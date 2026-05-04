#!/usr/bin/env python3
"""Intensive Legal-BERT (LoRA) clause training for Colab / GPU.

Targets higher Macro-F1 on the 41-way CUAD-style task via:
  - larger LoRA rank, longer training, early stopping on validation Macro-F1
  - linear warmup + cosine LR schedule
  - optional gradient accumulation and mixed precision (CUDA)
  - class-weighted cross-entropy (same spirit as Notebook 07)

Outputs (under ``--output-dir``, default ``results/phase2``):
  - ``models/legal_bert_lora_adapter/`` (best validation Macro-F1)
  - ``results.json`` (best val macro-F1, accuracy, test metrics if run)
  - ``training_history.json`` (per-epoch train/val loss + val macro-F1)

Macro-F1 ~0.85 on this dataset is not guaranteed (label noise, segment ambiguity,
and class imbalance); this script maximizes reproducible training quality.

Example (Colab, after unzipping repo to /content/agastya)::

    cd /content/agastya
    pip install -r colab/minimal_requirements_phase2.txt
    python3 scripts/train_legal_bert_intensive.py \\
        --project-root /content/agastya \\
        --epochs 30 --patience 8 --batch-size 16 --grad-accum 2
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.phase2.data.dataset import ContractDataset
from src.phase2.data.dataset_loader import load_label2id, load_split_csv
from src.phase2.models.bert_classifier import BertWithLengthClassifier
from src.phase2.models.bert_lora_classifier import (
    apply_lora,
    load_lora_adapter,
    merge_lora_into_base,
    save_lora_adapter,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _class_weights_from_train(train_df, label2id: dict[str, int], device: torch.device) -> torch.Tensor:
    counts = train_df.groupby("label_id").size().reindex(range(len(label2id)), fill_value=0)
    # inverse frequency; avoid div by zero
    w = 1.0 / counts.replace(0, np.nan).fillna(counts.max() + 1).astype(float)
    w = w / w.mean()
    return torch.tensor(w.values, dtype=torch.float32, device=device)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    crit_eval = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    all_y: list[int] = []
    all_p: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            length_feat = batch["length_feat"].to(device, non_blocking=True)
            if device.type == "cuda" and use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids, attention_mask, length_feat)
                    loss = crit_eval(logits, labels)
            else:
                logits = model(input_ids, attention_mask, length_feat)
                loss = crit_eval(logits, labels)
            total_loss += float(loss.item())
            n_batches += 1
            preds = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            all_p.extend(int(x) for x in preds)
            all_y.extend(labels.detach().cpu().numpy().tolist())
    avg_loss = total_loss / max(1, n_batches)
    macro_f1 = float(f1_score(all_y, all_p, average="macro", zero_division=0))
    return avg_loss, macro_f1, all_y, all_p


def main() -> None:
    p = argparse.ArgumentParser(description="Intensive Legal-BERT LoRA training (Phase 2).")
    p.add_argument("--project-root", type=Path, default=_PROJECT)
    p.add_argument("--train-csv", type=Path, default=None)
    p.add_argument("--val-csv", type=Path, default=None)
    p.add_argument("--test-csv", type=Path, default=None)
    p.add_argument("--label2id", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None, help="Usually results/phase2")
    p.add_argument("--model-name", default="nlpaueb/legal-bert-base-uncased")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=8, help="Early stopping on val Macro-F1.")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-merged", action="store_true", help="Also save merged full BERT weights (large).")
    p.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast.")
    args = p.parse_args()

    root = args.project_root.resolve()
    out = args.output_dir or (root / "results/phase2")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = models_dir / "legal_bert_lora_adapter"

    train_csv = args.train_csv or (root / "data/processed/train.csv")
    val_csv = args.val_csv or (root / "data/processed/val.csv")
    test_csv = args.test_csv or (root / "data/processed/test.csv")
    label2id_path = args.label2id or (root / "data/processed/label2id.json")
    if not label2id_path.exists():
        label2id_path = root / "results/phase2/label2id.json"

    _set_seed(args.seed)
    device = _device()
    use_amp = device.type == "cuda" and not args.no_amp

    label2id = load_label2id(label2id_path)
    num_classes = len(label2id)

    train_df = load_split_csv(train_csv)
    val_df = load_split_csv(val_csv)
    test_df = load_split_csv(test_csv) if test_csv.exists() else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = ContractDataset(train_df, tokenizer, max_length=args.max_length)
    val_ds = ContractDataset(val_df, tokenizer, max_length=args.max_length)
    loader_kw = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = (
        DataLoader(ContractDataset(test_df, tokenizer, max_length=args.max_length), shuffle=False, **loader_kw)
        if test_df is not None
        else None
    )

    model = BertWithLengthClassifier(
        args.model_name,
        num_classes=num_classes,
        dropout=0.1,
        use_length_feature=True,
    )
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.to(device)

    weights = _class_weights_from_train(train_df, label2id, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    opt = AdamW(
        [x for x in model.parameters() if x.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: dict[str, list] = {
        "train_loss_per_epoch": [],
        "val_loss_per_epoch": [],
        "val_macro_f1_per_epoch": [],
        "val_acc_per_epoch": [],
    }

    best_f1 = -1.0
    patience_left = args.patience

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            length_feat = batch["length_feat"].to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids, attention_mask, length_feat)
                    loss = criterion(logits, labels) / args.grad_accum
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids, attention_mask, length_feat)
                loss = criterion(logits, labels) / args.grad_accum
                loss.backward()

            running += float(loss.item()) * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                if use_amp:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=f"{running / max(1, step + 1):.4f}")

        # leftover grad accum
        if len(train_loader) % args.grad_accum != 0:
            if use_amp:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        train_loss = running / max(1, len(train_loader))
        val_loss, val_f1, vy, vp = _evaluate(model, val_loader, device, use_amp)
        val_acc = float(accuracy_score(vy, vp))

        history["train_loss_per_epoch"].append(train_loss)
        history["val_loss_per_epoch"].append(val_loss)
        history["val_macro_f1_per_epoch"].append(val_f1)
        history["val_acc_per_epoch"].append(val_acc)

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}  val_acc={val_acc:.4f}"
        )

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            patience_left = args.patience
            save_lora_adapter(model, str(adapter_dir))
            print(f"  -> new best val_macro_f1={best_f1:.4f}; saved adapter -> {adapter_dir}")
        else:
            patience_left -= 1
            print(f"  (no improvement; patience {patience_left}/{args.patience})")
            if patience_left <= 0:
                print("Early stopping.")
                break

    # Persist history for debug / plots
    hist_path = out / "training_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Wrote", hist_path)

    # Reload best adapter for test evaluation
    eval_model = model
    if adapter_dir.exists():
        base_eval = BertWithLengthClassifier(
            args.model_name,
            num_classes=num_classes,
            dropout=0.1,
            use_length_feature=True,
        )
        eval_model = load_lora_adapter(base_eval, str(adapter_dir), device=device)

    test_macro: float | None = None
    test_acc: float | None = None
    if test_loader is not None and adapter_dir.exists():
        _, test_macro, ty, tp = _evaluate(eval_model, test_loader, device, use_amp)
        test_acc = float(accuracy_score(ty, tp))
        print(f"Test macro_f1={test_macro:.4f}  test_acc={test_acc:.4f}")

    results = {
        "test_macro_f1": float(test_macro) if test_macro is not None else None,
        "test_accuracy": float(test_acc) if test_acc is not None else None,
        "val_macro_f1_best": float(best_f1),
        "macro_f1": float(test_macro) if test_macro is not None else float(best_f1),
        "accuracy": float(test_acc) if test_acc is not None else None,
        "notes": "macro_f1/accuracy mirror test when available, else val best only.",
        "model_name": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs_ran": len(history["train_loss_per_epoch"]),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote", out / "results.json")

    # Copy label map next to results for packaging convenience
    label_dst = out / "label2id.json"
    label_dst.write_text(label2id_path.read_text(encoding="utf-8"), encoding="utf-8")

    if args.save_merged and adapter_dir.exists():
        base = BertWithLengthClassifier(
            args.model_name,
            num_classes=num_classes,
            dropout=0.1,
            use_length_feature=True,
        )
        model_m = load_lora_adapter(base, str(adapter_dir), device=torch.device("cpu"))
        merged = merge_lora_into_base(copy.deepcopy(model_m).cpu())
        ckpt = models_dir / "legal_bert_phase2.pt"
        torch.save(merged.state_dict(), ckpt)
        print("Wrote merged checkpoint ->", ckpt)


if __name__ == "__main__":
    main()
