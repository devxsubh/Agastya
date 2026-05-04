#!/usr/bin/env python3
"""Debug fine-tuned Legal-BERT (Phase 2 LoRA clause classifier).

Produces:
  - Printed table: sample text (truncated), true label, predicted label, confidence, correct flag
  - CSV of the same rows under reports/phase2/debug_bert_samples.csv
  - Figure reports/phase2/figures/debug_legal_bert.png:
      * Loss curves (if ``results/phase2/training_history.json`` exists — see below)
      * Train-split class distribution (bar chart)

Training loss JSON (optional): save next to your adapter after training, e.g.::

    {
      "train_loss_per_epoch": [2.1, 1.4, 1.0],
      "val_loss_per_epoch":   [1.9, 1.35, 1.05]
    }

If the file is missing, the loss panel shows a short note instead of a curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.phase2.data.dataset_loader import load_label2id, load_split_csv
from src.phase2.models.bert_classifier import BertWithLengthClassifier
from src.phase2.models.bert_lora_classifier import load_lora_adapter


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    *,
    model_name: str,
    label2id_path: Path,
    adapter_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, int], dict[int, str]]:
    label2id = load_label2id(label2id_path)
    id2label = {i: lab for lab, i in label2id.items()}
    base = BertWithLengthClassifier(
        model_name=model_name,
        num_classes=len(label2id),
        dropout=0.1,
        use_length_feature=True,
        download_pretrained_backbone=True,
    )
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
    model = load_lora_adapter(base, str(adapter_path), device=device)
    model.eval()
    return model, label2id, id2label


@torch.inference_mode()
def predict_row(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    log_length: float,
    *,
    max_length: int,
    device: torch.device,
) -> tuple[int, torch.Tensor]:
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    length_feat = torch.tensor([[log_length]], dtype=torch.float32, device=device)
    logits = model(input_ids, attention_mask, length_feat)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred_id = int(torch.argmax(probs).item())
    return pred_id, probs.cpu()


def _has_loss_series(history: dict) -> bool:
    tr = history.get("train_loss_per_epoch")
    va = history.get("val_loss_per_epoch")
    return bool(tr) or bool(va)


def plot_figure(
    out_path: Path,
    train_counts: pd.Series,
    history: dict,
    *,
    history_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax0 = axes[0]
    if _has_loss_series(history):
        tr = history.get("train_loss_per_epoch") or []
        va = history.get("val_loss_per_epoch") or []
        if tr:
            ax0.plot(range(1, len(tr) + 1), tr, marker="o", label="train loss")
        if va:
            ax0.plot(range(1, len(va) + 1), va, marker="s", label="val loss")
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Loss")
        ax0.set_title("Training / validation loss")
        ax0.legend()
        ax0.grid(True, alpha=0.3)
    else:
        ax0.axis("off")
        ax0.text(
            0.5,
            0.5,
            "No loss history (or empty lists).\n\n"
            f"Add:\n  {history_path}\n\n"
            "Example:\n"
            '  {"train_loss_per_epoch": [2.1, 1.4], "val_loss_per_epoch": [1.9, 1.35]}',
            ha="center",
            va="center",
            fontsize=10,
            family="monospace",
            transform=ax0.transAxes,
        )
        ax0.set_title("Loss curve (optional)")

    ax1 = axes[1]
    counts = train_counts.sort_values(ascending=True)
    y = np.arange(len(counts))
    ax1.barh(y, counts.values, color="steelblue", alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(counts.index, fontsize=6)
    ax1.set_xlabel("Count (train split)")
    ax1.set_title("Class distribution — train.csv")
    ax1.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug Legal-BERT LoRA: samples, optional loss curve, class distribution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_PROJECT,
        help="Repository root (contains src/, data/, results/).",
    )
    parser.add_argument(
        "--model-name",
        default="nlpaueb/legal-bert-base-uncased",
        help="HuggingFace backbone id.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="LoRA adapter directory (default: <root>/results/phase2/models/legal_bert_lora_adapter).",
    )
    parser.add_argument(
        "--label2id",
        type=Path,
        default=None,
        help="label2id.json (default: <root>/results/phase2/label2id.json).",
    )
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument(
        "--history-json",
        type=Path,
        default=None,
        help="Per-epoch losses (default: <root>/results/phase2/training_history.json).",
    )
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Sample table CSV (default: <root>/reports/phase2/debug_bert_samples.csv).",
    )
    parser.add_argument(
        "--out-figure",
        type=Path,
        default=None,
        help="Loss + class distribution PNG (default: <root>/reports/phase2/figures/debug_legal_bert.png).",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    adapter = args.adapter_path or (root / "results/phase2/models/legal_bert_lora_adapter")
    label2id_path = args.label2id or (root / "results/phase2/label2id.json")
    train_csv = args.train_csv or (root / "data/processed/train.csv")
    test_csv = args.test_csv or (root / "data/processed/test.csv")
    history_path = args.history_json or (root / "results/phase2/training_history.json")
    out_csv = args.out_csv or (root / "reports/phase2/debug_bert_samples.csv")
    out_fig = args.out_figure or (root / "reports/phase2/figures/debug_legal_bert.png")

    device = _device()
    print("device:", device)

    train_df = load_split_csv(train_csv)
    test_df = load_split_csv(test_csv)

    rng = np.random.default_rng(args.seed)
    n = min(max(1, args.n_samples), len(test_df))
    idxs = rng.choice(len(test_df), size=n, replace=False)

    model, _label2id, id2label = load_model(
        model_name=args.model_name,
        label2id_path=label2id_path,
        adapter_path=adapter,
        device=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    rows_out: list[dict] = []
    print(f"\n{'='*80}\nSample predictions (n={n}, test split)\n{'='*80}")
    for i, idx in enumerate(idxs, start=1):
        r = test_df.iloc[int(idx)]
        text = str(r["text"])
        y_true = int(r["label_id"])
        log_len = float(r["log_length"])
        pred_id, probs = predict_row(
            model,
            tokenizer,
            text,
            log_len,
            max_length=args.max_length,
            device=device,
        )
        conf = float(probs[pred_id].item())
        true_name = id2label.get(y_true, str(y_true))
        pred_name = id2label.get(pred_id, str(pred_id))
        ok = pred_id == y_true
        snippet = text.replace("\n", " ")[:200] + ("…" if len(text) > 200 else "")
        print(f"\n--- Sample {i} (row {idx}) {'✓' if ok else '✗'} ---")
        print("text:", snippet)
        print("true :", true_name)
        print("pred :", pred_name, f"(p={conf:.4f})")
        top3 = torch.topk(probs, k=min(3, len(probs)))
        top_parts = [
            f"{id2label[int(top3.indices[j])]}={float(top3.values[j]):.3f}"
            for j in range(top3.indices.numel())
        ]
        print("top-3:", ", ".join(top_parts))
        rows_out.append(
            {
                "row_index": int(idx),
                "text_snippet": snippet,
                "true_label": true_name,
                "pred_label": pred_name,
                "confidence": conf,
                "correct": ok,
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"\nSaved sample table -> {out_csv}")

    label_col = "label" if "label" in train_df.columns else "label_id"
    train_counts = train_df[label_col].value_counts()

    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = {}

    plot_figure(out_fig, train_counts, history, history_path=history_path)
    print(f"Saved figure -> {out_fig}")


if __name__ == "__main__":
    main()
