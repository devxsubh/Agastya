"""Sweep Phase 2 confidence threshold for Hybrid v2 (BERT + RF).

This script quantifies how many clause labels survive thresholding and how
that propagates into contract-level risk macro-F1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

# Allow direct execution: `python3 scripts/threshold_sweep.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2.segmentation.clause_splitter import split_clauses
from src.phase3.hybrid_eval import build_contract_dataset
from src.phase3.interface.phase2_adapter import Phase2BertAdapter
from src.phase3.rf_reasoner import load_rf_reasoner, predict_risk


def _predict_present_labels(
    adapter: Phase2BertAdapter,
    text: str,
    threshold: float,
    feature_set: set[str],
) -> set[str]:
    clauses = split_clauses(text)
    present: set[str] = set()
    for clause in clauses:
        preds = adapter.predict(clause)
        for row in preds:
            label = str(row.get("phase2_label", "")).strip()
            conf = float(row.get("confidence", 0.0))
            if conf >= threshold and label in feature_set:
                present.add(label)
    return present


def main() -> None:
    reasoner = load_rf_reasoner("results/phase3/rf_reasoner.pkl")
    feature_labels = list(reasoner["feature_labels"])
    feature_set = set(feature_labels)

    adapter = Phase2BertAdapter(
        checkpoint_path="results/phase2/models/legal_bert_phase2.pt",
        label_map_path="results/phase2/label2id.json",
        adapter_path="results/phase2/models/legal_bert_lora_adapter",
    )

    test_df = pd.read_csv("data/processed/test.csv")
    contract_df = build_contract_dataset(test_df)
    y_true = contract_df["true_risk"].tolist()
    labels = ["Low", "Medium", "High"]

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_f1, best_t = 0.0, thresholds[0]

    print(f"{'Threshold':>10}  {'Macro-F1':>10}  {'Low':>6}  {'Med':>6}  {'High':>6}  {'AvgClauses':>12}")
    print("-" * 65)

    for threshold in thresholds:
        y_pred: list[str] = []
        clause_counts: list[int] = []
        for text in contract_df["text"].tolist():
            present = _predict_present_labels(adapter, text, threshold, feature_set)
            clause_counts.append(len(present))
            result = predict_risk(reasoner, present)
            y_pred.append(str(result["risk_level"]))

        macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        per_cls = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        avg_clauses = sum(clause_counts) / max(len(clause_counts), 1)

        if macro > best_f1:
            best_f1, best_t = macro, threshold

        print(
            f"{threshold:>10.2f}  {macro:>10.4f}  "
            f"{per_cls[0]:>6.3f}  {per_cls[1]:>6.3f}  {per_cls[2]:>6.3f}  "
            f"{avg_clauses:>12.1f}"
        )

    print(f"\nBest threshold: {best_t:.2f} -> macro-F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
