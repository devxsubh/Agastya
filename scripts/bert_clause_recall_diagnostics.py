"""Diagnose Phase 2 clause recall against contract ground truth labels."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Allow direct execution: `python3 scripts/bert_clause_recall_diagnostics.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2.segmentation.clause_splitter import split_clauses
from src.phase3.interface.phase2_adapter import Phase2BertAdapter
from src.phase3.rf_reasoner import load_rf_reasoner


def main() -> None:
    threshold = 0.30

    reasoner = load_rf_reasoner("results/phase3/rf_reasoner.pkl")
    feature_set = set(reasoner["feature_labels"])

    adapter = Phase2BertAdapter(
        checkpoint_path="results/phase2/models/legal_bert_phase2.pt",
        label_map_path="results/phase2/label2id.json",
        adapter_path="results/phase2/models/legal_bert_lora_adapter",
    )

    test_df = pd.read_csv("data/processed/test.csv")
    stats = Counter()
    missing = Counter()

    for _, frame in test_df.groupby("filename"):
        text = " ".join(str(x) for x in frame["text"].dropna().tolist())[:4000]
        clauses = split_clauses(text)

        detected: set[str] = set()
        for clause in clauses:
            for row in adapter.predict(clause):
                label = str(row.get("phase2_label", "")).strip()
                conf = float(row.get("confidence", 0.0))
                if conf >= threshold and label in feature_set:
                    detected.add(label)

        true_labels = {str(x).strip() for x in frame["label"].dropna().tolist()}
        for label in true_labels & feature_set:
            if label in detected:
                stats["hit"] += 1
            else:
                stats["miss"] += 1
                missing[label] += 1

    total = stats["hit"] + stats["miss"]
    recall = stats["hit"] / max(total, 1)
    print(
        f"BERT clause recall (threshold={threshold:.2f}): "
        f"{stats['hit']}/{total} = {recall:.1%}"
    )
    print("\nMost-missed clause types:")
    for label, count in missing.most_common(10):
        print(f"  {count:3d}x  {label}")


if __name__ == "__main__":
    main()
