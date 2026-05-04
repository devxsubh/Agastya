"""One-shot RF reasoner training script. Run from project root."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import classification_report

# Allow direct execution: `python3 scripts/train_rf_reasoner.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.rf_reasoner import build_dataset, get_feature_importances, train_rf_reasoner
from src.phase3.rf_reasoner import get_phase2_feature_labels

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"
TEST_CSV = "data/processed/test.csv"
SAVE_PATH = "results/phase3/rf_reasoner.pkl"
LABEL_MAP_PATH = "results/phase2/label2id.json"


def main() -> None:
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV) if Path(VAL_CSV).exists() else None
    test_df = pd.read_csv(TEST_CSV)

    feature_labels = get_phase2_feature_labels(LABEL_MAP_PATH)
    print(f"Training RF with {len(feature_labels)} feature labels")
    reasoner = train_rf_reasoner(
        train_df,
        val_df=val_df,
        save_path=SAVE_PATH,
        feature_labels=feature_labels,
    )

    x_test, y_test = build_dataset(test_df, feature_labels=feature_labels)
    y_pred = reasoner["model"].predict(x_test)
    print("\n=== FINAL TEST RESULTS ===")
    print(classification_report(y_test, y_pred, labels=["Low", "Medium", "High"]))

    print("\n=== TOP 10 MOST IMPORTANT CLAUSES ===")
    fi = get_feature_importances(reasoner)
    print(fi.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
