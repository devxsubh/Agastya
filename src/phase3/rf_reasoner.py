"""Random Forest contract-level risk reasoner.

Drop-in replacement for BN risk reasoning. It consumes detected CUAD labels
and predicts contract-level risk with calibrated probabilities.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

PAYMENT_LABELS = {
    "Revenue/Profit Sharing",
    "Minimum Commitment",
    "Price Restrictions",
    "Liquidated Damages",
    "Most Favored Nation",
    "Volume Restriction",
    "Unlimited/All-You-Can-Eat-License",
}
TERMINATION_LABELS = {
    "Termination For Convenience",
    "Notice Period To Terminate Renewal",
    "Post-Termination Services",
    "Renewal Term",
    "Expiration Date",
    "Change Of Control",
}
LIABILITY_LABELS = {
    "Cap On Liability",
    "Uncapped Liability",
    "Warranty Duration",
    "Insurance",
    "Liquidated Damages",
}
CONFIDENTIALITY_LABELS = {
    "Non-Compete",
    "Non-Disparagement",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "Source Code Escrow",
    "Non-Transferable License",
    "Irrevocable Or Perpetual License",
    "Affiliate License-Licensee",
    "Affiliate License-Licensor",
    "License Grant",
    "Competitive Restriction Exception",
    "Exclusivity",
    "Third Party Beneficiary",
    "Rofr/Rofo/Rofn",
    "Anti-Assignment",
}
DISPUTE_LABELS = {
    "Governing Law",
    "Covenant Not To Sue",
    "Audit Rights",
}

# Legacy 25-label feature universe (kept for backward compatibility only).
ALL_LABELS: list[str] = sorted(
    PAYMENT_LABELS
    | TERMINATION_LABELS
    | LIABILITY_LABELS
    | CONFIDENTIALITY_LABELS
    | DISPUTE_LABELS
)


def get_phase2_feature_labels(
    label_map_path: str = "results/phase2/label2id.json",
) -> list[str]:
    """Return all Phase 2 labels in stable id order (41-label full universe)."""
    path = Path(label_map_path)
    if not path.exists():
        raise FileNotFoundError(
            f"label2id.json not found at '{path}'. "
            "This file is required to ensure the RF feature space is aligned "
            "with the 41-label BERT universe. "
            "Run Phase 2 training first or pass the correct path."
        )
    payload = pd.read_json(path, typ="series")
    ordered = sorted(payload.items(), key=lambda kv: int(kv[1]))
    return [str(label) for label, _ in ordered]


def build_feature_vector(
    present_labels: list[str] | set[str],
    feature_labels: list[str] | None = None,
) -> np.ndarray:
    """Build a clause-frequency vector for one contract."""
    labels = feature_labels or ALL_LABELS
    if isinstance(present_labels, (list, tuple)):
        from collections import Counter
        counts = Counter(present_labels)
        return np.array([float(counts.get(lbl, 0)) for lbl in labels], dtype=np.float32)
    return np.array(
        [1.0 if lbl in present_labels else 0.0 for lbl in labels],
        dtype=np.float32,
    )


def build_dataset(
    df: pd.DataFrame,
    feature_labels: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build contract-level (X, y) from row-level CUAD-style dataframe."""
    from src.phase3.hybrid_eval import _derive_contract_risk

    labels = feature_labels or ALL_LABELS
    x_rows: list[np.ndarray] = []
    y: list[str] = []
    for _filename, frame in df.groupby("filename"):
        present = [str(x).strip() for x in frame["label"].dropna()]
        x_rows.append(build_feature_vector(present, labels))
        y.append(_derive_contract_risk(set(present)))
    return np.array(x_rows, dtype=np.float32), y


def train_rf_reasoner(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    save_path: str = "results/phase3/rf_reasoner.pkl",
    feature_labels: list[str] | None = None,
) -> dict:
    """Train and save calibrated RF reasoner."""
    labels = list(feature_labels or ALL_LABELS)
    x_train, y_train = build_dataset(train_df, labels)

    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        x_train, y_train = smote.fit_resample(x_train, y_train)
    except ImportError:
        print("Warning: imbalanced-learn not installed. Skipping SMOTE augmentation.")

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf = CalibratedClassifierCV(rf, cv=5, method="isotonic")
    clf.fit(x_train, y_train)

    if val_df is not None:
        x_val, y_val = build_dataset(val_df, labels)
        y_pred = clf.predict(x_val)
        val_f1 = f1_score(y_val, y_pred, average="macro", labels=["Low", "Medium", "High"])
        print(f"Validation macro-F1: {val_f1:.4f}")
        print(classification_report(y_val, y_pred, labels=["Low", "Medium", "High"]))

    bundle = {
        "model": clf,
        "feature_labels": labels,
    }
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(bundle, f)
    return bundle


def load_rf_reasoner(path: str = "results/phase3/rf_reasoner.pkl") -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # Backward compatibility with old pickles that stored only estimator.
    if isinstance(obj, dict) and "model" in obj and "feature_labels" in obj:
        return obj
    return {"model": obj, "feature_labels": list(ALL_LABELS)}


def predict_risk(reasoner: dict, present_labels: list[str] | set[str]) -> dict:
    """Predict risk using calibrated RF and return BN-compatible keys."""
    clf = reasoner["model"]
    feature_labels: list[str] = list(reasoner["feature_labels"])
    x = build_feature_vector(present_labels, feature_labels).reshape(1, -1)
    risk_level: str = str(clf.predict(x)[0])
    proba = clf.predict_proba(x)[0]
    classes = list(clf.classes_)

    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    for cls in ["Low", "Medium", "High"]:
        prob_dict.setdefault(cls, 0.0)

    return {
        "risk_level": risk_level,
        "probabilities": prob_dict,
        "distribution": prob_dict,
        "feature_vector": build_feature_vector(present_labels, feature_labels).tolist(),
    }


def get_feature_importances(reasoner: dict) -> pd.DataFrame:
    """Extract RF feature importances from calibrated estimator wrapper."""
    clf = reasoner["model"]
    feature_labels: list[str] = list(reasoner["feature_labels"])
    base_rf = clf.calibrated_classifiers_[0].estimator
    importances = base_rf.feature_importances_
    return (
        pd.DataFrame({"label": feature_labels, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
