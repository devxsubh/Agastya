"""Diagnostic tool to analyze 'Why' the pipeline fails on specific contracts.

Identifies mismatches between Ground Truth and Predictions and decomposes the error.
"""

import sys
from pathlib import Path

# Allow direct execution: `python3 src/phase3/failure_analysis.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.phase3.hybrid_eval import build_contract_dataset, _derive_contract_risk
from src.phase3.hybrid_pipeline import AgastyaHybridPipeline

def analyze_failures(n_cases: int = 3):
    print("Loading pipeline and test data...")
    test_df = pd.read_csv("data/processed/test.csv")
    contract_df = build_contract_dataset(test_df)
    
    pipeline = AgastyaHybridPipeline(
        rf_model_path="results/phase3/rf_reasoner.pkl",
        bn_model_path=None,
        bert_checkpoint_path="results/phase2/models/legal_bert_phase2.pt",
        label_map_path="results/phase2/label2id.json",
        adapter_path="results/phase2/models/legal_bert_lora_adapter",
    )
    # Use the optimized threshold
    pipeline.rf_label_confidence_floor = 0.11

    failures = []

    print(f"Analyzing {len(contract_df)} contracts for failures...")
    for _, row in tqdm(contract_df.iterrows(), total=len(contract_df)):
        res = pipeline.predict(row["text"])
        
        if res["risk_level"] != row["true_risk"]:
            # Decompose why
            ground_truth_labels = set(test_df[test_df["filename"] == row["filename"]]["label"].dropna())
            predicted_labels = set(res["clause_evidence"].keys())
            
            # Find missed risky clauses
            missed = ground_truth_labels - predicted_labels
            false_positives = predicted_labels - ground_truth_labels
            
            failures.append({
                "filename": row["filename"],
                "true_risk": row["true_risk"],
                "pred_risk": res["risk_level"],
                "missed_clauses": sorted(list(missed)),
                "false_positives": sorted(list(false_positives)),
                "bert_confidences": {d["phase2_label"]: round(d["confidence"], 3) for d in res["bert_details"] if d["phase2_label"] in missed}
            })

    print(f"\nFound {len(failures)} failures out of {len(contract_df)} contracts.")
    print("-" * 50)
    
    for i, fail in enumerate(failures[:n_cases]):
        print(f"CASE #{i+1}: {fail['filename']}")
        print(f"  ACTUAL: {fail['true_risk']} | PREDICTED: {fail['pred_risk']}")
        
        if fail["missed_clauses"]:
            print(f"  ❌ MISSED CLAUSES (Why it failed):")
            for m in fail["missed_clauses"]:
                conf = fail["bert_confidences"].get(m, "Not Predicted")
                print(f"    - {m} (BERT confidence: {conf})")
        
        if fail["false_positives"]:
            print(f"  👻 FALSE POSITIVES (Noise): {', '.join(fail['false_positives'][:3])}...")
        
        print("-" * 50)

    # Save to CSV for the report
    if failures:
        fail_df = pd.DataFrame(failures)
        out_path = "reports/phase3/failure_analysis.csv"
        fail_df.to_csv(out_path, index=False)
        print(f"Full failure analysis saved to: {out_path}")

if __name__ == "__main__":
    analyze_failures()
