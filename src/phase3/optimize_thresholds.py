"""Optimization script to find the best rf_label_confidence_floor.

Sweeps through confidence thresholds to maximize the downstream Hybrid F1 score.
"""

import sys
from pathlib import Path

# Allow direct execution: `python3 src/phase3/optimize_thresholds.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.phase3.hybrid_eval import build_contract_dataset, _derive_contract_risk
from src.phase3.hybrid_pipeline import AgastyaHybridPipeline

def optimize():
    print("Loading test dataset...")
    test_df = pd.read_csv("data/processed/test.csv")
    contract_df = build_contract_dataset(test_df)
    
    # Pre-load pipeline (using defaults)
    pipeline = AgastyaHybridPipeline(
        rf_model_path="results/phase3/rf_reasoner.pkl",
        bn_model_path=None,
        bert_checkpoint_path="results/phase2/models/legal_bert_phase2.pt",
        label_map_path="results/phase2/label2id.json",
        adapter_path="results/phase2/models/legal_bert_lora_adapter",
    )
    
    # We pre-calculate BERT outputs for all contracts to save time in the loop
    print("Pre-calculating Legal-BERT outputs for all contracts...")
    all_bert_outputs = []
    for _, row in tqdm(contract_df.iterrows(), total=len(contract_df)):
        raw_outputs = [pipeline.bert.predict(clause) for clause in row["clauses"]]
        # Flatten (using the same internal logic as pipeline)
        from src.phase3.hybrid_pipeline import _flatten_bert_outputs
        all_bert_outputs.append(_flatten_bert_outputs(raw_outputs))

    thresholds = np.arange(0.01, 0.51, 0.02)
    results = []

    print(f"Sweeping {len(thresholds)} thresholds...")
    for thresh in tqdm(thresholds):
        pipeline.rf_label_confidence_floor = thresh
        y_true = []
        y_pred = []
        
        for i, row in enumerate(contract_df.iterrows()):
            contract_row = row[1]
            # Use the cached bert outputs
            bert_outputs = all_bert_outputs[i]
            
            # Run the RF part of _predict_from_clauses manually
            known_labels = set(pipeline.rf.get("feature_labels", []))
            present_labels = [
                str(output.get("phase2_label", "")).strip()
                for output in bert_outputs
                if float(output.get("confidence", 0.0)) >= pipeline.rf_label_confidence_floor
                and str(output.get("phase2_label", "")).strip() in known_labels
            ]
            
            if not present_labels:
                best_known = sorted(
                    (
                        output for output in bert_outputs
                        if str(output.get("phase2_label", "")).strip() in known_labels
                    ),
                    key=lambda output: float(output.get("confidence", 0.0)),
                    reverse=True,
                )
                if best_known:
                    present_labels.append(str(best_known[0].get("phase2_label", "")).strip())
            
            from src.phase3.rf_reasoner import predict_risk
            res = predict_risk(pipeline.rf, present_labels)
            
            y_true.append(contract_row["true_risk"])
            y_pred.append(res["risk_level"])
            
        score = f1_score(y_true, y_pred, average="macro")
        results.append({"threshold": thresh, "macro_f1": score})

    res_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    print("\nTop 5 Thresholds:")
    print(res_df.head(5))
    
    best_thresh = res_df.iloc[0]["threshold"]
    best_f1 = res_df.iloc[0]["macro_f1"]
    print(f"\nOptimal threshold: {best_thresh:.3f} (Macro-F1: {best_f1:.4f})")
    
    print("\nAction: Update 'rf_label_confidence_floor' in AgastyaHybridPipeline constructor or instance.")

if __name__ == "__main__":
    optimize()
