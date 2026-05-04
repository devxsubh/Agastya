"""CLI wrapper to generate Hybrid eval JSON artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct execution: `python3 src/phase3/hybrid_eval_cli.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.hybrid_eval import generate_hybrid_eval_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 3 Hybrid evaluation artifact.")
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--out", default="reports/phase3/hybrid_eval.json")
    parser.add_argument("--bn-model", default="results/phase3/bayesian_network.pkl")
    parser.add_argument("--rf-model", default="results/phase3/rf_reasoner.pkl")
    parser.add_argument("--bert-checkpoint", default="results/phase2/models/legal_bert_phase2.pt")
    parser.add_argument("--label-map", default="results/phase2/label2id.json")
    parser.add_argument("--adapter-path", default="results/phase2/models/legal_bert_lora_adapter")
    parser.add_argument("--rf-threshold", type=float, default=0.11)
    args = parser.parse_args()

    payload = generate_hybrid_eval_artifact(
        test_csv_path=args.test_csv,
        output_json_path=args.out,
        bn_model_path=args.bn_model,
        rf_model_path=args.rf_model,
        bert_checkpoint_path=args.bert_checkpoint,
        label_map_path=args.label_map,
        adapter_path=args.adapter_path,
        rf_label_confidence_floor=args.rf_threshold,
    )
    print(f"Saved: {args.out}")
    print(payload)


if __name__ == "__main__":
    main()

