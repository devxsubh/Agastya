# Agastya

**Agastya** is a research system for **automated contract risk classification**: turning long-form legal agreements into structured risk signals using a three-phase neuro-symbolic pipeline. Given a contract PDF or text, Agastya identifies which clause types are present (Legal-BERT) and reasons over them to classify the contract as **Low / Medium / High** risk (Random Forest).

## Pipeline Overview

```
Contract Text / PDF
      │
      ▼
  [Phase 2] Legal-BERT (LoRA fine-tuned)
  Clause detection across 41 CUAD categories
      │
      ▼
  [Phase 3] Random Forest Reasoner
  Contract-level risk classification (Low / Medium / High)
      │
      ▼
  Risk Label + Clause Evidence + Feature Importances
```

---

## Contents

- [Results](#results)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Quickstart](#quickstart)
- [Notebooks](#notebooks)
- [Reproducible Setup](#reproducible-setup)
- [Development Practices](#development-practices)
- [Team](#team)
- [Acknowledgments](#acknowledgments)

---

## Results

All metrics are loaded from artifact files — never hardcoded. Run `python3 src/phase3/hybrid_eval_cli.py` to regenerate.

### Phase Progression

| Phase | Model | Task | Macro-F1 | Accuracy |
|-------|-------|------|----------|----------|
| **Phase 1** | LinearSVC + TF-IDF | Clause classification (41 CUAD) | **0.719** | 0.800 |
| **Phase 2** | Legal-BERT (LoRA, intensive retrain) | Clause classification (41 CUAD) | **0.759** | 0.826 |
| Phase 3 v1 ❌ | Legal-BERT + Bayesian Network | Contract risk (Low/Med/High) | 0.159 | — |
| **Phase 3 v2 ✅** | **Legal-BERT + Random Forest** | **Contract risk (Low/Med/High)** | **0.866** | **0.882** |

Phase 2 / Hybrid figures mirror `results/phase2/results.json` and `reports/phase3/hybrid_eval.json`; retrain Legal-BERT then rerun `python3 src/phase3/hybrid_eval_cli.py` to refresh.

> **Why did the Bayesian Network fail?** The BN required hand-coded Conditional Probability Tables encoding legal domain knowledge. With only 5 evidence nodes for 41 CUAD labels, 80% of the BERT signal was discarded before reasoning. EM training on the imbalanced dataset then collapsed the `Medium` class entirely (all predictions → High). The RF replaced the BN as the reasoning layer and learns clause interaction weights from data.

### RF Oracle (Upper Bound)

When given **ground-truth** clause labels (bypassing BERT extraction errors), the RF achieves **Macro-F1 = 0.903**. The gap between that oracle and live hybrid Macro-F1 (see `hybrid_eval.json`) shrinks as the Phase 2 encoder improves — remaining gap is largely Phase 2 clause noise feeding the RF.

---

## Architecture

### Phase 1 — Classical ML Baseline

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Features | TF-IDF (word + character n-grams) | Strong legal surface form baseline; sparse and inspectable |
| Model | LinearSVC (`class_weight="balanced"`) | Phase 1 mandate: scikit-learn only, no deep learning |
| Evaluation | Macro-F1 (primary) | Handles severe class imbalance across 41 clause categories |

### Phase 2 — Legal-BERT Clause Classifier

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Backbone | `nlpaueb/legal-bert-base-uncased` | Domain-specific pre-training for legal text |
| Fine-tuning | LoRA adapters | Parameter-efficient; preserves frozen BERT weights |
| Segmentation | Segment-based extraction (≤512 tokens) | Handles long contracts across multiple passes |
| Training | PyTorch + HuggingFace, AdamW, weighted loss | Standardized pipeline for transformer fine-tuning |

### Phase 3 — Hybrid Neuro-Symbolic Pipeline

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Feature extraction | Frozen Legal-BERT (Phase 2 adapter) | Converts clause text → 41-dim presence signals |
| Reasoner | Calibrated Random Forest (41 CUAD labels, SMOTE) | Data-driven over all labels; no manual CPT specification |
| Confidence floor | 0.11 (mathematically optimized) | Optimal signal-to-noise threshold for BERT → RF handoff |
| Feature vector | Clause frequency counts (not binary) | Preserves how many times each clause type appears |
| OCR ingestion | Tiered extractor (`pdfplumber` → EasyOCR fallback) | Supports both digital and scanned PDFs |
| Delivery surface | Streamlit app (`app/streamlit_app.py`) | Interactive upload-to-risk-analysis with traceable outputs |

### Key Design Decisions

- **No dummy fallbacks**: If Phase 2 artifacts are missing, the pipeline raises `FileNotFoundError` — no silent degradation to random or heuristic outputs.
- **No hardcoded metrics**: All displayed numbers are read from `results/phase*/results.json` and `reports/phase3/hybrid_eval.json` at runtime.
- **BN retained for ablation**: `results/phase3/bayesian_network_seed.pkl` is preserved for Notebook 10 ablation analysis only. It is never used in the live prediction path.

---

## Folder Structure

```
agastya/
├── app/
│   └── streamlit_app.py          # Interactive risk analysis UI
├── configs/                       # YAML defaults (model, training, data)
├── data/
│   ├── CUAD_v1/                   # Phase 1 copy (master_clauses.csv, JSON)
│   ├── raw/                       # Scanned PDFs and raw CUAD
│   ├── interim/                   # OCR outputs, segmented clauses
│   └── processed/                 # train.csv / val.csv / test.csv
├── notebooks/
│   ├── Phase_1/                   # Part_01 … Part_05 (classical ML)
│   ├── Phase_2/                   # 01 … 09 (OCR → BERT → evaluation)
│   └── Phase_3/                   # 10 … 13 (hybrid pipeline)
│       ├── 10_bn_structure_and_cpts.ipynb     # BN ablation (historical)
│       ├── 11_hybrid_pipeline_demo.ipynb      # End-to-end demo
│       ├── 12_ablation_study.ipynb            # ML vs DL vs Hybrid comparison
│       └── 13_interpretability_report.ipynb   # RF feature importances
├── reports/
│   └── phase3/
│       ├── hybrid_eval.json       # Live eval results (backend=rf)
│       ├── phase_progression_summary.csv
│       ├── ablation_results.csv
│       └── figures/               # rf_feature_importance.png, etc.
├── results/
│   ├── phase1/results.json        # Phase 1 SVM metrics
│   ├── phase2/
│   │   ├── results.json           # Phase 2 BERT metrics
│   │   ├── label2id.json          # 41-label CUAD map (required for RF)
│   │   └── models/                # legal_bert_phase2.pt, LoRA adapter
│   └── phase3/
│       ├── rf_reasoner.pkl        # Trained RF (41 labels, SMOTE, calibrated)
│       └── bayesian_network_seed.pkl  # BN ablation artifact only
├── src/
│   ├── phase2/                    # OCR, segmentation, BERT training, evaluation
│   └── phase3/
│       ├── hybrid_pipeline.py     # AgastyaHybridPipeline (BERT → RF)
│       ├── hybrid_eval.py         # Batch evaluation logic
│       ├── hybrid_eval_cli.py     # CLI: regenerates hybrid_eval.json
│       ├── rf_reasoner.py         # RF training, inference, importances
│       ├── ablation.py            # Ablation table builder
│       ├── interface/             # phase2_adapter, evidence_encoder, etc.
│       └── bayesian/              # BN code (ablation only)
├── scripts/                       # train.py, evaluate.py, run_ocr.py
├── tests/                         # pytest suite
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quickstart

### Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Upload any contract PDF and get a risk classification with clause-level evidence.

### Regenerate Phase 3 Evaluation Metrics

```bash
# From project root — overwrites reports/phase3/hybrid_eval.json
python3 src/phase3/hybrid_eval_cli.py
```

### Retrain the RF Reasoner

```bash
# Retrains on data/processed/train.csv, validates on val.csv
# Saves to results/phase3/rf_reasoner.pkl
python3 -c "
import pandas as pd
from src.phase3.rf_reasoner import train_rf_reasoner, get_phase2_feature_labels

labels = get_phase2_feature_labels()
train_df = pd.read_csv('data/processed/train.csv')
val_df   = pd.read_csv('data/processed/val.csv')
train_rf_reasoner(train_df, val_df, feature_labels=labels)
"
```

### Run Ablation Report

```bash
python3 -m src.phase3.ablation_cli
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Notebooks

All Phase 3 notebooks use a **path bootstrap** (Cell 1) that resolves `PROJECT_ROOT` from any working directory — no manual path adjustment needed.

| Notebook | Purpose | Key Output |
|----------|---------|------------|
| `Phase_1/Part_01` – `Part_05` | Classical ML baselines | SVM Macro-F1 = 0.719 |
| `Phase_2/01` – `09` | Legal-BERT fine-tuning and evaluation | Clause Macro-F1 ≈ 0.759 (see `results/phase2/results.json`) |
| `Phase_3/10_bn_structure_and_cpts` | **[Ablation]** Why BN failed | CPT collapse visualizations |
| `Phase_3/11_hybrid_pipeline_demo` | End-to-end demo on a real contract | Live risk classification |
| `Phase_3/12_ablation_study` | Phase 1 vs 2 vs 3 comparison | Reads all metrics from artifact files |
| `Phase_3/13_interpretability_report` | RF feature importances + BN ablation | `rf_feature_importance.png` |

**Run order for Phase 3:**
1. Notebook 11 (requires Phase 2 artifacts to be present)
2. Notebook 12 (reads from `hybrid_eval.json` — run CLI first)
3. Notebook 13 (reads from `rf_reasoner.pkl`)
4. Notebook 10 (standalone BN ablation — no dependency on RF)

---

## Reproducible Setup

**Prerequisites**

- Python 3.10+
- Poppler (for PDF rasterization in OCR fallback):
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `sudo apt install poppler-utils`

**Install**

```bash
git clone <repository-url> agastya
cd agastya

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Data:** CUAD v1 is included under `data/CUAD_v1/`. Phase 3 processed splits (`train.csv`, `val.csv`, `test.csv`) are generated by the Phase 2 segmentation pipeline. If starting fresh, run the Phase 2 preprocessing notebooks first.

**Determinism:** All training uses `random_state=42`. Re-run notebooks top-to-bottom after pulling changes.

---

## Development Practices

- **Commits:** Small, coherent commits with clear messages ([Conventional Commits](https://www.conventionalcommits.org/)).
- **No patchwork metrics:** All evaluation numbers are computed live from model artifacts. No hardcoded fallback values.
- **Phase boundaries:** Phase 1 — no transformers. Phase 2 — BERT fine-tuning, no contract-level reasoning. Phase 3 — hybrid neuro-symbolic pipeline.
- **Housekeeping:** `.venv/`, Jupyter checkpoints, and large model binaries are gitignored.

---

## Team

- Divyanshi Sachan
- Subham Mahapatra

---

## Acknowledgments

Analysis uses the **Contract Understanding Atticus Dataset (CUAD)**. Cite the dataset and license terms from the [official CUAD release](https://www.atticusprojectai.org/cuad) when publishing or redistributing.
