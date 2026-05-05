# Agastya

**Agastya** is our end-to-end research and engineering attempt to automate **contract risk classification** from long legal documents.
Given a contract PDF or text, Agastya:
1. detects clause categories across 41 CUAD labels, and
2. infers a contract-level risk class (**Low / Medium / High**) with evidence-backed outputs.

This repository is not just the final model. It documents our full journey: what we planned, what failed, what we debugged, what we changed, and why the current architecture is what it is.

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

- [What We Set Out To Build](#what-we-set-out-to-build)
- [Research Journey and Process](#research-journey-and-process)
- [Experiments We Ran](#experiments-we-ran)
- [Hurdles, Failures, and Fixes](#hurdles-failures-and-fixes)
- [Final Results](#final-results)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Quickstart](#quickstart)
- [Notebooks and Reproducibility Workflow](#notebooks-and-reproducibility-workflow)
- [Reproducible Setup](#reproducible-setup)
- [Development Practices](#development-practices)
- [Team](#team)
- [Acknowledgments](#acknowledgments)

---

## What We Set Out To Build

Our objective was to build a **research-grade but deployable** system for legal contract risk assessment that:

- handles real-world long contracts (including scanned PDFs),
- does not collapse under class imbalance,
- provides interpretable contract-level reasoning,
- avoids demo-only shortcuts (hardcoded metrics, silent fallbacks, hidden assumptions),
- and can be rerun by another researcher with minimal ambiguity.

We treated Agastya as a phased research program rather than a one-shot model training exercise.

---

## Research Journey and Process

### Phase 1 (Classical Baseline): Establish a Strong, Explainable Reference

We started with a strict classical-ML baseline (TF-IDF + LinearSVC) to answer:

- How much legal signal is recoverable from surface lexical patterns alone?
- What does the imbalance profile look like per CUAD label?
- What should be considered an acceptable transformer gain later?

This gave us a stable benchmark and a clean sanity anchor for later deep learning experiments.

### Phase 2 (Legal-BERT): Improve Clause Detection Under Real Contract Lengths

Once the baseline stabilized, we moved to Legal-BERT with LoRA adapters for efficient fine-tuning.  
Core work in this phase was not only model training but **data/path plumbing**:

- long-document segmentation into <=512-token chunks,
- OCR fallback for non-machine-readable PDFs,
- label map stability across train/infer/eval,
- and artifact discipline so downstream phases could consume outputs reliably.

### Phase 3 (Neuro-Symbolic Risk Reasoning): Convert Clause Signals into Risk Decisions

The major challenge in Phase 3 was that accurate clause classification is necessary but not sufficient: we needed reliable contract-level reasoning.

We first attempted a Bayesian Network (our original neuro-symbolic intent), then replaced it with a data-driven Random Forest reasoner after ablation exposed structural limits in BN for this setting.

### Productization Layer

After modeling stabilized, we aligned the research pipeline with usage:

- web UI for upload -> analysis,
- strict artifact loading (no fake outputs if model files are missing),
- CLI evaluation to regenerate reports from current artifacts,
- notebook sequence that mirrors the research storyline and can be rerun.

---

## Experiments We Ran

### 1) Classical ML experiments (Phase 1)

- TF-IDF feature space variants (word and character n-grams).
- LinearSVC with `class_weight="balanced"` to address skewed clause distribution.
- Macro-F1 selected as primary metric due to minority-class sensitivity.

Outcome: established a hard-to-beat baseline and identified where transformer improvements mattered vs where they were marginal.

### 2) Legal-BERT experiments (Phase 2)

- Domain model selection: `nlpaueb/legal-bert-base-uncased`.
- LoRA-based fine-tuning to keep training efficient.
- Loss weighting and optimizer tuning (AdamW).
- Segment aggregation behavior for long contracts.

Outcome: improved macro-F1 and better minority label behavior than Phase 1, while keeping an efficient training path.

### 3) Risk reasoning experiments (Phase 3)

- **v1 (failed):** Bayesian Network with hand-curated structure/CPT assumptions.
- **v2 (final):** Random Forest trained on all 41 clause features (+ SMOTE + calibration).
- Thresholding study for BERT -> RF handoff confidence floor.
- Binary vs frequency-count feature vectorization for clause evidence.

Outcome: RF materially outperformed BN and avoided the manual-CPT bottleneck.

### 4) Ablation and interpretability experiments

- Phase 1 vs Phase 2 vs Phase 3 performance comparison.
- BN retained for controlled ablation, not production inference.
- RF feature-importance study for explanation surfaces.
- Oracle analysis: RF with ground-truth clauses to estimate upper bound.

Outcome: remaining live-system gap was traced primarily to upstream clause extraction noise, not RF reasoning capacity.

---

## Hurdles, Failures, and Fixes

### Hurdle 1: Bayesian Network underperformed drastically

**Problem**
- BN required manually specified Conditional Probability Tables for complex legal interactions.
- Only a small evidence subset was effectively used, causing large information loss.
- EM optimization on imbalanced data collapsed predictions toward dominant classes.

**What we saw**
- Macro-F1 collapsed to `0.159`.
- `Medium` risk class effectively disappeared in predictions.

**Fix**
- Replaced BN with Random Forest using all 41 clause features.
- Added class-balancing and calibration in the RF pipeline.
- Kept BN artifacts only as historical ablation evidence.

### Hurdle 2: Long contract handling and OCR variability

**Problem**
- Real contracts were inconsistent: some digitally extractable, others scanned or noisy.
- Single-pass tokenization was insufficient for long legal documents.

**Fix**
- Tiered text extraction (`pdfplumber` then EasyOCR fallback).
- Segment-based processing to preserve long-document coverage.
- Unified preprocessing path so training and inference remain aligned.

### Hurdle 3: Artifact drift and silent mismatch risks

**Problem**
- In multi-phase pipelines, stale model files or label maps can produce invalid predictions without obvious crashes.

**Fix**
- Strict artifact loading with explicit failure (`FileNotFoundError`) instead of fallback behavior.
- Metric reporting wired directly to artifact outputs, never static README constants.
- Dedicated eval CLI to regenerate report files deterministically.

### Hurdle 4: Class imbalance and threshold sensitivity

**Problem**
- Rare clause and risk classes were volatile under default settings.
- Naive confidence cutoffs introduced either over-triggering or under-reporting.

**Fix**
- Weighted training in Phase 2.
- SMOTE + calibration in Phase 3 RF.
- Confidence floor optimization (selected `0.11`) based on empirical trade-offs.

---

## Final Results

All metrics are loaded from artifacts (not hardcoded prose values).  
Regenerate hybrid metrics with:

```bash
python3 src/phase3/hybrid_eval_cli.py
```

### Phase Progression

| Phase | Model | Task | Macro-F1 | Accuracy |
|-------|-------|------|----------|----------|
| **Phase 1** | LinearSVC + TF-IDF | Clause classification (41 CUAD) | **0.719** | 0.800 |
| **Phase 2** | Legal-BERT (LoRA, intensive retrain) | Clause classification (41 CUAD) | **0.759** | 0.826 |
| Phase 3 v1 ❌ | Legal-BERT + Bayesian Network | Contract risk (Low/Med/High) | 0.159 | — |
| **Phase 3 v2 ✅** | **Legal-BERT + Random Forest** | **Contract risk (Low/Med/High)** | **0.866** | **0.882** |

Phase 2 and Hybrid numbers mirror `results/phase2/results.json` and `reports/phase3/hybrid_eval.json`.

### RF Oracle (Upper Bound)

When RF is fed **ground-truth clause labels** (instead of predicted Phase 2 outputs), it reaches **Macro-F1 = 0.903**.  
This indicates the current ceiling is mostly constrained by upstream clause extraction noise, not downstream RF capacity.

---

## Architecture

### Phase 1 — Classical ML Baseline

| Component | Choice | Why this choice |
|-----------|--------|-----------------|
| Features | TF-IDF (word + character n-grams) | High-signal legal lexical baseline; sparse + inspectable |
| Model | LinearSVC (`class_weight="balanced"`) | Strong linear baseline under class imbalance |
| Eval objective | Macro-F1 | Prevent dominance by frequent labels |

### Phase 2 — Legal-BERT Clause Classifier

| Component | Choice | Why this choice |
|-----------|--------|-----------------|
| Backbone | `nlpaueb/legal-bert-base-uncased` | Domain-pretrained legal encoder |
| Fine-tuning | LoRA adapters | Parameter-efficient adaptation |
| Segmentation | Segment extraction (<=512 tokens) | Long-contract compatibility |
| Training | PyTorch + HuggingFace + weighted loss | Stable and reproducible tuning stack |

### Phase 3 — Hybrid Neuro-Symbolic Reasoning

| Component | Choice | Why this choice |
|-----------|--------|-----------------|
| Upstream evidence | Frozen Legal-BERT outputs | Reuses strongest clause detector |
| Reasoner | Calibrated Random Forest (+ SMOTE) | Learns clause interactions from data, no manual CPTs |
| Confidence floor | 0.11 | Best observed precision-recall trade-off for evidence handoff |
| Feature vector | Clause frequency counts | Keeps occurrence intensity, not just presence/absence |
| OCR ingestion | `pdfplumber` -> EasyOCR fallback | Handles digital and scanned contracts |
| Delivery | Web UI | Usable interface for non-ML users |

### Non-Negotiable Design Decisions

- **No silent fallback models:** Missing artifacts should fail loudly.
- **No hardcoded reported metrics:** Reports always read generated files.
- **Historical BN retained only for analysis:** It is excluded from the live inference path.

---

## Folder Structure

```
agastya/
├── app/
│   ├── api.py                           # FastAPI backend for web UI inference
│   ├── README.md                        # How to run backend + frontend
│   └── web/                             # React + Vite frontend
├── configs/                             # YAML defaults (model, training, data)
├── data/
│   ├── CUAD_v1/                         # Phase 1 copy (master_clauses.csv, JSON)
│   ├── raw/                             # Scanned PDFs and raw CUAD
│   ├── interim/                         # OCR outputs, segmented clauses
│   └── processed/                       # train.csv / val.csv / test.csv
├── notebooks/
│   ├── Phase_1/                         # Part_01 ... Part_05 (classical ML)
│   ├── Phase_2/                         # 01 ... 09 (OCR -> BERT -> eval)
│   └── Phase_3/                         # 10 ... 13 (hybrid pipeline + ablations)
│       ├── 10_bn_structure_and_cpts.ipynb     # BN ablation (historical)
│       ├── 11_hybrid_pipeline_demo.ipynb      # End-to-end demo
│       ├── 12_ablation_study.ipynb            # ML vs DL vs Hybrid
│       └── 13_interpretability_report.ipynb   # RF feature importances
├── reports/
│   └── phase3/
│       ├── hybrid_eval.json             # Live eval results (RF backend)
│       ├── phase_progression_summary.csv
│       ├── ablation_results.csv
│       └── figures/                     # rf_feature_importance.png, etc.
├── results/
│   ├── phase1/results.json              # Phase 1 metrics
│   ├── phase2/
│   │   ├── results.json                 # Phase 2 metrics
│   │   ├── label2id.json                # 41-label CUAD map
│   │   └── models/                      # legal_bert_phase2.pt, LoRA adapter
│   └── phase3/
│       ├── rf_reasoner.pkl              # Trained RF reasoner
│       └── bayesian_network_seed.pkl    # BN ablation artifact only
├── src/
│   ├── phase2/                          # OCR, segmentation, BERT training, eval
│   └── phase3/
│       ├── hybrid_pipeline.py           # AgastyaHybridPipeline (BERT -> RF)
│       ├── hybrid_eval.py               # Batch evaluation
│       ├── hybrid_eval_cli.py           # Regenerates hybrid_eval.json
│       ├── rf_reasoner.py               # RF training and inference
│       ├── ablation.py                  # Ablation table builder
│       ├── interface/                   # phase2 adapter, evidence encoder, etc.
│       └── bayesian/                    # BN code (analysis only)
├── scripts/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quickstart

### Use the Web UI

Upload a contract PDF in the web UI to get:
- risk class (Low/Medium/High),
- clause-level evidence used for inference,
- and feature-driven explainability outputs.

Run full web stack from project root:

```bash
# Terminal 1 - backend API
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - frontend
cd app/web
npm install
npm run dev
```

Optional frontend API override:

```bash
cd app/web
VITE_API_URL=http://localhost:8000 npm run dev
```

### Regenerate Phase 3 Evaluation Metrics

```bash
# From project root - overwrites reports/phase3/hybrid_eval.json
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

## Notebooks and Reproducibility Workflow

All Phase 3 notebooks start with a path bootstrap cell that resolves `PROJECT_ROOT` so they can be launched from any working directory.

| Notebook | Purpose | Key Output |
|----------|---------|------------|
| `Phase_1/Part_01` - `Part_05` | Classical ML baseline development | SVM baseline metrics |
| `Phase_2/01` - `09` | OCR, segmentation, Legal-BERT training + eval | Phase 2 clause metrics |
| `Phase_3/10_bn_structure_and_cpts` | BN failure analysis (ablation) | CPT/structure diagnostics |
| `Phase_3/11_hybrid_pipeline_demo` | End-to-end hybrid demo | Live contract risk prediction |
| `Phase_3/12_ablation_study` | Cross-phase performance comparison | Consolidated ablation table |
| `Phase_3/13_interpretability_report` | RF explainability + comparisons | `rf_feature_importance.png` |

Recommended Phase 3 execution order:
1. Notebook 11 (end-to-end sanity).
2. `python3 src/phase3/hybrid_eval_cli.py`.
3. Notebook 12 (ablation reads fresh report artifacts).
4. Notebook 13 (interpretability from current RF artifact).
5. Notebook 10 (historical BN analysis, independent of RF path).

---

## Reproducible Setup

**Prerequisites**

- Python 3.10+
- Poppler for PDF rasterization in OCR fallback:
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

**Data**

CUAD v1 is included under `data/CUAD_v1/`.  
Phase 3 processed splits (`train.csv`, `val.csv`, `test.csv`) come from the Phase 2 preprocessing/segmentation flow.

**Determinism**

Training uses fixed seeds (`random_state=42` where applicable).  
For full reproducibility after pulling updates, rerun notebooks top-to-bottom and regenerate reports via CLI.

---

## Development Practices

- **Small coherent commits:** keep history reviewable and hypothesis-driven.
- **No patchwork metrics:** all reported values come from artifacts generated by code.
- **Strict phase boundaries:** baseline -> encoder -> reasoner, each with explicit scope.
- **Fail loudly over failing silently:** missing artifacts should break fast.
- **Repository hygiene:** `.venv/`, checkpoints, and bulky binaries are excluded.

---

## Team

- Divyanshi Sachan
- Subham Mahapatra

---

## Acknowledgments

This work uses the **Contract Understanding Atticus Dataset (CUAD)**.  
Please cite CUAD and follow license/distribution constraints from the [official release](https://www.atticusprojectai.org/cuad).
