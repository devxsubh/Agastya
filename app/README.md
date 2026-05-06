# Agastya Web UI Module

This directory contains the complete web application layer for Agastya:

- `api.py` -> FastAPI backend serving model inference endpoints.
- `web/` -> React + Vite frontend for contract upload and analysis visualizations.

## Architecture

```
Browser (React/Vite)
   -> POST /analyze
   -> POST /classify-clause
FastAPI (`app/api.py`)
   -> AgastyaHybridPipeline (Legal-BERT + RF)
   -> returns risk scores, evidence, and feature importances
```

## Run locally

From project root:

1) Start backend

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

2) Start frontend (new terminal)

```bash
cd app/web
npm install
npm run dev
```

Frontend default URL is shown by Vite (usually `http://localhost:5173`).
Backend default URL is `http://localhost:8000`.

## Environment variable

The frontend reads API base URL from `VITE_API_URL`.

Example:

```bash
cd app/web
VITE_API_URL=http://localhost:8000 npm run dev
```

If unset, frontend falls back to `http://localhost:8000`.

## Backend artifact paths (deployment)

Inference needs trained files under `results/` (many are **gitignored** and must be uploaded or mounted on the host):

| Artifact | Default path |
|---------|----------------|
| Phase 2 label map | `results/phase2/label2id.json` |
| Legal-BERT LoRA adapter (directory) | `results/phase2/models/legal_bert_lora_adapter/` |
| Legal-BERT merged checkpoint (optional alternative) | `results/phase2/models/legal_bert_phase2.pt` |
| Phase 3 RF reasoner | `results/phase3/rf_reasoner.pkl` |

Override with environment variables (absolute paths or paths relative to the repo root):

- `AGASTYA_RF_MODEL_PATH`
- `AGASTYA_PHASE2_LABEL_MAP`
- `AGASTYA_PHASE2_ADAPTER_PATH` (set to empty string to use only the merged checkpoint)
- `AGASTYA_PHASE2_BERT_CHECKPOINT` (set to empty string to use only the LoRA adapter)

`GET /health` includes `artifacts_ready` and the resolved paths for debugging.
