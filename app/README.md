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
