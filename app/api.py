"""FastAPI backend for the Agastya web UI.

Exposes a single POST /analyze endpoint that accepts a contract file (PDF or TXT)
and returns the full risk analysis payload from the hybrid pipeline.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MAX_CONTRACT_PREVIEW_CHARS = 24_000
MAX_CLAUSE_CLASSIFY_CHARS = 8_000
BERT_DETAIL_TEXT_CHARS = 2_000

# Allow imports from project root
PROJECT_ROOT = next(
    (p for p in [Path(__file__).parent.parent, *Path(__file__).parent.parent.parents]
     if (p / "src").exists()),
    None,
)
if PROJECT_ROOT and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.hybrid_pipeline import AgastyaHybridPipeline
from src.phase3.ocr.extractor import extract_text
from src.phase3.rf_reasoner import get_feature_importances, load_rf_reasoner

app = FastAPI(title="Agastya API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton pipeline loaded once
_pipeline: AgastyaHybridPipeline | None = None
_feature_importances: list[dict] | None = None


def _get_pipeline() -> AgastyaHybridPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AgastyaHybridPipeline(
            rf_model_path=str(PROJECT_ROOT / "results/phase3/rf_reasoner.pkl"),
            bn_model_path=None,
            bert_checkpoint_path=str(PROJECT_ROOT / "results/phase2/models/legal_bert_phase2.pt"),
            label_map_path=str(PROJECT_ROOT / "results/phase2/label2id.json"),
            adapter_path=str(PROJECT_ROOT / "results/phase2/models/legal_bert_lora_adapter"),
        )
    return _pipeline


def _get_feature_importances() -> list[dict]:
    global _feature_importances
    if _feature_importances is None:
        reasoner = load_rf_reasoner(str(PROJECT_ROOT / "results/phase3/rf_reasoner.pkl"))
        fi_df = get_feature_importances(reasoner)
        _feature_importances = fi_df.to_dict(orient="records")
    return _feature_importances


class SegmentAnnotation(BaseModel):
    """One segmented passage + primary BERT label + heuristic clause-risk tier for UI markup."""

    segment_index: int
    text: str
    phase2_label: str
    mapped_bucket: str
    confidence: float
    clause_risk: str


class AnalysisResponse(BaseModel):
    risk_level: str
    risk_probabilities: dict[str, float]
    clause_evidence: dict[str, Any]
    bert_details: list[dict]
    feature_importances: list[dict]
    reasoner: str
    n_clauses: int
    contract_text_preview: str = ""
    contract_char_total: int = 0
    contract_truncated: bool = False
    segment_annotations: list[SegmentAnnotation] = []


class ClassifyClauseBody(BaseModel):
    """Run Legal-BERT only on a user-selected passage (clause classification)."""

    text: str = Field(..., min_length=4, max_length=MAX_CLAUSE_CLASSIFY_CHARS)


class ClausePredictionItem(BaseModel):
    phase2_label: str
    mapped_bucket: str
    confidence: float


class ClassifyClauseResponse(BaseModel):
    predictions: list[ClausePredictionItem]
    char_count: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "Legal-BERT + RF (Hybrid v2)"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    content = await file.read()

    try:
        if suffix == ".pdf":
            text = extract_text(io.BytesIO(content), suffix=suffix)
        else:
            text = content.decode("utf-8", errors="replace")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to extract text: {exc}") from exc

    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file")

    stripped = text.strip()
    contract_truncated = len(stripped) > MAX_CONTRACT_PREVIEW_CHARS
    preview = stripped[:MAX_CONTRACT_PREVIEW_CHARS]

    try:
        pipeline = _get_pipeline()
        result = pipeline.predict(text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    bert_details = result.get("bert_details", [])
    # Sanitize bert_details — remove numpy arrays (embeddings) for JSON serialisation
    clean_details = []
    for d in bert_details:
        clean_details.append({
            "clause_type": str(d.get("phase2_label", d.get("clause_type", "Other"))),
            "confidence": float(d.get("confidence", 0.0)),
            "text": str(d.get("clause_text", d.get("text", "")))[:BERT_DETAIL_TEXT_CHARS],
        })

    anno_raw = result.get("segment_annotations") or []
    segment_annotations: list[SegmentAnnotation] = []
    for row in anno_raw:
        if not isinstance(row, dict):
            continue
        segment_annotations.append(
            SegmentAnnotation(
                segment_index=int(row.get("segment_index", len(segment_annotations))),
                text=str(row.get("text", ""))[:BERT_DETAIL_TEXT_CHARS],
                phase2_label=str(row.get("phase2_label", "Unknown")),
                mapped_bucket=str(row.get("mapped_bucket", "Other")),
                confidence=float(row.get("confidence", 0.0)),
                clause_risk=str(row.get("clause_risk", "Low")),
            )
        )

    return AnalysisResponse(
        risk_level=result["risk_level"],
        risk_probabilities={k: float(v) for k, v in result.get("risk_probabilities", result.get("risk_probabilities", {})).items()},
        clause_evidence={k: str(v) for k, v in result.get("clause_evidence", {}).items()},
        bert_details=clean_details,
        feature_importances=_get_feature_importances()[:15],
        reasoner=result.get("reasoner", "rf"),
        n_clauses=len(clean_details),
        contract_text_preview=preview,
        contract_char_total=len(stripped),
        contract_truncated=contract_truncated,
        segment_annotations=segment_annotations,
    )


@app.post("/classify-clause", response_model=ClassifyClauseResponse)
def classify_clause(body: ClassifyClauseBody) -> ClassifyClauseResponse:
    """Top-K Legal-BERT predictions for an arbitrary passage (no RF contract risk)."""
    raw = body.text.strip()
    if len(raw) < 4:
        raise HTTPException(status_code=422, detail="Text too short.")
    try:
        pipeline = _get_pipeline()
        preds = pipeline.bert.predict(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classifier error: {exc}") from exc

    clean: list[ClausePredictionItem] = []
    for d in preds:
        clean.append(
            ClausePredictionItem(
                phase2_label=str(d.get("phase2_label", "Unknown")),
                mapped_bucket=str(d.get("clause_type", "Other")),
                confidence=float(d.get("confidence", 0.0)),
            )
        )
    return ClassifyClauseResponse(predictions=clean, char_count=len(raw))
