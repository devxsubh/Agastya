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
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
MAX_INFER_TEXT_CHARS = int(os.getenv("MAX_INFER_TEXT_CHARS", "220000"))

# Allow imports from project root
PROJECT_ROOT = next(
    (p for p in [Path(__file__).parent.parent, *Path(__file__).parent.parent.parents]
     if (p / "src").exists()),
    None,
)
if PROJECT_ROOT and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.hybrid_pipeline import AgastyaHybridPipeline
from src.phase3.interface.phase2_adapter import describe_phase2_artifact_gaps
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


def _require_project_root() -> Path:
    if PROJECT_ROOT is None:
        raise RuntimeError("Could not resolve project root (expected parent of src/).")
    return PROJECT_ROOT


def _repo_path(env_var: str, relative_default: str) -> str:
    """Absolute path: env overrides default under project root."""
    root = _require_project_root()
    raw = os.getenv(env_var)
    if raw is None or not str(raw).strip():
        return str((root / relative_default).resolve())
    p = Path(str(raw).strip())
    return str(p.resolve() if p.is_absolute() else (root / p).resolve())


def _repo_path_optional(env_var: str, relative_default: str) -> str | None:
    """Like _repo_path, but if env is set to empty string, return None."""
    root = _require_project_root()
    if env_var not in os.environ:
        return str((root / relative_default).resolve())
    raw = os.getenv(env_var, "").strip()
    if raw == "":
        return None
    p = Path(raw)
    return str(p.resolve() if p.is_absolute() else (root / p).resolve())


def _artifact_paths() -> tuple[str, str | None, str | None, str]:
    """Resolved RF path, label map, optional adapter dir, optional merged BERT ckpt."""
    rf_path = _repo_path("AGASTYA_RF_MODEL_PATH", "results/phase3/rf_reasoner.pkl")
    label_map = _repo_path("AGASTYA_PHASE2_LABEL_MAP", "results/phase2/label2id.json")
    adapter = _repo_path_optional("AGASTYA_PHASE2_ADAPTER_PATH", "results/phase2/models/legal_bert_lora_adapter")
    ckpt = _repo_path_optional("AGASTYA_PHASE2_BERT_CHECKPOINT", "results/phase2/models/legal_bert_phase2.pt")
    return rf_path, label_map, adapter, ckpt


def _ensure_inference_artifacts_or_raise() -> tuple[str, str | None, str | None, str]:
    rf_path, label_map, adapter, ckpt = _artifact_paths()
    errors = describe_phase2_artifact_gaps(ckpt, label_map, adapter)
    if not Path(rf_path).exists():
        errors.append(
            f"RF reasoner missing: {rf_path}. Set AGASTYA_RF_MODEL_PATH or add results/phase3/rf_reasoner.pkl."
        )
    if errors:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Inference artifacts missing or incomplete.",
                "hints": errors,
            },
        )
    return rf_path, label_map, adapter, ckpt


def _get_pipeline() -> AgastyaHybridPipeline:
    global _pipeline
    if _pipeline is None:
        rf_path, label_map, adapter, ckpt = _ensure_inference_artifacts_or_raise()
        _pipeline = AgastyaHybridPipeline(
            rf_model_path=rf_path,
            bn_model_path=None,
            bert_checkpoint_path=ckpt,
            label_map_path=label_map,
            adapter_path=adapter,
        )
    return _pipeline


def _get_feature_importances() -> list[dict]:
    global _feature_importances
    if _feature_importances is None:
        rf_path, _, _, _ = _ensure_inference_artifacts_or_raise()
        reasoner = load_rf_reasoner(rf_path)
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
    try:
        rf_path, label_map, adapter, ckpt = _artifact_paths()
        gaps = describe_phase2_artifact_gaps(ckpt, label_map, adapter)
        rf_ok = Path(rf_path).exists()
        ready = not gaps and rf_ok
        return {
            "status": "ok" if ready else "degraded",
            "model": "Legal-BERT + RF (Hybrid v2)",
            "artifacts_ready": ready,
            "paths": {
                "rf_reasoner": rf_path,
                "phase2_label_map": label_map,
                "phase2_adapter": adapter,
                "phase2_checkpoint": ckpt,
            },
        }
    except Exception:
        return {"status": "error", "model": "Legal-BERT + RF (Hybrid v2)", "artifacts_ready": False}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

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
    if len(stripped) > MAX_INFER_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                "Extracted contract text is too large for synchronous inference. "
                f"Current limit: {MAX_INFER_TEXT_CHARS} chars."
            ),
        )
    contract_truncated = len(stripped) > MAX_CONTRACT_PREVIEW_CHARS
    preview = stripped[:MAX_CONTRACT_PREVIEW_CHARS]

    try:
        pipeline = _get_pipeline()
        result = pipeline.predict(text)
    except HTTPException:
        raise
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
    except HTTPException:
        raise
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
