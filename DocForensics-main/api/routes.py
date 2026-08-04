import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile

from api import train_manager
from api.schemas import (AnalyzeResponse, DatasetCounts, DetectorResult,
                          TrainStartResponse, TrainStatusResponse)
from core.config import SUPPORTED_EXTS
from core.pipeline import analyze

router = APIRouter()


def _heatmap_to_base64(heatmap: np.ndarray) -> str:
    heat_u8  = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    colored  = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    _, buf   = cv2.imencode('.png', colored)
    return base64.b64encode(buf.tobytes()).decode()


@router.post('/analyze', response_model=AnalyzeResponse)
async def analyze_document(file: UploadFile):
    ext = Path(file.filename or '').suffix.lower()
    mime_to_ext = {
        'image/jpeg': '.jpg', 'image/png': '.png',
        'image/tiff': '.tiff', 'application/pdf': '.pdf',
    }
    if ext not in SUPPORTED_EXTS:
        ext = mime_to_ext.get(file.content_type or '', '')
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail=f'Unsupported file type: {file.filename!r} ({file.content_type})')

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        verdict = analyze(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return AnalyzeResponse(
        is_tampered=verdict.is_tampered,
        label=verdict.label,
        confidence=round(verdict.confidence, 4),
        evidence=verdict.evidence,
        per_detector=[
            DetectorResult(name=d.detector_name, score=round(d.score, 4), details=d.details)
            for d in verdict.per_detector
        ],
        heatmap_base64=_heatmap_to_base64(verdict.fused_heatmap),
        class_scores=verdict.class_scores,
    )


@router.get('/health')
def health():
    return {'status': 'ok'}


# ── Train section ────────────────────────────────────────────────────────
# Exposes model/train.py (TamperNet: Original vs Forged classifier + mask
# decoder) so the frontend can kick off and monitor training runs.

@router.post('/train/start', response_model=TrainStartResponse)
def start_training():
    started = train_manager.start_training()
    return TrainStartResponse(started=started, status=train_manager.get_status()['status'])


@router.get('/train/status', response_model=TrainStatusResponse)
def train_status():
    st = train_manager.get_status()
    return TrainStatusResponse(
        status=st['status'],
        current_epoch=st['current_epoch'],
        total_epochs=st['total_epochs'],
        history=st['history'],
        error=st['error'],
        dataset=DatasetCounts(**train_manager.dataset_counts()),
        has_checkpoint=train_manager.has_checkpoint(),
    )