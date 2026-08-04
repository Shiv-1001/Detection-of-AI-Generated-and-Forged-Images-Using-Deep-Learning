from pydantic import BaseModel


class DetectorResult(BaseModel):
    name: str
    score: float
    details: dict


class AnalyzeResponse(BaseModel):
    is_tampered: bool
    label: str
    confidence: float
    evidence: list[str]
    per_detector: list[DetectorResult]
    heatmap_base64: str
    # Ordered 3-way breakdown: Original -> AI Generated -> Forged
    class_scores: dict[str, float] = {}


class DatasetCounts(BaseModel):
    genuine: int
    tampered: int


class TrainStartResponse(BaseModel):
    started: bool
    status: str


class TrainStatusResponse(BaseModel):
    status: str                # idle | running | done | error
    current_epoch: int
    total_epochs: int
    history: list[dict]
    error: str | None = None
    dataset: DatasetCounts
    has_checkpoint: bool