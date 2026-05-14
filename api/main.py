import torch
import tempfile
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException

from api.schemas import PredictionResponse
from src.inference.predict import load_model, predict


CHECKPOINT = Path("checkpoints/best.pt")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    device = get_device()
    app.state.device = device
    app.state.model  = load_model(CHECKPOINT, device)
    print(f"Model loaded on {device}")
    yield
    # Cleanup on shutdown (nothing needed here)


app = FastAPI(
    title="Audio Deepfake Detector",
    description="Detects whether an audio clip is real human speech or AI-generated.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(audio_file: UploadFile = File(...)):
    # Validate file size
    contents = await audio_file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    # Validate file extension
    suffix = Path(audio_file.filename).suffix.lower()
    if suffix not in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff"}:
        raise HTTPException(status_code=400, detail="Unsupported format. Use wav, flac, mp3, m4a, ogg, or aiff.")

    # Write to temp file, run inference, clean up
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        result = predict(tmp_path, app.state.model, app.state.device)
    finally:
        tmp_path.unlink(missing_ok=True)

    return PredictionResponse(**result)
