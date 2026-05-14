# ──────────────────────────────────────────────────────────────────────────────
# Audio Deepfake Detector — FastAPI inference server
#
# Build:
#   docker build -t audio-deepfake-detector .
#
# Run (checkpoint must exist at ./checkpoints/best.pt on the host):
#   docker run -p 8000:8000 \
#     -v $(pwd)/checkpoints:/app/checkpoints:ro \
#     audio-deepfake-detector
#
# Then hit http://localhost:8000/docs for the interactive API.
# ──────────────────────────────────────────────────────────────────────────────

# Python 3.12-slim: smallest official image that has PyTorch CPU wheels.
# (PyTorch wheel support for newer Python versions lags by several months.)
FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
# ffmpeg: required by pydub so the /predict endpoint can accept mp3 / m4a files.
# Without it the endpoint still works for wav/flac/ogg — pydub is only the
# fallback path when soundfile cannot read the format directly.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only the dependency manifest first so Docker can cache this layer.
# Re-installing packages only happens when pyproject.toml changes, not on every
# source-code edit — a big time saver during iterative builds.
COPY pyproject.toml .

# Install CPU-only PyTorch first (explicit index URL keeps the CUDA wheels out).
# CPU torch ≈ 800 MB vs GPU torch ≈ 2.5 GB — essential for a portable image.
RUN pip install --no-cache-dir \
    torch==2.2.* \
    torchaudio==2.2.* \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the project dependencies.
# The editable install (-e .) also registers the `src` package on sys.path so
# imports like `from src.models.lcnn import LCNN` resolve correctly.
RUN pip install --no-cache-dir -e .

# ── Source code ───────────────────────────────────────────────────────────────
# Copy only the packages the server actually needs at runtime.
# Training scripts, notebooks, and data are intentionally excluded.
COPY src/  src/
COPY api/  api/

# ── Runtime configuration ─────────────────────────────────────────────────────
# The model checkpoint is NOT baked into the image — it is mounted at runtime.
# This keeps the image reusable across different trained checkpoints and avoids
# committing large binary files to the Docker layer cache.
#
# Expected mount: -v /path/to/checkpoints:/app/checkpoints:ro
# The server looks for:  /app/checkpoints/best.pt   (see api/main.py)

EXPOSE 8000

# --workers 1: single worker is correct here because the model is loaded once
# into app.state during the lifespan hook.  Multiple workers would each load
# their own copy, multiplying RAM usage without benefit for a CPU-bound model.
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
