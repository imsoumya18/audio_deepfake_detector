import torch
import torchaudio
import soundfile as sf
import numpy as np
import tempfile
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

SUPPORTED_BY_SOUNDFILE = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def _convert_to_wav(file_path: Path) -> Path:
    """Convert any audio format to a temporary wav file using pydub."""
    from pydub import AudioSegment
    audio = AudioSegment.from_file(str(file_path))
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    return Path(tmp.name)

SAMPLE_RATE = 16_000
MAX_SAMPLES = 4 * SAMPLE_RATE  # 4 seconds = 64,000 samples

LABEL_MAP = {"bonafide": 0, "spoof": 1}


def load_waveform(file_path: str | Path) -> torch.Tensor:
    """Load any audio file and return a mono 16kHz waveform of fixed length.

    Tries soundfile first (fast, no dependencies). Falls back to pydub for
    formats like mp3/m4a (requires ffmpeg installed on the system).
    Returns shape: [1, 64000]
    """
    file_path = Path(file_path)
    tmp_path  = None

    try:
        data, sr = sf.read(file_path, dtype="float32", always_2d=True)
    except Exception:
        # soundfile couldn't read it — try pydub conversion
        try:
            tmp_path  = _convert_to_wav(file_path)
            data, sr  = sf.read(tmp_path, dtype="float32", always_2d=True)
        except FileNotFoundError:
            raise ValueError(
                f"Cannot read '{file_path.suffix}' files. "
                "Install ffmpeg for mp3/m4a support, or use wav/flac."
            )
        finally:
            if tmp_path:
                tmp_path.unlink(missing_ok=True)
    # soundfile returns [samples, channels] — transpose to [channels, samples]
    waveform = torch.from_numpy(data.T)

    # Convert stereo to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if the file isn't already 16kHz
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Pad short clips with zeros, truncate long clips
    length = waveform.shape[1]
    if length < MAX_SAMPLES:
        pad = MAX_SAMPLES - length
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :MAX_SAMPLES]

    return waveform  # [1, 64000]


class ASVspoofDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        """
        df       : DataFrame from parse_protocol()
        transform: optional callable applied to the raw waveform tensor
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]

        waveform = load_waveform(row["file_path"])

        if self.transform:
            waveform = self.transform(waveform)

        label = LABEL_MAP[row["label"]]
        return waveform, label
