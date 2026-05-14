import torch
from pathlib import Path
from src.data.dataset import load_waveform
from src.data.transforms import MelSpectrogramTransform
from src.models.lcnn import LCNN

LABEL_NAMES = {0: "bonafide", 1: "spoof"}
_transform = MelSpectrogramTransform(augment=False)


def load_model(checkpoint_path: str | Path, device: torch.device) -> LCNN:
    model = LCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model


def predict(audio_path: str | Path, model: LCNN, device: torch.device) -> dict:
    """Run inference on a single audio file.

    Returns a dict with label, confidence, and per-class scores.
    """
    waveform = load_waveform(audio_path)
    mel      = _transform(waveform).unsqueeze(0).to(device)  # [1, 1, 128, 251]

    with torch.no_grad():
        logits = model(mel)
        probs  = torch.softmax(logits, dim=1)[0]

    bonafide_score = probs[0].item()
    spoof_score    = probs[1].item()
    predicted_class = int(probs.argmax().item())

    return {
        "label":      LABEL_NAMES[predicted_class],
        "confidence": max(bonafide_score, spoof_score),
        "scores": {
            "bonafide": round(bonafide_score, 4),
            "spoof":    round(spoof_score, 4),
        },
    }
