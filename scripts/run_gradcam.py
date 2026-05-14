import torch
from pathlib import Path
from src.data.protocol import parse_protocol
from src.data.dataset import load_waveform
from src.data.transforms import MelSpectrogramTransform
from src.models.lcnn import LCNN
from src.explainability.gradcam import GradCAM

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = LCNN()
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
gradcam = GradCAM(model, device)
transform = MelSpectrogramTransform(augment=False)

data_root = Path("data")
protocols = data_root / "ASVspoof2019_LA_cm_protocols"
df_eval   = parse_protocol(
    protocols / "ASVspoof2019.LA.cm.eval.trl.txt",
    data_root / "ASVspoof2019_LA_eval" / "flac",
)

# Pick one sample from each category
cases = {
    "bonafide":      df_eval[df_eval["label"] == "bonafide"].iloc[0],
    "spoof_A07_easy": df_eval[df_eval["attack_type"] == "A07"].iloc[0],
    "spoof_A17_hard": df_eval[df_eval["attack_type"] == "A17"].iloc[0],
    "spoof_A18_hard": df_eval[df_eval["attack_type"] == "A18"].iloc[0],
}

for name, row in cases.items():
    waveform = load_waveform(row["file_path"])
    mel      = transform(waveform).unsqueeze(0)   # [1, 1, 128, 251]

    # Get model prediction
    with torch.no_grad():
        logits = model(mel.to(device))
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = "spoof" if probs[1] > 0.5 else "bonafide"
        conf   = probs[1].item()

    print(f"\n{name}")
    print(f"  True label : {row['label']} ({row['attack_type']})")
    print(f"  Prediction : {pred}  (spoof confidence: {conf:.3f})")

    cam   = gradcam.compute(mel, target_class=1)
    title = f"{name} | true={row['label']} | pred={pred} ({conf:.2f})"
    gradcam.visualize(mel, cam, title=title, save_path=f"notebooks/gradcam_{name}.png")
