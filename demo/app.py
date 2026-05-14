import torch
import gradio as gr
from pathlib import Path

from src.inference.predict import load_model, predict
from src.data.dataset import load_waveform
from src.data.transforms import MelSpectrogramTransform
from src.explainability.gradcam import GradCAM

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Gradio
import matplotlib.pyplot as plt
import numpy as np

CHECKPOINT = Path("checkpoints/best.pt")

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device    = get_device()
model     = load_model(CHECKPOINT, device)
gradcam   = GradCAM(model, device)
transform = MelSpectrogramTransform(augment=False)


def run_inference(audio_path: str):
    if audio_path is None:
        return "No file uploaded", 0.0, None

    result   = predict(audio_path, model, device)
    label    = "🟢 Real (Bonafide)" if result["label"] == "bonafide" else "🔴 Fake (AI-Generated)"
    confidence = result["confidence"]

    # Generate Grad-CAM
    waveform = load_waveform(audio_path)
    mel      = transform(waveform).unsqueeze(0)
    cam      = gradcam.compute(mel, target_class=1)
    mel_np   = mel.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    axes[0].imshow(mel_np, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("Mel-Spectrogram")
    axes[0].set_xlabel("Time frames")
    axes[0].set_ylabel("Mel band")

    axes[1].imshow(mel_np, origin="lower", aspect="auto", cmap="magma", alpha=0.6)
    axes[1].imshow(cam,    origin="lower", aspect="auto", cmap="jet",   alpha=0.5)
    axes[1].set_title("Grad-CAM (model attention)")
    axes[1].set_xlabel("Time frames")
    axes[1].set_ylabel("Mel band")

    plt.tight_layout()

    return label, round(confidence, 4), fig


demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Audio(type="filepath", label="Upload audio clip (.wav or .flac)"),
    outputs=[
        gr.Textbox(label="Detection Result"),
        gr.Number(label="Confidence"),
        gr.Plot(label="Spectrogram + Grad-CAM"),
    ],
    title="Audio Deepfake Detector",
    description=(
        "Upload a speech clip to check if it is real human speech or AI-generated. "
        "The Grad-CAM heatmap shows which frequency regions the model focused on."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
