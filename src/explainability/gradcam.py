import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

from src.models.lcnn import LCNN


class GradCAM:
    """Grad-CAM for LCNN on mel-spectrograms.

    Hooks into the last ConvBlock of LCNN's feature extractor to produce
    a heatmap showing which time-frequency regions influenced the prediction.
    """

    def __init__(self, model: LCNN, device: torch.device):
        self.model  = model.to(device).eval()
        self.device = device

        self._activations = {}
        self._gradients   = {}

        # Hook into the last ConvBlock (index 9 in self.features)
        target_layer = model.features[9]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations["feat"] = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients["feat"] = grad_output[0].detach()

    def compute(self, mel: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        mel          : [1, 1, 128, T] — single mel-spectrogram on device
        target_class : 1 = spoof (default), 0 = bonafide

        Returns heatmap as numpy array [128, T], values in [0, 1].
        """
        mel = mel.to(self.device).requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(mel)                          # [1, 2]
        score  = logits[0, target_class]
        score.backward()

        # Grad-CAM: global average pool gradients → weights
        grads   = self._gradients["feat"]                 # [1, C, H, W]
        acts    = self._activations["feat"]               # [1, C, H, W]
        weights = grads.mean(dim=[2, 3], keepdim=True)    # [1, C, 1, 1]

        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # [1, 1, H, W]
        cam = F.interpolate(cam, size=mel.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def visualize(
        self,
        mel: torch.Tensor,
        cam: np.ndarray,
        title: str = "",
        save_path: str | Path | None = None,
    ):
        """Overlay Grad-CAM heatmap on the mel-spectrogram."""
        mel_np = mel.squeeze().cpu().numpy()  # [128, T]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Left: raw mel-spectrogram
        axes[0].imshow(mel_np, origin="lower", aspect="auto", cmap="magma")
        axes[0].set_title("Mel-Spectrogram")
        axes[0].set_xlabel("Time frames")
        axes[0].set_ylabel("Mel band")

        # Right: spectrogram with Grad-CAM overlay
        axes[1].imshow(mel_np, origin="lower", aspect="auto", cmap="magma", alpha=0.6)
        axes[1].imshow(cam,    origin="lower", aspect="auto", cmap="jet",   alpha=0.5)
        axes[1].set_title(f"Grad-CAM — {title}")
        axes[1].set_xlabel("Time frames")
        axes[1].set_ylabel("Mel band")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()
        plt.close()
