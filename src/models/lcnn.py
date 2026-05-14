import torch
import torch.nn as nn


def mfm(x: torch.Tensor) -> torch.Tensor:
    """Max-Feature-Map activation: splits channels in half, returns element-wise max."""
    x1, x2 = x.chunk(2, dim=1)
    return torch.max(x1, x2)


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → MFM. The basic building block of LCNN."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, padding=0):
        super().__init__()
        # out_channels*2 because MFM halves the channels
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, padding=padding)
        self.bn   = nn.BatchNorm2d(out_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mfm(self.bn(self.conv(x)))


class LCNN(nn.Module):
    """
    Light CNN with Max-Feature-Map activations for audio deepfake detection.
    Input : [batch, 1, 128, 251]  (log mel-spectrogram)
    Output: [batch, 2]            (logits for bonafide / spoof)
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,  32, kernel_size=5, padding=2),  # [b, 32, 128, 251]
            nn.MaxPool2d(2, 2),                            # [b, 32,  64, 125]

            ConvBlock(32, 32, kernel_size=1),              # [b, 32,  64, 125]
            ConvBlock(32, 32, kernel_size=3, padding=1),   # [b, 32,  64, 125]
            nn.MaxPool2d(2, 2),                            # [b, 32,  32,  62]

            ConvBlock(32, 32, kernel_size=1),              # [b, 32,  32,  62]
            ConvBlock(32, 32, kernel_size=3, padding=1),   # [b, 32,  32,  62]
            nn.MaxPool2d(2, 2),                            # [b, 32,  16,  31]

            ConvBlock(32, 64, kernel_size=1),              # [b, 64,  16,  31]
            ConvBlock(64, 32, kernel_size=3, padding=1),   # [b, 32,  16,  31]
            nn.MaxPool2d(2, 2),                            # [b, 32,   8,  15]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 15, 160),
            nn.Dropout(0.75),
            nn.Linear(160, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
