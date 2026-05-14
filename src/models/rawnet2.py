import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SincConv(nn.Module):
    """
    Learnable sinc-based bandpass filterbank operating on raw waveforms.
    Each filter is defined by two parameters: low cutoff f1 and high cutoff f2.
    """

    def __init__(self, n_filters: int = 70, kernel_size: int = 1024, sample_rate: int = 16000):
        super().__init__()
        assert kernel_size % 2 == 0, "kernel_size must be even"
        self.n_filters   = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Initialize cutoff frequencies on the mel scale (good starting point)
        low_hz  = 30.0
        high_hz = sample_rate / 2 - 100

        mel_low  = self._hz_to_mel(low_hz)
        mel_high = self._hz_to_mel(high_hz)
        mel_points = torch.linspace(mel_low, mel_high, n_filters + 2)
        hz_points  = self._mel_to_hz(mel_points)

        # f1 = lower cutoff, f2 = bandwidth (both learned)
        self.f1 = nn.Parameter(hz_points[:-2].unsqueeze(1))   # [n_filters, 1]
        self.f2 = nn.Parameter((hz_points[1:-1] - hz_points[:-2]).unsqueeze(1))  # [n_filters, 1]

        # Hamming window to reduce spectral leakage at filter edges
        n = torch.arange(kernel_size // 2)
        self.register_buffer("window", 0.54 - 0.46 * torch.cos(2 * np.pi * n / kernel_size))
        self.register_buffer("t", torch.arange(1, kernel_size // 2 + 1).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Constrain frequencies to valid range
        f1 = torch.abs(self.f1) / self.sample_rate
        f2 = torch.abs(self.f1 + torch.abs(self.f2)) / self.sample_rate
        f2 = torch.max(f2, f1 + 1e-6)           # f2 > f1
        f2 = torch.clamp(f2, max=0.5)           # f2 < Nyquist

        # Build bandpass filters: lowpass(f2) - lowpass(f1)
        t = self.t.unsqueeze(0)                       # [1, kernel_size//2]
        lp_f2 = 2 * f2 * torch.sinc(2 * f2 * t)      # [n_filters, kernel_size//2]
        lp_f1 = 2 * f1 * torch.sinc(2 * f1 * t)      # [n_filters, kernel_size//2]
        band   = (lp_f2 - lp_f1) * self.window       # apply Hamming window

        # Make filters symmetric (even function) for zero phase response
        filters = torch.cat([band.flip(1), band], dim=1).unsqueeze(1)  # [n_filters, 1, kernel_size]

        return F.conv1d(x, filters, stride=16, padding=self.kernel_size // 2)

    @staticmethod
    def _hz_to_mel(hz):
        return 2595 * torch.log10(torch.tensor(1.0 + hz / 700)) if isinstance(hz, float) \
               else 2595 * torch.log10(1.0 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)


class ResBlock(nn.Module):
    """1D residual block with optional downsampling via stride."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels,  out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        # Skip connection: match dimensions if channels or time length changed
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        out = self.bn2(self.conv2(out))
        return F.leaky_relu(out + self.shortcut(x), 0.01)


class RawNet2(nn.Module):
    """
    End-to-end anti-spoofing model operating on raw waveforms.
    Input : [batch, 1, 64000]
    Output: [batch, 2]
    """

    def __init__(self):
        super().__init__()

        self.sinc = SincConv(n_filters=70, kernel_size=1024)
        self.bn0  = nn.BatchNorm1d(70)

        self.res_blocks = nn.Sequential(
            ResBlock(70,  70),
            ResBlock(70,  70),
            ResBlock(70,  128, stride=3),
            ResBlock(128, 128),
            ResBlock(128, 256, stride=3),
            ResBlock(256, 256),
        )

        self.gru = nn.GRU(input_size=256, hidden_size=1024, batch_first=True)

        self.fc = nn.Linear(1024, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, 64000]
        x = self.sinc(x)                            # [batch, 70, ~2000]
        x = F.leaky_relu(self.bn0(x), 0.01)
        x = self.res_blocks(x)                      # [batch, 256, ~222]
        x = x.permute(0, 2, 1)                      # [batch, time, 256] — GRU expects this
        _, h_n = self.gru(x)                        # h_n: [1, batch, 1024]
        x = h_n.squeeze(0)                          # [batch, 1024]
        return self.fc(x)                           # [batch, 2]
