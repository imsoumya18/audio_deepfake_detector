import torch
import torchaudio.transforms as T


# Fixed parameters — changing these changes the input shape to the model
N_FFT      = 1024   # FFT window size (= 64ms at 16kHz)
HOP_LENGTH = 256    # step between windows (= 16ms at 16kHz)
N_MELS     = 128    # number of mel filterbank bands
F_MIN      = 0
F_MAX      = 8000
SAMPLE_RATE = 16_000


class MelSpectrogramTransform:
    """Converts a raw waveform [1, 64000] to a log mel-spectrogram [1, 128, 313]."""

    def __init__(self, augment: bool = False):
        self.augment = augment

        self.mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
        )

        # SpecAugment — only applied during training
        self.time_mask = T.TimeMasking(time_mask_param=30)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [1, 64000]
        mel = self.mel(waveform)            # [1, 128, time_frames]
        mel = torch.log(mel + 1e-8)         # log compression

        if self.augment:
            mel = self.time_mask(mel)
            mel = self.freq_mask(mel)

        return mel  # [1, 128, 313]
