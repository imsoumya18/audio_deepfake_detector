# Project 01 — Audio Deepfake Detector

Detect whether an audio clip is real human speech or AI-generated, trained from scratch on a standard academic benchmark.

---

## Why This Project

- Active, unsolved research problem — no shortcut API exists.
- Audio ML is rare in portfolios — immediately differentiates you.
- Relevant to Safety & Trust teams at Meta, Google, Microsoft.
- Results comparable to published papers (ASVspoof benchmark).

---

## Dataset

**ASVspoof 2019 — Logical Access (LA) partition**

- ~121k utterances: bonafide (real) + spoofed (19 TTS/VC attack types).
- Metric: **Equal Error Rate (EER)**. Baseline LFCC-GMM: ~8.09%.
- Source: [datasharing.ed.ac.uk/handle/10283/3336](https://datasharing.ed.ac.uk/handle/10283/3336)

---

## Architecture

### Model A — LCNN on Mel-Spectrogram
```
Audio → Mel-spectrogram (128 bands) → Light CNN with Max-Feature-Map → FC → Softmax
```

### Model B — RawNet2 on Raw Waveform
```
Raw audio (16kHz) → SincConv → ResBlocks → GRU → FC
```
End-to-end, no hand-crafted features.

> Implement both, compare EER scores, ensemble outputs.

---

## Training

| Setting | Value |
|---|---|
| Loss | Weighted Cross-Entropy (handle class imbalance) |
| Optimiser | Adam, lr=1e-4 |
| Scheduler | CosineAnnealingLR |
| Augmentation (Model A) | SpecAugment |
| Augmentation (Model B) | Additive noise + RIR |
| Tracking | TensorBoard |
| Stopping criterion | Early stopping on Dev EER |

---

## FAANG Signal Additions

- **Grad-CAM on spectrograms** — visualise which frequency bands trigger detection.
- **Manual error analysis** on misclassified clips — which attack types fool the model?
- **FastAPI endpoint + Gradio demo** deployed on Hugging Face Spaces.
- **README** with EER table vs baseline papers and failure analysis.

---

## Stack

`Python` · `PyTorch` · `Librosa` · `FastAPI` · `Gradio` · `TensorBoard` · `Docker`

---

## Interview Story

> "I built an audio deepfake detector on ASVspoof 2019. I implemented two architectures — LCNN on mel-spectrograms and RawNet2 on raw waveforms — compared their EER scores, and used Grad-CAM to visualise which acoustic features each model used. The key finding was that vocoder artifacts appear in high-frequency bands, which the spectrogram model catches well, but neural codec attacks fool it — which is why raw waveform models are gaining traction."
