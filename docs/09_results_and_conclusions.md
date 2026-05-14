# 09 — Results and Conclusions

## Table of Contents

1. [Complete Results Summary](#1-complete-results-summary)
2. [Per-Attack Detailed Analysis](#2-per-attack-detailed-analysis)
3. [Key Findings](#3-key-findings)
4. [Limitations](#4-limitations)
5. [Future Work](#5-future-work)
6. [Lessons Learned](#6-lessons-learned)

---

## 1. Complete Results Summary

### Overall EER

| Model | Parameters | Train time | Dev EER | Eval EER | vs. Baseline |
|---|---|---|---|---|---|
| LFCC-GMM Baseline | — | — | — | 8.0900% | — |
| CQCC-GMM Baseline | — | — | — | 9.5700% | — |
| **LCNN (ours)** | **699,938** | **~3 hours** | **0.0000%** | **7.0724%** | **-1.02 pp** |
| RawNet2 (ours) | 4,908,026 | ~12 hours | 2.4700% | 12.7814% | +4.69 pp |
| Ensemble (average) | — | — | — | 10.2392% | +2.15 pp |
| Ensemble (learned LR) | — | — | — | 9.9167% | +1.83 pp |

**pp = percentage points.**

### Ensemble Fusion Weights (from Logistic Regression)

```
P(spoof) = sigmoid(10.707 × lcnn_score + 3.065 × rawnet2_score + bias)
```

The logistic regression learned to weight LCNN scores 3.5× more than RawNet2 scores.

---

## 2. Per-Attack Detailed Analysis

### Full Per-Attack EER Table

| Attack | LCNN EER | RawNet2 EER | Ensemble avg | Attack category |
|---|---|---|---|---|
| A07 | 0.0000% | 1.2644% | ~0.6% | Neural vocoder (WaveNet-family) |
| A08 | 0.8158% | 6.9613% | ~3.5% | Waveform concatenation |
| A09 | 0.1224% | 1.0582% | ~0.5% | Neural source-filter |
| A10 | 0.5846% | 1.2237% | ~0.8% | Neural TTS + AR vocoder |
| A11 | 0.3663% | 1.1557% | ~0.7% | Transformer TTS |
| A12 | 0.7750% | 1.6859% | ~1.1% | WaveNet vocoder |
| A13 | 0.7937% | 2.6252% | ~1.6% | Waveform synthesis |
| A14 | 0.5088% | 1.6044% | ~1.0% | Statistical parametric |
| A15 | 1.5228% | 1.4652% | ~1.4% | Hybrid neural/statistical |
| A16 | 0.0000% | 2.1571% | ~1.0% | Waveform concatenation |
| **A17** | **36.8457%** | **40.3807%** | **~38%** | **Neural codec** |
| A18 | 9.7477% | 40.8221% | ~20% | Neural codec variant |
| A19 | 0.0611% | 12.6988% | ~6% | Traditional synthesis |

### Digest

LCNN achieves **< 1% EER on 9 of 13 eval attacks**, confirming that vocoder-based TTS synthesis is reliably detectable. The two outliers — A17 and A18 — break the pattern. Removing them from the aggregate would bring the LCNN's eval EER below 1%.

RawNet2 is systematically worse than LCNN across all attacks, with the single exception of A15 (1.47% vs 1.52% — essentially tied). The gap is smallest on vocoder-family attacks where temporal coherence (what RawNet2 can theoretically capture) might matter.

The ensemble's failure to beat LCNN across the board is the most informative result. It definitively shows that the two models' errors are correlated — they fail on the same attacks.

---

## 3. Key Findings

### Finding 1: LCNN Beats the LFCC-GMM Baseline with Fewer Assumptions

LFCC-GMM uses hand-crafted Linear Frequency Cepstral Coefficients (LFCC) features fed into a Gaussian Mixture Model. This is a shallow model with hand-designed features. Our LCNN, with 699,938 learned parameters operating on mel-spectrograms, outperforms it: **7.07% vs 8.09% EER**.

The improvement is not enormous (+1 percentage point), but it demonstrates that a learned neural approach can exceed the baseline without complex engineering. The LCNN's MFM activations and BatchNorm provide specific advantages for noisy discriminative learning that the GMM baseline cannot match.

### Finding 2: Spectrogram Models Excel at Vocoder Attack Detection

For A07–A16 and A19 (vocoder-based attacks), LCNN achieves 0.0–1.52% EER — performance that is essentially solved. Neural vocoders of the 2017–2019 era leave consistent high-frequency aliasing artifacts in the 4–8 kHz range that the LCNN detects with near-perfect reliability.

This finding is independently confirmed by Grad-CAM, which shows focused activation at 4–8 kHz for these attacks. The model's feature detector is aligned with the actual physical artifacts of vocoder synthesis.

### Finding 3: Neural Codec Attacks (A17–A18) Remain Unsolved

A17 achieves 36.8% EER — the worst result of any attack family, nearly at the random-classifier level (50% EER). A18 achieves 9.75% — much better, but still significantly worse than the vocoder attacks.

Both attacks use neural audio codec technology that generates perceptually indistinguishable speech without the high-frequency vocoder artifacts. The LCNN has no learned feature for detecting these attacks because no neural codec attacks appeared in the training set (A01–A06 are all vocoder-based).

This is the central failure of the project — and it is expected, given the dataset design. The eval set was constructed specifically to include unseen attack families to test generalization.

### Finding 4: Grad-CAM Confirms Model Focuses on High-Frequency Vocoder Artifacts

The Grad-CAM analysis (see `docs/07_explainability.md`) provides mechanistic confirmation of Findings 2 and 3:

| Input | Heatmap | Interpretation |
|---|---|---|
| A07 spoof (easy) | High freq 4–8 kHz, concentrated | Vocoder aliasing detected |
| A17 spoof (hard) | Diffuse mid-low freq, weak | No vocoder signal found |
| Bonafide | Very low freq, very weak | Absence of high-freq artifacts |

The model's decision rule is effectively: "high-frequency vocoder artifacts present → spoof; absent → bonafide." This is a valid rule for the training distribution but fails to generalize to codec-based synthesis.

### Finding 5: Ensemble Did Not Help Due to Correlated Failure Modes

Both LCNN and RawNet2 fail on A17/A18. Fusing their scores does not help because the errors are correlated — the ensemble inherits the failures of both models.

The learned ensemble (EER 9.92%) slightly outperforms the simple average (10.24%) because the logistic regression correctly upweights LCNN (10.707 vs 3.065 for RawNet2). But both are worse than LCNN alone (7.07%).

The lesson: architectural diversity (mel vs. raw waveform) is not sufficient for ensemble benefit. You need error diversity — models that fail on different samples.

### Finding 6: Raw Waveform Models Need Significantly More Training

RawNet2 reached only 24.37% dev EER after 50 epochs (the same budget as LCNN). After 100+ total epochs, it reached 2.47% dev EER but 12.78% eval EER.

LCNN converged in 28 epochs. The ~4× training time difference reflects the fundamental difference between:
- Using pre-computed, semantically aligned features (mel-spectrogram) that already encode useful information
- Learning features from scratch (SincConv filters) which must first discover which frequency bands are discriminative

For applications where training time is a constraint, LCNN-style models are clearly superior. For applications where maximum performance is needed regardless of training cost, RawNet2 with more epochs and better augmentation might close the gap.

---

## 4. Limitations

### 4.1 Training Distribution Limitation

The model is trained exclusively on A01–A06 attacks (vocoder-based TTS from 2015–2018). It has learned features specific to this era of synthesis technology. Any TTS system from 2020 onward (neural codecs, diffusion-based vocoders, language-model-based generation) will likely not be detected.

This is not a bug — it is a consequence of the dataset design. It correctly reveals that generalization to fundamentally new synthesis methods requires training data from those methods.

### 4.2 Single Language and Accent

VCTK corpus contains English speech from various British accents. The model has not been tested on non-English speech or accents significantly different from VCTK speakers. Synthesis artifacts may manifest differently for different phoneme systems, and the model's features may not transfer cross-lingually.

### 4.3 Near-Anechoic Recording Conditions

ASVspoof 2019 bonafide speech was recorded in a professional studio. Real-world bonafide speech contains room reverberation, background noise, microphone frequency response effects, and telephone channel artifacts. A model trained on studio recordings may incorrectly classify real speech recorded in noisy environments as spoof (due to unusual acoustic properties).

### 4.4 No Adversarial Robustness Testing

The model was not tested against adversarial attacks — modifications to synthetic speech designed to evade detection. A sophisticated attacker who knows the model's features (high-frequency vocoder artifacts) could post-process synthetic speech to suppress these artifacts, potentially reducing the model to chance-level detection.

### 4.5 Fixed 4-Second Window

The model only uses the first 4 seconds of audio. For utterances shorter than 4 seconds, silence padding may introduce artifacts. For utterances longer than 4 seconds, the model ignores content after the 4-second mark. An attack could in principle produce natural-sounding speech for the first 4 seconds and spoof speech afterward.

### 4.6 Mel-Spectrogram Discards Phase

By using a magnitude mel-spectrogram as input, the model cannot detect synthesis artifacts that exist only in the phase domain. Neural codecs in particular have been shown to have distinctive phase patterns that magnitude-only analysis cannot capture.

---

## 5. Future Work

### 5.1 Training on Neural Codec Attacks

**Priority: High. Effort: Medium.**

The most direct fix for A17/A18 failures. Obtain samples from ASVspoof 2021/2024 datasets, which include attacks from neural codec systems. Add these to the training mix. The model should learn codec-specific artifact features.

Expected result: A17 EER drops from 36.8% to something reasonable (< 5%). The high-frequency vocoder features are preserved for the existing attacks. Total eval EER should improve significantly.

### 5.2 Phase-Aware Features

**Priority: Medium. Effort: High.**

Compute instantaneous frequency (IF) features alongside the standard magnitude mel-spectrogram. Instantaneous frequency is the time derivative of the phase:

```
IF(t, f) = ∂φ(t,f) / ∂t
```

where `φ(t, f)` is the phase of the STFT. Natural speech has smooth IF in voiced regions (harmonic structure creates consistent phase evolution). Synthetic speech often has irregular IF due to vocoder processing.

Including IF as an additional input channel `[magnitude_mel, IF_mel]` gives the model access to phase information while keeping the 2D spectrogram structure that convolutions handle well.

### 5.3 AASIST Architecture

**Priority: High. Effort: High.**

**AASIST** (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks, Jung et al., 2022) is the current state-of-the-art countermeasure, achieving ~0.83% EER on ASVspoof 2019 eval.

AASIST uses a **graph neural network** on spectro-temporal features:
- Build a graph where each node represents a time-frequency region
- Edges connect time regions (temporal context) and frequency regions (spectral context)
- Graph attention mechanism learns which connections are most informative for spoof detection

This architecture can model non-local relationships — e.g., "the harmonic at 2 kHz at time 50ms is correlated with the harmonic at 4 kHz at time 50ms in bonafide speech but not in spoof speech." Standard CNNs can only capture local correlations within their receptive field.

### 5.4 Data Augmentation with Neural Codec Artifacts

**Priority: Medium. Effort: Low.**

Before training on real neural codec attacks (which require the ASVspoof 2021 dataset), we can synthetically generate codec-like artifacts by:

1. Encoding bonafide speech through a neural audio codec (EnCodec at low bitrate)
2. Decoding back to waveform
3. Treating the result as an additional "codec spoof" training example

This is a form of data augmentation that teaches the model what codec artifacts look like without needing a labeled dataset of codec attacks.

```python
from encodec import EncodecModel

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(1.5)  # low bitrate → more artifacts

# Encode real speech at low bitrate → decode → treat as spoof
encoded = model.encode(bonafide_waveform)
decoded = model.decode(encoded)
# decoded contains codec artifacts — label as spoof
```

### 5.5 Self-Supervised Pre-training Features

**Priority: Medium. Effort: Medium.**

Use features from self-supervised speech models (wav2vec 2.0, HuBERT, WavLM) as input to the classifier instead of or in addition to mel-spectrograms. These models were trained on thousands of hours of real speech and encode rich acoustic representations.

The intuition: features from a model trained on real speech will represent real speech naturally. Synthetic speech, forced through the same representation, may produce anomalous feature values that are detectable as out-of-distribution.

Recent work (Wang et al., 2021; Liu et al., 2023) shows that wav2vec 2.0 features combined with a simple linear classifier achieve near state-of-the-art on ASVspoof 2019 — suggesting the pre-trained features already capture much of the discriminative information.

### 5.6 Uncertainty Quantification

**Priority: Medium. Effort: Medium.**

The current model outputs a single confidence score. For a security application, we should also estimate **uncertainty** — the model's confidence in its own confidence.

Approaches:
- **Monte Carlo Dropout**: Run inference with Dropout enabled, multiple times. The variance across runs estimates epistemic uncertainty.
- **Temperature Scaling**: Post-training calibration that maps model confidence to actual empirical accuracy.
- **Deep Ensembles**: Train multiple models with different random seeds. Disagreement among models indicates uncertain predictions.

A production system should flag low-certainty predictions (near 50% score) for human review rather than making a hard decision.

### 5.7 Continual Learning

**Priority: Low (research). Effort: Very High.**

As new TTS systems are released, the model's performance will degrade on new attacks. A continual learning framework would:
1. Detect distribution shift (new attack patterns appearing in the input distribution)
2. Alert operators
3. Collect labeled samples of new attacks
4. Update the model without forgetting performance on old attacks (catastrophic forgetting mitigation via EWC, experience replay, etc.)

This is an active research area. For now, periodic retraining from scratch on updated datasets is more practical.

---

## 6. Lessons Learned

### 6.1 Feature Engineering Still Matters

The comparison between LCNN (mel-spectrogram input, 7.07% EER) and RawNet2 (raw waveform input, 12.78% EER) demonstrates that domain-appropriate feature engineering provides a substantial advantage, even in the deep learning era. The mel-spectrogram is not just a preprocessing step — it is decades of speech science knowledge encoded as an inductive bias.

End-to-end learning can in principle learn any feature. In practice, it requires far more data and compute to learn the same features that can be computed analytically. For constrained training budgets, leveraging domain knowledge pays off.

### 6.2 Evaluate on Held-Out Attack Families, Not Just Held-Out Speakers

A model that achieves 0% EER on the development set is not necessarily good. The development set in ASVspoof 2019 uses the same attack families as the training set (A01–A06). Zero dev EER means the model memorized the training attack fingerprints — not that it learned to detect synthesis in general.

The critical evaluation is on A07–A19 (held-out attack families). Only this evaluation reveals whether the model learned something fundamental about synthetic speech.

For any future dataset, this principle applies: always test on attack families the model has never seen.

### 6.3 Correlated Errors Make Ensembling Ineffective

Adding a second model to an ensemble is not automatically beneficial. Both models must fail on different samples for the ensemble to improve over the best single model. When the root cause of failures is shared (in our case: both models were trained on vocoder-attack data and fail on codec-attack data), ensembling only adds complexity without improving performance.

Before investing in ensemble infrastructure, analyze the error correlation between candidate models on the validation set. If EER improvements per attack are correlated, the ensemble will not help.

### 6.4 Grad-CAM Earns Its Place

The Grad-CAM analysis was not just a visualization exercise — it produced concrete, actionable insights:

1. It confirmed that the model uses high-frequency vocoder artifacts (providing confidence that the model is doing the right thing for the easy attacks)
2. It revealed that the model has no signal for A17 (providing a specific explanation for the failure)
3. It suggested what to do next (train on codec attacks, add phase features)

Explainability tools are often treated as optional "nice to have" additions. In this project, they were essential for understanding both successes and failures.

### 6.5 Raw Waveform Models Require Patient Training

The RawNet2 trajectory — 24.37% dev EER at 50 epochs, 2.47% dev EER at 100+ epochs — shows that end-to-end models from raw waveforms need substantially more training budget. A researcher who stopped at 50 epochs would conclude "RawNet2 doesn't work" — but 64 more epochs revealed it can reach competitive performance.

The lesson: when training raw waveform models, set generous epoch budgets and use early stopping with high patience. The optimization landscape for SincConv + ResBlocks + GRU is more complex than for pre-feature-extracted + CNN models, and gradient descent takes longer to navigate it.

### 6.6 soundfile > torchaudio for Reliable Cross-Platform Audio Loading

The torchcodec/torchaudio backend issue on Python 3.14 + Apple Silicon was a non-trivial obstacle. The solution — using soundfile as the primary audio reader — is more robust and has fewer dependencies. For any speech processing project, prefer soundfile or librosa for audio I/O over framework-specific loaders that may have backend compatibility issues.

### 6.7 The Arms Race Framing Is Correct

This project's results align exactly with the "detection arms race" framing from `docs/01_problem_and_motivation.md`. The LCNN learned features specific to 2017–2019 era TTS systems. Neural codec attacks (introduced in ~2019–2021) immediately broke it. A newer model would need to be retrained on codec attacks, and will likely be broken by the next generation of synthesis technology.

This is not a failure — it is the expected behavior of a model trained on a finite sample from an infinite and evolving distribution. Robust countermeasure deployment requires:
- Continuous monitoring for performance degradation
- Regular retraining on newly discovered attack families
- Defense in depth (multiple detection approaches)
- Human review for borderline cases

Audio deepfake detection is an ongoing process, not a solved problem.
