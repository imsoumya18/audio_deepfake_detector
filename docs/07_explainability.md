# 07 — Explainability: Grad-CAM

## Table of Contents

1. [Why Interpretability Matters](#1-why-interpretability-matters)
2. [Grad-CAM Theory](#2-grad-cam-theory)
3. [PyTorch Hook API](#3-pytorch-hook-api)
4. [Target Layer Selection](#4-target-layer-selection)
5. [Per-Case Findings](#5-per-case-findings)
6. [What the Model Learned](#6-what-the-model-learned)
7. [Implications for Future Work](#7-implications-for-future-work)

---

## 1. Why Interpretability Matters

### The Black Box Problem

LCNN achieves 7.07% EER on the eval set. But why? What acoustic features does it use to distinguish real from fake speech? Without understanding the model's reasoning, several problems arise:

1. **We cannot trust it**: A model that achieves good EER by exploiting a dataset artifact (e.g., different recording conditions between bonafide and spoof) is not doing genuine deepfake detection.

2. **We cannot debug failures**: Why does LCNN fail catastrophically on A17? Without understanding what the model looks for, we cannot identify what is missing from its feature set.

3. **We cannot improve it**: If we know the model detects high-frequency vocoder artifacts, we know what to add: either training data with different artifact types, or additional features that capture other artifact dimensions.

4. **We cannot deploy it confidently**: Regulatory and ethical requirements increasingly demand that automated systems provide explanations for their decisions, especially in security-critical applications.

Grad-CAM provides a visual explanation: a heatmap overlaid on the mel-spectrogram that shows which time-frequency regions drove the model's decision.

---

## 2. Grad-CAM Theory

**Grad-CAM** (Gradient-weighted Class Activation Mapping, Selvaraju et al., 2017) is a technique for producing visual explanations of convolutional network predictions. It works by measuring how strongly the final convolutional feature maps are correlated with the predicted class score.

### Step-by-Step Derivation

#### Step 1: Forward Pass

Given an input mel-spectrogram `I` of shape `[1, 1, 128, 251]`, the forward pass produces:
- Feature maps `A^k` at the target layer, shape `[1, C, H, W]` where `k ∈ {1...C}` indexes channels
- Class logit `y^c` for target class `c` (c=1 for spoof)

#### Step 2: Backward Pass to Get Gradients

Compute the gradient of the class score with respect to each feature map channel:

```
∂y^c / ∂A^k_{ij}   for each spatial position (i, j) and channel k
```

This gives a gradient tensor of shape `[1, C, H, W]` — same as the feature maps.

#### Step 3: Global Average Pool the Gradients

Compute the **importance weight** for each channel by averaging the gradients over all spatial positions:

```
α^c_k = (1 / Z) × Σ_i Σ_j (∂y^c / ∂A^k_{ij})
```

where `Z = H × W` is the number of spatial positions.

This is global average pooling of the gradient tensor. The result is `α^c_k` — a scalar weight for channel `k` representing how important that channel's activations are for predicting class `c`.

**Intuition**: If increasing the activation of channel `k` at any spatial position increases the class score (positive gradient), then channel `k` is important for predicting class `c`. The spatial average computes how consistently this is true across all positions.

#### Step 4: Weighted Sum of Feature Maps

Combine the feature maps using the importance weights:

```
L^c_{Grad-CAM} = ReLU(Σ_k α^c_k × A^k)
```

The result is `L^c_{Grad-CAM}` of shape `[H, W]` — a spatial heatmap.

**Why ReLU?** We only care about features that have a **positive influence** on the predicted class. Channels where `α^c_k × A^k < 0` are features that, if active, would *decrease* the class score — i.e., they are evidence *against* class `c`. ReLU discards these negative contributions.

#### Step 5: Upsample

The heatmap `L^c` has the spatial dimensions of the target convolutional layer, which is much smaller than the input (e.g., `[8, 15]` for our target layer). Bilinear interpolation upsamples it to the input size `[128, 251]`:

```python
cam = F.interpolate(cam, size=mel.shape[-2:], mode="bilinear", align_corners=False)
```

#### Step 6: Normalize

```python
cam -= cam.min()
if cam.max() > 0:
    cam /= cam.max()
```

The final heatmap has values in `[0, 1]`. Value 1 = the region most responsible for the prediction. Value 0 = region that had no influence.

### Complete Math Summary

```
Forward:
  A^k = feature_map[k]          # shape [H, W] for each channel k
  y^c = model_output[c]          # scalar class score

Backward:
  g^k_{ij} = ∂y^c / ∂A^k_{ij}  # gradient at position (i,j), channel k

Weights:
  α^c_k = (1/HW) × Σ_{i,j} g^k_{ij}   # global avg pooled gradient

CAM:
  L^c_{ij} = ReLU(Σ_k α^c_k × A^k_{ij})

Upsample + normalize:
  heatmap = bilinear_upsample(L^c, target_size=[128, 251])
  heatmap = (heatmap - heatmap.min()) / heatmap.max()
```

---

## 3. PyTorch Hook API

Grad-CAM requires access to both the **forward activations** (feature maps) and the **backward gradients** at a specific intermediate layer. PyTorch's hook API provides this.

### Why Hooks?

Without hooks, we would need to modify the `forward()` method of the model to return intermediate activations. This is invasive and changes the model's interface. Hooks allow attaching callbacks that fire automatically during forward and backward passes — non-invasively.

### forward_hook

```python
def _save_activation(self, module, input, output):
    self._activations["feat"] = output.detach()

target_layer.register_forward_hook(self._save_activation)
```

`register_forward_hook(hook_fn)` registers a function that is called every time the module's `forward()` runs. The arguments are:
- `module`: the layer itself
- `input`: tuple of inputs to the layer
- `output`: the layer's output tensor

We save `output.detach()` — the feature maps produced by the target layer during the forward pass. `.detach()` disconnects the saved tensor from the computation graph to prevent memory leaks.

### register_full_backward_hook

```python
def _save_gradient(self, module, grad_input, grad_output):
    self._gradients["feat"] = grad_output[0].detach()

target_layer.register_full_backward_hook(self._save_gradient)
```

`register_full_backward_hook(hook_fn)` fires during the backward pass when gradients flow through the layer. The arguments are:
- `module`: the layer
- `grad_input`: tuple of gradients w.r.t. the layer's inputs
- `grad_output`: tuple of gradients w.r.t. the layer's outputs

We save `grad_output[0]` — the gradient of the loss with respect to the output of the target layer. This is the `∂y^c / ∂A^k_{ij}` from the derivation above, before global averaging.

**Why `full_backward_hook` not `backward_hook`?** The deprecated `register_backward_hook` had a known issue where the `grad_input` tensor could be incorrect for modules with multiple inputs. `register_full_backward_hook` is the updated API that correctly handles all cases. For our target layer (a ConvBlock), this distinction matters because the ConvBlock has a sequential structure with multiple intermediate tensors.

### Complete Grad-CAM Forward Pass

```python
def compute(self, mel, target_class=1):
    mel = mel.to(self.device).requires_grad_(True)

    self.model.zero_grad()     # clear any existing gradients
    logits = self.model(mel)   # forward pass — fires _save_activation hook
    score  = logits[0, target_class]   # scalar: spoof class score
    score.backward()           # backward pass — fires _save_gradient hook

    grads   = self._gradients["feat"]   # [1, C, H, W]
    acts    = self._activations["feat"] # [1, C, H, W]
    weights = grads.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # [1, 1, H, W]
    cam = F.interpolate(cam, size=mel.shape[-2:], mode="bilinear", align_corners=False)
    ...
```

**`requires_grad_(True)` on the input**: Normally, input tensors don't require gradients (we only backpropagate through model parameters). Setting `requires_grad=True` on `mel` allows computing gradients all the way back to the input, but we don't use the input gradients in Grad-CAM. The reason we set it is that without it, some versions of PyTorch may not retain the intermediate gradients at the hook layer during backward.

---

## 4. Target Layer Selection

The Grad-CAM target is `model.features[9]` — the last ConvBlock in the LCNN feature extractor:

```python
# From src/explainability/gradcam.py
target_layer = model.features[9]   # ConvBlock(64→32, 3×3, pad=1)
```

This is the 10th element (0-indexed) of the `nn.Sequential` that comprises `model.features`:

```
model.features[0]  = ConvBlock(1→32,  5×5)
model.features[1]  = MaxPool2d
model.features[2]  = ConvBlock(32→32, 1×1)
model.features[3]  = ConvBlock(32→32, 3×3)
model.features[4]  = MaxPool2d
model.features[5]  = ConvBlock(32→32, 1×1)
model.features[6]  = ConvBlock(32→32, 3×3)
model.features[7]  = MaxPool2d
model.features[8]  = ConvBlock(32→64, 1×1)
model.features[9]  = ConvBlock(64→32, 3×3)  ← TARGET
model.features[10] = MaxPool2d
```

### Why the Last Conv Layer?

Grad-CAM works best at the **last convolutional layer** because:

1. **Richest semantic features**: Earlier layers capture low-level patterns (edges, textures). The last conv layer captures the highest-level semantic features — it "knows" whether the spectrogram is a spoof or bonafide in the most abstract sense.

2. **Most spatial specificity**: Later layers have seen more of the receptive field. The spatial position in `features[9]`'s output corresponds to a region in the input that is `16 × 16` pixels (due to 4 MaxPool2d(2,2) = 16x downsampling). This is a large-ish region, meaning the Grad-CAM heatmap is coarse. If we used an earlier layer, we'd get finer spatial resolution but lower semantic relevance.

3. **Best resolution trade-off**: `features[9]` output is `[B, 32, 16, 31]`. After bilinear upsampling to `[128, 251]`, we get adequate spatial resolution to see which frequency bands the model focuses on.

---

## 5. Per-Case Findings

The Grad-CAM analysis was run on representative samples from the evaluation set:
- Several bonafide utterances
- Easy spoof samples from A07 (0% EER attack)
- Hard spoof samples from A17 (36.8% EER attack)
- Moderate samples from A18 (9.7% EER attack)

The script `scripts/run_gradcam.py` generates heatmap visualizations saved to `notebooks/`.

### Case 1: Bonafide Speech

**Model prediction**: bonafide (high confidence)

**Grad-CAM (target_class=1, spoof)**:
The heatmap shows very **low activation magnitude overall**, with a small concentration at **very low frequency bands (0–1 kHz, the bottom rows of the mel-spectrogram)**.

**Interpretation**:

The model is not confidently detecting spoof, so the "spoof-driving" heatmap is appropriately flat. The slight low-frequency activation suggests the model is checking whether there are any artifacts even in the low frequencies — and finding none.

More interestingly, running Grad-CAM with `target_class=0` (bonafide) on bonafide speech shows stronger activation throughout the spectrogram — the model has learned what genuine speech looks like at low frequencies.

**What this tells us**: The LCNN's bonafide detection is essentially "no high-frequency artifacts detected → bonafide." It is not detecting positive features of human speech — it is detecting the absence of synthetic artifacts.

### Case 2: A07 Spoof (0% EER — Easy)

**Model prediction**: spoof (very high confidence, e.g., 0.997)

**Grad-CAM (target_class=1, spoof)**:
Strong, concentrated activation at **high frequency bands (4–8 kHz, the top portion of the mel-spectrogram)**. The activation is present across all time frames — it is not a transient feature but a sustained spectral signature.

**Interpretation**:

Neural vocoders of the 2017–2019 era (WaveNet family, which A07 likely belongs to) produce characteristic high-frequency artifacts. These arise because:

1. **Vocoder prediction at each frame**: The vocoder generates spectral envelope parameters frame-by-frame. The transitions between frames can produce aliasing artifacts at high frequencies.

2. **Missing acoustic coupling**: In natural speech, the vocal tract resonances create specific harmonic relationships that are correlated across frequency bands. Vocoders synthesize each spectral band somewhat independently, producing unnatural relationships at high frequencies.

3. **Bandwidth limiting**: Some vocoders explicitly limit synthesis to certain frequency bands and use simple extrapolation for others, producing telltale smooth or flat high-frequency content.

The mel-spectrogram makes these artifacts visible as **abnormally smooth, patterned energy** in the 4–8 kHz range. The LCNN learned to detect this pattern with perfect reliability (0% EER).

**Time-frequency heat concentration**: The heatmap shows that the activation is fairly uniform across time, meaning the artifact is present throughout the entire utterance — not just at specific phoneme boundaries. This is consistent with a vocoder that generates each frame independently, producing the same artifact signature regardless of the phoneme being synthesized.

### Case 3: A17 Spoof (36.8% EER — Very Hard)

**Model prediction**: often bonafide (model is confused; scores near 0.5)

**Grad-CAM (target_class=1, spoof)**:
The heatmap shows **diffuse, low-magnitude activation spread across mid-low frequencies (1–4 kHz)**. There is no concentrated "hot spot." The activations are weak and spatially scattered.

**Interpretation**:

A17 is a neural codec-based attack. Neural codecs (e.g., EnCodec, SoundStream) encode speech into discrete tokens and decode back to waveforms. The key differences from vocoders:

1. **End-to-end perceptual optimization**: Neural codecs are trained with perceptual losses (e.g., adversarial training with a discriminator) that specifically minimize audible differences. They do not leave the same high-frequency aliasing that vocoders produce.

2. **Distributed compression artifacts**: Any codec artifacts in neural codecs appear as subtle texture changes distributed across the entire spectrum, not concentrated at high frequencies.

3. **No frame-boundary artifacts**: Neural codecs operate on tokens that span many frames, avoiding the frame-boundary artifacts that per-frame vocoders produce.

The LCNN has **no learned feature** that activates for A17's artifact pattern. Its internal representations were trained exclusively on A01–A06 (vocoder-based attacks) and learned features that are useless for codec-based attacks. The "diffuse mid-low frequency" activation is essentially the model grasping at whatever correlates marginally with the slightly different statistics of codec speech — not a reliable artifact signal.

**Why mid-low frequencies?** This is likely because mid-low frequencies (formant region, 1–3 kHz) show the most statistical difference between the A17 spoof distribution and the bonafide distribution due to the codec's specific compression characteristics in that range. But this difference is subtle and inconsistent — explaining the near-random 36.8% EER.

### Case 4: A18 Spoof (9.7% EER — Hard but Partially Detectable)

**Model prediction**: often spoof (moderate confidence, ~0.7)

**Grad-CAM**:
Similar to A17, with slightly more concentrated activation in the **low-mid frequency range (500 Hz – 2 kHz)**, and slightly higher magnitude than A17.

**Interpretation**:

A18 is described as a neural codec variant. It may use a different codec architecture or different training settings than A17. The slightly better detection performance (9.7% vs 36.8%) suggests that A18's artifacts are more similar to the vocoder artifacts in the training set — possibly because:
- A18 uses a lower-quality codec setting with more quantization noise in the lower frequencies
- A18 was trained with a weaker adversarial loss, leaving more detectable artifacts
- A18's pitch generation method leaves artifacts that overlap with known vocoder patterns

The LCNN partially exploits these similarities, achieving 9.7% EER — still poor by the standard of the easy attacks (0–1.5%), but demonstrating that partial transfer from known features is possible.

---

## 6. What the Model Learned

Based on the Grad-CAM analysis, a complete picture of the LCNN's decision process emerges:

### The LCNN's Implicit Decision Rule

```
If high-frequency energy (4–8 kHz) shows characteristic vocoder aliasing pattern:
    → Predict SPOOF (high confidence)

Else if high-frequency energy looks smooth and natural:
    → Predict BONAFIDE (high confidence)

Else if the signal has unusual mid-low frequency statistics:
    → Maybe SPOOF (low confidence) — neural codec territory, model confused
```

### Feature Visualization Summary

| Input | Heatmap location | Magnitude | Model confidence |
|---|---|---|---|
| Bonafide | Very low freq (0–1 kHz) | Very low | High bonafide |
| A07 (vocoder, easy) | High freq (4–8 kHz) | Very high | High spoof |
| A09 (vocoder, easy) | High freq (4–8 kHz) | High | High spoof |
| A17 (neural codec, hard) | Mid-low freq (1–4 kHz) | Very low | Low / confused |
| A18 (neural codec, hard) | Low-mid freq (0.5–2 kHz) | Low | Moderate spoof |

### What Features LCNN Did NOT Learn

Noticeably absent from the Grad-CAM heatmaps:

1. **Phase coherence**: Synthetic speech has unnatural phase relationships between harmonics. The mel-spectrogram discards phase, so the LCNN cannot see this.

2. **Temporal envelope modulation**: Natural speech has specific micro-variations in energy over time (jitter, shimmer, tremor) that arise from physiological processes. Vocoders and codecs may have different modulation statistics. LCNN's global feature aggregation may not capture these temporal patterns.

3. **Formant transition dynamics**: Natural formant movements follow specific trajectories constrained by vocal tract physics. Synthetic systems approximate these transitions differently. These patterns live in the 0–4 kHz range but require precise temporal tracking — the LCNN's pooling aggregates away fine temporal structure.

---

## 7. Implications for Future Work

### Short-Term Improvements

**Add phase-aware features**: Compute instantaneous frequency (derivative of phase with respect to time) as an additional input channel. Synthetic speech has characteristic flat instantaneous frequency in regions where vocoders hold spectral parameters constant.

**Extend to A17/A18 training data**: The most direct fix. If A17/A18-type attacks are included in training, the model will learn their artifact signatures. The ASVspoof 2021 LA dataset includes attacks with neural codec elements.

**Use higher mel resolution at high frequencies**: Standard mel scaling compresses high frequencies. Using inverse mel (more resolution at high frequencies) might help the model detect high-frequency artifacts more precisely.

### Architectural Improvements

**AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention)**: The current state-of-the-art uses a graph attention network on spectro-temporal features. Instead of processing a 2D mel-spectrogram with convolutions, AASIST builds a graph where each node represents a time-frequency region and edges represent spectro-temporal relationships. Graph attention can model non-local spectral correlations that convolutions miss.

**Wav2Vec 2.0 features**: Using a self-supervised representation trained on large speech corpora as input features. These representations encode phonetic and prosodic information at a higher level than mel-spectrograms. Artifacts in synthesis that create unnatural phonetic sequences might be detectable from these features.

**Multi-scale analysis**: Process the spectrogram at multiple time-frequency resolutions simultaneously and fuse the representations. This captures both fine temporal structure (short-window STFT) and global spectral shape (long-window STFT).

### Debugging Insights

The Grad-CAM analysis of A17 provided the most valuable debugging insight of this entire project:

**The model is not detecting deepfakes in general. It is detecting vocoders specifically.**

This distinction is crucial for deployment. A production system based on this LCNN would reliably detect 2017–2019 era TTS systems but fail on modern neural codec-based speech synthesis. Any deployment would need:

1. Continuous monitoring for new attack types (e.g., flag samples where the model has low confidence — near 0.5 — for human review)
2. A defense-in-depth strategy combining this vocoder detector with complementary detectors
3. Regular retraining as new synthesis systems are discovered in the wild

The Grad-CAM analysis transforms a black-box failure into an understandable, actionable insight: the model needs to see neural codec attacks during training.
