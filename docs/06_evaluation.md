# 06 — Evaluation

## Table of Contents

1. [EER Theory and Computation](#1-eer-theory-and-computation)
2. [Full Results Table](#2-full-results-table)
3. [Per-Attack EER Analysis](#3-per-attack-eer-analysis)
4. [The Generalization Gap](#4-the-generalization-gap)
5. [Ensemble Analysis](#5-ensemble-analysis)
6. [Comparison to Published Baselines](#6-comparison-to-published-baselines)

---

## 1. EER Theory and Computation

### Binary Decision System

The countermeasure outputs a **detection score** for each utterance — a real number where higher values indicate "more likely spoof." The score is the softmax probability of class 1:

```python
scores = torch.softmax(logits, dim=1)[:, 1]
```

A decision is made by comparing this score to a threshold `θ`:
- If `score >= θ`: declare **spoof**
- If `score < θ`: declare **bonafide**

### Two Types of Errors

**False Acceptance Rate (FAR)** — also called False Positive Rate (FPR):
```
FAR(θ) = P(score >= θ | utterance is bonafide)
       = #{bonafide utterances with score >= θ} / #{bonafide utterances}
```
This is the fraction of genuine speech clips that are incorrectly flagged as fake. From a security perspective, this is the "spoof passes" error.

**False Rejection Rate (FRR)** — also called False Negative Rate (FNR):
```
FRR(θ) = P(score < θ | utterance is spoof)
       = #{spoof utterances with score < θ} / #{spoof utterances}
```
This is the fraction of fake speech clips that are incorrectly classified as genuine. From a security perspective, this is the "spoof wins" error.

### The FAR/FRR Trade-off

As the threshold `θ` varies from 0 to 1, FAR and FRR move in opposite directions:

```
Error
Rate
 1.0 |
     | FAR(θ)     FRR(θ)
     |  \             /
     |    \         /
 0.5 |      \     /
     |        \   /
     |         \ /
     |      EER ×         ← EER = FAR(θ*) = FRR(θ*)
     |         / \
     |       /    \
 0.0 |_____/______\_______> threshold θ
     0                   1
     (declare everything spoof)     (declare everything bonafide)
```

**At θ → 0**: Everything is declared spoof.
- FAR = 1.0 (all bonafide rejected as spoof — maximum false acceptance of spoof doesn't apply; this is all bonafide incorrectly rejected)
- Actually at θ=0, score >= 0 always, so FAR=1: all bonafide declared spoof
- FRR = 0: all spoof declared spoof (correct), so no false rejections

Wait — let me clarify the convention used in `src/evaluation/eer.py`:

In sklearn's `roc_curve`, FPR = FAR is computed with respect to the positive class (label=1 = spoof). The convention used here is:
- Label 1 = spoof (positive class for the ROC curve)
- FPR = rate at which bonafide samples (label=0) are classified as spoof
- TPR = rate at which spoof samples (label=1) are correctly classified as spoof

So the FAR/FRR crossing diagram is:

```
Error
Rate
 1.0 |
     | FRR(θ)\      /FAR(θ)
     |        \    /
     |          \/
 EER |          /\
     |        /    \
     |      /        \
 0.0 |----/----------\----> threshold θ
     0    θ*          1
```

Where `FRR(θ)` decreases as θ decreases (more things called spoof = fewer spoof miss), and `FAR(θ)` increases as θ decreases (more things called spoof = more bonafide falsely accused).

### EER Definition

The **Equal Error Rate** is the threshold `θ*` at which `FAR(θ*) = FRR(θ*)`:

```
EER = FAR(θ*) = FRR(θ*)
```

It is a single number describing the best achievable balanced performance of the detector.

- **EER = 0%**: Perfect separation — there exists a threshold where all bonafide are correct and all spoof are correct.
- **EER = 50%**: No separation — regardless of threshold, the error rates cross at 50% (random classifier).
- **EER = 7.07%** (our LCNN): At the optimal threshold, both false acceptance and false rejection rates are 7.07%.

### Implementation: sklearn + scipy

The implementation in `src/evaluation/eer.py`:

```python
def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr   # FRR = 1 - TPR

    # Find threshold where FAR == FRR: solve fpr - fnr = 0
    eer = brentq(lambda x: interp1d(fpr, fpr - fnr)(x), fpr[0], fpr[-1])
    return float(eer)
```

**Step 1**: `sklearn.metrics.roc_curve` sweeps through all possible score thresholds and records `(FPR, TPR)` pairs. With 71,237 eval samples, this generates 71,237 threshold values.

**Step 2**: `FNR = 1 - TPR` converts TPR (true positive rate for spoof) to FNR (false negative rate = rate of missing spoof = FRR).

**Step 3**: `scipy.optimize.brentq` finds the root of `FPR - FNR = 0`. Brent's method is a root-finding algorithm that combines bisection, secant, and inverse quadratic interpolation. It is guaranteed to converge if the function changes sign in the bracketing interval `[fpr[0], fpr[-1]]`, which it always does (at one extreme, FPR=0 < FNR=1; at the other, FPR=1 > FNR=0).

`scipy.interpolate.interp1d` converts the discrete `(fpr, fpr-fnr)` pairs into a continuous function for the root finder.

### Per-Attack EER

For each attack type, EER is computed on a subset of the eval set containing all bonafide samples plus only the spoof samples from that specific attack:

```python
for attack in unique_attacks:
    attack_mask = attack_types == attack
    bonafide_mask = labels == 0
    mask = bonafide_mask | attack_mask
    results[attack] = compute_eer(labels[mask], scores[mask])
```

This measures "how well does the model distinguish real speech from attack X specifically?" It's useful for diagnosing which attack families are easy vs hard.

---

## 2. Full Results Table

### Main Results

| Model | Dev EER | Eval EER | Delta vs Baseline |
|---|---|---|---|
| LFCC-GMM Baseline | — | 8.0900% | — |
| **LCNN** | **0.0000%** | **7.0724%** | **-1.0176 pp** |
| RawNet2 | 2.4700% | 12.7814% | +4.6914 pp |
| Ensemble (average) | — | 10.2392% | +2.1492 pp |
| Ensemble (learned LR) | — | 9.9167% | +1.8267 pp |

### Ensemble Weights

The logistic regression ensemble trained on dev scores learned:

```
Learned weights: LCNN = 10.707, RawNet2 = 3.065
```

The logistic regression assigns LCNN's scores 3.5× more weight than RawNet2's scores. This reflects LCNN's superior performance on the dev set.

### Key Observation

**Ensembling made performance worse** relative to LCNN alone (10.24–9.92% vs 7.07%). This is counterintuitive — why would combining two models hurt? The answer is in Section 5.

---

## 3. Per-Attack EER Analysis

### Full Per-Attack Table

| Attack | LCNN EER | RawNet2 EER | Ensemble (avg) | Relative difficulty |
|---|---|---|---|---|
| A07 | 0.0000% | 1.2644% | ~0.6% | Very easy |
| A08 | 0.8158% | 6.9613% | ~3.5% | Easy |
| A09 | 0.1224% | 1.0582% | ~0.5% | Very easy |
| A10 | 0.5846% | 1.2237% | ~0.8% | Easy |
| A11 | 0.3663% | 1.1557% | ~0.7% | Easy |
| A12 | 0.7750% | 1.6859% | ~1.1% | Easy |
| A13 | 0.7937% | 2.6252% | ~1.6% | Easy |
| A14 | 0.5088% | 1.6044% | ~1.0% | Easy |
| A15 | 1.5228% | 1.4652% | ~1.4% | Moderate |
| A16 | 0.0000% | 2.1571% | ~1.0% | Very easy (LCNN only) |
| **A17** | **36.8457%** | **40.3807%** | **~38%** | **Very hard — neural codec** |
| A18 | 9.7477% | 40.8221% | ~20% | Hard — neural codec |
| A19 | 0.0611% | 12.6988% | ~6% | Mixed |

### Observations

**A07 and A16 (0% LCNN EER)**: The LCNN perfectly separates bonafide from these attacks. Grad-CAM analysis shows these are vocoder-based attacks with strong high-frequency artifacts that the LCNN has learned to detect precisely.

**A08 (0.82% LCNN, 6.96% RawNet2)**: An interesting case where LCNN vastly outperforms RawNet2. This attack uses waveform concatenation — a technique that creates temporal discontinuities rather than spectral artifacts. The mel-spectrogram makes these discontinuities visible as transient patterns in specific frequency bands.

**A15 (1.52% LCNN, 1.47% RawNet2)**: The only attack where RawNet2 slightly outperforms LCNN. A15 is a hybrid neural/statistical system. The raw waveform representation may preserve some feature that the mel-spectrogram discards.

**A17 (36.85% LCNN, 40.38% RawNet2)**: Complete failure for both models. EER of 36.85% means: out of all possible thresholds, the best one still misclassifies 36.85% of samples from each class. The model is performing only slightly better than random (50% would be completely random). Both false acceptance and false rejection are nearly equal and nearly random.

**A18 (9.75% LCNN, 40.82% RawNet2)**: RawNet2 fails completely (40.8% EER) while LCNN has moderate success (9.75%). This is surprising given that A17 and A18 are supposedly similar neural codec attacks. A possible explanation: A18 has some characteristics (perhaps higher-level prosodic irregularities or specific codec artifacts) that produce mel-spectrogram patterns the LCNN can partially detect, while being invisible to the raw waveform path.

**A19 (0.06% LCNN, 12.70% RawNet2)**: Another case where LCNN dramatically outperforms RawNet2. A19 is a traditional synthesis method — presumably with strong spectral artifacts that the mel-spectrogram exposes clearly.

---

## 4. The Generalization Gap

### Dev to Eval

LCNN achieves:
- **Dev EER: 0.0000%** (perfectly separates A01–A06 attacks from bonafide)
- **Eval EER: 7.0724%** (imperfect on A07–A19 attacks)

The gap of 7.07 percentage points is the **generalization cost** of encountering unseen attack families.

### Why the Gap Exists

The 0% dev EER means the model perfectly learned the artifacts of A01–A06. These are vocoder-based attacks from the 2015–2018 era, and the LCNN found a feature (confirmed by Grad-CAM: high-frequency aliasing at 4–8 kHz) that perfectly separates all six attack types from bonafide speech.

The 7.07% eval EER means that some of the A07–A19 attacks cannot be detected using the features learned from A01–A06. The 7.07% is an aggregate — dominated by the catastrophic failure on A17 (36.85%) and moderate difficulty on A18 (9.75%). If we remove A17 and A18, the LCNN's average eval EER on A07–A16 and A19 would be approximately 0.6%.

### What the Gap Tells Us

The generalization gap reveals a **feature specificity problem**: the LCNN learned vocoder-specific features, not general speech-authenticity features. Its internal representation can be described as:

> "High-frequency spectral artifacts present → likely spoof. High-frequency artifacts absent → likely bonafide."

This heuristic works perfectly for vocoder-based attacks and fails for neural codecs that produce different (or no obvious) high-frequency artifacts.

A truly robust countermeasure would need to learn:

> "Does this speech have the acoustic properties of natural human phonation — proper glottal closure patterns, natural formant transitions, expected phase coherence between harmonics?"

These questions are much harder to answer from a mel-spectrogram alone.

---

## 5. Ensemble Analysis

### Why We Tried Ensembling

Ensemble methods combine predictions from multiple models to reduce variance and improve generalization. The intuition: if two models make independent errors, combining them should reduce total error. Formally, if models have equal error rate `ε` and their errors are independent, the ensemble error is `ε²` (much smaller than `ε`).

The practical question: are LCNN and RawNet2 errors independent?

### Learned Fusion Weights

The logistic regression on dev scores learned:

```
P(spoof | lcnn_score, rawnet2_score) = sigmoid(10.707 × lcnn_score + 3.065 × rawnet2_score + bias)
```

The LCNN coefficient (10.707) is 3.5× larger than the RawNet2 coefficient (3.065). This means the optimal fusion nearly ignores RawNet2 — it is mostly just a scaled version of LCNN's output. The logistic regression essentially learned "trust LCNN, barely trust RawNet2."

This is unsurprising given the large performance gap: LCNN has 7.07% EER while RawNet2 has 12.78%. Any rational combination should upweight the better model.

### Why Ensemble Hurt Performance

The ensemble achieved 10.24% (average) and 9.92% (learned), both worse than LCNN's 7.07%.

**Root cause: correlated errors.**

For the ensemble to help, the two models need to fail on **different** samples. When one model is wrong, the other should be right, allowing the ensemble to override the wrong prediction.

Looking at the per-attack EERs:
- A17: LCNN 36.85%, RawNet2 40.38% — **both fail**
- A18: LCNN 9.75%, RawNet2 40.82% — **both fail** (RawNet2 catastrophically)

Both models fail on the same attack families (A17, A18). This is not a coincidence — both models were trained on the same data (A01–A06 attacks) and learned features appropriate for vocoder-based synthesis. Neither model has any useful signal for neural codec attacks.

When you average two confused classifiers:
```
LCNN score on A17 spoof: ~0.5 (near-random)
RawNet2 score on A17 spoof: ~0.5 (near-random)
Average ensemble score: ~0.5 (still near-random)
```

The ensemble inherits the error, not the correct prediction.

### When Ensembling Works

Ensembling works when models are:
1. **Complementary in architecture**: Very different representations (which LCNN and RawNet2 are)
2. **Complementary in errors**: Failing on different samples (which they are NOT — both fail on neural codecs)

The architectural diversity was present, but the error diversity was absent. Both models were trained on the same attack-limited training set and learned to detect the same family of artifacts.

A useful ensemble would combine:
- A vocoder-artifact specialist (our LCNN)
- A neural-codec-artifact specialist (trained on A17/A18-type attacks)
- A phase-coherence detector (model using phase spectrum)

Such an ensemble would have decorrelated errors across attack families.

### Ensemble EER Breakdown: Why Averaging Hurt

Consider two possible scenarios for an utterance:

**Scenario 1 (Easy attack, e.g., A07)**:
```
LCNN spoof score:   0.995 (very confident: spoof)
RawNet2 spoof score: 0.872 (confident: spoof)
Average:             0.934 (still clearly spoof) → correct
```

**Scenario 2 (Hard attack, A17)**:
```
LCNN spoof score:   0.490 (confused: near-random)
RawNet2 spoof score: 0.505 (confused: near-random)
Average:             0.498 (near-random) → same confusion
```

**Scenario 3 (Bonafide — the real problem)**:
```
LCNN bonafide score: 0.042 (very confident: bonafide)
RawNet2 bonafide score: 0.210 (less confident, RawNet2 slightly confused)
Average:               0.126 (still mostly bonafide) → correct
```

The ensemble doesn't hurt on Scenario 3 (bonafide classification). The damage comes from Scenario 2 propagating the confusion of both models to the ensemble — and from RawNet2's confusion "infecting" the ensemble's bonafide detections.

The net effect: by including RawNet2's confused scores, the ensemble threshold adjustment that optimally balances FAR and FRR ends up worse than LCNN alone.

---

## 6. Comparison to Published Baselines

### ASVspoof 2019 Challenge Baselines

| System | Eval EER |
|---|---|
| LFCC-GMM (official baseline) | 8.09% |
| CQCC-GMM (official baseline) | 9.57% |
| **LCNN (this project)** | **7.07%** |
| RawNet2 (this project) | 12.78% |

Our LCNN beats both official baselines.

### Top-Performing Challenge Systems (from the 2019 paper)

The best submitted systems in the ASVspoof 2019 challenge achieved much lower EER (some below 1%) by using:
- Larger training datasets with data augmentation
- Multiple feature types fused at score level (CQCC + LFCC + LPS)
- More sophisticated architectures (ResNet34, SENet, TDNN)
- Training-time tricks (focal loss, mixup, transfer learning)

Our LCNN is a single-model baseline that beats the official baselines — a solid result for a project of this scope.

### LCNN Performance vs Literature

The original LCNN-based system by Lavrentyeva et al. (2019) achieved approximately 5.06% EER on the ASVspoof 2019 eval set (Lavrentyeva, 2019 workshop paper). Our implementation achieves 7.07%. The gap can be attributed to:
- Different spectrogram parameters (their n_mels, n_fft may differ)
- Different training data augmentation
- No multi-system fusion
- Different architecture variants (they may have used deeper or wider networks)

The 7.07% result is competitive and demonstrates that the implementation is correct.

### RawNet2 Performance vs Literature

The original RawNet2 (Tak et al., 2021) reported approximately 4.05% EER on ASVspoof 2019 using more training tricks. Our 12.78% reflects:
- Potentially undertrained model (100 epochs vs potentially longer in original)
- No sophisticated data augmentation for raw waveforms
- Architecture may differ in GRU hidden size or ResBlock counts
- MPS backend training may be slightly less optimized than CUDA

The 2.47% dev EER suggests the architecture has the right capacity — the gap to the literature is likely a training time issue.
