# 05 — Training

## Table of Contents

1. [Loss Function — Weighted CrossEntropy](#1-loss-function--weighted-crossentropy)
2. [Optimizer — Adam](#2-optimizer--adam)
3. [Learning Rate Schedule — CosineAnnealingLR](#3-learning-rate-schedule--cosineannealinglr)
4. [Early Stopping](#4-early-stopping)
5. [TensorBoard Logging](#5-tensorboard-logging)
6. [LCNN Training Story](#6-lcnn-training-story)
7. [RawNet2 Training Story](#7-rawnet2-training-story)
8. [Checkpoint Saving and Resuming](#8-checkpoint-saving-and-resuming)
9. [The Trainer Class](#9-the-trainer-class)

---

## 1. Loss Function — Weighted CrossEntropy

### Standard CrossEntropy

For a 2-class classification problem, CrossEntropy loss is:

```
CE(y, p) = -(y * log(p) + (1-y) * log(1-p))
```

where `y ∈ {0, 1}` is the true label and `p ∈ [0, 1]` is the predicted probability of class 1 (spoof).

In PyTorch's `nn.CrossEntropyLoss`, the input is raw logits (not softmax output), and the formula extends to K classes:

```
CE(z, y) = -log(exp(z_y) / sum_k exp(z_k))
         = -z_y + log(sum_k exp(z_k))
```

where `z` is the logit vector and `y` is the target class index.

### Weighted CrossEntropy

The weighted version multiplies the loss for each sample by the class weight:

```
WCE(z, y, w) = -w_y * z_y + w_y * log(sum_k exp(z_k))
```

Equivalently, the final per-batch loss is the weighted average of per-sample losses:

```
L = (1/N) * sum_i w_{y_i} * CE(z_i, y_i)
```

### Our Weights: [4.92, 0.56]

```python
weight_bonafide = 25380 / (2 × 2580)  = 4.9186 ≈ 4.92   # index 0
weight_spoof    = 25380 / (2 × 22800) = 0.5566 ≈ 0.56   # index 1
```

Every bonafide example's loss contribution is scaled by 4.92. Every spoof example's loss contribution is scaled by 0.56. The total expected loss from bonafide examples equals the total from spoof examples, regardless of class frequencies.

### Effect on Gradient

The gradient of the weighted loss with respect to model parameters is:

```
dL/dθ = (1/N) × sum_i w_{y_i} × dCE(z_i, y_i)/dθ
```

Bonafide samples produce gradients that are 4.92/0.56 ≈ 8.8× larger than spoof samples. The optimizer step is proportionally larger for errors on bonafide samples.

### Implementation

```python
# src/training/losses.py
weights = compute_class_weights(df).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
```

The weight tensor is moved to the same device as the model. PyTorch handles the rest automatically in the forward pass of CrossEntropyLoss.

---

## 2. Optimizer — Adam

### Why Adam and Not SGD

**SGD (Stochastic Gradient Descent)** updates parameters with:
```
θ ← θ - lr × dL/dθ
```

**Adam** (Adaptive Moment Estimation, Kingma & Ba, 2015) maintains running estimates of the first and second moments of the gradient:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          # first moment (mean)
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2         # second moment (variance)

m̂_t = m_t / (1 - beta1^t)   # bias-corrected
v̂_t = v_t / (1 - beta2^t)

θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + epsilon)
```

Default values: `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`.

Adam advantages for this project:

1. **Per-parameter learning rates**: Parameters with large consistent gradients (like the final Linear layer) are updated with smaller effective steps. Parameters with small or inconsistent gradients receive larger effective steps. This self-normalizing behavior is crucial for architectures like LCNN where different layers have very different gradient magnitudes.

2. **Momentum**: The first moment `m_t` provides momentum — the update continues in the direction of previous gradients even if the current gradient is small. This helps escape saddle points and flat regions of the loss surface.

3. **Robust to scale**: Adam is less sensitive to the choice of learning rate than SGD. With SGD, a learning rate that is too large causes divergence; with Adam, the adaptive denominator provides a natural bound.

4. **Fast convergence**: Adam typically converges in fewer epochs than SGD with momentum, which is why LCNN converges in just 28 epochs.

### Hyperparameters

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,           # initial learning rate
    weight_decay=1e-4  # L2 regularization coefficient
)
```

**lr=1e-4**: A conservative learning rate appropriate for fine-tuning and medium-sized models. `1e-3` is Adam's default and often too large for audio models; `1e-5` is sometimes used but leads to very slow convergence.

**weight_decay=1e-4**: L2 regularization. Adam applies weight decay by adding `weight_decay × θ` to the gradient before the update step. This penalizes large parameter values and prevents overfitting. The effect is that the loss function being minimized is:

```
L_regularized = L + (weight_decay / 2) × ||θ||^2
```

With `weight_decay=1e-4`, the regularization term becomes significant when `||θ||^2` is large — which happens when individual weights grow large. This is a gentle but effective safeguard against overfitting the 25K training examples with 700K parameters.

> Technically, `weight_decay` in PyTorch's Adam is not the correct AdamW (decoupled weight decay). AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the adaptive step. For this project, the standard Adam with weight decay was used — the difference is generally small for the learning rates used here.

---

## 3. Learning Rate Schedule — CosineAnnealingLR

### What the Cosine Schedule Does

`CosineAnnealingLR` varies the learning rate as a cosine function:

```
lr_t = eta_min + (lr_0 - eta_min) × (1 + cos(π × t / T_max)) / 2
```

where:
- `lr_0 = 1e-4` is the initial learning rate
- `eta_min = 1e-6` is the minimum learning rate
- `T_max = 50` (LCNN) or `T_max = 100` (RawNet2) is the period
- `t` is the current epoch

The learning rate follows this curve:

```
lr
1e-4 |*
     | *
     |   *
5e-5 |     *
     |       *
     |         *
1e-6 |           ****---> epoch
     0    25    50
```

At `t=0`: `lr = 1e-4` (maximum)
At `t=T_max/2 = 25`: `lr ≈ 5e-5` (midpoint)
At `t=T_max = 50`: `lr = 1e-6` (minimum)

### Why Cosine Annealing?

1. **Warm start to fast convergence**: The initial high learning rate allows the model to make large parameter updates and move quickly toward a good region of the loss surface.

2. **Gradual refinement**: As the learning rate decreases, the updates become smaller and the model refines its parameters within the found basin.

3. **No plateau issue**: Unlike step decay (reduce lr by 10× every N epochs), cosine annealing is smooth and continuous. There is no sudden change that might cause instability.

4. **Consistent with early stopping**: The schedule is designed for `T_max` epochs, but early stopping may terminate training before `T_max`. Since the schedule always moves toward `eta_min`, early termination simply means the model was trained with a still-decreasing learning rate — not harmful.

### Configuration

```yaml
# configs/lcnn.yaml
scheduler:
  T_max: 50      # LCNN: 50 epochs total budget
  eta_min: 0.000001   # final lr = 1e-6

# configs/rawnet2.yaml
scheduler:
  T_max: 100     # RawNet2: 100 epochs (much longer training)
  eta_min: 0.000001
```

---

## 4. Early Stopping

### Mechanism

Early stopping prevents overfitting by halting training when the validation metric stops improving:

```python
if dev_eer < self.best_eer:
    self.best_eer = dev_eer
    self.epochs_no_improve = 0
    torch.save(model.state_dict(), checkpoint_dir / "best.pt")
else:
    self.epochs_no_improve += 1

if self.epochs_no_improve >= self.patience:
    print(f"Early stopping at epoch {epoch}")
    break
```

- **LCNN patience=10**: If dev EER doesn't improve for 10 consecutive epochs, training stops.
- **RawNet2 patience=15**: More patient because raw waveform models can have more volatile training curves.

### Why Monitor EER, Not Loss?

Early stopping based on **development loss** has a known failure mode: the loss can decrease (model becomes more confident) while accuracy/EER remains constant or worsens (the model is just becoming overconfident about wrong predictions). This is particularly common with cross-entropy loss on imbalanced datasets.

Early stopping based on **development EER** directly tracks the metric we care about. The checkpoint is saved when the model achieves its best discrimination between bonafide and spoof — not when its logit magnitudes happen to be calibrated.

The EER is a threshold-free metric, so it is not affected by overconfidence. A model that outputs `[0.01, 0.99]` and a model that outputs `[0.0, 1.0]` both assign rank order "spoof" and produce the same EER.

### NaN EER Handling

```python
if dev_eer != dev_eer:   # NaN check (NaN != NaN is True)
    print("EER is NaN (too few samples) — skipping checkpoint")
    continue
```

`compute_eer` returns `float("nan")` when the development batch contains only one class (e.g., if a minibatch during early epochs happens to contain all bonafide or all spoof). The NaN check prevents `best.pt` from being saved with an invalid EER value.

---

## 5. TensorBoard Logging

The `Trainer` class writes the following scalars to TensorBoard at every epoch:

```python
self.writer.add_scalar("loss/train", train_loss, epoch)
self.writer.add_scalar("loss/dev",   dev_loss,   epoch)
self.writer.add_scalar("EER/dev",    dev_eer,    epoch)
self.writer.add_scalar("lr",         lr,         epoch)
```

### Log Directory Structure

```
runs/
├── lcnn/                   # LCNN TensorBoard logs
│   └── events.out.tfevents.xxxxx
└── rawnet2/                # RawNet2 TensorBoard logs
    └── events.out.tfevents.xxxxx
```

### Viewing Logs

```bash
tensorboard --logdir runs/
# Open http://localhost:6006 in a browser
```

TensorBoard shows:
- **loss/train vs loss/dev**: Divergence between these indicates overfitting
- **EER/dev**: The metric being minimized — should trend down and then plateau
- **lr**: Verifies the cosine schedule is working correctly

The TensorBoard logs are committed to git (unlike checkpoints and data) because they provide a complete record of the training run.

---

## 6. LCNN Training Story

### Setup

```bash
python scripts/train.py
# Config: configs/lcnn.yaml
# Device: MPS (Apple M-series chip)
# Batch size: 32
# Max epochs: 50
# Early stopping patience: 10
```

### What Happened

**Epochs 1–5**: Rapid loss decrease. The model quickly learned to distinguish the coarsest features — possibly speaker identity or average spectral shape — and began separating bonafide from spoof.

**Epochs 5–15**: Dev EER drops to near-zero. The model found the high-frequency vocoder artifact features (confirmed later by Grad-CAM). With 6 known attack families in the dev set (A01–A06) and the model already learning their shared high-frequency artifacts, near-perfect dev EER is achievable.

**Epochs 18–27**: Dev EER stable at 0.0%. The model improved confidence in its predictions (train loss continued decreasing) but the discrimination was already perfect.

**Epoch 28**: Early stopping triggered. The model had 10 consecutive epochs with dev EER = 0.0000% (no improvement). The best checkpoint was saved at the epoch when EER first reached 0.0%.

### Dev EER 0% — Is This Overfitting?

A dev EER of 0.0000% seems suspiciously perfect. However, this is not overfitting in the traditional sense:

1. **Different speakers**: Dev set uses speakers not seen in training
2. **Same attack families**: The dev attacks (A01–A06) are the same as training attacks (A01–A06)
3. **High separability**: The vocoder artifacts in A01–A06 are apparently so distinctive that the LCNN can reliably detect them

The true test is eval EER (7.07%) on unseen attacks. The gap between 0% dev and 7.07% eval represents the generalization cost of seeing unseen attack families.

### Training Duration

LCNN training on Apple Silicon MPS: approximately **2–3 hours** for 28 epochs of 25,380 training examples at batch size 32 = ~793 steps/epoch.

---

## 7. RawNet2 Training Story

### Setup

```bash
python scripts/train_rawnet2.py
# Config: configs/rawnet2.yaml
# Device: MPS (Apple M-series chip)
# Batch size: 32
# Max epochs: 100 (first run: 50, then resumed for 50 more)
# Early stopping patience: 15
```

### First Run: 50 Epochs — Failure

The first training run was configured for 50 epochs (same as LCNN). After 50 epochs:

- Dev EER: **24.37%** — barely better than random (50% would be random)
- The model had not converged

**Why did RawNet2 take so much longer?**

1. **Raw feature learning**: SincConv must learn which frequency bands are discriminative from scratch. The mel filterbank gives LCNN a warm start aligned with speech frequency structure.

2. **More parameters**: 4.9M vs 700K parameters. More parameters = more optimization steps needed to find a good configuration.

3. **GRU training difficulty**: The GRU has ~3.1M parameters that all interact through backpropagation through time (BPTT). The gradients must flow backward through 444 time steps. Despite skip connections in the ResBlocks preventing vanishing gradients in the feedforward path, the GRU itself has no skip connections.

4. **Harder optimization landscape**: The SincConv filters, ResBlocks, and GRU all interact. Small changes in SincConv filter cutoffs change the input distribution to the ResBlocks, which changes what the GRU sees. This coupling makes the optimization landscape more complex.

### Resuming Training: Epochs 51–114

Training was resumed from the epoch-50 checkpoint for an additional 50–64 epochs (using the same config with adjusted epoch counting):

```bash
python scripts/train_rawnet2.py  # modified to load epoch-50 checkpoint
```

The learning rate schedule was reset to continue from the cosine curve position at epoch 50:
```
lr at epoch 50 of 100: 1e-6 + (1e-4 - 1e-6) × (1 + cos(π × 50/100))/2
                     = 1e-6 + 9.9e-5 × (1 + cos(π/2))/2
                     = 1e-6 + 9.9e-5 × 0.5 = ~5e-5
```

**Best result**: Dev EER of **2.47%** at approximately epoch 114 total (64 epochs into the second run). Early stopping triggered at patience=15 from this best epoch.

### Lesson: Raw Waveform Models Need More Training

The RawNet2 story illustrates a fundamental principle: end-to-end models with learned feature extraction need significantly more training than models that use hand-crafted features. The LCNN started with a useful prior (mel features) while RawNet2 had to build all features from scratch.

For a production system where training time is not a constraint, running RawNet2 for 200+ epochs might close the gap with LCNN. The 2.47% dev EER suggests the model has real capacity — it just needs more optimization steps to exploit it.

---

## 8. Checkpoint Saving and Resuming

### Checkpoint Format

The Trainer saves only the model's state dictionary:

```python
torch.save(self.model.state_dict(), self.checkpoint_dir / "best.pt")
```

`state_dict()` is a Python dictionary mapping parameter names to tensors. It contains all learned weights and biases but not the model architecture, optimizer state, or training configuration.

**Why only state_dict, not the full model?**

1. **Smaller files**: Only parameters, not the Python class definition.
2. **Portability**: Can load into any compatible model architecture.
3. **Safety**: Avoids Python pickle issues with model class serialization.

The LCNN checkpoint is approximately 2.7 MB. The RawNet2 checkpoint is approximately 18.7 MB.

### Loading for Inference

```python
# src/inference/predict.py
model = LCNN()
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device).eval()
```

`map_location=device` ensures the checkpoint loads correctly regardless of the device it was saved on. A checkpoint saved on MPS can be loaded on CPU or CUDA without issues.

`model.eval()` switches the model from training mode to evaluation mode:
- BatchNorm uses population statistics instead of batch statistics
- Dropout is disabled (all neurons active)
- No gradient tracking (saves memory)

### Resuming Training

To resume RawNet2 from a checkpoint:

```python
model = RawNet2()
model.load_state_dict(torch.load("checkpoints/rawnet2/best.pt", map_location=device))
model.to(device).train()  # switch back to train mode

# Recreate optimizer and scheduler from scratch
# (optimizer state is not saved — this causes momentum reset, which is acceptable)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

Note that the optimizer state (Adam's `m_t` and `v_t` moments) is not saved. This means the resumed training starts with cold momentum, equivalent to a fresh Adam start from the current model weights. In practice, Adam's moments re-accumulate quickly within a few steps.

For future runs, a more robust approach would save the full training state:
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_eer': best_eer,
}, "checkpoint_full.pt")
```

---

## 9. The Trainer Class

The `Trainer` class in `src/training/trainer.py` encapsulates the complete training loop:

```python
class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device,
                 checkpoint_dir, log_dir, patience=10):
        ...

    def train_epoch(self, loader, epoch) -> float:
        """One epoch of gradient descent. Returns average train loss."""

    def eval_epoch(self, loader, epoch) -> tuple[float, float]:
        """One epoch of evaluation. Returns (avg_loss, EER)."""

    def fit(self, train_loader, dev_loader, epochs):
        """Main training loop with early stopping."""
```

### train_epoch Flow

```python
self.model.train()           # enable Dropout, use batch stats for BN
for x, labels in loader:
    x, labels = x.to(device), labels.to(device)
    optimizer.zero_grad()    # clear gradients from previous step
    logits = model(x)        # forward pass
    loss = criterion(logits, labels)  # weighted CE
    loss.backward()          # compute gradients
    optimizer.step()         # update parameters
```

### eval_epoch Flow

```python
self.model.eval()            # disable Dropout, use population stats for BN
with torch.no_grad():        # disable gradient tracking (saves memory)
    for x, labels in loader:
        logits = model(x)
        scores = softmax(logits)[:, 1]   # spoof probability
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())

eer = compute_eer(all_labels, all_scores)
```

The evaluation uses softmax to convert logits to probabilities. The spoof probability (class 1) is used as the detection score — higher means more likely spoof. EER is computed on the full dev set after collecting all scores.

### fit Loop

```python
for epoch in range(1, epochs + 1):
    train_loss = train_epoch(train_loader, epoch)
    dev_loss, dev_eer = eval_epoch(dev_loader, epoch)
    scheduler.step()
    # Log to TensorBoard
    # Save checkpoint if dev_eer improved
    # Check early stopping
```

`scheduler.step()` is called **after** evaluation, not inside the training loop. CosineAnnealingLR's step corresponds to one epoch. Calling it inside the training loop would advance the schedule by one step per batch.
