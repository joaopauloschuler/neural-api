# Normalization cheat sheet

A quick reference for the normalization layers in `neural/neuralnetwork.pas`. The
goal is to make it easy to pick the right layer: which **axes** it reduces over,
whether it has **learnable parameters**, what it computes, and when to reach for
it.

A few conventions used below:

* **Sample** means one item in the batch. neural-api processes one sample at a
  time, so "per-sample" statistics are computed from a single volume of shape
  `SizeX * SizeY * Depth` — there are **no batch statistics** anywhere in this
  list (this is the main difference from textbook BatchNorm).
* **Depth** is the channel axis; `(X, Y)` are the spatial axes.
* `gamma` = learnable scale, `beta` = learnable bias, `alpha` = learnable scalar.

## Summary table

| Layer (constructor) | Reduces over | Learnable params | Formula (per element) | Pick it when |
|---|---|---|---|---|
| `TNNetLayerNorm.Create()` | whole sample (X, Y **and** Depth) | gamma + beta, per-element (one per X*Y*Depth) | `y = gamma * (x - mean) / sqrt(var + eps) + beta` | Transformers / RNNs; the general-purpose batch-independent norm. |
| `TNNetRMSNorm.Create()` | whole sample (X, Y **and** Depth) | gamma only, per-element | `y = gamma * x / sqrt(mean(x^2) + eps)` | Cheaper LayerNorm for transformers; skip mean-centering. |
| `TNNetRMSNormGated.Create()` | whole sample (X, Y **and** Depth) for the RMS; gate is per-channel | gate logit `g[c]` only, per-channel (init 0) | `y = (x / sqrt(mean(x^2) + eps)) * sigmoid(g[c])` | RMSNorm with a learnable per-channel on/off gate instead of gamma. |
| `TNNetZScore.Create()` | whole sample (X, Y **and** Depth) | none | `y = (x - mean) / sqrt(var + eps)` | LayerNorm's normalization without the affine; fixed standardization. |
| `TNNetGroupNorm.Create(Groups)` | each contiguous channel group, over (X, Y + channels-in-group) | gamma + beta, per-element | `y = gamma * (x - mean_g) / sqrt(var_g + eps) + beta` | Vision / small batches where BatchNorm is unstable. |
| `TNNetInstanceNorm.Create()` | each single channel, over (X, Y) | gamma + beta, per-element | GroupNorm with `Groups = Depth` | Style transfer / generative vision; per-channel contrast. |
| `TNNetMovingStdNormalization.Create()` | whole sample (running mean/std) | 2 trainable scalars (shift, scale) | `y = (x - shift) / std` | Drop-in BatchNorm-ish standardization for the whole tensor. |
| `TNNetChannelStdNormalization.Create()` | per channel | per-channel scale (+ inherited zero-center) | per-channel zero-center then `* scale[c]` | Per-channel std normalization (the repo's "channel std norm"). |
| `TNNetPixelNorm.Create()` | per (X, Y) pixel, over Depth | none | `y = x / sqrt(mean_depth(x^2) + eps)` | StyleGAN-style per-pixel feature-vector norm; GAN generators. |
| `TNNetL2Normalize.Create([axis][,eps])` | axis 0 (default): per (X, Y) over Depth; axis 1: whole sample; axis 2: per-channel over (X, Y) | none | `y = x / sqrt(sum(x^2) + eps)` | Unit-length feature vectors (embeddings, cosine similarity). |
| `TNNetUnitNorm.Create()` | whole sample (flattened) | none | `y = x / sqrt(sum_all(x^2) + eps)` | Full-volume unit-L2 (Keras "UnitNorm"); alias of the line above. |
| `TNNetMinMaxNorm.Create([eps])` | whole sample (X, Y **and** Depth) | none | `y = (x - min) / (max - min + eps)` | Rescale a whole sample to ~`[0, 1]`. |
| `TNNetGRN.Create()` | per channel L2 over (X, Y), then across channels | gamma + beta, per-channel (both init 0) | `y = gamma[c] * (x * Nx[c]) + beta[c] + x` | ConvNeXt-V2 blocks; channel-competition contrast norm. |
| `TNNetDyT.Create()` | nothing (no statistics) | gamma + beta per channel, single alpha | `y = gamma[c] * tanh(alpha * x) + beta[c]` | Normalization-FREE drop-in LayerNorm replacement. |
| `TNNetLogitNormalize.Create([tau][,eps])` | per (X, Y) over Depth | none | `y = x / (tau * sqrt(sum_depth(x^2)) + eps)` | Pre-softmax logit regularizer for calibration / OOD. |

## Per-layer notes

### `TNNetLayerNorm`
Constructor: `Create()`. eps = `1e-5` (fixed). Normalizes each sample over **all**
its elements (`SizeX*SizeY*Depth`) to zero mean / unit variance, then applies a
learnable per-element `gamma` (init 1) and `beta` (init 0). gamma and beta have
one weight **per element** of the volume, not per channel. The general-purpose,
batch-independent norm — first choice for transformers and recurrent models.

### `TNNetRMSNorm`
Constructor: `Create()`. eps = `1e-5` (fixed). Like LayerNorm but divides by the
root-mean-square of the elements **without subtracting the mean**, then applies a
learnable per-element `gamma` (init 1). No `beta`. Cheaper than LayerNorm and a
common choice in modern transformer stacks.

### `TNNetRMSNormGated`
Constructor: `Create()`. eps = `1e-5` (fixed), and the RMS is taken over the
**whole sample** with **no mean subtraction**, exactly like `TNNetRMSNorm`. The
difference is the affine: instead of a per-element `gamma`, it applies a learnable
**per-channel sigmoid gate** `y[x,y,c] = n[x,y,c] * sigmoid(g[c])`, where
`n = x / sqrt(mean(x^2) + eps)` and there is **one gate logit `g[c]` per `Depth`
channel** (the `TNNetGatedResidual` storage pattern: `FNeurons[0].Weights` holds
the `Depth` logits). The logits are initialised to **0**, so at init every gate is
`sigmoid(0) = 0.5` and the layer simply halves the normalized activation; channels
then learn to open (→1) or close (→0) independently. The backward pass routes the
input error through both the per-channel scale `sigmoid(g[c])` and the shared
`invRMS` Jacobian (the RMS term couples all elements of the sample), reusing
RMSNorm's exact normalization Jacobian. Pick it when you want RMSNorm but with a
cheap, learnable per-channel gating instead of a full per-element scale.

### `TNNetZScore`
Constructor: `Create()`. The unparameterised core of LayerNorm:
`y = (x - mean) / sqrt(var + eps)` over the whole sample, with **no** learnable
gamma/beta. Use it when you want fixed standardization without an affine.

### `TNNetGroupNorm`
Constructor: `Create(Groups: integer)`. eps = `1e-5` (fixed). Splits `Depth` into
`Groups` contiguous channel groups and normalizes each group independently over
(X, Y + channels-in-group), then applies a learnable per-element `gamma` (init 1)
and `beta` (init 0) over the full volume. **Repo behavior note:** if `Depth` is
**not divisible** by `Groups`, it silently falls back to a **single group**
(equivalent to a per-sample LayerNorm-without-mean-split) rather than erroring.
Good for vision tasks and small-batch regimes where BatchNorm is noisy.

### `TNNetInstanceNorm`
Constructor: `Create()`. A `TNNetGroupNorm` with `Groups = Depth` — one channel
per group — resolved from the input depth at `SetPrevLayer` time. Each channel is
normalized independently over its spatial `(X, Y)` extent. Same learnable
per-element `gamma`/`beta` as GroupNorm. Typical in style transfer and generative
vision models.

### `TNNetMovingStdNormalization`
Constructor: `Create()`. The repo's batch-norm-style "moving" standardization:
subtracts a learned shift and divides by a learned standard-deviation scalar over
the whole tensor. It carries **2 trainable scalars** (shift and std). **Repo
behavior note:** the std update is deliberately damped (≈100x slower than the
zero-centering term, see `GetMaxAbsoluteDelta` returning `* 0.01`) to avoid
overflow spikes; the std divisor is only applied when `std > 0` and `std <> 1`.
Use it as a possible drop-in replacement for batch normalization on a whole
tensor (also reachable via `TNNet.AddMovingNorm`).

### `TNNetChannelStdNormalization`
Constructor: `Create()`. This is the repo's **per-channel std normalization**
(descends from `TNNetChannelZeroCenter`): it zero-centers per channel and then
multiplies each channel by a trainable per-channel scale (one weight per `Depth`
channel, init 1). **Repo behavior note:** the std-deviation learning is again
heavily damped (`-FLearningRate*0.01 / channelSize`) and, on the backward pass,
the channel-error scaling is clamped with `SetMin(1)` because "the direction of
the error is more important than its magnitude." Reachable per channel via
`TNNet.AddChannelMovingNorm`.

### `TNNetPixelNorm`
Constructor: `Create()`. eps = `1e-8` (fixed). StyleGAN-style per-pixel
feature-vector normalization: for each `(X, Y)` position the `Depth`-dimensional
vector is divided by its root-mean-square over the **depth axis**, giving each
pixel a unit-RMS feature vector. **Parameter-free.** Common in GAN generators.

### `TNNetL2Normalize`
Constructors: `Create()`, `Create(eps)`, `Create(axis)`, `Create(axis, eps)`.
eps default `1e-8`. **Selectable reduction scope** stored in `FStruct[0]`:

* `axis = 0` (**default**, and what bare `Create()` / `Create(eps)` give) —
  per spatial position `(X, Y)`, normalize the depth vector to unit L2 norm.
  This preserves the historical behavior.
* `axis = 1` — reduce sum-of-squares over the **entire flattened sample** so the
  whole volume has unit L2 norm.
* `axis = 2` — per **depth channel**, reduce sum-of-squares over the spatial
  positions `(X, Y)` so each channel's feature map is independently scaled to
  unit L2 norm (`n_d = sqrt(sum_{x,y} x[x,y,d]^2 + eps)`). The per-(X,Y)-over-depth
  transpose of `axis = 0`.

No learnable parameters; the exact `(I - y y^T)/n` Jacobian is applied on the
backward pass over the chosen scope. Use for unit-length embeddings / cosine
similarity. **Note:** this is a true L2 unit-norm, *not* a mean/variance
standardization.

### `TNNetUnitNorm`
Constructor: `Create()`. A thin subclass of `TNNetL2Normalize` whose default
constructor selects the **full-volume** scope (`axis = 1`, eps `1e-8`) — i.e. it
is the Keras "UnitNorm" name for full-volume L2 normalization. Serializes under
its own class name. Behaviorally identical to `TNNetL2Normalize.Create(1)`.

### `TNNetMinMaxNorm`
Constructors: `Create()`, `Create(eps)`. eps default `1e-7`. Rescales the whole
sample by its own global min/max — reduced over **all** positions (X, Y **and**
Depth) — to approximately `[0, 1]`: `y = (x - m) / ((M - m) + eps)`. No learnable
parameters. **Repo behavior note:** the backward pass is a true subgradient that
routes a bulk `1/denom` term to every element plus exact coupling corrections at
the argmin and argmax indices (held fixed); for a constant volume `eps` keeps it
finite and a single index absorbs both corrections.

### `TNNetGRN`
Constructor: `Create()`. eps = `1e-6` (fixed). Global Response Normalization
(ConvNeXt-V2, Woo et al. 2023). For each channel it computes an L2 response over
`(X, Y)`, divides by the **mean response across channels**, then applies a
learnable per-channel `gamma[c]` and `beta[c]` **plus a residual** add of the
input:
`Y[x,y,c] = gamma[c] * (X[x,y,c] * Nx[c]) + beta[c] + X[x,y,c]` where
`Nx[c] = Gx[c] / mean_c(Gx)`. gamma and beta init to **0**, so the layer is the
identity at start. Use inside ConvNeXt-V2-style blocks for channel competition.

### `TNNetDyT`
Constructor: `Create()`. Dynamic Tanh (Liu et al. 2025) — a **normalization-free**
drop-in LayerNorm replacement that uses **no batch or per-sample statistics**:
`Y[x,y,c] = gamma[c] * tanh(alpha * X[x,y,c]) + beta[c]`. Learnable params: a
**single** layer-wide scalar `alpha` (init 1.0) plus per-channel `gamma` (init 1)
and `beta` (init 0). Pick it to drop LayerNorm's per-token statistics while
keeping a squashing + affine response.

### `TNNetLogitNormalize`
Constructors: `Create()`, `Create(tau)`, `Create(tau, eps)`. tau default `1.0`,
eps default `1e-8`. A pre-softmax regularizer (Wei et al. 2022) that divides the
depth-axis logit vector at each `(X, Y)` by a tau-scaled L2 norm:
`y_i = x_i / (tau * sqrt(sum_j x_j^2 + safety) + eps)`. No learnable parameters.
**Repo behavior note:** with `tau = 1` and `eps = 0` it reduces exactly to
`TNNetL2Normalize` (axis 0). Improves calibration and OOD detection by bounding
logit magnitudes during training.
