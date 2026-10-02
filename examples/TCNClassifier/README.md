# TCNClassifier — multivariate Temporal Convolutional Network

A Temporal Convolutional Network (TCN; Bai, Kolter & Koltun 2018,
[arXiv:1803.01271](https://arxiv.org/abs/1803.01271)) that classifies rolling
windows of a **synthetic multivariate** time series (generated in-code; no data
files). The network is built with `TNNet.AddTCNBlock`. A non-dilated baseline
with the same layers and the same weight count is trained on the same data, so
the effect of dilation is visible.

## The task

- 4 sensor channels of Gaussian noise (std 0.3).
- **Cue:** every 30..60 steps, a 3-step pulse (height 3) on one random channel.
- **Distractors:** at each step, with probability 0.04, a 1-step pulse of the
  same height on a random channel.
- **Label** of a window `(64, 1, 4)`: the channel of the most recent cue. Chance
  is 25%.

Cues are sparse, so the most recent cue is often far back in the window. The
program prints, for each model, the fraction of test windows whose label
channel's last COMPLETE 3-step cue ends inside that model's receptive field:
98.6% for the dilated TCN, 24.0% for the baseline. A cue still in progress at
the window end counts as not visible.

About 1 window in 45 has its cue starting at the last step. That single pulse
looks exactly like a distractor, so the best possible accuracy is about 98-99%.

## The model

```text
NN.AddLayer(TNNetInput.Create(64, 1, 4));          // (WindowLen, 1, NumFeatures)
for Dilation in 1, 2, 4, 8, 16:
  NN.AddTCNBlock({Channels=}16, {KernelSize=}2, Dilation, DropoutRate, UseNormalization);
NN.AddLayer(TNNetCrop.Create(63, 0, 1, 1));        // last time step
NN.AddLayer(TNNetFullConnectLinear.Create(4));
NN.AddLayer(TNNetSoftMax.Create());
```

Each `AddTCNBlock` adds two dilated `TNNetCausalConv1D` layers, each followed by
ReLU, plus a residual sum with the block input (a 1x1
`TNNetPointwiseConvLinear` projection when the channel count changes), then a
final ReLU. With two convs per block the receptive field is
`1 + 2*(KernelSize-1)*sum(dilations)` = 63 steps (baseline with all dilations 1:
11 steps). The window length (64) is chosen to be at least the receptive field.

`TNNetCrop` keeps only the last time step. Every conv is causal, so the last
step is the position with the widest view: it sees the last 63 of the 64 steps.

### Options

- `--dropout <rate>` — `DropoutRate` for `TNNetSpatialDropout1D` after each
  ReLU in the branch. **Dropout is optional**: the default 0 adds no dropout
  layers. Use it if the model overfits.
- `--norm` — `UseNormalization = true` adds a `TNNetMovingStdNormalization`
  after each causal conv. **Normalization is off by default.** It uses stored
  running statistics, so the block stays causal (the paper uses weight
  normalization instead).

## Build & run

```
lazbuild examples/TCNClassifier/TCNClassifier.lpi
./bin/x86_64-linux/bin/TCNClassifier            # add --dropout 0.1 / --norm to try the options
```

Training uses `TNeuralFit` (20 epochs, batch 32, learning rate 0.003). It writes
`autosave.nn` / `autosave.csv` in the working directory.

## Measured result (default options, 4-core CPU, AVX2 build)

```
features=4  window=64  series=20000  windows train/val/test=3985/484/484
label = channel of the most recent cue (30..60 steps apart); chance = 25.0%
test windows whose last cue is inside the receptive field: dilated 98.6%, baseline 24.0%
TCN (dilations 1,2,4,8,16): receptive field = 63 steps, 4864 weights, 35 layers
TCN (dilations 1,2,4,8,16): test accuracy  97.11%  (trained in 101.0 s)
Baseline (all dilations 1): receptive field = 11 steps, 4864 weights, 35 layers
Baseline (all dilations 1): test accuracy  39.46%  (trained in 387.4 s)
Summary: dilated TCN 97.11%, non-dilated baseline 39.46%, chance 25.0%
```

Peak resident memory was 36 MB.

**Run time varies widely between runs, for a reason not yet diagnosed.** Two
measured runs of the same build: 8:08 wall clock (dilated 101 s, baseline
387 s) and 33:34 wall clock (dilated ~100 s, baseline 1915 s). The slow parts
are multi-second stalls between epochs (visible in `autosave.csv`), and they
can hit either model.
