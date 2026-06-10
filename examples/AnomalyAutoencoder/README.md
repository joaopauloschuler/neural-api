# AnomalyAutoencoder — reconstruction-error anomaly detection

Trains an undercomplete autoencoder on ONE "normal" data distribution, then
scores test samples by reconstruction error `||x − decode(encode(x))||²`. Normal
points reconstruct well (low error); held-out **anomalies** the autoencoder never
saw reconstruct poorly (high error). Ranking by reconstruction error separates
the two, quantified by **AUROC**. Fully **synthetic** (generated in-code; no data
files), pure CPU, runs in a few seconds.

## Data (8-D vectors)

- **NORMAL**: points lying near a 2-parameter *curved* manifold embedded in 8-D
  (two latent angles drive a fixed nonlinear embedding) plus small Gaussian
  jitter. The cloud lives near a 2-D surface — exactly what a 2-unit bottleneck
  can capture.
- **ANOMALY**: isotropic Gaussian blobs **off** that manifold (shifted centre,
  wider spread). They do not lie on the learned surface, so the trained AE
  reconstructs them badly.

## Model

```
Input(8) -> FC-ReLU(16) -> FC-ReLU(2 bottleneck) -> FC-ReLU(16) -> FC-Linear(8)
```

Trained with plain SGD to minimise MSE against the **input** (the AE target is
the input). Only NORMAL samples are used for training. All layers are existing
library layers.

## AUROC

Anomalies are treated as positives, held-in normal test points as negatives.
AUROC is computed in-code via the Mann-Whitney-U rank statistic with tie-averaged
ranks (helper copied from `examples/MahalanobisOOD`, kept local on purpose).

## Build & run

```
lazbuild examples/AnomalyAutoencoder/AnomalyAutoencoder.lpi
./bin/x86_64-linux/bin/AnomalyAutoencoder
```

## Sample output

```
=== AnomalyAutoencoder: reconstruction-error anomaly detection ===
dim=8  bottleneck=2  train_normal=2000  test_normal=600  test_anomaly=600
NORMAL = points near a 2-D curved manifold in 8-D; ANOMALY = off-manifold Gaussian blobs

autoencoder params = 320
Idx   Layer                            Output Shape                 Params      Neurons
---------------------------------------------------------------------------------------
0     TNNetInput                       (8, 1, 1)                         0            0
1     TNNetFullConnectReLU             (16, 1, 1)                      128           16
2     TNNetFullConnectReLU             (2, 1, 1)                        32            2
3     TNNetFullConnectReLU             (16, 1, 1)                       32           16
4     TNNetFullConnectLinear           (8, 1, 1)                       128            8
---------------------------------------------------------------------------------------
Totals: 5 layers, 320 weights, 42 neurons


training the autoencoder on NORMAL samples (40 epochs)...
  epoch   mean_recon_MSE
      1       0.285055
      5       0.174409
     10       0.158571
     15       0.145558
     20       0.137126
     25       0.131587
     30       0.130949
     35       0.132973
     40       0.131907

reconstruction error (anomaly score):
  NORMAL  test  n= 600  mean =    0.94782
  ANOMALY test  n= 600  mean =    6.38120

AUROC (normal vs anomaly by recon error) = 0.9706

PASS: AUROC 0.9706 > 0.85 -- reconstruction error separates anomalies from normal points.
```

## How to read it

- The **mean_recon_MSE** column is the training loss curve; it drops then
  plateaus once the AE has captured the 2-D manifold.
- The mean reconstruction error on ANOMALY points (~6.4) is far above that on
  NORMAL points (~0.95): off-manifold samples cannot be squeezed through the
  2-unit bottleneck and reconstructed.
- **AUROC ≈ 0.97** means a randomly chosen anomaly almost always has a higher
  reconstruction error than a randomly chosen normal point. The program
  self-checks and `Halt(1)`s if AUROC drops below 0.85.
