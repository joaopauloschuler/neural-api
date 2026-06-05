# NeuralCollapse

Tiny example for `TNNet.NeuralCollapseReport`, which measures the four canonical
**Neural-Collapse** metrics (Papyan, Han & Donoho 2020, *"Prevalence of neural
collapse during the terminal phase of deep learning training"*) on the
**penultimate-layer** features of a classifier.

Where the sibling `FeatureSeparabilityReport` stops at the `tr(Sw)` collapse and
the Fisher-ratio *magnitude* (a partial NC1), this report computes the full
headline result: the **simplex equiangular tight frame (ETF)** geometry (NC2),
**self-duality** (NC3) and the **nearest-class-mean** classifier collapse (NC4).

## The four metrics

| Metric | Meaning | Target |
| ------ | ------- | ------ |
| **NC1** | within-class variability collapse `tr(Sw·Sb⁺)/C` (trace-ratio surrogate `tr(Sw)/tr(Sb)`) | → 0 |
| **NC2** | class means form a simplex ETF: **equinorm** (CV of `‖mu_c − mu‖`) and **equiangular** (every pair's cosine equals `−1/(C−1)`; mean & max deviation printed) | CV → 0, dev → 0 |
| **NC3** | self-duality: cosine alignment between centered class means and the classifier weight rows | → 1 |
| **NC4** | classifier collapses to a nearest-class-mean decision rule | → 1 |

NC1 reuses `FeatureSeparabilityReport`'s class-mean / within-class scatter `Sw` /
between-class scatter `Sb` machinery (not re-derived). NC3 is **honestly skipped
with a printed flag** when the head is not a width-matched
`TNNetFullConnectLinear` / `TNNetPointwiseConvLinear`.

## What the example does

It builds a small softmax classifier
(`8 → FC16+ReLU → FC16+ReLU → FC16(linear, penultimate) → FC4(linear head) → SoftMax`)
on a synthetic **4-class Gaussian-blob** problem (class centres on a regular
polygon in the first two dims, the rest noise distractors), trains it **well past
zero train-error** into the *terminal phase of training*, and calls
`TNNet.NeuralCollapseReport` every N epochs on a **fixed, class-balanced probe**.

It prints an **ASCII trajectory** of the mean pairwise centered-class-mean cosine
marching onto the simplex-ETF target line `−1/(C−1) = −0.3333` — you watch the
simplex assemble itself — followed by the full final report (NC1–NC4 plus the
centered-mean cosine heatmap).

## Sample output

```
  epoch    1  train-acc=0.328  mean pairwise cosine=-0.2954
  epoch   60  train-acc=1.000  mean pairwise cosine=-0.3253
  ...
  epoch  600  train-acc=1.000  mean pairwise cosine=-0.3283

Mean pairwise centered-class-mean cosine over training
(| = current value, T = simplex-ETF target -0.3333):
  ep    1 -0.2954       T|
  ep   60 -0.3253       |
  ...
  ep  600 -0.3283       |

NC1 within-class variability collapse tr(Sw.Sb^+)/C ~= tr(Sw)/tr(Sb) = 0.039230  (-> 0)
NC2 simplex-ETF: equinorm CV(||mu_c-mu||)=0.122951 (-> 0); equiangular target -1/(C-1)=-0.3333
    mean pairwise cosine=-0.3283  mean|dev|=0.114840  max|dev|=0.285847  (-> 0)
NC3 self-duality: mean |cos(mu_c-mu, w_c)| = 0.8586  (-> 1)
NC4 classifier -> nearest-class-mean: 1.0000 of probes (160/160)  (-> 1)
```

The exact ETF angle `−1/(C−1)` is the deterministic target the cosines converge
toward; the smoke test pins it on a hand-built perfect simplex within `1e-4`.

## Build & run

```
cd examples/NeuralCollapse
lazbuild --build-mode=Release NeuralCollapse.lpi
../../bin/x86_64-linux/bin/NeuralCollapse
```

Pure CPU, no dataset download, runs in seconds (well under five minutes).
