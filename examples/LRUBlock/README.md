# LRU block — a transformer-style Linear Recurrent Unit tower

Demonstrates `TNNet.AddLRU`, the full **LRU block** builder, on a long-range
**delayed-recall** task. The LRU (Orvieto et al. 2023, "Resurrecting RNNs for
Long Sequences", [arXiv:2303.06349](https://arxiv.org/abs/2303.06349)) is a
stable complex-diagonal linear recurrence whose eigenvalues are parameterised to
sit just inside the unit circle, so it can carry information across very long
delays with negligible decay.

## The block

The leaf layer `TNNetLRU` is a bare recurrence cell (shape-preserving `D → D`
over a `(SeqLen, 1, D)` sequence, exact BPTT). The LRU paper wraps it in a
*block* that drops into a transformer-style residual stack — and that is exactly
what the two new builders provide:

`TNNet.AddLRUMixer()` — the LRU time-mixing **arm** (shape-preserving `D → D`):

```
x  ──► PointwiseConvLinear(D)  ──►  TNNetLRU  ──►  PointwiseConvLinear(2D) ──► SwiGLU ──► PointwiseConvLinear(D)
       (per-token input proj)      (recurrence)    └────────── GLU non-linearity ─────────┘   (output proj)
```

(No token-shift: the LRU is *itself* a temporal mixer, so the arm only mixes
channels around it. The GLU is realised as a `2D` projection followed by
`TNNetSwiGLU`, which halves the depth channel-wise back to `D`.)

`TNNet.AddLRU(d_ff, PreNorm = true, NormClass = nil)` — the full **block**,
mirroring `AddGatedLinearAttentionBlock`: two pre/post-norm residual sub-blocks,

```
x := x + LRUMixer(LayerNorm(x))     (LRU time-mixing residual)
x := x + FFN(LayerNorm(x))          (token-wise SwiGLU FFN residual)
```

`d_model` is inferred from the input depth and the output shape matches the
input, so blocks stack into a deep tower. `NormClass` defaults to
`TNNetLayerNorm`. Both functions require a `(SeqLen, 1, D)` sequence (`SizeY = 1`)
and use `PointwiseConvLinear` everywhere (a `FullConnect` would flatten/mix the
time axis and zero the input gradient).

## The task

A one-hot symbol is presented with a cue flag at position `0` of a length-24
sequence; positions `1 … L-2` are noise distractors; at the **final** position
the network must output the symbol seen at position `0`. Solving it requires
propagating information across the whole sequence — the LRU's design target.

```
pos 0      : [ symbol_onehot(s) | flag=1 ]
pos 1..L-2 : [ noise...         | flag=0 ]   (distractors)
pos L-1    : [ 0...0            | flag=0 ]    target = symbol_onehot(s)
```

The model is a per-token embedding → 2× `AddLRU` blocks → per-token softmax head;
the recall decision reads the final position.

## Running

```
lazbuild LRUBlock.lpi
../../bin/x86_64-linux/bin/LRUBlock
```

Pure CPU, ~18 s on 2 cores. Example output (chance = 16.7 %):

```
training (8000 steps), loss curve:
  step  1000   recall CE = 4.2857   recall acc = 26.0%
  step  2000   recall CE = 4.1890   recall acc = 42.5%
  step  3000   recall CE = 4.2296   recall acc = 33.5%
  step  4000   recall CE = 3.0582   recall acc = 35.5%
  step  5000   recall CE = 1.9771   recall acc = 62.5%
  step  6000   recall CE = 1.2019   recall acc = 62.5%
  step  7000   recall CE = 0.4067   recall acc = 82.0%
  step  8000   recall CE = 0.2608   recall acc = 95.0%

held-out recall accuracy over 500 sequences = 96.2%
```

The recall cross-entropy falls from ~4.3 to ~0.26 and accuracy climbs from chance
to ~96 % across the 24-step delay — the AddLRU tower learns the long-range
dependency.

Coded by Claude (AI).
