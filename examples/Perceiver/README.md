# Perceiver — a latent-bottleneck encoder whose cost is decoupled from input length

This example showcases **`TNNet.AddPerceiverEncoder(NumLatents, d_latent, Heads, Depth)`**,
a one-call **builder** (not a new leaf layer) that wires the **Perceiver**
latent bottleneck (Jaegle et al. 2021, *"Perceiver: General Perception with
Iterative Attention"*, arXiv:2103.03206; *Perceiver IO*, arXiv:2107.14795).

## The idea

A standard transformer encoder over an `N`-token sequence pays `O(N^2)`
attention **and** stacks every block directly on the `N` tokens, so going deep on
a long input gets expensive fast. The Perceiver decouples *depth* from *length*
with a small, **fixed-size learnable latent array** `Z` of `NumLatents` rows
(`NumLatents << N`):

1. **Cross-attention read** — the `NumLatents` latents cross-attend **once** over
   the `N` input tokens, absorbing the whole sequence into a fixed
   `(NumLatents, 1, d_latent)` summary. This is the *only* step that touches the
   input; its cost is **linear** in `N` (a single `NumLatents × N` attention map).
2. **Latent tower** — a stack of `Depth` self-attention + FFN blocks refines the
   latents, acting **only** over the `NumLatents` rows: `O(NumLatents^2)` per
   block, completely **independent of `N`**.

So you can pour a 1000-token (or 50k-pixel) input through a deep tower whose
weight count and per-block compute never grow with the input length.

## How the builder composes already-landed pieces

`AddPerceiverEncoder` adds no new class — it wires existing layers:

```
(InputSeqLen,1,d_model)
  -> [if d_model <> d_latent] TNNetPointwiseConvLinear(d_latent)   # token-wise width projection
  -> AddAttentionPooling(NumLatents, Heads)                        # latent CROSS-ATTENTION read
  -> AddTransformerEncoderBlock(Heads, d_ff) × Depth              # latent SELF-attention tower
(NumLatents,1,d_latent)
```

The key reuse: **`TNNetAttentionPooling` (PMA from the Set Transformer) IS the
latent cross-attention read** — its learnable seed bank is exactly the Perceiver
latent array `Z`: `NumLatents` learnable seeds cross-attend over the inputs and
collapse them to a fixed `(NumLatents,1,d_latent)` output, at a cost linear in
the input length.

### Distinct from the Set-Transformer builders

This is genuinely the **third mode**, not a near-duplicate:

| builder | output length | depth over latents? |
| --- | --- | --- |
| `AddInducedSetAttention` (ISAB) | **= input length** (projects back) | no |
| `AddAttentionPooling` (PMA) | `k` seeds | no (single pool step) |
| **`AddPerceiverEncoder`** | **`NumLatents`** | **yes (`Depth`-block tower)** |

## The demo

A tiny pure-CPU synthetic task makes the length-independence visible. Each sample
is a deliberately **long 256-token** `(SEQLEN,1,DMODEL)` tensor: a class-dependent
signal is planted into a few informative channels at positions **spread across
the whole sequence**, buried in Gaussian noise. The classifier is

```
Input(256,1,8)
  -> AddPerceiverEncoder(NumLatents=16, d_latent=16, Heads=4, Depth=2)   # -> (16,1,16)
  -> FullConnectLinear(NCLASS) -> SoftMax
```

### Headline payoff

Before training the demo builds the **same** net on `SEQLEN` and `2*SEQLEN` and
prints the trainable-weight count of each — they are **identical**:

```
Trainable weights @ SEQLEN=256   : 9088
Trainable weights @ SEQLEN=512   : 9088
  -> DOUBLING the input length added ZERO weights: the cost lives in the latent tower, not the input.
```

Then it trains on the long input and the accuracy climbs from chance (~25%) to
**100%**:

```
Test accuracy BEFORE training: 23.3%  (chance = 25.0%)
  epoch    0  test-acc= 50.0%
  epoch   50  test-acc= 96.3%
  ...
  epoch  399  test-acc=100.0%
Test accuracy AFTER training: 100.0%
```

Pure CPU, tiny dimensions and batches — runs in **≈55 s** on 2 cores with modest
memory. No binaries are committed.

## Building and running

```sh
lazbuild --bm=Release Perceiver.lpi
stdbuf -oL ../../bin/x86_64-linux/bin/Perceiver
```

(Use `stdbuf -oL` so the per-epoch lines are not lost to block buffering.)

## Follow-ups (not in v1)

- **Perceiver IO output cross-attention**: a final query array of shape
  `(NumOutputs,1,d)` reading the refined latents (decoupling the *output* shape
  too). A natural next builder.

---

*Coded by Claude (AI).*
