# MLP-Mixer block (all-MLP, attention-free sequence mixer)

A tiny all-MLP token-classification demo of **`TNNet.AddMLPMixerBlock`**, the
attention-free MLP-Mixer block.

Reference: Tolstikhin et al. 2021, *MLP-Mixer: An all-MLP Architecture for
Vision* (arXiv:2105.01601).

## The block

MLP-Mixer replaces self-attention over a `(Tokens, 1, Channels)` sequence with
**two pre-LayerNorm residual MLPs**, applied along different axes:

1. **Token-mixing MLP** — mixes information *across token positions*, sharing the
   same MLP over every channel. Implemented by transposing the token and channel
   axes (`TNNetTransposeXD`: `(T,1,C) -> (C,1,T)`), running a pointwise 2-layer
   MLP over the new Depth (= Tokens) axis, then transposing back:

   ```
   x -> LayerNorm -> TransposeXD -> Linear(tokensHidden) -> act -> Linear(T) -> TransposeXD -> + x
   ```

2. **Channel-mixing MLP** — a standard per-token pointwise FFN over the channels:

   ```
   x -> LayerNorm -> Linear(channelsHidden) -> act -> Linear(C) -> + x
   ```

Both sub-blocks are shape-preserving and wrapped in `AddPreNormResidual`. All
projections are pointwise 1×1 convolutions (`TNNetPointwiseConvLinear`) so the
sequence axis is never flattened (a `FullConnect` would mix the whole sequence).

```pascal
NN.AddMLPMixerBlock(TokensHidden, ChannelsHidden, TNNetReLU);
```

`Tokens` (= `SizeX`) and `Channels` (= `Depth`) are read from the current last
layer; the block does not change the shape, so blocks stack directly.

## The task (which-half classification — needs token mixing)

Each example is a length-`SEQ_LEN` sequence of `CHANNELS`-channel tokens filled
with small noise. A single spike of magnitude `SPIKE` is planted on channel 0 at
one random position. The label is **0 if the spike sits in the first half** of
the sequence and **1 if it sits in the second half**.

Deciding the class requires comparing information *across* token positions —
exactly what the token-mixing MLP provides. A purely channel-wise (per-token)
network cannot solve it.

## Network

```
Input(SEQ_LEN,1,CHANNELS)
  -> PointwiseConvLinear(CHANNELS)                 (token-wise embedding stem)
  -> AddMLPMixerBlock(TOKENS_HIDDEN, CHANNELS_HIDDEN, ReLU)  x NUM_BLOCKS
  -> LayerNorm
  -> AvgChannel                                    (mean-pool over tokens)
  -> FullConnectLinear(NUM_CLASSES) -> SoftMax
```

## Running

```
lazbuild examples/MLPMixer/MLPMixer.lpi
./bin/x86_64-linux/bin/MLPMixer
```

A 2-block Mixer stack (≈2320 weights) converges to **100% train and test
accuracy** with a smoothly falling cross-entropy loss in well under a minute on
CPU (≈6 s on 2 cores). Example trace:

```
 epoch    train loss   train acc    test acc
     0     0.69268    0.59750    0.56000
    10     0.51904    1.00000    1.00000
    30     0.19428    1.00000    1.00000
    59     0.10451    1.00000    1.00000

FINAL  train acc =  1.00000   test acc =  1.00000
```

This is a small CPU toy that demonstrates the builder rather than chasing SOTA.
