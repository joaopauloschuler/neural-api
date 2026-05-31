# TransformerDecoderBlock

Smallest possible end-to-end demo of `TNNet.AddTransformerDecoderBlock`:
a single transformer DECODER block wired to a (toy) encoder output and
trained on a tiny synthetic target so you can watch gradients flow
through the whole block in isolation. No tokenizer, no embedding, no
full seq2seq pipeline -- just the block plumbing.

## What it shows

A transformer decoder block stacks three residual sub-blocks on the
decoder stream:

1. **Causal multi-head self-attention** -- position `i` may only attend
   to positions `<= i`.
2. **Multi-head cross-attention** -- Query comes from the decoder
   stream, Key|Value come from the encoder output (the "encoder
   memory"). The two sequences may have different lengths.
3. **Token-wise SwiGLU feed-forward**.

Each sub-block is a residual (`x := x + SubBlock(LayerNorm(x))` in the
default PreNorm placement). Every projection is token-wise
(`TNNetPointwiseConvLinear`, a 1x1 conv) so the `(SeqLen, 1, d_model)`
sequence axis is preserved and the output shape matches the decoder
input -- blocks can be stacked.

The wiring uses two input branches:

```
DecoderInput(4, 1, 8)        # decoder stream, SeqLen=4
EncoderOutput(5, 1, 8)       # encoder memory, KVSeqLen=5 (may differ)
  -> AddTransformerDecoderBlock(d_model=8, Heads=2, d_ff=16, EncoderOutput)
     # output: (4, 1, 8)  -- on the decoder grid
```

The toy task regresses a fixed `cos`-shaped target from a fixed
(decoder input, encoder memory) pair. It is meaningless as a language
task -- it only exercises the plumbing and demonstrates finite
gradient flow as the loss falls toward zero.

## Build & run

```
lazbuild TransformerDecoderBlock.lpi
../../bin/x86_64-linux/bin/TransformerDecoderBlock
```

Or directly with fpc:

```
fpc -dRelease -dUseCThreads -O3 -Fu../../neural TransformerDecoderBlock.lpr
./TransformerDecoderBlock
```

Pure CPU, no external data, runs in well under a second. The run is
non-interactive (no trailing `ReadLn`).

## Expected output sketch

```
Decoder input shape : (4,1,8)
Encoder memory shape: (5,1,8)
Block output shape  : (4,1,8)

First-forward output sample:  0.35937,  1.18249, -0.04208

Toy training (forward + backward through the whole block):
  epoch    1  loss=27.413218
  epoch   50  loss= 0.084013
  epoch  100  loss= 0.001344
  epoch  150  loss= 0.000012
  epoch  200  loss= 0.000000

Done. The decreasing loss shows gradients flow end to end.
```

The output shape staying `(4,1,8)` (the decoder grid, not the
length-5 encoder grid) confirms cross-attention puts its result on the
query sequence, and the loss falling to zero confirms gradients flow
end to end through all three residual sub-blocks.
