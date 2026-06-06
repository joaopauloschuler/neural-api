# AttentionCopyTask

Smallest possible end-to-end attention training demo. A single
`TNNetScaledDotProductAttention` head learns to **copy** a 16-token
input sequence (vocabulary of 8) to its output. No external dataset,
no multi-head wrapper, no transformer block — just the raw SDPA layer
wired up with an embedding and a per-position softmax readout.

## What it shows

The model is the minimal stack that can solve a sequence-to-sequence
identity:

```
TNNetInput(SeqLen, 1, 1)                  # token IDs along X
  -> TNNetEmbedding(Vocab, d_model)       # learned token vectors
  -> TNNetSinusoidalPositionalEmbedding   # parameter-free positions
  -> TNNetPointwiseConvLinear(3*d_k)      # pack Q|K|V on depth
  -> TNNetScaledDotProductAttention(d_k)  # single non-causal head
  -> TNNetPointwiseConvLinear(Vocab)      # per-position logits
  -> TNNetPointwiseSoftMax(1)             # softmax across depth
```

The optimal solution is for each query position to attend almost
entirely to the matching key position (an identity-like attention
matrix) and copy the embedded token through to the readout.

## Build & run

```
lazbuild AttentionCopyTask.lpi
../../bin/x86_64-linux/bin/AttentionCopyTask
```

Pure CPU, well under a minute (~6 seconds on a single thread).

## Expected output sketch

```
Training for 400 steps of batch size 32...
  step    1 /  400   mean-CE=2.13992   elapsed=0.0s
  step   25 /  400   mean-CE=0.58127   elapsed=0.4s
  step   50 /  400   mean-CE=0.06702   elapsed=0.7s
  ...
  step  400 /  400   mean-CE=0.00020   elapsed=5.7s

Evaluation on fresh random sequences:
  INPUT     : 1 4 1 7 1 5 0 1 6 1 0 6 3 5 5 3
  PREDICTED : 1 4 1 7 1 5 0 1 6 1 0 6 3 5 5 3
  ...
Per-token accuracy on 5 held-out probes: 80 / 80 = 100.00%
```
