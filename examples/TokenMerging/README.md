# Token Merging (ToMe)

A demo of `TNNetTokenMerging`, a **weightless** (zero trainable parameters)
sequence-shortening layer for the transformer stack, after Bolya et al. 2023,
*"Token Merging: Your ViT But Faster"*, ICLR 2023
([arXiv:2210.09461](https://arxiv.org/abs/2210.09461)).

## What ToMe does

Self-attention costs `O(N^2)` in the token count `N`. Many tokens are redundant,
so instead of attending over all of them ToMe **fuses** the most-similar ones.
Operating on a `(SeqLen, 1, Depth)` token tensor it reduces it to a **static**
`(SeqLen - R, 1, Depth)` tensor by *bipartite soft matching*:

1. Split tokens into alternating sets A (odd indices) and B (even indices).
2. Compute each A-token's cosine similarity to its most-similar B-token.
3. Pick the top-`R` highest-similarity A→B edges.
4. **Merge** each chosen A-token into its B partner by (size-weighted) averaging;
   unmatched tokens pass through unchanged.
5. Carry a per-output-token *size* count so a B partner that absorbed `m`
   A-tokens averages over `m+1` tokens (the paper's proportional bookkeeping).

There are **no trainable parameters** and the output length is fixed given `R`,
so it drops straight into an existing encoder stack between blocks.

This is **distinct** from other sequence reducers in the library:
`TNNetAttentionPooling` / `AddPerceiverEncoder` *learn* fixed query slots;
`AddMixtureOfDepths` *routes/skips* tokens; pooling layers *collapse* the whole
axis. ToMe instead **fuses redundant tokens, weightlessly**.

### Backward / serialization

Each merge is a plain weighted average, so the merged-output gradient routes back
to every contributing token scaled by its size weight `1/size`. The top-`R` edge
selection is the non-differentiable boundary and is **frozen per forward pass**
(like MaxPool's argmax). `R` and the A/B parity round-trip through `FStruct`.

## This example

A tiny pure-CPU synthetic classification task. Each `(64, 1, 16)` sample carries a
per-class prototype signal planted (with noise) across the sequence. One shared
encoder front-end is compared in two configurations:

```
BASELINE : Input -> Embed -> EncBlock x2 (over all 64 tokens) -> AvgPool -> Linear -> SoftMax
ToMe     : Input -> Embed -> EncBlock
                -> TNNetTokenMerging(R=32)   <-- drops half the tokens
                -> EncBlock (over 32 tokens) -> AvgPool -> Linear -> SoftMax
```

Both are trained for the same budget; the example prints, for each, the token
count the deep block sees, the final test accuracy, and the wall-clock time.

### Sample run

```
BASELINE  : deep encoder block attends over 64 tokens, 13632 weights
ToMe      : deep encoder block attends over 32 tokens, 11584 weights (TNNetTokenMerging adds ZERO weights)

BASELINE  : test-acc= 66.3%   deep-block-tokens=64   wall=  61.4s
ToMe      : test-acc= 74.5%   deep-block-tokens=32   wall=  37.7s

Headline: ToMe dropped 32 of 64 tokens (50%) before the deep block
          while keeping accuracy (66.3% -> 74.5%), with no extra parameters.
```

ToMe halves the token count the deep block attends over, runs noticeably faster,
and keeps (here, slightly improves) accuracy — with zero added parameters.

## Build & run

```
lazbuild TokenMerging.lpi
../../bin/x86_64-linux/bin/TokenMerging
```

Pure CPU, tiny dimensions and batches — runs in well under 5 minutes on 2 cores
with modest memory. No binaries are committed.
