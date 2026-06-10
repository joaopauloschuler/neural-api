# IncrementalDecode: KV-cache decoding with TNNetScaledDotProductAttention (+ TNNetDiagonalSSM persisted state)

Autoregressive generation with a vanilla attention stack re-encodes the
**entire prefix** to sample every next token: the step at prefix length `t`
runs a full `t x t` causal attention pass, so generating `N` tokens costs
`O(N^3)` attention work in total. The KV cache removes the re-encode: the
layer keeps every past token's K and V slices in a persistent, preallocated
buffer and a decode step feeds **only the new token** — its query attends over
the cached keys/values `[0..t]`, one row of attention instead of a full pass.
Attending over exactly the cached prefix *is* the causal mask.

## API (on `TNNetScaledDotProductAttention`)

```pascal
SDPA.BeginIncrementalDecode(MaxContext); // preallocate K/V caches, start empty
// ... feed tokens one at a time (or a multi-token prompt prefill):
//     every forward appends K,V to the cache and attends over [0..t]
SDPA.ResetCache();                        // start a fresh sequence, keep buffers
SDPA.EndIncrementalDecode();              // back to the normal training forward
// Introspection: SDPA.CacheEnabled, SDPA.CacheLength, SDPA.MaxContext
```

* The cache is **inference only**: with the cache disabled (the default) the
  forward/backward path is bit-for-bit unchanged. Do not call `Backpropagate`
  in cached mode.
* The K and V caches are preallocated **once** at `MaxContext x d_k` each by
  `BeginIncrementalDecode`; decode steps never reallocate, and exceeding
  `MaxContext` is a hard error (call `ResetCache` or begin with a larger
  context).
* **Positional contract**: the SDPA layer applies no positional encoding
  itself. If your stack adds positions before the QKV projection
  (`TNNetAddPositionalEmbedding`, RoPE, ...), encode each streamed token with
  its **absolute** position `t = SDPA.CacheLength` (read before the step), not
  with position 0 derived from the length-1 input.

## What the demo does

1. **Faithfulness**: feeds the same random `Q|K|V` sequence through one full
   causal forward and token-at-a-time through the cached path, and checks the
   outputs match at every position to `< 1e-5` (they match exactly here).
2. **Timing**: measures per-token wall-clock at growing prefix lengths for
   both arms (`d_k = 64`).

Sample run (2-core CPU):

```
  prefix t | full re-encode (ms/token) | cached step (ms/token) | speedup
        64 |                     0.6946 |                 0.0330 |   21.1x
       128 |                     1.2854 |                 0.0239 |   53.7x
       256 |                     2.0913 |                 0.0074 |  281.2x
       512 |                    16.3052 |                 0.0111 | 1469.8x
      1024 |                    81.2312 |                 0.0538 | 1509.5x
```

The full re-encode column grows ~quadratically with the prefix; the cached
column stays nearly flat, so the speedup keeps widening with `t`.

## TNNetDiagonalSSM: O(1)-per-step incremental decode

A linear recurrence is the easy case: the **entire past is summarised by one
Depth-long state vector `h`**, so incremental decode needs no cache and no
preallocation budget at all — just carry `h` across single-token forwards.

```pascal
SSM.BeginIncrementalDecode();  // stateful mode, h := 0 (no MaxContext needed)
// ... feed tokens one at a time (or a multi-token prompt prefill):
//     every forward resumes from the persisted h and updates it
SSM.ResetState();              // start a fresh sequence (h := 0)
SSM.EndIncrementalDecode();    // back to the normal training forward
// Introspection: SSM.DecodeEnabled, SSM.DecodeSteps
```

Same contract as the attention KV cache: inference only (calling
`Backpropagate` in incremental mode is an error), and with the mode disabled
(the default) the training forward/backward is bit-for-bit unchanged. The
demo runs the same faithfulness + timing protocol (`Depth = 192`):

```
  prefix t | full re-encode (ms/token) | incremental (ms/token) | speedup
        64 |                     0.0298 |                 0.0082 |    3.6x
       128 |                     0.0353 |                 0.0334 |    1.1x
       256 |                     0.0665 |                 0.0081 |    8.2x
       512 |                     0.1444 |                 0.0082 |   17.7x
      1024 |                     0.6997 |                 0.0032 |  216.4x
```

For an SSM the full re-encode column grows **linearly** with the prefix (each
step re-sweeps the `t`-token recurrence); the incremental column is **flat by
construction** — a step costs `O(Depth)` regardless of `t`.

## Build

```
lazbuild IncrementalDecode.lpi
../../bin/x86_64-linux/bin/IncrementalDecode
```

Pure CPU, no training, finishes in a few seconds.
