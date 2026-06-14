# RWKVDecode — flat-memory recurrent decoding with `TNNetWKV`

Constant-memory autoregressive decoding is the headline of RWKV (and of the
`BuildRWKVFromSafeTensors` importer): unlike a transformer — whose KV cache and
per-step attention work **grow with context length** — an RWKV model summarises
the entire past in a **fixed-size recurrent state**. This example exercises that
on the core RWKV-4 time-mixing recurrence, `TNNetWKV`, through its new
incremental decode API.

## The incremental decode API

`TNNetWKV` now exposes a stateful, inference-only decode path that mirrors
`TNNetDiagonalSSM` and the SDPA KV-cache, so a decoder can drive every recurrent
layer type uniformly:

| method | effect |
|---|---|
| `BeginIncrementalDecode` | switch the layer into stateful mode; state starts fresh (`A=B=0, Q=-∞`) |
| `Compute` (one-token input) | advance **one token in O(1)**; resumes from and persists the running state |
| `ResetState` / `ResetCache` | start a fresh sequence (zero the state) |
| `EndIncrementalDecode` | return to the normal full-sequence forward |
| `CaptureState` / `RestoreState` | snapshot / restore the `(A,B,Q)` triple to fork a session |

The single-step update is the **exact algebraic equivalent** of one step of the
ordinary full-sequence scan (the same numerically-stable RWKV-v4
running-max exponent normalisation). The only differences are that it resumes
from the persisted stabilised numerator/denominator state `(A, B, Q)` instead of
restarting at `(0, 0, -∞)`, and it does not write the BPTT accumulator cache
(decode is inference-only). The carried state is a fixed `3·C`-float triple —
**independent of how many tokens have been decoded**.

## What the demo prints

**(A) Exact equivalence.** A 24-token sequence is fed all-at-once through the
full-sequence `Compute()` (the parallel/prefill path) and then token-by-token
through the incremental path. The two outputs match **bit-for-bit**
(`max abs error = 0.0`, asserted `< 1e-5`).

**(B) Flat work / flat memory.** After warming the state to position 64, four
chunks of 128 incremental decode steps each are timed. The reported `us/step`
**does not trend up with position** — decoding deep in the sequence costs the
same as the start, because the state size is constant. A transformer KV cache
would show per-step work growing roughly linearly with the start position.

## Build & run

Offline and self-contained (a tiny random-init `TNNetWKV` stack; no network, no
checkpoint). Run under a memory cap:

```
fpc -Mobjfpc -Sh -O2 -Fu../../neural RWKVDecode.lpr
ulimit -v 3000000 ./RWKVDecode
```

Runs in about a second on CPU.

## Follow-up

This lands the **layer-level** incremental API + the exact-equivalence assert.
A full block-level decode through `TNNet.AddRWKVTimeMix` (which also needs the
surrounding `TNNetTokenShift` layers to carry their one-token shift state) and
its wiring into `TNNetStreamingDecoder` is a noted follow-up.
