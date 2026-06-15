# MambaDecode — flat-memory recurrent decoding with `TNNetSelectiveSSM`

Constant-memory autoregressive decoding is the headline of Mamba (and of the
`BuildMambaFromSafeTensors` importer): unlike a transformer — whose KV cache and
per-step attention work **grow with context length** — a Mamba model summarises
the entire past in a **fixed-size recurrent state**, the `[d_inner × d_state]`
hidden matrix `h`. This example exercises that on the core Mamba/S6 selective
scan, `TNNetSelectiveSSM`, through its new incremental decode API. It is the
direct sibling of the `examples/RWKVDecode` demo for the RWKV-4 recurrence.

## The incremental decode API

`TNNetSelectiveSSM` now exposes a stateful, inference-only decode path that
mirrors `TNNetWKV`, `TNNetDiagonalSSM` and the SDPA KV-cache, so a decoder can
drive every recurrent layer type uniformly:

| method | effect |
|---|---|
| `BeginIncrementalDecode` | switch the layer into stateful mode; state starts fresh (`h = 0`) |
| `Compute` (one-token input) | advance **one token in O(1)**; resumes from and persists the running state `h` |
| `ResetState` / `ResetCache` | start a fresh sequence (zero `h`) |
| `EndIncrementalDecode` | return to the normal full-sequence forward |
| `CaptureState` / `RestoreState` | snapshot / restore the `[d_inner × d_state]` `h` matrix to fork a session |

The single-step update is the **exact algebraic equivalent** of one step of the
ordinary full-sequence selective scan

```
delta_t = softplus(x_t·W_d + b_d)      (per-channel positive step, input-dependent)
h_t = exp(-delta_t·exp(A))(*)h_{t-1} + delta_t·B_t·x_t
y_t = C_t·h_t + D·x_t                   (B_t, C_t input-dependent — the "selective" part)
```

The only difference is that it resumes from the persisted running state `h`
instead of restarting at `h = 0`, and it does not write the BPTT caches (decode
is inference-only). All three layer modes (legacy `DState=1`, the multi-state
real-Mamba `DState>1`, and the Jamba inner-norm variant) are handled. The carried
state is a fixed `Depth·DState`-float matrix — **independent of how many tokens
have been decoded**.

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

Offline and self-contained (a tiny random-init `TNNetSelectiveSSM` leaf; no
network, no checkpoint). Run under a memory cap:

```
fpc -Mobjfpc -Sh -O2 -Fu../../neural MambaDecode.lpr
ulimit -v 3000000 ./MambaDecode
```

Runs in about a second on CPU.

## Scope / follow-up

This lands the **layer-level** incremental API on the SSM recurrence leaf + the
exact-equivalence assert. A full Mamba **block** places a causal depthwise
`conv1d` (+ SiLU gating) before this scan; decoding the whole block
token-by-token additionally needs the conv to carry its `(kernel-1)`-token ring
buffer, and the surrounding in/out projections to be driven one token at a time.
That block-level conv-state decode (and its wiring into `TNNetStreamingDecoder`)
is a noted follow-up — exactly mirroring the RWKV landing, which deferred its
own `TNNetTokenShift` shift-state block integration.
