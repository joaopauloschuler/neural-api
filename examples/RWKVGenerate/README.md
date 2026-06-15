# RWKVGenerate — end-to-end flat-memory RWKV **block** decoding

[`examples/RWKVDecode`](../RWKVDecode) showed constant-memory decoding of the bare
RWKV-4 recurrence leaf `TNNetWKV`. The missing piece for driving a real RWKV
model token-by-token was the **other** stateful layer in an RWKV block:
`TNNetTokenShift`, the per-channel time-shift that mixes `x_t` with `x_{t-1}`.

This example decodes a **full** RWKV language-model net (token embedding → two
`AddRWKVBlock` stacks → final norm → vocab logits) one token at a time, with
**every** stateful layer advancing in lockstep.

## The incremental decode API on `TNNetTokenShift`

`TNNetTokenShift` now exposes the **same** stateful, inference-only decode path
as `TNNetWKV` / `TNNetSelectiveSSM`:

| method | effect |
|---|---|
| `BeginIncrementalDecode` | switch into stateful mode; state starts fresh (`x_{-1}=0`) |
| `Compute` (one-token input) | advance **one token in O(1)**; resumes from / persists the previous token |
| `ResetState` / `ResetCache` | start a fresh sequence (zero the carried previous token) |
| `EndIncrementalDecode` | return to the normal full-sequence forward |
| `CaptureState` / `RestoreState` | snapshot / restore the carried previous token to fork a session |

The single-step output is the **exact algebraic equivalent** of one step of the
full-sequence shift `y[t,c] = mix[c]·x[t,c] + (1-mix[c])·x[t-1,c]` (with
`x[-1,c]=0`); it just resumes `x_{t-1}` from a persisted **Depth-long** buffer
instead of re-scanning the sequence. That buffer is the **entire** carried state
— independent of how many tokens have been decoded.

## The net-wide driver

`TNNet` now broadcasts the decode mode across a whole net:

| method | effect |
|---|---|
| `BeginIncrementalDecode` | loop the layers; switch every recurrent leaf (`TNNetTokenShift`, `TNNetWKV`, `TNNetSelectiveSSM`, `TNNetDiagonalSSM`) onto its O(1)-per-step path. Returns the count switched on |
| `ResetIncrementalDecode` | fresh sequence on every recurrent leaf |
| `EndIncrementalDecode` | switch all of them back to the full-sequence forward |

Stateless per-token layers (pointwise projections, sums, concats, activations,
norms) need no state, so a whole `AddRWKVBlock` decodes token-by-token once its
`TokenShift`s and `WKV` are switched on. (Attention KV-cache layers take a
`MaxContext` budget and stay on the `TNNetStreamingDecoder` path; this driver is
for the zero-arg recurrent leaves.)

## What the demo prints

**(A) Exact equivalence.** A 20-token sequence is run all-at-once through the
full-sequence forward, then decoded token-by-token with
`TNNet.BeginIncrementalDecode`. The driver reports **8** recurrent leaves
switched on (2 blocks × (1 time-mix `TokenShift` + 1 `WKV` + 2 channel-mix
`TokenShift`)). The per-step next-token logits match **bit-for-bit**
(`max abs error = 0.0`) and the greedy argmax token agrees at **every** step
(`0 / 20` mismatches).

**(B) Flat work / flat memory.** After warming to position 64, four chunks of
incremental steps are timed. The `us/step` **does not trend up with position** —
deep in the sequence costs the same as the start, because the carried state is
fixed-size. A transformer KV cache would grow per-step work ~linearly with
position.

## Build & run

Offline and self-contained (a tiny random-init RWKV net; no network, no
checkpoint). Run under a memory cap:

```
fpc -Mobjfpc -Sh -O2 -Fu../../neural RWKVGenerate.lpr
ulimit -v 3000000 ./RWKVGenerate
```

Runs in about a second on CPU.
