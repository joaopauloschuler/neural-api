# MinimalRNN — minGRU / minLSTM vs a memoryless MLP on selective-copy recall

Demonstrates the two **minimal, fully-parallelizable recurrent cells** of
Feng, Tung, Hassani, Hamarneh & Ravanbakhsh 2024 (*"Were RNNs all we needed?"*,
[arXiv:2410.01201](https://arxiv.org/abs/2410.01201)):

* **`TNNetMinGRU`**  — `h_t = (1 - z_t)·h_{t-1} + z_t·h̃_t`
* **`TNNetMinLSTM`** — `h_t = f'_t·h_{t-1} + i'_t·h̃_t`, gates normalized `f/(f+i)`

Both cells drop the dependence of their gates on `h_{t-1}` — that is precisely
what makes them parallelizable. The example contrasts them head-to-head against a
**memoryless per-token MLP** that shares the same I/O contract but carries **no
state across time**, so it structurally cannot solve a task that requires
remembering an earlier token.

## The task — selective-copy recall

A sequence presents several value tokens; exactly **one** token is marked (a flag
channel = 1). A final **query** token (a separate flag) asks the network to
reproduce the marked token's value vector:

```
value t : [ value_vec | mark_flag in {0,1} | query_flag=0 ]
query   : [ 0...0      | 0                  | query_flag=1 ]
```

The target at the query position is the value vector of the marked token. Solving
it requires the cell to (a) detect the mark, (b) latch that value into its hidden
state, (c) hold it across the remaining tokens, and (d) emit it at the query. The
minimal cells' update gate can open on the marked token and stay near-closed
afterwards — a content-dependent latch. A per-token MLP, with no cross-time
memory, can only ever look at the all-zero query token and is stuck at chance
(`1 / cNumVals`).

Each sample is built by `MakeSample`: `cNumToks` value tokens carry random value
ids drawn from a fixed `ValueBank`, one random position is marked, and the trailing
query position holds the desired marked-value target.

## The three arms

All three arms share the **same** `(SeqLen, 1, cInDim)` I/O contract and differ
only in the middle:

| arm | architecture |
|-----|--------------|
| minGRU  | `TNNetInput` → `TNNetPointwiseConvLinear(d)` → `TNNetMinGRU` → `TNNetPointwiseConvLinear(value_dim)` |
| minLSTM | `TNNetInput` → `TNNetPointwiseConvLinear(d)` → `TNNetMinLSTM` → `TNNetPointwiseConvLinear(value_dim)` |
| MLP (memoryless) | `TNNetInput` → `TNNetPointwiseConvReLU(d)` → `TNNetPointwiseConvReLU(d)` → `TNNetPointwiseConvLinear(value_dim)` |

The `1×1` projections (`TNNetPointwiseConvLinear` / `TNNetPointwiseConvReLU`)
process each time position independently, so the **only** source of cross-time
memory in the recurrent arms is the cell itself.

Dimensions: `cNumVals=6` values, `cValueDim=4`, `cNumToks=4` value tokens
(`cSeqLen=5` with the query), `cModelDim=24`, `cInDim=6`. All arms are trained for
`cTrainSteps=40000` per-sample steps on the **same** replayed RNG stream
(`RandSeed:=999` before each), then evaluated over `cEvalSeqs=400` held-out
sequences (`RandSeed:=7`).

**Note:** BPTT through the unrolled recurrence is momentum-sensitive at this tiny
scale, so all arms train with `SetLearningRate(0.005, 0.0)` (plain SGD, momentum
0); higher momentum destabilizes the carried hidden-state gradient.

Evaluation (`Evaluate`) reports two numbers per arm: the **recall MSE** at the
query position and the **exact-recall accuracy** — the fraction of sequences whose
query output is nearest-neighbour-decoded (over the value bank) to the true value.

## Running

```
cd examples/MinimalRNN
fpc -O3 -Mobjfpc -Sh -Fu../../neural MinimalRNN.lpr
./MinimalRNN
```

(or open `MinimalRNN.lpi` in Lazarus). Pure CPU, tiny dims, finishes in a couple
of seconds on 2 cores.

## What to expect

The program prints the configuration, the per-arm weight counts, a training
progress line, and the held-out eval table:

```
=== MinimalRNN: selective-copy recall, minGRU / minLSTM vs memoryless MLP ===
...
eval over 400 held-out selective-copy sequences:
  minGRU         : recall MSE = ...   exact-recall acc = ...%
  minLSTM        : recall MSE = ...   exact-recall acc = ...%
  MLP (memoryless): recall MSE = ...   exact-recall acc = ...%
```

Trained on identical data for identical steps, both minimal recurrent arms reach
near-100% exact-recall accuracy while the memoryless MLP stays near chance
(`100 / cNumVals`%). The example asserts its own headline: if both recurrent arms
beat the MLP by more than 0.3 accuracy it prints

```
OK: both minimal recurrent cells solve selective-copy recall, while the memoryless MLP stays near chance.
```

otherwise a `WARNING` line. Exact numbers are seed-dependent; the contrast is the
point.

Coded by Claude (AI).
