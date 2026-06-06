# MemoryFootprintReport

Demonstrates `TNNet.MemoryFootprintReport`: a static, per-layer estimator of
activation / parameter / gradient memory for a `TNNet`. Pure structure
inspection — no probe batch, no forward pass.

## What it does

1. Builds a tiny MLP (`64 -> 256 -> 256 -> 10 + softmax`) and prints its
   per-layer memory table.
2. Builds a tiny CIFAR-style ConvNet (3 conv+pool stages, global avg
   channel, softmax head) and prints the same table.
3. Builds a tiny attention stack (`SeqLen=16`, `Emb=32`, `Dk=16`, two
   stacked `TNNetScaledDotProductAttention` blocks) and prints the same
   table.
4. Prints a one-line side-by-side bottleneck comparison so reviewers can
   eyeball how the shape of the memory bottleneck shifts across model
   families.

The default optimizer assumption is `adam` (`3x params`) and the default
budget is `2048 MiB`. Both are configurable via the call signature
`MemoryFootprintReport(NN, OptimizerKind, BudgetMiB)`.

## What is reported

For every layer:

| Column | Meaning |
| --- | --- |
| `ActMiB` | `Output.SizeX * SizeY * Depth * sizeof(TNeuralFloat)` in MiB |
| `ParamMiB` | Trainable weights + biases in MiB (zero for parameter-free layers) |
| `GradMiB` | Mirrors `ActMiB` — transient, recoverable via checkpointing |

Followed by:

- **Peak forward residency** (sum of activations kept alive for backward).
- **Parameters + optimizer-state baseline** scaled by `OptimizerKind`:
  `sgd` = `1x` params, `momentum` = `2x` params, `adam` = `3x` params
  (master copy + 1st moment + 2nd moment).
- **10-bin ASCII histogram** of per-layer activation MiB so memory-hot
  layers jump out at a glance.
- **Flag lists**: "activation-heavy" layers (`>10%` of activation total —
  natural gradient-checkpointing candidates) and "parameter-heavy"
  layers (`>10%` of parameter total — natural LoRA / quantization
  candidates).
- **Would-fit-in verdict** against `BudgetMiB` for `forward-only`,
  `train-SGD`, `train-Adam`, plus the user-requested optimizer.

## Bottleneck shape across families

Look at the side-by-side comparison at the end. The classic pattern is:

- **MLP**: parameter-heavy (dense `In x Out` matrices), activation-light.
  Natural LoRA / quantization target.
- **ConvNet**: activation-heavy in the early high-resolution stages,
  parameter-light. Natural gradient-checkpointing target.
- **Attention stack**: both-heavy. Activation grows with `SeqLen * Emb`
  and the attention map itself is `SeqLen^2` per layer; parameters grow
  with the Q|K|V projections.

`MemoryFootprintReport` is the natural input for any future checkpointing
or mixed-precision work — you need to know which layers cost what before
you can decide where to trade compute for memory.

## Running

```
cd examples/MemoryFootprintReport
lazbuild MemoryFootprintReport.lpi
../../bin/<arch>/bin/MemoryFootprintReport
```

Runs in under a second on CPU.
