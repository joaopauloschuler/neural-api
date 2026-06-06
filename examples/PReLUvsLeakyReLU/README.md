# PReLU vs LeakyReLU vs ReLU Bake-Off

Three-config activation comparison on the toy hypotenuse regression task
(`y = sqrt(X^2 + Y^2)`):

- `TNNetReLU`       baseline, no negative leak.
- `TNNetLeakyReLU`  fixed slope `0.01` on the negative half.
- `TNNetPReLU`      single learnable scalar slope shared across all
  elements (He et al. 2015, init alpha = 0.25). This is the
  channel-shared / "PReLU-shared" variant; for one alpha per channel
  see `TNNetPReLUChannel`.

Same architecture (`2 -> 32 (activation) -> 1`), same `RandSeed := 42`
before each variant's data generation and fit, so all three see
identical inputs and identical weight init. The only thing that
changes is the activation layer (and PReLU's one extra trainable
scalar).

## Build & run

```
cd examples/PReLUvsLeakyReLU
lazbuild PReLUvsLeakyReLU.lpi --build-mode=Default
../../bin/x86_64-linux/bin/PReLUvsLeakyReLU
```

Finishes in roughly two minutes on CPU.

## What it shows

- Final validation MSE on the original target scale (~0..141).
- Epochs-to-converge under threshold `5.0` (validation MSE).
- Trainable parameter count and delta vs ReLU, so the +1 learnable
  scalar from PReLU is visible.

## Expected output sketch

Real fragment from a recent run:

```
=== Results (CSV) ===
activation,final_val_loss,epochs_to_converge,total_epochs,trainable_weights,weight_delta_vs_relu
TNNetReLU,0.7779,24,150,96,+0
TNNetLeakyReLU,0.7593,25,150,96,+0
TNNetPReLU,0.2895,6,150,97,+1

Total wall time: 118.18 s
```

PReLU's learnable slope converges far faster (epoch ~6 vs ~24) and
reaches a lower final loss at the cost of exactly one extra trainable
parameter. ReLU and LeakyReLU end up close because on inputs already
in `[0, 1]` only a few hidden pre-activations go negative, so the
fixed `0.01` leak rarely fires; PReLU is free to learn a more useful
slope.
