# HighwayDepth — does the Highway carry keep deep stacks trainable?

This example showcases **`TNNetHighway`**, the input-dependent learned-gate
**Highway layer** (Srivastava, Greff & Schmidhuber 2015, *"Training Very Deep
Networks"*):

```
y = T(x) ⊙ H(x) + (1 - T(x)) ⊙ x ,   T(x) = sigmoid(W_T·x + b_T) ,
                                     H(x) = tanh(W_H·x + b_H)
```

The **transform gate** `T(x)` is computed *from the input* on every forward pass,
and the `(1 - T(x)) ⊙ x` term is an explicit **identity carry**. This is the
input-dependent learned-gate ancestor of the ResNet skip connection, and is
distinct from the other gated-residual mechanisms in the library:

| layer | gate | identity carry |
|-------|------|----------------|
| `TNNetReZero`        | one learned **scalar**, input-independent      | yes |
| `AddGatedResidual`   | per-channel learned **constant**, input-indep. | yes |
| `TNNetGLU`/`SwiGLU`  | multiplicative gating                          | **no** |
| **`TNNetHighway`**   | per-channel **`sigmoid(W_T·x+b_T)`**, input-dependent | **yes** |

The gate biases `b_T` are initialised **negative** (the paper's trick, default
`-1.5`), so a fresh deep stack begins life near the identity (`T ≈ 0 ⇒ y ≈ x`):
gradients flow through the carry path and the stack stays trainable as depth grows.

## The experiment

The regression target is an **"identity + small residual"** map,
`target = x + 0.3·W2·tanh(W1·x)` — close to the identity, exactly the kind of
function a deep near-identity stack should represent easily *if* it can carry `x`
through its layers. At each depth in `{2, 5, 10, 20, 40}` we train, head to head,
two stacks of shape-preserving width-12 blocks followed by the **same** linear
read-out:

| model | block | identity carry |
|-------|-------|----------------|
| **plain**   | `TNNetFullConnect(width)` (tanh) | none — must re-learn identity every layer |
| **highway** | `TNNetHighway(width)`            | gated `(1-T)·x` carry |

We chart **final test loss vs depth** (ASCII bars) and, for the Highway models,
report the **mean learned gate `T` per layer**.

## Running

```
cd examples/HighwayDepth
fpc -Mobjfpc -Sh -O3 -Fu../../neural -dRelease -dAVX2 HighwayDepth.lpr
./HighwayDepth
```

(or open `HighwayDepth.lpi` in Lazarus). Pure CPU, single thread, finishes in
about 40 seconds — comfortably under the 5-minute budget. The compiled binary is
git-ignored (see `.gitignore`).

## Representative output

```
depth | plain  tanh-MLP                                  test-MSE
    2 | ########################................        5.58E-003
    5 | ##############..........................        3.38E-002
   10 | ########................................        9.11E-002
   20 | ##......................................        2.46E-001
   40 | ........................................        3.79E-001
------------------------------------------------------------------------
depth | highway (gated carry)                            test-MSE
    2 | #############################...........        2.56E-003
    5 | #############################...........        2.83E-003
   10 | ############################............        3.12E-003
   20 | ###############.........................        2.95E-002
   40 | ........................................        3.94E-001
```

**Headline:** the plain tanh stack degrades **monotonically** with depth — its
test MSE climbs ~16× from depth 2 to depth 10 (5.6e-3 → 9.1e-3 → 9.1e-2) and
~44× by depth 20. The Highway stack stays essentially **flat** (2.6e-3 → 3.1e-3)
through depth 10 and is still ~8× better than plain at depth 20, because its
identity carry lets gradients reach the early layers. At the extreme depth 40
both stacks finally saturate near the residual's variance (a 40-layer dense-tanh
composition is pathological even with the carry at this width/step budget) — the
honest limit of a tiny CPU-only demo.

The **mean gate `T`** sits around `0.13–0.19` per layer: the layers keep most of
the carry (small `T`) and inject just the small residual they need, exactly the
near-identity regime the negative gate-bias init is designed to start in.

## Notes / follow-up

- The forward/backward passes route the output error through **both** branches
  and the gate's sigmoid; `x` appears in three places (carry, transform input,
  gate input) so all three input-error contributions are accumulated. This is
  verified by a numerical-gradient check in
  `tests/TestNeuralNumerical.pas` (`TestHighwayGradientCheck`).
- A natural follow-up is to pair `TNNetHighway` with the existing pre-/post-norm
  residual builders, or to push the depth-40 regime with a learning-rate schedule
  / warmup so the very deep Highway stack also converges.

Coded by Claude (AI).
