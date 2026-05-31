# Layer authoring & numerical-gradient debugging

Developer notes for adding a new `TNNet*` layer to `neural/neuralnetwork.pas` and
keeping its gradient tests honest. Every claim here is derived from real layers
(`TNNetDropBlock`, `TNNetSpatialDropout2D`/`1D`, the trainable `TNNetGRN`) and from
`tests/TestNeuralNumerical.pas` / `tests/TestNeuralLayersExtra.pas`. Use the symbol
names below as anchors; line numbers drift.

## 1. Layer authoring checklist

Follow this in order. Steps marked **(both)** have TWO places that must change.

1. **Pick a base class.** Subclass the closest existing behaviour, not `TNNetLayer`
   directly. Regularizers that are identity at inference subclass
   `TNNetAddNoiseBase` (e.g. `TNNetDropBlock`, `TNNetSpatialDropout2D`,
   `TNNetGaussianNoise`); per-channel trainable affine layers subclass the
   channel/norm bases (e.g. `TNNetGRN`). The base gives you `FEnabled`,
   `FStruct[]`, `FFloatSt[]`, the backprop-call counter guard, and serialization
   plumbing for free.

2. **Declare the class in the interface** `type` block (top of the unit, near the
   sibling layers — `TNNetSpatialDropout2D` is declared right beside `1D` and
   `TNNetDropBlock`). Put a doc comment above it stating: the semantics, what is
   stored in `FStruct[]`/`FFloatSt[]`, the inference behaviour, and "No trainable
   params" if applicable. Declare only the methods you override
   (`Create; override` / `Destroy; override` / `Compute; override` /
   `Backpropagate; override`) plus any helper (`RefreshMask`) and read-only
   properties for tests (`KeepMask`, `FreezeMask`).

3. **Constructor: store every parameter so it round-trips.** Clamp inputs, then
   write **integer** params to `FStruct[i]` and **float** params to `FFloatSt[i]`.
   `TNNetDropBlock.Create` does both: `FStruct[0] := pBlockSize; FFloatSt[0] := pDropProb;`.
   `TNNetSpatialDropout2D.Create` stores only `FFloatSt[0] := pDropProb`. These
   arrays are exactly what `SaveStructureToString` serializes (see step 6), so the
   constructor argument list and the `FStruct`/`FFloatSt` indices must line up
   one-to-one. Allocate owned buffers here (`FKeepMask := TNNetVolume.Create(...)`)
   and set `FEnabled := true` when the layer is actually active.

4. **`Destroy` (if you own buffers).** Free every `TNNetVolume` you created, then
   `inherited Destroy()`. (`TNNetDropBlock.Destroy` frees `FKeepMask`.)

5. **Register in BOTH `CreateLayer` dispatch tables (both).** `TNNet.CreateLayer`
   (`function TNNet.CreateLayer(strData: string)`) has two parallel reconstruction
   paths and a layer name appears in each:
   - the FPC `case ClassNameStr of` table — e.g.
     `'TNNetDropBlock' : Result := TNNetDropBlock.Create(St[0], Ft[0]);`
   - the non-FPC `if S[0] = '...' then ... else` ladder — e.g.
     `if S[0] = 'TNNetDropBlock' then Result := TNNetDropBlock.Create(St[0], Ft[0]) else`

   Both must pass the SAME constructor args, reading ints from `St[]` and floats
   from `Ft[]` in the same index order the constructor stored them. Miss one and
   the layer loads on only one compiler path. (`TNNetSpatialDropout2D` →
   `.Create(Ft[0])` in both.)

6. **Verify the SaveToString / LoadFromString round-trip.** Serialization is
   automatic via `SaveStructureToString` (`ClassName:FStruct;...::FFloatSt;...`),
   but it only works if step 3 and step 5 agree: the constructor args must
   reconstruct the layer exactly from `FStruct`/`FFloatSt`. Do NOT serialize
   sampled state (masks, RNG draws) — only hyperparameters. The round-trip test
   (step 8) is what proves this.

7. **`Compute` / `Backpropagate` (and `SetPrevLayer` if shape depends on input).**
   - `Compute`: start from `FOutput.CopyNoChecks(FPrevLayer.FOutput)`, resize any
     mask buffer to match `FOutput` when shapes change, branch on
     `FEnabled and (param > 0)` for the active path, and otherwise be the identity.
     Capture any per-sample random state needed for backward (DropBlock stores it
     in `FKeepMask`).
   - `Backpropagate`: the mandatory prologue is
     `Inc(FBackPropCallCurrentCnt); if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit; TestBackPropCallCurrCnt();`
     then apply the SAME gate/mask captured in `Compute` to `FOutputError`, and
     end with `BackpropagateNoTest();`. Gradient must route through the identical
     element-wise factor used forward (DropBlock multiplies `FOutputError` by the
     stored `FKeepMask`, including its rescale).

8. **Add the MANDATORY numerical-gradient test** in
   `tests/TestNeuralNumerical.pas`: declare the method in the published section of
   `TTestNeuralNumerical`, implement central-difference vs analytic (pattern in
   §2), and use `RandSeed := 424242` at the top if your forward draws RNG. For
   stochastic layers, freeze/reseed the mask so the probe and the analytic pass
   see an identical gate (`TNNetDropBlock.FreezeMask`, or reseed-before-every-
   forward as `TestSpatialDropout2DGradientCheck` does).

9. **Add a smoke / round-trip test** in `tests/TestNeuralLayersExtra.pas`
   (`TestDropBlockSmokeAndRoundTrip` is the template): assert (1) inference is an
   identity passthrough, (2) the active path actually does its thing, and (3)
   `SaveToString -> LoadFromString` preserves `SaveStructureToString()` and, under
   a fixed `RandSeed`, reproduces the output bit-for-bit.

10. **Build and run the whole suite:** `bash tests/RunAll.sh`. Use the runner, not
    a hand-rolled `fpc` line (the latter can spuriously fail on `UTF8Process`).

## 2. Reading a numerical-gradient failure

The harness compares an analytic gradient against a **central difference**. The
canonical loop (see `TestDropBlockGradientCheck`, `TestSpatialDropout2DGradientCheck`)
is:

```
numericalGrad  := (lossPlus - lossMinus) / (2 * epsilon);  // f(x+eps), f(x-eps)
analyticalGrad := NN.Layers[0].OutputError.Raw[i];          // from one backward pass
assert  Abs(numericalGrad - analyticalGrad) < tolerance;
```

with `epsilon` typically `0.0001`–`0.001` (some layers `0.01`) and the default
`tolerance = 0.01`. When it fails, the **magnitude and behaviour** of
`Abs(num - ana)` tells you which of three things you are looking at:

- **(a) Genuine analytic-gradient bug.** Error is large — order O(1) relative to
  the gradient, or it stays large / grows as you shrink `epsilon`. The central
  difference is converging on the true derivative and the analytic value simply
  disagrees. Fix the math in `Backpropagate` (wrong sign, missing rescale factor,
  forgot to route through the captured mask, missing chain-rule term). DropBlock's
  whole point is that backward reuses the SAME `FKeepMask` including its rescale —
  dropping the rescale would show up here.

- **(b) Tolerance too tight.** Error is small and just barely over the threshold,
  and it is **stable** as you vary `epsilon` (does not blow up, does not vanish).
  This is single-precision finite-difference noise, not a bug. Prefer shrinking
  `epsilon` or fixing accumulation order before loosening the bound (see §3).

- **(c) Discontinuity / kink near the probe step.** Error is small almost
  everywhere but **spikes at specific input positions**. The `eps` step straddles
  a non-differentiable point: a ReLU/abs kink, a clamp boundary, a saturating
  activation's flat tail, or — for stochastic layers — a different dropout mask
  being sampled on the `+eps` vs `-eps` forward. The analytic gradient is fine;
  the central difference is averaging across a kink. Mitigate by choosing inputs
  that avoid the kink, by reseeding so the SAME mask is used on every forward
  (`ComputeLossSeeded` reseeds `RandSeed := Seed` before each `NN.Compute`), or by
  freezing the mask (`FreezeMask := true`).

**Single-precision floor.** `TNeuralFloat` is single precision (~7 decimal
digits). You cannot drive `epsilon` arbitrarily small: too small and
`lossPlus - lossMinus` loses all its significant digits to catastrophic
cancellation, making the numerical gradient *worse*. `TestDropBlockGradientCheck`
explicitly bumps `epsilon := 0.01` for exactly this reason — its rescale amplifies
the loss, so a tiny step would cancel. There is a sweet spot; both ends hurt.

## 3. Picking a tolerance for numerical-gradient tests

Default to `< 0.01` (`1e-2`) — that is what nearly every test in
`TestNeuralNumerical.pas` uses (GELU, Mish, LayerNorm, RMSNorm, GroupNorm,
InstanceNorm, the residual builders, SpatialDropout2D, DropBlock, …). It is the
right bound for single-precision central differences on a non-trivial layer.

**When `1e-2` is fine.** Clean element-wise activations and standard
norm/residual layers with a moderate `epsilon` (`0.0001`–`0.001`). Most of the
file.

**When to loosen (to `2e-2`) — sparingly, and only with a reason in the comment.**
Layers that *amplify* finite-difference noise: a `sqrt` plus a mean-ratio that
couples all channels (`TestGRNInputGradientCheck` uses `< 0.02` and says so:
"GRN couples all channel positions through sqrt and a mean ratio, which amplifies
single-precision finite-difference noise"), or index-shuffling that spreads error
across many outputs (`TestPixelShuffle...` uses `< 2e-2`). The looseness must be
justified by the layer's math, not by a test that "won't pass otherwise".

**When to TIGHTEN.** If the op is exactly linear or a clean element-wise map,
the finite difference is accurate and you can assert harder — exact-reconstruction
and identity-init checks already use `< 1e-4`/`< 1e-5`/`< 1e-6` (e.g. reversible
round-trip `< 1e-4`, LoRA zero-init "starts as identity" `< 1e-6`, serialization
round-trip outputs `< 1e-5`). A tight bound there is a real correctness signal.

**The repo convention (most important):** if a gradient test is marginal, the
right move is almost never to loosen the tolerance. In order of preference:

1. **Fix the analytic gradient** — a real O(1) miss is a bug, not a tolerance
   problem (§2a).
2. **Shrink (or right-size) `epsilon`** and reduce accumulation noise — but mind
   the single-precision floor; sometimes the fix is a *larger* `epsilon`
   (DropBlock's `0.01`) to escape cancellation.
3. **Reseed / freeze stochastic state** so the probe and analytic pass share an
   identical mask (`RandSeed := 424242`; `ComputeLossSeeded`; `FreezeMask`).

Only after those, and only with a one-line comment explaining the amplification,
loosen the bound — and never past `2e-2`.
