# Optimization Guide

A running list of recommendations for keeping the neural-api codebase fast,
consistent, and maintainable. Each entry states the problem, the fix, and why
it matters. Recommendations are added incrementally as we discover them.

## 1. Prefer named size constants over `SizeOf(type)`

Replace repeated `SizeOf(<type>)` expressions with a predefined named size
constant. These constants live in `neural/neuralvolume.pas` and are visible to
every unit that `uses neuralvolume`:

```pascal
const
  csNeuralFloatSize  = SizeOf(TNeuralFloat);
  csNeuralFloat4Size = SizeOf(TNeuralFloat4);
  csLongintSize      = SizeOf(Longint);
  csIntegerSize      = SizeOf(Integer);
  csShortIntSize     = SizeOf(ShortInt);
```

**Do this:**

```pascal
Move(Src^, Dst^, Count * csNeuralFloatSize);
clSetKernelArg(k, 0, csLongintSize, @N);
Move(FKCacheCodes[(j + 1) * FDk], FKCacheCodes[j * FDk], FDk * csShortIntSize);
```

**Instead of:**

```pascal
Move(Src^, Dst^, Count * SizeOf(TNeuralFloat));
clSetKernelArg(k, 0, SizeOf(Longint), @N);
Move(FKCacheCodes[(j + 1) * FDk], FKCacheCodes[j * FDk], FDk * SizeOf(ShortInt));
```

**Why it matters**

- **Consistency** — one canonical name per element size across the whole
  codebase, so `Move`/allocation/stride/`clSetKernelArg` arithmetic all read the
  same way.
- **Single source of truth** — if a type's width ever changes, the intent stays
  expressed through one named constant rather than scattered `SizeOf` literals.
- **Readability** — `csNeuralFloatSize` signals *"one neural-float element"* at
  a glance, which is clearer than a bare `SizeOf` inside a byte-count
  expression.

**Adding a new size constant.** When you hit a repeated `SizeOf(<sometype>)`,
add a `cs<Type>Size = SizeOf(<sometype>);` line to the same `const` block in
`neuralvolume.pas` (keep the literal `SizeOf` in the definition itself), then
search/replace the literal across the codebase. Match case-insensitively — Pascal
is case-insensitive, so `SizeOf(integer)` and `SizeOf(Integer)` both occur.

**Caveats.**

- **Declaration order (Pascal).** A constant cannot be referenced before it is
  declared. The `cs*Size` definitions themselves, and any `type` declarations
  that precede the `const` block (e.g.
  `TNeuralFloatArr = array[0..Maxint div SizeOf(TNeuralFloat)] of ...`), must
  keep the literal `SizeOf`.
- **Only substitute matching types.** Replace `SizeOf(Longint)` with
  `csLongintSize`, not with `csIntegerSize` — on some targets `Integer` and
  `Longint` need not be the same width, so keep the constant tied to the exact
  type it names.

## 2. Never place expressions in a `for` bound; hoist them into a variable

Compute the loop bound once into a local variable, then loop over the variable.
Do not put an arithmetic (or property/function) expression directly in the
`to` clause of a `for`.

**Do this:**

```pascal
QLenM1 := QLen - 1;
for i := 0 to QLenM1 do
  ...
```

**Instead of:**

```pascal
for i := 0 to QLen - 1 do
  ...
```

**Why it matters**

- **No repeated evaluation** — the bound is evaluated once up front rather than
  potentially being recomputed. This is most important when the bound is a
  property getter or function call (e.g. `Volume.SizeX - 1`), where the
  expression can hide real work behind an innocent-looking loop header.
- **Readability** — a named bound like `QLenM1` documents the intent
  ("last query index") and keeps the loop header uncluttered.
- **Consistency** — matches the established style in the hot paths of this
  codebase, where loop bounds such as `SeqLenM1`, `DkM1`, and `CacheLenM1` are
  precomputed before the loop.

## 3. Access volume elements directly via `FData[]` in hot loops

`TNNetVolume`'s `[x, y, d]` accessor routes every read/write through
`Get`/`Set` → `GetRawPos`, which computes the flat offset:

```pascal
function TVolume.GetRawPos(x, y, d: integer): integer;
begin
  Result := ((FSizeX * y) + x) * FDepth + d;
end;
```

That is a multiply-add plus a method call on *every* element access. In hot
loops — especially forward/backprop inner loops — read and write the backing
array `FData` directly instead.

**Do this:**

```pascal
mu    := FMu.FData[c];
sigma := FSigma.FData[c];
```

**Instead of:**

```pascal
mu    := FMu[0, 0, c];
sigma := FSigma[0, 0, c];
```

**Why it matters**

- **Skips index arithmetic and the accessor call** — `FData[c]` is a plain
  array read; the `[0, 0, c]` form recomputes `((SizeX*0)+0)*Depth + c` and goes
  through `Get`/`GetRawPos` each time.

**When it is valid.** You must compute the flat index yourself, so this is only
safe when you know the volume's layout. The clean, common cases:

- **A `(1, 1, N)` volume** (per-channel vectors like `FMu`, `FSigma`): index
  `[0, 0, c]` collapses to `FData[c]` because `x = y = 0`.
- **General `[x, y, d]`**: the flat index is `((SizeX * y) + x) * Depth + d`.
  If you have already precomputed a raw offset (or a row pointer via
  `GetRawPtr`), prefer that.

**Compute the offset with `GetRawPos`, don't hand-roll it.** When the index is
non-trivial, let the volume compute its own flat offset via the public
`GetRawPos` (`inline` in Release) instead of re-deriving the arithmetic
yourself:

```pascal
base := FA.GetRawPos(t_, 0, c);   // = t_ * FDepth + c, but self-documenting
a  := FA.FData[base];
```

This is preferable to writing `t_ * FChannels + c` by hand: it reads the
volume's *actual* `FDepth` (so it stays correct if the layout assumption
changes), it documents intent, and — because it is inlined — it costs nothing in
Release. **Reuse one `base` across volumes that share a shape.** If several
volumes have identical dimensions, a single `GetRawPos` result indexes all of
them:

```pascal
// FA, H and HErr all share (T, 1, FChannels) — one offset addresses all three.
base := FA.GetRawPos(t_, 0, c);
a  := FA.FData[base];
hv := H.FData[base];
HErr.FData[base] := HErr.FData[base] + ...;
```

**The same offset can span different storage formats.** A slot's flat offset
depends only on the shape, so it addresses *any* array laid out with the same
per-slot stride — not just other `TNNetVolume.FData`. In the KV-cache eviction
shift, `FKCache`/`FVCache` (float rows in `.FData`) and
`FKCacheCodes`/`FVCacheCodes` (int8 code rows) all store `FDk` contiguous
elements per slot, so `j * FDk` indexes all four:

```pascal
// FKCache/FVCache are (MaxContext, 1, FDk) => GetRawPos(j,0,0) = j * FDk, which
// is also the int8 code offset. One carried offset (see #6) serves both formats.
Move(FKCache.FData[jP1Dk],   FKCache.FData[jDk],   RowBytesFP); // float rows
Move(FKCacheCodes[jP1Dk],    FKCacheCodes[jDk],    RowBytesI8); // int8 code rows
```

**`Move` and friends take untyped `var` parameters — pass the element, not a
pointer.** `Move(const source; var dest; count)` takes the *variables* and the
compiler passes their addresses; there is no `Addr`/`@` at the call site. So
`Move(V.FData[pos], ...)` is correct, and replacing `V.GetRawPtr(x,y,d)^` with
`V.FData[pos]` is a safe swap: `GetRawPtr` returns a pointer and the `^` already
dereferenced it back to a variable, which is exactly what `V.FData[pos]` is.
(Writing `Move(Addr(V.FData[pos])^, ...)` is equivalent but redundant; writing
`Move(Addr(V.FData[pos]), ...)` — no `^` — is a bug that copies the pointer
bytes.)

**Caveat.** `FData[]` does no bounds/shape checking and bypasses the accessor
entirely. Only use it where the shape is known and fixed (asserted at layer
setup), and keep the index expression obviously correct. For one-off or
non-hot-path accesses, the `[x, y, d]` form is clearer and the overhead is
irrelevant — reserve direct `FData` access for inner loops.

*Example:* `TNNetAttentiveStatsPooling.Backpropagate()` reads the per-channel
`FMu`/`FSigma` (both shaped `(1, 1, FChannels)`) once per channel in the outer
loop; switching those to `FData[c]` is exact and removes the accessor overhead.
Verified equivalent by the numerical gradient-check tests
(`TestAttentiveStatsPoolingFeatureGradientCheck` /
`...LogitGradientCheck`).

**Anti-pattern — do NOT follow this: inline `V.FData[V.GetRawPos(x,y,d)]` at a
single-use site.** The entire point of this rule is to *avoid recomputing the
flat offset*. Calling `GetRawPos` **inline**, with the same indices the accessor
would use, at a place where the result is used exactly once, achieves nothing —
it is the same work the accessor already does, just spelled out and made harder
to read.

**Do NOT do this:**

```pascal
FOutput.FData[FOutput.GetRawPos(t, 0, c)] := sum;   // pointless
err := FOutputError.FData[FOutputError.GetRawPos(X, Y, D)];   // pointless
```

**Because it is identical in cost to — and less readable than — the accessor:**

```pascal
FOutput[t, 0, c] := sum;
err := FOutputError[X, Y, D];
```

The `[x, y, d]` getter/setter *is* `FData[GetRawPos(x, y, d)]` internally, and
`GetRawPos` is `inline` in Release, so `V.FData[V.GetRawPos(x, y, d)]` and
`V[x, y, d]` compile to the same code. Writing the long form buys no speed and
loses clarity — and it silently bypasses bounds/shape checking for zero benefit.

**The rule only pays off when the offset is *reused*.** Direct `FData` is
worthwhile only in one of these shapes — otherwise keep the accessor:

- **Reused base:** compute `base := V.GetRawPos(x, y, 0)` *once*, then hit it
  more than once — `V.FData[base + k]`, `V.FData[base + HalfDepth]`, or across
  several same-shaped volumes (`A.FData[base]`, `B.FData[base]`).
- **Read-modify-write of one cell:** the accessor form
  `V[x, y, d] := V[x, y, d] * s` computes the offset *twice*; `pos := V.GetRawPos(x, y, d); V.FData[pos] := V.FData[pos] * s` computes it once. That is a real save (offset used twice via `pos`).
- **Running offset (strength reduction, #6):** `Inc(pos, stride)` /
  `idx := idx + OutDepth` down a loop, seeded once.
- **`(1,1,N)` / `(N,1,1)` collapse:** `FData[c]` / `FData[a]` directly, no
  `GetRawPos` at all.

**The test:** does the offset get used *at least twice* (or carried, or
collapsed)? If not — a lone read or a lone write with the same indices as the
accessor — leave it as `V[x, y, d]`. If you find yourself typing
`V.FData[V.GetRawPos(...)]`, either bind the offset to a local you reuse, or
revert to the accessor. This is exactly the over-application that had to be
reverted after a rule #3 sweep across the GLU-family Compute writes, Retention,
`DepthwiseConv1D`, `PixelShuffle`, `GetMinMaxAtDepth`/`FillAtDepth`, and others.

## 4. Hoist repeated subexpressions inside a loop body

If the same subexpression appears more than once in a loop body, compute it
once into a local and reuse it. Re-writing `j + 1` (or any index arithmetic)
several times per iteration re-does the work each time and obscures that all
those uses refer to the *same* value.

**Anti-example — do NOT follow this.** From the KV-cache eviction shift in
`TNNetScaledDotProductAttention.ComputeIncremental()`, `j + 1` is recomputed up
to six times per iteration, and `j * FDk` / `(j + 1) * FDk` twice each:

```pascal
for j := FEvictSinks to CacheLenM2 do
begin
  if FKVQuantInt8 then
  begin
    Move(FKCacheCodes[(j + 1) * FDk], FKCacheCodes[j * FDk], FDk * csShortIntSize);
    Move(FVCacheCodes[(j + 1) * FDk], FVCacheCodes[j * FDk], FDk * csShortIntSize);
    FKCacheScale[j] := FKCacheScale[j + 1];
    FVCacheScale[j] := FVCacheScale[j + 1];
  end
  else
  begin
    Move(FKCache.GetRawPtr(j + 1, 0, 0)^, FKCache.GetRawPtr(j, 0, 0)^, FDk * csNeuralFloatSize);
    Move(FVCache.GetRawPtr(j + 1, 0, 0)^, FVCache.GetRawPtr(j, 0, 0)^, FDk * csNeuralFloatSize);
  end;
end;
```

**Do this instead — compute each repeated value once:**

```pascal
for j := FEvictSinks to CacheLenM2 do
begin
  jP1 := j + 1;
  if FKVQuantInt8 then
  begin
    jDk   := j   * FDk;
    jP1Dk := jP1 * FDk;
    Move(FKCacheCodes[jP1Dk], FKCacheCodes[jDk], FDk * csShortIntSize);
    Move(FVCacheCodes[jP1Dk], FVCacheCodes[jDk], FDk * csShortIntSize);
    FKCacheScale[j] := FKCacheScale[jP1];
    FVCacheScale[j] := FVCacheScale[jP1];
  end
  else
  begin
    Move(FKCache.GetRawPtr(jP1, 0, 0)^, FKCache.GetRawPtr(j, 0, 0)^, FDk * csNeuralFloatSize);
    Move(FVCache.GetRawPtr(jP1, 0, 0)^, FVCache.GetRawPtr(j, 0, 0)^, FDk * csNeuralFloatSize);
  end;
end;
```

**Why it matters**

- **Less repeated work** — the index arithmetic is done once per iteration
  instead of once per use.
- **Readability** — a named `jP1` makes it explicit that every reference is the
  *same* neighbouring slot, not several independent computations.
- **Fewer edit hazards** — with one definition there is a single place to change
  if the offset logic ever moves, so the uses cannot drift out of sync.

(This eviction loop has further, larger problems beyond the repeated
subexpression — see later recommendations.)

## 5. Hoist loop-invariant computations out of the loop entirely

Recommendation #4 removes a subexpression recomputed *within one iteration*.
This one is stronger: if a value does not depend on the loop variable at all,
it is **loop-invariant** and should be computed *once before the loop* (or once
per call), not on every iteration.

The tell is that the expression contains none of the loop's induction
variables. In the same KV-cache path, the per-row byte size
`FDk * csNeuralFloatSize` (and `FDk * csShortIntSize`) depends only on `FDk`,
which is fixed for the whole call — yet it was recomputed at every `Move` in
both the eviction shift and the cache append.

**Anti-example — do NOT follow this** (recomputed every iteration):

```pascal
for p := 0 to SeqLenM1 do
begin
  ...
  Move(FKCache.GetRawPtr(j + 1, 0, 0)^, FKCache.GetRawPtr(j, 0, 0)^,
    FDk * csNeuralFloatSize);   // FDk is invariant — nothing here depends on p or j
  ...
end;
```

**Do this — compute once up front, then reuse:**

```pascal
RowBytesFP := FDk * csNeuralFloatSize;   // once per call, before the loop
RowBytesI8 := FDk * csShortIntSize;
for p := 0 to SeqLenM1 do
begin
  ...
  Move(FKCache.GetRawPtr(j + 1, 0, 0)^, FKCache.GetRawPtr(j, 0, 0)^, RowBytesFP);
  ...
end;
```

**Why it matters**

- **Work is done once, not N times** — the deeper the loop nest, the bigger the
  saving; an invariant hoisted out of an inner loop is removed from every
  iteration of every enclosing loop.
- **Readability** — a named `RowBytesFP` states *what* the quantity is ("bytes
  in one cache row") at the point it is computed.

**How far to hoist.** Lift the value to the outermost scope in which it is still
invariant. If it never changes across calls (a per-layer quantity), consider
caching it in a field computed at setup rather than recomputing it each call.
Do **not** hoist something whose inputs change inside the loop — that changes
behaviour, not just cost.

> Note: a good optimizing compiler may hoist trivial invariants on its own, but
> do not rely on it — hoisting explicitly guarantees the win, documents intent,
> and often exposes further simplifications (e.g. a shared row-pointer).

## 6. Replace loop multiplication with a running sum (strength reduction)

When an expression multiplies the loop variable by a constant — `j * FDk`,
`i * Stride`, `row * Width` — and the loop variable advances by a fixed step,
the product advances by a *fixed increment* too. So maintain it as a running
accumulator updated by addition, instead of recomputing the multiply every
iteration. This is **strength reduction**: trading a per-iteration multiply for
a per-iteration add (and a variable read), which is cheaper on essentially every
CPU.

It differs from the earlier recommendations:

- **#4 (CSE)** removes a subexpression repeated *within one iteration*.
- **#5 (invariant hoisting)** removes a value that does not change *at all*.
- **#6 (strength reduction)** handles a value that *does* change each iteration,
  but by a constant step — so it can be carried forward by `+=` instead of
  recomputed from scratch.

**Anti-example — do NOT follow this.** `j * FDk` and `(j + 1) * FDk` are each a
multiply recomputed every iteration (and duplicated across the K and V rows),
from the KV-cache eviction shift:

```pascal
for j := FEvictSinks to CacheLenM2 do
begin
  Move(FKCacheCodes[(j + 1) * FDk], FKCacheCodes[j * FDk], RowBytesI8);
  Move(FVCacheCodes[(j + 1) * FDk], FVCacheCodes[j * FDk], RowBytesI8);
  ...
end;
```

**Do this — carry the offset by addition, and update it once per iteration in
the loop body (not inside a branch) so every branch can share it:**

```pascal
jDk := FEvictSinks * FDk;              // seed once: the first j * FDk
for j := FEvictSinks to CacheLenM2 do
begin
  jP1Dk := jDk + FDk;                  // (j + 1) * FDk, by addition
  if FKVQuantInt8 then
  begin
    Move(FKCacheCodes[jP1Dk], FKCacheCodes[jDk], RowBytesI8);
    Move(FVCacheCodes[jP1Dk], FVCacheCodes[jDk], RowBytesI8);
    ...
  end
  else
  begin
    Move(FKCache.FData[jP1Dk], FKCache.FData[jDk], RowBytesFP);  // same offset (see #3)
    Move(FVCache.FData[jP1Dk], FVCache.FData[jDk], RowBytesFP);
  end;
  jDk := jP1Dk;                        // advance: next iteration's j * FDk
end;
```

The single seed multiply outside the loop replaces one multiply *per iteration*;
`jDk` and `jP1Dk` are each computed once and reused for both the K and V rows in
*both* storage formats (folding in #4's de-duplication and #3's shared offset).

**Why it matters**

- **Multiply → add** — the inner loop becomes add-only; only one multiply (the
  seed) remains, outside the loop.
- **Fewer operations overall** — combined with reusing each offset for the K and
  V rows, four multiplies per iteration collapse to one add plus one copy-forward.

**Caveats.**

- **The step must be constant.** Strength reduction is valid only because `j`
  advances by exactly 1 (so the product advances by exactly `FDk`). If the
  induction variable's step varies, the increment is not constant and this does
  not apply.
- **Advance the accumulator unconditionally, in the loop body.** Update `jDk`
  once per iteration outside any branch, so it stays correct no matter which
  branch runs and can be shared by all of them (here both the int8 and the FP32
  paths). Advancing it *inside* a branch is a trap: it is correct only if that
  branch runs on every iteration — e.g. a loop-invariant `if` — and silently
  desynchronises the offset the moment that stops being true.
- **Seed correctly.** The initial value must equal the product at the loop's
  first index (`FEvictSinks * FDk`, not `0`), or every subsequent offset is
  wrong.

## 7. Bind a repeatedly-accessed list element to a local

Indexing a list — `FNeurons[0]`, `FLayers[i]`, `Layer.Neurons[j]` — is **not** a
field read. The `[]` default property routes through the list's `Items[]` getter
(`GetItem`), which is a method call that indexes the backing storage (and, under
FPC's `TFPGObjectList`, range-checks and type-casts). Every `FNeurons[0].X` in a
loop pays that call again, even though the element reference never changes.

When you touch the same element more than once — whether several times in one
iteration or once per iteration across a loop — bind it to a local **once** and
go through the local thereafter.

**Do this:**

```pascal
Neuron0 := FNeurons[0];              // one list accessor call, up front
for t := 0 to SeqLenM1 do
begin
  Logit := TNNetVolume.DotProduct(
    Neuron0.FWeights.GetRawPtr(0, 0, 0), Prev.GetRawPtr(t, 0, 0),
    FFeatures) + Neuron0.FBiasWeight;
  ...
end;
```

**Instead of** (from `TNNetForgetGateBias.Compute()`, `neuralnetwork.pas` — two
`FNeurons[0]` list-accessor calls *per iteration*):

```pascal
for t := 0 to SeqLenM1 do
begin
  Logit := TNNetVolume.DotProduct(
    FNeurons[0].FWeights.GetRawPtr(0, 0, 0), Prev.GetRawPtr(t, 0, 0),
    FFeatures) + FNeurons[0].FBiasWeight;
  ...
end;
```

**Why it matters**

- **Skips the accessor call** — `Neuron0` is a plain reference; `FNeurons[0]`
  re-enters `GetItem` (call + index + optional range check/cast) on every use.
- **Readability** — `Neuron0` names *which* element the whole loop operates on,
  instead of repeating the list-lookup syntax at each site.
- **Consistency** — mirrors #3 (skip the volume element accessor) and the array
  mirror `FArrNeurons` this codebase already keeps for exactly this reason: a
  "fast (array) mirror of the `FNeurons` list" to avoid the list getter in hot
  paths.

**Caveat.** The binding is valid only while the list membership is fixed. If the
loop can add, remove, or reorder elements (or reallocate the list), the cached
reference goes stale — only hoist when the element identity is stable for the
binding's lifetime (it is, for the fixed per-layer neurons of a forward pass).

## 8. Hoist invariant *method* and *property* results, not just arithmetic

Recommendation #5 hoists loop-invariant *arithmetic* (`FDk * csNeuralFloatSize`).
The same rule applies — and pays off more — when the invariant is a **method
call or property getter** whose inputs are all constant across the loop. Such a
call *looks* like real per-iteration work, which is exactly why it hides: the eye
reads `GetRawPtr(0, 0, 0)` as "compute a pointer" and skips past the fact that
`(0, 0, 0)` never changes, so the pointer never changes either.

In the `TNNetForgetGateBias.Compute()` loop, `Neuron0.FWeights.GetRawPtr(0, 0, 0)`
returns the base address of the weight row — **the same pointer every
iteration** — and `Neuron0.FBiasWeight` is a fixed field read. Both are
loop-invariant and belong *above* the loop:

**Do this** (combining #7 and #8 — the fully hoisted loop):

```pascal
Neuron0     := FNeurons[0];                        // #7: list element, once
WeightsPtr  := Neuron0.FWeights.GetRawPtr(0, 0, 0); // #8: invariant pointer, once
Bias        := Neuron0.FBiasWeight;                 // #8: invariant field, once
for t := 0 to SeqLenM1 do
begin
  Logit := TNNetVolume.DotProduct(WeightsPtr, Prev.GetRawPtr(t, 0, 0), FFeatures)
           + Bias;
  FVal  := Sigmoid(Logit);
  FF.FData[t] := FVal;
  RunF  := RunF + pcr_logf(FVal);
  FLogF.FData[t] := RunF;
end;
```

`Prev.GetRawPtr(t, 0, 0)` **stays inside** the loop — it depends on the induction
variable `t`, so it is not invariant. Only the two arguments that do not depend
on `t` are hoisted.

**Why it matters**

- **Removes a call per iteration, not just a recompute** — unlike a hoisted
  arithmetic expression, hoisting a getter also removes the call overhead
  (dispatch, argument setup, any inlined offset math) from every iteration.
- **Exposes the intent** — `WeightsPtr` and `Bias` state plainly that these are
  fixed for the whole loop, which the inline `GetRawPtr(0,0,0)`/`FBiasWeight`
  spelling actively obscures.

**How to spot it.** Look at the *arguments*: if a call's arguments contain none
of the loop's induction variables (all literals or loop-invariant locals), its
result is loop-invariant — hoist it. If any argument depends on the loop
variable (like `t` above), it is not.

**Caveat.** Only hoist when the call is *pure with respect to the loop* — its
result must not depend on state the loop mutates. `GetRawPtr(0,0,0)` is safe here
because the weights volume is neither resized nor reallocated inside the loop; if
the loop could reallocate the backing buffer, the cached pointer would dangle.
Same rule as #5: hoist cost, never behaviour.

## 9. Bind the whole invariant access *chain*, above the outermost loop — worked example

Rules #4, #7, and #8 compose. When an access *chain* like
`FNeurons[0].FWeights` is invariant across a **nested** loop and appears more
than once per iteration, all three apply at once:

- **#7** — `FNeurons[0]` is a list-accessor call, not a field read.
- **#4** — the chain appears twice per iteration (once on each side of the
  assignment), so it is a repeated subexpression *within* one iteration.
- **#8 / #5** — the chain contains none of the loop variables, so it is
  loop-invariant; hoist it above the **outermost** loop it is invariant in, not
  merely to the top of the inner loop.

Bind the *entire* chain to one local, not just its first link. `FNeurons[0]`
alone still leaves `.FWeights` (a field deref) repeated; `FNeurons[0].FWeights`
as a single `W` collapses the whole thing.

**Anti-example — do NOT follow this** (from `TNNetModernHopfield.InitDefault()`,
`neuralnetwork.pas` — `FNeurons[0].FWeights` evaluated **twice per inner
iteration**, i.e. `2 * FNumPatterns * FDim` times, though it never changes):

```pascal
for p := 0 to NumPatternsM1 do
  for d := 0 to DimM1 do
    FNeurons[0].FWeights[p, 0, d] :=
      FNeurons[0].FWeights.RandomGaussianValue() * 0.1;
```

**Do this — bind the chain once, above both loops:**

```pascal
W := FNeurons[0].FWeights;              // list accessor + field deref, once total
for p := 0 to NumPatternsM1 do
  for d := 0 to DimM1 do
    W[p, 0, d] := W.RandomGaussianValue() * 0.1;
```

Now the list getter runs once instead of `2 * FNumPatterns * FDim` times, and
both the write target and the `RandomGaussianValue()` receiver go through the
same local. (This exact chain recurs in many `InitDefault`/init routines in
`neuralnetwork.pas` — the same one-line hoist applies to each.)

**Why it matters**

- **Collapses N² accessor calls to one** — the deeper the nest, the larger the
  multiplier removed; an invariant lifted above the outer loop is gone from every
  inner iteration.
- **Readability** — `W` names the one volume the whole nest initialises, instead
  of restating the `FNeurons[0].FWeights` path four times.

**Note on the remaining `[p, 0, d]` write.** It still goes through the volume
accessor (#3). Here that is fine: it is a lone write per cell with no reused
offset (the `p, 0, d` slot is touched once), so per #3's own test — "does the
offset get used at least twice?" — leave it as the accessor. This entry is about
hoisting the invariant *chain*, not about rewriting the per-cell store.

**Caveat.** As with #7, the binding assumes stable list membership and that the
weights volume is not reallocated inside the loop — true for a one-shot
initialiser like this. Hoist cost, never behaviour.

## 10. In hot paths, index `FArrNeurons` (the array mirror), not `FNeurons` (the list)

Rule #7 caches *one* repeatedly-accessed list element in a local. That does not
help when the index **varies** each iteration — `FNeurons[3*h]`,
`FNeurons[3*h+1]`, `FNeurons[3*h+2]` — because there is no single element to
bind. For exactly this case the codebase keeps `FArrNeurons`: a
`array of TNNetNeuron` holding the *same* `TNNetNeuron` references as `FNeurons`,
but indexed as a plain array — no `Items[]`/`GetItem` method call, no
property-getter bounds/cast overhead. Its own declaration comment calls it the
"fast (array) mirror of the `FNeurons` list … indexed directly without the
`TNNetNeuronList` method/bounds overhead." In hot compute/backprop loops, index
the mirror.

This example also recomputes `3 * h` three times per iteration — a repeated
subexpression (#4). Compute the base index once; since `h` advances by 1 the base
advances by a constant 3, so it is also a strength-reduction candidate (#6).

**Anti-example — do NOT follow this** (from the head loop in
`TNNetGroupedQueryAttention`-style `Compute()`/`Backpropagate()`,
`neuralnetwork.pas` ~32346 / ~32445 — a `FNeurons` list accessor *and* a
`3 * h` multiply, three of each per head):

```pascal
for h := 0 to HeadsM1 do
begin
  K1 := FNeurons[3 * h].FWeights;
  K2 := FNeurons[3 * h + 1].FWeights;
  V  := FNeurons[3 * h + 2].FWeights;
  ...
end;
```

**Do this — mirror index, base computed once:**

```pascal
for h := 0 to HeadsM1 do
begin
  i0 := 3 * h;                        // #4: one multiply, reused three ways
  K1 := FArrNeurons[i0    ].FWeights; // #10: array mirror, no list getter
  K2 := FArrNeurons[i0 + 1].FWeights;
  V  := FArrNeurons[i0 + 2].FWeights;
  ...
end;
```

**Why it matters**

- **No accessor call per element** — `FArrNeurons[i]` is a direct array read;
  `FNeurons[i]` re-enters `GetItem` (call + index + optional range check/cast)
  every time, and here the index differs each iteration so #7's bind-one-local
  trick cannot remove it.
- **One multiply, not three** — `i0 := 3 * h` computed once and reused for the
  `+0/+1/+2` slots (#4); it can be carried as a running `+= 3` accumulator across
  heads if you prefer (#6).
- **Consistency** — this is the mirror's intended use; the safetensors/GGUF
  loaders, savers, and several compute paths already read weights through
  `FArrNeurons` for the same reason.

**Precondition (rarely a concern in practice).** `FArrNeurons` is a snapshot of
references built by `BuildArrNeurons` (and re-run lazily by the compute loops /
at construction). Because it stores the *same* `TNNetNeuron` objects, edits to a
neuron's weight *contents* are always visible through either view — and a layer's
neurons are populated at construction and **do not normally change places**
afterwards, so by the time any forward/backward pass runs, the mirror is current.
The only situation that invalidates it is code that **adds, removes, or reorders**
neurons after the mirror was built; such setup/mutation code should rerun
`BuildArrNeurons` (the existing paths already do) or just use plain `FNeurons`.
For the ordinary hot compute/backprop loop, `FArrNeurons` is simply the correct
index.

## 11. Hoist to the innermost loop level at which a value is still invariant

Invariance is **relative to a particular loop**. A value can vary across an outer
loop yet be constant across the inner one — a *partial* invariant. Rule #5 says
"lift it to the outermost scope in which it is still invariant"; in a nested
loop that scope is often *between* the loops: the outer loop's body, **before**
the inner loop — not all the way out of both.

The tell (from #8) still works, applied to the *inner* loop: look at the call's
arguments and ask which loop's induction variable appears. If the inner index is
absent, the value is inner-loop-invariant even when an outer index is present —
so it belongs one level up.

**Anti-example — do NOT follow this** (attention backprop head loop,
`neuralnetwork.pas` ~31392 — `FOutputError.GetRawPtr(i, 0, QOfs)` is recomputed
**twice per `j` iteration**, i.e. `2 * SeqLen` times per `i`, though its
arguments are `i`, `0`, `QOfs` — none of them the inner variable `j`):

```pascal
for i := 0 to SeqLenM1 do
begin
  AttnRow := FAttn.GetRawPos(0, h * SeqLen + i, 0);   // already hoisted to the i level
  for j := 0 to SeqLenM1 do
  begin
    A := FAttn.FData[AttnRow + j];
    TNNetVolume.MulAdd(PrevErr.GetRawPtr(j, 0, VOfs),
      FOutputError.GetRawPtr(i, 0, QOfs), A, FDk);          // invariant across j
    FdAttnBuf[j] := TNNetVolume.DotProduct(
      FOutputError.GetRawPtr(i, 0, QOfs), Prev.GetRawPtr(j, 0, VOfs), FDk); // again
  end;
end;
```

**Do this — hoist the `i`-only pointer to the outer body, beside `AttnRow`:**

```pascal
for i := 0 to SeqLenM1 do
begin
  AttnRow := FAttn.GetRawPos(0, h * SeqLen + i, 0);
  dOutPtr := FOutputError.GetRawPtr(i, 0, QOfs);   // computed once per i, not per j
  for j := 0 to SeqLenM1 do
  begin
    A := FAttn.FData[AttnRow + j];
    TNNetVolume.MulAdd(PrevErr.GetRawPtr(j, 0, VOfs), dOutPtr, A, FDk);
    FdAttnBuf[j] := TNNetVolume.DotProduct(
      dOutPtr, Prev.GetRawPtr(j, 0, VOfs), FDk);
  end;
end;
```

`PrevErr.GetRawPtr(j, 0, VOfs)` and `Prev.GetRawPtr(j, 0, VOfs)` **stay inside**
the inner loop — they carry `j`, so they are not inner-invariant. Only the
`i`-indexed pointer moves up. This is the same discipline the code already
applies to `AttnRow` (hoisted to the `i` level); the `dOut` pointer simply
deserves the same.

**Why it matters**

- **Removes work from the whole inner loop** — an inner-invariant lifted one
  level runs `SeqLen`× fewer times per `i` (and it was computed *twice* per
  iteration here, so it also folds in #4).
- **Multiplies through nesting** — the deeper the nest, the more iterations a
  single correct hoist removes; a partial invariant left in place is the most
  common missed hoist in nested numeric code.

**How far to lift.** Place the binding at the *lowest* loop level whose body it
is still invariant in: one that uses only outer indices goes just inside the
outer loop; one that uses no loop index at all goes above every loop (#5/#8). Do
not lift a value past a loop whose variable it actually depends on — that changes
behaviour. (This same `GetRawPtr(i, 0, QOfs)` pattern recurs in the other
attention backprop paths around ~30806 / ~33415 / ~38104 — the identical
one-level hoist applies to each.)

## 12. Strength-reduce a per-iteration `GetRawPtr(loopvar, …)` into a carried offset

The pointer left *inside* the inner loop by #11 — `Prev.GetRawPtr(j, 0, VOfs)` —
is itself a strength-reduction target (#6), because `GetRawPtr` hides a multiply.
Recall the layout math (`neuralvolume.pas`):

```pascal
function TVolume.GetRawPos(x, y, d: integer): integer;
begin  Result := ((FSizeX * y) + x) * FDepth + d;  end;

function TVolume.GetRawPtr(x, y, d: integer): pointer;
begin  Result := Addr(FData[GetRawPos(x, y, d)]);  end;
```

With `y = 0`, `GetRawPtr(j, 0, VOfs)` addresses flat element `j * FDepth + VOfs`.
As `j` advances by 1 the offset advances by a **constant `FDepth`** — so the
per-iteration multiply `j * FDepth` can be replaced by a carried offset stepped
with `+=`. The step is exactly `GetRawPos(1, 0, 0)` (`= 1 * FDepth + 0 = FDepth`,
and independent of `FSizeX` because `y = 0`), which is a self-documenting way to
name the row stride without hardcoding `FDepth`. There is also a single-argument
`GetRawPtr(x) = Addr(FData[x])` (no multiply) — the natural consumer of a carried
flat offset.

Because `Prev` and `PrevErr` are the same previous layer's output and error, they
share one shape — so a **single** carried offset addresses both (rule #3's
shared-offset point), and it seeds at `VOfs` (`= GetRawPos(0, 0, VOfs)`).

**Anti-example — do NOT follow this** (`GetRawPtr(j, 0, VOfs)` recomputes
`j * FDepth` every iteration, for two volumes):

```pascal
for j := 0 to SeqLenM1 do
begin
  A := FAttn.FData[AttnRow + j];
  TNNetVolume.MulAdd(PrevErr.GetRawPtr(j, 0, VOfs), dOutPtr, A, FDk);
  FdAttnBuf[j] := TNNetVolume.DotProduct(
    dOutPtr, Prev.GetRawPtr(j, 0, VOfs), FDk);
end;
```

**Do this — carry one offset, advance by the row stride:**

```pascal
RowStride := Prev.GetRawPos(1, 0, 0);   // = FDepth; elements between consecutive j
posV      := VOfs;                       // = GetRawPos(0, 0, VOfs); the j = 0 offset
for j := 0 to SeqLenM1 do
begin
  A := FAttn.FData[AttnRow + j];
  TNNetVolume.MulAdd(PrevErr.GetRawPtr(posV), dOutPtr, A, FDk);   // one offset,
  FdAttnBuf[j] := TNNetVolume.DotProduct(
    dOutPtr, Prev.GetRawPtr(posV), FDk);                          // both volumes
  Inc(posV, RowStride);                  // next j's offset, by addition (#6)
end;
```

**Why it matters**

- **Multiply → add** — the inner-loop address math drops from a multiply-add per
  pointer to one shared `Inc`; `RowStride` and the seed are computed once.
- **One offset, two volumes** — folds in #3: same shape ⇒ `posV` indexes both
  `Prev` and `PrevErr`.

**Be honest about the magnitude — measure before spreading this.** Here each
`GetRawPtr` feeds a `MulAdd`/`DotProduct` over `FDk` elements; that vector kernel
dominates, and the single removed multiply is lost in the noise. The transform is
essentially free and correct, but do **not** expect a visible speedup when the
per-iteration body is a length-`FDk` kernel. Strength-reducing `GetRawPtr` pays
off where the inner body is *light* — scalar or very short-vector work, or a hot
offset used many times per iteration — and where the multiply is a real fraction
of the loop's cost. Treat this as the lowest-priority class of change in this
guide: apply it when you are already rewriting the loop for #11, not as a standalone
sweep.

**Caveats.** Same as #6: the induction step must be constant (`j` by 1 here), the
seed must equal the first offset (`VOfs`, not `0`), and `posV` must advance
unconditionally in the loop body. And the two volumes may share one offset only
while they genuinely share a shape — if `Prev` and `PrevErr` could ever differ in
`FDepth`, carry two offsets.
