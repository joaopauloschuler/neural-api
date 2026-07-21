# Optimization Guide

A running list of recommendations for keeping the neural-api codebase fast,
consistent, and maintainable. Each entry states the problem, the fix, and why
it matters. Recommendations are added incrementally as we discover them.

> ## ⛔ HARD PROHIBITION — read before applying ANY optimization
>
> **NEVER introduce a heap allocation on a compute path.** Do **NOT** add
> `SetLength`, `New`, `GetMem`, dynamic-array creation, `TXxx.Create`, or any
> other per-call allocation inside a layer's `Compute`, `ComputeCPU`,
> `ComputeRange`, `ComputeIncremental`, `Backpropagate`, or `Backpropagate*`
> method — nor inside **any** routine in `neuralvolume.pas`, nor inside the
> inference decode helpers (audio codecs, samplers, etc.).
>
> This is **the single most damaging mistake** you can make here. These methods
> run once per token / per sample / per training step; a `SetLength` there is a
> heap allocation (and often a zero-fill) on the hottest path in the system.
> Hoisting a `1-mix` vector or a section table "once per call" into a fresh
> `SetLength` array is **slower than the original scalar code you were trying to
> improve** — you traded a few arithmetic ops for a malloc. **This is very
> wrong. Do not do it.**
>
> If a rule below (e.g. #5 hoist-invariant, #13 bulk-method, App. D scratch)
> seems to call for a temporary buffer:
> 1. **Prefer no buffer at all** — most invariant hoists (a scalar, a base
>    offset, a bound) need only a local `integer`/`TNeuralFloat`, never an array.
> 2. If a buffer is genuinely required, it MUST be a **persistent, lazily-sized
>    layer field** (e.g. `FScratch: TNNetVolume` allocated in `SetPrevLayer`,
>    or a field resized only when `Length(field) <> needed`), **never** a
>    fresh `SetLength` of a local on every call.
> 3. If neither is possible, **do not apply the optimization** — leave the
>    original code and note it. A correct scalar loop beats an allocating
>    "optimization" every time.
>
> The only acceptable `SetLength` on these paths is a **lazy, amortized resize
> of a persistent field** guarded by a size check, so it allocates ~once over
> the lifetime of the layer, not once per call. See rule #17.

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
FDkIntSize := FDk * csShortIntSize;
FDkFloatSize := FDk * csNeuralFloatSize;
for j := FEvictSinks to CacheLenM2 do
begin
  jP1 := j + 1;
  if FKVQuantInt8 then
  begin
    jDk   := j   * FDk;
    jP1Dk := jP1 * FDk;
    Move(FKCacheCodes[jP1Dk], FKCacheCodes[jDk], FDkIntSize);
    Move(FVCacheCodes[jP1Dk], FVCacheCodes[jDk], FDkIntSize);
    FKCacheScale[j] := FKCacheScale[jP1];
    FVCacheScale[j] := FVCacheScale[jP1];
  end
  else
  begin
    Move(FKCache.GetRawPtr(jP1, 0)^, FKCache.GetRawPtr(j, 0)^, FDkFloatSize);
    Move(FVCache.GetRawPtr(jP1, 0)^, FVCache.GetRawPtr(j, 0)^, FDkFloatSize);
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

**Worked example — the residual *left behind* by an offset-hoist.** Hoisting the
base per #11 is not the finish line. A `base + D` combination can still be formed
several times *within a single iteration*, and that is #4's job. In
`TNNetGaussianReparameterize.Backpropagate` the bases are already hoisted, yet
`basePrev0 + D` is formed four times and `basePrev0 + D + HalfDepth` twice per `D`:

```pascal
for D := 0 to MaxD do
begin
  g      := FOutputError.FData[baseErr0 + D];
  logVar := FPrevLayer.FOutput.FData[basePrev0 + D + HalfDepth];
  sigma  := NeuralExp(0.5 * logVar);
  e      := FEps.FData[baseEps0 + D];
  FPrevLayer.FOutputError.FData[basePrev0 + D] :=
    FPrevLayer.FOutputError.FData[basePrev0 + D] + g;
  FPrevLayer.FOutputError.FData[basePrev0 + D + HalfDepth] :=
    FPrevLayer.FOutputError.FData[basePrev0 + D + HalfDepth] + g * 0.5 * sigma * e;
end;
```

**Do this — name the two indices once:**

```pascal
for D := 0 to MaxD do
begin
  idx   := basePrev0 + D;             // #4: the (base + D) add, done once
  idxHi := idx + HalfDepth;
  g      := FOutputError.FData[baseErr0 + D];
  logVar := FPrevLayer.FOutput.FData[idxHi];
  sigma  := NeuralExp(0.5 * logVar);
  e      := FEps.FData[baseEps0 + D];
  FPrevLayer.FOutputError.FData[idx]   := FPrevLayer.FOutputError.FData[idx]   + g;
  FPrevLayer.FOutputError.FData[idxHi] := FPrevLayer.FOutputError.FData[idxHi] + g * 0.5 * sigma * e;
end;
```

The two writes stay scalar here — the `dlogvar` term has a per-element `sigma`
and `e`, so it is *not* a #13 bulk candidate. (The `dmu` term `prevErr[idx] += g`
alone *is* a pure accumulate and can be promoted per #13; see that rule's
mixed-body split.)

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
  Move(FKCache.GetRawPtr(j + 1, 0)^, FKCache.GetRawPtr(j, 0)^, RowBytesFP);
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

**Worked example — an invariant buried inside a per-element write.** The scalar
`Coeff * E * InvT` is invariant across the whole `X/Y/D` nest, yet it is
re-multiplied on every element:

```pascal
for X := 0 to MaxX do
  for Y := 0 to MaxY do
  begin
    baseErr0 := FOutputError.GetRawPos(X, Y, 0);
    for D := 0 to EM1 do
    begin
      GradScale := Coeff * E * FFBuf[D] * InvT;   // Coeff*E*InvT recomputed every D
      FOutputError.FData[baseErr0 + D] := GradScale;
    end;
  end;
```

**Do this — hoist the invariant factor once; the inner loop becomes a pure scale:**

```pascal
kScale := Coeff * E * InvT;                        // #5: once for the whole call
for X := 0 to MaxX do
  for Y := 0 to MaxY do
  begin
    baseErr0 := FOutputError.GetRawPos(X, Y);
    for D := 0 to EM1 do
      FOutputError.FData[baseErr0 + D] := kScale * FFBuf[D];
  end;
```

With the invariant gone, the inner loop is `dst[D] := kScale * FFBuf[D]` — a
uniform scale, so it is *also* a **rule #13** candidate: copy `FFBuf` into the
output row, then `TNNetVolume.Mul(dstPtr, kScale, EM1 + 1)` in place (valid only
because `FFBuf` is a plain contiguous buffer with no per-element transcendental).
This is the common chain: **#5 exposes the invariant, #13 then vectorizes what is
left.**

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
    Neuron0.FWeights.GetRawPtr(0, 0), Prev.GetRawPtr(t, 0),
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
    FNeurons[0].FWeights.GetRawPtr(0, 0), Prev.GetRawPtr(t, 0),
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
WeightsPtr  := Neuron0.FWeights.GetRawPtr(0, 0); // #8: invariant pointer, once
Bias        := Neuron0.FBiasWeight;                 // #8: invariant field, once
for t := 0 to SeqLenM1 do
begin
  Logit := TNNetVolume.DotProduct(WeightsPtr, Prev.GetRawPtr(t, 0), FFeatures)
           + Bias;
  FVal  := Sigmoid(Logit);
  FF.FData[t] := FVal;
  RunF  := RunF + pcr_logf(FVal);
  FLogF.FData[t] := RunF;
end;
```

`Prev.GetRawPtr(t, 0)` **stays inside** the loop — it depends on the induction
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

**Worked variant — split an offset into an inner-invariant base plus the inner
index.** The hoist above lifted a whole *pointer*; the same logic applies to a
raw *flat offset* whose innermost index is only *added* at the end. Because

```pascal
GetRawPos(X, Y, D) = ((FSizeX * Y) + X) * FDepth + D
                   = GetRawPos(X, Y, 0) + D
```

the multiply-add `((FSizeX * Y) + X) * FDepth` is exactly `GetRawPos(X, Y, 0)` —
which is **invariant across a `D` loop** — and the inner index enters as a bare
`+ D`. So compute the base at `D = 0` once in the outer body and index
`base + D` inside, instead of recomputing the whole offset (with its multiply)
every `D`.

**Anti-example — do NOT follow this** (`GetRawPos(X, Y, D)` recomputes
`((FSizeX*Y)+X)*FDepth` on every `D`, though only the trailing `+ D` changes):

```pascal
for X := 0 to MaxX do
  for Y := 0 to MaxY do
    for D := 0 to MaxD do
    begin
      basePrev := FPrevLayer.FOutput.GetRawPos(X, Y, D);   // multiply-add every D
      a := FPrevLayer.FOutput.FData[basePrev];
      b := FPrevLayer.FOutput.FData[basePrev + HalfDepth];
      // Exact Gaussian CDF: Phi(b) = 0.5*(1 + erf(b/sqrt(2))).
      cdf := 0.5 * (1 + pcr_erff(b * INV_SQRT_2));
      FOutput[X, Y, D] := a * (b * cdf);
    end;
```

**Do this — hoist the base to `D = 0`, one level up, and add `D` inside — for
the output volume too:**

```pascal
for X := 0 to MaxX do
  for Y := 0 to MaxY do
  begin
    basePrev0 := FPrevLayer.FOutput.GetRawPos(X, Y);      // = ((FSizeX*Y)+X)*FInDepth, once per (X,Y)
    baseOut0  := FOutput.GetRawPos(X, Y);                 // FOutput's own base (different shape)
    for D := 0 to MaxD do
    begin
      basePrev := basePrev0 + D;                          // just an add
      a := FPrevLayer.FOutput.FData[basePrev];
      b := FPrevLayer.FOutput.FData[basePrev + HalfDepth];
      cdf := 0.5 * (1 + pcr_erff(b * INV_SQRT_2));
      FOutput.FData[baseOut0 + D] := a * (b * cdf);       // store, no accessor
    end;
  end;
```

Now the `((FSizeX*Y)+X)*FDepth` multiply-add runs once per `(X, Y)` instead of
once per `D` — on **both** sides. The read base is hit twice per iteration
(`basePrev` and `basePrev + HalfDepth`), so direct `FData` is warranted by #3's
"used twice" test outright. The *write* is a lone store, but it is still worth
hoisting: the justification is #11, not #3 — the base multiply
`((FSizeX*Y)+X)*FDepth` is **invariant across the whole `D` loop**, so lifting it
turns every store's offset from a multiply-add into a plain `baseOut0 + D` add.
That is a real per-iteration saving even for a single write, and it is *not* the
#3 anti-pattern: the anti-pattern is calling `GetRawPos` **inline** at a
single-use site (identical cost to the accessor); here the offset's expensive
part is computed once *outside* the loop and only an add remains inside.

**`FOutput` needs its own base — do not reuse `basePrev0`.** This is a gated
activation: `FOutput` has half the input depth (`FInDepth = 2 * FDepth`), so its
per-slot stride differs and `GetRawPos(X, Y, 0)` yields a different base. Rule
#3's "reuse one base across same-shaped volumes" applies only when the shapes
match — they do not here, so carry two bases. (If the layer *were* shape-
preserving, one base would index both.)

Since `D` advances by 1, both `basePrev` and `baseOut0 + D` advance by 1, so they
can equally be carried with `Inc` per #12 — but `base0 + D` is often the clearest
spelling and just as cheap.

**Caveat — the bare `+ D` decomposition only works for the *depth* index.** `D`
is the last term in `GetRawPos` with coefficient 1, which is why
`GetRawPos(X, Y, D) = GetRawPos(X, Y, 0) + D`. If instead the *inner* loop varied
`X`, one `X`-step moves the offset by `FDepth` (not 1); if it varied `Y`, by
`FSizeX * FDepth`. In those cases the constant step is the row/plane stride and
this becomes the carried-offset strength reduction of #12 (`Inc(pos, FDepth)`),
not a plain `+ innerIndex`.

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
RowStride := Prev.GetRawPos(1, 0);       // = FDepth; elements between consecutive j
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

**Apply it everywhere the pattern occurs — every cycle counts.** The transform is
always valid and essentially free, so apply it to *every* per-iteration
`GetRawPtr(loopvar, …)` / `GetRawPos(loopvar, …)`, not only the ones with an
obviously light inner body. The magnitude of the win does vary: where the inner
body is *light* — scalar or short-vector work, or a hot offset used many times per
iteration — the removed multiply is a real fraction of the loop's cost and the
speedup is directly visible; where the inner body is a length-`FDk`
`MulAdd`/`DotProduct` the vector kernel dominates and one removed multiply is a
smaller slice. But "smaller" is not "zero": across the whole forward/backward pass
these multiplies number in the millions, and this codebase is chasing the last few
percent to beat a hand-tuned C/GGML (ollama/llama.cpp-class) baseline — so we take
the cycles wherever the pattern appears. The only reason *not* to apply it at a
given site is a correctness blocker (a non-constant step, a per-element scatter
index — see the caveats), never "the body is too heavy to bother." Fold it in
whenever you are already rewriting a loop for #11, and it is also worth a
standalone sweep.

**Caveats.** Same as #6: the induction step must be constant (`j` by 1 here), the
seed must equal the first offset (`VOfs`, not `0`), and `posV` must advance
unconditionally in the loop body. And the two volumes may share one offset only
while they genuinely share a shape — if `Prev` and `PrevErr` could ever differ in
`FDepth`, carry two offsets.

## 13. Promote a degenerate elementwise inner loop to a `TNNetVolume` bulk method

Rules #11 and #12 make the inner loop's *addressing* cheap. But once the base is
hoisted, many inner loops turn out to be nothing more than a **uniform elementwise
op over a contiguous run with loop-invariant scalar coefficients** — a copy, a
scale, an accumulate. That whole loop is already implemented, once, as an
AVX-vectorized primitive on `TNNetVolume`. Recognise the shape and **call the
method** instead of hand-rolling a scalar `D` loop: you replace `N` scalar
operations *plus* `N` bounds checks *plus* the loop overhead with a single kernel
call. This is a bigger win than #11/#12 at the same site — it changes the
instruction mix (scalar → SIMD), not just one multiply into an add.

The primitives (all in `neuralvolume.pas`; pointer-form class methods are the ones
that consume a hoisted `GetRawPtr`):

| Inner loop (after #11 hoist) | Meaning | Call |
| --- | --- | --- |
| `dst[base+D] := src[base+D]` | contiguous copy | `Move(src.FData[base], dst.FData[base], N * csNeuralFloatSize)` (see #1) or `dst.Copy(src, N)` |
| `dst[D] := src[D] * k` (`k` invariant) | scale-copy | `Move` then `TNNetVolume.Mul(dstPtr, k, N)` (in-place `*= k`) |
| `dst[base+D] := dst[base+D] + src[base+D]` | accumulate | `TNNetVolume.Add(dstPtr, srcPtr, N)` |
| `dst[base+D] := dst[base+D] + src[base+D] * k` | scaled accumulate | `TNNetVolume.MulAdd(dstPtr, srcPtr, k, N)` |
| `dst[D] := dst[D] + a[D] * b[D]` | elementwise FMA | `TNNetVolume.MulAdd(dstPtr, aPtr, bPtr, N)` |
| `acc := acc + a[D] * b[D]` | reduction | `TNNetVolume.DotProduct(aPtr, bPtr, N)` |

**Anti-example — do NOT hand-roll a scalar map** (`neuralnetwork.pas` GEGLU, the
`B/sqrt(2)` write):

```pascal
outPtr := FOutput.GetRawPtr(X, Y, 0);
bPtr   := FPrevLayer.FOutput.GetRawPtr(X, Y, HalfDepth);
for D := 0 to MaxD do
  outPtr^[D] := bPtr^[D] * INV_SQRT_2;   // N scalar muls + N bound checks
```

**Do this — one vectorized call:**

```pascal
FOutput.Copy(FPrevLayer.FOutput, ... );          // or Move the B half in
TNNetVolume.Mul(outPtr, INV_SQRT_2, HalfDepth);  // AVX scale, in place
```

(The very next lines of that same method already do exactly this for the erf ride
via `TNNetVolume.VectorErf` — the scalar seed loop above it is the leftover.)

**The most degenerate shape — a pure contiguous copy — is `Move`.** When the
inner loop copies one run to another with no arithmetic at all:

```pascal
for D := 0 to MaxD do
  FOutput.FData[baseOut0 + D] := FPrevLayer.FOutput.FData[basePrev0 + D];  // scalar copy
```

it is exactly a `memmove`:

```pascal
Move(FPrevLayer.FOutput.FData[basePrev0], FOutput.FData[baseOut0],
  (MaxD + 1) * csNeuralFloatSize);   // one call; see #1 for the byte-size idiom
```

(or `FOutput.Copy(FPrevLayer.FOutput, MaxD + 1)`). This one *is* bit-exact — no
float op happens — so there is never a reason to leave it as a scalar loop.

**The one precondition: uniformity.** Promotion is valid only when *every* element
runs the *same* op with *only* loop-invariant scalars. The moment the body carries
a **per-element transcendental** (`NeuralExp(0.5*logVar)` recomputed each `D`) or a
**data-dependent factor** (`g * 0.5 * sigma * e`, where `sigma` and `e` vary per
`D`), it is not a uniform map and must stay scalar — *unless* you first
materialise the varying factor into a contiguous temp and then fire one bulk call.
A mixed body splits: in the VAE-reparam backward, the `dmu` branch
`prevErr[base+D] += g` **is** a pure `TNNetVolume.Add(dstPtr, gPtr, N)`, while the
`dlogvar` branch (per-element `sigma`, `e`) stays scalar. Promote the promotable
half; do not force the rest.

### Equal dimensions ⇒ equal base — compute it once

- **Equal dimensions ⇒ equal base — compute it once (flip side of #11).** If two
  volumes have identical `FSizeX`/`FSizeY`/`FDepth`, then `GetRawPos(X, Y, 0)` is
  the *same integer* for both, so one hoisted base indexes both (#12 already leans
  on this: "same shape ⇒ one offset addresses both"). The trap is the reverse,
  which #11 spells out: a **gated** layer's `FOutput` is half the input depth, so
  its base differs and must be carried separately. Rule of thumb: share a base
  only after you have verified the shapes match; never assume it.

  ```pascal
  // anti — two calls when the volumes provably share a shape:
  baseOut0 := FOutput.GetRawPos(X, Y, 0);
  baseErr0 := FOutputError.GetRawPos(X, Y, 0);   // same dims ⇒ identical integer
  // do — compute once, index both (ONLY after verifying the shapes match):
  base0 := FOutput.GetRawPos(X, Y, 0);           // a gated layer would NOT qualify
  ```
- **A second pointer into the *same* volume at a constant depth = first pointer +
  that constant.** Instead of a second accessor call
  `bPtr := V.GetRawPtr(X, Y, HalfDepth)` next to `aPtr := V.GetRawPtr(X, Y, 0)`,
  the offset differs by a compile-time constant, so `bPtr` is just `aPtr` advanced
  by `HalfDepth`. In index form this is the `FData[base + HalfDepth]` already shown
  in #11; in pointer form it needs an element-typed pointer (or `{$POINTERMATH ON}`)
  to write `aPtr + HalfDepth` — if that is not in scope, keep the index form rather
  than a redundant `GetRawPtr`.

  ```pascal
  // anti — a second accessor call for a constant-offset neighbour:
  aPtr := FPrevLayer.FOutput.GetRawPtr(X, Y, 0);
  bPtr := FPrevLayer.FOutput.GetRawPtr(X, Y, HalfDepth);   // same volume, +HalfDepth
  // do — one call, then add the constant (element-typed ptr / {$POINTERMATH ON}):
  aPtr := FPrevLayer.FOutput.GetRawPtr(X, Y, 0);
  bPtr := aPtr + HalfDepth;
  // if pointer math is not in scope, keep the index form: FData[base + HalfDepth]
  ```

### Use `GetRawPos(x, y)` instead of `GetRawPos(x, y, 0)` 
- Use `GetRawPos(x, y)` / `GetRawPtr(x, y)` instead of `GetRawPos(x, y, 0)` / `GetRawPtr(x, y, 0)`
as `GetRawPos(x, y)` / `GetRawPtr(x, y)` overloads are faster.

## 14. Rank on the cheapest order-equivalent quantity — skip transforms used only to compare

Rules #1–#13 make a given computation *cheaper*. This one is different in kind: it
**removes the computation entirely** when its result is used only to rank, compare,
or `argmax`. Two facts do the work:

- A **strictly monotonic** function preserves order: `f(a) > f(b) ⟺ a > b`. So if
  you only need the *position* of the maximum (not the transformed value), take the
  `argmax` of the cheap pre-image, not the transform.
- **Multiplying/dividing every candidate by the same positive constant** preserves
  order too. A normalizer that is identical across the candidates cancels out of any
  comparison between them.

The `argmax` of a softmax is the archetype: `softmax(L)_c = exp(L_c − M) / SumExp`.
`exp` is strictly monotonic and `SumExp` is one positive constant shared by every
class, so `argmax_c softmax(L)_c = argmax_c L_c` — the winning class is read
straight off the raw logits, with **no `exp` and no divide**.

**Anti-example — do NOT compute the transform per candidate just to rank**
(`DecodeDetrDetections`, best foreground class; `L[c]` is `Output.FData[qBase + c]`):

```pascal
SumExp := 0;
for c := 0 to NumLabels do SumExp := SumExp + Exp(L[c] - MaxLogit);
BestCls := 0; BestProb := -1;
for c := 0 to NumLabelsM1 do
begin
  Prob := Exp(L[c] - MaxLogit) / SumExp;   // NumLabels Exp + NumLabels divides, ONLY to rank
  if Prob > BestProb then begin BestProb := Prob; BestCls := c; end;
end;
```

**Do this — rank on the raw logit; compute the winner's value once:**

```pascal
// argmax of prob == argmax of logit (Exp monotonic; /SumExp a positive constant).
SumExp := 0; BestCls := 0;
for c := 0 to NumLabels do
begin
  SumExp := SumExp + Exp(L[c] - MaxLogit);                 // still needed: it is the score's denominator
  if (c <= NumLabelsM1) and (L[c] > L[BestCls]) then BestCls := c;   // no Exp, no divide
end;
BestProb := Exp(L[BestCls] - MaxLogit) / SumExp;           // ONE Exp/divide — for the winner only
```

The per-class `exp`+divide loop is gone; the selection folds into the `SumExp` pass
you were already running, and the probability is computed once for the class you
actually return.

**Keep the value only where the value is genuinely needed.** The point is not "never
compute the probability" — here `BestProb` is still needed as the score and for the
`>= Threshold` test, so it *is* computed, but **once for the winner** instead of once
per candidate. Same shape elsewhere: a `sigmoid(x) > t` gate with constant `t` is
`x > ln(t/(1−t))` (threshold on the pre-image — one `ln` at setup, none per element);
a nearest-neighbour search compares **squared** distances and takes the one `sqrt`
only on the reported minimum.

**Why it matters**

- **Fewer operations, not just cheaper ones** — unlike #13 (which vectorizes the
  same work), this changes the operation *count*: `N` transcendentals/divides drop to
  `O(1)`. When it applies it is the highest-leverage move in this guide.
- **Often collapses a loop** — the ranking merges into an existing pass (the `SumExp`
  accumulation above), removing a whole traversal.

**Caveats — this one needs a correctness argument, so state it in a comment.**

- **Monotonicity must actually hold, and in the right direction.** `exp`, `log`,
  `sigmoid`, `softplus`, `x²` *for x ≥ 0* are increasing; `x²` over signed `x` is
  **not** monotonic (don't rank signed values by their square). A decreasing
  transform flips `argmax`↔`argmin`.
- **The shared normalizer must truly be shared and positive.** `/SumExp` cancels only
  because every class in one query divides by the *same* `SumExp`; a per-candidate or
  possibly-negative divisor does not cancel.
- **Preserve the original tie-breaking.** Keep the same strict/non-strict comparison
  (`>` vs `>=`) the transformed version used, so that near-ties resolve to the same
  index. (Float `exp` is monotonic non-decreasing, so two extremely close logits can
  map to equal probabilities — but then either pick is equally correct, and matching
  the original `>`/`>=` keeps behaviour identical.)
- **This is an algebraic rewrite, not a mechanical one** — it is the one rule here a
  pattern-matching sweep will not surface. It requires reading what a value is *for*.

## 15. Strength-reduce `*`/`div` by a compile-time power of two to `shl`/`shr`

A multiply or divide by a constant power of two is a single bit-shift: `x * 2^k` is
`x shl k`, `x div 2^k` is `x shr k` (for non-negative `x`). A shift is one cheap ALU
op; the general `div` in particular is many cycles. In dimension arithmetic this
recurs constantly — `pHeadDim div 2`, `div 4`, `* 8`.

**Anti-example — do NOT write the arithmetic form for a constant power of two:**

```pascal
HalfD    := pHeadDim div 2;
QuarterD := pHeadDim div 4;
```

**Do this:**

```pascal
HalfD    := pHeadDim shr 1;   // div 2
QuarterD := pHeadDim shr 2;   // div 4
```

(and `x shl k` for `x * 2^k`).

**Why not just trust the compiler?** For an **unsigned** operand FPC does reduce
`div 2^k` to a shift. For a **signed** operand it legally cannot emit a bare `shr`:
`div` truncates toward zero but an arithmetic right shift rounds toward −∞, so the
compiler must add sign-correction (a conditional add of `2^k − 1` before shifting) —
extra instructions on every use. Writing `shr` yourself skips that correction.

**Caveats.**
- **`shr` is only equal to `div 2^k` when the value is non-negative.** Dimensions,
  sizes, counts, and offsets always are, so this is safe there. Do **not** blind-swap
  `div`→`shr` on a value that can be negative (a signed delta, a coordinate that can
  go negative) — the results differ for negatives.
- **Only for a genuine compile-time power of two.** `div 3`, `div 6`, `div FDepth`
  are not shifts; leave them (a `div` by a runtime value is not reducible here).
- Same idea for `mod`: `x mod 2^k` on a non-negative `x` is `x and (2^k − 1)`.

## 16. Use `NeuralExp` (the fast, trap-free exp), not the RTL `Exp`, in model math

`neuralvolume.pas` provides `NeuralExp` — a table + polynomial `exp` (a 64-entry
`2^(i/64)` LUT with a degree-5 minimax poly, the scalar sibling of the AVX
`VectorExp`). Two reasons it is the right default in the network's forward/backward
math:

- **Faster** than the RTL `System.Exp` (which does a full accurate range-reduced
  libm-style evaluation). The approximation is well within this codebase's tolerance
  (see the no-bit-parity policy) and is already the convention — it is used in over a
  hundred sites across `neuralnetwork.pas`.
- **Overflow-trap-free.** `NeuralExp` is a `{$Q-}`/`{$R-}` clone, so a large argument
  cannot raise a range/overflow exception under a debug (`-Criot`) build the way
  `System.Exp` can. This has bitten real model runs on padded/degenerate rows.

**Anti-example — RTL `Exp` in a hot activation / gate / softmax body:**

```pascal
sg := 1.0 / (1.0 + Exp(-x));          // sigmoid
e  := Exp(logit[c] - MaxLogit);       // softmax term
dec := Exp(-sp * pn);                 // gate decay
```

**Do this — the codebase's fast exp:**

```pascal
sg := 1.0 / (1.0 + NeuralExp(-x));
e  := NeuralExp(logit[c] - MaxLogit);
dec := NeuralExp(-sp * pn);
```

**Where NOT to swap.** `NeuralExp` is an *approximation*. Leave `System.Exp` in code
where exactness is the point, not speed:
- weight/activation **quantizer** scale computation and other numerically sensitive
  setup that runs once (cost is irrelevant, accuracy matters),
- importers / config / metric code outside the per-token or per-element path,
- any site a comment explicitly marks as needing a stable/exact evaluation.
When in doubt on a hot forward/backward path, prefer `NeuralExp`; on a
once-per-model setup path, leave `Exp`. (The same applies to the other `Neural*` /
`pcr_*` fast transcendentals versus their RTL equivalents.)

## 17. NEVER `SetLength`/allocate per call in `Compute`/`Backpropagate` (or in `neuralvolume.pas`)

See the **HARD PROHIBITION** box at the top of this guide. This rule exists because
the mistake was made repeatedly during an optimization sweep and each instance was
**slower than the code it replaced**.

**The anti-pattern (NEVER do this):**

```pascal
procedure TNNetTokenShift.Compute();
var
  oneMinusMix: array of TNeuralFloat;   // <-- local dynamic array
begin
  ...
  // "hoist the (1-mix) vector once per call"  <-- WRONG: this is a per-call malloc
  SetLength(oneMinusMix, Depth);
  for d := 0 to MaxD do oneMinusMix[d] := 1.0 - W.Raw[d];
  for t := 1 to MaxT do
    for d := 0 to MaxD do
      FOutput.FData[base + d] := mix * xt + oneMinusMix[d] * xtm1;
end;
```

The `SetLength` runs on **every forward call** (every token during decode). It
allocates + zero-fills a heap array to save one subtraction per element — a net
loss. The correct code keeps the trivial scalar:

```pascal
  for t := 1 to MaxT do
    for d := 0 to MaxD do
    begin
      mix := W.Raw[d];
      one_minus_mix := 1.0 - mix;               // one local, no allocation
      FOutput.FData[base + d] := mix * xt + one_minus_mix * xtm1;
    end;
```

Same verdict for a "section table" (`SetLength(secTab, HalfD)` to cache a
per-`k` branch selector), a "snapshot array" (`SetLength(Vols, Count)` in a
`TNNetVolumeList` method to avoid a list getter), a "repacked weight column"
(`SetLength(WT, ...)` in a decode conv), etc. **All were reverted.** The list
getter / branch / small recompute they replaced is cheaper than the allocation.

**Forbidden on every `Compute*`/`Backpropagate*` path and everywhere in
`neuralvolume.pas`:** `SetLength`, `New`, `GetMem`, `TXxx.Create`,
`Copy(dynarray)`, `Concat`, and any implicit dynamic-array/temporary-string
allocation.

**The only allowed form — lazy, amortized resize of a persistent field:**

```pascal
  // FScratch is a LAYER FIELD, not a local. This allocates ~once, then reuses.
  if FScratch.Size <> Depth then FScratch.ReSize(1, 1, Depth);   // amortized
  // ... use FScratch.FData[...] ...
```

Even then, reach for it only when an invariant genuinely needs a *vector* of
results across the loop. The overwhelming majority of hoists (rules #4/#5/#8/#11)
need a **scalar or an integer offset** — a plain local — and allocate nothing.
**If the only way you can see to apply an optimization is to allocate, the
correct action is to NOT apply it.**

## 18 TNNetVolume methods are faster than pure pascal equivalents
The below TNNetVolume methods are faster than equivalent pure pascal implementations
```pascal
      procedure Fill(c: Single = 0); {$IFDEF Release} inline; {$ENDIF}
      procedure Add(Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      class procedure Add(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure Sub(Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      function DotProduct(Original: TNNetVolume): TNeuralFloat; overload; {$IFDEF Release} inline; {$ENDIF}
      class function DotProduct(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single; overload; {$IFDEF Release} inline; {$ENDIF}
      procedure Mul(Value: Single); overload; {$IFDEF Release} inline; {$ENDIF}
      class procedure Mul(PtrA: TNeuralFloatArrPtr; MulOp: TNeuralFloat; pSize: integer); overload;
      class procedure Mul(PtrA, PtrB: TNeuralFloatArrPtr; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      class procedure MaxElements(PtrA, PtrB: TNeuralFloatArrPtr; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure MulAdd(Value: TNeuralFloat; Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure MulAdd(Original1, Original2: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure MulMulAdd(Value1, Value2: TNeuralFloat; Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure MulAdd(Value: TNeuralFloat; PtrB: TNeuralFloatArrPtr); overload; {$IFDEF Release} inline; {$ENDIF}
      class procedure MulAdd(PtrA, PtrB: TNeuralFloatArrPtr; Value: TNeuralFloat; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      class procedure MulAdd(PtrA, PtrB, PtrC: TNeuralFloatArrPtr; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure Divi(Value: Single); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure Copy(Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure CopyRelu(Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure CopyPadding(Original: TNNetVolume; Padding: integer); overload;
      procedure CopyPadding(Original: TNNetVolume; PaddingX, PaddingY: integer); {$IFDEF Release} inline; {$ENDIF} overload;
      procedure CopyNoChecks(Original: TNNetVolume);
      function GetSum(): TNeuralFloat; override;
      function GetSumSqr(): TNeuralFloat; override;
      function GetDistanceSqr(Original: TNNetVolume): TNeuralFloat;  overload; {$IFDEF Release} inline; {$ENDIF}
      function GetDistance(Original: TNNetVolume): TNeuralFloat;  overload; {$IFDEF Release} inline; {$ENDIF}
      function SumDiff(Original: TNNetVolume): TNeuralFloat; overload; {$IFDEF Release} inline; {$ENDIF}
```

## Appendix: the patterns agents miss most — extra worked examples

These reinforce rules already stated; they are collected here because review keeps
finding them un-applied. Same rules, more surface.

### A. Repeated `(t * Stride) + d` index arithmetic — rule #4/#5 (MOST-MISSED)

The single most-recurring miss. An index built from a loop variable and a stride is
recomputed several times per iteration, and the `t * Stride` part is invariant across
the inner loop.

**Anti-example:**

```pascal
FDelta.FData[(t * Depth) + d] := sp;
FBt.FData[(t * Depth) + d]    := acc;
FCt.FData[(t * Depth) + d]    := hprev;
FAt.FData[(t * Depth) + d]    := NeuralExp(-sp * FExpA.FData[d]);   // (see #16)
```

**Do this — one index for the group (`#4`), and hoist `t * Depth` to the `t` body
(`#5`):**

```pascal
tDepth := t * Depth;          // #5: once per t, before the d loop
...
idx := tDepth + d;            // #4: once per (t, d)
FDelta.FData[idx] := sp;
FBt.FData[idx]    := acc;
FCt.FData[idx]    := hprev;
FAt.FData[idx]    := NeuralExp(-sp * FExpA.FData[d]);
```

Same for `(t * FNumHeads) + h`, and for a **nested** build like
`hbase := ((t * Depth) + d) * NS` — hoist `tDepth := t * Depth` to the `t` body and
reuse `idx := tDepth + d` before multiplying by `NS`. Whenever the same
`loopvar * const` appears twice, it is this pattern.

### B. Consecutive fixed depth slots — write through one base — rule #11/#4

Writing a small fixed number of depth channels at `[p, 0, 0]`, `[p, 0, 1]`,
`[p, 0, 2]` routes each through the `[]` accessor (a `GetRawPos` multiply-add each).

**Anti-example:**

```pascal
FPhi[p, 0, 0] := 1.0;
FPhi[p, 0, 1] := pn;
FPhi[p, 0, 2] := pcr_sinf(2 * Pi * pn);
```

**Do this — one base, then `+0/+1/+2`:**

```pascal
b := FPhi.GetRawPos(p, 0);        // 2-arg form (#7); the row base
FPhi.FData[b]     := 1.0;
FPhi.FData[b + 1] := pn;
FPhi.FData[b + 2] := pcr_sinf(2 * Pi * pn);
```

### C. Contiguous copies/accumulates are `Move` / `TNNetVolume` ops — rule #13

Reviewers keep flagging plain element loops. Reflexively recognise them:

```pascal
// copy — rule #13:
for c := 0 to DepthM1 do FOutput.FData[baseOut0 + c] := Prev.FData[basePrev0 + c];
//   -> Move(Prev.FData[basePrev0], FOutput.FData[baseOut0], (DepthM1 + 1) * csNeuralFloatSize);

// accumulate — rule #13:
for c := 0 to DepthM1 do
  PrevErr.FData[basePrevErr0 + c] := PrevErr.FData[basePrevErr0 + c] + FOutputError.FData[baseOutErr0 + c];
//   -> TNNetVolume.Add(PrevErr.GetRawPtr(basePrevErr0), FOutputError.GetRawPtr(baseOutErr0), DepthM1 + 1);
```

**This applies to plain `array of TNeuralFloat` too, not only `TNNetVolume`.** Three
parallel raw-array copies are three `Move`s:

```pascal
for i := 0 to nM1 do begin FPosT[i] := pPosT[i]; FPosH[i] := pPosH[i]; FPosW[i] := pPosW[i]; end;
//   -> Move(pPosT[0], FPosT[0], n * SizeOf(TNeuralFloat));  (and FPosH, FPosW)  — n = nM1 + 1
```

A `dot` (`acc += W[k]*x[k]`) is `TNNetVolume.DotProduct`; a `mad`
(`buf[k] += W[k]*s`, `s` invariant) is `TNNetVolume.MulAdd(buf, W, s, N)`; a
`buf[k] := 0` fill is `FillChar`/`TVolume.Fill`. In one `HamiltonianCell` block the
reviewer found all four in a dozen lines — a `Move`, a zero-fill, a `dot`, and a
`mad` — none promoted. Read a scalar `for` over a contiguous run and ask *which
primitive is this* before leaving it.

### D. Elementwise math as a short op sequence — rule #13 (with scratch)

A uniform elementwise formula built from `+ - *` and already-vectorized transcendentals
can often be a short sequence of `TNNetVolume` ops over a scratch buffer:

```pascal
// tanh-approx GELU forward: out = SQRT_2_OVER_PI*(x + GELU_CONST*x^3)
for OutputCnt := 0 to SizeM1 do
begin
  x := LocalPrevOutput.FData[OutputCnt];
  FOutput.FData[OutputCnt] := SQRT_2_OVER_PI * (x + GELU_CONST * x*x*x);
end;
```

is expressible as: copy `x`; cube via elementwise `Mul` into a scratch; `MulAdd`/`Mul`
by the constants — worthwhile when the length is large and the ops are the AVX
primitives. **Precondition is still #13's uniformity:** if the body mixes a
per-element transcendental with data-dependent factors (e.g. a Swish backward
`t + x*(1 - t*t)*dg`), promote only the uniform sub-parts and leave the rest scalar —
do not force a single-call rewrite. If the scratch churn would exceed the scalar
loop's cost (short runs), leave it; note the decision.

### E. Reorder loops so the inner axis is contiguous — rule #13/#11

When the inner loop strides by `Depth` (channel-outer) the accesses are cache-hostile
and cannot be a bulk op; interchanging so the contiguous (depth) axis is innermost
often exposes a `Move`/`Add`/`MulAdd` and removes per-element `GetRawPos`:

```pascal
for c := 0 to DepthM1 do
  for t := 0 to FSeqLenM1 do
    ... Prev[s, 0, c] ...      // strided by Depth in the inner body
```

Consider `for t ... for c ...` (or hoisting a per-`(t,c)` base) so the depth run is
contiguous — but **only when it preserves the dependency structure** (watch
accumulations like `GdH[p,0,c] += ...` whose order across `c`/`t` must stay correct).
Reorder for locality, not blindly.

### F. Initializer loops are not exempt — rules #2/#3/#11

Setup/init code violates the guide too:

```pascal
Depth   := FNeurons[2].FWeights.SizeX;
DepthM1 := Depth - 1;
LpBnd68 := FNeurons[2].FWeights.Depth - 1;
for c := 0 to DepthM1 do
  for j := 0 to LpBnd68 do
    FNeurons[2].FWeights[c, 0, j] := FNeurons[2].FWeights.RandomGaussianValue() * 0.1;
```

`FNeurons[2].FWeights` is re-resolved on every write (bind it to a local, #3/#7); the
`[c, 0, j]` accessor recomputes `GetRawPos` per element when a hoisted base + `Move`-free
inner write would do (#11). Cost is one-time, so this is low priority — but it is
still a violation, and a reviewer will still flag it; apply the same reflexes.
