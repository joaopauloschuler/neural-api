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

**Do this — carry the offset by addition:**

```pascal
jDk := FEvictSinks * FDk;              // seed once: the first j * FDk
for j := FEvictSinks to CacheLenM2 do
begin
  jP1Dk := jDk + FDk;                  // (j + 1) * FDk, by addition
  Move(FKCacheCodes[jP1Dk], FKCacheCodes[jDk], RowBytesI8);
  Move(FVCacheCodes[jP1Dk], FVCacheCodes[jDk], RowBytesI8);
  ...
  jDk := jP1Dk;                        // advance: next iteration's j * FDk
end;
```

The single seed multiply outside the loop replaces one multiply *per iteration*;
`jDk` and `jP1Dk` are each computed once and reused for both the K and V rows
(folding in #4's de-duplication).

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
- **Keep the accumulator in sync with the branch that uses it.** Here `jDk` is
  advanced inside the `if FKVQuantInt8` branch, which is safe *only because the
  condition is loop-invariant* (it is the same every iteration, so the branch —
  and the advance — runs on every pass). If the branch could vary per iteration,
  advance the accumulator unconditionally in the loop body instead.
- **Seed correctly.** The initial value must equal the product at the loop's
  first index (`FEvictSinks * FDk`, not `0`), or every subsequent offset is
  wrong.
