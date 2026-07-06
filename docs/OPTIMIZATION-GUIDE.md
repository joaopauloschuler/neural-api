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
