# Optimization Guide

A running list of recommendations for keeping the neural-api codebase fast,
consistent, and maintainable. Each entry states the problem, the fix, and why
it matters. Recommendations are added incrementally as we discover them.

## 1. Prefer named size constants over `SizeOf(type)`

Replace repeated `SizeOf(TNeuralFloat)` (and similar element-size expressions)
with the predefined constant `csNeuralFloatSize`, declared in
`neural/neuralvolume.pas`:

```pascal
const
  csNeuralFloatSize = SizeOf(TNeuralFloat);
```

**Do this:**

```pascal
Move(Src^, Dst^, Count * csNeuralFloatSize);
```

**Instead of:**

```pascal
Move(Src^, Dst^, Count * SizeOf(TNeuralFloat));
```

**Why it matters**

- **Consistency** — one canonical name for the element size across the whole
  codebase, so `Move`/allocation/stride arithmetic all read the same way.
- **Single source of truth** — if the float type ever changes, the intent stays
  expressed through one named constant rather than scattered `SizeOf` literals.
- **Readability** — `csNeuralFloatSize` signals *"one neural-float element"* at
  a glance, which is clearer than a bare `SizeOf` inside a byte-count
  expression.

**Caveat: declaration order in Pascal.** A constant cannot be referenced before
it is declared. The `csNeuralFloatSize` definition itself, and the
`TNeuralFloatArr = array[0..Maxint div SizeOf(TNeuralFloat)] of ...` type
declarations that precede it in `neuralvolume.pas`, must keep the literal
`SizeOf(TNeuralFloat)`. Everything after the `const` block — and every other
unit that `uses neuralvolume` — should use `csNeuralFloatSize`.

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
