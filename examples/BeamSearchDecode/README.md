# BeamSearchDecode — a decoding-strategy bake-off

A decoding-strategy bake-off that shows **why beam search beats greedy argmax**.
A tiny char-level next-token model is trained on a SYNTHETIC corpus deliberately
built so that **greedy dead-ends**: right after the prompt the locally-likeliest
first token leads into a globally WORSE (lower total log-prob) continuation,
while a slightly-less-likely first token opens a strongly-deterministic,
high-probability long tail. Beam search keeps both branches alive and recovers
the higher TOTAL log-prob sequence.

## The trap corpus

Two branches fork off the prompt `"a"`:

- `ab{k,l}<EOS>` — `b` is locally the slightly-most-likely first move (48
  occurrences), BUT the `ab` branch is **high-entropy**: it forks into two
  equiprobable continuations (`k`, `l`), so every `ab?` continuation pays
  `~ln(1/2)` of log-prob and no `ab...` sequence can reach a high cumulative
  log-prob. The two equal forks make `p(k|ab)=p(l|ab)=0.5` the unique
  cross-entropy optimum, so a converged net cannot collapse it.
- `acdefgh<EOS>` — `c` is locally just below `b` (42 occurrences, so a single
  argmax skips it), but `ac` opens a long **near-deterministic** tail where every
  step has prob `~1.0` (`~0` log-prob cost), giving a far higher cumulative
  log-prob.

Greedy commits to `b` and pays the entropy tax forever; beam (`B>=2`) keeps the
`c` branch alive and recovers the globally better tail. The `b:c` first-token
ratio is kept close (`48:42`) so both survive into the beam — it is the
downstream entropy-vs-determinism gap that decides the winner.

## The model

A tiny char-level MLP over a one-hot context window (`csContextLen=8`,
`csVocabSize=128`):

```
Input(8, 1, 128)
 -> FullConnectReLU(48)
 -> FullConnectReLU(48)
 -> FullConnectLinear(128)
 -> SoftMax
```

Every `(prefix, next-char)` pair from every corpus line is flattened into a
training sample. Trained with `TNeuralFit` (`lr=0.005`, batch 8, 90 epochs over
the replicated corpus), it converges in well under a minute on CPU.

## What it demonstrates

Decoding strategies from `neural/neuraldecode.pas`:

- `DecodeGreedy` — single argmax per step.
- `DecodeBeamSearch(NN, Prompt, MaxLen, BeamWidth, alpha)` — beam search with a
  Wu et al. 2016 length penalty (`alpha`). Run at `B=2,4,8`.
- `DecodeBeamSearchAll` — returns the full ranked beam pool
  (`TNNetDecodeResultArray`), finished + survivors, with `SumLogProb`, `Score`
  and `Finished` flags.
- `TNNetSamplerTopK` / `TNNetSamplerTopP` (via `GenerateStringFromChars`) for the
  diversity contrast.

The program prints:

1. **Probes** of the trained `P(next|"a")`, `P(next|"ab")`, `P(next|"ac")` to
   confirm the trap is in place (argmax after `"a"` is `b`).
2. A **Greedy vs Beam(2,4,8)** table with total log-prob and length-penalised
   score, asserting beam recovered a higher total log-prob than greedy.
3. A **length-penalty contrast** at `B=4`: `alpha=0` (raw sum, short-biased) vs
   `alpha=0.7` (divides by `((5+L)/6)^alpha`, lifting longer tails).
4. The **final ranked beam** (top 4 of `DecodeBeamSearchAll`).
5. A **diversity contrast**: beam is sharp/deterministic (6 draws give the same
   sequence) while `TopK(3)` / `TopP(0.9)` re-sample per token and wander.

## How to run

```
cd examples/BeamSearchDecode
fpc -O3 -Mobjfpc -Sh -Fu../../neural BeamSearchDecode.lpr
./BeamSearchDecode
```

(or open `BeamSearchDecode.lpi` in Lazarus). Pure CPU, trains in-process in well
under a minute. No binaries are committed.

## Expected output

After training, the greedy decode follows `a -> b -> {k|l}` (the high-entropy
trap), while beam search returns the `acdefgh<EOS>` tail with a strictly higher
`sum_logp`, and the program prints `RESULT: beam recovered a HIGHER total
log-prob sequence than greedy.`. The length-penalty table shows `alpha>0` lifting
the longer, content-bearing tail, and the diversity contrast shows beam repeating
identically while the stochastic samplers vary across draws.
