# HellaSwagEval example

End-to-end multiple-choice scoring demo for `EvaluateMultipleChoice`
(`neural/neuralnlpmetrics.pas`) — the lm-evaluation-harness HellaSwag / ARC /
PIQA scoring pattern: for each item, score every candidate completion of a
shared context with `ScoreCompletion`, let the argmax win, and report

- **acc** — gold wins by **sum** of completion log-probs (lm-eval `acc`,
  short-biased), and
- **acc_norm** — gold wins by **mean** (length-normalized) log-prob
  (lm-eval `acc_norm`).

To stay within a tiny CPU / memory budget (no multi-GB checkpoint download)
this demo trains a small **char-level** autoregressive next-token model on a
handful of toy bigram phrases, then scores four hand-written multiple-choice
items through the **same** `EvaluateMultipleChoice` path an imported SmolLM2 /
pythia checkpoint would use. The scoring code is checkpoint-agnostic: swap the
toy `BuildModel`/`TrainModel` for a `BuildLlamaFromSafeTensors` (or any
safetensors / GGUF importer) and feed `TNeuralHFTokenizer`-produced token ids
into the same `TNNetMultipleChoiceItem` records — the harness is unchanged.

> The char-level model uses the single-next-token head, so the input encoding
> must match the one `ScoreSequence` uses internally
> (`CopyReversedNoChecksIntArr`: zero-filled window, prefix laid in reversed
> order). The training loop replicates that encoding exactly; mismatching it is
> the classic train/eval-skew bug that drives accuracy to chance.

Expected output: the toy model learns the bigrams perfectly, so both metrics
report `1.0000` (4 / 4). Pure CPU, well under a minute, a few MB RAM.

## Build

```
lazbuild HellaSwagEval.lpi
./HellaSwagEval
```

Or, without Lazarus installed (run under a memory cap to be safe):

```
fpc -Mobjfpc -Sh -O3 -Fu../../neural HellaSwagEval.lpr
ulimit -v 3000000
./HellaSwagEval
```
