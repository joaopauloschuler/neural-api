# MMLUEval example

End-to-end demo for the MMLU few-shot accuracy harness `EvaluateMMLU` /
`MMLUReport` (`neural/neuralnlpmetrics.pas`). MMLU (Hendrycks et al. 2021) is
the canonical 4-choice (A/B/C/D) knowledge benchmark; the HF
lm-evaluation-harness scores it by the log-probability of the **single
answer-letter token** that follows the standard k-shot prompt

```
<k same-subject demos, each "Question..\nA. ..\nB. ..\nC. ..\nD. ..\nAnswer: X\n\n">
Question ..
A. ..
B. ..
C. ..
D. ..
Answer:
```

— **not** by the perplexity of the whole answer string. That latter,
full-continuation pattern is what [`../HellaSwagEval`](../HellaSwagEval)
demonstrates via `EvaluateMultipleChoice`; the two scoring modes are kept
clearly separate. This example shows **both 0-shot and 5-shot** modes (5-shot
is the headline MMLU setting) and reports **per-subject** accuracy plus the
**macro-average** (mean over subjects, the headline MMLU number) and the
**micro-average** (pooled over questions).

To stay within a tiny CPU / memory budget (no network fetch, no multi-GB
download) this demo trains a small **char-level** autoregressive next-token
model on a **tiny embedded smoke subset** (two toy "subjects") and runs in a
few seconds under the 3 GB ulimit. The goal is the **harness mechanics**, not
a real accuracy number. The scoring path is checkpoint-agnostic: swap the toy
`BuildModel`/`TrainModel` for a `BuildLlamaFromSafeTensors` (or any
safetensors / GGUF importer) and feed `TNeuralHFTokenizer`-produced token ids
of the prompt and the four `" A".." D"` answer-letter tokens into the same
`TNNetMMLUQuestion` records — the harness is unchanged.

> The char-level model uses the single-next-token head, so the input encoding
> must match the one `ScoreSequence` uses internally
> (`CopyReversedNoChecksIntArr`: zero-filled window, prefix laid in reversed
> order). The training loop replicates that encoding exactly; mismatching it is
> the classic train/eval-skew bug that drives accuracy to chance.
>
> `EvaluateMMLU` is called with `LastWindow = true`: real MMLU k-shot prompts
> routinely exceed any fixed context window, so over-context prompts are scored
> over the model's trailing context window (the standard sliding-window LM
> eval) instead of raising. The trailing window always contains the `Answer: `
> region that determines the letter, and training clipped its prefix to the
> same window.

Expected output: the toy model learns the answer-letter mapping perfectly, so
both modes report `1.0000` (4 / 4). Pure CPU, a few seconds, a few MB RAM.

## Running against the real cais/mmlu dataset

This smoke build hard-codes its questions so it needs no fetch. To score the
real `cais/mmlu` (a.k.a. `hendrycks_test`) splits, dump the dev (few-shot) and
test splits to a small text file with the `datasets` package and feed the
questions through the same `FormatQuestion` / `BuildPrompt` builder plus a real
subword tokenizer. A `--full <path>` hook is a documented follow-up (see
`tasklist.md`).
