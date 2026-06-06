# PerplexityEval example

Runs `TNNet.PerplexityReport` on a tiny char-level next-token model trained
briefly on a perfectly-periodic 8-symbol alphabet (`abcdefgh...`), then
repeats the run with a `TNNetLogSoftMax` head to exercise the auto-detected
log-space code path.

The report prints, per evaluation:

- per-token mean cross-entropy in nats and bits,
- perplexity `exp(mean_CE)`,
- bits-per-character (BPC),
- top-1 and top-5 accuracy over the stream,
- a 10-bin ASCII histogram of per-token bits (long right tail = rare-token
  spikes),
- the K worst-predicted positions (token id, context window, bits).

Both heads should produce nearly identical perplexity, well below the
uniform baseline of 8.

## Build

```
lazbuild PerplexityEval.lpi
./PerplexityEval
```

Or, without Lazarus installed:

```
fpc -Mobjfpc -Sh -O3 -Fu../../neural PerplexityEval.lpr
```
