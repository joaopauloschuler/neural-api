# VARGenerate: coarse-to-fine visual autoregressive generation

An offline, CPU **VAR** (Visual AutoRegressive, *next-scale* prediction; Tian et
al. 2024, [arXiv:2404.02905](https://arxiv.org/abs/2404.02905)) image-generation
loop, end to end, over the repo's already-landed importers. Nothing here is a new
leaf layer — it is pure plumbing over `BuildVARFromSafeTensors`, the `VARGenerate`
/ `DecodeVARTokensToImage` helpers, and `BuildVqModelFromSafeTensors`.

The pipeline:

```
class label y
  -> class-conditional VAR transformer (BuildVARFromSafeTensors)
  -> next-scale autoregressive sampling loop (VARGenerate): for each pyramid
     level s, run the forward over the partially-filled multi-scale token
     sequence, read the next-scale logits at scale s, sample/argmax the
     tokens, write them back so finer scales attend to them
  -> the final scale's token grid is a VQ token map
  -> residual/discrete VQ decode to pixels (DecodeVARTokensToImage)
  -> RGB image -> P6 PPM.
```

## Build / run

```
cd examples/VARGenerate
lazbuild VARGenerate.lpi --build-mode=Release       # program VARGenerateDemo, binary VARGenerate
# run from the repo root so the fixture path resolves:
../../bin/x86_64-linux/bin/VARGenerate
../../bin/x86_64-linux/bin/VARGenerate var.safetensors vq.safetensors
```

Optional trailing flags: `--class N` (class label, default 0), `--temp T`
(sampling temperature, default 0 = greedy argmax), `--seed N` (RNG seed, default
424242; only matters when `--temp > 0`), `--smoke` (run, assert finiteness, print
`SMOKE OK`/`SMOKE FAIL` and exit without writing a PPM).

## Input

Real checkpoints are not obtainable offline, so by default it falls back to the
committed pico VAR (`tests/fixtures/tiny_var.*`) and the **matched** pico VQModel
(`tests/fixtures/tiny_var_vqmodel.*`), sized so the VAR final `3×3` token map is
exactly the VQ token grid. The pico fixtures have **random** weights, so this is a
**wiring/throughput smoke** — it proves the multi-scale loop + VQ decode run
offline and produce a finite image, not a real picture. Pass your own
`var.safetensors vq.safetensors` (with sibling `config.json` files) to use real
checkpoints.

> Faithfulness note: canonical FoundationVision/var carries the cross-scale
> residual-VQ feature accumulation INSIDE the VQ tokenizer; this importer's input
> contract is plain codebook indices, so the coarse→fine information flow is
> carried purely by the transformer attention over already-sampled coarser tokens.

## Output

Writes `var_generate.ppm` (the decoded RGB image) and prints the configs, the
sampled per-scale token maps and an ASCII preview. Regression-tested by
`TestVARGenerateSmoke`. Pure CPU.
