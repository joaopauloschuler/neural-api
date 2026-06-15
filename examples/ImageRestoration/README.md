# NAFNet Image Restoration

Denoises a small image end-to-end on the CPU with the repo's **NAFNet**
image-restoration importer (`BuildNAFNetFromSafeTensors`, `neuralpretrained.pas`)
— the repo's **first non-diffusion image-to-image restoration import that is NOT
super-resolution** (the RRDBNet/ESRGAN path is x4 upscaling only). NAFNet
restores (denoise / deblur / dejpeg) at the **same resolution** (Chen et al.
2022, [*Simple Baselines for Image Restoration*](https://arxiv.org/abs/2204.04676)).

## What's new

* **`TNNetSimpleGate`** — a parameter-free leaf layer: split the channel
  (depth) axis in half and multiply the halves elementwise
  (`out[c] = in[c] · in[C+c]`), a GLU with no activation. Backward is the
  product rule. This is the core nonlinearity of every NAFBlock.
  Numerical-gradient-checked in `tests/TestNeuralNumerical.pas`.
* **Simplified Channel Attention (SCA)** — reuses landed pieces: global average
  pool (`TNNetAvgChannel`) → 1×1 conv → channelwise multiply
  (`TNNetChannelMulByLayer`), no activation.

Everything else reuses landed layers: 1×1 / 3×3 conv, depthwise 3×3 conv,
per-pixel LayerNorm2d (`TNNetTokenLayerNorm`), stride-2 down conv, PixelShuffle
up via `TNNetDepthToSpace`, and residual adds with learnable per-channel
`beta`/`gamma` scales.

## Architecture

```
intro 3x3 -> [enc NAFBlocks -> stride-2 down conv (2x ch)]* -> middle NAFBlocks
           -> [1x1 conv (2x ch) -> PixelShuffle(2) -> + encoder skip
               -> dec NAFBlocks]* -> ending 3x3 -> + input (global skip)
```

NAFBlock(C):

```
x  = LayerNorm2d(inp) -> conv1(1x1,C->2C) -> dwconv2(3x3,2C->2C)
     -> SimpleGate(2C->C) -> x * SCA(x) -> conv3(1x1,C->C)
y  = inp + beta .* x
x2 = LayerNorm2d(y) -> conv4(1x1,C->2C) -> SimpleGate(2C->C) -> conv5(1x1,C->C)
out = y + gamma .* x2
```

## Run

```
./ImageRestoration                          # uses the committed pico fixture
./ImageRestoration model.safetensors [config.json]   # your own checkpoint
```

No network access is required: the official NAFNet checkpoints are large and not
obtainable offline, so — exactly like the repo's RRDBNet / VAE-decoder pico
fixtures — this falls back to the committed **config-faithful random pico
NAFNet** (`tests/fixtures/tiny_nafnet.*`, built by `tools/nafnet_tiny_fixture.py`
and parity-checked `< 1e-4` against a float64 numpy oracle in
`tests/TestNeuralPretrained.pas`, `TestNAFNetParity`). Because the pico weights
are random (not trained to denoise), this is a **wiring / throughput smoke
demo**: it builds the net, adds synthetic Gaussian noise to a deterministic test
image, runs the restoration forward pass, writes `nafnet_noisy.ppm` /
`nafnet_restored.ppm` plus an ASCII preview, and reports the round-trip RMSE.
Point it at a real `.safetensors` (+ `config.json` sibling) to restore with your
own trained checkpoint.

Pure CPU, well under a second on the fixture.

## Config (`config.json`)

| field            | meaning                                  | default |
| ---------------- | ---------------------------------------- | ------- |
| `model_type`     | `"nafnet"`                               | nafnet  |
| `img_channel`    | input/output channels                    | 3       |
| `width`          | base channel width (must be even)        | —       |
| `enc_blk_nums`   | NAFBlocks per encoder stage (list)       | `[1]`   |
| `middle_blk_num` | NAFBlocks in the bottleneck              | 1       |
| `dec_blk_nums`   | NAFBlocks per decoder stage (list)       | `[1]`   |
| `input_size`     | input grid H=W for the `Input` layer     | —       |

`enc_blk_nums` and `dec_blk_nums` must have the same length (one entry per
U-Net level).

Coded by Claude (AI).
