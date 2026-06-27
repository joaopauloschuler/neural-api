#!/usr/bin/env python3
"""Generate a tiny RANDOM diffusers T2IAdapter (full_adapter) parity fixture for
tests/TestNeuralPretrained.pas (TestT2IAdapterParity).

diffusers is NOT installed in the reusable venv, so the reference forward is a
self-contained numpy float64 oracle that mirrors the CAI importer's forward path
EXACTLY (the make_pico recipe). The weights use the exact diffusers T2IAdapter
key scheme (adapter.conv_in.*, adapter.body.{i}.in_conv.*,
adapter.body.{i}.resnets.{j}.block1/block2.*) so the importer is exercised on a
real key layout.

T2I-Adapter is the lighter sibling of ControlNet: a small conv encoder over a
spatial hint (canny/sketch/depth) producing a PYRAMID of per-resolution feature
maps that get ADDED into the SD UNet down-block hidden state. Architecture
(diffusers FullAdapter):
  * PixelUnshuffle(downscale_factor): (C,H,W) -> (C*f*f, H/f, W/f), reshaping the
    hint onto the latent grid.
  * conv_in: 3x3 pad1 -> channels[0].
  * len(channels) AdapterBlocks. Block 0 keeps the grid; blocks 1.. AvgPool2d(2)
    first. Each block: optional 1x1 in_conv on a channel change, then
    num_res_blocks adapter ResnetBlocks. The output of each block is one feature.
  * adapter ResnetBlock: block1(3x3 pad1) -> ReLU -> block2(1x1) -> + identity.

The N features map 1:1 to the SD UNet's N down blocks and are added into `sample`
before each block's downsampler (diffusers down_intrablock_additional_residuals).

PICO config (architecturally complete, tiny; matches the pico SD UNet latent
grid so the combined add is shape-compatible):
  in_channels (hint)  = 3 (RGB)
  channels            = [16, 32]   (2 down stages, like the pico SD UNet)
  num_res_blocks      = 2
  downscale_factor    = 2          (hint 16x16 -> latent 8x8)
  cond_size           = 16

The CAI importer feeds the unshuffle via TNNetSpaceToDepth whose channel order
((dx*f + dy)*C + ic) differs from torch PixelUnshuffle (ic*f*f + dy*f + dx); the
importer PERMUTES conv_in's input columns at load so the two convolutions are
identical. This oracle computes the TRUE torch PixelUnshuffle + torch-ordered
conv_in, which the permutation makes bit-identical.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/t2i_adapter_tiny_fixture.py
writes tests/fixtures/tiny_t2i_adapter{.safetensors,_config.json,_io.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

# ---- pico config ----
COND_CHANNELS = 3
CHANNELS = [16, 32]
NUM_RES_BLOCKS = 2
DOWNSCALE = 2
COND_GRID = 16            # hint image grid (16/2 -> 8 latent grid)
LATENT = COND_GRID // DOWNSCALE

rng = np.random.default_rng(20260627)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def conv_w(out_ch, in_ch, k, std=0.10):
    return randn(out_ch, in_ch, k, k, std=std)


# ===========================================================================
# numpy float64 oracle.
# ===========================================================================
def conv2d(x, w, b, pad, stride=1):
    I, H, W = x.shape
    O, _, k, _ = w.shape
    Ho = (H - k + 2 * pad) // stride + 1
    Wo = (W - k + 2 * pad) // stride + 1
    xp = np.zeros((I, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    out = np.zeros((O, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            patch = xp[:, oy * stride:oy * stride + k, ox * stride:ox * stride + k]
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3],
                                                          [0, 1, 2])) + b
    return out


def relu(x):
    return np.maximum(x, 0.0)


def pixel_unshuffle(x, r):
    """torch.nn.PixelUnshuffle: (C,H,W) -> (C*r*r, H/r, W/r). Output channel
    oc = ic*r*r + dy*r + dx samples input (oy*r+dy, ox*r+dx)."""
    C, H, W = x.shape
    out = np.zeros((C * r * r, H // r, W // r), dtype=np.float64)
    for ic in range(C):
        for dy in range(r):
            for dx in range(r):
                oc = ic * r * r + dy * r + dx
                out[oc] = x[ic, dy::r, dx::r]
    return out


def avgpool2(x):
    """nn.AvgPool2d(2, stride=2, ceil_mode=True). Pico grids are even so ceil is
    a no-op; plain 2x2 mean pooling."""
    C, H, W = x.shape
    return x.reshape(C, H // 2, 2, W // 2, 2).mean(axis=(2, 4))


def adapter_resnet(x, sd, prefix):
    h = conv2d(x, sd[prefix + 'block1.weight'], sd[prefix + 'block1.bias'], 1)
    h = relu(h)
    h = conv2d(h, sd[prefix + 'block2.weight'], sd[prefix + 'block2.bias'], 0)
    return h + x


def adapter_block(x, sd, i, in_ch, out_ch, down):
    if down:
        x = avgpool2(x)
    if in_ch != out_ch:
        x = conv2d(x, sd[f'adapter.body.{i}.in_conv.weight'],
                   sd[f'adapter.body.{i}.in_conv.bias'], 0)
    for j in range(NUM_RES_BLOCKS):
        x = adapter_resnet(x, sd, f'adapter.body.{i}.resnets.{j}.')
    return x


def forward(hint, sd):
    x = pixel_unshuffle(hint, DOWNSCALE)
    x = conv2d(x, sd['adapter.conv_in.weight'], sd['adapter.conv_in.bias'], 1)
    features = []
    in_ch = CHANNELS[0]
    for i, out_ch in enumerate(CHANNELS):
        x = adapter_block(x, sd, i, in_ch, out_ch, down=(i > 0))
        features.append(x)
        in_ch = out_ch
    return features


def build_state_dict():
    sd = {}
    unshuffled_in = COND_CHANNELS * DOWNSCALE * DOWNSCALE  # 12
    sd['adapter.conv_in.weight'] = conv_w(CHANNELS[0], unshuffled_in, 3)
    sd['adapter.conv_in.bias'] = randn(CHANNELS[0], std=0.05)
    in_ch = CHANNELS[0]
    for i, out_ch in enumerate(CHANNELS):
        if in_ch != out_ch:
            sd[f'adapter.body.{i}.in_conv.weight'] = conv_w(out_ch, in_ch, 1)
            sd[f'adapter.body.{i}.in_conv.bias'] = randn(out_ch, std=0.05)
        for j in range(NUM_RES_BLOCKS):
            p = f'adapter.body.{i}.resnets.{j}.'
            sd[p + 'block1.weight'] = conv_w(out_ch, out_ch, 3)
            sd[p + 'block1.bias'] = randn(out_ch, std=0.05)
            sd[p + 'block2.weight'] = conv_w(out_ch, out_ch, 1)
            sd[p + 'block2.bias'] = randn(out_ch, std=0.05)
        in_ch = out_ch
    return sd


def main():
    sd = build_state_dict()

    # Pinned hint image (deterministic dyadic values; exact in f32 + JSON).
    hint = np.zeros((COND_CHANNELS, COND_GRID, COND_GRID), dtype=np.float64)
    for c in range(COND_CHANNELS):
        for y in range(COND_GRID):
            for x in range(COND_GRID):
                hint[c, y, x] = (((c * 256 + y * 16 + x) * 3) % 17 - 8) / 16.0

    feats = forward(hint, sd)
    print('t2i-adapter features:', [f.shape for f in feats])
    for i, f in enumerate(feats):
        print(f'  feature[{i}] {f.shape} min {f.min():.4f} max {f.max():.4f} '
              f'mean {f.mean():.4f}')

    save_file({k: v.astype(np.float32) for k, v in sd.items()},
              'tests/fixtures/tiny_t2i_adapter.safetensors')

    cfg = {
        '_class_name': 'T2IAdapter',
        'in_channels': COND_CHANNELS,
        'channels': CHANNELS,
        'num_res_blocks': NUM_RES_BLOCKS,
        'downscale_factor': DOWNSCALE,
        'cond_size': COND_GRID,
    }
    with open('tests/fixtures/tiny_t2i_adapter_config.json', 'w') as f:
        json.dump(cfg, f, indent=1)

    io = {
        'cond': hint.tolist(),                          # [C][H][W]
        'features': [f.tolist() for f in feats],        # list of [C][H][W]
        'num_features': len(feats),
    }
    with open('tests/fixtures/tiny_t2i_adapter_io.json', 'w') as f:
        json.dump(io, f)
    print('wrote tests/fixtures/tiny_t2i_adapter{.safetensors,_config.json,'
          '_io.json}')


if __name__ == '__main__':
    main()
