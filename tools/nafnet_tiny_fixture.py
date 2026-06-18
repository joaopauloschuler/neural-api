#!/usr/bin/env python3
"""Generate a tiny RANDOM NAFNet image-restoration parity fixture for
tests/TestNeuralPretrained.pas.

NAFNet (Chen et al. 2022, "Simple Baselines for Image Restoration",
https://arxiv.org/abs/2204.04676) is a U-Net of NAFBlocks. There is NO official
tiny checkpoint, so the reference forward is a self-contained numpy float64
oracle that mirrors the CAI importer's forward path EXACTLY (same idiom as
tools/rrdbnet_tiny_fixture.py). Weights use the canonical NAFNet state_dict key
scheme so the importer is exercised on a realistic key layout.

NAFBlock(C):
  LayerNorm2d (per-pixel over channels, learnable gamma/beta [C])
    -> conv1 (1x1, C->2C, +bias)
    -> conv2 (3x3 depthwise, 2C->2C, pad1, groups=2C, +bias)
    -> SimpleGate (split depth in half, multiply: 2C -> C)
    -> SCA: avgpool(spatial) -> conv (1x1, C->C, +bias) -> channel multiply
    -> conv3 (1x1, C->C, +bias)
  y = inp + x * beta          (beta learnable per-channel [C], init 1 here)
  LayerNorm2d
    -> conv4 (1x1, C->2C, +bias)
    -> SimpleGate (2C -> C)
    -> conv5 (1x1, C->C, +bias)
  out = y + x2 * gamma        (gamma learnable per-channel [C])

U-Net (this pico): width W
  intro   (3x3, in->W, +bias)
  enc[0]: NB NAFBlocks @ W
  down0   (conv 2x2 stride2, W->2W, no pad, +bias)          # H/2
  middle: NB NAFBlocks @ 2W
  up0     (conv 1x1, 2W->4W, +bias) -> PixelShuffle(2) -> W  # H
          decoder input = up0 + enc[0] output     (skip ADD)
  dec[0]: NB NAFBlocks @ W
  ending  (3x3, W->in, +bias)
  out = ending(...) + inp                                    # global residual

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/nafnet_tiny_fixture.py
writes tests/fixtures/tiny_nafnet{.safetensors,_config.json,_io.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

WIDTH = 6          # base channel width (must be even for SimpleGate)
ENC_BLOCKS = 1     # NAFBlocks per encoder stage
MIDDLE_BLOCKS = 1  # NAFBlocks in the bottleneck
DEC_BLOCKS = 1     # NAFBlocks per decoder stage
IN_CH = 3
INPUT = 8          # input grid H=W (even, so /2 then *2 is exact)
LN_EPS = 1e-5

rng = np.random.default_rng(20260615)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


# ---------------------------------------------------------------------------
# State dict (canonical NAFNet keys).
# ---------------------------------------------------------------------------
def add_conv(sd, name, out_ch, in_ch, k, std=0.15):
    sd[name + '.weight'] = randn(out_ch, in_ch, k, k, std=std)
    sd[name + '.bias'] = randn(out_ch, std=0.05)


def add_dwconv(sd, name, ch, k, std=0.15):
    sd[name + '.weight'] = randn(ch, 1, k, k, std=std)
    sd[name + '.bias'] = randn(ch, std=0.05)


def add_nafblock(sd, prefix, C):
    # norm1 / norm2 gamma+beta
    sd[prefix + 'norm1.weight'] = randn(C, std=0.3) + 1.0
    sd[prefix + 'norm1.bias'] = randn(C, std=0.1)
    sd[prefix + 'norm2.weight'] = randn(C, std=0.3) + 1.0
    sd[prefix + 'norm2.bias'] = randn(C, std=0.1)
    add_conv(sd, prefix + 'conv1', 2 * C, C, 1)
    add_dwconv(sd, prefix + 'conv2', 2 * C, 3)
    add_conv(sd, prefix + 'sca.1', C, C, 1)   # SCA 1x1 conv
    add_conv(sd, prefix + 'conv3', C, C, 1)
    add_conv(sd, prefix + 'conv4', 2 * C, C, 1)
    add_conv(sd, prefix + 'conv5', C, C, 1)
    sd[prefix + 'beta'] = randn(C, std=0.3) + 1.0
    sd[prefix + 'gamma'] = randn(C, std=0.3) + 1.0


def build_state_dict():
    sd = {}
    W = WIDTH
    add_conv(sd, 'intro', W, IN_CH, 3)
    for i in range(ENC_BLOCKS):
        add_nafblock(sd, f'encoders.0.{i}.', W)
    # down: 2x2 stride2 conv, W -> 2W
    sd['downs.0.weight'] = randn(2 * W, W, 2, 2, std=0.15)
    sd['downs.0.bias'] = randn(2 * W, std=0.05)
    for i in range(MIDDLE_BLOCKS):
        add_nafblock(sd, f'middle_blks.{i}.', 2 * W)
    # up: 1x1 conv 2W -> 4W, then pixelshuffle(2) -> W
    sd['ups.0.weight'] = randn(4 * W, 2 * W, 1, 1, std=0.15)
    sd['ups.0.bias'] = randn(4 * W, std=0.05)
    for i in range(DEC_BLOCKS):
        add_nafblock(sd, f'decoders.0.{i}.', W)
    add_conv(sd, 'ending', IN_CH, W, 3)
    return sd


# ---------------------------------------------------------------------------
# numpy float64 oracle (volumes (C,H,W); conv weights [O,I,kh,kw]).
# ---------------------------------------------------------------------------
def conv2d(x, w, b, pad, stride=1):
    I, H, Wd = x.shape
    O, _, k, _ = w.shape
    xp = np.zeros((I, H + 2 * pad, Wd + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + Wd] = x
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (Wd + 2 * pad - k) // stride + 1
    out = np.zeros((O, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            patch = xp[:, oy * stride:oy * stride + k, ox * stride:ox * stride + k]
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3],
                                                          [0, 1, 2])) + b
    return out


def dwconv2d(x, w, b, pad):
    # depthwise: w [C,1,k,k], output channel c uses only input channel c.
    C, H, Wd = x.shape
    k = w.shape[2]
    xp = np.zeros((C, H + 2 * pad, Wd + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + Wd] = x
    out = np.zeros((C, H, Wd), dtype=np.float64)
    for c in range(C):
        for oy in range(H):
            for ox in range(Wd):
                patch = xp[c, oy:oy + k, ox:ox + k]
                out[c, oy, ox] = np.sum(w[c, 0] * patch) + b[c]
    return out


def layernorm2d(x, g, beta, eps=LN_EPS):
    # per-pixel normalization over channel axis (axis 0 of (C,H,W)).
    mean = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)           # population variance
    xn = (x - mean) / np.sqrt(var + eps)
    return xn * g[:, None, None] + beta[:, None, None]


def simple_gate(x):
    C = x.shape[0] // 2
    return x[:C] * x[C:]


def pixel_shuffle(x, r):
    # CAI TNNetDepthToSpace mapping (the importer uses this, NOT nn.PixelShuffle):
    #   out[X=ix*r+sx, Y=iy*r+sy, ic] = in[X=ix, Y=iy, (sx*r+sy)*C + ic]
    # In the (C, H=Y, W=X) oracle layout the X offset is sx and the Y offset sy:
    #   out[ic, iy*r+sy, ix*r+sx] = in[(sx*r+sy)*C + ic, iy, ix]
    C2, H, Wd = x.shape
    C = C2 // (r * r)
    out = np.zeros((C, H * r, Wd * r), dtype=np.float64)
    for iy in range(H):
        for ix in range(Wd):
            for sx in range(r):
                for sy in range(r):
                    for ic in range(C):
                        out[ic, iy * r + sy, ix * r + sx] = \
                            x[(sx * r + sy) * C + ic, iy, ix]
    return out


def nafblock(inp, sd, prefix):
    C = inp.shape[0]
    x = layernorm2d(inp, sd[prefix + 'norm1.weight'], sd[prefix + 'norm1.bias'])
    x = conv2d(x, sd[prefix + 'conv1.weight'], sd[prefix + 'conv1.bias'], 0)
    x = dwconv2d(x, sd[prefix + 'conv2.weight'], sd[prefix + 'conv2.bias'], 1)
    x = simple_gate(x)
    # SCA
    pooled = x.mean(axis=(1, 2)).reshape(C, 1, 1)
    sca = conv2d(pooled, sd[prefix + 'sca.1.weight'], sd[prefix + 'sca.1.bias'], 0)
    x = x * sca                          # broadcast per-channel
    x = conv2d(x, sd[prefix + 'conv3.weight'], sd[prefix + 'conv3.bias'], 0)
    y = inp + x * sd[prefix + 'beta'][:, None, None]

    x2 = layernorm2d(y, sd[prefix + 'norm2.weight'], sd[prefix + 'norm2.bias'])
    x2 = conv2d(x2, sd[prefix + 'conv4.weight'], sd[prefix + 'conv4.bias'], 0)
    x2 = simple_gate(x2)
    x2 = conv2d(x2, sd[prefix + 'conv5.weight'], sd[prefix + 'conv5.bias'], 0)
    out = y + x2 * sd[prefix + 'gamma'][:, None, None]
    return out


def forward(inp, sd):
    x = conv2d(inp, sd['intro.weight'], sd['intro.bias'], 1)
    for i in range(ENC_BLOCKS):
        x = nafblock(x, sd, f'encoders.0.{i}.')
    enc_skip = x
    x = conv2d(x, sd['downs.0.weight'], sd['downs.0.bias'], 0, stride=2)
    for i in range(MIDDLE_BLOCKS):
        x = nafblock(x, sd, f'middle_blks.{i}.')
    x = conv2d(x, sd['ups.0.weight'], sd['ups.0.bias'], 0)
    x = pixel_shuffle(x, 2)
    x = x + enc_skip
    for i in range(DEC_BLOCKS):
        x = nafblock(x, sd, f'decoders.0.{i}.')
    x = conv2d(x, sd['ending.weight'], sd['ending.bias'], 1)
    return x + inp


def main():
    sd = build_state_dict()
    # Round-trip every weight through float32 (CAI loads float32).
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    # Pinned input image: deterministic dyadic values (exact in f32 + JSON).
    x = np.zeros((IN_CH, INPUT, INPUT), dtype=np.float64)
    for c in range(IN_CH):
        for y in range(INPUT):
            for px in range(INPUT):
                x[c, y, px] = (((c * 64 + y * 8 + px) * 5) % 13 - 6) / 8.0

    img = forward(x, sd)
    print(f'input {x.shape} -> image {img.shape}')
    print(f'image stats: min {img.min():.4f} max {img.max():.4f} '
          f'mean {img.mean():.4f}')

    save_file(sd_f32, 'tests/fixtures/tiny_nafnet.safetensors')
    config = {
        'model_type': 'nafnet',
        'img_channel': IN_CH,
        'width': WIDTH,
        'enc_blk_nums': [ENC_BLOCKS],
        'middle_blk_num': MIDDLE_BLOCKS,
        'dec_blk_nums': [DEC_BLOCKS],
        'input_size': INPUT,
    }
    with open('tests/fixtures/tiny_nafnet_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_nafnet_io.json', 'w') as f:
        json.dump({
            'input': x.tolist(),
            'image': img.tolist(),
            'image_size': img.shape[1],
        }, f)
    print(f'wrote tiny_nafnet.safetensors ({len(sd_f32)} tensors) + config + io')

    # ---- fixture self-checks: each major piece must MATTER. ----
    base = img.copy()

    def perturb(key):
        alt = dict(sd)
        alt[key] = np.zeros_like(sd[key])
        return np.abs(forward(x, alt) - base).max()

    for key in ('encoders.0.0.sca.1.weight', 'middle_blks.0.conv2.weight',
                'downs.0.weight', 'ups.0.weight',
                'decoders.0.0.gamma', 'encoders.0.0.beta'):
        d = perturb(key)
        assert d > 1e-4, f'{key} had no effect ({d})'
        print(f'{key} effect: max|diff| = {d:.4f}')

    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
