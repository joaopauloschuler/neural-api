#!/usr/bin/env python3
"""Generate a FULL torchvision inception_v3 parity fixture for
tests/TestNeuralPretrained.pas.

torchvision is NOT installed in the reusable venv, so -- exactly like
tools/inceptionv3_tiny_fixture.py and tools/resnet18_tiny_fixture.py -- the
reference logits are computed by a self-contained numpy float64 oracle that
replicates the CAI forward path bit-for-bit (the SAME conv-BN fold + CAI
maxpool floor semantics + count_include_pad=False average pool + channel
concat). The state_dict uses the EXACT torchvision inception_v3 key scheme and
the EXACT torchvision channel widths, so the importer (BuildInceptionV3Full)
loads a real-shaped checkpoint and must match the oracle < 1e-4.

WHAT IS EXERCISED (the full module sequence, the genuinely-new work):
  * full stem: Conv2d_1a_3x3 (stride2) .. Conv2d_4a_3x3 + two stride-2 maxpools.
  * InceptionA x3 (Mixed_5b/5c/5d), size-preserving, with the REAL
    count_include_pad=False avg-pool branch (TNNetGridAvgPool), not the pico
    maxpool stand-in.
  * InceptionB (Mixed_6a) + InceptionD (Mixed_7a): strided grid reductions
    (parallel stride-2 conv + stride-2 maxpool branches).
  * InceptionC x4 (Mixed_6b/6c/6d/6e): 7x7-factorized asymmetric 1x7 / 7x1
    convs with asymmetric (0,3)/(3,0) padding.
  * InceptionE x2 (Mixed_7b/7c): the wide split (1x3 || 3x1) module, 2048-d out.
  * conv-BN fold (eps 1e-3) on every BasicConv2d; global avg pool -> fc.

The image is SMALL (so the test is cheap) but every channel width is the real
torchvision width -- the importer's hardcoded widths must line up exactly.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/inceptionv3_full_fixture.py
writes tests/fixtures/tiny_inceptionv3_full{.safetensors,_config.json,_logits.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

EPS = 1e-3
NUM_CHANNELS = 3
NUM_CLASSES = 4
# Small input that keeps a >=3x3 grid through the InceptionE modules (so the
# 3x3 convs there are genuine 3x3, not clamped to the input spatial size by
# CAI's TNNetConvolution kernel auto-shrink) and a >=7x7 grid through the
# 7x7-factorized InceptionC modules. IMAGE=160 -> stem 17, afterB 8 (C grid),
# E grid 3. Real torchvision is 299 (C grid 17, E grid 8); 160 is the smallest
# cheap size that exercises every kernel at full size.
IMAGE = 160

# Channel-width divisor: the real torchvision inception_v3 is ~24M params
# (~90MB of f32 weights) -- far too big to commit as a test fixture. We scale
# EVERY channel width down by WIDTH_DIV (topology / kernels / concat / asymmetric
# convs all unchanged) so the committed safetensors stays ~1MB. The importer's
# config carries the SAME width_div, so a real width_div=1 (2048-d) checkpoint
# loads identically. The canonical widths are all multiples of 32, so WIDTH_DIV
# in {1,2,4,8,16,32} divides exactly.
WIDTH_DIV = 8

rng = np.random.default_rng(20260626)


def w(n):
    """Scale a torchvision channel width by WIDTH_DIV (exact for 32-multiples)."""
    return max(1, n // WIDTH_DIV)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def make_bn(prefix, sd, channels):
    sd[prefix + '.weight'] = randn(channels, std=0.1) + 0.8
    sd[prefix + '.bias'] = randn(channels, std=0.1)
    sd[prefix + '.running_mean'] = randn(channels, std=0.1)
    sd[prefix + '.running_var'] = np.abs(randn(channels, std=0.2)) + 0.8
    sd[prefix + '.num_batches_tracked'] = np.array(0, dtype=np.int64)


def conv(prefix, sd, out_ch, in_ch, kh, kw, std=0.08):
    # Scale weight std down by sqrt(fan-in) so activations stay O(1) through all
    # 11 modules (keeps the absolute-tolerance parity check meaningful instead
    # of comparing 1e7-scale logits). Plus a BN that does not blow up.
    fan_in = in_ch * kh * kw
    sd[prefix + '.conv.weight'] = randn(out_ch, in_ch, kh, kw,
                                        std=1.4 / np.sqrt(fan_in))
    make_bn(prefix + '.bn', sd, out_ch)


# --------------------------------------------------------------------------
# state_dict in the exact torchvision inception_v3 key scheme + widths.
# --------------------------------------------------------------------------
def inception_a(prefix, sd, in_ch, pool_features):
    conv(prefix + '.branch1x1', sd, w(64), in_ch, 1, 1)
    conv(prefix + '.branch5x5_1', sd, w(48), in_ch, 1, 1)
    conv(prefix + '.branch5x5_2', sd, w(64), w(48), 5, 5)
    conv(prefix + '.branch3x3dbl_1', sd, w(64), in_ch, 1, 1)
    conv(prefix + '.branch3x3dbl_2', sd, w(96), w(64), 3, 3)
    conv(prefix + '.branch3x3dbl_3', sd, w(96), w(96), 3, 3)
    conv(prefix + '.branch_pool', sd, w(pool_features), in_ch, 1, 1)
    return w(64) + w(64) + w(96) + w(pool_features)


def inception_b(prefix, sd, in_ch):
    conv(prefix + '.branch3x3', sd, w(384), in_ch, 3, 3)
    conv(prefix + '.branch3x3dbl_1', sd, w(64), in_ch, 1, 1)
    conv(prefix + '.branch3x3dbl_2', sd, w(96), w(64), 3, 3)
    conv(prefix + '.branch3x3dbl_3', sd, w(96), w(96), 3, 3)
    return w(384) + w(96) + in_ch


def inception_c(prefix, sd, in_ch, c7):
    conv(prefix + '.branch1x1', sd, w(192), in_ch, 1, 1)
    conv(prefix + '.branch7x7_1', sd, w(c7), in_ch, 1, 1)
    conv(prefix + '.branch7x7_2', sd, w(c7), w(c7), 1, 7)
    conv(prefix + '.branch7x7_3', sd, w(192), w(c7), 7, 1)
    conv(prefix + '.branch7x7dbl_1', sd, w(c7), in_ch, 1, 1)
    conv(prefix + '.branch7x7dbl_2', sd, w(c7), w(c7), 7, 1)
    conv(prefix + '.branch7x7dbl_3', sd, w(c7), w(c7), 1, 7)
    conv(prefix + '.branch7x7dbl_4', sd, w(c7), w(c7), 7, 1)
    conv(prefix + '.branch7x7dbl_5', sd, w(192), w(c7), 1, 7)
    conv(prefix + '.branch_pool', sd, w(192), in_ch, 1, 1)
    return w(192) * 4


def inception_d(prefix, sd, in_ch):
    conv(prefix + '.branch3x3_1', sd, w(192), in_ch, 1, 1)
    conv(prefix + '.branch3x3_2', sd, w(320), w(192), 3, 3)
    conv(prefix + '.branch7x7x3_1', sd, w(192), in_ch, 1, 1)
    conv(prefix + '.branch7x7x3_2', sd, w(192), w(192), 1, 7)
    conv(prefix + '.branch7x7x3_3', sd, w(192), w(192), 7, 1)
    conv(prefix + '.branch7x7x3_4', sd, w(192), w(192), 3, 3)
    return w(320) + w(192) + in_ch


def inception_e(prefix, sd, in_ch):
    conv(prefix + '.branch1x1', sd, w(320), in_ch, 1, 1)
    conv(prefix + '.branch3x3_1', sd, w(384), in_ch, 1, 1)
    conv(prefix + '.branch3x3_2a', sd, w(384), w(384), 1, 3)
    conv(prefix + '.branch3x3_2b', sd, w(384), w(384), 3, 1)
    conv(prefix + '.branch3x3dbl_1', sd, w(448), in_ch, 1, 1)
    conv(prefix + '.branch3x3dbl_2', sd, w(384), w(448), 3, 3)
    conv(prefix + '.branch3x3dbl_3a', sd, w(384), w(384), 1, 3)
    conv(prefix + '.branch3x3dbl_3b', sd, w(384), w(384), 3, 1)
    conv(prefix + '.branch_pool', sd, w(192), in_ch, 1, 1)
    return w(320) + w(384) + w(384) + w(384) + w(384) + w(192)


def build_state_dict():
    sd = {}
    conv('Conv2d_1a_3x3', sd, w(32), NUM_CHANNELS, 3, 3)
    conv('Conv2d_2a_3x3', sd, w(32), w(32), 3, 3)
    conv('Conv2d_2b_3x3', sd, w(64), w(32), 3, 3)
    conv('Conv2d_3b_1x1', sd, w(80), w(64), 1, 1)
    conv('Conv2d_4a_3x3', sd, w(192), w(80), 3, 3)
    ch = w(192)
    ch = inception_a('Mixed_5b', sd, ch, 32)
    ch = inception_a('Mixed_5c', sd, ch, 64)
    ch = inception_a('Mixed_5d', sd, ch, 64)
    ch = inception_b('Mixed_6a', sd, ch)
    ch = inception_c('Mixed_6b', sd, ch, 128)
    ch = inception_c('Mixed_6c', sd, ch, 160)
    ch = inception_c('Mixed_6d', sd, ch, 160)
    ch = inception_c('Mixed_6e', sd, ch, 192)
    ch = inception_d('Mixed_7a', sd, ch)
    ch = inception_e('Mixed_7b', sd, ch)
    ch = inception_e('Mixed_7c', sd, ch)
    sd['fc.weight'] = randn(NUM_CLASSES, ch, std=0.05)
    sd['fc.bias'] = randn(NUM_CLASSES, std=0.05)
    return sd, ch


# --------------------------------------------------------------------------
# numpy float64 oracle, replicating the CAI forward path exactly.
# Volumes (C,H,W); conv weights torchvision [O,I,kh,kw]; BN FOLDED.
# --------------------------------------------------------------------------
def fold_bn(w, bn_w, bn_b, bn_m, bn_v):
    scale = bn_w / np.sqrt(bn_v + EPS)
    return w * scale[:, None, None, None], bn_b - bn_w * bn_m / np.sqrt(bn_v + EPS)


def conv2d(x, w, b, stride, pad_h, pad_w):
    I, H, W = x.shape
    O, _, kh, kw = w.shape
    Ho = (H - kh + 2 * pad_h) // stride + 1
    Wo = (W - kw + 2 * pad_w) // stride + 1
    xp = np.zeros((I, H + 2 * pad_h, W + 2 * pad_w), dtype=np.float64)
    xp[:, pad_h:pad_h + H, pad_w:pad_w + W] = x
    out = np.zeros((O, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            iy, ix = oy * stride, ox * stride
            patch = xp[:, iy:iy + kh, ix:ix + kw]
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3], [0, 1, 2])) + b
    return out


def relu(x):
    return np.maximum(x, 0.0)


def basicconv(x, sd, prefix, stride=1, pad_h=0, pad_w=0):
    w_f, b_f = fold_bn(sd[prefix + '.conv.weight'],
                       sd[prefix + '.bn.weight'], sd[prefix + '.bn.bias'],
                       sd[prefix + '.bn.running_mean'],
                       sd[prefix + '.bn.running_var'])
    return relu(conv2d(x, w_f, b_f, stride, pad_h, pad_w))


def maxpool(x, k, stride, pad):
    """Floor-sized (TNNetMaxPoolPortable) maxpool, ZERO pad (== torchvision for
    non-negative ReLU'd inputs)."""
    I, H, W = x.shape
    xp = np.zeros((I, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    Ho = (H - k + 2 * pad) // stride + 1
    Wo = (W - k + 2 * pad) // stride + 1
    out = np.zeros((I, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            iy, ix = oy * stride, ox * stride
            out[:, oy, ox] = xp[:, iy:iy + k, ix:ix + k].reshape(I, -1).max(axis=1)
    return out


def grid_avgpool(x, k, pad):
    """count_include_pad=False stride-1 KxK average (TNNetGridAvgPool): divide
    each window by the number of REAL (non-pad) cells in it."""
    I, H, W = x.shape
    Ho = H + 2 * pad - k + 1
    Wo = W + 2 * pad - k + 1
    out = np.zeros((I, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        iy0 = oy - pad
        for ox in range(Wo):
            ix0 = ox - pad
            acc = np.zeros(I, dtype=np.float64)
            cnt = 0
            for ky in range(k):
                iy = iy0 + ky
                if iy < 0 or iy >= H:
                    continue
                for kx in range(k):
                    ix = ix0 + kx
                    if ix < 0 or ix >= W:
                        continue
                    acc += x[:, iy, ix]
                    cnt += 1
            out[:, oy, ox] = acc / cnt
    return out


def f_a(x, sd, prefix):
    b1 = basicconv(x, sd, prefix + '.branch1x1')
    b5 = basicconv(x, sd, prefix + '.branch5x5_1')
    b5 = basicconv(b5, sd, prefix + '.branch5x5_2', pad_h=2, pad_w=2)
    b3 = basicconv(x, sd, prefix + '.branch3x3dbl_1')
    b3 = basicconv(b3, sd, prefix + '.branch3x3dbl_2', pad_h=1, pad_w=1)
    b3 = basicconv(b3, sd, prefix + '.branch3x3dbl_3', pad_h=1, pad_w=1)
    bp = grid_avgpool(x, 3, 1)
    bp = basicconv(bp, sd, prefix + '.branch_pool')
    return np.concatenate([b1, b5, b3, bp], axis=0)


def f_b(x, sd, prefix):
    b3 = basicconv(x, sd, prefix + '.branch3x3', stride=2)
    b3d = basicconv(x, sd, prefix + '.branch3x3dbl_1')
    b3d = basicconv(b3d, sd, prefix + '.branch3x3dbl_2', pad_h=1, pad_w=1)
    b3d = basicconv(b3d, sd, prefix + '.branch3x3dbl_3', stride=2)
    bp = maxpool(x, 3, 2, 0)
    return np.concatenate([b3, b3d, bp], axis=0)


def f_c(x, sd, prefix):
    b1 = basicconv(x, sd, prefix + '.branch1x1')
    b7 = basicconv(x, sd, prefix + '.branch7x7_1')
    b7 = basicconv(b7, sd, prefix + '.branch7x7_2', pad_h=0, pad_w=3)
    b7 = basicconv(b7, sd, prefix + '.branch7x7_3', pad_h=3, pad_w=0)
    b7d = basicconv(x, sd, prefix + '.branch7x7dbl_1')
    b7d = basicconv(b7d, sd, prefix + '.branch7x7dbl_2', pad_h=3, pad_w=0)
    b7d = basicconv(b7d, sd, prefix + '.branch7x7dbl_3', pad_h=0, pad_w=3)
    b7d = basicconv(b7d, sd, prefix + '.branch7x7dbl_4', pad_h=3, pad_w=0)
    b7d = basicconv(b7d, sd, prefix + '.branch7x7dbl_5', pad_h=0, pad_w=3)
    bp = grid_avgpool(x, 3, 1)
    bp = basicconv(bp, sd, prefix + '.branch_pool')
    return np.concatenate([b1, b7, b7d, bp], axis=0)


def f_d(x, sd, prefix):
    b3 = basicconv(x, sd, prefix + '.branch3x3_1')
    b3 = basicconv(b3, sd, prefix + '.branch3x3_2', stride=2)
    b7 = basicconv(x, sd, prefix + '.branch7x7x3_1')
    b7 = basicconv(b7, sd, prefix + '.branch7x7x3_2', pad_h=0, pad_w=3)
    b7 = basicconv(b7, sd, prefix + '.branch7x7x3_3', pad_h=3, pad_w=0)
    b7 = basicconv(b7, sd, prefix + '.branch7x7x3_4', stride=2)
    bp = maxpool(x, 3, 2, 0)
    return np.concatenate([b3, b7, bp], axis=0)


def f_e(x, sd, prefix):
    b1 = basicconv(x, sd, prefix + '.branch1x1')
    b3 = basicconv(x, sd, prefix + '.branch3x3_1')
    b3a = basicconv(b3, sd, prefix + '.branch3x3_2a', pad_h=0, pad_w=1)
    b3b = basicconv(b3, sd, prefix + '.branch3x3_2b', pad_h=1, pad_w=0)
    b3d = basicconv(x, sd, prefix + '.branch3x3dbl_1')
    b3d = basicconv(b3d, sd, prefix + '.branch3x3dbl_2', pad_h=1, pad_w=1)
    b3da = basicconv(b3d, sd, prefix + '.branch3x3dbl_3a', pad_h=0, pad_w=1)
    b3db = basicconv(b3d, sd, prefix + '.branch3x3dbl_3b', pad_h=1, pad_w=0)
    bp = grid_avgpool(x, 3, 1)
    bp = basicconv(bp, sd, prefix + '.branch_pool')
    return np.concatenate([b1, b3a, b3b, b3da, b3db, bp], axis=0)


def forward(x, sd, trace=False):
    out = basicconv(x, sd, 'Conv2d_1a_3x3', stride=2)
    out = basicconv(out, sd, 'Conv2d_2a_3x3')
    out = basicconv(out, sd, 'Conv2d_2b_3x3', pad_h=1, pad_w=1)
    out = maxpool(out, 3, 2, 0)
    out = basicconv(out, sd, 'Conv2d_3b_1x1')
    out = basicconv(out, sd, 'Conv2d_4a_3x3')
    out = maxpool(out, 3, 2, 0)
    if trace:
        print(f'  after stem: {out.shape}')
    out = f_a(out, sd, 'Mixed_5b')
    out = f_a(out, sd, 'Mixed_5c')
    out = f_a(out, sd, 'Mixed_5d')
    if trace:
        print(f'  after 5d (A): {out.shape}')
    out = f_b(out, sd, 'Mixed_6a')
    out = f_c(out, sd, 'Mixed_6b')
    out = f_c(out, sd, 'Mixed_6c')
    out = f_c(out, sd, 'Mixed_6d')
    out = f_c(out, sd, 'Mixed_6e')
    if trace:
        print(f'  after 6e (C): {out.shape}')
    out = f_d(out, sd, 'Mixed_7a')
    out = f_e(out, sd, 'Mixed_7b')
    out = f_e(out, sd, 'Mixed_7c')
    if trace:
        print(f'  after 7c (E): {out.shape}')
    pooled = out.reshape(out.shape[0], -1).mean(axis=1)
    logits = sd['fc.weight'] @ pooled + sd['fc.bias']
    return logits, out.shape, pooled


def main():
    sd, pooled_dim = build_state_dict()

    pixels = np.zeros((NUM_CHANNELS, IMAGE, IMAGE), dtype=np.float64)
    for c in range(NUM_CHANNELS):
        for y in range(IMAGE):
            for x in range(IMAGE):
                pixels[c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()
              if not k.endswith('num_batches_tracked')}
    sd64 = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    logits, feat_shape, pooled = forward(pixels, sd64, trace=True)
    print(f'pooled_dim={pooled_dim}, feat {feat_shape}, logits {logits.shape}')
    print(f'logits: {logits.tolist()}')

    save_file(sd_f32, 'tests/fixtures/tiny_inceptionv3_full.safetensors')

    config = {
        'model_type': 'inception',
        'architectures': ['InceptionV3'],
        'full_arch': True,
        'width_div': WIDTH_DIV,
        'image_size': IMAGE,
        'num_channels': NUM_CHANNELS,
        'num_labels': NUM_CLASSES,
        'bn_eps': EPS,
    }
    with open('tests/fixtures/tiny_inceptionv3_full_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_inceptionv3_full_logits.json', 'w') as f:
        json.dump({
            'pixels': pixels.tolist(),
            'logits': [logits.tolist()],
            'pooled': pooled.tolist(),
            'num_labels': NUM_CLASSES,
            'pooled_dim': pooled_dim,
        }, f)
    print(f'wrote tiny_inceptionv3_full.* ({len(sd_f32)} tensors)')

    # ---- fixture self-checks: every distinctive path must matter ----
    base = logits.copy()
    for key, desc in [
        ('Conv2d_1a_3x3.conv.weight', 'stem 1a conv'),
        ('Mixed_5b.branch_pool.conv.weight', 'A avgpool branch'),
        ('Mixed_6a.branch3x3.conv.weight', 'B stride-2 conv'),
        ('Mixed_6b.branch7x7_2.conv.weight', 'C 1x7 conv'),
        ('Mixed_6b.branch7x7_3.conv.weight', 'C 7x1 conv'),
        ('Mixed_7a.branch7x7x3_2.conv.weight', 'D 1x7 conv'),
        ('Mixed_7b.branch3x3_2a.conv.weight', 'E 1x3 split-a'),
        ('Mixed_7b.branch3x3_2b.conv.weight', 'E 3x1 split-b'),
        ('Mixed_7c.branch_pool.conv.weight', 'E avgpool branch'),
        ('fc.bias', 'fc bias'),
    ]:
        alt = dict(sd64)
        alt[key] = np.zeros_like(sd64[key])
        d = np.abs(forward(pixels, alt)[0] - base).max()
        assert d > 1e-4, f'{desc} had no effect ({d})'
        print(f'{desc} effect: max|diff| = {d:.4f}')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
