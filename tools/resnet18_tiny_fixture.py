#!/usr/bin/env python3
"""Generate a tiny RANDOM torchvision-ResNet18 parity fixture for
tests/TestNeuralPretrained.pas (no network access, no torchvision needed:
the model is a pico random-init ResNet whose state_dict mirrors
torchvision's exact key scheme, and the reference logits are computed by a
self-contained numpy float64 oracle).

This is the FIRST conv-net (non-transformer / non-ViT) weight import in the
repo, so the oracle is hand-written in numpy (torchvision is not installed
in the reusable venv) - mirroring tools/internlm2_tiny_fixture.py, which
builds its own numpy oracle when the model is not in the library.

Architecture (torchvision ResNet18 = BasicBlock, layout blocks=[2,2,2,2]):
  stem:   conv1 7x7 stride2 pad3 (no bias) -> bn1 -> relu -> maxpool 3x3 s2 p1
  layer1: 2 BasicBlocks, width W0, stride 1 (NO downsample)
  layer2: 2 BasicBlocks, width W1, first block stride 2 + 1x1 downsample
  layer3: 2 BasicBlocks, width W2, first block stride 2 + 1x1 downsample
  layer4: 2 BasicBlocks, width W3, first block stride 2 + 1x1 downsample
  head:   global avg pool -> fc (Linear W3 -> num_classes)
each BasicBlock: conv1 3x3(pad1) -> bn1 -> relu -> conv2 3x3 pad1 -> bn2,
add shortcut (identity or downsample 1x1 conv + bn), then relu.
torchvision folds NO bias into conv (Conv2d bias=False); every conv is
followed by a BatchNorm2d (weight/bias/running_mean/running_var, eps 1e-5).

CONV-BN FOLD parity: the importer folds each BN into the preceding conv at
load time (w' = w*gamma/sqrt(var+eps); b' = beta - gamma*mean/sqrt(var+eps)).
This oracle does the SAME fold then runs the folded conv, so the test checks
the fold + the CAI forward path end to end.

POOLING parity note: the CAI TNNetMaxPool with padding uses ceil() output
sizing, zero-padding, and edge-clamped windows (NOT PyTorch's floor() +
-inf padding). The numpy oracle here replicates the CAI maxpool semantics
exactly (see cai_maxpool), so this is a true parity check of the importer
plus the CAI forward path. Matching torchvision's maxpool bit-for-bit is a
follow-up (would need a maxpool variant with PyTorch sizing).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/resnet18_tiny_fixture.py
writes tests/fixtures/tiny_resnet18{.safetensors,_config.json,_logits.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

EPS = 1e-5
# IMAGE 64 is the SMALLEST input that keeps every spatial map >= the 3x3
# kernel through all 4 stages (stem /2, maxpool /2, layer2/3/4 /2 each).
# CAI's TNNetConvolution CLAMPS the kernel to the input size when the input
# is smaller than the kernel (FFeatureSize := Min(FFeatureSize, InputSize)),
# which diverges from PyTorch; keeping maps >= 3 avoids that path entirely
# so the importer is exercised on the real (un-clamped) ResNet conv arithmetic.
IMAGE = 64
NUM_CHANNELS = 3
NUM_CLASSES = 5
WIDTHS = [4, 6, 8, 12]           # layer1..layer4 widths (BasicBlock, expansion 1)
BLOCKS = [2, 2, 2, 2]            # resnet18 layout
STEM = WIDTHS[0]

rng = np.random.default_rng(20260614)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def make_bn(prefix, sd, channels):
    """Random, NON-trivial BatchNorm so the fold is actually exercised."""
    sd[prefix + '.weight'] = randn(channels, std=0.3) + 1.0       # gamma
    sd[prefix + '.bias'] = randn(channels, std=0.25)              # beta
    sd[prefix + '.running_mean'] = randn(channels, std=0.3)
    sd[prefix + '.running_var'] = np.abs(randn(channels, std=0.3)) + 0.5
    sd[prefix + '.num_batches_tracked'] = np.array(0, dtype=np.int64)


def make_conv(prefix, sd, out_ch, in_ch, k, std=0.15):
    sd[prefix + '.weight'] = randn(out_ch, in_ch, k, k, std=std)  # [O,I,kh,kw]


def make_basic_block(prefix, sd, in_ch, out_ch, stride):
    make_conv(prefix + '.conv1', sd, out_ch, in_ch, 3)
    make_bn(prefix + '.bn1', sd, out_ch)
    make_conv(prefix + '.conv2', sd, out_ch, out_ch, 3)
    make_bn(prefix + '.bn2', sd, out_ch)
    if stride != 1 or in_ch != out_ch:
        make_conv(prefix + '.downsample.0', sd, out_ch, in_ch, 1)
        make_bn(prefix + '.downsample.1', sd, out_ch)


def build_state_dict():
    sd = {}
    make_conv('conv1', sd, STEM, NUM_CHANNELS, 7)
    make_bn('bn1', sd, STEM)
    in_ch = STEM
    for li, (width, nblocks) in enumerate(zip(WIDTHS, BLOCKS), start=1):
        stride = 1 if li == 1 else 2
        for bi in range(nblocks):
            s = stride if bi == 0 else 1
            make_basic_block(f'layer{li}.{bi}', sd, in_ch, width, s)
            in_ch = width
    sd['fc.weight'] = randn(NUM_CLASSES, WIDTHS[-1], std=0.4)     # [out,in]
    sd['fc.bias'] = randn(NUM_CLASSES, std=0.3)
    return sd


# --------------------------------------------------------------------------
# numpy float64 oracle, replicating the CAI forward path exactly.
# Volumes here are kept in (C, H, W) for clarity; conv weights are torchvision
# [O, I, kh, kw]. BN is FOLDED into the conv (matching the importer).
# --------------------------------------------------------------------------
def fold_bn(w, bn_w, bn_b, bn_m, bn_v):
    """w:[O,I,k,k] folded with BN params -> (w', b')."""
    scale = bn_w / np.sqrt(bn_v + EPS)                # [O]
    w_f = w * scale[:, None, None, None]
    b_f = bn_b - bn_w * bn_m / np.sqrt(bn_v + EPS)    # [O]
    return w_f, b_f


def conv2d(x, w, b, stride, pad):
    """x:[I,H,W] w:[O,I,k,k] b:[O] -> [O,Ho,Wo]. Zero pad, standard conv
    (output size (H - k + 2p)//stride + 1 == CAI and PyTorch agree)."""
    I, H, W = x.shape
    O, _, k, _ = w.shape
    Ho = (H - k + 2 * pad) // stride + 1
    Wo = (W - k + 2 * pad) // stride + 1
    xp = np.zeros((I, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    out = np.zeros((O, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            iy, ix = oy * stride, ox * stride
            patch = xp[:, iy:iy + k, ix:ix + k]       # [I,k,k]
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3],
                                                          [0, 1, 2])) + b
    return out


def relu(x):
    return np.maximum(x, 0.0)


def cai_maxpool(x, pool, stride, pad):
    """Replicate TNNetMaxPool with padding: ZERO pad, ceil() output size,
    edge-clamped windows. out size = ceil((H + 2p) / stride)."""
    I, H, W = x.shape
    xp = np.zeros((I, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    PH, PW = H + 2 * pad, W + 2 * pad
    Ho = -(-PH // stride)        # ceil
    Wo = -(-PW // stride)
    out = np.zeros((I, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            iy, ix = oy * stride, ox * stride
            iy_max = min(iy + pool - 1, PH - 1)       # edge clamp
            ix_max = min(ix + pool - 1, PW - 1)
            win = xp[:, iy:iy_max + 1, ix:ix_max + 1]
            out[:, oy, ox] = win.reshape(I, -1).max(axis=1)
    return out


def conv_bn(x, sd, conv_prefix, bn_prefix, stride, pad):
    w_f, b_f = fold_bn(sd[conv_prefix + '.weight'],
                       sd[bn_prefix + '.weight'], sd[bn_prefix + '.bias'],
                       sd[bn_prefix + '.running_mean'],
                       sd[bn_prefix + '.running_var'])
    return conv2d(x, w_f, b_f, stride, pad)


def basic_block(x, sd, prefix, stride, has_downsample):
    out = conv_bn(x, sd, prefix + '.conv1', prefix + '.bn1', stride, 1)
    out = relu(out)
    out = conv_bn(out, sd, prefix + '.conv2', prefix + '.bn2', 1, 1)
    if has_downsample:
        ident = conv_bn(x, sd, prefix + '.downsample.0',
                        prefix + '.downsample.1', stride, 0)
    else:
        ident = x
    return relu(out + ident)


def forward(x, sd):
    out = conv_bn(x, sd, 'conv1', 'bn1', 2, 3)
    out = relu(out)
    out = cai_maxpool(out, 3, 2, 1)
    for li, (nblocks,) in enumerate(zip(BLOCKS), start=1):
        stride = 1 if li == 1 else 2
        for bi in range(nblocks):
            s = stride if bi == 0 else 1
            has_ds = (bi == 0) and (s != 1 or out.shape[0] != WIDTHS[li - 1])
            out = basic_block(out, sd, f'layer{li}.{bi}', s, has_ds)
    pooled = out.reshape(out.shape[0], -1).mean(axis=1)    # global avg pool
    logits = sd['fc.weight'] @ pooled + sd['fc.bias']
    return logits, out.shape


def main():
    sd = build_state_dict()

    # Pinned input: deterministic dyadic pixel values (exact in f32 + JSON).
    pixels = np.zeros((NUM_CHANNELS, IMAGE, IMAGE), dtype=np.float64)
    for c in range(NUM_CHANNELS):
        for y in range(IMAGE):
            for x in range(IMAGE):
                pixels[c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

    # Round-trip every weight through float32 BEFORE computing the oracle:
    # CAI loads float32 weights, so the oracle must fold/forward from the
    # SAME float32 values. This removes weight-quantization mismatch, leaving
    # only the float64-vs-float32 forward-arithmetic gap (well under 1e-4 at
    # these activation magnitudes over ResNet18 depth).
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()
              if not k.endswith('num_batches_tracked')}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    logits, feat_shape = forward(pixels, sd)
    print(f'feature map before pool: {feat_shape}, logits {logits.shape}')
    print(f'logits: {logits.tolist()}')

    save_file(sd_f32, 'tests/fixtures/tiny_resnet18.safetensors')

    config = {
        'model_type': 'resnet',
        'architectures': ['ResNetForImageClassification'],
        'depth': 18,
        'image_size': IMAGE,
        'num_channels': NUM_CHANNELS,
        'num_labels': NUM_CLASSES,
        'widths': WIDTHS,
        'blocks_per_stage': BLOCKS,
        'bn_eps': EPS,
    }
    with open('tests/fixtures/tiny_resnet18_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_resnet18_logits.json', 'w') as f:
        json.dump({
            'pixels': pixels.tolist(),            # [3][IMAGE][IMAGE]
            'logits': [logits.tolist()],          # [1][NUM_CLASSES]
            'num_labels': NUM_CLASSES,
        }, f)
    print(f'wrote tiny_resnet18.safetensors ({len(sd_f32)} tensors) + '
          f'config + oracle')
    for k in sorted(sd_f32):
        print(f'  {k} {list(sd_f32[k].shape)}')

    # ---- fixture self-checks ----
    base = logits.copy()
    # 1. BN fold must matter: zeroing bn1.weight (gamma) must move logits.
    alt = dict(sd)
    alt['bn1.weight'] = np.zeros_like(sd['bn1.weight'])
    d = np.abs(forward(pixels, alt)[0] - base).max()
    assert d > 1e-3, f'bn1 gamma had no effect ({d})'
    print(f'bn1 gamma effect: max|diff| = {d:.4f}')
    # 2. downsample shortcut must matter (layer2.0.downsample).
    alt = dict(sd)
    alt['layer2.0.downsample.0.weight'] = \
        np.zeros_like(sd['layer2.0.downsample.0.weight'])
    d = np.abs(forward(pixels, alt)[0] - base).max()
    assert d > 1e-3, f'layer2 downsample had no effect ({d})'
    print(f'layer2 downsample effect: max|diff| = {d:.4f}')
    # 3. fc bias must matter.
    alt = dict(sd)
    alt['fc.bias'] = np.zeros_like(sd['fc.bias'])
    d = np.abs(forward(pixels, alt)[0] - base).max()
    assert d > 1e-3, f'fc bias had no effect ({d})'
    print(f'fc bias effect: max|diff| = {d:.4f}')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
