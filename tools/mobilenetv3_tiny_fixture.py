#!/usr/bin/env python3
"""Generate a tiny RANDOM torchvision-MobileNetV3 parity fixture for
tests/TestNeuralPretrained.pas (no network access, no torchvision needed:
the model is a pico random-init MobileNetV3 whose state_dict mirrors
torchvision's exact key scheme, and the reference logits are computed by a
self-contained numpy float64 oracle).

torchvision is NOT installed in the reusable venv, so - mirroring
tools/resnet18_tiny_fixture.py and tools/internlm2_tiny_fixture.py - the
oracle is hand-written in numpy. This exercises EVERY branch of the
BuildMobileNetV3 importer (neuralpretrained.pas):
  - stem conv 3x3 stride2 pad1 + folded BN + hard-swish
  - MBConv blocks covering all branch combos:
      * NO expand (exp == in)               -> depthwise sub-block is .0
      * WITH expand (exp != in)             -> expand .0, depthwise .1
      * WITH squeeze-excite (fc1 reduce / fc2 expand 1x1 convs WITH bias)
      * WITHOUT squeeze-excite
      * stride 2 (spatial downsample)
      * stride 1 & in == out                -> residual add
      * ReLU blocks and hard-swish blocks
  - head conv 1x1 + folded BN + hard-swish -> global avg pool
  - classifier.0 Linear -> hard-swish -> classifier.3 Linear

torchvision Conv2d has bias=False everywhere except the SE 1x1 convs (bias);
every conv (stem / expand / depthwise / project / head) is followed by a
BatchNorm2d with eps = 1e-3 (torchvision MobileNetV3, NOT 1e-5). The importer
FOLDS each BN into the preceding conv at load time; this oracle does the SAME
fold then runs the folded conv, so the test checks the fold + the CAI forward
path end to end.

state_dict key scheme (torchvision mobilenet_v3):
  features.0.0.weight                  stem conv
  features.0.1.{weight,bias,running_mean,running_var}   stem BN
  for MBConv block at features.{i+1}.block.{j}...:
    if expand present: j=0 is expand (.0 conv, .1 bn), j=1 is depthwise
    else:              j=0 is depthwise
    depthwise sub-block: .0 conv [C,1,k,k] groups=C, .1 bn
    if SE present:      next j is SE: .fc1.{weight,bias} reduce 1x1,
                                      .fc2.{weight,bias} expand 1x1
    project:            last j is .0 conv 1x1, .1 bn
  features.{nblocks+1}.0.weight        head conv 1x1
  features.{nblocks+1}.1.*             head BN
  classifier.0.{weight,bias}           Linear (head_conv -> hidden)
  classifier.3.{weight,bias}           Linear (hidden -> num_labels)

POOLING parity note: CAI's TNNetAvgChannel is a TNNetAvgPool with pool == its
input width and stride == width, so it divides by width*width. That equals a
true global mean ONLY when the map is SQUARE. This fixture keeps every spatial
map square (square image, stride-preserving pad = k//2) so AvgChannel is an
exact global average; the oracle uses .mean() over H*W to match.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/mobilenetv3_tiny_fixture.py
writes tests/fixtures/tiny_mobilenetv3{.safetensors,_config.json,_logits.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

EPS = 1e-3                  # torchvision MobileNetV3 BatchNorm2d eps
# Image 32 (square) keeps every spatial map square AND >= the largest depthwise
# kernel (5) after the single stride-2 downsample, avoiding CAI's
# kernel-clamp-to-input path (FFeatureSize := Min(FFeatureSize, InputSize)).
# Map sizes: stem/2 -> 16, blk0 s1 -> 16, blk1 s2 -> 8, blk2/blk3 s1 -> 8.
IMAGE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 5
STEM = 4
HEAD_CONV = 24
CLASSIFIER_HIDDEN = 12

# Block table: each block exercises a distinct branch combination.
# (kernel, exp, out, stride, se, se_channels, hswish)
# blk0: NO expand (exp==in==STEM=4), SE, stride1, in!=out (no residual), ReLU
# blk1: expand, NO SE, stride2 (downsample), hard-swish
# blk2: expand, SE, stride1, in==out -> RESIDUAL, hard-swish
# blk3: expand, NO SE, stride1, in!=out (no residual), ReLU
BLOCKS = [
    dict(kernel=3, exp=4,  out=8,  stride=1, se=True,  se_channels=4, hswish=False),
    dict(kernel=3, exp=16, out=12, stride=2, se=False, se_channels=0, hswish=True),
    dict(kernel=5, exp=24, out=12, stride=1, se=True,  se_channels=8, hswish=True),
    dict(kernel=3, exp=20, out=16, stride=1, se=False, se_channels=0, hswish=False),
]

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


def make_conv(prefix, sd, out_ch, in_ch, k, std=0.2):
    sd[prefix + '.weight'] = randn(out_ch, in_ch, k, k, std=std)  # [O,I,kh,kw]


def make_dwconv(prefix, sd, channels, k, std=0.3):
    sd[prefix + '.weight'] = randn(channels, 1, k, k, std=std)    # [C,1,kh,kw]


def make_se(prefix, sd, exp_ch, se_ch):
    # torchvision SqueezeExcitation: fc1 (reduce) / fc2 (expand) 1x1 convs WITH
    # bias, weight [O,I,1,1].
    sd[prefix + '.fc1.weight'] = randn(se_ch, exp_ch, 1, 1, std=0.3)
    sd[prefix + '.fc1.bias'] = randn(se_ch, std=0.2)
    sd[prefix + '.fc2.weight'] = randn(exp_ch, se_ch, 1, 1, std=0.3)
    sd[prefix + '.fc2.bias'] = randn(exp_ch, std=0.2)


def build_state_dict():
    sd = {}
    make_conv('features.0.0', sd, STEM, NUM_CHANNELS, 3)
    make_bn('features.0.1', sd, STEM)
    in_ch = STEM
    for i, blk in enumerate(BLOCKS):
        p = f'features.{i + 1}.block.'
        has_expand = blk['exp'] != in_ch
        j = 0
        if has_expand:
            make_conv(p + f'{j}.0', sd, blk['exp'], in_ch, 1)   # expand 1x1
            make_bn(p + f'{j}.1', sd, blk['exp'])
            j += 1
        make_dwconv(p + f'{j}.0', sd, blk['exp'], blk['kernel'])  # depthwise
        make_bn(p + f'{j}.1', sd, blk['exp'])
        j += 1
        if blk['se']:
            make_se(p + f'{j}', sd, blk['exp'], blk['se_channels'])
            j += 1
        make_conv(p + f'{j}.0', sd, blk['out'], blk['exp'], 1)  # project 1x1
        make_bn(p + f'{j}.1', sd, blk['out'])
        in_ch = blk['out']
    last = len(BLOCKS) + 1
    make_conv(f'features.{last}.0', sd, HEAD_CONV, in_ch, 1)    # head conv 1x1
    make_bn(f'features.{last}.1', sd, HEAD_CONV)
    sd['classifier.0.weight'] = randn(CLASSIFIER_HIDDEN, HEAD_CONV, std=0.3)
    sd['classifier.0.bias'] = randn(CLASSIFIER_HIDDEN, std=0.2)
    sd['classifier.3.weight'] = randn(NUM_CLASSES, CLASSIFIER_HIDDEN, std=0.3)
    sd['classifier.3.bias'] = randn(NUM_CLASSES, std=0.2)
    return sd


# --------------------------------------------------------------------------
# numpy float64 oracle, replicating the CAI forward path exactly.
# Volumes here are kept in (C, H, W); conv weights are torchvision [O,I,kh,kw].
# BN is FOLDED into the conv (matching the importer).
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


def dwconv2d(x, w, b, stride, pad):
    """Depthwise conv. x:[C,H,W] w:[C,1,k,k] b:[C] -> [C,Ho,Wo]."""
    C, H, W = x.shape
    _, _, k, _ = w.shape
    Ho = (H - k + 2 * pad) // stride + 1
    Wo = (W - k + 2 * pad) // stride + 1
    xp = np.zeros((C, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    out = np.zeros((C, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            iy, ix = oy * stride, ox * stride
            patch = xp[:, iy:iy + k, ix:ix + k]       # [C,k,k]
            out[:, oy, ox] = (w[:, 0, :, :] * patch).reshape(C, -1).sum(axis=1) + b
    return out


def relu(x):
    return np.maximum(x, 0.0)


def relu6(x):
    return np.clip(x, 0.0, 6.0)


def hard_swish(x):
    return x * relu6(x + 3.0) / 6.0


def hard_sigmoid(x):
    return np.clip((x + 3.0) / 6.0, 0.0, 1.0)


def conv_bn(x, sd, conv_prefix, bn_prefix, stride, pad):
    w_f, b_f = fold_bn(sd[conv_prefix + '.weight'],
                       sd[bn_prefix + '.weight'], sd[bn_prefix + '.bias'],
                       sd[bn_prefix + '.running_mean'],
                       sd[bn_prefix + '.running_var'])
    return conv2d(x, w_f, b_f, stride, pad)


def dwconv_bn(x, sd, conv_prefix, bn_prefix, stride, pad):
    # The importer folds the BN SCALE into the depthwise weights and routes the
    # SHIFT through a following TNNetChannelBias; net result is conv + (w*scale)
    # then + shift, which equals folding both into a biased depthwise. We fold
    # both here for parity.
    w = sd[conv_prefix + '.weight']
    bn_w = sd[bn_prefix + '.weight']; bn_b = sd[bn_prefix + '.bias']
    bn_m = sd[bn_prefix + '.running_mean']; bn_v = sd[bn_prefix + '.running_var']
    scale = bn_w / np.sqrt(bn_v + EPS)
    shift = bn_b - bn_w * bn_m / np.sqrt(bn_v + EPS)
    w_f = w * scale[:, None, None, None]
    return dwconv2d(x, w_f, shift, stride, pad)


def se_module(x, sd, prefix):
    """Squeeze-excite on x:[C,H,W]. pool -> 1x1 reduce -> relu -> 1x1 expand ->
    hard-sigmoid -> per-channel multiply. The 1x1 convs are plain matmuls on the
    (C,) pooled vector (NO BN)."""
    C = x.shape[0]
    pooled = x.reshape(C, -1).mean(axis=1)               # global avg pool [C]
    w1 = sd[prefix + '.fc1.weight'][:, :, 0, 0]          # [se, C]
    b1 = sd[prefix + '.fc1.bias']
    reduced = relu(w1 @ pooled + b1)                     # [se]
    w2 = sd[prefix + '.fc2.weight'][:, :, 0, 0]          # [C, se]
    b2 = sd[prefix + '.fc2.bias']
    gate = hard_sigmoid(w2 @ reduced + b2)               # [C]
    return x * gate[:, None, None]


def mbconv(x, sd, idx, blk, in_ch):
    p = f'features.{idx + 1}.block.'
    has_expand = blk['exp'] != in_ch
    has_residual = (blk['stride'] == 1) and (in_ch == blk['out'])
    pad = blk['kernel'] // 2
    act = hard_swish if blk['hswish'] else relu
    inp = x
    j = 0
    out = x
    if has_expand:
        out = conv_bn(out, sd, p + f'{j}.0', p + f'{j}.1', 1, 0)  # expand 1x1
        out = act(out)
        j += 1
    out = dwconv_bn(out, sd, p + f'{j}.0', p + f'{j}.1', blk['stride'], pad)
    out = act(out)
    j += 1
    if blk['se']:
        out = se_module(out, sd, p + f'{j}')
        j += 1
    out = conv_bn(out, sd, p + f'{j}.0', p + f'{j}.1', 1, 0)      # project 1x1
    # NO activation after project.
    if has_residual:
        out = out + inp
    return out


def forward(x, sd):
    out = conv_bn(x, sd, 'features.0.0', 'features.0.1', 2, 1)    # stem
    out = hard_swish(out)
    in_ch = STEM
    for i, blk in enumerate(BLOCKS):
        out = mbconv(out, sd, i, blk, in_ch)
        in_ch = blk['out']
    last = len(BLOCKS) + 1
    out = conv_bn(out, sd, f'features.{last}.0', f'features.{last}.1', 1, 0)
    out = hard_swish(out)
    pooled = out.reshape(out.shape[0], -1).mean(axis=1)          # global avg pool
    h = sd['classifier.0.weight'] @ pooled + sd['classifier.0.bias']
    h = hard_swish(h)
    logits = sd['classifier.3.weight'] @ h + sd['classifier.3.bias']
    return logits, out.shape


def main():
    sd = build_state_dict()

    # Pinned input: deterministic dyadic pixel values (exact in f32 + JSON).
    pixels = np.zeros((NUM_CHANNELS, IMAGE, IMAGE), dtype=np.float64)
    for c in range(NUM_CHANNELS):
        for y in range(IMAGE):
            for xx in range(IMAGE):
                pixels[c, y, xx] = (((c * 1024 + y * 32 + xx) * 5) % 17 - 8) / 8.0

    # Round-trip every weight through float32 BEFORE the oracle: CAI loads f32
    # weights, so the oracle must fold/forward from the SAME f32 values.
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()
              if not k.endswith('num_batches_tracked')}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    logits, feat_shape = forward(pixels, sd)
    print(f'feature map before head pool: {feat_shape}, logits {logits.shape}')
    print(f'logits: {logits.tolist()}')

    save_file(sd_f32, 'tests/fixtures/tiny_mobilenetv3.safetensors')

    config = {
        'model_type': 'mobilenet_v3',
        'variant': 'small',
        'image_size': IMAGE,
        'num_channels': NUM_CHANNELS,
        'num_labels': NUM_CLASSES,
        'stem_width': STEM,
        'head_conv_width': HEAD_CONV,
        'classifier_hidden': CLASSIFIER_HIDDEN,
        'bn_eps': EPS,
        'blocks': [
            dict(kernel=b['kernel'], exp=b['exp'], out=b['out'],
                 stride=b['stride'], se=b['se'],
                 se_channels=b['se_channels'], hswish=b['hswish'])
            for b in BLOCKS
        ],
    }
    with open('tests/fixtures/tiny_mobilenetv3_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_mobilenetv3_logits.json', 'w') as f:
        json.dump({
            'pixels': pixels.tolist(),            # [3][IMAGE][IMAGE]
            'logits': [logits.tolist()],          # [1][NUM_CLASSES]
            'num_labels': NUM_CLASSES,
        }, f)
    print(f'wrote tiny_mobilenetv3.safetensors ({len(sd_f32)} tensors) + '
          f'config + oracle')
    for k in sorted(sd_f32):
        print(f'  {k} {list(sd_f32[k].shape)}')

    # ---- fixture self-checks: each branch must move the logits ----
    base = logits.copy()

    def effect(key):
        alt = dict(sd)
        alt[key] = np.zeros_like(sd[key])
        return np.abs(forward(pixels, alt)[0] - base).max()

    # Each branch must measurably MOVE the logits (small threshold: these are
    # tiny random weights so effects are modest, but must be clearly non-zero).
    TH = 1e-5
    # stem BN gamma
    d = effect('features.0.1.weight')
    assert d > TH, f'stem BN gamma had no effect ({d})'
    print(f'stem BN gamma effect: {d:.6f}')
    # SE in blk0 (no-expand SE block, depthwise=.0 SE=.1 project=.2)
    d = effect('features.1.block.1.fc2.weight')
    assert d > TH, f'blk0 SE had no effect ({d})'
    print(f'blk0 SE fc2 effect: {d:.6f}')
    # expand conv in blk1 (has expand: expand=.0)
    d = effect('features.2.block.0.0.weight')
    assert d > TH, f'blk1 expand had no effect ({d})'
    print(f'blk1 expand effect: {d:.6f}')
    # residual block blk2 (has expand .0, dw .1, SE .2, project .3)
    d = effect('features.3.block.3.0.weight')
    assert d > TH, f'blk2 project had no effect ({d})'
    print(f'blk2 project effect: {d:.6f}')
    # classifier.0 bias
    d = effect('classifier.0.bias')
    assert d > TH, f'classifier.0 bias had no effect ({d})'
    print(f'classifier.0 bias effect: {d:.6f}')
    # classifier.3 bias
    d = effect('classifier.3.bias')
    assert d > TH, f'classifier.3 bias had no effect ({d})'
    print(f'classifier.3 bias effect: {d:.6f}')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
