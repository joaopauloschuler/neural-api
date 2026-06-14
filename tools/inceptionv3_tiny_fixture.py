#!/usr/bin/env python3
"""Generate a tiny RANDOM Inception-v3-shaped parity fixture for
tests/TestNeuralPretrained.pas (no network access, no torchvision needed:
the model is a pico random-init Inception whose state_dict mirrors the
torchvision inception_v3 key scheme for the modules it exercises, and the
reference logits are computed by a self-contained numpy float64 oracle).

torchvision is NOT installed in the reusable venv, so -- exactly like
tools/resnet18_tiny_fixture.py -- the oracle is a hand-written numpy float64
forward that replicates the CAI forward path bit-for-bit (the SAME conv-BN
fold + CAI pool/concat semantics), so the test checks the importer (the
multi-branch concat block builder + the conv-BN fold) end to end.

WHAT IS EXERCISED (the genuinely-new Inception work):
  * the branch-concatenation block builder: each Inception module runs N
    parallel conv/pool branches and concatenates them on the CHANNEL (depth)
    axis (TNNetDeepConcat). This is structurally distinct from every other
    landed CNN (ResNet/VGG/MobileNet are sequential/residual, never a
    channel-concat of parallel branches).
  * the 5x5-as-two-3x3 factorization (branch 3 below).
  * a pool branch (size-preserving 3x3 pool -> 1x1 conv).
  * conv-BN fold (reused from the ResNet path): every Conv2d has bias=False
    and is followed by a BatchNorm2d (eps 1e-3 in torchvision inception),
    FOLDED into the conv at load.
  * global avg pool -> fc logits head (the pooled feature is the FID backbone
    tap; here it is the input to fc).

PICO SIMPLIFICATIONS (documented importer limitations, mirrored in the
oracle so parity is exact):
  * SQUARE kernels only. torchvision Inception-v3's InceptionC uses asymmetric
    1x7 / 7x1 factorized convs; CAI's TNNetConvolution is square-kernel only,
    so the asymmetric-conv InceptionC and the grid-reduction InceptionB/D
    (strided, size-shrinking) are a documented follow-up. The pico exercises
    the InceptionA-shaped module (1x1 / 3x3 / 5x5-as-two-3x3 / pool), which is
    the canonical multi-branch concat block and the new builder.
  * SIZE-PRESERVING modules: every branch keeps the 8x8 grid (stride-1,
    same-padding) so the four branches can be channel-concatenated. The pool
    branch uses a PyTorch-sized (floor) 3x3 stride-1 pad-1 MAXpool
    (TNNetMaxPoolPortable) which preserves the grid; torchvision's pool branch
    is an AVGpool, a documented difference (the oracle uses the same maxpool
    the importer builds, so this is a true end-to-end parity check).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/inceptionv3_tiny_fixture.py
writes tests/fixtures/tiny_inceptionv3{.safetensors,_config.json,_logits.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

EPS = 1e-3                 # torchvision inception BatchNorm2d eps
IMAGE = 8
NUM_CHANNELS = 3
NUM_CLASSES = 5

# Pico module widths. The state_dict key scheme mirrors torchvision's
# Inception module naming (BasicConv2d sub-modules: .conv / .bn). We use the
# InceptionA branch names: branch1x1, branch5x5_1/_2, branch3x3dbl_1/_2/_3,
# branch_pool. Channel counts are tiny but the topology is faithful.
STEM = 4                   # stem conv out channels
# Per-InceptionA-module branch out channels (square-kernel pico version):
B1 = 3                     # branch1x1: 1x1
B5_R = 2                   # branch5x5_1: 1x1 reduce
B5 = 3                     # branch5x5_2: here a 3x3 (pico stand-in for 5x5)
B3_R = 2                   # branch3x3dbl_1: 1x1 reduce
B3_M = 3                   # branch3x3dbl_2: 3x3
B3 = 4                     # branch3x3dbl_3: 3x3  (the 5x5-as-two-3x3 path)
BP = 3                     # branch_pool: 1x1 after pool
# concatenated module output channels:
MOD_OUT = B1 + B5 + B3 + BP
NUM_MODULES = 2            # InceptionA x2

rng = np.random.default_rng(20260614)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def make_bn(prefix, sd, channels):
    sd[prefix + '.weight'] = randn(channels, std=0.3) + 1.0       # gamma
    sd[prefix + '.bias'] = randn(channels, std=0.25)              # beta
    sd[prefix + '.running_mean'] = randn(channels, std=0.3)
    sd[prefix + '.running_var'] = np.abs(randn(channels, std=0.3)) + 0.5
    sd[prefix + '.num_batches_tracked'] = np.array(0, dtype=np.int64)


def make_basicconv(prefix, sd, out_ch, in_ch, k, std=0.2):
    """torchvision BasicConv2d = conv (bias=False) + bn, keys .conv / .bn."""
    sd[prefix + '.conv.weight'] = randn(out_ch, in_ch, k, k, std=std)
    make_bn(prefix + '.bn', sd, out_ch)


def make_inception_a(prefix, sd, in_ch):
    make_basicconv(prefix + '.branch1x1', sd, B1, in_ch, 1)
    make_basicconv(prefix + '.branch5x5_1', sd, B5_R, in_ch, 1)
    make_basicconv(prefix + '.branch5x5_2', sd, B5, B5_R, 3)   # pico: 3x3
    make_basicconv(prefix + '.branch3x3dbl_1', sd, B3_R, in_ch, 1)
    make_basicconv(prefix + '.branch3x3dbl_2', sd, B3_M, B3_R, 3)
    make_basicconv(prefix + '.branch3x3dbl_3', sd, B3, B3_M, 3)
    make_basicconv(prefix + '.branch_pool', sd, BP, in_ch, 1)


def build_state_dict():
    sd = {}
    # Stem (single 3x3 same-pad conv in the pico; torchvision has
    # Conv2d_1a_3x3 .. Conv2d_4a_3x3 + maxpools, a documented simplification).
    make_basicconv('Conv2d_1a_3x3', sd, STEM, NUM_CHANNELS, 3)
    in_ch = STEM
    for m in range(NUM_MODULES):
        make_inception_a(f'Mixed_5{chr(ord("b") + m)}', sd, in_ch)
        in_ch = MOD_OUT
    sd['fc.weight'] = randn(NUM_CLASSES, MOD_OUT, std=0.4)        # [out,in]
    sd['fc.bias'] = randn(NUM_CLASSES, std=0.3)
    return sd


# --------------------------------------------------------------------------
# numpy float64 oracle, replicating the CAI forward path exactly.
# Volumes are (C, H, W); conv weights torchvision [O, I, kh, kw]; BN FOLDED.
# --------------------------------------------------------------------------
def fold_bn(w, bn_w, bn_b, bn_m, bn_v):
    scale = bn_w / np.sqrt(bn_v + EPS)
    w_f = w * scale[:, None, None, None]
    b_f = bn_b - bn_w * bn_m / np.sqrt(bn_v + EPS)
    return w_f, b_f


def conv2d(x, w, b, stride, pad):
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
            patch = xp[:, iy:iy + k, ix:ix + k]
            out[:, oy, ox] = np.tensordot(
                w, patch, axes=([1, 2, 3], [0, 1, 2])) + b
    return out


def relu(x):
    return np.maximum(x, 0.0)


def portable_maxpool(x, pool, stride, pad):
    """Replicate TNNetMaxPoolPortable: ZERO pad, floor() output size
    ((H - pool + 2p)//stride + 1), plain (non-clamped) windows."""
    I, H, W = x.shape
    xp = np.full((I, H + 2 * pad, W + 2 * pad), 0.0, dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    Ho = (H - pool + 2 * pad) // stride + 1
    Wo = (W - pool + 2 * pad) // stride + 1
    out = np.zeros((I, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            iy, ix = oy * stride, ox * stride
            win = xp[:, iy:iy + pool, ix:ix + pool]
            out[:, oy, ox] = win.reshape(I, -1).max(axis=1)
    return out


def basicconv(x, sd, prefix, stride, pad):
    w_f, b_f = fold_bn(sd[prefix + '.conv.weight'],
                       sd[prefix + '.bn.weight'], sd[prefix + '.bn.bias'],
                       sd[prefix + '.bn.running_mean'],
                       sd[prefix + '.bn.running_var'])
    return relu(conv2d(x, w_f, b_f, stride, pad))   # BasicConv2d ends in ReLU


def inception_a(x, sd, prefix):
    b1 = basicconv(x, sd, prefix + '.branch1x1', 1, 0)
    b5 = basicconv(x, sd, prefix + '.branch5x5_1', 1, 0)
    b5 = basicconv(b5, sd, prefix + '.branch5x5_2', 1, 1)        # 3x3 pad1
    b3 = basicconv(x, sd, prefix + '.branch3x3dbl_1', 1, 0)
    b3 = basicconv(b3, sd, prefix + '.branch3x3dbl_2', 1, 1)
    b3 = basicconv(b3, sd, prefix + '.branch3x3dbl_3', 1, 1)
    bp = portable_maxpool(x, 3, 1, 1)                            # size-preserve
    bp = basicconv(bp, sd, prefix + '.branch_pool', 1, 0)
    return np.concatenate([b1, b5, b3, bp], axis=0)             # CHANNEL concat


def forward(x, sd):
    out = basicconv(x, sd, 'Conv2d_1a_3x3', 1, 1)               # 3x3 same-pad
    for m in range(NUM_MODULES):
        out = inception_a(out, sd, f'Mixed_5{chr(ord("b") + m)}')
    pooled = out.reshape(out.shape[0], -1).mean(axis=1)        # global avg pool
    logits = sd['fc.weight'] @ pooled + sd['fc.bias']
    return logits, out.shape, pooled


def main():
    sd = build_state_dict()

    pixels = np.zeros((NUM_CHANNELS, IMAGE, IMAGE), dtype=np.float64)
    for c in range(NUM_CHANNELS):
        for y in range(IMAGE):
            for x in range(IMAGE):
                pixels[c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

    # Round-trip every weight through float32 BEFORE the oracle (CAI loads f32).
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()
              if not k.endswith('num_batches_tracked')}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    logits, feat_shape, pooled = forward(pixels, sd)
    print(f'feature map before pool: {feat_shape} (== {MOD_OUT} channels), '
          f'pooled {pooled.shape}, logits {logits.shape}')
    print(f'logits: {logits.tolist()}')

    save_file(sd_f32, 'tests/fixtures/tiny_inceptionv3.safetensors')

    config = {
        'model_type': 'inception',
        'architectures': ['InceptionV3'],
        'image_size': IMAGE,
        'num_channels': NUM_CHANNELS,
        'num_labels': NUM_CLASSES,
        'stem_width': STEM,
        'num_modules': NUM_MODULES,
        'bn_eps': EPS,
        # branch channel table for the pico InceptionA module:
        'branch1x1': B1,
        'branch5x5_reduce': B5_R, 'branch5x5': B5,
        'branch3x3dbl_reduce': B3_R, 'branch3x3dbl_mid': B3_M, 'branch3x3dbl': B3,
        'branch_pool': BP,
    }
    with open('tests/fixtures/tiny_inceptionv3_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_inceptionv3_logits.json', 'w') as f:
        json.dump({
            'pixels': pixels.tolist(),
            'logits': [logits.tolist()],
            'pooled': pooled.tolist(),
            'num_labels': NUM_CLASSES,
            'pooled_dim': MOD_OUT,
        }, f)
    print(f'wrote tiny_inceptionv3.safetensors ({len(sd_f32)} tensors) + '
          f'config + oracle')
    for k in sorted(sd_f32):
        print(f'  {k} {list(sd_f32[k].shape)}')

    # ---- fixture self-checks (every exercised path must matter) ----
    base = logits.copy()
    for key, desc in [
        ('Mixed_5b.branch1x1.bn.weight', 'branch1x1 BN gamma'),
        ('Mixed_5b.branch5x5_2.conv.weight', 'branch5x5 (5x5-as-3x3) conv'),
        ('Mixed_5b.branch3x3dbl_3.conv.weight', '3x3dbl last conv'),
        ('Mixed_5b.branch_pool.conv.weight', 'pool-branch 1x1 conv'),
        ('fc.bias', 'fc bias'),
    ]:
        alt = dict(sd)
        alt[key] = np.zeros_like(sd[key])
        d = np.abs(forward(pixels, alt)[0] - base).max()
        assert d > 1e-4, f'{desc} had no effect ({d})'
        print(f'{desc} effect: max|diff| = {d:.4f}')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
