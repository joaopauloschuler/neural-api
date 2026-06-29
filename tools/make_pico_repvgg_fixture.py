#!/usr/bin/env python3
"""Generate a tiny RANDOM RepVGG parity fixture for
tests/TestNeuralPretrained.pas (no network access, no timm/torchvision: the
model is a pico random-init RepVGG whose state_dict mirrors the OFFICIAL
RepVGG key scheme, and the reference logits are computed by a self-contained
numpy float64 oracle that runs the MULTI-BRANCH TRAINING graph).

RepVGG (Ding et al., CVPR 2021) trains a multi-branch block and FUSES the
branches into a single 3x3 conv for inference. This fixture pins that fusion:
the oracle computes the TRAINING forward (3x3 conv-BN + 1x1 conv-BN +
identity-BN, summed, then ReLU); the Pascal importer fuses the three branches
into one 3x3 conv at LOAD time. The parity test asserts the FUSED inference
forward matches the multi-branch TRAINING oracle to max|diff| < 1e-4.

Architecture (official RepVGG, scaled down to a pico size):
  stem   = stage0: 1 RepVGG block, in=3 -> StemWidth, stride 2 (no identity)
  stage1..4: NUM_BLOCKS[i] blocks each, first block stride 2 (no identity,
             width change), the rest stride 1 with width unchanged -> identity
             branch present.
  head: global avg pool -> linear (Linear last_width -> num_classes).

each RepVGG block (training graph):
  rbr_dense    = Conv2d(3x3, pad1, bias=False) -> BatchNorm2d
  rbr_1x1      = Conv2d(1x1, pad0, bias=False) -> BatchNorm2d
  rbr_identity = BatchNorm2d (ONLY when in==out and stride==1)
  y = ReLU(dense(x) + onexone(x) + identity(x))

State-dict keys (official RepVGG):
  stageN.B.rbr_dense.conv.weight              [O,I,3,3]
  stageN.B.rbr_dense.bn.{weight,bias,running_mean,running_var}   [O]
  stageN.B.rbr_1x1.conv.weight                [O,I,1,1]
  stageN.B.rbr_1x1.bn.{...}                    [O]
  stageN.B.rbr_identity.{weight,bias,running_mean,running_var}   [O]  (opt)
  linear.{weight,bias}
(stem uses prefix "stage0.<single block>" -> here just "stage0").

POOLING note: RepVGG has NO max-pool; all spatial reduction is via strided
convs. CAI's TNNetConvolutionReLU and this oracle agree on conv arithmetic
(zero pad, output (H - k + 2p)//s + 1), so this is a clean parity check of the
re-parameterization fusion + the CAI forward path.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_repvgg_fixture.py
writes tests/fixtures/tiny_repvgg{.safetensors,_config.json,_logits.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

EPS = 1e-5
# IMAGE 64: stem/2 -> 32, then 4 stages each /2 -> 16,8,4,2. Keeping maps >= 3
# until the last stage avoids CAI's kernel-clamp path; the final stride-2 3x3
# on a >=4 map stays un-clamped. (Stage4 maps 4->2 still >= the kernel after
# clamp-free arithmetic at the strided step.)
IMAGE = 64
NUM_CHANNELS = 3
NUM_CLASSES = 5
WIDTHS = [8, 12, 16, 24]          # stage1..stage4 OUTPUT widths
NUM_BLOCKS = [2, 2, 2, 1]         # pico layout (stage4 single block)
STEM = 6                          # stem out (min(64,*) analogue, tiny here)

rng = np.random.default_rng(20260627)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def make_bn(prefix, sd, channels):
    """Random NON-trivial BN so the fold + identity branch are exercised."""
    sd[prefix + '.weight'] = randn(channels, std=0.3) + 1.0       # gamma
    sd[prefix + '.bias'] = randn(channels, std=0.25)              # beta
    sd[prefix + '.running_mean'] = randn(channels, std=0.3)
    sd[prefix + '.running_var'] = np.abs(randn(channels, std=0.3)) + 0.5
    sd[prefix + '.num_batches_tracked'] = np.array(0, dtype=np.int64)


def make_block(prefix, sd, in_ch, out_ch, has_identity):
    sd[prefix + '.rbr_dense.conv.weight'] = randn(out_ch, in_ch, 3, 3, std=0.15)
    make_bn(prefix + '.rbr_dense.bn', sd, out_ch)
    sd[prefix + '.rbr_1x1.conv.weight'] = randn(out_ch, in_ch, 1, 1, std=0.2)
    make_bn(prefix + '.rbr_1x1.bn', sd, out_ch)
    if has_identity:
        assert in_ch == out_ch
        make_bn(prefix + '.rbr_identity', sd, out_ch)


def block_plan():
    """Yield (prefix, in_ch, out_ch, stride, has_identity) for every block."""
    plan = []
    plan.append(('stage0', NUM_CHANNELS, STEM, 2, False))
    in_ch = STEM
    for si, (w, nb) in enumerate(zip(WIDTHS, NUM_BLOCKS), start=1):
        for bi in range(nb):
            stride = 2 if bi == 0 else 1
            has_id = (stride == 1) and (in_ch == w)
            plan.append((f'stage{si}.{bi}', in_ch, w, stride, has_id))
            in_ch = w
    return plan


def build_state_dict(plan):
    sd = {}
    for prefix, in_ch, out_ch, stride, has_id in plan:
        make_block(prefix, sd, in_ch, out_ch, has_id)
    sd['linear.weight'] = randn(NUM_CLASSES, WIDTHS[-1], std=0.4)   # [out,in]
    sd['linear.bias'] = randn(NUM_CLASSES, std=0.3)
    return sd


# --------------------------------------------------------------------------
# numpy float64 oracle: MULTI-BRANCH TRAINING forward (NOT the fused conv).
# Volumes in (C,H,W); conv weights torch [O,I,kh,kw]. BN applied as the
# training-time inference BN (running stats), branches SUMMED before ReLU.
# --------------------------------------------------------------------------
def conv2d(x, w, stride, pad):
    """x:[I,H,W] w:[O,I,k,k] (NO bias) -> [O,Ho,Wo]. Zero pad."""
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
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3],
                                                          [0, 1, 2]))
    return out


def bn_apply(x, sd, prefix):
    """Inference BatchNorm over channel axis of x:[C,H,W]."""
    g = sd[prefix + '.weight'][:, None, None]
    b = sd[prefix + '.bias'][:, None, None]
    m = sd[prefix + '.running_mean'][:, None, None]
    v = sd[prefix + '.running_var'][:, None, None]
    return (x - m) / np.sqrt(v + EPS) * g + b


def relu(x):
    return np.maximum(x, 0.0)


def block_forward(x, sd, prefix, stride, has_identity):
    dense = bn_apply(conv2d(x, sd[prefix + '.rbr_dense.conv.weight'], stride, 1),
                     sd, prefix + '.rbr_dense.bn')
    onexone = bn_apply(conv2d(x, sd[prefix + '.rbr_1x1.conv.weight'], stride, 0),
                       sd, prefix + '.rbr_1x1.bn')
    out = dense + onexone
    if has_identity:
        out = out + bn_apply(x, sd, prefix + '.rbr_identity')
    return relu(out)


def forward(x, sd, plan):
    out = x
    for prefix, in_ch, out_ch, stride, has_id in plan:
        out = block_forward(out, sd, prefix, stride, has_id)
    pooled = out.reshape(out.shape[0], -1).mean(axis=1)         # global avg pool
    logits = sd['linear.weight'] @ pooled + sd['linear.bias']
    return logits, out.shape


def main():
    plan = block_plan()
    sd = build_state_dict(plan)

    pixels = np.zeros((NUM_CHANNELS, IMAGE, IMAGE), dtype=np.float64)
    for c in range(NUM_CHANNELS):
        for y in range(IMAGE):
            for x in range(IMAGE):
                pixels[c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

    # Round-trip weights through float32 (CAI loads f32) so only the
    # f64-vs-f32 forward gap remains.
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()
              if not k.endswith('num_batches_tracked')}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    logits, feat_shape = forward(pixels, sd, plan)
    print(f'feature map before pool: {feat_shape}, logits {logits.shape}')
    print(f'logits: {logits.tolist()}')

    save_file(sd_f32, 'tests/fixtures/tiny_repvgg.safetensors')

    config = {
        'model_type': 'repvgg',
        'architectures': ['RepVGGForImageClassification'],
        'image_size': IMAGE,
        'num_channels': NUM_CHANNELS,
        'num_labels': NUM_CLASSES,
        'widths': WIDTHS,
        'num_blocks': NUM_BLOCKS,
        'stem_width': STEM,
        'bn_eps': EPS,
    }
    with open('tests/fixtures/tiny_repvgg_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_repvgg_logits.json', 'w') as f:
        json.dump({
            'pixels': pixels.tolist(),
            'logits': [logits.tolist()],
            'num_labels': NUM_CLASSES,
        }, f)
    print(f'wrote tiny_repvgg.safetensors ({len(sd_f32)} tensors) + '
          f'config + oracle')
    for k in sorted(sd_f32):
        print(f'  {k} {list(sd_f32[k].shape)}')

    # ---- fixture self-checks: each branch must measurably affect the output ----
    base = logits.copy()
    # 1. The 1x1 branch of stage1.0 must matter (zeroing its conv).
    alt = dict(sd)
    alt['stage1.0.rbr_1x1.conv.weight'] = \
        np.zeros_like(sd['stage1.0.rbr_1x1.conv.weight'])
    d = np.abs(forward(pixels, alt, plan)[0] - base).max()
    assert d > 1e-3, f'stage1.0 1x1 branch had no effect ({d})'
    print(f'stage1.0 1x1 branch effect: max|diff| = {d:.4f}')
    # 2. An IDENTITY branch must matter. Find the first block that has one.
    id_prefix = next(p for p, i, o, s, h in plan if h) + '.rbr_identity'
    alt = dict(sd)
    # Zero the identity BN gamma/beta => kills the identity contribution.
    alt[id_prefix + '.weight'] = np.zeros_like(sd[id_prefix + '.weight'])
    alt[id_prefix + '.bias'] = np.zeros_like(sd[id_prefix + '.bias'])
    d = np.abs(forward(pixels, alt, plan)[0] - base).max()
    assert d > 1e-3, f'identity branch ({id_prefix}) had no effect ({d})'
    print(f'identity branch {id_prefix} effect: max|diff| = {d:.4f}')
    # 3. dense BN gamma must matter.
    alt = dict(sd)
    alt['stage0.rbr_dense.bn.weight'] = \
        np.zeros_like(sd['stage0.rbr_dense.bn.weight'])
    d = np.abs(forward(pixels, alt, plan)[0] - base).max()
    assert d > 1e-3, f'stage0 dense BN had no effect ({d})'
    print(f'stage0 dense BN effect: max|diff| = {d:.4f}')
    # 4. linear bias must matter.
    alt = dict(sd)
    alt['linear.bias'] = np.zeros_like(sd['linear.bias'])
    d = np.abs(forward(pixels, alt, plan)[0] - base).max()
    assert d > 1e-3, f'linear bias had no effect ({d})'
    print(f'linear bias effect: max|diff| = {d:.4f}')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
