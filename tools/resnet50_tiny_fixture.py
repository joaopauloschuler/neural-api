#!/usr/bin/env python3
"""Generate tiny RANDOM torchvision-ResNet parity fixtures for the Bottleneck
(resnet50-shaped) and the deeper-BasicBlock (resnet34-shaped) importer paths in
tests/TestNeuralPretrained.pas.

The resnet18 fixture (tools/resnet18_tiny_fixture.py) only exercises the
BasicBlock path with the shallow [2,2,2,2] layout. The torchvision ResNet
importer (BuildResNetFromSafeTensors in neural/neuralpretrained.pas) ALSO
supports:
  - Bottleneck blocks (resnet50/101/152, 1x1 -> 3x3 -> 1x1, expansion 4), and
  - the deeper resnet34 BasicBlock layout (3-4-6-3), which exercises the
    intra-stage NON-downsample identity shortcut across >2 blocks per stage.
Both were CODED but UNTESTED. This generator emits pico fixtures for both,
each with a self-contained numpy float64 oracle (torchvision is NOT installed),
reusing the resnet18 generator's helpers so the BN-fold / conv / maxpool /
oracle semantics are IDENTICAL to the proven resnet18 path.

Block kind is selected in the importer by config "depth" (18/34/50). The tiny
"widths" / "blocks_per_stage" config overrides only the per-stage widths and
counts (NOT the kind), so the fixture config keeps depth=34 / depth=50 to pick
BasicBlock / Bottleneck, with tiny widths + few blocks.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/resnet50_tiny_fixture.py
writes tests/fixtures/tiny_resnet50{.safetensors,_config.json,_logits.json}
  and tests/fixtures/tiny_resnet34{...}. Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

# Reuse the EXACT helpers from the resnet18 generator so the oracle math
# (BN fold, conv2d, cai_maxpool, relu) is shared and proven.
import resnet18_tiny_fixture as r18

EPS = r18.EPS
# Same IMAGE 64 as resnet18: smallest input keeping every spatial map >= the
# 3x3 kernel through all 4 stages (stem /2, maxpool /2, layer2/3/4 /2 each).
IMAGE = r18.IMAGE
NUM_CHANNELS = 3
NUM_CLASSES = 5


def make_bottleneck_block(prefix, sd, in_ch, base_ch, stride, expansion):
    """Bottleneck: conv1 1x1 (in->base), conv2 3x3 stride (base->base),
    conv3 1x1 (base->base*expansion), + 1x1 downsample on the shortcut when
    channels change or stride>1. Mirrors AddResNetBlock's Bottleneck path."""
    out_ch = base_ch * expansion
    r18.make_conv(prefix + '.conv1', sd, base_ch, in_ch, 1)
    r18.make_bn(prefix + '.bn1', sd, base_ch)
    r18.make_conv(prefix + '.conv2', sd, base_ch, base_ch, 3)
    r18.make_bn(prefix + '.bn2', sd, base_ch)
    r18.make_conv(prefix + '.conv3', sd, out_ch, base_ch, 1)
    r18.make_bn(prefix + '.bn3', sd, out_ch)
    if stride != 1 or in_ch != out_ch:
        r18.make_conv(prefix + '.downsample.0', sd, out_ch, in_ch, 1)
        r18.make_bn(prefix + '.downsample.1', sd, out_ch)


def bottleneck_block(x, sd, prefix, stride, has_downsample):
    """numpy float64 oracle for one Bottleneck. torchvision puts the stride on
    the 3x3 (conv2); conv1/conv3 are 1x1 stride1 pad0; downsample 1x1 stride."""
    out = r18.conv_bn(x, sd, prefix + '.conv1', prefix + '.bn1', 1, 0)
    out = r18.relu(out)
    out = r18.conv_bn(out, sd, prefix + '.conv2', prefix + '.bn2', stride, 1)
    out = r18.relu(out)
    out = r18.conv_bn(out, sd, prefix + '.conv3', prefix + '.bn3', 1, 0)
    if has_downsample:
        ident = r18.conv_bn(x, sd, prefix + '.downsample.0',
                            prefix + '.downsample.1', stride, 0)
    else:
        ident = x
    return r18.relu(out + ident)


def build_state_dict(widths, blocks, expansion, kind, stem):
    sd = {}
    r18.make_conv('conv1', sd, stem, NUM_CHANNELS, 7)
    r18.make_bn('bn1', sd, stem)
    in_ch = stem
    for li, (width, nblocks) in enumerate(zip(widths, blocks), start=1):
        stride = 1 if li == 1 else 2
        for bi in range(nblocks):
            s = stride if bi == 0 else 1
            prefix = f'layer{li}.{bi}'
            if kind == 'bottleneck':
                make_bottleneck_block(prefix, sd, in_ch, width, s, expansion)
                in_ch = width * expansion
            else:
                r18.make_basic_block(prefix, sd, in_ch, width, s)
                in_ch = width
    last = widths[-1] * expansion
    sd['fc.weight'] = r18.randn(NUM_CLASSES, last, std=0.4)
    sd['fc.bias'] = r18.randn(NUM_CLASSES, std=0.3)
    return sd


def forward(x, sd, widths, blocks, expansion, kind):
    out = r18.conv_bn(x, sd, 'conv1', 'bn1', 2, 3)
    out = r18.relu(out)
    out = r18.cai_maxpool(out, 3, 2, 1)
    for li, (width, nblocks) in enumerate(zip(widths, blocks), start=1):
        stride = 1 if li == 1 else 2
        out_ch = width * expansion
        for bi in range(nblocks):
            s = stride if bi == 0 else 1
            has_ds = (bi == 0) and (s != 1 or out.shape[0] != out_ch)
            prefix = f'layer{li}.{bi}'
            if kind == 'bottleneck':
                out = bottleneck_block(out, sd, prefix, s, has_ds)
            else:
                out = r18.basic_block(out, sd, prefix, s, has_ds)
    pooled = out.reshape(out.shape[0], -1).mean(axis=1)
    logits = sd['fc.weight'] @ pooled + sd['fc.bias']
    return logits, out.shape


def make_pixels(seed_offset):
    pixels = np.zeros((NUM_CHANNELS, IMAGE, IMAGE), dtype=np.float64)
    for c in range(NUM_CHANNELS):
        for y in range(IMAGE):
            for x in range(IMAGE):
                pixels[c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0
    return pixels


def emit(name, depth, widths, blocks, expansion, kind, stem, seed):
    r18.rng = np.random.default_rng(seed)
    sd = build_state_dict(widths, blocks, expansion, kind, stem)
    pixels = make_pixels(0)

    # Round-trip weights through float32 (CAI loads float32) before the oracle.
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()
              if not k.endswith('num_batches_tracked')}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    logits, feat_shape = forward(pixels, sd, widths, blocks, expansion, kind)
    print(f'[{name}] feature map before pool: {feat_shape}, '
          f'logits {logits.shape}')

    save_file(sd_f32, f'tests/fixtures/{name}.safetensors')
    config = {
        'model_type': 'resnet',
        'architectures': ['ResNetForImageClassification'],
        'depth': depth,
        'image_size': IMAGE,
        'num_channels': NUM_CHANNELS,
        'num_labels': NUM_CLASSES,
        'widths': widths,
        'blocks_per_stage': blocks,
        'bn_eps': EPS,
        # This oracle replicates CAI (ceil-sized, edge-clamped) maxpool; opt
        # the importer back to it (the default is PyTorch floor-sized maxpool).
        'cai_maxpool': True,
    }
    with open(f'tests/fixtures/{name}_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open(f'tests/fixtures/{name}_logits.json', 'w') as f:
        json.dump({
            'pixels': pixels.tolist(),
            'logits': [logits.tolist()],
            'num_labels': NUM_CLASSES,
        }, f)
    print(f'[{name}] wrote {name}.safetensors ({len(sd_f32)} tensors) '
          f'+ config + oracle')

    # ---- fixture self-checks (each must perturb the logits) ----
    base = logits.copy()

    def perturb(key):
        alt = dict(sd)
        alt[key] = np.zeros_like(sd[key])
        d = np.abs(forward(pixels, alt, widths, blocks,
                           expansion, kind)[0] - base).max()
        return d

    d = perturb('bn1.weight')
    assert d > 1e-3, f'[{name}] bn1 gamma had no effect ({d})'
    print(f'[{name}] bn1 gamma effect: max|diff| = {d:.4f}')
    d = perturb('layer2.0.downsample.0.weight')
    assert d > 1e-3, f'[{name}] layer2 downsample had no effect ({d})'
    print(f'[{name}] layer2 downsample effect: max|diff| = {d:.4f}')
    if kind == 'bottleneck':
        # conv3 (the expansion 1x1) MUST matter -> guards the expansion=4 path.
        d = perturb('layer1.0.conv3.weight')
        assert d > 1e-3, f'[{name}] layer1.0.conv3 had no effect ({d})'
        print(f'[{name}] conv3 (expansion) effect: max|diff| = {d:.4f}')
    else:
        # A non-block-0 (identity-shortcut) conv MUST matter -> guards the
        # intra-stage identity shortcut across the deeper stack.
        d = perturb('layer1.2.conv2.weight')
        assert d > 1e-3, f'[{name}] layer1.2.conv2 had no effect ({d})'
        print(f'[{name}] deep identity-block effect: max|diff| = {d:.4f}')
    d = perturb('fc.bias')
    assert d > 1e-3, f'[{name}] fc bias had no effect ({d})'
    print(f'[{name}] fc bias effect: max|diff| = {d:.4f}')
    print(f'[{name}] all fixture self-checks passed')


def main():
    # resnet50-shaped Bottleneck: tiny base widths, expansion 4, [2,1,1,1].
    # layer1 has 2 blocks -> exercises Bottleneck identity shortcut too.
    emit('tiny_resnet50', depth=50,
         widths=[4, 6, 8, 12], blocks=[2, 1, 1, 1],
         expansion=4, kind='bottleneck', stem=4, seed=20260615)
    # resnet34-shaped BasicBlock, deeper than resnet18: [3,2,2,2] (layer1 has
    # 3 blocks -> 2 intra-stage identity shortcuts, deeper than resnet18's 2).
    emit('tiny_resnet34', depth=34,
         widths=[4, 6, 8, 12], blocks=[3, 2, 2, 2],
         expansion=1, kind='basic', stem=4, seed=20260616)


if __name__ == '__main__':
    main()
