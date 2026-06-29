#!/usr/bin/env python3
"""Generate tiny RANDOM RegNet parity fixtures for TestNeuralPretrained.pas.

Builds two pico `transformers.RegNetForImageClassification` models -- a regnet_x
variant (layer_type "x", NO squeeze-excite) and a regnet_y variant (layer_type
"y", WITH SE) -- with small random weights, runs each through the REAL HF model
in float64, and writes the committed parity contract under tests/fixtures/:
  tiny_regnet_{x,y}.safetensors      (the exact HF `regnet.`-prefixed keys)
  tiny_regnet_{x,y}_config.json      (transformers RegNetConfig + image_size)
  tiny_regnet_{x,y}_logits.json      (pinned pixels + float64 reference logits)

The Pascal BuildRegNet importer reads the SAME safetensors + config and must
reproduce the logits to < 1e-4.

Architecture exercised (both variants):
  - stem: conv 3x3 stride2 pad1 + folded BN + ReLU
          (regnet.embedder.embedder.convolution / .normalization)
  - 4 stages of bottleneck blocks (regnet.encoder.stages.{s}.layers.{l}):
      layer.0 = 1x1 conv+BN+ReLU (in -> out)
      layer.1 = GROUPED 3x3 conv+BN+ReLU (out -> out, groups=out/group_width,
                stride on the first block of each stage)
      [y only] layer.2 = SE (.attention.0 reduce conv +bias, ReLU,
                .attention.2 expand conv +bias, sigmoid, per-channel scale)
      last    = 1x1 conv+BN (no activation)
      shortcut (first block of a stage) = 1x1 stride conv+BN
  - residual add + ReLU
  - head: global avg pool -> Linear (classifier.1)

CAI's TNNetAvgChannel divides by H*W, an exact global mean only on SQUARE maps,
so the config keeps every map square. Image 64 + downsample_in_first_stage=False:
stem/2 -> 32, stage0 s1 -> 32, stage1/2 -> 16, stage2/2 -> 8, stage3/2 -> 4
(all square, >= the 3x3 grouped kernel).

group_width divides every hidden size so groups = out/group_width is integral and
> 1 (exercises the grouped-conv per-group weight slicing path).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_regnet_fixture.py
Needs torch + transformers + safetensors (in the reusable venv).

Coded by Claude (AI).
"""
import json
import numpy as np
import torch
from transformers import RegNetConfig, RegNetForImageClassification
from safetensors.torch import save_file

IMAGE = 64
NUM_CHANNELS = 3
NUM_CLASSES = 5
EMBED = 8
# group_width 4 divides every hidden size -> groups 2 / 3 / 4 / 6 (all > 1).
GROUP_WIDTH = 4
HIDDEN = [8, 12, 16, 24]
DEPTHS = [1, 2, 1, 2]


def pinned_pixels():
    """Deterministic dyadic pixel values (exact in f32 + JSON)."""
    px = np.zeros((NUM_CHANNELS, IMAGE, IMAGE), dtype=np.float64)
    for c in range(NUM_CHANNELS):
        for y in range(IMAGE):
            for x in range(IMAGE):
                px[c, y, x] = (((c * 4096 + y * 64 + x) * 5) % 17 - 8) / 8.0
    return px


def randomize(model, seed):
    """Replace the (kaiming) init with small NON-trivial random weights so every
    BN fold / SE gate is actually exercised, and BN running stats are non-zero."""
    g = torch.Generator().manual_seed(seed)
    sd = model.state_dict()
    for k, v in sd.items():
        if k.endswith('num_batches_tracked'):
            continue
        if k.endswith('running_var'):
            sd[k] = (torch.rand(v.shape, generator=g, dtype=torch.float64) * 0.5
                     + 0.5)
        elif k.endswith('running_mean'):
            sd[k] = torch.randn(v.shape, generator=g, dtype=torch.float64) * 0.3
        elif k.endswith('normalization.weight'):  # BN gamma
            sd[k] = (torch.randn(v.shape, generator=g, dtype=torch.float64) * 0.3
                     + 1.0)
        elif k.endswith('normalization.bias'):    # BN beta
            sd[k] = torch.randn(v.shape, generator=g, dtype=torch.float64) * 0.25
        else:  # conv / linear / SE conv weights & biases
            std = 0.3 if 'attention' in k or 'classifier' in k else 0.2
            sd[k] = torch.randn(v.shape, generator=g, dtype=torch.float64) * std
    model.load_state_dict(sd)


def make_variant(layer_type, seed, tag):
    cfg = RegNetConfig(
        num_channels=NUM_CHANNELS,
        embedding_size=EMBED,
        hidden_sizes=HIDDEN,
        depths=DEPTHS,
        groups_width=GROUP_WIDTH,
        layer_type=layer_type,
        hidden_act='relu',
        downsample_in_first_stage=False,
        num_labels=NUM_CLASSES,
    )
    model = RegNetForImageClassification(cfg).double().eval()
    randomize(model, seed)

    px = pinned_pixels()
    with torch.no_grad():
        x = torch.from_numpy(px).unsqueeze(0)  # (1,3,H,W) float64
        logits = model(pixel_values=x).logits  # (1, NUM_CLASSES)
    logits = logits.squeeze(0).numpy().astype(np.float64)
    print(f'[{tag}] logits: {logits.tolist()}')

    # Round the weights through float32 (CAI loads f32) and re-check the f32
    # forward so the committed reference is the f32 oracle.
    sd_f32 = {k: v.to(torch.float32) for k, v in model.state_dict().items()
              if not k.endswith('num_batches_tracked')}
    model32 = RegNetForImageClassification(cfg).double().eval()
    sd64 = {k: v.to(torch.float64) for k, v in sd_f32.items()}
    # restore BN running stats / num_batches_tracked buffers
    full = model32.state_dict()
    for k in full:
        if k in sd64:
            full[k] = sd64[k]
    model32.load_state_dict(full)
    with torch.no_grad():
        logits32 = model32(pixel_values=torch.from_numpy(px).unsqueeze(0)
                           ).logits.squeeze(0).numpy().astype(np.float64)
    print(f'[{tag}] f32-weight logits: {logits32.tolist()}')

    save_file(sd_f32, f'tests/fixtures/tiny_regnet_{tag}.safetensors')

    config = {
        'model_type': 'regnet',
        'layer_type': layer_type,
        'image_size': IMAGE,
        'num_channels': NUM_CHANNELS,
        'embedding_size': EMBED,
        'hidden_sizes': HIDDEN,
        'depths': DEPTHS,
        'groups_width': GROUP_WIDTH,
        'downsample_in_first_stage': False,
        'hidden_act': 'relu',
        'num_labels': NUM_CLASSES,
        'bn_eps': 1e-5,
    }
    with open(f'tests/fixtures/tiny_regnet_{tag}_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    # NOTE: pixels are NOT stored (they are the deterministic dyadic formula
    # below, regenerated bit-for-bit in the Pascal test) -- keeps the committed
    # logits.json tiny (~200 B vs the 80 KB a 64x64x3 array would cost).
    with open(f'tests/fixtures/tiny_regnet_{tag}_logits.json', 'w') as f:
        json.dump({
            'logits': [logits32.tolist()],
            'num_labels': NUM_CLASSES,
            'image_size': IMAGE,
            'num_channels': NUM_CHANNELS,
            'pixel_formula': '(((c*4096 + y*64 + x)*5) mod 17 - 8) / 8',
        }, f)
    print(f'[{tag}] wrote tiny_regnet_{tag}.safetensors ({len(sd_f32)} tensors)')
    for k in sorted(sd_f32):
        print(f'    {k} {list(sd_f32[k].shape)}')


def main():
    make_variant('x', 20260627, 'x')
    make_variant('y', 20260628, 'y')


if __name__ == '__main__':
    main()
