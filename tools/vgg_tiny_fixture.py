#!/usr/bin/env python3
"""Generate a tiny RANDOM torchvision-VGG parity fixture for
tests/TestNeuralPretrained.pas (no network access, no torchvision needed:
the model is a pico random-init VGG whose state_dict mirrors torchvision's
exact key scheme, and the reference logits + intermediate tap activations are
computed by a self-contained numpy float64 oracle).

torchvision VGG is a straight Conv3x3(pad1)->ReLU stack split into 5 stages by
2x2 stride-2 MaxPool, followed by AdaptiveAvgPool2d((7,7)) and a 3-layer FC
classifier (4096,4096,num_classes) with ReLU+Dropout (dropout is a no-op at
inference). The state_dict keys are SEQUENTIAL inside nn.Sequential containers:
  features.{i}.weight / features.{i}.bias   (i = position in features Sequential)
  classifier.{0,3,6}.weight / .bias
For the PLAIN config the features Sequential is [Conv,ReLU,Conv,ReLU,MaxPool,...]
so the convs land on positions 0,2,5,7,10,... For the _bn variant a BatchNorm2d
is inserted after every conv (Conv,BN,ReLU,...) and the convs/BNs shift; this
fixture exercises the PLAIN config (the _bn fold reuses the ResNet conv-BN fold
path and is a documented follow-up).

PERCEPTUAL TAPS: VGG-16's LPIPS / style-transfer taps relu1_2, relu2_2,
relu3_3, relu4_3 are the ReLU activations after the LAST conv of stages 1..4.
This fixture emits those tap activations so the importer's named-tap mechanism
(tap layer indices returned by BuildVGG) can be verified.

POOLING parity note: CAI's TNNetMaxPool with NO padding and an even input is
exactly torchvision's MaxPool2d(2, stride=2) (floor sizing, no -inf padding
needed because the windows never run off the edge). The fixture keeps every
spatial map even so CAI and torchvision agree bit-for-bit.

ADAPTIVE POOL no-op: torchvision's AdaptiveAvgPool2d((7,7)) is a no-op when the
feature map already has the target grid. This fixture sizes the input so the
post-stack feature map (2x2 after 5 pools from a 64x64 input) is collapsed by
AdaptiveAvgPool2d((1,1)) to a global mean per channel, the documented no-op
global-pool case (a TNNetAvgChannel global pool is used, equivalent).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/vgg_tiny_fixture.py
writes tests/fixtures/tiny_vgg16{.safetensors,_config.json,_logits.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

# IMAGE 64 keeps every conv's spatial map >= the 3x3 kernel (CAI's conv CLAMPS
# the kernel to the input size when smaller, diverging from PyTorch; with 5
# stages the stage-4 conv input is 64/2^4 = 4 >= 3, so the un-clamped conv
# arithmetic is exercised). After all 5 pools the map is 64/2^5 = 2x2; with
# AdaptivePool=1 the AdaptiveAvgPool2d((1,1)) collapses that 2x2 to a global
# mean per channel (== TNNetAvgChannel), the no-op global-pool case.
IMAGE = 64
NUM_CHANNELS = 3
NUM_CLASSES = 5
# Tiny VGG-16-shaped stage layout: conv COUNTS per stage [2,2,3,3,3] (VGG-16);
# widths shrunk drastically so the fixture stays a few KB.
STAGE_CONVS = [2, 2, 3, 3, 3]
STAGE_WIDTHS = [4, 6, 8, 8, 8]
FC_HIDDEN = [10, 10]          # tiny stand-ins for (4096, 4096)
ADAPT = 1                     # adaptive-pool target grid (1x1 here)

rng = np.random.default_rng(20260614)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def build_state_dict():
    """Builds a torchvision-PLAIN-VGG state_dict (features.{seq_idx}.* +
    classifier.{0,3,6}.*) and records, per conv, its features Sequential
    position and (in/out) channels."""
    sd = {}
    feat_pos = 0           # running position inside the features Sequential
    in_ch = NUM_CHANNELS
    conv_meta = []         # list of (features_idx, out_ch, in_ch)
    tap_idx = []           # features index of the LAST conv ReLU per stage
    for si, (nconv, width) in enumerate(zip(STAGE_CONVS, STAGE_WIDTHS)):
        for ci in range(nconv):
            w = randn(width, in_ch, 3, 3, std=0.15)
            b = randn(width, std=0.1)
            sd[f'features.{feat_pos}.weight'] = w
            sd[f'features.{feat_pos}.bias'] = b
            conv_meta.append((feat_pos, width, in_ch))
            in_ch = width
            relu_pos = feat_pos + 1
            feat_pos += 2          # Conv occupies feat_pos, ReLU feat_pos+1
        # record the ReLU after the LAST conv of this stage (relu{si+1}_{nconv})
        tap_idx.append(relu_pos)
        feat_pos += 1              # MaxPool position
    # classifier: Linear(flatten -> FC0) ReLU Dropout Linear(FC0->FC1) ReLU
    #             Dropout Linear(FC1->num_classes). torchvision indices 0,3,6.
    flat = STAGE_WIDTHS[-1] * ADAPT * ADAPT
    dims = [flat] + FC_HIDDEN + [NUM_CLASSES]
    for li, idx in enumerate([0, 3, 6]):
        sd[f'classifier.{idx}.weight'] = randn(dims[li + 1], dims[li], std=0.3)
        sd[f'classifier.{idx}.bias'] = randn(dims[li + 1], std=0.2)
    return sd, conv_meta, tap_idx


# --------------------------------------------------------------------------
# numpy float64 oracle replicating the CAI forward path. Volumes are (C,H,W).
# --------------------------------------------------------------------------
def conv2d(x, w, b, pad):
    I, H, W = x.shape
    O, _, k, _ = w.shape
    Ho = (H - k + 2 * pad) + 1
    Wo = (W - k + 2 * pad) + 1
    xp = np.zeros((I, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    out = np.zeros((O, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            patch = xp[:, oy:oy + k, ox:ox + k]
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3],
                                                          [0, 1, 2])) + b
    return out


def relu(x):
    return np.maximum(x, 0.0)


def maxpool2(x):
    """MaxPool2d(2, stride=2), floor sizing, no padding (input kept even)."""
    I, H, W = x.shape
    Ho, Wo = H // 2, W // 2
    out = np.zeros((I, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            win = x[:, 2 * oy:2 * oy + 2, 2 * ox:2 * ox + 2]
            out[:, oy, ox] = win.reshape(I, -1).max(axis=1)
    return out


def forward(x, sd):
    out = x
    taps = {}
    feat_pos = 0
    for si, (nconv, width) in enumerate(zip(STAGE_CONVS, STAGE_WIDTHS)):
        for cidx in range(nconv):
            w = sd[f'features.{feat_pos}.weight']
            b = sd[f'features.{feat_pos}.bias']
            out = relu(conv2d(out, w, b, 1))
            feat_pos += 2
        taps[f'relu{si + 1}_{nconv}'] = out.copy()
        out = maxpool2(out)
        feat_pos += 1
    # adaptive avg pool to ADAPTxADAPT (no-op here, grid already ADAPT) ==
    # global mean per channel into (C, ADAPT, ADAPT). Flatten row-major.
    pooled = out.reshape(out.shape[0], -1).mean(axis=1)   # (C,) for ADAPT=1
    h = pooled
    for li, idx in enumerate([0, 3, 6]):
        h = sd[f'classifier.{idx}.weight'] @ h + sd[f'classifier.{idx}.bias']
        if idx != 6:
            h = relu(h)
    return h, taps, out.shape


def main():
    sd, conv_meta, tap_idx = build_state_dict()

    pixels = np.zeros((NUM_CHANNELS, IMAGE, IMAGE), dtype=np.float64)
    for c in range(NUM_CHANNELS):
        for y in range(IMAGE):
            for x in range(IMAGE):
                pixels[c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

    # Round-trip weights through float32 before computing the oracle (CAI loads
    # float32 weights).
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    logits, taps, feat_shape = forward(pixels, sd)
    print(f'feature map before pool: {feat_shape}, logits {logits.shape}')
    print(f'logits: {logits.tolist()}')
    for k in sorted(taps):
        print(f'  tap {k}: shape {taps[k].shape} '
              f'mean {taps[k].mean():.5f} max {taps[k].max():.5f}')

    save_file(sd_f32, 'tests/fixtures/tiny_vgg16.safetensors')

    config = {
        'model_type': 'vgg',
        'architectures': ['VGG'],
        'depth': 16,
        'image_size': IMAGE,
        'num_channels': NUM_CHANNELS,
        'num_labels': NUM_CLASSES,
        'batch_norm': False,
        'stage_convs': STAGE_CONVS,
        'stage_widths': STAGE_WIDTHS,
        'fc_hidden': FC_HIDDEN,
        'adaptive_pool': ADAPT,
    }
    with open('tests/fixtures/tiny_vgg16_config.json', 'w') as f:
        json.dump(config, f, indent=1)

    # emit tap activations as (C,H,W)-flattened lists keyed by tap name.
    taps_out = {k: v.reshape(-1).tolist() for k, v in taps.items()}
    tap_shapes = {k: list(v.shape) for k, v in taps.items()}
    with open('tests/fixtures/tiny_vgg16_logits.json', 'w') as f:
        json.dump({
            'pixels': pixels.tolist(),
            'logits': [logits.tolist()],
            'num_labels': NUM_CLASSES,
            'taps': taps_out,
            'tap_shapes': tap_shapes,
        }, f)
    print(f'wrote tiny_vgg16.safetensors ({len(sd_f32)} tensors) + config + '
          f'oracle; tap relu positions {tap_idx}')
    for k in sorted(sd_f32):
        print(f'  {k} {list(sd_f32[k].shape)}')

    # ---- fixture self-checks ----
    base = logits.copy()
    alt = dict(sd)
    alt['features.0.bias'] = np.zeros_like(sd['features.0.bias'])
    d = np.abs(forward(pixels, alt)[0] - base).max()
    assert d > 1e-4, f'features.0 bias had no effect ({d})'
    print(f'features.0 bias effect: max|diff| = {d:.5f}')
    alt = dict(sd)
    alt['classifier.6.bias'] = np.zeros_like(sd['classifier.6.bias'])
    d = np.abs(forward(pixels, alt)[0] - base).max()
    assert d > 1e-4, f'classifier.6 bias had no effect ({d})'
    print(f'classifier.6 bias effect: max|diff| = {d:.5f}')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
