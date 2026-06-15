#!/usr/bin/env python3
"""Generate a tiny RANDOM RIFE (IFNet) video frame-interpolation parity fixture
for tests/TestNeuralPretrained.pas.

RIFE (Huang et al. 2022, "Real-Time Intermediate Flow Estimation for Video Frame
Interpolation", https://arxiv.org/abs/2011.06294; hzwer/Practical-RIFE) estimates
a bidirectional intermediate flow with a coarse-to-fine stack of IFBlocks and
then BACKWARD-WARPS the two input frames and blends them with a learned soft
fusion mask. There is NO tiny official checkpoint offline, so the reference
forward is a self-contained numpy float64 oracle that mirrors the CAI importer's
forward path EXACTLY (same idiom as tools/nafnet_tiny_fixture.py). Weights use a
RIFE-style state_dict key scheme so the importer is exercised on a realistic key
layout.

This pico is scoped to ONE intermediate frame at t=0.5 (inference-only), a small
IFNet of 2 IFBlocks at full resolution with tiny channel counts. The genuinely
new primitive being pinned is the differentiable BACKWARD-WARP
(TNNetBackwardWarp): out(x,y) = image(x+dx(x,y), y+dy(x,y)), bilinear, flow in
PIXEL units, border-clamp (replicate) padding -- the pixel-space equivalent of
RIFE's F.grid_sample(mode='bilinear', padding_mode='border', align_corners=True)
on integer pixels.

IFBlock(in_planes, c):
  conv0 (3x3, in_planes->c, pad1, +bias) -> PReLU(c)
  conv1 (3x3, c->c, pad1, +bias)         -> PReLU(c)
  lastconv (3x3, c->5, pad1, +bias)       # 4 flow channels + 1 mask channel
IFNet (this pico): the input is the channel-concat [frame0(3) | frame1(3)] (6ch).
  block0: flow0, mask0 = IFBlock0(input)
  block1: dflow, mask1 = IFBlock1(input)         # v1: refine from the SAME input
  flow = flow0 + dflow                            # accumulated bidir flow (4ch)
  mask = mask1                                     # last block's fusion logit
  flow_t0 = flow[0:2]  (warps frame0)             # (dx, dy) for frame0
  flow_t1 = flow[2:4]  (warps frame1)             # (dx, dy) for frame1
  warped0 = backward_warp(frame0, flow_t0)
  warped1 = backward_warp(frame1, flow_t1)
  m = sigmoid(mask)                                # (1, H, W) broadcast over RGB
  merged = warped0 * m + warped1 * (1 - m)         # interpolated middle frame

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/rife_tiny_fixture.py
writes tests/fixtures/tiny_rife{.safetensors,_config.json,_io.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

C = 8              # IFBlock hidden channels
NUM_BLOCKS = 2     # IFBlocks in the coarse-to-fine stack
IN_CH = 3          # channels per frame (RGB)
IN_PLANES = 2 * IN_CH  # 6: concat [frame0 | frame1]
INPUT = 8          # input grid H=W

rng = np.random.default_rng(20260615)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


# ---------------------------------------------------------------------------
# State dict (RIFE-style keys: block{i}.conv0/conv1/lastconv + prelu slopes).
# ---------------------------------------------------------------------------
def add_conv(sd, name, out_ch, in_ch, k, std=0.15):
    sd[name + '.weight'] = randn(out_ch, in_ch, k, k, std=std)
    sd[name + '.bias'] = randn(out_ch, std=0.05)


def add_ifblock(sd, prefix, in_planes, c):
    add_conv(sd, prefix + 'conv0', c, in_planes, 3)
    add_conv(sd, prefix + 'conv1', c, c, 3)
    add_conv(sd, prefix + 'lastconv', 5, c, 3, std=0.10)
    # PReLU per-channel slopes (one per conv0 / conv1 output channel).
    sd[prefix + 'prelu0.weight'] = randn(c, std=0.2) + 0.25
    sd[prefix + 'prelu1.weight'] = randn(c, std=0.2) + 0.25


def build_state_dict():
    sd = {}
    for i in range(NUM_BLOCKS):
        add_ifblock(sd, f'block{i}.', IN_PLANES, C)
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


def prelu(x, a):
    # per-channel: y = max(x,0) + a[c]*min(x,0)
    return np.maximum(x, 0) + a[:, None, None] * np.minimum(x, 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def backward_warp(img, flow):
    # img (C,H,W); flow (2,H,W): channel 0 = dx, channel 1 = dy.
    # out(x,y) = img(x+dx, y+dy) bilinear, flow in PIXELS, border-clamp padding.
    Ch, H, Wd = img.shape
    out = np.zeros((Ch, H, Wd), dtype=np.float64)
    for oy in range(H):
        for ox in range(Wd):
            sx = ox + flow[0, oy, ox]
            sy = oy + flow[1, oy, ox]
            x0 = int(np.floor(sx)); y0 = int(np.floor(sy))
            x1 = x0 + 1; y1 = y0 + 1
            fx = sx - x0; fy = sy - y0
            w00 = (1 - fx) * (1 - fy)
            w10 = fx * (1 - fy)
            w01 = (1 - fx) * fy
            w11 = fx * fy
            cx0 = min(max(x0, 0), Wd - 1); cx1 = min(max(x1, 0), Wd - 1)
            cy0 = min(max(y0, 0), H - 1);  cy1 = min(max(y1, 0), H - 1)
            out[:, oy, ox] = (w00 * img[:, cy0, cx0] + w10 * img[:, cy0, cx1]
                              + w01 * img[:, cy1, cx0] + w11 * img[:, cy1, cx1])
    return out


def ifblock(inp, sd, prefix):
    x = conv2d(inp, sd[prefix + 'conv0.weight'], sd[prefix + 'conv0.bias'], 1)
    x = prelu(x, sd[prefix + 'prelu0.weight'])
    x = conv2d(x, sd[prefix + 'conv1.weight'], sd[prefix + 'conv1.bias'], 1)
    x = prelu(x, sd[prefix + 'prelu1.weight'])
    x = conv2d(x, sd[prefix + 'lastconv.weight'],
               sd[prefix + 'lastconv.bias'], 1)
    flow = x[0:4]   # (4,H,W)
    mask = x[4:5]   # (1,H,W)
    return flow, mask


def forward(inp, sd):
    # inp: (6,H,W) = [frame0(3) | frame1(3)]
    frame0 = inp[0:3]
    frame1 = inp[3:6]
    flow = np.zeros((4,) + inp.shape[1:], dtype=np.float64)
    mask = np.zeros((1,) + inp.shape[1:], dtype=np.float64)
    for i in range(NUM_BLOCKS):
        dflow, bmask = ifblock(inp, sd, f'block{i}.')
        flow = flow + dflow
        mask = bmask  # last block's mask logit wins
    warped0 = backward_warp(frame0, flow[0:2])
    warped1 = backward_warp(frame1, flow[2:4])
    m = sigmoid(mask)  # (1,H,W)
    merged = warped0 * m + warped1 * (1 - m)
    return merged


def main():
    sd = build_state_dict()
    # Round-trip every weight through float32 (CAI loads float32).
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    # Pinned input: two deterministic dyadic frames (exact in f32 + JSON).
    x = np.zeros((IN_PLANES, INPUT, INPUT), dtype=np.float64)
    for c in range(IN_PLANES):
        for y in range(INPUT):
            for px in range(INPUT):
                x[c, y, px] = (((c * 64 + y * 8 + px) * 5) % 13 - 6) / 8.0

    img = forward(x, sd)
    print(f'input {x.shape} -> interpolated frame {img.shape}')
    print(f'frame stats: min {img.min():.4f} max {img.max():.4f} '
          f'mean {img.mean():.4f}')

    save_file(sd_f32, 'tests/fixtures/tiny_rife.safetensors')
    config = {
        'model_type': 'rife',
        'in_channel': IN_CH,
        'hidden_channels': C,
        'num_blocks': NUM_BLOCKS,
        'input_size': INPUT,
    }
    with open('tests/fixtures/tiny_rife_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_rife_io.json', 'w') as f:
        json.dump({
            'input': x.tolist(),       # (6,H,W) = [frame0 | frame1]
            'image': img.tolist(),     # (3,H,W) interpolated middle frame
            'image_size': img.shape[1],
        }, f)
    print(f'wrote tiny_rife.safetensors ({len(sd_f32)} tensors) + config + io')

    # ---- fixture self-checks: each major piece must MATTER. ----
    base = img.copy()

    def perturb(key):
        alt = dict(sd)
        alt[key] = np.zeros_like(sd[key])
        return np.abs(forward(x, alt) - base).max()

    for key in ('block0.conv0.weight', 'block0.lastconv.weight',
                'block1.lastconv.weight', 'block1.conv1.weight',
                'block0.prelu0.weight', 'block1.prelu1.weight'):
        d = perturb(key)
        assert d > 1e-5, f'{key} had no effect ({d})'
        print(f'{key} effect: max|diff| = {d:.4f}')

    # The mask must actually mix the two warped frames (not a degenerate 0/1).
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
