#!/usr/bin/env python3
"""Generate a tiny RANDOM ultralytics-YOLOv8 (anchor-free, DFL) parity fixture
for tests/TestNeuralPretrained.pas.

The ultralytics package is NOT installed (and is a heavy dependency), so the
reference forward is a self-contained numpy float64 oracle that mirrors the CAI
importer's forward path EXACTLY (same idiom as tools/rrdbnet_tiny_fixture.py).
The weights use the canonical ultralytics YOLOv8 state_dict key scheme
("model.{i}...."), so the importer is exercised on the real key layout.

YOLOv8 (n-style, but PICO widths/depths) architecture:
  Conv(k,s)        = conv2d(no bias) -> BatchNorm2d -> SiLU      (ultralytics Conv)
  Bottleneck(c, shortcut) = Conv(3x3,1) -> Conv(3x3,1); +x if shortcut
  C2f(cin,cout,n,shortcut):
      y = Conv(1x1)(x)                     -> 2*cmid channels
      a, b = split(y, [cmid, cmid])        (channel split)
      outs = [a, b]
      cur = b
      for _ in range(n): cur = Bottleneck(cmid, shortcut)(cur); outs.append(cur)
      Conv(1x1)(concat(outs))              -> cout
  SPPF(cin,cout,k=5):
      x = Conv(1x1)(x)                     -> cmid
      p1 = maxpool(k,1,k//2)(x); p2 = maxpool(...)(p1); p3 = maxpool(...)(p2)
      Conv(1x1)(concat([x,p1,p2,p3]))      -> cout

  Backbone (indices = ultralytics model.{i}):
    0 Conv(3->w0, 3x3, s2)
    1 Conv(w0->w1, 3x3, s2)
    2 C2f(w1, w1, d0, True)
    3 Conv(w1->w2, 3x3, s2)
    4 C2f(w2, w2, d1, True)         -> P3 (saved)
    5 Conv(w2->w3, 3x3, s2)
    6 C2f(w3, w3, d1, True)         -> P4 (saved)
    7 Conv(w3->w4, 3x3, s2)
    8 C2f(w4, w4, d0, True)
    9 SPPF(w4, w4)                  -> P5 (saved)
  Neck (PAN top-down + bottom-up):
    10 Upsample(2x nearest)(9)
    11 Concat([10, P4])
    12 C2f(w4+w3 -> w3, d0, False)  -> N4 (saved)
    13 Upsample(2x)(12)
    14 Concat([13, P3])
    15 C2f(w3+w2 -> w2, d0, False)  -> head P3 input (saved)
    16 Conv(w2->w2, 3x3, s2)(15)
    17 Concat([16, 12])
    18 C2f(w2+w3 -> w3, d0, False)  -> head P4 input (saved)
    19 Conv(w3->w3, 3x3, s2)(18)
    20 Concat([19, 9])
    21 C2f(w3+w4 -> w4, d0, False)  -> head P5 input (saved)
  Detect head (22): decoupled, on inputs [15(w2), 18(w3), 21(w4)], 3 strides.
    per stride i, channels ci:
      box branch cv2[i]: Conv(ci->cb,3x3) -> Conv(cb->cb,3x3) -> Conv2d(cb->4*reg_max, +bias)
      cls branch cv3[i]: Conv(ci->cc,3x3) -> Conv(cc->cc,3x3) -> Conv2d(cc->nc, +bias)
    raw per-cell output (this fixture pins the RAW head, like DETR): for each
    cell concat[box_dist(4*reg_max), cls_logits(nc)]. DFL decode + NMS is a CPU
    post-process and is checked separately by the decode unit test.

  The CAI net flattens each stride to (H*W, 1, 4*reg_max+nc) and DeepConcats the
  three strides along the token axis, so the final output is
  (sum_i Hi*Wi, 1, 4*reg_max+nc). Cells are ordered stride-major, row-major
  within a stride (the (y*W+x) raster CAI uses).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/yolo_tiny_fixture.py
writes tests/fixtures/tiny_yolo{.safetensors,_config.json,_io.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

# Pico config: tiny widths/depths but the FULL YOLOv8 structure.
W0, W1, W2, W3, W4 = 4, 6, 8, 12, 16     # stage widths
D0, D1 = 1, 2                            # C2f repeats (n)
NC = 3                                   # classes
REG_MAX = 8                              # DFL bins per side (ultralytics uses 16)
IN_CH = 3
INPUT = 96                              # /8,/16,/32 -> 12,6,3 grids
# NOTE: CAI's conv clamps its kernel to the input spatial size
# (FFeatureSizeX := Min(K, prev.SizeX)), so the smallest feature map must stay
# >= 3 for the 3x3 convs (head + deep C2f) to keep a full 3x3 kernel. /32 = 3 at
# INPUT 96 is the minimum that avoids the clamp; real yolov8 runs 640 (/32=20).
CB = 8                                   # box-branch hidden width
CC = 8                                   # cls-branch hidden width
BN_EPS = 1e-3                            # ultralytics BN eps

rng = np.random.default_rng(20260615)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


# ---------------------------------------------------------------------------
# State dict (canonical ultralytics keys: "model.{i}.<sub>.{conv|bn}.<...>").
# ---------------------------------------------------------------------------
def add_conv_bn(sd, name, out_ch, in_ch, k):
    # ultralytics Conv: conv (no bias) + bn. name = the Conv module prefix.
    sd[name + '.conv.weight'] = randn(out_ch, in_ch, k, k, std=0.15)
    sd[name + '.bn.weight'] = randn(out_ch, std=0.3) + 1.0
    sd[name + '.bn.bias'] = randn(out_ch, std=0.1)
    sd[name + '.bn.running_mean'] = randn(out_ch, std=0.1)
    sd[name + '.bn.running_var'] = np.abs(randn(out_ch, std=0.2)) + 0.5


def add_conv2d(sd, name, out_ch, in_ch, k):
    # raw nn.Conv2d (head final layers): weight + bias.
    sd[name + '.weight'] = randn(out_ch, in_ch, k, k, std=0.15)
    sd[name + '.bias'] = randn(out_ch, std=0.1)


def add_bottleneck(sd, prefix, c, k=3):
    add_conv_bn(sd, prefix + '.cv1', c, c, k)
    add_conv_bn(sd, prefix + '.cv2', c, c, k)


def add_c2f(sd, idx, cin, cout, n):
    cmid = cout // 2
    add_conv_bn(sd, f'model.{idx}.cv1', 2 * cmid, cin, 1)
    add_conv_bn(sd, f'model.{idx}.cv2', cout, (2 + n) * cmid, 1)
    for j in range(n):
        add_bottleneck(sd, f'model.{idx}.m.{j}', cmid)


def add_sppf(sd, idx, cin, cout):
    cmid = cin // 2
    add_conv_bn(sd, f'model.{idx}.cv1', cmid, cin, 1)
    add_conv_bn(sd, f'model.{idx}.cv2', cout, 4 * cmid, 1)


def build_state_dict():
    sd = {}
    add_conv_bn(sd, 'model.0', W0, IN_CH, 3)
    add_conv_bn(sd, 'model.1', W1, W0, 3)
    add_c2f(sd, 2, W1, W1, D0)
    add_conv_bn(sd, 'model.3', W2, W1, 3)
    add_c2f(sd, 4, W2, W2, D1)
    add_conv_bn(sd, 'model.5', W3, W2, 3)
    add_c2f(sd, 6, W3, W3, D1)
    add_conv_bn(sd, 'model.7', W4, W3, 3)
    add_c2f(sd, 8, W4, W4, D0)
    add_sppf(sd, 9, W4, W4)
    # neck
    add_c2f(sd, 12, W4 + W3, W3, D0)
    add_c2f(sd, 15, W3 + W2, W2, D0)
    add_conv_bn(sd, 'model.16', W2, W2, 3)
    add_c2f(sd, 18, W2 + W3, W3, D0)
    add_conv_bn(sd, 'model.19', W3, W3, 3)
    add_c2f(sd, 21, W3 + W4, W4, D0)
    # head: 3 strides, channels [W2, W3, W4]
    head_ch = [W2, W3, W4]
    for i, ci in enumerate(head_ch):
        add_conv_bn(sd, f'model.22.cv2.{i}.0', CB, ci, 3)
        add_conv_bn(sd, f'model.22.cv2.{i}.1', CB, CB, 3)
        add_conv2d(sd, f'model.22.cv2.{i}.2', 4 * REG_MAX, CB, 1)
        add_conv_bn(sd, f'model.22.cv3.{i}.0', CC, ci, 3)
        add_conv_bn(sd, f'model.22.cv3.{i}.1', CC, CC, 3)
        add_conv2d(sd, f'model.22.cv3.{i}.2', NC, CC, 1)
    return sd


# ---------------------------------------------------------------------------
# numpy float64 oracle (volumes kept (C,H,W); conv weights [O,I,kh,kw]).
# ---------------------------------------------------------------------------
def conv2d(x, w, b, pad, stride):
    I, H, W = x.shape
    O, _, k, _ = w.shape
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (W + 2 * pad - k) // stride + 1
    xp = np.zeros((I, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    out = np.zeros((O, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            patch = xp[:, oy * stride:oy * stride + k, ox * stride:ox * stride + k]
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3], [0, 1, 2]))
            if b is not None:
                out[:, oy, ox] += b
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def conv_bn(x, sd, name, k, stride):
    w = sd[name + '.conv.weight']
    g = sd[name + '.bn.weight']
    bt = sd[name + '.bn.bias']
    rm = sd[name + '.bn.running_mean']
    rv = sd[name + '.bn.running_var']
    pad = k // 2
    y = conv2d(x, w, None, pad, stride)
    scale = g / np.sqrt(rv + BN_EPS)
    shift = bt - g * rm / np.sqrt(rv + BN_EPS)
    y = y * scale[:, None, None] + shift[:, None, None]
    return silu(y)


def conv2d_raw(x, sd, name, k):
    return conv2d(x, sd[name + '.weight'], sd[name + '.bias'], k // 2, 1)


def maxpool(x, k, stride, pad):
    # CAI's TNNetMaxPool zero-PADS the border before taking the window max
    # (CopyPadding fills 0), so the oracle must pad with 0 too (NOT -inf) to
    # stay byte-exact. SiLU outputs can be slightly negative, so this matters.
    I, H, W = x.shape
    xp = np.zeros((I, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (W + 2 * pad - k) // stride + 1
    out = np.zeros((I, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            out[:, oy, ox] = xp[:, oy * stride:oy * stride + k,
                                ox * stride:ox * stride + k].max(axis=(1, 2))
    return out


def bottleneck(x, sd, prefix, shortcut):
    y = conv_bn(x, sd, prefix + '.cv1', 3, 1)
    y = conv_bn(y, sd, prefix + '.cv2', 3, 1)
    return x + y if shortcut else y


def c2f(x, sd, idx, n, shortcut):
    y = conv_bn(x, sd, f'model.{idx}.cv1', 1, 1)
    cmid = y.shape[0] // 2
    a, b = y[:cmid], y[cmid:]
    outs = [a, b]
    cur = b
    for j in range(n):
        cur = bottleneck(cur, sd, f'model.{idx}.m.{j}', shortcut)
        outs.append(cur)
    cat = np.concatenate(outs, axis=0)
    return conv_bn(cat, sd, f'model.{idx}.cv2', 1, 1)


def sppf(x, sd, idx):
    y = conv_bn(x, sd, f'model.{idx}.cv1', 1, 1)
    p1 = maxpool(y, 5, 1, 2)
    p2 = maxpool(p1, 5, 1, 2)
    p3 = maxpool(p2, 5, 1, 2)
    cat = np.concatenate([y, p1, p2, p3], axis=0)
    return conv_bn(cat, sd, f'model.{idx}.cv2', 1, 1)


def upsample2x(x):
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def head_branch(x, sd, i):
    # box: cv2[i] -> 4*reg_max ; cls: cv3[i] -> nc
    bx = conv_bn(x, sd, f'model.22.cv2.{i}.0', 3, 1)
    bx = conv_bn(bx, sd, f'model.22.cv2.{i}.1', 3, 1)
    bx = conv2d_raw(bx, sd, f'model.22.cv2.{i}.2', 1)        # (4*reg_max,H,W)
    cl = conv_bn(x, sd, f'model.22.cv3.{i}.0', 3, 1)
    cl = conv_bn(cl, sd, f'model.22.cv3.{i}.1', 3, 1)
    cl = conv2d_raw(cl, sd, f'model.22.cv3.{i}.2', 1)        # (nc,H,W)
    return np.concatenate([bx, cl], axis=0)                  # (4*reg_max+nc,H,W)


def forward(x, sd):
    f0 = conv_bn(x, sd, 'model.0', 3, 2)
    f1 = conv_bn(f0, sd, 'model.1', 3, 2)
    f2 = c2f(f1, sd, 2, D0, True)
    f3 = conv_bn(f2, sd, 'model.3', 3, 2)
    P3 = c2f(f3, sd, 4, D1, True)        # /8
    f5 = conv_bn(P3, sd, 'model.5', 3, 2)
    P4 = c2f(f5, sd, 6, D1, True)        # /16
    f7 = conv_bn(P4, sd, 'model.7', 3, 2)
    f8 = c2f(f7, sd, 8, D0, True)
    P5 = sppf(f8, sd, 9)                 # /32
    # top-down
    u10 = upsample2x(P5)
    c11 = np.concatenate([u10, P4], axis=0)
    N4 = c2f(c11, sd, 12, D0, False)
    u13 = upsample2x(N4)
    c14 = np.concatenate([u13, P3], axis=0)
    H3 = c2f(c14, sd, 15, D0, False)     # head P3 input (/8)
    # bottom-up
    d16 = conv_bn(H3, sd, 'model.16', 3, 2)
    c17 = np.concatenate([d16, N4], axis=0)
    H4 = c2f(c17, sd, 18, D0, False)     # head P4 input (/16)
    d19 = conv_bn(H4, sd, 'model.19', 3, 2)
    c20 = np.concatenate([d19, P5], axis=0)
    H5 = c2f(c20, sd, 21, D0, False)     # head P5 input (/32)
    heads = [head_branch(H3, sd, 0),
             head_branch(H4, sd, 1),
             head_branch(H5, sd, 2)]
    # flatten stride-major, row-major (y*W+x): (sum Hi*Wi, 4*reg_max+nc)
    rows = []
    for h in heads:
        C, Hh, Ww = h.shape
        for yy in range(Hh):
            for xx in range(Ww):
                rows.append(h[:, yy, xx])
    return np.array(rows), [(h.shape[1], h.shape[2]) for h in heads]


def main():
    sd = build_state_dict()
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    # Pinned input image: deterministic dyadic values (exact in f32 + JSON).
    x = np.zeros((IN_CH, INPUT, INPUT), dtype=np.float64)
    for c in range(IN_CH):
        for y in range(INPUT):
            for px in range(INPUT):
                x[c, y, px] = (((c * 1024 + y * 32 + px) * 7) % 17 - 8) / 16.0

    out, grids = forward(x, sd)
    n_cells = out.shape[0]
    out_dim = out.shape[1]
    print(f'input {x.shape} -> head ({n_cells}, {out_dim}); grids {grids}')
    print(f'out stats: min {out.min():.4f} max {out.max():.4f} '
          f'mean {out.mean():.4f}')

    save_file(sd_f32, 'tests/fixtures/tiny_yolo.safetensors')
    config = {
        'model_type': 'yolov8',
        'widths': [W0, W1, W2, W3, W4],
        'depths': [D0, D1],
        'num_classes': NC,
        'reg_max': REG_MAX,
        'box_hidden': CB,
        'cls_hidden': CC,
        'num_channels': IN_CH,
        'image_size': INPUT,
        'bn_eps': BN_EPS,
        'strides': [8, 16, 32],
    }
    with open('tests/fixtures/tiny_yolo_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_yolo_io.json', 'w') as f:
        json.dump({
            # FLAT lists (the Pascal test indexes them flat):
            # input flat (c, y, x); output flat (cell, channel).
            'input': x.reshape(-1).tolist(),      # 3*INPUT*INPUT
            'output': out.reshape(-1).tolist(),   # n_cells*(4*reg_max+nc)
            'num_cells': n_cells,
            'out_dim': out_dim,
            'grids': grids,
        }, f)
    print(f'wrote tiny_yolo.safetensors ({len(sd_f32)} tensors) + config + io')

    # ---- fixture self-checks: each major piece must MATTER. ----
    base = out.copy()

    def effect(key):
        alt = dict(sd)
        alt[key] = np.zeros_like(sd[key])
        d = np.abs(forward(x, alt)[0] - base).max()
        return d

    for key in ['model.2.m.0.cv1.conv.weight',     # C2f bottleneck
                'model.9.cv2.conv.weight',          # SPPF fuse
                'model.12.cv1.conv.weight',         # top-down C2f
                'model.16.conv.weight',             # bottom-up downsample
                'model.22.cv2.0.2.bias',            # box head bias
                'model.22.cv3.2.2.weight']:         # cls head P5
        d = effect(key)
        # 1e-5 floor: well above f32 round-trip noise; some early-stage pico
        # paths have a genuinely small (but nonzero) influence on the head.
        assert d > 1e-5, f'{key} had no effect ({d})'
        print(f'{key:32s} effect: max|diff| = {d:.4f}')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
