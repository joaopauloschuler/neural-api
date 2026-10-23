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

Segment mode (--seg): the ultralytics Segment(Detect) head used by YOLOv8-seg
and FastSAM, on the same pico backbone with nc=1 (FastSAM's single class):
  proto (on head input 15): Conv(w2->npr,3x3) -> ConvTranspose2d(npr,npr,2,s2,
      +bias) -> Conv(npr->npr,3x3) -> Conv(npr->nm,1x1)    (stride-4 map)
  cv4[i] per stride: Conv(ci->c4,3x3) -> Conv(c4->c4,3x3) -> Conv2d(c4->nm,+bias)
  c4 = max(w2 // 4, nm). Per cell the CAI head is [box, cls, nm mask coefs].
  The oracle then runs DFL decode + NMS, FastSAM's border snap / full-box
  replacement, and ultralytics 8.0 process_mask (sigmoid(coef @ proto), crop
  at proto resolution, bilinear upsample, > 0.5) in torch float64. It also
  writes the retina masks (retina_masks=True, FastSAM's CLI default): 8.0
  process_mask_native with shape = the net input, i.e. upsample first, then
  crop with the pixel boxes (no letterbox unpad: the input is the image).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/yolo_tiny_fixture.py
writes tests/fixtures/tiny_yolo{.safetensors,_config.json,_io.json}.
  /home/bpsa/x/bin/python tools/yolo_tiny_fixture.py --seg
writes tests/fixtures/tiny_yolo_seg{.safetensors,_config.json,_io.safetensors}
(needs torch for the seg oracle; the detect mode needs numpy + safetensors only).
"""
import sys
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


def build_state_dict(nc=NC):
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
        add_conv2d(sd, f'model.22.cv3.{i}.2', nc, CC, 1)
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
    # torch nn.MaxPool2d pads with -inf. SiLU outputs can be negative, so a
    # zero pad would change border cells; BuildYolo shifts the SPPF input to
    # keep CAI's zero-padded maxpool exact.
    I, H, W = x.shape
    xp = np.full((I, H + 2 * pad, W + 2 * pad), -np.inf, dtype=np.float64)
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


def neck(x, sd):
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
    return H3, H4, H5


def flatten_heads(heads):
    # flatten stride-major, row-major (y*W+x): (sum Hi*Wi, per-cell channels)
    rows = []
    for h in heads:
        C, Hh, Ww = h.shape
        for yy in range(Hh):
            for xx in range(Ww):
                rows.append(h[:, yy, xx])
    return np.array(rows), [(h.shape[1], h.shape[2]) for h in heads]


def forward(x, sd):
    H3, H4, H5 = neck(x, sd)
    return flatten_heads([head_branch(H3, sd, 0),
                          head_branch(H4, sd, 1),
                          head_branch(H5, sd, 2)])


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


# ---------------------------------------------------------------------------
# Segment mode (YOLOv8-seg / FastSAM).
# ---------------------------------------------------------------------------
NC_SEG = 1                               # FastSAM: one class ("object")
NM = 4                                   # mask coefficients (ultralytics 32)
NPR = 8                                  # proto hidden width (ultralytics 256*w)
C4 = max(W2 // 4, NM)                    # cv4 hidden width
SEG_IOU = 0.5                            # NMS IoU threshold pinned in the io
SEG_KEEP = 24                            # candidates above the score threshold


def add_seg_head(sd):
    add_conv_bn(sd, 'model.22.proto.cv1', NPR, W2, 3)
    sd['model.22.proto.upsample.weight'] = randn(NPR, NPR, 2, 2, std=0.3)
    sd['model.22.proto.upsample.bias'] = randn(NPR, std=0.1)
    add_conv_bn(sd, 'model.22.proto.cv2', NPR, NPR, 3)
    add_conv_bn(sd, 'model.22.proto.cv3', NM, NPR, 1)
    for i, ci in enumerate([W2, W3, W4]):
        add_conv_bn(sd, f'model.22.cv4.{i}.0', C4, ci, 3)
        add_conv_bn(sd, f'model.22.cv4.{i}.1', C4, C4, 3)
        sd[f'model.22.cv4.{i}.2.weight'] = randn(NM, C4, 1, 1, std=3.0)
        sd[f'model.22.cv4.{i}.2.bias'] = randn(NM, std=1.0)


def proto_hidden(H3, sd):
    import torch
    p = conv_bn(H3, sd, 'model.22.proto.cv1', 3, 1)
    p = torch.nn.functional.conv_transpose2d(
        torch.from_numpy(p)[None],
        torch.from_numpy(sd['model.22.proto.upsample.weight']),
        torch.from_numpy(sd['model.22.proto.upsample.bias']), stride=2)[0].numpy()
    return conv_bn(p, sd, 'model.22.proto.cv2', 3, 1)


def calibrate_proto_bn(x, sd):
    # Set proto.cv3's BN stats to the real pre-BN statistics so the proto map
    # varies across space (random stats leave it nearly constant).
    H3 = neck(x, sd)[0]
    pre = conv2d(proto_hidden(H3, sd), sd['model.22.proto.cv3.conv.weight'],
                 None, 0, 1)
    sd['model.22.proto.cv3.bn.running_mean'] = pre.mean(axis=(1, 2))
    sd['model.22.proto.cv3.bn.running_var'] = pre.var(axis=(1, 2))
    sd['model.22.proto.cv3.bn.weight'] = randn(NM, std=0.5) + 3.0
    sd['model.22.proto.cv3.bn.bias'] = randn(NM, std=0.5)


def seg_forward(x, sd):
    H3, H4, H5 = neck(x, sd)
    heads = []
    for i, Hi in enumerate([H3, H4, H5]):
        mc = conv_bn(Hi, sd, f'model.22.cv4.{i}.0', 3, 1)
        mc = conv_bn(mc, sd, f'model.22.cv4.{i}.1', 3, 1)
        mc = conv2d_raw(mc, sd, f'model.22.cv4.{i}.2', 1)
        heads.append(np.concatenate([head_branch(Hi, sd, i), mc], axis=0))
    proto = conv_bn(proto_hidden(H3, sd), sd, 'model.22.proto.cv3', 1, 1)
    out, grids = flatten_heads(heads)
    return out, grids, proto


def dfl_candidates(out, grids):
    # ultralytics Detect inference decode: DFL expectation -> xyxy pixels.
    rows = []
    cell = 0
    for (gh, gw), stride in zip(grids, [8, 16, 32]):
        for gy in range(gh):
            for gx in range(gw):
                v = out[cell]
                d = []
                for side in range(4):
                    b = v[side * REG_MAX:(side + 1) * REG_MAX]
                    e = np.exp(b - b.max())
                    d.append((e * np.arange(REG_MAX)).sum() / e.sum())
                cls = v[4 * REG_MAX:4 * REG_MAX + NC_SEG]
                score = 1.0 / (1.0 + np.exp(-cls.max()))
                cx, cy = gx + 0.5, gy + 0.5
                rows.append([(cx - d[0]) * stride, (cy - d[1]) * stride,
                             (cx + d[2]) * stride, (cy + d[3]) * stride,
                             score, int(cls.argmax()), cell])
                cell += 1
    return rows


def box_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_nms(cands, iou_thr):
    order = sorted(cands, key=lambda r: -r[4])
    keep, margin = [], 1.0
    for r in order:
        ok = True
        for k in keep:
            if k[5] == r[5]:
                iou = box_iou(k, r)
                margin = min(margin, abs(iou - iou_thr))
                if iou > iou_thr:
                    ok = False
                    break
        if ok:
            keep.append(list(r))
    return keep, margin


def fastsam_adjust(dets, size, border=20.0, full_iou=0.9):
    # FastSAM postprocess: bbox_iou snaps EVERY box to the image border in
    # place, then boxes with IoU > 0.9 vs the full image become the full box.
    for d in dets:
        if d[0] < border: d[0] = 0.0
        if d[1] < border: d[1] = 0.0
        if d[2] > size - border: d[2] = float(size)
        if d[3] > size - border: d[3] = float(size)
    full = [0.0, 0.0, float(size), float(size)]
    for d in dets:
        if box_iou(full, d) > full_iou:
            d[0:4] = full
            d[5] = 0
    return dets


def crop_mask(m, boxes):
    # ultralytics ops.crop_mask: keep x1 <= x < x2, y1 <= y < y2 per mask.
    import torch
    n, h, w = m.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, dtype=torch.float64)[None, None, :]
    cc = torch.arange(h, dtype=torch.float64)[None, :, None]
    return m * ((r >= x1) * (r < x2) * (cc >= y1) * (cc < y2))


def mask_logits_sigmoid(proto, coefs):
    import torch
    protos = torch.from_numpy(proto)
    c, mh, mw = protos.shape
    return (torch.from_numpy(coefs) @ protos.view(c, -1)).sigmoid().view(-1, mh, mw)


def process_mask(proto, coefs, boxes, size):
    # ultralytics 8.0 ops.process_mask(upsample=True), the predictor default path.
    import torch
    import torch.nn.functional as F
    m = mask_logits_sigmoid(proto, coefs)
    mh, mw = m.shape[1:]
    db = torch.from_numpy(np.array(boxes, dtype=np.float64)).clone()
    db[:, 0] *= mw / size; db[:, 2] *= mw / size
    db[:, 1] *= mh / size; db[:, 3] *= mh / size
    m = crop_mask(m, db)
    m = F.interpolate(m[None], (size, size), mode='bilinear',
                      align_corners=False)[0]
    return m.numpy()


def process_mask_native(proto, coefs, boxes, size):
    # ultralytics 8.0 ops.process_mask_native(protos, masks_in, bboxes, shape)
    # with shape = (size, size): gain = mh / size, so the letterbox unpad
    # slice is the whole map; bilinear upsample, then crop in pixels.
    import torch
    import torch.nn.functional as F
    m = mask_logits_sigmoid(proto, coefs)
    mh, mw = m.shape[1:]
    gain = min(mh / size, mw / size)
    pad = (mw - size * gain) / 2, (mh - size * gain) / 2
    top, left = int(pad[1]), int(pad[0])
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    m = m[:, top:bottom, left:right]
    m = F.interpolate(m[None], (size, size), mode='bilinear',
                      align_corners=False)[0]
    m = crop_mask(m, torch.from_numpy(np.array(boxes, dtype=np.float64)))
    return m.numpy()


def main_seg():
    global rng
    rng = np.random.default_rng(20261023)
    sd = build_state_dict(NC_SEG)
    add_seg_head(sd)
    x = np.zeros((IN_CH, INPUT, INPUT), dtype=np.float64)
    for c in range(IN_CH):
        for y in range(INPUT):
            for px in range(INPUT):
                x[c, y, px] = (((c * 1024 + y * 32 + px) * 7) % 17 - 8) / 16.0
    calibrate_proto_bn(x, sd)
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}
    out, grids, proto = seg_forward(x, sd)
    cands = dfl_candidates(out, grids)
    scores = sorted([r[4] for r in cands], reverse=True)
    # cut at the widest score gap among ranks SEG_KEEP..2*SEG_KEEP
    k = max(range(SEG_KEEP, 2 * SEG_KEEP), key=lambda j: scores[j - 1] - scores[j])
    conf = 0.5 * (scores[k - 1] + scores[k])
    score_margin = conf - scores[k]
    dets, iou_margin = greedy_nms([r for r in cands if r[4] > conf], SEG_IOU)
    dets = fastsam_adjust(dets, INPUT)
    coef_ofs = 4 * REG_MAX + NC_SEG
    coefs = np.array([out[d[6], coef_ofs:coef_ofs + NM] for d in dets])
    soft = process_mask(proto, coefs, [d[0:4] for d in dets], INPUT)
    retina = process_mask_native(proto, coefs, [d[0:4] for d in dets], INPUT)
    masks = (soft > 0.5).astype(np.float32)
    # pixels this close to 0.5 may binarize differently in float32; the
    # Pascal test compares the soft masks and skips them in the 0/1 check.
    near = int((np.abs(soft - 0.5) < 1e-4).sum())
    retina_near = int((np.abs(retina - 0.5) < 1e-4).sum())
    retina_differs = int(((retina > 0.5) != (soft > 0.5)).sum())
    print(f'head {out.shape}; proto {proto.shape}; conf {conf:.6f} '
          f'(margin {score_margin:.2e}); NMS IoU margin {iou_margin:.2e}')
    print(f'{len(dets)} dets; mask pixels on: {masks.sum(axis=(1, 2))}; '
          f'pixels within 1e-4 of 0.5: {near}')
    print(f'retina masks: pixels within 1e-4 of 0.5: {retina_near}; '
          f'binary pixels that differ from process_mask: {retina_differs}')
    assert score_margin > 1e-4 and iou_margin > 1e-3
    assert retina_differs > 0

    save_file(sd_f32, 'tests/fixtures/tiny_yolo_seg.safetensors')
    config = {
        'model_type': 'yolov8',
        'task': 'segment',
        'widths': [W0, W1, W2, W3, W4],
        'depths': [D0, D1],
        'num_classes': NC_SEG,
        'reg_max': REG_MAX,
        'box_hidden': CB,
        'cls_hidden': CC,
        'num_masks': NM,
        'proto_hidden': NPR,
        'num_channels': IN_CH,
        'image_size': INPUT,
        'bn_eps': BN_EPS,
        'strides': [8, 16, 32],
    }
    with open('tests/fixtures/tiny_yolo_seg_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    f32 = lambda a: np.ascontiguousarray(np.array(a, dtype=np.float32))
    save_file({
        'input': f32(x),                               # (3, H, W)
        'head': f32(out),                              # (cells, 4*rm+nc+nm)
        'proto': f32(proto),                           # (nm, H/4, W/4)
        'conf': f32([conf]),
        'iou': f32([SEG_IOU]),
        'det_boxes': f32([d[0:4] for d in dets]),      # FastSAM-adjusted xyxy
        'det_scores': f32([d[4] for d in dets]),
        'det_cells': f32([d[6] for d in dets]),
        'soft_masks': f32(soft),                       # (N, H, W); mask = > 0.5
        'retina_soft_masks': f32(retina),              # (N, H, W); mask = > 0.5
    }, 'tests/fixtures/tiny_yolo_seg_io.safetensors')
    print('wrote tiny_yolo_seg.safetensors + config + io')


if __name__ == '__main__':
    if '--seg' in sys.argv[1:]:
        main_seg()
    else:
        main()
