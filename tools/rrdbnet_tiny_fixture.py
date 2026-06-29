#!/usr/bin/env python3
"""Generate a tiny RANDOM Real-ESRGAN / ESRGAN RRDBNet (scale x4) parity fixture
for tests/TestNeuralPretrained.pas.

The realesrgan / basicsr packages are NOT installed, so the reference forward is
a self-contained numpy float64 oracle that mirrors the CAI importer's forward
path EXACTLY (same idiom as tools/vae_decoder_tiny_fixture.py). The weights use
the canonical xinntao RRDBNet state_dict key scheme so the importer is exercised
on the real key layout.

RRDBNet (scale x4) architecture:
  x (3, H, W)
    conv_first   (3x3 pad1, 3 -> nf, +bias)            -> feat
  body: N RRDB blocks; each RRDB output = feat + 0.2 * rrdb(feat)
    each RRDB = 3 ResidualDenseBlocks (rdb1, rdb2, rdb3) chained:
      rdb_out = rdb_in + 0.2 * rdb(rdb_in)
    each RDB = 5 conv (3x3) with DENSE skips along channels:
      x1 = lrelu(conv1(cat[x]))                in nf      -> gc
      x2 = lrelu(conv2(cat[x,x1]))             in nf+gc   -> gc
      x3 = lrelu(conv3(cat[x,x1,x2]))          in nf+2gc  -> gc
      x4 = lrelu(conv4(cat[x,x1,x2,x3]))       in nf+3gc  -> gc
      x5 =       conv5(cat[x,x1,x2,x3,x4])     in nf+4gc  -> nf
      rdb_out = x + 0.2 * x5
    lrelu = LeakyReLU(negative_slope=0.2)
  conv_body  (3x3 pad1, nf -> nf, +bias); feat = feat + conv_body(body(feat))
  UPSAMPLE x4 (nearest interpolate x2 + conv + lrelu, twice):
    feat = lrelu(conv_up1(nearest2x(feat)))
    feat = lrelu(conv_up2(nearest2x(feat)))
  conv_hr    (3x3 pad1, nf -> nf, +bias); feat = lrelu(conv_hr(feat))
  conv_last  (3x3 pad1, nf -> 3, +bias)  -> out (3, 4H, 4W)

Nearest upsample replicates each pixel into a 2x2 block (F.interpolate
mode='nearest' == CAI TNNetDeMaxPool(2) spacing 0).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/rrdbnet_tiny_fixture.py
writes tests/fixtures/tiny_rrdbnet{.safetensors,_config.json,_io.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

# Pico config: tiny widths, 1 RRDB block.
NF = 8            # num_feat
GC = 4            # num_grow_ch
NUM_BLOCK = 1     # number of RRDB blocks in body
IN_CH = 3
OUT_CH = 3
INPUT = 6         # input image grid (scale x4 -> 24x24, scale x2 -> 12x12)
LRELU_SLOPE = 0.2

rng = np.random.default_rng(20260614)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def conv_w(out_ch, in_ch, k, std=0.1):
    return randn(out_ch, in_ch, k, k, std=std)   # [O,I,kh,kw]


# ---------------------------------------------------------------------------
# State dict (canonical xinntao RRDBNet keys).
# ---------------------------------------------------------------------------
def add_conv(sd, name, out_ch, in_ch, k=3, std=0.1):
    sd[name + '.weight'] = conv_w(out_ch, in_ch, k, std=std)
    sd[name + '.bias'] = randn(out_ch, std=0.05)


def add_rdb(sd, prefix):
    add_conv(sd, prefix + 'conv1', GC, NF)
    add_conv(sd, prefix + 'conv2', GC, NF + GC)
    add_conv(sd, prefix + 'conv3', GC, NF + 2 * GC)
    add_conv(sd, prefix + 'conv4', GC, NF + 3 * GC)
    add_conv(sd, prefix + 'conv5', NF, NF + 4 * GC)


def build_state_dict(scale):
    sd = {}
    add_conv(sd, 'conv_first', NF, IN_CH)
    for i in range(NUM_BLOCK):
        for r in (1, 2, 3):
            add_rdb(sd, f'body.{i}.rdb{r}.')
    add_conv(sd, 'conv_body', NF, NF)
    add_conv(sd, 'conv_up1', NF, NF)
    if scale == 4:
        add_conv(sd, 'conv_up2', NF, NF)
    add_conv(sd, 'conv_hr', NF, NF)
    add_conv(sd, 'conv_last', OUT_CH, NF)
    return sd


# ---------------------------------------------------------------------------
# numpy float64 oracle (volumes kept (C,H,W); conv weights [O,I,kh,kw]).
# ---------------------------------------------------------------------------
def conv2d(x, w, b, pad):
    I, H, W = x.shape
    O, _, k, _ = w.shape
    Ho = H - k + 2 * pad + 1
    Wo = W - k + 2 * pad + 1
    xp = np.zeros((I, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    out = np.zeros((O, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            patch = xp[:, oy:oy + k, ox:ox + k]
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3],
                                                          [0, 1, 2])) + b
    return out


def lrelu(x):
    return np.where(x > 0, x, x * LRELU_SLOPE)


def upsample_nearest(x):
    # Replicate each pixel into a 2x2 block (depth preserved).
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def rdb(x, sd, prefix):
    # Dense block: concat along channel axis (axis 0 in (C,H,W)).
    x1 = lrelu(conv2d(x, sd[prefix + 'conv1.weight'],
                      sd[prefix + 'conv1.bias'], 1))
    x2 = lrelu(conv2d(np.concatenate([x, x1], axis=0),
                      sd[prefix + 'conv2.weight'], sd[prefix + 'conv2.bias'], 1))
    x3 = lrelu(conv2d(np.concatenate([x, x1, x2], axis=0),
                      sd[prefix + 'conv3.weight'], sd[prefix + 'conv3.bias'], 1))
    x4 = lrelu(conv2d(np.concatenate([x, x1, x2, x3], axis=0),
                      sd[prefix + 'conv4.weight'], sd[prefix + 'conv4.bias'], 1))
    x5 = conv2d(np.concatenate([x, x1, x2, x3, x4], axis=0),
                sd[prefix + 'conv5.weight'], sd[prefix + 'conv5.bias'], 1)
    return x + 0.2 * x5


def rrdb(x, sd, prefix):
    out = rdb(x, sd, prefix + 'rdb1.')
    out = rdb(out, sd, prefix + 'rdb2.')
    out = rdb(out, sd, prefix + 'rdb3.')
    return x + 0.2 * out


def forward(x, sd, scale):
    feat = conv2d(x, sd['conv_first.weight'], sd['conv_first.bias'], 1)
    body = feat
    for i in range(NUM_BLOCK):
        body = rrdb(body, sd, f'body.{i}.')
    body = conv2d(body, sd['conv_body.weight'], sd['conv_body.bias'], 1)
    feat = feat + body
    # Upsample (nearest-2x + conv + lrelu): two stages for x4, one for x2.
    feat = lrelu(conv2d(upsample_nearest(feat),
                        sd['conv_up1.weight'], sd['conv_up1.bias'], 1))
    if scale == 4:
        feat = lrelu(conv2d(upsample_nearest(feat),
                            sd['conv_up2.weight'], sd['conv_up2.bias'], 1))
    feat = lrelu(conv2d(feat, sd['conv_hr.weight'], sd['conv_hr.bias'], 1))
    out = conv2d(feat, sd['conv_last.weight'], sd['conv_last.bias'], 1)
    return out


def pinned_input():
    # Pinned input image: deterministic dyadic values (exact in f32 + JSON).
    x = np.zeros((IN_CH, INPUT, INPUT), dtype=np.float64)
    for c in range(IN_CH):
        for y in range(INPUT):
            for px in range(INPUT):
                x[c, y, px] = (((c * 36 + y * 6 + px) * 5) % 13 - 6) / 8.0
    return x


def emit(scale, stem):
    sd = build_state_dict(scale)
    # Round-trip every weight through float32 (CAI loads float32).
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    x = pinned_input()
    img = forward(x, sd, scale)
    print(f'[scale x{scale}] input {x.shape} -> image {img.shape}; '
          f'min {img.min():.4f} max {img.max():.4f} mean {img.mean():.4f}')

    save_file(sd_f32, f'tests/fixtures/{stem}.safetensors')
    config = {
        'model_type': 'rrdbnet',
        'num_in_ch': IN_CH,
        'num_out_ch': OUT_CH,
        'num_feat': NF,
        'num_block': NUM_BLOCK,
        'num_grow_ch': GC,
        'scale': scale,
        'input_size': INPUT,
    }
    with open(f'tests/fixtures/{stem}_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open(f'tests/fixtures/{stem}_io.json', 'w') as f:
        json.dump({
            'input': x.tolist(),                  # [3][INPUT][INPUT]
            'image': img.tolist(),                # [3][IMG][IMG]
            'image_size': img.shape[1],
        }, f)
    print(f'  wrote {stem}.safetensors ({len(sd_f32)} tensors) + config + io')
    return sd, sd_f32, x, img


def write_pth_params_ema(sd_f32, path):
    # Save the SAME float32 weights as a Real-ESRGAN-style .pth: the actual
    # state_dict is nested under a 'params_ema' top-level key. torch.save uses
    # a STORED (uncompressed) zip, which TNNetTorchBinReader requires.
    import torch
    params = {k: torch.from_numpy(np.ascontiguousarray(v))
              for k, v in sd_f32.items()}
    torch.save({'params_ema': params}, path)
    print(f'  wrote {path} (params_ema-wrapped .pth)')


def main():
    # ---- scale x4 (canonical: two upsample stages) ----
    sd4, sd4_f32, x, img4 = emit(4, 'tiny_rrdbnet')
    write_pth_params_ema(sd4_f32, 'tests/fixtures/tiny_rrdbnet.pth')

    # ---- scale x2 (one upsample stage; conv_up2 absent) ----
    emit(2, 'tiny_rrdbnet_x2')

    # ---- fixture self-checks (scale x4): each major piece must MATTER. ----
    base = img4.copy()

    alt = dict(sd4)
    alt['body.0.rdb2.conv3.weight'] = np.zeros_like(sd4['body.0.rdb2.conv3.weight'])
    d = np.abs(forward(x, alt, 4) - base).max()
    assert d > 1e-4, f'dense conv3 had no effect ({d})'
    print(f'dense skip conv3 effect: max|diff| = {d:.4f}')

    alt = dict(sd4)
    alt['conv_body.bias'] = np.zeros_like(sd4['conv_body.bias'])
    d = np.abs(forward(x, alt, 4) - base).max()
    assert d > 1e-4, f'conv_body bias had no effect ({d})'
    print(f'conv_body bias effect: max|diff| = {d:.4f}')

    alt = dict(sd4)
    alt['conv_up2.weight'] = np.zeros_like(sd4['conv_up2.weight'])
    d = np.abs(forward(x, alt, 4) - base).max()
    assert d > 1e-4, f'conv_up2 had no effect ({d})'
    print(f'conv_up2 effect: max|diff| = {d:.4f}')

    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
