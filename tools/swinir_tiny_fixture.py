#!/usr/bin/env python3
"""Generate a tiny RANDOM SwinIR image-restoration parity fixture for
tests/TestNeuralPretrained.pas.

SwinIR (Liang et al. 2021, "SwinIR: Image Restoration Using Swin Transformer",
https://arxiv.org/abs/2108.10257) is a transformer restoration net: a shallow
conv stem, a stack of Residual Swin Transformer Blocks (RSTB), and a
pixel-shuffle upsample tail (classical super-resolution variant). There is NO
official tiny checkpoint, so the reference forward is a self-contained numpy
float64 oracle that mirrors the CAI importer's forward path EXACTLY (same idiom
as tools/nafnet_tiny_fixture.py and tools/rrdbnet_tiny_fixture.py). Weights use
the official SwinIR repo state_dict key scheme so the importer is exercised on a
realistic key layout.

Architecture (this pico, classical SR, upscale=2):
  conv_first       (3x3, 3->E, pad1, +bias)                       -> feat map F0
  layers[L] RSTB:   x = block_in (token seq over the HxW grid)
      residual_group.blocks[B] Swin layer (W-MSA / SW-MSA + MLP)
      residual_group.conv (3x3, E->E, pad1, +bias) over the map
      x = x + block_in                                            (RSTB residual)
  norm              (token LayerNorm over E)
  conv_after_body  (3x3, E->E, pad1, +bias)  ; + F0   (deep-feature residual)
  conv_before_upsample (3x3, E->Eup, pad1, +bias) -> LeakyReLU(0.2)
  upsample.0       (3x3, Eup->4*Eup, pad1, +bias) -> PixelShuffle(2)  (x2)
  conv_last        (3x3, Eup->3, pad1, +bias)

One Swin layer (HF/SwinIR semantics, token sequence over the HxW grid):
  shortcut = x
  x = norm1(x)  (token LayerNorm over E)
  window-partition (+cyclic shift on odd blocks) -> per-window per-head W-MSA
    with relative_position_bias + (shifted) attention mask
  x = shortcut + attn
  x = x + mlp(norm2(x))   (fc1 -> GELU -> fc2)

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/swinir_tiny_fixture.py
writes tests/fixtures/tiny_swinir{.safetensors,_config.json,_io.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

# Pico config: deliberately exercises the full shifted-window path.
EMBED = 6          # embed dim (divisible by NUM_HEADS)
NUM_HEADS = 2
DEPTH = 2          # Swin layers per RSTB (>=2 -> one W-MSA + one SW-MSA)
NUM_RSTB = 1       # RSTB blocks
WINDOW = 2         # window size (grid 4 -> 2x2 windows, shift = 1)
MLP_RATIO = 2.0
IN_CH = 3
IMG = 4            # input grid H=W (multiple of WINDOW)
UPSCALE = 2
NUM_FEAT = 4       # conv_before_upsample channels (Eup)
LN_EPS = 1e-5

rng = np.random.default_rng(20260615)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


# ---------------------------------------------------------------------------
# State dict (official SwinIR repo keys).
# ---------------------------------------------------------------------------
def add_conv(sd, name, out_ch, in_ch, k, std=0.15):
    sd[name + '.weight'] = randn(out_ch, in_ch, k, k, std=std)
    sd[name + '.bias'] = randn(out_ch, std=0.05)


def add_linear(sd, name, out_f, in_f, bias=True, std=0.15):
    sd[name + '.weight'] = randn(out_f, in_f, std=std)
    if bias:
        sd[name + '.bias'] = randn(out_f, std=0.05)


def add_swin_layer(sd, prefix):
    # layernorms
    sd[prefix + 'norm1.weight'] = randn(EMBED, std=0.2) + 1.0
    sd[prefix + 'norm1.bias'] = randn(EMBED, std=0.1)
    sd[prefix + 'norm2.weight'] = randn(EMBED, std=0.2) + 1.0
    sd[prefix + 'norm2.bias'] = randn(EMBED, std=0.1)
    # attention q/k/v packed (SwinIR uses a single qkv linear) + proj
    add_linear(sd, prefix + 'attn.qkv', 3 * EMBED, EMBED)
    add_linear(sd, prefix + 'attn.proj', EMBED, EMBED)
    # relative position bias table [(2*ws-1)^2, num_heads]
    sd[prefix + 'attn.relative_position_bias_table'] = \
        randn((2 * WINDOW - 1) ** 2, NUM_HEADS, std=0.1)
    # mlp
    hidden = int(EMBED * MLP_RATIO)
    add_linear(sd, prefix + 'mlp.fc1', hidden, EMBED)
    add_linear(sd, prefix + 'mlp.fc2', EMBED, hidden)


def build_state_dict():
    sd = {}
    add_conv(sd, 'conv_first', EMBED, IN_CH, 3)
    for li in range(NUM_RSTB):
        for bi in range(DEPTH):
            add_swin_layer(sd, f'layers.{li}.residual_group.blocks.{bi}.')
        add_conv(sd, f'layers.{li}.conv', EMBED, EMBED, 3)
    sd['norm.weight'] = randn(EMBED, std=0.2) + 1.0
    sd['norm.bias'] = randn(EMBED, std=0.1)
    add_conv(sd, 'conv_after_body', EMBED, EMBED, 3)
    add_conv(sd, 'conv_before_upsample.0', NUM_FEAT, EMBED, 3)
    add_conv(sd, 'upsample.0', 4 * NUM_FEAT, NUM_FEAT, 3)
    add_conv(sd, 'conv_last', IN_CH, NUM_FEAT, 3)
    return sd


# ---------------------------------------------------------------------------
# numpy float64 oracle.  Image volumes are (C, H, W); conv weights [O,I,kh,kw].
# Token sequences are (N=H*W, E) in row-major (y*W + x) order.
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


def gelu(x):
    # exact erf GELU (matches TNNetGELU).
    from math import sqrt
    import scipy.special as sp  # noqa
    return 0.5 * x * (1.0 + sp.erf(x / sqrt(2.0)))


def gelu_np(x):
    # erf without scipy.
    from math import erf, sqrt
    f = np.vectorize(lambda v: 0.5 * v * (1.0 + erf(v / sqrt(2.0))))
    return f(x)


def layernorm_tokens(x, g, beta, eps=LN_EPS):
    # x (N, E), normalize over the E axis per token.
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)        # population variance
    xn = (x - mean) / np.sqrt(var + eps)
    return xn * g[None, :] + beta[None, :]


def map_to_tokens(x):
    # (E, H, W) -> (H*W, E) row-major (y*W + x).
    E, H, Wd = x.shape
    return x.transpose(1, 2, 0).reshape(H * Wd, E)


def tokens_to_map(t, H, Wd):
    # (H*W, E) -> (E, H, W).
    E = t.shape[1]
    return t.reshape(H, Wd, E).transpose(2, 0, 1)


def rel_pos_index(ws):
    # within-window relative position index, (ws2, ws2).
    coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws), indexing='ij'),
                      axis=0).reshape(2, -1)          # (2, ws2)
    rel = coords[:, :, None] - coords[:, None, :]      # (2, ws2, ws2)
    rel = rel + (ws - 1)
    idx = rel[0] * (2 * ws - 1) + rel[1]
    return idx                                         # (ws2, ws2)


def region_ids(H, Wd, ws, shift):
    # HF get_attn_mask region image: 9 regions from the 3 height/width slices.
    img = np.zeros((H, Wd), dtype=np.int64)
    hs = [slice(0, -ws), slice(-ws, -shift), slice(-shift, None)]
    cnt = 0
    for hsl in hs:
        for wsl in hs:
            img[hsl, wsl] = cnt
            cnt += 1
    return img


def swin_layer(x, sd, prefix, H, Wd, shifted):
    # x: (N=H*W, E) token sequence.
    E = x.shape[1]
    ws = WINDOW
    ws2 = ws * ws
    heads = NUM_HEADS
    hd = E // heads
    shift = ws // 2 if shifted else 0

    shortcut = x
    xn = layernorm_tokens(x, sd[prefix + 'norm1.weight'], sd[prefix + 'norm1.bias'])
    grid = xn.reshape(H, Wd, E)
    if shift > 0:
        grid = np.roll(grid, shift=(-shift, -shift), axis=(0, 1))

    # window partition
    wper = H // ws
    ridx = rel_pos_index(ws)                      # (ws2, ws2)
    bias_table = sd[prefix + 'attn.relative_position_bias_table']  # ((2ws-1)^2, heads)
    # bias[head] gathered to (ws2, ws2)
    rpb = bias_table[ridx]                        # (ws2, ws2, heads)
    rpb = rpb.transpose(2, 0, 1)                  # (heads, ws2, ws2)

    reg = None
    if shift > 0:
        reg = region_ids(H, Wd, ws, shift)        # (H, W) region image (pre-roll grid space)

    qkv_w = sd[prefix + 'attn.qkv.weight']        # (3E, E)
    qkv_b = sd[prefix + 'attn.qkv.bias']          # (3E,)
    proj_w = sd[prefix + 'attn.proj.weight']      # (E, E)
    proj_b = sd[prefix + 'attn.proj.bias']        # (E,)

    out_grid = np.zeros((H, Wd, E), dtype=np.float64)
    for wy in range(wper):
        for wx in range(wper):
            win = grid[wy * ws:(wy + 1) * ws, wx * ws:(wx + 1) * ws, :]
            win = win.reshape(ws2, E)             # (ws2, E)
            qkv = win @ qkv_w.T + qkv_b           # (ws2, 3E)
            q = qkv[:, :E].reshape(ws2, heads, hd).transpose(1, 0, 2)
            k = qkv[:, E:2 * E].reshape(ws2, heads, hd).transpose(1, 0, 2)
            v = qkv[:, 2 * E:].reshape(ws2, heads, hd).transpose(1, 0, 2)
            scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)  # (heads, ws2, ws2)
            scores = scores + rpb
            if shift > 0:
                # mask: tokens in this window that map to different regions
                # cannot attend.  Build the window's region vector.
                rwin = reg[wy * ws:(wy + 1) * ws, wx * ws:(wx + 1) * ws].reshape(ws2)
                m = (rwin[:, None] != rwin[None, :])
                scores = np.where(m[None, :, :], scores - 1e9, scores)
            scores = scores - scores.max(axis=-1, keepdims=True)
            ex = np.exp(scores)
            attn = ex / ex.sum(axis=-1, keepdims=True)
            ctx = attn @ v                         # (heads, ws2, hd)
            ctx = ctx.transpose(1, 0, 2).reshape(ws2, E)
            ctx = ctx @ proj_w.T + proj_b
            out_grid[wy * ws:(wy + 1) * ws, wx * ws:(wx + 1) * ws, :] = \
                ctx.reshape(ws, ws, E)

    if shift > 0:
        out_grid = np.roll(out_grid, shift=(shift, shift), axis=(0, 1))
    attn_out = out_grid.reshape(H * Wd, E)
    x = shortcut + attn_out

    # MLP
    xn2 = layernorm_tokens(x, sd[prefix + 'norm2.weight'], sd[prefix + 'norm2.bias'])
    h = xn2 @ sd[prefix + 'mlp.fc1.weight'].T + sd[prefix + 'mlp.fc1.bias']
    h = gelu_np(h)
    h = h @ sd[prefix + 'mlp.fc2.weight'].T + sd[prefix + 'mlp.fc2.bias']
    return x + h


def pixel_shuffle(x, r):
    # CAI TNNetDepthToSpace mapping (NOT nn.PixelShuffle):
    #   out[ic, iy*r+sy, ix*r+sx] = in[(sx*r+sy)*C + ic, iy, ix]
    C2, H, Wd = x.shape
    C = C2 // (r * r)
    out = np.zeros((C, H * r, Wd * r), dtype=np.float64)
    for iy in range(H):
        for ix in range(Wd):
            for sx in range(r):
                for sy in range(r):
                    for ic in range(C):
                        out[ic, iy * r + sy, ix * r + sx] = \
                            x[(sx * r + sy) * C + ic, iy, ix]
    return out


def forward(inp, sd):
    H = inp.shape[1]
    Wd = inp.shape[2]
    f0 = conv2d(inp, sd['conv_first.weight'], sd['conv_first.bias'], 1)  # (E,H,W)
    x = map_to_tokens(f0)                                # (N, E)
    for li in range(NUM_RSTB):
        block_in = x
        for bi in range(DEPTH):
            shifted = (bi % 2 == 1)
            x = swin_layer(x, sd, f'layers.{li}.residual_group.blocks.{bi}.',
                           H, Wd, shifted)
        # RSTB conv over the map then back to tokens, + block input.
        xm = tokens_to_map(x, H, Wd)
        xm = conv2d(xm, sd[f'layers.{li}.conv.weight'],
                    sd[f'layers.{li}.conv.bias'], 1)
        x = map_to_tokens(xm) + block_in
    x = layernorm_tokens(x, sd['norm.weight'], sd['norm.bias'])
    xm = tokens_to_map(x, H, Wd)
    xm = conv2d(xm, sd['conv_after_body.weight'], sd['conv_after_body.bias'], 1)
    feat = xm + f0                                       # deep-feature residual
    # SR tail.
    u = conv2d(feat, sd['conv_before_upsample.0.weight'],
               sd['conv_before_upsample.0.bias'], 1)
    u = np.where(u >= 0, u, 0.2 * u)                     # LeakyReLU(0.2)
    u = conv2d(u, sd['upsample.0.weight'], sd['upsample.0.bias'], 1)
    u = pixel_shuffle(u, UPSCALE)
    out = conv2d(u, sd['conv_last.weight'], sd['conv_last.bias'], 1)
    return out


def main():
    sd = build_state_dict()
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    # Pinned input image: deterministic dyadic values (exact in f32 + JSON).
    x = np.zeros((IN_CH, IMG, IMG), dtype=np.float64)
    for c in range(IN_CH):
        for y in range(IMG):
            for px in range(IMG):
                x[c, y, px] = (((c * 64 + y * 8 + px) * 5) % 13 - 6) / 8.0

    img = forward(x, sd)
    print(f'input {x.shape} -> image {img.shape}')
    print(f'image stats: min {img.min():.4f} max {img.max():.4f} '
          f'mean {img.mean():.4f}')

    save_file(sd_f32, 'tests/fixtures/tiny_swinir.safetensors')
    config = {
        'model_type': 'swinir',
        'upscale': UPSCALE,
        'in_chans': IN_CH,
        'img_size': IMG,
        'window_size': WINDOW,
        'embed_dim': EMBED,
        'depths': [DEPTH] * NUM_RSTB,
        'num_heads': [NUM_HEADS] * NUM_RSTB,
        'mlp_ratio': MLP_RATIO,
        'num_feat': NUM_FEAT,
        'layer_norm_eps': LN_EPS,
    }
    with open('tests/fixtures/tiny_swinir_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_swinir_io.json', 'w') as f:
        json.dump({
            'input': x.tolist(),
            'image': img.tolist(),
            'image_size': img.shape[1],
        }, f)
    print(f'wrote tiny_swinir.safetensors ({len(sd_f32)} tensors) + config + io')

    # ---- fixture self-checks: each major piece must MATTER. ----
    base = img.copy()

    def perturb(key):
        alt = dict(sd)
        alt[key] = np.zeros_like(sd[key])
        return np.abs(forward(x, alt) - base).max()

    for key in ('conv_first.weight',
                'layers.0.residual_group.blocks.0.attn.qkv.weight',
                'layers.0.residual_group.blocks.1.attn.relative_position_bias_table',
                'layers.0.residual_group.blocks.0.mlp.fc1.weight',
                'layers.0.conv.weight', 'conv_after_body.weight',
                'conv_before_upsample.0.weight', 'upsample.0.weight',
                'conv_last.weight'):
        d = perturb(key)
        assert d > 1e-4, f'{key} had no effect ({d})'
        print(f'{key:62s} effect: max|diff| = {d:.4f}')

    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
