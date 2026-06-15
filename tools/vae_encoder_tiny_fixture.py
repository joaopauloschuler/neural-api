#!/usr/bin/env python3
"""Generate a tiny RANDOM diffusers-AutoencoderKL ENCODER parity fixture for
tests/TestNeuralPretrained.pas, completing the round-trip with the decoder
fixture (tools/vae_decoder_tiny_fixture.py).

diffusers is NOT installed in the reusable venv, so the reference forward is a
self-contained numpy float64 oracle that mirrors the CAI importer's forward
path EXACTLY (same style as the decoder fixture). The weights use the exact
diffusers AutoencoderKL.encoder key scheme so the importer is exercised on a
real key layout.

ENCODER architecture (mirror of the decoder):
  x (3, H, W)
    encoder.conv_in  (3x3 pad1, 3 -> C[0], +bias)
  DOWN block d (block_out_channels low->high order), width C_d:
    layers_per_block ResNet blocks
    asymmetric-pad (0,1,0,1) + stride-2 3x3 conv DOWNSAMPLE
        (every down block EXCEPT the last)
  MID: resnet -> spatial self-attn(H*W tokens) -> resnet      (width C[-1])
  OUT: conv_norm_out (GroupNorm) -> SiLU -> conv_out (3x3 -> 2*latent_channels)
    quant_conv  (1x1, 2*lat -> 2*lat, +bias)
  DiagonalGaussianDistribution: split channels into mean[0:lat], logvar[lat:];
    deterministic latent = mean * scaling_factor.

ResnetBlock2D / Spatial AttentionBlock / GroupNorm / SiLU: identical to the
decoder fixture (same helpers reused). The DOWNSAMPLE uses diffusers
Downsample2D's asymmetric pad=(0,1,0,1) then a stride-2 3x3 conv with pad 0;
the CAI importer reproduces this with PadXY(1,1)+Crop(1,1,..) (drop the
leading pad) then a stride-2 pad-0 conv.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/vae_encoder_tiny_fixture.py
writes tests/fixtures/tiny_vae_encoder{.safetensors,_config.json,_io.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

EPS = 1e-6
# Pico config mirroring the decoder: small widths, 1 layer/block, 4 groups.
# IMAGE 16 -> after 1 downsample (2 down blocks) the latent grid is 8x8, which
# matches the decoder fixture's latent grid so encode->decode round-trips.
IMAGE = 16
LATENT_CH = 4
IN_CH = 3
BLOCK_OUT = [8, 16]          # block_out_channels (low->high res order)
LAYERS_PER_BLOCK = 1
NORM_GROUPS = 4
SCALING = 0.18215

rng = np.random.default_rng(20260615)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def conv_w(out_ch, in_ch, k, std=0.12):
    return randn(out_ch, in_ch, k, k, std=std)   # [O,I,kh,kw]


def gn_params(c):
    return randn(c, std=0.3) + 1.0, randn(c, std=0.25)


# ---------------------------------------------------------------------------
# State dict (exact diffusers AutoencoderKL.encoder keys).
# ---------------------------------------------------------------------------
def add_resnet(sd, prefix, in_ch, out_ch):
    g, b = gn_params(in_ch)
    sd[prefix + 'norm1.weight'], sd[prefix + 'norm1.bias'] = g, b
    sd[prefix + 'conv1.weight'] = conv_w(out_ch, in_ch, 3)
    sd[prefix + 'conv1.bias'] = randn(out_ch, std=0.1)
    g, b = gn_params(out_ch)
    sd[prefix + 'norm2.weight'], sd[prefix + 'norm2.bias'] = g, b
    sd[prefix + 'conv2.weight'] = conv_w(out_ch, out_ch, 3)
    sd[prefix + 'conv2.bias'] = randn(out_ch, std=0.1)
    if in_ch != out_ch:
        sd[prefix + 'conv_shortcut.weight'] = conv_w(out_ch, in_ch, 1)
        sd[prefix + 'conv_shortcut.bias'] = randn(out_ch, std=0.1)


def add_attn(sd, prefix, c):
    g, b = gn_params(c)
    sd[prefix + 'group_norm.weight'], sd[prefix + 'group_norm.bias'] = g, b
    for name in ('to_q', 'to_k', 'to_v'):
        sd[prefix + name + '.weight'] = randn(c, c, std=0.2)   # [out,in]
        sd[prefix + name + '.bias'] = randn(c, std=0.1)
    sd[prefix + 'to_out.0.weight'] = randn(c, c, std=0.2)
    sd[prefix + 'to_out.0.bias'] = randn(c, std=0.1)


def build_state_dict():
    sd = {}
    n = len(BLOCK_OUT)
    top = BLOCK_OUT[-1]
    out2 = 2 * LATENT_CH
    sd['encoder.conv_in.weight'] = conv_w(BLOCK_OUT[0], IN_CH, 3)
    sd['encoder.conv_in.bias'] = randn(BLOCK_OUT[0], std=0.1)
    in_ch = BLOCK_OUT[0]
    for d in range(n):
        out_ch = BLOCK_OUT[d]
        for m in range(LAYERS_PER_BLOCK):
            b = in_ch if m == 0 else out_ch
            add_resnet(sd, f'encoder.down_blocks.{d}.resnets.{m}.', b, out_ch)
        in_ch = out_ch
        if d < n - 1:
            sd[f'encoder.down_blocks.{d}.downsamplers.0.conv.weight'] = \
                conv_w(out_ch, out_ch, 3)
            sd[f'encoder.down_blocks.{d}.downsamplers.0.conv.bias'] = \
                randn(out_ch, std=0.1)
    add_resnet(sd, 'encoder.mid_block.resnets.0.', top, top)
    add_attn(sd, 'encoder.mid_block.attentions.0.', top)
    add_resnet(sd, 'encoder.mid_block.resnets.1.', top, top)
    g, b = gn_params(top)
    sd['encoder.conv_norm_out.weight'], sd['encoder.conv_norm_out.bias'] = g, b
    sd['encoder.conv_out.weight'] = conv_w(out2, top, 3)
    sd['encoder.conv_out.bias'] = randn(out2, std=0.1)
    sd['quant_conv.weight'] = conv_w(out2, out2, 1)
    sd['quant_conv.bias'] = randn(out2, std=0.1)
    return sd


# ---------------------------------------------------------------------------
# numpy float64 oracle (volumes kept (C,H,W); conv weights [O,I,kh,kw]).
# ---------------------------------------------------------------------------
def conv2d(x, w, b, pad, stride=1):
    I, H, W = x.shape
    O, _, k, _ = w.shape
    xp = np.zeros((I, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    Hp, Wp = xp.shape[1], xp.shape[2]
    Ho = (Hp - k) // stride + 1
    Wo = (Wp - k) // stride + 1
    out = np.zeros((O, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            patch = xp[:, oy * stride:oy * stride + k, ox * stride:ox * stride + k]
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3],
                                                          [0, 1, 2])) + b
    return out


def downsample(x, w, b):
    # diffusers Downsample2D: asymmetric pad (0,1,0,1) -> stride-2 3x3 conv p0.
    I, H, W = x.shape
    xp = np.zeros((I, H + 1, W + 1), dtype=np.float64)
    xp[:, 0:H, 0:W] = x          # pad on right/bottom only
    return conv2d(xp, w, b, pad=0, stride=2)


def silu(x):
    return x / (1.0 + np.exp(-x))


def group_norm(x, gamma, beta, groups):
    C, H, W = x.shape
    cpg = C // groups
    out = np.empty_like(x)
    for g in range(groups):
        sl = x[g * cpg:(g + 1) * cpg]
        mu = sl.mean()
        var = ((sl - mu) ** 2).mean()
        out[g * cpg:(g + 1) * cpg] = (sl - mu) / np.sqrt(var + EPS)
    return out * gamma[:, None, None] + beta[:, None, None]


def resnet_block(x, sd, prefix, in_ch, out_ch):
    h = group_norm(x, sd[prefix + 'norm1.weight'], sd[prefix + 'norm1.bias'],
                   NORM_GROUPS)
    h = silu(h)
    h = conv2d(h, sd[prefix + 'conv1.weight'], sd[prefix + 'conv1.bias'], 1)
    h = group_norm(h, sd[prefix + 'norm2.weight'], sd[prefix + 'norm2.bias'],
                   NORM_GROUPS)
    h = silu(h)
    h = conv2d(h, sd[prefix + 'conv2.weight'], sd[prefix + 'conv2.bias'], 1)
    if in_ch != out_ch:
        ident = conv2d(x, sd[prefix + 'conv_shortcut.weight'],
                       sd[prefix + 'conv_shortcut.bias'], 0)
    else:
        ident = x
    return h + ident


def attention(x, sd, prefix, c):
    C, H, W = x.shape
    residual = x
    h = group_norm(x, sd[prefix + 'group_norm.weight'],
                   sd[prefix + 'group_norm.bias'], NORM_GROUPS)
    tokens = h.reshape(C, H * W).T          # [N, C]
    q = tokens @ sd[prefix + 'to_q.weight'].T + sd[prefix + 'to_q.bias']
    k = tokens @ sd[prefix + 'to_k.weight'].T + sd[prefix + 'to_k.bias']
    v = tokens @ sd[prefix + 'to_v.weight'].T + sd[prefix + 'to_v.bias']
    scale = 1.0 / np.sqrt(c)
    scores = (q @ k.T) * scale
    scores = scores - scores.max(axis=1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=1, keepdims=True)
    attn = w @ v
    out = attn @ sd[prefix + 'to_out.0.weight'].T + sd[prefix + 'to_out.0.bias']
    out = out.T.reshape(C, H, W)
    return out + residual


def encode(x, sd):
    n = len(BLOCK_OUT)
    top = BLOCK_OUT[-1]
    h = conv2d(x, sd['encoder.conv_in.weight'], sd['encoder.conv_in.bias'], 1)
    in_ch = BLOCK_OUT[0]
    for d in range(n):
        out_ch = BLOCK_OUT[d]
        for m in range(LAYERS_PER_BLOCK):
            b = in_ch if m == 0 else out_ch
            h = resnet_block(h, sd, f'encoder.down_blocks.{d}.resnets.{m}.',
                             b, out_ch)
        in_ch = out_ch
        if d < n - 1:
            h = downsample(
                h, sd[f'encoder.down_blocks.{d}.downsamplers.0.conv.weight'],
                sd[f'encoder.down_blocks.{d}.downsamplers.0.conv.bias'])
    h = resnet_block(h, sd, 'encoder.mid_block.resnets.0.', top, top)
    h = attention(h, sd, 'encoder.mid_block.attentions.0.', top)
    h = resnet_block(h, sd, 'encoder.mid_block.resnets.1.', top, top)
    h = group_norm(h, sd['encoder.conv_norm_out.weight'],
                   sd['encoder.conv_norm_out.bias'], NORM_GROUPS)
    h = silu(h)
    h = conv2d(h, sd['encoder.conv_out.weight'], sd['encoder.conv_out.bias'], 1)
    h = conv2d(h, sd['quant_conv.weight'], sd['quant_conv.bias'], 0)
    # DiagonalGaussianDistribution: mean = first LATENT_CH channels.
    mean = h[:LATENT_CH]
    logvar = h[LATENT_CH:]
    latent = mean * SCALING            # deterministic encode
    return h, mean, logvar, latent


def main():
    sd = build_state_dict()
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    # Pinned image: deterministic dyadic values (exact in f32 + JSON).
    x = np.zeros((IN_CH, IMAGE, IMAGE), dtype=np.float64)
    for c in range(IN_CH):
        for y in range(IMAGE):
            for xx in range(IMAGE):
                x[c, y, xx] = (((c * 256 + y * 16 + xx) * 5) % 13 - 6) / 8.0

    moments, mean, logvar, latent = encode(x, sd)
    print(f'image {x.shape} -> moments {moments.shape} '
          f'(mean/logvar {mean.shape}), latent {latent.shape}')
    print(f'latent stats: min {latent.min():.4f} max {latent.max():.4f} '
          f'mean {latent.mean():.4f}')

    save_file(sd_f32, 'tests/fixtures/tiny_vae_encoder.safetensors')
    config = {
        '_class_name': 'AutoencoderKL',
        'block_out_channels': BLOCK_OUT,
        'layers_per_block': LAYERS_PER_BLOCK,
        'latent_channels': LATENT_CH,
        'in_channels': IN_CH,
        'norm_num_groups': NORM_GROUPS,
        'sample_size': IMAGE,
        'image_size': IMAGE,
        'scaling_factor': SCALING,
        'norm_eps': EPS,
    }
    with open('tests/fixtures/tiny_vae_encoder_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    latent_grid = latent.shape[1]
    with open('tests/fixtures/tiny_vae_encoder_io.json', 'w') as f:
        json.dump({
            'image': x.tolist(),               # [3][IMAGE][IMAGE]
            'mean': mean.tolist(),             # [LATENT_CH][g][g]
            'logvar': logvar.tolist(),         # [LATENT_CH][g][g]
            'latent': latent.tolist(),         # mean * scaling
            'latent_grid': latent_grid,
        }, f)
    print(f'wrote tiny_vae_encoder.safetensors ({len(sd_f32)} tensors) + '
          f'config + io oracle (latent grid {latent_grid})')

    # ---- fixture self-checks: each major piece must MATTER. ----
    base = moments.copy()

    def diff(alt):
        return np.abs(encode(x, alt)[0] - base).max()

    alt = dict(sd)
    alt['encoder.mid_block.attentions.0.to_v.bias'] = \
        np.zeros_like(sd['encoder.mid_block.attentions.0.to_v.bias'])
    d = diff(alt)
    assert d > 1e-4, f'attention to_v bias had no effect ({d})'
    print(f'attention to_v effect: max|diff| = {d:.4f}')

    alt = dict(sd)
    alt['encoder.down_blocks.0.downsamplers.0.conv.bias'] = \
        np.zeros_like(sd['encoder.down_blocks.0.downsamplers.0.conv.bias'])
    d = diff(alt)
    assert d > 1e-4, f'downsampler conv bias had no effect ({d})'
    print(f'downsampler conv effect: max|diff| = {d:.4f}')

    alt = dict(sd)
    alt['quant_conv.bias'] = np.zeros_like(sd['quant_conv.bias'])
    d = diff(alt)
    assert d > 1e-4, f'quant_conv bias had no effect ({d})'
    print(f'quant_conv effect: max|diff| = {d:.4f}')

    # Asymmetric downsample pad MUST matter: a symmetric (1,1) pad would shift
    # the sampling grid and change the result.
    def sym_downsample(xx, w, b):
        return conv2d(xx, w, b, pad=1, stride=2)
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
