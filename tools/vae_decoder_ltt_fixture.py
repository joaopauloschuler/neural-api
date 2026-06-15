#!/usr/bin/env python3
"""Generate the tiny VAE DECODER fixture used by examples/LatentTextToImage and
its regression test TestLatentTextToImageSmoke.

This is the SAME diffusers-AutoencoderKL decoder oracle as
tools/vae_decoder_tiny_fixture.py, but sized to match the committed pico PixArt
fixture (tools/make_pico_pixart_fixture.py): LATENT grid 6 (PixArt sample_size)
and latent_channels 4 (PixArt in_channels), so the sampled PixArt latent flows
straight into this VAE decoder with NO reshaping. The end-to-end chain is

  caller T5 states -> PixArt CFG DDIM denoise -> (6,6,4) latent
                   -> VAE decode (/0.18215) -> (24,24,3) RGB image

(2 up blocks => one 2x upsample => 6 -> 12, then conv keeps 12; conv_out stays
12... wait: with 2 blocks there is exactly ONE upsample, 6 -> 12. The image is
12x12x3.) See the printed shape line for the authoritative numbers.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/vae_decoder_ltt_fixture.py
writes tests/fixtures/tiny_vae_decoder_ltt{.safetensors,_config.json,_io.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

EPS = 1e-6
# Matched to the pico PixArt fixture: latent grid 6, latent channels 4.
LATENT = 6
LATENT_CH = 4
OUT_CH = 3
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
    top = BLOCK_OUT[-1]
    sd['post_quant_conv.weight'] = conv_w(LATENT_CH, LATENT_CH, 1)
    sd['post_quant_conv.bias'] = randn(LATENT_CH, std=0.1)
    sd['decoder.conv_in.weight'] = conv_w(top, LATENT_CH, 3)
    sd['decoder.conv_in.bias'] = randn(top, std=0.1)
    add_resnet(sd, 'decoder.mid_block.resnets.0.', top, top)
    add_attn(sd, 'decoder.mid_block.attentions.0.', top)
    add_resnet(sd, 'decoder.mid_block.resnets.1.', top, top)
    n_resnet = LAYERS_PER_BLOCK + 1
    in_ch = top
    n = len(BLOCK_OUT)
    for r in range(n):
        out_ch = BLOCK_OUT[n - 1 - r]
        for m in range(n_resnet):
            b = in_ch if m == 0 else out_ch
            add_resnet(sd, f'decoder.up_blocks.{r}.resnets.{m}.', b, out_ch)
        in_ch = out_ch
        if r < n - 1:
            sd[f'decoder.up_blocks.{r}.upsamplers.0.conv.weight'] = \
                conv_w(out_ch, out_ch, 3)
            sd[f'decoder.up_blocks.{r}.upsamplers.0.conv.bias'] = \
                randn(out_ch, std=0.1)
    g, b = gn_params(BLOCK_OUT[0])
    sd['decoder.conv_norm_out.weight'], sd['decoder.conv_norm_out.bias'] = g, b
    sd['decoder.conv_out.weight'] = conv_w(OUT_CH, BLOCK_OUT[0], 3)
    sd['decoder.conv_out.bias'] = randn(OUT_CH, std=0.1)
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
    scores = (q @ k.T) * scale              # [N, N]
    scores = scores - scores.max(axis=1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=1, keepdims=True)
    attn = w @ v                            # [N, C]
    out = attn @ sd[prefix + 'to_out.0.weight'].T + sd[prefix + 'to_out.0.bias']
    out = out.T.reshape(C, H, W)
    return out + residual


def upsample_nearest(x):
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def forward(z, sd):
    top = BLOCK_OUT[-1]
    h = z * (1.0 / SCALING)
    h = conv2d(h, sd['post_quant_conv.weight'], sd['post_quant_conv.bias'], 0)
    h = conv2d(h, sd['decoder.conv_in.weight'], sd['decoder.conv_in.bias'], 1)
    h = resnet_block(h, sd, 'decoder.mid_block.resnets.0.', top, top)
    h = attention(h, sd, 'decoder.mid_block.attentions.0.', top)
    h = resnet_block(h, sd, 'decoder.mid_block.resnets.1.', top, top)
    n_resnet = LAYERS_PER_BLOCK + 1
    in_ch = top
    n = len(BLOCK_OUT)
    for r in range(n):
        out_ch = BLOCK_OUT[n - 1 - r]
        for m in range(n_resnet):
            b = in_ch if m == 0 else out_ch
            h = resnet_block(h, sd, f'decoder.up_blocks.{r}.resnets.{m}.',
                             b, out_ch)
        in_ch = out_ch
        if r < n - 1:
            h = upsample_nearest(h)
            h = conv2d(h, sd[f'decoder.up_blocks.{r}.upsamplers.0.conv.weight'],
                       sd[f'decoder.up_blocks.{r}.upsamplers.0.conv.bias'], 1)
    h = group_norm(h, sd['decoder.conv_norm_out.weight'],
                   sd['decoder.conv_norm_out.bias'], NORM_GROUPS)
    h = silu(h)
    h = conv2d(h, sd['decoder.conv_out.weight'], sd['decoder.conv_out.bias'], 1)
    return h


def main():
    sd = build_state_dict()
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    # Pinned latent: deterministic dyadic values (exact in f32 + JSON).
    z = np.zeros((LATENT_CH, LATENT, LATENT), dtype=np.float64)
    for c in range(LATENT_CH):
        for y in range(LATENT):
            for x in range(LATENT):
                z[c, y, x] = (((c * 64 + y * 8 + x) * 5) % 13 - 6) / 8.0

    img = forward(z, sd)
    print(f'latent {z.shape} -> image {img.shape}')
    print(f'image stats: min {img.min():.4f} max {img.max():.4f} '
          f'mean {img.mean():.4f}')

    save_file(sd_f32, 'tests/fixtures/tiny_vae_decoder_ltt.safetensors')
    config = {
        '_class_name': 'AutoencoderKL',
        'block_out_channels': BLOCK_OUT,
        'layers_per_block': LAYERS_PER_BLOCK,
        'latent_channels': LATENT_CH,
        'out_channels': OUT_CH,
        'norm_num_groups': NORM_GROUPS,
        'latent_size': LATENT,
        'scaling_factor': SCALING,
        'norm_eps': EPS,
    }
    with open('tests/fixtures/tiny_vae_decoder_ltt_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_vae_decoder_ltt_io.json', 'w') as f:
        json.dump({
            'latent': z.tolist(),
            'image': img.tolist(),
            'image_size': img.shape[1],
        }, f)
    print(f'wrote tiny_vae_decoder_ltt.safetensors ({len(sd_f32)} tensors) + '
          f'config + io oracle')


if __name__ == '__main__':
    main()
