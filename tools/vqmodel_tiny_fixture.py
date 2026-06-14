#!/usr/bin/env python3
"""Generate a tiny RANDOM diffusers-VQModel (VQGAN / discrete image tokenizer)
parity fixture for tests/TestNeuralPretrained.pas.

diffusers is NOT installed in the reusable venv, so the reference forward is a
self-contained numpy float64 oracle that mirrors the CAI importer's forward
path EXACTLY (same style as the VAE encoder/decoder fixtures). The weights use
the exact diffusers VQModel key scheme so the importer is exercised on a real
key layout.

VQModel (diffusers) = the SAME encoder/decoder ResNet+attn blocks as
AutoencoderKL, but with a DISCRETE quantizer in between (and NO 0.18215 latent
scaling - that is AutoencoderKL only):

  ENCODE (image -> token-id grid):
    x (3, H, W)
      encoder.conv_in (3x3 pad1, 3 -> C[0], +bias)
    DOWN block d (block_out_channels low->high), width C_d:
      layers_per_block ResNet blocks
      asymmetric-pad (0,1,0,1) + stride-2 3x3 conv DOWNSAMPLE
          (every down block EXCEPT the last)
    MID: resnet -> spatial self-attn(H*W tokens) -> resnet      (width C[-1])
    OUT: conv_norm_out (GroupNorm) -> SiLU -> conv_out (3x3 -> latent_channels)
      quant_conv (1x1, latent_channels -> vq_embed_dim, +bias)
    VectorQuantizer: for each spatial (h,w) latent vector z, pick the codebook
      row e_k minimizing ||z - e_k||^2 -> integer token id.  (n_e x vq_embed_dim
      embedding table = quantize.embedding.weight)

  DECODE (token-id grid -> image):
    gather each id's codebook embedding -> (vq_embed_dim, H', W') latent
      post_quant_conv (1x1, vq_embed_dim -> latent_channels, +bias)
      decoder.conv_in (3x3, latent_channels -> C[-1])
    MID: resnet -> self-attn -> resnet
    UP block r (reversed block_out_channels): (layers_per_block+1) ResNet blocks,
      nearest 2x upsample + conv (every up block EXCEPT the last)
    OUT: conv_norm_out (GroupNorm) -> SiLU -> conv_out (3x3 -> 3 RGB, raw)

ResnetBlock2D / spatial AttentionBlock / GroupNorm / SiLU / asymmetric
downsample / nearest upsample: identical to the VAE fixtures (same helpers).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/vqmodel_tiny_fixture.py
writes tests/fixtures/tiny_vqmodel{.safetensors,_config.json,_io.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

EPS = 1e-6
# Pico config: small widths, 1 layer/block, 4 groups. IMAGE 16 -> after 1
# downsample (2 down blocks) the latent grid is 8x8. n_e=16 codebook entries.
IMAGE = 16
LATENT_CH = 4          # diffusers latent_channels (encoder conv_out width)
VQ_EMBED_DIM = 4       # diffusers vq_embed_dim (codebook vector dim); often==lat
IN_CH = 3
OUT_CH = 3
BLOCK_OUT = [8, 16]    # block_out_channels (low->high res order)
LAYERS_PER_BLOCK = 1
NORM_GROUPS = 4
NUM_VQ_EMBEDDINGS = 16 # n_e (codebook size)

rng = np.random.default_rng(20260616)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def conv_w(out_ch, in_ch, k, std=0.12):
    return randn(out_ch, in_ch, k, k, std=std)   # [O,I,kh,kw]


def gn_params(c):
    return randn(c, std=0.3) + 1.0, randn(c, std=0.25)


# ---------------------------------------------------------------------------
# State dict (exact diffusers VQModel keys).
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
    # ---- ENCODER ----
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
    sd['encoder.conv_out.weight'] = conv_w(LATENT_CH, top, 3)  # NOT 2*lat
    sd['encoder.conv_out.bias'] = randn(LATENT_CH, std=0.1)
    # ---- quant convs ----
    sd['quant_conv.weight'] = conv_w(VQ_EMBED_DIM, LATENT_CH, 1)
    sd['quant_conv.bias'] = randn(VQ_EMBED_DIM, std=0.1)
    sd['post_quant_conv.weight'] = conv_w(LATENT_CH, VQ_EMBED_DIM, 1)
    sd['post_quant_conv.bias'] = randn(LATENT_CH, std=0.1)
    # ---- codebook (n_e x vq_embed_dim). Spread entries so argmin is stable. ----
    emb = randn(NUM_VQ_EMBEDDINGS, VQ_EMBED_DIM, std=0.6)
    sd['quantize.embedding.weight'] = emb
    # ---- DECODER ----
    sd['decoder.conv_in.weight'] = conv_w(top, LATENT_CH, 3)
    sd['decoder.conv_in.bias'] = randn(top, std=0.1)
    add_resnet(sd, 'decoder.mid_block.resnets.0.', top, top)
    add_attn(sd, 'decoder.mid_block.attentions.0.', top)
    add_resnet(sd, 'decoder.mid_block.resnets.1.', top, top)
    in_ch = top
    nres = LAYERS_PER_BLOCK + 1
    for r in range(n):
        rev = n - 1 - r
        out_ch = BLOCK_OUT[rev]
        for m in range(nres):
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


def upsample_nearest(x):
    # diffusers Upsample2D(mode="nearest") 2x: replicate each pixel 2x2.
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


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


def encode_to_latent(x, sd):
    """image -> (vq_embed_dim, H', W') latent after quant_conv (pre-quantize)."""
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
    return h     # (vq_embed_dim, H', W')


def quantize(latent, emb):
    """latent (E,H,W), emb (n_e,E) -> token ids (H,W) by argmin L2."""
    E, H, W = latent.shape
    z = latent.reshape(E, H * W).T          # [N, E]
    # ||z - e||^2 = |z|^2 - 2 z.e + |e|^2 ; argmin over e.
    d = (np.sum(z ** 2, axis=1, keepdims=True)
         - 2.0 * z @ emb.T
         + np.sum(emb ** 2, axis=1)[None, :])   # [N, n_e]
    ids = np.argmin(d, axis=1)                  # [N]
    return ids.reshape(H, W)


def decode_from_ids(ids, sd):
    """token ids (H',W') -> RGB image (3, IMG, IMG)."""
    emb = sd['quantize.embedding.weight']       # (n_e, E)
    H, W = ids.shape
    E = emb.shape[1]
    z = emb[ids.reshape(-1)].T.reshape(E, H, W)  # gather -> (E,H,W)
    n = len(BLOCK_OUT)
    top = BLOCK_OUT[-1]
    h = conv2d(z, sd['post_quant_conv.weight'], sd['post_quant_conv.bias'], 0)
    h = conv2d(h, sd['decoder.conv_in.weight'], sd['decoder.conv_in.bias'], 1)
    h = resnet_block(h, sd, 'decoder.mid_block.resnets.0.', top, top)
    h = attention(h, sd, 'decoder.mid_block.attentions.0.', top)
    h = resnet_block(h, sd, 'decoder.mid_block.resnets.1.', top, top)
    in_ch = top
    nres = LAYERS_PER_BLOCK + 1
    for r in range(n):
        rev = n - 1 - r
        out_ch = BLOCK_OUT[rev]
        for m in range(nres):
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

    # Pinned image: deterministic dyadic values (exact in f32 + JSON).
    x = np.zeros((IN_CH, IMAGE, IMAGE), dtype=np.float64)
    for c in range(IN_CH):
        for y in range(IMAGE):
            for xx in range(IMAGE):
                x[c, y, xx] = (((c * 256 + y * 16 + xx) * 5) % 13 - 6) / 8.0

    latent = encode_to_latent(x, sd)
    emb = sd['quantize.embedding.weight']
    ids = quantize(latent, emb)
    decoded = decode_from_ids(ids, sd)
    grid = ids.shape[0]
    print(f'image {x.shape} -> latent {latent.shape} -> ids {ids.shape}')
    print(f'token ids:\n{ids}')
    print(f'decoded {decoded.shape}, min {decoded.min():.4f} '
          f'max {decoded.max():.4f}')

    save_file(sd_f32, 'tests/fixtures/tiny_vqmodel.safetensors')
    config = {
        '_class_name': 'VQModel',
        'block_out_channels': BLOCK_OUT,
        'layers_per_block': LAYERS_PER_BLOCK,
        'latent_channels': LATENT_CH,
        'vq_embed_dim': VQ_EMBED_DIM,
        'num_vq_embeddings': NUM_VQ_EMBEDDINGS,
        'in_channels': IN_CH,
        'out_channels': OUT_CH,
        'norm_num_groups': NORM_GROUPS,
        'sample_size': IMAGE,
        'image_size': IMAGE,
        'norm_eps': EPS,
    }
    with open('tests/fixtures/tiny_vqmodel_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_vqmodel_io.json', 'w') as f:
        json.dump({
            'image': x.tolist(),                # [3][IMAGE][IMAGE]
            'image_size': IMAGE,
            'latent_grid': grid,
            'latent': latent.tolist(),          # [E][g][g] pre-quantize
            'token_ids': ids.tolist(),          # [g][g] integer codebook ids
            'decoded': decoded.tolist(),        # [3][IMAGE][IMAGE] RGB raw
        }, f)
    print(f'wrote tiny_vqmodel.safetensors ({len(sd_f32)} tensors) + config + io')

    # ---- fixture self-checks: each major piece must MATTER. ----
    base_ids = ids.copy()
    base_dec = decoded.copy()

    # The codebook MUST drive the token ids: scrambling it changes ids.
    alt = dict(sd)
    alt2_emb = emb.copy()
    alt2_emb[:] = emb[::-1]
    alt['quantize.embedding.weight'] = alt2_emb
    ids2 = quantize(encode_to_latent(x, alt), alt2_emb)
    assert not np.array_equal(ids2, base_ids), 'codebook had no effect on ids'
    print('codebook reorder changes ids: OK')

    # quant_conv MUST matter for ids.
    alt = dict(sd)
    alt['quant_conv.bias'] = np.zeros_like(sd['quant_conv.bias'])
    ids3 = quantize(encode_to_latent(x, alt), emb)
    assert not np.array_equal(ids3, base_ids), 'quant_conv bias no effect on ids'
    print('quant_conv bias changes ids: OK')

    # post_quant_conv MUST matter for the decode.
    alt = dict(sd)
    alt['post_quant_conv.bias'] = np.zeros_like(sd['post_quant_conv.bias'])
    dec2 = decode_from_ids(base_ids, alt)
    assert np.abs(dec2 - base_dec).max() > 1e-4, 'post_quant_conv no effect'
    print('post_quant_conv bias changes decode: OK')

    # The ids must NOT be all-identical (a degenerate codebook would be useless).
    assert len(np.unique(base_ids)) > 1, 'all token ids identical (degenerate)'
    print(f'distinct token ids used: {len(np.unique(base_ids))} '
          f'of {NUM_VQ_EMBEDDINGS}')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
