#!/usr/bin/env python3
"""Generate a tiny RANDOM diffusers UNet2DConditionModel parity fixture that
exercises the SD2.x / SDXL variant features for tests/TestNeuralPretrained.pas
(TestSDUNetLinearProjParity):

  * use_linear_projection = True
        proj_in / proj_out are nn.Linear (weight stored [out, in]) applied AFTER
        flatten (H,W,C)->(H*W,C), NOT the 1x1 conv ([out,in,1,1]) of SD1.x. The
        per-token map is IDENTICAL; only the stored tensor shape differs.
  * transformer_layers_per_block = 2
        each Transformer2DModel stacks 2 BasicTransformerBlocks
        (transformer_blocks.0, transformer_blocks.1), each loaded independently.

Everything else mirrors tools/sd_unet_tiny_fixture.py EXACTLY (same numpy
float64 oracle, same key scheme). diffusers is NOT installed; the reference
forward is the self-contained oracle below.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/sd_unet_linproj_tiny_fixture.py
writes tests/fixtures/tiny_sd_unet_linproj{.safetensors,_config.json,_io.json}.
"""
import json
import math
import numpy as np
from safetensors.numpy import save_file

try:
    from scipy.special import erf as _erf
except Exception:
    _verf = np.vectorize(math.erf)
    def _erf(x):
        return _verf(x)

RES_EPS = 1e-5
TR_EPS = 1e-6
LN_EPS = 1e-5

# ---- pico config (same shape as the v1 fixture, plus the two variant knobs) ----
IN_CH = 4
OUT_CH = 4
BLOCK_OUT = [16, 32]
LAYERS_PER_BLOCK = 1
CROSS_DIM = 12
NORM_GROUPS = 4
HEADS = 2
LATENT = 8
TEXT_SEQ = 5
TIME_EMBED_DIM = BLOCK_OUT[0] * 4
DOWN_TYPES = ['CrossAttnDownBlock2D', 'DownBlock2D']
UP_TYPES = ['UpBlock2D', 'CrossAttnUpBlock2D']
USE_LINEAR_PROJ = True
TRANSFORMER_DEPTH = 2     # transformer_layers_per_block

rng = np.random.default_rng(20260627)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def conv_w(out_ch, in_ch, k, std=0.10):
    return randn(out_ch, in_ch, k, k, std=std)


def lin_w(out_f, in_f, std=0.12):
    return randn(out_f, in_f, std=std)


def gn_params(c):
    return randn(c, std=0.3) + 1.0, randn(c, std=0.25)


# ===========================================================================
# numpy float64 oracle.
# ===========================================================================
def conv2d(x, w, b, pad, stride=1):
    I, H, W = x.shape
    O, _, k, _ = w.shape
    Ho = (H - k + 2 * pad) // stride + 1
    Wo = (W - k + 2 * pad) // stride + 1
    xp = np.zeros((I, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    out = np.zeros((O, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            patch = xp[:, oy * stride:oy * stride + k, ox * stride:ox * stride + k]
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3], [0, 1, 2])) + b
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_erf(x):
    return x * 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def group_norm(x, gamma, beta, groups, eps):
    C, H, W = x.shape
    cpg = C // groups
    out = np.empty_like(x)
    for g in range(groups):
        sl = x[g * cpg:(g + 1) * cpg]
        mu = sl.mean()
        var = ((sl - mu) ** 2).mean()
        out[g * cpg:(g + 1) * cpg] = (sl - mu) / np.sqrt(var + eps)
    return out * gamma[:, None, None] + beta[:, None, None]


def layer_norm(x, gamma, beta, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = np.exp(-math.log(10000.0) * np.arange(half) / half)
    emb = t * freqs
    return np.concatenate([np.cos(emb), np.sin(emb)]).astype(np.float64)


def resnet_block(x, temb, sd, prefix, in_ch, out_ch):
    h = group_norm(x, sd[prefix + 'norm1.weight'], sd[prefix + 'norm1.bias'],
                   NORM_GROUPS, RES_EPS)
    h = silu(h)
    h = conv2d(h, sd[prefix + 'conv1.weight'], sd[prefix + 'conv1.bias'], 1)
    t = silu(temb)
    t = t @ sd[prefix + 'time_emb_proj.weight'].T + sd[prefix + 'time_emb_proj.bias']
    h = h + t[:, None, None]
    h = group_norm(h, sd[prefix + 'norm2.weight'], sd[prefix + 'norm2.bias'],
                   NORM_GROUPS, RES_EPS)
    h = silu(h)
    h = conv2d(h, sd[prefix + 'conv2.weight'], sd[prefix + 'conv2.bias'], 1)
    if in_ch != out_ch:
        ident = conv2d(x, sd[prefix + 'conv_shortcut.weight'],
                       sd[prefix + 'conv_shortcut.bias'], 0)
    else:
        ident = x
    return h + ident


def attention(q_tokens, kv_tokens, sd, prefix, c, heads):
    head_dim = c // heads
    scale = 1.0 / math.sqrt(head_dim)
    q = q_tokens @ sd[prefix + 'to_q.weight'].T
    k = kv_tokens @ sd[prefix + 'to_k.weight'].T
    v = kv_tokens @ sd[prefix + 'to_v.weight'].T
    Nq = q.shape[0]
    out = np.zeros((Nq, c), dtype=np.float64)
    for hh in range(heads):
        sl = slice(hh * head_dim, (hh + 1) * head_dim)
        qh, kh, vh = q[:, sl], k[:, sl], v[:, sl]
        scores = (qh @ kh.T) * scale
        scores = scores - scores.max(axis=1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(axis=1, keepdims=True)
        out[:, sl] = w @ vh
    out = out @ sd[prefix + 'to_out.0.weight'].T + sd[prefix + 'to_out.0.bias']
    return out


def feed_forward(x, sd, prefix):
    proj = x @ sd[prefix + 'net.0.proj.weight'].T + sd[prefix + 'net.0.proj.bias']
    inner = proj.shape[1] // 2
    h, g = proj[:, :inner], proj[:, inner:]
    h = h * gelu_erf(g)
    return h @ sd[prefix + 'net.2.weight'].T + sd[prefix + 'net.2.bias']


def basic_block(tokens, enc, sd, prefix, c, heads):
    x = tokens
    h = layer_norm(x, sd[prefix + 'norm1.weight'], sd[prefix + 'norm1.bias'], LN_EPS)
    x = attention(h, h, sd, prefix + 'attn1.', c, heads) + x
    h = layer_norm(x, sd[prefix + 'norm2.weight'], sd[prefix + 'norm2.bias'], LN_EPS)
    x = attention(h, enc, sd, prefix + 'attn2.', c, heads) + x
    h = layer_norm(x, sd[prefix + 'norm3.weight'], sd[prefix + 'norm3.bias'], LN_EPS)
    x = feed_forward(h, sd, prefix + 'ff.') + x
    return x


def transformer2d(x, enc, sd, prefix, c, heads):
    C, H, W = x.shape
    residual = x
    h = group_norm(x, sd[prefix + 'norm.weight'], sd[prefix + 'norm.bias'],
                   NORM_GROUPS, TR_EPS)
    if USE_LINEAR_PROJ:
        # nn.Linear applied AFTER flatten (H,W,C)->(N,C).
        tokens = h.reshape(C, H * W).T                    # (N, C)
        tokens = tokens @ sd[prefix + 'proj_in.weight'].T + sd[prefix + 'proj_in.bias']
    else:
        h = conv2d(h, sd[prefix + 'proj_in.weight'], sd[prefix + 'proj_in.bias'], 0)
        tokens = h.reshape(C, H * W).T
    for b in range(TRANSFORMER_DEPTH):
        tokens = basic_block(tokens, enc, sd,
                             prefix + f'transformer_blocks.{b}.', c, heads)
    if USE_LINEAR_PROJ:
        tokens = tokens @ sd[prefix + 'proj_out.weight'].T + sd[prefix + 'proj_out.bias']
        h = tokens.T.reshape(C, H, W)
    else:
        h = tokens.T.reshape(C, H, W)
        h = conv2d(h, sd[prefix + 'proj_out.weight'], sd[prefix + 'proj_out.bias'], 0)
    return h + residual


def upsample_nearest(x):
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


# ===========================================================================
# State dict.
# ===========================================================================
def add_resnet(sd, prefix, in_ch, out_ch):
    g, b = gn_params(in_ch)
    sd[prefix + 'norm1.weight'], sd[prefix + 'norm1.bias'] = g, b
    sd[prefix + 'conv1.weight'] = conv_w(out_ch, in_ch, 3)
    sd[prefix + 'conv1.bias'] = randn(out_ch, std=0.1)
    sd[prefix + 'time_emb_proj.weight'] = lin_w(out_ch, TIME_EMBED_DIM)
    sd[prefix + 'time_emb_proj.bias'] = randn(out_ch, std=0.1)
    g, b = gn_params(out_ch)
    sd[prefix + 'norm2.weight'], sd[prefix + 'norm2.bias'] = g, b
    sd[prefix + 'conv2.weight'] = conv_w(out_ch, out_ch, 3)
    sd[prefix + 'conv2.bias'] = randn(out_ch, std=0.1)
    if in_ch != out_ch:
        sd[prefix + 'conv_shortcut.weight'] = conv_w(out_ch, in_ch, 1)
        sd[prefix + 'conv_shortcut.bias'] = randn(out_ch, std=0.1)


def add_attn(sd, prefix, c):
    inner = c * 4
    g, b = gn_params(c)
    sd[prefix + 'norm.weight'], sd[prefix + 'norm.bias'] = g, b
    # proj_in / proj_out: nn.Linear [out,in] when use_linear_projection.
    if USE_LINEAR_PROJ:
        sd[prefix + 'proj_in.weight'] = lin_w(c, c)
        sd[prefix + 'proj_out.weight'] = lin_w(c, c)
    else:
        sd[prefix + 'proj_in.weight'] = conv_w(c, c, 1)
        sd[prefix + 'proj_out.weight'] = conv_w(c, c, 1)
    sd[prefix + 'proj_in.bias'] = randn(c, std=0.1)
    sd[prefix + 'proj_out.bias'] = randn(c, std=0.1)
    for blk in range(TRANSFORMER_DEPTH):
        bp = prefix + f'transformer_blocks.{blk}.'
        sd[bp + 'norm1.weight'], sd[bp + 'norm1.bias'] = gn_params(c)
        sd[bp + 'norm2.weight'], sd[bp + 'norm2.bias'] = gn_params(c)
        sd[bp + 'norm3.weight'], sd[bp + 'norm3.bias'] = gn_params(c)
        sd[bp + 'attn1.to_q.weight'] = lin_w(c, c)
        sd[bp + 'attn1.to_k.weight'] = lin_w(c, c)
        sd[bp + 'attn1.to_v.weight'] = lin_w(c, c)
        sd[bp + 'attn1.to_out.0.weight'] = lin_w(c, c)
        sd[bp + 'attn1.to_out.0.bias'] = randn(c, std=0.1)
        sd[bp + 'attn2.to_q.weight'] = lin_w(c, c)
        sd[bp + 'attn2.to_k.weight'] = lin_w(c, CROSS_DIM)
        sd[bp + 'attn2.to_v.weight'] = lin_w(c, CROSS_DIM)
        sd[bp + 'attn2.to_out.0.weight'] = lin_w(c, c)
        sd[bp + 'attn2.to_out.0.bias'] = randn(c, std=0.1)
        sd[bp + 'ff.net.0.proj.weight'] = lin_w(2 * inner, c)
        sd[bp + 'ff.net.0.proj.bias'] = randn(2 * inner, std=0.1)
        sd[bp + 'ff.net.2.weight'] = lin_w(c, inner)
        sd[bp + 'ff.net.2.bias'] = randn(c, std=0.1)


def build_state_dict():
    sd = {}
    n = len(BLOCK_OUT)
    sd['conv_in.weight'] = conv_w(BLOCK_OUT[0], IN_CH, 3)
    sd['conv_in.bias'] = randn(BLOCK_OUT[0], std=0.1)
    sd['time_embedding.linear_1.weight'] = lin_w(TIME_EMBED_DIM, BLOCK_OUT[0])
    sd['time_embedding.linear_1.bias'] = randn(TIME_EMBED_DIM, std=0.1)
    sd['time_embedding.linear_2.weight'] = lin_w(TIME_EMBED_DIM, TIME_EMBED_DIM)
    sd['time_embedding.linear_2.bias'] = randn(TIME_EMBED_DIM, std=0.1)
    out_ch = BLOCK_OUT[0]
    for i, dtype in enumerate(DOWN_TYPES):
        in_ch = out_ch
        out_ch = BLOCK_OUT[i]
        is_final = (i == n - 1)
        for j in range(LAYERS_PER_BLOCK):
            rin = in_ch if j == 0 else out_ch
            add_resnet(sd, f'down_blocks.{i}.resnets.{j}.', rin, out_ch)
            if dtype == 'CrossAttnDownBlock2D':
                add_attn(sd, f'down_blocks.{i}.attentions.{j}.', out_ch)
        if not is_final:
            sd[f'down_blocks.{i}.downsamplers.0.conv.weight'] = conv_w(out_ch, out_ch, 3)
            sd[f'down_blocks.{i}.downsamplers.0.conv.bias'] = randn(out_ch, std=0.1)
    mid_ch = BLOCK_OUT[-1]
    add_resnet(sd, 'mid_block.resnets.0.', mid_ch, mid_ch)
    add_attn(sd, 'mid_block.attentions.0.', mid_ch)
    add_resnet(sd, 'mid_block.resnets.1.', mid_ch, mid_ch)
    rev = list(reversed(BLOCK_OUT))
    output_channel = rev[0]
    for i, utype in enumerate(UP_TYPES):
        prev_output_channel = output_channel
        output_channel = rev[i]
        input_channel = rev[min(i + 1, n - 1)]
        for j in range(LAYERS_PER_BLOCK + 1):
            res_skip = input_channel if j == LAYERS_PER_BLOCK else output_channel
            resnet_in = prev_output_channel if j == 0 else output_channel
            add_resnet(sd, f'up_blocks.{i}.resnets.{j}.',
                       resnet_in + res_skip, output_channel)
            if utype == 'CrossAttnUpBlock2D':
                add_attn(sd, f'up_blocks.{i}.attentions.{j}.', output_channel)
        if i != n - 1:
            sd[f'up_blocks.{i}.upsamplers.0.conv.weight'] = conv_w(output_channel, output_channel, 3)
            sd[f'up_blocks.{i}.upsamplers.0.conv.bias'] = randn(output_channel, std=0.1)
    g, b = gn_params(BLOCK_OUT[0])
    sd['conv_norm_out.weight'], sd['conv_norm_out.bias'] = g, b
    sd['conv_out.weight'] = conv_w(OUT_CH, BLOCK_OUT[0], 3)
    sd['conv_out.bias'] = randn(OUT_CH, std=0.1)
    return sd


def forward(latent, t, enc, sd):
    n = len(BLOCK_OUT)
    temb = timestep_embedding(t, BLOCK_OUT[0])
    temb = temb @ sd['time_embedding.linear_1.weight'].T + sd['time_embedding.linear_1.bias']
    temb = silu(temb)
    temb = temb @ sd['time_embedding.linear_2.weight'].T + sd['time_embedding.linear_2.bias']
    h = conv2d(latent, sd['conv_in.weight'], sd['conv_in.bias'], 1)
    skips = [h]
    out_ch = BLOCK_OUT[0]
    for i, dtype in enumerate(DOWN_TYPES):
        out_ch = BLOCK_OUT[i]
        is_final = (i == n - 1)
        c = out_ch
        in_ch = skips[-1].shape[0]
        for j in range(LAYERS_PER_BLOCK):
            rin = in_ch if j == 0 else out_ch
            h = resnet_block(h, temb, sd, f'down_blocks.{i}.resnets.{j}.', rin, out_ch)
            if dtype == 'CrossAttnDownBlock2D':
                h = transformer2d(h, enc, sd, f'down_blocks.{i}.attentions.{j}.', c, HEADS)
            skips.append(h)
        if not is_final:
            h = conv2d(h, sd[f'down_blocks.{i}.downsamplers.0.conv.weight'],
                       sd[f'down_blocks.{i}.downsamplers.0.conv.bias'], 1, stride=2)
            skips.append(h)
    mid_ch = BLOCK_OUT[-1]
    h = resnet_block(h, temb, sd, 'mid_block.resnets.0.', mid_ch, mid_ch)
    h = transformer2d(h, enc, sd, 'mid_block.attentions.0.', mid_ch, HEADS)
    h = resnet_block(h, temb, sd, 'mid_block.resnets.1.', mid_ch, mid_ch)
    rev = list(reversed(BLOCK_OUT))
    output_channel = rev[0]
    for i, utype in enumerate(UP_TYPES):
        prev_output_channel = output_channel
        output_channel = rev[i]
        input_channel = rev[min(i + 1, n - 1)]
        for j in range(LAYERS_PER_BLOCK + 1):
            res = skips.pop()
            h = np.concatenate([h, res], axis=0)
            resnet_in = prev_output_channel if j == 0 else output_channel
            res_skip = input_channel if j == LAYERS_PER_BLOCK else output_channel
            h = resnet_block(h, temb, sd, f'up_blocks.{i}.resnets.{j}.',
                             resnet_in + res_skip, output_channel)
            if utype == 'CrossAttnUpBlock2D':
                h = transformer2d(h, enc, sd, f'up_blocks.{i}.attentions.{j}.',
                                  output_channel, HEADS)
        if i != n - 1:
            h = upsample_nearest(h)
            h = conv2d(h, sd[f'up_blocks.{i}.upsamplers.0.conv.weight'],
                       sd[f'up_blocks.{i}.upsamplers.0.conv.bias'], 1)
    h = group_norm(h, sd['conv_norm_out.weight'], sd['conv_norm_out.bias'],
                   NORM_GROUPS, RES_EPS)
    h = silu(h)
    h = conv2d(h, sd['conv_out.weight'], sd['conv_out.bias'], 1)
    return h


def main():
    sd = build_state_dict()
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    latent = np.zeros((IN_CH, LATENT, LATENT), dtype=np.float64)
    for c in range(IN_CH):
        for y in range(LATENT):
            for x in range(LATENT):
                latent[c, y, x] = (((c * 64 + y * 8 + x) * 5) % 13 - 6) / 8.0
    enc = np.zeros((TEXT_SEQ, CROSS_DIM), dtype=np.float64)
    for s in range(TEXT_SEQ):
        for d in range(CROSS_DIM):
            enc[s, d] = (((s * 16 + d) * 3) % 11 - 5) / 8.0
    t = 17.0

    noise = forward(latent, t, enc, sd)
    print(f'latent {latent.shape} t={t} enc {enc.shape} -> noise {noise.shape}')
    print(f'noise stats: min {noise.min():.4f} max {noise.max():.4f} '
          f'mean {noise.mean():.4f}')

    save_file(sd_f32, 'tests/fixtures/tiny_sd_unet_linproj.safetensors')
    config = {
        '_class_name': 'UNet2DConditionModel',
        'in_channels': IN_CH,
        'out_channels': OUT_CH,
        'block_out_channels': BLOCK_OUT,
        'down_block_types': DOWN_TYPES,
        'up_block_types': UP_TYPES,
        'layers_per_block': LAYERS_PER_BLOCK,
        'cross_attention_dim': CROSS_DIM,
        'attention_head_dim': BLOCK_OUT[0] // HEADS,
        'norm_num_groups': NORM_GROUPS,
        'flip_sin_to_cos': True,
        'freq_shift': 0,
        'latent_size': LATENT,
        'text_seq_len': TEXT_SEQ,
        'use_linear_projection': USE_LINEAR_PROJ,
        'transformer_layers_per_block': TRANSFORMER_DEPTH,
    }
    with open('tests/fixtures/tiny_sd_unet_linproj_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_sd_unet_linproj_io.json', 'w') as f:
        json.dump({
            'latent': latent.tolist(),
            'timestep': t,
            'encoder_hidden_states': enc.tolist(),
            'noise': noise.tolist(),
        }, f)
    print(f'wrote tiny_sd_unet_linproj.safetensors ({len(sd_f32)} tensors) + config + io')

    # ---- fixture self-checks ----
    base = noise.copy()

    def effect(mut, name):
        alt = dict(sd)
        for k, v in mut.items():
            alt[k] = v
        d = np.abs(forward(latent, t, enc, alt) - base).max()
        assert d > 1e-4, f'{name} had no effect ({d})'
        print(f'{name} effect: max|diff| = {d:.4f}')

    # the SECOND stacked transformer block must MATTER (depth>1 actually loops).
    effect({'mid_block.attentions.0.transformer_blocks.1.attn2.to_v.weight':
            np.zeros_like(sd['mid_block.attentions.0.transformer_blocks.1.attn2.to_v.weight'])},
           'mid transformer_blocks.1 cross-attn to_v')
    # the linear proj_in must MATTER.
    effect({'mid_block.attentions.0.proj_in.weight':
            np.zeros_like(sd['mid_block.attentions.0.proj_in.weight'])},
           'mid linear proj_in')
    effect({'mid_block.attentions.0.proj_out.bias':
            np.zeros_like(sd['mid_block.attentions.0.proj_out.bias'])},
           'mid linear proj_out bias')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
