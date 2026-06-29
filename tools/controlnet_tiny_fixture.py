#!/usr/bin/env python3
"""Generate a tiny RANDOM diffusers ControlNetModel parity fixture for
tests/TestNeuralPretrained.pas (TestControlNetParity).

diffusers is NOT installed in the reusable venv, so the reference forward is a
self-contained numpy float64 oracle that mirrors the CAI importer's forward
path EXACTLY (the make_pico recipe). The weights use the exact diffusers
ControlNetModel key scheme so the importer is exercised on a real key layout.

ControlNet adds spatial control (a canny edge / depth / pose map -> image) to
latent diffusion. Architecture (diffusers ControlNetModel):
  * A trainable COPY of the SD UNet ENCODER (conv_in + time_embedding + down
    blocks) + the MID block. ControlNet has NO up blocks.
  * A small conv "hint" stem (controlnet_cond_embedding): conv_in (3x3 pad1) +
    a few (conv 3x3 pad1, conv 3x3 stride2 pad1) pairs each followed by SiLU,
    then a ZERO-initialised conv_out (3x3 pad1) -> block_out[0] channels. Its
    output is ADDED to the conv_in output before the first down block.
  * Per-resolution residual taps: the conv_in-plus-hint output and every down
    block / downsampler / mid output is passed through a ZERO-initialised 1x1
    conv (controlnet_down_blocks.{i}, controlnet_mid_block). These become the
    down_block_res_samples + mid_block_res_sample that get added into the
    frozen base UNet's decoder skip connections.

This oracle / fixture reuses the SD UNet pico encoder shapes (the encoder is
identical), cf. tools/sd_unet_tiny_fixture.py.

PICO config (architecturally complete, tiny):
  block_out_channels  = [16, 32]
  down_block_types    = [CrossAttnDownBlock2D, DownBlock2D]
  layers_per_block    = 1
  cross_attention_dim = 12
  num heads           = 2    (head_dim = C // 2)
  norm_num_groups     = 4
  latent grid         = 8x8, in_channels = 4
  text seq len        = 5
  conditioning_channels = 3 (RGB control image)
  conditioning_embedding_out_channels = [8, 16]  (cond embed dims)
  control image grid  = 16x16 (1 stride-2 stage -> 8x8 latent grid)

Down-block residual taps follow the diffusers ControlNet ordering: tap[0] is
the conv_in (+hint) output, then one tap per down resnet(+attn) output, then
one tap per downsampler output. controlnet_mid_block taps the mid output.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/controlnet_tiny_fixture.py
writes tests/fixtures/tiny_controlnet{.safetensors,_config.json,_io.json}.
Needs numpy + safetensors + scipy(erf, optional) only.
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

# ---- pico config ----
IN_CH = 4
BLOCK_OUT = [16, 32]
LAYERS_PER_BLOCK = 1
CROSS_DIM = 12
NORM_GROUPS = 4
HEADS = 2
LATENT = 8
TEXT_SEQ = 5
TIME_EMBED_DIM = BLOCK_OUT[0] * 4   # 64
DOWN_TYPES = ['CrossAttnDownBlock2D', 'DownBlock2D']
COND_CHANNELS = 3
COND_EMBED_OUT = [8, 16]            # conditioning_embedding_out_channels
COND_GRID = 16                     # control image grid (1 stride-2 -> 8)

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
# numpy float64 oracle (shared with the SD UNet fixture).
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
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3],
                                                          [0, 1, 2])) + b
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
    h = conv2d(h, sd[prefix + 'proj_in.weight'], sd[prefix + 'proj_in.bias'], 0)
    tokens = h.reshape(C, H * W).T
    tokens = basic_block(tokens, enc, sd, prefix + 'transformer_blocks.0.', c, heads)
    h = tokens.T.reshape(C, H, W)
    h = conv2d(h, sd[prefix + 'proj_out.weight'], sd[prefix + 'proj_out.bias'], 0)
    return h + residual


# ===========================================================================
# State dict (exact diffusers ControlNetModel keys).
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
    sd[prefix + 'proj_in.weight'] = conv_w(c, c, 1)
    sd[prefix + 'proj_in.bias'] = randn(c, std=0.1)
    bp = prefix + 'transformer_blocks.0.'
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
    sd[prefix + 'proj_out.weight'] = conv_w(c, c, 1)
    sd[prefix + 'proj_out.bias'] = randn(c, std=0.1)


def build_state_dict():
    sd = {}
    n = len(BLOCK_OUT)
    # conv_in
    sd['conv_in.weight'] = conv_w(BLOCK_OUT[0], IN_CH, 3)
    sd['conv_in.bias'] = randn(BLOCK_OUT[0], std=0.1)
    # time MLP
    sd['time_embedding.linear_1.weight'] = lin_w(TIME_EMBED_DIM, BLOCK_OUT[0])
    sd['time_embedding.linear_1.bias'] = randn(TIME_EMBED_DIM, std=0.1)
    sd['time_embedding.linear_2.weight'] = lin_w(TIME_EMBED_DIM, TIME_EMBED_DIM)
    sd['time_embedding.linear_2.bias'] = randn(TIME_EMBED_DIM, std=0.1)
    # ---- controlnet_cond_embedding (hint stem) ----
    sd['controlnet_cond_embedding.conv_in.weight'] = conv_w(COND_EMBED_OUT[0], COND_CHANNELS, 3)
    sd['controlnet_cond_embedding.conv_in.bias'] = randn(COND_EMBED_OUT[0], std=0.1)
    # blocks: for each i, (conv same-dim, conv stride2 to next dim).
    bi = 0
    for i in range(len(COND_EMBED_OUT) - 1):
        ch_in = COND_EMBED_OUT[i]
        ch_out = COND_EMBED_OUT[i + 1]
        sd[f'controlnet_cond_embedding.blocks.{bi}.weight'] = conv_w(ch_in, ch_in, 3)
        sd[f'controlnet_cond_embedding.blocks.{bi}.bias'] = randn(ch_in, std=0.1)
        bi += 1
        sd[f'controlnet_cond_embedding.blocks.{bi}.weight'] = conv_w(ch_out, ch_in, 3)
        sd[f'controlnet_cond_embedding.blocks.{bi}.bias'] = randn(ch_out, std=0.1)
        bi += 1
    # conv_out: ZERO-init, maps last cond dim -> block_out[0].
    sd['controlnet_cond_embedding.conv_out.weight'] = np.zeros(
        (BLOCK_OUT[0], COND_EMBED_OUT[-1], 3, 3), dtype=np.float64)
    sd['controlnet_cond_embedding.conv_out.bias'] = np.zeros(BLOCK_OUT[0], dtype=np.float64)
    # ---- down blocks (encoder copy) ----
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
    # ---- mid block ----
    mid_ch = BLOCK_OUT[-1]
    add_resnet(sd, 'mid_block.resnets.0.', mid_ch, mid_ch)
    add_attn(sd, 'mid_block.attentions.0.', mid_ch)
    add_resnet(sd, 'mid_block.resnets.1.', mid_ch, mid_ch)
    # ---- controlnet zero-conv taps ----
    # one tap per down_block_res_sample: tap0 = conv_in(+hint); then per
    # resnet(+attn) skip; then per downsampler skip. All 1x1 ZERO convs.
    tap_channels = tap_channel_list()
    for k, ch in enumerate(tap_channels):
        sd[f'controlnet_down_blocks.{k}.weight'] = np.zeros((ch, ch, 1, 1), dtype=np.float64)
        sd[f'controlnet_down_blocks.{k}.bias'] = np.zeros(ch, dtype=np.float64)
    sd['controlnet_mid_block.weight'] = np.zeros((mid_ch, mid_ch, 1, 1), dtype=np.float64)
    sd['controlnet_mid_block.bias'] = np.zeros(mid_ch, dtype=np.float64)
    return sd


def tap_channel_list():
    """Channels of each down_block_res_sample (in tap order)."""
    n = len(BLOCK_OUT)
    taps = [BLOCK_OUT[0]]      # conv_in(+hint) output
    out_ch = BLOCK_OUT[0]
    for i in range(n):
        out_ch = BLOCK_OUT[i]
        is_final = (i == n - 1)
        for j in range(LAYERS_PER_BLOCK):
            taps.append(out_ch)
        if not is_final:
            taps.append(out_ch)
    return taps


def cond_embedding(cond, sd):
    h = conv2d(cond, sd['controlnet_cond_embedding.conv_in.weight'],
               sd['controlnet_cond_embedding.conv_in.bias'], 1)
    h = silu(h)
    bi = 0
    for i in range(len(COND_EMBED_OUT) - 1):
        h = conv2d(h, sd[f'controlnet_cond_embedding.blocks.{bi}.weight'],
                   sd[f'controlnet_cond_embedding.blocks.{bi}.bias'], 1)
        h = silu(h)
        bi += 1
        h = conv2d(h, sd[f'controlnet_cond_embedding.blocks.{bi}.weight'],
                   sd[f'controlnet_cond_embedding.blocks.{bi}.bias'], 1, stride=2)
        h = silu(h)
        bi += 1
    h = conv2d(h, sd['controlnet_cond_embedding.conv_out.weight'],
               sd['controlnet_cond_embedding.conv_out.bias'], 1)
    return h


# ===========================================================================
# Full ControlNet forward -> (down_block_res_samples, mid_block_res_sample).
# ===========================================================================
def forward(latent, t, enc, cond, sd):
    n = len(BLOCK_OUT)
    temb = timestep_embedding(t, BLOCK_OUT[0])
    temb = temb @ sd['time_embedding.linear_1.weight'].T + sd['time_embedding.linear_1.bias']
    temb = silu(temb)
    temb = temb @ sd['time_embedding.linear_2.weight'].T + sd['time_embedding.linear_2.bias']
    # conv_in + hint
    h = conv2d(latent, sd['conv_in.weight'], sd['conv_in.bias'], 1)
    h = h + cond_embedding(cond, sd)
    skips = [h]
    # ---- down ----
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
    # ---- mid ----
    mid_ch = BLOCK_OUT[-1]
    h = resnet_block(h, temb, sd, 'mid_block.resnets.0.', mid_ch, mid_ch)
    h = transformer2d(h, enc, sd, 'mid_block.attentions.0.', mid_ch, HEADS)
    h = resnet_block(h, temb, sd, 'mid_block.resnets.1.', mid_ch, mid_ch)
    # ---- zero-conv taps ----
    down_res = []
    for k, s in enumerate(skips):
        r = conv2d(s, sd[f'controlnet_down_blocks.{k}.weight'],
                   sd[f'controlnet_down_blocks.{k}.bias'], 0)
        down_res.append(r)
    mid_res = conv2d(h, sd['controlnet_mid_block.weight'],
                     sd['controlnet_mid_block.bias'], 0)
    return down_res, mid_res


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
    cond = np.zeros((COND_CHANNELS, COND_GRID, COND_GRID), dtype=np.float64)
    for c in range(COND_CHANNELS):
        for y in range(COND_GRID):
            for x in range(COND_GRID):
                cond[c, y, x] = (((c * 1024 + y * 32 + x) * 7) % 17 - 8) / 16.0
    t = 17.0

    # With zero-init taps the residuals are all zero; for a MEANINGFUL parity
    # test we fill the zero convs with small random weights AFTER the
    # architectural zero-init (real fine-tuned ControlNets have non-zero taps).
    # The importer must load whatever is in the checkpoint, so we exercise that.
    # Fill f32 first, then re-derive f64 so both hold IDENTICAL f32 values.
    fill_taps(sd_f32)
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    down_res, mid_res = forward(latent, t, enc, cond, sd)
    print(f'latent {latent.shape} t={t} enc {enc.shape} cond {cond.shape}')
    print(f'down_block_res_samples: {len(down_res)} tensors '
          f'shapes {[r.shape for r in down_res]}')
    print(f'mid_block_res_sample shape {mid_res.shape}')

    save_file(sd_f32, 'tests/fixtures/tiny_controlnet.safetensors')
    config = {
        '_class_name': 'ControlNetModel',
        'in_channels': IN_CH,
        'block_out_channels': BLOCK_OUT,
        'down_block_types': DOWN_TYPES,
        'layers_per_block': LAYERS_PER_BLOCK,
        'cross_attention_dim': CROSS_DIM,
        'attention_head_dim': BLOCK_OUT[0] // HEADS,
        'norm_num_groups': NORM_GROUPS,
        'flip_sin_to_cos': True,
        'freq_shift': 0,
        'latent_size': LATENT,
        'text_seq_len': TEXT_SEQ,
        'conditioning_channels': COND_CHANNELS,
        'conditioning_embedding_out_channels': COND_EMBED_OUT,
        'cond_size': COND_GRID,
    }
    with open('tests/fixtures/tiny_controlnet_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_controlnet_io.json', 'w') as f:
        json.dump({
            'latent': latent.tolist(),
            'timestep': t,
            'encoder_hidden_states': enc.tolist(),
            'controlnet_cond': cond.tolist(),
            'down_block_res_samples': [r.tolist() for r in down_res],
            'mid_block_res_sample': mid_res.tolist(),
        }, f)
    print(f'wrote tiny_controlnet.safetensors ({len(sd_f32)} tensors) + config + io')

    # ---- fixture self-checks: every major piece must MATTER. ----
    base_down, base_mid = down_res, mid_res

    def maxdiff(dr, mr):
        d = max(np.abs(a - b).max() for a, b in zip(dr, base_down))
        d = max(d, np.abs(mr - base_mid).max())
        return d

    # timestep matters
    dr, mr = forward(latent, 99.0, enc, cond, sd)
    d = maxdiff(dr, mr); assert d > 1e-4, f'timestep no effect ({d})'
    print(f'timestep effect: {d:.4f}')
    # text matters
    dr, mr = forward(latent, t, enc + 0.5, cond, sd)
    d = maxdiff(dr, mr); assert d > 1e-4, f'text no effect ({d})'
    print(f'text effect: {d:.4f}')
    # control image matters
    dr, mr = forward(latent, t, enc, cond + 0.3, sd)
    d = maxdiff(dr, mr); assert d > 1e-4, f'cond no effect ({d})'
    print(f'control-image effect: {d:.4f}')

    def effect(mut, name):
        alt = dict(sd)
        alt.update(mut)
        dr, mr = forward(latent, t, enc, cond, alt)
        d = maxdiff(dr, mr)
        assert d > 1e-4, f'{name} no effect ({d})'
        print(f'{name} effect: {d:.4f}')

    # hint stem matters (conv_in of cond embedding)
    effect({'controlnet_cond_embedding.conv_in.bias':
            sd['controlnet_cond_embedding.conv_in.bias'] + 1.0}, 'hint conv_in')
    # a mid-block zero conv tap matters
    effect({'controlnet_mid_block.bias':
            sd['controlnet_mid_block.bias'] + 1.0}, 'mid tap bias')
    # a down zero conv tap matters
    effect({'controlnet_down_blocks.0.bias':
            sd['controlnet_down_blocks.0.bias'] + 1.0}, 'down tap0 bias')
    print('all fixture self-checks passed')


def fill_taps(sd):
    """Replace the architecturally-zero tap convs (and the hint conv_out) with
    small random f32 weights so the parity test is meaningful (a real
    fine-tuned ControlNet has non-zero taps). Mutates sd (f32) in place.
    """
    tap_channels = tap_channel_list()
    for k, ch in enumerate(tap_channels):
        sd[f'controlnet_down_blocks.{k}.weight'] = conv_w(ch, ch, 1).astype(np.float32)
        sd[f'controlnet_down_blocks.{k}.bias'] = randn(ch, std=0.1).astype(np.float32)
    mid_ch = BLOCK_OUT[-1]
    sd['controlnet_mid_block.weight'] = conv_w(mid_ch, mid_ch, 1).astype(np.float32)
    sd['controlnet_mid_block.bias'] = randn(mid_ch, std=0.1).astype(np.float32)
    sd['controlnet_cond_embedding.conv_out.weight'] = conv_w(
        BLOCK_OUT[0], COND_EMBED_OUT[-1], 3).astype(np.float32)
    sd['controlnet_cond_embedding.conv_out.bias'] = randn(
        BLOCK_OUT[0], std=0.1).astype(np.float32)


if __name__ == '__main__':
    main()
