#!/usr/bin/env python3
# Coded by Claude (AI).
#
# Self-contained float64 oracle + tiny safetensors fixture for the F5-TTS
# flow-matching DiT velocity field. F5-TTS (SWivid/F5-TTS) is NOT in
# transformers, so this re-implements the official architecture
# (model/backbones/dit.py) in pure numpy float64 -- the same stance as the
# InternLM2 / Pyannote tiny fixtures.
#
# The F5 DiT velocity field v_theta(x_t, cond, text, t):
#   text_embed  = char-embed(text) + ConvNeXt-V2-1D blocks            (S, text_dim)
#   x           = InputEmbedding( cat([x_t, cond, text_embed], -1) )  (S, dim)
#                  = Linear(2*n_mel + text_dim -> dim) then conv-pos residual
#   t_emb       = SiLU-MLP( sinusoidal(t * 1000) )                    (dim,)
#   for blk in blocks:  x = DiTBlock(x, t_emb)  (adaLN-zero, RoPE self-attn)
#   x           = adaLN-norm-out(x, t_emb)
#   v           = proj_out(x)                                          (S, n_mel)
#
# Run:  /home/bpsa/x/bin/python tools/f5_tiny_fixture.py
#
# Emits (all tiny, committed under tests/fixtures/):
#   tiny_f5.safetensors    float32 weights
#   tiny_f5_config.json    F5 config (model_type "f5tts")
#   tiny_f5_expected.txt   header + flattened inputs + float64 velocity oracle

import json
import struct
import numpy as np

rng = np.random.default_rng(20260626)

# ---- tiny config (kept genuinely small; ~12KB committed) --------------------
DIM        = 12     # transformer width
DEPTH      = 2      # number of DiT blocks
HEADS      = 2      # attention heads (head_dim = 6, even for RoPE)
FF_MULT    = 2      # FFN mult (ff_dim = FF_MULT*DIM)
N_MEL      = 6      # mel channels
TEXT_DIM   = 8      # text embedding width (conv_text_dim)
VOCAB      = 9      # character vocab (incl filler id 0)
CONV_LAYERS= 2      # ConvNeXt-V2 text blocks
CONV_MULT  = 2      # ConvNeXt inner mult
S          = 5      # sequence length (frames)
EPS        = 1e-6
ROPE_THETA = 10000.0

HEAD_DIM = DIM // HEADS
FF_DIM   = FF_MULT * DIM


def randn(*shape, scale=0.45):
    return (rng.standard_normal(shape) * scale).astype(np.float64)


W = {}

# text embedding: char table (+1 filler row at index 0 already inside VOCAB)
W['text_embed.weight']   = randn(VOCAB, TEXT_DIM)
# ConvNeXt-V2 1D blocks over the text sequence (depthwise k=7 + GRN)
for i in range(CONV_LAYERS):
    p = f'text_conv.{i}.'
    W[p + 'dwconv.weight'] = randn(TEXT_DIM, 7)            # depthwise (C, k)
    W[p + 'dwconv.bias']   = randn(TEXT_DIM)
    W[p + 'pwconv1.weight']= randn(CONV_MULT*TEXT_DIM, TEXT_DIM)
    W[p + 'pwconv1.bias']  = randn(CONV_MULT*TEXT_DIM)
    W[p + 'grn.gamma']     = randn(CONV_MULT*TEXT_DIM, scale=0.2)
    W[p + 'grn.beta']      = randn(CONV_MULT*TEXT_DIM, scale=0.2)
    W[p + 'pwconv2.weight']= randn(TEXT_DIM, CONV_MULT*TEXT_DIM)
    W[p + 'pwconv2.bias']  = randn(TEXT_DIM)

# input embedding: Linear(2*n_mel + text_dim -> dim) + conv positional residual
W['input_embed.proj.weight'] = randn(DIM, 2*N_MEL + TEXT_DIM)
W['input_embed.proj.bias']   = randn(DIM)
# conv positional embedding: two depthwise-grouped 1D convs (k=31 in F5; small here)
W['input_embed.conv.0.weight'] = randn(DIM, 7)   # depthwise (C, k)
W['input_embed.conv.0.bias']   = randn(DIM)
W['input_embed.conv.1.weight'] = randn(DIM, 7)
W['input_embed.conv.1.bias']   = randn(DIM)

# time embedding: sinusoidal(dim) -> Linear(dim->dim) -> SiLU -> Linear(dim->dim)
W['time_embed.mlp.0.weight'] = randn(DIM, DIM)
W['time_embed.mlp.0.bias']   = randn(DIM)
W['time_embed.mlp.2.weight'] = randn(DIM, DIM)
W['time_embed.mlp.2.bias']   = randn(DIM)

# DiT blocks (adaLN-zero): adaLN(dim->6*dim), qkv(dim->3*dim), attn out(dim->dim),
# ff.0(dim->ff_dim), ff.2(ff_dim->dim)
for i in range(DEPTH):
    p = f'blocks.{i}.'
    W[p + 'adaln.weight'] = randn(6*DIM, DIM, scale=0.3)
    W[p + 'adaln.bias']   = randn(6*DIM, scale=0.3)
    W[p + 'attn.qkv.weight'] = randn(3*DIM, DIM)
    W[p + 'attn.qkv.bias']   = randn(3*DIM)
    W[p + 'attn.proj.weight']= randn(DIM, DIM)
    W[p + 'attn.proj.bias']  = randn(DIM)
    W[p + 'ff.0.weight'] = randn(FF_DIM, DIM)
    W[p + 'ff.0.bias']   = randn(FF_DIM)
    W[p + 'ff.2.weight'] = randn(DIM, FF_DIM)
    W[p + 'ff.2.bias']   = randn(DIM)

# final adaLN norm-out (shift,scale) + proj_out(dim->n_mel)
W['norm_out.weight'] = randn(2*DIM, DIM, scale=0.3)
W['norm_out.bias']   = randn(2*DIM, scale=0.3)
W['proj_out.weight'] = randn(N_MEL, DIM)
W['proj_out.bias']   = randn(N_MEL)


# ---- numpy float64 forward oracle -------------------------------------------
def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * x**3)))


from math import erf as _erf
_verf = np.vectorize(_erf)


def gelu_erf(x):
    # exact erf GELU, matches neural-api TNNetGELUErf
    return 0.5 * x * (1.0 + _verf(x / np.sqrt(2.0)))


def layernorm(x, w, b, eps=EPS):
    # per-row (last axis) LayerNorm, biased variance
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu)**2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def layernorm_noaffine(x, eps=EPS):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu)**2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def depthwise_conv1d(x, w, b, pad):
    # x: (S, C), w: (C, k) depthwise, 'same' padding with zeros
    S_, C = x.shape
    k = w.shape[1]
    xp = np.zeros((S_ + 2*pad, C))
    xp[pad:pad+S_] = x
    out = np.zeros((S_, C))
    for c in range(C):
        for j in range(k):
            out[:, c] += xp[j:j+S_, c] * w[c, j]
        out[:, c] += b[c]
    return out


def grn(x, gamma, beta, eps=1e-6):
    # global response norm over sequence axis (matches neural-api TNNetGRN
    # exactly: eps INSIDE the sqrt, plain mean divisor, residual add).
    gx = np.sqrt((x**2).sum(0, keepdims=True) + eps)    # L2 over S -> (1, C)
    nx = gx / gx.mean(-1, keepdims=True)                # (1, C)
    return gamma * (x * nx) + beta + x


def convnext_block(x, p):
    inp = x
    h = depthwise_conv1d(x, W[p+'dwconv.weight'], W[p+'dwconv.bias'], pad=3)
    h = layernorm_noaffine(h)   # TNNetTokenLayerNorm is affine-free
    h = h @ W[p+'pwconv1.weight'].T + W[p+'pwconv1.bias']
    h = gelu_erf(h)
    h = grn(h, W[p+'grn.gamma'], W[p+'grn.beta'])
    h = h @ W[p+'pwconv2.weight'].T + W[p+'pwconv2.bias']
    return inp + h


def rope_cos_sin(S_, head_dim, theta=ROPE_THETA):
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, half) / half))   # (half,)
    pos = np.arange(S_)[:, None] * freqs[None, :]          # (S, half)
    # interleave-free (GPT-NeoX style, matching neural-api SDPA): first/second half
    cos = np.concatenate([np.cos(pos), np.cos(pos)], -1)   # (S, head_dim)
    sin = np.concatenate([np.sin(pos), np.sin(pos)], -1)
    return cos, sin


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], -1)


def apply_rope(x, cos, sin):
    # x: (S, head_dim)
    return x * cos + rotate_half(x) * sin


def attention(x):
    # x: (S, DIM). Per-head RoPE SDPA, no causal mask, no qk-norm.
    S_, _ = x.shape
    qkv = x @ W_attn_qkv.T + W_attn_qkv_b
    q = qkv[:, :DIM]; k = qkv[:, DIM:2*DIM]; v = qkv[:, 2*DIM:]
    cos, sin = rope_cos_sin(S_, HEAD_DIM)
    out = np.zeros((S_, DIM))
    for h in range(HEADS):
        sl = slice(h*HEAD_DIM, (h+1)*HEAD_DIM)
        qh = apply_rope(q[:, sl], cos, sin)
        kh = apply_rope(k[:, sl], cos, sin)
        vh = v[:, sl]
        scores = qh @ kh.T / np.sqrt(HEAD_DIM)
        scores -= scores.max(-1, keepdims=True)
        w = np.exp(scores); w /= w.sum(-1, keepdims=True)
        out[:, sl] = w @ vh
    return out @ W_attn_proj.T + W_attn_proj_b


def dit_block(x, c, p):
    global W_attn_qkv, W_attn_qkv_b, W_attn_proj, W_attn_proj_b
    W_attn_qkv   = W[p+'attn.qkv.weight'];  W_attn_qkv_b  = W[p+'attn.qkv.bias']
    W_attn_proj  = W[p+'attn.proj.weight']; W_attn_proj_b = W[p+'attn.proj.bias']
    # adaLN: SiLU(c) -> 6*dim, chunk [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f]
    mod = silu(c) @ W[p+'adaln.weight'].T + W[p+'adaln.bias']    # (6*dim,)
    sa, ca, ga, sf, cf, gf = np.split(mod, 6)
    # attention sub-block
    h = layernorm_noaffine(x) * (1 + ca) + sa
    h = attention(h)
    x = x + ga * h
    # feed-forward sub-block
    h = layernorm_noaffine(x) * (1 + cf) + sf
    h = h @ W[p+'ff.0.weight'].T + W[p+'ff.0.bias']
    h = gelu_tanh(h)
    h = h @ W[p+'ff.2.weight'].T + W[p+'ff.2.bias']
    x = x + gf * h
    return x


def time_embedding(t):
    # sinusoidal (DDPM order: sin|cos) of scaled time, then MLP
    tt = t * 1000.0
    half = DIM // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
    args = tt * freqs
    emb = np.concatenate([np.sin(args), np.cos(args)])          # (DIM,)
    emb = emb @ W['time_embed.mlp.0.weight'].T + W['time_embed.mlp.0.bias']
    emb = silu(emb)
    emb = emb @ W['time_embed.mlp.2.weight'].T + W['time_embed.mlp.2.bias']
    return emb


def text_embedding(text_ids):
    e = W['text_embed.weight'][text_ids]                       # (S, TEXT_DIM)
    for i in range(CONV_LAYERS):
        e = convnext_block(e, f'text_conv.{i}.')
    return e


def input_embedding(x_noised, x_cond, text_emb):
    cat = np.concatenate([x_noised, x_cond, text_emb], -1)     # (S, 2*N_MEL+TEXT_DIM)
    x = cat @ W['input_embed.proj.weight'].T + W['input_embed.proj.bias']
    # conv positional residual: two depthwise convs with gelu in between
    h = depthwise_conv1d(x, W['input_embed.conv.0.weight'],
                         W['input_embed.conv.0.bias'], pad=3)
    h = gelu_erf(h)
    h = depthwise_conv1d(h, W['input_embed.conv.1.weight'],
                         W['input_embed.conv.1.bias'], pad=3)
    return x + h


def forward(x_noised, x_cond, text_ids, t):
    # NOTE: t is the continuous flow-matching time in [0,1], scaled by 1000
    # before the sinusoidal table (the F5 / FlowMatching convention). The
    # importer / test feeds t*1000 as the scalar time input.
    text_emb = text_embedding(text_ids)
    x = input_embedding(x_noised, x_cond, text_emb)
    c = time_embedding(t)
    for i in range(DEPTH):
        x = dit_block(x, c, f'blocks.{i}.')
    # final adaLN norm-out
    mod = silu(c) @ W['norm_out.weight'].T + W['norm_out.bias']
    shift, scale = np.split(mod, 2)
    x = layernorm_noaffine(x) * (1 + scale) + shift
    return x @ W['proj_out.weight'].T + W['proj_out.bias']      # (S, N_MEL)


# ---- deterministic inputs + oracle ------------------------------------------
x_noised = randn(S, N_MEL, scale=1.0)
x_cond   = randn(S, N_MEL, scale=1.0)
text_ids = np.array([(3*i + 2) % VOCAB for i in range(S)], dtype=np.int64)
t_val    = 0.37

velocity = forward(x_noised, x_cond, text_ids, t_val)          # (S, N_MEL)


# ---- emit safetensors (hand-rolled, float32) --------------------------------
def save_safetensors(path, tensors):
    header = {}
    blobs = []
    offset = 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr.astype(np.float32))
        b = a.tobytes()
        header[name] = {'dtype': 'F32', 'shape': list(a.shape),
                        'data_offsets': [offset, offset + len(b)]}
        blobs.append(b)
        offset += len(b)
    hjson = json.dumps(header, separators=(',', ':')).encode('utf-8')
    pad = (8 - len(hjson) % 8) % 8
    hjson += b' ' * pad
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(hjson)))
        fh.write(hjson)
        for b in blobs:
            fh.write(b)


save_safetensors('tests/fixtures/tiny_f5.safetensors', W)

config = {
    'model_type': 'f5tts',
    'dim': DIM,
    'depth': DEPTH,
    'heads': HEADS,
    'ff_mult': FF_MULT,
    'n_mel_channels': N_MEL,
    'text_dim': TEXT_DIM,
    'text_num_embeds': VOCAB,
    'conv_layers': CONV_LAYERS,
    'conv_mult': CONV_MULT,
    'rope_theta': ROPE_THETA,
    'layer_norm_eps': EPS,
}
with open('tests/fixtures/tiny_f5_config.json', 'w') as fh:
    json.dump(config, fh, indent=2)

with open('tests/fixtures/tiny_f5_expected.txt', 'w') as fh:
    fh.write(f'{S} {N_MEL} {VOCAB} {t_val:.8f}\n')
    fh.write(' '.join(str(int(i)) for i in text_ids) + '\n')
    fh.write(' '.join(f'{v:.10f}' for v in x_noised.flatten()) + '\n')
    fh.write(' '.join(f'{v:.10f}' for v in x_cond.flatten()) + '\n')
    for r in range(S):
        fh.write(' '.join(f'{velocity[r, c]:.10f}' for c in range(N_MEL)) + '\n')

print('wrote tiny_f5.safetensors, tiny_f5_config.json, tiny_f5_expected.txt')
print('velocity[0]:', velocity[0])
print('|velocity| max:', np.abs(velocity).max())
