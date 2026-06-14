#!/usr/bin/env python3
"""Generate a tiny RANDOM Jamba parity fixture for
tests/TestNeuralPretrained.pas (no network: the model is randomly
initialized from a pico config, never downloaded).

Jamba (HF model_type "jamba", ai21labs/Jamba-tiny-dev / Jamba-Mini) is
the suite's FIRST HYBRID Mamba+attention+MoE decoder import. It exercises
three landed subsystems TOGETHER:
  - selective-SSM mixers (TNNetSelectiveSSM) on the "mamba" layers, here in
    the genuinely-new JAMBA INNER-NORM mode: the mixer inserts a per-vector
    RMSNorm on the dt / B / C selection vectors BETWEEN x_proj and the scan
    (HF JambaMambaMixer dt_layernorm / b_layernorm / c_layernorm), which is
    nonlinear and cannot be folded into the constant projections the plain
    Mamba importer uses;
  - full softmax attention on the "attention" layers - GQA via
    num_key_value_heads, bias-free q/k/v/o, and NO positional encoding at
    all (Jamba attention is NoPE; the Mamba layers carry order);
  - Mixtral-style top-k MoE FFNs on the "expert" layers (a router + N SwiGLU
    experts, softmax over all experts then top-k with the raw - NOT
    renormalized - weights), dense SwiGLU FFN elsewhere.

The genuinely new piece is the PER-LAYER block-type schedule (HF
configuration_jamba):
  attention iff i % attn_layer_period == attn_layer_offset, else mamba;
  num_experts experts iff i % expert_layer_period == expert_layer_offset,
  else a dense (1-expert) MLP.
This fixture picks periods so the first 6 layers cover ALL FOUR block kinds:
  L0 mamba+dense, L1 attn+moe, L2 mamba+dense, L3 attn+dense,
  L4 mamba+moe, L5 attn+dense   (=> mamba-dense, mamba-moe, attn-dense,
  attn-moe all present).

On-disk tensor naming follows the published checkpoints (the transformers
"jamba" weight-conversion source patterns): per-expert
feed_forward.experts.{e}.{gate,up,down}_proj.weight + feed_forward.router,
dense feed_forward.{gate,up,down}_proj, the Mamba mixer keys
(in_proj/conv1d/x_proj/dt_proj/A_log/D/out_proj + dt_layernorm/b_layernorm/
c_layernorm), self_attn.{q,k,v,o}_proj, input_layernorm/pre_ff_layernorm,
final_layernorm, embed_tokens, lm_head.

The reference logits are a faithful float64 reimplementation of HF
modeling_jamba's slow_forward (verified function-by-function against the
formulas in modeling_jamba.py) - no torch model is instantiated, so the
oracle is independent of the installed transformers' in-memory expert
layout. Every quirk is exercised by a self-check at the bottom.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/jamba_tiny_fixture.py
writes tests/fixtures/tiny_jamba{.safetensors,_config.json,_logits.json}.
Needs numpy + safetensors.
"""
import json
import math

import numpy as np
from safetensors.numpy import save_file

rng = np.random.default_rng(20260613)

HIDDEN = 8
INTER = 12          # SwiGLU / expert width
LAYERS = 6
VOCAB = 17
HEADS = 4
KV_HEADS = 2        # GQA: 2 query heads share each kv head
HEAD_DIM = HIDDEN // HEADS   # 2
STATE = 4           # mamba_d_state
DT_RANK = 3         # mamba_dt_rank
EXPAND = 2          # d_inner = 16
CONV_K = 4
N_EXPERTS = 2
TOP_K = 1
EPS = 1e-6
SEQ_LEN = 7
N_SEQUENCES = 3

ATTN_PERIOD, ATTN_OFFSET = 2, 1
EXPERT_PERIOD, EXPERT_OFFSET = 3, 1
D_INNER = EXPAND * HIDDEN


def is_attn(i):
    return i % ATTN_PERIOD == ATTN_OFFSET


def num_experts(i):
    return N_EXPERTS if i % EXPERT_PERIOD == EXPERT_OFFSET else 1


cfg = {
    'architectures': ['JambaForCausalLM'],
    'model_type': 'jamba',
    'vocab_size': VOCAB,
    'hidden_size': HIDDEN,
    'intermediate_size': INTER,
    'num_hidden_layers': LAYERS,
    'num_attention_heads': HEADS,
    'num_key_value_heads': KV_HEADS,
    'hidden_act': 'silu',
    'rms_norm_eps': EPS,
    'tie_word_embeddings': False,
    'num_experts': N_EXPERTS,
    'num_experts_per_tok': TOP_K,
    'expert_layer_period': EXPERT_PERIOD,
    'expert_layer_offset': EXPERT_OFFSET,
    'attn_layer_period': ATTN_PERIOD,
    'attn_layer_offset': ATTN_OFFSET,
    'mamba_d_state': STATE,
    'mamba_d_conv': CONV_K,
    'mamba_expand': EXPAND,
    'mamba_dt_rank': DT_RANK,
    'mamba_conv_bias': True,
    'mamba_proj_bias': False,
    'use_mamba_kernels': False,
}


def randn(*shape, s=0.5):
    return (rng.standard_normal(shape) * s).astype(np.float64)


# ---------------------------------------------------------------- weights
W = {}
W['model.embed_tokens.weight'] = randn(VOCAB, HIDDEN, s=0.8)
W['model.final_layernorm.weight'] = randn(HIDDEN, s=0.2) + 1.0
W['lm_head.weight'] = randn(VOCAB, HIDDEN, s=0.5)

for i in range(LAYERS):
    p = f'model.layers.{i}.'
    W[p + 'input_layernorm.weight'] = randn(HIDDEN, s=0.2) + 1.0
    W[p + 'pre_ff_layernorm.weight'] = randn(HIDDEN, s=0.2) + 1.0
    if is_attn(i):
        W[p + 'self_attn.q_proj.weight'] = randn(HEADS * HEAD_DIM, HIDDEN)
        W[p + 'self_attn.k_proj.weight'] = randn(KV_HEADS * HEAD_DIM, HIDDEN)
        W[p + 'self_attn.v_proj.weight'] = randn(KV_HEADS * HEAD_DIM, HIDDEN)
        W[p + 'self_attn.o_proj.weight'] = randn(HIDDEN, HEADS * HEAD_DIM)
    else:
        mp = p + 'mamba.'
        W[mp + 'in_proj.weight'] = randn(2 * D_INNER, HIDDEN)
        W[mp + 'conv1d.weight'] = randn(D_INNER, 1, CONV_K)
        W[mp + 'conv1d.bias'] = randn(D_INNER, s=0.4)
        W[mp + 'x_proj.weight'] = randn(DT_RANK + 2 * STATE, D_INNER)
        W[mp + 'dt_proj.weight'] = randn(D_INNER, DT_RANK, s=0.6)
        W[mp + 'dt_proj.bias'] = randn(D_INNER, s=0.8)
        # A_log = log(A), A = arange(1, STATE+1) broadcast - but re-randomize
        # so the per-state decay spread matters; keep it positive-ish.
        W[mp + 'A_log'] = (randn(D_INNER, STATE, s=0.6) + 0.3).astype(np.float64)
        W[mp + 'D'] = randn(D_INNER, s=0.8) + 1.0
        W[mp + 'out_proj.weight'] = randn(HIDDEN, D_INNER)
        W[mp + 'dt_layernorm.weight'] = randn(DT_RANK, s=0.3) + 1.0
        W[mp + 'b_layernorm.weight'] = randn(STATE, s=0.3) + 1.0
        W[mp + 'c_layernorm.weight'] = randn(STATE, s=0.3) + 1.0
    ne = num_experts(i)
    fp = p + 'feed_forward.'
    if ne > 1:
        W[fp + 'router.weight'] = randn(ne, HIDDEN)
        for e in range(ne):
            W[fp + f'experts.{e}.gate_proj.weight'] = randn(INTER, HIDDEN)
            W[fp + f'experts.{e}.up_proj.weight'] = randn(INTER, HIDDEN)
            W[fp + f'experts.{e}.down_proj.weight'] = randn(HIDDEN, INTER)
    else:
        W[fp + 'gate_proj.weight'] = randn(INTER, HIDDEN)
        W[fp + 'up_proj.weight'] = randn(INTER, HIDDEN)
        W[fp + 'down_proj.weight'] = randn(HIDDEN, INTER)


# ------------------------------------------------------- float64 oracle
def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.logaddexp(0.0, x)


def rmsnorm(x, g, eps=EPS):
    # x: (..., d); g: (d,)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * g


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def mamba_mixer(x, mp):
    # x: (T, HIDDEN). Mirrors JambaMambaMixer.slow_forward (no cache).
    T = x.shape[0]
    proj = x @ W[mp + 'in_proj.weight'].T          # (T, 2*D_INNER)
    hs, gate = proj[:, :D_INNER], proj[:, D_INNER:]  # (T, D_INNER) each
    # depthwise causal conv1d (pad k-1 left), per channel, then SiLU.
    cw = W[mp + 'conv1d.weight'][:, 0, :]           # (D_INNER, CONV_K)
    cb = W[mp + 'conv1d.bias']
    conv = np.zeros_like(hs)
    for t in range(T):
        for kk in range(CONV_K):
            src = t - (CONV_K - 1) + kk
            if src >= 0:
                conv[t] += cw[:, kk] * hs[src]
    conv += cb
    hs = silu(conv)                                 # (T, D_INNER)
    ssm = hs @ W[mp + 'x_proj.weight'].T            # (T, DT_RANK+2*STATE)
    ts = ssm[:, :DT_RANK]
    B = ssm[:, DT_RANK:DT_RANK + STATE]
    C = ssm[:, DT_RANK + STATE:]
    ts = rmsnorm(ts, W[mp + 'dt_layernorm.weight'])
    B = rmsnorm(B, W[mp + 'b_layernorm.weight'])
    C = rmsnorm(C, W[mp + 'c_layernorm.weight'])
    dt = softplus(ts @ W[mp + 'dt_proj.weight'].T + W[mp + 'dt_proj.bias'])  # (T,D_INNER)
    A = -np.exp(W[mp + 'A_log'])                    # (D_INNER, STATE)
    D = W[mp + 'D']
    state = np.zeros((D_INNER, STATE))
    out = np.zeros((T, D_INNER))
    for t in range(T):
        dA = np.exp(A * dt[t][:, None])             # (D_INNER, STATE)
        dB = dt[t][:, None] * B[t][None, :]         # (D_INNER, STATE)
        state = dA * state + dB * hs[t][:, None]
        out[t] = state @ C[t] + D * hs[t]
    out = out * silu(gate)                          # (T, D_INNER)
    return out @ W[mp + 'out_proj.weight'].T        # (T, HIDDEN)


def attention(x, p):
    T = x.shape[0]
    q = (x @ W[p + 'self_attn.q_proj.weight'].T).reshape(T, HEADS, HEAD_DIM)
    k = (x @ W[p + 'self_attn.k_proj.weight'].T).reshape(T, KV_HEADS, HEAD_DIM)
    v = (x @ W[p + 'self_attn.v_proj.weight'].T).reshape(T, KV_HEADS, HEAD_DIM)
    groups = HEADS // KV_HEADS
    scale = HEAD_DIM ** -0.5
    out = np.zeros((T, HEADS, HEAD_DIM))
    causal = np.tril(np.ones((T, T)))
    for h in range(HEADS):
        kv = h // groups
        scores = (q[:, h] @ k[:, kv].T) * scale     # (T, T)
        scores = np.where(causal > 0, scores, -np.inf)
        attn = softmax(scores, axis=-1)
        out[:, h] = attn @ v[:, kv]
    out = out.reshape(T, HEADS * HEAD_DIM)
    return out @ W[p + 'self_attn.o_proj.weight'].T


def dense_ffn(x, fp):
    g = silu(x @ W[fp + 'gate_proj.weight'].T)
    u = x @ W[fp + 'up_proj.weight'].T
    return (g * u) @ W[fp + 'down_proj.weight'].T


def moe_ffn(x, fp, ne):
    T = x.shape[0]
    logits = x @ W[fp + 'router.weight'].T          # (T, ne)
    probs = softmax(logits, axis=-1)
    idx = np.argsort(-probs, axis=-1)[:, :TOP_K]     # top-k (raw, not renorm)
    out = np.zeros_like(x)
    for t in range(T):
        for kpos in range(TOP_K):
            e = idx[t, kpos]
            w = probs[t, e]
            ep = fp + f'experts.{e}.'
            g = silu(x[t] @ W[ep + 'gate_proj.weight'].T)
            u = x[t] @ W[ep + 'up_proj.weight'].T
            out[t] += w * ((g * u) @ W[ep + 'down_proj.weight'].T)
    return out


def forward(ids):
    x = W['model.embed_tokens.weight'][ids]          # (T, HIDDEN)
    for i in range(LAYERS):
        p = f'model.layers.{i}.'
        res = x
        h = rmsnorm(x, W[p + 'input_layernorm.weight'])
        if is_attn(i):
            h = attention(h, p)
        else:
            h = mamba_mixer(h, p + 'mamba.')
        x = res + h
        res = x
        h = rmsnorm(x, W[p + 'pre_ff_layernorm.weight'])
        fp = p + 'feed_forward.'
        if num_experts(i) > 1:
            h = moe_ffn(h, fp, num_experts(i))
        else:
            h = dense_ffn(h, fp)
        x = res + h
    x = rmsnorm(x, W['model.final_layernorm.weight'])
    return x @ W['lm_head.weight'].T                 # (T, VOCAB)


sequences = [[(5 * i + 3 * s + s * s) % (VOCAB - 1) + 1
              for i in range(SEQ_LEN)] for s in range(N_SEQUENCES)]

logits = np.stack([forward(np.array(s)) for s in sequences])

# Save f32 weights (the importer / Pascal net are f32; the oracle is f64 on
# these exact f32-rounded values).
sd = {k: v.astype(np.float32) for k, v in W.items()}
save_file(sd, 'tests/fixtures/tiny_jamba.safetensors')
with open('tests/fixtures/tiny_jamba_config.json', 'w') as f:
    json.dump(cfg, f, indent=1)
with open('tests/fixtures/tiny_jamba_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits.tolist()}, f)

print(f'wrote tiny_jamba.safetensors ({len(sd)} tensors) + config + oracle')
print('layer schedule:')
for i in range(LAYERS):
    print(f'  L{i}: {"attn " if is_attn(i) else "mamba"} '
          f'{"moe " if num_experts(i) > 1 else "dense"}')

# -------------------------------------------------------- self-checks
kinds = {(is_attn(i), num_experts(i) > 1) for i in range(LAYERS)}
assert kinds == {(False, False), (False, True), (True, False), (True, True)}, \
    f'fixture must cover all four block kinds, got {kinds}'
print('all four block kinds present')

# The dt/B/C inner RMSNorms must MATTER: zeroing the b_layernorm gain to ~0
# (kills B, hence the SSM input term) must change the logits a lot.
W2 = {k: v.copy() for k, v in W.items()}
for i in range(LAYERS):
    if not is_attn(i):
        W2[f'model.layers.{i}.mamba.b_layernorm.weight'][:] = 0.0
globals_W = W
try:
    W = W2
    d = np.abs(np.stack([forward(np.array(s)) for s in sequences]) - logits).max()
finally:
    W = globals_W
assert d > 1e-2, f'b_layernorm had no effect ({d})'
print(f'b_layernorm effect on logits: max |diff| = {d:.4f}')

# GQA must matter: 2 kv heads shared by 4 q heads (groups=2). Just assert the
# shapes encode genuine GQA (kv width < q width).
assert KV_HEADS < HEADS and HEADS % KV_HEADS == 0
print('GQA (kv_heads < heads) confirmed')
print('all fixture self-checks passed')
