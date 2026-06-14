#!/usr/bin/env python3
"""Generate a tiny HAND-BUILT InternLM2 parity fixture for
tests/TestNeuralPretrained.pas.

InternLM2's modeling code (modeling_internlm2.py) is NOT bundled in
transformers - loading internlm/internlm2_5-* needs trust_remote_code and a
network download, neither available offline. So this fixture is constructed
by hand: a random pico checkpoint is written under the EXACT InternLM2 HF
tensor names and a config.json with model_type "internlm2", and the
reference next-token logits are computed by a self-contained numpy float64
forward pass that reimplements InternLM2ForCausalLM (the oracle convention
of the other hand-built fixtures).

The ONE genuinely InternLM2-specific piece exercised here is the FUSED
attention.wqkv packing, which is NEITHER contiguous Q|K|V thirds NOR
GPT-NeoX per-head interleaving:

  wqkv is reshaped [num_kv_heads, (q_per_kv + 2), head_dim, hidden] and,
  per KV group, concatenates its q_per_kv query slices, then its single K,
  then its single V slice. Unpacking to standard separate W_q / W_k / W_v
  needs the group-interleaved row reorder (all Q heads of all groups, then
  all K, then all V), after which the usual Llama rotate_half q/k de-permute
  for RoPE rides the existing path.

GQA is exercised (4 query heads sharing 2 kv heads -> q_per_kv = 2), as is
the SwiGLU FFN (w1=gate, w3=up, w2=down) and the standard attention_norm /
ffn_norm pre-norms, untied output head (the InternLM2 default).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/internlm2_tiny_fixture.py
writes tests/fixtures/tiny_internlm2{.safetensors,_config.json,_logits.json}.
Needs numpy + safetensors only (no torch, no transformers).
"""
import json

import numpy as np
from safetensors.numpy import save_file

N_LAYER = 2
N_HEAD = 4
N_KV_HEAD = 2
D_MODEL = 16            # head_dim = 4
HEAD_DIM = D_MODEL // N_HEAD
Q_PER_KV = N_HEAD // N_KV_HEAD     # 2
D_FF = 24
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
RMS_EPS = 1e-5
ROPE_THETA = 10000.0

rng = np.random.default_rng(20260613)


def randn(*shape):
    # O(1)-scale weights so every path (RoPE, GQA grouping, the wqkv
    # repack) moves the logits well above the 1e-4 parity gate; a std-0.02
    # HF init would leave a pico net almost linear (the ModernBERT lesson).
    return (rng.standard_normal(shape) * 0.8).astype(np.float64)


# ---- hand-built weights (float64 reference; saved as float32) ----
W = {}
W['model.tok_embeddings.weight'] = randn(VOCAB, D_MODEL)
W['model.norm.weight'] = randn(D_MODEL) * 0.3 + 1.0
W['output.weight'] = randn(VOCAB, D_MODEL)
for b in range(N_LAYER):
    p = f'model.layers.{b}.'
    W[p + 'attention_norm.weight'] = randn(D_MODEL) * 0.3 + 1.0
    W[p + 'ffn_norm.weight'] = randn(D_MODEL) * 0.3 + 1.0
    # wqkv: [num_kv_heads * (q_per_kv + 2) * head_dim, hidden], group-packed.
    wqkv_rows = N_KV_HEAD * (Q_PER_KV + 2) * HEAD_DIM
    W[p + 'attention.wqkv.weight'] = randn(wqkv_rows, D_MODEL)
    W[p + 'attention.wo.weight'] = randn(D_MODEL, N_HEAD * HEAD_DIM)
    W[p + 'feed_forward.w1.weight'] = randn(D_FF, D_MODEL)   # gate
    W[p + 'feed_forward.w3.weight'] = randn(D_FF, D_MODEL)   # up
    W[p + 'feed_forward.w2.weight'] = randn(D_MODEL, D_FF)   # down


def split_wqkv(wqkv):
    """InternLM2 wqkv -> (Wq, Wk, Wv) in standard HF row order.

    wqkv is [num_kv_heads*(q_per_kv+2)*head_dim, hidden]; view it as
    [num_kv_heads, q_per_kv+2, head_dim, hidden] and gather, per group, the
    q_per_kv query slices into Wq, the single K slice into Wk, the single V
    slice into Wv. The query heads come out group-major
    (kv0's q heads, then kv1's, ...), exactly the [num_heads*head_dim] order
    HF's reshape produces."""
    qkv = wqkv.reshape(N_KV_HEAD, Q_PER_KV + 2, HEAD_DIM, D_MODEL)
    q = qkv[:, :Q_PER_KV, :, :].reshape(N_HEAD * HEAD_DIM, D_MODEL)
    k = qkv[:, -2, :, :].reshape(N_KV_HEAD * HEAD_DIM, D_MODEL)
    v = qkv[:, -1, :, :].reshape(N_KV_HEAD * HEAD_DIM, D_MODEL)
    return q, k, v


def rmsnorm(x, w):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + RMS_EPS) * w


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def rope_cos_sin(seqlen):
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2,
                                               dtype=np.float64) / HEAD_DIM))
    pos = np.arange(seqlen, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)                 # [seq, head_dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)   # [seq, head_dim]
    return np.cos(emb), np.sin(emb)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def forward(seq):
    seqlen = len(seq)
    h = W['model.tok_embeddings.weight'][np.array(seq)]   # [seq, d]
    cos, sin = rope_cos_sin(seqlen)
    causal = np.triu(np.full((seqlen, seqlen), -1e30), k=1)
    for b in range(N_LAYER):
        p = f'model.layers.{b}.'
        Wq, Wk, Wv = split_wqkv(W[p + 'attention.wqkv.weight'])
        # ---- attention ----
        x = rmsnorm(h, W[p + 'attention_norm.weight'])
        q = (x @ Wq.T).reshape(seqlen, N_HEAD, HEAD_DIM)
        k = (x @ Wk.T).reshape(seqlen, N_KV_HEAD, HEAD_DIM)
        v = (x @ Wv.T).reshape(seqlen, N_KV_HEAD, HEAD_DIM)
        # RoPE on q,k (HF rotate_half layout), per head.
        c = cos[:, None, :]
        s = sin[:, None, :]
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        # GQA: repeat each kv head q_per_kv times.
        k = np.repeat(k, Q_PER_KV, axis=1)            # [seq, n_head, hd]
        v = np.repeat(v, Q_PER_KV, axis=1)
        attn_out = np.zeros((seqlen, N_HEAD, HEAD_DIM), dtype=np.float64)
        scale = 1.0 / np.sqrt(HEAD_DIM)
        for hd in range(N_HEAD):
            scores = (q[:, hd, :] @ k[:, hd, :].T) * scale + causal
            w = softmax(scores, axis=-1)
            attn_out[:, hd, :] = w @ v[:, hd, :]
        attn_flat = attn_out.reshape(seqlen, N_HEAD * HEAD_DIM)
        h = h + attn_flat @ W[p + 'attention.wo.weight'].T
        # ---- SwiGLU FFN ----
        x = rmsnorm(h, W[p + 'ffn_norm.weight'])
        gate = x @ W[p + 'feed_forward.w1.weight'].T
        up = x @ W[p + 'feed_forward.w3.weight'].T
        silu = gate / (1.0 + np.exp(-gate))
        h = h + (silu * up) @ W[p + 'feed_forward.w2.weight'].T
    h = rmsnorm(h, W['model.norm.weight'])
    return h @ W['output.weight'].T                   # [seq, vocab]


internlm2_cfg = {
    'architectures': ['InternLM2ForCausalLM'],
    'model_type': 'internlm2',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': RMS_EPS,
    'rope_theta': ROPE_THETA,
    'bias': False,
    'tie_word_embeddings': False,
    'hidden_act': 'silu',
    'bos_token_id': 1,
    'eos_token_id': 2,
    'pad_token_id': 0,
}

sd = {k: v.astype(np.float32) for k, v in W.items()}
save_file(sd, 'tests/fixtures/tiny_internlm2.safetensors')
with open('tests/fixtures/tiny_internlm2_config.json', 'w') as f:
    json.dump(internlm2_cfg, f, indent=1)

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
logits = [forward(seq).tolist() for seq in sequences]
with open('tests/fixtures/tiny_internlm2_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_internlm2.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')

# Sanity: a NAIVE contiguous-thirds split (the WRONG unpacking) must give
# DIFFERENT q/k/v than the group-interleaved split, otherwise the fixture
# would not actually test the repack.
w0 = W['model.layers.0.attention.wqkv.weight']
Wq, Wk, Wv = split_wqkv(w0)
qw = N_HEAD * HEAD_DIM
naive_q = w0[:qw]
diff = np.abs(Wq - naive_q).max()
assert diff > 1e-6, \
    f'group-interleaved split equals contiguous-thirds split ({diff})'
print(f'wqkv repack vs contiguous-thirds: max |diff| = {diff:.4f}')
