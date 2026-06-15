#!/usr/bin/env python3
"""Generate a tiny RANDOM MiniCPM parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

MiniCPM (openbmb/MiniCPM-1.2B / 2B-sft/dpo, HF MiniCPMForCausalLM) is a plain
Llama backbone (RMSNorm + RoPE + SwiGLU, GQA, tied embeddings, bias-free) plus
OpenBMB's muP-style depth/width rescaling. The distinguishing pieces, all
constant FOLDS at load (no new layer types - the Granite fold machinery is
reused by the importer):

  scale_emb                       -> multiplies the token embeddings AFTER
                                     lookup (folded into the embedding rows;
                                     the tied LM head reads the UNSCALED rows).
  scale_depth / sqrt(num_layers)  -> rescales EVERY residual branch (each
                                     attention AND each MLP sublayer output) -
                                     a PER-SUBLAYER residual scale, folded into
                                     the o_proj and down_proj rows of each block.
  hidden_size / dim_model_base    -> DIVIDES the final logits before softmax
                                     (folded as 1/scale into the tied LM head).

MiniCPM's modeling code ships via trust_remote_code (no `MiniCPMConfig` /
`MiniCPMForCausalLM` class in transformers 5.x), and this box has no network,
so the reference forward is implemented HERE in float64 from the documented
MiniCPM math (the HF Llama block + the three muP folds above). The forward is
tiny and fully specified; it IS the oracle the Pascal importer must match.

Each of the three muP knobs is set to a non-trivial value and ASSERTED to move
the logits (resetting it to its no-op must change the reference logits), so a
fold-blind import cannot silently pass.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/minicpm_tiny_fixture.py
writes tests/fixtures/tiny_minicpm{.safetensors,_config.json,_logits.json}
(and reuses the shared tiny SentencePiece .model tokenizer fixture).
Needs torch + safetensors + sentencepiece.
"""
import json
import math
import os

import torch
from safetensors.torch import save_file

FIXDIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1
D_MODEL = 8
HEAD_DIM = D_MODEL // N_HEAD          # 4
D_FF = 12
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
RMS_EPS = 1e-5
ROPE_THETA = 10000.0

# muP knobs (non-trivial; each must visibly move the logits or the asserts
# below fail). dim_model_base < hidden_size so the logits divide by > 1.
SCALE_EMB = 1.7
SCALE_DEPTH = 1.4
DIM_MODEL_BASE = 5                    # logits divide by D_MODEL / DIM_MODEL_BASE

SEQUENCES = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]


def rand_weights(seed):
    """A full set of HF-Llama-named random weights for the pico MiniCPM.
    O(1)-scale (not std-0.02) so the attention pattern and every fold are
    genuinely exercised (HF's std-0.02 init makes the scores ~0 / vacuous)."""
    g = torch.Generator().manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    w = {}
    w['model.embed_tokens.weight'] = rn(VOCAB, D_MODEL)
    for L in range(N_LAYER):
        p = f'model.layers.{L}.'
        w[p + 'self_attn.q_proj.weight'] = rn(N_HEAD * HEAD_DIM, D_MODEL)
        w[p + 'self_attn.k_proj.weight'] = rn(N_KV_HEAD * HEAD_DIM, D_MODEL)
        w[p + 'self_attn.v_proj.weight'] = rn(N_KV_HEAD * HEAD_DIM, D_MODEL)
        w[p + 'self_attn.o_proj.weight'] = rn(D_MODEL, N_HEAD * HEAD_DIM)
        w[p + 'mlp.gate_proj.weight'] = rn(D_FF, D_MODEL)
        w[p + 'mlp.up_proj.weight'] = rn(D_FF, D_MODEL)
        w[p + 'mlp.down_proj.weight'] = rn(D_MODEL, D_FF)
        # RMSNorm gains: re-randomized around 1.0 so the norm load is not vacuous.
        w[p + 'input_layernorm.weight'] = 1.0 + 0.5 * rn(D_MODEL)
        w[p + 'post_attention_layernorm.weight'] = 1.0 + 0.5 * rn(D_MODEL)
    w['model.norm.weight'] = 1.0 + 0.5 * rn(D_MODEL)
    return w


def rmsnorm(x, gain, eps=RMS_EPS):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * gain


def rope_cos_sin(seq_len):
    # HF Llama rotate_half RoPE: freqs over the FIRST half = SECOND half (the
    # [theta_0..theta_{d/2-1}, theta_0..] layout), positions 0..seq_len-1.
    half = HEAD_DIM // 2
    inv_freq = ROPE_THETA ** (-torch.arange(0, half, dtype=torch.float64) / half)
    pos = torch.arange(seq_len, dtype=torch.float64)
    freqs = torch.outer(pos, inv_freq)                 # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)            # [T, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    # x: [T, n_head, head_dim]; cos/sin: [T, head_dim]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return x * cos + rotate_half(x) * sin


def forward(w, ids, scale_emb=SCALE_EMB, scale_depth=SCALE_DEPTH,
            dim_model_base=DIM_MODEL_BASE):
    """Float64 reference MiniCPM forward. Returns logits [T, VOCAB]."""
    T = len(ids)
    res_scale = scale_depth / math.sqrt(N_LAYER)       # per-sublayer residual
    idt = torch.tensor(ids, dtype=torch.long)
    # scale_emb multiplies the embeddings after lookup.
    h = w['model.embed_tokens.weight'][idt] * scale_emb   # [T, D]
    cos, sin = rope_cos_sin(T)
    causal = torch.triu(torch.full((T, T), float('-inf'), dtype=torch.float64),
                        diagonal=1)
    for L in range(N_LAYER):
        p = f'model.layers.{L}.'
        # ---- attention sublayer ----
        x = rmsnorm(h, w[p + 'input_layernorm.weight'])
        q = (x @ w[p + 'self_attn.q_proj.weight'].T).view(T, N_HEAD, HEAD_DIM)
        k = (x @ w[p + 'self_attn.k_proj.weight'].T).view(T, N_KV_HEAD, HEAD_DIM)
        v = (x @ w[p + 'self_attn.v_proj.weight'].T).view(T, N_KV_HEAD, HEAD_DIM)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        rep = N_HEAD // N_KV_HEAD                       # GQA expand
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        # scores [n_head, T, T] with the standard 1/sqrt(head_dim) scale.
        qh = q.permute(1, 0, 2)
        kh = k.permute(1, 0, 2)
        vh = v.permute(1, 0, 2)
        scores = (qh @ kh.transpose(-1, -2)) / math.sqrt(HEAD_DIM) + causal
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ vh                                 # [n_head, T, head_dim]
        ctx = ctx.permute(1, 0, 2).reshape(T, N_HEAD * HEAD_DIM)
        attn_out = ctx @ w[p + 'self_attn.o_proj.weight'].T
        h = h + attn_out * res_scale                    # per-sublayer residual
        # ---- MLP sublayer ----
        x = rmsnorm(h, w[p + 'post_attention_layernorm.weight'])
        gate = x @ w[p + 'mlp.gate_proj.weight'].T
        up = x @ w[p + 'mlp.up_proj.weight'].T
        act = torch.nn.functional.silu(gate) * up
        mlp_out = act @ w[p + 'mlp.down_proj.weight'].T
        h = h + mlp_out * res_scale                     # per-sublayer residual
    h = rmsnorm(h, w['model.norm.weight'])
    # Tied LM head, then DIVIDE logits by hidden_size / dim_model_base.
    logits = h @ w['model.embed_tokens.weight'].T
    logits = logits / (D_MODEL / dim_model_base)
    return logits


def build_config():
    return {
        'architectures': ['MiniCPMForCausalLM'],
        'model_type': 'minicpm',
        'hidden_size': D_MODEL,
        'intermediate_size': D_FF,
        'num_hidden_layers': N_LAYER,
        'num_attention_heads': N_HEAD,
        'num_key_value_heads': N_KV_HEAD,
        'vocab_size': VOCAB,
        'max_position_embeddings': MAX_POS,
        'rms_norm_eps': RMS_EPS,
        'rope_theta': ROPE_THETA,
        'attention_bias': False,
        'tie_word_embeddings': True,
        'hidden_act': 'silu',
        'scale_emb': SCALE_EMB,
        'scale_depth': SCALE_DEPTH,
        'dim_model_base': DIM_MODEL_BASE,
    }


def assert_folds_matter(w):
    """Resetting each muP knob to its no-op MUST change the logits, so a
    fold-blind import cannot reproduce the reference."""
    seq = SEQUENCES[0]
    base = forward(w, seq)
    variants = {
        'scale_emb': dict(scale_emb=1.0),
        'scale_depth': dict(scale_depth=math.sqrt(N_LAYER)),  # res_scale -> 1
        'dim_model_base': dict(dim_model_base=D_MODEL),       # logits / 1
    }
    for name, patch in variants.items():
        alt = forward(w, seq, **{**dict(scale_emb=SCALE_EMB,
                                        scale_depth=SCALE_DEPTH,
                                        dim_model_base=DIM_MODEL_BASE), **patch})
        eff = (base - alt).abs().max().item()
        assert eff > 1e-3, f'{name} had no effect on the logits ({eff})'
        print(f'minicpm: {name} effect: max |diff| = {eff:.4f}')


def ensure_spm():
    """Reuse the shared pico SentencePiece .model tokenizer fixture (built by
    tools/make_pico_spm_fixture.py); MiniCPM ships a SentencePiece tokenizer."""
    spm_path = os.path.join(FIXDIR, 'tiny_spm.model')
    if os.path.exists(spm_path):
        return spm_path
    try:
        import sentencepiece  # noqa: F401
        import subprocess
        import sys
        subprocess.check_call([sys.executable,
            os.path.join(os.path.dirname(__file__), 'make_pico_spm_fixture.py')])
    except Exception as e:        # tokenizer is optional for the parity tests
        print(f'(skipping spm fixture: {e})')
    return spm_path


def main():
    torch.manual_seed(20260614)
    w = rand_weights(20260614)
    assert_folds_matter(w)

    sd = {k: v.to(torch.float32).contiguous() for k, v in w.items()}
    save_file(sd, os.path.join(FIXDIR, 'tiny_minicpm.safetensors'))
    with open(os.path.join(FIXDIR, 'tiny_minicpm_config.json'), 'w') as f:
        json.dump(build_config(), f, indent=1)
    logits = [forward(w, ids=seq).tolist() for seq in SEQUENCES]
    with open(os.path.join(FIXDIR, 'tiny_minicpm_logits.json'), 'w') as f:
        json.dump({'sequences': SEQUENCES, 'logits': logits}, f)
    print(f'wrote tiny_minicpm.safetensors ({len(sd)} tensors) + config '
          f'+ logits ({N_SEQUENCES} sequences of {MAX_POS})')
    ensure_spm()


if __name__ == '__main__':
    main()
