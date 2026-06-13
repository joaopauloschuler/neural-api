#!/usr/bin/env python3
"""Generate tiny RANDOM Falcon parity fixtures for
tests/TestNeuralPretrained.pas (no network access: the models are randomly
initialized from pico configs, never downloaded).

Falcon is the GPT-NeoX cousin (fused query_key_value, RoPE, parallel
attention+MLP residual) with two Falcon-specific twists this fixture pins:

  - the FUSED MULTI-QUERY / GQA query_key_value slab, in BOTH layouts;
  - the SINGLE-vs-DUAL parallel-residual LayerNorm switch.

Pinned in tests/fixtures/ (~20 KB total), two genuinely different models:

  tiny_falcon_mq.safetensors + tiny_falcon_mq_config.json +
  tiny_falcon_mq_logits.json:
      FalconForCausalLM, multi_query=True, new_decoder_architecture=False
      (the falcon-7b / falcon-rw branch):
        - ONE shared K head + ONE shared V head fanned out across all query
          heads (query_key_value rows = num_heads*head_dim + 2*head_dim,
          HF view(num_heads + 2, head_dim));
        - SINGLE input_layernorm feeding BOTH the attention and MLP branches
          (parallel_attn=True): x := x + Attn(ln(x)) + MLP(ln(x));
        - full-head RoPE (Llama rotate_half), bias-free Linears,
          exact-erf GELU MLP, tied lm_head.

  tiny_falcon_nda.safetensors + tiny_falcon_nda_config.json +
  tiny_falcon_nda_logits.json:
      FalconForCausalLM, new_decoder_architecture=True, num_kv_heads=2
      (the falcon-40b branch):
        - num_kv_heads GQA groups, each (num_heads/num_kv_heads query heads +
          1 K + 1 V) INTERLEAVED per group
          (HF view(-1, num_heads//num_kv_heads + 2, head_dim));
        - TWO separate LayerNorms ln_attn (attention) and ln_mlp (MLP), both
          reading the block input (num_ln_in_parallel_attn=2).

intermediate (ffn_hidden_size) is deliberately NOT 4*hidden to catch a
hardcoded 4x anywhere. Reference logits come from HF transformers in float64
(the oracle convention of the committed fixtures). The script ASSERTS that
RoPE genuinely matters (zeroing the rotary effect changes the logits far
above the parity gate) so a missing/wrong rotary FAILS the test.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/falcon_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import FalconConfig, FalconForCausalLM

MAX_POS = 16
N_SEQUENCES = 3


def boost_qk(model, n_head, n_kv, head_dim, group_first):
    """HF inits with std 0.02; at pico width the attention scores are ~0 and
    softmax is near-uniform, making RoPE numerically invisible. Boost the q/k
    rows of the fused query_key_value so the scores are O(1) and the rotary
    path genuinely matters. Layout differs per branch (see _split_heads)."""
    with torch.no_grad():
        for layer in model.transformer.h:
            w = layer.self_attention.query_key_value.weight  # [qkv_out, hid]
            hid = w.shape[1]
            if group_first:  # new arch: view(n_kv, group + 2, head_dim)
                grp = n_head // n_kv
                w3 = w.view(n_kv, grp + 2, head_dim, hid)
                w3[:, :grp].normal_(0.0, 0.5)   # query heads
                w3[:, grp].normal_(0.0, 0.5)    # K head
            else:  # multi_query: view(n_head + 2, head_dim)
                w3 = w.view(n_head + 2, head_dim, hid)
                w3[:n_head].normal_(0.0, 0.5)   # query heads
                w3[n_head].normal_(0.0, 0.5)    # the single K head


def logits_of(m, seqs):
    with torch.no_grad():
        return [m(input_ids=torch.tensor([seq])).logits[0] for seq in seqs]


def write_fixture(tag, cfg_dict, n_head, n_kv, head_dim, group_first):
    model = FalconForCausalLM(FalconConfig(**cfg_dict,
                                           attn_implementation='eager'))
    boost_qk(model, n_head, n_kv, head_dim, group_first)
    model = model.double().eval()

    sd = {k: v.to(torch.float32).contiguous()
          for k, v in model.state_dict().items()}
    save_file(sd, f'tests/fixtures/tiny_falcon_{tag}.safetensors')
    with open(f'tests/fixtures/tiny_falcon_{tag}_config.json', 'w') as f:
        json.dump(cfg_dict, f, indent=1)

    vocab = cfg_dict['vocab_size']
    sequences = [[(7 * i + 3 * s + s * s) % vocab for i in range(MAX_POS)]
                 for s in range(N_SEQUENCES)]
    ref = logits_of(model, sequences)
    with open(f'tests/fixtures/tiny_falcon_{tag}_logits.json', 'w') as f:
        json.dump({'sequences': sequences,
                   'logits': [l.tolist() for l in ref]}, f)
    print(f'wrote tiny_falcon_{tag}.safetensors ({len(sd)} tensors) + '
          f'config + logits ({N_SEQUENCES}x{MAX_POS})')

    # RoPE must be non-vacuous: a model with theta blown up to ~infinity (no
    # rotation) must produce different logits, otherwise the fixture would
    # not pin RoPE at all.
    no_rope_cfg = dict(cfg_dict)
    no_rope_cfg['rope_theta'] = 1e30
    no_rope = FalconForCausalLM(FalconConfig(**no_rope_cfg,
                                             attn_implementation='eager'))
    no_rope.load_state_dict(model.state_dict())
    no_rope = no_rope.double().eval()
    rope_effect = max((a - b).abs().max().item()
                      for a, b in zip(ref, logits_of(no_rope, sequences)))
    assert rope_effect > 1e-3, \
        f'[{tag}] RoPE had no effect on the logits ({rope_effect})'
    print(f'  RoPE effect on logits: max |diff| = {rope_effect:.4f}')

    print(f'  tensor names ({tag}):')
    for k in sorted(sd):
        print('   ', k, list(sd[k].shape))
    return model


torch.manual_seed(20260613)

# ---- multi_query branch (falcon-7b / falcon-rw): single shared K/V, one LN.
N_HEAD = 4
HEAD_DIM = 8
D_MODEL = N_HEAD * HEAD_DIM  # 32
mq_cfg = {
    'architectures': ['FalconForCausalLM'],
    'model_type': 'falcon',
    'hidden_size': D_MODEL,
    'ffn_hidden_size': 48,        # deliberately NOT 4*hidden (=128)
    'num_hidden_layers': 2,
    'num_attention_heads': N_HEAD,
    'vocab_size': 11,
    'max_position_embeddings': MAX_POS,
    'layer_norm_epsilon': 1e-5,
    'rope_theta': 10000.0,
    'multi_query': True,
    'new_decoder_architecture': False,
    'parallel_attn': True,
    'bias': False,
    'alibi': False,
    'tie_word_embeddings': True,
}
write_fixture('mq', mq_cfg, N_HEAD, 1, HEAD_DIM, group_first=False)

# ---- new_decoder_architecture branch (falcon-40b): GQA groups, two LNs.
N_HEAD2 = 4
HEAD_DIM2 = 8
N_KV = 2
D_MODEL2 = N_HEAD2 * HEAD_DIM2  # 32
nda_cfg = {
    'architectures': ['FalconForCausalLM'],
    'model_type': 'falcon',
    'hidden_size': D_MODEL2,
    'ffn_hidden_size': 48,        # deliberately NOT 4*hidden
    'num_hidden_layers': 2,
    'num_attention_heads': N_HEAD2,
    'num_kv_heads': N_KV,
    'vocab_size': 11,
    'max_position_embeddings': MAX_POS,
    'layer_norm_epsilon': 1e-5,
    'rope_theta': 10000.0,
    'new_decoder_architecture': True,
    'bias': False,
    'alibi': False,
    'tie_word_embeddings': False,
}
write_fixture('nda', nda_cfg, N_HEAD2, N_KV, HEAD_DIM2, group_first=True)

print('done.')
