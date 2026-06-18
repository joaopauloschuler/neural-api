#!/usr/bin/env python3
"""Generate a tiny RANDOM BLIP-2 Q-Former parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

The Q-Former is the genuinely NEW bridging module BLIP-2 contributes: a
small BERT-style transformer fed a fixed set of LEARNED query tokens that,
in each block, self-attend among themselves AND cross-attend into the
FROZEN ViT patch features (encoder_hidden_states), producing the query
embeddings later projected into the LLM token space.

This fixture isolates the parity-critical piece -- HF Blip2QFormerModel --
fed two FIXED inputs:
  - query_embeds  : the (num_query_tokens, hidden) query-token vectors
                    (in the full model these are the learned query_tokens;
                    here a fixed random O(1) tensor so the test is
                    self-contained and does not depend on the soft-prompt
                    bank reproduction);
  - encoder_hidden_states : the (num_patches, encoder_hidden_size) frozen
                    ViT features (a fixed random O(1) tensor).
The reference is HF's float64 last_hidden_state (the 32 query embeddings).

Q-Former block (use_qformer_text_input=False, the BLIP-2 query-only path):
  1. self-attention over the query tokens, post-LN:
       q := LN(q + attn.output.dense(MHA(q,q,q)))
  2. cross-attention into the ViT features, post-LN (only on layers where
       layer_idx % cross_attention_frequency == 0):
       q := LN(q + crossattention.output.dense(MHA(q, kv=ViT)))
  3. FFN via intermediate_query/output_query, post-LN:
       q := LN(q + output_query.dense(gelu(intermediate_query.dense(q))))
Plus the model-level embeddings.LayerNorm applied to query_embeds first.

cross_attention_frequency = 2 here (the real BLIP-2 default): layer 0
cross-attends into the ViT features, layer 1 is a pure self-attn + FFN
block -- so the parity test exercises BOTH the cross-attn and the
self-attn-only block paths.
encoder_hidden_size != hidden_size, so the cross-attn K/V projections are
genuinely RECTANGULAR (the importer's two-source wiring is tested).
hidden_act "gelu" is the EXACT erf form (the script asserts it differs from
the tanh approximation).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/blip2_qformer_tiny_fixture.py
writes tests/fixtures/tiny_blip2_qformer{.safetensors,_config.json,
_qformer.json}.  Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import Blip2QFormerConfig
from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM        # 8  (Q-Former hidden_size)
INTERMEDIATE = 16
ENC_HIDDEN = 12                    # ViT feature width (encoder_hidden_size)
NUM_QUERY = 4                      # query tokens (BLIP-2 uses 32)
NUM_PATCHES = 5                    # ViT patch tokens fed as encoder states

torch.manual_seed(20260615)

cfg_dict = {
    'model_type': 'blip_2_qformer',
    'num_query_tokens': NUM_QUERY,
    'hidden_size': D_MODEL,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'intermediate_size': INTERMEDIATE,
    'encoder_hidden_size': ENC_HIDDEN,
    'cross_attention_frequency': 2,
    'hidden_act': 'gelu',
    'layer_norm_eps': 1e-12,
    'use_qformer_text_input': False,
}
cfg = Blip2QFormerConfig(**cfg_dict)
model = Blip2QFormerModel(cfg)
# HF inits weights with std 0.02; at this pico width the attention scores
# are ~0 and softmax near-uniform. Boost q/k/v and the FFN so the
# self/cross-attention patterns are O(1)-structured and the gelu path
# genuinely matters (the ModernBERT fixture lesson: std-0.02 makes parity
# trivially pass on near-zero tensors).
with torch.no_grad():
    for layer in model.encoder.layer:
        a = layer.attention.attention
        a.query.weight.normal_(0.0, 0.8)
        a.key.weight.normal_(0.0, 0.8)
        a.value.weight.normal_(0.0, 0.6)
        if getattr(layer, 'has_cross_attention', False):
            c = layer.crossattention.attention
            c.query.weight.normal_(0.0, 0.8)
            c.key.weight.normal_(0.0, 0.8)
            c.value.weight.normal_(0.0, 0.6)
        layer.intermediate_query.dense.weight.normal_(0.0, 1.3)
model = model.double().eval()

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_blip2_qformer.safetensors')
with open('tests/fixtures/tiny_blip2_qformer_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

# Fixed inputs (O(1) scale, batch 1).
torch.manual_seed(424242)
query_embeds = (torch.randn(1, NUM_QUERY, D_MODEL, dtype=torch.float64))
encoder_hidden_states = (torch.randn(1, NUM_PATCHES, ENC_HIDDEN,
                                     dtype=torch.float64))

with torch.no_grad():
    out = model(query_embeds=query_embeds,
                encoder_hidden_states=encoder_hidden_states)
    hidden = out.last_hidden_state[0].tolist()      # (NUM_QUERY, D_MODEL)

with open('tests/fixtures/tiny_blip2_qformer_qformer.json', 'w') as f:
    json.dump({
        'query_embeds': query_embeds[0].tolist(),
        'encoder_hidden_states': encoder_hidden_states[0].tolist(),
        'hidden': hidden,
    }, f)
print(f'wrote tiny_blip2_qformer.safetensors ({len(sd)} tensors) '
      f'+ config + query embeddings ({NUM_QUERY} x {D_MODEL})')

# ---- a SECOND fixture: the full-blip2-checkpoint LAYOUT for
# BuildBlip2FromSafeTensors. Re-prefix the Q-Former tensors with "qformer."
# and add the two BLIP-2-specific top-level pieces: the learned query_tokens
# (1, NUM_QUERY, D_MODEL) and the language_projection nn.Linear
# (D_MODEL -> TEXT_HIDDEN). The ViT tower and the LLM are NOT included (v1
# scope -- BuildBlip2FromSafeTensors builds only the Q-Former + these two).
TEXT_HIDDEN = 10
torch.manual_seed(20260616)
full_sd = {'qformer.' + k: v for k, v in sd.items()}
query_tokens = torch.randn(1, NUM_QUERY, D_MODEL) * 0.7
proj_w = torch.randn(TEXT_HIDDEN, D_MODEL) * 0.5
proj_b = torch.randn(TEXT_HIDDEN) * 0.3
full_sd['query_tokens'] = query_tokens.contiguous()
full_sd['language_projection.weight'] = proj_w.contiguous()
full_sd['language_projection.bias'] = proj_b.contiguous()
save_file(full_sd, 'tests/fixtures/tiny_blip2_full.safetensors')
full_cfg = {
    'model_type': 'blip-2',
    'num_query_tokens': NUM_QUERY,
    'text_config': {'hidden_size': TEXT_HIDDEN, 'model_type': 't5'},
    'vision_config': {'hidden_size': ENC_HIDDEN, 'model_type': 'blip_2_vision_model'},
    'qformer_config': cfg_dict,
}
with open('tests/fixtures/tiny_blip2_full_config.json', 'w') as f:
    json.dump(full_cfg, f, indent=1)

# Reference for the full bridge: the projected query embeddings the LLM would
# be spliced with = language_projection(qformer(query_tokens, ViT)).
qt64 = query_tokens.to(torch.float64)
with torch.no_grad():
    qf = model(query_embeds=qt64,
               encoder_hidden_states=encoder_hidden_states).last_hidden_state
    projected = torch.nn.functional.linear(
        qf, proj_w.to(torch.float64), proj_b.to(torch.float64))[0]
with open('tests/fixtures/tiny_blip2_full_projected.json', 'w') as f:
    json.dump({
        'encoder_hidden_states': encoder_hidden_states[0].tolist(),
        'projected': projected.tolist(),  # (NUM_QUERY, TEXT_HIDDEN)
    }, f)
print(f'wrote tiny_blip2_full.safetensors ({len(full_sd)} tensors) '
      f'+ config + projected query embeddings ({NUM_QUERY} x {TEXT_HIDDEN})')

# ---- fixture self-checks: every quirk must be visible in the reference ----
import math
hmax = max(abs(v) for row in hidden for v in row)
assert hmax > 0.1, f'reference query embeddings are vacuously small ({hmax})'
print(f'reference query-embedding magnitude: max |h| = {hmax:.4f}')

with torch.no_grad():
    # 1. cross-attention must matter: perturbing the ViT features must move
    # the query embeddings (an importer missing cross-attn fails parity).
    enc2 = encoder_hidden_states.clone()
    enc2[0, 0] += 1.0
    out2 = model(query_embeds=query_embeds, encoder_hidden_states=enc2)
    cross_effect = (out.last_hidden_state -
                    out2.last_hidden_state).abs().max().item()
    assert cross_effect > 1e-2, \
        f'ViT features had no effect on query embeddings ({cross_effect})'
    print(f'cross-attention effect (perturb ViT feature): '
          f'max |diff| = {cross_effect:.4f}')

    # 2. the query tokens must genuinely drive the output (the self-attention
    # value path is exercised): zeroing query_embeds must move the result. The
    # output is cross-attention-dominated, so the precise query->query mixing is
    # numerically small; this asserts the queries matter at all (an importer
    # that ignored query_embeds, e.g. dropped the self-attn residual, fails).
    out_zero = model(query_embeds=torch.zeros_like(query_embeds),
                     encoder_hidden_states=encoder_hidden_states)
    query_effect = (out.last_hidden_state -
                    out_zero.last_hidden_state).abs().max().item()
    assert query_effect > 1e-1, \
        f'query_embeds had no effect on the output ({query_effect})'
    print(f'query-token effect (query_embeds vs zero): '
          f'max |diff| = {query_effect:.4f}')
    # The 32 query embeddings must be DISTINCT (not collapsed to one vector):
    # confirms the per-query self-attention produced per-row structure.
    rows = out.last_hidden_state[0]
    row_spread = (rows - rows.mean(dim=0)).abs().max().item()
    assert row_spread > 1e-1, \
        f'query embeddings collapsed to a single vector ({row_spread})'
    print(f'per-query output spread: max |row - mean| = {row_spread:.4f}')

    # 3. exact-vs-tanh GELU must differ in the reference.
    tanh_cfg = Blip2QFormerConfig(**{**cfg_dict,
                                     'hidden_act': 'gelu_pytorch_tanh'})
    tanh_model = Blip2QFormerModel(tanh_cfg)
    tanh_model.load_state_dict(model.state_dict())
    tanh_model = tanh_model.double().eval()
    tanh_out = tanh_model(query_embeds=query_embeds,
                          encoder_hidden_states=encoder_hidden_states)
    gelu_effect = (out.last_hidden_state -
                   tanh_out.last_hidden_state).abs().max().item()
    assert gelu_effect > 2.5e-5, \
        f'exact vs tanh GELU invisible in the fixture ({gelu_effect})'
    print(f'exact-vs-tanh GELU effect: max |diff| = {gelu_effect:.2e}')
