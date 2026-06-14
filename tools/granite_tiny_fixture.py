#!/usr/bin/env python3
"""Generate tiny RANDOM IBM Granite 3.x parity fixtures for
tests/TestNeuralPretrained.pas (no network access: the models are randomly
initialized from pico configs, never downloaded).

Two fixtures, ~10 KB each, pinned in tests/fixtures/:

  tiny_granite.*: GraniteForCausalLM (model_type "granite") - the Llama block
      (RMSNorm + RoPE + SwiGLU, GQA, bias-free attention) plus the FOUR Granite
      scalar multipliers Llama lacks, all RE-RANDOMIZED to non-1.0 values and
      each one ASSERTED to move the HF logits, so a multiplier-blind import
      cannot pass:
        - embedding_multiplier (scale token embeddings after lookup),
        - residual_multiplier  (scale each sublayer output before the residual),
        - attention_multiplier (replaces 1/sqrt(head_dim) in the SDPA scale),
        - logits_scaling       (DIVIDES the final logits before softmax).
      GQA (1 kv head < 2 query heads) and a TIED lm_head are on.

  tiny_granitemoe.*: GraniteMoeForCausalLM (model_type "granitemoe") - the same
      four multipliers on the Mixtral-style sparse top-k MoE FFN (fused 3-D
      input_linear/output_linear/router slabs). shared_intermediate_size is 0
      (no parallel shared expert - that path is deferred in the importer).

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures). GraniteForCausalLM and
GraniteMoeForCausalLM are present in the venv (transformers 5.x).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/granite_tiny_fixture.py
writes tests/fixtures/tiny_granite{,moe}{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (>= the Granite release) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import (GraniteConfig, GraniteForCausalLM,
                          GraniteMoeConfig, GraniteMoeForCausalLM)

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1
D_MODEL = 8
D_FF = 12
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13

# Non-1.0 multipliers: each one must visibly move the logits or the asserts
# below fail (a multiplier-blind import would then silently pass).
EMB_MULT = 1.7
RES_MULT = 0.6
ATTN_MULT = 2.0
LOGITS_SCALING = 2.5


def common_cfg(model_type):
    return {
        'model_type': model_type,
        'hidden_size': D_MODEL,
        'intermediate_size': D_FF,
        'num_hidden_layers': N_LAYER,
        'num_attention_heads': N_HEAD,
        'num_key_value_heads': N_KV_HEAD,
        'vocab_size': VOCAB,
        'max_position_embeddings': MAX_POS,
        'rms_norm_eps': 1e-5,
        'rope_theta': 10000.0,
        'attention_bias': False,
        'tie_word_embeddings': True,
        'hidden_act': 'silu',
        'embedding_multiplier': EMB_MULT,
        'residual_multiplier': RES_MULT,
        'attention_multiplier': ATTN_MULT,
        'logits_scaling': LOGITS_SCALING,
    }


SEQUENCES = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]


def randomize_norms(model):
    # HF ones-inits every RMSNorm gain; re-randomize so the norm load paths
    # are not vacuous (matches the other fixtures). HF also std-0.02 inits
    # every Linear, which makes the attention scores ~0 and the softmax
    # ~uniform - so attention_multiplier would have a near-vacuous effect.
    # Re-randomize every parameter to an O(1) scale so the attention pattern
    # (and hence the score-scale multiplier) is genuinely exercised.
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 1.0)
        for layer in model.model.layers:
            layer.input_layernorm.weight.normal_(1.0, 0.5)
            layer.post_attention_layernorm.weight.normal_(1.0, 0.5)
        model.model.norm.weight.normal_(1.0, 0.5)


def build_and_dump(name, cfg_dict, model):
    model.double().eval()
    sd = {k: v.to(torch.float32).contiguous()
          for k, v in model.state_dict().items()}
    save_file(sd, f'tests/fixtures/tiny_{name}.safetensors')
    with open(f'tests/fixtures/tiny_{name}_config.json', 'w') as f:
        json.dump(cfg_dict, f, indent=1)
    with torch.no_grad():
        logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
                  for seq in SEQUENCES]
    with open(f'tests/fixtures/tiny_{name}_logits.json', 'w') as f:
        json.dump({'sequences': SEQUENCES, 'logits': logits}, f)
    print(f'wrote tiny_{name}.safetensors ({len(sd)} tensors) + config '
          f'+ logits ({N_SEQUENCES} sequences of {MAX_POS})')


def assert_multipliers_matter(name, ModelCls, ConfigCls, cfg_dict, model):
    """Each of the four multipliers reset to its no-op MUST change the
    logits, so a multiplier-blind import cannot reproduce them."""
    with torch.no_grad():
        base = model(input_ids=torch.tensor([SEQUENCES[0]])).logits

    def logits_with(patch):
        d = dict(cfg_dict)
        d.update(patch)
        m2 = ModelCls(ConfigCls(**d, attn_implementation='eager'))
        m2.load_state_dict(model.state_dict())
        m2.double().eval()
        with torch.no_grad():
            return m2(input_ids=torch.tensor([SEQUENCES[0]])).logits

    for field, noop in (('embedding_multiplier', 1.0),
                        ('residual_multiplier', 1.0),
                        ('attention_multiplier', 1.0 / (D_MODEL // N_HEAD) ** 0.5),
                        ('logits_scaling', 1.0)):
        eff = (base - logits_with({field: noop})).abs().max().item()
        assert eff > 1e-3, \
            f'{name}: {field} had no effect on the logits ({eff})'
        print(f'{name}: {field} effect: max |diff| = {eff:.4f}')


# ---------------- dense granite ----------------
torch.manual_seed(20260613)
granite_cfg = common_cfg('granite')
granite_cfg['architectures'] = ['GraniteForCausalLM']
gmodel = GraniteForCausalLM(GraniteConfig(**{k: v for k, v in granite_cfg.items()
                                             if k != 'architectures'},
                                          attn_implementation='eager'))
randomize_norms(gmodel)
build_and_dump('granite', granite_cfg, gmodel)
assert_multipliers_matter('granite', GraniteForCausalLM, GraniteConfig,
                          {k: v for k, v in granite_cfg.items()
                           if k != 'architectures'}, gmodel)

# ---------------- granitemoe ----------------
torch.manual_seed(20260614)
moe_cfg = common_cfg('granitemoe')
moe_cfg['architectures'] = ['GraniteMoeForCausalLM']
moe_cfg['num_local_experts'] = 3
moe_cfg['num_experts_per_tok'] = 2
moe_cfg['shared_intermediate_size'] = 0
mmodel = GraniteMoeForCausalLM(GraniteMoeConfig(
    **{k: v for k, v in moe_cfg.items() if k != 'architectures'},
    attn_implementation='eager'))
randomize_norms(mmodel)
build_and_dump('granitemoe', moe_cfg, mmodel)
assert_multipliers_matter('granitemoe', GraniteMoeForCausalLM, GraniteMoeConfig,
                          {k: v for k, v in moe_cfg.items()
                           if k != 'architectures'}, mmodel)
