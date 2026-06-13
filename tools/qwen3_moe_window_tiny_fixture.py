#!/usr/bin/env python3
"""Generate a tiny RANDOM Qwen3-MoE SLIDING-WINDOW parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

This is the SLIDING-WINDOW sibling of tools/qwen3_moe_tiny_fixture.py. It pins
the use_sliding_window=true import path: HF Qwen3MoeConfig sets

    self.sliding_window = sliding_window if use_sliding_window else None

and the model bands EVERY layer's causal mask with that window when it is not
None (create_sliding_window_causal_mask; NO max_window_layers gating in this
revision). The Pascal importer mirrors this onto the Mistral convention
(Config.SlidingWindow > 0 => every layer local), threading the window into each
TNNetScaledDotProductAttention via pWindow / FStruct[2].

Config: a uniform all-MoE stack (decoder_sparse_step=1, mlp_only_layers=[]) over
2 layers, with use_sliding_window=true and a SMALL sliding_window (3) over a
length-16 sequence so the band genuinely masks tokens (a missing window would
attend full-context and change the logits). The script ASSERTS the window is
non-vacuous by re-running the float64 oracle with use_sliding_window disabled
(full attention) and checking the logits move.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/qwen3_moe_window_tiny_fixture.py
writes tests/fixtures/tiny_qwen3_moe_window{.safetensors,_config.json,
_logits.json}. Needs torch + transformers (with Qwen3-MoE) + safetensors.
"""
import json

import torch
import torch.nn.functional as F
import transformers.models.qwen3_moe.modeling_qwen3_moe as Q3M
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM


# Newer transformers fuses the experts into 3-D gate_up_proj/down_proj slabs and
# dispatches to a grouped kernel that does not support float64. Install a plain
# float64-safe per-expert SwiGLU eager loop as the experts forward.
def eager_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(top_k_index,
                            num_classes=self.num_experts).permute(2, 1, 0)
    for e in range(self.num_experts):
        pos, tok = torch.where(expert_mask[e])
        if tok.numel() == 0:
            continue
        cur = hidden_states[tok]
        gate, up = F.linear(cur, self.gate_up_proj[e]).chunk(2, dim=-1)
        cur = self.act_fn(gate) * up
        cur = F.linear(cur, self.down_proj[e])
        cur = cur * top_k_weights[tok, pos, None]
        final.index_add_(0, tok, cur.to(final.dtype))
    return final


Q3M.Qwen3MoeExperts.forward = eager_experts_forward

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1
HEAD_DIM = 6           # != D_MODEL // N_HEAD = 4: decoupled head_dim
D_MODEL = 8
D_FF = 12
D_MOE_FF = 5
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
N_EXPERTS = 4
TOP_K = 2
SLIDING_WINDOW = 3     # tiny window over the length-16 sequence

torch.manual_seed(20260613)
qwen3_moe_cfg = {
    'architectures': ['Qwen3MoeForCausalLM'],
    'model_type': 'qwen3_moe',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'moe_intermediate_size': D_MOE_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'head_dim': HEAD_DIM,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-5,
    'rope_theta': 10000.0,
    'attention_bias': False,
    'use_sliding_window': True,
    'sliding_window': SLIDING_WINDOW,
    'max_window_layers': 0,
    'tie_word_embeddings': False,
    'hidden_act': 'silu',
    'num_experts': N_EXPERTS,
    'num_experts_per_tok': TOP_K,
    'norm_topk_prob': True,
    'decoder_sparse_step': 1,
    'mlp_only_layers': [],
}
model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**qwen3_moe_cfg,
                                           attn_implementation='eager'))
assert N_HEAD * HEAD_DIM != D_MODEL, 'head_dim decoupling not exercised'
assert model.config.sliding_window == SLIDING_WINDOW, \
    'sliding_window did not survive config (use_sliding_window gating)'

# O(1)-scale re-randomization (the pico recipe) so the experts genuinely move
# the logits and the window band has something to mask. NON-ONE q/k norm gains.
with torch.no_grad():
    for p in model.parameters():
        if p.dim() >= 2:
            p.normal_(0.0, 0.5)
    for layer in model.model.layers:
        layer.self_attn.q_norm.weight.normal_(1.0, 0.5)
        layer.self_attn.k_norm.weight.normal_(1.0, 0.5)
        if isinstance(layer.mlp, Q3M.Qwen3MoeSparseMoeBlock):
            layer.mlp.gate.weight.normal_(0.0, 0.3)

model = model.double().eval()
state = model.state_dict()


def perexpert_state_dict(sd):
    """Rewrite the fused (transformers>=5) Qwen3-MoE experts back to the on-disk
    per-expert layout mlp.experts.{e}.gate_proj/up_proj/down_proj."""
    out = {}
    for k, v in sd.items():
        if k.endswith('mlp.experts.gate_up_proj'):
            base = k[:-len('mlp.experts.gate_up_proj')]
            for e in range(v.shape[0]):
                gate, up = v[e].chunk(2, dim=0)
                out[f'{base}mlp.experts.{e}.gate_proj.weight'] = \
                    gate.contiguous().clone()
                out[f'{base}mlp.experts.{e}.up_proj.weight'] = \
                    up.contiguous().clone()
        elif k.endswith('mlp.experts.down_proj'):
            base = k[:-len('mlp.experts.down_proj')]
            for e in range(v.shape[0]):
                out[f'{base}mlp.experts.{e}.down_proj.weight'] = \
                    v[e].contiguous().clone()
        else:
            out[k] = v.clone()
    return out


sd = {k: v.to(torch.float32).contiguous()
      for k, v in perexpert_state_dict(state).items()}
save_file(sd, 'tests/fixtures/tiny_qwen3_moe_window.safetensors')
with open('tests/fixtures/tiny_qwen3_moe_window_config.json', 'w') as f:
    json.dump(qwen3_moe_cfg, f, indent=1)
sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_qwen3_moe_window_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_qwen3_moe_window.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS}); '
      f'sliding_window={SLIDING_WINDOW}')

# --- sliding-window non-vacuity assertion ---------------------------------
# Re-run with full attention (use_sliding_window disabled) and confirm the
# window genuinely changes the logits, so a missing/zero window FAILS parity.
full_cfg = dict(qwen3_moe_cfg)
full_cfg['use_sliding_window'] = False
full = Qwen3MoeForCausalLM(Qwen3MoeConfig(**full_cfg,
                                          attn_implementation='eager'))
full.load_state_dict(model.state_dict())
full = full.double().eval()
assert full.config.sliding_window is None, 'full-attention oracle still windowed'
with torch.no_grad():
    lg = model(input_ids=torch.tensor([sequences[0]])).logits
    lf = full(input_ids=torch.tensor([sequences[0]])).logits
window_effect = (lg - lf).abs().max().item()
assert window_effect > 1e-3, \
    f'sliding window had no effect on the logits ({window_effect}) - the ' \
    f'fixture does not pin Qwen3-MoE windowing (try a smaller window)'
print(f'sliding-window effect on logits: max |diff| = {window_effect:.4f}')
