#!/usr/bin/env python3
"""Generate a tiny RANDOM Qwen3.5/3.6-MoE (model_type qwen3_5_moe) parity
fixture for tests/TestNeuralPretrained.pas (no network access needed: the
model is randomly initialized from a pico config, never downloaded).

One fixture, ~40 KB, pinned in tests/fixtures/:

  tiny_qwen3_5_moe.*: Qwen3_5MoeForCausalLM - the hybrid DeltaNet+attention
      stack of tiny_qwen3_5 (4 layers, one (L L L F) pattern) with EVERY MLP
      replaced by the Qwen3.5-MoE sparse block:
      - router mlp.gate.weight [E, H] -> per-token softmax over ALL experts
        -> hard top-k (num_experts_per_tok=2) with the survivors ALWAYS
        renormalized (norm_topk_prob hard-coded on in HF);
      - experts stored FUSED 3-D (the on-disk layout):
        mlp.experts.gate_up_proj [E, 2I, H] (per expert gate rows 0..I-1
        then up rows I..2I-1, chunk(2) order) and
        mlp.experts.down_proj [E, H, I], moe_intermediate_size=5;
      - PLUS an always-on shared SwiGLU expert (width
        shared_expert_intermediate_size=6, DISTINCT from the routed width so
        a swap cannot cancel) scaled by sigmoid(mlp.shared_expert_gate(x))
        - a [1, hidden] linear -> ONE sigmoid scalar per token.

  The safetensors keeps the FLAT text-only "model." prefix (the dense
  fixture pins the multimodal "model.language_model." + visual/mtp-skip
  path); the config still nests the text fields under "text_config"
  (top-level model_type "qwen3_5_moe", inner "qwen3_5_moe_text") so the
  nesting parse is pinned by both fixtures.

  The script ASSERTS that zeroing the shared_expert_gate weights (sigmoid
  -> 0.5) MOVES the logits, so the parity test genuinely pins the gated
  shared expert.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/qwen3_5_moe_tiny_fixture.py
writes tests/fixtures/tiny_qwen3_5_moe{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (with qwen3_5_moe) + safetensors.
"""
import json

import torch
import torch.nn.functional as F
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as QM
from safetensors.torch import save_file
from transformers import Qwen3_5MoeForCausalLM
from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig


# transformers >= 5 dispatches the fused experts to a grouped-mm kernel that
# does not support float64 - it would raise on the float64 oracle. Install a
# plain, float64-safe eager loop (the per-expert SwiGLU, the modeling-source
# semantics) as the experts forward for the whole run.
def eager_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(top_k_index,
                            num_classes=self.num_experts).permute(2, 1, 0)
    for e in range(self.num_experts):
        top_k_pos, token_idx = torch.where(expert_mask[e])
        if token_idx.numel() == 0:
            continue
        cur = hidden_states[token_idx]
        gate, up = F.linear(cur, self.gate_up_proj[e]).chunk(2, dim=-1)
        cur = self.act_fn(gate) * up
        cur = F.linear(cur, self.down_proj[e])
        cur = cur * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, cur.to(final.dtype))
    return final


QM.Qwen3_5MoeExperts.forward = eager_experts_forward

N_LAYER = 4
LAYER_TYPES = ['linear_attention', 'linear_attention', 'linear_attention',
               'full_attention']
N_HEAD = 2
N_KV_HEAD = 1
HEAD_DIM = 8           # rotary_dim = 8 * 0.25 = 2 (even, partial rotary)
D_MODEL = 8
D_FF = 12              # dense intermediate_size (unused: every FFN is MoE)
D_MOE_FF = 5           # moe_intermediate_size (routed expert width)
D_SHARED_FF = 6        # shared_expert_intermediate_size (DISTINCT width)
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
N_EXPERTS = 4
TOP_K = 2
LIN_K_HEADS = 2
LIN_V_HEADS = 4
LIN_K_DIM = 4
LIN_V_DIM = 4
CONV_K = 4

torch.manual_seed(20260719)
text_cfg = {
    'model_type': 'qwen3_5_moe_text',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'moe_intermediate_size': D_MOE_FF,
    'shared_expert_intermediate_size': D_SHARED_FF,
    'num_experts': N_EXPERTS,
    'num_experts_per_tok': TOP_K,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'head_dim': HEAD_DIM,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-6,
    'attention_bias': False,
    'tie_word_embeddings': False,
    'hidden_act': 'silu',
    'layer_types': LAYER_TYPES,
    'linear_num_key_heads': LIN_K_HEADS,
    'linear_num_value_heads': LIN_V_HEADS,
    'linear_key_head_dim': LIN_K_DIM,
    'linear_value_head_dim': LIN_V_DIM,
    'linear_conv_kernel_dim': CONV_K,
    'rope_parameters': {
        'rope_type': 'default',
        'rope_theta': 10000.0,
        'partial_rotary_factor': 0.25,
        'mrope_section': [1, 0, 0],  # rotary_dim/2 = 1 frequency pair
        'mrope_interleaved': True,   # a NO-OP for 1-D text positions
    },
}
config = Qwen3_5MoeTextConfig(
    **{k: v for k, v in text_cfg.items() if k != 'model_type'})
model = Qwen3_5MoeForCausalLM(config)
model.config._attn_implementation = 'eager'

# Re-randomize to O(1) scale (see qwen3_5_tiny_fixture.py for the why per
# family); a flat-ish router keeps the top-2 probs comparable so the
# always-on renorm genuinely matters.
with torch.no_grad():
    for p in model.parameters():
        if p.dim() >= 2:
            p.normal_(0.0, 0.2)
    for layer in model.model.layers:
        layer.input_layernorm.weight.normal_(0.0, 0.3)
        layer.post_attention_layernorm.weight.normal_(0.0, 0.3)
        if hasattr(layer, 'self_attn'):
            layer.self_attn.q_norm.weight.normal_(0.0, 0.3)
            layer.self_attn.k_norm.weight.normal_(0.0, 0.3)
        if hasattr(layer, 'linear_attn'):
            layer.linear_attn.norm.weight.normal_(1.0, 0.3)
            layer.linear_attn.A_log.uniform_(-2.0, 1.0)
            layer.linear_attn.dt_bias.normal_(0.0, 0.5)
        layer.mlp.gate.weight.normal_(0.0, 0.3)
        layer.mlp.experts.gate_up_proj.normal_(0.0, 0.5)
        layer.mlp.experts.down_proj.normal_(0.0, 0.5)
    model.model.norm.weight.normal_(0.0, 0.3)

model = model.double().eval()
state = model.state_dict()

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]

# --- shared-expert-gate assertion (pins sigmoid(shared_expert_gate)*out) --
zeroed = Qwen3_5MoeForCausalLM(config)
sd_zero = {k: v.clone() for k, v in state.items()}
for name, w in sd_zero.items():
    if name.endswith('mlp.shared_expert_gate.weight'):
        w.zero_()  # gate logit 0 -> sigmoid = 0.5
zeroed.load_state_dict(sd_zero)
zeroed.config._attn_implementation = 'eager'
zeroed = zeroed.double().eval()
with torch.no_grad():
    lg = model(input_ids=torch.tensor([sequences[0]])).logits
    l0 = zeroed(input_ids=torch.tensor([sequences[0]])).logits
gate_effect = (lg - l0).abs().max().item()
assert gate_effect > 1e-3, \
    f'shared_expert_gate had no effect on the logits ({gate_effect})'
print(f'shared-expert-gate effect on logits: max |diff| = {gate_effect:.4f}')

# --- f32-consistency assertion: HF's OWN float32 forward must track the
# f64 oracle far inside the Pascal test's 1e-4 gate, or the fixture is too
# numerically hot for any faithful f32 importer to pass. ---
m32 = Qwen3_5MoeForCausalLM(config)
m32.load_state_dict({k: v.float() for k, v in state.items()})
m32.config._attn_implementation = 'eager'
m32 = m32.float().eval()
with torch.no_grad():
    f32_drift = max(
        (m32(input_ids=torch.tensor([seq])).logits[0].double() -
         torch.tensor(lgt)).abs().max().item()
        for seq, lgt in zip(sequences, logits))
assert f32_drift < 3e-5, \
    f'fixture too hot: HF f32 drifts {f32_drift} from the f64 oracle'
print(f'HF f32 vs f64 oracle drift: max |diff| = {f32_drift:.2e}')

sd = {k: v.to(torch.float32).contiguous() for k, v in state.items()}
# The fused expert slabs must be present under the on-disk names.
assert any(k.endswith('mlp.experts.gate_up_proj') for k in sd)
assert any(k.endswith('mlp.experts.down_proj') for k in sd)
save_file(sd, 'tests/fixtures/tiny_qwen3_5_moe.safetensors')

wrapper_cfg = {
    'architectures': ['Qwen3_5MoeForConditionalGeneration'],
    'model_type': 'qwen3_5_moe',
    'text_config': text_cfg,
}
with open('tests/fixtures/tiny_qwen3_5_moe_config.json', 'w') as f:
    json.dump(wrapper_cfg, f, indent=1)
with open('tests/fixtures/tiny_qwen3_5_moe_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_qwen3_5_moe.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')
