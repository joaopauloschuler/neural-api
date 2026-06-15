#!/usr/bin/env python3
"""Generate a tiny RANDOM OLMoE parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, ~15 KB, pinned in tests/fixtures/:

  tiny_olmoe.*: OlmoeForCausalLM - the SPARSE-MoE sibling of dense OLMo-2,
      the CROSS of the ORIGINAL-OLMo PRE-NORM attention and a Mixtral-style
      top-k Mixture-of-Experts FFN (allenai/OLMoE-1B-0924):
      - ATTENTION/NORM = the ORIGINAL OLMo PRE-NORM block (NOT OLMo-2's
        reordered post-norm): standard input_layernorm before the attention
        residual and post_attention_layernorm before the FFN residual
        (HF modeling_olmoe.OlmoeDecoderLayer.forward). q/k RMSNorm is over the
        FULL flattened projection width (q_norm [num_heads*head_dim], k_norm
        [num_kv_heads*head_dim]) BEFORE the head split + RoPE - the OLMo-2
        QKNormFullWidth placement. GQA (1 kv head < 2 query heads), no q/k/v
        bias, clip_qkv=null. The gains are re-randomized (HF ones-inits every
        RMSNorm) with PER-CHANNEL (non-head-periodic) values so a per-head
        (Qwen3-style) RMS mix-up cannot pass, and the script ASSERTS that
        resetting them to ones moves the logits.
      - FFN = Mixtral-style sparse MoE wired EXACTLY like Qwen3-MoE: a router
        mlp.gate.weight (num_experts logits) -> per-token softmax over ALL
        experts -> hard top-k (num_experts_per_tok=2) gate, the top-k subset
        renormalized (norm_topk_prob=True), feeding num_experts independent
        SwiGLU experts. OLMoE experts are the plain intermediate_size (NOT a
        separate moe_intermediate_size). NO shared expert, NO per-expert/router
        bias. The script ASSERTS that DISABLING the top-k renorm moves the
        logits so the parity test pins the norm_topk_prob knob.

      OLMoE is UNIFORMLY MoE: EVERY decoder layer is a sparse block (no
      mlp_only_layers / decoder_sparse_step in HF), so the all-MoE path is
      exercised.

      The committed safetensors uses the per-expert tensor names
      (mlp.gate.weight + mlp.experts.{i}.gate_proj/up_proj/down_proj.weight,
      the layout BuildOlmoeFromSafeTensors imports - the same Qwen3-MoE
      dialect the released OLMoE checkpoint ships). transformers>=5 fuses the
      experts into 3-D gate_up_proj/down_proj slabs internally; this script
      de-fuses them back to the per-expert keys.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/olmoe_tiny_fixture.py
writes tests/fixtures/tiny_olmoe{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (with OLMoE) + safetensors.
"""
import json
import types

import torch
import torch.nn.functional as F
import transformers.models.olmoe.modeling_olmoe as OLM
from safetensors.torch import save_file
from transformers import OlmoeConfig, OlmoeForCausalLM


# transformers>=5 fuses the experts into 3-D gate_up_proj/down_proj slabs and
# dispatches to a grouped kernel that does not support float64 - it would raise
# on the float64 oracle. Install a plain, float64-safe eager loop (the
# per-expert SwiGLU) as the experts forward for the whole run.
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


OLM.OlmoeExperts.forward = eager_experts_forward

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1          # GQA: 1 kv head < 2 query heads
HEAD_DIM = 4           # OLMoE has NO decoupled head_dim: = hidden // heads
D_MODEL = 8
D_FF = 5               # intermediate_size = the per-expert SwiGLU width
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
N_EXPERTS = 4
TOP_K = 2

torch.manual_seed(20260613)
olmoe_cfg = {
    'architectures': ['OlmoeForCausalLM'],
    'model_type': 'olmoe',
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
    'clip_qkv': None,
    'tie_word_embeddings': False,
    'hidden_act': 'silu',
    'num_experts': N_EXPERTS,
    'num_experts_per_tok': TOP_K,
    'norm_topk_prob': True,
    'eos_token_id': 2,
    'bos_token_id': 1,
    'pad_token_id': 0,
}
model = OlmoeForCausalLM(OlmoeConfig(**olmoe_cfg, attn_implementation='eager'))
assert N_HEAD * HEAD_DIM == D_MODEL, 'OLMoE head_dim must be hidden // heads'

# HF ones-inits the q_norm/k_norm RMSNorm gains; carry NON-ONE gains or the
# gain-loading path would be vacuously "verified". HF also inits these pico
# models at std ~0.02, which makes the expert MLPs numerically negligible
# against the residual stream - the renorm assertion would then be vacuous.
# Re-randomize every projection to O(1) scale (the ModernBERT/Qwen3-MoE pico
# recipe) so the routed experts genuinely move the logits, and use a flat-ish
# gate so the top-2 probs stay comparable (survivor renorm matters). The q/k
# norm gains are PER-CHANNEL random (NOT head-periodic) so a per-head
# (Qwen3-style) RMS placement could not pass the full-width parity.
with torch.no_grad():
    for p in model.parameters():
        if p.dim() >= 2:
            p.normal_(0.0, 0.5)
    for layer in model.model.layers:
        layer.self_attn.q_norm.weight.normal_(1.0, 0.5)
        layer.self_attn.k_norm.weight.normal_(1.0, 0.5)
        layer.mlp.gate.weight.normal_(0.0, 0.3)

model = model.double().eval()
state = model.state_dict()


def perexpert_state_dict(sd):
    """Rewrite the fused (transformers>=5) OLMoE experts back to the on-disk
    per-expert layout: mlp.experts.{e}.gate_proj/up_proj/down_proj.
    gate_up_proj[e] is [2*I, H] -> F.linear -> chunk -> [gate(I) | up(I)];
    down_proj[e] is [H, I]. The router mlp.gate.weight is already per-layer."""
    out = {}
    for k, v in sd.items():
        if k.endswith('mlp.experts.gate_up_proj'):
            base = k[:-len('mlp.experts.gate_up_proj')]
            for e in range(v.shape[0]):
                gate, up = v[e].chunk(2, dim=0)  # [I,H] each
                out[f'{base}mlp.experts.{e}.gate_proj.weight'] = \
                    gate.contiguous().clone()
                out[f'{base}mlp.experts.{e}.up_proj.weight'] = \
                    up.contiguous().clone()
        elif k.endswith('mlp.experts.down_proj'):
            base = k[:-len('mlp.experts.down_proj')]
            for e in range(v.shape[0]):
                out[f'{base}mlp.experts.{e}.down_proj.weight'] = \
                    v[e].contiguous().clone()  # [H,I]
        else:
            out[k] = v.clone()
    return out


sd = {k: v.to(torch.float32).contiguous()
      for k, v in perexpert_state_dict(state).items()}
save_file(sd, 'tests/fixtures/tiny_olmoe.safetensors')
with open('tests/fixtures/tiny_olmoe_config.json', 'w') as f:
    json.dump(olmoe_cfg, f, indent=1)
sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_olmoe_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_olmoe.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')
print('per-expert keys present:',
      sorted(k for k in sd if 'mlp.experts' in k and 'layers.0' in k)[:6])

# --- q/k norm gain assertion (pins the full-width RMSNorm loading path) ----
sd_ones = {k: v.clone() for k, v in model.state_dict().items()}
for k in sd_ones:
    if k.endswith(('q_norm.weight', 'k_norm.weight')):
        sd_ones[k] = torch.ones_like(sd_ones[k])
ones = OlmoeForCausalLM(OlmoeConfig(**olmoe_cfg, attn_implementation='eager'))
ones.load_state_dict(sd_ones)
ones = ones.double().eval()
with torch.no_grad():
    lg = model(input_ids=torch.tensor([sequences[0]])).logits
    l1 = ones(input_ids=torch.tensor([sequences[0]])).logits
norm_effect = (lg - l1).abs().max().item()
assert norm_effect > 1e-3, \
    f'q/k norm gains had no effect on the logits ({norm_effect})'
print(f'qk-norm gain effect on logits: max |diff| = {norm_effect:.4f}')

# --- top-k renorm assertion (pins the norm_topk_prob routing knob) --------
# Monkeypatch every block's router to SKIP the survivor renorm and confirm
# dropping it MOVES the logits, so the parity test genuinely pins
# pRenormalize=true (norm_topk_prob).
def router_forward_no_renorm(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_probs = F.softmax(router_logits.float(), dim=-1)
    router_top_value, router_indices = torch.topk(
        router_probs, self.top_k, dim=-1)
    # (renorm intentionally OMITTED)
    router_top_value = router_top_value.to(router_logits.dtype)
    return router_logits, router_top_value, router_indices


patched = OlmoeForCausalLM(OlmoeConfig(**olmoe_cfg,
                                       attn_implementation='eager'))
patched.load_state_dict(model.state_dict())
patched = patched.double().eval()
for layer in patched.model.layers:
    router = layer.mlp.gate
    router.forward = types.MethodType(router_forward_no_renorm, router)
with torch.no_grad():
    no_renorm = patched(input_ids=torch.tensor([sequences[0]])).logits
norenorm_effect = (lg - no_renorm).abs().max().item()
assert norenorm_effect > 1e-3, \
    f'disabling the top-k renorm had no effect on the logits ' \
    f'({norenorm_effect}) - the fixture does not pin OLMoE routing'
print(f'top-k renorm effect on logits: max |diff| = {norenorm_effect:.4f}')
