#!/usr/bin/env python3
"""Generate a tiny RANDOM Mixtral parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, ~10 KB, pinned in tests/fixtures/:

  tiny_mixtral.*: a MixtralForCausalLM exercising the block_sparse_moe FFN -
      a stock Mistral decoder (full MHA/GQA, RoPE) whose every FFN is a
      sparse MoE: a router gate linear (num_local_experts logits) + N
      independent SwiGLU experts, softmax over ALL experts then top-k
      (num_experts_per_tok=2) routing, output = sum of the selected experts
      weighted by the RENORMALIZED top-k softmax probs (HF normalizes the
      top-k subset). GQA (1 kv head < 2 query heads) and an UNTIED lm_head
      are also on.

      The committed safetensors uses the LEGACY per-expert tensor names
      (block_sparse_moe.gate.weight + block_sparse_moe.experts.{i}.w1/w2/w3,
      the layout every released mistralai/Mixtral-* checkpoint ships and the
      one BuildMixtralFromSafeTensors imports). Newer transformers fuses the
      experts into 3-D gate_up_proj/down_proj slabs internally; this script
      de-fuses them back to the legacy keys so the fixture matches both the
      Pascal importer AND the on-disk checkpoints.

      The script ASSERTS that DISABLING the top-k renorm (keeping the raw,
      un-renormalized survivor softmax weights) MOVES the logits, so the
      parity test genuinely pins the renormalize=true routing knob - the one
      that distinguishes Mixtral from DeepSeek-V2 (norm_topk_prob=false).

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/mixtral_tiny_fixture.py
writes tests/fixtures/tiny_mixtral{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (with Mixtral) + safetensors.
"""
import json
import types

import torch
import torch.nn.functional as F
import transformers.models.mixtral.modeling_mixtral as MX
from safetensors.torch import save_file
from transformers import MixtralConfig, MixtralForCausalLM


# Newer transformers fuses the experts into 3-D gate_up_proj/down_proj slabs
# and dispatches to a grouped_mm kernel that only supports fp16/bf16/fp32 -
# it raises on the float64 oracle. Install a plain, float64-safe eager loop
# (the legacy per-expert SwiGLU) as the experts forward for the whole run.
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


MX.MixtralExperts.forward = eager_experts_forward

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1
D_MODEL = 8
D_FF = 12
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
N_EXPERTS = 4
TOP_K = 2

torch.manual_seed(20260612)
mixtral_cfg = {
    'architectures': ['MixtralForCausalLM'],
    'model_type': 'mixtral',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-5,
    'rope_theta': 10000.0,
    'tie_word_embeddings': False,
    'hidden_act': 'silu',
    'num_local_experts': N_EXPERTS,
    'num_experts_per_tok': TOP_K,
    'sliding_window': None,
    'output_router_logits': False,
}
model = MixtralForCausalLM(MixtralConfig(**mixtral_cfg,
                                         attn_implementation='eager'))

# Re-randomize the per-block router gates a touch wider so the top-2
# selection varies across tokens and experts (a flat gate would make the
# renorm assertion below vacuous).
# HF initializes these tiny models at std ~0.02, which makes the expert MLPs
# (and hence the whole MoE branch) numerically negligible against the
# residual stream - the renorm assertion would then be vacuous. Re-randomize
# every projection to O(1) scale (the ModernBERT pico-fixture recipe) so the
# routed experts genuinely move the logits, and use a FLAT gate so the top-2
# probs stay comparable (survivor renorm = 1/sum, sum well below 1, matters).
with torch.no_grad():
    for p in model.parameters():
        if p.dim() >= 2:
            p.normal_(0.0, 0.5)
    for layer in model.model.layers:
        layer.mlp.gate.weight.normal_(0.0, 0.3)

model = model.double().eval()
state = model.state_dict()


def legacy_state_dict(sd):
    """Rewrite the fused (transformers>=5) Mixtral experts back to the
    legacy on-disk layout: per-expert w1 (gate_proj), w3 (up_proj) and w2
    (down_proj), plus block_sparse_moe.gate.weight. gate_up_proj[e] is
    [2*I, H] -> linear -> [gate(I) | up(I)]; down_proj[e] is [H, I]."""
    out = {}
    for k, v in sd.items():
        if k.endswith('mlp.gate.weight'):
            out[k.replace('mlp.gate.weight',
                          'block_sparse_moe.gate.weight')] = v.clone()
        elif k.endswith('mlp.experts.gate_up_proj'):
            base = k[:-len('mlp.experts.gate_up_proj')]
            for e in range(v.shape[0]):
                gate, up = v[e].chunk(2, dim=0)  # [I,H] each
                out[f'{base}block_sparse_moe.experts.{e}.w1.weight'] = \
                    gate.contiguous().clone()
                out[f'{base}block_sparse_moe.experts.{e}.w3.weight'] = \
                    up.contiguous().clone()
        elif k.endswith('mlp.experts.down_proj'):
            base = k[:-len('mlp.experts.down_proj')]
            for e in range(v.shape[0]):
                out[f'{base}block_sparse_moe.experts.{e}.w2.weight'] = \
                    v[e].contiguous().clone()  # [H,I]
        else:
            out[k] = v.clone()
    return out


sd = {k: v.to(torch.float32).contiguous()
      for k, v in legacy_state_dict(state).items()}
save_file(sd, 'tests/fixtures/tiny_mixtral.safetensors')
with open('tests/fixtures/tiny_mixtral_config.json', 'w') as f:
    json.dump(mixtral_cfg, f, indent=1)
sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_mixtral_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_mixtral.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')
print('legacy expert keys present:',
      sorted(k for k in sd if 'block_sparse_moe' in k and '.0.' in k)[:6])

# --- top-k renorm assertion (pins the renormalize=true routing knob) -----
# Monkeypatch every block's router to SKIP the survivor renorm (HF hardcodes
# router_top_value /= router_top_value.sum() in MixtralTopKRouter.forward)
# and confirm dropping it MOVES the logits - so the parity test genuinely
# pins pRenormalize=true (the knob that splits Mixtral from DeepSeek-V2).
def router_forward_no_renorm(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_probs = F.softmax(router_logits.float(), dim=-1)
    router_top_value, router_indices = torch.topk(
        router_probs, self.top_k, dim=-1)
    # (renorm intentionally OMITTED)
    return router_logits, router_top_value, router_indices


with torch.no_grad():
    base = model(input_ids=torch.tensor([sequences[0]])).logits

patched = MixtralForCausalLM(MixtralConfig(**mixtral_cfg,
                                           attn_implementation='eager'))
patched.load_state_dict(model.state_dict())
patched = patched.double().eval()
for layer in patched.model.layers:
    router = layer.mlp.gate
    router.forward = types.MethodType(router_forward_no_renorm, router)
with torch.no_grad():
    no_renorm = patched(input_ids=torch.tensor([sequences[0]])).logits
norenorm_effect = (base - no_renorm).abs().max().item()
assert norenorm_effect > 1e-3, \
    f'disabling the top-k renorm had no effect on the logits ' \
    f'({norenorm_effect}) - the fixture does not pin Mixtral routing'
print(f'top-k renorm effect on logits: max |diff| = {norenorm_effect:.4f}')
