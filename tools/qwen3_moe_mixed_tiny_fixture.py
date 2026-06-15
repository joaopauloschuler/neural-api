#!/usr/bin/env python3
"""Generate a tiny RANDOM Qwen3-MoE MIXED dense/MoE parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

This is the MIXED-LAYER sibling of tools/qwen3_moe_tiny_fixture.py. Where that
fixture exercises the uniform all-MoE stack (decoder_sparse_step=1,
mlp_only_layers=[]), THIS one interleaves dense SwiGLU FFN layers with MoE
layers so the per-layer gate (LlamaLayerIsMoE in neuralpretrained.pas, which
mirrors HF modeling_qwen3_moe.Qwen3MoeDecoderLayer.__init__ EXACTLY) is
genuinely exercised:

  A decoder layer i is MoE iff
    (i not in mlp_only_layers) AND (num_experts > 0)
    AND ((i + 1) mod decoder_sparse_step == 0)
  otherwise it is a plain dense SwiGLU FFN of full intermediate_size
  (mlp.gate_proj/up_proj/down_proj.weight).

The config below uses decoder_sparse_step=2 over 3 layers, so:
  layer 0 -> (0+1)%2 = 1 != 0 -> DENSE
  layer 1 -> (1+1)%2 = 0      -> MoE
  layer 2 -> (2+1)%2 = 1 != 0 -> DENSE
i.e. at least one dense AND one MoE layer coexist. The script ASSERTS this
split against the actual instantiated HF module types so the recipe can never
silently drift to all-dense or all-MoE.

Everything else matches the uniform fixture: dense-Qwen3 attention VERBATIM
(per-head q/k RMSNorm BEFORE RoPE with NON-ONE gains, decoupled head_dim, GQA,
no q/k/v bias), per-expert tensor names de-fused from the transformers>=5 3-D
expert slabs, float64 oracle logits, and the renorm/qk-norm "non-vacuous"
assertions.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/qwen3_moe_mixed_tiny_fixture.py
writes tests/fixtures/tiny_qwen3_moe_mixed{.safetensors,_config.json,
_logits.json}. Needs torch + transformers (with Qwen3-MoE) + safetensors.
"""
import json
import types

import torch
import torch.nn.functional as F
import transformers.models.qwen3_moe.modeling_qwen3_moe as Q3M
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM


# Newer transformers fuses the experts into 3-D gate_up_proj/down_proj slabs
# and dispatches to a grouped kernel that does not support float64 - it would
# raise on the float64 oracle. Install a plain, float64-safe eager loop (the
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


Q3M.Qwen3MoeExperts.forward = eager_experts_forward

N_LAYER = 3
N_HEAD = 2
N_KV_HEAD = 1
HEAD_DIM = 6           # != D_MODEL // N_HEAD = 4: decoupled head_dim
D_MODEL = 8
D_FF = 12              # dense intermediate_size (USED by the dense layers)
D_MOE_FF = 5           # moe_intermediate_size: NARROWER per-expert width
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
N_EXPERTS = 4
TOP_K = 2
SPARSE_STEP = 2        # mixed stack: layers 0,2 dense, layer 1 MoE

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
    'use_sliding_window': False,
    'tie_word_embeddings': False,
    'hidden_act': 'silu',
    'num_experts': N_EXPERTS,
    'num_experts_per_tok': TOP_K,
    'norm_topk_prob': True,
    'decoder_sparse_step': SPARSE_STEP,
    'mlp_only_layers': [],
}
model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**qwen3_moe_cfg,
                                           attn_implementation='eager'))
assert N_HEAD * HEAD_DIM != D_MODEL, 'head_dim decoupling not exercised'
assert D_MOE_FF != D_FF, 'moe_intermediate_size not distinct from dense FF'

# Confirm the mixed stack actually has BOTH layer kinds, and that the split
# matches the HF (layer_idx+1) % decoder_sparse_step rule the Pascal importer
# mirrors. Qwen3MoeSparseMoeBlock = MoE; Qwen3MoeMLP = dense SwiGLU FFN.
kinds = []
for i, layer in enumerate(model.model.layers):
    is_moe = isinstance(layer.mlp, Q3M.Qwen3MoeSparseMoeBlock)
    expect_moe = ((i + 1) % SPARSE_STEP == 0)
    assert is_moe == expect_moe, \
        f'layer {i}: HF mlp kind {type(layer.mlp).__name__} disagrees with ' \
        f'the (i+1)%step rule (expect_moe={expect_moe})'
    kinds.append('MoE' if is_moe else 'dense')
assert 'MoE' in kinds and 'dense' in kinds, \
    f'mixed stack must contain BOTH kinds, got {kinds}'
print('layer kinds:', kinds)

# HF ones-inits the q_norm/k_norm RMSNorm gains; carry NON-ONE gains or the
# gain-loading path would be vacuously "verified". HF also inits these pico
# models at std ~0.02, which makes the FFN numerically negligible against the
# residual stream - the renorm/dense assertions would then be vacuous.
# Re-randomize every projection to O(1) scale (the ModernBERT pico recipe) so
# the experts AND the dense FFN genuinely move the logits, and use a flat-ish
# gate so the top-2 probs stay comparable (survivor renorm matters).
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
    """Rewrite the fused (transformers>=5) Qwen3-MoE experts back to the
    on-disk per-expert layout: mlp.experts.{e}.gate_proj/up_proj/down_proj.
    gate_up_proj[e] is [2*I, H] -> F.linear -> chunk -> [gate(I) | up(I)];
    down_proj[e] is [H, I]. Dense layers' mlp.{gate,up,down}_proj.weight and
    the router mlp.gate.weight pass through untouched."""
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
save_file(sd, 'tests/fixtures/tiny_qwen3_moe_mixed.safetensors')
with open('tests/fixtures/tiny_qwen3_moe_mixed_config.json', 'w') as f:
    json.dump(qwen3_moe_cfg, f, indent=1)
sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_qwen3_moe_mixed_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_qwen3_moe_mixed.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')
# Confirm both a dense and an MoE FFN tensor set are on disk.
print('dense layer-0 FFN keys:',
      sorted(k for k in sd if k.startswith('model.layers.0.mlp.')
             and 'experts' not in k and 'gate.weight' not in k)[:3])
print('MoE layer-1 expert keys:',
      sorted(k for k in sd if 'layers.1.mlp.experts' in k)[:3])

# --- q/k norm gain assertion (pins the per-head RMSNorm loading path) -----
ones = Qwen3MoeForCausalLM(Qwen3MoeConfig(**qwen3_moe_cfg,
                                          attn_implementation='eager'))
sd_ones = {k: v.clone() for k, v in model.state_dict().items()}
for k in sd_ones:
    if k.endswith(('q_norm.weight', 'k_norm.weight')):
        sd_ones[k] = torch.ones_like(sd_ones[k])
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
def router_forward_no_renorm(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_probs = F.softmax(router_logits.float(), dim=-1)
    router_top_value, router_indices = torch.topk(
        router_probs, self.top_k, dim=-1)
    # (renorm intentionally OMITTED)
    router_top_value = router_top_value.to(router_logits.dtype)
    return router_logits, router_top_value, router_indices


patched = Qwen3MoeForCausalLM(Qwen3MoeConfig(**qwen3_moe_cfg,
                                             attn_implementation='eager'))
patched.load_state_dict(model.state_dict())
patched = patched.double().eval()
for layer in patched.model.layers:
    if isinstance(layer.mlp, Q3M.Qwen3MoeSparseMoeBlock):
        router = layer.mlp.gate
        router.forward = types.MethodType(router_forward_no_renorm, router)
with torch.no_grad():
    no_renorm = patched(input_ids=torch.tensor([sequences[0]])).logits
norenorm_effect = (lg - no_renorm).abs().max().item()
assert norenorm_effect > 1e-3, \
    f'disabling the top-k renorm had no effect on the logits ' \
    f'({norenorm_effect}) - the fixture does not pin Qwen3-MoE routing'
print(f'top-k renorm effect on logits: max |diff| = {norenorm_effect:.4f}')
