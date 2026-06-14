#!/usr/bin/env python3
"""Generate a tiny RANDOM GPT-OSS (model_type "gpt_oss") parity fixture for
tests/TestNeuralPretrained.pas. No network access: the model is randomly
initialized from a pico config, never downloaded.

One fixture, ~tens of KB, pinned in tests/fixtures/:

  tiny_gpt_oss.*: GptOssForCausalLM - OpenAI gpt-oss, the CROSS of several
      subsystems no other importer combines:
      - ATTENTION with a LEARNED PER-HEAD SINK logit appended to the softmax
        denominator (the sink leaks softmax mass off the real keys but
        contributes nothing to the value mix); GQA (1 kv head < 4 q heads);
        DECOUPLED head_dim; q/k/v/o ALL carry biases. The script re-randomizes
        the sinks to NON-ZERO values (HF inits them ~0.02) and ASSERTS that
        zeroing them moves the logits, so the sink-loading path is genuinely
        pinned.
      - ALTERNATING sliding-window / full attention per layer (layer_types:
        layer 0 sliding, layer 1 full).
      - YaRN RoPE (rope_parameters.rope_type "yarn") on every layer.
      - FFN = top-k routed MoE (num_experts_per_tok=2 of num_local_experts=4)
        with gpt-oss's CLAMPED-SwiGLU expert activation (alpha=1.702,
        limit=7.0), the gate|up projection INTERLEAVED (gate=even, up=odd),
        and BOTH the expert projections AND the router carrying biases.

      The committed safetensors stores the experts in the DENSE batched F32
      form (mlp.experts.gate_up_proj [E,hidden,2*inter] +
      mlp.experts.down_proj [E,inter,hidden], with the matching _bias tensors)
      - the same layout HF save_pretrained writes for an unquantized model and
      BuildGptOssFromSafeTensors imports straight. Real 20B/120B checkpoints
      ship these experts as MXFP4 4-bit blocks; the importer's MXFP4
      dequant-at-load path is exercised separately by TestMXFP4Dequant* and the
      dense form keeps this fixture tiny.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures). gpt_oss is eager-only (_supports_sdpa is
False), and its eager_attention_forward implements the sinks exactly; the
expert loop is replaced with a plain float64-safe SwiGLU loop.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/gpt_oss_tiny_fixture.py
writes tests/fixtures/tiny_gpt_oss{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (with gpt_oss) + safetensors.
"""
import json
import math

import numpy as np
import torch
import transformers.models.gpt_oss.modeling_gpt_oss as GO
from transformers.integrations.mxfp4 import FP4_VALUES
from safetensors.torch import save_file
from transformers import GptOssConfig, GptOssForCausalLM


# The fused grouped-experts kernel does not run in float64; install a plain,
# float64-safe eager loop that matches GptOssExperts exactly (interleaved
# gate/up split, clamped-SwiGLU _apply_gate, per-expert bias).
def eager_experts_forward(self, hidden_states, router_indices=None,
                          routing_weights=None):
    next_states = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(
        router_indices, num_classes=self.num_experts).permute(2, 1, 0)
    for e in range(self.num_experts):
        top_k_pos, tok = torch.where(expert_mask[e])
        if tok.numel() == 0:
            continue
        cur = hidden_states[tok]
        gate_up = cur @ self.gate_up_proj[e] + self.gate_up_proj_bias[e]
        gated = self._apply_gate(gate_up)
        out = gated @ self.down_proj[e] + self.down_proj_bias[e]
        out = out * routing_weights[tok, top_k_pos, None]
        next_states.index_add_(0, tok, out.to(next_states.dtype))
    return next_states


GO.GptOssExperts.forward = eager_experts_forward

N_LAYER = 2
N_HEAD = 4
N_KV_HEAD = 1
HEAD_DIM = 10          # != D_MODEL // N_HEAD = 8: decoupled head_dim
# hidden and intermediate are multiples of the MXFP4 block size (32) so the
# SAME pico can be re-emitted with MXFP4-packed experts (the experts are
# blocked along hidden for gate_up and along intermediate for down).
D_MODEL = 32
D_FF = 32              # per-expert SwiGLU width (intermediate_size)
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
N_EXPERTS = 4
TOP_K = 2
SLIDING_WINDOW = 4

torch.manual_seed(20260613)
gpt_oss_cfg = {
    'architectures': ['GptOssForCausalLM'],
    'model_type': 'gpt_oss',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'head_dim': HEAD_DIM,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-5,
    'sliding_window': SLIDING_WINDOW,
    'num_local_experts': N_EXPERTS,
    'num_experts_per_tok': TOP_K,
    'tie_word_embeddings': False,
    'attention_bias': True,
    'layer_types': ['sliding_attention', 'full_attention'],
    'rope_parameters': {
        'rope_type': 'yarn',
        'factor': 4.0,
        'beta_fast': 32.0,
        'beta_slow': 1.0,
        'truncate': False,
        'original_max_position_embeddings': 4,
        'rope_theta': 10000.0,
    },
}
model = GptOssForCausalLM(GptOssConfig(**gpt_oss_cfg,
                                       attn_implementation='eager'))
assert N_HEAD * HEAD_DIM != D_MODEL, 'head_dim decoupling not exercised'

# HF inits at std ~0.02, which makes the routed experts numerically negligible
# against the residual stream and the sinks ~0. Re-randomize to O(1) scale so
# the experts genuinely move the logits, give the sinks NON-ZERO values, and
# use a flat-ish router so the top-2 probs stay comparable.
with torch.no_grad():
    for p in model.parameters():
        if p.dim() >= 2:
            p.normal_(0.0, 0.5)
    for layer in model.model.layers:
        layer.self_attn.sinks.normal_(0.0, 1.0)
        layer.mlp.router.weight.normal_(0.0, 0.3)
        layer.mlp.router.bias.normal_(0.0, 0.1)
        # expert biases at a small but non-trivial scale
        layer.mlp.experts.gate_up_proj_bias.normal_(0.0, 0.2)
        layer.mlp.experts.down_proj_bias.normal_(0.0, 0.2)

model = model.double().eval()
state = model.state_dict()
sd = {k: v.to(torch.float32).contiguous() for k, v in state.items()}
save_file(sd, 'tests/fixtures/tiny_gpt_oss.safetensors')
with open('tests/fixtures/tiny_gpt_oss_config.json', 'w') as f:
    json.dump(gpt_oss_cfg, f, indent=1)

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_gpt_oss_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_gpt_oss.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')

# --- sink-logit assertion (pins the per-head sink-loading path) -----------
sd_zero = {k: v.clone() for k, v in model.state_dict().items()}
for k in sd_zero:
    if k.endswith('self_attn.sinks'):
        sd_zero[k] = torch.zeros_like(sd_zero[k])
zero = GptOssForCausalLM(GptOssConfig(**gpt_oss_cfg,
                                      attn_implementation='eager'))
zero.load_state_dict(sd_zero)
zero = zero.double().eval()
with torch.no_grad():
    lg = model(input_ids=torch.tensor([sequences[0]])).logits
    lz = zero(input_ids=torch.tensor([sequences[0]])).logits
sink_effect = (lg - lz).abs().max().item()
assert sink_effect > 1e-3, \
    f'zeroing the attention sinks had no effect ({sink_effect}) - ' \
    f'the fixture does not pin the gpt-oss sink path'
print(f'sink-logit effect on logits: max |diff| = {sink_effect:.4f}')

# --- sliding-window assertion (pins the alternating-window path) -----------
# Make layer 0 FULL instead of sliding and confirm the logits move, so the
# per-layer window genuinely matters at this sequence length.
full_cfg = dict(gpt_oss_cfg)
full_cfg['layer_types'] = ['full_attention', 'full_attention']
full = GptOssForCausalLM(GptOssConfig(**full_cfg, attn_implementation='eager'))
full.load_state_dict(model.state_dict())
full = full.double().eval()
with torch.no_grad():
    lf = full(input_ids=torch.tensor([sequences[0]])).logits
window_effect = (lg - lf).abs().max().item()
assert window_effect > 1e-3, \
    f'making the sliding layer full had no effect ({window_effect}) - ' \
    f'the fixture does not pin the alternating-window path'
print(f'sliding-window effect on logits: max |diff| = {window_effect:.4f}')


# --- MXFP4 variant (exercises the importer's MXFP4 dequant-at-load) ---------
# Real gpt-oss checkpoints ship the MoE experts as MXFP4 4-bit blocks. Emit a
# SECOND fixture whose experts are stored as the "_blocks" (uint8 packed
# nibbles, [E,OUT,IN/32,16]) + "_scales" (E8M0, [E,OUT,IN/32]) pair - the exact
# inverse of transformers _convert_moe_packed_tensors (which transposes the
# last two axes, so the dense [E,IN,OUT] is transposed to [E,OUT,IN] before
# packing along IN). The reference logits use the DEQUANTIZED experts (exactly
# what BuildGptOssFromSafeTensors reconstructs), so importer parity is tight
# despite the lossy 4-bit weights.
FP4 = np.array(FP4_VALUES, dtype=np.float64)


def quant_block(vals):  # vals: (32,) float64 -> (blk[16], scale_byte, idx[32])
    amax = float(np.max(np.abs(vals)))
    exp = int(math.floor(math.log2(amax / 6.0))) if amax > 0 else 0
    exp = max(min(exp + 127, 254), 0)
    scale = 2.0 ** (exp - 127)
    q = vals / scale
    idx = np.abs(q[:, None] - FP4[None, :]).argmin(axis=1).astype(np.uint8)
    blk = np.zeros(16, dtype=np.uint8)
    for k in range(16):
        blk[k] = (idx[2 * k] & 0x0F) | ((idx[2 * k + 1] & 0x0F) << 4)
    return blk, np.uint8(exp), idx


def quantize_last_axis(t):  # t: torch [E,OUT,IN] -> blocks, scales, dequant
    a = t.double().cpu().numpy()
    E, O, I = a.shape
    G = I // 32
    blocks = np.zeros((E, O, G, 16), dtype=np.uint8)
    scales = np.zeros((E, O, G), dtype=np.uint8)
    deq = np.zeros((E, O, I), dtype=np.float64)
    for e in range(E):
        for o in range(O):
            for g in range(G):
                blk, sc, idx = quant_block(a[e, o, g * 32:(g + 1) * 32])
                blocks[e, o, g] = blk
                scales[e, o, g] = sc
                deq[e, o, g * 32:(g + 1) * 32] = FP4[idx] * (2.0 ** (int(sc) - 127))
    return torch.from_numpy(blocks), torch.from_numpy(scales), torch.from_numpy(deq)


msd = {}
deq_map = {}
for k, v in sd.items():
    if k.endswith('experts.gate_up_proj') or k.endswith('experts.down_proj'):
        blocks, scales, deq = quantize_last_axis(v.transpose(1, 2).contiguous())
        msd[k + '_blocks'] = blocks
        msd[k + '_scales'] = scales
        deq_map[k] = deq.transpose(1, 2).contiguous()  # back to [E,IN,OUT]
    else:
        msd[k] = v.clone()
save_file({kk: (vv if vv.dtype == torch.uint8 else vv.float()).contiguous()
           for kk, vv in msd.items()},
          'tests/fixtures/tiny_gpt_oss_mxfp4.safetensors')

mx = GptOssForCausalLM(GptOssConfig(**gpt_oss_cfg, attn_implementation='eager'))
mx_state = {k: v.double() for k, v in sd.items()}
for k, deq in deq_map.items():
    mx_state[k] = deq.double()
mx.load_state_dict(mx_state)
mx = mx.double().eval()
with torch.no_grad():
    mx_logits = [mx(input_ids=torch.tensor([s])).logits[0].tolist()
                 for s in sequences]
with open('tests/fixtures/tiny_gpt_oss_mxfp4_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': mx_logits}, f)
max_qerr = max((deq_map[k].double() - sd[k].double()).abs().max().item()
               for k in deq_map)
print(f'wrote tiny_gpt_oss_mxfp4.safetensors + logits '
      f'(max dense-vs-dequant weight error {max_qerr:.4f})')
