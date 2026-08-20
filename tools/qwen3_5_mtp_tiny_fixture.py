#!/usr/bin/env python3
"""Generate a tiny RANDOM parity fixture for the Qwen3.5/3.8 MTP module
(multi-token prediction), for tests/TestNeuralPretrained.pas. No network
access needed: the weights are randomly initialized from a pico config.

HF transformers 5.11 does NOT implement the MTP module - Qwen3_5ForCausalLM
lists `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` and drops those
tensors. So the oracle here follows vLLM's Qwen3_5MultiTokenPredictor
(vllm/model_executor/models/qwen3_5_mtp.py), reimplemented in torch on top
of HF's own Qwen3_5RMSNorm / Qwen3_5DecoderLayer:

  inputs_embeds = pre_fc_norm_embedding( embed_tokens[x_{r+1}] )
  hidden        = pre_fc_norm_hidden( h_r )
  h             = fc( concat([inputs_embeds, hidden], dim=-1) )
  h             = layers[0](h)          # one full_attention block, causal
  h'            = norm(h)
  logits        = lm_head(h')           # the TRUNK's lm_head (no mtp copy)

The concat puts the EMBEDDING half first: mtp.fc.weight is
[hidden, 2*hidden] and its columns 0..hidden-1 consume the embedding.
vLLM enters a fused-residual decoder layer with residual=None and finishes
with norm(hidden_states, residual); HF's Qwen3_5DecoderLayer is the
non-fused form that already adds both residuals internally, so the same
computation is layer(h) followed by norm(h_out).

Row convention, pinned by the emitted mtp_positions array: row r consumes
the trunk hidden state h_r (post final norm, i.e. what lm_head consumes)
together with embed_tokens[x_{r+1}], and uses rotary position r+1 - the
position of the embedded token. A sequence of T tokens yields T-1 rows.

The safetensors mimics the REAL multimodal checkpoint layout: the text
backbone is renamed under "model.language_model.", lm_head.weight stays
top-level, a dummy model.visual.* tensor the text importer must SKIP is
added, and the 15 REAL mtp.* tensors of Qwen3.8-27B are emitted at pico
scale. The config nests the text fields under "text_config" and carries
mtp_num_hidden_layers / mtp_use_dedicated_embeddings.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/qwen3_5_mtp_tiny_fixture.py
writes tests/fixtures/tiny_qwen3_5_mtp{.safetensors,_config.json,
_oracle.json}. Needs torch + transformers (with qwen3_5) + safetensors.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5RMSNorm)

N_LAYER = 8
LAYER_TYPES = ['linear_attention', 'linear_attention', 'linear_attention',
               'full_attention'] * 2
N_HEAD = 2
N_KV_HEAD = 1
HEAD_DIM = 8           # rotary_dim = 8 * 0.25 = 2 (even, partial rotary)
D_MODEL = 8
D_FF = 12
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
LIN_K_HEADS = 2
LIN_V_HEADS = 4
LIN_K_DIM = 4
LIN_V_DIM = 4
CONV_K = 4
MTP_N_LAYER = 1
MTP_POSITIONS = list(range(1, MAX_POS))
MTP_TENSORS = [
    'mtp.fc.weight',
    'mtp.pre_fc_norm_embedding.weight',
    'mtp.pre_fc_norm_hidden.weight',
    'mtp.norm.weight',
    'mtp.layers.0.input_layernorm.weight',
    'mtp.layers.0.post_attention_layernorm.weight',
    'mtp.layers.0.self_attn.q_proj.weight',
    'mtp.layers.0.self_attn.k_proj.weight',
    'mtp.layers.0.self_attn.v_proj.weight',
    'mtp.layers.0.self_attn.o_proj.weight',
    'mtp.layers.0.self_attn.q_norm.weight',
    'mtp.layers.0.self_attn.k_norm.weight',
    'mtp.layers.0.mlp.gate_proj.weight',
    'mtp.layers.0.mlp.up_proj.weight',
    'mtp.layers.0.mlp.down_proj.weight',
]


class Qwen3_5MultiTokenPredictor(nn.Module):
    """The Qwen3.8 MTP module, named after vLLM's class: two pre-fc norms,
    fc, one full_attention decoder layer, and a final norm."""

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        mtp_config = copy.deepcopy(config)
        mtp_config.num_hidden_layers = MTP_N_LAYER
        mtp_config.layer_types = ['full_attention'] * MTP_N_LAYER
        hidden = config.hidden_size
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            hidden, eps=config.rms_norm_eps)
        self.norm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(mtp_config, i) for i in range(MTP_N_LAYER)])

    def forward(self, trunk_hidden, next_embeds, position_embeddings,
                position_ids, causal_mask, swap_concat=False):
        halves = [self.pre_fc_norm_embedding(next_embeds),
                  self.pre_fc_norm_hidden(trunk_hidden)]
        if swap_concat:
            halves.reverse()
        hidden_states = self.fc(torch.cat(halves, dim=-1))
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids)
        return self.norm(hidden_states)


torch.manual_seed(20260820)
text_cfg = {
    'model_type': 'qwen3_5_text',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
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
    'mtp_num_hidden_layers': MTP_N_LAYER,
    'mtp_use_dedicated_embeddings': False,
    'rope_parameters': {
        'rope_type': 'default',
        'rope_theta': 10000.0,
        'partial_rotary_factor': 0.25,
        'mrope_section': [1, 0, 0],  # rotary_dim/2 = 1 frequency pair
        'mrope_interleaved': True,   # a NO-OP for 1-D text positions
    },
}
config = Qwen3_5TextConfig(
    **{k: v for k, v in text_cfg.items() if k != 'model_type'})


def randomize(module):
    """Re-randomize every loading path: projections std 0.2, zero-centered
    RMSNorm offsets N(0, 0.3), plain DeltaNet gated-norm gain N(1, 0.3)."""
    with torch.no_grad():
        for p in module.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.2)
        for layer in module.modules():
            if isinstance(layer, Qwen3_5DecoderLayer):
                layer.input_layernorm.weight.normal_(0.0, 0.3)
                layer.post_attention_layernorm.weight.normal_(0.0, 0.3)
                if hasattr(layer, 'self_attn'):
                    layer.self_attn.q_norm.weight.normal_(0.0, 0.3)
                    layer.self_attn.k_norm.weight.normal_(0.0, 0.3)
                if hasattr(layer, 'linear_attn'):
                    layer.linear_attn.norm.weight.normal_(1.0, 0.3)
                    layer.linear_attn.A_log.uniform_(-2.0, 1.0)
                    layer.linear_attn.dt_bias.normal_(0.0, 0.5)


trunk = Qwen3_5ForCausalLM(config)
trunk.config._attn_implementation = 'eager'
randomize(trunk)
with torch.no_grad():
    trunk.model.norm.weight.normal_(0.0, 0.3)
trunk = trunk.double().eval()

mtp = Qwen3_5MultiTokenPredictor(config)
mtp.layers[0].self_attn.config._attn_implementation = 'eager'
randomize(mtp)
with torch.no_grad():
    for w in (mtp.pre_fc_norm_embedding.weight, mtp.pre_fc_norm_hidden.weight,
              mtp.norm.weight):
        w.normal_(0.0, 0.3)
mtp = mtp.double().eval()

emitted = {'mtp.' + k for k in mtp.state_dict()}
assert emitted == set(MTP_TENSORS), \
    f'MTP tensor names drifted from the real checkpoint: {emitted}'

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
CAUSAL_MASK = torch.full((1, 1, MAX_POS - 1, MAX_POS - 1),
                         float('-inf')).triu(1)


def run(trunk_model, mtp_module, swap_concat=False):
    """Trunk + MTP over every sequence; returns the four oracle arrays."""
    dtype = trunk_model.lm_head.weight.dtype
    trunk_hidden, trunk_logits, mtp_hidden, mtp_logits = [], [], [], []
    position_ids = torch.tensor([MTP_POSITIONS])
    with torch.no_grad():
        for seq in sequences:
            input_ids = torch.tensor([seq])
            hidden = trunk_model.model(input_ids=input_ids).last_hidden_state
            trunk_hidden.append(hidden[0])
            trunk_logits.append(trunk_model.lm_head(hidden)[0])
            next_embeds = trunk_model.model.embed_tokens(input_ids)[:, 1:, :]
            rows = hidden[:, :-1, :]
            position_embeddings = trunk_model.model.rotary_emb(
                rows, position_ids)
            predicted = mtp_module(rows, next_embeds, position_embeddings,
                                   position_ids, CAUSAL_MASK.to(dtype),
                                   swap_concat)
            mtp_hidden.append(predicted[0])
            mtp_logits.append(trunk_model.lm_head(predicted)[0])
    return trunk_hidden, trunk_logits, mtp_hidden, mtp_logits


trunk_hidden, trunk_logits, mtp_hidden, mtp_logits = run(trunk, mtp)


def max_logit_diff(other_logits):
    return max((a - b).abs().max().item()
               for a, b in zip(mtp_logits, other_logits))


def variant(mutate):
    """Deep-copy the MTP module, mutate it, and rerun the whole oracle."""
    other = copy.deepcopy(mtp)
    with torch.no_grad():
        mutate(other)
    return run(trunk, other)[3]


# --- the MTP head must not reproduce the trunk's own logits ---------------
trunk_effect = max(
    (m - t[:MAX_POS - 1]).abs().max().item()
    for m, t in zip(mtp_logits, trunk_logits))
assert trunk_effect > 1e-3, \
    f'MTP logits match the trunk logits ({trunk_effect})'
print(f'MTP vs trunk logits: max |diff| = {trunk_effect:.4f}')

# --- concat order: embedding half first ----------------------------------
swap_effect = max_logit_diff(run(trunk, mtp, swap_concat=True)[3])
assert swap_effect > 1e-3, \
    f'swapping the fc concat halves had no effect ({swap_effect})'
print(f'fc concat order effect: max |diff| = {swap_effect:.4f}')

# --- the embedding half of fc really feeds the block ---------------------
def zero_embedding_half(module):
    module.fc.weight[:, :D_MODEL] = 0.0


embed_effect = max_logit_diff(variant(zero_embedding_half))
assert embed_effect > 1e-3, \
    f'the embedding half of mtp.fc had no effect ({embed_effect})'
print(f'fc embedding-half effect: max |diff| = {embed_effect:.4f}')

# --- attention output gate (per-head [query|gate] q_proj split) ----------
def zero_output_gate(module):
    weight = module.layers[0].self_attn.q_proj.weight.view(
        N_HEAD, 2 * HEAD_DIM, D_MODEL)
    weight[:, HEAD_DIM:, :] = 0.0  # gate logits -> 0 -> sigmoid = 0.5


gate_effect = max_logit_diff(variant(zero_output_gate))
assert gate_effect > 1e-3, \
    f'MTP attention output gate had no effect ({gate_effect})'
print(f'MTP attn output-gate effect: max |diff| = {gate_effect:.4f}')

# --- f32-consistency: a faithful float32 importer must land far inside the
# Pascal test's 1e-4 parity tolerance. ---
trunk32 = Qwen3_5ForCausalLM(config)
trunk32.load_state_dict({k: v.float() for k, v in trunk.state_dict().items()})
trunk32.config._attn_implementation = 'eager'
trunk32 = trunk32.float().eval()
mtp32 = Qwen3_5MultiTokenPredictor(config)
mtp32.load_state_dict({k: v.float() for k, v in mtp.state_dict().items()})
mtp32.layers[0].self_attn.config._attn_implementation = 'eager'
mtp32 = mtp32.float().eval()
f32_drift = max((a.double() - b).abs().max().item()
                for a, b in zip(run(trunk32, mtp32)[3], mtp_logits))
assert f32_drift < 3e-5, \
    f'fixture too hot: f32 drifts {f32_drift} from the f64 oracle'
print(f'f32 vs f64 oracle drift: max |diff| = {f32_drift:.2e}')

# --- serialize in the REAL multimodal checkpoint layout ------------------
sd = {}
for k, v in trunk.state_dict().items():
    if k == 'lm_head.weight':
        sd[k] = v.to(torch.float32).contiguous()
    elif k.startswith('model.'):
        sd['model.language_model.' + k[len('model.'):]] = \
            v.to(torch.float32).contiguous()
    else:
        raise AssertionError(f'unexpected key {k}')
for k, v in mtp.state_dict().items():
    sd['mtp.' + k] = v.to(torch.float32).contiguous()
# Dummy vision-tower tensor the TEXT importer must SKIP.
sd['model.visual.patch_embed.proj.weight'] = torch.zeros(2, 3)
save_file(sd, 'tests/fixtures/tiny_qwen3_5_mtp.safetensors')

wrapper_cfg = {
    'architectures': ['Qwen3_5ForConditionalGeneration'],
    'model_type': 'qwen3_5',
    'text_config': text_cfg,
}
with open('tests/fixtures/tiny_qwen3_5_mtp_config.json', 'w') as f:
    json.dump(wrapper_cfg, f, indent=1)
with open('tests/fixtures/tiny_qwen3_5_mtp_oracle.json', 'w') as f:
    json.dump({'sequences': sequences,
               'mtp_positions': MTP_POSITIONS,
               'trunk_hidden': [h.tolist() for h in trunk_hidden],
               'mtp_hidden': [h.tolist() for h in mtp_hidden],
               'mtp_logits': [l.tolist() for l in mtp_logits]}, f)
print(f'wrote tiny_qwen3_5_mtp.safetensors ({len(sd)} tensors) '
      f'+ config + oracle ({N_SEQUENCES} sequences of {MAX_POS}, '
      f'{MAX_POS - 1} MTP rows each)')
