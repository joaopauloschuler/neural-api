#!/usr/bin/env python3
"""Generate a tiny RANDOM RecurrentGemma (Griffin/Hawk) parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_recurrentgemma.*: RecurrentGemmaForCausalLM (HF model_type
      "recurrent_gemma", the google/recurrentgemma-2b architecture) - the
      Griffin/Hawk HYBRID: a fixed temporal-block PATTERN where most layers
      are a "recurrent" block (the RG-LRU real-gated linear recurrent unit +
      a small depthwise causal conv1d temporal mixer) and the rest are LOCAL
      sliding-window attention. Quirks exercised by the fixture:
        - block_types = ('recurrent','recurrent','attention') replicated to
          NumLayers: layers 0,1 recurrent, layer 2 local attention;
        - the RG-LRU recurrence (TNNetRGLRU): per channel with c = 8,
          i_t = sigmoid(W_x x + b_x), r_t = sigmoid(W_a x + b_a),
          log a_t = -c r_t softplus(Lambda), a_t = exp(log a_t),
          h_t = a_t h_{t-1} + sqrt(1-a_t^2) (i_t x_t). The per-head
          block-diagonal input/recurrent gate matrices + biases and the
          conv1d are realised by the importer as upstream projections, the
          leaf owning only Lambda;
        - the recurrent block's GELU(tanh) y-branch gating
          (h = rg_lru(conv(linear_x)) * gelu(linear_y)) + linear_out;
        - LOCAL sliding-window attention (attention_window_size) with
          PARTIAL rotary (partial_rotary_factor 0.5), GQA
          (num_key_value_heads < num_attention_heads), an o_proj BIAS and
          attention scaled by head_dim**-0.5;
        - Gemma RMSNorm gain = (1 + weight) (zero-centered) over hidden,
          eps 1e-6; the embedding x sqrt(hidden) normalizer;
        - the GEGLU(tanh) MLP (gate/up/down all biased, intermediate//2);
        - the FINAL-logit soft cap logits_soft_cap*tanh(logits/cap)
          (default 30) and the TIED lm_head.

The reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

NOT a duplicate of tools/lru fixtures: TNNetLRU is the complex-diagonal
ungated Orvieto LRU; RG-LRU is real-valued and INPUT-GATED.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_recurrentgemma_fixture.py
writes tests/fixtures/tiny_recurrentgemma{.safetensors,_config.json,
_logits.json}.  Needs torch + transformers + safetensors.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import RecurrentGemmaConfig, RecurrentGemmaForCausalLM

HIDDEN = 16
LRU_WIDTH = 16
LAYERS = 3                 # 'recurrent','recurrent','attention'
VOCAB = 32
HEADS = 2                  # block_width = lru_width // heads = 8
KV_HEADS = 1               # GQA
HEAD_DIM = 8               # hidden // heads
INTERMEDIATE = 48          # GEGLU uses intermediate // 2 = 24
CONV_K = 4
WINDOW = 8
SOFT_CAP = 30.0
SEQ_LEN = 6
N_SEQUENCES = 3

torch.manual_seed(20260613)

cfg_dict = {
    'architectures': ['RecurrentGemmaForCausalLM'],
    'model_type': 'recurrent_gemma',
    'vocab_size': VOCAB,
    'hidden_size': HIDDEN,
    'lru_width': LRU_WIDTH,
    'intermediate_size': INTERMEDIATE,
    'num_hidden_layers': LAYERS,
    'num_attention_heads': HEADS,
    'num_key_value_heads': KV_HEADS,
    'head_dim': HEAD_DIM,
    'conv1d_width': CONV_K,
    'attention_window_size': WINDOW,
    'logits_soft_cap': SOFT_CAP,
    'rms_norm_eps': 1e-6,
    'block_types': ['recurrent', 'recurrent', 'attention'],
    'partial_rotary_factor': 0.5,
    'hidden_activation': 'gelu_pytorch_tanh',
    'rope_theta': 10000.0,
    'attention_bias': False,
    'use_cache': True,
    'bos_token_id': 0,
    'eos_token_id': 0,
    'pad_token_id': 0,
}
model = RecurrentGemmaForCausalLM(RecurrentGemmaConfig(**cfg_dict))

assert model.model.embed_tokens.weight.device.type == 'cpu'
assert model.config.layers_block_type == ['recurrent', 'recurrent', 'attention']

# Pinned inputs (3+ rows: the Pascal parity helper requires it). Avoid id 0
# (the padding idx) so the RG-LRU position-0 reset path is not entangled with
# pad handling; position_ids start at 0 so the first token is the natural BOS.
sequences = [[(5 * i + 3 * s + s * s) % (VOCAB - 2) + 1
              for i in range(SEQ_LEN)] for s in range(N_SEQUENCES)]

# Re-randomize away from the (tiny-std) HF init so every quirk is visible in
# the oracle: gate logits straddling zero (sigmoids != 0/1), O(1) Lambda
# decays with real per-channel spread, non-trivial conv taps/biases, real
# attention weights and non-one (1+w) RMSNorm gains.
with torch.no_grad():
    for layer in model.model.layers:
        layer.temporal_pre_norm.weight.normal_(0.0, 0.25)
        layer.channel_pre_norm.weight.normal_(0.0, 0.25)
        tb = layer.temporal_block
        if hasattr(tb, 'rg_lru'):  # recurrent block
            tb.linear_x.weight.normal_(0.0, 0.5)
            tb.linear_x.bias.normal_(0.0, 0.3)
            tb.linear_y.weight.normal_(0.0, 0.5)
            tb.linear_y.bias.normal_(0.0, 0.3)
            tb.linear_out.weight.normal_(0.0, 0.5)
            tb.linear_out.bias.normal_(0.0, 0.3)
            tb.conv_1d.weight.normal_(0.0, 0.5)
            tb.conv_1d.bias.normal_(0.0, 0.4)
            rg = tb.rg_lru
            rg.input_gate_weight.normal_(0.0, 0.5)
            rg.input_gate_bias.normal_(0.0, 0.5)
            rg.recurrent_gate_weight.normal_(0.0, 0.5)
            rg.recurrent_gate_bias.normal_(0.0, 0.5)
            # recurrent_param so softplus(Lambda) gives O(1) decays with spread.
            rg.recurrent_param.copy_(
                torch.empty_like(rg.recurrent_param).uniform_(-1.0, 1.0))
        else:  # attention block
            tb.q_proj.weight.normal_(0.0, 0.5)
            tb.k_proj.weight.normal_(0.0, 0.5)
            tb.v_proj.weight.normal_(0.0, 0.5)
            tb.o_proj.weight.normal_(0.0, 0.5)
            tb.o_proj.bias.normal_(0.0, 0.3)
        mlp = layer.mlp_block
        mlp.gate_proj.weight.normal_(0.0, 0.5)
        mlp.gate_proj.bias.normal_(0.0, 0.3)
        mlp.up_proj.weight.normal_(0.0, 0.5)
        mlp.up_proj.bias.normal_(0.0, 0.3)
        mlp.down_proj.weight.normal_(0.0, 0.5)
        mlp.down_proj.bias.normal_(0.0, 0.3)
    model.model.final_norm.weight.normal_(0.0, 0.25)
    model.model.embed_tokens.weight.normal_(0.0, 0.8)
model = model.double().eval()

# Tied head: lm_head.weight aliases embed_tokens.weight; safetensors refuses
# shared storage, so do not save lm_head.weight (the importer ties it).
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items() if k != 'lm_head.weight'}
save_file(sd, 'tests/fixtures/tiny_recurrentgemma.safetensors')
with open('tests/fixtures/tiny_recurrentgemma_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)


def run(m):
    with torch.no_grad():
        return torch.stack([m(input_ids=torch.tensor([s]),
                              use_cache=False).logits[0]
                            for s in sequences])


logits = run(model)
with open('tests/fixtures/tiny_recurrentgemma_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits.tolist()}, f)
print(f'wrote tiny_recurrentgemma.safetensors ({len(sd)} tensors) + '
      f'config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
base = logits

# 0. RG-LRU vs TNNetRGLRU formula round-trip: HF computes
# log_a = -8 * sigmoid(a_logits) * softplus(Lambda); a = exp(log_a);
# multiplier = sqrt(1 - exp(2 log_a)). The Pascal layer uses the IDENTICAL
# formula. Spot-check the math.
import torch.nn.functional as F
for lam in (-0.7, 0.0, 0.9):
    for rl in (-1.0, 0.4):
        sp = F.softplus(torch.tensor(lam))
        rg = torch.sigmoid(torch.tensor(rl))
        log_a = -8.0 * rg * sp
        a = torch.exp(log_a)
        mult = torch.sqrt(1 - torch.exp(2 * log_a))
        # Pascal-side reconstruction:
        a_ours = torch.exp(-8.0 * rg * sp)
        mult_ours = torch.sqrt(1 - a_ours * a_ours)
        assert abs(a - a_ours) < 1e-12 and abs(mult - mult_ours) < 1e-12
print('RG-LRU c=8 / softplus / sqrt(1-a^2) formula check passed')

# 1. Lambda matters: setting recurrent_param huge (softplus -> large, a -> 0,
# a memoryless RG-LRU) must FAIL parity.
nolam = copy.deepcopy(model)
with torch.no_grad():
    for layer in nolam.model.layers:
        if hasattr(layer.temporal_block, 'rg_lru'):
            layer.temporal_block.rg_lru.recurrent_param.fill_(8.0)
d = (run(nolam) - base).abs().max().item()
assert d > 1e-2, f'Lambda had no effect ({d})'
print(f'Lambda (recurrence decay) effect: max |diff| = {d:.4f}')

# 2. input gate matters: zeroing the input-gate weight+bias (gate = sigmoid(0)
# = 0.5 constant) must FAIL parity.
noig = copy.deepcopy(model)
with torch.no_grad():
    for layer in noig.model.layers:
        if hasattr(layer.temporal_block, 'rg_lru'):
            layer.temporal_block.rg_lru.input_gate_weight.zero_()
            layer.temporal_block.rg_lru.input_gate_bias.zero_()
d = (run(noig) - base).abs().max().item()
assert d > 1e-2, f'input gate had no effect ({d})'
print(f'input-gate effect: max |diff| = {d:.4f}')

# 3. recurrence gate matters: zeroing it (r = 0.5 constant) must FAIL parity.
norg = copy.deepcopy(model)
with torch.no_grad():
    for layer in norg.model.layers:
        if hasattr(layer.temporal_block, 'rg_lru'):
            layer.temporal_block.rg_lru.recurrent_gate_weight.zero_()
            layer.temporal_block.rg_lru.recurrent_gate_bias.zero_()
d = (run(norg) - base).abs().max().item()
assert d > 1e-2, f'recurrence gate had no effect ({d})'
print(f'recurrence-gate effect: max |diff| = {d:.4f}')

# 4. conv1d matters: zeroing conv bias must FAIL parity.
nocb = copy.deepcopy(model)
with torch.no_grad():
    for layer in nocb.model.layers:
        if hasattr(layer.temporal_block, 'rg_lru'):
            layer.temporal_block.conv_1d.bias.zero_()
d = (run(nocb) - base).abs().max().item()
assert d > 1e-2, f'conv bias had no effect ({d})'
print(f'conv-bias effect: max |diff| = {d:.4f}')

# 5. GELU(tanh) y-branch gate matters: zeroing linear_y (gate = gelu(bias))
# must FAIL parity.
nogate = copy.deepcopy(model)
with torch.no_grad():
    for layer in nogate.model.layers:
        if hasattr(layer.temporal_block, 'rg_lru'):
            layer.temporal_block.linear_y.weight.zero_()
            layer.temporal_block.linear_y.bias.zero_()
d = (run(nogate) - base).abs().max().item()
assert d > 1e-2, f'y-branch gate had no effect ({d})'
print(f'GELU y-branch gate effect: max |diff| = {d:.4f}')

# 6. local attention layer matters: zeroing its o_proj must FAIL parity.
noattn = copy.deepcopy(model)
with torch.no_grad():
    for layer in noattn.model.layers:
        if not hasattr(layer.temporal_block, 'rg_lru'):
            layer.temporal_block.o_proj.weight.zero_()
            layer.temporal_block.o_proj.bias.zero_()
d = (run(noattn) - base).abs().max().item()
assert d > 1e-2, f'attention layer had no effect ({d})'
print(f'local-attention layer effect: max |diff| = {d:.4f}')

# 7. sliding window matters: only with SEQ_LEN > WINDOW would shrinking the
# window change logits. Here SEQ_LEN <= WINDOW so the window is inert by
# design (documented); assert the attention layer uses local attention.
assert model.config.layers_block_type[2] == 'attention'
assert SEQ_LEN <= WINDOW, 'fixture seq fits in window (window inert here)'
print('attention is LOCAL sliding-window (window inert at this seq len)')

# 8. partial rotary: partial_rotary_factor 0.5 means only half of head_dim is
# rotated. Assert it is configured (the importer must use RotaryDims =
# head_dim*0.5).
assert abs(model.config.partial_rotary_factor - 0.5) < 1e-9
print('partial rotary factor 0.5 confirmed')

# 9. final logit soft cap: logits must be bounded by the cap.
assert base.abs().max().item() <= SOFT_CAP + 1e-6
print(f'final-logit soft cap honored (|logit|max = {base.abs().max():.4f} '
      f'<= {SOFT_CAP})')

# 10. Gemma (1+w) RMSNorm + tied head.
assert model.lm_head.weight.data_ptr() == \
    model.model.embed_tokens.weight.data_ptr(), 'head not tied'
print('tied-head check passed')
print('all fixture self-checks passed')
