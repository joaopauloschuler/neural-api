#!/usr/bin/env python3
"""Generate a tiny RANDOM RWKV-4 parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_rwkv.*: RwkvForCausalLM (HF model_type "rwkv", the RWKV-4
      architecture of RWKV/rwkv-4-169m-pile) - the suite's first
      NON-TRANSFORMER importer target: a recurrent WKV mixer with
      constant decode memory and no KV cache. Quirks exercised:
        - the WKV decay convention: the checkpoint stores a RAW
          time_decay vector applied as a per-step decay factor
          exp(-exp(time_decay)) (HF's kernel computes
          time_decay = -torch.exp(time_decay) internally), and a
          time_first bonus u added to the current token's key;
        - token-shift mixing: per-channel lerp between x_t and x_{t-1}
          with SEPARATE learned time_mix_{key,value,receptance}
          vectors per stream (zero-padded at t=0);
        - the channel-mix FFN: squared-ReLU keying
          square(relu(W_k xk)) and sigmoid receptance gating;
        - plain BIASED LayerNorm everywhere (NOT RMSNorm): ln1/ln2 per
          block, the extra pre_ln embedding norm in block 0 only, and
          the final ln_out;
        - bias-free nn.Linear everywhere (only LayerNorms carry biases);
        - an UNTIED head (tie_word_embeddings false, separate
          head.weight).

The reference logits are computed by HF transformers in float64 (the
oracle convention of the committed fixtures). rescale_every=6 is kept at
the real-checkpoint value: with 2 layers the inference-time rescaling
divides by 2**(layer_id // 6) = 1, a no-op (and it is mathematically an
identity for ANY depth because LayerNorm is scale-invariant, so the
importer loads raw weights).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/rwkv_tiny_fixture.py
writes tests/fixtures/tiny_rwkv{.safetensors,_config.json,_logits.json}.
Needs torch + transformers + safetensors.
"""
import copy
import json
import math
import types

import torch
from safetensors.torch import save_file
from transformers import RwkvConfig, RwkvForCausalLM

HIDDEN = 8
LAYERS = 2
VOCAB = 13
INTERMEDIATE = 16
SEQ_LEN = 8
N_SEQUENCES = 3

torch.manual_seed(20260612)

cfg_dict = {
    'architectures': ['RwkvForCausalLM'],
    'model_type': 'rwkv',
    'vocab_size': VOCAB,
    'context_length': SEQ_LEN,
    'hidden_size': HIDDEN,
    'num_hidden_layers': LAYERS,
    'attention_hidden_size': HIDDEN,
    'intermediate_size': INTERMEDIATE,
    'layer_norm_epsilon': 1e-5,
    'rescale_every': 6,
    'tie_word_embeddings': False,
    'use_cache': True,
    'bos_token_id': 0,
    'eos_token_id': 0,
}
model = RwkvForCausalLM(RwkvConfig(**cfg_dict))

# Pinned inputs (3+ rows: the Pascal parity helper requires it).
sequences = [[(5 * i + 3 * s + s * s) % (VOCAB - 1) + 1
              for i in range(SEQ_LEN)] for s in range(N_SEQUENCES)]

# HF inits to zeros/tiny values at this pico width; re-randomize so every
# quirk is visible in the oracle: O(1) decays both above and below 1
# effective half-life, non-trivial token-shift mixes, FFN pre-activations
# straddling zero (so squared-ReLU != ReLU), non-one LN gains and non-zero
# LN biases.
with torch.no_grad():
    for block in model.rwkv.blocks:
        att, ff = block.attention, block.feed_forward
        att.time_decay.uniform_(-1.5, 1.0)   # exp(td) in [0.22, 2.72]
        att.time_first.normal_(0.0, 0.8)
        for p in (att.time_mix_key, att.time_mix_value,
                  att.time_mix_receptance, ff.time_mix_key,
                  ff.time_mix_receptance):
            p.uniform_(0.2, 0.9)
        for lin in (att.key, att.value, att.receptance, att.output,
                    ff.key, ff.receptance, ff.value):
            lin.weight.normal_(0.0, 0.5)
        for ln in (block.ln1, block.ln2):
            ln.weight.normal_(1.0, 0.25)
            ln.bias.normal_(0.0, 0.2)
    pre = model.rwkv.blocks[0].pre_ln
    pre.weight.normal_(1.0, 0.25)
    pre.bias.normal_(0.0, 0.3)
    model.rwkv.ln_out.weight.normal_(1.0, 0.25)
    model.rwkv.ln_out.bias.normal_(0.0, 0.2)
    model.rwkv.embeddings.weight.normal_(0.0, 0.8)
    model.head.weight.normal_(0.0, 0.6)
model = model.double().eval()

sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_rwkv.safetensors')
with open('tests/fixtures/tiny_rwkv_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)


def run(m):
    with torch.no_grad():
        return torch.stack([m(input_ids=torch.tensor([s])).logits[0]
                            for s in sequences])


logits = run(model)
with open('tests/fixtures/tiny_rwkv_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits.tolist()}, f)
print(f'wrote tiny_rwkv.safetensors ({len(sd)} tensors) + config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
base = logits

# 0. The softplus-inverse decay mapping the Pascal importer applies must
# be EXACT: TNNetWKV's w = softplus(w_raw) with the raw>30 shortcut must
# reproduce exp(time_decay) bit-tight per channel.
for block in model.rwkv.blocks:
    for td in block.attention.time_decay.tolist():
        x = math.exp(td)
        w_raw = x if x > 30 else math.log(math.exp(x) - 1.0)
        w = w_raw if w_raw > 30 else math.log(1.0 + math.exp(w_raw))
        assert abs(w - x) < 1e-12 * max(1.0, x), (td, x, w)
print('softplus-inverse decay round-trip check passed')

# 1. DECAY CONVENTION: an importer that copies time_decay raw into
# TNNetWKV's w_raw slot (effective decay softplus(td) instead of exp(td))
# must FAIL parity. Emulate: exp(td') = softplus(td) -> td' = log(softplus).
wrong = copy.deepcopy(model)
with torch.no_grad():
    for block in wrong.rwkv.blocks:
        td = block.attention.time_decay
        td.copy_(torch.log(torch.nn.functional.softplus(td)))
d = (run(wrong) - base).abs().max().item()
assert d > 1e-2, f'decay convention had no effect ({d})'
print(f'decay-convention effect on logits: max |diff| = {d:.4f}')

# 2. SQUARED ReLU: a plain-ReLU channel-mix must FAIL parity.


def ff_forward_plain_relu(self, hidden, state=None):
    shifted = self.time_shift(hidden)
    key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
    receptance = (hidden * self.time_mix_receptance +
                  shifted * (1 - self.time_mix_receptance))
    key = torch.relu(self.key(key))  # NOT squared
    value = self.value(key)
    receptance = torch.sigmoid(self.receptance(receptance))
    return receptance * value, state


plain = copy.deepcopy(model)
for block in plain.rwkv.blocks:
    block.feed_forward.forward = types.MethodType(
        ff_forward_plain_relu, block.feed_forward)
d = (run(plain) - base).abs().max().item()
assert d > 1e-2, f'squared-ReLU had no effect ({d})'
print(f'squared-ReLU effect on logits: max |diff| = {d:.4f}')

# 3. TOKEN SHIFT: forcing every time_mix to 1.0 (ignore x_{t-1}) must
# FAIL parity - both the attention's three mixes and the FFN's two.
noshift = copy.deepcopy(model)
with torch.no_grad():
    for block in noshift.rwkv.blocks:
        att, ff = block.attention, block.feed_forward
        for p in (att.time_mix_key, att.time_mix_value,
                  att.time_mix_receptance, ff.time_mix_key,
                  ff.time_mix_receptance):
            p.fill_(1.0)
d = (run(noshift) - base).abs().max().item()
assert d > 1e-2, f'token shift had no effect ({d})'
print(f'token-shift effect on logits: max |diff| = {d:.4f}')

# 4. ln0 (pre_ln): skipping the block-0 extra embedding LayerNorm must
# FAIL parity.
noln0 = copy.deepcopy(model)
noln0.rwkv.blocks[0].pre_ln = torch.nn.Identity()
d = (run(noln0) - base).abs().max().item()
assert d > 1e-2, f'pre_ln had no effect ({d})'
print(f'ln0 (pre_ln) effect on logits: max |diff| = {d:.4f}')

# 5. time_first bonus u: zeroing it must FAIL parity.
nobonus = copy.deepcopy(model)
with torch.no_grad():
    for block in nobonus.rwkv.blocks:
        block.attention.time_first.zero_()
d = (run(nobonus) - base).abs().max().item()
assert d > 1e-2, f'time_first bonus had no effect ({d})'
print(f'time_first-bonus effect on logits: max |diff| = {d:.4f}')

# 6. Biased LayerNorm (NOT RMSNorm): zeroing every LN bias must FAIL
# parity (an RMSNorm import has no bias slot at all and also loses the
# mean subtraction).
nobias = copy.deepcopy(model)
with torch.no_grad():
    for m in nobias.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.bias.zero_()
d = (run(nobias) - base).abs().max().item()
assert d > 1e-2, f'LayerNorm biases had no effect ({d})'
print(f'LN-bias effect on logits: max |diff| = {d:.4f}')

# 7. Bias-free Linear everywhere (the RWKV convention).
for m in model.modules():
    if isinstance(m, torch.nn.Linear):
        assert m.bias is None, 'unexpected Linear bias in RWKV'
print('bias-free Linear check passed')
print('all fixture self-checks passed')
