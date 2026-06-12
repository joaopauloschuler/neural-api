#!/usr/bin/env python3
"""Generate a tiny RANDOM GPT-Neo parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, ~10 KB, pinned in tests/fixtures/:

  tiny_gptneo.*: GPTNeoForCausalLM with the architecture quirks that
      distinguish GPT-Neo from GPT-2:
        - ALTERNATING attention_types ["global", "local"] with
          window_size=4 SMALLER than the 16-token test sequences, so the
          locally-banded layer genuinely masks (the script ASSERTS that
          the window changes the logits vs window_size >= seqlen);
        - UNSCALED attention (no /sqrt(d_head) on the scores);
        - separate bias-free nn.Linear q/k/v projections ([out, in]
          storage, NOT GPT-2's transposed Conv1D); out_proj and the MLP
          carry biases;
        - intermediate_size left null (the 4*hidden_size default);
        - tied lm_head (GPT-Neo default).

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/gptneo_tiny_fixture.py
writes tests/fixtures/tiny_gptneo{.safetensors,_config.json,_logits.json}.
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import GPTNeoConfig, GPTNeoForCausalLM

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 8
MAX_POS = 16
N_SEQUENCES = 3
WINDOW_SIZE = 4  # < MAX_POS so the local band genuinely masks
VOCAB = 11

torch.manual_seed(20260612)

cfg_dict = {
    'architectures': ['GPTNeoForCausalLM'],
    'model_type': 'gpt_neo',
    'hidden_size': D_MODEL,
    'num_layers': N_LAYER,
    'num_heads': N_HEAD,
    'attention_types': [[['global', 'local'], 1]],
    'window_size': WINDOW_SIZE,
    'intermediate_size': None,  # GPT-Neo default: 4 * hidden_size
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'layer_norm_epsilon': 1e-5,
    'activation_function': 'gelu_new',
    'tie_word_embeddings': True,
}
model = GPTNeoForCausalLM(
    GPTNeoConfig(**cfg_dict, attn_implementation='eager'))
# HF inits weights with std 0.02; at this pico width the attention scores
# are then ~0 and softmax is near-uniform, making the UNSCALED-attention
# quirk numerically invisible. Boost q/k so the scores are O(1) and the
# no-scale and local-window paths are both genuinely exercised.
with torch.no_grad():
    for layer in model.transformer.h:
        layer.attn.attention.q_proj.weight.normal_(0.0, 0.5)
        layer.attn.attention.k_proj.weight.normal_(0.0, 0.5)
model = model.double().eval()

# attention_layers is what the Pascal importer consumes when present; HF
# serializes it alongside attention_types - pin both, like real configs do.
cfg_dict['attention_layers'] = model.config.attention_layers
assert model.config.attention_layers == ['global', 'local']

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_gptneo.safetensors')
with open('tests/fixtures/tiny_gptneo_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_gptneo_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_gptneo.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')

# The local band MUST change the logits vs a window covering the whole
# sequence, otherwise the fixture would not test the window path at all.
wide_cfg = dict(cfg_dict)
del wide_cfg['attention_layers']
wide_cfg['window_size'] = MAX_POS  # band covers every causal key: vacuous
wide = GPTNeoForCausalLM(
    GPTNeoConfig(**wide_cfg, attn_implementation='eager'))
wide.load_state_dict(model.state_dict())
wide = wide.double().eval()
with torch.no_grad():
    lw = model(input_ids=torch.tensor([sequences[0]])).logits
    lf = wide(input_ids=torch.tensor([sequences[0]])).logits
window_effect = (lw - lf).abs().max().item()
assert window_effect > 1e-3, \
    f'local window had no effect on the logits ({window_effect})'
print(f'local-window effect on logits: max |diff| = {window_effect:.4f}')

# The UNSCALED attention must matter too: scaling q by 1/sqrt(d_head)
# (i.e. pretending GPT-Neo used standard scaled attention) must change the
# logits, so the parity test genuinely covers the no-scale quirk.
scaled = GPTNeoForCausalLM(
    GPTNeoConfig(**wide_cfg | {'window_size': WINDOW_SIZE},
                 attn_implementation='eager'))
sd2 = {k: v.clone() for k, v in model.state_dict().items()}
for k in sd2:
    if k.endswith('q_proj.weight'):
        sd2[k] /= HEAD_DIM ** 0.5
scaled.load_state_dict(sd2)
scaled = scaled.double().eval()
with torch.no_grad():
    ls = scaled(input_ids=torch.tensor([sequences[0]])).logits
scale_effect = (lw - ls).abs().max().item()
assert scale_effect > 1e-3, \
    f'attention scaling had no effect on the logits ({scale_effect})'
print(f'no-scale quirk effect on logits: max |diff| = {scale_effect:.4f}')

print('tensor names:')
for k in sorted(sd):
    print(' ', k, list(sd[k].shape))
