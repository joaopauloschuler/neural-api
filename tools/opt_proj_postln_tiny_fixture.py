#!/usr/bin/env python3
"""Generate a tiny RANDOM opt-350m-SHAPED OPT parity fixture for
tests/TestNeuralPretrained.pas (no network: randomly initialized from a pico
config, never downloaded).

The default tiny_opt fixture (opt_tiny_fixture.py) is opt-125m-shaped:
word_embed_proj_dim == hidden_size (so project_in/project_out are ABSENT) and
do_layer_norm_before=True (PRE-LN). That leaves two importer code paths wired
but untested. The opt-350m architecture exercises BOTH at once:

  - word_embed_proj_dim != hidden_size  -> the bias-free project_in
    (embed -> hidden) after the token embedding and the bias-free project_out
    (hidden -> embed) before the LM head;
  - do_layer_norm_before=False          -> POST-LN block wiring
    (x := N(x + sublayer(x))), distinct from the 125m PRE-LN wiring;
  - _remove_final_layer_norm=False with post-LN still keeps a decoder-level
    final_layer_norm (HF: present iff do_layer_norm_before and not
    _remove_final_layer_norm) -> opt-350m has do_layer_norm_before=False so
    final_layer_norm is ABSENT here, also distinct from the 125m fixture.

So this single fixture covers the project_in/out path AND the post-LN path.

Pinned in tests/fixtures/ (~30 KB):
  tiny_opt350.safetensors + tiny_opt350_config.json + tiny_opt350_logits.json

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures). RE-RANDOMIZED O(1)-scale weights are
used on the attention q/k AND the two projections so the new paths genuinely
matter (the ModernBERT-fixture lesson: HF std-0.02 init is near-vacuous at
pico width).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/opt_proj_postln_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import OPTConfig, OPTForCausalLM

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 8
D_MODEL = N_HEAD * HEAD_DIM        # 16  (hidden_size)
WORD_EMBED_PROJ_DIM = 12          # != hidden_size -> project_in/project_out
FFN_DIM = 24                       # deliberately NOT 4*hidden
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 11

torch.manual_seed(20260626)

cfg_dict = {
    'architectures': ['OPTForCausalLM'],
    'model_type': 'opt',
    'hidden_size': D_MODEL,
    'word_embed_proj_dim': WORD_EMBED_PROJ_DIM,
    'ffn_dim': FFN_DIM,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'layer_norm_elementwise_affine': True,
    # opt-350m architecture: POST-LN.
    'do_layer_norm_before': False,
    'enable_bias': True,
    # opt-350m keeps _remove_final_layer_norm False; with post-LN that means
    # final_layer_norm is ABSENT (HasFinalLayerNorm = do_layer_norm_before and
    # not _remove_final_layer_norm = False).
    '_remove_final_layer_norm': False,
    'activation_function': 'relu',
    'dropout': 0.0,
    'attention_dropout': 0.0,
    'layerdrop': 0.0,
    'pad_token_id': 1,
    'bos_token_id': 2,
    'eos_token_id': 2,
    'tie_word_embeddings': True,
}

model = OPTForCausalLM(OPTConfig(**cfg_dict, attn_implementation='eager'))
# HF inits with std 0.02; at pico width attention scores are then ~0 and the
# projections are near-identity-scale, making the new paths numerically
# invisible. Boost q/k (attention path) and the project_in/project_out
# (the wide<->hidden remap) to O(1) so they genuinely matter.
with torch.no_grad():
    for layer in model.model.decoder.layers:
        layer.self_attn.q_proj.weight.normal_(0.0, 0.5)
        layer.self_attn.k_proj.weight.normal_(0.0, 0.5)
    model.model.decoder.project_in.weight.normal_(0.0, 0.5)
    model.model.decoder.project_out.weight.normal_(0.0, 0.5)
model = model.double().eval()

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_opt350.safetensors')
with open('tests/fixtures/tiny_opt350_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]


def logits_of(m, seqs):
    with torch.no_grad():
        return [m(input_ids=torch.tensor([seq])).logits[0] for seq in seqs]


ref = logits_of(model, sequences)
with open('tests/fixtures/tiny_opt350_logits.json', 'w') as f:
    json.dump({'sequences': sequences,
               'logits': [l.tolist() for l in ref]}, f)
print(f'wrote tiny_opt350.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')

# ---- the project_in/project_out path must matter: replacing project_in with
# an identity-style remap (here, zeroing it) must move the logits, proving the
# learned wide<->hidden projection is genuinely exercised.
with torch.no_grad():
    nopin = OPTForCausalLM(OPTConfig(**cfg_dict, attn_implementation='eager'))
    nopin.load_state_dict(model.state_dict())
    nopin.model.decoder.project_in.weight.zero_()
nopin = nopin.double().eval()
proj_effect = max((a - b).abs().max().item()
                  for a, b in zip(ref, logits_of(nopin, sequences)))
assert proj_effect > 1e-3, \
    f'project_in had no effect on the logits ({proj_effect})'
print(f'project_in effect on logits: max |diff| = {proj_effect:.4f}')

# ---- POST-LN must differ from PRE-LN: flipping do_layer_norm_before must move
# the logits (the same weights wired the two ways produce different outputs).
preln = OPTForCausalLM(OPTConfig(**{**cfg_dict, 'do_layer_norm_before': True},
                                 attn_implementation='eager'))
preln.load_state_dict(model.state_dict(), strict=False)
preln = preln.double().eval()
ln_effect = max((a - b).abs().max().item()
                for a, b in zip(ref, logits_of(preln, sequences)))
assert ln_effect > 1e-3, \
    f'pre-LN vs post-LN had no effect on the logits ({ln_effect})'
print(f'pre-LN-vs-post-LN effect on logits: max |diff| = {ln_effect:.4f}')

print('tensor names:')
for k in sorted(sd):
    print(' ', k, list(sd[k].shape))
