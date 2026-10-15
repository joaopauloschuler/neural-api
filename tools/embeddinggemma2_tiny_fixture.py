#!/usr/bin/env python3
"""Generate a tiny RANDOM EmbeddingGemma 2 parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the text tower is
randomly initialized from a pico config, never downloaded - the same recipe
as tools/modernbert_tiny_fixture.py).

One fixture, ~10 KB, pinned in tests/fixtures/:

  tiny_embeddinggemma2.*: EmbeddingGemma2TextModel (model_type
      "embedding_gemma2_text") saved the way the COMPOSITE
      google/embeddinggemma-2 export ships it: the text tower under the
      "language_model." prefix, a composite config.json (model_type
      "embedding_gemma2", the text config under "text_config"), plus one
      dummy "vision_tower.*" tensor the importer must SKIP. Every
      EmbeddingGemma 2 delta is genuinely exercised (each one is ASSERTED
      non-vacuous below by re-running the oracle with that single delta
      disabled):
      - BIDIRECTIONAL attention with a SYMMETRIC sliding window on the
        sliding layers: sliding_window=2 (an inclusive RADIUS, |i-j| <= 2)
        << seq len 12.
      - PER-LAYER geometry: sliding layers 2 heads x head_dim 4 with ONE kv
        head (GQA 2:1); full layers (per_layer_config) head_dim 6 with 2 kv
        heads.
      - PER-LAYER-TYPE RoPE theta (full 1000 vs sliding 10) over each type's
        head_dim.
      - q/k RMSNorm with learned gains, v_norm WITHOUT a gain, attention
        scaling = 1.0.
      - projection-only PLE (per-layer embeddings) and the per-layer
        ple_block gate.
      - a stored per-layer layer_scalar (randomized, not 1).
      - GeGLU (gelu_pytorch_tanh) MLP with sandwich norms, Gemma-4 RMSNorm
        x_hat * w (NOT 1 + w), sqrt(hidden) embedding scale, final norm +
        embedding_projection.

Reference per-token states (last_hidden_state, already projected to
embedding_dim) are computed by HF transformers in float64 and stored under
the "logits" key so the test reuses AssertLogitParityWithFixture with
Vocab=embedding_dim; the mean-pooled + L2-normalized sentence embeddings
(the sentence-transformers head) are stored under "embeddings".

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/embeddinggemma2_tiny_fixture.py
writes tests/fixtures/tiny_embeddinggemma2{.safetensors,_config.json,
_logits.json}. Needs torch + transformers (>= the EmbeddingGemma 2 release)
+ safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel

N_LAYER = 5
LAYER_TYPES = ['sliding_attention', 'sliding_attention', 'full_attention',
               'sliding_attention', 'full_attention']
N_HEAD = 2
D_MODEL = 8
HEAD_DIM = 4            # sliding layers
N_KV = 1                # sliding layers: GQA 2:1
GLOBAL_HEAD_DIM = 6     # full layers (per_layer_config)
GLOBAL_N_KV = 2         # full layers: plain MHA
D_FF = 6
PLE = 4
EMB_DIM = 6
SEQ_LEN = 12
SLIDING_WINDOW = 2      # inclusive radius << SEQ_LEN: the mask bites
FULL_THETA = 1000.0
SLIDING_THETA = 10.0
N_SEQUENCES = 3
VOCAB = 13

torch.manual_seed(20261014)
text_cfg = {
    'model_type': 'embedding_gemma2_text',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV,
    'head_dim': HEAD_DIM,
    'hidden_activation': 'gelu_pytorch_tanh',
    'max_position_embeddings': 64,
    'rms_norm_eps': 1e-6,
    'pad_token_id': 0,
    'eos_token_id': 1,
    'bos_token_id': 2,
    'rope_parameters': {
        'full_attention': {'rope_type': 'default', 'rope_theta': FULL_THETA},
        'sliding_attention': {'rope_type': 'default',
                              'rope_theta': SLIDING_THETA},
    },
    'sliding_window': SLIDING_WINDOW,
    'layer_types': LAYER_TYPES,
    'hidden_size_per_layer_input': PLE,
    'embedding_dim': EMB_DIM,
    'vocab_size': VOCAB,
    'per_layer_config': {
        f'{i:02d}': {'head_dim': GLOBAL_HEAD_DIM,
                     'num_key_value_heads': GLOBAL_N_KV}
        for i, t in enumerate(LAYER_TYPES) if t == 'full_attention'},
}
composite_cfg = {
    'architectures': ['EmbeddingGemma2Model'],
    'model_type': 'embedding_gemma2',
    'text_config': text_cfg,
}


def make_model(**overrides):
    cfg = {**text_cfg, **overrides}
    cfg.pop('model_type')
    pl = cfg.pop('per_layer_config')
    if pl is not None:
        cfg['per_layer_config'] = {int(k): v for k, v in pl.items()}
    return EmbeddingGemma2TextModel(EmbeddingGemma2TextConfig(
        **cfg, attn_implementation='eager'))


model = make_model()
for i, layer in enumerate(model.layers):
    want = GLOBAL_HEAD_DIM if LAYER_TYPES[i] == 'full_attention' else HEAD_DIM
    assert layer.self_attn.head_dim == want, f'layer {i} head_dim'
assert model.layers[0].self_attn.num_key_value_groups == N_HEAD // N_KV

# HF's _init_weights draws every Linear at std 0.02 - at pico scale every
# branch would then be ~1e-3 of the residual stream and no delta could clear
# the 1e-3 non-vacuity floor. Re-randomize at O(1) scale, give every norm a
# NON-TRIVIAL gain (so a skipped/unloaded gain - or a 1+w misread - breaks
# parity) and every layer a non-unit layer_scalar.
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith('norm.weight') or 'layernorm' in name:
            p.normal_(1.0, 0.5)
        else:
            p.normal_(0.0, 0.5)
    for layer in model.layers:
        layer.layer_scalar.uniform_(0.4, 1.2)

model = model.double().eval()
sd = {'language_model.' + k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
# The composite export also carries the vision/audio towers: one dummy tensor
# outside the text prefix that the importer must skip, not reject.
sd['vision_tower.patch_embedder.weight'] = torch.randn(3, 4)
save_file(sd, 'tests/fixtures/tiny_embeddinggemma2.safetensors')
with open('tests/fixtures/tiny_embeddinggemma2_config.json', 'w') as f:
    json.dump(composite_cfg, f, indent=1)
sequences = [[(5 * i + 2 * s + s * s + 3) % VOCAB for i in range(SEQ_LEN)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    hidden = [model(input_ids=torch.tensor([seq])).last_hidden_state[0]
              for seq in sequences]
embeddings = [torch.nn.functional.normalize(h.mean(0), dim=0).tolist()
              for h in hidden]
with open('tests/fixtures/tiny_embeddinggemma2_logits.json', 'w') as f:
    json.dump({'sequences': sequences,
               'logits': [h.tolist() for h in hidden],
               'embeddings': embeddings}, f)
print(f'wrote tiny_embeddinggemma2.safetensors ({len(sd)} tensors) '
      f'+ config + states ({N_SEQUENCES} sequences of {SEQ_LEN})')


def hidden_with(state_dict=None, **overrides):
    """Oracle states for sequence 0 with one delta disabled."""
    other = make_model(**overrides)
    other.load_state_dict(state_dict or model.state_dict())
    other = other.double().eval()
    with torch.no_grad():
        return other(input_ids=torch.tensor([sequences[0]])
                     ).last_hidden_state


def edited(fn):
    out = {k: v.clone() for k, v in model.state_dict().items()}
    fn(out)
    return out


def swap_gate_up(sd_):
    for k in list(sd_):
        if k.endswith('mlp.gate_proj.weight'):
            u = k.replace('gate_proj', 'up_proj')
            sd_[k], sd_[u] = sd_[u].clone(), sd_[k].clone()


def unit_scalars(sd_):
    for k in sd_:
        if k.endswith('layer_scalar'):
            sd_[k].fill_(1.0)


def zero_ple(sd_):
    sd_['ple.per_layer_model_projection.weight'].zero_()


def norm_plus_one(sd_):
    # what a Gemma 1-3 style (1 + w) import would compute
    for k in sd_:
        if k.endswith('layernorm.weight'):
            sd_[k] += 1.0


with torch.no_grad():
    ref = model(input_ids=torch.tensor([sequences[0]])).last_hidden_state

checks = {
    'sliding window (radius 2 -> 64 = full)':
        hidden_with(sliding_window=64),
    'per-layer-type rope theta (sliding 10 -> 1000)':
        hidden_with(rope_parameters={
            'full_attention': {'rope_type': 'default',
                               'rope_theta': FULL_THETA},
            'sliding_attention': {'rope_type': 'default',
                                  'rope_theta': FULL_THETA}}),
    'GeGLU gate/up packing (swapped)': hidden_with(edited(swap_gate_up)),
    'layer_scalar (-> 1)': hidden_with(edited(unit_scalars)),
    'PLE (projection zeroed)': hidden_with(edited(zero_ple)),
    'RMSNorm gain (w -> 1 + w)': hidden_with(edited(norm_plus_one)),
}
for name, alt in checks.items():
    effect = (ref - alt).abs().max().item()
    assert effect > 1e-3, f'{name} had no effect on the output ({effect})'
    print(f'{name}: max |diff| = {effect:.4f}')
