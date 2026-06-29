#!/usr/bin/env python3
"""Generate a tiny RANDOM XLM-RoBERTa parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

XLM-RoBERTa (FacebookAI/xlm-roberta-base, the multilingual encoder behind the
sentence-transformers / reranker / ColBERT multilingual backbones) is the
SAME NETWORK as RoBERTa: same post-LN bidirectional encoder, same tensor
names (XLMRobertaModel exports unprefixed), the SAME create_position_ids +2
offset and type_vocab_size = 1. The ONLY real difference is the tokenizer
(SentencePiece sentencepiece.bpe.model vs RoBERTa BPE) - which lives OUTSIDE
this network builder. So this fixture's job is to PIN that the importer keeps
treating model_type "xlm-roberta" as the RoBERTa encoder, and that the three
deltas that distinguish it from plain BERT are exercised:

  - THE position-id OFFSET: create_position_ids_from_input_ids starts
    real-token positions at padding_idx+1 = pad_token_id+1 = 2, so checkpoint
    position rows 0 and 1 are NEVER read and the usable context is
    max_position_embeddings - 2. The script ASSERTS rows 0/1 are unused AND
    that an importer reading rows 0..L-1 instead of 2..L+1 (the natural bug)
    fails the 2e-5 Pascal parity gate;
  - type_vocab_size = 1: the token-type branch degenerates to a single
    constant row added at every position (the "no real segment embedding"
    delta);
  - otherwise the exact BERT skeleton: same tensor names, POST-LN blocks,
    bidirectional attention, exact erf GELU (asserted visible), pooler head.

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): last_hidden_state + pooler_output for
every sequence. Sequences are MAX_POS-2 = 14 tokens long (the full usable
context) and contain NO pad token (id 1) - a pad token would get a padding
position id and break the consecutive-positions assumption of the importer.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_xlmroberta_fixture.py
writes tests/fixtures/tiny_xlmroberta{.safetensors,_config.json,_hidden.json}.
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import XLMRobertaConfig, XLMRobertaModel

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 8
INTERMEDIATE = 16
MAX_POS = 16
PAD_ID = 1
OFFSET = PAD_ID + 1            # first usable position row = 2
SEQ_LEN = MAX_POS - OFFSET     # 14: the full usable context
N_SEQUENCES = 3
VOCAB = 11

torch.manual_seed(20260628)

cfg_dict = {
    'architectures': ['XLMRobertaModel'],
    'model_type': 'xlm-roberta',
    'hidden_size': D_MODEL,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'intermediate_size': INTERMEDIATE,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'type_vocab_size': 1,
    'pad_token_id': PAD_ID,
    'hidden_act': 'gelu',
    'layer_norm_eps': 1e-5,
}
model = XLMRobertaModel(
    XLMRobertaConfig(**cfg_dict, attn_implementation='eager'))
# Same pico-width boosts as tools/roberta_tiny_fixture.py: O(1)-structured
# attention and FFN pre-activations where exact-vs-tanh GELU is visible
# (HF std-0.02 init is vacuous at pico scale - cf. the ModernBERT note).
with torch.no_grad():
    for layer in model.encoder.layer:
        layer.attention.self.query.weight.normal_(0.0, 0.7)
        layer.attention.self.key.weight.normal_(0.0, 0.7)
        layer.attention.self.value.weight.normal_(0.0, 0.5)
        layer.intermediate.dense.weight.normal_(0.0, 1.3)
    # Make the never-read padding-position rows 0/1 LOUD: if any importer
    # ever reads them, the parity diff is O(10), not O(0.02).
    model.embeddings.position_embeddings.weight[:OFFSET].normal_(0.0, 10.0)
model = model.double().eval()

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_xlmroberta.safetensors')
with open('tests/fixtures/tiny_xlmroberta_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

# NO pad token (id 1) inside the sequences: HF would give it a padding
# position id and the consecutive-positions importer premise would break.
sequences = [[tid if tid != PAD_ID else (tid + 3) % VOCAB
              for tid in ((7 * i + 3 * s + s * s) % VOCAB
                          for i in range(SEQ_LEN))]
             for s in range(N_SEQUENCES)]
# type_vocab_size = 1: token-type ids can only be 0 (the constant row).
token_types = [[0] * SEQ_LEN for _ in range(N_SEQUENCES)]

hidden = []
pooler = []
with torch.no_grad():
    for seq in sequences:
        out = model(input_ids=torch.tensor([seq]))
        hidden.append(out.last_hidden_state[0].tolist())
        pooler.append(out.pooler_output[0].tolist())
with open('tests/fixtures/tiny_xlmroberta_hidden.json', 'w') as f:
    json.dump({'sequences': sequences, 'token_types': token_types,
               'hidden': hidden, 'pooler': pooler}, f)
print(f'wrote tiny_xlmroberta.safetensors ({len(sd)} tensors) '
      f'+ config + hidden states ({N_SEQUENCES} sequences of {SEQ_LEN})')

# ---- fixture self-checks: every quirk must be visible in the reference ----
with torch.no_grad():
    ids = torch.tensor([sequences[0]])
    base = model(input_ids=ids)
    # 0. HF itself must use positions 2..SEQ_LEN+1 for these inputs.
    pos_ids = model.embeddings.create_position_ids_from_input_ids(
        ids, model.embeddings.padding_idx)
    assert pos_ids[0].tolist() == list(range(OFFSET, OFFSET + SEQ_LEN)), \
        pos_ids[0].tolist()
    print(f'HF position ids: {pos_ids[0, 0].item()}..'
          f'{pos_ids[0, -1].item()} (offset {OFFSET} confirmed)')
    # 1. position rows 0/1 are UNUSED: rewriting them must not move the
    # reference at all.
    scrub = XLMRobertaModel(XLMRobertaConfig(**cfg_dict,
                                             attn_implementation='eager'))
    scrub.load_state_dict(model.state_dict())
    scrub = scrub.double().eval()
    with torch.no_grad():
        scrub.embeddings.position_embeddings.weight[:OFFSET] = 123.0
    scrub_diff = (base.last_hidden_state -
                  scrub(input_ids=ids).last_hidden_state).abs().max().item()
    assert scrub_diff == 0.0, \
        f'position rows 0/1 leaked into the reference ({scrub_diff})'
    print('position rows 0/1 rewritten -> hidden states unchanged (unused)')
    # 2. THE bug to catch: an importer reading position rows 0..L-1 instead
    # of 2..L+1. Simulate it by shifting the table so unshifted reads see
    # the true rows' predecessors, and assert the parity gate would fail.
    bug = XLMRobertaModel(XLMRobertaConfig(**cfg_dict,
                                           attn_implementation='eager'))
    bug.load_state_dict(model.state_dict())
    bug = bug.double().eval()
    with torch.no_grad():
        w = bug.embeddings.position_embeddings.weight
        w[OFFSET:OFFSET + SEQ_LEN] = \
            model.embeddings.position_embeddings.weight[:SEQ_LEN].double()
    bug_diff = (base.last_hidden_state -
                bug(input_ids=ids).last_hidden_state).abs().max().item()
    assert bug_diff > 2e-5, \
        f'unshifted-position bug invisible in the fixture ({bug_diff})'
    print(f'unshifted-position-rows bug effect: max |diff| = {bug_diff:.3f}')
    # 3. the single token-type row must matter (it is summed everywhere).
    tt_norm = model.embeddings.token_type_embeddings.weight.abs().max()
    assert tt_norm.item() > 1e-3, 'token-type row is ~zero'
    # 4. bidirectionality must matter (causal importers fail at position 0).
    perturbed = list(sequences[0])
    perturbed[-1] = 5 if perturbed[-1] != 5 else 6
    pert = model(input_ids=torch.tensor([perturbed]))
    causal_effect = (base.last_hidden_state[0, 0] -
                     pert.last_hidden_state[0, 0]).abs().max().item()
    assert causal_effect > 1e-3, \
        f'last token did not affect position 0 ({causal_effect})'
    print(f'bidirectional flow (last token -> position 0): '
          f'max |diff| = {causal_effect:.4f}')
    # 5. exact-vs-tanh GELU must differ beyond the 2e-5 Pascal gate.
    tanh_cfg = XLMRobertaConfig(**{**cfg_dict,
                                   'hidden_act': 'gelu_pytorch_tanh'},
                                attn_implementation='eager')
    tanh_model = XLMRobertaModel(tanh_cfg)
    tanh_model.load_state_dict(model.state_dict())
    tanh_model = tanh_model.double().eval()
    tanh_out = tanh_model(input_ids=ids)
    gelu_effect = (base.last_hidden_state - tanh_out.last_hidden_state) \
        .abs().max().item()
    assert gelu_effect > 2.5e-5, \
        f'exact vs tanh GELU invisible in the fixture ({gelu_effect})'
    print(f'exact-vs-tanh GELU effect: max |diff| = {gelu_effect:.2e}')
    # 6. model_type must be the multilingual XLM-R one (the importer dispatch
    # key), NOT plain "roberta".
    assert cfg_dict['model_type'] == 'xlm-roberta', cfg_dict['model_type']
    print('model_type = xlm-roberta confirmed')
