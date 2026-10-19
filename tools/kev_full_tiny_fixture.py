#!/usr/bin/env python3
"""Generate the Kev full-weight parity fixture for tests/TestNeuralDecisionHeads.pas.

A Kev full-weight checkpoint (kev-27b's layout) holds the whole backbone as
`save_pretrained` of a Qwen3_5TextModel: config.json with architectures
["Qwen3_5TextModel"] and a flat model_type "qwen3_5_text", safetensors keys
without the "model." prefix (embed_tokens.weight, layers.N..., norm.weight), no
lm_head, plus head.pt with weights="full" and the tokenizer files.

The backbone here is the kev_qwen3_5 fixture (tools/kev_tiny_fixture.py, run it
first) with its LoRA adapter merged by kev's own loader in float32, saved
untied the way kev-27b is (tie_word_embeddings false). The oracle is kev's own
full-weight path: Checkpoint(dir).load() (backbone from the directory itself,
float64) + admit + DecisionModel.probs.

Writes tests/fixtures/kev_full_qwen3_5/:
  config.json, model.safetensors, tokenizer.json, tokenizer_config.json, head.pt
  kev_full_oracle.json: the request, ids, positions and kev's probabilities.

Coded by Claude (AI).

Usage (from the repo root, inside the ~/x venv):
  python3 tools/kev_full_tiny_fixture.py
"""
import json
import os
import shutil

import torch
from safetensors import safe_open

from kev.api import SystemOneRequest, to_answers, to_record
from kev.checkpoint import Checkpoint, LoadOptions, Meta, write_meta
from kev.model import admit, rows_of

from kev_tiny_fixture import (ADAPTER_DIR, BASE_DIR, POINTER, REQUEST, ROOT,
                              TEMPERATURE, write_json)

FIXTURE = os.path.join('tests', 'fixtures', 'kev_full_qwen3_5')
LORA_ORACLE = os.path.join('tests', 'fixtures', 'kev_qwen3_5', 'kev_oracle.json')


def main():
    os.chdir(ROOT)
    shutil.rmtree(FIXTURE, ignore_errors=True)
    os.makedirs(FIXTURE)
    # kev's LoRA load merges in float32 (one rounding), the full checkpoint keeps it.
    _, merged = Checkpoint(ADAPTER_DIR).load('cpu', LoadOptions(dtype=torch.float32))
    backbone = merged.lm
    assert type(backbone).__name__ == 'Qwen3_5TextModel', type(backbone).__name__
    backbone.config.tie_word_embeddings = False
    backbone.save_pretrained(FIXTURE, safe_serialization=True)
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        shutil.copy(os.path.join(BASE_DIR, name), os.path.join(FIXTURE, name))
    for name in os.listdir(FIXTURE):
        if name not in ('config.json', 'model.safetensors', 'tokenizer.json',
                        'tokenizer_config.json'):
            os.remove(os.path.join(FIXTURE, name))
    write_meta(FIXTURE, Meta(base=FIXTURE, head=merged.head.state_dict(), head_dim=POINTER,
                             temperature=TEMPERATURE, weights='full', weights_dtype='fp32'))

    with open(os.path.join(FIXTURE, 'config.json'), encoding='utf-8') as f:
        config = json.load(f)
    assert config['architectures'] == ['Qwen3_5TextModel'], config['architectures']
    assert config['model_type'] == 'qwen3_5_text' and 'text_config' not in config
    with safe_open(os.path.join(FIXTURE, 'model.safetensors'), 'pt') as f:
        keys = list(f.keys())
    assert 'embed_tokens.weight' in keys and 'norm.weight' in keys, keys[:4]
    assert not any(k.startswith('model.') or k.startswith('lm_head') for k in keys)

    # kev's full-weight serving load and request path.
    ck = Checkpoint(FIXTURE)
    assert ck.full
    tok, model = ck.load('cpu', LoadOptions(dtype=torch.float64))
    assert model.hybrid and next(model.lm.parameters()).dtype == torch.float64
    assert model.head.temperature == TEMPERATURE
    request = SystemOneRequest.model_validate(REQUEST)
    rec, meta_q = to_record(request)
    enc = admit(model, tok, rec)
    state_ids, _, rows = rows_of(enc)
    with torch.no_grad():
        probs = [p.double() for p in model.probs(enc)]
    with open(LORA_ORACLE, encoding='utf-8') as f:
        lora_oracle = json.load(f)
    lora_gap = max(abs(a - b) for q, p in zip(lora_oracle['questions'], probs)
                   for a, b in zip(q['probs'], p.tolist()))
    oracle = {
        'kev_source': 'kev.checkpoint.Checkpoint.load (weights=full) + kev.model.admit + DecisionModel.probs',
        'settings': {'backbone_dtype': 'float64', 'saved_dtype': 'float32', 'temperature': TEMPERATURE,
                     'head_dim': POINTER, 'tensors': len(keys)},
        'request': json.dumps(REQUEST, ensure_ascii=False),
        'state_ids': list(state_ids),
        'questions': [{'id': m['id'], 'branch_ids': r['ids'], 'probs': p.tolist()}
                      for m, r, p in zip(meta_q, rows, probs)],
        'answers': to_answers([p.tolist() for p in probs], meta_q),
        'checks': {'gap_to_lora_fixture': lora_gap},
    }
    write_json(os.path.join(FIXTURE, 'kev_full_oracle.json'), oracle, indent=None)
    total = 0
    for name in sorted(os.listdir(FIXTURE)):
        size = os.path.getsize(os.path.join(FIXTURE, name))
        total += size
        print(f'  {os.path.join(FIXTURE, name)}: {size}')
    print(f'total {total} bytes; {len(keys)} tensors; gap to the LoRA fixture oracle {lora_gap:.2e}')
    for q in oracle['questions']:
        print(f"  {q['id']}: {[round(x, 4) for x in q['probs']]}")


if __name__ == '__main__':
    main()
