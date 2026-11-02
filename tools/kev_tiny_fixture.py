#!/usr/bin/env python3
"""Generate the Kev scorer parity fixture for tests/TestNeuralDecisionHeads.pas.

The oracle is kev's own code (jaredpalmer/kev, installed --no-deps in ~/x):
kev.model.DecisionModel builds the PEFT LoRA exactly as kev.train does, the
adapter and the pointer head are saved the kev way (save_pretrained +
kev.checkpoint.write_meta), and kev.checkpoint.Checkpoint(...).load() reloads
them (base in float64, LoRA merged, fp32 pointer head) before
kev.model.admit + DecisionModel.probs score the request (the serving path of
a hybrid backbone: the state once, then one causal row per question from its
cache). Nothing is downloaded: the base is a seeded RANDOM pico Qwen3.5
hybrid written locally, and head.pt names its directory as `base` (a local
path, which kev's loader accepts as a model name).

Writes tests/fixtures/kev_qwen3_5/:
  base/      config.json (Qwen3_5ForConditionalGeneration, text_config),
             model.safetensors (keys under model.language_model., tied
             embeddings), tokenizer.json, tokenizer_config.json. The tokenizer
             is tiny_bpe_split_qwen35_tokenizer.json plus an NFC normalizer
             and the five kev delimiters as special added tokens.
  adapter/   adapter_config.json + adapter_model.safetensors (PEFT, r=2,
             lora_alpha=4 on every kev hybrid target, random non-zero B) and
             head.pt (PointerHead d=16 -> dp=8, temperature 1.7).
  head_option_isolation.pt, head_special_embeddings.pt: the same head with
             one unsupported Meta flag set.
  kev_oracle.json: the request, the state and branch token ids, the hidden
             positions, kev's probabilities at the head temperature and at
             1.0, the answers, and the same head in float64 as a reference.

Coded by Claude (AI).

Usage (from the repo root, inside the ~/x venv):
  python3 tools/kev_tiny_fixture.py
"""
import json
import math
import os
import shutil

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from kev.api import SystemOneRequest, to_answers, to_record
from kev.checkpoint import Checkpoint, LoadOptions, Meta, write_meta
from kev.model import SPECIAL, DecisionModel, admit, rows_of

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FIXTURE = os.path.join('tests', 'fixtures', 'kev_qwen3_5')
BASE_DIR = os.path.join(FIXTURE, 'base')
ADAPTER_DIR = os.path.join(FIXTURE, 'adapter')
BASE_TOKENIZER = os.path.join('tests', 'fixtures', 'tiny_bpe_split_qwen35_tokenizer.json')

VOCAB = 408            # padded past the 405 tokenizer ids, like real Qwen
HIDDEN = 16
POINTER = 8            # dp
LORA_RANK = 2          # kev: lora_alpha = 2 * rank
TEMPERATURE = 1.7
WEIGHT_STD = 0.4
LORA_STD = 0.3
HEAD_STD = 0.15       # keeps the pointer logits near 1, away from saturation
FIRST_SPECIAL_ID = 400

TEXT_CONFIG = {
    'model_type': 'qwen3_5_text',
    'hidden_size': HIDDEN,
    'intermediate_size': 24,
    'num_hidden_layers': 4,
    'num_attention_heads': 2,
    'num_key_value_heads': 1,
    'head_dim': 8,
    'vocab_size': VOCAB,
    'max_position_embeddings': 4096,
    'rms_norm_eps': 1e-06,
    'attention_bias': False,
    'tie_word_embeddings': True,
    'hidden_act': 'silu',
    'layer_types': ['linear_attention', 'linear_attention', 'linear_attention',
                    'full_attention'],
    'linear_num_key_heads': 2,
    'linear_num_value_heads': 4,
    'linear_key_head_dim': 4,
    'linear_value_head_dim': 4,
    'linear_conv_kernel_dim': 4,
    'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.0,
                        'partial_rotary_factor': 0.25, 'mrope_section': [1, 0, 0],
                        'mrope_interleaved': True},
    'bos_token_id': None,
    'eos_token_id': 399,
    'pad_token_id': 398,
}

REQUEST = {
    'model': 'kev-parity',
    'state': {
        'ticket': 'Card declined twice; the app froze <|fim_suffix|> again.',
        'customer': {'name': 'Ana', 'plan': 'pro', 'seats': 12, 'verified': True},
        'history': ['refund asked in May', {'channel': 'chat', 'mood': 'upset'}],
    },
    'questions': {
        'team': {'type': 'choice', 'instructions': 'Which team takes it?',
                 'criteria': {'billing': 'payments and refunds', 'technical': None,
                              'sales': 'more <|box_end|> seats'}},
        'urgency': {'type': 'score', 'instructions': {'ask': 'How urgent?', 'scale': 'low to high'},
                    'criteria': ['low', 'medium', 'high', {'level': 'critical'}]},
        'angry': {'type': 'noul', 'instructions': 'Is the customer angry?',
                  'criteria': {'true': 'raised voice', 'false': 'calm'}},
        'plain': {'type': 'noul', 'instructions': 'Refund due?'},
    },
}


def added_token(token_id, content):
    return {'id': token_id, 'content': content, 'single_word': False, 'lstrip': False,
            'rstrip': False, 'normalized': False, 'special': True}


def write_json(path, data, indent=1):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write('\n')


def write_base():
    with open(BASE_TOKENIZER, encoding='utf-8') as f:
        tok = json.load(f)
    tok['normalizer'] = {'type': 'NFC'}
    assert max(tok['model']['vocab'].values()) == FIRST_SPECIAL_ID - 1
    tok['added_tokens'] += [added_token(FIRST_SPECIAL_ID + i, s) for i, s in enumerate(SPECIAL)]
    write_json(os.path.join(BASE_DIR, 'tokenizer.json'), tok, indent=None)
    write_json(os.path.join(BASE_DIR, 'tokenizer_config.json'), {
        'tokenizer_class': 'PreTrainedTokenizerFast',
        'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>',
        'added_tokens_decoder': {str(t['id']): {k: v for k, v in t.items() if k != 'id'}
                                 for t in tok['added_tokens']}})
    write_json(os.path.join(BASE_DIR, 'config.json'), {
        'architectures': ['Qwen3_5ForConditionalGeneration'], 'model_type': 'qwen3_5',
        'tie_word_embeddings': True, 'text_config': TEXT_CONFIG})
    config = Qwen3_5TextConfig(**{k: v for k, v in TEXT_CONFIG.items() if k != 'model_type'})
    model = Qwen3_5ForCausalLM(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith('norm.weight'):
                p.normal_(1.0, 0.2)
            elif name.endswith('A_log') or name.endswith('dt_bias'):
                p.normal_(0.0, 0.5)
            else:
                p.normal_(0.0, WEIGHT_STD)
    state = {'model.language_model.' + k[len('model.'):]: v.float().contiguous()
             for k, v in model.state_dict().items() if k.startswith('model.')}
    save_file(state, os.path.join(BASE_DIR, 'model.safetensors'), metadata={'format': 'pt'})
    return state


def main():
    os.chdir(ROOT)
    shutil.rmtree(FIXTURE, ignore_errors=True)
    os.makedirs(BASE_DIR)
    os.makedirs(ADAPTER_DIR)
    torch.manual_seed(20261019)
    written = write_base()

    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    raw_tok = Tokenizer.from_file(os.path.join(BASE_DIR, 'tokenizer.json'))
    assert [tok.convert_tokens_to_ids(s) for s in SPECIAL] == list(range(FIRST_SPECIAL_ID, FIRST_SPECIAL_ID + 5))

    # kev.train's construction: the base, PEFT on kev's hybrid targets, the head.
    trainer = DecisionModel(BASE_DIR, tok, 'cpu', lora=LORA_RANK, head_dim=POINTER)
    loaded = {('model.language_model.' + k): v for k, v in trainer.lm.base_model.model.state_dict().items()
              if 'lora_' not in k}
    for k, v in written.items():
        key = k.replace('.weight', '.base_layer.weight') if k not in loaded else k
        assert torch.equal(loaded[key], v), f'base reload differs at {k}'
    with torch.no_grad():
        for name, p in trainer.lm.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                p.normal_(0.0, LORA_STD)
        for p in trainer.head.parameters():
            p.normal_(0.0, HEAD_STD)
    trainer.lm.save_pretrained(ADAPTER_DIR)
    readme = os.path.join(ADAPTER_DIR, 'README.md')
    if os.path.exists(readme):
        os.remove(readme)
    meta = Meta(base=BASE_DIR, head=trainer.head.state_dict(), lora=LORA_RANK, head_dim=POINTER,
                temperature=TEMPERATURE, weights='lora')
    write_meta(ADAPTER_DIR, meta)
    for flag in ('option_isolation', 'special_embeddings'):
        refused = Meta.from_dict(meta.to_dict())
        setattr(refused, flag, True)
        torch.save(refused.to_dict(), os.path.join(FIXTURE, f'head_{flag}.pt'))

    # kev's serving load and request path.
    ck = Checkpoint(ADAPTER_DIR)
    tok, model = ck.load('cpu', LoadOptions(dtype=torch.float64))
    assert model.hybrid and next(model.lm.parameters()).dtype == torch.float64
    assert model.head.temperature == TEMPERATURE
    _, raw_model = ck.load('cpu', LoadOptions(dtype=torch.float64, temperature=1.0))
    _, base_model = ck.load('cpu', LoadOptions(dtype=torch.float64, lora_scale=0.0))
    request = SystemOneRequest.model_validate(REQUEST)
    rec, meta_q = to_record(request)
    enc = admit(model, tok, rec)
    state_ids, _, rows = rows_of(enc)
    with torch.no_grad():
        probs = [p.double() for p in model.probs(enc)]
        probs_raw = [p.double() for p in raw_model.probs(enc)]
        lora_effect = max((a - b.double()).abs().max().item()
                          for a, b in zip(probs, base_model.probs(enc)))
        assert lora_effect > 0.05, f'the adapter barely moves the answers ({lora_effect})'
        # Cross-check: the same rows recomputed from scratch (forward()).
        full = [torch.softmax(z.double(), -1) for z in model.forward(enc)]
        recompute_drift = max((a - b).abs().max().item() for a, b in zip(probs, full))
        # The same pointer head in float64 on kev's float64 hidden states.
        S, Sp, _ = rows_of(enc)
        head = {k: v.double() for k, v in model.head.state_dict().items()}
        probs_f64 = []
        for r in rows:
            h = model.lm(input_ids=torch.tensor([S + r['ids']]),
                         position_ids=torch.tensor([Sp + r['pos']])).last_hidden_state[0]
            q = h[len(S) + r['decide']] @ head['q.weight'].T + head['q.bias']
            k = h[[len(S) + o for o in r['opts']]] @ head['k.weight'].T + head['k.bias']
            probs_f64.append(torch.softmax(k @ q / math.sqrt(POINTER) / TEMPERATURE, -1))
    head_drift = max((a - b).abs().max().item() for a, b in zip(probs, probs_f64))
    answers = to_answers([p.tolist() for p in probs], meta_q)
    # The full-row text encodes like the raw tokenizers library.
    assert tok(rec['state'], add_special_tokens=False).input_ids == raw_tok.encode(rec['state'], add_special_tokens=False).ids
    oracle = {
        'kev_source': 'kev.checkpoint.Checkpoint.load + kev.model.admit + DecisionModel.probs',
        'settings': {'backbone_dtype': 'float64', 'head_dtype': 'float32', 'merge': True,
                     'temperature': TEMPERATURE, 'head_dim': POINTER, 'lora': LORA_RANK},
        'request': json.dumps(REQUEST, ensure_ascii=False),
        'state_text': rec['state'],
        'state_ids': list(state_ids),
        'questions': [{'id': m['id'], 'instructions': q['instr'], 'options': q['options'],
                       'branch_ids': r['ids'], 'positions': r['pos'],
                       'opt_positions': r['opts'], 'decide_position': r['decide'],
                       'probs': p.tolist(), 'probs_temperature_1': pr.tolist(),
                       'probs_f64_head': pf.tolist()}
                      for m, q, r, p, pr, pf in zip(meta_q, rec['questions'], rows, probs, probs_raw, probs_f64)],
        'answers': answers,
        'input_tokens': len(state_ids) + sum(len(r['ids']) for r in rows),
        'checks': {'lora_effect': lora_effect, 'recompute_drift': recompute_drift, 'f32_head_drift': head_drift},
    }
    write_json(os.path.join(FIXTURE, 'kev_oracle.json'), oracle, indent=None)
    total = 0
    for dirpath, _, files in os.walk(FIXTURE):
        for name in sorted(files):
            size = os.path.getsize(os.path.join(dirpath, name))
            total += size
            print(f'  {os.path.join(dirpath, name)}: {size}')
    print(f'total {total} bytes; state {len(state_ids)} tokens; branches '
          f'{[len(r["ids"]) for r in rows]}; recompute drift {recompute_drift:.2e}; '
          f'f32 head drift {head_drift:.2e}; LoRA moves p by {lora_effect:.3f}')
    for q in oracle['questions']:
        print(f"  {q['id']}: {[round(x, 4) for x in q['probs']]} raw {[round(x, 4) for x in q['probs_temperature_1']]}")


if __name__ == '__main__':
    main()
