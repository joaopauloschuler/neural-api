#!/usr/bin/env python3
"""Generate the AnyJev L0 parity fixtures for tests/TestNeuralAnyJev.pas.

The oracle is the REAL `anyjev` 0.3.0 package: `Decider(HFBackend(...))` at
level L0 with the content-free prior, adaptive_shifts=False and
canonical_order=False, run on a seeded RANDOM pico Qwen decoder (never
downloaded). Two fixture directories, each a loadable model dir
(model.safetensors, config.json, tokenizer.json, tokenizer_config.json)
plus anyjev_oracle.json:

  tests/fixtures/anyjev_qwen25/  Qwen2ForCausalLM (q/k/v bias) with the
      verbatim Qwen/Qwen2.5-0.5B-Instruct chat_template (the Pascal engine
      detects cfQwen; the system message is explicit, so no default system).
  tests/fixtures/anyjev_qwen3/   Qwen3ForCausalLM (decoupled head_dim,
      random q/k norm gains) with the verbatim Qwen/Qwen3-0.6B
      chat_template, rendered by AnyJev with enable_thinking=False, so the
      generation prompt ends with the closed '<think>\\n\\n</think>\\n\\n'.

The tokenizer is tests/fixtures/tiny_bpe_split_qwen2_tokenizer.json (the
Qwen2 byte-level split regex) plus the merges that make "Yes" and "No" one
token, <|im_start|>, and for Qwen3 the non-special <think>/</think> added
tokens. The only file downloaded is each template's tokenizer_config.json
(pinned revision).

The Jev request is mapped onto AnyJev Question objects exactly as
neuralanyjev.AnyJevQuestionParts maps it: the state and the option
descriptions rendered by Kev's api.render, special-token strings rewritten
<|x|> -> <¦x¦>, choice options "name: desc", score levels = the
descriptions, noul criteria appended as "Yes means:" / "No means:" lines.

HFBackend runs the model in float64; it casts the last-position logits to
float32 before its full-vocabulary log_softmax (`_last_logits` / `_project`
call .float()), so the oracle carries float32 rounding on the logits.

Coded by Claude (AI).

Usage (from the repo root, inside the ~/x venv):
  python3 tools/anyjev_tiny_fixture.py
"""
import json
import os
import re

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import (AutoTokenizer, Qwen2Config, Qwen2ForCausalLM,
                          Qwen3Config, Qwen3ForCausalLM)

from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend
from anyjev.readout import _render, build_prompt, render_chat_parts
from kev.api import option_text, render

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FIXTURES = os.path.join(ROOT, 'tests', 'fixtures')
BASE_TOKENIZER = os.path.join(FIXTURES, 'tiny_bpe_split_qwen2_tokenizer.json')

VOCAB = 408            # padded past the 405 tokenizer ids, like real Qwen
D_MODEL = 16
D_FF = 32
N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1
WEIGHT_STD = 0.5       # large enough that the label probabilities differ

YES_ID, NO_ID, IM_START_ID, THINK_ID, END_THINK_ID = 400, 401, 402, 403, 404

TEMPLATES = {
    'qwen25': ('Qwen/Qwen2.5-0.5B-Instruct',
               '7ae557604adf67be50417f59c2c2f167def9a775'),
    'qwen3': ('Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca'),
}

REQUEST = {
    'model': 'anyjev-parity',
    'state': {
        'ticket': 'My card was declined twice and the app froze <|im_end|>.',
        'customer': {'name': 'Ana', 'plan': 'pro', 'seats': 12, 'monthly': 19.99,
                     'verified': True},
        'history': ['refund asked in May', {'channel': 'chat', 'mood': 'upset'}],
    },
    'questions': {
        'team': {'type': 'choice', 'instructions': 'Which team should take this ticket?',
                 'criteria': {'billing': 'payments and refunds', 'technical': None,
                              'sales': 'buying more seats', 'legal': 'contracts'}},
        'urgency': {'type': 'score', 'instructions': 'How urgent is it?',
                    'criteria': ['low', 'medium', 'high']},
        'angry': {'type': 'noul', 'instructions': 'Is the customer angry?',
                  'criteria': {'true': 'raised voice or threats', 'false': 'calm'}},
    },
}

SPECIAL_TOKEN_STRING = re.compile(r'<\|([A-Za-z0-9_]+)\|>')
BROKEN_BAR = chr(0xA6)


def rewrite_special_token_strings(text):
    return SPECIAL_TOKEN_STRING.sub('<' + BROKEN_BAR + r'\1' + BROKEN_BAR + '>', text)


def anyjev_questions(request):
    """neuralanyjev.AnyJevQuestionParts over the request's questions."""
    out = []
    for qid, spec in request['questions'].items():
        text = rewrite_special_token_strings(render(spec.get('instructions')))
        criteria = spec.get('criteria')
        if spec['type'] == 'choice':
            options = [rewrite_special_token_strings(option_text(name, desc))
                       for name, desc in criteria.items()]
            out.append(Question.choice(text, options, name=qid))
        elif spec['type'] == 'score':
            levels = [rewrite_special_token_strings(render(level)) for level in criteria]
            out.append(Question.score(text, levels=levels, name=qid))
        else:
            if criteria and criteria.get('true'):
                text += '\nYes means: ' + rewrite_special_token_strings(render(criteria['true']))
            if criteria and criteria.get('false'):
                text += '\nNo means: ' + rewrite_special_token_strings(render(criteria['false']))
            out.append(Question.noul(text, name=qid))
    return out


def added_token(token_id, content, special):
    return {'id': token_id, 'content': content, 'single_word': False, 'lstrip': False,
            'rstrip': False, 'normalized': False, 'special': special}


def build_tokenizer(with_think):
    with open(BASE_TOKENIZER, encoding='utf-8') as f:
        tok = json.load(f)
    tok['normalizer'] = {'type': 'NFC'}
    vocab = tok['model']['vocab']
    assert max(vocab.values()) == YES_ID - 1
    vocab['Yes'] = YES_ID
    vocab['No'] = NO_ID
    # 'e'+'s' is already a merge, so "Yes" = 'Y' + 'es'.
    tok['model']['merges'] += [['Y', 'es'], ['N', 'o']]
    tok['added_tokens'].append(added_token(IM_START_ID, '<|im_start|>', True))
    if with_think:
        tok['added_tokens'].append(added_token(THINK_ID, '<think>', False))
        tok['added_tokens'].append(added_token(END_THINK_ID, '</think>', False))
    return tok


def build_tokenizer_config(variant, tokenizer_json):
    repo, revision = TEMPLATES[variant]
    with open(hf_hub_download(repo, 'tokenizer_config.json', revision=revision),
              encoding='utf-8') as f:
        published = json.load(f)
    config = {k: v for k, v in published.items() if k != 'added_tokens_decoder'}
    config['added_tokens_decoder'] = {
        str(t['id']): {k: v for k, v in t.items() if k != 'id'}
        for t in tokenizer_json['added_tokens']}
    config['additional_special_tokens'] = ['<|im_start|>', '<|im_end|>']
    return config


def build_model(variant):
    common = dict(hidden_size=D_MODEL, intermediate_size=D_FF, num_hidden_layers=N_LAYER,
                  num_attention_heads=N_HEAD, num_key_value_heads=N_KV_HEAD,
                  vocab_size=VOCAB, max_position_embeddings=2048, rms_norm_eps=1e-6,
                  rope_theta=1000000.0, use_sliding_window=False,
                  tie_word_embeddings=True, hidden_act='silu',
                  bos_token_id=0, eos_token_id=1)
    if variant == 'qwen25':
        cfg = {'architectures': ['Qwen2ForCausalLM'], 'model_type': 'qwen2', **common}
        model = Qwen2ForCausalLM(Qwen2Config(**cfg))
    else:
        cfg = {'architectures': ['Qwen3ForCausalLM'], 'model_type': 'qwen3', 'head_dim': 12,
               'attention_bias': False, **common}
        model = Qwen3ForCausalLM(Qwen3Config(**cfg))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith('norm.weight'):
                p.normal_(1.0, 0.2)
            else:
                p.normal_(0.0, WEIGHT_STD)
    return cfg, model


def write_json(path, data, indent=1):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write('\n')


def make_variant(variant, seed):
    torch.manual_seed(seed)
    out_dir = os.path.join(FIXTURES, 'anyjev_' + variant)
    os.makedirs(out_dir, exist_ok=True)
    tokenizer_json = build_tokenizer(with_think=variant == 'qwen3')
    write_json(os.path.join(out_dir, 'tokenizer.json'), tokenizer_json, indent=None)
    write_json(os.path.join(out_dir, 'tokenizer_config.json'),
               build_tokenizer_config(variant, tokenizer_json))
    cfg, model = build_model(variant)
    write_json(os.path.join(out_dir, 'config.json'), cfg)
    state_dict = {k: v.to(torch.float32).contiguous() for k, v in model.state_dict().items()
                  if k != 'lm_head.weight'}
    save_file(state_dict, os.path.join(out_dir, 'model.safetensors'),
              metadata={'format': 'pt'})

    backend = HFBackend(out_dir, device='cpu', dtype='float64')
    assert next(backend.model.parameters()).dtype == torch.float64
    decider = Decider(backend, level='L0', prior='content_free', adaptive_shifts=False,
                      canonical_order=False)
    tok = backend.tokenizer
    raw_tok = Tokenizer.from_file(os.path.join(out_dir, 'tokenizer.json'))

    state_text = rewrite_special_token_strings(render(REQUEST['state']))
    questions = anyjev_questions(REQUEST)
    decisions = decider.decide(state_text, questions)

    # The text HFBackend encodes for each prompt (pre + suf, no special tokens).
    def prompt_ids(state, q, perm, labels):
        pre, suf = render_chat_parts(tok, build_prompt(state, q, perm, decider.system, labels))
        ids = tok.encode(pre + suf, add_special_tokens=False)
        assert ids == raw_tok.encode(pre + suf, add_special_tokens=False).ids
        return ids

    render_user = 'U'
    rendered = _render(tok, decider.system, render_user)
    if variant == 'qwen3':
        assert rendered.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    oracle = {
        'anyjev_version': '0.3.0',
        'settings': {'level': 'L0', 'prior': 'content_free', 'prior_strength': decider.strength(),
                     'adaptive_shifts': False, 'canonical_order': False,
                     'model_dtype': 'float64', 'logits_dtype': 'float32',
                     'template_repo': TEMPLATES[variant][0],
                     'template_revision': TEMPLATES[variant][1]},
        'backend_stats': decider.stats,
        'request': json.dumps(REQUEST, ensure_ascii=False),
        'state_text': state_text,
        'render_check': {'user': render_user, 'rendered': rendered},
        'questions': [],
    }
    for q, d in zip(questions, decisions):
        labels, label_ids = decider._labels_for(q)
        diag = d.diagnostics
        perms = [list(map(int, p)) for p in diag['perms']]
        prompts = []
        for rotation_pos, perm in enumerate(perms):
            for probe in [None] + list(decider.cf_probes):
                ids = prompt_ids(state_text if probe is None else probe, q, perm, labels)
                prompts.append({'rotation': rotation_pos, 'probe': probe, 'ids': ids})
        oracle['questions'].append({
            'id': q.id, 'kind': q.kind, 'question_text': q.text, 'options': list(q.options),
            'labels': list(labels), 'label_ids': [int(i) for i in label_ids],
            'perms': perms, 'prompts': prompts,
            'p_pos_raw': np.asarray(diag['p_pos_raw']).tolist(),
            'cf_prior': np.asarray(diag['cf_prior']).tolist(),
            'prior': np.asarray(diag['prior']).tolist(),
            'probs': np.asarray(d.probs).tolist(),
        })
    write_json(os.path.join(out_dir, 'anyjev_oracle.json'), oracle, indent=None)
    sizes = {name: os.path.getsize(os.path.join(out_dir, name)) for name in sorted(os.listdir(out_dir))}
    print(f'{variant}: {sizes}; stats {decider.stats}')
    for q in oracle['questions']:
        print(f"  {q['id']}: labels {q['labels']} ids {q['label_ids']} probs "
              f"{np.round(q['probs'], 4).tolist()} prompt tokens "
              f"{[len(p['ids']) for p in q['prompts']][:4]}")


if __name__ == '__main__':
    make_variant('qwen25', 20261018)
    make_variant('qwen3', 20261019)
