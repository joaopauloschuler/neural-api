#!/usr/bin/env python3
"""Generate the Open-Jev scorer parity fixture for tests/TestNeuralDecisionHeads.pas.

The oracle is Open-Jev's own code (Zefan-Cai/Open-Jev, package `jev`, installed
--no-deps in ~/x): jev.api.compile_request + candidate_prompts build the
prompts, jev.model.DecisionModel.forward (the uncached path; score_cached needs the
transformers 5.10 cache classes jev pins) scores them, and jev.serving.Predictor.predict applies the temperature
and formats the answers. DecisionModel.__init__ is bypassed because it loads
a Hub vision-language checkpoint and seeds the head from single-token Yes/No
rows; the oracle sets the same attributes DecisionModel.load leaves (the
Qwen3_5TextModel backbone in float64 with the PEFT adapter attached, the fp32
Linear(d, 1) head loaded from head.pt, max_length from model.json).
Nothing is downloaded: the base is a seeded RANDOM pico Qwen3.5 hybrid, and
its tokenizer_config.json carries the verbatim Qwen/Qwen3.5-2B chat template
(revision 15852e8c, the base Open-Jev-2B pins).

Writes tests/fixtures/openjev_qwen3_5/:
  base/      config.json, model.safetensors (model.language_model. keys, tied
             embeddings), tokenizer.json (tiny_bpe_split_qwen35_tokenizer.json
             + NFC + <|im_start|> special, <think> and </think> not special,
             as in Qwen3.5), tokenizer_config.json.
  package/checkpoint/  the Open-Jev release layout: adapter/ (PEFT r=2,
             lora_alpha=4 on jev's targets, random non-zero B), head.pt
             (Linear(16, 1) state_dict, storages tagged 'cuda:0'), model.json
             (max_length 512), temperature.json.
  openjev_oracle.json: the request, per question the candidate prompts, the
             chat-template strings and their ids, jev's raw scores,
             probabilities at the package temperature and at 1.0,
             the answers, a float64-head reference, and json.dumps cases.

Coded by Claude (AI).

Usage (from the repo root, inside the ~/x venv):
  python3 tools/openjev_tiny_fixture.py
"""
import json
import os
import shutil

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5TextModel

from jev.api import _render, candidate_prompts, compile_request
from jev.model import DecisionModel
from jev.serving import Predictor, TorchScorer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FIXTURE = os.path.join('tests', 'fixtures', 'openjev_qwen3_5')
BASE_DIR = os.path.join(FIXTURE, 'base')
CHECKPOINT_DIR = os.path.join(FIXTURE, 'package', 'checkpoint')
ADAPTER_DIR = os.path.join(CHECKPOINT_DIR, 'adapter')
BASE_TOKENIZER = os.path.join('tests', 'fixtures', 'tiny_bpe_split_qwen35_tokenizer.json')
# Qwen/Qwen3.5-2B chat_template.jinja at revision 15852e8c16360a2fea060d615a32b45270f8a8fc,
# passed as an argument (fetched beside the fixture, never committed).
TEMPLATE_ENV = 'QWEN35_2B_CHAT_TEMPLATE'

VOCAB = 408
HIDDEN = 16
LORA_RANK = 2          # jev: lora_alpha = 2 * rank
MAX_LENGTH = 512
TEMPERATURE = 1.37
WEIGHT_STD = 0.4
LORA_STD = 0.3
HEAD_STD = 1.0
FIRST_ADDED_ID = 400
ADDED = [('<|im_start|>', True), ('<think>', False), ('</think>', False)]

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
    'eos_token_id': 1,
    'pad_token_id': 0,
}

REQUEST = {
    'model': 'open-jev-parity',
    'state': {
        'ticket': 'Card declined twice; the app froze again.\nCustomer: "fix it"',
        'customer': {'name': 'Ana Lúcia', 'plan': 'pro', 'seats': 12, 'verified': True,
                     'balance': -12.5, 'ratio': 1e-07},
        'history': ['refund asked in May', {'mood': 'upset', 'channel': 'chat'}],
        'Notes': None,
    },
    'questions': {
        'team': {'type': 'choice', 'instructions': 'Which team takes it?',
                 'criteria': {'billing': 'payments and refunds', 'technical': None,
                              'sales': {'scope': 'seats', 'max': 50}, 'legal': ''}},
        'urgency': {'type': 'score', 'instructions': {'scale': 'low to high', 'ask': 'How urgent?'},
                    'criteria': ['low', 'medium', 'high', {'level': 'critical'}]},
        'angry': {'type': 'noul', 'instructions': 'Is the customer angry?',
                  'criteria': {'false': 'calm', 'true': 'raised voice'}},
        'plain': {'type': 'noul', 'instructions': 'Refund due?'},
        'only_false': {'type': 'noul', 'instructions': 'Escalate?',
                       'criteria': {'false': ['wait', 'monitor']}},
        'single': {'type': 'choice', 'instructions': 'Pick one.',
                   'criteria': {'only': 'the one option'}},
    },
}

BS = chr(92)  # a backslash, kept out of the literals below


def dumps_cases():
    """JSON texts and jev's _render of them (json.dumps(sort_keys=True))."""
    u = BS + 'u'
    escaped = ('"q' + BS + '"b' + BS + BS + 's' + BS + '/n' + BS + 'n' + BS + 't' + BS + 'r' +
               BS + 'b' + BS + 'f' + u + '0001' + u + '001f' + u + '007f' + u + '2028' +
               u + '00e9' + u + 'd83d' + u + 'de00 end"')
    texts = [
        '{"b": 1, "a": [true, false, null], "B": {"y": 2.5, "x": "q"}}',
        '{"é": 1, "z": 2, "Z": 3, "aa": 4, "a": 5, "": 6, "\U0001F600": 7, "ÿ": 8, "_": 9}',
        '[1e16, 1.5e-7, 0.1, 100.0, 1E2, -0.0, 5e-324, 1.7976931348623157e308, 123456789,'
        ' -42, 0.30000000000000004, 1e-5, 0.0001, 9007199254740993, 1.0e15, 2.5E+3, 0]',
        '[1e-323, 2.5e-323, 9.5e-323, 1e-310, 2.2250738585072014e-308, 4.9e-324, 9.8e-324]',
        '{"k": ' + escaped + '}',
        '{"nested": {"empty_obj": {}, "empty_list": [], "list": [{"b": 1, "a": 2}, "x", [1, [2]]]}}',
        '"plain text' + BS + 'nsecond line"',
        '{"a' + BS + '"b": 1, "a' + BS + 'nb": 2, "a b": 3}',
        '[]',
        '{}',
    ]
    return [[text, _render(json.loads(text))] for text in texts]


def added_token(token_id, content, special):
    return {'id': token_id, 'content': content, 'single_word': False, 'lstrip': False,
            'rstrip': False, 'normalized': False, 'special': special}


def write_json(path, data, indent=1):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write('\n')


def write_base(chat_template):
    with open(BASE_TOKENIZER, encoding='utf-8') as f:
        tok = json.load(f)
    tok['normalizer'] = {'type': 'NFC'}
    assert max(tok['model']['vocab'].values()) == FIRST_ADDED_ID - 1
    tok['added_tokens'] += [added_token(FIRST_ADDED_ID + i, s, special)
                            for i, (s, special) in enumerate(ADDED)]
    write_json(os.path.join(BASE_DIR, 'tokenizer.json'), tok, indent=None)
    write_json(os.path.join(BASE_DIR, 'tokenizer_config.json'), {
        'tokenizer_class': 'PreTrainedTokenizerFast', 'add_bos_token': False, 'bos_token': None,
        'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>',
        'added_tokens_decoder': {str(t['id']): {k: v for k, v in t.items() if k != 'id'}
                                 for t in tok['added_tokens']},
        'chat_template': chat_template})
    write_json(os.path.join(BASE_DIR, 'config.json'), {
        'architectures': ['Qwen3_5ForConditionalGeneration'], 'model_type': 'qwen3_5',
        'tie_word_embeddings': True, 'text_config': TEXT_CONFIG})
    config = Qwen3_5TextConfig(**{k: v for k, v in TEXT_CONFIG.items() if k != 'model_type'})
    model = Qwen3_5TextModel(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith('norm.weight'):
                p.normal_(1.0, 0.2)
            elif name.endswith('A_log') or name.endswith('dt_bias'):
                p.normal_(0.0, 0.5)
            else:
                p.normal_(0.0, WEIGHT_STD)
    state = {'model.language_model.' + k: v.float().contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(BASE_DIR, 'model.safetensors'), metadata={'format': 'pt'})
    return config


def load_backbone(config, dtype):
    model = Qwen3_5TextModel(config)
    prefix = 'model.language_model.'
    weights = {k[len(prefix):]: v for k, v in load_file(os.path.join(BASE_DIR, 'model.safetensors')).items()}
    model.load_state_dict(weights, strict=True)
    return model.to(dtype)


def save_head_tagged_cuda(head, path):
    """torch.save with every storage location written as 'cuda:0' (CPU bytes)."""
    registry = torch.serialization._package_registry
    before = list(registry)
    torch.serialization.register_package(0, lambda storage: 'cuda:0', lambda storage, location: None)
    try:
        torch.save(head.state_dict(), path)
    finally:
        registry[:] = before


def oracle_model(config, tok, lora_scale=1.0):
    """The attributes DecisionModel.load leaves, on a float64 backbone."""
    model = DecisionModel.__new__(DecisionModel)
    nn.Module.__init__(model)
    model.model_id, model.revision = BASE_DIR, None
    model.max_length, model.device_name = MAX_LENGTH, 'cpu'
    model.tokenizer = tok
    model.sharded = False
    model.backbone = PeftModel.from_pretrained(load_backbone(config, torch.float64), ADAPTER_DIR)
    if lora_scale != 1.0:
        for module in model.backbone.modules():
            if hasattr(module, 'scaling') and isinstance(module.scaling, dict):
                module.scaling = {k: v * lora_scale for k, v in module.scaling.items()}
    model.lora_rank = LORA_RANK
    model.head = nn.Linear(HIDDEN, 1, bias=True, dtype=torch.float32)
    model.head.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'head.pt'),
                                          map_location='cpu', weights_only=True))
    return model.eval()


def answer_probs(response, record):
    answer = response['answers'][record['id']]
    if record['kind'] == 'noul':
        return [1.0 - answer['noul'], answer['noul']]
    return list(answer['probabilities'].values())


def main():
    template_file = os.environ.get(TEMPLATE_ENV)
    if not template_file:
        raise SystemExit(f'set {TEMPLATE_ENV} to the Qwen/Qwen3.5-2B chat_template.jinja')
    with open(template_file, encoding='utf-8') as f:
        chat_template = f.read()
    os.chdir(ROOT)
    shutil.rmtree(FIXTURE, ignore_errors=True)
    os.makedirs(BASE_DIR)
    os.makedirs(ADAPTER_DIR)
    torch.manual_seed(20261019)
    config = write_base(chat_template)
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    tok.padding_side = 'right'
    assert tok.convert_tokens_to_ids('<|im_start|>') == FIRST_ADDED_ID

    # jev's training construction: PEFT on the Qwen3_5TextModel backbone, jev's targets.
    backbone = load_backbone(config, torch.float32)
    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'out_proj']
    available = {name.rsplit('.', 1)[-1] for name, _ in backbone.named_modules()}
    targets = [name for name in targets if name in available]
    assert len(targets) == 6, targets
    peft_model = get_peft_model(backbone, LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK * 2,
                                                     target_modules=targets, lora_dropout=0.0,
                                                     bias='none'))
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                p.normal_(0.0, LORA_STD)
    peft_model.save_pretrained(ADAPTER_DIR)
    readme = os.path.join(ADAPTER_DIR, 'README.md')
    if os.path.exists(readme):
        os.remove(readme)
    head = nn.Linear(HIDDEN, 1, bias=True, dtype=torch.float32)
    with torch.no_grad():
        head.weight.normal_(0.0, HEAD_STD)
        head.bias.normal_(0.0, HEAD_STD)
    save_head_tagged_cuda(head, os.path.join(CHECKPOINT_DIR, 'head.pt'))
    with open(os.path.join(CHECKPOINT_DIR, 'head.pt'), 'rb') as f:
        assert b'cuda:0' in f.read()
    write_json(os.path.join(CHECKPOINT_DIR, 'model.json'), {
        'model_id': 'Qwen/Qwen3.5-2B', 'revision': '15852e8c16360a2fea060d615a32b45270f8a8fc',
        'max_length': MAX_LENGTH, 'lora_rank': LORA_RANK,
        'method': 'independent_candidate_lora_nll_brier'}, indent=2)
    write_json(os.path.join(CHECKPOINT_DIR, 'temperature.json'), {
        'temperature': TEMPERATURE, 'split': 'calibration', 'n': 512,
        'ids_sha256': '0' * 64}, indent=2)

    model = oracle_model(config, tok)
    base_model = oracle_model(config, tok, lora_scale=0.0)
    records = compile_request(REQUEST['state'], REQUEST['questions'])
    with torch.no_grad():
        uncached = [row.double() for row in model(records)]
        base_rows = [row.double() for row in base_model(records)]
    lora_effect = max((a - b).abs().max().item() for a, b in zip(uncached, base_rows))
    assert lora_effect > 0.05, f'the adapter barely moves the scores ({lora_effect})'
    predictor = Predictor(TorchScorer(model), model_name='open-jev-parity', temperature=TEMPERATURE)
    response = predictor.predict(REQUEST)
    raw_response = Predictor(TorchScorer(model), model_name='open-jev-parity',
                             temperature=1.0).predict(REQUEST)

    head64 = {k: v.double() for k, v in model.head.state_dict().items()}
    questions = []
    prompt_lengths = []
    for record, row in zip(records, uncached):
        prompts = candidate_prompts(record)
        rendered = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
                    for p in prompts]
        ids = [tok(r).input_ids for r in rendered]
        assert ids == [tok(r, add_special_tokens=False).input_ids for r in rendered]
        prompt_lengths += [len(x) for x in ids]
        with torch.no_grad():
            scores64 = []
            for x in ids:
                h = model.backbone(input_ids=torch.tensor([x]), use_cache=False).last_hidden_state[0, -1]
                scores64.append((h @ head64['weight'][0] + head64['bias'][0]).item())
        questions.append({
            'id': record['id'], 'kind': record['kind'], 'question': record['question'],
            'options': record['options'], 'prompts': prompts, 'rendered': rendered, 'ids': ids,
            'scores': row.tolist(), 'scores_f64_head': scores64,
            'probs': answer_probs(response, record),
            'probs_temperature_1': answer_probs(raw_response, record),
        })
    assert max(prompt_lengths) < MAX_LENGTH
    oracle = {
        'jev_source': 'jev.api.compile_request/candidate_prompts + DecisionModel.forward/score_cached'
                      ' + jev.serving.Predictor.predict',
        'settings': {'backbone_dtype': 'float64', 'head_dtype': 'float32', 'temperature': TEMPERATURE,
                     'max_length': MAX_LENGTH, 'lora': LORA_RANK, 'hidden': HIDDEN},
        'request': json.dumps(REQUEST, ensure_ascii=False),
        'state_rendered': _render(REQUEST['state']),
        'questions': questions,
        'response': response,
        'response_temperature_1': raw_response,
        'logical_input_tokens': sum(prompt_lengths),
        'checks': {'lora_effect': lora_effect},
        'json_dumps_cases': dumps_cases(),
    }
    write_json(os.path.join(FIXTURE, 'openjev_oracle.json'), oracle, indent=None)
    total = 0
    for dirpath, _, files in os.walk(FIXTURE):
        for name in sorted(files):
            size = os.path.getsize(os.path.join(dirpath, name))
            total += size
            print(f'  {os.path.join(dirpath, name)}: {size}')
    print(f'total {total} bytes; prompts {prompt_lengths}; LoRA moves s by {lora_effect:.3f}')
    for q in questions:
        print(f"  {q['id']}: s {[round(x, 4) for x in q['scores']]} p {[round(x, 4) for x in q['probs']]}")


if __name__ == '__main__':
    main()
