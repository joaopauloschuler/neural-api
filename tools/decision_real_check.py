#!/usr/bin/env python3
"""Real-checkpoint parity check: ChatServer --decision-adapter against kev's or
Open-Jev's own scoring code on the same base and adapter.

The script sends one fixed Jev request (a choice, a score and a noul question;
model "jev-latest") to a running ChatServer launched as
  ChatServer <base> --decision-adapter <adapter> --fp32 ...
then scores the same request in Python and prints both probability vectors
per question (choice and score in request order, noul as (no, yes)).

The layout is detected the way ChatServer does it:
  - Open-Jev (head.pt + temperature.json in <adapter>, <adapter>/checkpoint or
    <adapter>/package/checkpoint): jev.api.compile_request + candidate_prompts,
    jev.model.DecisionModel.forward (the uncached path: jev's score_cached
    needs the transformers 5.10 cache classes jev pins) and
    jev.serving.Predictor.predict. DecisionModel.__init__ is bypassed (it loads
    a Hub vision-language checkpoint and seeds the head from the LM head); the
    script sets what DecisionModel.load leaves: the Qwen3_5TextModel backbone
    of <base> in float32 with <checkpoint>/adapter attached (PEFT, unmerged),
    the Linear(d, 1) head from head.pt, max_length from model.json.
  - Kev LoRA (head.pt + adapter_config.json in <adapter>):
    kev.checkpoint.Checkpoint(<adapter>).load('cpu', LoadOptions(float32)),
    kev's exact path (LoRA merged in fp32), then kev.api.to_record +
    kev.model.admit + DecisionModel.probs. head.pt's `base` is replaced by
    <base>, so kev reads the same local files as the server (nothing is
    downloaded).
  - Kev full weights (head.pt + config.json + model*.safetensors, kev-27b):
    the same kev path; Checkpoint.load takes the backbone from <adapter>
    itself (nothing merged) and the tokenizer from <base>. ChatServer is
    launched with that directory as both <model-dir> and --decision-adapter,
    so pass it twice here too.

Launch ChatServer with --fp32: its default int8 weights move the
probabilities by far more than the FP32 summation-order noise this check
bounds. The default threshold 5e-3 is the max absolute probability
difference allowed: FP32 against FP32 through 24 layers is expected near
1e-4..1e-3, the server and jev round to 4 decimals (5e-5 each), and a real
mismatch (a wrong token, row position, missing LoRA module or temperature)
moves probabilities by 1e-2 or more. Pass the same --temperature to both
sides when the server ran with --decision-temperature.

Needs torch, transformers, peft and safetensors, plus kev (Kev) or jev
(Open-Jev), installed --no-deps:
  pip install --no-deps git+https://github.com/jaredpalmer/kev@1f04cfa28e6a708485b28a9585010b9e436b776e
  pip install --no-deps git+https://github.com/Zefan-Cai/Open-Jev@bd4118882f733574a3250a4b65fe4d884130c08b
HTTP uses only the standard library.

Usage:
  python3 tools/decision_real_check.py <base> <adapter> --server http://127.0.0.1:8080

Coded by Claude (AI).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_THRESHOLD = 5e-3
CONNECT_ATTEMPTS = 30      # the server may still be loading the checkpoint
CONNECT_RETRY_SECONDS = 10

REQUEST = {
    'model': 'jev-latest',
    'state': {
        'ticket': 'My card was declined twice and the app froze when I tried again.',
        'customer': {'plan': 'pro', 'seats': 12, 'verified': True},
        'history': ['refund asked in May', 'two calls this week'],
    },
    'questions': {
        'team': {'type': 'choice', 'instructions': 'Which team should take this ticket?',
                 'criteria': {'billing': 'payments and refunds',
                              'technical': 'app bugs and crashes', 'sales': None}},
        'urgency': {'type': 'score', 'instructions': 'How urgent is it?',
                    'criteria': ['low', 'medium', 'high']},
        'angry': {'type': 'noul', 'instructions': 'Is the customer angry?',
                  'criteria': {'true': 'raised voice or threats', 'false': 'calm'}},
    },
}


def post_systemone(server, request, timeout):
    """The server's JSON answer; retries while nothing listens yet."""
    body = json.dumps(request, ensure_ascii=False).encode('utf-8')
    url = server.rstrip('/') + '/v1/systemone'
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        req = urllib.request.Request(url, data=body, method='POST',
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            sys.exit(f'{url} answered HTTP {e.code}: {e.read().decode("utf-8", "replace")}')
        except (urllib.error.URLError, ConnectionError) as e:
            if attempt == CONNECT_ATTEMPTS:
                sys.exit(f'{url} unreachable after {attempt} attempts: {e}')
            print(f'[{url} not answering yet ({e}); retry {attempt}/{CONNECT_ATTEMPTS - 1}'
                  f' in {CONNECT_RETRY_SECONDS} s]', flush=True)
            time.sleep(CONNECT_RETRY_SECONDS)


def answer_probabilities(request, answers):
    """Per question id, the probabilities of a Jev response in request option order."""
    out = {}
    for qid, spec in request['questions'].items():
        answer = answers[qid]
        if spec['type'] == 'choice':
            out[qid] = [answer['probabilities'][name] for name in spec['criteria']]
        elif spec['type'] == 'score':
            out[qid] = [answer['probabilities'][str(level)]
                        for level in range(len(spec['criteria']))]
        else:
            out[qid] = [1.0 - answer['noul'], answer['noul']]
    return out


def option_labels(spec):
    if spec['type'] == 'choice':
        return list(spec['criteria'])
    if spec['type'] == 'score':
        return [str(level) for level in spec['criteria']]
    return ['no', 'yes']


def openjev_checkpoint_dir(adapter):
    """neuraldecisionheads.OpenJevCheckpointDir: the directory holding head.pt and temperature.json."""
    for sub in ('', 'checkpoint', os.path.join('package', 'checkpoint')):
        candidate = os.path.join(adapter, sub)
        if all(os.path.isfile(os.path.join(candidate, f)) for f in ('head.pt', 'temperature.json')):
            return candidate
    return None


def load_text_backbone(base, dtype):
    """<base>'s text decoder (Qwen3_5TextModel for a Qwen3.5 checkpoint) with its weights."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel
    text_config = AutoConfig.from_pretrained(base).get_text_config()
    backbone = AutoModel.from_config(text_config).to(dtype)
    index = os.path.join(base, 'model.safetensors.index.json')
    if os.path.isfile(index):
        with open(index, encoding='utf-8') as f:
            shards = sorted(set(json.load(f)['weight_map'].values()))
    else:
        shards = ['model.safetensors']
    expected = set(backbone.state_dict())
    weights = {}
    for shard in shards:
        for key, value in load_file(os.path.join(base, shard)).items():
            for prefix in ('model.language_model.', 'model.', ''):
                if key.startswith(prefix) and key[len(prefix):] in expected:
                    weights[key[len(prefix):]] = value.to(dtype)
                    break
    backbone.load_state_dict(weights, strict=True)
    return backbone.eval()


def openjev_reference(base, checkpoint, request, temperature, dtype):
    import torch
    from peft import PeftModel
    from torch import nn
    from transformers import AutoTokenizer
    from jev.model import DecisionModel
    from jev.serving import Predictor, TorchScorer

    with open(os.path.join(checkpoint, 'model.json'), encoding='utf-8') as f:
        config = json.load(f)
    if temperature is None:
        with open(os.path.join(checkpoint, 'temperature.json'), encoding='utf-8') as f:
            temperature = json.load(f)['temperature']
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = 'right'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = DecisionModel.__new__(DecisionModel)
    nn.Module.__init__(model)
    model.model_id, model.revision = base, None
    model.max_length, model.device_name = config['max_length'], 'cpu'
    model.tokenizer, model.sharded = tok, False
    model.backbone = load_text_backbone(base, getattr(torch, dtype))
    model.lora_rank = config['lora_rank']
    if model.lora_rank:
        model.backbone = PeftModel.from_pretrained(model.backbone, os.path.join(checkpoint, 'adapter'))
    head = torch.load(os.path.join(checkpoint, 'head.pt'), map_location='cpu', weights_only=True)
    model.head = nn.Linear(head['weight'].shape[1], 1, bias=True, dtype=torch.float32)
    model.head.load_state_dict(head)
    model.eval()
    response = Predictor(TorchScorer(model), model_name='open-jev-check',
                         temperature=temperature).predict(request)
    return answer_probabilities(request, response['answers']), temperature


def kev_reference(base, adapter, request, temperature, dtype):
    import torch
    from kev.api import SystemOneRequest, to_record
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import admit

    ck = Checkpoint(adapter)
    print(f'[head.pt records base {ck.meta.base} at revision {ck.meta.base_revision};'
          f' reading {base} instead]', flush=True)
    ck.meta.base, ck.meta.base_revision = base, None
    tok, model = ck.load('cpu', LoadOptions(dtype=getattr(torch, dtype), temperature=temperature))
    rec, meta = to_record(SystemOneRequest.model_validate(request))
    enc = admit(model, tok, rec)
    with torch.no_grad():
        probs = [p.double().tolist() for p in model.probs(enc)]
    return {m['id']: p for m, p in zip(meta, probs)}, model.head.temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('base', help='the base checkpoint ChatServer loads (its <model-dir>)')
    parser.add_argument('adapter', help='the --decision-adapter directory')
    parser.add_argument('--server', default='http://127.0.0.1:8080')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--temperature', type=float, default=None,
                        help="override the checkpoint's temperature (as --decision-temperature)")
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32',
                        help='the Python reference backbone dtype (bfloat16 halves its RAM:'
                             ' 27B checkpoints; the threshold must then be loosened)')
    parser.add_argument('--timeout', type=float, default=1800,
                        help='seconds one /v1/systemone request may take')
    args = parser.parse_args()

    checkpoint = openjev_checkpoint_dir(args.adapter)
    kev_lora = os.path.isfile(os.path.join(args.adapter, 'adapter_config.json'))
    kev_full = os.path.isfile(os.path.join(args.adapter, 'head.pt')) and \
        os.path.isfile(os.path.join(args.adapter, 'config.json')) and \
        any(name.startswith('model') and name.endswith('.safetensors')
            for name in os.listdir(args.adapter))
    if checkpoint is None and not kev_lora and not kev_full:
        sys.exit(f'{args.adapter}: neither an Open-Jev checkpoint nor a Kev LoRA or'
                 ' full-weight checkpoint')
    print(f'POST {args.server}/v1/systemone ...', flush=True)
    started = time.time()
    response = post_systemone(args.server, REQUEST, args.timeout)
    print(f'[server answered in {time.time() - started:.1f} s as model'
          f' "{response.get("model")}", usage {response.get("usage")}]', flush=True)
    served = answer_probabilities(REQUEST, response['answers'])

    started = time.time()
    if checkpoint is not None:
        print(f'jev on {args.base} + {checkpoint} ({args.dtype}, CPU, uncached forward) ...', flush=True)
        reference, temperature = openjev_reference(args.base, checkpoint, REQUEST, args.temperature,
                                                   args.dtype)
        name = 'jev'
    else:
        layout = 'LoRA merged' if kev_lora else 'full weights'
        print(f'kev on {args.base} + {args.adapter} ({layout}, {args.dtype}, CPU) ...', flush=True)
        reference, temperature = kev_reference(args.base, args.adapter, REQUEST, args.temperature,
                                               args.dtype)
        name = 'kev'
    print(f'[{name} answered in {time.time() - started:.1f} s at temperature {temperature}]')

    max_diff = 0.0
    for qid, spec in REQUEST['questions'].items():
        print(f'\n{qid} ({spec["type"]}):')
        print(f'  {"option":<30} {"ChatServer":>10} {name:>10} {"diff":>10}')
        for option, a, b in zip(option_labels(spec), served[qid], reference[qid]):
            max_diff = max(max_diff, abs(a - b))
            print(f'  {option[:30]:<30} {a:10.4f} {b:10.4f} {a - b:10.2e}')
    verdict = 'PASS' if max_diff <= args.threshold else 'FAIL'
    print(f'\nmax abs difference {max_diff:.2e} (threshold {args.threshold:.0e}): {verdict}')
    sys.exit(0 if verdict == 'PASS' else 1)


if __name__ == '__main__':
    main()
