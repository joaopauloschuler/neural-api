#!/usr/bin/env python3
"""Real-checkpoint parity check: ChatServer's POST /v1/systemone against the
real `anyjev` 0.3.0 Decider on the same checkpoint.

The script sends the fixed Jev request of tools/anyjev_tiny_fixture.py (one
choice, one score and one noul question; model "jev-latest") to a running
ChatServer, then runs `Decider(HFBackend(<model dir>, device='cpu',
dtype='float32'))` at level L0 with the content-free prior,
adaptive_shifts=False and canonical_order=False on the same request, and
prints both probability vectors per question in AnyJev's option order
(choice and score as given, noul (Yes, No)).

Launch ChatServer with --fp32: its default int8 weights (and the int8 KV
cache that follows them) move the probabilities by far more than the FP32
summation-order noise this check is meant to bound. The default threshold
5e-3 is the max absolute probability difference allowed. FP32 against FP32
on Qwen2.5-0.5B-Instruct is expected near 1e-4..1e-3 (different summation
order through 24 layers); the server also rounds every probability to 4
decimals (up to 5e-5). A real mismatch (a wrong prompt token, label id,
template or prior) moves probabilities by 1e-2 or more.

Needs anyjev 0.3.0 and kev (Kev's api.render, used to render the request
the way neuraldecision does), torch and transformers. HTTP uses only the
standard library.

Usage (on the box that holds the checkpoint):
  git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct q25
  bin/x86_64-linux/bin/ChatServer q25/ --fp32 --ctx 4096 --port 8080 &
  python3 tools/anyjev_real_check.py q25/ --server http://127.0.0.1:8080

Coded by Claude (AI).
"""
import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anyjev_tiny_fixture import REQUEST, anyjev_questions, rewrite_special_token_strings  # noqa: E402

DEFAULT_THRESHOLD = 5e-3
CONNECT_ATTEMPTS = 30      # the server may still be loading the checkpoint
CONNECT_RETRY_SECONDS = 10


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


def server_probabilities(request, answers):
    """Per question id, the server's probabilities in AnyJev's option order."""
    out = {}
    for qid, spec in request['questions'].items():
        answer = answers[qid]
        if spec['type'] == 'choice':
            out[qid] = [answer['probabilities'][name] for name in spec['criteria']]
        elif spec['type'] == 'score':
            out[qid] = [answer['probabilities'][str(level)]
                        for level in range(len(spec['criteria']))]
        else:
            out[qid] = [answer['noul'], 1.0 - answer['noul']]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('model_dir', help='local HF snapshot, e.g. Qwen2.5-0.5B-Instruct')
    parser.add_argument('--server', default='http://127.0.0.1:8080')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--timeout', type=float, default=1800,
                        help='seconds one /v1/systemone request may take')
    args = parser.parse_args()

    request = copy.deepcopy(REQUEST)
    request['model'] = 'jev-latest'
    print(f'POST {args.server}/v1/systemone ...', flush=True)
    started = time.time()
    response = post_systemone(args.server, request, args.timeout)
    print(f'[server answered in {time.time() - started:.1f} s as model'
          f' "{response.get("model")}", usage {response.get("usage")}]', flush=True)
    served = server_probabilities(request, response['answers'])

    from anyjev import Decider
    from anyjev.backends.hf import HFBackend
    from kev.api import render

    print(f'anyjev Decider on {args.model_dir} (float32, CPU) ...', flush=True)
    started = time.time()
    backend = HFBackend(args.model_dir, device='cpu', dtype='float32')
    decider = Decider(backend, level='L0', prior='content_free', adaptive_shifts=False,
                      canonical_order=False)
    state_text = rewrite_special_token_strings(render(request['state']))
    questions = anyjev_questions(request)
    decisions = decider.decide(state_text, questions)
    print(f'[anyjev answered in {time.time() - started:.1f} s; stats {decider.stats}]')

    max_diff = 0.0
    for q, d in zip(questions, decisions):
        reference = [float(p) for p in d.probs]
        mine = served[q.id]
        print(f'\n{q.id} ({q.kind}):')
        print(f'  {"option":<40} {"ChatServer":>10} {"anyjev":>10} {"diff":>10}')
        for option, a, b in zip(q.options, mine, reference):
            max_diff = max(max_diff, abs(a - b))
            print(f'  {option[:40]:<40} {a:10.4f} {b:10.4f} {a - b:10.2e}')
    verdict = 'PASS' if max_diff <= args.threshold else 'FAIL'
    print(f'\nmax abs difference {max_diff:.2e} (threshold {args.threshold:.0e}): {verdict}')
    sys.exit(0 if verdict == 'PASS' else 1)


if __name__ == '__main__':
    main()
