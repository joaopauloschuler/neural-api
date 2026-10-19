# ChatServer: the same engine over HTTP

`ChatServer` (in this folder) is a minimal OpenAI-style HTTP server over the
same shared engine (`neural/neuralchatengine.pas`, `TChatEngine`), so
neural-api models can be called from any codebase that speaks the OpenAI
REST shape. It takes the same command line as ChatTerminal — the model directory and
the common flags in [README.md](README.md), minus the terminal-only
one-shot `-p` — plus the server flags below:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--host ADDR` | listen address | `127.0.0.1` (loopback only) |
| `--port N` | listen port | `8080` |
| `--decision-level l0\|raw` | how `POST /v1/systemone` reads the model: `l0` scores every rotation of the options and corrects each by the label prior; `raw` scores one prompt with the options in the order given | `l0` |
| `--decision-prior content_free\|none` | the label prior of `l0` (ignored by `raw`) | `content_free` |
| `--decision-max-rotations N` | cap on the cyclic shifts of a choice question under `l0` (fewer prompts, less position-bias correction) | `0` (one per option) |
| `--decision-adapter DIR` | answer `POST /v1/systemone` with the Kev or Open-Jev checkpoint in `DIR` instead of AnyJev L0; the server becomes decision-only and loads the model without its LM head (see *Trained decision checkpoints*; for Kev full weights `DIR` is the model directory too). Refused together with the three AnyJev flags above | none (AnyJev L0) |
| `--decision-temperature T` | with `--decision-adapter`: replace the checkpoint's calibration temperature (`1` = raw head scores; Kev's `KEV_TEMPERATURE`) | the checkpoint's |

```
lazbuild neural-api/examples/ChatTerminal/ChatServer.lpi
```

```
$ ChatServer /path/to/model --temperature 0.7 --top-p 0.9 --port 8080
...
Serving model on http://127.0.0.1:8080/v1 (SSE streaming with "stream":true; Ctrl+C stops)

$ curl http://127.0.0.1:8080/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"Hi!"}],"max_tokens":64}'

$ curl -N http://127.0.0.1:8080/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"Hi!"}],"stream":true}'
data: {"id":"chatcmpl-1","object":"chat.completion.chunk", ... "delta":{"role":"assistant","content":""} ...}
data: {"id":"chatcmpl-1","object":"chat.completion.chunk", ... "delta":{"content":"Hello"} ...}
...
data: [DONE]
```

## Endpoints

Endpoints: `POST /v1/chat/completions` (messages rendered through the
model's chat template), `POST /v1/completions` (plain completion, no
template - the `--format raw` path), `POST /v1/systemone` (typed
decisions, see *Typed decisions* below), `GET /v1/models`. A message
`content` is either a string or the OpenAI content-parts array
(`[{"type":"text","text":"..."}, ...]`) that current SDKs such as
openai-python and smolagents send by default; text parts are joined
with newlines and any non-text part (`image_url`, ...) is a 400, since
the server is text-only. With
`"stream": true` both POST endpoints stream the reply as OpenAI-style
Server-Sent Events - one `data:` chunk per decoded token, a
`finish_reason` chunk, then `data: [DONE]`.
`"stream_options": {"include_usage": true}` appends the usage chunk.
Only a literal JSON boolean is accepted for `stream` (a `"true"` string
is a 400, never a mis-parsed hang), and `"n"` other than 1 is rejected.
Response headers go out with the first token, so pre-generation failures
(bad template, context overflow) are still ordinary JSON 400s; if the
client disconnects mid-stream, generation aborts and the engine
invalidates its KV cache so the next request decodes cleanly.
Request fields `temperature`, `top_p`,
`top_k`, `min_p`, `repetition_penalty`, `frequency_penalty`,
`presence_penalty` and `max_tokens`/`max_completion_tokens` override the
launch defaults per request; absent fields fall back to them.

`POST /v1/completions` encodes the prompt without any chat template in
every mode, so it is the way to drive a base model. A server launched with
`--format raw` has no template at all: it answers `POST /v1/chat/completions`
with a 400 that points at `/v1/completions`. Without a template there is no
end-of-turn marker, generation stops on the tokenizer's EOS id or at
`max_tokens` only, and base models rarely emit EOS, so pass a small
`max_tokens`.

## Typed decisions: `POST /v1/systemone`

`POST /v1/systemone` takes a TypeSafe/Jev decision request: a state plus
typed questions in, option probabilities out, and no text generated. The
JSON field names follow TypeSafe's OpenAPI 0.2.0 (`api.typesafe.ai`).
`model` is the name `GET /v1/models` lists for the decision model (the
model directory's name, or the adapter directory's name under
`--decision-adapter`) or the alias `jev-latest`. `state`, `instructions` and
the option descriptions may each be a string, an object or an array.

```
$ curl http://127.0.0.1:8080/v1/systemone -d '{
    "model": "jev-latest",
    "state": {"ticket": "My card was declined twice and the app froze."},
    "questions": {
      "team":    {"type": "choice", "instructions": "Which team should take this ticket?",
                  "criteria": {"billing": "payments and refunds", "technical": null}},
      "urgency": {"type": "score", "instructions": "How urgent is it?",
                  "criteria": ["low", "medium", "high"]},
      "angry":   {"type": "noul", "instructions": "Is the customer angry?",
                  "criteria": {"true": "raised voice or threats", "false": "calm"}}}}'
{"model":"jev-latest","answers":{
  "team":{"type":"choice","choice":"billing","probabilities":{"billing":0.81,"technical":0.19},"confidence":0.62},
  "urgency":{"type":"score","score":1.3,"legend":{"0":"low","1":"medium","2":"high"},
             "probabilities":{"0":0.1,"1":0.5,"2":0.4},"confidence":0.25},
  "angry":{"type":"noul","noul":0.73}},
 "usage":{"input_tokens":2140,"output_tokens":0}}
```

The numbers above show the shape only (the server sends the reply on one
line). A `choice` answer carries the most likely option and the probability
of each option by name; a `score` answer carries the expected level
`score = sum i*p_i` over the levels `0..n-1`; a `noul` answer carries the
probability of `true`. The server rounds every number to 4 decimals.
`confidence` follows Jev's published formulas, computed from the
probabilities alone.

**Method.** `TAnyJevL0Scorer` (`neural/neuralanyjev.pas`) answers with
AnyJev's L0 readout on the loaded chat model, so the server loads nothing
extra. For each question the scorer renders one chat prompt per rotation of
the options (the cyclic shifts of a choice, both phrasings of a yes/no
question), reads the model's next-token probabilities of the answer labels
(`A`, `B`, ... / `1`, `2`, ... / `Yes`, `No`), divides each rotation by the
content-free prior (the same prompts with the states `N/A`, `""` and
`[MASK]`), and combines the rotations by their log-mean. The scorer prefills
the token prefix that all prompts of one state share once, so a request
costs about one prefix plus the prompt tails.

**Status codes.** 200 on success. 422 with the FastAPI detail list
(`{"detail":[{"loc":...,"msg":...,"type":...}]}`) on a malformed request,
including a `box` question or image `assets`. 422 with `{"detail": "..."}` on
a model name the server does not serve and on a valid request the scorer
cannot answer: a choice question with more than 26 options (the labels are
the letters `A`-`Z`), a prompt longer than the context (`--ctx`), or a model
whose tokenizer cannot spell an answer label (`Yes`, `No`, a letter) as one
token. 500 with `{"detail": "..."}` when scoring fails.

**Not supported.** `box` questions and image `assets`; more than 26 options
in a choice question; AnyJev's running-mean batch prior (the scorer takes
the prior from the content-free probes of each request); AnyJev's L1/L2
levels and adaptive shifts.

**Status.** AnyJev L0 is planned (coded): `TestNeuralAnyJev` matches the
real `anyjev` 0.3.0 package within 3.1e-6 on two pico Qwen fixtures, but
nobody has run it on a real checkpoint yet. The check below is how the user
runs it on Qwen2.5-0.5B-Instruct.

**Real-checkpoint check.** `tools/anyjev_real_check.py` sends one fixed
request (a choice, a score and a yes/no question) to a running ChatServer,
runs the real `anyjev` Decider (`HFBackend`, float32, CPU, L0, content-free
prior) on the same checkpoint, prints both probability vectors side by side,
and exits with status 1 when the largest difference exceeds 5e-3. Launch the
server with `--fp32`: the default int8 weights move the probabilities far
more than the FP32 rounding the check bounds. The script needs `anyjev`
0.3.0, `kev` (for Kev's value renderer), `torch` and `transformers`; it
retries for up to 5 minutes while the server is still loading.

```
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct q25
lazbuild -B neural-api/examples/ChatTerminal/ChatServer.lpi
neural-api/bin/x86_64-linux/bin/ChatServer q25/ --fp32 --ctx 4096 --port 8080 &
pip install anyjev==0.3.0
pip install --no-deps git+https://github.com/jaredpalmer/kev@1f04cfa28e6a708485b28a9585010b9e436b776e  # kev pins torch<2.9
python3 neural-api/tools/anyjev_real_check.py q25/ --server http://127.0.0.1:8080
kill %1
```

### Trained decision checkpoints: `--decision-adapter`

`--decision-adapter DIR` replaces AnyJev L0 with a trained decision
checkpoint. For a LoRA checkpoint the positional model directory stays the
BASE checkpoint the adapter was trained on; ChatServer merges the adapter's
LoRA into it while loading (`TChatOptions.LoRAAdapterDir`, FP32 merge in HF
tensor space), reads the head and scores with it. A Kev full-weight
checkpoint is its own model: pass its directory both as the model directory
and as `DIR` (`ChatServer kev-27b/ --decision-adapter kev-27b/`); nothing is
merged and the tokenizer is the checkpoint's copy. ChatServer detects the
layout of `DIR` (`DetectDecisionAdapterLayout` in
`neural/neuraldecisionheads.pas`):

| Layout | Files | Scorer |
| --- | --- | --- |
| Open-Jev (`ZefanCai/Open-Jev-2B`, ...) | `head.pt` and `temperature.json` in `DIR`, `DIR/checkpoint` or `DIR/package/checkpoint` (the HF repo root works), plus `model.json` and `adapter/` | `TOpenJevScorer`: one chat prompt per candidate, Linear(d, 1) head; the adapter is `<checkpoint>/adapter` when `model.json` has `lora_rank` > 0 |
| Kev LoRA (`jaredpalmer/kev-0.8b`, `-4b`, `-9b`) | `head.pt`, `adapter_config.json` and `adapter_model.safetensors` in `DIR` | `TKevScorer`: one row per question, pointer head; the adapter is `DIR` |
| Kev full weights (`jaredpalmer/kev-27b`) | `head.pt`, `config.json`, `model*.safetensors`, no `adapter_config.json` | `TKevScorer` as for Kev LoRA; the weights are `DIR` itself (a `Qwen3_5TextModel` export: flat `qwen3_5_text` config, keys without the `model.` prefix, no `lm_head`), so the model directory must be `DIR` too, else startup refuses. A model directory in this layout without `--decision-adapter` is refused too (it has no LM head to chat with) |

Any other directory is refused at startup with a message that lists the
three layouts. At startup ChatServer prints the base the checkpoint records
(Kev: `head.pt` `base` and `base_revision`; Open-Jev: `model.json`
`model_id` and `revision`). It cannot check that the model directory holds
that exact revision; it refuses only a plain mismatch: a head that reads
hidden states of another size than the base's `config.json` `hidden_size`.

**Decision-only.** With `--decision-adapter` the merged LoRA (or Kev's full
fine-tune) has changed the base weights for decision scoring (and Kev bases
are pretrained `-Base` models with no chat tuning), and the model is loaded
without its LM head (below), so `POST /v1/chat/completions` and
`POST /v1/completions` answer 501 with that reason, and `GET /v1/models`
lists only the decision model. Its name is the adapter directory's name
(`kev-0.8b`, `Open-Jev-2B`; for `.../package/checkpoint` the repo root's
name); `jev-latest` is accepted too. `--decision-level`, `--decision-prior`
and `--decision-max-rotations` tune AnyJev only and are refused with
`--decision-adapter`.

**Memory.** Kev and Open-Jev read the final hidden states and never
compute the LM head, so with `--decision-adapter` ChatServer sets
`TChatOptions.OmitLMHead`: `BuildFromPretrained(..., pOmitLMHead)` ends the
net at the final RMSNorm, and the `--prefill-window` twins are built the
same way. The vocab x hidden LM-head matrix is never allocated: 248320 x 1024
= 254M weights for Kev-0.8B (1.0 GB in FP32, about 0.25 GB in int8; the
base ties the head to the embedding, but the build holds a separate copy of
those rows, and the embedding still loads) and 248320 x 5120 = 1.27G for
Kev-27B (5.1 GB FP32, about 1.3 GB int8). The probabilities do not change: `TestNeuralDecisionHeads`
measures them bit-identical with and without the LM head on both fixtures.

**Status.** Planned (coded), not supported: `TestNeuralDecisionHeads` matches
kev's and jev's own code on pico fixtures (Kev LoRA within 4.1e-6, Kev full
weights within 2.2e-6, Open-Jev within 6.4e-7), and
`tools/decision_real_check.py` passed against all three fixtures through a
running server (max difference 4.6e-5, the server's 4-decimal rounding), but
nobody has run a real Kev or Open-Jev checkpoint yet.

**Real-checkpoint checks.** `tools/decision_real_check.py BASE ADAPTER`
sends one fixed request (a choice, a score and a yes/no question) to the
running server, scores it in Python with the checkpoint's own code on the
same local base (Kev: `kev.checkpoint.Checkpoint(ADAPTER).load` in FP32 with
`head.pt`'s base replaced by `BASE`, then `admit` + `DecisionModel.probs`;
Open-Jev: jev's `compile_request` + `DecisionModel.forward` (the uncached
path) + `Predictor.predict`, FP32 backbone), prints both side by side and
exits with status 1 when the largest difference exceeds 5e-3. Launch the
server with `--fp32`: int8 weights move the probabilities far more than the
FP32 rounding the check bounds. With `--decision-temperature T`, pass
`--temperature T` to the script too. kev and jev pin older torch and
transformers versions, so install them `--no-deps` next to torch,
transformers, peft and safetensors.

Kev-0.8B (base `Qwen/Qwen3.5-0.8B-Base` at the revision `head.pt` records):

```
git clone https://huggingface.co/Qwen/Qwen3.5-0.8B-Base q08b
git -C q08b checkout dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
git clone https://huggingface.co/jaredpalmer/kev-0.8b
lazbuild -B neural-api/examples/ChatTerminal/ChatServer.lpi
neural-api/bin/x86_64-linux/bin/ChatServer q08b/ --decision-adapter kev-0.8b/ --fp32 --ctx 4096 --port 8080 &
pip install --no-deps git+https://github.com/jaredpalmer/kev@1f04cfa28e6a708485b28a9585010b9e436b776e
python3 neural-api/tools/decision_real_check.py q08b/ kev-0.8b/ --server http://127.0.0.1:8080
kill %1
```

Open-Jev-2B (base `Qwen/Qwen3.5-2B` at the revision `model.json` records):

```
git clone https://huggingface.co/Qwen/Qwen3.5-2B q2b
git -C q2b checkout 15852e8c16360a2fea060d615a32b45270f8a8fc
git clone https://huggingface.co/ZefanCai/Open-Jev-2B
neural-api/bin/x86_64-linux/bin/ChatServer q2b/ --decision-adapter Open-Jev-2B/ --fp32 --ctx 4096 --port 8080 &
pip install --no-deps git+https://github.com/Zefan-Cai/Open-Jev@bd4118882f733574a3250a4b65fe4d884130c08b
python3 neural-api/tools/decision_real_check.py q2b/ Open-Jev-2B/ --server http://127.0.0.1:8080
kill %1
```

Kev-27B (full weights, fine-tuned from `Qwen/Qwen3.8-27B`; the same
directory is the model directory, the `--decision-adapter` directory and both
script arguments). 27B in FP32 needs about 110 GB on each side, so this check
runs the server on the GPU with its default int8 weights and the Python
reference in bfloat16 (about 55 GB of host RAM); both move the probabilities
by more than FP32 rounding, so the threshold is loosened to 5e-2 and the
printed table is the result to read (the same argmax per question). On a
GPU that cannot hold the int8 weights, add `--int4`.

```
git clone https://huggingface.co/jaredpalmer/kev-27b
neural-api/bin/x86_64-linux/bin/ChatServer kev-27b/ --decision-adapter kev-27b/ --gpu --ctx 4096 --port 8080 &
python3 neural-api/tools/decision_real_check.py kev-27b/ kev-27b/ --dtype bfloat16 --threshold 5e-2 --server http://127.0.0.1:8080
kill %1
```

### Calling it: `DecisionClient` and curl

Start a server with one of the three scorers (the checkpoints are the ones
cloned above):

```
# AnyJev L0 on a plain chat model
neural-api/bin/x86_64-linux/bin/ChatServer q25/ --fp32 --ctx 4096 --port 8080 &
# Kev (LoRA adapter on its base)
neural-api/bin/x86_64-linux/bin/ChatServer q08b/ --decision-adapter kev-0.8b/ --fp32 --ctx 4096 --port 8080 &
# Open-Jev (LoRA adapter on its base)
neural-api/bin/x86_64-linux/bin/ChatServer q2b/ --decision-adapter Open-Jev-2B/ --fp32 --ctx 4096 --port 8080 &
```

`DecisionClient` (`DecisionClient.lpr` in this folder) sends one request and
prints each answer: the choice or the expected level, the probabilities in
the server's order (`*` marks the chosen option, a score line shows its
legend), the confidence and the token usage. It uses only FCL units
(`fphttpclient`, `fpjson`), none of the neural-api units, so it builds in
about a second and shows that any HTTP client can call the endpoint. Without
`--request` it sends a built-in support-ticket request (an object state; a
choice of team with one `null` description, a 3-level frustration score and
an urgent yes/no question). The folder `decision_examples/` holds three
request files: ticket routing, review sentiment and a nested-object state.

```
lazbuild -B neural-api/examples/ChatTerminal/DecisionClient.lpi
cd neural-api/examples/ChatTerminal
../../bin/x86_64-linux/bin/DecisionClient --retries 300
../../bin/x86_64-linux/bin/DecisionClient --request decision_examples/ticket_routing.json
../../bin/x86_64-linux/bin/DecisionClient --request decision_examples/review_sentiment.json --raw
curl -s -d @decision_examples/ticket_routing.json http://127.0.0.1:8080/v1/systemone
curl -s -d @decision_examples/nested_state.json http://127.0.0.1:8080/v1/systemone
```

```
model: jev-latest
team (choice): billing   confidence 0.2794
    * billing    0.4596
      technical  0.1395
      shipping   0.2395
      other      0.1614
priority (score): 1.3656 on levels 0..3   confidence 0.2216
      0  0.2064  low
      1  0.3515  normal
      2  0.3121  high
      3  0.1299  critical
needs_refund (noul): P(true) = 0.3064
usage: 5234 input tokens, 0 output tokens
```

(The numbers above come from the random pico fixture
`tests/fixtures/anyjev_qwen25` and mean nothing.)

| Option | Meaning | Default |
| --- | --- | --- |
| `--server URL` | server root (a trailing `/v1` is accepted) | `http://127.0.0.1:8080` |
| `--model NAME` | the request's `model`; a request file keeps its own model unless `--model` is given | `jev-latest` |
| `--request FILE` | send the JSON request in `FILE` | the built-in demo |
| `--api-key KEY` | send `Authorization: Bearer KEY` | none |
| `--retries N` | retry a refused connection `N` times, 1 s apart (ChatServer listens only after the model has loaded) | `0` |
| `--raw` | print the raw JSON response body | off |
| `--selftest` | run the offline checks and exit | |

On a status other than 200, `DecisionClient` prints the status and the
`detail` (a string, or one `loc: msg (type)` line per entry of a 422 list)
and exits with 1; it exits with 2 when it cannot connect (with a hint to
start ChatServer) and with 3 on a bad command line or request file.

**Authentication.** ChatServer has none: it ignores any `Authorization`
header and listens on `127.0.0.1` unless `--host` says otherwise.
`--api-key` exists for other servers that speak the same wire format.

TypeSafe's Python SDK (`typesafe-sdk` 0.7.5, checked by reading its source,
not run against this server) takes the server root through its `base_url`
constructor argument or the `TYPESAFE_BASE_URL` environment variable
(`http://127.0.0.1:8080`, without `/v1`) and sends its default model
`jev-latest`, which this server accepts; it refuses to start without an API
key (`api_key` or `TYPESAFE_API_KEY`; any value works, since this server
ignores it), and it zstd-compresses request bodies of 1024 bytes or more
when zstd is installed, which this server does not decode, so set its
`compression` argument to false.

## Concurrency and cache reuse

Requests are handled strictly one at a time: one model, one live KV cache,
and the non-threaded accept loop serializes them. A `POST /v1/systemone`
request runs on the same loop and the same engine, so it never interleaves
with a chat request. Before it scores, `TChatEngine.PrefillDecisionPrefix`
saves the live conversation to a conversation slot (when the slots are on,
as they are by default here), so the next chat request of that conversation
resumes from the slot instead of prefilling it again. The KV-cache prefix reuse
applies across requests: a growing conversation re-sent in full each turn
only prefills the new tail, so time-to-first-token stays roughly flat.

The live KV cache holds one conversation. When two clients take turns, each
request leaves the other client's conversation, so it shares only the ids up
to where the two differ, often just the system prompt. Two mechanisms keep
the other conversation (see the ChatTerminal [README](README.md), *cache
checkpoints* and *conversation slots*):

- **Cache checkpoints** (`--cache-checkpoints N`, on by default for
  hybrid/recurrent nets) hold the recurrent state at message boundaries of
  every recent conversation. On a net with attention layers a checkpoint is
  resumable only within the ids the prompt shares with the live KV cache, so
  across conversations it helps only up to the shared system prompt.
- **Conversation slots** (`--kv-slots N`, **4 by default in ChatServer**;
  `--kv-slots 0` turns them off) save the conversation a request leaves
  (its K/V rows and recurrent state) and resume it when its client comes
  back. Each slot costs one KV cache at its conversation's length (host
  RAM); on hybrid/recurrent nets also one recurrent state (host RAM) and one
  recurrent state at the last user message (OpenCL memory under `--gpu`),
  plus one more such copy held by the engine (see *Memory* under
  *conversation slots* in the README).

A server for a few alternating clients (4 conversation slots by default),
and the same server without slots:

```
$ ChatServer /path/to/model --gpu --stats --port 8080
$ ChatServer /path/to/model --gpu --stats --port 8080 --kv-slots 0
```

`--stats` prints, per request, where the prompt resumed from (`live cache`,
`checkpoint` or `conversation slot`) and the memory the slots hold. The
slots have not been timed on a real model yet.

## Testing

`ChatServer --selftest` runs the offline request-parsing and
parameter-overlay checks (the `messages` array, per-request overrides of
the launch defaults, `max_completion_tokens` over `max_tokens`, the ignored
`stop` field, the `stream` flag, the `--kv-slots` server default, the
`--decision-*` flags and their refusals, the `--decision-adapter` layout
detection on placeholder files, and the `/v1/systemone` status codes on a
fixed test scorer, including the decision-only 501s) without needing any
model files. `DecisionClient --selftest` checks the client offline: the
built-in request, the request-file handling, and the printing and exit
status of canned 200, 422 and 501 replies.
