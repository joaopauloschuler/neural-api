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
`model` is the name `GET /v1/models` lists for the chat model (the model
directory's name) or the alias `jev-latest`. `state`, `instructions` and
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
levels and adaptive shifts; trained decision heads (Kev, Open-Jev).

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
`--decision-*` flags, and the `/v1/systemone` status codes on a fixed test
scorer) without needing any model files.
