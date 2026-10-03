# ChatServer: the same engine over HTTP

`ChatServer` (in this folder) is a minimal OpenAI-style HTTP server over the
same shared engine (`neural/neuralchatengine.pas`, `TChatEngine`), so
neural-api models can be called from any codebase that speaks the OpenAI
REST shape. It takes the same command line as ChatTerminal — the model directory and
the common flags in [README.md](README.md), minus the terminal-only
one-shot `-p` — plus two server flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--host ADDR` | listen address | `127.0.0.1` (loopback only) |
| `--port N` | listen port | `8080` |

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
template - the `--format raw` path), `GET /v1/models`. A message
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

## Concurrency and cache reuse

Requests are handled strictly one at a time: one model, one live KV cache,
and the non-threaded accept loop serializes them. The KV-cache prefix reuse
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
- **Conversation slots** (`--kv-slots N`, **off by default**) save the
  conversation a request leaves (its K/V rows and recurrent state) and
  resume it when its client comes back. Each slot costs one KV cache at its
  conversation's length (host RAM); on hybrid/recurrent nets also one
  recurrent state (host RAM) and one recurrent state at the last user
  message (OpenCL memory under `--gpu`), plus one more such copy held by the
  engine (see *Memory* under *conversation slots* in the README).

A server for a few alternating clients:

```
$ ChatServer /path/to/model --gpu --kv-slots 4 --stats --port 8080
```

`--stats` prints, per request, where the prompt resumed from (`live cache`,
`checkpoint` or `conversation slot`) and the memory the slots hold. The
slots have not been timed on a real model yet.

## Testing

`ChatServer --selftest` runs the offline request-parsing and
parameter-overlay checks (the `messages` array, per-request overrides of
the launch defaults, `max_completion_tokens` over `max_tokens`, the ignored
`stop` field, the `stream` flag) without needing any model files.
