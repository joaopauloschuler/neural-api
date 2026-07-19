program ChatServer;
(*
ChatServer: a minimal OpenAI-style HTTP inference server over any checkpoint
the generic importer dispatch understands (neuralpretrained.pas
BuildFromPretrained: gpt2, llama, mistral, qwen2/3, gemma/2/3, phi/phi3,
gpt_neo(x), gptj, rwkv, mamba, bloom, deepseek_v2, ...), so neural-api
models can be called from any codebase that speaks the OpenAI REST shape.

It is the HTTP sibling of the ChatTerminal REPL: both are thin frontends
over the shared engine unit neural/neuralchatengine.pas (TChatEngine), and
both take the SAME command line (model directory + sampling/memory/GPU
flags). Server-specific flags: --host (default 127.0.0.1, loopback only)
and --port (default 8080).

  ChatServer /path/to/model --temperature 0.8 --top-p 0.9 --port 8080

Endpoints (all responses application/json):

  POST /v1/chat/completions   {"messages":[{"role":...,"content":...}], ...}
       The messages array is rendered through the model's chat template
       (autodetected exactly as in ChatTerminal; --format overrides) and
       one assistant reply is generated. "stream":true streams the reply
       as OpenAI-style SSE (see STREAMING below); any non-boolean stream
       spelling is rejected with a clear error (fail loudly, never hang
       an SSE client). "n" other than 1 is rejected. Unavailable in
       --format raw.
  POST /v1/completions        {"prompt":"...", ...}
       Plain text completion: the prompt is encoded WITHOUT any chat
       template and the model continues it (the --format raw path, but
       available in every mode). Also honors "stream":true.
  GET  /v1/models             lists the single loaded model.

STREAMING: with "stream":true the reply is sent as Server-Sent Events -
one "data: {chunk JSON}" frame per decoded token (chat.completion.chunk
objects with a delta, or text_completion chunks with a text delta), a
final frame carrying finish_reason, then "data: [DONE]". This is the wire
shape the OpenAI client libraries expect from stream=True. The connection
carries "Connection: close" (fcl-web 3.2.2 serves one request per
connection anyway) and the response has no Content-Length - the stream
ends when the connection closes after [DONE].
"stream_options":{"include_usage":true} appends the usage chunk before
[DONE], as api.openai.com does. Headers are only sent when the first
token is ready, so pre-generation failures (bad template, context full)
still produce ordinary JSON 400s. If generation dies MID-stream the
server emits a best-effort "data: {"error":...}" frame and closes; the
engine invalidates its KV cache on that path, so the next request is
clean. A client that disconnects mid-stream simply aborts generation via
the failed socket write (same cache-invalidation path).

PER-REQUEST SAMPLING: temperature, top_p, top_k, min_p, repetition_penalty,
frequency_penalty, presence_penalty and max_tokens (or the newer
max_completion_tokens) may be set per request; each ABSENT field falls back
to the value resolved at server launch (explicit CLI flag >
generation_config.json > built-in fallback - the ChatTerminal precedence).
Absence is tracked at the JSON level, so 0.0 in a request is an explicit
value, not "unset". A request "stop" field is IGNORED (the engine already
stops on EOS + the chat format's end-of-turn marker); ignoring is noted on
the console once per request.

CONCURRENCY: the engine is a single model + a single KV-cache session, so
the server runs the fcl-web HTTP server in its default non-threaded mode -
requests are handled strictly one at a time on the accept loop, which IS
the serialization. The KV-cache prefix reuse still pays off: consecutive
requests of a growing conversation share their prompt prefix, so only the
new tail is prefilled (TTFT stays roughly flat as the conversation grows).

--selftest runs the offline request-parsing/overlay unit checks (no model
needed) and exits.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  // cmem is skipped in the Debug build mode: it enables Valgrind (-gv), and
  // FPC then pulls in cmem itself, so naming it here is a duplicate.
  {$IFDEF UNIX}cthreads, {$IFNDEF Debug}cmem,{$ENDIF}{$ENDIF}
  Classes, SysUtils, DateUtils, fpjson, jsonparser, fphttpserver,
  neuralvolume, neuralchat, neuralchatengine;

type
  // TFPHTTPServer keeps the listen Address protected; republish it so the
  // server can honor --host.
  TChatHTTPServer = class(TFPHTTPServer)
  public
    property Address;
  end;

  // TFPHTTPConnectionResponse keeps Connection protected. Streaming needs the
  // raw socket (one write per token, after SendHeaders), so the same
  // republishing trick as Address: a cast to this same-unit subclass makes
  // Connection.Socket reachable. The socket writes unbuffered and fcl-web
  // already sets MSG_NOSIGNAL, so a vanished client raises a stream error
  // instead of killing the process with SIGPIPE.
  TStreamableResponse = class(TFPHTTPConnectionResponse)
  public
    property Connection;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: ChatServer <model-dir> [options]');
  WriteLn;
  WriteLn('<model-dir> holds config.json, model.safetensors (or a sharded');
  WriteLn('index / pytorch_model.bin), tokenizer.json and (for chat-format');
  WriteLn('autodetection) tokenizer_config.json.');
  WriteLn;
  WriteLn('An OpenAI-style HTTP server: POST /v1/chat/completions,');
  WriteLn('POST /v1/completions, GET /v1/models. Request fields temperature, top_p,');
  WriteLn('top_k, min_p, repetition_penalty, frequency_penalty, presence_penalty and');
  WriteLn('max_tokens override the launch defaults per request; absent fields fall');
  WriteLn('back to them. "stream":true streams SSE chunks (one per token) the way');
  WriteLn('the OpenAI client libraries expect; "n">1 is rejected.');
  WriteLn;
  WriteLn('Server options:');
  WriteLn('  --host ADDR           listen address (default 127.0.0.1, loopback only)');
  WriteLn('  --port N              listen port (default 8080)');
  WriteLn;
  PrintChatOptionsHelp();
end;

// ---------------------------------------------------------------------------
// Request parsing (pure: JSON in, records out - exercised by --selftest).
// ---------------------------------------------------------------------------

// Reads the "messages" array of a chat completions request into TChatMessages.
// False + ErrMsg when the field is missing/malformed (role and content must
// both be strings; content may be empty, role may not).
function ParseRequestMessages(Req: TJSONObject; out Msgs: TChatMessages;
  out ErrMsg: string): boolean;
var
  Node: TJSONData;
  Arr: TJSONArray;
  Item: TJSONObject;
  RoleNode, ContentNode: TJSONData;
  Cnt: integer;
begin
  Result := false;
  ErrMsg := '';
  SetLength(Msgs, 0);
  Node := Req.Find('messages');
  if not (Assigned(Node) and (Node.JSONType = jtArray)) then
  begin
    ErrMsg := '"messages" must be an array';
    exit;
  end;
  Arr := TJSONArray(Node);
  if Arr.Count = 0 then
  begin
    ErrMsg := '"messages" is empty';
    exit;
  end;
  SetLength(Msgs, Arr.Count);
  for Cnt := 0 to Arr.Count - 1 do
  begin
    if Arr.Items[Cnt].JSONType <> jtObject then
    begin
      ErrMsg := Format('messages[%d] is not an object', [Cnt]);
      exit;
    end;
    Item := TJSONObject(Arr.Items[Cnt]);
    RoleNode := Item.Find('role');
    ContentNode := Item.Find('content');
    if not (Assigned(RoleNode) and (RoleNode.JSONType = jtString) and
      (RoleNode.AsString <> '')) then
    begin
      ErrMsg := Format('messages[%d].role must be a non-empty string', [Cnt]);
      exit;
    end;
    if not (Assigned(ContentNode) and (ContentNode.JSONType = jtString)) then
    begin
      ErrMsg := Format('messages[%d].content must be a string', [Cnt]);
      exit;
    end;
    Msgs[Cnt] := ChatMessage(RoleNode.AsString, ContentNode.AsString);
  end;
  Result := true;
end;

// Overlays the request's sampling fields on the launch defaults: each field
// PRESENT in the request replaces the launch value; absent fields keep it
// (absence is JSON-level, so an explicit 0.0 counts as set). Stream reports
// a literal "stream":true (any non-boolean spelling is rejected) and
// IncludeUsage a "stream_options":{"include_usage":true}. Rejects "n" <> 1.
// IgnoredStop reports a present "stop" field so the caller can log that it
// is ignored.
function OverlayRequestOptions(const Base: TChatOptions; Req: TJSONObject;
  out GenOpt: TChatOptions; out ErrMsg: string; out IgnoredStop: boolean;
  out Stream: boolean; out IncludeUsage: boolean): boolean;
var
  Node, SubNode: TJSONData;
begin
  Result := false;
  ErrMsg := '';
  IgnoredStop := false;
  Stream := false;
  IncludeUsage := false;
  GenOpt := Base;
  // Only a literal JSON boolean is accepted (not a string "true", a 1, a
  // null...): a client that asked for SSE in a spelling we would silently
  // read as false must fail loudly, never sit waiting on a JSON body it
  // will mis-parse as an SSE stream (or the other way around).
  Node := Req.Find('stream');
  if Assigned(Node) then
  begin
    if Node.JSONType <> jtBoolean then
    begin
      ErrMsg := '"stream" must be a JSON boolean';
      exit;
    end;
    Stream := Node.AsBoolean;
  end;
  // stream_options is only meaningful on a streaming request (the OpenAI
  // API rejects it otherwise too); include_usage asks for the usage chunk
  // before [DONE].
  Node := Req.Find('stream_options');
  if Assigned(Node) then
  begin
    if Node.JSONType <> jtObject then
    begin
      ErrMsg := '"stream_options" must be an object';
      exit;
    end;
    if not Stream then
    begin
      ErrMsg := '"stream_options" requires "stream": true';
      exit;
    end;
    SubNode := TJSONObject(Node).Find('include_usage');
    if Assigned(SubNode) then
    begin
      if SubNode.JSONType <> jtBoolean then
      begin
        ErrMsg := '"stream_options.include_usage" must be a JSON boolean';
        exit;
      end;
      IncludeUsage := SubNode.AsBoolean;
    end;
  end;
  // AsFloat, not AsInteger: 1.0 must pass, 1.5 must not round to "close
  // enough", and a huge value must not overflow Round into a 500.
  Node := Req.Find('n');
  if Assigned(Node) and
    not ((Node.JSONType = jtNumber) and (Node.AsFloat = 1.0)) then
  begin
    ErrMsg := 'only "n": 1 is supported';
    exit;
  end;
  Node := Req.Find('temperature');
  if Assigned(Node) and (Node.JSONType = jtNumber) then
    GenOpt.Temperature := Node.AsFloat;
  Node := Req.Find('top_p');
  if Assigned(Node) and (Node.JSONType = jtNumber) then
    GenOpt.TopP := Node.AsFloat;
  // A request top_k maps to the HF-style WEIGHTED top-k (this library's plain
  // top-k draws uniformly - the same mapping generation_config top_k gets).
  // Range-checked through AsFloat first so an absurd value 400s instead of
  // overflowing Round into a 500.
  Node := Req.Find('top_k');
  if Assigned(Node) and (Node.JSONType = jtNumber) then
  begin
    if (Node.AsFloat < 0) or (Node.AsFloat > 1e9) then
    begin
      ErrMsg := 'top_k out of range';
      exit;
    end;
    GenOpt.TopK := Node.AsInteger;
    GenOpt.WeightedTopK := true;
  end;
  Node := Req.Find('min_p');
  if Assigned(Node) and (Node.JSONType = jtNumber) then
    GenOpt.MinP := Node.AsFloat;
  Node := Req.Find('repetition_penalty');
  if Assigned(Node) and (Node.JSONType = jtNumber) then
    GenOpt.RepetitionPenalty := Node.AsFloat;
  Node := Req.Find('frequency_penalty');
  if Assigned(Node) and (Node.JSONType = jtNumber) then
    GenOpt.FrequencyPenalty := Node.AsFloat;
  Node := Req.Find('presence_penalty');
  if Assigned(Node) and (Node.JSONType = jtNumber) then
    GenOpt.PresencePenalty := Node.AsFloat;
  // max_tokens is the classic name, max_completion_tokens the newer one;
  // when both are present the newer wins. Same range stance as top_k.
  Node := Req.Find('max_completion_tokens');
  if not (Assigned(Node) and (Node.JSONType = jtNumber)) then
    Node := Req.Find('max_tokens');
  if Assigned(Node) and (Node.JSONType = jtNumber) then
  begin
    if (Node.AsFloat < 1) or (Node.AsFloat > 1e9) then
    begin
      ErrMsg := 'max_tokens must be between 1 and 1e9';
      exit;
    end;
    GenOpt.MaxNewTokens := Node.AsInteger;
  end;
  IgnoredStop := Assigned(Req.Find('stop'));
  Result := true;
end;

// ---------------------------------------------------------------------------
// The server: one engine, one accept loop (non-threaded = serialized).
// ---------------------------------------------------------------------------
type
  TServerApp = class(TObject)
  public
    Engine: TChatEngine;
    ModelName: string;          // reported in responses: the model dir's name
    RequestCount: integer;      // completion ids + console log lines
    procedure NoticeOut(const S: string);
    procedure HandleRequest(Sender: TObject;
      var ARequest: TFPHTTPConnectionRequest;
      var AResponse: TFPHTTPConnectionResponse);
  private
    // Live only while a streaming request is being served (the server is
    // non-threaded, so one at a time by construction).
    SSEResp: TFPHTTPConnectionResponse;
    SSEHeadersSent: boolean;
    SSEChat: boolean;           // chat.completion.chunk vs text_completion
    SSEIncludeUsage: boolean;
    SSEId: string;
    SSECreated: int64;
    procedure SendJSON(var AResponse: TFPHTTPConnectionResponse;
      Code: integer; Obj: TJSONObject);
    procedure SendError(var AResponse: TFPHTTPConnectionResponse;
      Code: integer; const Msg: string);
    procedure HandleChatCompletions(Req: TJSONObject;
      var AResponse: TFPHTTPConnectionResponse);
    procedure HandleCompletions(Req: TJSONObject;
      var AResponse: TFPHTTPConnectionResponse);
    procedure HandleModels(var AResponse: TFPHTTPConnectionResponse);
    function UsageObject(): TJSONObject;
    function StreamChunkFrame(const ContentDelta: string; WithRole: boolean;
      const FinishReason: string): string;
    procedure StreamWrite(const S: string);
    procedure StreamEnsureHeaders();
    procedure StreamTokenOut(const S: string);
    procedure StreamFinish();
    procedure RunStreaming(ChatMode: boolean; const Msgs: TChatMessages;
      const PromptIds: TNeuralIntegerArray; const GenOpt: TChatOptions;
      IncludeUsage: boolean; var AResponse: TFPHTTPConnectionResponse);
  end;

procedure TServerApp.NoticeOut(const S: string);
begin
  WriteLn(S);
end;

// Serializes Obj into the response and frees it. fpjson's AsJSON does the
// string escaping, so reply text with quotes/newlines/UTF-8 is safe.
procedure TServerApp.SendJSON(var AResponse: TFPHTTPConnectionResponse;
  Code: integer; Obj: TJSONObject);
begin
  try
    AResponse.Code := Code;
    AResponse.ContentType := 'application/json';
    AResponse.Content := Obj.AsJSON;
  finally
    Obj.Free;
  end;
end;

// The OpenAI error envelope: {"error":{"message":...,"type":...}}.
procedure TServerApp.SendError(var AResponse: TFPHTTPConnectionResponse;
  Code: integer; const Msg: string);
var
  Root, Err: TJSONObject;
  Kind: string;
begin
  if Code >= 500 then Kind := 'server_error'
  else Kind := 'invalid_request_error';
  Err := TJSONObject.Create(['message', Msg, 'type', Kind]);
  Root := TJSONObject.Create(['error', Err]);
  SendJSON(AResponse, Code, Root);
  WriteLn('[error ', Code, '] ', Msg);
end;

function TServerApp.UsageObject(): TJSONObject;
begin
  Result := TJSONObject.Create([
    'prompt_tokens', Engine.LastPromptTokens,
    'completion_tokens', Engine.LastCompletionTokens,
    'total_tokens', Engine.LastPromptTokens + Engine.LastCompletionTokens]);
end;

// ---------------------------------------------------------------------------
// Streaming (SSE): one "data: {chunk}" frame per decoded token, then a
// finish_reason frame, an optional usage frame, and "data: [DONE]".
// ---------------------------------------------------------------------------

// One SSE frame carrying an OpenAI streaming chunk. Chat mode emits
// chat.completion.chunk objects with a delta ({"role":"assistant",
// "content":""} on the first frame, {"content": token} afterwards, {} with
// the finish_reason on the last); completions mode emits text_completion
// chunks whose delta is the "text" field itself.
function TServerApp.StreamChunkFrame(const ContentDelta: string;
  WithRole: boolean; const FinishReason: string): string;
var
  Root, Choice, Delta: TJSONObject;
  ObjKind: string;
begin
  if SSEChat then
  begin
    Delta := TJSONObject.Create();
    if WithRole then
    begin
      Delta.Add('role', 'assistant');
      Delta.Add('content', '');
    end
    else if ContentDelta <> '' then
      Delta.Add('content', ContentDelta);
    Choice := TJSONObject.Create(['index', 0, 'delta', Delta]);
    ObjKind := 'chat.completion.chunk';
  end
  else
  begin
    Choice := TJSONObject.Create(['index', 0, 'text', ContentDelta]);
    ObjKind := 'text_completion';
  end;
  if FinishReason = '' then
    Choice.Add('finish_reason', TJSONNull.Create())
  else
    Choice.Add('finish_reason', FinishReason);
  Root := TJSONObject.Create([
    'id', SSEId,
    'object', ObjKind,
    'created', SSECreated,
    'model', ModelName,
    'choices', TJSONArray.Create([Choice])]);
  try
    Result := 'data: ' + Root.AsJSON + #10#10;
  finally
    Root.Free;
  end;
end;

// Raw socket write: TSocketStream is unbuffered, so every frame is on the
// wire before the next token decodes - that per-token flush IS the point of
// streaming. A disconnected client raises here (MSG_NOSIGNAL is set by
// fcl-web), which aborts generation through the engine's exception path.
procedure TServerApp.StreamWrite(const S: string);
begin
  if (SSEResp = nil) or (S = '') then exit;
  TStreamableResponse(SSEResp).Connection.Socket.WriteBuffer(S[1], Length(S));
end;

// Headers go out LAZILY, with the first token: anything that fails before
// decoding starts (template error, context full) can still be answered with
// an ordinary JSON 400, because nothing is on the wire yet. No
// Content-Length is ever set (so none is emitted - CollectHeaders skips
// unset headers): the stream ends when the connection closes after [DONE],
// which fcl-web 3.2.2 does after every request anyway.
procedure TServerApp.StreamEnsureHeaders();
begin
  if SSEHeadersSent then exit;
  SSEResp.Code := 200;
  SSEResp.ContentType := 'text/event-stream';
  SSEResp.SetCustomHeader('Cache-Control', 'no-cache');
  SSEResp.SetCustomHeader('Connection', 'close');
  SSEResp.SendHeaders();
  SSEHeadersSent := true;
  // The first chat chunk carries the role so clients can build the
  // assistant message before any text arrives (api.openai.com does this).
  if SSEChat then StreamWrite(StreamChunkFrame('', true, ''));
end;

// The engine's OnToken sink while a streaming request is served.
procedure TServerApp.StreamTokenOut(const S: string);
begin
  StreamEnsureHeaders();
  if S <> '' then StreamWrite(StreamChunkFrame(S, false, ''));
end;

// Trailer after a successful generation: the finish_reason frame, the usage
// frame when stream_options.include_usage asked for it, then [DONE]. A
// zero-token reply reaches this without headers sent - EnsureHeaders covers
// it (role chunk + finish frame, no content frames).
procedure TServerApp.StreamFinish();
var
  Root: TJSONObject;
begin
  StreamEnsureHeaders();
  StreamWrite(StreamChunkFrame('', false, Engine.LastFinishReason));
  if SSEIncludeUsage then
  begin
    Root := TJSONObject.Create([
      'id', SSEId,
      'object', 'chat.completion.chunk',
      'created', SSECreated,
      'model', ModelName,
      'choices', TJSONArray.Create(),
      'usage', UsageObject()]);
    if not SSEChat then Root.Strings['object'] := 'text_completion';
    try
      StreamWrite('data: ' + Root.AsJSON + #10#10);
    finally
      Root.Free;
    end;
  end;
  StreamWrite('data: [DONE]'#10#10);
  SSEResp := nil;
end;

// Serves one streaming generation end to end (both endpoints: ChatMode
// picks ChatReply(Msgs) vs GenerateFromIds(PromptIds), the unused argument
// is nil). Pre-decode failures 400 as plain JSON; a failure AFTER the
// status line is on the wire can only be reported in-band, so it becomes a
// best-effort "data: {error}" frame and the connection closes. The engine
// invalidates its KV cache on any generation exception, so the next
// request decodes cleanly either way.
procedure TServerApp.RunStreaming(ChatMode: boolean;
  const Msgs: TChatMessages; const PromptIds: TNeuralIntegerArray;
  const GenOpt: TChatOptions; IncludeUsage: boolean;
  var AResponse: TFPHTTPConnectionResponse);
var
  TStart: QWord;
  Err: TJSONObject;
  Kind, What: string;
begin
  Inc(RequestCount);
  if ChatMode then
  begin
    SSEId := 'chatcmpl-' + IntToStr(RequestCount);
    Kind := 'chat.completion';
    What := 'conversation';
  end
  else
  begin
    SSEId := 'cmpl-' + IntToStr(RequestCount);
    Kind := 'text_completion';
    What := 'prompt';
  end;
  SSEChat := ChatMode;
  SSECreated := DateTimeToUnix(Now, false);
  SSEIncludeUsage := IncludeUsage;
  SSEResp := AResponse;
  SSEHeadersSent := false;
  TStart := GetTickCount64();
  Engine.OnToken := @StreamTokenOut;
  try
    try
      if ChatMode then
        Engine.ChatReply(Msgs, GenOpt)
      else
        Engine.GenerateFromIds(PromptIds, GenOpt);
    finally
      Engine.OnToken := nil;
    end;
  except
    on E: ENeuralChatError do
    begin
      // Template rendering fails before any decode step, so nothing is on
      // the wire yet and a plain 400 is still possible.
      SSEResp := nil;
      SendError(AResponse, 400, 'chat template error: ' + E.Message);
      exit;
    end;
    on E: Exception do
    begin
      if not SSEHeadersSent then
      begin
        SSEResp := nil;
        raise; // the HandleRequest catch-all turns it into a JSON 500
      end;
      Err := TJSONObject.Create(['error', TJSONObject.Create([
        'message', E.ClassName + ': ' + E.Message,
        'type', 'server_error'])]);
      try
        try
          StreamWrite('data: ' + Err.AsJSON + #10#10);
        except
          // the usual cause IS a dead client socket - nothing to tell it
        end;
      finally
        Err.Free;
      end;
      SSEResp := nil;
      WriteLn('[error mid-stream] ', E.ClassName, ': ', E.Message);
      exit;
    end;
  end;
  if Engine.ContextFull then
  begin
    // The engine bails out before the first decode step, so no frame has
    // been written and this is still a plain JSON 400.
    SSEResp := nil;
    SendError(AResponse, 400, Format('prompt is %d tokens but the context' +
      ' window is %d - shorten the %s or relaunch with a larger --ctx',
      [Engine.LastPromptTokens, Engine.SeqLen, What]));
    exit;
  end;
  StreamFinish();
  WriteLn(Format('[%s %d streamed] %d prompt + %d completion tokens,' +
    ' %.1fs, finish %s', [Kind, RequestCount, Engine.LastPromptTokens,
    Engine.LastCompletionTokens, (GetTickCount64() - TStart) / 1000,
    Engine.LastFinishReason]));
end;

procedure TServerApp.HandleChatCompletions(Req: TJSONObject;
  var AResponse: TFPHTTPConnectionResponse);
var
  Msgs: TChatMessages;
  GenOpt: TChatOptions;
  Reply, ErrMsg: string;
  IgnoredStop, Stream, IncludeUsage: boolean;
  Root, Choice, Msg: TJSONObject;
  TStart: QWord;
begin
  if Engine.RawMode then
  begin
    SendError(AResponse, 400, 'the server was launched with --format raw' +
      ' (no chat template) - use POST /v1/completions');
    exit;
  end;
  if not ParseRequestMessages(Req, Msgs, ErrMsg) then
  begin
    SendError(AResponse, 400, ErrMsg);
    exit;
  end;
  if not OverlayRequestOptions(Engine.Opt, Req, GenOpt, ErrMsg,
    IgnoredStop, Stream, IncludeUsage) then
  begin
    SendError(AResponse, 400, ErrMsg);
    exit;
  end;
  if IgnoredStop then
    WriteLn('[request "stop" ignored - the engine stops on EOS and the chat',
      ' format''s end-of-turn marker]');
  if Stream then
  begin
    RunStreaming(true, Msgs, nil, GenOpt, IncludeUsage, AResponse);
    exit;
  end;
  TStart := GetTickCount64();
  try
    Reply := Engine.ChatReply(Msgs, GenOpt);
  except
    on E: ENeuralChatError do
    begin
      // e.g. a system message on a chat format without a system role.
      SendError(AResponse, 400, 'chat template error: ' + E.Message);
      exit;
    end;
  end;
  if Engine.ContextFull then
  begin
    SendError(AResponse, 400, Format('prompt is %d tokens but the context' +
      ' window is %d - shorten the conversation or relaunch with a larger' +
      ' --ctx', [Engine.LastPromptTokens, Engine.SeqLen]));
    exit;
  end;
  Inc(RequestCount);
  Msg := TJSONObject.Create(['role', 'assistant', 'content', Reply]);
  Choice := TJSONObject.Create(['index', 0, 'message', Msg,
    'finish_reason', Engine.LastFinishReason]);
  Root := TJSONObject.Create([
    'id', 'chatcmpl-' + IntToStr(RequestCount),
    'object', 'chat.completion',
    'created', DateTimeToUnix(Now, false),
    'model', ModelName,
    'choices', TJSONArray.Create([Choice]),
    'usage', UsageObject()]);
  SendJSON(AResponse, 200, Root);
  WriteLn(Format('[chat.completion %d] %d prompt + %d completion tokens,' +
    ' %.1fs, finish %s', [RequestCount, Engine.LastPromptTokens,
    Engine.LastCompletionTokens, (GetTickCount64() - TStart) / 1000,
    Engine.LastFinishReason]));
end;

procedure TServerApp.HandleCompletions(Req: TJSONObject;
  var AResponse: TFPHTTPConnectionResponse);
var
  Node: TJSONData;
  GenOpt: TChatOptions;
  PromptIds: TNeuralIntegerArray;
  Reply, ErrMsg: string;
  IgnoredStop, Stream, IncludeUsage: boolean;
  Root, Choice: TJSONObject;
  TStart: QWord;
begin
  Node := Req.Find('prompt');
  if not (Assigned(Node) and (Node.JSONType = jtString)) then
  begin
    SendError(AResponse, 400, '"prompt" must be a string');
    exit;
  end;
  if not OverlayRequestOptions(Engine.Opt, Req, GenOpt, ErrMsg,
    IgnoredStop, Stream, IncludeUsage) then
  begin
    SendError(AResponse, 400, ErrMsg);
    exit;
  end;
  if IgnoredStop then
    WriteLn('[request "stop" ignored - the engine stops on EOS]');
  TStart := GetTickCount64();
  // No chat template: the prompt is encoded verbatim and the model continues
  // it (the ChatTerminal --format raw path, available in every server mode).
  PromptIds := Engine.Tokenizer.Encode(Node.AsString);
  if Length(PromptIds) = 0 then
  begin
    // A BOS-less tokenizer encodes '' to zero ids; there is no last token
    // to feed the first decode step, so decoding cannot start.
    SendError(AResponse, 400, '"prompt" encodes to 0 tokens');
    exit;
  end;
  if Stream then
  begin
    RunStreaming(false, nil, PromptIds, GenOpt, IncludeUsage, AResponse);
    exit;
  end;
  Reply := Engine.GenerateFromIds(PromptIds, GenOpt);
  if Engine.ContextFull then
  begin
    SendError(AResponse, 400, Format('prompt is %d tokens but the context' +
      ' window is %d - shorten the prompt or relaunch with a larger --ctx',
      [Engine.LastPromptTokens, Engine.SeqLen]));
    exit;
  end;
  Inc(RequestCount);
  Choice := TJSONObject.Create(['index', 0, 'text', Reply,
    'finish_reason', Engine.LastFinishReason]);
  Root := TJSONObject.Create([
    'id', 'cmpl-' + IntToStr(RequestCount),
    'object', 'text_completion',
    'created', DateTimeToUnix(Now, false),
    'model', ModelName,
    'choices', TJSONArray.Create([Choice]),
    'usage', UsageObject()]);
  SendJSON(AResponse, 200, Root);
  WriteLn(Format('[text_completion %d] %d prompt + %d completion tokens,' +
    ' %.1fs, finish %s', [RequestCount, Engine.LastPromptTokens,
    Engine.LastCompletionTokens, (GetTickCount64() - TStart) / 1000,
    Engine.LastFinishReason]));
end;

procedure TServerApp.HandleModels(var AResponse: TFPHTTPConnectionResponse);
var
  Model: TJSONObject;
begin
  Model := TJSONObject.Create([
    'id', ModelName,
    'object', 'model',
    'created', DateTimeToUnix(Now, false),
    'owned_by', 'neural-api']);
  SendJSON(AResponse, 200, TJSONObject.Create([
    'object', 'list',
    'data', TJSONArray.Create([Model])]));
end;

procedure TServerApp.HandleRequest(Sender: TObject;
  var ARequest: TFPHTTPConnectionRequest;
  var AResponse: TFPHTTPConnectionResponse);
var
  Path: string;
  Parser: TJSONParser;
  Root: TJSONData;
begin
  Path := ARequest.URI;
  try
    if (ARequest.Method = 'GET') and (Path = '/v1/models') then
    begin
      HandleModels(AResponse);
      exit;
    end;
    if (ARequest.Method = 'POST') and
      ((Path = '/v1/chat/completions') or (Path = '/v1/completions')) then
    begin
      Root := nil;
      try
        try
          Parser := TJSONParser.Create(ARequest.Content, []);
          try
            Root := Parser.Parse();
          finally
            Parser.Free;
          end;
        except
          on E: Exception do
          begin
            SendError(AResponse, 400, 'invalid JSON: ' + E.Message);
            exit;
          end;
        end;
        if not (Assigned(Root) and (Root.JSONType = jtObject)) then
        begin
          SendError(AResponse, 400, 'request body must be a JSON object');
          exit;
        end;
        if Path = '/v1/chat/completions' then
          HandleChatCompletions(TJSONObject(Root), AResponse)
        else
          HandleCompletions(TJSONObject(Root), AResponse);
      finally
        Root.Free;
      end;
      exit;
    end;
    SendError(AResponse, 404, 'no such endpoint: ' + ARequest.Method + ' ' +
      Path + ' (POST /v1/chat/completions, POST /v1/completions,' +
      ' GET /v1/models)');
  except
    // Nothing may escape into the accept loop: any unexpected failure
    // becomes a 500 and the server keeps serving.
    on E: Exception do
      SendError(AResponse, 500, E.ClassName + ': ' + E.Message);
  end;
end;

// ---------------------------------------------------------------------------
// --selftest: offline unit checks (no model files needed).
// ---------------------------------------------------------------------------
procedure SelfTest();
var
  Failures: integer;

  procedure Check(Condition: boolean; const What: string);
  begin
    if Condition then WriteLn('PASS: ', What)
    else
    begin
      WriteLn('FAIL: ', What);
      Inc(Failures);
    end;
  end;

  function ParseObj(const S: string): TJSONObject;
  var
    Parser: TJSONParser;
  begin
    Parser := TJSONParser.Create(S, []);
    try
      Result := TJSONObject(Parser.Parse());
    finally
      Parser.Free;
    end;
  end;

var
  Base, GenOpt: TChatOptions;
  Req: TJSONObject;
  Msgs: TChatMessages;
  ErrMsg: string;
  IgnoredStop, Stream, IncludeUsage: boolean;
begin
  Failures := 0;
  Base := DefaultChatOptions();
  Base.Temperature := 0.7;
  Base.TopP := 0.8;
  Base.MaxNewTokens := 100;

  // Absent fields keep the launch values.
  Req := ParseObj('{"messages":[{"role":"user","content":"hi"}]}');
  try
    Check(OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'empty overlay accepted');
    Check((Abs(GenOpt.Temperature - 0.7) < 1e-6) and
      (Abs(GenOpt.TopP - 0.8) < 1e-6) and (GenOpt.MaxNewTokens = 100),
      'absent fields keep the launch defaults');
    Check(not IgnoredStop, 'no stop field reported');
    Check(ParseRequestMessages(Req, Msgs, ErrMsg) and (Length(Msgs) = 1) and
      (Msgs[0].Role = 'user') and (Msgs[0].Content = 'hi'),
      'messages array parses');
  finally
    Req.Free;
  end;

  // Present fields override - including explicit zeros (JSON-level
  // presence, not value sentinels).
  Req := ParseObj('{"temperature":0.0,"top_p":0.5,"max_tokens":7,' +
    '"presence_penalty":0.25,"stop":["x"]}');
  try
    Check(OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'override overlay accepted');
    Check(Abs(GenOpt.Temperature) < 1e-6, 'explicit 0.0 temperature applies');
    Check(Abs(GenOpt.TopP - 0.5) < 1e-6, 'request top_p overrides');
    Check(GenOpt.MaxNewTokens = 7, 'request max_tokens overrides');
    Check(Abs(GenOpt.PresencePenalty - 0.25) < 1e-6,
      'request presence_penalty overrides');
    Check(IgnoredStop, 'stop field reported as ignored');
  finally
    Req.Free;
  end;

  // top_k maps to the weighted top-k; max_completion_tokens beats max_tokens.
  Req := ParseObj('{"top_k":20,"max_tokens":5,"max_completion_tokens":9}');
  try
    Check(OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage) and
      (GenOpt.TopK = 20) and GenOpt.WeightedTopK,
      'request top_k maps to weighted top-k');
    Check(GenOpt.MaxNewTokens = 9, 'max_completion_tokens wins');
  finally
    Req.Free;
  end;

  // stream must be a literal boolean; stream_options rides on stream:true.
  Req := ParseObj('{"stream":true}');
  try
    Check(OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage) and Stream and not IncludeUsage,
      'stream:true accepted and reported');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"stream":false}');
  try
    Check(OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage) and not Stream,
      'stream:false accepted, not streaming');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"messages":[]}');
  try
    Check(OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage) and not Stream and not IncludeUsage,
      'absent stream defaults to non-streaming');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"stream":true,"stream_options":{"include_usage":true}}');
  try
    Check(OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage) and Stream and IncludeUsage,
      'stream_options.include_usage reported');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"stream":true,"stream_options":{"include_usage":false}}');
  try
    Check(OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage) and Stream and not IncludeUsage,
      'include_usage:false stays off');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"stream_options":{"include_usage":true}}');
  try
    Check(not OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'stream_options without stream:true rejected');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"stream":true,"stream_options":{"include_usage":"yes"}}');
  try
    Check(not OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'non-boolean include_usage rejected');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"stream":1}');
  try
    Check(not OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'stream as a number rejected');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"n":2}');
  try
    Check(not OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'n=2 rejected');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"n":1}');
  try
    Check(OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'n=1 accepted');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"n":1.0}');
  try
    Check(OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'n=1.0 accepted (float one)');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"stream":"true"}');
  try
    Check(not OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'stream as a string rejected (only a literal boolean streams or not)');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"n":"1"}');
  try
    Check(not OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'n as a string rejected');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"max_tokens":0}');
  try
    Check(not OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'max_tokens=0 rejected');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"max_tokens":1e12}');
  try
    Check(not OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'absurd max_tokens rejected as 400, not a Round overflow 500');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"top_k":-1}');
  try
    Check(not OverlayRequestOptions(Base, Req, GenOpt, ErrMsg, IgnoredStop,
      Stream, IncludeUsage),
      'negative top_k rejected');
  finally
    Req.Free;
  end;

  // Malformed messages arrays.
  Req := ParseObj('{}');
  try
    Check(not ParseRequestMessages(Req, Msgs, ErrMsg),
      'missing messages rejected');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"messages":[]}');
  try
    Check(not ParseRequestMessages(Req, Msgs, ErrMsg),
      'empty messages rejected');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"messages":[{"role":"user"}]}');
  try
    Check(not ParseRequestMessages(Req, Msgs, ErrMsg),
      'message without content rejected');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"messages":[{"role":"","content":"x"}]}');
  try
    Check(not ParseRequestMessages(Req, Msgs, ErrMsg),
      'empty role rejected');
  finally
    Req.Free;
  end;
  Req := ParseObj('{"messages":[{"role":"system","content":"a"},' +
    '{"role":"user","content":"b"}]}');
  try
    Check(ParseRequestMessages(Req, Msgs, ErrMsg) and (Length(Msgs) = 2) and
      (Msgs[0].Role = 'system') and (Msgs[1].Content = 'b'),
      'multi-message array parses in order');
  finally
    Req.Free;
  end;

  if Failures = 0 then WriteLn('SELFTEST OK')
  else
  begin
    WriteLn('SELFTEST FAILED: ', Failures, ' check(s)');
    Halt(1);
  end;
end;

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------
var
  Opt: TChatOptions;
  Args: TStringList;
  App: TServerApp;
  Server: TChatHTTPServer;
  Cnt: integer;
  ErrorMsg: string;
begin
  Args := TStringList.Create();
  for Cnt := 1 to ParamCount do Args.Add(ParamStr(Cnt));
  if not ParseArgs(Args, Opt) then
  begin
    WriteLn('Error: ', Opt.ErrorMsg);
    WriteLn;
    PrintUsage();
    Args.Free;
    Halt(1);
  end;
  Args.Free;
  if Opt.SelfTest then
  begin
    SelfTest();
    Halt(0);
  end;
  if Opt.ShowHelp or (Opt.ModelDir = '') then
  begin
    PrintUsage();
    if Opt.ModelDir = '' then Halt(1);
    Halt(0);
  end;

  App := TServerApp.Create();
  App.Engine := TChatEngine.Create();
  App.Engine.OnNotice := @App.NoticeOut; // load progress + notices to console
  App.RequestCount := 0;
  if not App.Engine.LoadModel(Opt, ErrorMsg) then
  begin
    WriteLn(ErrorMsg);
    App.Engine.Free;
    App.Free;
    Halt(1);
  end;
  App.ModelName :=
    ExtractFileName(ExcludeTrailingPathDelimiter(App.Engine.Opt.ModelDir));
  if App.ModelName = '' then App.ModelName := App.Engine.ModelType;

  Server := TChatHTTPServer.Create(nil);
  try
    Server.Address := App.Engine.Opt.Host;
    Server.Port := App.Engine.Opt.Port;
    Server.Threaded := false; // one request at a time on the accept loop:
                              // the engine's serialization guarantee
    Server.OnRequest := @App.HandleRequest;
    WriteLn(Format('Serving %s on http://%s:%d/v1 (SSE streaming with' +
      ' "stream":true; Ctrl+C stops)', [App.ModelName, App.Engine.Opt.Host,
      App.Engine.Opt.Port]));
    Server.Active := true; // blocks: the accept/serve loop
  finally
    Server.Free;
    App.Engine.Free;
    App.Free;
  end;
end.
