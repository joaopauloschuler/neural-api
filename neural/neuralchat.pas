(*
neuralchat
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

// neuralchat -- chat templates (the HF transformers apply_chat_template
// equivalent) for the instruct checkpoints whose base architectures the
// importers in neuralpretrained.pas load, so an imported instruct model
// can be prompted with a correctly formatted conversation end to end:
//
//   Messages := [ChatMessage('system', 'Be brief.'),
//                ChatMessage('user', 'Hi!')];
//   Prompt := ApplyChatTemplate(cfChatML, Messages, {AddGenPrompt=}true);
//   Ids    := EncodeChat(Tokenizer, cfChatML, Messages, true);
//
// Full Jinja2 is OUT OF SCOPE. Instead the seven well-known turn formats
// are hardcoded and pinned, byte for byte, against ground truth produced
// by rendering the AUTHENTIC published chat_template Jinja strings with
// the exact transformers Jinja environment (trim_blocks + lstrip_blocks;
// see tools/chat_template_fixture.py and
// tests/fixtures/chat_template_cases.json):
//
//   cfChatML  <|im_start|>role\n...<|im_end|>\n     Qwen2/Qwen2.5/Qwen3,
//             Yi and the wider ChatML family. The Qwen templates only add
//             a default system message on top -- pass one explicitly for
//             byte parity with Qwen's apply_chat_template.
//   cfLlama2  <s>[INST] ... [/INST] answer </s>     Llama-2-chat; the
//             system message is folded into the first user turn inside
//             <<SYS>>\n...\n<</SYS>>\n\n; contents are stripped; roles
//             must alternate user/assistant (raises otherwise, like HF).
//   cfLlama3  <|begin_of_text|><|start_header_id|>role<|end_header_id|>
//             \n\ncontent<|eot_id|>                  Llama-3/3.1-Instruct.
//   cfZephyr  <|role|>\ncontent</s>\n                zephyr-7b-beta and
//             TinyLlama-1.1B-Chat (identical template); unknown roles are
//             silently skipped (HF template has no else branch).
//   cfGemma   <bos><start_of_turn>user|model\n...<end_of_turn>\n
//             gemma-it (1/2/3 share the turn format). NO system role --
//             raises like the HF template; roles must alternate.
//   cfPhi3    <|role|>\ncontent<|end|>\n             Phi-3-mini-instruct;
//             without a generation prompt the HF template appends
//             <|endoftext|>.
//   cfMistral <s>[INST] ... [/INST]answer</s>        Mistral-7B-Instruct
//             v0.1/v0.2; NO system role and strict alternation (raises).
//   cfDeepSeek <｜begin▁of▁sentence｜>system User: ...\n\n
//             Assistant: answer<｜end▁of▁sentence｜>        DeepSeek-V2/V3-Chat;
//             the bos / end-of-sentence tokens use the fullwidth pipe
//             U+FF5C and the one-eighth block U+2581; system content is
//             emitted verbatim with no role tag; generation prompt is the
//             bare 'Assistant:'. No content strip; no role alternation check.
//   cfPhi4Mini <|system|>...<|end|><|user|>...<|end|><|assistant|>
//             Phi-4-mini-instruct ChatML-style tool-aware template; like
//             cfPhi3 but the tags carry NO trailing newline and there is no
//             eos fallback when add_generation_prompt is false.
//
// Auto-detection does NOT interpret Jinja: DetectChatFormat fingerprints
// the chat_template string from tokenizer_config.json by its distinctive
// control tokens (e.g. '<|im_start|>' -> cfChatML) and the explicit
// format parameter is the fallback for anything unrecognized.

unit neuralchat;
{$include neuralnetwork.inc}
{$H+}

interface

uses
  Classes, SysUtils, neuralvolume, neuralhftokenizer;

type
  ENeuralChatError = class(Exception);
  // Raised by the mini-Jinja interpreter on an unsupported construct or a
  // template-side raise_exception(...) call. A subclass of ENeuralChatError
  // so callers that already catch ENeuralChatError keep working.
  EChatTemplateError = class(ENeuralChatError);

  TNeuralChatFormat = (
    cfUnknown,
    cfChatML,   // Qwen / Yi / ChatML family
    cfLlama2,   // Llama-2-chat [INST] + <<SYS>>
    cfLlama3,   // Llama-3-Instruct header format
    cfZephyr,   // zephyr-7b-beta / TinyLlama-1.1B-Chat
    cfGemma,    // gemma-it <start_of_turn>
    cfPhi3,     // Phi-3-mini-instruct <|user|>...<|end|>
    cfMistral,  // Mistral-7B-Instruct [INST] without system
    cfDeepSeek, // DeepSeek-V2/V3-Chat <｜begin▁of▁sentence｜>User: ...
    cfPhi4Mini  // Phi-4-mini-instruct ChatML-style (no-newline tags)
  );

  TChatMessage = record
    Role: string;    // 'system' | 'user' | 'assistant'
    Content: string;
  end;
  TChatMessages = array of TChatMessage;

// Convenience constructor: ChatMessage('user', 'Hi!').
function ChatMessage(const Role, Content: string): TChatMessage;

// Renders the conversation in the given format. AddGenerationPrompt
// appends the assistant turn opener so the model continues as the
// assistant (formats where the template ignores the flag -- cfLlama2,
// cfMistral -- behave like HF: the [/INST] tail already cues the model).
// Raises ENeuralChatError exactly where the HF template raises
// (cfGemma system turn, cfLlama2/cfGemma/cfMistral role alternation,
// cfMistral non-user/assistant role).
function ApplyChatTemplate(ChatFormat: TNeuralChatFormat;
  const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean = true): string;

// Resolves an arbitrary chat_template string to a rendered prompt. First
// tries DetectChatFormat: a recognized family routes to the fast hardcoded
// renderer (ignoring the supplied template body). Otherwise the mini-Jinja
// interpreter renders the template directly (the v2 fallback path), with
// BosToken/EosToken substituted for {{ bos_token }}/{{ eos_token }}. Raises
// EChatTemplateError on an unsupported Jinja construct.
function ApplyChatTemplateString(const ChatTemplate: string;
  const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean = true;
  const BosToken: string = ''; const EosToken: string = ''): string;

// 'chatml' <-> cfChatML etc; ChatFormatFromName returns cfUnknown for
// unrecognized names.
function ChatFormatName(ChatFormat: TNeuralChatFormat): string;
function ChatFormatFromName(const Name: string): TNeuralChatFormat;

// Fingerprints a chat_template Jinja string (no Jinja interpretation --
// substring matching on the distinctive control tokens). cfUnknown when
// nothing matches.
function DetectChatFormat(const ChatTemplate: string): TNeuralChatFormat;

// Reads the raw chat_template string from a HF tokenizer_config.json
// ('' when the file has none). The newer list-of-named-templates form
// ([{"name": ..., "template": ...}]) resolves to the 'default' entry or,
// absent one, the first entry. When tokenizer_config.json carries no
// chat_template field, newer transformers exports it to a sibling
// chat_template.jinja in the same directory; that file is read as the
// fallback.
function LoadChatTemplateString(const TokenizerConfigFile: string): string;

// LoadChatTemplateString + DetectChatFormat in one call.
function DetectChatFormatFromConfigFile(
  const TokenizerConfigFile: string): TNeuralChatFormat;

// Mini-Jinja subset INTERPRETER (chat templates v2). Renders an HF
// chat_template Jinja string directly, for checkpoints whose template does
// NOT match any hardcoded TNeuralChatFormat. This is the fallback path: it
// reproduces, byte for byte, the output of every hardcoded format when fed
// that family's authentic chat_template string, and raises a clean
// EChatTemplateError on any construct outside the supported subset (rather
// than silently emitting garbage).
//
// Supported constructs (the union actually used by the real published HF
// chat templates this repo bundles):
//   * {{ expr }} output, {%- / -%} whitespace control, and the implicit
//     trim_blocks + lstrip_blocks transformers uses by default.
//   * {% for x in seq %}...{% endfor %} over messages (and message slices),
//     with loop.index0 / loop.index / loop.first / loop.last.
//   * {% if c %}...{% elif c %}...{% else %}...{% endif %}.
//   * {% set name = expr %} (incl. list slices like messages[1:]).
//   * Variable / index / attribute access: messages, messages[0],
//     message['role'], message.content, message['content'], loop.last.
//   * Specials add_generation_prompt, bos_token, eos_token.
//   * Operators: + (string concat / int add), ~ (concat), % (mod),
//     == != < <= > >= , and / or / not , 'is defined'.
//   * String/int/bool literals (true/false), parentheses.
//   * Filters .strip() / | trim, | default(value), and the
//     raise_exception('msg') call (raises EChatTemplateError).
// Anything else raises EChatTemplateError.
function RenderChatTemplate(const Template: string;
  const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean;
  const BosToken: string = ''; const EosToken: string = ''): string;

// ApplyChatTemplate followed by Tokenizer.Encode. The rendered template
// embeds control tokens (<|im_start|>, <s>, ...) as TEXT; the tokenizer
// matches them verbatim through its added-tokens mechanism, so they map
// to single ids exactly when they exist as added tokens in tokenizer.json
// (true for every supported instruct family).
procedure EncodeChat(Tokenizer: TNeuralHFTokenizer;
  ChatFormat: TNeuralChatFormat; const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean; Ids: TIntegerList); overload;
function EncodeChat(Tokenizer: TNeuralHFTokenizer;
  ChatFormat: TNeuralChatFormat; const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean = true): TNeuralIntegerArray; overload;

implementation

uses
  fpjson;

const
  csAlternateError =
    'Conversation roles must alternate user/assistant/user/assistant/...';
  // DeepSeek special tokens: '<' + U+FF5C + 'begin' + U+2581 + 'of' +
  // U+2581 + 'sentence' + U+FF5C + '>' (and 'end' for the eos). Written as
  // raw UTF-8 byte sequences so the source stays encoding-agnostic.
  csPipe = #$EF#$BD#$9C;    // U+FF5C FULLWIDTH VERTICAL LINE
  csBlock = #$E2#$96#$81;   // U+2581 LOWER ONE EIGHTH BLOCK
  csDeepSeekBos =
    '<' + csPipe + 'begin' + csBlock + 'of' + csBlock + 'sentence' +
    csPipe + '>';
  csDeepSeekEos =
    '<' + csPipe + 'end' + csBlock + 'of' + csBlock + 'sentence' +
    csPipe + '>';

// Python str.strip() / Jinja "| trim" equivalent: trims ASCII whitespace
// only (space, \t, \n, \r, \v, \f), NOT the other control chars that
// Pascal's Trim also eats.
function PyStrip(const S: string): string;
var
  First, Last: integer;
begin
  First := 1;
  Last := Length(S);
  while (First <= Last) and (S[First] in [' ', #9, #10, #13, #11, #12]) do
    Inc(First);
  while (Last >= First) and (S[Last] in [' ', #9, #10, #13, #11, #12]) do
    Dec(Last);
  Result := Copy(S, First, Last - First + 1);
end;

function ChatMessage(const Role, Content: string): TChatMessage;
begin
  Result.Role := Role;
  Result.Content := Content;
end;

function ChatFormatName(ChatFormat: TNeuralChatFormat): string;
begin
  case ChatFormat of
    cfChatML: Result := 'chatml';
    cfLlama2: Result := 'llama2';
    cfLlama3: Result := 'llama3';
    cfZephyr: Result := 'zephyr';
    cfGemma: Result := 'gemma';
    cfPhi3: Result := 'phi3';
    cfMistral: Result := 'mistral';
    cfDeepSeek: Result := 'deepseek';
    cfPhi4Mini: Result := 'phi4mini';
    else Result := 'unknown';
  end;
end;

function ChatFormatFromName(const Name: string): TNeuralChatFormat;
var
  Lowered: string;
begin
  Lowered := LowerCase(Name);
  if Lowered = 'chatml' then Result := cfChatML
  else if Lowered = 'llama2' then Result := cfLlama2
  else if Lowered = 'llama3' then Result := cfLlama3
  else if Lowered = 'zephyr' then Result := cfZephyr
  else if Lowered = 'gemma' then Result := cfGemma
  else if Lowered = 'phi3' then Result := cfPhi3
  else if Lowered = 'mistral' then Result := cfMistral
  else if Lowered = 'deepseek' then Result := cfDeepSeek
  else if Lowered = 'phi4mini' then Result := cfPhi4Mini
  else Result := cfUnknown;
end;

function DetectChatFormat(const ChatTemplate: string): TNeuralChatFormat;
begin
  // ordered by specificity: every fingerprint below is unique to its
  // family among the published templates (cfZephyr and cfPhi3 share
  // '<|user|>', so cfPhi3's '<|end|>' is tested first; '[INST]' is
  // shared by cfLlama2 and cfMistral, so cfLlama2's '<<SYS>>' wins).
  if Pos('<|im_start|>', ChatTemplate) > 0 then Result := cfChatML
  else if Pos('<|start_header_id|>', ChatTemplate) > 0 then
    Result := cfLlama3
  else if Pos('<<SYS>>', ChatTemplate) > 0 then Result := cfLlama2
  else if Pos('<start_of_turn>', ChatTemplate) > 0 then Result := cfGemma
  // DeepSeek: either the rendered begin-of-sentence token (fullwidth pipe
  // U+FF5C + block U+2581) or, in the raw Jinja (where the token hides
  // behind {{ bos_token }}), the distinctive 'User: '/'Assistant: ' prefix
  // literal pair is unique to the family.
  else if (Pos(csDeepSeekBos, ChatTemplate) > 0) or
    ((Pos('''User: ''', ChatTemplate) > 0) and
     (Pos('''Assistant: ''', ChatTemplate) > 0)) then Result := cfDeepSeek
  // cfPhi3 and cfPhi4Mini share <|user|> + <|end|>; cfPhi3's tags carry a
  // trailing newline ('<|user|>\n'), cfPhi4Mini's do not -- test cfPhi3
  // first so the Phi-4-mini no-newline ChatML lands in its own branch.
  else if (Pos('<|end|>', ChatTemplate) > 0) and
    (Pos('<|user|>' + #10, ChatTemplate) > 0) then Result := cfPhi3
  else if (Pos('<|end|>', ChatTemplate) > 0) and
    (Pos('<|user|>', ChatTemplate) > 0) then Result := cfPhi4Mini
  else if Pos('<|user|>', ChatTemplate) > 0 then Result := cfZephyr
  else if Pos('[INST]', ChatTemplate) > 0 then Result := cfMistral
  else Result := cfUnknown;
end;

// Reads an entire file into a string (raw bytes, no encoding translation).
function ReadFileToString(const FileName: string): string;
var
  FS: TFileStream;
begin
  Result := '';
  FS := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    SetLength(Result, FS.Size);
    if FS.Size > 0 then FS.ReadBuffer(Result[1], FS.Size);
  finally
    FS.Free;
  end;
end;

// Extracts the chat_template field from an already-loaded tokenizer_config
// JSON string ('' when absent).
function ExtractChatTemplateFromJson(const RawJson, ConfigFile: string): string;
var
  Root, Node, Entry: TJSONData;
  Arr: TJSONArray;
  Cnt: integer;
begin
  Result := '';
  // same fpjson stance as neuralhftokenizer: decode \uXXXX up front and
  // parse raw so non-ASCII survives without a widestring manager.
  Root := HFParseJSONRaw(HFDecodeUnicodeEscapes(RawJson));
  try
    if not (Root is TJSONObject) then
      raise ENeuralChatError.Create(ConfigFile + ': root is not an object');
    Node := TJSONObject(Root).Find('chat_template');
    if (Node = nil) or Node.IsNull then exit;
    if Node is TJSONArray then
    begin // [{"name": ..., "template": ...}, ...]
      Arr := TJSONArray(Node);
      for Cnt := 0 to Arr.Count - 1 do
      begin
        Entry := Arr.Items[Cnt];
        if (Entry is TJSONObject) and
          (TJSONObject(Entry).Get('name', '') = 'default') then
          Exit(TJSONObject(Entry).Get('template', ''));
      end;
      if (Arr.Count > 0) and (Arr.Items[0] is TJSONObject) then
        Result := TJSONObject(Arr.Items[0]).Get('template', '');
    end
    else
      Result := Node.AsString;
  finally
    Root.Free;
  end;
end;

function LoadChatTemplateString(const TokenizerConfigFile: string): string;
var
  JinjaFile: string;
begin
  Result := ExtractChatTemplateFromJson(
    ReadFileToString(TokenizerConfigFile), TokenizerConfigFile);
  if Result <> '' then exit;
  // Newer transformers exports the template to a sibling chat_template.jinja
  // instead of embedding it in tokenizer_config.json. The .jinja file holds
  // the raw Jinja source directly (no JSON wrapper).
  JinjaFile := ExtractFilePath(TokenizerConfigFile) + 'chat_template.jinja';
  if FileExists(JinjaFile) then
    Result := ReadFileToString(JinjaFile);
end;

function DetectChatFormatFromConfigFile(
  const TokenizerConfigFile: string): TNeuralChatFormat;
begin
  Result := DetectChatFormat(LoadChatTemplateString(TokenizerConfigFile));
end;

// One renderer per format; each mirrors its Jinja template line by line
// (the comment above every branch quotes the construct it reproduces).

function RenderChatML(const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean): string;
var
  Cnt: integer;
begin
  Result := '';
  for Cnt := 0 to High(Messages) do
    Result := Result + '<|im_start|>' + Messages[Cnt].Role + #10 +
      Messages[Cnt].Content + '<|im_end|>' + #10;
  if AddGenerationPrompt then
    Result := Result + '<|im_start|>assistant' + #10;
end;

function RenderLlama2(const Messages: array of TChatMessage): string;
var
  Cnt, LoopFirst, LoopIdx: integer;
  SystemMessage, Content: string;
  HasSystem: boolean;
begin
  Result := '';
  HasSystem := (Length(Messages) > 0) and (Messages[0].Role = 'system');
  if HasSystem then
  begin
    SystemMessage := Messages[0].Content;
    LoopFirst := 1;
  end
  else
    LoopFirst := 0;
  for Cnt := LoopFirst to High(Messages) do
  begin
    LoopIdx := Cnt - LoopFirst;
    // {% if (role == 'user') != (loop.index0 % 2 == 0) %}raise{% endif %}
    if (Messages[Cnt].Role = 'user') <> (LoopIdx mod 2 = 0) then
      raise ENeuralChatError.Create(csAlternateError);
    if (LoopIdx = 0) and HasSystem then
      Content := '<<SYS>>' + #10 + SystemMessage + #10 + '<</SYS>>' +
        #10#10 + Messages[Cnt].Content
    else
      Content := Messages[Cnt].Content;
    if Messages[Cnt].Role = 'user' then
      Result := Result + '<s>[INST] ' + PyStrip(Content) + ' [/INST]'
    else if Messages[Cnt].Role = 'assistant' then
      Result := Result + ' ' + PyStrip(Content) + ' </s>';
    // other roles: the template has no branch for them -> emit nothing
  end;
  // add_generation_prompt is a no-op: the trailing [/INST] cues the model
end;

function RenderLlama3(const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean): string;
var
  Cnt: integer;
begin
  Result := '';
  for Cnt := 0 to High(Messages) do
  begin
    if Cnt = 0 then Result := Result + '<|begin_of_text|>';
    Result := Result + '<|start_header_id|>' + Messages[Cnt].Role +
      '<|end_header_id|>' + #10#10 + PyStrip(Messages[Cnt].Content) +
      '<|eot_id|>';
  end;
  if AddGenerationPrompt then
    Result := Result + '<|start_header_id|>assistant<|end_header_id|>' +
      #10#10;
end;

function RenderZephyr(const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean): string;
var
  Cnt: integer;
  Role: string;
begin
  Result := '';
  for Cnt := 0 to High(Messages) do
  begin
    Role := Messages[Cnt].Role;
    if (Role = 'user') or (Role = 'system') or (Role = 'assistant') then
      Result := Result + '<|' + Role + '|>' + #10 + Messages[Cnt].Content +
        '</s>' + #10;
    // {% if loop.last and add_generation_prompt %} -- INSIDE the loop, so
    // an empty conversation gets no generation prompt (HF parity).
    if (Cnt = High(Messages)) and AddGenerationPrompt then
      Result := Result + '<|assistant|>' + #10;
  end;
end;

function RenderGemma(const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean): string;
var
  Cnt: integer;
  Role: string;
begin
  Result := '<bos>';
  if (Length(Messages) > 0) and (Messages[0].Role = 'system') then
    raise ENeuralChatError.Create('System role not supported');
  for Cnt := 0 to High(Messages) do
  begin
    if (Messages[Cnt].Role = 'user') <> (Cnt mod 2 = 0) then
      raise ENeuralChatError.Create(csAlternateError);
    if Messages[Cnt].Role = 'assistant' then
      Role := 'model'
    else
      Role := Messages[Cnt].Role;
    Result := Result + '<start_of_turn>' + Role + #10 +
      PyStrip(Messages[Cnt].Content) + '<end_of_turn>' + #10;
  end;
  if AddGenerationPrompt then
    Result := Result + '<start_of_turn>model' + #10;
end;

function RenderPhi3(const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean): string;
var
  Cnt: integer;
  Role: string;
begin
  Result := '';
  for Cnt := 0 to High(Messages) do
  begin
    Role := Messages[Cnt].Role;
    if (Role = 'system') or (Role = 'user') or (Role = 'assistant') then
      Result := Result + '<|' + Role + '|>' + #10 + Messages[Cnt].Content +
        '<|end|>' + #10;
  end;
  if AddGenerationPrompt then
    Result := Result + '<|assistant|>' + #10
  else
    Result := Result + '<|endoftext|>'; // {% else %}{{ eos_token }}
end;

function RenderMistral(const Messages: array of TChatMessage): string;
var
  Cnt: integer;
begin
  Result := '<s>';
  for Cnt := 0 to High(Messages) do
  begin
    if (Messages[Cnt].Role = 'user') <> (Cnt mod 2 = 0) then
      raise ENeuralChatError.Create(csAlternateError);
    if Messages[Cnt].Role = 'user' then
      Result := Result + '[INST] ' + Messages[Cnt].Content + ' [/INST]'
    else if Messages[Cnt].Role = 'assistant' then
      Result := Result + Messages[Cnt].Content + '</s> '
    else
      raise ENeuralChatError.Create(
        'Only user and assistant roles are supported!');
  end;
end;

function RenderDeepSeek(const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean): string;
var
  Cnt: integer;
  Role: string;
begin
  // {{ bos_token }} -- always emitted, exactly once, before the loop.
  Result := csDeepSeekBos;
  for Cnt := 0 to High(Messages) do
  begin
    Role := Messages[Cnt].Role;
    if Role = 'user' then
      Result := Result + 'User: ' + Messages[Cnt].Content + #10#10
    else if Role = 'assistant' then
      Result := Result + 'Assistant: ' + Messages[Cnt].Content + csDeepSeekEos
    else if Role = 'system' then
      Result := Result + Messages[Cnt].Content + #10#10;
    // other roles: the template has no branch for them -> emit nothing.
    // NOTE: no content strip and no role-alternation check (HF parity).
  end;
  if AddGenerationPrompt then
    Result := Result + 'Assistant:';
end;

function RenderPhi4Mini(const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean): string;
var
  Cnt: integer;
  Role: string;
begin
  // Like cfPhi3 but the <|...|> tags carry NO trailing newline, and there
  // is no eos fallback when add_generation_prompt is false.
  Result := '';
  for Cnt := 0 to High(Messages) do
  begin
    Role := Messages[Cnt].Role;
    if (Role = 'system') or (Role = 'user') or (Role = 'assistant') then
      Result := Result + '<|' + Role + '|>' + Messages[Cnt].Content +
        '<|end|>';
  end;
  if AddGenerationPrompt then
    Result := Result + '<|assistant|>';
end;

// ===================================================================
// Mini-Jinja subset interpreter (chat templates v2).
// ===================================================================
//
// Pipeline: Tokenize (split into TEXT / {{...}} / {%...%} chunks honouring
// {%- -%} and trim_blocks+lstrip_blocks) -> recursive descent over the
// chunk stream evaluating expressions against a dynamic scope. There is no
// separate AST: control blocks are executed by scanning the matching
// end tag, because the template language is tiny and linear.

type
  // A dynamic value. Lists/objects are modelled lazily: a value points back
  // into the host Messages array by index range rather than copying.
  TJValueKind = (jvUndefined, jvString, jvInt, jvBool, jvMessages,
    jvMessage, jvLoop);
  TJValue = record
    Kind: TJValueKind;
    Str: string;
    IntV: integer;
    BoolV: boolean;
    // jvMessages: a (possibly sliced) window over the host messages.
    SliceLo, SliceHi: integer;   // inclusive..exclusive host indices
    // jvMessage: index into the host messages array.
    MsgIdx: integer;
    // jvLoop: the loop bookkeeping (index0 within the iterated slice).
    LoopIndex0, LoopCount: integer;
  end;

  TJVarBinding = record
    Name: string;
    Value: TJValue;
  end;

  // The interpreter object. One instance per RenderChatTemplate call.
  TJInterp = class
  private
    FTemplate: string;
    FMessages: array of TChatMessage;
    FAddGen: boolean;
    FBos, FEos: string;
    FVars: array of TJVarBinding;
    FOutput: string;
    // chunk stream
    FChunks: array of record
      Kind: char;   // 'T' text, 'O' output {{ }}, 'S' statement {% %}
      Text: string; // payload (for 'O'/'S' the inner expression, trimmed)
    end;
    procedure Tokenize;
    procedure Fail(const Msg: string);
    function LookupVar(const Name: string; out V: TJValue): boolean;
    procedure SetVar(const Name: string; const V: TJValue);
    function MakeStr(const S: string): TJValue;
    function MakeInt(I: integer): TJValue;
    function MakeBool(B: boolean): TJValue;
    function MakeUndefined: TJValue;
    function ToStr(const V: TJValue): string;
    function Truthy(const V: TJValue): boolean;
    function ValuesEqual(const A, B: TJValue): boolean;
    // expression parser (operates on a single statement/output string)
    function ParseExpr(const S: string; var P: integer): TJValue;
    function ParseOr(const S: string; var P: integer): TJValue;
    function ParseAnd(const S: string; var P: integer): TJValue;
    function ParseNot(const S: string; var P: integer): TJValue;
    function ParseCompare(const S: string; var P: integer): TJValue;
    function ParseConcat(const S: string; var P: integer): TJValue;
    function ParseAddSub(const S: string; var P: integer): TJValue;
    function ParseMulMod(const S: string; var P: integer): TJValue;
    function ParsePostfix(const S: string; var P: integer): TJValue;
    function ParsePrimary(const S: string; var P: integer): TJValue;
    procedure SkipWs(const S: string; var P: integer);
    function PeekWord(const S: string; P: integer; const W: string): boolean;
    // block execution: render chunk range [Lo..Hi) into FOutput
    procedure ExecRange(Lo, Hi: integer);
    procedure ExecIf(Lo, EndIdx: integer; const FirstCond: string);
    function EvalFullExpr(const S: string): TJValue;
    function FindMatchingEnd(Lo: integer;
      const OpenKw, EndKw: string): integer;
  public
    constructor Create(const ATemplate: string;
      const AMessages: array of TChatMessage; AAddGen: boolean;
      const ABos, AEos: string);
    function Render: string;
  end;

constructor TJInterp.Create(const ATemplate: string;
  const AMessages: array of TChatMessage; AAddGen: boolean;
  const ABos, AEos: string);
var
  I: integer;
begin
  inherited Create;
  FTemplate := ATemplate;
  SetLength(FMessages, Length(AMessages));
  for I := 0 to High(AMessages) do FMessages[I] := AMessages[I];
  FAddGen := AAddGen;
  FBos := ABos;
  FEos := AEos;
end;

procedure TJInterp.Fail(const Msg: string);
begin
  raise EChatTemplateError.Create('chat_template: ' + Msg);
end;

function TJInterp.MakeStr(const S: string): TJValue;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.Kind := jvString;
  Result.Str := S;
end;

function TJInterp.MakeInt(I: integer): TJValue;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.Kind := jvInt;
  Result.IntV := I;
end;

function TJInterp.MakeBool(B: boolean): TJValue;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.Kind := jvBool;
  Result.BoolV := B;
end;

function TJInterp.MakeUndefined: TJValue;
begin
  FillChar(Result, SizeOf(Result), 0);
  Result.Kind := jvUndefined;
end;

function TJInterp.ToStr(const V: TJValue): string;
begin
  case V.Kind of
    jvString: Result := V.Str;
    jvInt: Result := IntToStr(V.IntV);
    jvBool: if V.BoolV then Result := 'True' else Result := 'False';
    jvUndefined: Result := '';
    else
      Fail('cannot stringify a list/object value');
  end;
end;

function TJInterp.Truthy(const V: TJValue): boolean;
begin
  case V.Kind of
    jvUndefined: Result := false;
    jvBool: Result := V.BoolV;
    jvString: Result := V.Str <> '';
    jvInt: Result := V.IntV <> 0;
    jvMessages: Result := (V.SliceHi - V.SliceLo) > 0;
    jvMessage: Result := true;
    else Result := true;
  end;
end;

function TJInterp.ValuesEqual(const A, B: TJValue): boolean;
begin
  // bool vs bool, int vs int, string vs string; undefined never equals a
  // concrete value except undefined==undefined.
  if (A.Kind = jvUndefined) or (B.Kind = jvUndefined) then
    Result := (A.Kind = jvUndefined) and (B.Kind = jvUndefined)
  else if (A.Kind = jvBool) or (B.Kind = jvBool) then
    Result := (A.Kind = jvBool) and (B.Kind = jvBool) and (A.BoolV = B.BoolV)
  else if (A.Kind = jvInt) and (B.Kind = jvInt) then
    Result := A.IntV = B.IntV
  else if (A.Kind = jvString) and (B.Kind = jvString) then
    Result := A.Str = B.Str
  else
    Result := false;
end;

function TJInterp.LookupVar(const Name: string; out V: TJValue): boolean;
var
  I: integer;
begin
  // user-set variables take precedence (a {% set %} can shadow specials).
  for I := High(FVars) downto 0 do
    if FVars[I].Name = Name then
    begin
      V := FVars[I].Value;
      Exit(true);
    end;
  if Name = 'messages' then
  begin
    FillChar(V, SizeOf(V), 0);
    V.Kind := jvMessages;
    V.SliceLo := 0;
    V.SliceHi := Length(FMessages);
    Exit(true);
  end
  else if Name = 'add_generation_prompt' then
  begin
    V := MakeBool(FAddGen);
    Exit(true);
  end
  else if Name = 'bos_token' then begin V := MakeStr(FBos); Exit(true); end
  else if Name = 'eos_token' then begin V := MakeStr(FEos); Exit(true); end
  else if Name = 'true' then begin V := MakeBool(true); Exit(true); end
  else if Name = 'false' then begin V := MakeBool(false); Exit(true); end
  else if Name = 'none' then begin V := MakeUndefined; Exit(true); end;
  V := MakeUndefined;
  Result := false;
end;

procedure TJInterp.SetVar(const Name: string; const V: TJValue);
var
  I: integer;
begin
  for I := 0 to High(FVars) do
    if FVars[I].Name = Name then
    begin
      FVars[I].Value := V;
      Exit;
    end;
  SetLength(FVars, Length(FVars) + 1);
  FVars[High(FVars)].Name := Name;
  FVars[High(FVars)].Value := V;
end;

// --- Tokenizer: split FTemplate into TEXT / output / statement chunks,
// applying whitespace control. trim_blocks: a block tag {% %} that is
// immediately followed by a newline swallows that newline. lstrip_blocks:
// whitespace from line start up to a block tag is stripped. {%- strips all
// whitespace before; -%} strips all whitespace after (overrides the block
// defaults). Same for {{- -}}.
procedure TJInterp.Tokenize;
var
  P, Len, TagStart, TagEnd: integer;
  T: string;
  IsStmt, TrimLeft, TrimRight: boolean;
  Inner, PrevText: string;
  K: integer;

  procedure AddChunk(Kind: char; const Txt: string);
  begin
    SetLength(FChunks, Length(FChunks) + 1);
    FChunks[High(FChunks)].Kind := Kind;
    FChunks[High(FChunks)].Text := Txt;
  end;

begin
  T := FTemplate;
  Len := Length(T);
  P := 1;
  while P <= Len do
  begin
    // find next tag opener "{{" or "{%"
    TagStart := 0;
    K := P;
    while K < Len do
    begin
      if (T[K] = '{') and ((T[K + 1] = '{') or (T[K + 1] = '%')) then
      begin
        TagStart := K;
        Break;
      end;
      Inc(K);
    end;
    if TagStart = 0 then
    begin
      // trailing text
      if P <= Len then AddChunk('T', Copy(T, P, Len - P + 1));
      Break;
    end;
    // emit preceding text chunk
    if TagStart > P then AddChunk('T', Copy(T, P, TagStart - P));
    IsStmt := T[TagStart + 1] = '%';
    // detect left trim marker "{{-" / "{%-"
    TrimLeft := (TagStart + 2 <= Len) and (T[TagStart + 2] = '-');
    // find tag end
    TagEnd := 0;
    K := TagStart + 2;
    while K < Len do
    begin
      if IsStmt and (T[K] = '%') and (T[K + 1] = '}') then
      begin TagEnd := K; Break; end;
      if (not IsStmt) and (T[K] = '}') and (T[K + 1] = '}') then
      begin TagEnd := K; Break; end;
      Inc(K);
    end;
    if TagEnd = 0 then Fail('unterminated tag');
    TrimRight := (T[TagEnd - 1] = '-');
    Inner := Copy(T, TagStart + 2, TagEnd - (TagStart + 2));
    if TrimLeft then Delete(Inner, 1, 1);
    if TrimRight then Delete(Inner, Length(Inner), 1);
    Inner := Trim(Inner);

    // apply left-side whitespace control to the previously emitted TEXT.
    if Length(FChunks) > 0 then
    begin
      if FChunks[High(FChunks)].Kind = 'T' then
      begin
        PrevText := FChunks[High(FChunks)].Text;
        if TrimLeft then
        begin
          // strip ALL trailing whitespace
          while (Length(PrevText) > 0) and
            (PrevText[Length(PrevText)] in [' ', #9, #10, #13]) do
            Delete(PrevText, Length(PrevText), 1);
        end
        else if IsStmt then
        begin
          // lstrip_blocks: strip spaces/tabs back to the last newline.
          K := Length(PrevText);
          while (K > 0) and (PrevText[K] in [' ', #9]) do Dec(K);
          if (K = 0) or (PrevText[K] = #10) then
            SetLength(PrevText, K);
        end;
        FChunks[High(FChunks)].Text := PrevText;
      end;
    end;

    if IsStmt then AddChunk('S', Inner) else AddChunk('O', Inner);

    P := TagEnd + 2;
    // right-side whitespace control on the text that FOLLOWS.
    if TrimRight then
    begin
      while (P <= Len) and (T[P] in [' ', #9, #10, #13]) do Inc(P);
    end
    else if IsStmt then
    begin
      // trim_blocks: swallow a single trailing newline (and a preceding CR).
      if (P <= Len) and (T[P] = #13) and (P + 1 <= Len) and (T[P + 1] = #10)
        then Inc(P, 2)
      else if (P <= Len) and (T[P] = #10) then Inc(P);
    end;
  end;
end;

procedure TJInterp.SkipWs(const S: string; var P: integer);
begin
  while (P <= Length(S)) and (S[P] in [' ', #9, #10, #13]) do Inc(P);
end;

function TJInterp.PeekWord(const S: string; P: integer;
  const W: string): boolean;
var
  E: integer;
begin
  // matches keyword W at P followed by a non-identifier char (word boundary)
  if Copy(S, P, Length(W)) <> W then Exit(false);
  E := P + Length(W);
  Result := (E > Length(S)) or not (S[E] in
    ['a'..'z', 'A'..'Z', '0'..'9', '_']);
end;

// --- expression evaluation (recursive descent) ---

function TJInterp.ParseExpr(const S: string; var P: integer): TJValue;
begin
  Result := ParseOr(S, P);
end;

function TJInterp.ParseOr(const S: string; var P: integer): TJValue;
var
  R: TJValue;
begin
  Result := ParseAnd(S, P);
  SkipWs(S, P);
  while PeekWord(S, P, 'or') do
  begin
    Inc(P, 2);
    R := ParseAnd(S, P);
    Result := MakeBool(Truthy(Result) or Truthy(R));
    SkipWs(S, P);
  end;
end;

function TJInterp.ParseAnd(const S: string; var P: integer): TJValue;
var
  R: TJValue;
begin
  Result := ParseNot(S, P);
  SkipWs(S, P);
  while PeekWord(S, P, 'and') do
  begin
    Inc(P, 3);
    R := ParseNot(S, P);
    Result := MakeBool(Truthy(Result) and Truthy(R));
    SkipWs(S, P);
  end;
end;

function TJInterp.ParseNot(const S: string; var P: integer): TJValue;
var
  R: TJValue;
begin
  SkipWs(S, P);
  if PeekWord(S, P, 'not') then
  begin
    Inc(P, 3);
    R := ParseNot(S, P);
    Result := MakeBool(not Truthy(R));
  end
  else
    Result := ParseCompare(S, P);
end;

function TJInterp.ParseCompare(const S: string; var P: integer): TJValue;
var
  R: TJValue;
  Op: string;
  Cmp: integer;
  Res: boolean;
begin
  Result := ParseConcat(S, P);
  SkipWs(S, P);
  // 'is defined' / 'is not defined'
  if PeekWord(S, P, 'is') then
  begin
    Inc(P, 2);
    SkipWs(S, P);
    if PeekWord(S, P, 'not') then
    begin
      Inc(P, 3); SkipWs(S, P);
      if not PeekWord(S, P, 'defined') then Fail('expected ''defined''');
      Inc(P, 7);
      Result := MakeBool(Result.Kind = jvUndefined);
    end
    else if PeekWord(S, P, 'defined') then
    begin
      Inc(P, 7);
      Result := MakeBool(Result.Kind <> jvUndefined);
    end
    else
      Fail('unsupported ''is'' test');
    Exit;
  end;
  // relational / equality operators
  Op := '';
  if Copy(S, P, 2) = '==' then Op := '=='
  else if Copy(S, P, 2) = '!=' then Op := '!='
  else if Copy(S, P, 2) = '<=' then Op := '<='
  else if Copy(S, P, 2) = '>=' then Op := '>='
  else if (P <= Length(S)) and (S[P] = '<') then Op := '<'
  else if (P <= Length(S)) and (S[P] = '>') then Op := '>';
  if Op = '' then Exit;
  Inc(P, Length(Op));
  R := ParseConcat(S, P);
  case Op of
    '==': Result := MakeBool(ValuesEqual(Result, R));
    '!=': Result := MakeBool(not ValuesEqual(Result, R));
    else
    begin
      if (Result.Kind <> jvInt) or (R.Kind <> jvInt) then
        Fail('relational comparison needs integers');
      if Result.IntV < R.IntV then Cmp := -1
      else if Result.IntV > R.IntV then Cmp := 1
      else Cmp := 0;
      case Op of
        '<': Res := Cmp < 0;
        '<=': Res := Cmp <= 0;
        '>': Res := Cmp > 0;
        '>=': Res := Cmp >= 0;
        else Res := false;
      end;
      Result := MakeBool(Res);
    end;
  end;
end;

function TJInterp.ParseConcat(const S: string; var P: integer): TJValue;
var
  R: TJValue;
begin
  Result := ParseAddSub(S, P);
  SkipWs(S, P);
  while (P <= Length(S)) and (S[P] = '~') do
  begin
    Inc(P);
    R := ParseAddSub(S, P);
    Result := MakeStr(ToStr(Result) + ToStr(R));
    SkipWs(S, P);
  end;
end;

function TJInterp.ParseAddSub(const S: string; var P: integer): TJValue;
var
  R: TJValue;
  Op: char;
begin
  Result := ParseMulMod(S, P);
  SkipWs(S, P);
  while (P <= Length(S)) and ((S[P] = '+') or (S[P] = '-')) do
  begin
    // ensure it's not part of "->" or similar; here only + and - binary
    Op := S[P];
    Inc(P);
    R := ParseMulMod(S, P);
    if (Op = '+') and ((Result.Kind = jvString) or (R.Kind = jvString)) then
      Result := MakeStr(ToStr(Result) + ToStr(R))
    else if (Result.Kind = jvInt) and (R.Kind = jvInt) then
    begin
      if Op = '+' then Result := MakeInt(Result.IntV + R.IntV)
      else Result := MakeInt(Result.IntV - R.IntV);
    end
    else
      Fail('cannot apply ''' + Op + ''' to these operands');
    SkipWs(S, P);
  end;
end;

function TJInterp.ParseMulMod(const S: string; var P: integer): TJValue;
var
  R: TJValue;
begin
  Result := ParsePostfix(S, P);
  SkipWs(S, P);
  while (P <= Length(S)) and (S[P] = '%') do
  begin
    Inc(P);
    R := ParsePostfix(S, P);
    if (Result.Kind <> jvInt) or (R.Kind <> jvInt) or (R.IntV = 0) then
      Fail('modulo needs nonzero integers');
    Result := MakeInt(Result.IntV mod R.IntV);
    SkipWs(S, P);
  end;
end;

// postfix: indexing [..], attribute .name, filter |name(args), method .m()
function TJInterp.ParsePostfix(const S: string; var P: integer): TJValue;
var
  Key, FilterName, AttrName: string;
  Idx, Start, ColonPos, Lo, Hi: integer;
  IdxVal: TJValue;
  HasLo, HasHi: boolean;
  Inner: string;
begin
  Result := ParsePrimary(S, P);
  SkipWs(S, P);
  while P <= Length(S) do
  begin
    if S[P] = '[' then
    begin
      // subscript: string key or integer index or slice a:b
      Inc(P);
      SkipWs(S, P);
      // gather inner up to matching ]
      Start := P;
      while (P <= Length(S)) and (S[P] <> ']') do Inc(P);
      if P > Length(S) then Fail('unterminated subscript');
      Inner := Trim(Copy(S, Start, P - Start));
      Inc(P); // skip ]
      ColonPos := Pos(':', Inner);
      if ColonPos > 0 then
      begin
        // slice messages[a:b] -- only valid on jvMessages
        if Result.Kind <> jvMessages then
          Fail('slice on non-list');
        HasLo := Trim(Copy(Inner, 1, ColonPos - 1)) <> '';
        HasHi := Trim(Copy(Inner, ColonPos + 1, MaxInt)) <> '';
        if HasLo then Lo := StrToInt(Trim(Copy(Inner, 1, ColonPos - 1)))
        else Lo := 0;
        if HasHi then
          Hi := StrToInt(Trim(Copy(Inner, ColonPos + 1, MaxInt)))
        else Hi := Result.SliceHi - Result.SliceLo;
        IdxVal := Result;
        Result.SliceLo := IdxVal.SliceLo + Lo;
        Result.SliceHi := IdxVal.SliceLo + Hi;
      end
      else if (Length(Inner) > 0) and
        ((Inner[1] = '''') or (Inner[1] = '"')) then
      begin
        // string key on a message
        Key := Copy(Inner, 2, Length(Inner) - 2);
        if Result.Kind <> jvMessage then Fail('string key on non-object');
        if Key = 'role' then Result := MakeStr(FMessages[Result.MsgIdx].Role)
        else if Key = 'content' then
          Result := MakeStr(FMessages[Result.MsgIdx].Content)
        else
          Fail('unknown message key ''' + Key + '''');
      end
      else
      begin
        // integer index into messages
        Idx := StrToInt(Inner);
        if Result.Kind <> jvMessages then Fail('index on non-list');
        if (Idx < 0) or (Result.SliceLo + Idx >= Result.SliceHi) then
          Fail('list index out of range');
        IdxVal := Result;
        FillChar(Result, SizeOf(Result), 0);
        Result.Kind := jvMessage;
        Result.MsgIdx := IdxVal.SliceLo + Idx;
      end;
    end
    else if (S[P] = '.') then
    begin
      Inc(P);
      Start := P;
      while (P <= Length(S)) and (S[P] in
        ['a'..'z', 'A'..'Z', '0'..'9', '_']) do Inc(P);
      AttrName := Copy(S, Start, P - Start);
      SkipWs(S, P);
      if (P <= Length(S)) and (S[P] = '(') then
      begin
        // method call: only .strip() supported
        Inc(P);
        SkipWs(S, P);
        if (P <= Length(S)) and (S[P] = ')') then Inc(P)
        else Fail('method args not supported');
        if AttrName = 'strip' then
          Result := MakeStr(PyStrip(ToStr(Result)))
        else
          Fail('unsupported method .' + AttrName + '()');
      end
      else
      begin
        // attribute: message.role / message.content / loop.xxx
        if Result.Kind = jvMessage then
        begin
          if AttrName = 'role' then
            Result := MakeStr(FMessages[Result.MsgIdx].Role)
          else if AttrName = 'content' then
            Result := MakeStr(FMessages[Result.MsgIdx].Content)
          else
            Fail('unknown message attribute .' + AttrName);
        end
        else if Result.Kind = jvLoop then
        begin
          if AttrName = 'index0' then Result := MakeInt(Result.LoopIndex0)
          else if AttrName = 'index' then
            Result := MakeInt(Result.LoopIndex0 + 1)
          else if AttrName = 'first' then
            Result := MakeBool(Result.LoopIndex0 = 0)
          else if AttrName = 'last' then
            Result := MakeBool(Result.LoopIndex0 = Result.LoopCount - 1)
          else
            Fail('unknown loop attribute .' + AttrName);
        end
        else
          Fail('attribute .' + AttrName + ' on unsupported value');
      end;
    end
    else if S[P] = '|' then
    begin
      Inc(P);
      SkipWs(S, P);
      Start := P;
      while (P <= Length(S)) and (S[P] in
        ['a'..'z', 'A'..'Z', '0'..'9', '_']) do Inc(P);
      FilterName := Copy(S, Start, P - Start);
      SkipWs(S, P);
      if FilterName = 'trim' then
        Result := MakeStr(PyStrip(ToStr(Result)))
      else if FilterName = 'default' then
      begin
        if (P > Length(S)) or (S[P] <> '(') then
          Fail('default filter needs an argument');
        Inc(P);
        IdxVal := ParseExpr(S, P);
        SkipWs(S, P);
        if (P > Length(S)) or (S[P] <> ')') then Fail('expected )');
        Inc(P);
        if Result.Kind = jvUndefined then Result := IdxVal;
      end
      else
        Fail('unsupported filter |' + FilterName);
    end
    else
      Break;
    SkipWs(S, P);
  end;
end;

function TJInterp.ParsePrimary(const S: string; var P: integer): TJValue;
var
  Start: integer;
  Ident, Lit, FnName, Quote: string;
  V: TJValue;
  Arg: TJValue;
begin
  SkipWs(S, P);
  if P > Length(S) then Fail('unexpected end of expression');
  if S[P] = '(' then
  begin
    Inc(P);
    Result := ParseExpr(S, P);
    SkipWs(S, P);
    if (P > Length(S)) or (S[P] <> ')') then Fail('expected )');
    Inc(P);
    Exit;
  end;
  if (S[P] = '''') or (S[P] = '"') then
  begin
    Quote := S[P];
    Inc(P);
    Start := P;
    while (P <= Length(S)) and (S[P] <> Quote[1]) do Inc(P);
    if P > Length(S) then Fail('unterminated string literal');
    Lit := Copy(S, Start, P - Start);
    Inc(P);
    Result := MakeStr(Lit);
    Exit;
  end;
  if S[P] in ['0'..'9'] then
  begin
    Start := P;
    while (P <= Length(S)) and (S[P] in ['0'..'9']) do Inc(P);
    Result := MakeInt(StrToInt(Copy(S, Start, P - Start)));
    Exit;
  end;
  if S[P] in ['a'..'z', 'A'..'Z', '_'] then
  begin
    Start := P;
    while (P <= Length(S)) and (S[P] in
      ['a'..'z', 'A'..'Z', '0'..'9', '_']) do Inc(P);
    Ident := Copy(S, Start, P - Start);
    SkipWs(S, P);
    // function call?
    if (P <= Length(S)) and (S[P] = '(') then
    begin
      FnName := Ident;
      Inc(P);
      SkipWs(S, P);
      if FnName = 'raise_exception' then
      begin
        Arg := ParseExpr(S, P);
        SkipWs(S, P);
        if (P > Length(S)) or (S[P] <> ')') then Fail('expected )');
        Inc(P);
        raise EChatTemplateError.Create(ToStr(Arg));
      end
      else
        Fail('unsupported function ' + FnName + '()');
    end;
    if LookupVar(Ident, V) then Result := V
    else Result := MakeUndefined;
    Exit;
  end;
  Fail('unexpected character ''' + S[P] + ''' in expression');
end;

// --- block execution ---

function TJInterp.FindMatchingEnd(Lo: integer;
  const OpenKw, EndKw: string): integer;
var
  I, Depth, P: integer;
  Kw, Body: string;
begin
  Depth := 1;
  I := Lo;
  while I < Length(FChunks) do
  begin
    if FChunks[I].Kind = 'S' then
    begin
      Body := FChunks[I].Text;
      P := 1;
      while (P <= Length(Body)) and (Body[P] in
        ['a'..'z', 'A'..'Z', '_']) do Inc(P);
      Kw := Copy(Body, 1, P - 1);
      if Kw = OpenKw then Inc(Depth)
      else if Kw = EndKw then
      begin
        Dec(Depth);
        if Depth = 0 then Exit(I);
      end;
    end;
    Inc(I);
  end;
  Fail('missing {% ' + EndKw + ' %}');
  Result := -1;
end;

function TJInterp.EvalFullExpr(const S: string): TJValue;
var
  P: integer;
begin
  P := 1;
  Result := ParseExpr(S, P);
  SkipWs(S, P);
  if P <= Length(S) then Fail('trailing characters in expression: ' +
    Copy(S, P, MaxInt));
end;

// Splits a statement chunk into its leading keyword and the remainder.
procedure StatementKeyword(const Body: string;
  out Kw, Rest: string);
var
  P: integer;
begin
  P := 1;
  while (P <= Length(Body)) and (Body[P] in
    ['a'..'z', 'A'..'Z', '_']) do Inc(P);
  Kw := Copy(Body, 1, P - 1);
  Rest := Trim(Copy(Body, P, MaxInt));
end;

// Executes an {% if %}...{% endif %} block. Lo points just past the {% if %}
// chunk; EndIdx is the {% endif %} chunk index. FirstCond is the if
// condition. Walks the elif/else clauses at nesting depth 0.
procedure TJInterp.ExecIf(Lo, EndIdx: integer; const FirstCond: string);
var
  I, Depth, ClauseStart: integer;
  Kw, Rest, CurCond: string;
  Taken: boolean;
begin
  CurCond := FirstCond;
  ClauseStart := Lo;
  Taken := false;
  I := Lo;
  Depth := 0;
  while I < EndIdx do
  begin
    if FChunks[I].Kind = 'S' then
    begin
      StatementKeyword(FChunks[I].Text, Kw, Rest);
      if (Kw = 'if') or (Kw = 'for') then Inc(Depth)
      else if (Kw = 'endif') or (Kw = 'endfor') then Dec(Depth)
      else if (Depth = 0) and ((Kw = 'elif') or (Kw = 'else')) then
      begin
        // close the current clause; evaluate it if not yet taken.
        if (not Taken) and Truthy(EvalFullExpr(CurCond)) then
        begin
          ExecRange(ClauseStart, I);
          Taken := true;
        end;
        if Kw = 'elif' then CurCond := Rest
        else CurCond := 'true'; // else clause always passes if reached
        ClauseStart := I + 1;
      end;
    end;
    Inc(I);
  end;
  if (not Taken) and Truthy(EvalFullExpr(CurCond)) then
    ExecRange(ClauseStart, EndIdx);
end;

procedure TJInterp.ExecRange(Lo, Hi: integer);
var
  I, P, EndIdx, BodyStart: integer;
  Body, Kw, Rest, VarName, IterName: string;
  IterVal, Item, LoopVal, SetVal: TJValue;
  M: integer;
begin
  I := Lo;
  while I < Hi do
  begin
    case FChunks[I].Kind of
      'T': begin FOutput := FOutput + FChunks[I].Text; Inc(I); end;
      'O':
        begin
          FOutput := FOutput + ToStr(EvalFullExpr(FChunks[I].Text));
          Inc(I);
        end;
      'S':
        begin
          StatementKeyword(FChunks[I].Text, Kw, Rest);
          if Kw = 'for' then
          begin
            EndIdx := FindMatchingEnd(I + 1, 'for', 'endfor');
            // parse "var in expr"
            P := 1;
            SkipWs(Rest, P);
            BodyStart := P;
            while (P <= Length(Rest)) and (Rest[P] in
              ['a'..'z', 'A'..'Z', '0'..'9', '_']) do Inc(P);
            IterName := Copy(Rest, BodyStart, P - BodyStart);
            SkipWs(Rest, P);
            if not PeekWord(Rest, P, 'in') then Fail('for: expected ''in''');
            Inc(P, 2);
            IterVal := ParseExpr(Rest, P);
            SkipWs(Rest, P);
            if P <= Length(Rest) then
              Fail('for: trailing characters after iterable');
            if IterVal.Kind <> jvMessages then
              Fail('for: only message lists are iterable');
            for M := IterVal.SliceLo to IterVal.SliceHi - 1 do
            begin
              FillChar(Item, SizeOf(Item), 0);
              Item.Kind := jvMessage;
              Item.MsgIdx := M;
              SetVar(IterName, Item);
              FillChar(LoopVal, SizeOf(LoopVal), 0);
              LoopVal.Kind := jvLoop;
              LoopVal.LoopIndex0 := M - IterVal.SliceLo;
              LoopVal.LoopCount := IterVal.SliceHi - IterVal.SliceLo;
              SetVar('loop', LoopVal);
              ExecRange(I + 1, EndIdx);
            end;
            I := EndIdx + 1;
          end
          else if Kw = 'if' then
          begin
            EndIdx := FindMatchingEnd(I + 1, 'if', 'endif');
            ExecIf(I + 1, EndIdx, Rest);
            I := EndIdx + 1;
          end
          else if Kw = 'set' then
          begin
            P := Pos('=', Rest);
            if P = 0 then Fail('set: expected ''=''');
            VarName := Trim(Copy(Rest, 1, P - 1));
            SetVal := EvalFullExpr(Trim(Copy(Rest, P + 1, MaxInt)));
            SetVar(VarName, SetVal);
            Inc(I);
          end
          else if (Kw = 'endfor') or (Kw = 'endif') or (Kw = 'else') or
            (Kw = 'elif') then
            Fail('unexpected {% ' + Kw + ' %}')
          else
            Fail('unsupported statement {% ' + Kw + ' %}');
        end;
    end;
  end;
end;

function TJInterp.Render: string;
begin
  Tokenize;
  FOutput := '';
  ExecRange(0, Length(FChunks));
  Result := FOutput;
end;

function RenderChatTemplate(const Template: string;
  const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean;
  const BosToken: string = ''; const EosToken: string = ''): string;
var
  Interp: TJInterp;
begin
  Interp := TJInterp.Create(Template, Messages, AddGenerationPrompt,
    BosToken, EosToken);
  try
    Result := Interp.Render;
  finally
    Interp.Free;
  end;
end;

function ApplyChatTemplate(ChatFormat: TNeuralChatFormat;
  const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean = true): string;
begin
  case ChatFormat of
    cfChatML: Result := RenderChatML(Messages, AddGenerationPrompt);
    cfLlama2: Result := RenderLlama2(Messages);
    cfLlama3: Result := RenderLlama3(Messages, AddGenerationPrompt);
    cfZephyr: Result := RenderZephyr(Messages, AddGenerationPrompt);
    cfGemma: Result := RenderGemma(Messages, AddGenerationPrompt);
    cfPhi3: Result := RenderPhi3(Messages, AddGenerationPrompt);
    cfMistral: Result := RenderMistral(Messages);
    cfDeepSeek: Result := RenderDeepSeek(Messages, AddGenerationPrompt);
    cfPhi4Mini: Result := RenderPhi4Mini(Messages, AddGenerationPrompt);
    else
      raise ENeuralChatError.Create('Unknown chat format. Pass one of ' +
        'cfChatML/cfLlama2/cfLlama3/cfZephyr/cfGemma/cfPhi3/cfMistral/' +
        'cfDeepSeek/cfPhi4Mini ' +
        '(auto-detection via DetectChatFormat did not recognize the ' +
        'model''s chat_template).');
  end;
end;

function ApplyChatTemplateString(const ChatTemplate: string;
  const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean = true;
  const BosToken: string = ''; const EosToken: string = ''): string;
var
  Detected: TNeuralChatFormat;
begin
  Detected := DetectChatFormat(ChatTemplate);
  if Detected <> cfUnknown then
    // Recognized family: the hardcoded renderer is the byte-pinned ground
    // truth, so prefer it over re-interpreting the Jinja.
    Result := ApplyChatTemplate(Detected, Messages, AddGenerationPrompt)
  else if Trim(ChatTemplate) = '' then
    raise EChatTemplateError.Create(
      'no chat_template available to render')
  else
    // Fallback: render the raw Jinja template via the mini-Jinja interpreter.
    Result := RenderChatTemplate(ChatTemplate, Messages,
      AddGenerationPrompt, BosToken, EosToken);
end;

procedure EncodeChat(Tokenizer: TNeuralHFTokenizer;
  ChatFormat: TNeuralChatFormat; const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean; Ids: TIntegerList);
begin
  Tokenizer.Encode(
    ApplyChatTemplate(ChatFormat, Messages, AddGenerationPrompt), Ids);
end;

function EncodeChat(Tokenizer: TNeuralHFTokenizer;
  ChatFormat: TNeuralChatFormat; const Messages: array of TChatMessage;
  AddGenerationPrompt: boolean = true): TNeuralIntegerArray;
begin
  Result := Tokenizer.Encode(
    ApplyChatTemplate(ChatFormat, Messages, AddGenerationPrompt));
end;

end.
