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

  TNeuralChatFormat = (
    cfUnknown,
    cfChatML,   // Qwen / Yi / ChatML family
    cfLlama2,   // Llama-2-chat [INST] + <<SYS>>
    cfLlama3,   // Llama-3-Instruct header format
    cfZephyr,   // zephyr-7b-beta / TinyLlama-1.1B-Chat
    cfGemma,    // gemma-it <start_of_turn>
    cfPhi3,     // Phi-3-mini-instruct <|user|>...<|end|>
    cfMistral   // Mistral-7B-Instruct [INST] without system
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
// absent one, the first entry.
function LoadChatTemplateString(const TokenizerConfigFile: string): string;

// LoadChatTemplateString + DetectChatFormat in one call.
function DetectChatFormatFromConfigFile(
  const TokenizerConfigFile: string): TNeuralChatFormat;

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
  else if (Pos('<|end|>', ChatTemplate) > 0) and
    (Pos('<|user|>', ChatTemplate) > 0) then Result := cfPhi3
  else if Pos('<|user|>', ChatTemplate) > 0 then Result := cfZephyr
  else if Pos('[INST]', ChatTemplate) > 0 then Result := cfMistral
  else Result := cfUnknown;
end;

function LoadChatTemplateString(const TokenizerConfigFile: string): string;
var
  FS: TFileStream;
  RawJson: string;
  Root, Node, Entry: TJSONData;
  Arr: TJSONArray;
  Cnt: integer;
begin
  Result := '';
  FS := TFileStream.Create(TokenizerConfigFile,
    fmOpenRead or fmShareDenyWrite);
  try
    SetLength(RawJson, FS.Size);
    if FS.Size > 0 then FS.ReadBuffer(RawJson[1], FS.Size);
  finally
    FS.Free;
  end;
  // same fpjson stance as neuralhftokenizer: decode \uXXXX up front and
  // parse raw so non-ASCII survives without a widestring manager.
  Root := HFParseJSONRaw(HFDecodeUnicodeEscapes(RawJson));
  try
    if not (Root is TJSONObject) then
      raise ENeuralChatError.Create(TokenizerConfigFile +
        ': root is not an object');
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
    else
      raise ENeuralChatError.Create('Unknown chat format. Pass one of ' +
        'cfChatML/cfLlama2/cfLlama3/cfZephyr/cfGemma/cfPhi3/cfMistral ' +
        '(auto-detection via DetectChatFormat did not recognize the ' +
        'model''s chat_template).');
  end;
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
