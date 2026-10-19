unit neuraldecisionheads;

(*
neuraldecisionheads
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

// neuraldecisionheads -- /v1/systemone scorers that read a trained head
// (head.pt) over the final hidden states of a TChatEngine whose checkpoint
// was loaded with the scorer's LoRA adapter merged (TChatOptions.
// LoRAAdapterDir). Nothing is generated and the LM head is never computed.
//
//   - TKevScorer: jaredpalmer/kev (kev/model.py encode + PointerHead, kev/api.py
//     to_record). One row per request: [<|fim_prefix|>] + state, then per
//     question a branch [<|fim_middle|>] + instructions + ([<|box_start|>]
//     option [<|box_end|>])* + [<|fim_suffix|>] continuing after the state.
//     The state is fed once; each branch is scored from its saved state.
//     z_i = (K h_</opt>_i + k.bias) . (Q h_decide + q.bias) / sqrt(dp) / T.
//   - TOpenJevScorer: Zefan-Cai/Open-Jev (jev/api.py compile_request +
//     candidate_prompts, jev/model.py forward). One prompt per candidate (noul:
//     one), each a single user turn under the Qwen3.5 chat frame with thinking
//     off; s = w . h_last + b, choice/score softmax(s / T), noul sigmoid(s / T).
//     The prompts' longest common token prefix is fed once.

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralvolume, neuralchat, neuralchatengine, neuraldecision,
  neuraltorchbin;

const
  // kev.model.SPECIAL: state, question, option open, option close, decide.
  csKevStateToken = '<|fim_prefix|>';
  csKevQuestionToken = '<|fim_middle|>';
  csKevOptionOpenToken = '<|box_start|>';
  csKevOptionCloseToken = '<|box_end|>';
  csKevDecideToken = '<|fim_suffix|>';
  // kev.model.SERVE_MAX_STATE (the state token included) and SERVE_MAX_BRANCH
  // (state + one branch).
  csKevMaxStateTokens = 65536;
  csKevMaxRowTokens = csKevMaxStateTokens + 8192;
  // An Open-Jev checkpoint directory (package/checkpoint in the HF release).
  csOpenJevAdapterSubdir = 'adapter';
  csOpenJevHeadFile = 'head.pt';
  csOpenJevModelFile = 'model.json';
  csOpenJevTemperatureFile = 'temperature.json';

type
  // A head.pt this scorer cannot use (missing tensor, wrong shape, a Meta
  // option it does not implement).
  EDecisionHeadError = class(Exception);

  // One question as the last request fed it: the branch ids and, within them,
  // the </opt> index per option and the decide index.
  TKevQuestionRow = record
    BranchIds: TNeuralIntegerArray;
    OptionPositions: TNeuralIntegerArray;
    DecidePosition: integer;
  end;
  TKevQuestionRows = array of TKevQuestionRow;

  // A scorer reading a trained head over the final hidden states of FEngine,
  // with a calibration temperature from the checkpoint that callers may override.
  TDecisionHeadScorer = class(TDecisionScorer)
  protected
    FEngine: TChatEngine;
    FHiddenSize: integer;
    FHeadTemperature, FTemperature: double;
    procedure SetTemperature(Value: double);
    // FHeadTemperature := Value, and the override back to it; Source names the file.
    procedure SetHeadTemperature(Value: double; const Source: string);
  public
    // AEngine is loaded (adapter merged) and not owned.
    constructor Create(AEngine: TChatEngine; const AModelName: string);
    // The checkpoint's temperature; 1.0 = raw logits.
    property Temperature: double read FTemperature write SetTemperature;
    property HeadTemperature: double read FHeadTemperature;
  end;

  TKevScorer = class(TDecisionHeadScorer)
  private
    FQueryWeights, FQueryBias, FKeyWeights, FKeyBias: TNNetVolume;
    FPointerDim: integer;
    FStateId, FQuestionId, FOptionOpenId, FOptionCloseId, FDecideId: integer;
    FLastStateIds: TNeuralIntegerArray;
    FLastRows: TKevQuestionRows;
  public
    // AEngine is loaded (adapter merged) and not owned; AHeadFile is kev's head.pt.
    constructor Create(AEngine: TChatEngine; const AModelName, AHeadFile: string);
    destructor Destroy(); override;
    procedure Score(const Request: TDecisionRequest; out Answers: TDecisionAnswers;
      out InputTokens, OutputTokens: integer); override;
    // [<|fim_prefix|>] + the state's ids (special-token strings rewritten).
    function BuildStateIds(const StateText: string): TNeuralIntegerArray;
    function BuildQuestionRow(const Question: TDecisionQuestion): TKevQuestionRow;
    // Softmax of the pointer logits of one branch's final hidden states:
    // rows 0..K-1 = the </opt> tokens, row K = the decide token.
    function PointerProbabilities(Hidden: TNNetVolume): TDecisionProbabilities;
    property PointerDim: integer read FPointerDim;
    property LastStateIds: TNeuralIntegerArray read FLastStateIds;
    property LastRows: TKevQuestionRows read FLastRows;
  end;

  // One Open-Jev candidate prompt as the last request fed it.
  TOpenJevPrompt = record
    QuestionPos: integer;
    Text: string;                // the chat-framed prompt
    Ids: TNeuralIntegerArray;
  end;
  TOpenJevPrompts = array of TOpenJevPrompt;

  TOpenJevScorer = class(TDecisionHeadScorer)
  private
    FHeadWeights: TNNetVolume;
    FHeadBias: double;
    FMaxPromptTokens, FLoRARank: integer;
    FBaseModelId, FBaseRevision: string;
    FSharePrefix: boolean;
    FLastPrompts: TOpenJevPrompts;
    FLastScores: TDecisionProbabilities;
    FLastSharedPrefixLen: integer;
    procedure SetMaxPromptTokens(Value: integer);
  public
    // ACheckpointDir: anything OpenJevCheckpointDir resolves. AEngine was
    // loaded with <checkpoint>/adapter merged when model.json's lora_rank > 0.
    constructor Create(AEngine: TChatEngine; const AModelName, ACheckpointDir: string);
    destructor Destroy(); override;
    procedure Score(const Request: TDecisionRequest; out Answers: TDecisionAnswers;
      out InputTokens, OutputTokens: integer); override;
    // CandidatePrompt (special-token strings rewritten) as one user turn, thinking off.
    function RenderPrompt(const CandidatePrompt: string): string;
    // w . h + b over row 0 of Hidden, in double.
    function HeadScore(Hidden: TNNetVolume): double;
    // model.json max_length: a longer prompt answers 422 (jev's --max-length overrides it).
    property MaxPromptTokens: integer read FMaxPromptTokens write SetMaxPromptTokens;
    property LoRARank: integer read FLoRARank;
    property BaseModelId: string read FBaseModelId;
    property BaseRevision: string read FBaseRevision;
    // Feed the prompts' common token prefix once (default true).
    property SharePrefix: boolean read FSharePrefix write FSharePrefix;
    property LastPrompts: TOpenJevPrompts read FLastPrompts;
    // The head score s of every LastPrompts entry.
    property LastScores: TDecisionProbabilities read FLastScores;
    property LastSharedPrefixLen: integer read FLastSharedPrefixLen;
  end;

  // model.json of an Open-Jev checkpoint.
  TOpenJevModelConfig = record
    MaxPromptTokens, LoRARank: integer;
    BaseModelId, BaseRevision: string;
  end;

  TDecisionAdapterLayout = (dalUnknown, dalKevLoRA, dalKevFullWeights, dalOpenJev);

  // A trained decision checkpoint as found on disk, read without its base model.
  TDecisionAdapterInfo = record
    Layout: TDecisionAdapterLayout;
    CheckpointDir: string;              // holds head.pt
    LoRAAdapterDir: string;             // the PEFT adapter to merge; '' = none
    BaseModelId, BaseRevision: string;  // the base it records; '' = not recorded
    HeadInputSize: integer;             // the hidden size the head reads
  end;

// kev.api.to_record option texts: noul ("no[: false]", "yes[: true]"),
// choice "name[: desc]", score the level texts.
function KevOptionTexts(const Question: TDecisionQuestion): TStringArray;
// The directory holding head.pt and temperature.json: Dir itself, Dir/checkpoint
// or Dir/package/checkpoint (an Open-Jev HF repo clone); '' when none does.
function OpenJevCheckpointDir(const Dir: string): string;
// jev.api.compile_request + candidate_prompts for one question: choice/score one
// prompt per option, noul one. Caller text is NOT rewritten here.
function OpenJevCandidatePrompts(const Request: TDecisionRequest;
  const Question: TDecisionQuestion): TStringArray;
// Raises EDecisionHeadError when model.json is missing or has no positive max_length.
function ReadOpenJevModelConfig(const CheckpointDir: string): TOpenJevModelConfig;
// Kev LoRA: head.pt, adapter_config.json and adapter weights in Dir; Kev full weights:
// head.pt, config.json and model*.safetensors; Open-Jev: OpenJevCheckpointDir(Dir) <> ''.
function DetectDecisionAdapterLayout(const Dir: string): TDecisionAdapterLayout;
// Reads head.pt (and model.json); raises EDecisionHeadError on dalUnknown, naming the layouts.
function ReadDecisionAdapterInfo(const Dir: string): TDecisionAdapterInfo;
// Dir's own name, or the repository root's for <root>/checkpoint and <root>/package/checkpoint.
function DecisionAdapterModelName(const Dir: string): string;

implementation

uses
  Math, fpjson, jsonparser;

// Dest := the flat tensor pName after checking it is [Rows, Cols] (Cols = 0: 1-D [Rows]).
procedure LoadDecisionHeadTensor(Reader: TNNetTorchBinReader;
  const HeadFile, pName: string; Rows, Cols: integer; Dest: TNNetVolume);
var
  Expected: string;
  ShapeOK: boolean;
begin
  if not Reader.HasTensor(pName) then
    raise EDecisionHeadError.CreateFmt('%s: no tensor "%s"', [HeadFile, pName]);
  if Cols = 0 then
  begin
    Expected := Format('[%d]', [Rows]);
    ShapeOK := (Reader.DimCount(pName) = 1) and (Reader.DimSize(pName, 0) = Rows);
  end
  else
  begin
    Expected := Format('[%d, %d]', [Rows, Cols]);
    ShapeOK := (Reader.DimCount(pName) = 2) and (Reader.DimSize(pName, 0) = Rows) and
      (Reader.DimSize(pName, 1) = Cols);
  end;
  if not ShapeOK then
    raise EDecisionHeadError.CreateFmt('%s: "%s" is %s, expected %s',
      [HeadFile, pName, Reader.ShapeAsString(pName), Expected]);
  Reader.LoadTensorFlat(pName, Dest);
end;

function KevOptionTexts(const Question: TDecisionQuestion): TStringArray;
var
  OptionPos, MaxOptionPos: integer;
begin
  Result := nil;
  SetLength(Result, Length(Question.OptionNames));
  MaxOptionPos := High(Result);
  for OptionPos := 0 to MaxOptionPos do
    case Question.Kind of
      dqkNoul:
        if OptionPos = 0 then
          Result[OptionPos] := DecisionOptionText('no', Question.OptionDescriptions[0])
        else
          Result[OptionPos] := DecisionOptionText('yes', Question.OptionDescriptions[1]);
      dqkChoice:
        Result[OptionPos] := DecisionOptionText(Question.OptionNames[OptionPos],
          Question.OptionDescriptions[OptionPos]);
      else
        Result[OptionPos] := Question.OptionDescriptions[OptionPos];
    end;
end;

procedure SoftMaxInPlace(var Values: TDecisionProbabilities);
var
  ValuePos, MaxValuePos: integer;
  Largest, Total: double;
begin
  MaxValuePos := High(Values);
  Largest := MaxValue(Values);
  Total := 0;
  for ValuePos := 0 to MaxValuePos do
  begin
    Values[ValuePos] := Exp(Values[ValuePos] - Largest);
    Total := Total + Values[ValuePos];
  end;
  for ValuePos := 0 to MaxValuePos do
    Values[ValuePos] := Values[ValuePos] / Total;
end;

{ TDecisionHeadScorer }

constructor TDecisionHeadScorer.Create(AEngine: TChatEngine;
  const AModelName: string);
begin
  inherited Create(AModelName);
  FEngine := AEngine;
  FHiddenSize := FEngine.Session.HiddenState().Depth;
  FHeadTemperature := 1.0;
  FTemperature := 1.0;
end;

procedure TDecisionHeadScorer.SetTemperature(Value: double);
begin
  if not (Value > 0) or IsInfinite(Value) then
    raise EDecisionHeadError.CreateFmt('the decision temperature must be' +
      ' positive and finite, not %g', [Value]);
  FTemperature := Value;
end;

procedure TDecisionHeadScorer.SetHeadTemperature(Value: double;
  const Source: string);
begin
  if not (Value > 0) or IsInfinite(Value) then
    raise EDecisionHeadError.CreateFmt('%s: temperature %g', [Source, Value]);
  FHeadTemperature := Value;
  FTemperature := Value;
end;

{ TKevScorer }

constructor TKevScorer.Create(AEngine: TChatEngine;
  const AModelName, AHeadFile: string);
var
  Reader: TNNetTorchBinReader;

  function SpecialId(const Token: string): integer;
  begin
    Result := FEngine.Tokenizer.TokenToId(Token);
    if Result < 0 then
      raise EDecisionHeadError.CreateFmt('the tokenizer of %s has no %s token,' +
        ' which Kev rows need', [FEngine.Opt.ModelDir, Token]);
  end;

begin
  inherited Create(AEngine, AModelName);
  FQueryWeights := TNNetVolume.Create();
  FQueryBias := TNNetVolume.Create();
  FKeyWeights := TNNetVolume.Create();
  FKeyBias := TNNetVolume.Create();
  FStateId := SpecialId(csKevStateToken);
  FQuestionId := SpecialId(csKevQuestionToken);
  FOptionOpenId := SpecialId(csKevOptionOpenToken);
  FOptionCloseId := SpecialId(csKevOptionCloseToken);
  FDecideId := SpecialId(csKevDecideToken);
  Reader := TNNetTorchBinReader.Create(AHeadFile);
  try
    // Both need what a row-per-question pass cannot give: a packed attention
    // mask between option spans, and trained embedding rows in the adapter.
    if Reader.GetMetadataBool('option_isolation', false) then
      raise EDecisionHeadError.CreateFmt('%s: option_isolation is true; Kev''s' +
        ' option isolation needs its packed block mask, which is not supported' +
        ' (kev refuses it on the hybrid Qwen3.5 bases too)', [AHeadFile]);
    if Reader.GetMetadataBool('special_embeddings', false) then
      raise EDecisionHeadError.CreateFmt('%s: special_embeddings is true; trained' +
        ' delimiter embeddings (PEFT trainable_token_indices) are not supported',
        [AHeadFile]);
    FPointerDim := Reader.GetMetadataInt('head_dim', 256);
    if FPointerDim <= 0 then
      raise EDecisionHeadError.CreateFmt('%s: head_dim %d', [AHeadFile, FPointerDim]);
    LoadDecisionHeadTensor(Reader, AHeadFile, 'head.q.weight', FPointerDim,
      FHiddenSize, FQueryWeights);
    LoadDecisionHeadTensor(Reader, AHeadFile, 'head.q.bias', FPointerDim, 0, FQueryBias);
    LoadDecisionHeadTensor(Reader, AHeadFile, 'head.k.weight', FPointerDim,
      FHiddenSize, FKeyWeights);
    LoadDecisionHeadTensor(Reader, AHeadFile, 'head.k.bias', FPointerDim, 0, FKeyBias);
    SetHeadTemperature(Reader.GetMetadataFloat('temperature', 1.0), AHeadFile);
  finally
    Reader.Free;
  end;
end;

destructor TKevScorer.Destroy();
begin
  FKeyBias.Free;
  FKeyWeights.Free;
  FQueryBias.Free;
  FQueryWeights.Free;
  inherited Destroy();
end;

function TKevScorer.BuildStateIds(const StateText: string): TNeuralIntegerArray;
var
  TextIds: TNeuralIntegerArray;
begin
  TextIds := FEngine.Tokenizer.Encode(RewriteSpecialTokenStrings(StateText));
  Result := nil;
  SetLength(Result, Length(TextIds) + 1);
  Result[0] := FStateId;
  if Length(TextIds) > 0 then
    Move(TextIds[0], Result[1], Length(TextIds) * csIntegerSize);
end;

function TKevScorer.BuildQuestionRow(const Question: TDecisionQuestion): TKevQuestionRow;
var
  Options: TStringArray;
  OptionPos, MaxOptionPos, RowLen: integer;

  procedure Append(const Ids: TNeuralIntegerArray);
  begin
    if Length(Ids) = 0 then exit;
    if RowLen + Length(Ids) > Length(Result.BranchIds) then
      SetLength(Result.BranchIds, 2 * (RowLen + Length(Ids)));
    Move(Ids[0], Result.BranchIds[RowLen], Length(Ids) * csIntegerSize);
    Inc(RowLen, Length(Ids));
  end;

  procedure AppendId(Id: integer);
  begin
    if RowLen = Length(Result.BranchIds) then
      SetLength(Result.BranchIds, 2 * RowLen + 16);
    Result.BranchIds[RowLen] := Id;
    Inc(RowLen);
  end;

  function TextIds(const Text: string): TNeuralIntegerArray;
  begin
    Result := FEngine.Tokenizer.Encode(RewriteSpecialTokenStrings(Text));
  end;

begin
  Options := KevOptionTexts(Question);
  Result.BranchIds := nil;
  Result.OptionPositions := nil;
  SetLength(Result.OptionPositions, Length(Options));
  RowLen := 0;
  AppendId(FQuestionId);
  Append(TextIds(Question.Instructions));
  MaxOptionPos := High(Options);
  for OptionPos := 0 to MaxOptionPos do
  begin
    AppendId(FOptionOpenId);
    Append(TextIds(Options[OptionPos]));
    Result.OptionPositions[OptionPos] := RowLen;
    AppendId(FOptionCloseId);
  end;
  Result.DecidePosition := RowLen;
  AppendId(FDecideId);
  SetLength(Result.BranchIds, RowLen);
end;

function TKevScorer.PointerProbabilities(Hidden: TNNetVolume): TDecisionProbabilities;
var
  OptionCount, OptionPos, MaxOptionPos, ElemPos, MaxElemPos: integer;
  Keys, Query: TNNetVolume;
  QueryOffset, KeyOffset: integer;
  Scale, Total, QueryElem: double;
begin
  OptionCount := Hidden.SizeX - 1;
  MaxOptionPos := OptionCount - 1;
  MaxElemPos := FPointerDim - 1;
  Result := nil;
  SetLength(Result, OptionCount);
  Keys := TNNetVolume.Create(Hidden.SizeX, 1, FPointerDim);
  Query := TNNetVolume.Create(Hidden.SizeX, 1, FPointerDim);
  try
    // Q and K are fp32 Linear layers in kev; the logits are summed in double.
    Keys.DotProducts(FPointerDim, 0, MaxOptionPos, FHiddenSize, FKeyWeights, Hidden);
    Query.DotProducts(FPointerDim, OptionCount, OptionCount, FHiddenSize,
      FQueryWeights, Hidden);
    QueryOffset := OptionCount * FPointerDim;
    Scale := 1 / (Sqrt(FPointerDim) * FTemperature);
    for OptionPos := 0 to MaxOptionPos do
    begin
      KeyOffset := OptionPos * FPointerDim;
      Total := 0;
      for ElemPos := 0 to MaxElemPos do
      begin
        QueryElem := Query.FData[QueryOffset + ElemPos] + FQueryBias.FData[ElemPos];
        Total := Total + (Keys.FData[KeyOffset + ElemPos] +
          FKeyBias.FData[ElemPos]) * QueryElem;
      end;
      Result[OptionPos] := Total * Scale;
    end;
  finally
    Query.Free;
    Keys.Free;
  end;
  SoftMaxInPlace(Result);
end;

procedure TKevScorer.Score(const Request: TDecisionRequest;
  out Answers: TDecisionAnswers; out InputTokens, OutputTokens: integer);
var
  QuestionCount, QuestionPos, MaxQuestionPos, OptionCount, StateLen, BranchLen: integer;
  HiddenPositions: TNeuralIntegerArray;
  Hidden: TNNetVolume;
  Question: TDecisionQuestion;
begin
  InputTokens := 0;
  OutputTokens := 0;
  Answers := nil;
  QuestionCount := Length(Request.Questions);
  MaxQuestionPos := QuestionCount - 1;
  FLastStateIds := BuildStateIds(Request.State);
  StateLen := Length(FLastStateIds);
  if StateLen > csKevMaxStateTokens then
    raise EDecisionUnsupported.CreateFmt('state is %d tokens, over the %d-token' +
      ' limit (the <state> token included): shorten the document or split it' +
      ' across requests', [StateLen, csKevMaxStateTokens]);
  FLastRows := nil;
  SetLength(FLastRows, QuestionCount);
  // Every row is built and checked before the model reads anything.
  for QuestionPos := 0 to MaxQuestionPos do
  begin
    Question := Request.Questions[QuestionPos];
    if (Question.Kind = dqkChoice) and
      (Length(Question.OptionNames) > csDecisionMaxChoiceOptions) then
      raise EDecisionUnsupported.CreateFmt('question "%s": %d options, over the' +
        ' %d Kev reads', [Question.Id, Length(Question.OptionNames),
        csDecisionMaxChoiceOptions]);
    FLastRows[QuestionPos] := BuildQuestionRow(Question);
    BranchLen := Length(FLastRows[QuestionPos].BranchIds);
    if BranchLen > csKevMaxRowTokens - StateLen then
      raise EDecisionUnsupported.CreateFmt('question "%s": branch too long: %d' +
        ' tokens with a %d-token state (row limit %d)',
        [Question.Id, BranchLen, StateLen, csKevMaxRowTokens]);
    if StateLen + BranchLen > FEngine.SeqLen then
      raise EDecisionUnsupported.CreateFmt('question "%s": the %d-token state' +
        ' and its %d-token branch exceed the %d-token context',
        [Question.Id, StateLen, BranchLen, FEngine.SeqLen]);
  end;
  SetLength(Answers, QuestionCount);
  FEngine.PrefillDecisionPrefix(FLastStateIds);
  InputTokens := StateLen;
  Hidden := TNNetVolume.Create();
  try
    for QuestionPos := 0 to MaxQuestionPos do
    begin
      OptionCount := Length(FLastRows[QuestionPos].OptionPositions);
      HiddenPositions := Copy(FLastRows[QuestionPos].OptionPositions);
      SetLength(HiddenPositions, OptionCount + 1);
      HiddenPositions[OptionCount] := FLastRows[QuestionPos].DecidePosition;
      FEngine.ScoreDecisionBranch(FLastRows[QuestionPos].BranchIds,
        HiddenPositions, nil, Hidden);
      InputTokens := InputTokens + Length(FLastRows[QuestionPos].BranchIds);
      Answers[QuestionPos] := DecisionAnswerFromProbabilities(
        Request.Questions[QuestionPos].Kind, PointerProbabilities(Hidden));
    end;
  finally
    Hidden.Free;
  end;
end;

function OpenJevCheckpointDir(const Dir: string): string;
const
  Subdirs: array[0..2] of string = ('', 'checkpoint',
    'package' + DirectorySeparator + 'checkpoint');
var
  SubdirPos: integer;
  Candidate: string;
begin
  for SubdirPos := 0 to High(Subdirs) do
  begin
    Candidate := IncludeTrailingPathDelimiter(IncludeTrailingPathDelimiter(Dir) +
      Subdirs[SubdirPos]);
    if FileExists(Candidate + csOpenJevHeadFile) and
      FileExists(Candidate + csOpenJevTemperatureFile) then
      Exit(Candidate);
  end;
  Result := '';
end;

// The option text jev.api.compile_request builds: choice "name" or "name: desc", score the level.
function OpenJevOptionText(const Question: TDecisionQuestion; OptionPos: integer): string;
begin
  if Question.Kind = dqkScore then
    Result := Question.OpenJevDescriptions[OptionPos]
  else if Question.OptionHasDescription[OptionPos] then
    Result := Question.OptionNames[OptionPos] + ': ' +
      Question.OpenJevDescriptions[OptionPos]
  else
    Result := Question.OptionNames[OptionPos];
end;

function OpenJevCandidatePrompts(const Request: TDecisionRequest;
  const Question: TDecisionQuestion): TStringArray;
var
  Prefix, QuestionText: string;
  OptionPos, MaxOptionPos: integer;
begin
  Result := nil;
  QuestionText := Question.OpenJevInstructions;
  if Question.Kind = dqkNoul then
  begin
    // compile_request appends the true description first.
    if Question.OptionHasDescription[1] then
      QuestionText := QuestionText + #10'Yes means: ' + Question.OpenJevDescriptions[1];
    if Question.OptionHasDescription[0] then
      QuestionText := QuestionText + #10'No means: ' + Question.OpenJevDescriptions[0];
  end;
  Prefix := 'Context:'#10 + Request.OpenJevState + #10#10'Question: ' +
    QuestionText + #10;
  if Question.Kind = dqkNoul then
  begin
    SetLength(Result, 1);
    Result[0] := Prefix + 'Is the answer to this question yes? Answer Yes or No.';
    exit;
  end;
  SetLength(Result, Length(Question.OptionNames));
  MaxOptionPos := High(Result);
  for OptionPos := 0 to MaxOptionPos do
    Result[OptionPos] := Prefix + 'Proposed answer: ' +
      OpenJevOptionText(Question, OptionPos) +
      #10'Is this proposed answer correct? Answer Yes or No.';
end;

// The JSON object in FileName; raises EDecisionHeadError when it is missing or not an object.
function ReadDecisionJSONObject(const FileName: string): TJSONObject;
var
  Text: TStringList;
  Root: TJSONData;
begin
  if not FileExists(FileName) then
    raise EDecisionHeadError.CreateFmt('%s: file not found', [FileName]);
  Text := TStringList.Create;
  try
    Text.LoadFromFile(FileName);
    try
      Root := GetJSON(Text.Text);
    except
      on E: Exception do
        raise EDecisionHeadError.CreateFmt('%s: %s', [FileName, E.Message]);
    end;
  finally
    Text.Free;
  end;
  if not (Root is TJSONObject) then
  begin
    Root.Free;
    raise EDecisionHeadError.CreateFmt('%s: not a JSON object', [FileName]);
  end;
  Result := TJSONObject(Root);
end;

function ReadOpenJevModelConfig(const CheckpointDir: string): TOpenJevModelConfig;
var
  FileName: string;
  Config: TJSONObject;
  Field: TJSONData;

  function ConfigText(const Name: string): string;
  var
    TextField: TJSONData;
  begin
    TextField := Config.Find(Name);
    if (TextField <> nil) and (TextField.JSONType = jtString) then
      Result := TextField.AsString
    else
      Result := '';
  end;

begin
  FileName := IncludeTrailingPathDelimiter(CheckpointDir) + csOpenJevModelFile;
  Config := ReadDecisionJSONObject(FileName);
  try
    Field := Config.Find('max_length');
    if (Field = nil) or not (Field is TJSONIntegerNumber) or (Field.AsInteger <= 0) then
      raise EDecisionHeadError.CreateFmt('%s: max_length must be a positive' +
        ' integer', [FileName]);
    Result.MaxPromptTokens := Field.AsInteger;
    Result.LoRARank := 0;
    Field := Config.Find('lora_rank');
    if (Field <> nil) and (Field is TJSONIntegerNumber) then
      Result.LoRARank := Field.AsInteger;
    Result.BaseModelId := ConfigText('model_id');
    Result.BaseRevision := ConfigText('revision');
  finally
    Config.Free;
  end;
end;

{ TOpenJevScorer }

constructor TOpenJevScorer.Create(AEngine: TChatEngine;
  const AModelName, ACheckpointDir: string);
var
  CheckpointDir, HeadFile: string;
  ModelConfig: TOpenJevModelConfig;
  Config: TJSONObject;
  Field: TJSONData;
  Reader: TNNetTorchBinReader;
  BiasVolume: TNNetVolume;
begin
  inherited Create(AEngine, AModelName);
  FSharePrefix := true;
  FHeadWeights := TNNetVolume.Create();
  CheckpointDir := OpenJevCheckpointDir(ACheckpointDir);
  if CheckpointDir = '' then
    raise EDecisionHeadError.CreateFmt('%s: no Open-Jev checkpoint (%s and %s)' +
      ' in it, in checkpoint/ or in package/checkpoint/', [ACheckpointDir,
      csOpenJevHeadFile, csOpenJevTemperatureFile]);
  if (FEngine.Tokenizer.TokenToId('<|im_start|>') < 0) or
    (FEngine.Tokenizer.TokenToId('<|im_end|>') < 0) then
    raise EDecisionHeadError.CreateFmt('the tokenizer of %s has no <|im_start|>' +
      ' or <|im_end|> token, which Open-Jev''s chat prompts need',
      [FEngine.Opt.ModelDir]);
  ModelConfig := ReadOpenJevModelConfig(CheckpointDir);
  FMaxPromptTokens := ModelConfig.MaxPromptTokens;
  FLoRARank := ModelConfig.LoRARank;
  FBaseModelId := ModelConfig.BaseModelId;
  FBaseRevision := ModelConfig.BaseRevision;
  Config := ReadDecisionJSONObject(CheckpointDir + csOpenJevTemperatureFile);
  try
    Field := Config.Find('temperature');
    if (Field = nil) or (Field.JSONType <> jtNumber) then
      raise EDecisionHeadError.CreateFmt('%s: no numeric "temperature"',
        [CheckpointDir + csOpenJevTemperatureFile]);
    SetHeadTemperature(Field.AsFloat, CheckpointDir + csOpenJevTemperatureFile);
  finally
    Config.Free;
  end;
  HeadFile := CheckpointDir + csOpenJevHeadFile;
  BiasVolume := TNNetVolume.Create();
  Reader := TNNetTorchBinReader.Create(HeadFile);
  try
    LoadDecisionHeadTensor(Reader, HeadFile, 'weight', 1, FHiddenSize, FHeadWeights);
    LoadDecisionHeadTensor(Reader, HeadFile, 'bias', 1, 0, BiasVolume);
    FHeadBias := BiasVolume.FData[0];
  finally
    Reader.Free;
    BiasVolume.Free;
  end;
end;

destructor TOpenJevScorer.Destroy();
begin
  FHeadWeights.Free;
  inherited Destroy();
end;

procedure TOpenJevScorer.SetMaxPromptTokens(Value: integer);
begin
  if Value <= 0 then
    raise EDecisionHeadError.CreateFmt('Open-Jev max_length must be positive,' +
      ' not %d', [Value]);
  FMaxPromptTokens := Value;
end;

function TOpenJevScorer.RenderPrompt(const CandidatePrompt: string): string;
var
  Msgs: array[0..0] of TChatMessage;
begin
  // The Qwen3.5-2B/9B and Qwen3.8-27B templates render one user turn alike
  // with thinking off; the 2B template detects as plain ChatML, so the frame is fixed.
  Msgs[0] := ChatMessage('user', RewriteSpecialTokenStrings(CandidatePrompt));
  Result := ApplyChatTemplate(cfQwen3_5, Msgs, ChatTemplateOptions(
    {AddGenerationPrompt=}true, {ContinueFinalMessage=}false, reOff));
end;

function TOpenJevScorer.HeadScore(Hidden: TNNetVolume): double;
var
  ElemPos, MaxElemPos: integer;
begin
  Result := FHeadBias;
  MaxElemPos := FHiddenSize - 1;
  for ElemPos := 0 to MaxElemPos do
    Result := Result + FHeadWeights.FData[ElemPos] * Hidden.FData[ElemPos];
end;

procedure TOpenJevScorer.Score(const Request: TDecisionRequest;
  out Answers: TDecisionAnswers; out InputTokens, OutputTokens: integer);
var
  QuestionPos, MaxQuestionPos, CandidatePos, MaxCandidatePos: integer;
  PromptCount, PromptPos, MaxPromptPos, PromptLen, PrefixLen, OptionCount: integer;
  Candidates: TStringArray;
  Question: TDecisionQuestion;
  HiddenPositions: TNeuralIntegerArray;
  Hidden: TNNetVolume;
  P: TDecisionProbabilities;
begin
  InputTokens := 0;
  OutputTokens := 0;
  Answers := nil;
  FLastPrompts := nil;
  FLastScores := nil;
  FLastSharedPrefixLen := 0;
  MaxQuestionPos := High(Request.Questions);
  PromptCount := 0;
  // Every prompt is built and checked before the model reads anything.
  for QuestionPos := 0 to MaxQuestionPos do
  begin
    Question := Request.Questions[QuestionPos];
    Candidates := OpenJevCandidatePrompts(Request, Question);
    SetLength(FLastPrompts, PromptCount + Length(Candidates));
    MaxCandidatePos := High(Candidates);
    for CandidatePos := 0 to MaxCandidatePos do
    begin
      FLastPrompts[PromptCount].QuestionPos := QuestionPos;
      FLastPrompts[PromptCount].Text := RenderPrompt(Candidates[CandidatePos]);
      FLastPrompts[PromptCount].Ids := FEngine.Tokenizer.Encode(
        FLastPrompts[PromptCount].Text);
      PromptLen := Length(FLastPrompts[PromptCount].Ids);
      if PromptLen > FMaxPromptTokens then
        raise EDecisionUnsupported.CreateFmt('question "%s": a %d-token prompt' +
          ' exceeds the checkpoint''s max_length of %d tokens',
          [Question.Id, PromptLen, FMaxPromptTokens]);
      if PromptLen > FEngine.SeqLen then
        raise EDecisionUnsupported.CreateFmt('question "%s": a %d-token prompt' +
          ' exceeds the %d-token context', [Question.Id, PromptLen, FEngine.SeqLen]);
      Inc(PromptCount);
    end;
  end;
  if PromptCount = 0 then exit;
  MaxPromptPos := PromptCount - 1;
  // jev's common_prefix_length: at least one token of every prompt is left to feed.
  PrefixLen := 0;
  if FSharePrefix then
  begin
    PrefixLen := Length(FLastPrompts[0].Ids) - 1;
    for PromptPos := 1 to MaxPromptPos do
      PrefixLen := Min(PrefixLen, Min(Length(FLastPrompts[PromptPos].Ids) - 1,
        CommonPrefixLen(FLastPrompts[0].Ids, FLastPrompts[PromptPos].Ids)));
  end;
  FLastSharedPrefixLen := PrefixLen;
  SetLength(FLastScores, PromptCount);
  FEngine.PrefillDecisionPrefix(Copy(FLastPrompts[0].Ids, 0, PrefixLen));
  InputTokens := PrefixLen;
  HiddenPositions := nil;
  SetLength(HiddenPositions, 1);
  Hidden := TNNetVolume.Create();
  try
    for PromptPos := 0 to MaxPromptPos do
    begin
      PromptLen := Length(FLastPrompts[PromptPos].Ids);
      HiddenPositions[0] := PromptLen - PrefixLen - 1;
      FEngine.ScoreDecisionBranch(Copy(FLastPrompts[PromptPos].Ids, PrefixLen,
        MaxInt), HiddenPositions, nil, Hidden);
      FLastScores[PromptPos] := HeadScore(Hidden);
      InputTokens := InputTokens + PromptLen - PrefixLen;
    end;
  finally
    Hidden.Free;
  end;
  SetLength(Answers, Length(Request.Questions));
  PromptPos := 0;
  for QuestionPos := 0 to MaxQuestionPos do
  begin
    Question := Request.Questions[QuestionPos];
    if Question.Kind = dqkNoul then
    begin
      // jev reads noul as the softmax of [0, s]: p(true) = sigmoid(s / T).
      P := nil;
      SetLength(P, 2);
      P[0] := 0;
      P[1] := FLastScores[PromptPos] / FTemperature;
      Inc(PromptPos);
    end
    else
    begin
      OptionCount := Length(Question.OptionNames);
      P := Copy(FLastScores, PromptPos, OptionCount);
      MaxCandidatePos := OptionCount - 1;
      for CandidatePos := 0 to MaxCandidatePos do
        P[CandidatePos] := P[CandidatePos] / FTemperature;
      Inc(PromptPos, OptionCount);
    end;
    SoftMaxInPlace(P);
    Answers[QuestionPos] := DecisionAnswerFromProbabilities(Question.Kind, P);
  end;
end;

function DetectDecisionAdapterLayout(const Dir: string): TDecisionAdapterLayout;
var
  DirPath: string;
  Shard: TSearchRec;
  HasShards: boolean;
begin
  if OpenJevCheckpointDir(Dir) <> '' then Exit(dalOpenJev);
  DirPath := IncludeTrailingPathDelimiter(Dir);
  Result := dalUnknown;
  if not FileExists(DirPath + csOpenJevHeadFile) then exit;
  if FileExists(DirPath + 'adapter_config.json') then
  begin
    if FileExists(DirPath + 'adapter_model.safetensors') or
      FileExists(DirPath + 'adapter_model.bin') then
      Result := dalKevLoRA;
    exit;
  end;
  HasShards := FindFirst(DirPath + 'model*.safetensors', faAnyFile, Shard) = 0;
  FindClose(Shard);
  if HasShards and FileExists(DirPath + 'config.json') then
    Result := dalKevFullWeights;
end;

function ReadDecisionAdapterInfo(const Dir: string): TDecisionAdapterInfo;
var
  HeadFile, InputTensor: string;
  Reader: TNNetTorchBinReader;
  ModelConfig: TOpenJevModelConfig;

  function HeadMetadataText(const Name: string): string;
  var
    MetadataPos: integer;
  begin
    MetadataPos := Reader.FindMetadata(Name);
    if (MetadataPos >= 0) and (Reader.MetadataAt(MetadataPos).Kind = tmkStr) then
      Result := Reader.MetadataAt(MetadataPos).StrVal
    else
      Result := '';
  end;

begin
  Result.Layout := DetectDecisionAdapterLayout(Dir);
  Result.LoRAAdapterDir := '';
  Result.BaseModelId := '';
  Result.BaseRevision := '';
  Result.HeadInputSize := 0;
  case Result.Layout of
    dalUnknown:
      raise EDecisionHeadError.CreateFmt('%s is not a decision checkpoint: expected' +
        ' a Kev LoRA checkpoint (head.pt, adapter_config.json and' +
        ' adapter_model.safetensors), a Kev full-weight checkpoint (head.pt,' +
        ' config.json and model*.safetensors) or an Open-Jev checkpoint (%s and' +
        ' %s in it, in checkpoint/ or in package/checkpoint/)',
        [Dir, csOpenJevHeadFile, csOpenJevTemperatureFile]);
    dalOpenJev:
      begin
        Result.CheckpointDir := OpenJevCheckpointDir(Dir);
        ModelConfig := ReadOpenJevModelConfig(Result.CheckpointDir);
        if ModelConfig.LoRARank > 0 then
          Result.LoRAAdapterDir := Result.CheckpointDir + csOpenJevAdapterSubdir;
        Result.BaseModelId := ModelConfig.BaseModelId;
        Result.BaseRevision := ModelConfig.BaseRevision;
        InputTensor := 'weight';
      end;
    else
      begin
        Result.CheckpointDir := IncludeTrailingPathDelimiter(Dir);
        if Result.Layout = dalKevLoRA then
          Result.LoRAAdapterDir := ExcludeTrailingPathDelimiter(Dir);
        InputTensor := 'head.q.weight';
      end;
  end;
  HeadFile := Result.CheckpointDir + csOpenJevHeadFile;
  Reader := TNNetTorchBinReader.Create(HeadFile);
  try
    if Result.Layout <> dalOpenJev then
    begin
      Result.BaseModelId := HeadMetadataText('base');
      Result.BaseRevision := HeadMetadataText('base_revision');
    end;
    if Reader.HasTensor(InputTensor) and (Reader.DimCount(InputTensor) = 2) then
      Result.HeadInputSize := Reader.DimSize(InputTensor, 1);
  finally
    Reader.Free;
  end;
end;

function DecisionAdapterModelName(const Dir: string): string;
var
  Path: string;
begin
  Path := ExcludeTrailingPathDelimiter(ExpandFileName(Dir));
  if ExtractFileName(Path) = 'checkpoint' then
  begin
    Path := ExtractFileDir(Path);
    if ExtractFileName(Path) = 'package' then Path := ExtractFileDir(Path);
  end;
  Result := ExtractFileName(Path);
end;

end.
