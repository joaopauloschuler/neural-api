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

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralvolume, neuralchatengine, neuraldecision,
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

  TKevScorer = class(TDecisionScorer)
  private
    FEngine: TChatEngine;
    FQueryWeights, FQueryBias, FKeyWeights, FKeyBias: TNNetVolume;
    FHiddenSize, FPointerDim: integer;
    FHeadTemperature, FTemperature: double;
    FStateId, FQuestionId, FOptionOpenId, FOptionCloseId, FDecideId: integer;
    FLastStateIds: TNeuralIntegerArray;
    FLastRows: TKevQuestionRows;
    procedure SetTemperature(Value: double);
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
    // head.pt's temperature; 1.0 = raw logits (kev's KEV_TEMPERATURE).
    property Temperature: double read FTemperature write SetTemperature;
    property HeadTemperature: double read FHeadTemperature;
    property PointerDim: integer read FPointerDim;
    property LastStateIds: TNeuralIntegerArray read FLastStateIds;
    property LastRows: TKevQuestionRows read FLastRows;
  end;

// kev.api.to_record option texts: noul ("no[: false]", "yes[: true]"),
// choice "name[: desc]", score the level texts.
function KevOptionTexts(const Question: TDecisionQuestion): TStringArray;

implementation

uses
  Math;

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
  inherited Create(AModelName);
  FEngine := AEngine;
  FQueryWeights := TNNetVolume.Create();
  FQueryBias := TNNetVolume.Create();
  FKeyWeights := TNNetVolume.Create();
  FKeyBias := TNNetVolume.Create();
  FStateId := SpecialId(csKevStateToken);
  FQuestionId := SpecialId(csKevQuestionToken);
  FOptionOpenId := SpecialId(csKevOptionOpenToken);
  FOptionCloseId := SpecialId(csKevOptionCloseToken);
  FDecideId := SpecialId(csKevDecideToken);
  FHiddenSize := FEngine.Session.HiddenState().Depth;
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
    FHeadTemperature := Reader.GetMetadataFloat('temperature', 1.0);
    if not (FHeadTemperature > 0) or IsInfinite(FHeadTemperature) then
      raise EDecisionHeadError.CreateFmt('%s: temperature %g', [AHeadFile,
        FHeadTemperature]);
    FTemperature := FHeadTemperature;
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

procedure TKevScorer.SetTemperature(Value: double);
begin
  if not (Value > 0) or IsInfinite(Value) then
    raise EDecisionHeadError.CreateFmt('Kev temperature must be positive and' +
      ' finite, not %g', [Value]);
  FTemperature := Value;
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
  Scale, Largest, Total, QueryElem: double;
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
  Largest := MaxValue(Result);
  Total := 0;
  for OptionPos := 0 to MaxOptionPos do
  begin
    Result[OptionPos] := Exp(Result[OptionPos] - Largest);
    Total := Total + Result[OptionPos];
  end;
  for OptionPos := 0 to MaxOptionPos do
    Result[OptionPos] := Result[OptionPos] / Total;
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

end.
