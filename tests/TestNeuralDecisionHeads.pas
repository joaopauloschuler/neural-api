unit TestNeuralDecisionHeads;
(*
Tests for neuraldecisionheads.pas, the trained-head /v1/systemone scorers.
TKevScorer runs on tests/fixtures/kev_qwen3_5/ (tools/kev_tiny_fixture.py):
a random pico Qwen3.5 hybrid base, a PEFT LoRA adapter on kev's targets and a
pointer head written through kev's own code, against kev's own scoring
(Checkpoint.load + admit + DecisionModel.probs, float64 backbone).
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, StrUtils, Math, fpcunit, testregistry, fpjson, jsonparser,
  neuralvolume, neuralchatengine, neuraldecision, neuraldecisionheads;

type
  TTestNeuralDecisionHeads = class(TTestCase)
  private
    function FixturePath(const FileName: string): string;
    function LoadOracle(): TJSONObject;
    // The fixture base with the fixture adapter merged, FP32 on the CPU.
    function NewKevEngine(): TChatEngine;
  published
    // Token ids, readout positions, probabilities and answers against kev; then
    // the temperature override, and a chat equal to a fresh engine's.
    procedure TestKevParityWithKev;
    procedure TestKevHeadRefusesUnsupportedOptions;
    // A state over 65536 tokens and a row over the context answer 422.
    procedure TestKevLimits;
  end;

implementation

const
  // FP32 forward of the merged pico hybrid against kev's float64 backbone and
  // fp32 head; the largest measured gap is 4.1e-6 (at T = 1).
  csKevParityTolerance = 2e-5;
  csKevContext = 1024;

procedure AssertIdsEqual(const Msg: string; Expected: TJSONArray;
  const Actual: TNeuralIntegerArray);
var
  IdPos: integer;
begin
  for IdPos := 0 to Min(Expected.Count, Length(Actual)) - 1 do
    if Expected.Integers[IdPos] <> Actual[IdPos] then
      TAssert.Fail(Format('%s: first difference at token %d: expected %d, got %d',
        [Msg, IdPos, Expected.Integers[IdPos], Actual[IdPos]]));
  TAssert.AssertEquals(Msg + ': token count', Expected.Count, Length(Actual));
end;

function TTestNeuralDecisionHeads.FixturePath(const FileName: string): string;
begin
  Result := 'fixtures' + DirectorySeparator + FileName;
  if not FileExists(Result) and not DirectoryExists(Result) then
    Result := 'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      FileName;
  if not FileExists(Result) and not DirectoryExists(Result) then
    Fail('Fixture not found: ' + FileName);
end;

function TTestNeuralDecisionHeads.LoadOracle(): TJSONObject;
var
  Stream: TFileStream;
begin
  Stream := TFileStream.Create(FixturePath('kev_qwen3_5' + DirectorySeparator +
    'kev_oracle.json'), fmOpenRead or fmShareDenyWrite);
  try
    Result := GetJSON(Stream) as TJSONObject;
  finally
    Stream.Free;
  end;
end;

function TTestNeuralDecisionHeads.NewKevEngine(): TChatEngine;
var
  Args: TStringList;
  Opt: TChatOptions;
  ErrorMsg: string;
begin
  ErrorMsg := '';
  Result := TChatEngine.Create();
  Args := TStringList.Create();
  try
    try
      Args.AddStrings([FixturePath('kev_qwen3_5' + DirectorySeparator + 'base'),
        '--greedy', '--fp32', '--cpu', '--serial', '--ctx', IntToStr(csKevContext),
        '--max-new-tokens', '3']);
      AssertTrue('chat options parse', ParseArgs(Args, Opt));
      Opt.LoRAAdapterDir := FixturePath('kev_qwen3_5' + DirectorySeparator + 'adapter');
      AssertTrue('LoadModel: ' + ErrorMsg, Result.LoadModel(Opt, ErrorMsg));
    except
      Result.Free;
      raise;
    end;
  finally
    Args.Free;
  end;
end;

procedure TTestNeuralDecisionHeads.TestKevParityWithKev;
var
  Oracle, QuestionOracle, AnswerOracle: TJSONObject;
  Engine, Fresh: TChatEngine;
  Scorer: TKevScorer;
  Request: TDecisionRequest;
  Errors: TDecisionErrorDetails;
  Answers, RawAnswers: TDecisionAnswers;
  InputTokens, OutputTokens, QuestionPos, OptionPos: integer;
  Where: string;
  Expected: TJSONArray;
  ChatPrompt: TNeuralIntegerArray;
  Reply, FreshReply: string;
  Answer: TDecisionAnswer;
  TemperatureEffect: double;
begin
  Oracle := LoadOracle();
  Engine := nil;
  Fresh := nil;
  Scorer := nil;
  try
    Engine := NewKevEngine();
    Scorer := TKevScorer.Create(Engine, 'kev-parity', FixturePath('kev_qwen3_5' +
      DirectorySeparator + 'adapter' + DirectorySeparator + 'head.pt'));
    AssertEquals('head temperature', Oracle.Objects['settings'].Floats['temperature'],
      Scorer.HeadTemperature, 1e-12);
    AssertEquals('pointer dim', Oracle.Objects['settings'].Integers['head_dim'],
      Scorer.PointerDim);
    AssertTrue('parse', ParseDecisionRequest(Oracle.Strings['request'], Request, Errors));
    AssertEquals('rendered state', Oracle.Strings['state_text'], Request.State);
    Scorer.Score(Request, Answers, InputTokens, OutputTokens);
    AssertIdsEqual('state ids', Oracle.Arrays['state_ids'], Scorer.LastStateIds);
    AssertEquals('question count', Oracle.Arrays['questions'].Count, Length(Answers));
    for QuestionPos := 0 to High(Answers) do
    begin
      QuestionOracle := Oracle.Arrays['questions'].Objects[QuestionPos];
      Where := QuestionOracle.Strings['id'];
      AssertEquals(Where + ': id', Where, Request.Questions[QuestionPos].Id);
      Expected := QuestionOracle.Arrays['options'];
      AssertEquals(Where + ': options', Expected.Count,
        Length(KevOptionTexts(Request.Questions[QuestionPos])));
      for OptionPos := 0 to Expected.Count - 1 do
        AssertEquals(Where + ': option text', Expected.Strings[OptionPos],
          KevOptionTexts(Request.Questions[QuestionPos])[OptionPos]);
      AssertIdsEqual(Where + ': branch ids', QuestionOracle.Arrays['branch_ids'],
        Scorer.LastRows[QuestionPos].BranchIds);
      AssertIdsEqual(Where + ': </opt> positions', QuestionOracle.Arrays['opt_positions'],
        Scorer.LastRows[QuestionPos].OptionPositions);
      AssertEquals(Where + ': decide position', QuestionOracle.Integers['decide_position'],
        Scorer.LastRows[QuestionPos].DecidePosition);
      Expected := QuestionOracle.Arrays['probs'];
      AssertEquals(Where + ': probabilities', Expected.Count,
        Length(Answers[QuestionPos].Probabilities));
      for OptionPos := 0 to Expected.Count - 1 do
        AssertEquals(Where + ': p[' + IntToStr(OptionPos) + ']',
          Expected.Floats[OptionPos], Answers[QuestionPos].Probabilities[OptionPos],
          csKevParityTolerance);
      // kev's answers are rounded to 4 decimals.
      AnswerOracle := Oracle.Objects['answers'].Objects[Where];
      Answer := Answers[QuestionPos];
      case Answer.Kind of
        dqkChoice:
          begin
            AssertEquals(Where + ': choice', AnswerOracle.Strings['choice'],
              Request.Questions[QuestionPos].OptionNames[Answer.ChoiceIndex]);
            AssertEquals(Where + ': confidence', AnswerOracle.Floats['confidence'],
              Answer.Confidence, 1e-4);
          end;
        dqkScore:
          begin
            AssertEquals(Where + ': score', AnswerOracle.Floats['score'], Answer.Score, 1e-4);
            AssertEquals(Where + ': confidence', AnswerOracle.Floats['confidence'],
              Answer.Confidence, 1e-4);
          end;
        dqkNoul:
          AssertEquals(Where + ': noul', AnswerOracle.Floats['noul'], Answer.Noul, 1e-4);
      end;
    end;
    AssertEquals('input tokens', Oracle.Integers['input_tokens'], InputTokens);
    AssertEquals('nothing generated', 0, OutputTokens);

    // KEV_TEMPERATURE=1: raw pointer logits.
    Scorer.Temperature := 1.0;
    Scorer.Score(Request, RawAnswers, InputTokens, OutputTokens);
    TemperatureEffect := 0;
    for QuestionPos := 0 to High(RawAnswers) do
    begin
      QuestionOracle := Oracle.Arrays['questions'].Objects[QuestionPos];
      Expected := QuestionOracle.Arrays['probs_temperature_1'];
      for OptionPos := 0 to Expected.Count - 1 do
      begin
        AssertEquals(QuestionOracle.Strings['id'] + ': T=1 p[' + IntToStr(OptionPos) + ']',
          Expected.Floats[OptionPos], RawAnswers[QuestionPos].Probabilities[OptionPos],
          csKevParityTolerance);
        TemperatureEffect := Max(TemperatureEffect, Abs(
          RawAnswers[QuestionPos].Probabilities[OptionPos] -
          Answers[QuestionPos].Probabilities[OptionPos]));
      end;
    end;
    AssertTrue('the head temperature matters', TemperatureEffect > 0.05);

    // The engine still chats like a fresh one after scoring.
    ChatPrompt := Copy(Scorer.LastStateIds, 0, 12);
    Reply := Engine.GenerateFromIds(ChatPrompt, Engine.Opt);
    Fresh := NewKevEngine();
    FreshReply := Fresh.GenerateFromIds(ChatPrompt, Fresh.Opt);
    AssertEquals('chat after scoring: reply', FreshReply, Reply);
    AssertEquals('chat after scoring: completion', Fresh.LastCompletionTokens,
      Engine.LastCompletionTokens);
    AssertTrue('chat after scoring: last logits row',
      Engine.Session.Output().SumDiff(Fresh.Session.Output()) < 1e-5);
  finally
    Fresh.Free;
    Scorer.Free;
    Engine.Free;
    Oracle.Free;
  end;
end;

procedure TTestNeuralDecisionHeads.TestKevHeadRefusesUnsupportedOptions;
var
  Engine: TChatEngine;

  procedure AssertRefused(const Flag: string);
  var
    Message: string;
    Scorer: TKevScorer;
  begin
    Message := '';
    Scorer := nil;
    try
      Scorer := TKevScorer.Create(Engine, 'kev', FixturePath('kev_qwen3_5' +
        DirectorySeparator + 'head_' + Flag + '.pt'));
    except
      on E: EDecisionHeadError do Message := E.Message;
    end;
    Scorer.Free;
    AssertTrue(Flag + ' refused, naming it: ' + Message, Pos(Flag + ' is true', Message) > 0);
  end;

begin
  Engine := NewKevEngine();
  try
    AssertRefused('option_isolation');
    AssertRefused('special_embeddings');
  finally
    Engine.Free;
  end;
end;

procedure TTestNeuralDecisionHeads.TestKevLimits;
var
  Engine: TChatEngine;
  Scorer: TKevScorer;
  Request: TDecisionRequest;
  Errors: TDecisionErrorDetails;
  Answers: TDecisionAnswers;
  InputTokens, OutputTokens: integer;

  function ScoreError(const State: string): string;
  begin
    Result := '';
    AssertTrue('parse', ParseDecisionRequest('{"model":"kev","state":"' + State +
      '","questions":{"q":{"type":"noul","instructions":"Refund due?"}}}',
      Request, Errors));
    try
      Scorer.Score(Request, Answers, InputTokens, OutputTokens);
    except
      on E: EDecisionUnsupported do Result := E.Message;
    end;
  end;

begin
  Engine := NewKevEngine();
  Scorer := nil;
  try
    Scorer := TKevScorer.Create(Engine, 'kev', FixturePath('kev_qwen3_5' +
      DirectorySeparator + 'adapter' + DirectorySeparator + 'head.pt'));
    AssertTrue('state limit', Pos('over the 65536-token limit',
      ScoreError(DupeString('z ', 66000))) > 0);
    AssertTrue('context limit', Pos('exceed the 1024-token context',
      ScoreError(DupeString('z ', 1100))) > 0);
    AssertEquals('nothing was fed', 0, Length(Engine.CachedTokens));
    AssertEquals('a short state still scores', '', ScoreError('z z'));
    AssertEquals('one answer', 1, Length(Answers));
  finally
    Scorer.Free;
    Engine.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralDecisionHeads);
end.
