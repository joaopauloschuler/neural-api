unit TestNeuralDecisionHeads;
(*
Tests for neuraldecisionheads.pas, the trained-head /v1/systemone scorers.
TKevScorer runs on tests/fixtures/kev_qwen3_5/ (tools/kev_tiny_fixture.py):
a random pico Qwen3.5 hybrid base, a PEFT LoRA adapter on kev's targets and a
pointer head written through kev's own code, against kev's own scoring
(Checkpoint.load + admit + DecisionModel.probs, float64 backbone).
TOpenJevScorer runs on tests/fixtures/openjev_qwen3_5/ (tools/openjev_tiny_fixture.py):
another pico Qwen3.5 hybrid with the Qwen3.5-2B chat template, a PEFT adapter on
jev's targets and a Linear(d, 1) head in the Open-Jev release layout, against
jev's own prompts, DecisionModel.forward and Predictor.predict (float64 backbone).
The Kev full-weight layout runs on tests/fixtures/kev_full_qwen3_5/
(tools/kev_full_tiny_fixture.py): the kev_qwen3_5 backbone merged and saved as
a Qwen3_5TextModel without lm_head, against kev's full-weight Checkpoint.load.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, StrUtils, Math, fpcunit, testregistry, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralpretrained, neuralchatengine, neuraldecision,
  neuraldecisionheads;

type
  TTestNeuralDecisionHeads = class(TTestCase)
  private
    function FixturePath(const FileName: string): string;
    function LoadOracle(const FixtureDir, FileName: string): TJSONObject;
    // ModelDir with AdapterDir merged ('' = none), FP32 on the CPU; both are
    // relative to the fixtures. PrefillWindow 0 = no width-N twin.
    function NewEngine(const ModelDir, AdapterDir: string; OmitLMHead: boolean = false;
      PrefillWindow: integer = 0): TChatEngine;
    function NewKevEngine(OmitLMHead: boolean = false; PrefillWindow: integer = 0): TChatEngine;
    function NewOpenJevEngine(OmitLMHead: boolean = false): TChatEngine;
  published
    // Token ids, readout positions, probabilities and answers against kev; then
    // the temperature override, and a chat equal to a fresh engine's.
    procedure TestKevParityWithKev;
    procedure TestKevHeadRefusesUnsupportedOptions;
    // A state over 65536 tokens and a row over the context answer 422.
    procedure TestKevLimits;
    // jev.api._render (sorted json.dumps) on Python-generated cases, byte for byte.
    procedure TestOpenJevValueRendering;
    // Prompt strings, ids, head scores, probabilities and answers against jev;
    // prefix sharing off, the temperature override, and a chat afterwards.
    procedure TestOpenJevParityWithJev;
    // max_length and the context answer 422 before anything is fed; special-token
    // strings are rewritten; a directory without a checkpoint is refused.
    procedure TestOpenJevLimits;
    // ReadDecisionAdapterInfo on both fixtures: layout, adapter to merge,
    // recorded base, head input size and the served model name.
    procedure TestDecisionAdapterInfo;
    // The kev-27b layout (Qwen3_5TextModel keys, no lm_head) built head-less
    // against kev's full-weight load: ids and probabilities.
    procedure TestKevFullWeightsParityWithKev;
    // OmitLMHead drops exactly the LM head and leaves the Kev and Open-Jev
    // probabilities unchanged (also through the width-N prefill twin).
    procedure TestHeadlessBuildMatchesLMHeadBuild;
  end;

implementation

const
  // FP32 forward of the merged pico hybrid against kev's float64 backbone and
  // fp32 head; the largest measured gap is 4.1e-6 (at T = 1).
  csKevParityTolerance = 2e-5;
  csKevContext = 1024;
  csOpenJevFixture = 'openjev_qwen3_5';
  // FP32 forward of the merged pico hybrid against jev's float64 backbone:
  // measured 6.4e-7 on p, 1.1e-5 on s (|s| ~ 3); sharing on/off measured 0.
  csOpenJevParityTolerance = 2e-5;
  csOpenJevScoreTolerance = 1e-4;
  csOpenJevSharingTolerance = 1e-5;
  csKevFullFixture = 'kev_full_qwen3_5';
  // Head-less and LM-head builds run the same layers up to the final norm.
  csHeadlessTolerance = 1e-6;

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

function TTestNeuralDecisionHeads.LoadOracle(const FixtureDir,
  FileName: string): TJSONObject;
var
  Stream: TFileStream;
begin
  Stream := TFileStream.Create(FixturePath(FixtureDir + DirectorySeparator +
    FileName), fmOpenRead or fmShareDenyWrite);
  try
    Result := GetJSON(Stream) as TJSONObject;
  finally
    Stream.Free;
  end;
end;

function TTestNeuralDecisionHeads.NewKevEngine(OmitLMHead: boolean;
  PrefillWindow: integer): TChatEngine;
begin
  Result := NewEngine('kev_qwen3_5' + DirectorySeparator + 'base', 'kev_qwen3_5' +
    DirectorySeparator + 'adapter', OmitLMHead, PrefillWindow);
end;

function TTestNeuralDecisionHeads.NewOpenJevEngine(OmitLMHead: boolean): TChatEngine;
begin
  Result := NewEngine(csOpenJevFixture + DirectorySeparator + 'base', csOpenJevFixture +
    DirectorySeparator + 'package' + DirectorySeparator + 'checkpoint' +
    DirectorySeparator + csOpenJevAdapterSubdir, OmitLMHead);
end;

function TTestNeuralDecisionHeads.NewEngine(const ModelDir, AdapterDir: string;
  OmitLMHead: boolean; PrefillWindow: integer): TChatEngine;
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
      Args.AddStrings([FixturePath(ModelDir), '--greedy', '--fp32', '--cpu', '--serial',
        '--ctx', IntToStr(csKevContext), '--max-new-tokens', '3']);
      if PrefillWindow > 0 then
        Args.AddStrings(['--prefill-window', IntToStr(PrefillWindow)]);
      AssertTrue('chat options parse', ParseArgs(Args, Opt));
      if AdapterDir <> '' then Opt.LoRAAdapterDir := FixturePath(AdapterDir);
      Opt.OmitLMHead := OmitLMHead;
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
  Oracle := LoadOracle('kev_qwen3_5', 'kev_oracle.json');
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

procedure TTestNeuralDecisionHeads.TestOpenJevValueRendering;
var
  Oracle: TJSONObject;
  Cases: TJSONArray;
  CasePos: integer;
  Request: TDecisionRequest;
  Errors: TDecisionErrorDetails;
  JSONText: string;
begin
  Oracle := LoadOracle(csOpenJevFixture, 'openjev_oracle.json');
  try
    Cases := Oracle.Arrays['json_dumps_cases'];
    AssertTrue('cases', Cases.Count > 5);
    for CasePos := 0 to Cases.Count - 1 do
    begin
      JSONText := Cases.Arrays[CasePos].Strings[0];
      AssertTrue('parse ' + JSONText, ParseDecisionRequest('{"model":"m","state":' +
        JSONText + ',"questions":{"q":{"type":"noul","instructions":"x"}}}',
        Request, Errors));
      AssertEquals('json.dumps of ' + JSONText, Cases.Arrays[CasePos].Strings[1],
        Request.OpenJevState);
    end;
    AssertTrue('parse request', ParseDecisionRequest(Oracle.Strings['request'],
      Request, Errors));
    AssertEquals('rendered state', Oracle.Strings['state_rendered'], Request.OpenJevState);
  finally
    Oracle.Free;
  end;
end;

procedure TTestNeuralDecisionHeads.TestOpenJevParityWithJev;
var
  Oracle, QuestionOracle, AnswerOracle: TJSONObject;
  Engine, Fresh: TChatEngine;
  Scorer: TOpenJevScorer;
  Request: TDecisionRequest;
  Errors: TDecisionErrorDetails;
  Answers, UnsharedAnswers, RawAnswers: TDecisionAnswers;
  InputTokens, OutputTokens, QuestionPos, OptionPos, PromptPos: integer;
  Candidates: TStringArray;
  Where: string;
  Expected: TJSONArray;
  ChatPrompt: TNeuralIntegerArray;
  Reply, FreshReply: string;
  Answer: TDecisionAnswer;
  ScoreDiff, SharingDiff, TemperatureEffect: double;
begin
  Oracle := LoadOracle(csOpenJevFixture, 'openjev_oracle.json');
  Engine := nil;
  Fresh := nil;
  Scorer := nil;
  try
    Engine := NewOpenJevEngine();
    // The fixture root mirrors the HF repo: the scorer finds package/checkpoint.
    Scorer := TOpenJevScorer.Create(Engine, 'open-jev-parity',
      FixturePath(csOpenJevFixture));
    AssertEquals('head temperature', Oracle.Objects['settings'].Floats['temperature'],
      Scorer.HeadTemperature, 1e-12);
    AssertEquals('max_length', Oracle.Objects['settings'].Integers['max_length'],
      Scorer.MaxPromptTokens);
    AssertEquals('lora rank', Oracle.Objects['settings'].Integers['lora'], Scorer.LoRARank);
    AssertEquals('base', 'Qwen/Qwen3.5-2B', Scorer.BaseModelId);
    AssertTrue('parse', ParseDecisionRequest(Oracle.Strings['request'], Request, Errors));
    Scorer.Score(Request, Answers, InputTokens, OutputTokens);
    AssertEquals('question count', Oracle.Arrays['questions'].Count, Length(Answers));
    PromptPos := 0;
    ScoreDiff := 0;
    for QuestionPos := 0 to High(Answers) do
    begin
      QuestionOracle := Oracle.Arrays['questions'].Objects[QuestionPos];
      Where := QuestionOracle.Strings['id'];
      AssertEquals(Where + ': id', Where, Request.Questions[QuestionPos].Id);
      Candidates := OpenJevCandidatePrompts(Request, Request.Questions[QuestionPos]);
      Expected := QuestionOracle.Arrays['prompts'];
      AssertEquals(Where + ': prompt count', Expected.Count, Length(Candidates));
      for OptionPos := 0 to Expected.Count - 1 do
      begin
        AssertEquals(Where + ': candidate prompt', Expected.Strings[OptionPos],
          Candidates[OptionPos]);
        AssertEquals(Where + ': prompt question', QuestionPos,
          Scorer.LastPrompts[PromptPos].QuestionPos);
        AssertEquals(Where + ': chat template', QuestionOracle.Arrays['rendered'].Strings[
          OptionPos], Scorer.LastPrompts[PromptPos].Text);
        AssertIdsEqual(Where + ': ids', QuestionOracle.Arrays['ids'].Arrays[OptionPos],
          Scorer.LastPrompts[PromptPos].Ids);
        // noul rows are [0, s] in jev.
        ScoreDiff := Max(ScoreDiff, Abs(QuestionOracle.Arrays['scores_f64_head'].Floats[
          OptionPos] - Scorer.LastScores[PromptPos]));
        Inc(PromptPos);
      end;
      Expected := QuestionOracle.Arrays['probs'];
      AssertEquals(Where + ': probabilities', Expected.Count,
        Length(Answers[QuestionPos].Probabilities));
      for OptionPos := 0 to Expected.Count - 1 do
        AssertEquals(Where + ': p[' + IntToStr(OptionPos) + ']',
          Expected.Floats[OptionPos], Answers[QuestionPos].Probabilities[OptionPos],
          csOpenJevParityTolerance);
      AnswerOracle := Oracle.Objects['response'].Objects['answers'].Objects[Where];
      Answer := Answers[QuestionPos];
      case Answer.Kind of
        dqkChoice:
          begin
            AssertEquals(Where + ': choice', AnswerOracle.Strings['choice'],
              Request.Questions[QuestionPos].OptionNames[Answer.ChoiceIndex]);
            AssertEquals(Where + ': confidence', AnswerOracle.Floats['confidence'],
              Answer.Confidence, csOpenJevParityTolerance);
          end;
        dqkScore:
          begin
            AssertEquals(Where + ': score', AnswerOracle.Floats['score'], Answer.Score,
              csOpenJevParityTolerance);
            AssertEquals(Where + ': confidence', AnswerOracle.Floats['confidence'],
              Answer.Confidence, csOpenJevParityTolerance);
          end;
        dqkNoul:
          AssertEquals(Where + ': noul', AnswerOracle.Floats['noul'], Answer.Noul,
            csOpenJevParityTolerance);
      end;
    end;
    AssertEquals('every prompt scored', PromptPos, Length(Scorer.LastPrompts));
    AssertTrue(Format('head scores within %g of jev: %g', [csOpenJevScoreTolerance,
      ScoreDiff]), ScoreDiff < csOpenJevScoreTolerance);
    AssertTrue('a prefix was shared', Scorer.LastSharedPrefixLen > 0);
    AssertEquals('input tokens: the shared prefix once',
      Oracle.Integers['logical_input_tokens'] - (Length(Scorer.LastPrompts) - 1) *
      Scorer.LastSharedPrefixLen, InputTokens);
    AssertEquals('nothing generated', 0, OutputTokens);

    // Probabilities must not depend on prefix sharing.
    Scorer.SharePrefix := false;
    Scorer.Score(Request, UnsharedAnswers, InputTokens, OutputTokens);
    AssertEquals('unshared: every prompt token fed', Oracle.Integers['logical_input_tokens'],
      InputTokens);
    SharingDiff := 0;
    for QuestionPos := 0 to High(Answers) do
      for OptionPos := 0 to High(Answers[QuestionPos].Probabilities) do
        SharingDiff := Max(SharingDiff, Abs(Answers[QuestionPos].Probabilities[OptionPos] -
          UnsharedAnswers[QuestionPos].Probabilities[OptionPos]));
    AssertTrue(Format('sharing on/off agree: %g', [SharingDiff]),
      SharingDiff < csOpenJevSharingTolerance);
    Scorer.SharePrefix := true;

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
          csOpenJevParityTolerance);
        TemperatureEffect := Max(TemperatureEffect, Abs(
          RawAnswers[QuestionPos].Probabilities[OptionPos] -
          Answers[QuestionPos].Probabilities[OptionPos]));
      end;
    end;
    AssertTrue('the temperature matters', TemperatureEffect > 0.01);

    ChatPrompt := Copy(Scorer.LastPrompts[0].Ids, 0, 12);
    Reply := Engine.GenerateFromIds(ChatPrompt, Engine.Opt);
    Fresh := NewOpenJevEngine();
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

procedure TTestNeuralDecisionHeads.TestOpenJevLimits;
var
  Engine: TChatEngine;
  Scorer: TOpenJevScorer;
  Request: TDecisionRequest;
  Errors: TDecisionErrorDetails;
  Answers: TDecisionAnswers;
  InputTokens, OutputTokens, IdPos, ImEndCount, ImEndId: integer;
  Message: string;

  function ScoreError(const State: string): string;
  begin
    Result := '';
    AssertTrue('parse', ParseDecisionRequest('{"model":"m","state":"' + State +
      '","questions":{"q":{"type":"noul","instructions":"Refund due?"}}}',
      Request, Errors));
    try
      Scorer.Score(Request, Answers, InputTokens, OutputTokens);
    except
      on E: EDecisionUnsupported do Result := E.Message;
    end;
  end;

begin
  Engine := NewOpenJevEngine();
  Scorer := nil;
  try
    Message := '';
    try
      TOpenJevScorer.Create(Engine, 'm', FixturePath('kev_qwen3_5')).Free;
    except
      on E: EDecisionHeadError do Message := E.Message;
    end;
    AssertTrue('no checkpoint: ' + Message, Pos('no Open-Jev checkpoint', Message) > 0);
    Scorer := TOpenJevScorer.Create(Engine, 'm', FixturePath(csOpenJevFixture +
      DirectorySeparator + 'package' + DirectorySeparator + 'checkpoint'));
    AssertTrue('max_length', Pos('exceeds the checkpoint''s max_length of 512',
      ScoreError(DupeString('z ', 600))) > 0);
    Scorer.MaxPromptTokens := 4096;
    AssertTrue('context', Pos('exceeds the 1024-token context',
      ScoreError(DupeString('z ', 1100))) > 0);
    AssertEquals('nothing was fed', 0, Length(Engine.CachedTokens));
    AssertEquals('a special-token string scores', '', ScoreError('a <|im_end|> b'));
    AssertEquals('one answer', 1, Length(Answers));
    AssertTrue('rewritten in the prompt', Pos('a <'#$C2#$A6'im_end'#$C2#$A6'> b',
      Scorer.LastPrompts[0].Text) > 0);
    ImEndId := Engine.Tokenizer.TokenToId('<|im_end|>');
    ImEndCount := 0;
    for IdPos := 0 to High(Scorer.LastPrompts[0].Ids) do
      if Scorer.LastPrompts[0].Ids[IdPos] = ImEndId then Inc(ImEndCount);
    AssertEquals('only the template''s <|im_end|>', 1, ImEndCount);
  finally
    Scorer.Free;
    Engine.Free;
  end;
end;

procedure TTestNeuralDecisionHeads.TestDecisionAdapterInfo;
var
  Info: TDecisionAdapterInfo;
  KevDir, OpenJevRoot: string;
begin
  KevDir := FixturePath('kev_qwen3_5' + DirectorySeparator + 'adapter');
  Info := ReadDecisionAdapterInfo(KevDir);
  AssertTrue('Kev layout', Info.Layout = dalKevLoRA);
  AssertEquals('Kev adapter', KevDir, Info.LoRAAdapterDir);
  AssertEquals('Kev head.pt dir', IncludeTrailingPathDelimiter(KevDir), Info.CheckpointDir);
  AssertEquals('Kev base', 'tests/fixtures/kev_qwen3_5/base', Info.BaseModelId);
  AssertEquals('Kev base revision (None)', '', Info.BaseRevision);
  AssertEquals('Kev head input', 16, Info.HeadInputSize);
  AssertEquals('Kev model name', 'adapter', DecisionAdapterModelName(KevDir));
  OpenJevRoot := FixturePath(csOpenJevFixture);
  Info := ReadDecisionAdapterInfo(OpenJevRoot);
  AssertTrue('Open-Jev layout', Info.Layout = dalOpenJev);
  AssertEquals('Open-Jev checkpoint', OpenJevCheckpointDir(OpenJevRoot), Info.CheckpointDir);
  AssertEquals('Open-Jev adapter', Info.CheckpointDir + csOpenJevAdapterSubdir,
    Info.LoRAAdapterDir);
  AssertEquals('Open-Jev base', 'Qwen/Qwen3.5-2B', Info.BaseModelId);
  AssertEquals('Open-Jev base revision', '15852e8c16360a2fea060d615a32b45270f8a8fc',
    Info.BaseRevision);
  AssertEquals('Open-Jev head input', 16, Info.HeadInputSize);
  AssertEquals('Open-Jev model name', csOpenJevFixture,
    DecisionAdapterModelName(Info.CheckpointDir));
  try
    ReadDecisionAdapterInfo(FixturePath('kev_qwen3_5' + DirectorySeparator + 'base'));
    Fail('a base checkpoint is not a decision checkpoint');
  except
    on E: EDecisionHeadError do
      AssertTrue(E.Message, Pos('not a decision checkpoint', E.Message) > 0);
  end;
end;

// The largest |p - q| over every option of every question.
function MaxProbabilityDiff(const A, B: TDecisionAnswers): double;
var
  QuestionPos, OptionPos: integer;
begin
  Result := 0;
  TAssert.AssertEquals('answer count', Length(A), Length(B));
  for QuestionPos := 0 to High(A) do
    for OptionPos := 0 to High(A[QuestionPos].Probabilities) do
      Result := Max(Result, Abs(A[QuestionPos].Probabilities[OptionPos] -
        B[QuestionPos].Probabilities[OptionPos]));
end;

procedure TTestNeuralDecisionHeads.TestKevFullWeightsParityWithKev;
var
  Oracle, QuestionOracle: TJSONObject;
  Engine: TChatEngine;
  Scorer: TKevScorer;
  Request: TDecisionRequest;
  Errors: TDecisionErrorDetails;
  Answers: TDecisionAnswers;
  Info: TDecisionAdapterInfo;
  InputTokens, OutputTokens, QuestionPos, OptionPos: integer;
  Where, CheckpointDir, Message: string;
  Expected: TJSONArray;
  ProbabilityDiff: double;
begin
  Oracle := LoadOracle(csKevFullFixture, 'kev_full_oracle.json');
  Engine := nil;
  Scorer := nil;
  try
    CheckpointDir := FixturePath(csKevFullFixture);
    Engine := NewEngine(csKevFullFixture, '', {OmitLMHead=}true);
    Scorer := TKevScorer.Create(Engine, 'kev-full-parity', CheckpointDir +
      DirectorySeparator + csOpenJevHeadFile);
    AssertTrue('parse', ParseDecisionRequest(Oracle.Strings['request'], Request, Errors));
    Scorer.Score(Request, Answers, InputTokens, OutputTokens);
    AssertIdsEqual('state ids', Oracle.Arrays['state_ids'], Scorer.LastStateIds);
    AssertEquals('question count', Oracle.Arrays['questions'].Count, Length(Answers));
    ProbabilityDiff := 0;
    for QuestionPos := 0 to High(Answers) do
    begin
      QuestionOracle := Oracle.Arrays['questions'].Objects[QuestionPos];
      Where := QuestionOracle.Strings['id'];
      AssertIdsEqual(Where + ': branch ids', QuestionOracle.Arrays['branch_ids'],
        Scorer.LastRows[QuestionPos].BranchIds);
      Expected := QuestionOracle.Arrays['probs'];
      AssertEquals(Where + ': probabilities', Expected.Count,
        Length(Answers[QuestionPos].Probabilities));
      for OptionPos := 0 to Expected.Count - 1 do
        ProbabilityDiff := Max(ProbabilityDiff, Abs(Expected.Floats[OptionPos] -
          Answers[QuestionPos].Probabilities[OptionPos]));
    end;
    AssertTrue(Format('probabilities within %g of kev: %g', [csKevParityTolerance,
      ProbabilityDiff]), ProbabilityDiff < csKevParityTolerance);
    // No LM head: the net ends at the final norm and refuses to generate.
    AssertTrue('last layer is the final norm', Engine.NN.GetLastLayer() is TNNetTokenRMSNorm);
    AssertTrue('hidden state = last layer', Engine.Session.HiddenLayer() =
      Engine.NN.GetLastLayer());
    Message := '';
    try
      Engine.GenerateFromIds(Copy(Scorer.LastStateIds, 0, 4), Engine.Opt);
    except
      on E: Exception do Message := E.Message;
    end;
    AssertTrue('chat refused: ' + Message, Pos('without its LM head', Message) > 0);
    Info := ReadDecisionAdapterInfo(CheckpointDir);
    AssertTrue('full-weight layout', Info.Layout = dalKevFullWeights);
    AssertEquals('nothing to merge', '', Info.LoRAAdapterDir);
    AssertEquals('head input', 16, Info.HeadInputSize);
    // An LM-head build of this layout has no lm_head tensor to load.
    Message := '';
    try
      BuildFromPretrained(CheckpointDir, 8, false).Free;
    except
      on E: EPretrainedImportError do Message := E.Message;
    end;
    AssertTrue('LM-head build refused: ' + Message, Pos('lm_head.weight', Message) > 0);
  finally
    Scorer.Free;
    Engine.Free;
    Oracle.Free;
  end;
end;

procedure TTestNeuralDecisionHeads.TestHeadlessBuildMatchesLMHeadBuild;
var
  Oracle: TJSONObject;
  Request: TDecisionRequest;
  Errors: TDecisionErrorDetails;
  WithHead, Headless: TDecisionAnswers;
  WithHeadEngine, HeadlessEngine: TChatEngine;
  HiddenSize, VocabSize: integer;

  function ScoreKev(Engine: TChatEngine): TDecisionAnswers;
  var
    Scorer: TKevScorer;
    InputTokens, OutputTokens: integer;
  begin
    Scorer := TKevScorer.Create(Engine, 'kev', FixturePath('kev_qwen3_5' +
      DirectorySeparator + 'adapter' + DirectorySeparator + csOpenJevHeadFile));
    try
      Scorer.Score(Request, Result, InputTokens, OutputTokens);
    finally
      Scorer.Free;
    end;
  end;

  function ScoreOpenJev(Engine: TChatEngine): TDecisionAnswers;
  var
    Scorer: TOpenJevScorer;
    InputTokens, OutputTokens: integer;
  begin
    Scorer := TOpenJevScorer.Create(Engine, 'open-jev', FixturePath(csOpenJevFixture));
    try
      Scorer.Score(Request, Result, InputTokens, OutputTokens);
    finally
      Scorer.Free;
    end;
  end;

  procedure AssertOnlyLMHeadDropped(const Where: string);
  begin
    AssertEquals(Where + ': one layer fewer', WithHeadEngine.NN.CountLayers() - 1,
      HeadlessEngine.NN.CountLayers());
    AssertEquals(Where + ': the vocab x hidden weights fewer',
      WithHeadEngine.NN.CountWeights() - int64(VocabSize) * HiddenSize,
      HeadlessEngine.NN.CountWeights());
    AssertTrue(Where + ': ends at the final norm',
      HeadlessEngine.NN.GetLastLayer() is TNNetTokenRMSNorm);
  end;

begin
  WithHeadEngine := nil;
  HeadlessEngine := nil;
  Oracle := LoadOracle('kev_qwen3_5', 'kev_oracle.json');
  try
    AssertTrue('parse', ParseDecisionRequest(Oracle.Strings['request'], Request, Errors));
    WithHeadEngine := NewKevEngine();
    HeadlessEngine := NewKevEngine({OmitLMHead=}true);
    VocabSize := WithHeadEngine.VocabSize;
    HiddenSize := WithHeadEngine.Session.HiddenState().Depth;
    AssertEquals('Kev: head-less hidden size', HiddenSize,
      HeadlessEngine.Session.HiddenState().Depth);
    AssertOnlyLMHeadDropped('Kev');
    WithHead := ScoreKev(WithHeadEngine);
    Headless := ScoreKev(HeadlessEngine);
    AssertTrue(Format('Kev: head-less = LM-head build: %g',
      [MaxProbabilityDiff(WithHead, Headless)]),
      MaxProbabilityDiff(WithHead, Headless) <= csHeadlessTolerance);
    FreeAndNil(HeadlessEngine);
    // The width-4 prefill twin borrows from the head-less net.
    HeadlessEngine := NewKevEngine({OmitLMHead=}true, {PrefillWindow=}4);
    AssertTrue('twin built', Assigned(HeadlessEngine.WindowNN));
    AssertEquals('twin is head-less too', HeadlessEngine.NN.CountLayers(),
      HeadlessEngine.WindowNN.CountLayers());
    Headless := ScoreKev(HeadlessEngine);
    AssertTrue('the twin fed whole windows', HeadlessEngine.LastPrefillWindows > 0);
    AssertTrue(Format('Kev: head-less with a prefill twin = LM-head build: %g',
      [MaxProbabilityDiff(WithHead, Headless)]),
      MaxProbabilityDiff(WithHead, Headless) <= csKevParityTolerance);
    FreeAndNil(HeadlessEngine);
    FreeAndNil(WithHeadEngine);
    FreeAndNil(Oracle);

    Oracle := LoadOracle(csOpenJevFixture, 'openjev_oracle.json');
    AssertTrue('parse', ParseDecisionRequest(Oracle.Strings['request'], Request, Errors));
    WithHeadEngine := NewOpenJevEngine();
    HeadlessEngine := NewOpenJevEngine({OmitLMHead=}true);
    VocabSize := WithHeadEngine.VocabSize;
    HiddenSize := WithHeadEngine.Session.HiddenState().Depth;
    AssertOnlyLMHeadDropped('Open-Jev');
    WithHead := ScoreOpenJev(WithHeadEngine);
    Headless := ScoreOpenJev(HeadlessEngine);
    AssertTrue(Format('Open-Jev: head-less = LM-head build: %g',
      [MaxProbabilityDiff(WithHead, Headless)]),
      MaxProbabilityDiff(WithHead, Headless) <= csHeadlessTolerance);
  finally
    HeadlessEngine.Free;
    WithHeadEngine.Free;
    Oracle.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralDecisionHeads);
end.
