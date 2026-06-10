program DecodeFeaturesBakeoff;
(*
DecodeFeaturesBakeoff: benchmarks the recently-landed DECODE-EFFICIENCY
features on a REAL tokenized NLP workload (TinyStories, 3k-token vocabulary),
one phase per feature, selected by CLI argument:

    ./DecodeFeaturesBakeoff --phase N        (N in 1..5)

Each phase is self-budgeted to finish within 270 seconds of wall clock:
training is time-boxed INSIDE the program (TNeuralDataLoadingFit.ShouldQuit
is raised from OnAfterStep when the train budget expires), leaving the rest
of the budget for the decode benchmark itself.

  Phase 1 (commit 2dacc95): KV-cache incremental decode vs full re-encode.
           Trains a small RoPE transformer decoder, saves it to
           bakeoff-phase1.nn, then greedy-generates from fixed prompts two
           ways -- (a) full re-encode of the whole prefix per token (status
           quo) and (b) token-at-a-time through a weight-copied 1-token step
           net with BeginIncrementalDecode + RoPE PositionOffset per step.
           HARD ASSERT: both arms emit the SAME tokens. Prints ms/token.
  Phase 2 (commit 4d95378): MTP-heads self-speculative decode. Trains the
           SAME trunk as phase 1 but with TNNet.AddMultiTokenPrediction
           (NumFuture=3): head 0 is the ordinary next-token head, heads 1..2
           forecast t+2/t+3 and double as a built-in draft -- no second
           network. Greedy self-speculative decoding: each full re-encode
           forward verifies the pending drafts (accept-until-first-mismatch;
           a rejection still commits head-0's corrected token; full
           acceptance yields a bonus token) and drafts the next block from
           heads 1..2 at the last committed row. KV-cache composition is NOT
           attempted (drafts come from forwards a cache never saw after
           rejection -- known open problem), so both arms re-encode fully and
           the win measured is forwards/token and wall clock. HARD ASSERT:
           self-speculative output == plain greedy from the same model.
  Phase 3 (commit 80dd830): DiagonalSSM O(1)-per-step decode.    [stub]
  Phase 4 (commit a8f3077): MLA decoupled-RoPE latent KV cache.  [stub]
  Phase 5 (commit 42dd5dd): speculative decoding COMPOSED with the KV cache
           (TruncateCache rollback). Requires phase 1's checkpoint. Trains a
           cheap TokenShift draft briefly, then greedy speculative decoding
           with cached verification: prefill once, short multi-token verify
           passes, TruncateCache on rejection. HARD ASSERT: speculative
           output == plain greedy output. Prints accept rate, tokens/pass,
           target-forwards/token and a three-way ms/token comparison
           (plain-full vs plain-cached vs speculative-cached).

MODEL. All phases share one config (constants below): RoPE positional scheme
everywhere -- TNNetEmbedding (token-only) + UseRoPE=true in
AddTransformerEncoderBlock -- because TNNetRotaryEmbedding.PositionOffset is
what makes EXACT streamed decode possible (learned-absolute
TokenAndPositionalEmbedding has no offset and must not be used here). The
norm class is TNNetDyT (Dynamic Tanh): DyT is per-element with NO cross-token
statistics, so a width-1 step net computes exactly what the width-48 full net
computes at that position. (TNNetLayerNorm/TNNetRMSNorm normalize over the
WHOLE (SeqLen,1,Depth) sample, sequence axis included, which breaks
full-vs-incremental token exactness and must not be used here either.)

DATASET (NOT in the repository; download first):
  git clone https://huggingface.co/datasets/schuler/TinyStories4Pascal-Tokenized-v2
  unzip TinyStories4Pascal-Tokenized-v2/tinystories-100k-tokenized3k.csv.zip -d datasets/
  unzip TinyStories4Pascal-Tokenized-v2/tinystories-vocab-3k-cai.csv.zip -d datasets/
The first run filters the first csMaxRows usable rows into
datasets/bakeoff_temp.csv and later runs reuse that cache, so data loading
stays a small slice of the 270 s budget.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, BaseUnix, Unix, {$ENDIF}
  Classes,
  neuralnetwork,
  neuralvolume,
  neuralfit,
  neuralthread,
  neuraldatasets,
  neuralab,
  neuraltokenizer,
  CustApp,
  Math,
  sysutils;

const
  // ---- shared model config (all phases) ----
  csContextLen     = 48;
  csEmbedDim       = 128;   // d_k = 128/8 = 16 (even, as RoPE requires)
  csHeads          = 8;
  csBlocks         = 2;
  csFFNDim         = 512;
  csModelVocabSize = 3000;
  csBatchSize      = 32;
  // ---- data ----
  csTrainingFileName = 'datasets/tinystories-100k-tokenized3k.csv';
  csVocabFileName    = 'datasets/tinystories-vocab-3k-cai.csv';
  csFilteredFileName = 'datasets/bakeoff_temp.csv'; // cached filter output
  csMaxRows          = 24000; // subsample: load time stays small vs 270 s
  // ---- budgets (seconds of wall clock from process start) ----
  csTotalBudgetSec   = 270;
  csBenchReserveSec  = 60;   // kept free for the decode benchmark
  csTrainCapSec      = 185;  // phase 1: never train longer than this
  csTrainCapSecMTP   = 195;  // phase 2: MTP decode bench is cheap, train more
  csDraftCapSec      = 160;  // phase 5: draft training cap
  // ---- decode benchmark ----
  csGenTokens        = 40;   // tokens generated per prompt per arm
  csSpecK            = 4;    // speculative block size (draft proposes K)
  csCheckpointFile   = 'bakeoff-phase1.nn';
  // ---- phase 5 draft model ----
  csDraftEmbedDim    = 64;
  csDraftFFNDim      = 256;
  // ---- phase 2 MTP heads ----
  csNumFuture        = 3;    // t+1 (committed) + t+2,t+3 (the built-in draft)

const
  csPrompts: array[0..1] of string = ('one day', 'once upon a time');

// ---------------------------------------------------------------------------
// Microsecond-resolution wall clock in ms since the first call (pattern from
// examples/IncrementalDecode: FPC types the 1000.0 literal as SINGLE, so at
// absolute Unix-epoch scale the arithmetic would quantize; rebase to start).
// ---------------------------------------------------------------------------
{$IFDEF UNIX}
var
  GBaseSec: int64 = -1;

function NowMs(): double;
var
  tv: TTimeVal;
begin
  fpGetTimeOfDay(@tv, nil);
  if GBaseSec < 0 then GBaseSec := tv.tv_sec;
  Result := (tv.tv_sec - GBaseSec) * 1000.0 + tv.tv_usec / 1000.0;
end;
{$ELSE}
var
  GBaseMs: double = -1;

function NowMs(): double;
begin
  if GBaseMs < 0 then GBaseMs := Now() * 24 * 3600 * 1000;
  Result := Now() * 24 * 3600 * 1000 - GBaseMs;
end;
{$ENDIF}

function ElapsedSec(): double;
begin
  Result := NowMs() / 1000.0;
end;

type
  TTokenBuf = array[0..csContextLen - 1] of integer;

  // A weight-copied twin of the trained net at a short input width, with every
  // per-head SDPA in KV-cache incremental-decode mode and every per-head RoPE
  // layer collected so PositionOffset can be set to the ABSOLUTE position of
  // the streamed window before each cached forward.
  TStepNet = record
    Net: TNNet;
    Width: integer;
    SDPAs: array of TNNetScaledDotProductAttention;
    Ropes: array of TNNetRotaryEmbedding;
    InV: TNNetVolume;
  end;

  { TDecodeBakeoff }
  TDecodeBakeoff = class(TCustomApplication)
  protected
    FDataset: array of array of integer;
    FDatasetSize: integer;
    FDictionary: TNeuralTokenizer;
    FNN: TNNet;                       // the net currently being trained
    NFit: TNeuralDataLoadingFit;      // the fit currently running
    FTrainDeadlineMs: double;         // OnAfterStep raises ShouldQuit past this
    FNumFuture: integer;              // >1 only in phase 2 (MTP target layout)
    procedure DoRun; override;
    // shared helpers
    procedure RequireDatasetFiles;
    procedure LoadDataset;
    function BuildModel(ContextLen: integer): TNNet;
    function BuildDraft(ContextLen: integer): TNNet;
    procedure TrainTimeBoxed(NN: TNNet; BudgetSec: double;
      SamplesPerEpoch: integer; LearningRate: TNeuralFloat);
    procedure CreateStepNet(out Step: TStepNet; Width: integer; Source: TNNet);
    procedure FreeStepNet(var Step: TStepNet);
    procedure CachedWindowForward(var Step: TStepNet; const Seq: TTokenBuf;
      StartPos, NReal: integer);
    function TokensToText(const Toks: TTokenBuf; StartIdx, Len: integer): string;
    function PromptToTokens(const Prompt: string; out Buf: TTokenBuf): integer;
    // greedy decode arms
    procedure GreedyFull(NN: TNNet; var Toks: TTokenBuf; PromptLen, MaxNew: integer;
      out OutLen: integer; out Ms: double; out Fwds: integer);
    procedure GreedyCached(var Step: TStepNet; var Toks: TTokenBuf;
      PromptLen, MaxNew: integer; out OutLen: integer; out Ms: double;
      out Fwds: integer);
    procedure GreedySpecCached(var Step: TStepNet; Draft: TNNet;
      var Toks: TTokenBuf; PromptLen, MaxNew: integer; out OutLen: integer;
      out Ms: double; out TgtFwds, DraftFwds, Passes, Proposed, Accepted: integer);
    // phase 2 (MTP heads as built-in draft; full re-encode arms)
    function BuildModelMTP(ContextLen: integer): TNNet;
    function MTPHeadArgmax(NN: TNNet; Row, Head: integer): integer;
    procedure MTPGreedyFull(NN: TNNet; var Toks: TTokenBuf;
      PromptLen, MaxNew: integer; out OutLen: integer; out Ms: double;
      out Fwds: integer);
    procedure MTPSelfSpec(NN: TNNet; var Toks: TTokenBuf;
      PromptLen, MaxNew: integer; out OutLen: integer; out Ms: double;
      out Fwds, Passes, Committed: integer;
      var Verified, Accepted: array of integer);
    // phases
    procedure RunPhase1;
    procedure RunPhase2;
    procedure RunPhase5;
    procedure RunStub(Phase: integer);
  public
    procedure OnAfterStep(Sender: TObject);
    procedure OnAfterEpoch(Sender: TObject);
    procedure GetTrainingPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetValidationPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetTestPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
  end;

// ---------------------------------------------------------------------------
// Data pipeline (pattern copied from TransformerWithTokenizer.lpr, with a
// fail-fast download hint and a cached, row-capped filter step).
// ---------------------------------------------------------------------------
procedure TDecodeBakeoff.RequireDatasetFiles;
begin
  if FileExists(csTrainingFileName) and FileExists(csVocabFileName) then exit;
  WriteLn('ERROR: dataset files not found:');
  WriteLn('  ', csTrainingFileName);
  WriteLn('  ', csVocabFileName);
  WriteLn('Download them first (from this example''s directory):');
  WriteLn('  git clone https://huggingface.co/datasets/schuler/TinyStories4Pascal-Tokenized-v2');
  WriteLn('  unzip TinyStories4Pascal-Tokenized-v2/tinystories-100k-tokenized3k.csv.zip -d datasets/');
  WriteLn('  unzip TinyStories4Pascal-Tokenized-v2/tinystories-vocab-3k-cai.csv.zip -d datasets/');
  Halt(1);
end;

procedure TDecodeBakeoff.LoadDataset;
begin
  RequireDatasetFiles();
  WriteLn('Loading vocabulary: ', csVocabFileName);
  FDictionary.LoadVocabularyFromFile(csVocabFileName);
  if FileExists(csFilteredFileName) then
    WriteLn('Reusing cached filtered CSV: ', csFilteredFileName)
  else
  begin
    WriteLn('Filtering first ', csMaxRows, ' usable rows into ', csFilteredFileName);
    FilterCSVWithNumbersUpToMax(csTrainingFileName, csFilteredFileName,
      csModelVocabSize - 1, csMaxRows);
  end;
  LoadIntegersInCSV(csFilteredFileName, FDataset);
  FDatasetSize := Length(FDataset);
  WriteLn('Loaded dataset with ', FDatasetSize, ' rows. [t=',
    ElapsedSec():0:1, 's]');
end;

// ---------------------------------------------------------------------------
// Models.
// ---------------------------------------------------------------------------
// The shared decoder at an arbitrary input width: every layer's parameter
// shapes are sequence-length independent, so a width-1 (or width-K+1) twin can
// CopyWeights() from the width-csContextLen trained net.
// Norm class is TNNetDyT on purpose: see header (per-element, no cross-token
// statistics, so cached/streamed decode is exact).
function TDecodeBakeoff.BuildModel(ContextLen: integer): TNNet;
var
  B: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, 1));
  // Token-only embedding: position is injected by RoPE inside attention.
  Result.AddLayer(TNNetEmbedding.Create(csModelVocabSize, csEmbedDim, 0, 0.02));
  for B := 1 to csBlocks do
    Result.AddTransformerEncoderBlock(csHeads, csFFNDim,
      {PreNorm=}true, {CausalMask=}true, {UseRoPE=}true,
      {NormClass=}TNNetDyT);
  Result.AddLayer([
    TNNetPointwiseConvLinear.Create(csEmbedDim),
    TNNetPointwiseConvLinear.Create(csModelVocabSize),
    TNNetPointwiseSoftMax.Create(1)
  ]);
end;

// Cheap attention-free draft for phase 5: TokenShift mixing (causal by
// construction) + a narrow pointwise MLP. Far fewer MACs per forward than the
// target, yet bigram-ish enough to agree with greedy TinyStories continuations
// at a useful rate.
function TDecodeBakeoff.BuildDraft(ContextLen: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(csModelVocabSize, csDraftEmbedDim, 0, 0.02));
  Result.AddLayer(TNNetTokenShift.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(csDraftFFNDim));
  Result.AddLayer(TNNetReLU.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(csModelVocabSize));
  Result.AddLayer(TNNetPointwiseSoftMax.Create(1));
end;

// Phase 2 model: the SAME trunk as BuildModel, but the single next-token head
// is replaced by TNNet.AddMultiTokenPrediction(csNumFuture): NumFuture
// parallel per-token (PointwiseConvLinear + softmax) heads tapping the shared
// trunk, concatenated along depth. Output (ContextLen,1,csNumFuture*Vocab);
// head h occupies depth slab [h*Vocab .. (h+1)*Vocab-1] and forecasts, at row
// t, the token at position t+1+h.
function TDecodeBakeoff.BuildModelMTP(ContextLen: integer): TNNet;
var
  B: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(csModelVocabSize, csEmbedDim, 0, 0.02));
  for B := 1 to csBlocks do
    Result.AddTransformerEncoderBlock(csHeads, csFFNDim,
      {PreNorm=}true, {CausalMask=}true, {UseRoPE=}true,
      {NormClass=}TNNetDyT);
  Result.AddLayer(TNNetPointwiseConvLinear.Create(csEmbedDim));
  Result.AddMultiTokenPrediction(csNumFuture, csModelVocabSize);
end;

// ---------------------------------------------------------------------------
// Time-boxed training. TNeuralDataLoadingFit exposes a public ShouldQuit flag
// that the training loop checks after every batch (and before validation /
// test), so a clean mid-fit abort is just "raise it from OnAfterStep".
// ---------------------------------------------------------------------------
procedure TDecodeBakeoff.OnAfterStep(Sender: TObject);
begin
  if NowMs() > FTrainDeadlineMs then
    NFit.ShouldQuit := true;
end;

procedure TDecodeBakeoff.OnAfterEpoch(Sender: TObject);
var
  Toks: TTokenBuf;
  PLen, OLen, Fw: integer;
  SampleMs: double;
begin
  // Convergence clue after each completed epoch.
  if FNumFuture > 1 then
  begin
    // MTP net: GenerateStringFromCasualNN would argmax over the WHOLE
    // NumFuture*Vocab concat depth (token ids >= Vocab); sample head 0 only.
    PLen := PromptToTokens('one day', Toks);
    MTPGreedyFull(NFit.NN, Toks, PLen, 16, OLen, SampleMs, Fw);
    WriteLn('[t=', ElapsedSec():0:1, 's] sample (head 0): one day ',
      TokensToText(Toks, PLen, OLen - PLen), '.');
  end
  else
    WriteLn('[t=', ElapsedSec():0:1, 's] sample: ',
      GenerateStringFromCasualNN(NFit.NN, FDictionary, 'one day', nil,
        csNeuralEncodingMethodIntChar), '.');
end;

procedure TDecodeBakeoff.TrainTimeBoxed(NN: TNNet; BudgetSec: double;
  SamplesPerEpoch: integer; LearningRate: TNeuralFloat);
var
  Opt: TNeuralOptimizerAdam;
begin
  WriteLn('Training time box: ', BudgetSec:0:0, 's  (', SamplesPerEpoch,
    ' samples/epoch, batch ', csBatchSize, ', Adam lr=', LearningRate:0:5, ')');
  FNN := NN; // GetTrainingPair sizes its volumes from FNN
  Opt := TNeuralOptimizerAdam.Create(0.9, 0.98);
  NFit := TNeuralDataLoadingFit.Create();
  try
    NFit.LogEveryBatches := 25;
    NFit.InitialLearningRate := LearningRate;
    NFit.Optimizer := Opt;
    NFit.LearningRateDecay := 0.00;
    NFit.StaircaseEpochs := 1;
    NFit.L2Decay := 0.1;
    NFit.EnableMultiClassLoss();
    NFit.EnableClassComparisonInLastPixel();
    NFit.AvgWeightEpochCount := 1;
    NFit.OnAfterEpoch := @OnAfterEpoch;
    NFit.OnAfterStep := @OnAfterStep;
    FTrainDeadlineMs := NowMs() + BudgetSec * 1000.0;
    NFit.FitLoading(
      NN,
      {TrainingVolumesCount=}SamplesPerEpoch,
      {ValidationVolumesCount=}SamplesPerEpoch div 20,
      {TestVolumesCount=}0,
      {batchsize=}csBatchSize,
      {epochs=}500,
      @GetTrainingPair, @GetValidationPair, @GetTestPair
    );
  finally
    NFit.Free;
    NFit := nil;
    Opt.Free;
  end;
  WriteLn('Training stopped. [t=', ElapsedSec():0:1, 's]');
end;

// Training pairs: identical scheme to TransformerWithTokenizer.lpr (predict
// the last token of a random-length prefix; per-position one-hot targets).
procedure TDecodeBakeoff.GetTrainingPair(Idx: integer; ThreadId: integer;
  pInput, pOutput: TNNetVolume);
var
  SampleId, SampleLen, SampleCutPosition, ExpectedTokenInt: integer;
  AIntegerArray: array of integer;
  t, h, fut: integer;
begin
  if FNN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(FNN.GetFirstLayer().Output);
  if FNN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(FNN.GetLastLayer().Output);
  SampleId := Random(FDatasetSize);
  SampleLen := Min(Length(FDataset[SampleId]), pInput.SizeX);
  SampleCutPosition := SampleLen;
  ExpectedTokenInt := FDataset[SampleId][SampleCutPosition - 1];
  AIntegerArray := Copy(FDataset[SampleId], 0, SampleCutPosition);
  pInput.Fill(0);
  pInput.CopyNoChecksIntArr(AIntegerArray);
  if FNumFuture > 1 then
  begin
    // MTP targets: head h's slab at row t supervises the token at position
    // t+1+h (reading past the input window into the same story is fine; rows
    // past the story end carry no supervision).
    pOutput.Fill(0);
    for t := 0 to SampleCutPosition - 1 do
      for h := 0 to FNumFuture - 1 do
      begin
        fut := t + 1 + h;
        if fut < Length(FDataset[SampleId]) then
          pOutput[t, 0, h * csModelVocabSize + FDataset[SampleId][fut]] := 1.0;
      end;
  end
  else
  begin
    AIntegerArray := Copy(FDataset[SampleId], 1, SampleCutPosition);
    pOutput.OneHotEncoding(AIntegerArray);
  end;
  pOutput.Tag := ExpectedTokenInt;
end;

procedure TDecodeBakeoff.GetValidationPair(Idx: integer; ThreadId: integer;
  pInput, pOutput: TNNetVolume);
var
  SampleId, SampleLen, SampleCutPosition, ExpectedTokenInt: integer;
  AIntegerArray: array of integer;
  t, h, fut: integer;
begin
  if FNN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(FNN.GetFirstLayer().Output);
  if FNN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(FNN.GetLastLayer().Output);
  SampleId := (Idx * 20) mod FDatasetSize;
  SampleLen := Min(Length(FDataset[SampleId]), pInput.SizeX);
  SampleCutPosition := SampleLen;
  ExpectedTokenInt := FDataset[SampleId][SampleCutPosition - 1];
  AIntegerArray := Copy(FDataset[SampleId], 0, SampleCutPosition);
  pInput.Fill(0);
  pInput.CopyNoChecksIntArr(AIntegerArray);
  if FNumFuture > 1 then
  begin
    pOutput.Fill(0);
    for t := 0 to SampleCutPosition - 1 do
      for h := 0 to FNumFuture - 1 do
      begin
        fut := t + 1 + h;
        if fut < Length(FDataset[SampleId]) then
          pOutput[t, 0, h * csModelVocabSize + FDataset[SampleId][fut]] := 1.0;
      end;
  end
  else
  begin
    AIntegerArray := Copy(FDataset[SampleId], 1, SampleCutPosition);
    pOutput.OneHotEncoding(AIntegerArray);
  end;
  pOutput.Tag := ExpectedTokenInt;
end;

procedure TDecodeBakeoff.GetTestPair(Idx: integer; ThreadId: integer;
  pInput, pOutput: TNNetVolume);
begin
  GetValidationPair(Idx, ThreadId, pInput, pOutput);
end;

// ---------------------------------------------------------------------------
// Step-net plumbing (KV-cache incremental decode; pattern from
// examples/SpeculativeDecoding extended with the RoPE PositionOffset contract).
// ---------------------------------------------------------------------------
procedure TDecodeBakeoff.CreateStepNet(out Step: TStepNet; Width: integer;
  Source: TNNet);
var
  i, n: integer;
begin
  Step.Width := Width;
  Step.Net := BuildModel(Width);
  // Same builder, so the layer-by-layer weight copy is exact (all parameter
  // shapes are sequence-length independent).
  Step.Net.CopyWeights(Source);
  SetLength(Step.SDPAs, 0);
  SetLength(Step.Ropes, 0);
  for i := 0 to Step.Net.Layers.Count - 1 do
  begin
    if Step.Net.Layers[i] is TNNetScaledDotProductAttention then
    begin
      n := Length(Step.SDPAs);
      SetLength(Step.SDPAs, n + 1);
      Step.SDPAs[n] := TNNetScaledDotProductAttention(Step.Net.Layers[i]);
    end;
    if Step.Net.Layers[i] is TNNetRotaryEmbedding then
    begin
      n := Length(Step.Ropes);
      SetLength(Step.Ropes, n + 1);
      Step.Ropes[n] := TNNetRotaryEmbedding(Step.Net.Layers[i]);
    end;
  end;
  // Worst transient cache load: full context committed + one whole window.
  for i := 0 to High(Step.SDPAs) do
    Step.SDPAs[i].BeginIncrementalDecode(csContextLen + Width);
  Step.InV := TNNetVolume.Create(Width, 1, 1);
end;

procedure TDecodeBakeoff.FreeStepNet(var Step: TStepNet);
begin
  Step.InV.Free;
  Step.Net.Free;
  SetLength(Step.SDPAs, 0);
  SetLength(Step.Ropes, 0);
end;

// One cached window forward: tokens Seq[StartPos..StartPos+NReal-1] fill the
// window (zero padding AFTER every real token, so the cached causal path never
// lets a real query see a pad; pad K/V is discarded by the caller's next
// TruncateCache). Every RoPE layer is shifted so the window is rotated at its
// ABSOLUTE positions. Output row r is the softmax at position StartPos+r.
procedure TDecodeBakeoff.CachedWindowForward(var Step: TStepNet;
  const Seq: TTokenBuf; StartPos, NReal: integer);
var
  t: integer;
begin
  Step.InV.Fill(0);
  for t := 0 to NReal - 1 do Step.InV.FData[t] := Seq[StartPos + t];
  for t := 0 to High(Step.Ropes) do Step.Ropes[t].PositionOffset := StartPos;
  Step.Net.Compute(Step.InV);
end;

// ---------------------------------------------------------------------------
// Prompt/text helpers.
// ---------------------------------------------------------------------------
function TDecodeBakeoff.PromptToTokens(const Prompt: string;
  out Buf: TTokenBuf): integer;
var
  Tokens: TNeuralIntegerArray;
  i: integer;
begin
  FDictionary.Tokenize(Prompt, Tokens);
  Result := Length(Tokens);
  for i := 0 to csContextLen - 1 do Buf[i] := 0;
  for i := 0 to Result - 1 do Buf[i] := Tokens[i];
  SetLength(Tokens, 0);
end;

// Detokenize Toks[StartIdx..StartIdx+Len-1] for display; stops at the first
// special token (<2) for readability.
function TDecodeBakeoff.TokensToText(const Toks: TTokenBuf;
  StartIdx, Len: integer): string;
var
  i: integer;
begin
  Result := '';
  for i := StartIdx to StartIdx + Len - 1 do
  begin
    if Toks[i] < 2 then break;
    if FDictionary.TokenizerHasSeparator and (Result <> '') then
      Result := Result + ' ' + FDictionary.DeTokenize(Toks[i])
    else
      Result := Result + FDictionary.DeTokenize(Toks[i]);
  end;
end;

// ---------------------------------------------------------------------------
// Decode arms. All three are GREEDY (argmax) so token-exact equality is the
// correctness gate. Generation is fixed-length (no early EOS stop) so the
// arms stay step-for-step comparable; display trims at the first EOS.
// ---------------------------------------------------------------------------

// (a) Status quo: re-encode the WHOLE zero-padded prefix for every token and
// read the argmax at the last real row (exactly what a cache-less sampler,
// e.g. GenerateStringFromCasualNN, pays per token).
procedure TDecodeBakeoff.GreedyFull(NN: TNNet; var Toks: TTokenBuf;
  PromptLen, MaxNew: integer; out OutLen: integer; out Ms: double;
  out Fwds: integer);
var
  InV: TNNetVolume;
  t: integer;
  T0: double;
begin
  InV := TNNetVolume.Create(csContextLen, 1, 1);
  try
    OutLen := PromptLen;
    Fwds := 0;
    T0 := NowMs();
    while (OutLen - PromptLen < MaxNew) and (OutLen < csContextLen) do
    begin
      InV.Fill(0);
      for t := 0 to OutLen - 1 do InV.FData[t] := Toks[t];
      NN.Compute(InV);
      Inc(Fwds);
      Toks[OutLen] := NN.GetLastLayer().Output.GetClassOnPixel(OutLen - 1, 0);
      Inc(OutLen);
    end;
    Ms := NowMs() - T0;
  finally
    InV.Free;
  end;
end;

// (b) KV cache: prefill the prompt token-at-a-time through a width-1 step net
// (each step's RoPE offset = its absolute position = the running cache
// length), then decode one token per single-token cached forward.
procedure TDecodeBakeoff.GreedyCached(var Step: TStepNet; var Toks: TTokenBuf;
  PromptLen, MaxNew: integer; out OutLen: integer; out Ms: double;
  out Fwds: integer);
var
  t: integer;
  T0: double;
begin
  Fwds := 0;
  T0 := NowMs();
  for t := 0 to High(Step.SDPAs) do Step.SDPAs[t].ResetCache();
  // Prefill tokens 0..PromptLen-2; the LAST prompt token is the first decode
  // step's input (its output row predicts the first new token).
  for t := 0 to PromptLen - 2 do
  begin
    CachedWindowForward(Step, Toks, t, 1);
    Inc(Fwds);
  end;
  OutLen := PromptLen;
  while (OutLen - PromptLen < MaxNew) and (OutLen < csContextLen) do
  begin
    CachedWindowForward(Step, Toks, OutLen - 1, 1);
    Inc(Fwds);
    Toks[OutLen] := Step.Net.GetLastLayer().Output.GetClassOnPixel(0, 0);
    Inc(OutLen);
  end;
  Ms := NowMs() - T0;
end;

// (c) Speculative + KV cache (phase 5). The draft proposes up to K greedy
// tokens; `Draft` here is a WIDTH-2 weight-copied twin of the trained draft:
// a TokenShift+pointwise-MLP output at position t depends ONLY on tokens t-1
// and t, so feeding [prev|cur] and reading row 1 reproduces the full-width
// draft prediction exactly at ~1/csContextLen of the MACs (the draft's
// equivalent of a KV cache); the target verifies the whole block in ONE cached
// forward over
// the window [last committed token | drafts] (multi-token prefill semantics).
// Greedy acceptance: accept while draft token == target argmax; on the first
// mismatch commit the target's argmax and TruncateCache back to the committed
// prefix (rejected/pad K/V become stale and are discarded). If all are
// accepted, the bonus token (target argmax at the last row) is free from the
// SAME forward. Every committed token is the target's argmax given its
// prefix, so the output equals plain greedy decoding token-for-token.
procedure TDecodeBakeoff.GreedySpecCached(var Step: TStepNet; Draft: TNNet;
  var Toks: TTokenBuf; PromptLen, MaxNew: integer; out OutLen: integer;
  out Ms: double; out TgtFwds, DraftFwds, Passes, Proposed, Accepted: integer);
var
  DraftIn: TNNetVolume;
  DraftToks: array[0..csSpecK - 1] of integer;
  i, t, nP, tgt, done, w: integer;
  rejected: boolean;
  T0: double;
begin
  DraftIn := TNNetVolume.Create(2, 1, 1);
  try
    TgtFwds := 0; DraftFwds := 0; Passes := 0; Proposed := 0; Accepted := 0;
    T0 := NowMs();
    // PREFILL once: cache the K/V of tokens 0..PromptLen-2 in short windows
    // (the last committed token is always re-fed as the next window's first
    // query, so its softmax row is fresh each pass).
    for t := 0 to High(Step.SDPAs) do Step.SDPAs[t].ResetCache();
    done := 0;
    while done < PromptLen - 1 do
    begin
      w := Min(Step.Width, PromptLen - 1 - done);
      CachedWindowForward(Step, Toks, done, w);
      Inc(TgtFwds);
      done := done + w;
      for t := 0 to High(Step.SDPAs) do Step.SDPAs[t].TruncateCache(done);
    end;

    OutLen := PromptLen;
    while (OutLen - PromptLen < MaxNew) and (OutLen < csContextLen) do
    begin
      Inc(Passes);
      // 1. draft proposes up to K greedy tokens over the growing prefix.
      nP := 0;
      for i := 0 to csSpecK - 1 do
      begin
        if OutLen + i >= csContextLen then break;
        // width-2 window [prev|cur]; row 1 = the full-width draft's row at
        // absolute position OutLen+i-1 (TokenShift reach is exactly 1 token).
        DraftIn.FData[0] := Toks[OutLen + i - 2];
        DraftIn.FData[1] := Toks[OutLen + i - 1];
        Draft.Compute(DraftIn);
        Inc(DraftFwds);
        DraftToks[i] := Draft.GetLastLayer().Output.GetClassOnPixel(1, 0);
        Toks[OutLen + i] := DraftToks[i];
        Inc(nP);
      end;
      if nP = 0 then break;
      Inc(Proposed, nP);

      // 2. ONE cached verification forward over [last committed | drafts]:
      //    row i is the target's distribution at context length OutLen+i.
      CachedWindowForward(Step, Toks, OutLen - 1, nP + 1);
      Inc(TgtFwds);

      // 3. greedy accept/reject walk.
      rejected := false;
      i := 0;
      while i < nP do
      begin
        tgt := Step.Net.GetLastLayer().Output.GetClassOnPixel(i, 0);
        if tgt = DraftToks[i] then
        begin
          Inc(Accepted);
          Inc(i);
        end
        else
        begin
          Toks[OutLen + i] := tgt;  // commit the target's token instead
          Inc(i);                   // i tokens committed in total
          rejected := true;
          break;
        end;
      end;

      if not rejected then
      begin
        OutLen := OutLen + nP;
        if OutLen < csContextLen then
        begin
          // bonus token: free from row nP of the same cached forward.
          Toks[OutLen] := Step.Net.GetLastLayer().Output.GetClassOnPixel(nP, 0);
          Inc(OutLen);
        end;
      end
      else
        OutLen := OutLen + i;

      // 4. ROLLBACK: the caches transiently hold the whole window (pads and
      //    rejected drafts included). Keep exactly the committed prefix minus
      //    its last token (re-fed as the next window's first query).
      for t := 0 to High(Step.SDPAs) do Step.SDPAs[t].TruncateCache(OutLen - 1);
    end;
    // A fully-accepted final pass may overshoot MaxNew by up to K; trim so the
    // exactness comparison and ms/token are over exactly MaxNew tokens.
    if OutLen > PromptLen + MaxNew then OutLen := PromptLen + MaxNew;
    Ms := NowMs() - T0;
  finally
    DraftIn.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Phase 1: KV-cache incremental decode vs full re-encode (commit 2dacc95).
// ---------------------------------------------------------------------------
procedure TDecodeBakeoff.RunPhase1;
var
  Trained: TNNet;
  Step: TStepNet;
  FullToks, CachedToks: TTokenBuf;
  PromptLen, FullLen, CachedLen: integer;
  FullMs, CachedMs: double;
  FullFwds, CachedFwds: integer;
  p, t, NewToks: integer;
  TrainSec: double;
begin
  WriteLn('=== Phase 1: KV-cache incremental decode vs full re-encode (2dacc95) ===');
  LoadDataset();

  Trained := BuildModel(csContextLen);
  WriteLn('Model: ctx=', csContextLen, ' d_model=', csEmbedDim, ' blocks=',
    csBlocks, ' heads=', csHeads, ' ffn=', csFFNDim, ' vocab=',
    csModelVocabSize, '  params=', Trained.CountWeights());

  TrainSec := Min(csTrainCapSec,
    csTotalBudgetSec - ElapsedSec() - csBenchReserveSec);
  if TrainSec < 10 then
  begin
    WriteLn('ERROR: no time left to train (data loading took too long).');
    Halt(1);
  end;
  TrainTimeBoxed(Trained, TrainSec, {SamplesPerEpoch=}3200, {lr=}0.0005);

  Trained.SaveToFile(csCheckpointFile);
  WriteLn('Checkpoint saved: ', csCheckpointFile);

  CreateStepNet(Step, {Width=}1, Trained);
  try
    WriteLn;
    WriteLn('Greedy decode showdown (', csGenTokens, ' tokens per prompt):');
    for p := 0 to High(csPrompts) do
    begin
      PromptLen := PromptToTokens(csPrompts[p], FullToks);
      CachedToks := FullToks;

      GreedyFull(Trained, FullToks, PromptLen, csGenTokens,
        FullLen, FullMs, FullFwds);
      GreedyCached(Step, CachedToks, PromptLen, csGenTokens,
        CachedLen, CachedMs, CachedFwds);

      // HARD ASSERT: token-exact equality between the two arms.
      if FullLen <> CachedLen then
      begin
        WriteLn('ASSERT FAILED: full arm emitted ', FullLen - PromptLen,
          ' tokens but cached arm emitted ', CachedLen - PromptLen, '.');
        Halt(1);
      end;
      for t := PromptLen to FullLen - 1 do
        if FullToks[t] <> CachedToks[t] then
        begin
          WriteLn('ASSERT FAILED: decode arms diverge at position ', t,
            ' (full=', FullToks[t], ' cached=', CachedToks[t],
            ') for prompt "', csPrompts[p], '".');
          Halt(1);
        end;

      NewToks := FullLen - PromptLen;
      WriteLn;
      WriteLn('Prompt: "', csPrompts[p], '"');
      WriteLn('  text: ', csPrompts[p], ' ',
        TokensToText(FullToks, PromptLen, NewToks));
      WriteLn(Format('  full re-encode : %8.2f ms total  %6.2f ms/token  (%d forwards)',
        [FullMs, FullMs / Max(1, NewToks), FullFwds]));
      WriteLn(Format('  KV-cached step : %8.2f ms total  %6.2f ms/token  (%d forwards incl. prefill)',
        [CachedMs, CachedMs / Max(1, NewToks), CachedFwds]));
      WriteLn(Format('  speedup        : %6.2fx   tokens match: YES (%d/%d)',
        [FullMs / Max(CachedMs, 1e-9), NewToks, NewToks]));
    end;
    WriteLn;
    WriteLn('Phase 1: PASS (cached decode is token-exact). [t=',
      ElapsedSec():0:1, 's]');
  finally
    FreeStepNet(Step);
    Trained.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Phase 2: MTP heads as a built-in draft -- self-speculative greedy decode
// from ONE model (commit 4d95378; pattern from examples/SelfSpeculativeDecoding
// ported to this real tokenized workload).
//
// Both decode arms below use FULL RE-ENCODE forwards on purpose: composing the
// KV cache with MTP drafting is a known open problem (the drafts for the next
// pass come from a forward whose window the cache never saw after a rejection),
// so the win measured here is forwards/token and wall clock at equal
// per-forward cost -- NOT a cache composition.
// ---------------------------------------------------------------------------

// Greedy argmax of head Head's softmax slab at row Row: the MTP model's
// forecast for the token at position Row+1+Head.
function TDecodeBakeoff.MTPHeadArgmax(NN: TNNet; Row, Head: integer): integer;
var
  v: integer;
  p, bestP: TNeuralFloat;
begin
  Result := 0;
  bestP := -1;
  for v := 0 to csModelVocabSize - 1 do
  begin
    p := NN.GetLastLayer().Output[Row, 0, Head * csModelVocabSize + v];
    if p > bestP then begin bestP := p; Result := v; end;
  end;
end;

// Plain greedy from the MTP model: one full re-encode forward per token, head
// 0 only (GreedyFull can't be reused -- GetClassOnPixel would argmax over the
// WHOLE NumFuture*Vocab concat depth).
procedure TDecodeBakeoff.MTPGreedyFull(NN: TNNet; var Toks: TTokenBuf;
  PromptLen, MaxNew: integer; out OutLen: integer; out Ms: double;
  out Fwds: integer);
var
  InV: TNNetVolume;
  t: integer;
  T0: double;
begin
  InV := TNNetVolume.Create(csContextLen, 1, 1);
  try
    OutLen := PromptLen;
    Fwds := 0;
    T0 := NowMs();
    while (OutLen - PromptLen < MaxNew) and (OutLen < csContextLen) do
    begin
      InV.Fill(0);
      for t := 0 to OutLen - 1 do InV.FData[t] := Toks[t];
      NN.Compute(InV);
      Inc(Fwds);
      Toks[OutLen] := MTPHeadArgmax(NN, OutLen - 1, 0);
      Inc(OutLen);
    end;
    Ms := NowMs() - T0;
  finally
    InV.Free;
  end;
end;

// SELF-SPECULATIVE greedy decode: ONE forward per pass; the same pass verifies
// the previous block's drafts AND drafts the next block from heads 1..N-1.
//   1. Forward the committed prefix PLUS the pending drafts (full re-encode).
//   2. Verify drafts left-to-right: draft j (position len+j, proposed by head
//      j+1 last pass) is accepted iff it equals head-0's argmax at row
//      len+j-1 -- exactly plain greedy's token there, because every row left
//      of the first mismatch sees only committed-or-accepted context.
//   3. First mismatch: head-0's argmax at that row IS the correct greedy
//      token, so commit it (a rejection still yields one token) and discard
//      the rest. Full acceptance: head 0 at the last draft row yields a BONUS
//      token from the same forward.
//   4. New drafts come from heads 1..N-1 at the row that produced the LAST
//      committed token -- a row whose causal context is fully committed, so
//      the proposals are well-defined.
// Every committed token is head-0's argmax over committed-only context =>
// output is token-identical to plain greedy (the phase's HARD ASSERT).
// Verified[j]/Accepted[j] count draft slot j (head j+1, distance t+2+j).
procedure TDecodeBakeoff.MTPSelfSpec(NN: TNNet; var Toks: TTokenBuf;
  PromptLen, MaxNew: integer; out OutLen: integer; out Ms: double;
  out Fwds, Passes, Committed: integer;
  var Verified, Accepted: array of integer);
var
  InV: TNNetVolume;
  len, m, j, h, lastRow, newLen, pos, g, t: integer;
  mismatch: boolean;
  T0: double;
begin
  InV := TNNetVolume.Create(csContextLen, 1, 1);
  try
    len := PromptLen;   // committed length
    m := 0;             // pending drafts at positions len..len+m-1
    Fwds := 0; Passes := 0; Committed := 0;
    T0 := NowMs();
    while (len - PromptLen < MaxNew) and (len < csContextLen) do
    begin
      // One forward over committed prefix + pending drafts.
      InV.Fill(0);
      for t := 0 to len + m - 1 do InV.FData[t] := Toks[t];
      NN.Compute(InV);
      Inc(Fwds);
      Inc(Passes);

      // Verify pending drafts left-to-right.
      j := 0;
      mismatch := false;
      while (j < m) and (not mismatch) do
      begin
        g := MTPHeadArgmax(NN, len + j - 1, 0);
        Inc(Verified[j]);
        if g = Toks[len + j] then
        begin
          Inc(Accepted[j]);
          Inc(j);
        end
        else
        begin
          Toks[len + j] := g;   // corrected token: still exact greedy
          mismatch := true;
        end;
      end;

      if mismatch then
      begin
        lastRow := len + j - 1; // row that produced the corrected token
        newLen := len + j + 1;  // j accepted drafts + 1 corrected token
      end
      else
      begin
        // All m drafts accepted; bonus token from head 0 at the last draft
        // row (with m=0 this is the plain first-pass commit).
        lastRow := len + m - 1;
        if len + m < csContextLen then
        begin
          Toks[len + m] := MTPHeadArgmax(NN, lastRow, 0);
          newLen := len + m + 1;
        end
        else
          newLen := len + m;
      end;
      Committed := Committed + (newLen - len);

      // Draft the next block from heads 1..N-1 at the last committed row.
      m := 0;
      for h := 1 to csNumFuture - 1 do
      begin
        pos := lastRow + 1 + h;     // = newLen-1+h
        if pos >= csContextLen then break;
        Toks[pos] := MTPHeadArgmax(NN, lastRow, h);
        Inc(m);
      end;
      len := newLen;
    end;
    // A fully-accepted final pass may overshoot MaxNew; trim so the exactness
    // comparison and ms/token are over exactly MaxNew tokens.
    if len > PromptLen + MaxNew then len := PromptLen + MaxNew;
    OutLen := len;
    Ms := NowMs() - T0;
  finally
    InV.Free;
  end;
end;

procedure TDecodeBakeoff.RunPhase2;
var
  Trained: TNNet;
  PlainToks, SpecToks: TTokenBuf;
  Verified, Accepted: array[0..csNumFuture - 2] of integer;
  PromptLen, PlainLen, SpecLen: integer;
  PlainMs, SpecMs: double;
  PlainFwds, SpecFwds, Passes, Committed: integer;
  SumNew, SumPlainFwds, SumSpecFwds, SumPasses, SumCommitted: integer;
  SumPlainMs, SumSpecMs: double;
  p, t, h, NewToks: integer;
  TrainSec: double;
begin
  WriteLn('=== Phase 2: MTP-heads self-speculative decode (4d95378) ===');
  LoadDataset();

  FNumFuture := csNumFuture; // switches Get*Pair + OnAfterEpoch to MTP layout
  Trained := BuildModelMTP(csContextLen);
  WriteLn('Model: ctx=', csContextLen, ' d_model=', csEmbedDim, ' blocks=',
    csBlocks, ' heads=', csHeads, ' ffn=', csFFNDim, ' vocab=',
    csModelVocabSize, ' NumFuture=', csNumFuture, '  params=',
    Trained.CountWeights());

  TrainSec := Min(csTrainCapSecMTP,
    csTotalBudgetSec - ElapsedSec() - csBenchReserveSec);
  if TrainSec < 10 then
  begin
    WriteLn('ERROR: no time left to train (data loading took too long).');
    Halt(1);
  end;
  // lr=0.001 (vs phase 1's 0.0005): the 3-head MTP forward+backward is ~2x
  // the cost of phase 1's single-head step, so fewer examples fit in the box;
  // the higher rate recovers a comparable loss drop.
  TrainTimeBoxed(Trained, TrainSec, {SamplesPerEpoch=}3200, {lr=}0.001);

  try
    for h := 0 to csNumFuture - 2 do
    begin
      Verified[h] := 0;
      Accepted[h] := 0;
    end;
    SumNew := 0; SumPlainFwds := 0; SumSpecFwds := 0;
    SumPasses := 0; SumCommitted := 0;
    SumPlainMs := 0; SumSpecMs := 0;

    WriteLn;
    WriteLn('Greedy decode showdown (', csGenTokens,
      ' tokens per prompt; both arms full re-encode, same MTP model):');
    for p := 0 to High(csPrompts) do
    begin
      PromptLen := PromptToTokens(csPrompts[p], PlainToks);
      SpecToks := PlainToks;

      MTPGreedyFull(Trained, PlainToks, PromptLen, csGenTokens,
        PlainLen, PlainMs, PlainFwds);
      MTPSelfSpec(Trained, SpecToks, PromptLen, csGenTokens,
        SpecLen, SpecMs, SpecFwds, Passes, Committed, Verified, Accepted);

      // HARD ASSERT: self-speculative output token-identical to plain greedy.
      if SpecLen <> PlainLen then
      begin
        WriteLn('ASSERT FAILED: plain arm emitted ', PlainLen - PromptLen,
          ' tokens but self-speculative arm emitted ', SpecLen - PromptLen, '.');
        Halt(1);
      end;
      for t := PromptLen to PlainLen - 1 do
        if SpecToks[t] <> PlainToks[t] then
        begin
          WriteLn('ASSERT FAILED: self-speculative diverges from plain greedy',
            ' at position ', t, ' (greedy=', PlainToks[t], ' spec=',
            SpecToks[t], ') for prompt "', csPrompts[p], '".');
          Halt(1);
        end;

      NewToks := PlainLen - PromptLen;
      Inc(SumNew, NewToks);
      Inc(SumPlainFwds, PlainFwds); Inc(SumSpecFwds, SpecFwds);
      Inc(SumPasses, Passes); Inc(SumCommitted, Committed);
      SumPlainMs := SumPlainMs + PlainMs;
      SumSpecMs := SumSpecMs + SpecMs;

      WriteLn;
      WriteLn('Prompt: "', csPrompts[p], '"  (both arms token-identical)');
      WriteLn('  text: ', csPrompts[p], ' ',
        TokensToText(PlainToks, PromptLen, NewToks));
      WriteLn(Format('  plain greedy     : %8.2f ms total  %6.2f ms/token  (%d forwards)',
        [PlainMs, PlainMs / Max(1, NewToks), PlainFwds]));
      WriteLn(Format('  self-speculative : %8.2f ms total  %6.2f ms/token  (%d forwards)',
        [SpecMs, SpecMs / Max(1, NewToks), SpecFwds]));
    end;

    WriteLn;
    WriteLn('Aggregate over ', Length(csPrompts), ' prompts (', SumNew,
      ' tokens):');
    WriteLn('  accept rate per head distance (head h forecasts t+1+h):');
    for h := 1 to csNumFuture - 1 do
      WriteLn(Format('    head %d (t+%d): %5.1f%%  (%d/%d drafts verified)',
        [h, h + 1, 100.0 * Accepted[h - 1] / Max(1, Verified[h - 1]),
         Accepted[h - 1], Verified[h - 1]]));
    WriteLn(Format('  mean committed tokens per verification pass: %.2f',
      [SumCommitted / Max(1, SumPasses)]));
    WriteLn('  arm                ms/token   target-fwd/token');
    WriteLn(Format('  plain greedy       %8.2f   %16.2f',
      [SumPlainMs / Max(1, SumNew), SumPlainFwds / Max(1, SumNew)]));
    WriteLn(Format('  self-speculative   %8.2f   %16.2f',
      [SumSpecMs / Max(1, SumNew), SumSpecFwds / Max(1, SumNew)]));
    WriteLn(Format('  speedup            : %.2fx wall clock   %.2fx forwards',
      [SumPlainMs / Max(SumSpecMs, 1e-9),
       SumPlainFwds / Max(1.0, SumSpecFwds * 1.0)]));
    WriteLn;
    WriteLn('Phase 2: PASS (self-speculative decode is token-exact). [t=',
      ElapsedSec():0:1, 's]');
  finally
    Trained.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Phase 5: speculative decoding composed with the KV cache (commit 42dd5dd).
// ---------------------------------------------------------------------------
procedure TDecodeBakeoff.RunPhase5;
var
  Target, Draft, DraftStep: TNNet;
  Step1, StepK: TStepNet;
  FullToks, CachedToks, SpecToks: TTokenBuf;
  PromptLen, FullLen, CachedLen, SpecLen: integer;
  FullMs, CachedMs, SpecMs: double;
  FullFwds, CachedFwds: integer;
  TgtFwds, DraftFwds, Passes, Proposed, Accepted: integer;
  SumNew, SumPasses, SumProposed, SumAccepted: integer;
  SumFullMs, SumCachedMs, SumSpecMs: double;
  SumFullFwds, SumCachedFwds, SumTgtFwds, SumDraftFwds: integer;
  p, t, NewToks: integer;
  DraftSec: double;
begin
  WriteLn('=== Phase 5: speculative decoding + KV cache (TruncateCache rollback) (42dd5dd) ===');
  if not FileExists(csCheckpointFile) then
  begin
    WriteLn('ERROR: checkpoint ', csCheckpointFile, ' not found.');
    WriteLn('Run --phase 1 first (it trains and saves the target model).');
    Halt(1);
  end;
  LoadDataset();

  Target := TNNet.Create();
  Target.LoadFromFile(csCheckpointFile);
  WriteLn('Loaded target from ', csCheckpointFile, '  params=',
    Target.CountWeights());

  Draft := BuildDraft(csContextLen);
  WriteLn('Draft: TokenShift, d=', csDraftEmbedDim, ' ffn=', csDraftFFNDim,
    '  params=', Draft.CountWeights());
  DraftSec := Min(csDraftCapSec,
    csTotalBudgetSec - ElapsedSec() - csBenchReserveSec);
  if DraftSec < 10 then
  begin
    WriteLn('ERROR: no time left to train the draft.');
    Halt(1);
  end;
  TrainTimeBoxed(Draft, DraftSec, {SamplesPerEpoch=}3200, {lr=}0.001);

  // Width-2 weight-copied draft twin: TokenShift+pointwise output at row 1
  // only depends on the 2 tokens fed, so proposals cost ~2/csContextLen of a
  // full-width draft forward and stay prediction-exact.
  DraftStep := BuildDraft(2);
  DraftStep.CopyWeights(Draft);

  CreateStepNet(Step1, {Width=}1, Target);          // plain-cached arm
  CreateStepNet(StepK, {Width=}csSpecK + 1, Target); // speculative verify arm
  try
    SumNew := 0; SumPasses := 0; SumProposed := 0; SumAccepted := 0;
    SumFullMs := 0; SumCachedMs := 0; SumSpecMs := 0;
    SumFullFwds := 0; SumCachedFwds := 0; SumTgtFwds := 0; SumDraftFwds := 0;
    WriteLn;
    WriteLn('Greedy decode three-way (', csGenTokens, ' tokens per prompt, K=',
      csSpecK, '):');
    for p := 0 to High(csPrompts) do
    begin
      PromptLen := PromptToTokens(csPrompts[p], FullToks);
      CachedToks := FullToks;
      SpecToks := FullToks;

      GreedyFull(Target, FullToks, PromptLen, csGenTokens,
        FullLen, FullMs, FullFwds);
      GreedyCached(Step1, CachedToks, PromptLen, csGenTokens,
        CachedLen, CachedMs, CachedFwds);
      GreedySpecCached(StepK, DraftStep, SpecToks, PromptLen, csGenTokens,
        SpecLen, SpecMs, TgtFwds, DraftFwds, Passes, Proposed, Accepted);

      // HARD ASSERTS: all three arms emit identical tokens.
      if (CachedLen <> FullLen) or (SpecLen <> FullLen) then
      begin
        WriteLn('ASSERT FAILED: lengths differ (full=', FullLen - PromptLen,
          ' cached=', CachedLen - PromptLen, ' spec=', SpecLen - PromptLen, ').');
        Halt(1);
      end;
      for t := PromptLen to FullLen - 1 do
      begin
        if CachedToks[t] <> FullToks[t] then
        begin
          WriteLn('ASSERT FAILED: plain-cached diverges from plain-full at ',
            t, ' (full=', FullToks[t], ' cached=', CachedToks[t], ').');
          Halt(1);
        end;
        if SpecToks[t] <> FullToks[t] then
        begin
          WriteLn('ASSERT FAILED: speculative diverges from plain greedy at ',
            t, ' (greedy=', FullToks[t], ' spec=', SpecToks[t], ').');
          Halt(1);
        end;
      end;

      NewToks := FullLen - PromptLen;
      Inc(SumNew, NewToks);
      Inc(SumPasses, Passes); Inc(SumProposed, Proposed);
      Inc(SumAccepted, Accepted);
      SumFullMs := SumFullMs + FullMs;
      SumCachedMs := SumCachedMs + CachedMs;
      SumSpecMs := SumSpecMs + SpecMs;
      Inc(SumFullFwds, FullFwds); Inc(SumCachedFwds, CachedFwds);
      Inc(SumTgtFwds, TgtFwds); Inc(SumDraftFwds, DraftFwds);

      WriteLn;
      WriteLn('Prompt: "', csPrompts[p], '"  (all three arms token-identical)');
      WriteLn('  text: ', csPrompts[p], ' ',
        TokensToText(FullToks, PromptLen, NewToks));
      WriteLn(Format('  accept rate %5.1f%%  (%d/%d draft tokens)  committed/pass %.2f',
        [100.0 * Accepted / Max(1, Proposed), Accepted, Proposed,
         NewToks / Max(1, Passes)]));
    end;

    WriteLn;
    WriteLn('Aggregate over ', Length(csPrompts), ' prompts (', SumNew,
      ' tokens):');
    WriteLn(Format('  accept rate          : %5.1f%%   tokens/pass: %.2f',
      [100.0 * SumAccepted / Max(1, SumProposed),
       SumNew / Max(1, SumPasses)]));
    WriteLn('  arm                    ms/token   target-fwd/token   draft-fwd/token');
    WriteLn(Format('  plain full re-encode   %8.2f   %16.2f   %15.2f',
      [SumFullMs / SumNew, SumFullFwds / SumNew, 0.0]));
    WriteLn(Format('  plain KV-cached        %8.2f   %16.2f   %15.2f',
      [SumCachedMs / SumNew, SumCachedFwds / SumNew, 0.0]));
    WriteLn(Format('  speculative KV-cached  %8.2f   %16.2f   %15.2f',
      [SumSpecMs / SumNew, SumTgtFwds / SumNew, SumDraftFwds / SumNew]));
    WriteLn(Format('  speedup vs plain-full  : %.2fx (cached)  %.2fx (spec+cached)',
      [SumFullMs / Max(SumCachedMs, 1e-9), SumFullMs / Max(SumSpecMs, 1e-9)]));
    WriteLn;
    WriteLn('Phase 5: PASS (speculative+cached decode is token-exact). [t=',
      ElapsedSec():0:1, 's]');
  finally
    FreeStepNet(StepK);
    FreeStepNet(Step1);
    DraftStep.Free;
    Draft.Free;
    Target.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Stubs for phases owned by later work. They exit 0 on purpose so the driver
// script can run all five phases today.
// ---------------------------------------------------------------------------
procedure TDecodeBakeoff.RunStub(Phase: integer);
begin
  case Phase of
    3: WriteLn('phase 3: not yet implemented ',
         '(DiagonalSSM O(1)-per-step decode variant, commit 80dd830)');
    4: WriteLn('phase 4: not yet implemented ',
         '(MLA decoupled-RoPE latent KV-cache economics, commit a8f3077)');
  end;
  // TODO(phase owner): implement here. Reuse the shared helpers --
  // LoadDataset, BuildModel, TrainTimeBoxed (raises NFit.ShouldQuit at the
  // deadline), CreateStepNet/CachedWindowForward (KV cache + RoPE offsets),
  // GreedyFull/GreedyCached (timed greedy arms), NowMs/ElapsedSec (timing).
end;

procedure TDecodeBakeoff.DoRun;
var
  Phase, i: integer;
  s: string;
begin
  NowMs(); // rebase the clock at process start
  // Mask FPU exceptions so transient early-training underflows don't abort
  // (matches sibling examples).
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 424242;

  Phase := 0;
  for i := 1 to ParamCount do
  begin
    s := ParamStr(i);
    if (s = '--phase') or (s = '-p') then
    begin
      if i < ParamCount then Phase := StrToIntDef(ParamStr(i + 1), 0);
    end;
  end;
  if (Phase < 1) or (Phase > 5) then
  begin
    WriteLn('Usage: DecodeFeaturesBakeoff --phase N   (N in 1..5)');
    WriteLn('  1: KV-cache incremental decode vs full re-encode (2dacc95)');
    WriteLn('  2: MTP-heads self-speculative decode (4d95378)');
    WriteLn('  3: DiagonalSSM O(1)-per-step decode (80dd830) [stub]');
    WriteLn('  4: MLA decoupled-RoPE latent KV cache (a8f3077) [stub]');
    WriteLn('  5: speculative decoding + KV cache rollback (42dd5dd)');
    Terminate;
    ExitCode := 2;
    exit;
  end;

  FDictionary := TNeuralTokenizer.Create();
  try
    case Phase of
      1: RunPhase1;
      2: RunPhase2;
      5: RunPhase5;
    else
      RunStub(Phase);
    end;
  finally
    FDictionary.Free;
  end;
  Terminate;
end;

var
  Application: TDecodeBakeoff;
begin
  Application := TDecodeBakeoff.Create(nil);
  Application.Title := 'Decode-efficiency features bakeoff';
  Application.Run;
  Application.Free;
end.
