program TinyGPT;
(*
Copyright (C) 2024 Joao Paulo Schwarz Schuler

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

// TinyGPT -- a char-level, decoder-only (GPT-style) transformer trained
// end-to-end on a tiny, self-contained text corpus. It builds a causal-masked
// transformer, trains it to predict the next character, streams the dropping
// loss, then autoregressively generates a short sample from a seed prompt.
//
// Everything is intentionally tiny so the demo trains WELL under 5 minutes on
// a pure-CPU machine and never exhausts memory. The corpus is a small set of
// hardcoded sentences; the honest headline of this capstone is that the model
// learns/memorizes the structure of this tiny corpus -- enough to continue a
// seed prompt into coherent corpus-style text.

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes,
  SysUtils,
  neuralnetwork,
  neuralvolume,
  neuralfit,
  neuralthread,
  neuraldatasets,
  CustApp,
  Math;

const
  // ---- Tiny GPT configuration (sized for a <5 min pure-CPU budget) ----
  csContextLen = 24;   // context window (SeqLen) in characters
  csVocabSize  = 128;  // char-level vocabulary (ASCII), one-hot encoded
  csMinSampleSize = 3; // minimum prefix length used as input
  csDModel = 64;       // residual-stream width (d_model)
  csHeads  = 4;        // attention heads (d_model must divide by Heads)
  csDFF    = 64;       // feed-forward inner width
  csBlocks = 2;        // number of stacked causal transformer blocks

type

  { TTinyGPT }

  TTinyGPT = class(TCustomApplication)
  protected
    FDataset: TStringList;
    FDatasetSize: integer;
    FNN: TNNet;
    NFit: TNeuralDataLoadingFit;
    FSampler: TNNetSamplerBase;
    procedure BuildCorpus;
    procedure DoRun; override;
  public
    procedure OnAfterEpoch(Sender: TObject);
    procedure GetTrainingPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetValidationPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetTestPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
  end;

  // A small, self-contained corpus. Each line is one training "sentence".
  // chr(1) is appended as an end-of-sequence marker (token < 2 stops sampling),
  // matching the GenerateStringFromChars convention in neuraldatasets.pas.
  procedure TTinyGPT.BuildCorpus;
  const
    csLines: array[0..7] of string = (
      'the quick brown fox jumps over the lazy dog. ',
      'the lazy dog sleeps while the quick fox runs. ',
      'a quick brown fox is faster than a lazy dog. ',
      'the dog and the fox are friends in the forest. ',
      'foxes are quick and dogs are lazy in the sun. ',
      'over the river the quick fox jumps with joy. ',
      'the brown dog runs after the quick brown fox. ',
      'in the forest the fox and the dog play all day. '
    );
  var
    I, Rep: integer;
  begin
    FDataset := TStringList.Create();
    // Repeat the corpus a few times so each epoch sees many samples and the
    // tiny model has enough gradient signal to memorize the structure.
    for Rep := 1 to 16 do
      for I := Low(csLines) to High(csLines) do
        FDataset.Add(LowerCase(csLines[I]) + chr(1));
    FDatasetSize := FDataset.Count;
    WriteLn('Built corpus with ', FDatasetSize, ' rows (',
      Length(csLines), ' unique sentences).');
  end;

  procedure TTinyGPT.DoRun;
  var
    I: integer;
  begin
    BuildCorpus();
    FNN := TNNet.Create();
    NFit := TNeuralDataLoadingFit.Create();
    // TopP sampling adds a little variety while staying close to the corpus.
    FSampler := TNNetSamplerTopP.Create(0.6);

    // ---- Decoder-only (GPT) architecture ----
    // One-hot char input -> project to d_model -> absolute positional
    // embedding -> stacked CAUSAL transformer blocks -> next-char head.
    // We use AddTransformerEncoderBlock with CausalMask=true: that is exactly a
    // GPT decoder block (causal self-attention + SwiGLU FFN, no cross-attention).
    FNN.AddLayer([
      TNNetInput.Create(csContextLen, 1, csVocabSize),
      // Token projection: one-hot (depth=VocabSize) -> d_model, per token.
      // Pointwise (1x1) conv keeps the token/sequence axis intact.
      TNNetPointwiseConvLinear.Create(csDModel),
      TNNetAddPositionalEmbedding.Create(10000)
    ]);

    for I := 1 to csBlocks do
      FNN.AddTransformerEncoderBlock(
        {Heads=}csHeads, {d_ff=}csDFF,
        {PreNorm=}true, {CausalMask=}true,
        {UseRoPE=}false, {NormClass=}nil);

    // Next-char prediction head. FullConnect flattens the whole context window
    // into a single next-token distribution (the standard LM head here).
    FNN.AddLayer([
      TNNetPointwiseConvReLU.Create(csDModel),
      TNNetFullConnectReLU.Create(csDModel),
      TNNetFullConnectLinear.Create(csVocabSize),
      TNNetSoftMax.Create()
    ]);

    DebugThreadCount();
    FNN.DebugStructure;

    WriteLn('Computing...');
    NFit.LogEveryBatches := 10;      // stream the loss (see README)
    NFit.InitialLearningRate := 0.01;
    NFit.LearningRateDecay := 0;
    NFit.Inertia := 0.9;
    NFit.L2Decay := 0;
    NFit.EnableClassComparison();
    NFit.EnableDefaultLoss();
    NFit.AvgWeightEpochCount := 1;
    NFit.OnAfterEpoch := @OnAfterEpoch;
    // Each "epoch" draws many random (prefix -> next-char) samples from the
    // tiny corpus; we use a high volume count + a few epochs so the model sees
    // enough steps to memorize the corpus inside the CPU time budget.
    NFit.FitLoading(
      FNN,
      {TrainingVolumesCount=}1024,
      {ValidationVolumesCount=}64,
      {TestVolumesCount=}64,
      {batchsize=}32,
      {epochs=}8,
      @GetTrainingPair, @GetValidationPair, @GetTestPair
    );

    WriteLn();
    WriteLn('==== Final autoregressive samples (seed -> generated) ====');
    OnAfterEpoch(Self);

    FSampler.Free;
    NFit.Free;
    FNN.Free;
    FDataset.Free;
    Terminate;
  end;

  procedure TTinyGPT.OnAfterEpoch(Sender: TObject);
  begin
    WriteLn('  "the quick" -> ',
      GenerateStringFromChars(NFit.NN, 'the quick', FSampler));
    WriteLn('  "the lazy"  -> ',
      GenerateStringFromChars(NFit.NN, 'the lazy', FSampler));
    WriteLn('  "in the for"-> ',
      GenerateStringFromChars(NFit.NN, 'in the for', FSampler));
  end;

  procedure TTinyGPT.GetTrainingPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    SampleId: integer;
    SampleLen: integer;
    SampleCutPosition: integer;
    ExpectedTokenChar: char;
    ExpectedTokenInt: integer;
  begin
    if FNN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(FNN.GetFirstLayer().Output);
    if FNN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(FNN.GetLastLayer().Output);
    SampleId := Random(FDatasetSize);
    SampleLen := Min(Length(FDataset[SampleId]), pInput.SizeX);
    SampleCutPosition := Random(SampleLen-csMinSampleSize)+csMinSampleSize;
    // The expected token is the next character in the string.
    ExpectedTokenChar := FDataset[SampleId][SampleCutPosition+1];
    ExpectedTokenInt := Min(Ord(ExpectedTokenChar),pInput.Depth-1);
    // One-hot encode the prefix; target is the next char.
    pInput.OneHotEncodingReversed(copy(FDataset[SampleId], 1, SampleCutPosition));
    pOutput.SetClassForSoftMax(ExpectedTokenInt);
    pOutput.Tag := ExpectedTokenInt;
  end;

  procedure TTinyGPT.GetValidationPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    SampleId: integer;
    SampleLen: integer;
    SampleCutPosition: integer;
    ExpectedTokenChar: char;
    ExpectedTokenInt: integer;
  begin
    if FNN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(FNN.GetFirstLayer().Output);
    if FNN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(FNN.GetLastLayer().Output);
    SampleId := Idx mod FDatasetSize;
    SampleLen := Min(Length(FDataset[SampleId]), pInput.SizeX);
    SampleCutPosition := (Idx mod (1+SampleLen-csMinSampleSize))+csMinSampleSize-1;
    ExpectedTokenChar := FDataset[SampleId][SampleCutPosition+1];
    ExpectedTokenInt := Min(Ord(ExpectedTokenChar),pInput.Depth-1);
    pInput.OneHotEncodingReversed(copy(FDataset[SampleId], 1, SampleCutPosition));
    pOutput.SetClassForSoftMax(ExpectedTokenInt);
    pOutput.Tag := ExpectedTokenInt;
  end;

  procedure TTinyGPT.GetTestPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    GetValidationPair(Idx, ThreadId, pInput, pOutput);
  end;

var
  Application: TTinyGPT;
begin
  Application := TTinyGPT.Create(nil);
  Application.Title:='TinyGPT char-level transformer';
  Application.Run;
  Application.Free;
end.
