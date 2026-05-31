program TinyTransformerFFN;
(*
TinyTransformerFFN: the feed-forward HALF of a transformer block, trained
end-to-end on a tiny pure-CPU synthetic per-token task. There is NO
multi-head attention here on purpose: this demo isolates the FFN sub-block
(the part of a transformer layer that mixes FEATURES within each token, not
information ACROSS tokens) and shows that a deep pre-norm residual stack of
those sub-blocks composes and trains stably.

Each transformer FFN sub-block is the LLaMA-style pre-norm residual:

    y = x + SwiGLU_FFN( RMSNorm(x) )

built from two existing library builders:
  - TNNet.AddRMSNormResidual([...])        { y = x + Sublayer(RMSNorm(x)) }
  - TNNet.AddSwiGLUFeedForward(D,H,D)       { Dense(2H) -> SwiGLU -> Dense(D) }

SHAPE CONTRACT (the gotcha this demo is careful about):
The residual RMSNorm + the residual TNNetSum operate on the DEPTH axis, so the
block-carrying tensor is 1 x 1 x WIDTH (features in Depth). But
AddSwiGLUFeedForward ends in a TNNetFullConnectLinear, which lays its output
along the X axis (shape WIDTH x 1 x 1). So we append a TNNetReshape(1,1,WIDTH)
INSIDE the residual sublayer list to move the feature vector back into Depth
before the residual Sum. (Without it you get "Size doesn't match ...
Should be:(1 1 W) It is:(W 1 1)".)

TASK (per-token denoising; no sequence mixing needed, so an FFN suffices):
Each token is a feature vector v in R^WIDTH. The CLEAN target for that token is
a fixed deterministic nonlinear elementwise map of a few of its own
coordinates. The network sees a NOISED copy of v and must reconstruct the
clean target. Because every token's target depends only on that token's own
features, a per-position FFN can solve it with no attention.

Headline outputs: param count, per-epoch train-loss trace, final train/val
MSE, and a small before/after (noised input vs predicted vs clean) sample.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralfit;

const
  WIDTH      = 16;   // d_model: per-token feature dim (lives in the Depth axis)
  HIDDEN     = 32;   // SwiGLU hidden dim
  NUM_BLOCKS = 4;    // depth of the FFN-only residual stack
  TRAIN_SIZE = 2000;
  VAL_SIZE   = 400;
  NUM_EPOCHS = 60;
  BATCH_SIZE = 32;
  NOISE_STD  = 0.15;
  SEED       = 42;

type
  TLossTracker = class
  public
    Trace: array[0..NUM_EPOCHS - 1] of TNeuralFloat;
    TraceCount: integer;
    procedure Reset;
    procedure OnAfterEpoch(Sender: TObject);
  end;

var
  GTracker: TLossTracker;

procedure TLossTracker.Reset;
begin
  TraceCount := 0;
end;

procedure TLossTracker.OnAfterEpoch(Sender: TObject);
var
  Fit: TNeuralFit;
begin
  Fit := Sender as TNeuralFit;
  if TraceCount <= High(Trace) then
  begin
    Trace[TraceCount] := Fit.CurrentTrainingError;
    Inc(TraceCount);
  end;
end;

function RandUniform: TNeuralFloat;
begin
  Result := (Random * 2.0) - 1.0;
end;

// Box-Muller-ish cheap Gaussian noise (two uniforms averaged is enough here;
// we just want a mild perturbation the FFN learns to denoise away).
function RandNoise: TNeuralFloat;
begin
  Result := NOISE_STD * (RandUniform + RandUniform) * 0.5 * 2.0;
end;

// CLEAN per-token target: a fixed deterministic nonlinear elementwise map of
// the token's own coordinates. Each output channel is a smooth nonlinearity of
// one or two input channels, so the FFN must learn a real (but learnable)
// per-feature transform - not the identity.
procedure CleanTarget(const Inp: TNNetVolume; const Outp: TNNetVolume);
var
  d: integer;
  a, b: TNeuralFloat;
begin
  for d := 0 to WIDTH - 1 do
  begin
    a := Inp.FData[d];
    b := Inp.FData[(d + 1) mod WIDTH];
    // Smooth, well-scaled (~[-1,1]) per-channel nonlinearity of two of the
    // token's own coordinates. An FFN can fit this; the residual stack just
    // needs to denoise the perturbed input back onto this clean manifold.
    Outp.FData[d] := 0.6 * Tanh(1.5 * a) + 0.4 * a * b;
  end;
end;

// Builds one (clean-feature -> noised-input, clean-target) training pair.
// Both volumes are shaped 1 x 1 x WIDTH so the feature vector lives in Depth,
// exactly what RMSNorm / the residual Sum expect.
function CreatePair: TNNetVolumePair;
var
  Clean, Noised, Target: TNNetVolume;
  d: integer;
begin
  Clean  := TNNetVolume.Create(1, 1, WIDTH);
  Noised := TNNetVolume.Create(1, 1, WIDTH);
  Target := TNNetVolume.Create(1, 1, WIDTH);
  for d := 0 to WIDTH - 1 do
    Clean.FData[d] := RandUniform;
  CleanTarget(Clean, Target);
  for d := 0 to WIDTH - 1 do
    Noised.FData[d] := Clean.FData[d] + RandNoise;
  Clean.Free;
  Result := TNNetVolumePair.Create(Noised, Target); // pair owns Noised+Target
end;

function CreatePairList(MaxCnt: integer): TNNetVolumePairList;
var
  Cnt: integer;
begin
  Result := TNNetVolumePairList.Create();
  for Cnt := 1 to MaxCnt do
    Result.Add(CreatePair);
end;

// A "hit" = mean abs error over the WIDTH channels under 0.1.
function LocalCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
var
  d: integer;
  s: TNeuralFloat;
begin
  s := 0;
  for d := 0 to WIDTH - 1 do
    s := s + Abs(A.FData[d] - B.FData[d]);
  Result := (s / WIDTH) < 0.1;
end;

// Builds the FFN-only transformer stack. The carried tensor is 1 x 1 x WIDTH
// throughout. Each block is a pre-norm SwiGLU FFN residual; the trailing
// reshape moves the SwiGLU's FullConnect output (W x 1 x 1) back into Depth so
// the residual Sum is shape-valid.
procedure BuildModel(NN: TNNet);
var
  i: integer;
begin
  NN.AddLayer( TNNetInput.Create(1, 1, WIDTH) );
  for i := 1 to NUM_BLOCKS do
    NN.AddRMSNormResidual([
      TNNetFullConnectLinear.Create(1, 1, 2 * HIDDEN), // gate || value on depth
      TNNetSwiGLU.Create(),                            // value * SiLU(gate)
      TNNetFullConnectLinear.Create(WIDTH),            // back to d_model (on X)
      TNNetReshape.Create(1, 1, WIDTH)                 // -> Depth for the Sum
    ]);
  // Per-token output head: shape-preserving projection over Depth.
  NN.AddLayer( TNNetPointwiseConvLinear.Create(WIDTH) );
end;

function EvaluateMSE(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I, d: integer;
  SumSq: Double;
  Outp: TNNetVolume;
begin
  SumSq := 0;
  Outp := TNNetVolume.Create(1, 1, WIDTH);
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    NN.GetOutput(Outp);
    for d := 0 to WIDTH - 1 do
      SumSq := SumSq + Sqr(Outp.FData[d] - Pairs[I].O.FData[d]);
  end;
  Outp.Free;
  if Pairs.Count > 0 then
    Result := SumSq / (Pairs.Count * WIDTH)
  else
    Result := 0;
end;

function SafeF(V: TNeuralFloat; Width, Decimals: integer): string;
begin
  if IsNan(V) then Result := 'NaN'
  else if IsInfinite(V) then Result := 'Inf'
  else Result := FloatToStrF(V, ffFixed, Width, Decimals);
end;

procedure RunAlgo();
var
  NN: TNNet;
  NFit: TNeuralFit;
  TrainingPairs, ValidationPairs: TNNetVolumePairList;
  Outp: TNNetVolume;
  i, d: integer;
  StartTime, EndTime: TDateTime;
  BaselineMSE, FinalVal: TNeuralFloat;
  SumSq: Double;
begin
  WriteLn('TinyTransformerFFN: FFN-only (no-attention) transformer half-block.');
  WriteLn('Stack: ', NUM_BLOCKS, ' x [ x + SwiGLU_FFN(RMSNorm(x)) ],  d_model=',
          WIDTH, ', hidden=', HIDDEN, '.');
  WriteLn('Task: per-token denoising of a nonlinear elementwise target ',
          '(noise std=', SafeF(NOISE_STD, 6, 3), ').');
  WriteLn;

  RandSeed := SEED;
  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  TrainingPairs   := CreatePairList(TRAIN_SIZE);
  ValidationPairs := CreatePairList(VAL_SIZE);
  try
    BuildModel(NN);

    WriteLn('Network structure:');
    NN.DebugStructure();
    WriteLn('Trainable parameter count: ', NN.CountWeights());
    WriteLn;

    // Baseline: MSE if the model did nothing and just echoed the noised input.
    SumSq := 0;
    for i := 0 to ValidationPairs.Count - 1 do
      for d := 0 to WIDTH - 1 do
        SumSq := SumSq + Sqr(ValidationPairs[i].I.FData[d] - ValidationPairs[i].O.FData[d]);
    BaselineMSE := SumSq / (ValidationPairs.Count * WIDTH);
    WriteLn('Baseline val MSE (echo noised input, no learning): ',
            SafeF(BaselineMSE, 12, 6));
    WriteLn;

    GTracker.Reset;
    NFit.FileNameBase := GetTempDir + 'TinyTransformerFFN_autosave';
    NFit.InitialLearningRate := 0.005;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.MaxThreadNum := 1; // determinism on this tiny task
    NFit.HideMessages();
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.InferHitFn := @LocalCompare;
    NFit.OnAfterEpoch := @GTracker.OnAfterEpoch;

    WriteLn('Training ', NUM_EPOCHS, ' epochs (LR=0.005, batch=', BATCH_SIZE, ')...');
    StartTime := Now;
    NFit.Fit(NN, TrainingPairs, ValidationPairs, nil, BATCH_SIZE, NUM_EPOCHS);
    EndTime := Now;

    WriteLn;
    WriteLn('Train-loss trace (every 5th epoch + final):');
    i := 0;
    while i < GTracker.TraceCount do
    begin
      WriteLn('  epoch ', (i + 1):3, ': train_err = ', SafeF(GTracker.Trace[i], 12, 6));
      Inc(i, 5);
    end;
    if (GTracker.TraceCount > 0) and (((GTracker.TraceCount - 1) mod 5) <> 0) then
      WriteLn('  epoch ', GTracker.TraceCount:3, ': train_err = ',
              SafeF(GTracker.Trace[GTracker.TraceCount - 1], 12, 6));
    WriteLn;

    FinalVal := EvaluateMSE(NN, ValidationPairs);
    WriteLn('Final validation MSE (clean target): ', SafeF(FinalVal, 12, 6));
    WriteLn('  vs baseline echo MSE             : ', SafeF(BaselineMSE, 12, 6),
            '   (', SafeF(BaselineMSE / Max(FinalVal, 1e-9), 8, 2), 'x better)');
    WriteLn;

    // Before/after sample on the first val token (first 6 channels).
    Outp := TNNetVolume.Create(1, 1, WIDTH);
    NN.Compute(ValidationPairs[0].I);
    NN.GetOutput(Outp);
    WriteLn('Sample token (first 6 of ', WIDTH, ' channels):');
    WriteLn('  ch | noised_in  predicted   clean_tgt');
    for d := 0 to 5 do
      WriteLn('  ', d:2, ' | ',
              ValidationPairs[0].I.FData[d]:9:4, '  ',
              Outp.FData[d]:9:4, '  ',
              ValidationPairs[0].O.FData[d]:9:4);
    Outp.Free;

    WriteLn;
    WriteLn('Wall time: ', FormatFloat('0.00', (EndTime - StartTime) * 86400), ' s');
  finally
    ValidationPairs.Free;
    TrainingPairs.Free;
    NFit.Free;
    NN.Free;
  end;
end;

var
  Application: record Title: string; end;

begin
  Application.Title := 'Tiny Transformer FFN Example';
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  GTracker := TLossTracker.Create;
  try
    RunAlgo();
  finally
    GTracker.Free;
  end;
end.
