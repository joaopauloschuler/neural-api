program PreNormVsPostNorm;
(*
PreNormVsPostNorm: compares the SAME deepish residual stack wired three
different ways on the toy hypotenuse task: y = sqrt(X^2 + Y^2).

The only variable is the residual builder used for every block:
  - TNNet.AddPreNormResidual  -> y = x + Sublayer(LayerNorm(x))
  - TNNet.AddRMSNormResidual  -> y = x + Sublayer(RMSNorm(x))   (LLaMA-style)
  - TNNet.AddPostNormResidual -> y = LayerNorm(Sublayer(x) + x)

Each arm uses the SAME architecture (a stack of NUM_BLOCKS residual blocks of
a small fixed width), the SAME synthetic data, the SAME seed, LR and epochs.
The well-known PreNorm-vs-PostNorm training-stability gap shows up as deeper
post-norm stacks training more slowly / less stably than the pre-norm variants.

For each arm we print a short loss-vs-epoch trace plus a final comparison
table (final training loss, final validation MSE in original hypotenuse
units), and a one-line verdict. Printing is guarded against NaN / Inf so a
diverging arm reports cleanly instead of crashing.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralfit;

type
  TNormKind = (nkPreNorm, nkRMSNorm, nkPostNorm);

const
  // Inputs are normalized to [0,1] before training (X/100, Y/100, hypot/200).
  INPUT_SCALE  = 100.0;
  TARGET_SCALE = 200.0;
  TRAIN_SIZE   = 800;
  VAL_SIZE     = 200;
  NUM_BLOCKS   = 12;   // deepish residual stack
  WIDTH        = 24;   // small fixed residual width (lives in the Depth axis)
  NUM_EPOCHS   = 30;
  BATCH_SIZE   = 32;
  SEED         = 42;

type
  TArmResult = record
    Name: string;
    FinalTrainLoss: TNeuralFloat;      // last-epoch training error reported by fit
    FinalValLoss: TNeuralFloat;        // validation MSE in original target units
    Trace: array[0..NUM_EPOCHS - 1] of TNeuralFloat; // per-epoch training error
    TraceCount: integer;
    Diverged: boolean;
  end;

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

function CreateHypotenusePairList(MaxCnt: integer): TNNetVolumePairList;
var
  Cnt: integer;
  LocalX, LocalY, Hypotenuse: TNeuralFloat;
begin
  Result := TNNetVolumePairList.Create();
  for Cnt := 1 to MaxCnt do
  begin
    LocalX := Random(100);
    LocalY := Random(100);
    Hypotenuse := sqrt(LocalX*LocalX + LocalY*LocalY);

    Result.Add(
      TNNetVolumePair.Create(
        TNNetVolume.Create([LocalX / INPUT_SCALE, LocalY / INPUT_SCALE]),
        TNNetVolume.Create([Hypotenuse / TARGET_SCALE])
      )
    );
  end;
end;

function LocalFloatCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
begin
  Result := ( Abs(A.FData[0]-B.FData[0])<0.005 );
end;

function NormName(Kind: TNormKind): string;
begin
  case Kind of
    nkPreNorm:  Result := 'PreNorm  (AddPreNormResidual)';
    nkRMSNorm:  Result := 'RMSNorm  (AddRMSNormResidual)';
    nkPostNorm: Result := 'PostNorm (AddPostNormResidual)';
  end;
end;

// Builds the SAME deepish residual stack, wired with the requested builder.
// The residual-carrying tensor is 1 x 1 x WIDTH (feature dim in Depth), which
// is exactly what TNNetPointwiseConvLinear + LayerNorm/RMSNorm + Sum expect.
procedure BuildArm(NN: TNNet; Kind: TNormKind);
var
  i: integer;
begin
  NN.AddLayer( TNNetInput.Create(2) );
  NN.AddLayer( TNNetFullConnectLinear.Create(WIDTH) ); // project to WIDTH features
  // FullConnectLinear lays WIDTH out along X (shape WIDTH x 1 x 1). The residual
  // builders normalise/convolve along the Depth axis (TNNetPointwiseConvLinear),
  // so reshape the feature vector into Depth: shape 1 x 1 x WIDTH.
  NN.AddLayer( TNNetReshape.Create(1, 1, WIDTH) );
  for i := 1 to NUM_BLOCKS do
  begin
    case Kind of
      nkPreNorm:  NN.AddPreNormResidual ([ TNNetPointwiseConvLinear.Create(WIDTH), TNNetReLU.Create() ]);
      nkRMSNorm:  NN.AddRMSNormResidual ([ TNNetPointwiseConvLinear.Create(WIDTH), TNNetReLU.Create() ]);
      nkPostNorm: NN.AddPostNormResidual([ TNNetPointwiseConvLinear.Create(WIDTH), TNNetReLU.Create() ]);
    end;
  end;
  NN.AddLayer( TNNetFullConnectLinear.Create(1) ); // regression head
end;

function EvaluateMSE(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I: integer;
  Pred: TNeuralFloat;
  SumSq: Double;
begin
  SumSq := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    Pred := NN.GetLastLayer().Output.FData[0];
    SumSq := SumSq + Sqr((Pred - Pairs[I].O.FData[0]) * TARGET_SCALE);
  end;
  if Pairs.Count > 0 then
    Result := SumSq / Pairs.Count
  else
    Result := 0;
end;

function RunOne(Kind: TNormKind;
                Train, Validation: TNNetVolumePairList): TArmResult;
var
  NN: TNNet;
  NFit: TNeuralFit;
  i: integer;
begin
  Result.Name := NormName(Kind);
  GTracker.Reset;

  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    BuildArm(NN, Kind);

    NFit.FileNameBase := GetTempDir + 'PreNormVsPostNorm_autosave';
    NFit.InitialLearningRate := 0.01;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HideMessages(); // keep stdout to our own trace / table only
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.InferHitFn := @LocalFloatCompare;
    NFit.OnAfterEpoch := @GTracker.OnAfterEpoch;
    NFit.Fit(NN, Train, Validation, nil, BATCH_SIZE, NUM_EPOCHS);

    Result.TraceCount := GTracker.TraceCount;
    for i := 0 to GTracker.TraceCount - 1 do
      Result.Trace[i] := GTracker.Trace[i];

    Result.FinalValLoss := EvaluateMSE(NN, Validation);
    if Result.TraceCount > 0 then
      Result.FinalTrainLoss := Result.Trace[Result.TraceCount - 1]
    else
      Result.FinalTrainLoss := NaN;
    Result.Diverged := IsNan(Result.FinalValLoss) or IsInfinite(Result.FinalValLoss) or
                       IsNan(Result.FinalTrainLoss) or IsInfinite(Result.FinalTrainLoss);
  finally
    NFit.Free;
    NN.Free;
  end;
end;

// Safe float-to-string that never crashes on NaN / Inf.
function SafeF(V: TNeuralFloat; Width, Decimals: integer): string;
begin
  if IsNan(V) then
    Result := 'NaN'
  else if IsInfinite(V) then
    Result := 'Inf'
  else
    Result := FloatToStrF(V, ffFixed, Width, Decimals);
end;

procedure PrintTrace(const R: TArmResult);
var
  i, step: integer;
begin
  WriteLn('  ', R.Name, ' loss-vs-epoch trace:');
  // Print at most ~9 sample points so the trace stays compact.
  step := 5;
  i := 0;
  while i < R.TraceCount do
  begin
    WriteLn('    epoch ', (i + 1):3, ': train_err = ', SafeF(R.Trace[i], 12, 6));
    Inc(i, step);
  end;
  // Always show the final epoch.
  if (R.TraceCount > 0) and (((R.TraceCount - 1) mod step) <> 0) then
    WriteLn('    epoch ', R.TraceCount:3, ': train_err = ', SafeF(R.Trace[R.TraceCount - 1], 12, 6));
end;

procedure RunAlgo();
var
  TrainingPairs, ValidationPairs: TNNetVolumePairList;
  Kind: TNormKind;
  Results: array[TNormKind] of TArmResult;
  StartTime, EndTime: TDateTime;
  BestKind: TNormKind;
  BestVal: TNeuralFloat;
  AnyFinite: boolean;
begin
  WriteLn('PreNorm vs RMSNorm vs PostNorm residual-stack bake-off (hypotenuse toy task).');
  WriteLn('Same arch wired three ways: ', NUM_BLOCKS, ' residual blocks of width ', WIDTH, '.');
  WriteLn('Block sublayer = [PointwiseConvLinear(', WIDTH, '), ReLU]; ',
          NUM_EPOCHS, ' epochs, ', TRAIN_SIZE, ' train pairs, LR=0.01, RandSeed=', SEED, '.');
  WriteLn;

  StartTime := Now;
  for Kind := Low(TNormKind) to High(TNormKind) do
  begin
    // Reseed before generating data AND before each fit so every arm sees the
    // same data and the same weight initialization.
    RandSeed := SEED;
    TrainingPairs   := CreateHypotenusePairList(TRAIN_SIZE);
    ValidationPairs := CreateHypotenusePairList(VAL_SIZE);
    try
      RandSeed := SEED;
      Write('Training ', NormName(Kind), ' ...');
      Results[Kind] := RunOne(Kind, TrainingPairs, ValidationPairs);
      WriteLn(' done.');
    finally
      ValidationPairs.Free;
      TrainingPairs.Free;
    end;
  end;
  EndTime := Now;

  WriteLn;
  WriteLn('=== Loss-vs-epoch traces ===');
  for Kind := Low(TNormKind) to High(TNormKind) do
  begin
    PrintTrace(Results[Kind]);
    WriteLn;
  end;

  WriteLn('=== Results (CSV) ===');
  WriteLn('builder,final_train_loss,final_val_mse,diverged');
  for Kind := Low(TNormKind) to High(TNormKind) do
    WriteLn(Results[Kind].Name, ',',
            SafeF(Results[Kind].FinalTrainLoss, 12, 6), ',',
            SafeF(Results[Kind].FinalValLoss, 12, 4), ',',
            BoolToStr(Results[Kind].Diverged, 'YES', 'no'));
  WriteLn;

  // Verdict: lowest finite validation MSE wins.
  AnyFinite := false;
  BestKind := Low(TNormKind);
  BestVal := Infinity;
  for Kind := Low(TNormKind) to High(TNormKind) do
    if (not Results[Kind].Diverged) and (Results[Kind].FinalValLoss < BestVal) then
    begin
      BestVal := Results[Kind].FinalValLoss;
      BestKind := Kind;
      AnyFinite := true;
    end;

  if AnyFinite then
    WriteLn('Verdict: ', NormName(BestKind),
            ' converged lowest/most stably (val MSE = ', SafeF(BestVal, 12, 4), ').')
  else
    WriteLn('Verdict: every arm diverged at this LR/depth.');

  WriteLn;
  WriteLn('Total wall time: ', FormatFloat('0.00', (EndTime - StartTime)*86400), ' s');
end;

var
  Application: record Title: string; end;

begin
  Application.Title := 'PreNorm vs PostNorm';
  // A diverging (post-norm) arm can produce NaN / Inf. Mask the FPU exceptions
  // so those propagate as float VALUES we can detect and report cleanly,
  // instead of raising EInvalidOp and crashing the whole bake-off.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := SEED;
  GTracker := TLossTracker.Create;
  try
    RunAlgo();
  finally
    GTracker.Free;
  end;
end.
