program ReZeroVsGatedResidual;
(*
ReZeroVsGatedResidual: trains the SAME deepish residual MLP on the toy
hypotenuse task (y = sqrt(X^2 + Y^2)), wired two ways that differ ONLY in the
residual gate placed on each block's branch:

  - Arm A "ReZero":         y = x + alpha       * Sublayer(x)   (one SCALAR alpha
                                                                 per block)
  - Arm B "GatedResidual":  y = x + alpha[d]    * Sublayer(x)   (one alpha PER
                                                                 CHANNEL/Depth)

Both gates initialise to 0.0, so every block starts as the identity and the
gate "opens" during training. TNNetReZero (in this repo) holds a SINGLE
learnable scalar (SetNumWeightsForAllNeurons(1,1,1)); TNNetGatedResidual is its
per-channel generalisation (SetNumWeightsForAllNeurons(1,1,Depth)). Arm B is a
straight builder swap (TNNet.AddGatedResidual); Arm A is wired manually,
mirroring AddGatedResidual but substituting TNNetReZero for the gate:
  BranchInput := GetLastLayer();
  AddLayer(Sublayer); AddLayer(TNNetReZero.Create());
  AddLayer(TNNetSum.Create([GetLastLayer(), BranchInput])).

HEADLINE: after training we dump the learned gate value(s) for every residual
block. For ReZero that is the single scalar; for GatedResidual it is the
per-channel vector, shown as an ASCII bar chart plus min/mean/max, so the reader
can SEE whether the per-channel gate opens UNEVENLY across channels (some
channels grow, others stay near 0) versus the single ReZero scalar. We also
print final training loss + validation MSE for both arms so convergence is
comparable. All printing is guarded against NaN / Inf.

The gate weights live in Layer.Neurons[0].Weights (a TNNetVolume): one element
for ReZero (.Raw[0]); Depth elements for GatedResidual (.Raw[d]).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralfit;

type
  TGateKind = (gkReZero, gkGated);

const
  // Inputs are normalized to [0,1] before training (X/100, Y/100, hypot/200).
  INPUT_SCALE  = 100.0;
  TARGET_SCALE = 200.0;
  TRAIN_SIZE   = 800;
  VAL_SIZE     = 200;
  NUM_BLOCKS   = 12;   // deepish residual stack (same budget as PreNormVsPostNorm)
  WIDTH        = 24;   // small fixed residual width (lives in the Depth axis)
  NUM_EPOCHS   = 30;
  BATCH_SIZE   = 32;
  SEED         = 42;

type
  TArmResult = record
    Name: string;
    FinalTrainLoss: TNeuralFloat;      // last-epoch training error reported by fit
    FinalValLoss: TNeuralFloat;        // validation MSE in original target units
    // Per-block gate LAYER INDICES into NN.Layers. We store indices (not layer
    // refs) because TNeuralFit.Fit reloads the best model at the end via
    // FNN.LoadFromFile, which rebuilds every layer instance -> any captured
    // layer reference would be stale. Indices stay valid (same structure).
    GateIdx: array[0..NUM_BLOCKS - 1] of integer;
    Diverged: boolean;
  end;

  TLossTracker = class
  public
    LastError: TNeuralFloat;
    procedure Reset;
    procedure OnAfterEpoch(Sender: TObject);
  end;

var
  GTracker: TLossTracker;
  // Gate-layer indices captured during the build of the current arm.
  GGateIdx: array[0..NUM_BLOCKS - 1] of integer;

procedure TLossTracker.Reset;
begin
  LastError := NaN;
end;

procedure TLossTracker.OnAfterEpoch(Sender: TObject);
begin
  LastError := (Sender as TNeuralFit).CurrentTrainingError;
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

function GateName(Kind: TGateKind): string;
begin
  case Kind of
    gkReZero: Result := 'ReZero        (scalar gate, manual x + alpha*Sub(x))';
    gkGated:  Result := 'GatedResidual (per-channel gate, AddGatedResidual)';
  end;
end;

// Builds the SAME deepish residual stack, wired with the requested gate. The
// residual-carrying tensor is 1 x 1 x WIDTH (feature dim in Depth), exactly
// what TNNetPointwiseConvLinear + the gate + Sum expect (a residual sublayer
// MUST be shape-preserving; PointwiseConvLinear over Depth preserves shape,
// FullConnectLinear would not). Captures the gate layer of each block into
// GGateLayers so its trained weights can be dumped afterwards.
procedure BuildArm(NN: TNNet; Kind: TGateKind);
var
  i: integer;
  BranchInput: TNNetLayer;
begin
  NN.AddLayer( TNNetInput.Create(2) );
  NN.AddLayer( TNNetFullConnectLinear.Create(WIDTH) ); // project to WIDTH features
  // FullConnectLinear lays WIDTH out along X (shape WIDTH x 1 x 1). The gate /
  // sum operate along the Depth axis, so reshape the feature vector into Depth:
  // shape 1 x 1 x WIDTH.
  NN.AddLayer( TNNetReshape.Create(1, 1, WIDTH) );
  for i := 1 to NUM_BLOCKS do
  begin
    case Kind of
      gkGated:
        begin
          // Builder swap: y = x + GatedResidual(Sublayer(x)).
          NN.AddGatedResidual([ TNNetPointwiseConvLinear.Create(WIDTH), TNNetReLU.Create() ]);
          // The gate is the layer just before the closing TNNetSum.
          GGateIdx[i - 1] := NN.GetLastLayer().LayerIdx - 1;
        end;
      gkReZero:
        begin
          // Manual ReZero residual mirroring AddGatedResidual but with a scalar
          // gate: y = x + alpha * Sublayer(x).
          BranchInput := NN.GetLastLayer();
          NN.AddLayer( TNNetPointwiseConvLinear.Create(WIDTH) );
          NN.AddLayer( TNNetReLU.Create() );
          GGateIdx[i - 1] := NN.AddLayer( TNNetReZero.Create() ).LayerIdx;
          NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        end;
    end;
  end;
  NN.AddLayer( TNNetFullConnectLinear.Create(1) ); // regression head
end;

procedure DumpGates(NN: TNNet; const R: TArmResult; Kind: TGateKind); forward;

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

function RunOne(Kind: TGateKind;
                Train, Validation: TNNetVolumePairList): TArmResult;
var
  NN: TNNet;
  NFit: TNeuralFit;
  i: integer;
begin
  Result.Name := GateName(Kind);
  GTracker.Reset;
  for i := 0 to NUM_BLOCKS - 1 do GGateIdx[i] := -1;

  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    BuildArm(NN, Kind);

    NFit.FileNameBase := GetTempDir + 'ReZeroVsGatedResidual_autosave';
    NFit.InitialLearningRate := 0.01;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HideMessages(); // keep stdout to our own dump / table only
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.InferHitFn := @LocalFloatCompare;
    NFit.OnAfterEpoch := @GTracker.OnAfterEpoch;
    NFit.Fit(NN, Train, Validation, nil, BATCH_SIZE, NUM_EPOCHS);

    Result.FinalTrainLoss := GTracker.LastError;
    Result.FinalValLoss := EvaluateMSE(NN, Validation);
    // Snapshot the gate-layer indices so weights can be read off the (possibly
    // reloaded) trained net afterwards.
    for i := 0 to NUM_BLOCKS - 1 do
      Result.GateIdx[i] := GGateIdx[i];
    Result.Diverged := IsNan(Result.FinalValLoss) or IsInfinite(Result.FinalValLoss) or
                       IsNan(Result.FinalTrainLoss) or IsInfinite(Result.FinalTrainLoss);
    // Dump must happen before NN.Free, so do it here while the net is alive.
    DumpGates(NN, Result, Kind);
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

// One row of an ASCII bar chart for a gate magnitude, scaled to Scale (the max
// |gate| in the block, so the largest channel fills the bar). Guards NaN/Inf.
function GateBar(V, Scale: TNeuralFloat; BarLen: integer): string;
var
  n, k: integer;
begin
  if IsNan(V) or IsInfinite(V) or IsNan(Scale) or IsInfinite(Scale) or (Scale <= 0) then
  begin
    Result := '';
    Exit;
  end;
  n := Round( (Abs(V) / Scale) * BarLen );
  if n < 0 then n := 0;
  if n > BarLen then n := BarLen;
  Result := '';
  for k := 1 to n do Result := Result + '#';
end;

procedure RunAlgo();
var
  TrainingPairs, ValidationPairs: TNNetVolumePairList;
  Kind: TGateKind;
  Results: array[TGateKind] of TArmResult;
  StartTime, EndTime: TDateTime;
begin
  WriteLn('ReZero vs GatedResidual depth ablation (hypotenuse toy task).');
  WriteLn('Same arch wired two ways: ', NUM_BLOCKS, ' residual blocks of width ', WIDTH, '.');
  WriteLn('Block sublayer = [PointwiseConvLinear(', WIDTH, '), ReLU]; ',
          NUM_EPOCHS, ' epochs, ', TRAIN_SIZE, ' train pairs, LR=0.01, RandSeed=', SEED, '.');
  WriteLn('Only difference: ReZero = ONE scalar gate per block; ',
          'GatedResidual = one gate per channel (Depth=', WIDTH, ').');
  WriteLn('Both gates init 0.0 => every block starts as identity; gates open while training.');
  WriteLn;

  StartTime := Now;
  for Kind := Low(TGateKind) to High(TGateKind) do
  begin
    // Reseed before generating data AND before each fit so every arm sees the
    // same data and the same weight initialization.
    RandSeed := SEED;
    TrainingPairs   := CreateHypotenusePairList(TRAIN_SIZE);
    ValidationPairs := CreateHypotenusePairList(VAL_SIZE);
    try
      RandSeed := SEED;
      Write('Training ', GateName(Kind), ' ...');
      Results[Kind] := RunOne(Kind, TrainingPairs, ValidationPairs);
      WriteLn(' done.');
    finally
      ValidationPairs.Free;
      TrainingPairs.Free;
    end;
  end;
  EndTime := Now;

  WriteLn;
  WriteLn('=== Convergence (CSV) ===');
  WriteLn('arm,final_train_loss,final_val_mse,diverged');
  for Kind := Low(TGateKind) to High(TGateKind) do
    WriteLn(Results[Kind].Name, ',',
            SafeF(Results[Kind].FinalTrainLoss, 12, 6), ',',
            SafeF(Results[Kind].FinalValLoss, 12, 4), ',',
            BoolToStr(Results[Kind].Diverged, 'YES', 'no'));
  WriteLn;

  WriteLn('Total wall time: ', FormatFloat('0.00', (EndTime - StartTime)*86400), ' s');
end;

// Reads the trained gate weights off each block and prints them. For ReZero
// the gate is a single scalar (Neurons[0].Weights.Raw[0]); for GatedResidual it
// is a Depth-length per-channel vector (Neurons[0].Weights.Raw[d]) printed as an
// ASCII bar chart plus min/mean/max so uneven opening is visible.
procedure DumpGates(NN: TNNet; const R: TArmResult; Kind: TGateKind);
const
  OPEN_THRESH = 0.01; // a gate is considered "open" past this magnitude
var
  b, d, n, shown, idx: integer;
  GateLayer: TNNetLayer;
  W: TNNetVolume;
  v, vmin, vmax, vsum, vmean, vscale: TNeuralFloat;
  openCount: integer;       // channels whose |gate| moved meaningfully off 0
begin
  WriteLn;
  WriteLn('=== Learned gate dump: ', R.Name, ' ===');
  for b := 0 to NUM_BLOCKS - 1 do
  begin
    idx := R.GateIdx[b];
    if (idx < 0) or (idx >= NN.Layers.Count) then
    begin
      WriteLn('  block ', (b + 1):2, ': <no gate layer captured>');
      Continue;
    end;
    GateLayer := NN.Layers[idx];
    if (GateLayer = nil) or (GateLayer.Neurons.Count < 1) then
    begin
      WriteLn('  block ', (b + 1):2, ': <no gate weights>');
      Continue;
    end;
    W := GateLayer.Neurons[0].Weights;
    n := W.Size;

    if Kind = gkReZero then
    begin
      // Single scalar.
      v := W.Raw[0];
      WriteLn('  block ', (b + 1):2, ': alpha = ', SafeF(v, 10, 6));
    end
    else
    begin
      // Per-channel vector: stats + a compact bar chart.
      vmin := Infinity; vmax := -Infinity; vsum := 0; openCount := 0;
      for d := 0 to n - 1 do
      begin
        v := W.Raw[d];
        if IsNan(v) or IsInfinite(v) then Continue;
        if v < vmin then vmin := v;
        if v > vmax then vmax := v;
        vsum := vsum + v;
        if Abs(v) >= OPEN_THRESH then Inc(openCount);
      end;
      if n > 0 then vmean := vsum / n else vmean := NaN;
      vscale := Max(Abs(vmin), Abs(vmax));

      WriteLn('  block ', (b + 1):2, ': per-channel alpha[0..', n - 1,
              ']  min=', SafeF(vmin, 9, 5),
              ' mean=', SafeF(vmean, 9, 5),
              ' max=', SafeF(vmax, 9, 5),
              '  (', openCount, '/', n, ' |gate|>=', SafeF(OPEN_THRESH, 4, 2), ')');
      // Show a bar per channel (WIDTH is small enough to print all of them).
      shown := 0;
      for d := 0 to n - 1 do
      begin
        v := W.Raw[d];
        WriteLn('        ch', d:2, ' ', SafeF(v, 9, 5), ' |',
                GateBar(v, vscale, 32));
        Inc(shown);
        if shown >= WIDTH then Break;
      end;
    end;
  end;
end;

var
  Application: record Title: string; end;

begin
  Application.Title := 'ReZero vs GatedResidual';
  // A diverging arm could produce NaN / Inf. Mask the FPU exceptions so those
  // propagate as float VALUES we can detect and report cleanly, instead of
  // raising EInvalidOp and crashing the whole ablation.
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
