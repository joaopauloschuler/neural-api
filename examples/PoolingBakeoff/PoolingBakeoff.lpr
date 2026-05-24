program PoolingBakeoff;
(*
PoolingBakeoff: a pooling-HEAD bake-off. The exact same tiny conv
classifier is trained on the exact same synthetic image task; the ONLY
thing that changes between arms is the global pooling layer that reduces
the 12x12 feature map to 1x1 before the linear classifier:

  - TNNetAvgPool                       (plain mean over the window)
  - TNNetMaxPool                       (single largest activation)
  - TNNetLpPool  with p in {1,2,4,8}   (power-mean; p=1 -> mean,
                                        p -> inf approaches max)
  - TNNetSoftPool with beta in {0.5,1,2,8}
                                       (softmax-weighted mean;
                                        beta -> 0 -> mean,
                                        beta -> inf approaches max)

The synthetic task is a 4-class problem on 12x12x3 images. Each class
plants a bright blob in a class-specific quadrant on top of uniform
noise, so WHERE the activation energy sits (not just its average) is
discriminative. That makes the choice of spatial pooling matter and
lets the avg<->max interpolation of LpPool's p and SoftPool's beta show
up empirically in the final loss / accuracy.

Every arm reseeds RandSeed to the same value before generating its data
and before building/initialising the net, so all arms see identical
inputs and identical weight init; only the pooling layer differs.

Pure CPU, no external dataset, finishes well under a few minutes.

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
  neuralvolume;

const
  cImgSize     = 12;     // 12 x 12 images
  cChannels    = 3;
  cNumClasses  = 4;      // one bright-blob quadrant per class
  cTrainPer    = 300;    // 1200 train samples total
  cValPer      = 80;     // 320 validation samples total
  cEpochs      = 16;
  cLearnRate   = 0.01;
  cInertia     = 0.9;
  cSeed        = 42;
  cBlobAmp     = 0.9;    // peak blob brightness added on top of noise
  cNoiseAmp    = 0.35;   // uniform background noise amplitude

type
  TPoolKind = (pkAvg, pkMax, pkLp, pkSoft);

  TArm = record
    Name : string;
    Kind : TPoolKind;
    Param: TNeuralFloat;  // p for Lp, beta for Soft, ignored otherwise
  end;

  TArmResult = record
    Name        : string;
    FinalValLoss: TNeuralFloat;
    FinalValAcc : TNeuralFloat;
    Trace       : array[1..cEpochs] of TNeuralFloat; // per-epoch train NLL
  end;

const
  cArms: array[0..9] of TArm =
  (
    (Name: 'TNNetAvgPool';          Kind: pkAvg;  Param: 0.0),
    (Name: 'TNNetMaxPool';          Kind: pkMax;  Param: 0.0),
    (Name: 'TNNetLpPool(p=1)';      Kind: pkLp;   Param: 1.0),
    (Name: 'TNNetLpPool(p=2)';      Kind: pkLp;   Param: 2.0),
    (Name: 'TNNetLpPool(p=4)';      Kind: pkLp;   Param: 4.0),
    (Name: 'TNNetLpPool(p=8)';      Kind: pkLp;   Param: 8.0),
    (Name: 'TNNetSoftPool(beta=0.5)'; Kind: pkSoft; Param: 0.5),
    (Name: 'TNNetSoftPool(beta=1)';   Kind: pkSoft; Param: 1.0),
    (Name: 'TNNetSoftPool(beta=2)';   Kind: pkSoft; Param: 2.0),
    (Name: 'TNNetSoftPool(beta=8)';   Kind: pkSoft; Param: 8.0)
  );

// Quadrant origins (x0,y0) for the 6x6 blob region of each class.
function ClassQuadrant(ClassId: integer; out X0, Y0: integer): integer;
const
  Half = cImgSize div 2;
begin
  case ClassId of
    0: begin X0 := 0;    Y0 := 0;    end; // top-left
    1: begin X0 := Half; Y0 := 0;    end; // top-right
    2: begin X0 := 0;    Y0 := Half; end; // bottom-left
  else begin X0 := Half; Y0 := Half; end; // bottom-right
  end;
  Result := ClassId;
end;

// Build one synthetic sample: uniform noise everywhere + a soft bright
// blob centered in the class quadrant, replicated across all channels.
procedure MakeSample(ClassId: integer; out X, Y: TNNetVolume);
var
  PX, PY, C, X0, Y0, CX, CY: integer;
  D2, Falloff, V: TNeuralFloat;
begin
  X := TNNetVolume.Create(cImgSize, cImgSize, cChannels);
  Y := TNNetVolume.Create(cNumClasses, 1, 1);
  Y.Fill(0);
  Y.FData[ClassId] := 1.0;

  // background noise
  for PX := 0 to cImgSize - 1 do
    for PY := 0 to cImgSize - 1 do
      for C := 0 to cChannels - 1 do
        X.Data[PX, PY, C] := Random * cNoiseAmp;

  ClassQuadrant(ClassId, X0, Y0);
  CX := X0 + (cImgSize div 2) div 2; // blob center inside the quadrant
  CY := Y0 + (cImgSize div 2) div 2;

  for PX := X0 to X0 + (cImgSize div 2) - 1 do
    for PY := Y0 to Y0 + (cImgSize div 2) - 1 do
    begin
      D2 := Sqr(PX - CX) + Sqr(PY - CY);
      Falloff := Exp(-D2 / 4.0); // gaussian-ish blob
      V := cBlobAmp * Falloff;
      for C := 0 to cChannels - 1 do
        X.Data[PX, PY, C] := X.Data[PX, PY, C] + V;
    end;
end;

procedure BuildSet(out Pairs: TNNetVolumePairList; PerClass: integer);
var
  C, I: integer;
  X, Y: TNNetVolume;
begin
  Pairs := TNNetVolumePairList.Create();
  for C := 0 to cNumClasses - 1 do
    for I := 1 to PerClass do
    begin
      MakeSample(C, X, Y);
      Pairs.Add(TNNetVolumePair.Create(X, Y));
    end;
end;

// Shared backbone + the swappable pooling head.
procedure BuildNet(out NN: TNNet; Arm: TArm);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cImgSize, cImgSize, cChannels));
  // Two padded 3x3 convolutions keep the 12x12 spatial size so the
  // pooling window below is a clean global 12x12 reduction.
  NN.AddLayer(TNNetConvolutionReLU.Create(12, 3, 1, 1)); // 12x12x12
  NN.AddLayer(TNNetConvolutionReLU.Create(12, 3, 1, 1)); // 12x12x12

  // --- swappable global pooling head (12x12 -> 1x1) ---
  case Arm.Kind of
    pkAvg : NN.AddLayer(TNNetAvgPool.Create(cImgSize));
    pkMax : NN.AddLayer(TNNetMaxPool.Create(cImgSize));
    pkLp  : NN.AddLayer(TNNetLpPool.Create(cImgSize, cImgSize, 0, Arm.Param));
    pkSoft: NN.AddLayer(TNNetSoftPool.Create(cImgSize, cImgSize, 0, Arm.Param));
  end;

  NN.AddLayer(TNNetFullConnectLinear.Create(cNumClasses)); // 1x1x4
  NN.AddLayer(TNNetSoftMax.Create());
end;

// Mean cross-entropy (NLL) and top-1 accuracy over a pair list.
procedure Evaluate(NN: TNNet; Pairs: TNNetVolumePairList;
  out MeanNLL, Acc: TNeuralFloat);
var
  I, Hits, Tgt, Pred: integer;
  P: TNNetVolume;
  Sum: Double;
begin
  Sum := 0;
  Hits := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    P := NN.GetLastLayer().Output;
    Tgt := Pairs[I].O.GetClass();
    Sum := Sum - Ln(Max(1e-9, P.FData[Tgt]));
    Pred := P.GetClass();
    if Pred = Tgt then Inc(Hits);
  end;
  if Pairs.Count > 0 then
  begin
    MeanNLL := Sum / Pairs.Count;
    Acc := Hits / Pairs.Count;
  end
  else
  begin
    MeanNLL := 0;
    Acc := 0;
  end;
end;

function RunArm(Arm: TArm): TArmResult;
var
  NN: TNNet;
  TrainSet, ValSet: TNNetVolumePairList;
  Epoch, I: integer;
  P: TNNetVolume;
  Sum: Double;
  ValNLL, ValAcc: TNeuralFloat;
begin
  Result.Name := Arm.Name;

  // Reseed BEFORE data gen and BEFORE net build so every arm sees the
  // same inputs and the same weight initialisation.
  RandSeed := cSeed;
  BuildSet(TrainSet, cTrainPer);
  BuildSet(ValSet, cValPer);

  RandSeed := cSeed;
  BuildNet(NN, Arm);
  try
    NN.SetLearningRate(cLearnRate, cInertia);

    for Epoch := 1 to cEpochs do
    begin
      Sum := 0;
      for I := 0 to TrainSet.Count - 1 do
      begin
        NN.Compute(TrainSet[I].I);
        NN.Backpropagate(TrainSet[I].O);
        P := NN.GetLastLayer().Output;
        Sum := Sum - Ln(Max(1e-9, P.FData[TrainSet[I].O.GetClass()]));
      end;
      Result.Trace[Epoch] := Sum / TrainSet.Count;
    end;

    Evaluate(NN, ValSet, ValNLL, ValAcc);
    Result.FinalValLoss := ValNLL;
    Result.FinalValAcc  := ValAcc;
  finally
    NN.Free;
    ValSet.Free;
    TrainSet.Free;
  end;
end;

function SafeF(V: TNeuralFloat; Dec: integer): string;
begin
  if IsNan(V) or IsInfinite(V) then
    Result := 'NaN'
  else
    Result := FloatToStrF(V, ffFixed, 8, Dec);
end;

procedure RunBakeoff();
var
  K, Epoch: integer;
  Results: array[0..High(cArms)] of TArmResult;
  StartTime, EndTime: TDateTime;
begin
  WriteLn('Pooling-head bake-off on a synthetic 4-class blob-quadrant task.');
  WriteLn('Backbone: Input(', cImgSize, 'x', cImgSize, 'x', cChannels,
          ') -> Conv12(3x3,pad1)+ReLU -> Conv12(3x3,pad1)+ReLU');
  WriteLn('          -> [POOL ', cImgSize, 'x', cImgSize, ' global]',
          ' -> FullConnectLinear(', cNumClasses, ') -> SoftMax');
  WriteLn('Train=', cNumClasses * cTrainPer, ' Val=', cNumClasses * cValPer,
          '  Epochs=', cEpochs, '  LR=', SafeF(cLearnRate, 3),
          '  RandSeed=', cSeed);
  WriteLn('Only the pooling layer changes between arms (same data, same init).');
  WriteLn;

  StartTime := Now;
  for K := 0 to High(cArms) do
  begin
    Write('Training ', cArms[K].Name, ' ...');
    Results[K] := RunArm(cArms[K]);
    WriteLn(' done.  val_loss=', SafeF(Results[K].FinalValLoss, 4),
            '  val_acc=', SafeF(Results[K].FinalValAcc, 4));
  end;
  EndTime := Now;

  WriteLn;
  WriteLn('=== Results (CSV) ===');
  WriteLn('pooling,final_val_loss,final_val_accuracy');
  for K := 0 to High(cArms) do
    WriteLn(Results[K].Name, ',',
            SafeF(Results[K].FinalValLoss, 4), ',',
            SafeF(Results[K].FinalValAcc, 4));

  WriteLn;
  WriteLn('=== Train NLL vs epoch (per arm) ===');
  Write('epoch');
  for Epoch := 1 to cEpochs do Write(',', Epoch);
  WriteLn;
  for K := 0 to High(cArms) do
  begin
    Write(Results[K].Name);
    for Epoch := 1 to cEpochs do
      Write(',', SafeF(Results[K].Trace[Epoch], 4));
    WriteLn;
  end;

  WriteLn;
  WriteLn('Total wall time: ',
          FormatFloat('0.00', (EndTime - StartTime) * 86400), ' s');
end;

begin
  RandSeed := cSeed;
  RunBakeoff();
end.
