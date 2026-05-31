program APLBakeoff;
(*
 APLBakeoff: APL vs PReLU vs ReLU activation bake-off on the hypotenuse toy.

 Trains the SAME small MLP - Input(2) -> FullConnect(H) -> activation ->
 FullConnect(H) -> activation -> FullConnectLinear(1) - on the hypotenuse
 regression toy (predict sqrt(a^2+b^2) from (a,b)), changing ONLY the hidden
 activation between arms:

   ReLU      - plain rectifier (no extra parameters)
   PReLU     - one learnable negative slope per unit
   APL S=1   - Adaptive Piecewise Linear with 1 hinge  (2 params / unit)
   APL S=2   - Adaptive Piecewise Linear with 2 hinges (4 params / unit)
   APL S=4   - Adaptive Piecewise Linear with 4 hinges (8 params / unit)

 Question (from tasklist): does the extra piecewise capacity of APL buy a
 lower final loss? The table prints the trainable-parameter count for every
 arm so the comparison stays honest (APL adds 2*S params per channel).

 Whole run is a few thousand samples / modest epochs, finishes well under
 5 minutes on CPU (about 2 s on the dev machine). A fixed RandSeed makes
 every arm reproducible.

 Copyright (C) 2026 Joao Paulo Schwarz Schuler

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume,
  neuralfit;

type
  { which hidden activation an arm uses }
  TActivationKind = (akReLU, akPReLU, akAPL);

  TActivationSpec = record
    Name: string;
    Kind: TActivationKind;
    Hinges: integer; { only used when Kind = akAPL }
  end;

const
  FIXED_SEED   = 424242;
  HIDDEN       = 32;     { hidden width, identical for every arm }
  TRAIN_COUNT  = 2000;
  TEST_COUNT   = 500;
  NUM_EPOCHS   = 60;
  BATCH_SIZE   = 1000;
  LR           = 0.001;

{ Builds one hypotenuse pair list.  Inputs and target are scaled by 0.01 so
  they sit in a friendly range for the network. }
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
        TNNetVolume.Create([0.01*LocalX, 0.01*LocalY]),
        TNNetVolume.Create([0.01*Hypotenuse])
      )
    );
  end;
end;

{ Mean squared error of a trained network over a pair list (single output). }
function MeanSquaredError(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I: integer;
  V: TNeuralFloat;
  Sum: Double;
begin
  Sum := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    V := NN.GetLastLayer().Output.FData[0] - Pairs[I].O.FData[0];
    Sum := Sum + V*V;
  end;
  if Pairs.Count > 0 then
    Result := Sum / Pairs.Count
  else
    Result := 0;
end;

procedure AddActivation(NN: TNNet; const Spec: TActivationSpec);
begin
  case Spec.Kind of
    akReLU:  NN.AddLayer(TNNetReLU.Create());
    akPReLU: NN.AddLayer(TNNetPReLU.Create());
    akAPL:   NN.AddLayer(TNNetAPL.Create(Spec.Hinges));
  end;
end;

function BuildNet(const Spec: TActivationSpec): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(2));
  Result.AddLayer(TNNetFullConnect.Create(HIDDEN));
  AddActivation(Result, Spec);
  Result.AddLayer(TNNetFullConnect.Create(HIDDEN));
  AddActivation(Result, Spec);
  Result.AddLayer(TNNetFullConnectLinear.Create(1));
end;

procedure RunAlgo();
var
  Arms: array of TActivationSpec;
  TrainPairs, TestPairs: TNNetVolumePairList;
  NN: TNNet;
  NFit: TNeuralFit;
  TrainLoss, ValLoss: array of TNeuralFloat;
  Params: array of integer;
  I: integer;
begin
  { the activation arms to compare }
  SetLength(Arms, 5);
  Arms[0].Name := 'ReLU';     Arms[0].Kind := akReLU;  Arms[0].Hinges := 0;
  Arms[1].Name := 'PReLU';    Arms[1].Kind := akPReLU; Arms[1].Hinges := 0;
  Arms[2].Name := 'APL S=1';  Arms[2].Kind := akAPL;   Arms[2].Hinges := 1;
  Arms[3].Name := 'APL S=2';  Arms[3].Kind := akAPL;   Arms[3].Hinges := 2;
  Arms[4].Name := 'APL S=4';  Arms[4].Kind := akAPL;   Arms[4].Hinges := 4;

  SetLength(TrainLoss, Length(Arms));
  SetLength(ValLoss, Length(Arms));
  SetLength(Params, Length(Arms));

  WriteLn('APL vs PReLU vs ReLU bake-off on the hypotenuse toy (regression)');
  WriteLn('Same MLP: Input(2) -> FC(', HIDDEN, ') -> act -> FC(', HIDDEN,
          ') -> act -> FCLinear(1)');
  WriteLn(TRAIN_COUNT, ' train / ', TEST_COUNT, ' test samples, ',
          NUM_EPOCHS, ' epochs, seed=', FIXED_SEED);
  WriteLn('');

  for I := 0 to High(Arms) do
  begin
    { reseed before EACH arm so data + init are identical across arms }
    RandSeed := FIXED_SEED;

    TrainPairs := CreateHypotenusePairList(TRAIN_COUNT);
    TestPairs := CreateHypotenusePairList(TEST_COUNT);

    NN := BuildNet(Arms[I]);
    Params[I] := NN.CountWeights();

    NFit := TNeuralFit.Create();
    try
      NFit.InitialLearningRate := LR;
      NFit.LearningRateDecay := 0;
      NFit.L2Decay := 0;
      NFit.InferHitFn := nil;
      NFit.Verbose := False;
      NFit.Fit(NN, TrainPairs, TestPairs, nil, BATCH_SIZE, NUM_EPOCHS);

      { measure final MSE directly (regression toy) }
      TrainLoss[I] := MeanSquaredError(NN, TrainPairs);
      ValLoss[I] := MeanSquaredError(NN, TestPairs);
    finally
      NFit.Free;
      NN.Free;
      TestPairs.Free;
      TrainPairs.Free;
    end;

    WriteLn(Format('  done: %-8s  train MSE=%.6f  val MSE=%.6f  params=%d',
            [Arms[I].Name, TrainLoss[I], ValLoss[I], Params[I]]));
  end;

  WriteLn('');
  WriteLn('=== Activation bake-off results ===');
  WriteLn('Arm        Train MSE     Val MSE     Params');
  WriteLn('-------------------------------------------');
  for I := 0 to High(Arms) do
    WriteLn(Format('%-9s  %10.6f  %10.6f  %6d',
            [Arms[I].Name, TrainLoss[I], ValLoss[I], Params[I]]));
  WriteLn('-------------------------------------------');
  WriteLn('Note: APL adds 2*S learnable params per hidden unit (slope+knee per');
  WriteLn('hinge), so its param count is higher than ReLU/PReLU at equal width.');
end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'APLBakeoff';
  RandSeed := FIXED_SEED;
  RunAlgo();
end.
