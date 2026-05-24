program EnergyHeads;
(*
EnergyHeads: a tiny side-by-side comparison of two final-feature
"energy" heads on a regression task. Two networks share an identical
body and differ only in the penultimate-to-output transformation:

  - L1-energy head: TNNetAbs    (elementwise |x|)
  - L2-energy head: TNNetSquare (elementwise  x^2)

Pipeline (per net):
  TNNetInput(4)
    -> TNNetFullConnectReLU(16)
    -> TNNetFullConnectReLU(8)
    -> [ TNNetAbs | TNNetSquare ]   { the only difference }
    -> TNNetFullConnectLinear(1)    { regression head }

Target function (Euclidean norm of the input vector):
  y = sqrt(x0^2 + x1^2 + x2^2 + x3^2)
with x0..x3 sampled uniformly from [-1, 1]. An energy/magnitude head
fits this target naturally: |x| linearises the L1 magnitudes a Dense
layer can already aggregate, and x^2 linearises the squared norm so a
linear head can recover sqrt-like behaviour from the learned features.

Both runs use the same RandSeed and the same synthetic data, so the
final test MSE comparison isolates the effect of the energy layer.

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
  cDIn     = 4;
  cDOut    = 1;
  cSeed    = 42;
  cTrainN  = 3000;
  cValN    = 500;
  cTestN   = 500;
  cEpochs  = 10;
  cBatch   = 32;

type
  TEnergyKind = (ekAbs, ekSquare);

  function RandUniform: TNeuralFloat;
  begin
    Result := (Random * 2.0) - 1.0;
  end;

  function TargetFn(x0, x1, x2, x3: TNeuralFloat): TNeuralFloat;
  begin
    Result := Sqrt(x0*x0 + x1*x1 + x2*x2 + x3*x3);
  end;

  function CreateRegressionPairList(MaxCnt: integer): TNNetVolumePairList;
  var
    Cnt: integer;
    x0, x1, x2, x3, y: TNeuralFloat;
  begin
    Result := TNNetVolumePairList.Create();
    for Cnt := 1 to MaxCnt do
    begin
      x0 := RandUniform;
      x1 := RandUniform;
      x2 := RandUniform;
      x3 := RandUniform;
      y  := TargetFn(x0, x1, x2, x3);

      Result.Add(
        TNNetVolumePair.Create(
          TNNetVolume.Create([x0, x1, x2, x3]),
          TNNetVolume.Create([y])
        )
      );
    end;
  end;

  // Counts a prediction as a "hit" when within 0.1 of the target.
  function LocalFloatCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
  begin
    Result := ( Abs(A.FData[0]-B.FData[0]) < 0.1 );
  end;

  function ComputeTestMSE(NN: TNNet; TestPairs: TNNetVolumePairList): TNeuralFloat;
  var
    pOut: TNNetVolume;
    Cnt: integer;
    Diff, Sum: TNeuralFloat;
  begin
    pOut := TNNetVolume.Create(1, 1, cDOut, 0);
    Sum := 0;
    for Cnt := 0 to TestPairs.Count - 1 do
    begin
      NN.Compute(TestPairs[Cnt].I);
      NN.GetOutput(pOut);
      Diff := pOut.Raw[0] - TestPairs[Cnt].O.FData[0];
      Sum := Sum + Diff*Diff;
    end;
    Result := Sum / TestPairs.Count;
    pOut.Free;
  end;

  function BuildNet(Kind: TEnergyKind): TNNet;
  begin
    Result := TNNet.Create();
    Result.AddLayer( TNNetInput.Create(cDIn) );
    Result.AddLayer( TNNetFullConnectReLU.Create(16) );
    Result.AddLayer( TNNetFullConnectReLU.Create(8) );
    case Kind of
      ekAbs:    Result.AddLayer( TNNetAbs.Create() );
      ekSquare: Result.AddLayer( TNNetSquare.Create() );
    end;
    Result.AddLayer( TNNetFullConnectLinear.Create(cDOut) );
  end;

  function KindName(Kind: TEnergyKind): string;
  begin
    case Kind of
      ekAbs:    Result := 'TNNetAbs    (L1-energy)';
      ekSquare: Result := 'TNNetSquare (L2-energy)';
    else
      Result := '?';
    end;
  end;

  procedure TrainOne(Kind: TEnergyKind;
                    TrainingPairs, ValidationPairs, TestPairs: TNNetVolumePairList;
                    out FinalMSE: TNeuralFloat;
                    out NNOut: TNNet);
  var
    NN: TNNet;
    NFit: TNeuralFit;
  begin
    // Reseed before each run so the two nets start from comparable
    // weight initialisations.
    RandSeed := cSeed;
    NN := BuildNet(Kind);

    WriteLn('=========================================================');
    WriteLn('Energy head: ', KindName(Kind));
    WriteLn('=========================================================');
    WriteLn('Layers:');
    NN.DebugStructure();

    NFit := TNeuralFit.Create();
    try
      NFit.InitialLearningRate := 0.003;
      NFit.LearningRateDecay := 0;
      NFit.L2Decay := 0;
      NFit.InferHitFn := @LocalFloatCompare;
      NFit.Fit(NN, TrainingPairs, ValidationPairs, TestPairs,
        {batchsize=}cBatch, {epochs=}cEpochs);
    finally
      NFit.Free;
    end;

    FinalMSE := ComputeTestMSE(NN, TestPairs);
    NNOut := NN;
  end;

  procedure RunAlgo();
  var
    TrainingPairs, ValidationPairs, TestPairs: TNNetVolumePairList;
    NNAbs, NNSquare: TNNet;
    MSEAbs, MSESquare: TNeuralFloat;
    pAbs, pSquare: TNNetVolume;
    Cnt: integer;
  begin
    // Generate the shared datasets once with a fixed seed.
    RandSeed := cSeed;
    TrainingPairs   := CreateRegressionPairList(cTrainN);
    ValidationPairs := CreateRegressionPairList(cValN);
    TestPairs       := CreateRegressionPairList(cTestN);

    TrainOne(ekAbs,    TrainingPairs, ValidationPairs, TestPairs, MSEAbs,    NNAbs);
    TrainOne(ekSquare, TrainingPairs, ValidationPairs, TestPairs, MSESquare, NNSquare);

    WriteLn('');
    WriteLn('=========================================================');
    WriteLn(' Final test-MSE comparison (Euclidean-norm regression)');
    WriteLn('=========================================================');
    WriteLn(' Head                      |   Test MSE');
    WriteLn(' --------------------------+--------------');
    WriteLn(' ', KindName(ekAbs),    ' | ', MSEAbs:10:6);
    WriteLn(' ', KindName(ekSquare), ' | ', MSESquare:10:6);
    WriteLn('');

    pAbs    := TNNetVolume.Create(1, 1, cDOut, 0);
    pSquare := TNNetVolume.Create(1, 1, cDOut, 0);
    WriteLn('Sample predictions on held-out test data (first 5):');
    WriteLn('  inputs                            |   target |    Abs |  Square');
    WriteLn('  ----------------------------------+----------+--------+--------');
    for Cnt := 0 to 4 do
    begin
      NNAbs.Compute(TestPairs[Cnt].I);
      NNAbs.GetOutput(pAbs);
      NNSquare.Compute(TestPairs[Cnt].I);
      NNSquare.GetOutput(pSquare);
      WriteLn
      ( '  (',
        TestPairs[Cnt].I.FData[0]:6:3,', ',
        TestPairs[Cnt].I.FData[1]:6:3,', ',
        TestPairs[Cnt].I.FData[2]:6:3,', ',
        TestPairs[Cnt].I.FData[3]:6:3,
        ') | ', TestPairs[Cnt].O.FData[0]:7:4,
        ' | ', pAbs.Raw[0]:6:3,
        ' | ', pSquare.Raw[0]:6:3
      );
    end;
    pAbs.Free;
    pSquare.Free;

    NNAbs.Free;
    NNSquare.Free;
    TestPairs.Free;
    ValidationPairs.Free;
    TrainingPairs.Free;
  end;

var
  // Stops Lazarus errors
  Application: record Title:string; end;

begin
  Application.Title:='Energy Heads Example';
  RunAlgo();
end.
