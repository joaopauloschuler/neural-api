program GLUFeedForward;
(*
GLUFeedForward: a tiny transformer-style feed-forward block
demo. Trains the canonical "Dense -> GLU -> Dense" sandwich on a
synthetic regression task so you can see GLU end-to-end in
isolation, without an attention block or embedding around it.

Pipeline:
  TNNetInput(D_in)
    -> TNNetFullConnectLinear(2*D_hidden)   { gate || value packed on depth }
    -> TNNetGLU                             { value * Sigmoid(gate), halves depth }
    -> TNNetFullConnectLinear(D_out)        { regression head }

Target function (deterministic, learnable, mildly nonlinear):
  y = sin(x0) + 0.5*x1*x2 - 0.3*x3
with x0..x3 sampled uniformly from [-1, 1].

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
  cDHidden = 16;
  cDOut    = 1;

  function RandUniform: TNeuralFloat;
  begin
    Result := (Random * 2.0) - 1.0;
  end;

  function TargetFn(x0, x1, x2, x3: TNeuralFloat): TNeuralFloat;
  begin
    Result := Sin(x0) + 0.5 * x1 * x2 - 0.3 * x3;
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

  procedure RunAlgo();
  var
    NN: TNNet;
    NFit: TNeuralFit;
    TrainingPairs, ValidationPairs, TestPairs: TNNetVolumePairList;
    Cnt: integer;
    pOutPut: TNNetVolume;
  begin
    RandSeed := 42;
    NN := TNNet.Create();
    NFit := TNeuralFit.Create();
    TrainingPairs   := CreateRegressionPairList(3000);
    ValidationPairs := CreateRegressionPairList(500);
    TestPairs       := CreateRegressionPairList(500);

    NN.AddLayer( TNNetInput.Create(cDIn) );
    // Dense(2*D_hidden) -> GLU -> Dense(D_out) in a single call.
    NN.AddGLUFeedForward(cDIn, cDHidden, cDOut);

    WriteLn('Layers:');
    NN.DebugStructure();

    WriteLn('Training Dense -> GLU -> Dense FFN on synthetic regression...');
    NFit.InitialLearningRate := 0.003;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.InferHitFn := @LocalFloatCompare;
    NFit.Fit(NN, TrainingPairs, ValidationPairs, TestPairs,
      {batchsize=}32, {epochs=}20);
    NN.DebugWeights();

    pOutPut := TNNetVolume.Create({pSizeX=}1, {pSizeY=}1, {pDepth=}cDOut, {FillValue=}0);

    WriteLn('Sample predictions on held-out test data:');
    for Cnt := 0 to 9 do
    begin
      NN.Compute(TestPairs[Cnt].I);
      NN.GetOutput(pOutPut);
      WriteLn
      ( '  inputs=(',
        TestPairs[Cnt].I.FData[0]:6:3,', ',
        TestPairs[Cnt].I.FData[1]:6:3,', ',
        TestPairs[Cnt].I.FData[2]:6:3,', ',
        TestPairs[Cnt].I.FData[3]:6:3,
        ')  predicted=', pOutPut.Raw[0]:7:4,
        '  target=', TestPairs[Cnt].O.FData[0]:7:4
      );
    end;
    pOutPut.Free;
    TestPairs.Free;
    ValidationPairs.Free;
    TrainingPairs.Free;
    NFit.Free;
    NN.Free;
  end;

var
  // Stops Lazarus errors
  Application: record Title:string; end;

begin
  Application.Title:='GLU Feed-Forward Example';
  RunAlgo();
end.
