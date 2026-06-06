program SeededReproducibility;
(*
SeededReproducibility: trains the same tiny network twice from the
same RandSeed and checks that every trainable weight (and bias) is
*bit-for-bit identical* between the two runs.

Why: when a regression sneaks non-determinism into the training loop
(unseeded RNG, racing worker threads, etc.), this test catches it
immediately. The check is intentionally strict — max abs diff must be
exactly 0. On FAIL, ExitCode is set to 1 so this can be wired up as a
CI guard later.

To keep the comparison sound, the fit is run on a single worker thread
(NFit.MaxThreadNum := 1), because parallel weight updates from
multiple threads do not have a deterministic reduction order.

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

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralfit;

const
  cSeed         = 42;
  cTrainSamples = 500;
  cValSamples   = 100;
  cTestSamples  = 100;
  cBatchSize    = 32;
  cEpochs       = 3;

type
  TFloatArray = array of TNeuralFloat;

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
          TNNetVolume.Create([LocalX, LocalY]),
          TNNetVolume.Create([Hypotenuse])
        )
      );
    end;
  end;

  function LocalFloatCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
  begin
    Result := ( Abs(A.FData[0]-B.FData[0]) < 0.1 );
  end;

  // Walks every trainable weight and bias in the network and
  // concatenates them into a single flat array, in deterministic
  // layer/neuron/weight order. The two snapshots are compared
  // element-by-element after both runs finish.
  procedure SnapshotWeights(NN: TNNet; out Snap: TFloatArray);
  var
    LayerIdx, NeuronIdx, WIdx: integer;
    Layer: TNNetLayer;
    Neuron: TNNetNeuron;
    Total, Cursor: integer;
  begin
    Total := 0;
    for LayerIdx := 0 to NN.CountLayers() - 1 do
    begin
      Layer := NN.Layers[LayerIdx];
      if Layer.Neurons.Count = 0 then Continue;
      for NeuronIdx := 0 to Layer.Neurons.Count - 1 do
      begin
        Neuron := Layer.Neurons[NeuronIdx];
        if Assigned(Neuron.Weights) then
          Inc(Total, Neuron.Weights.Size);
        Inc(Total); // bias
      end;
    end;

    SetLength(Snap, Total);
    Cursor := 0;
    for LayerIdx := 0 to NN.CountLayers() - 1 do
    begin
      Layer := NN.Layers[LayerIdx];
      if Layer.Neurons.Count = 0 then Continue;
      for NeuronIdx := 0 to Layer.Neurons.Count - 1 do
      begin
        Neuron := Layer.Neurons[NeuronIdx];
        if Assigned(Neuron.Weights) then
        begin
          for WIdx := 0 to Neuron.Weights.Size - 1 do
          begin
            Snap[Cursor] := Neuron.Weights.FData[WIdx];
            Inc(Cursor);
          end;
        end;
        Snap[Cursor] := Neuron.Bias;
        Inc(Cursor);
      end;
    end;
  end;

  procedure TrainOnce(out Snap: TFloatArray);
  var
    NN: TNNet;
    NFit: TNeuralFit;
    TrainingPairs, ValidationPairs, TestPairs: TNNetVolumePairList;
  begin
    RandSeed := cSeed;

    NN := TNNet.Create();
    NFit := TNeuralFit.Create();
    NFit.MaxThreadNum := 1; // single-threaded → deterministic reductions
    NFit.Verbose := False;

    TrainingPairs   := CreateHypotenusePairList(cTrainSamples);
    ValidationPairs := CreateHypotenusePairList(cValSamples);
    TestPairs       := CreateHypotenusePairList(cTestSamples);

    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectLinear.Create(1)
    ]);

    NFit.InitialLearningRate := 0.00001;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.InferHitFn := @LocalFloatCompare;
    NFit.Fit(NN, TrainingPairs, ValidationPairs, TestPairs,
      cBatchSize, cEpochs);

    SnapshotWeights(NN, Snap);

    TestPairs.Free;
    ValidationPairs.Free;
    TrainingPairs.Free;
    NFit.Free;
    NN.Free;
  end;

  procedure RunCheck();
  var
    SnapA, SnapB: TFloatArray;
    Idx: integer;
    Diff, MaxDiff: TNeuralFloat;
    MismatchCount: integer;
  begin
    WriteLn('SeededReproducibility: training twice with RandSeed=', cSeed, '...');
    WriteLn('Run 1 of 2:');
    TrainOnce(SnapA);
    WriteLn('Run 2 of 2:');
    TrainOnce(SnapB);

    WriteLn;
    if Length(SnapA) <> Length(SnapB) then
    begin
      WriteLn('FAIL: snapshot sizes differ (', Length(SnapA), ' vs ',
        Length(SnapB), ')');
      ExitCode := 1;
      Exit;
    end;

    MaxDiff := 0;
    MismatchCount := 0;
    for Idx := 0 to High(SnapA) do
    begin
      Diff := Abs(SnapA[Idx] - SnapB[Idx]);
      if Diff <> 0 then Inc(MismatchCount);
      if Diff > MaxDiff then MaxDiff := Diff;
    end;

    WriteLn('Weights compared : ', Length(SnapA));
    WriteLn('Mismatching      : ', MismatchCount);
    WriteLn('Max abs diff     : ', MaxDiff:0:20);

    if MaxDiff = 0 then
    begin
      WriteLn('PASS: bit-for-bit identical weights across runs.');
      ExitCode := 0;
    end
    else
    begin
      WriteLn('FAIL: same-seed runs diverged.');
      ExitCode := 1;
    end;
  end;

var
  // Stops Lazarus errors
  Application: record Title:string; end;

begin
  Application.Title:='Seeded Reproducibility Check';
  RunCheck();
end.
