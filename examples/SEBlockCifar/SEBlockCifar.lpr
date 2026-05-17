program SEBlockCifar;
(*
SEBlockCifar: minimal demo of TNNet.AddSEBlock (Squeeze-and-Excitation).

The example builds a tiny conv net that classifies a small synthetic
"CIFAR-like" dataset (32x32x3 -> 2 classes) and trains for a couple of
epochs. The point is to exercise AddSEBlock end-to-end (forward + backward
through the channel gating) rather than to reach a competitive accuracy.

License: LGPL with linking exception (same as the rest of neural-api).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume,
  neuralfit;

  function CreateSyntheticPairs(Count: integer): TNNetVolumePairList;
  var
    Cnt, Idx, ClassIdx: integer;
    Inp, Tgt: TNNetVolume;
  begin
    Result := TNNetVolumePairList.Create();
    for Cnt := 1 to Count do
    begin
      Inp := TNNetVolume.Create(32, 32, 3);
      Tgt := TNNetVolume.Create(2);
      ClassIdx := Random(2);
      // Class 0: low-mean noise. Class 1: high-mean noise. Trivially separable.
      for Idx := 0 to Inp.Size - 1 do
        Inp.FData[Idx] := Random + ClassIdx * 0.7 - 0.5;
      Tgt.Fill(0);
      Tgt.FData[ClassIdx] := 1.0;
      Result.Add(TNNetVolumePair.Create(Inp, Tgt));
    end;
  end;

  procedure RunDemo();
  var
    NN: TNNet;
    NFit: TNeuralFit;
    ConvOut: TNNetLayer;
    TrainPairs, ValPairs, TestPairs: TNNetVolumePairList;
  begin
    RandSeed := 42;
    NN := TNNet.Create();
    NFit := TNeuralFit.Create();
    TrainPairs := CreateSyntheticPairs(256);
    ValPairs   := CreateSyntheticPairs(64);
    TestPairs  := CreateSyntheticPairs(64);

    // Tiny net: input -> conv -> SE block -> avg-pool -> softmax(2).
    NN.AddLayer( TNNetInput.Create(32, 32, 3) );
    ConvOut := NN.AddLayer( TNNetConvolutionReLU.Create({features=}8, {fsize=}3, {pad=}1, {stride=}1) );
    NN.AddSEBlock(ConvOut, {ReductionRatio=}4);
    NN.AddLayer( TNNetAvgChannel.Create() );
    NN.AddLayer( TNNetFullConnectLinear.Create(2) );
    NN.AddLayer( TNNetSoftMax.Create() );

    WriteLn('SEBlockCifar: tiny net summary');
    NN.DebugStructure();

    NFit.InitialLearningRate := 0.001;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Fit(NN, TrainPairs, ValPairs, TestPairs,
      {batchsize=}16, {epochs=}2);

    TestPairs.Free;
    ValPairs.Free;
    TrainPairs.Free;
    NFit.Free;
    NN.Free;
  end;

var
  // Stops Lazarus errors.
  Application: record Title: string; end;

begin
  Application.Title := 'SEBlockCifar Example';
  RunDemo();
end.
