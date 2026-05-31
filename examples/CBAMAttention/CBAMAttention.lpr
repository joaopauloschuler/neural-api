program CBAMAttention;
(*
CBAMAttention: minimal demo of TNNet.AddCBAM (Convolutional Block Attention
Module, Woo et al. 2018).

The example builds a tiny conv net that classifies a small synthetic image
dataset (16x16x3 -> 2 classes) and trains for a few epochs. The point is to
exercise AddCBAM end-to-end (forward + backward through BOTH the channel and
spatial attention sub-modules) and to show the loss decreasing, rather than to
reach a competitive accuracy. It is CPU-only and runs in well under a minute.

The feature map is kept SQUARE (16x16) on purpose: the channel sub-module's
global max-pool (TNNetMaxChannel) assumes SizeX = SizeY.

License: LGPL with linking exception (same as the rest of neural-api).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume,
  neuralfit;

  // Mean squared error + classification accuracy of NN over a pair list. Used
  // to demonstrate, explicitly and independently of TNeuralFit's internal log
  // cadence, that the CBAM net's loss goes DOWN with training.
  procedure EvalLossAndAcc(NN: TNNet; Pairs: TNNetVolumePairList;
    out Loss, Acc: TNeuralFloat);
  var
    P, K, Hits: integer;
    Diff: TNeuralFloat;
    Pair: TNNetVolumePair;
  begin
    Loss := 0;
    Hits := 0;
    for P := 0 to Pairs.Count - 1 do
    begin
      Pair := Pairs[P];
      NN.Compute(Pair.A);
      for K := 0 to NN.GetLastLayer.Output.Size - 1 do
      begin
        Diff := NN.GetLastLayer.Output.FData[K] - Pair.B.FData[K];
        Loss := Loss + 0.5 * Diff * Diff;
      end;
      if NN.GetLastLayer.Output.GetClass() = Pair.B.GetClass() then Inc(Hits);
    end;
    Loss := Loss / Pairs.Count;
    Acc := Hits / Pairs.Count;
  end;

  function CreateSyntheticPairs(Count: integer): TNNetVolumePairList;
  var
    Cnt, Idx, ClassIdx: integer;
    Inp, Tgt: TNNetVolume;
  begin
    Result := TNNetVolumePairList.Create();
    for Cnt := 1 to Count do
    begin
      Inp := TNNetVolume.Create(16, 16, 3);
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
  const
    Rounds = 5;
    EpochsPerRound = 2;
  var
    NN: TNNet;
    NFit: TNeuralFit;
    ConvOut: TNNetLayer;
    TrainPairs, ValPairs, TestPairs: TNNetVolumePairList;
    R: integer;
    Loss, Acc: TNeuralFloat;
  begin
    RandSeed := 42;
    NN := TNNet.Create();
    TrainPairs := CreateSyntheticPairs(256);
    ValPairs   := CreateSyntheticPairs(64);
    TestPairs  := CreateSyntheticPairs(64);

    // Tiny net: input -> conv -> CBAM (channel+spatial attention) -> pool -> softmax(2).
    NN.AddLayer( TNNetInput.Create(16, 16, 3) );
    ConvOut := NN.AddLayer( TNNetConvolutionReLU.Create({features=}8, {fsize=}3, {pad=}1, {stride=}1) );
    NN.AddCBAM(ConvOut, {ReductionRatio=}4, {SpatialKernelSize=}7);
    NN.AddLayer( TNNetAvgChannel.Create() );
    NN.AddLayer( TNNetFullConnectLinear.Create(2) );
    NN.AddLayer( TNNetSoftMax.Create() );

    WriteLn('CBAMAttention: tiny net summary');
    NN.DebugStructure();

    // Train in short rounds, printing the training loss/accuracy after each so
    // the loss is visibly DECREASING (this is independent of TNeuralFit's own
    // batch-logging cadence, which stays quiet on a dataset this small).
    EvalLossAndAcc(NN, TrainPairs, Loss, Acc);
    WriteLn(Format('Before training   train loss=%.5f  train acc=%.4f', [Loss, Acc]));
    for R := 1 to Rounds do
    begin
      NFit := TNeuralFit.Create();
      NFit.MaxThreadNum := 1;
      NFit.InitialLearningRate := 0.01;
      NFit.LearningRateDecay := 0;
      NFit.L2Decay := 0;
      NFit.Verbose := false;
      NFit.Fit(NN, TrainPairs, ValPairs, TestPairs,
        {batchsize=}16, {epochs=}EpochsPerRound);
      NFit.Free;
      EvalLossAndAcc(NN, TrainPairs, Loss, Acc);
      WriteLn(Format('After %2d epochs   train loss=%.5f  train acc=%.4f',
        [R * EpochsPerRound, Loss, Acc]));
    end;

    TestPairs.Free;
    ValPairs.Free;
    TrainPairs.Free;
    NN.Free;
  end;

var
  // Stops Lazarus errors.
  Application: record Title: string; end;

begin
  Application.Title := 'CBAMAttention Example';
  RunDemo();
end.
