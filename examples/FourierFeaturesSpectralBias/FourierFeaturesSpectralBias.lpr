program FourierFeaturesSpectralBias;
(*
FourierFeaturesSpectralBias: the headline Tancik et al. 2020 "spectral bias"
micro-experiment for TNNetFourierFeatures.

A small ReLU coordinate-MLP is asked to fit a high-frequency 1D target

    y = sin(20*x) + 0.5*sin(53*x),   x in [-1, 1]

TWICE, with the IDENTICAL hidden head:

  raw      : Input(1) ->                          FullConnectReLU(h) ->
             FullConnectReLU(h) -> FullConnectLinear(1)
  fourier  : Input(1) -> FourierFeatures(M,sigma) -> FullConnectReLU(h) ->
             FullConnectReLU(h) -> FullConnectLinear(1)

The headline result: the raw-coordinate MLP suffers from "spectral bias" and
cannot fit the high frequencies (large MSE), while the Fourier-mapped one fits
them well (much smaller MSE). The program prints the side-by-side MSE gap and a
small predicted-vs-truth table at 11 points.

It then runs the sigma BANDWIDTH SWEEP: the same Fourier fit with
sigma in {0.5, 2, 8, 32}, printing final MSE vs sigma. Too small a sigma stays
low-pass (underfits); too large is noisy (overfits). The single-knob story.

Pure CPU, no dataset download, finishes well under two minutes.

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
  NumTrain = 384;
  NumEpochs = 400;
  BatchSize = 16;
  LearningRate = 0.01;
  HiddenWidth = 64;
  NumFeatures = 64;   // M; Fourier output Depth = 2*M
  DefaultSigma = 4.0; // good bandwidth for this target (see sigma sweep)

  // The high-frequency target. Deliberately mixes a moderate (20) and a high
  // (53) angular frequency so the raw MLP cannot fit both.
  function Target(X: TNeuralFloat): TNeuralFloat;
  begin
    Result := Sin(20.0 * X) + 0.5 * Sin(53.0 * X);
  end;

  function CreateTargetPairList(MaxCnt: integer): TNNetVolumePairList;
  var
    Cnt: integer;
    X: TNeuralFloat;
  begin
    Result := TNNetVolumePairList.Create();
    for Cnt := 1 to MaxCnt do
    begin
      X := 2.0 * Random - 1.0;  // x in [-1, 1)
      Result.Add(
        TNNetVolumePair.Create(
          TNNetVolume.Create([X]),
          TNNetVolume.Create([Target(X)])
        )
      );
    end;
  end;

  // Computes MSE on a clean 200-point test grid over [-1, 1].
  function CleanGridMSE(NN: TNNet): TNeuralFloat;
  var
    I: integer;
    X, Tgt, Pred, Diff, Sum: TNeuralFloat;
    Input, Output: TNNetVolume;
  begin
    Input := TNNetVolume.Create([TNeuralFloat(0)]);
    Output := TNNetVolume.Create(1, 1, 1);
    Sum := 0;
    for I := 0 to 199 do
    begin
      X := -1.0 + 2.0 * I / 199.0;
      Tgt := Target(X);
      Input.FData[0] := X;
      NN.Compute(Input);
      NN.GetOutput(Output);
      Pred := Output.Raw[0];
      Diff := Pred - Tgt;
      Sum := Sum + Diff * Diff;
    end;
    Input.Free;
    Output.Free;
    Result := Sum / 200.0;
  end;

  // Fisher-Yates shuffle of an integer index array.
  procedure ShuffleIndices(var Idx: array of integer);
  var
    I, J, Tmp: integer;
  begin
    for I := High(Idx) downto 1 do
    begin
      J := Random(I + 1);
      Tmp := Idx[I];
      Idx[I] := Idx[J];
      Idx[J] := Tmp;
    end;
  end;

  // Builds the coordinate-MLP. When UseFourier is set, the front-end is a
  // TNNetFourierFeatures lift of the scalar x; otherwise the raw x feeds the
  // identical ReLU head. The hidden head is byte-for-byte the same in both.
  function BuildNet(UseFourier: boolean; Sigma: TNeuralFloat): TNNet;
  begin
    Result := TNNet.Create();
    Result.AddLayer(TNNetInput.Create(1));
    if UseFourier then
      Result.AddLayer(TNNetFourierFeatures.Create(NumFeatures, Sigma, {seed=}0));
    Result.AddLayer(TNNetFullConnectReLU.Create(HiddenWidth));
    Result.AddLayer(TNNetFullConnectReLU.Create(HiddenWidth));
    Result.AddLayer(TNNetFullConnectLinear.Create(1));
  end;

  // Trains one network with plain shuffled mini-batch SGD and returns the
  // final clean-grid MSE. Verbose prints a few progress lines.
  function TrainNet(NN: TNNet; TrainingPairs: TNNetVolumePairList;
    Verbose: boolean): TNeuralFloat;
  var
    Epoch, Step, I: integer;
    Pair: TNNetVolumePair;
    Order: array of integer;
  begin
    SetLength(Order, TrainingPairs.Count);
    for I := 0 to High(Order) do Order[I] := I;

    NN.SetLearningRate(LearningRate, {Momentum=}0.9);
    NN.SetL2Decay(0.0);

    for Epoch := 1 to NumEpochs do
    begin
      ShuffleIndices(Order);
      Step := 0;
      NN.ClearDeltas();
      for I := 0 to TrainingPairs.Count - 1 do
      begin
        Pair := TrainingPairs[Order[I]];
        NN.Compute(Pair.I);
        NN.Backpropagate(Pair.O);
        Inc(Step);
        if Step = BatchSize then
        begin
          NN.UpdateWeights();
          NN.ClearDeltas();
          Step := 0;
        end;
      end;
      if Step > 0 then
      begin
        NN.UpdateWeights();
        NN.ClearDeltas();
      end;

      if Verbose and ((Epoch = 1) or (Epoch mod 100 = 0) or (Epoch = NumEpochs)) then
        WriteLn('    Epoch ', Epoch:4, '  clean-grid MSE = ', CleanGridMSE(NN):0:6);
    end;

    Result := CleanGridMSE(NN);
  end;

  procedure PrintPredTable(NNraw, NNfourier: TNNet);
  var
    I: integer;
    X, Tgt, PRaw, PFour: TNeuralFloat;
    Input, Output: TNNetVolume;
  begin
    WriteLn;
    WriteLn('Predicted vs truth on 11 evenly-spaced points in [-1, 1]:');
    WriteLn('       x          truth         raw-MLP      fourier-MLP');
    Input := TNNetVolume.Create([TNeuralFloat(0)]);
    Output := TNNetVolume.Create(1, 1, 1);
    for I := 0 to 10 do
    begin
      X := -1.0 + I / 5.0;
      Tgt := Target(X);
      Input.FData[0] := X;
      NNraw.Compute(Input);
      NNraw.GetOutput(Output);
      PRaw := Output.Raw[0];
      NNfourier.Compute(Input);
      NNfourier.GetOutput(Output);
      PFour := Output.Raw[0];
      WriteLn(X:8:4, Tgt:15:6, PRaw:15:6, PFour:15:6);
    end;
    Input.Free;
    Output.Free;
  end;

  procedure RunAlgo();
  var
    TrainingPairs: TNNetVolumePairList;
    NNraw, NNfourier, NNsweep: TNNet;
    RawMSE, FourierMSE, SweepMSE: TNeuralFloat;
    Sigmas: array[0..3] of TNeuralFloat;
    S: integer;
  begin
    RandSeed := 42;
    TrainingPairs := CreateTargetPairList(NumTrain);

    WriteLn('Fourier Features and Spectral Bias (Tancik et al. 2020)');
    WriteLn('Target: y = sin(20*x) + 0.5*sin(53*x) on x in [-1, 1]');
    WriteLn('Train samples=', NumTrain, '  hidden width=', HiddenWidth,
      '  epochs=', NumEpochs, '  lr=', LearningRate:0:4);
    WriteLn;

    // -- Headline: raw scalar x vs Fourier front-end, identical head. --------
    WriteLn('[1] RAW coordinate MLP: Input(1) -> ReLU(', HiddenWidth,
      ') -> ReLU(', HiddenWidth, ') -> Linear(1)');
    RandSeed := 42;
    NNraw := BuildNet({UseFourier=}False, 0.0);
    RawMSE := TrainNet(NNraw, TrainingPairs, {Verbose=}True);

    WriteLn;
    WriteLn('[2] FOURIER coordinate MLP: Input(1) -> FourierFeatures(M=',
      NumFeatures, ', sigma=', DefaultSigma:0:1, ') -> same ReLU head');
    RandSeed := 42;
    NNfourier := BuildNet({UseFourier=}True, DefaultSigma);
    FourierMSE := TrainNet(NNfourier, TrainingPairs, {Verbose=}True);

    WriteLn;
    WriteLn('=== HEADLINE: final clean-grid MSE ===');
    WriteLn('  raw-coordinate MLP   MSE = ', RawMSE:0:6);
    WriteLn('  fourier-feature MLP  MSE = ', FourierMSE:0:6);
    if FourierMSE > 0 then
      WriteLn('  improvement factor (raw / fourier) = ',
        (RawMSE / FourierMSE):0:2, 'x');

    PrintPredTable(NNraw, NNfourier);

    NNraw.Free;
    NNfourier.Free;

    // -- Sigma bandwidth sweep. ---------------------------------------------
    Sigmas[0] := 0.5;
    Sigmas[1] := 2.0;
    Sigmas[2] := 8.0;
    Sigmas[3] := 32.0;

    WriteLn;
    WriteLn('=== SIGMA BANDWIDTH SWEEP (Fourier front-end, M=', NumFeatures, ') ===');
    WriteLn('  too small = low-pass / underfit;  too large = noisy / overfit');
    WriteLn('     sigma        final MSE');
    for S := 0 to High(Sigmas) do
    begin
      RandSeed := 42;
      NNsweep := BuildNet({UseFourier=}True, Sigmas[S]);
      SweepMSE := TrainNet(NNsweep, TrainingPairs, {Verbose=}False);
      WriteLn(Sigmas[S]:10:1, SweepMSE:15:6);
      NNsweep.Free;
    end;

    TrainingPairs.Free;
  end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'Fourier Features Spectral Bias Example';
  RunAlgo();
end.
