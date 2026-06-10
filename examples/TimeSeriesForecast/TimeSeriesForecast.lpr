program TimeSeriesForecast;
(*
TimeSeriesForecast: a one-screen univariate time-series forecasting demo on a
SYNTHETIC seasonal + trend + noise signal (generated in-code; no data files).

Signal:
  x[t] = linear_trend(t)
       + seasonal_a * sin(2*pi*t / period_a)
       + seasonal_b * sin(2*pi*t / period_b + phase)
       + small Gaussian noise

Model: a small causal 1-D convolutional stack that maps a sliding window of the
last cWindow samples to a one-step-ahead prediction:
  Input (cWindow,1,1)
    -> TNNetCausalConv1D(16, k=3)  + ReLU      (local patterns)
    -> TNNetCausalConv1D(16, k=3, dilation=2) + ReLU (wider receptive field)
    -> TNNetPointwiseConvLinear(8) + ReLU      (channel mix per time step)
    -> the LAST time step's features are read out by a linear head to 1 value.
We read the last position of the sequence (the only one allowed to see the whole
window causally) and regress the next sample. A multi-step forecast is produced
auto-regressively by feeding each prediction back into the window.

The series is split chronologically into train / validation. We report the
train and validation MSE every few epochs (a short loss curve), then roll out a
multi-step forecast over the held-out tail and print forecast-vs-truth for a few
horizon points plus the horizon MAE and RMSE.

Pure CPU, tiny dims, finishes in well under a minute on 2 cores.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSeriesLen   = 900;   // total synthetic samples
  cWindow      = 32;    // sliding input window length
  cTrainEnd    = 700;   // first cTrainEnd samples are train; rest is val/test
  cHorizon     = 24;    // multi-step forecast horizon (auto-regressive)
  cEpochs      = 40;
  cLearnRate   = 0.001;

  cTrendSlope  = 0.004;
  cSeasonA     = 1.0;
  cPeriodA     = 40.0;
  cSeasonB     = 0.5;
  cPeriodB     = 13.0;
  cPhaseB      = 0.7;
  cNoiseStd    = 0.05;

var
  // Raw series and the normalisation stats used to standardise it.
  Series: array[0..cSeriesLen - 1] of TNeuralFloat;
  Norm:   array[0..cSeriesLen - 1] of TNeuralFloat; // standardised series
  Mean, Std: TNeuralFloat;

function RandomGauss(): TNeuralFloat;
var U1, U2: TNeuralFloat;
begin
  repeat U1 := Random; until U1 > 1e-9;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

procedure BuildSeries();
var t: integer; s: TNeuralFloat;
begin
  for t := 0 to cSeriesLen - 1 do
  begin
    s := cTrendSlope * t
       + cSeasonA * Sin(2 * Pi * t / cPeriodA)
       + cSeasonB * Sin(2 * Pi * t / cPeriodB + cPhaseB)
       + cNoiseStd * RandomGauss();
    Series[t] := s;
  end;
end;

// Standardise using TRAIN statistics only (avoid leaking val/test scale).
procedure Standardise();
var t: integer; v: TNeuralFloat;
begin
  Mean := 0;
  for t := 0 to cTrainEnd - 1 do Mean := Mean + Series[t];
  Mean := Mean / cTrainEnd;
  Std := 0;
  for t := 0 to cTrainEnd - 1 do
  begin v := Series[t] - Mean; Std := Std + v * v; end;
  Std := Sqrt(Std / cTrainEnd);
  if Std < 1e-6 then Std := 1;
  for t := 0 to cSeriesLen - 1 do Norm[t] := (Series[t] - Mean) / Std;
end;

function BuildNet(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cWindow, 1, 1));
  Result.AddLayer(TNNetCausalConv1D.Create(16, 3));
  Result.AddLayer(TNNetReLU.Create());
  Result.AddLayer(TNNetCausalConv1D.Create(16, 3, 0, 2)); // dilation 2
  Result.AddLayer(TNNetReLU.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(8));
  Result.AddLayer(TNNetReLU.Create());
  // Collapse the whole window to one feature vector, then to a scalar.
  Result.AddLayer(TNNetFullConnectReLU.Create(16));
  Result.AddLayer(TNNetFullConnectLinear.Create(1));
end;

// Fill Input with the standardised window ending at position (endIdx-1), i.e.
// samples [endIdx-cWindow .. endIdx-1]; target is the standardised sample at endIdx.
procedure FillWindow(Input: TNNetVolume; endIdx: integer);
var k: integer;
begin
  for k := 0 to cWindow - 1 do
    Input[k, 0, 0] := Norm[endIdx - cWindow + k];
end;

function ComputeMSE(NN: TNNet; Input, Desired: TNNetVolume;
  startIdx, endIdx: integer): TNeuralFloat;
var i: integer; diff, acc: TNeuralFloat; n: integer;
begin
  acc := 0; n := 0;
  for i := startIdx to endIdx - 1 do
  begin
    FillWindow(Input, i);
    NN.Compute(Input);
    diff := NN.GetLastLayer.Output.FData[0] - Norm[i];
    acc := acc + diff * diff;
    Inc(n);
  end;
  if n = 0 then Result := 0 else Result := acc / n;
end;

var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  epoch, i, h, valStart: integer;
  trainMSE, valMSE: TNeuralFloat;
  // forecast rollout buffer (standardised)
  RollWin: array[0..cWindow - 1] of TNeuralFloat;
  pred, truth, err, mae, rmse: TNeuralFloat;
  predReal, truthReal: TNeuralFloat;
  fcStart: integer;
  fcInput: TNNetVolume;
begin
  RandSeed := 20260607;
  BuildSeries();
  Standardise();

  WriteLn('=== TimeSeriesForecast: causal-conv 1-step forecaster ===');
  WriteLn('series_len=', cSeriesLen, '  window=', cWindow,
          '  train_end=', cTrainEnd, '  horizon=', cHorizon);
  WriteLn('signal = trend + sin(period ', cPeriodA:0:0, ') + sin(period ',
          cPeriodB:0:0, ') + noise(std ', cNoiseStd:0:2, ')');
  WriteLn;

  NN := BuildNet();
  WriteLn('model params = ', NN.CountWeights());
  NN.SetLearningRate(cLearnRate, 0.9);
  NN.DebugWeights();
  WriteLn;

  Input   := TNNetVolume.Create(cWindow, 1, 1);
  Desired := TNNetVolume.Create(1, 1, 1);

  // first trainable target index is cWindow (needs a full window before it)
  valStart := cTrainEnd;

  WriteLn('training (', cEpochs, ' epochs)...');
  WriteLn('  epoch    train_MSE      val_MSE');
  for epoch := 1 to cEpochs do
  begin
    // one pass over the training targets [cWindow .. cTrainEnd-1]
    for i := cWindow to cTrainEnd - 1 do
    begin
      FillWindow(Input, i);
      Desired.FData[0] := Norm[i];
      NN.Compute(Input);
      NN.Backpropagate(Desired);
    end;
    if (epoch = 1) or (epoch mod 5 = 0) then
    begin
      trainMSE := ComputeMSE(NN, Input, Desired, cWindow, cTrainEnd);
      valMSE   := ComputeMSE(NN, Input, Desired, valStart, cSeriesLen);
      WriteLn('  ', epoch:5, '   ', trainMSE:10:6, '   ', valMSE:10:6);
    end;
  end;
  WriteLn;

  // ---------------- Multi-step auto-regressive forecast ----------------
  // Seed the rollout window with the real (standardised) samples ending right
  // before the forecast start, then predict cHorizon steps feeding predictions
  // back in. Compare against the true held-out samples.
  fcStart := cSeriesLen - cHorizon;          // forecast covers the final tail
  for i := 0 to cWindow - 1 do
    RollWin[i] := Norm[fcStart - cWindow + i];

  fcInput := TNNetVolume.Create(cWindow, 1, 1);
  mae := 0; rmse := 0;
  WriteLn('multi-step forecast vs truth (last ', cHorizon, ' samples, original scale):');
  WriteLn('  step      forecast        truth        abs_err');
  for h := 0 to cHorizon - 1 do
  begin
    for i := 0 to cWindow - 1 do fcInput[i, 0, 0] := RollWin[i];
    NN.Compute(fcInput);
    pred := NN.GetLastLayer.Output.FData[0];      // standardised prediction
    truth := Norm[fcStart + h];
    // slide window: drop oldest, append the prediction (auto-regressive)
    for i := 0 to cWindow - 2 do RollWin[i] := RollWin[i + 1];
    RollWin[cWindow - 1] := pred;
    // de-standardise for human-readable comparison
    predReal  := pred * Std + Mean;
    truthReal := truth * Std + Mean;
    err := Abs(predReal - truthReal);
    mae := mae + err;
    rmse := rmse + (predReal - truthReal) * (predReal - truthReal);
    if (h < 8) or (h = cHorizon - 1) then
      WriteLn('  ', (h + 1):4, '   ', predReal:12:5, ' ', truthReal:12:5,
              '   ', err:12:5);
  end;
  mae := mae / cHorizon;
  rmse := Sqrt(rmse / cHorizon);
  WriteLn;
  WriteLn('horizon MAE  = ', mae:0:5);
  WriteLn('horizon RMSE = ', rmse:0:5);
  WriteLn;
  // Naive persistence baseline (predict last value for the whole horizon).
  mae := 0;
  for h := 0 to cHorizon - 1 do
    mae := mae + Abs(Series[fcStart - 1] - Series[fcStart + h]);
  mae := mae / cHorizon;
  WriteLn('naive persistence MAE = ', mae:0:5, '  (last-value baseline)');

  fcInput.Free;
  Desired.Free;
  Input.Free;
  NN.Free;
end.
