program LearningRateFinder;
(*
LearningRateFinder: Leslie Smith's LR-range test on the hypotenuse toy.

The program trains a tiny MLP (2 -> 16 -> 16 -> 1) on y = sqrt(x1^2 + x2^2),
with x1, x2 ~ U(0, 1).  Across ~100 mini-batches the learning rate is swept
exponentially from 1e-6 to 1e+1, the per-step loss is recorded and smoothed
with an exponential moving average (beta = 0.98), and an ASCII chart of
log10(LR) vs smoothed loss is printed.  The steepest negative slope on the
smoothed curve is highlighted with '*' and reported as the suggested LR.

This is a CPU-only, single-threaded demo; it should finish in a few seconds.

Copyright (C) 2026 Joao Paulo Schwarz Schuler
Released under the GNU General Public License v2 or later.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes,
  SysUtils,
  Math,
  neuralnetwork,
  neuralvolume;

const
  NUM_STEPS        = 100;
  BATCH_SIZE       = 32;
  LR_MIN           = 1e-6;
  LR_MAX           = 1e+1;
  EMA_BETA         = 0.98;
  CHART_WIDTH      = 60;
  DIVERGENCE_RATIO = 4.0;  // stop chart early if loss explodes 4x best

procedure RunLRFinder();
var
  NN: TNNet;
  Input, Target, Output: TNNetVolume;
  Step, B, Row, Col, BestIdx, SuggestIdx: Integer;
  LR, LogLR, Mult, Beta, BatchLoss, Diff: TNeuralFloat;
  EmaLoss, BiasCorr, Smoothed, BestSmoothed: TNeuralFloat;
  X1, X2, Y: TNeuralFloat;
  LRs, RawLosses, SmoothedLosses: array of TNeuralFloat;
  Active: array of Boolean;
  CsvFile: TextFile;
  MinSm, MaxSm, Frac, Slope, BestSlope: TNeuralFloat;
  ChartLine: string;
  StarCol: Integer;
  StepsActive: Integer;
begin
  RandSeed := 25557;
  Randomize;

  // Build the tiny MLP: 2 -> 16 -> 16 -> 1.
  NN := TNNet.Create();
  NN.AddLayer([
    TNNetInput.Create(2),
    TNNetFullConnectReLU.Create(16),
    TNNetFullConnectReLU.Create(16),
    TNNetFullConnectLinear.Create(1)
  ]);
  NN.SetLearningRate(LR_MIN, 0.0);
  NN.SetL2Decay(0.0);

  Input  := TNNetVolume.Create(2, 1, 1);
  Target := TNNetVolume.Create(1, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);

  SetLength(LRs, NUM_STEPS);
  SetLength(RawLosses, NUM_STEPS);
  SetLength(SmoothedLosses, NUM_STEPS);
  SetLength(Active, NUM_STEPS);

  Mult := Exp( Ln(LR_MAX / LR_MIN) / (NUM_STEPS - 1) );
  LR := LR_MIN;
  EmaLoss := 0.0;
  BestSmoothed := MaxSingle;
  StepsActive := 0;

  AssignFile(CsvFile, 'lr_finder.csv');
  Rewrite(CsvFile);
  WriteLn(CsvFile, 'step,log10_lr,raw_loss,smoothed_loss');

  WriteLn('Leslie Smith LR-range test');
  WriteLn('  network : 2 -> 16 -> 16 -> 1 (ReLU hidden, Linear head)');
  WriteLn('  task    : y = sqrt(x1^2 + x2^2), x1,x2 ~ U(0,1)');
  WriteLn('  sweep   : LR ', LR_MIN:0:8, ' -> ', LR_MAX:0:4,
          ' over ', NUM_STEPS, ' mini-batches of ', BATCH_SIZE);
  WriteLn('  ema beta: ', EMA_BETA:0:4);
  WriteLn;
  WriteLn(Format('%5s  %10s  %12s  %12s', ['step', 'log10(LR)', 'raw_loss', 'smooth_loss']));

  for Step := 0 to NUM_STEPS - 1 do
  begin
    NN.SetLearningRate(LR, 0.0);
    NN.ClearDeltas();

    BatchLoss := 0.0;
    for B := 1 to BATCH_SIZE do
    begin
      X1 := Random;
      X2 := Random;
      Y  := Sqrt(X1*X1 + X2*X2);

      Input.FData[0]  := X1;
      Input.FData[1]  := X2;
      Target.FData[0] := Y;

      NN.Compute(Input);
      NN.GetOutput(Output);
      Diff := Output.FData[0] - Y;
      BatchLoss := BatchLoss + Diff * Diff;

      NN.Backpropagate(Target);
    end;
    BatchLoss := BatchLoss / BATCH_SIZE;
    NN.UpdateWeights();

    // EMA with bias correction (Leslie Smith / fast.ai recipe).
    Beta := EMA_BETA;
    EmaLoss := Beta * EmaLoss + (1 - Beta) * BatchLoss;
    BiasCorr := 1 - Power(Beta, Step + 1);
    Smoothed := EmaLoss / BiasCorr;

    LogLR := Log10(LR);
    LRs[Step]            := LR;
    RawLosses[Step]      := BatchLoss;
    SmoothedLosses[Step] := Smoothed;

    if IsNan(Smoothed) or IsInfinite(Smoothed) then
    begin
      Active[Step] := False;
    end
    else
    begin
      Active[Step] := True;
      Inc(StepsActive);
      if Smoothed < BestSmoothed then BestSmoothed := Smoothed;
    end;

    WriteLn(Format('%5d  %10.4f  %12.6f  %12.6f',
      [Step, LogLR, BatchLoss, Smoothed]));
    WriteLn(CsvFile, Format('%d,%.6f,%.6f,%.6f',
      [Step, LogLR, BatchLoss, Smoothed]));

    // Diverged: stop running but keep printed steps for the chart.
    if not Active[Step] then
    begin
      WriteLn('  [diverged: loss became NaN/Inf; stopping sweep]');
      Break;
    end;
    if (Step > 5) and (Smoothed > DIVERGENCE_RATIO * BestSmoothed) then
    begin
      WriteLn('  [diverged: smoothed loss exceeds ',
        DIVERGENCE_RATIO:0:1, 'x best; stopping sweep]');
      Break;
    end;

    LR := LR * Mult;
  end;

  CloseFile(CsvFile);

  // Compute chart bounds from the active rows only.  Clip points that
  // already exceed the divergence threshold so the chart isn't squashed
  // by a single exploding step.
  MinSm :=  MaxSingle;
  MaxSm := -MaxSingle;
  for Row := 0 to NUM_STEPS - 1 do
  begin
    if not Active[Row] then Continue;
    if SmoothedLosses[Row] > DIVERGENCE_RATIO * BestSmoothed then Continue;
    if SmoothedLosses[Row] < MinSm then MinSm := SmoothedLosses[Row];
    if SmoothedLosses[Row] > MaxSm then MaxSm := SmoothedLosses[Row];
  end;
  if MaxSm <= MinSm then MaxSm := MinSm + 1e-6;

  // Find steepest negative slope on the smoothed curve (suggested LR).
  BestSlope  :=  MaxSingle;  // most negative wins
  SuggestIdx := -1;
  BestIdx    := -1;
  // Skip a short warm-up so the bias-corrected EMA stabilises before we
  // pick the steepest-descent point.
  for Row := 10 to NUM_STEPS - 1 do
  begin
    if not (Active[Row] and Active[Row - 1]) then Continue;
    Slope := (SmoothedLosses[Row] - SmoothedLosses[Row - 1]) /
             (Log10(LRs[Row]) - Log10(LRs[Row - 1]));
    if Slope < BestSlope then
    begin
      BestSlope := Slope;
      BestIdx   := Row;
    end;
  end;
  // Leslie Smith's heuristic: pick the LR a bit before the minimum
  // (i.e. the steepest descent), not the absolute min.
  SuggestIdx := BestIdx;

  WriteLn;
  WriteLn('ASCII chart  (x = log10 LR, y = smoothed loss, ', CHART_WIDTH, ' cols)');
  WriteLn('  loss range: ', MinSm:0:6, ' .. ', MaxSm:0:6);
  WriteLn('  ', StringOfChar('-', CHART_WIDTH + 18));
  for Row := 0 to NUM_STEPS - 1 do
  begin
    if not Active[Row] then Continue;
    if SmoothedLosses[Row] > DIVERGENCE_RATIO * BestSmoothed then Continue;
    Frac := (SmoothedLosses[Row] - MinSm) / (MaxSm - MinSm);
    if Frac < 0 then Frac := 0;
    if Frac > 1 then Frac := 1;
    Col := Round(Frac * (CHART_WIDTH - 1));
    SetLength(ChartLine, CHART_WIDTH);
    FillChar(ChartLine[1], CHART_WIDTH, ' ');
    StarCol := -1;
    if Row = SuggestIdx then StarCol := Col;
    for B := 1 to CHART_WIDTH do
      if B - 1 = Col then ChartLine[B] := '#'
      else                 ChartLine[B] := ' ';
    if StarCol >= 0 then ChartLine[StarCol + 1] := '*';
    WriteLn(Format('  %7.3f |%s| %.4f',
      [Log10(LRs[Row]), ChartLine, SmoothedLosses[Row]]));
  end;
  WriteLn('  ', StringOfChar('-', CHART_WIDTH + 18));
  WriteLn('  legend: ''#'' = smoothed loss   ''*'' = steepest descent');
  WriteLn;
  if SuggestIdx >= 0 then
  begin
    WriteLn(Format('Suggested LR (steepest descent): %.6g  (log10 = %.3f, slope = %.4f / decade)',
      [LRs[SuggestIdx], Log10(LRs[SuggestIdx]), BestSlope]));
  end
  else
  begin
    WriteLn('Suggested LR (steepest descent): could not determine from this sweep.');
  end;
  WriteLn('CSV written to lr_finder.csv (', StepsActive, ' active rows).');

  Input.Free;
  Target.Free;
  Output.Free;
  NN.Free;
end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'Learning Rate Finder Example';
  // Allow weights to overflow into Inf/NaN at high LRs without aborting.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RunLRFinder();
end.
