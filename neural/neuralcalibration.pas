(*
neuralcalibration
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).
*)

unit neuralcalibration;

(*
Forward-only model-calibration / reliability diagnostics for a trained
classifier. Given a network, a list of input volumes and their integer
labels, this unit measures *how well the model's confidence matches its
accuracy*:

  - Expected Calibration Error (ECE): the sample-weighted average over
    confidence bins of |accuracy(bin) - confidence(bin)|. 0 == perfect.
  - Maximum Calibration Error (MCE): the worst bin gap.
  - Brier score: mean squared error between the predicted probability
    vector and the one-hot label (a proper scoring rule).
  - A reliability diagram: per-bin (mean confidence, accuracy, count),
    rendered as an ASCII chart and dumped as a P2 (ASCII) PGM image.
  - Temperature scaling: a single scalar T that, dividing the logits,
    minimises validation NLL (Guo et al. 2017). Found by a coarse-then-
    fine 1-D grid scan over T in [0.5, 5.0] -- no autograd needed.

LOGIT ASSUMPTION
================
Calibration metrics and temperature scaling need *pre-softmax logits*.
A net that ends in TNNetSoftMax only exposes probabilities, so we read
the final-layer output and reconstruct pseudo-logits as the elementwise
log of the probability vector: z_i := ln(max(p_i, eps)). Up to an
additive constant (which softmax is invariant to) this recovers the true
logits whenever the head is a plain softmax over a linear layer, so
softmax(z / T) is exactly temperature scaling on the original logits.
If the net ends in a non-softmax layer the output is treated as raw
logits directly. Either way the backbone is never trained, never
back-propagated through and its weights are never mutated -- this is a
pure forward-only measurement.
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math,
  neuralvolume, neuralnetwork;

type
  // Holds the scalar metrics plus the per-bin reliability-diagram arrays so
  // callers can post-process (re-plot, export, threshold, ...). All arrays
  // have BinCount entries; bin b covers confidence range
  // [b/BinCount, (b+1)/BinCount).
  TNeuralCalibrationReport = record
    BinCount:   integer;
    NumSamples: integer;
    NumClasses: integer;
    ECE:        TNeuralFloat;   // Expected Calibration Error in [0, 1]
    MCE:        TNeuralFloat;   // Maximum Calibration Error in [0, 1]
    Brier:      TNeuralFloat;   // Brier score (>= 0)
    Accuracy:   TNeuralFloat;   // overall top-1 accuracy in [0, 1]
    BinCount_:  array of integer;       // samples in each bin
    BinConf:    array of TNeuralFloat;  // mean top-1 confidence per bin
    BinAcc:     array of TNeuralFloat;  // accuracy (frac correct) per bin
  end;

// Computes the calibration metrics + reliability diagram for NN over the
// given inputs/labels. Forward-only. Result.NumSamples = 0 signals an empty
// or invalid input (the caller can inspect it directly).
function ComputeCalibration(
  NN: TNNet;
  Inputs: TNNetVolumeList;
  const Labels: array of integer;
  BinCount: integer): TNeuralCalibrationReport;

// Primary entry point: an ASCII summary (metrics + reliability diagram).
// Guards nil NN / empty set gracefully (returns a one-line message).
function CalibrationReport(
  NN: TNNet;
  Inputs: TNNetVolumeList;
  const Labels: array of integer;
  BinCount: integer = 10): string;

// Fits the scalar temperature T in [0.5, 5.0] that minimises validation NLL
// when the (pseudo-)logits are divided by T. Coarse-then-fine grid scan.
// Returns 1.0 on invalid input (a no-op temperature). Forward-only.
function FitTemperature(
  NN: TNNet;
  Inputs: TNNetVolumeList;
  const Labels: array of integer): TNeuralFloat;

// Writes a P2 (ASCII) PGM reliability diagram: per-bin accuracy bars plotted
// against the y=x perfect-calibration reference line. Returns True on success.
function WriteReliabilityPGM(
  const Report: TNeuralCalibrationReport;
  const FileName: string): boolean;

implementation

const
  cEps = 1e-9;

// Reads NN's final-layer output for Input and fills Probs with a proper
// probability distribution and Logits with pseudo-logits (ln of probs). When
// the output is already a probability vector (softmax head) Probs is a copy;
// otherwise the raw output is softmaxed into Probs and used as logits.
procedure ForwardProbsAndLogits(
  NN: TNNet; Input: TNNetVolume;
  out Probs: array of TNeuralFloat;
  out Logits: array of TNeuralFloat);
var
  Output: TNNetVolume;
  I, N, NM1: integer;
  Sum, MaxV, Acc, V, AccClamped: TNeuralFloat;
  IsProb: boolean;
begin
  NN.Compute(Input);
  Output := NN.GetLastLayer().Output;
  N := Output.Size;
  NM1 := N - 1;
  // Detect whether the output already looks like a probability simplex
  // (non-negative entries summing to ~1) -> softmax head.
  Sum := 0;
  IsProb := True;
  for I := 0 to NM1 do
  begin
    if Output.FData[I] < -cEps then IsProb := False;
    Sum := Sum + Output.FData[I];
  end;
  if Abs(Sum - 1.0) > 1e-3 then IsProb := False;

  if IsProb then
  begin
    for I := 0 to NM1 do
    begin
      Probs[I]  := Max(cEps, Output.FData[I]);
      Logits[I] := Ln(Probs[I]); // pseudo-logits, see unit header
    end;
  end
  else
  begin
    // Treat raw output as logits; softmax it into Probs.
    MaxV := Output.FData[0];
    for I := 1 to NM1 do
      if Output.FData[I] > MaxV then MaxV := Output.FData[I];
    Acc := 0;
    for I := 0 to NM1 do
    begin
      Logits[I] := Output.FData[I];
      V := Exp(Output.FData[I] - MaxV);
      Probs[I] := V;
      Acc := Acc + V;
    end;
    AccClamped := Max(cEps, Acc);
    for I := 0 to NM1 do
      Probs[I] := Probs[I] / AccClamped;
  end;
end;

// Softmax of Logits/T into Probs.
procedure SoftmaxTemp(
  const Logits: array of TNeuralFloat; T: TNeuralFloat;
  out Probs: array of TNeuralFloat; N: integer);
var
  I, NM1: integer;
  MaxV, Acc, MaxRaw, V, AccClamped: TNeuralFloat;
begin
  NM1 := N - 1;
  MaxRaw := Logits[0];
  for I := 1 to NM1 do
    if Logits[I] > MaxRaw then MaxRaw := Logits[I];
  MaxV := MaxRaw / T;
  Acc := 0;
  for I := 0 to NM1 do
  begin
    V := Exp(Logits[I] / T - MaxV);
    Probs[I] := V;
    Acc := Acc + V;
  end;
  AccClamped := Max(cEps, Acc);
  for I := 0 to NM1 do
    Probs[I] := Probs[I] / AccClamped;
end;

function ComputeCalibration(
  NN: TNNet;
  Inputs: TNNetVolumeList;
  const Labels: array of integer;
  BinCount: integer): TNeuralCalibrationReport;
var
  I, B, N, Total, InputCount: integer;
  NM1, BinCountM1, InputCountM1: integer;
  Probs, Logits: array of TNeuralFloat;
  PredClass, TrueClass, J: integer;
  Conf, BrierAcc, Diff: TNeuralFloat;
  Correct: integer;
  BinSumConf, BinSumAcc: array of TNeuralFloat;
begin
  Result.BinCount   := BinCount;
  Result.NumSamples := 0;
  Result.NumClasses := 0;
  Result.ECE := 0; Result.MCE := 0; Result.Brier := 0; Result.Accuracy := 0;
  SetLength(Result.BinCount_, 0);
  SetLength(Result.BinConf, 0);
  SetLength(Result.BinAcc, 0);

  if BinCount <= 0 then Exit;
  if NN = nil then Exit;
  if (Inputs = nil) or (Inputs.Count = 0) then Exit;

  N := NN.GetLastLayer().Output.Size;
  if N <= 0 then Exit;
  NM1 := N - 1;
  BinCountM1 := BinCount - 1;
  Result.NumClasses := N;

  SetLength(Probs, N);
  SetLength(Logits, N);
  SetLength(Result.BinCount_, BinCount);
  SetLength(Result.BinConf, BinCount);
  SetLength(Result.BinAcc, BinCount);
  SetLength(BinSumConf, BinCount);
  SetLength(BinSumAcc, BinCount);
  for B := 0 to BinCountM1 do
  begin
    Result.BinCount_[B] := 0;
    BinSumConf[B] := 0;
    BinSumAcc[B] := 0;
  end;

  Total := 0;
  Correct := 0;
  BrierAcc := 0;
  InputCount := Inputs.Count;
  InputCountM1 := InputCount - 1;
  for I := 0 to InputCountM1 do
  begin
    if Inputs[I] = nil then Continue;
    if I >= Length(Labels) then Break;
    TrueClass := Labels[I];
    if (TrueClass < 0) or (TrueClass >= N) then Continue;

    ForwardProbsAndLogits(NN, Inputs[I], Probs, Logits);

    // top-1 prediction + confidence.
    PredClass := 0;
    Conf := Probs[0];
    for J := 1 to NM1 do
      if Probs[J] > Conf then
      begin
        Conf := Probs[J];
        PredClass := J;
      end;

    // Brier: sum over classes of (p_j - onehot_j)^2.
    for J := 0 to NM1 do
    begin
      if J = TrueClass then Diff := Probs[J] - 1.0
      else Diff := Probs[J];
      BrierAcc := BrierAcc + Diff * Diff;
    end;

    Inc(Total);
    if PredClass = TrueClass then Inc(Correct);

    // bin by top-1 confidence; clamp the p=1.0 edge into the last bin.
    B := Trunc(Conf * BinCount);
    if B >= BinCount then B := BinCount - 1;
    if B < 0 then B := 0;
    Inc(Result.BinCount_[B]);
    BinSumConf[B] := BinSumConf[B] + Conf;
    if PredClass = TrueClass then BinSumAcc[B] := BinSumAcc[B] + 1.0;
  end;

  Result.NumSamples := Total;
  if Total = 0 then Exit;

  Result.Accuracy := Correct / Total;
  Result.Brier := BrierAcc / Total;

  Result.ECE := 0;
  Result.MCE := 0;
  for B := 0 to BinCountM1 do
  begin
    if Result.BinCount_[B] > 0 then
    begin
      Result.BinConf[B] := BinSumConf[B] / Result.BinCount_[B];
      Result.BinAcc[B]  := BinSumAcc[B] / Result.BinCount_[B];
      Diff := Abs(Result.BinAcc[B] - Result.BinConf[B]);
      Result.ECE := Result.ECE + (Result.BinCount_[B] / Total) * Diff;
      if Diff > Result.MCE then Result.MCE := Diff;
    end
    else
    begin
      Result.BinConf[B] := 0;
      Result.BinAcc[B]  := 0;
    end;
  end;
end;

function CalibrationReport(
  NN: TNNet;
  Inputs: TNNetVolumeList;
  const Labels: array of integer;
  BinCount: integer = 10): string;
var
  Lines: TStringList;
  R: TNeuralCalibrationReport;
  B, BarW, NConf, NAcc, RBinCountM1: integer;
  Lo, Hi: TNeuralFloat;
begin
  Result := '';
  if NN = nil then
  begin
    Result := 'CalibrationReport: NN is nil.' + sLineBreak;
    Exit;
  end;
  if (Inputs = nil) or (Inputs.Count = 0) then
  begin
    Result := 'CalibrationReport: empty input list.' + sLineBreak;
    Exit;
  end;
  if BinCount <= 0 then
  begin
    Result := 'CalibrationReport: BinCount must be > 0.' + sLineBreak;
    Exit;
  end;

  R := ComputeCalibration(NN, Inputs, Labels, BinCount);
  if R.NumSamples = 0 then
  begin
    Result := 'CalibrationReport: no valid (input, label) pairs found.' +
      sLineBreak;
    Exit;
  end;

  Lines := TStringList.Create();
  try
    Lines.Add('CalibrationReport');
    Lines.Add(Format('Samples: %d   Classes: %d   Bins: %d',
      [R.NumSamples, R.NumClasses, R.BinCount]));
    Lines.Add('');
    Lines.Add('(a) Scalar metrics');
    Lines.Add(Format('  Top-1 accuracy           : %8.4f', [R.Accuracy]));
    Lines.Add(Format('  Expected Calib. Error ECE: %8.4f', [R.ECE]));
    Lines.Add(Format('  Maximum  Calib. Error MCE: %8.4f', [R.MCE]));
    Lines.Add(Format('  Brier score              : %8.4f', [R.Brier]));
    Lines.Add('');
    Lines.Add('(b) Reliability diagram (per confidence bin)');
    Lines.Add('  bin   range        count     conf      acc   gap');
    // ASCII chart: two bars per bin -- conf (-) and acc (#) -- over a 0..1
    // width-BarW axis. Where acc < conf the model is over-confident.
    BarW := 24;
    RBinCountM1 := R.BinCount - 1;
    for B := 0 to RBinCountM1 do
    begin
      Lo := B / R.BinCount;
      Hi := (B + 1) / R.BinCount;
      if R.BinCount_[B] > 0 then
      begin
        NConf := Round(R.BinConf[B] * BarW);
        NAcc  := Round(R.BinAcc[B] * BarW);
        if NConf > BarW then NConf := BarW;
        if NAcc  > BarW then NAcc  := BarW;
        Lines.Add(Format('  %3d [%4.2f,%4.2f] %6d  %7.4f  %7.4f  %8.4f',
          [B, Lo, Hi, R.BinCount_[B], R.BinConf[B], R.BinAcc[B],
           R.BinAcc[B] - R.BinConf[B]]));
        Lines.Add('        acc |' + StringOfChar('#', NAcc));
        Lines.Add('        cnf |' + StringOfChar('-', NConf));
      end
      else
        Lines.Add(Format('  %3d [%4.2f,%4.2f] %6d  (empty)',
          [B, Lo, Hi, 0]));
    end;
    Lines.Add('');
    Lines.Add('  Legend: # = bin accuracy, - = bin mean confidence.');
    Lines.Add('  Over-confident bins have acc (#) shorter than conf (-).');
    Result := Lines.Text;
  finally
    Lines.Free;
  end;
end;

function FitTemperature(
  NN: TNNet;
  Inputs: TNNetVolumeList;
  const Labels: array of integer): TNeuralFloat;
var
  N, I, J, Step, TrueClass, Total, InputCount: integer;
  NM1, InputCountM1, TotalM1: integer;
  Logits, Probs: array of TNeuralFloat;
  // cached pseudo-logits for every valid sample (one forward pass total).
  AllLogits: array of array of TNeuralFloat;
  AllLabel: array of integer;
  T, BestT, NLL, BestNLL, Lo, Hi, GridStep: TNeuralFloat;
begin
  Result := 1.0;
  if NN = nil then Exit;
  if (Inputs = nil) or (Inputs.Count = 0) then Exit;
  N := NN.GetLastLayer().Output.Size;
  if N <= 0 then Exit;
  NM1 := N - 1;

  InputCount := Inputs.Count;
  InputCountM1 := InputCount - 1;
  SetLength(Logits, N);
  SetLength(Probs, N);
  SetLength(AllLogits, InputCount);
  SetLength(AllLabel, InputCount);

  // Single forward pass over the set; cache logits so the grid scan is pure
  // arithmetic (the backbone is touched exactly once and never mutated).
  Total := 0;
  for I := 0 to InputCountM1 do
  begin
    if Inputs[I] = nil then Continue;
    if I >= Length(Labels) then Break;
    TrueClass := Labels[I];
    if (TrueClass < 0) or (TrueClass >= N) then Continue;
    ForwardProbsAndLogits(NN, Inputs[I], Probs, Logits);
    SetLength(AllLogits[Total], N);
    Move(Logits[0], AllLogits[Total][0], N * csNeuralFloatSize);
    AllLabel[Total] := TrueClass;
    Inc(Total);
  end;
  if Total = 0 then Exit;
  TotalM1 := Total - 1;

  // Coarse-then-fine grid scan over T in [0.5, 5.0], minimising mean NLL.
  Lo := 0.5;
  Hi := 5.0;
  BestT := 1.0;
  BestNLL := MaxSingle;
  for Step := 0 to 1 do
  begin
    GridStep := (Hi - Lo) / 40;
    if GridStep < 1e-4 then GridStep := 1e-4;
    T := Lo;
    while T <= Hi + 1e-9 do
    begin
      NLL := 0;
      for I := 0 to TotalM1 do
      begin
        SoftmaxTemp(AllLogits[I], T, Probs, N);
        NLL := NLL - Ln(Max(cEps, Probs[AllLabel[I]]));
      end;
      NLL := NLL / Total;
      if NLL < BestNLL then
      begin
        BestNLL := NLL;
        BestT := T;
      end;
      T := T + GridStep;
    end;
    // Refine around the current best for the second pass.
    Lo := Max(0.5, BestT - GridStep);
    Hi := Min(5.0, BestT + GridStep);
  end;

  Result := BestT;
end;

function WriteReliabilityPGM(
  const Report: TNeuralCalibrationReport;
  const FileName: string): boolean;
const
  cW = 256;   // image width  (pixels)
  cH = 256;   // image height (pixels)
  cMax = 255; // PGM max gray value
  cWM1 = cW - 1;
  cHM1 = cH - 1;
var
  F: TextFile;
  Img: array of array of integer; // [row][col], 0=black .. 255=white
  X, Y, B, Col, BarTop, RefRow: integer;
  BinW, BinWM1, ReportBinCountM1: integer;
  Acc: TNeuralFloat;
  Line: string;
begin
  Result := False;
  if Report.BinCount <= 0 then Exit;

  SetLength(Img, cH, cW);
  for Y := 0 to cHM1 do
    for X := 0 to cWM1 do
      Img[Y][X] := cMax; // white background

  BinW := cW div Report.BinCount;
  if BinW < 1 then BinW := 1;
  BinWM1 := BinW - 1;

  // y = x reference (perfect-calibration) diagonal, mid-gray.
  for X := 0 to cWM1 do
  begin
    RefRow := (cH - 1) - Round((X / (cW - 1)) * (cH - 1));
    if (RefRow >= 0) and (RefRow < cH) then Img[RefRow][X] := 128;
  end;

  // Per-bin accuracy bars in dark gray, anchored at the bottom.
  ReportBinCountM1 := Report.BinCount - 1;
  for B := 0 to ReportBinCountM1 do
  begin
    if Report.BinCount_[B] <= 0 then Continue;
    Acc := Report.BinAcc[B];
    if Acc < 0 then Acc := 0;
    if Acc > 1 then Acc := 1;
    BarTop := (cH - 1) - Round(Acc * (cH - 1));
    for Col := 0 to BinWM1 do
    begin
      X := B * BinW + Col;
      if X >= cW then Break;
      for Y := BarTop to cHM1 do
        if Img[Y][X] = cMax then Img[Y][X] := 64
        else Img[Y][X] := 32; // darker where the bar crosses the diagonal
    end;
  end;

  AssignFile(F, FileName);
  try
    Rewrite(F);
    WriteLn(F, 'P2');
    WriteLn(F, '# reliability diagram: per-bin accuracy bars vs y=x reference');
    WriteLn(F, cW, ' ', cH);
    WriteLn(F, cMax);
    for Y := 0 to cHM1 do
    begin
      Line := '';
      for X := 0 to cWM1 do
      begin
        if Line <> '' then Line := Line + ' ';
        Line := Line + IntToStr(Img[Y][X]);
      end;
      WriteLn(F, Line);
    end;
    CloseFile(F);
    Result := True;
  except
    Result := False;
  end;
end;

end.
