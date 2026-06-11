program NeuralCollapse;
(*
NeuralCollapse: trains a small softmax classifier WELL PAST zero train-error (the
"terminal phase of training") on a synthetic few-class Gaussian-blob problem and
charts the four canonical Neural-Collapse metrics (Papyan, Han & Donoho 2020) via
TNNet.NeuralCollapseReport on a fixed, class-balanced probe set:

  NC1 within-class variability collapse tr(Sw.Sb^+)/C        -> 0
  NC2 simplex-ETF geometry: equinorm CV + equiangular cosine -> -1/(C-1)
  NC3 self-duality cos(centered class mean, classifier row)   -> 1
  NC4 classifier collapses to nearest-class-mean rule         -> 1

The headline (NC2) is the centered class means assembling themselves into a
simplex equiangular tight frame: every pair of centered class means converges to
the SAME angle, whose cosine is exactly -1/(C-1). The program calls the report
every N epochs on the fixed probe and prints an ASCII trajectory of the mean
pairwise cosine marching onto the -1/(C-1) line as training proceeds - you watch
the simplex assemble.

Pure CPU, no dataset download, well under five minutes.

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
  cInDim    = 8;
  cHidden   = 16;
  cClasses  = 4;
  cEpochs   = 600;   // train WELL past zero train-error (terminal phase)
  cReportEv = 60;    // call the report every N epochs
  cProbeCnt = 160;   // fixed probe batch (multiple of cClasses => balanced)
  cBatch    = 64;
  cSep      = 2.0;   // class-centre separation
  cSpread   = 0.45;  // per-blob spread (small => learnable, collapse emerges)

  // Tiny MLP classifier: Input -> FC+ReLU x2 -> FC(linear, penultimate
  // features) -> FC(linear head, width=cClasses) -> SoftMax.
  // The second-to-last FullConnectLinear is the penultimate feature layer the
  // report auto-selects; the final FullConnectLinear is the width-matched head
  // so NC3 (self-duality) is COMPUTED, not skipped.
  procedure BuildNet(out NN: TNNet; LR: TNeuralFloat);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));     // penultimate feats
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));     // linear head
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(LR, 0.9);
    NN.InitWeights();
  end;

  // Synthetic few-class Gaussian-blob sample. Each class centre sits on a
  // regular polygon in the first two dims; the rest are noise distractors.
  procedure MakeSample(out X, Y: TNNetVolume; Cls: integer);
  var
    I: integer;
    Angle, Cx, Cy: TNeuralFloat;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Y.Fill(0);
    Y.Raw[Cls] := 1.0;
    for I := 0 to cInDim - 1 do
      X.Raw[I] := (Random - 0.5) * 2.0 * cSpread;
    Angle := (2.0 * Pi * Cls) / cClasses;
    Cx := cSep * Cos(Angle);
    Cy := cSep * Sin(Angle);
    X.Raw[0] := Cx + (Random - 0.5) * 2.0 * cSpread;
    X.Raw[1] := Cy + (Random - 0.5) * 2.0 * cSpread;
  end;

  procedure BuildProbe(out Batch: TNNetVolumePairList; Count: integer);
  var
    I: integer;
    X, Y: TNNetVolume;
  begin
    Batch := TNNetVolumePairList.Create();
    for I := 0 to Count - 1 do
    begin
      MakeSample(X, Y, I mod cClasses);
      Batch.Add(TNNetVolumePair.Create(X, Y));
    end;
  end;

  // Pull the "mean pairwise cosine=" value out of the report text.
  function ExtractMeanCos(const Rep: string): TNeuralFloat;
  var
    A, B: integer;
    S: string;
    FS: TFormatSettings;
  begin
    Result := 0;
    FS := DefaultFormatSettings;
    FS.DecimalSeparator := '.';
    A := Pos('mean pairwise cosine=', Rep);
    if A <= 0 then Exit;
    A := A + Length('mean pairwise cosine=');
    B := A;
    while (B <= Length(Rep)) and
          (Rep[B] in ['0'..'9', '.', '-', '+', 'e', 'E']) do Inc(B);
    S := Trim(Copy(Rep, A, B - A));
    Result := StrToFloatDef(S, 0, FS);
  end;

var
  NN: TNNet;
  Probe: TNNetVolumePairList;
  Ep, B, Cls, Correct: integer;
  X, Y: TNNetVolume;
  Acc, Target, CosV, Lo, Hi: TNeuralFloat;
  TrajEp: array of integer;
  TrajCos: array of TNeuralFloat;
  NTraj, I, PosC, PosT, Col: integer;
  Bar: string;
  FinalReport: string;

begin
  RandSeed := 424242;
  Target := -1.0 / (cClasses - 1);

  WriteLn('NeuralCollapseReport demo: ', cClasses,
    '-class softmax MLP on synthetic Gaussian blobs, trained into the ',
    'terminal phase.');
  WriteLn('Watch the centered class means assemble into a simplex ETF: the ',
    'mean pairwise cosine -> -1/(C-1) = ', Format('%.4f', [Target]), '.');
  WriteLn;

  BuildProbe(Probe, cProbeCnt);
  BuildNet(NN, 0.01);

  SetLength(TrajEp, 0);
  SetLength(TrajCos, 0);
  NTraj := 0;

  try
    for Ep := 1 to cEpochs do
    begin
      Correct := 0;
      for B := 1 to cBatch do
      begin
        Cls := Random(cClasses);
        MakeSample(X, Y, Cls);
        try
          NN.Compute(X);
          if NN.GetLastLayer.Output.GetClass() = Cls then Inc(Correct);
          NN.Backpropagate(Y);
        finally
          X.Free; Y.Free;
        end;
      end;
      Acc := Correct / cBatch;

      if (Ep = 1) or (Ep mod cReportEv = 0) or (Ep = cEpochs) then
      begin
        CosV := ExtractMeanCos(
          TNNet.NeuralCollapseReport(NN, Probe, cClasses));
        SetLength(TrajEp, NTraj + 1);
        SetLength(TrajCos, NTraj + 1);
        TrajEp[NTraj] := Ep;
        TrajCos[NTraj] := CosV;
        Inc(NTraj);
        WriteLn(Format('  epoch %4d  train-acc=%.3f  mean pairwise cosine=%.4f',
          [Ep, Acc, CosV]));
      end;
    end;

    // --- ASCII trajectory: mean pairwise cosine converging onto -1/(C-1). ---
    WriteLn;
    WriteLn('Mean pairwise centered-class-mean cosine over training');
    WriteLn('(| = current value, T = simplex-ETF target ',
      Format('%.4f', [Target]), '):');
    Lo := Target - 0.15;
    Hi := 1.0;
    for I := 0 to NTraj - 1 do
    begin
      Col := 50;
      PosT := Round((Target - Lo) / (Hi - Lo) * Col);
      PosC := Round((TrajCos[I] - Lo) / (Hi - Lo) * Col);
      if PosT < 0 then PosT := 0; if PosT > Col then PosT := Col;
      if PosC < 0 then PosC := 0; if PosC > Col then PosC := Col;
      Bar := StringOfChar(' ', Col + 1);
      Bar[PosT + 1] := 'T';
      Bar[PosC + 1] := '|';
      WriteLn(Format('  ep %4d %7.4f  %s', [TrajEp[I], TrajCos[I], Bar]));
    end;
    WriteLn('            axis: ', Format('%.2f', [Lo]),
      ' ........................................ ', Format('%.2f', [Hi]));

    // --- final full report. ---
    WriteLn;
    WriteLn(StringOfChar('=', 78));
    WriteLn('Final NeuralCollapseReport (terminal phase):');
    WriteLn(StringOfChar('=', 78));
    FinalReport := TNNet.NeuralCollapseReport(NN, Probe, cClasses);
    Write(FinalReport);

    WriteLn;
    WriteLn('Read it as: NC1 -> 0 (clusters collapse to their class mean), ',
      'NC2 mean|dev| -> 0 (the class means form a simplex ETF, every pair at ',
      'the same angle cos = -1/(C-1)), NC3 -> 1 (classifier rows align with ',
      'the class means: self-duality), NC4 -> 1 (the classifier becomes a ',
      'nearest-class-mean rule). The trajectory above shows the mean pairwise ',
      'cosine marching onto the target line as the simplex assembles.');
  finally
    NN.Free;
    Probe.Free;
  end;
end.
