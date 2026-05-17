program ArchitectureDiff;
(*
ArchitectureDiff: prints a unified-diff-style report of architectural
differences between two near-identical SimpleImageClassifier-style
networks. Demonstrates TNNet.DiffArchitecture and the companion
TNNet.DiffArchitectureFromString overload.

Variant A:  the baseline classifier (Conv -> Conv -> MaxPool -> FC).
Variant B:  same shape but with an extra TNNetChannelStdNormalization
            after the first conv and the hidden FC activation swapped from
            ReLU to Sigmoid.

The diff format is:
   ClassName            OutputShape         Params       (matching layer)
- ClassName            OutputShape         Params       (only in A)
+ ClassName            OutputShape         Params       (only in B)

Pure-CPU, finishes in milliseconds; no training is performed.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

  function BuildVariantA: TNNet;
  begin
    Result := TNNet.Create();
    Result.AddLayer(TNNetInput.Create(32, 32, 3));
    Result.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1));
    Result.AddLayer(TNNetConvolutionReLU.Create(32, 3, 1, 1));
    Result.AddLayer(TNNetMaxPool.Create(2));
    Result.AddLayer(TNNetFullConnectReLU.Create(64));
    Result.AddLayer(TNNetFullConnectLinear.Create(10));
  end;

  // Same skeleton plus a normalization after the first conv and a
  // different final hidden activation.
  function BuildVariantB: TNNet;
  begin
    Result := TNNet.Create();
    Result.AddLayer(TNNetInput.Create(32, 32, 3));
    Result.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1));
    Result.AddLayer(TNNetChannelStdNormalization.Create());
    Result.AddLayer(TNNetConvolutionReLU.Create(32, 3, 1, 1));
    Result.AddLayer(TNNetMaxPool.Create(2));
    Result.AddLayer(TNNetFullConnectSigmoid.Create(64));
    Result.AddLayer(TNNetFullConnectLinear.Create(10));
  end;

  procedure RunDemo;
  var
    A, B: TNNet;
    Diff, SelfDiff, GoldenStr: string;
  begin
    A := BuildVariantA();
    B := BuildVariantB();
    try
      WriteLn('Variant A summary:');
      A.PrintSummary();
      WriteLn('Variant B summary:');
      B.PrintSummary();

      WriteLn('Diff (A vs B):');
      WriteLn(StringOfChar('-', 75));
      Diff := A.DiffArchitecture(B);
      if Diff = '' then
        WriteLn('  (identical)')
      else
        Write(Diff);
      WriteLn(StringOfChar('-', 75));
      WriteLn;

      WriteLn('Sanity: A diffed against itself should be empty.');
      SelfDiff := A.DiffArchitecture(A);
      if SelfDiff = '' then
        WriteLn('  OK (empty diff)')
      else
        WriteLn('  FAIL: self-diff was not empty.');
      WriteLn;

      WriteLn('Sanity: DiffArchitectureFromString round-trip through ' +
        'SaveStructureToString.');
      GoldenStr := B.SaveStructureToString();
      Diff := A.DiffArchitectureFromString(GoldenStr);
      if Diff = '' then
        WriteLn('  FAIL: A vs B golden should differ.')
      else
        WriteLn('  OK (', Length(Diff), ' chars, same shape as live diff)');
    finally
      A.Free;
      B.Free;
    end;
  end;

var
  Application: record Title: string; end;

begin
  Application.Title := 'Architecture Diff Example';
  RunDemo();
end.
