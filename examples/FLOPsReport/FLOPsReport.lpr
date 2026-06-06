program FLOPsReport;
(*
FLOPsReport: builds two small networks (a tiny MLP and a tiny CIFAR-style
convolutional stack) WITHOUT training them, then prints per-layer estimated
forward-pass FLOPs via TNNet.CountFLOPsPerLayer for each, finishing with a
one-line side-by-side total comparison.

Pure construction; runs in well under a second.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

  procedure BuildMLP(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(64, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(128));
    NN.AddLayer(TNNetFullConnectReLU.Create(64));
    NN.AddLayer(TNNetFullConnectLinear.Create(10));
    NN.AddLayer(TNNetSoftMax.Create());
  end;

  procedure BuildConvNet(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(32, 32, 3));
    NN.AddLayer(TNNetConvolutionReLU.Create({features=}16, {fsize=}3,
      {pad=}1, {stride=}1));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetConvolutionReLU.Create(32, 3, 1, 1));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 1));
    NN.AddLayer(TNNetAvgChannel.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(10));
    NN.AddLayer(TNNetSoftMax.Create());
  end;

  function TotalFLOPsOf(const Report: string): string;
  var
    P: integer;
    Line: string;
  begin
    // The last non-empty line of the report is "TOTAL FLOPs: N (uncovered..)".
    Result := '';
    P := Pos('TOTAL FLOPs:', Report);
    if P = 0 then Exit;
    Line := Copy(Report, P, Length(Report) - P + 1);
    P := Pos(sLineBreak, Line);
    if P > 0 then Line := Copy(Line, 1, P - 1);
    Result := Line;
  end;

var
  MLP, ConvNet: TNNet;
  MLPReport, ConvReport: string;
begin
  WriteLn('=== Tiny MLP (64 -> 128 -> 64 -> 10) ===');
  BuildMLP(MLP);
  try
    MLPReport := TNNet.CountFLOPsPerLayer(MLP);
    Write(MLPReport);
  finally
    MLP.Free;
  end;

  WriteLn;
  WriteLn('=== Tiny CIFAR-style ConvNet ===');
  BuildConvNet(ConvNet);
  try
    ConvReport := TNNet.CountFLOPsPerLayer(ConvNet);
    Write(ConvReport);
  finally
    ConvNet.Free;
  end;

  WriteLn;
  WriteLn('Comparison:');
  WriteLn('  MLP     ', TotalFLOPsOf(MLPReport));
  WriteLn('  ConvNet ', TotalFLOPsOf(ConvReport));
end.
