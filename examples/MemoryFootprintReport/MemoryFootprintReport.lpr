program MemoryFootprintReport;
(*
MemoryFootprintReport: builds three small networks (a tiny MLP, a small
CIFAR-style convolutional stack, and a small attention stack) WITHOUT
training them, then prints per-layer activation/parameter/gradient memory
via TNNet.MemoryFootprintReport for each, finishing with a one-line
side-by-side bottleneck comparison.

Pure construction; runs in well under a second.

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

  procedure BuildMLP(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(64, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(256));
    NN.AddLayer(TNNetFullConnectReLU.Create(256));
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

  procedure BuildAttentionStack(out NN: TNNet);
  const
    cSeqLen = 16;
    cEmb    = 32;
    cDk     = 16;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, cEmb));
    // Project embeddings to a Q|K|V concatenation of width 3*Dk.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDk));
    // First SDPA block (non-causal): full bidirectional attention.
    NN.AddLayer(TNNetScaledDotProductAttention.Create(cDk, False));
    // Re-pack d_k -> 3*d_k for a second SDPA block.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDk));
    // Second SDPA block (causal): every query may only see past keys.
    NN.AddLayer(TNNetScaledDotProductAttention.Create(cDk, True));
    // Read-out back to embedding dim.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cEmb));
  end;

  function ExtractLine(const Report, Needle: string): string;
  var
    P: integer;
    Line: string;
  begin
    Result := '';
    P := Pos(Needle, Report);
    if P = 0 then Exit;
    Line := Copy(Report, P, Length(Report) - P + 1);
    P := Pos(sLineBreak, Line);
    if P > 0 then Line := Copy(Line, 1, P - 1);
    Result := Line;
  end;

var
  MLP, ConvNet, AttNet: TNNet;
  ReportMLP, ReportConv, ReportAtt: string;
begin
  WriteLn('=== Tiny MLP (64 -> 256 -> 256 -> 10) ===');
  BuildMLP(MLP);
  try
    ReportMLP := TNNet.MemoryFootprintReport(MLP, 'adam', 2048.0);
    Write(ReportMLP);
  finally
    MLP.Free;
  end;

  WriteLn;
  WriteLn('=== Tiny CIFAR-style ConvNet ===');
  BuildConvNet(ConvNet);
  try
    ReportConv := TNNet.MemoryFootprintReport(ConvNet, 'adam', 2048.0);
    Write(ReportConv);
  finally
    ConvNet.Free;
  end;

  WriteLn;
  WriteLn('=== Tiny Attention Stack (SeqLen=16, Emb=32, Dk=16) ===');
  BuildAttentionStack(AttNet);
  try
    ReportAtt := TNNet.MemoryFootprintReport(AttNet, 'adam', 2048.0);
    Write(ReportAtt);
  finally
    AttNet.Free;
  end;

  WriteLn;
  WriteLn('Bottleneck shape comparison (peak-forward residency line):');
  WriteLn('  MLP     ', ExtractLine(ReportMLP, 'Peak forward residency'));
  WriteLn('  ConvNet ', ExtractLine(ReportConv, 'Peak forward residency'));
  WriteLn('  AttNet  ', ExtractLine(ReportAtt, 'Peak forward residency'));
  WriteLn('Parameter baseline comparison:');
  WriteLn('  MLP     ', ExtractLine(ReportMLP, 'Parameters:'));
  WriteLn('  ConvNet ', ExtractLine(ReportConv, 'Parameters:'));
  WriteLn('  AttNet  ', ExtractLine(ReportAtt, 'Parameters:'));
end.
