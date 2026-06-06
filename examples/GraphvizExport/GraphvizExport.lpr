program GraphvizExport;
(*
GraphvizExport: builds a few small networks and prints the Graphviz DOT
description of each one's layer DAG via TNNet.ToGraphvizDot, then writes the
branched net's DOT to a file (branched_net.dot) so it can be rendered:
  dot -Tpng branched_net.dot -o branched_net.png

Three nets are shown:
  1. A plain sequential MLP (single chain of edges).
  2. A BRANCHED residual net with a TNNetSum (two incoming edges into the
     sum node make the multi-input DAG visible).
  3. A TNNetDeepConcat branch (another multi-input merge).

Pure structure / forward-only: ToGraphvizDot never trains. Self-contained and
synthetic - no dataset download, runs in well under a minute.

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

  // Plain sequential MLP: a single chain of layers.
  function BuildSequentialMLP(): TNNet;
  begin
    Result := TNNet.Create();
    Result.AddLayer(TNNetInput.Create(8, 1, 1));
    Result.AddLayer(TNNetFullConnectLinear.Create(16));
    Result.AddLayer(TNNetReLU.Create());
    Result.AddLayer(TNNetFullConnectLinear.Create(16));
    Result.AddLayer(TNNetReLU.Create());
    Result.AddLayer(TNNetFullConnectLinear.Create(1));
  end;

  // Branched residual net: a short cut and a longer path merged by a
  // TNNetSum, so the sum node has two incoming edges.
  function BuildBranchedNet(): TNNet;
  var
    InputLayer, ShortCut, LongPath: TNNetLayer;
  begin
    Result := TNNet.Create();
    InputLayer := Result.AddLayer(TNNetInput.Create(8, 1, 1));
    ShortCut := Result.AddLayer(TNNetFullConnectLinear.Create(16));
    // Longer path branching from the same shortcut output.
    Result.AddLayerAfter(TNNetFullConnectLinear.Create(16), ShortCut);
    Result.AddLayer(TNNetReLU.Create());
    LongPath := Result.AddLayer(TNNetFullConnectLinear.Create(16));
    // Residual merge: ShortCut + LongPath.
    Result.AddLayer(TNNetSum.Create([ShortCut, LongPath]));
    Result.AddLayer(TNNetReLU.Create());
    Result.AddLayer(TNNetFullConnectLinear.Create(1));
    if InputLayer = nil then ; // silence unused-var on some FPC configs
  end;

  // A net whose two branches are merged by a TNNetDeepConcat (depth concat).
  function BuildConcatNet(): TNNet;
  var
    Stem, BranchA, BranchB: TNNetLayer;
  begin
    Result := TNNet.Create();
    Result.AddLayer(TNNetInput.Create(8, 8, 3));
    Stem := Result.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
    BranchA := Result.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
    Result.AddLayerAfter(TNNetConvolutionReLU.Create(8, 3, 1, 1), Stem);
    BranchB := Result.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
    Result.AddLayer(TNNetDeepConcat.Create([BranchA, BranchB]));
    Result.AddLayer(TNNetReLU.Create());
  end;

var
  NN: TNNet;
  Dot: string;
  DotFile: TStringList;
begin
  WriteLn('GraphvizExport demo: TNNet.ToGraphvizDot on three small nets.');
  WriteLn('Render any block below with, e.g.:  dot -Tpng net.dot -o net.png');

  // ---- 1. Plain sequential MLP ----
  WriteLn;
  WriteLn(StringOfChar('=', 72));
  WriteLn('NET 1: plain sequential MLP');
  WriteLn(StringOfChar('=', 72));
  NN := BuildSequentialMLP();
  try
    WriteLn(NN.ToGraphvizDot('SequentialMLP'));
  finally
    NN.Free;
  end;

  // ---- 2. Branched residual net (TNNetSum) ----
  WriteLn;
  WriteLn(StringOfChar('=', 72));
  WriteLn('NET 2: branched residual net (TNNetSum has two incoming edges)');
  WriteLn(StringOfChar('=', 72));
  NN := BuildBranchedNet();
  try
    Dot := NN.ToGraphvizDot('BranchedResidualNet');
    WriteLn(Dot);
    // Persist this one so it can be rendered.
    DotFile := TStringList.Create;
    try
      DotFile.Text := Dot;
      DotFile.SaveToFile('branched_net.dot');
    finally
      DotFile.Free;
    end;
    WriteLn('(written to branched_net.dot - render: dot -Tpng branched_net.dot -o branched_net.png)');
  finally
    NN.Free;
  end;

  // ---- 3. TNNetDeepConcat branch ----
  WriteLn;
  WriteLn(StringOfChar('=', 72));
  WriteLn('NET 3: TNNetDeepConcat merge (two incoming edges)');
  WriteLn(StringOfChar('=', 72));
  NN := BuildConcatNet();
  try
    WriteLn(NN.ToGraphvizDot('ConcatNet'));
  finally
    NN.Free;
  end;

  // ---- Edge case: empty net ----
  WriteLn;
  WriteLn(StringOfChar('=', 72));
  WriteLn('NET 4: empty network (valid empty digraph)');
  WriteLn(StringOfChar('=', 72));
  NN := TNNet.Create();
  try
    WriteLn(NN.ToGraphvizDot());
  finally
    NN.Free;
  end;
end.
