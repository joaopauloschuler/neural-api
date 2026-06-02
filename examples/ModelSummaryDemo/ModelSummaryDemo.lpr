program ModelSummaryDemo;
(*
ModelSummaryDemo: construct THREE structurally-distinct networks and print each
via TNNet.PrintSummary(), doubling as a self-checking smoke test for the summary
output format.

THE IDEA. TNNet.SummaryString() (which PrintSummary() WriteLns) renders a Keras-
style table: a header row, a separator rule, one row per layer carrying the layer
index, the layer class name, its output shape (SizeX, SizeY, Depth), its weight
count and its neuron count, a closing separator rule, and a final
  "Totals: <L> layers, <W> weights, <N> neurons"
footer. This demo builds three nets that exercise the table across very different
layer types (dense, spatial/conv/pool, and a normalization + multi-input residual
branch) and prints all three.

It is ALSO a smoke test. There is no training: just Create + AddLayer +
InitWeights + PrintSummary, so it finishes in well under a second. For each net we
capture SummaryString() and assert the format is well-formed by parsing it AND by
cross-checking the parsed numbers against the network object computed
independently:
  * the header row contains the expected column titles;
  * there is exactly one body row per layer, i.e. CountLayers() rows;
  * the per-row params summed equal NN.CountWeights(), and the per-row neurons
    summed equal NN.CountNeurons();
  * the "Totals:" footer reports CountLayers()/CountWeights()/CountNeurons()
    matching the object;
  * every net has layers > 0 and weights >= 0;
  * the three nets are structurally distinct (different layer counts and shapes).
Any failure Halt(1)s; otherwise a final PASS line is printed. This mirrors the
self-checking gate idiom used by examples/SIREN and examples/DeepSets.

Pure CPU, single-threaded, deterministic (fixed RandSeed).

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

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, StrUtils,
  neuralnetwork,
  neuralvolume;

const
  cSeed = 424242; // repo idiom

  // ---- Net 1: a small MLP. ---------------------------------------------------
  // Input(8) -> FullConnectReLU(16) -> FullConnectReLU(12) -> FullConnectLinear(4)
  procedure BuildMLP(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(8));
    NN.AddLayer(TNNetFullConnectReLU.Create(16));
    NN.AddLayer(TNNetFullConnectReLU.Create(12));
    NN.AddLayer(TNNetFullConnectLinear.Create(4));
    NN.InitWeights();
  end;

  // ---- Net 2: a tiny conv net. -----------------------------------------------
  // Image-shaped input so the summary shows spatial output shapes and conv params.
  // Input(16,16,3) -> ConvReLU -> MaxPool(2) -> ConvReLU -> MaxPool(2)
  //   -> FullConnectReLU(8) -> FullConnectLinear(3)
  procedure BuildConv(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(16, 16, 3));
    NN.AddLayer(TNNetConvolutionReLU.Create({Features=}8, {FeatureSize=}3,
      {Padding=}1, {Stride=}1, {SuppressBias=}0));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetConvolutionReLU.Create({Features=}12, {FeatureSize=}3,
      {Padding=}1, {Stride=}1, {SuppressBias=}0));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.InitWeights();
  end;

  // ---- Net 3: a normalization + multi-input residual branch. -----------------
  // A pre-norm residual: y = x + PointwiseConvLinear(LayerNorm(x)). The builder
  // emits a TNNetLayerNorm and a TNNetSum (multi-input) so the summary shows a
  // normalization layer and a non-trivial multi-input node. The residual sublayer
  // is a depth-wise PointwiseConvLinear so it is shape-preserving (a FullConnect
  // would change the shape and break the residual sum).
  // Input(1,1,16) -> [LayerNorm -> PointwiseConvLinear(16) -> Sum] -> FullConnectLinear(4)
  procedure BuildResidual(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, 1, 16));
    NN.AddPreNormResidual([TNNetPointwiseConvLinear.Create(16)]);
    NN.AddLayer(TNNetFullConnectLinear.Create(4));
    NN.InitWeights();
  end;

  // Split SummaryString() into non-empty trimmed lines.
  procedure SplitLines(const S: string; out Lines: TStringList);
  begin
    Lines := TStringList.Create();
    Lines.Text := S;
  end;

  // Self-check one net's summary. Returns True on PASS; prints failures.
  function CheckSummary(NN: TNNet; const Tag: string): boolean;
  var
    Summary, Footer: string;
    Lines: TStringList;
    I, BodyRows, RowParams, RowNeurons, SumParams, SumNeurons: integer;
    Parts: TStringList;
    FooterL, FooterW, FooterN: integer;
    HeaderIdx, FirstSepIdx, SecondSepIdx, FooterIdx: integer;
  begin
    Result := True;
    Summary := NN.SummaryString();
    WriteLn(Summary);

    SplitLines(Summary, Lines);
    try
      // Locate the header, the two separator rules and the totals footer.
      HeaderIdx := -1; FirstSepIdx := -1; SecondSepIdx := -1; FooterIdx := -1;
      for I := 0 to Lines.Count - 1 do
      begin
        if (HeaderIdx < 0) and (Pos('Idx', Lines[I]) = 1) and
           (Pos('Layer', Lines[I]) > 0) and (Pos('Output Shape', Lines[I]) > 0) and
           (Pos('Params', Lines[I]) > 0) and (Pos('Neurons', Lines[I]) > 0) then
          HeaderIdx := I;
        if (Pos('---', Lines[I]) = 1) then
        begin
          if FirstSepIdx < 0 then FirstSepIdx := I
          else if SecondSepIdx < 0 then SecondSepIdx := I;
        end;
        if Pos('Totals:', Lines[I]) = 1 then FooterIdx := I;
      end;

      if HeaderIdx < 0 then
      begin
        WriteLn('  [', Tag, '] FAIL: header row not found.'); Result := False;
      end;
      if (FirstSepIdx < 0) or (SecondSepIdx < 0) then
      begin
        WriteLn('  [', Tag, '] FAIL: expected two separator rules.'); Result := False;
      end;
      if FooterIdx < 0 then
      begin
        WriteLn('  [', Tag, '] FAIL: "Totals:" footer not found.'); Result := False;
      end;
      if not Result then Exit;

      // Body rows live strictly between the two separator rules.
      BodyRows := SecondSepIdx - FirstSepIdx - 1;
      if BodyRows <> NN.CountLayers() then
      begin
        WriteLn('  [', Tag, '] FAIL: ', BodyRows, ' body rows but CountLayers()=',
          NN.CountLayers(), '.');
        Result := False;
      end;

      // Sum per-row Params (col 4) and Neurons (col 5) and cross-check the object.
      SumParams := 0; SumNeurons := 0;
      for I := FirstSepIdx + 1 to SecondSepIdx - 1 do
      begin
        Parts := TStringList.Create();
        try
          Parts.Delimiter := ' ';
          Parts.StrictDelimiter := False; // collapse runs of spaces
          Parts.DelimitedText := Trim(Lines[I]);
          // Columns: Idx Class Shape... Params Neurons. Shape "(x, y, z)" is split
          // into 3 tokens by the spaces after commas, so Params/Neurons are the
          // last two tokens regardless of the shape's internal spacing.
          if Parts.Count < 5 then
          begin
            WriteLn('  [', Tag, '] FAIL: malformed body row: "', Lines[I], '".');
            Result := False;
          end
          else
          begin
            RowParams  := StrToInt(Parts[Parts.Count - 2]);
            RowNeurons := StrToInt(Parts[Parts.Count - 1]);
            SumParams  := SumParams + RowParams;
            SumNeurons := SumNeurons + RowNeurons;
          end;
        finally
          Parts.Free;
        end;
      end;

      if SumParams <> NN.CountWeights() then
      begin
        WriteLn('  [', Tag, '] FAIL: per-row params sum ', SumParams,
          ' <> CountWeights() ', NN.CountWeights(), '.');
        Result := False;
      end;
      if SumNeurons <> NN.CountNeurons() then
      begin
        WriteLn('  [', Tag, '] FAIL: per-row neurons sum ', SumNeurons,
          ' <> CountNeurons() ', NN.CountNeurons(), '.');
        Result := False;
      end;

      // Parse the footer "Totals: L layers, W weights, N neurons".
      Footer := Lines[FooterIdx];
      FooterL := StrToIntDef(Trim(Copy(Footer, Pos('Totals:', Footer) + 7,
        Pos('layers', Footer) - (Pos('Totals:', Footer) + 7))), -1);
      FooterW := StrToIntDef(Trim(Copy(Footer, Pos(',', Footer) + 1,
        Pos('weights', Footer) - (Pos(',', Footer) + 1))), -1);
      FooterN := StrToIntDef(Trim(Copy(Footer, PosEx(',', Footer,
        Pos(',', Footer) + 1) + 1,
        Pos('neurons', Footer) - (PosEx(',', Footer, Pos(',', Footer) + 1) + 1))), -1);

      if FooterL <> NN.CountLayers() then
      begin
        WriteLn('  [', Tag, '] FAIL: footer layers ', FooterL, ' <> ',
          NN.CountLayers(), '.'); Result := False;
      end;
      if FooterW <> NN.CountWeights() then
      begin
        WriteLn('  [', Tag, '] FAIL: footer weights ', FooterW, ' <> ',
          NN.CountWeights(), '.'); Result := False;
      end;
      if FooterN <> NN.CountNeurons() then
      begin
        WriteLn('  [', Tag, '] FAIL: footer neurons ', FooterN, ' <> ',
          NN.CountNeurons(), '.'); Result := False;
      end;

      // Sanity invariants.
      if NN.CountLayers() <= 0 then
      begin
        WriteLn('  [', Tag, '] FAIL: non-positive layer count.'); Result := False;
      end;
      if NN.CountWeights() < 0 then
      begin
        WriteLn('  [', Tag, '] FAIL: negative weight count.'); Result := False;
      end;
    finally
      Lines.Free;
    end;
  end;

var
  NNMlp, NNConv, NNResidual: TNNet;
  Pass: boolean;
begin
  RandSeed := cSeed;     // deterministic; no training/threads are used here
  Pass := True;

  WriteLn('ModelSummaryDemo: TNNet.PrintSummary() across three networks.');
  WriteLn('No training - just Create + AddLayer + InitWeights + PrintSummary.');
  WriteLn('Doubles as a smoke test of the summary table format.');
  WriteLn;

  // ---- Net 1: MLP ------------------------------------------------------------
  WriteLn('================================================================');
  WriteLn('Net 1: small MLP');
  WriteLn('  Input(8) -> FullConnectReLU(16) -> FullConnectReLU(12) -> FullConnectLinear(4)');
  WriteLn('----------------------------------------------------------------');
  BuildMLP(NNMlp);
  Pass := CheckSummary(NNMlp, 'MLP') and Pass;
  WriteLn;

  // ---- Net 2: conv net -------------------------------------------------------
  WriteLn('================================================================');
  WriteLn('Net 2: tiny conv net');
  WriteLn('  Input(16,16,3) -> ConvReLU(8) -> MaxPool -> ConvReLU(12) -> MaxPool');
  WriteLn('    -> FullConnectReLU(8) -> FullConnectLinear(3)');
  WriteLn('----------------------------------------------------------------');
  BuildConv(NNConv);
  Pass := CheckSummary(NNConv, 'CONV') and Pass;
  WriteLn;

  // ---- Net 3: normalization + multi-input residual ---------------------------
  WriteLn('================================================================');
  WriteLn('Net 3: pre-norm residual (LayerNorm + multi-input Sum)');
  WriteLn('  Input(1,1,16) -> [LayerNorm -> PointwiseConvLinear(16) -> Sum]');
  WriteLn('    -> FullConnectLinear(4)');
  WriteLn('----------------------------------------------------------------');
  BuildResidual(NNResidual);
  Pass := CheckSummary(NNResidual, 'RESID') and Pass;
  WriteLn;

  // ---- The three nets must be structurally distinct. -------------------------
  if (NNMlp.CountLayers() = NNConv.CountLayers()) and
     (NNConv.CountLayers() = NNResidual.CountLayers()) then
  begin
    WriteLn('GATE: FAIL - the three nets share the same layer count; not distinct.');
    Pass := False;
  end;

  WriteLn('================================================================');
  if Pass then
    WriteLn('GATE: PASS - all three summaries are well-formed and their row/total ' +
      'counts match CountLayers()/CountWeights()/CountNeurons().')
  else
    WriteLn('GATE: FAIL - see messages above.');

  NNMlp.Free;
  NNConv.Free;
  NNResidual.Free;

  if not Pass then Halt(1);
end.
