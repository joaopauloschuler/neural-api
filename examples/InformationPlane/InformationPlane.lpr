program InformationPlane;
(*
InformationPlane: reproduces the INFORMATION-PLANE TRAJECTORY of the Information
Bottleneck story (Tishby & Zaslavsky 2015; Shwartz-Ziv & Tishby 2017, "Opening
the Black Box of Deep Neural Networks via Information"). For a tiny fully-connected
classifier on a small synthetic binary task, it tracks the mutual-information pair
(I(X;T), I(T;Y)) of EACH hidden layer T across training epochs and prints every
layer's path through the 2-D information plane as an ASCII scatter.

Narrative target: two reported phases. A fast FITTING/ERM phase where both I(X;T)
and I(T;Y) rise, followed by a slow COMPRESSION phase where I(X;T) DROPS while
I(T;Y) stays high (the layer forgets input detail irrelevant to the label).

MI estimator = the original BINNING estimator. Each neuron's bounded activation is
discretized into B equal-width bins; the per-sample bin-tuple is that sample's
discrete code for layer T. We compute plug-in entropies from empirical histograms:
  I(X;T) = H(T) - H(T|X). The net is deterministic and each input is unique, so
           H(T|X) = 0 and I(X;T) = H(T).
  I(T;Y) = H(T) - H(T|Y), with H(T|Y) = sum_y p(y) * H(T | Y=y).
All quantities are in BITS (log base 2). No new gradient machinery -- only a
forward pass (Net.Compute) over the full dataset at each logged epoch, reading
each hidden layer's Output volume.

HONEST headline (house "what did NOT reproduce" style): the binning estimator
REQUIRES a saturating activation to show compression. Two arms are shipped:
  ARM A (headline) -- a TANH trunk (TNNetFullConnect, bounded to [-1,1]) so the
         fixed-width bins are meaningful and the compression bend can appear;
  ARM B (contrast) -- a ReLU trunk (TNNetFullConnectReLU, unbounded -> fixed-width
         binning is ill-defined; the clean compression bend vanishes),
demonstrating the Saxe et al. 2018 controversy ("On the Information Bottleneck
Theory of Deep Learning") rather than overclaiming.

Pitfalls (see README): MI is upper-bounded by log2(#samples) and by B^width, so
width/B/sample-count are kept balanced. Absolute MI values are estimator-dependent;
the robust reproducible signal is the SHAPE of the trajectory and the tanh-vs-ReLU
difference, not the nats. Pure CPU, tiny MLP, well under 5 minutes.

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
  cInputDim       = 12;     // input dimensionality (12-bit symmetric task)
  cNumSamples     = 2048;   // full dataset; I(X;T) ceiling = log2(2048) = 11 bits
  cHiddenWidth    = 4;      // hidden width; B^width code space stays bounded
  cBins           = 30;     // B equal-width bins per neuron
  cEpochs         = 200;    // total training epochs
  cLogEvery       = 10;     // log MI every this many epochs
  cLearningRate   = 0.02;

type
  TInfoPoint = record
    Epoch: integer;
    Ixt, Ity: TNeuralFloat;
  end;
  TInfoTrack = array of TInfoPoint;   // trajectory for one layer

// Box-Muller N(0,1) -- no extra dependency.
function RandomGauss(): TNeuralFloat;
var U1, U2: TNeuralFloat;
begin
  repeat U1 := Random; until U1 > 1e-9;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

// Synthetic binary task: 12 noisy bits in {-1,+1}; label = parity-like majority
// vote over a fixed informative subset (first 5 bits). The remaining 7 bits are
// nuisance: relevant to X but NOT to Y, so a compressing layer should discard
// them. Gaussian jitter makes each input unique (so H(T|X)=0 holds).
procedure MakeSample(out X, Y: TNNetVolume);
var
  I, S: integer;
  Bit: integer;
begin
  X := TNNetVolume.Create(cInputDim, 1, 1);
  Y := TNNetVolume.Create(2, 1, 1);
  S := 0;
  for I := 0 to cInputDim - 1 do
  begin
    Bit := Random(2) * 2 - 1;                 // -1 or +1
    if I < 5 then S := S + Bit;               // first 5 bits carry the label
    X.FData[I] := Bit + RandomGauss() * 0.15; // jitter -> unique inputs
  end;
  Y.Fill(0);
  if S > 0 then Y.FData[1] := 1.0 else Y.FData[0] := 1.0;
end;

procedure BuildSet(out Pairs: TNNetVolumePairList; N: integer);
var
  I: integer;
  X, Y: TNNetVolume;
begin
  Pairs := TNNetVolumePairList.Create();
  for I := 1 to N do
  begin
    MakeSample(X, Y);
    Pairs.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// Plug-in Shannon entropy (bits) of a histogram of counts over Total samples.
function EntropyFromCounts(Counts: TStringList; Total: integer): TNeuralFloat;
var
  I, C: integer;
  P: TNeuralFloat;
begin
  Result := 0;
  for I := 0 to Counts.Count - 1 do
  begin
    C := PtrInt(Counts.Objects[I]);
    if C > 0 then
    begin
      P := C / Total;
      Result := Result - P * (Ln(P) / Ln(2.0));
    end;
  end;
end;

// Increments the count stored against Key in a sorted string list (count kept in
// the Objects[] slot as a tagged integer). Sorted+Find = O(log n) lookup.
procedure BumpCount(L: TStringList; const Key: string);
var
  Idx: integer;
begin
  if L.Find(Key, Idx) then
    L.Objects[Idx] := TObject(PtrInt(L.Objects[Idx]) + 1)
  else
    L.AddObject(Key, TObject(PtrInt(1)));
end;

function NewCountList(): TStringList;
begin
  Result := TStringList.Create;
  Result.Sorted := True;
  Result.Duplicates := dupError;
end;

// Maps one activation value to a bin index [0..cBins-1] over [Lo,Hi].
function BinIndex(V, Lo, Hi: TNeuralFloat): integer;
begin
  if Hi <= Lo then Exit(0);
  Result := Trunc((V - Lo) / (Hi - Lo) * cBins);
  if Result < 0 then Result := 0;
  if Result >= cBins then Result := cBins - 1;
end;

// Computes (I(X;T), I(T;Y)) in bits for a single layer's activations, using the
// binning estimator. Activations is one TNNetVolume per sample (already computed
// for that layer); Labels[i] is the class of sample i. Range [Lo,Hi] bounds the
// bins (tanh: [-1,1]; ReLU: observed [0, max]).
procedure LayerMI(const Activations: array of TNNetVolume;
  const Labels: array of integer; Width: integer; Lo, Hi: TNeuralFloat;
  out Ixt, Ity: TNeuralFloat);
var
  N, S, J, Cls: integer;
  Code: string;
  AllT: TStringList;                 // p(T)
  PerClass: array[0..1] of TStringList;  // p(T | Y=y)
  ClassCount: array[0..1] of integer;
  Ht, HtGivenY: TNeuralFloat;
begin
  N := Length(Activations);
  AllT := NewCountList();
  PerClass[0] := NewCountList();
  PerClass[1] := NewCountList();
  ClassCount[0] := 0; ClassCount[1] := 0;
  try
    for S := 0 to N - 1 do
    begin
      Code := '';
      for J := 0 to Width - 1 do
        Code := Code + IntToStr(BinIndex(Activations[S].FData[J], Lo, Hi)) + ',';
      Cls := Labels[S];
      BumpCount(AllT, Code);
      BumpCount(PerClass[Cls], Code);
      Inc(ClassCount[Cls]);
    end;

    Ht := EntropyFromCounts(AllT, N);
    // H(T|Y) = sum_y p(y) * H(T | Y=y)
    HtGivenY := 0;
    for Cls := 0 to 1 do
      if ClassCount[Cls] > 0 then
        HtGivenY := HtGivenY +
          (ClassCount[Cls] / N) *
          EntropyFromCounts(PerClass[Cls], ClassCount[Cls]);

    // H(T|X)=0 for a deterministic net with unique inputs -> I(X;T)=H(T).
    Ixt := Ht;
    Ity := Ht - HtGivenY;
    if Ity < 0 then Ity := 0;  // guard tiny negative round-off
  finally
    AllT.Free;
    PerClass[0].Free;
    PerClass[1].Free;
  end;
end;

// One epoch of plain SGD over Pairs.
procedure TrainEpoch(NN: TNNet; Pairs: TNNetVolumePairList);
var
  I: integer;
  Pair: TNNetVolumePair;
begin
  for I := 0 to Pairs.Count - 1 do
  begin
    Pair := Pairs[I];
    NN.Compute(Pair.I);
    NN.Backpropagate(Pair.O);
  end;
end;

// Returns mean cross-entropy NLL over the set (one forward pass per sample).
function MeanNLL(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I: integer;
  Pair: TNNetVolumePair;
begin
  Result := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    Pair := Pairs[I];
    NN.Compute(Pair.I);
    Result := Result -
      Ln(Max(1e-9, NN.GetLastLayer().Output.FData[Pair.O.GetClass()]));
  end;
  Result := Result / Pairs.Count;
end;

// Renders all layer trajectories on one ASCII information plane.
// x-axis = I(X;T) (bits), y-axis = I(T;Y) (bits). Each layer gets a distinct
// glyph; lowercase marks the path, UPPERCASE marks the final logged epoch.
procedure PrintInfoPlane(const ArmName: string;
  const HiddenIdx: array of integer; const Tracks: array of TInfoTrack);
const
  W = 56;  // plot width  (columns)
  H = 22;  // plot height (rows)
  cGlyph: array[0..2] of char = ('a', 'b', 'c');  // layer markers
var
  Grid: array of array of char;
  R, C, L, K: integer;
  MaxX, MaxY: TNeuralFloat;
  Px, Py: integer;
  Ch: char;
begin
  MaxX := Log2(cNumSamples);  // I(X;T) ceiling = log2(N)
  MaxY := 1.0;                // binary label -> I(T;Y) <= 1 bit

  SetLength(Grid, H);
  for R := 0 to H - 1 do
  begin
    SetLength(Grid[R], W);
    for C := 0 to W - 1 do Grid[R][C] := ' ';
  end;

  for L := 0 to High(Tracks) do
    for K := 0 to High(Tracks[L]) do
    begin
      Px := Round(Tracks[L][K].Ixt / MaxX * (W - 1));
      Py := Round(Tracks[L][K].Ity / MaxY * (H - 1));
      if Px < 0 then Px := 0;
      if Px > W - 1 then Px := W - 1;
      if Py < 0 then Py := 0;
      if Py > H - 1 then Py := H - 1;
      R := (H - 1) - Py;  // invert: high I(T;Y) at top
      if K = High(Tracks[L]) then Ch := UpCase(cGlyph[L])  // final epoch
      else Ch := cGlyph[L];
      Grid[R][Px] := Ch;
    end;

  WriteLn;
  WriteLn('Information plane (', ArmName, ')  --  lowercase=path, UPPER=final epoch');
  WriteLn('  y = I(T;Y) bits  (top=', MaxY:0:1, ')      x = I(X;T) bits  (right=',
    MaxX:0:1, ')');
  for L := 0 to High(HiddenIdx) do
    WriteLn('    layer L', HiddenIdx[L], ' = ''', cGlyph[L], '''');
  WriteLn('  ', StringOfChar('-', W));
  for R := 0 to H - 1 do
  begin
    Write('  |');
    for C := 0 to W - 1 do Write(Grid[R][C]);
    WriteLn('|');
  end;
  WriteLn('  ', StringOfChar('-', W));
  WriteLn('   I(X;T) ->  (compression = a layer''s path bends LEFT while ' +
    'staying high)');
end;

// ----- one experiment arm (tanh or ReLU trunk) -----
// HiddenIdx lists the network layer indices of the hidden trunk layers to probe.
// ActLo/ActHi is the binning range; for ReLU ActHi is rescanned per epoch.
procedure RunArm(const ArmName: string; UseTanh: boolean;
  Pairs: TNNetVolumePairList);
var
  NN: TNNet;
  HiddenIdx: array of integer;
  Tracks: array of TInfoTrack;   // one trajectory per hidden layer
  Labels: array of integer;
  Acts: array of array of TNNetVolume; // [layer][sample] activation snapshots
  Epoch, L, S, Li: integer;
  ActLo, ActHi, Obs: TNeuralFloat;
  Ixt, Ity: TNeuralFloat;
  Pair: TNNetVolumePair;
begin
  WriteLn;
  WriteLn('############################################################');
  WriteLn('# ARM ', ArmName);
  if UseTanh then
    WriteLn('#   tanh trunk (bounded [-1,1]) -- bins meaningful')
  else
    WriteLn('#   ReLU trunk (unbounded)     -- fixed-width bins ill-defined');
  WriteLn('############################################################');

  RandSeed := 424242;  // determinism

  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cInputDim, 1, 1));
  if UseTanh then
  begin
    NN.AddLayer(TNNetFullConnect.Create(cHiddenWidth));      // tanh
    NN.AddLayer(TNNetFullConnect.Create(cHiddenWidth));      // tanh
    NN.AddLayer(TNNetFullConnect.Create(cHiddenWidth));      // tanh
  end
  else
  begin
    NN.AddLayer(TNNetFullConnectReLU.Create(cHiddenWidth));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHiddenWidth));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHiddenWidth));
  end;
  NN.AddLayer(TNNetFullConnectLinear.Create(2));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.InitWeights();
  NN.SetLearningRate(cLearningRate, 0.9);

  // hidden trunk layers = indices 1,2,3 (after the input layer)
  SetLength(HiddenIdx, 3);
  HiddenIdx[0] := 1; HiddenIdx[1] := 2; HiddenIdx[2] := 3;

  SetLength(Tracks, Length(HiddenIdx));
  for L := 0 to High(Tracks) do
    SetLength(Tracks[L], 0);

  // cache labels once
  SetLength(Labels, Pairs.Count);
  for S := 0 to Pairs.Count - 1 do
    Labels[S] := Pairs[S].O.GetClass();

  SetLength(Acts, Length(HiddenIdx));

  WriteLn;
  WriteLn(' epoch  meanNLL | layer  I(X;T)   I(T;Y)   (bits)');
  WriteLn(' -----  ------- | -----  ------   ------');

  for Epoch := 0 to cEpochs do
  begin
    if (Epoch mod cLogEvery = 0) then
    begin
      // collect activations over the full dataset
      for L := 0 to High(HiddenIdx) do
      begin
        SetLength(Acts[L], Pairs.Count);
        for S := 0 to Pairs.Count - 1 do
          Acts[L][S] := nil;
      end;
      for S := 0 to Pairs.Count - 1 do
      begin
        Pair := Pairs[S];
        NN.Compute(Pair.I);
        for L := 0 to High(HiddenIdx) do
          Acts[L][S] := TNNetVolume.Create(NN.Layers[HiddenIdx[L]].Output);
      end;

      for L := 0 to High(HiddenIdx) do
      begin
        if UseTanh then
        begin
          ActLo := -1.0; ActHi := 1.0;
        end
        else
        begin
          // ReLU: bins over observed [0, max]; this very rescaling is the
          // ill-definedness that washes out the compression bend.
          ActLo := 0.0; ActHi := 1e-6;
          for S := 0 to Pairs.Count - 1 do
            for Li := 0 to cHiddenWidth - 1 do
            begin
              Obs := Acts[L][S].FData[Li];
              if Obs > ActHi then ActHi := Obs;
            end;
        end;

        LayerMI(Acts[L], Labels, cHiddenWidth, ActLo, ActHi, Ixt, Ity);

        SetLength(Tracks[L], Length(Tracks[L]) + 1);
        Tracks[L][High(Tracks[L])].Epoch := Epoch;
        Tracks[L][High(Tracks[L])].Ixt := Ixt;
        Tracks[L][High(Tracks[L])].Ity := Ity;

        if L = 0 then
          Write(Epoch:6, ' ', MeanNLL(NN, Pairs):7:4, ' | ')
        else
          Write('                | ');
        WriteLn('L', HiddenIdx[L], '    ', Ixt:6:3, '   ', Ity:6:3);
      end;

      // free snapshots
      for L := 0 to High(HiddenIdx) do
        for S := 0 to Pairs.Count - 1 do
          Acts[L][S].Free;
    end;

    if Epoch < cEpochs then TrainEpoch(NN, Pairs);
  end;

  // ----- ASCII information-plane scatter -----
  PrintInfoPlane(ArmName, HiddenIdx, Tracks);

  NN.Free;
end;

var
  DataSet: TNNetVolumePairList;
begin
  WriteLn('========================================================');
  WriteLn(' Information Bottleneck -- information-plane trajectory');
  WriteLn(' binning MI estimator, ', cNumSamples, ' samples, B=', cBins,
    ' bins, width=', cHiddenWidth);
  WriteLn(' I(X;T) ceiling = log2(N) = ', Log2(cNumSamples):0:2, ' bits; ',
    'I(T;Y) ceiling = 1 bit');
  WriteLn('========================================================');

  // Single fixed-seed dataset shared by both arms (so the only difference
  // between arms is the activation function, not the data).
  RandSeed := 20260607;
  BuildSet(DataSet, cNumSamples);

  RunArm('A (tanh -- headline)', True,  DataSet);
  RunArm('B (ReLU -- contrast)', False, DataSet);

  DataSet.Free;
end.
