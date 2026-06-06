program DeadReLUDiagnostic;
(*
DeadReLUDiagnostic: trains the SAME small classifier FOUR times - identical in
every way except the hidden activation function (ReLU, LeakyReLU, GELU, Swish) -
under a deliberately aggressive learning rate, and tracks the PER-EPOCH fraction
of dead hidden units along the whole training trajectory. It finishes with a
summary table comparing the dead-unit fraction across the four activations.

A hidden unit is "dead" when its activation output stays at (numerically) zero
for EVERY probe sample: such a unit produces a zero gradient and can never
recover. Plain ReLU is prone to this under an aggressive learning rate, because
once a unit's pre-activation is pushed negative for all inputs it is stuck.
LeakyReLU / GELU / Swish keep a non-zero output (and gradient) for negative
inputs, so far fewer of their units die. This is the whole point of the demo.

How this DIFFERS from examples/DeadNeuronReport/:
  - DeadNeuronReport prints a single STATIC report of one already-trained net.
  - This example reports a TRAJECTORY (dead fraction at every epoch) AND a
    cross-activation COMPARISON of four otherwise-identical networks.

The dead-unit fraction is measured the same way TNNet.DeadNeuronReport defines
it (see neural/neuralnetwork.pas): push every probe sample through the net and,
for each hidden unit, track the maximum absolute activation; a unit is dead if
that maximum is <= a small threshold. Here we read the number directly per
epoch instead of the formatted string report.

Pure CPU, single-threaded, well under a minute.

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
  neuralvolume,
  neuralfit;

const
  cInDim     = 6;     // input dimensionality (blobs lifted into 6 dims)
  cClasses   = 3;     // number of Gaussian blobs / output classes
  cHidden    = 64;    // hidden units per layer
  cEpochs    = 60;    // training epochs
  cBatch     = 40;    // pairs per epoch
  cProbeCnt  = 96;    // probe samples for the dead-unit measurement
  cLR        = 0.5;   // aggressive learning rate (induces dying ReLU)
  cDeadThr   = 1e-6;  // same default threshold as TNNet.DeadNeuronReport
  cSeed      = 2026;

type
  TArmKind = (akReLU, akLeakyReLU, akGELU, akSwish);

var
  // Fixed blob centres, shared across all four arms so the task is identical.
  Centres: array[0 .. cClasses - 1, 0 .. cInDim - 1] of TNeuralFloat;
  // Final dead fraction recorded per arm for the summary table.
  FinalDeadFrac: array[TArmKind] of TNeuralFloat;
  PeakDeadFrac: array[TArmKind] of TNeuralFloat;

  function ArmName(Kind: TArmKind): string;
  begin
    case Kind of
      akReLU:      Result := 'TNNetReLU';
      akLeakyReLU: Result := 'TNNetLeakyReLU';
      akGELU:      Result := 'TNNetGELU';
      akSwish:     Result := 'TNNetSwish';
    else
      Result := '?';
    end;
  end;

  // One shared activation factory keeps the four arms byte-for-byte identical
  // apart from the activation class itself.
  function NewActivation(Kind: TArmKind): TNNetLayer;
  begin
    case Kind of
      akReLU:      Result := TNNetReLU.Create();
      akLeakyReLU: Result := TNNetLeakyReLU.Create();
      akGELU:      Result := TNNetGELU.Create();
      akSwish:     Result := TNNetSwish.Create();
    else
      Result := TNNetReLU.Create();
    end;
  end;

  procedure InitCentres();
  var
    C, I: integer;
  begin
    for C := 0 to cClasses - 1 do
      for I := 0 to cInDim - 1 do
        Centres[C][I] := (Random - 0.5) * 6.0;
  end;

  // Draw one (x, one-hot y) pair: a Gaussian blob around a random class centre.
  procedure MakePair(out X, Y: TNNetVolume);
  var
    Cls, I: integer;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Cls := Random(cClasses);
    for I := 0 to cInDim - 1 do
      X.Raw[I] := Centres[Cls][I] + (Random - 0.5) * 1.2;
    Y.SetClassForSoftMax(Cls);
  end;

  procedure BuildProbes(out Probes: TNNetVolumeList);
  var
    K, Cls, I: integer;
    V: TNNetVolume;
  begin
    Probes := TNNetVolumeList.Create(True);
    for K := 0 to cProbeCnt - 1 do
    begin
      V := TNNetVolume.Create(cInDim, 1, 1);
      Cls := K mod cClasses;
      for I := 0 to cInDim - 1 do
        V.Raw[I] := Centres[Cls][I] + (Random - 0.5) * 1.2;
      Probes.Add(V);
    end;
  end;

  // Build the classifier. Two hidden FullConnectLinear+activation blocks plus a
  // softmax head. We remember the two activation layer indices so we can probe
  // their outputs for dead units.
  procedure BuildNet(out NN: TNNet; Kind: TArmKind;
    out ActIdxA, ActIdxB: integer);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));

    NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));
    ActIdxA := NN.AddLayer(NewActivation(Kind)).LayerIdx;

    NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));
    ActIdxB := NN.AddLayer(NewActivation(Kind)).LayerIdx;

    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(cLR, 0.9);
  end;

  // Fraction of dead units across the two hidden activation layers, measured
  // exactly as TNNet.DeadNeuronReport defines it: a unit is dead if its maximum
  // absolute output over ALL probe samples is <= cDeadThr.
  function MeasureDeadFrac(NN: TNNet; Probes: TNNetVolumeList;
    const ActIdxA, ActIdxB: integer): TNeuralFloat;
  var
    MaxAbsA, MaxAbsB: array of TNeuralFloat;
    LayerA, LayerB: TNNetLayer;
    UnitsA, UnitsB, S, U, Dead, Total: integer;
    V: TNeuralFloat;
  begin
    LayerA := NN.Layers[ActIdxA];
    LayerB := NN.Layers[ActIdxB];
    UnitsA := LayerA.Output.Size;
    UnitsB := LayerB.Output.Size;
    SetLength(MaxAbsA, UnitsA);
    SetLength(MaxAbsB, UnitsB);
    for U := 0 to UnitsA - 1 do MaxAbsA[U] := 0;
    for U := 0 to UnitsB - 1 do MaxAbsB[U] := 0;

    for S := 0 to Probes.Count - 1 do
    begin
      NN.Compute(Probes[S]);
      for U := 0 to UnitsA - 1 do
      begin
        V := Abs(LayerA.Output.Raw[U]);
        if V > MaxAbsA[U] then MaxAbsA[U] := V;
      end;
      for U := 0 to UnitsB - 1 do
      begin
        V := Abs(LayerB.Output.Raw[U]);
        if V > MaxAbsB[U] then MaxAbsB[U] := V;
      end;
    end;

    Dead := 0;
    for U := 0 to UnitsA - 1 do if MaxAbsA[U] <= cDeadThr then Inc(Dead);
    for U := 0 to UnitsB - 1 do if MaxAbsB[U] <= cDeadThr then Inc(Dead);
    Total := UnitsA + UnitsB;
    if Total = 0 then Result := 0 else Result := Dead / Total;
  end;

var
  // The probe set is shared by all arms; declared here so RunArm can reach it.
  GProbes: TNNetVolumeList;

  procedure RunArm(Kind: TArmKind);
  var
    NN: TNNet;
    ActIdxA, ActIdxB: integer;
    Ep, B, I: integer;
    X, Yt, Out0: TNNetVolume;
    TotalLoss, Diff, DeadFrac, Peak: TNeuralFloat;
  begin
    // Reseed identically before every arm: same centres are already fixed, and
    // the same RNG stream now drives identical weight init + identical training
    // draws, so the ONLY difference between arms is the activation function.
    RandSeed := cSeed;
    BuildNet(NN, Kind, ActIdxA, ActIdxB);
    Peak := 0;
    try
      WriteLn;
      WriteLn(StringOfChar('=', 64));
      WriteLn(Format('ARM: %s  (LR=%.2f, %d hidden units x 2 layers)',
        [ArmName(Kind), cLR, cHidden]));
      WriteLn(StringOfChar('=', 64));
      WriteLn(Format('%-7s %-12s %-10s', ['epoch', 'mean-loss', 'dead%']));

      for Ep := 1 to cEpochs do
      begin
        TotalLoss := 0;
        for B := 1 to cBatch do
        begin
          MakePair(X, Yt);
          try
            NN.Compute(X);
            Out0 := NN.GetLastLayer.Output;
            for I := 0 to Out0.Size - 1 do
            begin
              Diff := Out0.Raw[I] - Yt.Raw[I];
              TotalLoss := TotalLoss + Diff * Diff;
            end;
            NN.Backpropagate(Yt);
          finally
            X.Free;
            Yt.Free;
          end;
        end;

        DeadFrac := MeasureDeadFrac(NN, GProbes, ActIdxA, ActIdxB);
        if DeadFrac > Peak then Peak := DeadFrac;

        if (Ep = 1) or (Ep mod 10 = 0) or (Ep = cEpochs) then
          WriteLn(Format('%-7d %-12.6f %8.2f%%',
            [Ep, TotalLoss / cBatch, DeadFrac * 100.0]));
      end;

      FinalDeadFrac[Kind] := MeasureDeadFrac(NN, GProbes, ActIdxA, ActIdxB);
      PeakDeadFrac[Kind] := Peak;
    finally
      NN.Free;
    end;
  end;

var
  K: TArmKind;
begin
  // Manual Compute/Backpropagate loop is inherently single-threaded; with a
  // fixed seed the whole run is fully reproducible.
  RandSeed := cSeed;

  // Fixed blob centres are drawn once and shared by every arm so the
  // classification task is literally identical across the four activations.
  InitCentres();
  BuildProbes(GProbes);

  try
    WriteLn('DeadReLUDiagnostic: per-epoch dead-unit trajectory across four');
    WriteLn('activations on a shared 3-blob classification task.');
    WriteLn('A unit is DEAD if |activation| <= ', cDeadThr:0:9,
      ' for every probe sample.');

    for K := Low(TArmKind) to High(TArmKind) do
      RunArm(K);

    WriteLn;
    WriteLn(StringOfChar('=', 64));
    WriteLn('SUMMARY: dead hidden-unit fraction (lower is healthier)');
    WriteLn(StringOfChar('=', 64));
    WriteLn(Format('%-16s %-12s %-12s', ['activation', 'peak dead%', 'final dead%']));
    WriteLn(StringOfChar('-', 64));
    for K := Low(TArmKind) to High(TArmKind) do
      WriteLn(Format('%-16s %10.2f%% %10.2f%%',
        [ArmName(K), PeakDeadFrac[K] * 100.0, FinalDeadFrac[K] * 100.0]));
    WriteLn;
    WriteLn('Expect TNNetReLU to show the highest dead fraction: once its');
    WriteLn('pre-activations go negative under the aggressive LR the units are');
    WriteLn('stuck at zero gradient. LeakyReLU/GELU/Swish leak a non-zero');
    WriteLn('gradient for negative inputs, so far fewer of their units die.');
  finally
    GProbes.Free;
  end;
end.
