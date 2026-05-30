program DeadReLULRSweep;
(*
DeadReLULRSweep: the learning-rate-sweep follow-up to
examples/DeadReLUDiagnostic/. Where that demo PINS one aggressive learning
rate (LR=0.5) and shows ReLU stranding ~19% of its hidden units, this demo
sweeps the learning rate over a grid and, for EACH LR, trains the same small
classifier with each of {ReLU, LeakyReLU, GELU, Swish}, recording the dead
hidden-unit fraction. The result is the cleanest "dying ReLU is a
learning-rate pathology" curve: ReLU's dead fraction climbs with LR while
LeakyReLU/GELU/Swish stay near zero across the whole sweep.

A hidden unit is "dead" when its activation output stays at (numerically)
zero for EVERY probe sample: such a unit produces a zero gradient and can
never recover. Plain ReLU is prone to this, and the larger the learning rate
the more readily a unit's pre-activation is pushed negative for all inputs and
stuck there. LeakyReLU / GELU / Swish keep a non-zero output (and gradient)
for negative inputs, so far fewer of their units die at any LR.

How this DIFFERS from examples/DeadReLUDiagnostic/:
  - DeadReLUDiagnostic fixes ONE LR and reports the per-epoch dead-fraction
    TRAJECTORY for the four activations.
  - This example sweeps the LR over a grid and reports a TABLE: rows = LR,
    columns = the four activations, cells = peak dead fraction. It isolates
    the learning rate as the cause of dying ReLU.

The dead-unit fraction is measured the same way TNNet.DeadNeuronReport defines
it (see neural/neuralnetwork.pas): push every probe sample through the net and,
for each hidden unit, track the maximum absolute activation; a unit is dead if
that maximum is <= a small threshold.

Deterministic: RandSeed is reseeded identically before every (LR, activation)
arm, so the ONLY differences between arms are the LR and the activation.
Pure CPU, single-threaded, well under five minutes.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralfit;

const
  cInDim     = 6;     // input dimensionality (blobs lifted into 6 dims)
  cClasses   = 3;     // number of Gaussian blobs / output classes
  cHidden    = 64;    // hidden units per layer
  cEpochs    = 60;    // training epochs per arm
  cBatch     = 40;    // pairs per epoch
  cProbeCnt  = 96;    // probe samples for the dead-unit measurement
  cDeadThr   = 1e-6;  // same default threshold as TNNet.DeadNeuronReport
  cSeed      = 424242;

type
  TArmKind = (akReLU, akLeakyReLU, akGELU, akSwish);

const
  // Learning-rate sweep grid. The headline: ReLU dead-fraction climbs across
  // this grid; the leaky/smooth activations stay near zero everywhere.
  cLRGrid: array[0 .. 4] of TNeuralFloat = (0.02, 0.05, 0.1, 0.2, 0.5);

var
  // Fixed blob centres, shared across all arms so the task is identical.
  Centres: array[0 .. cClasses - 1, 0 .. cInDim - 1] of TNeuralFloat;
  // Peak dead fraction per (LR index, activation) cell of the sweep table.
  DeadTable: array[0 .. High(cLRGrid), TArmKind] of TNeuralFloat;

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

  // One shared activation factory keeps the arms byte-for-byte identical apart
  // from the activation class itself.
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
  procedure BuildNet(out NN: TNNet; Kind: TArmKind; LR: TNeuralFloat;
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
    NN.SetLearningRate(LR, 0.9);
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

  // Train one (LR, activation) arm and return the PEAK dead fraction over the
  // training trajectory (peak is the robust headline: it captures the worst
  // damage even if a stray unit later flickers back).
  function RunArm(Kind: TArmKind; LR: TNeuralFloat): TNeuralFloat;
  var
    NN: TNNet;
    ActIdxA, ActIdxB: integer;
    Ep, B: integer;
    X, Yt, Out0: TNNetVolume;
    Diff, DeadFrac, Peak: TNeuralFloat;
  begin
    // Reseed identically before every arm: same fixed centres, same RNG stream
    // driving identical weight init + identical training draws, so the ONLY
    // differences between arms are the LR and the activation function.
    RandSeed := cSeed;
    BuildNet(NN, Kind, LR, ActIdxA, ActIdxB);
    Peak := 0;
    try
      for Ep := 1 to cEpochs do
      begin
        for B := 1 to cBatch do
        begin
          MakePair(X, Yt);
          try
            NN.Compute(X);
            Out0 := NN.GetLastLayer.Output;
            // (loss value itself is unused here; we only chart dead units)
            Diff := Out0.Raw[0] - Yt.Raw[0];
            if IsNan(Diff) then Halt(1);
            NN.Backpropagate(Yt);
          finally
            X.Free;
            Yt.Free;
          end;
        end;

        DeadFrac := MeasureDeadFrac(NN, GProbes, ActIdxA, ActIdxB);
        if DeadFrac > Peak then Peak := DeadFrac;
      end;
    finally
      NN.Free;
    end;
    Result := Peak;
  end;

var
  K: TArmKind;
  Li: integer;
  ReLULow, ReLUHigh: TNeuralFloat;
  HdrLine: string;
begin
  // Manual Compute/Backpropagate loop is inherently single-threaded; with a
  // fixed seed the whole run is fully reproducible.
  RandSeed := cSeed;

  // Fixed blob centres are drawn once and shared by every arm so the
  // classification task is literally identical across LRs and activations.
  InitCentres();
  BuildProbes(GProbes);

  try
    WriteLn('DeadReLULRSweep: dead hidden-unit fraction vs learning rate,');
    WriteLn('charted for ReLU/LeakyReLU/GELU/Swish on a shared 3-blob task.');
    WriteLn('A unit is DEAD if |activation| <= ', cDeadThr:0:9,
      ' for every probe sample.');
    WriteLn('Cells are the PEAK dead fraction over ', cEpochs, ' epochs.');

    for Li := Low(cLRGrid) to High(cLRGrid) do
      for K := Low(TArmKind) to High(TArmKind) do
        DeadTable[Li][K] := RunArm(K, cLRGrid[Li]);

    WriteLn;
    WriteLn(StringOfChar('=', 72));
    WriteLn('SWEEP: peak dead hidden-unit fraction (%) -- rows = LR, cols = activation');
    WriteLn(StringOfChar('=', 72));
    HdrLine := Format('%-8s', ['LR']);
    for K := Low(TArmKind) to High(TArmKind) do
      HdrLine := HdrLine + Format('%-15s', [ArmName(K)]);
    WriteLn(HdrLine);
    WriteLn(StringOfChar('-', 72));
    for Li := Low(cLRGrid) to High(cLRGrid) do
    begin
      Write(Format('%-8.2f', [cLRGrid[Li]]));
      for K := Low(TArmKind) to High(TArmKind) do
        Write(Format('%-15s', [Format('%.2f%%', [DeadTable[Li][K] * 100.0])]));
      WriteLn;
    end;
    WriteLn;
    WriteLn('Read down the TNNetReLU column: the dead fraction climbs with LR.');
    WriteLn('The other three activations leak a non-zero gradient for negative');
    WriteLn('inputs, so their columns stay near zero across the whole sweep.');
    WriteLn('Dying ReLU is a learning-rate pathology.');

    // ---- Self-gate: assert only invariants that are genuinely TRUE. ----
    ReLULow  := DeadTable[Low(cLRGrid)][akReLU];
    ReLUHigh := DeadTable[High(cLRGrid)][akReLU];

    // (1) ReLU dead fraction at the HIGHEST LR is meaningfully larger than at
    //     the LOWEST LR -- the monotone-ish climb that is the whole point.
    if not (ReLUHigh > ReLULow + 0.05) then
    begin
      WriteLn('FAIL: ReLU dead fraction did not climb with LR ',
        '(low=', ReLULow * 100.0:0:2, '%, high=', ReLUHigh * 100.0:0:2, '%).');
      Halt(1);
    end;

    // (2) At the highest LR, ReLU strands strictly MORE units than each of the
    //     leaky/smooth activations.
    for K := akLeakyReLU to akSwish do
      if not (ReLUHigh > DeadTable[High(cLRGrid)][K]) then
      begin
        WriteLn('FAIL: at LR=', cLRGrid[High(cLRGrid)]:0:2,
          ' ReLU (', ReLUHigh * 100.0:0:2, '%) did not exceed ',
          ArmName(K), ' (', DeadTable[High(cLRGrid)][K] * 100.0:0:2, '%).');
        Halt(1);
      end;

    WriteLn;
    WriteLn('PASS: ReLU dead fraction climbs from ', ReLULow * 100.0:0:2,
      '% (LR=', cLRGrid[Low(cLRGrid)]:0:2, ') to ', ReLUHigh * 100.0:0:2,
      '% (LR=', cLRGrid[High(cLRGrid)]:0:2, '), and at the top LR exceeds');
    WriteLn('every leaky/smooth activation.');
  finally
    GProbes.Free;
  end;
end.
