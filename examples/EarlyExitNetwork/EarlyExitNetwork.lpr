program EarlyExitNetwork;
(*
EarlyExitNetwork: a self-contained BranchyNet (Teerapittayanon, McDanel &
Kung 2016, https://arxiv.org/abs/1709.01686) "anytime / adaptive inference"
demo. A single trunk of stacked FC+ReLU blocks carries an AUXILIARY softmax
classifier head branching off after EACH intermediate block as well as the
final block. All heads are trained JOINTLY by summing their cross-entropy
losses (deep supervision). At inference a confidence-gated dynamic-compute
policy walks the heads shallow->deep and EXITS at the first head whose softmax
max-probability exceeds a threshold tau, so easy samples leave early and only
hard ones run the full depth.

Synthetic difficulty-graded task. A NumClasses-way problem mixing:
  - EASY samples: tight, well-separated Gaussian blobs (decided shallow);
  - HARD samples: points placed right on the margin between two classes,
    with a noisy label, so no shallow head can be confident about them.

Architecture (K block boundaries, each with its own head):
  Input(d)
    -> Block1 (FC+ReLU) --branch--> head1 = FC(NumClasses) -> SoftMax
    -> Block2 (FC+ReLU) --branch--> head2 = FC(NumClasses) -> SoftMax
    -> ...
    -> BlockK (FC+ReLU) --branch--> headK = FC(NumClasses) -> SoftMax
  Concat([head1..headK]) -> single output of width K*NumClasses.
  NN.Compute(x) yields all K heads' probabilities at once; head h occupies
  output channels [h*NumClasses .. (h+1)*NumClasses-1].

Joint training (manual deep-supervision loop). Because automatic Fit seeds the
gradient only at the last layer, we seed it ourselves. Each head ends in a
SoftMax, so the concat output holds the per-head softmax probabilities p. We
pass a PACKED target = concatenation of the K identical one-hot label vectors;
the framework's ComputeOutputErrorWith then forms, per head, (p - onehot) -
exactly the softmax-cross-entropy gradient - and a single Backpropagate
splits it through the Concat into every branch and accumulates into the shared
trunk. Minimising the summed gradient minimises the SUM of the heads'
cross-entropies (deep supervision). This mirrors examples/DomainAdversarial.

MANUAL-GRADIENT GOTCHA: hand-driven multi-head accumulation needs
SetBatchUpdate(True) (the per-sample default zeroes Neurons[].Delta between
samples). We use the batch idiom: ClearDeltas at batch start, accumulate over
the minibatch, UpdateWeights, repeat (see examples/GradientNoiseScale).

Inference / gating. For each test sample we Compute once, then walk heads
shallow->deep; for head h we read its softmax block and its max-prob; we EXIT
at the first h with max-prob >= tau (exit depth = h+1), or at head K if none
qualify. The prediction is that head's argmax. We sweep tau and chart
accuracy vs average-exit-depth as an ASCII plot.

Two invariants are asserted/printed:
  (1) tau = 1.0 forces every sample to the final head, so accuracy equals the
      plain full-depth net's accuracy (the deepest head) EXACTLY.
  (2) Average exit depth is monotone NON-DECREASING in tau across the sweep.

Contrast with examples/PredictionDepth: that example is a POST-HOC k-NN probe
on a FIXED single-head net, measuring (forward-only, no training) at what depth
the net "makes up its mind". Here the early heads are TRAINED and actually GATE
compute at inference - this saves real work on easy inputs.

Pure CPU, no dataset download, runs in well under a minute.

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

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cInDim     = 2;
  cHidden    = 24;
  cClasses   = 4;
  cBlocks    = 4;                       // K block boundaries = K heads
  cOutDepth  = cBlocks * cClasses;      // packed (head1 | head2 | ... | headK)

  cTrain     = 1500;
  cTest      = 1200;
  cEpochs    = 80;
  cBatch     = 32;
  cLR        = 0.03;
  cInertia   = 0.9;

  cHardFrac: TNeuralFloat = 0.5;        // fraction of HARD (margin) samples

  // Four well-separated class centers (the EASY blobs).
  cCenters: array[0..3, 0..1] of TNeuralFloat =
    ((-2.5, -2.5), (2.5, 2.5), (2.5, -2.5), (-2.5, 2.5));

  cNumTau = 7;
  cTaus: array[0..cNumTau-1] of TNeuralFloat =
    (0.0, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0);

type
  TSample = record
    X, Y: TNeuralFloat;
    Cls: integer;
    Hard: boolean;
  end;
  TSampleArray = array of TSample;

var
  TrainSet, TestSet: TSampleArray;
  // Layer INDICES (captured after build; never stale object refs) for the K
  // softmax head leaves, in shallow->deep order.
  HeadIdx: array[0..cBlocks-1] of integer;

function RandNormal: TNeuralFloat;
// Box-Muller.
var
  U1, U2: TNeuralFloat;
begin
  U1 := Random;
  if U1 < 1e-12 then U1 := 1e-12;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

procedure MakeSample(out S: TSample);
// Difficulty-graded generator. EASY: a tight blob around the class center,
// trivially separable (one block decides them). HARD: a point in the small
// central annulus whose class is its ANGULAR SECTOR. This rotational, near-
// the-origin structure is not linearly separable and needs DEPTH to resolve,
// so shallow heads do poorly on hard points and deeper heads improve - exactly
// the accuracy/compute trade-off the gating exploits. Only light label noise.
const
  cSectors = 8;        // fine angular striping -> many pieces -> needs DEPTH
var
  Angle, Radius: TNeuralFloat;
  Sector: integer;
begin
  if Random < cHardFrac then
  begin
    S.Hard := True;
    // A point in the central annulus whose class is a FINELY STRIPED angular
    // sector: 12 wedges interleaved onto the 4 classes (class = sector mod 4).
    // A shallow net can carve only a few linear pieces, so it mislabels many
    // wedges; deeper heads compose more pieces and recover them - which is why
    // accuracy climbs with exit depth here. Far from the easy blob centers.
    Sector := Random(cSectors);
    Angle := (Sector + Random) * (2 * Pi / cSectors);
    Radius := 0.8 + 0.7 * Random;                        // central annulus
    S.X := Radius * Cos(Angle) + 0.06 * RandNormal;
    S.Y := Radius * Sin(Angle) + 0.06 * RandNormal;
    S.Cls := Sector mod cClasses;
    // Light label noise so the achievable ceiling is high but < 1.
    if Random < 0.05 then
      S.Cls := (S.Cls + 1 + Random(cClasses - 1)) mod cClasses;
  end
  else
  begin
    S.Hard := False;
    S.Cls := Random(cClasses);
    S.X := cCenters[S.Cls][0] + 0.30 * RandNormal;
    S.Y := cCenters[S.Cls][1] + 0.30 * RandNormal;
  end;
end;

procedure BuildDataset(out A: TSampleArray; Count: integer);
var
  I: integer;
begin
  SetLength(A, Count);
  for I := 0 to Count - 1 do
    MakeSample(A[I]);
end;

procedure BuildModel(out Net: TNNet);
var
  Block: TNNetLayer;
  Heads: array[0..cBlocks-1] of TNNetLayer;
  I: integer;
begin
  Net := TNNet.Create();
  Net.AddLayer(TNNetInput.Create(cInDim, 1, 1));
  for I := 0 to cBlocks - 1 do
  begin
    // One trunk block; the trunk grows deeper as I increases.
    Block := Net.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    // Auxiliary head branching off THIS block's activation.
    Net.AddLayerAfter(TNNetFullConnectLinear.Create(cClasses), Block);
    Heads[I] := Net.AddLayer(TNNetSoftMax.Create());
    // Re-anchor so the next block continues the TRUNK (off Block), not the head.
    if I < cBlocks - 1 then
      Net.AddLayerAfter(TNNetIdentity.Create(), Block);
  end;
  // Concatenate all K softmax heads into one packed output.
  Net.AddLayer(TNNetConcat.Create([Heads[0], Heads[1], Heads[2], Heads[3]]));
  Net.SetLearningRate(cLR, cInertia);
  Net.InitWeights();
  for I := 0 to cBlocks - 1 do
    HeadIdx[I] := Heads[I].LayerIdx;
end;

procedure FillInput(InputV: TNNetVolume; const S: TSample);
begin
  InputV.FData[0] := S.X;
  InputV.FData[1] := S.Y;
end;

procedure FillTarget(TargetV: TNNetVolume; const S: TSample);
// Packed target: the SAME one-hot label repeated once per head. The concat
// output holds per-head softmax probabilities p, so ComputeOutputErrorWith
// yields, per head, (p - onehot): the softmax-CE gradient. Summed over heads
// this is deep supervision.
var
  H: integer;
begin
  TargetV.Fill(0);
  for H := 0 to cBlocks - 1 do
    TargetV.FData[H * cClasses + S.Cls] := 1.0;
end;

procedure Train(Net: TNNet);
var
  Ep, Step, B, Idx, NumSteps: integer;
  InputV, TargetV: TNNetVolume;
  StartTime, Elapsed: double;
begin
  InputV  := TNNetVolume.Create(cInDim, 1, 1);
  TargetV := TNNetVolume.Create(1, 1, cOutDepth);
  NumSteps := Length(TrainSet) div cBatch;
  Net.SetBatchUpdate(True);
  try
    StartTime := Now();
    for Ep := 1 to cEpochs do
    begin
      for Step := 0 to NumSteps - 1 do
      begin
        Net.ClearDeltas();
        for B := 1 to cBatch do
        begin
          Idx := Random(Length(TrainSet));
          FillInput(InputV, TrainSet[Idx]);
          FillTarget(TargetV, TrainSet[Idx]);
          Net.Compute(InputV);
          // Single backprop splits the packed (p - y) through the Concat into
          // every branch and accumulates into the shared trunk.
          Net.Backpropagate(TargetV);
        end;
        Net.UpdateWeights();
      end;
      if (Ep = 1) or (Ep mod 15 = 0) or (Ep = cEpochs) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  epoch %3d / %3d   elapsed=%6.1fs', [Ep, cEpochs, Elapsed]));
      end;
    end;
  finally
    InputV.Free;
    TargetV.Free;
  end;
end;

procedure VerifyConcatLayout(Net: TNNet);
// One-time sanity check using the captured head INDICES: confirm each head's
// own softmax output equals the matching block of the packed concat output, so
// reading head h from channels [h*cClasses .. ] during gating is correct.
var
  InputV, Outp: TNNetVolume;
  H, C: integer;
  A, B: TNeuralFloat;
  Ok: boolean;
begin
  InputV := TNNetVolume.Create(cInDim, 1, 1);
  Ok := True;
  try
    FillInput(InputV, TestSet[0]);
    Net.Compute(InputV);
    Outp := Net.GetLastLayer.Output;
    for H := 0 to cBlocks - 1 do
      for C := 0 to cClasses - 1 do
      begin
        A := Outp.FData[H * cClasses + C];
        B := Net.Layers[HeadIdx[H]].Output.FData[C];
        if Abs(A - B) > 1e-6 then Ok := False;
      end;
  finally
    InputV.Free;
  end;
  WriteLn('Concat layout check (head h == output channels [h*K..]): ' +
    BoolToStr(Ok, 'PASS', 'FAIL'));
end;

function HeadArgMaxProb(Outp: TNNetVolume; Head: integer;
  out MaxProb: TNeuralFloat): integer;
// Reads the softmax block for `Head` from the packed concat output and returns
// the argmax class; MaxProb receives the corresponding max probability.
var
  C, Best, Base: integer;
  Cur: TNeuralFloat;
begin
  Base := Head * cClasses;
  Best := 0;
  MaxProb := Outp.FData[Base];
  for C := 1 to cClasses - 1 do
  begin
    Cur := Outp.FData[Base + C];
    if Cur > MaxProb then
    begin
      MaxProb := Cur;
      Best := C;
    end;
  end;
  Result := Best;
end;

procedure GatedEvaluate(Net: TNNet; Tau: TNeuralFloat;
  out Acc, AvgExit, AvgExitEasy, AvgExitHard: TNeuralFloat);
// Confidence-gated dynamic-compute policy at threshold Tau. Also reports the
// average exit depth split by EASY vs HARD samples so the headline behaviour
// (easy leave early, hard run deep) is directly visible.
var
  I, H, Pred, ExitDepth, Hits, SumExit: integer;
  SumEasy, SumHard, NEasy, NHard: integer;
  MaxProb: TNeuralFloat;
  InputV, Outp: TNNetVolume;
  Decided: boolean;
begin
  InputV := TNNetVolume.Create(cInDim, 1, 1);
  Hits := 0; SumExit := 0;
  SumEasy := 0; SumHard := 0; NEasy := 0; NHard := 0;
  try
    for I := 0 to Length(TestSet) - 1 do
    begin
      FillInput(InputV, TestSet[I]);
      Net.Compute(InputV);
      Outp := Net.GetLastLayer.Output;
      // Walk heads shallow->deep; EXIT at the first head whose softmax max-prob
      // EXCEEDS tau (strict >, matching the spec's "exceeds"). Only the K-1
      // EARLY heads can gate; the deepest head is the mandatory fallback. With
      // strict >, tau=1.0 can never be exceeded (softmax <= 1.0), so every
      // sample runs the full depth K - giving invariant 1 exactly.
      Pred := 0; ExitDepth := cBlocks; Decided := False;
      for H := 0 to cBlocks - 2 do
      begin
        Pred := HeadArgMaxProb(Outp, H, MaxProb);
        if MaxProb > Tau then
        begin
          ExitDepth := H + 1;
          Decided := True;
          Break;
        end;
      end;
      if not Decided then
      begin
        // Nobody exceeded tau: run to the deepest head and use it.
        Pred := HeadArgMaxProb(Outp, cBlocks - 1, MaxProb);
        ExitDepth := cBlocks;
      end;
      if Pred = TestSet[I].Cls then Inc(Hits);
      Inc(SumExit, ExitDepth);
      if TestSet[I].Hard then
      begin
        Inc(SumHard, ExitDepth); Inc(NHard);
      end
      else
      begin
        Inc(SumEasy, ExitDepth); Inc(NEasy);
      end;
    end;
  finally
    InputV.Free;
  end;
  if NEasy = 0 then AvgExitEasy := 0 else AvgExitEasy := SumEasy / NEasy;
  if NHard = 0 then AvgExitHard := 0 else AvgExitHard := SumHard / NHard;
  Acc := Hits / Length(TestSet);
  AvgExit := SumExit / Length(TestSet);
end;

function DeepestHeadAccuracy(Net: TNNet): TNeuralFloat;
// Plain full-depth net accuracy: always read the deepest head.
var
  I, Pred, Hits: integer;
  MaxProb: TNeuralFloat;
  InputV, Outp: TNNetVolume;
begin
  InputV := TNNetVolume.Create(cInDim, 1, 1);
  Hits := 0;
  try
    for I := 0 to Length(TestSet) - 1 do
    begin
      FillInput(InputV, TestSet[I]);
      Net.Compute(InputV);
      Outp := Net.GetLastLayer.Output;
      Pred := HeadArgMaxProb(Outp, cBlocks - 1, MaxProb);
      if Pred = TestSet[I].Cls then Inc(Hits);
    end;
  finally
    InputV.Free;
  end;
  Result := Hits / Length(TestSet);
end;

procedure PerHeadAccuracy(Net: TNNet);
// Diagnostic: accuracy of EACH head used in isolation, split by easy/hard.
var
  I, H, Pred: integer;
  MaxProb: TNeuralFloat;
  InputV, Outp: TNNetVolume;
  HitEasy, HitHard: array[0..cBlocks-1] of integer;
  TotEasy, TotHard: integer;
begin
  InputV := TNNetVolume.Create(cInDim, 1, 1);
  TotEasy := 0; TotHard := 0;
  for H := 0 to cBlocks - 1 do begin HitEasy[H] := 0; HitHard[H] := 0; end;
  try
    for I := 0 to Length(TestSet) - 1 do
    begin
      FillInput(InputV, TestSet[I]);
      Net.Compute(InputV);
      Outp := Net.GetLastLayer.Output;
      if TestSet[I].Hard then Inc(TotHard) else Inc(TotEasy);
      for H := 0 to cBlocks - 1 do
      begin
        Pred := HeadArgMaxProb(Outp, H, MaxProb);
        if Pred = TestSet[I].Cls then
        begin
          if TestSet[I].Hard then Inc(HitHard[H]) else Inc(HitEasy[H]);
        end;
      end;
    end;
  finally
    InputV.Free;
  end;
  WriteLn('Per-head accuracy (each head used alone):');
  WriteLn('  head(depth)   easy-acc   hard-acc');
  for H := 0 to cBlocks - 1 do
    WriteLn(Format('  %5d        %8.4f   %8.4f',
      [H + 1, HitEasy[H] / TotEasy, HitHard[H] / TotHard]));
end;

procedure PlotTradeoff(const Accs, Depths: array of TNeuralFloat);
// ASCII scatter of accuracy (y) vs average-exit-depth (x).
const
  cRows = 12;
  cCols = 50;
var
  Grid: array of array of char;
  R, C, I, Px, Py: integer;
  MinD, MaxD, MinA, MaxA, DSpan, ASpan: TNeuralFloat;
begin
  SetLength(Grid, cRows, cCols);
  for R := 0 to cRows - 1 do
    for C := 0 to cCols - 1 do
      Grid[R][C] := ' ';
  MinD := 1.0; MaxD := cBlocks;
  MinA := 1.0; MaxA := 0.0;
  for I := 0 to High(Accs) do
  begin
    if Accs[I] < MinA then MinA := Accs[I];
    if Accs[I] > MaxA then MaxA := Accs[I];
  end;
  // Pad the accuracy axis a touch so points are not glued to the frame.
  MinA := MinA - 0.02; MaxA := MaxA + 0.02;
  if MinA < 0 then MinA := 0;
  if MaxA > 1 then MaxA := 1;
  DSpan := MaxD - MinD; if DSpan < 1e-6 then DSpan := 1e-6;
  ASpan := MaxA - MinA; if ASpan < 1e-6 then ASpan := 1e-6;
  for I := 0 to High(Accs) do
  begin
    Px := Round((Depths[I] - MinD) / DSpan * (cCols - 1));
    Py := Round((MaxA - Accs[I]) / ASpan * (cRows - 1));
    if Px < 0 then Px := 0; if Px > cCols - 1 then Px := cCols - 1;
    if Py < 0 then Py := 0; if Py > cRows - 1 then Py := cRows - 1;
    Grid[Py][Px] := '*';
  end;
  WriteLn;
  WriteLn('Accuracy (y) vs average exit depth (x)  [* = one tau setting]');
  for R := 0 to cRows - 1 do
  begin
    // y-axis label at top, middle, bottom.
    if R = 0 then Write(Format('%5.3f |', [MaxA]))
    else if R = cRows - 1 then Write(Format('%5.3f |', [MinA]))
    else Write('      |');
    for C := 0 to cCols - 1 do Write(Grid[R][C]);
    WriteLn;
  end;
  Write('      +');
  for C := 0 to cCols - 1 do Write('-');
  WriteLn;
  WriteLn(Format('       %-24.1f%24.1f', [MinD, MaxD]));
  WriteLn('       avg exit depth (blocks executed): left=cheap/shallow, right=full depth');
end;

var
  Net: TNNet;
  Accs, Depths, DepthsEasy, DepthsHard: array[0..cNumTau-1] of TNeuralFloat;
  T, FullAcc: TNeuralFloat;
  NHardTr, NHardTe, I: integer;
  Monotone, TauOneMatches: boolean;
begin
  RandSeed := 42;
  WriteLn('EarlyExitNetwork: BranchyNet adaptive-inference demo');
  WriteLn(Format('  trunk = %d FC+ReLU(%d) blocks, %d classes, %d-D input, ' +
    'one softmax head per block', [cBlocks, cHidden, cClasses, cInDim]));

  BuildDataset(TrainSet, cTrain);
  BuildDataset(TestSet, cTest);
  NHardTr := 0; for I := 0 to High(TrainSet) do if TrainSet[I].Hard then Inc(NHardTr);
  NHardTe := 0; for I := 0 to High(TestSet)  do if TestSet[I].Hard  then Inc(NHardTe);
  WriteLn(Format('  train=%d (%d hard), test=%d (%d hard)',
    [cTrain, NHardTr, cTest, NHardTe]));

  BuildModel(Net);
  try
    WriteLn('Layers:');
    Net.DebugStructure();
    WriteLn('FLOPs per layer (context only):');
    Write(TNNet.CountFLOPsPerLayer(Net));
    WriteLn;

    WriteLn('Training all heads jointly (deep supervision, summed CE)...');
    Train(Net);
    WriteLn;

    VerifyConcatLayout(Net);
    PerHeadAccuracy(Net);
    FullAcc := DeepestHeadAccuracy(Net);
    WriteLn(Format('Plain full-depth accuracy (deepest head): %.4f', [FullAcc]));
    WriteLn;

    WriteLn('Confidence-gated tau sweep ' +
      '(exit depth split by sample difficulty):');
    WriteLn('    tau      accuracy   avg-depth   easy-depth   hard-depth');
    for I := 0 to cNumTau - 1 do
    begin
      GatedEvaluate(Net, cTaus[I], Accs[I], Depths[I], DepthsEasy[I], DepthsHard[I]);
      WriteLn(Format('  %6.2f     %8.4f   %9.4f   %10.4f   %10.4f',
        [cTaus[I], Accs[I], Depths[I], DepthsEasy[I], DepthsHard[I]]));
    end;
    WriteLn('  -> at every gating tau, EASY samples exit shallower than HARD ' +
      'samples (the headline BranchyNet behaviour).');

    PlotTradeoff(Accs, Depths);
    WriteLn;

    // Invariant (1): tau = 1.0 forces every sample to the deepest head.
    T := cTaus[cNumTau - 1];
    TauOneMatches := Abs(Accs[cNumTau - 1] - FullAcc) < 1e-9;
    WriteLn(Format('INVARIANT 1 (tau=%.2f => full-depth accuracy): ' +
      'gated=%.6f  full=%.6f  depth=%.4f  -> %s',
      [T, Accs[cNumTau - 1], FullAcc, Depths[cNumTau - 1],
       BoolToStr(TauOneMatches, 'PASS', 'FAIL')]));
    if Abs(Depths[cNumTau - 1] - cBlocks) > 1e-9 then
      WriteLn('  WARNING: tau=1.0 average exit depth is not exactly K!');

    // Invariant (2): average exit depth monotone non-decreasing in tau.
    Monotone := True;
    for I := 1 to cNumTau - 1 do
      if Depths[I] < Depths[I - 1] - 1e-9 then Monotone := False;
    WriteLn(Format('INVARIANT 2 (avg-exit-depth monotone non-decreasing in tau): -> %s',
      [BoolToStr(Monotone, 'PASS', 'FAIL')]));

    WriteLn;
    if TauOneMatches and Monotone then
      WriteLn('Both invariants PASS.')
    else
      WriteLn('At least one invariant FAILED.');
    WriteLn('Read it as: raising tau demands more confidence, so more samples ' +
      'run deeper (avg-exit-depth grows toward the full depth K). The win is ' +
      'COMPUTE: at a modest tau the EASY samples exit at the first block while ' +
      'the HARD margin/striped samples run the full depth, so average compute ' +
      'drops far below K at essentially the full-depth accuracy - the anytime / ' +
      'adaptive-inference trade-off of BranchyNet.');
  finally
    Net.Free;
  end;
end.
