program CumSumPositionEncoding;
(*
CumSumPositionEncoding: demonstration of how TNNetCumSum can be used to
manufacture a *learned-free* linear position feature, PLUS a train-time
bake-off proving the feature measurably helps on a position-dependent task.

The trick:
  If the input is a constant 1.0 along the depth axis, then
      CumSum_depth(Input)[c] = c + 1
  This is a strictly increasing per-position ramp — exactly the kind
  of "where am I in the sequence?" signal that a permutation-invariant
  downstream layer needs in order to behave position-aware.

PART 1 (forward-only demo, retained from the original landing): a pure
forward pass through a network whose only meaningful layer is TNNetCumSum,
verifying the three canonical patterns:

  1. all-ones input  → [1, 2, 3, ...] linear ramp,
  2. arbitrary input → standard prefix-sum,
  3. multi-row input → each (X,Y) location accumulates independently
     along its own depth column.

PART 2 (train-time bake-off, the open follow-up): we actually TRAIN a
tiny permutation-invariant model on a task whose answer depends on token
POSITION, twice -- once WITHOUT a position feature and once WITH the
CumSum ramp concatenated -- and chart the loss/accuracy delta.

  Task ("find the marker"): a length-cSeqLen sequence of one-hot tokens.
  Exactly one position holds a special MARKER token; the target class is
  the POSITION INDEX of that marker (cSeqLen classes). The model is
  deliberately permutation-invariant over positions: each token is passed
  through a SHARED pointwise transform (same weights at every position)
  and then the positions are AVERAGE-pooled into a single summary vector
  before classification. Shared per-token weights + an order-agnostic pool
  means the network has NO way to know which slot a token sits in -- unless
  each token carries its own position in its feature vector.

    NoPos arm : the marker token is identical wherever it sits, so the
                pooled summary is the same regardless of position -> the
                net cannot beat chance (1/cSeqLen).
    CumSum arm: a TNNetCumSum-of-constant ramp is concatenated to every
                token's features, stamping each position with a distinct
                increasing value. The marker now rides a position-specific
                value into the pool, so the net can read the position off
                the summary and solve the task.

  Both arms use the IDENTICAL architecture, identical seed, identical
  epochs and an identical-shape input (the NoPos arm pads with a constant
  channel so parameter counts match exactly); only the extra channel's
  CONTENT differs (constant vs CumSum ramp). The bake-off prints final
  cross-entropy loss and accuracy for each arm plus the delta, and gates
  (Halt(1) on failure) that the CumSum arm's loss is meaningfully below the
  NoPos arm and that it clears a real accuracy bar while NoPos stays at
  chance.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSeqLen = 8;

  // Builds a tiny 1-layer pipeline:  Input(1,1,Depth) -> CumSum
  procedure BuildCumSum1D(out NN: TNNet; ADepth: integer);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, 1, ADepth));
    NN.AddLayer(TNNetCumSum.Create());
  end;

  // Builds a 2D pipeline so we can show per-row independence:
  //   Input(1, Rows, Depth) -> CumSum  (each row sums on its own)
  procedure BuildCumSum2D(out NN: TNNet; ARows, ADepth: integer);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, ARows, ADepth));
    NN.AddLayer(TNNetCumSum.Create());
  end;

  procedure PrintDepth(const Tag: string; V: TNNetVolume);
  var
    D: integer;
  begin
    Write(Tag, ' [');
    for D := 0 to V.Depth - 1 do
    begin
      if D > 0 then Write(', ');
      Write(V.FData[D]:6:2);
    end;
    WriteLn(']');
  end;

  procedure DemoLinearRamp();
  var
    NN: TNNet;
    Input, Output: TNNetVolume;
    D: integer;
    AllOk: boolean;
  begin
    WriteLn('--- Demo 1: constant-1 input  ->  linear position ramp ---');
    BuildCumSum1D(NN, cSeqLen);
    Input  := TNNetVolume.Create(1, 1, cSeqLen, 1.0); // all ones
    Output := TNNetVolume.Create(1, 1, cSeqLen, 0.0);

    NN.Compute(Input);
    NN.GetOutput(Output);

    PrintDepth('  input  =', Input);
    PrintDepth('  cumsum =', Output);

    AllOk := True;
    for D := 0 to cSeqLen - 1 do
      if Abs(Output.FData[D] - (D + 1)) > 1e-6 then AllOk := False;
    if AllOk then
      WriteLn('  OK: CumSum of all-ones is the position index + 1.')
    else
      WriteLn('  FAIL: ramp does not match expected values.');

    Output.Free;
    Input.Free;
    NN.Free;
    WriteLn;
  end;

  procedure DemoArbitraryPrefixSum();
  var
    NN: TNNet;
    Input, Output: TNNetVolume;
    Vals: array[0..5] of TNeuralFloat = (1, 2, 3, 4, 5, 6);
    Expected: array[0..5] of TNeuralFloat = (1, 3, 6, 10, 15, 21);
    D: integer;
    AllOk: boolean;
  begin
    WriteLn('--- Demo 2: arbitrary input   ->  standard prefix-sum ---');
    BuildCumSum1D(NN, Length(Vals));
    Input  := TNNetVolume.Create(1, 1, Length(Vals), 0.0);
    Output := TNNetVolume.Create(1, 1, Length(Vals), 0.0);
    for D := 0 to Length(Vals) - 1 do Input.FData[D] := Vals[D];

    NN.Compute(Input);
    NN.GetOutput(Output);

    PrintDepth('  input  =', Input);
    PrintDepth('  cumsum =', Output);

    AllOk := True;
    for D := 0 to Length(Vals) - 1 do
      if Abs(Output.FData[D] - Expected[D]) > 1e-6 then AllOk := False;
    if AllOk then
      WriteLn('  OK: prefix-sum matches the textbook result.')
    else
      WriteLn('  FAIL: prefix-sum mismatched.');

    Output.Free;
    Input.Free;
    NN.Free;
    WriteLn;
  end;

  procedure DemoPerRowIndependence();
  const
    cRows  = 3;
    cDepth = 5;
  var
    NN: TNNet;
    Input, Output: TNNetVolume;
    Y, D: integer;
    RowVal: TNeuralFloat;
    AllOk: boolean;
  begin
    WriteLn('--- Demo 3: multi-row input   ->  each row sums independently ---');
    BuildCumSum2D(NN, cRows, cDepth);
    Input  := TNNetVolume.Create(1, cRows, cDepth, 0.0);
    Output := TNNetVolume.Create(1, cRows, cDepth, 0.0);

    // Row y has constant value (y + 1) along the depth axis.
    // After CumSum, row y should be:  [(y+1)*1, (y+1)*2, ..., (y+1)*cDepth].
    for Y := 0 to cRows - 1 do
      for D := 0 to cDepth - 1 do
        Input[0, Y, D] := (Y + 1);

    NN.Compute(Input);
    NN.GetOutput(Output);

    AllOk := True;
    for Y := 0 to cRows - 1 do
    begin
      Write('  row ', Y, '  in=');
      for D := 0 to cDepth - 1 do
      begin
        if D > 0 then Write(',');
        Write(Input[0, Y, D]:4:1);
      end;
      Write('   out=');
      for D := 0 to cDepth - 1 do
      begin
        RowVal := (Y + 1) * (D + 1);
        if D > 0 then Write(',');
        Write(Output[0, Y, D]:5:1);
        if Abs(Output[0, Y, D] - RowVal) > 1e-6 then AllOk := False;
      end;
      WriteLn;
    end;

    if AllOk then
      WriteLn('  OK: each row accumulates only its own column.')
    else
      WriteLn('  FAIL: row coupling detected.');

    Output.Free;
    Input.Free;
    NN.Free;
    WriteLn;
  end;

  procedure DemoUsageHint();
  begin
    WriteLn('--- Usage hint ---');
    WriteLn('  To inject a learned-free position feature, concatenate a');
    WriteLn('  CumSum-of-constant branch alongside your real features:');
    WriteLn;
    WriteLn('    Real    : TNNetInput(1,1,Depth)');
    WriteLn('    PosFeat : TNNetInput(1,1,Depth, FillValue=1) -> TNNetCumSum');
    WriteLn('    Merged  : TNNetConcat([Real, PosFeat])  -> downstream layers');
    WriteLn;
    WriteLn('  The downstream model now sees a strictly-increasing position');
    WriteLn('  index in the extra channels, without any trainable parameter.');
    WriteLn;
  end;

// ===========================================================================
// PART 2: train-time bake-off on a POSITION-DEPENDENT task.
// ===========================================================================
//
// Task "find the marker": a length-cSeqLen sequence of one-hot tokens over a
// tiny vocab. Exactly one position holds the MARKER token (vocab index 0);
// every other position holds a uniformly-random non-marker token. The target
// class is the POSITION of the marker (cSeqLen classes).
//
// Input volume layout: (X=cSeqLen positions, Y=1, Depth=cFeat features).
// Per-token feature vector (cFeat = cVocab + 1):
//     [ one-hot(token) over cVocab slots , extra-channel ]
// where the extra channel is:
//     NoPos  arm : constant 1.0           (carries NO position information)
//     CumSum arm : the CumSum ramp value  (position index + 1; distinct per slot)
//
// The model is permutation-invariant over positions on purpose:
//     Input(cSeqLen,1,cFeat)
//       -> PointwiseConvReLU(cHidden)   // SHARED transform applied per token
//       -> MaxChannel                   // order-agnostic pool over positions
//       -> FullConnectLinear(cSeqLen)
//       -> SoftMax
// Shared per-token weights + an order-agnostic pool => the net cannot tell
// which slot a token occupies from token identity alone. Only the CumSum arm,
// whose extra channel stamps each position with a distinct value, can route
// the marker's position through the pool.
// ===========================================================================

const
  cBakeSeed   = 424242;
  cVocab      = 4;            // token vocabulary size (index 0 = MARKER)
  cFeat       = cVocab + 1;   // one-hot + 1 extra (position/constant) channel
  cHidden     = 48;           // shared per-token hidden width
  cTrainN     = 800;          // training sequences
  cTestN      = 800;          // held-out test sequences
  cBakeEpochs = 300;          // identical epoch budget for both arms
  cBakeBatch  = 20;           // mini-batch size
  cBakeLR     = 0.02;
  cBakeMom    = 0.9;
  cBakeChance = 1.0 / cSeqLen;

type
  TBakeResult = record
    Name     : string;
    Loss     : TNeuralFloat;   // mean cross-entropy on the test set
    Acc      : TNeuralFloat;   // argmax accuracy on the test set
    Params   : integer;
  end;

// Build ONE labelled sequence into (Pos)/(Target). MarkerPos is the slot that
// holds the marker; the target class equals MarkerPos. The extra channel is
// filled by the caller's arm (left at the raw value here).
//   UsePos = True  -> extra channel = CumSum ramp (pos + 1)
//   UsePos = False -> extra channel = constant 1.0
procedure MakeSequence(X, Y: TNNetVolume; UsePos: boolean);
var
  Pos, MarkerPos, Tok: integer;
begin
  X.Fill(0);
  MarkerPos := Random(cSeqLen);
  for Pos := 0 to cSeqLen - 1 do
  begin
    if Pos = MarkerPos then
      Tok := 0                       // the MARKER token
    else
      Tok := 1 + Random(cVocab - 1); // a random non-marker token
    X[Pos, 0, Tok] := 1.0;           // one-hot over the first cVocab channels
    // extra channel (index cVocab): position ramp or a flat constant.
    if UsePos then
      X[Pos, 0, cVocab] := Pos + 1   // == CumSum-of-ones ramp at this slot
    else
      X[Pos, 0, cVocab] := 1.0;      // constant: no positional signal
  end;
  Y.Fill(0);
  Y.SetClassForSoftMax(MarkerPos);
end;

// Build a paired train/test set. The two arms (UsePos True/False) are built
// from the SAME RNG stream by reseeding before each call, so the marker
// positions and distractor tokens are identical across arms -- only the extra
// channel's CONTENT differs.
procedure BuildBakeSet(out Pairs: TNNetVolumePairList; Count: integer;
  UsePos: boolean);
var
  I: integer;
  X, Y: TNNetVolume;
begin
  Pairs := TNNetVolumePairList.Create();
  for I := 0 to Count - 1 do
  begin
    X := TNNetVolume.Create(cSeqLen, 1, cFeat);
    Y := TNNetVolume.Create(cSeqLen, 1, 1);
    MakeSequence(X, Y, UsePos);
    Pairs.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// The FIXED permutation-invariant classifier (identical for both arms).
procedure BuildBakeNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cSeqLen, 1, cFeat));
  NN.AddLayer(TNNetPointwiseConvReLU.Create(cHidden)); // shared per-token transform
  NN.AddLayer(TNNetMaxChannel.Create());               // order-agnostic pool
  NN.AddLayer(TNNetFullConnectLinear.Create(cSeqLen));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(cBakeLR, cBakeMom);
  // A small weight decay keeps the SoftMax logits from running away, so the
  // reported cross-entropy reflects how well each arm SOLVES the task rather
  // than how overconfident a handful of wrong predictions became.
  NN.SetL2Decay(0.001);
  NN.SetBatchUpdate(True);
end;

// Mean cross-entropy + argmax accuracy of NN over a pair list.
procedure EvalBake(NN: TNNet; Pairs: TNNetVolumePairList;
  out Loss, Acc: TNeuralFloat);
var
  I, Hits: integer;
  P, SumLoss: TNeuralFloat;
  Out: TNNetVolume;
begin
  Hits := 0;
  SumLoss := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    Out := NN.GetLastLayer().Output;
    if Out.GetClass() = Pairs[I].O.GetClass() then Inc(Hits);
    P := Out.FData[Pairs[I].O.GetClass()];   // SoftMax prob of the true class
    SumLoss := SumLoss - Ln(Max(P, 1e-12));
  end;
  if Pairs.Count > 0 then
  begin
    Loss := SumLoss / Pairs.Count;
    Acc  := Hits / Pairs.Count;
  end
  else begin Loss := 0; Acc := 0; end;
end;

// Train one arm: build the net under the fixed seed (identical weight init),
// run cBakeEpochs of mini-batch SGD, then report test loss/accuracy.
function RunBakeArm(const Name: string; UsePos: boolean): TBakeResult;
var
  NN: TNNet;
  Train, Test: TNNetVolumePairList;
  Epoch, I, J, B, Tmp, InBatch: integer;
  Order: array of integer;
begin
  Result.Name := Name;
  // Same data seed for both arms => identical sequences/marker positions.
  RandSeed := cBakeSeed;
  BuildBakeSet(Train, cTrainN, UsePos);
  BuildBakeSet(Test,  cTestN,  UsePos);
  // Reseed before build so weight init is IDENTICAL across both arms.
  RandSeed := cBakeSeed + 7;
  BuildBakeNet(NN);
  Result.Params := NN.CountWeights();
  SetLength(Order, Train.Count);
  for I := 0 to High(Order) do Order[I] := I;
  try
    for Epoch := 1 to cBakeEpochs do
    begin
      for I := High(Order) downto 1 do
      begin
        J := Random(I + 1);
        Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
      end;
      InBatch := 0;
      NN.ClearDeltas();
      for B := 0 to High(Order) do
      begin
        NN.Compute(Train[Order[B]].I);
        NN.Backpropagate(Train[Order[B]].O);
        Inc(InBatch);
        if (InBatch >= cBakeBatch) or (B = High(Order)) then
        begin
          NN.UpdateWeights();
          NN.ClearDeltas();
          InBatch := 0;
        end;
      end;
    end;
    EvalBake(NN, Test, Result.Loss, Result.Acc);
  finally
    NN.Free;
    Train.Free;
    Test.Free;
  end;
end;

procedure RunBakeOff();
var
  NoPos, WithPos: TBakeResult;
  LossDelta: TNeuralFloat;
  PassLoss, PassAcc, PassChance, PassAll: boolean;
const
  cLossMargin = 0.20;  // CumSum loss must be at least this much below NoPos
  cAccBar     = 0.70;  // CumSum arm must clear this accuracy (>> chance)
  cChanceTol  = 0.10;  // NoPos arm must stay within chance + this
begin
  WriteLn('================================================================');
  WriteLn('PART 2: train-time bake-off on a POSITION-DEPENDENT task.');
  WriteLn('================================================================');
  WriteLn(Format('Task "find the marker": SeqLen=%d, vocab=%d, %d classes '
    + '(chance=%.3f).', [cSeqLen, cVocab, cSeqLen, cBakeChance]));
  WriteLn(Format('Net: Input(%d,1,%d)->PointwiseConvReLU(%d)->MaxChannel'
    + '->FullConnectLinear(%d)->SoftMax  (shared per-token + order-agnostic pool).',
    [cSeqLen, cFeat, cHidden, cSeqLen]));
  WriteLn(Format('Mini-batch SGD  batch=%d  LR=%.3f  mom=%.2f  epochs=%d  seed=%d.',
    [cBakeBatch, cBakeLR, cBakeMom, cBakeEpochs, cBakeSeed]));
  WriteLn('Same net/seed/epochs; only the extra channel differs (constant vs CumSum ramp).');
  WriteLn;

  Write('Training NoPos arm  (extra channel = constant 1.0) ... ');
  NoPos := RunBakeArm('NoPos (constant)', False);
  WriteLn('done.');
  Write('Training CumSum arm (extra channel = CumSum ramp)  ... ');
  WithPos := RunBakeArm('CumSum (ramp)   ', True);
  WriteLn('done.');
  WriteLn;

  LossDelta := NoPos.Loss - WithPos.Loss;
  WriteLn('=== Bake-off results (held-out test set) ===');
  WriteLn('arm                params    TEST loss    TEST acc');
  WriteLn(Format('%-17s  %6d    %8.4f    %6.2f%%',
    [NoPos.Name, NoPos.Params, NoPos.Loss, NoPos.Acc * 100]));
  WriteLn(Format('%-17s  %6d    %8.4f    %6.2f%%',
    [WithPos.Name, WithPos.Params, WithPos.Loss, WithPos.Acc * 100]));
  WriteLn(Format('chance accuracy (1/SeqLen) = %.2f%%', [cBakeChance * 100]));
  WriteLn(Format('loss delta (NoPos - CumSum) = %.4f   (positive => CumSum wins)',
    [LossDelta]));
  WriteLn;

  WriteLn('=== Correctness gate ===');
  PassLoss := LossDelta >= cLossMargin;
  WriteLn(Format('[%s] CumSum loss %.4f is >= %.2f below NoPos loss %.4f.',
    [BoolToStr(PassLoss, 'PASS', 'FAIL'), WithPos.Loss, cLossMargin, NoPos.Loss]));
  PassAcc := WithPos.Acc >= cAccBar;
  WriteLn(Format('[%s] CumSum test acc = %.2f%% (must be >= %.0f%%): solves the task.',
    [BoolToStr(PassAcc, 'PASS', 'FAIL'), WithPos.Acc * 100, cAccBar * 100]));
  PassChance := NoPos.Acc <= cBakeChance + cChanceTol;
  WriteLn(Format('[%s] NoPos test acc = %.2f%% (must be <= chance+%.0f%% = %.2f%%): '
    + 'positions invisible.',
    [BoolToStr(PassChance, 'PASS', 'FAIL'), NoPos.Acc * 100, cChanceTol * 100,
     (cBakeChance + cChanceTol) * 100]));
  WriteLn;
  WriteLn('TAKEAWAY: with positions invisible the permutation-invariant model is');
  WriteLn('stuck at chance; concatenating the parameter-free CumSum position ramp');
  WriteLn('lets the SAME model read off the marker''s slot and solve the task.');
  WriteLn;

  PassAll := PassLoss and PassAcc and PassChance;
  if PassAll then
    WriteLn('=> BAKE-OFF CHECKS PASS: the CumSum position feature measurably helps.')
  else
    WriteLn('=> BAKE-OFF CHECKS FAILED (see above).');

  if not PassAll then Halt(1);
end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'CumSum Position Encoding Demo';
  WriteLn('CumSumPositionEncoding: forward-only demo + train-time bake-off.');
  WriteLn;
  WriteLn('################################################################');
  WriteLn('PART 1: forward-only CumSum demonstration.');
  WriteLn('################################################################');
  WriteLn;
  DemoLinearRamp();
  DemoArbitraryPrefixSum();
  DemoPerRowIndependence();
  DemoUsageHint();
  WriteLn;
  RunBakeOff();
end.
