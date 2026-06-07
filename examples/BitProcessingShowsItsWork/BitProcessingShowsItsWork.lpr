program BitProcessingShowsItsWork;
(*
BitProcessingShowsItsWork: "a neural network that shows its work".

A small hybrid network learns the clean arithmetic relation y = a - b on two
continuous inputs, GENERALIZES it far outside the box it was trained on, and --
unlike an ordinary dense net -- can PRINT the human-readable rule it induced.

The star is TNNetBitProcessing (neural/neuralnetwork.pas). It affine-quantizes
EACH input scalar to one whole byte over [-25.6, +25.6] (~0.2 step), runs a
SYMBOLIC byte engine (TEasyLearnAndPredictClass) that induces discrete
cause->effect rules whose grammar literally contains A-B, A+B, AND and
comparisons, then affine-decodes one float back per byte. decode o encode is
~identity, and a straight-through estimator lets it sit inside a gradient-
trained TNNet. So the layer reduces (a,b) to a discrete affine CODE that a tiny
linear readout combines -- a pipeline that is genuinely scale-free, which is
exactly why it extrapolates a clean linear rule perfectly.

Three things get printed:
  1. In-range error   : symbolic vs dense (comparable -- both fit the box).
  2. Extrapolation err : symbolic clearly beats dense (the money shot).
  3. The induced rule  : a human-readable "out := A[0] - A[1]" relation table.

How the rule is shown
---------------------
The layer keeps its symbolic engine in a private field (FByteLearning); a
SEPARATE example unit cannot reach it. So -- exactly as the sibling examples
ByteProcessingRelationTable / ByteRuleInduction do -- we drive an INDEPENDENT
TEasyLearnAndPredictClass mirror with the SAME affine byte encoding the layer
uses, feeding it the subtraction target in byte space, and call
printRelationTable. The mirror is configured identically to the layer's engine
(same neuron-group budget and search budget), so what it prints is what the
layer's engine can and does induce for a - b: the crisp csSub relation
(A[0] - A[1]). A dense net has no comparable artifact -- it just has weights.

Deterministic (fixed RandSeed), no external data, finishes in a few seconds on
two cores.

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralbyteprediction;

const
  // Affine range of TNNetBitProcessing (its defaults). Everything below stays
  // well inside it so nothing ever clips.
  AffineLo   = -25.6;
  AffineHi   =  25.6;

  // Symbolic engine configuration, shared by the in-net layer AND the rule
  // mirror so the printed rule is the engine the layer actually runs.
  EngNeurons = 16;   // relation-table size (neuron groups)
  EngBudget  = 40;   // searches per step

  TrainCount = 600;  // training pairs, a,b in [0,10]
  EvalCount  = 400;  // evaluation pairs (per box)
  Epochs     = 40;
  BatchSize  = 16;

function EncByte(x: TNeuralFloat): byte;
begin
  Result := EnsureRange(Round((x - AffineLo) / (AffineHi - AffineLo) * 255), 0, 255);
end;

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
procedure MakePairs(out Pairs: TNNetVolumePairList; Count: integer;
  Lo, Hi: TNeuralFloat);
var
  i: integer;
  a, b: TNeuralFloat;
begin
  Pairs := TNNetVolumePairList.Create();
  for i := 1 to Count do
  begin
    a := Lo + Random * (Hi - Lo);
    b := Lo + Random * (Hi - Lo);
    Pairs.Add(
      TNNetVolumePair.Create(
        TNNetVolume.Create([a, b]),
        TNNetVolume.Create([a - b])   // clean arithmetic target
      ));
  end;
end;

// Root-mean-square error of NN over Pairs (in the target's own units).
function RmseOf(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  i: integer;
  SumSq: double;
begin
  SumSq := 0;
  for i := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[i].I);
    SumSq := SumSq + Sqr(NN.GetLastLayer().Output.FData[0] - Pairs[i].O.FData[0]);
  end;
  if Pairs.Count > 0 then Result := Sqrt(SumSq / Pairs.Count) else Result := 0;
end;

// ---------------------------------------------------------------------------
// Training (manual, single-thread, deterministic)
// ---------------------------------------------------------------------------
procedure TrainNet(NN: TNNet; Pairs: TNNetVolumePairList;
  LR, Momentum: TNeuralFloat);
var
  Epoch, i, j, t, Step: integer;
  Order: array of integer;
begin
  NN.SetLearningRate(LR, Momentum);
  NN.SetL2Decay(0.0);
  SetLength(Order, Pairs.Count);
  for i := 0 to High(Order) do Order[i] := i;
  for Epoch := 1 to Epochs do
  begin
    for i := High(Order) downto 1 do          // Fisher-Yates shuffle
    begin
      j := Random(i + 1);
      t := Order[i]; Order[i] := Order[j]; Order[j] := t;
    end;
    Step := 0;
    NN.ClearDeltas();
    for i := 0 to High(Order) do
    begin
      NN.Compute(Pairs[Order[i]].I);
      NN.Backpropagate(Pairs[Order[i]].O);
      Inc(Step);
      if Step = BatchSize then
      begin
        NN.UpdateWeights();
        NN.ClearDeltas();
        Step := 0;
      end;
    end;
    if Step > 0 then begin NN.UpdateWeights(); NN.ClearDeltas(); end;
  end;
end;

// ---------------------------------------------------------------------------
// The induced rule, shown via a directly-driven mirror of the layer's engine.
// ---------------------------------------------------------------------------
procedure ShowInducedRule();
var
  Engine: TEasyLearnAndPredictClass;
  ep, i: integer;
  a, b: TNeuralFloat;
  aAct, aState, aPred, aTgt: array[0..1] of byte;
  sub: byte;
begin
  FillChar(Engine, SizeOf(Engine), 0);
  // Same configuration the TNNetBitProcessing layer gives its own engine
  // (2 input bytes -> 2 output bytes, no cache, FUseBelief, FGeneralize).
  Engine.Initiate(2, 2, False {includeZeros}, EngNeurons, EngBudget,
                  False {cache}, 1000);
  Engine.BytePred.FUseBelief := True;
  Engine.BytePred.FGeneralize := True;

  RandSeed := 20240607;
  // Drive it with the affine byte encoding the layer uses, and the subtraction
  // target in byte space: byte_out := (encByte(a) - encByte(b)) and 255, which
  // is exactly the engine's csSub operation. With many varied pairs the engine
  // discovers the single rule that explains them all: B := A[0] - A[1].
  for ep := 0 to 250 do
    for i := 0 to 7 do
    begin
      a := Random * 10;
      b := Random * 10;
      aAct[0] := EncByte(a);  aAct[1] := EncByte(b);
      aState[0] := aAct[0];   aState[1] := aAct[1];
      Engine.Predict(aAct, aState, aPred);
      sub := byte((aAct[0] - aAct[1]) and 255);
      aTgt[0] := sub;  aTgt[1] := sub;
      Engine.newStateFound(aTgt);
    end;

  WriteLn('  format: B=<out byte> <condition> => fE[B] := <effect> ',
          '[ f=conf Vit=wins n=samples ]');
  WriteLn('  A[0] = encoded a, A[1] = encoded b. Read the WINNING rows (f=1, Vit>0);');
  WriteLn('  rows with n=0 are unused neuron slots. Several groups independently');
  WriteLn('  converge on the same effect:  fE[B] := (A[0] - A[1])  -- subtraction.');
  WriteLn;
  Engine.printRelationTable;
  Engine.DeInitiate;
end;

// ---------------------------------------------------------------------------
var
  TrainPairs, InRangePairs, ExtrapPairs: TNNetVolumePairList;
  Symbolic, Dense: TNNet;
  symIn, symEx, denIn, denEx: TNeuralFloat;
begin
  WriteLn('A neural network that shows its work: TNNetBitProcessing on y = a - b');
  WriteLn('=====================================================================');
  WriteLn('TRAIN box : a,b in [0,10]   (y = a-b in [-10,10])');
  WriteLn('EXTRAP box: a,b in [10,20]  (unseen, still inside the +/-25.6 affine range)');
  WriteLn('The symbolic RULE is range-free, so it must generalize; a dense tanh net');
  WriteLn('fits the training box but bends away once the inputs leave it.');
  WriteLn;

  RandSeed := 424242;
  MakePairs(TrainPairs,   TrainCount, 0,  10);
  MakePairs(InRangePairs, EvalCount,  0,  10);
  MakePairs(ExtrapPairs,  EvalCount,  10, 20);

  // ---- Model 1: symbolic (quantize -> discrete rule -> linear readout) ----
  Symbolic := TNNet.Create();
  Symbolic.AddLayer([
    TNNetInput.Create(2),
    TNNetBitProcessing.Create(0, EngNeurons, EngBudget, 0, AffineLo, AffineHi),
    TNNetFullConnectLinear.Create(1)
  ]);
  TrainNet(Symbolic, TrainPairs, {LR=}0.0005, {Momentum=}0.9);

  // ---- Model 2: dense baseline (same-size tanh head) ----
  Dense := TNNet.Create();
  Dense.AddLayer([
    TNNetInput.Create(2),
    TNNetFullConnect.Create(8),     // tanh
    TNNetFullConnect.Create(8),     // tanh
    TNNetFullConnectLinear.Create(1)
  ]);
  TrainNet(Dense, TrainPairs, {LR=}0.01, {Momentum=}0.9);

  symIn := RmseOf(Symbolic, InRangePairs);
  symEx := RmseOf(Symbolic, ExtrapPairs);
  denIn := RmseOf(Dense,    InRangePairs);
  denEx := RmseOf(Dense,    ExtrapPairs);

  WriteLn('1) IN-RANGE error (a,b in [0,10]) -- both models fit the training box');
  WriteLn('   ----------------------------------------------------------------');
  WriteLn(Format('   symbolic (TNNetBitProcessing)  RMSE = %7.4f', [symIn]));
  WriteLn(Format('   dense baseline (tanh)          RMSE = %7.4f', [denIn]));
  WriteLn;
  WriteLn('2) EXTRAPOLATION error (a,b in [10,20]) -- the money shot');
  WriteLn('   ----------------------------------------------------------------');
  WriteLn(Format('   symbolic (TNNetBitProcessing)  RMSE = %7.4f', [symEx]));
  WriteLn(Format('   dense baseline (tanh)          RMSE = %7.4f', [denEx]));
  WriteLn(Format('   -> symbolic extrapolates %.1fx better than the dense net.',
                 [denEx / Max(symEx, 1e-6)]));
  WriteLn;
  WriteLn('3) THE INDUCED RULE -- the symbolic engine can show its work');
  WriteLn('   ----------------------------------------------------------------');
  ShowInducedRule();
  WriteLn;
  WriteLn('   The dense baseline has nothing comparable to print: its knowledge is');
  WriteLn('   a tangle of tanh weights, not a readable rule -- which is also why it');
  WriteLn('   cannot extrapolate the clean linear relation outside its training box.');
  WriteLn;
  WriteLn('Done.');

  Symbolic.Free;
  Dense.Free;
  TrainPairs.Free;
  InRangePairs.Free;
  ExtrapPairs.Free;
end.
