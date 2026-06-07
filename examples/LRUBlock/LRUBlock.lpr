// Linear Recurrent Unit (LRU) block example
//
// Long-range DELAYED-RECALL over a sequence, showcasing the full LRU block
// builder TNNet.AddLRU (Orvieto et al. 2023, "Resurrecting RNNs for Long
// Sequences", arXiv:2303.06349). A stack of AddLRU blocks drops into a
// transformer-style residual tower exactly like AddGatedLinearAttentionBlock /
// AddRetention: each block is
//   x := x + LRU(LayerNorm(x))     (LRU time-mixing residual)
//   x := x + FFN(LayerNorm(x))     (token-wise SwiGLU FFN residual)
// where the LRU arm is a per-token input projection -> the stable
// complex-diagonal LRU recurrence (TNNetLRU) -> a GLU non-linearity -> a
// per-token output projection.
//
// The task (DELAYED RECALL). A one-hot symbol is presented at position 0 of a
// long sequence; every later position is a "distractor" carrying noise but no
// symbol. At the FINAL position the network must output the symbol seen at
// position 0. Solving it requires propagating information across the whole
// sequence with negligible decay - precisely what the LRU's stable
// complex-diagonal recurrence (eigenvalues parameterised to sit just inside the
// unit circle) is built for.
//
//   pos 0      : [ symbol_onehot(s) | flag=1 ]
//   pos 1..L-2 : [ noise...         | flag=0 ]  (distractors)
//   pos L-1    : [ 0...0            | flag=0 ]  target = symbol_onehot(s)
//
// Headline: training a small 2-block AddLRU tower on this stream drives the
// recall cross-entropy down and the recall accuracy up over the SeqLen-long
// delay, printing a decreasing training-loss curve and a held-out accuracy that
// climbs well above the 1/cNumSymbols chance level. Pure CPU, tiny dims,
// finishes well under a minute on 2 cores.
//
// Coded by Claude (AI).
program LRUBlock;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cNumSymbols = 6;                 // recall vocabulary
  cSeqLen     = 24;                // long delay between cue and recall
  cInDim      = cNumSymbols + 1;   // symbol one-hot | cue flag
  cModelDim   = 24;                // residual-stream width (d_model)
  cFFDim      = 32;                // SwiGLU FFN inner width (d_ff)
  cNumBlocks  = 2;                 // depth of the AddLRU tower
  cTrainSteps = 8000;
  cEvalSeqs   = 500;

// Build one delayed-recall sequence. A random symbol is shown (one-hot, cue
// flag=1) at position 0; positions 1..L-2 are noise distractors; the target at
// position L-1 is the one-hot of the position-0 symbol.
procedure MakeSample(Input, Desired: TNNetVolume; out Symbol: integer);
var pos, j: integer;
begin
  Input.Fill(0);
  Desired.Fill(0);
  Symbol := Random(cNumSymbols);
  // Cue at position 0.
  Input[0, 0, Symbol] := 1.0;
  Input[0, 0, cNumSymbols] := 1.0;          // cue flag
  // Distractor noise (no symbol, no flag) at positions 1..L-2.
  for pos := 1 to cSeqLen - 2 do
    for j := 0 to cNumSymbols - 1 do
      Input[pos, 0, j] := (Random - 0.5) * 0.2;
  // Recall target at the final position.
  Desired[cSeqLen - 1, 0, Symbol] := 1.0;
end;

// LRU tower: per-token embedding into the residual stream, cNumBlocks stacked
// AddLRU blocks, then a per-token softmax classifier head at every position
// (we only read the final position for the recall decision).
function BuildLRU(): TNNet;
var b: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  // Embed each token into the residual stream (pointwise keeps the time axis).
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  for b := 0 to cNumBlocks - 1 do
    Result.AddLRU(cFFDim);
  // Per-token classifier head -> softmax over the symbol vocabulary.
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cNumSymbols));
  Result.AddLayer(TNNetSoftMax.Create());
end;

// Held-out recall accuracy: fraction of sequences whose final-position argmax
// matches the cued symbol.
function Evaluate(NN: TNNet; N: integer): TNeuralFloat;
var
  Input, Desired: TNNetVolume;
  seq, j, sym, bestJ, correct: integer;
  bestV: TNeuralFloat;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, cInDim);
  Desired := TNNetVolume.Create(cSeqLen, 1, cNumSymbols);
  correct := 0;
  try
    for seq := 0 to N - 1 do
    begin
      MakeSample(Input, Desired, sym);
      NN.Compute(Input);
      bestJ := 0; bestV := NN.GetLastLayer.Output[cSeqLen - 1, 0, 0];
      for j := 1 to cNumSymbols - 1 do
        if NN.GetLastLayer.Output[cSeqLen - 1, 0, j] > bestV then
        begin
          bestV := NN.GetLastLayer.Output[cSeqLen - 1, 0, j];
          bestJ := j;
        end;
      if bestJ = sym then Inc(correct);
    end;
    Result := correct / N;
  finally
    Input.Free; Desired.Free;
  end;
end;

// Cross-entropy of the final-position softmax against the recall target,
// averaged over N sequences (for the training-loss curve).
function FinalCE(NN: TNNet; N: integer): TNeuralFloat;
var
  Input, Desired: TNNetVolume;
  seq, sym: integer;
  p, ce: TNeuralFloat;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, cInDim);
  Desired := TNNetVolume.Create(cSeqLen, 1, cNumSymbols);
  ce := 0;
  try
    for seq := 0 to N - 1 do
    begin
      MakeSample(Input, Desired, sym);
      NN.Compute(Input);
      p := NN.GetLastLayer.Output[cSeqLen - 1, 0, sym];
      ce := ce - Ln(Max(p, 1e-9));
    end;
    Result := ce / N;
  finally
    Input.Free; Desired.Free;
  end;
end;

var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  i, sym, epoch: integer;
  ce, acc: TNeuralFloat;
begin
  RandSeed := 12345;

  WriteLn('=== LRU block (TNNet.AddLRU): long-range delayed recall ===');
  WriteLn('symbols=', cNumSymbols, '  seq_len=', cSeqLen,
          '  d_model=', cModelDim, '  d_ff=', cFFDim,
          '  blocks=', cNumBlocks);

  NN := BuildLRU();
  WriteLn('LRU tower params = ', NN.CountWeights(),
          '   (chance recall acc = ', (100.0 / cNumSymbols):0:1, '%)');
  WriteLn;

  Input := TNNetVolume.Create(cSeqLen, 1, cInDim);
  Desired := TNNetVolume.Create(cSeqLen, 1, cNumSymbols);
  NN.SetLearningRate(0.004, 0.9);

  WriteLn('training (', cTrainSteps, ' steps), loss curve:');
  RandSeed := 999;
  for epoch := 0 to 7 do
  begin
    // 1/8th of the training budget per reported point.
    for i := 0 to (cTrainSteps div 8) - 1 do
    begin
      MakeSample(Input, Desired, sym);
      NN.Compute(Input);
      NN.Backpropagate(Desired);
    end;
    RandSeed := 7 + epoch;
    ce := FinalCE(NN, 200);
    acc := Evaluate(NN, 200);
    WriteLn('  step ', ((epoch + 1) * (cTrainSteps div 8)):5,
            '   recall CE = ', ce:0:4,
            '   recall acc = ', (acc * 100):0:1, '%');
  end;
  WriteLn;

  RandSeed := 4242;
  acc := Evaluate(NN, cEvalSeqs);
  WriteLn('held-out recall accuracy over ', cEvalSeqs, ' sequences = ',
          (acc * 100):0:1, '%');
  if acc > (2.0 / cNumSymbols) then
    WriteLn('OK: the AddLRU tower recalls the cued symbol across the ',
            cSeqLen, '-step delay far above chance.')
  else
    WriteLn('WARNING: the LRU tower did not learn the long-range recall.');

  Input.Free; Desired.Free; NN.Free;
end.
