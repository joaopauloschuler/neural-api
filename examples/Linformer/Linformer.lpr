// Linformer example
//
// Linformer self-attention (Wang et al. 2020, "Linformer: Self-Attention with
// Linear Complexity") on a small synthetic sequence task, contrasting
// TNNetLinformerAttention against full quadratic TNNetScaledDotProductAttention
// at the SAME sequence length.
//
// Linformer projects the Key and Value sequences DOWN along the sequence axis
// from SeqLen to a small fixed rank k (k << SeqLen) with two learnable matrices
// E, F (each k x SeqLen):
//   K' = E.K, V' = F.V  ->  Attn = softmax(Q.K'^T / sqrt(d_k))  ->  Out = Attn.V'
// The score matrix is SeqLen x k instead of SeqLen x SeqLen, so attention costs
// O(SeqLen * k) instead of O(SeqLen^2).
//
// The task -- MAJORITY VALUE (a GLOBAL aggregate, which is exactly the regime
// where attention is approximately low-rank, so Linformer's k-rank projection
// loses little). A sequence carries cSeqLen tokens, each a one-hot value label
// drawn from cNumVals classes; the target (read out at a dedicated query
// position) is the MOST FREQUENT value class in the sequence. Answering needs
// the network to POOL evidence across the whole sequence -- a low-rank,
// permutation-friendly read that a handful of projected "summary" rows captures
// well.
//
// Both arms share an identical I/O contract and differ ONLY in the attention
// layer:
//   front 1x1 projection (-> 3*d_k = Q|K|V) -> attention -> 1x1 readout -> softmax.
//
// Headline. At the SAME SeqLen the Linformer arm uses FEWER attention FLOPs (its
// score matrix is SeqLen x k, not SeqLen x SeqLen) and still learns the task
// well above chance -- the classic Linformer trade: trading a little accuracy
// for LINEAR attention cost. The program prints the per-query attention
// score-matrix size (the FLOP contrast) and the held-out classification
// accuracy for each arm. Pure CPU, tiny dims, finishes in well under a couple of
// minutes on 2 cores.
//
// Coded by Claude (AI).
program Linformer;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils,
  neuralnetwork, neuralvolume;

const
  cNumVals    = 5;     // number of value classes (the labels to count)
  cSeqLen     = 17;    // sequence length (16 value tokens + 1 query position)
  cNumTokens  = cSeqLen - 1;             // value-carrying tokens
  cModelDim   = 24;    // attention head dim d_k
  cProjDim    = 4;     // Linformer projection rank k (k << SeqLen)
  cInDim      = cNumVals + 1;            // value one-hot | query flag
  cTrainSteps = 12000;
  cEvalSeqs   = 600;

// Build one majority-value sequence: cNumTokens random value labels followed by
// a query position; the target (returned only at the query) is the one-hot of
// the most frequent label.
procedure MakeSample(Input, Desired: TNNetVolume);
var
  pos, v, best, bestCount, j: integer;
  counts: array[0..cNumVals - 1] of integer;
begin
  Input.Fill(0);
  Desired.Fill(0);
  for v := 0 to cNumVals - 1 do counts[v] := 0;
  for pos := 0 to cNumTokens - 1 do
  begin
    v := Random(cNumVals);
    Input[pos, 0, v] := 1.0;                 // value one-hot
    Input[pos, 0, cNumVals] := 0.0;          // flag = data token
    Inc(counts[v]);
  end;
  // Query position: no value, flag set.
  Input[cSeqLen - 1, 0, cNumVals] := 1.0;
  // Target = argmax frequency (ties broken toward the lowest class id).
  best := 0; bestCount := counts[0];
  for j := 1 to cNumVals - 1 do
    if counts[j] > bestCount then begin bestCount := counts[j]; best := j; end;
  Desired[cSeqLen - 1, 0, best] := 1.0;
end;

// Classification accuracy at the query position over N held-out sequences.
function Evaluate(NN: TNNet; N: integer): TNeuralFloat;
var
  Input, Desired: TNNetVolume;
  seq, j, pred, truth, correct: integer;
  best, t: TNeuralFloat;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, cInDim);
  Desired := TNNetVolume.Create(cSeqLen, 1, cNumVals);
  correct := 0;
  try
    for seq := 0 to N - 1 do
    begin
      MakeSample(Input, Desired);
      NN.Compute(Input);
      pred := 0; best := NN.GetLastLayer.Output[cSeqLen - 1, 0, 0];
      truth := 0;
      for j := 0 to cNumVals - 1 do
      begin
        t := NN.GetLastLayer.Output[cSeqLen - 1, 0, j];
        if t > best then begin best := t; pred := j; end;
        if Desired[cSeqLen - 1, 0, j] > 0.5 then truth := j;
      end;
      if pred = truth then Inc(correct);
    end;
    Result := correct / N;
  finally
    Input.Free; Desired.Free;
  end;
end;

// Linformer arm: 1x1 projection to Q|K|V -> low-rank Linformer attention ->
// readout -> softmax.
function BuildLinformer(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(3 * cModelDim));
  Result.AddLayer(TNNetLinformerAttention.Create(cModelDim, cProjDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cNumVals));
  Result.AddLayer(TNNetPointwiseSoftMax.Create());
end;

// Full-attention baseline: same I/O, full quadratic SDPA instead of Linformer.
function BuildFullAttention(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(3 * cModelDim));
  Result.AddLayer(TNNetScaledDotProductAttention.Create(cModelDim, {Causal=}false));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cNumVals));
  Result.AddLayer(TNNetPointwiseSoftMax.Create());
end;

procedure Train(NN: TNNet; Steps: integer; LR: TNeuralFloat);
var
  Input, Desired: TNNetVolume;
  i: integer;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, cInDim);
  Desired := TNNetVolume.Create(cSeqLen, 1, cNumVals);
  NN.SetLearningRate(LR, 0.9);
  try
    for i := 0 to Steps - 1 do
    begin
      MakeSample(Input, Desired);
      NN.Compute(Input);
      NN.Backpropagate(Desired);
    end;
  finally
    Input.Free; Desired.Free;
  end;
end;

var
  LNet, FNet: TNNet;
  lAcc, fAcc: TNeuralFloat;
  fullScores, linScores: integer;
begin
  RandSeed := 12345;

  WriteLn('=== Linformer: low-rank attention vs full SDPA at the SAME SeqLen ===');
  WriteLn('num_classes=', cNumVals, '  tokens/seq=', cNumTokens,
          '  model_dim=', cModelDim, '  seq_len=', cSeqLen,
          '  proj_rank_k=', cProjDim);
  WriteLn('task: predict the MAJORITY value class (a global aggregate).');
  WriteLn;

  LNet := BuildLinformer();
  FNet := BuildFullAttention();

  // Per-query attention score-matrix size (the FLOP contrast). Full SDPA scores
  // every query against every key: SeqLen x SeqLen. Linformer scores every query
  // against k projected keys: SeqLen x k.
  fullScores := cSeqLen * cSeqLen;
  linScores  := cSeqLen * cProjDim;
  WriteLn('attention score-matrix size (per head, per sequence):');
  WriteLn('  Full SDPA  : ', cSeqLen, ' x ', cSeqLen, ' = ', fullScores, ' scores');
  WriteLn('  Linformer  : ', cSeqLen, ' x ', cProjDim, ' = ', linScores, ' scores  (',
          (100.0 * linScores / fullScores):0:1, '% of full)');
  WriteLn;
  WriteLn('total trainable params:');
  WriteLn('  Linformer arm    = ', LNet.CountWeights());
  WriteLn('  Full-SDPA arm    = ', FNet.CountWeights(),
          '   (SDPA itself has 0 attention params; Linformer adds E,F = ',
          2 * cProjDim * cSeqLen, ')');
  WriteLn;

  WriteLn('training both arms on the SAME stream (', cTrainSteps, ' steps each)...');
  RandSeed := 999;
  Train(LNet, cTrainSteps, 0.01);
  RandSeed := 999;
  Train(FNet, cTrainSteps, 0.01);
  WriteLn;

  RandSeed := 7;
  lAcc := Evaluate(LNet, cEvalSeqs);
  RandSeed := 7;
  fAcc := Evaluate(FNet, cEvalSeqs);

  WriteLn('eval over ', cEvalSeqs, ' held-out sequences (chance = ',
          (100.0 / cNumVals):0:1, '%):');
  WriteLn('  Linformer (rank ', cProjDim, ') : majority-class accuracy = ',
          (lAcc * 100):0:1, '%');
  WriteLn('  Full SDPA          : majority-class accuracy = ', (fAcc * 100):0:1, '%');
  WriteLn;

  // Linformer trades a little accuracy for LINEAR attention cost: the success
  // bar is that it (a) clearly learns the task -- well above chance -- and
  // (b) does so with a strictly smaller attention score matrix than full SDPA.
  if (linScores < fullScores) and (lAcc > 1.5 * (1.0 / cNumVals)) then
    WriteLn('OK: Linformer learns the task well above chance using only ',
            (100.0 * linScores / fullScores):0:1,
            '% of the attention score-matrix size of full SDPA.')
  else
    WriteLn('WARNING: Linformer did not learn the task above chance.');

  LNet.Free; FNet.Free;
end.
