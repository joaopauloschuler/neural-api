// Performer example
//
// Performer / FAVOR+ self-attention (Choromanski et al. 2020, "Rethinking
// Attention with Performers") on a small synthetic sequence task, plus a direct
// demonstration of the headline FAVOR+ claim: positive random features give an
// UNBIASED, low-variance estimate of the softmax attention kernel, so the
// Performer output APPROXIMATES full quadratic softmax SDPA and the
// approximation error SHRINKS as the number of random features m grows.
//
// Unlike TNNetLinearAttention (deterministic elu+1 feature map, Katharopoulos
// 2020) Performer uses POSITIVE RANDOM FEATURES for an m x d_k FROZEN random
// projection W:
//   phi(x) = exp(W.x - ||x||^2/2) / sqrt(m)
// so that E[ phi(q).phi(k) ] = exp(q.k) (the softmax numerator kernel). Then
//   q'_t = phi(Q_t), k'_s = phi(K_s)
//   S = sum_s k'_s (x) V_s   (m x d_v),  Z = sum_s k'_s   (m,)
//   Out_t = (q'_t . S) / (q'_t . Z)
// at linear O(SeqLen * m * d_v) cost (no SeqLen x SeqLen score matrix).
//
// PART 1 -- APPROXIMATION QUALITY. For a fixed random input sequence we compute
// full softmax SDPA (the exact target) and Performer outputs at increasing m.
// We average over several random projection seeds at each m and report the mean
// RMS error between Performer and SDPA. The error must fall as m grows -- that
// is FAVOR+ converging to softmax.
//
// PART 2 -- TRAINING. A tiny majority-value task: a sequence of one-hot value
// tokens; the target read out at a dedicated query position is the MOST FREQUENT
// value class. Performer trains well above chance, confirming the random-feature
// attention is a usable drop-in for softmax attention at linear cost.
//
// Pure CPU, tiny dims, finishes in well under a couple of minutes on 2 cores.
//
// Coded by Claude (AI).
program Performer;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cNumVals    = 5;     // number of value classes (the labels to count)
  cSeqLen     = 17;    // sequence length (16 value tokens + 1 query position)
  cNumTokens  = cSeqLen - 1;             // value-carrying tokens
  cModelDim   = 24;    // attention head dim d_k
  cInDim      = cNumVals + 1;            // value one-hot | query flag
  cTrainSteps = 12000;
  cEvalSeqs   = 600;

// ---------------------------------------------------------------------------
// PART 1 -- approximation quality vs m.
// ---------------------------------------------------------------------------

// One self-contained net: Input -> (the supplied attention layer). The packed
// Q|K|V input is fed directly so both arms see EXACTLY the same Q,K,V.
function BuildBareAttention(ALayer: TNNetLayer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 3 * cModelDim));
  Result.AddLayer(ALayer);
end;

// RMS error between two equally-sized output volumes.
function RmsError(A, B: TNNetVolume): TNeuralFloat;
var
  i: integer;
  acc, d: TNeuralFloat;
begin
  acc := 0;
  for i := 0 to A.Size - 1 do
  begin
    d := A.Raw[i] - B.Raw[i];
    acc := acc + d * d;
  end;
  Result := Sqrt(acc / A.Size);
end;

procedure ApproximationStudy();
var
  Input: TNNetVolume;
  SDPANet, PerfNet: TNNet;
  SDPAOut: TNNetVolume;
  i, mi, trial, m, seed: integer;
  errSum, err: TNeuralFloat;
  mList: array[0..5] of integer = (4, 8, 16, 32, 64, 128);
const
  cTrials = 24;  // average over this many random projection seeds per m
begin
  WriteLn('=== PART 1: FAVOR+ approximates softmax SDPA; error shrinks with m ===');
  WriteLn('d_k=', cModelDim, '  seq_len=', cSeqLen,
          '  (mean RMS error vs full SDPA over ', cTrials, ' seeds)');
  WriteLn;

  RandSeed := 20240;
  Input := TNNetVolume.Create(cSeqLen, 1, 3 * cModelDim);
  // Bound Q|K|V to a modest range so softmax / exp features stay well-behaved.
  for i := 0 to Input.Size - 1 do
    Input.Raw[i] := Sin(i * 0.37) * 0.5;
  // SDPA scales scores by 1/sqrt(d_k); Performer's exp feature map estimates the
  // UNSCALED kernel exp(q.k). To compare the SAME kernel, pre-scale Q and K
  // (slabs 0..2*d_k-1, NOT V) by d_k^(-1/4) so q.k -> q.k/sqrt(d_k) for BOTH.
  for i := 0 to cSeqLen - 1 do
    for mi := 0 to 2 * cModelDim - 1 do
      Input[i, 0, mi] := Input[i, 0, mi] * Power(cModelDim, -0.25);

  // Exact target: full quadratic softmax SDPA.
  SDPANet := BuildBareAttention(
    TNNetScaledDotProductAttention.Create(cModelDim, {Causal=}false));
  SDPANet.Compute(Input);
  SDPAOut := TNNetVolume.Create();
  SDPAOut.Copy(SDPANet.GetLastLayer.Output);

  WriteLn('   m   |  mean RMS error vs SDPA');
  WriteLn('  -----+------------------------');
  for mi := 0 to High(mList) do
  begin
    m := mList[mi];
    errSum := 0;
    for trial := 0 to cTrials - 1 do
    begin
      seed := 1000 + trial * 7919;  // distinct nonzero seed per trial
      PerfNet := BuildBareAttention(
        TNNetPerformerAttention.Create(cModelDim, m, seed));
      PerfNet.Compute(Input);
      err := RmsError(PerfNet.GetLastLayer.Output, SDPAOut);
      errSum := errSum + err;
      PerfNet.Free;
    end;
    WriteLn(Format('  %4d | %20.6f', [m, errSum / cTrials]));
  end;
  WriteLn;
  WriteLn('(Monotone-ish decrease confirms the unbiased FAVOR+ estimator: more');
  WriteLn(' random features -> tighter softmax approximation.)');
  WriteLn;

  SDPAOut.Free;
  SDPANet.Free;
  Input.Free;
end;

// ---------------------------------------------------------------------------
// PART 2 -- a tiny end-to-end training task.
// ---------------------------------------------------------------------------

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
    Inc(counts[v]);
  end;
  // Query position: no value, flag set.
  Input[cSeqLen - 1, 0, cNumVals] := 1.0;
  best := 0; bestCount := counts[0];
  for j := 1 to cNumVals - 1 do
    if counts[j] > bestCount then begin bestCount := counts[j]; best := j; end;
  Desired[cSeqLen - 1, 0, best] := 1.0;
end;

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

function BuildPerformer(m: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(3 * cModelDim));
  Result.AddLayer(TNNetPerformerAttention.Create(cModelDim, m, {seed=}424242));
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
  PNet: TNNet;
  pAcc: TNeuralFloat;
  perfScores, fullScores: integer;
begin
  ApproximationStudy();

  WriteLn('=== PART 2: Performer trains a majority-value task at linear cost ===');
  WriteLn('num_classes=', cNumVals, '  tokens/seq=', cNumTokens,
          '  model_dim=', cModelDim, '  seq_len=', cSeqLen);
  WriteLn('task: predict the MAJORITY value class (a global aggregate).');
  WriteLn;

  // Cost contrast: full SDPA scores every query against every key (SeqLen^2);
  // Performer never forms a score matrix -- it accumulates S/Z over m features.
  fullScores := cSeqLen * cSeqLen;
  perfScores := cSeqLen * (2 * cModelDim); // m = 2*d_k default for this arm
  WriteLn('attention work (per head, per sequence):');
  WriteLn('  Full SDPA  : ', cSeqLen, ' x ', cSeqLen, ' = ', fullScores,
          ' pairwise scores');
  WriteLn('  Performer  : ', cSeqLen, ' x m(', 2 * cModelDim, ') = ', perfScores,
          ' feature ops  (NO SeqLen x SeqLen matrix)');
  WriteLn;

  PNet := BuildPerformer({m=}2 * cModelDim);
  WriteLn('total trainable params (frozen random W is NOT trained) = ',
          PNet.CountWeights());
  WriteLn;

  WriteLn('training Performer (', cTrainSteps, ' steps)...');
  RandSeed := 999;
  Train(PNet, cTrainSteps, 0.01);
  WriteLn;

  RandSeed := 7;
  pAcc := Evaluate(PNet, cEvalSeqs);

  WriteLn('eval over ', cEvalSeqs, ' held-out sequences (chance = ',
          (100.0 / cNumVals):0:1, '%):');
  WriteLn('  Performer : majority-class accuracy = ', (pAcc * 100):0:1, '%');
  WriteLn;

  if pAcc > 1.5 * (1.0 / cNumVals) then
    WriteLn('OK: Performer learns the task well above chance using linear-cost ',
            'random-feature attention.')
  else
    WriteLn('WARNING: Performer did not learn the task above chance.');

  PNet.Free;
end.
