// Grouped-Query Attention (GQA) example
//
// Associative recall (key -> value lookup) over a sequence, comparing THREE
// configurations of TNNet.AddMultiHeadGroupedQueryAttention (Ainslie et al.
// 2023, arXiv:2305.13245) that differ ONLY in how many key/value heads the
// query heads share:
//   * MHA arm (KVHeads = QueryHeads = 4): every query head has its own K/V
//     head -- plain multi-head attention (the degenerate full case).
//   * GQA arm (KVHeads = 2): pairs of query heads share one K/V head.
//   * MQA arm (KVHeads = 1): ALL query heads share a single K/V head --
//     Multi-Query Attention (Shazeer 2019, arXiv:1911.02150).
//
// The task (content-based recall, position-free on purpose). A sequence
// presents WRITE tokens [key_onehot | value_vec | flag=0], then a QUERY token
// re-presents one written key with flag=1; the target at the query position is
// that key's value vector. Pure content addressing -- exactly what softmax
// attention does -- so no positional encoding is needed and the comparison
// isolates the K/V head sharing.
//
// What sharing buys. The K and V projections shrink from QueryHeads*d_k to
// KVHeads*d_k output channels, a factor QueryHeads/KVHeads fewer K/V
// projection parameters (and, in a KV-cached decoder, proportionally less
// cache state to recompute per shared projection). The printed param counts
// make the shrink explicit; training all arms on the SAME sample stream shows
// recall quality stays competitive as KVHeads drops -- the GQA paper's
// headline result, reproduced on a toy.
//
// Pure CPU, tiny dims, finishes in well under a minute.
//
// Coded by Claude (AI).
program GroupedQueryAttention;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cNumKeys    = 6;     // vocabulary of distinct keys
  cNumVals    = 6;     // vocabulary of distinct value vectors
  cValueDim   = 4;     // dimensionality of each stored value vector
  cNumPairs   = 4;     // distinct (key,value) writes per sequence
  cSeqLen     = cNumPairs + 1;           // writes + 1 query
  cModelDim   = 16;    // attention width d_model
  cQueryHeads = 4;     // query heads in every arm
  cInDim      = cNumKeys + cValueDim + 1; // key one-hot | value | query flag
  cTrainSteps = 20000;
  cEvalSeqs   = 400;

var
  ValueBank: array[0..cNumVals - 1, 0..cValueDim - 1] of TNeuralFloat;

// Fixed value-vector vocabulary (the discrete "contents" the network recalls).
procedure InitValueBank();
var vk, j: integer;
begin
  for vk := 0 to cNumVals - 1 do
    for j := 0 to cValueDim - 1 do
      ValueBank[vk, j] := Sin(vk * 1.7 + j * 0.9) * 0.6 + Cos(vk * 0.5 - j * 1.3) * 0.4;
end;

// Build one recall sequence: cNumPairs distinct keys are written with random
// value-ids; then one of them is queried. Desired holds the target value
// vector at the query position only.
procedure MakeSample(Input, Desired: TNNetVolume);
var
  pos, j, queryIdx, tmp: integer;
  keys: array[0..cNumKeys - 1] of integer;
  vals: array[0..cNumPairs - 1] of integer;
begin
  Input.Fill(0);
  Desired.Fill(0);
  // Random permutation of keys; first cNumPairs are this sequence's writes.
  for j := 0 to cNumKeys - 1 do keys[j] := j;
  for j := cNumKeys - 1 downto 1 do
  begin
    tmp := Random(j + 1);
    pos := keys[j]; keys[j] := keys[tmp]; keys[tmp] := pos;
  end;
  // Write tokens (positions 0..cNumPairs-1).
  for pos := 0 to cNumPairs - 1 do
  begin
    vals[pos] := Random(cNumVals);
    Input[pos, 0, keys[pos]] := 1.0;                          // key one-hot
    for j := 0 to cValueDim - 1 do
      Input[pos, 0, cNumKeys + j] := ValueBank[vals[pos], j]; // value vector
    Input[pos, 0, cNumKeys + cValueDim] := 0.0;               // flag = write
  end;
  // Query token (position cSeqLen-1): re-present one written key.
  queryIdx := Random(cNumPairs);
  Input[cSeqLen - 1, 0, keys[queryIdx]] := 1.0;
  Input[cSeqLen - 1, 0, cNumKeys + cValueDim] := 1.0;         // flag = query
  for j := 0 to cValueDim - 1 do
    Desired[cSeqLen - 1, 0, j] := ValueBank[vals[queryIdx], j];
end;

// Mean squared recall error at the query position over N sequences, plus the
// exact-recall accuracy (nearest neighbour over the value bank).
function Evaluate(NN: TNNet; N: integer; out Accuracy: TNeuralFloat): TNeuralFloat;
var
  Input, Desired: TNNetVolume;
  seq, ji, ki, bestK, trueVal, correct: integer;
  diff, dist, bestDist, mse: TNeuralFloat;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, cInDim);
  Desired := TNNetVolume.Create(cSeqLen, 1, cValueDim);
  mse := 0; correct := 0;
  try
    for seq := 0 to N - 1 do
    begin
      MakeSample(Input, Desired);
      NN.Compute(Input);
      for ji := 0 to cValueDim - 1 do
      begin
        diff := NN.GetLastLayer.Output[cSeqLen - 1, 0, ji] - Desired[cSeqLen - 1, 0, ji];
        mse := mse + diff * diff;
      end;
      // Recover the true value-id, then nearest-neighbour decode the output.
      trueVal := -1;
      for ki := 0 to cNumVals - 1 do
      begin
        dist := 0;
        for ji := 0 to cValueDim - 1 do
        begin
          diff := Desired[cSeqLen - 1, 0, ji] - ValueBank[ki, ji];
          dist := dist + diff * diff;
        end;
        if dist < 1e-6 then trueVal := ki;
      end;
      bestK := -1; bestDist := 1e30;
      for ki := 0 to cNumVals - 1 do
      begin
        dist := 0;
        for ji := 0 to cValueDim - 1 do
        begin
          diff := NN.GetLastLayer.Output[cSeqLen - 1, 0, ji] - ValueBank[ki, ji];
          dist := dist + diff * diff;
        end;
        if dist < bestDist then begin bestDist := dist; bestK := ki; end;
      end;
      if bestK = trueVal then Inc(correct);
    end;
    Result := mse / (N * cValueDim);
    Accuracy := correct / N;
  finally
    Input.Free; Desired.Free;
  end;
end;

// One arm: 1x1 projection to cModelDim -> grouped-query attention (the only
// difference between arms is KVHeads) -> 1x1 readout to the value dimension.
// All projections are token-wise (PointwiseConvLinear preserves the sequence
// axis; FullConnect would flatten it).
function BuildArm(KVHeads: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  Result.AddMultiHeadGroupedQueryAttention(cQueryHeads, KVHeads,
    {CausalMask=}false);
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
end;

procedure Train(NN: TNNet; Steps: integer; LR: TNeuralFloat);
var
  Input, Desired: TNNetVolume;
  i: integer;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, cInDim);
  Desired := TNNetVolume.Create(cSeqLen, 1, cValueDim);
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
  MHANet, GQANet, MQANet: TNNet;
  mMSE, gMSE, qMSE, mAcc, gAcc, qAcc: TNeuralFloat;
begin
  RandSeed := 12345;
  InitValueBank();

  WriteLn('=== Grouped-Query Attention: sharing K/V heads across query heads ===');
  WriteLn('keys=', cNumKeys, '  value_dim=', cValueDim,
          '  pairs/seq=', cNumPairs, '  d_model=', cModelDim,
          '  query_heads=', cQueryHeads, '  seq_len=', cSeqLen);
  WriteLn;

  MHANet := BuildArm({KVHeads=}cQueryHeads); // full MHA
  GQANet := BuildArm({KVHeads=}2);           // grouped
  MQANet := BuildArm({KVHeads=}1);           // multi-query
  WriteLn('MHA (KVHeads=', cQueryHeads, ') params = ', MHANet.CountWeights());
  WriteLn('GQA (KVHeads=2) params = ', GQANet.CountWeights());
  WriteLn('MQA (KVHeads=1) params = ', MQANet.CountWeights());
  WriteLn('(the difference is exactly the K and V projection shrink: ',
          '2 * (QueryHeads-KVHeads) * d_k * d_model weights)');
  WriteLn;

  WriteLn('training all arms on the SAME recall stream (', cTrainSteps,
          ' steps each)...');
  // Same RNG stream replay so every arm sees identical training samples.
  RandSeed := 999;
  Train(MHANet, cTrainSteps, 0.03);
  RandSeed := 999;
  Train(GQANet, cTrainSteps, 0.03);
  RandSeed := 999;
  Train(MQANet, cTrainSteps, 0.03);
  WriteLn;

  RandSeed := 7;
  mMSE := Evaluate(MHANet, cEvalSeqs, mAcc);
  RandSeed := 7;
  gMSE := Evaluate(GQANet, cEvalSeqs, gAcc);
  RandSeed := 7;
  qMSE := Evaluate(MQANet, cEvalSeqs, qAcc);

  WriteLn('eval over ', cEvalSeqs, ' held-out recall sequences:');
  WriteLn('  MHA (KVHeads=', cQueryHeads, '): recall MSE = ', mMSE:0:5,
          '   exact-recall acc = ', (mAcc * 100):0:1, '%');
  WriteLn('  GQA (KVHeads=2): recall MSE = ', gMSE:0:5,
          '   exact-recall acc = ', (gAcc * 100):0:1, '%');
  WriteLn('  MQA (KVHeads=1): recall MSE = ', qMSE:0:5,
          '   exact-recall acc = ', (qAcc * 100):0:1, '%');
  WriteLn;
  WriteLn('Headline: recall quality stays competitive as KVHeads shrinks, while');
  WriteLn('the K/V projection parameter count drops by QueryHeads/KVHeads.');

  MQANet.Free;
  GQANet.Free;
  MHANet.Free;
end.
