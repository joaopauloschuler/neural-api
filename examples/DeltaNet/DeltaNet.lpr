// DeltaNet example
//
// Associative recall (key -> value lookup) over a sequence, contrasting the
// delta-rule linear-attention layer TNNetDeltaNet against a param-matched
// fixed-decay linear-attention baseline (TNNetRetention).
//
// The task (OVERWRITE recall -- the regime where the delta rule shines). A
// sequence presents WRITE tokens, and one of the keys is written TWICE with two
// different values (an overwrite); then a QUERY re-presents that key:
//   write t :  [ key_onehot(k_t) | value_vec | flag=0 ]
//   query   :  [ key_onehot(k_q) | 0...0     | flag=1 ]
// The target is the key's MOST RECENT value. Retrieving it requires REMOVING the
// stale association and keeping the fresh one -- exactly the delta rule's
// read-then-correct write S_t = S_{t-1} + beta_t k_t (x) (v_t - S_{t-1}^T k_t).
// A fixed-decay accumulator (Retention) instead BLENDS the old and new values by
// recency and cannot cleanly overwrite, so it lands between the two values.
//
// Two arms share the SAME I/O contract and a matched parameter budget:
//   * DeltaNet arm : front 1x1 projection -> TNNetDeltaNet -> 1x1 readout.
//       DeltaNet maintains a (d,d) matrix memory updated by the delta rule
//       S_t = S_{t-1} + beta_t k_t (x) (v_t - S_{t-1}^T k_t): it reads the
//       current prediction, then writes back ONLY the correction. It can store a
//       fresh association and read it back later -- exactly what recall needs.
//   * Retention arm : front 1x1 projection (to 3*d_k = Q|K|V) -> TNNetRetention
//       (learnable decay) -> 1x1 readout. Plain decayed linear attention with NO
//       error-corrected write-back; it blends associations by recency rather than
//       retrieving an exact stored value.
//
// Headline: trained on the SAME data for the SAME number of steps, the DeltaNet
// arm reaches markedly lower recall error (and higher exact-recall accuracy) than
// the fixed-decay Retention arm. Pure CPU, tiny dims, finishes in a couple of
// minutes on 2 cores.
//
// Coded by Claude (AI).
program DeltaNet;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cNumKeys   = 6;     // vocabulary of distinct keys
  cNumVals   = 6;     // vocabulary of distinct value vectors
  cValueDim  = 4;     // dimensionality of each stored value vector
  cNumPairs  = 4;     // initial distinct (key,value) writes per sequence
  cSeqLen    = cNumPairs + 2;            // writes + 1 overwrite + 1 query
  cModelDim  = 16;    // memory width (DeltaNet head dim / Retention d_k)
  cInDim     = cNumKeys + cValueDim + 1; // key one-hot | value | query flag
  cTrainSteps= 6000;
  cEvalSeqs  = 400;

var
  ValueBank: array[0..cNumVals - 1, 0..cValueDim - 1] of TNeuralFloat;

// Fixed value-vector vocabulary (the discrete "contents" the network recalls).
procedure InitValueBank();
var k, j: integer;
begin
  for k := 0 to cNumVals - 1 do
    for j := 0 to cValueDim - 1 do
      ValueBank[k, j] := Sin(k * 1.7 + j * 0.9) * 0.6 + Cos(k * 0.5 - j * 1.3) * 0.4;
end;

// Build one OVERWRITE-recall sequence. cNumPairs distinct keys are written with
// random value-ids; then key[overIdx] is RE-WRITTEN with a different value-id
// (the overwrite); finally that key is queried and the target is its LATEST
// value. Returns (in Desired) only the query-position target.
procedure MakeSample(Input, Desired: TNNetVolume);
var
  pos, j, overIdx, queriedKey, newVal, tmp: integer;
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
  // Initial write tokens (positions 0..cNumPairs-1).
  for pos := 0 to cNumPairs - 1 do
  begin
    vals[pos] := Random(cNumVals);
    Input[pos, 0, keys[pos]] := 1.0;                       // key one-hot
    for j := 0 to cValueDim - 1 do
      Input[pos, 0, cNumKeys + j] := ValueBank[vals[pos], j]; // value vector
    Input[pos, 0, cNumKeys + cValueDim] := 0.0;            // flag = write
  end;
  // Overwrite token (position cNumPairs): pick one written key, give it a NEW
  // value-id distinct from its first one.
  overIdx := Random(cNumPairs);
  queriedKey := keys[overIdx];
  repeat newVal := Random(cNumVals); until newVal <> vals[overIdx];
  Input[cNumPairs, 0, queriedKey] := 1.0;
  for j := 0 to cValueDim - 1 do
    Input[cNumPairs, 0, cNumKeys + j] := ValueBank[newVal, j];
  Input[cNumPairs, 0, cNumKeys + cValueDim] := 0.0;        // flag = write
  // Query token (position cSeqLen-1): re-present the overwritten key; target is
  // its MOST RECENT value (newVal), NOT the stale one.
  Input[cSeqLen - 1, 0, queriedKey] := 1.0;
  Input[cSeqLen - 1, 0, cNumKeys + cValueDim] := 1.0;      // flag = query
  for j := 0 to cValueDim - 1 do
    Desired[cSeqLen - 1, 0, j] := ValueBank[newVal, j];
end;

// Mean squared recall error at the query position, averaged over N sequences.
// Also returns the EXACT-recall accuracy: fraction of sequences whose query
// output is closest (nearest neighbour over the value bank) to the true value.
function Evaluate(NN: TNNet; N: integer; out Accuracy: TNeuralFloat): TNeuralFloat;
var
  Input, Desired: TNNetVolume;
  seq, ji, ki, bestK, trueKey, correct: integer;
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
      // Squared error at the query position.
      for ji := 0 to cValueDim - 1 do
      begin
        diff := NN.GetLastLayer.Output[cSeqLen - 1, 0, ji] - Desired[cSeqLen - 1, 0, ji];
        mse := mse + diff * diff;
      end;
      // Nearest-neighbour decode over the value bank -> exact-recall check.
      // Recover the true value-id from the Desired value (its bank entry).
      trueKey := -1;
      for ki := 0 to cNumVals - 1 do
      begin
        dist := 0;
        for ji := 0 to cValueDim - 1 do
        begin
          diff := Desired[cSeqLen - 1, 0, ji] - ValueBank[ki, ji];
          dist := dist + diff * diff;
        end;
        if dist < 1e-6 then trueKey := ki;
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
      if bestK = trueKey then Inc(correct);
    end;
    Result := mse / (N * cValueDim);
    Accuracy := correct / N;
  finally
    Input.Free; Desired.Free;
  end;
end;

// DeltaNet arm: 1x1 projection to cModelDim -> delta-rule cell -> 1x1 readout.
function BuildDeltaNet(): TNNet;
var Cell: TNNetDeltaNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  Cell := TNNetDeltaNet.Create();
  Result.AddLayer(Cell);
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
  // The layer defaults to a near-zero write strength (b_beta strongly negative)
  // for numerical tameness at init; for a from-scratch recall task warm-start the
  // gate to beta~=0.5 (b_beta=0) so the memory actually writes from step one and
  // the gate escapes its cold-start saturation.
  Cell.Neurons[4].Weights.Raw[0] := 0.0;
end;

// Retention arm: 1x1 projection to 3*cModelDim (Q|K|V) -> learnable-decay
// retention (output d_k = cModelDim) -> 1x1 readout. Param budget matched to the
// DeltaNet arm (DeltaNet uses 3 d x d projections internally; here the front 1x1
// produces 3*d channels, the same projection budget).
function BuildRetention(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(3 * cModelDim));
  Result.AddLayer(TNNetRetention.Create(cModelDim, 0.9, {LearnGamma=}true));
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
  DNet, RNet: TNNet;
  dMSE, rMSE, dAcc, rAcc: TNeuralFloat;
begin
  RandSeed := 12345;
  InitValueBank();

  WriteLn('=== DeltaNet: associative recall, delta-rule vs fixed-decay ===');
  WriteLn('keys=', cNumKeys, '  value_dim=', cValueDim,
          '  pairs/seq=', cNumPairs, '  model_dim=', cModelDim,
          '  seq_len=', cSeqLen);
  WriteLn;

  DNet := BuildDeltaNet();
  RNet := BuildRetention();
  WriteLn('DeltaNet  params = ', DNet.CountWeights());
  WriteLn('Retention params = ', RNet.CountWeights());
  WriteLn;

  WriteLn('training both arms on the SAME recall stream (', cTrainSteps, ' steps each)...');
  // Same RNG stream replay so both arms see identical training samples.
  RandSeed := 999;
  Train(DNet, cTrainSteps, 0.01);
  RandSeed := 999;
  Train(RNet, cTrainSteps, 0.01);
  WriteLn;

  RandSeed := 7;
  dMSE := Evaluate(DNet, cEvalSeqs, dAcc);
  RandSeed := 7;
  rMSE := Evaluate(RNet, cEvalSeqs, rAcc);

  WriteLn('eval over ', cEvalSeqs, ' held-out recall sequences:');
  WriteLn('  DeltaNet (delta rule)  : recall MSE = ', dMSE:0:5,
          '   exact-recall acc = ', (dAcc * 100):0:1, '%');
  WriteLn('  Retention (fixed decay): recall MSE = ', rMSE:0:5,
          '   exact-recall acc = ', (rAcc * 100):0:1, '%');
  WriteLn;

  if (dMSE < rMSE) and (dAcc >= rAcc) then
    WriteLn('OK: the delta-rule layer recalls stored values better than the ',
            'fixed-decay baseline.')
  else
    WriteLn('WARNING: DeltaNet did not beat the Retention baseline on recall.');

  DNet.Free; RNet.Free;
end.
