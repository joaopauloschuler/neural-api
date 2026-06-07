// Gated Linear Attention (GLA) example
//
// Associative recall (key -> value lookup) over a sequence, showcasing the
// data-dependent VECTOR forget gate of TNNetGatedLinearAttention (Yang et al.
// 2023, arXiv:2312.06635) head-to-head with the delta-rule layer TNNetDeltaNet
// and a fixed-decay linear-attention baseline (TNNetRetention) on the SAME toy.
//
// The task (OVERWRITE recall). A sequence presents WRITE tokens; one key is
// written TWICE with two different values (an overwrite); then a QUERY
// re-presents that key:
//   write t :  [ key_onehot(k_t) | value_vec | flag=0 ]
//   query   :  [ key_onehot(k_q) | 0...0     | flag=1 ]
// The target is the key's MOST RECENT value. Retrieving it requires forgetting
// the stale association and keeping the fresh one.
//
// How each arm tackles it (same I/O contract, matched parameter budget):
//   * GLA arm       : 1x1 projection -> TNNetGatedLinearAttention -> 1x1 readout.
//       GLA carries a (d,d) matrix memory S updated by a DATA-DEPENDENT
//       PER-CHANNEL forget gate alpha_t = sigmoid(W_a x_t):
//         S_t[d,e] = alpha_t[d] * S_{t-1}[d,e] + k_t[d] v_t[e].
//       Each key channel decays its own slice of memory by its OWN
//       input-dependent factor, so on the overwrite token it can selectively
//       forget the channels carrying the stale value while retaining the rest.
//   * DeltaNet arm  : 1x1 projection -> TNNetDeltaNet -> 1x1 readout. The delta
//       rule S_t = S_{t-1} + beta_t k_t (x) (v_t - S_{t-1}^T k_t) reads the
//       current prediction then writes back ONLY the correction (a scalar write
//       gate, no multiplicative forget).
//   * Retention arm : 1x1 projection (to 3*d_k = Q|K|V) -> TNNetRetention
//       (a single learnable SCALAR decay). Plain decayed linear attention; it
//       blends old/new values by recency and cannot cleanly overwrite.
//
// Headline: trained on the SAME data for the SAME number of steps, the GLA and
// DeltaNet arms reach markedly lower recall error (and higher exact-recall
// accuracy) than the single-scalar-decay Retention arm; GLA's per-channel gate
// is competitive with the delta rule on overwrite recall. Pure CPU, tiny dims,
// finishes in a couple of minutes on 2 cores.
//
// Coded by Claude (AI).
program GatedLinearAttention;

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
  cModelDim  = 16;    // memory width (GLA / DeltaNet head dim / Retention d_k)
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

// GLA arm: 1x1 projection to cModelDim -> gated-linear-attention cell -> 1x1
// readout. The cell's per-channel gate defaults to alpha~=1 (near-perfect
// retention) for numerical tameness at init; warm-start it toward alpha~=0.5
// (b_a=0) so the gate actually modulates memory from step one and escapes the
// cold-start saturation.
function BuildGLA(): TNNet;
var Cell: TNNetGatedLinearAttention; d: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  Cell := TNNetGatedLinearAttention.Create();
  Result.AddLayer(Cell);
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
  for d := 0 to Cell.Neurons[4].Weights.Size - 1 do
    Cell.Neurons[4].Weights.Raw[d] := 0.0; // b_a = 0 -> alpha ~= 0.5 at init
end;

// GLA-builder arm: the SAME projection front-end, but the gated-linear-attention
// mixer is wired by the TNNet.AddGatedLinearAttention builder (token-shift +
// per-token projection into the leaf + receptance/output gating) instead of the
// bare leaf. Shows the drop-in time-mixing block reaches comparable recall.
function BuildGLABuilder(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  Result.AddGatedLinearAttention();
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
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
  // Warm-start the write gate to beta~=0.5 (b_beta=0) so the memory writes from
  // step one (the layer defaults to a near-zero write strength).
  Cell.Neurons[4].Weights.Raw[0] := 0.0;
end;

// Retention arm: 1x1 projection to 3*cModelDim (Q|K|V) -> learnable-scalar-decay
// retention -> 1x1 readout. Param budget comparable to the matrix-state arms.
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
  GNet, BNet, DNet, RNet: TNNet;
  gMSE, bMSE, dMSE, rMSE, gAcc, bAcc, dAcc, rAcc: TNeuralFloat;
begin
  RandSeed := 12345;
  InitValueBank();

  WriteLn('=== Gated Linear Attention: per-channel forget gate vs delta rule vs fixed decay ===');
  WriteLn('keys=', cNumKeys, '  value_dim=', cValueDim,
          '  pairs/seq=', cNumPairs, '  model_dim=', cModelDim,
          '  seq_len=', cSeqLen);
  WriteLn;

  GNet := BuildGLA();
  BNet := BuildGLABuilder();
  DNet := BuildDeltaNet();
  RNet := BuildRetention();
  WriteLn('GLA (leaf)     params = ', GNet.CountWeights());
  WriteLn('GLA (builder)  params = ', BNet.CountWeights());
  WriteLn('DeltaNet       params = ', DNet.CountWeights());
  WriteLn('Retention      params = ', RNet.CountWeights());
  WriteLn;

  WriteLn('training all arms on the SAME recall stream (', cTrainSteps, ' steps each)...');
  // Same RNG stream replay so every arm sees identical training samples.
  RandSeed := 999;
  Train(GNet, cTrainSteps, 0.01);
  RandSeed := 999;
  Train(BNet, cTrainSteps, 0.01);
  RandSeed := 999;
  Train(DNet, cTrainSteps, 0.01);
  RandSeed := 999;
  Train(RNet, cTrainSteps, 0.01);
  WriteLn;

  RandSeed := 7;
  gMSE := Evaluate(GNet, cEvalSeqs, gAcc);
  RandSeed := 7;
  bMSE := Evaluate(BNet, cEvalSeqs, bAcc);
  RandSeed := 7;
  dMSE := Evaluate(DNet, cEvalSeqs, dAcc);
  RandSeed := 7;
  rMSE := Evaluate(RNet, cEvalSeqs, rAcc);

  WriteLn('eval over ', cEvalSeqs, ' held-out recall sequences:');
  WriteLn('  GLA (leaf, per-chan gate): recall MSE = ', gMSE:0:5,
          '   exact-recall acc = ', (gAcc * 100):0:1, '%');
  WriteLn('  GLA (AddGatedLinearAtt.) : recall MSE = ', bMSE:0:5,
          '   exact-recall acc = ', (bAcc * 100):0:1, '%');
  WriteLn('  DeltaNet (delta rule)  : recall MSE = ', dMSE:0:5,
          '   exact-recall acc = ', (dAcc * 100):0:1, '%');
  WriteLn('  Retention (fixed decay): recall MSE = ', rMSE:0:5,
          '   exact-recall acc = ', (rAcc * 100):0:1, '%');
  WriteLn;

  if (gMSE < rMSE) and (gAcc >= rAcc) then
    WriteLn('OK: the gated-linear-attention layer recalls stored values better ',
            'than the single-scalar fixed-decay baseline.')
  else
    WriteLn('WARNING: GLA did not beat the Retention baseline on recall.');

  GNet.Free; BNet.Free; DNet.Free; RNet.Free;
end.
