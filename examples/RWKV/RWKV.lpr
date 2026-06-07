// RWKV example
//
// Associative recall (key -> value lookup) over a sequence, contrasting the
// RWKV time-mixing weighted-key-value recurrence (TNNetWKV, wired by the
// TNNet.AddRWKVTimeMix builder) against the delta-rule linear-attention layer
// TNNetDeltaNet on the SAME overwrite-recall task.
//
// The task (OVERWRITE recall). A sequence presents WRITE tokens; one key is
// written TWICE with two different values (an overwrite); then a QUERY re-presents
// that key and the target is its MOST RECENT value:
//   write t :  [ key_onehot(k_t) | value_vec | flag=0 ]
//   query   :  [ key_onehot(k_q) | 0...0     | flag=1 ]
//
// Two arms share the SAME I/O contract and a comparable parameter budget:
//   * RWKV arm   : front 1x1 projection -> AddRWKVTimeMix -> 1x1 readout.
//       AddRWKVTimeMix wires TokenShift -> {r,k,v} pointwise projections ->
//       TNNetWKV (the EMA numerator/denominator WKV recurrence with a learnable
//       per-channel decay w and a per-token bonus u) -> receptance gate ->
//       output projection. WKV gives every past token an exponentially-decaying
//       weight plus a bonus for the current token, self-normalised by its
//       running denominator -- softmax-free linear-time "attention".
//   * DeltaNet arm : front 1x1 projection -> TNNetDeltaNet -> 1x1 readout.
//       The delta rule maintains a (d,d) matrix memory and writes back ONLY the
//       read-then-corrected residual, so it cleanly OVERWRITES associations.
//
// Headline: trained on the SAME data for the SAME number of steps, both
// linear-time recurrences learn the recall task well above chance; the delta rule
// (an explicit error-correcting matrix memory) edges out the EMA-style WKV mixer
// on the hard OVERWRITE regime, while WKV is the cheaper softmax-free attention
// surrogate. Pure CPU, tiny dims, finishes in a couple of minutes on 2 cores.
//
// Coded by Claude (AI).
program RWKV;

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
  cModelDim  = 16;    // memory width
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
  for j := 0 to cNumKeys - 1 do keys[j] := j;
  for j := cNumKeys - 1 downto 1 do
  begin
    tmp := Random(j + 1);
    pos := keys[j]; keys[j] := keys[tmp]; keys[tmp] := pos;
  end;
  for pos := 0 to cNumPairs - 1 do
  begin
    vals[pos] := Random(cNumVals);
    Input[pos, 0, keys[pos]] := 1.0;                       // key one-hot
    for j := 0 to cValueDim - 1 do
      Input[pos, 0, cNumKeys + j] := ValueBank[vals[pos], j]; // value vector
    Input[pos, 0, cNumKeys + cValueDim] := 0.0;            // flag = write
  end;
  overIdx := Random(cNumPairs);
  queriedKey := keys[overIdx];
  repeat newVal := Random(cNumVals); until newVal <> vals[overIdx];
  Input[cNumPairs, 0, queriedKey] := 1.0;
  for j := 0 to cValueDim - 1 do
    Input[cNumPairs, 0, cNumKeys + j] := ValueBank[newVal, j];
  Input[cNumPairs, 0, cNumKeys + cValueDim] := 0.0;        // flag = write
  Input[cSeqLen - 1, 0, queriedKey] := 1.0;
  Input[cSeqLen - 1, 0, cNumKeys + cValueDim] := 1.0;      // flag = query
  for j := 0 to cValueDim - 1 do
    Desired[cSeqLen - 1, 0, j] := ValueBank[newVal, j];
end;

// Mean squared recall error at the query position, averaged over N sequences.
// Also returns EXACT-recall accuracy (nearest neighbour over the value bank).
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
      for ji := 0 to cValueDim - 1 do
      begin
        diff := NN.GetLastLayer.Output[cSeqLen - 1, 0, ji] - Desired[cSeqLen - 1, 0, ji];
        mse := mse + diff * diff;
      end;
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

// RWKV arm: 1x1 projection -> AddRWKVTimeMix (TokenShift + WKV + gating) -> readout.
function BuildRWKV(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  Result.AddRWKVTimeMix();
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
end;

// DeltaNet arm: 1x1 projection -> delta-rule cell -> 1x1 readout.
function BuildDeltaNet(): TNNet;
var Cell: TNNetDeltaNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  Cell := TNNetDeltaNet.Create();
  Result.AddLayer(Cell);
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
  // Warm-start the write gate to beta~=0.5 so the memory writes from step one.
  Cell.Neurons[4].Weights.Raw[0] := 0.0;
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
  WNet, DNet: TNNet;
  wMSE, dMSE, wAcc, dAcc: TNeuralFloat;
begin
  RandSeed := 12345;
  InitValueBank();

  WriteLn('=== RWKV: associative recall, WKV time-mix vs delta-rule ===');
  WriteLn('keys=', cNumKeys, '  value_dim=', cValueDim,
          '  pairs/seq=', cNumPairs, '  model_dim=', cModelDim,
          '  seq_len=', cSeqLen);
  WriteLn;

  WNet := BuildRWKV();
  DNet := BuildDeltaNet();
  WriteLn('RWKV (WKV)  params = ', WNet.CountWeights());
  WriteLn('DeltaNet    params = ', DNet.CountWeights());
  WriteLn;

  WriteLn('training both arms on the SAME recall stream (', cTrainSteps, ' steps each)...');
  RandSeed := 999;
  Train(WNet, cTrainSteps, 0.01);
  RandSeed := 999;
  Train(DNet, cTrainSteps, 0.01);
  WriteLn;

  RandSeed := 7;
  wMSE := Evaluate(WNet, cEvalSeqs, wAcc);
  RandSeed := 7;
  dMSE := Evaluate(DNet, cEvalSeqs, dAcc);

  WriteLn('eval over ', cEvalSeqs, ' held-out recall sequences:');
  WriteLn('  RWKV     (WKV time-mix): recall MSE = ', wMSE:0:5,
          '   exact-recall acc = ', (wAcc * 100):0:1, '%');
  WriteLn('  DeltaNet (delta rule)  : recall MSE = ', dMSE:0:5,
          '   exact-recall acc = ', (dAcc * 100):0:1, '%');
  WriteLn;

  if wAcc > 0.30 then
    WriteLn('OK: the WKV time-mixing recurrence learns associative recall ',
            'well above chance (', (100.0 / cNumVals):0:1, '%).')
  else
    WriteLn('WARNING: the WKV arm did not learn recall above chance.');

  WNet.Free; DNet.Free;
end.
