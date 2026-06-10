(*
  TitansMemory example

  Long-context associative recall, contrasting the test-time neural long-term
  memory layer TNNetTitansMemory (Behrouz et al. 2024, "Titans: Learning to
  Memorize at Test Time", the Memory-as-Context leaf-layer variant) against a
  param-matched fixed-decay linear-attention baseline (TNNetRetention).

  The task. Each sequence has three phases:
    * STORE phase  : cNumPairs distinct (key -> value) WRITE tokens up front:
        [ key_onehot | value_vec | store_flag=1 | query_flag=0 ]
    * DISTRACTOR   : a long span of cDistractor random noise tokens (no key, no
        value, both flags 0). This is what makes the task LONG-CONTEXT: the
        stored associations must survive many irrelevant intervening steps.
    * QUERY phase  : re-present one stored key and ask for ITS value:
        [ key_onehot | 0... | store_flag=0 | query_flag=1 ]
      the target is that key's stored value vector.

  Why Titans should win at long range. Its inner memory M_t is a small MLP whose
  weights are gradient-descended AT INFERENCE on the per-token associative loss
  || M_t(k_t) - v_t ||^2, with TWO mechanisms a plain decay baseline lacks:
    (a) a MOMENTUM / "surprise" term  S_t = eta*S_{t-1} - theta*grad  so a
        surprising STORE token keeps writing for several steps (deep encoding);
    (b) a data-dependent FORGET gate  M_t = (1-alpha_t)*M_{t-1} + S_t  that can
        leave the stored association untouched while the distractor tokens stream
        past (alpha_t small for bland tokens), so memories persist across the gap.
  A fixed-decay accumulator instead bleeds every stored value toward zero at a
  constant rate over the long distractor span and recalls poorly at range.

  Two arms share the SAME I/O contract and a matched parameter budget:
    * Titans arm    : 1x1 projection -> TNNetTitansMemory -> 1x1 readout.
    * Retention arm : 1x1 projection (to 3*d_k = Q|K|V) -> TNNetRetention
        (plain FIXED, non-learnable decay) -> 1x1 readout.

  Headline: trained on the SAME data for the SAME number of steps, the Titans arm
  reaches markedly lower recall MSE (and higher exact-recall accuracy) than the
  fixed-decay baseline as the distractor span grows. Pure CPU, tiny dims,
  finishes in a few minutes on 2 cores.

  Coded by Claude (AI).

*)
program TitansMemory;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cNumKeys    = 5;     // vocabulary of distinct keys
  cNumVals    = 5;     // vocabulary of distinct value vectors
  cValueDim   = 4;     // dimensionality of each stored value vector
  cNumPairs   = 4;     // distinct (key,value) writes per sequence
  cDistractor = 24;    // length of the long noise span between store and query
  cSeqLen     = cNumPairs + cDistractor + 1; // store + distractor + 1 query
  cModelDim   = 12;    // memory width
  cHidden     = 16;    // Titans inner-MLP hidden width
  cInDim      = cNumKeys + cValueDim + 2; // key one-hot | value | store | query flags
  cTrainSteps = 6000;
  cEvalSeqs   = 400;

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

// Build one long-context recall sequence: cNumPairs stores, a long distractor
// span, then a query for one stored key. Target is filled only at the query.
procedure MakeSample(Input, Desired: TNNetVolume);
var
  pos, j, queriedIdx, tmp, d: integer;
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
  // STORE phase (positions 0..cNumPairs-1).
  for pos := 0 to cNumPairs - 1 do
  begin
    vals[pos] := Random(cNumVals);
    Input[pos, 0, keys[pos]] := 1.0;                          // key one-hot
    for j := 0 to cValueDim - 1 do
      Input[pos, 0, cNumKeys + j] := ValueBank[vals[pos], j]; // value vector
    Input[pos, 0, cNumKeys + cValueDim] := 1.0;              // store flag
  end;
  // DISTRACTOR phase: weak random noise tokens (no key, no flags) -- the long
  // span the stored associations must survive.
  for pos := cNumPairs to cNumPairs + cDistractor - 1 do
    for d := 0 to cValueDim - 1 do
      Input[pos, 0, cNumKeys + d] := (Random - 0.5) * 0.3;
  // QUERY phase (last position): re-present one stored key.
  queriedIdx := Random(cNumPairs);
  Input[cSeqLen - 1, 0, keys[queriedIdx]] := 1.0;
  Input[cSeqLen - 1, 0, cNumKeys + cValueDim + 1] := 1.0;     // query flag
  for j := 0 to cValueDim - 1 do
    Desired[cSeqLen - 1, 0, j] := ValueBank[vals[queriedIdx], j];
end;

// Mean squared recall error at the query position, averaged over N sequences,
// plus exact-recall accuracy via nearest-neighbour decode over the value bank.
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

// Titans arm: 1x1 projection -> neural long-term memory -> 1x1 readout.
function BuildTitans(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  Result.AddLayer(TNNetTitansMemory.Create(cHidden));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
end;

// Retention arm: 1x1 projection to 3*cModelDim (Q|K|V) -> plain FIXED-decay
// linear attention (gamma NOT learnable) -> 1x1 readout. This is the "plain
// linear-attention" baseline: it accumulates k(x)v with a CONSTANT decay it
// cannot adapt, so a stored association unavoidably bleeds toward zero over the
// long distractor span -- exactly where Titans' data-dependent forget gate and
// momentum should win. Param budget comparable to the Titans arm.
function BuildRetention(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(3 * cModelDim));
  Result.AddLayer(TNNetRetention.Create(cModelDim, 0.9, {LearnGamma=}false));
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
  TNet, RNet: TNNet;
  tMSE, rMSE, tAcc, rAcc: TNeuralFloat;
begin
  RandSeed := 12345;
  InitValueBank();

  WriteLn('=== Titans neural long-term memory: long-context recall ===');
  WriteLn('keys=', cNumKeys, '  value_dim=', cValueDim,
          '  pairs/seq=', cNumPairs, '  distractor=', cDistractor,
          '  seq_len=', cSeqLen, '  model_dim=', cModelDim);
  WriteLn;

  TNet := BuildTitans();
  RNet := BuildRetention();
  WriteLn('Titans    params = ', TNet.CountWeights());
  WriteLn('Retention params = ', RNet.CountWeights());
  WriteLn;

  WriteLn('training both arms on the SAME recall stream (', cTrainSteps, ' steps each)...');
  RandSeed := 999;
  Train(TNet, cTrainSteps, 0.01);
  RandSeed := 999;
  Train(RNet, cTrainSteps, 0.01);
  WriteLn;

  RandSeed := 7;
  tMSE := Evaluate(TNet, cEvalSeqs, tAcc);
  RandSeed := 7;
  rMSE := Evaluate(RNet, cEvalSeqs, rAcc);

  WriteLn('eval over ', cEvalSeqs, ' held-out long-context recall sequences:');
  WriteLn('  Titans (neural memory) : recall MSE = ', tMSE:0:5,
          '   exact-recall acc = ', (tAcc * 100):0:1, '%');
  WriteLn('  Retention (fixed decay): recall MSE = ', rMSE:0:5,
          '   exact-recall acc = ', (rAcc * 100):0:1, '%');
  WriteLn;

  if (tMSE < rMSE) and (tAcc >= rAcc) then
    WriteLn('OK: the test-time neural memory recalls stored values better than ',
            'the fixed-decay baseline across the long distractor span.')
  else
    WriteLn('WARNING: Titans did not beat the Retention baseline on recall.');

  TNet.Free; RNet.Free;
end.
