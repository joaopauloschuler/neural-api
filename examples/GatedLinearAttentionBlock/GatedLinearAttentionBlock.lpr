// Gated Linear Attention BLOCK example
//
// Shows the TNNet.AddGatedLinearAttentionBlock composite builder: a full
// transformer-style block that wraps the gated-linear-attention time mixer
// (TNNet.AddGatedLinearAttention, itself around the TNNetGatedLinearAttention
// leaf, Yang et al. 2023, arXiv:2312.06635) in a PRE-NORM residual structure
// plus a token-wise SwiGLU feed-forward residual:
//   x := x + GLA(LayerNorm(x))
//   x := x + FFN(LayerNorm(x))   (PointwiseConvLinear -> SwiGLU -> PointwiseConvLinear)
// mirroring AddTransformerEncoderBlock but with the self-attention arm replaced
// by gated linear attention. It is shape-preserving over a (SeqLen,1,d_model)
// sequence, so blocks STACK into a deep tower.
//
// The task (OVERWRITE recall). A sequence presents WRITE tokens; one key is
// written TWICE with two different values (an overwrite); then a QUERY
// re-presents that key and the target is the key's MOST RECENT value. Retrieving
// it requires forgetting the stale association and keeping the fresh one -- a
// natural fit for GLA's data-dependent per-channel forget gate.
//
// Two arms on the SAME data / step budget / matched-ish width:
//   * Bare mixer : 1x1 projection -> AddGatedLinearAttention -> 1x1 readout.
//       A single GLA time-mix, no residual norm and no feed-forward.
//   * Block tower: 1x1 projection -> AddGatedLinearAttentionBlock x cDepth ->
//       1x1 readout. The same mixer, but each layer is residual-wrapped with
//       LayerNorm + a SwiGLU FFN and several are stacked for depth.
//
// Headline: the residual+FFN block tower trains stably and reaches markedly
// lower recall error (and higher exact-recall accuracy) than the single bare
// mixer -- the LayerNorm/residual/FFN scaffolding the block adds is what makes
// deep stacking of the mixer pay off. Pure CPU, tiny dims, finishes in well
// under a couple of minutes on 2 cores.
//
// Coded by Claude (AI).
program GatedLinearAttentionBlock;

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
  cModelDim  = 16;    // residual-stream width
  cFF        = 32;    // feed-forward inner width inside each block
  cDepth     = 3;     // number of stacked GLA blocks in the tower arm
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
// random value-ids; then one written key is RE-WRITTEN with a different value-id
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
    Input[pos, 0, keys[pos]] := 1.0;                          // key one-hot
    for j := 0 to cValueDim - 1 do
      Input[pos, 0, cNumKeys + j] := ValueBank[vals[pos], j]; // value vector
    Input[pos, 0, cNumKeys + cValueDim] := 0.0;               // flag = write
  end;
  // Overwrite token (position cNumPairs): pick one written key, give it a NEW
  // value-id distinct from its first one.
  overIdx := Random(cNumPairs);
  queriedKey := keys[overIdx];
  repeat newVal := Random(cNumVals); until newVal <> vals[overIdx];
  Input[cNumPairs, 0, queriedKey] := 1.0;
  for j := 0 to cValueDim - 1 do
    Input[cNumPairs, 0, cNumKeys + j] := ValueBank[newVal, j];
  Input[cNumPairs, 0, cNumKeys + cValueDim] := 0.0;           // flag = write
  // Query token (position cSeqLen-1): re-present the overwritten key; target is
  // its MOST RECENT value (newVal), NOT the stale one.
  Input[cSeqLen - 1, 0, queriedKey] := 1.0;
  Input[cSeqLen - 1, 0, cNumKeys + cValueDim] := 1.0;         // flag = query
  for j := 0 to cValueDim - 1 do
    Desired[cSeqLen - 1, 0, j] := ValueBank[newVal, j];
end;

// Mean squared recall error at the query position, averaged over N sequences.
// Also returns EXACT-recall accuracy: fraction of sequences whose query output
// is closest (nearest neighbour over the value bank) to the true value.
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
      // Nearest-neighbour decode over the value bank -> exact-recall check.
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

// Bare-mixer arm: a single AddGatedLinearAttention time mixer between projections.
function BuildBareMixer(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  Result.AddGatedLinearAttention();
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
end;

// Block-tower arm: cDepth stacked AddGatedLinearAttentionBlock (the SAME mixer,
// but each layer is a LayerNorm/residual + SwiGLU-FFN block) between projections.
function BuildBlockTower(): TNNet;
var d: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  for d := 0 to cDepth - 1 do
    Result.AddGatedLinearAttentionBlock(cFF);
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
  MNet, TNet: TNNet;
  mMSE, tMSE, mAcc, tAcc: TNeuralFloat;
begin
  RandSeed := 12345;
  InitValueBank();

  WriteLn('=== AddGatedLinearAttentionBlock: residual+FFN block tower vs a bare mixer ===');
  WriteLn('keys=', cNumKeys, '  value_dim=', cValueDim,
          '  pairs/seq=', cNumPairs, '  model_dim=', cModelDim,
          '  d_ff=', cFF, '  depth=', cDepth, '  seq_len=', cSeqLen);
  WriteLn;

  MNet := BuildBareMixer();
  TNet := BuildBlockTower();
  WriteLn('bare GLA mixer       params = ', MNet.CountWeights());
  WriteLn('GLA block tower (x', cDepth, ') params = ', TNet.CountWeights());
  WriteLn;

  WriteLn('training both arms on the SAME recall stream (', cTrainSteps, ' steps each)...');
  // Same RNG stream replay so both arms see identical training samples.
  RandSeed := 999;
  Train(MNet, cTrainSteps, 0.005);
  RandSeed := 999;
  Train(TNet, cTrainSteps, 0.005);
  WriteLn;

  RandSeed := 7;
  mMSE := Evaluate(MNet, cEvalSeqs, mAcc);
  RandSeed := 7;
  tMSE := Evaluate(TNet, cEvalSeqs, tAcc);

  WriteLn('eval over ', cEvalSeqs, ' held-out recall sequences:');
  WriteLn('  bare GLA mixer        : recall MSE = ', mMSE:0:5,
          '   exact-recall acc = ', (mAcc * 100):0:1, '%');
  WriteLn('  GLA block tower (x', cDepth, ') : recall MSE = ', tMSE:0:5,
          '   exact-recall acc = ', (tAcc * 100):0:1, '%');
  WriteLn;

  if (tMSE < mMSE) and (tAcc >= mAcc) then
    WriteLn('OK: the residual+FFN block tower recalls stored values better than ',
            'the single bare gated-linear-attention mixer.')
  else
    WriteLn('WARNING: the block tower did not beat the bare mixer on recall.');

  MNet.Free; TNet.Free;
end.
