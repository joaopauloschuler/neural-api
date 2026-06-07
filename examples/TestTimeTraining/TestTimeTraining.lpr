// Test-Time Training (TTT) example
//
// Associative recall + copy demo contrasting THREE recurrent sequence mixers at
// a MATCHED memory width, to show the headline property of Test-Time Training
// (Sun et al. 2024, arXiv:2407.04620): when the inner "hidden state" is itself a
// small NON-LINEAR model trained on the fly (TTT-MLP), it solves a binding task
// that the LINEAR-state mixers (TTT-Linear and the landed DeltaNet) plateau on.
//
// The task (NON-LINEAR binding recall). A sequence presents WRITE tokens that
// each carry TWO key fields (a one-hot "row" key and a one-hot "col" key) and a
// query flag. The value to be recalled at a queried key is the bank entry indexed
// by the XOR of the two key ids -- a deliberately NON-LINEAR (parity-like)
// function of the inputs that a single matrix associative memory cannot cleanly
// separate, but a 2-layer inner net (GeLU) can:
//   write t :  [ row_onehot(r_t) | col_onehot(c_t) | value_vec(bank[r_t xor c_t]) | flag=0 ]
//   query   :  [ row_onehot(r_q) | col_onehot(c_q) | 0...0                        | flag=1 ]
// The target at the query position is bank[r_q xor c_q] for the re-presented key.
//
// Three arms share the SAME I/O contract and a MATCHED memory width (cModelDim):
//   * TTT-MLP   : front 1x1 projection -> TNNetTestTimeTraining(variant=MLP) ->
//                 1x1 readout. The inner state is W2*GeLU(W1 .) trained one SGD
//                 step per token -> a genuine non-linear fast-weight memory.
//   * TTT-Linear: identical, but the inner state is a single matrix (variant=0);
//                 its per-token update is a rank-1 MSE-gradient (delta-rule-like)
//                 write.
//   * DeltaNet  : the landed TNNetDeltaNet matrix delta-rule recurrence (sigmoid
//                 write gate, L2-normalized key) -- the other linear-state arm.
//
// Headline: trained on the SAME data for the SAME number of steps, the TTT-MLP
// arm reaches markedly lower recall error (and higher exact-recall accuracy) than
// BOTH linear-state arms, which plateau on the parity binding. Pure CPU, tiny
// dims, short sequences, finishes in well under five minutes on 2 cores.
//
// Coded by Claude (AI).
program TestTimeTraining;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cNumRows   = 4;     // row-key vocabulary
  cNumCols   = 4;     // col-key vocabulary
  cNumVals   = cNumRows;       // bank size = max(r xor c)+1 over {0..3}x{0..3} = 4
  cValueDim  = 4;     // dimensionality of each stored value vector
  cNumPairs  = 3;     // distinct writes per sequence
  cSeqLen    = cNumPairs + 1;            // writes + 1 query
  cModelDim  = 16;    // shared memory width
  cHidden    = 24;    // TTT-MLP inner hidden width
  cInDim     = cNumRows + cNumCols + cValueDim + 1; // row|col one-hot | value | flag
  cTrainSteps= 8000;
  cEvalSeqs  = 400;

var
  ValueBank: array[0..cNumVals - 1, 0..cValueDim - 1] of TNeuralFloat;

procedure InitValueBank();
var k, j: integer;
begin
  for k := 0 to cNumVals - 1 do
    for j := 0 to cValueDim - 1 do
      ValueBank[k, j] := Sin(k * 1.7 + j * 0.9) * 0.6 + Cos(k * 0.5 - j * 1.3) * 0.4;
end;

// Build one non-linear-binding recall sequence. cNumPairs distinct (row,col)
// pairs are written, each carrying bank[row xor col]; then one written key is
// re-presented as a query and the target is its bank value.
procedure MakeSample(Input, Desired: TNNetVolume);
var
  pos, j, queryIdx, valId: integer;
  rows, cols: array[0..cNumPairs - 1] of integer;
  usedKey: array[0..cNumRows * cNumCols - 1] of boolean;
  rr, cc, kk: integer;
begin
  Input.Fill(0);
  Desired.Fill(0);
  for j := 0 to cNumRows * cNumCols - 1 do usedKey[j] := false;
  // Distinct (row,col) write keys.
  for pos := 0 to cNumPairs - 1 do
  begin
    repeat
      rr := Random(cNumRows);
      cc := Random(cNumCols);
      kk := rr * cNumCols + cc;
    until not usedKey[kk];
    usedKey[kk] := true;
    rows[pos] := rr; cols[pos] := cc;
    valId := rr xor cc; // NON-LINEAR (parity) binding
    Input[pos, 0, rr] := 1.0;                       // row one-hot
    Input[pos, 0, cNumRows + cc] := 1.0;            // col one-hot
    for j := 0 to cValueDim - 1 do
      Input[pos, 0, cNumRows + cNumCols + j] := ValueBank[valId, j];
    Input[pos, 0, cInDim - 1] := 0.0;               // flag = write
  end;
  // Query token: re-present one written key; target is bank[row xor col].
  queryIdx := Random(cNumPairs);
  rr := rows[queryIdx]; cc := cols[queryIdx];
  Input[cSeqLen - 1, 0, rr] := 1.0;
  Input[cSeqLen - 1, 0, cNumRows + cc] := 1.0;
  Input[cSeqLen - 1, 0, cInDim - 1] := 1.0;         // flag = query
  valId := rr xor cc;
  for j := 0 to cValueDim - 1 do
    Desired[cSeqLen - 1, 0, j] := ValueBank[valId, j];
end;

// Mean squared recall error + exact (nearest-neighbour) recall accuracy.
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

// Shared scaffold: 1x1 projection -> mixer -> 1x1 readout. The caller supplies
// the recurrent mixer layer.
function BuildArm(Mixer: TNNetLayer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  Result.AddLayer(Mixer);
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
  MLPNet, LinNet, DNet: TNNet;
  mMSE, lMSE, dMSE, mAcc, lAcc, dAcc: TNeuralFloat;
begin
  RandSeed := 12345;
  InitValueBank();

  WriteLn('=== Test-Time Training: non-linear binding recall ===');
  WriteLn('rows=', cNumRows, ' cols=', cNumCols, ' value_dim=', cValueDim,
          ' pairs/seq=', cNumPairs, ' model_dim=', cModelDim,
          ' mlp_hidden=', cHidden, ' seq_len=', cSeqLen);
  WriteLn('value index = row XOR col  (a non-linear binding of the two keys)');
  WriteLn;

  MLPNet := BuildArm(TNNetTestTimeTraining.Create({variant=}1, cHidden));
  LinNet := BuildArm(TNNetTestTimeTraining.Create({variant=}0, 0));
  DNet   := BuildArm(TNNetDeltaNet.Create());
  WriteLn('TTT-MLP    params = ', MLPNet.CountWeights());
  WriteLn('TTT-Linear params = ', LinNet.CountWeights());
  WriteLn('DeltaNet   params = ', DNet.CountWeights());
  WriteLn;

  WriteLn('training all three arms on the SAME recall stream (', cTrainSteps, ' steps each)...');
  RandSeed := 999; Train(MLPNet, cTrainSteps, 0.01);
  RandSeed := 999; Train(LinNet, cTrainSteps, 0.01);
  RandSeed := 999; Train(DNet,   cTrainSteps, 0.01);
  WriteLn;

  RandSeed := 7; mMSE := Evaluate(MLPNet, cEvalSeqs, mAcc);
  RandSeed := 7; lMSE := Evaluate(LinNet, cEvalSeqs, lAcc);
  RandSeed := 7; dMSE := Evaluate(DNet,   cEvalSeqs, dAcc);

  WriteLn('eval over ', cEvalSeqs, ' held-out recall sequences:');
  WriteLn('  TTT-MLP    (non-linear inner state): recall MSE = ', mMSE:0:5,
          '   exact-recall acc = ', (mAcc * 100):0:1, '%');
  WriteLn('  TTT-Linear (matrix inner state)    : recall MSE = ', lMSE:0:5,
          '   exact-recall acc = ', (lAcc * 100):0:1, '%');
  WriteLn('  DeltaNet   (matrix delta rule)     : recall MSE = ', dMSE:0:5,
          '   exact-recall acc = ', (dAcc * 100):0:1, '%');
  WriteLn;

  if (mMSE < lMSE) and (mMSE < dMSE) then
    WriteLn('OK: the non-linear TTT-MLP inner state beats BOTH linear-state ',
            'mixers on the parity binding task.')
  else
    WriteLn('WARNING: TTT-MLP did not beat both linear baselines on this run.');

  MLPNet.Free; LinNet.Free; DNet.Free;
end.
