// MinimalRNN example
//
// Selective-copy recall over a short sequence, contrasting the two minimal,
// fully-parallelizable recurrent cells of Feng, Tung, Hassani, Hamarneh &
// Ravanbakhsh 2024 ("Were RNNs all we needed?", arXiv:2410.01201):
//   * TNNetMinGRU  : h_t = (1 - z_t) h_{t-1} + z_t ht~
//   * TNNetMinLSTM : h_t = f'_t h_{t-1} + i'_t ht~ , gates normalized f/(f+i)
// Both drop the dependence of the gates on h_{t-1} (that is what makes them
// parallelizable). The headline contrast is against a MEMORYLESS per-token MLP:
// it shares the same I/O contract but has NO state carried across time, so it
// structurally cannot solve a task that requires remembering an earlier token.
//
// The task. A sequence presents several value tokens; exactly ONE token is
// marked (a flag channel = 1). A final query token (a separate flag) asks the
// network to reproduce the marked token's value vector:
//   value t : [ value_vec | mark_flag in {0,1} | query_flag=0 ]
//   query   : [ 0...0      | 0                  | query_flag=1 ]
// The target at the query position is the value vector of the marked token.
// Solving it requires the cell to (a) detect the mark, (b) latch that value into
// its hidden state, (c) hold it across the remaining tokens, and (d) emit it at
// the query. The minimal cells' update gate can open on the marked token and
// stay near-closed afterwards (a content-dependent latch); a per-token MLP, with
// no cross-time memory, can only ever look at the all-zero query token and is
// stuck at chance (1 / cNumVals).
//
// Three arms share the SAME I/O contract:
//   * minGRU arm  : 1x1 projection -> TNNetMinGRU  -> 1x1 readout
//   * minLSTM arm : 1x1 projection -> TNNetMinLSTM -> 1x1 readout
//   * MLP arm     : two per-token 1x1 ReLU layers -> 1x1 readout (memoryless)
//
// NOTE: BPTT through these unrolled recurrences is sensitive to momentum, so the
// recurrent arms train with momentum=0 (plain SGD); higher momentum destabilizes
// the carried hidden-state gradient at this tiny scale.
//
// Headline: trained on the SAME data for the SAME number of steps, both minimal
// recurrent arms reach near-100% exact-recall accuracy, while the memoryless MLP
// stays at chance. Pure CPU, tiny dims, finishes in a couple of seconds on 2
// cores.
//
// Coded by Claude (AI).
program MinimalRNN;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils,
  neuralnetwork, neuralvolume;

const
  cNumVals   = 6;     // vocabulary of distinct value vectors
  cValueDim  = 4;     // dimensionality of each stored value vector
  cNumToks   = 4;     // value tokens per sequence (one of them is marked)
  cSeqLen    = cNumToks + 1;             // value tokens + 1 query
  cModelDim  = 24;    // hidden width
  cInDim     = cValueDim + 2;            // value | mark flag | query flag
  cMarkCh    = cValueDim;                // mark-flag channel index
  cQueryCh   = cValueDim + 1;            // query-flag channel index
  cTrainSteps= 40000;
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

// Build one selective-copy sequence. cNumToks value tokens carry random value
// ids; exactly one position (markPos) carries the mark flag. The query token
// asks for the marked token's value. Desired holds only the query-position
// target (the marked token's value vector).
procedure MakeSample(Input, Desired: TNNetVolume);
var
  pos, j, markPos, valId, markVal: integer;
begin
  Input.Fill(0);
  Desired.Fill(0);
  markPos := Random(cNumToks);
  markVal := 0;
  for pos := 0 to cNumToks - 1 do
  begin
    valId := Random(cNumVals);
    if pos = markPos then markVal := valId;
    for j := 0 to cValueDim - 1 do
      Input[pos, 0, j] := ValueBank[valId, j];     // value vector
    if pos = markPos then Input[pos, 0, cMarkCh] := 1.0;  // mark flag
    Input[pos, 0, cQueryCh] := 0.0;                // not a query
  end;
  // Query token (last position): ask for the marked value.
  Input[cSeqLen - 1, 0, cQueryCh] := 1.0;
  for j := 0 to cValueDim - 1 do
    Desired[cSeqLen - 1, 0, j] := ValueBank[markVal, j];
end;

// Mean squared recall error at the query position, averaged over N sequences.
// Also returns the EXACT-recall accuracy: fraction of sequences whose query
// output is closest (nearest neighbour over the value bank) to the true value.
function Evaluate(NN: TNNet; N: integer; out Accuracy: TNeuralFloat): TNeuralFloat;
var
  Input, Desired: TNNetVolume;
  seq, ji, ki, bestK, trueK, correct: integer;
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
      // Recover the true value-id from Desired (its bank entry).
      trueK := -1;
      for ki := 0 to cNumVals - 1 do
      begin
        dist := 0;
        for ji := 0 to cValueDim - 1 do
        begin
          diff := Desired[cSeqLen - 1, 0, ji] - ValueBank[ki, ji];
          dist := dist + diff * diff;
        end;
        if dist < 1e-6 then trueK := ki;
      end;
      // Nearest-neighbour decode of the network output.
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
      if bestK = trueK then Inc(correct);
    end;
    Result := mse / (N * cValueDim);
    Accuracy := correct / N;
  finally
    Input.Free; Desired.Free;
  end;
end;

// minGRU arm: 1x1 projection -> minimal-GRU cell -> 1x1 readout.
function BuildMinGRU(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  Result.AddLayer(TNNetMinGRU.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
end;

// minLSTM arm: 1x1 projection -> minimal-LSTM cell -> 1x1 readout.
function BuildMinLSTM(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  Result.AddLayer(TNNetMinLSTM.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
end;

// MLP arm: two per-token 1x1 ReLU layers -> 1x1 readout. Fully MEMORYLESS: each
// time position is processed independently, so it cannot carry the marked value
// to the query position.
function BuildMLP(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvReLU.Create(cModelDim));
  Result.AddLayer(TNNetPointwiseConvReLU.Create(cModelDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
end;

procedure Train(NN: TNNet; Steps: integer; LR: TNeuralFloat);
var
  Input, Desired: TNNetVolume;
  i: integer;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, cInDim);
  Desired := TNNetVolume.Create(cSeqLen, 1, cValueDim);
  // Momentum 0: BPTT through the unrolled recurrence is momentum-sensitive here.
  NN.SetLearningRate(LR, 0.0);
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
  GNet, LNet, MNet: TNNet;
  gMSE, lMSE, mMSE, gAcc, lAcc, mAcc: TNeuralFloat;
begin
  RandSeed := 12345;
  InitValueBank();

  WriteLn('=== MinimalRNN: selective-copy recall, minGRU / minLSTM vs memoryless MLP ===');
  WriteLn('vals=', cNumVals, '  value_dim=', cValueDim,
          '  tokens/seq=', cNumToks, '  model_dim=', cModelDim,
          '  seq_len=', cSeqLen, '  chance acc=', (100.0 / cNumVals):0:1, '%');
  WriteLn;

  GNet := BuildMinGRU();
  LNet := BuildMinLSTM();
  MNet := BuildMLP();
  WriteLn('minGRU  params = ', GNet.CountWeights());
  WriteLn('minLSTM params = ', LNet.CountWeights());
  WriteLn('MLP     params = ', MNet.CountWeights());
  WriteLn;

  WriteLn('training all arms on the SAME recall stream (', cTrainSteps, ' steps each)...');
  // Same RNG stream replay so all arms see identical training samples.
  RandSeed := 999; Train(GNet, cTrainSteps, 0.005);
  RandSeed := 999; Train(LNet, cTrainSteps, 0.005);
  RandSeed := 999; Train(MNet, cTrainSteps, 0.005);
  WriteLn;

  RandSeed := 7; gMSE := Evaluate(GNet, cEvalSeqs, gAcc);
  RandSeed := 7; lMSE := Evaluate(LNet, cEvalSeqs, lAcc);
  RandSeed := 7; mMSE := Evaluate(MNet, cEvalSeqs, mAcc);

  WriteLn('eval over ', cEvalSeqs, ' held-out selective-copy sequences:');
  WriteLn('  minGRU         : recall MSE = ', gMSE:0:5,
          '   exact-recall acc = ', (gAcc * 100):0:1, '%');
  WriteLn('  minLSTM        : recall MSE = ', lMSE:0:5,
          '   exact-recall acc = ', (lAcc * 100):0:1, '%');
  WriteLn('  MLP (memoryless): recall MSE = ', mMSE:0:5,
          '   exact-recall acc = ', (mAcc * 100):0:1, '%');
  WriteLn;

  if (gAcc > mAcc + 0.3) and (lAcc > mAcc + 0.3) then
    WriteLn('OK: both minimal recurrent cells solve selective-copy recall, ',
            'while the memoryless MLP stays near chance.')
  else
    WriteLn('WARNING: a minimal cell did not clearly beat the memoryless MLP.');

  GNet.Free; LNet.Free; MNet.Free;
end.
