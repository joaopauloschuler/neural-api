// CrossWKV example
//
// TWO-SOURCE external-memory recall with TNNetCrossWKV -- a copy/recall task
// that single-source TNNetWKV CANNOT express.
//
// TNNetWKV (the RWKV-4 weighted-key-value time-mixing recurrence) splits its
// OWN input into the key|value pair that drives its running WKV state, so the
// memory it accumulates and the stream that reads it are ONE sequence.
// TNNetCrossWKV instead reads the key|value stream from a SEPARATE source than
// the receptance/query stream (exactly as TNNetCrossAttention generalises
// self-attention's packed Q|K|V to two sources). That makes "build a memory
// over sequence M, then read it out with a DIFFERENT query sequence Q" a single
// layer -- which is what decode-time external memory needs.
//
// The task -- CROSS-COPY (the equal-length, position-aligned regime this v1
// supports). A MEMORY sequence carries a stream of random value vectors; a
// SEPARATE QUERY sequence carries a per-step "read" selector. The read-out at
// position t must reproduce the MEMORY value at position t -- a value that
// lives ONLY in the memory tensor, never in the query tensor:
//   memory token t (src B = key|value): [ select_onehot(t) | value_vec_t ]
//   query  token t (src A = receptance): [ read_pulse(t) ]
//   target at t                        : value_vec_t   (from the memory stream)
// The value stream is re-randomised every sequence, so the answer cannot be
// memorised: it MUST be read out of the cross-built WKV memory whose key|value
// comes from source B while the receptance comes from source A.
//
// Architecture (single CrossWKV layer between two projected sources):
//   memory branch (src B): Input(M) -> 1x1 Linear -> 2*C  (k|v packed)
//   query  branch (src A): Input(Q) -> 1x1 Linear ->   C  (receptance)
//   TNNetCrossWKV(memSrc) reads B's WKV state, gates by A's receptance
//   -> 1x1 Linear readout -> value_dim
// SEQLEN CONTRACT: this v1 of TNNetCrossWKV requires EQUAL seqlen on both
// sources; the read-out at t uses the WKV state accumulated over the memory
// source up to t (current-token bonus from the memory key at t), gated by the
// query receptance at t. (Asymmetric/full-context cross -- summarise the memory
// once, query with a different-length stream -- is a documented follow-up.)
//
// CONTRAST ARM: a single-source TNNetWKV that can only see the QUERY stream
// (it has NO access to the memory tensor). Since the value stream changes every
// sequence, it is structurally blind to the stored values and cannot beat
// chance -- which is the whole point of the two-source layer.
//
// Headline: the CrossWKV arm reaches high exact-recall accuracy reading values
// out of the cross-built memory; the memory-blind single-source WKV arm stays
// near chance. Pure CPU, tiny dims, finishes in well under a minute on 2 cores.
//
// Coded by Claude (AI).
program CrossWKV;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils,
  neuralnetwork, neuralvolume;

const
  cNumKeys   = 6;     // number of positions (= seqlen, position one-hot width)
  cNumVals   = 6;     // number of value vectors in the random value bank
  cValueDim  = 4;     // dimensionality of each stored value vector
  cSeqLen    = cNumKeys;                 // memory writes = query reads
  cModelDim  = 16;    // CrossWKV memory width (C)
  cMemDim    = cNumKeys + cValueDim;     // memory token: position one-hot | value
  cQryDim    = cNumKeys;                 // query token: position read pulse
  cTrainSteps= 4000;
  cEvalSeqs  = 400;

var
  // Fixed pool of candidate value vectors; each position draws one at random
  // per sequence (so the value stream is sequence-specific and unmemorisable).
  ValueBank: array[0..cNumVals - 1, 0..cValueDim - 1] of TNeuralFloat;

procedure InitValueBank();
var k, j: integer;
begin
  for k := 0 to cNumVals - 1 do
    for j := 0 to cValueDim - 1 do
      ValueBank[k, j] := Sin(k * 1.7 + j * 0.9) * 0.6 + Cos(k * 0.5 - j * 1.3) * 0.4;
end;

// Build one cross-COPY sample with a FRESH random per-position value stream.
// MemInput is the key|value source (B): position one-hot | value vector.
// QryInput the receptance source (A): a read pulse at each position. Desired
// holds the per-position target value (= the memory value at that position).
// PosVal[t] is the value-bank id at position t (returned so Evaluate decodes).
procedure MakeSample(MemInput, QryInput, Desired: TNNetVolume;
  out PosVal: array of integer);
var
  pos, j: integer;
begin
  MemInput.Fill(0);
  QryInput.Fill(0);
  Desired.Fill(0);
  for pos := 0 to cSeqLen - 1 do
  begin
    PosVal[pos] := Random(cNumVals);
    // Memory (src B): position one-hot | this position's random value vector.
    MemInput[pos, 0, pos] := 1.0;
    for j := 0 to cValueDim - 1 do
      MemInput[pos, 0, cNumKeys + j] := ValueBank[PosVal[pos], j];
    // Query (src A): a read pulse selecting "emit the memory value here".
    QryInput[pos, 0, pos] := 1.0;
    // Target: the memory value at this position (lives only in src B).
    for j := 0 to cValueDim - 1 do
      Desired[pos, 0, j] := ValueBank[PosVal[pos], j];
  end;
end;

// Drive both source tensors and run the forward pass. For the cross arm the
// memory tensor is copied onto Layers[1] (the side input); both arms read the
// query stream on Layers[0].
procedure Forward(NN: TNNet; MemInput, QryInput: TNNetVolume; CrossArm: boolean);
begin
  NN.Layers[0].Output.Copy(QryInput);
  if CrossArm then NN.Layers[1].Output.Copy(MemInput);
  NN.Compute(NN.Layers[0].Output);
end;

// Mean squared recall error over ALL query positions, plus exact-recall
// accuracy (nearest-neighbour decode over the value bank).
function Evaluate(NN: TNNet; N: integer; out Accuracy: TNeuralFloat;
  CrossArm: boolean): TNeuralFloat;
var
  MemInput, QryInput, Desired: TNNetVolume;
  PosVal: array[0..cNumKeys - 1] of integer;
  seq, pos, ji, ki, bestK, trueKey, correct, total: integer;
  diff, dist, bestDist, mse: TNeuralFloat;
begin
  MemInput := TNNetVolume.Create(cSeqLen, 1, cMemDim);
  QryInput := TNNetVolume.Create(cSeqLen, 1, cQryDim);
  Desired := TNNetVolume.Create(cSeqLen, 1, cValueDim);
  mse := 0; correct := 0; total := 0;
  try
    for seq := 0 to N - 1 do
    begin
      MakeSample(MemInput, QryInput, Desired, PosVal);
      Forward(NN, MemInput, QryInput, CrossArm);
      for pos := 0 to cSeqLen - 1 do
      begin
        // recover true value id from the target value vector.
        trueKey := -1;
        for ki := 0 to cNumVals - 1 do
        begin
          dist := 0;
          for ji := 0 to cValueDim - 1 do
          begin
            diff := Desired[pos, 0, ji] - ValueBank[ki, ji];
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
            diff := NN.GetLastLayer.Output[pos, 0, ji] - ValueBank[ki, ji];
            dist := dist + diff * diff;
          end;
          if dist < bestDist then begin bestDist := dist; bestK := ki; end;
        end;
        for ji := 0 to cValueDim - 1 do
        begin
          diff := NN.GetLastLayer.Output[pos, 0, ji] - Desired[pos, 0, ji];
          mse := mse + diff * diff;
        end;
        if bestK = trueKey then Inc(correct);
        Inc(total);
      end;
    end;
    Result := mse / (total * cValueDim);
    Accuracy := correct / total;
  finally
    MemInput.Free; QryInput.Free; Desired.Free;
  end;
end;

// CrossWKV arm. Two source branches share ONE TNNet; the memory branch is wired
// as a side input via AddLayerAfter(...,0). Layer 0 = query (receptance) input,
// Layer 1 = memory (key|value) input.
function BuildCrossWKV(): TNNet;
var
  QryIn, MemIn, MemProj: TNNetLayer;
begin
  Result := TNNet.Create();
  QryIn := Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cQryDim));
  MemIn := Result.AddLayerAfter(TNNetInput.Create(cSeqLen, 1, cMemDim), 0);
  // Project the memory tokens to the packed k|v width (2*C) and the query
  // tokens to the receptance width (C).
  MemProj := Result.AddLayerAfter(TNNetPointwiseConvLinear.Create(2 * cModelDim), MemIn);
  Result.AddLayerAfter(TNNetPointwiseConvLinear.Create(cModelDim), QryIn);
  // CrossWKV: receptance from the query branch (its PrevLayer), key|value from
  // the projected memory branch.
  Result.AddLayer(TNNetCrossWKV.Create(MemProj));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
end;

// Memory-BLIND single-source baseline: a plain TNNetWKV that only ever sees the
// QUERY stream. It has NO access to the memory tensor, and the value table is
// re-randomised per sequence, so it cannot recall the stored values -- the
// structural contrast the two-source layer is built for.
function BuildBlindWKV(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cQryDim));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(2 * cModelDim)); // k|v split
  Result.AddLayer(TNNetWKV.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cValueDim));
end;

procedure Train(NN: TNNet; Steps: integer; LR: TNeuralFloat; CrossArm: boolean);
var
  MemInput, QryInput, Desired: TNNetVolume;
  PosVal: array[0..cNumKeys - 1] of integer;
  i: integer;
begin
  MemInput := TNNetVolume.Create(cSeqLen, 1, cMemDim);
  QryInput := TNNetVolume.Create(cSeqLen, 1, cQryDim);
  Desired := TNNetVolume.Create(cSeqLen, 1, cValueDim);
  NN.SetLearningRate(LR, 0.9);
  try
    for i := 0 to Steps - 1 do
    begin
      MakeSample(MemInput, QryInput, Desired, PosVal);
      Forward(NN, MemInput, QryInput, CrossArm);
      NN.Backpropagate(Desired);
    end;
  finally
    MemInput.Free; QryInput.Free; Desired.Free;
  end;
end;

var
  CNet, BNet: TNNet;
  cMSE, bMSE, cAcc, bAcc, chance: TNeuralFloat;
begin
  RandSeed := 12345;
  InitValueBank();

  WriteLn('=== CrossWKV: two-source external-memory recall ===');
  WriteLn('keys=', cNumKeys, '  values=', cNumVals, '  value_dim=', cValueDim,
          '  model_dim(C)=', cModelDim, '  seq_len=', cSeqLen);
  WriteLn('memory and query are SEPARATE sequences; key->value table is ',
          're-randomised per sequence.');
  WriteLn;

  CNet := BuildCrossWKV();
  BNet := BuildBlindWKV();
  WriteLn('CrossWKV (two-source)  params = ', CNet.CountWeights());
  WriteLn('blind WKV (query-only) params = ', BNet.CountWeights());
  WriteLn;

  WriteLn('training both arms on the SAME recall stream (', cTrainSteps,
          ' steps each)...');
  RandSeed := 999;
  Train(CNet, cTrainSteps, 0.01, {CrossArm=}true);
  RandSeed := 999;
  Train(BNet, cTrainSteps, 0.01, {CrossArm=}false);
  WriteLn;

  RandSeed := 7;
  cMSE := Evaluate(CNet, cEvalSeqs, cAcc, {CrossArm=}true);
  RandSeed := 7;
  bMSE := Evaluate(BNet, cEvalSeqs, bAcc, {CrossArm=}false);
  chance := 1.0 / cNumVals;

  WriteLn('eval over ', cEvalSeqs, ' held-out cross-recall sequences:');
  WriteLn('  CrossWKV (two-source)  : recall MSE = ', cMSE:0:5,
          '   exact-recall acc = ', (cAcc * 100):0:1, '%');
  WriteLn('  blind WKV (query-only) : recall MSE = ', bMSE:0:5,
          '   exact-recall acc = ', (bAcc * 100):0:1, '%');
  WriteLn('  chance accuracy        : ', (chance * 100):0:1, '%');
  WriteLn;

  if (cAcc > bAcc) and (cAcc > 2 * chance) then
    WriteLn('OK: TNNetCrossWKV recalls values from the cross-built memory; ',
            'the memory-blind single-source WKV cannot.')
  else
    WriteLn('NOTE: expected the two-source arm to dominate the memory-blind arm.');
end.
