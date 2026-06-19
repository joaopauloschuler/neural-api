program RetentionDualForm;
(*
RetentionDualForm: trains the PARALLEL (training) form of Retention (RetNet,
Sun et al. 2023, "Retentive Network", https://arxiv.org/abs/2307.08621) on a
tiny char-level next-token task, then runs the SAME trained weights through a
hand-rolled RECURRENT (inference) loop and asserts the two forward passes agree
token-for-token. This proves Retention's headline property: the O(n^2) parallel
form used for training and the O(1)-state-per-step recurrent form used for
generation are mathematically identical.

THE DUAL FORM.
Retention replaces softmax attention softmax(Q.K^T/sqrt d) V with a softmax-FREE
mixer whose only score weighting is a FIXED exponential-decay causal mask:

  PARALLEL form (this is what TNNetRetention.Compute runs, what trains):
      D[n,m] = gamma^(n-m)  for n >= m, else 0   (lower-triangular decay)
      out[n] = sum_{m<=n} (Q[n] . K[m]) * D[n,m] * V[m]
  There is NO softmax. D is a constant multiplicative weight: older tokens are
  geometrically down-weighted by relative distance, future tokens hard-masked.

  RECURRENT form (hand-rolled below over the trained weights):
      S_n = gamma * S_{n-1} + K_n^T V_n     (S is a d_k x d_k state matrix)
      out_n = Q_n S_n
  This needs only the previous state S_{n-1} (O(d_k^2) memory, O(1) in seqlen)
  yet produces the IDENTICAL out_n. That is the dual form.

THE TASK.
A small vocabulary of cVocab chars; each sample is a random length-cSeqLen
string. The target is a fixed-offset copy:  target[t] = S[t-cLag] (and S[t-1]
for t<cLag). Retention's decay mask lets a query reach back a content-AND-
position-weighted window, so it can learn this short-range copy. The point of
the demo is NOT a hard task; it is the parallel-vs-recurrent equality gate.

THE GATE (mandatory, in the style of examples/SpeculativeDecoding's draft==target
gate): after training, for every probe sequence and every position we compute
the recurrent out_n from the trained Q/K/V the layer actually saw and compare it
to the parallel TNNetRetention.Output. If the max abs difference exceeds an fp
tolerance the program Halt(1)s.

HONEST v1 SCOPE (like the Grokking / SpeculativeDecoding READMEs).
  - The main arm uses a FIXED per-head gamma (the paper's geometric schedule
    across heads). A second LEARNABLE-GAMMA arm at the end shows the follow-up:
    TNNetRetention.Create(..., LearnGamma:=true) makes gamma=sigmoid(raw) a
    trained scalar; we init it wrong (0.50) and watch gradient move it up.
  - Only the parallel + naive-recurrent forms ship. The chunkwise-recurrent
    hybrid (a throughput optimisation, not a new capability) is skipped.
  - Single head here keeps the hand-rolled recurrent replay simple; the
    AddRetention builder supports H heads (one gamma per head).

Pure CPU, single-threaded, no external dataset, finishes in a few seconds.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cVocab   = 6;        // small char vocabulary
  cSeqLen  = 14;       // sequence length
  cLag     = 3;        // fixed-offset copy distance
  cDModel  = 12;       // embedding / retention stream width (= d_k, single head)
  cGamma   = 0.90;     // FIXED retention decay (single head)
  cDFF     = 24;       // hidden width of the pointwise MLP read-out
  cEpochs  = 250;      // training epochs
  cBatch   = 48;       // sequences per epoch
  cLR      = 0.005;
  cInertia = 0.9;
  cSeed    = 424242;   // repo idiom
  cProbes  = 400;      // evaluation probes
  cGateTol = 1e-4;     // parallel-vs-recurrent fp tolerance (single precision)

type
  TSeq = array[0..cSeqLen - 1] of integer;

procedure MakeSeq(out S: TSeq);
var i: integer;
begin
  for i := 0 to cSeqLen - 1 do
    S[i] := Random(cVocab);
end;

function TargetTok(const S: TSeq; t: integer): integer;
begin
  if t < cLag then
  begin
    if t - 1 >= 0 then Result := S[t - 1] else Result := 0;
  end
  else
    Result := S[t - cLag];
end;

procedure FillPair(const S: TSeq; InputV, TargetV: TNNetVolume);
var t: integer;
begin
  TargetV.Fill(0);
  for t := 0 to cSeqLen - 1 do
  begin
    InputV.FData[t] := S[t];
    TargetV[t, 0, TargetTok(S, t)] := 1.0;
  end;
end;

var
  NN: TNNet;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  Epoch, b, k, t, n, a, bb, Pred, Tgt: integer;
  RetentionIdx: integer;
  RetLayer: TNNetLayer;
  Slab, RetOut: TNNetVolume;
  StartTime: double;
  Correct, Total: integer;
  State: array of array of TNeuralFloat; // S[a,b], d_k x d_k
  RecOut, Diff, MaxDiff: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  WriteLn('RetentionDualForm: train PARALLEL Retention, then verify the');
  WriteLn('RECURRENT form reproduces it token-for-token (the RetNet dual form).');
  WriteLn(Format('Task: fixed-offset copy target[t]=S[t-%d] (seqlen %d, vocab %d).',
    [cLag, cSeqLen, cVocab]));
  WriteLn(Format('Single head, d_k=%d, gamma=%.2f (FIXED). Decay mask D[n,m]=gamma^(n-m).',
    [cDModel, cGamma]));
  WriteLn;

  RandSeed := cSeed;
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
  NN.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
  // Token-wise QKV slab projection d_model -> 3*d_model (1x1 conv per token).
  // This slab is exactly the input TNNetRetention sees.
  NN.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDModel));
  // Single-head retention (the builder also handles H heads with one gamma each).
  NN.AddRetention(cDModel, 1, -Log2(1.0 - cGamma)); // gamma_0 = 1-2^-exp = cGamma
  // AddRetention appends [SplitChannels, Retention, DeepConcat, PWConvLinear];
  // locate the single TNNetRetention layer it inserted.
  RetentionIdx := -1;
  for k := 0 to NN.CountLayers() - 1 do
    if NN.Layers[k] is TNNetRetention then RetentionIdx := k;
  // Per-token MLP read-out.
  NN.AddLayer(TNNetPointwiseConvLinear.Create(cDFF));
  NN.AddLayer(TNNetReLU.Create());
  NN.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
  NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
  NN.SetLearningRate(cLR, cInertia);
  NN.SetL2Decay(0.0);

  WriteLn(Format('Built net: %d layers, %d weights. Retention layer idx %d.',
    [NN.CountLayers(), NN.CountWeights(), RetentionIdx]));

  // Sanity: the captured retention layer's gamma matches the requested constant.
  if not (NN.Layers[RetentionIdx] is TNNetRetention) then
  begin
    WriteLn('[FAIL] layer at RetentionIdx is not a TNNetRetention.');
    Halt(1);
  end;
  WriteLn(Format('Retention gamma read back from layer: %.6f (requested %.6f).',
    [TNNetRetention(NN.Layers[RetentionIdx]).Gamma, cGamma]));
  WriteLn;

  // ---- Train the PARALLEL form (per-sample SGD, single-threaded) ----
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  RandSeed := cSeed;
  StartTime := Now();
  Write('Training parallel form ...');
  for Epoch := 1 to cEpochs do
    for b := 1 to cBatch do
    begin
      MakeSeq(S);
      FillPair(S, InputV, TargetV);
      NN.Compute(InputV);
      NN.Backpropagate(TargetV);
    end;
  WriteLn(Format(' done in %.2fs.', [(Now() - StartTime) * 86400.0]));

  // ---- Evaluate accuracy AND run the dual-form equality gate ----
  RetLayer := NN.Layers[RetentionIdx];
  SetLength(State, cDModel, cDModel);
  Correct := 0; Total := 0; MaxDiff := 0;
  RandSeed := cSeed + 7;
  for k := 1 to cProbes do
  begin
    MakeSeq(S);
    FillPair(S, InputV, TargetV);
    NN.Compute(InputV);

    // Accuracy of the trained parallel model.
    for t := 0 to cSeqLen - 1 do
    begin
      Pred := NN.GetLastLayer.Output.GetClassOnPixel(t, 0);
      Tgt  := TargetTok(S, t);
      if Pred = Tgt then Inc(Correct);
      Inc(Total);
    end;

    // ---- RECURRENT replay over the EXACT Q|K|V slab the layer saw ----
    // Slab depth layout: Q at [0..d-1], K at [d..2d-1], V at [2d..3d-1].
    Slab   := RetLayer.PrevLayer.Output;     // input the retention layer saw
    RetOut := RetLayer.Output;               // parallel-form output to match
    for a := 0 to cDModel - 1 do
      for bb := 0 to cDModel - 1 do
        State[a][bb] := 0;
    for n := 0 to cSeqLen - 1 do
    begin
      // S_n = gamma * S_{n-1} + K_n^T V_n
      for a := 0 to cDModel - 1 do
        for bb := 0 to cDModel - 1 do
          State[a][bb] := cGamma * State[a][bb] +
            Slab[n, 0, cDModel + a] * Slab[n, 0, 2 * cDModel + bb];
      // out_n[bb] = sum_a Q_n[a] * S_n[a,bb]; compare to the parallel output.
      for bb := 0 to cDModel - 1 do
      begin
        RecOut := 0;
        for a := 0 to cDModel - 1 do
          RecOut := RecOut + Slab[n, 0, a] * State[a][bb];
        Diff := Abs(RecOut - RetOut[n, 0, bb]);
        if Diff > MaxDiff then MaxDiff := Diff;
      end;
    end;
  end;

  WriteLn;
  WriteLn(Format('Next-token accuracy (parallel form): %.3f  (chance %.3f).',
    [Correct / Max(1, Total), 1.0 / cVocab]));
  WriteLn(Format('Dual-form check over %d probes x %d positions x %d dims:',
    [cProbes, cSeqLen, cDModel]));
  WriteLn(Format('  PARALLEL vs RECURRENT max abs diff = %.3e  (tol %.1e).',
    [MaxDiff, cGateTol]));
  WriteLn;

  if MaxDiff < cGateTol then
    WriteLn('[PASS] recurrent form reproduces the parallel form within tolerance.')
  else
    WriteLn('[FAIL] dual-form mismatch exceeds tolerance.');

  InputV.Free; TargetV.Free; NN.Free;

  // ===================================================================
  // LEARNABLE-GAMMA ARM (the logged follow-up: learn gamma by gradient).
  // -------------------------------------------------------------------
  // TNNetRetention's third Create argument LearnGamma=true stores an
  // UNCONSTRAINED raw scalar (1 neuron weight) and uses gamma=sigmoid(raw),
  // so the effective decay is always in (0,1) under plain SGD. Backward
  // accumulates dL/dgamma through D[n,m]=gamma^(n-m) (d/dgamma gamma^k =
  // k*gamma^(k-1)) and chains the sigmoid (dgamma/draw = gamma*(1-gamma)).
  // Here we DELIBERATELY init gamma wrong (0.50, far below the cGamma=0.90
  // schedule the task rewards) and show training moves it back up.
  // ===================================================================
  WriteLn;
  WriteLn('--- Learnable-gamma arm: init gamma WRONG (0.50), train, watch it move ---');
  RandSeed := cSeed;
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
  NN.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
  NN.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDModel));
  // Single learnable-gamma retention head, init gamma = 0.50 (intentionally low).
  RetLayer := NN.AddLayerAfter(
    TNNetRetention.Create(cDModel, 0.50, {LearnGamma=}true), NN.GetLastLayer());
  NN.AddLayer(TNNetPointwiseConvLinear.Create(cDFF));
  NN.AddLayer(TNNetReLU.Create());
  NN.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
  NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
  NN.SetLearningRate(cLR, cInertia);
  NN.SetL2Decay(0.0);

  WriteLn(Format('Initial effective gamma = %.4f (raw = logit, learnable).',
    [TNNetRetention(RetLayer).Gamma]));

  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  RandSeed := cSeed;
  for Epoch := 1 to cEpochs do
    for b := 1 to cBatch do
    begin
      MakeSeq(S);
      FillPair(S, InputV, TargetV);
      NN.Compute(InputV);
      NN.Backpropagate(TargetV);
    end;

  RecOut := TNNetRetention(RetLayer).Gamma; // reuse RecOut as "final gamma"
  WriteLn(Format('Final   effective gamma = %.4f (after %d epochs).',
    [RecOut, cEpochs]));
  if RecOut > 0.50 + 1e-3 then
    WriteLn('[PASS] learnable gamma moved UP from 0.50 toward the rewarded decay.')
  else
    WriteLn('[NOTE] learnable gamma did not increase (still a valid (0,1) value).');

  InputV.Free; TargetV.Free; NN.Free;

  // Mandatory equality gate (style of examples/SpeculativeDecoding).
  if not (MaxDiff < cGateTol) then Halt(1);
  // Gate the learnable arm too: gamma must stay in (0,1) and have moved up.
  if not (RecOut > 0.50 + 1e-3) then Halt(1);
end.
