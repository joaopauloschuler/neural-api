program CausalMaskSanity;
(*
CausalMaskSanity: a controlled demonstration that a self-attention next-token
model trained WITHOUT a causal mask "cheats" -- it peeks at FUTURE tokens to
drive its TRAIN loss to near-zero, yet is useless for honest left-to-right
generation. The SAME model trained WITH a strictly-causal mask is forced to
predict token t using only tokens <= t, so its train loss is honestly higher
but it actually generalizes to autoregressive next-token prediction.

THE TASK (constructed so peeking ahead is both helpful AND non-generalizing).
A sequence over a small vocabulary V follows a 2nd-order recurrence:
    tok[0], tok[1] ~ uniform(0..V-1)
    tok[t]         = (tok[t-1] + 2*tok[t-2] + 1) mod V        (t >= 2)
The model is trained on NEXT-token prediction: at every position t it must
predict tok[t+1]. Crucially the correct answer at position t, namely
(tok[t] + 2*tok[t-1] + 1) mod V, is a function ONLY of positions <= t. So:
  * A CAUSAL model (query t may attend only to keys <= t) CAN learn the rule,
    but it must COMBINE TWO past tokens (current and previous) -- a genuinely
    non-trivial computation. It generalizes to real left-to-right generation.
  * A NON-CAUSAL model has a far EASIER, illegitimate shortcut: the target at
    position t is literally tok[t+1], which is sitting RIGHT THERE at input
    position t+1. The unmasked attention head just routes query t to key t+1
    and COPIES it -- one hop, no arithmetic. Train loss collapses to ~0 -- but
    this is pure cheating: at generation time tok[t+1] does not exist yet, so
    the shortcut evaporates and accuracy craters.

TWO ARMS, identical in every respect (architecture, seed, LR, epochs, batch,
data) EXCEPT the causal mask:
  Input(SeqLen) -> Embedding(V,d) -> SinusoidalPositionalEmbedding
    -> PointwiseConvLinear(3*d_k)        { pack Q | K | V along depth }
    -> ScaledDotProductAttention(d_k, CAUSAL?)   { the ONLY difference }
    -> PointwiseConvLinear(V)            { per-position vocab logits }
    -> PointwiseSoftMax(1)               { softmax across depth }

THE MASK. TNNetScaledDotProductAttention's CausalMask flag applies, before the
row-softmax, exactly the strictly-upper-triangular additive fill (score[j>i] :=
-1e9) that the standalone layers TNNetMaskedFill / TNNetTriangularCausalMask
apply to an explicit (key,query) score matrix -- same upper triangle, same
-1e9 constant. We use the built-in flag because SDPA computes its scores
internally (there is no exposed score tensor to slot a separate TNNetMaskedFill
in front of); the masking OPERATION is identical.

HEADLINE / GATES (Halt(1) on failure, suite idiom):
  1. CHEATING: the UNMASKED arm reaches a strictly LOWER train cross-entropy
     than the masked arm (it exploits the future-token shortcut).
  2. GENERALIZATION: under TRUE autoregressive generation (feed only the past;
     unknown future positions are blanked so the model literally cannot peek),
     the MASKED arm's next-token accuracy is strictly HIGHER than the unmasked
     arm's. The cheat does not survive honest generation.
  3. EVIDENCE (where the cheat lives): averaged over probe sequences, the
     UNMASKED head places substantial attention weight on FUTURE key positions
     (j > i), while the MASKED head places ~0 there (read straight off the
     read-only AttentionWeights map, layout X=key j, Y=query i).

Pure CPU, single-threaded, deterministic (fixed RandSeed). Runs in well under
two minutes. Mirrors the gate/Halt idiom of examples/SIREN and examples/DeepSets.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cVocab   = 6;       // small vocabulary
  cSeqLen  = 7;       // short sequence
  cDModel  = 24;      // embedding width
  cDk      = 24;      // attention head width (= d_model here)
  cEpochs  = 160;     // shared epoch budget for both arms
  cNumTrain= 256;     // fixed synthetic training set (shared by both arms)
  cLR      = 0.005;   // per-sample SGD (same idiom/LR as AttentionCopyTask)
  cInertia = 0.9;
  cSeed    = 424242;  // repo idiom

type
  TSeq = array[0..cSeqLen - 1] of integer;

// Generate one sequence from a 2nd-order recurrence:
//   tok[0], tok[1] ~ uniform(0..V-1)
//   tok[t]         = (tok[t-1] + 2*tok[t-2] + 1) mod V     (t >= 2)
// The next token at position t, tok[t+1], is therefore a deterministic
// function of (tok[t], tok[t-1]) -- it depends ONLY on positions <= t, so a
// causal model CAN learn it; but it must COMBINE TWO past tokens (current and
// previous), which is genuinely harder than the unmasked shortcut of copying
// the literal answer sitting one slot ahead in the input.
procedure MakeSeq(out S: TSeq);
var
  t: integer;
begin
  S[0] := Random(cVocab);
  S[1] := Random(cVocab);
  for t := 2 to cSeqLen - 1 do
    S[t] := (S[t - 1] + 2 * S[t - 2] + 1) mod cVocab;
end;

// The "true next token" at position t under the recurrence. For t<SeqLen-1
// this is exactly S[t+1]; at the last position it is the valid continuation
// (S[t]+2*S[t-1]+1) mod V. Supervising EVERY position (no all-zero target row)
// keeps a clean one-hot target everywhere so the softmax+CE stays well-posed.
function NextTok(const S: TSeq; t: integer): integer;
begin
  if t < cSeqLen - 1 then Result := S[t + 1]
  else if t >= 1 then Result := (S[t] + 2 * S[t - 1] + 1) mod cVocab
  else Result := S[t];
end;

// Lay a sequence onto the input (token IDs along X) and build the next-token
// target: target[t] = NextTok(t) (one-hot over the depth axis).
procedure FillPair(const S: TSeq; InputV, TargetV: TNNetVolume);
var
  t: integer;
begin
  TargetV.Fill(0);
  for t := 0 to cSeqLen - 1 do
  begin
    InputV.FData[t] := S[t];
    TargetV[t, 0, NextTok(S, t)] := 1.0;
  end;
end;

function BuildModel(Causal: boolean): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
  Result.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDk));     // Q | K | V slab
  // The ONLY difference between the two arms: the causal mask flag. When True,
  // SDPA fills score[j>i] := -1e9 before the row-softmax (the same strictly
  // upper-triangular mask as TNNetMaskedFill / TNNetTriangularCausalMask).
  Result.AddLayer(TNNetScaledDotProductAttention.Create(cDk, Causal));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));       // per-pos logits
  Result.AddLayer(TNNetPointwiseSoftMax.Create(1));               // softmax/depth
  Result.SetLearningRate(cLR, cInertia);
  Result.SetL2Decay(0.0);
  // Per-sample SGD: Backpropagate() applies the update immediately (library
  // default, FBatchUpdate=False), exactly as examples/AttentionCopyTask does.
end;

// Mean per-position cross-entropy over all SeqLen supervised positions.
function CrossEntropy(Output, Target: TNNetVolume): TNeuralFloat;
var
  i: integer;
  P: TNeuralFloat;
begin
  Result := 0;
  for i := 0 to Output.Size - 1 do
    if Target.FData[i] > 0 then
    begin
      P := Output.FData[i];
      if P < 1e-12 then P := 1e-12;
      Result := Result - Target.FData[i] * Ln(P);
    end;
  Result := Result / cSeqLen;
end;

function ArgMaxDepth(V: TNNetVolume; Pos: integer): integer;
var
  d, Best: integer;
  BestVal, Cur: TNeuralFloat;
begin
  Best := 0;
  BestVal := V[Pos, 0, 0];
  for d := 1 to cVocab - 1 do
  begin
    Cur := V[Pos, 0, d];
    if Cur > BestVal then begin BestVal := Cur; Best := d; end;
  end;
  Result := Best;
end;

// Train one arm on the shared training set; return its final mean train CE.
function TrainArm(NN: TNNet; const TrainSeqs: array of TSeq;
  const Tag: string): TNeuralFloat;
var
  Epoch, i: integer;
  InputV, TargetV: TNNetVolume;
  SumCE: TNeuralFloat;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  Result := 0;
  try
    for Epoch := 1 to cEpochs do
    begin
      SumCE := 0;
      for i := 0 to High(TrainSeqs) do
      begin
        FillPair(TrainSeqs[i], InputV, TargetV);
        NN.Compute(InputV);
        SumCE := SumCE + CrossEntropy(NN.GetLastLayer.Output, TargetV);
        NN.Backpropagate(TargetV);   // per-sample weight update (auto)
      end;
      Result := SumCE / Length(TrainSeqs);
      if (Epoch = 1) or (Epoch mod 40 = 0) or (Epoch = cEpochs) then
        WriteLn(Format('  [%-8s] epoch %3d  mean train-CE = %.6f',
          [Tag, Epoch, Result]));
    end;
  finally
    InputV.Free;
    TargetV.Free;
  end;
end;

// "Teacher-forced" train-style next-token accuracy: the FULL sequence is on
// the input, so a non-causal model can see future positions. This is the
// metric the unmasked arm games.
function TeacherForcedAcc(NN: TNNet; const Seqs: array of TSeq): TNeuralFloat;
var
  i, t, Pred, Correct, Total: integer;
  InputV, TargetV: TNNetVolume;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  Correct := 0; Total := 0;
  try
    for i := 0 to High(Seqs) do
    begin
      FillPair(Seqs[i], InputV, TargetV);
      NN.Compute(InputV);
      for t := 0 to cSeqLen - 2 do
      begin
        Pred := ArgMaxDepth(NN.GetLastLayer.Output, t);
        if Pred = Seqs[i][t + 1] then Inc(Correct);
        Inc(Total);
      end;
    end;
  finally
    InputV.Free; TargetV.Free;
  end;
  if Total = 0 then Result := 0 else Result := Correct / Total;
end;

// TRUE autoregressive next-token accuracy. For each target position t we feed
// ONLY the past: positions 0..t hold the real tokens, every FUTURE position
// (>t) is blanked to a fixed sentinel (token 0) so the model literally cannot
// read tok[t+1]. We then read the prediction at position t. This is honest
// left-to-right generation; the unmasked arm's "copy the future" trick dies.
function CausalGenAcc(NN: TNNet; const Seqs: array of TSeq): TNeuralFloat;
var
  i, t, j, Pred, Correct, Total: integer;
  InputV, TargetV: TNNetVolume;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  Correct := 0; Total := 0;
  try
    for i := 0 to High(Seqs) do
      for t := 0 to cSeqLen - 2 do
      begin
        // Build a context that reveals only positions 0..t.
        for j := 0 to cSeqLen - 1 do
          if j <= t then InputV.FData[j] := Seqs[i][j]
          else InputV.FData[j] := 0;          // future blanked (sentinel)
        NN.Compute(InputV);
        Pred := ArgMaxDepth(NN.GetLastLayer.Output, t);
        if Pred = Seqs[i][t + 1] then Inc(Correct);
        Inc(Total);
      end;
  finally
    InputV.Free; TargetV.Free;
  end;
  if Total = 0 then Result := 0 else Result := Correct / Total;
end;

// Average fraction of attention mass a head puts on FUTURE keys (j>i),
// averaged over supervised query rows i and probe sequences. Reads the
// read-only AttentionWeights map (X=key j, Y=query i, rows sum to 1).
function MeanFutureAttention(NN: TNNet; AttnIdx: integer;
  const Seqs: array of TSeq): TNeuralFloat;
var
  i, t, j: integer;
  InputV, TargetV: TNNetVolume;
  Attn: TNNetVolume;
  RowFuture, Sum: TNeuralFloat;
  Rows: integer;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  Sum := 0; Rows := 0;
  try
    for i := 0 to High(Seqs) do
    begin
      FillPair(Seqs[i], InputV, TargetV);
      NN.Compute(InputV);
      Attn := TNNetScaledDotProductAttention(NN.Layers[AttnIdx]).AttentionWeights;
      // Query rows 0..SeqLen-2 are the supervised (next-token) rows.
      for t := 0 to cSeqLen - 2 do
      begin
        RowFuture := 0;
        for j := t + 1 to cSeqLen - 1 do
          RowFuture := RowFuture + Attn[j, t, 0];
        Sum := Sum + RowFuture;
        Inc(Rows);
      end;
    end;
  finally
    InputV.Free; TargetV.Free;
  end;
  if Rows = 0 then Result := 0 else Result := Sum / Rows;
end;

var
  NNMasked, NNUnmasked: TNNet;
  TrainSeqs: array[0..cNumTrain - 1] of TSeq;
  EvalSeqs:  array[0..199] of TSeq;
  i, AttnIdx: integer;
  CEMasked, CEUnmasked: TNeuralFloat;
  TFMasked, TFUnmasked: TNeuralFloat;
  GenMasked, GenUnmasked: TNeuralFloat;
  FutMasked, FutUnmasked: TNeuralFloat;
  Pass: boolean;
begin
  // Fast-math kernels can transiently underflow during early training; mask
  // FPU exceptions so the host doesn't abort (matches AttentionCopyTask).
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := cSeed;
  // Manual Compute/Backpropagate runs single-threaded by construction (as in
  // examples/SIREN and examples/DeepSets), so no thread-count knob is needed.

  WriteLn('CausalMaskSanity: masked vs unmasked next-token self-attention.');
  WriteLn(Format('Task: 2nd-order recurrence  tok[t] = (tok[t-1]+2*tok[t-2]+1)'
    + ' mod %d  (vocab %d, seqlen %d).', [cVocab, cVocab, cSeqLen]));
  WriteLn('Both arms identical except the SDPA causal-mask flag '
    + '(score[j>i] := -1e9, the TNNetMaskedFill upper-triangle mask).');
  WriteLn;

  // Shared, fixed datasets so both arms see EXACTLY the same data.
  for i := 0 to cNumTrain - 1 do MakeSeq(TrainSeqs[i]);
  for i := 0 to High(EvalSeqs) do MakeSeq(EvalSeqs[i]);

  // ---- UNMASKED arm (may peek at future tokens) ----------------------------
  NNUnmasked := BuildModel({Causal=}False);
  NNUnmasked.InitWeights();
  WriteLn('Training UNMASKED arm (non-causal: query t may attend to ANY key)...');
  CEUnmasked := TrainArm(NNUnmasked, TrainSeqs, 'unmasked');
  WriteLn;

  // ---- MASKED arm (strictly causal) ----------------------------------------
  // Reseed so both arms get the SAME weight initialization stream.
  RandSeed := cSeed;
  NNMasked := BuildModel({Causal=}True);
  NNMasked.InitWeights();
  WriteLn('Training MASKED arm (causal: query t may attend only to keys <= t)...');
  CEMasked := TrainArm(NNMasked, TrainSeqs, 'masked');
  WriteLn;

  // Index of the SDPA layer (same in both models): Input,Emb,PosEmb,QKV,SDPA.
  AttnIdx := 4;

  // ---- metrics --------------------------------------------------------------
  TFUnmasked  := TeacherForcedAcc(NNUnmasked, EvalSeqs);
  TFMasked    := TeacherForcedAcc(NNMasked,   EvalSeqs);
  GenUnmasked := CausalGenAcc(NNUnmasked, EvalSeqs);
  GenMasked   := CausalGenAcc(NNMasked,   EvalSeqs);
  FutUnmasked := MeanFutureAttention(NNUnmasked, AttnIdx, EvalSeqs);
  FutMasked   := MeanFutureAttention(NNMasked,   AttnIdx, EvalSeqs);

  WriteLn(StringOfChar('=', 70));
  WriteLn('RESULTS');
  WriteLn(StringOfChar('=', 70));
  WriteLn('Final train cross-entropy (lower = "fits" the train set better):');
  WriteLn(Format('  UNMASKED : %.6f   <-- cheats, drives loss lower', [CEUnmasked]));
  WriteLn(Format('  MASKED   : %.6f   <-- honestly higher', [CEMasked]));
  WriteLn;
  WriteLn('Teacher-forced next-token accuracy (FULL sequence visible -- the');
  WriteLn('metric the unmasked arm games by reading tok[t+1] off the input):');
  WriteLn(Format('  UNMASKED : %6.2f%%', [100.0 * TFUnmasked]));
  WriteLn(Format('  MASKED   : %6.2f%%', [100.0 * TFMasked]));
  WriteLn;
  WriteLn('TRUE autoregressive generation accuracy (future positions blanked,');
  WriteLn('model sees only the past) -- the honest test:');
  WriteLn(Format('  UNMASKED : %6.2f%%   <-- the cheat evaporates', [100.0 * GenUnmasked]));
  WriteLn(Format('  MASKED   : %6.2f%%   <-- generalizes to generation', [100.0 * GenMasked]));
  WriteLn;
  WriteLn('Mean attention mass on FUTURE keys (j>i) over supervised query rows:');
  WriteLn(Format('  UNMASKED : %.4f   <-- the head literally looks ahead', [FutUnmasked]));
  WriteLn(Format('  MASKED   : %.4f   <-- masked out (~0)', [FutMasked]));
  WriteLn(StringOfChar('=', 70));
  WriteLn;

  // ---- SELF-CHECKING GATES --------------------------------------------------
  Pass :=
    (CEUnmasked < CEMasked) and                 // 1. cheating: lower train loss
    (GenMasked  > GenUnmasked) and              // 2. masked wins honest generation
    (FutUnmasked > 0.10) and (FutMasked < 0.01); // 3. evidence: where the cheat lives
  if Pass then
  begin
    WriteLn('GATE: PASS');
    WriteLn('  - UNMASKED reached a LOWER train loss by peeking at future tokens (cheating).');
    WriteLn('  - MASKED wins TRUE autoregressive generation (it actually generalizes).');
    WriteLn('  - UNMASKED attention puts real weight on future keys; MASKED puts ~0.');
  end
  else
  begin
    WriteLn('GATE: FAIL');
    if not (CEUnmasked < CEMasked) then
      WriteLn(Format('  - expected unmasked train CE %.6f < masked %.6f (no cheat seen).',
        [CEUnmasked, CEMasked]));
    if not (GenMasked > GenUnmasked) then
      WriteLn(Format('  - expected masked gen acc %.2f%% > unmasked %.2f%%.',
        [100.0 * GenMasked, 100.0 * GenUnmasked]));
    if not (FutUnmasked > 0.10) then
      WriteLn(Format('  - expected unmasked future-attention %.4f > 0.10.', [FutUnmasked]));
    if not (FutMasked < 0.01) then
      WriteLn(Format('  - expected masked future-attention %.4f < 0.01.', [FutMasked]));
  end;

  NNMasked.Free;
  NNUnmasked.Free;

  if not Pass then Halt(1);
end.
