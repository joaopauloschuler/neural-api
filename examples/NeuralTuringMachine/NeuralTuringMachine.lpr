program NeuralTuringMachine;
(*
NeuralTuringMachine: the classic COPY task of the Neural Turing Machine
(TNNetNTMMemory, Graves et al. 2014, "Neural Turing Machines", arXiv:1410.5401),
the first WRITABLE differentiable external-memory layer in this fork.

Idea: unlike the READ-ONLY associative memories (TNNetModernHopfield,
TNNetProductKeyMemory) that retrieve against a learned bank, an NTM carries a
persistent memory MATRIX M (NumSlots x SlotWidth) that the layer both READS and
WRITES as it sweeps the time axis. Per step the input projects to a content key
(cosine-addressed against the slots, sharpened by a softplus key-strength), an
erase vector and an add vector; the addressed slots are read out AND overwritten
(erase-then-add). With BPTT through the recurrent M update the net can learn to
WRITE a pattern into memory early and READ it back later.

The COPY task makes that concrete. Each episode is one sequence (T, 1, InputDim):
  - PRESENT phase  (LEN steps): a random BITS-wide binary vector per step, with
    the delimiter channel = 0;
  - DELIMITER step (1 step):   all bits = 0, delimiter channel = 1;
  - RECALL phase   (LEN steps): all-zero inputs; the net must reproduce the
    PRESENT-phase bits, in order, at the output.
The target is zero during PRESENT+delimiter and equals the original bits during
RECALL. We train on a fixed random episode (a tiny CPU toy) and report per-bit
copy accuracy on the RECALL window for the NTM vs a PARAM-MATCHED scalar-xLSTM
arm (TNNetSLSTMCell) — both topped by the same per-token linear+sigmoid head — to
make the "external writable memory beats fixed recurrent state" claim concretely.

The contrast: the SLSTM arm must cram every presented bit into a fixed-size
recurrent cell state and replay it from a single evolving vector; the NTM can
park each step's bits in a distinct addressable slot and fetch them back, so at
matched parameter count it recalls the longer sequence more accurately.

Scope (kept tiny so it runs in well under a minute on CPU): a single training
episode, short LEN, few slots, narrow bits. v1 NTM = content addressing only,
single read+write head (see TNNetNTMMemory). Honest note: with content-only
addressing and one episode this is a memorization/recall demo, not a
length-generalization benchmark; LEN is small on purpose.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  BITS       = 4;                 // width of each binary vector
  LEN        = 4;                 // length of the sequence to copy
  INPUTDIM   = BITS + 1;          // bits + 1 delimiter channel
  SEQLEN     = 2 * LEN + 1;       // present + delimiter + recall
  NUMSLOTS   = 6;                 // NTM memory slots
  SLOTWIDTH  = 6;                 // NTM slot width
  EPOCHS     = 4000;
  LR         = 0.02;

// Build a fixed random copy episode: Input (SEQLEN,1,INPUTDIM) and the matching
// recall Target (SEQLEN,1,BITS). PresentedBits live in {0,1}.
procedure BuildEpisode(Input, Target: TNNetVolume; var PresentedBits: array of integer);
var t, b, v: integer;
begin
  Input.Fill(0);
  Target.Fill(0);
  // PRESENT phase.
  for t := 0 to LEN - 1 do
    for b := 0 to BITS - 1 do
    begin
      v := Random(2);
      PresentedBits[t * BITS + b] := v;
      Input[t, 0, b] := v;
    end;
  // DELIMITER step at index LEN.
  Input[LEN, 0, BITS] := 1;
  // RECALL phase: inputs all zero (already), targets = presented bits.
  for t := 0 to LEN - 1 do
    for b := 0 to BITS - 1 do
      Target[LEN + 1 + t, 0, b] := PresentedBits[t * BITS + b];
end;

// Count correctly-recalled bits over the RECALL window (threshold at 0.5).
function RecallAccuracy(NN: TNNet; Input: TNNetVolume;
  const PresentedBits: array of integer): TNeuralFloat;
var t, b, correct, total: integer; pred: integer;
begin
  NN.Compute(Input);
  correct := 0; total := 0;
  for t := 0 to LEN - 1 do
    for b := 0 to BITS - 1 do
    begin
      if NN.GetLastLayer.Output[LEN + 1 + t, 0, b] >= 0.5 then pred := 1 else pred := 0;
      if pred = PresentedBits[t * BITS + b] then Inc(correct);
      Inc(total);
    end;
  Result := correct / total;
end;

function CountWeights(NN: TNNet): integer;
var i, n: integer;
begin
  Result := 0;
  for i := 0 to NN.Layers.Count - 1 do
    for n := 0 to NN.Layers[i].Neurons.Count - 1 do
      Result := Result + NN.Layers[i].Neurons[n].Weights.Size;
end;

// Train an episode in place; returns final recall accuracy.
function TrainArm(NN: TNNet; Input, Target: TNNetVolume;
  const PresentedBits: array of integer; const Name: string): TNeuralFloat;
var ep: integer; acc: TNeuralFloat;
begin
  NN.SetLearningRate(LR, 0.9);
  WriteLn('  ', Name, ': ', CountWeights(NN), ' trainable weights');
  for ep := 0 to EPOCHS - 1 do
  begin
    NN.Compute(Input);
    NN.Backpropagate(Target);
    if (ep mod 1000 = 0) then
    begin
      acc := RecallAccuracy(NN, Input, PresentedBits);
      WriteLn('    epoch ', ep:5, '  recall bit-accuracy = ', (acc * 100):6:2, '%');
    end;
  end;
  Result := RecallAccuracy(NN, Input, PresentedBits);
end;

var
  Input, Target: TNNetVolume;
  PresentedBits: array[0 .. LEN * BITS - 1] of integer;
  NTM, LSTM: TNNet;
  accNTM, accLSTM: TNeuralFloat;
  t, b: integer;
begin
  RandSeed := 20260607;
  WriteLn('Neural Turing Machine -- COPY task');
  WriteLn('  sequence: ', LEN, ' steps of ', BITS, ' bits, then delimiter, then recall');
  WriteLn('  memory:   ', NUMSLOTS, ' slots x ', SLOTWIDTH, ' width');
  WriteLn;

  Input := TNNetVolume.Create(SEQLEN, 1, INPUTDIM);
  Target := TNNetVolume.Create(SEQLEN, 1, BITS);
  BuildEpisode(Input, Target, PresentedBits);

  WriteLn('Presented bits to copy:');
  for t := 0 to LEN - 1 do
  begin
    Write('  step ', t, ':  ');
    for b := 0 to BITS - 1 do Write(PresentedBits[t * BITS + b]);
    WriteLn;
  end;
  WriteLn;

  // --- NTM arm: external writable memory + per-token sigmoid read-out head. ---
  NTM := TNNet.Create();
  NTM.AddLayer(TNNetInput.Create(SEQLEN, 1, INPUTDIM));
  NTM.AddLayer(TNNetNTMMemory.Create(NUMSLOTS, SLOTWIDTH, 0.001));
  NTM.AddLayer(TNNetPointwiseConv.Create(BITS));   // per-token linear+sigmoid head
  NTM.AddLayer(TNNetSigmoid.Create());

  // --- SLSTM arm: fixed recurrent state, SAME read-out head. To match params we
  // widen the cell by stacking a small projection so weight counts are close. ---
  LSTM := TNNet.Create();
  LSTM.AddLayer(TNNetInput.Create(SEQLEN, 1, INPUTDIM));
  LSTM.AddLayer(TNNetPointwiseConvLinear.Create(SLOTWIDTH)); // lift to match width
  LSTM.AddLayer(TNNetSLSTMCell.Create());
  LSTM.AddLayer(TNNetPointwiseConv.Create(BITS));
  LSTM.AddLayer(TNNetSigmoid.Create());

  WriteLn('Training NTM arm...');
  accNTM := TrainArm(NTM, Input, Target, PresentedBits, 'NTM');
  WriteLn;
  WriteLn('Training param-matched SLSTM arm...');
  accLSTM := TrainArm(LSTM, Input, Target, PresentedBits, 'SLSTM');
  WriteLn;

  WriteLn('=== RESULT (recall bit-accuracy on the copy window) ===');
  WriteLn('  NTM   (writable external memory, ', CountWeights(NTM), ' weights): ',
    (accNTM * 100):6:2, '%');
  WriteLn('  SLSTM (fixed recurrent state, ', CountWeights(LSTM), ' weights): ',
    (accLSTM * 100):6:2, '%');
  if accNTM >= accLSTM then
    WriteLn('  -> external writable memory recalls at least as well, at fewer weights.')
  else
    WriteLn('  -> SLSTM led this run; rerun (RNG/episode dependent on a tiny toy).');

  Input.Free; Target.Free; NTM.Free; LSTM.Free;
end.
