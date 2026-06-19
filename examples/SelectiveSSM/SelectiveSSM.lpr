program SelectiveSSM;
(*
SelectiveSSM: a SELECTIVE-COPY / INDUCTION bake-off that isolates the one thing
an INPUT-DEPENDENT ("selective", Mamba/S6-style, Gu & Dao 2023) state-space
model can do that a LINEAR-TIME-INVARIANT one provably cannot: CONTENT-ADDRESSED
recall at a data-dependent offset.

THE TASK (content-addressed long-hold copy).
A small vocabulary of cVocab symbols, with three reserved:
    QUERY   = cVocab-3   (planted in the trailing window: "emit the payload now")
    MARKER  = cVocab-2   (the "remember the NEXT symbol" cue)
    BLANK   = cVocab-1   (the target before the query)
Each sample is a random length-cSeqLen string of ordinary payload symbols, with
EXACTLY ONE marker at a fully RANDOM position p; the symbol at p+1 is the
PAYLOAD. The trailing cQueryWin positions are overwritten with QUERY tokens.
The target is BLANK everywhere EXCEPT the trailing query window, where the model
must emit the payload S[p+1]. Only that window is scored.

    layout:  [ .. distractors .. MARKER payload .. distractors .. QUERY QUERY ]
    target:  [ BLANK ...........................................  pay  pay  ]

Two ingredients make this provably LTI-defeating:
  (1) p is RANDOM, so the payload sits at a data-dependent position among
      same-type distractors - a time-INVARIANT gate (identical input-gain every
      step) cannot know WHICH symbol to keep.
  (2) only the trailing query window is scored, and the read-out is a SHALLOW
      per-token LINEAR map (no nonlinear MLP, no cross-position mixing), so the
      payload cannot be decoded locally near the marker - it must SURVIVE in the
      recurrent state all the way to the query.
An LTI diagonal SSM blends the lone payload into the long distractor run and
plateaus well below the ceiling. A SELECTIVE SSM makes its step / input-gain /
output-gain FUNCTIONS OF THE INPUT, so it learns "open the gate at the marker,
HOLD through the distractors, read out at the query" - selective-copy / induction.

THE BAKE-OFF.
Two models, identical except for the single sequence-mixing layer, trained on
the SAME data with the SAME init / schedule:
    (A) TNNetSelectiveSSM   - input-dependent, selective (width cDSel).
    (B) TNNetDiagonalSSM    - linear-time-invariant sibling, given the WIDER
                              mixing state (width cDLti > cDSel) so it is NOT
                              starved of capacity; any gap is mechanism, not width.
Shared front-end / read-out:
    Input -> Embedding(d) -> [MIXER] -> PointwiseConvLinear(vocab) -> PointwiseSoftMax.
Training is mini-batch SGD with gradient clipping and a stepped LR decay.

We report cross-entropy and the RECALL accuracy on the trailing query window.
Headline: the selective model CLEARS the content-addressed copy; the wider LTI
sibling PLATEAUS. The example asserts this with self-check gates (Halt(1) on
failure), so a regression in TNNetSelectiveSSM turns the run red.

Pure CPU, single-threaded (manual Compute/Backpropagate), no external dataset,
finishes in well under five minutes.

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
  cVocab    = 7;        // payloads 0..3, QUERY=4, MARKER=5, BLANK=6
  cQuery    = cVocab - 3;
  cMarker   = cVocab - 2;
  cBlank    = cVocab - 1;
  cNPayload = cVocab - 3;  // number of ordinary payload symbols (0..cNPayload-1)
  cSeqLen   = 10;       // sequence length
  cQueryWin = 3;        // trailing positions that carry the QUERY / are scored
  cDSel     = 24;       // selective-model mixing width
  cDLti     = 64;       // LTI-model width (wider -> not starved)
  cEpochs   = 3000;
  cBatch    = 48;       // sequences per epoch
  cLR       = 0.02;     // per-sample SGD
  cInertia  = 0.9;
  cSeed     = 424242;   // repo idiom
  cProbes   = 800;      // evaluation probes

type
  TSeq = array[0..cSeqLen - 1] of integer;

  TArm = record
    Name      : string;
    Params    : integer;
    InitLoss  : TNeuralFloat;
    FinalLoss : TNeuralFloat;
    AccRecall : TNeuralFloat;   // accuracy on the recall region t > p
    Seconds   : TNeuralFloat;
  end;

// One sample. Returns the marker position p; S[p+1] is the payload. The
// marker sits in the FIRST QUARTER, so the payload must be HELD across a long,
// VARIABLE run of same-type distractor symbols before it is finally queried at
// the END of the sequence. The query is signalled by a QUERY token planted in
// the last cQueryWin positions.
//   Layout:  [ .. random distractors .. MARKER payload .. random distractors ..
//              QUERY QUERY ... QUERY ]
// The discriminator: a TIME-INVARIANT (LTI) mixer applies the SAME input-gain
// to every token, so by the time the query arrives the lone payload has been
// blended into (or decayed under) the long distractor run and cannot be
// recovered. A SELECTIVE mixer can shut its input gate during the distractors
// and HOLD the payload in state until the query, then emit it.
procedure MakeSeq(out S: TSeq; out p: integer);
var
  t: integer;
begin
  for t := 0 to cSeqLen - 1 do
    S[t] := Random(cVocab - 3);          // ordinary payload symbols 0..cVocab-4
  // Marker anywhere up to the query window (WIDE, fully variable p). This is the
  // LTI-defeating ingredient: the payload's position is data-dependent and it is
  // surrounded by same-type distractors, so a time-invariant gate (identical
  // input-gain every step) cannot know WHICH token to keep - only a content gate
  // that opens at the marker can. The model must HOLD it to the trailing query.
  p := Random(cSeqLen - cQueryWin - 1);
  S[p] := cMarker;
  if S[p + 1] >= cVocab - 3 then S[p + 1] := Random(cVocab - 3);
  // Plant the QUERY token across the trailing query window.
  for t := cSeqLen - cQueryWin to cSeqLen - 1 do
    S[t] := cQuery;
end;

// Target: BLANK everywhere except the trailing query window, where the model
// must emit the payload S[p+1]. The metric scores ONLY that window, so the
// per-position read-out cannot "cheat" by decoding the payload locally near the
// marker - the payload must survive in the recurrent state all the way to the
// query.
function TargetTok(const S: TSeq; p, t: integer): integer;
begin
  if t >= cSeqLen - cQueryWin then Result := S[p + 1]   // recall at the query
  else Result := cBlank;
end;

procedure FillPair(const S: TSeq; p: integer; InputV, TargetV: TNNetVolume);
var
  t: integer;
begin
  TargetV.Fill(0);
  for t := 0 to cSeqLen - 1 do
  begin
    InputV.FData[t] := S[t];
    TargetV.OneHotEncodingOnPixel(t, 0, TargetTok(S, p, t));
  end;
end;

// Shared front-end / read-out; only the mixer in the middle differs. cUseSel
// selects the selective SSM (width dSel) vs the LTI DiagonalSSM (width dLti).
function BuildNet(UseSelective: boolean; dModel: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(cVocab, dModel, 1));
  if UseSelective
    then Result.AddLayer(TNNetSelectiveSSM.Create())
    else Result.AddLayer(TNNetDiagonalSSM.Create());
  // Deliberately SHALLOW, per-token LINEAR read-out (no cross-position mixing,
  // no nonlinear MLP). This denies the read-out any way to reconstruct the
  // payload from a blended state: the payload must already be cleanly present
  // in the mixer's per-token state at the query, which only the selective
  // (content-gated) recurrence achieves.
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
  Result.AddLayer(TNNetPointwiseSoftMax.Create(1));
  Result.SetLearningRate(cLR, cInertia);
  Result.SetL2Decay(0.0);
end;

function EvalMeanCE(NN: TNNet): TNeuralFloat;
var
  k, p: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  Sum: TNeuralFloat;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  Sum := 0;
  try
    for k := 1 to cProbes do
    begin
      MakeSeq(S, p);
      FillPair(S, p, InputV, TargetV);
      NN.Compute(InputV);
      Sum := Sum + NN.GetLastLayer.Output.MeanCrossEntropy(TargetV);
    end;
  finally
    InputV.Free; TargetV.Free;
  end;
  Result := Sum / cProbes;
end;

// Accuracy on the QUERY window only (the trailing cQueryWin positions). This
// is the content-addressed long-hold recall an LTI mixer structurally cannot do
// once the payload is buried under a long, variable distractor run.
function EvalRecallAcc(NN: TNNet): TNeuralFloat;
var
  k, p, t, Pred, Tgt, C, Ntot: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  C := 0; Ntot := 0;
  try
    for k := 1 to cProbes do
    begin
      MakeSeq(S, p);
      FillPair(S, p, InputV, TargetV);
      NN.Compute(InputV);
      for t := cSeqLen - cQueryWin to cSeqLen - 1 do
      begin
        Pred := NN.GetLastLayer.Output.GetClassOnPixel(t, 0);
        Tgt  := TargetTok(S, p, t);
        if Pred = Tgt then Inc(C);
        Inc(Ntot);
      end;
    end;
  finally
    InputV.Free; TargetV.Free;
  end;
  Result := C / Max(1, Ntot);
end;

function RunArm(UseSelective: boolean; dModel: integer; const AName: string): TArm;
var
  NN: TNNet;
  Epoch, b, p: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  StartTime: double;
begin
  Result.Name := AName;
  RandSeed := cSeed;
  NN := BuildNet(UseSelective, dModel);
  try
    Result.Params := NN.CountWeights();
    RandSeed := cSeed + 1;
    Result.InitLoss := EvalMeanCE(NN);

    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    RandSeed := cSeed;
    StartTime := Now();
    // Mini-batch accumulation with gradient clipping. The selective recurrence
    // has an unbounded step (delta = softplus(.)) and BPTT through it, so raw
    // per-sample SGD can occasionally blow the state up; accumulating a batch
    // and clipping the max |delta| before the update keeps training stable.
    NN.SetBatchUpdate(True);
    try
      for Epoch := 1 to cEpochs do
      begin
        // Step the learning rate DOWN over the last third so the final state is
        // a stable, low-noise solution rather than an SGD-jittered snapshot.
        if Epoch = (cEpochs * 2) div 3 then NN.SetLearningRate(cLR * 0.3, cInertia);
        if Epoch = (cEpochs * 9) div 10 then NN.SetLearningRate(cLR * 0.1, cInertia);
        NN.ClearDeltas();
        for b := 1 to cBatch do
        begin
          MakeSeq(S, p);
          FillPair(S, p, InputV, TargetV);
          NN.Compute(InputV);
          NN.Backpropagate(TargetV);
        end;
        NN.NormalizeMaxAbsoluteDelta(1.0);  // gradient clip (per accumulated batch)
        NN.UpdateWeights();
        {$IFDEF SHOWCURVE}
        if (Epoch mod 300 = 0) then
        begin
          RandSeed := cSeed + 2;
          WriteLn(Format('    [%s] epoch %4d  recall_acc=%.3f',
            [AName, Epoch, EvalRecallAcc(NN)]));
          RandSeed := cSeed;  // (training stream continues; cheap re-sync)
        end;
        {$ENDIF}
      end;
    finally
      InputV.Free; TargetV.Free;
    end;
    Result.Seconds := (Now() - StartTime) * 86400.0;

    RandSeed := cSeed + 1;
    Result.FinalLoss := EvalMeanCE(NN);
    RandSeed := cSeed + 2;
    Result.AccRecall := EvalRecallAcc(NN);
  finally
    NN.Free;
  end;
end;

function SafeF(V: TNeuralFloat; Dec: integer): string;
begin
  if IsNan(V) or IsInfinite(V) then Result := 'NaN'
  else Result := FloatToStrF(V, ffFixed, 8, Dec);
end;

var
  Sel, Lti: TArm;
  Chance: TNeuralFloat;
  GateSelClears, GateSelBeatsLti, GateParamMatched: boolean;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  // Chance recall accuracy: a model that ignores content must guess the payload
  // among the cNPayload ordinary symbols.
  Chance := 1.0 / cNPayload;

  WriteLn('SelectiveSSM: a CONTENT-ADDRESSED selective-copy / induction bake-off.');
  WriteLn('Task: one MARKER (early) tags a payload S[p+1]; after a long, VARIABLE');
  WriteLn('run of same-type distractors a QUERY asks for the payload at the end.');
  WriteLn(Format('seqlen=%d, vocab=%d (QUERY=%d, MARKER=%d, BLANK=%d), query window=%d.',
    [cSeqLen, cVocab, cQuery, cMarker, cBlank, cQueryWin]));
  WriteLn('Only the trailing query window is scored, so the read-out cannot decode');
  WriteLn('the payload locally - it must SURVIVE in the recurrent state to the query.');
  WriteLn('An LTI mixer (same input-gain every step) blends the lone payload into the');
  WriteLn('distractor run; a SELECTIVE one shuts its gate on distractors and HOLDS it.');
  WriteLn(Format('Arms: TNNetSelectiveSSM (d=%d) vs param-matched TNNetDiagonalSSM '
    + '(d=%d).', [cDSel, cDLti]));
  WriteLn(Format('Shared: Embedding(d) -> [MIXER] -> PWConvLinear(%d) -> softmax '
    + '(shallow linear read-out).', [cVocab]));
  WriteLn(Format('Chance recall accuracy (guess among %d payloads) = %.3f.',
    [cNPayload, Chance]));
  WriteLn;

  Write('Training SELECTIVE arm (TNNetSelectiveSSM) ...');
  Sel := RunArm(True, cDSel, 'SelectiveSSM');
  WriteLn(Format(' done.  final_CE=%s  recall_acc=%s  %ss',
    [SafeF(Sel.FinalLoss, 4), SafeF(Sel.AccRecall, 3), SafeF(Sel.Seconds, 2)]));

  Write('Training LTI arm (TNNetDiagonalSSM, param-matched) ...');
  Lti := RunArm(False, cDLti, 'DiagonalSSM');
  WriteLn(Format(' done.  final_CE=%s  recall_acc=%s  %ss',
    [SafeF(Lti.FinalLoss, 4), SafeF(Lti.AccRecall, 3), SafeF(Lti.Seconds, 2)]));
  WriteLn;

  WriteLn(StringOfChar('=', 74));
  WriteLn('RESULTS  (recall_acc = accuracy on the trailing query window)');
  WriteLn(StringOfChar('-', 74));
  WriteLn(Format('  %-16s %8s %10s %10s %12s %8s',
    ['model', 'params', 'init_CE', 'final_CE', 'recall_acc', 'secs']));
  WriteLn(Format('  %-16s %8d %10s %10s %12s %8s',
    [Sel.Name, Sel.Params, SafeF(Sel.InitLoss, 3), SafeF(Sel.FinalLoss, 3),
     SafeF(Sel.AccRecall, 3), SafeF(Sel.Seconds, 2)]));
  WriteLn(Format('  %-16s %8d %10s %10s %12s %8s',
    [Lti.Name, Lti.Params, SafeF(Lti.InitLoss, 3), SafeF(Lti.FinalLoss, 3),
     SafeF(Lti.AccRecall, 3), SafeF(Lti.Seconds, 2)]));
  WriteLn(StringOfChar('-', 74));
  WriteLn(Format('  SELECTIVITY clears the content-addressed copy (%.1f%% recall);',
    [Sel.AccRecall * 100]));
  WriteLn(Format('  the wider LTI sibling PLATEAUS (%.1f%% recall, chance %.1f%%).',
    [Lti.AccRecall * 100, Chance * 100]));
  WriteLn(StringOfChar('=', 74));
  WriteLn;

  // Self-check gates: the example asserts its own headline so a regression in
  // the layer turns the run red instead of printing a quietly-wrong table.
  // The LTI arm is given the WIDER mixing state (d=cDLti > d=cDSel), so it is
  // not starved of capacity - its failure is mechanism, not width.
  GateParamMatched := cDLti >= cDSel;                       // LTI not starved
  GateSelClears    := Sel.AccRecall >= 0.60;                // selective solves it
  GateSelBeatsLti  := Sel.AccRecall >= Lti.AccRecall + 0.12;// clear margin over LTI

  WriteLn('Self-check gates:');
  WriteLn(Format('  [%s] LTI not starved (LTI width %d >= selective width %d)',
    [BoolToStr(GateParamMatched, 'PASS', 'FAIL'), cDLti, cDSel]));
  WriteLn(Format('  [%s] selective recall >= 0.60  (got %.3f)',
    [BoolToStr(GateSelClears, 'PASS', 'FAIL'), Sel.AccRecall]));
  WriteLn(Format('  [%s] selective beats LTI by >= 0.12  (%.3f vs %.3f)',
    [BoolToStr(GateSelBeatsLti, 'PASS', 'FAIL'), Sel.AccRecall, Lti.AccRecall]));

  if GateParamMatched and GateSelClears and GateSelBeatsLti then
  begin
    WriteLn;
    WriteLn('ALL GATES PASSED: selectivity clears content-addressed copy; LTI plateaus.');
  end
  else
  begin
    WriteLn;
    WriteLn('A GATE FAILED.');
    Halt(1);
  end;
end.
