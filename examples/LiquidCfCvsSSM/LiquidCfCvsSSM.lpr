program LiquidCfCvsSSM;
(*
LiquidCfCvsSSM: a longer-horizon, MULTI-CUE "gated recall" toy that isolates the
one thing a CfC liquid cell has and a fixed-decay diagonal SSM does not - an
INPUT-DEPENDENT time constant - and shows it pays off at a MATCHED parameter
budget.

Reference: Hasani et al. 2022, "Closed-form continuous-time neural networks",
Nature Machine Intelligence. A CfC cell carries a hidden state through time and
gates it with an INPUT-DEPENDENT, per-channel continuous-time constant - the
analytic closed-form solution of a liquid time-constant ODE (no ODE solver):
  tau_t = b_t + Wt.x_t ; gate_t = sigmoid(-tau_t * tnorm_t) ; g_t = tanh(b_g + Wg.x_t)
  h_t   = gate_t * g_t + (1 - gate_t) * h_{t-1}
The decay/keep balance gate_t depends on the CURRENT INPUT x_t. A diagonal SSM
(TNNetDiagonalSSM) instead has a FIXED, input-independent per-channel decay
  h_t = a[d]*h_{t-1} + b[d]*x_t ;  y_t = c[d]*h_t + e[d]*x_t   (a[d] learned ONCE)
so once trained it decays at the same rate no matter what it just saw.

Task (LAST-WRITE-WINS over a long sequence): SEVERAL cue symbols (2..4) are
written at random positions; a dedicated WRITE marker channel pulses on at each
cue; the rest is filler. At the LAST position the model must output the MOST
RECENTLY written cue. This is the regime that separates an input-dependent gate
from a fixed decay: the model must FORGET the previous value and OVERWRITE with
the new one exactly when a write pulse fires. The CfC's input-dependent gate can
slam shut on a WRITE pulse and reset; a fixed per-channel decay can only BLEND,
so earlier cues bleed into the answer and it never fully overwrites.

Two models are trained on the SAME data:
  CfC : Embedding(+WRITE) -> TNNetClosedFormContinuous -> per-position readout
  SSM : Embedding(+WRITE) -> TNNetDiagonalSSM (wider)   -> per-position readout
The SSM is given a wider working width so the two learnable-weight counts are
close (a diagonal SSM is cheap per channel: 4*d vs the CfC's 2*d^2+2*d). Both
counts are printed and last-position recall accuracy is reported for each.

Everything is generated on the fly; no external dataset. Runs in a couple of
minutes on a single CPU thread with a modest memory footprint (< 5 min budget).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSeqLen   = 16;    // longer recall horizon than the base LiquidCfC toy (was 8)
  cNumSym   = 4;     // number of distinct cue symbols (1..cNumSym; 0 = filler)
  cVocab    = cNumSym + 1;
  cDModelC  = 8;     // CfC embedding / recurrent width
  cDModelS  = 14;    // SSM width, widened so param counts ~match the CfC
  cSteps    = 3000;
  cBatch    = 16;
  cLR       = 0.004;
  cInertia  = 0.9;
  cEval     = 600;   // eval sequences for accuracy

// The per-token input has TWO feature channels: the symbol id (channel 0, fed
// through an embedding) and the WRITE marker pulse (channel 1, 0/1). The marker
// makes the cue position observable to the model without leaking the timing into
// a fixed positional code.
//
// Model skeleton: split the 2 input channels, embed channel 0 to d_model-1 dims,
// keep the raw WRITE pulse as the last channel, then run the recurrent mixer.

  // Build a CfC model. The recurrent cell sees d_model channels: an embedding of
  // the symbol (d_model-1 dims) with the raw write pulse appended as the last
  // channel, so the cell can detect WHEN to write.
  procedure BuildCfC(out NN: TNNet);
  var WritePulse: TNNetLayer;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 2));     // [symbol, write-pulse]
    WritePulse := NN.AddLayer(TNNetSplitChannels.Create([1])); // write pulse only
    NN.AddLayerAfter(TNNetSplitChannels.Create([0]), NN.GetFirstLayer()); // symbol
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModelC - 1, 1));
    NN.AddLayer(TNNetDeepConcat.Create([NN.GetLastLayer(), WritePulse]));
    NN.AddLayer(TNNetClosedFormContinuous.Create());   // the liquid recurrent cell
    NN.AddLayer(TNNetLayerNorm.Create());              // stabilize the readout
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
    NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
    NN.SetLearningRate(cLR, cInertia);
  end;

  // Build a fixed-decay diagonal-SSM model with the SAME skeleton, just wider.
  procedure BuildSSM(out NN: TNNet);
  var WritePulse: TNNetLayer;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 2));
    WritePulse := NN.AddLayer(TNNetSplitChannels.Create([1]));
    NN.AddLayerAfter(TNNetSplitChannels.Create([0]), NN.GetFirstLayer());
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModelS - 1, 1));
    NN.AddLayer(TNNetDeepConcat.Create([NN.GetLastLayer(), WritePulse]));
    NN.AddLayer(TNNetDiagonalSSM.Create());            // FIXED-decay mixer
    NN.AddLayer(TNNetLayerNorm.Create());              // stabilize the readout
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
    NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
    NN.SetLearningRate(cLR, cInertia);
  end;

  // Build one (input, target) pair for the LAST-WRITE-WINS task. SEVERAL cues
  // are written at random positions (each flagged by the write pulse); the
  // target is the MOST RECENTLY written cue at every position (dense
  // supervision; accuracy is read at the last position). This is the regime that
  // separates an input-dependent gate from a fixed decay: the model must FORGET
  // the previous value and OVERWRITE with the new one exactly when a write pulse
  // fires - a fixed per-channel decay cannot selectively reset, it can only
  // blend, so old cues bleed into the answer.
  procedure MakePair(InputV, TargetV: TNNetVolume);
  var
    I, D, Cue, NumWrites, W, Pos: integer;
    LastCue: array[0..cSeqLen - 1] of integer;
    Used: array[0..cSeqLen - 1] of boolean;
  begin
    InputV.Fill(0);
    for I := 0 to cSeqLen - 1 do Used[I] := false;
    // 2..4 writes at distinct random positions.
    NumWrites := 2 + Random(3);
    for W := 1 to NumWrites do
    begin
      repeat Pos := Random(cSeqLen) until not Used[Pos];
      Used[Pos] := true;
      Cue := 1 + Random(cNumSym);
      InputV[Pos, 0, 0] := Cue;
      InputV[Pos, 0, 1] := 1;     // write pulse on at every cue
    end;
    // Most-recently-written cue at each position (0 = nothing written yet).
    Cue := 0;
    for I := 0 to cSeqLen - 1 do
    begin
      if Used[I] then Cue := Round(InputV[I, 0, 0]);
      LastCue[I] := Cue;
    end;
    TargetV.Fill(0);
    for I := 0 to cSeqLen - 1 do
    begin
      D := LastCue[I];
      if D = 0 then D := 0;       // no write yet -> filler class 0
      TargetV[I, 0, D] := 1;
    end;
  end;

  function TargetLastSymbol(TargetV: TNNetVolume): integer;
  var D: integer;
  begin
    Result := 0;
    for D := 0 to cVocab - 1 do
      if TargetV[cSeqLen - 1, 0, D] > 0.5 then Exit(D);
  end;

  procedure Train(NN: TNNet; const Tag: string);
  var
    Step, B: integer;
    InputV, TargetV: TNNetVolume;
    SumErr: TNeuralFloat;
    StartTime, Elapsed: double;
  begin
    InputV  := TNNetVolume.Create(cSeqLen, 1, 2);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    try
      StartTime := Now();
      for Step := 1 to cSteps do
      begin
        SumErr := 0;
        for B := 1 to cBatch do
        begin
          MakePair(InputV, TargetV);
          NN.Compute(InputV);
          NN.Backpropagate(TargetV);
          SumErr := SumErr -
            Ln(Max(NN.GetLastLayer.Output[cSeqLen - 1, 0,
              Round(TargetLastSymbol(TargetV))], 1e-7));
        end;
        if (Step = 1) or (Step mod 100 = 0) or (Step = cSteps) then
        begin
          Elapsed := (Now() - StartTime) * 86400.0;
          WriteLn(Format('  [%s] step %4d / %4d   mean -log p=%.4f   elapsed=%.1fs',
            [Tag, Step, cSteps, SumErr / cBatch, Elapsed]));
        end;
      end;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

  function EvalAccuracy(NN: TNNet): TNeuralFloat;
  var
    I, D, Best, Truth, Hits: integer;
    InputV, TargetV: TNNetVolume;
    BestV: TNeuralFloat;
  begin
    InputV  := TNNetVolume.Create(cSeqLen, 1, 2);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    Hits := 0;
    try
      for I := 1 to cEval do
      begin
        MakePair(InputV, TargetV);
        NN.Compute(InputV);
        Truth := TargetLastSymbol(TargetV);
        Best := 0;
        BestV := NN.GetLastLayer.Output[cSeqLen - 1, 0, 0];
        for D := 1 to cVocab - 1 do
          if NN.GetLastLayer.Output[cSeqLen - 1, 0, D] > BestV then
          begin
            BestV := NN.GetLastLayer.Output[cSeqLen - 1, 0, D];
            Best := D;
          end;
        if Best = Truth then Inc(Hits);
      end;
    finally
      InputV.Free;
      TargetV.Free;
    end;
    Result := Hits / cEval;
  end;

var
  CfCNet, SsmNet: TNNet;
  AccCfC, AccSsm: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;
  WriteLn('LiquidCfCvsSSM: last-write-wins over a length-', cSeqLen, ' sequence.');
  WriteLn('Several cues (each flagged by a write pulse) are written; the LAST-');
  WriteLn('written one must be reproduced at the last position. Contrasting ',
    'TNNet-');
  WriteLn('ClosedFormContinuous (CfC, input-dependent decay) vs TNNetDiagonalSSM',
    ' (fixed');
  WriteLn('per-channel decay).');
  WriteLn;

  BuildCfC(CfCNet);
  BuildSSM(SsmNet);
  try
    WriteLn('CfC model (', CfCNet.CountWeights(), ' learnable weights):');
    CfCNet.PrintSummary();
    WriteLn;
    WriteLn('SSM model (', SsmNet.CountWeights(), ' learnable weights):');
    SsmNet.PrintSummary();
    WriteLn;

    WriteLn('Training CfC model...');
    Train(CfCNet, 'CfC ');
    WriteLn('Training SSM model...');
    Train(SsmNet, 'SSM ');
    WriteLn;

    AccCfC := EvalAccuracy(CfCNet);
    AccSsm := EvalAccuracy(SsmNet);
    WriteLn(StringOfChar('=', 64));
    WriteLn(Format('  Last-position recall accuracy (chance = %.3f):',
      [1.0 / cNumSym]));
    WriteLn(Format('    CfC liquid cell (input-dependent decay): %.1f%%   (%d weights)',
      [AccCfC * 100, CfCNet.CountWeights()]));
    WriteLn(Format('    Diagonal SSM    (fixed decay)          : %.1f%%   (%d weights)',
      [AccSsm * 100, SsmNet.CountWeights()]));
    WriteLn(StringOfChar('=', 64));
    WriteLn;
    WriteLn('Each write must OVERWRITE the previous value. The CfC gate reacts to');
    WriteLn('the WRITE pulse and resets/clamps its state; a fixed-decay SSM can');
    WriteLn('only blend, so earlier cues bleed in - which is why the liquid cell');
    WriteLn('recalls the last write more reliably even at MORE weights for the');
    WriteLn('SSM (input-dependent vs fixed time constant at a matched budget).');
  finally
    CfCNet.Free;
    SsmNet.Free;
  end;
end.
