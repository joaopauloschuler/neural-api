program LiquidCfC;
(*
LiquidCfC: end-to-end "remember-then-recall" toy that contrasts the CfC
"liquid" recurrent cell TNNetClosedFormContinuous against a single
scaled-dot-product attention (SDPA) head at MATCHED parameter count.

Reference: Hasani et al. 2022, "Closed-form continuous-time neural networks",
Nature Machine Intelligence. A CfC cell carries a hidden state through time and
gates it with an INPUT-DEPENDENT, per-channel continuous-time constant - the
analytic closed-form solution of a liquid time-constant ODE (no ODE solver):
  tau_t = b_t + Wt.x_t ; gate_t = sigmoid(-tau_t * tnorm_t) ; g_t = tanh(b_g + Wg.x_t)
  h_t   = gate_t * g_t + (1 - gate_t) * h_{t-1}

Task (delayed copy / "remember-then-recall"): a CUE symbol in {1..cNumSym} is
shown at position 0; the rest of the sequence is filler (symbol 0); at the LAST
position the model must output the cue. This needs the model to carry one piece
of information across the whole sequence - a clean recurrence vs attention test.

Two models are trained on the SAME data:
  CfC : Embedding -> TNNetClosedFormContinuous -> per-position readout
  SDPA: Embedding -> Q|K|V proj -> SDPA head -> per-position readout
Their learnable-weight counts are printed; the SDPA d_k is chosen so the two
counts are close. We report final recall accuracy at the last position for both.

Everything is generated on the fly; no external dataset. Runs in well under a
minute on a single CPU thread with a modest memory footprint.

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
  cSeqLen  = 8;     // sequence length (recall horizon)
  cNumSym  = 4;     // number of distinct cue symbols (1..cNumSym; 0 = filler)
  cVocab   = cNumSym + 1;
  cDModel  = 8;     // embedding / recurrent width
  cDk      = 6;     // SDPA head width, picked so param counts ~match CfC
  cSteps   = 2000;
  cBatch   = 16;
  cLR      = 0.01;
  cInertia = 0.9;
  cEval    = 400;   // eval sequences for accuracy

  // Build a CfC model: Embedding -> CfC liquid cell -> per-position readout.
  procedure BuildCfC(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
    NN.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
    // The liquid recurrent cell (the layer this example showcases).
    NN.AddLayer(TNNetClosedFormContinuous.Create());
    // Per-position linear projection to vocab logits + softmax over depth.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
    NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
    NN.SetLearningRate(cLR, cInertia);
  end;

  // Build a single-head SDPA model with the SAME embedding + readout skeleton.
  procedure BuildSDPA(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
    NN.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
    // Pack Q|K|V (depth = 3*d_k) for a single causal attention head.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDk));
    NN.AddLayer(TNNetScaledDotProductAttention.Create(cDk, True));
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
    NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
    NN.SetLearningRate(cLR, cInertia);
  end;

  // Build one (input, target) pair. Cue symbol at position 0, filler elsewhere,
  // and the target at the LAST position equals the cue (one-hot over depth).
  procedure MakePair(InputV, TargetV: TNNetVolume);
  var
    I, Cue: integer;
  begin
    Cue := 1 + Random(cNumSym);
    InputV.Fill(0);
    InputV.FData[0] := Cue;          // cue shown only at position 0
    for I := 1 to cSeqLen - 1 do InputV.FData[I] := 0; // filler symbol 0
    // Dense supervision: EVERY position must report the cue, so the model has to
    // carry it forward (a recurrence holds it in state; attention looks back to
    // position 0). Accuracy is read at the last position (the longest horizon).
    TargetV.Fill(0);
    for I := 0 to cSeqLen - 1 do TargetV[I, 0, Cue] := 1;
  end;

  // Find which symbol the last-position target one-hot encodes.
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
    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
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
          // Cross-entropy-ish: -log prob of the true last-position symbol.
          SumErr := SumErr -
            Ln(Max(NN.GetLastLayer.Output[cSeqLen - 1, 0,
              Round(TargetLastSymbol(TargetV))], 1e-7));
        end;
        if (Step = 1) or (Step mod 50 = 0) or (Step = cSteps) then
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

  // Last-position recall accuracy over freshly generated sequences.
  function EvalAccuracy(NN: TNNet): TNeuralFloat;
  var
    I, D, Best, Truth, Hits: integer;
    InputV, TargetV: TNNetVolume;
    BestV: TNeuralFloat;
  begin
    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
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
  CfCNet, SdpaNet: TNNet;
  AccCfC, AccSdpa: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;
  WriteLn('LiquidCfC: remember-then-recall over a length-', cSeqLen,
    ' sequence.');
  WriteLn('A cue symbol shown at position 0 must be reproduced at the last ',
    'position.');
  WriteLn('Contrasting TNNetClosedFormContinuous (CfC liquid cell) vs a ',
    'single SDPA head.');
  WriteLn;

  BuildCfC(CfCNet);
  BuildSDPA(SdpaNet);
  try
    WriteLn('CfC model (', CfCNet.CountWeights(), ' learnable weights):');
    CfCNet.PrintSummary();
    WriteLn;
    WriteLn('SDPA model (', SdpaNet.CountWeights(), ' learnable weights):');
    SdpaNet.PrintSummary();
    WriteLn;

    WriteLn('Training CfC model...');
    Train(CfCNet, 'CfC ');
    WriteLn('Training SDPA model...');
    Train(SdpaNet, 'SDPA');
    WriteLn;

    AccCfC := EvalAccuracy(CfCNet);
    AccSdpa := EvalAccuracy(SdpaNet);
    WriteLn(StringOfChar('=', 64));
    WriteLn(Format('  Last-position recall accuracy (chance = %.3f):',
      [1.0 / cNumSym]));
    WriteLn(Format('    CfC  liquid cell : %.1f%%   (%d weights)',
      [AccCfC * 100, CfCNet.CountWeights()]));
    WriteLn(Format('    SDPA attention   : %.1f%%   (%d weights)',
      [AccSdpa * 100, SdpaNet.CountWeights()]));
    WriteLn(StringOfChar('=', 64));
    WriteLn;
    WriteLn('Both should approach 100%: the CfC cell solves the recall by ',
      'carrying');
    WriteLn('the cue in its liquid hidden state, the SDPA head by attending ',
      'back to');
    WriteLn('position 0 - two different mechanisms at a matched parameter ',
      'budget.');
  finally
    CfCNet.Free;
    SdpaNet.Free;
  end;
end.
