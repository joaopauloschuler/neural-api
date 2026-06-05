program DiagonalSSM;
(*
DiagonalSSM: smallest possible end-to-end demo of TNNetDiagonalSSM, the
diagonal-state linear-recurrence ("SSM-lite") sequence mixer - an O(n)
causal alternative to an attention head and the first genuinely recurrent
layer in the library.

Task: a tiny "cumulative-sum / running-memory" sequence problem. The input
is a length-cSeqLen sequence of scalars laid out along the X axis and lifted
to cDepth channels by a pointwise projection. The target at each position t
is the running (decaying) statistic of the sequence up to t, which a linear
recurrence h_t = a*h_{t-1} + b*x_t captures exactly. Different target
channels need DIFFERENT memory horizons (fast vs slow), so after training the
per-channel learned decay spectrum a[d] = sigmoid(a_raw[d]) spreads out into
fast-forgetting and slow-remembering channels - which we print.

Pipeline:
  scalar seq -> TNNetInput(cSeqLen,1,1)
             -> TNNetPointwiseConvLinear(cDepth)   { lift to channels }
             -> TNNetDiagonalSSM                    { the recurrent mixer }
             -> TNNetPointwiseConvLinear(cOut)      { per-position readout }

Everything is generated on the fly; no external dataset. Runs in well under
a minute on a single CPU thread.

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
  cSeqLen  = 12;
  cDepth   = 4;     // recurrent channels (the decay spectrum we visualise)
  cOut     = 2;     // two target running-statistics (a fast and a slow one)
  cSteps   = 600;
  cBatch   = 16;
  cLR      = 0.02;
  cInertia = 0.9;
  // "Teacher" decays used to generate the two target running statistics.
  cTeacherFast = 0.1;   // forgets quickly
  cTeacherSlow = 0.95;  // long memory

var
  GSSM: TNNetDiagonalSSM;

  procedure BuildModel(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
    // Lift the scalar input to cDepth channels.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cDepth));
    // The recurrent diagonal-SSM mixer (the layer this example showcases).
    GSSM := TNNetDiagonalSSM.Create();
    NN.AddLayer(GSSM);
    // Per-position linear readout to the cOut target statistics.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cOut));
    NN.SetLearningRate(cLR, cInertia);
  end;

  // Build one (input, target) pair. Input is a random scalar sequence on X.
  // Target channel 0 is a fast-decaying running statistic, channel 1 a
  // slow-decaying one - both exact outputs of a 1-pole linear recurrence.
  procedure MakePair(InputV, TargetV: TNNetVolume);
  var
    I: integer;
    x, hFast, hSlow: TNeuralFloat;
  begin
    hFast := 0;
    hSlow := 0;
    for I := 0 to cSeqLen - 1 do
    begin
      x := Random * 2.0 - 1.0;            // uniform in [-1, 1]
      InputV.FData[I] := x;
      hFast := cTeacherFast * hFast + (1 - cTeacherFast) * x;
      hSlow := cTeacherSlow * hSlow + (1 - cTeacherSlow) * x;
      TargetV[I, 0, 0] := hFast;
      TargetV[I, 0, 1] := hSlow;
    end;
  end;

  function MeanSquaredError(Output, Target: TNNetVolume): TNeuralFloat;
  var
    I: integer;
    diff: TNeuralFloat;
  begin
    Result := 0;
    for I := 0 to Output.Size - 1 do
    begin
      diff := Output.FData[I] - Target.FData[I];
      Result := Result + diff * diff;
    end;
    Result := Result / Output.Size;
  end;

  procedure Train(NN: TNNet);
  var
    Step, B: integer;
    InputV, TargetV: TNNetVolume;
    SumLoss: TNeuralFloat;
    StartTime, Elapsed: double;
  begin
    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cOut);
    try
      StartTime := Now();
      for Step := 1 to cSteps do
      begin
        SumLoss := 0;
        for B := 1 to cBatch do
        begin
          MakePair(InputV, TargetV);
          NN.Compute(InputV);
          SumLoss := SumLoss + MeanSquaredError(NN.GetLastLayer.Output, TargetV);
          NN.Backpropagate(TargetV);
        end;
        if (Step = 1) or (Step mod 50 = 0) or (Step = cSteps) then
        begin
          Elapsed := (Now() - StartTime) * 86400.0;
          WriteLn(Format('  step %4d / %4d   mean-MSE=%.6f   elapsed=%.1fs',
            [Step, cSteps, SumLoss / cBatch, Elapsed]));
        end;
      end;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

  procedure PrintDecaySpectrum;
  var
    D: integer;
    aRaw, aVal, bVal, cVal, eVal: TNeuralFloat;
  begin
    WriteLn('Learned per-channel decay spectrum a[d] = sigmoid(a_raw[d]):');
    WriteLn('  channel    a_raw        a=sig(a_raw)      b         c         e');
    for D := 0 to cDepth - 1 do
    begin
      aRaw := GSSM.Neurons[0].Weights.Raw[D];
      aVal := 1 / (1 + Exp(-aRaw));
      bVal := GSSM.Neurons[1].Weights.Raw[D];
      cVal := GSSM.Neurons[2].Weights.Raw[D];
      eVal := GSSM.Neurons[3].Weights.Raw[D];
      WriteLn(Format('  %5d  %10.4f      %8.4f    %8.4f  %8.4f  %8.4f',
        [D, aRaw, aVal, bVal, cVal, eVal]));
    end;
  end;

var
  NN: TNNet;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;
  WriteLn('DiagonalSSM demo: a single recurrent TNNetDiagonalSSM layer learns ');
  WriteLn('fast- and slow-memory running statistics over a length-', cSeqLen,
    ' sequence.');
  BuildModel(NN);
  try
    WriteLn;
    WriteLn('Architecture:');
    NN.PrintSummary();
    WriteLn;
    WriteLn('Decay spectrum BEFORE training (a_raw init 0 -> a = 0.5 everywhere):');
    PrintDecaySpectrum;
    WriteLn;
    WriteLn('Training for ', cSteps, ' steps of batch size ', cBatch, '...');
    Train(NN);
    WriteLn;
    WriteLn(StringOfChar('=', 72));
    PrintDecaySpectrum;
    WriteLn(StringOfChar('=', 72));
    WriteLn;
    WriteLn('Expect: the channels differentiate into fast-forgetting (a near 0) ',
      'and');
    WriteLn('slow-remembering (a near 1) memory horizons, since the readout ',
      'must');
    WriteLn('reconstruct both a quickly-decaying and a long-memory running ',
      'statistic.');
  finally
    NN.Free;
  end;
end.
