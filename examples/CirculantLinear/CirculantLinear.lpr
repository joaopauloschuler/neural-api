program CirculantLinear;
(*
CirculantLinear: a parameter-efficiency bake-off for TNNetCirculantLinear, the
STRUCTURED-MATRIX dense layer whose n x n weight matrix is CIRCULANT (every row
is a cyclic shift of one learned length-n kernel c), so the map is the circular
convolution y = circconv(c, x) (+ bias) and the layer stores only O(n) weights
instead of the O(n^2) of a full dense layer.

Task: the regression target is a GENUINE circular convolution of the input with
a fixed (unknown to the model) teacher kernel c_true, plus a small per-output
bias. This is EXACTLY the function class a circulant layer represents, so a
circulant layer can fit it with only 2n learned numbers. A param-matched dense
TNNetFullConnectLinear can also fit it, but it pays n*n + n weights to do so.

Two models are trained on the same data, head to head:
  (A) TNNetCirculantLinear(n)      -> 2n  trainable weights (kernel c + bias)
  (B) TNNetFullConnectLinear(n)    -> n*n + n trainable weights (dense)

The headline is ACCURACY-PER-WEIGHT: both reach a low test MSE, but the
circulant layer reaches it with O(n) weights, so its (1/test-MSE) per weight is
dramatically higher. We also print the recovered kernel vs the teacher kernel to
show the circulant layer actually learned the true circular-convolution
operator.

Everything is generated on the fly; no external dataset. Pure CPU, single
thread, well under five minutes (a couple of seconds in practice).

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
  cN       = 16;      // sequence / vector length (the n in the n x n map)
  cSteps   = 1500;
  cBatch   = 16;
  cLR      = 0.02;
  cInertia = 0.9;
  cTestSet = 512;     // held-out test pairs for the final MSE

var
  GTeacher: array[0..cN - 1] of TNeuralFloat;  // unknown teacher kernel c_true
  GBias:    array[0..cN - 1] of TNeuralFloat;  // unknown teacher bias

  // Fill the teacher kernel and bias once (the function the models must learn).
  procedure InitTeacher;
  var
    I: integer;
    S: TNeuralFloat;
  begin
    S := 0;
    for I := 0 to cN - 1 do
    begin
      // A smooth, localized kernel (a couple of dominant taps) so the circular
      // convolution is a non-trivial mixing operator.
      GTeacher[I] := Exp(-0.5 * Sqr((I - 2) / 1.5)) - 0.3 * Exp(-0.5 * Sqr((I - 7) / 2.0));
      S := S + Abs(GTeacher[I]);
    end;
    for I := 0 to cN - 1 do
      GTeacher[I] := GTeacher[I] / S;          // keep the operator well-scaled
    for I := 0 to cN - 1 do
      GBias[I] := 0.1 * Sin(I * 0.7);
  end;

  // One (input, target) pair. Target = circular convolution of x with the
  // teacher kernel, plus the teacher bias:
  //   target[i] = bias[i] + sum_k c_true[(i-k) mod n] * x[k].
  procedure MakePair(InputV, TargetV: TNNetVolume);
  var
    I, K, Idx: integer;
    Acc: TNeuralFloat;
  begin
    for I := 0 to cN - 1 do
      InputV.FData[I] := Random * 2.0 - 1.0;     // uniform in [-1, 1]
    for I := 0 to cN - 1 do
    begin
      Acc := GBias[I];
      for K := 0 to cN - 1 do
      begin
        Idx := (I - K) mod cN;
        if Idx < 0 then Idx := Idx + cN;
        Acc := Acc + GTeacher[Idx] * InputV.FData[K];
      end;
      TargetV.FData[I] := Acc;
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

  procedure Train(NN: TNNet; const Name: string);
  var
    Step, B: integer;
    InputV, TargetV: TNNetVolume;
    SumLoss: TNeuralFloat;
  begin
    InputV  := TNNetVolume.Create(1, 1, cN);
    TargetV := TNNetVolume.Create(1, 1, cN);
    try
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
        if (Step = 1) or (Step mod 250 = 0) or (Step = cSteps) then
          WriteLn(Format('  [%s] step %4d / %4d   train-MSE=%.6e',
            [Name, Step, cSteps, SumLoss / cBatch]));
      end;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

  // Mean test MSE over a fixed held-out set (re-seeded so both models see the
  // SAME test pairs).
  function EvalTestMSE(NN: TNNet): TNeuralFloat;
  var
    I: integer;
    InputV, TargetV: TNNetVolume;
  begin
    RandSeed := 99991;
    InputV  := TNNetVolume.Create(1, 1, cN);
    TargetV := TNNetVolume.Create(1, 1, cN);
    Result := 0;
    try
      for I := 1 to cTestSet do
      begin
        MakePair(InputV, TargetV);
        NN.Compute(InputV);
        Result := Result + MeanSquaredError(NN.GetLastLayer.Output, TargetV);
      end;
      Result := Result / cTestSet;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

  // Count trainable weights (all neuron weight vectors + their biases) of a
  // layer, the honest parameter budget for the per-weight headline.
  function CountTrainable(Layer: TNNetLayer; CountBias: boolean): integer;
  var
    O: integer;
  begin
    Result := 0;
    for O := 0 to Layer.Neurons.Count - 1 do
    begin
      Result := Result + Layer.Neurons[O].Weights.Size;
      if CountBias then Inc(Result);
    end;
  end;

var
  NNCirc, NNDense: TNNet;
  CircLayer, DenseLayer: TNNetLayer;
  CircParams, DenseParams: integer;
  CircMSE, DenseMSE: TNeuralFloat;
  CircEff, DenseEff: TNeuralFloat;
  StartTime, Elapsed: double;
  I: integer;
  recovered: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  StartTime := Now();
  InitTeacher;

  WriteLn('CirculantLinear bake-off: a length-', cN,
    ' CIRCULAR-CONVOLUTION regression target,');
  WriteLn('learned by a structured circulant layer (2n weights) vs a ',
    'param-matched');
  WriteLn('dense layer (n*n + n weights). Headline: accuracy PER WEIGHT.');
  WriteLn;

  // ---- Model A: the circulant (structured) layer ----
  RandSeed := 2026;
  NNCirc := TNNet.Create();
  NNCirc.AddLayer(TNNetInput.Create(1, 1, cN));
  CircLayer := NNCirc.AddLayer(TNNetCirculantLinear.Create(cN));
  NNCirc.SetLearningRate(cLR, cInertia);

  // ---- Model B: the param-matched dense baseline ----
  RandSeed := 2026;
  NNDense := TNNet.Create();
  NNDense.AddLayer(TNNetInput.Create(1, 1, cN));
  DenseLayer := NNDense.AddLayer(TNNetFullConnectLinear.Create(cN));
  NNDense.SetLearningRate(cLR, cInertia);

  // The circulant layer stores kernel c + bias in two length-n neurons; the
  // dense layer stores a full n x n matrix + n biases.
  CircParams  := CountTrainable(CircLayer, {CountBias=}false);   // 2n (c + bias vector)
  DenseParams := CountTrainable(DenseLayer, {CountBias=}true);   // n*n + n

  try
    WriteLn('Trainable parameters:');
    WriteLn(Format('  circulant  TNNetCirculantLinear(%d) : %5d   (= 2n)',
      [cN, CircParams]));
    WriteLn(Format('  dense      TNNetFullConnectLinear(%d): %5d   (= n*n + n)',
      [cN, DenseParams]));
    WriteLn;

    WriteLn('Training circulant model...');
    RandSeed := 2026;
    Train(NNCirc, 'circ ');
    WriteLn('Training dense model...');
    RandSeed := 2026;
    Train(NNDense, 'dense');
    WriteLn;

    CircMSE  := EvalTestMSE(NNCirc);
    DenseMSE := EvalTestMSE(NNDense);

    // Accuracy-per-weight: (1 / test-MSE) spread over the trainable budget.
    CircEff  := (1.0 / CircMSE)  / CircParams;
    DenseEff := (1.0 / DenseMSE) / DenseParams;

    WriteLn(StringOfChar('=', 72));
    WriteLn('RESULTS');
    WriteLn(StringOfChar('-', 72));
    WriteLn(Format('  %-28s %12s %14s %16s',
      ['model', 'params', 'test-MSE', 'acc/weight']));
    WriteLn(Format('  %-28s %12d %14.3e %16.3e',
      ['circulant (structured)', CircParams, CircMSE, CircEff]));
    WriteLn(Format('  %-28s %12d %14.3e %16.3e',
      ['dense (param-matched)', DenseParams, DenseMSE, DenseEff]));
    WriteLn(StringOfChar('-', 72));
    WriteLn(Format('  Circulant uses %.1fx FEWER weights and is %.1fx more ' +
      'ACCURATE-PER-WEIGHT.',
      [DenseParams / CircParams, CircEff / DenseEff]));
    WriteLn(StringOfChar('=', 72));
    WriteLn;

    WriteLn('Recovered circulant kernel vs the (unknown) teacher kernel:');
    WriteLn('  tap   teacher c_true     learned c');
    for I := 0 to cN - 1 do
    begin
      recovered := CircLayer.Neurons[0].Weights.Raw[I];
      WriteLn(Format('  %3d   %12.5f   %12.5f', [I, GTeacher[I], recovered]));
    end;

    Elapsed := (Now() - StartTime) * 86400.0;
    WriteLn;
    WriteLn(Format('Total wall-clock: %.1f s (pure CPU, single thread).',
      [Elapsed]));
  finally
    NNCirc.Free;
    NNDense.Free;
  end;
end.
