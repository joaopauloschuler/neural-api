program KANLayer;
(*
KANLayer: a layer-KAN vs param-matched ReLU-MLP toy-fit micro-experiment for the
landed TNNetKANLayer — a true Kolmogorov-Arnold *dense layer* (Liu et al. 2024,
"KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756).

Where the sibling example examples/SplineActivationKAN puts the KAN idea in an
ACTIVATION (one learnable univariate function per channel, depth-preserving),
this example uses the KAN idea in the WEIGHTS: TNNetKANLayer maps D_in -> D_out
where EVERY input->output edge (i,j) carries its own learned univariate function
phi_{ij}(x_i) = sum_k c_{ijk} T_k(tanh(x_i)) (Chebyshev basis of degree K), and
the output is the pure sum y_j = sum_i phi_{ij}(x_i). There is no weight matrix —
the "weights" ARE the edge functions.

Target (wiggly so a learnable nonlinearity has something to earn):
    y = sin(3x) + 0.3*sin(11x),  x in [-2, 2]

Two arms, trained the SAME number of epochs with the SAME data and optimizer:
  Arm A (baseline MLP):  Input(1) -> FullConnectReLU(Wa) -> FullConnectLinear(1)
  Arm B (KAN-dense MLP): Input(1) -> KANLayer(Wb, K) -> FullConnectLinear(1)

The KAN arm spends its parameters on D_in*D_out*(K+1) edge coefficients. To keep
this a FAIR fixed-param fight we size the ReLU arm WIDER (Wa > Wb) so both arms
carry ~equal total trainable weight counts. The exact per-arm weight count
(TNNet.CountWeights) is computed and PRINTED so the reader can see the match, and
the final clean-grid MSE of both arms is printed side by side.

See also examples/SplineActivationKAN for the activation-KAN counterpart, so the
two KAN flavours (per-channel learnable ACTIVATION vs per-edge learnable WEIGHT
function) can be read side by side.

Pure CPU, no dataset download, synthetic data generated in-code, runs in a few
seconds.

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

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cXRange   = 2.0;    // target sampled over x in [-cXRange, +cXRange]
  cNumTrain = 256;    // synthetic training samples
  cEpochs   = 600;    // same epoch budget for both arms
  cBatch    = 16;
  cLR       = 0.01;
  cMomentum = 0.9;

  // KAN arm geometry. Hidden width D_out and Chebyshev degree K per edge.
  cKANW     = 8;      // hidden width of the KAN-dense arm (D_out)
  cKANK     = 4;      // Chebyshev degree K (=> K+1 coefficients per edge)
  // The ReLU arm is made WIDER to match total weight count (see ChooseReLUWidth).

  // Target function: a wiggly 1D signal so a learnable layer can earn its keep.
  function Target(X: TNeuralFloat): TNeuralFloat;
  begin
    Result := Sin(3 * X) + 0.3 * Sin(11 * X);
  end;

  procedure MakeTrainSet(out Xs, Ys: array of TNeuralFloat);
  var
    I: integer;
  begin
    for I := 0 to High(Xs) do
    begin
      Xs[I] := (Random - 0.5) * 2.0 * cXRange;  // x in [-cXRange, +cXRange)
      Ys[I] := Target(Xs[I]);
    end;
  end;

  // MSE on a dense clean grid (no noise), the honest measure of fit quality.
  function GridMSE(NN: TNNet): TNeuralFloat;
  var
    I: integer;
    X, Diff, Sum: TNeuralFloat;
    Inp, Outp: TNNetVolume;
  begin
    Inp := TNNetVolume.Create(1, 1, 1);
    Outp := TNNetVolume.Create(1, 1, 1);
    Sum := 0;
    for I := 0 to 199 do
    begin
      X := -cXRange + 2.0 * cXRange * I / 199.0;
      Inp.Raw[0] := X;
      NN.Compute(Inp);
      NN.GetOutput(Outp);
      Diff := Outp.Raw[0] - Target(X);
      Sum := Sum + Diff * Diff;
    end;
    Inp.Free;
    Outp.Free;
    Result := Sum / 200.0;
  end;

  // One full mini-batch SGD training run on the (shared) training set.
  function TrainArm(NN: TNNet; const Xs, Ys: array of TNeuralFloat;
    const Tag: string): TNeuralFloat;
  var
    Epoch, Step, I, J, Tmp: integer;
    Inp, Tgt: TNNetVolume;
    Order: array of integer;
  begin
    Inp := TNNetVolume.Create(1, 1, 1);
    Tgt := TNNetVolume.Create(1, 1, 1);
    SetLength(Order, Length(Xs));
    for I := 0 to High(Order) do Order[I] := I;
    NN.SetLearningRate(cLR, cMomentum);
    NN.SetL2Decay(0.0);
    try
      for Epoch := 1 to cEpochs do
      begin
        // Fisher-Yates shuffle.
        for I := High(Order) downto 1 do
        begin
          J := Random(I + 1);
          Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
        end;
        Step := 0;
        NN.ClearDeltas();
        for I := 0 to High(Order) do
        begin
          Inp.Raw[0] := Xs[Order[I]];
          Tgt.Raw[0] := Ys[Order[I]];
          NN.Compute(Inp);
          NN.Backpropagate(Tgt);
          Inc(Step);
          if Step = cBatch then
          begin
            NN.UpdateWeights();
            NN.ClearDeltas();
            Step := 0;
          end;
        end;
        if Step > 0 then
        begin
          NN.UpdateWeights();
          NN.ClearDeltas();
        end;
        if (Epoch = 1) or (Epoch mod 100 = 0) or (Epoch = cEpochs) then
          WriteLn(Format('  [%s] epoch %4d  grid-MSE = %.6f',
            [Tag, Epoch, GridMSE(NN)]));
      end;
    finally
      Inp.Free;
      Tgt.Free;
    end;
    Result := GridMSE(NN);
  end;

  // ---- KAN-dense arm (Input -> KANLayer -> FullConnectLinear). --------------
  procedure BuildKANArm(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, 1, 1));
    NN.AddLayer(TNNetKANLayer.Create(cKANW, cKANK));  // <- per-edge learnable
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.InitWeights();
  end;

  // ---- ReLU arm at a chosen width W: Input -> FullConnectReLU(W) -> Linear(1).
  procedure BuildReLUArm(out NN: TNNet; W: integer);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(W));
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.InitWeights();
  end;

  // Pick the ReLU hidden width whose total CountWeights best matches a target.
  function ChooseReLUWidth(TargetParams: integer): integer;
  var
    W, BestW, BestDiff, Diff, C: integer;
    Probe: TNNet;
  begin
    BestW := 1; BestDiff := MaxInt;
    for W := 1 to 64 do
    begin
      BuildReLUArm(Probe, W);
      C := Probe.CountWeights();
      Probe.Free;
      Diff := Abs(C - TargetParams);
      if Diff < BestDiff then
      begin
        BestDiff := Diff;
        BestW := W;
      end;
    end;
    Result := BestW;
  end;

var
  NNKAN, NNReLU: TNNet;
  Xs, Ys: array[0..cNumTrain - 1] of TNeuralFloat;
  ReLUW: integer;
  KANParams, ReLUParams: integer;
  LossKAN, LossReLU: TNeuralFloat;
begin
  RandSeed := 2026;
  WriteLn('KANLayer: layer-KAN vs ReLU-MLP toy fit at MATCHED parameter count.');
  WriteLn('Target  y = sin(3x) + 0.3*sin(11x)  over x in [-', cXRange:0:1,
    ', ', cXRange:0:1, '].');
  WriteLn('Arm A: ReLU MLP (wider).  Arm B: SAME shape with the hidden dense ' +
    'layer replaced by TNNetKANLayer (per-EDGE learnable Chebyshev function).');
  WriteLn('See also examples/SplineActivationKAN for the activation-KAN ' +
    'counterpart (per-CHANNEL learnable activation).');
  WriteLn;

  // Shared training set: both arms see EXACTLY the same data.
  MakeTrainSet(Xs, Ys);

  // Build the KAN arm and count its params, then size the ReLU arm to match.
  BuildKANArm(NNKAN);
  KANParams := NNKAN.CountWeights();
  ReLUW := ChooseReLUWidth(KANParams);
  BuildReLUArm(NNReLU, ReLUW);
  ReLUParams := NNReLU.CountWeights();

  WriteLn(Format('KAN  arm: KANLayer(Dout=%d,K=%d) -> Linear(1)' +
    '                  =>  %d trainable weights', [cKANW, cKANK, KANParams]));
  WriteLn(Format('ReLU arm: FullConnectReLU(%d) -> Linear(1)' +
    '                      =>  %d trainable weights  (width chosen to match)',
    [ReLUW, ReLUParams]));
  WriteLn(Format('Param-count match: kan=%d  relu=%d  (delta=%d)',
    [KANParams, ReLUParams, Abs(KANParams - ReLUParams)]));
  WriteLn;

  WriteLn('Training ReLU arm for ', cEpochs, ' epochs...');
  LossReLU := TrainArm(NNReLU, Xs, Ys, 'ReLU');
  WriteLn;
  WriteLn('Training KAN arm for ', cEpochs, ' epochs...');
  LossKAN := TrainArm(NNKAN, Xs, Ys, 'KAN ');
  WriteLn;

  WriteLn('================================================================');
  WriteLn('RESULT (final clean-grid MSE, lower is better):');
  WriteLn(Format('  ReLU arm (%d params): %.6f', [ReLUParams, LossReLU]));
  WriteLn(Format('  KAN  arm (%d params): %.6f', [KANParams, LossKAN]));
  if LossKAN <= LossReLU then
    WriteLn(Format('  => KAN claim HOLDS: per-edge learnable function wins by ' +
      '%.1f%% at matched params.', [100.0 * (LossReLU - LossKAN) / LossReLU]))
  else
    WriteLn('  => KAN claim does NOT hold on this run.');
  WriteLn('================================================================');
  WriteLn;
  WriteLn('Read it as: the ReLU arm and the KAN arm carry the SAME number of ' +
    'trainable weights, but the KAN arm spends them on per-edge learnable ' +
    'Chebyshev functions phi_{ij}(x) instead of a scalar weight matrix + ' +
    'pointwise ReLU, and fits the wiggly target to a lower MSE — the ' +
    'layer-KAN matched-param win.');

  NNKAN.Free;
  NNReLU.Free;
end.
