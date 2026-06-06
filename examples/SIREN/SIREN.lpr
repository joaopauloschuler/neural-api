program SIREN;
(*
SIREN: a coordinate-MLP 1D fit reproducing the headline of Sitzmann et al. 2020,
"Implicit Neural Representations with Periodic Activation Functions" (SIREN), on
a tiny pure-CPU target using the existing TNNetSin periodic activation.

THE IDEA. An implicit neural representation maps a coordinate x to the signal
value f(x). Sitzmann et al. show that a plain ReLU (or Tanh) coordinate-MLP has
a strong SPECTRAL BIAS: it learns smooth, low-frequency content quickly but
fights to represent fine high-frequency detail. Swapping the activation for a
plain SINE (and initializing so the pre-activations land in the sine's useful
regime) removes that bias -- the same width/depth net fits high-frequency
content far better. The whole trick is the periodic activation plus its init.

TARGET (high-frequency 1D signal):
    y = sin(3x) + 0.3*sin(11x),  x in [-1, 1]
The 11x term is the fine detail a ReLU/Tanh MLP of this tiny size struggles to
reproduce while a SIREN of the SAME size nails it.

BAKE-OFF. The SAME architecture, width, depth, seed, epoch budget, batch size,
learning rate and optimizer is trained TWICE:
  Arm SIREN : Input(1) -> [FullConnectLinear(H) -> TNNetSin]xL -> FullConnectLinear(1)
  Arm TANH  : Input(1) -> [FullConnectLinear(H) -> TNNetHyperbolicTangent]xL -> FullConnectLinear(1)
Only the hidden activation differs. We report final dense-grid MSE for each; the
headline result is that the SIREN arm reaches a substantially LOWER MSE.

SIREN INITIALIZATION (this is what makes it actually work).
The paper's init: the FIRST layer weights ~ U(-1/n, 1/n) but the pre-activation
is then multiplied by a frequency omega_0 (~30) so the first sine sees inputs
spanning many periods; hidden layers ~ U(-sqrt(6/fan_in)/omega_0,
+sqrt(6/fan_in)/omega_0) so that each sine's pre-activation keeps a unit-ish
std across depth.

The library does not expose SIREN's exact init directly, but TNNetLayer.InitUniform(s)
sets every weight ~ U(-s, +s) (TVolume.InitUniform yields U(-1,1), then scales by s).
That is exactly the building block SIREN needs, so we reproduce the scheme by
hand AFTER InitWeights():
  * input dim n = 1, so the paper's first-layer U(-1/n, 1/n) = U(-1, 1); we fold
    the omega_0 frequency factor straight into the first layer by initializing it
    to U(-omega_0, +omega_0) -- i.e. InitUniform(omega_0). This is mathematically
    identical to "U(-1,1) weights, then multiply the pre-activation by omega_0",
    and needs no input rescaling or extra layer.
  * each hidden sine layer gets InitUniform(sqrt(6/fan_in)/omega_0).
We use omega_0 = 12 (a touch below the paper's 30; with x in [-1,1] and this
tiny net, 12 keeps the high-frequency advantage while staying numerically calm
and deterministic). The TANH arm uses the library's default InitWeights() (its
standard init for that activation) -- a fair, conventional baseline.

The bias terms are left at their InitWeights() values (0) for both arms.

Pure CPU, single-threaded, deterministic (fixed RandSeed). Runs in a few
seconds. The self-checking gate Halt(1)s on failure, mirroring the suite idiom
used by examples/DeepSets, examples/MaxBlurPool and examples/BitLinearBakeoff.

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
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cXRange   = 1.0;    // target sampled over x in [-cXRange, +cXRange]
  cNumTrain = 256;    // synthetic training samples
  cEpochs   = 400;    // same epoch budget for both arms
  cBatch    = 16;
  cLR       = 0.005;  // same optimizer settings for both arms
  cMomentum = 0.9;
  cHidden   = 24;     // hidden width (shared by both arms)
  cDepth    = 3;      // number of hidden (FullConnect + activation) blocks
  cOmega0   = 12.0;   // SIREN first-layer frequency factor
  cSeed     = 424242; // repo idiom

  // High-frequency target: the 0.3*sin(11x) term is the fine detail a tiny
  // Tanh MLP fights to represent while a SIREN of the same size captures it.
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

  // MSE on a dense clean grid (no noise) -- the honest measure of fit quality.
  function GridMSE(NN: TNNet): TNeuralFloat;
  var
    I: integer;
    X, Diff, Sum: TNeuralFloat;
    Inp, Outp: TNNetVolume;
  begin
    Inp := TNNetVolume.Create(1, 1, 1);
    Outp := TNNetVolume.Create(1, 1, 1);
    Sum := 0;
    for I := 0 to 399 do
    begin
      X := -cXRange + 2.0 * cXRange * I / 399.0;
      Inp.Raw[0] := X;
      NN.Compute(Inp);
      NN.GetOutput(Outp);
      Diff := Outp.Raw[0] - Target(X);
      Sum := Sum + Diff * Diff;
    end;
    Inp.Free;
    Outp.Free;
    Result := Sum / 400.0;
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

  // Build a coordinate-MLP. UseSin=True -> TNNetSin hidden activations (SIREN);
  // UseSin=False -> TNNetHyperbolicTangent (the conventional baseline arm).
  // Same width/depth/param-count for both; only the activation class differs.
  procedure BuildArm(out NN: TNNet; UseSin: boolean);
  var
    L: integer;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, 1, 1));
    for L := 1 to cDepth do
    begin
      NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));
      if UseSin then
        NN.AddLayer(TNNetSin.Create())
      else
        NN.AddLayer(TNNetHyperbolicTangent.Create());
    end;
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.InitWeights();
  end;

  // Reproduce SIREN's init on top of the default InitWeights(). Walk the layers;
  // the FullConnectLinear feeding the FIRST sine gets U(-omega_0, +omega_0)
  // (input dim n=1, folding the omega_0 frequency into the weights), every later
  // FullConnectLinear feeding a sine gets U(-sqrt(6/fan_in)/omega_0, +...).
  // The final linear read-out keeps its default init.
  procedure ApplySirenInit(NN: TNNet);
  var
    Idx, FanIn: integer;
    IsFirstSine: boolean;
    Scale: TNeuralFloat;
  begin
    IsFirstSine := True;
    for Idx := 0 to NN.GetLastLayerIdx() do
    begin
      // A sine activation directly follows the FullConnectLinear we must init.
      if (NN.Layers[Idx] is TNNetSin) and (Idx >= 1) then
      begin
        FanIn := NN.Layers[Idx - 1].PrevLayer.Output.Size; // fan-in of that FC
        if IsFirstSine then
        begin
          NN.Layers[Idx - 1].InitUniform(cOmega0);
          IsFirstSine := False;
        end
        else
        begin
          Scale := Sqrt(6.0 / FanIn) / cOmega0;
          NN.Layers[Idx - 1].InitUniform(Scale);
        end;
      end;
    end;
  end;

var
  NNSiren, NNTanh: TNNet;
  Xs, Ys: array[0..cNumTrain - 1] of TNeuralFloat;
  LossSiren, LossTanh, Improvement: TNeuralFloat;
  Pass: boolean;
const
  cGateRatio = 0.5;   // SIREN MSE must be < half the Tanh MSE to PASS
begin
  RandSeed := cSeed;  // deterministic; manual Compute/Backpropagate run single-threaded

  WriteLn('SIREN: periodic-activation coordinate-MLP 1D fit (Sitzmann et al. 2020).');
  WriteLn(Format('Target  y = sin(3x) + 0.3*sin(11x)  over x in [-%.1f, %.1f].',
    [cXRange, cXRange]));
  WriteLn(Format('Net (both arms): Input(1) -> [FullConnectLinear(%d) -> act]x%d'
    + ' -> FullConnectLinear(1)', [cHidden, cDepth]));
  WriteLn('  SIREN arm act = TNNetSin (omega_0 = ', cOmega0:0:1,
    ', SIREN init);  baseline arm act = TNNetHyperbolicTangent (default init).');
  WriteLn;

  // Shared training set: both arms see EXACTLY the same data.
  MakeTrainSet(Xs, Ys);

  // ---- baseline (Tanh) arm: same geometry, conventional activation + init. ---
  BuildArm(NNTanh, {UseSin=}False);
  WriteLn('Training baseline TANH arm for ', cEpochs, ' epochs...');
  LossTanh := TrainArm(NNTanh, Xs, Ys, 'TANH');
  WriteLn;

  // ---- SIREN arm: same geometry, sine activation + SIREN init. ---------------
  BuildArm(NNSiren, {UseSin=}True);
  ApplySirenInit(NNSiren);
  WriteLn('Training SIREN arm for ', cEpochs, ' epochs...');
  LossSiren := TrainArm(NNSiren, Xs, Ys, 'SINE');
  WriteLn;

  WriteLn('================================================================');
  WriteLn('RESULT (final dense-grid MSE, lower is better):');
  WriteLn(Format('  TANH  baseline arm : %.6f', [LossTanh]));
  WriteLn(Format('  SIREN (TNNetSin)   : %.6f', [LossSiren]));
  if LossTanh > 0 then
  begin
    Improvement := 100.0 * (LossTanh - LossSiren) / LossTanh;
    WriteLn(Format('  SIREN reduces MSE by %.1f%% vs the Tanh baseline.',
      [Improvement]));
  end;
  WriteLn('================================================================');
  WriteLn;

  // ---- SELF-CHECKING GATE ----------------------------------------------------
  // SIREN's headline is that periodic activations fit high-frequency detail far
  // better; require its MSE be MEANINGFULLY lower (< half) the Tanh baseline.
  Pass := (LossSiren < cGateRatio * LossTanh) and (LossSiren < 0.05);
  if Pass then
    WriteLn('GATE: PASS - the periodic-activation (SIREN) arm fits the ' +
      'high-frequency target substantially better than the Tanh baseline.')
  else
  begin
    WriteLn('GATE: FAIL');
    if not (LossSiren < cGateRatio * LossTanh) then
      WriteLn(Format('  - SIREN MSE %.6f not < %.1f%% of Tanh MSE %.6f.',
        [LossSiren, 100.0 * cGateRatio, LossTanh]));
    if not (LossSiren < 0.05) then
      WriteLn(Format('  - SIREN MSE %.6f did not reach the absolute fit bar 0.05.',
        [LossSiren]));
  end;

  NNSiren.Free;
  NNTanh.Free;

  if not Pass then Halt(1);
end.
