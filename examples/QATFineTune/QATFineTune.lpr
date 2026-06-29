program QATFineTune;
(*
QATFineTune: quantization-aware training (QAT) demonstration.

This small, self-contained example shows the three accuracy points that
motivate QAT on a tiny synthetic image-classification task:

  1. FLOAT  - a small conv/dense classifier trained in full precision.
  2. PTQ    - the SAME trained weights, int8-quantized (TNNet.QuantizeWeightsInt8)
              with calibrated TNNetFakeQuantize activations but NO retraining.
              This drops accuracy versus float (quantization noise).
  3. QAT    - the same quantized topology, TRAINED with the fake-quant rounding
              in the forward pass (straight-through gradient) so the weights
              converge to a quantization-robust optimum and recover accuracy.

The dataset is generated synthetically (a low signal-to-noise 4-class image
task: each class is a faint bright patch in a class-specific quadrant buried
in strong additive noise) so the example downloads nothing and runs in
seconds. The low SNR keeps the float net at a moderate margin, which is the
regime where int8 quantization costs accuracy and QAT can recover it.
It is a DEMONSTRATION / smoke example, not a benchmark - it only needs to
show the QAT idea, not reach state of the art.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).

*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume,
  neuralfit;

const
  cImgSize    = 10;
  cNumClasses = 4;
  cTrainCount = 800;
  cTestCount  = 400;
  cNoise      = 2.2;  // moderate noise: keeps float accuracy at a tight margin.
  // Activation quantization step (qmax) for the TNNetFakeQuantize layers. We
  // use an AGGRESSIVE low-bit setting on purpose: at 8 bits (qmax = 127) the
  // calibrated fake-quant is nearly lossless on this small net, so there would
  // be almost no PTQ gap for QAT to recover. Coarse quantization makes the
  // post-training drop clearly visible and gives QAT (which trains the weights
  // to tolerate the rounding) real headroom to recover - the textbook low-bit
  // QAT story.
  cActQMax    = 5;    // very low-bit symmetric activations: levels in {-5..+5}

// Builds one synthetic cImgSize x cImgSize x 1 example for the given class.
procedure MakeSample(Vol: TNNetVolume; ClassId: integer);
var
  X, Y, Cx, Cy: integer;
  V: TNeuralFloat;
begin
  // Each class is identified by a faint bright 2x2 patch in a class-specific
  // quadrant; everything else (and the patch itself) is buried in strong
  // additive noise. The signal-to-noise ratio is low on purpose so the float
  // net only reaches a moderate margin - which is exactly the regime where
  // low-bit activation quantization (PTQ) costs accuracy and QAT can recover it.
  case ClassId of
    0: begin Cx := 1; Cy := 1; end;
    1: begin Cx := cImgSize - 3; Cy := 1; end;
    2: begin Cx := 1; Cy := cImgSize - 3; end;
  else begin Cx := cImgSize - 3; Cy := cImgSize - 3; end;
  end;
  for Y := 0 to cImgSize - 1 do
    for X := 0 to cImgSize - 1 do
    begin
      V := (Random - 0.5) * cNoise;                      // strong noise floor
      if (X >= Cx) and (X <= Cx + 1) and
         (Y >= Cy) and (Y <= Cy + 1) then
        V := V + 0.9;                                    // faint class signal
      Vol[X, Y, 0] := V;
    end;
end;

// Fills a pair list with Count random-class synthetic examples.
procedure BuildDataset(Pairs: TNNetVolumePairList; Count: integer);
var
  Cnt, ClassId: integer;
  InVol, OutVol: TNNetVolume;
begin
  for Cnt := 0 to Count - 1 do
  begin
    ClassId := Random(cNumClasses);
    InVol := TNNetVolume.Create(cImgSize, cImgSize, 1);
    OutVol := TNNetVolume.Create(cNumClasses, 1, 1);
    MakeSample(InVol, ClassId);
    OutVol.Fill(0);
    OutVol[ClassId, 0, 0] := 1;
    Pairs.Add(TNNetVolumePair.Create(InVol, OutVol));
  end;
end;

// Classification accuracy (argmax) of NN over the pair list.
function Accuracy(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  Cnt, Hits, Total: integer;
  Pred: TNNetVolume;
begin
  Hits := 0;
  Total := Pairs.Count;
  Pred := TNNetVolume.Create(cNumClasses, 1, 1);
  try
    for Cnt := 0 to Total - 1 do
    begin
      NN.Compute(Pairs[Cnt].I);
      NN.GetOutput(Pred);
      if Pred.GetClass() = Pairs[Cnt].O.GetClass() then Inc(Hits);
    end;
  finally
    Pred.Free;
  end;
  if Total > 0 then Result := Hits / Total else Result := 0;
end;

// Builds the float classifier (no fake-quant layers).
function BuildFloatNet(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer([
    TNNetInput.Create(cImgSize, cImgSize, 1),
    TNNetConvolutionReLU.Create({Features=}8, {FeatureSize=}3, {Padding=}1, {Stride=}1),
    TNNetMaxPool.Create(2),
    TNNetFullConnectReLU.Create(16),
    TNNetFullConnectLinear.Create(cNumClasses),
    TNNetSoftMax.Create()
  ]);
end;

// Builds the QAT network: same backbone, but with TNNetFakeQuantize layers
// inserted on the activations so training sees quantization noise.
function BuildQATNet(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer([
    TNNetInput.Create(cImgSize, cImgSize, 1),
    TNNetConvolutionReLU.Create({Features=}8, {FeatureSize=}3, {Padding=}1, {Stride=}1),
    TNNetFakeQuantize.Create({pQMax=}cActQMax, {pMomentum=}0.90, {pRunningMaxAbs=}0.0, {pFrozen=}0),
    TNNetMaxPool.Create(2),
    TNNetFullConnectReLU.Create(16),
    TNNetFakeQuantize.Create({pQMax=}cActQMax, {pMomentum=}0.90, {pRunningMaxAbs=}0.0, {pFrozen=}0),
    TNNetFullConnectLinear.Create(cNumClasses),
    TNNetSoftMax.Create()
  ]);
end;

procedure RunAlgo();
var
  TrainPairs, TestPairs: TNNetVolumePairList;
  FloatNet, PTQNet, QATNet: TNNet;
  Fit: TNeuralFit;
  AccFloat, AccPTQ, AccQAT: TNeuralFloat;
  LayerCnt: integer;
begin
  RandSeed := 424242;
  WriteLn('QAT fine-tune demonstration (synthetic ', cNumClasses, '-class task)');
  WriteLn('Building synthetic dataset...');
  TrainPairs := TNNetVolumePairList.Create();
  TestPairs := TNNetVolumePairList.Create();
  BuildDataset(TrainPairs, cTrainCount);
  BuildDataset(TestPairs, cTestCount);

  // ---- 1. Train the float baseline -------------------------------------
  WriteLn('Training float baseline...');
  FloatNet := BuildFloatNet();
  Fit := TNeuralFit.Create();
  Fit.Verbose := false;
  Fit.MaxThreadNum := 1; // deterministic, reproducible across runs
  Fit.HideMessages();
  Fit.InitialLearningRate := 0.01;
  Fit.LearningRateDecay := 0;
  Fit.L2Decay := 0;
  Fit.Fit(FloatNet, TrainPairs, nil, nil, {batchsize=}32, {epochs=}30);
  Fit.Free;
  AccFloat := Accuracy(FloatNet, TestPairs);

  // ---- 2. Post-training quantization (PTQ) -----------------------------
  // Build the quantized network (int8 weights + TNNetFakeQuantize on the
  // activations) and warm-start it from the trained float weights, but do
  // NOT retrain. We only CALIBRATE the activation observers by running the
  // training data through with every trainable layer's LearningRate set to 0
  // (the SGD update delta = -lr*grad is exactly zero, so weights never move
  // while the fake-quant observers populate their running max-abs). Then we
  // freeze the observers and int8-quantize the weights. This is the standard
  // PTQ baseline: quantization applied AFTER training, with no fine-tuning.
  WriteLn('Calibrating + applying post-training quantization (PTQ)...');
  PTQNet := BuildQATNet();
  PTQNet.Layers[1].CopyWeights(FloatNet.Layers[1]); // ConvolutionReLU
  PTQNet.Layers[4].CopyWeights(FloatNet.Layers[3]); // FullConnectReLU
  PTQNet.Layers[6].CopyWeights(FloatNet.Layers[4]); // FullConnectLinear
  for LayerCnt := 0 to PTQNet.GetLastLayerIdx() do
    PTQNet.Layers[LayerCnt].LearningRate := 0;        // freeze ALL weights
  Fit := TNeuralFit.Create();
  Fit.Verbose := false;
  Fit.MaxThreadNum := 1; // deterministic, reproducible across runs
  Fit.HideMessages();
  Fit.InitialLearningRate := 0;                       // calibration only
  Fit.LearningRateDecay := 0;
  Fit.L2Decay := 0;
  Fit.Fit(PTQNet, TrainPairs, nil, nil, {batchsize=}32, {epochs=}2);
  Fit.Free;
  for LayerCnt := 0 to PTQNet.GetLastLayerIdx() do
    if PTQNet.Layers[LayerCnt].ClassType = TNNetFakeQuantize then
      TNNetFakeQuantize(PTQNet.Layers[LayerCnt]).Freeze();
  PTQNet.QuantizeWeightsInt8();
  AccPTQ := Accuracy(PTQNet, TestPairs);

  // ---- 3. QAT: insert fake-quant layers, train from scratch ------------
  // Same quantized topology, trained from scratch (NOT warm-started from the
  // float weights): the fake-quant rounding is in the forward pass and
  // gradients flow back through the straight-through estimator, so the weights
  // converge to a quantization-robust optimum rather than the float net's
  // sharp (quantization-sensitive) minimum. Observers calibrate during
  // training; we freeze them and int8-quantize the weights before measuring
  // inference accuracy.
  WriteLn('QAT fine-tuning with TNNetFakeQuantize activations...');
  QATNet := BuildQATNet();

  Fit := TNeuralFit.Create();
  Fit.Verbose := false;
  Fit.MaxThreadNum := 1; // deterministic, reproducible across runs
  Fit.HideMessages();
  Fit.InitialLearningRate := 0.01;
  Fit.LearningRateDecay := 0;
  Fit.L2Decay := 0;
  Fit.Fit(QATNet, TrainPairs, nil, nil, {batchsize=}32, {epochs=}100);
  Fit.Free;

  // Freeze the observers, then int8-quantize the weights for inference.
  for LayerCnt := 0 to QATNet.GetLastLayerIdx() do
    if QATNet.Layers[LayerCnt].ClassType = TNNetFakeQuantize then
      TNNetFakeQuantize(QATNet.Layers[LayerCnt]).Freeze();
  QATNet.QuantizeWeightsInt8();
  AccQAT := Accuracy(QATNet, TestPairs);

  // ---- Report ----------------------------------------------------------
  WriteLn('');
  WriteLn('================ QAT demonstration results ================');
  WriteLn('Stage                         Test accuracy');
  WriteLn('-----------------------------------------------------------');
  WriteLn('FLOAT (full precision)         ', (AccFloat * 100):6:2, ' %');
  WriteLn('PTQ   (low-bit acts, no QAT)   ', (AccPTQ   * 100):6:2, ' %');
  WriteLn('QAT   (low-bit acts, fine-tune)', (AccQAT   * 100):6:2, ' %');
  WriteLn('===========================================================');
  WriteLn('PTQ drop vs float : ', ((AccPTQ - AccFloat) * 100):6:2, ' pts');
  WriteLn('QAT vs PTQ        : ', ((AccQAT - AccPTQ) * 100):6:2, ' pts');

  QATNet.Free;
  PTQNet.Free;
  FloatNet.Free;
  TestPairs.Free;
  TrainPairs.Free;
end;

var
  Application: record Title: string; end;

begin
  Application.Title := 'QAT Fine-Tune Example';
  RunAlgo();
end.
