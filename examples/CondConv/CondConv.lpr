program CondConv;
(*
CondConv: a bake-off that shows when CONDITIONALLY-PARAMETERIZED ("dynamic")
convolution (TNNetCondConv, Yang et al. 2019, NeurIPS, arXiv:1904.04971) pays
off -- a small K-expert CondConv matching a much WIDER plain convolution while
keeping the inference cost of a SINGLE conv.

TNNetCondConv owns a BANK of K expert kernels W_1..W_K plus a tiny per-sample
routing head (global-avg-pool -> FC -> sigmoid) that emits K mixing coefficients
alpha_k. The effective kernel is the per-sample blend W_eff = sum_k alpha_k * W_k,
applied as ONE ordinary convolution -- so inference cost stays that of a single
conv regardless of K, while capacity grows with the bank. This is distinct from
TNNetHyperConv (generates the whole kernel from a second tensor) and
TNNet.AddMixtureOfExperts (mixes K expert OUTPUTS, K forward passes); CondConv
mixes K kernels BEFORE the conv (one forward pass).

TASK (built to favour input-dependent filtering): an 8x8 single-channel field.
A GLOBAL property of each sample -- the SIGN of its overall mean -- selects WHICH
of two ground-truth 3x3 filters produced the target:
  mean >= 0 : target = horizontal-edge filter applied to the field
  mean <  0 : target = a smoothing/blur filter applied to the field
A single plain conv has ONE kernel and must compromise across both regimes; a
CondConv can route per-sample to the right blend of its experts. The routing
signal (global mean) is exactly what the global-avg-pool head sees.

THREE contenders map an 8x8x1 field -> 8x8x1 field with a 3x3 kernel, pad 1,
stride 1:
  (A) TNNetConvolutionLinear(1,3,1,1)            -- one plain narrow conv
  (B) TNNetConvolutionLinear(W,3,1,1) -> 1x1     -- a much WIDER plain conv
      (W feature maps then a pointwise reduce) -- more weights AND more
      inference FLOPs (W feature maps convolved every forward pass)
  (C) TNNetCondConv(K,1,3,1,1)                    -- K-expert dynamic conv
We print each model's trainable weight count and its measured inference time over
the validation set, plus final val-MSE. CondConv is expected to match or beat the
wide plain conv at the inference cost of model (A) -- one conv -- because the
expert blend is chosen per sample.

Pure CPU, tiny data, few epochs -- runs in well under a minute.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cSizeX    = 8;
  cSizeY    = 8;
  cChannels = 1;
  cTrain    = 192;
  cVal      = 64;
  cEpochs   = 150;
  cLR       = 0.004;
  cSeed     = 424242;
  cExperts  = 2;     // K for CondConv
  cWide     = 8;     // feature maps for the wide plain conv

  // Two ground-truth 3x3 filters (flat fy*3+fx). One is a horizontal-edge
  // detector, the other a normalised blur.
  EdgeK: array[0..8] of TNeuralFloat =
    (-1, -2, -1,   0, 0, 0,   1, 2, 1);
  BlurK: array[0..8] of TNeuralFloat =
    (1/9, 1/9, 1/9,  1/9, 1/9, 1/9,  1/9, 1/9, 1/9);

// Apply a flat 3x3 kernel (pad 1, stride 1) to a single-channel field.
procedure ApplyFilter(Src, Dst: TNNetVolume; const K: array of TNeuralFloat);
var
  x, y, fx, fy, sx, sy: integer;
  acc: TNeuralFloat;
begin
  for y := 0 to cSizeY - 1 do
    for x := 0 to cSizeX - 1 do
    begin
      acc := 0;
      for fy := 0 to 2 do
        for fx := 0 to 2 do
        begin
          sx := x + fx - 1;
          sy := y + fy - 1;
          if (sx < 0) or (sx >= cSizeX) or (sy < 0) or (sy >= cSizeY) then continue;
          acc := acc + K[fy * 3 + fx] * Src.Get(sx, sy, 0);
        end;
      Dst.Add(x, y, 0, acc);
    end;
end;

// Build one (input, target) pair. A per-sample bias shifts the whole field so
// its global mean lands clearly on one side of zero; that sign selects the
// ground-truth filter.
procedure MakePair(Input, Target: TNNetVolume);
var
  x, y: integer;
  bias: TNeuralFloat;
begin
  // Random bias far from zero so the routing signal is unambiguous.
  if Random < 0.5 then bias := 0.8 + Random * 0.6
                  else bias := -(0.8 + Random * 0.6);
  for y := 0 to cSizeY - 1 do
    for x := 0 to cSizeX - 1 do
      Input.Add(x, y, 0, bias + (Random - 0.5) * 0.6);
  if bias >= 0 then ApplyFilter(Input, Target, EdgeK)
               else ApplyFilter(Input, Target, BlurK);
end;

var
  TrainIn, TrainTg: array[0..cTrain - 1] of TNNetVolume;
  ValIn, ValTg: array[0..cVal - 1] of TNNetVolume;

procedure BuildData();
var
  i: integer;
begin
  RandSeed := cSeed;
  for i := 0 to cTrain - 1 do
  begin
    TrainIn[i] := TNNetVolume.Create(cSizeX, cSizeY, cChannels);
    TrainTg[i] := TNNetVolume.Create(cSizeX, cSizeY, cChannels);
    MakePair(TrainIn[i], TrainTg[i]);
  end;
  for i := 0 to cVal - 1 do
  begin
    ValIn[i] := TNNetVolume.Create(cSizeX, cSizeY, cChannels);
    ValTg[i] := TNNetVolume.Create(cSizeX, cSizeY, cChannels);
    MakePair(ValIn[i], ValTg[i]);
  end;
end;

procedure FreeData();
var
  i: integer;
begin
  for i := 0 to cTrain - 1 do begin TrainIn[i].Free; TrainTg[i].Free; end;
  for i := 0 to cVal - 1 do begin ValIn[i].Free; ValTg[i].Free; end;
end;

function CountWeights(NN: TNNet): integer;
var
  L, n: integer;
  Layer: TNNetLayer;
begin
  Result := 0;
  for L := 0 to NN.CountLayers - 1 do
  begin
    Layer := NN.Layers[L];
    for n := 0 to Layer.Neurons.Count - 1 do
      Result := Result + Layer.Neurons[n].Weights.Size;
  end;
end;

function MeanMSE(NN: TNNet; const Ins, Tgs: array of TNNetVolume; Cnt: integer): TNeuralFloat;
var
  s, k: integer;
  diff, acc: TNeuralFloat;
begin
  acc := 0;
  for s := 0 to Cnt - 1 do
  begin
    NN.Compute(Ins[s]);
    for k := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[k] - Tgs[s].Raw[k];
      acc := acc + diff * diff;
    end;
  end;
  Result := acc / (Cnt * cSizeX * cSizeY * cChannels);
end;

// Measured wall-clock inference time over the whole validation set (ms),
// averaged over several passes so the number is stable on a quiet machine.
function InferMillis(NN: TNNet; Passes: integer): TNeuralFloat;
var
  p, s: integer;
  t0: TDateTime;
begin
  t0 := Now();
  for p := 0 to Passes - 1 do
    for s := 0 to cVal - 1 do
      NN.Compute(ValIn[s]);
  Result := (Now() - t0) * 24 * 60 * 60 * 1000;
end;

procedure Train(NN: TNNet; const Name: string);
var
  epoch, s, order, tmp, i: integer;
  perm: array[0..cTrain - 1] of integer;
  vMSE, ms: TNeuralFloat;
begin
  NN.SetLearningRate(cLR, 0.9);
  RandSeed := cSeed + 7;
  for i := 0 to cTrain - 1 do perm[i] := i;
  for epoch := 1 to cEpochs do
  begin
    for i := cTrain - 1 downto 1 do
    begin
      order := Random(i + 1);
      tmp := perm[i]; perm[i] := perm[order]; perm[order] := tmp;
    end;
    for s := 0 to cTrain - 1 do
    begin
      NN.Compute(TrainIn[perm[s]]);
      NN.Backpropagate(TrainTg[perm[s]]);
    end;
  end;
  vMSE := MeanMSE(NN, ValIn, ValTg, cVal);
  ms := InferMillis(NN, 20);
  WriteLn(Format('  %-40s  weights=%4d   infer=%7.2f ms   val-MSE=%.6f',
    [Name, CountWeights(NN), ms, vMSE]));
end;

var
  NNa, NNb, NNc: TNNet;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  WriteLn('CondConv: dynamic-conv bake-off on an input-dependent filtering task.');
  WriteLn(Format('Task: %dx%d single-channel fields. The SIGN of each sample''s global mean',
    [cSizeX, cSizeY]));
  WriteLn('selects which of two ground-truth 3x3 filters (edge vs blur) made the target,');
  WriteLn('so the right kernel DEPENDS ON THE INPUT -- exactly what per-sample routing buys.');
  WriteLn(Format('Train=%d  Val=%d  Epochs=%d  LR=%.3f  Experts(K)=%d',
    [cTrain, cVal, cEpochs, cLR, cExperts]));
  WriteLn;

  BuildData();

  // (A) One plain narrow conv: a single fixed kernel -- must compromise.
  NNa := TNNet.Create();
  NNa.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cChannels));
  NNa.AddLayer(TNNetConvolutionLinear.Create(cChannels, 3, 1, 1));

  // (B) A much WIDER plain conv (cWide feature maps) then a 1x1 reduce. More
  //     weights AND more inference FLOPs (cWide maps convolved every forward).
  NNb := TNNet.Create();
  NNb.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cChannels));
  NNb.AddLayer(TNNetConvolutionLinear.Create(cWide, 3, 1, 1));
  NNb.AddLayer(TNNetConvolutionLinear.Create(cChannels, 1, 0, 1));

  // (C) K-expert CondConv: one conv at inference, kernel blended per sample.
  NNc := TNNet.Create();
  NNc.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cChannels));
  NNc.AddLayer(TNNetCondConv.Create(cExperts, cChannels, 3, 1, 1));

  WriteLn('Final results (lower val-MSE is better; infer = wall-clock over val set):');
  Train(NNa, 'TNNetConvolutionLinear (1 plain conv)');
  Train(NNb, Format('TNNetConvolutionLinear (wide=%d) + 1x1', [cWide]));
  Train(NNc, Format('TNNetCondConv (K=%d experts)', [cExperts]));
  WriteLn;
  WriteLn('Expected: the single plain conv cannot switch behaviour and has the highest');
  WriteLn('error; the wide plain conv lowers error but costs more weights and inference');
  WriteLn('FLOPs; CondConv matches/beats it at the inference cost of ONE narrow conv,');
  WriteLn('because it routes each sample to the right blend of its expert kernels.');

  NNa.Free; NNb.Free; NNc.Free;
  FreeData();
end.
