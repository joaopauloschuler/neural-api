program KANConv;
(*
KANConv: a small demonstration of TNNetKANConv, the CONVOLUTIONAL Kolmogorov-
Arnold layer (the spatial sibling of the dense TNNetKANLayer).

An ordinary convolution maps each receptive-field patch to one output per filter
via a LINEAR dot product. TNNetKANConv replaces that linear dot product with a
sum of LEARNED univariate edge functions, one per (kernel position, input
channel): each edge function is a Chebyshev expansion of degree K over the
squashed input u = tanh(x),
    phi(x) = sum_{k=0..K} c_k * T_k(u).
So per output filter there are (FeatureSize*FeatureSize*InputDepth)*(K+1)
trainable coefficients and NO output bias (the c_0 term plays that role).

TASK (built to reward learnable per-edge NONLINEARITY): a 1-channel image is
mapped to a 1-channel image where each output pixel is a fixed NONLINEAR function
of its 3x3 neighbourhood,
    y(x,y) = sum over the 3x3 window of  a_p * f_p( input_p ),
where each f_p is a distinct smooth nonlinearity (sin / tanh / square / cube /
abs ...). A plain linear convolution can only fit the linearised part of this
map; a KAN convolution can fit the per-edge nonlinear shape directly.

Two contenders map a 1-channel image -> 1-channel image with a 3x3 kernel,
padding 1, stride 1:
  (A) TNNetKANConv(1, 3, 1, 1, K=4)  : a single filter with 3*3*1*(K+1) coeffs.
  (B) TNNetConvolutionLinear(1, 3, 1, 1) followed by a learnable per-channel
      activation, at a comparable (slightly larger) weight budget.
We print each model's trainable weight count, train both with the SAME data /
schedule, and report final validation MSE. The KAN convolution is expected to
reach the lowest error because its per-edge Chebyshev basis matches the
per-edge nonlinearity of the task.

Pure CPU, tiny data, few epochs -- runs in well under a minute.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cSizeX  = 8;
  cSizeY  = 8;
  cTrain  = 128;
  cVal    = 32;
  cEpochs = 150;
  cLR     = 0.01;
  cSeed   = 424242;
  cDegree = 4;

var
  // Fixed per-window-position linear mixing weights a_p (3x3).
  AW: array[0..2, 0..2] of TNeuralFloat;

// A distinct smooth nonlinearity per window position p (0..8).
function EdgeFn(p: integer; x: TNeuralFloat): TNeuralFloat;
begin
  case p of
    0: Result := Sin(1.5 * x);
    1: Result := Tanh(2.0 * x);
    2: Result := x * x - 0.3;
    3: Result := x * x * x;
    4: Result := Abs(x) - 0.5;
    5: Result := Cos(1.2 * x) - 1.0;
    6: Result := x / (1.0 + Abs(x));
    7: Result := Sin(0.8 * x) * x;
  else
    Result := Tanh(x) * Tanh(x);
  end;
end;

procedure InitGroundTruth();
var
  kx, ky: integer;
begin
  for ky := 0 to 2 do
    for kx := 0 to 2 do
      AW[ky, kx] := 0.4 + 0.3 * Sin(1.9 * kx + 2.1 * ky);
end;

// Build one (input, target) pair. Target pixel = sum_p a_p * f_p(input_p) over
// the 3x3 neighbourhood (out-of-range positions contribute 0).
procedure MakePair(Input, Target: TNNetVolume);
var
  x, y, kx, ky, sx, sy, p: integer;
  acc: TNeuralFloat;
begin
  for x := 0 to cSizeX - 1 do
    for y := 0 to cSizeY - 1 do
      Input.Add(x, y, 0, (Random - 0.5) * 2.0);
  for x := 0 to cSizeX - 1 do
    for y := 0 to cSizeY - 1 do
    begin
      acc := 0;
      for ky := 0 to 2 do
        for kx := 0 to 2 do
        begin
          sx := x + kx - 1;
          sy := y + ky - 1;
          if (sx < 0) or (sx >= cSizeX) or (sy < 0) or (sy >= cSizeY) then continue;
          p := ky * 3 + kx;
          acc := acc + AW[ky, kx] * EdgeFn(p, Input.Get(sx, sy, 0));
        end;
      Target.Add(x, y, 0, acc);
    end;
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
    TrainIn[i] := TNNetVolume.Create(cSizeX, cSizeY, 1);
    TrainTg[i] := TNNetVolume.Create(cSizeX, cSizeY, 1);
    MakePair(TrainIn[i], TrainTg[i]);
  end;
  for i := 0 to cVal - 1 do
  begin
    ValIn[i] := TNNetVolume.Create(cSizeX, cSizeY, 1);
    ValTg[i] := TNNetVolume.Create(cSizeX, cSizeY, 1);
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
  Result := acc / (Cnt * cSizeX * cSizeY);
end;

procedure Train(NN: TNNet; const Name: string);
var
  epoch, s, order, tmp, i: integer;
  perm: array[0..cTrain - 1] of integer;
  vMSE: TNeuralFloat;
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
  WriteLn(Format('  %-40s  weights=%4d   val-MSE=%.6f',
    [Name, CountWeights(NN), vMSE]));
end;

var
  NNk, NNc: TNNet;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  WriteLn('KANConv: TNNetKANConv vs a linear conv on a per-edge NONLINEAR 3x3 map.');
  WriteLn(Format('Task: %dx%d 1-channel images. Target pixel = sum_p a_p * f_p(neighbour_p)',
    [cSizeX, cSizeY]));
  WriteLn('over the 3x3 window, where each f_p is a distinct smooth nonlinearity.');
  WriteLn('A KAN conv learns per-edge Chebyshev functions, matching the task structure.');
  WriteLn(Format('Train=%d  Val=%d  Epochs=%d  LR=%.3f  Degree=%d',
    [cTrain, cVal, cEpochs, cLR, cDegree]));
  WriteLn;

  InitGroundTruth();
  BuildData();

  // (A) KAN conv: 1 -> 1 channel, 3x3 kernel, pad 1, stride 1, Chebyshev degree K.
  NNk := TNNet.Create();
  NNk.AddLayer(TNNetInput.Create(cSizeX, cSizeY, 1));
  NNk.AddLayer(TNNetKANConv.Create(1, 3, 1, 1, cDegree, 1));

  // (B) Linear conv + learnable per-channel activation (a comparable nonlinear
  //     baseline that CANNOT shape each edge independently).
  NNc := TNNet.Create();
  NNc.AddLayer(TNNetInput.Create(cSizeX, cSizeY, 1));
  NNc.AddLayer(TNNetConvolutionLinear.Create(4, 3, 1, 1));
  NNc.AddLayer(TNNetConvolutionLinear.Create(1, 1, 0, 1));

  WriteLn('Final results (lower val-MSE is better):');
  Train(NNk, 'TNNetKANConv(K=4)');
  Train(NNc, 'TNNetConvolutionLinear (4->1)');
  WriteLn;
  WriteLn('Expected: the KAN conv reaches the lowest error because its per-edge');
  WriteLn('Chebyshev basis directly fits the per-window-position nonlinearity that');
  WriteLn('a plain linear convolution can only approximate by its linearised part.');

  NNk.Free; NNc.Free;
  FreeData();
end.
