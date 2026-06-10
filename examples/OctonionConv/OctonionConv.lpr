program OctonionConv;
(*
OctonionConv: a PARAMETER-MATCHED bake-off that shows when the structured weight
sharing of TNNetOctonionConv (the spatial sibling of TNNetOctonionLinear and the
8D analogue of TNNetQuaternionConv) pays off.

TNNetOctonionConv reinterprets the input/output Depth (multiples of 8) as packed
octonions and learns, for every kernel tap, an (OutO x InO) grid of octonion
weights W applied by the Cayley-Dickson product W . X = M(W) * X. So ONE
octonion's 8 reals control a whole 8x8 channel block at each spatial position ->
the layer stores ~1/8 of the weights of a real conv of the same input/output
width while still mixing all eight channel components.

TASK (built to favour 8-component spatial coupling): a small "octonion left-mul +
blur" image map. Each sample is a HxW image with 8 input channels (1 input
octonion per pixel). The TARGET applies, at every output pixel, the SAME
ground-truth octonion left-multiplication by a fixed octonion g to a small fixed-
weight spatial average of the 3x3 neighbourhood. This target is EXACTLY an
octonion convolution, so a model whose inductive bias matches the structure
should win at equal parameter count.

TWO param-matched contenders map an 8-channel image -> 8-channel image with a 3x3
kernel, padding 1, stride 1:
  (A) TNNetOctonionConv(8, 3, 1, 1) : OutO*InO*3*3*8 = 1*1*9*8 = 72 weights
  (B) TNNetConvolutionLinear with an 8 -> 1 -> 8 channel bottleneck so its weight
      count is comparable; we print both counts so the comparison is honest.
We report each model's trainable weight count, train both with the SAME data /
schedule, and print final MSE. The octonion model is expected to reach the lowest
error per parameter because its weight sharing is exactly the symmetry of the
task (a full 8x8-channel 3x3 real conv would need 9*8*8 = 576 weights).

Pure CPU, tiny data, few epochs -- runs in well under a minute.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cSizeX   = 8;
  cSizeY   = 8;
  cChannels = 8;            // one input octonion per pixel
  cTrain   = 128;
  cVal     = 32;
  cEpochs  = 120;
  cLR      = 0.01;
  cSeed    = 424242;

  // Octonion multiplication table (matches OCT_SGN / OCT_SRC inside
  // TNNetOctonionConv): product P . Q has component
  //   r[i] = sum_j SGN[i][j] * P[ i xor j ] * Q[j].
  OCT_SRC: array[0..7, 0..7] of integer =
  (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (1, 0, 3, 2, 5, 4, 7, 6),
    (2, 3, 0, 1, 6, 7, 4, 5),
    (3, 2, 1, 0, 7, 6, 5, 4),
    (4, 5, 6, 7, 0, 1, 2, 3),
    (5, 4, 7, 6, 1, 0, 3, 2),
    (6, 7, 4, 5, 2, 3, 0, 1),
    (7, 6, 5, 4, 3, 2, 1, 0)
  );
  OCT_SGN: array[0..7, 0..7] of TNeuralFloat =
  (
    ( 1, -1, -1, -1, -1, -1, -1, -1),
    ( 1,  1, -1,  1, -1,  1,  1, -1),
    ( 1,  1,  1, -1, -1, -1,  1,  1),
    ( 1, -1,  1,  1, -1,  1, -1,  1),
    ( 1,  1,  1,  1,  1, -1, -1, -1),
    ( 1, -1,  1, -1,  1,  1,  1, -1),
    ( 1, -1, -1,  1,  1, -1,  1,  1),
    ( 1,  1, -1, -1,  1,  1, -1,  1)
  );

type
  TOct = array[0..7] of TNeuralFloat;

// Octonion product p . q.
function OMul(const p, q: TOct): TOct;
var
  i, j: integer;
  acc: TNeuralFloat;
begin
  for i := 0 to 7 do
  begin
    acc := 0;
    for j := 0 to 7 do
      acc := acc + OCT_SGN[i, j] * p[OCT_SRC[i, j]] * q[j];
    Result[i] := acc;
  end;
end;

var
  gOct: TOct;                               // ground-truth fixed octonion
  Kern: array[0..2, 0..2] of TNeuralFloat;  // fixed 3x3 spatial average

procedure InitGroundTruth();
var
  c, kx, ky: integer;
  n, s: TNeuralFloat;
begin
  // A unit octonion: normalised mix of all eight components.
  gOct[0] := 0.6;
  for c := 1 to 7 do gOct[c] := Sin(0.7 * c + 0.3) * 0.4;
  n := 0;
  for c := 0 to 7 do n := n + gOct[c] * gOct[c];
  n := Sqrt(n);
  for c := 0 to 7 do gOct[c] := gOct[c] / n;
  // A small fixed (non-symmetric) spatial kernel, normalised to sum 1.
  s := 0;
  for ky := 0 to 2 do
    for kx := 0 to 2 do
    begin
      Kern[ky, kx] := 0.5 + 0.4 * Sin(1.7 * kx + 2.3 * ky);
      s := s + Kern[ky, kx];
    end;
  for ky := 0 to 2 do
    for kx := 0 to 2 do
      Kern[ky, kx] := Kern[ky, kx] / s;
end;

// Build one (input, target) pair. Target pixel = g . (spatial-average of the
// 3x3 neighbourhood of the input octonion field).
procedure MakePair(Input, Target: TNNetVolume);
var
  x, y, c, kx, ky, sx, sy: integer;
  acc, rot: TOct;
begin
  for x := 0 to cSizeX - 1 do
    for y := 0 to cSizeY - 1 do
      for c := 0 to cChannels - 1 do
        Input.Add(x, y, c, (Random - 0.5) * 2.0);
  for x := 0 to cSizeX - 1 do
    for y := 0 to cSizeY - 1 do
    begin
      for c := 0 to 7 do acc[c] := 0;
      for ky := 0 to 2 do
        for kx := 0 to 2 do
        begin
          sx := x + kx - 1;
          sy := y + ky - 1;
          if (sx < 0) or (sx >= cSizeX) or (sy < 0) or (sy >= cSizeY) then continue;
          for c := 0 to 7 do
            acc[c] := acc[c] + Kern[ky, kx] * Input.Get(sx, sy, c);
        end;
      rot := OMul(gOct, acc);
      for c := 0 to 7 do
        Target.Add(x, y, c, rot[c]);
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
  WriteLn(Format('  %-34s  weights=%4d   val-MSE=%.6f',
    [Name, CountWeights(NN), vMSE]));
end;

var
  NNo, NNc: TNNet;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  WriteLn('OctonionConv: parameter-matched bake-off on an octonion left-mul + blur task.');
  WriteLn(Format('Task: %dx%d images, %d channels (1 octonion/pixel). Target = fixed unit',
    [cSizeX, cSizeY, cChannels]));
  WriteLn('octonion left-multiplication of a fixed 3x3 spatial average. Target is EXACTLY');
  WriteLn('an octonion convolution, so 8-component spatial coupling should help.');
  WriteLn(Format('Train=%d  Val=%d  Epochs=%d  LR=%.3f', [cTrain, cVal, cEpochs, cLR]));
  WriteLn;

  InitGroundTruth();
  BuildData();

  // (A) Octonion conv: 8 -> 8 channels, 3x3 kernel, pad 1, stride 1.
  NNo := TNNet.Create();
  NNo.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cChannels));
  NNo.AddLayer(TNNetOctonionConv.Create(cChannels, 3, 1, 1));

  // (B) Real conv bottleneck 8 -> 1 -> 8 (a low-rank spatial factorisation at a
  //     comparable weight budget).
  NNc := TNNet.Create();
  NNc.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cChannels));
  NNc.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1));
  NNc.AddLayer(TNNetConvolutionLinear.Create(cChannels, 3, 1, 1));

  WriteLn('Final results (lower val-MSE is better):');
  Train(NNo, 'TNNetOctonionConv');
  Train(NNc, 'TNNetConvolutionLinear(bottleneck)');
  WriteLn;
  WriteLn('Expected: the octonion conv reaches the lowest error because its weight');
  WriteLn('sharing matches the task symmetry, at ~1/8 the weights of a full');
  WriteLn('8x8-channel 3x3 real conv (9*8*8 = 576).');

  NNo.Free; NNc.Free;
  FreeData();
end.
