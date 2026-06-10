program QuaternionConv;
(*
QuaternionConv: a PARAMETER-MATCHED bake-off that shows when the structured
weight sharing of TNNetQuaternionConv (the spatial sibling of
TNNetQuaternionLinear) pays off.

TNNetQuaternionConv reinterprets the input/output Depth (multiples of 4) as
packed quaternions and learns, for every kernel tap, an (OutQ x InQ) grid of
quaternion weights q = r + x i + y j + z k applied by the Hamilton product
M(q)*(a,b,c,d). So ONE quaternion's 4 reals control a whole 4x4 channel block at
each spatial position -> the layer stores ~1/4 of the weights of a real conv of
the same input/output width while still mixing all four channel components.

TASK (built to favour 4-component spatial coupling): a small "colour-rotation +
blur" image map. Each sample is a HxW image with 4 input channels (1 input
quaternion per pixel: think RGB + alpha packed as a quaternion). The TARGET
applies, at every output pixel, the SAME ground-truth Hamilton rotation by a
fixed unit quaternion g to a small fixed-weight spatial average of the 3x3
neighbourhood. This target is EXACTLY a quaternion convolution, so a model whose
inductive bias matches the structure should win at equal parameter count.

TWO param-matched contenders map a 4-channel image -> 4-channel image with a 3x3
kernel, padding 1, stride 1:
  (A) TNNetQuaternionConv(4, 3, 1, 1) : OutQ*InQ*3*3*4 = 1*1*9*4 = 36 weights
  (B) TNNetConvolutionLinear with a 4 -> 1 -> 4 channel bottleneck so its weight
      count is comparable; we print both counts so the comparison is honest.
We report each model's trainable weight count, train both with the SAME data /
schedule, and print final MSE. The quaternion model is expected to reach the
lowest error per parameter because its weight sharing is exactly the symmetry of
the task.

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
  cChannels = 4;            // one input quaternion per pixel
  cTrain   = 128;
  cVal     = 32;
  cEpochs  = 120;
  cLR      = 0.01;
  cSeed    = 424242;

type
  TQuat = array[0..3] of TNeuralFloat;

// Hamilton product p (x) q.
function QMul(const p, q: TQuat): TQuat;
begin
  Result[0] := p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3];
  Result[1] := p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2];
  Result[2] := p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1];
  Result[3] := p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0];
end;

var
  gRot: TQuat;                       // ground-truth fixed rotation quaternion
  Kern: array[0..2, 0..2] of TNeuralFloat;  // fixed 3x3 spatial average

procedure InitGroundTruth();
var
  c, kx, ky: integer;
  n, s: TNeuralFloat;
begin
  // A unit quaternion (cos t, sin t * axis): a real 3D rotation of the
  // imaginary part.
  gRot[0] := Cos(0.6);
  gRot[1] := Sin(0.6) * 0.3;
  gRot[2] := Sin(0.6) * 0.9;
  gRot[3] := Sin(0.6) * 0.2;
  n := Sqrt(gRot[0]*gRot[0] + gRot[1]*gRot[1] + gRot[2]*gRot[2] + gRot[3]*gRot[3]);
  for c := 0 to 3 do gRot[c] := gRot[c] / n;
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

// Build one (input, target) pair. Target pixel = g (x) (spatial-average of the
// 3x3 neighbourhood of the input quaternion field).
procedure MakePair(Input, Target: TNNetVolume);
var
  x, y, c, kx, ky, sx, sy: integer;
  acc, rot: TQuat;
begin
  for x := 0 to cSizeX - 1 do
    for y := 0 to cSizeY - 1 do
      for c := 0 to cChannels - 1 do
        Input.Add(x, y, c, (Random - 0.5) * 2.0);
  for x := 0 to cSizeX - 1 do
    for y := 0 to cSizeY - 1 do
    begin
      for c := 0 to 3 do acc[c] := 0;
      for ky := 0 to 2 do
        for kx := 0 to 2 do
        begin
          sx := x + kx - 1;
          sy := y + ky - 1;
          if (sx < 0) or (sx >= cSizeX) or (sy < 0) or (sy >= cSizeY) then continue;
          for c := 0 to 3 do
            acc[c] := acc[c] + Kern[ky, kx] * Input.Get(sx, sy, c);
        end;
      rot := QMul(gRot, acc);
      for c := 0 to 3 do
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
  WriteLn(Format('  %-30s  weights=%4d   val-MSE=%.6f',
    [Name, CountWeights(NN), vMSE]));
end;

var
  NNq, NNc: TNNet;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  WriteLn('QuaternionConv: parameter-matched bake-off on a colour-rotation + blur task.');
  WriteLn(Format('Task: %dx%d images, %d channels (1 quaternion/pixel). Target = fixed unit',
    [cSizeX, cSizeY, cChannels]));
  WriteLn('quaternion rotation of a fixed 3x3 spatial average. Target is EXACTLY a');
  WriteLn('quaternion convolution, so 4-component spatial coupling should help.');
  WriteLn(Format('Train=%d  Val=%d  Epochs=%d  LR=%.3f', [cTrain, cVal, cEpochs, cLR]));
  WriteLn;

  InitGroundTruth();
  BuildData();

  // (A) Quaternion conv: 4 -> 4 channels, 3x3 kernel, pad 1, stride 1.
  NNq := TNNet.Create();
  NNq.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cChannels));
  NNq.AddLayer(TNNetQuaternionConv.Create(cChannels, 3, 1, 1));

  // (B) Real conv bottleneck 4 -> 1 -> 4 (a low-rank spatial factorisation at a
  //     comparable weight budget).
  NNc := TNNet.Create();
  NNc.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cChannels));
  NNc.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1));
  NNc.AddLayer(TNNetConvolutionLinear.Create(cChannels, 3, 1, 1));

  WriteLn('Final results (lower val-MSE is better):');
  Train(NNq, 'TNNetQuaternionConv');
  Train(NNc, 'TNNetConvolutionLinear(bottleneck)');
  WriteLn;
  WriteLn('Expected: the quaternion conv reaches the lowest error because its');
  WriteLn('weight sharing matches the task symmetry, at ~1/4 the weights of a');
  WriteLn('full 4x4-channel 3x3 real conv (9*4*4 = 144).');

  NNq.Free; NNc.Free;
  FreeData();
end.
