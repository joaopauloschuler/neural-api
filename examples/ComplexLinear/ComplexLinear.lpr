program ComplexLinear;
(*
ComplexLinear: a PARAMETER-MATCHED bake-off that shows when the structured
weight sharing of TNNetComplexLinear (the 2-dimensional base rung of the same
Cayley-Dickson hypercomplex ladder as TNNetQuaternionLinear (4D) and
TNNetOctonionLinear (8D)) actually pays off.

TNNetComplexLinear reinterprets the input/output Depth (multiples of 2) as
packed COMPLEX numbers (group g holds Re=chan[2g], Im=chan[2g+1]) and learns an
(OutC x InC) grid of complex weights w = a + b i. The forward pass is the
complex product y = w . x, i.e. a real 2x2-block matrix where each learned
complex's 2 reals drive a whole 2x2 block
    M(w) = [[a,-b],[b,a]]  ->  Re' = a*Re - b*Im,  Im' = a*Im + b*Re.
So ONE complex's 2 reals control 4 matrix entries -> the layer stores ~1/2 the
weights of a dense layer of equal width while still mixing the real and
imaginary parts of every input complex.

TASK (built to favour complex coupling): each sample is X = 2 input complex
numbers (4 real channels). The TARGET multiplies, on the LEFT, each input
complex by the SAME fixed ground-truth complex g (a pure PHASE ROTATION plus
gain), then adds a small fixed complex-valued cross coupling between the two
complex numbers. This target is EXACTLY a complex-linear map, so a model whose
inductive bias matches the structure should win at equal parameter count.

THREE param-matched contenders map 4 -> 4 channels:
  (A) TNNetComplexLinear(4)           : OutC*InC*2 = 2*2*2 = 8 weights
  (B) TNNetFullConnectLinear bottleneck 4 -> 1 -> 4 : 4*1 + 1*4 = 8 weights
  (C) AddGroupedFullConnect(2 groups) : block-diagonal, 2*(2*2) = 8 weights
We report each model's trainable weight count so the comparison is honest, then
train all three with the SAME data / schedule and print final MSE. The complex
model is expected to reach the lowest error because its weight sharing is
exactly the symmetry of the task, at ~1/2 the weights of a full 4x4 dense
layer (16).

It then verifies the algebraic guarantee the layer is built on -- norm/phase
behaviour |w.X| = |w|.|X| (a single complex multiply scales magnitude by |w|
and rotates phase by arg(w)) -- exactly the way the OctonionLinear example
checks octonion norm multiplicativity.

Pure CPU, tiny data, few epochs -- runs in well under a minute.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cComplex  = 2;                  // input/output complex numbers
  cChannels = cComplex * 2;       // 4 real channels
  cTrain    = 256;
  cVal      = 64;
  cEpochs   = 200;
  cLR       = 0.02;
  cSeed     = 424242;

type
  TCpx = array[0..1] of TNeuralFloat;  // [0]=Re, [1]=Im

// Complex product w . x  ->  Re'=a*re-b*im, Im'=a*im+b*re.
function CMul(const w, x: TCpx): TCpx;
begin
  Result[0] := w[0] * x[0] - w[1] * x[1];
  Result[1] := w[0] * x[1] + w[1] * x[0];
end;

var
  gRot: TCpx;                                   // fixed ground-truth complex
  CrossMix: array[0..cComplex - 1, 0..cComplex - 1] of TCpx;

procedure InitGroundTruth();
var
  i, j: integer;
  angle, gain, n: TNeuralFloat;
begin
  // Ground-truth = a pure phase rotation of 50 degrees with gain 1.3.
  angle := 50.0 * Pi / 180.0;
  gain  := 1.3;
  gRot[0] := gain * Cos(angle);
  gRot[1] := gain * Sin(angle);
  n := Sqrt(gRot[0] * gRot[0] + gRot[1] * gRot[1]); // = gain
  for i := 0 to cComplex - 1 do
    for j := 0 to cComplex - 1 do
      if i = j then
      begin
        CrossMix[i, j, 0] := 0;
        CrossMix[i, j, 1] := 0;
      end
      else
      begin
        CrossMix[i, j, 0] := 0.12 * Cos(1.3 * (i + 1) * (j + 2));
        CrossMix[i, j, 1] := 0.12 * Sin(1.3 * (i + 1) * (j + 2));
      end;
  WriteLn(Format('Ground-truth complex g = %.4f + %.4f i  (|g|=%.4f, arg=%.1f deg)',
    [gRot[0], gRot[1], n, ArcTan2(gRot[1], gRot[0]) * 180.0 / Pi]));
end;

// Target = (g . x_i) + sum_{j<>i} CrossMix[i,j] . x_j.
procedure MakePair(Input, Target: TNNetVolume);
var
  i, j, c: integer;
  xi, xj, rot, contrib: TCpx;
begin
  for i := 0 to cChannels - 1 do
    Input.Raw[i] := (Random - 0.5) * 2.0;
  for i := 0 to cComplex - 1 do
  begin
    for c := 0 to 1 do xi[c] := Input.Raw[i * 2 + c];
    rot := CMul(gRot, xi);
    for c := 0 to 1 do Target.Raw[i * 2 + c] := rot[c];
    for j := 0 to cComplex - 1 do
      if j <> i then
      begin
        for c := 0 to 1 do xj[c] := Input.Raw[j * 2 + c];
        contrib := CMul(CrossMix[i, j], xj);
        for c := 0 to 1 do
          Target.Raw[i * 2 + c] := Target.Raw[i * 2 + c] + contrib[c];
      end;
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
    TrainIn[i] := TNNetVolume.Create(1, 1, cChannels);
    TrainTg[i] := TNNetVolume.Create(1, 1, cChannels);
    MakePair(TrainIn[i], TrainTg[i]);
  end;
  for i := 0 to cVal - 1 do
  begin
    ValIn[i] := TNNetVolume.Create(1, 1, cChannels);
    ValTg[i] := TNNetVolume.Create(1, 1, cChannels);
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
    for k := 0 to cChannels - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[k] - Tgs[s].Raw[k];
      acc := acc + diff * diff;
    end;
  end;
  Result := acc / (Cnt * cChannels);
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
  WriteLn(Format('  %-26s  weights=%4d   val-MSE=%.6f',
    [Name, CountWeights(NN), vMSE]));
end;

// Verify the norm/phase guarantee on the trained complex layer: for a single
// complex (InC=OutC=1, no bias) layer, |w.x| = |w|.|x| and arg(w.x) = arg(w)+arg(x).
procedure VerifyNormPhase();
var
  NN: TNNet;
  Input: TNNetVolume;
  CL: TNNetComplexLinear;
  trial: integer;
  a, b, re, im, nw, nx, ny, refNorm, maxNormErr, maxPhaseErr: TNeuralFloat;
  outRe, outIm, refPhase, outPhase, dPhase: TNeuralFloat;
begin
  RandSeed := cSeed + 99;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 2);
  maxNormErr := 0;
  maxPhaseErr := 0;
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 2, 1));
    CL := TNNetComplexLinear.Create(2, {SuppressBias}1);
    NN.AddLayer(CL);
    for trial := 0 to 9 do
    begin
      a  := (Random - 0.5) * 2.0;
      b  := (Random - 0.5) * 2.0;
      re := (Random - 0.5) * 2.0;
      im := (Random - 0.5) * 2.0;
      CL.Neurons[0].Weights.Raw[0] := a;
      CL.Neurons[0].Weights.Raw[1] := b;
      Input.Raw[0] := re;
      Input.Raw[1] := im;
      NN.Compute(Input);
      outRe := NN.GetLastLayer.Output.Raw[0];
      outIm := NN.GetLastLayer.Output.Raw[1];

      nw := Sqrt(a*a + b*b);
      nx := Sqrt(re*re + im*im);
      ny := Sqrt(outRe*outRe + outIm*outIm);
      refNorm := nw * nx;
      maxNormErr := Max(maxNormErr, Abs(ny - refNorm));

      refPhase := ArcTan2(b, a) + ArcTan2(im, re);
      outPhase := ArcTan2(outIm, outRe);
      dPhase := outPhase - refPhase;
      while dPhase >  Pi do dPhase := dPhase - 2*Pi;
      while dPhase < -Pi do dPhase := dPhase + 2*Pi;
      maxPhaseErr := Max(maxPhaseErr, Abs(dPhase));
    end;
    WriteLn(Format('Norm/phase check  |w.x|=|w||x| max-abs-error=%.2e ; '+
      'arg(w.x)=arg(w)+arg(x) max-abs-error=%.2e rad',
      [maxNormErr, maxPhaseErr]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

var
  NNc, NNd, NNg: TNNet;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  WriteLn('ComplexLinear: parameter-matched bake-off on a complex-product task.');
  WriteLn(Format('Task: left-multiply %d input complex numbers (%d channels) by a fixed',
    [cComplex, cChannels]));
  WriteLn('complex (phase rotation + gain) + small cross-complex coupling. Target is');
  WriteLn('EXACTLY a complex-linear map, so real/imag coupling should help.');
  WriteLn(Format('Train=%d  Val=%d  Epochs=%d  LR=%.3f', [cTrain, cVal, cEpochs, cLR]));
  WriteLn;

  InitGroundTruth();
  BuildData();

  // (A) Complex-linear: 4 -> 4, OutC*InC*2 = 8 weights.
  NNc := TNNet.Create();
  NNc.AddLayer(TNNetInput.Create(1, 1, cChannels));
  NNc.AddLayer(TNNetComplexLinear.Create(cChannels));

  // (B) Dense bottleneck 4 -> 1 -> 4 (a low-rank dense factorisation).
  NNd := TNNet.Create();
  NNd.AddLayer(TNNetInput.Create(1, 1, cChannels));
  NNd.AddLayer(TNNetFullConnectLinear.Create(1));
  NNd.AddLayer(TNNetFullConnectLinear.Create(cChannels));

  // (C) Grouped (block-diagonal) full connect: 2 groups, 4 -> 4.
  NNg := TNNet.Create();
  NNg.AddLayer(TNNetInput.Create(1, 1, cChannels));
  NNg.AddGroupedFullConnect(TNNetFullConnectLinear, 2, cChannels);

  WriteLn;
  WriteLn('Final results (lower val-MSE is better):');
  Train(NNc, 'TNNetComplexLinear');
  Train(NNd, 'TNNetFullConnectLinear(BN)');
  Train(NNg, 'AddGroupedFullConnect(2)');
  WriteLn;
  WriteLn('Expected: the complex layer reaches the lowest error because its weight');
  WriteLn('sharing matches the task symmetry, at ~1/2 the weights of a full 4x4');
  WriteLn('dense layer (16).');
  WriteLn;

  VerifyNormPhase();

  NNc.Free; NNd.Free; NNg.Free;
  FreeData();
end.
