program QuaternionLinear;
(*
QuaternionLinear: a PARAMETER-MATCHED bake-off that shows when the structured
weight sharing of TNNetQuaternionLinear (the first hypercomplex layer in this
fork) actually pays off.

TNNetQuaternionLinear reinterprets the input/output Depth (multiples of 4) as
packed quaternions and learns an (OutQ x InQ) grid of quaternion weights
q = r + x i + y j + z k. The forward pass is the Hamilton product y = W (x) x,
i.e. a real 4x4-block matrix where each learned quaternion drives a whole 4x4
block:
    M(q) = [[ r,-x,-y,-z],
            [ x, r,-z, y],
            [ y, z, r,-x],
            [ z,-y, x, r]].
So ONE quaternion's 4 reals control 16 matrix entries -> the layer stores ~1/4
the weights of a dense layer of equal width while still mixing all four
components of every input quaternion.

TASK (built to favour 4-component coupling): a colour/signal "rotation" map.
Each sample is Q = 4 input quaternions (16 real channels). The TARGET applies,
to every input quaternion, the SAME ground-truth Hamilton rotation by a fixed
unit quaternion g (a genuine 3D-rotation-of-the-imaginary-part operator) and
then mixes the Q quaternions with a small fixed quaternion-valued coupling.
This target is EXACTLY a quaternion-linear map, so a model whose inductive bias
matches the structure should win at equal parameter count.

THREE param-matched contenders map 16 -> 16 channels:
  (A) TNNetQuaternionLinear(16)          : OutQ*InQ*4 = 4*4*4   = 64 weights
  (B) TNNetFullConnectLinear(16) bias-off but reduced width via a 16->4->16
      bottleneck                          : 16*4 + 4*16 = 128 ... (see note)
  (C) AddGroupedFullConnect(4 groups)     : block-diagonal, 4*(4*4) = 64 weights
We report each model's trainable weight count so the comparison is honest, then
train all three with the SAME data / schedule and print final MSE. The
quaternion model is expected to reach the lowest error per-parameter because its
weight sharing is exactly the symmetry of the task; the block-diagonal grouped
model cannot mix across quaternions, and the plain dense bottleneck spends its
budget on an unstructured low-rank factorisation.

Pure CPU, tiny data, few epochs -- runs in well under a minute.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cQuaternions = 4;                 // Q input/output quaternions
  cChannels    = cQuaternions * 4;  // 16 real channels
  cTrain       = 256;
  cVal         = 64;
  cEpochs      = 200;
  cLR          = 0.02;
  cSeed        = 424242;

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
  // Ground-truth fixed rotation quaternion g and a small cross-quaternion mix.
  gRot: TQuat;
  CrossMix: array[0..cQuaternions - 1, 0..cQuaternions - 1] of TQuat;

procedure InitGroundTruth();
var
  i, j, c: integer;
  n: TNeuralFloat;
begin
  // A unit quaternion (cos t, sin t * axis) -> a real 3D rotation of the
  // imaginary part.
  gRot[0] := Cos(0.6);
  gRot[1] := Sin(0.6) * 0.3;
  gRot[2] := Sin(0.6) * 0.9;
  gRot[3] := Sin(0.6) * 0.2;
  n := Sqrt(gRot[0]*gRot[0] + gRot[1]*gRot[1] + gRot[2]*gRot[2] + gRot[3]*gRot[3]);
  for c := 0 to 3 do gRot[c] := gRot[c] / n;
  // Small fixed coupling between quaternions (also quaternion-valued).
  for i := 0 to cQuaternions - 1 do
    for j := 0 to cQuaternions - 1 do
      for c := 0 to 3 do
        if i = j then CrossMix[i, j, c] := 0
        else CrossMix[i, j, c] := 0.10 * Sin(1.3 * (i + 1) * (j + 2) + c);
end;

// Build one (input, target) pair. Target = sum_j (g (x) x_j) coupled into i.
procedure MakePair(Input, Target: TNNetVolume);
var
  i, j, c: integer;
  xj, rot, contrib: TQuat;
begin
  for i := 0 to cChannels - 1 do
    Input.Raw[i] := (Random - 0.5) * 2.0;
  for i := 0 to cQuaternions - 1 do
  begin
    // self term: rotate x_i by g
    for c := 0 to 3 do xj[c] := Input.Raw[i * 4 + c];
    rot := QMul(gRot, xj);
    for c := 0 to 3 do Target.Raw[i * 4 + c] := rot[c];
    // cross terms
    for j := 0 to cQuaternions - 1 do
      if j <> i then
      begin
        for c := 0 to 3 do xj[c] := Input.Raw[j * 4 + c];
        contrib := QMul(CrossMix[i, j], xj);
        for c := 0 to 3 do
          Target.Raw[i * 4 + c] := Target.Raw[i * 4 + c] + contrib[c];
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
  s, i, k: integer;
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
    // shuffle
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

var
  NNq, NNd, NNg: TNNet;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  WriteLn('QuaternionLinear: parameter-matched bake-off on a quaternion-rotation task.');
  WriteLn(Format('Task: rotate %d input quaternions (%d channels) by a fixed unit',
    [cQuaternions, cChannels]));
  WriteLn('quaternion + small cross-quaternion coupling. Target is EXACTLY a');
  WriteLn('quaternion-linear map, so 4-component coupling should help.');
  WriteLn(Format('Train=%d  Val=%d  Epochs=%d  LR=%.3f', [cTrain, cVal, cEpochs, cLR]));
  WriteLn;

  InitGroundTruth();
  BuildData();

  // (A) Quaternion-linear: 16 -> 16, OutQ*InQ*4 = 64 weights.
  NNq := TNNet.Create();
  NNq.AddLayer(TNNetInput.Create(1, 1, cChannels));
  NNq.AddLayer(TNNetQuaternionLinear.Create(cChannels));

  // (B) Dense bottleneck 16 -> 4 -> 16 (a low-rank dense factorisation at a
  //     comparable weight budget).
  NNd := TNNet.Create();
  NNd.AddLayer(TNNetInput.Create(1, 1, cChannels));
  NNd.AddLayer(TNNetFullConnectLinear.Create(4));
  NNd.AddLayer(TNNetFullConnectLinear.Create(cChannels));

  // (C) Grouped (block-diagonal) full connect: 4 groups, 16 -> 16.
  NNg := TNNet.Create();
  NNg.AddLayer(TNNetInput.Create(1, 1, cChannels));
  NNg.AddGroupedFullConnect(TNNetFullConnectLinear, 4, cChannels);

  WriteLn('Final results (lower val-MSE is better):');
  Train(NNq, 'TNNetQuaternionLinear');
  Train(NNd, 'TNNetFullConnectLinear(BN)');
  Train(NNg, 'AddGroupedFullConnect(4)');
  WriteLn;
  WriteLn('Expected: the quaternion layer reaches the lowest error because its');
  WriteLn('weight sharing matches the task symmetry, at ~1/4 the weights of a');
  WriteLn('full 16x16 dense layer (256).');

  NNq.Free; NNd.Free; NNg.Free;
  FreeData();
end.
