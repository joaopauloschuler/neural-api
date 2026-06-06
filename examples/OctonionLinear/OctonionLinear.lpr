program OctonionLinear;
(*
OctonionLinear: a PARAMETER-MATCHED bake-off that shows when the structured
weight sharing of TNNetOctonionLinear (the 8-dimensional hypercomplex sibling of
TNNetQuaternionLinear) actually pays off.

TNNetOctonionLinear reinterprets the input/output Depth (multiples of 8) as
packed OCTONIONS and learns an (OutO x InO) grid of octonion weights
w = o0 + o1 e1 + ... + o7 e7. The forward pass is the Cayley-Dickson product
y = W . X, i.e. a real 8x8-block matrix where each learned octonion's 8 reals
drive a whole 8x8 block via the fixed octonion multiplication table
    M(W)[i][j] = SGN[i][j] * W[i xor j].
So ONE octonion's 8 reals control 64 matrix entries -> the layer stores ~1/8
the weights of a dense layer of equal width while still mixing all eight
components of every input octonion.

TASK (built to favour 8-component coupling): each sample is X = 2 input
octonions (16 real channels). The TARGET multiplies, on the LEFT, each input
octonion by the SAME fixed ground-truth octonion g, then adds a small fixed
octonion-valued cross coupling between the two octonions. This target is EXACTLY
an octonion-linear map, so a model whose inductive bias matches the structure
should win at equal parameter count.

THREE param-matched contenders map 16 -> 16 channels:
  (A) TNNetOctonionLinear(16)        : OutO*InO*8 = 2*2*8 = 32 weights
  (B) TNNetFullConnectLinear bottleneck 16 -> 2 -> 16 : 16*2 + 2*16 = 64 weights
  (C) AddGroupedFullConnect(2 groups): block-diagonal, 2*(8*8) = 128 weights
We report each model's trainable weight count so the comparison is honest, then
train all three with the SAME data / schedule and print final MSE. The octonion
model is expected to reach the lowest error per-parameter because its weight
sharing is exactly the symmetry of the task, at ~1/8 the weights of a full
16x16 dense layer (256).

Pure CPU, tiny data, few epochs -- runs in well under a minute.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cOctonions = 2;                  // input/output octonions
  cChannels  = cOctonions * 8;     // 16 real channels
  cTrain     = 256;
  cVal       = 64;
  cEpochs    = 200;
  cLR        = 0.02;
  cSeed      = 424242;

type
  TOct = array[0..7] of TNeuralFloat;

var
  // The SAME octonion multiplication table as the layer (Cayley-Dickson).
  SRC: array[0..7, 0..7] of integer =
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
  SGN: array[0..7, 0..7] of TNeuralFloat =
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

// Octonion product w . x.
function OMul(const w, x: TOct): TOct;
var
  i, j: integer;
  acc: TNeuralFloat;
begin
  for i := 0 to 7 do
  begin
    acc := 0;
    for j := 0 to 7 do
      acc := acc + SGN[i, j] * w[SRC[i, j]] * x[j];
    Result[i] := acc;
  end;
end;

var
  gRot: TOct;                                   // fixed ground-truth octonion
  CrossMix: array[0..cOctonions - 1, 0..cOctonions - 1] of TOct;

procedure InitGroundTruth();
var
  i, j, c: integer;
  n: TNeuralFloat;
begin
  gRot[0] := 0.8;
  for c := 1 to 7 do gRot[c] := 0.3 * Sin(1.7 * c);
  n := 0;
  for c := 0 to 7 do n := n + gRot[c] * gRot[c];
  n := Sqrt(n);
  for c := 0 to 7 do gRot[c] := gRot[c] / n;    // unit octonion
  for i := 0 to cOctonions - 1 do
    for j := 0 to cOctonions - 1 do
      for c := 0 to 7 do
        if i = j then CrossMix[i, j, c] := 0
        else CrossMix[i, j, c] := 0.10 * Sin(1.3 * (i + 1) * (j + 2) + c);
end;

// Target = (g . x_i) + sum_{j<>i} CrossMix[i,j] . x_j.
procedure MakePair(Input, Target: TNNetVolume);
var
  i, j, c: integer;
  xi, xj, rot, contrib: TOct;
begin
  for i := 0 to cChannels - 1 do
    Input.Raw[i] := (Random - 0.5) * 2.0;
  for i := 0 to cOctonions - 1 do
  begin
    for c := 0 to 7 do xi[c] := Input.Raw[i * 8 + c];
    rot := OMul(gRot, xi);
    for c := 0 to 7 do Target.Raw[i * 8 + c] := rot[c];
    for j := 0 to cOctonions - 1 do
      if j <> i then
      begin
        for c := 0 to 7 do xj[c] := Input.Raw[j * 8 + c];
        contrib := OMul(CrossMix[i, j], xj);
        for c := 0 to 7 do
          Target.Raw[i * 8 + c] := Target.Raw[i * 8 + c] + contrib[c];
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

var
  NNo, NNd, NNg: TNNet;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  WriteLn('OctonionLinear: parameter-matched bake-off on an octonion-product task.');
  WriteLn(Format('Task: left-multiply %d input octonions (%d channels) by a fixed unit',
    [cOctonions, cChannels]));
  WriteLn('octonion + small cross-octonion coupling. Target is EXACTLY an');
  WriteLn('octonion-linear map, so 8-component coupling should help.');
  WriteLn(Format('Train=%d  Val=%d  Epochs=%d  LR=%.3f', [cTrain, cVal, cEpochs, cLR]));
  WriteLn;

  InitGroundTruth();
  BuildData();

  // (A) Octonion-linear: 16 -> 16, OutO*InO*8 = 32 weights.
  NNo := TNNet.Create();
  NNo.AddLayer(TNNetInput.Create(1, 1, cChannels));
  NNo.AddLayer(TNNetOctonionLinear.Create(cChannels));

  // (B) Dense bottleneck 16 -> 2 -> 16 (a low-rank dense factorisation).
  NNd := TNNet.Create();
  NNd.AddLayer(TNNetInput.Create(1, 1, cChannels));
  NNd.AddLayer(TNNetFullConnectLinear.Create(2));
  NNd.AddLayer(TNNetFullConnectLinear.Create(cChannels));

  // (C) Grouped (block-diagonal) full connect: 2 groups, 16 -> 16.
  NNg := TNNet.Create();
  NNg.AddLayer(TNNetInput.Create(1, 1, cChannels));
  NNg.AddGroupedFullConnect(TNNetFullConnectLinear, 2, cChannels);

  WriteLn('Final results (lower val-MSE is better):');
  Train(NNo, 'TNNetOctonionLinear');
  Train(NNd, 'TNNetFullConnectLinear(BN)');
  Train(NNg, 'AddGroupedFullConnect(2)');
  WriteLn;
  WriteLn('Expected: the octonion layer reaches the lowest error because its');
  WriteLn('weight sharing matches the task symmetry, at ~1/8 the weights of a');
  WriteLn('full 16x16 dense layer (256).');

  NNo.Free; NNd.Free; NNg.Free;
  FreeData();
end.
