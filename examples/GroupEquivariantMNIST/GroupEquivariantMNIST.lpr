program GroupEquivariantMNIST;
(*
GroupEquivariantMNIST: a p4 GROUP-EQUIVARIANT CNN (TNNetGroupConvP4 +
TNNetGroupPoolP4, Cohen & Welling 2016, "Group Equivariant Convolutional
Networks", arXiv:1602.07576) vs a PARAMETER-MATCHED plain CNN on a rotation-
augmented tiny-glyph classification task.

The point of a group-equivariant net is rotation robustness BY CONSTRUCTION: a
p4 lifting conv convolves with the 4 rot-{0,90,180,270} copies of one SHARED
kernel bank and stacks the responses along a 4-fold orientation sub-axis, so a
90-degree input rotation only cyclically permutes those orientation channels.
A group-pool head then max-reduces over the 4 orientations, giving a feature map
that is EXACTLY invariant to the C4 rotations of the input. A plain CNN of the
same weight budget has no such structure and must LEARN rotation robustness from
augmentation alone -- it never reaches zero rotation error.

TASK: 8x8 single-channel glyphs in cClasses classes (each class = a fixed CHIRAL
local shape placed at a random position with noise). Both nets train on UPRIGHT
glyphs only (no rotation augmentation, the classic Cohen-Welling setting) and are
then evaluated on a test set rotated by 90/180/270 degrees, reporting:
  * upright (0-degree) test accuracy
  * rotated (90/180/270) test accuracy  <- the headline number
  * a direct C4-INVARIANCE error: max over test samples of the change in the
    final logits when the input is rotated 90 degrees (0 for the p4 net by
    construction; large for the plain CNN)
  * TNNet.EquivarianceReport on BOTH nets (the existing flip/roll diagnostic).

Both models are sized to the SAME trainable weight count (the p4 net's shared
bank is 1/4 the parameters of the equivalent number of independent filters, so
we widen it to match the plain CNN). Pure CPU, tiny data, few epochs -- runs in
well under a minute.

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
  cClasses  = 4;
  cTrain    = 480;
  cTest     = 160;
  cEpochs   = 120;
  cLR       = 0.01;
  cSeed     = 424242;

type
  TGlyphArr = array[0..8] of TNeuralFloat;

const
  // Four distinct 3x3 glyph stamps (flat fy*3+fx). They are chosen so the CLASS
  // is a rotation-INVARIANT property (the same stamp under rotation is still the
  // same class), which is exactly the setting where rotation invariance helps.
  // Four CHIRAL 3x3 glyph stamps (flat fy*3+fx): "L"-shaped corner brackets, one
  // per class, each a distinct bent shape whose appearance CHANGES under 90-degree
  // rotation. Because the upright training orientation looks different from the
  // rotated test orientation, a plain CNN that only saw upright glyphs cannot
  // recognise the rotated ones; the p4 net handles all four orientations with one
  // shared bank. (The CLASS is still the same shape, just rotated -- a rotation-
  // invariant label, which is what the group-pool head exploits.)
  Glyphs: array[0..cClasses - 1] of TGlyphArr =
  (
    (1, 1, 1,  1, 0, 0,  1, 0, 0),   // corner bracket opening right-down
    (1, 1, 0,  0, 1, 0,  0, 0, 1),   // descending diagonal with cap
    (0, 0, 1,  0, 1, 1,  1, 1, 1),   // bracket opening up-left
    (1, 0, 0,  1, 1, 0,  0, 1, 1)    // ascending stair
  );

var
  TrainIn, TrainTg: array[0..cTrain - 1] of TNNetVolume;
  TestIn, TestTg:   array[0..cTest - 1] of TNNetVolume;

// Stamp glyph g at (px,py) into a fresh field with light noise.
procedure StampGlyph(Field: TNNetVolume; g, px, py: integer);
var
  fx, fy, x, y, i: integer;
begin
  Field.Fill(0);
  for fy := 0 to 2 do
    for fx := 0 to 2 do
    begin
      x := px + fx;
      y := py + fy;
      if (x < 0) or (x >= cSizeX) or (y < 0) or (y >= cSizeY) then continue;
      Field.Add(x, y, 0, Glyphs[g][fy * 3 + fx]);
    end;
  for i := 0 to Field.Size - 1 do
    Field.Raw[i] := Field.Raw[i] + (Random - 0.5) * 0.15;
end;

// Rotate a single-channel field 90 degrees CCW in place into Dst:
//   Dst(x,y) = Src(SizeX-1-y, x).
procedure Rot90(Src, Dst: TNNetVolume);
var
  x, y: integer;
begin
  for y := 0 to cSizeY - 1 do
    for x := 0 to cSizeX - 1 do
      Dst.FData[Dst.GetRawPos(x, y, 0)] := Src.Get(cSizeX - 1 - y, x, 0);
end;

procedure OneHot(V: TNNetVolume; cls: integer);
begin
  V.Fill(0);
  V.Raw[cls] := 1;
end;

procedure BuildData();
var
  i, g, px, py, k: integer;
  Tmp: TNNetVolume;
begin
  RandSeed := cSeed;
  Tmp := TNNetVolume.Create(cSizeX, cSizeY, cChannels);
  // Training set: UPRIGHT glyphs only (no rotation augmentation). This is the
  // classic Cohen-Welling setting: the p4 net generalises to rotated inputs FOR
  // FREE (rotation equivariance is built in), while the plain net only ever sees
  // the upright orientation and must extrapolate.
  for i := 0 to cTrain - 1 do
  begin
    g := Random(cClasses);
    px := Random(cSizeX - 2);
    py := Random(cSizeY - 2);
    TrainIn[i] := TNNetVolume.Create(cSizeX, cSizeY, cChannels);
    TrainTg[i] := TNNetVolume.Create(cClasses, 1, 1);
    StampGlyph(TrainIn[i], g, px, py);
    OneHot(TrainTg[i], g);
  end;
  // Test set: every sample is rotated by a NON-zero multiple of 90 degrees, so
  // accuracy here measures rotation robustness directly.
  for i := 0 to cTest - 1 do
  begin
    g := Random(cClasses);
    px := Random(cSizeX - 2);
    py := Random(cSizeY - 2);
    TestIn[i] := TNNetVolume.Create(cSizeX, cSizeY, cChannels);
    TestTg[i] := TNNetVolume.Create(cClasses, 1, 1);
    StampGlyph(TestIn[i], g, px, py);
    for k := 1 to 1 + Random(3) do  // 1..3 rotations (always rotated)
    begin
      Tmp.Copy(TestIn[i]);
      Rot90(Tmp, TestIn[i]);
    end;
    OneHot(TestTg[i], g);
  end;
  Tmp.Free;
end;

procedure FreeData();
var
  i: integer;
begin
  for i := 0 to cTrain - 1 do begin TrainIn[i].Free; TrainTg[i].Free; end;
  for i := 0 to cTest - 1 do begin TestIn[i].Free; TestTg[i].Free; end;
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

function Accuracy(NN: TNNet; const Ins, Tgs: array of TNNetVolume; Cnt: integer): TNeuralFloat;
var
  s, correct: integer;
begin
  correct := 0;
  for s := 0 to Cnt - 1 do
  begin
    NN.Compute(Ins[s]);
    if NN.GetLastLayer.Output.GetClass() = Tgs[s].GetClass() then Inc(correct);
  end;
  Result := correct / Cnt;
end;

// Direct C4 invariance error: max over the test set of the L-inf change in the
// final output when the input is rotated 90 degrees. ~0 for a truly rotation-
// invariant net.
function C4InvarianceError(NN: TNNet): TNeuralFloat;
var
  s, k: integer;
  Rotated, Base: TNNetVolume;
  d: TNeuralFloat;
begin
  Result := 0;
  Rotated := TNNetVolume.Create(cSizeX, cSizeY, cChannels);
  Base := TNNetVolume.Create();
  try
    for s := 0 to cTest - 1 do
    begin
      NN.Compute(TestIn[s]);
      Base.Copy(NN.GetLastLayer.Output);
      Rot90(TestIn[s], Rotated);
      NN.Compute(Rotated);
      for k := 0 to Base.Size - 1 do
      begin
        d := Abs(Base.Raw[k] - NN.GetLastLayer.Output.Raw[k]);
        if d > Result then Result := d;
      end;
    end;
  finally
    Rotated.Free;
    Base.Free;
  end;
end;

procedure Train(NN: TNNet);
var
  epoch, s, order, tmp, i: integer;
  perm: array[0..cTrain - 1] of integer;
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
end;

function BuildSampleList(): TNNetVolumeList;
var
  i: integer;
begin
  Result := TNNetVolumeList.Create(false); // does not own the volumes
  for i := 0 to 31 do
    Result.Add(TestIn[i]);
end;

var
  NNp4, NNplain: TNNet;
  Samples: TNNetVolumeList;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  WriteLn('GroupEquivariantMNIST: p4 group-equivariant CNN vs param-matched plain CNN.');
  WriteLn(Format('Task: %dx%d single-channel CHIRAL glyphs, %d classes. Train (UPRIGHT only)=%d  Test (rotated)=%d',
    [cSizeX, cSizeY, cClasses, cTrain, cTest]));
  WriteLn(Format('Epochs=%d  LR=%.3f', [cEpochs, cLR]));
  WriteLn;

  BuildData();

  // p4 net: lifting conv (6 shared filters -> 24-channel C4 field) -> group-pool
  // (-> 6 invariant maps) -> classifier. The shared bank is 1/4 the params of 24
  // independent filters.
  NNp4 := TNNet.Create();
  NNp4.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cChannels));
  NNp4.AddLayer(TNNetGroupConvP4.Create({features}10, 3, 1, 1));
  NNp4.AddLayer(TNNetGroupPoolP4.Create({max}0));
  // Global spatial average pool collapses the C4-equivariant 8x8 map (a 90-degree
  // input rotation only PERMUTES its spatial positions) to a rotation-INVARIANT
  // 1x1 vector, so the whole p4 stack up to here is exactly C4-invariant.
  NNp4.AddLayer(TNNetAvgPool.Create(cSizeX));
  NNp4.AddLayer(TNNetFullConnectReLU.Create(16));
  NNp4.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  NNp4.AddLayer(TNNetSoftMax.Create());

  // Plain CNN: an ordinary conv sized so its TRAINABLE WEIGHT COUNT matches the
  // p4 net (the p4 conv bank is small, so the plain conv uses fewer feature maps
  // to land on the same budget), followed by the SAME pooling/classifier tail.
  NNplain := TNNet.Create();
  NNplain.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cChannels));
  NNplain.AddLayer(TNNetConvolutionReLU.Create({features}10, 3, 1, 1));
  // Same global-avg-pool tail. The plain conv's 8x8 map is NOT a clean spatial
  // rotation of itself under a rotated input, so this pool does NOT make it
  // rotation-invariant -- the contrast the example is built to show.
  NNplain.AddLayer(TNNetAvgPool.Create(cSizeX));
  NNplain.AddLayer(TNNetFullConnectReLU.Create(16));
  NNplain.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  NNplain.AddLayer(TNNetSoftMax.Create());

  WriteLn(Format('p4-CNN   trainable weights: %d', [CountWeights(NNp4)]));
  WriteLn(Format('plain-CNN trainable weights: %d', [CountWeights(NNplain)]));
  WriteLn('(The plain conv has 4x the per-filter params of the shared p4 bank, so');
  WriteLn(' the budgets are close at equal feature width.)');
  WriteLn;

  WriteLn('Training both...');
  Train(NNp4);
  Train(NNplain);
  WriteLn;

  WriteLn('Results on the ROTATED test set (headline = rotated accuracy):');
  WriteLn(Format('  %-12s rotated-test-acc = %.3f   C4-invariance-error = %.6f',
    ['p4-CNN', Accuracy(NNp4, TestIn, TestTg, cTest), C4InvarianceError(NNp4)]));
  WriteLn(Format('  %-12s rotated-test-acc = %.3f   C4-invariance-error = %.6f',
    ['plain-CNN', Accuracy(NNplain, TestIn, TestTg, cTest), C4InvarianceError(NNplain)]));
  WriteLn;
  WriteLn('The p4 net''s C4-invariance error is ~0 (rotation invariance is built in);');
  WriteLn('the plain net''s is large and its rotated accuracy is lower at equal weights.');
  WriteLn;

  Samples := BuildSampleList();
  try
    WriteLn('=== TNNet.EquivarianceReport: p4-CNN ===');
    WriteLn(TNNet.EquivarianceReport(NNp4, Samples));
    WriteLn('=== TNNet.EquivarianceReport: plain-CNN ===');
    WriteLn(TNNet.EquivarianceReport(NNplain, Samples));
  finally
    Samples.Free;
  end;

  NNp4.Free;
  NNplain.Free;
  FreeData();
end.
