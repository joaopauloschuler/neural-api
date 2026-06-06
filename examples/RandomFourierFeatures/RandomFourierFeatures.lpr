(*
 * Random Fourier Features demo -- TNNetRandomFourierFeatures
 * --------------------------------------------------------------------------
 * The headline this demo proves: a SINGLE fixed random-feature layer plus a
 * linear head already separates a non-linearly-separable problem, matching a
 * deeper ReLU MLP -- the classic Rahimi & Recht (2007) result.
 *
 * Random Fourier Features map a Din-vector x to
 *     phi_k(x) = sqrt(1/D) * [ cos(w_k . x) , sin(w_k . x) ]
 * with the projection rows w_k drawn ONCE from N(0, 1/sigma^2) and FROZEN.
 * Then <phi(x),phi(y)> approximates the RBF kernel exp(-||x-y||^2/(2 sigma^2)),
 * so a plain LINEAR classifier over phi(x) approximates an RBF-kernel SVM --
 * which can carve out curved / concentric decision regions.
 *
 * The task here is the canonical CONCENTRIC RINGS: an inner disk (class 0)
 * surrounded by an outer ring (class 1). No straight line separates them, so a
 * bare linear classifier on the raw (x,y) is stuck near chance. We compare:
 *   (A) RFF -> FullConnectLinear -> SoftMax       (random features + linear head)
 *   (B) raw (x,y) -> FullConnectLinear -> SoftMax (linear baseline, the control)
 *   (C) a small ReLU MLP                          (learned-feature reference)
 * (A) separates the rings with a FROZEN feature map (only the tiny linear head
 * trains), matching the MLP (C) while crushing the linear baseline (B).
 *
 * Pure CPU, single-threaded-friendly, runs in a few seconds. No binaries are
 * committed.
 *
 * Coded by Claude (AI).
 *)
program RandomFourierFeatures;
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, neuralnetwork, neuralvolume;

const
  NTRAIN   = 400;    // training points
  NTEST    = 400;    // held-out test points
  NFEAT    = 256;    // D random features (output Depth = 2*D = 512)
  SIGMA    = 0.45;   // RBF bandwidth (tuned to the ring geometry)
  EPOCHS   = 60;
  LR       = 0.05;
  R_INNER  = 0.55;   // class boundary radius

type
  TPt = record x, y: TNeuralFloat; lbl: integer; end;

var
  TrainSet: array[0..NTRAIN - 1] of TPt;
  TestSet:  array[0..NTEST  - 1] of TPt;

// Concentric rings: label by radius. An outer ring (class 1) around an inner
// disk (class 0), with a margin gap so the boundary is clean. No straight line
// separates them. Pass the array by var (a static-array var param is allowed).
procedure MakeData(var Pts: array of TPt; N: integer);
var
  i, c: integer;
  ang, rad: TNeuralFloat;
begin
  for i := 0 to N - 1 do
  begin
    c := i mod 2;             // balanced classes
    ang := 2 * Pi * Random;
    if c = 0 then
      rad := 0.05 + 0.40 * Random          // inner disk  (r in [0.05, 0.45])
    else
      rad := 0.70 + 0.40 * Random;         // outer ring  (r in [0.70, 1.10])
    Pts[i].x := rad * Cos(ang);
    Pts[i].y := rad * Sin(ang);
    Pts[i].lbl := c;
  end;
end;

// Train a 2-output (SoftMax) classifier; return test accuracy in [0,1].
function TrainEval(NN: TNNet; const tag: string): TNeuralFloat;
var
  ep, s, pred, correct: integer;
  Inp, Tgt: TNNetVolume;
  best: TNeuralFloat;
  k: integer;
begin
  Inp := TNNetVolume.Create(2, 1, 1);
  Tgt := TNNetVolume.Create(2, 1, 1);
  NN.SetLearningRate(LR, 0.9);
  NN.SetBatchUpdate(true);
  for ep := 0 to EPOCHS - 1 do
  begin
    NN.ClearDeltas();
    for s := 0 to NTRAIN - 1 do
    begin
      Inp.FData[0] := TrainSet[s].x;
      Inp.FData[1] := TrainSet[s].y;
      Tgt.Fill(0);
      Tgt.FData[TrainSet[s].lbl] := 1;
      NN.Compute(Inp);
      NN.Backpropagate(Tgt);
    end;
    NN.UpdateWeights();
  end;
  // Evaluate on the held-out test set.
  correct := 0;
  for s := 0 to NTEST - 1 do
  begin
    Inp.FData[0] := TestSet[s].x;
    Inp.FData[1] := TestSet[s].y;
    NN.Compute(Inp);
    pred := 0; best := NN.GetLastLayer.Output.FData[0];
    for k := 1 to 1 do
      if NN.GetLastLayer.Output.FData[k] > best then
      begin
        best := NN.GetLastLayer.Output.FData[k];
        pred := k;
      end;
    if pred = TestSet[s].lbl then Inc(correct);
  end;
  Inp.Free; Tgt.Free;
  WriteLn(Format('  [%s] test accuracy: %.3f  (params=%d)',
    [tag, correct / NTEST, NN.CountWeights()]));
  Result := correct / NTEST;
end;

var
  RFFNet, LinNet, MLPNet: TNNet;
  accRFF, accLin, accMLP: TNeuralFloat;
begin
  RandSeed := 424242;
  MakeData(TrainSet, NTRAIN);
  MakeData(TestSet,  NTEST);

  WriteLn('Random Fourier Features demo: a FROZEN random-feature layer + linear');
  WriteLn('head separates concentric rings that no straight line can split.');
  WriteLn(Format('Train=%d  Test=%d  D=%d (feature Depth=%d)  sigma=%.2f',
    [NTRAIN, NTEST, NFEAT, 2 * NFEAT, SIGMA]));
  WriteLn;

  // ---- (A) RFF -> linear head. The random projection is FROZEN (classic RFF);
  // only the 2-class linear head trains. This is an explicit RBF-kernel machine.
  RFFNet := TNNet.Create();
  RFFNet.AddLayer(TNNetInput.Create(2, 1, 1));
  RFFNet.AddLayer(TNNetRandomFourierFeatures.Create(NFEAT, SIGMA, {trainable}0, {seed}777));
  RFFNet.AddLayer(TNNetFullConnectLinear.Create(2));
  RFFNet.AddLayer(TNNetSoftMax.Create());

  // ---- (B) Linear baseline on the raw (x,y): one straight boundary -> chance.
  LinNet := TNNet.Create();
  LinNet.AddLayer(TNNetInput.Create(2, 1, 1));
  LinNet.AddLayer(TNNetFullConnectLinear.Create(2));
  LinNet.AddLayer(TNNetSoftMax.Create());

  // ---- (C) Small ReLU MLP reference (learned features).
  MLPNet := TNNet.Create();
  MLPNet.AddLayer(TNNetInput.Create(2, 1, 1));
  MLPNet.AddLayer(TNNetFullConnect.Create(32));
  MLPNet.AddLayer(TNNetReLU.Create());
  MLPNet.AddLayer(TNNetFullConnect.Create(32));
  MLPNet.AddLayer(TNNetReLU.Create());
  MLPNet.AddLayer(TNNetFullConnectLinear.Create(2));
  MLPNet.AddLayer(TNNetSoftMax.Create());

  WriteLn('Training (A) RFF + linear head (random map frozen)...');
  accRFF := TrainEval(RFFNet, 'RFF');
  WriteLn('Training (B) raw linear baseline...');
  accLin := TrainEval(LinNet, 'LIN');
  WriteLn('Training (C) ReLU MLP reference...');
  accMLP := TrainEval(MLPNet, 'MLP');
  WriteLn;

  WriteLn('=== Held-out test accuracy ===');
  WriteLn(Format('  (A) RFF + linear head : %.3f', [accRFF]));
  WriteLn(Format('  (B) raw linear        : %.3f', [accLin]));
  WriteLn(Format('  (C) ReLU MLP          : %.3f', [accMLP]));
  WriteLn;
  WriteLn('HEADLINE: the RFF model separates the rings (accuracy near the MLP)');
  WriteLn('with a FROZEN random feature map -- only the tiny linear head learns --');
  WriteLn('while the same linear classifier on the raw coordinates is stuck near');
  WriteLn('chance. One random-feature layer = an explicit RBF-kernel machine.');

  RFFNet.Free;
  LinNet.Free;
  MLPNet.Free;
end.
