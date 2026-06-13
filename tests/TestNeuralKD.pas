unit TestNeuralKD;
(*
Tests for neuralkd.pas: classic Hinton knowledge distillation (KD) trainer.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralnetwork, neuralvolume, neuralkd;

type
  TTestNeuralKD = class(TTestCase)
  private
    // Tiny classifier: Input(InW,1,InD) -> FC ReLU -> FC Linear(Vocab) -> SoftMax.
    function BuildTinyNet(InW, InD, Hidden, Vocab: integer): TNNet;
    // softmax(logits) into Dest.
    procedure ReadProbs(NN: TNNet; pInput: TNNetVolume; Dest: TNNetVolume);
    // Snapshot every weight + bias of NN into a flat array.
    function Snapshot(NN: TNNet): TNeuralFloatDynArr;
  published
    // alpha=1 (pure hard label): one KD step must equal an ordinary
    // cross-entropy SGD step (identical weight movement to a hand-run
    // softmax+CE Backpropagate on the SAME init).
    procedure TestAlphaOneMatchesPlainCE;
    // alpha=0, T=1: distilling toward a fixed teacher must drive the
    // student's softmax toward the teacher's; KL decreases monotonically.
    procedure TestAlphaZeroKLDecreasesMonotonically;
    // The teacher net is frozen: every teacher weight is bit-identical after
    // many KD steps.
    procedure TestTeacherWeightsUnchanged;
    // Finite-difference check of dLoss/dWeight (blended alpha, T>1) against
    // the implemented backward (LR=1, inertia=0, batch mode: grad = -Delta).
    procedure TestFiniteDifferenceGradient;
  end;

implementation

const
  csInW    = 3;
  csInD    = 5;
  csHidden = 8;
  csVocab  = 6;

function TTestNeuralKD.BuildTinyNet(InW, InD, Hidden, Vocab: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(InW, 1, InD));
  Result.AddLayer(TNNetFullConnectReLU.Create(Hidden));
  Result.AddLayer(TNNetFullConnectLinear.Create(Vocab));
  Result.AddLayer(TNNetSoftMax.Create());
end;

procedure TTestNeuralKD.ReadProbs(NN: TNNet; pInput: TNNetVolume; Dest: TNNetVolume);
begin
  NN.Compute(pInput);
  Dest.Copy(NN.GetLastLayer().Output);
end;

function TTestNeuralKD.Snapshot(NN: TNNet): TNeuralFloatDynArr;
var
  L, N, I, Cnt: integer;
begin
  Cnt := 0;
  for L := 0 to NN.Layers.Count - 1 do
    for N := 0 to NN.Layers[L].Neurons.Count - 1 do
      Cnt := Cnt + NN.Layers[L].Neurons[N].Weights.Size + 1; // +1 for bias
  SetLength(Result, Cnt);
  Cnt := 0;
  for L := 0 to NN.Layers.Count - 1 do
    for N := 0 to NN.Layers[L].Neurons.Count - 1 do
    begin
      for I := 0 to NN.Layers[L].Neurons[N].Weights.Size - 1 do
      begin
        Result[Cnt] := NN.Layers[L].Neurons[N].Weights.FData[I];
        Inc(Cnt);
      end;
      Result[Cnt] := NN.Layers[L].Neurons[N].BiasWeight;
      Inc(Cnt);
    end;
end;

procedure TTestNeuralKD.TestAlphaOneMatchesPlainCE;
var
  Teacher, Student, Plain: TNNet;
  Trainer: TNeuralKDTrainer;
  Input: TNNetVolume;
  Probs: TNNetVolume;
  KDW, PlainW: TNeuralFloatDynArr;
  HardLabel, I: integer;
  MaxDiff: TNeuralFloat;
begin
  RandSeed := 424242;
  Input := TNNetVolume.Create(csInW, 1, csInD);
  Input.Randomize();
  HardLabel := 2;

  // Build two identical students (same RNG init) and an arbitrary teacher.
  RandSeed := 13579;
  Student := BuildTinyNet(csInW, csInD, csHidden, csVocab);
  RandSeed := 13579;
  Plain := BuildTinyNet(csInW, csInD, csHidden, csVocab);
  RandSeed := 999;
  Teacher := BuildTinyNet(csInW, csInD, csHidden, csVocab);

  Student.SetLearningRate(0.1, 0);
  Plain.SetLearningRate(0.1, 0);

  // KD step with alpha=1 -> the teacher is irrelevant, pure hard-label CE.
  Trainer := TNeuralKDTrainer.Create(Teacher, Student, {alpha=}1.0, {T=}3.0);
  Trainer.Step(Input, HardLabel);

  // Reference: an ordinary softmax + cross-entropy SGD step, computed the
  // framework-canonical way. TNNetSoftMax applies the FULL softmax Jacobian
  //   dL/dx_i = y_i*(e_i - sum_j y_j e_j)
  // on its OutputError e. Seeding e_target = -1/y_target (zero elsewhere)
  // makes the dot product sum_j y_j e_j = -1, so the logit gradient becomes
  //   x_target: y_target*(-1/y_target + 1) = y_target - 1
  //   x_i!=t  : y_i*(0 + 1)               = y_i
  // i.e. EXACTLY the textbook cross-entropy gradient (p1 - onehot). This is
  // an independent route to that gradient (KD injects it directly into the
  // logit layer instead).
  Probs := TNNetVolume.Create();
  Plain.SetBatchUpdate(true);
  Plain.ClearDeltas();
  ReadProbs(Plain, Input, Probs);   // Probs = softmax(logits) = p1
  // Build the pseudo-target pt such that Backpropagate sets the softmax
  // OutputError = (output - pt) = e, i.e. pt = output - e. We want
  // e_target = -1/y_target and e_i = 0 elsewhere -> pt = output except
  // pt_target = y_target + 1/y_target.
  Plain.Compute(Input);
  Probs.Copy(Plain.GetLastLayer().Output);
  Probs.FData[HardLabel] := Probs.FData[HardLabel] +
    1 / Plain.GetLastLayer().Output.FData[HardLabel];
  Plain.Backpropagate(Probs);
  Plain.UpdateWeights();
  Plain.SetBatchUpdate(false);

  KDW := Snapshot(Student);
  PlainW := Snapshot(Plain);
  MaxDiff := 0;
  for I := 0 to High(KDW) do
    MaxDiff := Max(MaxDiff, Abs(KDW[I] - PlainW[I]));
  AssertTrue('alpha=1 KD step must equal an ordinary CE step (max weight ' +
    'diff ' + FloatToStr(MaxDiff) + ')', MaxDiff < 1e-6);

  Probs.Free;
  Trainer.Free;
  Teacher.Free;
  Plain.Free;
  Student.Free;
  Input.Free;
end;

procedure TTestNeuralKD.TestAlphaZeroKLDecreasesMonotonically;
var
  Teacher, Student: TNNet;
  Trainer: TNeuralKDTrainer;
  Input: TNNetVolume;
  PrevKL: TNeuralFloat;
  Step: integer;
begin
  RandSeed := 424242;
  Input := TNNetVolume.Create(csInW, 1, csInD);
  Input.Randomize();

  RandSeed := 222;
  Teacher := BuildTinyNet(csInW, csInD, csHidden, csVocab);
  RandSeed := 777;
  Student := BuildTinyNet(csInW, csInD, csHidden, csVocab);
  Student.SetLearningRate(0.2, 0);

  // alpha=0, T=1: pure soft distillation toward the (frozen) teacher.
  Trainer := TNeuralKDTrainer.Create(Teacher, Student, {alpha=}0.0, {T=}1.0);

  // The hard label is irrelevant at alpha=0; pick anything in range.
  Trainer.ComputeLoss(Input, 0);
  PrevKL := Trainer.LastKL;
  AssertTrue('initial KL must be positive', PrevKL > 1e-4);

  for Step := 1 to 12 do
  begin
    Trainer.Step(Input, 0);
    AssertTrue('KD KL divergence must decrease monotonically (step ' +
      IntToStr(Step) + ': ' + FloatToStr(Trainer.LastKL) + ' >= prev ' +
      FloatToStr(PrevKL) + ')', Trainer.LastKL < PrevKL + 1e-9);
    PrevKL := Trainer.LastKL;
  end;
  // After enough steps the student should closely match the teacher.
  AssertTrue('KL must shrink toward zero, got ' + FloatToStr(PrevKL),
    PrevKL < 0.05);

  Trainer.Free;
  Student.Free;
  Teacher.Free;
  Input.Free;
end;

procedure TTestNeuralKD.TestTeacherWeightsUnchanged;
var
  Teacher, Student: TNNet;
  Trainer: TNeuralKDTrainer;
  Input: TNNetVolume;
  Before, After: TNeuralFloatDynArr;
  Step, I: integer;
begin
  RandSeed := 424242;
  Input := TNNetVolume.Create(csInW, 1, csInD);
  Input.Randomize();

  RandSeed := 222;
  Teacher := BuildTinyNet(csInW, csInD, csHidden, csVocab);
  RandSeed := 777;
  Student := BuildTinyNet(csInW, csInD, csHidden, csVocab);
  Student.SetLearningRate(0.3, 0);

  Trainer := TNeuralKDTrainer.Create(Teacher, Student, {alpha=}0.5, {T=}2.5);
  Before := Snapshot(Teacher);
  for Step := 1 to 20 do
    Trainer.Step(Input, Step mod csVocab);
  After := Snapshot(Teacher);

  for I := 0 to High(Before) do
    AssertTrue('teacher weight ' + IntToStr(I) + ' moved during distillation',
      Before[I] = After[I]);

  Trainer.Free;
  Student.Free;
  Teacher.Free;
  Input.Free;
end;

procedure TTestNeuralKD.TestFiniteDifferenceGradient;
const
  csEps = 1e-3;
var
  Teacher, Student: TNNet;
  Trainer: TNeuralKDTrainer;
  Input: TNNetVolume;
  HardLabel, LayerIdx, NeuronIdx, WeightIdx, N, I, Checked: integer;
  W: TNNetVolume;
  Saved, LossPlus, LossMinus, FDGrad, ImplGrad, Best: TNeuralFloat;
begin
  RandSeed := 424242;
  Input := TNNetVolume.Create(csInW, 1, csInD);
  Input.Randomize();
  HardLabel := 3;

  RandSeed := 222;
  Teacher := BuildTinyNet(csInW, csInD, csHidden, csVocab);
  RandSeed := 777;
  Student := BuildTinyNet(csInW, csInD, csHidden, csVocab);
  // Keep probabilities away from 0/1 so FD of the log terms stays accurate.
  Student.MulWeights(0.5);
  Teacher.MulWeights(0.5);
  Student.SetLearningRate(1.0, 0);   // LR=1, inertia=0 -> Delta = -gradient.

  Trainer := TNeuralKDTrainer.Create(Teacher, Student, {alpha=}0.4, {T=}2.0);

  Student.SetBatchUpdate(true);
  Student.ClearDeltas();
  Trainer.AccumulateGradients(Input, HardLabel);
  Student.SetBatchUpdate(false);

  // FD-check the largest-gradient weight in each trainable student layer.
  Checked := 0;
  for LayerIdx := 1 to 2 do
  begin
    Best := -1; NeuronIdx := 0; WeightIdx := 0;
    for N := 0 to Student.Layers[LayerIdx].Neurons.Count - 1 do
      for I := 0 to Student.Layers[LayerIdx].Neurons[N].Weights.Size - 1 do
        if Abs(Student.Layers[LayerIdx].Neurons[N].Delta.FData[I]) > Best then
        begin
          Best := Abs(Student.Layers[LayerIdx].Neurons[N].Delta.FData[I]);
          NeuronIdx := N; WeightIdx := I;
        end;
    AssertTrue('student layer ' + IntToStr(LayerIdx) +
      ' must have a non-zero gradient', Best > 1e-4);
    ImplGrad := -Student.Layers[LayerIdx].Neurons[NeuronIdx].Delta.FData[WeightIdx];

    W := Student.Layers[LayerIdx].Neurons[NeuronIdx].Weights;
    Saved := W.FData[WeightIdx];
    W.FData[WeightIdx] := Saved + csEps;
    LossPlus := Trainer.ComputeLoss(Input, HardLabel);
    W.FData[WeightIdx] := Saved - csEps;
    LossMinus := Trainer.ComputeLoss(Input, HardLabel);
    W.FData[WeightIdx] := Saved;
    FDGrad := (LossPlus - LossMinus) / (2 * csEps);

    AssertTrue(Format(
      'KD finite-difference mismatch at layer %d neuron %d weight %d: ' +
      'fd=%g implemented=%g', [LayerIdx, NeuronIdx, WeightIdx, FDGrad, ImplGrad]),
      Abs(FDGrad - ImplGrad) / Max(Abs(ImplGrad), 1e-4) < 0.02);
    Inc(Checked);
  end;
  AssertTrue('must have FD-checked two layers', Checked = 2);
  Student.ClearDeltas();

  Trainer.Free;
  Student.Free;
  Teacher.Free;
  Input.Free;
end;

initialization
  RegisterTest(TTestNeuralKD);
end.
