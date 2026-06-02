program KnowledgeDistillation;
(*
KnowledgeDistillation: demonstrates the TNNetKLDivergence loss head doing
knowledge distillation on a SYNTHETIC multi-class toy, contrasted against a
hard-label cross-entropy baseline of identical student capacity. No external
dataset, pure CPU, finishes in well under a second.

WHAT IT SHOWS
-------------
Three classes of 2D Gaussian blobs (K=3) are classified by a small MLP.
First a relatively LARGE "teacher" MLP is trained on hard one-hot labels.
Then, for every training point, the teacher's TEMPERATURE-SOFTENED output
distribution is recorded (softmax of teacher logits / T, with T=3). These
softened distributions carry the teacher's "dark knowledge": the relative
confidences across the wrong classes, not just the argmax.

A SMALLER "student" MLP is then trained TWO ways from the same body:
  (a) DISTILLATION: student head = SoftMax -> TNNetKLDivergence, with the
      teacher's soft distribution as the target. The KL head minimises
      KL(p_teacher || q_student).
  (b) BASELINE:     student head = SoftMax trained against the HARD one-hot
      label via the framework's standard (output - target) cross-entropy
      gradient.

The program prints periodic loss checkpoints for both students and a final
comparison of TEST accuracy: teacher vs distilled-student vs hard-label
student. Numbers are reported honestly: whatever the run produces is printed,
and the README repeats the actual captured output.

HOW THE KL HEAD IS WIRED (read from neural/neuralnetwork.pas)
------------------------------------------------------------
TNNetKLDivergence is a TNNetIdentity descendant. Its doc-comment states its
input q must be a PROBABILITY distribution (e.g. the output of a SoftMax) and
its target p is the reference distribution. Forward is an identity passthrough.
The framework seeds the last layer's FOutputError with (output - target) =
(q - p); Backpropagate recovers p = q - FOutputError and rewrites the residual
with the analytic gradient dL/dq_i = -p_i / q_i. So to distil we:
  - end the student with SoftMax (produces q), then KLDivergence;
  - feed the teacher's softened distribution as the TARGET volume.
The SoftMax then backprops the KL gradient through its Jacobian.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cNumClasses     = 3;
  cSamplesPerCls  = 120;   // training samples per class
  cTestPerCls     = 60;    // held-out test samples per class
  cTeacherEpochs  = 60;
  cStudentEpochs  = 80;
  cTeacherLR      = 0.05;
  cStudentLR      = 0.05;
  cTemperature    = 3.0;   // distillation temperature for softening teacher
  cSeed           = 42;
  cSigma          = 0.85;  // blob spread (deliberately overlapping classes)

  // Three 2D Gaussian blob centers in a triangle, with deliberate overlap so
  // the teacher's soft targets actually carry useful inter-class structure.
  cCenters: array[0..cNumClasses - 1, 0..1] of TNeuralFloat =
    (( 0.0,  1.4),
     (-1.2, -0.8),
     ( 1.2, -0.8));

type
  TSample = record
    X, Y: TNeuralFloat;
    Cls:  integer;
  end;
  TSampleArray = array of TSample;
  // Soft targets: one probability row per training sample.
  TSoftArray = array of array of TNeuralFloat;

// Why: Box-Muller gives N(0,1) samples without an extra dependency.
function RandomGauss(): TNeuralFloat;
var
  U1, U2: TNeuralFloat;
begin
  repeat
    U1 := Random;
  until U1 > 1e-9;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

// Builds a synthetic dataset of PerCls 2D Gaussian points per class.
procedure BuildDataset(out Data: TSampleArray; PerCls: integer);
var
  C, I, Idx: integer;
begin
  SetLength(Data, cNumClasses * PerCls);
  Idx := 0;
  for C := 0 to cNumClasses - 1 do
    for I := 1 to PerCls do
    begin
      Data[Idx].X := cCenters[C][0] + RandomGauss() * cSigma;
      Data[Idx].Y := cCenters[C][1] + RandomGauss() * cSigma;
      Data[Idx].Cls := C;
      Inc(Idx);
    end;
end;

// Loads a single 2D point into the input volume.
procedure FillInput(V: TNNetVolume; const S: TSample);
begin
  V[0, 0, 0] := S.X;
  V[0, 0, 1] := S.Y;
end;

// Teacher: a comparatively wide MLP ending SoftMax (hard-label classifier).
function BuildTeacher(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(1, 1, 2));
  Result.AddLayer(TNNetFullConnectReLU.Create(32));
  Result.AddLayer(TNNetFullConnectReLU.Create(32));
  Result.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  Result.AddLayer(TNNetSoftMax.Create());
  Result.InitWeights();
end;

// Student body: a deliberately SMALLER MLP, shared by both training modes.
// pDistil = True appends SoftMax -> KLDivergence (soft-target distillation);
// pDistil = False appends just SoftMax (hard-label cross-entropy baseline).
function BuildStudent(pDistil: boolean): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(1, 1, 2));
  Result.AddLayer(TNNetFullConnectReLU.Create(6));
  Result.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  Result.AddLayer(TNNetSoftMax.Create());
  if pDistil then
    // KL head consumes the SoftMax probabilities q; target is the teacher dist.
    Result.AddLayer(TNNetKLDivergence.Create());
  Result.InitWeights();
end;

// One epoch of hard-label training: target is a one-hot volume; the SoftMax
// head's (output - target) seed yields the standard cross-entropy gradient.
function TrainHardEpoch(Net: TNNet; const Data: TSampleArray): TNeuralFloat;
var
  Step, NumSamples, Order, Tmp, K: integer;
  Input, Target: TNNetVolume;
  Perm: array of integer;
  P, TotalCE: TNeuralFloat;
begin
  NumSamples := Length(Data);
  Input := TNNetVolume.Create(1, 1, 2);
  Target := TNNetVolume.Create(1, 1, cNumClasses);
  SetLength(Perm, NumSamples);
  for Step := 0 to NumSamples - 1 do Perm[Step] := Step;
  TotalCE := 0;
  try
    // Shuffle for stochastic order.
    for Step := NumSamples - 1 downto 1 do
    begin
      Order := Random(Step + 1);
      Tmp := Perm[Step]; Perm[Step] := Perm[Order]; Perm[Order] := Tmp;
    end;
    for Step := 0 to NumSamples - 1 do
    begin
      Order := Perm[Step];
      FillInput(Input, Data[Order]);
      Target.Fill(0);
      Target[0, 0, Data[Order].Cls] := 1.0;
      Net.Compute(Input);
      Net.Backpropagate(Target);
      // Report cross-entropy -log q[true class] for progress.
      P := Net.GetLastLayer().Output.FData[Data[Order].Cls];
      if P < 1e-7 then P := 1e-7;
      TotalCE := TotalCE - Ln(P);
    end;
  finally
    Target.Free;
    Input.Free;
  end;
  K := NumSamples;
  Result := TotalCE / K;
end;

// One epoch of distillation: target is the teacher's softened distribution.
// Reports mean KL(p_teacher || q_student) for progress.
function TrainSoftEpoch(Net: TNNet; const Data: TSampleArray;
  const Soft: TSoftArray): TNeuralFloat;
var
  Step, NumSamples, Order, Tmp, C: integer;
  Input, Target: TNNetVolume;
  Perm: array of integer;
  Q, Pt, TotalKL: TNeuralFloat;
begin
  NumSamples := Length(Data);
  Input := TNNetVolume.Create(1, 1, 2);
  Target := TNNetVolume.Create(1, 1, cNumClasses);
  SetLength(Perm, NumSamples);
  for Step := 0 to NumSamples - 1 do Perm[Step] := Step;
  TotalKL := 0;
  try
    for Step := NumSamples - 1 downto 1 do
    begin
      Order := Random(Step + 1);
      Tmp := Perm[Step]; Perm[Step] := Perm[Order]; Perm[Order] := Tmp;
    end;
    for Step := 0 to NumSamples - 1 do
    begin
      Order := Perm[Step];
      FillInput(Input, Data[Order]);
      for C := 0 to cNumClasses - 1 do
        Target[0, 0, C] := Soft[Order][C];
      Net.Compute(Input);
      Net.Backpropagate(Target);
      // KL(p||q) = sum p*log(p/q) over classes (SoftMax output is the q head).
      for C := 0 to cNumClasses - 1 do
      begin
        Pt := Soft[Order][C];
        if Pt > 1e-7 then
        begin
          Q := Net.GetLastLayer().Output.FData[C];
          if Q < 1e-7 then Q := 1e-7;
          TotalKL := TotalKL + Pt * Ln(Pt / Q);
        end;
      end;
    end;
  finally
    Target.Free;
    Input.Free;
  end;
  Result := TotalKL / NumSamples;
end;

// Computes the teacher's TEMPERATURE-SOFTENED distribution for each sample.
// We read the teacher's already-softmaxed probabilities, recover logits as
// ln(prob), divide by T, and re-softmax. (ln of a softmax recovers the logits
// up to an additive constant, which softmax is invariant to, so this yields
// exactly softmax(logits / T).)
procedure ComputeSoftTargets(Teacher: TNNet; const Data: TSampleArray;
  out Soft: TSoftArray);
var
  I, C, NumSamples: integer;
  Input: TNNetVolume;
  Logit, MaxL, SumE, E: TNeuralFloat;
  L: array[0..cNumClasses - 1] of TNeuralFloat;
begin
  NumSamples := Length(Data);
  SetLength(Soft, NumSamples, cNumClasses);
  Input := TNNetVolume.Create(1, 1, 2);
  try
    for I := 0 to NumSamples - 1 do
    begin
      FillInput(Input, Data[I]);
      Teacher.Compute(Input);
      MaxL := -1e30;
      for C := 0 to cNumClasses - 1 do
      begin
        Logit := Teacher.GetLastLayer().Output.FData[C];
        if Logit < 1e-7 then Logit := 1e-7;
        L[C] := Ln(Logit) / cTemperature;  // softened logit
        if L[C] > MaxL then MaxL := L[C];
      end;
      SumE := 0;
      for C := 0 to cNumClasses - 1 do
      begin
        E := Exp(L[C] - MaxL);
        L[C] := E;
        SumE := SumE + E;
      end;
      for C := 0 to cNumClasses - 1 do
        Soft[I][C] := L[C] / SumE;
    end;
  finally
    Input.Free;
  end;
end;

// Classification accuracy of a SoftMax-headed net over a dataset.
// For the distillation student the SoftMax is the second-to-last layer (the
// KL head is an identity passthrough), so the argmax is identical either way.
function Accuracy(Net: TNNet; const Data: TSampleArray): TNeuralFloat;
var
  I, C, NumSamples, Pred, Correct: integer;
  Input: TNNetVolume;
  Best, V: TNeuralFloat;
begin
  NumSamples := Length(Data);
  Input := TNNetVolume.Create(1, 1, 2);
  Correct := 0;
  try
    for I := 0 to NumSamples - 1 do
    begin
      FillInput(Input, Data[I]);
      Net.Compute(Input);
      Pred := 0;
      Best := -1e30;
      for C := 0 to cNumClasses - 1 do
      begin
        V := Net.GetLastLayer().Output.FData[C];
        if V > Best then begin Best := V; Pred := C; end;
      end;
      if Pred = Data[I].Cls then Inc(Correct);
    end;
  finally
    Input.Free;
  end;
  Result := Correct / NumSamples;
end;

procedure RunAlgo();
var
  Teacher, StudentKD, StudentHard: TNNet;
  Train, Test: TSampleArray;
  Soft: TSoftArray;
  Epoch: integer;
  Loss, AccTeacher, AccKD, AccHard, AccTeacherTrain: TNeuralFloat;
begin
  RandSeed := cSeed;
  WriteLn('KnowledgeDistillation: KL-divergence distillation vs hard-label baseline');
  WriteLn('Classes: ', cNumClasses, '  train/class: ', cSamplesPerCls,
    '  test/class: ', cTestPerCls, '  temperature: ', cTemperature:0:1);
  WriteLn('(deliberately overlapping blobs, sigma=', cSigma:0:2,
    ', so soft targets carry inter-class structure)');
  WriteLn;

  BuildDataset(Train, cSamplesPerCls);
  BuildDataset(Test, cTestPerCls);

  // ---- 1. Train the teacher on hard labels. -------------------------------
  Teacher := BuildTeacher();
  Teacher.SetLearningRate(cTeacherLR, 0.9);
  WriteLn('[1] Training teacher (32-32 MLP) on hard labels for ',
    cTeacherEpochs, ' epochs...');
  for Epoch := 1 to cTeacherEpochs do
  begin
    Loss := TrainHardEpoch(Teacher, Train);
    if (Epoch = 1) or (Epoch mod 15 = 0) then
      WriteLn(Format('    epoch %3d   train_CE=%8.4f', [Epoch, Loss]));
  end;
  AccTeacherTrain := Accuracy(Teacher, Train);
  AccTeacher := Accuracy(Teacher, Test);
  WriteLn(Format('    teacher train_acc=%6.3f   test_acc=%6.3f',
    [AccTeacherTrain, AccTeacher]));
  WriteLn;

  // ---- 2. Record the teacher's softened distributions as soft targets. ----
  WriteLn('[2] Computing teacher temperature-softened soft targets (T=',
    cTemperature:0:1, ')...');
  ComputeSoftTargets(Teacher, Train, Soft);
  WriteLn(Format('    example soft target for a train point: [%5.3f %5.3f %5.3f]',
    [Soft[0][0], Soft[0][1], Soft[0][2]]));
  WriteLn;

  // ---- 3a. Distillation student (SoftMax -> KLDivergence). ----------------
  StudentKD := BuildStudent(True);
  StudentKD.SetLearningRate(cStudentLR, 0.9);
  WriteLn('[3a] Training DISTILLATION student (6-unit MLP, KL head) for ',
    cStudentEpochs, ' epochs...');
  for Epoch := 1 to cStudentEpochs do
  begin
    Loss := TrainSoftEpoch(StudentKD, Train, Soft);
    if (Epoch = 1) or (Epoch mod 20 = 0) then
      WriteLn(Format('    epoch %3d   mean_KL=%8.4f', [Epoch, Loss]));
  end;
  WriteLn;

  // ---- 3b. Hard-label baseline student (same body, SoftMax + CE). ---------
  StudentHard := BuildStudent(False);
  StudentHard.SetLearningRate(cStudentLR, 0.9);
  WriteLn('[3b] Training HARD-LABEL student (same 6-unit MLP, CE head) for ',
    cStudentEpochs, ' epochs...');
  for Epoch := 1 to cStudentEpochs do
  begin
    Loss := TrainHardEpoch(StudentHard, Train);
    if (Epoch = 1) or (Epoch mod 20 = 0) then
      WriteLn(Format('    epoch %3d   train_CE=%8.4f', [Epoch, Loss]));
  end;
  WriteLn;

  // ---- 4. Compare test accuracy. ------------------------------------------
  AccKD := Accuracy(StudentKD, Test);
  AccHard := Accuracy(StudentHard, Test);
  WriteLn('==================== TEST-SET ACCURACY ====================');
  WriteLn(Format('  teacher (32-32)                : %6.3f', [AccTeacher]));
  WriteLn(Format('  student, distilled (KL, 6)     : %6.3f', [AccKD]));
  WriteLn(Format('  student, hard-label (CE, 6)    : %6.3f', [AccHard]));
  WriteLn('===========================================================');
  if AccKD > AccHard + 1e-6 then
    WriteLn('  -> On this run distillation BEAT the hard-label baseline.')
  else if AccKD < AccHard - 1e-6 then
    WriteLn('  -> On this run distillation did NOT beat the hard-label baseline.')
  else
    WriteLn('  -> On this run distillation TIED the hard-label baseline.');
  WriteLn('  (Toy problem; reported numbers are exactly what this run produced.)');

  StudentHard.Free;
  StudentKD.Free;
  Teacher.Free;
end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'KnowledgeDistillation Example';
  RunAlgo();
end.
