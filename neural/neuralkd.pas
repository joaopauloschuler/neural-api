unit neuralkd;
(*
neuralkd: Classic Hinton knowledge distillation (KD) trainer.
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).

--------------------------------------------------------------------------
WHAT THIS UNIT IS
--------------------------------------------------------------------------
Classic Hinton knowledge distillation (Hinton, Vinyals & Dean 2015,
https://arxiv.org/abs/1503.02531; the "transformers DistillationTrainer"):
a small trainable STUDENT net is trained to mimic a large FROZEN TEACHER
net while still fitting the ordinary hard labels. The blended loss is

  L = alpha * CE(hard_label, softmax(z_s))
    + (1-alpha) * T^2 * KL( softmax(z_t/T) || softmax(z_s/T) )

where z_s / z_t are the STUDENT / TEACHER logits (the pre-softmax layer
output), T >= 1 is the distillation temperature that softens both
distributions, and alpha in [0,1] blends the hard-label cross-entropy with
the soft-target KL term. The T^2 multiplier compensates for the 1/T^2
shrinkage that softening introduces into the soft-target gradient, so the
two terms keep comparable magnitudes as T changes.

GRADIENT DERIVATION (what Step() backpropagates into the STUDENT logits)
--------------------------------------------------------------------------
Write p1 = softmax(z_s) (the hard-label, T=1 student distribution),
pS = softmax(z_s/T) and qS = softmax(z_t/T) (the temperature-softened
student / teacher distributions). The standard results are

  d CE(onehot, p1) / d z_s            = (p1 - onehot)
  d KL(qS || pS)  / d z_s             = (1/T) * (pS - qS)

(the KL term's logit gradient carries one explicit 1/T from the chain rule
d(z_s/T)/dz_s; the cross-entropy-style (pS - qS) shape is the usual softmax
gradient). Multiplying the soft term by the (1-alpha)*T^2 weight,

  dL/d z_s = alpha*(p1 - onehot) + (1-alpha)*T^2*(1/T)*(pS - qS)
           = alpha*(p1 - onehot) + (1-alpha)*T*(pS - qS).

This is the dense logit gradient this unit injects.

HOW THE GRADIENT IS INJECTED (frozen teacher, student-only update)
--------------------------------------------------------------------------
Both nets must end in a LINEAR logit layer followed by a SoftMax layer
(see EXPECTED MODEL SHAPE). The teacher is run forward only (Compute), so
its weights never move - it is frozen exactly like the DPO reference net.

For the student, this unit does NOT backpropagate through the trailing
softmax layer (whose Jacobian would re-map a dense error in an inconvenient
way). Instead it computes the analytic logit gradient dL/dz_s above and
writes it straight into the student's LINEAR logit layer's OutputError,
then calls that layer's Backpropagate(). A TNNetFullConnectLinear has the
identity activation derivative and uses OutputError directly as the logit
gradient (Delta := -LR*OutputError outer PrevOutput), so the gradient that
reaches the student weights is EXACTLY dL/dz_s. The softmax layer is never
touched on the backward pass. This matches the framework's sign convention
OutputError = output - target = +dL/dlogit (gradient descent then does
w -= LR*dL/dw).

EXPECTED MODEL SHAPE
--------------------------------------------------------------------------
Teacher and student must share the same VocabSize and produce a SINGLE
distribution per forward pass, ending in:
  ... -> TNNetFullConnectLinear(Vocab)   { the logits z }
      -> TNNetSoftMax / TNNetPointwiseSoftMax
This is the examples/TinyGPT / examples/SimpleNLP next-token head as well
as any plain classifier with a linear+softmax tail. The two nets need NOT
share input shape or hidden architecture - only the logit width (Vocab)
must agree. The hard label is a class/token id in [0, Vocab).

API
--------------------------------------------------------------------------
  Trainer := TNeuralKDTrainer.Create(Teacher, Student, {alpha=}0.5, {T=}2.0);
  Loss := Trainer.Step(Input, HardLabel);     // one student SGD KD update
After any Step/ComputeLoss the scalar diagnostics are available:
  Trainer.LastLoss      the blended loss L
  Trainer.LastHardLoss  CE(hard_label, softmax(z_s))
  Trainer.LastKL        KL(softmax(z_t/T) || softmax(z_s/T))
ComputeLoss() is forward-only (no update). AccumulateGradients() does
forward+backward WITHOUT ClearDeltas/UpdateWeights (the student must
already be in batch-update mode) for gradient inspection / mini-batching.
Weight updates use the student's per-layer learning rate/inertia
(set them with Student.SetLearningRate beforehand).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, neuralnetwork, neuralvolume, pascoremath32;

type
  /// Trainer-level helper implementing classic Hinton knowledge distillation:
  /// a frozen teacher's temperature-softened logits supervise a trainable
  /// student through a KL term blended with the ordinary hard-label loss.
  // Coded by Claude (AI).
  TNeuralKDTrainer = class(TObject)
    private
      FTeacher: TNNet;
      FStudent: TNNet;
      FOwnsTeacher: boolean;
      FAlpha: TNeuralFloat;
      FTemperature: TNeuralFloat;
      FProbFloor: TNeuralFloat;
      FLastLoss: TNeuralFloat;
      FLastHardLoss: TNeuralFloat;
      FLastKL: TNeuralFloat;
      // Cached softened/hard distributions and logit gradient (Vocab-sized).
      FStudentSoft: TNNetVolume;   // softmax(z_s / T)
      FTeacherSoft: TNNetVolume;   // softmax(z_t / T)
      FStudentHard: TNNetVolume;   // softmax(z_s)      (T = 1)
      FLogitGrad: TNNetVolume;     // dL/dz_s
      // Returns the LINEAR logit layer (the one feeding the trailing softmax)
      // and validates the net ends in linear-logits -> softmax.
      function LogitLayer(NN: TNNet): TNNetLayer;
      // Softmax of (Logits / Temp) into Dest (numerically stable).
      procedure SoftenLogits(Logits: TNNetVolume; Temp: TNeuralFloat;
        Dest: TNNetVolume);
    public
      constructor Create(pTeacher, pStudent: TNNet;
        pAlpha: TNeuralFloat = 0.5; pTemperature: TNeuralFloat = 2.0;
        pOwnsTeacher: boolean = false);
      destructor Destroy(); override;

      // Forward-only blended KD loss for one (input, hard-label) example;
      // refreshes LastLoss/LastHardLoss/LastKL. Runs both nets forward.
      function ComputeLoss(pInput: TNNetVolume;
        HardLabel: integer): TNeuralFloat;
      // Forward + backward: computes the loss, builds dL/dz_s and backprops it
      // into the STUDENT only (accumulates deltas; requires the student to be
      // in batch-update mode; does NOT ClearDeltas nor UpdateWeights).
      function AccumulateGradients(pInput: TNNetVolume;
        HardLabel: integer): TNeuralFloat;
      // One full KD SGD step on one example (batch mode + ClearDeltas +
      // backward + UpdateWeights). Returns the blended loss.
      function Step(pInput: TNNetVolume; HardLabel: integer): TNeuralFloat;

      property Alpha: TNeuralFloat read FAlpha write FAlpha;
      property Temperature: TNeuralFloat read FTemperature write FTemperature;
      property ProbFloor: TNeuralFloat read FProbFloor write FProbFloor;
      property Teacher: TNNet read FTeacher;
      property Student: TNNet read FStudent;
      property LastLoss: TNeuralFloat read FLastLoss;
      property LastHardLoss: TNeuralFloat read FLastHardLoss;
      property LastKL: TNeuralFloat read FLastKL;
  end;

implementation

{ TNeuralKDTrainer }

constructor TNeuralKDTrainer.Create(pTeacher, pStudent: TNNet;
  pAlpha: TNeuralFloat; pTemperature: TNeuralFloat; pOwnsTeacher: boolean);
begin
  inherited Create();
  FTeacher := pTeacher;
  FStudent := pStudent;
  FOwnsTeacher := pOwnsTeacher;
  FAlpha := pAlpha;
  FTemperature := pTemperature;
  FProbFloor := 1e-9;
  FLastLoss := 0; FLastHardLoss := 0; FLastKL := 0;
  FStudentSoft := TNNetVolume.Create();
  FTeacherSoft := TNNetVolume.Create();
  FStudentHard := TNNetVolume.Create();
  FLogitGrad := TNNetVolume.Create();
  // Validate both nets up-front (raises if the tail is not linear -> softmax).
  LogitLayer(FTeacher);
  LogitLayer(FStudent);
end;

destructor TNeuralKDTrainer.Destroy();
begin
  FLogitGrad.Free;
  FStudentHard.Free;
  FTeacherSoft.Free;
  FStudentSoft.Free;
  if FOwnsTeacher then FTeacher.Free;
  inherited Destroy();
end;

function TNeuralKDTrainer.LogitLayer(NN: TNNet): TNNetLayer;
var
  Last: TNNetLayer;
begin
  Last := NN.GetLastLayer();
  if not ( (Last is TNNetSoftMax) or (Last is TNNetPointwiseSoftMax) ) then
    raise Exception.Create(
      'TNeuralKDTrainer requires the net to end in a softmax layer ' +
      '(TNNetSoftMax or TNNetPointwiseSoftMax). Found: ' + Last.ClassName + '.');
  Result := Last.PrevLayer;
  if Result = nil then
    raise Exception.Create(
      'TNeuralKDTrainer: the softmax layer has no preceding logit layer.');
  if not (Result is TNNetFullConnectLinear) then
    raise Exception.Create(
      'TNeuralKDTrainer requires a TNNetFullConnectLinear logit layer ' +
      'immediately before the softmax. Found: ' + Result.ClassName + '. ' +
      'Use TNNetFullConnectLinear(Vocab) as the LM/classifier head so the ' +
      'analytic logit gradient can be injected directly.');
end;

procedure TNeuralKDTrainer.SoftenLogits(Logits: TNNetVolume;
  Temp: TNeuralFloat; Dest: TNNetVolume);
var
  I, LogitsSizeM1: integer;
  MaxLogit, SumExp, V: TNeuralFloat;
begin
  if Dest.Size <> Logits.Size then Dest.ReSize(Logits);
  LogitsSizeM1 := Logits.Size - 1;
  // Numerically stable softmax over (Logits / Temp).
  MaxLogit := Logits.FData[0];
  for I := 1 to LogitsSizeM1 do
    if Logits.FData[I] > MaxLogit then MaxLogit := Logits.FData[I];
  MaxLogit := MaxLogit / Temp;
  SumExp := 0;
  for I := 0 to LogitsSizeM1 do
  begin
    V := NeuralExp((Logits.FData[I] / Temp) - MaxLogit);
    Dest.FData[I] := V;
    SumExp := SumExp + V;
  end;
  for I := 0 to LogitsSizeM1 do
    Dest.FData[I] := Dest.FData[I] / SumExp;
end;

function TNeuralKDTrainer.ComputeLoss(pInput: TNNetVolume;
  HardLabel: integer): TNeuralFloat;
var
  StudentLogits, TeacherLogits: TNNetVolume;
  I, TeacherSoftSizeM1: integer;
  PHard: TNeuralFloat;
begin
  // Forward passes. The teacher is run forward only -> never updated.
  FStudent.Compute(pInput);
  FTeacher.Compute(pInput);
  StudentLogits := LogitLayer(FStudent).Output;
  TeacherLogits := LogitLayer(FTeacher).Output;

  // Hard-label cross-entropy uses the T=1 student distribution.
  SoftenLogits(StudentLogits, 1.0, FStudentHard);
  PHard := Max(FStudentHard.FData[HardLabel], FProbFloor);
  FLastHardLoss := -Ln(PHard);

  // Soft KL term uses the temperature-softened distributions.
  SoftenLogits(StudentLogits, FTemperature, FStudentSoft);
  SoftenLogits(TeacherLogits, FTemperature, FTeacherSoft);
  // KL(qS || pS) = sum_i qS_i * (ln qS_i - ln pS_i).
  FLastKL := 0;
  TeacherSoftSizeM1 := FTeacherSoft.Size - 1;
  for I := 0 to TeacherSoftSizeM1 do
    FLastKL := FLastKL + FTeacherSoft.FData[I] *
      ( pcr_logf(Max(FTeacherSoft.FData[I], FProbFloor)) -
        pcr_logf(Max(FStudentSoft.FData[I], FProbFloor)) );

  FLastLoss := FAlpha * FLastHardLoss
             + (1 - FAlpha) * Sqr(FTemperature) * FLastKL;
  Result := FLastLoss;
end;

function TNeuralKDTrainer.AccumulateGradients(pInput: TNNetVolume;
  HardLabel: integer): TNeuralFloat;
var
  Logit: TNNetLayer;
  I, LogitGradSizeM1: integer;
  HardW, SoftW: TNeuralFloat;
begin
  Result := ComputeLoss(pInput, HardLabel);
  // dL/dz_s = alpha*(p1 - onehot) + (1-alpha)*T*(pS - qS).
  if FLogitGrad.Size <> FStudentHard.Size then FLogitGrad.ReSize(FStudentHard);
  HardW := FAlpha;
  SoftW := (1 - FAlpha) * FTemperature;   // (1-alpha)*T^2 * (1/T)
  LogitGradSizeM1 := FLogitGrad.Size - 1;
  for I := 0 to LogitGradSizeM1 do
    FLogitGrad.FData[I] :=
      HardW * FStudentHard.FData[I] +
      SoftW * (FStudentSoft.FData[I] - FTeacherSoft.FData[I]);
  FLogitGrad.FData[HardLabel] := FLogitGrad.FData[HardLabel] - HardW; // -onehot

  // Inject the analytic logit gradient straight into the student's LINEAR
  // logit layer and backprop FROM it (the trailing softmax is skipped). The
  // student must already be in batch-update mode (Step handles that).
  Logit := LogitLayer(FStudent);
  // ResetBackpropCallCurrCnt zeroes every layer's OutputError and resets the
  // per-pass call counters. The logit layer already has DepartingBranchesCnt=1
  // (the trailing softmax was added on top of it), so a single Backpropagate()
  // call from it fires immediately.
  FStudent.ResetBackpropCallCurrCnt();
  Logit.OutputError.Copy(FLogitGrad);           // OutputError := +dL/dz_s
  Logit.Backpropagate();
end;

function TNeuralKDTrainer.Step(pInput: TNNetVolume;
  HardLabel: integer): TNeuralFloat;
begin
  FStudent.SetBatchUpdate(true);
  FStudent.ClearDeltas();
  Result := AccumulateGradients(pInput, HardLabel);
  FStudent.UpdateWeights();
  FStudent.SetBatchUpdate(false);
end;

end.
