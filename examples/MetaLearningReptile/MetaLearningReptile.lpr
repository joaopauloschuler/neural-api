program MetaLearningReptile;
(*
MetaLearningReptile: Reptile first-order meta-learning (Nichol, Achiam &
Schulman 2018, "On First-Order Meta-Learning Algorithms",
https://arxiv.org/abs/1803.02999).

Trains an INITIALIZATION across a DISTRIBUTION of tiny sine-regression tasks
y = A*sin(x+p) (A, p randomised per task), so that a few SGD steps adapt it to
a freshly sampled task. Shows that from the Reptile meta-init, a held-out task
fits in k in {1,2,4} gradient steps to far lower loss than the same k steps from
a random init or from a net pre-trained on the AVERAGE task.

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
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cNumPts     = 20;     // x-samples per sine task (over [-5, 5])
  cInnerLR    = 0.0007; // inner-loop SGD learning rate (stable for the summed
                        // 20-point gradient on this tiny ReLU net)
  cInnerSteps = 16;     // SGD steps used to ADAPT during meta-training
  cEps        = 0.1;    // outer Reptile interpolation step
  cMetaIters  = 10000;  // outer Reptile iterations
  cJointIters = 12000;  // joint-training (average-task baseline) iterations

  function BuildNet: TNNet;
  begin
    Result := TNNet.Create();
    Result.AddLayer([
      TNNetInput.Create(1),
      TNNetFullConnectReLU.Create(16),
      TNNetFullConnectReLU.Create(16),
      TNNetFullConnectLinear.Create(1)
    ]);
  end;

  // Samples a random sine task.
  procedure SampleTask(out A, P: TNeuralFloat);
  begin
    A := 0.5 + Random * 2.5;   // amplitude in [0.5, 3]
    P := Random * Pi;          // phase in [0, pi)
  end;

  // Runs InnerSteps full-batch SGD steps adapting NN to one sine task.
  procedure AdaptToTask(NN: TNNet; A, P: TNeuralFloat; InnerSteps: integer);
  var
    Inp, Des: TNNetVolume;
    Step, I: integer;
    X: TNeuralFloat;
  begin
    Inp := TNNetVolume.Create(1, 1, 1);
    Des := TNNetVolume.Create(1, 1, 1);
    try
      NN.SetLearningRate(cInnerLR, 0.0);
      for Step := 1 to InnerSteps do
      begin
        for I := 0 to cNumPts - 1 do
        begin
          X := -5.0 + 10.0 * (I / (cNumPts - 1));
          Inp.Raw[0] := X * 0.2; // normalise x to [-1,1] for well-conditioned SGD
          Des.Raw[0] := A * Sin(X + P);
          NN.Compute(Inp);
          NN.Backpropagate(Des);
        end;
        NN.UpdateWeights();
      end;
    finally
      Inp.Free;
      Des.Free;
    end;
  end;

  // Mean absolute error of NN on a sine task.
  function TaskLoss(NN: TNNet; A, P: TNeuralFloat): TNeuralFloat;
  var
    Inp, Out: TNNetVolume;
    I: integer;
    X, Err: TNeuralFloat;
  begin
    Inp := TNNetVolume.Create(1, 1, 1);
    Out := TNNetVolume.Create(1, 1, 1);
    Err := 0;
    try
      for I := 0 to cNumPts - 1 do
      begin
        X := -5.0 + 10.0 * (I / (cNumPts - 1));
        Inp.Raw[0] := X * 0.2;
        NN.Compute(Inp);
        NN.GetOutput(Out);
        Err := Err + Abs(Out.Raw[0] - A * Sin(X + P));
      end;
    finally
      Inp.Free;
      Out.Free;
    end;
    Result := Err / cNumPts;
  end;

  // Evaluates: average held-out loss over NumTasks tasks after adapting K steps
  // from a fixed Init network.
  function EvalAtK(Init: TNNet; K, NumTasks: integer; SeedTasks: integer): TNeuralFloat;
  var
    Adapt: TNNet;
    T: integer;
    A, P, Sum: TNeuralFloat;
  begin
    RandSeed := SeedTasks; // same held-out tasks for every init/K
    Sum := 0;
    for T := 0 to NumTasks - 1 do
    begin
      SampleTask(A, P);
      Adapt := BuildNet();
      try
        Adapt.CopyWeights(Init);
        AdaptToTask(Adapt, A, P, K);
        Sum := Sum + TaskLoss(Adapt, A, P);
      finally
        Adapt.Free;
      end;
    end;
    Result := Sum / NumTasks;
  end;

  procedure RunAlgo();
  var
    MetaNet, RandInit, JointNet, Worker: TNNet;
    Trainer: TNNetReptileMetaTrainer;
    Iter: integer;
    A, P: TNeuralFloat;
    Inp, Des: TNNetVolume;
    I: integer;
    X: TNeuralFloat;
    KList: array[0..2] of integer = (1, 2, 4);
    Ki: integer;
    K: integer;
    LMeta, LRand, LJoint: TNeuralFloat;
  const
    cEvalTasks = 20;
    cEvalSeed  = 999777;
  begin
    // This example uses the manual Compute/Backpropagate/UpdateWeights path,
    // which is single-threaded by construction (the MaxThreadNum := 1 budget
    // requested by the spec applies to TNeuralFit, which we do not use here).
    // --- random init baseline (frozen) ---
    RandSeed := 424242;
    RandInit := BuildNet();

    // --- Reptile meta-init (seeded from the same random init) ---
    MetaNet := BuildNet();
    MetaNet.CopyWeights(RandInit);
    Trainer := TNNetReptileMetaTrainer.Create(MetaNet, cEps);

    WriteLn('Reptile meta-training over ', cMetaIters, ' sine tasks...');
    for Iter := 1 to cMetaIters do
    begin
      SampleTask(A, P);
      Worker := Trainer.BeginTask();
      AdaptToTask(Worker, A, P, cInnerSteps);
      // Safeguard: a rare inner-loop divergence yields a non-finite phi; merging
      // it would permanently poison the meta-init theta. Skip those tasks. The
      // worker is reseeded from theta on the next BeginTask, so nothing leaks.
      if not IsNan(Worker.GetWeightSum()) then
        Trainer.MergeTask();
      if (Iter mod 2000) = 0 then
        WriteLn('  meta iter ', Iter, '/', cMetaIters);
    end;

    // --- joint-training baseline: train ONE net on the pooled DISTRIBUTION ---
    // Each update averages the gradient over a batch of DIFFERENT tasks, so the
    // net converges to the best single fit to the whole distribution (the
    // "average task", which here is ~0 since E[A sin(x+p)] = 0). It does NOT
    // learn a fast-adapting init -- that is exactly the contrast with Reptile.
    WriteLn('Joint-training baseline over ', cJointIters, ' steps...');
    JointNet := BuildNet();
    JointNet.CopyWeights(RandInit);
    JointNet.SetLearningRate(cInnerLR, 0.0);
    Inp := TNNetVolume.Create(1, 1, 1);
    Des := TNNetVolume.Create(1, 1, 1);
    try
      for Iter := 1 to cJointIters do
      begin
        // Pool 4 tasks per update -> a stable gradient toward the mean function.
        for I := 0 to 4 * cNumPts - 1 do
        begin
          if (I mod cNumPts) = 0 then SampleTask(A, P);
          X := -5.0 + 10.0 * ((I mod cNumPts) / (cNumPts - 1));
          Inp.Raw[0] := X * 0.2;
          Des.Raw[0] := A * Sin(X + P);
          JointNet.Compute(Inp);
          JointNet.Backpropagate(Des);
        end;
        JointNet.UpdateWeights();
      end;
    finally
      Inp.Free;
      Des.Free;
    end;

    WriteLn;
    WriteLn('Held-out adaptation (mean abs error over ', cEvalTasks,
      ' tasks, lower is better):');
    WriteLn('  k     Reptile-init   Random-init    Joint-init');
    for Ki := 0 to 2 do
    begin
      K := KList[Ki];
      LMeta  := EvalAtK(MetaNet,  K, cEvalTasks, cEvalSeed);
      LRand  := EvalAtK(RandInit, K, cEvalTasks, cEvalSeed);
      LJoint := EvalAtK(JointNet, K, cEvalTasks, cEvalSeed);
      WriteLn('  ', K, '     ',
        LMeta:10:4, '     ', LRand:10:4, '    ', LJoint:10:4);
    end;

    Trainer.Free;
    MetaNet.Free;
    RandInit.Free;
    JointNet.Free;
  end;

var
  Application: record Title: string; end;

begin
  // The neural-api numeric kernels expect FP exceptions masked (denormals/over-
  // flow are saturated, not signalled) — same convention as the other examples.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  Application.Title := 'MetaLearningReptile Example';
  RunAlgo();
end.
