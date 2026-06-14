program KnowledgeDistillation;
(*
KnowledgeDistillation: end-to-end demo of TNeuralKDTrainer
(neural/neuralkd.pas), classic Hinton knowledge distillation.

A LARGER char-level next-token TEACHER is Pascal-trained on a structured
synthetic corpus until it is a non-trivial predictor. A SMALL student is then
trained two ways, at MATCHED steps / examples-seen and identical RNG
initialisation:

  (A) WITH KD  -- soft teacher targets blended with the hard label
                  (alpha < 1, temperature T > 1), via TNeuralKDTrainer.
  (B) HARD-ONLY -- the SAME trainer with alpha = 1 (the soft term vanishes,
                  so this is an ordinary cross-entropy SGD step; see the
                  TestAlphaOneMatchesPlainCE unit test). Same data order,
                  same learning rate, same number of Step() calls.

Both students are then evaluated with TNNet.PerplexityReport on a held-out
slice of the stream. The KD student should reach LOWER perplexity than the
hard-label-only student at the same budget: the teacher's softened
distribution carries "dark knowledge" (relative probabilities of the wrong
classes) that a single hard label cannot.

The teacher here is a small Pascal-built model so the whole demo runs on pure
CPU in well under a minute and inside a 3 GB memory cap. To distill from a
REAL pretrained teacher (GPT-2 / TinyStories) instead, import it with the
safetensors / torch.bin importers and pass it as the teacher (see README.md);
the only requirement is that teacher and student share the vocabulary width
and both end in TNNetFullConnectLinear(Vocab) -> SoftMax.

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
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralkd;

const
  cVocab          = 12;
  cContextLen     = 8;
  cTeacherHidden  = 96;    // teacher is comfortably larger than the student
  cStudentHidden  = 12;    // small student
  cStudentEmbed   = 8;
  cTeacherEmbed   = 24;
  cTeacherPasses  = 120;   // train the teacher well first
  cTeacherLR      = 0.05;
  cStudentSteps   = 6000;  // MATCHED budget for both students (Step() calls)
  cStudentLR      = 0.05;
  cKDAlpha        = 0.3;   // 0.3*hard CE + 0.7*soft KL
  cKDTemperature  = 3.0;
  cTrainStreamN   = 1024;
  cEvalStreamN    = 512;

  // A structured-but-not-trivial stream: a deterministic mixing recurrence
  // over the vocabulary. Each token depends on the two preceding ones, so the
  // next-token distribution is genuinely learnable (not uniform, not a single
  // trivial period), giving the teacher real "dark knowledge" to transfer.
  procedure MakeStream(out S: array of integer; Seed: integer);
  var
    I, A, B: integer;
  begin
    A := Seed mod cVocab;
    B := (Seed * 7 + 3) mod cVocab;
    for I := 0 to High(S) do
    begin
      S[I] := (A * 5 + B * 3 + 1) mod cVocab;
      A := B;
      B := S[I];
    end;
  end;

  procedure BuildTeacher(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cContextLen, 1, 1));
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cTeacherEmbed));
    NN.AddLayer(TNNetFullConnectReLU.Create(cTeacherHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cTeacherHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cVocab));   // logits z_t
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(cTeacherLR, 0.9);
  end;

  procedure BuildStudent(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cContextLen, 1, 1));
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cStudentEmbed));
    NN.AddLayer(TNNetFullConnectReLU.Create(cStudentHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cVocab));   // logits z_s
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(cStudentLR, 0.9);
  end;

  // Ordinary supervised SGD to train the (frozen-afterwards) teacher.
  procedure TrainTeacher(NN: TNNet; const Stream: array of integer);
  var
    Input, Target: TNNetVolume;
    Pass, T, D: integer;
  begin
    Input := TNNetVolume.Create(cContextLen, 1, 1);
    Target := TNNetVolume.Create(NN.GetLastLayer().Output);
    try
      for Pass := 1 to cTeacherPasses do
        for T := cContextLen to High(Stream) do
        begin
          for D := 0 to cContextLen - 1 do
            Input.FData[D] := Stream[T - cContextLen + D];
          Target.Fill(0);
          Target.FData[Stream[T]] := 1.0;
          NN.Compute(Input);
          NN.Backpropagate(Target);
        end;
    finally
      Target.Free;
      Input.Free;
    end;
  end;

  // Train a student through TNeuralKDTrainer for exactly cStudentSteps Step()
  // calls over the stream (looping the stream if needed). The SAME procedure
  // is used for the KD student (alpha<1) and the hard-label-only student
  // (alpha=1), so the two runs differ ONLY in the soft-target term.
  procedure DistillStudent(Trainer: TNeuralKDTrainer;
    const Stream: array of integer; const Title: string);
  var
    Input: TNNetVolume;
    Step, T, D: integer;
    AccLoss, AccHard, AccKL: TNeuralFloat;
  begin
    Input := TNNetVolume.Create(cContextLen, 1, 1);
    AccLoss := 0; AccHard := 0; AccKL := 0;
    try
      WriteLn(Title, ' (alpha=', FloatToStrF(Trainer.Alpha, ffFixed, 3, 2),
        ', T=', FloatToStrF(Trainer.Temperature, ffFixed, 3, 1),
        ', ', cStudentSteps, ' steps):');
      for Step := 0 to cStudentSteps - 1 do
      begin
        // Slide a window across the (looped) stream.
        T := cContextLen + (Step mod (Length(Stream) - cContextLen));
        for D := 0 to cContextLen - 1 do
          Input.FData[D] := Stream[T - cContextLen + D];
        Trainer.Step(Input, Stream[T]);
        AccLoss := AccLoss + Trainer.LastLoss;
        AccHard := AccHard + Trainer.LastHardLoss;
        AccKL := AccKL + Trainer.LastKL;
        if (Step + 1) mod 1500 = 0 then
        begin
          WriteLn('  step ', Step + 1:5,
            '  mean blended=', FloatToStrF(AccLoss / 1500, ffFixed, 8, 4),
            '  hardCE=', FloatToStrF(AccHard / 1500, ffFixed, 8, 4),
            '  KL=', FloatToStrF(AccKL / 1500, ffFixed, 8, 4));
          AccLoss := 0; AccHard := 0; AccKL := 0;
        end;
      end;
    finally
      Input.Free;
    end;
  end;

var
  Teacher, StudentKD, StudentHard: TNNet;
  TrainerKD, TrainerHard: TNeuralKDTrainer;
  TrainStream: array[0..cTrainStreamN - 1] of integer;
  EvalStream: array[0..cEvalStreamN - 1] of integer;
  TeacherPPL, KDPPL, HardPPL: string;
begin
  WriteLn('Knowledge-distillation demo: char-level next-token model.');
  WriteLn('Teacher hidden=', cTeacherHidden, '  student hidden=', cStudentHidden,
    '  vocab=', cVocab, '  context=', cContextLen);
  WriteLn;

  MakeStream(TrainStream, 1);
  MakeStream(EvalStream, 99);   // held-out stream (different seed)

  // ---- 1. Train the teacher -------------------------------------------------
  RandSeed := 2026;
  BuildTeacher(Teacher);
  WriteLn('Training teacher (', cTeacherPasses, ' passes over ',
    cTrainStreamN, ' tokens) ...');
  TrainTeacher(Teacher, TrainStream);
  WriteLn('Teacher trained. Held-out perplexity:');
  TeacherPPL := TNNet.PerplexityReport(Teacher, EvalStream, cContextLen, 0);
  Write(TeacherPPL);
  WriteLn;

  // ---- 2. Two identical small students (same RNG init) ----------------------
  RandSeed := 555;
  BuildStudent(StudentKD);
  RandSeed := 555;
  BuildStudent(StudentHard);

  // ---- 3a. KD student: soft teacher targets + hard label --------------------
  TrainerKD := TNeuralKDTrainer.Create(Teacher, StudentKD,
    cKDAlpha, cKDTemperature);
  DistillStudent(TrainerKD, TrainStream, 'Student WITH KD');
  WriteLn;

  // ---- 3b. Hard-label-only student: alpha=1 (soft term off) -----------------
  // Same trainer plumbing, alpha=1 -> pure cross-entropy SGD, MATCHED steps.
  TrainerHard := TNeuralKDTrainer.Create(Teacher, StudentHard,
    {alpha=}1.0, {T=}cKDTemperature);
  DistillStudent(TrainerHard, TrainStream, 'Student HARD-LABEL ONLY');
  WriteLn;

  // ---- 4. Compare held-out perplexity ---------------------------------------
  WriteLn(StringOfChar('=', 78));
  WriteLn('Held-out perplexity comparison (lower is better):');
  WriteLn(StringOfChar('=', 78));
  KDPPL := TNNet.PerplexityReport(StudentKD, EvalStream, cContextLen, 0);
  HardPPL := TNNet.PerplexityReport(StudentHard, EvalStream, cContextLen, 0);
  WriteLn('--- Student WITH KD ---');
  Write(KDPPL);
  WriteLn('--- Student HARD-LABEL ONLY ---');
  Write(HardPPL);
  WriteLn;
  WriteLn('Expect: the KD student reaches LOWER perplexity than the ',
    'hard-label-only student at the SAME number of steps. The soft teacher ',
    'targets regularise the small student toward the teacher''s relative ',
    'class probabilities, which generalises better to the held-out stream ',
    'than memorising hard labels alone (uniform baseline = ', cVocab, ').');

  TrainerHard.Free;
  TrainerKD.Free;
  StudentHard.Free;
  StudentKD.Free;
  Teacher.Free;
end.
