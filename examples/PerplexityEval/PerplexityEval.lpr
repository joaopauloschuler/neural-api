program PerplexityEval;
(*
PerplexityEval: builds a tiny char-level next-token model on a repeating
synthetic alphabet ('abcdefgh'), trains it briefly, then prints
TNNet.PerplexityReport on a held-out portion of the stream. Demonstrates
both auto-detection paths by rebuilding the same stack with a
TNNetLogSoftMax head and reporting again.

Pure CPU, well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cVocab        = 8;
  cContextLen   = 8;
  cEmbedDim     = 16;
  cTrainPasses  = 80;     // tiny — keeps the demo under ~30 seconds CPU
  cLearningRate = 0.05;
  cTrainStreamN = 256;
  cEvalStreamN  = 128;

  procedure MakeRepeatingStream(out S: array of integer);
  var
    I: integer;
  begin
    for I := 0 to High(S) do
      S[I] := I mod cVocab;
  end;

  procedure BuildModel(out NN: TNNet; UseLogSoftMax: boolean);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cContextLen, 1, 1));
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cEmbedDim));
    NN.AddLayer(TNNetFullConnectReLU.Create(32));
    if UseLogSoftMax then
    begin
      // LogSoftMax normalises across the depth axis, so emit the vocab in
      // the depth dimension via the (SizeX, SizeY, Depth) constructor.
      NN.AddLayer(TNNetFullConnectLinear.Create(1, 1, cVocab));
      NN.AddLayer(TNNetLogSoftMax.Create());
    end
    else
    begin
      // TNNetSoftMax normalises across the whole volume, so any axis works.
      NN.AddLayer(TNNetFullConnectLinear.Create(cVocab));
      NN.AddLayer(TNNetSoftMax.Create());
    end;
    NN.SetLearningRate(cLearningRate, 0.9);
  end;

  procedure TrainBriefly(NN: TNNet; const Stream: array of integer);
  // Online SGD over the stream: for each window of length cContextLen we
  // predict the next token. Targets are one-hot; we ReSize the target to
  // the actual output shape so both SoftMax (V,1,1) and LogSoftMax (1,1,V)
  // heads work without further branching. No batching; the whole demo aims
  // at < 30 s CPU.
  var
    Input, Target: TNNetVolume;
    Pass, T, D: integer;
  begin
    Input := TNNetVolume.Create(cContextLen, 1, 1);
    Target := TNNetVolume.Create(NN.GetLastLayer().Output);
    try
      for Pass := 1 to cTrainPasses do
      begin
        for T := cContextLen to High(Stream) do
        begin
          for D := 0 to cContextLen - 1 do
            Input.FData[D] := Stream[T - cContextLen + D];
          Target.Fill(0);
          Target.FData[Stream[T]] := 1.0;
          NN.Compute(Input);
          NN.Backpropagate(Target);
        end;
      end;
    finally
      Target.Free;
      Input.Free;
    end;
  end;

  procedure RunTrained(const Title: string;
    const TrainStream, EvalStream: array of integer);
  // Train a SoftMax head and print the report on a held-out stream. With a
  // perfectly-periodic alphabet, perplexity should be close to 1.0 and
  // top-1 accuracy close to 1.0 after a brief training run.
  var
    NN: TNNet;
    Report: string;
  begin
    RandSeed := 1234;
    BuildModel(NN, False);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 92));
      WriteLn(Title);
      WriteLn(StringOfChar('=', 92));
      WriteLn('Architecture:');
      NN.PrintSummary();
      WriteLn;
      WriteLn('Training (', cTrainPasses, ' passes over ',
        Length(TrainStream), ' tokens) ...');
      TrainBriefly(NN, TrainStream);
      WriteLn('Done. Evaluating on held-out stream of ',
        Length(EvalStream), ' tokens (context ', cContextLen, ').');
      WriteLn;
      Report := TNNet.PerplexityReport(NN, EvalStream, cContextLen, 5);
      Write(Report);
    finally
      NN.Free;
    end;
  end;

  procedure RunLogSoftMaxAutoDetect(const Title: string;
    const TrainStream, EvalStream: array of integer);
  // Same stack but with a TNNetLogSoftMax head (vocab in the depth axis).
  // Exercises the log-space auto-detect path; numbers should be close to
  // the SoftMax run after a brief training pass.
  var
    NN: TNNet;
    Report: string;
  begin
    RandSeed := 1234;
    BuildModel(NN, True);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 92));
      WriteLn(Title);
      WriteLn(StringOfChar('=', 92));
      WriteLn('Architecture:');
      NN.PrintSummary();
      WriteLn;
      WriteLn('Training (', cTrainPasses, ' passes over ',
        Length(TrainStream), ' tokens) ...');
      TrainBriefly(NN, TrainStream);
      WriteLn('Done. Evaluating on held-out stream of ',
        Length(EvalStream), ' tokens (context ', cContextLen, ').');
      WriteLn;
      Report := TNNet.PerplexityReport(NN, EvalStream, cContextLen, 5);
      Write(Report);
    finally
      NN.Free;
    end;
  end;

var
  TrainStream: array[0..cTrainStreamN - 1] of integer;
  EvalStream: array[0..cEvalStreamN - 1] of integer;
begin
  WriteLn('PerplexityEval demo: tiny char-level model on a repeating ',
    cVocab, '-symbol alphabet.');
  MakeRepeatingStream(TrainStream);
  MakeRepeatingStream(EvalStream);
  RunTrained('Trained SoftMax head (probability-space auto-detect)',
    TrainStream, EvalStream);
  RunLogSoftMaxAutoDetect(
    'Trained LogSoftMax head (log-space auto-detect)',
    TrainStream, EvalStream);
  WriteLn;
  WriteLn(
    'Expect: both runs show perplexity well below ', cVocab,
    ' (uniform baseline), top-1 close to 1.0, and a tight per-token bits ',
    'histogram. The two heads exercise different auto-detect paths.');
end.
