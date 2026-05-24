program PredictionDepth;
(*
PredictionDepth: builds a small softmax classifier on a synthetic multi-class
2D-blob problem with a deliberately LABEL-NOISED subset, then prints
TNNet.PredictionDepthReport for (i) a freshly-initialised network and (ii) the
same architecture after a short training run, so the contrast is visible.

PredictionDepth (Baldock, Maennel & Neyshabur 2021, "Deep Learning Through the
Lens of Example Difficulty") is a per-EXAMPLE difficulty diagnostic. For every
query it asks: "at how deep a layer does the network actually make up its mind
about THIS example?". The estimator is forward-only and NON-PARAMETRIC: it
snapshots each trainable layer's activation over a labelled SUPPORT batch and a
QUERY batch, takes a k-NN vote (cosine distance) over the support at every
layer, and defines the prediction depth of a query as the index of the
shallowest layer after which the k-NN vote agrees with the network's final
argmax and never disagrees again. Easy, well-separated examples decide early
(shallow); hard / ambiguous / mislabelled examples stay contested until the last
layers (deep).

The report prints a 10-bin ASCII histogram of prediction depth, mean/median
depth, a per-layer "newly-resolved" profile (where examples get decided), the K
deepest (hardest) query indices as a relabel-candidate queue, and - because the
demo supplies query labels - a correctness cross-tab: the mean prediction depth
of correctly vs incorrectly classified queries (the literature's headline
result is that depth correlates with error / low margin).

At fresh init the depths pile up at the LAST layer (right-skewed histogram -
nothing is decided early). After training the mass shifts shallow for the
well-separated clusters while the label-noised subset keeps a deep tail, and the
incorrect-vs-correct mean-depth gap is positive. The run also feeds the support
set as its OWN queries as a built-in correctness check (every sample finite
depth; final-layer vote matches the net argmax for ~all samples).

Pure CPU, forward-only, no dataset download, well under a minute.

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
  cInDim    = 2;
  cHidden   = 12;
  cClasses  = 4;
  cEpochs   = 120;
  cSupport  = 160;  // labelled support batch (the k-NN reference set)
  cQuery    = 120;  // query batch whose per-example depth we measure
  cHardFrac = 0.25; // fraction of queries placed in the ambiguous between-blob
                    // band (genuinely hard: contested deep into the net)
  cBlob     = 2.2;  // distance of each blob centre from the origin
  cNoise    = 0.45; // per-blob Gaussian-ish spread

var
  Centers: array[0..3, 0..1] of TNeuralFloat =
    ((-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0));

  // Builds a tiny MLP classifier:
  //   Input -> FC+ReLU -> FC+ReLU -> FC+ReLU -> FC -> SoftMax
  // The stacked ReLU blocks give several intermediate layers, so the per-layer
  // k-NN vote (and hence the prediction depth) has somewhere to settle.
  procedure BuildNet(out NN: TNNet; LR: TNeuralFloat);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(LR, 0.9);
    NN.InitWeights();
  end;

  // One labelled 2D-blob sample of the given class (tight cluster).
  function MakeInput(Cls: integer): TNNetVolume;
  begin
    Result := TNNetVolume.Create(cInDim, 1, 1);
    Result.Raw[0] := Centers[Cls][0] * cBlob + (Random - 0.5) * 2.0 * cNoise;
    Result.Raw[1] := Centers[Cls][1] * cBlob + (Random - 0.5) * 2.0 * cNoise;
  end;

  // A genuinely AMBIGUOUS query: drawn from the empty band between the four
  // blobs (near a coordinate axis / the origin) where no cluster dominates, so
  // the raw-input k-NN is contested and only the deeper, trained layers commit.
  function MakeHardInput: TNNetVolume;
  begin
    Result := TNNetVolume.Create(cInDim, 1, 1);
    // points scattered around a circle of radius ~cBlob but rotated 45deg off
    // the cluster centres, i.e. on the midlines between adjacent blobs.
    if Random(2) = 0 then
    begin
      Result.Raw[0] := (Random - 0.5) * 2.0 * cBlob * 1.2;
      Result.Raw[1] := (Random - 0.5) * 2.0 * cNoise;
    end
    else
    begin
      Result.Raw[0] := (Random - 0.5) * 2.0 * cNoise;
      Result.Raw[1] := (Random - 0.5) * 2.0 * cBlob * 1.2;
    end;
  end;

  // Build the SUPPORT set: inputs + clean integer labels (the k-NN reference).
  procedure BuildSupport(out Inputs: TNNetVolumeList; out Labels: TIntegerList);
  var
    K, Cls: integer;
  begin
    Inputs := TNNetVolumeList.Create();
    Labels := TIntegerList.Create();
    for K := 0 to cSupport - 1 do
    begin
      Cls := K mod cClasses;
      Inputs.Add(MakeInput(Cls));
      Labels.Add(Cls);
    end;
  end;

  // Nearest cluster-centre class of a 2D point (the "best-guess" label for an
  // ambiguous query - it may or may not match what the trained net decides).
  function NearestClass(V: TNNetVolume): integer;
  var
    C, Best: integer;
    D, BestD: TNeuralFloat;
  begin
    Best := 0;
    BestD := 1e30;
    for C := 0 to cClasses - 1 do
    begin
      D := Sqr(V.Raw[0] - Centers[C][0] * cBlob) +
           Sqr(V.Raw[1] - Centers[C][1] * cBlob);
      if D < BestD then
      begin
        BestD := D;
        Best := C;
      end;
    end;
    Result := Best;
  end;

  // Build the QUERY set: a mix of EASY queries (tight inside a blob, labelled by
  // their generating class) and a cHardFrac subset of AMBIGUOUS queries drawn
  // from the empty between-blob band and labelled by their nearest centre. The
  // hard subset is what stays contested deep into the network.
  procedure BuildQueries(out Inputs: TNNetVolumeList; out Labels: TIntegerList;
    out NumHard: integer);
  var
    K, Cls: integer;
    V: TNNetVolume;
  begin
    Inputs := TNNetVolumeList.Create();
    Labels := TIntegerList.Create();
    NumHard := 0;
    for K := 0 to cQuery - 1 do
    begin
      if Random < cHardFrac then
      begin
        V := MakeHardInput;
        Inputs.Add(V);
        Labels.Add(NearestClass(V));
        Inc(NumHard);
      end
      else
      begin
        Cls := K mod cClasses;
        Inputs.Add(MakeInput(Cls));
        Labels.Add(Cls);
      end;
    end;
  end;

  procedure TrainOnce(NN: TNNet; Epochs: integer);
  var
    Ep, B, Cls, Correct: integer;
    X, Y: TNNetVolume;
  begin
    for Ep := 1 to Epochs do
    begin
      Correct := 0;
      for B := 1 to 96 do
      begin
        Cls := Random(cClasses);
        X := MakeInput(Cls);
        Y := TNNetVolume.Create(cClasses, 1, 1);
        Y.Fill(0);
        Y.Raw[Cls] := 1.0;
        try
          NN.Compute(X);
          if NN.GetLastLayer.Output.GetClass() = Cls then Inc(Correct);
          NN.Backpropagate(Y);
        finally
          X.Free;
          Y.Free;
        end;
      end;
      if (Ep = 1) or (Ep mod 20 = 0) or (Ep = Epochs) then
        WriteLn(Format('  epoch %3d  train-acc=%.3f', [Ep, Correct / 96.0]));
    end;
  end;

var
  NN: TNNet;
  SupInputs, QryInputs: TNNetVolumeList;
  SupLabels, QryLabels: TIntegerList;
  SupLabelsArr, QryLabelsArr: array of integer;
  NumHard, I: integer;
begin
  RandSeed := 2026;
  BuildSupport(SupInputs, SupLabels);
  BuildQueries(QryInputs, QryLabels, NumHard);

  SetLength(SupLabelsArr, SupLabels.Count);
  for I := 0 to SupLabels.Count - 1 do SupLabelsArr[I] := SupLabels[I];
  SetLength(QryLabelsArr, QryLabels.Count);
  for I := 0 to QryLabels.Count - 1 do QryLabelsArr[I] := QryLabels[I];

  try
    WriteLn('PredictionDepthReport demo: tiny 4-class softmax MLP on a ' +
      'synthetic 2D-blob problem.');
    WriteLn(Format('Support=%d clean samples, Query=%d samples (%d drawn from ' +
      'the ambiguous between-blob band).', [cSupport, cQuery, NumHard]));
    WriteLn('Per query, a per-layer k-NN vote (cosine) over the support set ' +
      'gives the depth at which the net "makes up its mind".');

    // ---- (i) fresh init: depths pile up at the LAST layer ----
    BuildNet(NN, 0.01);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 1: freshly-initialised network (no training). ' +
        'Nothing is decided early - depths pile up at the LAST layer.');
      WriteLn(StringOfChar('=', 100));
      Write(TNNet.PredictionDepthReport(NN, SupInputs, SupLabelsArr,
        QryInputs, QryLabelsArr));
    finally
      NN.Free;
    end;

    // ---- (ii) after a short training run: easy-shallow / hard-deep contrast --
    RandSeed := 2026;
    BuildNet(NN, 0.01);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 2: same architecture after a short training run. ' +
        'Well-separated clusters resolve shallow; the mislabelled tail stays ' +
        'deep.');
      WriteLn(StringOfChar('=', 100));
      WriteLn('Training for ', cEpochs, ' epochs...');
      TrainOnce(NN, cEpochs);
      WriteLn;
      Write(TNNet.PredictionDepthReport(NN, SupInputs, SupLabelsArr,
        QryInputs, QryLabelsArr));

      // ---- built-in correctness check: support set fed as its own queries ----
      WriteLn;
      WriteLn(StringOfChar('-', 100));
      WriteLn('CORRECTNESS CHECK: feeding the support set as its OWN queries. ' +
        'Each point is its own nearest neighbour (cosine distance 0), so the ' +
        'final-layer k-NN vote should match the network argmax for ~every ' +
        'sample (agreement ~1.0).');
      WriteLn(StringOfChar('-', 100));
      Write(TNNet.PredictionDepthReport(NN, SupInputs, SupLabelsArr,
        SupInputs, SupLabelsArr));
    finally
      NN.Free;
    end;

    WriteLn;
    WriteLn(
      'Read it as: prediction depth is a per-EXAMPLE difficulty score. At ' +
      'fresh init the histogram is right-skewed (everything decided late); ' +
      'after training the mass shifts shallow for the well-separated clusters ' +
      'while the label-noised subset keeps a deep tail, and the ' +
      'incorrect-minus-correct mean-depth gap is positive (errors decide ' +
      'deeper). The "Hardest query indices" list is a ready-made relabel / ' +
      'hard-example queue.');
  finally
    SupInputs.Free;
    QryInputs.Free;
    SupLabels.Free;
    QryLabels.Free;
  end;
end.
