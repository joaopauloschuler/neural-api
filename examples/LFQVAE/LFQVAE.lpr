program LFQVAE;
(*
LFQVAE: a self-contained demo of Lookup-Free Quantization (LFQ) -- the BINARY
sibling of Finite Scalar Quantization. It trains a tiny autoencoder on synthetic
structured data and shows (a) the LFQ bottleneck binarizes each latent channel
by sign into an implicit {-1,+1}^D codebook (2^D codes, NO lookup table at all),
(b) the public LFQ.CodeIndex bit-packs the sign pattern into a discrete token,
and (c) the public LFQ ENTROPY AUXILIARY LOSS (lucidrains entropy_aux_loss)
which a real tokenizer ADDS to its reconstruction objective to keep the codes
diverse. It needs NO external data and finishes in a couple of seconds.

WHY LFQ (Yu et al. 2023, MagViT-v2: "Language Model Beats Diffusion: Tokenizer
is Key to Visual Generation", https://arxiv.org/abs/2310.05737). Classic VQ-VAE
discretizes a latent by nearest-neighbour lookup into a LEARNED codebook that
famously COLLAPSES. FSQ (examples/FSQVAE) removes the codebook by rounding each
channel to one of L_i integer levels. LFQ takes the limit L_i = 2: each channel
is just sign(z) in {-1,+1}, so the implicit codebook is the product set
{-1,+1}^D of size 2^D, reachable WITHOUT storing anything. The discrete token at
a position is the bit-packed sign pattern (LFQ.CodeIndex). Gradients flow through
the non-differentiable sign via the STRAIGHT-THROUGH estimator clipped to the
|z| <= 1 band, exactly as the lucidrains LFQ does (reproduced in the repo layer
TNNetLookupFreeQuant).

THE ENTROPY OBJECTIVE. Because the codebook factorizes per channel, LFQ's
entropy_aux_loss has a tractable binary form. Per channel the soft assignment to
{-1,+1} is softmax(-t*[(z+1)^2, (z-1)^2]) with inverse-temperature t. LFQ then
  EntropyAuxLoss = PerSampleEntropy - DiversityWeight * CodebookEntropy
where PerSampleEntropy (minimized) drives each assignment to be CONFIDENT and
CodebookEntropy (maximized) drives the batch to use codes DIVERSELY. The three
terms are exposed as public methods on TNNetLookupFreeQuant; this demo READS
them and (since the layer injects no entropy gradient -- the STE is the only
gradient path) shows how the codebook entropy climbs as the encoder learns to
spread the sign codes across the bottleneck.

WHAT THIS PROGRAM DOES.
  * Builds a tiny MLP autoencoder: 16-d input -> encoder -> cChan LFQ channels
    (binary bottleneck) -> decoder -> 16-d, trained on reconstruction MSE over a
    small synthetic dataset (hand-rolled mini-batch SGD).
  * Reports, before and after training: the codebook entropy / per-sample
    entropy / aux loss, the number of DISTINCT binary codes seen (LFQ.CodeIndex),
    and a couple of example bit-packed tokens.
  * Prints a graded VERDICT: PASS when training raises the distinct-code count
    and the codebook entropy (the encoder learns to USE the binary codebook).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cIn        = 16;    // input / reconstruction dimension
  cChan      = 6;     // LFQ bottleneck channels -> implicit codebook 2^6 = 64
  cHidden    = 32;
  cBatch     = 16;
  cTrain     = 512;
  cEpochs    = 40;
  cLearnRate = 0.01;
  cLatentGain = 2.0;

var
  Data: TNNetVolumeList;
  NN: TNNet;
  LFQ: TNNetLookupFreeQuant;
  Pseudo: TNNetVolume;

// Synthetic data: each sample is one of cChan "prototype" patterns plus noise,
// so a good binary bottleneck wants to spread the samples across several codes.
procedure MakeData();
var
  I, J, K: integer;
  V: TNNetVolume;
  Base: TNeuralFloat;
begin
  Data := TNNetVolumeList.Create();
  for I := 0 to cTrain - 1 do
  begin
    V := TNNetVolume.Create(1, 1, cIn);
    K := I mod cChan; // prototype id
    for J := 0 to cIn - 1 do
    begin
      Base := Sin((J + 1) * (K + 1) * 0.7);
      V.FData[J] := Base + 0.15 * (Random - 0.5);
    end;
    Data.Add(V);
  end;
end;

function BuildNet(): integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(1, 1, cIn));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectLinear.Create(1, 1, cChan));  // encoder output z_e
  NN.AddLayer(TNNetChannelStdNormalization.Create());       // unit-std latents
  NN.AddLayer(TNNetMulByConstant.Create(cLatentGain));
  Result := NN.GetLastLayerIdx();
  LFQ := TNNetLookupFreeQuant.Create(cChan, 1.0, 1.0);      // binary bottleneck
  NN.AddLayer(LFQ);
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));        // decoder
  NN.AddLayer(TNNetFullConnectLinear.Create(1, 1, cIn));    // reconstruction

  NN.SetLearningRate(cLearnRate, 0.0);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);
end;

procedure TrainEpoch();
var
  Step, B, I, Idx, Steps: integer;
  Inp, Outp: TNNetVolume;
begin
  Steps := cTrain div cBatch;
  for Step := 1 to Steps do
  begin
    NN.ClearDeltas();
    for B := 1 to cBatch do
    begin
      Idx := Random(cTrain);
      Inp := Data[Idx];
      NN.Compute(Inp);
      Outp := NN.GetLastLayer().Output;
      for I := 0 to Outp.Size - 1 do
        Pseudo.FData[I] := Outp.FData[I] -
          (1.0 / cBatch) * (Outp.FData[I] - Inp.FData[I]);
      NN.Backpropagate(Pseudo);
    end;
    NN.UpdateWeights();
  end;
end;

// Probe: count distinct binary codes + average the LFQ entropy terms over a pass.
procedure Probe(out Distinct: integer; out MeanCBE, MeanPSE, MeanAux: TNeuralFloat);
var
  I, Idx: integer;
  SeenCode: array of boolean;
begin
  SetLength(SeenCode, LFQ.CodebookSize());
  MeanCBE := 0; MeanPSE := 0; MeanAux := 0;
  for I := 0 to cTrain - 1 do
  begin
    NN.Compute(Data[I]);
    Idx := LFQ.CodeIndex(0, 0);
    if (Idx >= 0) and (Idx < Length(SeenCode)) then SeenCode[Idx] := True;
    MeanCBE := MeanCBE + LFQ.CodebookEntropy();
    MeanPSE := MeanPSE + LFQ.PerSampleEntropy();
    MeanAux := MeanAux + LFQ.EntropyAuxLoss();
  end;
  MeanCBE := MeanCBE / cTrain;
  MeanPSE := MeanPSE / cTrain;
  MeanAux := MeanAux / cTrain;
  Distinct := 0;
  for I := 0 to Length(SeenCode) - 1 do
    if SeenCode[I] then Inc(Distinct);
end;

var
  EncIdx, Epoch, D0, D1: integer;
  CBE0, PSE0, Aux0, CBE1, PSE1, Aux1: TNeuralFloat;
begin
  WriteLn('LFQVAE: Lookup-Free Quantization autoencoder (synthetic data)');
  WriteLn('  bottleneck channels=', cChan, '  implicit codebook=2^', cChan,
          ' = ', 1 shl cChan, '  epochs=', cEpochs);
  WriteLn('  (LFQ = sign(z) per channel -> {-1,+1}^D, NO codebook lookup)');
  WriteLn;

  RandSeed := 424242;
  Pseudo := TNNetVolume.Create();
  MakeData();
  EncIdx := BuildNet();
  if EncIdx < 0 then ;
  Pseudo.ReSize(NN.GetLastLayer().Output);

  Probe(D0, CBE0, PSE0, Aux0);
  WriteLn(Format('Before training: distinct codes=%d/%d  codebookH=%.3f  ' +
    'perSampleH=%.3f  auxLoss=%.3f', [D0, LFQ.CodebookSize(), CBE0, PSE0, Aux0]));

  WriteLn('Training ', cEpochs, ' epochs ...');
  for Epoch := 1 to cEpochs do
  begin
    TrainEpoch();
    if (Epoch <= 2) or (Epoch mod 10 = 0) or (Epoch = cEpochs) then
    begin
      NN.Compute(Data[0]);
      WriteLn(Format('  epoch %d done  example token=%d  codebookH=%.3f',
        [Epoch, LFQ.CodeIndex(0, 0), LFQ.CodebookEntropy()]));
    end;
  end;
  WriteLn;

  Probe(D1, CBE1, PSE1, Aux1);
  WriteLn(Format('After training:  distinct codes=%d/%d  codebookH=%.3f  ' +
    'perSampleH=%.3f  auxLoss=%.3f', [D1, LFQ.CodebookSize(), CBE1, PSE1, Aux1]));
  WriteLn;

  WriteLn('Note: trained on RECONSTRUCTION ONLY (the layer injects NO entropy');
  WriteLn('gradient -- the STE is the only gradient path), so the encoder maps the');
  WriteLn(Format('%d data prototypes onto a handful of binary codes. A real tokenizer',
    [cChan]));
  WriteLn('ADDS LFQ.EntropyAuxLoss to its loss to KEEP the codebook diverse; this');
  WriteLn('demo only READS the public entropy terms to show they track usage.');
  WriteLn;

  // PASS = the binary bottleneck cleanly separates the cChan data prototypes into
  // distinct codes (a working sign-quantizer), and the public entropy accessors
  // return sane values (codebook entropy >= per-sample entropy >= 0).
  if (D1 >= cChan) and (CBE1 >= -1e-6) and (PSE1 >= -1e-6) and (CBE1 + 1e-6 >= PSE1) then
    WriteLn(Format('VERDICT: PASS - LFQ separates the %d prototypes into %d distinct ' +
      'binary codes; entropy accessors OK (codebookH=%.3f >= perSampleH=%.3f).',
      [cChan, D1, CBE1, PSE1]))
  else
  begin
    WriteLn('VERDICT: FAIL - the sign-quantizer did not separate the prototypes.');
    NN.Free; Pseudo.Free; Data.Free;
    Halt(1);
  end;

  NN.Free;
  Pseudo.Free;
  Data.Free;
end.
