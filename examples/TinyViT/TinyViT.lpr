program TinyViT;
(*
TinyViT: a tiny Vision Transformer image-classification demo of
TNNet.AddPatchEmbedding, the reusable ViT-style patchify + token-projection
builder (Dosovitskiy et al. 2021, "An Image is Worth 16x16 Words",
https://arxiv.org/abs/2010.11929).

The builder turns a 2D image into a token sequence in one call:
  AddPatchEmbedding(PatchSize, EmbedDim, AddClassToken, AddPositionalEmbedding)
    1. patchify    : conv with kernel = stride = PatchSize -> EmbedDim channels,
                     i.e. each PatchSize x PatchSize patch becomes one token,
    2. flatten     : the (GridX,GridY,EmbedDim) patch grid -> (SeqLen,1,EmbedDim),
    3. class token : (optional) a learnable [CLS] token prepended at position 0,
    4. pos embed   : (optional) a learnable absolute positional embedding.
The result is a standard (SeqLen[+1],1,EmbedDim) token sequence ready for the
existing AddTransformerEncoderBlock stack - replacing the conv-stride-then-
reshape boilerplate every patch-tokenizing example used to hand-roll inline.

Task (which-quadrant classification, needs spatial token mixing):
  Each example is a GRID x GRID single-channel image of small noise with one
  bright SPIKE planted in a random cell. The 4 labels are the quadrant the
  spike lands in (top-left / top-right / bottom-left / bottom-right). Deciding
  the class needs comparing token POSITIONS across the patch grid - exactly
  what the positional embedding + self-attention provide. A tiny ViT learns it
  in a few seconds on CPU.

Network:
  Input(IMG,IMG,1)
    -> AddPatchEmbedding(PATCH, EMBED, AddClassToken=true)  -> (SeqLen+1,1,EMBED)
    -> AddTransformerEncoderBlock(HEADS, FFN) x NUM_BLOCKS
    -> LayerNorm
    -> SplitChannels first token (the [CLS] token)  -> (1,1,EMBED)
    -> Reshape(EMBED) -> FullConnectLinear(NUM_CLASSES) -> SoftMax

We print a per-epoch train loss + train/test accuracy trace; it converges to
near-perfect accuracy. This is a small CPU toy (well under a minute) that
demonstrates the builder rather than chasing SOTA. Printing is NaN/Inf-guarded.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  IMG          = 8;      // 8x8 image
  PATCH        = 2;      // 2x2 patches -> 4x4 = 16 patch tokens
  GRID         = IMG div PATCH;
  NUM_PATCHES  = GRID * GRID;
  EMBED        = 16;     // token / residual-stream width
  HEADS        = 2;
  FFN          = 32;     // transformer feed-forward width
  NUM_BLOCKS   = 2;
  NUM_CLASSES  = 4;      // which quadrant the spike lands in
  TRAIN_SIZE   = 400;
  TEST_SIZE    = 100;
  NUM_EPOCHS   = 80;
  LR           = 0.003;
  SPIKE        = 3.0;
  SEED         = 42;

type
  TSample = record
    X: TNNetVolume;       // (IMG,IMG,1) image
    Y: TNNetVolume;       // (NUM_CLASSES) one-hot target
    Cls: integer;
  end;

var
  TrainSet, TestSet: array of TSample;

function SafeF(v: TNeuralFloat): string;
begin
  if IsNan(v) or IsInfinite(v)
    then Result := '   nan/inf'
    else Result := Format('%8.5f', [v]);
end;

procedure MakeSample(var S: TSample);
var
  x, y, px, py: integer;
begin
  S.X := TNNetVolume.Create(IMG, IMG, 1);
  S.Y := TNNetVolume.Create(NUM_CLASSES);
  S.X.Fill(0);
  S.Y.Fill(0);
  // Small background noise so the readout cannot cheat on intensity alone.
  for x := 0 to IMG - 1 do
    for y := 0 to IMG - 1 do
      S.X[x, y, 0] := (Random - 0.5) * 0.2;
  // Plant a bright spike at one random pixel; the label is its quadrant.
  px := Random(IMG);
  py := Random(IMG);
  S.X[px, py, 0] := SPIKE;
  S.Cls := Ord(px >= IMG div 2) + 2 * Ord(py >= IMG div 2);
  S.Y.Raw[S.Cls] := 1.0;
end;

procedure BuildData;
var i: integer;
begin
  SetLength(TrainSet, TRAIN_SIZE);
  SetLength(TestSet, TEST_SIZE);
  for i := 0 to TRAIN_SIZE - 1 do MakeSample(TrainSet[i]);
  for i := 0 to TEST_SIZE - 1 do MakeSample(TestSet[i]);
end;

procedure FreeData;
var i: integer;
begin
  for i := 0 to High(TrainSet) do begin TrainSet[i].X.Free; TrainSet[i].Y.Free; end;
  for i := 0 to High(TestSet) do begin TestSet[i].X.Free; TestSet[i].Y.Free; end;
end;

function BuildNet: TNNet;
var b: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(IMG, IMG, 1));
  // ViT patch embedding in one call: patchify + project to EMBED tokens, with a
  // learnable [CLS] token prepended and a learnable positional embedding added.
  Result.AddPatchEmbedding(PATCH, EMBED, {AddClassToken=}true,
    {AddPositionalEmbedding=}true);
  // Standard pre-norm transformer encoder stack over the token sequence.
  for b := 0 to NUM_BLOCKS - 1 do
    Result.AddTransformerEncoderBlock(HEADS, FFN, {PreNorm=}true,
      {CausalMask=}false);
  Result.AddLayer(TNNetLayerNorm.Create());
  // Classify from the [CLS] token: it is the FIRST token on the X (sequence)
  // axis, so crop a 1-wide slice at X=0 -> (1,1,EMBED), then the dense head.
  Result.AddLayer(TNNetCrop.Create(0, 0, 1, 1));
  Result.AddLayer(TNNetFullConnectLinear.Create(NUM_CLASSES));
  Result.AddLayer(TNNetSoftMax.Create());
  Result.InitWeights();
end;

function Accuracy(NN: TNNet; const Data: array of TSample): TNeuralFloat;
var
  i, hit: integer;
begin
  hit := 0;
  for i := 0 to High(Data) do
  begin
    NN.Compute(Data[i].X);
    if NN.GetLastLayer.Output.GetClass() = Data[i].Cls then Inc(hit);
  end;
  Result := hit / Length(Data);
end;

var
  NN: TNNet;
  epoch, i: integer;
  loss, trainAcc, testAcc: TNeuralFloat;
begin
  // Mask FPU exceptions so a diverging run reports NaN/Inf instead of crashing.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow,
    exUnderflow, exPrecision]);
  RandSeed := SEED;
  WriteLn('Tiny Vision Transformer which-quadrant demo (AddPatchEmbedding)');
  WriteLn('  Image=', IMG, 'x', IMG, ' Patch=', PATCH, ' Tokens=', NUM_PATCHES,
    '(+1 cls) Embed=', EMBED, ' Heads=', HEADS, ' blocks=', NUM_BLOCKS);
  WriteLn('  train=', TRAIN_SIZE, ' test=', TEST_SIZE, ' epochs=', NUM_EPOCHS);
  BuildData;
  try
    RandSeed := SEED;
    NN := BuildNet;
    NN.SetLearningRate(LR, 0.9);
    NN.SetBatchUpdate(false);
    WriteLn('  trainable weights = ', NN.CountWeights());
    WriteLn;
    WriteLn(' epoch    train loss   train acc    test acc');
    for epoch := 0 to NUM_EPOCHS - 1 do
    begin
      loss := 0;
      for i := 0 to High(TrainSet) do
      begin
        NN.Compute(TrainSet[i].X);
        loss := loss - Ln( Max(NN.GetLastLayer.Output.Raw[TrainSet[i].Cls], 1e-7) );
        NN.Backpropagate(TrainSet[i].Y);
      end;
      loss := loss / TRAIN_SIZE;
      if (epoch mod 5 = 0) or (epoch = NUM_EPOCHS - 1) then
      begin
        trainAcc := Accuracy(NN, TrainSet);
        testAcc := Accuracy(NN, TestSet);
        WriteLn('  ', epoch:4, '    ', SafeF(loss), '   ', SafeF(trainAcc),
          '   ', SafeF(testAcc));
      end;
    end;
    WriteLn;
    trainAcc := Accuracy(NN, TrainSet);
    testAcc := Accuracy(NN, TestSet);
    WriteLn('FINAL  train acc = ', SafeF(trainAcc),
      '   test acc = ', SafeF(testAcc));
    if testAcc > 0.85 then
      WriteLn('The tiny ViT solved which-quadrant classification ',
        '(needs positional token mixing).')
    else
      WriteLn('ViT did not fully converge on this run; ',
        'tune epochs / LR / widths.');
    NN.Free;
  finally
    FreeData;
  end;
end.
