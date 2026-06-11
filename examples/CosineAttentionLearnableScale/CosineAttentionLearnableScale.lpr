program CosineAttentionLearnableScale;
(*
CosineAttentionLearnableScale: trains the single learnable `scale` scalar of
TNNetCosineSimilarityAttention on a tiny synthetic next-token task and prints
its trajectory, to check whether training drives it toward the cargo-culted
1/tau temperatures used in cosine-attention papers.

Background: cosine attention bounds the pre-softmax logits to [-scale, +scale].
Papers commonly hard-code scale = 1/tau with tau ~ 0.05..0.1 (so scale ~ 10..20)
because, with unit-norm Q/K, a fixed scale of 1.0 makes the softmax too flat to
ever become confident. This experiment makes `scale` a learnable scalar (init
1.0) and watches whether gradient descent on a peaked-attention task pushes it
up toward those large values on its own.

The model is a single cosine-attention layer over a 3*d_k packed Q|K|V tensor;
the target is a sharp copy from one specific key position, which can only be
matched if the softmax is sharp -> scale must grow. No downloads, no data files.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

  procedure RunAlgo();
  const
    SeqLen = 4;
    Dk = 6;
    Epochs = 4000;
    LR = 0.5;
  var
    NN: TNNet;
    Attn: TNNetCosineSimilarityAttention;
    Input, Desired: TNNetVolume;
    i, d, epoch, q, kpos: integer;
    diff, loss: TNeuralFloat;
  begin
    Randomize;
    RandSeed := 1234;
    NN := TNNet.Create();
    Input := TNNetVolume.Create(SeqLen, 1, 3 * Dk);
    Desired := TNNetVolume.Create(SeqLen, 1, Dk);
    try
      NN.AddLayer(TNNetInput.Create(SeqLen, 1, 3 * Dk, 1));
      // Learnable scale, initialised to the fixed default 1.0.
      Attn := TNNetCosineSimilarityAttention.Create(Dk, {causal=}false,
        {scale=}1.0, {learnable=}true);
      NN.AddLayer(Attn);
      NN.SetLearningRate(LR, 0.0);
      NN.SetBatchUpdate(false);

      // Build a task that REWARDS a sharp softmax. Each query q is made to
      // point (in cosine direction) at exactly one key kpos = (q+1) mod SeqLen
      // by giving query q and key kpos the same one-hot-ish direction (a 1.0 in
      // slot (q mod Dk)), while the other keys point elsewhere. With unit-norm
      // Q/K the correct pair has cos ~ 1 and the rest cos ~ 0, so loss drops
      // only as the softmax sharpens -> only as `scale` grows.
      Input.Fill(0);
      for q := 0 to SeqLen - 1 do
      begin
        kpos := (q + 1) mod SeqLen;
        // Q[q]: one-hot in slot (q mod Dk).
        Input[q, 0, q mod Dk] := 1.0;
        // K[kpos] (depth offset Dk): same direction as Q[q].
        Input[kpos, 0, Dk + (q mod Dk)] := 1.0;
        // V[kpos] (depth offset 2*Dk): a distinct payload per key.
        for d := 0 to Dk - 1 do
          Input[kpos, 0, 2 * Dk + d] := Sin((kpos * Dk + d) * 0.7);
      end;

      // Desired output for query q = the V of its matching key kpos, attained
      // only when the query attends almost entirely to that one key.
      for q := 0 to SeqLen - 1 do
      begin
        kpos := (q + 1) mod SeqLen;
        for d := 0 to Dk - 1 do
          Desired[q, 0, d] := Input[kpos, 0, 2 * Dk + d];
      end;

      WriteLn('Cosine-attention learnable-scale trajectory');
      WriteLn('(papers hard-code scale = 1/tau ~ 10..20; init here = 1.0)');
      WriteLn('epoch       scale        loss');
      for epoch := 0 to Epochs do
      begin
        NN.Compute(Input);
        // MSE loss + its derivative as the output error.
        loss := 0;
        for i := 0 to Desired.Size - 1 do
        begin
          diff := NN.GetLastLayer.Output.Raw[i] - Desired.Raw[i];
          loss := loss + 0.5 * diff * diff;
        end;
        if (epoch mod 400 = 0) then
          WriteLn(Format('%5d   %10.4f   %10.6f',
            [epoch, Attn.Scale, loss]));
        NN.Backpropagate(Desired);
      end;
      WriteLn;
      WriteLn(Format('Final learned scale = %.4f', [Attn.Scale]));
      WriteLn('If this climbed well above 1.0 toward ~10, training rediscovered');
      WriteLn('the cargo-culted 1/tau temperature on its own.');
    finally
      NN.Free;
      Input.Free;
      Desired.Free;
    end;
  end;

var
  Application: record Title:string; end;

begin
  Application.Title:='Cosine Attention Learnable Scale';
  RunAlgo();
end.
