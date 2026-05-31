program TransformerDecoderBlock;
(*
TransformerDecoderBlock: a tiny end-to-end demonstration of the
TNNet.AddTransformerDecoderBlock builder. It wires a (toy) encoder
output into a single transformer DECODER block and runs a forward +
backward pass, printing the output shape and a short toy-train loss
curve so you can see the block learn end to end -- without a full
seq2seq pipeline around it.

A transformer decoder block stacks three residual sub-blocks on the
decoder stream:
  1. CAUSAL multi-head self-attention (position i sees only <= i).
  2. Multi-head CROSS-attention: Query from the decoder stream,
     Key|Value from the encoder output.
  3. Token-wise SwiGLU feed-forward.
Every projection is token-wise (1x1 conv), so the (SeqLen,1,d_model)
sequence axis is preserved and the output shape matches the decoder
input.

Wiring (two input branches):
  DecoderInput(SeqLen,1,d_model)   -- the decoder stream
  EncoderOutput(KVSeqLen,1,d_model)-- the (toy) encoder memory, K|V source
  AddTransformerDecoderBlock(d_model, Heads, d_ff, EncoderOutput)

Toy task: regress a deterministic function of both the decoder input
and the encoder memory, just to show finite gradients flowing through
the whole block. This is NOT a meaningful language task -- it only
exercises the plumbing quickly.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cDModel  = 8;   // model width
  cHeads   = 2;   // attention heads (d_model must be divisible by Heads)
  cDFF     = 16;  // feed-forward inner width
  cSeqLen  = 4;   // decoder sequence length
  cKVSeqLen = 5;  // encoder-memory length (may differ from decoder length)
  cEpochs  = 200; // toy-train iterations (tiny, runs in a blink)

  procedure FillSeq(V: TNNetVolume; Phase: TNeuralFloat);
  var i: integer;
  begin
    for i := 0 to V.Size - 1 do
      V.Raw[i] := Sin(i * 0.53 + Phase) * 0.9 + 0.1;
  end;

  procedure RunAlgo();
  var
    NN: TNNet;
    DecIn, EncOut, BlockOut: TNNetLayer;
    DecData, EncData, Desired: TNNetVolume;
    Epoch, i: integer;
    Loss, diff: TNeuralFloat;
  begin
    RandSeed := 42;
    NN := TNNet.Create();

    // Two input branches: the decoder stream and the (toy) encoder memory.
    DecIn  := NN.AddLayer(TNNetInput.Create(cSeqLen, 1, cDModel, 1));
    EncOut := NN.AddLayerAfter(TNNetInput.Create(cKVSeqLen, 1, cDModel, 1), 0);
    // Bring the active layer back to the decoder stream before building the
    // block (AddLayerAfter above left the encoder input as the last layer).
    NN.AddLayerAfter(TNNetIdentity.Create(), DecIn);

    // The whole decoder block in a single call. Q from the decoder stream,
    // K|V from EncOut. PreNorm=True (default).
    BlockOut := NN.AddTransformerDecoderBlock(cDModel, cHeads, cDFF, EncOut);

    NN.SetLearningRate(0.01, 0.0);
    NN.SetBatchUpdate(true);

    WriteLn('Decoder block structure:');
    NN.DebugStructure();
    WriteLn;
    WriteLn('Decoder input shape : (', cSeqLen, ',1,', cDModel, ')');
    WriteLn('Encoder memory shape: (', cKVSeqLen, ',1,', cDModel, ')');
    WriteLn('Block output shape  : (', BlockOut.Output.SizeX, ',',
      BlockOut.Output.SizeY, ',', BlockOut.Output.Depth, ')');
    WriteLn;

    DecData := TNNetVolume.Create(cSeqLen, 1, cDModel);
    EncData := TNNetVolume.Create(cKVSeqLen, 1, cDModel);
    Desired := TNNetVolume.Create(cSeqLen, 1, cDModel);

    // One fixed toy (decoder input, encoder memory, target) triple.
    FillSeq(DecData, 0.0);
    FillSeq(EncData, 1.3);
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.31);

    // Forward once and confirm the output is finite.
    DecIn.Output.Copy(DecData);
    EncOut.Output.Copy(EncData);
    NN.Compute(DecIn.Output);
    WriteLn('First-forward output sample: ',
      BlockOut.Output.Raw[0]:8:5, ', ', BlockOut.Output.Raw[1]:8:5, ', ',
      BlockOut.Output.Raw[2]:8:5);
    WriteLn;

    WriteLn('Toy training (forward + backward through the whole block):');
    for Epoch := 1 to cEpochs do
    begin
      DecIn.Output.Copy(DecData);
      EncOut.Output.Copy(EncData);
      NN.Compute(DecIn.Output);

      Loss := 0;
      for i := 0 to BlockOut.Output.Size - 1 do
      begin
        diff := BlockOut.Output.Raw[i] - Desired.Raw[i];
        Loss := Loss + 0.5 * diff * diff;
      end;

      NN.Backpropagate(Desired);
      NN.UpdateWeights();

      if (Epoch = 1) or (Epoch mod 50 = 0) then
        WriteLn('  epoch ', Epoch:4, '  loss=', Loss:9:6);
    end;
    WriteLn;
    WriteLn('Done. The decreasing loss shows gradients flow end to end.');

    DecData.Free;
    EncData.Free;
    Desired.Free;
    NN.Free;
  end;

var
  // Stops Lazarus errors
  Application: record Title:string; end;

begin
  Application.Title:='Transformer Decoder Block Example';
  RunAlgo();
end.
