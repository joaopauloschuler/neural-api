program ReceptiveFieldReport;
(*
ReceptiveFieldReport: builds two small networks WITHOUT training them and
prints TNNet.ReceptiveFieldReport for each, so reviewers can eyeball how the
receptive field grows layer by layer.

  (i)  a plain VGG-style 3x3 stride-1 convolutional stack (RF grows by 2 per
       3x3 conv, jump stays 1), and
  (ii) a stride-2 downsampling stack (the jump doubles at every stride-2 layer,
       so the receptive field grows aggressively and reaches the whole input
       far sooner).

Purely analytical: no probe batch, no forward pass, no backward pass. Runs in
well under a second.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

  // Plain VGG-style 3x3 stride-1 stack on a 32x32 input.
  procedure BuildVGGStack(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(32, 32, 3));
    NN.AddLayer(TNNetConvolutionReLU.Create({features=}16, {fsize=}3,
      {pad=}1, {stride=}1));
    NN.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(32, 3, 1, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(32, 3, 1, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(10));
    NN.AddLayer(TNNetSoftMax.Create());
  end;

  // Stride-2-downsampling stack on a 32x32 input. Each stride-2 layer doubles
  // the jump, so the receptive field grows fast.
  procedure BuildStride2Stack(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(32, 32, 3));
    NN.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(32, 3, 1, {stride=}2));
    NN.AddLayer(TNNetMaxPool.Create({pool=}2));
    NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 2));
    NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(10));
    NN.AddLayer(TNNetSoftMax.Create());
  end;

var
  VGGNet, Stride2Net: TNNet;
begin
  WriteLn('=== (i) Plain VGG-style 3x3 stride-1 stack (32x32 input) ===');
  BuildVGGStack(VGGNet);
  try
    Write(TNNet.ReceptiveFieldReport(VGGNet));
  finally
    VGGNet.Free;
  end;

  WriteLn;
  WriteLn('=== (ii) Stride-2 downsampling stack (32x32 input) ===');
  BuildStride2Stack(Stride2Net);
  try
    Write(TNNet.ReceptiveFieldReport(Stride2Net));
  finally
    Stride2Net.Free;
  end;
end.
