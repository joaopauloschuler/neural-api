program EffectiveReceptiveField;
(*
EffectiveReceptiveField: builds two small networks WITHOUT training them and
prints TNNet.EffectiveReceptiveFieldReport for each, so reviewers can eyeball
how much of the THEORETICAL receptive field a deep output unit ACTUALLY
weights (Luo et al. 2016: the effective RF is typically far smaller and more
Gaussian than the theoretical window).

This is the EMPIRICAL (gradient-measured) counterpart of the analytical
TNNet.ReceptiveFieldReport: it picks the centre output unit of the final
spatial layer, back-propagates a one-hot output error (reusing
TNNet.EnableInputGradient), and accumulates |d out_centre / d input| over a
synthetic probe batch into a per-(x,y) input-plane heatmap.

  (i)  a STACK of 3x3 stride-1 convolutions (RF grows by 2 per conv): many
       composed 3x3 kernels make the effective RF concentrate sharply near the
       centre, so the effective/theoretical ratio is well below 1, and
  (ii) a SINGLE large-kernel (9x9) convolution: one flat kernel weights its
       whole window much more uniformly, so the effective RF fills a far larger
       fraction of the theoretical window.

The two stems are sized to reach a comparable THEORETICAL RF on the same small
input, so the contrast in the effective/theoretical RATIO is the story.

Forward+backward only (the centre-unit gradient), on a frozen, untrained net:
no weight update, no dataset download. Runs in well under a second.

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
  cInputSize = 15;   // 15x15x1 synthetic input plane
  cProbes    = 24;   // synthetic probe batch size

  // Stack of four 3x3 stride-1 convs on a 15x15x1 input.
  // Theoretical RF = 1 + 4*(3-1) = 9 along each axis.
  procedure BuildStack3x3(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInputSize, cInputSize, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create({features=}8, {fsize=}3,
      {pad=}1, {stride=}1));
    NN.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
    NN.InitWeights();
  end;

  // A single large 9x9 conv on the same 15x15x1 input.
  // Theoretical RF = 9 along each axis (same as the 3x3 stack), but a single
  // flat kernel weights its window far more evenly.
  procedure BuildLargeKernel(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInputSize, cInputSize, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create({features=}8, {fsize=}9,
      {pad=}4, {stride=}1));
    NN.InitWeights();
  end;

  // Deterministic synthetic probe batch (no dataset download): smooth random
  // patterns so the centre-unit gradient is well defined everywhere.
  procedure BuildProbes(out Probes: TNNetVolumeList);
  var
    I, X, Y: integer;
    V: TNNetVolume;
  begin
    Probes := TNNetVolumeList.Create();
    RandSeed := 20260524;
    for I := 1 to cProbes do
    begin
      V := TNNetVolume.Create(cInputSize, cInputSize, 1);
      for Y := 0 to cInputSize - 1 do
        for X := 0 to cInputSize - 1 do
          V[X, Y, 0] := Random - 0.5;
      Probes.Add(V);
    end;
  end;

var
  Stack3x3, LargeKernel: TNNet;
  Probes: TNNetVolumeList;
begin
  BuildProbes(Probes);
  try
    WriteLn('=== (i) Stack of four 3x3 stride-1 convs (15x15x1 input) ===');
    BuildStack3x3(Stack3x3);
    try
      Write(TNNet.EffectiveReceptiveFieldReport(Stack3x3, Probes));
    finally
      Stack3x3.Free;
    end;

    WriteLn;
    WriteLn('=== (ii) Single 9x9 conv (15x15x1 input) ===');
    BuildLargeKernel(LargeKernel);
    try
      Write(TNNet.EffectiveReceptiveFieldReport(LargeKernel, Probes));
    finally
      LargeKernel.Free;
    end;
  finally
    Probes.Free;
  end;
end.
