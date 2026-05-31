program GatherChannelsRouting;
(*
GatherChannelsRouting: a tiny, self-contained forward-pass demo of the
TNNet.AddGatherChannels builder for channel routing / pruning.

A small stem (PointwiseConv) emits an 8-channel feature map. We then use the
one-line TNNet.AddGatherChannels builder to keep / reorder a hand-picked
subset of those channels -- here [5, 1, 1, 6] -- demonstrating:
  - channel pruning  (8 channels -> 4 kept),
  - channel reorder  (channel 5 routed to output position 0, etc.),
  - channel reuse     (channel 1 duplicated into two output positions).

The program prints the stem Depth before and after the gather, the gathered
activation values next to their hand-picked SOURCE channel values, and asserts
they match exactly (gather is a learnable-free copy/route, not a transform),
printing a clear PASS / FAIL. It does NOT train -- a forward pass is enough to
show the routing.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  STEM_CHANNELS = 8;
  // Hand-picked routing: keep channel 5 first, then 1 (twice -> reuse), then 6.
  // Net effect: prune 8 -> 4 output channels, reordered and with one duplicate.
  ROUTE: array[0..3] of integer = (5, 1, 1, 6);

var
  NN: TNNet;
  Input, StemOut: TNNetVolume;
  StemLayer, GatherLayer: TNNetLayer;
  i, k, srcCh: integer;
  gathered, expected: TNeuralFloat;
  allOk: boolean;

begin
  // Pure forward-pass demo: single-threaded, no fitting object to configure.
  WriteLn('GatherChannelsRouting: channel routing / pruning demo');
  WriteLn('------------------------------------------------------');

  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 3);  // 3 raw input features
  try
    // Stem: project 3 raw features up to STEM_CHANNELS channels.
    NN.AddLayer(TNNetInput.Create(1, 1, 3, 1));
    StemLayer := NN.AddLayer(TNNetPointwiseConvReLU.Create(STEM_CHANNELS));

    // One-line builder: keep/reorder/duplicate a channel subset.
    GatherLayer := NN.AddGatherChannels(ROUTE);

    // Deterministic-ish input.
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := (i + 1) * 0.37 - 0.5;

    NN.Compute(Input);

    StemOut := StemLayer.Output;

    WriteLn('Stem output Depth   : ', StemOut.Depth);
    WriteLn('Gathered output Depth: ', GatherLayer.Output.Depth,
      '  (route = [5,1,1,6])');
    WriteLn('');
    WriteLn('out[k]   <- stem ch   gathered        source');
    WriteLn('-----------------------------------------------');

    allOk := True;
    for k := 0 to GatherLayer.Output.Depth - 1 do
    begin
      srcCh := ROUTE[k];
      gathered := GatherLayer.Output[0, 0, k];
      expected := StemOut[0, 0, srcCh];
      WriteLn(Format('out[%d]   <- ch %d    %10.6f    %10.6f',
        [k, srcCh, gathered, expected]));
      if Abs(gathered - expected) > 1e-6 then allOk := False;
    end;

    WriteLn('');
    if allOk and (GatherLayer.Output.Depth = Length(ROUTE)) then
      WriteLn('PASS: gathered channels match the hand-picked source channels.')
    else
      WriteLn('FAIL: gathered channels do not match.');
  finally
    NN.Free;
    Input.Free;
  end;
end.
