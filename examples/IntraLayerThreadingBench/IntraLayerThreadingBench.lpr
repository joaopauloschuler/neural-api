program IntraLayerThreadingBench;
(*
IntraLayerThreadingBench: validates the intra-layer threading decision.

  * Experiment A sweeps single-layer FC / pointwise / conv shapes across the
    crossover, reports whether each shape is eligible under the 1M rule, the
    OFF-vs-ON speedup, and a bit-parity check (threaded output MUST equal
    serial output - only independent outputs are partitioned).
  * Experiment B times a few realistic multi-layer nets OFF vs ON end to end.

A speedup > 1.00x means threading pays off at that size; <= 1.00x means the
serial path (or a higher crossover) would be faster. Read the eligible column
against the speedup to judge the 1M line: eligible shapes should show a gain,
sub-threshold shapes should be ones threading would not have helped.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, DateUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralthread;

const
  CROSSOVER = 1024*1024; // the hardcoded 1M work proxy under test

// Warm up, then best-of-5 timing (ms per Compute). Best-of rejects scheduler
// jitter / turbo wobble far better than an average on a noisy box. Parallel
// MUST be threaded through to Compute: EnableIntraLayerThreading only arms the
// chunk path - Compute still runs serial unless Parallel=true is passed, so
// timing the threaded case with the default (serial) overload measures nothing.
function BestTimeMs(NN: TNNet; MyIn: TNNetVolume; Parallel: boolean): double;
var
  r, i, iters: integer;
  StartT: TDateTime;
  el, best: double;
begin
  iters := 1;
  repeat
    StartT := Now();
    for i := 1 to iters do NN.Compute(MyIn, 0, Parallel);
    el := MilliSecondsBetween(Now(), StartT);
    if el >= 150 then break;
    if el < 1 then iters := iters * 8
    else iters := Max(iters + 1, Round(iters * (180 / el)));
  until false;
  iters := Max(1, Round(iters * (80 / Max(el, 1))));
  best := 1e30;
  for r := 0 to 4 do
  begin
    StartT := Now();
    for i := 1 to iters do NN.Compute(MyIn, 0, Parallel);
    el := MilliSecondsBetween(Now(), StartT) / iters;
    if el < best then best := el;
  end;
  Result := best;
end;

// =============================== Experiment A =================================
type
  TShape = record Name: string; Kind, A, B, C, D: integer; end;
var
  ShA: array of TShape;

// Kind 0 = FullConnect(input A -> B neurons)
//      1 = Convolution(input AxAxB, C features, DxD kernel, stride 1, pad 0)
//      2 = PointwiseConv(input AxAxB -> C features)
procedure AddA(const N: string; K, A, B, C, D: integer);
var idx: integer;
begin
  idx := Length(ShA);
  SetLength(ShA, idx + 1);
  ShA[idx].Name := N; ShA[idx].Kind := K;
  ShA[idx].A := A; ShA[idx].B := B; ShA[idx].C := C; ShA[idx].D := D;
end;

function BuildOne(const S: TShape): TNNet;
var NN: TNNet;
begin
  NN := TNNet.Create();
  case S.Kind of
    0: begin NN.AddLayer(TNNetInput.Create(S.A));
             NN.AddLayer(TNNetFullConnectReLU.Create(S.B)); end;
    1: begin NN.AddLayer(TNNetInput.Create(S.A, S.A, S.B));
             NN.AddLayer(TNNetConvolutionReLU.Create(S.C, S.D, 0, 1)); end;
    2: begin NN.AddLayer(TNNetInput.Create(S.A, S.A, S.B));
             NN.AddLayer(TNNetPointwiseConvReLU.Create(S.C)); end;
  end;
  NN.GetLastLayer().InitDefault();
  Result := NN;
end;

// The unified work proxy the eligibility rule compares against 1M.
function WorkProxy(NN: TNNet): int64;
begin
  Result := int64(NN.Layers[0].Output.Size) * NN.GetLastLayer().Output.Size;
end;

procedure ExperimentA;
var
  si, i: integer;
  S: TShape;
  NN: TNNet;
  MyIn, OutSerial: TNNetVolume;
  proxy: int64;
  Ts, Tt, sp, maxdiff, d: double;
  elig: string;
begin
  AddA('FC 256x256',        0, 256,  256,  0, 0);
  AddA('FC 512x512',        0, 512,  512,  0, 0);
  AddA('FC 768x768',        0, 768,  768,  0, 0);
  AddA('FC 1024x1024',      0, 1024, 1024, 0, 0);
  AddA('FC 1536x1536',      0, 1536, 1536, 0, 0);
  AddA('FC 2048x2048',      0, 2048, 2048, 0, 0);
  AddA('FC 3072x3072',      0, 3072, 3072, 0, 0);
  AddA('FC 768x3072 up',    0, 768,  3072, 0, 0);
  AddA('FC 3072x768 dn',    0, 3072, 768,  0, 0);
  AddA('Pw 32x32x256->256', 2, 32, 256, 256, 0);
  AddA('Pw 16x16x512->512', 2, 16, 512, 512, 0);
  AddA('Conv 16x16x64 k3',  1, 16, 64,  128, 3);
  AddA('Conv 32x32x64 k3',  1, 32, 64,  64,  3);
  AddA('Conv 8x8x256 k3',   1, 8,  256, 256, 3);

  WriteLn('=== Experiment A: per-layer  threading OFF vs ON  (1M crossover) ===');
  WriteLn('Cores (NeuralDefaultThreadCount) = ', NeuralDefaultThreadCount());
  WriteLn(Format('%-20s %13s %8s %9s %9s %8s %9s',
    ['shape', 'proxy', '>=1M?', 'off ms', 'on ms', 'speedup', 'maxdiff']));
  WriteLn(StringOfChar('-', 82));
  for si := 0 to High(ShA) do
  begin
    S := ShA[si];
    NN := BuildOne(S);
    MyIn := TNNetVolume.Create(NN.Layers[0].Output);
    MyIn.Randomize();
    proxy := WorkProxy(NN);

    // Serial reference (threading off), then snapshot for the parity check.
    NN.EnableIntraLayerThreading(false);
    NN.Compute(MyIn, 0, {parallel=}false);
    OutSerial := TNNetVolume.Create(NN.GetLastLayer().Output);
    Ts := BestTimeMs(NN, MyIn, {parallel=}false);

    // Same net, same weights, threading on: arm the chunk path AND route
    // Compute through the parallel scheduler (SetTrainable(False) is the
    // inference-only contract the parallel pass expects).
    NN.SetTrainable(False);
    NN.EnableIntraLayerThreading(true);
    NN.Compute(MyIn, 0, {parallel=}true);
    maxdiff := 0;
    for i := 0 to OutSerial.Size - 1 do
    begin
      d := Abs(OutSerial.FData[i] - NN.GetLastLayer().Output.FData[i]);
      if d > maxdiff then maxdiff := d;
    end;
    Tt := BestTimeMs(NN, MyIn, {parallel=}true);

    if Tt > 0 then sp := Ts / Tt else sp := 0;
    if proxy >= CROSSOVER then elig := 'yes' else elig := 'no';
    WriteLn(Format('%-20s %13d %8s %9.3f %9.3f %7.2fx %9.2g',
      [S.Name, proxy, elig, Ts, Tt, sp, maxdiff]));

    OutSerial.Free;
    MyIn.Free;
    NN.Free;
  end;
  WriteLn;
end;

// =============================== Experiment B =================================
// Transformer-decoder-ish forward: a stack of per-token MLP blocks over a short
// sequence - the shape where intra-layer threading is meant to pay off.
function BuildStack(dModel, dHidden, nBlocks, seqLen: integer;
  Threaded: boolean): TNNet;
var NN: TNNet; b: integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(dModel, seqLen, 1));
  for b := 0 to nBlocks - 1 do
  begin
    NN.AddLayer(TNNetPointwiseConvReLU.Create(dHidden));
    NN.AddLayer(TNNetPointwiseConvLinear.Create(dModel));
  end;
  for b := 0 to NN.CountLayers - 1 do NN.Layers[b].InitDefault();
  NN.SetTrainable(False);
  NN.EnableIntraLayerThreading(Threaded);
  Result := NN;
end;

procedure SweepNet(const Tag: string; dModel, dHidden, nBlocks, seqLen: integer);
var
  MyIn: TNNetVolume; NN: TNNet; off, on_: double;
begin
  NN := BuildStack(dModel, dHidden, nBlocks, seqLen, false);
  MyIn := TNNetVolume.Create(NN.Layers[0].Output);
  MyIn.Randomize();
  off := BestTimeMs(NN, MyIn, {parallel=}false);
  NN.Free;

  NN := BuildStack(dModel, dHidden, nBlocks, seqLen, true);
  on_ := BestTimeMs(NN, MyIn, {parallel=}true);
  NN.Free;

  WriteLn(Format('  %-16s dM=%-5d dH=%-5d blk=%d seq=%-3d  off=%8.3f  on=%8.3f  %6.2fx',
    [Tag, dModel, dHidden, nBlocks, seqLen, off, on_, off / on_]));
  MyIn.Free;
end;

begin
  Randomize;
  ExperimentA;
  WriteLn('=== Experiment B: realistic nets  threading OFF vs ON (end-to-end) ===');
  SweepNet('small GPT MLP', 512,  2048, 6, 8);
  SweepNet('mid GPT MLP',   768,  3072, 6, 16);
  SweepNet('wide 1-token',  2048, 8192, 4, 1);
  WriteLn;
  WriteLn('speedup > 1.00x => the 1M threading decision is winning on this box.');
end.
