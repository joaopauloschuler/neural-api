program LinearAttention;
(*
LinearAttention: wall-clock SCALING PROBE for TNNetLinearAttention, the
first sub-quadratic (softmax-free / kernelized) attention layer in this
repo (Katharopoulos et al. 2020, "Transformers are RNNs",
https://arxiv.org/abs/2006.16236).

TNNetLinearAttention replaces softmax with a positive feature map
phi(x) = elu(x)+1 applied to Q and K, then exploits associativity:
  S = sum_s phi(K_s) (x) V_s   (a d_k x d_v matrix, accumulated ONCE)
  Z = sum_s phi(K_s)
  Out_t = (phi(Q_t).S) / (phi(Q_t).Z)
so the per-forward cost is O(SeqLen * d_k * d_v) -- LINEAR in sequence
length -- and the SeqLen x SeqLen score matrix is never materialised.

This probe builds a 1-layer LinearAttention net, runs a forward pass at
SeqLen in {16, 32, 64, 128, 256} with d_k held fixed, times each, and
prints a table. The time-per-token (and the time ratio vs. the previous
row) should stay roughly FLAT as SeqLen doubles -- i.e. total time grows
~linearly, not quadratically. (Quadratic softmax attention would roughly
QUADRUPLE the time each time SeqLen doubles; here it ~doubles.)

Runs in a few seconds on a single CPU thread and uses modest memory.

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
  cDk         = 64;   // fixed feature width; input depth = 3*cDk (Q|K|V)
  cReps       = 200;  // forward passes per SeqLen (averaged for a stable time)

  // Build a 1-layer LinearAttention net for the given sequence length.
  procedure BuildModel(out NN: TNNet; SeqLen: integer);
  begin
    NN := TNNet.Create();
    // Input: SeqLen positions on X, depth = 3*d_k packing Q | K | V.
    NN.AddLayer(TNNetInput.Create(SeqLen, 1, 3 * cDk));
    // The probe target: non-causal linear attention. No softmax, no NxN matrix.
    NN.AddLayer(TNNetLinearAttention.Create(cDk));
  end;

  // Fill the input volume with a deterministic, finite Q|K|V pattern.
  procedure FillInput(V: TNNetVolume);
  var i: integer;
  begin
    for i := 0 to V.Size - 1 do
      V.Raw[i] := Sin(i * 0.017) * 0.8 + 0.1;
  end;

var
  SeqLens: array[0..4] of integer = (16, 32, 64, 128, 256);
  Idx, Rep, SeqLen: integer;
  NN: TNNet;
  Input: TNNetVolume;
  StartTime, TotalMs, PerForwardMs, PerTokenUs: double;
  PrevTotalMs: double;
  RatioStr: string;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;

  WriteLn('TNNetLinearAttention scaling probe');
  WriteLn('  feature width d_k = d_v = ', cDk,
    ' (input depth = 3*d_k = ', 3 * cDk, ')');
  WriteLn('  forward passes timed per SeqLen = ', cReps);
  WriteLn;
  WriteLn('Linear attention is O(SeqLen * d_k * d_v): total forward time');
  WriteLn('should grow ~LINEARLY (time/token roughly flat, time ratio ~2x');
  WriteLn('per SeqLen doubling), NOT quadratically (which would be ~4x).');
  WriteLn;
  WriteLn(StringOfChar('-', 74));
  WriteLn(Format('%8s %14s %16s %14s %10s',
    ['SeqLen', 'total ms', 'ms / forward', 'us / token', 'ratio']));
  WriteLn(StringOfChar('-', 74));

  PrevTotalMs := 0;
  for Idx := 0 to High(SeqLens) do
  begin
    SeqLen := SeqLens[Idx];
    BuildModel(NN, SeqLen);
    Input := TNNetVolume.Create(SeqLen, 1, 3 * cDk);
    try
      FillInput(Input);
      // Warm up (allocation / cache) so it doesn't skew the first timed row.
      NN.Compute(Input);

      StartTime := Now();
      for Rep := 1 to cReps do
        NN.Compute(Input);
      TotalMs := (Now() - StartTime) * 86400.0 * 1000.0;

      PerForwardMs := TotalMs / cReps;
      PerTokenUs := (PerForwardMs * 1000.0) / SeqLen;

      if PrevTotalMs > 0 then
        // Expected ~ (SeqLen / PrevSeq) = 2.0 for linear; ~4.0 for quadratic.
        RatioStr := Format('%.2fx', [TotalMs / PrevTotalMs])
      else
        RatioStr := '   -';

      WriteLn(Format('%8d %14.2f %16.4f %14.3f %10s',
        [SeqLen, TotalMs, PerForwardMs, PerTokenUs, RatioStr]));

      PrevTotalMs := TotalMs;
    finally
      Input.Free;
      NN.Free;
    end;
  end;
  WriteLn(StringOfChar('-', 74));
  WriteLn;
  WriteLn('Read the "ratio" column: each row doubles SeqLen, so a value near');
  WriteLn('2x confirms linear scaling. The "us / token" column should stay');
  WriteLn('roughly constant -- the hallmark of sub-quadratic attention.');
end.
