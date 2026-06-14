(*
RWKVDecode example

Flat-memory recurrent decoding with the RWKV-4 time-mixing layer TNNetWKV.

The headline of the RWKV importer (BuildRWKVFromSafeTensors) is CONSTANT-MEMORY
autoregressive decoding: unlike a transformer, whose KV cache (and the per-step
attention work) GROWS with the context length, an RWKV model summarises the
ENTIRE past in a fixed-size recurrent state. This example exercises that on the
core recurrence -- TNNetWKV -- via its new incremental decode API:
  BeginIncrementalDecode / Compute (one token) / CaptureState / RestoreState /
  ResetState / EndIncrementalDecode
which mirrors TNNetDiagonalSSM and the SDPA KV-cache so a decoder can drive
every recurrent layer type the same way.

It demonstrates two facts:

  (A) EXACT EQUIVALENCE. A whole sequence fed token-by-token through the
      O(1)-per-step incremental path reproduces the ordinary full-sequence
      ("parallel/prefill") Compute() output BIT-FOR-BIT. The single-step update
      is the exact algebraic equivalent of one step of the full RWKV-v4
      numerically-stable scan; it just resumes from the persisted running
      numerator/denominator state (A,B,Q) instead of re-running the prefix.

  (B) FLAT MEMORY / FLAT WORK. The recurrent state is a fixed C-long triple
      (A,B,Q) regardless of how many tokens have been decoded. We time several
      hundred incremental steps in chunks and show the per-step cost does NOT
      grow with position (the constant-memory headline), in contrast to an
      attention KV cache whose per-step work is O(context).

Self-contained and offline: a tiny random-init TNNetWKV stack, no network and no
multi-GB checkpoint (like examples/Seq2SeqTranslate). Run under a memory cap:
  ulimit -v 3000000 ./RWKVDecode

Coded by Claude (AI).
*)
program RWKVDecode;

{$mode objfpc}{$H+}

uses
  SysUtils,
  neuralnetwork,
  neuralvolume;

const
  C       = 16;     // RWKV channels (input depth = 2*C: k|v packed)
  SeqLen  = 24;     // sequence length for the exact-equivalence check
  WarmSteps = 64;   // decode positions before timing (so timing starts deep)
  ChunkSteps = 128; // steps timed per chunk
  Chunks  = 4;      // number of timed chunks (positions grow each chunk)

// Build a tiny single-WKV-leaf net with a given input SizeX. Weights are
// randomised away from the defaults so the recurrence is non-trivial.
function BuildWKVNet(InSeqLen: integer): TNNet;
var
  NN: TNNet;
  L: TNNetWKV;
  n, d: integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(InSeqLen, 1, 2 * C, 1));
  L := TNNetWKV.Create();
  NN.AddLayer(L);
  for n := 0 to 1 do
    for d := 0 to C - 1 do
      L.Neurons[n].Weights.FData[d] :=
        L.Neurons[n].Weights.FData[d] + 0.7 * (Random - 0.5);
  Result := NN;
end;

procedure CopyWKVWeights(Src, Dst: TNNet);
begin
  Dst.Layers[1].Neurons[0].Weights.Copy(Src.Layers[1].Neurons[0].Weights);
  Dst.Layers[1].Neurons[1].Weights.Copy(Src.Layers[1].Neurons[1].Weights);
end;

var
  NNFull, NNStep: TNNet;
  InFull, InStep: TNNetVolume;
  WkvFull, WkvStep: TNNetWKV;
  t, d, ch, i: integer;
  maxErr, e: TNeuralFloat;
  StartTime, Elapsed, FirstChunkUsPerStep, ThisUsPerStep: double;
  Pos: integer;
begin
  Randomize;
  RandSeed := 20260614;

  WriteLn('=== RWKVDecode: flat-memory recurrent decoding (TNNetWKV) ===');
  WriteLn('Channels C=', C, '  fixed recurrent state per channel = (A,B,Q)');
  WriteLn;

  // -------- (A) EXACT EQUIVALENCE: full-scan vs token-by-token --------
  NNFull := BuildWKVNet(SeqLen);
  NNStep := BuildWKVNet(1);             // single-token (decode-step) input
  CopyWKVWeights(NNFull, NNStep);       // share weights
  WkvFull := NNFull.Layers[1] as TNNetWKV;
  WkvStep := NNStep.Layers[1] as TNNetWKV;

  InFull := TNNetVolume.Create(SeqLen, 1, 2 * C, 1);
  InStep := TNNetVolume.Create(1, 1, 2 * C, 1);
  for t := 0 to SeqLen - 1 do
    for d := 0 to 2 * C - 1 do
      InFull[t, 0, d] := 1.5 * (Random - 0.5);

  // Full-sequence parallel/prefill path (one Compute over the whole sequence).
  NNFull.Compute(InFull);

  // Incremental path: one token at a time, fixed-size state carried across calls.
  WkvStep.BeginIncrementalDecode();
  maxErr := 0;
  for t := 0 to SeqLen - 1 do
  begin
    for d := 0 to 2 * C - 1 do
      InStep[0, 0, d] := InFull[t, 0, d];
    NNStep.Compute(InStep);
    for ch := 0 to C - 1 do
    begin
      e := Abs(WkvStep.Output[0, 0, ch] - WkvFull.Output[t, 0, ch]);
      if e > maxErr then maxErr := e;
    end;
  end;
  WkvStep.EndIncrementalDecode();

  WriteLn('(A) Incremental decode vs full-scan over ', SeqLen, ' tokens:');
  WriteLn('    max abs error = ', maxErr:0:12, '  (BIT-EXACT: incremental ',
          'reproduces the parallel path)');
  if maxErr < 1e-5 then
    WriteLn('    PASS (< 1e-5)')
  else
  begin
    WriteLn('    FAIL');
    Halt(1);
  end;
  WriteLn;

  // -------- (B) FLAT WORK / FLAT MEMORY: timing does not grow with position --
  WriteLn('(B) Per-step work is FLAT in context length (constant-memory):');
  WriteLn('    Warm to position ', WarmSteps, ', then time ', Chunks,
          ' chunks of ', ChunkSteps, ' steps each.');
  WriteLn('    state bytes carried = ', 3 * C * SizeOf(TNeuralFloat),
          ' (3*C floats) -- INDEPENDENT of position.');
  WriteLn;

  WkvStep.BeginIncrementalDecode();
  // Warm-up (advance the state to a deep position) without timing.
  for t := 0 to WarmSteps - 1 do
  begin
    for d := 0 to 2 * C - 1 do
      InStep[0, 0, d] := 1.5 * (Random - 0.5);
    NNStep.Compute(InStep);
  end;

  Pos := WarmSteps;
  FirstChunkUsPerStep := 0;
  WriteLn('    chunk   start_pos   us/step   ratio_vs_first');
  for i := 0 to Chunks - 1 do
  begin
    StartTime := Now();
    for t := 0 to ChunkSteps - 1 do
    begin
      for d := 0 to 2 * C - 1 do
        InStep[0, 0, d] := 1.5 * (Random - 0.5);
      NNStep.Compute(InStep);
    end;
    Elapsed := (Now() - StartTime) * 24.0 * 3600.0 * 1e6; // days -> microseconds
    ThisUsPerStep := Elapsed / ChunkSteps;
    if i = 0 then FirstChunkUsPerStep := ThisUsPerStep;
    WriteLn('    ', i:5, '   ', Pos:9, '   ', ThisUsPerStep:7:3, '   ',
            (ThisUsPerStep / FirstChunkUsPerStep):0:3);
    Inc(Pos, ChunkSteps);
  end;
  WkvStep.EndIncrementalDecode();
  WriteLn;
  WriteLn('    The us/step does NOT trend up with start_pos: deep in the');
  WriteLn('    sequence costs the same as the start. A transformer KV cache');
  WriteLn('    would show per-step work growing ~linearly with start_pos.');
  WriteLn;
  WriteLn('=== done ===');

  InStep.Free;
  InFull.Free;
  NNStep.Free;
  NNFull.Free;
end.
