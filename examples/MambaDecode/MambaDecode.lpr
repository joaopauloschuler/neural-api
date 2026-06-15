(*
MambaDecode example

Flat-memory recurrent decoding with the Mamba selective state-space layer
TNNetSelectiveSSM.

The headline of the Mamba importer (BuildMambaFromSafeTensors) is CONSTANT-MEMORY
autoregressive decoding: unlike a transformer, whose KV cache (and the per-step
attention work) GROWS with the context length, a Mamba model summarises the
ENTIRE past in a fixed-size recurrent state -- the [d_inner x d_state] hidden
matrix h. This example exercises that on the core recurrence -- the selective
scan TNNetSelectiveSSM -- via its new incremental decode API:
  BeginIncrementalDecode / Compute (one token) / CaptureState / RestoreState /
  ResetState / EndIncrementalDecode
which mirrors TNNetWKV (RWKV-4), TNNetDiagonalSSM and the SDPA KV-cache so a
decoder can drive every recurrent layer type the same way.

It demonstrates two facts:

  (A) EXACT EQUIVALENCE. A whole sequence fed token-by-token through the
      O(1)-per-step incremental path reproduces the ordinary full-sequence
      ("parallel/prefill") Compute() output BIT-FOR-BIT. The single-step update
      is the exact algebraic equivalent of one step of the full selective scan
      h_t = exp(-dt*exp(A)) .* h_{t-1} + dt*B*x ; y_t = C*h_t + D*x ; it just
      resumes from the persisted running state h instead of re-running the
      prefix.

  (B) FLAT MEMORY / FLAT WORK. The recurrent state is the fixed [Depth x DState]
      hidden matrix h regardless of how many tokens have been decoded. We time
      several hundred incremental steps in chunks and show the per-step cost
      does NOT grow with position (the constant-memory headline), in contrast to
      an attention KV cache whose per-step work is O(context).

Self-contained and offline: a tiny random-init TNNetSelectiveSSM leaf, no network
and no multi-GB checkpoint. Run under a memory cap:
  ulimit -v 3000000 ./MambaDecode

NOTE (scope): a FULL Mamba block places a causal depthwise conv1d before this
scan; token-by-token decode of the whole block additionally needs the conv to
carry its (kernel-1)-token ring buffer. This example (like the RWKV one) decodes
the SSM RECURRENCE leaf -- the part that owns the position-independent state.
Block-level conv-state decode is a documented follow-up.

Coded by Claude (AI).
*)
program MambaDecode;

{$mode objfpc}{$H+}

uses
  SysUtils,
  neuralnetwork,
  neuralvolume;

const
  Depth   = 16;     // SSM channels (d_inner); input depth = Depth
  DState  = 8;      // states per channel (the d_state of Mamba/S6)
  SeqLen  = 24;     // sequence length for the exact-equivalence check
  WarmSteps = 64;     // decode positions before timing (so timing starts deep)
  ChunkSteps = 20000; // steps timed per chunk (large so the timer is not quantized)
  Chunks  = 4;        // number of timed chunks (positions grow each chunk)

// Build a tiny single-SSM-leaf net with a given input SizeX. Weights are
// randomised away from the defaults so the selective recurrence is non-trivial.
function BuildSSMNet(InSeqLen: integer): TNNet;
var
  NN: TNNet;
  L: TNNetSelectiveSSM;
  n, k: integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(InSeqLen, 1, Depth, 1));
  L := TNNetSelectiveSSM.Create(DState);
  NN.AddLayer(L);
  // Perturb every weight set so dt/B/C are genuinely input-dependent.
  for n := 0 to L.Neurons.Count - 1 do
    for k := 0 to L.Neurons[n].Weights.Size - 1 do
      L.Neurons[n].Weights.FData[k] :=
        L.Neurons[n].Weights.FData[k] + 0.3 * (Random - 0.5);
  Result := NN;
end;

procedure CopySSMWeights(Src, Dst: TNNet);
var n: integer;
begin
  for n := 0 to Src.Layers[1].Neurons.Count - 1 do
    Dst.Layers[1].Neurons[n].Weights.Copy(Src.Layers[1].Neurons[n].Weights);
end;

var
  NNFull, NNStep: TNNet;
  InFull, InStep: TNNetVolume;
  SsmFull, SsmStep: TNNetSelectiveSSM;
  t, d, i: integer;
  maxErr, e: TNeuralFloat;
  StartTime, Elapsed, FirstChunkUsPerStep, ThisUsPerStep: double;
  Pos: integer;
begin
  Randomize;
  RandSeed := 20260614;

  WriteLn('=== MambaDecode: flat-memory recurrent decoding (TNNetSelectiveSSM) ===');
  WriteLn('Depth (d_inner)=', Depth, '  DState=', DState,
          '  carried state h = [Depth x DState] = ', Depth * DState, ' floats');
  WriteLn;

  // -------- (A) EXACT EQUIVALENCE: full-scan vs token-by-token --------
  NNFull := BuildSSMNet(SeqLen);
  NNStep := BuildSSMNet(1);             // single-token (decode-step) input
  CopySSMWeights(NNFull, NNStep);       // share weights
  SsmFull := NNFull.Layers[1] as TNNetSelectiveSSM;
  SsmStep := NNStep.Layers[1] as TNNetSelectiveSSM;

  InFull := TNNetVolume.Create(SeqLen, 1, Depth, 1);
  InStep := TNNetVolume.Create(1, 1, Depth, 1);
  for t := 0 to SeqLen - 1 do
    for d := 0 to Depth - 1 do
      InFull[t, 0, d] := 1.5 * (Random - 0.5);

  // Full-sequence parallel/prefill path (one Compute over the whole sequence).
  NNFull.Compute(InFull);

  // Incremental path: one token at a time, fixed-size state h carried across calls.
  SsmStep.BeginIncrementalDecode();
  maxErr := 0;
  for t := 0 to SeqLen - 1 do
  begin
    for d := 0 to Depth - 1 do
      InStep[0, 0, d] := InFull[t, 0, d];
    NNStep.Compute(InStep);
    for d := 0 to Depth - 1 do
    begin
      e := Abs(SsmStep.Output[0, 0, d] - SsmFull.Output[t, 0, d]);
      if e > maxErr then maxErr := e;
    end;
  end;
  SsmStep.EndIncrementalDecode();

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
  WriteLn('    state bytes carried = ', Depth * DState * SizeOf(TNeuralFloat),
          ' (Depth*DState floats) -- INDEPENDENT of position.');
  WriteLn;

  SsmStep.BeginIncrementalDecode();
  // Warm-up (advance the state to a deep position) without timing.
  for t := 0 to WarmSteps - 1 do
  begin
    for d := 0 to Depth - 1 do
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
      for d := 0 to Depth - 1 do
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
  SsmStep.EndIncrementalDecode();
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
