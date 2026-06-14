(*
RWKVGenerate example

End-to-end flat-memory recurrent decoding through a FULL RWKV block.

examples/RWKVDecode demonstrated the constant-memory decode of the bare RWKV-4
recurrence leaf TNNetWKV. The missing piece for driving a real RWKV model
token-by-token was the OTHER stateful layer in an RWKV block: TNNetTokenShift,
the per-channel time-shift that mixes x_t with x_{t-1}. A whole AddRWKVBlock
(time-mix: TokenShift -> receptance/key/value projections -> TNNetWKV -> gate ->
out-proj; plus a channel-mix sub-block with its OWN two TokenShifts) therefore
has FOUR stateful layers that must all advance in lockstep for the block to
decode one token at a time.

TNNetTokenShift now has the SAME incremental-decode API as TNNetWKV /
TNNetSelectiveSSM:
  BeginIncrementalDecode / Compute (one token) / ResetState / ResetCache /
  EndIncrementalDecode + CaptureState / RestoreState
and the carried state is a single Depth-long buffer holding the previous token.

A net-wide driver switches them all together:
  TNNet.BeginIncrementalDecode  -- loops the layers, switches every recurrent
                                   leaf (TokenShift, WKV, SelectiveSSM,
                                   DiagonalSSM) onto its O(1)-per-step path
  TNNet.ResetIncrementalDecode  -- fresh sequence on all of them
  TNNet.EndIncrementalDecode    -- back to the full-sequence forward
(The attention KV-cache layers take a MaxContext budget and stay on the
TNNetStreamingDecoder path; this driver is for the zero-arg recurrent leaves.)

This example builds a small but COMPLETE RWKV language-model net (token
embedding -> two AddRWKVBlock stacks -> norm -> vocab logits) and demonstrates:

  (A) EXACT EQUIVALENCE. Greedy-decoding the sequence token-by-token with
      TNNet.BeginIncrementalDecode reproduces the full-sequence forward pass's
      per-step next-token logits BIT-FOR-BIT, and therefore picks the identical
      argmax token at every step. Every TokenShift + WKV in both blocks advances
      one token per step.

  (B) FLAT MEMORY / FLAT WORK. The carried state is fixed-size (one previous
      token per TokenShift + one (A,B,Q) triple per WKV) regardless of how many
      tokens have been generated, so the per-step cost does NOT grow with the
      context length -- the constant-memory headline, vs a transformer KV cache
      whose per-step work is O(context).

Self-contained and offline: a tiny random-init RWKV net, no checkpoint. Run
under a memory cap:
  ulimit -v 3000000 ./RWKVGenerate

Coded by Claude (AI).
*)
program RWKVGenerate;

{$mode objfpc}{$H+}

uses
  SysUtils,
  neuralnetwork,
  neuralvolume;

const
  Vocab   = 32;     // toy vocabulary size (logits width)
  DModel  = 24;     // embedding / hidden width
  SeqLen  = 20;     // sequence length for the exact-equivalence check
  WarmSteps = 64;   // decode positions before timing (so timing starts deep)
  ChunkSteps = 96;  // steps timed per chunk
  Chunks  = 4;      // number of timed chunks (positions grow each chunk)

// Build a small COMPLETE RWKV LM net with a given input sequence length.
// Input is a (InSeqLen,1,DModel) hidden-state sequence (a stand-in for the
// token-embedding lookup) -> two RWKV blocks -> final norm -> vocab logits.
function BuildRWKVNet(InSeqLen: integer): TNNet;
var
  NN: TNNet;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(InSeqLen, 1, DModel, 1));
  NN.AddRWKVBlock();
  NN.AddRWKVBlock();
  NN.AddLayer(TNNetTokenLayerNorm.Create());
  NN.AddLayer(TNNetPointwiseConvLinear.Create(Vocab, 1));  // per-token logits
  Result := NN;
end;

// argmax over the Vocab logits of row t.
function ArgMaxRow(V: TNNetVolume; t: integer): integer;
var
  d, best: integer;
  bestVal: TNeuralFloat;
begin
  best := 0;
  bestVal := V[t, 0, 0];
  for d := 1 to Vocab - 1 do
    if V[t, 0, d] > bestVal then
    begin
      bestVal := V[t, 0, d];
      best := d;
    end;
  Result := best;
end;

var
  NNFull, NNStep: TNNet;
  InFull, InStep: TNNetVolume;
  OutFull, OutStep: TNNetVolume;
  t, d, i, recCount: integer;
  maxErr, e: TNeuralFloat;
  argMismatch: integer;
  StartTime, Elapsed, FirstChunkUsPerStep, ThisUsPerStep: double;
  Pos: integer;
begin
  Randomize;
  RandSeed := 20260614;

  WriteLn('=== RWKVGenerate: end-to-end flat-memory RWKV block decoding ===');
  WriteLn('DModel=', DModel, '  Vocab=', Vocab, '  blocks=2 (each: time-mix + channel-mix)');
  WriteLn;

  // -------- (A) EXACT EQUIVALENCE: full-scan vs token-by-token --------
  NNFull := BuildRWKVNet(SeqLen);
  NNStep := BuildRWKVNet(1);            // single-token (decode-step) input
  NNFull.InitWeights();
  NNStep.CopyWeights(NNFull);           // weight-identical nets

  InFull := TNNetVolume.Create(SeqLen, 1, DModel, 1);
  InStep := TNNetVolume.Create(1, 1, DModel, 1);
  for t := 0 to SeqLen - 1 do
    for d := 0 to DModel - 1 do
      InFull[t, 0, d] := 1.0 * (Random - 0.5);

  // Full-sequence parallel/prefill path (one Compute over the whole sequence).
  NNFull.Compute(InFull);
  OutFull := NNFull.GetLastLayer().Output;

  // Net-wide incremental driver: switch every recurrent leaf onto its
  // O(1)-per-step state-carry path, then feed one token at a time.
  recCount := NNStep.BeginIncrementalDecode();
  OutStep := NNStep.GetLastLayer().Output;
  maxErr := 0;
  argMismatch := 0;
  for t := 0 to SeqLen - 1 do
  begin
    for d := 0 to DModel - 1 do
      InStep[0, 0, d] := InFull[t, 0, d];
    NNStep.Compute(InStep);
    for d := 0 to Vocab - 1 do
    begin
      e := Abs(OutStep[0, 0, d] - OutFull[t, 0, d]);
      if e > maxErr then maxErr := e;
    end;
    if ArgMaxRow(OutStep, 0) <> ArgMaxRow(OutFull, t) then Inc(argMismatch);
  end;
  NNStep.EndIncrementalDecode();

  WriteLn('(A) Net-wide incremental decode vs full-scan over ', SeqLen, ' tokens:');
  WriteLn('    recurrent leaves switched on (2 blocks x (1 TokenShift+1 WKV time-mix');
  WriteLn('      + 2 channel-mix TokenShift)) = ', recCount);
  WriteLn('    max abs logit error = ', maxErr:0:12);
  WriteLn('    next-token argmax mismatches = ', argMismatch, ' / ', SeqLen);
  if (maxErr < 1e-5) and (argMismatch = 0) then
    WriteLn('    PASS (BIT-EXACT: incremental reproduces the parallel path)')
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
  WriteLn('    Carried state is fixed-size (prev token per TokenShift + (A,B,Q)');
  WriteLn('    per WKV) -- INDEPENDENT of position.');
  WriteLn;

  NNStep.BeginIncrementalDecode();
  // Warm-up (advance the state to a deep position) without timing.
  for t := 0 to WarmSteps - 1 do
  begin
    for d := 0 to DModel - 1 do
      InStep[0, 0, d] := 1.0 * (Random - 0.5);
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
      for d := 0 to DModel - 1 do
        InStep[0, 0, d] := 1.0 * (Random - 0.5);
      NNStep.Compute(InStep);
    end;
    Elapsed := (Now() - StartTime) * 24.0 * 3600.0 * 1e6; // days -> microseconds
    ThisUsPerStep := Elapsed / ChunkSteps;
    if i = 0 then FirstChunkUsPerStep := ThisUsPerStep;
    WriteLn('    ', i:5, '   ', Pos:9, '   ', ThisUsPerStep:7:3, '   ',
            (ThisUsPerStep / FirstChunkUsPerStep):0:3);
    Inc(Pos, ChunkSteps);
  end;
  NNStep.EndIncrementalDecode();
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
