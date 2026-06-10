// IncrementalDecode example
//
// KV-cache incremental decode on TNNetScaledDotProductAttention.
//
// Autoregressive generation with a vanilla attention stack re-encodes the
// ENTIRE prefix to sample every next token: the step at prefix length t pays
// for a full t x t causal attention pass (O(t^2) work), so generating N
// tokens costs O(N^3) attention work in total. The KV cache fixes this: the
// layer keeps every past token's K and V slices in a persistent preallocated
// buffer (BeginIncrementalDecode(MaxContext)), and a decode step feeds ONLY
// the new token — its query attends over the cached keys/values [0..t], so
// the step costs one row of attention (O(t)) instead of a full re-encode
// (O(t^2)). Attending over exactly the cached prefix IS the causal mask.
//
// This demo does two things:
//   1) Faithfulness: feeds the SAME random Q|K|V sequence through (a) one
//      full causal forward and (b) token-at-a-time through the cached path,
//      and checks the outputs match at EVERY position to < 1e-5.
//   2) Timing: measures per-token wall-clock at growing prefix lengths for
//      both arms and prints the table — the full re-encode column grows
//      ~quadratically with the prefix while the cached column stays nearly
//      flat, so the speedup ratio keeps widening with t.
//
// Positional contract reminder: the SDPA layer applies no positional encoding
// itself. If your stack uses TNNetAddPositionalEmbedding / RoPE before the
// QKV projection, encode each streamed token with its ABSOLUTE position
// (= SDPA.CacheLength before the step), not position 0.
//
// Pure CPU, no training, finishes in a few seconds.
//
// Coded by Claude (AI).
program IncrementalDecode;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, BaseUnix, Unix,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cDk        = 64;    // head dimension
  cMaxLen    = 1024;  // longest prefix probed (= cache preallocation)
  cWindow    = 64;    // timing window (steps averaged per measurement)
  cCheckLen  = 48;    // sequence length for the faithfulness check
  cCheckpoints: array[0..4] of integer = (64, 128, 256, 512, 1024);

var
  MasterSeq: TNNetVolume; // one shared random Q|K|V stream [cMaxLen,1,3*cDk]

// Microsecond-resolution wall clock in milliseconds since the first call.
// SysUtils.Now ticks at ~1 ms on Linux, far too coarse for a cached step (a
// few microseconds). Rebasing to the first call keeps the value small: FPC
// types the 1000.0 literal as SINGLE, and at an absolute Unix-epoch scale
// (~1.8e12 ms) single-precision quantization (~131 s!) would freeze the clock.
{$IFDEF UNIX}
var
  GBaseSec: int64 = -1;

function NowMs(): double;
var
  tv: TTimeVal;
begin
  fpGetTimeOfDay(@tv, nil);
  if GBaseSec < 0 then GBaseSec := tv.tv_sec;
  Result := (tv.tv_sec - GBaseSec) * 1000.0 + tv.tv_usec / 1000.0;
end;
{$ELSE}
var
  GBaseMs: double = -1;

function NowMs(): double;
begin
  if GBaseMs < 0 then GBaseMs := Now() * 24 * 3600 * 1000;
  Result := Now() * 24 * 3600 * 1000 - GBaseMs;
end;
{$ENDIF}

// ---------------------------------------------------------------------------
// 1) Faithfulness: full forward vs token-at-a-time cached decode.
// ---------------------------------------------------------------------------
function RunFaithfulnessCheck(): boolean;
var
  NNFull, NNStep: TNNet;
  SDPAStep: TNNetScaledDotProductAttention;
  FullIn, StepIn, FullOut, StepOut: TNNetVolume;
  T, D: integer;
  Diff, MaxDiff: TNeuralFloat;
begin
  NNFull := TNNet.Create();
  NNFull.AddLayer(TNNetInput.Create(cCheckLen, 1, 3 * cDk));
  NNFull.AddLayer(TNNetScaledDotProductAttention.Create(cDk, {CausalMask=}true));
  NNStep := TNNet.Create();
  NNStep.AddLayer(TNNetInput.Create(1, 1, 3 * cDk));
  SDPAStep := TNNetScaledDotProductAttention.Create(cDk, {CausalMask=}true);
  NNStep.AddLayer(SDPAStep);
  FullIn := TNNetVolume.Create(cCheckLen, 1, 3 * cDk);
  StepIn := TNNetVolume.Create(1, 1, 3 * cDk);
  FullOut := TNNetVolume.Create();
  StepOut := TNNetVolume.Create();
  try
    for T := 0 to cCheckLen - 1 do
      for D := 0 to 3 * cDk - 1 do
        FullIn[T, 0, D] := MasterSeq[T, 0, D];
    NNFull.Compute(FullIn);
    NNFull.GetOutput(FullOut);

    SDPAStep.BeginIncrementalDecode(cCheckLen);
    MaxDiff := 0;
    for T := 0 to cCheckLen - 1 do
    begin
      for D := 0 to 3 * cDk - 1 do
        StepIn[0, 0, D] := FullIn[T, 0, D];
      NNStep.Compute(StepIn);
      NNStep.GetOutput(StepOut);
      for D := 0 to cDk - 1 do
      begin
        Diff := Abs(FullOut[T, 0, D] - StepOut[0, 0, D]);
        if Diff > MaxDiff then MaxDiff := Diff;
      end;
    end;
    WriteLn('Faithfulness: max |full - cached| over all ', cCheckLen,
      ' positions x ', cDk, ' dims = ', MaxDiff: 12: 9);
    Result := MaxDiff < 1e-5;
    if Result then
      WriteLn('Faithfulness: PASS (< 1e-5)')
    else
      WriteLn('Faithfulness: FAIL (>= 1e-5)');
  finally
    StepOut.Free;
    FullOut.Free;
    StepIn.Free;
    FullIn.Free;
    NNStep.Free;
    NNFull.Free;
  end;
end;

// ---------------------------------------------------------------------------
// 2) Timing: per-token wall-clock vs prefix length, both arms.
// ---------------------------------------------------------------------------

// Full re-encode arm: per-token cost at prefix length Len = one full causal
// forward over Len tokens (that is exactly what a cache-less sampler pays for
// every generated token). Averaged over cWindow repetitions.
function TimeFullForwardPerStep(Len: integer): double;
var
  NN: TNNet;
  InV: TNNetVolume;
  T, D, Rep, Reps: integer;
  T0: double;
begin
  // Each full forward is large; a handful of reps is plenty for stable ms
  // numbers and keeps the t=1024 row fast.
  Reps := Max(4, 1024 div Len);
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(Len, 1, 3 * cDk));
  NN.AddLayer(TNNetScaledDotProductAttention.Create(cDk, true));
  InV := TNNetVolume.Create(Len, 1, 3 * cDk);
  try
    for T := 0 to Len - 1 do
      for D := 0 to 3 * cDk - 1 do
        InV[T, 0, D] := MasterSeq[T, 0, D];
    NN.Compute(InV); // warm-up
    T0 := NowMs();
    for Rep := 1 to Reps do
      NN.Compute(InV);
    Result := (NowMs() - T0) / Reps; // ms per step
  finally
    InV.Free;
    NN.Free;
  end;
end;

// Cached arm: stream the master sequence token-at-a-time through ONE step net
// and record, for each checkpoint t, the average wall-clock of the cWindow
// single-token steps ending at cache length t.
procedure TimeCachedStream(var PerStepMs: array of double);
var
  NN: TNNet;
  SDPA: TNNetScaledDotProductAttention;
  StepIn: TNNetVolume;
  T, D, C: integer;
  T0: double;
  WindowStart: array[0..High(cCheckpoints)] of integer;
begin
  for C := 0 to High(cCheckpoints) do
    WindowStart[C] := cCheckpoints[C] - cWindow; // steps [start..chk-1]
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(1, 1, 3 * cDk));
  SDPA := TNNetScaledDotProductAttention.Create(cDk, true);
  NN.AddLayer(SDPA);
  StepIn := TNNetVolume.Create(1, 1, 3 * cDk);
  try
    SDPA.BeginIncrementalDecode(cMaxLen);
    T0 := 0;
    for T := 0 to cMaxLen - 1 do
    begin
      for D := 0 to 3 * cDk - 1 do
        StepIn[0, 0, D] := MasterSeq[T, 0, D];
      for C := 0 to High(cCheckpoints) do
        if T = WindowStart[C] then T0 := NowMs();
      NN.Compute(StepIn);
      for C := 0 to High(cCheckpoints) do
        if T = cCheckpoints[C] - 1 then
          PerStepMs[C] := (NowMs() - T0) / cWindow;
    end;
  finally
    StepIn.Free;
    NN.Free;
  end;
end;

var
  T, D, C: integer;
  CachedMs: array[0..High(cCheckpoints)] of double;
  FullMs: double;
begin
  RandSeed := 20260610;
  WriteLn('KV-cache incremental decode: TNNetScaledDotProductAttention');
  WriteLn('d_k=', cDk, '  max context=', cMaxLen, '  timing window=',
    cWindow, ' steps');
  WriteLn;

  MasterSeq := TNNetVolume.Create(cMaxLen, 1, 3 * cDk);
  for T := 0 to cMaxLen - 1 do
    for D := 0 to 3 * cDk - 1 do
      MasterSeq[T, 0, D] := (Random(2000) - 1000) / 1000;

  if not RunFaithfulnessCheck() then
  begin
    MasterSeq.Free;
    WriteLn('ERROR: cached path does not match the full forward.');
    Halt(1);
  end;
  WriteLn;

  TimeCachedStream(CachedMs);
  WriteLn('Per-token wall-clock vs prefix length t:');
  WriteLn('  prefix t | full re-encode (ms/token) | cached step (ms/token) | speedup');
  for C := 0 to High(cCheckpoints) do
  begin
    FullMs := TimeFullForwardPerStep(cCheckpoints[C]);
    WriteLn(Format('  %8d | %26.4f | %22.4f | %6.1fx',
      [cCheckpoints[C], FullMs, CachedMs[C], FullMs / Max(CachedMs[C], 1e-9)]));
  end;
  WriteLn;
  WriteLn('The full re-encode column grows ~quadratically with the prefix');
  WriteLn('(every step re-runs t x t attention); the cached column stays nearly');
  WriteLn('flat (one query row over the cached K/V), so the speedup widens with t.');
  MasterSeq.Free;
end.
