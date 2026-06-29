program OpenCLForwardBenchmark;
(*
OpenCLForwardBenchmark: forward-pass CPU-vs-OpenCL timing sweep across every
single-input layer type that has an OpenCL forward path.

For each layer it builds a two-layer net (TNNetInput -> TheLayer), times the
forward pass on the CPU, then calls EnableOpenCL and times the same forward on
the device, and prints one table row: representative output shape, mean
microseconds per forward (CPU and GPU), the CPU/GPU speedup, and whether the
GPU path actually fired. The last column matters: several layers only dispatch
to the device above a size threshold (e.g. SDPA needs SeqLen >=
csSDPAOpenCLMinSeqLen) and otherwise silently run on the CPU - the per-layer
ForwardGPUCnt dispatch counter is read back to tell a real offload from a
fallback, so a "speedup" computed against a CPU fallback is flagged rather than
reported as a win.

Timing is wall-clock and auto-scaled: each measurement repeats the forward pass
until at least cMinSeconds of work has run (capped at cMaxIters), so the coarse
resolution of Now() does not dominate the small layers. A warmup pass is run
and discarded before each measurement; the first device call additionally pays
kernel compilation and buffer upload, which the warmup absorbs.

This is a microbenchmark of isolated single layers, not a model: the numbers
show where the device beats the host for each operator in isolation. Multi-input
layers (cross-attention, grid-sample, warp, AdaIN, correlation) need a second
source branch and are out of scope for this v1; they are listed at the end.

Wall-clock numbers are inherently machine/run dependent. PASS/FAIL is not a
concept here - the program always exits 0 (it is a benchmark, not a test). With
no OpenCL platform available it prints SKIP and exits 0 (harmless in CPU-only
CI).

Build (from this directory):
  fpc -Mobjfpc -Sh -O3 -dAVX2 -dRelease -dOpenCL -Fu../../neural OpenCLForwardBenchmark.lpr

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX}cthreads,{$ENDIF}
  SysUtils, neuralvolume, neuralnetwork
  {$IFDEF OpenCL}, neuralopencl, cl{$ENDIF};

{$IFDEF OpenCL}
const
  cMinSeconds = 0.40;  // grow the repeat count until a measurement runs this long
  cMaxIters   = 8192;  // ... but never exceed this many forwards per measurement
  cWarmups    = 3;     // discarded forwards before each timed measurement

var
  EasyCL: TEasyOpenCL;
  PlatformId: cl_platform_id;
  DeviceId: cl_device_id;
  AnyFallback: boolean;
  AnyError: boolean;

// Build a fresh (TNNetInput -> ?) net and a randomized matching input volume.
// The caller appends the layer under test, then hands both to Bench (which frees
// them). Input values are uniform in [-1, 1]; see FillTokens for the embedding.
function MakeNet(X, Y, D: integer; out Inp: TNNetVolume): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(X, Y, D));
  Inp := TNNetVolume.Create(X, Y, D);
  Inp.Randomize();
end;

// Overwrite an input volume with valid integer token ids for TNNetEmbedding.
procedure FillTokens(Inp: TNNetVolume; VocabSize: integer);
var
  Cnt: integer;
begin
  for Cnt := 0 to Inp.Size - 1 do
    Inp.FData[Cnt] := Random(VocabSize);
end;

// Mean microseconds per forward. Repeats Compute in doubling batches until the
// batch spans at least cMinSeconds (so Now()'s coarse resolution is negligible)
// or the batch hits cMaxIters. Returns us/forward; reports the batch size used.
function TimeForward(NN: TNNet; Inp: TNNetVolume; out Iters: integer): double;
var
  Cnt, N: integer;
  T0, ElapsedSec: double;
begin
  N := 8;
  while True do
  begin
    T0 := Now();
    for Cnt := 1 to N do NN.Compute(Inp);
    ElapsedSec := (Now() - T0) * 86400.0; // TDateTime span (days) -> seconds
    if (ElapsedSec >= cMinSeconds) or (N >= cMaxIters) then break;
    N := N * 2;
  end;
  Iters := N;
  if N > 0 then Result := ElapsedSec / N * 1e6 else Result := 0;
end;

// CPU forward vs OpenCL forward of the same one-layer net; prints one row.
// Frees NN and Inp.
procedure Bench(const Name: string; NN: TNNet; Inp: TNNetVolume);
var
  Cnt, ItersC, ItersG: integer;
  CpuUs, GpuUs, Speed: double;
  GpuCnt: integer;
  Last: TNNetLayer;
  Shape, Fired: string;
begin
  try
    Last := NN.GetLastLayer();
    Shape := Format('%dx%dx%d',
      [Last.Output.SizeX, Last.Output.SizeY, Last.Output.Depth]);
    // A single layer whose OpenCL path faults (bad buffer size, kernel arg
    // failure, etc.) must not abort the whole sweep: catch it, print an ERROR
    // row, and move on to the next layer.
    try
      // CPU path.
      for Cnt := 1 to cWarmups do NN.Compute(Inp);
      CpuUs := TimeForward(NN, Inp, ItersC);

      // OpenCL path (warmup absorbs kernel compile + first upload).
      NN.EnableOpenCL(PlatformId, DeviceId);
      for Cnt := 1 to cWarmups do NN.Compute(Inp);
      NN.ClearTime(); // zero the dispatch counters so GpuCnt covers only the measure
      GpuUs := TimeForward(NN, Inp, ItersG);
      GpuCnt := NN.GetLastLayer().ForwardGPUCnt;

      if GpuCnt > 0 then Fired := 'yes'
      else
      begin
        Fired := 'NO-cpu';
        AnyFallback := True;
      end;

      if (GpuCnt > 0) and (GpuUs > 0) then
        Speed := CpuUs / GpuUs
      else
        Speed := 0;

      if Speed > 0 then
        WriteLn(Format('%-32s %-14s %12.1f %12.1f %9.2fx  %s',
          [Name, Shape, CpuUs, GpuUs, Speed, Fired]))
      else
        WriteLn(Format('%-32s %-14s %12.1f %12.1f %10s  %s',
          [Name, Shape, CpuUs, GpuUs, '-', Fired]));
    except
      on E: Exception do
      begin
        WriteLn(Format('%-32s %-14s  *** ERROR: %s', [Name, Shape, E.Message]));
        AnyError := True;
      end;
    end;
  finally
    NN.Free;
    Inp.Free;
  end;
end;

var
  NN: TNNet;
  Inp: TNNetVolume;

begin
  EasyCL := TEasyOpenCL.Create();
  try
    if EasyCL.GetPlatformCount() = 0 then
    begin
      WriteLn('SKIP: no OpenCL platform found.');
      Halt(0);
    end;
    EasyCL.SetCurrentPlatform(EasyCL.PlatformIds[0]);
    if EasyCL.GetDeviceCount() = 0 then
    begin
      WriteLn('SKIP: no OpenCL device on platform ', EasyCL.PlatformNames[0]);
      Halt(0);
    end;
    EasyCL.SetCurrentDevice(EasyCL.Devices[0]);
    PlatformId := EasyCL.PlatformIds[0];
    DeviceId := EasyCL.Devices[0];

    RandSeed := 424242; // deterministic inputs across runs
    AnyFallback := False;
    AnyError := False;

    WriteLn('OpenCL: ', EasyCL.PlatformNames[0], ' / ', EasyCL.DeviceNames[0]);
    WriteLn('Profiles: SEQ=(256,1,D)  VIS=(32,32,C)  d_k=64  d_model=512');
    WriteLn(Format('Auto-scaled timing: >= %.2fs/measurement, cap %d forwards, %d warmups',
      [cMinSeconds, cMaxIters, cWarmups]));
    WriteLn;
    WriteLn(Format('%-32s %-14s %12s %12s %10s  %s',
      ['layer', 'out shape', 'cpu us/fwd', 'gpu us/fwd', 'speedup', 'gpu?']));
    WriteLn(StringOfChar('-', 96));

    // --- Convolution family (VIS 32x32x64) ---
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetConvolution.Create(128, 3, 1, 1));
    Bench('TNNetConvolution', NN, Inp);
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetConvolutionLinear.Create(128, 3, 1, 1));
    Bench('TNNetConvolutionLinear', NN, Inp);
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetDeconvolution.Create(64, 3, 2, 1, 1));
    Bench('TNNetDeconvolution', NN, Inp);
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetDepthwiseConv.Create(2, 3, 1, 1));
    Bench('TNNetDepthwiseConv', NN, Inp);
    NN := MakeNet(256, 1, 512, Inp); NN.AddLayer(TNNetDepthwiseConv1D.Create(3, True));
    Bench('TNNetDepthwiseConv1D', NN, Inp);
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetGroupConvP4.Create(32, 3, 1, 1));
    Bench('TNNetGroupConvP4', NN, Inp);
    // TNNetKANConv is intentionally excluded: its OpenCL forward path requests a
    // pathological buffer (~81 GB at this shape) and faults inside the device
    // driver, which is an unrecoverable native segfault (a try/except cannot
    // catch it, so it would abort the whole sweep). Flagged as a library issue
    // worth a separate look; see the "Known device-path faults" note at the end.
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetDeformableConv.Create(64, 3, 1, 1));
    Bench('TNNetDeformableConv', NN, Inp);

    // --- Dense / embedding ---
    NN := MakeNet(1, 1, 2048, Inp); NN.AddLayer(TNNetFullConnect.Create(2048));
    Bench('TNNetFullConnect', NN, Inp);
    NN := MakeNet(256, 1, 1, Inp); FillTokens(Inp, 32000);
    NN.AddLayer(TNNetEmbedding.Create(32000, 512));
    Bench('TNNetEmbedding', NN, Inp);

    // --- Attention (SEQ 256x1x192, QKV-packed, d_k=64) ---
    NN := MakeNet(256, 1, 192, Inp); NN.AddLayer(TNNetScaledDotProductAttention.Create(64, True, 0, 0));
    Bench('TNNetScaledDotProductAttention', NN, Inp);
    NN := MakeNet(256, 1, 192, Inp); NN.AddLayer(TNNetLinearAttention.Create(64));
    Bench('TNNetLinearAttention', NN, Inp);
    NN := MakeNet(256, 1, 192, Inp); NN.AddLayer(TNNetCosineSimilarityAttention.Create(64, True));
    Bench('TNNetCosineSimilarityAttention', NN, Inp);
    NN := MakeNet(256, 1, 192, Inp); NN.AddLayer(TNNetDisentangledAttention.Create(64, True));
    Bench('TNNetDisentangledAttention', NN, Inp);
    NN := MakeNet(256, 1, 192, Inp); NN.AddLayer(TNNetConformerRelPosAttention.Create(64, True));
    Bench('TNNetConformerRelPosAttention', NN, Inp);
    NN := MakeNet(256, 1, 192, Inp); NN.AddLayer(TNNetALiBiAttention.Create(64, True));
    Bench('TNNetALiBiAttention', NN, Inp);

    // --- Rotary position embeddings ---
    NN := MakeNet(256, 1, 128, Inp); NN.AddLayer(TNNetRotaryEmbedding.Create());
    Bench('TNNetRotaryEmbedding', NN, Inp);
    // TNNetMRotaryEmbedding is excluded: it requires SetPositions (a per-token
    // T/H/W multimodal position grid) before any forward pass, which the generic
    // single-layer harness does not supply; without it the forward faults.

    // --- Normalization ---
    NN := MakeNet(256, 1, 512, Inp); NN.AddLayer(TNNetRMSNorm.Create());
    Bench('TNNetRMSNorm', NN, Inp);
    NN := MakeNet(256, 1, 512, Inp); NN.AddLayer(TNNetTokenRMSNorm.Create());
    Bench('TNNetTokenRMSNorm', NN, Inp);
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetGroupNorm.Create(8));
    Bench('TNNetGroupNorm', NN, Inp);
    NN := MakeNet(256, 1, 512, Inp); NN.AddLayer(TNNetLayerNorm.Create());
    Bench('TNNetLayerNorm', NN, Inp);
    NN := MakeNet(256, 1, 512, Inp); NN.AddLayer(TNNetTokenLayerNorm.Create());
    Bench('TNNetTokenLayerNorm', NN, Inp);
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetPixelNorm.Create());
    Bench('TNNetPixelNorm', NN, Inp);
    NN := MakeNet(256, 1, 512, Inp); NN.AddLayer(TNNetL2Normalize.Create());
    Bench('TNNetL2Normalize', NN, Inp);

    // --- Gated activations (even depth -> output halves) ---
    NN := MakeNet(256, 1, 1024, Inp); NN.AddLayer(TNNetSwiGLU.Create());
    Bench('TNNetSwiGLU', NN, Inp);
    NN := MakeNet(256, 1, 1024, Inp); NN.AddLayer(TNNetGLU.Create());
    Bench('TNNetGLU', NN, Inp);
    NN := MakeNet(256, 1, 1024, Inp); NN.AddLayer(TNNetGEGLU.Create());
    Bench('TNNetGEGLU', NN, Inp);
    NN := MakeNet(256, 1, 1024, Inp); NN.AddLayer(TNNetGEGLUErf.Create());
    Bench('TNNetGEGLUErf', NN, Inp);
    NN := MakeNet(256, 1, 512, Inp); NN.AddLayer(TNNetPointwiseSoftMax.Create());
    Bench('TNNetPointwiseSoftMax', NN, Inp);

    // --- Pooling / spatial resize ---
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetMaxPool.Create(2, 2));
    Bench('TNNetMaxPool', NN, Inp);
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetBilinearResize.Create(64, 64, 0));
    Bench('TNNetBilinearResize', NN, Inp);
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetBicubicUpsample.Create(2, 0));
    Bench('TNNetBicubicUpsample', NN, Inp);
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetBilinearUpsample.Create(2));
    Bench('TNNetBilinearUpsample', NN, Inp);
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetPixelShuffle.Create(2));
    Bench('TNNetPixelShuffle', NN, Inp);
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetResize2D.Create(64, 64, 1, 0));
    Bench('TNNetResize2D', NN, Inp);
    NN := MakeNet(32, 32, 64, Inp); NN.AddLayer(TNNetGramMatrix.Create());
    Bench('TNNetGramMatrix', NN, Inp);

    // --- Recurrent cells (shape-preserving, read depth from prev layer) ---
    NN := MakeNet(256, 1, 256, Inp); NN.AddLayer(TNNetLSTMCell.Create());
    Bench('TNNetLSTMCell', NN, Inp);
    NN := MakeNet(256, 1, 256, Inp); NN.AddLayer(TNNetGRUCell.Create());
    Bench('TNNetGRUCell', NN, Inp);

    WriteLn(StringOfChar('-', 96));
    if AnyFallback then
      WriteLn('Note: rows marked NO-cpu never dispatched to the device (below a '
        + 'size threshold or no GPU path for that input) - their speedup is omitted.');
    if AnyError then
      WriteLn('Note: rows marked *** ERROR faulted on the device path for the '
        + 'chosen shape and were skipped (the sweep continued).');
    WriteLn;
    WriteLn('Out of scope (multi-input, need a second source branch): '
      + 'TNNetCrossAttention, TNNetGridSample, TNNetAffineGridSample, '
      + 'TNNetBackwardWarp, TNNetFlowWarp, TNNetAdaIN, TNNetCorrelationVolume, '
      + 'TNNetCorrelationLookup.');
    WriteLn('Known device-path faults (excluded): TNNetKANConv - its OpenCL '
      + 'forward requests a ~81 GB buffer at the benchmark shape and segfaults '
      + 'inside the driver (uncatchable). Worth a separate investigation. '
      + 'TNNetMRotaryEmbedding needs SetPositions before a forward and is also '
      + 'excluded from this generic harness.');
  finally
    EasyCL.Free;
  end;
end.
{$ELSE}
begin
  WriteLn('SKIP: built without -dOpenCL (rebuild with -dOpenCL to run).');
end.
{$ENDIF}
