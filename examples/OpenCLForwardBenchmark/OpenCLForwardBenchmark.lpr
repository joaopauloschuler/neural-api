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

An optional integer input scale factor (argv[1], default 1) multiplies, FOR EACH
layer, only the ONE dimension that drives that layer's CPU-vs-GPU crossover - the
sequence length for the per-token SEQ ops (norms, gated activations, softmax,
RoPE, embedding) and attention, the spatial width for the VIS spatial ops
(pooling, resize/upsample, shuffle, pixel/group norm, Gram), the output-feature
count for the convolutions, and the cell width for the recurrent cells. Holding
every other dimension at its base keeps the sweep near-linear (instead of the
k^3..k^4 blow-up of scaling all dimensions at once), so the same binary walks
each operator across its threshold without a rebuild:
  OpenCLForwardBenchmark        # base sizes (default, == scale 1)
  OpenCLForwardBenchmark 8      # each layer's primary dimension x8

Every row is measured with NN.ForceOpenCL(True) so the GPU path always fires and
both timings are charted; the 'verdict' column separately reports each layer's
own FShouldOpenCL size decision (what production dispatch would choose at this
size), so the gap between where 'verdict' flips to yes and where 'speedup'
crosses 1x shows whether the per-layer threshold is set well.

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

  cVocab   = 32000;  // embedding vocabulary (table size, not a per-forward size)

// Input profiles. The values below are the moderate BASE sizes. The optional
// command-line scale factor (argv[1], default 1) is NOT applied here; instead
// each Bench call site multiplies it into the ONE dimension that drives that
// layer's crossover (see the program header), leaving every other dimension at
// base. An INTEGER factor is used on purpose: it preserves every divisibility
// invariant the layers need - the scaled dimension keeps its own property
// (even depths stay even, channels stay /4 and /8, 3*d_k stays a multiple of 3,
// pool/shuffle spatial stays even) because integer multiplication keeps
// divisibility, so no per-dimension rounding is needed and no factor can produce
// an illegal shape. The resolved factor is echoed in the banner. With the
// dispatch gate at 2^16 the base sizes already exercise the device; raise the
// factor to push a real GPU (the depthwise/KAN result-buffer OOMs are fixed, so
// large factors are safe - see the GEMV-diagonal note in tasklist.md).
var
  cVisX:    integer = 32;    // VIS spatial width
  cVisY:    integer = 32;    // VIS spatial height
  cVisC:    integer = 64;    // VIS input channels
  cVisFeat: integer = 128;   // VIS conv output features
  cSeqLen:  integer = 256;   // SEQ sequence length
  cDModel:  integer = 512;   // SEQ model width (norms, softmax, pointwise gates)
  cDk:      integer = 64;    // attention head dim; QKV-packed input depth = 3*cDk
  cGateIn:  integer = 1024;  // gated-activation input depth (even; output halves)
  cFCIn:    integer = 2048;  // FullConnect flattened input/output width
  cRoPEDim: integer = 128;   // rotary embedding depth (even)
  cRnnDim:  integer = 256;   // LSTM/GRU cell width
  cScale:   integer = 1;     // argv[1]: multiplies the dimensions above

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
  Shape, Fired, Verdict: string;
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

      // OpenCL path (warmup absorbs kernel compile + first upload). ForceOpenCL
      // overrides each layer's per-layer size verdict so every OpenCL-capable
      // layer dispatches to the device regardless of tensor size - that is what
      // charts the per-layer crossover (the <1x rows are the point).
      NN.EnableOpenCL(PlatformId, DeviceId);
      // The natural per-layer size verdict (frozen in SetPrevLayer), captured
      // BEFORE the force override so the column reports what production dispatch
      // would actually choose at this size.
      if NN.GetLastLayer().ShouldOpenCL then Verdict := 'yes' else Verdict := 'no';
      NN.ForceOpenCL(True);
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
        WriteLn(Format('%-32s %-14s %12.1f %12.1f %9.2fx  %-6s %s',
          [Name, Shape, CpuUs, GpuUs, Speed, Fired, Verdict]))
      else
        WriteLn(Format('%-32s %-14s %12.1f %12.1f %10s  %-6s %s',
          [Name, Shape, CpuUs, GpuUs, '-', Fired, Verdict]));
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

    // Optional scale factor (argv[1], default 1). It is NOT applied to the global
    // profiles here; each Bench call site below multiplies it into the single
    // crossover-relevant dimension for that layer (see the program header). A
    // non-numeric/<1 arg falls back to 1, leaving the run identical to the
    // no-argument default.
    if ParamCount >= 1 then cScale := StrToIntDef(ParamStr(1), 1);
    if cScale < 1 then cScale := 1;

    WriteLn('OpenCL: ', EasyCL.PlatformNames[0], ' / ', EasyCL.DeviceNames[0]);
    WriteLn(Format('Scale factor: %d (argv[1]; default 1) - applied per layer to '
      + 'its primary dimension', [cScale]));
    WriteLn(Format('Base profiles: SEQ=(%d,1,%d)  VIS=(%d,%d,%d)  d_k=%d',
      [cSeqLen, cDModel, cVisX, cVisY, cVisC, cDk]));
    WriteLn('Dispatch: all layers FORCED via NN.ForceOpenCL(True); the verdict '
      + 'column shows each layer''s own FShouldOpenCL size decision');
    WriteLn(Format('Auto-scaled timing: >= %.2fs/measurement, cap %d forwards, %d warmups',
      [cMinSeconds, cMaxIters, cWarmups]));
    WriteLn;
    WriteLn(Format('%-32s %-14s %12s %12s %10s  %-6s %s',
      ['layer', 'out shape', 'cpu us/fwd', 'gpu us/fwd', 'speedup', 'gpu?', 'verdict']));
    WriteLn(StringOfChar('-', 104));

    // --- Convolution family (VIS) --- primary dim = output FEATURE count
    // (cVisFeat * cScale), the GEMM width that moves both the dispatch metric
    // (X*Y*Feat) and the arithmetic. Depthwise has no separate feature count, so
    // it scales input CHANNELS; DepthwiseConv1D is a SEQ op, so it scales SeqLen.
    NN := MakeNet(cVisX, cVisY, cVisC, Inp); NN.AddLayer(TNNetConvolution.Create(cVisFeat * cScale, 3, 1, 1));
    Bench('TNNetConvolution', NN, Inp);
    NN := MakeNet(cVisX, cVisY, cVisC, Inp); NN.AddLayer(TNNetConvolutionLinear.Create(cVisFeat * cScale, 3, 1, 1));
    Bench('TNNetConvolutionLinear', NN, Inp);
    NN := MakeNet(cVisX, cVisY, cVisC, Inp); NN.AddLayer(TNNetDeconvolution.Create(cVisC * cScale, 3, 2, 1, 1));
    Bench('TNNetDeconvolution', NN, Inp);
    NN := MakeNet(cVisX, cVisY, cVisC * cScale, Inp); NN.AddLayer(TNNetDepthwiseConv.Create(2, 3, 1, 1));
    Bench('TNNetDepthwiseConv', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, cDModel, Inp); NN.AddLayer(TNNetDepthwiseConv1D.Create(3, True));
    Bench('TNNetDepthwiseConv1D', NN, Inp);
    NN := MakeNet(cVisX, cVisY, cVisC, Inp); NN.AddLayer(TNNetGroupConvP4.Create((cVisFeat * cScale) div 4, 3, 1, 1));
    Bench('TNNetGroupConvP4', NN, Inp);
    NN := MakeNet(cVisX, cVisY, cVisC, Inp); NN.AddLayer(TNNetKANConv.Create(cVisFeat * cScale, 3, 1, 1));
    Bench('TNNetKANConv', NN, Inp);
    NN := MakeNet(cVisX, cVisY, cVisC, Inp); NN.AddLayer(TNNetDeformableConv.Create(cVisFeat * cScale, 3, 1, 1));
    Bench('TNNetDeformableConv', NN, Inp);

    // --- Dense / embedding --- FullConnect scales its OUTPUT width (the dispatch
    // metric; input width held at base keeps work linear); Embedding scales the
    // SEQ length (number of row lookups).
    NN := MakeNet(1, 1, cFCIn, Inp); NN.AddLayer(TNNetFullConnect.Create(cFCIn * cScale));
    Bench('TNNetFullConnect', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, 1, Inp); FillTokens(Inp, cVocab);
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModel));
    Bench('TNNetEmbedding', NN, Inp);

    // --- Attention (SEQ, QKV-packed, depth = 3*d_k) --- primary dim = SeqLen,
    // the O(SeqLen^2) term that decides the offload (d_k held at base).
    NN := MakeNet(cSeqLen * cScale, 1, 3 * cDk, Inp); NN.AddLayer(TNNetScaledDotProductAttention.Create(cDk, True, 0, 0));
    Bench('TNNetScaledDotProductAttention', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, 3 * cDk, Inp); NN.AddLayer(TNNetLinearAttention.Create(cDk));
    Bench('TNNetLinearAttention', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, 3 * cDk, Inp); NN.AddLayer(TNNetCosineSimilarityAttention.Create(cDk, True));
    Bench('TNNetCosineSimilarityAttention', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, 3 * cDk, Inp); NN.AddLayer(TNNetDisentangledAttention.Create(cDk, True));
    Bench('TNNetDisentangledAttention', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, 3 * cDk, Inp); NN.AddLayer(TNNetConformerRelPosAttention.Create(cDk, True));
    Bench('TNNetConformerRelPosAttention', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, 3 * cDk, Inp); NN.AddLayer(TNNetALiBiAttention.Create(cDk, True));
    Bench('TNNetALiBiAttention', NN, Inp);

    // --- Rotary position embeddings --- primary dim = SeqLen (RoPE depth base).
    NN := MakeNet(cSeqLen * cScale, 1, cRoPEDim, Inp); NN.AddLayer(TNNetRotaryEmbedding.Create());
    Bench('TNNetRotaryEmbedding', NN, Inp);
    // TNNetMRotaryEmbedding is excluded: it requires SetPositions (a per-token
    // T/H/W multimodal position grid) before any forward pass, which the generic
    // single-layer harness does not supply; without it the forward faults.

    // --- Normalization --- per-token SEQ norms scale SeqLen; the VIS per-pixel /
    // grouped norms scale spatial width (cVisX * cScale).
    NN := MakeNet(cSeqLen * cScale, 1, cDModel, Inp); NN.AddLayer(TNNetRMSNorm.Create());
    Bench('TNNetRMSNorm', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, cDModel, Inp); NN.AddLayer(TNNetTokenRMSNorm.Create());
    Bench('TNNetTokenRMSNorm', NN, Inp);
    NN := MakeNet(cVisX * cScale, cVisY, cVisC, Inp); NN.AddLayer(TNNetGroupNorm.Create(8));
    Bench('TNNetGroupNorm', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, cDModel, Inp); NN.AddLayer(TNNetLayerNorm.Create());
    Bench('TNNetLayerNorm', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, cDModel, Inp); NN.AddLayer(TNNetTokenLayerNorm.Create());
    Bench('TNNetTokenLayerNorm', NN, Inp);
    NN := MakeNet(cVisX * cScale, cVisY, cVisC, Inp); NN.AddLayer(TNNetPixelNorm.Create());
    Bench('TNNetPixelNorm', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, cDModel, Inp); NN.AddLayer(TNNetL2Normalize.Create());
    Bench('TNNetL2Normalize', NN, Inp);

    // --- Gated activations (even depth -> output halves) --- primary dim = SeqLen
    // (gate width held at base, stays even).
    NN := MakeNet(cSeqLen * cScale, 1, cGateIn, Inp); NN.AddLayer(TNNetSwiGLU.Create());
    Bench('TNNetSwiGLU', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, cGateIn, Inp); NN.AddLayer(TNNetGLU.Create());
    Bench('TNNetGLU', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, cGateIn, Inp); NN.AddLayer(TNNetGEGLU.Create());
    Bench('TNNetGEGLU', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, cGateIn, Inp); NN.AddLayer(TNNetGEGLUErf.Create());
    Bench('TNNetGEGLUErf', NN, Inp);
    NN := MakeNet(cSeqLen * cScale, 1, cDModel, Inp); NN.AddLayer(TNNetPointwiseSoftMax.Create());
    Bench('TNNetPointwiseSoftMax', NN, Inp);

    // --- Pooling / spatial resize (VIS) --- primary dim = spatial WIDTH
    // (cVisX * cScale); height held at base so work grows linearly. Resize/upsample
    // targets track the scaled input width.
    NN := MakeNet(cVisX * cScale, cVisY, cVisC, Inp); NN.AddLayer(TNNetMaxPool.Create(2, 2));
    Bench('TNNetMaxPool', NN, Inp);
    NN := MakeNet(cVisX * cScale, cVisY, cVisC, Inp); NN.AddLayer(TNNetBilinearResize.Create(cVisX * cScale * 2, cVisY * 2, 0));
    Bench('TNNetBilinearResize', NN, Inp);
    NN := MakeNet(cVisX * cScale, cVisY, cVisC, Inp); NN.AddLayer(TNNetBicubicUpsample.Create(2, 0));
    Bench('TNNetBicubicUpsample', NN, Inp);
    NN := MakeNet(cVisX * cScale, cVisY, cVisC, Inp); NN.AddLayer(TNNetBilinearUpsample.Create(2));
    Bench('TNNetBilinearUpsample', NN, Inp);
    NN := MakeNet(cVisX * cScale, cVisY, cVisC, Inp); NN.AddLayer(TNNetPixelShuffle.Create(2));
    Bench('TNNetPixelShuffle', NN, Inp);
    NN := MakeNet(cVisX * cScale, cVisY, cVisC, Inp); NN.AddLayer(TNNetResize2D.Create(cVisX * cScale * 2, cVisY * 2, 1, 0));
    Bench('TNNetResize2D', NN, Inp);
    NN := MakeNet(cVisX * cScale, cVisY, cVisC, Inp); NN.AddLayer(TNNetGramMatrix.Create());
    Bench('TNNetGramMatrix', NN, Inp);

    // --- Recurrent cells (shape-preserving, read depth from prev layer) ---
    // primary dim = cell WIDTH (cRnnDim * cScale), the per-step gate matmul size.
    NN := MakeNet(cSeqLen, 1, cRnnDim * cScale, Inp); NN.AddLayer(TNNetLSTMCell.Create());
    Bench('TNNetLSTMCell', NN, Inp);
    NN := MakeNet(cSeqLen, 1, cRnnDim * cScale, Inp); NN.AddLayer(TNNetGRUCell.Create());
    Bench('TNNetGRUCell', NN, Inp);

    WriteLn(StringOfChar('-', 104));
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
    WriteLn('Excluded: TNNetMRotaryEmbedding needs SetPositions (a multimodal '
      + 'T/H/W position grid) before a forward and is not supported by this '
      + 'generic single-input harness.');
  finally
    EasyCL.Free;
  end;
end.
{$ELSE}
begin
  WriteLn('SKIP: built without -dOpenCL (rebuild with -dOpenCL to run).');
end.
{$ENDIF}
