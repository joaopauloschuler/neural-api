program SDPAFlashBench;
(*
SDPAFlashBench: times the OpenCL non-causal cached attention of TNNetFusedSDPA
(TNNetFusedSDPACL.ComputeNonCausal) for each FlashVariant: 0 is
cai_sdpa_noncausal_tiled, 1..3 are cai_sdpa_flash_v1..v3. No model is loaded.

Before timing, every variant runs once on a small shape against the CPU twin
(TNNetFusedSDPA on the host) at the requested head dimension; a variant whose
max |diff| exceeds 1e-5 * max(1, max |y|) is reported as FAIL and not timed.

Timing is the wall time of the in-order OpenCL pipeline, enqueue included:
the packed [Q|K|V] input is uploaded once and bound as the step source, the
KV cache prefix is uploaded once, and the result stays in OpenCL memory; each
forward also re-appends the step's K/V rows into the same cache slots (one
cai_sdpa_append_kv launch, small next to the attention). The figure is the
median of 3 timed blocks, each at least --iters forwards and at least
200 ms. FLOPs = 4 * QHeads * Dk * (sum over token rows of the cache rows each
row attends).

Memory guard: a shape runs only if its host volumes (X, Y, K and V cache) fit
600 MB, counted twice on a CPU OpenCL device, whose buffers are host RAM too.
The default shape needs about 400 MB of host volumes.

--sweep-crossover times the flash path against the split-row decode path
(TNNetFusedSDPACL.Compute, cai_sdpa_decode_split + merge) for 1..512 token
rows over 8k and 32k caches. CAVEAT: the flash path runs mask mode none
(every row attends the whole window) while the decode path is causal (row t
attends up to its own slot), so their FLOPs differ slightly; compare ms per
forward and the per-key figure, not the masks.

Diagnostics: NEURAL_SDPA_DUMP_PTX=<file> writes the built program (PTX on
NVIDIA). NEURAL_OPENCL_BUILD_LOG=1 prints the first build log of the process;
with NEURAL_OPENCL_BUILD_OPTIONS=-cl-nv-verbose it holds NVIDIA's per-kernel
register and spill report (set CUDA_CACHE_DISABLE=1 so the driver compiles
instead of reusing its cache). NEURAL_OPENCL_BUILD_OPTIONS is appended to every
program build in the process, is cut at 255 characters with the default
options, and an option the driver does not know (-cl-nv-verbose on PoCL)
fails every build. The table prints CL_KERNEL_PRIVATE_MEM_SIZE of every flash
variant.

Usage:
  SDPAFlashBench [--q-heads 32] [--kv-heads 32] [--dk 128] [--tokens 4096]
    [--prefix 0] [--mode none] [--window 0] [--iters 20] [--warmup 3]
    [--variants 0,1,2,3] [--sweep-crossover] [--gpu-platform 0]
    [--gpu-device 0]

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, {$IFNDEF Debug}cmem,{$ENDIF}{$ENDIF}
  SysUtils, Math, neuralvolume, neuralnetwork
  {$IFDEF OpenCL}, neuralopencl, cl, ctypes{$ENDIF};

{$IFDEF OpenCL}
const
  // Parity shape: GQA groups of 2, more rows than one tile, a cache prefix.
  csParityQHeads = 4;
  csParityKVHeads = 2;
  csParityTokens = 300;
  csParityPrefix = 40;
  // Host RAM the timing volumes may take before the run refuses to start.
  csMaxHostBytes = 600 * 1024 * 1024;
  csMinBlockSeconds = 0.2;
  csTimedBlocks = 3;

type
  TVariantList = array of integer;

var
  QHeads, KVHeads, Dk, TokenCnt, PrefixLen, Window, Iters, Warmup: integer;
  PlatformIdx, DeviceIdx: integer;
  SweepCrossover: boolean;
  Variants: TVariantList;
  PlatformId: cl_platform_id;
  DeviceId: cl_device_id;
  DeviceName: string;

procedure PrintUsageAndHalt(const Problem: string);
begin
  if Problem <> '' then WriteLn('Error: ', Problem);
  WriteLn('Usage: SDPAFlashBench [--q-heads N] [--kv-heads N] [--dk N] ' +
    '[--tokens N] [--prefix N] [--mode none] [--window N] [--iters N] ' +
    '[--warmup N] [--variants 0,1,2,3] [--sweep-crossover] ' +
    '[--gpu-platform N] [--gpu-device N]');
  Halt(2);
end;

function ParseVariants(const Text: string): TVariantList;
var
  Parts: TStringArray;
  PartIdx, Variant: integer;
begin
  Result := nil;
  Parts := Text.Split([',']);
  SetLength(Result, Length(Parts));
  for PartIdx := 0 to High(Parts) do
  begin
    if (not TryStrToInt(Trim(Parts[PartIdx]), Variant)) or (Variant < 0) or
      (Variant > csFusedSDPAFlashVariants) then
      PrintUsageAndHalt('--variants takes numbers 0..' +
        IntToStr(csFusedSDPAFlashVariants));
    Result[PartIdx] := Variant;
  end;
end;

procedure ParseArguments();
var
  ArgIdx: integer;
  Arg: string;

  function NextInt(): integer;
  begin
    Inc(ArgIdx);
    if (ArgIdx > ParamCount) or (not TryStrToInt(ParamStr(ArgIdx), Result)) then
      PrintUsageAndHalt(Arg + ' needs an integer');
  end;

begin
  QHeads := 32;
  KVHeads := 32;
  Dk := 128;
  TokenCnt := 4096;
  PrefixLen := 0;
  Window := 0;
  Iters := 20;
  Warmup := 3;
  PlatformIdx := 0;
  DeviceIdx := 0;
  SweepCrossover := false;
  Variants := ParseVariants('0,1,2,3');
  ArgIdx := 1;
  while ArgIdx <= ParamCount do
  begin
    Arg := ParamStr(ArgIdx);
    if Arg = '--q-heads' then QHeads := NextInt()
    else if Arg = '--kv-heads' then KVHeads := NextInt()
    else if Arg = '--dk' then Dk := NextInt()
    else if Arg = '--tokens' then TokenCnt := NextInt()
    else if Arg = '--prefix' then PrefixLen := NextInt()
    else if Arg = '--window' then Window := NextInt()
    else if Arg = '--iters' then Iters := NextInt()
    else if Arg = '--warmup' then Warmup := NextInt()
    else if Arg = '--gpu-platform' then PlatformIdx := NextInt()
    else if Arg = '--gpu-device' then DeviceIdx := NextInt()
    else if Arg = '--sweep-crossover' then SweepCrossover := true
    else if Arg = '--variants' then
    begin
      Inc(ArgIdx);
      if ArgIdx > ParamCount then PrintUsageAndHalt('--variants needs a list');
      Variants := ParseVariants(ParamStr(ArgIdx));
    end
    else if Arg = '--mode' then
    begin
      Inc(ArgIdx);
      if (ArgIdx > ParamCount) or (ParamStr(ArgIdx) <> 'none') then
        PrintUsageAndHalt('--mode: only "none" exists so far');
    end
    else PrintUsageAndHalt('unknown argument ' + Arg);
    Inc(ArgIdx);
  end;
  if (QHeads < 1) or (KVHeads < 1) or (QHeads mod KVHeads <> 0) then
    PrintUsageAndHalt('--q-heads must be a positive multiple of --kv-heads');
  if (Dk < 1) or (TokenCnt < 1) or (PrefixLen < 0) or (Window < 0) or
    (Iters < 1) or (Warmup < 0) then
    PrintUsageAndHalt('sizes must be positive');
end;

// Cache rows each token row attends in mask mode none.
function LiveKeysPerRow(CacheLen, StepTokens, pWindow: integer): integer;
begin
  Result := CacheLen + StepTokens;
  if (pWindow > 0) and (Result > pWindow) then Result := pWindow;
end;

// Max |CPU - OpenCL| of one non-causal forward of the parity shape at Variant,
// and the variant that actually ran; MaxAbsY receives max |y| of the CPU.
function ParityMaxDiff(Variant: integer; out MaxAbsY: TNeuralFloat;
  out RanVariant: integer): TNeuralFloat;
var
  NNCpu, NNGpu: TNNet;
  LCpu, LGpu: TNNetFusedSDPA;
  StepIn, PrefixK, PrefixV: TNNetVolume;
  InDepth, KW, Pos: integer;
begin
  InDepth := (csParityQHeads + 2 * csParityKVHeads) * Dk;
  KW := csParityKVHeads * Dk;
  NNCpu := TNNet.Create();
  NNGpu := TNNet.Create();
  StepIn := TNNetVolume.Create(csParityTokens, 1, InDepth);
  PrefixK := TNNetVolume.Create(csParityPrefix, 1, KW);
  PrefixV := TNNetVolume.Create(csParityPrefix, 1, KW);
  try
    NNCpu.AddLayer(TNNetInput.Create(csParityTokens, 1, InDepth, 1));
    LCpu := TNNetFusedSDPA.Create(csParityQHeads, csParityKVHeads, Dk, False,
      Window, 0, {pCachedForwardNonCausal=}True);
    LCpu.BeginIncrementalDecode(csParityPrefix + csParityTokens);
    NNCpu.AddLayer(LCpu);
    NNCpu.SetTrainable(False, False);
    NNGpu.AddLayer(TNNetInput.Create(csParityTokens, 1, InDepth, 1));
    LGpu := TNNetFusedSDPA.Create(csParityQHeads, csParityKVHeads, Dk, False,
      Window, 0, {pCachedForwardNonCausal=}True);
    LGpu.BeginIncrementalDecode(csParityPrefix + csParityTokens);
    NNGpu.AddLayer(LGpu);
    NNGpu.SetTrainable(False, False);
    NNGpu.EnableOpenCL(PlatformId, DeviceId);
    LGpu.FusedSDPACL.FlashVariant := Variant;
    StepIn.Randomize();
    PrefixK.Randomize();
    PrefixV.Randomize();
    LCpu.AppendCacheRowsFrom(PrefixK, PrefixV);
    LGpu.AppendCacheRowsFrom(PrefixK, PrefixV);
    NNCpu.Compute(StepIn);
    NNGpu.Compute(StepIn);
    if LGpu.ForwardGPUCnt = 0
      then RanVariant := -1
      else RanVariant := LGpu.FusedSDPACL.LastFlashVariant;
    Result := 0;
    MaxAbsY := 0;
    for Pos := 0 to LCpu.Output.Size - 1 do
    begin
      Result := Max(Result, Abs(LCpu.Output.FData[Pos] -
        LGpu.Output.FData[Pos]));
      MaxAbsY := Max(MaxAbsY, Abs(LCpu.Output.FData[Pos]));
    end;
  finally
    PrefixV.Free; PrefixK.Free; StepIn.Free; NNGpu.Free; NNCpu.Free;
  end;
end;

// The resident buffers and host volumes of one timing shape.
type
  TTimingShape = record
    Tokens, CacheMax, CacheLen: integer;
    X, Y, K, V: TNNetVolume;
    BufX: cl_mem;
  end;

procedure CreateShape(Helper: TNNetFusedSDPACL; Tokens, CacheLen: integer;
  out Shape: TTimingShape);
begin
  Shape.Tokens := Tokens;
  Shape.CacheLen := CacheLen;
  Shape.CacheMax := CacheLen + Tokens;
  Shape.X := TNNetVolume.Create(Tokens, 1, (QHeads + 2 * KVHeads) * Dk);
  Shape.Y := TNNetVolume.Create(Tokens, 1, QHeads * Dk);
  Shape.K := TNNetVolume.Create(KVHeads * Shape.CacheMax * Dk, 1, 1);
  Shape.V := TNNetVolume.Create(KVHeads * Shape.CacheMax * Dk, 1, 1);
  Shape.X.Randomize();
  Shape.K.Randomize();
  Shape.V.Randomize();
  Shape.BufX := Helper.ForwardKernel.CreateBuffer(CL_MEM_READ_WRITE, Shape.X);
  Helper.ForwardKernel.WriteBuffer(Shape.BufX, Shape.X, CL_TRUE);
  Helper.UploadCache(Shape.K, Shape.V, KVHeads, Shape.CacheMax, CacheLen, Dk);
end;

procedure FreeShape(var Shape: TTimingShape);
begin
  clReleaseMemObject(Shape.BufX);
  Shape.V.Free; Shape.K.Free; Shape.Y.Free; Shape.X.Free;
end;

// One forward of Shape on the flash path (Causal false) or the split-row
// decode path (Causal true); the result stays in OpenCL memory.
procedure RunForward(Helper: TNNetFusedSDPACL; const Shape: TTimingShape;
  Causal: boolean);
var
  InvSqrtDk: TNeuralFloat;
begin
  InvSqrtDk := 1 / Sqrt(Dk);
  if Causal then
    Helper.Compute(Shape.X, Shape.Y, Shape.K, Shape.V, QHeads, KVHeads,
      QHeads div KVHeads, Dk, Shape.CacheMax, Shape.CacheLen, QHeads * Dk,
      KVHeads * Dk, Window, InvSqrtDk, 0, 0, Shape.BufX, true)
  else
    Helper.ComputeNonCausal(Shape.X, Shape.Y, Shape.K, Shape.V, QHeads,
      KVHeads, QHeads div KVHeads, Dk, Shape.CacheMax, Shape.CacheLen,
      QHeads * Dk, KVHeads * Dk, Window, InvSqrtDk, 0, 0, Shape.BufX, true);
end;

// Median milliseconds per forward over csTimedBlocks blocks.
function TimeForwards(Helper: TNNetFusedSDPACL; const Shape: TTimingShape;
  Causal: boolean): double;
var
  BlockMs: array[0..csTimedBlocks - 1] of double;
  BlockIdx, IterIdx, BlockIters: integer;
  StartTime: TDateTime;
  WarmupSeconds: double;
begin
  StartTime := Now();
  for IterIdx := 1 to Max(Warmup, 1) do RunForward(Helper, Shape, Causal);
  Helper.ForwardKernel.Finish();
  WarmupSeconds := (Now() - StartTime) * 86400.0 / Max(Warmup, 1);
  BlockIters := Iters;
  if WarmupSeconds > 0 then
    BlockIters := Max(Iters, Ceil(csMinBlockSeconds / WarmupSeconds));
  for BlockIdx := 0 to csTimedBlocks - 1 do
  begin
    StartTime := Now();
    for IterIdx := 1 to BlockIters do RunForward(Helper, Shape, Causal);
    Helper.ForwardKernel.Finish();
    BlockMs[BlockIdx] := (Now() - StartTime) * 86400.0 * 1000.0 / BlockIters;
  end;
  // Median of three.
  Result := Max(Min(BlockMs[0], BlockMs[1]),
    Min(Max(BlockMs[0], BlockMs[1]), BlockMs[2]));
end;

function Tflops(LiveKeys: int64; Ms: double): double;
begin
  if Ms <= 0 then exit(0);
  Result := 4.0 * QHeads * Dk * LiveKeys / (Ms * 1e-3) / 1e12;
end;

// True when the selected OpenCL device is a CPU (its buffers are host RAM).
function DeviceIsCPU(): boolean;
var
  DeviceType: cl_device_type;
  BytesWritten: csize_t;
begin
  DeviceType := 0;
  BytesWritten := 0;
  Result := (clGetDeviceInfo(DeviceId, CL_DEVICE_TYPE_INFO, SizeOf(DeviceType),
    @DeviceType, BytesWritten) = CL_SUCCESS) and
    ((DeviceType and CL_DEVICE_TYPE_CPU) <> 0);
end;

// Host RAM a timing shape takes: its volumes, twice on a CPU device.
function HostBytes(Tokens, CacheMax: integer): int64;
begin
  Result := (int64(Tokens) * ((QHeads + 2 * KVHeads) * Dk + QHeads * Dk) +
    2 * int64(KVHeads) * CacheMax * Dk) * SizeOf(TNeuralFloat);
  if DeviceIsCPU() then Result := 2 * Result;
end;

function PrivateMemText(Helper: TNNetFusedSDPACL; Variant: integer): string;
begin
  if Variant = 0
    then Result := '-'
    else Result := IntToStr(Helper.ForwardKernel.KernelPrivateMemSize(
      Helper.FlashKernel(Variant)));
end;

procedure RunParity(var Passed: array of boolean);
var
  VariantIdx, RanVariant: integer;
  MaxDiff, MaxAbsY, Bound: TNeuralFloat;
begin
  WriteLn(Format('Parity vs CPU: Hq=%d Hkv=%d Dk=%d tokens=%d prefix=%d ' +
    'window=%d', [csParityQHeads, csParityKVHeads, Dk, csParityTokens,
    csParityPrefix, Window]));
  for VariantIdx := 0 to High(Variants) do
  begin
    RandSeed := 20261005;
    MaxDiff := ParityMaxDiff(Variants[VariantIdx], MaxAbsY, RanVariant);
    Bound := 1e-5 * Max(1, MaxAbsY);
    Passed[VariantIdx] := (MaxDiff <= Bound) and (RanVariant >= 0);
    WriteLn(Format('  variant %d (ran %d): max|diff| = %.3e (bound %.1e) %s',
      [Variants[VariantIdx], RanVariant, MaxDiff, Bound,
       BoolToStr(Passed[VariantIdx], 'PASS', 'FAIL')]));
  end;
end;

procedure RunMainTiming(Helper: TNNetFusedSDPACL;
  const Passed: array of boolean);
var
  Shape: TTimingShape;
  VariantIdx, Variant: integer;
  LiveKeys: int64;
  Ms: double;
begin
  if HostBytes(TokenCnt, PrefixLen + TokenCnt) > csMaxHostBytes then
  begin
    WriteLn(Format('Shape needs %d MB of host RAM (limit %d MB): skipped.',
      [HostBytes(TokenCnt, PrefixLen + TokenCnt) shr 20, csMaxHostBytes shr 20]));
    exit;
  end;
  CreateShape(Helper, TokenCnt, PrefixLen, Shape);
  try
    LiveKeys := int64(TokenCnt) * LiveKeysPerRow(PrefixLen, TokenCnt, Window);
    WriteLn(Format('Timing: Hq=%d Hkv=%d Dk=%d tokens=%d prefix=%d window=%d ' +
      'mode=none, %d live keys per row', [QHeads, KVHeads, Dk, TokenCnt,
      PrefixLen, Window, LiveKeys div TokenCnt]));
    WriteLn('  variant ran  tiles   local B  private B       ms   TFLOPS');
    for VariantIdx := 0 to High(Variants) do
    begin
      Variant := Variants[VariantIdx];
      if not Passed[VariantIdx] then
      begin
        WriteLn(Format('  %7d  parity FAIL: not timed', [Variant]));
        continue;
      end;
      Helper.FlashVariant := Variant;
      Ms := TimeForwards(Helper, Shape, {Causal=}false);
      WriteLn(Format('  %7d %3d %3dx%-3d %8d %10s %8.3f %8.2f', [Variant,
        Helper.LastFlashVariant, Helper.LastQueryTileRows,
        Helper.LastKeyTileRows, int64(Helper.LastScratchBytes),
        PrivateMemText(Helper, Variant), Ms, Tflops(LiveKeys, Ms)]));
    end;
  finally
    FreeShape(Shape);
  end;
end;

procedure RunSweep(Helper: TNNetFusedSDPACL; const Passed: array of boolean);
const
  csSweepCaches: array[0..1] of integer = (8192, 32768);
  csSweepTokens: array[0..9] of integer = (1, 4, 8, 16, 32, 64, 128, 256,
    384, 512);
var
  Shape: TTimingShape;
  CacheIdx, TokenIdx, VariantIdx, Tokens: integer;
  Ms, DecodeMs: double;
  Line: string;
begin
  WriteLn;
  WriteLn('Crossover sweep, flash (mask mode none) against the split-row ' +
    'decode (causal).');
  WriteLn('  CAVEAT: the masks differ; ms per forward is the comparable figure.');
  for CacheIdx := 0 to High(csSweepCaches) do
  begin
    if HostBytes(512, csSweepCaches[CacheIdx] + 512) > csMaxHostBytes then
    begin
      WriteLn(Format('  cache %d: needs %d MB of host RAM (limit %d MB); ' +
        'use fewer --kv-heads.', [csSweepCaches[CacheIdx],
        HostBytes(512, csSweepCaches[CacheIdx] + 512) shr 20,
        csMaxHostBytes shr 20]));
      continue;
    end;
    Line := Format('  cache %6d  tokens  decode ms', [csSweepCaches[CacheIdx]]);
    for VariantIdx := 0 to High(Variants) do
      Line := Line + Format('   v%d ms', [Variants[VariantIdx]]);
    WriteLn(Line);
    for TokenIdx := 0 to High(csSweepTokens) do
    begin
      Tokens := csSweepTokens[TokenIdx];
      CreateShape(Helper, Tokens, csSweepCaches[CacheIdx], Shape);
      try
        DecodeMs := TimeForwards(Helper, Shape, {Causal=}true);
        Line := Format('               %6d %10.4f', [Tokens, DecodeMs]);
        for VariantIdx := 0 to High(Variants) do
        begin
          if not Passed[VariantIdx] then
          begin
            Line := Line + '     FAIL';
            continue;
          end;
          Helper.FlashVariant := Variants[VariantIdx];
          Ms := TimeForwards(Helper, Shape, {Causal=}false);
          Line := Line + Format(' %8.4f', [Ms]);
        end;
        WriteLn(Line);
      finally
        FreeShape(Shape);
      end;
    end;
  end;
end;

procedure RunBenchmark();
var
  NN: TNNet;
  Helper: TNNetFusedSDPACL;
  Passed: array of boolean;
  VariantIdx, Variant: integer;
begin
  Passed := nil;
  SetLength(Passed, Length(Variants));
  RunParity(Passed);
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 1));
    NN.EnableOpenCL(PlatformId, DeviceId);
    Helper := TNNetFusedSDPACL.Create(NN);
    try
      WriteLn('Device: ', DeviceName, ', local memory ',
        Helper.ForwardKernel.DeviceLocalMemSize(), ' B, compute units ',
        Helper.ForwardKernel.DeviceMaxComputeUnits());
      for VariantIdx := 0 to High(Variants) do
      begin
        Variant := Variants[VariantIdx];
        if Variant = 0 then continue;
        WriteLn(Format('  cai_sdpa_flash_v%d: %d lanes, max work-group %d, ' +
          'private %s B, tiles fit at Dk %d: %s', [Variant,
          csFusedSDPAFlashLanes[Variant],
          Helper.ForwardKernel.KernelMaxWorkGroupSize(
            Helper.FlashKernel(Variant)), PrivateMemText(Helper, Variant), Dk,
          BoolToStr(Helper.FlashTilesFit(Variant, Dk, false), 'yes', 'no')]));
      end;
      RunMainTiming(Helper, Passed);
      if SweepCrossover then RunSweep(Helper, Passed);
    finally
      Helper.Free;
    end;
  finally
    NN.Free;
  end;
  for VariantIdx := 0 to High(Passed) do
    if not Passed[VariantIdx] then Halt(1);
end;

var
  EasyCL: TEasyOpenCL;
  OpenCLProblem: string;
{$ENDIF}

begin
  {$IFDEF OpenCL}
  ParseArguments();
  EasyCL := TEasyOpenCL.Create();
  try
    if not EasyCL.SelectPlatformAndDevice(PlatformIdx, DeviceIdx,
      OpenCLProblem) then
    begin
      WriteLn('No OpenCL device: ', OpenCLProblem);
      Halt(1);
    end;
    PlatformId := EasyCL.PlatformIds[PlatformIdx];
    DeviceId := EasyCL.Devices[DeviceIdx];
    DeviceName := EasyCL.PlatformNames[PlatformIdx] + ' / ' +
      EasyCL.DeviceNames[DeviceIdx];
    RunBenchmark();
  finally
    EasyCL.Free;
  end;
  {$ELSE}
  WriteLn('SDPAFlashBench needs a build with -dOpenCL.');
  {$ENDIF}
end.
