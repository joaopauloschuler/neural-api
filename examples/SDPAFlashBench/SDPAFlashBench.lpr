program SDPAFlashBench;
(*
SDPAFlashBench: times the OpenCL flash attention of TNNetFusedSDPA
(TNNetFusedSDPACL.ComputeFlash on cai_sdpa_flash, or with --int8
ComputeFlashInt8 on cai_sdpa_flash_int8 over the int8 KV cache) over a
resident KV cache, in mask mode none or causal. No model is loaded.

Before timing, the kernel runs once on a small shape against the CPU twin
(TNNetFusedSDPA on the host, same cache format) at the requested head
dimension; a max |diff| above 1e-5 (int8: 1e-3) * max(1, max |y|) is reported
as FAIL and nothing is timed.

Timing is the wall time of the in-order OpenCL pipeline, enqueue included:
the packed [Q|K|V] input is uploaded once and bound as the step source, the
KV cache prefix is uploaded once, and the result stays in OpenCL memory; each
forward also re-appends the step's K/V rows into the same cache slots (one
cai_sdpa_append_kv(_int8) launch, small next to the attention). With --int8
the cache holds random codes and row scales near 1/127 (values of order 1).
The figure is the median of 3 timed blocks, each at least --iters forwards
and at least 200 ms. FLOPs = 4 * QHeads * Dk * (sum over token rows of the
cache rows each row attends).

Memory guard: a shape runs only if its host volumes (X, Y, K and V cache) fit
600 MB, counted twice on a CPU OpenCL device, whose buffers are host RAM too.
The default shape needs about 400 MB of host volumes.

--sweep-crossover times the flash path against the split-row decode path
(TNNetFusedSDPACL.Compute / ComputeInt8, cai_sdpa_decode_split(_int8) +
merge) for 1..512 token rows over 8k and 32k caches. With --mode causal both
compute the same attention; with --mode none the flash rows see the whole
window while the decode path stays causal. Key splits (merged by
cai_sdpa_decode_merge) are printed per flash run.

Diagnostics: NEURAL_SDPA_DUMP_PTX=<file> writes the built program (PTX on
NVIDIA). NEURAL_OPENCL_BUILD_LOG=1 prints the first build log of the process;
with NEURAL_OPENCL_BUILD_OPTIONS=-cl-nv-verbose it holds NVIDIA's per-kernel
register and spill report (set CUDA_CACHE_DISABLE=1 so the driver compiles
instead of reusing its cache). NEURAL_OPENCL_BUILD_OPTIONS is appended to every
program build in the process, is cut at 255 characters with the default
options, and an option the driver does not know (-cl-nv-verbose on PoCL)
fails every build. The table prints CL_KERNEL_PRIVATE_MEM_SIZE of the flash
kernel.

Usage:
  SDPAFlashBench [--q-heads 32] [--kv-heads 32] [--dk 128] [--tokens 4096]
    [--prefix 0] [--mode none|causal] [--window 0] [--iters 20] [--warmup 3]
    [--int8] [--sweep-crossover] [--gpu-platform 0] [--gpu-device 0]

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

var
  QHeads, KVHeads, Dk, TokenCnt, PrefixLen, Window, Iters, Warmup: integer;
  PlatformIdx, DeviceIdx: integer;
  SweepCrossover, Causal, Int8KV: boolean;
  PlatformId: cl_platform_id;
  DeviceId: cl_device_id;
  DeviceName: string;

procedure PrintUsageAndHalt(const Problem: string);
begin
  if Problem <> '' then WriteLn('Error: ', Problem);
  WriteLn('Usage: SDPAFlashBench [--q-heads N] [--kv-heads N] [--dk N] ' +
    '[--tokens N] [--prefix N] [--mode none|causal] [--window N] ' +
    '[--iters N] [--warmup N] [--int8] [--sweep-crossover] ' +
    '[--gpu-platform N] [--gpu-device N]');
  Halt(2);
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
  Causal := false;
  Int8KV := false;
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
    else if Arg = '--int8' then Int8KV := true
    else if Arg = '--mode' then
    begin
      Inc(ArgIdx);
      if (ArgIdx > ParamCount) or ((ParamStr(ArgIdx) <> 'none') and
        (ParamStr(ArgIdx) <> 'causal')) then
        PrintUsageAndHalt('--mode takes none or causal');
      Causal := ParamStr(ArgIdx) = 'causal';
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

// Cache rows summed over the token rows of a causal step.
function CausalLiveKeys(CacheLen, StepTokens, pWindow: integer): int64;
var
  RowIdx, RowKeys: integer;
begin
  Result := 0;
  for RowIdx := 0 to StepTokens - 1 do
  begin
    RowKeys := CacheLen + RowIdx + 1;
    if (pWindow > 0) and (RowKeys > pWindow) then RowKeys := pWindow;
    Inc(Result, RowKeys);
  end;
end;

// Cache rows summed over the token rows of a step in the selected mode.
function StepLiveKeys(CacheLen, StepTokens: integer): int64;
begin
  if Causal
    then Result := CausalLiveKeys(CacheLen, StepTokens, Window)
    else Result := int64(StepTokens) * LiveKeysPerRow(CacheLen, StepTokens,
      Window);
end;

// Max |CPU - OpenCL| of one forward of the parity shape; MaxAbsY receives
// max |y| of the CPU, RanFlash whether the OpenCL forward took flash.
function ParityMaxDiff(out MaxAbsY: TNeuralFloat;
  out RanFlash: boolean): TNeuralFloat;
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
      Window, 0, {pCachedForwardNonCausal=}not Causal);
    LCpu.BeginIncrementalDecode(csParityPrefix + csParityTokens, Int8KV);
    NNCpu.AddLayer(LCpu);
    NNCpu.SetTrainable(False, False);
    NNGpu.AddLayer(TNNetInput.Create(csParityTokens, 1, InDepth, 1));
    LGpu := TNNetFusedSDPA.Create(csParityQHeads, csParityKVHeads, Dk, False,
      Window, 0, {pCachedForwardNonCausal=}not Causal);
    LGpu.BeginIncrementalDecode(csParityPrefix + csParityTokens, Int8KV);
    NNGpu.AddLayer(LGpu);
    NNGpu.SetTrainable(False, False);
    NNGpu.EnableOpenCL(PlatformId, DeviceId);
    StepIn.Randomize();
    PrefixK.Randomize();
    PrefixV.Randomize();
    LCpu.AppendCacheRowsFrom(PrefixK, PrefixV);
    LGpu.AppendCacheRowsFrom(PrefixK, PrefixV);
    NNCpu.Compute(StepIn);
    NNGpu.Compute(StepIn);
    RanFlash := (LGpu.ForwardGPUCnt > 0) and
      (LGpu.FusedSDPACL.LastPath = sdpaPathFlash);
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
    KQ, VQ: TNNetVolumeQuant8;
    BufX: cl_mem;
  end;

// Random codes in [-127, 127] and row scales of about 1/127.
procedure RandomizeInt8Cache(Cache: TNNetVolumeQuant8);
var
  Pos, MaxPos: integer;
begin
  MaxPos := Cache.Size - 1;
  for Pos := 0 to MaxPos do Cache.SetRaw(Pos, Random(255) - 127);
  MaxPos := Cache.ScaleCount - 1;
  for Pos := 0 to MaxPos do Cache.ScalePtr^[Pos] := (0.5 + Random) / 127;
end;

procedure CreateShape(Helper: TNNetFusedSDPACL; Tokens, CacheLen: integer;
  out Shape: TTimingShape);
begin
  Shape.Tokens := Tokens;
  Shape.CacheLen := CacheLen;
  Shape.CacheMax := CacheLen + Tokens;
  Shape.X := TNNetVolume.Create(Tokens, 1, (QHeads + 2 * KVHeads) * Dk);
  Shape.Y := TNNetVolume.Create(Tokens, 1, QHeads * Dk);
  Shape.X.Randomize();
  Shape.BufX := Helper.ForwardKernel.CreateBuffer(CL_MEM_READ_WRITE, Shape.X);
  Helper.ForwardKernel.WriteBuffer(Shape.BufX, Shape.X, CL_TRUE);
  Shape.K := nil;
  Shape.V := nil;
  Shape.KQ := nil;
  Shape.VQ := nil;
  if Int8KV then
  begin
    Shape.KQ := TNNetVolumeQuant8.Create();
    Shape.VQ := TNNetVolumeQuant8.Create();
    Shape.KQ.ReSize(Shape.CacheMax, KVHeads, Dk);
    Shape.VQ.ReSize(Shape.CacheMax, KVHeads, Dk);
    RandomizeInt8Cache(Shape.KQ);
    RandomizeInt8Cache(Shape.VQ);
    Helper.UploadCacheInt8(Shape.KQ, Shape.VQ, KVHeads, Shape.CacheMax,
      CacheLen, Dk);
  end
  else
  begin
    Shape.K := TNNetVolume.Create(KVHeads * Shape.CacheMax * Dk, 1, 1);
    Shape.V := TNNetVolume.Create(KVHeads * Shape.CacheMax * Dk, 1, 1);
    Shape.K.Randomize();
    Shape.V.Randomize();
    Helper.UploadCache(Shape.K, Shape.V, KVHeads, Shape.CacheMax, CacheLen, Dk);
  end;
end;

procedure FreeShape(var Shape: TTimingShape);
begin
  clReleaseMemObject(Shape.BufX);
  Shape.VQ.Free; Shape.KQ.Free;
  Shape.V.Free; Shape.K.Free; Shape.Y.Free; Shape.X.Free;
end;

// One forward of Shape on the split-row decode path (Decode) or the flash
// path in the selected mode; the result stays in OpenCL memory.
procedure RunForward(Helper: TNNetFusedSDPACL; const Shape: TTimingShape;
  Decode: boolean);
var
  InvSqrtDk: TNeuralFloat;
begin
  InvSqrtDk := 1 / Sqrt(Dk);
  if Decode and Int8KV then
    Helper.ComputeInt8(Shape.X, Shape.Y, Shape.KQ, Shape.VQ, QHeads, KVHeads,
      QHeads div KVHeads, Dk, Shape.CacheMax, Shape.CacheLen, QHeads * Dk,
      KVHeads * Dk, Window, InvSqrtDk, 0, 0, Shape.BufX, true)
  else if Decode then
    Helper.Compute(Shape.X, Shape.Y, Shape.K, Shape.V, QHeads, KVHeads,
      QHeads div KVHeads, Dk, Shape.CacheMax, Shape.CacheLen, QHeads * Dk,
      KVHeads * Dk, Window, InvSqrtDk, 0, 0, Shape.BufX, true)
  else if Int8KV then
    Helper.ComputeFlashInt8(Shape.X, Shape.Y, Shape.KQ, Shape.VQ, QHeads,
      KVHeads, QHeads div KVHeads, Dk, Shape.CacheMax, Shape.CacheLen,
      QHeads * Dk, KVHeads * Dk, Window, Causal, [], 0, InvSqrtDk, 0, 0,
      Shape.BufX, true)
  else
    Helper.ComputeFlash(Shape.X, Shape.Y, Shape.K, Shape.V, QHeads,
      KVHeads, QHeads div KVHeads, Dk, Shape.CacheMax, Shape.CacheLen,
      QHeads * Dk, KVHeads * Dk, Window, Causal, [], 0, InvSqrtDk, 0, 0,
      Shape.BufX, true);
end;

// Median milliseconds per forward over csTimedBlocks blocks.
function TimeForwards(Helper: TNNetFusedSDPACL; const Shape: TTimingShape;
  Decode: boolean): double;
var
  BlockMs: array[0..csTimedBlocks - 1] of double;
  BlockIdx, IterIdx, BlockIters: integer;
  StartTime: TDateTime;
  WarmupSeconds: double;
begin
  StartTime := Now();
  for IterIdx := 1 to Max(Warmup, 1) do RunForward(Helper, Shape, Decode);
  Helper.ForwardKernel.Finish();
  WarmupSeconds := (Now() - StartTime) * 86400.0 / Max(Warmup, 1);
  BlockIters := Iters;
  if WarmupSeconds > 0 then
    BlockIters := Max(Iters, Ceil(csMinBlockSeconds / WarmupSeconds));
  for BlockIdx := 0 to csTimedBlocks - 1 do
  begin
    StartTime := Now();
    for IterIdx := 1 to BlockIters do RunForward(Helper, Shape, Decode);
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
  Result := int64(Tokens) * ((QHeads + 2 * KVHeads) * Dk + QHeads * Dk) *
    SizeOf(TNeuralFloat);
  if Int8KV
    then Inc(Result, 2 * int64(KVHeads) * CacheMax * (Dk + SizeOf(TNeuralFloat)))
    else Inc(Result, 2 * int64(KVHeads) * CacheMax * Dk * SizeOf(TNeuralFloat));
  if DeviceIsCPU() then Result := 2 * Result;
end;

function PrivateMemText(Helper: TNNetFusedSDPACL): string;
begin
  Result := IntToStr(Helper.ForwardKernel.KernelPrivateMemSize(
    Helper.FlashKernel(Int8KV)));
end;

function RunParity(): boolean;
var
  RanFlash: boolean;
  MaxDiff, MaxAbsY, Bound: TNeuralFloat;
begin
  WriteLn(Format('Parity vs CPU: Hq=%d Hkv=%d Dk=%d tokens=%d prefix=%d ' +
    'window=%d mode=%s cache=%s', [csParityQHeads, csParityKVHeads, Dk,
    csParityTokens, csParityPrefix, Window,
    BoolToStr(Causal, 'causal', 'none'), BoolToStr(Int8KV, 'int8', 'fp32')]));
  RandSeed := 20261005;
  MaxDiff := ParityMaxDiff(MaxAbsY, RanFlash);
  Bound := IfThen(Int8KV, 1e-3, 1e-5) * Max(1, MaxAbsY);
  Result := (MaxDiff <= Bound) and RanFlash;
  WriteLn(Format('  flash ran: %s, max|diff| = %.3e (bound %.1e) %s',
    [BoolToStr(RanFlash, 'yes', 'no'), MaxDiff, Bound,
     BoolToStr(Result, 'PASS', 'FAIL')]));
end;

procedure RunMainTiming(Helper: TNNetFusedSDPACL);
var
  Shape: TTimingShape;
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
    LiveKeys := StepLiveKeys(PrefixLen, TokenCnt);
    WriteLn(Format('Timing: Hq=%d Hkv=%d Dk=%d tokens=%d prefix=%d window=%d ' +
      'mode=%s cache=%s, %d live keys per row on average', [QHeads, KVHeads,
      Dk, TokenCnt, PrefixLen, Window, BoolToStr(Causal, 'causal', 'none'),
      BoolToStr(Int8KV, 'int8', 'fp32'), LiveKeys div TokenCnt]));
    WriteLn('    tiles  splits  local B  private B       ms   TFLOPS');
    Ms := TimeForwards(Helper, Shape, {Decode=}false);
    WriteLn(Format('  %3dx%-3d %6d %8d %10s %8.3f %8.2f', [
      Helper.LastQueryTileRows, Helper.LastKeyTileRows,
      Helper.LastFlashSplits, int64(Helper.LastScratchBytes),
      PrivateMemText(Helper), Ms, Tflops(LiveKeys, Ms)]));
  finally
    FreeShape(Shape);
  end;
end;

procedure RunSweep(Helper: TNNetFusedSDPACL);
const
  csSweepCaches: array[0..1] of integer = (8192, 32768);
  csSweepTokens: array[0..11] of integer = (1, 2, 3, 4, 8, 16, 32, 64, 128,
    256, 384, 512);
var
  Shape: TTimingShape;
  CacheIdx, TokenIdx, Tokens: integer;
  Ms, DecodeMs: double;
begin
  WriteLn;
  WriteLn(Format('Crossover sweep, flash (mode %s) against the split-row ' +
    'decode (causal); "/n" = flash key splits.',
    [BoolToStr(Causal, 'causal', 'none')]));
  if not Causal then
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
    WriteLn(Format('  cache %6d  tokens  decode ms    flash ms',
      [csSweepCaches[CacheIdx]]));
    for TokenIdx := 0 to High(csSweepTokens) do
    begin
      Tokens := csSweepTokens[TokenIdx];
      CreateShape(Helper, Tokens, csSweepCaches[CacheIdx], Shape);
      try
        DecodeMs := TimeForwards(Helper, Shape, {Decode=}true);
        Ms := TimeForwards(Helper, Shape, {Decode=}false);
        WriteLn(Format('               %6d %10.4f %8.4f/%-2d', [Tokens,
          DecodeMs, Ms, Helper.LastFlashSplits]));
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
begin
  if not RunParity() then Halt(1);
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 1));
    NN.EnableOpenCL(PlatformId, DeviceId);
    Helper := TNNetFusedSDPACL.Create(NN);
    try
      WriteLn('Device: ', DeviceName, ', local memory ',
        Helper.ForwardKernel.DeviceLocalMemSize(), ' B, compute units ',
        Helper.ForwardKernel.DeviceMaxComputeUnits());
      WriteLn(Format('  %s: %d lanes, max work-group %d, private %s B, ' +
        'tiles fit at Dk %d: %s', [BoolToStr(Int8KV, 'cai_sdpa_flash_int8',
        'cai_sdpa_flash'), csFusedSDPAFlashLanes,
        Helper.ForwardKernel.KernelMaxWorkGroupSize(Helper.FlashKernel(Int8KV)),
        PrivateMemText(Helper), Dk,
        BoolToStr(Helper.FlashTilesFit(Dk, Int8KV), 'yes', 'no')]));
      RunMainTiming(Helper);
      if SweepCrossover then RunSweep(Helper);
    finally
      Helper.Free;
    end;
  finally
    NN.Free;
  end;
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
