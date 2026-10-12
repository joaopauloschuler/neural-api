program GemmBench;
(*
GemmBench: times the tiled int8 / int4 OpenCL GEMM that TNNetPointwiseConvLinear
runs for a window of tokens (TDotProductSharedKernel.ComputeInt8 / ComputeInt4
-> RunTiledGemm on cai_dot_product_int8_tiled / cai_dot_product_int4_tiled) at
the Qwen-Image-2.1 projection shapes: Q/K/V/O 4096 -> 4096, GateUp 4096 ->
24576, Down 12288 -> 4096, over --tokens columns. No model is loaded.

First it prints each tiled code kernel's work-group cap, local memory and
private (spilled) bytes as the device compiler reports them. Before timing,
each weight mode runs once on two small shapes (516 and 515
rows, a ragged reduction axis, 37 columns) against a plain Pascal loop over the
same codes; a max |diff| above 1e-4 * max(1, max |y|) is reported as FAIL and
nothing is timed. With FP16 B (--b fp16|both) the int8 parity also runs on
the half kernels: max |diff| against the Pascal loop over B rounded to half
must stay within the same 1e-4 bound, and against the unrounded B within
2^-11 * sum |w * b| (half rounding) plus that bound; the first diff must be
the smaller one, which proves the kernel read B as half.

Timing: the codes, scales and the B operand are uploaded once; each launch
reads them in OpenCL memory and leaves its result there. The figure is the
median of 3 timed blocks of the same launch count, at least --iters and grown
until a block lasts 200 ms (GetTickCount64, monotonic). FLOPs = 2 * rows *
reduction * tokens. Memory guard: a shape runs only if its host volumes fit
600 MB, counted with the OpenCL buffers on a CPU device. Each timing line
also prints the bytes of A (codes + scales) and B per launch, and GB/s =
(A + B + result bytes, each counted once) / time, a lower bound on the
traffic, to compare against the device's DRAM bandwidth (split-K partials and
L2 re-reads are not counted).

Usage:
  GemmBench [--tokens 4096] [--iters 10] [--int8 | --int4] [--pico]
    [--rows R --reduction K] [--grid auto|large|block] [--split-k auto|off|N]
    [--b fp32|fp16|both] [--gpu-platform 0] [--gpu-device 0]
--pico times hidden 64 / MLP 192 over 32 tokens (a smoke run). --rows and
--reduction time that one shape instead (e.g. an LLM projection at a prefill
window: --rows 1024 --reduction 2560 --tokens 64). --grid forces the large
(512 rows x 16 columns) or block (128 x 128) work-groups of the code kernels;
auto picks per shape, as TNNetPointwiseConvLinear does
(SetTiledGemmCodesGrid). --split-k cuts the large grid's reduction into N
K-splits (raw partials + merge), off never splits, auto picks per shape
(SetTiledGemmSplitK); the block grid never splits. Each timing line reports
the lanes and K-splits that ran. Parity runs on both grids and on the large
grid at 3 K-splits.
--b picks the B operand of the int8 rows (default fp32). fp16 arms the layer
FP16 activation mode (PrepareForComputeInt8 pFP16 = True, the --gpu-fp16 path)
so the _h kernels run, and prints two rows: "fp16" times the GEMM alone on the
resident half B; "fp16+cast" binds a resident FP32 B as a layer's resident
source is bound, so each launch also runs cai_f32_to_half, as a real layer
pays. int4 has no FP16-B kernel: it runs with FP32 B only (none under fp16).

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, {$IFNDEF Debug}cmem,{$ENDIF}{$ENDIF}
  SysUtils, Math, neuralvolume
  {$IFDEF OpenCL}, neuralopencl, cl, ctypes{$ENDIF};

{$IFDEF OpenCL}
const
  csHidden = 4096;
  csMlpHidden = 12288;
  csPicoHidden = 64;
  csPicoMlpHidden = 192;
  csPicoTokens = 32;
  csParityColumns = 37;
  csParityReduction = 1003; // int4 rounds it down to whole blocks of 32
  csParitySplits = 3;
  csMaxHostBytes = 600 * 1024 * 1024;
  csMinBlockSeconds = 0.2;
  csTimedBlocks = 3;

type
  TGemmShape = record
    Name: string;
    Rows, Reduction: integer;
  end;

var
  TokenCnt, Iters, PlatformIdx, DeviceIdx, CustomRows, CustomReduction: integer;
  SplitK: integer;
  Grid: TTiledGemmCodesGrid;
  RunInt8, RunInt4, Pico, RunBFP32, RunBFP16: boolean;
  PlatformId: cl_platform_id;
  DeviceId: cl_device_id;
  DeviceName: string;

procedure PrintUsageAndHalt(const Problem: string);
begin
  if Problem <> '' then WriteLn('Error: ', Problem);
  WriteLn('Usage: GemmBench [--tokens N] [--iters N] [--int8 | --int4] ' +
    '[--pico] [--rows R --reduction K] [--grid auto|large|block] ' +
    '[--split-k auto|off|N] [--b fp32|fp16|both] [--gpu-platform N] ' +
    '[--gpu-device N]');
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
  TokenCnt := 0;
  Iters := 10;
  PlatformIdx := 0;
  DeviceIdx := 0;
  RunInt8 := true;
  RunInt4 := true;
  Pico := false;
  CustomRows := 0;
  CustomReduction := 0;
  Grid := tgcAuto;
  SplitK := csTiledGemmSplitKAuto;
  RunBFP32 := true;
  RunBFP16 := false;
  ArgIdx := 1;
  while ArgIdx <= ParamCount do
  begin
    Arg := ParamStr(ArgIdx);
    if Arg = '--tokens' then TokenCnt := NextInt()
    else if Arg = '--iters' then Iters := NextInt()
    else if Arg = '--gpu-platform' then PlatformIdx := NextInt()
    else if Arg = '--gpu-device' then DeviceIdx := NextInt()
    else if Arg = '--int8' then RunInt4 := false
    else if Arg = '--int4' then RunInt8 := false
    else if Arg = '--pico' then Pico := true
    else if Arg = '--rows' then CustomRows := NextInt()
    else if Arg = '--reduction' then CustomReduction := NextInt()
    else if Arg = '--grid' then
    begin
      Inc(ArgIdx);
      if ArgIdx > ParamCount then PrintUsageAndHalt('--grid needs a value');
      if ParamStr(ArgIdx) = 'large' then Grid := tgcLarge
      else if ParamStr(ArgIdx) = 'block' then Grid := tgcBlock
      else if ParamStr(ArgIdx) <> 'auto' then
        PrintUsageAndHalt('--grid takes auto, large or block');
    end
    else if Arg = '--split-k' then
    begin
      Inc(ArgIdx);
      if ArgIdx > ParamCount then PrintUsageAndHalt('--split-k needs a value');
      if ParamStr(ArgIdx) = 'auto' then SplitK := csTiledGemmSplitKAuto
      else if ParamStr(ArgIdx) = 'off' then SplitK := 0
      else if (not TryStrToInt(ParamStr(ArgIdx), SplitK)) or (SplitK < 0) then
        PrintUsageAndHalt('--split-k takes auto, off or a split count');
    end
    else if Arg = '--b' then
    begin
      Inc(ArgIdx);
      if ArgIdx > ParamCount then PrintUsageAndHalt('--b needs a value');
      if (ParamStr(ArgIdx) <> 'fp32') and (ParamStr(ArgIdx) <> 'fp16') and
        (ParamStr(ArgIdx) <> 'both') then
        PrintUsageAndHalt('--b takes fp32, fp16 or both');
      RunBFP32 := ParamStr(ArgIdx) <> 'fp16';
      RunBFP16 := ParamStr(ArgIdx) <> 'fp32';
    end
    else PrintUsageAndHalt('unknown argument ' + Arg);
    Inc(ArgIdx);
  end;
  if not (RunInt8 or RunInt4) then
    PrintUsageAndHalt('--int8 and --int4 exclude each other');
  if RunBFP16 and not RunInt8 then
    PrintUsageAndHalt('--b fp16 and both need the int8 rows (drop --int4)');
  if TokenCnt = 0 then
  begin
    if Pico then TokenCnt := csPicoTokens else TokenCnt := 4096;
  end;
  if (TokenCnt < 1) or (Iters < 1) then PrintUsageAndHalt('sizes must be positive');
  if (CustomRows <> 0) or (CustomReduction <> 0) then
  begin
    if (CustomRows < 1) or (CustomReduction < 1) then
      PrintUsageAndHalt('--rows and --reduction go together and must be positive');
    if RunInt4 and RunBFP32 and (CustomReduction mod 32 <> 0) then
      PrintUsageAndHalt('int4 needs --reduction to be a multiple of 32 (add --int8)');
  end;
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

function WeightBytes(Rows, Reduction: integer; Int4: boolean): int64;
begin
  if Int4
    then Result := int64(Rows) * (Reduction div 2) +
      int64(Rows) * (Reduction div 32) * SizeOf(TNeuralFloat)
    else Result := int64(Rows) * Reduction + int64(Rows) * SizeOf(TNeuralFloat);
end;

// Host volumes of a shape (B operand, weights), plus the OpenCL buffers on a
// CPU device (weights, result, B; with BHalf FP32 staging, half and resident B).
function HostBytes(Rows, Reduction, Columns: integer; Int4, BHalf: boolean): int64;
var
  BBytes, DeviceBBytes: int64;
begin
  BBytes := int64(Reduction) * Columns * SizeOf(TNeuralFloat);
  Result := BBytes + WeightBytes(Rows, Reduction, Int4);
  if BHalf then DeviceBBytes := 2 * BBytes + BBytes div 2 else DeviceBBytes := BBytes;
  if DeviceIsCPU() then
    Inc(Result, DeviceBBytes + WeightBytes(Rows, Reduction, Int4) +
      int64(Rows) * Columns * SizeOf(TNeuralFloat));
end;

// Random int8 codes [a + k*Rows] and row scales, or Q4_0 packed pairs
// [a + p*Rows] and block scales [a + blk*Rows]; values keep |y| of order 1.
procedure FillWeights(Rows, Reduction: integer; Int4: boolean;
  out Codes: TBytes; out Scales: TNeuralFloatDynArr);
var
  Pos, MaxPos: integer;
  ScaleBase: TNeuralFloat;
begin
  if Int4 then
  begin
    SetLength(Codes, int64(Rows) * (Reduction div 2));
    SetLength(Scales, int64(Rows) * (Reduction div 32));
    ScaleBase := 1 / (8 * Sqrt(Reduction));
  end
  else
  begin
    SetLength(Codes, int64(Rows) * Reduction);
    SetLength(Scales, Rows);
    ScaleBase := 1 / (127 * Sqrt(Reduction));
  end;
  MaxPos := High(Codes);
  for Pos := 0 to MaxPos do Codes[Pos] := Random(256);
  // int8 codes are symmetric: -128 is not a valid code.
  if not Int4 then
    for Pos := 0 to MaxPos do
      if Codes[Pos] = 128 then Codes[Pos] := 129;
  MaxPos := High(Scales);
  for Pos := 0 to MaxPos do Scales[Pos] := ScaleBase * (0.5 + Random);
end;

// Weight (row a, element k) of the codes FillWeights produced.
function WeightAt(const Codes: TBytes; const Scales: TNeuralFloatDynArr;
  Rows, a, k: integer; Int4: boolean): TNeuralFloat;
var
  PairByte: integer;
begin
  if Int4 then
  begin
    PairByte := Codes[a + (k shr 1) * Rows];
    if (k and 1) = 1 then PairByte := PairByte shr 4;
    Result := ((PairByte and 15) - 8) * Scales[a + (k shr 5) * Rows];
  end
  else
    Result := ShortInt(Codes[a + k * Rows]) * Scales[a];
end;

// Arms DotCL with the weights of a shape and B for Columns columns; BHalf
// (int8 only) arms the FP16 activation mode, as a --gpu-fp16 layer does.
function PrepareShape(DotCL: TDotProductSharedKernel; Rows, Reduction: integer;
  Int4, BHalf: boolean; const Codes: TBytes; const Scales: TNeuralFloatDynArr;
  B: TNNetVolume): boolean;
begin
  if Int4
    then Result := DotCL.PrepareForComputeInt4(@Codes[0], @Scales[0], Rows,
      Reduction, B) = CL_SUCCESS
    else Result := DotCL.PrepareForComputeInt8(@Codes[0], @Scales[0], Rows,
      Reduction, B, BHalf) = CL_SUCCESS;
end;

// One launch over the resident B (uploaded when NewB). A non-nil ResidentB is
// bound as a resident FP32 source; the FP16 mode narrows it on every launch.
procedure RunGemm(DotCL: TDotProductSharedKernel; B: TNNetVolume;
  Int4, NewB: boolean; ResidentB: cl_mem = nil);
begin
  if Int4
    then DotCL.ComputeInt4(B, {ActFN=}0, NewB)
    else DotCL.ComputeInt8(B, {ActFN=}0, NewB, nil, true, ResidentB);
end;

function ModeName(Int4: boolean): string;
begin
  if Int4 then Result := 'int4' else Result := 'int8';
end;

// The code kernel the last launch ran, from its lanes and K-splits and the
// B format (TiledCodesKernelName adds _h when the FP16 mode is armed).
function TiledKernelName(DotCL: TDotProductSharedKernel; Int4, BHalf: boolean): string;
begin
  Result := ModeName(Int4) + '_tiled';
  if BHalf then Result := Result + '_h';
  if DotCL.LastTiledGemmLanes = csTiledGemmBlockLanes then Result := Result + '_block';
  if DotCL.LastTiledGemmSplits > 1 then Result := Result + '_splitk';
end;

// Max |OpenCL - Pascal| of one launch at Rows x Reduction x csParityColumns;
// with BHalf the reference reads B rounded to half (see the header).
function ParityShape(DotCL: TDotProductSharedKernel; Rows, Reduction: integer;
  Int4, BHalf: boolean): boolean;
const
  csHalfUnitRoundoff = 1 / 2048; // 2^-11
var
  Codes: TBytes;
  Scales: TNeuralFloatDynArr;
  HalfBits: array of Word;
  B, BRef, Y: TNNetVolume;
  a, k, Col, LaunchesBefore: integer;
  Weight, Sum, SumExact, SumAbs, Diff, MaxDiff, MaxDiffExact, MaxAbsY,
    Bound: TNeuralFloat;
  Ran, ExactOk, HalfSeen: boolean;
begin
  FillWeights(Rows, Reduction, Int4, Codes, Scales);
  B := TNNetVolume.Create(Reduction * csParityColumns, 1, 1);
  BRef := TNNetVolume.Create(Reduction * csParityColumns, 1, 1);
  Y := TNNetVolume.Create(Rows * csParityColumns, 1, 1);
  try
    B.Randomize();
    BRef.Copy(B);
    if BHalf then
    begin
      SetLength(HalfBits, B.Size);
      TNNetVolume.EncodeF16(@HalfBits[0], TNeuralFloatArrPtr(@B.FData[0]), B.Size);
      TNNetVolume.DecodeF16(TNeuralFloatArrPtr(@BRef.FData[0]), @HalfBits[0], B.Size);
    end;
    Result := PrepareShape(DotCL, Rows, Reduction, Int4, BHalf, Codes, Scales, B);
    if not Result then exit;
    LaunchesBefore := DotCL.TiledGemmLaunchCount;
    RunGemm(DotCL, B, Int4, true);
    DotCL.FinishAndLoadResult(Y);
    Ran := DotCL.TiledGemmLaunchCount > LaunchesBefore;
    MaxDiff := 0;
    MaxDiffExact := 0;
    MaxAbsY := 0;
    ExactOk := true;
    for Col := 0 to csParityColumns - 1 do
      for a := 0 to Rows - 1 do
      begin
        Sum := 0;
        SumExact := 0;
        SumAbs := 0;
        for k := 0 to Reduction - 1 do
        begin
          Weight := WeightAt(Codes, Scales, Rows, a, k, Int4);
          Sum := Sum + Weight * BRef.FData[Col * Reduction + k];
          SumExact := SumExact + Weight * B.FData[Col * Reduction + k];
          SumAbs := SumAbs + Abs(Weight * B.FData[Col * Reduction + k]);
        end;
        MaxDiff := Max(MaxDiff, Abs(Sum - Y.FData[Col * Rows + a]));
        MaxAbsY := Max(MaxAbsY, Abs(Sum));
        Diff := Abs(SumExact - Y.FData[Col * Rows + a]);
        MaxDiffExact := Max(MaxDiffExact, Diff);
        if Diff > csHalfUnitRoundoff * SumAbs + 1e-4 * Max(1, Abs(SumExact)) then
          ExactOk := false;
      end;
    Bound := 1e-4 * Max(1, MaxAbsY);
    Result := Ran and (MaxDiff <= Bound);
    if BHalf then
    begin
      HalfSeen := MaxDiff < MaxDiffExact;
      Result := Result and ExactOk and HalfSeen;
    end;
    Write(Format('  %s %d rows x %d x %d columns, %s, %d lanes, %d K-splits: ' +
      'tiled ran: %s, max|diff| = %.3e (bound %.1e)', [ModeName(Int4), Rows,
      Reduction, csParityColumns, TiledKernelName(DotCL, Int4, BHalf),
      DotCL.LastTiledGemmLanes, DotCL.LastTiledGemmSplits,
      BoolToStr(Ran, 'yes', 'no'), MaxDiff, Bound]));
    if BHalf then
      Write(Format(', vs unrounded B %.3e (bound 2^-11*sum|w*b|: %s), B read ' +
        'as half: %s', [MaxDiffExact, BoolToStr(ExactOk, 'met', 'EXCEEDED'),
        BoolToStr(HalfSeen, 'yes', 'NO')]));
    WriteLn(' ', BoolToStr(Result, 'PASS', 'FAIL'));
  finally
    Y.Free;
    BRef.Free;
    B.Free;
  end;
end;

// Work-group size cap, local and private (spilled) bytes of each tiled code
// kernel, as the device compiler reports them for Kernel's program.
procedure PrintKernelResources(Kernel: TNeuralKernel);
const
  csCodeKernels: array[0..5] of string = ('cai_dot_product_int8_tiled',
    'cai_dot_product_int8_tiled_block', 'cai_dot_product_int8_tiled_h',
    'cai_dot_product_int8_tiled_h_block', 'cai_dot_product_int4_tiled',
    'cai_dot_product_int4_tiled_block');
var
  KernelIdx: integer;
  K: cl_kernel;
begin
  WriteLn('Kernel resources (max work-group, local bytes, private bytes):');
  for KernelIdx := 0 to High(csCodeKernels) do
  begin
    K := Kernel.CreateKernel(csCodeKernels[KernelIdx]);
    if not Assigned(K) then
    begin
      WriteLn('  ', csCodeKernels[KernelIdx], ': not created');
      continue;
    end;
    WriteLn(Format('  %-34s %5d %7d %7d', [csCodeKernels[KernelIdx],
      Kernel.KernelMaxWorkGroupSize(K), Kernel.KernelLocalMemSize(K),
      Kernel.KernelPrivateMemSize(K)]));
    clReleaseKernel(K);
  end;
end;

// Median milliseconds per launch over csTimedBlocks blocks of equal length.
function TimeLaunches(Kernel: TNeuralKernel; DotCL: TDotProductSharedKernel;
  B: TNNetVolume; Int4: boolean; ResidentB: cl_mem): double;
var
  BlockMs: array[0..csTimedBlocks - 1] of double;
  BlockIdx, BlockIters: integer;
  StartTick, Elapsed: QWord;

  function TimeBlock(pIters: integer): QWord;
  var
    LaunchIdx: integer;
  begin
    StartTick := GetTickCount64();
    for LaunchIdx := 1 to pIters do RunGemm(DotCL, B, Int4, false, ResidentB);
    Kernel.Finish();
    Result := GetTickCount64() - StartTick;
  end;

begin
  TimeBlock(2); // warm-up
  BlockIters := Iters;
  Elapsed := TimeBlock(BlockIters);
  while Elapsed < Round(csMinBlockSeconds * 1000) do
  begin
    if Elapsed = 0
      then BlockIters := BlockIters * 10
      else BlockIters := Ceil(BlockIters * csMinBlockSeconds * 1000 * 1.1 / Elapsed);
    Elapsed := TimeBlock(BlockIters);
  end;
  for BlockIdx := 0 to csTimedBlocks - 1 do
    BlockMs[BlockIdx] := TimeBlock(BlockIters) / BlockIters;
  Result := Max(Min(BlockMs[0], BlockMs[1]),
    Min(Max(BlockMs[0], BlockMs[1]), BlockMs[2]));
end;

// Times one launch form and prints its row; CastBytes are the extra bytes a
// cai_f32_to_half pass moves per launch (0 without one).
procedure TimeAndPrintRow(Kernel: TNeuralKernel; DotCL: TDotProductSharedKernel;
  const Shape: TGemmShape; B: TNNetVolume; Int4, BHalf: boolean;
  ResidentB: cl_mem; const BName: string; CastBytes: int64);
var
  Ms, Tflops, GBs: double;
  ABytes, BBytes, YBytes: int64;
begin
  Ms := TimeLaunches(Kernel, DotCL, B, Int4, ResidentB);
  ABytes := WeightBytes(Shape.Rows, Shape.Reduction, Int4);
  BBytes := int64(Shape.Reduction) * TokenCnt * SizeOf(TNeuralFloat);
  if BHalf then BBytes := BBytes div 2;
  YBytes := int64(Shape.Rows) * TokenCnt * SizeOf(TNeuralFloat);
  Tflops := 0;
  GBs := 0;
  if Ms > 0 then
  begin
    Tflops := 2.0 * Shape.Rows * Shape.Reduction * TokenCnt / (Ms * 1e-3) / 1e12;
    GBs := (ABytes + BBytes + YBytes + CastBytes) / (Ms * 1e-3) / 1e9;
  end;
  WriteLn(Format('  %-6s %s %-9s %5dx%5dx%5d %-24s %3d %2d %9.3f %7.2f %7.1f %7.1f %6.0f',
    [Shape.Name, ModeName(Int4), BName, Shape.Rows, Shape.Reduction, TokenCnt,
    TiledKernelName(DotCL, Int4, BHalf), DotCL.LastTiledGemmLanes,
    DotCL.LastTiledGemmSplits, Ms, Tflops, ABytes / 1e6, BBytes / 1e6,
    GBs]));
end;

procedure TimeShape(Kernel: TNeuralKernel; DotCL: TDotProductSharedKernel;
  const Shape: TGemmShape; Int4, BHalf: boolean);
var
  Codes: TBytes;
  Scales: TNeuralFloatDynArr;
  B: TNNetVolume;
  ResidentB: cl_mem;
  LaunchesBefore: integer;
  BName: string;
begin
  if HostBytes(Shape.Rows, Shape.Reduction, TokenCnt, Int4, BHalf) > csMaxHostBytes then
  begin
    WriteLn(Format('  %-6s %s: needs %d MB of host RAM (limit %d MB), skipped.',
      [Shape.Name, ModeName(Int4), HostBytes(Shape.Rows, Shape.Reduction,
      TokenCnt, Int4, BHalf) shr 20, csMaxHostBytes shr 20]));
    exit;
  end;
  FillWeights(Shape.Rows, Shape.Reduction, Int4, Codes, Scales);
  B := TNNetVolume.Create(Shape.Reduction * TokenCnt, 1, 1);
  ResidentB := nil;
  try
    B.Randomize();
    if not PrepareShape(DotCL, Shape.Rows, Shape.Reduction, Int4, BHalf, Codes,
      Scales, B) then
    begin
      WriteLn('  ', Shape.Name, ': arming the weights failed.');
      exit;
    end;
    // The upload copies are no longer needed.
    SetLength(Codes, 0);
    SetLength(Scales, 0);
    LaunchesBefore := DotCL.TiledGemmLaunchCount;
    // Uploads B (and in the FP16 mode narrows it into the half buffer, which
    // the NewB = false launches below then read as is).
    RunGemm(DotCL, B, Int4, true);
    Kernel.Finish();
    if DotCL.TiledGemmLaunchCount = LaunchesBefore then
      WriteLn('  ', Shape.Name, ': WARNING, the tiled GEMM did not run.');
    if BHalf then BName := 'fp16' else BName := 'fp32';
    TimeAndPrintRow(Kernel, DotCL, Shape, B, Int4, BHalf, nil, BName, 0);
    if BHalf then
    begin
      ResidentB := Kernel.CreateInputBuffer(B);
      if (not Assigned(ResidentB)) or
        (Kernel.WriteBuffer(ResidentB, B, CL_TRUE) <> CL_SUCCESS) then
        WriteLn('  ', Shape.Name, ': uploading the resident FP32 B failed.')
      else
        // The cast reads FP32 B and writes half B.
        TimeAndPrintRow(Kernel, DotCL, Shape, B, Int4, BHalf, ResidentB,
          'fp16+cast', int64(B.Size) * (SizeOf(TNeuralFloat) + 2));
    end;
  finally
    DotCL.UnprepareForCompute();
    if Assigned(ResidentB) then clReleaseMemObject(ResidentB);
    B.Free;
  end;
end;

procedure RunBenchmark();
var
  Kernel, Int8Kernel, FP16Kernel: TNeuralKernel;
  DotCL: TDotProductSharedKernel;
  Shapes: array of TGemmShape;
  Hidden, MlpHidden, ShapeIdx, ParityReduction: integer;
  Int4, BHalf, ParityOk: boolean;
  ParityPass: integer;

  // True when the (Int4, BHalf) variant was asked for and has a kernel.
  function VariantRuns(pInt4, pBHalf: boolean): boolean;
  begin
    if pInt4
      then Result := RunInt4 and RunBFP32 and not pBHalf
      else Result := RunInt8 and ((pBHalf and RunBFP16) or
        ((not pBHalf) and RunBFP32));
  end;

begin
  if Pico then
  begin
    Hidden := csPicoHidden;
    MlpHidden := csPicoMlpHidden;
  end
  else
  begin
    Hidden := csHidden;
    MlpHidden := csMlpHidden;
  end;
  if CustomRows > 0 then
  begin
    SetLength(Shapes, 1);
    Shapes[0].Name := 'Custom';
    Shapes[0].Rows := CustomRows;
    Shapes[0].Reduction := CustomReduction;
  end
  else
  begin
    SetLength(Shapes, 3);
    Shapes[0].Name := 'QKVO';
    Shapes[0].Rows := Hidden;
    Shapes[0].Reduction := Hidden;
    Shapes[1].Name := 'GateUp';
    Shapes[1].Rows := 2 * MlpHidden;
    Shapes[1].Reduction := Hidden;
    Shapes[2].Name := 'Down';
    Shapes[2].Rows := Hidden;
    Shapes[2].Reduction := MlpHidden;
  end;
  Kernel := TNeuralKernel.Create(PlatformId, DeviceId, 'cai_dot_product', true);
  if not Assigned(Kernel.Kernel) then
  begin
    WriteLn('Could not build neural.cl (run GemmBench from examples/GemmBench ' +
      'so ../../neural/neural.cl is found).');
    Halt(1);
  end;
  Int8Kernel := TNeuralKernel.CreateFromProgram(Kernel, 'cai_dot_product_int8');
  // The FP16-B entry point a --gpu-fp16 layer injects (TNNetPointwiseConvLinear).
  FP16Kernel := nil;
  if RunBFP16 then
  begin
    FP16Kernel := TNeuralKernel.CreateFromProgram(Kernel, 'cai_dot_product_int8_h');
    if not Assigned(FP16Kernel.Kernel) then
    begin
      WriteLn('cai_dot_product_int8_h was not built on this device: no fp16 rows.');
      FreeAndNil(FP16Kernel);
      RunBFP16 := false;
      if not RunBFP32 then Halt(1);
    end;
  end;
  if RunBFP16 and RunInt4 then
    if RunBFP32
      then WriteLn('int4 has no FP16-B kernel: int4 runs with FP32 B only.')
      else WriteLn('int4 has no FP16-B kernel: int4 skipped.');
  DotCL := TDotProductSharedKernel.Create(Kernel, Int8Kernel, FP16Kernel);
  DotCL.HideMessages();
  try
    WriteLn('Device: ', DeviceName, ', compute units ',
      Kernel.DeviceMaxComputeUnits(), ', local memory ',
      Kernel.DeviceLocalMemSize(), ' B');
    PrintKernelResources(Int8Kernel);
    RandSeed := 20261006;
    WriteLn('Parity vs Pascal:');
    ParityOk := true;
    // Large unsplit, block, large at csParitySplits K-splits.
    for ParityPass := 0 to 2 do
    begin
      if ParityPass = 1
        then SetTiledGemmCodesGrid(tgcBlock)
        else SetTiledGemmCodesGrid(tgcLarge);
      if ParityPass = 2
        then SetTiledGemmSplitK(csParitySplits)
        else SetTiledGemmSplitK(0);
      for Int4 := false to true do
        for BHalf := false to true do
          if VariantRuns(Int4, BHalf) then
          begin
            ParityReduction := csParityReduction;
            if Int4 then ParityReduction := ParityReduction - ParityReduction mod 32;
            ParityOk := ParityShape(DotCL, 516, ParityReduction, Int4, BHalf) and
              ParityOk;
            ParityOk := ParityShape(DotCL, 515, ParityReduction, Int4, BHalf) and
              ParityOk;
            DotCL.UnprepareForCompute();
          end;
    end;
    if not ParityOk then Halt(1);
    SetTiledGemmCodesGrid(Grid);
    SetTiledGemmSplitK(SplitK);
    WriteLn('Timing (A = codes + scales, B per launch, MB = 1e6 bytes; GB/s = ' +
      '(A + B + result [+ cast]) / time,');
    WriteLn('  a compulsory-traffic lower bound: L2 re-reads and split-K ' +
      'partials are not counted):');
    WriteLn(Format('  %-6s %s %-9s %-17s %-24s %3s %2s %9s %7s %7s %7s %6s',
      ['shape', 'mode', 'B', 'rows x K x tokens', 'kernel', 'lns', 'sp', 'ms',
      'TFLOPS', 'A MB', 'B MB', 'GB/s']));
    for Int4 := false to true do
      for ShapeIdx := 0 to High(Shapes) do
        for BHalf := false to true do
          if VariantRuns(Int4, BHalf) then
            TimeShape(Kernel, DotCL, Shapes[ShapeIdx], Int4, BHalf);
  finally
    DotCL.Free;
    FP16Kernel.Free;
    Int8Kernel.Free;
    Kernel.Free;
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
  WriteLn('GemmBench needs a build with -dOpenCL.');
  {$ENDIF}
end.
