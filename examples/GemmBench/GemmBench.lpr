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
nothing is timed.

Timing: the codes, scales and the B operand are uploaded once; each launch
reads them in OpenCL memory and leaves its result there. The figure is the
median of 3 timed blocks of the same launch count, at least --iters and grown
until a block lasts 200 ms (GetTickCount64, monotonic). FLOPs = 2 * rows *
reduction * tokens. Memory guard: a shape runs only if its host volumes fit
600 MB, counted with the OpenCL buffers on a CPU device.

Usage:
  GemmBench [--tokens 4096] [--iters 10] [--int8 | --int4] [--pico]
    [--rows R --reduction K] [--grid auto|large|block]
    [--gpu-platform 0] [--gpu-device 0]
--pico times hidden 64 / MLP 192 over 32 tokens (a smoke run). --rows and
--reduction time that one shape instead (e.g. an LLM projection at a prefill
window: --rows 1024 --reduction 2560 --tokens 64). --grid forces the large
(512 rows x 16 columns) or block (128 x 128) work-groups of the code kernels;
auto picks per shape, as TNNetPointwiseConvLinear does
(SetTiledGemmCodesGrid). Parity runs on both grids.

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
  Grid: TTiledGemmCodesGrid;
  RunInt8, RunInt4, Pico: boolean;
  PlatformId: cl_platform_id;
  DeviceId: cl_device_id;
  DeviceName: string;

procedure PrintUsageAndHalt(const Problem: string);
begin
  if Problem <> '' then WriteLn('Error: ', Problem);
  WriteLn('Usage: GemmBench [--tokens N] [--iters N] [--int8 | --int4] ' +
    '[--pico] [--rows R --reduction K] [--grid auto|large|block] ' +
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
    else PrintUsageAndHalt('unknown argument ' + Arg);
    Inc(ArgIdx);
  end;
  if not (RunInt8 or RunInt4) then
    PrintUsageAndHalt('--int8 and --int4 exclude each other');
  if TokenCnt = 0 then
  begin
    if Pico then TokenCnt := csPicoTokens else TokenCnt := 4096;
  end;
  if (TokenCnt < 1) or (Iters < 1) then PrintUsageAndHalt('sizes must be positive');
  if (CustomRows <> 0) or (CustomReduction <> 0) then
  begin
    if (CustomRows < 1) or (CustomReduction < 1) then
      PrintUsageAndHalt('--rows and --reduction go together and must be positive');
    if RunInt4 and (CustomReduction mod 32 <> 0) then
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

// Host volumes of a shape (B operand, weights), plus the OpenCL buffers
// (B, weights, result) on a CPU device.
function HostBytes(Rows, Reduction, Columns: integer; Int4: boolean): int64;
var
  BBytes: int64;
begin
  BBytes := int64(Reduction) * Columns * SizeOf(TNeuralFloat);
  Result := BBytes + WeightBytes(Rows, Reduction, Int4);
  if DeviceIsCPU() then
    Inc(Result, BBytes + WeightBytes(Rows, Reduction, Int4) +
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

// Arms DotCL with the weights of a shape and B for Columns columns.
function PrepareShape(DotCL: TDotProductSharedKernel; Rows, Reduction: integer;
  Int4: boolean; const Codes: TBytes; const Scales: TNeuralFloatDynArr;
  B: TNNetVolume): boolean;
begin
  if Int4
    then Result := DotCL.PrepareForComputeInt4(@Codes[0], @Scales[0], Rows,
      Reduction, B) = CL_SUCCESS
    else Result := DotCL.PrepareForComputeInt8(@Codes[0], @Scales[0], Rows,
      Reduction, B) = CL_SUCCESS;
end;

// One launch over the resident B (uploaded when NewB).
procedure RunGemm(DotCL: TDotProductSharedKernel; B: TNNetVolume;
  Int4, NewB: boolean);
begin
  if Int4
    then DotCL.ComputeInt4(B, {ActFN=}0, NewB)
    else DotCL.ComputeInt8(B, {ActFN=}0, NewB);
end;

function ModeName(Int4: boolean): string;
begin
  if Int4 then Result := 'int4' else Result := 'int8';
end;

// Max |OpenCL - Pascal| of one launch at Rows x Reduction x csParityColumns.
function ParityShape(DotCL: TDotProductSharedKernel; Rows, Reduction: integer;
  Int4: boolean): boolean;
var
  Codes: TBytes;
  Scales: TNeuralFloatDynArr;
  B, Y: TNNetVolume;
  a, k, Col, LaunchesBefore: integer;
  Sum, MaxDiff, MaxAbsY, Bound: TNeuralFloat;
  Ran: boolean;
begin
  FillWeights(Rows, Reduction, Int4, Codes, Scales);
  B := TNNetVolume.Create(Reduction * csParityColumns, 1, 1);
  Y := TNNetVolume.Create(Rows * csParityColumns, 1, 1);
  try
    B.Randomize();
    Result := PrepareShape(DotCL, Rows, Reduction, Int4, Codes, Scales, B);
    if not Result then exit;
    LaunchesBefore := DotCL.TiledGemmLaunchCount;
    RunGemm(DotCL, B, Int4, true);
    DotCL.FinishAndLoadResult(Y);
    Ran := DotCL.TiledGemmLaunchCount > LaunchesBefore;
    MaxDiff := 0;
    MaxAbsY := 0;
    for Col := 0 to csParityColumns - 1 do
      for a := 0 to Rows - 1 do
      begin
        Sum := 0;
        for k := 0 to Reduction - 1 do
          Sum := Sum + WeightAt(Codes, Scales, Rows, a, k, Int4) *
            B.FData[Col * Reduction + k];
        MaxDiff := Max(MaxDiff, Abs(Sum - Y.FData[Col * Rows + a]));
        MaxAbsY := Max(MaxAbsY, Abs(Sum));
      end;
    Bound := 1e-4 * Max(1, MaxAbsY);
    Result := Ran and (MaxDiff <= Bound);
    WriteLn(Format('  %s %d rows x %d x %d columns, %d lanes: tiled ran: %s, ' +
      'max|diff| = %.3e (bound %.1e) %s', [ModeName(Int4), Rows, Reduction,
      csParityColumns, DotCL.LastTiledGemmLanes, BoolToStr(Ran, 'yes', 'no'),
      MaxDiff, Bound,
      BoolToStr(Result, 'PASS', 'FAIL')]));
  finally
    Y.Free;
    B.Free;
  end;
end;

// Work-group size cap, local and private (spilled) bytes of each tiled code
// kernel, as the device compiler reports them for Kernel's program.
procedure PrintKernelResources(Kernel: TNeuralKernel);
const
  csCodeKernels: array[0..3] of string = ('cai_dot_product_int8_tiled',
    'cai_dot_product_int8_tiled_block', 'cai_dot_product_int4_tiled',
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
  B: TNNetVolume; Int4: boolean): double;
var
  BlockMs: array[0..csTimedBlocks - 1] of double;
  BlockIdx, BlockIters: integer;
  StartTick, Elapsed: QWord;

  function TimeBlock(pIters: integer): QWord;
  var
    LaunchIdx: integer;
  begin
    StartTick := GetTickCount64();
    for LaunchIdx := 1 to pIters do RunGemm(DotCL, B, Int4, false);
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

procedure TimeShape(Kernel: TNeuralKernel; DotCL: TDotProductSharedKernel;
  const Shape: TGemmShape; Int4: boolean);
var
  Codes: TBytes;
  Scales: TNeuralFloatDynArr;
  B: TNNetVolume;
  LaunchesBefore: integer;
  Ms, Tflops: double;
begin
  if HostBytes(Shape.Rows, Shape.Reduction, TokenCnt, Int4) > csMaxHostBytes then
  begin
    WriteLn(Format('  %-8s %s: needs %d MB of host RAM (limit %d MB), skipped.',
      [Shape.Name, ModeName(Int4), HostBytes(Shape.Rows, Shape.Reduction,
      TokenCnt, Int4) shr 20, csMaxHostBytes shr 20]));
    exit;
  end;
  FillWeights(Shape.Rows, Shape.Reduction, Int4, Codes, Scales);
  B := TNNetVolume.Create(Shape.Reduction * TokenCnt, 1, 1);
  try
    B.Randomize();
    if not PrepareShape(DotCL, Shape.Rows, Shape.Reduction, Int4, Codes,
      Scales, B) then
    begin
      WriteLn('  ', Shape.Name, ': arming the weights failed.');
      exit;
    end;
    // The upload copies are no longer needed.
    SetLength(Codes, 0);
    SetLength(Scales, 0);
    LaunchesBefore := DotCL.TiledGemmLaunchCount;
    RunGemm(DotCL, B, Int4, true);
    Kernel.Finish();
    if DotCL.TiledGemmLaunchCount = LaunchesBefore then
      WriteLn('  ', Shape.Name, ': WARNING, the tiled GEMM did not run.');
    Ms := TimeLaunches(Kernel, DotCL, B, Int4);
    if Ms > 0
      then Tflops := 2.0 * Shape.Rows * Shape.Reduction * TokenCnt / (Ms * 1e-3) / 1e12
      else Tflops := 0;
    WriteLn(Format('  %-8s %s %6d x %6d x %5d %4d lanes %10.3f %8.2f',
      [Shape.Name, ModeName(Int4), Shape.Rows, Shape.Reduction, TokenCnt,
      DotCL.LastTiledGemmLanes, Ms, Tflops]));
  finally
    DotCL.UnprepareForCompute();
    B.Free;
  end;
end;

procedure RunBenchmark();
var
  Kernel, Int8Kernel: TNeuralKernel;
  DotCL: TDotProductSharedKernel;
  Shapes: array of TGemmShape;
  Hidden, MlpHidden, ShapeIdx, ParityReduction: integer;
  Int4, ParityOk: boolean;
  ParityGrid: TTiledGemmCodesGrid;
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
  DotCL := TDotProductSharedKernel.Create(Kernel, Int8Kernel);
  DotCL.HideMessages();
  try
    WriteLn('Device: ', DeviceName, ', compute units ',
      Kernel.DeviceMaxComputeUnits(), ', local memory ',
      Kernel.DeviceLocalMemSize(), ' B');
    PrintKernelResources(Int8Kernel);
    RandSeed := 20261006;
    WriteLn('Parity vs Pascal:');
    ParityOk := true;
    for ParityGrid := tgcLarge to tgcBlock do
    begin
      SetTiledGemmCodesGrid(ParityGrid);
      for Int4 := false to true do
        if (Int4 and RunInt4) or ((not Int4) and RunInt8) then
        begin
          ParityReduction := csParityReduction;
          if Int4 then ParityReduction := ParityReduction - ParityReduction mod 32;
          ParityOk := ParityShape(DotCL, 516, ParityReduction, Int4) and ParityOk;
          ParityOk := ParityShape(DotCL, 515, ParityReduction, Int4) and ParityOk;
          DotCL.UnprepareForCompute();
        end;
    end;
    if not ParityOk then Halt(1);
    SetTiledGemmCodesGrid(Grid);
    WriteLn('Timing (rows x reduction x tokens, work-group lanes, ms per ' +
      'launch, TFLOPS):');
    for Int4 := false to true do
      if (Int4 and RunInt4) or ((not Int4) and RunInt8) then
        for ShapeIdx := 0 to High(Shapes) do
          TimeShape(Kernel, DotCL, Shapes[ShapeIdx], Int4);
  finally
    DotCL.Free;
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
