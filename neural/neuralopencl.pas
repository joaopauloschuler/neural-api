(*
neuralopencl
Copyright (C) 2017 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*)

unit neuralopencl;
// Coded and adapted by Joao Paulo Schwarz Schuler
// https://sourceforge.net/p/cai/

// This code was initially inspired on explamples found at:
// fpc\3.0.2\source\packages\opencl\examples

// This code was also inspired on:
// https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/opencl/trillion-test/uopencl_trillion_test.pas

// Delphi developers require these OpenCL headers:
// https://github.com/CWBudde/PasOpenCL

{$IFDEF FPC} {$mode objfpc}{$H+} {$ENDIF}

interface

uses
  Classes, SysUtils, cl, {$IFDEF FPC}ctypes{$ELSE}Winapi.Windows,AnsiStrings,CL_Platform{$ENDIF}, neuralvolume;

type
  {$IFDEF FPC}
  TNeuralStrBuffer = array[0..999] of Char;
  TNeuralPChar     = PChar;
  {$ELSE}
  TNeuralStrBuffer = array[0..999] of AnsiChar;
  TNeuralPChar     = PAnsiChar;
  csize_t          = NativeUInt;
  cl_bool          = TCL_bool;
  cl_int           = TCL_int;
  cl_uint          = TCL_uint;
  cl_platform_id   = PCL_platform_id;
  cl_device_id     = PCL_device_id;
  cl_context       = PCL_context;
  cl_command_queue = PCL_command_queue;
  cl_program       = PCL_program;
  cl_map_flags     = TCL_map_flags;
  cl_mem_flags     = TCL_mem_flags;
  cl_mem           = PCL_mem ;
  cl_kernel        = PCL_kernel;
  {$ENDIF}

const
  csCLMemSize = SizeOf(cl_mem);
  /// Bytes per OpenCL half. There is no Pascal type for it: the FP16 B
  /// operand is only ever written and read by device kernels.
  csHalfSize = 2;

type
  TPlatformNames = array of string;
  TPlatforms = array of cl_platform_id;
  TDeviceNames = array of string;
  TDevices = array of cl_device_id;

  { TEasyOpenCL }
  TEasyOpenCL = class(TMObject)
  private
    FPlatformNames: TPlatformNames;
    FPlatformIds: TPlatforms;
    FDeviceNames: TDeviceNames;
    FDevices: TDevices;
    FCurrentPlatform: cl_platform_id;
    FCurrentDevice: cl_device_id;
    // Cached CL_DEVICE_MAX_COMPUTE_UNITS of FCurrentDevice, 0 while not
    // yet queried. The int8 launch sizer asks for it per GEMM per token.
    FMaxComputeUnits: integer;
    FOpenCLProgramSource: TStringList;

    FContext: cl_context;        // OpenCL compute context
    FCommands: cl_command_queue; // OpenCL compute command queue
    FProg: cl_program;           // OpenCL compute program
    // When true the context/program above are BORROWED from another
    // TEasyOpenCL (the shared dot-product kernel) and must NOT be released by
    // this instance. Set by TNeuralKernel.CreateFromProgram so a helper kernel
    // can bind the already-compiled neural.cl program instead of recompiling it
    // per layer. (Coded by Claude (AI).)
    FBorrowedContext: boolean;
    // Same for the command queue, which is borrowed independently: a helper
    // kernel either shares the owner's queue or creates its own.
    FBorrowedQueue: boolean;
    {$IFDEF FPC}
    FCompilerOptions: string[255];
    {$ELSE}
    FCompilerOptions: ShortString;
    {$ENDIF}

    procedure LoadPlatforms();
    procedure FreeContext();
    procedure CompileProgram(); overload;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    procedure printDevicesInfo();
    function GetPlatformCount(): integer;
    function GetDeviceCount(): integer;
    procedure GetDevicesFromPlatform(PlatformId: cl_platform_id;
      out pDeviceNames: TDeviceNames; out pDevices: TDevices);

    procedure SetCurrentPlatform(pPlatformId: cl_platform_id);
    procedure SetCurrentDevice(pDeviceId: cl_device_id);

    procedure CompileProgramFromFile(filename:string); overload;
    procedure CompileProgram(programsource: TStrings); overload;
    procedure CompileProgram(programsource: string);   overload;

    function CreateContext(): cl_context;
    function CreateCommandQueue(): cl_command_queue;
    function CreateBuffer(flags: cl_mem_flags; size: csize_t; ptr: Pointer = nil): cl_mem; overload;
    function MapBuffer(buffer: cl_mem; cb: csize_t; map_flags: cl_map_flags; blocking: cl_bool = CL_TRUE): Pointer; overload;
    function MapHostInputBuffer(buffer: cl_mem; cb: csize_t): Pointer; overload;
    function UnmapMemObject(buffer: cl_mem; mapped_ptr: Pointer): cl_int;
    function RefreshHostInputBufferCache(buffer: cl_mem; cb: csize_t): cl_int;
    function WriteBuffer(buffer: cl_mem; cb: csize_t; ptr: Pointer; blocking: cl_bool = CL_FALSE): integer; overload;
    function ReadBuffer(buffer: cl_mem; cb: csize_t; ptr: Pointer; blocking: cl_bool = CL_TRUE): integer; overload;
    // Partial transfer: moves cb bytes into/out of the buffer starting at
    // offsetBytes, so a caller can refresh one live slice of a large
    // persistent buffer without moving the whole allocation.
    function WriteBufferAt(buffer: cl_mem; offsetBytes, cb: csize_t; ptr: Pointer; blocking: cl_bool = CL_FALSE): integer;
    function ReadBufferAt(buffer: cl_mem; offsetBytes, cb: csize_t; ptr: Pointer; blocking: cl_bool = CL_TRUE): integer;

    function CreateInputBuffer(size: csize_t): cl_mem; overload; {$IFDEF Release} inline; {$ENDIF}
    function CreateHostInputBuffer(size: csize_t; ptr: Pointer): cl_mem; overload; {$IFDEF Release} inline; {$ENDIF}
    function CreateOutputBuffer(size: csize_t): cl_mem; overload; {$IFDEF Release} inline; {$ENDIF}
    function CreateBuffer(size: csize_t): cl_mem;  overload; {$IFDEF Release} inline; {$ENDIF}

    function CreateKernel(kernelname: string): cl_kernel;
    // CL_DEVICE_MAX_COMPUTE_UNITS of the current device, 1 when the query
    // fails. Cached after the first call. Sizes launches that must fill the
    // device to run at speed.
    function DeviceMaxComputeUnits(): integer;
    function RunKernel(pkernel:cl_kernel; ThreadCount: integer): integer;
    function RunKernel2D(pkernel:cl_kernel; d1size, d2size: csize_t): integer; overload;
    function RunKernel2D(pkernel:cl_kernel; d1size, d2size, d1groupsize, d2groupsize: csize_t): integer; overload;
    function RunKernel3D(pkernel:cl_kernel; d1size, d2size, d3size: csize_t): integer; overload;
    function RunKernel3D(pkernel:cl_kernel; d1size, d2size, d3size, d1groupsize, d2groupsize, d3groupsize: csize_t): integer; overload;
    function Finish():integer;

    property PlatformNames: TPlatformNames read FPlatformNames;
    property PlatformIds: TPlatforms read FPlatformIds;
    property DeviceNames: TDeviceNames read FDeviceNames;
    property Devices: TDevices read FDevices;

    property CurrentPlatform: cl_platform_id read FCurrentPlatform;
    property CurrentDevice: cl_device_id read FCurrentDevice;

    property Context: cl_context read FContext;
    property Commands: cl_command_queue read FCommands;
    property Prog: cl_program read FProg;
    property CompilerOptions: ShortString read FCompilerOptions write FCompilerOptions;
  end;

  /// EasyOpenCL with TVolume support
  TEasyOpenCLV = class (TEasyOpenCL)
    public
      function CreateBuffer(flags: cl_mem_flags; V: TNNetVolume): cl_mem; overload; {$IFDEF Release} inline; {$ENDIF}
      function CreateInputBuffer(V: TNNetVolume): cl_mem; overload; {$IFDEF Release} inline; {$ENDIF}
      function CreateHostInputBuffer(V: TNNetVolume): cl_mem; overload; {$IFDEF Release} inline; {$ENDIF}
      function CreateOutputBuffer(V: TNNetVolume): cl_mem; overload; {$IFDEF Release} inline; {$ENDIF}
      function CreateBuffer(V: TNNetVolume): cl_mem;  overload; {$IFDEF Release} inline; {$ENDIF}

      function WriteBuffer(buffer: cl_mem; V: TNNetVolume; blocking: cl_bool = CL_FALSE): integer; overload; {$IFDEF Release} inline; {$ENDIF}
      function ReadBuffer(buffer: cl_mem; V: TNNetVolume; blocking: cl_bool = CL_TRUE): integer; overload; {$IFDEF Release} inline; {$ENDIF}

      function CreateAndWriteBuffer(V: TNNetVolume; var buffer: cl_mem): integer; overload; {$IFDEF Release} inline; {$ENDIF}
      function CreateAndWriteBuffer(V: TNNetVolume): cl_mem; overload; {$IFDEF Release} inline; {$ENDIF}
      function CreateWriteSetArgument(V: TNNetVolume; kernel:cl_kernel; arg_index: cl_uint): cl_mem; {$IFDEF Release} inline; {$ENDIF}
      function CreateOutputSetArgument(V: TNNetVolume; kernel:cl_kernel; arg_index: cl_uint): cl_mem; {$IFDEF Release} inline; {$ENDIF}

      // Grow-only persistent device buffer, the convolution buffer-reuse model
      // applied to the auxiliary helper kernels: pass the SAME (buf, capBytes)
      // pair every forward and the buffer is (re)allocated only when the needed
      // size exceeds the current capacity, otherwise reused in place. This
      // replaces the per-forward CreateBuffer/clReleaseMemObject churn that made
      // deep transformer models allocate dozens of tiny device buffers per pass.
      // The owning helper must release the buffer once in its destructor and
      // zero-init (buf=nil, capBytes=0). (Coded by Claude (AI).)
      function EnsureBuffer(var buf: cl_mem; var capBytes: csize_t;
        flags: cl_mem_flags; neededBytes: csize_t): cl_mem;
      // Ensure a persistent buffer big enough for V, then upload V into it.
      // DoWrite=false skips the upload (reuse the resident contents) - safe when
      // V is unchanged since the last write: a reallocation (first call or any
      // growth) uploads regardless, because the fresh handle holds nothing.
      function EnsureWriteBuffer(var buf: cl_mem; var capBytes: csize_t;
        V: TNNetVolume; DoWrite: boolean = true): cl_mem;
      // Ensure a persistent output buffer big enough for V (no upload).
      function EnsureOutputBuffer(var buf: cl_mem; var capBytes: csize_t;
        V: TNNetVolume): cl_mem;
  end;

  TNeuralKernel = class(TEasyOpenCLV)
    private
      /// OpenCL Kernel
      FKernel: cl_kernel;
      function PrepareKernel(kernelname: string = 'cai_dot_product'): integer;
      procedure UnprepareKernel();
    public
      constructor Create(pCurrentPlatform: cl_platform_id; pCurrentDevice: cl_device_id; kernelname: string = 'cai_dot_product'; pHideMessages: boolean = false);
      // Binds a kernel entry point against the ALREADY-COMPILED program of a
      // shared kernel (e.g. the net-wide dot-product kernel) instead of
      // recompiling neural.cl. The context and program are borrowed from
      // SharedKernel, and so is its command queue unless pSharedQueue is False;
      // borrowed handles are not released by this instance. Only the kernel
      // handle and the per-instance buffers are owned here. This is the
      // shared-program form of the auxiliary helper kernels (RoPE, softmax,
      // norms, gathers, ...). (Coded by Claude (AI).)
      constructor CreateFromProgram(SharedKernel: TEasyOpenCL;
        kernelname: string; pHideMessages: boolean = true;
        pSharedQueue: boolean = true);
      destructor Destroy(); override;

      property Kernel: cl_kernel read FKernel;
  end;

  // Do not use this class. It's under development
  TNNetVolumeCL = class(TNNetVolume)
    private
      // OpenCL Kernel
      FKernel: TNeuralKernel;
      // OpenCL Buffer
      FBufferCL: cl_mem;
    public
      procedure ReSize(pSizeX, pSizeY, pDepth: integer); override;
      procedure WriteToDevice(blocking: cl_bool = CL_FALSE);
      procedure ReadFromDevice(blocking: cl_bool = CL_TRUE);
      destructor Destroy(); override;

      property Kernel: TNeuralKernel read FKernel write FKernel;
  end;

  { TDotProductSharedKernel }
  TDotProductSharedKernel = class(TMObject)
    private
      /// Kernel Input Buffer A
      FInputBufferAs: cl_mem;
      /// Kernel Input Buffer B
      FInputBufferBs: cl_mem;
      /// Kernel Result Buffer
      FResultBuffer: cl_mem;
      /// Kernel parameters: number of vector As and Bs
      FNumAs, FNumBs: longint;
      /// Kernel parameters: size of vector As and vector Bs
      FSize: longint;
      /// Kernel parameter: activation function flag (1 means relu).
      FActFun: longint;
      /// Kernel parameter: number of OpenCL threads.
      FThreadCount: longint;
      /// OpenCL Group Sizes;
      FGroupSizeA, FGroupSizeB: longint;
      /// Average Previous Computing Time
      FPreviousComputeTime: TDateTime;
      /// Indicates if buffers should be stored on host.
      FHostInput: boolean;
      /// Byte capacities of the three device buffers above. Used by
      /// ReallocateBuffersIfRequired for grow-only reuse; kept in sync with the
      /// actual allocations (reset to 0 whenever a buffer is released).
      FCapAs, FCapBs, FCapResult: csize_t;
      /// Optional resident fused-bias buffer (arg 9 of cai_dot_product). Only
      /// allocated when a Compute call passes a bias volume; grow-only like the
      /// operands, uploaded only when NewVBias (or on first/grown allocation).
      /// nil (and UseBias=0) for every bias-less caller. Coded by Claude (AI).
      FBiasBuffer: cl_mem;
      FCapBias: csize_t;
      /// Optional resident source buffer for device-side im2col: holds the small
      /// (padded) convolution input that BuildInputColsOnDevice gathers into
      /// FInputBufferBs. Grow-only and re-uploaded only when the source changed;
      /// nil until the first inference-only non-pointwise conv forward that opts in.
      /// Coded by Claude (AI).
      FIm2ColSrcBuffer: cl_mem;
      FCapIm2ColSrc: csize_t;
      /// INT8 WEIGHT MODE (cai_dot_product_int8). The A operand is per-row
      /// symmetric int8 codes + per-row FP32 scales, both RESIDENT and
      /// IMMUTABLE (quantized layers are inference-only, so unlike the FP32
      /// weights they are uploaded exactly once, at PrepareForComputeInt8
      /// time, and never re-uploaded). FInt8Kernel is the net-wide
      /// cai_dot_product_int8 handle, INJECTED at Create time by the owning
      /// layer (which borrows it from TNNet) and NOT owned here - the net
      /// frees it. nil on every FP32-only instance. Buffers and enqueues ride
      /// the shared in-order queue via FDotProductKernel. Coded by Claude (AI).
      FInt8Kernel: TNeuralKernel;
      FDotProductKernel: TNeuralKernel;
      FCodesBuffer: cl_mem;
      FScalesBuffer: cl_mem;
      FCapCodes, FCapScales: csize_t;
      FInt8Ready: boolean;
      /// SPLIT-K INT8 MODE. cai_dot_product_int8 gives one work-item per
      /// (output row, sample), so a decode GEMV (FNumBs=1) launches only
      /// FNumAs work-items - far too few to fill a GPU. These two entry points
      /// are bound against FInt8Kernel's already-compiled program (so they ride
      /// the same queue) and split the reduction across a third grid axis:
      /// pass 1 writes raw slab sums into FPartialBuffer, pass 2 reduces them
      /// and applies scale/bias/activation. Owned here (raw cl_kernel handles),
      /// created lazily on the first split launch. Coded by Claude (AI).
      FSplitKKernel: cl_kernel;
      FSplitKReduceKernel: cl_kernel;
      FPartialBuffer: cl_mem;
      FCapPartial: csize_t;
      /// FP16 ACTIVATION MODE (cai_dot_product_int8_h and its split-K twin).
      /// The A operand is unchanged - int8 codes and FP32 scales; only B, the
      /// column matrix every one of the FNumAs output rows re-reads, narrows to
      /// half and lives in FInputBufferBsFP16 instead of FInputBufferBs. half is
      /// a STORAGE format here: the kernels read it with vload_half and
      /// accumulate in float, so the result buffer, the scales and the bias stay
      /// FP32 and the round trip to the CPU is Single-precision in both modes.
      /// The split-K partials narrow the same way (see FSplitKFP16Kernel).
      /// FFP16Kernel is net-owned and INJECTED at Create time like FInt8Kernel;
      /// the cl_kernel handles below are bound against its program and released
      /// here. FFP16Activations is the resolved verdict - PrepareForComputeInt8
      /// only sets it when the caller asked AND the kernel bound, so a device
      /// that rejected cai_dot_product_int8_h falls back to the FP32 B operand.
      /// Coded by Claude (AI).
      FFP16Kernel: TNeuralKernel;
      FFP16Activations: boolean;
      FInputBufferBsFP16: cl_mem;
      FCapBsFP16: csize_t;
      /// Both split-K passes have a half twin: pass 1 writes its raw slab sums
      /// as half, so pass 2 must read FPartialBuffer as half too. The buffer is
      /// sized in half bytes in that mode, which is why PrepareSplitK derives
      /// its element size from FFP16Activations. Coded by Claude (AI).
      FSplitKFP16Kernel: cl_kernel;
      FSplitKReduceFP16Kernel: cl_kernel;
      FCastFP16Kernel: cl_kernel;
      /// Last (shape, buffer, kernel) tuple bound into the split-K entry points
      /// by BindSplitKInvariantArgs. The split-K handles are per-instance, so
      /// eleven of the sixteen arguments survive from launch to launch and are
      /// re-set only when one of these changes. FSplitKArgsBound is cleared
      /// whenever a buffer this tuple names is released. Coded by Claude (AI).
      FSplitKArgsBound: boolean;
      FBoundSplitKKernel, FBoundSplitKReduceKernel: cl_kernel;
      FBoundSplits, FBoundNumAs, FBoundNumBs, FBoundSize: longint;
      FBoundPartialBuffer, FBoundCodesBuffer: cl_mem;
      FBoundResultBuffer, FBoundScalesBuffer: cl_mem;

      /// How many slabs to cut the reduction axis into for the current shape:
      /// 1 means the launch already fills the device, so ComputeInt8 keeps the
      /// single-pass kernel. Coded by Claude (AI).
      function Int8SplitCount(): integer;
      /// Binds the two split-K entry points and sizes FPartialBuffer for
      /// pSplits. False when the device rejected either kernel, which sends
      /// ComputeInt8 back to the single-pass path. Coded by Claude (AI).
      function PrepareSplitK(pSplits: integer): boolean;
      /// Sets the split-K arguments that do not change between launches of the
      /// same shape, skipping the eleven clSetKernelArg calls when the cached
      /// tuple still matches. Coded by Claude (AI).
      function BindSplitKInvariantArgs(pKernel, pReduceKernel: cl_kernel;
        pSplits: longint): integer;
      /// Narrows ElementCount floats of pSrcFP32 into FInputBufferBsFP16 with
      /// cai_f32_to_half, on the shared in-order queue. Coded by Claude (AI).
      function CastBOperandToFP16(pSrcFP32: cl_mem; ElementCount: longint): integer;
      /// The B operand ComputeInt8 binds. In FP16 mode this narrows whatever the
      /// caller supplied into FInputBufferBsFP16 first; NewVBs = false means the
      /// caller's cai_im2col_h already wrote that buffer, so nothing is done.
      /// Coded by Claude (AI).
      function PrepareInt8BOperand(VBs: TNNetVolume; NewVBs: boolean;
        pExternalVBs: cl_mem; var err: integer): cl_mem;


      function Kernel(): cl_kernel; {$IFDEF Release} inline; {$ENDIF}
    public
      constructor Create(DotProductKernel: TNeuralKernel;
        pInt8Kernel: TNeuralKernel = nil; pFP16Kernel: TNeuralKernel = nil);
      destructor Destroy(); override;

      procedure UnprepareForCompute();
      function PrepareForCompute(VAs, VBs: TNNetVolume; pSize: longint; GroupSizeA: integer=0; GroupSizeB: integer=0): integer;
      /// Grow-only sibling of PrepareForCompute: sets the same scalar shape state
      /// but only releases+recreates a device buffer when the needed byte size
      /// exceeds its current capacity, otherwise reuses it in place. Safe because
      /// Compute() re-uploads both operands every call, so stale buffer contents
      /// never leak. Use this from per-forward attention/gram ComputeOpenCL paths
      /// to avoid churning three clCreateBuffer/clReleaseMemObject pairs per GEMM.
      /// Coded by Claude (AI).
      procedure ReallocateBuffersIfRequired(VAs, VBs: TNNetVolume; pSize: longint; GroupSizeA: integer=0; GroupSizeB: integer=0);
      /// Device-side im2col: gathers SrcVol (the padded conv input) into the
      /// resident B-operand buffer (FInputBufferBs) using the shared cai_im2col
      /// kernel, so the host never builds nor uploads the inflated column matrix.
      /// Must be called AFTER the B buffer is sized (PrepareForCompute in
      /// EnableOpenCL) and BEFORE the matching Compute(..., NewVBs=false), on the
      /// same in-order command queue so the gather is ordered before the GEMM.
      /// Coded by Claude (AI).
      /// pExternalSrc BORROWS an already-resident gather source (a producing
      /// layer's output buffer) in place of FIm2ColSrcBuffer: nothing is
      /// uploaded and nothing is released here. SrcVol then only carries the
      /// shape. Coded by Claude (AI).
      procedure BuildInputColsOnDevice(Im2ColKernel: TNeuralKernel; SrcVol: TNNetVolume;
        OutSizeX, ColDepth, RowSpan, InSizeX, InDepth, Stride: longint; NewSrc: boolean = true;
        pExternalSrc: cl_mem = nil);
      /// pExternalVBs BORROWS a B operand that is already on the device (a
      /// producing layer's output buffer): it is bound instead of
      /// FInputBufferBs, never uploaded and never released here. VBs then only
      /// carries the shape. Coded by Claude (AI).
      procedure Compute(VAs, VBs: TNNetVolume; pActFN: longint; NewVAs:boolean = true; NewVBs:boolean = true; VBias: TNNetVolume = nil; NewVBias: boolean = true; pExternalVBs: cl_mem = nil);
      /// Arms the int8 weight mode: binds cai_dot_product_int8, uploads the
      /// interleaved codes (pCodes, NumAs*pSize bytes, layout
      /// codes[a + i*NumAs]) and per-row scales (pScales, NumAs floats) as
      /// resident immutable buffers, and sizes the B/result buffers for VBs.
      /// Blocking uploads: the caller's staging arrays may be freed on
      /// return. Coded by Claude (AI).
      function PrepareForComputeInt8(pCodes, pScales: Pointer;
        NumAs, pSize: longint; VBs: TNNetVolume;
        pFP16: boolean = false): integer;
      /// Int8 twin of Compute: same B upload, fused bias and activation
      /// semantics, but the A operand is the resident code buffer and the
      /// per-row scales ride as kernel arg 10 (deferred dequantization).
      /// Coded by Claude (AI).
      procedure ComputeInt8(VBs: TNNetVolume; pActFN: longint;
        NewVBs: boolean = true; VBias: TNNetVolume = nil;
        NewVBias: boolean = true; pExternalVBs: cl_mem = nil);
      procedure FinishAndLoadResult(Results: TNNetVolume; SaveCPU: TNeuralFloat = 0); overload;

      /// The underlying device kernel shared by this instance. Exposed so a layer
      /// can spin up a second TDotProductSharedKernel (e.g. a dedicated backward
      /// instance) bound to the same kernel without re-deriving it.
      property DotProductKernel: TNeuralKernel read FDotProductKernel;
      /// True after a successful PrepareForComputeInt8 (cleared by
      /// UnprepareForCompute). Layers gate their int8 device route on this,
      /// falling back to the fused CPU path when unarmed.
      property Int8Ready: boolean read FInt8Ready;
      /// The buffer Compute/ComputeInt8 leave their result in and
      /// FinishAndLoadResult reads back from. Exposed so a layer can bind it
      /// (device residency) instead of downloading. Still owned here: released
      /// by UnprepareForCompute, replaced when PrepareForCompute or
      /// ReallocateBuffersIfRequired resizes it. Read it per use, never cache.
      property ResultBuffer: cl_mem read FResultBuffer;
  end;

  /// Class that does dot products via OpenCL
  TDotProductCL = class (TNeuralKernel)
    private
      /// Kernel parameters: number of vector As and Bs
      FNumAs, FNumBs: longint;
      /// Kernel parameters: size of vector As and vector Bs
      FSize: longint;
      /// Kernel parameter: activation function flag (1 means relu).
      FActFun: longint;
      /// Kernel parameter: number of OpenCL threads.
      FThreadCount: longint;
      /// Average Previous Computing Time
      FPreviousComputeTime: TDateTime;
      /// Indicates if buffers should be stored on host.
      FHostInput: boolean;

      FInputBufferAs: cl_mem;
      FInputBufferBs: cl_mem;
      FResultBuffer: cl_mem;

      /// OpenCL Group Sizes;
      FGroupSizeA, FGroupSizeB: longint;
    public
      constructor Create(pCurrentPlatform: cl_platform_id; pCurrentDevice: cl_device_id; kernelname: string = 'cai_dot_product'; pHideMessages: boolean = false);
      destructor Destroy(); override;

      procedure UnprepareForCompute();
      function PrepareForCompute(VAs, VBs: TNNetVolume; pSize: longint; kernelname: string = 'cai_dot_product'; GroupSizeA: integer=0; GroupSizeB: integer=0): integer;

      procedure Compute(VAs, VBs: TNNetVolume; pActFN: longint);
      procedure FinishAndLoadResult(Results: TNNetVolume; SaveCPU: TNeuralFloat = 0); overload;
  end;

implementation
uses math;

const
  platform_str_info: array[1..5] of record
      id: dword;
      Name: PChar
    end
  =
    (
    (id: CL_PLATFORM_PROFILE; Name: 'PROFILE'),
    (id: CL_PLATFORM_VERSION; Name: 'VERSION'),
    (id: CL_PLATFORM_NAME; Name: 'NAME'),
    (id: CL_PLATFORM_VENDOR; Name: 'VENDOR'),
    (id: CL_PLATFORM_EXTENSIONS; Name: 'EXTENSIONS')
    );

  device_str_info: array[1..5] of record
      id: dword;
      Name: PChar
    end
  =
    (
    (id: CL_DEVICE_NAME; Name: 'DEVICE NAME'),
    (id: CL_DEVICE_VENDOR; Name: 'DEVICE VENDOR'),
    (id: CL_DEVICE_VERSION; Name: 'DEVICE VERSION'),
    (id: CL_DEVICE_PROFILE; Name: 'DEVICE PROFILE'),
    (id: CL_DEVICE_EXTENSIONS; Name: 'DEVICE EXTENSIONS')
    );

  device_word_info: array[1..10] of record
      id: dword;
      Name: PChar
    end
  =
    (
    (id: {$IFDEF FPC}CL_DEVICE_TYPE_INFO{$ELSE}CL_DEVICE_TYPE{$ENDIF}; Name: 'DEVICE TYPE'),
    (id: CL_DEVICE_MAX_WORK_GROUP_SIZE; Name: 'DEVICE MAX WORK GROUP SIZE'),
    (id: CL_DEVICE_MAX_COMPUTE_UNITS; Name: 'DEVICE MAX COMPUTE UNITS'),
    (id: CL_DEVICE_IMAGE3D_MAX_WIDTH; Name: 'DEVICE IMAGE3D MAX WIDTH'),
    (id: CL_DEVICE_IMAGE3D_MAX_HEIGHT; Name: 'DEVICE IMAGE3D MAX HEIGHT'),
    (id: CL_DEVICE_GLOBAL_MEM_SIZE; Name: 'DEVICE GLOBAL MEM SIZE'),
    (id: CL_DEVICE_LOCAL_MEM_SIZE; Name: 'DEVICE LOCAL MEM SIZE'),
    (id: CL_DEVICE_COMPILER_AVAILABLE; Name: 'DEVICE COMPILER AVAILABLE'),
    (id: CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE; Name: 'DEVICE MAX CONSTANT BUFFER SIZE'),
    (id: CL_DEVICE_MAX_CONSTANT_ARGS; Name: 'DEVICE MAX CONSTANT ARGS')
    );

{ TNNetVolumeCL }

procedure TNNetVolumeCL.ReSize(pSizeX, pSizeY, pDepth: integer);
begin
  inherited ReSize(pSizeX, pSizeY, pDepth);
  if Assigned(FBufferCL) then
  begin
    clReleaseMemObject(FBufferCL);
  end;

  if Assigned(FKernel) then
  begin
    FBufferCL := FKernel.CreateBuffer(Self);
  end;
end;

procedure TNNetVolumeCL.WriteToDevice(blocking: cl_bool);
begin
  FKernel.WriteBuffer(FBufferCL, Self, blocking);
end;

procedure TNNetVolumeCL.ReadFromDevice(blocking: cl_bool);
begin
  FKernel.ReadBuffer(FBufferCL, Self, blocking);
end;

destructor TNNetVolumeCL.Destroy();
begin
  if Assigned(FBufferCL) then
  begin
    clReleaseMemObject(FBufferCL);
  end;
  inherited Destroy();
end;

function TDotProductSharedKernel.Kernel(): cl_kernel;
begin
  Kernel := FDotProductKernel.Kernel;
end;

constructor TDotProductSharedKernel.Create(DotProductKernel: TNeuralKernel;
  pInt8Kernel: TNeuralKernel = nil; pFP16Kernel: TNeuralKernel = nil);
begin
  inherited Create();
  FDotProductKernel := DotProductKernel;
  FInt8Kernel := pInt8Kernel;
  FFP16Kernel := pFP16Kernel;
  FHostInput := False;
end;

destructor TDotProductSharedKernel.Destroy();
begin
  UnprepareForCompute();
  // FInt8Kernel and FFP16Kernel are net-owned shared handles - not freed here.
  inherited Destroy();
end;

procedure TDotProductSharedKernel.UnprepareForCompute();
begin
  if Assigned(FInputBufferAs) then clReleaseMemObject(FInputBufferAs);
  if Assigned(FInputBufferBs) then clReleaseMemObject(FInputBufferBs);
  if Assigned(FResultBuffer)  then clReleaseMemObject(FResultBuffer);
  if Assigned(FBiasBuffer)    then clReleaseMemObject(FBiasBuffer);
  if Assigned(FIm2ColSrcBuffer) then clReleaseMemObject(FIm2ColSrcBuffer);
  if Assigned(FCodesBuffer)   then clReleaseMemObject(FCodesBuffer);
  if Assigned(FScalesBuffer)  then clReleaseMemObject(FScalesBuffer);
  if Assigned(FPartialBuffer) then clReleaseMemObject(FPartialBuffer);
  if Assigned(FInputBufferBsFP16) then clReleaseMemObject(FInputBufferBsFP16);
  // Owned here, unlike FInt8Kernel and FFP16Kernel: these came from CreateKernel
  // against the net's program, so this instance releases them.
  if Assigned(FSplitKKernel)       then clReleaseKernel(FSplitKKernel);
  if Assigned(FSplitKReduceKernel) then clReleaseKernel(FSplitKReduceKernel);
  if Assigned(FSplitKFP16Kernel)   then clReleaseKernel(FSplitKFP16Kernel);
  if Assigned(FSplitKReduceFP16Kernel) then
    clReleaseKernel(FSplitKReduceFP16Kernel);
  if Assigned(FCastFP16Kernel)     then clReleaseKernel(FCastFP16Kernel);
  FPartialBuffer := nil;
  FInputBufferBsFP16 := nil;
  FSplitKKernel := nil;
  FSplitKReduceKernel := nil;
  FSplitKFP16Kernel := nil;
  FSplitKReduceFP16Kernel := nil;
  FCastFP16Kernel := nil;
  FCapPartial := 0;
  FCapBsFP16 := 0;
  FFP16Activations := false;

  FInputBufferAs := nil;
  FInputBufferBs := nil;
  FResultBuffer  := nil;
  FBiasBuffer    := nil;
  FIm2ColSrcBuffer := nil;
  FCodesBuffer   := nil;
  FScalesBuffer  := nil;

  FCapAs := 0;
  FCapBs := 0;
  FCapResult := 0;
  FCapBias := 0;
  FCapIm2ColSrc := 0;
  FCapCodes := 0;
  FCapScales := 0;
  FInt8Ready := false;
  FSplitKArgsBound := false;
end;

// Grow-only buffer management for the per-forward attention/gram callers. Unlike
// PrepareForCompute (which unconditionally frees + recreates all three buffers),
// this keeps a buffer whenever it is already big enough. The scalar shape state
// (FNumAs/FNumBs/FSize/FThreadCount/FGroupSize*) is still set on every call, so
// the two different-shaped GEMMs of a single attention forward remain correct.
// CreateHostInputBuffer binds a buffer to a specific host pointer, so it cannot
// be reused across volumes; the FHostInput path therefore still allocates fresh.
// Coded by Claude (AI).
procedure TDotProductSharedKernel.ReallocateBuffersIfRequired(VAs, VBs: TNNetVolume;
  pSize: longint; GroupSizeA: integer; GroupSizeB: integer);
var
  NeededAs, NeededBs, NeededResult: csize_t;
begin
  FNumAs := VAs.Size div pSize;
  FNumBs := VBs.Size div pSize;
  FThreadCount := FNumAs * FNumBs;
  FSize := pSize;
  FGroupSizeA := GroupSizeA;
  FGroupSizeB := GroupSizeB;

  NeededResult := FNumAs * FNumBs * csNeuralFloatSize;

  if (FHostInput) then
  begin
    // Host-pointer buffers cannot be reused across volumes: free + recreate.
    if Assigned(FInputBufferAs) then clReleaseMemObject(FInputBufferAs);
    if Assigned(FInputBufferBs) then clReleaseMemObject(FInputBufferBs);
    FInputBufferAs := FDotProductKernel.CreateHostInputBuffer(VAs);
    FInputBufferBs := FDotProductKernel.CreateHostInputBuffer(VBs);
    FCapAs := VAs.GetMemSize();
    FCapBs := VBs.GetMemSize();
  end
  else
  begin
    NeededAs := VAs.GetMemSize();
    NeededBs := VBs.GetMemSize();
    if (FInputBufferAs = nil) or (NeededAs > FCapAs) then
    begin
      if Assigned(FInputBufferAs) then clReleaseMemObject(FInputBufferAs);
      FInputBufferAs := FDotProductKernel.CreateInputBuffer(NeededAs);
      FCapAs := NeededAs;
    end;
    if (FInputBufferBs = nil) or (NeededBs > FCapBs) then
    begin
      if Assigned(FInputBufferBs) then clReleaseMemObject(FInputBufferBs);
      FInputBufferBs := FDotProductKernel.CreateInputBuffer(NeededBs);
      FCapBs := NeededBs;
    end;
  end;

  if (FResultBuffer = nil) or (NeededResult > FCapResult) then
  begin
    if Assigned(FResultBuffer) then clReleaseMemObject(FResultBuffer);
    FResultBuffer := FDotProductKernel.CreateOutputBuffer(NeededResult);
    FCapResult := NeededResult;
  end;

  FPreviousComputeTime := 0;
end;

function TDotProductSharedKernel.PrepareForCompute(VAs, VBs: TNNetVolume;
  pSize: longint; GroupSizeA: integer; GroupSizeB: integer): integer;
begin
  UnprepareForCompute();
  FNumAs := VAs.Size div pSize;
  FNumBs := VBs.Size div pSize;
  FThreadCount := FNumAs * FNumBs;
  FSize := pSize;
  FGroupSizeA := GroupSizeA;
  FGroupSizeB := GroupSizeB;

  if (FHostInput) then
  begin
    FInputBufferAs := FDotProductKernel.CreateHostInputBuffer(VAs);
    FInputBufferBs := FDotProductKernel.CreateHostInputBuffer(VBs);
  end
  else
  begin
    FInputBufferAs := FDotProductKernel.CreateInputBuffer(VAs);
    FInputBufferBs := FDotProductKernel.CreateInputBuffer(VBs);
  end;

  FResultBuffer  := FDotProductKernel.CreateOutputBuffer(FNumAs * FNumBs * csNeuralFloatSize);
  FPreviousComputeTime := 0;

  PrepareForCompute := CL_SUCCESS;
end;

procedure TDotProductSharedKernel.BuildInputColsOnDevice(Im2ColKernel: TNeuralKernel;
  SrcVol: TNNetVolume; OutSizeX, ColDepth, RowSpan, InSizeX, InDepth, Stride: longint;
  NewSrc: boolean = true; pExternalSrc: cl_mem = nil);
var
  k: cl_kernel;
  N: longint;
  err: integer;
  NeededSrc: csize_t;
  SrcBuffer, ColsBuffer: cl_mem;
begin
  k := Im2ColKernel.Kernel;
  // The caller must pass cai_im2col_h whenever this instance is in FP16 mode:
  // the gather writes whichever B buffer the GEMM will read, and both verdicts
  // come from the one layer flag so they cannot disagree.
  if FFP16Activations
    then ColsBuffer := FInputBufferBsFP16
    else ColsBuffer := FInputBufferBs;
  // Total column-matrix elements = FInputBufferBs capacity (already sized to
  // FInputPrepared by PrepareForCompute). FNumBs*FSize == FInputPrepared.Size.
  N := FNumBs * FSize;

  err := CL_SUCCESS;
  if pExternalSrc <> nil then
  begin
    SrcBuffer := pExternalSrc;
  end
  else
  begin
    // Resident, grow-only source buffer (same model as the operand/bias buffers).
    NeededSrc := SrcVol.GetMemSize();
    if (FIm2ColSrcBuffer = nil) or (NeededSrc > FCapIm2ColSrc) then
    begin
      if Assigned(FIm2ColSrcBuffer) then clReleaseMemObject(FIm2ColSrcBuffer);
      FIm2ColSrcBuffer := FDotProductKernel.CreateInputBuffer(NeededSrc);
      FCapIm2ColSrc := NeededSrc;
      NewSrc := true; // fresh/grown buffer: force upload regardless of caller
    end;
    SrcBuffer := FIm2ColSrcBuffer;
    if NewSrc then err := FDotProductKernel.WriteBuffer(FIm2ColSrcBuffer, SrcVol);
  end;

  err := err or clSetKernelArg(k, 0, csLongintSize, @N);
  err := err or clSetKernelArg(k, 1, csLongintSize, @OutSizeX);
  err := err or clSetKernelArg(k, 2, csLongintSize, @ColDepth);
  err := err or clSetKernelArg(k, 3, csLongintSize, @RowSpan);
  err := err or clSetKernelArg(k, 4, csLongintSize, @InSizeX);
  err := err or clSetKernelArg(k, 5, csLongintSize, @InDepth);
  err := err or clSetKernelArg(k, 6, csLongintSize, @Stride);
  err := err or clSetKernelArg(k, 7, csCLMemSize, @SrcBuffer);
  err := err or clSetKernelArg(k, 8, csCLMemSize, @ColsBuffer);
  if (err <> CL_SUCCESS) then
    ErrorProc('Error: BuildInputColsOnDevice - failed setting parameters: ' + IntToStr(err));

  // Enqueue the gather on the DOT-PRODUCT kernel's queue rather than on whatever
  // queue the im2col kernel holds. The source upload above, this gather and the
  // Compute GEMM that reads FInputBufferBs are only ordered while all three ride
  // one in-order queue; queues are unordered with respect to each other, so this
  // stays correct even when the im2col kernel was built with its own queue
  // (CreateFromProgram's pSharedQueue = False). No event or Finish is needed, and
  // the cross-kernel enqueue is legal because both kernels share the same context
  // (ComputeInt8 runs the int8 kernel on this queue for the same reason).
  FDotProductKernel.RunKernel(k, N);
end;

procedure TDotProductSharedKernel.Compute
(
  VAs, VBs: TNNetVolume;
  pActFN: longint;
  NewVAs:boolean = true; NewVBs:boolean = true;
  VBias: TNNetVolume = nil; NewVBias: boolean = true;
  pExternalVBs: cl_mem = nil
);
var
  err: integer;
  UseBias: longint;
  NeededBias: csize_t;
  BufferBs: cl_mem;
  K: cl_kernel;
begin
  K := Kernel();
  FActFun := pActFN;
  if pExternalVBs <> nil then BufferBs := pExternalVBs else BufferBs := FInputBufferBs;

  if (VAs.Size = FSize * FNumAs) then
  begin
    if (VBs.Size = FSize * FNumBs) then
    begin
      err := clSetKernelArg(K, 0, csLongintSize, @FThreadCount);
      if (err <> CL_SUCCESS) then ErrorProc('0 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(K, 1, csLongintSize, @FNumAs);
      if (err <> CL_SUCCESS) then ErrorProc('1 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(K, 2, csLongintSize, @FNumBs);
      if (err <> CL_SUCCESS) then ErrorProc('2 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(K, 3, csLongintSize, @FSize);
      if (err <> CL_SUCCESS) then ErrorProc('3 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(K, 4, csLongintSize, @FActFun);
      if (err <> CL_SUCCESS) then ErrorProc('4 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(K, 5, csCLMemSize,  @FInputBufferAs);
      if (err <> CL_SUCCESS) then ErrorProc('5 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(K, 6, csCLMemSize,  @BufferBs);
      if (err <> CL_SUCCESS) then ErrorProc('6 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(K, 7, csCLMemSize,  @FResultBuffer);
      if (err <> CL_SUCCESS) then ErrorProc('7 Error: Failed to set kernel arguments:' + IntToStr(err));

      // Fused bias (arg 8 UseBias, arg 9 FBiasBuffer). Both args MUST be set every
      // call: cai_dot_product now has 10 args and the enqueue rejects any unset
      // one. A bias-less caller (VBias=nil) passes UseBias=0 and a NULL buffer
      // (legal - the kernel never reads it). With a bias volume, keep it resident
      // grow-only and re-upload only when NewVBias or the buffer was just
      // (re)allocated. Coded by Claude (AI).
      if VBias <> nil then
      begin
        NeededBias := VBias.GetMemSize();
        if (FBiasBuffer = nil) or (NeededBias > FCapBias) then
        begin
          if Assigned(FBiasBuffer) then clReleaseMemObject(FBiasBuffer);
          FBiasBuffer := FDotProductKernel.CreateInputBuffer(NeededBias);
          FCapBias := NeededBias;
          NewVBias := true; // fresh/grown buffer: force upload regardless of caller
        end;
        if NewVBias then err := err or FDotProductKernel.WriteBuffer(FBiasBuffer, VBias);
        UseBias := 1;
      end
      else
        UseBias := 0;

      err := err or clSetKernelArg(K, 8, csLongintSize, @UseBias);
      if (err <> CL_SUCCESS) then ErrorProc('8 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(K, 9, csCLMemSize, @FBiasBuffer);
      if (err <> CL_SUCCESS) then ErrorProc('9 Error: Failed to set kernel arguments:' + IntToStr(err));

      if (FHostInput) then
      begin
        //TODO: Fix this refresh.
        //if NewVAs then err := err or FDotProductKernel.RefreshHostInputBufferCache(FInputBufferAs, VAs.GetMemSize());
        //if NewVBs then err := err or FDotProductKernel.RefreshHostInputBufferCache(FInputBufferBs, VBs.GetMemSize())
        if NewVAs then err := err or FDotProductKernel.WriteBuffer(FInputBufferAs, VAs);
        if NewVBs and (pExternalVBs = nil) then err := err or FDotProductKernel.WriteBuffer(FInputBufferBs, VBs);
      end
      else
      begin
        if NewVAs then err := err or FDotProductKernel.WriteBuffer(FInputBufferAs, VAs);
        if NewVBs and (pExternalVBs = nil) then err := err or FDotProductKernel.WriteBuffer(FInputBufferBs, VBs);
      end;

      if (err <> CL_SUCCESS) then ErrorProc('Failed at WriteBuffer(input):' + IntToStr(err));

      if err = CL_SUCCESS then
      begin

        if (FGroupSizeA > 0) and (FGroupSizeB > 0)  then
        begin
          FDotProductKernel.RunKernel2D(Kernel, FNumAs, FNumBs, FGroupSizeA, FGroupSizeB);
        end
        else
        begin
          FDotProductKernel.RunKernel2D(Kernel, FNumAs, FNumBs);
        end;

      end
      else
      begin
        ErrorProc
        (
          'Error: TDotProductCL.Compute - ' +
          ' Failed setting parameters: ' + IntToStr(err)
        );
      end;
    end
    else
    begin
      ErrorProc
      (
        'Error: TDotProductCL.Compute - VB size: ' +
        IntToStr(VAs.Size) +
        ' FSize: ' + IntToStr(FSize) +
        ' NumBs:' + IntToStr(FNumBs)
      );
    end;
  end
  else
  begin
    ErrorProc
    (
      'Error: TDotProductCL.Compute - VA size: ' +
      IntToStr(VAs.Size) +
      ' FSize: ' + IntToStr(FSize) +
      ' NumAs:' + IntToStr(FNumAs)
    );
  end;
end;

function TDotProductSharedKernel.PrepareForComputeInt8(pCodes, pScales: Pointer;
  NumAs, pSize: longint; VBs: TNNetVolume; pFP16: boolean = false): integer;
var
  NeededCodes, NeededScales, NeededResult: csize_t;
  err: integer;
begin
  UnprepareForCompute();
  if not Assigned(FInt8Kernel) then
  begin
    ErrorProc('Error: PrepareForComputeInt8 - this TDotProductSharedKernel was ' +
      'built without a cai_dot_product_int8 handle.');
    PrepareForComputeInt8 := CL_INVALID_KERNEL;
    exit;
  end;
  FNumAs := NumAs;
  FNumBs := VBs.Size div pSize;
  FThreadCount := FNumAs * FNumBs;
  FSize := pSize;
  FGroupSizeA := 0;
  FGroupSizeB := 0;

  // Asked for AND available: a device that rejected cai_dot_product_int8_h
  // keeps the FP32 B operand rather than failing the layer.
  FFP16Activations := pFP16 and Assigned(FFP16Kernel) and
    Assigned(FFP16Kernel.Kernel);

  NeededCodes := FNumAs * FSize; // 1 byte per code
  NeededScales := FNumAs * csNeuralFloatSize;
  NeededResult := FNumAs * FNumBs * csNeuralFloatSize;

  FCodesBuffer := FDotProductKernel.CreateInputBuffer(NeededCodes);
  FScalesBuffer := FDotProductKernel.CreateInputBuffer(NeededScales);
  FResultBuffer := FDotProductKernel.CreateOutputBuffer(NeededResult);
  FCapCodes := NeededCodes;
  FCapScales := NeededScales;
  FCapResult := NeededResult;

  if FFP16Activations then
  begin
    // READ_WRITE, not CreateInputBuffer: cai_im2col_h and cai_f32_to_half write
    // this buffer and cai_dot_product_int8_h reads it.
    FCapBsFP16 := csize_t(FNumBs) * FSize * csHalfSize;
    FInputBufferBsFP16 := FDotProductKernel.CreateBuffer(FCapBsFP16);
    // The FP32 staging copy exists only on the host-upload path, so
    // PrepareInt8BOperand allocates it there instead of here.
    FInputBufferBs := nil;
    FCapBs := 0;
  end
  else
  begin
    FInputBufferBs := FDotProductKernel.CreateInputBuffer(VBs);
    FCapBs := VBs.GetMemSize();
  end;

  // One-time blocking uploads: the codes/scales never change afterwards
  // (quantized layers are inference-only) and the caller's staging arrays
  // may be freed as soon as this returns.
  err := FDotProductKernel.WriteBuffer(FCodesBuffer, NeededCodes, pCodes, CL_TRUE);
  err := err or FDotProductKernel.WriteBuffer(FScalesBuffer, NeededScales,
    pScales, CL_TRUE);
  if (err <> CL_SUCCESS) then
    ErrorProc('Error: PrepareForComputeInt8 - failed uploading codes/scales: '
      + IntToStr(err));

  FPreviousComputeTime := 0;
  FInt8Ready := (err = CL_SUCCESS);
  PrepareForComputeInt8 := err;
end;

const
  /// Work-items per compute unit the int8 launch aims for before it stops
  /// splitting. Enough to cover memory latency without cutting rows so thin
  /// that the reduce pass and the extra round trip dominate.
  csInt8SplitKThreadsPerUnit = 1024;
  /// Never cut a row into slabs shorter than this: below it the per-work-item
  /// setup outweighs the reduction work.
  csInt8SplitKMinSlab = 128;
  csInt8SplitKMaxSplits = 64;

function TDotProductSharedKernel.Int8SplitCount(): integer;
var
  Rows, TargetThreads, MaxSplitsBySize: integer;
begin
  Result := 1;
  Rows := FNumAs * FNumBs;
  if (Rows < 1) or (FSize < 1) then exit;
  TargetThreads := FDotProductKernel.DeviceMaxComputeUnits() *
    csInt8SplitKThreadsPerUnit;
  // Already fills the device (prefill, or a vocab-sized head): one pass wins,
  // because splitting would only add a partial buffer and a second launch.
  if Rows >= TargetThreads then exit;
  MaxSplitsBySize := FSize div csInt8SplitKMinSlab;
  if MaxSplitsBySize < 2 then exit;
  Result := (TargetThreads + Rows - 1) div Rows;
  if Result > MaxSplitsBySize then Result := MaxSplitsBySize;
  if Result > csInt8SplitKMaxSplits then Result := csInt8SplitKMaxSplits;
end;

function TDotProductSharedKernel.PrepareSplitK(pSplits: integer): boolean;
var
  NeededPartial, PartialElementSize: csize_t;
begin
  Result := false;
  if not Assigned(FInt8Kernel) then exit;
  // Both passes have an FP16 twin: pass 1 stores its raw slab sums as half, so
  // pass 2 must read them as half and FPartialBuffer is sized in half bytes.
  if FFP16Activations then
  begin
    if not Assigned(FSplitKFP16Kernel) then
      FSplitKFP16Kernel :=
        FFP16Kernel.CreateKernel('cai_dot_product_int8_splitk_h');
    if not Assigned(FSplitKFP16Kernel) then exit;
    if not Assigned(FSplitKReduceFP16Kernel) then
      FSplitKReduceFP16Kernel :=
        FFP16Kernel.CreateKernel('cai_dot_product_int8_splitk_reduce_h');
    if not Assigned(FSplitKReduceFP16Kernel) then exit;
    PartialElementSize := csHalfSize;
  end
  else
  begin
    if not Assigned(FSplitKKernel) then
      FSplitKKernel := FInt8Kernel.CreateKernel('cai_dot_product_int8_splitk');
    if not Assigned(FSplitKKernel) then exit;
    if not Assigned(FSplitKReduceKernel) then
      FSplitKReduceKernel :=
        FInt8Kernel.CreateKernel('cai_dot_product_int8_splitk_reduce');
    if not Assigned(FSplitKReduceKernel) then exit;
    PartialElementSize := SizeOf(TNeuralFloat);
  end;

  NeededPartial := csize_t(FNumAs) * FNumBs * pSplits * PartialElementSize;
  if (FPartialBuffer = nil) or (NeededPartial > FCapPartial) then
  begin
    if Assigned(FPartialBuffer) then clReleaseMemObject(FPartialBuffer);
    FPartialBuffer := FDotProductKernel.CreateBuffer(NeededPartial);
    FCapPartial := NeededPartial;
  end;
  Result := Assigned(FPartialBuffer);
end;

function TDotProductSharedKernel.CastBOperandToFP16(pSrcFP32: cl_mem;
  ElementCount: longint): integer;
var
  k: cl_kernel;
begin
  if not Assigned(FCastFP16Kernel) then
    FCastFP16Kernel := FFP16Kernel.CreateKernel('cai_f32_to_half');
  k := FCastFP16Kernel;
  if not Assigned(k) then
  begin
    ErrorProc('Error: CastBOperandToFP16 - cai_f32_to_half is not available.');
    Result := CL_INVALID_KERNEL;
    exit;
  end;
  Result := clSetKernelArg(k, 0, csLongintSize, @ElementCount);
  Result := Result or clSetKernelArg(k, 1, csCLMemSize, @pSrcFP32);
  Result := Result or clSetKernelArg(k, 2, csCLMemSize, @FInputBufferBsFP16);
  // Same in-order queue as the GEMM that reads it, so no wait is needed.
  if Result = CL_SUCCESS then FDotProductKernel.RunKernel(k, ElementCount);
end;

function TDotProductSharedKernel.PrepareInt8BOperand(VBs: TNNetVolume;
  NewVBs: boolean; pExternalVBs: cl_mem; var err: integer): cl_mem;
var
  NeededBs: csize_t;
begin
  if not FFP16Activations then
  begin
    if pExternalVBs <> nil then Result := pExternalVBs
    else
    begin
      Result := FInputBufferBs;
      if NewVBs then err := err or FDotProductKernel.WriteBuffer(FInputBufferBs, VBs);
    end;
    exit;
  end;

  Result := FInputBufferBsFP16;
  // Nothing borrowed and nothing new: cai_im2col_h already wrote the half
  // buffer during this same forward.
  if (pExternalVBs = nil) and (not NewVBs) then exit;

  if pExternalVBs <> nil then
  begin
    // A borrowed resident source is FP32 - every other consumer of a producing
    // layer's output reads it as FP32 - so it is narrowed rather than bound.
    err := err or CastBOperandToFP16(pExternalVBs, VBs.Size);
    exit;
  end;

  // Host-supplied B: upload FP32, then narrow on the device. This staging copy
  // is reached only here, which is why PrepareForComputeInt8 does not size it.
  NeededBs := VBs.GetMemSize();
  if (FInputBufferBs = nil) or (NeededBs > FCapBs) then
  begin
    if Assigned(FInputBufferBs) then clReleaseMemObject(FInputBufferBs);
    FInputBufferBs := FDotProductKernel.CreateInputBuffer(NeededBs);
    FCapBs := NeededBs;
  end;
  err := err or FDotProductKernel.WriteBuffer(FInputBufferBs, VBs);
  err := err or CastBOperandToFP16(FInputBufferBs, VBs.Size);
end;

function TDotProductSharedKernel.BindSplitKInvariantArgs(pKernel,
  pReduceKernel: cl_kernel; pSplits: longint): integer;
begin
  Result := CL_SUCCESS;
  if FSplitKArgsBound and (pKernel = FBoundSplitKKernel) and
    (pReduceKernel = FBoundSplitKReduceKernel) and (pSplits = FBoundSplits) and
    (FNumAs = FBoundNumAs) and (FNumBs = FBoundNumBs) and (FSize = FBoundSize) and
    (FPartialBuffer = FBoundPartialBuffer) and (FCodesBuffer = FBoundCodesBuffer) and
    (FResultBuffer = FBoundResultBuffer) and (FScalesBuffer = FBoundScalesBuffer)
  then exit;

  FSplitKArgsBound := false;
  Result := clSetKernelArg(pKernel, 0, csLongintSize, @FNumAs);
  Result := Result or clSetKernelArg(pKernel, 1, csLongintSize, @FNumBs);
  Result := Result or clSetKernelArg(pKernel, 2, csLongintSize, @FSize);
  Result := Result or clSetKernelArg(pKernel, 3, csLongintSize, @pSplits);
  Result := Result or clSetKernelArg(pKernel, 4, csCLMemSize, @FCodesBuffer);
  Result := Result or clSetKernelArg(pKernel, 6, csCLMemSize, @FPartialBuffer);

  Result := Result or clSetKernelArg(pReduceKernel, 0, csLongintSize, @FNumAs);
  Result := Result or clSetKernelArg(pReduceKernel, 1, csLongintSize, @FNumBs);
  Result := Result or clSetKernelArg(pReduceKernel, 2, csLongintSize, @pSplits);
  Result := Result or clSetKernelArg(pReduceKernel, 4, csCLMemSize, @FPartialBuffer);
  Result := Result or clSetKernelArg(pReduceKernel, 5, csCLMemSize, @FResultBuffer);
  Result := Result or clSetKernelArg(pReduceKernel, 8, csCLMemSize, @FScalesBuffer);
  if Result <> CL_SUCCESS then exit;

  FBoundSplitKKernel := pKernel;
  FBoundSplitKReduceKernel := pReduceKernel;
  FBoundSplits := pSplits;
  FBoundNumAs := FNumAs;
  FBoundNumBs := FNumBs;
  FBoundSize := FSize;
  FBoundPartialBuffer := FPartialBuffer;
  FBoundCodesBuffer := FCodesBuffer;
  FBoundResultBuffer := FResultBuffer;
  FBoundScalesBuffer := FScalesBuffer;
  FSplitKArgsBound := true;
end;

procedure TDotProductSharedKernel.ComputeInt8(VBs: TNNetVolume;
  pActFN: longint; NewVBs: boolean = true; VBias: TNNetVolume = nil;
  NewVBias: boolean = true; pExternalVBs: cl_mem = nil);
var
  err: integer;
  UseBias: longint;
  NeededBias: csize_t;
  K, KReduce: cl_kernel;
  BufferBs: cl_mem;
  Splits: longint;
begin
  if not FInt8Ready then
  begin
    ErrorProc('Error: TDotProductSharedKernel.ComputeInt8 without ' +
      'PrepareForComputeInt8.');
    exit;
  end;
  if (VBs.Size <> FSize * FNumBs) then
  begin
    ErrorProc('Error: TDotProductSharedKernel.ComputeInt8 - VB size: ' +
      IntToStr(VBs.Size) + ' FSize: ' + IntToStr(FSize) +
      ' NumBs:' + IntToStr(FNumBs));
    exit;
  end;
  FActFun := pActFN;

  // Fused bias: same contract as Compute (a bias-less caller passes UseBias=0
  // and a NULL buffer the kernel never reads). Resident grow-only buffer,
  // re-uploaded only when NewVBias or just (re)allocated.
  err := CL_SUCCESS;
  if VBias <> nil then
  begin
    NeededBias := VBias.GetMemSize();
    if (FBiasBuffer = nil) or (NeededBias > FCapBias) then
    begin
      if Assigned(FBiasBuffer) then clReleaseMemObject(FBiasBuffer);
      FBiasBuffer := FDotProductKernel.CreateInputBuffer(NeededBias);
      FCapBias := NeededBias;
      NewVBias := true; // fresh/grown buffer: force upload regardless of caller
    end;
    if NewVBias then err := err or FDotProductKernel.WriteBuffer(FBiasBuffer, VBias);
    UseBias := 1;
  end
  else
    UseBias := 0;

  // Binds (FP32) or narrows into (FP16) the B operand the GEMM below reads.
  BufferBs := PrepareInt8BOperand(VBs, NewVBs, pExternalVBs, err);

  Splits := Int8SplitCount();
  if (Splits > 1) and PrepareSplitK(Splits) then
  begin
    // Pass 1: raw slab sums. Pass 2: reduce, scale, bias, activation. Both on
    // the same in-order queue, so pass 2 is ordered after pass 1 with no wait.
    if FFP16Activations then K := FSplitKFP16Kernel else K := FSplitKKernel;
    if FFP16Activations
      then KReduce := FSplitKReduceFP16Kernel
      else KReduce := FSplitKReduceKernel;
    // Shape, codes, scales, partial and result stay bound across launches of
    // the same shape; only the four arguments below change per call.
    err := err or BindSplitKInvariantArgs(K, KReduce, Splits);
    err := err or clSetKernelArg(K, 5, csCLMemSize, @BufferBs);
    err := err or clSetKernelArg(KReduce, 3, csLongintSize, @FActFun);
    err := err or clSetKernelArg(KReduce, 6, csLongintSize, @UseBias);
    err := err or clSetKernelArg(KReduce, 7, csCLMemSize, @FBiasBuffer);

    if err = CL_SUCCESS then
    begin
      FDotProductKernel.RunKernel3D(K, FNumAs, FNumBs, Splits);
      FDotProductKernel.RunKernel2D(KReduce, FNumAs, FNumBs);
    end
    else
    begin
      ErrorProc('Error: TDotProductSharedKernel.ComputeInt8 - ' +
        'failed setting split-K parameters: ' + IntToStr(err));
    end;
    exit;
  end;

  if FFP16Activations then K := FFP16Kernel.Kernel else K := FInt8Kernel.Kernel;
  err := err or clSetKernelArg(K, 0, csLongintSize, @FThreadCount);
  err := err or clSetKernelArg(K, 1, csLongintSize, @FNumAs);
  err := err or clSetKernelArg(K, 2, csLongintSize, @FNumBs);
  err := err or clSetKernelArg(K, 3, csLongintSize, @FSize);
  err := err or clSetKernelArg(K, 4, csLongintSize, @FActFun);
  err := err or clSetKernelArg(K, 5, csCLMemSize, @FCodesBuffer);
  err := err or clSetKernelArg(K, 6, csCLMemSize, @BufferBs);
  err := err or clSetKernelArg(K, 7, csCLMemSize, @FResultBuffer);
  err := err or clSetKernelArg(K, 8, csLongintSize, @UseBias);
  err := err or clSetKernelArg(K, 9, csCLMemSize, @FBiasBuffer);
  err := err or clSetKernelArg(K, 10, csCLMemSize, @FScalesBuffer);

  if err = CL_SUCCESS then
  begin
    FDotProductKernel.RunKernel2D(K, FNumAs, FNumBs);
  end
  else
  begin
    ErrorProc('Error: TDotProductSharedKernel.ComputeInt8 - ' +
      'failed setting parameters: ' + IntToStr(err));
  end;
end;

procedure TDotProductSharedKernel.FinishAndLoadResult(Results: TNNetVolume;
  SaveCPU: TNeuralFloat);
var
  ResultSize: integer;
  err: integer; // error code returned from api calls
  finishTime, startTime: TDateTime;
begin
  ResultSize := FNumAs * FNumBs;
  if (ResultSize > Results.Size) then
  begin
    Results.ReSize(ResultSize,1,1);
    MessageProc
    (
      'Expected Result Size is: ' + IntToStr(ResultSize) +
      ' Found Result Size is:' + IntToStr(Results.Size)
    );
  end;

  if SaveCPU > 0 then
  begin
    //if Random(100)=0 then WriteLn(FPreviousComputeTime:6:4);
    // Time Collection
    if (Random(10)=0) then
    begin
      startTime := now();
      err := FDotProductKernel.ReadBuffer(FResultBuffer, Results);
      finishTime := now();
      FPreviousComputeTime := FPreviousComputeTime * 0.99 + (finishTime - startTime)* 24 * 60 * 60 * 1000 * 0.01;
    end
    else
    begin
      if FPreviousComputeTime*SaveCPU > 1 then // 1 ms
      begin
        Sleep(Floor(FPreviousComputeTime*SaveCPU));
      end;
      err := FDotProductKernel.ReadBuffer(FResultBuffer, Results);
    end;
  end
  else
  begin
    err := FDotProductKernel.ReadBuffer(FResultBuffer, Results);
  end;
  if (err <> CL_SUCCESS) then
  begin
    ErrorProc(' Error reading result buffer:' + IntToStr(err));
  end;
end;

function TNeuralKernel.PrepareKernel(kernelname: string): integer;
begin
  UnprepareKernel();
  FKernel := CreateKernel(kernelname);
  PrepareKernel := CL_SUCCESS;
end;

procedure TNeuralKernel.UnprepareKernel();
begin
  if Assigned(FKernel) then clReleaseKernel(FKernel);
  FKernel := nil;
end;

// Locates neural.cl, returning '' when no candidate exists. The same relative
// candidates are tried twice: first against the current directory, then against
// the directory holding the running binary. The executable-relative pass is what
// lets a program be started from anywhere - the test suite run from the repo
// root, or an example launched by an IDE with its own working directory - since
// otherwise the kernel silently fails to build and every offload returns
// garbage. (Coded by Claude (AI).)
function FindNeuralKernelSource(): string;
const
  // Ordered from the deepest example directory outwards, matching where the
  // binaries this library builds actually sit relative to neural/.
  Candidates: array[0..5] of string = (
    '../../../neural/neural.cl',
    'neural.cl',
    'neural-api/neural/neural.cl',
    '../neural/neural.cl',
    '../../neural/neural.cl',
    'neural/neural.cl'
  );
var
  I: integer;
  ExePath: string;
begin
  for I := Low(Candidates) to High(Candidates) do
  begin
    if FileExists(Candidates[I]) then
    begin
      Result := Candidates[I];
      exit;
    end;
  end;
  ExePath := ExtractFilePath(ParamStr(0));
  if ExePath <> '' then
  begin
    for I := Low(Candidates) to High(Candidates) do
    begin
      if FileExists(ExePath + Candidates[I]) then
      begin
        Result := ExePath + Candidates[I];
        exit;
      end;
    end;
  end;
  Result := '';
end;

constructor TNeuralKernel.Create(pCurrentPlatform: cl_platform_id;
  pCurrentDevice: cl_device_id; kernelname: string = 'cai_dot_product';
  pHideMessages: boolean = false);
var
  KernelSource: string;
begin
  inherited Create();
  // Optionally suppress the routine "clCreateContext/clBuildProgram/
  // clCreateKernel ... OK!" progress chatter emitted while compiling neural.cl:
  // a net enabling OpenCL builds this kernel once, but a process that spins up
  // many nets (or the per-layer benchmark) repeats the whole banner each time.
  // Default false keeps the historical verbose behaviour for existing callers;
  // pass pHideMessages=true to opt in. HideMessages only gags FMessageProc;
  // FErrorProc (build failures, missing neural.cl below) is untouched, so real
  // problems still surface. (Coded by Claude (AI).)
  if pHideMessages then HideMessages();
  SetCurrentPlatform(pCurrentPlatform);
  SetCurrentDevice(pCurrentDevice);

  // Create the OpenCL Kernel Here:
  KernelSource := FindNeuralKernelSource();
  if KernelSource <> '' then
  begin
    CompileProgramFromFile(KernelSource);
  end
  else
  begin
    // Report through ErrorProc, not MessageProc: a missing kernel means the
    // offload silently builds nothing and returns garbage, so this must stay
    // visible even when the routine progress messages are hidden.
    FErrorProc('File neural.cl could not be found.');
  end;
  PrepareKernel(kernelname);
end;

constructor TNeuralKernel.CreateFromProgram(SharedKernel: TEasyOpenCL;
  kernelname: string; pHideMessages: boolean = true;
  pSharedQueue: boolean = true);
begin
  inherited Create();
  // Suppress the per-layer "clCreateKernel ... OK!" chatter by default: a model
  // with many transformer blocks binds the same helper kernel dozens of times.
  if pHideMessages then HideMessages();
  // Borrow the shared kernel's context/program, and its command queue when
  // pSharedQueue. FBorrowedContext/FBorrowedQueue keep the destructor from
  // releasing what it does not own (they outlive this helper). No
  // CompileProgramFromFile call here: neural.cl was already built once when the
  // shared dot-product kernel was created.
  FBorrowedContext := true;
  FBorrowedQueue := pSharedQueue;
  FCurrentPlatform := SharedKernel.CurrentPlatform;
  FCurrentDevice   := SharedKernel.CurrentDevice;
  FContext  := SharedKernel.Context;
  FProg     := SharedKernel.Prog;
  if pSharedQueue
    then FCommands := SharedKernel.Commands
    else FCommands := CreateCommandQueue(); // the command queue is given per kernel
  // Bind our own kernel handle into the shared (already-built) program.
  PrepareKernel(kernelname);
end;

destructor TNeuralKernel.Destroy();
begin
  UnprepareKernel();
  inherited Destroy();
end;

{ TDotProductCL }
constructor TDotProductCL.Create(pCurrentPlatform: cl_platform_id; pCurrentDevice: cl_device_id; kernelname: string = 'cai_dot_product'; pHideMessages: boolean = false);
begin
  inherited Create(pCurrentPlatform, pCurrentDevice, kernelname, pHideMessages);
  FInputBufferAs := nil;
  FInputBufferBs := nil;
  FResultBuffer  := nil;
  FHostInput     := False;

  FNumAs := 0;
  FNumBs := 0;
  FSize := 0;
end;

destructor TDotProductCL.Destroy();
begin
  UnprepareForCompute();

  inherited Destroy();
end;

procedure TDotProductCL.UnprepareForCompute();
begin
  if Assigned(FInputBufferAs) then clReleaseMemObject(FInputBufferAs);
  if Assigned(FInputBufferBs) then clReleaseMemObject(FInputBufferBs);
  if Assigned(FResultBuffer)  then clReleaseMemObject(FResultBuffer);

  FInputBufferAs := nil;
  FInputBufferBs := nil;
  FResultBuffer  := nil;
  UnprepareKernel();
end;

function TDotProductCL.PrepareForCompute(VAs, VBs: TNNetVolume; pSize: longint;
  kernelname: string; GroupSizeA: integer; GroupSizeB: integer): integer;
var
  err: integer; // error code returned from api calls
  UseBiasZero: longint; // this class never fuses bias
  NilBias: cl_mem;
begin
  UnprepareForCompute();

  FNumAs := VAs.Size div pSize;
  FNumBs := VBs.Size div pSize;
  FThreadCount := FNumAs * FNumBs;
  FSize := pSize;
  FGroupSizeA := GroupSizeA;
  FGroupSizeB := GroupSizeB;

  if (FHostInput) then
  begin
    FInputBufferAs := CreateHostInputBuffer(VAs);
    FInputBufferBs := CreateHostInputBuffer(VBs);
  end
  else
  begin
    FInputBufferAs := CreateInputBuffer(VAs);
    FInputBufferBs := CreateInputBuffer(VBs);
  end;
  FResultBuffer  := CreateOutputBuffer(FNumAs * FNumBs * csNeuralFloatSize);
  FPreviousComputeTime := 0;

  err := PrepareKernel(kernelname);

  err := err or clSetKernelArg(FKernel, 0, csLongintSize, @FThreadCount);
  if (err <> CL_SUCCESS) then ErrorProc('0 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 1, csLongintSize, @FNumAs);
  if (err <> CL_SUCCESS) then ErrorProc('1 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 2, csLongintSize, @FNumBs);
  if (err <> CL_SUCCESS) then ErrorProc('2 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 3, csLongintSize, @FSize);
  if (err <> CL_SUCCESS) then ErrorProc('3 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 4, csLongintSize, @FActFun);
  if (err <> CL_SUCCESS) then ErrorProc('4 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 5, csCLMemSize,  @FInputBufferAs);
  if (err <> CL_SUCCESS) then ErrorProc('5 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 6, csCLMemSize,  @FInputBufferBs);
  if (err <> CL_SUCCESS) then ErrorProc('6 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 7, csCLMemSize,  @FResultBuffer);
  if (err <> CL_SUCCESS) then ErrorProc('7 Error: Failed to set kernel arguments:' + IntToStr(err));

  // cai_dot_product gained two fused-bias args (8 UseBias, 9 FBiasOutput). This
  // class never fuses bias, but every arg must be set once before enqueue, so pin
  // UseBias=0 and a NULL bias buffer here (clSetKernelArg copies the value
  // immediately, so the locals are safe). Coded by Claude (AI).
  UseBiasZero := 0;
  NilBias := nil;
  err := err or clSetKernelArg(FKernel, 8, csLongintSize, @UseBiasZero);
  if (err <> CL_SUCCESS) then ErrorProc('8 Error: Failed to set kernel arguments:' + IntToStr(err));
  err := err or clSetKernelArg(FKernel, 9, csCLMemSize, @NilBias);
  if (err <> CL_SUCCESS) then ErrorProc('9 Error: Failed to set kernel arguments:' + IntToStr(err));

  PrepareForCompute := err;
end;

procedure TDotProductCL.Compute(VAs, VBs: TNNetVolume; pActFN: longint);
var
  err: integer;
begin
  if (VAs.Size = FSize * FNumAs) then
  begin
    if (VBs.Size = FSize * FNumBs) then
    begin
      if (FHostInput) then
      begin
        err :=
          //TODO: Fix this refresh.
          //RefreshHostInputBufferCache(FInputBufferAs, VAs.GetMemSize()) or
          //RefreshHostInputBufferCache(FInputBufferBs, VBs.GetMemSize())
          WriteBuffer(FInputBufferAs, VAs) or
          WriteBuffer(FInputBufferBs, VBs);
        ;
      end
      else
      begin
        err :=
          WriteBuffer(FInputBufferAs, VAs) or
          WriteBuffer(FInputBufferBs, VBs);
      end;

      FActFun := pActFN;

      err := err or clSetKernelArg(FKernel, 4, csLongintSize, @FActFun);

      if err = CL_SUCCESS then
      begin

        if (FGroupSizeA > 0) and (FGroupSizeB > 0)  then
        begin
          RunKernel2D(FKernel, FNumAs, FNumBs, FGroupSizeA, FGroupSizeB);
        end
        else
        begin
          RunKernel2D(FKernel, FNumAs, FNumBs);
        end;

      end
      else
      begin
        ErrorProc
        (
          'Error: TDotProductCL.Compute - ' +
          ' Failed setting parameters: ' + IntToStr(err)
        );
      end;
    end
    else
    begin
      ErrorProc
      (
        'Error: TDotProductCL.Compute - VB size: ' +
        IntToStr(VAs.Size) +
        ' FSize: ' + IntToStr(FSize) +
        ' NumBs:' + IntToStr(FNumBs)
      );
    end;
  end
  else
  begin
    ErrorProc
    (
      'Error: TDotProductCL.Compute - VA size: ' +
      IntToStr(VAs.Size) +
      ' FSize: ' + IntToStr(FSize) +
      ' NumAs:' + IntToStr(FNumAs)
    );
  end;
end;

procedure TDotProductCL.FinishAndLoadResult(Results: TNNetVolume; SaveCPU: TNeuralFloat);
var
  ResultSize: integer;
  err: integer; // error code returned from api calls
  finishTime, startTime: TDateTime;
begin
  ResultSize := FNumAs * FNumBs;
  if (ResultSize > Results.Size) then
  begin
    Results.ReSize(ResultSize,1,1);
    MessageProc
    (
      'Expected Result Size is: ' + IntToStr(ResultSize) +
      ' Found Result Size is:' + IntToStr(Results.Size)
    );
  end;

  if SaveCPU > 0 then
  begin
    if (Random(10)=0) then
    begin
      startTime := now();
      err := ReadBuffer(FResultBuffer, Results);
      finishTime := now();
      FPreviousComputeTime := FPreviousComputeTime * 0.99 + (finishTime - startTime)* 24 * 60 * 60 * 1000 * 0.01;
    end
    else
    begin
      if FPreviousComputeTime*SaveCPU > 1 then // 1 ms
      begin
        Sleep(Floor(FPreviousComputeTime*SaveCPU));
      end;
      err := ReadBuffer(FResultBuffer, Results);
    end;
  end
  else
  begin
    err := ReadBuffer(FResultBuffer, Results);
  end;
  if (err <> CL_SUCCESS) then
  begin
    ErrorProc(' Error reading result buffer:' + IntToStr(err));
  end;
end;

{ TEasyOpenCLV }
function TEasyOpenCLV.CreateBuffer(flags: cl_mem_flags; V: TNNetVolume): cl_mem;
begin
  Result := CreateBuffer(flags, V.GetMemSize());
end;

function TEasyOpenCLV.CreateInputBuffer(V: TNNetVolume): cl_mem;
begin
  Result := CreateBuffer(CL_MEM_READ_ONLY, V.GetMemSize());
end;

function TEasyOpenCLV.CreateHostInputBuffer(V: TNNetVolume): cl_mem;
begin
  Result := CreateBuffer(CL_MEM_READ_ONLY or CL_MEM_USE_HOST_PTR, V.GetMemSize(), V.DataPtr);
end;

function TEasyOpenCLV.CreateOutputBuffer(V: TNNetVolume): cl_mem;
begin
  Result := CreateBuffer(CL_MEM_WRITE_ONLY, V.GetMemSize());
end;

function TEasyOpenCLV.CreateBuffer(V: TNNetVolume): cl_mem;
begin
  Result := CreateBuffer(CL_MEM_READ_WRITE, V.GetMemSize());
end;

function TEasyOpenCLV.WriteBuffer(buffer: cl_mem; V: TNNetVolume; blocking: cl_bool): integer;
begin
  Result := WriteBuffer(buffer, V.GetMemSize(), V.DataPtr, blocking);
end;

function TEasyOpenCLV.ReadBuffer(buffer: cl_mem; V: TNNetVolume; blocking: cl_bool): integer;
begin
  Result := ReadBuffer(buffer, V.GetMemSize(), V.DataPtr, blocking);
end;

function TEasyOpenCLV.CreateAndWriteBuffer(V: TNNetVolume; var buffer: cl_mem
  ): integer;
begin
  buffer := CreateBuffer(V);
  Result := WriteBuffer(buffer,V);
end;

function TEasyOpenCLV.CreateAndWriteBuffer(V: TNNetVolume): cl_mem;
begin
  Result := nil;
  CreateAndWriteBuffer(V,Result);
end;

function TEasyOpenCLV.CreateWriteSetArgument(V: TNNetVolume; kernel:cl_kernel; arg_index: cl_uint): cl_mem;
begin
  Result := nil;
  CreateAndWriteBuffer(V, Result);
  clSetKernelArg(kernel, arg_index, csCLMemSize, @Result);
end;

function TEasyOpenCLV.CreateOutputSetArgument(V: TNNetVolume;
  kernel: cl_kernel; arg_index: cl_uint): cl_mem;
begin
  Result := CreateOutputBuffer(V);
  clSetKernelArg(kernel, arg_index, csCLMemSize, @Result);
end;

function TEasyOpenCLV.EnsureBuffer(var buf: cl_mem; var capBytes: {$IFDEF FPC}PtrUInt{$ELSE}UIntPtr{$ENDIF};
  flags: cl_mem_flags; neededBytes: {$IFDEF FPC}PtrUInt{$ELSE}UIntPtr{$ENDIF}): cl_mem;
begin
  // Grow only: reuse the existing allocation whenever it is large enough (a
  // buffer bigger than needed is harmless - WriteBuffer/ReadBuffer move exactly
  // V.GetMemSize() bytes and the kernels are bounded by explicit size args).
  if (buf = nil) or (neededBytes > capBytes) then
  begin
    if Assigned(buf) then clReleaseMemObject(buf);
    buf := CreateBuffer(flags, neededBytes);
    capBytes := neededBytes;
  end;
  Result := buf;
end;

function TEasyOpenCLV.EnsureWriteBuffer(var buf: cl_mem; var capBytes: csize_t;
  V: TNNetVolume; DoWrite: boolean = true): cl_mem;
var
  PreviousBuffer: cl_mem;
begin
  PreviousBuffer := buf;
  // READ_WRITE (not READ_ONLY) so one persistent buffer can back either an
  // input or output role across shapes without flag mismatches.
  Result := EnsureBuffer(buf, capBytes, CL_MEM_READ_WRITE, V.GetMemSize());
  // DoWrite=false leaves the resident device copy in place (weights unchanged),
  // but a fresh handle holds nothing, so the first call and any growth upload
  // whatever the caller asked for.
  if DoWrite or (Result <> PreviousBuffer) then WriteBuffer(Result, V, CL_FALSE);
end;

function TEasyOpenCLV.EnsureOutputBuffer(var buf: cl_mem; var capBytes: csize_t;
  V: TNNetVolume): cl_mem;
begin
  Result := EnsureBuffer(buf, capBytes, CL_MEM_READ_WRITE, V.GetMemSize());
end;

{ TEasyOpenCL }
procedure TEasyOpenCL.LoadPlatforms();
var
  err: integer; // error code returned from api calls
  firstpointer, local_platformids: {$IFDEF FPC}pcl_platform_id{$ELSE}ppcl_platform_id{$ENDIF};
  local_platforms: cl_uint;
  i: integer;
  buf: TNeuralStrBuffer;
  bufwritten: csize_t;
  local_platformsM1: integer;
begin
  bufwritten := 0;
  err := clGetPlatformIDs(0, nil, @local_platforms);
  if (err <> CL_SUCCESS) then
  begin
    FErrorProc('Error: Cannot get number of platforms!');
    exit;
  end;
  getmem(local_platformids, local_platforms * sizeof(cl_platform_id));
  firstpointer := local_platformids;
  err := clGetPlatformIDs(local_platforms, local_platformids, nil);
  if (err <> CL_SUCCESS) then
  begin
    FErrorProc('Error: Cannot get platforms!');
    freemem(local_platformids);
    exit;
  end;

  SetLength(FPlatformNames, local_platforms);
  SetLength(FPlatformIds, local_platforms);

  if (local_platforms > 0) then
  begin
    local_platformsM1 := local_platforms - 1;
    for i := 0 to local_platformsM1 do
    begin
      {$IFDEF FPC}
      err := clGetPlatformInfo(local_platformids[i], CL_PLATFORM_NAME, sizeof(buf), @buf, bufwritten);
      FPlatformNames[i] := buf;
      FPlatformIds[i]   := local_platformids[i];
      {$ELSE}
      err := clGetPlatformInfo(local_platformids^, CL_PLATFORM_NAME, sizeof(buf), @buf, @bufwritten);
      FPlatformNames[i] := buf;
      FPlatformIds[i]   := local_platformids^;
      Inc(local_platformids);
      {$ENDIF}
    end;
  end;
  freemem(firstpointer);
end;

procedure TEasyOpenCL.FreeContext();
begin
  // A borrowed context, program or command queue belongs to the shared kernel
  // that created it; releasing one here would tear it out from under every other
  // borrower, so those references are only dropped. (Coded by Claude (AI).)
  if (not FBorrowedQueue) and Assigned(FCommands) then
    clReleaseCommandQueue(FCommands);
  FCommands := nil;
  if FBorrowedContext then
  begin
    FProg := nil;
    FContext := nil;
  end
  else
  begin
    if Assigned(FProg) then clReleaseProgram(FProg);
    if Assigned(FContext) then clReleaseContext(FContext);
    FProg := nil;
    FContext := nil;
  end;
end;

procedure TEasyOpenCL.CompileProgram();
var
  localKernelSource: TNeuralPChar;
  errorlog, localCompilerOptions: TNeuralPChar;
  err: integer; // error code returned from api calls
  errorlogstr: TNeuralStrBuffer;
  loglen: csize_t;
begin
  err := 0;
  FreeContext();

  {$IFDEF FPC}
  localKernelSource := FOpenCLProgramSource.GetText();
  {$ELSE}
  localKernelSource := AnsiStrings.StrNew(PAnsiChar(AnsiString(FOpenCLProgramSource.Text)));
  {$ENDIF}

  // Create a compute context
  FContext := CreateContext();
  if FContext = nil then exit;

  // Create a command queue
  FCommands := CreateCommandQueue();

  // Create the compute program from the source buffer
  {$IFDEF FPC}
  FProg := clCreateProgramWithSource(context, 1, PPChar(@localKernelSource), nil,  err);
  {$ELSE}
  FProg := clCreateProgramWithSource(context, 1, PPAnsiChar(@localKernelSource), nil,  @err);
  {$ENDIF}
  if FProg = nil then
  begin
    FMessageProc(localKernelSource);
    FErrorProc('Error: Failed to create compute program:' + IntToStr(err));
    exit;
  end
  else
    FMessageProc('clCreateProgramWithSource OK!');

  localCompilerOptions := {$IFDEF FPC}StrAlloc{$ELSE}AnsiStrAlloc{$ENDIF}(length(FCompilerOptions)+1);
  {$IFDEF FPC}StrPCopy{$ELSE}AnsiStrings.StrPCopy{$ENDIF}(localCompilerOptions,FCompilerOptions);

  // Build the program executable
  err := clBuildProgram(FProg, 0, nil, localCompilerOptions, nil, nil);

  {$IFDEF FPC}StrDispose{$ELSE}AnsiStrings.StrDispose{$ENDIF}(localCompilerOptions);

  if (err <> CL_SUCCESS) then
  begin
    errorlog := @errorlogstr[1];
    loglen := SizeOf(errorlogstr);
    clGetProgramBuildInfo(FProg, FCurrentDevice, CL_PROGRAM_BUILD_LOG, SizeOf(errorlogstr), errorlog, {$IFDEF FPC}loglen{$ELSE}@loglen{$ENDIF});
    FErrorProc('Error: Failed to build program executable:' + IntToStr(err) + ' ' + errorlog);
    exit;
  end
  else
    FMessageProc('clBuildProgram OK!');
end;

procedure TEasyOpenCL.printDevicesInfo();
var
  local_devices: TDeviceNames;
  local_deviceids: TDevices;
  i, j, k: integer;
  buf: TNeuralStrBuffer;
  bufwritten: csize_t;
  FPlatformIdsHi, platform_str_infoHi, device_str_infoHi: integer;
  device_word_infoHi, local_deviceidsHi: integer;
begin
  bufwritten := 0;
  platform_str_infoHi := high(platform_str_info);
  device_str_infoHi := high(device_str_info);
  device_word_infoHi := high(device_word_info);
  if GetPlatformCount()>0 then
  begin
    FPlatformIdsHi := High(FPlatformIds);
    for i := Low(FPlatformIds) to FPlatformIdsHi do
    begin
      FMessageProc('Platform info: ' + IntToStr(i) + ' ---------------------');
      for k := low(platform_str_info) to platform_str_infoHi do
      begin
        clGetPlatformInfo(FPlatformIds[i], platform_str_info[k].id, sizeof(buf), @buf, {$IFDEF FPC}bufwritten{$ELSE}@bufwritten{$ENDIF});
        MessageProc(platform_str_info[k].Name + ': ' + buf);
      end;

      GetDevicesFromPlatform(FPlatformIds[i], local_devices, local_deviceids);

      if Length(local_devices)>0 then
      begin
        local_deviceidsHi := High(local_deviceids);
        for j := Low(local_deviceids) to local_deviceidsHi do
        begin
          MessageProc('Device info: ' + IntToStr(j) + ' ------------');
          for k := low(device_str_info) to device_str_infoHi do
          begin
            clGetDeviceInfo(local_deviceids[j], device_str_info[k].id, sizeof(buf), @buf, {$IFDEF FPC}bufwritten{$ELSE}@bufwritten{$ENDIF});
            MessageProc(device_str_info[k].Name + ': ' + buf);
          end;

          for k := low(device_word_info) to device_word_infoHi do
          begin
            clGetDeviceInfo(local_deviceids[j], device_word_info[k].id, sizeof(buf), @buf, {$IFDEF FPC}bufwritten{$ELSE}@bufwritten{$ENDIF});
            MessageProc(device_word_info[k].Name + ': ' + IntToStr(pdword(@buf)^));
          end;
        end;
      end;
    end;
  end;
end;

function TEasyOpenCL.GetPlatformCount(): integer;
begin
  Result := Length(FPlatformNames);
end;

function TEasyOpenCL.GetDeviceCount(): integer;
begin
  Result := Length(FDeviceNames);
end;

procedure TEasyOpenCL.GetDevicesFromPlatform(PlatformId: cl_platform_id; out pDeviceNames: TDeviceNames; out pDevices: TDevices);
var
  err: integer; // error code returned from api calls
  local_devices: cl_uint;
  firstpointer, local_deviceids: {$IFDEF FPC}pcl_device_id{$ELSE}ppcl_device_id{$ENDIF};
  j: integer;
  buf: TNeuralStrBuffer;
  bufwritten: csize_t;
  local_devicesM1: integer;
begin
  bufwritten := 0;
  err := clGetDeviceIDs(PlatformId, CL_DEVICE_TYPE_ALL, 0, nil, @local_devices);
  if (err <> CL_SUCCESS) then
  begin
    FErrorProc('ERROR: Cannot get number of devices for platform.');
  end
  else
  begin
    SetLength(pDeviceNames, local_devices);
    SetLength(pDevices, local_devices);

    getmem(local_deviceids, local_devices * sizeof(cl_device_id));
    firstpointer := local_deviceids;
    err := clGetDeviceIDs(PlatformId, CL_DEVICE_TYPE_ALL, local_devices, local_deviceids, nil);

    if (local_devices > 0) then
    begin
      local_devicesM1 := local_devices - 1;
      for j := 0 to local_devicesM1 do
      begin
        {$IFDEF FPC}
        err := clGetDeviceInfo(local_deviceids[j], CL_DEVICE_NAME, sizeof(buf), @buf, bufwritten);
        pDeviceNames[j] := buf;
        pDevices[j] := local_deviceids[j];
        {$ELSE}
        err := clGetDeviceInfo(local_deviceids^, CL_DEVICE_NAME, sizeof(buf), @buf, @bufwritten);
        pDeviceNames[j] := buf;
        pDevices[j] := local_deviceids^;
        Inc(local_deviceids);
        {$ENDIF}
      end;
    end;
    freemem(firstpointer);
  end;
end;

procedure TEasyOpenCL.SetCurrentPlatform(pPlatformId: cl_platform_id);
begin
  FCurrentPlatform := pPlatformId;
  GetDevicesFromPlatform(pPlatformId, FDeviceNames, FDevices);
end;

procedure TEasyOpenCL.SetCurrentDevice(pDeviceId: cl_device_id);
begin
  FCurrentDevice := pDeviceId;
  FMaxComputeUnits := 0;
end;

procedure TEasyOpenCL.CompileProgramFromFile(filename: string);
begin
  if FileExists(filename) then
  begin
    FOpenCLProgramSource.LoadFromFile(filename);
    CompileProgram();
  end
  else
  begin
    ErrorProc('File not found:' + filename);
  end;
end;

procedure TEasyOpenCL.CompileProgram(programsource: TStrings);
begin
  FOpenCLProgramSource.Text := programsource.Text;
  CompileProgram();
end;

procedure TEasyOpenCL.CompileProgram(programsource: string);
begin
  FOpenCLProgramSource.Text := programsource;
  CompileProgram();
end;

function TEasyOpenCL.CreateContext(): cl_context;
var
  err: integer; // error code returned from api calls
begin
  err := 0;
  Result := clCreateContext(nil, 1, @FCurrentDevice, nil, nil, {$IFDEF FPC}err{$ELSE}@err{$ENDIF});

  if Result = nil then
  begin
    FErrorProc('Error: Failed to create a compute context:' + IntToStr(err));
    exit;
  end
  else
    FMessageProc('clCreateContext OK!');
end;

function TEasyOpenCL.CreateCommandQueue(): cl_command_queue;
var
  err: integer; // error code returned from api calls
begin
  // FPC's cl.pp mistypes clCreateCommandQueue's errcode_ret as a by-value
  // cl_int instead of a pointer, so whatever err holds is passed to the driver
  // as the address it writes the status to. Zero means "no status wanted": any
  // other value makes the driver write to a wild pointer and segfault.
  err := 0;
  Result := clCreateCommandQueue(context, FCurrentDevice, 0,  {$IFDEF FPC}err{$ELSE}@err{$ENDIF});
  if Result = nil then
  begin
    FErrorProc('Error: Failed to create a command queue.');
    exit;
  end
  else
    FMessageProc('clCreateCommandQueue OK!');
end;

function TEasyOpenCL.CreateBuffer(flags: cl_mem_flags; size: csize_t; ptr: Pointer = nil): cl_mem;
var
  err: integer; // error code returned from api calls
begin
  err := 0;
  Result := clCreateBuffer(FContext, flags, size, ptr, {$IFDEF FPC}err{$ELSE}@err{$ENDIF});

  if (err <> CL_SUCCESS) or (Result = nil) then
  begin
    FErrorProc('clCreateBuffer :'+ IntToStr(err)+ ' Size:'+ IntToStr(size)+' bytes.');
  end;
end;

function TEasyOpenCL.MapBuffer(buffer: cl_mem; cb: csize_t;
  map_flags: cl_map_flags;
  blocking: cl_bool): Pointer;
var
  err: integer; // error code returned from api calls
begin
  err := 0;
  Result := clEnqueueMapBuffer(FCommands, buffer, blocking, map_flags, {offset=}0, cb,
    {num_events=}0, {events_list=}nil, {event=}nil, {$IFDEF FPC}err{$ELSE}@err{$ENDIF});
  if (err <> CL_SUCCESS) then
  begin
    FErrorProc('clEnqueueMapBuffer :'+ IntToStr(err)+ ' Size:'+ IntToStr(cb)+' bytes.');
  end;
end;

function TEasyOpenCL.MapHostInputBuffer(buffer: cl_mem; cb: csize_t): Pointer;
begin
  Result := MapBuffer(buffer, cb, {map_flags=}CL_MAP_READ, {blocking=}CL_TRUE);
end;

function TEasyOpenCL.UnmapMemObject(buffer: cl_mem; mapped_ptr: Pointer): cl_int;
begin
  Result := clEnqueueUnmapMemObject(FCommands, buffer, mapped_ptr, {num_events=}0, {events_list=}nil, {event=}nil);
  if (Result <> CL_SUCCESS) then
  begin
    FErrorProc('UnmapMemObject :'+ IntToStr(Result)+'.');
  end;
end;

function TEasyOpenCL.RefreshHostInputBufferCache(buffer: cl_mem; cb: csize_t
  ): cl_int;
var
  mapped_ptr: Pointer;
begin
  mapped_ptr := MapHostInputBuffer(buffer, cb);
  Result := UnmapMemObject(buffer, mapped_ptr);
end;

function TEasyOpenCL.WriteBuffer(buffer: cl_mem; cb: csize_t; ptr: Pointer; blocking: cl_bool): integer;
begin
  Result := clEnqueueWriteBuffer(FCommands, buffer, blocking, 0, cb, ptr, 0, nil, nil);
  if (Result <> CL_SUCCESS) then
  begin
    FErrorProc('clCreateBuffer :'+ IntToStr(Result)+ ' Size:'+ IntToStr(cb)+' bytes.');
  end;
end;

function TEasyOpenCL.ReadBuffer(buffer: cl_mem; cb: csize_t; ptr: Pointer; blocking: cl_bool): integer;
begin
  Result := clEnqueueReadBuffer(FCommands, buffer, blocking, 0, cb, ptr, 0, nil, nil);
  if (Result <> CL_SUCCESS) then
  begin
    if (Result = CL_OUT_OF_RESOURCES)
    then FErrorProc('ERROR: Out of computing resources - probably out of memory.')
    else FErrorProc('ERROR: Failed to read output array: ' + IntToStr(Result));
  end
end;

function TEasyOpenCL.WriteBufferAt(buffer: cl_mem; offsetBytes, cb: csize_t; ptr: Pointer; blocking: cl_bool): integer;
begin
  Result := clEnqueueWriteBuffer(FCommands, buffer, blocking, offsetBytes, cb, ptr, 0, nil, nil);
  if (Result <> CL_SUCCESS) then
  begin
    FErrorProc('ERROR: Failed to write buffer slice: ' + IntToStr(Result) +
      ' Offset:' + IntToStr(offsetBytes) + ' Size:' + IntToStr(cb) + ' bytes.');
  end;
end;

function TEasyOpenCL.ReadBufferAt(buffer: cl_mem; offsetBytes, cb: csize_t; ptr: Pointer; blocking: cl_bool): integer;
begin
  Result := clEnqueueReadBuffer(FCommands, buffer, blocking, offsetBytes, cb, ptr, 0, nil, nil);
  if (Result <> CL_SUCCESS) then
  begin
    FErrorProc('ERROR: Failed to read buffer slice: ' + IntToStr(Result) +
      ' Offset:' + IntToStr(offsetBytes) + ' Size:' + IntToStr(cb) + ' bytes.');
  end;
end;

function TEasyOpenCL.CreateInputBuffer(size: csize_t): cl_mem;
begin
  Result := CreateBuffer(CL_MEM_READ_ONLY,size);
end;

function TEasyOpenCL.CreateHostInputBuffer(size: csize_t; ptr: Pointer): cl_mem;
begin
  Result := CreateBuffer(CL_MEM_READ_ONLY or CL_MEM_USE_HOST_PTR, size, ptr);
end;

function TEasyOpenCL.CreateOutputBuffer(size: csize_t): cl_mem;
begin
  Result := CreateBuffer(CL_MEM_WRITE_ONLY,size);
end;

function TEasyOpenCL.CreateBuffer(size: csize_t): cl_mem;
begin
  Result := CreateBuffer(CL_MEM_READ_WRITE,size);
end;

function TEasyOpenCL.CreateKernel(kernelname: string): cl_kernel;
var
  localKernelName: TNeuralPChar;
  err: integer; // error code returned from api calls
begin
  err := 0;
  localKernelName := {$IFDEF FPC}StrAlloc{$ELSE}AnsiStrAlloc{$ENDIF}(length(kernelname)+1);
  {$IFDEF FPC}StrPCopy{$ELSE}AnsiStrings.StrPCopy{$ENDIF}(localKernelName,kernelname);

  // Create the compute kernel in the program we wish to run
  Result := clCreateKernel(prog, localKernelName, {$IFDEF FPC}err{$ELSE}@err{$ENDIF});
  if (Result = nil) or (err <> CL_SUCCESS) then
  begin
    FErrorProc('Error: Failed to create compute kernel:'+kernelname);
  end
  else
  begin
    FMessageProc('clCreateKernel '+kernelname+' OK!');
  end;
  {$IFDEF FPC}StrDispose{$ELSE}AnsiStrings.StrDispose{$ENDIF}(localKernelName);
end;

function TEasyOpenCL.DeviceMaxComputeUnits(): integer;
var
  Units: cl_uint;
  BytesWritten: csize_t;
begin
  Result := FMaxComputeUnits;
  if Result > 0 then exit;
  Result := 1;
  if FCurrentDevice = nil then exit;
  Units := 0;
  if clGetDeviceInfo(FCurrentDevice, CL_DEVICE_MAX_COMPUTE_UNITS,
    SizeOf(Units), @Units, {$IFDEF FPC}BytesWritten{$ELSE}@BytesWritten{$ENDIF}) = CL_SUCCESS then
  begin
    if Units > 0 then Result := Units;
  end;
  FMaxComputeUnits := Result;
end;

function TEasyOpenCL.RunKernel(pkernel: cl_kernel; ThreadCount: integer): integer;
var
  GlobalThreadCount: csize_t;
  work_dim: cl_uint;
begin
  GlobalThreadCount := ThreadCount;
  work_dim := 1;
  Result := clEnqueueNDRangeKernel(FCommands, pkernel, work_dim, nil, @GlobalThreadCount, nil, 0, nil, nil);
  if (Result <> CL_SUCCESS) then
  begin
    if (Result = CL_INVALID_WORK_GROUP_SIZE)
    then FErrorProc('ERROR: Invalid work group size.')
    else FErrorProc('ERROR: Failed to execute kernel. Error:' + IntToStr(Result));
  end;
end;

function TEasyOpenCL.RunKernel2D(pkernel: cl_kernel;
  d1size, d2size: csize_t): integer;
var
  work_dim: cl_uint;
  dim_sizes: array[0..1] of csize_t;
begin
  work_dim := 2;
  dim_sizes[0] := d1size;
  dim_sizes[1] := d2size;

  Result := clEnqueueNDRangeKernel(FCommands, pkernel, work_dim, nil, @dim_sizes[0], nil, 0, nil, nil);

  if (Result <> CL_SUCCESS) then
  begin
    if (Result = CL_INVALID_WORK_GROUP_SIZE)
    then FErrorProc('ERROR: Invalid work group size.')
    else FErrorProc('ERROR: Failed to execute kernel. Error:' + IntToStr(Result));
  end;
end;

function TEasyOpenCL.RunKernel2D(pkernel: cl_kernel; d1size, d2size,
  d1groupsize, d2groupsize: csize_t): integer;
var
  work_dim: cl_uint;
  dim_sizes, group_sizes: array[0..1] of csize_t;
begin
  work_dim := 2;
  dim_sizes[0] := d1size;
  dim_sizes[1] := d2size;

  group_sizes[0] := d1groupsize;
  group_sizes[1] := d2groupsize;

  Result := clEnqueueNDRangeKernel(FCommands, pkernel, work_dim, nil, @dim_sizes[0], @group_sizes[0], 0, nil, nil);

  if (Result <> CL_SUCCESS) then
  begin
    if (Result = CL_INVALID_WORK_GROUP_SIZE)
    then FErrorProc('ERROR: Invalid work group size. Global (' +
      IntToStr(Int64(d1size)) + ', ' + IntToStr(Int64(d2size)) +
      ') group (' + IntToStr(Int64(d1groupsize)) + ', ' +
      IntToStr(Int64(d2groupsize)) + ').')
    else FErrorProc('ERROR: Failed to execute kernel. Error:' + IntToStr(Result));
  end;
end;

function TEasyOpenCL.RunKernel3D(pkernel: cl_kernel; d1size, d2size,
  d3size: csize_t): integer;
var
  work_dim: cl_uint;
  dim_sizes: array[0..2] of csize_t;
begin
  work_dim := 3;
  dim_sizes[0] := d1size;
  dim_sizes[1] := d2size;
  dim_sizes[2] := d3size;

  Result := clEnqueueNDRangeKernel(FCommands, pkernel, work_dim, nil, @dim_sizes[0], nil, 0, nil, nil);

  if (Result <> CL_SUCCESS) then
  begin
    if (Result = CL_INVALID_WORK_GROUP_SIZE)
    then FErrorProc('ERROR: Invalid work group size.')
    else FErrorProc('ERROR: Failed to execute kernel. Error:' + IntToStr(Result));
  end;
end;

function TEasyOpenCL.RunKernel3D(pkernel: cl_kernel; d1size, d2size, d3size,
  d1groupsize, d2groupsize, d3groupsize: csize_t): integer;
var
  work_dim: cl_uint;
  dim_sizes, group_sizes: array[0..2] of csize_t;
begin
  work_dim := 3;
  dim_sizes[0] := d1size;
  dim_sizes[1] := d2size;
  dim_sizes[2] := d3size;

  group_sizes[0] := d1groupsize;
  group_sizes[1] := d2groupsize;
  group_sizes[2] := d3groupsize;

  Result := clEnqueueNDRangeKernel(FCommands, pkernel, work_dim, nil, @dim_sizes[0], @group_sizes[0], 0, nil, nil);

  if (Result <> CL_SUCCESS) then
  begin
    if (Result = CL_INVALID_WORK_GROUP_SIZE)
    then FErrorProc('ERROR: Invalid work group size. Global (' +
      IntToStr(Int64(d1size)) + ', ' + IntToStr(Int64(d2size)) + ', ' +
      IntToStr(Int64(d3size)) + ') group (' + IntToStr(Int64(d1groupsize)) +
      ', ' + IntToStr(Int64(d2groupsize)) + ', ' +
      IntToStr(Int64(d3groupsize)) + ').')
    else FErrorProc('ERROR: Failed to execute kernel. Error:' + IntToStr(Result));
  end;
end;

function TEasyOpenCL.Finish(): integer;
begin
  Result := clFinish(FCommands);

  if (Result = CL_SUCCESS) then
    FMessageProc('clFinish OK!')
  else
  begin
    if Result = CL_INVALID_COMMAND_QUEUE
    then FErrorProc('ERROR while running OpenCL code.')
    else FErrorProc('Error at clFinish:' + IntToStr(Result));
  end;
end;

constructor TEasyOpenCL.Create();
begin
  inherited Create();
  FOpenCLProgramSource := TStringList.Create();
  {$IFDEF FPC}
  MessageProc := @Self.DefaultMessageProc;
  ErrorProc := @Self.DefaultErrorProc;
  {$ELSE}
  MessageProc := Self.DefaultMessageProc;
  ErrorProc := Self.DefaultErrorProc;
  {$ENDIF}
  LoadPlatforms();
  SetLength(FDeviceNames, 0);
  SetLength(FDevices, 0);

  FCompilerOptions := '-cl-fast-relaxed-math -cl-mad-enable';

  FContext := nil;        // compute context
  FCommands := nil;       // compute command queue
  FProg := nil;           // compute program
  FBorrowedContext := false;
  FBorrowedQueue := false;
end;

destructor TEasyOpenCL.Destroy();
begin
  FreeContext();
  FOpenCLProgramSource.Free;
  SetLength(FPlatformNames, 0);
  SetLength(FPlatformIds, 0);
  SetLength(FDeviceNames, 0);
  SetLength(FDevices, 0);
  inherited Destroy;
end;

{$IFNDEF FPC}
initialization
InitOpenCL;
{$ENDIF}

end.
