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
    FOpenCLProgramSource: TStringList;

    FContext: cl_context;        // OpenCL compute context
    FCommands: cl_command_queue; // OpenCL compute command queue
    FProg: cl_program;           // OpenCL compute program
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

    function CreateBuffer(flags: cl_mem_flags; size: csize_t; ptr: Pointer = nil): cl_mem; overload;
    function MapBuffer(buffer: cl_mem; cb: csize_t; map_flags: cl_map_flags; blocking: cl_bool = CL_TRUE): Pointer; overload;
    function MapHostInputBuffer(buffer: cl_mem; cb: csize_t): Pointer; overload;
    function UnmapMemObject(buffer: cl_mem; mapped_ptr: Pointer): cl_int;
    function RefreshHostInputBufferCache(buffer: cl_mem; cb: csize_t): cl_int;
    function WriteBuffer(buffer: cl_mem; cb: csize_t; ptr: Pointer; blocking: cl_bool = CL_FALSE): integer; overload;
    function ReadBuffer(buffer: cl_mem; cb: csize_t; ptr: Pointer; blocking: cl_bool = CL_TRUE): integer; overload;

    function CreateInputBuffer(size: csize_t): cl_mem; overload; {$IFDEF Release} inline; {$ENDIF}
    function CreateHostInputBuffer(size: csize_t; ptr: Pointer): cl_mem; overload; {$IFDEF Release} inline; {$ENDIF}
    function CreateOutputBuffer(size: csize_t): cl_mem; overload; {$IFDEF Release} inline; {$ENDIF}
    function CreateBuffer(size: csize_t): cl_mem;  overload; {$IFDEF Release} inline; {$ENDIF}

    function CreateKernel(kernelname: string): cl_kernel;
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
  end;

  TNeuralKernel = class(TEasyOpenCLV)
    private
      /// OpenCL Kernel
      FKernel: cl_kernel;
      function PrepareKernel(kernelname: string = 'cai_dot_product'): integer;
      procedure UnprepareKernel();
    public
      constructor Create(pCurrentPlatform: cl_platform_id; pCurrentDevice: cl_device_id; kernelname: string = 'cai_dot_product');
      destructor Destroy(); override;

      property Kernel: cl_kernel read FKernel;
  end;

  TDotProductKernel = class(TNeuralKernel);

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

      FDotProductKernel: TDotProductKernel;

      function Kernel(): cl_kernel; {$IFDEF Release} inline; {$ENDIF}
    public
      constructor Create(DotProductKernel: TDotProductKernel);
      destructor Destroy(); override;

      procedure UnprepareForCompute();
      function PrepareForCompute(VAs, VBs: TNNetVolume; pSize: longint; GroupSizeA: integer=0; GroupSizeB: integer=0): integer;
      procedure Compute(VAs, VBs: TNNetVolume; pActFN: longint; NewVAs:boolean = true; NewVBs:boolean = true);
      procedure FinishAndLoadResult(Results: TNNetVolume; SaveCPU: TNeuralFloat = 0); overload;
  end;

  /// Class that does dot products via OpenCL
  TDotProductCL = class (TDotProductKernel)
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
      constructor Create(pCurrentPlatform: cl_platform_id; pCurrentDevice: cl_device_id; kernelname: string = 'cai_dot_product');
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

constructor TDotProductSharedKernel.Create(DotProductKernel: TDotProductKernel);
begin
  inherited Create();
  FDotProductKernel := DotProductKernel;
  FHostInput := False;
end;

destructor TDotProductSharedKernel.Destroy();
begin
  UnprepareForCompute();
  inherited Destroy();
end;

procedure TDotProductSharedKernel.UnprepareForCompute();
begin
  if Assigned(FInputBufferAs) then clReleaseMemObject(FInputBufferAs);
  if Assigned(FInputBufferBs) then clReleaseMemObject(FInputBufferBs);
  if Assigned(FResultBuffer)  then clReleaseMemObject(FResultBuffer);

  FInputBufferAs := nil;
  FInputBufferBs := nil;
  FResultBuffer  := nil;
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

  FResultBuffer  := FDotProductKernel.CreateOutputBuffer(FNumAs * FNumBs * SizeOf(TNeuralFloat));
  FPreviousComputeTime := 0;

  PrepareForCompute := CL_SUCCESS;
end;

procedure TDotProductSharedKernel.Compute
(
  VAs, VBs: TNNetVolume;
  pActFN: longint;
  NewVAs:boolean = true; NewVBs:boolean = true
);
var
  err: integer;
begin
  FActFun := pActFN;

  if (VAs.Size = FSize * FNumAs) then
  begin
    if (VBs.Size = FSize * FNumBs) then
    begin
      err := clSetKernelArg(Kernel, 0, SizeOf(longint), @FThreadCount);
      if (err <> CL_SUCCESS) then ErrorProc('0 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(Kernel, 1, SizeOf(longint), @FNumAs);
      if (err <> CL_SUCCESS) then ErrorProc('1 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(Kernel, 2, SizeOf(longint), @FNumBs);
      if (err <> CL_SUCCESS) then ErrorProc('2 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(Kernel, 3, SizeOf(longint), @FSize);
      if (err <> CL_SUCCESS) then ErrorProc('3 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(Kernel, 4, SizeOf(longint), @FActFun);
      if (err <> CL_SUCCESS) then ErrorProc('4 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(Kernel, 5, SizeOf(cl_mem),  @FInputBufferAs);
      if (err <> CL_SUCCESS) then ErrorProc('5 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(Kernel, 6, SizeOf(cl_mem),  @FInputBufferBs);
      if (err <> CL_SUCCESS) then ErrorProc('6 Error: Failed to set kernel arguments:' + IntToStr(err));

      err := err or clSetKernelArg(Kernel, 7, SizeOf(cl_mem),  @FResultBuffer);
      if (err <> CL_SUCCESS) then ErrorProc('7 Error: Failed to set kernel arguments:' + IntToStr(err));

      if (FHostInput) then
      begin
        //TODO: Fix this refresh.
        //if NewVAs then err := err or FDotProductKernel.RefreshHostInputBufferCache(FInputBufferAs, VAs.GetMemSize());
        //if NewVBs then err := err or FDotProductKernel.RefreshHostInputBufferCache(FInputBufferBs, VBs.GetMemSize())
        if NewVAs then err := err or FDotProductKernel.WriteBuffer(FInputBufferAs, VAs);
        if NewVBs then err := err or FDotProductKernel.WriteBuffer(FInputBufferBs, VBs);
      end
      else
      begin
        if NewVAs then err := err or FDotProductKernel.WriteBuffer(FInputBufferAs, VAs);
        if NewVBs then err := err or FDotProductKernel.WriteBuffer(FInputBufferBs, VBs);
      end;

      if (err <> CL_SUCCESS) then ErrorProc('Failed at WriteBuffer(input):' + IntToStr(err));

      err := err or clSetKernelArg(Kernel, 4, SizeOf(longint), @FActFun);
      if (err <> CL_SUCCESS) then ErrorProc('Failed at clSetKernelArg 4:' + IntToStr(err));

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

constructor TNeuralKernel.Create(pCurrentPlatform: cl_platform_id;
  pCurrentDevice: cl_device_id; kernelname: string = 'cai_dot_product');
begin
  inherited Create();
  SetCurrentPlatform(pCurrentPlatform);
  SetCurrentDevice(pCurrentDevice);

  // Create the OpenCL Kernel Here:
  if FileExists('../../../neural/neural.cl') then
  begin
    CompileProgramFromFile('../../../neural/neural.cl');
  end
  else if FileExists('neural.cl') then
  begin
    CompileProgramFromFile('neural.cl');
  end
  else
  begin
    MessageProc('File neural.cl could not be found.');
  end;
  PrepareKernel(kernelname);
end;

destructor TNeuralKernel.Destroy();
begin
  UnprepareKernel();
  inherited Destroy();
end;

{ TDotProductCL }
constructor TDotProductCL.Create(pCurrentPlatform: cl_platform_id; pCurrentDevice: cl_device_id; kernelname: string = 'cai_dot_product');
begin
  inherited Create(pCurrentPlatform, pCurrentDevice, kernelname);
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
  FResultBuffer  := CreateOutputBuffer(FNumAs * FNumBs * SizeOf(TNeuralFloat));
  FPreviousComputeTime := 0;

  err := PrepareKernel(kernelname);

  err := err or clSetKernelArg(FKernel, 0, SizeOf(longint), @FThreadCount);
  if (err <> CL_SUCCESS) then ErrorProc('0 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 1, SizeOf(longint), @FNumAs);
  if (err <> CL_SUCCESS) then ErrorProc('1 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 2, SizeOf(longint), @FNumBs);
  if (err <> CL_SUCCESS) then ErrorProc('2 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 3, SizeOf(longint), @FSize);
  if (err <> CL_SUCCESS) then ErrorProc('3 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 4, SizeOf(longint), @FActFun);
  if (err <> CL_SUCCESS) then ErrorProc('4 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 5, SizeOf(cl_mem),  @FInputBufferAs);
  if (err <> CL_SUCCESS) then ErrorProc('5 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 6, SizeOf(cl_mem),  @FInputBufferBs);
  if (err <> CL_SUCCESS) then ErrorProc('6 Error: Failed to set kernel arguments:' + IntToStr(err));

  err := err or clSetKernelArg(FKernel, 7, SizeOf(cl_mem),  @FResultBuffer);
  if (err <> CL_SUCCESS) then ErrorProc('7 Error: Failed to set kernel arguments:' + IntToStr(err));

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

      err := err or clSetKernelArg(FKernel, 4, SizeOf(longint), @FActFun);

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
  clSetKernelArg(kernel, arg_index, sizeof(cl_mem), @Result);
end;

function TEasyOpenCLV.CreateOutputSetArgument(V: TNNetVolume;
  kernel: cl_kernel; arg_index: cl_uint): cl_mem;
begin
  Result := CreateOutputBuffer(V);
  clSetKernelArg(kernel, arg_index, sizeof(cl_mem), @Result);
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
    for i := 0 to local_platforms - 1 do
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
  if Assigned(FProg) then clReleaseProgram(FProg);
  if Assigned(FCommands) then clReleaseCommandQueue(FCommands);
  if Assigned(FContext) then clReleaseContext(FContext);
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
  FContext := clCreateContext(nil, 1, @FCurrentDevice, nil, nil, {$IFDEF FPC}err{$ELSE}@err{$ENDIF});

  if FContext = nil then
  begin
    FErrorProc('Error: Failed to create a compute context:' + IntToStr(err));
    exit;
  end
  else
    FMessageProc('clCreateContext OK!');

  // Create a command commands
  FCommands := clCreateCommandQueue(context, FCurrentDevice, 0,  {$IFDEF FPC}err{$ELSE}@err{$ENDIF});
  if FCommands = nil then
  begin
    FErrorProc('Error: Failed to create a command commands:' + IntToStr(err));
    exit;
  end
  else
    FMessageProc('clCreateCommandQueue OK!');

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
begin
  bufwritten := 0;
  if GetPlatformCount()>0 then
  begin
    for i := Low(FPlatformIds) to High(FPlatformIds) do
    begin
      FMessageProc('Platform info: ' + IntToStr(i) + ' ---------------------');
      for k := low(platform_str_info) to high(platform_str_info) do
      begin
        clGetPlatformInfo(FPlatformIds[i], platform_str_info[k].id, sizeof(buf), @buf, {$IFDEF FPC}bufwritten{$ELSE}@bufwritten{$ENDIF});
        MessageProc(platform_str_info[k].Name + ': ' + buf);
      end;

      GetDevicesFromPlatform(FPlatformIds[i], local_devices, local_deviceids);

      if Length(local_devices)>0 then
      begin
        for j := Low(local_deviceids) to High(local_deviceids) do
        begin
          MessageProc('Device info: ' + IntToStr(j) + ' ------------');
          for k := low(device_str_info) to high(device_str_info) do
          begin
            clGetDeviceInfo(local_deviceids[j], device_str_info[k].id, sizeof(buf), @buf, {$IFDEF FPC}bufwritten{$ELSE}@bufwritten{$ENDIF});
            MessageProc(device_str_info[k].Name + ': ' + buf);
          end;

          for k := low(device_word_info) to high(device_word_info) do
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
      for j := 0 to local_devices - 1 do
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
    then FErrorProc('ERROR: Invalid work group size.')
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
  dim_sizes[2] := d2size;

  group_sizes[0] := d1groupsize;
  group_sizes[1] := d2groupsize;
  group_sizes[2] := d2groupsize;

  Result := clEnqueueNDRangeKernel(FCommands, pkernel, work_dim, nil, @dim_sizes[0], @group_sizes[0], 0, nil, nil);

  if (Result <> CL_SUCCESS) then
  begin
    if (Result = CL_INVALID_WORK_GROUP_SIZE)
    then FErrorProc('ERROR: Invalid work group size.')
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
