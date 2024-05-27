(*
Neural Threads
Copyright (C) 2021 Joao Paulo Schwarz Schuler

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
(*
USAGE:
This API has easy to use, lightweight and platform independent parallel
processing API methods.

As an example, assuming that you need to run a procedure 10 times in parallel,
you can create 10 thread workers as follows:
FProcs := TNeuralThreadList.Create( 10 );

As an example, this is the procedure that we intend to run in parallel:
procedure MyClassName.RunNNThread(index, threadnum: integer);
begin
  WriteLn('This is thread ',index,' out of ',threadnum,' threads.');
end;

Then, to run the procedure RunNNThread passed as parameter 10 times in parallel, do this:
FProcs.StartProc({$IFDEF FPC}@RunNNThread{$ELSE}RunNNThread{$ENDIF});

You can control the blocking mode (waiting threads to finish
before the program continues) as per declaration:
procedure StartProc(pProc: TNeuralProc; pBlock: boolean = true);

Or, if you prefer, you can specifically say when to wait for threads to finish
as per this example:
FProcs.StartProc({$IFDEF FPC}@RunNNThread{$ELSE}RunNNThread{$ENDIF}, false);
// insert your code here
FProcs.WaitForProc(); // waits until all threads are finished.

When you are done, you should call:
FProcs.Free;
*)

unit neuralthread;
{$include neuralnetwork.inc}

interface

uses
  Classes, SysUtils,
  {$IFDEF FPC}
  fgl {$IFNDEF WINDOWS} , MTPCPU {$ENDIF}
    {$IFDEF WINDOWS}
    ,windows
    {$ELSE}
    ,BaseUnix
    {$ENDIF}
  {$ELSE}
  Generics.Collections, Windows
  {$ENDIF}
  , syncobjs;

type
  TNeuralProc = procedure(index, threadnum: integer) of object;

  { TNeuralThread }
  TNeuralThread = class(TThread)
  protected
    FProc: TNeuralProc;
    FShouldStart: boolean;
    FRunning: boolean;
    FProcFinished: boolean;
    FIndex, FThreadNum: integer;
    FNeuronStart:  {$IFDEF FPC}TEventObject{$ELSE}TEvent {$ENDIF};
    FNeuronFinish: {$IFDEF FPC}TEventObject{$ELSE}TEvent {$ENDIF};

    procedure Execute; override;
  public
    constructor Create(CreateSuspended : boolean; pIndex: integer);
    destructor Destroy(); override;

    procedure StartProc(pProc: TNeuralProc; pIndex, pThreadNum: integer); {$IFDEF Release} inline; {$ENDIF}
    procedure WaitForProc(); {$IFDEF Release} inline; {$ENDIF}
    procedure DoSomething(); {$IFDEF Release} inline; {$ENDIF}
    property ShouldStart: boolean read FShouldStart;
    property Running: boolean read FRunning;
    property ProcFinished: boolean read FProcFinished;
  end;

  { TNeuralThreadList }
  {$IFDEF FPC}
  TNeuralThreadList = class (specialize TFPGObjectList<TNeuralThread>)
  {$ELSE}
  TNeuralThreadList = class (TObjectList<TNeuralThread>)
  {$ENDIF}
  private
    FStarted: boolean;
  public
    constructor Create(pSize: integer);
    destructor Destroy; override;

    class procedure CalculateWorkingRange(pIndex, pThreadnum, pSize: integer; out
      StartPos, FinishPos: integer);
    class function GetRandomNumberOnWorkingRange(pIndex, pThreadnum, pSize: integer): integer; overload;
    class function GetRandomNumberOnWorkingRange(pIndex, pThreadnum, pSize: integer; out MaxLen: integer): integer; overload;

    procedure StartEngine();
    procedure StopEngine();

    procedure StartProc(pProc: TNeuralProc; pBlock: boolean = true);
    procedure WaitForProc(); {$IFDEF Release} inline; {$ENDIF}
  end;

  procedure NeuralThreadListCreate(pSize: integer);
  procedure NeuralThreadListFree();
  function fNTL: TNeuralThreadList; {$IFDEF FPC}{$IFDEF Release} inline; {$ENDIF}{$ENDIF}
  procedure CreateNeuralThreadListIfRequired();
  function NeuralDefaultThreadCount: integer;
  procedure NeuralInitCriticalSection(var pCritSec: TRTLCriticalSection);
  procedure NeuralDoneCriticalSection(var pCritSec: TRTLCriticalSection);
  function GetProcessId(): {$IFDEF FPC}integer{$ELSE}integer{$ENDIF};
  procedure DebugThreadCount();

implementation

procedure NeuralInitCriticalSection(var pCritSec: TRTLCriticalSection);
begin
  {$IFDEF FPC}
  InitCriticalSection(pCritSec);
  {$ELSE}
  InitializeCriticalSection(pCritSec);
  {$ENDIF}
end;

procedure NeuralDoneCriticalSection(var pCritSec: TRTLCriticalSection);
begin
  {$IFDEF FPC}
  DoneCriticalsection(pCritSec);
  {$ELSE}
  DeleteCriticalSection(pCritSec);
  {$ENDIF}
end;

var
  vNTL: TNeuralThreadList;

procedure NeuralThreadListCreate(pSize: integer);
begin
  vNTL := TNeuralThreadList.Create(pSize);
end;

procedure NeuralThreadListFree();
begin
  if Assigned(vNTL) then
  begin
    vNTL.StopEngine();
    vNTL.Free;
    vNTL := nil;
  end;
end;

procedure CreateNeuralThreadListIfRequired();
begin
  if Not(Assigned(vNTL)) then
  begin
    NeuralThreadListCreate(System.CPUCount);
  end;
end;

{$UNDEF IMPORT_AFFINITAPI}
{$IFDEF MSWINDOWS}
{$IFNDEF FPC}
{$IF CompilerVersion <= 23}
{$DEFINE IMPORT_AFFINITAPI}
{$IFEND}
{$ELSE}
{$DEFINE IMPORT_AFFINITAPI}
{$ENDIF}
{$ENDIF}


{$IFDEF IMPORT_AFFINITAPI}
// delphi 2010 does not define the following functions and constants

const ALL_PROCESSOR_GROUPS = $ffff;

  //
  // Structure to represent a group-specific affinity, such as that of a
  // thread.  Specifies the group number and the affinity within that group.
  //
type
  KAFFINITY = ULONG_PTR;
  _GROUP_AFFINITY = record
      Mask: KAFFINITY;
      Group: WORD;
      Reserved: array[0..2] of WORD;
  end;
  {$EXTERNALSYM _GROUP_AFFINITY}
  GROUP_AFFINITY = _GROUP_AFFINITY;
  {$EXTERNALSYM GROUP_AFFINITY}
  PGROUP_AFFINITY = ^_GROUP_AFFINITY;
  {$EXTERNALSYM PGROUP_AFFINITY}
  TGroupAffinity = _GROUP_AFFINITY;
  PGroupAffinity = PGROUP_AFFINITY;

function GetActiveProcessorCount(GroupNumber: WORD): DWORD; stdcall; external 'kernel32.dll';
function SetThreadGroupAffinity(hThread: THandle; var GroupAffinity: TGroupAffinity;
  PreviousGroupAffinity: PGroupAffinity): ByteBool; stdcall; external kernel32 name 'GetThreadGroupAffinity';
function GetThreadGroupAffinity(hThread: THandle; var GroupAffinity: TGroupAffinity): ByteBool; stdcall; external kernel32 name 'GetThreadGroupAffinity';
function GetActiveProcessorGroupCount: WORD; stdcall; external kernel32 name 'GetThreadGroupAffinity';
{$ENDIF}

function NeuralDefaultThreadCount: integer;
begin
  {$IFDEF MSWINDOWS}
  // for systems with more than 64 cores thes System.CPUCount only returns 64 at max
  // -> we need to group them together so count the cpu's differntly
  // https://learn.microsoft.com/en-us/windows/win32/procthread/processor-groups
  Result := GetActiveProcessorCount(ALL_PROCESSOR_GROUPS); // get all of all groups
  {$ELSE}
  {$IFDEF FPC}
  Result := GetSystemThreadCount;
  {$ELSE}
  Result := System.CPUCount;
  {$ENDIF}
  {$ENDIF}
end;

{$IFDEF FPC}
function GetProcessId(): integer;
begin
  GetProcessId := {$IFDEF WINDOWS}GetCurrentProcessId(){$ELSE}fpgetppid(){$ENDIF};
end;
{$ELSE}
function GetProcessId(): integer;
begin
  {$IFDEF WINDOWS}
  Result := GetCurrentProcessId()
  {$ELSE}
  //TODO: properly implement process ID for non windows delphi
  GetProcessId := Random(MaxInt);
  {$ENDIF}
end;
{$ENDIF}

procedure DebugThreadCount;
begin
  WriteLn('CPU threads reported by the operating system: ', NeuralDefaultThreadCount,'.');
end;

function fNTL: TNeuralThreadList;
begin
  Result := vNTL;
end;

{ TNeuralThreadList }
constructor TNeuralThreadList.Create(pSize: integer);
var
  I: integer;
begin
  inherited Create(true);

  FStarted := false;

  for I := 1 to pSize do
  begin
    Self.Add( TNeuralThread.Create(true, I));
  end;
end;

destructor TNeuralThreadList.Destroy;
begin
  StopEngine();
  inherited Destroy;
end;

class procedure TNeuralThreadList.CalculateWorkingRange(pIndex, pThreadnum, pSize: integer; out StartPos, FinishPos: integer);
var
  BlockSize: integer;
begin
  BlockSize := pSize div pThreadnum {$IFDEF MakeQuick}div 10{$ENDIF};
  StartPos  := BlockSize * pIndex;
  FinishPos := BlockSize * (pIndex + 1) - 1;

  if pIndex = (pThreadnum-1) then
  begin
    FinishPos := pSize - 1;
  end;
end;

class function TNeuralThreadList.GetRandomNumberOnWorkingRange(pIndex,
  pThreadnum, pSize: integer): integer;
var
  StartPos, FinishPos: integer;
begin
  CalculateWorkingRange(pIndex, pThreadnum, pSize, StartPos, FinishPos);
  Result := StartPos + Random(FinishPos - StartPos + 1);
end;

class function TNeuralThreadList.GetRandomNumberOnWorkingRange(pIndex,
  pThreadnum, pSize: integer; out MaxLen: integer): integer;
var
  StartPos, FinishPos: integer;
begin
  CalculateWorkingRange(pIndex, pThreadnum, pSize, StartPos, FinishPos);
  Result := StartPos + Random(FinishPos - StartPos + 1);
  MaxLen := FinishPos - Result;
end;

procedure TNeuralThreadList.StartEngine();
var
  I: integer;
  localCount: integer;
begin
  if not(FStarted) then
  begin
    localCount := Count;
    for I := 0 to localCount - 1 do
    begin
      Self.Items[I].Start();
    end;
    FStarted := true;
  end;
end;

procedure TNeuralThreadList.StopEngine();
var
  I: integer;
  localCount: integer;
begin
  if (FStarted) then
  begin
    localCount := Count;
    for I := 0 to localCount - 1 do
    begin
      Self.Items[I].Terminate();
      Self.Items[I].DoSomething();
    end;

    for I := 0 to localCount - 1 do
    begin
      Self.Items[I].DoSomething();
      Self.Items[I].WaitFor();
    end;
    FStarted := false;
  end;
end;

procedure TNeuralThreadList.StartProc(pProc: TNeuralProc; pBlock: boolean = true);
var
  I: integer;
  localCount: integer;
begin
  localCount := Count;

  if localCount = 1 then
  begin
    pProc(0,1);
  end
  else
  begin
    if not(FStarted) then StartEngine();
    for I := 0 to localCount - 1 do
    begin
      Self.Items[I].StartProc(pProc, I, localCount);
    end;

    if pBlock then WaitForProc();
  end;
end;

procedure TNeuralThreadList.WaitForProc();
var
  I: integer;
  MaxCount: integer;
begin
  MaxCount := Count - 1;
  for I := 0 to MaxCount do
  begin
    Self.Items[I].WaitForProc();
  end;
end;

{ TNeuralThread }
procedure TNeuralThread.Execute;
{$IFDEF MSWINDOWS}

var i : integer;
    numGroups : integer;
    maxIdxInGroup : integer;
    ga : TGroupAffinity;
{$ENDIF}
begin
  {$IFDEF MSWINDOWS}
  // set group affinity
  maxIdxInGroup := -1;
  numGroups := GetActiveProcessorGroupCount;

  // set affinity to physical cpu's - leave it as is otherwise
  for i := 0 to numGroups - 1 do
  begin
       maxIdxInGroup := maxIdxInGroup + Integer(GetActiveProcessorCount(i));
       if maxIdxInGroup >= FIndex then
       begin
            FillChar( ga, sizeof(ga), 0);
            GetThreadGroupAffinity(GetCurrentThread, ga);
            ga.Group := Word(i);
            if not SetThreadGroupAffinity( Handle, ga, nil) then
               RaiseLastOSError;

            break;
       end;
  end;
  {$ENDIF}



  // ###########################################
  // #### do the work
  while (not Terminated) do
  begin
    FNeuronStart.WaitFor(INFINITE);
    FNeuronStart.ResetEvent;
    if (FShouldStart) then
    begin
      FRunning := true;
      FShouldStart := false;

      FProc(FIndex, FThreadNum);

      FRunning := false;
      FNeuronFinish.SetEvent;
      FProcFinished := true;
    end;
  end;
end;

constructor TNeuralThread.Create(CreateSuspended: boolean; pIndex: integer);
var
  NStartName, NFinishName: string;
  PidAndIndexStr: string;
begin
  inherited Create(CreateSuspended);
  FProc := nil;
  FIndex := pIndex;
  FThreadNum := 1;
  FShouldStart := false;
  FProcFinished := false;
  PidAndIndexStr := IntToStr(GetProcessId())+'-'+IntToStr(pIndex);
  NStartName := 'NStart-'+PidAndIndexStr;
  NFinishName := 'NFinish-'+PidAndIndexStr;
  {$IFDEF FPC}
  FNeuronStart := TEventObject.Create(nil, True, False, NStartName) ;
  FNeuronFinish := TEventObject.Create(nil, True, False, NFinishName) ;
  {$ELSE}
  FNeuronStart := TEvent.Create(nil, True, False, NStartName) ;
  FNeuronFinish := TEvent.Create(nil, True, False, NFinishName) ;
  {$ENDIF}
end;

destructor TNeuralThread.Destroy();
begin
  FNeuronStart.Free;
  FNeuronFinish.Free;
  inherited Destroy();
end;

procedure TNeuralThread.StartProc(pProc: TNeuralProc; pIndex,
  pThreadNum: integer);
begin
  FNeuronStart.ResetEvent;
  FNeuronFinish.ResetEvent;
  FProcFinished := false;
  FProc := pProc;
  FIndex := pIndex;
  FThreadNum := pThreadNum;
  FShouldStart := true;
  FNeuronStart.SetEvent;
end;

procedure TNeuralThread.WaitForProc();
begin
  FNeuronFinish.WaitFor(INFINITE);
  FNeuronFinish.ResetEvent;
end;

procedure TNeuralThread.DoSomething();
begin
  FNeuronStart.SetEvent;
end;

initialization
vNTL := nil;

finalization
NeuralThreadListFree();

end.

