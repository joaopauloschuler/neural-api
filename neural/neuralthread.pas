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
  fgl, MTPCPU
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
    NeuralThreadListCreate(TThread.ProcessorCount);
  end;
end;

function NeuralDefaultThreadCount: integer;
begin
  {$IFDEF FPC}
  Result := GetSystemThreadCount;
  {$ELSE}
  Result := TThread.ProcessorCount;
  {$ENDIF}
end;

procedure DebugThreadCount;
begin
  WriteLn('CPU threads reported by the operating system: ', NeuralDefaultThreadCount,'.');
end;

{$IFDEF FPC}
function GetProcessId(): integer;
begin
  GetProcessId := {$IFDEF WINDOWS}GetCurrentProcessId(){$ELSE}fpgetppid(){$ENDIF};
end;

{$ELSE}
//TODO: properly implement process ID for delphi
function GetProcessId(): integer;
begin
  GetProcessId := Random(MaxInt);
end;
{$ENDIF}

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
begin
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

