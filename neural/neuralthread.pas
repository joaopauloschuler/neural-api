// Neural Threads
unit neuralthread;
{$include neuralnetwork.inc}

interface

uses
  Classes, SysUtils,
  {$IFDEF FPC}
  fgl, MTPCPU
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

    procedure StartEngine();
    procedure StopEngine();

    procedure StartProc(pProc: TNeuralProc; pBlock: boolean = true);
    procedure WaitForProc(); {$IFDEF Release} inline; {$ENDIF}
  end;

  procedure NeuralThreadListCreate(pSize: integer);
  procedure NeuralThreadListFree();
  function fNTL: TNeuralThreadList; {$IFDEF Release} inline; {$ENDIF}
  procedure CreateNeuralThreadListIfRequired();
  function NeuralDefaultThreadCount: integer;
  procedure NeuralInitCriticalSection(var pCritSec: TRTLCriticalSection);
  procedure NeuralDoneCriticalSection(var pCritSec: TRTLCriticalSection);

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
begin
  inherited Create(CreateSuspended);
  FProc := nil;
  FIndex := pIndex;
  FThreadNum := 1;
  FShouldStart := false;
  FProcFinished := false;
  {$IFDEF FPC}
  FNeuronStart := TEventObject.Create(nil, True, False, 'NStart '+IntToStr(pIndex)) ;
  FNeuronFinish := TEventObject.Create(nil, True, False, 'NFinish '+IntToStr(pIndex)) ;
  {$ELSE}
  FNeuronStart := TEvent.Create(nil, True, False, 'NStart '+IntToStr(pIndex)) ;
  FNeuronFinish := TEvent.Create(nil, True, False, 'NFinish '+IntToStr(pIndex)) ;
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

