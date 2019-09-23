// Neural Threads
unit neuralthread;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fgl, math, syncobjs;

type
  TNeuronProc = procedure(index, threadnum: integer) of object;

  { TNeuronThread }
  TNeuronThread = class(TThread)
  protected
    FProc: TNeuronProc;
    FShouldStart: boolean;
    FRunning: boolean;
    FProcFinished: boolean;
    FIndex, FThreadNum: integer;
    FNeuronStart: TEventObject;
    FNeuronFinish: TEventObject;

    procedure Execute; override;
  public
    constructor Create(CreateSuspended : boolean; pIndex: integer);
    destructor Destroy(); override;

    procedure StartProc(pProc: TNeuronProc; pIndex, pThreadNum: integer); {$IFDEF Release} inline; {$ENDIF}
    procedure WaitForProc(); {$IFDEF Release} inline; {$ENDIF}
    procedure DoSomething(); {$IFDEF Release} inline; {$ENDIF}
    property ShouldStart: boolean read FShouldStart;
    property Running: boolean read FRunning;
    property ProcFinished: boolean read FProcFinished;
  end;

  { TNeuronThreadList }
  TNeuronThreadList = class (specialize TFPGObjectList<TNeuronThread>)
  private
    FStarted: boolean;
  public
    constructor Create(pSize: integer);

    procedure StartEngine();
    procedure StopEngine();

    procedure StartProc(pProc: TNeuronProc; pBlock: boolean = true);
    procedure WaitForProc(); {$IFDEF Release} inline; {$ENDIF}
  end;

  procedure NeuronThreadListCreate(pSize: integer);
  procedure NeuronThreadListFree();
  function fNTL: TNeuronThreadList; {$IFDEF Release} inline; {$ENDIF}
  procedure CreateNeuronThreadListIfRequired();

implementation

var
  vNTL: TNeuronThreadList;

procedure NeuronThreadListCreate(pSize: integer);
begin
  vNTL := TNeuronThreadList.Create(pSize);
end;

procedure NeuronThreadListFree();
begin
  if Assigned(vNTL) then
  begin
    vNTL.StopEngine();
    vNTL.Free;
    vNTL := nil;
  end;
end;

procedure CreateNeuronThreadListIfRequired();
begin
  if Not(Assigned(vNTL)) then
  begin
    NeuronThreadListCreate(TThread.ProcessorCount);
  end;
end;

function fNTL: TNeuronThreadList;
begin
  Result := vNTL;
end;

{ TNeuronThreadList }
constructor TNeuronThreadList.Create(pSize: integer);
var
  I: integer;
begin
  inherited Create(true);

  FStarted := false;

  for I := 1 to pSize do
  begin
    Self.Add( TNeuronThread.Create(true, I));
  end;
end;

procedure TNeuronThreadList.StartEngine();
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

procedure TNeuronThreadList.StopEngine();
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

procedure TNeuronThreadList.StartProc(pProc: TNeuronProc; pBlock: boolean = true);
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

procedure TNeuronThreadList.WaitForProc();
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

{ TNeuronThread }
procedure TNeuronThread.Execute;
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

constructor TNeuronThread.Create(CreateSuspended: boolean; pIndex: integer);
begin
  inherited Create(CreateSuspended);
  FProc := nil;
  FIndex := pIndex;
  FThreadNum := 1;
  FShouldStart := false;
  FProcFinished := false;
  FNeuronStart := TEventObject.Create(nil, True, False, 'NStart '+IntToStr(pIndex)) ;
  FNeuronFinish := TEventObject.Create(nil, True, False, 'NFinish '+IntToStr(pIndex)) ;
end;

destructor TNeuronThread.Destroy();
begin
  FNeuronStart.Free;
  FNeuronFinish.Free;
  inherited Destroy();
end;

procedure TNeuronThread.StartProc(pProc: TNeuronProc; pIndex,
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

procedure TNeuronThread.WaitForProc();
begin
  FNeuronFinish.WaitFor(INFINITE);
  FNeuronFinish.ResetEvent;
end;

procedure TNeuronThread.DoSomething();
begin
  FNeuronStart.SetEvent;
end;

initialization
vNTL := nil;

finalization
NeuronThreadListFree();

end.

