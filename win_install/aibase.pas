unit aibase;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

const
  AI_SUITE_VERSION = '1.10.0';

type
  TAIComponentCategory = (
    ccInput,
    ccOutput,
    ccAction,
    ccModel,
    ccProject,
    ccSafety,
    ccSimulation,
    ccOther
  );

  TAILogLevel = (llDebug, llInfo, llWarning, llError);

  TAILogEvent = procedure(
    Sender: TObject;
    Level: TAILogLevel;
    const Message: string
  ) of object;

  { TAIBaseComponent }

  TAIBaseComponent = class(TComponent)
  protected
    FPrompt: string;
    FLastError: string;
    FLastResult: string;
    FLastSuccess: Boolean;
    FCategory: TAIComponentCategory;
    FOnLog: TAILogEvent;
    
    procedure ClearError;
    procedure SetError(const AMessage: string);
    procedure Log(ALevel: TAILogLevel; const AMessage: string); virtual;
  public
    constructor Create(AOwner: TComponent); override;
    property LastSuccess: Boolean read FLastSuccess;
  published
    property Prompt: string read FPrompt write FPrompt;
    property LastError: string read FLastError;
    property LastResult: string read FLastResult;
    property Category: TAIComponentCategory read FCategory write FCategory default ccOther;
    property OnLog: TAILogEvent read FOnLog write FOnLog;
  end;

function GetAISuiteVersion: string;

implementation

function GetAISuiteVersion: string;
begin
  Result := AI_SUITE_VERSION;
end;

{ TAIBaseComponent }

constructor TAIBaseComponent.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FPrompt := '';
  FLastError := '';
  FLastResult := '';
  FLastSuccess := True;
  FCategory := ccOther;
  FOnLog := nil;
end;

procedure TAIBaseComponent.ClearError;
begin
  FLastError := '';
  FLastSuccess := True;
  Log(llDebug, 'Error cleared.');
end;

procedure TAIBaseComponent.SetError(const AMessage: string);
begin
  FLastError := AMessage;
  FLastSuccess := False;
  Log(llError, AMessage);
end;

procedure TAIBaseComponent.Log(ALevel: TAILogLevel; const AMessage: string);
begin
  if Assigned(FOnLog) then
    FOnLog(Self, ALevel, AMessage);
end;

end.
