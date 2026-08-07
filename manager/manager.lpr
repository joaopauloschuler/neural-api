program manager;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  {$IFDEF WINDOWS}
  Windows,
  {$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms, mainform;

{$R *.res}

{$IFDEF WINDOWS}
var
  hMutex: HANDLE;
{$ENDIF}

begin
  {$IFDEF WINDOWS}
  hMutex := CreateMutex(nil, True, 'ChatServerManager_SingleInstance_Mutex');
  if (hMutex <> 0) and (GetLastError = ERROR_ALREADY_EXISTS) then
  begin
    CloseHandle(hMutex);
    Halt(0);
  end;
  {$ENDIF}

  RequireDerivedFormResource:=True;
  Application.Scaled:=True;
  Application.Initialize;
  Application.CreateForm(TfrmManager, frmManager);
  Application.Run;
end.
