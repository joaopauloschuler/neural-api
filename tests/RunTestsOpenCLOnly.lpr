program RunTestsOpenCLOnly;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}
  cthreads, cmem,
  {$ENDIF}
  Classes, consoletestrunner,
  TestNeuralOpenCL;

type
  TMyTestRunner = class(TTestRunner)
  protected
    function GetShortOpts: string; override;
  end;

function TMyTestRunner.GetShortOpts: string;
begin
  Result := inherited GetShortOpts + 'c';
end;

var
  Application: TMyTestRunner;

begin
  Application := TMyTestRunner.Create(nil);
  Application.Initialize;
  Application.Title := 'CAI Neural API OpenCL Test Suite';
  Application.Run;
  Application.Free;
end.
