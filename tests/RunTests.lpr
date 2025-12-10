program RunTests;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}
  cthreads, cmem,
  {$ENDIF}
  Classes, consoletestrunner,
  TestNeuralVolume, TestNeuralLayers, TestNeuralThread,
  TestNeuralFit, TestNeuralVolumePairs;

type
  TMyTestRunner = class(TTestRunner)
  protected
    function GetShortOpts: string; override;
  end;

function TMyTestRunner.GetShortOpts: string;
begin
  Result := inherited GetShortOpts + 'x';
end;

var
  Application: TMyTestRunner;

begin
  Application := TMyTestRunner.Create(nil);
  Application.Initialize;
  Application.Title := 'CAI Neural API Test Suite';
  Application.Run;
  Application.Free;
end.
