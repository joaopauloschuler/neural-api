program SelfTest;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume,
  Math, neuraldatasets, neuralfit, neuralthread;

type
  TTestCNNAlgo = class(TCustomApplication)
  protected
    procedure DoRun; override;
  end;

  procedure TTestCNNAlgo.DoRun;
  begin
    WriteLn('Testing Volumes API ...');
    TestTNNetVolume();
    TestKMeans();

    WriteLn('Testing Convolutional API ...');
    TestConvolutionAPI();

    WriteLn('Press ENTER to quit.');
    ReadLn();
    Terminate;
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='Neural API self-test';
  Application.Run;
  Application.Free;
end.
