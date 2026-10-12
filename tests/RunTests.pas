// This program requires the LazUtils package to run.
program RunTests;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}
  cthreads, cmem,
  {$ENDIF}
  Classes, consoletestrunner,
  TestNeuralVolume, TestNeuralLayers, TestNeuralThread,
  TestNeuralFit, TestNeuralVolumePairs, TestNeuralSamplers,
  TestNeuralLayersExtra, TestNeuralTraining, TestNeuralNumerical,
  TestNeuralScheduler, TestNeuralDecode, TestNeuralPretrained,
  TestNeuralDPO, TestNeuralGRPO, TestNeuralPreference,
  TestNeuralRewardModel,
  TestNeuralKD, TestNeuralNLPMetrics, TestNeuralMinHash, TestNeuralPacking,
  TestNeuralLengthGrouped,
  TestNeuralMaskedLM,
  TestNeuralHFTokenizer, TestNeuralHFHub, TestNeuralNumpy,
  TestNeuralTokenizer, TestNeuralImagePreprocess,
  TestNeuralDiffusion, TestNeuralImageMetrics, TestNeuralAudio,
  TestNeuralAugment, TestNeuralRegistry, TestNeuralCallbacks,
  TestNeuralSWA, TestNeuralFusedSDPA, TestNeuralABFun, TestNeuralReduction,
  TestNeuralBytePrediction{$IFDEF OpenCL}, neuralopencl{$ENDIF};

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
  {$IFDEF OpenCL}
  // Few distinct local sizes keep PoCL's per-size kernel compiles low; the
  // NEURAL_OPENCL_LOCAL_SIZE_CAP environment variable picks another cap.
  if OpenCLLocalSizeCap = 0 then OpenCLLocalSizeCap := 8;
  {$ENDIF}
  Application := TMyTestRunner.Create(nil);
  Application.Initialize;
  Application.Title := 'CAI Neural API Test Suite';
  Application.Run;
  Application.Free;
end.
