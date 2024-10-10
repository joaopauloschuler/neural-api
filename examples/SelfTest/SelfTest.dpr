program SelfTest;

{$APPTYPE CONSOLE}

uses
  Classes,
  SysUtils,
  Math,
  CPUFeatures in '..\..\neural\CPUFeatures.pas',
  neuralab in '..\..\neural\neuralab.pas',
  neuralabfun in '..\..\neural\neuralabfun.pas',
  neuralbit in '..\..\neural\neuralbit.pas',
  neuralbyteprediction in '..\..\neural\neuralbyteprediction.pas',
  neuralcache in '..\..\neural\neuralcache.pas',
  neuraldatasets in '..\..\neural\neuraldatasets.pas',
  neuraldatasetsv in '..\..\neural\neuraldatasetsv.pas',
  neuralevolutionary in '..\..\neural\neuralevolutionary.pas',
  neuralfit in '..\..\neural\neuralfit.pas',
  neuralgeneric in '..\..\neural\neuralgeneric.pas',
  neuralnetwork in '..\..\neural\neuralnetwork.pas',
  neuralopencl in '..\..\neural\neuralopencl.pas',
  neuralopenclv in '..\..\neural\neuralopenclv.pas',
  neuralplanbuilder in '..\..\neural\neuralplanbuilder.pas',
  neuralthread in '..\..\neural\neuralthread.pas',
  neuralvolume in '..\..\neural\neuralvolume.pas',
  neuralvolumev in '..\..\neural\neuralvolumev.pas',
  NeuralAVX in '..\..\neural\NeuralAVX.pas',
  NeuralAVXx64 in '..\..\neural\NeuralAVXx64.pas',
  neuralavxcore in '..\..\neural\neuralavxcore.pas';

begin
    Writeln('Testing Delphi AVX...');
    TestAVX;

    WriteLn('Testing Volumes API ...');
    TestTNNetVolume();
    TestKMeans();

    WriteLn('Testing Convolutional API ...');
    TestConvolutionAPI;

    WriteLn('Press ENTER to quit.');
    ReadLn;
end.
