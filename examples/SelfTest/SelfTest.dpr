program SelfTest;

{$APPTYPE CONSOLE}

uses
  Classes,
  SysUtils,
  Math,
  NeuralAVX in '..\..\Neural\NeuralAVX.pas',
  NeuralAVXx64 in '..\..\Neural\NeuralAVXx64.pas',
  neuralab in '..\..\Neural\neuralab.pas',
  neuralabfun in '..\..\Neural\neuralabfun.pas',
  neuralbit in '..\..\Neural\neuralbit.pas',
  neuralbyteprediction in '..\..\Neural\neuralbyteprediction.pas',
  neuralcache in '..\..\Neural\neuralcache.pas',
  neuraldatasets in '..\..\Neural\neuraldatasets.pas',
  neuraldatasetsv in '..\..\Neural\neuraldatasetsv.pas',
  neuralevolutionary in '..\..\Neural\neuralevolutionary.pas',
  neuralfit in '..\..\Neural\neuralfit.pas',
  neuralgeneric in '..\..\Neural\neuralgeneric.pas',
  neuralnetwork in '..\..\Neural\neuralnetwork.pas',
  neuralopencl in '..\..\Neural\neuralopencl.pas',
  neuralopenclv in '..\..\Neural\neuralopenclv.pas',
  neuralplanbuilder in '..\..\Neural\neuralplanbuilder.pas',
  neuralthread in '..\..\Neural\neuralthread.pas',
  neuralvolume in '..\..\Neural\neuralvolume.pas',
  neuralvolumev in '..\..\Neural\neuralvolumev.pas';

begin
     Writeln('Test AVX');
     TestAVX;

     WriteLn('Testing Volumes API ...');
     TestTNNetVolume();
     TestKMeans();

     WriteLn('Testing Convolutional API ...');
     TestConvolutionAPI();

     WriteLn('Press ENTER to quit.');
     ReadLn;
end.
