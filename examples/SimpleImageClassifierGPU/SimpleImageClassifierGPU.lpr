program SimpleImageClassifierGPU;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math,
  neuraldatasets, neuralfit, neuralopencl;

type
  TTestCNNAlgo = class(TCustomApplication)
  protected
    procedure DoRun; override;
  end;

  procedure TTestCNNAlgo.DoRun;
  var
    NN: THistoricalNets;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    EasyOpenCL: TEasyOpenCL;
  begin
    if not CheckCIFARFile() then
    begin
      Terminate;
      exit;
    end;

    EasyOpenCL := TEasyOpenCL.Create();
    if EasyOpenCL.GetPlatformCount() = 0 then
    begin
      WriteLn('No OpenCL capable platform has been found.');
      exit;
    end;
    WriteLn('Setting platform to: ', EasyOpenCL.PlatformNames[0]);
    EasyOpenCL.SetCurrentPlatform(EasyOpenCL.PlatformIds[0]);
    if EasyOpenCL.GetDeviceCount() = 0 then
    begin
      WriteLn('No OpenCL capable device has been found for platform ',EasyOpenCL.PlatformNames[0]);
      exit;
    end;
    EasyOpenCL.SetCurrentDevice(EasyOpenCL.Devices[0]);
    WriteLn('Setting device to: ', EasyOpenCL.DeviceNames[0]);
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}5, {Padding=}2, {Stride=}1, {SuppressBias=}1).InitBasicPatterns(),
      TNNetMaxPool.Create(4),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create({SkipBackpropDerivative=}1)
    ]);
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleImageClassifierGPU';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.EnableOpenCL(EasyOpenCL.PlatformIds[0], EasyOpenCL.Devices[0]);
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}128, {epochs=}50);
    NeuralFit.Free;

    EasyOpenCL.Free;
    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    Terminate;
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='CIFAR-10 Classification Example OpenCL';
  Application.Run;
  Application.Free;
end.
