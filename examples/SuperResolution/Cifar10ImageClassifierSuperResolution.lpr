program Cifar10ImageClassifierSuperResolution;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math, neuraldatasets,
  neuralfit, usuperresolutionexample;

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
    ImgTrainingVolumes64, ImgValidationVolumes64, ImgTestVolumes64: TNNetVolumeList;
  begin
    ImgTrainingVolumes64 := TNNetVolumeList.Create();
    ImgValidationVolumes64 := TNNetVolumeList.Create();
    ImgTestVolumes64 := TNNetVolumeList.Create();
    if not CheckCIFARFile() then
    begin
      Terminate;
      exit;
    end;
    WriteLn('Creating neural network for upscaling the dataset.');
    NN := CreateResizingNN(32, 32, csExampleFileName);
    NN.DebugStructure();
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    WriteLn('Upscaling test data.');
    NN.Compute(ImgTestVolumes, ImgTestVolumes64);
    ImgTestVolumes.Clear;
    WriteLn('Upscaling validation data.');
    NN.Compute(ImgValidationVolumes, ImgValidationVolumes64);
    ImgValidationVolumes.Clear;
    WriteLn('Upscaling training data.');
    NN.Compute(ImgTrainingVolumes, ImgTrainingVolumes64);
    ImgTrainingVolumes.Clear;

    NN.Clear();

    WriteLn('Creating neural network for image classification.');
    NN.AddLayer([
      TNNetInput.Create(64, 64, 3),
      TNNetConvolutionLinear.Create(64, 5, 2, 1, 1),
      TNNetMaxPool.Create(4),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(64, 3, 1, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 1, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 1, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 1, 1, 1),
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);
    NN.DebugStructure();

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SuperResolutionImageClassifierResize64';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    //NeuralFit.MaxCropSize := 16;
    //NeuralFit.MaxThreadNum := 16;
    NeuralFit.Fit(NN, ImgTrainingVolumes64, ImgValidationVolumes64, ImgTestVolumes64, {NumClasses=}10, {batchsize=}64, {epochs=}50);
    NeuralFit.Free;

    NN.Free;
    ImgTrainingVolumes64.Free;
    ImgValidationVolumes64.Free;
    ImgTestVolumes64.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    Terminate;
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='CIFAR-10 Classification With Super Resolution';
  Application.Run;
  Application.Free;
end.
