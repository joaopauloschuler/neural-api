program SimpleImageClassifier;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math, neuraldatasets, neuralfit;

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
  begin
    if not CheckCIFARFile() then exit;
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionLinear.Create(64, 5, 2, 1, 1).InitBasicPatterns(),
      TNNetMaxPool.Create(4),
      TNNetConvolutionReLU.Create(64, 3, 0, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 0, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 0, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 0, 1, 1),
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleImageClassifier';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.995;
    NeuralFit.StaircaseEpochs := 17;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}128, {epochs=}100);
    NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='CIFAR-10 Simple Image Classifier Example';
  Application.Run;
  Application.Free;
end.
