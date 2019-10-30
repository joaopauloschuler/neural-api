program ImageClassifierSELU;
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
    if not CheckCIFARFile() then
    begin
      Terminate;
      exit;
    end;
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionLinear.Create(64, 5, 2, 1, 1).InitBasicPatterns(),
      TNNetMaxPool.Create(4),
      TNNetSELU.Create(),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionLinear.Create(64, 3, 1, 1, 1).InitSELU(),
      TNNetSELU.Create(),
      TNNetConvolutionLinear.Create(64, 3, 1, 1, 1).InitSELU(),
      TNNetSELU.Create(),
      TNNetConvolutionLinear.Create(64, 3, 1, 1, 1).InitSELU(),
      TNNetSELU.Create(),
      TNNetConvolutionLinear.Create(64, 3, 1, 1, 1).InitSELU(),
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetSELU.Create(),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'ImageClassifierSELU';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NN.DebugWeights();
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}64, {epochs=}50);
    NeuralFit.Free;

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
  Application.Title:='CIFAR-10 SELU Classification Example';
  Application.Run;
  Application.Free;
end.
