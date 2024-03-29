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

    NN.AddLayer( TNNetInput.Create(32, 32, 3) );
    NN.AddLayer( TNNetConvolutionLinear.Create(64, 5, 2, 1, 1) ).InitSELU().InitBasicPatterns();
    NN.AddLayer( TNNetMaxPool.Create(4) );
    NN.AddLayer( TNNetSELU.Create() );
    NN.AddLayer( TNNetMovingStdNormalization.Create() );
    NN.AddLayer( TNNetConvolutionLinear.Create(64, 3, 1, 1, 1) ).InitSELU();
    NN.AddLayer( TNNetSELU.Create() );
    NN.AddLayer( TNNetConvolutionLinear.Create(64, 3, 1, 1, 1) ).InitSELU();
    NN.AddLayer( TNNetSELU.Create() );
    NN.AddLayer( TNNetConvolutionLinear.Create(64, 3, 1, 1, 1) ).InitSELU();
    NN.AddLayer( TNNetSELU.Create() );
    NN.AddLayer( TNNetConvolutionLinear.Create(64, 3, 1, 1, 1) ).InitSELU();
    NN.AddLayer( TNNetDropout.Create(0.5) );
    NN.AddLayer( TNNetMaxPool.Create(2) );
    NN.AddLayer( TNNetSELU.Create() );
    NN.AddLayer( TNNetFullConnectLinear.Create(10) );
    NN.AddLayer( TNNetSoftMax.Create({SkipBackpropDerivative=}1) );

    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'ImageClassifierSELU';
    NeuralFit.InitialLearningRate := 0.0004; // SELU seems to work better with smaller learning rates.
    NeuralFit.LearningRateDecay := 0.03;
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
