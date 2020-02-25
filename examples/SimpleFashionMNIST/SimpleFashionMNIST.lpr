program SimpleFashionMNIST;
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
    NN: TNNet;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
  begin
    if Not(CheckMNISTFile('train', {IsFashion=}true)) or
      Not(CheckMNISTFile('t10k', {IsFashion=}true)) then
    begin
      Terminate;
      exit;
    end;
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    NN.AddLayer([
      TNNetInput.Create(28, 28, 1),
      TNNetConvolutionLinear.Create(64, 5, 2, 1, 1),
      TNNetMaxPool.Create(4),
      TNNetConvolutionReLU.Create(64, 3, 1, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 1, 1, 1),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);
    CreateMNISTVolumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,
      'train', 't10k', {Verbose=}true, {IsFashion=}true);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleFashionMNIST';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.HasFlipX := true;
    NeuralFit.HasFlipY := false;
    NeuralFit.MaxCropSize := 4;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}128, {epochs=}50);
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
  Application.Title:='Simple Fashion MNIST Classification Example';
  Application.Run;
  Application.Free;
end.
