program SeparableConvolution;
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
    if not CheckCIFARFile() then
    begin
      Terminate;
      exit;
    end;

    WriteLn('Creating Neural Network...');
    NN := TNNet.Create();
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionLinear.Create({NumFeatures=}64, {featureSize=}5, {padding=}2, {stride=}1, {SuppressBias=}0).InitBasicPatterns(),
      TNNetMaxPool.Create(4)
    ]);
    NN.AddSeparableConvReLU({NumFeatures=}64,{featureSize=}3,{padding=}1,{stride=}1,{DepthMultiplier=}1,{SuppressBias=}0);
    NN.AddSeparableConvReLU({NumFeatures=}64,{featureSize=}3,{padding=}1,{stride=}1,{DepthMultiplier=}1,{SuppressBias=}0);
    NN.AddSeparableConvReLU({NumFeatures=}64,{featureSize=}3,{padding=}1,{stride=}1,{DepthMultiplier=}1,{SuppressBias=}0);
    NN.AddSeparableConvReLU({NumFeatures=}64,{featureSize=}3,{padding=}1,{stride=}1,{DepthMultiplier=}1,{SuppressBias=}0);
    NN.AddLayer([
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create({SkipBackpropDerivative=}1)
    ]);
    NN.DebugWeights();
    NN.DebugStructure();
    WriteLn('Layers: ', NN.CountLayers());
    WriteLn('Neurons: ', NN.CountNeurons());
    WriteLn('Weights: ', NN.CountWeights());

    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleImageClassifier';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
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
  Application.Title:='CIFAR-10 Separable Convolutions Example';
  Application.Run;
  Application.Free;
end.
