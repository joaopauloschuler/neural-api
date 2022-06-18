program CaiResNet20;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume,
  Math, neuraldatasets, neuralfit, neuralthread;

type
  TTestCNNAlgo = class(TCustomApplication)
  protected
    procedure DoRun; override;
  end;

  const
  // Padding and cropping constants.
  csPadding = 4;
  csCropSize = csPadding * 2;

procedure CaiOptimizedResnetUnit(pNN: TNNet; pNeurons: integer);
var
  PreviousLayer, ShortCut, LongPath: TNNetLayer;
  Stride: integer;
begin
  PreviousLayer := pNN.GetLastLayer();
  if PreviousLayer.Output.Depth = pNeurons
    then Stride := 1
    else Stride := 2;
  LongPath := pNN.AddLayer([
    TNNetConvolutionLinear.Create(pNeurons, {featuresize}3, {padding}1, Stride),
    TNNetReLU.Create(),
    TNNetConvolutionLinear.Create(pNeurons, {featuresize}3, {padding}1, {stride}1),
    TNNetReLUL.Create(-3, 3, 0)
  ]);
  if PreviousLayer.Output.Depth = pNeurons then
  begin
    pNN.AddLayer( TNNetSum.Create([PreviousLayer, LongPath]) );
  end
  else
  begin
    ShortCut := pNN.AddLayerAfter([
      TNNetConvolutionReLU.Create(pNeurons, {featuresize}3, {padding}1, Stride)
    ], PreviousLayer);
    pNN.AddLayer( TNNetSum.Create([ShortCut, LongPath]) );
  end;
end;

  procedure TTestCNNAlgo.DoRun;
  var
    NN: THistoricalNets;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    ModuleCount, ModuleNumber: integer;
  begin
    ModuleNumber := 3;
    if not CheckCIFARFile() then
    begin
      Terminate;
      exit;
    end;
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    NN.AddLayer(TNNetInput.Create(32, 32, 3));
    NN.AddLayer([
      TNNetConvolutionLinear.Create({neurons=}16, {featuresize}3, {padding}1, {stride}1),
      TNNetReLU6.Create()
    ]);
    for ModuleCount := 1 to ModuleNumber do CaiOptimizedResnetUnit(NN, 16);
    for ModuleCount := 1 to ModuleNumber do CaiOptimizedResnetUnit(NN, 32);
    for ModuleCount := 1 to ModuleNumber do CaiOptimizedResnetUnit(NN, 64);

    NN.AddLayer([
      TNNetAvgChannel.Create(),
      TNNetFullConnectLinear.Create(10), // The original implementation uses avg pooling.
      TNNetSoftMax.Create()
    ]);

    NN.DebugStructure();
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes,
      ImgTestVolumes, csEncodeRGB, {ValidationSampleSize=}2000);

    // Add padding to dataset
    WriteLn
    (
      'Original image size: ',
      ImgTrainingVolumes[0].SizeX,',',
      ImgTrainingVolumes[0].SizeY,' px.'
    );
    ImgTrainingVolumes.AddPadding(csPadding);
    ImgValidationVolumes.AddPadding(csPadding);
    ImgTestVolumes.AddPadding(csPadding);
    WriteLn
    (
      'New image size after padding: ',
      ImgTrainingVolumes[0].SizeX,',',
      ImgTrainingVolumes[0].SizeY,' px.'
    );

    NeuralFit := TNeuralImageFit.Create;
    // Enable cropping while fitting.
    NeuralFit.HasImgCrop := true;
    NeuralFit.MaxCropSize := csCropSize;
    NeuralFit.FileNameBase := 'SimpleImageClassifier-'+IntToStr(GetProcessId());
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}32, {epochs=}50);
    NeuralFit.Free;
    ReadLn();
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
  Application.Title:='CIFAR-10 Classification Example';
  Application.Run;
  Application.Free;
end.
