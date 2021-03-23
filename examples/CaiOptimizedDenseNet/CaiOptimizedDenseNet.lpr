program CaiOptimizedDenseNet;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
 This command line tool trains a CAI Optimized DenseNet with CIFAR-10.
 This code is inspired on:
 https://github.com/liuzhuang13/DenseNet
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes,
  SysUtils,
  CustApp,
  neuralnetwork,
  neuralvolume,
  Math,
  neuraldatasets,
  neuralfit;

const
  // Padding and cropping constants.
  csPadding = 4;
  csCropSize = csPadding * 2;

type
  TTestCNNAlgo = class(TCustomApplication)
  protected
    fLearningRate, fInertia, fTarget: single;
    bPaddingCropping: boolean;
    bSeparable: boolean;
    iInnerConvNum: integer;
    iBottleneck: integer;
    iConvNeuronCount: integer;
    procedure DoRun; override;
    procedure Train();
  public
    constructor Create(TheOwner: TComponent); override;
    procedure WriteHelp; virtual;
  end;

  procedure TTestCNNAlgo.DoRun;
  var
    ParStr: string;
  begin
    // parse parameters
    if HasOption('h', 'help') then
    begin
      WriteHelp;
      Terminate;
      Exit;
    end;

    fLearningRate := 0.001;
    if HasOption('l', 'learningrate') then
    begin
      ParStr := GetOptionValue('l', 'learningrate');
      fLearningRate := StrToFloat(ParStr);
    end;

    fInertia := 0.9;
    if HasOption('i', 'inertia') then
    begin
      ParStr := GetOptionValue('i', 'inertia');
      fInertia := StrToFloat(ParStr);
    end;

    fTarget := 1;
    if HasOption('t', 'target') then
    begin
      ParStr := GetOptionValue('t', 'target');
      fTarget := StrToFloat(ParStr);
    end;

    bSeparable := HasOption('s', 'separable');
    bPaddingCropping := HasOption('p', 'padding');

    iInnerConvNum := 12;
    if HasOption('c', 'convolutions') then
    begin
      ParStr := GetOptionValue('c', 'convolutions');
      iInnerConvNum := StrToInt(ParStr);
    end;

    iBottleneck := 32;
    if HasOption('b', 'bottleneck') then
    begin
      ParStr := GetOptionValue('b', 'bottleneck');
      iBottleneck := StrToInt(ParStr);
    end;

    iConvNeuronCount := 32;
    if HasOption('n', 'neurons') then
    begin
      ParStr := GetOptionValue('n', 'neurons');
      iConvNeuronCount := StrToInt(ParStr);
    end;

    Train();

    Terminate;
  end;

  procedure TTestCNNAlgo.Train();
  var
    NN: THistoricalNets;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    NumClasses: integer;
    fileNameBase: string;
    HasMovingNorm: boolean;
  begin
    if not CheckCIFARFile() then exit;
    WriteLn('Creating Neural Network...');
    NumClasses  := 10;
    NN := THistoricalNets.Create();
    fileNameBase := 'CaiOptimizedDenseNet';
    HasMovingNorm := true;
    NN.AddLayer( TNNetInput.Create(32, 32, 3).EnableErrorCollection() );
    // First block shouldn't be separable.
    NN.AddDenseNetBlockCAI(iInnerConvNum div 6, iConvNeuronCount, {supressBias=}1, TNNetConvolutionReLU, {IsSeparable=}false, {HasMovingNorm=}HasMovingNorm, {pBeforeNorm=}nil, {pAfterNorm=}nil, {BottleNeck=}iBottleneck, {Compression=}0, {Dropout=}0, {RandomAdd=}1, {RandomMul=}1);
    NN.AddDenseNetBlockCAI(iInnerConvNum div 6, iConvNeuronCount, {supressBias=}1, TNNetConvolutionReLU, {IsSeparable=}bSeparable, {HasMovingNorm=}HasMovingNorm, {pBeforeNorm=}nil, {pAfterNorm=}nil, {BottleNeck=}iBottleneck, {Compression=}0, {Dropout=}0, {RandomAdd=}1, {RandomMul=}1);
    NN.AddCompression(1);
    NN.AddLayer( TNNetMaxPool.Create(2) );
    NN.AddDenseNetBlockCAI(iInnerConvNum div 3, iConvNeuronCount, {supressBias=}1, TNNetConvolutionReLU, {IsSeparable=}bSeparable, {HasMovingNorm=}HasMovingNorm, {pBeforeNorm=}nil, {pAfterNorm=}nil, {BottleNeck=}iBottleneck, {Compression=}0, {Dropout=}0, {RandomAdd=}1, {RandomMul=}1);
    NN.AddCompression(1);
    NN.AddLayer( TNNetMaxPool.Create(2) );
    NN.AddDenseNetBlockCAI(iInnerConvNum div 3, iConvNeuronCount, {IsSeparable=}1, TNNetConvolutionReLU, {IsSeparable=}bSeparable, {HasMovingNorm=}HasMovingNorm, {pBeforeNorm=}nil, {pAfterNorm=}nil, {BottleNeck=}iBottleneck, {Compression=}0, {Dropout=}0, {RandomAdd=}1, {RandomMul=}1);
    NN.AddCompression(1);
    NN.AddLayer( TNNetDropout.Create(0.25) );
    NN.AddLayer( TNNetMaxChannel.Create() );
    NN.AddLayer( TNNetFullConnectLinear.Create(NumClasses) );
    NN.AddLayer( TNNetSoftMax.Create() );
    NN.Layers[ NN.GetFirstImageNeuronalLayerIdx() ].InitBasicPatterns();

    WriteLn('Learning rate set to: [',fLearningRate:7:5,']');
    WriteLn('Inertia set to: [',fInertia:7:5,']');
    WriteLn('Target set to: [',fTarget:7:5,']');

    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    if bPaddingCropping then
    begin
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
    end;

    WriteLn('Neural Network will minimize error with:');
    WriteLn(' Layers: ', NN.CountLayers());
    WriteLn(' Neurons:', NN.CountNeurons());
    WriteLn(' Weights:', NN.CountWeights());
    NN.DebugWeights();
    NN.DebugStructure();

    NeuralFit := TNeuralImageFit.Create;

    if bPaddingCropping then
    begin
      // Enable cropping while fitting.
      NeuralFit.HasImgCrop := true;
      NeuralFit.MaxCropSize := csCropSize;
    end;

    NeuralFit.FileNameBase := fileNameBase;
    NeuralFit.InitialLearningRate := fLearningRate;
    NeuralFit.LearningRateDecay := 0.02;
    NeuralFit.CyclicalLearningRateLen := 100;
    NeuralFit.StaircaseEpochs := 15;
    NeuralFit.Inertia := fInertia;
    NeuralFit.TargetAccuracy := fTarget;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, NumClasses, {batchsize=}64, {epochs=}300);
    NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
  end;

  constructor TTestCNNAlgo.Create(TheOwner: TComponent);
  begin
    inherited Create(TheOwner);
    StopOnException := True;
  end;

  procedure TTestCNNAlgo.WriteHelp;
  begin
    WriteLn
    (
      'CIFAR-10 CAI Optimized DenseNet Classification Example by Joao Paulo Schwarz Schuler',sLineBreak,
      'Command Line Example: CaiOptimizedDenseNet -i 0.8', sLineBreak,
      ' -h : displays this help. ', sLineBreak,
      ' -l : defines learing rate. Default is -l 0.001. ', sLineBreak,
      ' -i : defines inertia. Default is -i 0.9.', sLineBreak,
      ' -s : enables separable convolutions (less weights and faster).', sLineBreak,
      ' -c : defines the number of convolutions. Default is 12.', sLineBreak,
      ' -b : defines the bottleneck. Default is 32.', sLineBreak,
      ' -n : defines convolutional neurons (growth rate). Default is 32.', sLineBreak,
      ' -p : enables padding and cropping data augmentation.', sLineBreak,
      ' https://github.com/joaopauloschuler/neural-api/tree/master/examples/CaiOptimizedDenseNet',sLineBreak,
      ' More info at:',sLineBreak,
      '   https://github.com/joaopauloschuler/neural-api'
    );
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='CIFAR-10 Classification Example';
  Application.Run;
  Application.Free;
end.
