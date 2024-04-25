program kOptimizedDenseNet;
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
  csChannelsPerGroup = 16;

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

    iBottleneck := 128;
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
    HasRandomAddBias: integer;
    SupressBias: integer;
    HasIntergroup: boolean;
  begin
    if not CheckCIFARFile() then exit;
    WriteLn('Creating Neural Network...');
    NumClasses  := 10;
    NN := THistoricalNets.Create();
    fileNameBase := 'kCaiOptimizedDenseNet';
    HasMovingNorm := false;
    HasRandomAddBias := 0;
    SupressBias := 1;
    HasIntergroup := false;
    NN.AddLayer( TNNetInput.Create(32, 32, 3) );
    NN.AddLayerDeepConcatingInputOutput( TNNetConvolutionReLU.Create(csChannelsPerGroup-3,3,1,1,0) ); // makes the input multiple
    NN.AddkDenseNetBlock(iInnerConvNum div 6, iConvNeuronCount, SupressBias, TNNetGroupedPointwiseConvLinear, {IsSeparable=}true, {HasMovingNorm=}HasMovingNorm, {pBeforeBot=}nil, {pAfterBot=}TNNetReLU6, {pBeforeConv=}nil, {pAfterConv=}TNNetReLU6, {BottleNeck=}iBottleneck, {Compression=}0, {Dropout=}0, {RandomBias=}HasRandomAddBias, {RandomMul=}HasRandomAddBias, {FeatureSize=}3, {MinGroupSize=}csChannelsPerGroup);
    NN.AddLayer( TNNetMaxPool.Create(2) );
    NN.AddkDenseNetBlock(iInnerConvNum div 6, iConvNeuronCount, SupressBias, TNNetGroupedPointwiseConvLinear, {IsSeparable=}bSeparable, {HasMovingNorm=}HasMovingNorm, {pBeforeBot=}nil, {pAfterBot=}TNNetReLU6, {pBeforeConv=}nil, {pAfterConv=}TNNetReLU6, {BottleNeck=}iBottleneck, {Compression=}0, {Dropout=}0, {RandomAdd=}HasRandomAddBias, {RandomMul=}HasRandomAddBias, {FeatureSize=}3, {MinGroupSize=}csChannelsPerGroup);
    NN.AddGroupedCompression(1, csChannelsPerGroup, 1, HasIntergroup);
    NN.AddLayer( TNNetMaxPool.Create(2) );
    NN.AddLayer( TNNetReLU6.Create() );
    NN.AddkDenseNetBlock(iInnerConvNum div 3, iConvNeuronCount, SupressBias, TNNetGroupedPointwiseConvLinear, {IsSeparable=}bSeparable, {HasMovingNorm=}HasMovingNorm, {pBeforeBot=}nil, {pAfterBot=}TNNetSwish6, {pBeforeConv=}nil, {pAfterConv=}TNNetSwish6, {BottleNeck=}iBottleneck, {Compression=}0, {Dropout=}0, {RandomAdd=}HasRandomAddBias, {RandomMul=}HasRandomAddBias, {FeatureSize=}3, {MinGroupSize=}csChannelsPerGroup);
    NN.AddLayer( TNNetInterleaveChannels.Create(12) );
    NN.AddGroupedCompression(1, csChannelsPerGroup, 1, HasIntergroup);
    NN.AddLayer( TNNetMaxPool.Create(2) );
    NN.AddLayer( TNNetReLU6.Create() );
    NN.AddkDenseNetBlock(iInnerConvNum div 3, iConvNeuronCount, SupressBias, TNNetGroupedPointwiseConvLinear, {IsSeparable=}bSeparable, {HasMovingNorm=}HasMovingNorm, {pBeforeBot=}nil, {pAfterBot=}TNNetSwish6, {pBeforeConv=}nil, {pAfterConv=}TNNetSwish6, {BottleNeck=}iBottleneck, {Compression=}0, {Dropout=}0, {RandomAdd=}HasRandomAddBias, {RandomMul=}HasRandomAddBias, {FeatureSize=}3, {MinGroupSize=}csChannelsPerGroup);
    NN.AddLayer( TNNetInterleaveChannels.Create(16) );
    NN.AddGroupedCompression(1, csChannelsPerGroup, 1, HasIntergroup);

    NN.AddLayer( TNNetMaxChannel.Create() );
    NN.AddLayer( TNNetReLU6.Create() );
    NN.AddLayer( TNNetFullConnectLinear.Create(NumClasses) );
    NN.AddLayer( TNNetSoftMax.Create({SkipBackpropDerivative=}1) );
    NN.Layers[ NN.GetFirstImageNeuronalLayerIdx() ].InitBasicPatterns();

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

    NN.DebugWeights();
    NN.DebugStructure();

    WriteLn('Learning rate set to: [',fLearningRate:7:5,']');
    WriteLn('Inertia set to: [',fInertia:7:5,']');
    WriteLn('Target set to: [',fTarget:7:5,']');

    WriteLn('Has AVX: ', ImgTrainingVolumes[0].HasAVX);
    WriteLn('Has AVX2: ', ImgTrainingVolumes[0].HasAVX2);
    WriteLn('Has AVX512: ', ImgTrainingVolumes[0].HasAVX512);

    WriteLn('Neural Network will minimize error with:');
    WriteLn(' Layers: ', NN.CountLayers());
    WriteLn(' Neurons: ', NN.CountNeurons());
    WriteLn(' Weights: ', NN.CountWeights());

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
    NeuralFit.ClipDelta := 0.01;
    //NeuralFit.MaxThreadNum := 16;
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
