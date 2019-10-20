program Cifar100CaiDenseNet;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
 This command line tool trains a CAI Optimized DenseNet with CIFAR-100.
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

type
  TTestCNNAlgo = class(TCustomApplication)
  protected
    fLearningRate, fInertia, fTarget: single;
    bSeparable: boolean;
    iInnerConvNum: integer;
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

    iInnerConvNum := 12;
    if HasOption('c', 'convolutions') then
    begin
      ParStr := GetOptionValue('c', 'convolutions');
      iInnerConvNum := StrToInt(ParStr);
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
    ConvNeuronCount: integer;
    Bottleneck: integer;
    HasMovingNorm: boolean;
    //I,Idx: integer;
  begin
    if not CheckCIFAR100File() then exit;
    WriteLn('Creating Neural Network...');
    NumClasses  := 100;
    NN := THistoricalNets.Create();
    fileNameBase := 'Cifar100CaiDenseNet';
    ConvNeuronCount := 32;
    Bottleneck := 32;
    HasMovingNorm := true;
    NN.AddLayer( TNNetInput.Create(32, 32, 3).EnableErrorCollection() );
    // First block shouldn't be separable.
    NN.AddDenseNetBlockCAI(iInnerConvNum div 6, ConvNeuronCount, {supressBias=}0, TNNetConvolutionReLU, {IsSeparable=}false, {HasMovingNorm=}HasMovingNorm, {pBeforeNorm=}nil, {pAfterNorm=}nil, {BottleNeck=}Bottleneck, {Compression=}0, {Dropout=}0);
    NN.AddDenseNetBlockCAI(iInnerConvNum div 6, ConvNeuronCount, {supressBias=}0, TNNetConvolutionReLU, {IsSeparable=}bSeparable, {HasMovingNorm=}HasMovingNorm, {pBeforeNorm=}nil, {pAfterNorm=}nil, {BottleNeck=}Bottleneck, {Compression=}0, {Dropout=}0);
    NN.AddLayer( TNNetMaxPool.Create(2) );
    NN.AddDenseNetBlockCAI(iInnerConvNum div 3, ConvNeuronCount, {supressBias=}0, TNNetConvolutionReLU, {IsSeparable=}bSeparable, {HasMovingNorm=}HasMovingNorm, {pBeforeNorm=}nil, {pAfterNorm=}nil, {BottleNeck=}Bottleneck, {Compression=}0, {Dropout=}0);
    NN.AddLayer( TNNetMaxPool.Create(2) );
    NN.AddDenseNetBlockCAI(iInnerConvNum div 3, ConvNeuronCount, {IsSeparable=}0, TNNetConvolutionReLU, {IsSeparable=}bSeparable, {HasMovingNorm=}HasMovingNorm, {pBeforeNorm=}nil, {pAfterNorm=}nil, {BottleNeck=}Bottleneck, {Compression=}0, {Dropout=}0);
    NN.AddLayer( TNNetDropout.Create(0.2) );
    NN.AddLayer( TNNetMaxChannel.Create() );
    NN.AddLayer( TNNetFullConnectReLU.Create(128) );
    NN.AddLayer( TNNetFullConnectLinear.Create(NumClasses) );
    NN.AddLayer( TNNetSoftMax.Create() );
    NN.Layers[ NN.GetFirstImageNeuronalLayerIdx() ].InitBasicPatterns();

    WriteLn('Learning rate set to: [',fLearningRate:7:5,']');
    WriteLn('Inertia set to: [',fInertia:7:5,']');
    WriteLn('Target set to: [',fTarget:7:5,']');

    NN.DebugWeights();
    NN.DebugStructure();
    WriteLn('Neural Network will minimize error with:');
    WriteLn(' Layers: ', NN.CountLayers());
    WriteLn(' Neurons:', NN.CountNeurons());
    WriteLn(' Weights:' ,NN.CountWeights());
    CreateCifar100Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);
    (*
    for I := 1 to 100 do
    begin
      Idx := Random(ImgTrainingVolumes.Count);
      WriteLn(
        ImgTrainingVolumes[Idx].Tag,
        ' ', ImgTrainingVolumes[Idx].Tags[1],
        ' ', ImgTrainingVolumes[Idx].GetMin():6:4,
        ' ',ImgTrainingVolumes[Idx].GetMax():6:4
      );
    end;
    ReadLn;
    *)
    NeuralFit := TNeuralImageFit.Create;
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
      'CIFAR-100 Cai Optimized DenseNet Example by Joao Paulo Schwarz Schuler',sLineBreak,
      'Command Line Example: Cifar100CaiDenseNet -i 0.8', sLineBreak,
      ' -h : displays this help. ', sLineBreak,
      ' -l : defines learing rate. Default is -l 0.001. ', sLineBreak,
      ' -i : defines inertia. Default is -i 0.9.', sLineBreak,
      ' -s : enables separable convolutions (less weights and faster).', sLineBreak,
      ' -c : defines the number of convolutions. Default is 12.', sLineBreak,
      ' https://github.com/joaopauloschuler/neural-api/tree/master/examples/Cifar100CaiDenseNet',sLineBreak,
      ' More info at:',sLineBreak,
      '   https://github.com/joaopauloschuler/neural-api'
    );
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='CIFAR-100 Cai Optimized DenseNet Example';
  Application.Run;
  Application.Free;
end.
