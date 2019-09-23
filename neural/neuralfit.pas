// This unit is under development - do not use it.
unit neuralfit;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralnetwork, neuralvolume, MTProcs;

type
  // This is a base class. Do not use it directly.
  TNeuralFitBase = class(TMObject)
    protected
      FBatchSize: integer;
      FMaxThreadNum, FThreadNum: integer;
      FThreadNN: TNNetDataParallelism;
      FAvgWeights: TNNetDataParallelism;
      FAvgWeight: TNNet;
      FAvgWeightEpochCount: integer;
      FNN: TNNet;
      FGlobalHit: integer;
      FGlobalMiss: integer;
      FGlobalTotalLoss: single;
      FGlobalErrorSum: single;
      FWorkingVolumes: TNNetVolumeList;
      FFinishedThread: TNNetVolume;
      FCritSec: TRTLCriticalSection;
      FMultipleSamplesAtValidation: boolean;
      FDataAugmentation: boolean;
      FVerbose: boolean;
      FStaircaseEpochs: integer;
      FStepSize: integer;
      FLearningRateDecay: single;
      FInitialLearningRate: single;
      FInitialEpoch: integer;
      FMaxEpochs: integer;
      FMinLearnRate: single;
      FCurrentLearningRate: single;
      FInertia: single;
      FL2Decay: TNeuralFloat;
      FFileNameBase: string;
      FClipDelta: single;
      FTargetAccuracy: single;
      procedure CheckLearningRate(iEpochCount: integer);
    public
      constructor Create();
      destructor Destroy(); override;

      property AvgWeightEpochCount: integer read FAvgWeightEpochCount write FAvgWeightEpochCount;
      property ClipDelta: single read FClipDelta write FClipDelta;
      property CurrentLearningRate: single read FCurrentLearningRate;
      property FileNameBase: string read FFileNameBase write FFileNameBase;
      property Inertia: single read FInertia write FInertia;
      property InitialEpoch: integer read FInitialEpoch write FInitialEpoch;
      property InitialLearningRate: single read FInitialLearningRate write FInitialLearningRate;
      property LearningRateDecay: single read FLearningRateDecay write FLearningRateDecay;
      property L2Decay: single read FL2Decay write FL2Decay;
      property Momentum: single read FInertia write FInertia;
      property MultipleSamplesAtValidation: boolean read FMultipleSamplesAtValidation write FMultipleSamplesAtValidation;
      property StaircaseEpochs: integer read FStaircaseEpochs write FStaircaseEpochs;
      property TargetAccuracy: single read FTargetAccuracy write FTargetAccuracy;
  end;

  { TNeuralImageFit }

  TNeuralImageFit = class(TNeuralFitBase)
    protected
      FNumClasses: integer;
      FMaxCrop: integer;
      FRunning: boolean;
      FImgVolumes, FImgValidationVolumes, FImgTestVolumes: TNNetVolumeList;
      FImgCrop: boolean;
      FIsSoftmax: boolean;
      FColorEncoding: integer;
    public
      constructor Create();
      destructor Destroy(); override;

      procedure Fit(pNN: TNNet;
        pImgVolumes, pImgValidationVolumes, pImgTestVolumes: TNNetVolumeList;
        pNumClasses, pBatchSize, Epochs: integer);
      procedure RunNNThread(Index: PtrInt; Data: Pointer;
        Item: TMultiThreadProcItem);
      procedure TestNNThread(Index: PtrInt; Data: Pointer;
        Item: TMultiThreadProcItem);
  end;

implementation
uses
  math;

{ TNeuralFitBase }

constructor TNeuralFitBase.Create();
begin
  inherited Create();
  FFinishedThread := TNNetVolume.Create();
  FMaxThreadNum := ProcThreadPool.MaxThreadCount;
  FDataAugmentation := true;
  InitCriticalSection(FCritSec);
  FMultipleSamplesAtValidation := true;
  FVerbose := true;
  FStaircaseEpochs := 1;
  FStepSize := 1;
  FAvgWeightEpochCount := 10;
  FLearningRateDecay := 0.99;
  FInitialLearningRate := 0.001;
  FInitialEpoch := 0;
  fMinLearnRate := FInitialLearningRate * 0.01;
  FInertia := 0.9;
  FClipDelta := 0.0;
  FFileNameBase := 'autosave';
  FL2Decay := 0.0000001;
  FTargetAccuracy := 1;
end;

destructor TNeuralFitBase.Destroy();
begin
  DoneCriticalSection(FCritSec);
  FFinishedThread.Free;
  inherited Destroy();
end;

procedure TNeuralFitBase.CheckLearningRate(iEpochCount: integer);
var
  iStairCount: integer;
  fNewLearningRate: single;
begin
  if FStaircaseEpochs > 1 then
  begin
    iStairCount := iEpochCount div FStaircaseEpochs;
    fNewLearningRate := (fInitialLearningRate * power(fLearningRateDecay, iStairCount));
  end
  else
  begin
    fNewLearningRate := (fInitialLearningRate * power(fLearningRateDecay,iEpochCount));
  end;
  if ( ( fNewLearningRate >= fMinLearnRate ) and (fNewLearningRate <> FCurrentLearningRate) ) then
  begin
    FCurrentLearningRate := fNewLearningRate;
    FThreadNN.SetLearningRate(FCurrentLearningRate, FInertia);
    FNN.SetLearningRate(FCurrentLearningRate, FInertia);
    FNN.ClearInertia();
    if FVerbose then
    WriteLn
    (
      'Learning rate set to:',FCurrentLearningRate:7:5
    );
  end;
end;

{ TNeuralImageFit }

constructor TNeuralImageFit.Create();
begin
  inherited Create();
  FMaxCrop := 8;
  FImgCrop := false;
  FIsSoftmax := true;
  FColorEncoding := 0;
  FMultipleSamplesAtValidation := true;
end;

destructor TNeuralImageFit.Destroy();
begin
  inherited Destroy;
end;


procedure TNeuralImageFit.Fit(pNN: TNNet;
  pImgVolumes, pImgValidationVolumes, pImgTestVolumes: TNNetVolumeList;
  pNumClasses, pBatchSize, Epochs: integer);
var
  I: integer;
  startTime, totalTimeSeconds: double;
  globalStartTime: double;
  CurrentRate: double;
  fileName, FileNameCSV: string;
  TrainingError, TrainingLoss, TrainingRate: TNeuralFloat;
  ValidationError, ValidationLoss, ValidationRate: TNeuralFloat;
  TestError, TestLoss, TestRate: TNeuralFloat;
  CSVFile: TextFile;
  CurrentAccuracy, AccuracyWithInertia: TNeuralFloat;
  MaxDelta: TNeuralFloat;
  ValidationRecord: TNeuralFloat;
  iEpochCount: integer;
begin
  iEpochCount := FInitialEpoch;
  FCurrentLearningRate := FInitialLearningRate;
  FImgVolumes := pImgVolumes;
  FImgValidationVolumes := pImgValidationVolumes;
  FImgTestVolumes := pImgTestVolumes;
  FRunning := true;
  FThreadNum := FMaxThreadNum;
  FBatchSize := pBatchSize;
  FMaxEpochs := Epochs;
  if FBatchSize >= FThreadNum then
  begin
    FBatchSize := (FBatchSize div FThreadNum) * FThreadNum;
  end
  else if FBatchSize > 0 then
  begin
    FThreadNum := FBatchSize;
  end
  else
  begin
    FErrorProc('Batch size has to be bigger than zero.');
    exit;
  end;
  FStepSize := FBatchSize;
  if Assigned(FThreadNN) then FThreadNN.Free;
  FThreadNN := TNNetDataParallelism.Create(pNN, FThreadNum);
  FFinishedThread.Resize(1, 1, FThreadNum);
  FNN := pNN;
  FNumClasses := pNumClasses;
  AccuracyWithInertia := 100 / FNumClasses;
  TrainingError := 0;
  TrainingLoss := 0;
  TrainingRate := 0;
  ValidationRate := 0;
  ValidationLoss := 0;
  ValidationError := 0;
  TestRate := 0;
  TestLoss := 0;
  TestError := 0;
  ValidationRecord := 0;
  totalTimeSeconds := 0;

  FMessageProc('File name is: ' + FFileNameBase);

  FileNameCSV := FFileNameBase + '.csv';
  FileName := FFileNameBase + '.nn';

  AssignFile(CSVFile, FileNameCSV);
  ReWrite(CSVFile);
  WriteLn(CSVFile, 'epoch,training accuracy,training loss,training error,validation accuracy,validation loss,validation error,learning rate,time,test accuracy,test loss,test error');

  FAvgWeights := nil;
  FAvgWeight := FNN.Clone();

  if FVerbose then
  begin
    WriteLn('Learning rate:',FCurrentLearningRate:8:6,
      ' L2 decay:', FL2Decay:8:6,
      ' Inertia:', FInertia:8:6,
      ' Batch size:', FBatchSize,
      ' Step size:', FStepSize,
      ' Staircase ephocs:',FStaircaseEpochs);
    WriteLn(
      'Training images:', FImgVolumes.Count,
      ' Validation images: ', FImgValidationVolumes.Count,
      ' Test images: ', FImgTestVolumes.Count);
  end;

  FThreadNN.SetLearningRate(FCurrentLearningRate, FInertia);
  FThreadNN.SetBatchUpdate(true);
  // in batch update mode, threaded NN should not apply L2 (L2 is applied in the main thread).
  FThreadNN.SetL2Decay(0);

  FNN.SetL2Decay(FL2Decay);
  FNN.SetLearningRate(FCurrentLearningRate, FInertia);

  if FVerbose then
  begin
    MessageProc('Computing...');
  end;

  globalStartTime := Now();
  while ( FRunning and (FMaxEpochs > iEpochCount) ) do
  begin
    FGlobalErrorSum := 0;
    startTime := Now();
    CheckLearningRate(iEpochCount);
    for I := 1 to (FImgVolumes.Count div FStepSize) {$IFDEF MakeQuick}div 10{$ENDIF} do
    begin
      FGlobalHit       := 0;
      FGlobalMiss      := 0;
      FGlobalTotalLoss := 0;
      FGlobalErrorSum  := 0;
      FFinishedThread.Fill(0);
      FNN.ClearTime();
      ProcThreadPool.DoParallel(@RunNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);
      if FClipDelta > 0 then
      begin
        MaxDelta := FNN.ForceMaxAbsoluteDelta(FClipDelta);
      end
      else
      begin
        MaxDelta := FNN.NormalizeMaxAbsoluteDelta();
        if MaxDelta < 1 then
        begin
          MessageProc('Deltas have been multiplied by:'+FloatToStr(MaxDelta) );
        end;
      end;
      FNN.UpdateWeights();
      FNN.ComputeL2Decay();

      if (FGlobalHit > 0) then
      begin
        CurrentRate := FGlobalHit / (FGlobalHit + FGlobalMiss);
        TrainingError := FGlobalErrorSum / (FGlobalHit + FGlobalMiss);
        TrainingLoss  := FGlobalTotalLoss / (FGlobalHit + FGlobalMiss);
        TrainingRate  := CurrentRate;
        CurrentAccuracy := (FGlobalHit*100) div (FGlobalHit+FGlobalMiss);
        if (FStepSize < 100) then
        begin
          AccuracyWithInertia := AccuracyWithInertia*0.99 + CurrentAccuracy*0.01;
        end
        else if (FStepSize < 500) then
        begin
          AccuracyWithInertia := AccuracyWithInertia*0.9 + CurrentAccuracy*0.1;
        end
        else
        begin
          AccuracyWithInertia := CurrentAccuracy;
        end;
      end;

      if ( (FGlobalHit > 0) and (I mod 10 = 0) ) then
      begin
        totalTimeSeconds := (Now() - startTime) * 24 * 60 * 60;

        if FVerbose then WriteLn
        (
          (FGlobalHit + FGlobalMiss)*I + iEpochCount*FImgVolumes.Count,
          ' Examples seen. Accuracy:', (AccuracyWithInertia/100):6:4,
          ' Error:', TrainingError:10:5,
          ' Loss:', TrainingLoss:7:5,
          ' Threads: ', FThreadNum,
          ' Forward time:', (FNN.ForwardTime * 24 * 60 * 60):6:2,'s',
          ' Backward time:', (FNN.BackwardTime * 24 * 60 * 60):6:2,'s',
          ' Step time:', totalTimeSeconds:6:2,'s'
        );

        startTime := Now();
      end;
    end;

    Inc(iEpochCount);

    if not Assigned(FAvgWeights) then
    begin
      FAvgWeight.CopyWeights(FNN);
      if FAvgWeightEpochCount > 1
      then FAvgWeights := TNNetDataParallelism.Create(FNN, FAvgWeightEpochCount)
      else FAvgWeights := TNNetDataParallelism.Create(FNN, 1)
    end;

    if (FRunning) then
    begin
      if FAvgWeightEpochCount > 1 then
      begin
        FAvgWeights.ReplaceAtIdxAndUpdateWeightAvg(iEpochCount mod FAvgWeightEpochCount, FNN, FAvgWeight);
      end
      else
      begin
        FAvgWeight.CopyWeights(FNN);
      end;

      if Assigned(FImgValidationVolumes) then
      begin
        FWorkingVolumes := FImgValidationVolumes;
        FGlobalHit       := 0;
        FGlobalMiss      := 0;
        FGlobalTotalLoss := 0;
        FGlobalErrorSum  := 0;
        FMessageProc('Starting Validation.');
        ProcThreadPool.DoParallel(@TestNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);

        if FGlobalHit + FGlobalMiss > 0 then
        begin
          ValidationRate  := FGlobalHit / (FGlobalHit + FGlobalMiss);
          ValidationLoss  := FGlobalTotalLoss / (FGlobalHit + FGlobalMiss);
          ValidationError := FGlobalErrorSum / (FGlobalHit + FGlobalMiss);
        end;

        if (ValidationRate > ValidationRecord) then
        begin
          ValidationRecord := ValidationRate;
          FMessageProc('VALIDATION RECORD! Saving NN at '+fileName);
          FAvgWeight.SaveToFile(fileName);
        end;
        if (FGlobalHit > 0) and (FVerbose) then
        begin
          WriteLn(
            'Epochs: ',iEpochCount,
            ' Examples seen:', iEpochCount * FImgVolumes.Count,
            ' Validation Accuracy: ', ValidationRate:6:4,
            ' Validation Error: ', ValidationError:6:4,
            ' Validation Loss: ', ValidationLoss:6:4,
            ' Total time: ', (((Now() - globalStartTime)) * 24 * 60): 6: 2, 'min'
          );
        end;

        if (ValidationRate >= FTargetAccuracy) then
        begin
          FRunning := false;
          break;
        end;
      end;// Assigned(FImgValidationVolumes)

      if (iEpochCount mod FThreadNN.Count = 0) and (FVerbose) then
      begin
        FNN.DebugWeights();
      end;

      if Assigned(FImgTestVolumes) then
      begin
        if ( (iEpochCount mod 10 = 0) and (iEpochCount > 0) ) then
        begin
          FWorkingVolumes := FImgTestVolumes;
          FGlobalHit       := 0;
          FGlobalMiss      := 0;
          FGlobalTotalLoss := 0;
          FGlobalErrorSum  := 0;
          FMessageProc('Starting Testing.');
          ProcThreadPool.DoParallel(@TestNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);

          if FGlobalHit + FGlobalMiss > 0 then
          begin
            TestRate  := FGlobalHit / (FGlobalHit + FGlobalMiss);
            TestLoss  := FGlobalTotalLoss / (FGlobalHit + FGlobalMiss);
            TestError := FGlobalErrorSum / (FGlobalHit + FGlobalMiss);
          end;
          if (FGlobalHit > 0) and (FVerbose) then
          begin
            WriteLn(
              'Epochs: ',iEpochCount,
              ' Examples seen:', iEpochCount * FImgVolumes.Count,
              ' Test Accuracy: ', TestRate:6:4,
              ' Test Error: ', TestError:6:4,
              ' Test Loss: ', TestLoss:6:4,
              ' Total time: ', (((Now() - globalStartTime)) * 24 * 60): 6: 2, 'min'
            );
          end;
        end;
        if (TestRate >= FTargetAccuracy) then
        begin
          FRunning := false;
          break;
        end;
      end;

      if ( (iEpochCount mod 10 = 0) and (iEpochCount > 0) ) then
      begin
        WriteLn
        (
          CSVFile,
          iEpochCount,',',
          (AccuracyWithInertia/100):6:4,',',
          TrainingLoss:6:4,',',
          TrainingError:6:4,',',
          ValidationRate:6:4,',',
          ValidationLoss:6:4,',',
          ValidationError:6:4,',',
          FCurrentLearningRate:9:7,',',
          Round( (Now() - globalStartTime) * 24 * 60 * 60),',',
          TestRate:6:4,',',
          TestLoss:6:4,',',
          TestError:6:4
        );
      end
      else
      begin
        WriteLn
        (
          CSVFile,
          iEpochCount,',',
          (AccuracyWithInertia/100):6:4,',',
          TrainingLoss:6:4,',',
          TrainingError:6:4,',',
          ValidationRate:6:4,',',
          ValidationLoss:6:4,',',
          ValidationError:6:4,',',
          FCurrentLearningRate:9:7,',',
          Round( (Now() - globalStartTime) * 24 * 60 * 60)
        );
      end;
      CloseFile(CSVFile);
      AssignFile(CSVFile, FileNameCSV);
      Append(CSVFile);

      MessageProc(
        'Epoch time: ' + FloatToStrF( totalTimeSeconds*(50000/(FStepSize*10))/60,ffGeneral,1,4)+' minutes.' +
        ' 100 epochs: ' + FloatToStrF( 100*totalTimeSeconds*(50000/(FStepSize*10))/3600,ffGeneral,1,4)+' hours.');

      MessageProc(
        'Epochs: '+IntToStr(iEpochCount)+
        '. Working time: '+FloatToStrF(Round((Now() - globalStartTime)*2400)/100,ffGeneral,4,2)+' hours.');

    end;
  end;

  FAvgWeight.Free;
  if Assigned(FAvgWeights) then FAvgWeights.Free;
  FThreadNN.Free;
  CloseFile(CSVFile);
  FRunning := false;
end;

procedure TNeuralImageFit.RunNNThread(Index: PtrInt; Data: Pointer;
  Item: TMultiThreadProcItem);
var
  BlockSize, BlockSizeRest, CropSizeX, CropSizeY: integer;
  LocalNN: TNNet;
  ImgInput, ImgInputCp: TNNetVolume;
  pOutput, vOutput, LocalClassHits: TNNetVolume;
  I, ImgIdx: integer;
  OutputValue, CurrentLoss: TNeuralFloat;
  LocalHit, LocalMiss: integer;
  LocalTotalLoss, LocalErrorSum: TNeuralFloat;
begin
  ImgInput := TNNetVolume.Create();
  ImgInputCp := TNNetVolume.Create();
  pOutput := TNNetVolume.Create(FNumClasses,1,1);
  vOutput := TNNetVolume.Create(FNumClasses,1,1);
  LocalClassHits := TNNetVolume.Create(1, FNumClasses, 2, 0);

  LocalHit := 0;
  LocalMiss := 0;
  LocalTotalLoss := 0;
  LocalErrorSum := 0;

  BlockSize := FBatchSize div FThreadNum;
  BlockSizeRest := FBatchSize mod FThreadNum;

  if (Index < BlockSizeRest) then Inc(BlockSize);

  LocalNN := FThreadNN[Index];
  LocalNN.CopyWeights(FNN);
  LocalNN.ClearTime();
  LocalNN.ClearDeltas();
  LocalNN.EnableDropouts(true);
  for I := 1 to BlockSize do
  begin
    if not(FRunning) then Break;
    ImgIdx := Random(FImgVolumes.Count);

    if FDataAugmentation then
    begin
      if FImgCrop then
      begin
        ImgInput.CopyCropping(FImgVolumes[ImgIdx], random(8), random(8), 24, 24);
      end
      else
      begin
        CropSizeX := random(FMaxCrop + 1);
        CropSizeY := random(FMaxCrop + 1);
        ImgInputCp.CopyCropping(FImgVolumes[ImgIdx], random(CropSizeX), random(CropSizeY),FImgVolumes[ImgIdx].SizeX-CropSizeX, FImgVolumes[ImgIdx].SizeY-CropSizeY);
        ImgInput.CopyResizing(ImgInputCp, FImgVolumes[ImgIdx].SizeX, FImgVolumes[ImgIdx].SizeY);
      end;

      if (ImgInput.Depth = 3) then
      begin
        if (Random(1000) > 750) then
        begin
          ImgInput.MakeGray(FColorEncoding);
        end;
      end;

      // flip is always used in training
      if Random(1000) > 500 then
      begin
        ImgInput.FlipX();
      end;
    end
    else begin
      ImgInput.Copy(FImgVolumes[ImgIdx]);
    end;
    ImgInput.Tag := FImgVolumes[ImgIdx].Tag;

    LocalNN.Compute( ImgInput );
    LocalNN.GetOutput( pOutput );

    if FIsSoftmax
      then vOutput.SetClassForSoftMax( ImgInput.Tag )
      else vOutput.SetClass( ImgInput.Tag, +0.9, -0.1);

    LocalErrorSum += vOutput.SumDiff( pOutput );
    OutputValue := pOutput.FData[ ImgInput.Tag ];
    if Not(FIsSoftmax) then
    begin
      OutputValue := Max(OutputValue, 0.001);
      LocalNN.Backpropagate(vOutput);
    end
    else
    begin
      LocalNN.Backpropagate(vOutput);
    end;

    if (OutputValue > 0) then
    begin
      CurrentLoss := -Ln(OutputValue);
    end
    else
    begin
      WriteLn('Error - invalid output value:',OutputValue);
      CurrentLoss := 1;
    end;
    LocalTotalLoss += CurrentLoss;

    if pOutput.GetClass() = FImgVolumes[ImgIdx].Tag then
    begin
      Inc(LocalHit);
    end
    else
    begin
      Inc(LocalMiss);
    end;
  end; // of for

  if Index and 1 = 0 then
  begin
    if Index + 1 < FThreadNum then
    begin
      while FFinishedThread.FData[Index + 1] = 0 do;
      LocalNN.SumDeltasNoChecks(FThreadNN[Index + 1]);
      FFinishedThread.FData[Index] += FFinishedThread.FData[Index + 1];
    end;
  end;
  FFinishedThread.FData[Index] += 1;
  if Index and 3 = 0 then
  begin
    if Index + 2 < FThreadNum then
    begin
      while FFinishedThread.FData[Index + 2] = 0 do;
      LocalNN.SumDeltasNoChecks(FThreadNN[Index + 2]);
      FFinishedThread.FData[Index] += FFinishedThread.FData[Index + 2];
    end;
  end;
//  WriteLn('Index:',Index,' [',FFinishedThread.FData[Index],']');

  EnterCriticalSection(FCritSec);
  FGlobalHit       += LocalHit;
  FGlobalMiss      += LocalMiss;
  FGlobalTotalLoss += LocalTotalLoss;
  FGlobalErrorSum  += LocalErrorSum;

  FNN.ForwardTime := FNN.ForwardTime + LocalNN.ForwardTime;
  FNN.BackwardTime := FNN.BackwardTime + LocalNN.BackwardTime;
  {$IFDEF Debug}
  if Index and 3 = 0 then FNN.SumDeltas(LocalNN);
  {$ELSE}
  if Index and 3 = 0 then FNN.SumDeltasNoChecks(LocalNN);
  {$ENDIF}
  LocalClassHits.Free;
  LeaveCriticalSection(FCritSec);
  ImgInputCp.Free;
  ImgInput.Free;
  vOutput.Free;
  pOutput.Free;
end;

procedure TNeuralImageFit.TestNNThread(Index: PtrInt; Data: Pointer;
  Item: TMultiThreadProcItem);
var
  BlockSize: integer;
  LocalNN: TNNet;
  ImgInput, ImgInputCp: TNNetVolume;
  pOutput, vOutput, sumOutput, LocalFrequency: TNNetVolume;
  I, ImgIdx: integer;
  StartPos, FinishPos: integer;
  OutputValue, CurrentLoss: TNeuralFloat;
  LocalHit, LocalMiss: integer;
  LocalTotalLoss, LocalErrorSum: TNeuralFloat;
  PredictedClass: integer;
begin
  ImgInput := TNNetVolume.Create();
  ImgInputCp := TNNetVolume.Create();
  pOutput := TNNetVolume.Create(FNumClasses,1,1);
  vOutput := TNNetVolume.Create(FNumClasses,1,1);
  sumOutput := TNNetVolume.Create(FNumClasses,1,1);
  LocalFrequency := TNNetVolume.Create();

  LocalHit := 0;
  LocalMiss := 0;
  LocalTotalLoss := 0;
  LocalErrorSum := 0;

  BlockSize := (FWorkingVolumes.Count div FThreadNum) {$IFDEF MakeQuick}div 10{$ENDIF};
  StartPos  := BlockSize * index;
  FinishPos := BlockSize * (index + 1) - 1;

  LocalNN := FThreadNN[Index];
  LocalNN.CopyWeights(FAvgWeight);
  LocalNN.EnableDropouts(false);
  for I := StartPos to FinishPos - 1 do
  begin
    if not(FRunning) then Break;
    sumOutput.Fill(0);
    ImgIdx := I;

    if FImgCrop then
    begin
      ImgInput.CopyCropping(FWorkingVolumes[ImgIdx], 4, 4, 24, 24);
    end
    else
    begin
      ImgInput.Copy(FWorkingVolumes[ImgIdx]);
    end;

    ImgInput.Tag := FWorkingVolumes[ImgIdx].Tag;

    LocalNN.Compute( ImgInput );
    LocalNN.GetOutput( pOutput );
    if FMultipleSamplesAtValidation then
    begin
      sumOutput.Add( pOutput );

      ImgInput.FlipX();

      LocalNN.Compute( ImgInput );
      LocalNN.GetOutput( pOutput );
      sumOutput.Add( pOutput );

      if ImgInput.SizeX >= 32 then
      begin
        ImgInputCp.CopyCropping(ImgInput, FMaxCrop div 2, FMaxCrop div 2, ImgInput.SizeX - FMaxCrop, ImgInput.SizeY - FMaxCrop);
        ImgInput.CopyResizing(ImgInputCp, ImgInput.SizeX, ImgInput.SizeY);
        LocalNN.Compute( ImgInput );
        LocalNN.GetOutput( pOutput );
        sumOutput.Add( pOutput );
        sumOutput.Divi(3);
      end
      else
      begin
        sumOutput.Divi(2);
      end;

      pOutput.Copy(sumOutput);
    end;

    vOutput.SetClassForSoftMax( ImgInput.Tag );
    LocalErrorSum += vOutput.SumDiff( pOutput );

    OutputValue := pOutput.FData[ ImgInput.Tag ];
    if Not(FIsSoftmax) then OutputValue += 0.5001;

    if (OutputValue > 0) then
    begin
      CurrentLoss := -Ln(OutputValue);
    end
    else
    begin
      WriteLn('Error - invalid output value:',OutputValue);
      CurrentLoss := 1;
    end;
    LocalTotalLoss += CurrentLoss;

    PredictedClass := pOutput.GetClass();
    if PredictedClass = ImgInput.Tag then
    begin
      Inc(LocalHit);
    end
    else
    begin
      Inc(LocalMiss);
    end;
  end; // of for
  LocalNN.EnableDropouts(true);

  EnterCriticalSection(FCritSec);
  FGlobalHit       += LocalHit;
  FGlobalMiss      += LocalMiss;
  FGlobalTotalLoss += LocalTotalLoss;
  FGlobalErrorSum  += LocalErrorSum;
  LeaveCriticalSection(FCritSec);

  LocalFrequency.Free;
  sumOutput.Free;
  ImgInputCp.Free;
  ImgInput.Free;
  vOutput.Free;
  pOutput.Free;
  //WriteLn('Fixes: bad=',cntBAD,' yay=',cntYAY);
end;


end.

