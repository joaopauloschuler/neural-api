program sentimentAnalysis;
(*
sentimentAnalysis: learns how the sentiment in the sst2 dataset:
https://huggingface.co/datasets/sst2
Copyright (C) 2023 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes,
  neuralnetwork,
  neuralvolume,
  neuralfit,
  neuralthread,
  neuraldatasets,
  CustApp,
  Math,
  SysUtils;

const
  csContextLen = 81;
  csTrainingFileName = 'sst2_local/sst2_train.txt';
  csValidationFileName = 'sst2_local/sst2_validation.txt';
  csAutosavedFileName = 'sentiment.nn';
  csVocabSize  = 128; // ASCII character based vocabulary/dictionary.
  csMinSampleSize = 3; // Minimum of 3 characters.
  csClassToStr: array[0..1] of string = ('negative','positive');

type

  { TTestFitLoading }

  TTestFitLoading = class(TCustomApplication)
  private
    procedure TestFromFile;
  protected
    FDataset: TStringList;
    FDatasetClasses: TIntegerList;
    FDatasetSize: integer;
    FDatasetValidation: TStringList;
    FDatasetValidationClasses: TIntegerList;
    FDatasetValidationSize: integer;
    FNN: TNNet;
    NFit: TNeuralDataLoadingFit;
    procedure LoadDataset;
    procedure DoRun; override;
    procedure WriteClassFromChars(str: string);
  public
    procedure OnAfterEpoch(Sender: TObject);
    procedure GetTrainingPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetValidationPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetTestPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
  end;

function LoadDatasetFromFile(filename: string; DsText: TStringList; DsClasses: TIntegerList): integer;
var
  LargeFile: TextFile;
  StrLine: string;
  ClassChar: char;
begin
  WriteLn('Loading: ',filename);
  AssignFile(LargeFile, filename);
  Reset(LargeFile);
  while not Eof(LargeFile) do
  begin
    ReadLn(LargeFile, StrLine);
    if Length(StrLine)>csMinSampleSize then
    begin
      StrLine := LowerCase(StrLine);
      ClassChar := StrLine[1];
      if (ClassChar = '0') or (ClassChar = '1') then
      begin
        DsClasses.Add(StrToInt(ClassChar));
        DsText.Add(Copy(StrLine, 3, Length(StrLine) - 2));
      end;
    end;
  end;
  CloseFile(LargeFile);
  WriteLn('Loaded ',filename,' dataset with ', DsText.Count, ' rows');
  result := DsText.Count;
end;


  procedure TTestFitLoading.LoadDataset;
  var
    I: integer;
  begin
    FDatasetSize := LoadDatasetFromFile(csTrainingFileName, FDataset, FDatasetClasses);
    // creates a dataset validation removing elements from the training;
    if FDatasetSize > 0 then
    begin
      for I := FDatasetSize-1 downto 0 do
      begin
        if (I mod 21 = 0) then
        begin
          FDatasetValidation.Add(FDataset[I]);
          FDatasetValidationClasses.Add(FDatasetClasses[I]);
          FDataset.Delete(I);
          FDatasetClasses.Delete(I);
        end;
      end;
    end;
    FDatasetSize := FDataset.Count;
    FDatasetValidationSize := FDatasetValidation.Count;
    // FDatasetValidationSize := LoadDatasetFromFile(csValidationFileName, FDatasetValidation, FDatasetValidationClasses);
  end;

  procedure TTestFitLoading.DoRun;
  begin
    FDataset := TStringList.Create();
    FDatasetClasses := TIntegerList.Create();
    FDatasetValidation := TStringList.Create();
    FDatasetValidationClasses := TIntegerList.Create();

    LoadDataset();
    FNN := TNNet.Create();
    NFit := TNeuralDataLoadingFit.Create();

    FNN.AddLayer([
      TNNetInput.Create(csContextLen, 1, csVocabSize),
      TNNetPointwiseConv.Create(32,1),
      TNNetDropout.Create(0.5),
      TNNetPadXY.Create(1,0),
      TNNetConvolutionReLU.Create(64,3,0,1,1),
      TNNetMaxPool.Create(3),
      TNNetPadXY.Create(1,0),
      TNNetConvolutionReLU.Create(128,3,0,1,1),
      TNNetPointwiseConvReLU.Create(1024,1),
      TNNetMaxPoolWithPosition.Create(27,27,0,1,0),
      // TNNetFullConnectReLU.Create(1024),
      TNNetFullConnectLinear.Create(2, 1),
      TNNetSoftMax.Create()
    ]);
    DebugThreadCount();
    FNN.DebugStructure;

    WriteLn('Computing...');
    //NFit.MaxThreadNum := 1;
    NFit.LogEveryBatches := 100;
    NFit.InitialLearningRate := 0.001;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.EnableClassComparison();
    NFit.EnableDefaultLoss();
    NFit.AvgWeightEpochCount := 1;
    NFit.OnAfterEpoch := @OnAfterEpoch;
    NFit.FitLoading(
      FNN,
      {TrainingVolumesCount=}FDatasetSize,
      {ValidationVolumesCount=}FDatasetValidationSize,
      {TestVolumesCount=}FDatasetValidationSize,
      {batchsize=}32,
      {epochs=}50,
      @GetTrainingPair, @GetValidationPair, @GetTestPair
    );
    FNN.DebugWeights();
    OnAfterEpoch(Self);

    NFit.Free;
    FNN.Free;
    FDatasetValidationClasses.Free;
    FDatasetValidation.Free;
    FDatasetClasses.Free;
    FDataset.Free;
    Write('Press ENTER to exit.');
    ReadLn;
    Terminate;
  end;

  procedure TTestFitLoading.WriteClassFromChars(str: String);
  var
    ClassId: integer;
  begin
    ClassId := GetClassFromChars(NFit.NN, str);
    WriteLn('"',str, '" is ',csClassToStr[ClassId],'.');
  end;


  procedure TTestFitLoading.OnAfterEpoch(Sender: TObject);
  begin
    WriteLn('Testing.');
    WriteClassFromChars('the is fantastic!');
    WriteClassFromChars('I hated this movie.');
    WriteClassFromChars('Horrible. I do not recommend.');
  end;

  procedure TTestFitLoading.GetTrainingPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    SampleId: integer;
    SampleLen: integer;
    SampleStr: string;
    SampleClass: integer;
  begin
    // Make sure that expected input and output have the proper sizes.
    if FNN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(FNN.GetFirstLayer().Output);
    if FNN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(FNN.GetLastLayer().Output);
    // Get the input sample
    SampleId := Random(FDatasetSize);
    SampleStr := RemoveRandomChars(FDataset[SampleId], Length(FDataset[SampleId]) div 10);
    //SampleStr := RemoveRandomWord(FDataset[SampleId]);
    SampleClass := FDatasetClasses[SampleId];
    SampleLen := Length(SampleStr);
    if SampleLen > pInput.SizeX then SampleStr := copy(SampleStr, 1, pInput.SizeX);
    // Encode the input and output volumes
    pInput.OneHotEncodingReversed(SampleStr);
    pOutput.SetClassForSoftMax(SampleClass);
    pOutput.Tag := SampleClass;
  end;

  procedure TTestFitLoading.GetValidationPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    SampleId: integer;
    SampleLen: integer;
    SampleStr: string;
    SampleClass: integer;
  begin
    // Make sure that expected input and output have the proper sizes.
    if FNN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(FNN.GetFirstLayer().Output);
    if FNN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(FNN.GetLastLayer().Output);
    // Get the input sample
    SampleId := Random(FDatasetValidationSize);
    SampleStr := FDatasetValidation[SampleId];
    SampleClass := FDatasetValidationClasses[SampleId];
    SampleLen := Length(SampleStr);
    if SampleLen > pInput.SizeX then SampleStr := copy(SampleStr, 1, pInput.SizeX);
    // Encode the input and output volumes
    pInput.OneHotEncodingReversed(SampleStr);
    pOutput.SetClassForSoftMax(SampleClass);
    pOutput.Tag := SampleClass;
  end;

  procedure TTestFitLoading.GetTestPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    GetValidationPair(Idx, ThreadId, pInput, pOutput);
  end;

  procedure TTestFitLoading.TestFromFile;
  var
    S: string;
    NN: TNNet;
  begin
    NN := TNNet.Create();
    WriteLn('Loading neural network.');
    NN.LoadFromFile(csAutosavedFileName);
    NN.DebugStructure();
    WriteLn();
    WriteLn('Write something and I will tell you if it is positive or negative.');
    repeat
      Write('User: ');
      ReadLn(S);
      WriteClassFromChars(S);
    until S = 'exit';
    NN.Free;
  end;

var
  Application: TTestFitLoading;
begin
  Application := TTestFitLoading.Create(nil);
  Application.Title:='Nano Covolutional Based NLP Trained from File';
  //Application.TestFromFile;
  Application.Run;
  Application.Free;
end.

