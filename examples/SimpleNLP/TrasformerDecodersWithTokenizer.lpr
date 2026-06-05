program TrasformerDecodersWithTokenizer;
(*
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

The dataset can be found at:
git clone https://huggingface.co/datasets/schuler/TinyStories4Pascal-Tokenized-v2
unzip TinyStories4Pascal-Tokenized-v2/tinystories-100k-tokenized3k.csv.zip
unzip TinyStories4Pascal-Tokenized-v2/tinystories-vocab-3k-cai.csv.zip
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
  neuralab,
  neuraltokenizer,
  CustApp,
  Math,
  sysutils;

const
  csContextLen = 80;
  csTrainingFileName = 'datasets/tinystories-100k-tokenized3k.csv';
  csVocabFileName = 'datasets/tinystories-vocab-3k-cai.csv';
  csMinSampleSize = 3; // Minimum of 3 tokens.
  csEmbedDim = 512;
  csModelVocabSize = 3000;

type
  TTestFitLoading = class(TCustomApplication)
  protected
    FDataset: array of array of integer;
    FDictionary: TNeuralTokenizer;
    FDatasetSize: integer;
    FNN: THistoricalNets;
    NFit: TNeuralDataLoadingFit;
    FSampler: TNNetSamplerBase;
    FMaxPredictCharPos: integer;
    FVocabSize: integer; // Character based vocabulary/dictionary.
    procedure LoadDataset;
    procedure DoRun; override;
  public
    procedure OnAfterEpoch(Sender: TObject);
    procedure OnAfterStep(Sender: TObject);
    procedure GetTrainingPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetValidationPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetTestPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
  end;

  procedure TTestFitLoading.LoadDataset;
  var
    Tokens: array of integer;
  begin
    WriteLn('Loading vocabulary: ', csVocabFileName);
    FDictionary.LoadVocabularyFromFile(csVocabFileName);
    FDictionary.Tokenize('one day a', Tokens);
    WriteLn('one day a: ',Tokens[0],' ',Tokens[1],' ',Tokens[2]);

    WriteLn('Dic : ', FDictionary.DeTokenize(Tokens[0]));
    WriteLn('Dic : ', FDictionary.DeTokenize(Tokens[1]));
    WriteLn('Dic : ', FDictionary.DeTokenize(Tokens[2]));

    WriteLn('Filtering CSV.');
    FilterCSVWithNumbersUpToMax(csTrainingFileName, 'temp.csv', csModelVocabSize-1);
    WriteLn('Loading temp.csv.');
    LoadIntegersInCSV('temp.csv', FDataset);

    FVocabSize := FDictionary.Count;
    FDatasetSize := Length(FDataSet);

    WriteLn('Loaded dataset with ', FDatasetSize, ' rows');
  end;

  function CyclicalAdvLRScheduler25b(Epoch: integer): single;
  var
    BaseLearning: single;
    LocalEpoch: integer;
  begin
    BaseLearning := 0.0005;
    LocalEpoch := Epoch mod 25;
    Result := BaseLearning;
    if Epoch < 25 then
    begin
      if LocalEpoch < 7 then
      begin
        Result := BaseLearning * (1 + 0.5 * LocalEpoch);
      end
      else
      begin
        Result := (BaseLearning * 4) * Power(0.90, (LocalEpoch - 7));
      end;
    end
    else
    begin
      Result := 0.0001;
    end;
  end;

  procedure TTestFitLoading.DoRun;
  var
    W: TNNetLayer;
    I: integer;
    Opt: TNeuralOptimizerAdam;
  begin
    FDictionary := TNeuralTokenizer.Create();
    LoadDataset();
    FNN := THistoricalNets.Create();
    Opt := TNeuralOptimizerAdam.Create(0.9, 0.98);
    NFit := TNeuralDataLoadingFit.Create();
    FMaxPredictCharPos := csContextLen;
    FSampler := TNNetSamplerTopP.Create(0.4);
    FNN.AddLayer([
      TNNetInput.Create(csContextLen, 1, 1),
      TNNetTokenAndPositionalEmbedding.Create(csModelVocabSize, csEmbedDim, 0, 0.02, 0.01)
    ]);

    for I := 1 to 2 do FNN.AddTransformerBlockCAI(8, 2048, true, false, false);

    FNN.AddLayer([
      TNNetPointwiseConvLinear.Create(csEmbedDim),
      TNNetPointwiseConvLinear.Create(csModelVocabSize),
      TNNetPointwiseSoftMax.Create(1)
    ]);

    DebugThreadCount();
    FNN.DebugStructure;
    FNN.DebugWeights();

    WriteLn('Computing...');
    NFit.LogEveryBatches := 10;
    NFit.CustomLearningRateScheduleFn:=@CyclicalAdvLRScheduler25b;
    NFit.Optimizer := Opt;
    NFit.LearningRateDecay := 0.00;
    NFit.StaircaseEpochs := 1;
    NFit.L2Decay := 0.1;
    NFit.EnableMultiClassLoss();
    NFit.EnableClassComparisonInLastPixel();
    NFit.AvgWeightEpochCount := 1;
    NFit.OnAfterEpoch := @OnAfterEpoch;
    NFit.OnAfterStep := @OnAfterStep;
    //NFit.MaxThreadNum := 1;
    NFit.FitLoading(
      FNN,
      {TrainingVolumesCount=}48000*1,
      {ValidationVolumesCount=}48000*1 div 20,
      {TestVolumesCount=}48000*1 div 20,
      {batchsize=}48,
      {epochs=}500,
      @GetTrainingPair, @GetValidationPair, @GetTestPair
    );
    FNN.DebugWeights();
    OnAfterEpoch(Self);
    FSampler.Free;
    Opt.Free;
    NFit.Free;
    FNN.Free;
    FDictionary.Free;
    Terminate;
  end;

  procedure TTestFitLoading.OnAfterEpoch(Sender: TObject);
  begin
    WriteLn(GenerateStringFromCasualNN(NFit.NN, FDictionary, 'one day', nil, csNeuralEncodingMethodIntChar),'.');
    WriteLn(GenerateStringFromCasualNN(NFit.NN, FDictionary, 'once upon a', nil, csNeuralEncodingMethodIntChar),'.');
    WriteLn(GenerateStringFromCasualNN(NFit.NN, FDictionary, 'once upon a time', nil, csNeuralEncodingMethodIntChar),'.');
    WriteLn(GenerateStringFromCasualNN(NFit.NN, FDictionary, 'once upon', nil, csNeuralEncodingMethodIntChar),'.');
    WriteLn(GenerateStringFromCasualNN(NFit.NN, FDictionary, 'billy', FSampler, csNeuralEncodingMethodIntChar),'.');
  end;

  procedure TTestFitLoading.OnAfterStep(Sender: TObject);
  begin
    //if Random(100)=0 then OnAfterEpoch(Sender);
    //NFit.ThreadNN[0].DebugWeights();
  end;

  procedure TTestFitLoading.GetTrainingPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    SampleId: integer;
    SampleLen: integer;
    SampleCutPosition: integer;
    ExpectedTokenInt: integer;
    AIntegerArray: array of integer;
  begin
    // Make sure that expected input and output have the proper sizes.
    if FNN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(FNN.GetFirstLayer().Output);
    if FNN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(FNN.GetLastLayer().Output);
    // Get the input sample
    SampleId := Random(FDatasetSize);
    SampleLen := Min(Length(FDataset[SampleId]), pInput.SizeX);
    SampleLen := Min(FMaxPredictCharPos, SampleLen);
    SampleCutPosition := SampleLen;
    // The expected token is the next character in the string
    ExpectedTokenInt := FDataset[SampleId][SampleCutPosition-1];
    // Encode the input volume
    AIntegerArray := Copy(FDataset[SampleId], 0, SampleCutPosition);
    pInput.Fill(0);
    pInput.CopyNoChecksIntArr( AIntegerArray );
    // Encode the output volume (includes the predicted word)
    AIntegerArray := Copy(FDataset[SampleId], 1, SampleCutPosition);
    pOutput.OneHotEncoding(AIntegerArray) ;
    pOutput.Tag := ExpectedTokenInt;
  end;

  procedure TTestFitLoading.GetValidationPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    SampleId: integer;
    SampleLen: integer;
    SampleCutPosition: integer;
    ExpectedTokenChar: char;
    ExpectedTokenInt: integer;
    AIntegerArray: array of integer;
  begin
    // Make sure that expected input and output have the proper sizes.
    if FNN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(FNN.GetFirstLayer().Output);
    if FNN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(FNN.GetLastLayer().Output);
    // Get the input sample
    SampleId := (Idx * 20) mod FDatasetSize;
    SampleLen := Min(Length(FDataset[SampleId]), pInput.SizeX);
    SampleCutPosition := SampleLen;
    // The expected token is the next character in the string
    ExpectedTokenInt := FDataset[SampleId][SampleCutPosition-1];
    // Encode the input and output volumes
    AIntegerArray := Copy(FDataset[SampleId], 0, SampleCutPosition);
    pInput.Fill(0);
    pInput.CopyNoChecksIntArr( AIntegerArray );
    // Encode the output volume (includes the predicted word)
    AIntegerArray := Copy(FDataset[SampleId], 1, SampleCutPosition);
    pOutput.OneHotEncoding(AIntegerArray) ;
    pOutput.Tag := ExpectedTokenInt;
  end;

  procedure TTestFitLoading.GetTestPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    GetValidationPair(Idx, ThreadId, pInput, pOutput);
  end;

var
  Application: TTestFitLoading;
begin
  Application := TTestFitLoading.Create(nil);
  Application.Title := 'Transformer decoder with tokenizer';
  Application.Run;
  Application.Free;
end.
