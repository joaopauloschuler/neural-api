///This file has an implementation to test a trained model for
//plant leaf diseases image classification. You can get the dataset at
//https://data.mendeley.com/datasets/tywbtsjrjv/1/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/Plant_leaf_diseases_dataset_without_augmentation.zip?dl=1 .
//Folders with plant diseases will need to be stored inside of a folder named "plant".
program TestPlantLeafDiseaseTrainedModel;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math, neuraldatasets,
  neuralfit;

type
  TTestCNNAlgo = class(TCustomApplication)
  protected
    procedure DoRun; override;
  end;

  procedure TestVolumes(NN: TNNet; ImgTestVolumes: TNNetVolumeList; pLabel:string);
  var
    NeuralFit: TNeuralImageFit;
    imgCnt: integer;
    vOutput: TNNetVolume;
    hitsCnt: integer;
  begin
    NeuralFit := TNeuralImageFit.Create;
    vOutput := TNNetVolume.Create(39); // 39 classes of images
    hitsCnt := 0;
    for imgCnt := 0 to ImgTestVolumes.Count - 1 do
    begin
      vOutput.Fill(0);
      NeuralFit.ClassifyImage(NN, ImgTestVolumes[imgCnt], vOutput);
      // is the actual class the same as the predicted class?
      if (ImgTestVolumes[imgCnt].Tag = vOutput.GetClass()) then inc(hitsCnt);
    end;
    WriteLn(pLabel, ' - Correct Classifications: ', hitsCnt);
    WriteLn(pLabel, ' - Tested images: ', ImgTestVolumes.Count);
    WriteLn(pLabel, ' - Precision: ', ((hitsCnt*100)/ImgTestVolumes.Count):6:3,'%');
    vOutput.Free;
    NeuralFit.Free;
  end;

  procedure TTestCNNAlgo.DoRun;
  var
    NN: TNNet;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    ProportionToLoad: Single;
  begin
    WriteLn('Loading Neural Network...');
    NN := TNNet.Create;
    NN.LoadFromFile('SimplePlantLeafDisease-20230720.nn');
    NN.DebugStructure();
    // change ProportionToLoad to a smaller number if you don't have available 16GB of RAM.
    ProportionToLoad := 0.1;
    WriteLn('Loading ', Round(ProportionToLoad*100), '% of the Plant leave disease dataset into memory.');
    CreateVolumesFromImagesFromFolder
    (
      ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,
      {FolderName=}'plant', {pImageSubFolder=}'',
      {color_encoding=}0{RGB},
      {TrainingProp=}0.9*ProportionToLoad,
      {ValidationProp=}0.05*ProportionToLoad,
      {TestProp=}0.05*ProportionToLoad,
      {NewSizeX=}64, {NewSizeY=}64
    );

    WriteLn
    (
      'Training Images:', ImgTrainingVolumes.Count,
      ' Validation Images:', ImgValidationVolumes.Count,
      ' Test Images:', ImgTestVolumes.Count
    );

    TestVolumes(NN, ImgTrainingVolumes, 'Training Images');
    TestVolumes(NN, ImgValidationVolumes, 'Validation Images');
    TestVolumes(NN, ImgTestVolumes, 'Test Images');
    WriteLn('Press ENTER to quit.');
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
  Application.Title:='Plant Leaf Disease Classification';
  Application.Run;
  Application.Free;
end.
