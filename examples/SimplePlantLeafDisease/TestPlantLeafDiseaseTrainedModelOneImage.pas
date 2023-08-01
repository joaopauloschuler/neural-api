///This file has an implementation to test a trained model for
//plant leaf diseases image classification. You can get the dataset at
//https://data.mendeley.com/datasets/tywbtsjrjv/1/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/Plant_leaf_diseases_dataset_without_augmentation.zip?dl=1 .
//Folders with plant diseases will need to be stored inside of a folder named "plant".
program TestPlantLeafDiseaseTrainedModelOneImage;
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

  { TTestCNNAlgo }
  TTestCNNAlgo = class(TCustomApplication)
  protected
    procedure DoRun; override;
    procedure DoRunSimple;
    procedure DoRunDetailed;
  end;

  procedure TTestCNNAlgo.DoRun;
  var
    RunSimple: boolean;
  begin
    RunSimple := true;
    if RunSimple
    then DoRunSimple
    else DoRunDetailed;
  end;

  procedure TTestCNNAlgo.DoRunSimple;
  var
    NN: TNNet;
    ImageFileName: string;
    NeuralFit: TNeuralImageFit;
  begin
    WriteLn('Loading Neural Network...');
    NN := TNNet.Create;
    NN.LoadFromFile('SimplePlantLeafDisease-20230720.nn');
    NN.DebugStructure();
    NeuralFit := TNeuralImageFit.Create;

    ImageFileName := 'plant/Apple___Black_rot/image (1).JPG';
    WriteLn('Processing image: ',ImageFileName);
    WriteLn(
      'The class of the image is: ',
      NeuralFit.ClassifyImageFromFile(NN, ImageFileName)
    );

    WriteLn('Press ENTER to quit.');
    ReadLn();

    NeuralFit.Free;
    NN.Free;
    Terminate;
  end;

  procedure TTestCNNAlgo.DoRunDetailed;
  var
    NN: TNNet;
    ImageFileName: string;
    NeuralFit: TNeuralImageFit;
    vInputImage, vOutput: TNNetVolume;
    InputSizeX, InputSizeY, NumberOfClasses: integer;
  begin
    WriteLn('Loading Neural Network...');
    NN := TNNet.Create;
    NN.LoadFromFile('SimplePlantLeafDisease-20230720.nn');
    NN.DebugStructure();
    InputSizeX := NN.Layers[0].Output.SizeX;
    InputSizeY := NN.Layers[0].Output.SizeY;
    NumberOfClasses := NN.GetLastLayer().Output.Size;

    NeuralFit := TNeuralImageFit.Create;
    vInputImage := TNNetVolume.Create();
    vOutput := TNNetVolume.Create(NumberOfClasses);

    ImageFileName := 'plant/Apple___Black_rot/image (1).JPG';
    WriteLn('Loading image: ',ImageFileName);

    if LoadImageFromFileIntoVolume(
      ImageFileName, vInputImage, InputSizeX, InputSizeY,
      {EncodeNeuronalInput=}csEncodeRGB) then
    begin
      WriteLn('Classifying the image:', ImageFileName);
      vOutput.Fill(0);
      NeuralFit.ClassifyImage(NN, vInputImage, vOutput);
      WriteLn('The image belongs to the class of images: ', vOutput.GetClass());
    end
    else
    begin
      WriteLn('Failed loading image: ',ImageFileName);
    end;

    vInputImage.Free;
    vOutput.Free;

    WriteLn('Press ENTER to quit.');
    ReadLn();

    NeuralFit.Free;
    NN.Free;
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
