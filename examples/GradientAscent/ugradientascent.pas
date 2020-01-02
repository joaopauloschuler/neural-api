{
Copyright (C) 2018 Joao Paulo Schwarz Schuler

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
}

unit ugradientascent;

{$mode objfpc}{$H+}

interface

uses
  {$ifdef unix}
  cmem, // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, neuralnetwork, neuralvolumev, neuraldatasets,
  neuralvolume;

type
  { TFormVisualLearning }
  TFormVisualLearning = class(TForm)
    ButLearn: TButton;
    ButLoadFile: TButton;
    ChkStrongInput: TCheckBox;
    ComboLayer: TComboBox;
    GrBoxNeurons: TGroupBox;
    ImgSample: TImage;
    LabLayer: TLabel;
    OpenDialogNN: TOpenDialog;
    procedure ButLearnClick(Sender: TObject);
    procedure ButLoadFileClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
  private
    { private declarations }
    FRunning: boolean;
    aImage: array of TImage;
    aLabelX, aLabelY: array of TLabel;
    FileName: string;

    iEpochCount, iEpochCountAfterLoading: integer;

    FNN: TNNet;
    FLastLayerIdx: integer;
    FFilterSize: integer;
    FImagesPerRow: integer;
    FOutputSize: integer;
    procedure Learn(Sender: TObject);
    procedure EnableComponents(flag: boolean);
    procedure SaveScreenshot(pfilename: string);
    procedure SaveNeuronsImage(pfilename: string);
  public
    procedure ProcessMessages();
  end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
{$R *.lfm}

uses LCLIntf, LCLType, neuraldatasetsv;

{ TFormVisualLearning }

procedure TFormVisualLearning.ButLearnClick(Sender: TObject);
begin
  if (FRunning) then
  begin
    FRunning := false;
  end
  else
  begin
    FRunning := true;
    ButLearn.Caption := 'Stop';
    EnableComponents(false);
    Learn(Sender);
    EnableComponents(true);
    FRunning := false;
  end;
  ButLearn.Caption := 'Restart';
end;

procedure TFormVisualLearning.ButLoadFileClick(Sender: TObject);
begin
  if (OpenDialogNN.Execute()) then
  begin
    if FileExists(OpenDialogNN.FileName) then
    begin
      ButLearn.Enabled := false;
      FileName := OpenDialogNN.FileName;
      if not(FileExists(FileName)) then
      begin
        if FileName = 'SimpleImageClassifier.nn' then
        begin
          WriteLn('Please run the Simple Image Classifier example before running this example.');
          WriteLn('https://github.com/joaopauloschuler/neural-api/tree/master/examples/SimpleImageClassifier');
        end
        else
        begin
          WriteLn('File not found:', FileName);
        end;
        exit;
      end;

      FNN.LoadFromFile(FileName);
      FNN.DebugStructure();
      FNN.DebugWeights();
      FNN.SetBatchUpdate( true );
      FNN.SetLearningRate(0.001,0.0);
      TNNetInput(FNN.Layers[0]).EnableErrorCollection();
      WriteLn('Neural network has: ');
      WriteLn(' Layers: ', FNN.CountLayers()  );
      WriteLn(' Neurons:', FNN.CountNeurons() );
      WriteLn(' Weights:', FNN.CountWeights() );
      ButLearn.Enabled := true;
      LoadNNLayersIntoCombo(FNN, ComboLayer);
    end;
  end;
end;

procedure TFormVisualLearning.FormClose(Sender: TObject;
  var CloseAction: TCloseAction);
begin
  FRunning := false;
end;

procedure TFormVisualLearning.FormCreate(Sender: TObject);
begin
  WriteLn('Creating Neural Network...');
  FNN := TNNet.Create();

  FRunning := false;
  SetLength(aImage, 0);
  SetLength(aLabelX, 0);
  SetLength(aLabelY, 0);
end;

procedure TFormVisualLearning.FormDestroy(Sender: TObject);
begin
  FRunning := false;
  FreeNeuronImages(aImage,  aLabelX,  aLabelY);
  FNN.Free;
end;

procedure TFormVisualLearning.Learn(Sender: TObject);
var
  vInput, pOutput, vDisplay: TNNetVolume;
  OutputCount, K: integer;
  InputSize: integer;
begin
  iEpochCount := 0;
  iEpochCountAfterLoading := 0;
  FLastLayerIdx := ComboLayer.ItemIndex;
  if (FNN.Layers[FLastLayerIdx].OutputError.Depth > 1)
    then FOutputSize := FNN.Layers[FLastLayerIdx].OutputError.Depth
    else FOutputSize := FNN.Layers[FLastLayerIdx].OutputError.Size;

  if FOutputSize > 32 then
  begin
    FFilterSize := 64;
    FImagesPerRow := 16;
  end
  else
  begin
    FFilterSize := 128;
    FImagesPerRow := 8;
  end;

  vInput  := TNNetVolume.Create(FNN.Layers[0].Output);
  vDisplay := TNNetVolume.Create(vInput);
  pOutput := TNNetVolume.Create(FNN.GetLastLayer().Output);
  InputSize := FNN.Layers[0].Output.SizeX;

  FreeNeuronImages(aImage,  aLabelX,  aLabelY);
  CreateAscentImages
  (
    GrBoxNeurons,
    aImage, aLabelX, aLabelY,
    FOutputSize,
    InputSize, FFilterSize, FImagesPerRow
  );

  FormVisualLearning.Width := GrBoxNeurons.Left + GrBoxNeurons.Width + 10;
  FormVisualLearning.Height := GrBoxNeurons.Top + GrBoxNeurons.Height + 10;
  Application.ProcessMessages;

  for OutputCount := 0 to FOutputSize - 1 do
  begin
    //vInput.Randomize(10000, 5000, 5000000);
    vInput.RandomizeGaussian(2); vInput.Sub(1);
    if ChkStrongInput.Checked
      then vInput.NormalizeMax(20)
      else vInput.NormalizeMax(0.002);
    Write('Random Input: '); vInput.PrintDebug(); WriteLn;
    //vInput.Fill(0.1);
    for K := 1 to 100 do
    begin
      FNN.Compute(vInput);
      FNN.GetOutput(pOutput);
      FNN.BackpropagateFromLayerAndNeuron(FLastLayerIdx, OutputCount, 20);
      vInput.MulAdd(-1, FNN.Layers[0].OutputError);
      FNN.ClearDeltas();
      FNN.ClearInertia();
      if K mod 100 = 0 then
      begin
        vDisplay.Copy(vInput);
        vDisplay.NeuronalWeightToImg(0);
        LoadVolumeIntoTImage(vDisplay, aImage[OutputCount], 0);
        aImage[OutputCount].Width   := FFilterSize;
        aImage[OutputCount].Height  := FFilterSize;
      end;
      Application.ProcessMessages();
      if Not(FRunning) then break;
    end;
    FNN.DebugWeights();
  end;

  vDisplay.Free;
  vInput.Free;
  pOutput.Free;
end;

procedure TFormVisualLearning.EnableComponents(flag: boolean);
var
  i : Integer;
begin
  for i := 0 to ComponentCount-1 do
  begin
    if (Components[i] is TEdit) then
      TEdit(Components[i]).Enabled := flag;

    if (Components[i] is TComboBox) then
       TComboBox(Components[i]).Enabled := flag;

    if (Components[i] is TCheckBox) then
       TCheckBox(Components[i]).Enabled := flag;

    if (Components[i] is TRadioButton) then
       TRadioButton(Components[i]).Enabled := flag;
  end;

  Application.ProcessMessages;
end;

procedure TFormVisualLearning.SaveScreenshot(pfilename: string);
begin
  WriteLn(' Saving ',pfilename,'.');
  SaveHandleToBitmap(pfilename, Self.Handle);
end;

procedure TFormVisualLearning.SaveNeuronsImage(pfilename: string);
begin
  WriteLn(' Saving ',pfilename,'.');
  SaveHandleToBitmap(pfilename, GrBoxNeurons.Handle);
end;

procedure TFormVisualLearning.ProcessMessages();
begin
  Application.ProcessMessages();
end;

end.
