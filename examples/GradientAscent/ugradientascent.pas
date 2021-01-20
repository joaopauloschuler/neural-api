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
    ChkForceInputRange: TCheckBox;
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
      FNN.SetLearningRate(0.01,0.0);
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
  StatingSum, CurrentSum, InputForce: TNeuralFloat;
  LayerHasDepth: boolean;
  StopRatio: TNeuralFloat;
  vOutputSum: TNeuralFloat;
begin
  iEpochCount := 0;
  iEpochCountAfterLoading := 0;
  FLastLayerIdx := ComboLayer.ItemIndex;
  LayerHasDepth := (FNN.Layers[FLastLayerIdx].OutputError.Depth > 1);
  if LayerHasDepth
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

  // if ChkForceInputRange.Checked
  // then StopRatio := 10
  // else StopRatio := vInput.Size;
  StopRatio := 100;
  for OutputCount := 0 to FOutputSize - 1 do
  begin
    //vInput.Randomize(10000, 5000, 5000000);
    if LayerHasDepth then
    begin
      vInput.Fill(0.0);
      vInput.AddSaltAndPepper(4, 2, -2, true);
    end
    else
    begin
      vInput.RandomizeGaussian(2); vInput.Sub(1);
    end;
    StatingSum := vInput.GetSumAbs();
    if ChkStrongInput.Checked
      then InputForce := 2
      else InputForce := 0.002;
    vInput.NormalizeMax(InputForce);
    Write('Random Input: '); vInput.PrintDebug(); WriteLn;
    for K := 1 to 10000 do
    begin
      FNN.Compute(vInput);
      FNN.GetOutput(pOutput);
      FNN.BackpropagateFromLayerAndNeuron(FLastLayerIdx, OutputCount, 20);
      vInput.MulAdd(-1, FNN.Layers[0].OutputError);
      if ChkForceInputRange.Checked then vInput.ForceMaxRange(2);
      FNN.ClearDeltas();
      FNN.ClearInertia();
      vOutputSum := FNN.Layers[FLastLayerIdx].Output.GetSumAbs();
      if ((K mod 100 = 0) or (vOutputSum > 1000000000)) then
      begin
        vDisplay.Copy(vInput);
        vDisplay.NeuronalWeightToImg(0);
        LoadVolumeIntoTImage(vDisplay, aImage[OutputCount], 0);
        aImage[OutputCount].Width   := FFilterSize;
        aImage[OutputCount].Height  := FFilterSize;
        Application.ProcessMessages();
        CurrentSum := vInput.GetSumAbs();
        WriteLn(StatingSum,' - ', StopRatio, ' - ',CurrentSum,' - ',(CurrentSum/StatingSum),' - ', vOutputSum);
        if (CurrentSum > StopRatio*StatingSum) or Not(LayerHasDepth)
          then break
          else vInput.AddSaltAndPepper(4, InputForce/10, -InputForce/10, true);
      end;
      Application.ProcessMessages();
      if Not(FRunning) then break;
    end;
    //FNN.DebugWeights();
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
