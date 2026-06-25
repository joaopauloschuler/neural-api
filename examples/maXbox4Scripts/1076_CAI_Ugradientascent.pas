{
Copyright (C) 18 Joao Paulo Schwarz Schuler

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

unit ugradientascent_mX476;

//{$mode objfpc}{$H+}

interface

//uses
  {$ifdef unix}
  cmem; // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  //Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  //ExtCtrls, neuralnetwork, neuralvolumev, neuraldatasets,
  //neuralvolume;

type
  { TFormVisualLearning }
  TFormVisualLearning = {class(}TForm;
  var
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
  //private
    { private declarations }
   var 
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
    procedure TFormVisualLearningEnableComponents(flag: boolean);
    procedure SaveScreenshot(pfilename: string);
    procedure SaveNeuronsImage(pfilename: string);
  //public
    procedure TFormVisualLearningProcessMessages();
 // end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
//{$R *.lfm}

//uses LCLIntf, LCLType, neuraldatasetsv;

{ TFormVisualLearning }

procedure ButLearnClick(Sender: TObject);
begin
  if (FRunning) then
  begin
    FRunning := false;
  end
  else
  begin
    FRunning := true;
    ButLearn.Caption := 'Stop';
    TFormVisualLearningEnableComponents(false);
    Learn(Sender);
    TFormVisualLearningEnableComponents(true);
    FRunning := false;
  end;
  ButLearn.Caption := 'Restart';
end;

procedure ButLoadFileClick(Sender: TObject);
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
          WriteLn('File not found:'+FileName);
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
      WriteLn(' Layers: '+ itoa(FNN.CountLayers()  ));
      WriteLn(' Neurons:'+ itoa(FNN.CountNeurons() ));
      WriteLn(' Weights:'+ itoa(FNN.CountWeights() ));
      ButLearn.Enabled := true;
      LoadNNLayersIntoCombo(FNN, ComboLayer);
    end;
  end;
end;

procedure FormClose(Sender: TObject;
  var CloseAction: TCloseAction);
begin
  FRunning := false;
  closeaction:= cafree;
  writeln('form closed...');
end;

procedure FormCreate(Sender: TObject);
var ab: TBitmap;
begin
  WriteLn('Creating Neural Network...');
  FNN := TNNet.Create();

  FRunning := false;
  SetLength(aImage, 0);
  SetLength(aLabelX, 0);
  SetLength(aLabelY, 0);
  ab:= TBitmap.create;
  ab.loadfromresourcename(0,'MOON_FULL');
  imgsample.picture.bitmap:= ab;
  ab.Free;
end;

procedure FormDestroy(Sender: TObject);
begin
  FRunning := false;
  FreeNeuronImages(aImage,  aLabelX,  aLabelY);
  FNN.Free;
  writeln('destroy neural images');
end;

procedure Learn(Sender: TObject);
var
  vInput, pOutput, vDisplay: TNNetVolume;
  OutputCount, K: integer;
  InputSize: integer;
  StatingSum, CurrentSum, InputForce: TNeuralFloat;
  LayerHasDepth: boolean;
  StopRatio: TNeuralFloat;
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

  //RegisterMethod('Constructor Create3( Original : TVolume);');
  vInput  := TNNetVolume.Create3(FNN.Layers[0].Output);
  vDisplay := TNNetVolume.Create3(vInput);
  pOutput := TNNetVolume.Create3(FNN.GetLastLayer().Output);
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
  FormVisualLearning.Height := GrBoxNeurons.Top + GrBoxNeurons.Height + 50;
  Application.ProcessMessages;

  if ChkForceInputRange.Checked
  then StopRatio := 100
  else StopRatio := vInput.Size;

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
    // RegisterMethod('Procedure Sub19( Value : TNeuralFloat);');
      vInput.RandomizeGaussian(2); vInput.Sub19(1);
    end;
    StatingSum := vInput.GetSumAbs();
    if ChkStrongInput.Checked
      then InputForce := 2
      else InputForce := 0.002;
    vInput.NormalizeMax(InputForce);
    Write('Random Input: '); vInput.PrintDebug(); WriteLn('');
    for K := 1 to 10000 do
    begin
      FNN.Compute65(vInput,0);
      FNN.GetOutput(pOutput);
      FNN.BackpropagateFromLayerAndNeuron(FLastLayerIdx, OutputCount, 20);
      vInput.MulAdd69(-1, FNN.Layers[0].OutputError);
      if ChkForceInputRange.Checked then vInput.ForceMaxRange(2);
      FNN.ClearDeltas();
      FNN.ClearInertia();
      if K mod 100 = 0 then
      begin
        vDisplay.Copy76(vInput);
        vDisplay.NeuronalWeightToImg54(0);
        LoadVolumeIntoTImage(vDisplay, aImage[OutputCount], 0);
        aImage[OutputCount].Width   := FFilterSize;
        aImage[OutputCount].Height  := FFilterSize;
        Application.ProcessMessages();
        CurrentSum := vInput.GetSumAbs();
        WriteLn(flots(StatingSum)+' - '+flots(CurrentSum)+' - '+flots((CurrentSum/StatingSum)));
        if (CurrentSum > StopRatio*StatingSum) or Not(LayerHasDepth)
          then break
          else vInput.AddSaltAndPepper(4, InputForce/10, -InputForce/10, true);
      end;
      Application.ProcessMessages();
      if Not(FRunning) then break;
    end;
    //FNN.DebugWeights();
    writeln('neural learn part has been finished!')
  end;

  vDisplay.Free;
  vInput.Free;
  pOutput.Free;
end;

procedure TFormVisualLearningEnableComponents(flag: boolean);
var
  i : Integer;
begin
  for i := 0 to FormVisualLearning.ComponentCount-1 do
  begin
    if (FormVisualLearning.Components[i] is TEdit) then
      TEdit(FormVisualLearning.Components[i]).Enabled := flag;

    if (FormVisualLearning.Components[i] is TComboBox) then
       TComboBox(FormVisualLearning.Components[i]).Enabled := flag;

    if (FormVisualLearning.Components[i] is TCheckBox) then
       TCheckBox(FormVisualLearning.Components[i]).Enabled := flag;

    if (FormVisualLearning.Components[i] is TRadioButton) then
       TRadioButton(FormVisualLearning.Components[i]).Enabled := flag;
  end;

  Application.ProcessMessages;
end;

procedure SaveScreenshot(pfilename: string);
begin
  WriteLn(' Saving '+pfilename+'.');
  SaveHandleToBitmap(pfilename, Self.Handle);
end;

procedure SaveNeuronsImage(pfilename: string);
begin
  WriteLn(' Saving '+pfilename+'.');
  SaveHandleToBitmap(pfilename, GrBoxNeurons.Handle);
end;

procedure TFormVisualLearningProcessMessages();
begin
  Application.ProcessMessages();
end;


procedure loadvisualneuralForm;
begin
FormVisualLearning:= TFormVisualLearning.create(self);
ImgSample:= TImage.create(self);
with  FormVisualLearning do begin
  Left := 88
  Height := 704
  Top := 61
  Width := 1106
  Caption := 'CAI Gradient Ascent Example by maXbox4'
  ClientHeight := 702
  ClientWidth := 1104
  //DesignTimePPI := 120
  OnClose := @FormClose;
  OnCreate := @FormCreate;
  OnDestroy := @FormDestroy;
  //show;
  FormCreate(Self);
  ShoW;
  Position := poScreenCenter
  //LCLVersion := '2.0.2.0'
  ButLearn:= TButton.create(self)
  with butlearn do begin
   parent:= formvisuallearning;
    Left := 624
    Height := 30
    Top := 15
    Width := 112
    Caption := 'Start'
    Enabled := False
    OnClick := @ButLearnClick;
    ParentFont := False
    TabOrder := 0
  end;
  //ImgSample:= TImage.create(self)
  with imgsample do begin
   parent:= formvisuallearning;
    Left := 445
    Height := 32
    Top := 120
    Width := 32
    Stretch := True
  end;
  GrBoxNeurons:= TGroupBox.create(self)
  with grboxneurons do begin
   parent:= formvisuallearning;
    Left := 24
    Height := 312
    Top := 64
    Width := 344
    ParentFont := False
    TabOrder := 1
  end;
  //object ButLoadFile: TButton
  ButLoadFile:= TButton.create(self)
  with ButLoadFile do begin
   parent:= formvisuallearning;  
    Left := 30
    Height := 30
    Top := 16
    Width := 160
    Caption := 'Load Neural Network...'
    OnClick := @ButLoadFileClick
    ParentFont := False
    TabOrder := 2
  end;
  LabLayer:= TLabel.create(self)
  with lablayer do begin
   parent:= formvisuallearning;
    Left := 296
    Height := 20
    Top := 21
    Width := 38
    Caption := 'Layer:'
    ParentColor := False
  end ;
  ComboLayer:= TComboBox.create(self)
  with combolayer do begin
   parent:= formvisuallearning;
    Left := 344
    Height := 30
    Top := 16
    Width := 264
    Enabled := False
    ItemHeight := 25
    Style := csDropDownList
    TabOrder := 3
  end ;
  ChkStrongInput:= TCheckBox.create(self)
  with chkstronginput do begin
   parent:= formvisuallearning;
    Left := 768
    Height := 24
    Top := 8
    Width := 106
    Caption := 'Strong Input'
    Checked := True
    State := cbChecked
    TabOrder := 4
  end;
  //object ChkForceInputRange: TCheckBox
  ChkForceInputRange:= TCheckBox.create(self)
  with ChkForceInputRange do begin
   parent:= formvisuallearning;
    Left := 768
    Height := 24
    Top := 32
    Width := 144
    Caption := 'Force Input Range'
    TabOrder := 5
  end;
  OpenDialogNN:= TOpenDialog.create(self);
  with opendialogNN do begin
    Title := 'Open existing Neural Network File'
    Filter := 'Neural Network|*.nn'
    //left := 488
    //top := 120
  end;
 end;
end; 


begin  //@main

  writeln('CPUSpeed '+cpuspeed);
  loadvisualneuralForm;

End.

{Doc: # Gradient Ascent Example
<img src="https://github.com/joaopauloschuler/neural-api/blob/master/docs/gradientascent.jpg" height="192">

It's usually very hard to understand neuron by neuron how a neural network dedicated to image classification internally works. 
One technique used to help with the understanding about what individual neurons represent is called Gradient Ascent.

In this technique, an arbitrary neuron is required to activate and then the same backpropagation method used for learning is
applied to an input image producing an image that this neuron expects to see.

To be able to run this example, you'll need to load an already trained neural network file and then select the layer you intend to visualize.

Deeper convolutional layers tend to produce more complex patterns. Above image was produced from the a first convolutional layer. The following image was produced from a third convolutional layer. Notice that patterns are more complex.

<img src="https://github.com/joaopauloschuler/neural-api/blob/master/docs/gradientascent3layer.jpg" height="192">

This is the API method used for an arbitrary neuron backpropagation (Gradient Ascent):
```
procedure TNNet.BackpropagateFromLayerAndNeuron(LayerIdx, NeuronIdx: integer; Error: TNeuralFloat);
```

Errors on the input image aren't enabled by default. In this example, errors regarding the input image are enabled with this:
```
TNNetInput(FNN.Layers[0]).EnableErrorCollection();
```

Then, errors are added to the input with this:
```
vInput.MulAdd(-1, FNN.Layers[0].OutputError);
FNN.ClearDeltas();
FNN.ClearInertia();
```

You can find more about Gradient Ascent at:
* [Lecture 12: Visualizing and Understanding - CS231n - Stanford](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf)
* [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/deepvis)}
