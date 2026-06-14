/// Self-supervised image colorization on CIFAR-10.
///
/// The task: predict the chroma (CIELAB a* and b* channels) of an image from
/// its luminance (L) channel alone. A network that learns this can take any
/// grayscale photo and hallucinate a plausible coloring of it.
///
/// Pipeline:
///   * each CIFAR-10 RGB image is converted to CIELAB (sRGB->XYZ(D65)->Lab,
///     via TNNetVolume.RgbToLab);
///   * the L channel (1 channel, normalized to ~[0,1]) is the network INPUT;
///   * the a*,b* channels (2 channels, normalized to ~[-1,1]) are the TARGET;
///   * a small conv encoder-decoder (TNNet.AddUNet) is trained with plain
///     per-pixel L2 (MSE) regression on the a*,b* channels.
///
/// At the end a handful of validation images are colorized: the predicted
/// a*,b* are combined with the TRUE L, mapped back to RGB and written to disk
/// next to their grayscale inputs, so you can eyeball gray-in vs color-out.
///
/// Coded by Joao Paulo Schwarz Schuler with Claude (AI).
/// https://github.com/joaopauloschuler/neural-api
program Colorization;

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, CustApp, Math,
  neuralnetwork, neuralvolume, neuraldatasets, neuralfit;

const
  // Normalization constants. CIELAB L is in [0,100]; a*,b* roughly [-128,127].
  csLScale  = 100.0; // L  -> [0,1]
  csABScale = 110.0; // ab -> ~[-1.16, 1.15]
  csOutputDir = 'colorized';
  csTrainSamples = 2000; // CPU-friendly subset; raise for better quality.
  csValSamples   = 200;

type
  { TColorizationExample }
  TColorizationExample = class(TCustomApplication)
  protected
    // Img*Lab hold the full CIELAB volume (depth 3: L,a,b) per image.
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    ImgTrainingLab, ImgValidationLab: TNNetVolumeList;
    procedure DoRun; override;
  private
    procedure BuildLabLists;
    procedure GetTrainingPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetValidationPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    // Splits a CIELAB volume into the (normalized) L input and a*b* target.
    procedure LabToInputOutput(Lab, pInput, pOutput: TNNetVolume);
    // Recombines normalized L input + (normalized) ab prediction back into RGB.
    procedure InputOutputToRgb(pInput, pAB, pRgb: TNNetVolume);
    procedure SaveColorizedSamples(NN: TNNet; Count: integer);
  end;

procedure TColorizationExample.LabToInputOutput(Lab, pInput, pOutput: TNNetVolume);
var
  X, Y: integer;
begin
  pInput.ReSize(Lab.SizeX, Lab.SizeY, 1);
  pOutput.ReSize(Lab.SizeX, Lab.SizeY, 2);
  for X := 0 to Lab.SizeX - 1 do
    for Y := 0 to Lab.SizeY - 1 do
    begin
      pInput[X, Y, 0]  := Lab[X, Y, 0] / csLScale;
      pOutput[X, Y, 0] := Lab[X, Y, 1] / csABScale;
      pOutput[X, Y, 1] := Lab[X, Y, 2] / csABScale;
    end;
end;

procedure TColorizationExample.InputOutputToRgb(pInput, pAB, pRgb: TNNetVolume);
var
  X, Y: integer;
begin
  pRgb.ReSize(pInput.SizeX, pInput.SizeY, 3);
  for X := 0 to pInput.SizeX - 1 do
    for Y := 0 to pInput.SizeY - 1 do
    begin
      pRgb[X, Y, 0] := pInput[X, Y, 0] * csLScale;  // L
      pRgb[X, Y, 1] := pAB[X, Y, 0] * csABScale;    // a*
      pRgb[X, Y, 2] := pAB[X, Y, 1] * csABScale;    // b*
    end;
  pRgb.LabToRgb(); // in-place CIELAB -> RGB (0..255, gamut clamped)
end;

procedure TColorizationExample.BuildLabLists;
var
  I: integer;
  Lab: TNNetVolume;

  procedure FillFrom(Src: TNNetVolumeList; Dst: TNNetVolumeList; MaxCount: integer);
  var J, N: integer;
  begin
    N := Min(MaxCount, Src.Count);
    for J := 0 to N - 1 do
    begin
      Lab := TNNetVolume.Create;
      Lab.Copy(Src[J]); // RGB 0..255
      Lab.RgbToLab();
      Dst.Add(Lab);
    end;
  end;

begin
  ImgTrainingLab := TNNetVolumeList.Create();
  ImgValidationLab := TNNetVolumeList.Create();
  FillFrom(ImgTrainingVolumes, ImgTrainingLab, csTrainSamples);
  FillFrom(ImgValidationVolumes, ImgValidationLab, csValSamples);
  WriteLn('CIELAB lists built: ', ImgTrainingLab.Count, ' training, ',
    ImgValidationLab.Count, ' validation.');
  I := 0; // silence hint
  if I <> 0 then ;
end;

procedure TColorizationExample.GetTrainingPair(Idx: integer; ThreadId: integer;
  pInput, pOutput: TNNetVolume);
var
  LocalIdx: integer;
begin
  LocalIdx := Random(ImgTrainingLab.Count);
  LabToInputOutput(ImgTrainingLab[LocalIdx], pInput, pOutput);
  if Random(1000) > 500 then
  begin
    pInput.FlipX();
    pOutput.FlipX();
  end;
end;

procedure TColorizationExample.GetValidationPair(Idx: integer; ThreadId: integer;
  pInput, pOutput: TNNetVolume);
begin
  LabToInputOutput(ImgValidationLab[Idx], pInput, pOutput);
end;

procedure TColorizationExample.SaveColorizedSamples(NN: TNNet; Count: integer);
var
  I, X, Y: integer;
  Input, Pred, Rgb, Gray: TNNetVolume;
  N: integer;
begin
  if not DirectoryExists(csOutputDir) then
    CreateDir(csOutputDir);
  Input := TNNetVolume.Create;
  Pred := TNNetVolume.Create;
  Rgb := TNNetVolume.Create;
  Gray := TNNetVolume.Create;
  try
    N := Min(Count, ImgValidationLab.Count);
    for I := 0 to N - 1 do
    begin
      // Gray-in: L channel only, replicated across RGB.
      Gray.ReSize(ImgValidationLab[I].SizeX, ImgValidationLab[I].SizeY, 3);
      for X := 0 to Gray.SizeX - 1 do
        for Y := 0 to Gray.SizeY - 1 do
        begin
          Gray[X, Y, 0] := ImgValidationLab[I][X, Y, 0];
          Gray[X, Y, 1] := 0;
          Gray[X, Y, 2] := 0;
        end;
      Gray.LabToRgb();
      SaveImageFromVolumeIntoFile(Gray,
        csOutputDir + PathDelim + 'sample' + IntToStr(I) + '_gray.png');

      // Color-out: predicted ab + true L -> RGB.
      LabToInputOutput(ImgValidationLab[I], Input, Pred); // Pred reused as scratch shape
      NN.Compute(Input);
      NN.GetOutput(Pred);
      InputOutputToRgb(Input, Pred, Rgb);
      SaveImageFromVolumeIntoFile(Rgb,
        csOutputDir + PathDelim + 'sample' + IntToStr(I) + '_color.png');
    end;
    WriteLn('Wrote ', N, ' gray/color image pairs into ./', csOutputDir, '/');
  finally
    Input.Free;
    Pred.Free;
    Rgb.Free;
    Gray.Free;
  end;
end;

procedure TColorizationExample.DoRun;
var
  NN: TNNet;
  NeuralFit: TNeuralDataLoadingFit;
  EncoderTaps: TNeuralIntegerArray;
begin
  if not CheckCIFARFile() then
  begin
    WriteLn('CIFAR-10 data (data_batch_1.bin ...) not found - cannot run.');
    Terminate;
    Exit;
  end;

  WriteLn('Loading CIFAR-10...');
  CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);
  BuildLabLists;

  WriteLn('Building colorization U-Net (L -> a*b*)...');
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(32, 32, 1));
  // Small CPU-friendly encoder-decoder. OutputChannels=2 (a*,b*).
  NN.AddUNet({Depth=}2, {BaseFeatures=}16, {OutputChannels=}2,
    EncoderTaps, {UseNorm=}true);
  // Bound the regression output to the normalized ab range.
  NN.AddLayer(TNNetHardTanh.Create());
  NN.DebugStructure();

  NeuralFit := TNeuralDataLoadingFit.Create;
  try
    NeuralFit.FileNameBase := 'colorization';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 5;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0;
    NeuralFit.Verbose := true;
    NeuralFit.AvgWeightEpochCount := 1;
    NeuralFit.EnableDefaultLoss(); // per-pixel L2 (MSE) regression reporting
    // Pure per-pixel regression on the a*,b* channels.
    NeuralFit.FitLoading(NN,
      ImgTrainingLab.Count, ImgValidationLab.Count, 0,
      {batchsize=}32, {epochs=}10,
      @GetTrainingPair, @GetValidationPair, @GetValidationPair);

    SaveColorizedSamples(NN, 8);
  finally
    NeuralFit.Free;
  end;

  NN.Free;
  ImgTrainingLab.Free;
  ImgValidationLab.Free;
  ImgTrainingVolumes.Free;
  ImgValidationVolumes.Free;
  ImgTestVolumes.Free;
  Terminate;
end;

var
  Application: TColorizationExample;
begin
  Application := TColorizationExample.Create(nil);
  Application.Title := 'CIFAR-10 Colorization';
  Application.Run;
  Application.Free;
end.
