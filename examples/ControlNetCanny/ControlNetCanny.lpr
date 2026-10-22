program ControlNetCanny;
(*
ControlNetCanny: Stable Diffusion 1.x text-to-image guided by a ControlNet
(diffusers StableDiffusionControlNetPipeline), through TStableDiffusionPipeline
with a TStableDiffusionControlNet conditioner in neuralpretrained.pas.

  * no --model / --controlnet: the committed random pico fixtures
    tests/fixtures/tiny_sd15 + tests/fixtures/tiny_sd15_controlnet (16x16,
    random weights): this checks wiring and I/O only.
  * --model DIR: a diffusers Stable Diffusion 1.x folder, e.g.
    runwayml/stable-diffusion-v1-5; --controlnet DIR: a diffusers
    ControlNetModel folder, e.g. lllyasviel/sd-controlnet-canny.

--control FILE is the control image (for the canny ControlNet: white edges on
black, e.g. the output of cv2.Canny); it is resized to the image size. --photo
FILE instead resizes a photo to the image size and extracts its edges with
CannyEdges (a cv2.Canny port in neuralimageresize.pas). Without either the
example draws a circle and a square outline.

--control-start / --control-end limit the ControlNet to that fraction of the
steps and --guess-mode runs it in the prompt pass only (diffusers
control_guidance_start / _end and guess_mode). --ip-adapter FILE --ip-image PNG
[--ip-scale S] adds an h94/IP-Adapter image prompt on top of the ControlNet
(TStableDiffusionMultiConditioner).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuraldatasets, neuralpretrained,
  neuralimageresize;

const
  PicoModelFolder = '../../tests/fixtures/tiny_sd15';
  PicoControlNetFolder = '../../tests/fixtures/tiny_sd15_controlnet';

type
  TControlNetCannyOptions = record
    ModelFolder, ControlNetFolder, ControlFile, PhotoFile, EdgesFile, Prompt,
      NegativePrompt, OutputFile: string;
    // A side < 1: the checkpoint default (512 for SD 1.5).
    Steps, ImageWidth, ImageHeight: integer;
    GuidanceScale, ConditioningScale: double;
    ControlStart, ControlEnd: double;
    GuessMode: boolean;
    // --ip-adapter weights file and --ip-image; '' for none.
    IPAdapterFile, IPImageFile: string;
    IPScale: double;
    // cv2.Canny thresholds and the optional GaussianBlurImage radius before it.
    CannyLow, CannyHigh, CannyBlur: double;
    Seed: cardinal;
    UseOpenCL, UseInt8: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

  // Prints the pipeline's stages and steps as they run.
  TControlNetReporter = class(TObject)
  public
    procedure OnPhase(Phase: TStableDiffusionPipelinePhase);
    procedure OnStep(StepIndex, StepCount: integer; Timestep: double;
      Latents: TNNetVolume);
  end;

procedure TControlNetReporter.OnPhase(Phase: TStableDiffusionPipelinePhase);
const
  PhaseNames: array[TStableDiffusionPipelinePhase] of string = (
    'loading the text encoder', 'encoding the prompt',
    'loading the UNet and the ControlNet', 'denoising',
    'loading the VAE decoder', 'decoding', 'done');
begin
  WriteLn('  ', PhaseNames[Phase]);
end;

procedure TControlNetReporter.OnStep(StepIndex, StepCount: integer;
  Timestep: double; Latents: TNNetVolume);
begin
  WriteLn(Format('    step %d/%d  t=%.0f', [StepIndex + 1, StepCount, Timestep]));
end;

procedure PrintUsage();
begin
  WriteLn('Usage: ControlNetCanny [--model DIR] [--controlnet DIR] [options]');
  WriteLn('  --model DIR       diffusers Stable Diffusion 1.x folder ',
    '(default: the pico fixture)');
  WriteLn('  --controlnet DIR  diffusers ControlNetModel folder ',
    '(default: the pico fixture)');
  WriteLn('  --control FILE    control image, e.g. a canny edge map ',
    '(default: drawn outlines)');
  WriteLn('  --photo FILE      photo to extract canny edges from instead ',
    'of --control');
  WriteLn('  --canny-low T     canny low threshold (default 100)');
  WriteLn('  --canny-high T    canny high threshold (default 200)');
  WriteLn('  --canny-blur R    Gaussian blur radius before canny ',
    '(default 0: none, as cv2.Canny)');
  WriteLn('  --edges-out PNG   also save the extracted edge map');
  WriteLn('  --prompt TEXT     prompt (default "a futuristic city, ',
    'detailed, sharp focus")');
  WriteLn('  --negative TEXT   negative prompt (default empty)');
  WriteLn('  --steps N         denoising steps (default 25)');
  WriteLn('  --guidance G      classifier-free guidance scale (default 7.5)');
  WriteLn('  --scale S         controlnet_conditioning_scale (default 1.0)');
  WriteLn('  --control-start F fraction of the steps before the ControlNet ',
    'joins (default 0)');
  WriteLn('  --control-end F   fraction of the steps after which it stops ',
    '(default 1)');
  WriteLn('  --guess-mode      ControlNet in the prompt pass only, as ',
    'diffusers guess_mode');
  WriteLn('  --ip-adapter FILE IP-Adapter ip-adapter_sd15 (.safetensors/.bin) ',
    'with image_encoder/ beside it; needs --ip-image');
  WriteLn('  --ip-image PNG    the image prompt of --ip-adapter');
  WriteLn('  --ip-scale S      IP-Adapter scale (default 1.0)');
  WriteLn('  --width N         image width (default: the checkpoint''s, ',
    '512 for SD 1.5)');
  WriteLn('  --height N        image height (default: the checkpoint''s)');
  WriteLn('  --size N          sets both --width and --height');
  WriteLn('  --seed N          noise seed (default 0)');
  WriteLn('  --out PNG         output file (default controlnet_canny.png)');
  WriteLn('  --int8            UNet, ControlNet and text encoder weights in ',
    'int8');
  WriteLn('  --gpu             run the nets on OpenCL (needs a build with ',
    '-dOpenCL)');
  WriteLn('  --gpu-platform N  OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N    OpenCL device index (default 0)');
end;

function ParseOptions(out Opt: TControlNetCannyOptions): boolean;
var
  ArgPos: integer;
  Arg: string;

  function NextArg(): string;
  begin
    Inc(ArgPos);
    if ArgPos > ParamCount then
      raise Exception.Create(Arg + ' needs a value');
    Result := ParamStr(ArgPos);
  end;

begin
  Opt.ModelFolder := PicoModelFolder;
  Opt.ControlNetFolder := PicoControlNetFolder;
  Opt.ControlFile := '';
  Opt.PhotoFile := '';
  Opt.EdgesFile := '';
  Opt.CannyLow := 100;
  Opt.CannyHigh := 200;
  Opt.CannyBlur := 0;
  Opt.Prompt := 'a futuristic city, detailed, sharp focus';
  Opt.NegativePrompt := '';
  Opt.OutputFile := 'controlnet_canny.png';
  Opt.Steps := 25;
  Opt.ImageWidth := 0;
  Opt.ImageHeight := 0;
  Opt.GuidanceScale := 7.5;
  Opt.ConditioningScale := 1.0;
  Opt.ControlStart := 0;
  Opt.ControlEnd := 1;
  Opt.GuessMode := false;
  Opt.IPAdapterFile := '';
  Opt.IPImageFile := '';
  Opt.IPScale := 1.0;
  Opt.Seed := 0;
  Opt.UseOpenCL := false;
  Opt.UseInt8 := false;
  Opt.OpenCLPlatform := 0;
  Opt.OpenCLDevice := 0;
  Result := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--model' then Opt.ModelFolder := NextArg()
    else if Arg = '--controlnet' then Opt.ControlNetFolder := NextArg()
    else if Arg = '--control' then Opt.ControlFile := NextArg()
    else if Arg = '--photo' then Opt.PhotoFile := NextArg()
    else if Arg = '--edges-out' then Opt.EdgesFile := NextArg()
    else if Arg = '--canny-low' then
      Opt.CannyLow := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--canny-high' then
      Opt.CannyHigh := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--canny-blur' then
      Opt.CannyBlur := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--prompt' then Opt.Prompt := NextArg()
    else if Arg = '--negative' then Opt.NegativePrompt := NextArg()
    else if Arg = '--out' then Opt.OutputFile := NextArg()
    else if Arg = '--steps' then Opt.Steps := StrToInt(NextArg())
    else if Arg = '--width' then Opt.ImageWidth := StrToInt(NextArg())
    else if Arg = '--height' then Opt.ImageHeight := StrToInt(NextArg())
    else if Arg = '--size' then
    begin
      Opt.ImageWidth := StrToInt(NextArg());
      Opt.ImageHeight := Opt.ImageWidth;
    end
    else if Arg = '--guidance' then
      Opt.GuidanceScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--scale' then
      Opt.ConditioningScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--control-start' then
      Opt.ControlStart := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--control-end' then
      Opt.ControlEnd := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--guess-mode' then Opt.GuessMode := true
    else if Arg = '--ip-adapter' then Opt.IPAdapterFile := NextArg()
    else if Arg = '--ip-image' then Opt.IPImageFile := NextArg()
    else if Arg = '--ip-scale' then
      Opt.IPScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--seed' then Opt.Seed := StrToInt64(NextArg())
    else if Arg = '--int8' then Opt.UseInt8 := true
    else if Arg = '--gpu' then Opt.UseOpenCL := true
    else if Arg = '--gpu-platform' then Opt.OpenCLPlatform := StrToInt(NextArg())
    else if Arg = '--gpu-device' then Opt.OpenCLDevice := StrToInt(NextArg())
    else if (Arg = '--help') or (Arg = '-h') then
    begin
      PrintUsage();
      Result := false;
      exit;
    end
    else
      raise Exception.Create('unknown argument: ' + Arg);
    Inc(ArgPos);
  end;
  if Opt.Steps < 1 then raise Exception.Create('--steps must be >= 1');
  if (Opt.ControlFile <> '') and (Opt.PhotoFile <> '') then
    raise Exception.Create('pass --control or --photo, not both');
  if (Opt.IPAdapterFile = '') <> (Opt.IPImageFile = '') then
    raise Exception.Create('--ip-adapter and --ip-image go together');
end;

// A (Width, Height, 3) edge map in 0..255: one-pixel white outlines of a
// centered circle and square (sized by the shorter side) on black.
procedure DrawEdgeMap(Image: TNNetVolume; Width, Height: integer);
var
  X, Y, MaxX, MaxY, CenterX, CenterY, Radius, HalfSquare: integer;
  IsEdge: boolean;
begin
  Image.ReSize(Width, Height, 3);
  Image.Fill(0);
  MaxX := Width - 1;
  MaxY := Height - 1;
  CenterX := Width div 2;
  CenterY := Height div 2;
  Radius := Min(Width, Height) * 3 div 8;
  HalfSquare := Min(Width, Height) div 4;
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      IsEdge := Abs(Round(Sqrt(Sqr(X - CenterX) + Sqr(Y - CenterY))) -
        Radius) = 0;
      IsEdge := IsEdge or ((Abs(X - CenterX) = HalfSquare) and
        (Abs(Y - CenterY) <= HalfSquare)) or
        ((Abs(Y - CenterY) = HalfSquare) and (Abs(X - CenterX) <= HalfSquare));
      if IsEdge then
      begin
        Image[X, Y, 0] := 255;
        Image[X, Y, 1] := 255;
        Image[X, Y, 2] := 255;
      end;
    end;
end;

procedure Run(const Opt: TControlNetCannyOptions);
var
  Pipeline: TStableDiffusionPipeline;
  ControlNet: TStableDiffusionControlNet;
  IPAdapter: TStableDiffusionIPAdapter;
  Multi: TStableDiffusionMultiConditioner;
  Reporter: TControlNetReporter;
  Control, Image: TNNetVolume;
  ComputeText: string;
  ImageWidth, ImageHeight: integer;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLText: string;
  {$ENDIF}
begin
  Pipeline := TStableDiffusionPipeline.Create(Opt.ModelFolder);
  ControlNet := TStableDiffusionControlNet.Create(Opt.ControlNetFolder);
  IPAdapter := nil;
  Multi := nil;
  Reporter := TControlNetReporter.Create;
  Control := TNNetVolume.Create;
  Image := TNNetVolume.Create;
  try
    ComputeText := 'CPU';
    if Opt.UseOpenCL then
    begin
      {$IFDEF OpenCL}
      if Pipeline.EnableOpenCLDevice(Opt.OpenCLPlatform, Opt.OpenCLDevice,
        OpenCLText) then ComputeText := 'OpenCL on ' + OpenCLText
      else ComputeText := 'CPU (' + OpenCLText + ')';
      {$ELSE}
      ComputeText := 'CPU (built without -dOpenCL)';
      {$ENDIF}
    end;
    ImageWidth := Opt.ImageWidth;
    if ImageWidth < 1 then ImageWidth := Pipeline.DefaultImageSize;
    ImageHeight := Opt.ImageHeight;
    if ImageHeight < 1 then ImageHeight := Pipeline.DefaultImageSize;
    if Opt.ControlFile <> '' then
    begin
      if not LoadImageFromFileIntoVolume(Opt.ControlFile, Control) then
        raise Exception.Create('could not read ' + Opt.ControlFile);
    end
    else if Opt.PhotoFile <> '' then
    begin
      if not LoadImageFromFileIntoVolume(Opt.PhotoFile, Image) then
        raise Exception.Create('could not read ' + Opt.PhotoFile);
      ResizeImagePIL(Image, Image, ImageWidth, ImageHeight, irfLanczos, true);
      GaussianBlurImage(Image, Image, Opt.CannyBlur);
      CannyEdges(Image, Control, Opt.CannyLow, Opt.CannyHigh, false, 3);
      if (Opt.EdgesFile <> '') and
        not SaveImageFromVolumeIntoFile(Control, Opt.EdgesFile) then
        raise Exception.Create('could not save ' + Opt.EdgesFile);
    end
    else DrawEdgeMap(Control, ImageWidth, ImageHeight);
    ControlNet.SetControlImage(Control);
    ControlNet.ConditioningScale := Opt.ConditioningScale;
    ControlNet.GuidanceStart := Opt.ControlStart;
    ControlNet.GuidanceEnd := Opt.ControlEnd;
    ControlNet.GuessMode := Opt.GuessMode;
    Pipeline.Conditioner := ControlNet;
    if Opt.IPAdapterFile <> '' then
    begin
      IPAdapter := TStableDiffusionIPAdapter.Create(Opt.IPAdapterFile);
      if not LoadImageFromFileIntoVolume(Opt.IPImageFile, Image) then
        raise Exception.Create('could not read ' + Opt.IPImageFile);
      IPAdapter.SetImage(Image);
      IPAdapter.Scale := Opt.IPScale;
      Multi := TStableDiffusionMultiConditioner.Create();
      Multi.Add(ControlNet);
      Multi.Add(IPAdapter);
      Pipeline.Conditioner := Multi;
    end;
    Pipeline.GuidanceScale := Opt.GuidanceScale;
    Pipeline.NegativePrompt := Opt.NegativePrompt;
    Pipeline.UNetInt8 := Opt.UseInt8;
    Pipeline.TextEncoderInt8 := Opt.UseInt8;
    Pipeline.OnPhase := @Reporter.OnPhase;
    Pipeline.OnStep := @Reporter.OnStep;
    WriteLn('Model:      ', Opt.ModelFolder);
    WriteLn('ControlNet: ', Opt.ControlNetFolder);
    if Opt.ControlFile <> '' then WriteLn('Control:    ', Opt.ControlFile)
    else if Opt.PhotoFile <> '' then
      WriteLn(Format('Control:    canny edges of %s (low %.0f, high %.0f, ' +
        'blur %.2f)', [Opt.PhotoFile, Opt.CannyLow, Opt.CannyHigh,
        Opt.CannyBlur]))
    else WriteLn('Control:    drawn circle and square outlines');
    if (Opt.ControlStart > 0) or (Opt.ControlEnd < 1) or Opt.GuessMode then
      WriteLn(Format('Window:     steps %.2f..%.2f of the run, guess mode %s',
        [Opt.ControlStart, Opt.ControlEnd, BoolToStr(Opt.GuessMode, 'on',
        'off')]));
    if Assigned(IPAdapter) then
      WriteLn(Format('IP-Adapter: %s, image %s (%dx%d), scale %.2f',
        [Opt.IPAdapterFile, Opt.IPImageFile, Image.SizeX, Image.SizeY,
         Opt.IPScale]));
    WriteLn('Prompt:     "', Opt.Prompt, '"');
    WriteLn(Format('Image:      %dx%d, %d steps, guidance %.2f, scale %.2f, ' +
      'seed %d, %s', [ImageWidth, ImageHeight, Opt.Steps, Opt.GuidanceScale,
      Opt.ConditioningScale, Opt.Seed, ComputeText]));
    if Pipeline.SamplerNotice <> '' then
      WriteLn('Notice:     ', Pipeline.SamplerNotice);
    // Euler a draws its per-step noise from the global RNG.
    RandSeed := Opt.Seed xor $9E3779B9;
    StartTime := Now();
    Pipeline.Generate(Opt.Prompt, ImageWidth, ImageHeight, Opt.Steps,
      Opt.Seed, Image);
    Image.Mul(255.0);
    if not SaveImageFromVolumeIntoFile(Image, Opt.OutputFile) then
      raise Exception.Create('could not save ' + Opt.OutputFile);
    WriteLn(Format('Total:      %.1f s', [(Now() - StartTime) * 86400.0]));
    WriteLn('Wrote ', Opt.OutputFile);
  finally
    Image.Free;
    Control.Free;
    Reporter.Free;
    Pipeline.Free;
    Multi.Free;
    IPAdapter.Free;
    ControlNet.Free;
  end;
end;

var
  Options: TControlNetCannyOptions;
begin
  try
    if not ParseOptions(Options) then exit;
    if (Options.ModelFolder = PicoModelFolder) or
      (Options.ControlNetFolder = PicoControlNetFolder) then
      WriteLn('Pico fixture in use: random weights (wiring check only).');
    Run(Options);
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
