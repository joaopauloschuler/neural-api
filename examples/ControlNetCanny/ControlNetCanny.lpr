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
black, e.g. the output of cv2.Canny); it is resized to the image size. Without
it the example draws a circle and a square outline. ControlNet runs at square
sizes only.

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
  neuralnetwork, neuralvolume, neuraldatasets, neuralpretrained;

const
  PicoModelFolder = '../../tests/fixtures/tiny_sd15';
  PicoControlNetFolder = '../../tests/fixtures/tiny_sd15_controlnet';

type
  TControlNetCannyOptions = record
    ModelFolder, ControlNetFolder, ControlFile, Prompt, NegativePrompt,
      OutputFile: string;
    // ImageSize < 1: the checkpoint default (512 for SD 1.5).
    Steps, ImageSize: integer;
    GuidanceScale, ConditioningScale: double;
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
  WriteLn('  --prompt TEXT     prompt (default "a futuristic city, ',
    'detailed, sharp focus")');
  WriteLn('  --negative TEXT   negative prompt (default empty)');
  WriteLn('  --steps N         denoising steps (default 25)');
  WriteLn('  --guidance G      classifier-free guidance scale (default 7.5)');
  WriteLn('  --scale S         controlnet_conditioning_scale (default 1.0)');
  WriteLn('  --size N          square image side (default: the checkpoint''s)');
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
  Opt.Prompt := 'a futuristic city, detailed, sharp focus';
  Opt.NegativePrompt := '';
  Opt.OutputFile := 'controlnet_canny.png';
  Opt.Steps := 25;
  Opt.ImageSize := 0;
  Opt.GuidanceScale := 7.5;
  Opt.ConditioningScale := 1.0;
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
    else if Arg = '--prompt' then Opt.Prompt := NextArg()
    else if Arg = '--negative' then Opt.NegativePrompt := NextArg()
    else if Arg = '--out' then Opt.OutputFile := NextArg()
    else if Arg = '--steps' then Opt.Steps := StrToInt(NextArg())
    else if Arg = '--size' then Opt.ImageSize := StrToInt(NextArg())
    else if Arg = '--guidance' then
      Opt.GuidanceScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--scale' then
      Opt.ConditioningScale := StrToFloat(NextArg(), DefaultFormatSettings)
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
end;

// A (Size, Size, 3) edge map in 0..255: one-pixel white outlines of a circle
// and of a square on black.
procedure DrawEdgeMap(Image: TNNetVolume; Size: integer);
var
  X, Y, MaxPos, Center, Radius, SquareLow, SquareHigh: integer;
  IsEdge: boolean;
begin
  Image.ReSize(Size, Size, 3);
  Image.Fill(0);
  MaxPos := Size - 1;
  Center := Size div 2;
  Radius := Size * 3 div 8;
  SquareLow := Size div 4;
  SquareHigh := Size * 3 div 4;
  for Y := 0 to MaxPos do
    for X := 0 to MaxPos do
    begin
      IsEdge := Abs(Round(Sqrt(Sqr(X - Center) + Sqr(Y - Center))) - Radius) = 0;
      IsEdge := IsEdge or (((X = SquareLow) or (X = SquareHigh)) and
        (Y >= SquareLow) and (Y <= SquareHigh)) or
        (((Y = SquareLow) or (Y = SquareHigh)) and
        (X >= SquareLow) and (X <= SquareHigh));
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
  Reporter: TControlNetReporter;
  Control, Image: TNNetVolume;
  ComputeText: string;
  ImageSize: integer;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLText: string;
  {$ENDIF}
begin
  Pipeline := TStableDiffusionPipeline.Create(Opt.ModelFolder);
  ControlNet := TStableDiffusionControlNet.Create(Opt.ControlNetFolder);
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
    ImageSize := Opt.ImageSize;
    if ImageSize < 1 then ImageSize := Pipeline.DefaultImageSize;
    if Opt.ControlFile <> '' then
    begin
      if not LoadImageFromFileIntoVolume(Opt.ControlFile, Control) then
        raise Exception.Create('could not read ' + Opt.ControlFile);
    end
    else DrawEdgeMap(Control, ImageSize);
    ControlNet.SetControlImage(Control);
    ControlNet.ConditioningScale := Opt.ConditioningScale;
    Pipeline.Conditioner := ControlNet;
    Pipeline.GuidanceScale := Opt.GuidanceScale;
    Pipeline.NegativePrompt := Opt.NegativePrompt;
    Pipeline.UNetInt8 := Opt.UseInt8;
    Pipeline.TextEncoderInt8 := Opt.UseInt8;
    Pipeline.OnPhase := @Reporter.OnPhase;
    Pipeline.OnStep := @Reporter.OnStep;
    WriteLn('Model:      ', Opt.ModelFolder);
    WriteLn('ControlNet: ', Opt.ControlNetFolder);
    if Opt.ControlFile <> '' then WriteLn('Control:    ', Opt.ControlFile)
    else WriteLn('Control:    drawn circle and square outlines');
    WriteLn('Prompt:     "', Opt.Prompt, '"');
    WriteLn(Format('Image:      %dx%d, %d steps, guidance %.2f, scale %.2f, ' +
      'seed %d, %s', [ImageSize, ImageSize, Opt.Steps, Opt.GuidanceScale,
      Opt.ConditioningScale, Opt.Seed, ComputeText]));
    if Pipeline.SamplerNotice <> '' then
      WriteLn('Notice:     ', Pipeline.SamplerNotice);
    // Euler a draws its per-step noise from the global RNG.
    RandSeed := Opt.Seed xor $9E3779B9;
    StartTime := Now();
    Pipeline.Generate(Opt.Prompt, ImageSize, ImageSize, Opt.Steps, Opt.Seed,
      Image);
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
