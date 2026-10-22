program DiffusionInpainting;
(*
DiffusionInpainting: Stable Diffusion 1.x inpainting (diffusers
StableDiffusionInpaintPipeline) through TStableDiffusionPipeline with a
TStableDiffusionInpainter conditioner in neuralpretrained.pas. The prompt
regenerates the white part of the mask; the rest follows the image. With
--strength below 1 the run starts from the image latents re-noised to a later
timestep and runs only the last steps.

  * a 9-channel inpainting UNet (e.g. runwayml/stable-diffusion-inpainting)
    sees latents | mask | masked-image latents at every step;
  * a standard 4-channel UNet (e.g. runwayml/stable-diffusion-v1-5) gets the
    kept region re-noised and blended back after every step.

  * no --model: the committed random pico folder tests/fixtures/tiny_sd15_inpaint
    (9 channels, 16x16, random weights): wiring and I/O only;
    tests/fixtures/tiny_sd15 runs the 4-channel mode.

Without --image / --mask the example uses a color ramp and a right-half mask.

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
  PicoModelFolder = '../../tests/fixtures/tiny_sd15_inpaint';

type
  TDiffusionInpaintingOptions = record
    ModelFolder, ImageFile, MaskFile, Prompt, NegativePrompt,
      OutputFile: string;
    // A side < 1: the checkpoint default (512 for SD 1.5).
    Steps, ImageWidth, ImageHeight: integer;
    GuidanceScale, Strength: double;
    Seed: cardinal;
    UseOpenCL, UseInt8: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

  // Prints the pipeline's stages and steps as they run.
  TInpaintReporter = class(TObject)
  public
    procedure OnPhase(Phase: TStableDiffusionPipelinePhase);
    procedure OnStep(StepIndex, StepCount: integer; Timestep: double;
      Latents: TNNetVolume);
  end;

procedure TInpaintReporter.OnPhase(Phase: TStableDiffusionPipelinePhase);
const
  PhaseNames: array[TStableDiffusionPipelinePhase] of string = (
    'loading the text encoder', 'encoding the prompt',
    'loading the UNet, encoding the image and the masked image', 'denoising',
    'loading the VAE decoder', 'decoding', 'done');
begin
  WriteLn('  ', PhaseNames[Phase]);
end;

procedure TInpaintReporter.OnStep(StepIndex, StepCount: integer;
  Timestep: double; Latents: TNNetVolume);
begin
  WriteLn(Format('    step %d/%d  t=%.0f', [StepIndex + 1, StepCount, Timestep]));
end;

procedure PrintUsage();
begin
  WriteLn('Usage: DiffusionInpainting [--model DIR] [options]');
  WriteLn('  --model DIR       diffusers Stable Diffusion 1.x folder, an ',
    'inpainting one or not');
  WriteLn('                    (default: the 9-channel pico fixture)');
  WriteLn('  --image FILE      image to edit (default: a color ramp)');
  WriteLn('  --mask FILE       mask, white = regenerate (default: the right ',
    'half)');
  WriteLn('  --prompt TEXT     prompt (default "a red brick wall")');
  WriteLn('  --negative TEXT   negative prompt (default empty)');
  WriteLn('  --steps N         denoising steps (default 25)');
  WriteLn('  --guidance G      classifier-free guidance scale (default 7.5)');
  WriteLn('  --strength S      0 < S <= 1: how far the image is re-noised; ',
    'runs int(steps * S) steps (default 1)');
  WriteLn('  --width N         image width (default: the checkpoint''s)');
  WriteLn('  --height N        image height (default: the checkpoint''s)');
  WriteLn('  --seed N          noise seed (default 0)');
  WriteLn('  --out PNG         output file (default inpainted.png)');
  WriteLn('  --int8            UNet and text encoder weights in int8');
  WriteLn('  --gpu             run the nets on OpenCL (needs a build with ',
    '-dOpenCL)');
  WriteLn('  --gpu-platform N  OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N    OpenCL device index (default 0)');
end;

function ParseOptions(out Opt: TDiffusionInpaintingOptions): boolean;
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
  Opt.ImageFile := '';
  Opt.MaskFile := '';
  Opt.Prompt := 'a red brick wall';
  Opt.NegativePrompt := '';
  Opt.OutputFile := 'inpainted.png';
  Opt.Steps := 25;
  Opt.ImageWidth := 0;
  Opt.ImageHeight := 0;
  Opt.GuidanceScale := 7.5;
  Opt.Strength := 1;
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
    else if Arg = '--image' then Opt.ImageFile := NextArg()
    else if Arg = '--mask' then Opt.MaskFile := NextArg()
    else if Arg = '--prompt' then Opt.Prompt := NextArg()
    else if Arg = '--negative' then Opt.NegativePrompt := NextArg()
    else if Arg = '--out' then Opt.OutputFile := NextArg()
    else if Arg = '--steps' then Opt.Steps := StrToInt(NextArg())
    else if Arg = '--width' then Opt.ImageWidth := StrToInt(NextArg())
    else if Arg = '--height' then Opt.ImageHeight := StrToInt(NextArg())
    else if Arg = '--guidance' then
      Opt.GuidanceScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--strength' then
      Opt.Strength := StrToFloat(NextArg(), DefaultFormatSettings)
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
  if not ((Opt.Strength > 0) and (Opt.Strength <= 1)) then
    raise Exception.Create('--strength must be in (0, 1]');
end;

// A (Width, Height, 3) color ramp in 0..255: red left to right, green top to
// bottom, a blue ripple.
procedure DrawColorRamp(Image: TNNetVolume; Width, Height: integer);
var
  X, Y, MaxX, MaxY: integer;
begin
  Image.ReSize(Width, Height, 3);
  MaxX := Width - 1;
  MaxY := Height - 1;
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      Image[X, Y, 0] := Round(255 * X / Max(1, MaxX));
      Image[X, Y, 1] := Round(255 * Y / Max(1, MaxY));
      Image[X, Y, 2] := Round(127.5 + 100 * Sin((X + Y) * 0.6));
    end;
end;

// A (Width, Height, 1) mask in 0..255: white (regenerate) on the right half.
procedure DrawRightHalfMask(Mask: TNNetVolume; Width, Height: integer);
var
  X, Y, MaxX, MaxY: integer;
begin
  Mask.ReSize(Width, Height, 1);
  Mask.Fill(0);
  MaxX := Width - 1;
  MaxY := Height - 1;
  for Y := 0 to MaxY do
    for X := Width div 2 to MaxX do
      Mask[X, Y, 0] := 255;
end;

procedure LoadOrDie(const FileName: string; Image: TNNetVolume);
begin
  if not LoadImageFromFileIntoVolume(FileName, Image) then
    raise Exception.Create('could not read ' + FileName);
end;

procedure Run(const Opt: TDiffusionInpaintingOptions);
var
  Pipeline: TStableDiffusionPipeline;
  Inpainter: TStableDiffusionInpainter;
  Reporter: TInpaintReporter;
  SourceImage, Mask, Image: TNNetVolume;
  ComputeText, ModeText: string;
  ImageWidth, ImageHeight: integer;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLText: string;
  {$ENDIF}
begin
  Pipeline := TStableDiffusionPipeline.Create(Opt.ModelFolder);
  Inpainter := TStableDiffusionInpainter.Create();
  Reporter := TInpaintReporter.Create;
  SourceImage := TNNetVolume.Create;
  Mask := TNNetVolume.Create;
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
    if Opt.ImageFile <> '' then LoadOrDie(Opt.ImageFile, SourceImage)
    else DrawColorRamp(SourceImage, ImageWidth, ImageHeight);
    if Opt.MaskFile <> '' then LoadOrDie(Opt.MaskFile, Mask)
    else DrawRightHalfMask(Mask, ImageWidth, ImageHeight);
    Inpainter.SetImage(SourceImage);
    Inpainter.SetMask(Mask);
    Inpainter.Strength := Opt.Strength;
    Pipeline.Conditioner := Inpainter;
    Pipeline.GuidanceScale := Opt.GuidanceScale;
    Pipeline.NegativePrompt := Opt.NegativePrompt;
    Pipeline.UNetInt8 := Opt.UseInt8;
    Pipeline.TextEncoderInt8 := Opt.UseInt8;
    Pipeline.OnPhase := @Reporter.OnPhase;
    Pipeline.OnStep := @Reporter.OnStep;
    if Pipeline.UNetConfig.InChannels > Pipeline.UNetConfig.OutChannels then
      ModeText := 'inpainting UNet (latents | mask | masked-image latents)'
    else ModeText := 'standard UNet, kept region blended after each step';
    WriteLn('Model:    ', Opt.ModelFolder);
    WriteLn('Mode:     ', ModeText);
    if Opt.ImageFile <> '' then WriteLn('Image:    ', Opt.ImageFile)
    else WriteLn('Image:    a color ramp');
    if Opt.MaskFile <> '' then WriteLn('Mask:     ', Opt.MaskFile)
    else WriteLn('Mask:     the right half');
    WriteLn('Prompt:   "', Opt.Prompt, '"');
    WriteLn(Format('Output:   %dx%d, %d steps, strength %.2f, guidance %.2f, ' +
      'seed %d, %s', [ImageWidth, ImageHeight, Opt.Steps, Opt.Strength,
      Opt.GuidanceScale, Opt.Seed, ComputeText]));
    if Pipeline.SamplerNotice <> '' then
      WriteLn('Notice:   ', Pipeline.SamplerNotice);
    // Euler a and the VAE latent samples draw their noise from the global RNG.
    RandSeed := Opt.Seed xor $9E3779B9;
    StartTime := Now();
    Pipeline.Generate(Opt.Prompt, ImageWidth, ImageHeight, Opt.Steps,
      Opt.Seed, Image);
    Image.Mul(255.0);
    if not SaveImageFromVolumeIntoFile(Image, Opt.OutputFile) then
      raise Exception.Create('could not save ' + Opt.OutputFile);
    WriteLn(Format('Total:    %.1f s', [(Now() - StartTime) * 86400.0]));
    WriteLn('Wrote ', Opt.OutputFile);
  finally
    Image.Free;
    Mask.Free;
    SourceImage.Free;
    Reporter.Free;
    Pipeline.Free;
    Inpainter.Free;
  end;
end;

var
  Options: TDiffusionInpaintingOptions;
begin
  try
    if not ParseOptions(Options) then exit;
    if Options.ModelFolder = PicoModelFolder then
      WriteLn('No --model: the random pico fixture (wiring check only).');
    Run(Options);
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
