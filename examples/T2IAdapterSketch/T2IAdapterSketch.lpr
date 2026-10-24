program T2IAdapterSketch;
(*
T2IAdapterSketch: Stable Diffusion 1.x text-to-image guided by a T2I-Adapter
(diffusers StableDiffusionAdapterPipeline), through TStableDiffusionPipeline
with a TStableDiffusionT2IAdapter conditioner in neuralpretrained.pas.

  * no --model / --adapter: the committed random pico fixtures
    tests/fixtures/tiny_sd15 + tests/fixtures/tiny_sd15_t2i_adapter (16x16,
    random weights, a one-channel sketch adapter): wiring and I/O only.
  * --model DIR: a diffusers Stable Diffusion 1.x folder; --adapter DIR: a
    diffusers T2IAdapter folder, e.g. TencentARC/t2iadapter_sketch_sd15v2.

--hint FILE is the hint image (for a sketch adapter: white strokes on black);
a one-channel adapter reads it as PIL 'L' gray, and it is resized to the image
size. Without it the example draws a house outline. Both sides must be a
multiple of the adapter's total downscale factor (64 for SD 1.5 adapters).

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
  PicoAdapterFolder = '../../tests/fixtures/tiny_sd15_t2i_adapter';

type
  TT2IAdapterSketchOptions = record
    ModelFolder, AdapterFolder, HintFile, Prompt, NegativePrompt,
      OutputFile: string;
    // A side < 1: the checkpoint default (512 for SD 1.5).
    Steps, ImageWidth, ImageHeight: integer;
    GuidanceScale, ConditioningScale: double;
    Seed: cardinal;
    UseOpenCL, UseInt8: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

  // Prints the pipeline's stages and steps as they run.
  TAdapterReporter = class(TObject)
  public
    procedure OnPhase(Phase: TStableDiffusionPipelinePhase);
    procedure OnStep(StepIndex, StepCount: integer; Timestep: double;
      Latents: TNNetVolume);
  end;

procedure TAdapterReporter.OnPhase(Phase: TStableDiffusionPipelinePhase);
const
  PhaseNames: array[TStableDiffusionPipelinePhase] of string = (
    'loading the text encoder', 'encoding the prompt',
    'loading the UNet, running the T2I-Adapter', 'denoising',
    'loading the VAE decoder', 'decoding', 'done');
begin
  WriteLn('  ', PhaseNames[Phase]);
end;

procedure TAdapterReporter.OnStep(StepIndex, StepCount: integer;
  Timestep: double; Latents: TNNetVolume);
begin
  WriteLn(Format('    step %d/%d  t=%.0f', [StepIndex + 1, StepCount, Timestep]));
end;

procedure PrintUsage();
begin
  WriteLn('Usage: T2IAdapterSketch [--model DIR] [--adapter DIR] [options]');
  WriteLn('  --model DIR       diffusers Stable Diffusion 1.x folder ',
    '(default: the pico fixture)');
  WriteLn('  --adapter DIR     diffusers T2IAdapter folder ',
    '(default: the pico fixture)');
  WriteLn('  --hint FILE       hint image, e.g. a sketch (default: a drawn ',
    'house outline)');
  WriteLn('  --prompt TEXT     prompt (default "a cozy cottage in a ',
    'meadow, watercolor")');
  WriteLn('  --negative TEXT   negative prompt (default empty)');
  WriteLn('  --steps N         denoising steps (default 25)');
  WriteLn('  --guidance G      classifier-free guidance scale (default 7.5)');
  WriteLn('  --scale S         adapter_conditioning_scale (default 1.0)');
  WriteLn('  --width N         image width (default: the checkpoint''s, ',
    '512 for SD 1.5)');
  WriteLn('  --height N        image height (default: the checkpoint''s)');
  WriteLn('  --size N          sets both --width and --height');
  WriteLn('  --seed N          noise seed (default 0)');
  WriteLn('  --out PNG         output file (default t2i_adapter_sketch.png)');
  WriteLn('  --int8            UNet and text encoder weights in int8');
  WriteLn('  --gpu             run the nets on OpenCL (needs a build with ',
    '-dOpenCL)');
  WriteLn('  --gpu-platform N  OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N    OpenCL device index (default 0)');
end;

function ParseOptions(out Opt: TT2IAdapterSketchOptions): boolean;
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
  Opt.AdapterFolder := PicoAdapterFolder;
  Opt.HintFile := '';
  Opt.Prompt := 'a cozy cottage in a meadow, watercolor';
  Opt.NegativePrompt := '';
  Opt.OutputFile := 't2i_adapter_sketch.png';
  Opt.Steps := 25;
  Opt.ImageWidth := 0;
  Opt.ImageHeight := 0;
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
    else if Arg = '--adapter' then Opt.AdapterFolder := NextArg()
    else if Arg = '--hint' then Opt.HintFile := NextArg()
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

// A (Width, Height, 1) sketch in 0..255: white one-pixel strokes of a house (a
// rectangular body under a triangular roof) on black.
procedure DrawHouseSketch(Image: TNNetVolume; Width, Height: integer);
var
  X, Y, MaxX, MaxY, Left, Right, Top, Bottom, Apex, Center: integer;
  IsStroke: boolean;
begin
  Image.ReSize(Width, Height, 1);
  Image.Fill(0);
  MaxX := Width - 1;
  MaxY := Height - 1;
  Left := Width div 4;
  Right := Width * 3 div 4;
  Top := Height div 2;
  Bottom := Height * 7 div 8;
  Apex := Height div 8;
  Center := Width div 2;
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      IsStroke := (((X = Left) or (X = Right)) and (Y >= Top) and
        (Y <= Bottom)) or (((Y = Top) or (Y = Bottom)) and (X >= Left) and
        (X <= Right));
      // Roof: the two lines from (Left, Top) and (Right, Top) to the apex.
      if (Y >= Apex) and (Y <= Top) then
        IsStroke := IsStroke or
          (X = Round(Center - (Center - Left) * (Y - Apex) / Max(1, Top - Apex))) or
          (X = Round(Center + (Right - Center) * (Y - Apex) / Max(1, Top - Apex)));
      if IsStroke then Image[X, Y, 0] := 255;
    end;
end;

procedure Run(const Opt: TT2IAdapterSketchOptions);
var
  Pipeline: TStableDiffusionPipeline;
  Adapter: TStableDiffusionT2IAdapter;
  Reporter: TAdapterReporter;
  Hint, Image: TNNetVolume;
  ComputeText: string;
  ImageWidth, ImageHeight: integer;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLText: string;
  {$ENDIF}
begin
  Pipeline := TStableDiffusionPipeline.Create(Opt.ModelFolder);
  Adapter := TStableDiffusionT2IAdapter.Create(Opt.AdapterFolder);
  Reporter := TAdapterReporter.Create;
  Hint := TNNetVolume.Create;
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
    if Opt.HintFile <> '' then
    begin
      if not LoadImageFromFileIntoVolume(Opt.HintFile, Hint) then
        raise Exception.Create('could not read ' + Opt.HintFile);
    end
    else DrawHouseSketch(Hint, ImageWidth, ImageHeight);
    Adapter.SetHintImage(Hint);
    Adapter.ConditioningScale := Opt.ConditioningScale;
    Pipeline.Conditioner := Adapter;
    Pipeline.GuidanceScale := Opt.GuidanceScale;
    Pipeline.NegativePrompt := Opt.NegativePrompt;
    Pipeline.UNetInt8 := Opt.UseInt8;
    Pipeline.TextEncoderInt8 := Opt.UseInt8;
    Pipeline.OnPhase := @Reporter.OnPhase;
    Pipeline.OnStep := @Reporter.OnStep;
    WriteLn('Model:    ', Opt.ModelFolder);
    WriteLn('Adapter:  ', Opt.AdapterFolder, ' (', Adapter.Config.CondChannels,
      '-channel hint)');
    if Opt.HintFile <> '' then WriteLn('Hint:     ', Opt.HintFile)
    else WriteLn('Hint:     drawn house outline');
    WriteLn('Prompt:   "', Opt.Prompt, '"');
    WriteLn(Format('Image:    %dx%d, %d steps, guidance %.2f, scale %.2f, ' +
      'seed %d, %s', [ImageWidth, ImageHeight, Opt.Steps, Opt.GuidanceScale,
      Opt.ConditioningScale, Opt.Seed, ComputeText]));
    if Pipeline.SamplerNotice <> '' then
      WriteLn('Notice:   ', Pipeline.SamplerNotice);
    // Euler a draws its per-step noise from the global RNG.
    RandSeed := Opt.Seed xor $9E3779B9;
    StartTime := Now();
    Pipeline.Generate(Opt.Prompt, ImageWidth, ImageHeight, Opt.Steps, Opt.Seed,
      Image);
    Image.Mul(255.0);
    if not SaveImageFromVolumeIntoFile(Image, Opt.OutputFile) then
      raise Exception.Create('could not save ' + Opt.OutputFile);
    WriteLn(Format('Total:    %.1f s', [(Now() - StartTime) * 86400.0]));
    WriteLn('Wrote ', Opt.OutputFile);
  finally
    Image.Free;
    Hint.Free;
    Reporter.Free;
    Pipeline.Free;
    Adapter.Free;
  end;
end;

var
  Options: TT2IAdapterSketchOptions;
begin
  try
    if not ParseOptions(Options) then exit;
    if (Options.ModelFolder = PicoModelFolder) or
      (Options.AdapterFolder = PicoAdapterFolder) then
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
