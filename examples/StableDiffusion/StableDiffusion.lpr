program StableDiffusion;
(*
StableDiffusion: text-to-image with a diffusers Stable Diffusion 1.x folder
through TStableDiffusionPipeline in neuralpretrained.pas.

  * no --model: runs the committed random pico folder tests/fixtures/tiny_sd15
    (16x16, random weights): this checks wiring and I/O only.
  * --model DIR: a diffusers StableDiffusionPipeline folder (model_index.json,
    text_encoder/, tokenizer/, unet/, vae/, scheduler/), e.g.
    runwayml/stable-diffusion-v1-5 with F32 *.safetensors weights.

The scheduler config gives the betas, timestep spacing and prediction type,
and its _class_name the sampler (DDIM, PNDM/PLMS, Euler, Euler a or
DPM-Solver++ 2M; other classes run DDIM). --sampler overrides it.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  {$IFDEF OpenCL}neuralopencl,{$ENDIF}
  neuralnetwork, neuralvolume, neuraldatasets, neuraldiffusion,
  neuralpretrained;

const
  PicoModelFolder = '../../tests/fixtures/tiny_sd15';
  // --sampler names of the methods TStableDiffusionPipeline maps to.
  SamplerOptionNames: array[0..4] of string =
    ('ddim', 'pndm', 'euler', 'euler-a', 'dpm++2m');
  SamplerOptionMethods: array[0..4] of TNNetSamplerMethod =
    (smDDIM, smPNDM, smEuler, smEulerAncestral, smDPMSolverPP2M);

type
  TStableDiffusionOptions = record
    ModelFolder, Prompt, NegativePrompt, OutputFile: string;
    // 'auto' (the scheduler config's) or one of SamplerOptionNames.
    SamplerName: string;
    // ImageSize < 1: the checkpoint default (unet sample_size * VAE factor).
    Steps, ImageSize: integer;
    GuidanceScale: double;
    Seed: cardinal;
    UseOpenCL: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

  // Prints the pipeline's stages and steps as they run.
  TStableDiffusionReporter = class(TObject)
  private
    FPhaseStart: TDateTime;
  public
    procedure OnPhase(Phase: TStableDiffusionPipelinePhase);
    procedure OnStep(StepIndex, StepCount: integer; Timestep: double;
      Latents: TNNetVolume);
  end;

procedure TStableDiffusionReporter.OnPhase(
  Phase: TStableDiffusionPipelinePhase);
const
  PhaseNames: array[TStableDiffusionPipelinePhase] of string = (
    'loading the text encoder', 'encoding the prompt', 'loading the UNet',
    'denoising', 'loading the VAE decoder', 'decoding', 'done');
begin
  if FPhaseStart <> 0 then
    WriteLn(Format('    (%.1f s)', [(Now() - FPhaseStart) * 86400.0]));
  FPhaseStart := Now();
  WriteLn('  ', PhaseNames[Phase]);
  if Phase = sdphDone then FPhaseStart := 0;
end;

procedure TStableDiffusionReporter.OnStep(StepIndex, StepCount: integer;
  Timestep: double; Latents: TNNetVolume);
begin
  WriteLn(Format('    step %d/%d  t=%.0f', [StepIndex + 1, StepCount, Timestep]));
end;

procedure PrintUsage();
begin
  WriteLn('Usage: StableDiffusion [--model DIR] [options]');
  WriteLn('  --model DIR       diffusers StableDiffusionPipeline folder ',
    '(default: the pico fixture)');
  WriteLn('  --prompt TEXT     prompt (default "a photo of an astronaut ',
    'riding a horse on mars")');
  WriteLn('  --negative TEXT   negative prompt (default empty)');
  WriteLn('  --steps N         denoising steps (default 25)');
  WriteLn('  --guidance G      classifier-free guidance scale (default 7.5; ',
    '<= 1 turns it off)');
  WriteLn('  --size N          square image side (default: the checkpoint''s, ',
    '512 for SD 1.5)');
  WriteLn('  --seed N          noise seed (default 0)');
  WriteLn('  --sampler NAME    auto (default: from the scheduler config), ddim,');
  WriteLn('                    pndm, euler, euler-a or dpm++2m');
  WriteLn('  --output PNG      output file (default stable_diffusion.png)');
  WriteLn('  --gpu             run the nets on OpenCL (needs a build with ',
    '-dOpenCL)');
  WriteLn('  --gpu-platform N  OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N    OpenCL device index (default 0)');
end;

// Position of Name in SamplerOptionNames, or -1.
function SamplerOptionIndex(const Name: string): integer;
var
  NamePos: integer;
begin
  Result := -1;
  for NamePos := 0 to High(SamplerOptionNames) do
    if SamplerOptionNames[NamePos] = Name then Result := NamePos;
end;

function SamplerText(Method: TNNetSamplerMethod): string;
var
  NamePos: integer;
begin
  Result := 'method ' + IntToStr(Ord(Method));
  for NamePos := 0 to High(SamplerOptionMethods) do
    if SamplerOptionMethods[NamePos] = Method then
      Result := SamplerOptionNames[NamePos];
end;

function ParseOptions(out Opt: TStableDiffusionOptions): boolean;
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
  Opt.Prompt := 'a photo of an astronaut riding a horse on mars';
  Opt.NegativePrompt := '';
  Opt.OutputFile := 'stable_diffusion.png';
  Opt.SamplerName := 'auto';
  Opt.Steps := 25;
  Opt.ImageSize := 0;
  Opt.GuidanceScale := 7.5;
  Opt.Seed := 0;
  Opt.UseOpenCL := false;
  Opt.OpenCLPlatform := 0;
  Opt.OpenCLDevice := 0;
  Result := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--model' then Opt.ModelFolder := NextArg()
    else if Arg = '--prompt' then Opt.Prompt := NextArg()
    else if Arg = '--negative' then Opt.NegativePrompt := NextArg()
    else if Arg = '--output' then Opt.OutputFile := NextArg()
    else if Arg = '--steps' then Opt.Steps := StrToInt(NextArg())
    else if Arg = '--size' then Opt.ImageSize := StrToInt(NextArg())
    else if Arg = '--guidance' then
      Opt.GuidanceScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--seed' then Opt.Seed := StrToInt64(NextArg())
    else if Arg = '--sampler' then Opt.SamplerName := LowerCase(NextArg())
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
  if (Opt.SamplerName <> 'auto') and
    (SamplerOptionIndex(Opt.SamplerName) < 0) then
    raise Exception.Create('unknown --sampler: ' + Opt.SamplerName);
end;

procedure Run(const Opt: TStableDiffusionOptions);
var
  Pipeline: TStableDiffusionPipeline;
  Reporter: TStableDiffusionReporter;
  Image: TNNetVolume;
  ComputeText: string;
  ImageSize: integer;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  OpenCLProblem: string;
  OpenCLPlatform, OpenCLDevice: integer;
  {$ENDIF}
begin
  Pipeline := TStableDiffusionPipeline.Create(Opt.ModelFolder);
  Reporter := TStableDiffusionReporter.Create;
  Image := TNNetVolume.Create;
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
    ComputeText := 'CPU';
    if Opt.UseOpenCL then
    begin
      {$IFDEF OpenCL}
      OpenCLDevices := TEasyOpenCL.Create();
      OpenCLPlatform := Opt.OpenCLPlatform;
      OpenCLDevice := Opt.OpenCLDevice;
      if OpenCLDevices.SelectPlatformAndDevice(OpenCLPlatform, OpenCLDevice,
        OpenCLProblem) then
      begin
        Pipeline.EnableOpenCL(OpenCLDevices.PlatformIds[OpenCLPlatform],
          OpenCLDevices.Devices[OpenCLDevice]);
        ComputeText := 'OpenCL on ' +
          OpenCLDevices.PlatformNames[OpenCLPlatform] + ' / ' +
          OpenCLDevices.DeviceNames[OpenCLDevice];
      end
      else
        ComputeText := 'CPU (' + OpenCLProblem + ')';
      {$ELSE}
      ComputeText := 'CPU (built without -dOpenCL)';
      {$ENDIF}
    end;
    ImageSize := Opt.ImageSize;
    if ImageSize < 1 then ImageSize := Pipeline.DefaultImageSize;
    Pipeline.GuidanceScale := Opt.GuidanceScale;
    Pipeline.NegativePrompt := Opt.NegativePrompt;
    Pipeline.OnPhase := @Reporter.OnPhase;
    Pipeline.OnStep := @Reporter.OnStep;
    WriteLn('Model:    ', Opt.ModelFolder);
    WriteLn('Prompt:   "', Opt.Prompt, '"');
    WriteLn(Format('Image:    %dx%d, %d steps, guidance %.2f, seed %d, %s',
      [ImageSize, ImageSize, Opt.Steps, Opt.GuidanceScale, Opt.Seed,
       ComputeText]));
    if Opt.SamplerName <> 'auto' then
      Pipeline.Sampler := SamplerOptionMethods[
        SamplerOptionIndex(Opt.SamplerName)];
    WriteLn('Sampler:  ', SamplerText(Pipeline.Sampler), ' with the ',
      Pipeline.SchedulerClassName, ' config''s betas and timestep spacing');
    if (Opt.SamplerName = 'auto') and (Pipeline.SamplerNotice <> '') then
      WriteLn('Notice:   ', Pipeline.SamplerNotice);
    // Euler a draws its per-step noise from the global RNG; the initial
    // latents use Opt.Seed itself, so this stream starts elsewhere.
    RandSeed := Opt.Seed xor $9E3779B9;
    StartTime := Now();
    Pipeline.Generate(Opt.Prompt, ImageSize, ImageSize, Opt.Steps, Opt.Seed,
      Image);
    WriteLn(Format('Total:    %.1f s', [(Now() - StartTime) * 86400.0]));
    Image.Mul(255.0);
    if not SaveImageFromVolumeIntoFile(Image, Opt.OutputFile) then
      raise Exception.Create('could not save ' + Opt.OutputFile);
    WriteLn('Wrote ', Opt.OutputFile);
  finally
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    Image.Free;
    Reporter.Free;
    Pipeline.Free;
  end;
end;

var
  Options: TStableDiffusionOptions;
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
