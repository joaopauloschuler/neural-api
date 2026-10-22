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

Each component is loaded when its stage starts and freed when it ends, so
only one is in memory at a time. --repeat N > 1 makes N images from one
prompt encode and keeps the three components loaded. --int8 stores the UNet
and text encoder weights in int8. --stats prints a per-image stage table
(wall ms; host<->OpenCL MB with --gpu). --embedding FILE[:TOKEN] (repeatable)
loads a textual inversion concept, as diffusers load_textual_inversion.
--ip-adapter FILE --ip-image PNG [--ip-scale S] adds an image prompt through an
h94/IP-Adapter ip-adapter_sd15 file (its image_encoder/ folder beside it), as
diffusers load_ip_adapter + set_ip_adapter_scale. --init-image PNG
[--strength S] turns it into image-to-image (diffusers
StableDiffusionImg2ImgPipeline): the output takes the image's size.

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
    // --ip-adapter weights file and --ip-image; '' for none.
    IPAdapterFile, IPImageFile: string;
    IPScale: double;
    // --init-image ('' for text-to-image) and its --strength.
    InitImageFile: string;
    Strength: double;
    // --embedding values: FILE or FILE:TOKEN, in load order.
    Embeddings: array of string;
    // 'auto' (the scheduler config's) or one of SamplerOptionNames.
    SamplerName: string;
    // A side < 1: the checkpoint default (unet sample_size * VAE factor).
    Steps, ImageWidth, ImageHeight, RepeatCount: integer;
    GuidanceScale: double;
    Seed: cardinal;
    UseOpenCL, UseInt8, UseStats: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

  // Prints the pipeline's stages and steps as they run.
  TStableDiffusionReporter = class(TObject)
  public
    procedure OnPhase(Phase: TStableDiffusionPipelinePhase);
    procedure OnStep(StepIndex, StepCount: integer; Timestep: double;
      Latents: TNNetVolume);
  end;

const
  PhaseNames: array[TStableDiffusionPipelinePhase] of string = (
    'loading the text encoder', 'encoding the prompt', 'loading the UNet',
    'denoising', 'loading the VAE decoder', 'decoding', 'done');
  PhaseRowNames: array[TStableDiffusionPipelinePhase] of string = (
    'load text encoder', 'encode prompt', 'load UNet', 'denoise', 'load VAE',
    'VAE decode', 'done');
  BytesPerMB = 1024 * 1024;

procedure TStableDiffusionReporter.OnPhase(
  Phase: TStableDiffusionPipelinePhase);
begin
  WriteLn('  ', PhaseNames[Phase]);
end;

// The --stats table of image ImageNumber: one row per phase that ran (the
// prompt rows only when HasEncode), then the PNG save and the sum.
procedure PrintStageTable(const Stats: TStableDiffusionImageStats;
  ImageNumber: integer; HasEncode: boolean; PngSaveMs: double);
var
  Phase: TStableDiffusionPipelinePhase;
  TotalMs: double;
  Detail: string;

  procedure AddRow(const Stage, Ms, Pct, RowDetail: string);
  begin
    WriteLn(TrimRight(Format('  %-18s %10s %6s  %s', [Stage, Ms, Pct,
      RowDetail])));
  end;

  procedure AddTimedRow(const Stage: string; Ms: double;
    const RowDetail: string);
  var
    Share: double;
  begin
    if TotalMs > 0 then Share := 100.0 * Ms / TotalMs else Share := 0;
    AddRow(Stage, Format('%.1f', [Ms]), Format('%.1f', [Share]), RowDetail);
  end;

begin
  TotalMs := PngSaveMs;
  for Phase := Low(TStableDiffusionPipelinePhase) to sdphDecode do
    if HasEncode or not (Phase in [sdphLoadTextEncoder, sdphEncodePrompt]) then
      TotalMs := TotalMs + Stats.PhaseMs[Phase];
  WriteLn('Stages of image ', ImageNumber, ' (wall time):');
  AddRow('Stage', 'ms', '%', '');
  for Phase := Low(TStableDiffusionPipelinePhase) to sdphDecode do
  begin
    if not HasEncode and (Phase in [sdphLoadTextEncoder, sdphEncodePrompt]) then
    begin
      if Phase = sdphEncodePrompt then
        AddRow(PhaseRowNames[Phase], '-', '-', 'encoded once, before image 1');
      continue;
    end;
    // A load the kept components skipped is left out.
    if (Stats.PhaseMs[Phase] = 0) and (Phase in [sdphLoadTextEncoder,
      sdphLoadUNet, sdphLoadVae]) then continue;
    Detail := '';
    if (Phase = sdphDenoise) and (Stats.UNetPassCount > 0) then
      Detail := Format('%d UNet passes x %.1f ms', [Stats.UNetPassCount,
        Stats.PhaseMs[Phase] / Stats.UNetPassCount]);
    if Stats.TransfersCounted and ((Stats.PhaseUploadBytes[Phase] > 0) or
      (Stats.PhaseDownloadBytes[Phase] > 0)) then
      Detail := Trim(Detail + Format('  up %.2f MB, down %.2f MB',
        [Stats.PhaseUploadBytes[Phase] / BytesPerMB,
         Stats.PhaseDownloadBytes[Phase] / BytesPerMB]));
    AddTimedRow(PhaseRowNames[Phase], Stats.PhaseMs[Phase], Detail);
  end;
  AddTimedRow('PNG save', PngSaveMs, '');
  AddRow('sum of stages', Format('%.1f', [TotalMs]), '100.0', '');
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
  WriteLn('  --width N         image width (default: the checkpoint''s, ',
    '512 for SD 1.5)');
  WriteLn('  --height N        image height (default: the checkpoint''s)');
  WriteLn('  --size N          sets both --width and --height');
  WriteLn('  --seed N          noise seed (default 0)');
  WriteLn('  --sampler NAME    auto (default: from the scheduler config), ddim,');
  WriteLn('                    pndm, euler, euler-a or dpm++2m');
  WriteLn('  --output PNG      output file (default stable_diffusion.png)');
  WriteLn('  --repeat N        N images, seeds --seed, --seed+1, ...; the ',
    'prompt is encoded');
  WriteLn('                    once. N > 1 keeps the components loaded and ',
    'numbers the files');
  WriteLn('                    (stable_diffusion_0001.png, ...; default 1)');
  WriteLn('  --embedding F[:T] textual inversion embedding (diffusers ',
    'learned_embeds');
  WriteLn('                    .safetensors/.bin or A1111 .pt) under token T ',
    '(default: the');
  WriteLn('                    file''s); use the token in the prompts; ',
    'repeatable');
  WriteLn('  --ip-adapter FILE IP-Adapter ip-adapter_sd15 (.safetensors/.bin) ',
    'with its');
  WriteLn('                    image_encoder/ folder beside it; needs ',
    '--ip-image');
  WriteLn('  --ip-image PNG    the image prompt of --ip-adapter');
  WriteLn('  --ip-scale S      IP-Adapter scale (default 1.0; 0 ignores the ',
    'image)');
  WriteLn('  --init-image PNG  image-to-image from this image; the output ',
    'takes its size');
  WriteLn('                    (rounded down to the VAE factor) unless ',
    '--width/--height');
  WriteLn('  --strength S      how much of --init-image to repaint, (0, 1] ',
    '(default 0.8);');
  WriteLn('                    runs the last int(steps * S) steps');
  WriteLn('  --int8            UNet and text encoder weights in int8 ',
    '(default FP32)');
  WriteLn('  --stats           after each image, a stage table (wall ms; ',
    'with --gpu also');
  WriteLn('                    the host<->OpenCL MB of each stage)');
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
  Opt.IPAdapterFile := '';
  Opt.IPImageFile := '';
  Opt.IPScale := 1.0;
  Opt.InitImageFile := '';
  Opt.Strength := -1;
  Opt.SamplerName := 'auto';
  Opt.Steps := 25;
  Opt.ImageWidth := 0;
  Opt.ImageHeight := 0;
  Opt.GuidanceScale := 7.5;
  Opt.Seed := 0;
  Opt.RepeatCount := 1;
  Opt.UseOpenCL := false;
  Opt.UseInt8 := false;
  Opt.UseStats := false;
  Opt.OpenCLPlatform := 0;
  Opt.OpenCLDevice := 0;
  SetLength(Opt.Embeddings, 0);
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
    else if Arg = '--width' then Opt.ImageWidth := StrToInt(NextArg())
    else if Arg = '--height' then Opt.ImageHeight := StrToInt(NextArg())
    else if Arg = '--size' then
    begin
      Opt.ImageWidth := StrToInt(NextArg());
      Opt.ImageHeight := Opt.ImageWidth;
    end
    else if Arg = '--guidance' then
      Opt.GuidanceScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--seed' then Opt.Seed := StrToInt64(NextArg())
    else if Arg = '--sampler' then Opt.SamplerName := LowerCase(NextArg())
    else if Arg = '--repeat' then Opt.RepeatCount := StrToInt(NextArg())
    else if Arg = '--embedding' then
    begin
      SetLength(Opt.Embeddings, Length(Opt.Embeddings) + 1);
      Opt.Embeddings[High(Opt.Embeddings)] := NextArg();
    end
    else if Arg = '--ip-adapter' then Opt.IPAdapterFile := NextArg()
    else if Arg = '--ip-image' then Opt.IPImageFile := NextArg()
    else if Arg = '--ip-scale' then
      Opt.IPScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--init-image' then Opt.InitImageFile := NextArg()
    else if Arg = '--strength' then
      Opt.Strength := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--int8' then Opt.UseInt8 := true
    else if Arg = '--stats' then Opt.UseStats := true
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
  if Opt.RepeatCount < 1 then raise Exception.Create('--repeat must be >= 1');
  if (Opt.IPAdapterFile = '') <> (Opt.IPImageFile = '') then
    raise Exception.Create('--ip-adapter and --ip-image go together');
  if (Opt.Strength >= 0) and (Opt.InitImageFile = '') then
    raise Exception.Create('--strength needs --init-image');
  if (Opt.InitImageFile <> '') and (Opt.IPAdapterFile <> '') then
    raise Exception.Create('--init-image and --ip-adapter cannot be combined');
  if (Opt.SamplerName <> 'auto') and
    (SamplerOptionIndex(Opt.SamplerName) < 0) then
    raise Exception.Create('unknown --sampler: ' + Opt.SamplerName);
end;

// Splits an --embedding value at its last ':' unless that is a drive letter's
// or is followed by a path; Token '' leaves the token to the file.
procedure SplitEmbeddingSpec(const Spec: string; out FileName, Token: string);
var
  ColonPos: integer;
begin
  FileName := Spec;
  Token := '';
  ColonPos := LastDelimiter(':', Spec);
  if (ColonPos <= 2) or (LastDelimiter('/\', Spec) > ColonPos) then exit;
  FileName := Copy(Spec, 1, ColonPos - 1);
  Token := Copy(Spec, ColonPos + 1, Length(Spec));
  if Token = '' then
    raise Exception.Create('--embedding ' + Spec + ': empty token');
end;

// --output for one image; else Base with _0001, _0002, ... before the extension.
function ImageFileName(const Base: string; ImagePos, ImageCount: integer): string;
begin
  if ImageCount = 1 then exit(Base);
  Result := ChangeFileExt(Base, '') + Format('_%.4d', [ImagePos + 1]) +
    ExtractFileExt(Base);
end;

procedure Run(const Opt: TStableDiffusionOptions);
var
  Pipeline: TStableDiffusionPipeline;
  Reporter: TStableDiffusionReporter;
  IPAdapter: TStableDiffusionIPAdapter;
  Img2Img: TStableDiffusionImg2Img;
  Image, PromptEmbeds: TNNetVolume;
  ComputeText, ImageFile, EmbeddingFile, EmbeddingToken: string;
  ImageWidth, ImageHeight, ImagePos, EmbeddingPos: integer;
  Seed: cardinal;
  StartTime, SaveStart: TDateTime;
  PngSaveMs: double;
  {$IFDEF OpenCL}
  OpenCLText: string;
  {$ENDIF}
begin
  Pipeline := TStableDiffusionPipeline.Create(Opt.ModelFolder);
  Reporter := TStableDiffusionReporter.Create;
  IPAdapter := nil;
  Img2Img := nil;
  Image := TNNetVolume.Create;
  PromptEmbeds := TNNetVolume.Create;
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
    ImageWidth := Pipeline.DefaultImageSize;
    ImageHeight := Pipeline.DefaultImageSize;
    if Opt.InitImageFile <> '' then
    begin
      Img2Img := TStableDiffusionImg2Img.Create();
      if not LoadImageFromFileIntoVolume(Opt.InitImageFile, Image) then
        raise Exception.Create('could not load ' + Opt.InitImageFile);
      Img2Img.SetImage(Image);
      if Opt.Strength >= 0 then Img2Img.Strength := Opt.Strength;
      Img2Img.ImageSizeFor(Pipeline.VaeScaleFactor, ImageWidth, ImageHeight);
      Pipeline.Conditioner := Img2Img;
    end;
    if Opt.ImageWidth >= 1 then ImageWidth := Opt.ImageWidth;
    if Opt.ImageHeight >= 1 then ImageHeight := Opt.ImageHeight;
    Pipeline.GuidanceScale := Opt.GuidanceScale;
    Pipeline.NegativePrompt := Opt.NegativePrompt;
    Pipeline.UNetInt8 := Opt.UseInt8;
    Pipeline.TextEncoderInt8 := Opt.UseInt8;
    {$IFDEF OpenCL}
    if Opt.UseStats then OpenCLTransferCounting := true;
    {$ENDIF}
    Pipeline.OnPhase := @Reporter.OnPhase;
    Pipeline.OnStep := @Reporter.OnStep;
    WriteLn('Model:    ', Opt.ModelFolder);
    WriteLn('Prompt:   "', Opt.Prompt, '"');
    WriteLn(Format('Image:    %dx%d, %d steps, guidance %.2f, seed %d, %s',
      [ImageWidth, ImageHeight, Opt.Steps, Opt.GuidanceScale, Opt.Seed,
       ComputeText]));
    if Opt.SamplerName <> 'auto' then
      Pipeline.Sampler := SamplerOptionMethods[
        SamplerOptionIndex(Opt.SamplerName)];
    WriteLn('Sampler:  ', SamplerText(Pipeline.Sampler), ' with the ',
      Pipeline.SchedulerClassName, ' config''s betas and timestep spacing');
    if (Opt.SamplerName = 'auto') and (Pipeline.SamplerNotice <> '') then
      WriteLn('Notice:   ', Pipeline.SamplerNotice);
    if Opt.UseInt8 then WriteLn('Weights:  UNet int8, text encoder int8')
    else WriteLn('Weights:  FP32');
    for EmbeddingPos := 0 to High(Opt.Embeddings) do
    begin
      SplitEmbeddingSpec(Opt.Embeddings[EmbeddingPos], EmbeddingFile,
        EmbeddingToken);
      EmbeddingToken := Pipeline.LoadTextualInversion(EmbeddingFile,
        EmbeddingToken);
      WriteLn('Embedding: ', EmbeddingToken, ' from ', EmbeddingFile);
    end;
    if Assigned(Img2Img) then
      WriteLn(Format('Init image: %s (%dx%d), strength %.2f, %d of %d steps',
        [Opt.InitImageFile, Image.SizeX, Image.SizeY, Img2Img.Strength,
         Opt.Steps - TNNetDiffusionScheduler.Img2ImgStartStep(Opt.Steps,
         Img2Img.Strength), Opt.Steps]));
    if Opt.IPAdapterFile <> '' then
    begin
      IPAdapter := TStableDiffusionIPAdapter.Create(Opt.IPAdapterFile);
      if not LoadImageFromFileIntoVolume(Opt.IPImageFile, Image) then
        raise Exception.Create('could not load ' + Opt.IPImageFile);
      IPAdapter.SetImage(Image);
      IPAdapter.Scale := Opt.IPScale;
      Pipeline.Conditioner := IPAdapter;
      WriteLn(Format('IP-Adapter: %s, image %s (%dx%d), scale %.2f',
        [Opt.IPAdapterFile, Opt.IPImageFile, Image.SizeX, Image.SizeY,
         Opt.IPScale]));
    end;
    if Opt.RepeatCount > 1 then
    begin
      WriteLn('Loading the text encoder, the UNet and the VAE decoder to ',
        'keep them in memory.');
      StartTime := Now();
      Pipeline.LoadComponents(ImageWidth, ImageHeight);
      WriteLn(Format('Loaded:   %.1f s', [(Now() - StartTime) * 86400.0]));
      Pipeline.EncodePrompt(Opt.Prompt, PromptEmbeds);
    end;
    Seed := Opt.Seed;
    for ImagePos := 0 to Opt.RepeatCount - 1 do
    begin
      ImageFile := ImageFileName(Opt.OutputFile, ImagePos, Opt.RepeatCount);
      if Opt.RepeatCount > 1 then
        WriteLn('Image ', ImagePos + 1, '/', Opt.RepeatCount, ', seed ', Seed);
      // Euler a and the img2img VAE sample draw their noise from the global
      // RNG; the initial latents use Seed itself, so this stream starts elsewhere.
      RandSeed := Seed xor $9E3779B9;
      StartTime := Now();
      if Opt.RepeatCount > 1 then
        Pipeline.GenerateFromEmbeds(PromptEmbeds, ImageWidth, ImageHeight,
          Opt.Steps, Seed, Image)
      else
        Pipeline.Generate(Opt.Prompt, ImageWidth, ImageHeight, Opt.Steps,
          Seed, Image);
      SaveStart := Now();
      Image.Mul(255.0);
      if not SaveImageFromVolumeIntoFile(Image, ImageFile) then
        raise Exception.Create('could not save ' + ImageFile);
      PngSaveMs := (Now() - SaveStart) * MSecsPerDay;
      WriteLn(Format('Total:    %.1f s', [(Now() - StartTime) * 86400.0]));
      WriteLn('Wrote ', ImageFile);
      if Opt.UseStats then
        PrintStageTable(Pipeline.ImageStats, ImagePos + 1,
          (Opt.RepeatCount = 1) or (ImagePos = 0), PngSaveMs);
      // The seed wraps from High(cardinal) to 0.
      {$PUSH}{$Q-}{$R-}
      Inc(Seed);
      {$POP}
    end;
  finally
    PromptEmbeds.Free;
    Image.Free;
    Pipeline.Free;
    IPAdapter.Free;
    Img2Img.Free;
    Reporter.Free;
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
