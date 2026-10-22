program StableDiffusion;
(*
StableDiffusion: text-to-image with a diffusers Stable Diffusion 1.x / 2.x folder
through TStableDiffusionPipeline in neuralpretrained.pas.

  * no --model: runs the committed random pico folder tests/fixtures/tiny_sd15
    (16x16, random weights): this checks wiring and I/O only.
  * --model DIR: a diffusers StableDiffusionPipeline folder (model_index.json,
    text_encoder/, tokenizer/, unet/, vae/, scheduler/), e.g.
    runwayml/stable-diffusion-v1-5 with F32 *.safetensors weights (SD 2.x
    folders are coded and pico-tested only).

The scheduler config gives the betas, timestep spacing and prediction type,
and its _class_name the sampler (DDIM, PNDM/PLMS, Euler, Euler a,
DPM-Solver++ 2M or LCM; other classes run DDIM). --sampler overrides it.
--sampler lcm with an LCM-LoRA (--lora, e.g. latent-consistency/
lcm-lora-sdv1-5) samples in 2-8 steps (default 4, guidance 1.0). An
LCM-distilled folder (SimianLuo/LCM_Dreamshaper_v7) needs no LoRA: its UNet
embeds the guidance (default 8.5) and runs one pass per step.

Each component is loaded when its stage starts and freed when it ends, so
only one is in memory at a time. --repeat N > 1 makes N images from one
prompt encode and keeps the three components loaded. --int8 stores the UNet
and text encoder weights in int8. --stats prints a per-image stage table
(wall ms; host<->OpenCL MB with --gpu). --embedding FILE[:TOKEN] (repeatable)
loads a textual inversion concept, as diffusers load_textual_inversion.
--lora FILE[:SCALE] (repeatable) merges a LoRA into the UNet and text encoder
weights as they load, as diffusers load_lora_weights + fuse_lora.
--ip-adapter FILE --ip-image PNG [--ip-scale S] adds an image prompt through an
h94/IP-Adapter ip-adapter_sd15 file (its image_encoder/ folder beside it), as
diffusers load_ip_adapter + set_ip_adapter_scale. --init-image PNG
[--strength S] turns it into image-to-image (diffusers
StableDiffusionImg2ImgPipeline): the output takes the image's size.
--init-image on a 5-channel depth UNet folder such as
stabilityai/stable-diffusion-2-depth runs depth-to-image (diffusers
StableDiffusionDepth2ImgPipeline): the folder's depth_estimator/ (DPT-Hybrid)
estimates the depth map, or --depth-image PNG supplies it (read as its luma,
brighter = nearer).
--edit-image PNG [--image-guidance S] edits the image as the prompt instructs
with an InstructPix2Pix folder such as timbrooks/instruct-pix2pix (diffusers
StableDiffusionInstructPix2PixPipeline); the output takes the image's size.
--ip-adapter combines with either (TStableDiffusionMultiConditioner).
--upscale-image PNG [--noise-level N] upscales the image 4x with an x4
upscaler folder such as stabilityai/stable-diffusion-x4-upscaler (diffusers
StableDiffusionUpscalePipeline; default 75 steps, guidance 9.0).
--freeu s1,s2,b1,b2 (or the paper's presets sd15 / sd21) turns on FreeU, as
diffusers enable_freeu: no extra UNet pass.
--deepcache N [--deepcache-branch B] turns on DeepCache (Ma et al. 2023): the
whole UNet runs on every N-th step, the steps between run only its shallow
branch and reuse the deep branch output of the last whole run.
--pag-scale S [--pag-layers L1,L2] [--pag-adaptive-scale A] turns on
Perturbed-Attention Guidance (Ahn et al. 2024), as diffusers
StableDiffusionPAGPipeline: one more UNet pass per step with the self-attention
of the chosen layers (default mid) replaced by identity attention.

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
  SamplerOptionNames: array[0..5] of string =
    ('ddim', 'pndm', 'euler', 'euler-a', 'dpm++2m', 'lcm');
  SamplerOptionMethods: array[0..5] of TNNetSamplerMethod =
    (smDDIM, smPNDM, smEuler, smEulerAncestral, smDPMSolverPP2M, smLCM);
  // --freeu presets of the FreeU paper (s1, s2, b1, b2).
  FreeUPresetNames: array[0..1] of string = ('sd15', 'sd21');
  FreeUPresetFactors: array[0..1, 0..3] of TNeuralFloat =
    ((0.9, 0.2, 1.5, 1.6), (0.9, 0.2, 1.4, 1.6));

type
  TStableDiffusionOptions = record
    ModelFolder, Prompt, NegativePrompt, OutputFile: string;
    // --ip-adapter weights file and --ip-image; '' for none.
    IPAdapterFile, IPImageFile: string;
    IPScale: double;
    // --init-image ('' for text-to-image), its --strength and --depth-image.
    InitImageFile, DepthImageFile: string;
    Strength: double;
    // --edit-image ('' for none) and its --image-guidance (< 0: default).
    EditImageFile: string;
    ImageGuidanceScale: double;
    // --upscale-image ('' for none) and its --noise-level (< 0: default).
    UpscaleImageFile: string;
    NoiseLevel: integer;
    // --embedding values: FILE or FILE:TOKEN, in load order.
    Embeddings: array of string;
    // --lora files and their scales, in load order.
    LoRAFiles: array of string;
    LoRAScales: array of TNeuralFloat;
    // 'auto' (the scheduler config's) or one of SamplerOptionNames.
    SamplerName: string;
    // --karras-sigmas 1, --no-karras-sigmas 0, -1: the scheduler config's.
    KarrasSigmas: integer;
    // A side < 1: the checkpoint default (unet sample_size * VAE factor).
    // Steps -1 / GuidanceScale < 0: the sampler's default (LCM: 4 / 1.0;
    // an LCM-distilled UNet: 4 / 8.5).
    Steps, ImageWidth, ImageHeight, RepeatCount: integer;
    GuidanceScale: double;
    Seed: cardinal;
    UseOpenCL, UseInt8, UseStats: boolean;
    // --no-prompt-weighting clears UsePromptWeighting; --max-prompt-chunks;
    // --clip-skip.
    UsePromptWeighting: boolean;
    MaxPromptChunkCount, ClipSkip: integer;
    OpenCLPlatform, OpenCLDevice: integer;
    // --freeu s1, s2, b1, b2; UseFreeU false when not given.
    UseFreeU: boolean;
    FreeUFactors: array[0..3] of TNeuralFloat;
    // --deepcache N (1 = off); --deepcache-branch B (-1: the pipeline default).
    DeepCacheInterval, DeepCacheBranch: integer;
    // --pag-scale (<= 0: off), --pag-adaptive-scale and the --pag-layers
    // patterns (diffusers pag_applied_layers).
    PAGScale, PAGAdaptiveScale: double;
    PAGLayers: TStringArray;
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
    if (Phase = sdphDenoise) and (Stats.UNetCachedPassCount > 0) then
      Detail := Detail + Format(' (%d shallow only)',
        [Stats.UNetCachedPassCount]);
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
  WriteLn(Format('    step %d/%d  t=%.6g', [StepIndex + 1, StepCount, Timestep]));
end;

procedure PrintUsage();
begin
  WriteLn('Usage: StableDiffusion [--model DIR] [options]');
  WriteLn('  --model DIR       diffusers StableDiffusionPipeline folder ',
    '(default: the pico fixture)');
  WriteLn('  --prompt TEXT     prompt (default "a photo of an astronaut ',
    'riding a horse on mars")');
  WriteLn('  --negative TEXT   negative prompt (default empty)');
  WriteLn('                    Both take A1111 weights: (word:1.3), (word) = ',
    'x1.1, [word] = /1.1,');
  WriteLn('                    \( \) \[ \] for literal brackets; prompts ',
    'longer than 75 tokens');
  WriteLn('                    are encoded in chunks of 75 (lpw_stable_diffusion)');
  WriteLn('  --no-prompt-weighting  read brackets in the prompts as plain text');
  WriteLn('  --max-prompt-chunks N  keep at most N chunks of 75 prompt tokens ',
    '(default 3;');
  WriteLn('                    0: no limit); longer prompts are cut with a notice');
  WriteLn('  --clip-skip N     skip the last N text encoder layers (diffusers ',
    'clip_skip;');
  WriteLn('                    A1111 "Clip skip" N+1; default 0)');
  WriteLn('  --steps N         denoising steps (default 25; 4 with lcm)');
  WriteLn('  --guidance G      classifier-free guidance scale (default 7.5, ',
    '1.0 with lcm;');
  WriteLn('                    <= 1 turns it off); an LCM-distilled UNet ',
    'embeds it (default 8.5)');
  WriteLn('  --width N         image width (default: the checkpoint''s, ',
    '512 for SD 1.5, 768 for SD 2.1)');
  WriteLn('  --height N        image height (default: the checkpoint''s)');
  WriteLn('  --size N          sets both --width and --height');
  WriteLn('  --seed N          noise seed (default 0)');
  WriteLn('  --sampler NAME    auto (default: from the scheduler config), ddim,');
  WriteLn('                    pndm, euler, euler-a, dpm++2m or lcm (Latent ',
    'Consistency;');
  WriteLn('                    an LCM-LoRA via --lora or an LCM-distilled ',
    'UNet, 2-8 steps)');
  WriteLn('  --karras-sigmas   Karras sigmas for euler, euler-a and dpm++2m ',
    '(diffusers');
  WriteLn('                    use_karras_sigmas); --no-karras-sigmas turns ',
    'them off');
  WriteLn('                    (default: the scheduler config''s)');
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
  WriteLn('  --lora FILE[:SCALE] merge a LoRA .safetensors (diffusers/PEFT or ',
    'kohya/A1111');
  WriteLn('                    keys; UNet and text encoder) at SCALE ',
    '(default 1.0); repeatable');
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
  WriteLn('                    runs the last int(steps * S) steps (lcm: all ',
    'steps over');
  WriteLn('                    the first int(50 * S) LCM timesteps)');
  WriteLn('  --depth-image PNG depth-to-image: the depth map of --init-image ',
    '(luma, brighter =');
  WriteLn('                    nearer; any size) for a 5-channel depth UNet ',
    'folder, e.g.');
  WriteLn('                    stabilityai/stable-diffusion-2-depth; without ',
    'it the folder''s');
  WriteLn('                    depth_estimator/ estimates the depth');
  WriteLn('  --edit-image PNG  InstructPix2Pix: edit this image as the prompt ',
    'says (needs an');
  WriteLn('                    8-channel UNet folder, e.g. ',
    'timbrooks/instruct-pix2pix); the output');
  WriteLn('                    takes its size (rounded down to the VAE factor)');
  WriteLn('  --image-guidance S how closely --edit-image is kept (default 1.5; ',
    '< 1 turns all');
  WriteLn('                    guidance off)');
  WriteLn('  --upscale-image PNG x4 upscaler: upscale this image (needs a ',
    '7-channel UNet folder,');
  WriteLn('                    e.g. stabilityai/stable-diffusion-x4-upscaler); ',
    'the output is');
  WriteLn('                    VAE factor x its size (sides rounded down to the ',
    'factor);');
  WriteLn('                    default 75 steps, guidance 9.0');
  WriteLn('  --noise-level N   noise added to --upscale-image, 0..max_noise_level ',
    '(350;');
  WriteLn('                    default 20); more noise lets the UNet invent ',
    'more detail');
  WriteLn('  --freeu S1,S2,B1,B2 FreeU (diffusers enable_freeu): skip scales ',
    'S1, S2 and backbone');
  WriteLn('                    scales B1, B2 of the first two up blocks; ',
    'presets sd15');
  WriteLn('                    (0.9,0.2,1.5,1.6) and sd21 (0.9,0.2,1.4,1.6)');
  WriteLn('  --deepcache N     DeepCache: the whole UNet every N-th step, only ',
    'its shallow');
  WriteLn('                    branch in between (default 1 = off; ',
    'DeepCache''s SD example uses 3)');
  WriteLn('  --deepcache-branch B shallow branch size, 0..layers_per_block: ',
    'B resnets of');
  WriteLn('                    down block 0 and B + 1 of the last up block run ',
    '(default');
  WriteLn('                    layers_per_block: both blocks whole; 0 is the ',
    'fastest)');
  WriteLn('  --pag-scale S     Perturbed-Attention Guidance scale (default 0 = ',
    'off; diffusers');
  WriteLn('                    uses 3.0): one more UNet pass per step');
  WriteLn('  --pag-layers L1,L2 the self-attentions PAG perturbs, diffusers ',
    'pag_applied_layers');
  WriteLn('                    patterns (regular expressions over names such ',
    'as');
  WriteLn('                    mid_block.attentions.0.transformer_blocks.0.',
    'attn1; default mid)');
  WriteLn('  --pag-adaptive-scale A lowers the PAG scale by A per timestep ',
    'below 1000');
  WriteLn('                    (diffusers pag_adaptive_scale, default 0)');
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

// --freeu value: a preset name or four comma-separated factors.
procedure ParseFreeUSpec(const Spec: string; out Factors: array of TNeuralFloat);
var
  Parts: TStringArray;
  PresetPos, FactorPos: integer;
begin
  for PresetPos := 0 to High(FreeUPresetNames) do
    if LowerCase(Spec) = FreeUPresetNames[PresetPos] then
    begin
      for FactorPos := 0 to 3 do
        Factors[FactorPos] := FreeUPresetFactors[PresetPos, FactorPos];
      exit;
    end;
  Parts := Spec.Split([',']);
  if Length(Parts) <> 4 then
    raise Exception.Create('--freeu wants s1,s2,b1,b2, sd15 or sd21, not "' +
      Spec + '"');
  for FactorPos := 0 to 3 do
    Factors[FactorPos] := StrToFloat(Trim(Parts[FactorPos]),
      DefaultFormatSettings);
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
  Opt.DepthImageFile := '';
  Opt.Strength := -1;
  Opt.EditImageFile := '';
  Opt.ImageGuidanceScale := -1;
  Opt.UpscaleImageFile := '';
  Opt.NoiseLevel := -1;
  Opt.SamplerName := 'auto';
  Opt.KarrasSigmas := -1;
  Opt.Steps := -1;
  Opt.ImageWidth := 0;
  Opt.ImageHeight := 0;
  Opt.GuidanceScale := -1;
  Opt.Seed := 0;
  Opt.RepeatCount := 1;
  Opt.UseOpenCL := false;
  Opt.UseInt8 := false;
  Opt.UseStats := false;
  Opt.UsePromptWeighting := true;
  Opt.MaxPromptChunkCount := 3;
  Opt.ClipSkip := 0;
  Opt.OpenCLPlatform := 0;
  Opt.OpenCLDevice := 0;
  Opt.UseFreeU := false;
  Opt.DeepCacheInterval := 1;
  Opt.DeepCacheBranch := -1;
  Opt.PAGScale := 0;
  Opt.PAGAdaptiveScale := 0;
  Opt.PAGLayers := nil;
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
    else if Arg = '--karras-sigmas' then Opt.KarrasSigmas := 1
    else if Arg = '--no-karras-sigmas' then Opt.KarrasSigmas := 0
    else if Arg = '--repeat' then Opt.RepeatCount := StrToInt(NextArg())
    else if Arg = '--embedding' then
    begin
      SetLength(Opt.Embeddings, Length(Opt.Embeddings) + 1);
      Opt.Embeddings[High(Opt.Embeddings)] := NextArg();
    end
    else if Arg = '--lora' then
    begin
      SetLength(Opt.LoRAFiles, Length(Opt.LoRAFiles) + 1);
      SetLength(Opt.LoRAScales, Length(Opt.LoRAScales) + 1);
      SplitLoRASpec(NextArg(), Opt.LoRAFiles[High(Opt.LoRAFiles)],
        Opt.LoRAScales[High(Opt.LoRAScales)]);
      if not FileExists(Opt.LoRAFiles[High(Opt.LoRAFiles)]) then
        raise Exception.Create('--lora: file "' +
          Opt.LoRAFiles[High(Opt.LoRAFiles)] + '" not found');
    end
    else if Arg = '--ip-adapter' then Opt.IPAdapterFile := NextArg()
    else if Arg = '--ip-image' then Opt.IPImageFile := NextArg()
    else if Arg = '--ip-scale' then
      Opt.IPScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--init-image' then Opt.InitImageFile := NextArg()
    else if Arg = '--depth-image' then Opt.DepthImageFile := NextArg()
    else if Arg = '--strength' then
      Opt.Strength := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--edit-image' then Opt.EditImageFile := NextArg()
    else if Arg = '--image-guidance' then
      Opt.ImageGuidanceScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--upscale-image' then Opt.UpscaleImageFile := NextArg()
    else if Arg = '--noise-level' then Opt.NoiseLevel := StrToInt(NextArg())
    else if Arg = '--freeu' then
    begin
      ParseFreeUSpec(NextArg(), Opt.FreeUFactors);
      Opt.UseFreeU := true;
    end
    else if Arg = '--deepcache' then Opt.DeepCacheInterval := StrToInt(NextArg())
    else if Arg = '--deepcache-branch' then
      Opt.DeepCacheBranch := StrToInt(NextArg())
    else if Arg = '--pag-scale' then
      Opt.PAGScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--pag-adaptive-scale' then
      Opt.PAGAdaptiveScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--pag-layers' then
      Opt.PAGLayers := NextArg().Split([','])
    else if Arg = '--no-prompt-weighting' then Opt.UsePromptWeighting := false
    else if Arg = '--max-prompt-chunks' then
      Opt.MaxPromptChunkCount := StrToInt(NextArg())
    else if Arg = '--clip-skip' then Opt.ClipSkip := StrToInt(NextArg())
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
  if (Opt.Steps < 1) and (Opt.Steps <> -1) then
    raise Exception.Create('--steps must be >= 1');
  if Opt.RepeatCount < 1 then raise Exception.Create('--repeat must be >= 1');
  if Opt.DeepCacheInterval < 1 then
    raise Exception.Create('--deepcache must be >= 1');
  if (Opt.DeepCacheBranch >= 0) and (Opt.DeepCacheInterval < 2) then
    raise Exception.Create('--deepcache-branch needs --deepcache 2 or more');
  if ((Length(Opt.PAGLayers) > 0) or (Opt.PAGAdaptiveScale <> 0)) and
    (Opt.PAGScale <= 0) then
    raise Exception.Create('--pag-layers and --pag-adaptive-scale need ' +
      '--pag-scale > 0');
  if (Opt.PAGScale > 0) and (Opt.EditImageFile <> '') then
    raise Exception.Create('--pag-scale cannot be combined with --edit-image');
  if (Opt.IPAdapterFile = '') <> (Opt.IPImageFile = '') then
    raise Exception.Create('--ip-adapter and --ip-image go together');
  if (Opt.Strength >= 0) and (Opt.InitImageFile = '') then
    raise Exception.Create('--strength needs --init-image');
  if (Opt.DepthImageFile <> '') and (Opt.InitImageFile = '') then
    raise Exception.Create('--depth-image needs --init-image');
  if (Opt.ImageGuidanceScale >= 0) and (Opt.EditImageFile = '') then
    raise Exception.Create('--image-guidance needs --edit-image');
  if (Opt.EditImageFile <> '') and (Opt.InitImageFile <> '') then
    raise Exception.Create('--edit-image cannot be combined with ' +
      '--init-image');
  if (Opt.NoiseLevel >= 0) and (Opt.UpscaleImageFile = '') then
    raise Exception.Create('--noise-level needs --upscale-image');
  if (Opt.UpscaleImageFile <> '') and ((Opt.InitImageFile <> '') or
    (Opt.EditImageFile <> '')) then
    raise Exception.Create('--upscale-image cannot be combined with ' +
      '--init-image or --edit-image');
  if (Opt.UpscaleImageFile <> '') and ((Opt.ImageWidth >= 1) or
    (Opt.ImageHeight >= 1)) then
    raise Exception.Create('--upscale-image sets the output size; drop ' +
      '--width / --height / --size');
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
  Multi: TStableDiffusionMultiConditioner;
  Img2Img: TStableDiffusionImg2Img;
  Depth2Img: TStableDiffusionDepth2Img;
  InstructPix2Pix: TStableDiffusionInstructPix2Pix;
  Upscaler: TStableDiffusionUpscaler;
  ImageConditioner: TStableDiffusionImageConditioner;
  Image, PromptEmbeds: TNNetVolume;
  SourceImageFile: string;
  ComputeText, ImageFile, EmbeddingFile, EmbeddingToken: string;
  ImageWidth, ImageHeight, ImagePos, EmbeddingPos, LoRAPos: integer;
  PerturbedGuidance: TNNetPerturbedGuidance;
  SelfAttentionNames: TStringArray;
  PAGLayerIndices: TNeuralIntegerArray;
  PAGLayerPos: integer;
  StepCount, Img2ImgStepCount: integer;
  IsDepthUNet, IsUpscalerUNet: boolean;
  GuidanceScale: double;
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
  Multi := nil;
  Img2Img := nil;
  Depth2Img := nil;
  InstructPix2Pix := nil;
  Upscaler := nil;
  ImageConditioner := nil;
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
    // A depth UNet takes latents | depth; without --depth-image the folder's
    // depth_estimator/ computes the depth from the init image.
    IsDepthUNet := Pipeline.UNetConfig.InChannels =
      Pipeline.UNetConfig.OutChannels + 1;
    if IsDepthUNet and (Opt.InitImageFile = '') then
      raise Exception.Create('this is a depth-to-image UNet: give ' +
        '--init-image (and optionally --depth-image)');
    // An x4 upscaler UNet takes latents | the noised low-res RGB image.
    IsUpscalerUNet := Pipeline.UNetConfig.InChannels =
      Pipeline.UNetConfig.OutChannels + 3;
    if IsUpscalerUNet and (Opt.UpscaleImageFile = '') then
      raise Exception.Create('this is an x4 upscaler UNet: give ' +
        '--upscale-image');
    if IsDepthUNet or (Opt.DepthImageFile <> '') then
    begin
      Depth2Img := TStableDiffusionDepth2Img.Create();
      if Opt.DepthImageFile <> '' then
      begin
        if not LoadImageFromFileIntoVolume(Opt.DepthImageFile, Image,
          {pWithAlpha=}false, {pFullPrecision=}true) then
          raise Exception.Create('could not load ' + Opt.DepthImageFile);
        Depth2Img.SetDepthMap(Image);
      end;
      Img2Img := Depth2Img;
    end
    else if Opt.InitImageFile <> '' then
      Img2Img := TStableDiffusionImg2Img.Create();
    if Assigned(Img2Img) then
    begin
      if Opt.Strength >= 0 then Img2Img.Strength := Opt.Strength;
      ImageConditioner := Img2Img;
      SourceImageFile := Opt.InitImageFile;
    end
    else if Opt.EditImageFile <> '' then
    begin
      InstructPix2Pix := TStableDiffusionInstructPix2Pix.Create();
      if Opt.ImageGuidanceScale >= 0 then
        InstructPix2Pix.ImageGuidanceScale := Opt.ImageGuidanceScale;
      ImageConditioner := InstructPix2Pix;
      SourceImageFile := Opt.EditImageFile;
    end
    else if Opt.UpscaleImageFile <> '' then
    begin
      Upscaler := TStableDiffusionUpscaler.Create();
      if Opt.NoiseLevel >= 0 then Upscaler.NoiseLevel := Opt.NoiseLevel;
      ImageConditioner := Upscaler;
      SourceImageFile := Opt.UpscaleImageFile;
    end;
    if Assigned(ImageConditioner) then
    begin
      if not LoadImageFromFileIntoVolume(SourceImageFile, Image) then
        raise Exception.Create('could not load ' + SourceImageFile);
      ImageConditioner.SetImage(Image);
      ImageConditioner.ImageSizeFor(Pipeline.VaeScaleFactor, ImageWidth,
        ImageHeight);
      Pipeline.Conditioner := ImageConditioner;
    end;
    if Opt.ImageWidth >= 1 then ImageWidth := Opt.ImageWidth;
    if Opt.ImageHeight >= 1 then ImageHeight := Opt.ImageHeight;
    if Opt.SamplerName <> 'auto' then
      Pipeline.Sampler := SamplerOptionMethods[
        SamplerOptionIndex(Opt.SamplerName)];
    if Opt.KarrasSigmas >= 0 then Pipeline.KarrasSigmas := Opt.KarrasSigmas = 1;
    StepCount := Opt.Steps;
    GuidanceScale := Opt.GuidanceScale;
    if Pipeline.Sampler = smLCM then
    begin
      if StepCount < 1 then StepCount := 4;
      // diffusers LatentConsistencyModelPipeline's default for distilled UNets.
      if (GuidanceScale < 0) and (Pipeline.UNetConfig.TimeCondProjDim > 0) then
        GuidanceScale := 8.5;
      if GuidanceScale < 0 then GuidanceScale := 1.0;
    end
    else if Assigned(Upscaler) then
    begin
      // diffusers StableDiffusionUpscalePipeline's defaults.
      if StepCount < 1 then StepCount := 75;
      if GuidanceScale < 0 then GuidanceScale := 9.0;
    end
    else
    begin
      if StepCount < 1 then StepCount := 25;
      if GuidanceScale < 0 then GuidanceScale := 7.5;
    end;
    Pipeline.GuidanceScale := GuidanceScale;
    Pipeline.NegativePrompt := Opt.NegativePrompt;
    Pipeline.PromptWeighting := Opt.UsePromptWeighting;
    Pipeline.MaxPromptChunkCount := Opt.MaxPromptChunkCount;
    Pipeline.ClipSkip := Opt.ClipSkip;
    Pipeline.UNetInt8 := Opt.UseInt8;
    Pipeline.TextEncoderInt8 := Opt.UseInt8;
    if Opt.UseFreeU then
      Pipeline.EnableFreeU(Opt.FreeUFactors[0], Opt.FreeUFactors[1],
        Opt.FreeUFactors[2], Opt.FreeUFactors[3]);
    Pipeline.DeepCacheInterval := Opt.DeepCacheInterval;
    if Opt.DeepCacheBranch >= 0 then
      Pipeline.DeepCacheBranch := Opt.DeepCacheBranch;
    if Opt.PAGScale > 0 then
    begin
      PerturbedGuidance := TNNetPerturbedGuidance.Create(1, Opt.PAGScale);
      Pipeline.PerturbedGuidance := PerturbedGuidance;
      PerturbedGuidance.AdaptiveScale := Opt.PAGAdaptiveScale;
      if Length(Opt.PAGLayers) > 0 then
        PAGLayerIndices := SDUNetPAGLayerIndices(Pipeline.UNetConfig,
          Opt.PAGLayers)
      else PAGLayerIndices := SDUNetPAGLayerIndices(Pipeline.UNetConfig,
        ['mid']);
      PerturbedGuidance.SetPerturbedLayers(PAGLayerIndices);
    end;
    {$IFDEF OpenCL}
    if Opt.UseStats then OpenCLTransferCounting := true;
    {$ENDIF}
    Pipeline.OnPhase := @Reporter.OnPhase;
    Pipeline.OnStep := @Reporter.OnStep;
    WriteLn('Model:    ', Opt.ModelFolder);
    WriteLn('Prompt:   "', Opt.Prompt, '"');
    WriteLn(Format('Image:    %dx%d, %d steps, guidance %.2f, seed %d, %s',
      [ImageWidth, ImageHeight, StepCount, GuidanceScale, Opt.Seed,
       ComputeText]));
    if Pipeline.UNetConfig.TimeCondProjDim > 0 then
      WriteLn('UNet:     LCM-distilled, guidance embedded (one pass per step)');
    if Pipeline.Sampler = smLCM then
      WriteLn('Sampler:  lcm with the ', Pipeline.SchedulerClassName,
        ' config''s betas and the LCM origin timesteps')
    else
      WriteLn('Sampler:  ', SamplerText(Pipeline.Sampler), ' with the ',
        Pipeline.SchedulerClassName, ' config''s betas and timestep spacing');
    if Pipeline.KarrasSigmas and
      (Pipeline.Sampler in [smEuler, smEulerAncestral, smDPMSolverPP2M]) then
      WriteLn('Sigmas:   Karras (rho 7)');
    if (Opt.SamplerName = 'auto') and (Pipeline.SamplerNotice <> '') then
      WriteLn('Notice:   ', Pipeline.SamplerNotice);
    if Opt.UseInt8 then WriteLn('Weights:  UNet int8, text encoder int8')
    else WriteLn('Weights:  FP32');
    if Pipeline.UNetConfig.UseFreeU then
      WriteLn(Format('FreeU:    s1 %.2f, s2 %.2f, b1 %.2f, b2 %.2f',
        [Opt.FreeUFactors[0], Opt.FreeUFactors[1], Opt.FreeUFactors[2],
         Opt.FreeUFactors[3]]));
    if Pipeline.DeepCacheInterval > 1 then
      WriteLn(Format('DeepCache: interval %d, branch %d',
        [Pipeline.DeepCacheInterval, Pipeline.DeepCacheBranch]));
    if Opt.PAGScale > 0 then
    begin
      WriteLn(Format('PAG:      scale %.2f, adaptive scale %g, %d ' +
        'self-attention(s):', [Opt.PAGScale, Opt.PAGAdaptiveScale,
        Length(PAGLayerIndices)]));
      SelfAttentionNames := SDUNetSelfAttentionNames(Pipeline.UNetConfig);
      for PAGLayerPos := 0 to High(PAGLayerIndices) do
        WriteLn('          ', SelfAttentionNames[PAGLayerIndices[PAGLayerPos]]);
    end;
    for EmbeddingPos := 0 to High(Opt.Embeddings) do
    begin
      SplitEmbeddingSpec(Opt.Embeddings[EmbeddingPos], EmbeddingFile,
        EmbeddingToken);
      EmbeddingToken := Pipeline.LoadTextualInversion(EmbeddingFile,
        EmbeddingToken);
      WriteLn('Embedding: ', EmbeddingToken, ' from ', EmbeddingFile);
    end;
    for LoRAPos := 0 to High(Opt.LoRAFiles) do
    begin
      Pipeline.AddLoRA(Opt.LoRAFiles[LoRAPos], Opt.LoRAScales[LoRAPos]);
      WriteLn(Format('LoRA:     %s x %.2f', [Opt.LoRAFiles[LoRAPos],
        Opt.LoRAScales[LoRAPos]]));
    end;
    if Assigned(Img2Img) then
    begin
      // LCM runs every step over a strength-scaled timestep grid.
      if Pipeline.Sampler = smLCM then Img2ImgStepCount := StepCount
      else Img2ImgStepCount := StepCount -
        TNNetDiffusionScheduler.Img2ImgStartStep(StepCount, Img2Img.Strength);
      WriteLn(Format('Init image: %s (%dx%d), strength %.2f, %d of %d steps',
        [Opt.InitImageFile, Image.SizeX, Image.SizeY, Img2Img.Strength,
         Img2ImgStepCount, StepCount]));
      if Assigned(Depth2Img) and (Opt.DepthImageFile <> '') then
        WriteLn('Depth image: ', Opt.DepthImageFile, ' (depth-to-image)')
      else if Assigned(Depth2Img) then
        WriteLn('Depth: estimated by depth_estimator/ (depth-to-image)');
    end;
    if Assigned(InstructPix2Pix) then
      WriteLn(Format('Edit image: %s (%dx%d), image guidance %.2f',
        [Opt.EditImageFile, Image.SizeX, Image.SizeY,
         InstructPix2Pix.ImageGuidanceScale]));
    if Assigned(Upscaler) then
      WriteLn(Format('Upscale image: %s (%dx%d), noise level %d',
        [Opt.UpscaleImageFile, Image.SizeX, Image.SizeY,
         Upscaler.NoiseLevel]));
    if Opt.IPAdapterFile <> '' then
    begin
      IPAdapter := TStableDiffusionIPAdapter.Create(Opt.IPAdapterFile);
      if not LoadImageFromFileIntoVolume(Opt.IPImageFile, Image) then
        raise Exception.Create('could not load ' + Opt.IPImageFile);
      IPAdapter.SetImage(Image);
      IPAdapter.Scale := Opt.IPScale;
      if Assigned(ImageConditioner) then
      begin
        Multi := TStableDiffusionMultiConditioner.Create();
        Multi.Add(ImageConditioner);
        Multi.Add(IPAdapter);
        Pipeline.Conditioner := Multi;
      end
      else Pipeline.Conditioner := IPAdapter;
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
      // Euler a, LCM and the img2img VAE sample draw their noise from the
      // global RNG; the initial latents use Seed itself, so this stream starts
      // elsewhere.
      RandSeed := Seed xor $9E3779B9;
      StartTime := Now();
      if Opt.RepeatCount > 1 then
        Pipeline.GenerateFromEmbeds(PromptEmbeds, ImageWidth, ImageHeight,
          StepCount, Seed, Image)
      else
        Pipeline.Generate(Opt.Prompt, ImageWidth, ImageHeight, StepCount,
          Seed, Image);
      // The prompt and the negative prompt are encoded by the first image.
      if (ImagePos = 0) and (Pipeline.PromptNotice <> '') then
        WriteLn('Notice:   ', Pipeline.PromptNotice);
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
    Multi.Free;
    IPAdapter.Free;
    ImageConditioner.Free;
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
