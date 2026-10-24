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
DPM-Solver++ 2M, its SDE variant or LCM; other classes run DDIM). --sampler
overrides it.
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
--change-map PNG with --init-image is Differential Diffusion (Levin & Fried
2024): each pixel's gray level sets how much it may change (white = most).
--init-image on a 5-channel depth UNet folder such as
stabilityai/stable-diffusion-2-depth runs depth-to-image (diffusers
StableDiffusionDepth2ImgPipeline): the folder's depth_estimator/ (DPT-Hybrid)
estimates the depth map, or --depth-image PNG supplies it (read as its luma,
brighter = nearer).
--edit-image PNG [--image-guidance S] edits the image as the prompt instructs
with an InstructPix2Pix folder such as timbrooks/instruct-pix2pix (diffusers
StableDiffusionInstructPix2PixPipeline); the output takes the image's size.
--ip-adapter combines with either (TStableDiffusionMultiConditioner).
--gligen FILE --box "PHRASE:X0,Y0,X1,Y1" (repeatable, at most 30) places each
phrase inside its box (normalized xyxy, 0..1, origin top-left) through a GLIGEN
text-box UNet file such as masterful/gligen-1-4-generation-text-box, as diffusers
StableDiffusionGLIGENPipeline; --gligen-beta B (default 0.3) grounds the first
int(B * steps) steps. --gligen-inpaint-image PNG with a GLIGEN inpainting folder
(masterful/gligen-1-4-inpainting-text-box) regenerates only the boxes of the
image (diffusers gligen_inpaint_image).
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
--sag-scale S turns on Self-Attention Guidance (Hong et al. 2023), as diffusers
StableDiffusionSAGPipeline: one more UNet pass per step on latents blurred where
the mid-block self-attention concentrates.
--invert-image PNG [--invert-steps N] [--invert-guidance G] runs DDIM inversion
(diffusers DDIMInverseScheduler): the image's VAE latents are walked to noise
with the prompt, then sampled back with DDIM and the prompt; guidance defaults
to 1, where the round trip reconstructs the image best. --null-text adds
null-text inversion (Mokady et al. 2022): per step, the unconditional embeds are
fitted so sampling back at the CFG guidance (default 7.5) retraces the inversion.
The fit backpropagates through the sampling UNet (int8 with --int8) without
weight gradients; on --gpu it builds a trainable FP32 UNet on the CPU.
--relight PNG --iclight FILE relights a cut-out subject with IC-Light (Zhang
et al. 2025, lllyasviel/IC-Light gradio_demo.py): the iclight_sd15_fc offset
is added to the UNet weights, the prompt describes the light, --light
left|right|top|bottom starts from a light gradient, --relight-background PNG
(with iclight_sd15_fbc) relights against a background, and a highres pass
upscales by 1.5. The subject's alpha (RGBA), or --birefnet FILE (BiRefNet /
RMBG-2.0; --birefnet-config FILE when its config.json is elsewhere), mattes
it on gray.
--panorama WxH runs MultiDiffusion (Bar-Tal et al. 2023, diffusers
StableDiffusionPanoramaPipeline): each step runs the UNet, with its own CFG,
on overlapping --width x --height windows (default the checkpoint size) of a
WxH canvas every --panorama-stride pixels (default 64, diffusers' 8 latent
cells), averages the predictions where windows overlap and steps the whole
canvas once; --panorama-wrap wraps the windows in X for a 360-degree panorama.
--vae-tile N decodes and encodes in N-pixel tiles (diffusers
enable_vae_tiling).
--tile-upscale N --init-image PNG upscales the image N times: PIL bicubic to N
times its size, then image-to-image at a low --strength (default 0.35) over
--panorama windows every half window, encoded and decoded with --vae-tile; --controlnet DIR
(a ControlNet folder such as lllyasviel/control_v11f1e_sd15_tile) takes the
upscaled image as its hint, each window its own crop of it. --controlnet DIR
--control-image PNG adds a ControlNet to any other run.
--edit-prompt TEXT runs Prompt-to-Prompt (Hertz et al. 2022, the
google/prompt-to-prompt controllers): the prompt and the edit prompt sample
together from one latent with DDIM, and the edit takes the prompt's
cross-attention maps (--edit-mode swap|refine|reweight, --cross-replace F) and
self-attention maps (--self-replace F); both images are written. With
--invert-image both start from the image's inverted latents.

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
  {$IFDEF OpenCL}neuralopencl,{$ENDIF}
  neuralnetwork, neuralvolume, neuraldatasets, neuraldiffusion,
  neuralimageresize, neuralpretrained;

const
  PicoModelFolder = '../../tests/fixtures/tiny_sd15';
  // --sampler names of the methods TStableDiffusionPipeline maps to.
  SamplerOptionNames: array[0..6] of string =
    ('ddim', 'pndm', 'euler', 'euler-a', 'dpm++2m', 'dpm++2m-sde', 'lcm');
  SamplerOptionMethods: array[0..6] of TNNetSamplerMethod =
    (smDDIM, smPNDM, smEuler, smEulerAncestral, smDPMSolverPP2M,
     smDPMSolverPP2M, smLCM);
  // The DPM-Solver++ algorithm of each name (read for dpm++2m* only).
  SamplerOptionDPMAlgorithms: array[0..6] of TNNetDPMSolverAlgorithm =
    (dpaDPMSolverPP, dpaDPMSolverPP, dpaDPMSolverPP, dpaDPMSolverPP,
     dpaDPMSolverPP, dpaSDEDPMSolverPP, dpaDPMSolverPP);
  // --freeu presets of the FreeU paper (s1, s2, b1, b2).
  FreeUPresetNames: array[0..1] of string = ('sd15', 'sd21');
  FreeUPresetFactors: array[0..1, 0..3] of TNeuralFloat =
    ((0.9, 0.2, 1.5, 1.6), (0.9, 0.2, 1.4, 1.6));
  // --light names of TSDLightDirection.
  LightOptionNames: array[TSDLightDirection] of string =
    ('none', 'left', 'right', 'top', 'bottom');
  // IC-Light gradio_demo.py's added and negative prompts.
  ICLightAddedPrompt = 'best quality';
  ICLightNegativePrompt = 'lowres, bad anatomy, bad hands, cropped, ' +
    'worst quality';

type
  TStableDiffusionOptions = record
    ModelFolder, Prompt, NegativePrompt, OutputFile: string;
    // --ip-adapter weights file and --ip-image; '' for none.
    IPAdapterFile, IPImageFile: string;
    IPScale: double;
    // --gligen UNet file ('' for none), its --box specs and --gligen-beta.
    GLIGENFile: string;
    GLIGENBoxes: array of string;
    GLIGENBeta: double;
    // --gligen-inpaint-image ('' for none).
    GLIGENInpaintImageFile: string;
    // --init-image ('' for text-to-image), its --strength and --depth-image.
    InitImageFile, DepthImageFile: string;
    Strength: double;
    // --change-map ('' for none) and --invert-change-map.
    ChangeMapFile: string;
    InvertChangeMap: boolean;
    // --edit-image ('' for none) and its --image-guidance (< 0: default).
    EditImageFile: string;
    ImageGuidanceScale: double;
    // --invert-image ('' for none), --invert-steps (< 1: --steps) and
    // --invert-guidance (< 0: 1.0).
    InvertImageFile: string;
    InvertSteps: integer;
    InvertGuidanceScale: double;
    // --null-text and its --null-text-inner Adam steps per sampling step.
    NullText: boolean;
    NullTextInnerSteps: integer;
    // --edit-prompt ('' for none), --edit-mode, --cross-replace,
    // --self-replace and the --reweight words and scales.
    EditPrompt, EditMode: string;
    CrossReplaceFraction, SelfReplaceFraction: double;
    ReweightWords: TStringArray;
    ReweightScales: TNeuralFloatDynArr;
    // --upscale-image ('' for none) and its --noise-level (< 0: default).
    UpscaleImageFile: string;
    NoiseLevel: integer;
    // --relight subject ('' for none), --iclight offset, --relight-background,
    // --birefnet (and --birefnet-config, '' for the config.json beside it),
    // --light, --relight-added-prompt and the highres pass
    // (--highres-scale, --highres-strength; < 0: default).
    RelightImageFile, ICLightFile, RelightBackgroundFile, BiRefNetFile: string;
    BiRefNetConfigFile: string;
    LightDirection: TSDLightDirection;
    RelightAddedPrompt: string;
    HighResScale, HighResStrength: double;
    // True when --negative was given.
    HasNegativePrompt: boolean;
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
    // --sag-scale (<= 0: off).
    SAGScale: double;
    // --panorama canvas (0: off), --panorama-stride (0: the pipeline's
    // default) and --panorama-wrap; the windows are --width x --height.
    PanoramaWidth, PanoramaHeight, PanoramaStride: integer;
    PanoramaWrap: boolean;
    // --vae-tile pixels (0: one decode).
    VaeTileSize: integer;
    // --tile-upscale factor (0: off).
    TileUpscale: double;
    // --controlnet folder ('' for none), --control-image ('' under
    // --tile-upscale: the upscaled image) and --controlnet-scale.
    ControlNetFolder, ControlImageFile: string;
    ControlNetScale: double;
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
    if (Phase = sdphDenoise) and (Stats.PanoramaWindowCount > 0) then
      Detail := Detail + Format(', %d panorama windows',
        [Stats.PanoramaWindowCount]);
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
  WriteLn('                    are encoded in chunks of 75 (lpw_stable_diffusion);');
  WriteLn('                    a standalone BREAK starts a new chunk (A1111)');
  WriteLn('  --no-prompt-weighting  read brackets and BREAK in the prompts as ',
    'plain text');
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
  WriteLn('                    pndm, euler, euler-a, dpm++2m, dpm++2m-sde ',
    '(diffusers');
  WriteLn('                    sde-dpmsolver++) or lcm (Latent Consistency;');
  WriteLn('                    an LCM-LoRA via --lora or an LCM-distilled ',
    'UNet, 2-8 steps)');
  WriteLn('  --karras-sigmas   Karras sigmas for euler, euler-a and dpm++2m* ',
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
  WriteLn('  --gligen FILE     GLIGEN text-box UNet .safetensors (fuser.* and ',
    'position_net.*);');
  WriteLn('                    needs --box');
  WriteLn('  --box "PHRASE:X0,Y0,X1,Y1" a --gligen phrase and its normalized ',
    'box (0..1,');
  WriteLn('                    origin top-left); repeatable, at most 30');
  WriteLn('  --gligen-beta B   ground the first int(B * steps) steps (default ',
    '0.3)');
  WriteLn('  --gligen-inpaint-image PNG  regenerate only the --box regions of ',
    'this image');
  WriteLn('                    (a GLIGEN inpainting folder, 9-channel UNet); ',
    'the image is');
  WriteLn('                    center-cropped and resized to the output size');
  WriteLn('  --init-image PNG  image-to-image from this image; the output ',
    'takes its size');
  WriteLn('                    (rounded down to the VAE factor) unless ',
    '--width/--height');
  WriteLn('  --strength S      how much of --init-image to repaint, (0, 1] ',
    '(default 0.8;');
  WriteLn('                    --relight --light: 0.9);');
  WriteLn('                    runs the last int(steps * S) steps (lcm: all ',
    'steps over');
  WriteLn('                    the first int(50 * S) LCM timesteps)');
  WriteLn('  --change-map PNG  Differential Diffusion: a gray map of how much ',
    'to change each');
  WriteLn('                    pixel of --init-image (white = change most, ',
    'black = keep;');
  WriteLn('                    any size), instead of one global strength; ',
    '--strength (default');
  WriteLn('                    1) still truncates the schedule');
  WriteLn('  --invert-change-map  read --change-map as white = keep (the ',
    'reference''s maps)');
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
  WriteLn('  --invert-image PNG DDIM inversion: walk this image''s latents to ',
    'noise with the');
  WriteLn('                    prompt, then sample them back with DDIM; the ',
    'output takes its');
  WriteLn('                    size (rounded down to the VAE factor); guidance ',
    'default 1.0');
  WriteLn('  --invert-steps N  inversion steps (default: --steps)');
  WriteLn('  --invert-guidance G CFG weight of the inversion (default 1.0)');
  WriteLn('  --null-text       with --invert-image: null-text inversion, per-step ',
    'unconditional');
  WriteLn('                    embeds fitted so DDIM at --guidance (default ',
    '7.5) retraces the');
  WriteLn('                    inversion; also for --edit-prompt');
  WriteLn('  --null-text-inner N Adam steps per sampling step (default 10)');
  WriteLn('  --edit-prompt TEXT Prompt-to-Prompt: also sample this prompt from ',
    'the same latent');
  WriteLn('                    (DDIM), its attention steered by --prompt''s; ',
    'writes --output');
  WriteLn('                    and its _edit twin; with --invert-image both ',
    'start from the');
  WriteLn('                    inverted image');
  WriteLn('  --edit-mode M     swap (word swap, equal word counts), refine ',
    '(added or changed');
  WriteLn('                    words keep their own maps; default) or ',
    'reweight (refine plus');
  WriteLn('                    --reweight)');
  WriteLn('  --cross-replace F the edit takes --prompt''s cross-attention for ',
    'the first');
  WriteLn('                    int(F * (steps + 1)) steps (default 0.8)');
  WriteLn('  --self-replace F  ... and its self-attention (maps of <= 16x16 ',
    'tokens) for the');
  WriteLn('                    first int(F * steps) steps (default 0.4)');
  WriteLn('  --reweight W=S[,W=S] scale the edit''s cross-attention to the ',
    'words W of');
  WriteLn('                    --edit-prompt by S (any mode)');
  WriteLn('  --relight PNG     IC-Light: relight this subject as the prompt ',
    'describes; the');
  WriteLn('                    alpha of an RGBA file (or --birefnet) mattes it ',
    'on gray');
  WriteLn('  --iclight FILE    iclight_sd15_fc.safetensors (or _fbc with ',
    '--relight-background),');
  WriteLn('                    added to the UNet weights; any SD 1.5 folder ',
    'as --model');
  WriteLn('  --light DIR       left, right, top or bottom: start from that ',
    'light gradient');
  WriteLn('                    at --strength (default 0.9); none (default) ',
    'starts from noise');
  WriteLn('  --relight-background PNG the background of an fbc offset');
  WriteLn('  --birefnet FILE   BiRefNet / RMBG-2.0 model.safetensors: matte ',
    'the --relight');
  WriteLn('                    image with it instead of its alpha');
  WriteLn('  --birefnet-config FILE the BiRefNet config.json (default: the one ',
    'beside the');
  WriteLn('                    --birefnet file, else the BiRefNet defaults)');
  WriteLn('  --relight-added-prompt TEXT appended to the prompt after ", " ',
    '(default');
  WriteLn('                    "best quality"; "" adds nothing)');
  WriteLn('  --highres-scale S the highres pass size factor, sides rounded to ',
    'multiples of');
  WriteLn('                    64 (default 1.5; 0 skips the pass)');
  WriteLn('  --highres-strength S the highres pass strength (default 0.5)');
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
  WriteLn('  --sag-scale S     Self-Attention Guidance scale (default 0 = off; ',
    'diffusers');
  WriteLn('                    uses 0.75): one more UNet pass per step');
  WriteLn('  --panorama WxH    MultiDiffusion: a WxH canvas (multiples of the ',
    'VAE factor)');
  WriteLn('                    denoised in --width x --height windows (default ',
    'the checkpoint''s');
  WriteLn('                    size), each with its own CFG, averaged where ',
    'they overlap');
  WriteLn('  --panorama-stride N window step in pixels, a multiple of the VAE ',
    'factor no larger');
  WriteLn('                    than a window side (default 64 = diffusers'' 8 ',
    'latent cells)');
  WriteLn('  --panorama-wrap   wrap the windows past the right edge to the left ',
    'edge (360-degree');
  WriteLn('                    panorama, diffusers circular_padding)');
  WriteLn('  --vae-tile N      VAE-decode and -encode images larger than N ',
    'pixels in overlapping');
  WriteLn('                    N x N tiles (diffusers enable_vae_tiling; ',
    'default 0 = one pass)');
  WriteLn('  --tile-upscale N  upscale --init-image N times (PIL bicubic), then ',
    'image-to-image');
  WriteLn('                    over --width x --height windows every half ',
    'window (--panorama-stride)');
  WriteLn('                    at --strength (default 0.35), VAE-encoded and ',
    '-decoded in');
  WriteLn('                    window-size tiles (--vae-tile)');
  WriteLn('  --controlnet DIR  a diffusers ControlNet folder; under ',
    '--tile-upscale (e.g.');
  WriteLn('                    lllyasviel/control_v11f1e_sd15_tile) its hint is ',
    'the upscaled image,');
  WriteLn('                    each window seeing its own crop; else give ',
    '--control-image');
  WriteLn('  --control-image PNG the --controlnet hint (resized to the output ',
    'size)');
  WriteLn('  --controlnet-scale S ControlNet conditioning scale (default 1.0)');
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

function SamplerText(Method: TNNetSamplerMethod;
  DPMAlgorithm: TNNetDPMSolverAlgorithm): string;
var
  NamePos: integer;
begin
  Result := 'method ' + IntToStr(Ord(Method));
  for NamePos := High(SamplerOptionMethods) downto 0 do
    if (SamplerOptionMethods[NamePos] = Method) and
      ((Method <> smDPMSolverPP2M) or
      (SamplerOptionDPMAlgorithms[NamePos] = DPMAlgorithm)) then
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

// A WxH value of Option (e.g. 2048x512): both sides positive.
procedure ParseSizeSpec(const Option, Spec: string; out Width, Height: integer);
var
  Parts: TStringArray;
begin
  Parts := LowerCase(Spec).Split(['x']);
  if (Length(Parts) <> 2) or not TryStrToInt(Trim(Parts[0]), Width) or
    not TryStrToInt(Trim(Parts[1]), Height) or (Width < 1) or (Height < 1) then
    raise Exception.Create(Option + ' wants WxH, e.g. 2048x512, not "' +
      Spec + '"');
end;

// --reweight value: WORD=SCALE pairs separated by commas, appended to Words
// and Scales.
procedure ParseReweightSpec(const Spec: string; var Words: TStringArray;
  var Scales: TNeuralFloatDynArr);
var
  Pairs: TStringArray;
  PairPos, EqualPos: integer;
begin
  Pairs := Spec.Split([',']);
  for PairPos := 0 to High(Pairs) do
  begin
    EqualPos := Pos('=', Pairs[PairPos]);
    if EqualPos <= 1 then
      raise Exception.Create('--reweight wants WORD=SCALE[,WORD=SCALE], not "' +
        Spec + '"');
    SetLength(Words, Length(Words) + 1);
    SetLength(Scales, Length(Scales) + 1);
    Words[High(Words)] := Trim(Copy(Pairs[PairPos], 1, EqualPos - 1));
    Scales[High(Scales)] := StrToFloat(Trim(Copy(Pairs[PairPos], EqualPos + 1,
      Length(Pairs[PairPos]))), DefaultFormatSettings);
  end;
end;

// --light value: one of LightOptionNames.
function ParseLightDirection(const Name: string): TSDLightDirection;
var
  Direction: TSDLightDirection;
begin
  for Direction := Low(TSDLightDirection) to High(TSDLightDirection) do
    if LowerCase(Name) = LightOptionNames[Direction] then exit(Direction);
  raise Exception.Create('--light wants none, left, right, top or bottom, ' +
    'not "' + Name + '"');
end;

// ICLight.SetMatte of Image's BiRefNet matte (BiRefNetFile model; ConfigFile
// '' reads the config.json beside it).
procedure MatteWithBiRefNet(const BiRefNetFile, ConfigFile: string;
  Image: TNNetVolume; ICLight: TStableDiffusionICLight);
var
  NN: TNNet;
  Config: TBiRefNetConfig;
  RGB, Mask: TNNetVolume;
begin
  NN := nil;
  RGB := TNNetVolume.Create;
  Mask := TNNetVolume.Create;
  try
    NN := BuildBiRefNetFromSafeTensors(BiRefNetFile, Config, {pTrainable=}false,
      ConfigFile);
    RGB.CopyChannels(Image, [0, 1, 2]);
    BiRefNetRemoveBackground(NN, Config, RGB, Mask);
    ICLight.SetMatte(Mask);
  finally
    Mask.Free;
    RGB.Free;
    NN.Free;
  end;
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
  Opt.GLIGENFile := '';
  Opt.GLIGENBoxes := nil;
  Opt.GLIGENBeta := 0.3;
  Opt.GLIGENInpaintImageFile := '';
  Opt.InitImageFile := '';
  Opt.DepthImageFile := '';
  Opt.Strength := -1;
  Opt.ChangeMapFile := '';
  Opt.InvertChangeMap := false;
  Opt.EditImageFile := '';
  Opt.ImageGuidanceScale := -1;
  Opt.UpscaleImageFile := '';
  Opt.NoiseLevel := -1;
  Opt.RelightImageFile := '';
  Opt.ICLightFile := '';
  Opt.RelightBackgroundFile := '';
  Opt.BiRefNetFile := '';
  Opt.BiRefNetConfigFile := '';
  Opt.LightDirection := sdlNone;
  Opt.RelightAddedPrompt := ICLightAddedPrompt;
  Opt.HighResScale := -1;
  Opt.HighResStrength := -1;
  Opt.HasNegativePrompt := false;
  Opt.InvertImageFile := '';
  Opt.InvertSteps := -1;
  Opt.InvertGuidanceScale := -1;
  Opt.NullText := false;
  Opt.NullTextInnerSteps := -1;
  Opt.EditPrompt := '';
  Opt.EditMode := '';
  Opt.CrossReplaceFraction := -1;
  Opt.SelfReplaceFraction := -1;
  Opt.ReweightWords := nil;
  Opt.ReweightScales := nil;
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
  Opt.SAGScale := 0;
  Opt.PanoramaWidth := 0;
  Opt.PanoramaHeight := 0;
  Opt.PanoramaStride := 0;
  Opt.PanoramaWrap := false;
  Opt.VaeTileSize := 0;
  Opt.TileUpscale := 0;
  Opt.ControlNetFolder := '';
  Opt.ControlImageFile := '';
  Opt.ControlNetScale := 1.0;
  SetLength(Opt.Embeddings, 0);
  Result := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--model' then Opt.ModelFolder := NextArg()
    else if Arg = '--prompt' then Opt.Prompt := NextArg()
    else if Arg = '--negative' then
    begin
      Opt.NegativePrompt := NextArg();
      Opt.HasNegativePrompt := true;
    end
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
    else if Arg = '--gligen' then Opt.GLIGENFile := NextArg()
    else if Arg = '--box' then
    begin
      SetLength(Opt.GLIGENBoxes, Length(Opt.GLIGENBoxes) + 1);
      Opt.GLIGENBoxes[High(Opt.GLIGENBoxes)] := NextArg();
    end
    else if Arg = '--gligen-beta' then
      Opt.GLIGENBeta := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--gligen-inpaint-image' then
      Opt.GLIGENInpaintImageFile := NextArg()
    else if Arg = '--init-image' then Opt.InitImageFile := NextArg()
    else if Arg = '--depth-image' then Opt.DepthImageFile := NextArg()
    else if Arg = '--change-map' then Opt.ChangeMapFile := NextArg()
    else if Arg = '--invert-change-map' then Opt.InvertChangeMap := true
    else if Arg = '--mask' then
      raise Exception.Create('--mask (binary inpainting) is not an option ' +
        'here; give a soft --change-map with --init-image instead')
    else if Arg = '--strength' then
      Opt.Strength := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--edit-image' then Opt.EditImageFile := NextArg()
    else if Arg = '--image-guidance' then
      Opt.ImageGuidanceScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--upscale-image' then Opt.UpscaleImageFile := NextArg()
    else if Arg = '--relight' then Opt.RelightImageFile := NextArg()
    else if Arg = '--iclight' then Opt.ICLightFile := NextArg()
    else if Arg = '--relight-background' then
      Opt.RelightBackgroundFile := NextArg()
    else if Arg = '--birefnet' then Opt.BiRefNetFile := NextArg()
    else if Arg = '--birefnet-config' then Opt.BiRefNetConfigFile := NextArg()
    else if Arg = '--light' then
      Opt.LightDirection := ParseLightDirection(NextArg())
    else if Arg = '--relight-added-prompt' then
      Opt.RelightAddedPrompt := NextArg()
    else if Arg = '--highres-scale' then
      Opt.HighResScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--highres-strength' then
      Opt.HighResStrength := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--noise-level' then Opt.NoiseLevel := StrToInt(NextArg())
    else if Arg = '--invert-image' then Opt.InvertImageFile := NextArg()
    else if Arg = '--invert-steps' then Opt.InvertSteps := StrToInt(NextArg())
    else if Arg = '--invert-guidance' then
      Opt.InvertGuidanceScale := StrToFloat(NextArg())
    else if Arg = '--null-text' then Opt.NullText := true
    else if Arg = '--null-text-inner' then
      Opt.NullTextInnerSteps := StrToInt(NextArg())
    else if Arg = '--edit-prompt' then Opt.EditPrompt := NextArg()
    else if Arg = '--edit-mode' then Opt.EditMode := LowerCase(NextArg())
    else if Arg = '--cross-replace' then
      Opt.CrossReplaceFraction := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--self-replace' then
      Opt.SelfReplaceFraction := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--reweight' then
      ParseReweightSpec(NextArg(), Opt.ReweightWords, Opt.ReweightScales)
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
    else if Arg = '--sag-scale' then
      Opt.SAGScale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--panorama' then
      ParseSizeSpec(Arg, NextArg(), Opt.PanoramaWidth, Opt.PanoramaHeight)
    else if Arg = '--panorama-stride' then
      Opt.PanoramaStride := StrToInt(NextArg())
    else if Arg = '--panorama-wrap' then Opt.PanoramaWrap := true
    else if Arg = '--vae-tile' then Opt.VaeTileSize := StrToInt(NextArg())
    else if Arg = '--tile-upscale' then
      Opt.TileUpscale := StrToFloat(NextArg(), DefaultFormatSettings)
    else if Arg = '--controlnet' then Opt.ControlNetFolder := NextArg()
    else if Arg = '--control-image' then Opt.ControlImageFile := NextArg()
    else if Arg = '--controlnet-scale' then
      Opt.ControlNetScale := StrToFloat(NextArg(), DefaultFormatSettings)
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
  if (Opt.SAGScale > 0) and (Opt.EditImageFile <> '') then
    raise Exception.Create('--sag-scale cannot be combined with --edit-image');
  if (Opt.IPAdapterFile = '') <> (Opt.IPImageFile = '') then
    raise Exception.Create('--ip-adapter and --ip-image go together');
  if (Opt.GLIGENFile = '') <> (Length(Opt.GLIGENBoxes) = 0) then
    raise Exception.Create('--gligen and --box go together');
  if (Opt.GLIGENBeta < 0) or (Opt.GLIGENBeta > 1) then
    raise Exception.Create('--gligen-beta must be in [0, 1]');
  if (Opt.GLIGENInpaintImageFile <> '') and (Opt.GLIGENFile = '') then
    raise Exception.Create('--gligen-inpaint-image needs --gligen and --box');
  if (Opt.GLIGENInpaintImageFile <> '') and ((Opt.InitImageFile <> '') or
    (Opt.EditImageFile <> '') or (Opt.UpscaleImageFile <> '')) then
    raise Exception.Create('--gligen-inpaint-image cannot be combined with ' +
      '--init-image, --edit-image or --upscale-image');
  if (Opt.TileUpscale <> 0) and (Opt.TileUpscale < 1) then
    raise Exception.Create('--tile-upscale must be >= 1');
  if (Opt.TileUpscale > 0) and ((Opt.InitImageFile = '') or
    (Opt.DepthImageFile <> '') or (Opt.PanoramaWidth > 0)) then
    raise Exception.Create('--tile-upscale upscales --init-image (no ' +
      '--depth-image) and sets the canvas itself; drop --panorama');
  if (Opt.ControlNetFolder = '') and ((Opt.ControlImageFile <> '') or
    (Opt.ControlNetScale <> 1.0)) then
    raise Exception.Create('--control-image and --controlnet-scale need ' +
      '--controlnet');
  if (Opt.ControlNetFolder <> '') and (Opt.ControlImageFile = '') and
    (Opt.TileUpscale = 0) then
    raise Exception.Create('--controlnet needs --control-image (or ' +
      '--tile-upscale, whose hint is the upscaled image)');
  if (Opt.ControlNetFolder <> '') and ((Opt.EditPrompt <> '') or
    (Opt.InvertImageFile <> '') or (Opt.RelightImageFile <> '')) then
    raise Exception.Create('--controlnet cannot be combined with ' +
      '--edit-prompt, --invert-image or --relight');
  if (Opt.Strength >= 0) and (Opt.InitImageFile = '') and
    (Opt.RelightImageFile = '') then
    raise Exception.Create('--strength needs --init-image or --relight');
  if (Opt.RelightImageFile <> '') <> (Opt.ICLightFile <> '') then
    raise Exception.Create('--relight and --iclight go together');
  if (Opt.RelightImageFile = '') and ((Opt.RelightBackgroundFile <> '') or
    (Opt.BiRefNetFile <> '') or (Opt.LightDirection <> sdlNone) or
    (Opt.RelightAddedPrompt <> ICLightAddedPrompt) or
    (Opt.HighResScale >= 0) or (Opt.HighResStrength >= 0)) then
    raise Exception.Create('--relight-background, --birefnet, --light, ' +
      '--relight-added-prompt and --highres-* need --relight');
  if (Opt.BiRefNetConfigFile <> '') and (Opt.BiRefNetFile = '') then
    raise Exception.Create('--birefnet-config needs --birefnet');
  if (Opt.RelightImageFile <> '') and ((Opt.InitImageFile <> '') or
    (Opt.EditImageFile <> '') or (Opt.UpscaleImageFile <> '') or
    (Opt.InvertImageFile <> '') or (Opt.EditPrompt <> '') or
    (Opt.GLIGENFile <> '') or (Opt.IPAdapterFile <> '')) then
    raise Exception.Create('--relight cannot be combined with --init-image, ' +
      '--edit-image, --upscale-image, --invert-image, --edit-prompt, ' +
      '--gligen or --ip-adapter');
  if (Opt.DepthImageFile <> '') and (Opt.InitImageFile = '') then
    raise Exception.Create('--depth-image needs --init-image');
  if (Opt.ChangeMapFile <> '') and ((Opt.InitImageFile = '') or
    (Opt.DepthImageFile <> '') or (Opt.TileUpscale > 0)) then
    raise Exception.Create('--change-map needs --init-image (and no ' +
      '--depth-image or --tile-upscale)');
  if Opt.InvertChangeMap and (Opt.ChangeMapFile = '') then
    raise Exception.Create('--invert-change-map needs --change-map');
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
  if ((Opt.InvertSteps <> -1) or (Opt.InvertGuidanceScale >= 0)) and
    (Opt.InvertImageFile = '') then
    raise Exception.Create('--invert-steps and --invert-guidance need ' +
      '--invert-image');
  if (Opt.InvertSteps < 1) and (Opt.InvertSteps <> -1) then
    raise Exception.Create('--invert-steps must be >= 1');
  if (Opt.InvertImageFile <> '') and ((Opt.InitImageFile <> '') or
    (Opt.EditImageFile <> '') or (Opt.UpscaleImageFile <> '') or
    (Opt.GLIGENInpaintImageFile <> '')) then
    raise Exception.Create('--invert-image cannot be combined with ' +
      '--init-image, --edit-image, --upscale-image or --gligen-inpaint-image');
  if (Opt.InvertImageFile <> '') and not ((Opt.SamplerName = 'auto') or
    (Opt.SamplerName = 'ddim')) then
    raise Exception.Create('--invert-image samples with ddim');
  if (Opt.InvertImageFile <> '') and (Opt.RepeatCount > 1) then
    raise Exception.Create('--invert-image makes one image; drop --repeat');
  if (Opt.NullText or (Opt.NullTextInnerSteps <> -1)) and
    (Opt.InvertImageFile = '') then
    raise Exception.Create('--null-text and --null-text-inner need ' +
      '--invert-image');
  if (Opt.NullTextInnerSteps <> -1) and not Opt.NullText then
    raise Exception.Create('--null-text-inner needs --null-text');
  if Opt.NullText and (Opt.NullTextInnerSteps < 0) and
    (Opt.NullTextInnerSteps <> -1) then
    raise Exception.Create('--null-text-inner must be >= 0');
  if Opt.NullText and ((Opt.InvertSteps <> -1) or
    (Opt.InvertGuidanceScale >= 0)) then
    raise Exception.Create('--null-text inverts at guidance 1 over --steps; ' +
      'drop --invert-steps and --invert-guidance');
  if (Opt.EditPrompt = '') and ((Opt.EditMode <> '') or
    (Opt.CrossReplaceFraction >= 0) or (Opt.SelfReplaceFraction >= 0) or
    (Length(Opt.ReweightWords) > 0)) then
    raise Exception.Create('--edit-mode, --cross-replace, --self-replace and ' +
      '--reweight need --edit-prompt');
  if not ((Opt.EditMode = '') or (Opt.EditMode = 'swap') or
    (Opt.EditMode = 'refine') or (Opt.EditMode = 'reweight')) then
    raise Exception.Create('unknown --edit-mode: ' + Opt.EditMode);
  if (Opt.EditMode = 'reweight') and (Length(Opt.ReweightWords) = 0) then
    raise Exception.Create('--edit-mode reweight needs --reweight');
  if (Opt.CrossReplaceFraction > 1) or (Opt.SelfReplaceFraction > 1) then
    raise Exception.Create('--cross-replace and --self-replace must be in ' +
      '[0, 1]');
  if (Opt.EditPrompt <> '') and ((Opt.InitImageFile <> '') or
    (Opt.EditImageFile <> '') or (Opt.UpscaleImageFile <> '') or
    (Opt.GLIGENFile <> '') or (Opt.IPAdapterFile <> '')) then
    raise Exception.Create('--edit-prompt cannot be combined with ' +
      '--init-image, --edit-image, --upscale-image, --gligen or --ip-adapter');
  if (Opt.EditPrompt <> '') and ((Opt.DeepCacheInterval > 1) or
    (Opt.PAGScale > 0) or (Opt.SAGScale > 0)) then
    raise Exception.Create('--edit-prompt cannot be combined with ' +
      '--deepcache, --pag-scale or --sag-scale');
  if (Opt.EditPrompt <> '') and not ((Opt.SamplerName = 'auto') or
    (Opt.SamplerName = 'ddim')) then
    raise Exception.Create('--edit-prompt samples with ddim');
  if (Opt.EditPrompt <> '') and (Opt.RepeatCount > 1) then
    raise Exception.Create('--edit-prompt makes one image pair; drop --repeat');
  if (Opt.PanoramaWidth = 0) and ((Opt.PanoramaStride <> 0) or
    Opt.PanoramaWrap) then
    raise Exception.Create('--panorama-stride and --panorama-wrap need ' +
      '--panorama');
  if Opt.PanoramaStride < 0 then
    raise Exception.Create('--panorama-stride must be >= 1');
  if Opt.VaeTileSize < 0 then
    raise Exception.Create('--vae-tile must be >= 0');
  // The windows share one latent canvas and one sampler; these keep a
  // second track, a per-pass cache or image coordinates.
  if ((Opt.PanoramaWidth > 0) or (Opt.TileUpscale > 0)) and
    ((Opt.EditPrompt <> '') or
    (Opt.InvertImageFile <> '') or (Opt.GLIGENFile <> '') or
    (Opt.DeepCacheInterval > 1) or (Opt.UpscaleImageFile <> '') or
    (Opt.RelightImageFile <> '')) then
    raise Exception.Create('--panorama and --tile-upscale cannot be combined ' +
      'with --edit-prompt, --invert-image, --gligen, --deepcache, ' +
      '--upscale-image or --relight');
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

// Splits a --box value "PHRASE:X0,Y0,X1,Y1" at its last ':'.
procedure SplitBoxSpec(const Spec: string; out Phrase: string;
  out Box: array of double);
var
  ColonPos, CoordPos: integer;
  Coords: TStringArray;
begin
  ColonPos := LastDelimiter(':', Spec);
  if ColonPos <= 1 then
    raise Exception.Create('--box ' + Spec + ': expected PHRASE:X0,Y0,X1,Y1');
  Phrase := Copy(Spec, 1, ColonPos - 1);
  Coords := Copy(Spec, ColonPos + 1, Length(Spec)).Split([',']);
  if Length(Coords) <> 4 then
    raise Exception.Create('--box ' + Spec + ': expected 4 coordinates');
  for CoordPos := 0 to 3 do
    Box[CoordPos] := StrToFloat(Trim(Coords[CoordPos]), DefaultFormatSettings);
  if (Box[0] >= Box[2]) or (Box[1] >= Box[3]) then
    raise Exception.Create('--box ' + Spec + ': needs X0 < X1 and Y0 < Y1');
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
  ControlNet: TStableDiffusionControlNet;
  GLIGEN: TStableDiffusionGLIGEN;
  Multi: TStableDiffusionMultiConditioner;
  Img2Img: TStableDiffusionImg2Img;
  GLIGENInpainter: TStableDiffusionGLIGENInpainter;
  Depth2Img: TStableDiffusionDepth2Img;
  Differential: TStableDiffusionDifferentialImg2Img;
  InstructPix2Pix: TStableDiffusionInstructPix2Pix;
  Upscaler: TStableDiffusionUpscaler;
  ICLight: TStableDiffusionICLight;
  ImageConditioner: TStableDiffusionImageConditioner;
  RelightPrompt: string;
  NegativeEmbeds: TNNetVolume;
  Image, PromptEmbeds: TNNetVolume;
  SourceImageFile: string;
  ComputeText, ImageFile, EmbeddingFile, EmbeddingToken, EditText: string;
  ImageWidth, ImageHeight, ImagePos, EmbeddingPos, LoRAPos: integer;
  BoxPos: integer;
  BoxPhrase: string;
  Box: array[0..3] of double;
  PerturbedGuidance: TNNetPerturbedGuidance;
  SelfAttentionNames: TStringArray;
  PAGLayerIndices: TNeuralIntegerArray;
  PAGLayerPos: integer;
  StepCount, Img2ImgStepCount, InvertStepCount: integer;
  InvertGuidanceScale: double;
  InvertImage, InvertedLatents: TNNetVolume;
  NullTextEmbeds: TNNetVolumeList;
  Edit: TSDPromptToPromptEdit;
  EditMode: TSDPromptEditMode;
  EditEmbeds, EditedImage, EditInitialLatents: TNNetVolume;
  EditWords: TStringArray;
  EditTokenIds, EditTokenLengths: TNeuralIntegerArray;
  EditImageFile: string;
  WordPos: integer;
  IsDepthUNet, IsUpscalerUNet: boolean;
  WindowWidth, WindowHeight, PanoramaStride, CanvasWidth, CanvasHeight,
    SourceWidth, SourceHeight: integer;
  UpscaledImage: TNNetVolume;
  PanoramaText: string;
  GuidanceScale: double;
  Seed: cardinal;
  StartTime, SaveStart: TDateTime;
  PngSaveMs: double;
  NullTextInnerSteps: integer;
  {$IFDEF OpenCL}
  OpenCLText: string;
  {$ENDIF}

  // InvertedLatents of InvertImage: null-text inversion (filling NullTextEmbeds)
  // with --null-text, else DDIM inversion.
  procedure InvertForSampling();
  begin
    if Opt.NullText then
    begin
      WriteLn('Null-text inversion:');
      Pipeline.NullTextInversion(InvertImage, PromptEmbeds, ImageWidth,
        ImageHeight, StepCount, Pipeline.GuidanceScale, InvertedLatents,
        NullTextEmbeds, NullTextInnerSteps);
    end
    else
    begin
      WriteLn('Inverting:');
      Pipeline.InvertImage(InvertImage, PromptEmbeds, ImageWidth, ImageHeight,
        InvertStepCount, InvertGuidanceScale, InvertedLatents);
    end;
  end;

begin
  Pipeline := TStableDiffusionPipeline.Create(Opt.ModelFolder);
  Reporter := TStableDiffusionReporter.Create;
  IPAdapter := nil;
  ControlNet := nil;
  GLIGEN := nil;
  Multi := nil;
  Img2Img := nil;
  Depth2Img := nil;
  Differential := nil;
  InstructPix2Pix := nil;
  Upscaler := nil;
  ICLight := nil;
  ImageConditioner := nil;
  Image := TNNetVolume.Create;
  PromptEmbeds := TNNetVolume.Create;
  InvertImage := TNNetVolume.Create;
  InvertedLatents := TNNetVolume.Create;
  NullTextEmbeds := nil;
  Edit := nil;
  EditEmbeds := TNNetVolume.Create;
  EditedImage := TNNetVolume.Create;
  NegativeEmbeds := TNNetVolume.Create;
  UpscaledImage := TNNetVolume.Create;
  SourceWidth := 0;
  SourceHeight := 0;
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
    else if Opt.ChangeMapFile <> '' then
    begin
      Differential := TStableDiffusionDifferentialImg2Img.Create();
      if not LoadImageFromFileIntoVolume(Opt.ChangeMapFile, Image,
        {pWithAlpha=}false) then
        raise Exception.Create('could not load ' + Opt.ChangeMapFile);
      // The reference's own maps are white = keep.
      if Opt.InvertChangeMap then
      begin
        Image.Mul(-1);
        Image.Add(255);
      end;
      Differential.SetChangeMap(Image);
      Img2Img := Differential;
    end
    else if Opt.InitImageFile <> '' then
      Img2Img := TStableDiffusionImg2Img.Create();
    if Assigned(Img2Img) then
    begin
      if Opt.Strength >= 0 then Img2Img.Strength := Opt.Strength
      // Tile upscaling keeps the image and repaints its detail.
      else if Opt.TileUpscale > 0 then Img2Img.Strength := 0.35;
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
    if Opt.RelightImageFile <> '' then
    begin
      Pipeline.UNetOffsetFile := Opt.ICLightFile;
      ICLight := TStableDiffusionICLight.Create();
      ImageConditioner := ICLight;
      if not LoadImageFromFileIntoVolume(Opt.RelightImageFile, Image,
        {pWithAlpha=}true) then
        raise Exception.Create('could not load ' + Opt.RelightImageFile);
      ICLight.SetImage(Image);
      if Opt.BiRefNetFile <> '' then
      begin
        WriteLn('Matting ', Opt.RelightImageFile, ' with ', Opt.BiRefNetFile);
        MatteWithBiRefNet(Opt.BiRefNetFile, Opt.BiRefNetConfigFile, Image,
          ICLight);
      end;
      if Opt.RelightBackgroundFile <> '' then
      begin
        if not LoadImageFromFileIntoVolume(Opt.RelightBackgroundFile, Image) then
          raise Exception.Create('could not load ' + Opt.RelightBackgroundFile);
        ICLight.SetBackground(Image);
      end;
      if not ICLight.AcceptsUNetInChannels(Pipeline.UNetConfig.InChannels,
        Pipeline.UNetConfig.OutChannels) then
        raise Exception.Create(Format('--iclight %s widens the UNet to %d ' +
          'input channels: iclight_sd15_fc (8) relights from the prompt, ' +
          'iclight_sd15_fbc (12) needs --relight-background',
          [Opt.ICLightFile, Pipeline.UNetConfig.InChannels]));
      ICLight.LightDirection := Opt.LightDirection;
      if Opt.Strength >= 0 then ICLight.Strength := Opt.Strength;
      if Opt.HighResScale >= 0 then ICLight.HighResScale := Opt.HighResScale;
      if Opt.HighResStrength >= 0 then
        ICLight.HighResStrength := Opt.HighResStrength;
      // gradio_demo.py samples with DPM++ 2M SDE Karras.
      if Opt.SamplerName = 'auto' then
      begin
        Pipeline.SelectSampler(smDPMSolverPP2M, dpaSDEDPMSolverPP);
        if Opt.KarrasSigmas < 0 then Pipeline.KarrasSigmas := true;
      end;
      Pipeline.Conditioner := ICLight;
    end;
    if Assigned(ImageConditioner) and not Assigned(ICLight) then
    begin
      if not LoadImageFromFileIntoVolume(SourceImageFile, Image) then
        raise Exception.Create('could not load ' + SourceImageFile);
      if Opt.TileUpscale > 0 then
      begin
        // PIL bicubic to N times the size, sides down to the VAE factor.
        SourceWidth := Image.SizeX;
        SourceHeight := Image.SizeY;
        CanvasWidth := Trunc(SourceWidth * Opt.TileUpscale);
        CanvasHeight := Trunc(SourceHeight * Opt.TileUpscale);
        CanvasWidth := CanvasWidth - CanvasWidth mod Pipeline.VaeScaleFactor;
        CanvasHeight := CanvasHeight - CanvasHeight mod Pipeline.VaeScaleFactor;
        if (CanvasWidth < 1) or (CanvasHeight < 1) then
          raise Exception.Create('--tile-upscale: the image is too small');
        ResizeImagePIL(Image, UpscaledImage, CanvasWidth, CanvasHeight,
          irfBicubic);
        Image.Copy(UpscaledImage);
      end;
      ImageConditioner.SetImage(Image);
      ImageConditioner.ImageSizeFor(Pipeline.VaeScaleFactor, ImageWidth,
        ImageHeight);
      Pipeline.Conditioner := ImageConditioner;
    end;
    if Opt.InvertImageFile <> '' then
    begin
      if not LoadImageFromFileIntoVolume(Opt.InvertImageFile, InvertImage) then
        raise Exception.Create('could not load ' + Opt.InvertImageFile);
      ImageWidth := InvertImage.SizeX - InvertImage.SizeX mod
        Pipeline.VaeScaleFactor;
      ImageHeight := InvertImage.SizeY - InvertImage.SizeY mod
        Pipeline.VaeScaleFactor;
      Pipeline.Sampler := smDDIM;
    end;
    if Opt.EditPrompt <> '' then Pipeline.Sampler := smDDIM;
    if Opt.ImageWidth >= 1 then ImageWidth := Opt.ImageWidth;
    if Opt.ImageHeight >= 1 then ImageHeight := Opt.ImageHeight;
    if (Opt.PanoramaWidth > 0) or (Opt.TileUpscale > 0) then
    begin
      // --width x --height (default the checkpoint's) are the windows.
      if Opt.ImageWidth >= 1 then WindowWidth := Opt.ImageWidth
      else WindowWidth := Pipeline.DefaultImageSize;
      if Opt.ImageHeight >= 1 then WindowHeight := Opt.ImageHeight
      else WindowHeight := Pipeline.DefaultImageSize;
      Pipeline.PanoramaWindowWidth := WindowWidth;
      Pipeline.PanoramaWindowHeight := WindowHeight;
      Pipeline.PanoramaStride := Opt.PanoramaStride;
      Pipeline.PanoramaCircular := Opt.PanoramaWrap;
      if Opt.TileUpscale > 0 then
      begin
        // diffusers' 8-cell stride over a 4x canvas is hundreds of windows;
        // half a window keeps the seams overlapped at a quarter of the passes.
        if Opt.PanoramaStride = 0 then
          Pipeline.PanoramaStride := Max(Pipeline.VaeScaleFactor,
            (Min(WindowWidth, WindowHeight) div 2) -
            (Min(WindowWidth, WindowHeight) div 2) mod Pipeline.VaeScaleFactor);
        ImageWidth := UpscaledImage.SizeX;
        ImageHeight := UpscaledImage.SizeY;
      end
      else
      begin
        ImageWidth := Opt.PanoramaWidth;
        ImageHeight := Opt.PanoramaHeight;
      end;
    end;
    Pipeline.VaeTileSize := Opt.VaeTileSize;
    if (Opt.TileUpscale > 0) and (Opt.VaeTileSize = 0) then
      Pipeline.VaeTileSize := Min(Pipeline.PanoramaWindowWidth,
        Pipeline.PanoramaWindowHeight);
    if Opt.SamplerName <> 'auto' then
      Pipeline.SelectSampler(
        SamplerOptionMethods[SamplerOptionIndex(Opt.SamplerName)],
        SamplerOptionDPMAlgorithms[SamplerOptionIndex(Opt.SamplerName)]);
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
    else if Assigned(ICLight) then
    begin
      // gradio_demo.py's defaults.
      if StepCount < 1 then StepCount := 25;
      if GuidanceScale < 0 then GuidanceScale := 2.0;
    end
    else if (Opt.InvertImageFile <> '') and not Opt.NullText then
    begin
      // Guidance 1 keeps the inversion and the sampling on the same path.
      if StepCount < 1 then StepCount := 25;
      if GuidanceScale < 0 then GuidanceScale := 1.0;
    end
    else
    begin
      if StepCount < 1 then StepCount := 25;
      if GuidanceScale < 0 then GuidanceScale := 7.5;
    end;
    InvertStepCount := Opt.InvertSteps;
    if InvertStepCount < 1 then InvertStepCount := StepCount;
    InvertGuidanceScale := Opt.InvertGuidanceScale;
    if InvertGuidanceScale < 0 then InvertGuidanceScale := 1.0;
    NullTextInnerSteps := Opt.NullTextInnerSteps;
    if NullTextInnerSteps < 0 then NullTextInnerSteps := 10;
    if Opt.NullText then NullTextEmbeds := TNNetVolumeList.Create();
    Pipeline.GuidanceScale := GuidanceScale;
    Pipeline.NegativePrompt := Opt.NegativePrompt;
    RelightPrompt := Opt.Prompt;
    if Assigned(ICLight) then
    begin
      if Opt.RelightAddedPrompt <> '' then
        RelightPrompt := Opt.Prompt + ', ' + Opt.RelightAddedPrompt;
      if not Opt.HasNegativePrompt then
        Pipeline.NegativePrompt := ICLightNegativePrompt;
    end;
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
    Pipeline.SAGScale := Opt.SAGScale;
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
      WriteLn('Sampler:  ', SamplerText(Pipeline.Sampler,
        Pipeline.Scheduler.DPMSolverAlgorithm), ' with the ',
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
    if Opt.SAGScale > 0 then
      WriteLn(Format('SAG:      scale %.2f, mid-block self-attention mask',
        [Opt.SAGScale]));
    if Opt.TileUpscale > 0 then
      WriteLn(Format('Tile upscale: %s %dx%d x %g -> %dx%d (PIL bicubic)',
        [Opt.InitImageFile, SourceWidth, SourceHeight, Opt.TileUpscale,
         ImageWidth, ImageHeight]));
    if (Opt.PanoramaWidth > 0) or (Opt.TileUpscale > 0) then
    begin
      PanoramaText := '';
      if Opt.PanoramaWrap then PanoramaText := ', wrapping in X';
      PanoramaStride := Pipeline.PanoramaStride;
      if PanoramaStride = 0 then PanoramaStride := 8 * Pipeline.VaeScaleFactor;
      WriteLn(Format('Panorama: %dx%d windows every %d pixels%s',
        [Pipeline.PanoramaWindowWidth, Pipeline.PanoramaWindowHeight,
         PanoramaStride, PanoramaText]));
    end;
    if Pipeline.VaeTileSize > 0 then
      WriteLn(Format('VAE:      decoded in %d-pixel tiles',
        [Pipeline.VaeTileSize]));
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
      if Assigned(Differential) then
        WriteLn('Change map: ', Opt.ChangeMapFile, ' (Differential Diffusion',
          BoolToStr(Opt.InvertChangeMap, ', inverted', ''), ')');
      if Assigned(Depth2Img) and (Opt.DepthImageFile <> '') then
        WriteLn('Depth image: ', Opt.DepthImageFile, ' (depth-to-image)')
      else if Assigned(Depth2Img) then
        WriteLn('Depth: estimated by depth_estimator/ (depth-to-image)');
    end;
    if Opt.InvertImageFile <> '' then
      if Opt.NullText then
        WriteLn(Format('Invert image: %s (%dx%d), null-text inversion over ' +
          '%d steps, %d inner steps', [Opt.InvertImageFile, InvertImage.SizeX,
          InvertImage.SizeY, StepCount, NullTextInnerSteps]))
      else
        WriteLn(Format('Invert image: %s (%dx%d), %d DDIM inversion steps, ' +
          'guidance %.2f', [Opt.InvertImageFile, InvertImage.SizeX,
          InvertImage.SizeY, InvertStepCount, InvertGuidanceScale]));
    if Opt.EditPrompt <> '' then
    begin
      if Opt.EditMode = 'swap' then EditMode := sdpemSwap
      else EditMode := sdpemRefine;
      Pipeline.PromptWordTokens(Opt.EditPrompt, EditWords, EditTokenIds,
        EditTokenLengths);
      for WordPos := 0 to High(Opt.ReweightWords) do
        if SDPromptWordTokenPositions(EditWords, EditTokenLengths,
          Opt.ReweightWords[WordPos]) = nil then
          raise Exception.Create('--reweight: "' + Opt.ReweightWords[WordPos] +
            '" is not a word of --edit-prompt');
      Edit := Pipeline.CreatePromptToPromptEdit(Opt.Prompt, Opt.EditPrompt,
        EditMode, Opt.ReweightWords, Opt.ReweightScales);
      if Opt.CrossReplaceFraction >= 0 then
        Edit.CrossReplaceFraction := Opt.CrossReplaceFraction;
      if Opt.SelfReplaceFraction >= 0 then
        Edit.SelfReplaceFraction := Opt.SelfReplaceFraction;
      if EditMode = sdpemSwap then EditText := 'swap'
      else if Length(Opt.ReweightWords) > 0 then EditText := 'reweight'
      else EditText := 'refine';
      WriteLn('Edit:     "', Opt.EditPrompt, '"');
      WriteLn(Format('          Prompt-to-Prompt %s; cross-attention for %d, ' +
        'self-attention for %d of %d steps', [EditText,
        Min(StepCount, Trunc(Edit.CrossReplaceFraction * (StepCount + 1))),
        Trunc(Edit.SelfReplaceFraction * StepCount), StepCount]));
      for WordPos := 0 to High(Opt.ReweightWords) do
        WriteLn(Format('          reweight "%s" x %.2f',
          [Opt.ReweightWords[WordPos], Opt.ReweightScales[WordPos]]));
    end;
    if Assigned(InstructPix2Pix) then
      WriteLn(Format('Edit image: %s (%dx%d), image guidance %.2f',
        [Opt.EditImageFile, Image.SizeX, Image.SizeY,
         InstructPix2Pix.ImageGuidanceScale]));
    if Assigned(Upscaler) then
      WriteLn(Format('Upscale image: %s (%dx%d), noise level %d',
        [Opt.UpscaleImageFile, Image.SizeX, Image.SizeY,
         Upscaler.NoiseLevel]));
    if Assigned(ICLight) then
    begin
      WriteLn(Format('Relight:  %s, IC-Light offset %s (%d-channel UNet)',
        [Opt.RelightImageFile, Opt.ICLightFile,
         Pipeline.UNetConfig.InChannels]));
      if Opt.RelightBackgroundFile <> '' then
        WriteLn('Background: ', Opt.RelightBackgroundFile);
      WriteLn('Full prompt: "', RelightPrompt, '"; negative "',
        Pipeline.NegativePrompt, '"');
      if ICLight.LightDirection <> sdlNone then
        WriteLn(Format('Light:    %s gradient, strength %.2f, %d steps of %d',
          [LightOptionNames[ICLight.LightDirection], ICLight.Strength,
           ICLightStepCount(StepCount, ICLight.Strength) -
           TNNetDiffusionScheduler.Img2ImgStartStep(
           ICLightStepCount(StepCount, ICLight.Strength), ICLight.Strength),
           ICLightStepCount(StepCount, ICLight.Strength)]))
      else WriteLn('Light:    none (starts from noise)');
      if ICLight.HighResScale > 0 then
        WriteLn(Format('Highres:  %dx%d, strength %.2f',
          [ICLightHighResSide(ImageWidth, ICLight.HighResScale,
           ICLight.HighResMultiple), ICLightHighResSide(ImageHeight,
           ICLight.HighResScale, ICLight.HighResMultiple),
           ICLight.HighResStrength]))
      else WriteLn('Highres:  off');
    end;
    if Opt.IPAdapterFile <> '' then
    begin
      IPAdapter := TStableDiffusionIPAdapter.Create(Opt.IPAdapterFile);
      if not LoadImageFromFileIntoVolume(Opt.IPImageFile, Image) then
        raise Exception.Create('could not load ' + Opt.IPImageFile);
      IPAdapter.SetImage(Image);
      IPAdapter.Scale := Opt.IPScale;
      WriteLn(Format('IP-Adapter: %s, image %s (%dx%d), scale %.2f',
        [Opt.IPAdapterFile, Opt.IPImageFile, Image.SizeX, Image.SizeY,
         Opt.IPScale]));
    end;
    if Opt.GLIGENFile <> '' then
    begin
      GLIGEN := TStableDiffusionGLIGEN.Create(Opt.GLIGENFile);
      GLIGEN.ScheduledSamplingBeta := Opt.GLIGENBeta;
      WriteLn(Format('GLIGEN:   %s, beta %.2f (%d of %d steps grounded)',
        [Opt.GLIGENFile, Opt.GLIGENBeta, Trunc(Opt.GLIGENBeta * StepCount),
         StepCount]));
      for BoxPos := 0 to High(Opt.GLIGENBoxes) do
      begin
        SplitBoxSpec(Opt.GLIGENBoxes[BoxPos], BoxPhrase, Box);
        if GLIGEN.BoxCount() = csGLIGENMaxObjects then
        begin
          WriteLn('Notice:   only the first ', csGLIGENMaxObjects,
            ' boxes are used');
          break;
        end;
        GLIGEN.AddBox(BoxPhrase, Box[0], Box[1], Box[2], Box[3]);
        WriteLn(Format('  box:    "%s" at %.3f,%.3f - %.3f,%.3f',
          [BoxPhrase, Box[0], Box[1], Box[2], Box[3]]));
      end;
      if Opt.GLIGENInpaintImageFile <> '' then
      begin
        GLIGENInpainter := TStableDiffusionGLIGENInpainter.Create(GLIGEN);
        ImageConditioner := GLIGENInpainter;
        if not LoadImageFromFileIntoVolume(Opt.GLIGENInpaintImageFile, Image) then
          raise Exception.Create('could not load ' + Opt.GLIGENInpaintImageFile);
        GLIGENInpainter.SetImage(Image);
        WriteLn(Format('GLIGEN inpaint image: %s (%dx%d), the boxes ' +
          'regenerated', [Opt.GLIGENInpaintImageFile, Image.SizeX,
          Image.SizeY]));
      end;
    end;
    if Opt.ControlNetFolder <> '' then
    begin
      ControlNet := TStableDiffusionControlNet.Create(Opt.ControlNetFolder);
      // Tile upscaling: the hint is the upscaled image itself; each window's
      // passes see its own crop of it.
      if Opt.ControlImageFile <> '' then
      begin
        if not LoadImageFromFileIntoVolume(Opt.ControlImageFile, Image) then
          raise Exception.Create('could not load ' + Opt.ControlImageFile);
        ControlNet.SetControlImage(Image);
        WriteLn(Format('ControlNet: %s, hint %s (%dx%d), scale %.2f',
          [Opt.ControlNetFolder, Opt.ControlImageFile, Image.SizeX,
           Image.SizeY, Opt.ControlNetScale]));
      end
      else
      begin
        ControlNet.SetControlImage(UpscaledImage);
        WriteLn(Format('ControlNet: %s, hint the upscaled image, scale %.2f',
          [Opt.ControlNetFolder, Opt.ControlNetScale]));
      end;
      ControlNet.ConditioningScale := Opt.ControlNetScale;
    end;
    if Assigned(IPAdapter) or Assigned(GLIGEN) or Assigned(ControlNet) then
    begin
      if Ord(Assigned(ImageConditioner)) + Ord(Assigned(IPAdapter)) +
        Ord(Assigned(GLIGEN)) + Ord(Assigned(ControlNet)) > 1 then
      begin
        Multi := TStableDiffusionMultiConditioner.Create();
        if Assigned(ImageConditioner) then Multi.Add(ImageConditioner);
        if Assigned(IPAdapter) then Multi.Add(IPAdapter);
        if Assigned(GLIGEN) then Multi.Add(GLIGEN);
        if Assigned(ControlNet) then Multi.Add(ControlNet);
        Pipeline.Conditioner := Multi;
      end
      else if Assigned(IPAdapter) then Pipeline.Conditioner := IPAdapter
      else if Assigned(GLIGEN) then Pipeline.Conditioner := GLIGEN
      else Pipeline.Conditioner := ControlNet;
    end;
    // --null-text fits the embeds on the sampling UNet (error-only backprop)
    // and samples back with it: kept, its weights are read once. On OpenCL the
    // fit builds its own trainable CPU UNet, so nothing would be shared.
    if (Opt.RepeatCount > 1) or (Opt.NullText and not Opt.UseOpenCL) then
    begin
      WriteLn('Loading the text encoder, the UNet and the VAE decoder to ',
        'keep them in memory.');
      StartTime := Now();
      // The UNet runs at the window size; the VAE builds for the image later.
      if Pipeline.PanoramaWindowWidth > 0 then
        Pipeline.LoadComponents(Pipeline.PanoramaWindowWidth,
          Pipeline.PanoramaWindowHeight)
      else Pipeline.LoadComponents(ImageWidth, ImageHeight);
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
      if Opt.EditPrompt <> '' then
      begin
        Pipeline.EncodePrompts([Opt.Prompt, Opt.EditPrompt],
          [PromptEmbeds, EditEmbeds]);
        EditInitialLatents := nil;
        if Opt.InvertImageFile <> '' then
        begin
          InvertForSampling();
          EditInitialLatents := InvertedLatents;
        end;
        WriteLn('Sampling the prompt and the edit:');
        Pipeline.GenerateEditFromEmbeds(PromptEmbeds, EditEmbeds, ImageWidth,
          ImageHeight, StepCount, Seed, Edit, Image, EditedImage,
          EditInitialLatents, NullTextEmbeds);
        EditImageFile := ChangeFileExt(ImageFile, '') + '_edit' +
          ExtractFileExt(ImageFile);
        EditedImage.Mul(255.0);
        if not SaveImageFromVolumeIntoFile(EditedImage, EditImageFile) then
          raise Exception.Create('could not save ' + EditImageFile);
        WriteLn('Wrote ', EditImageFile);
      end
      else if Opt.InvertImageFile <> '' then
      begin
        Pipeline.EncodePrompt(Opt.Prompt, PromptEmbeds);
        InvertForSampling();
        WriteLn(Format('Inverted latents: mean abs %.4f',
          [InvertedLatents.GetSumAbs() / InvertedLatents.Size]));
        WriteLn('Sampling from the inverted latents:');
        Pipeline.GenerateFromEmbeds(PromptEmbeds, ImageWidth, ImageHeight,
          StepCount, Seed, Image, InvertedLatents, NullTextEmbeds);
      end
      else if Assigned(ICLight) then
      begin
        // Both prompts in one text encoder load; the shorter one repeats its
        // chunks (the reference encode_prompt_pair).
        if (ImagePos = 0) and (Pipeline.CFGGuidanceScale > 1) then
        begin
          Pipeline.EncodePromptsRepeatingChunks([RelightPrompt,
            Pipeline.NegativePrompt], [PromptEmbeds, NegativeEmbeds]);
          Pipeline.SetNegativePromptEmbeds(NegativeEmbeds);
        end
        else if ImagePos = 0 then
          Pipeline.EncodePrompt(RelightPrompt, PromptEmbeds);
        ICLight.Relight(Pipeline, PromptEmbeds, ImageWidth, ImageHeight,
          StepCount, Seed, Image);
      end
      else if Opt.RepeatCount > 1 then
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
    UpscaledImage.Free;
    NegativeEmbeds.Free;
    EditedImage.Free;
    EditEmbeds.Free;
    Edit.Free;
    NullTextEmbeds.Free;
    InvertedLatents.Free;
    InvertImage.Free;
    PromptEmbeds.Free;
    Image.Free;
    Pipeline.Free;
    Multi.Free;
    GLIGEN.Free;
    ControlNet.Free;
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
