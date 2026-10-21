program QwenImage;
(*
QwenImage: Qwen-Image-2.1 text-to-image and image editing from a diffusers
checkpoint folder
(Qwen/Qwen-Image-2.1: model_index.json, processor/, text_encoder/,
transformer/, vae/, scheduler/), through TQwenImage21Pipeline
(neural/neuralpretrained.pas). The transformer step pass (int8/int4 weights)
and the VAE decode run on OpenCL by default; everything else runs on the CPU:

  prompt -> processor/tokenizer.json -> Qwen3-VL text encoder
    -> transformer prefix pass (text K/V, once)
    -> N flow-matching steps (Euler or DPM-Solver++(2M)) over the image tokens
    -> VAE decode, tiled -> RGBA PNG.

One-shot (-p "prompt" or --token-ids): one image, then exit. Each component
is loaded when its stage starts and freed when it ends, so only one is in
memory at a time. --repeat N > 1 makes N images and keeps the three
components loaded, as the REPL does.

REPL (neither -p nor --token-ids): the three components load once and stay
in memory; stdin gives one prompt per line (a piped file is a batch, end of
input ends the session). Images go to numbered files from the --output base
(qwenimage.png -> qwenimage_0001.png, ...; existing files are skipped, never
overwritten). Commands: /size WxH, /steps N, /seed N (otherwise the seed
grows by one per image), /tile SIZE[,STRIDE], /repeat N PROMPT (N images of
PROMPT with consecutive seeds, the prompt encoded once), /image FILE,
/images [clear], /init-image FILE|off, /mask FILE|off, /mask-blur R,
/padding-mask-crop N|off, /outpaint L,T,R,B [OVERLAP]|off, /strength S,
/upscale FILE|off, /fbc X|off, /panorama WxH|off, /negative-prompt
TEXT|off, /guidance-scale W, /pag W|off, /profile on|off, /stats on|off,
/quit.

Editing (--image FILE, repeatable up to 10, or /image FILE): the prompt is
an instruction about the condition images (image 1 = the first --image).
Each image is resized to its own aspect at the area of --width x --height;
the vision tower and the VAE encoder (both CPU, untiled) encode it once, and
the output takes the LAST image's aspect at that area. Masks or marked-up
copies are just more condition images.

img2img (--init-image FILE or /init-image FILE): SDEdit. The VAE encoder
(CPU, untiled) encodes the init image once (posterior mean); each image
noises those latents to the sigma where the schedule is cut by --strength
(default 0.6) and runs only the remaining steps (the Qwen-Image v1 img2img
schedule). It re-styles the init image; it does not follow instructions.
The output keeps the init image's aspect ratio at the area of --width x
--height (sides multiples of 32, as diffusers' calculate_dimensions).
Images are resized with PIL's Lanczos coefficients in float. EXIF
orientation is not applied.

Inpainting (--init-image FILE --mask FILE, or /mask FILE in the REPL): img2img
that regenerates only the white region of the mask and keeps the rest of the
init image (diffusers QwenImageInpaintPipeline's blended mode, no retrained
model). The mask is read as grayscale (alpha ignored), optionally blurred at
its own size (--mask-blur R: a Gaussian of sigma R pixels, as diffusers'
mask_processor.blur), stretched to the
output size, binarized at 50% and taken per 16x16-pixel latent cell (each
cell's top-left pixel decides). After every step the kept cells are reset to
the init image's latents at the next noise level, and to the latents
themselves after the last step. --strength 1 regenerates the hole from pure
noise; lower values keep more of the init image inside the hole too.
--padding-mask-crop N (diffusers padding_mask_crop) generates only the mask's
bounding box plus N pixels, grown to the output aspect, at the output size,
then pastes it back into the init image at its own size, blended by the mask
(apply_overlay); a small hole in a large photo then gets full resolution.
Outpainting (--init-image FILE --outpaint L,T,R,B, or /outpaint in the REPL):
extends the init image by L, T, R and B pixels on the left, top, right and
bottom. BuildOutpaintCanvas (neuralimageresize) places the image on a canvas
grown to multiples of 32, fills the new border by repeating the edge pixels
and masks that border plus --outpaint-overlap pixels (default 16, one latent
cell) into the image; the inpaint path then regenerates the mask at strength 1
unless --strength is given. The output is the canvas itself (--width and
--height do not apply), cropped to the requested size, and the original pixels
outside the overlap band are pasted back unchanged (diffusers apply_overlay).
GPU memory: in the REPL and with --repeat N > 1 the transformer stays in
OpenCL memory while the VAE decodes, so both need room at once; /tile 128 (or --vae-tile 128) lowers the
VAE's share (~3.4 GB of layer buffers instead of ~10.7 GB at 256).

Upscaling (--upscale FILE or /upscale FILE): a Real-ESRGAN / ESRGAN RRDBNet
checkpoint (RealESRGAN_x4plus .pth or .safetensors) upscales the finished image
before the PNG save through TRRDBNetUpscaler, so 512x512 becomes 2048x2048 at x4.
The upscaler net is built after the VAE decode, for each image, and freed after
it; it runs tile by tile (--upscale-tile, default 32 pixels plus 10 of context
per side) on the --gpu device. Alpha is resized bilinearly.

Guidance (--negative-prompt TEXT --guidance-scale W > 1): each step runs the
transformer twice, on the prompt's and on the negative prompt's prefix
cache, and mixes the velocities as uncond + W * (cond - uncond) (diffusers
QwenImage21Pipeline true_cfg_scale; Qwen-Image-2.1 is meant to run without).
--apg ETA,MOMENTUM,THRESHOLD mixes them with Adaptive Projected Guidance on
the denoised x0 instead; --apg-velocity applies it to the raw velocities.

Perturbed-Attention Guidance (--pag-scale W, or /pag W): on the steps inside
the --pag-start..--pag-stop window (fractions of the steps run, strict at
both ends, default 0.01..0.2) one more transformer pass runs on the prompt
with the --pag-layers blocks' attention scores replaced by the identity, and
the velocity moves away from it by W (diffusers PerturbedAttentionGuidance).
--slg skips those blocks instead (Skip Layer Guidance). It adds to CFG, not
to --apg.

First-Block Cache (--fbc-threshold X or /fbc X): each step runs transformer
block 0 and compares its residual with the one of the last fully computed
step; when the relative change is at most X it skips the other blocks and
reuses their cached residual (diffusers apply_first_block_cache). 0 = off.

Panoramas and large canvases (--panorama WxH or /panorama WxH): MultiDiffusion
(Bar-Tal et al. 2023, diffusers StableDiffusionPanoramaPipeline). Each step
runs the transformer on overlapping --width x --height windows of the WxH
canvas (every --panorama-stride pixels, default half a window) and averages
the windows' velocities per latent cell, so memory stays at one window.
--panorama-wrap wraps the windows in X for a 360-degree panorama. Each window
sees only its own region, so the global composition is weaker than one pass.

The initial noise comes from the FPC RNG (--seed), so an image is repeatable
here but not equal to a diffusers image with the same seed.

USAGE
  QwenImage --model DIR [-p TEXT | --token-ids ID,ID,... [--drop-count N]]
            [--output FILE.png]
            [--width 1024] [--height 1024] [--steps 18] [--seed 42]
            [--repeat N] [--image FILE ... |
             --init-image FILE [--mask FILE [--mask-blur R]
             [--padding-mask-crop N] |
             --outpaint L,T,R,B [--outpaint-overlap 16]] [--strength 0.6]]
            [--int8 | --int4 | --fp32] [--int8-input] [--vae-tile SIZE[,STRIDE]]
            [--lora FILE[:SCALE] ...] [--upscale FILE [--upscale-tile 32]]
            [--fbc-threshold X]
            [--negative-prompt TEXT --guidance-scale W
             [--apg ETA,MOMENTUM,THRESHOLD [--apg-velocity]]]
            [--pag-scale W [--pag-layers I,J,...] [--pag-start F]
             [--pag-stop F] [--slg]]
            [--panorama WxH [--panorama-stride N] [--panorama-wrap]]
            [--serial] [--max-threads N]
            [--gpu | --cpu] [--gpu-platform N] [--gpu-device N]
            [--no-gpu-shared-kernel] [--profile] [--stats]
Run with --help for what each flag does.

Coded by Claude (AI).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)
{$mode objfpc}{$H+}

uses
  // cmem is skipped in the Debug build mode: it enables Valgrind (-gv), and
  // FPC then pulls in cmem itself, so naming it here is a duplicate.
  {$IFDEF UNIX}cthreads, {$IFNDEF Debug}cmem,{$ENDIF} termio, BaseUnix,{$ENDIF}
  SysUtils, Classes, Types, {$IFDEF OpenCL}neuralopencl,{$ENDIF}
  neuralvolume, neuralnetwork, neuralpretrained, neuraldatasets, neuralthread,
  neuraldiffusion, neuralimageresize;

const
  // Largest accepted image side in pixels (64x the 1024 default area).
  csMaxImageSide = 8192;
  // img2img strength when --strength is not given (diffusers' default).
  csDefaultStrength = 0.6;
  // Outpaint seam band when --outpaint-overlap is not given: one latent cell.
  csDefaultOutpaintOverlap = csQwenImage21PixelsPerLatent;
  // Largest --repeat / /repeat count (catches a mistyped count).
  csMaxRepeatCount = 1000;
  // --upscale-tile when not given: RRDBNet input pixels per tile side.
  csDefaultUpscaleTile = 32;
  csBytesPerMB = 1024.0 * 1024.0;
  // PAG / SLG when only some of the --pag-* flags are given: diffusers'
  // guider defaults, and blocks of the 32-block transformer (README.md).
  csDefaultPagScale = 2.8;
  csDefaultPagStart = 0.01;
  csDefaultPagStop = 0.2;
  csDefaultPagLayers: array[0..0] of integer = (16);
  csDefaultSlgLayers: array[0..2] of integer = (6, 7, 8);

type
  // Phase and step timings with the process memory. On a terminal each step
  // rewrites one line; a pipe or file keeps one line per step for logs.
  TQwenImageReporter = class(TObject)
  private
    FPhaseStart, FStepStart: QWord;
    FPhaseName: string;
    FRewritesStepLine, FStepLineOpen: boolean;
    procedure EndPhase();
    // Width of the stdout terminal in characters; 80 when unknown.
    function TerminalColumns(): integer;
  public
    constructor Create();
    // Ends a step line left open on the terminal; call before other output.
    procedure EndStepLine();
    procedure OnPhase(Phase: TQwenImage21PipelinePhase);
    procedure OnStep(StepIndex, StepCount: integer; Timestep: double;
      Latents: TNNetVolume);
  end;

// VmRSS and VmHWM of /proc/self/status in MB ('?' when unreadable); false
// without that file.
function ReadProcMemory(out Rss, Peak: string): boolean;
var
  Status: TStringList;
  LinePos: integer;
  Line: string;

  function Megabytes(const StatusLine: string): string;
  var
    Kilobytes: int64;
  begin
    Kilobytes := StrToInt64Def(Trim(StringReplace(Copy(StatusLine,
      Pos(':', StatusLine) + 1, MaxInt), 'kB', '', [])), -1);
    if Kilobytes < 0 then Result := '?'
    else Result := IntToStr(Kilobytes div 1024);
  end;

begin
  Rss := '?';
  Peak := '?';
  Result := FileExists('/proc/self/status');
  if not Result then exit;
  Status := TStringList.Create;
  try
    try
      Status.LoadFromFile('/proc/self/status');
    except
      exit;
    end;
    for LinePos := 0 to Status.Count - 1 do
    begin
      Line := Status[LinePos];
      if Pos('VmRSS:', Line) = 1 then Rss := Megabytes(Line)
      else if Pos('VmHWM:', Line) = 1 then Peak := Megabytes(Line);
    end;
  finally
    Status.Free;
  end;
end;

// 'RSS 123 MB, peak 456 MB' from /proc/self/status; '' elsewhere.
function MemoryReport(): string;
var
  Rss, Peak: string;
begin
  if ReadProcMemory(Rss, Peak) then
    Result := 'RSS ' + Rss + ' MB, peak ' + Peak + ' MB'
  else Result := '';
end;

// Resets the kernel's peak-RSS mark (VmHWM) to the current RSS; false when
// /proc/self/clear_refs is not writable.
function ResetPeakRss(): boolean;
{$IFDEF UNIX}
var
  Handle: cint;
  Command: char;
{$ENDIF}
begin
  Result := false;
  {$IFDEF UNIX}
  Handle := FpOpen('/proc/self/clear_refs', O_WRONLY);
  if Handle < 0 then exit;
  Command := '5';
  Result := FpWrite(Handle, Command, 1) = 1;
  FpClose(Handle);
  {$ENDIF}
end;

function PhaseName(Phase: TQwenImage21PipelinePhase): string;
begin
  case Phase of
    qppLoadTextEncoder: Result := 'load text encoder';
    qppEncodePrompt: Result := 'encode prompt';
    qppLoadVisionTower: Result := 'load vision tower';
    qppEncodeVision: Result := 'vision encode';
    qppLoadVaeEncoder: Result := 'load VAE encoder';
    qppEncodeImage: Result := 'VAE encode';
    qppLoadTransformer: Result := 'load transformer';
    qppEncodePrefix: Result := 'transformer prefix';
    qppDenoise: Result := 'denoise';
    qppLoadVae: Result := 'load VAE';
    qppDecode: Result := 'VAE decode';
  else
    Result := 'done';
  end;
end;

// The per-image stage table of --stats and --profile: one row per phase that
// ran, upscale (UpscaleRanOn '' = none), PNG save, their sum; then OpenCL
// transfers and bytes held, peak RSS.
procedure PrintStageTable(const Stats: TQwenImage21ImageStats;
  ImageNumber, EncodeImageNumber, InputImageEncodeNumber: integer;
  UpscaleMs: double; const UpscaleRanOn, UpscaleDetail: string;
  PngSaveMs: double; IsProfiled, IsPeakPerImage: boolean);
var
  Phase: TQwenImage21PipelinePhase;
  TotalMs: double;
  RanOn, Detail, Rss, Peak: string;
  HasEncode, UsedImages, HasImageEncode, IsOnOpenCL: boolean;

  // The phase ran for this image, not for an earlier one or not at all.
  function IsCountedHere(Phase: TQwenImage21PipelinePhase): boolean;
  begin
    case Phase of
      qppLoadTextEncoder, qppEncodePrompt: Result := HasEncode;
      qppLoadVisionTower, qppEncodeVision, qppLoadVaeEncoder, qppEncodeImage:
        Result := HasImageEncode;
    else
      Result := true;
    end;
  end;

  function Share(Ms: double): double;
  begin
    if TotalMs > 0 then Result := 100.0 * Ms / TotalMs else Result := 0;
  end;

  function Megabytes(Bytes: int64): double;
  begin
    Result := Bytes / csBytesPerMB;
  end;

  // Upload and download MB of Phase divided by Count.
  function TransferText(Phase: TQwenImage21PipelinePhase;
    Count: integer): string;
  begin
    if Count < 1 then Count := 1;
    Result := Format('up %.2f MB, down %.2f MB',
      [Megabytes(Stats.PhaseUploadBytes[Phase]) / Count,
       Megabytes(Stats.PhaseDownloadBytes[Phase]) / Count]);
  end;

  procedure AddRow(const Stage, Where, Ms, Pct, RowDetail: string);
  begin
    WriteLn(TrimRight(Format('  %-20s %-10s %10s %6s  %s', [Stage, Where, Ms,
      Pct, RowDetail])));
  end;

  procedure AddTimedRow(const Stage, Where: string; Ms: double;
    const RowDetail: string);
  begin
    AddRow(Stage, Where, Format('%.1f', [Ms]), Format('%.1f', [Share(Ms)]),
      RowDetail);
  end;

begin
  HasEncode := ImageNumber = EncodeImageNumber;
  UsedImages := Stats.UsedInitImage or (Stats.ConditionImageCount > 0);
  HasImageEncode := UsedImages and (ImageNumber = InputImageEncodeNumber);
  IsOnOpenCL := Stats.TransformerOnOpenCL or Stats.VaeOnOpenCL;
  TotalMs := PngSaveMs + UpscaleMs;
  for Phase := Low(TQwenImage21PipelinePhase) to qppDecode do
    if IsCountedHere(Phase) then
      TotalMs := TotalMs + Stats.PhaseMs[Phase];
  WriteLn('Stages of image ', ImageNumber, ' (wall time):');
  if IsProfiled and IsOnOpenCL then
    WriteLn('  (--profile drains OpenCL after every layer: these walls are ',
      'higher than without it)');
  AddRow('Stage', 'Ran on', 'ms', '%', '');
  for Phase := Low(TQwenImage21PipelinePhase) to qppDecode do
  begin
    if not IsCountedHere(Phase) then
    begin
      if Phase = qppEncodePrompt then
        AddRow(PhaseName(Phase), '-', '-', '-', 'encoded once, before image ' +
          IntToStr(EncodeImageNumber) + ' (counted there)')
      else if ((Phase = qppEncodeImage) and UsedImages) or
        ((Phase = qppEncodeVision) and (Stats.ConditionImageCount > 0)) then
        AddRow(PhaseName(Phase), '-', '-', '-', 'encoded once, before image ' +
          IntToStr(InputImageEncodeNumber) + ' (counted there)');
      continue;
    end;
    // A phase that did not run this image (no images, or no load) is left out.
    if (Stats.PhaseMs[Phase] = 0) and ((Phase in [qppLoadTextEncoder,
      qppLoadVisionTower, qppLoadVaeEncoder, qppLoadTransformer, qppLoadVae])
      or ((Phase = qppEncodeVision) and (Stats.ConditionImageCount = 0))) then
      continue;
    RanOn := 'CPU';
    Detail := '';
    case Phase of
      qppLoadTransformer, qppLoadVae:
        if Stats.PhaseUploadBytes[Phase] > 0 then
        begin
          RanOn := 'CPU+upload';
          Detail := Format('uploads %.1f MB',
            [Megabytes(Stats.PhaseUploadBytes[Phase])]);
        end;
      qppDenoise:
      begin
        if Stats.TransformerOnOpenCL then RanOn := 'OpenCL';
        if Stats.StepCount > 0 then
          Detail := Format('%d steps x %.1f ms', [Stats.StepCount,
            Stats.PhaseMs[Phase] / Stats.StepCount]);
        if Stats.PanoramaWindowCount > 0 then
          Detail := Detail + Format(', %d panorama windows',
            [Stats.PanoramaWindowCount]);
        if (Stats.SkippedStepCount > 0) and (Stats.PanoramaWindowCount > 0)
          then Detail := Detail + Format(', %d window passes with blocks ' +
            '1..N-1 skipped (FBC)', [Stats.SkippedStepCount])
        else if Stats.SkippedStepCount > 0 then
          Detail := Detail + Format(', %d with blocks 1..N-1 skipped (FBC)',
            [Stats.SkippedStepCount]);
        if Stats.PerturbedPassCount > 0 then
          Detail := Detail + Format(', %d perturbed passes (PAG/SLG)',
            [Stats.PerturbedPassCount]);
      end;
      qppDecode:
      begin
        if Stats.VaeOnOpenCL then RanOn := 'OpenCL';
        if Stats.VaeTileCount > 0 then
          Detail := Format('%d tile(s) x %.1f ms, %d net(s) built',
            [Stats.VaeTileCount, Stats.PhaseMs[Phase] / Stats.VaeTileCount,
             Stats.VaeNetCount]);
      end;
    end;
    AddTimedRow(PhaseName(Phase), RanOn, Stats.PhaseMs[Phase], Detail);
  end;
  if UpscaleRanOn <> '' then
    AddTimedRow('upscale', UpscaleRanOn, UpscaleMs, UpscaleDetail);
  AddTimedRow('PNG save', 'CPU', PngSaveMs, '');
  AddRow('sum of stages', '', Format('%.1f', [TotalMs]), '100.0', '');
  if IsOnOpenCL then
  begin
    if Stats.TransfersCounted then
      WriteLn('  host<->OpenCL: step pass ', TransferText(qppDenoise,
        Stats.StepCount), ' per step; VAE decode ', TransferText(qppDecode,
        Stats.VaeTileCount), ' per tile')
    else
      WriteLn('  host<->OpenCL: not counted (--stats turns the counting on)');
    WriteLn(Format('  OpenCL memory held: transformer %.1f MB after the last ' +
      'step, VAE %.1f MB (largest tile net); sampled, not every allocation',
      [Megabytes(Stats.TransformerOpenCLBytes),
       Megabytes(Stats.VaeOpenCLBytes)]));
  end;
  if ReadProcMemory(Rss, Peak) then
  begin
    if IsPeakPerImage then
      WriteLn('  peak RSS this image: ', Peak, ' MB')
    else
      WriteLn('  peak RSS of the process: ', Peak, ' MB');
  end;
end;

constructor TQwenImageReporter.Create();
begin
  inherited Create();
  FPhaseName := '';
  FRewritesStepLine :=
    {$IFDEF UNIX}IsATTY(StdOutputHandle) = 1{$ELSE}false{$ENDIF};
  FStepLineOpen := false;
end;

function TQwenImageReporter.TerminalColumns(): integer;
{$IFDEF UNIX}
var
  WinSize: TWinSize;
{$ENDIF}
begin
  Result := 80;
  {$IFDEF UNIX}
  WinSize := Default(TWinSize);
  if (FpIOCtl(StdOutputHandle, TIOCGWINSZ, @WinSize) = 0) and
    (WinSize.ws_col > 0) then Result := WinSize.ws_col;
  {$ENDIF}
end;

procedure TQwenImageReporter.EndStepLine();
begin
  if not FStepLineOpen then exit;
  WriteLn;
  FStepLineOpen := false;
end;

procedure TQwenImageReporter.EndPhase();
begin
  EndStepLine();
  if FPhaseName = '' then exit;
  WriteLn(Format('  %-20s %8.1f s   %s', [FPhaseName,
    (GetTickCount64 - FPhaseStart) / 1000, MemoryReport()]));
end;

procedure TQwenImageReporter.OnPhase(Phase: TQwenImage21PipelinePhase);
begin
  EndPhase();
  FPhaseStart := GetTickCount64;
  FStepStart := FPhaseStart;
  if Phase = qppDone then FPhaseName := '' else FPhaseName := PhaseName(Phase);
end;

procedure TQwenImageReporter.OnStep(StepIndex, StepCount: integer;
  Timestep: double; Latents: TNNetVolume);
var
  StepEnd: QWord;
  StepText: string;
begin
  StepEnd := GetTickCount64;
  StepText := Format('    step %d/%d  t=%7.2f  %6.1f s  |x| max %.3f  %s',
    [StepIndex + 1, StepCount, Timestep, (StepEnd - FStepStart) / 1000,
     Latents.GetMaxAbs(), MemoryReport()]);
  if FRewritesStepLine then
  begin
    Write(#13, Copy(StepText, 1, TerminalColumns() - 1), #27'[K');
    Flush(System.Output);
    FStepLineOpen := true;
  end
  else WriteLn(StepText);
  FStepStart := StepEnd;
end;

function IntegerListText(const Values: array of integer): string;
var
  ValuePos: integer;
begin
  Result := '';
  for ValuePos := 0 to High(Values) do
  begin
    if ValuePos > 0 then Result := Result + ',';
    Result := Result + IntToStr(Values[ValuePos]);
  end;
end;

function DefaultPagLayersText(IsSkipLayer: boolean): string;
begin
  if IsSkipLayer then Result := IntegerListText(csDefaultSlgLayers)
  else Result := IntegerListText(csDefaultPagLayers);
end;

procedure PrintHelp();
begin
  WriteLn('QwenImage: Qwen-Image-2.1 text-to-image and image editing (CPU; ',
    'OpenCL for the transformer');
  WriteLn('step pass and the VAE decode).');
  WriteLn('  --model DIR          diffusers folder with model_index.json (required)');
  WriteLn('  -p TEXT              one-shot: generate this prompt, write ',
    '--output and exit');
  WriteLn('  --output FILE        image file; .png keeps the alpha channel ',
    '(default qwenimage.png).');
  WriteLn('                       The REPL numbers it: qwenimage_0001.png, ',
    'qwenimage_0002.png, ...');
  WriteLn('  --width N, --height N  pixels, rounded DOWN to a multiple of 32 ',
    '(default 1024)');
  WriteLn('  --steps N            sampler steps (default 18)');
  WriteLn('  --sampler NAME       euler or dpm2m (DPM-Solver++(2M), second ',
    'order; default euler)');
  WriteLn('  --seed N             initial-noise seed for the FPC RNG (default 42)');
  WriteLn('  --image FILE         edit: a condition image; repeat it for up ',
    'to ', csQwenImage21MaxConditionImages, ' images (image 1, 2, ...');
  WriteLn('                       in the order given). The prompt says what ',
    'to change. Each image');
  WriteLn('                       is resized to its aspect at the area of ',
    '--width x --height; the');
  WriteLn('                       output takes the LAST image''s aspect at ',
    'that area. Needs');
  WriteLn('                       processor/preprocessor_config.json. Not ',
    'with --init-image.');
  WriteLn('  --init-image FILE    img2img (SDEdit) from this init image: the ',
    'output keeps its aspect');
  WriteLn('                       ratio at the area of --width x --height. It ',
    're-styles the image;');
  WriteLn('                       it does not follow edit instructions.');
  WriteLn('  --mask FILE          with --init-image: inpaint; only the ',
    'mask''s white region is');
  WriteLn('                       regenerated (grayscale, 50% threshold, ',
    'stretched to the output');
  WriteLn('                       size, decided per 16x16 cell), the rest ',
    'keeps the init image');
  WriteLn('  --mask-blur R        blur the mask by a Gaussian of sigma R ',
    'pixels at its own size before');
  WriteLn('                       the 50% threshold (diffusers ',
    'mask_processor.blur; default 0 = off)');
  WriteLn('  --padding-mask-crop N  with --mask: generate only the mask''s ',
    'bounding box plus N pixels');
  WriteLn('                       (grown to the output aspect) at the output ',
    'size, then paste it back into');
  WriteLn('                       the init image at its own size, blended by ',
    'the mask (diffusers');
  WriteLn('                       padding_mask_crop); the output keeps the ',
    'init image''s size');
  WriteLn('  --outpaint L,T,R,B   with --init-image: outpaint; add L, T, R, B ',
    'pixels on the left, top,');
  WriteLn('                       right and bottom. The output is the init ',
    'image plus that border at its');
  WriteLn('                       own scale (--width/--height do not apply); ',
    'the border and the overlap');
  WriteLn('                       band are generated, the other original ',
    'pixels are pasted back exactly.');
  WriteLn('                       Strength 1 unless --strength is given. Not ',
    'with --mask.');
  WriteLn('  --outpaint-overlap N pixels of the init image along each ',
    'extended side that are regenerated');
  WriteLn('                       too, so the seam blends (default ',
    csDefaultOutpaintOverlap, ')');
  WriteLn('  --strength S         img2img noise strength in (0, 1] (default ',
    csDefaultStrength:0:1, '): runs only the last');
  WriteLn('                       part of the schedule (18 steps at 0.6: the ',
    'last 11); 1 = from pure noise');
  WriteLn('  --repeat N           with -p or --token-ids: N images, seeds ',
    '--seed, --seed+1, ...; the prompt');
  WriteLn('                       is encoded once. N > 1 keeps the ',
    'components loaded and numbers the');
  WriteLn('                       files like the REPL (default 1, at most ',
    csMaxRepeatCount, ')');
  WriteLn('  --int8               transformer block weights in int8; text ',
    'encoder weights in int8 (DEFAULT)');
  WriteLn('  --int4               transformer block weights in Q4_0-style int4 ',
    '(block 32); text encoder weights in int8');
  WriteLn('  --fp32               both load FP32 (the 7B transformer then runs ',
    'a slow kernel)');
  WriteLn('                       Of --int8, --int4 and --fp32 the last one ',
    'given wins.');
  WriteLn('                       Norms, embeddings, the transformer''s ',
    'input/output/timestep nets and the VAE stay FP32.');
  WriteLn('  --int8-input         int8 activations into the transformer''s ',
    'int8/int4 projections (not with --fp32)');
  WriteLn('  --lora FILE[:SCALE]  merge a LoRA .safetensors (diffusers, ComfyUI ',
    'or kohya keys) into the');
  WriteLn('                       transformer weights at load, times SCALE ',
    '(default 1). Repeat it to stack');
  WriteLn('                       adapters.');
  WriteLn('  --vae-tile S[,T]     VAE tile S pixels every T pixels, multiples ',
    'of 16 (default 256,192)');
  WriteLn('  --upscale FILE       upscale each image with this Real-ESRGAN / ',
    'ESRGAN RRDBNet checkpoint');
  WriteLn('                       (.pth or .safetensors; config.json next to ',
    'it, else read from the tensors)');
  WriteLn('                       before the save: x4plus turns 512x512 into ',
    '2048x2048. Runs on the --gpu device.');
  WriteLn('  --upscale-tile N     upscaler tile in input pixels, 0 = the whole ',
    'image (default ', csDefaultUpscaleTile, ');');
  WriteLn('                       a tile holds about 2.7 MB per pixel of ',
    '(N + 20)^2 on the CPU');
  WriteLn('  --fbc-threshold X    First-Block Cache: skip transformer blocks ',
    '1..N-1 on a step whose block-0');
  WriteLn('                       residual changed by at most X (relative ',
    'mean |diff|) since the last full');
  WriteLn('                       step, reusing that step''s residual of ',
    'those blocks. 0 = off (DEFAULT);');
  WriteLn('                       0.05..0.2 trade detail for speed (see ',
    'README.md)');
  WriteLn('  --negative-prompt TEXT  what the image should not show; used ',
    'when --guidance-scale > 1');
  WriteLn('  --guidance-scale W   classifier-free guidance: uncond + W * ',
    '(cond - uncond), two transformer');
  WriteLn('                       passes per step. 1 = off (DEFAULT; ',
    'Qwen-Image-2.1 is meant to run without)');
  WriteLn('  --apg ETA,MOMENTUM,THRESHOLD  Adaptive Projected Guidance ',
    'instead of the CFG mix (paper');
  WriteLn('                       setting e.g. 0,-0.5,15), on the denoised ',
    'x0 = x - sigma * v');
  WriteLn('  --apg-velocity       apply --apg to the raw velocities instead ',
    'of x0');
  WriteLn('  --pag-scale W        Perturbed-Attention Guidance: one more ',
    'transformer pass per step inside');
  WriteLn('                       the window, with identity attention in ',
    '--pag-layers; 0 = off (DEFAULT).');
  WriteLn('                       --pag-layers, --pag-start, --pag-stop or ',
    '--slg alone use W = ', csDefaultPagScale:0:1, ' (diffusers)');
  WriteLn('  --pag-layers I,J,... the perturbed transformer blocks (from 0; ',
    'default ', DefaultPagLayersText(false), ',');
  WriteLn('                       with --slg ', DefaultPagLayersText(true),
    '; see README.md)');
  WriteLn('  --pag-start F        perturbed window start, a fraction of the ',
    'steps (default ', csDefaultPagStart:0:2, ')');
  WriteLn('  --pag-stop F         perturbed window stop (default ',
    csDefaultPagStop:0:2, '); both ends are strict, as diffusers');
  WriteLn('  --slg                Skip Layer Guidance: skip the --pag-layers ',
    'blocks in that pass instead');
  WriteLn('  --panorama WxH       MultiDiffusion: a WxH canvas (multiples of ',
    '32) denoised in overlapping');
  WriteLn('                       --width x --height windows whose velocities ',
    'are averaged per latent cell;');
  WriteLn('                       memory stays at one window. Not with ',
    '--init-image or --image (see README.md)');
  WriteLn('  --panorama-stride N  window step in pixels, a multiple of 16 no ',
    'larger than a window side');
  WriteLn('                       (default: half the window side)');
  WriteLn('  --panorama-wrap      wrap the windows past the right edge to the ',
    'left edge (360-degree panorama)');
  WriteLn('  --serial             single-threaded forward passes (default: ',
    'the parallel layer scheduler with intra-layer threading)');
  WriteLn('  --max-threads N      cap the parallel forward at N worker threads ',
    '(default: every CPU thread)');
  WriteLn('  --token-ids LIST     one-shot from comma-separated prompt token ',
    'ids instead of -p (no tokenizer');
  WriteLn('                       needed); with --image, one <|image_pad|> ',
    'id per image (expanded here)');
  WriteLn('  --drop-count N       leading system-prompt tokens to drop with ',
    '--token-ids (default 0)');
  WriteLn('  --gpu                OpenCL for the transformer step pass and ',
    'the VAE decode (DEFAULT when');
  WriteLn('                       built with -dOpenCL). With --fp32 the step ',
    'pass runs on the CPU.');
  WriteLn('                       The text encoder and the transformer ',
    'prefix pass always run on the CPU.');
  WriteLn('  --cpu                run everything on the CPU');
  WriteLn('  --gpu-platform N     OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N       OpenCL device index within the platform ',
    '(default 0)');
  WriteLn('  --no-gpu-shared-kernel  give each layer private OpenCL kernels ',
    'and command queue instead');
  WriteLn('                       of the net-wide shared ones (default: ',
    'shared, which is faster)');
  WriteLn('  --profile            after the image, per-layer time of the ',
    'transformer step pass (by block');
  WriteLn('                       role and by layer class, summed over the ',
    'blocks and steps), of the');
  WriteLn('                       prefix pass and of the VAE decode (by ',
    'layer class, one table per');
  WriteLn('                       tile shape). The OpenCL queue is drained ',
    'after every layer that fed it,');
  WriteLn('                       so each row includes its kernels and ',
    'transfers; steps run slower.');
  WriteLn('                       With --no-gpu-shared-kernel the layer ',
    'queues are private: the table header');
  WriteLn('                       says which queues are drained. Also ',
    'prints the --stats stage table.');
  WriteLn('  --stats              after each image, a stage table: where each ',
    'stage ran, its wall time');
  WriteLn('                       and share, ms per step and per VAE tile, ',
    'host<->OpenCL MB per step');
  WriteLn('                       and per tile, the OpenCL memory held and ',
    'the peak RSS of the image.');
  WriteLn('                       No per-layer drain; it counts transfers ',
    '(a few integer adds each).');
  WriteLn('                       --stats and --profile reset the kernel''s ',
    'peak-RSS mark when an image');
  WriteLn('                       starts, so getrusage ru_maxrss and ',
    '/usr/bin/time -v then report only');
  WriteLn('                       the peak since the last image started.');
  WriteLn;
  WriteLn('Without -p or --token-ids, QwenImage loads the text encoder, the ',
    'transformer and the VAE once');
  WriteLn('and reads one prompt per line from stdin (a piped file is a batch). ',
    'REPL commands:');
  WriteLn('  /size WxH            image size for the next prompts');
  WriteLn('  /steps N             sampler steps for the next prompts');
  WriteLn('  /seed N              seed of the next image (otherwise the seed ',
    'grows by one per image)');
  WriteLn('  /repeat N PROMPT     N images of PROMPT (encoded once), ',
    'consecutive seeds, numbered files');
  WriteLn('  /image FILE          add a condition image (edit) for the next ',
    'prompts, up to ', csQwenImage21MaxConditionImages);
  WriteLn('  /images              list the condition images; /images clear ',
    'removes them all and frees');
  WriteLn('                       the vision tower and the VAE encoder');
  WriteLn('  /init-image FILE     init image for the next prompts (img2img); ',
    '/init-image off clears it');
  WriteLn('  /mask FILE           inpaint mask for the init image (white = ',
    'regenerate); /mask off');
  WriteLn('                       clears it');
  WriteLn('  /mask-blur R         mask blur sigma in pixels, as --mask-blur; ',
    'reloads the current mask');
  WriteLn('  /padding-mask-crop N inpaint only the mask''s box plus N pixels, ',
    'as --padding-mask-crop;');
  WriteLn('                       /padding-mask-crop off inpaints the whole ',
    'image');
  WriteLn('  /outpaint L,T,R,B [OVERLAP]  outpaint the init image, as ',
    '--outpaint (and --outpaint-overlap);');
  WriteLn('                       /outpaint off returns to img2img');
  WriteLn('  /strength S          img2img strength for the next images, ',
    'in (0, 1]');
  WriteLn('  /upscale FILE        upscale the next images, as --upscale; ',
    '/upscale off stops it');
  WriteLn('  /tile SIZE[,STRIDE]  VAE tile, as --vae-tile. In the REPL (and ',
    'with --repeat N > 1) the');
  WriteLn('                       transformer stays in OpenCL memory ',
    'during the VAE decode; /tile 128');
  WriteLn('                       (or --vae-tile 128) lowers the VAE''s ',
    'memory if both do not fit.');
  WriteLn('  /fbc X|off           First-Block Cache threshold for the next ',
    'images, as --fbc-threshold');
  WriteLn('  /panorama WxH|off    --panorama canvas for the next prompts ',
    '(stride and wrap from the flags)');
  WriteLn('  /negative-prompt TEXT|off  --negative-prompt for the next prompts');
  WriteLn('  /guidance-scale W    --guidance-scale for the next prompts');
  WriteLn('  /pag W|off           --pag-scale for the next images (layers, ',
    'window and --slg from the flags)');
  WriteLn('  /profile on|off      --profile for the next images');
  WriteLn('  /stats on|off        --stats for the next images');
  WriteLn('  /quit                end the session (so does the end of the input)');
end;

// Parses SIZE[,STRIDE] (STRIDE defaults to 3/4 of SIZE, a multiple of 16);
// returns '' or what is wrong.
function ParseVaeTile(const Text: string;
  out TileSize, TileStride: integer): string;
var
  CommaPos: integer;
begin
  CommaPos := Pos(',', Text);
  if CommaPos > 0 then
  begin
    TileSize := StrToIntDef(Trim(Copy(Text, 1, CommaPos - 1)), -1);
    TileStride := StrToIntDef(Trim(Copy(Text, CommaPos + 1, MaxInt)), -1);
  end
  else
  begin
    TileSize := StrToIntDef(Trim(Text), -1);
    TileStride := (TileSize * 3 div 4) div 16 * 16;
    if TileStride < 16 then TileStride := 16;
  end;
  if (TileSize < 1) or (TileStride < 1) then
    exit('"' + Text + '" is not SIZE[,STRIDE] with positive integers.');
  Result := QwenImage21VaeTileProblem(TileSize, TileStride,
    csQwenImage21PixelsPerLatent);
end;

// '' when both sides are in csQwenImage21ImageSideMultiple..csMaxImageSide;
// otherwise what is wrong.
function ImageSideProblem(Width, Height: integer): string;
begin
  Result := '';
  if (Width < csQwenImage21ImageSideMultiple) or
     (Height < csQwenImage21ImageSideMultiple) or
     (Width > csMaxImageSide) or (Height > csMaxImageSide) then
    Result := 'image sides must be in ' +
      IntToStr(csQwenImage21ImageSideMultiple) + '..' +
      IntToStr(csMaxImageSide) + ' pixels.';
end;

// Parses WxH and rounds each side down to a multiple of 32; returns '' or
// what is wrong.
function ParseImageSize(const Text: string;
  out Width, Height: integer): string;
var
  CrossPos: integer;
begin
  CrossPos := Pos('x', LowerCase(Text));
  Width := -1;
  Height := -1;
  if CrossPos > 0 then
  begin
    Width := StrToIntDef(Trim(Copy(Text, 1, CrossPos - 1)), -1);
    Height := StrToIntDef(Trim(Copy(Text, CrossPos + 1, MaxInt)), -1);
  end;
  Result := ImageSideProblem(Width, Height);
  if Result <> '' then exit('"' + Text + '": ' + Result);
  Width := TQwenImage21Pipeline.RoundDownImageSide(Width);
  Height := TQwenImage21Pipeline.RoundDownImageSide(Height);
end;

// DefaultFormatSettings with a '.' decimal point, whatever the locale.
function PointFormat(): TFormatSettings;
begin
  Result := DefaultFormatSettings;
  Result.DecimalSeparator := '.';
end;

// Strength as ParseStrength reads it ('.' decimal point).
function StrengthText(Strength: double): string;
begin
  Result := FloatToStr(Strength, PointFormat());
end;

// Parses an img2img strength in (0, 1] (a '.' decimal point); returns '' or
// what is wrong.
function ParseStrength(const Text: string; out Strength: double): string;
begin
  if not TryStrToFloat(Trim(Text), Strength, PointFormat()) or
    not ((Strength > 0) and (Strength <= 1)) then
    Result := '"' + Text + '" is not a number in (0, 1].'
  else Result := '';
end;

// Parses a number >= 0 or 'off' (= 0) with a '.' decimal point: the
// First-Block Cache threshold and the PAG scale; returns '' or what is wrong.
function ParseNumberOrOff(const Text: string; out Value: double): string;
begin
  Result := '';
  if LowerCase(Trim(Text)) = 'off' then Value := 0
  else if not TryStrToFloat(Trim(Text), Value, PointFormat()) or
    not (Value >= 0) then
    Result := '"' + Text + '" is not a number >= 0 or off.';
end;

// Parses a guidance scale >= 0 (a '.' decimal point); returns '' or what is
// wrong.
function ParseGuidanceScale(const Text: string; out Scale: double): string;
begin
  Result := '';
  if not TryStrToFloat(Trim(Text), Scale, PointFormat()) or
    not (Scale >= 0) then
    Result := '"' + Text + '" is not a number >= 0.';
end;

function FbcThresholdText(Threshold: double): string;
begin
  if Threshold > 0 then
    Result := 'First-Block Cache threshold ' +
      FloatToStr(Threshold, PointFormat())
  else Result := 'First-Block Cache off';
end;

// Parses a --lora FILE[:SCALE] value (a '.' decimal point; no number after the
// last ':' leaves it in the file name); returns '' or what is wrong.
function ParseLoRAArg(const Text: string; out FileName: string;
  out Scale: double): string;
var
  ColonPos: integer;
begin
  FileName := Text;
  Scale := 1.0;
  ColonPos := LastDelimiter(':', Text);
  if (ColonPos > 0) and TryStrToFloat(Copy(Text, ColonPos + 1, MaxInt), Scale,
    PointFormat()) then
    FileName := Copy(Text, 1, ColonPos - 1)
  else Scale := 1.0;
  if not FileExists(FileName) then
    Result := 'file "' + FileName + '" not found.'
  else Result := '';
end;

// Parses a mask blur radius >= 0 in pixels (a '.' decimal point); returns ''
// or what is wrong.
function ParseMaskBlurRadius(const Text: string; out Radius: double): string;
begin
  if not TryStrToFloat(Trim(Text), Radius, PointFormat()) or
    not ((Radius >= 0) and (Radius <= csMaxImageSide)) then
    Result := '"' + Text + '" is not a number in [0, ' +
      IntToStr(csMaxImageSide) + '].'
  else Result := '';
end;

// Parses a pixel count in 0..csMaxImageSide (an outpaint overlap, a mask crop
// padding); returns '' or what is wrong.
function ParsePixelCount(const Text: string; out PixelCount: integer): string;
begin
  PixelCount := StrToIntDef(Trim(Text), -1);
  if (PixelCount < 0) or (PixelCount > csMaxImageSide) then
    Result := '"' + Text + '" is not an integer in 0..' +
      IntToStr(csMaxImageSide) + '.'
  else Result := '';
end;

// Parses a --panorama-stride: a positive multiple of 16 pixels; returns ''
// or what is wrong.
function ParsePanoramaStride(const Text: string; out Stride: integer): string;
begin
  Stride := StrToIntDef(Trim(Text), -1);
  if (Stride < csQwenImage21PixelsPerLatent) or (Stride > csMaxImageSide) or
     (Stride mod csQwenImage21PixelsPerLatent <> 0) then
    Result := '"' + Text + '" is not a multiple of ' +
      IntToStr(csQwenImage21PixelsPerLatent) + ' in ' +
      IntToStr(csQwenImage21PixelsPerLatent) + '..' + IntToStr(csMaxImageSide) +
      '.'
  else Result := '';
end;

// Parses a repeat count in 1..csMaxRepeatCount; returns '' or what is wrong.
function ParseRepeatCount(const Text: string; out RepeatCount: integer): string;
begin
  RepeatCount := StrToIntDef(Trim(Text), -1);
  if (RepeatCount < 1) or (RepeatCount > csMaxRepeatCount) then
    Result := '"' + Text + '" is not an integer in 1..' +
      IntToStr(csMaxRepeatCount) + '.'
  else Result := '';
end;

// The first BaseFile_NNNN file after FileNumber that does not exist yet
// (.png when BaseFile has no extension); FileNumber becomes its number.
function NextFreeOutputFile(const BaseFile: string;
  var FileNumber: integer): string;
var
  Extension: string;
begin
  Extension := ExtractFileExt(BaseFile);
  if Extension = '' then Extension := '.png';
  repeat
    Inc(FileNumber);
    Result := ChangeFileExt(BaseFile, '') + Format('_%.4d', [FileNumber]) +
      Extension;
  until not FileExists(Result);
end;

// Prompt shortened to MaxLength characters for a log line.
function ShortPrompt(const Prompt: string; MaxLength: integer): string;
begin
  if Length(Prompt) <= MaxLength then Result := Prompt
  else Result := Copy(Prompt, 1, MaxLength - 3) + '...';
end;

// Comma-separated integers (token ids, PAG layers); EConvertError otherwise.
function ParseIntegerList(const List: string): TNeuralIntegerArray;
var
  Parts: TStringList;
  PartPos: integer;
begin
  Parts := TStringList.Create;
  try
    Parts.StrictDelimiter := true;
    Parts.Delimiter := ',';
    Parts.DelimitedText := List;
    SetLength(Result, Parts.Count);
    for PartPos := 0 to Parts.Count - 1 do
      Result[PartPos] := StrToInt(Trim(Parts[PartPos]));
  finally
    Parts.Free;
  end;
end;

var
  ModelFolder, Prompt, OutputFile, TokenList, Arg, ArgProblem,
    OutpaintText: string;
  Width, Height, StepCount, DropCount, ArgPos: integer;
  RepeatCount, ImageNumber, FileNumber: integer;
  Seed: cardinal;
  HasPrompt, UseInt8Input, UseSerial, UseProfile, UseStats: boolean;
  // The image whose stage table shows the encode of the current prompt.
  EncodeImageNumber: integer;
  // img2img: the init image (0..255 RGBA) and its latents, encoded at
  // ImageLatentsWidth x ImageLatentsHeight (0: not encoded yet).
  HasInitImage: boolean;
  InitImageFile: string;
  InitImage, ImageLatents: TNNetVolume;
  ImageLatentsWidth, ImageLatentsHeight: integer;
  // Inpainting: the mask as read (0..255) and its latents, encoded at
  // MaskLatentsWidth x MaskLatentsHeight (0: not encoded yet). It only
  // applies while an init image is set.
  HasMask: boolean;
  MaskFile: string;
  // Gaussian sigma in mask pixels that LoadMask blurs the mask with (0: none).
  MaskBlurRadius: double;
  MaskImage, MaskLatents: TNNetVolume;
  MaskLatentsWidth, MaskLatentsHeight: integer;
  // --padding-mask-crop pixels (-1: off). While IsCropInpainting(): the mask
  // at the init image's size, its crop region and both crops at output size.
  PaddingMaskCrop: integer;
  InpaintCrop: TRect;
  InpaintFullMask, InpaintCropImage, InpaintCropMask: TNNetVolume;
  // The crop region the image latents were encoded from (empty: no crop).
  ImageLatentsCrop: TRect;
  // Outpainting: the pixels added per side of the init image and the seam
  // band; the canvas sides grow to multiples of 32 (cropped after decode).
  HasOutpaint: boolean;
  OutpaintLeft, OutpaintTop, OutpaintRight, OutpaintBottom,
    OutpaintOverlap: integer;
  CanvasLeft, CanvasTop, CanvasRight, CanvasBottom: integer;
  // The canvas (0..255) and its mask (0..255, blurred by MaskBlurRadius),
  // built from the init image while IsOutpainting().
  OutpaintCanvas, OutpaintMask: TNNetVolume;
  // Edit: the condition images as read (0..255) and their files; the
  // pipeline holds their encoding at ConditionEncodedArea pixels (0: none).
  // ImageArgFiles: the --image files, read once the output area is known.
  ConditionFiles, ImageArgFiles: TStringList;
  // The --lora files and their scales, in the order given.
  LoRAFiles: TStringList;
  LoRAScales: array of double;
  LoRAFile: string;
  LoRAScale: double;
  LoRAPos: integer;
  ConditionImages: array of TNNetVolume;
  ConditionEncodedArea: double;
  // The N+1 text runs of the encoded edit prompt.
  TextLengths: TNeuralIntegerArray;
  // The image whose stage table shows the init or condition image encode.
  InputImageEncodeNumber: integer;
  Strength: double;
  HasStrengthArg: boolean;
  // A --pag-* or --slg flag was given, and --pag-scale among them.
  HasPagArg, HasPagScaleArg: boolean;
  PagWindowValue: double;
  FbcThreshold: double;
  // --panorama canvas (0: off), --panorama-stride (0: half a window) and
  // --panorama-wrap; the windows are --width x --height.
  PanoramaWidth, PanoramaHeight, PanoramaStride: integer;
  PanoramaWrap: boolean;
  // --sampler: the flow-matching update rule of every Denoise.
  Sampler: TNNetFlowSampler;
  SamplerName: string;
  // --negative-prompt (HasNegativePrompt: given, even empty), --guidance-scale,
  // --apg (ApgSpec '': plain CFG) and --apg-velocity.
  NegativePrompt, ApgSpec: string;
  HasNegativePrompt, ApgOnVelocity: boolean;
  GuidanceScale: double;
  NegativeEmbeds: TNNetVolume;
  // --pag-scale (0: off), --pag-start, --pag-stop, --pag-layers (nil: the
  // default of the mode) and --slg (skip the blocks instead).
  PagScale, PagStart, PagStop: double;
  PagLayers: TNeuralIntegerArray;
  IsSkipLayerGuidance: boolean;
  // The kernel's peak-RSS mark was reset for the current image.
  IsPeakPerImage: boolean;
  WeightFormat: TQwenImage21WeightFormat;
  UseOpenCL, HasSharedKernel: boolean;
  OpenCLPlatform, OpenCLDevice: integer;
  ComputeText: string;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  OpenCLProblem: string;
  RequestedPlatform, RequestedDevice: integer;
  {$ENDIF}
  VaeTileSize, VaeTileStride, MaxThreads: integer;
  // --upscale / /upscale: nil when off; built from UpscaleTile.
  Upscaler: TRRDBNetUpscaler;
  UpscaleFile: string;
  UpscaleTile: integer;
  // The pipeline got an OpenCL device; the upscaler is armed on it too.
  IsOpenCLSelected: boolean;
  Pipeline: TQwenImage21Pipeline;
  Reporter: TQwenImageReporter;
  PromptEmbeds, Image: TNNetVolume;
  TokenIds: TNeuralIntegerArray;
  StartTime: QWord;

  function RanOnText(OnOpenCL: boolean): string;
  begin
    if OnOpenCL then Result := 'OpenCL'
    else Result := 'the CPU (see the notice above)';
  end;

  function NextArg(): string;
  begin
    Inc(ArgPos);
    if ArgPos > ParamCount then
    begin
      WriteLn('Missing value after ', Arg, '.');
      Halt(2);
    end;
    Result := ParamStr(ArgPos);
  end;

  procedure PrintComputeResult();
  begin
    if UseOpenCL then
      WriteLn('Compute    : the transformer step pass runs on ',
        RanOnText(Pipeline.TransformerOnOpenCL), ', the VAE decode on ',
        RanOnText(Pipeline.VaeOnOpenCL));
  end;

  function IsEditing(): boolean;
  begin
    Result := Length(ConditionImages) > 0;
  end;

  // Both prompts run each step (diffusers: a scale > 1 and a negative prompt).
  function IsGuided(): boolean;
  begin
    Result := (GuidanceScale > 1) and HasNegativePrompt;
  end;

  function GuidanceText(): string;
  begin
    if not IsGuided() then
    begin
      Result := 'off';
      if HasNegativePrompt then
        Result := Result + ' (the negative prompt needs --guidance-scale > 1)'
      else if GuidanceScale > 1 then
        Result := Result + ' (--guidance-scale needs --negative-prompt)';
      exit;
    end;
    Result := 'scale ' + FloatToStr(GuidanceScale, PointFormat()) +
      ', negative prompt "' + ShortPrompt(NegativePrompt, 40) + '", ';
    if ApgSpec = '' then Result := Result + 'CFG'
    else Result := Result + 'APG ' + ApgSpec + ' on ' +
      BoolToStr(ApgOnVelocity, 'velocity', 'x0');
  end;

  // --pag-layers, else the default blocks of the --slg / PAG mode.
  function PagLayersText(): string;
  begin
    if Length(PagLayers) > 0 then Result := IntegerListText(PagLayers)
    else Result := DefaultPagLayersText(IsSkipLayerGuidance);
  end;

  function PerturbedGuidanceText(): string;
  begin
    if PagScale = 0 then exit('off');
    Result := BoolToStr(IsSkipLayerGuidance, 'SLG', 'PAG') + ' scale ' +
      FloatToStr(PagScale, PointFormat()) + ' on blocks ' +
      PagLayersText() + ', steps in (' +
      FloatToStr(PagStart, PointFormat()) + ', ' +
      FloatToStr(PagStop, PointFormat()) + ') of the run';
    if IsGuided() and (ApgSpec <> '') then
      Result := Result + ' (refused with --apg while guided)';
  end;

  // Hands the PAG / SLG settings to the pipeline (nil when off).
  procedure ApplyPerturbedGuidance();
  var
    Guider: TNNetPerturbedGuidance;
  begin
    if PagScale = 0 then
    begin
      Pipeline.PerturbedGuidance := nil;
      exit;
    end;
    Guider := TNNetPerturbedGuidance.Create(1, PagScale, PagStart, PagStop);
    if IsSkipLayerGuidance then Guider.Mode := pgmSkipBlock;
    if Length(PagLayers) > 0 then Guider.SetPerturbedLayers(PagLayers)
    else if IsSkipLayerGuidance then
      Guider.SetPerturbedLayers(csDefaultSlgLayers)
    else Guider.SetPerturbedLayers(csDefaultPagLayers);
    Pipeline.PerturbedGuidance := Guider;
  end;

  // A panorama canvas is set and no init or condition image overrides it.
  function IsPanorama(): boolean;
  begin
    Result := (PanoramaWidth > 0) and not HasInitImage and not IsEditing();
  end;

  // The image size: --width x --height, or with SourceImage <> nil its aspect
  // ratio at that area (diffusers' calculate_dimensions).
  procedure GetOutputSizeFor(SourceImage: TNNetVolume; out OutputWidth,
    OutputHeight: integer);
  begin
    if Assigned(SourceImage) then
      QwenImage21SizeForAspect(double(Width) * Height,
        SourceImage.SizeX / SourceImage.SizeY, OutputWidth, OutputHeight)
    else
    begin
      OutputWidth := Width;
      OutputHeight := Height;
    end;
  end;

  // GetOutputSizeFor the init image, else the edit's size (the pipeline's
  // once encoded), else --width x --height.
  procedure GetOutputSize(out OutputWidth, OutputHeight: integer);
  begin
    if HasInitImage and HasOutpaint then
    begin
      OutputWidth := OutpaintCanvas.SizeX;
      OutputHeight := OutpaintCanvas.SizeY;
    end
    else if HasInitImage then GetOutputSizeFor(InitImage, OutputWidth, OutputHeight)
    else if IsEditing() and (ConditionEncodedArea = double(Width) * Height)
      then
    begin
      OutputWidth := Pipeline.ConditionOutputWidth;
      OutputHeight := Pipeline.ConditionOutputHeight;
    end
    else if IsEditing() then
      GetOutputSizeFor(ConditionImages[High(ConditionImages)], OutputWidth,
        OutputHeight)
    else if IsPanorama() then
    begin
      OutputWidth := PanoramaWidth;
      OutputHeight := PanoramaHeight;
    end
    else GetOutputSizeFor(nil, OutputWidth, OutputHeight);
  end;

  // '' when SourceImage's output size is usable; otherwise what is wrong.
  function OutputSizeProblemFor(SourceImage: TNNetVolume): string;
  var
    OutputWidth, OutputHeight: integer;
  begin
    GetOutputSizeFor(SourceImage, OutputWidth, OutputHeight);
    Result := ImageSideProblem(OutputWidth, OutputHeight);
    if (Result <> '') and Assigned(SourceImage) then
      Result := 'the image''s aspect gives ' + IntToStr(OutputWidth) +
        'x' + IntToStr(OutputHeight) + ': ' + Result;
  end;

  // Reads FileName (with alpha) into Dest when its size at the output area
  // works; otherwise returns what is wrong.
  function ReadImageFile(const FileName: string; Dest: TNNetVolume): string;
  begin
    Result := '';
    try
      if not LoadImageFromFileIntoVolume(FileName, Dest, true) then
        Result := 'could not read ' + FileName + '.';
    except
      on E: Exception do Result := 'could not read ' + FileName + ': ' +
        E.Message;
    end;
    if Result = '' then Result := OutputSizeProblemFor(Dest);
  end;

  // An init image and outpaint sides are set.
  function IsOutpainting(): boolean;
  begin
    Result := HasInitImage and HasOutpaint;
  end;

  // '' when Image outpainted by the given sides makes a usable canvas;
  // otherwise what is wrong.
  function OutpaintCanvasProblem(Image: TNNetVolume; Left, Top, Right,
    Bottom: integer): string;
  begin
    GrowOutpaintSidesToMultiple(Image.SizeX, Image.SizeY,
      csQwenImage21ImageSideMultiple, Left, Top, Right, Bottom);
    Result := ImageSideProblem(Image.SizeX + Left + Right,
      Image.SizeY + Top + Bottom);
    if Result <> '' then
      Result := 'the outpaint canvas would be ' +
        IntToStr(Image.SizeX + Left + Right) + 'x' +
        IntToStr(Image.SizeY + Top + Bottom) + ': ' + Result;
  end;

  // Rebuilds the canvas and its mask from the init image while outpainting;
  // the encoded latents no longer match them.
  procedure BuildOutpaint();
  var
    Mask: TNNetVolume;
  begin
    ImageLatentsWidth := 0;
    MaskLatentsWidth := 0;
    if not IsOutpainting() then exit;
    CanvasLeft := OutpaintLeft;
    CanvasTop := OutpaintTop;
    CanvasRight := OutpaintRight;
    CanvasBottom := OutpaintBottom;
    GrowOutpaintSidesToMultiple(InitImage.SizeX, InitImage.SizeY,
      csQwenImage21ImageSideMultiple, CanvasLeft, CanvasTop, CanvasRight,
      CanvasBottom);
    Mask := TNNetVolume.Create();
    try
      BuildOutpaintCanvas(InitImage, CanvasLeft, CanvasTop, CanvasRight,
        CanvasBottom, OutpaintOverlap, OutpaintCanvas, Mask);
      Mask.Mul(255);
      GaussianBlurImage(Mask, OutpaintMask, MaskBlurRadius);
    finally
      Mask.Free;
    end;
  end;

  // Makes Left..Bottom with Overlap the outpaint sides when the init image
  // (if any) gives a usable canvas; otherwise returns what is wrong.
  function SetOutpaint(Left, Top, Right, Bottom, Overlap: integer): string;
  begin
    Result := '';
    if HasInitImage then
      Result := OutpaintCanvasProblem(InitImage, Left, Top, Right, Bottom);
    if Result <> '' then exit;
    HasOutpaint := true;
    OutpaintLeft := Left;
    OutpaintTop := Top;
    OutpaintRight := Right;
    OutpaintBottom := Bottom;
    OutpaintOverlap := Overlap;
    BuildOutpaint();
  end;

  // Copies the requested outpaint area out of the decoded canvas Image (0..255)
  // and pastes the init image's kept pixels back over it.
  procedure FinishOutpaint();
  var
    Cropped: TNNetVolume;
  begin
    Cropped := TNNetVolume.Create();
    try
      Cropped.CopyCropping(Image, CanvasLeft - OutpaintLeft,
        CanvasTop - OutpaintTop,
        InitImage.SizeX + OutpaintLeft + OutpaintRight,
        InitImage.SizeY + OutpaintTop + OutpaintBottom);
      PasteOutpaintOriginal(InitImage, OutpaintLeft, OutpaintTop,
        OutpaintRight, OutpaintBottom, OutpaintOverlap, Cropped);
      Image.Copy(Cropped);
    finally
      Cropped.Free;
    end;
  end;

  // Loads FileName and makes it the init image when its output size (or its
  // outpaint canvas) works; otherwise returns what is wrong and changes nothing.
  function LoadInitImage(const FileName: string): string;
  var
    Loaded: TNNetVolume;
  begin
    Loaded := TNNetVolume.Create();
    try
      Result := ReadImageFile(FileName, Loaded);
      if (Result = '') and HasOutpaint then
        Result := OutpaintCanvasProblem(Loaded, OutpaintLeft, OutpaintTop,
          OutpaintRight, OutpaintBottom);
      if Result <> '' then exit;
      InitImage.Copy(Loaded);
    finally
      Loaded.Free;
    end;
    HasInitImage := true;
    InitImageFile := FileName;
    ImageLatentsWidth := 0;
    ImageLatentsHeight := 0;
    BuildOutpaint();
  end;

  // Loads FileName as the inpaint mask (any size: it is stretched to the
  // output size), blurred by MaskBlurRadius; otherwise returns what is wrong.
  function LoadMask(const FileName: string): string;
  var
    Loaded: TNNetVolume;
  begin
    Result := '';
    Loaded := TNNetVolume.Create();
    try
      try
        if not LoadImageFromFileIntoVolume(FileName, Loaded, true) then
          Result := 'could not read ' + FileName + '.';
      except
        on E: Exception do Result := 'could not read ' + FileName + ': ' +
          E.Message;
      end;
      if Result <> '' then exit;
      GaussianBlurImage(Loaded, MaskImage, MaskBlurRadius);
    finally
      Loaded.Free;
    end;
    HasMask := true;
    MaskFile := FileName;
    MaskLatentsWidth := 0;
    MaskLatentsHeight := 0;
  end;

  // An init image and a mask are set.
  function IsInpainting(): boolean;
  begin
    Result := HasInitImage and HasMask;
  end;

  // The mask latents for GenerateFromEmbeds: nil unless inpainting.
  function ActiveMaskLatents(): TNNetVolume;
  begin
    if IsInpainting() or IsOutpainting() then Result := MaskLatents
    else Result := nil;
  end;

  // Inpainting with --padding-mask-crop.
  function IsCropInpainting(): boolean;
  begin
    Result := IsInpainting() and (PaddingMaskCrop >= 0);
  end;

  // Sets InpaintFullMask, InpaintCrop (diffusers get_crop_region) and the two
  // crops resized as resize_mode "fill" to the output size; raises if no mask.
  procedure BuildInpaintCrop(OutputWidth, OutputHeight: integer);
  var
    Cropped: TNNetVolume;
    X1, Y1, X2, Y2: integer;
  begin
    if (MaskImage.SizeX = InitImage.SizeX) and
      (MaskImage.SizeY = InitImage.SizeY) then
    begin
      InpaintFullMask.ReSize(MaskImage);
      InpaintFullMask.Copy(MaskImage);
    end
    else ResizeImagePIL(MaskImage, InpaintFullMask, InitImage.SizeX,
      InitImage.SizeY, irfLanczos);
    GetMaskCropRegion(InpaintFullMask, OutputWidth, OutputHeight,
      PaddingMaskCrop, X1, Y1, X2, Y2);
    InpaintCrop := Rect(X1, Y1, X2, Y2);
    Cropped := TNNetVolume.Create();
    try
      Cropped.CopyCropping(InitImage, X1, Y1, X2 - X1, Y2 - Y1);
      ResizeImageAndFill(Cropped, InpaintCropImage, OutputWidth, OutputHeight,
        irfLanczos);
      Cropped.CopyCropping(InpaintFullMask, X1, Y1, X2 - X1, Y2 - Y1);
      ResizeImageAndFill(Cropped, InpaintCropMask, OutputWidth, OutputHeight,
        irfLanczos);
    finally
      Cropped.Free;
    end;
  end;

  // Appends FileName to the condition images when it reads and fits;
  // otherwise returns what is wrong and changes nothing.
  function AddConditionImage(const FileName: string): string;
  var
    Loaded: TNNetVolume;
  begin
    if Length(ConditionImages) >= csQwenImage21MaxConditionImages then
      exit('an edit takes at most ' +
        IntToStr(csQwenImage21MaxConditionImages) + ' images.');
    Loaded := TNNetVolume.Create();
    Result := ReadImageFile(FileName, Loaded);
    if Result <> '' then
    begin
      Loaded.Free;
      exit;
    end;
    SetLength(ConditionImages, Length(ConditionImages) + 1);
    ConditionImages[High(ConditionImages)] := Loaded;
    ConditionFiles.Add(FileName);
    ConditionEncodedArea := 0;
  end;

  procedure ClearConditionImages();
  var
    ImagePos: integer;
  begin
    for ImagePos := 0 to High(ConditionImages) do
      ConditionImages[ImagePos].Free;
    ConditionImages := nil;
    ConditionFiles.Clear;
    ConditionEncodedArea := 0;
    if Assigned(Pipeline) then Pipeline.ClearConditionImages();
  end;

  // One line per condition image: number, file, size as read.
  procedure ListConditionImages();
  var
    ImagePos: integer;
  begin
    for ImagePos := 0 to High(ConditionImages) do
      WriteLn('  image ', ImagePos + 1, ': ', ConditionFiles[ImagePos], ' (',
        ConditionImages[ImagePos].SizeX, 'x', ConditionImages[ImagePos].SizeY,
        ')');
  end;

  // Encodes the condition images unless the pipeline holds them at the
  // current output area; raises when an image's size cannot work.
  procedure PrepareConditionImages();
  var
    ImagePos: integer;
    Area: double;
    Problem: string;
  begin
    if not IsEditing() then exit;
    Area := double(Width) * Height;
    if (ConditionEncodedArea = Area) and
      (Pipeline.ConditionImageCount = Length(ConditionImages)) then exit;
    for ImagePos := 0 to High(ConditionImages) do
    begin
      Problem := OutputSizeProblemFor(ConditionImages[ImagePos]);
      if Problem <> '' then
        raise Exception.Create(ConditionFiles[ImagePos] + ': ' + Problem);
    end;
    ConditionEncodedArea := 0;
    Pipeline.EncodeConditionImages(ConditionImages, Area);
    Reporter.OnPhase(qppDone);
    ConditionEncodedArea := Area;
    InputImageEncodeNumber := ImageNumber + 1;
  end;

  // The prompt token ids of PromptText: the edit template while editing.
  function TokenizeFor(const PromptText: string): TNeuralIntegerArray;
  begin
    if IsEditing() then
      Result := Pipeline.TokenizeEditPrompt(PromptText,
        Length(ConditionImages), DropCount)
    else Result := Pipeline.TokenizePrompt(PromptText, DropCount);
  end;

  // Encodes the init image unless its latents already match the output size;
  // raises when the size or the strength cannot work.
  procedure PrepareInitLatents();
  var
    OutputWidth, OutputHeight: integer;
    Problem: string;
  begin
    if not HasInitImage then exit;
    if not IsOutpainting() then
    begin
      Problem := OutputSizeProblemFor(InitImage);
      if Problem <> '' then raise Exception.Create(Problem);
    end;
    // Raises when the strength leaves no step.
    TNNetFlowMatchEulerScheduler.Img2ImgStartStep(StepCount, Strength);
    GetOutputSize(OutputWidth, OutputHeight);
    if IsCropInpainting() then
    begin
      BuildInpaintCrop(OutputWidth, OutputHeight);
      // The crop mask also follows the init image; it is cheap to encode.
      MaskLatentsWidth := 0;
    end
    else InpaintCrop := Rect(0, 0, 0, 0);
    if not EqualRect(InpaintCrop, ImageLatentsCrop) then
    begin
      ImageLatentsWidth := 0;
      MaskLatentsWidth := 0;
      ImageLatentsCrop := InpaintCrop;
    end;
    if IsOutpainting() and ((MaskLatentsWidth <> OutputWidth) or
      (MaskLatentsHeight <> OutputHeight)) then
    begin
      MaskLatentsWidth := 0;
      Pipeline.EncodeMask(OutpaintMask, OutputWidth, OutputHeight,
        MaskLatents);
      MaskLatentsWidth := OutputWidth;
      MaskLatentsHeight := OutputHeight;
    end
    else if IsInpainting() and ((MaskLatentsWidth <> OutputWidth) or
      (MaskLatentsHeight <> OutputHeight)) then
    begin
      MaskLatentsWidth := 0;
      if IsCropInpainting() then
        Pipeline.EncodeMask(InpaintCropMask, OutputWidth, OutputHeight,
          MaskLatents)
      else
        Pipeline.EncodeMask(MaskImage, OutputWidth, OutputHeight, MaskLatents);
      MaskLatentsWidth := OutputWidth;
      MaskLatentsHeight := OutputHeight;
    end;
    if (ImageLatentsWidth = OutputWidth) and
      (ImageLatentsHeight = OutputHeight) then exit;
    ImageLatentsWidth := 0;
    if IsOutpainting() then
      Pipeline.EncodeImage(OutpaintCanvas, OutputWidth, OutputHeight,
        ImageLatents)
    else if IsCropInpainting() then
      Pipeline.EncodeImage(InpaintCropImage, OutputWidth, OutputHeight,
        ImageLatents)
    else
      Pipeline.EncodeImage(InitImage, OutputWidth, OutputHeight,
        ImageLatents);
    Reporter.OnPhase(qppDone);
    ImageLatentsWidth := OutputWidth;
    ImageLatentsHeight := OutputHeight;
    InputImageEncodeNumber := ImageNumber + 1;
  end;

  // The negative prompt's embeds (same template and condition images as the
  // prompt) while guidance is on; cleared otherwise.
  procedure EncodeNegativePrompt();
  var
    NegativeIds, NegativeTextLengths: TNeuralIntegerArray;
    NegativeDropCount: integer;
  begin
    if not IsGuided() then
    begin
      Pipeline.ClearNegativePrompt();
      exit;
    end;
    NegativeTextLengths := nil;
    if IsEditing() then
    begin
      NegativeIds := Pipeline.TokenizeEditPrompt(NegativePrompt,
        Length(ConditionImages), NegativeDropCount);
      Pipeline.EncodeEditTokenIds(NegativeIds, NegativeDropCount,
        NegativeEmbeds, NegativeTextLengths);
    end
    else
    begin
      NegativeIds := Pipeline.TokenizePrompt(NegativePrompt,
        NegativeDropCount);
      Pipeline.EncodeTokenIds(NegativeIds, NegativeDropCount, NegativeEmbeds);
    end;
    Pipeline.SetNegativePromptEmbeds(NegativeEmbeds, NegativeTextLengths);
  end;

  // Encodes the input images when needed (condition images or the init
  // image), then TokenIds into PromptEmbeds; ends the encode phase line.
  procedure EncodePrompt();
  begin
    IsPeakPerImage := (UseStats or UseProfile) and ResetPeakRss();
    PrepareConditionImages();
    PrepareInitLatents();
    // First, so the stage table's encode times are the prompt's.
    EncodeNegativePrompt();
    if IsEditing() then
      Pipeline.EncodeEditTokenIds(TokenIds, DropCount, PromptEmbeds,
        TextLengths)
    else Pipeline.EncodeTokenIds(TokenIds, DropCount, PromptEmbeds);
    Reporter.OnPhase(qppDone);
    EncodeImageNumber := ImageNumber + 1;
  end;

  // The Image and Init image lines of the run header.
  // Replaces the upscaler with FileName's (config read now, weights per
  // image); returns '' or what is wrong, keeping the previous upscaler then.
  function SetUpscaler(const FileName: string): string;
  var
    NewUpscaler: TRRDBNetUpscaler;
  begin
    Result := '';
    try
      NewUpscaler := TRRDBNetUpscaler.Create(FileName);
    except
      on E: Exception do exit(E.Message);
    end;
    NewUpscaler.Tile := UpscaleTile;
    {$IFDEF OpenCL}
    if IsOpenCLSelected then
      NewUpscaler.EnableOpenCL(OpenCLDevices.PlatformIds[OpenCLPlatform],
        OpenCLDevices.Devices[OpenCLDevice], HasSharedKernel);
    {$ENDIF}
    Upscaler.Free;
    Upscaler := NewUpscaler;
  end;

  function UpscalerText(): string;
  begin
    Result := Format('%s (x%d, tile %d)', [Upscaler.FileName,
      Upscaler.Config.Scale, Upscaler.Tile]);
  end;

  function PanoramaStrideText(): string;
  begin
    if PanoramaStride > 0 then Result := IntToStr(PanoramaStride) + ' pixels'
    else Result := 'half a window';
  end;

  // Arms the pipeline's MultiDiffusion windows while IsPanorama(), else
  // turns them off.
  procedure ApplyPanorama();
  begin
    if IsPanorama() then
    begin
      Pipeline.PanoramaWindowWidth := Width;
      Pipeline.PanoramaWindowHeight := Height;
    end
    else
    begin
      Pipeline.PanoramaWindowWidth := 0;
      Pipeline.PanoramaWindowHeight := 0;
    end;
    Pipeline.PanoramaStride := PanoramaStride;
    Pipeline.PanoramaCircular := PanoramaWrap;
  end;

  procedure PrintImageSettings();
  var
    OutputWidth, OutputHeight, StartStep: integer;
  begin
    GetOutputSize(OutputWidth, OutputHeight);
    WriteLn('Image      : ', OutputWidth, 'x', OutputHeight, ' (',
      (OutputWidth div 16) * (OutputHeight div 16), ' image tokens), ',
      StepCount, ' steps, seed ', Seed);
    if IsPanorama() then
      WriteLn('Panorama   : ', Width, 'x', Height, ' windows every ',
        PanoramaStrideText(), BoolToStr(PanoramaWrap, ', wrapping in X', ''))
    else if PanoramaWidth > 0 then
      WriteLn('Panorama   : off while an init or condition image is set');
    if IsEditing() then
    begin
      WriteLn('Edit       : ', Length(ConditionImages), ' condition ',
        'image(s); the output takes the last one''s aspect');
      ListConditionImages();
    end;
    if not HasInitImage then exit;
    // Raises when the strength leaves no step.
    StartStep := TNNetFlowMatchEulerScheduler.Img2ImgStartStep(StepCount,
      Strength);
    WriteLn('Init image : ', InitImageFile, ' (', InitImage.SizeX, 'x',
      InitImage.SizeY, '), strength ', StrengthText(Strength), ', the last ',
      StepCount - StartStep, ' of ', StepCount, ' steps');
    if IsOutpainting() then
    begin
      WriteLn('Outpaint   : ', OutpaintLeft, ',', OutpaintTop, ',',
        OutpaintRight, ',', OutpaintBottom, ' pixels, overlap ',
        OutpaintOverlap, '; canvas ', OutpaintCanvas.SizeX, 'x',
        OutpaintCanvas.SizeY, ' cropped to ', InitImage.SizeX + OutpaintLeft +
        OutpaintRight, 'x', InitImage.SizeY + OutpaintTop + OutpaintBottom);
      if MaskBlurRadius > 0 then
        WriteLn('Mask blur  : Gaussian sigma ', FloatToStr(MaskBlurRadius,
          PointFormat()), ' pixels');
    end;
    if IsInpainting() then
    begin
      WriteLn('Mask       : ', MaskFile, ' (', MaskImage.SizeX, 'x',
        MaskImage.SizeY, '), only its white region is regenerated');
      if IsCropInpainting() then
        WriteLn('Mask crop  : its box plus ', PaddingMaskCrop, ' pixels, ',
          'grown to the output aspect, is generated at ', OutputWidth, 'x',
          OutputHeight, ' and pasted into the ', InitImage.SizeX, 'x',
          InitImage.SizeY, ' init image');
      if MaskBlurRadius > 0 then
        WriteLn('Mask blur  : Gaussian sigma ', FloatToStr(MaskBlurRadius,
          PointFormat()), ' pixels');
    end;
  end;

  // Profiling counts transfers too (TNNet.LayerProfiling turns the counting
  // on, but only once a pass runs).
  procedure SetProfile(Value: boolean);
  begin
    UseProfile := Value;
    Pipeline.LayerProfiling := Value;
    {$IFDEF OpenCL}
    if Value then OpenCLTransferCounting := true;
    {$ENDIF}
  end;

  // Transfer counting stays on after /stats off: its cost per transfer is
  // four integer adds, two of them atomic.
  procedure SetStats(Value: boolean);
  begin
    UseStats := Value;
    {$IFDEF OpenCL}
    if Value then OpenCLTransferCounting := true;
    {$ENDIF}
  end;

  procedure AdvanceSeed();
  begin
    // The seed wraps from High(cardinal) to 0.
    {$PUSH}{$Q-}{$R-}
    Inc(Seed);
    {$POP}
  end;

  // Generates from PromptEmbeds at the current settings, writes ImageFile.
  procedure GenerateImage(const ImageFile: string);
  var
    ImageStart: QWord;
    SaveStart, UpscaleStart: TDateTime;
    PngSaveMs, UpscaleMs: double;
    OutputWidth, OutputHeight, GeneratedWidth, GeneratedHeight: integer;
    UpscaleRanOn, UpscaleDetail: string;
    TileConfig: TRRDBNetConfig;
  begin
    // The image that encoded the prompt had its mark reset before the encode.
    if ImageNumber <> EncodeImageNumber then
      IsPeakPerImage := (UseStats or UseProfile) and ResetPeakRss();
    ImageStart := GetTickCount64;
    GetOutputSize(OutputWidth, OutputHeight);
    ApplyPanorama();
    if HasInitImage then
      Pipeline.GenerateFromEmbeds(PromptEmbeds, OutputWidth, OutputHeight,
        StepCount, Seed, Image, nil, ImageLatents, Strength,
        ActiveMaskLatents())
    else if IsEditing() then
      Pipeline.GenerateEditFromEmbeds(PromptEmbeds, TextLengths, OutputWidth,
        OutputHeight, StepCount, Seed, Image)
    else
      Pipeline.GenerateFromEmbeds(PromptEmbeds, OutputWidth, OutputHeight,
        StepCount, Seed, Image);
    SaveStart := Now();
    Image.Mul(255);
    if IsOutpainting() then FinishOutpaint()
    else if IsCropInpainting() then
      PasteInpaintCrop(Image, InitImage, InpaintFullMask, InpaintCrop.Left,
        InpaintCrop.Top, InpaintCrop.Right, InpaintCrop.Bottom, irfLanczos,
        Image);
    UpscaleMs := 0;
    UpscaleRanOn := '';
    UpscaleDetail := '';
    if Assigned(Upscaler) then
    begin
      UpscaleStart := Now();
      GeneratedWidth := Image.SizeX;
      GeneratedHeight := Image.SizeY;
      TileConfig := Upscaler.TileConfigFor(GeneratedWidth, GeneratedHeight);
      Image.Divi(255);
      Upscaler.Upscale(Image, Image);
      Image.Mul(255);
      UpscaleMs := (Now() - UpscaleStart) * MSecsPerDay;
      if Upscaler.OnOpenCL then UpscaleRanOn := 'OpenCL'
      else UpscaleRanOn := 'CPU';
      UpscaleDetail := Format('x%d to %dx%d, net input %dx%d',
        [Upscaler.Config.Scale, Image.SizeX, Image.SizeY,
         TileConfig.InputWidth, TileConfig.InputHeight]);
      WriteLn('Upscaled ', GeneratedWidth, 'x', GeneratedHeight, ' to ',
        Image.SizeX, 'x', Image.SizeY, ' on ', UpscaleRanOn, ' in ',
        (UpscaleMs / 1000):0:1, ' s');
    end;
    if not SaveImageFromVolumeIntoFile(Image, ImageFile) then
      raise Exception.Create('could not write ' + ImageFile);
    PngSaveMs := (Now() - SaveStart) * MSecsPerDay - UpscaleMs;
    WriteLn('Wrote ', ImageFile, ' (', Image.SizeX, 'x', Image.SizeY, 'x',
      Image.Depth, ') in ', ((GetTickCount64 - ImageStart) / 1000):0:1,
      ' s; ', MemoryReport());
    if UseStats or UseProfile then
      PrintStageTable(Pipeline.ImageStats, ImageNumber, EncodeImageNumber,
        InputImageEncodeNumber, UpscaleMs, UpscaleRanOn, UpscaleDetail,
        PngSaveMs, UseProfile, IsPeakPerImage);
    if UseProfile then
    begin
      WriteLn;
      Write(Pipeline.TransformerProfileReport);
      Write(Pipeline.VaeProfileReport);
    end;
  end;

  // Loads the three components to keep them and prints the resident memory.
  procedure LoadAllComponents();
  begin
    WriteLn('Loading the text encoder, the transformer and the VAE decoder ',
      'to keep them in memory.');
    Pipeline.LoadComponents();
    WriteLn('Resident   : ', MemoryReport());
  end;

  // One image at Seed to the next numbered --output file (Numbered) or to
  // --output; the seed then grows by one, also when the image fails.
  procedure GenerateNextImage(Numbered: boolean);
  var
    ImageFile, StepText, ModeText: string;
    OutputWidth, OutputHeight: integer;
  begin
    Inc(ImageNumber);
    if Numbered then ImageFile := NextFreeOutputFile(OutputFile, FileNumber)
    else ImageFile := OutputFile;
    GetOutputSize(OutputWidth, OutputHeight);
    StepText := IntToStr(StepCount) + ' steps';
    if IsOutpainting() then ModeText := 'outpaint'
    else if IsCropInpainting() then ModeText := 'inpaint (mask crop)'
    else if IsInpainting() then ModeText := 'inpaint'
    else ModeText := 'img2img';
    if HasInitImage then
      StepText := Format('%s strength %s, the last %d of %d steps',
        [ModeText, StrengthText(Strength), StepCount -
        TNNetFlowMatchEulerScheduler.Img2ImgStartStep(StepCount, Strength),
        StepCount])
    else if IsEditing() then
      StepText := Format('edit of %d image(s), %d steps',
        [Length(ConditionImages), StepCount]);
    WriteLn('Image ', ImageNumber, ': ', OutputWidth, 'x', OutputHeight, ', ',
      StepText, ', seed ', Seed, ' -> ', ImageFile);
    try
      GenerateImage(ImageFile);
    finally
      AdvanceSeed();
    end;
  end;

  // Ends the session after an allocation failure; returns false.
  function EndSessionOutOfMemory(const Message: string): boolean;
  begin
    Reporter.EndStepLine();
    WriteLn('Error: ', Message, ' - out of memory, ending the session.');
    ExitCode := 1;
    Result := false;
  end;

  // ImageCount images of PromptText (encoded once) with consecutive seeds to
  // numbered files; an image that raises is skipped. False after out of memory.
  function GenerateReplImages(const PromptText: string;
    ImageCount: integer): boolean;
  var
    ImageIndex: integer;
  begin
    Result := true;
    WriteLn('Prompt     : "', ShortPrompt(PromptText, 60), '"');
    // The stages free the per-image passes on the way out; the loaded
    // weights stay valid, except after an allocation failure.
    try
      TokenIds := TokenizeFor(PromptText);
      EncodePrompt();
    except
      on E: EOutOfMemory do exit(EndSessionOutOfMemory(E.Message));
      on E: Exception do
      begin
        Reporter.OnPhase(qppDone);
        WriteLn('Error: ', E.Message, ' - prompt skipped.');
        exit;
      end;
    end;
    for ImageIndex := 1 to ImageCount do
    try
      GenerateNextImage({Numbered=}true);
    except
      on E: EOutOfMemory do exit(EndSessionOutOfMemory(E.Message));
      on E: Exception do
      begin
        Reporter.OnPhase(qppDone);
        WriteLn('Error: ', E.Message, ' - image ', ImageNumber, ' skipped.');
      end;
    end;
  end;

  // --output, or RepeatCount numbered files with consecutive seeds; an error
  // ends the run.
  procedure GenerateOneShotImages();
  var
    ImageIndex: integer;
  begin
    if RepeatCount > 1 then LoadAllComponents();
    EncodePrompt();
    for ImageIndex := 1 to RepeatCount do
    try
      GenerateNextImage({Numbered=}RepeatCount > 1);
    except
      if RepeatCount > 1 then
      begin
        Reporter.EndStepLine();
        WriteLn(ImageIndex - 1, ' of ', RepeatCount, ' images written.');
      end;
      raise;
    end;
  end;

  // Parses on/off for /Command; false (and prints why) otherwise.
  function ParseOnOff(const Command, Argument: string;
    out Value: boolean): boolean;
  begin
    Result := true;
    if LowerCase(Argument) = 'on' then Value := true
    else if LowerCase(Argument) = 'off' then Value := false
    else
    begin
      WriteLn('[/', Command, ': expected on or off]');
      Result := false;
    end;
  end;

  // /size, /steps, /seed, /tile, /repeat, /image, /images, /init-image, /mask,
  // /mask-blur, /outpaint, /strength, /upscale, /fbc, /panorama,
  // /negative-prompt, /guidance-scale, /pag, /profile, /stats, /quit;
  // false on /quit or when /repeat ran out of memory.
  function RunReplCommand(const Line: string): boolean;
  var
    IsOn: boolean;
    SpacePos, NewWidth, NewHeight, NewTileSize, NewTileStride: integer;
    NewRepeatCount: integer;
    NewValue: int64;
    NewStrength, NewMaskBlurRadius, NewFbcThreshold, NewGuidanceScale,
      NewPagScale: double;
    NewPaddingMaskCrop: integer;
    NewLeft, NewTop, NewRight, NewBottom, NewOverlap: integer;
    Command, Argument, Problem: string;
  begin
    Result := true;
    SpacePos := Pos(' ', Line);
    if SpacePos = 0 then
    begin
      Command := LowerCase(Copy(Line, 2, MaxInt));
      Argument := '';
    end
    else
    begin
      Command := LowerCase(Copy(Line, 2, SpacePos - 2));
      Argument := Trim(Copy(Line, SpacePos + 1, MaxInt));
    end;
    if Command = 'quit' then exit(false)
    else if Command = 'size' then
    begin
      Problem := ParseImageSize(Argument, NewWidth, NewHeight);
      if Problem <> '' then WriteLn('[/size: ', Problem, ']')
      else
      begin
        Width := NewWidth;
        Height := NewHeight;
        if IsOutpainting() then
          WriteLn('[size ', Width, 'x', Height, '; outpainting keeps its ',
            OutpaintCanvas.SizeX, 'x', OutpaintCanvas.SizeY, ' canvas]')
        else if HasInitImage or IsEditing() then
        begin
          GetOutputSize(NewWidth, NewHeight);
          WriteLn('[size ', Width, 'x', Height, '; the image''s aspect ',
            'gives ', NewWidth, 'x', NewHeight, ']');
        end
        else WriteLn('[size ', Width, 'x', Height, ']');
      end;
    end
    else if Command = 'steps' then
    begin
      NewValue := StrToInt64Def(Argument, 0);
      if (NewValue < 1) or (NewValue > MaxInt) then
        WriteLn('[/steps: "', Argument, '" is not a positive integer]')
      else
      begin
        StepCount := NewValue;
        WriteLn('[steps ', StepCount, ']');
      end;
    end
    else if Command = 'seed' then
    begin
      NewValue := StrToInt64Def(Argument, -1);
      if (NewValue < 0) or (NewValue > High(cardinal)) then
        WriteLn('[/seed: "', Argument, '" is not an integer in 0..',
          High(cardinal), ']')
      else
      begin
        Seed := NewValue;
        WriteLn('[seed ', Seed, ' for the next image]');
      end;
    end
    else if Command = 'tile' then
    begin
      Problem := ParseVaeTile(Argument, NewTileSize, NewTileStride);
      if Problem <> '' then WriteLn('[/tile: ', Problem, ']')
      else
      begin
        Pipeline.VaeTileSize := NewTileSize;
        Pipeline.VaeTileStride := NewTileStride;
        WriteLn('[VAE tile ', NewTileSize, ' px every ', NewTileStride,
          ' px]');
      end;
    end
    else if Command = 'repeat' then
    begin
      SpacePos := Pos(' ', Argument);
      if SpacePos = 0 then
        WriteLn('[/repeat: expected /repeat N PROMPT]')
      else
      begin
        Problem := ParseRepeatCount(Copy(Argument, 1, SpacePos - 1),
          NewRepeatCount);
        if Problem <> '' then WriteLn('[/repeat: ', Problem, ']')
        else
          Result := GenerateReplImages(Trim(Copy(Argument, SpacePos + 1,
            MaxInt)), NewRepeatCount);
      end;
    end
    else if Command = 'image' then
    begin
      if Argument = '' then
        WriteLn('[/image: expected /image FILE]')
      else if LowerCase(Argument) = 'off' then
        WriteLn('[/image off: /images clear removes the condition images, ',
          '/init-image off the img2img init image]')
      else if HasInitImage then
        WriteLn('[/image: an init image (img2img) is set; /init-image off ',
          'first]')
      else
      begin
        Problem := '';
        if not IsEditing() then
        try
          Pipeline.CheckEditSupport();
        except
          on E: Exception do Problem := E.Message;
        end;
        if Problem = '' then Problem := AddConditionImage(Argument);
        if Problem <> '' then WriteLn('[/image: ', Problem, ']')
        else
        begin
          GetOutputSize(NewWidth, NewHeight);
          WriteLn('[condition image ', Length(ConditionImages), ': ',
            Argument, ' (', ConditionImages[High(ConditionImages)].SizeX, 'x',
            ConditionImages[High(ConditionImages)].SizeY, '); output ',
            NewWidth, 'x', NewHeight, ' from the last image]');
        end;
      end;
    end
    else if Command = 'images' then
    begin
      if LowerCase(Argument) = 'clear' then
      begin
        ClearConditionImages();
        Pipeline.UnloadImageEncoders();
        WriteLn('[condition images cleared, vision tower and VAE encoder ',
          'freed: text-to-image]');
      end
      else if Argument <> '' then
        WriteLn('[/images: expected /images or /images clear]')
      else if not IsEditing() then WriteLn('[no condition images]')
      else ListConditionImages();
    end
    else if Command = 'init-image' then
    begin
      if Argument = '' then
        WriteLn('[/init-image: expected /init-image FILE or /init-image off]')
      else if LowerCase(Argument) = 'off' then
      begin
        HasInitImage := false;
        ImageLatentsWidth := 0;
        WriteLn('[init image off: text-to-image]');
      end
      else if IsEditing() then
        WriteLn('[/init-image: condition images (edit) are set; /images ',
          'clear first]')
      else
      begin
        Problem := LoadInitImage(Argument);
        if Problem <> '' then
        begin
          if HasInitImage then
            WriteLn('[/init-image: ', Problem, ' - the init image ',
              InitImageFile, ' stays active]')
          else WriteLn('[/init-image: ', Problem, ' - no init image is set]');
        end
        else
        begin
          GetOutputSize(NewWidth, NewHeight);
          WriteLn('[init image ', InitImageFile, ' (', InitImage.SizeX, 'x',
            InitImage.SizeY, '), output ', NewWidth, 'x', NewHeight,
            ', strength ', StrengthText(Strength), ']');
        end;
      end;
    end
    else if Command = 'mask' then
    begin
      if Argument = '' then
        WriteLn('[/mask: expected /mask FILE or /mask off]')
      else if LowerCase(Argument) = 'off' then
      begin
        HasMask := false;
        MaskLatentsWidth := 0;
        if HasInitImage then WriteLn('[mask off: img2img]')
        else WriteLn('[mask off]');
      end
      else if HasOutpaint then
        WriteLn('[/mask: outpainting is on; /outpaint off first]')
      else
      begin
        Problem := LoadMask(Argument);
        if Problem <> '' then
        begin
          if HasMask then
            WriteLn('[/mask: ', Problem, ' - the mask ', MaskFile,
              ' stays active]')
          else WriteLn('[/mask: ', Problem, ' - no mask is set]');
        end
        else if HasInitImage then
          WriteLn('[mask ', MaskFile, ' (', MaskImage.SizeX, 'x',
            MaskImage.SizeY, '): inpainting ', InitImageFile, ']')
        else WriteLn('[mask ', MaskFile, ' (', MaskImage.SizeX, 'x',
          MaskImage.SizeY, '): applies once /init-image sets an init image]');
      end;
    end
    else if Command = 'mask-blur' then
    begin
      Problem := ParseMaskBlurRadius(Argument, NewMaskBlurRadius);
      if Problem <> '' then WriteLn('[/mask-blur: ', Problem, ']')
      else
      begin
        MaskBlurRadius := NewMaskBlurRadius;
        if HasOutpaint then
        begin
          BuildOutpaint();
          WriteLn('[mask blur ', FloatToStr(MaskBlurRadius, PointFormat()),
            ' for the outpaint mask]');
        end
        else if HasMask then
        begin
          Problem := LoadMask(MaskFile);
          if Problem <> '' then
            WriteLn('[/mask-blur: ', Problem, ' - the mask keeps its ',
              'previous blur]')
          else WriteLn('[mask blur ', FloatToStr(MaskBlurRadius,
            PointFormat()), ': ', MaskFile, ' reloaded]');
        end
        else WriteLn('[mask blur ', FloatToStr(MaskBlurRadius, PointFormat()),
          ' for the next /mask]');
      end;
    end
    else if Command = 'padding-mask-crop' then
    begin
      if LowerCase(Argument) = 'off' then
      begin
        PaddingMaskCrop := -1;
        WriteLn('[padding mask crop off: the whole image is inpainted]');
      end
      else
      begin
        Problem := ParsePixelCount(Argument, NewPaddingMaskCrop);
        if Problem <> '' then WriteLn('[/padding-mask-crop: ', Problem, ']')
        else
        begin
          PaddingMaskCrop := NewPaddingMaskCrop;
          if IsInpainting() then
            WriteLn('[padding mask crop ', PaddingMaskCrop, ' pixels]')
          else WriteLn('[padding mask crop ', PaddingMaskCrop, ' pixels: ',
            'applies once /init-image and /mask are set]');
        end;
      end;
    end
    else if Command = 'strength' then
    begin
      Problem := ParseStrength(Argument, NewStrength);
      if Problem <> '' then WriteLn('[/strength: ', Problem, ']')
      else
      begin
        Strength := NewStrength;
        HasStrengthArg := true;
        WriteLn('[strength ', StrengthText(Strength), ']');
      end;
    end
    else if Command = 'outpaint' then
    begin
      if LowerCase(Argument) = 'off' then
      begin
        HasOutpaint := false;
        BuildOutpaint();
        if not HasStrengthArg then Strength := csDefaultStrength;
        if HasInitImage then
          WriteLn('[outpaint off: img2img, strength ', StrengthText(Strength),
            ']')
        else WriteLn('[outpaint off]');
      end
      else if HasMask then
        WriteLn('[/outpaint: a mask is set; /mask off first]')
      else
      begin
        SpacePos := Pos(' ', Argument);
        NewOverlap := OutpaintOverlap;
        Problem := '';
        if SpacePos > 0 then
          Problem := ParsePixelCount(Copy(Argument, SpacePos + 1, MaxInt),
            NewOverlap)
        else SpacePos := Length(Argument) + 1;
        if Problem = '' then
          Problem := ParseOutpaintSides(Copy(Argument, 1, SpacePos - 1),
            NewLeft, NewTop, NewRight, NewBottom);
        if Problem = '' then
          Problem := SetOutpaint(NewLeft, NewTop, NewRight, NewBottom,
            NewOverlap);
        if Problem <> '' then
          WriteLn('[/outpaint: expected /outpaint L,T,R,B [OVERLAP] or ',
            '/outpaint off; ', Problem, ']')
        else
        begin
          if not HasStrengthArg then Strength := 1;
          if HasInitImage then
            WriteLn('[outpaint ', InitImageFile, ' by ', OutpaintLeft, ',',
              OutpaintTop, ',', OutpaintRight, ',', OutpaintBottom,
              ' pixels, overlap ', OutpaintOverlap, ': canvas ',
              OutpaintCanvas.SizeX, 'x', OutpaintCanvas.SizeY, ', strength ',
              StrengthText(Strength), ']')
          else WriteLn('[outpaint ', OutpaintLeft, ',', OutpaintTop, ',',
            OutpaintRight, ',', OutpaintBottom, ': applies once /init-image ',
            'sets an init image]');
        end;
      end;
    end
    else if Command = 'upscale' then
    begin
      if Argument = '' then
      begin
        if Assigned(Upscaler) then
          WriteLn('[upscale ', UpscalerText(), '; /upscale off stops it]')
        else WriteLn('[/upscale: expected /upscale FILE or /upscale off]');
      end
      else if LowerCase(Argument) = 'off' then
      begin
        FreeAndNil(Upscaler);
        WriteLn('[upscale off]');
      end
      else
      begin
        Problem := SetUpscaler(Argument);
        if Problem = '' then
          WriteLn('[upscale ', UpscalerText(), ' from the next image]')
        else if Assigned(Upscaler) then
          WriteLn('[/upscale: ', Problem, ' - ', UpscalerText(),
            ' stays active]')
        else WriteLn('[/upscale: ', Problem, ' - upscaling stays off]');
      end;
    end
    else if Command = 'panorama' then
    begin
      if Argument = '' then
      begin
        if PanoramaWidth > 0 then
          WriteLn('[panorama ', PanoramaWidth, 'x', PanoramaHeight,
            '; /panorama WxH|off changes it]')
        else WriteLn('[panorama off; /panorama WxH sets a canvas]');
      end
      else if LowerCase(Argument) = 'off' then
      begin
        PanoramaWidth := 0;
        PanoramaHeight := 0;
        WriteLn('[panorama off]');
      end
      else
      begin
        Problem := ParseImageSize(Argument, NewWidth, NewHeight);
        if Problem <> '' then WriteLn('[/panorama: ', Problem, ']')
        else
        begin
          PanoramaWidth := NewWidth;
          PanoramaHeight := NewHeight;
          WriteLn('[panorama ', PanoramaWidth, 'x', PanoramaHeight, ' in ',
            Width, 'x', Height, ' windows]');
          if not IsPanorama() then
            WriteLn('[/panorama: off while an init or condition image is ',
              'set]');
        end;
      end;
    end
    else if Command = 'fbc' then
    begin
      if Argument = '' then
        WriteLn('[', FbcThresholdText(Pipeline.FirstBlockCacheThreshold),
          '; /fbc X|off changes it]')
      else
      begin
        Problem := ParseNumberOrOff(Argument, NewFbcThreshold);
        if Problem <> '' then WriteLn('[/fbc: ', Problem, ']')
        else
        begin
          Pipeline.FirstBlockCacheThreshold := NewFbcThreshold;
          WriteLn('[', FbcThresholdText(NewFbcThreshold),
            ' from the next image]');
        end;
      end;
    end
    else if Command = 'negative-prompt' then
    begin
      if Argument = '' then
        WriteLn('[guidance ', GuidanceText(), '; /negative-prompt TEXT|off ',
          'changes it]')
      else
      begin
        HasNegativePrompt := LowerCase(Argument) <> 'off';
        if HasNegativePrompt then NegativePrompt := Argument
        else NegativePrompt := '';
        WriteLn('[guidance ', GuidanceText(), ' from the next prompt]');
      end;
    end
    else if Command = 'guidance-scale' then
    begin
      Problem := ParseGuidanceScale(Argument, NewGuidanceScale);
      if Problem <> '' then WriteLn('[/guidance-scale: ', Problem, ']')
      else
      begin
        GuidanceScale := NewGuidanceScale;
        Pipeline.GuidanceScale := GuidanceScale;
        WriteLn('[guidance ', GuidanceText(), ' from the next prompt]');
      end;
    end
    else if Command = 'pag' then
    begin
      if Argument = '' then
        WriteLn('[perturbed guidance ', PerturbedGuidanceText(),
          '; /pag W|off changes it]')
      else
      begin
        Problem := ParseNumberOrOff(Argument, NewPagScale);
        if Problem <> '' then WriteLn('[/pag: ', Problem, ']')
        else
        begin
          PagScale := NewPagScale;
          ApplyPerturbedGuidance();
          WriteLn('[perturbed guidance ', PerturbedGuidanceText(),
            ' from the next image]');
        end;
      end;
    end
    else if Command = 'profile' then
    begin
      if ParseOnOff(Command, Argument, IsOn) then
      begin
        SetProfile(IsOn);
        WriteLn('[profile ', LowerCase(Argument), ' from the next image]');
      end;
    end
    else if Command = 'stats' then
    begin
      if ParseOnOff(Command, Argument, IsOn) then
      begin
        SetStats(IsOn);
        WriteLn('[stats ', LowerCase(Argument), ' from the next image]');
      end;
    end
    else WriteLn('[unknown command /', Command, ' - /size, /steps, /seed, ',
      '/tile, /repeat, /image, /images, /init-image, /mask, /mask-blur, ',
      '/outpaint, /strength, /upscale, /fbc, /panorama, /negative-prompt, ',
      '/guidance-scale, /pag, /profile, /stats, /quit]');
  end;

  // One prompt per stdin line until /quit or the end of the input.
  procedure RunRepl();
  var
    Line: string;
    InteractiveInput: boolean;
  begin
    LoadAllComponents();
    PrintComputeResult();
    WriteLn('Type a prompt per line; /size WxH, /steps N, /seed N, ',
      '/tile SIZE[,STRIDE], /repeat N PROMPT, /image FILE, /images [clear], ',
      '/init-image FILE|off, /mask FILE|off, /mask-blur R, ',
      '/padding-mask-crop N|off, /outpaint L,T,R,B [OVERLAP]|off, /strength S, ',
      '/upscale FILE|off, /fbc X|off, /negative-prompt TEXT|off, ',
      '/guidance-scale W, /pag W|off, /profile on|off, /stats on|off, /quit.');
    // A piped batch gets no '> ' markers, so its log has one line per event.
    InteractiveInput :=
      {$IFDEF UNIX}IsATTY(StdInputHandle) = 1{$ELSE}true{$ENDIF};
    while true do
    begin
      if InteractiveInput then
      begin
        Write('> ');
        Flush(System.Output);
      end;
      if EOF(System.Input) then
      begin
        if InteractiveInput then WriteLn;
        break;
      end;
      ReadLn(Line);
      Line := Trim(Line);
      if Line = '' then continue;
      if Line[1] = '/' then
      begin
        if RunReplCommand(Line) then continue else break;
      end;
      if not GenerateReplImages(Line, 1) then break;
    end;
    WriteLn('Bye.');
  end;

begin
  ModelFolder := '';
  Prompt := '';
  HasPrompt := false;
  OutputFile := 'qwenimage.png';
  TokenList := '';
  Width := 1024;
  Height := 1024;
  StepCount := 18;
  Seed := 42;
  DropCount := 0;
  RepeatCount := 1;
  ImageNumber := 0;
  FileNumber := 0;
  WeightFormat := qiwInt8;
  UseInt8Input := false;
  UseSerial := false;
  UseProfile := false;
  UseStats := false;
  EncodeImageNumber := 0;
  IsPeakPerImage := false;
  HasInitImage := false;
  InitImageFile := '';
  ImageLatentsWidth := 0;
  ImageLatentsHeight := 0;
  HasMask := false;
  MaskFile := '';
  MaskBlurRadius := 0;
  MaskLatentsWidth := 0;
  PaddingMaskCrop := -1;
  InpaintCrop := Rect(0, 0, 0, 0);
  ImageLatentsCrop := Rect(0, 0, 0, 0);
  MaskLatentsHeight := 0;
  HasOutpaint := false;
  OutpaintText := '';
  OutpaintOverlap := csDefaultOutpaintOverlap;
  InputImageEncodeNumber := 0;
  ConditionFiles := TStringList.Create;
  ImageArgFiles := TStringList.Create;
  LoRAFiles := TStringList.Create;
  LoRAScales := nil;
  ConditionImages := nil;
  ConditionEncodedArea := 0;
  Pipeline := nil;
  Strength := csDefaultStrength;
  HasStrengthArg := false;
  FbcThreshold := 0;
  PanoramaWidth := 0;
  PanoramaHeight := 0;
  PanoramaStride := 0;
  PanoramaWrap := false;
  Sampler := fsEuler;
  NegativePrompt := '';
  HasNegativePrompt := false;
  GuidanceScale := 1;
  ApgSpec := '';
  ApgOnVelocity := false;
  PagScale := 0;
  PagStart := csDefaultPagStart;
  PagStop := csDefaultPagStop;
  PagLayers := nil;
  IsSkipLayerGuidance := false;
  HasPagArg := false;
  HasPagScaleArg := false;
  MaxThreads := 0;
  Upscaler := nil;
  UpscaleFile := '';
  UpscaleTile := csDefaultUpscaleTile;
  IsOpenCLSelected := false;
  VaeTileSize := 256;
  VaeTileStride := 192;
  UseOpenCL := {$IFDEF OpenCL}true{$ELSE}false{$ENDIF};
  OpenCLPlatform := 0;
  OpenCLDevice := 0;
  HasSharedKernel := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--model' then ModelFolder := NextArg()
    else if Arg = '-p' then
    begin
      Prompt := NextArg();
      HasPrompt := true;
    end
    else if Arg = '--output' then OutputFile := NextArg()
    else if Arg = '--width' then Width := StrToIntDef(NextArg(), -1)
    else if Arg = '--height' then Height := StrToIntDef(NextArg(), -1)
    else if Arg = '--steps' then StepCount := StrToInt(NextArg())
    else if Arg = '--seed' then Seed := StrToInt64(NextArg())
    else if Arg = '--repeat' then
    begin
      ArgProblem := ParseRepeatCount(NextArg(), RepeatCount);
      if ArgProblem <> '' then
      begin
        WriteLn('--repeat: ', ArgProblem);
        Halt(2);
      end;
    end
    else if Arg = '--image' then
    begin
      if ImageArgFiles.Count >= csQwenImage21MaxConditionImages then
      begin
        WriteLn('--image: an edit takes at most ',
          csQwenImage21MaxConditionImages, ' images.');
        Halt(2);
      end;
      ImageArgFiles.Add(NextArg());
    end
    else if Arg = '--init-image' then InitImageFile := NextArg()
    else if Arg = '--mask' then MaskFile := NextArg()
    else if Arg = '--mask-blur' then
    begin
      ArgProblem := ParseMaskBlurRadius(NextArg(), MaskBlurRadius);
      if ArgProblem <> '' then
      begin
        WriteLn('--mask-blur: ', ArgProblem);
        Halt(2);
      end;
    end
    else if Arg = '--padding-mask-crop' then
    begin
      ArgProblem := ParsePixelCount(NextArg(), PaddingMaskCrop);
      if ArgProblem <> '' then
      begin
        WriteLn('--padding-mask-crop: ', ArgProblem);
        Halt(2);
      end;
    end
    else if Arg = '--outpaint' then OutpaintText := NextArg()
    else if Arg = '--outpaint-overlap' then
    begin
      ArgProblem := ParsePixelCount(NextArg(), OutpaintOverlap);
      if ArgProblem <> '' then
      begin
        WriteLn('--outpaint-overlap: ', ArgProblem);
        Halt(2);
      end;
    end
    else if Arg = '--strength' then
    begin
      HasStrengthArg := true;
      ArgProblem := ParseStrength(NextArg(), Strength);
      if ArgProblem <> '' then
      begin
        WriteLn('--strength: ', ArgProblem);
        Halt(2);
      end;
    end
    else if Arg = '--int8' then WeightFormat := qiwInt8
    else if Arg = '--int4' then WeightFormat := qiwInt4
    else if Arg = '--fp32' then WeightFormat := qiwFP32
    else if Arg = '--int8-input' then UseInt8Input := true
    else if Arg = '--lora' then
    begin
      ArgProblem := ParseLoRAArg(NextArg(), LoRAFile, LoRAScale);
      if ArgProblem <> '' then
      begin
        WriteLn('--lora: ', ArgProblem);
        Halt(2);
      end;
      LoRAFiles.Add(LoRAFile);
      SetLength(LoRAScales, Length(LoRAScales) + 1);
      LoRAScales[High(LoRAScales)] := LoRAScale;
    end
    else if Arg = '--serial' then UseSerial := true
    else if Arg = '--max-threads' then MaxThreads := StrToInt(NextArg())
    else if Arg = '--fbc-threshold' then
    begin
      ArgProblem := ParseNumberOrOff(NextArg(), FbcThreshold);
      if ArgProblem <> '' then
      begin
        WriteLn('--fbc-threshold: ', ArgProblem);
        Halt(2);
      end;
    end
    else if Arg = '--panorama' then
    begin
      ArgProblem := ParseImageSize(NextArg(), PanoramaWidth, PanoramaHeight);
      if ArgProblem <> '' then
      begin
        WriteLn('--panorama: ', ArgProblem);
        Halt(2);
      end;
    end
    else if Arg = '--panorama-stride' then
    begin
      ArgProblem := ParsePanoramaStride(NextArg(), PanoramaStride);
      if ArgProblem <> '' then
      begin
        WriteLn('--panorama-stride: ', ArgProblem);
        Halt(2);
      end;
    end
    else if Arg = '--panorama-wrap' then PanoramaWrap := true
    else if Arg = '--sampler' then
    begin
      SamplerName := NextArg();
      if SamplerName = 'euler' then Sampler := fsEuler
      else if SamplerName = 'dpm2m' then Sampler := fsDPMSolverPP2M
      else
      begin
        WriteLn('--sampler: "', SamplerName, '" is not euler or dpm2m.');
        Halt(2);
      end;
    end
    else if Arg = '--negative-prompt' then
    begin
      NegativePrompt := NextArg();
      HasNegativePrompt := true;
    end
    else if Arg = '--guidance-scale' then
    begin
      ArgProblem := ParseGuidanceScale(NextArg(), GuidanceScale);
      if ArgProblem <> '' then
      begin
        WriteLn('--guidance-scale: ', ArgProblem);
        Halt(2);
      end;
    end
    else if Arg = '--apg' then
    begin
      ApgSpec := NextArg();
      try
        TNNetAdaptiveProjectedGuidance.CreateFromSpec(ApgSpec).Free;
      except
        on E: EConvertError do
        begin
          WriteLn('--apg: ', E.Message);
          Halt(2);
        end;
      end;
    end
    else if Arg = '--apg-velocity' then ApgOnVelocity := true
    else if Arg = '--pag-scale' then
    begin
      ArgProblem := ParseNumberOrOff(NextArg(), PagScale);
      if ArgProblem <> '' then
      begin
        WriteLn('--pag-scale: ', ArgProblem);
        Halt(2);
      end;
      HasPagScaleArg := true;
    end
    else if Arg = '--pag-layers' then
    begin
      try
        PagLayers := ParseIntegerList(NextArg());
      except
        on E: EConvertError do
        begin
          WriteLn('--pag-layers: ', E.Message);
          Halt(2);
        end;
      end;
      HasPagArg := true;
    end
    else if (Arg = '--pag-start') or (Arg = '--pag-stop') then
    begin
      if not TryStrToFloat(Trim(NextArg()), PagWindowValue, PointFormat()) then
      begin
        WriteLn(Arg, ': "', ParamStr(ArgPos), '" is not a number.');
        Halt(2);
      end;
      if Arg = '--pag-start' then PagStart := PagWindowValue
      else PagStop := PagWindowValue;
      HasPagArg := true;
    end
    else if Arg = '--slg' then
    begin
      IsSkipLayerGuidance := true;
      HasPagArg := true;
    end
    else if Arg = '--token-ids' then TokenList := NextArg()
    else if Arg = '--drop-count' then DropCount := StrToInt(NextArg())
    else if Arg = '--gpu' then UseOpenCL := true
    else if Arg = '--cpu' then UseOpenCL := false
    else if Arg = '--gpu-platform' then OpenCLPlatform := StrToInt(NextArg())
    else if Arg = '--gpu-device' then OpenCLDevice := StrToInt(NextArg())
    else if Arg = '--no-gpu-shared-kernel' then HasSharedKernel := false
    else if Arg = '--profile' then UseProfile := true
    else if Arg = '--stats' then UseStats := true
    else if Arg = '--vae-tile' then
    begin
      ArgProblem := ParseVaeTile(NextArg(), VaeTileSize, VaeTileStride);
      if ArgProblem <> '' then
      begin
        WriteLn('--vae-tile: ', ArgProblem);
        Halt(2);
      end;
    end
    else if Arg = '--upscale' then UpscaleFile := NextArg()
    else if Arg = '--upscale-tile' then
    begin
      ArgProblem := ParsePixelCount(NextArg(), UpscaleTile);
      if ArgProblem <> '' then
      begin
        WriteLn('--upscale-tile: ', ArgProblem);
        Halt(2);
      end;
    end
    else if (Arg = '--help') or (Arg = '-h') then
    begin
      PrintHelp();
      Halt(0);
    end
    else
    begin
      WriteLn('Unknown argument: ', Arg, ' (see --help).');
      Halt(2);
    end;
    Inc(ArgPos);
  end;
  if ModelFolder = '' then
  begin
    PrintHelp();
    Halt(2);
  end;
  if ImageSideProblem(Width, Height) <> '' then
  begin
    WriteLn('--width/--height: ', ImageSideProblem(Width, Height));
    Halt(2);
  end;
  if HasPrompt and (TokenList <> '') then
  begin
    WriteLn('-p and --token-ids are exclusive.');
    Halt(2);
  end;
  if (RepeatCount > 1) and not (HasPrompt or (TokenList <> '')) then
  begin
    WriteLn('--repeat needs -p or --token-ids (in the REPL: /repeat N PROMPT).');
    Halt(2);
  end;
  if UseInt8Input and (WeightFormat = qiwFP32) then
  begin
    WriteLn('--int8-input needs int8 or int4 weights, not --fp32.');
    Halt(2);
  end;
  if MaxThreads < 0 then
  begin
    WriteLn('--max-threads: must be at least 1.');
    Halt(2);
  end;
  if (ImageArgFiles.Count > 0) and (InitImageFile <> '') then
  begin
    WriteLn('--image (edit) and --init-image (img2img) are exclusive.');
    Halt(2);
  end;
  if (MaskFile <> '') and (InitImageFile = '') and
    (HasPrompt or (TokenList <> '')) then
  begin
    WriteLn('--mask needs --init-image (the image it keeps).');
    Halt(2);
  end;
  if (PaddingMaskCrop >= 0) and (MaskFile = '') and
    (HasPrompt or (TokenList <> '')) then
  begin
    WriteLn('--padding-mask-crop needs --mask (the region it crops to).');
    Halt(2);
  end;
  if OutpaintText <> '' then
  begin
    ArgProblem := ParseOutpaintSides(OutpaintText, OutpaintLeft, OutpaintTop,
      OutpaintRight, OutpaintBottom);
    if ArgProblem <> '' then
    begin
      WriteLn('--outpaint: ', ArgProblem);
      Halt(2);
    end;
    if MaskFile <> '' then
    begin
      WriteLn('--mask and --outpaint are exclusive.');
      Halt(2);
    end;
    if (InitImageFile = '') and (HasPrompt or (TokenList <> '')) then
    begin
      WriteLn('--outpaint needs --init-image (the image it extends).');
      Halt(2);
    end;
    if not HasStrengthArg then Strength := 1;
  end;
  if (PanoramaWidth > 0) and ((InitImageFile <> '') or
    (ImageArgFiles.Count > 0)) then
  begin
    WriteLn('--panorama runs text-to-image only: not with --init-image or ',
      '--image.');
    Halt(2);
  end;
  if (PanoramaWidth > 0) and (PanoramaStride > 0) and
    ((PanoramaStride > Width) or (PanoramaStride > Height)) then
  begin
    WriteLn('--panorama-stride: ', PanoramaStride, ' is larger than the ',
      Width, 'x', Height, ' window.');
    Halt(2);
  end;
  if (PanoramaWidth = 0) and (PanoramaWrap or (PanoramaStride > 0)) then
  begin
    if HasPrompt or (TokenList <> '') then
      WriteLn('[--panorama-stride and --panorama-wrap have no effect ',
        'without --panorama]')
    else WriteLn('[--panorama-stride and --panorama-wrap apply once ',
      '/panorama sets a canvas]');
  end;
  if (UpscaleTile <> csDefaultUpscaleTile) and (UpscaleFile = '') then
  begin
    if HasPrompt or (TokenList <> '') then
      WriteLn('[--upscale-tile has no effect without --upscale]')
    else WriteLn('[--upscale-tile applies once /upscale sets an upscaler]');
  end;
  if HasPagArg and not HasPagScaleArg then PagScale := csDefaultPagScale;
  try
    TNNetPerturbedGuidance.Create(1, PagScale, PagStart, PagStop).Free;
  except
    on E: EArgumentException do
    begin
      WriteLn('--pag-start / --pag-stop: ', E.Message);
      Halt(2);
    end;
  end;
  if (PagScale <> 0) and IsGuided() and (ApgSpec <> '') then
  begin
    WriteLn('--pag-scale: perturbed guidance mixes with CFG, not with --apg; ',
      'drop one of the two.');
    Halt(2);
  end;
  if HasStrengthArg and (InitImageFile = '') then
  begin
    if HasPrompt or (TokenList <> '') then
      WriteLn('[--strength has no effect without --init-image]')
    else WriteLn('[--strength applies once /init-image sets an init image]');
  end;

  StartTime := GetTickCount64;
  Reporter := TQwenImageReporter.Create();
  PromptEmbeds := TNNetVolume.Create();
  NegativeEmbeds := TNNetVolume.Create();
  Image := TNNetVolume.Create();
  InitImage := TNNetVolume.Create();
  ImageLatents := TNNetVolume.Create();
  MaskImage := TNNetVolume.Create();
  InpaintFullMask := TNNetVolume.Create();
  InpaintCropImage := TNNetVolume.Create();
  InpaintCropMask := TNNetVolume.Create();
  MaskLatents := TNNetVolume.Create();
  OutpaintCanvas := TNNetVolume.Create();
  OutpaintMask := TNNetVolume.Create();
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
  try
    Pipeline := TQwenImage21Pipeline.Create(ModelFolder);
    Pipeline.TransformerFormat := WeightFormat;
    Pipeline.TextEncoderInt8 := WeightFormat <> qiwFP32;
    Pipeline.Int8Input := UseInt8Input;
    for LoRAPos := 0 to LoRAFiles.Count - 1 do
      Pipeline.AddLoRA(LoRAFiles[LoRAPos], LoRAScales[LoRAPos]);
    Pipeline.VaeTileSize := VaeTileSize;
    Pipeline.VaeTileStride := VaeTileStride;
    Pipeline.Parallel := not UseSerial;
    Pipeline.MaxThreads := MaxThreads;
    Pipeline.FirstBlockCacheThreshold := FbcThreshold;
    Pipeline.Scheduler.Sampler := Sampler;
    Pipeline.GuidanceScale := GuidanceScale;
    if ApgSpec <> '' then
      Pipeline.AdaptiveProjectedGuidance :=
        TNNetAdaptiveProjectedGuidance.CreateFromSpec(ApgSpec);
    Pipeline.ApgOnVelocity := ApgOnVelocity;
    ApplyPerturbedGuidance();
    if FbcThreshold > 0 then WriteLn('[', FbcThresholdText(FbcThreshold), ']');
    SetProfile(UseProfile);
    SetStats(UseStats);
    Pipeline.OnPhase := @Reporter.OnPhase;
    Pipeline.OnStep := @Reporter.OnStep;
    if UseOpenCL then ComputeText := 'CPU' else ComputeText := 'CPU (--cpu)';
    {$IFDEF OpenCL}
    if UseOpenCL then
    begin
      OpenCLDevices := TEasyOpenCL.Create();
      RequestedPlatform := OpenCLPlatform;
      RequestedDevice := OpenCLDevice;
      if OpenCLDevices.SelectPlatformAndDevice(OpenCLPlatform, OpenCLDevice,
        OpenCLProblem) then
      begin
        if OpenCLPlatform <> RequestedPlatform then
          WriteLn('[--gpu-platform ', RequestedPlatform, ' out of range, ',
            'using ', OpenCLPlatform, ']');
        if OpenCLDevice <> RequestedDevice then
          WriteLn('[--gpu-device ', RequestedDevice, ' out of range, using ',
            OpenCLDevice, ']');
        Pipeline.EnableOpenCL(OpenCLDevices.PlatformIds[OpenCLPlatform],
          OpenCLDevices.Devices[OpenCLDevice], HasSharedKernel);
        IsOpenCLSelected := true;
        ComputeText := 'OpenCL requested on ' +
          OpenCLDevices.PlatformNames[OpenCLPlatform] + ' / ' +
          OpenCLDevices.DeviceNames[OpenCLDevice];
        if WeightFormat <> qiwFP32 then
          ComputeText := ComputeText + ' (transformer step pass and VAE decode'
        else
          ComputeText := ComputeText + ' (VAE decode; FP32 weights keep the ' +
            'transformer step pass on the CPU';
        if not HasSharedKernel then
          ComputeText := ComputeText + ', per-layer kernels';
        ComputeText := ComputeText + '); the rest on the CPU';
      end
      else
      begin
        WriteLn('[--gpu: ', OpenCLProblem, ' - falling back to CPU]');
        ComputeText := 'CPU (' + OpenCLProblem + ')';
      end;
    end;
    {$ELSE}
    ComputeText := 'CPU (built without OpenCL)';
    UseOpenCL := false;
    {$ENDIF}
    Width := TQwenImage21Pipeline.RoundDownImageSide(Width);
    Height := TQwenImage21Pipeline.RoundDownImageSide(Height);
    if OutpaintText <> '' then
    begin
      ArgProblem := SetOutpaint(OutpaintLeft, OutpaintTop, OutpaintRight,
        OutpaintBottom, OutpaintOverlap);
      if ArgProblem <> '' then
        raise Exception.Create('--outpaint: ' + ArgProblem);
      if InitImageFile = '' then
        WriteLn('[--outpaint applies once /init-image sets an init image]');
    end;
    if InitImageFile <> '' then
    begin
      ArgProblem := LoadInitImage(InitImageFile);
      if ArgProblem <> '' then
        raise Exception.Create('--init-image: ' + ArgProblem);
    end;
    if MaskFile <> '' then
    begin
      ArgProblem := LoadMask(MaskFile);
      if ArgProblem <> '' then
        raise Exception.Create('--mask: ' + ArgProblem);
      if InitImageFile = '' then
        WriteLn('[--mask applies once /init-image sets an init image]');
    end;
    for ArgPos := 0 to ImageArgFiles.Count - 1 do
    begin
      ArgProblem := AddConditionImage(ImageArgFiles[ArgPos]);
      if ArgProblem <> '' then
        raise Exception.Create('--image: ' + ArgProblem);
    end;
    if UpscaleFile <> '' then
    begin
      ArgProblem := SetUpscaler(UpscaleFile);
      if ArgProblem <> '' then
        raise Exception.Create('--upscale: ' + ArgProblem);
    end;
    // Refuses a checkpoint that cannot edit before anything loads.
    if IsEditing() then Pipeline.CheckEditSupport();
    WriteLn('Model      : ', ModelFolder);
    PrintImageSettings();
    case WeightFormat of
      qiwInt8: WriteLn('Weights    : transformer int8, text encoder int8');
      qiwInt4: WriteLn('Weights    : transformer int4, text encoder int8');
      qiwFP32: WriteLn('Weights    : FP32 (--fp32)');
    end;
    WriteLn('Compute    : ', ComputeText);
    WriteLn('VAE tiles  : ', VaeTileSize, ' px every ', VaeTileStride, ' px');
    if Assigned(Upscaler) then WriteLn('Upscale    : ', UpscalerText());
    WriteLn('Guidance   : ', GuidanceText());
    if PagScale <> 0 then
      WriteLn('Perturbed  : ', PerturbedGuidanceText());
    for LoRAPos := 0 to LoRAFiles.Count - 1 do
      WriteLn('LoRA       : ', LoRAFiles[LoRAPos], ' x ',
        FloatToStr(LoRAScales[LoRAPos], PointFormat()));
    if UseSerial then WriteLn('Threads    : serial (single-threaded)')
    else if MaxThreads > 0 then
      WriteLn('Threads    : parallel, at most ', MaxThreads, ' workers')
    else WriteLn('Threads    : parallel, every CPU thread (',
      NeuralDefaultThreadCount, ')');
    if not (HasPrompt or (TokenList <> '')) then RunRepl()
    else
    begin
      if TokenList <> '' then
      begin
        TokenIds := ParseIntegerList(TokenList);
        WriteLn('Prompt     : ', Length(TokenIds), ' token ids, drop ',
          DropCount);
      end
      else
      begin
        WriteLn('Prompt     : ', Prompt);
        TokenIds := TokenizeFor(Prompt);
        WriteLn('Tokens     : ', Length(TokenIds), ' (', DropCount,
          ' system-prompt tokens dropped)');
      end;
      GenerateOneShotImages();
      PrintComputeResult();
      WriteLn('Total      : ', ((GetTickCount64 - StartTime) / 1000):0:1,
        ' s; ', MemoryReport());
    end;
  finally
    Reporter.EndStepLine();
    ClearConditionImages();
    Upscaler.Free;
    ImageArgFiles.Free;
    LoRAFiles.Free;
    ConditionFiles.Free;
    Pipeline.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    OutpaintMask.Free;
    OutpaintCanvas.Free;
    MaskLatents.Free;
    InpaintCropMask.Free;
    InpaintCropImage.Free;
    InpaintFullMask.Free;
    MaskImage.Free;
    ImageLatents.Free;
    InitImage.Free;
    Image.Free;
    NegativeEmbeds.Free;
    PromptEmbeds.Free;
    Reporter.Free;
  end;
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
