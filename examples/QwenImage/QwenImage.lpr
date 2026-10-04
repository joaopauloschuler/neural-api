program QwenImage;
(*
QwenImage: Qwen-Image-2.1 text-to-image from a diffusers checkpoint folder
(Qwen/Qwen-Image-2.1: model_index.json, processor/, text_encoder/,
transformer/, vae/, scheduler/), through TQwenImage21Pipeline
(neural/neuralpretrained.pas). The transformer step pass (int8/int4 weights)
and the VAE decode run on OpenCL by default; everything else runs on the CPU:

  prompt -> processor/tokenizer.json -> Qwen3-VL text encoder
    -> transformer prefix pass (text K/V, once)
    -> N flow-matching Euler steps over the image tokens
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
PROMPT with consecutive seeds, the prompt encoded once), /image FILE|off,
/strength S, /profile on|off, /stats on|off, /quit.

img2img (--image FILE or /image FILE): SDEdit. The VAE encoder (CPU,
untiled) encodes the init image once (posterior mean); each image noises
those latents to the
sigma where the schedule is cut by --strength (default 0.6) and runs only
the remaining steps (the Qwen-Image v1 img2img schedule). It re-styles the
init image; it does not follow edit instructions. The output keeps the init
image's aspect ratio at the area of --width x --height (sides multiples of
32, as diffusers' calculate_dimensions); the init image is resized with
PIL's Lanczos coefficients in float. EXIF orientation is not applied.
GPU memory: in the REPL and with --repeat N > 1 the transformer stays in
OpenCL memory while the VAE decodes, so both need room at once; /tile 128 (or --vae-tile 128) lowers the
VAE's share (~3.4 GB of layer buffers instead of ~10.7 GB at 256).

The initial noise comes from the FPC RNG (--seed), so an image is repeatable
here but not equal to a diffusers image with the same seed.

USAGE
  QwenImage --model DIR [-p TEXT | --token-ids ID,ID,... [--drop-count N]]
            [--output FILE.png]
            [--width 1024] [--height 1024] [--steps 18] [--seed 42]
            [--repeat N] [--image FILE [--strength 0.6]]
            [--int8 | --int4 | --fp32] [--int8-input] [--vae-tile SIZE[,STRIDE]]
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
  SysUtils, Classes, {$IFDEF OpenCL}neuralopencl,{$ENDIF}
  neuralvolume, neuralnetwork, neuralpretrained, neuraldatasets, neuralthread,
  neuraldiffusion;

const
  // Largest accepted image side in pixels (64x the 1024 default area).
  csMaxImageSide = 8192;
  // img2img strength when --strength is not given (diffusers' default).
  csDefaultStrength = 0.6;
  // Largest --repeat / /repeat count (catches a mistyped count).
  csMaxRepeatCount = 1000;
  csBytesPerMB = 1024.0 * 1024.0;

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
    qppLoadVaeEncoder: Result := 'load VAE encoder';
    qppEncodeImage: Result := 'encode init image';
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
// ran, PNG save, their sum; then OpenCL transfers and bytes held, peak RSS.
procedure PrintStageTable(const Stats: TQwenImage21ImageStats;
  ImageNumber, EncodeImageNumber, InitEncodeImageNumber: integer;
  PngSaveMs: double; IsProfiled, IsPeakPerImage: boolean);
var
  Phase: TQwenImage21PipelinePhase;
  TotalMs: double;
  RanOn, Detail, Rss, Peak: string;
  HasEncode, HasInitEncode, IsOnOpenCL: boolean;

  // The phase ran for this image, not for an earlier one or not at all.
  function IsCountedHere(Phase: TQwenImage21PipelinePhase): boolean;
  begin
    case Phase of
      qppLoadTextEncoder, qppEncodePrompt: Result := HasEncode;
      qppLoadVaeEncoder, qppEncodeImage: Result := HasInitEncode;
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
  HasInitEncode := Stats.UsedInitImage and
    (ImageNumber = InitEncodeImageNumber);
  IsOnOpenCL := Stats.TransformerOnOpenCL or Stats.VaeOnOpenCL;
  TotalMs := PngSaveMs;
  for Phase := qppLoadTextEncoder to qppDecode do
    if IsCountedHere(Phase) then
      TotalMs := TotalMs + Stats.PhaseMs[Phase];
  WriteLn('Stages of image ', ImageNumber, ' (wall time):');
  if IsProfiled and IsOnOpenCL then
    WriteLn('  (--profile drains OpenCL after every layer: these walls are ',
      'higher than without it)');
  AddRow('Stage', 'Ran on', 'ms', '%', '');
  for Phase := qppLoadTextEncoder to qppDecode do
  begin
    if not IsCountedHere(Phase) then
    begin
      if Phase = qppEncodePrompt then
        AddRow(PhaseName(Phase), '-', '-', '-', 'encoded once, before image ' +
          IntToStr(EncodeImageNumber) + ' (counted there)')
      else if (Phase = qppEncodeImage) and Stats.UsedInitImage then
        AddRow(PhaseName(Phase), '-', '-', '-', 'encoded once, before image ' +
          IntToStr(InitEncodeImageNumber) + ' (counted there)');
      continue;
    end;
    if (Stats.PhaseMs[Phase] = 0) and (Phase in [qppLoadTextEncoder,
      qppLoadVaeEncoder, qppLoadTransformer, qppLoadVae]) then continue;
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

procedure PrintHelp();
begin
  WriteLn('QwenImage: Qwen-Image-2.1 text-to-image (CPU; OpenCL for the ',
    'transformer step pass and the VAE decode).');
  WriteLn('  --model DIR          diffusers folder with model_index.json (required)');
  WriteLn('  -p TEXT              one-shot: generate this prompt, write ',
    '--output and exit');
  WriteLn('  --output FILE        image file; .png keeps the alpha channel ',
    '(default qwenimage.png).');
  WriteLn('                       The REPL numbers it: qwenimage_0001.png, ',
    'qwenimage_0002.png, ...');
  WriteLn('  --width N, --height N  pixels, rounded DOWN to a multiple of 32 ',
    '(default 1024)');
  WriteLn('  --steps N            Euler steps (default 18)');
  WriteLn('  --seed N             initial-noise seed for the FPC RNG (default 42)');
  WriteLn('  --image FILE         img2img (SDEdit) from this init image: the ',
    'output keeps its aspect');
  WriteLn('                       ratio at the area of --width x --height. It ',
    're-styles the image;');
  WriteLn('                       it does not follow edit instructions.');
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
  WriteLn('  --vae-tile S[,T]     VAE tile S pixels every T pixels, multiples ',
    'of 16 (default 256,192)');
  WriteLn('  --serial             single-threaded forward passes (default: ',
    'the parallel layer scheduler with intra-layer threading)');
  WriteLn('  --max-threads N      cap the parallel forward at N worker threads ',
    '(default: every CPU thread)');
  WriteLn('  --token-ids LIST     one-shot from comma-separated prompt token ',
    'ids instead of -p (no processor/ needed)');
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
  WriteLn('  /steps N             Euler steps for the next prompts');
  WriteLn('  /seed N              seed of the next image (otherwise the seed ',
    'grows by one per image)');
  WriteLn('  /repeat N PROMPT     N images of PROMPT (encoded once), ',
    'consecutive seeds, numbered files');
  WriteLn('  /image FILE          init image for the next prompts (img2img); ',
    '/image off clears it');
  WriteLn('  /strength S          img2img strength for the next images, ',
    'in (0, 1]');
  WriteLn('  /tile SIZE[,STRIDE]  VAE tile, as --vae-tile. In the REPL (and ',
    'with --repeat N > 1) the');
  WriteLn('                       transformer stays in OpenCL memory ',
    'during the VAE decode; /tile 128');
  WriteLn('                       (or --vae-tile 128) lowers the VAE''s ',
    'memory if both do not fit.');
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

function ParseTokenIds(const List: string): TNeuralIntegerArray;
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
  ModelFolder, Prompt, OutputFile, TokenList, Arg, ArgProblem: string;
  Width, Height, StepCount, DropCount, ArgPos: integer;
  RepeatCount, ImageNumber, FileNumber: integer;
  Seed: cardinal;
  HasPrompt, UseInt8Input, UseSerial, UseProfile, UseStats: boolean;
  // The image whose stage table shows the encode of the current prompt.
  EncodeImageNumber: integer;
  // img2img: the init image (0..255 RGBA) and its latents, encoded at
  // ImageLatentsWidth x ImageLatentsHeight (0: not encoded yet) before image
  // InitEncodeImageNumber.
  HasInitImage: boolean;
  InitImageFile: string;
  InitImage, ImageLatents: TNNetVolume;
  ImageLatentsWidth, ImageLatentsHeight, InitEncodeImageNumber: integer;
  Strength: double;
  HasStrengthArg: boolean;
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

  // Encodes TokenIds into PromptEmbeds and ends the encode phase line.
  procedure EncodePrompt();
  begin
    IsPeakPerImage := (UseStats or UseProfile) and ResetPeakRss();
    Pipeline.EncodeTokenIds(TokenIds, DropCount, PromptEmbeds);
    Reporter.OnPhase(qppDone);
    EncodeImageNumber := ImageNumber + 1;
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

  // GetOutputSizeFor the active init image (nil without one).
  procedure GetOutputSize(out OutputWidth, OutputHeight: integer);
  begin
    if HasInitImage then GetOutputSizeFor(InitImage, OutputWidth, OutputHeight)
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
      Result := 'the init image''s aspect gives ' + IntToStr(OutputWidth) +
        'x' + IntToStr(OutputHeight) + ': ' + Result;
  end;

  // Loads FileName (with alpha) and makes it the init image when its output
  // size works; otherwise returns what is wrong and changes nothing.
  function LoadInitImage(const FileName: string): string;
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
      if Result = '' then Result := OutputSizeProblemFor(Loaded);
      if Result <> '' then exit;
      InitImage.Copy(Loaded);
    finally
      Loaded.Free;
    end;
    HasInitImage := true;
    InitImageFile := FileName;
    ImageLatentsWidth := 0;
    ImageLatentsHeight := 0;
  end;

  // Encodes the init image unless its latents already match the output size;
  // raises when the size or the strength cannot work.
  procedure PrepareInitLatents();
  var
    OutputWidth, OutputHeight: integer;
    Problem: string;
  begin
    if not HasInitImage then exit;
    Problem := OutputSizeProblemFor(InitImage);
    if Problem <> '' then raise Exception.Create(Problem);
    // Raises when the strength leaves no step.
    TNNetFlowMatchEulerScheduler.Img2ImgStartStep(StepCount, Strength);
    GetOutputSize(OutputWidth, OutputHeight);
    if (ImageLatentsWidth = OutputWidth) and
      (ImageLatentsHeight = OutputHeight) then exit;
    ImageLatentsWidth := 0;
    Pipeline.EncodeImage(InitImage, OutputWidth, OutputHeight, ImageLatents);
    Reporter.OnPhase(qppDone);
    ImageLatentsWidth := OutputWidth;
    ImageLatentsHeight := OutputHeight;
    InitEncodeImageNumber := ImageNumber + 1;
  end;

  // The Image and Init image lines of the run header.
  procedure PrintImageSettings();
  var
    OutputWidth, OutputHeight, StartStep: integer;
  begin
    GetOutputSize(OutputWidth, OutputHeight);
    WriteLn('Image      : ', OutputWidth, 'x', OutputHeight, ' (',
      (OutputWidth div 16) * (OutputHeight div 16), ' image tokens), ',
      StepCount, ' steps, seed ', Seed);
    if not HasInitImage then exit;
    // Raises when the strength leaves no step.
    StartStep := TNNetFlowMatchEulerScheduler.Img2ImgStartStep(StepCount,
      Strength);
    WriteLn('Init image : ', InitImageFile, ' (', InitImage.SizeX, 'x',
      InitImage.SizeY, '), strength ', StrengthText(Strength), ', the last ',
      StepCount - StartStep, ' of ', StepCount, ' steps');
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
    SaveStart: TDateTime;
    PngSaveMs: double;
    OutputWidth, OutputHeight: integer;
  begin
    // The image that encoded the prompt had its mark reset before the encode.
    if ImageNumber <> EncodeImageNumber then
      IsPeakPerImage := (UseStats or UseProfile) and ResetPeakRss();
    ImageStart := GetTickCount64;
    GetOutputSize(OutputWidth, OutputHeight);
    if HasInitImage then
      Pipeline.GenerateFromEmbeds(PromptEmbeds, OutputWidth, OutputHeight,
        StepCount, Seed, Image, nil, ImageLatents, Strength)
    else
      Pipeline.GenerateFromEmbeds(PromptEmbeds, OutputWidth, OutputHeight,
        StepCount, Seed, Image);
    SaveStart := Now();
    Image.Mul(255);
    if not SaveImageFromVolumeIntoFile(Image, ImageFile) then
      raise Exception.Create('could not write ' + ImageFile);
    PngSaveMs := (Now() - SaveStart) * MSecsPerDay;
    WriteLn('Wrote ', ImageFile, ' (', Image.SizeX, 'x', Image.SizeY, 'x',
      Image.Depth, ') in ', ((GetTickCount64 - ImageStart) / 1000):0:1,
      ' s; ', MemoryReport());
    if UseStats or UseProfile then
      PrintStageTable(Pipeline.ImageStats, ImageNumber, EncodeImageNumber,
        InitEncodeImageNumber, PngSaveMs, UseProfile, IsPeakPerImage);
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
    ImageFile, StepText: string;
    OutputWidth, OutputHeight: integer;
  begin
    Inc(ImageNumber);
    if Numbered then ImageFile := NextFreeOutputFile(OutputFile, FileNumber)
    else ImageFile := OutputFile;
    GetOutputSize(OutputWidth, OutputHeight);
    StepText := IntToStr(StepCount) + ' steps';
    if HasInitImage then
      StepText := Format('img2img strength %s, the last %d of %d steps',
        [StrengthText(Strength), StepCount -
        TNNetFlowMatchEulerScheduler.Img2ImgStartStep(StepCount, Strength),
        StepCount]);
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
      TokenIds := Pipeline.TokenizePrompt(PromptText, DropCount);
      EncodePrompt();
      PrepareInitLatents();
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
    PrepareInitLatents();
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

  // /size, /steps, /seed, /tile, /repeat, /image, /strength, /profile, /stats
  // and /quit; false on /quit or when /repeat ran out of memory.
  function RunReplCommand(const Line: string): boolean;
  var
    IsOn: boolean;
    SpacePos, NewWidth, NewHeight, NewTileSize, NewTileStride: integer;
    NewRepeatCount: integer;
    NewValue: int64;
    NewStrength: double;
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
        if HasInitImage then
        begin
          GetOutputSize(NewWidth, NewHeight);
          WriteLn('[size ', Width, 'x', Height, '; the init image''s aspect ',
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
        WriteLn('[/image: expected /image FILE or /image off]')
      else if LowerCase(Argument) = 'off' then
      begin
        HasInitImage := false;
        ImageLatentsWidth := 0;
        WriteLn('[init image off: text-to-image]');
      end
      else
      begin
        Problem := LoadInitImage(Argument);
        if Problem <> '' then
        begin
          if HasInitImage then
            WriteLn('[/image: ', Problem, ' - the init image ', InitImageFile,
              ' stays active]')
          else WriteLn('[/image: ', Problem, ' - no init image is set]');
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
    else if Command = 'strength' then
    begin
      Problem := ParseStrength(Argument, NewStrength);
      if Problem <> '' then WriteLn('[/strength: ', Problem, ']')
      else
      begin
        Strength := NewStrength;
        WriteLn('[strength ', StrengthText(Strength), ']');
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
      '/tile, /repeat, /image, /strength, /profile, /stats, /quit]');
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
      '/tile SIZE[,STRIDE], /repeat N PROMPT, /image FILE|off, /strength S, ',
      '/profile on|off, /stats on|off, /quit.');
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
  InitEncodeImageNumber := 0;
  Strength := csDefaultStrength;
  HasStrengthArg := false;
  MaxThreads := 0;
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
    else if Arg = '--image' then InitImageFile := NextArg()
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
    else if Arg = '--serial' then UseSerial := true
    else if Arg = '--max-threads' then MaxThreads := StrToInt(NextArg())
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
  if HasStrengthArg and (InitImageFile = '') then
  begin
    if HasPrompt or (TokenList <> '') then
      WriteLn('[--strength has no effect without --image]')
    else WriteLn('[--strength applies once /image sets an init image]');
  end;

  StartTime := GetTickCount64;
  Reporter := TQwenImageReporter.Create();
  PromptEmbeds := TNNetVolume.Create();
  Image := TNNetVolume.Create();
  InitImage := TNNetVolume.Create();
  ImageLatents := TNNetVolume.Create();
  Pipeline := nil;
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
  try
    Pipeline := TQwenImage21Pipeline.Create(ModelFolder);
    Pipeline.TransformerFormat := WeightFormat;
    Pipeline.TextEncoderInt8 := WeightFormat <> qiwFP32;
    Pipeline.Int8Input := UseInt8Input;
    Pipeline.VaeTileSize := VaeTileSize;
    Pipeline.VaeTileStride := VaeTileStride;
    Pipeline.Parallel := not UseSerial;
    Pipeline.MaxThreads := MaxThreads;
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
    if InitImageFile <> '' then
    begin
      ArgProblem := LoadInitImage(InitImageFile);
      if ArgProblem <> '' then
        raise Exception.Create('--image: ' + ArgProblem);
    end;
    WriteLn('Model      : ', ModelFolder);
    PrintImageSettings();
    case WeightFormat of
      qiwInt8: WriteLn('Weights    : transformer int8, text encoder int8');
      qiwInt4: WriteLn('Weights    : transformer int4, text encoder int8');
      qiwFP32: WriteLn('Weights    : FP32 (--fp32)');
    end;
    WriteLn('Compute    : ', ComputeText);
    WriteLn('VAE tiles  : ', VaeTileSize, ' px every ', VaeTileStride, ' px');
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
        TokenIds := ParseTokenIds(TokenList);
        WriteLn('Prompt     : ', Length(TokenIds), ' token ids, drop ',
          DropCount);
      end
      else
      begin
        WriteLn('Prompt     : ', Prompt);
        TokenIds := Pipeline.TokenizePrompt(Prompt, DropCount);
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
    Pipeline.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    ImageLatents.Free;
    InitImage.Free;
    Image.Free;
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
