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
memory at a time.

REPL (neither -p nor --token-ids): the three components load once and stay
in memory; stdin gives one prompt per line (a piped file is a batch, end of
input ends the session). Images go to numbered files from the --output base
(qwenimage.png -> qwenimage_0001.png, ...; existing files are skipped, never
overwritten). Commands: /size WxH, /steps N, /seed N (otherwise the seed
grows by one per prompt), /tile SIZE[,STRIDE], /quit.
GPU memory: in the REPL the transformer stays in OpenCL memory while the VAE
decodes, so both need room at once; /tile 128 (or --vae-tile 128) lowers the
VAE's share (~3.4 GB of layer buffers instead of ~10.7 GB at 256).

The initial noise comes from the FPC RNG (--seed), so an image is repeatable
here but not equal to a diffusers image with the same seed.

USAGE
  QwenImage --model DIR [-p TEXT | --token-ids ID,ID,... [--drop-count N]]
            [--output FILE.png]
            [--width 1024] [--height 1024] [--steps 18] [--seed 42]
            [--int8 | --int4 | --fp32] [--int8-input] [--vae-tile SIZE[,STRIDE]]
            [--serial] [--max-threads N]
            [--gpu | --cpu] [--gpu-platform N] [--gpu-device N]
            [--no-gpu-shared-kernel] [--profile]
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
  neuralvolume, neuralnetwork, neuralpretrained, neuraldatasets, neuralthread;

const
  // Largest accepted image side in pixels (64x the 1024 default area).
  csMaxImageSide = 8192;

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

// 'VmRSS 123 MB, peak 456 MB' from /proc/self/status; '' elsewhere.
function MemoryReport(): string;
var
  Status: TStringList;
  LinePos: integer;
  Line, Rss, Peak: string;

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
  Result := '';
  if not FileExists('/proc/self/status') then exit;
  Rss := '?';
  Peak := '?';
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
  Result := 'RSS ' + Rss + ' MB, peak ' + Peak + ' MB';
end;

function PhaseName(Phase: TQwenImage21PipelinePhase): string;
begin
  case Phase of
    qppLoadTextEncoder: Result := 'load text encoder';
    qppEncodePrompt: Result := 'encode prompt';
    qppLoadTransformer: Result := 'load transformer';
    qppEncodePrefix: Result := 'transformer prefix';
    qppDenoise: Result := 'denoise';
    qppLoadVae: Result := 'load VAE';
    qppDecode: Result := 'VAE decode';
  else
    Result := 'done';
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
  WriteLn('                       says which queues are drained.');
  WriteLn;
  WriteLn('Without -p or --token-ids, QwenImage loads the text encoder, the ',
    'transformer and the VAE once');
  WriteLn('and reads one prompt per line from stdin (a piped file is a batch). ',
    'REPL commands:');
  WriteLn('  /size WxH            image size for the next prompts');
  WriteLn('  /steps N             Euler steps for the next prompts');
  WriteLn('  /seed N              seed of the next prompt (otherwise the seed ',
    'grows by one per prompt)');
  WriteLn('  /tile SIZE[,STRIDE]  VAE tile, as --vae-tile. In the REPL the ',
    'transformer stays in OpenCL');
  WriteLn('                       memory during the VAE decode; /tile 128 (or ',
    '--vae-tile 128) lowers the');
  WriteLn('                       VAE''s memory if both do not fit.');
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
  ModelFolder, Prompt, OutputFile, TokenList, Arg, TileProblem: string;
  Width, Height, StepCount, DropCount, ArgPos: integer;
  Seed: cardinal;
  HasPrompt, UseInt8Input, UseSerial, UseProfile: boolean;
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

  // Encodes TokenIds, generates at the current settings and writes ImageFile.
  procedure GenerateImage(const ImageFile: string);
  var
    ImageStart: QWord;
  begin
    ImageStart := GetTickCount64;
    Pipeline.EncodeTokenIds(TokenIds, DropCount, PromptEmbeds);
    Pipeline.GenerateFromEmbeds(PromptEmbeds, Width, Height, StepCount, Seed,
      Image);
    Image.Mul(255);
    if not SaveImageFromVolumeIntoFile(Image, ImageFile) then
      raise Exception.Create('could not write ' + ImageFile);
    WriteLn('Wrote ', ImageFile, ' (', Image.SizeX, 'x', Image.SizeY, 'x',
      Image.Depth, ') in ', ((GetTickCount64 - ImageStart) / 1000):0:1,
      ' s; ', MemoryReport());
    if UseProfile then
    begin
      WriteLn;
      Write(Pipeline.TransformerProfileReport);
      Write(Pipeline.VaeProfileReport);
    end;
  end;

  // /size, /steps, /seed, /tile and /quit; false on /quit.
  function RunReplCommand(const Line: string): boolean;
  var
    SpacePos, NewWidth, NewHeight, NewTileSize, NewTileStride: integer;
    NewValue: int64;
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
        WriteLn('[size ', Width, 'x', Height, ']');
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
        WriteLn('[seed ', Seed, ' for the next prompt]');
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
    else WriteLn('[unknown command /', Command,
      ' - /size, /steps, /seed, /tile, /quit]');
  end;

  // One prompt per stdin line until /quit or the end of the input.
  procedure RunRepl();
  var
    Line, ImageFile: string;
    ImageNumber, FileNumber: integer;
    InteractiveInput: boolean;
  begin
    WriteLn('Loading the text encoder, the transformer and the VAE decoder ',
      'to keep them in memory.');
    Pipeline.LoadComponents();
    WriteLn('Resident   : ', MemoryReport());
    PrintComputeResult();
    WriteLn('Type a prompt per line; /size WxH, /steps N, /seed N, ',
      '/tile SIZE[,STRIDE], /quit.');
    ImageNumber := 0;
    FileNumber := 0;
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
      Inc(ImageNumber);
      ImageFile := NextFreeOutputFile(OutputFile, FileNumber);
      WriteLn('Image ', ImageNumber, ': ', Width, 'x', Height, ', ', StepCount,
        ' steps, seed ', Seed, ' -> ', ImageFile, ': "', ShortPrompt(Line, 60),
        '"');
      try
        TokenIds := Pipeline.TokenizePrompt(Line, DropCount);
        GenerateImage(ImageFile);
      except
        // The stages free the per-image passes on the way out; the loaded
        // weights stay valid, except after an allocation failure.
        on E: EOutOfMemory do
        begin
          Reporter.EndStepLine();
          WriteLn('Error: ', E.Message, ' - out of memory, ending the ',
            'session.');
          ExitCode := 1;
          break;
        end;
        on E: Exception do
        begin
          Reporter.OnPhase(qppDone);
          WriteLn('Error: ', E.Message, ' - image ', ImageNumber, ' skipped.');
        end;
      end;
      // The seed wraps from High(cardinal) to 0.
      {$PUSH}{$Q-}{$R-}
      Inc(Seed);
      {$POP}
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
  WeightFormat := qiwInt8;
  UseInt8Input := false;
  UseSerial := false;
  UseProfile := false;
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
    else if Arg = '--vae-tile' then
    begin
      TileProblem := ParseVaeTile(NextArg(), VaeTileSize, VaeTileStride);
      if TileProblem <> '' then
      begin
        WriteLn('--vae-tile: ', TileProblem);
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

  StartTime := GetTickCount64;
  Reporter := TQwenImageReporter.Create();
  PromptEmbeds := TNNetVolume.Create();
  Image := TNNetVolume.Create();
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
    Pipeline.LayerProfiling := UseProfile;
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
    WriteLn('Model      : ', ModelFolder);
    WriteLn('Image      : ', Width, 'x', Height, ' (', (Width div 16) *
      (Height div 16), ' image tokens), ', StepCount, ' steps, seed ', Seed);
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
      GenerateImage(OutputFile);
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
