program QwenImage;
(*
QwenImage: Qwen-Image-2.1 text-to-image from a diffusers checkpoint folder
(Qwen/Qwen-Image-2.1: model_index.json, processor/, text_encoder/,
transformer/, vae/, scheduler/), on the CPU, through TQwenImage21Pipeline
(neural/neuralpretrained.pas):

  prompt -> processor/tokenizer.json -> Qwen3-VL text encoder (freed)
    -> transformer prefix pass (text K/V, once)
    -> N flow-matching Euler steps over the image tokens (transformer freed)
    -> VAE decode, tiled -> RGBA PNG.

Only one component is in memory at a time. The initial noise comes from the
FPC RNG (--seed), so an image is repeatable here but not equal to a diffusers
image with the same seed.

USAGE
  QwenImage --model DIR [--prompt TEXT] [--output FILE.png]
            [--width 1024] [--height 1024] [--steps 40] [--seed 42]
            [--int8 | --int4] [--int8-input] [--vae-tile SIZE[,STRIDE]]
            [--token-ids ID,ID,... --drop-count N]
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
  SysUtils, Classes,
  neuralvolume, neuralnetwork, neuralpretrained, neuraldatasets;

const
  csDefaultPrompt = 'A capybara wearing a wizard hat, reading a book by ' +
    'candlelight, oil painting';

type
  // Prints phase and step timings with the process memory (Linux /proc).
  TQwenImageReporter = class(TObject)
  private
    FPhaseStart, FStepStart: QWord;
    FPhaseName: string;
    procedure EndPhase();
  public
    constructor Create();
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
end;

procedure TQwenImageReporter.EndPhase();
begin
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
begin
  StepEnd := GetTickCount64;
  WriteLn(Format('    step %d/%d  t=%7.2f  %6.1f s  |x| max %.3f  %s',
    [StepIndex + 1, StepCount, Timestep, (StepEnd - FStepStart) / 1000,
     Latents.GetMaxAbs(), MemoryReport()]));
  FStepStart := StepEnd;
end;

procedure PrintHelp();
begin
  WriteLn('QwenImage: Qwen-Image-2.1 text-to-image (CPU).');
  WriteLn('  --model DIR          diffusers folder with model_index.json (required)');
  WriteLn('  --prompt TEXT        prompt (default: "', csDefaultPrompt, '")');
  WriteLn('  --output FILE        image file; .png keeps the alpha channel ',
    '(default qwenimage.png)');
  WriteLn('  --width N, --height N  pixels, rounded DOWN to a multiple of 32 ',
    '(default 1024)');
  WriteLn('  --steps N            Euler steps (default 40)');
  WriteLn('  --seed N             initial-noise seed for the FPC RNG (default 42)');
  WriteLn('  --int8               transformer block weights in int8; text ',
    'encoder weights in int8');
  WriteLn('  --int4               transformer block weights in Q4_0-style int4 ',
    '(block 32); text encoder weights in int8');
  WriteLn('                       Without either, both load FP32 (the 7B ',
    'transformer then runs a slow kernel).');
  WriteLn('                       Norms, embeddings, the transformer''s ',
    'input/output/timestep nets and the VAE stay FP32.');
  WriteLn('  --int8-input         int8 activations into the transformer''s ',
    'int8/int4 projections (needs --int8 or --int4)');
  WriteLn('  --vae-tile S[,T]     VAE tile S pixels every T pixels, multiples ',
    'of 16 (default 128,96)');
  WriteLn('  --token-ids LIST     comma-separated prompt token ids instead of ',
    '--prompt (no processor/ needed)');
  WriteLn('  --drop-count N       leading system-prompt tokens to drop with ',
    '--token-ids (default 0)');
  WriteLn('OpenCL is not offered: the transformer''s weight swap runs on the ',
    'CPU only (tasklist B1).');
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
  ModelFolder, Prompt, OutputFile, TokenList, Arg, TileArg: string;
  Width, Height, StepCount, DropCount, ArgPos, CommaPos: integer;
  Seed: cardinal;
  UseInt8, UseInt4, UseInt8Input: boolean;
  VaeTileSize, VaeTileStride: integer;
  Pipeline: TQwenImage21Pipeline;
  Reporter: TQwenImageReporter;
  PromptEmbeds, Image: TNNetVolume;
  TokenIds: TNeuralIntegerArray;
  StartTime: QWord;

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

begin
  ModelFolder := '';
  Prompt := csDefaultPrompt;
  OutputFile := 'qwenimage.png';
  TokenList := '';
  Width := 1024;
  Height := 1024;
  StepCount := 40;
  Seed := 42;
  DropCount := 0;
  UseInt8 := false;
  UseInt4 := false;
  UseInt8Input := false;
  VaeTileSize := 128;
  VaeTileStride := 96;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--model' then ModelFolder := NextArg()
    else if Arg = '--prompt' then Prompt := NextArg()
    else if Arg = '--output' then OutputFile := NextArg()
    else if Arg = '--width' then Width := StrToInt(NextArg())
    else if Arg = '--height' then Height := StrToInt(NextArg())
    else if Arg = '--steps' then StepCount := StrToInt(NextArg())
    else if Arg = '--seed' then Seed := StrToInt64(NextArg())
    else if Arg = '--int8' then UseInt8 := true
    else if Arg = '--int4' then UseInt4 := true
    else if Arg = '--int8-input' then UseInt8Input := true
    else if Arg = '--token-ids' then TokenList := NextArg()
    else if Arg = '--drop-count' then DropCount := StrToInt(NextArg())
    else if Arg = '--vae-tile' then
    begin
      TileArg := NextArg();
      CommaPos := Pos(',', TileArg);
      if CommaPos > 0 then
      begin
        VaeTileSize := StrToInt(Copy(TileArg, 1, CommaPos - 1));
        VaeTileStride := StrToInt(Copy(TileArg, CommaPos + 1, MaxInt));
      end
      else
      begin
        VaeTileSize := StrToInt(TileArg);
        VaeTileStride := (VaeTileSize * 3 div 4) div 16 * 16;
        if VaeTileStride < 16 then VaeTileStride := 16;
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
  if UseInt8 and UseInt4 then
  begin
    WriteLn('--int8 and --int4 are exclusive.');
    Halt(2);
  end;
  if UseInt8Input and not (UseInt8 or UseInt4) then
  begin
    WriteLn('--int8-input needs --int8 or --int4.');
    Halt(2);
  end;

  StartTime := GetTickCount64;
  Reporter := TQwenImageReporter.Create();
  PromptEmbeds := TNNetVolume.Create();
  Image := TNNetVolume.Create();
  Pipeline := nil;
  try
  try
    Pipeline := TQwenImage21Pipeline.Create(ModelFolder);
    if UseInt8 then Pipeline.TransformerFormat := qiwInt8
    else if UseInt4 then Pipeline.TransformerFormat := qiwInt4;
    Pipeline.TextEncoderInt8 := UseInt8 or UseInt4;
    Pipeline.Int8Input := UseInt8Input;
    Pipeline.VaeTileSize := VaeTileSize;
    Pipeline.VaeTileStride := VaeTileStride;
    Pipeline.OnPhase := @Reporter.OnPhase;
    Pipeline.OnStep := @Reporter.OnStep;
    Width := TQwenImage21Pipeline.RoundDownImageSide(Width);
    Height := TQwenImage21Pipeline.RoundDownImageSide(Height);
    WriteLn('Model      : ', ModelFolder);
    WriteLn('Image      : ', Width, 'x', Height, ' (', (Width div 16) *
      (Height div 16), ' image tokens), ', StepCount, ' steps, seed ', Seed);
    if UseInt8 then WriteLn('Weights    : transformer int8, text encoder int8')
    else if UseInt4 then
      WriteLn('Weights    : transformer int4, text encoder int8')
    else WriteLn('Weights    : FP32');
    WriteLn('VAE tiles  : ', VaeTileSize, ' px every ', VaeTileStride, ' px');
    if TokenList <> '' then
    begin
      TokenIds := ParseTokenIds(TokenList);
      WriteLn('Prompt     : ', Length(TokenIds), ' token ids, drop ', DropCount);
    end
    else
    begin
      WriteLn('Prompt     : ', Prompt);
      TokenIds := Pipeline.TokenizePrompt(Prompt, DropCount);
      WriteLn('Tokens     : ', Length(TokenIds), ' (', DropCount,
        ' system-prompt tokens dropped)');
    end;
    Pipeline.EncodeTokenIds(TokenIds, DropCount, PromptEmbeds);
    Pipeline.GenerateFromEmbeds(PromptEmbeds, Width, Height, StepCount, Seed,
      Image);
    Image.Mul(255);
    if not SaveImageFromVolumeIntoFile(Image, OutputFile) then
      raise Exception.Create('could not write ' + OutputFile);
    WriteLn('Wrote ', OutputFile, ' (', Image.SizeX, 'x', Image.SizeY, 'x',
      Image.Depth, ') in ', ((GetTickCount64 - StartTime) / 1000):0:1,
      ' s; ', MemoryReport());
  finally
    Pipeline.Free;
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
