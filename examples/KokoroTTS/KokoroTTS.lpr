program KokoroTTS;
(*
KokoroTTS: Kokoro-82M text-to-speech (hexgrad/Kokoro-82M, Apache-2.0,
StyleTTS2 + iSTFTNet) from a Hugging Face snapshot folder (config.json,
kokoro-v1_0.pth, voices/<name>.pt), through TNNetKokoro
(neural/neuralpretrained.pas). Writes a 16-bit 24 kHz mono WAV.

  phonemes -> ids -> stage A (PL-BERT, text encoder, durations)
    -> stage B (F0 and energy curves) -> stage-C decoder
    -> harmonic source + STFT (host) -> stage-C generator -> iSTFT (host).

The input is a phoneme string (misaki's English G2P output); text -> phoneme
conversion is not included yet. Zero noise: the same input gives the same WAV.

USAGE
  KokoroTTS --model-dir DIR --phonemes "IPA" [--voice af_heart] [--speed 1]
            [--out kokoro.wav] [--serial] [--max-threads N]
            [--gpu] [--gpu-platform N] [--gpu-device N]
            [--no-gpu-shared-kernel]

Coded by Claude (AI).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)
{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  SysUtils, Classes, {$IFDEF OpenCL}neuralopencl,{$ENDIF}
  neuralvolume, neuralpretrained, neuralthread, neuralaudio,
  neuralsafetensors;

// Peak resident memory (VmHWM) in MB from /proc/self/status, or '?'.
function PeakRssMegabytes(): string;
var
  Status: TStringList;
  LinePos: integer;
  Line: string;
  Kilobytes: int64;
begin
  Result := '?';
  if not FileExists('/proc/self/status') then exit;
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
      if Pos('VmHWM:', Line) = 1 then
      begin
        Kilobytes := StrToInt64Def(Trim(StringReplace(Copy(Line, 7, MaxInt),
          'kB', '', [])), -1);
        if Kilobytes >= 0 then Result := IntToStr(Kilobytes div 1024);
        exit;
      end;
    end;
  finally
    Status.Free;
  end;
end;

procedure PrintUsage();
begin
  WriteLn('Usage: KokoroTTS --model-dir DIR --phonemes "IPA" [options]');
  WriteLn('  --model-dir DIR      hexgrad/Kokoro-82M snapshot folder ',
    '(config.json, *.pth, voices/)');
  WriteLn('  --phonemes TEXT      phoneme string (misaki English G2P output), ',
    'e.g. "h', #$C9#$99, 'l', #$CB#$88, 'O w', #$CB#$88, #$C9#$9C, #$C9#$B9,
    'ld."');
  WriteLn('  --voice NAME         voices/NAME.pt (default af_heart)');
  WriteLn('  --speed S            speaking rate, > 0 (default 1)');
  WriteLn('  --out FILE           output WAV, 16-bit 24 kHz mono ',
    '(default kokoro.wav)');
  WriteLn('  --serial             single-threaded forward passes');
  WriteLn('  --max-threads N      cap the parallel forward at N worker threads ',
    '(default: every CPU thread)');
  WriteLn('  --gpu                run the stage nets on OpenCL (needs a build ',
    'with -dOpenCL; default CPU)');
  WriteLn('  --gpu-platform N     OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N       OpenCL device index within the platform ',
    '(default 0)');
  WriteLn('  --no-gpu-shared-kernel  give each layer private OpenCL kernels');
end;

var
  ModelDir, VoiceName, Phonemes, OutFile, Arg, SkippedSymbols,
    ComputeText: string;
  Speed: TNeuralFloat;
  UseSerial, UseOpenCL, HasSharedKernel: boolean;
  MaxThreads, OpenCLPlatform, OpenCLDevice, ArgPos, SamplesPerFrame,
    RatePos: integer;
  DotFormat: TFormatSettings;
  Config: TKokoroConfig;
  Voice: TKokoroVoicePack;
  Reader: TNNetSafeTensorsReader;
  Kokoro: TNNetKokoro;
  Audio: TNNetVolume;
  TokenIds: TNeuralIntegerArray;
  StartTime, StageTime, LoadMs, SynthesisMs, WriteMs: QWord;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  OpenCLProblem: string;
  {$ENDIF}

  function NextArg(): string;
  begin
    Inc(ArgPos);
    if ArgPos > ParamCount then
    begin
      WriteLn(Arg, ': missing value.');
      Halt(2);
    end;
    Result := ParamStr(ArgPos);
  end;

begin
  DotFormat := DefaultFormatSettings;
  DotFormat.DecimalSeparator := '.';
  ModelDir := '';
  VoiceName := 'af_heart';
  Phonemes := '';
  OutFile := 'kokoro.wav';
  Speed := 1;
  UseSerial := false;
  UseOpenCL := false;
  HasSharedKernel := true;
  MaxThreads := 0;
  OpenCLPlatform := 0;
  OpenCLDevice := 0;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--model-dir' then ModelDir := NextArg()
    else if Arg = '--voice' then VoiceName := NextArg()
    else if Arg = '--phonemes' then Phonemes := NextArg()
    else if Arg = '--speed' then Speed := StrToFloatDef(NextArg(), -1, DotFormat)
    else if Arg = '--out' then OutFile := NextArg()
    else if Arg = '--serial' then UseSerial := true
    else if Arg = '--max-threads' then MaxThreads := StrToIntDef(NextArg(), -1)
    else if Arg = '--gpu' then UseOpenCL := true
    else if Arg = '--gpu-platform' then OpenCLPlatform := StrToIntDef(NextArg(), 0)
    else if Arg = '--gpu-device' then OpenCLDevice := StrToIntDef(NextArg(), 0)
    else if Arg = '--no-gpu-shared-kernel' then HasSharedKernel := false
    else if (Arg = '--help') or (Arg = '-h') then
    begin
      PrintUsage();
      Halt(0);
    end
    else
    begin
      WriteLn('Unknown argument: ', Arg);
      PrintUsage();
      Halt(2);
    end;
    Inc(ArgPos);
  end;
  if (ModelDir = '') or (Phonemes = '') then
  begin
    WriteLn('--model-dir and --phonemes are required (text -> phoneme ',
      'conversion is not included yet).');
    PrintUsage();
    Halt(2);
  end;
  if Speed <= 0 then
  begin
    WriteLn('--speed: must be a number greater than 0.');
    Halt(2);
  end;
  if MaxThreads < 0 then
  begin
    WriteLn('--max-threads: must be at least 1.');
    Halt(2);
  end;

  StartTime := GetTickCount64;
  Kokoro := nil;
  Audio := TNNetVolume.Create();
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
  try
    WriteLn('Model      : ', ModelDir);
    WriteLn('Voice      : ', VoiceName);
    Reader := OpenKokoroSnapshot(ModelDir, VoiceName, Config, Voice);
    try
      Kokoro := TNNetKokoro.Create(Reader, Config);
    finally
      Reader.Free;
    end;
    Kokoro.Parallel := not UseSerial;
    Kokoro.MaxThreads := MaxThreads;
    ComputeText := 'CPU';
    {$IFDEF OpenCL}
    if UseOpenCL then
    begin
      OpenCLDevices := TEasyOpenCL.Create();
      if OpenCLDevices.SelectPlatformAndDevice(OpenCLPlatform, OpenCLDevice,
        OpenCLProblem) then
      begin
        if Kokoro.EnableOpenCL(OpenCLDevices.PlatformIds[OpenCLPlatform],
          OpenCLDevices.Devices[OpenCLDevice], HasSharedKernel) then
          ComputeText := 'OpenCL on ' +
            OpenCLDevices.PlatformNames[OpenCLPlatform] + ' / ' +
            OpenCLDevices.DeviceNames[OpenCLDevice] +
            ' (stage nets; host steps on the CPU)'
        else ComputeText := 'CPU (the OpenCL program did not build)';
      end
      else ComputeText := 'CPU (' + OpenCLProblem + ')';
    end;
    {$ELSE}
    if UseOpenCL then ComputeText := 'CPU (built without OpenCL)';
    {$ENDIF}
    LoadMs := GetTickCount64 - StartTime;
    WriteLn('Compute    : ', ComputeText);
    if UseSerial then WriteLn('Threads    : serial (single-threaded)')
    else if MaxThreads > 0 then
      WriteLn('Threads    : parallel, at most ', MaxThreads, ' workers')
    else WriteLn('Threads    : parallel, every CPU thread (',
      NeuralDefaultThreadCount, ')');
    WriteLn('Loaded in  : ', (LoadMs / 1000):0:2, ' s');

    WriteLn('Phonemes   : ', Phonemes);
    TokenIds := KokoroPhonemesToIds(Kokoro.Config, Phonemes, SkippedSymbols);
    if SkippedSymbols <> '' then
      WriteLn('Skipped    : ', SkippedSymbols, ' (not in the vocab)');
    WriteLn('Tokens     : ', Length(TokenIds), ' (with the two wrapping 0s)');

    StageTime := GetTickCount64;
    Kokoro.SynthesizePhonemes(Phonemes, Voice, Speed, Audio, SkippedSymbols);
    SynthesisMs := GetTickCount64 - StageTime;
    // Each predicted frame is x2 in the decoder, then the generator
    // upsamples and the iSTFT hop adds the rest.
    SamplesPerFrame := 2 * Kokoro.Config.ISTFTHopSize;
    for RatePos := 0 to High(Kokoro.Config.Generator.UpsampleRates) do
      SamplesPerFrame := SamplesPerFrame *
        Kokoro.Config.Generator.UpsampleRates[RatePos];
    WriteLn('Frames     : ', Audio.Size div SamplesPerFrame, ' (',
      SamplesPerFrame, ' samples each)');
    WriteLn('Audio      : ', Audio.Size, ' samples = ',
      (Audio.Size / KokoroSampleRate):0:2, ' s at ', KokoroSampleRate, ' Hz');
    WriteLn('Synthesis  : ', (SynthesisMs / 1000):0:2, ' s');

    StageTime := GetTickCount64;
    SaveVolumeToWav16(Audio, OutFile, KokoroSampleRate);
    WriteMs := GetTickCount64 - StageTime;
    WriteLn('Wrote      : ', OutFile, ' in ', (WriteMs / 1000):0:2, ' s');
    WriteLn('Total      : ', ((GetTickCount64 - StartTime) / 1000):0:2,
      ' s; peak RSS ', PeakRssMegabytes(), ' MB');
  finally
    Kokoro.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    Audio.Free;
  end;
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
