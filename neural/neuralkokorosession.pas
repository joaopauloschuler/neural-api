unit neuralkokorosession;

(*
neuralkokorosession
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).
*)

// neuralkokorosession -- a loaded Kokoro-82M snapshot folder for the
// frontends (examples/KokoroTTS, a later TTS HTTP server), as
// neuralchatengine is for ChatTerminal and ChatServer.
//
// TKokoroSession.Create loads config.json, the checkpoint and the first
// voice pack into a TNNetKokoro, applies the thread settings, optionally
// enables OpenCL and sets the English dictionary folder (default
// <model folder>/misaki). Per call it speaks one line of text (or one
// phoneme string) into a 16-bit 24 kHz WAV through TNNetKokoro.SynthesizeText
// / SynthesizePhonemes and keeps the result (report, timing) of the last
// line. SelectVoice switches to another voices/<name>.pt of the same folder;
// TNNetKokoro.EnglishG2PForVoice re-creates the G2P when the accent changes.
// The session runs one call at a time.

{$mode objfpc}{$H+}

interface

uses
  {$IFDEF OpenCL}
  neuralopencl,
  {$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralpretrained, neuralsafetensors, neuralg2p,
  neuralhftokenizer, neuralaudio, neuralthread;

type
  // What a frontend passes to TKokoroSession.Create.
  TKokoroSessionOptions = record
    ModelDir: string;        // hexgrad/Kokoro-82M snapshot folder
    DictionaryDir: string;   // misaki's dictionaries; '' = <ModelDir>/misaki
    VoiceName: string;       // voices/<VoiceName>.pt
    Speed: TNeuralFloat;     // speaking rate, > 0
    Serial: boolean;         // single-threaded forward passes
    MaxThreads: integer;     // 0 = every CPU thread
    Gpu: boolean;            // stage nets on OpenCL (builds with -dOpenCL)
    GpuPlatform: integer;
    GpuDevice: integer;
    GpuSharedKernel: boolean;
  end;

  // One line spoken by TKokoroSession: the WAV, the per-chunk report and
  // the time spent synthesizing and writing it.
  TKokoroLineResult = record
    FileName: string;
    Report: TKokoroTextReport;
    PhonemeCount: integer;   // code points over every chunk
    SampleCount: integer;    // at KokoroSampleRate
    SynthesisMs: QWord;
    WriteMs: QWord;
  end;

  TKokoroSession = class(TObject)
  private
    FOptions: TKokoroSessionOptions;
    FKokoro: TNNetKokoro;
    FVoice: TKokoroVoicePack;
    FSpeed: TNeuralFloat;
    FComputeDescription: string;
    FLastLine: TKokoroLineResult;
    FHasLastLine: boolean;
    {$IFDEF OpenCL}
    FOpenCLDevices: TEasyOpenCL;
    {$ENDIF}
    procedure SetSpeed(Value: TNeuralFloat);
    function GetDictionaryDir(): string;
    // Writes Audio to FileName and records the line as LastLine.
    function FinishLine(Audio: TNNetVolume; const FileName: string;
      const Report: TKokoroTextReport; SynthesisMs: QWord): TKokoroLineResult;
  public
    // Loads the snapshot folder and Options.VoiceName; raises
    // EPretrainedImportError when a file is missing or malformed.
    constructor Create(const Options: TKokoroSessionOptions);
    destructor Destroy(); override;
    // Loads the English G2P for the current voice now (dictionary files,
    // accent); raises ENeuralG2PError naming the missing files.
    procedure CheckTextInput();
    // Switches to voices/<VoiceName>.pt; with CheckText the G2P for its accent
    // is loaded first. On any error the current voice stays.
    procedure SelectVoice(const VoiceName: string; CheckText: boolean = true);
    // Speaks Text (English, chunked as kokoro KPipeline) into FileName.
    function SpeakTextToWav(const Text, FileName: string): TKokoroLineResult;
    // Speaks the phoneme string as given (no G2P, one chunk) into FileName.
    function SpeakPhonemesToWav(const Phonemes, FileName: string):
      TKokoroLineResult;
    property Options: TKokoroSessionOptions read FOptions;
    property Kokoro: TNNetKokoro read FKokoro;
    property Voice: TKokoroVoicePack read FVoice;
    // Raises EPretrainedImportError when not > 0.
    property Speed: TNeuralFloat read FSpeed write SetSpeed;
    property DictionaryDir: string read GetDictionaryDir;
    // 'CPU', or the OpenCL platform and device, or why OpenCL is off.
    property ComputeDescription: string read FComputeDescription;
    property LastLine: TKokoroLineResult read FLastLine;
    property HasLastLine: boolean read FHasLastLine;
  end;

function DefaultKokoroSessionOptions(): TKokoroSessionOptions;

// Thread setting in words: serial, at most N workers or every CPU thread.
function DescribeKokoroThreads(const Options: TKokoroSessionOptions): string;

implementation

function DefaultKokoroSessionOptions(): TKokoroSessionOptions;
begin
  Result.ModelDir := '';
  Result.DictionaryDir := '';
  Result.VoiceName := 'af_heart';
  Result.Speed := 1;
  Result.Serial := false;
  Result.MaxThreads := 0;
  Result.Gpu := false;
  Result.GpuPlatform := 0;
  Result.GpuDevice := 0;
  Result.GpuSharedKernel := true;
end;

function DescribeKokoroThreads(const Options: TKokoroSessionOptions): string;
begin
  if Options.Serial then Result := 'serial (single-threaded)'
  else if Options.MaxThreads > 0 then
    Result := 'parallel, at most ' + IntToStr(Options.MaxThreads) + ' workers'
  else
    Result := 'parallel, every CPU thread (' +
      IntToStr(NeuralDefaultThreadCount) + ')';
end;

constructor TKokoroSession.Create(const Options: TKokoroSessionOptions);
var
  Config: TKokoroConfig;
  Reader: TNNetSafeTensorsReader;
  {$IFDEF OpenCL}
  OpenCLProblem: string;
  {$ENDIF}
begin
  inherited Create();
  FOptions := Options;
  FHasLastLine := false;
  if Options.ModelDir = '' then
    raise EPretrainedImportError.Create('Kokoro: no model folder given.');
  SetSpeed(Options.Speed);
  Reader := OpenKokoroSnapshot(Options.ModelDir, Options.VoiceName, Config,
    FVoice);
  try
    FKokoro := TNNetKokoro.Create(Reader, Config);
  finally
    Reader.Free;
  end;
  FKokoro.Parallel := not Options.Serial;
  FKokoro.MaxThreads := Options.MaxThreads;
  FKokoro.EnglishDictionaryDir := GetDictionaryDir();
  FComputeDescription := 'CPU';
  {$IFDEF OpenCL}
  if Options.Gpu then
  begin
    FOpenCLDevices := TEasyOpenCL.Create();
    if FOpenCLDevices.SelectPlatformAndDevice(FOptions.GpuPlatform,
      FOptions.GpuDevice, OpenCLProblem) then
    begin
      if FKokoro.EnableOpenCL(FOpenCLDevices.PlatformIds[FOptions.GpuPlatform],
        FOpenCLDevices.Devices[FOptions.GpuDevice], Options.GpuSharedKernel) then
        FComputeDescription := 'OpenCL on ' +
          FOpenCLDevices.PlatformNames[FOptions.GpuPlatform] + ' / ' +
          FOpenCLDevices.DeviceNames[FOptions.GpuDevice] +
          ' (stage nets; host steps on the CPU)'
      else FComputeDescription := 'CPU (the OpenCL program did not build)';
    end
    else FComputeDescription := 'CPU (' + OpenCLProblem + ')';
  end;
  {$ELSE}
  if Options.Gpu then FComputeDescription := 'CPU (built without OpenCL)';
  {$ENDIF}
end;

destructor TKokoroSession.Destroy();
begin
  FKokoro.Free;
  {$IFDEF OpenCL}
  FOpenCLDevices.Free;
  {$ENDIF}
  inherited Destroy();
end;

procedure TKokoroSession.SetSpeed(Value: TNeuralFloat);
begin
  if not (Value > 0) then
    raise EPretrainedImportError.Create(
      'Kokoro: the speed must be a number greater than 0.');
  FSpeed := Value;
end;

function TKokoroSession.GetDictionaryDir(): string;
begin
  if FOptions.DictionaryDir <> '' then Result := FOptions.DictionaryDir
  else Result := KokoroEnglishDictionaryDir(FOptions.ModelDir);
end;

procedure TKokoroSession.CheckTextInput();
begin
  FKokoro.EnglishG2PForVoice(FVoice);
end;

procedure TKokoroSession.SelectVoice(const VoiceName: string;
  CheckText: boolean);
var
  NewVoice: TKokoroVoicePack;
begin
  if (VoiceName = '') or (Pos('/', VoiceName) > 0) or
    (Pos('\', VoiceName) > 0) or (Pos('..', VoiceName) > 0) then
    raise EPretrainedImportError.Create('Kokoro: "' + VoiceName +
      '" is not a voice name (voices/<name>.pt of the model folder).');
  NewVoice := LoadKokoroVoicePack(KokoroVoiceFileName(FOptions.ModelDir,
    VoiceName), FKokoro.Config.StyleDim);
  if CheckText then FKokoro.EnglishG2PForVoice(NewVoice);
  FVoice := NewVoice;
end;

function TKokoroSession.FinishLine(Audio: TNNetVolume; const FileName: string;
  const Report: TKokoroTextReport; SynthesisMs: QWord): TKokoroLineResult;
var
  StartTime: QWord;
  ChunkPos: integer;
begin
  StartTime := GetTickCount64();
  SaveVolumeToWav16(Audio, FileName, KokoroSampleRate);
  Result.WriteMs := GetTickCount64() - StartTime;
  Result.FileName := FileName;
  Result.Report := Report;
  Result.PhonemeCount := 0;
  for ChunkPos := 0 to High(Report) do
    Inc(Result.PhonemeCount, CodePointCount(Report[ChunkPos].G2PChunk.Phonemes));
  Result.SampleCount := Audio.Size;
  Result.SynthesisMs := SynthesisMs;
  FLastLine := Result;
  FHasLastLine := true;
end;

function TKokoroSession.SpeakTextToWav(const Text, FileName: string):
  TKokoroLineResult;
var
  Audio: TNNetVolume;
  Report: TKokoroTextReport;
  StartTime: QWord;
begin
  Audio := TNNetVolume.Create();
  try
    StartTime := GetTickCount64();
    FKokoro.SynthesizeText(Text, FVoice, FSpeed, Audio, Report);
    Result := FinishLine(Audio, FileName, Report,
      GetTickCount64() - StartTime);
  finally
    Audio.Free;
  end;
end;

function TKokoroSession.SpeakPhonemesToWav(const Phonemes, FileName: string):
  TKokoroLineResult;
var
  Audio: TNNetVolume;
  Report: TKokoroTextReport;
  StartTime: QWord;
begin
  Audio := TNNetVolume.Create();
  try
    StartTime := GetTickCount64();
    SetLength(Report, 1);
    Report[0].G2PChunk := Default(TNeuralG2PChunk);
    Report[0].G2PChunk.Phonemes := Phonemes;
    FKokoro.SynthesizePhonemes(Phonemes, FVoice, FSpeed, Audio,
      Report[0].SkippedSymbols);
    Report[0].FirstSample := 0;
    Report[0].SampleCount := Audio.Size;
    Result := FinishLine(Audio, FileName, Report,
      GetTickCount64() - StartTime);
  finally
    Audio.Free;
  end;
end;

end.
