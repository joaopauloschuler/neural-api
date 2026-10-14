program KokoroTTS;
(*
KokoroTTS: Kokoro-82M text-to-speech (hexgrad/Kokoro-82M, Apache-2.0,
StyleTTS2 + iSTFTNet) from a Hugging Face snapshot folder (config.json,
kokoro-v1_0.pth, voices/<name>.pt), through TNNetKokoro
(neural/neuralpretrained.pas). Writes 16-bit 24 kHz mono WAVs.

  KokoroTTS <model-dir> [options]

  text -> English G2P (neuralg2p, misaki's dictionaries) -> chunks
    -> per chunk: stage A (PL-BERT, text encoder, durations)
    -> stage B (F0 and energy curves) -> stage-C decoder
    -> harmonic source + STFT (host) -> stage-C generator -> iSTFT (host)
    -> chunks joined -> WAV.

The shared part (loading the folder, voice switching, text -> audio -> WAV)
is TKokoroSession in neural/neuralkokorosession.pas, so a later TTS server
can reuse it. This file is the terminal part: argument parsing, the REPL,
/commands and the external programs run after each WAV.

REPL (default): the model is loaded once; each typed line is spoken into the
next output file. Commands: /exit, /help, /voice <name>, /speed <x>,
/out <pattern>, /phonemes <IPA>, /show.

-p "text" speaks one line and exits without reading stdin (exit code 0, or
1 on any error, including a failing --play-cmd / --encode-cmd).
--phonemes "IPA" does the same with a phoneme string (no G2P).

Output names: --out <pattern> (default kokoro_%03d.wav). %d, %Nd and %0Nd
take the line counter (1 for the first line written, +1 per WAV written);
%% is a literal %. A pattern without a counter is used as is, so every line
overwrites that one file (-p "Hi" --out hello.wav writes hello.wav).

External programs (--play-cmd, --encode-cmd) run after each WAV, without a
shell: SplitCommandTemplate splits the template into arguments, then
ExpandCommandArguments replaces {wav} and {base} inside each argument.

--stats prints timing to stderr. --selftest runs the argument-parsing and
REPL-command checks (no model needed) and exits.

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
  SysUtils, Classes, process,
  neuralvolume, neuralpretrained, neuralg2p, neuralkokorosession;

type
  TKokoroArgs = record
    Session: TKokoroSessionOptions;
    Prompt: string;           // -p: one line of text, then exit
    Phonemes: string;         // --phonemes: one phoneme string, then exit
    OutPattern: string;       // --out
    PlayCommand: string;      // --play-cmd template ('' = none)
    EncodeCommand: string;    // --encode-cmd template ('' = none)
    KeepWav: boolean;         // --keep-wav true|false
    Stats: boolean;
    SelfTest: boolean;
    ShowHelp: boolean;
    ErrorMsg: string;
  end;

  TKokoroReplCommand = (krcText, krcExit, krcHelp, krcVoice, krcSpeed, krcOut,
    krcPhonemes, krcShow);

const
  csDefaultOutPattern = 'kokoro_%03d.wav';

procedure PrintUsage();
begin
  WriteLn('Usage: KokoroTTS <model-dir> [options]');
  WriteLn;
  WriteLn('<model-dir> is a hexgrad/Kokoro-82M snapshot folder: config.json, ',
    'kokoro-v1_0.pth,');
  WriteLn('voices/<name>.pt and (for text) misaki/{us,gb}_{gold,silver}.json.');
  WriteLn;
  WriteLn('  -p "text"            speak one line, write one WAV and exit ',
    '(stdin is not read)');
  WriteLn('  --phonemes "IPA"     the same with a phoneme string (expert; no ',
    'G2P), e.g.');
  WriteLn('                       "h', #$C9#$99, 'l', #$CB#$88, 'O w', #$CB#$88,
    #$C9#$9C, #$C9#$B9, 'ld."');
  WriteLn('  --dict-dir DIR       misaki''s dictionaries (default ',
    '<model-dir>/misaki)');
  WriteLn('  --voice NAME         voices/NAME.pt (default af_heart; a* US ',
    'English, b* British)');
  WriteLn('  --speed S            speaking rate, > 0 (default 1)');
  WriteLn('  --out PATTERN        output WAV names (default ', csDefaultOutPattern,
    '); %d/%03d = line');
  WriteLn('                       counter, %% = %; no counter = one file, ',
    'overwritten');
  WriteLn('  --play-cmd "CMD"     run after each WAV, e.g. "aplay {wav}"');
  WriteLn('  --encode-cmd "CMD"   run after each WAV, e.g. ',
    '"lame --quiet {wav} {base}.mp3"');
  WriteLn('                       {wav} = WAV path, {base} = it without .wav; ',
    'no shell');
  WriteLn('  --keep-wav BOOL      false deletes the WAV after a successful ',
    'encode (default true)');
  WriteLn('  --stats              timing to stderr (load; per line: chunks, ',
    'phonemes, audio,');
  WriteLn('                       synthesis time, real-time factor)');
  WriteLn('  --serial             single-threaded forward passes');
  WriteLn('  --max-threads N      cap the parallel forward at N worker threads ',
    '(default: every CPU thread)');
  WriteLn('  --gpu                run the stage nets on OpenCL (needs a build ',
    'with -dOpenCL; default CPU)');
  WriteLn('  --gpu-platform N     OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N       OpenCL device index within the platform ',
    '(default 0)');
  WriteLn('  --no-gpu-shared-kernel  give each layer private OpenCL kernels');
  WriteLn('  --selftest           run the offline checks and exit');
  WriteLn;
  WriteLn('REPL commands: /exit, /help, /voice <name>, /speed <x>, ',
    '/out <pattern>,');
  WriteLn('               /phonemes <IPA>, /show');
end;

procedure PrintReplHelp();
begin
  WriteLn('Type English text; each line is spoken into the next WAV.');
  WriteLn('  /voice <name>    switch to voices/<name>.pt (no name: show the ',
    'current voice)');
  WriteLn('  /speed <x>       speaking rate, > 0 (no value: show it)');
  WriteLn('  /out <pattern>   output names, e.g. take_%02d.wav (no pattern: ',
    'show it)');
  WriteLn('  /phonemes <IPA>  speak this phoneme string once (no G2P)');
  WriteLn('  /show            chunks, phonemes and unknown words of the last ',
    'line');
  WriteLn('  /help            this list');
  WriteLn('  /exit            quit');
end;

// Locale-independent float ('.' decimal separator).
function TryParseFloat(const S: string; out Value: TNeuralFloat): boolean;
var
  Parsed: double;
  Code: integer;
begin
  Val(S, Parsed, Code);
  Result := (Code = 0) and (S <> '');
  if Result then Value := Parsed;
end;

// Expands Pattern for line Counter: %d / %Nd / %0Nd -> Counter, %% -> %.
// Any other % sequence, or an empty result, is an error.
function ExpandOutputPattern(const Pattern: string; Counter: integer;
  out FileName, ErrorMsg: string): boolean;
var
  CharPos, Width: integer;
  ZeroPad: boolean;
  Digits: string;
begin
  FileName := '';
  ErrorMsg := '';
  CharPos := 1;
  while CharPos <= Length(Pattern) do
  begin
    if Pattern[CharPos] <> '%' then
    begin
      FileName := FileName + Pattern[CharPos];
      Inc(CharPos);
      continue;
    end;
    Inc(CharPos);
    if (CharPos <= Length(Pattern)) and (Pattern[CharPos] = '%') then
    begin
      FileName := FileName + '%';
      Inc(CharPos);
      continue;
    end;
    ZeroPad := (CharPos <= Length(Pattern)) and (Pattern[CharPos] = '0');
    if ZeroPad then Inc(CharPos);
    Width := 0;
    while (CharPos <= Length(Pattern)) and (Pattern[CharPos] in ['0'..'9']) and
      (Width < 100) do
    begin
      Width := Width * 10 + Ord(Pattern[CharPos]) - Ord('0');
      Inc(CharPos);
    end;
    if (CharPos > Length(Pattern)) or (Pattern[CharPos] <> 'd') or
      (Width > 9) then
    begin
      ErrorMsg := 'output pattern "' + Pattern + '": only %d, %Nd, %0Nd ' +
        '(N up to 9) and %% are allowed';
      exit(false);
    end;
    Inc(CharPos);
    Digits := IntToStr(Counter);
    while Length(Digits) < Width do
      if ZeroPad then Digits := '0' + Digits else Digits := ' ' + Digits;
    FileName := FileName + Digits;
  end;
  if FileName = '' then
  begin
    ErrorMsg := 'output pattern is empty';
    exit(false);
  end;
  Result := true;
end;

// Splits a command template into arguments without a shell: spaces and tabs
// separate; '...' and "..." group (quotes removed, no escapes); "" is an empty
// argument. False on an unbalanced quote or an empty command.
function SplitCommandTemplate(const Template: string; Arguments: TStrings;
  out ErrorMsg: string): boolean;
var
  CharPos: integer;
  Current: string;
  InWord: boolean;
  Quote: char;
begin
  Arguments.Clear;
  ErrorMsg := '';
  Current := '';
  InWord := false;
  Quote := #0;
  for CharPos := 1 to Length(Template) do
  begin
    if Quote <> #0 then
    begin
      if Template[CharPos] = Quote then Quote := #0
      else Current := Current + Template[CharPos];
    end
    else if Template[CharPos] in [' ', #9] then
    begin
      if InWord then Arguments.Add(Current);
      Current := '';
      InWord := false;
    end
    else if Template[CharPos] in ['''', '"'] then
    begin
      Quote := Template[CharPos];
      InWord := true;
    end
    else
    begin
      Current := Current + Template[CharPos];
      InWord := true;
    end;
  end;
  if Quote <> #0 then
  begin
    ErrorMsg := 'command "' + Template + '": unbalanced ' + Quote + ' quote';
    exit(false);
  end;
  if InWord then Arguments.Add(Current);
  if (Arguments.Count = 0) or (Arguments[0] = '') then
  begin
    ErrorMsg := 'command "' + Template + '": no program name';
    exit(false);
  end;
  Result := true;
end;

// The WAV path without a final .wav (any case); the whole path otherwise.
function WavBaseName(const WavFile: string): string;
begin
  if LowerCase(ExtractFileExt(WavFile)) = '.wav' then
    Result := Copy(WavFile, 1, Length(WavFile) - 4)
  else Result := WavFile;
end;

// Replaces {wav} and {base} inside each argument in one pass (a substituted
// path is not scanned again).
procedure ExpandCommandArguments(Template, Expanded: TStrings;
  const WavFile: string);
var
  ArgPos, CharPos: integer;
  Source, Target: string;
begin
  Expanded.Clear;
  for ArgPos := 0 to Template.Count - 1 do
  begin
    Source := Template[ArgPos];
    Target := '';
    CharPos := 1;
    while CharPos <= Length(Source) do
    begin
      if Copy(Source, CharPos, 5) = '{wav}' then
      begin
        Target := Target + WavFile;
        Inc(CharPos, 5);
      end
      else if Copy(Source, CharPos, 6) = '{base}' then
      begin
        Target := Target + WavBaseName(WavFile);
        Inc(CharPos, 6);
      end
      else
      begin
        Target := Target + Source[CharPos];
        Inc(CharPos);
      end;
    end;
    Expanded.Add(Target);
  end;
end;

function ParseKokoroArgs(Args: TStrings; out Opt: TKokoroArgs): boolean;
var
  ArgPos: integer;
  Arg, Value, Ignored: string;
  Parts: TStringList;

  function NextValue(out FlagValue: string): boolean;
  begin
    if ArgPos + 1 >= Args.Count then
    begin
      Opt.ErrorMsg := Arg + ' needs a value';
      exit(false);
    end;
    Inc(ArgPos);
    FlagValue := Args[ArgPos];
    Result := true;
  end;

  function NextInt(out IntValue: integer): boolean;
  begin
    Result := NextValue(Value);
    if not Result then exit;
    Result := TryStrToInt(Value, IntValue);
    if not Result then Opt.ErrorMsg := Arg + ': not an integer: ' + Value;
  end;

  function CheckCommand(const Template: string): boolean;
  begin
    Result := SplitCommandTemplate(Template, Parts, Opt.ErrorMsg);
    if not Result then Opt.ErrorMsg := Arg + ': ' + Opt.ErrorMsg;
  end;

var
  IntValue: integer;
  FloatValue: TNeuralFloat;
  KeepWavSet: boolean;
begin
  Opt := Default(TKokoroArgs);
  Opt.Session := DefaultKokoroSessionOptions();
  Opt.OutPattern := csDefaultOutPattern;
  Opt.KeepWav := true;
  KeepWavSet := false;
  Result := false;
  Parts := TStringList.Create;
  try
    ArgPos := 0;
    while ArgPos < Args.Count do
    begin
      Arg := Args[ArgPos];
      if Arg = '-p' then
      begin
        if not NextValue(Opt.Prompt) then exit;
        if Trim(Opt.Prompt) = '' then
        begin
          Opt.ErrorMsg := '-p needs some text';
          exit;
        end;
      end
      else if Arg = '--phonemes' then
      begin
        if not NextValue(Opt.Phonemes) then exit;
        if Trim(Opt.Phonemes) = '' then
        begin
          Opt.ErrorMsg := '--phonemes needs a phoneme string';
          exit;
        end;
      end
      else if Arg = '--dict-dir' then
      begin
        if not NextValue(Opt.Session.DictionaryDir) then exit;
      end
      else if Arg = '--voice' then
      begin
        if not NextValue(Opt.Session.VoiceName) then exit;
      end
      else if Arg = '--speed' then
      begin
        if not NextValue(Value) then exit;
        if not TryParseFloat(Value, FloatValue) or not (FloatValue > 0) then
        begin
          Opt.ErrorMsg := '--speed: must be a number greater than 0: ' + Value;
          exit;
        end;
        Opt.Session.Speed := FloatValue;
      end
      else if Arg = '--out' then
      begin
        if not NextValue(Opt.OutPattern) then exit;
        if not ExpandOutputPattern(Opt.OutPattern, 1, Ignored,
          Opt.ErrorMsg) then exit;
      end
      else if Arg = '--play-cmd' then
      begin
        if not NextValue(Opt.PlayCommand) or
          not CheckCommand(Opt.PlayCommand) then exit;
      end
      else if Arg = '--encode-cmd' then
      begin
        if not NextValue(Opt.EncodeCommand) or
          not CheckCommand(Opt.EncodeCommand) then exit;
      end
      else if Arg = '--keep-wav' then
      begin
        if not NextValue(Value) then exit;
        if LowerCase(Value) = 'true' then Opt.KeepWav := true
        else if LowerCase(Value) = 'false' then Opt.KeepWav := false
        else
        begin
          Opt.ErrorMsg := '--keep-wav: true or false, not ' + Value;
          exit;
        end;
        KeepWavSet := true;
      end
      else if Arg = '--stats' then Opt.Stats := true
      else if Arg = '--serial' then Opt.Session.Serial := true
      else if Arg = '--max-threads' then
      begin
        if not NextInt(IntValue) then exit;
        if IntValue < 1 then
        begin
          Opt.ErrorMsg := '--max-threads: must be at least 1';
          exit;
        end;
        Opt.Session.MaxThreads := IntValue;
      end
      else if Arg = '--gpu' then Opt.Session.Gpu := true
      else if Arg = '--gpu-platform' then
      begin
        if not NextInt(Opt.Session.GpuPlatform) then exit;
      end
      else if Arg = '--gpu-device' then
      begin
        if not NextInt(Opt.Session.GpuDevice) then exit;
      end
      else if Arg = '--no-gpu-shared-kernel' then
        Opt.Session.GpuSharedKernel := false
      else if Arg = '--selftest' then Opt.SelfTest := true
      else if (Arg = '--help') or (Arg = '-h') then Opt.ShowHelp := true
      else if Arg = '--model-dir' then
      begin
        Opt.ErrorMsg := '--model-dir was removed: pass the model folder as ' +
          'the first argument (KokoroTTS <model-dir> [options])';
        exit;
      end
      else if (Arg <> '') and (Arg[1] = '-') then
      begin
        Opt.ErrorMsg := 'unknown option: ' + Arg;
        exit;
      end
      else if Opt.Session.ModelDir = '' then Opt.Session.ModelDir := Arg
      else
      begin
        Opt.ErrorMsg := 'unexpected argument: ' + Arg + ' (the model folder ' +
          'is already ' + Opt.Session.ModelDir + ')';
        exit;
      end;
      Inc(ArgPos);
    end;
    if (Opt.Prompt <> '') and (Opt.Phonemes <> '') then
    begin
      Opt.ErrorMsg := '-p and --phonemes cannot be combined';
      exit;
    end;
    if KeepWavSet and not Opt.KeepWav and (Opt.EncodeCommand = '') then
    begin
      Opt.ErrorMsg := '--keep-wav false needs --encode-cmd (the WAV is ' +
        'deleted only after a successful encode)';
      exit;
    end;
    Result := true;
  finally
    Parts.Free;
  end;
end;

// Classifies one REPL line (already trimmed). Plain text is krcText with Arg
// = the line. False with ErrorMsg for an unknown command or a bad argument.
function ParseKokoroReplLine(const Line: string;
  out Command: TKokoroReplCommand; out Arg, ErrorMsg: string): boolean;
var
  SpacePos: integer;
  Name, Ignored: string;
  Speed: TNeuralFloat;
begin
  Command := krcText;
  Arg := Line;
  ErrorMsg := '';
  Result := true;
  if (Line = '') or (Line[1] <> '/') then exit;
  SpacePos := Pos(' ', Line);
  if SpacePos = 0 then
  begin
    Name := LowerCase(Copy(Line, 2, MaxInt));
    Arg := '';
  end
  else
  begin
    Name := LowerCase(Copy(Line, 2, SpacePos - 2));
    Arg := Trim(Copy(Line, SpacePos + 1, MaxInt));
  end;
  if Name = 'exit' then Command := krcExit
  else if Name = 'help' then Command := krcHelp
  else if Name = 'show' then Command := krcShow
  else if Name = 'voice' then Command := krcVoice
  else if Name = 'speed' then
  begin
    Command := krcSpeed;
    if (Arg <> '') and (not TryParseFloat(Arg, Speed) or not (Speed > 0)) then
    begin
      ErrorMsg := '/speed needs a number greater than 0, not ' + Arg;
      exit(false);
    end;
  end
  else if Name = 'out' then
  begin
    Command := krcOut;
    if (Arg <> '') and not ExpandOutputPattern(Arg, 1, Ignored, ErrorMsg) then
      exit(false);
  end
  else if Name = 'phonemes' then
  begin
    Command := krcPhonemes;
    if Arg = '' then
    begin
      ErrorMsg := '/phonemes needs a phoneme string';
      exit(false);
    end;
  end
  else
  begin
    ErrorMsg := 'unknown command /' + Name + ' (/help lists the commands)';
    exit(false);
  end;
end;

// Runs Arguments[0] with the rest as its arguments (no shell), waiting for
// it. False with ErrorMsg when it cannot start or exits with a non-zero code.
function RunExternalCommand(Arguments: TStrings; out ErrorMsg: string): boolean;
var
  Child: TProcess;
  ArgPos: integer;
begin
  ErrorMsg := '';
  Child := TProcess.Create(nil);
  try
    Child.Executable := Arguments[0];
    for ArgPos := 1 to Arguments.Count - 1 do
      Child.Parameters.Add(Arguments[ArgPos]);
    Child.Options := [poWaitOnExit];
    try
      Child.Execute;
    except
      on E: Exception do
      begin
        ErrorMsg := 'could not start ' + Arguments[0] + ': ' + E.Message;
        exit(false);
      end;
    end;
    // After poWaitOnExit, ExitStatus is the exit code (FPC on Unix: minus
    // the wait status when a signal ended the program).
    Result := Child.ExitStatus = 0;
    if Child.ExitStatus > 0 then
      ErrorMsg := Arguments[0] + ' failed with exit code ' +
        IntToStr(Child.ExitStatus)
    else if Child.ExitStatus < 0 then
      ErrorMsg := Arguments[0] + ' was ended by a signal (wait status ' +
        IntToStr(-Child.ExitStatus) + ')';
  finally
    Child.Free;
  end;
end;

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

// ---------------------------------------------------------------------------
// --selftest: offline checks (no model files needed).
// ---------------------------------------------------------------------------
procedure SelfTest();
var
  Failures: integer;
  Args, Parts, Expanded: TStringList;
  Opt: TKokoroArgs;
  FileName, ErrorMsg, Arg: string;
  Command: TKokoroReplCommand;

  procedure Check(Condition: boolean; const What: string);
  begin
    if Condition then WriteLn('PASS: ', What)
    else
    begin
      WriteLn('FAIL: ', What);
      Inc(Failures);
    end;
  end;

  // Parses the space-separated words of Line (no quoting) as a command line.
  function ParseWords(const Line: string): boolean;
  begin
    Args.Clear;
    Args.Delimiter := ' ';
    Args.StrictDelimiter := true;
    if Line <> '' then Args.DelimitedText := Line;
    Result := ParseKokoroArgs(Args, Opt);
  end;

begin
  Failures := 0;
  Args := TStringList.Create;
  Parts := TStringList.Create;
  Expanded := TStringList.Create;
  try
    // Positional model folder and the defaults.
    Check(ParseWords('/tmp/kokoro'), 'model folder alone parses');
    Check(Opt.Session.ModelDir = '/tmp/kokoro', 'model folder is the positional argument');
    Check((Opt.Prompt = '') and (Opt.Phonemes = ''), 'REPL by default (no -p, no --phonemes)');
    Check(Opt.OutPattern = 'kokoro_%03d.wav', 'default output pattern');
    Check(Opt.Session.DictionaryDir = '', 'dictionary folder defaults to <model-dir>/misaki');
    Check(Opt.Session.VoiceName = 'af_heart', 'default voice af_heart');
    Check(Abs(Opt.Session.Speed - 1) < 1e-6, 'default speed 1');
    Check(Opt.KeepWav and (Opt.PlayCommand = '') and (Opt.EncodeCommand = ''),
      'no external programs by default, WAV kept');
    Check(not Opt.Stats and not Opt.Session.Gpu and not Opt.Session.Serial and
      (Opt.Session.MaxThreads = 0) and Opt.Session.GpuSharedKernel,
      'stats off, CPU, parallel, every thread, shared OpenCL kernels');
    Check(ParseWords('--voice bf_emma /tmp/kokoro') and
      (Opt.Session.ModelDir = '/tmp/kokoro') and (Opt.Session.VoiceName = 'bf_emma'),
      'model folder after an option');
    Check(ParseWords('--selftest') and Opt.SelfTest and (Opt.Session.ModelDir = ''),
      '--selftest needs no model folder');
    Check(ParseWords('-h') and Opt.ShowHelp, '-h parses');
    Check(not ParseWords('/tmp/a /tmp/b'), 'a second positional argument rejected');
    Check(not ParseWords('--model-dir /tmp/kokoro') and
      (Pos('first argument', Opt.ErrorMsg) > 0), '--model-dir removed, with a hint');
    Check(not ParseWords('/tmp/kokoro --bogus'), 'unknown option rejected');

    // Every option with a value.
    Check(ParseWords('/tmp/kokoro --voice am_adam --speed 1.25 --dict-dir /d ' +
      '--serial --max-threads 3 --gpu --gpu-platform 1 --gpu-device 2 ' +
      '--no-gpu-shared-kernel --stats'), 'full option set parses');
    Check(Opt.Session.VoiceName = 'am_adam', '--voice');
    Check(Abs(Opt.Session.Speed - 1.25) < 1e-6, '--speed');
    Check(Opt.Session.DictionaryDir = '/d', '--dict-dir');
    Check(Opt.Session.Serial and (Opt.Session.MaxThreads = 3), '--serial, --max-threads');
    Check(Opt.Session.Gpu and (Opt.Session.GpuPlatform = 1) and
      (Opt.Session.GpuDevice = 2) and not Opt.Session.GpuSharedKernel,
      '--gpu, --gpu-platform, --gpu-device, --no-gpu-shared-kernel');
    Check(Opt.Stats, '--stats');
    Check(not ParseWords('/tmp/kokoro --speed 0'), '--speed 0 rejected');
    Check(not ParseWords('/tmp/kokoro --speed fast'), '--speed text rejected');
    Check(not ParseWords('/tmp/kokoro --max-threads 0'), '--max-threads 0 rejected');
    Check(not ParseWords('/tmp/kokoro --voice'), 'option without a value rejected');

    // -p and --phonemes (one-shot modes).
    Args.Clear;
    Args.Add('/tmp/kokoro'); Args.Add('-p'); Args.Add('Hello there.');
    Check(ParseKokoroArgs(Args, Opt) and (Opt.Prompt = 'Hello there.'), '-p parses');
    Args.Clear;
    Args.Add('/tmp/kokoro'); Args.Add('--phonemes'); Args.Add('h' + #$C9#$99 + 'lO.');
    Check(ParseKokoroArgs(Args, Opt) and (Opt.Phonemes = 'h' + #$C9#$99 + 'lO.'),
      '--phonemes parses');
    Args.Clear;
    Args.Add('/tmp/kokoro'); Args.Add('-p'); Args.Add('Hi.');
    Args.Add('--phonemes'); Args.Add('hI.');
    Check(not ParseKokoroArgs(Args, Opt), '-p with --phonemes rejected');
    Args.Clear;
    Args.Add('/tmp/kokoro'); Args.Add('-p'); Args.Add('  ');
    Check(not ParseKokoroArgs(Args, Opt), '-p with blank text rejected');
    Check(not ParseWords('/tmp/kokoro -p'), '-p without a value rejected');

    // --out patterns.
    Check(ParseWords('/tmp/kokoro --out take_%02d.wav') and
      (Opt.OutPattern = 'take_%02d.wav'), '--out parses');
    Check(ExpandOutputPattern('kokoro_%03d.wav', 7, FileName, ErrorMsg) and
      (FileName = 'kokoro_007.wav'), '%03d zero-pads the counter');
    Check(ExpandOutputPattern('line%d.wav', 12, FileName, ErrorMsg) and
      (FileName = 'line12.wav'), '%d is the plain counter');
    Check(ExpandOutputPattern('l%3d.wav', 5, FileName, ErrorMsg) and
      (FileName = 'l  5.wav'), '%3d pads with spaces');
    Check(ExpandOutputPattern('hello.wav', 9, FileName, ErrorMsg) and
      (FileName = 'hello.wav'), 'a pattern without a counter is used as is');
    Check(ExpandOutputPattern('100%%_%d.wav', 2, FileName, ErrorMsg) and
      (FileName = '100%_2.wav'), '%% is a literal percent');
    Check(not ExpandOutputPattern('out_%s.wav', 1, FileName, ErrorMsg),
      '%s rejected');
    Check(not ExpandOutputPattern('out_%', 1, FileName, ErrorMsg),
      'a trailing % rejected');
    Check(not ExpandOutputPattern('', 1, FileName, ErrorMsg), 'empty pattern rejected');
    Check(not ParseWords('/tmp/kokoro --out x_%x.wav'), '--out with a bad pattern rejected');

    // External command templates: splitting and placeholders.
    Check(SplitCommandTemplate('aplay {wav}', Parts, ErrorMsg) and
      (Parts.Count = 2) and (Parts[0] = 'aplay') and (Parts[1] = '{wav}'),
      'command split at spaces');
    Check(SplitCommandTemplate('  lame   --quiet'#9'{wav} {base}.mp3 ', Parts,
      ErrorMsg) and (Parts.Count = 4) and (Parts[3] = '{base}.mp3'),
      'runs of spaces and tabs separate once');
    Check(SplitCommandTemplate('play "{wav}" --title=''a "b" c''', Parts,
      ErrorMsg) and (Parts.Count = 3) and (Parts[1] = '{wav}') and
      (Parts[2] = '--title=a "b" c'), 'quotes group and are removed');
    Check(SplitCommandTemplate('cmd "" x', Parts, ErrorMsg) and
      (Parts.Count = 3) and (Parts[1] = ''), '"" is an empty argument');
    Check(SplitCommandTemplate('echo a|b>c;$HOME', Parts, ErrorMsg) and
      (Parts.Count = 2) and (Parts[1] = 'a|b>c;$HOME'),
      'shell characters are literal (no shell)');
    Check(not SplitCommandTemplate('aplay "{wav}', Parts, ErrorMsg),
      'unbalanced quote rejected');
    Check(not SplitCommandTemplate('   ', Parts, ErrorMsg), 'empty command rejected');
    SplitCommandTemplate('lame --quiet {wav} {base}.mp3', Parts, ErrorMsg);
    ExpandCommandArguments(Parts, Expanded, 'my dir/kokoro_001.wav');
    Check((Expanded.Count = 4) and (Expanded[2] = 'my dir/kokoro_001.wav') and
      (Expanded[3] = 'my dir/kokoro_001.mp3'),
      '{wav} and {base} replaced per argument (a space stays inside)');
    SplitCommandTemplate('cp {wav} {wav}.{base}', Parts, ErrorMsg);
    ExpandCommandArguments(Parts, Expanded, 'a.WAV');
    Check(Expanded[2] = 'a.WAV.a', 'two placeholders in one argument; .WAV any case');
    ExpandCommandArguments(Parts, Expanded, '{base}.wav');
    Check(Expanded[1] = '{base}.wav', 'a substituted path is not expanded again');
    Check(WavBaseName('out.flac') = 'out.flac', '{base} of a non-.wav name is the name');
    Args.Clear;
    Args.Add('/tmp/kokoro');
    Args.Add('--play-cmd'); Args.Add('aplay -q {wav}');
    Args.Add('--encode-cmd'); Args.Add('lame --quiet {wav} {base}.mp3');
    Args.Add('--keep-wav'); Args.Add('false');
    Check(ParseKokoroArgs(Args, Opt) and (Opt.PlayCommand = 'aplay -q {wav}') and
      (Opt.EncodeCommand = 'lame --quiet {wav} {base}.mp3') and not Opt.KeepWav,
      '--play-cmd, --encode-cmd, --keep-wav false parse');
    Args.Clear;
    Args.Add('/tmp/kokoro'); Args.Add('--play-cmd'); Args.Add('aplay "{wav}');
    Check(not ParseKokoroArgs(Args, Opt), '--play-cmd with an unbalanced quote rejected');
    Check(ParseWords('/tmp/kokoro --keep-wav TRUE') and Opt.KeepWav,
      '--keep-wav true (any case)');
    Check(not ParseWords('/tmp/kokoro --keep-wav no'), '--keep-wav other value rejected');
    Check(not ParseWords('/tmp/kokoro --keep-wav false'),
      '--keep-wav false without --encode-cmd rejected');

    // REPL lines.
    Check(ParseKokoroReplLine('Hello world.', Command, Arg, ErrorMsg) and
      (Command = krcText) and (Arg = 'Hello world.'), 'plain text is spoken');
    Check(ParseKokoroReplLine('/exit', Command, Arg, ErrorMsg) and
      (Command = krcExit), '/exit');
    Check(ParseKokoroReplLine('/EXIT', Command, Arg, ErrorMsg) and
      (Command = krcExit), 'command names ignore case');
    Check(ParseKokoroReplLine('/help', Command, Arg, ErrorMsg) and
      (Command = krcHelp), '/help');
    Check(ParseKokoroReplLine('/show', Command, Arg, ErrorMsg) and
      (Command = krcShow), '/show');
    Check(ParseKokoroReplLine('/voice  bf_emma ', Command, Arg, ErrorMsg) and
      (Command = krcVoice) and (Arg = 'bf_emma'), '/voice <name>');
    Check(ParseKokoroReplLine('/voice', Command, Arg, ErrorMsg) and
      (Command = krcVoice) and (Arg = ''), '/voice alone shows the voice');
    Check(ParseKokoroReplLine('/speed 0.8', Command, Arg, ErrorMsg) and
      (Command = krcSpeed) and (Arg = '0.8'), '/speed <x>');
    Check(ParseKokoroReplLine('/speed', Command, Arg, ErrorMsg) and
      (Command = krcSpeed) and (Arg = ''), '/speed alone shows the speed');
    Check(not ParseKokoroReplLine('/speed -1', Command, Arg, ErrorMsg),
      '/speed -1 rejected');
    Check(not ParseKokoroReplLine('/speed fast', Command, Arg, ErrorMsg),
      '/speed text rejected');
    Check(ParseKokoroReplLine('/out take_%02d.wav', Command, Arg, ErrorMsg) and
      (Command = krcOut) and (Arg = 'take_%02d.wav'), '/out <pattern>');
    Check(not ParseKokoroReplLine('/out x%q', Command, Arg, ErrorMsg),
      '/out with a bad pattern rejected');
    Check(ParseKokoroReplLine('/phonemes h' + #$C9#$99 + 'lO.', Command, Arg,
      ErrorMsg) and (Command = krcPhonemes) and (Arg = 'h' + #$C9#$99 + 'lO.'),
      '/phonemes <IPA>');
    Check(not ParseKokoroReplLine('/phonemes', Command, Arg, ErrorMsg),
      '/phonemes without phonemes rejected');
    Check(not ParseKokoroReplLine('/bogus', Command, Arg, ErrorMsg) and
      (Pos('/help', ErrorMsg) > 0), 'unknown command rejected, pointing to /help');
  finally
    Expanded.Free;
    Parts.Free;
    Args.Free;
  end;
  if Failures = 0 then WriteLn('SELFTEST OK')
  else
  begin
    WriteLn('SELFTEST FAILED: ', Failures, ' check(s)');
    Halt(1);
  end;
end;

// ---------------------------------------------------------------------------
// Main: load once, then one line (-p / --phonemes) or the REPL.
// ---------------------------------------------------------------------------
var
  Opt: TKokoroArgs;
  Args, PlayTemplate, EncodeTemplate: TStringList;
  Session: TKokoroSession;
  OutPattern: string;
  LineCounter: integer;

// The external programs for WavFile: play, then encode, then (with
// --keep-wav false) delete the WAV after a successful encode.
function RunAfterWav(const WavFile: string): boolean;
var
  Expanded: TStringList;
  ErrorMsg: string;
begin
  Result := true;
  // The child writes to the same terminal: print our lines first.
  Flush(System.Output);
  Expanded := TStringList.Create;
  try
    if PlayTemplate.Count > 0 then
    begin
      ExpandCommandArguments(PlayTemplate, Expanded, WavFile);
      if not RunExternalCommand(Expanded, ErrorMsg) then
      begin
        WriteLn('Play command: ', ErrorMsg);
        Result := false;
      end;
    end;
    if EncodeTemplate.Count > 0 then
    begin
      ExpandCommandArguments(EncodeTemplate, Expanded, WavFile);
      if not RunExternalCommand(Expanded, ErrorMsg) then
      begin
        WriteLn('Encode command: ', ErrorMsg, '; ', WavFile, ' kept');
        Result := false;
      end
      else if not Opt.KeepWav then
      begin
        if DeleteFile(WavFile) then WriteLn('Deleted    : ', WavFile, ' (encoded)')
        else WriteLn('Could not delete ', WavFile);
      end;
    end;
  finally
    Expanded.Free;
  end;
end;

// Speaks one line (text, or phonemes when IsPhonemes) into the next output
// file, prints the result and runs the external programs. False on any error.
function SpeakLine(const Line: string; IsPhonemes: boolean): boolean;
var
  FileName, ErrorMsg: string;
  LineResult: TKokoroLineResult;
  AudioSeconds: double;
begin
  Result := false;
  if not ExpandOutputPattern(OutPattern, LineCounter, FileName, ErrorMsg) then
  begin
    WriteLn('Error: ', ErrorMsg);
    exit;
  end;
  try
    if IsPhonemes then LineResult := Session.SpeakPhonemesToWav(Line, FileName)
    else LineResult := Session.SpeakTextToWav(Line, FileName);
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      exit;
    end;
  end;
  Inc(LineCounter);
  AudioSeconds := LineResult.SampleCount / KokoroSampleRate;
  WriteLn('Wrote      : ', FileName, ' (', Format('%.2f', [AudioSeconds]),
    ' s of audio, ',
    Length(LineResult.Report), ' chunk(s))');
  if Opt.Stats then
  begin
    Write(StdErr, '[stats] chunks ', Length(LineResult.Report), ', phonemes ',
      LineResult.PhonemeCount, ', audio ', Format('%.2f', [AudioSeconds]),
      ' s, synthesis ',
      (LineResult.SynthesisMs / 1000):0:2, ' s');
    if AudioSeconds > 0 then
      Write(StdErr, ', real-time factor ',
        (LineResult.SynthesisMs / 1000 / AudioSeconds):0:3);
    WriteLn(StdErr, ', WAV write ', (LineResult.WriteMs / 1000):0:2,
      ' s, peak RSS ', PeakRssMegabytes(), ' MB');
  end;
  Result := RunAfterWav(FileName);
end;

procedure ShowLastLine();
var
  Lines: TStringList;
  LinePos: integer;
begin
  if not Session.HasLastLine then
  begin
    WriteLn('Nothing spoken yet.');
    exit;
  end;
  Lines := TStringList.Create;
  try
    DescribeKokoroTextReport(Session.LastLine.Report, Lines);
    WriteLn('Last line  : ', Session.LastLine.FileName);
    for LinePos := 0 to Lines.Count - 1 do WriteLn(Lines[LinePos]);
  finally
    Lines.Free;
  end;
end;

procedure RunRepl();
var
  Line, Arg, ErrorMsg, Ignored: string;
  Command: TKokoroReplCommand;
  Speed: TNeuralFloat;
begin
  WriteLn('Type English text; each line is written to the next WAV. ',
    '/help lists the commands, /exit quits.');
  while true do
  begin
    Write('> ');
    Flush(System.Output);
    if EOF(System.Input) then break;
    ReadLn(Line);
    Line := Trim(Line);
    if Line = '' then continue;
    if not ParseKokoroReplLine(Line, Command, Arg, ErrorMsg) then
    begin
      WriteLn('Error: ', ErrorMsg);
      continue;
    end;
    case Command of
      krcText: SpeakLine(Arg, false);
      krcPhonemes: SpeakLine(Arg, true);
      krcExit: break;
      krcHelp: PrintReplHelp();
      krcShow: ShowLastLine();
      krcVoice:
        if Arg = '' then WriteLn('Voice      : ', Session.Voice.Name)
        else
          try
            Session.SelectVoice(Arg);
            WriteLn('Voice      : ', Session.Voice.Name);
          except
            on E: Exception do
              WriteLn('Error: ', E.Message, ' (voice stays ',
                Session.Voice.Name, ')');
          end;
      krcSpeed:
        begin
          if Arg <> '' then
          begin
            TryParseFloat(Arg, Speed);
            Session.Speed := Speed;
          end;
          WriteLn('Speed      : ', Session.Speed:0:2);
        end;
      krcOut:
        begin
          if Arg <> '' then OutPattern := Arg;
          ExpandOutputPattern(OutPattern, LineCounter, Ignored, ErrorMsg);
          WriteLn('Output     : ', OutPattern, ' (next: ', Ignored, ')');
        end;
    end;
  end;
  WriteLn('Bye.');
end;

var
  ArgPos: integer;
  StartTime: QWord;
  OneShotOK: boolean;
  ErrorMsg: string;
begin
  Args := TStringList.Create;
  for ArgPos := 1 to ParamCount do Args.Add(ParamStr(ArgPos));
  if not ParseKokoroArgs(Args, Opt) then
  begin
    WriteLn('Error: ', Opt.ErrorMsg);
    WriteLn;
    PrintUsage();
    Args.Free;
    Halt(1);
  end;
  Args.Free;
  if Opt.SelfTest then
  begin
    SelfTest();
    Halt(0);
  end;
  if Opt.ShowHelp or (Opt.Session.ModelDir = '') then
  begin
    PrintUsage();
    if Opt.Session.ModelDir = '' then Halt(1);
    Halt(0);
  end;

  OutPattern := Opt.OutPattern;
  LineCounter := 1;
  PlayTemplate := TStringList.Create;
  EncodeTemplate := TStringList.Create;
  if Opt.PlayCommand <> '' then
    SplitCommandTemplate(Opt.PlayCommand, PlayTemplate, ErrorMsg);
  if Opt.EncodeCommand <> '' then
    SplitCommandTemplate(Opt.EncodeCommand, EncodeTemplate, ErrorMsg);
  Session := nil;
  StartTime := GetTickCount64();
  try
    WriteLn('Model      : ', Opt.Session.ModelDir);
    WriteLn('Voice      : ', Opt.Session.VoiceName);
    Session := TKokoroSession.Create(Opt.Session);
    // Text needs the dictionaries: load them now so a missing file stops here.
    if Opt.Phonemes = '' then
    begin
      WriteLn('Dictionary : ', Session.DictionaryDir);
      Session.CheckTextInput();
    end;
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      if E is ENeuralG2PError then
        WriteLn('The dictionaries are misaki/data/{us,gb}_{gold,silver}.json ',
          'from https://github.com/hexgrad/misaki; put them in ',
          '<model-dir>/misaki or pass --dict-dir <folder>.');
      Session.Free;
      PlayTemplate.Free;
      EncodeTemplate.Free;
      Halt(1);
    end;
  end;
  WriteLn('Compute    : ', Session.ComputeDescription);
  WriteLn('Threads    : ', DescribeKokoroThreads(Opt.Session));
  WriteLn('Loaded in  : ', ((GetTickCount64() - StartTime) / 1000):0:2, ' s');
  if Opt.Stats then
    WriteLn(StdErr, '[stats] load ', ((GetTickCount64() - StartTime) / 1000):0:2,
      ' s (model, voice', BoolToStr(Opt.Phonemes = '', ', dictionaries', ''),
      '), peak RSS ', PeakRssMegabytes(), ' MB');

  if (Opt.Prompt <> '') or (Opt.Phonemes <> '') then
  begin
    if Opt.Phonemes <> '' then OneShotOK := SpeakLine(Opt.Phonemes, true)
    else OneShotOK := SpeakLine(Opt.Prompt, false);
    Session.Free;
    PlayTemplate.Free;
    EncodeTemplate.Free;
    if OneShotOK then Halt(0) else Halt(1);
  end;
  RunRepl();
  Session.Free;
  PlayTemplate.Free;
  EncodeTemplate.Free;
end.
