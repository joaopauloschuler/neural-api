program MusicGenProbe;
(*
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

// MusicGenProbe -- COMPONENT-ISOLATION diagnostic for the MusicGenText
// pipeline. The end-to-end demo chains T5 -> MusicGen decoder -> EnCodec; when
// the output is noise instead of music the fault could be in ANY of those
// three. This probe exercises the EnCodec audio codec ALONE -- no T5, no LM --
// so the codec can be cleared or convicted on its own.
//
// WHY THE CODEC FIRST
// -------------------
// If the EnCodec decoder is broken, the clip is noise no matter how good the
// MusicGen codes are. And the codec is the CHEAPEST component to run: it has
// no autoregressive loop, so memory is bounded only by the clip length (which
// --seconds caps), unlike the LM whose peak RAM grows with the generated
// sequence. So it is the right first bisection step under a RAM budget.
//
// WHAT IT DOES (reference-free)
// -----------------------------
// 1. Builds ONLY the EnCodec model (real facebook/encodec_32khz with
//    --download, else the committed pico fixture).
// 2. Gets an input waveform: a real WAV via --in, otherwise a synthesized
//    multi-tone "chord+sweep" so the probe is useful with no input file.
// 3. Reconstruct(): encode -> RVQ codes -> decode, the codec's full round
//    trip. Reports, WITHOUT any external reference:
//       - per-codebook code-index range (must stay in [0, codebook_size))
//       - input vs output energy ratio and Pearson correlation
//       - NaN / Inf counts and output peak
//    A working codec reconstructs recognizable audio: correlation well above
//    zero and an energy ratio near 1. NOISE (correlation ~ 0, wild energy
//    ratio, or NaNs) convicts the codec. On the PICO fixture the weights are
//    random, so low correlation is EXPECTED there -- pico only proves the
//    probe runs; point it at --download for the real verdict.
// 4. Writes the reconstruction to a WAV so it can be listened to directly.
//
// Each stage is timed (load, encode+decode) so the per-step cost is visible.
//
// Usage:
//   MusicGenProbe                              # pico fixture, synth tone
//   MusicGenProbe --download                   # real encodec_32khz, synth tone
//   MusicGenProbe --download --in song.wav --seconds 4
//   MusicGenProbe --download --quantizers 4    # decode only K RVQ stages
// Repo override: --encodec-repo. Output WAV: --out (default below).

{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralvolume, neuralnetwork, neuralpretrained, neuralaudio, neuralhfhub;

const
  PicoFixtureDir = '../../tests/fixtures/';
  DefaultEnCodecRepo = 'facebook/encodec_32khz';
  DefaultSeconds = 3.0;        // input cap; memory scales with this
  DefaultOutWav = 'musicgen_probe_recon.wav';

var
  Codec: TEnCodecModel;
  CodecCfg: TEnCodecConfig;
  EcSafe, EcCfg, EcDir: string;
  InWav, OutWav: string;
  RealMode: boolean;
  Seconds: TNeuralFloat;
  Quantizers, SampleRate, i, k_i, NumSamples, FrameCount, n: integer;
  TickStart: QWord;
  InVol: TNNetVolume;
  Input, Recon: TNeuralFloatDynArr;
  Codes: TNNetIntArr2D;
  cmin, cmax: integer;
  sumIn, sumOut, sumInOut, sumIn2, sumOut2, mIn, mOut: double;
  energyRatio, corr, peak: double;
  nanCount, infCount: integer;

  function HasFlag(const Name: string): boolean;
  var a: integer;
  begin
    Result := False;
    for a := 1 to ParamCount do if ParamStr(a) = Name then Result := True;
  end;

  function ParseStrArg(const Name, Def: string): string;
  var a: integer;
  begin
    Result := Def;
    for a := 1 to ParamCount - 1 do
      if ParamStr(a) = Name then Result := ParamStr(a + 1);
  end;

  function ParseFloatArg(const Name: string; Def: TNeuralFloat): TNeuralFloat;
  var a: integer; v: TNeuralFloat;
  begin
    Result := Def;
    for a := 1 to ParamCount - 1 do
      if (ParamStr(a) = Name) and TryStrToFloat(ParamStr(a + 1), v) then Result := v;
  end;

  function ParseIntArg(const Name: string; Def: integer): integer;
  var a, v: integer;
  begin
    Result := Def;
    for a := 1 to ParamCount - 1 do
      if (ParamStr(a) = Name) and TryStrToInt(ParamStr(a + 1), v) then Result := v;
  end;

  // Milliseconds since StartTick, formatted "NNN ms" / "N.NN s".
  function Elapsed(const StartTick: QWord): string;
  var ms: QWord;
  begin
    ms := GetTickCount64 - StartTick;
    if ms >= 1000 then Result := FormatFloat('0.00', ms / 1000.0) + ' s'
    else Result := IntToStr(ms) + ' ms';
  end;

  // A deterministic test signal at SR Hz, Secs long: a 220/277/330 Hz triad
  // plus a slow rising sweep. Recognizable enough that a correct reconstruction
  // sounds tonal and a broken codec is obviously noise, even by ear.
  procedure SynthTone(SR: integer; Secs: TNeuralFloat; out Sig: TNeuralFloatDynArr);
  var j, total: integer; tsec, sweep: double;
  begin
    total := Round(SR * Secs);
    if total < 1 then total := 1;
    SetLength(Sig, total);
    for j := 0 to total - 1 do
    begin
      tsec := j / SR;
      sweep := 110.0 + 220.0 * (j / total);  // rising 110 -> 330 Hz
      Sig[j] := 0.25 * (
          Sin(2 * Pi * 220.0 * tsec)
        + Sin(2 * Pi * 277.0 * tsec)
        + Sin(2 * Pi * 330.0 * tsec)
        + Sin(2 * Pi * sweep * tsec));
    end;
  end;

begin
  WriteLn('MusicGen PROBE - EnCodec codec isolation (no T5, no LM)');
  WriteLn('======================================================');

  RealMode := HasFlag('--download');
  Seconds := ParseFloatArg('--seconds', DefaultSeconds);
  Quantizers := ParseIntArg('--quantizers', 0);   // 0 = all RVQ stages
  InWav := ParseStrArg('--in', '');
  OutWav := ParseStrArg('--out', DefaultOutWav);

  if RealMode then
  begin
    WriteLn('Fetching EnCodec via HuggingFace Hub (cache: ', HubGetCacheDir, ')...');
    TickStart := GetTickCount64;
    EcDir := IncludeTrailingPathDelimiter(
      HubFetchModel(ParseStrArg('--encodec-repo', DefaultEnCodecRepo)));
    WriteLn('[time] fetch ', ParseStrArg('--encodec-repo', DefaultEnCodecRepo),
      ': ', Elapsed(TickStart));
    EcSafe := EcDir + 'model.safetensors';
    EcCfg := EcDir + 'config.json';
  end
  else
  begin
    EcSafe := PicoFixtureDir + 'tiny_musicgen_encodec.safetensors';
    EcCfg := PicoFixtureDir + 'tiny_musicgen_encodec_config.json';
    if not FileExists(EcSafe) then
    begin
      WriteLn('Pico fixture not found: ', EcSafe);
      WriteLn('Run from examples/MusicGenProbe, or pass --download.');
      Halt(1);
    end;
    WriteLn('Pico fixture mode (RANDOM weights): reconstruction WILL be noisy.');
    WriteLn('This only proves the probe runs; use --download for a real verdict.');
  end;
  WriteLn;

  Codec := nil;
  InVol := TNNetVolume.Create;
  try
    // ---- 1. Build the codec. -----------------------------------------------
    CodecCfg := ReadEnCodecConfigFromJSONFile(EcCfg);
    TickStart := GetTickCount64;
    Codec := BuildEnCodecFromSafeTensors(EcSafe, CodecCfg, EcCfg);
    WriteLn('[time] EnCodec load (', EcSafe, '): ', Elapsed(TickStart));
    WriteLn('EnCodec: ', EnCodecConfigToString(CodecCfg));
    WriteLn('Quantizers available: ', Codec.NumCodebooks,
      '; decode stages: ',
      IfThen(Quantizers <= 0, Codec.NumCodebooks, Quantizers));
    WriteLn;

    // ---- 2. Acquire the input waveform (capped to --seconds). --------------
    if InWav <> '' then
    begin
      SampleRate := LoadWav16ToVolume(InWav, InVol);
      WriteLn('Loaded ', InWav, ': ', InVol.Size, ' samples at ', SampleRate, ' Hz');
      if SampleRate <> CodecCfg.SamplingRate then
        WriteLn('WARNING: input is ', SampleRate, ' Hz but the codec expects ',
          CodecCfg.SamplingRate, ' Hz - pitch/length will be off, but the ',
          'reconstruction-vs-input correlation is still a valid codec check.');
      NumSamples := Round(Seconds * SampleRate);
      if NumSamples < 1 then NumSamples := 1;
      if NumSamples > InVol.Size then NumSamples := InVol.Size;
      SetLength(Input, NumSamples);
      for i := 0 to NumSamples - 1 do Input[i] := InVol.FData[i];
      if NumSamples < InVol.Size then
        WriteLn('Capped to ', Seconds:0:1, ' s (', NumSamples, ' samples).');
    end
    else
    begin
      SampleRate := CodecCfg.SamplingRate;
      SynthTone(SampleRate, Seconds, Input);
      WriteLn('Synthesized ', Seconds:0:1, ' s tone at ', SampleRate, ' Hz (',
        Length(Input), ' samples) - no --in given.');
    end;
    WriteLn;

    // ---- 3. Encode -> codes (range sanity), then full Reconstruct. ---------
    TickStart := GetTickCount64;
    Codec.EncodeAudioToCodes(Input, Codes, FrameCount);
    WriteLn('[time] EnCodec encode: ', Elapsed(TickStart),
      ' -> ', Length(Codes), ' codebooks x ', FrameCount, ' frames');
    for k_i := 0 to Length(Codes) - 1 do
    begin
      cmin := MaxInt; cmax := -MaxInt;
      for i := 0 to Length(Codes[k_i]) - 1 do
      begin
        if Codes[k_i][i] < cmin then cmin := Codes[k_i][i];
        if Codes[k_i][i] > cmax then cmax := Codes[k_i][i];
      end;
      Write('  cb', k_i, ' index range [', cmin, ', ', cmax, ']');
      if (cmin < 0) or (cmax >= CodecCfg.CodebookSize) then
        Write('  <-- OUT OF RANGE for codebook_size=', CodecCfg.CodebookSize);
      WriteLn;
    end;

    TickStart := GetTickCount64;
    if Quantizers > 0 then
      Codec.DecodeCodesToAudio(Codes, Recon, Quantizers)
    else
      Codec.Reconstruct(Input, Recon);
    WriteLn('[time] EnCodec decode: ', Elapsed(TickStart),
      ' -> ', Length(Recon), ' samples');
    WriteLn;

    // ---- 4. Reference-free verdict statistics. -----------------------------
    n := Length(Input);
    if Length(Recon) < n then n := Length(Recon);
    if n = 0 then
    begin
      WriteLn('Empty reconstruction - codec produced no samples.');
      Halt(1);
    end;
    sumIn := 0; sumOut := 0; nanCount := 0; infCount := 0; peak := 0;
    for i := 0 to Length(Recon) - 1 do
    begin
      if IsNan(Recon[i]) then Inc(nanCount)
      else if IsInfinite(Recon[i]) then Inc(infCount)
      else if Abs(Recon[i]) > peak then peak := Abs(Recon[i]);
    end;
    // Pearson correlation + energy ratio over the overlapping region.
    for i := 0 to n - 1 do
    begin
      sumIn := sumIn + Input[i];
      sumOut := sumOut + Recon[i];
    end;
    mIn := sumIn / n; mOut := sumOut / n;
    sumInOut := 0; sumIn2 := 0; sumOut2 := 0;
    for i := 0 to n - 1 do
    begin
      sumInOut := sumInOut + (Input[i] - mIn) * (Recon[i] - mOut);
      sumIn2 := sumIn2 + Sqr(Input[i] - mIn);
      sumOut2 := sumOut2 + Sqr(Recon[i] - mOut);
    end;
    if (sumIn2 > 0) and (sumOut2 > 0) then
      corr := sumInOut / Sqrt(sumIn2 * sumOut2)
    else
      corr := 0;
    if sumIn2 > 0 then energyRatio := sumOut2 / sumIn2 else energyRatio := 0;

    WriteLn('---- VERDICT (reference-free) ----');
    WriteLn('  output peak amplitude : ', peak:0:4);
    WriteLn('  NaN / Inf samples     : ', nanCount, ' / ', infCount);
    WriteLn('  energy ratio out/in   : ', energyRatio:0:4, '  (healthy ~ 0.2 .. 5)');
    WriteLn('  input/output corr     : ', corr:0:4,
      '  (healthy codec >> 0; ~0 = noise)');
    if (nanCount > 0) or (infCount > 0) then
      WriteLn('  >> NaN/Inf present: the codec path is numerically broken.')
    else if RealMode and (corr < 0.2) then
      WriteLn('  >> Low correlation on REAL weights: codec is the prime suspect.')
    else if RealMode then
      WriteLn('  >> Codec reconstructs structure: bug is likely UPSTREAM (LM/T5).')
    else
      WriteLn('  >> Pico random weights: low correlation here is EXPECTED.');
    WriteLn;

    // ---- 5. Write the reconstruction so it can be heard. -------------------
    InVol.ReSize(Length(Recon), 1, 1);
    for i := 0 to Length(Recon) - 1 do InVol.FData[i] := Recon[i];
    SaveVolumeToWav16(InVol, OutWav, CodecCfg.SamplingRate);
    WriteLn('Wrote ', OutWav, ' (', Length(Recon), ' samples, ',
      (Length(Recon) / CodecCfg.SamplingRate):0:2, ' s at ',
      CodecCfg.SamplingRate, ' Hz).');
  finally
    Codec.Free;
    InVol.Free;
  end;
end.
