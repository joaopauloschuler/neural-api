program SpeakerDiarization;
(*
SpeakerDiarization: a SMOKE for the FIRST "who speaks when" model in the
library - frame-level MULTI-speaker activity via the pyannote/segmentation-3.0
importer (BuildPyannoteSegmentationFromSafeTensors). This is deliberately
DISTINCT from the transcription / CTC paths (Whisper, Wav2Vec2): instead of
"what was said", it answers "WHO was speaking WHEN".

THE PIPELINE
------------
A raw mono waveform feeds a learnable SincNet band-pass front-end
(TNNetSincConv1D - each filter is materialized from just two scalars, a low
cutoff and a bandwidth, windowed by a Hamming window), then a conv/pool/LayerNorm
stack, a bidirectional minimal-LSTM temporal trunk (TNNetMinLSTM forward +
time-reversed), a linear layer, and a per-frame POWERSET multilabel head: the
3.0 model emits 7 classes covering every subset of <= 3 concurrent speakers
(silence; speaker 1/2/3; pairs 1+2/1+3/2+3). PyannotePowersetDecode turns the
per-frame argmax back into a per-speaker binary ACTIVITY matrix.

WHAT THIS SMOKE DOES
--------------------
It loads the committed pico fixture (a tiny RE-RANDOMIZED checkpoint - NOT a
real pyannote model, so the activity is illustrative, not meaningful), synthesizes
a short two-tone "two speaker" waveform at runtime, runs the net, prints a
per-frame speaker-activity TIMELINE and one RTTM line per speaker TURN, and saves
the synthetic clip to a WAV (via the landed SaveVolumeToWav16) so the audio
front-end utilities are exercised too. Pairs naturally with WhisperTranscribe for
"who said what".

To keep it inside a small pure-CPU / low-RAM budget everything is tiny; swap the
fixture for a real pyannote/segmentation-3.0 checkpoint (and a longer real clip)
to get genuine diarization.
*)
{$mode objfpc}{$H+}

uses
  SysUtils, Classes,
  neuralvolume, neuralnetwork, neuralpretrained, neuralaudio;

const
  // Locate the committed pico fixture relative to this example directory.
  FixtureDir = '../../tests/fixtures/';

var
  NN: TNNet;
  Cfg: TPyannoteConfig;
  Wave, Logits: TNNetVolume;
  NumSamples, Frames, fr, sp, t: integer;
  SampleRate, FrameDurMs, ClipStartMs: double;
  Active: array of boolean;
  PrevActive: array of boolean;
  TurnStart: array of integer;
  ConfigPath, ModelPath, WavPath: string;

  function SecOf(FrameIdx: integer): double;
  begin
    Result := (ClipStartMs + FrameIdx * FrameDurMs) / 1000.0;
  end;

begin
  ConfigPath := FixtureDir + 'tiny_pyannote_config.json';
  ModelPath  := FixtureDir + 'tiny_pyannote.safetensors';
  WavPath    := 'diarization_demo.wav';
  if not FileExists(ModelPath) then
  begin
    WriteLn('Pico fixture not found at ', ModelPath);
    WriteLn('Run tools/make_pico_pyannote_fixture.py first.');
    Halt(1);
  end;

  Cfg := ReadPyannoteConfigFromJSONFile(ConfigPath);
  SampleRate := Cfg.SampleRate;
  WriteLn('Pyannote config: ', PyannoteConfigToString(Cfg));

  // ---- synthesize a short two-"speaker" waveform -------------------------
  // Speaker A: a low tone in the first half; speaker B: a higher tone in the
  // second half, with a brief overlap in the middle. (Illustrative only.)
  // NumSamples chosen so every max-pool window divides evenly (clean,
  // non-overlapping pooling) for this pico fixture's shape.
  NumSamples := 1001;
  ClipStartMs := 0;
  Wave := TNNetVolume.Create(NumSamples, 1, 1);
  for t := 0 to NumSamples - 1 do
  begin
    Wave.FData[t] := 0;
    if t < (NumSamples * 6) div 10 then
      Wave.FData[t] := Wave.FData[t] + 0.5 * Sin(2 * Pi * 180 * t / SampleRate);
    if t > (NumSamples * 4) div 10 then
      Wave.FData[t] := Wave.FData[t] + 0.5 * Sin(2 * Pi * 540 * t / SampleRate);
  end;
  SaveVolumeToWav16(Wave, WavPath, Round(SampleRate));
  WriteLn('Synthesized ', NumSamples, ' samples (',
    (NumSamples / SampleRate):0:3, ' s) -> ', WavPath);

  // ---- build the net for this clip length and run it ---------------------
  NN := BuildPyannoteSegmentationFromSafeTensorsEx(ModelPath, Cfg, NumSamples,
    {pInferenceOnly=}true, ConfigPath);
  Frames := PyannoteFrameCount(Cfg, NumSamples);
  FrameDurMs := 1000.0 * NumSamples / SampleRate / Frames;
  WriteLn('Net frames: ', Frames, '  (', FrameDurMs:0:1, ' ms/frame)  classes: ',
    Cfg.NumPowersetClasses, '  max speakers: ', Cfg.MaxSpeakers);

  Logits := TNNetVolume.Create;
  NN.Compute(Wave);
  NN.GetOutput(Logits);

  // ---- per-frame speaker-activity timeline -------------------------------
  SetLength(Active, Cfg.MaxSpeakers);
  SetLength(PrevActive, Cfg.MaxSpeakers);
  SetLength(TurnStart, Cfg.MaxSpeakers);
  for sp := 0 to Cfg.MaxSpeakers - 1 do
  begin
    PrevActive[sp] := False;
    TurnStart[sp] := 0;
  end;

  WriteLn;
  WriteLn('Per-frame speaker activity ( . = silent, # = active ):');
  Write('         ');
  for sp := 0 to Cfg.MaxSpeakers - 1 do Write('spk', sp + 1, ' ');
  WriteLn;
  for fr := 0 to Frames - 1 do
  begin
    PyannotePowersetDecode(Logits, fr, Cfg, Active);
    Write('  ', SecOf(fr):6:2, 's  ');
    for sp := 0 to Cfg.MaxSpeakers - 1 do
      if Active[sp] then Write(' #   ') else Write(' .   ');
    WriteLn;
    // Emit an RTTM SPEAKER line on each falling edge (end of a turn).
    for sp := 0 to Cfg.MaxSpeakers - 1 do
    begin
      if Active[sp] and (not PrevActive[sp]) then TurnStart[sp] := fr;
      if (not Active[sp]) and PrevActive[sp] then
        WriteLn(Format('SPEAKER demo 1 %.3f %.3f <NA> <NA> spk%d <NA> <NA>',
          [SecOf(TurnStart[sp]), SecOf(fr) - SecOf(TurnStart[sp]), sp + 1]));
      PrevActive[sp] := Active[sp];
    end;
  end;
  // Close any turns still open at the end of the clip.
  for sp := 0 to Cfg.MaxSpeakers - 1 do
    if PrevActive[sp] then
      WriteLn(Format('SPEAKER demo 1 %.3f %.3f <NA> <NA> spk%d <NA> <NA>',
        [SecOf(TurnStart[sp]), SecOf(Frames) - SecOf(TurnStart[sp]), sp + 1]));

  WriteLn;
  WriteLn('Done. (Pico fixture - activity is illustrative, not a real model.)');
  Logits.Free;
  Wave.Free;
  NN.Free;
end.
