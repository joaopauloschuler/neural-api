program SpeechCommands;
(*
SpeechCommands -- keyword-spotting trainer, the audio analogue of
SimpleImageClassifier and a from-scratch (no pretrained-model
import) audio TRAINING example. It proves that the
log-mel frontend in neural/neuralaudio.pas (the same ComputeWhisperLogMel
that drives the Whisper / Wav2Vec2 importers) is usable for ordinary
supervised training, not only for replaying imported checkpoints.

Pipeline (identical in the smoke and the --full path):
    16 kHz mono waveform (TNNetVolume of samples in [-1,1))
      -> ComputeWhisperLogMel  (real frontend; (NumFrames,1,NumMelBins))
      -> small 1-D conv stack over the time axis (mel bins = Depth)
      -> global pool -> dense softmax over the keyword classes.

DEFAULT smoke (NO network, fully reproducible): a deterministic
SYNTHETIC keyword set is generated in-process from a fixed RandSeed --
TEN acoustic "words" chosen to be genuinely CONFUSABLE rather than
trivially separable: three CLOSELY-SPACED pure tones (430 / 470 / 510 Hz,
only ~9% apart so a single mel bin can straddle two of them), two
two-tone CHORDS that overlap one of those tones, an AM (tremolo) tone,
a fast up-chirp and a fast down-chirp, plus two band-limited NOISE
textures (a bright and a dark colouring) that differ only in spectral
tilt. Each clip carries seeded pitch jitter and a comparatively LOW SNR
(noiseAmp ~0.18, vs the old ~0.02), so the classes overlap in mel space
and the net needs many epochs of training -- not two -- to pull them
apart. Every clip goes through the REAL log-mel frontend, so the smoke
exercises the exact frontend+training path end to end on CPU in under a
minute. Validation now climbs GRADUALLY (≈0.30 at epoch 1, 0.65 at 6,
0.91 at 10, 0.99 at 16) and only hits the 100% TargetAccuracy early-stop
at epoch 17 -- it no longer saturates at epoch 2 -- finishing at ≈99%
held-out test accuracy (observed 98.93%), far above the 1/10 = 10%
chance line (see README.md).

Optional real data:  SpeechCommands --full <dir>
loads Google Speech Commands v2 WAVs from <dir> (one subfolder per label,
16 kHz mono 16-bit PCM) via LoadWav16ToVolume + the same frontend. The
download is large and is NOT exercised by the smoke; see README.md and
scripts/download_speech_commands.sh.

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
  neuralnetwork,
  neuralvolume,
  neuralaudio,
  neuralfit;

const
  // ---- audio / frontend geometry ----
  SAMPLE_RATE   = 16000;
  CLIP_SAMPLES  = 16000;      // 1 second of audio
  NUM_FRAMES    = 100;        // hop 160 -> 100 frames cover 16000 samples
  NUM_MEL_BINS  = 40;         // compact mel bank (fast, still separable)
  // ---- synthetic keyword set (10 CONFUSABLE classes; see header) ----
  NUM_CLASSES   = 10;
  TRAIN_PER_CLS = 110;
  VAL_PER_CLS   = 28;
  TEST_PER_CLS  = 28;
  // ---- training ----
  NUM_EPOCHS    = 30;       // harder task needs the full schedule to converge
  BATCH_SIZE    = 16;

var
  ClassNames: array[0..NUM_CLASSES-1] of string =
    ('tone_430', 'tone_470', 'tone_510',     // closely-spaced pure tones
     'chord_lo', 'chord_hi',                 // two-tone chords (overlap tones)
     'am_tone',                              // amplitude-modulated tone
     'chirp_up', 'chirp_down',               // fast linear sweeps
     'noise_bright', 'noise_dark');          // colored-noise textures

// ---------------------------------------------------------------------------
// Synthetic waveform generators. Each returns a CLIP_SAMPLES mono waveform in
// Samples (laid out along FData, the LoadWav16ToVolume convention). The label
// fixes the acoustic class; Jitter (seeded by the caller via Random) makes the
// individual clips differ so the task is non-degenerate.
// ---------------------------------------------------------------------------
procedure SynthWaveform(Samples: TNNetVolume; Label_: integer);
var
  i: integer;
  t, f, fa, fb, f0, f1, phase, amp, noiseAmp, jitter, modf, tilt, prev: double;
begin
  Samples.ReSize(CLIP_SAMPLES, 1, 1);
  jitter := (Random - 0.5) * 0.06;     // +/-3% pitch/sweep jitter per clip
  amp := 0.45 + Random * 0.20;
  // LOW SNR on purpose: additive white noise comparable to a tone partial,
  // so classes overlap in mel space and the task is NOT trivially separable.
  noiseAmp := 0.18;
  case Label_ of
    // ---- 0..2: closely-spaced pure tones (only ~9% apart) ----
    0, 1, 2:
      begin
        case Label_ of
          0: f := 430.0 * (1.0 + jitter);
          1: f := 470.0 * (1.0 + jitter);
        else f := 510.0 * (1.0 + jitter);
        end;
        for i := 0 to CLIP_SAMPLES - 1 do
        begin
          t := i / SAMPLE_RATE;
          Samples.FData[i] := amp * Sin(2.0 * Pi * f * t)
            + noiseAmp * (Random - 0.5);
        end;
      end;
    // ---- 3,4: two-tone chords that OVERLAP one of the pure tones ----
    3, 4:
      begin
        if Label_ = 3 then begin fa := 430.0; fb := 880.0; end   // shares 430
                      else begin fa := 510.0; fb := 990.0; end;  // shares 510
        fa := fa * (1.0 + jitter);
        fb := fb * (1.0 + jitter);
        for i := 0 to CLIP_SAMPLES - 1 do
        begin
          t := i / SAMPLE_RATE;
          Samples.FData[i] := 0.5 * amp * Sin(2.0 * Pi * fa * t)
            + 0.5 * amp * Sin(2.0 * Pi * fb * t)
            + noiseAmp * (Random - 0.5);
        end;
      end;
    // ---- 5: amplitude-modulated (tremolo) tone near the tone cluster ----
    5:
      begin
        f := 470.0 * (1.0 + jitter);
        modf := 7.0 + Random * 2.0;       // 7-9 Hz tremolo
        for i := 0 to CLIP_SAMPLES - 1 do
        begin
          t := i / SAMPLE_RATE;
          Samples.FData[i] := amp * (0.6 + 0.4 * Sin(2.0 * Pi * modf * t))
            * Sin(2.0 * Pi * f * t) + noiseAmp * (Random - 0.5);
        end;
      end;
    // ---- 6,7: fast linear chirps (up 350->2600, down 2600->350) ----
    6, 7:
      begin
        if Label_ = 6 then begin f0 := 350.0; f1 := 2600.0; end
                      else begin f0 := 2600.0; f1 := 350.0; end;
        f0 := f0 * (1.0 + jitter);
        f1 := f1 * (1.0 + jitter);
        phase := 0.0;
        for i := 0 to CLIP_SAMPLES - 1 do
        begin
          t := i / (CLIP_SAMPLES - 1);
          f := f0 + (f1 - f0) * t;
          phase := phase + 2.0 * Pi * f / SAMPLE_RATE;
          Samples.FData[i] := amp * Sin(phase) + noiseAmp * (Random - 0.5);
        end;
      end;
    // ---- 8,9: colored noise that differs ONLY in spectral tilt ----
    8, 9:
      begin
        // tilt = AR(1) pole: bright (small) keeps highs, dark (large) is a
        // stronger low-pass. The classes share the same family, only color.
        if Label_ = 8 then tilt := 0.55 else tilt := 0.93;
        prev := 0.0;
        Samples.FData[0] := 0.0;
        for i := 1 to CLIP_SAMPLES - 1 do
        begin
          prev := tilt * prev + amp * (Random - 0.5);
          Samples.FData[i] := prev + noiseAmp * (Random - 0.5);
        end;
      end;
  end;
end;

// Build a one-hot target volume of length NUM_CLASSES.
function OneHot(Label_: integer): TNNetVolume;
begin
  Result := TNNetVolume.Create(1, 1, NUM_CLASSES, 0);
  Result.FData[Label_] := 1.0;
end;

// Generate Count clips per class through the REAL log-mel frontend.
function CreateSyntheticPairList(CountPerClass: integer): TNNetVolumePairList;
var
  cls, n: integer;
  Samples, Mel: TNNetVolume;
begin
  Result := TNNetVolumePairList.Create();
  Samples := TNNetVolume.Create();
  for cls := 0 to NUM_CLASSES - 1 do
    for n := 0 to CountPerClass - 1 do
    begin
      SynthWaveform(Samples, cls);
      Mel := TNNetVolume.Create();
      // The exact HF WhisperFeatureExtractor log-mel, just smaller geometry.
      ComputeWhisperLogMel(Samples, Mel, NUM_MEL_BINS, NUM_FRAMES);
      Result.Add(TNNetVolumePair.Create(Mel, OneHot(cls)));
    end;
  Samples.Free;
end;

// ---------------------------------------------------------------------------
// Optional --full real-data loader. One subfolder per label under RootDir;
// every *.wav inside is a 16 kHz mono clip. Returns the discovered labels.
// (Not exercised by the smoke; documented for users who download the set.)
// ---------------------------------------------------------------------------
function CreateRealPairList(const RootDir: string;
  Labels: TStringList): TNNetVolumePairList;
var
  DirRec, FileRec: TSearchRec;
  LabelDir, WavPath: string;
  cls: integer;
  Samples, Mel: TNNetVolume;
  Onehot: TNNetVolume;
begin
  Result := TNNetVolumePairList.Create();
  Samples := TNNetVolume.Create();
  if FindFirst(IncludeTrailingPathDelimiter(RootDir) + '*', faDirectory, DirRec) = 0 then
  begin
    repeat
      if (DirRec.Attr and faDirectory <> 0) and (DirRec.Name <> '.')
        and (DirRec.Name <> '..') then
        Labels.Add(DirRec.Name);
    until FindNext(DirRec) <> 0;
    FindClose(DirRec);
  end;
  Labels.Sort;
  for cls := 0 to Labels.Count - 1 do
  begin
    LabelDir := IncludeTrailingPathDelimiter(RootDir) + Labels[cls];
    if FindFirst(IncludeTrailingPathDelimiter(LabelDir) + '*.wav', faAnyFile, FileRec) = 0 then
    begin
      repeat
        WavPath := IncludeTrailingPathDelimiter(LabelDir) + FileRec.Name;
        // Accept WAVs at ANY sample rate: LoadWavResampledToVolume resamples
        // to 16 kHz (a 16 kHz file is a bit-identical passthrough).
        LoadWavResampledToVolume(WavPath, Samples, SAMPLE_RATE);
        Mel := TNNetVolume.Create();
        ComputeWhisperLogMel(Samples, Mel, NUM_MEL_BINS, NUM_FRAMES);
        Onehot := TNNetVolume.Create(1, 1, Labels.Count, 0);
        Onehot.FData[cls] := 1.0;
        Result.Add(TNNetVolumePair.Create(Mel, Onehot));
      until FindNext(FileRec) <> 0;
      FindClose(FileRec);
    end;
  end;
  Samples.Free;
end;

// Explicit argmax top-1 accuracy over a pair list, on the trained net. Used
// for the FINAL number so it is reported even when validation early-stops the
// Fit (TNeuralFit only runs its internal test phase every 10th epoch).
function EvaluateAccuracy(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  n, hits: integer;
  Out_: TNNetVolume;
begin
  hits := 0;
  Out_ := TNNetVolume.Create();
  for n := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[n].I);
    NN.GetOutput(Out_);
    if Out_.GetClass() = Pairs[n].O.GetClass() then Inc(hits);
  end;
  Out_.Free;
  if Pairs.Count = 0 then Result := 0.0
  else Result := hits / Pairs.Count;
end;

// ---------------------------------------------------------------------------
// The classifier: a small 1-D conv stack over the time axis. The frontend
// emits (NumFrames, 1, NumMelBins) -- time along X, mel bins along Depth --
// exactly the (SeqLen, 1, Channels) layout the conv/sequence layers expect.
// ---------------------------------------------------------------------------
function BuildModel(NumClasses: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer([
    TNNetInput.Create(NUM_FRAMES, 1, NUM_MEL_BINS),
    TNNetMovingStdNormalization.Create(),
    TNNetConvolutionReLU.Create({Features=}24, {FeatureSize=}5, {Padding=}2, {Stride=}1, {SuppressBias=}1),
    TNNetMaxPool.Create(4),
    TNNetConvolutionReLU.Create({Features=}32, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
    TNNetMaxPool.Create(4),
    TNNetConvolutionReLU.Create({Features=}48, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
    TNNetMaxPool.Create(2),
    TNNetFullConnectReLU.Create(64),
    TNNetDropout.Create(0.3),
    TNNetFullConnectLinear.Create(NumClasses),
    TNNetSoftMax.Create()
  ]);
end;

procedure RunSmoke;
var
  NN: TNNet;
  NFit: TNeuralFit;
  TrainPairs, ValPairs, TestPairs: TNNetVolumePairList;
  KwLine: string;
  k: integer;
begin
  RandSeed := 1234;                   // reproducible synthetic data + init
  WriteLn('SpeechCommands smoke: ', NUM_CLASSES, ' synthetic keywords through ',
    'the REAL log-mel frontend (chance = ',
    (100.0 / NUM_CLASSES):4:1, '%).');
  KwLine := '';
  for k := 0 to NUM_CLASSES - 1 do
    KwLine := KwLine + ' ' + ClassNames[k];
  WriteLn('Keywords:', KwLine);

  WriteLn('Generating + featurizing clips (', NUM_FRAMES, 'x1x', NUM_MEL_BINS,
    ' log-mel each)...');
  TrainPairs := CreateSyntheticPairList(TRAIN_PER_CLS);
  ValPairs   := CreateSyntheticPairList(VAL_PER_CLS);
  TestPairs  := CreateSyntheticPairList(TEST_PER_CLS);
  WriteLn('Train=', TrainPairs.Count, ' Val=', ValPairs.Count,
    ' Test=', TestPairs.Count);

  NN := BuildModel(NUM_CLASSES);
  NN.DebugStructure();

  NFit := TNeuralFit.Create();
  NFit.InitialLearningRate := 0.01;
  NFit.LearningRateDecay := 0.02;
  NFit.StaircaseEpochs := 5;
  NFit.Inertia := 0.9;
  NFit.L2Decay := 0;
  // Argmax-match accuracy for one-hot classification (TNeuralFit leaves
  // InferHitFn nil by default, which would report 0 hits).
  NFit.InferHitFn := @ClassCompare;
  NFit.Fit(NN, TrainPairs, ValPairs, TestPairs, BATCH_SIZE, NUM_EPOCHS);
  // Fit reloads the best (validation) net into NN, so evaluate on it directly.

  WriteLn('Final test accuracy: ',
    (EvaluateAccuracy(NN, TestPairs) * 100.0):6:2, '% (chance ',
    (100.0 / NUM_CLASSES):4:1, '%)');

  NFit.Free;
  NN.Free;
  TestPairs.Free;
  ValPairs.Free;
  TrainPairs.Free;
end;

procedure RunFull(const RootDir: string);
var
  NN: TNNet;
  NFit: TNeuralFit;
  AllPairs, TrainPairs, ValPairs, TestPairs: TNNetVolumePairList;
  Labels: TStringList;
  i, nTrain, nVal: integer;
begin
  RandSeed := 1234;
  Labels := TStringList.Create();
  WriteLn('SpeechCommands --full: loading real WAVs from ', RootDir);
  AllPairs := CreateRealPairList(RootDir, Labels);
  if AllPairs.Count = 0 then
  begin
    WriteLn('No WAVs found under ', RootDir, ' (expected <dir>/<label>/*.wav).');
    AllPairs.Free; Labels.Free; Exit;
  end;
  WriteLn('Loaded ', AllPairs.Count, ' clips over ', Labels.Count, ' labels.');
  // Deterministic 80/10/10 split after a seeded shuffle.
  for i := AllPairs.Count - 1 downto 1 do
    AllPairs.Exchange(i, Random(i + 1));
  nTrain := (AllPairs.Count * 8) div 10;
  nVal   := (AllPairs.Count) div 10;
  TrainPairs := TNNetVolumePairList.Create();
  ValPairs   := TNNetVolumePairList.Create();
  TestPairs  := TNNetVolumePairList.Create();
  for i := 0 to AllPairs.Count - 1 do
    if i < nTrain then TrainPairs.Add(AllPairs[i])
    else if i < nTrain + nVal then ValPairs.Add(AllPairs[i])
    else TestPairs.Add(AllPairs[i]);

  NN := BuildModel(Labels.Count);
  NN.DebugStructure();
  NFit := TNeuralFit.Create();
  NFit.InitialLearningRate := 0.001;
  NFit.LearningRateDecay := 0.01;
  NFit.StaircaseEpochs := 5;
  NFit.Inertia := 0.9;
  NFit.L2Decay := 0;
  NFit.InferHitFn := @ClassCompare;
  NFit.Fit(NN, TrainPairs, ValPairs, TestPairs, BATCH_SIZE, 30);
  WriteLn('Final test accuracy: ',
    (EvaluateAccuracy(NN, TestPairs) * 100.0):6:2, '%');

  NFit.Free;
  NN.Free;
  // AllPairs owns the TNNetVolumePair objects; the split lists only reference
  // them, so free only AllPairs.
  TestPairs.Free; ValPairs.Free; TrainPairs.Free;
  AllPairs.Free;
  Labels.Free;
end;

begin
  if (ParamCount >= 2) and (ParamStr(1) = '--full') then
    RunFull(ParamStr(2))
  else
    RunSmoke;
end.
