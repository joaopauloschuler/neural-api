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
six distinct acoustic "words" (low/mid/high pure tones, an up-chirp, a
down-chirp, and band-limited noise), each with a touch of seeded jitter
and additive noise so the classes are non-trivial but clearly separable.
Every clip goes through the REAL log-mel frontend, so the smoke exercises
the exact frontend+training path end to end on CPU in well under a
minute, and prints a final test-accuracy number far above the 1/6 chance
line.

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
  // ---- synthetic keyword set ----
  NUM_CLASSES   = 6;          // low/mid/high tone, up-chirp, down-chirp, noise
  TRAIN_PER_CLS = 120;
  VAL_PER_CLS   = 30;
  TEST_PER_CLS  = 30;
  // ---- training ----
  NUM_EPOCHS    = 20;       // test phase fires at epoch 10 and 20
  BATCH_SIZE    = 16;

var
  ClassNames: array[0..NUM_CLASSES-1] of string =
    ('tone_low', 'tone_mid', 'tone_high', 'chirp_up', 'chirp_down', 'noise');

// ---------------------------------------------------------------------------
// Synthetic waveform generators. Each returns a CLIP_SAMPLES mono waveform in
// Samples (laid out along FData, the LoadWav16ToVolume convention). The label
// fixes the acoustic class; Jitter (seeded by the caller via Random) makes the
// individual clips differ so the task is non-degenerate.
// ---------------------------------------------------------------------------
procedure SynthWaveform(Samples: TNNetVolume; Label_: integer);
var
  i: integer;
  t, f, f0, f1, phase, amp, noiseAmp, jitter: double;
begin
  Samples.ReSize(CLIP_SAMPLES, 1, 1);
  jitter := (Random - 0.5) * 0.10;     // +/-5% pitch/sweep jitter per clip
  amp := 0.45 + Random * 0.20;
  noiseAmp := 0.02;
  case Label_ of
    0: f := 220.0  * (1.0 + jitter);   // low tone
    1: f := 660.0  * (1.0 + jitter);   // mid tone
    2: f := 1760.0 * (1.0 + jitter);   // high tone
  else
    f := 0.0;
  end;
  case Label_ of
    0, 1, 2:
      for i := 0 to CLIP_SAMPLES - 1 do
      begin
        t := i / SAMPLE_RATE;
        Samples.FData[i] := amp * Sin(2.0 * Pi * f * t)
          + noiseAmp * (Random - 0.5);
      end;
    3, 4:  // linear chirp (up: 300->3000 Hz, down: 3000->300 Hz)
      begin
        if Label_ = 3 then begin f0 := 300.0; f1 := 3000.0; end
                      else begin f0 := 3000.0; f1 := 300.0; end;
        f0 := f0 * (1.0 + jitter);
        f1 := f1 * (1.0 + jitter);
        phase := 0.0;
        for i := 0 to CLIP_SAMPLES - 1 do
        begin
          t := i / (CLIP_SAMPLES - 1);
          f := f0 + (f1 - f0) * t;       // instantaneous frequency
          phase := phase + 2.0 * Pi * f / SAMPLE_RATE;
          Samples.FData[i] := amp * Sin(phase) + noiseAmp * (Random - 0.5);
        end;
      end;
    5:  // band-limited noise burst (random walk, mild smoothing)
      begin
        Samples.FData[0] := 0.0;
        for i := 1 to CLIP_SAMPLES - 1 do
          Samples.FData[i] := 0.95 * Samples.FData[i-1]
            + amp * (Random - 0.5);
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
        if LoadWav16ToVolume(WavPath, Samples) <> SAMPLE_RATE then
          raise Exception.Create('SpeechCommands --full: ' + WavPath +
            ' is not 16 kHz (convert with: ffmpeg -ar 16000 -ac 1).');
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
