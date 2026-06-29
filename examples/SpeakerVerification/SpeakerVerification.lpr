program SpeakerVerification;
(*
SpeakerVerification: a SMOKE for the ECAPA-TDNN speaker-EMBEDDING importer
(BuildEcapaTdnnFromSafeTensors). This is the COMPANION to SpeakerDiarization:
pyannote answers "who speaks WHEN" within one window; ECAPA-TDNN turns a whole
utterance into a single fixed-length speaker EMBEDDING so two clips can be
COMPARED - the core of speaker VERIFICATION ("are these two the same person?")
and the cross-window clustering a full diarization pipeline needs.

THE MODEL
---------
On a (T, 1, NumMel) log-mel frame sequence:
  conv_pre (dilated TDNN conv + ReLU)
  -> 3x SE-Res2Block: a Res2Net hierarchical-residual dilated TDNN cascade
     (TNNetTDNNConv1D, growing receptive field within the block) with
     squeeze-excitation channel gating (the landed AddSEBlock)
  -> multi-layer feature aggregation (concat the 3 block outputs) + 1x1 conv
  -> ATTENTIVE STATISTICS POOLING (TNNetAttentiveStatsPooling): a per-frame
     attention head whose softmax weights give a context-weighted MEAN and
     STANDARD-DEVIATION over time, concatenated into the utterance vector
  -> linear -> the speaker embedding.
AAM-softmax (ArcFace) is training-only; verification reads the embedding
directly and scores two clips by COSINE similarity (EcapaCosineScore).

WHAT THIS SMOKE DOES
--------------------
Loads the committed pico fixture (a tiny RE-RANDOMIZED checkpoint - NOT a real
speechbrain model, so the scores are illustrative, not meaningful), synthesizes
three short log-mel "clips": two phrasings of speaker A (same tonal makeup,
different content) and one clip of speaker B (different tonal makeup). It embeds
all three, prints the cosine score for the same-speaker pair and the
different-speaker pair, and shows that the same-speaker score is the higher of
the two - the verification decision. Pairs naturally with SpeakerDiarization to
cluster turns across windows.

To keep it inside a small pure-CPU / low-RAM budget everything is tiny; swap the
fixture for a real speechbrain/spkrec-ecapa-voxceleb checkpoint (and real
log-mel features via neuralaudio) for genuine verification.

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses
  SysUtils, Classes,
  neuralvolume, neuralnetwork, neuralpretrained;

const
  // Locate the committed pico fixture relative to this example directory.
  FixtureDir = '../../tests/fixtures/';

var
  NN: TNNet;
  Cfg: TEcapaTdnnConfig;
  Frames: integer;
  ConfigPath, ModelPath: string;
  MelA1, MelA2, MelB: TNNetVolume;
  EmbA1, EmbA2, EmbB: TNNetVolume;
  ScoreSame, ScoreDiff: TNeuralFloat;

  // Synthesize a tiny deterministic log-mel-like clip. Speaker "identity" is
  // carried by Tone (the per-mel-bin tonal makeup); Phrase varies the content.
  procedure SynthMel(M: TNNetVolume; Tone, Phrase: double);
  var
    t, b: integer;
  begin
    M.ReSize(Frames, 1, Cfg.NumMel);
    for t := 0 to Frames - 1 do
      for b := 0 to Cfg.NumMel - 1 do
        M[t, 0, b] :=
          0.6 * Sin(Tone * (b + 1) + 0.21 * t) +
          0.3 * Cos(Phrase * t + 0.5 * b);
  end;

  function Embed(M: TNNetVolume): TNNetVolume;
  begin
    Result := TNNetVolume.Create;
    NN.Compute(M);
    NN.GetOutput(Result);
  end;

begin
  ConfigPath := FixtureDir + 'tiny_ecapa_config.json';
  ModelPath := FixtureDir + 'tiny_ecapa.safetensors';
  if not FileExists(ModelPath) then
  begin
    WriteLn('Fixture not found: ', ModelPath);
    WriteLn('Run tools/make_pico_ecapa_fixture.py first.');
    Halt(1);
  end;

  Frames := 12; // short clip; the importer is length-agnostic.

  WriteLn('Building ECAPA-TDNN speaker encoder from ', ModelPath, ' ...');
  NN := BuildEcapaTdnnFromSafeTensorsEx(ModelPath, Cfg, Frames,
    {pTrainable=}false, ConfigPath);
  try
    WriteLn(Format('  num_mel=%d channels=%d scale=%d mfa=%d emb_dim=%d',
      [Cfg.NumMel, Cfg.Channels, Cfg.Scale, Cfg.MFAChannels, Cfg.EmbDim]));
    WriteLn;

    MelA1 := TNNetVolume.Create; MelA2 := TNNetVolume.Create;
    MelB := TNNetVolume.Create;
    try
      // Two clips of speaker A (same tonal makeup, different phrasing) and one
      // clip of speaker B (a different tonal makeup).
      SynthMel(MelA1, {Tone=}0.40, {Phrase=}0.21);
      SynthMel(MelA2, {Tone=}0.40, {Phrase=}0.55);
      SynthMel(MelB,  {Tone=}1.15, {Phrase=}0.33);

      EmbA1 := Embed(MelA1);
      EmbA2 := Embed(MelA2);
      EmbB := Embed(MelB);
      try
        ScoreSame := EcapaCosineScore(EmbA1, EmbA2);
        ScoreDiff := EcapaCosineScore(EmbA1, EmbB);
        WriteLn('Speaker-verification cosine scores (higher => same speaker):');
        WriteLn(Format('  same  speaker (A1 vs A2): %8.5f', [ScoreSame]));
        WriteLn(Format('  diff. speaker (A1 vs B ): %8.5f', [ScoreDiff]));
        WriteLn;
        if ScoreSame > ScoreDiff then
          WriteLn('VERIFIED: the same-speaker pair scores higher (as expected).')
        else
          WriteLn('NOTE: pico fixture is random; ordering is illustrative only.');
      finally
        EmbA1.Free; EmbA2.Free; EmbB.Free;
      end;
    finally
      MelA1.Free; MelA2.Free; MelB.Free;
    end;
  finally
    NN.Free;
  end;
end.
