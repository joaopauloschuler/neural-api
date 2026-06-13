program Summarize;
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

// Summarize -- abstractive text summarization on CPU with a pretrained
// HuggingFace BART checkpoint (facebook/bart-large-cnn,
// sshleifer/distilbart-cnn-12-6, ...) imported by BuildBartFromSafeTensors.
// BART is the dominant pretrained encoder-decoder for summarization: a
// bidirectional encoder reads the article, a causal decoder with
// cross-attention writes the summary. This driver encodes the article with
// the checkpoint's GPT-2 byte-level BPE tokenizer.json, runs the landed
// token-id BEAM SEARCH (DecodeSeq2SeqBeamSearch), decodes the summary back
// to text and -- when a reference summary is supplied -- reports ROUGE-1 /
// ROUGE-2 / ROUGE-L F1 (neuralnlpmetrics).
//
// Usage:
//   Summarize <checkpoint_dir> [options]
//
//   <checkpoint_dir> - a directory holding model.safetensors, config.json
//                      and tokenizer.json (the standard HF BART layout).
//   -a "article"     - the source article text (default: a built-in demo
//                      article + its reference summary).
//   -af <file>       - read the article from a UTF-8 text file.
//   -ref "summary"   - a reference summary for ROUGE scoring.
//   -rf <file>       - read the reference summary from a file.
//   -enc <N>         - encoder window to build (default 256 tokens; the
//                      article is truncated to fit, with bos/eos).
//   -dec <N>         - max summary length to build/generate (default 64).
//   -beam <N>        - beam width (default 4; 1 = greedy).
//   -lp <f>          - length penalty alpha (default 1.0; Wu et al.).
//
// CPU note: build/run under `ulimit -v 3000000`. distilbart-cnn-12-6 is the
// fastest published CNN checkpoint; bart-large-cnn is heavier. Keep -enc /
// -dec small for a quick CPU demo.

{$mode objfpc}{$H+}

uses
  SysUtils, Classes,
  neuralvolume, neuralnetwork, neuralpretrained, neuraldecode,
  neuralhftokenizer, neuralnlpmetrics;

const
  // A short built-in demo article (CNN/DailyMail style) and a human
  // reference summary, so the example runs with no input files.
  DemoArticle =
    'The European Space Agency announced on Tuesday that its newest ' +
    'weather satellite had reached its final orbit and begun returning ' +
    'data. The satellite, the first of a new generation, carries an ' +
    'imager that scans the full disc of the Earth every ten minutes, ' +
    'twice as fast as the previous fleet. Forecasters say the sharper, ' +
    'more frequent images will improve early warnings for storms and ' +
    'flooding across Europe and Africa. The agency said the remaining ' +
    'instruments would be switched on over the coming weeks before the ' +
    'satellite enters routine service early next year.';
  DemoReference =
    'A new European weather satellite has reached orbit and started ' +
    'sending data, scanning the Earth every ten minutes to improve ' +
    'storm and flood warnings.';

var
  CheckpointDir, ModelPath, ConfigPath, TokPath: string;
  ArticleText, ReferenceText, SummaryText: string;
  EncSeqLen, DecSeqLen, BeamWidth: integer;
  LengthPenalty: TNeuralFloat;
  Enc, Dec: TNNet;
  Config: TBartConfig;
  Tok: TNeuralHFTokenizer;
  ArticleIds, SourceIds, SummaryIds: TNeuralIntegerArray;
  i, p, MaxArticle: integer;
  R1, R2, RL: TNNetRougeScore;
  FloatFmt: TFormatSettings;

  function ReadTextFile(const FN: string): string;
  var
    SL: TStringList;
  begin
    SL := TStringList.Create;
    try
      SL.LoadFromFile(FN);
      Result := SL.Text;
    finally
      SL.Free;
    end;
  end;

begin
  FloatFmt := DefaultFormatSettings;
  FloatFmt.DecimalSeparator := '.';
  ArticleText := DemoArticle;
  ReferenceText := DemoReference;
  EncSeqLen := 256;
  DecSeqLen := 64;
  BeamWidth := 4;
  LengthPenalty := 1.0;

  if ParamCount < 1 then
  begin
    WriteLn('Usage: Summarize <checkpoint_dir> [options]');
    WriteLn('  -a "article" | -af file   source article (default: demo)');
    WriteLn('  -ref "summary" | -rf file reference for ROUGE');
    WriteLn('  -enc N -dec N             encoder/decoder windows');
    WriteLn('  -beam N -lp f             beam width / length penalty');
    WriteLn('Needs model.safetensors + config.json + tokenizer.json in ' +
      'the directory (HF BART layout: facebook/bart-large-cnn, ' +
      'sshleifer/distilbart-cnn-12-6, ...).');
    Halt(1);
  end;

  CheckpointDir := IncludeTrailingPathDelimiter(ParamStr(1));
  i := 2;
  while i <= ParamCount do
  begin
    if (ParamStr(i) = '-a') and (i < ParamCount) then
    begin Inc(i); ArticleText := ParamStr(i); ReferenceText := ''; end
    else if (ParamStr(i) = '-af') and (i < ParamCount) then
    begin Inc(i); ArticleText := ReadTextFile(ParamStr(i)); end
    else if (ParamStr(i) = '-ref') and (i < ParamCount) then
    begin Inc(i); ReferenceText := ParamStr(i); end
    else if (ParamStr(i) = '-rf') and (i < ParamCount) then
    begin Inc(i); ReferenceText := ReadTextFile(ParamStr(i)); end
    else if (ParamStr(i) = '-enc') and (i < ParamCount) then
    begin Inc(i); EncSeqLen := StrToIntDef(ParamStr(i), EncSeqLen); end
    else if (ParamStr(i) = '-dec') and (i < ParamCount) then
    begin Inc(i); DecSeqLen := StrToIntDef(ParamStr(i), DecSeqLen); end
    else if (ParamStr(i) = '-beam') and (i < ParamCount) then
    begin Inc(i); BeamWidth := StrToIntDef(ParamStr(i), BeamWidth); end
    else if (ParamStr(i) = '-lp') and (i < ParamCount) then
    begin Inc(i);
      LengthPenalty := StrToFloatDef(ParamStr(i), LengthPenalty, FloatFmt);
    end;
    Inc(i);
  end;

  ModelPath := CheckpointDir + 'model.safetensors';
  ConfigPath := CheckpointDir + 'config.json';
  TokPath := CheckpointDir + 'tokenizer.json';
  if not FileExists(ModelPath) then
  begin
    WriteLn('ERROR: not found: ', ModelPath);
    Halt(2);
  end;
  if not FileExists(TokPath) then
  begin
    WriteLn('ERROR: tokenizer.json not found: ', TokPath);
    WriteLn('BART ships a GPT-2 byte-level BPE tokenizer.json - needed to ' +
      'encode the article.');
    Halt(2);
  end;

  Enc := nil; Dec := nil; Tok := nil;
  try
    WriteLn('Loading BART checkpoint from ', CheckpointDir, ' ...');
    BuildBartFromSafeTensors(ModelPath, Enc, Dec, Config,
      EncSeqLen, DecSeqLen, {pInferenceOnly=}true, ConfigPath);
    WriteLn(BartConfigToString(Config));
    WriteLn('Encoder window ', EncSeqLen, ', decoder window ', DecSeqLen,
      ', beam ', BeamWidth, ', length penalty ',
      FormatFloat('0.00', LengthPenalty, FloatFmt));

    Tok := TNeuralHFTokenizer.Create();
    Tok.LoadFromFile(TokPath);

    // Encode the article and frame it as BART expects: bos ... eos, then
    // truncate to the encoder window (the encoder is built for exactly
    // EncSeqLen tokens; DecodeSeq2SeqBeamSearch pads the rest internally).
    ArticleIds := Tok.Encode(ArticleText);
    MaxArticle := EncSeqLen - 2; // room for bos + eos
    if Length(ArticleIds) > MaxArticle then
      SetLength(ArticleIds, MaxArticle);
    SetLength(SourceIds, Length(ArticleIds) + 2);
    SourceIds[0] := Config.BosTokenId;
    for p := 0 to High(ArticleIds) do SourceIds[p + 1] := ArticleIds[p];
    SourceIds[High(SourceIds)] := Config.EosTokenId;
    WriteLn('Article: ', Length(SourceIds), ' tokens (incl. bos/eos).');

    // BART summarization decoding: decoder_start_token_id is eos (BART's
    // shift_tokens_right), the sequence terminates on eos.
    SummaryIds := DecodeSeq2SeqBeamSearch(Enc, Dec, SourceIds,
      {StartTokenId=}Config.DecoderStartTokenId,
      {EOSTokenId=}Config.EosTokenId,
      {MaxNewTokens=}DecSeqLen - 1, BeamWidth, LengthPenalty);

    SummaryText := Tok.Decode(SummaryIds, {SkipSpecialTokens=}true);
    WriteLn;
    WriteLn('==== SUMMARY ====');
    WriteLn(Trim(SummaryText));
    WriteLn('=================');

    if Trim(ReferenceText) <> '' then
    begin
      R1 := RougeN(SummaryText, ReferenceText, 1);
      R2 := RougeN(SummaryText, ReferenceText, 2);
      RL := RougeL(SummaryText, ReferenceText);
      WriteLn;
      WriteLn('Reference: ', Trim(ReferenceText));
      WriteLn('ROUGE-1 F1: ', FormatFloat('0.0000', R1.F1, FloatFmt));
      WriteLn('ROUGE-2 F1: ', FormatFloat('0.0000', R2.F1, FloatFmt));
      WriteLn('ROUGE-L F1: ', FormatFloat('0.0000', RL.F1, FloatFmt));
    end;
  finally
    Tok.Free;
    Dec.Free;
    Enc.Free;
  end;
end.
