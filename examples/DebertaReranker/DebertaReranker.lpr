program DebertaReranker;
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

// DebertaReranker -- the canonical RAG cross-encoder reranking demo on a
// pretrained DeBERTa-v3 ForSequenceClassification checkpoint (the ms-marco
// family: cross-encoder/ms-marco-... or naver/trecdl... rerankers). A
// cross-encoder concatenates the QUERY and a candidate PASSAGE into ONE
// sequence ([CLS] query [SEP] passage [SEP]) and the classification head
// emits a single relevance LOGIT; the higher the logit the more relevant
// the passage. This is the second-stage reranker of a RAG pipeline: a fast
// bi-encoder / BM25 retriever returns the top-N candidates and this model
// re-scores them jointly with the query (far more accurate than cosine
// similarity of independent embeddings, because every query token can attend
// to every passage token).
//
// The model is imported with BuildDebertaV2FromSafeTensorsEx(...,
// pSeqClsHead=true): the row-0 ([CLS]) classifier logits are the relevance
// scores. DeBERTa-v3's disentangled attention (TNNetDisentangledAttention)
// is the encoder; see the DEBERTA-V2 IMPORT section of neuralpretrained.pas.
//
// Usage (from the repo root):
//   DebertaReranker <model.safetensors> -q "query" -p "passage A" -p "..."
//   DebertaReranker <model.safetensors> [SeqLen] -ids cls q... sep p... sep
//
//   -q "text"   the query (text mode; needs tokenizer.json next to the
//               checkpoint - DeBERTa-v3 ships a Unigram tokenizer.json that
//               the TNeuralHFTokenizer Unigram reader handles).
//   -p "text"   a candidate passage; repeat for several (ranked together).
//   SeqLen      built context window (default 64).
//   -cfg path   config.json path ('' = next to the checkpoint; the pico
//               fixture's config has a _config.json suffix, so pass it).
//   -cls N      [CLS] token id (default 1, the deberta-v3 convention).
//   -sep N      [SEP] token id (default 2).
//   -ids ...    raw token-id mode for the committed pico fixture (the tiny
//               seq-cls fixture has no tokenizer): every -ids list is ONE
//               already-assembled [CLS] q [SEP] p [SEP] id sequence; repeat
//               -ids for several passages.
//
// Try it with the tiny committed seq-cls fixture (num_labels 2; the demo
// uses logit[0] as the relevance score), assembling two raw id sequences:
//   DebertaReranker tests/fixtures/tiny_debertav2_seqcls.safetensors 16 \
//     -cfg tests/fixtures/tiny_debertav2_seqcls_config.json \
//     -ids 1 5 2 7 9 2 0 0 0 0 0 0 0 0 0 0 \
//     -ids 1 5 2 3 4 2 0 0 0 0 0 0 0 0 0 0
// or with a real ms-marco DeBERTa-v3 reranker (tokenizer.json beside it):
//   DebertaReranker /tmp/ms-marco/model.safetensors -q "what is rag?" \
//     -p "Retrieval augmented generation..." -p "An unrelated sentence."

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralnetwork,
  neuralsafetensors, neuralpretrained, neuralhftokenizer;

const
  csDefaultSeqLen = 64;

type
  TCandidate = record
    Text: string;          // passage text (text mode) or '' (id mode)
    Ids: array of integer; // assembled token ids (padded to SeqLen)
    Score: TNeuralFloat;   // relevance logit (row 0, label 0)
  end;

var
  ModelPath, Query: string;
  Cand: array of TCandidate;
  CandCnt, SeqLen, ClsId, SepId, i, j, ArgCnt: integer;
  NN: TNNet;
  Config: TDebertaV2Config;
  Tok: TNeuralHFTokenizer;
  Input, Output: TNNetVolume;
  TokenizerPath, ConfigPath, Arg: string;
  Order: array of integer;
  Tmp: integer;

  procedure AddCandidate(const PassageText: string);
  begin
    SetLength(Cand, CandCnt + 1);
    Cand[CandCnt].Text := PassageText;
    SetLength(Cand[CandCnt].Ids, 0);
    Inc(CandCnt);
  end;

  // Assembles [CLS] query [SEP] passage [SEP], truncated/padded to SeqLen.
  procedure AssembleTextIds(var C: TCandidate);
  var
    QIds, PIds: TNeuralIntegerArray;
    k, n: integer;
  begin
    QIds := Tok.Encode(Query);
    PIds := Tok.Encode(C.Text);
    SetLength(C.Ids, SeqLen);
    for k := 0 to SeqLen - 1 do C.Ids[k] := 0; // pad id 0
    n := 0;
    C.Ids[n] := ClsId; Inc(n);
    for k := 0 to Length(QIds) - 1 do
      if n < SeqLen - 1 then begin C.Ids[n] := QIds[k]; Inc(n); end;
    if n < SeqLen then begin C.Ids[n] := SepId; Inc(n); end;
    for k := 0 to Length(PIds) - 1 do
      if n < SeqLen - 1 then begin C.Ids[n] := PIds[k]; Inc(n); end;
    if n < SeqLen then begin C.Ids[n] := SepId; Inc(n); end;
  end;

begin
  if ParamCount < 1 then
  begin
    WriteLn('Usage: DebertaReranker <model.safetensors> [SeqLen] ' +
      '-q "query" -p "passage" [-p ...]');
    WriteLn('   or: DebertaReranker <model.safetensors> [SeqLen] ' +
      '-ids <id list> [-ids ...]   (raw-id mode for the pico fixture)');
    Halt(1);
  end;
  ModelPath := ParamStr(1);
  SeqLen := csDefaultSeqLen;
  ClsId := 1; // deberta-v3 [CLS]
  SepId := 2; // deberta-v3 [SEP]
  Query := '';
  ConfigPath := ''; // '' = config.json next to the checkpoint
  CandCnt := 0;
  SetLength(Cand, 0);

  // Parse args: an optional leading numeric SeqLen, then -q/-p/-ids/-cls/-sep.
  ArgCnt := 2;
  if (ParamCount >= 2) and (StrToIntDef(ParamStr(2), -1) > 0) then
  begin
    SeqLen := StrToInt(ParamStr(2));
    ArgCnt := 3;
  end;
  while ArgCnt <= ParamCount do
  begin
    Arg := ParamStr(ArgCnt);
    if Arg = '-q' then begin Inc(ArgCnt); Query := ParamStr(ArgCnt); end
    else if Arg = '-p' then begin Inc(ArgCnt); AddCandidate(ParamStr(ArgCnt)); end
    else if Arg = '-cfg' then begin Inc(ArgCnt); ConfigPath := ParamStr(ArgCnt); end
    else if Arg = '-cls' then begin Inc(ArgCnt); ClsId := StrToInt(ParamStr(ArgCnt)); end
    else if Arg = '-sep' then begin Inc(ArgCnt); SepId := StrToInt(ParamStr(ArgCnt)); end
    else if Arg = '-ids' then
    begin
      // Consume the following numeric tokens as one assembled id sequence.
      AddCandidate('');
      SetLength(Cand[CandCnt - 1].Ids, SeqLen);
      for i := 0 to SeqLen - 1 do Cand[CandCnt - 1].Ids[i] := 0;
      i := 0;
      while (ArgCnt + 1 <= ParamCount) and
            (StrToIntDef(ParamStr(ArgCnt + 1), -999999) <> -999999) and
            (Copy(ParamStr(ArgCnt + 1), 1, 1) <> '-') do
      begin
        Inc(ArgCnt);
        if i < SeqLen then Cand[CandCnt - 1].Ids[i] := StrToInt(ParamStr(ArgCnt));
        Inc(i);
      end;
    end;
    Inc(ArgCnt);
  end;
  if CandCnt = 0 then
  begin
    WriteLn('No candidate passages given (-p or -ids).');
    Halt(1);
  end;

  WriteLn('Loading DeBERTa-v3 reranker: ', ModelPath);
  NN := BuildDebertaV2FromSafeTensorsEx(ModelPath, Config, SeqLen,
    {pTrainable=}false, ConfigPath, {pSeqClsHead=}true);
  WriteLn(DebertaV2ConfigToString(Config));
  WriteLn('Labels (output depth): ', NN.GetLastLayer().Output.Depth);

  // Text mode needs the tokenizer.json sibling (DeBERTa-v3 Unigram).
  Tok := nil;
  TokenizerPath := ExtractFilePath(ModelPath) + 'tokenizer.json';
  if (Query <> '') and FileExists(TokenizerPath) then
  begin
    Tok := TNeuralHFTokenizer.Create();
    Tok.LoadFromFile(TokenizerPath);
    WriteLn('Tokenizer: ', TokenizerPath);
  end;

  Input := TNNetVolume.Create(SeqLen, 1, 2);  // ch0 ids, ch1 token-type=0
  Output := TNNetVolume.Create();
  try
    if Query <> '' then WriteLn('Query: ', Query);
    WriteLn('Scoring ', CandCnt, ' candidate passage(s)...');
    for i := 0 to CandCnt - 1 do
    begin
      if (Cand[i].Text <> '') then
      begin
        if Tok = nil then
        begin
          WriteLn('  (text passage given but no tokenizer.json found - skip)');
          Cand[i].Score := -1e30;
          continue;
        end;
        AssembleTextIds(Cand[i]);
      end;
      // Feed the assembled ids (token-type all 0; type_vocab_size=0 in v3).
      for j := 0 to SeqLen - 1 do
      begin
        Input.FData[j * 2] := Cand[i].Ids[j];
        Input.FData[j * 2 + 1] := 0;
      end;
      NN.Compute(Input);
      NN.GetOutput(Output);
      // Row 0 ([CLS]) carries the classifier logits; label 0 is the
      // relevance score (single-logit ms-marco rerankers have num_labels 1).
      Cand[i].Score := Output.FData[0];
    end;

    // Rank descending by relevance.
    SetLength(Order, CandCnt);
    for i := 0 to CandCnt - 1 do Order[i] := i;
    for i := 0 to CandCnt - 2 do
      for j := i + 1 to CandCnt - 1 do
        if Cand[Order[j]].Score > Cand[Order[i]].Score then
        begin
          Tmp := Order[i]; Order[i] := Order[j]; Order[j] := Tmp;
        end;

    WriteLn;
    WriteLn('Reranked passages (most relevant first):');
    for i := 0 to CandCnt - 1 do
    begin
      Write('  #', i + 1, '  score=', FormatFloat('0.0000',
        Cand[Order[i]].Score), '  ');
      if Cand[Order[i]].Text <> '' then WriteLn('"', Cand[Order[i]].Text, '"')
      else WriteLn('(passage ', Order[i], ')');
    end;
  finally
    Output.Free;
    Input.Free;
    if Assigned(Tok) then Tok.Free;
    NN.Free;
  end;
end.
