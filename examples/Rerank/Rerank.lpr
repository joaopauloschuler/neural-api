program Rerank;
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

// Rerank -- the full two-stage retrieve-then-rerank RAG pipeline on CPU:
//   STAGE 1 (recall):    a sentence-transformers BI-ENCODER embeds the query
//                        and every corpus passage independently; cosine
//                        similarity returns the top-K candidates fast.
//   STAGE 2 (precision): a CROSS-ENCODER reranker (a BERT-family
//                        ForSequenceClassification with num_labels=1) re-scores
//                        each top-K candidate JOINTLY with the query -
//                        [CLS] query [SEP] passage [SEP] - and reorders them by
//                        the [CLS] relevance logit (sigmoid). Joint
//                        cross-attention between query and passage tokens is
//                        far more accurate than the bi-encoder's independent
//                        dot product, which is why production RAG always reranks.
//
// The cross-encoder pieces are the new neuralpretrained.pas helpers
// BertTokenizePair / CrossEncoderScore / RerankPassages / RerankReport (see the
// CROSS-ENCODER RERANKER section). The bi-encoder pieces are the existing
// BertEncodeSentence + CosineSimilarity (see SENTENCE EMBEDDINGS).
//
// Usage (real checkpoints):
//   Rerank -bi <biencoder_dir> -ce <crossencoder_dir> [-q "query"] [-k N]
//     -bi dir   sentence-transformers BI-encoder snapshot (model.safetensors +
//               config.json + tokenizer.json, e.g. all-MiniLM-L6-v2).
//     -ce dir   cross-encoder reranker snapshot with num_labels=1
//               (cross-encoder/ms-marco-MiniLM-L-6-v2, BAAI/bge-reranker-base,
//               ...), same three files.
//     -q text   query (default: a paraphrase of corpus #0).
//     -k N      number of bi-encoder candidates handed to the reranker
//               (default 5).
//     -int8     int8-quantize the cross-encoder backbone (the reranker forward
//               supports the BERT pQuantizeInt8 path).
//
// Demo without any download (committed pico fixture):
//   Rerank -demo
// runs the RerankReport BEFORE/AFTER bookkeeping on a tiny hand-wired
// cross-encoder so the pipeline shape is visible offline.

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralnetwork, neuralpretrained, neuralhftokenizer;

const
  csCorpusSize = 8;
  csCorpus: array[0..csCorpusSize - 1] of string = (
    'A cat is sitting on the windowsill in the sun.',
    'The weather forecast says it will rain tomorrow.',
    'She is cooking pasta for dinner tonight.',
    'The stock market dropped sharply this morning.',
    'A dog is sleeping on the porch.',
    'Scientists discovered a new species of frog in the rainforest.',
    'He fixed the leaking kitchen faucet himself.',
    'The orchestra performed a beautiful symphony last night.');
  csDefaultQuery = 'A kitty rests on the window ledge enjoying the sunshine.';

var
  BiDir, CeDir, BiWeights, CeWeights, CePath, Query: string;
  TopK: integer;
  UseInt8, DemoMode: boolean;
  BiTok, CeTok: TNeuralHFTokenizer;
  CeNet: TNNet;
  CeConfig: TBertConfig;
  // per-distinct-length bi-encoder nets (no attention mask -> exact parity,
  // see examples/SemanticSearch).
  BiLens: array of integer;
  BiNets: array of TNNet;
  CorpusEmb: array[0..csCorpusSize - 1] of TNNetVolume;
  QueryEmb: TNNetVolume;
  Sims: array[0..csCorpusSize - 1] of TNeuralFloat;
  Order: array[0..csCorpusSize - 1] of integer;
  ParamPos, i, j, Tmp: integer;
  Candidates: array of string;
  CandOrigIdx: array of integer;
  RerankOrder: TNeuralIntegerArray;
  RerankScores: TNeuralFloatDynArr;

function BiNetForLength(TokenCount: integer): TNNet;
var Cnt: integer;
begin
  for Cnt := 0 to High(BiLens) do
    if BiLens[Cnt] = TokenCount then Exit(BiNets[Cnt]);
  SetLength(BiLens, Length(BiLens) + 1);
  SetLength(BiNets, Length(BiNets) + 1);
  BiLens[High(BiLens)] := TokenCount;
  BiNets[High(BiNets)] := BuildBertFromSafeTensors(BiWeights, TokenCount,
    {pInferenceOnly=}true);
  Result := BiNets[High(BiNets)];
end;

procedure BiEmbed(const Text: string; Emb: TNNetVolume);
var Ids: TNeuralIntegerArray;
begin
  Ids := BertTokenizeSentence(BiTok, Text);
  BertEncodeSentence(BiNetForLength(Length(Ids)), BiTok, Text, Emb);
end;

procedure RunDemo;
var
  Tok: TNeuralHFTokenizer;
  NN: TNNet;
  Emb, Cls: TNNetLayer;
  V, k, GoldId: integer;
  Passages: array[0..2] of string;
  Rel: TNeuralIntegerArray;
  KList: array[0..0] of integer;
begin
  // A tiny hand-wired num_labels=1 cross-encoder: relevance logit = mean-pooled
  // embedding of the joint sequence; only the GOLD passage carries the marker
  // word. Mirrors tests/TestNeuralPretrained TestRerankReportLift.
  Tok := TNeuralHFTokenizer.Create();
  NN := TNNet.Create();
  try
    Tok.LoadFromFile('../../tests/fixtures/tiny_wordpiece_tokenizer.json');
    V := 255;
    NN.AddLayer(TNNetInput.Create(16, 1, 2));
    NN.AddLayer(TNNetSplitChannels.Create([0]));
    Emb := NN.AddLayer(TNNetEmbedding.Create(V, 4, 1));
    NN.AddLayer(TNNetAvgChannel.Create());
    Cls := NN.AddLayer(TNNetPointwiseConvLinear.Create(1));
    for k := 0 to 3 do Cls.Neurons[0].Weights.FData[k] := 1.0;
    Cls.Neurons[0].BiasWeight := 0; Cls.FlushWeightCache();
    for k := 0 to V * 4 - 1 do Emb.Neurons[0].Weights.FData[k] := 0.0;
    GoldId := Tok.TokenToId('fox');
    for k := 0 to 3 do Emb.Neurons[0].Weights.FData[GoldId * 4 + k] := 9.0;
    Emb.FlushWeightCache();
    Passages[0] := 'the quick brown dog';
    Passages[1] := 'lazy river runs slow';
    Passages[2] := 'a clever fox appears'; // gold, sits LAST initially
    SetLength(Rel, 1); Rel[0] := 2;
    KList[0] := 1;
    WriteLn('=== Rerank demo (committed pico fixture, no download) ===');
    WriteLn('query: "what jumped"');
    WriteLn('candidates (bi-encoder order):');
    for k := 0 to 2 do WriteLn('  #', k, ' ', Passages[k]);
    WriteLn;
    Write(RerankReport(NN, Tok, 'what jumped', Passages, Rel, KList));
  finally
    NN.Free; Tok.Free;
  end;
end;

begin
  Query := csDefaultQuery;
  TopK := 5;
  UseInt8 := false;
  DemoMode := false;
  BiDir := ''; CeDir := '';
  ParamPos := 1;
  while ParamPos <= ParamCount do
  begin
    case ParamStr(ParamPos) of
      '-demo': DemoMode := true;
      '-int8': UseInt8 := true;
      '-bi': begin Inc(ParamPos); BiDir := ParamStr(ParamPos); end;
      '-ce': begin Inc(ParamPos); CeDir := ParamStr(ParamPos); end;
      '-q':  begin Inc(ParamPos); Query := ParamStr(ParamPos); end;
      '-k':  begin Inc(ParamPos); TopK := StrToIntDef(ParamStr(ParamPos), 5); end;
    end;
    Inc(ParamPos);
  end;

  if DemoMode or ((BiDir = '') and (CeDir = '')) then
  begin
    RunDemo;
    Exit;
  end;

  if (BiDir = '') or (CeDir = '') then
  begin
    WriteLn('Usage: Rerank -bi <biencoder_dir> -ce <crossencoder_dir> ',
      '[-q "query"] [-k N] [-int8]   (or: Rerank -demo)');
    Halt(1);
  end;

  BiWeights := IncludeTrailingPathDelimiter(BiDir) + 'model.safetensors';
  CeWeights := IncludeTrailingPathDelimiter(CeDir) + 'model.safetensors';
  BiTok := TNeuralHFTokenizer.Create();
  CeTok := TNeuralHFTokenizer.Create();
  QueryEmb := TNNetVolume.Create();
  for i := 0 to csCorpusSize - 1 do CorpusEmb[i] := TNNetVolume.Create();
  try
    BiTok.LoadFromFile(IncludeTrailingPathDelimiter(BiDir) + 'tokenizer.json');
    CeTok.LoadFromFile(IncludeTrailingPathDelimiter(CeDir) + 'tokenizer.json');

    // ---- STAGE 1: bi-encoder recall ----
    WriteLn('Query: ', Query);
    WriteLn;
    WriteLn('=== Stage 1: bi-encoder cosine retrieval ===');
    BiEmbed(Query, QueryEmb);
    for i := 0 to csCorpusSize - 1 do
    begin
      BiEmbed(csCorpus[i], CorpusEmb[i]);
      Sims[i] := CosineSimilarity(QueryEmb, CorpusEmb[i]);
      Order[i] := i;
    end;
    for i := 1 to csCorpusSize - 1 do
    begin
      Tmp := Order[i]; j := i - 1;
      while (j >= 0) and (Sims[Order[j]] < Sims[Tmp]) do
      begin Order[j + 1] := Order[j]; Dec(j); end;
      Order[j + 1] := Tmp;
    end;
    if TopK > csCorpusSize then TopK := csCorpusSize;
    for i := 0 to TopK - 1 do
      WriteLn('  rank ', i + 1, '  cos=',
        FloatToStrF(Sims[Order[i]], ffFixed, 6, 4), '  ', csCorpus[Order[i]]);

    // ---- STAGE 2: cross-encoder rerank of the top-K ----
    WriteLn;
    WriteLn('=== Stage 2: cross-encoder joint rerank (top-', TopK, ') ===');
    CePath := IncludeTrailingPathDelimiter(CeDir) + 'config.json';
    CeNet := BuildBertForSequenceClassificationFromSafeTensorsEx(CeWeights,
      CeConfig, nil, {SeqLen=}0, {pInferenceOnly=}true, CePath, UseInt8);
    try
      if CeNet.GetLastLayer().Output.Depth <> 1 then
        WriteLn('  WARNING: cross-encoder is not num_labels=1 (depth=',
          CeNet.GetLastLayer().Output.Depth, '); using the row-0 logit.');
      SetLength(Candidates, TopK);
      SetLength(CandOrigIdx, TopK);
      for i := 0 to TopK - 1 do
      begin
        Candidates[i] := csCorpus[Order[i]];
        CandOrigIdx[i] := Order[i];
      end;
      RerankPassages(CeNet, CeTok, Query, Candidates, RerankOrder,
        RerankScores);
      for i := 0 to TopK - 1 do
        WriteLn('  rank ', i + 1, '  rel=',
          FloatToStrF(RerankScores[i], ffFixed, 6, 4), '  ',
          Candidates[RerankOrder[i]]);
    finally
      CeNet.Free;
    end;
  finally
    for i := 0 to csCorpusSize - 1 do CorpusEmb[i].Free;
    QueryEmb.Free;
    for i := 0 to High(BiNets) do BiNets[i].Free;
    CeTok.Free;
    BiTok.Free;
  end;
end.
