program ColBERTSearch;
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

// ColBERTSearch -- ColBERT late-interaction retrieval (colbert-ir/colbertv2.0)
// on CPU with an imported ColBERT checkpoint.
//
// ColBERT is the third RAG retrieval paradigm next to the bi-encoder
// (examples/SemanticSearch: one pooled vector + cosine) and the cross-encoder
// (examples/DebertaReranker: a joint q+passage scorer). It keeps the PER-TOKEN
// contextual embeddings of query and document (NO pooling), projects each
// token to a small dim (128) + L2-normalizes, and scores a (query, doc) pair
// by the MaxSim late-interaction sum
//   score = sum_{q in query} max_{d in doc} <E_q, E_d>
// -- cross-encoder-grade accuracy at bi-encoder cost: documents are encoded
// ONCE (their per-token matrices cached), then every query is scored by MaxSim
// against the cached matrices.
//
// Pipeline (see neuralpretrained.pas COLBERT LATE INTERACTION):
//   tokenizer.json WordPiece encode
//   -> [CLS][D] doc tokens [SEP] (docs)  /  [CLS][Q] query [SEP] [MASK]... (queries)
//   -> BuildColBERTFromSafeTensors encoder + [hidden->128] bias-free linear head
//   -> per-token projected + L2-normalized matrix (ColBERTEmbedTokens)
//   -> ColBERTMaxSimScore(query, doc) ranks the corpus.
//
// NOTE: the imported encoder carries NO attention padding mask. Documents are
// [PAD]-filled to the net's SeqLen and the pad rows are skipped from MaxSim,
// but real tokens still attend to pad positions, so a short doc in a long net
// is an approximation (exactly as examples/SemanticSearch documents). Build
// the net with -seqlen close to the real corpus length for best fidelity.
//
// Usage:
//   ColBERTSearch <modeldir> [-q "query text"] [-seqlen N]
//
//   modeldir  - directory with model.safetensors (or pytorch_model.bin) +
//               config.json + tokenizer.json of a ColBERT checkpoint
//               (must carry the "linear.weight" [128, hidden] projection head).
//   -q text   - query to rank the built-in corpus against.
//   -seqlen N - encoder context length (default 32).

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
  ModelDir, WeightsPath, Query: string;
  Tokenizer: TNeuralHFTokenizer;
  Net: TNNet;
  Markers: TColBERTMarkers;
  DocMats: array[0..csCorpusSize - 1] of TNNetVolume;
  QueryMat: TNNetVolume;
  ParamPos, SeqLen, DocCnt, BestIdx, i, j, TmpI: integer;
  Score, BestScore: TNeuralFloat;
  Scores: array[0..csCorpusSize - 1] of TNeuralFloat;
  Order: array[0..csCorpusSize - 1] of integer;

begin
  DefaultFormatSettings.DecimalSeparator := '.';
  if ParamCount < 1 then
  begin
    WriteLn('Usage: ColBERTSearch <modeldir> [-q "query"] [-seqlen N]');
    WriteLn('modeldir needs a ColBERT checkpoint: model.safetensors (with the');
    WriteLn('"linear.weight" projection head) + config.json + tokenizer.json.');
    WriteLn('See examples/ColBERTSearch/README.md.');
    Halt(1);
  end;
  ModelDir := ParamStr(1);
  WeightsPath := IncludeTrailingPathDelimiter(ModelDir) + 'model.safetensors';
  if not FileExists(WeightsPath) then
    WeightsPath := IncludeTrailingPathDelimiter(ModelDir) +
      'pytorch_model.bin';
  Query := csDefaultQuery;
  SeqLen := 32;
  ParamPos := 2;
  while ParamPos <= ParamCount do
  begin
    if (ParamStr(ParamPos) = '-q') and (ParamPos < ParamCount) then
    begin
      Inc(ParamPos); Query := ParamStr(ParamPos);
    end
    else if (ParamStr(ParamPos) = '-seqlen') and (ParamPos < ParamCount) then
    begin
      Inc(ParamPos); SeqLen := StrToInt(ParamStr(ParamPos));
    end
    else
    begin
      WriteLn('Unknown argument: ', ParamStr(ParamPos)); Halt(1);
    end;
    Inc(ParamPos);
  end;

  Tokenizer := TNeuralHFTokenizer.Create();
  Tokenizer.LoadFromFile(
    IncludeTrailingPathDelimiter(ModelDir) + 'tokenizer.json');
  WriteLn('Tokenizer loaded: vocab ', Tokenizer.GetVocabSize());

  WriteLn('Building ColBERT encoder (SeqLen ', SeqLen, ') ...');
  Net := BuildColBERTFromSafeTensors(WeightsPath, SeqLen,
    {pInferenceOnly=}true);
  Markers := ColBERTDefaultMarkers(Tokenizer);

  // Pre-encode the corpus ONCE: per-token L2-normalized doc matrices.
  WriteLn('Pre-encoding ', csCorpusSize, ' passages ...');
  for DocCnt := 0 to csCorpusSize - 1 do
  begin
    DocMats[DocCnt] := TNNetVolume.Create();
    ColBERTEmbedTokens(Net, Tokenizer, csCorpus[DocCnt], {IsQuery=}false,
      Markers, DocMats[DocCnt]);
  end;

  // Encode the query and score every passage by MaxSim.
  QueryMat := TNNetVolume.Create();
  try
    ColBERTEmbedTokens(Net, Tokenizer, Query, {IsQuery=}true, Markers,
      QueryMat);
    WriteLn;
    WriteLn('Query: ', Query);
    WriteLn('-----------------------------------------------------------');
    BestIdx := 0; BestScore := -1e30;
    for DocCnt := 0 to csCorpusSize - 1 do
    begin
      Scores[DocCnt] := ColBERTMaxSimScore(QueryMat, DocMats[DocCnt]);
      Order[DocCnt] := DocCnt;
      if Scores[DocCnt] > BestScore then
      begin
        BestScore := Scores[DocCnt]; BestIdx := DocCnt;
      end;
    end;
    // descending sort (insertion) for the ranked listing
    for i := 1 to csCorpusSize - 1 do
    begin
      TmpI := Order[i]; j := i - 1;
      while (j >= 0) and (Scores[Order[j]] < Scores[TmpI]) do
      begin
        Order[j + 1] := Order[j]; Dec(j);
      end;
      Order[j + 1] := TmpI;
    end;
    for i := 0 to csCorpusSize - 1 do
      WriteLn('  MaxSim=', FloatToStrF(Scores[Order[i]], ffFixed, 8, 4),
        '  ', csCorpus[Order[i]]);
    WriteLn('-----------------------------------------------------------');
    Score := BestScore;
    WriteLn('Best match (MaxSim=', FloatToStrF(Score, ffFixed, 8, 4), '): ',
      csCorpus[BestIdx]);
  finally
    QueryMat.Free;
    for DocCnt := 0 to csCorpusSize - 1 do DocMats[DocCnt].Free;
    Net.Free;
    Tokenizer.Free;
  end;
end.
