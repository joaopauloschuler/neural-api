program EmbeddingSearch;
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

// EmbeddingSearch -- the E5 / BGE retriever path through the landed
// neuralpretrained.pas SENTENCE EMBEDDINGS helpers: the POOLING-MODE
// selector (TNNetEmbedPooling) plus the INSTRUCTION-PREFIX table
// (TNNetEmbedInstruction). It is the self-contained sibling of
// examples/SemanticSearch (which needs a ~90 MB MiniLM download and only
// covers the mean-pool-no-prefix MiniLM case): this demo runs entirely on
// the committed pico fixture tests/fixtures/tiny_e5.* so it is CPU-fast
// (<1 s) and needs no network.
//
// It shows two things the MiniLM demo does not:
//   1. ApplyEmbedInstruction prepends the family-specific prefix that is
//      MANDATORY for parity -- E5's "query: "/"passage: " and BGE's
//      "Represent this sentence ..." query instruction (printed below);
//   2. PoolSentenceEmbedding wraps {forward -> pool(mode) -> L2 normalize}
//      for ANY of CLS (BGE), mean (E5/GTE), last-token (e5-mistral).
//
// Because the pico fixture ships no tokenizer.json, the instruction prefix
// is BAKED into the leading token ids of each sequence (ids 1/2/3 stand in
// for the "query: "/"passage: " prefix tokens) -- exactly the sequences
// the HF float64 parity oracle was computed from (tiny_e5_embed.json,
// TestE5EmbeddingParity). With a real downloaded E5 checkpoint you would
// instead tokenize ApplyEmbedInstruction(efE5, IsQuery, Text). The query
// and the shared-body passage win over the unrelated passage.
//
// Usage:
//   EmbeddingSearch [fixturedir]
//   fixturedir - directory holding tiny_e5.safetensors + tiny_e5_config.json
//                (default: tests/fixtures relative to the repo root).

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralnetwork, neuralpretrained;

const
  // The query (row 0) + two passages, baked-prefix + body token ids, padded
  // to 16 with [PAD]=0. Mirrors tiny_e5_embed.json exactly so the vectors
  // equal the HF float64 oracle. Leading ids 1/2 == "query: ", 1/3 ==
  // "passage: ".
  csSeqLen = 16;
  csCount = 3;
  csSequences: array[0..csCount - 1, 0..csSeqLen - 1] of integer = (
    (1, 2, 5, 8, 3, 9, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0),     // query
    (1, 3, 5, 8, 3, 9, 4, 7, 0, 0, 0, 0, 0, 0, 0, 0),     // passage 0 (shared body)
    (1, 3, 11, 6, 12, 10, 2, 8, 5, 0, 0, 0, 0, 0, 0, 0)); // passage 1 (unrelated)
  csRealTokens: array[0..csCount - 1] of integer = (7, 8, 9);
  csLabels: array[0..csCount - 1] of string = (
    'query    (baked "query: " prefix)',
    'passage0 (baked "passage: " prefix, shares the query body)',
    'passage1 (baked "passage: " prefix, unrelated body)');

var
  FixDir, WeightsPath, ConfigPath: string;
  NN: TNNet;
  Config: TBertConfig;
  Input, Hidden: TNNetVolume;
  Embs: array[0..csCount - 1] of TNNetVolume;
  Cnt, PosCnt, BestIdx: integer;
  Sim, BestSim: TNeuralFloat;
  Order: array[0..csCount - 1] of integer;
  TmpI: integer;

begin
  DefaultFormatSettings.DecimalSeparator := '.';
  if ParamCount >= 1 then FixDir := ParamStr(1)
  else FixDir := 'tests' + DirectorySeparator + 'fixtures';
  WeightsPath := IncludeTrailingPathDelimiter(FixDir) + 'tiny_e5.safetensors';
  ConfigPath := IncludeTrailingPathDelimiter(FixDir) + 'tiny_e5_config.json';
  if not FileExists(WeightsPath) then
  begin
    WriteLn('Fixture not found: ', WeightsPath);
    WriteLn('Run: python3 tools/e5_embed_tiny_fixture.py  (from the repo root)');
    WriteLn('or pass the fixture directory: EmbeddingSearch <dir>');
    Halt(1);
  end;

  // Show the instruction-prefix table (efE5 / efBGE) - the mandatory-for-
  // parity strings the helper prepends for a REAL checkpoint.
  WriteLn('Instruction-prefix table (neuralpretrained.pas):');
  WriteLn('  E5  query   : "', EmbedInstructionPrefix(efE5, True), '"');
  WriteLn('  E5  passage : "', EmbedInstructionPrefix(efE5, False), '"');
  WriteLn('  BGE query   : "', EmbedInstructionPrefix(efBGE, True), '"');
  WriteLn('  efE5 applied: "',
    ApplyEmbedInstruction(efE5, True, 'how tall is mount everest'), '"');
  WriteLn;

  NN := BuildBertFromSafeTensorsEx(WeightsPath, Config, {SeqLen=}csSeqLen,
    {pTrainable=}false, {pIncludePooler=}false, ConfigPath);
  Input := TNNetVolume.Create();
  Hidden := TNNetVolume.Create();
  try
    WriteLn('Imported pico E5 encoder: ', Config.NumLayers, ' layers, hidden ',
      Config.HiddenSize, '. Pooling = mean + L2-normalize (E5 recipe).');
    WriteLn;
    Input.ReSize(csSeqLen, 1, 2);
    for Cnt := 0 to csCount - 1 do
    begin
      Input.Fill(0); // token-type channel stays zero (single segment)
      for PosCnt := 0 to csSeqLen - 1 do
        Input.FData[PosCnt * 2] := csSequences[Cnt][PosCnt];
      NN.Compute(Input);
      NN.GetOutput(Hidden);
      Embs[Cnt] := TNNetVolume.Create();
      // the pooling-mode selector: epMean for E5/GTE (epCLS for BGE,
      // epLastToken for e5-mistral).
      PoolSentenceEmbedding(Hidden, csRealTokens[Cnt], epMean, Embs[Cnt],
        {Normalize=}True);
      WriteLn('Embedded (', Embs[Cnt].Size, ' dims): ', csLabels[Cnt]);
    end;

    WriteLn;
    WriteLn('Ranking passages against the query (cosine = dot, unit vectors):');
    for Cnt := 1 to csCount - 1 do Order[Cnt] := Cnt;
    // selection sort by cosine, descending (passages only, rows 1..)
    for Cnt := 1 to csCount - 2 do
    begin
      BestIdx := Cnt;
      BestSim := CosineSimilarity(Embs[Order[Cnt]], Embs[0]);
      for PosCnt := Cnt + 1 to csCount - 1 do
        if CosineSimilarity(Embs[Order[PosCnt]], Embs[0]) > BestSim then
        begin
          BestIdx := PosCnt;
          BestSim := CosineSimilarity(Embs[Order[PosCnt]], Embs[0]);
        end;
      TmpI := Order[Cnt]; Order[Cnt] := Order[BestIdx]; Order[BestIdx] := TmpI;
    end;
    for Cnt := 1 to csCount - 1 do
    begin
      Sim := CosineSimilarity(Embs[Order[Cnt]], Embs[0]);
      WriteLn(Format('  %d. %7.4f  %s', [Cnt, Sim, csLabels[Order[Cnt]]]));
    end;
    if Order[1] = 1 then
      WriteLn('OK: the shared-body passage ranks first.')
    else
      WriteLn('UNEXPECTED: shared-body passage did not rank first.');

    for Cnt := 0 to csCount - 1 do Embs[Cnt].Free;
  finally
    Hidden.Free;
    Input.Free;
    NN.Free;
  end;
end.
