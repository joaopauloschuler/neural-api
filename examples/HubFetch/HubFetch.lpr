// HubFetch example
//
// One-command HuggingFace Hub download -> local snapshot -> imported TNNet.
//
// Until now every importer example (GPT2Import, LlamaImport, SemanticSearch)
// began with "hand-download these files with python/huggingface-cli". The
// opt-in neuralhfhub.pas unit removes that step: HubFetchModel(repo) pulls
// config.json + tokenizer.json (when present) + the safetensors weights --
// transparently following model.safetensors.index.json to its shards when
// the checkpoint is sharded -- into a local cache (skip-if-present, so the
// second run is instant and offline) and returns the snapshot directory
// that BuildFromPretrained accepts. The core importers stay offline-only:
// ONLY programs that use neuralhfhub touch the network / OpenSSL.
//
// Usage:
//   HubFetch                  -> fetches hf-internal-testing/
//                                tiny-random-bert-sharded-safetensors, a
//                                ~100KB FIVE-SHARD checkpoint that exercises
//                                the index-json fallback, and builds it.
//   HubFetch <repo-id> [rev]  -> e.g. HubFetch prajjwal1/bert-tiny or
//                                HubFetch roneneldan/TinyStories-1M (note:
//                                both of those publish only
//                                pytorch_model.bin, no safetensors, so
//                                weight fetch fails with a clear error;
//                                sentence-transformers/all-MiniLM-L6-v2
//                                (~90MB) works end to end).
//
// Gated repos: export HF_TOKEN=hf_... before running.
// Cache: ~/.cache/neural-api/hub (override: NEURAL_API_HUB_CACHE env or
// HubSetCacheDir).
//
// This example is coded by Claude (AI).
program HubFetch;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  SysUtils,
  neuralnetwork,
  neuralhfhub,
  neuralpretrained;

var
  RepoId, Revision, SnapshotDir: string;
  Net: TNNet;
begin
  RepoId := 'hf-internal-testing/tiny-random-bert-sharded-safetensors';
  Revision := 'main';
  if ParamCount >= 1 then RepoId := ParamStr(1);
  if ParamCount >= 2 then Revision := ParamStr(2);

  WriteLn('Cache dir : ', HubGetCacheDir);
  WriteLn('Fetching  : ', RepoId, ' @ ', Revision);
  SnapshotDir := HubFetchModel(RepoId, Revision);
  WriteLn('Snapshot  : ', SnapshotDir);

  Net := BuildFromPretrained(SnapshotDir, {pSeqLen=}16,
    {pTrainable=}false);
  try
    WriteLn('Imported  : ', Net.CountLayers, ' layers, ',
      Net.CountWeights, ' weights.');
    WriteLn('Run me again: everything is cached, no network needed.');
  finally
    Net.Free;
  end;
end.
