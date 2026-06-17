(*
neuralhfhub
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

// neuralhfhub -- OPT-IN HuggingFace Hub download helper.
//
// The pretrained-checkpoint importers in neuralpretrained.pas are strictly
// offline: they take local file/directory paths and never touch the
// network. This unit is the small online companion that turns a Hub repo
// id into those local paths, so
//
//   Net := BuildFromPretrained(HubFetchModel('prajjwal1/bert-tiny'));
//
// works end to end. neuralpretrained.pas does NOT depend on this unit;
// only programs that explicitly add neuralhfhub to their uses clause pull
// in fphttpclient/OpenSSL.
//
// What it does:
//   * HubFetchFile(Repo, File) downloads
//       https://huggingface.co/{repo}/resolve/{rev}/{file}
//     (rev defaults to 'main') into a local cache directory and returns
//     the local path. Files already in the cache are NOT re-downloaded.
//   * HubFetchModel(Repo) fetches config.json, tokenizer.json (when the
//     repo has one) and the weights -- model.safetensors first, then the
//     sharded model.safetensors.index.json, then the torch.save
//     pytorch_model.bin and the sharded pytorch_model.bin.index.json
//     (following either index to its shards) -- and returns the local
//     snapshot DIRECTORY, which is exactly what BuildFromPretrained
//     accepts.
//   * Gated repos: pass a token explicitly or set the HF_TOKEN
//     environment variable; it is sent as "Authorization: Bearer ...".
//   * Cache layout: {cache}/{repo}/{revision}/{file}. The cache root
//     defaults to ~/.cache/neural-api/hub (XDG_CACHE_HOME aware) and can
//     be overridden with HubSetCacheDir or the NEURAL_API_HUB_CACHE
//     environment variable.
//
// Robustness notes:
//   * resolve/ URLs redirect to the CDN, so AllowRedirect is on.
//   * Downloads go to "<name>.part" first and are renamed only on
//     success, so an interrupted download never poisons the cache.
//   * 404s on OPTIONAL files (tokenizer.json, model.safetensors when the
//     checkpoint is sharded) are handled via HubTryFetchFile; everything
//     else raises EHubError with the HTTP status.

unit neuralhfhub;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

type
  EHubError = class(Exception);

  TStringArray = array of string;

// ---- cache directory -----------------------------------------------------
// Resolution order: HubSetCacheDir override > NEURAL_API_HUB_CACHE env >
// $XDG_CACHE_HOME/neural-api/hub > ~/.cache/neural-api/hub.
function HubGetCacheDir: string;
procedure HubSetCacheDir(const Dir: string); // '' restores the default

// ---- pure helpers (no network) ---------------------------------------------
// https://huggingface.co/{repo}/resolve/{rev}/{file}
function HubResolveURL(const RepoId, FileName: string;
  const Revision: string = 'main'): string;
// {cache}/{repo}/{rev}/{file} -- where HubFetchFile will put the file.
function HubLocalPath(const RepoId, FileName: string;
  const Revision: string = 'main'): string;
// Parses a model.safetensors.index.json (or pytorch_model.bin.index.json -
// same layout) TEXT and returns the unique shard file names referenced by
// its "weight_map", sorted. Raises EHubError on malformed input.
function HubShardListFromIndexJson(const IndexJsonText: string): TStringArray;

// ---- downloads -------------------------------------------------------------
// Downloads one file (skip-if-present) and returns its local path.
// Raises EHubError on any HTTP/network failure (including 404).
function HubFetchFile(const RepoId, FileName: string;
  const Revision: string = 'main'; const Token: string = '';
  ForceDownload: boolean = false): string;
// Same, but a 404 returns false instead of raising (other errors still
// raise). Used for optional files.
function HubTryFetchFile(const RepoId, FileName: string;
  out LocalPath: string; const Revision: string = 'main';
  const Token: string = ''; ForceDownload: boolean = false): boolean;
// Fetches config.json + tokenizer.json (if present) + the weights
// (model.safetensors, sharded safetensors index, pytorch_model.bin or
// sharded .bin index - single-file or index + shards) and returns the
// local snapshot directory -- pass it straight to BuildFromPretrained /
// BuildLlamaFromSafeTensors(dir + '/model.safetensors') etc.
function HubFetchModel(const RepoId: string; const Revision: string = 'main';
  const Token: string = ''): string;

implementation

uses
  fphttpclient, opensslsockets, fpjson, jsonparser;

var
  CacheDirOverride: string = '';

// Rejects repo ids / file names that could escape the cache directory.
procedure CheckPathComponent(const S, What: string);
var
  I: integer;
  PartsCount: integer;
  Parts: TStringList;
begin
  if S = '' then
    raise EHubError.Create('neuralhfhub: empty ' + What + '.');
  if (Pos('\', S) > 0) or (S[1] = '/') or (S[Length(S)] = '/') then
    raise EHubError.Create('neuralhfhub: invalid ' + What + ' "' + S + '".');
  Parts := TStringList.Create;
  try
    Parts.Delimiter := '/';
    Parts.StrictDelimiter := True;
    Parts.DelimitedText := S;
    PartsCount := Parts.Count;
    for I := 0 to PartsCount - 1 do
      if (Parts[I] = '') or (Parts[I] = '.') or (Parts[I] = '..') then
        raise EHubError.Create('neuralhfhub: invalid ' + What + ' "' + S +
          '".');
  finally
    Parts.Free;
  end;
end;

function HubGetCacheDir: string;
var
  Base: string;
begin
  if CacheDirOverride <> '' then Exit(CacheDirOverride);
  Result := GetEnvironmentVariable('NEURAL_API_HUB_CACHE');
  if Result <> '' then Exit(Result);
  Base := GetEnvironmentVariable('XDG_CACHE_HOME');
  if Base = '' then
    Base := IncludeTrailingPathDelimiter(GetUserDir) + '.cache';
  Result := IncludeTrailingPathDelimiter(Base) + 'neural-api' +
    DirectorySeparator + 'hub';
end;

procedure HubSetCacheDir(const Dir: string);
begin
  CacheDirOverride := Dir;
end;

function HubResolveURL(const RepoId, FileName: string;
  const Revision: string): string;
begin
  CheckPathComponent(RepoId, 'repo id');
  CheckPathComponent(FileName, 'file name');
  CheckPathComponent(Revision, 'revision');
  Result := 'https://huggingface.co/' + RepoId + '/resolve/' + Revision +
    '/' + FileName;
end;

function HubLocalPath(const RepoId, FileName: string;
  const Revision: string): string;
begin
  CheckPathComponent(RepoId, 'repo id');
  CheckPathComponent(FileName, 'file name');
  CheckPathComponent(Revision, 'revision');
  Result := IncludeTrailingPathDelimiter(HubGetCacheDir) +
    StringReplace(RepoId, '/', DirectorySeparator, [rfReplaceAll]) +
    DirectorySeparator +
    StringReplace(Revision, '/', DirectorySeparator, [rfReplaceAll]) +
    DirectorySeparator +
    StringReplace(FileName, '/', DirectorySeparator, [rfReplaceAll]);
end;

function HubShardListFromIndexJson(const IndexJsonText: string): TStringArray;
var
  Root: TJSONData;
  WeightMap: TJSONData;
  Shards: TStringList;
  I: integer;
  WeightMapCount, ShardsCount: integer;
begin
  Root := nil;
  Shards := TStringList.Create;
  try
    Shards.Sorted := True;
    Shards.Duplicates := dupIgnore;
    try
      Root := GetJSON(IndexJsonText);
    except
      on E: Exception do
        raise EHubError.Create(
          'neuralhfhub: index json is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      raise EHubError.Create('neuralhfhub: index json is not an object.');
    WeightMap := TJSONObject(Root).Find('weight_map');
    if not (WeightMap is TJSONObject) then
      raise EHubError.Create(
        'neuralhfhub: index json has no "weight_map" object.');
    WeightMapCount := TJSONObject(WeightMap).Count;
    if WeightMapCount = 0 then
      raise EHubError.Create('neuralhfhub: index "weight_map" is empty.');
    for I := 0 to WeightMapCount - 1 do
    begin
      if not (TJSONObject(WeightMap).Items[I].JSONType = jtString) then
        raise EHubError.Create('neuralhfhub: index weight_map entry "' +
          TJSONObject(WeightMap).Names[I] + '" is not a string.');
      Shards.Add(TJSONObject(WeightMap).Items[I].AsString);
    end;
    ShardsCount := Shards.Count;
    SetLength(Result, ShardsCount);
    for I := 0 to ShardsCount - 1 do
      Result[I] := Shards[I];
  finally
    Shards.Free;
    Root.Free;
  end;
end;

// Core download. Returns true on success, false on 404 when Allow404,
// raises EHubError otherwise.
function DoFetch(const RepoId, FileName, Revision, Token: string;
  ForceDownload, Allow404: boolean; out LocalPath: string): boolean;
var
  URL, PartPath, UseToken: string;
  Client: TFPHTTPClient;
  FileStream: TFileStream;
  Status: integer;
begin
  LocalPath := HubLocalPath(RepoId, FileName, Revision);
  if (not ForceDownload) and FileExists(LocalPath) then
    Exit(True); // cache hit: no network at all
  URL := HubResolveURL(RepoId, FileName, Revision);
  UseToken := Token;
  if UseToken = '' then UseToken := GetEnvironmentVariable('HF_TOKEN');
  if not ForceDirectories(ExtractFileDir(LocalPath)) then
    raise EHubError.Create('neuralhfhub: cannot create cache directory ' +
      ExtractFileDir(LocalPath) + '.');
  PartPath := LocalPath + '.part';
  Client := TFPHTTPClient.Create(nil);
  try
    Client.AllowRedirect := True;
    Client.MaxRedirects := 10;
    Client.IOTimeout := 60000;
    Client.AddHeader('User-Agent', 'neural-api/neuralhfhub');
    if UseToken <> '' then
      Client.AddHeader('Authorization', 'Bearer ' + UseToken);
    FileStream := TFileStream.Create(PartPath, fmCreate);
    try
      try
        Client.Get(URL, FileStream);
        Status := Client.ResponseStatusCode;
      except
        on E: Exception do
        begin
          Status := Client.ResponseStatusCode;
          if (Status = 404) and Allow404 then
          begin
            FreeAndNil(FileStream);
            DeleteFile(PartPath);
            Exit(False);
          end;
          FreeAndNil(FileStream);
          DeleteFile(PartPath);
          raise EHubError.Create('neuralhfhub: GET ' + URL + ' failed' +
            ' (HTTP ' + IntToStr(Status) + '): ' + E.Message);
        end;
      end;
    finally
      FileStream.Free;
    end;
    if Status <> 200 then
    begin
      DeleteFile(PartPath);
      if (Status = 404) and Allow404 then Exit(False);
      raise EHubError.Create('neuralhfhub: GET ' + URL +
        ' returned HTTP ' + IntToStr(Status) + '.');
    end;
  finally
    Client.Free;
  end;
  if ForceDownload and FileExists(LocalPath) then
    DeleteFile(LocalPath);
  if not RenameFile(PartPath, LocalPath) then
  begin
    DeleteFile(PartPath);
    raise EHubError.Create('neuralhfhub: cannot rename ' + PartPath +
      ' to ' + LocalPath + '.');
  end;
  Result := True;
end;

function HubFetchFile(const RepoId, FileName: string;
  const Revision: string; const Token: string;
  ForceDownload: boolean): string;
begin
  DoFetch(RepoId, FileName, Revision, Token, ForceDownload, False, Result);
end;

function HubTryFetchFile(const RepoId, FileName: string;
  out LocalPath: string; const Revision: string;
  const Token: string; ForceDownload: boolean): boolean;
begin
  Result := DoFetch(RepoId, FileName, Revision, Token, ForceDownload, True,
    LocalPath);
end;

function HubFetchModel(const RepoId: string; const Revision: string;
  const Token: string): string;
var
  ConfigPath, WeightsPath, IndexPath: string;

  // Downloads every shard referenced by an already-fetched index json
  // (model.safetensors.index.json / pytorch_model.bin.index.json - the
  // importers' sharded readers take the index path directly).
  procedure FetchIndexShards(const pIndexPath: string);
  var
    IndexText: TStringList;
    Shards: TStringArray;
    I: integer;
  begin
    IndexText := TStringList.Create;
    try
      IndexText.LoadFromFile(pIndexPath);
      Shards := HubShardListFromIndexJson(IndexText.Text);
    finally
      IndexText.Free;
    end;
    for I := 0 to High(Shards) do
      HubFetchFile(RepoId, Shards[I], Revision, Token);
  end;

begin
  // config.json is mandatory: every importer needs it and every HF model
  // repo has one. A missing config is a hard error (bad repo id / gating).
  ConfigPath := HubFetchFile(RepoId, 'config.json', Revision, Token);
  Result := ExtractFileDir(ConfigPath);
  // tokenizer.json is optional (e.g. older SentencePiece-only repos).
  HubTryFetchFile(RepoId, 'tokenizer.json', WeightsPath, Revision, Token);
  // Weights, in preference order: single-file model.safetensors, the
  // sharded safetensors index + its shards, the torch.save
  // pytorch_model.bin, then the sharded .bin index + its shards.
  if HubTryFetchFile(RepoId, 'model.safetensors', WeightsPath, Revision,
    Token) then
    exit;
  if HubTryFetchFile(RepoId, 'model.safetensors.index.json', IndexPath,
    Revision, Token) then
  begin
    FetchIndexShards(IndexPath);
    exit;
  end;
  if HubTryFetchFile(RepoId, 'pytorch_model.bin', WeightsPath, Revision,
    Token) then
    exit;
  if HubTryFetchFile(RepoId, 'pytorch_model.bin.index.json', IndexPath,
    Revision, Token) then
  begin
    FetchIndexShards(IndexPath);
    exit;
  end;
  raise EHubError.Create('neuralhfhub: no weights found in "' + RepoId +
    '" - tried model.safetensors, model.safetensors.index.json, ' +
    'pytorch_model.bin and pytorch_model.bin.index.json.');
end;

end.
