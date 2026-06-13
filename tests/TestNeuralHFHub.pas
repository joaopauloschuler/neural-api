unit TestNeuralHFHub;
// Tests for the opt-in HuggingFace Hub download helper (neuralhfhub.pas).
//
// Everything except TestLiveFetchSmallRepo runs OFFLINE: URL building,
// cache-path layout, cache-dir override, index-json shard parsing and the
// skip-if-present cache-hit path (a planted file must be returned without
// any network activity).
//
// TestLiveFetchSmallRepo actually downloads prajjwal1/bert-tiny's
// config.json (~285 bytes) from huggingface.co into a temp cache. It is
// gated behind the NEURAL_HUB_LIVE_TEST=1 environment variable so the
// suite still passes on offline machines; set the variable to exercise the
// real network path.
//
// This unit is coded by Claude (AI).
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, neuralhfhub;

type
  TTestNeuralHFHub = class(TTestCase)
  private
    FTempDir: string;
    function MakeTempCacheDir: string;
    procedure RemoveTree(const Dir: string);
  protected
    procedure TearDown; override;
  published
    procedure TestResolveURL;
    procedure TestResolveURLRejectsTraversal;
    procedure TestLocalPathLayoutAndCacheOverride;
    procedure TestShardListFromIndexJson;
    procedure TestShardListRejectsMalformedIndex;
    procedure TestCacheHitSkipsDownload;
    procedure TestLiveFetchSmallRepo;
  end;

implementation

function TTestNeuralHFHub.MakeTempCacheDir: string;
begin
  Result := IncludeTrailingPathDelimiter(GetTempDir) + 'neuralhfhub_test_' +
    IntToStr(GetProcessID) + '_' + IntToStr(Random(1000000));
  if not ForceDirectories(Result) then
    Fail('Cannot create temp dir ' + Result);
  FTempDir := Result;
end;

procedure TTestNeuralHFHub.RemoveTree(const Dir: string);
var
  Rec: TSearchRec;
begin
  if (Dir = '') or not DirectoryExists(Dir) then Exit;
  if FindFirst(IncludeTrailingPathDelimiter(Dir) + '*', faAnyFile, Rec) = 0
  then
  begin
    repeat
      if (Rec.Name = '.') or (Rec.Name = '..') then Continue;
      if (Rec.Attr and faDirectory) <> 0 then
        RemoveTree(IncludeTrailingPathDelimiter(Dir) + Rec.Name)
      else
        DeleteFile(IncludeTrailingPathDelimiter(Dir) + Rec.Name);
    until FindNext(Rec) <> 0;
    FindClose(Rec);
  end;
  RemoveDir(Dir);
end;

procedure TTestNeuralHFHub.TearDown;
begin
  HubSetCacheDir(''); // restore the default for the next test
  RemoveTree(FTempDir);
  FTempDir := '';
  inherited TearDown;
end;

procedure TTestNeuralHFHub.TestResolveURL;
begin
  AssertEquals(
    'https://huggingface.co/prajjwal1/bert-tiny/resolve/main/config.json',
    HubResolveURL('prajjwal1/bert-tiny', 'config.json'));
  AssertEquals(
    'https://huggingface.co/roneneldan/TinyStories-1M/resolve/v1/' +
    'model.safetensors',
    HubResolveURL('roneneldan/TinyStories-1M', 'model.safetensors', 'v1'));
  // nested file names keep their forward slashes in the URL
  AssertEquals(
    'https://huggingface.co/a/b/resolve/main/onnx/model.onnx',
    HubResolveURL('a/b', 'onnx/model.onnx'));
end;

procedure TTestNeuralHFHub.TestResolveURLRejectsTraversal;
var
  Raised: boolean;
begin
  Raised := False;
  try
    HubResolveURL('prajjwal1/bert-tiny', '../../etc/passwd');
  except
    on EHubError do Raised := True;
  end;
  AssertTrue('".." in file name must raise EHubError', Raised);
  Raised := False;
  try
    HubResolveURL('', 'config.json');
  except
    on EHubError do Raised := True;
  end;
  AssertTrue('empty repo id must raise EHubError', Raised);
  Raised := False;
  try
    HubLocalPath('a/b', '/abs/path.json');
  except
    on EHubError do Raised := True;
  end;
  AssertTrue('absolute file name must raise EHubError', Raised);
end;

procedure TTestNeuralHFHub.TestLocalPathLayoutAndCacheOverride;
var
  CacheDir, Expected: string;
begin
  CacheDir := MakeTempCacheDir;
  HubSetCacheDir(CacheDir);
  AssertEquals(CacheDir, HubGetCacheDir);
  Expected := IncludeTrailingPathDelimiter(CacheDir) + 'prajjwal1' +
    DirectorySeparator + 'bert-tiny' + DirectorySeparator + 'main' +
    DirectorySeparator + 'config.json';
  AssertEquals(Expected,
    HubLocalPath('prajjwal1/bert-tiny', 'config.json'));
  // restoring the default must change the root back
  HubSetCacheDir('');
  AssertTrue('default cache dir must not be the override',
    HubGetCacheDir <> CacheDir);
end;

procedure TTestNeuralHFHub.TestShardListFromIndexJson;
var
  Shards: TStringArray;
begin
  Shards := HubShardListFromIndexJson(
    '{"metadata": {"total_size": 42}, "weight_map": {' +
    '"model.layers.0.w": "model-00001-of-00002.safetensors",' +
    '"model.layers.1.w": "model-00002-of-00002.safetensors",' +
    '"model.layers.2.w": "model-00001-of-00002.safetensors"}}');
  AssertEquals('duplicates must collapse', 2, Length(Shards));
  AssertEquals('model-00001-of-00002.safetensors', Shards[0]);
  AssertEquals('model-00002-of-00002.safetensors', Shards[1]);
end;

procedure TTestNeuralHFHub.TestShardListRejectsMalformedIndex;
var
  Raised: boolean;
begin
  Raised := False;
  try
    HubShardListFromIndexJson('{"metadata": {}}');
  except
    on EHubError do Raised := True;
  end;
  AssertTrue('missing weight_map must raise EHubError', Raised);
  Raised := False;
  try
    HubShardListFromIndexJson('not json at all');
  except
    on EHubError do Raised := True;
  end;
  AssertTrue('invalid JSON must raise EHubError', Raised);
  Raised := False;
  try
    HubShardListFromIndexJson('{"weight_map": {}}');
  except
    on EHubError do Raised := True;
  end;
  AssertTrue('empty weight_map must raise EHubError', Raised);
end;

procedure TTestNeuralHFHub.TestCacheHitSkipsDownload;
var
  CacheDir, Planted, Got: string;
  Sink: TStringList;
begin
  // Plant a file where HubLocalPath says it would live, then fetch from a
  // deliberately NONEXISTENT repo: skip-if-present must return the planted
  // file without ever opening a connection (a network attempt would fail).
  CacheDir := MakeTempCacheDir;
  HubSetCacheDir(CacheDir);
  Planted := HubLocalPath('no-such-user-xyz/no-such-repo', 'config.json');
  ForceDirectories(ExtractFileDir(Planted));
  Sink := TStringList.Create;
  try
    Sink.Text := '{"model_type": "planted"}';
    Sink.SaveToFile(Planted);
  finally
    Sink.Free;
  end;
  Got := HubFetchFile('no-such-user-xyz/no-such-repo', 'config.json');
  AssertEquals(Planted, Got);
  Sink := TStringList.Create;
  try
    Sink.LoadFromFile(Got);
    AssertEquals('{"model_type": "planted"}', Trim(Sink.Text));
  finally
    Sink.Free;
  end;
end;

procedure TTestNeuralHFHub.TestLiveFetchSmallRepo;
var
  CacheDir, P1, P2, OptPath: string;
  Sink: TStringList;
  Age1: longint;
begin
  if GetEnvironmentVariable('NEURAL_HUB_LIVE_TEST') <> '1' then
    Exit; // offline machines: pass without touching the network
  CacheDir := MakeTempCacheDir;
  HubSetCacheDir(CacheDir);
  // ~285-byte download from a tiny repo
  P1 := HubFetchFile('prajjwal1/bert-tiny', 'config.json');
  AssertTrue('downloaded file must exist', FileExists(P1));
  Sink := TStringList.Create;
  try
    Sink.LoadFromFile(P1);
    AssertTrue('config.json must contain vocab_size',
      Pos('vocab_size', Sink.Text) > 0);
  finally
    Sink.Free;
  end;
  // second fetch must be a cache hit (same path, file untouched)
  Age1 := FileAge(P1);
  P2 := HubFetchFile('prajjwal1/bert-tiny', 'config.json');
  AssertEquals(P1, P2);
  AssertEquals('cache hit must not rewrite the file', Age1, FileAge(P1));
  // optional-file path: this repo has no tokenizer.json -> false, no raise
  AssertFalse('404 on optional file must return false',
    HubTryFetchFile('prajjwal1/bert-tiny', 'tokenizer.json', OptPath));
end;

initialization
  RegisterTest(TTestNeuralHFHub);

end.
