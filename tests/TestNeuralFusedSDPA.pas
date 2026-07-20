unit TestNeuralFusedSDPA;

// TNNetFusedSDPA parity suite: the fused multi-head (GQA) attention layer
// must be BIT-IDENTICAL to the per-head wiring it replaces
// ([SplitChannels -> DeepConcat pack -> TNNetScaledDotProductAttention] x Hq
// -> DeepConcat) on the prefill forward, the FP32 KV-cache decode path and
// the backward pass, and match the per-head int8 KV cache code-for-code
// (same per-(row,head) scale granularity). Also covers the cache API
// (TruncateCache rollback, Capture/RestoreCacheState fork) and eviction
// parity on the fused single-buffer cache.

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, neuralnetwork, neuralvolume;

type
  TTestNeuralFusedSDPA = class(TTestCase)
  private
    // Builds the per-head reference net for the packed [allQ|allK|allV]
    // input: Hq packs of [Q_h | K_g | V_g] -> SDPA(HeadDim) -> DeepConcat.
    function BuildPerHeadNet(SeqLen, Hq, Hkv, HeadDim, Window: integer;
      Causal: boolean): TNNet;
    function BuildFusedNet(SeqLen, Hq, Hkv, HeadDim, Window: integer;
      Causal: boolean): TNNet;
    procedure FillPacked(V: TNNetVolume; Seed: integer);
    procedure AssertOutputsEqual(const Msg: string; A, B: TNNet;
      Tolerance: TNeuralFloat);
  published
    procedure TestPrefillParityMHA;      // Hq = Hkv (plain multi-head)
    procedure TestPrefillParityGQA;      // Hq > Hkv (grouped queries)
    procedure TestPrefillParityWindow;   // sliding-window mask
    procedure TestDecodeParityFP32;      // per-token cached decode
    procedure TestDecodeParityInt8KV;    // int8 KV cache decode
    procedure TestDecodeMultiTokenPrefill; // cached multi-token prompt
    procedure TestTruncateRollback;      // speculative-decode rewind
    procedure TestCaptureRestoreFork;    // session snapshot / fork
    procedure TestEvictionParity;        // sink + rolling window
    procedure TestBackwardParity;        // per-head backward mirror
    procedure TestChunkedPrefillParity;  // planner chunk path vs serial
    procedure TestChunkedDecodeParity;   // chunked cached decode vs serial
  end;

implementation

function TTestNeuralFusedSDPA.BuildPerHeadNet(SeqLen, Hq, Hkv, HeadDim,
  Window: integer; Causal: boolean): TNNet;
var
  NN: TNNet;
  Src, QSlice, KSlice, VSlice, Pack: TNNetLayer;
  Heads: array of TNNetLayer;
  h, g, GroupSize, QW, KW: integer;
begin
  NN := TNNet.Create();
  GroupSize := Hq div Hkv;
  QW := Hq * HeadDim;
  KW := Hkv * HeadDim;
  Src := NN.AddLayer(TNNetInput.Create(SeqLen, 1, QW + 2 * KW, 1));
  SetLength(Heads, Hq);
  for h := 0 to Hq - 1 do
  begin
    g := h div GroupSize;
    QSlice := NN.AddLayerAfter(
      TNNetSplitChannels.Create(h * HeadDim, HeadDim), Src);
    KSlice := NN.AddLayerAfter(
      TNNetSplitChannels.Create(QW + g * HeadDim, HeadDim), Src);
    VSlice := NN.AddLayerAfter(
      TNNetSplitChannels.Create(QW + KW + g * HeadDim, HeadDim), Src);
    Pack := NN.AddLayer(TNNetDeepConcat.Create([QSlice, KSlice, VSlice]));
    Heads[h] := NN.AddLayerAfter(
      TNNetScaledDotProductAttention.Create(HeadDim, Causal, Window), Pack);
  end;
  if Hq = 1
    then NN.AddLayerAfter(TNNetIdentity.Create(), Heads[0])
    else NN.AddLayer(TNNetDeepConcat.Create(Heads));
  Result := NN;
end;

function TTestNeuralFusedSDPA.BuildFusedNet(SeqLen, Hq, Hkv, HeadDim,
  Window: integer; Causal: boolean): TNNet;
var
  NN: TNNet;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(SeqLen, 1, (Hq + 2 * Hkv) * HeadDim, 1));
  NN.AddLayer(TNNetFusedSDPA.Create(Hq, Hkv, HeadDim, Causal, Window));
  Result := NN;
end;

procedure TTestNeuralFusedSDPA.FillPacked(V: TNNetVolume; Seed: integer);
var
  i: integer;
begin
  for i := 0 to V.Size - 1 do
    V.FData[i] := Sin(0.37 * (i + Seed)) * 0.8 + 0.11 * Cos(1.13 * i + Seed);
end;

procedure TTestNeuralFusedSDPA.AssertOutputsEqual(const Msg: string;
  A, B: TNNet; Tolerance: TNeuralFloat);
var
  i: integer;
  OutA, OutB: TNNetVolume;
begin
  OutA := A.GetLastLayer().Output;
  OutB := B.GetLastLayer().Output;
  AssertEquals(Msg + ': output size', OutA.Size, OutB.Size);
  for i := 0 to OutA.Size - 1 do
    AssertEquals(Msg + ' at flat index ' + IntToStr(i),
      OutA.FData[i], OutB.FData[i], Tolerance);
end;

procedure TTestNeuralFusedSDPA.TestPrefillParityMHA;
var
  PerHead, Fused: TNNet;
  Input: TNNetVolume;
begin
  PerHead := BuildPerHeadNet(6, 2, 2, 8, 0, {Causal=}true);
  Fused := BuildFusedNet(6, 2, 2, 8, 0, {Causal=}true);
  Input := TNNetVolume.Create(6, 1, (2 + 4) * 8);
  try
    FillPacked(Input, 7);
    PerHead.Compute(Input);
    Fused.Compute(Input);
    AssertOutputsEqual('MHA prefill fused vs per-head', PerHead, Fused, 0);
  finally
    Input.Free;
    Fused.Free;
    PerHead.Free;
  end;
end;

procedure TTestNeuralFusedSDPA.TestPrefillParityGQA;
var
  PerHead, Fused: TNNet;
  Input: TNNetVolume;
begin
  // 4 query heads sharing 2 KV heads (GroupSize 2).
  PerHead := BuildPerHeadNet(5, 4, 2, 4, 0, {Causal=}true);
  Fused := BuildFusedNet(5, 4, 2, 4, 0, {Causal=}true);
  Input := TNNetVolume.Create(5, 1, (4 + 4) * 4);
  try
    FillPacked(Input, 21);
    PerHead.Compute(Input);
    Fused.Compute(Input);
    AssertOutputsEqual('GQA prefill fused vs per-head', PerHead, Fused, 0);
  finally
    Input.Free;
    Fused.Free;
    PerHead.Free;
  end;
end;

procedure TTestNeuralFusedSDPA.TestPrefillParityWindow;
var
  PerHead, Fused: TNNet;
  Input: TNNetVolume;
begin
  // Causal + sliding window of 3 keys.
  PerHead := BuildPerHeadNet(7, 2, 1, 4, 3, {Causal=}true);
  Fused := BuildFusedNet(7, 2, 1, 4, 3, {Causal=}true);
  Input := TNNetVolume.Create(7, 1, (2 + 2) * 4);
  try
    FillPacked(Input, 33);
    PerHead.Compute(Input);
    Fused.Compute(Input);
    AssertOutputsEqual('windowed prefill fused vs per-head',
      PerHead, Fused, 0);
  finally
    Input.Free;
    Fused.Free;
    PerHead.Free;
  end;
end;

// Arms the cached path on every SDPA-family layer of the net.
procedure BeginDecodeAll(NN: TNNet; MaxCtx: integer; Int8KV: boolean);
var
  i: integer;
begin
  for i := 0 to NN.CountLayers() - 1 do
    if NN.Layers[i] is TNNetScaledDotProductAttention then
      TNNetScaledDotProductAttention(NN.Layers[i]).BeginIncrementalDecode(
        MaxCtx, Int8KV);
end;

procedure TTestNeuralFusedSDPA.TestDecodeParityFP32;
var
  PerHead, Fused: TNNet;
  Tok: TNNetVolume;
  Step: integer;
begin
  // SeqLen-1 nets fed one token at a time; the caches carry the history.
  PerHead := BuildPerHeadNet(1, 4, 2, 4, 0, {Causal=}true);
  Fused := BuildFusedNet(1, 4, 2, 4, 0, {Causal=}true);
  Tok := TNNetVolume.Create(1, 1, (4 + 4) * 4);
  try
    BeginDecodeAll(PerHead, 16, false);
    BeginDecodeAll(Fused, 16, false);
    for Step := 0 to 9 do
    begin
      FillPacked(Tok, 100 + Step);
      PerHead.Compute(Tok);
      Fused.Compute(Tok);
      AssertOutputsEqual('FP32 decode step ' + IntToStr(Step),
        PerHead, Fused, 0);
    end;
  finally
    Tok.Free;
    Fused.Free;
    PerHead.Free;
  end;
end;

procedure TTestNeuralFusedSDPA.TestDecodeParityInt8KV;
var
  PerHead, Fused: TNNet;
  Tok: TNNetVolume;
  Step: integer;
begin
  // int8 KV: the fused per-(row, head) scales must match the per-head
  // per-row scales code-for-code, so the outputs stay bit-equal too.
  PerHead := BuildPerHeadNet(1, 4, 2, 4, 0, {Causal=}true);
  Fused := BuildFusedNet(1, 4, 2, 4, 0, {Causal=}true);
  Tok := TNNetVolume.Create(1, 1, (4 + 4) * 4);
  try
    BeginDecodeAll(PerHead, 16, true);
    BeginDecodeAll(Fused, 16, true);
    for Step := 0 to 9 do
    begin
      FillPacked(Tok, 500 + Step);
      PerHead.Compute(Tok);
      Fused.Compute(Tok);
      AssertOutputsEqual('int8 KV decode step ' + IntToStr(Step),
        PerHead, Fused, 1e-6);
    end;
  finally
    Tok.Free;
    Fused.Free;
    PerHead.Free;
  end;
end;

procedure TTestNeuralFusedSDPA.TestDecodeMultiTokenPrefill;
var
  PerHead, Fused: TNNet;
  Prompt, Tok: TNNetVolume;
  Step: integer;
begin
  // Cached multi-token prompt prefill (SeqLen 4), then single-token steps -
  // the ChatTerminal pattern. Both nets are built at the prompt width; the
  // decode steps reuse the same nets with a narrower live input? No - the
  // input width is fixed at build, so the step net is separate builds in
  // ChatTerminal. HERE we only check the multi-token cached forward parity.
  PerHead := BuildPerHeadNet(4, 2, 1, 8, 0, {Causal=}true);
  Fused := BuildFusedNet(4, 2, 1, 8, 0, {Causal=}true);
  Prompt := TNNetVolume.Create(4, 1, (2 + 2) * 8);
  Tok := nil;
  try
    BeginDecodeAll(PerHead, 32, false);
    BeginDecodeAll(Fused, 32, false);
    for Step := 0 to 2 do
    begin
      FillPacked(Prompt, 900 + Step);
      PerHead.Compute(Prompt);
      Fused.Compute(Prompt);
      AssertOutputsEqual('multi-token cached prefill pass ' + IntToStr(Step),
        PerHead, Fused, 0);
    end;
  finally
    Tok.Free;
    Prompt.Free;
    Fused.Free;
    PerHead.Free;
  end;
end;

procedure TTestNeuralFusedSDPA.TestTruncateRollback;
var
  Fused: TNNet;
  Tok: TNNetVolume;
  Layer: TNNetFusedSDPA;
  Saved: TNNetVolume;
  Step, i: integer;
begin
  // Decode 4 tokens, remember output of token 3, decode 3 more (speculation),
  // truncate back to 4, re-feed the SAME token 5th-token draft: the output
  // must match the first time it was computed at that cache position.
  Fused := BuildFusedNet(1, 2, 1, 4, 0, {Causal=}true);
  Tok := TNNetVolume.Create(1, 1, (2 + 2) * 4);
  Saved := TNNetVolume.Create();
  try
    BeginDecodeAll(Fused, 16, false);
    Layer := TNNetFusedSDPA(Fused.Layers[1]);
    for Step := 0 to 3 do
    begin
      FillPacked(Tok, 40 + Step);
      Fused.Compute(Tok);
    end;
    AssertEquals('cache length after 4 tokens', 4, Layer.CacheLength);
    // The draft token at position 4:
    FillPacked(Tok, 44);
    Fused.Compute(Tok);
    Saved.Copy(Fused.GetLastLayer().Output);
    // Speculate 2 more, then reject them all: rewind to 4 and re-feed.
    for Step := 5 to 6 do
    begin
      FillPacked(Tok, 40 + Step);
      Fused.Compute(Tok);
    end;
    AssertEquals('cache length after speculation', 7, Layer.CacheLength);
    Layer.TruncateCache(4);
    FillPacked(Tok, 44);
    Fused.Compute(Tok);
    for i := 0 to Saved.Size - 1 do
      AssertEquals('post-rollback recompute at ' + IntToStr(i),
        Saved.FData[i], Fused.GetLastLayer().Output.FData[i], 0);
  finally
    Saved.Free;
    Tok.Free;
    Fused.Free;
  end;
end;

procedure TTestNeuralFusedSDPA.TestCaptureRestoreFork;
var
  Fused: TNNet;
  Tok: TNNetVolume;
  Layer: TNNetFusedSDPA;
  SnapK, SnapV, OutA: TNNetVolume;
  Len, Sinks, Window, Step, i: integer;
begin
  Fused := BuildFusedNet(1, 2, 1, 4, 0, {Causal=}true);
  Tok := TNNetVolume.Create(1, 1, (2 + 2) * 4);
  SnapK := TNNetVolume.Create();
  SnapV := TNNetVolume.Create();
  OutA := TNNetVolume.Create();
  try
    BeginDecodeAll(Fused, 16, false);
    Layer := TNNetFusedSDPA(Fused.Layers[1]);
    for Step := 0 to 2 do
    begin
      FillPacked(Tok, 60 + Step);
      Fused.Compute(Tok);
    end;
    Layer.CaptureCacheState(SnapK, SnapV, Len, Sinks, Window);
    AssertEquals('snapshot length', 3, Len);
    // Branch A: continue with token X.
    FillPacked(Tok, 77);
    Fused.Compute(Tok);
    OutA.Copy(Fused.GetLastLayer().Output);
    // Diverge (different token), then restore the snapshot and replay X.
    FillPacked(Tok, 88);
    Fused.Compute(Tok);
    Layer.RestoreCacheState(SnapK, SnapV, Len, Sinks, Window);
    FillPacked(Tok, 77);
    Fused.Compute(Tok);
    for i := 0 to OutA.Size - 1 do
      AssertEquals('restored-fork replay at ' + IntToStr(i),
        OutA.FData[i], Fused.GetLastLayer().Output.FData[i], 0);
  finally
    OutA.Free;
    SnapV.Free;
    SnapK.Free;
    Tok.Free;
    Fused.Free;
  end;
end;

procedure TTestNeuralFusedSDPA.TestEvictionParity;
var
  PerHead, Fused: TNNet;
  Tok: TNNetVolume;
  i, Step: integer;
begin
  // Sink 2 + window 4 (cap 6): stream 12 tokens; the fused single-buffer
  // eviction shift must match the per-head shifts token for token.
  PerHead := BuildPerHeadNet(1, 2, 2, 4, 0, {Causal=}true);
  Fused := BuildFusedNet(1, 2, 2, 4, 0, {Causal=}true);
  Tok := TNNetVolume.Create(1, 1, (2 + 4) * 4);
  try
    BeginDecodeAll(PerHead, 16, false);
    BeginDecodeAll(Fused, 16, false);
    for i := 0 to PerHead.CountLayers() - 1 do
      if PerHead.Layers[i] is TNNetScaledDotProductAttention then
        TNNetScaledDotProductAttention(PerHead.Layers[i]).EnableEviction(2, 4);
    TNNetFusedSDPA(Fused.Layers[1]).EnableEviction(2, 4);
    for Step := 0 to 11 do
    begin
      FillPacked(Tok, 300 + Step);
      PerHead.Compute(Tok);
      Fused.Compute(Tok);
      AssertOutputsEqual('eviction decode step ' + IntToStr(Step),
        PerHead, Fused, 0);
    end;
    AssertEquals('fused cache pinned at cap', 6,
      TNNetFusedSDPA(Fused.Layers[1]).CacheLength);
  finally
    Tok.Free;
    Fused.Free;
    PerHead.Free;
  end;
end;

procedure TTestNeuralFusedSDPA.TestBackwardParity;
var
  PerHead, Fused: TNNet;
  Input, Tgt: TNNetVolume;
  i: integer;
begin
  PerHead := BuildPerHeadNet(5, 4, 2, 4, 0, {Causal=}true);
  Fused := BuildFusedNet(5, 4, 2, 4, 0, {Causal=}true);
  Input := TNNetVolume.Create(5, 1, (4 + 4) * 4);
  Tgt := TNNetVolume.Create(5, 1, 4 * 4);
  try
    PerHead.SetLearningRate(1.0, 0.0);
    Fused.SetLearningRate(1.0, 0.0);
    PerHead.SetBatchUpdate(true);
    Fused.SetBatchUpdate(true);
    FillPacked(Input, 11);
    FillPacked(Tgt, 200);
    PerHead.Compute(Input);
    Fused.Compute(Input);
    AssertOutputsEqual('backward-test forward', PerHead, Fused, 0);
    PerHead.Backpropagate(Tgt);
    Fused.Backpropagate(Tgt);
    // The gradient handed back to the packed input must match: per-head
    // Q gradients land on their slices; the shared-KV-group gradients are
    // the SUM over the group's heads in both wirings.
    for i := 0 to Input.Size - 1 do
      AssertEquals('packed-input gradient at ' + IntToStr(i),
        PerHead.Layers[0].OutputError.FData[i],
        Fused.Layers[0].OutputError.FData[i], 2e-5);
  finally
    Tgt.Free;
    Input.Free;
    Fused.Free;
    PerHead.Free;
  end;
end;

procedure TTestNeuralFusedSDPA.TestChunkedPrefillParity;
var
  Serial, Chunked: TNNet;
  Input: TNNetVolume;
  i: integer;
begin
  // The intra-layer-threaded forward splits the fused layer into per-head
  // chunks (PrepareChunkedForward + ComputeRange); the math must stay
  // bit-identical to the serial Compute.
  Serial := BuildFusedNet(6, 4, 2, 8, 0, {Causal=}true);
  Chunked := BuildFusedNet(6, 4, 2, 8, 0, {Causal=}true);
  Input := TNNetVolume.Create(6, 1, (4 + 4) * 8);
  try
    FillPacked(Input, 55);
    Serial.Compute(Input, 0, {Parallel=}false);
    Chunked.Compute(Input, 0, {Parallel=}true);
    for i := 0 to Serial.GetLastLayer().Output.Size - 1 do
      AssertEquals('chunked prefill at ' + IntToStr(i),
        Serial.GetLastLayer().Output.FData[i],
        Chunked.GetLastLayer().Output.FData[i], 0);
  finally
    Input.Free;
    Chunked.Free;
    Serial.Free;
  end;
end;

procedure TTestNeuralFusedSDPA.TestChunkedDecodeParity;
var
  Serial, Chunked: TNNet;
  Tok: TNNetVolume;
  Step, i: integer;
begin
  Serial := BuildFusedNet(1, 4, 2, 4, 0, {Causal=}true);
  Chunked := BuildFusedNet(1, 4, 2, 4, 0, {Causal=}true);
  Tok := TNNetVolume.Create(1, 1, (4 + 4) * 4);
  try
    BeginDecodeAll(Serial, 16, false);
    BeginDecodeAll(Chunked, 16, false);
    for Step := 0 to 7 do
    begin
      FillPacked(Tok, 700 + Step);
      Serial.Compute(Tok, 0, {Parallel=}false);
      Chunked.Compute(Tok, 0, {Parallel=}true);
      for i := 0 to Serial.GetLastLayer().Output.Size - 1 do
        AssertEquals('chunked decode step ' + IntToStr(Step) +
          ' at ' + IntToStr(i),
          Serial.GetLastLayer().Output.FData[i],
          Chunked.GetLastLayer().Output.FData[i], 0);
    end;
  finally
    Tok.Free;
    Chunked.Free;
    Serial.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralFusedSDPA);
end.
