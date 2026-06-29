program T2IAdapterSketch;
(*
T2IAdapterSketch: an OFFLINE, CPU smoke for END-TO-END base-UNet-plus-T2I-Adapter
single-step denoising -- the StableDiffusionAdapterPipeline inner loop wired over
the repo's already-landed importers. T2I-Adapter is the lighter sibling of
ControlNet (the natural successor): a small conv encoder over a spatial hint
(canny/sketch/depth) producing a PYRAMID of per-resolution feature maps that are
ADDED into the SD UNet down-block hidden state -- no transformer, no per-block
zero-conv, just a lightweight ResNet-ish ladder.

  spatial hint (sketch/canny-style image)
    -> T2I-Adapter (BuildT2IAdapterFromSafeTensors / T2IAdapterFeatures)
       produces a per-stage feature pyramid (one feature per UNet down block)
    -> base SD UNet built WITH adapter injection
       (BuildSDUNet(..., pWithAdapter=true))
    -> SDUNetDenoiseWithAdapter ADDS those features into `sample` at the end of
       each down block, exactly as diffusers does
       (down_intrablock_additional_residuals)
    -> predicted noise (the eps a sampler would step on).

Nothing here is a new leaf layer: pure plumbing over landed pieces. It is a
WIRING / THROUGHPUT smoke: it proves the base UNet + T2I-Adapter chain runs
offline and produces a FINITE noise prediction, NOT photorealism (the pico
fixtures carry random weights). The adapter feature pyramid itself is
parity-checked < 1e-4 in TestT2IAdapterParity (tools/t2i_adapter_tiny_fixture.py).

The committed pico fixtures are config-compatible: the adapter channels [16,32]
and grids (8x8, 4x4) match the pico SD UNet's two down-block stages, so the
adapter features add straight into the UNet hidden state with no reshaping.

SHAPES (pico fixtures, the default run):
  spatial hint    (16, 16, cond_channels=3)
    -> 2 adapter features: (16,8,8) + (32,4,4)
  latent          (8, 8, in_channels=4)
  text states     (text_seq=5, 1, cross_dim=12)
    -> predicted noise (8, 8, out_channels=4)

USAGE:
  ./T2IAdapterSketch                             use the committed pico fixtures
  ./T2IAdapterSketch <unet.st> <unet.cfg> <adapter.st> <adapter.cfg>
                                                 real diffusers checkpoints
                                                 (config-compatible base UNet +
                                                 T2I-Adapter)

NO NETWORK ACCESS / SELF-CONTAINED: falls back to the committed pico SD UNet
(tests/fixtures/tiny_sd_unet.* files) and the matched pico T2I-Adapter
(tests/fixtures/tiny_t2i_adapter.* files).
*)
{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralvolume, neuralnetwork, neuralpretrained;

const
  cUNetST   = '../../tests/fixtures/tiny_sd_unet.safetensors';
  cUNetCfg  = '../../tests/fixtures/tiny_sd_unet_config.json';
  cAdpST    = '../../tests/fixtures/tiny_t2i_adapter.safetensors';
  cAdpCfg   = '../../tests/fixtures/tiny_t2i_adapter_config.json';

// Deterministic, finite synthetic input fillers (no RNG -> reproducible smoke).
procedure FillLatent(V: TNNetVolume; Grid, Ch: integer);
var c, y, x: integer;
begin
  V.ReSize(Grid, Grid, Ch);
  for c := 0 to Ch - 1 do
    for y := 0 to Grid - 1 do
      for x := 0 to Grid - 1 do
        V.FData[(y * Grid + x) * Ch + c] :=
          (((c * 64 + y * Grid + x) * 7) mod 13 - 6) / 8.0;
end;

procedure FillText(V: TNNetVolume; SeqLen, Dim: integer);
var s, d: integer;
begin
  V.ReSize(SeqLen, 1, Dim);
  for s := 0 to SeqLen - 1 do
    for d := 0 to Dim - 1 do
      V.FData[s * Dim + d] := (((s * 16 + d) * 5) mod 11 - 5) / 8.0;
end;

// A toy "sketch" hint: a bright grid lattice (strokes every 4 px) on a dark
// field, replicated across the RGB channels. Just a viewable, finite map.
procedure FillSketch(V: TNNetVolume; Grid, Ch: integer);
var c, y, x: integer; edge: TNeuralFloat;
begin
  V.ReSize(Grid, Grid, Ch);
  for y := 0 to Grid - 1 do
    for x := 0 to Grid - 1 do
    begin
      if ((x mod 4) = 0) or ((y mod 4) = 0) then edge := 1.0 else edge := -1.0;
      for c := 0 to Ch - 1 do
        V.FData[(y * Grid + x) * Ch + c] := edge;
    end;
end;

procedure VolumeStats(V: TNNetVolume; out MinV, MaxV, MeanV: TNeuralFloat;
  out Finite: boolean);
var i: integer; s, val: TNeuralFloat;
begin
  MinV := 1e30; MaxV := -1e30; s := 0; Finite := true;
  for i := 0 to V.Size - 1 do
  begin
    val := V.FData[i];
    if IsNan(val) or IsInfinite(val) then Finite := false;
    if val < MinV then MinV := val;
    if val > MaxV then MaxV := val;
    s := s + val;
  end;
  if V.Size > 0 then MeanV := s / V.Size else MeanV := 0;
end;

var
  UNet, ANet: TNNet;
  UCfg: TSDUNetConfig;
  ACfg: TT2IAdapterConfig;
  UNetST, UNetCfg, AdpST, AdpCfg: string;
  Latent, Text, Cond, Noise: TNNetVolume;
  Feats: array of TNNetVolume;
  FeatCnt, i: integer;
  t, MinV, MaxV, MeanV: TNeuralFloat;
  Finite: boolean;
begin
  WriteLn('=== T2IAdapterSketch: base-UNet + T2I-Adapter single-step smoke ===');

  if ParamCount >= 4 then
  begin
    UNetST := ParamStr(1); UNetCfg := ParamStr(2);
    AdpST := ParamStr(3); AdpCfg := ParamStr(4);
    WriteLn('Using user checkpoints:');
    WriteLn('  base UNet   : ', UNetST);
    WriteLn('  t2i-adapter : ', AdpST);
  end
  else
  begin
    UNetST := cUNetST; UNetCfg := cUNetCfg;
    AdpST := cAdpST; AdpCfg := cAdpCfg;
    WriteLn('No checkpoints passed; using the committed pico fixtures.');
  end;

  // Base UNet built WITH adapter injection (extra zero-default main-path inputs).
  UCfg := ReadSDUNetConfigFromJSONFile(UNetCfg);
  UNet := BuildSDUNetFromSafeTensorsEx(UNetST, UCfg,
    {pTrainable=}false, {pWithControl=}false, {pWithAdapter=}true);
  ANet := BuildT2IAdapterFromSafeTensors(AdpST, ACfg,
    {pTrainable=}false, AdpCfg);
  WriteLn('Base UNet  : ', SDUNetConfigToString(UCfg));
  WriteLn('T2I-Adapter: ', T2IAdapterConfigToString(ACfg));

  FeatCnt := SDUNetAdapterFeatureCount(UCfg);
  if ACfg.NumChannels <> FeatCnt then
  begin
    WriteLn('FATAL: base UNet down-block count (', FeatCnt,
      ') <> adapter feature count (', ACfg.NumChannels,
      ') -- fixtures are not config-compatible.');
    Halt(1);
  end;

  Latent := TNNetVolume.Create;
  Text := TNNetVolume.Create;
  Cond := TNNetVolume.Create;
  Noise := TNNetVolume.Create;
  SetLength(Feats, FeatCnt);
  for i := 0 to FeatCnt - 1 do Feats[i] := TNNetVolume.Create;
  try
    FillLatent(Latent, UCfg.LatentGrid, UCfg.InChannels);
    FillText(Text, UCfg.TextSeqLen, UCfg.CrossAttentionDim);
    FillSketch(Cond, ACfg.CondGrid, ACfg.CondChannels);
    t := 23.0;

    // 1) T2I-Adapter -> per-stage feature pyramid.
    T2IAdapterFeatures(ANet, ACfg, Cond, Feats);
    WriteLn('T2I-Adapter produced ', FeatCnt, ' feature maps:');
    for i := 0 to FeatCnt - 1 do
      WriteLn(Format('  feature[%d]: (%d,%d,%d)', [i, Feats[i].Depth,
        Feats[i].SizeY, Feats[i].SizeX]));

    // 2) base UNet denoise WITH the features added into the down-block path.
    SDUNetDenoiseWithAdapter(UNet, UCfg, Latent, Text, t, Feats, Noise);

    VolumeStats(Noise, MinV, MaxV, MeanV, Finite);
    WriteLn(Format('Predicted noise (%d,%d,%d): min %.4f max %.4f mean %.4f',
      [Noise.Depth, Noise.SizeY, Noise.SizeX, MinV, MaxV, MeanV]));
    if Finite then
      WriteLn('OK: finite noise prediction (base UNet + T2I-Adapter wired).')
    else
    begin
      WriteLn('FAIL: noise prediction contains NaN/Inf.');
      Halt(2);
    end;
  finally
    for i := 0 to FeatCnt - 1 do Feats[i].Free;
    Noise.Free; Cond.Free; Text.Free; Latent.Free;
    ANet.Free; UNet.Free;
  end;
end.
