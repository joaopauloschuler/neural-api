program ControlNetCanny;
(*
ControlNetCanny: an OFFLINE, CPU smoke for END-TO-END base-UNet-plus-ControlNet
single-step denoising -- the StableDiffusionControlNetPipeline inner loop wired
over the repo's already-landed importers:

  noisy latent + text states + a (canny-edge-style) control image
    -> ControlNet (BuildControlNetFromSafeTensors / ControlNetResiduals)
       produces down_block_res_samples + mid_block_res_sample
    -> base SD UNet built WITH control injection
       (BuildSDUNet(..., pWithControl=true))
    -> SDUNetDenoiseWithControl ADDS those residuals into the decoder skip
       connections, exactly as diffusers does:
         down_block_res_samples = [d + c for d,c in zip(down, controlnet_down)]
         mid_block_res_sample  += controlnet_mid
    -> predicted noise (the eps a sampler would step on).

Nothing here is a new leaf layer: it is pure plumbing over landed pieces (the
ControlNet importer + ControlNetResiduals, the SD UNet importer + the new
SDUNetDenoiseWithControl driver). It is a WIRING / THROUGHPUT smoke: it proves
the base UNet + ControlNet chain runs offline and produces a FINITE noise
prediction, NOT photorealism (the pico fixtures carry random weights).

The committed pico fixtures are config-compatible (both share block_out_channels
[16,32], latent grid 8, text seq 5, cross dim 12), so the ControlNet residuals
flow straight into the base UNet skips with no reshaping. The combined forward is
parity-checked < 1e-4 in TestControlNetCombinedParity
(tools/controlnet_combined_fixture.py).

SHAPES (pico fixtures, the default run):
  latent          (8, 8, in_channels=4)
  text states     (text_seq=5, 1, cross_dim=12)
  control image   (16, 16, cond_channels=3)
    -> 4 down residuals + 1 mid residual
    -> predicted noise (8, 8, out_channels=4)

USAGE:
  ./ControlNetCanny                              use the committed pico fixtures
  ./ControlNetCanny <unet.st> <unet.cfg> <controlnet.st> <controlnet.cfg>
                                                 real diffusers checkpoints
                                                 (config-compatible base UNet +
                                                 ControlNet)

NO NETWORK ACCESS / SELF-CONTAINED: falls back to the committed pico SD UNet
(tests/fixtures/tiny_sd_unet.* files) and the matched pico ControlNet
(tests/fixtures/tiny_controlnet.* files).
*)
{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralvolume, neuralnetwork, neuralpretrained;

const
  cUNetST   = '../../tests/fixtures/tiny_sd_unet.safetensors';
  cUNetCfg  = '../../tests/fixtures/tiny_sd_unet_config.json';
  cCtrlST   = '../../tests/fixtures/tiny_controlnet.safetensors';
  cCtrlCfg  = '../../tests/fixtures/tiny_controlnet_config.json';

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

// A toy "canny edge" control image: a bright grid lattice (edges every 4 px) on
// a dark field, replicated across the RGB channels. Just a viewable, finite map.
procedure FillCanny(V: TNNetVolume; Grid, Ch: integer);
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
  UNet, CNet: TNNet;
  UCfg: TSDUNetConfig;
  CCfg: TControlNetConfig;
  UNetST, UNetCfg, CtrlST, CtrlCfg: string;
  Latent, Text, Cond, MidRes, Noise: TNNetVolume;
  DownRes: array of TNNetVolume;
  DownCnt, i: integer;
  t, MinV, MaxV, MeanV: TNeuralFloat;
  Finite: boolean;
begin
  WriteLn('=== ControlNetCanny: base-UNet + ControlNet single-step smoke ===');

  if ParamCount >= 4 then
  begin
    UNetST := ParamStr(1); UNetCfg := ParamStr(2);
    CtrlST := ParamStr(3); CtrlCfg := ParamStr(4);
    WriteLn('Using user checkpoints:');
    WriteLn('  base UNet : ', UNetST);
    WriteLn('  controlnet: ', CtrlST);
  end
  else
  begin
    UNetST := cUNetST; UNetCfg := cUNetCfg;
    CtrlST := cCtrlST; CtrlCfg := cCtrlCfg;
    WriteLn('No checkpoints passed; using the committed pico fixtures.');
  end;

  // Base UNet built WITH control injection (extra zero-default skip inputs).
  UCfg := ReadSDUNetConfigFromJSONFile(UNetCfg);
  UNet := BuildSDUNetFromSafeTensorsEx(UNetST, UCfg,
    {pTrainable=}false, {pWithControl=}true);
  CNet := BuildControlNetFromSafeTensors(CtrlST, CCfg,
    {pTrainable=}false, CtrlCfg);
  WriteLn('Base UNet : ', SDUNetConfigToString(UCfg));
  WriteLn('ControlNet: ', ControlNetConfigToString(CCfg));

  DownCnt := CCfg.NumDownResiduals;
  if SDUNetControlDownCount(UCfg) <> DownCnt then
  begin
    WriteLn('FATAL: base UNet skip count (', SDUNetControlDownCount(UCfg),
      ') <> controlnet down residual count (', DownCnt,
      ') -- fixtures are not config-compatible.');
    Halt(1);
  end;

  Latent := TNNetVolume.Create;
  Text := TNNetVolume.Create;
  Cond := TNNetVolume.Create;
  MidRes := TNNetVolume.Create;
  Noise := TNNetVolume.Create;
  SetLength(DownRes, DownCnt);
  for i := 0 to DownCnt - 1 do DownRes[i] := TNNetVolume.Create;
  try
    FillLatent(Latent, UCfg.LatentGrid, UCfg.InChannels);
    FillText(Text, UCfg.TextSeqLen, UCfg.CrossAttentionDim);
    FillCanny(Cond, CCfg.CondGrid, CCfg.CondChannels);
    t := 23.0;

    // 1) ControlNet -> residuals.
    ControlNetResiduals(CNet, CCfg, Latent, Text, Cond, t, DownRes, MidRes);
    WriteLn('ControlNet produced ', DownCnt,
      ' down residuals + 1 mid residual:');
    for i := 0 to DownCnt - 1 do
      WriteLn(Format('  down[%d]: (%d,%d,%d)', [i, DownRes[i].Depth,
        DownRes[i].SizeY, DownRes[i].SizeX]));
    WriteLn(Format('  mid    : (%d,%d,%d)',
      [MidRes.Depth, MidRes.SizeY, MidRes.SizeX]));

    // 2) base UNet denoise WITH the residuals injected into the decoder skips.
    SDUNetDenoiseWithControl(UNet, UCfg, Latent, Text, t,
      DownRes, MidRes, Noise);

    VolumeStats(Noise, MinV, MaxV, MeanV, Finite);
    WriteLn(Format('Predicted noise (%d,%d,%d): min %.4f max %.4f mean %.4f',
      [Noise.Depth, Noise.SizeY, Noise.SizeX, MinV, MaxV, MeanV]));
    if Finite then
      WriteLn('OK: finite noise prediction (base UNet + ControlNet wired).')
    else
    begin
      WriteLn('FAIL: noise prediction contains NaN/Inf.');
      Halt(2);
    end;
  finally
    for i := 0 to DownCnt - 1 do DownRes[i].Free;
    Noise.Free; MidRes.Free; Cond.Free; Text.Free; Latent.Free;
    CNet.Free; UNet.Free;
  end;
end.
