program VARGenerateDemo;
(*
VARGenerate: the OFFLINE, CPU coarse-to-fine VAR (Visual AutoRegressive,
NEXT-SCALE prediction; Tian et al. 2024, arXiv:2404.02905) image-generation
loop, end to end, over the repo's already-landed importers:

  class label y
    -> class-conditional VAR transformer (BuildVARFromSafeTensors)
    -> next-scale autoregressive SAMPLING loop (VARGenerate): for each pyramid
       level s = 0..K-1, run the forward over the partially-filled multi-scale
       token sequence, read the next-scale logits at scale s's positions,
       sample/argmax the VocabSize-way tokens, write them back so the FINER
       scales attend to them through the scale-block-causal mask
    -> the final scale's PatchNums[K-1] x PatchNums[K-1] token grid is a VQ
       token map
    -> residual/discrete VQ decode to pixels (BuildVqModelFromSafeTensors ->
       DecodeVARTokensToImage)
    -> RGB image -> P6 PPM.

Nothing here is a new leaf layer: it is pure plumbing over landed pieces (the
VAR importer + the new VARGenerate / DecodeVARTokensToImage helpers, and the
VQModel discrete-tokenizer importer).

FAITHFULNESS NOTE: canonical FoundationVision/var carries the cross-scale
residual-VQ feature accumulation (next-scale interpolation/up-sampling of the
running f_hat) INSIDE the VQ tokenizer that produces the input embeddings; this
importer's input contract is plain codebook INDICES embedded by word_embed, so
the coarse->fine information flow here is carried purely by the transformer
attention over the already-sampled coarser tokens. The committed pico fixtures
have RANDOM weights, so this is a WIRING / THROUGHPUT smoke -- it proves the
multi-scale loop + VQ decode run offline and produce a FINITE image, not a real
picture. The text-conditioned Infinity variant and real-checkpoint parity are
tracked follow-ups.

SHAPES (pico fixtures, the default run):
  VAR  : hidden 16, depth 2, vocab 12, 5 classes, pyramid [1,2,3] -> 14 tokens
  VQ   : latent grid 3 (= VAR final patch_num), codebook 12, image 6x6x3

NO NETWORK ACCESS / SELF-CONTAINED: falls back to the committed pico VAR
(tests/fixtures/tiny_var.*, tools/make_pico_var_fixture.py, parity-checked
< 1e-4 in TestVARParity) and the MATCHED pico VQModel
(tests/fixtures/tiny_var_vqmodel.*, tools/make_pico_var_vqmodel_fixture.py),
sized so the VAR final 3x3 token map is exactly the VQ token grid. Regression-
tested by TestVARGenerateSmoke (asserts the loop produces the expected token-
sequence shape deterministically and a finite image).

USAGE
  ./VARGenerate                        use the committed pico fixtures
  ./VARGenerate var.safetensors vq.safetensors
                                       use your own checkpoints (sibling
                                       config.json files)
Optional trailing flags:
  --class N      class label to generate (default 0)
  --temp T       sampling temperature (default 0 = greedy argmax)
  --seed N       RNG seed (default 424242; only matters when --temp > 0)
  --smoke        run, assert finiteness, print OK/FAIL and exit (no PPM)

OUTPUT
  Writes var_generate.ppm (the decoded RGB image) and prints the configs, the
  sampled per-scale token maps and an ASCII preview. With --smoke it prints
  "SMOKE OK" / "SMOKE FAIL" and sets the exit code (the build step exercises it).

Coded by Claude (AI).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)
{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math,
  neuralnetwork, neuralvolume, neuralpretrained;

const
  cFixtureVarST  = '../../tests/fixtures/tiny_var.safetensors';
  cFixtureVarCfg = '../../tests/fixtures/tiny_var_config.json';
  cFixtureVqST   = '../../tests/fixtures/tiny_var_vqmodel.safetensors';
  cFixtureVqCfg  = '../../tests/fixtures/tiny_var_vqmodel_config.json';
  cOutFile       = 'var_generate.ppm';

function ToByte(V: TNeuralFloat): byte;
begin
  V := (V + 1) * 0.5;
  if V < 0 then V := 0;
  if V > 1 then V := 1;
  Result := Round(V * 255);
end;

procedure WriteColorPPM(Img: TNNetVolume; const FileName: string);
var
  F: TextFile;
  Bin: TFileStream;
  X, Y, W, H: integer;
  R, G, B: byte;
begin
  W := Img.SizeX; H := Img.SizeY;
  AssignFile(F, FileName); Rewrite(F);
  WriteLn(F, 'P6'); WriteLn(F, W, ' ', H); WriteLn(F, 255);
  CloseFile(F);
  Bin := TFileStream.Create(FileName, fmOpenReadWrite);
  try
    Bin.Seek(0, soEnd);
    for Y := 0 to H - 1 do
      for X := 0 to W - 1 do
      begin
        R := ToByte(Img[X, Y, 0]);
        if Img.Depth > 1 then G := ToByte(Img[X, Y, 1]) else G := R;
        if Img.Depth > 2 then B := ToByte(Img[X, Y, 2]) else B := R;
        Bin.WriteByte(R); Bin.WriteByte(G); Bin.WriteByte(B);
      end;
  finally
    Bin.Free;
  end;
end;

procedure AsciiPreview(Img: TNNetVolume);
const cRamp = ' .:-=+*#%@';
var X, Y, idx: integer; v: TNeuralFloat;
begin
  for Y := 0 to Img.SizeY - 1 do
  begin
    for X := 0 to Img.SizeX - 1 do
    begin
      v := (Img[X, Y, 0] + Img[X, Y, 1] + Img[X, Y, 2]) / 3;
      v := (v + 1) * 0.5; if v < 0 then v := 0; if v > 1 then v := 1;
      idx := Round(v * (Length(cRamp) - 1)) + 1;
      Write(cRamp[idx]);
    end;
    WriteLn;
  end;
end;

function AllFinite(V: TNNetVolume): boolean;
var i: integer;
begin
  Result := true;
  for i := 0 to V.Size - 1 do
    if (V.FData[i] <> V.FData[i]) or (Abs(V.FData[i]) > 1e30) then
    begin
      Result := false;
      Exit;
    end;
end;

procedure PrintScaleMaps(const Config: TVARConfig;
  const Tokens: TNeuralIntegerArray);
var s, p, y, x, Start, ScalesM1: integer;
begin
  ScalesM1 := Config.NumScales - 1;
  for s := 0 to ScalesM1 do
  begin
    p := Config.PatchNums[s];
    Start := VARScaleStart(Config, s);
    WriteLn('  scale ', s, ' (', p, 'x', p, '):');
    for y := 0 to p - 1 do
    begin
      Write('    ');
      for x := 0 to p - 1 do
        Write(Tokens[Start + y * p + x]:4);
      WriteLn;
    end;
  end;
end;

var
  VarNet: TNNet;
  VarCfg: TVARConfig;
  VqModel: TNNetVqModel;
  VqCfg: TVqModelConfig;
  Tokens: TNeuralIntegerArray;
  Image: TNNetVolume;
  VarST, VarCfgPath, VqST, VqCfgPath, Arg: string;
  ClassId, SeedVal, i: integer;
  Temperature: TNeuralFloat;
  SmokeMode, ImageOk: boolean;
begin
  VarST := cFixtureVarST; VarCfgPath := cFixtureVarCfg;
  VqST := cFixtureVqST;   VqCfgPath := cFixtureVqCfg;
  ClassId := 0;
  Temperature := 0.0;
  SeedVal := 424242;
  SmokeMode := false;
  i := 1;
  if (ParamCount >= 1) and (Copy(ParamStr(1), 1, 2) <> '--') then
  begin
    VarST := ParamStr(1);
    VarCfgPath := ExtractFilePath(VarST) + 'config.json';
    i := 2;
    if (ParamCount >= 2) and (Copy(ParamStr(2), 1, 2) <> '--') then
    begin
      VqST := ParamStr(2);
      VqCfgPath := ExtractFilePath(VqST) + 'config.json';
      i := 3;
    end;
    WriteLn('Loading VAR + VQ from checkpoints: ', VarST, ' / ', VqST);
  end
  else
    WriteLn('No checkpoints passed; using the committed pico fixtures ' +
      '(random weights -- wiring SMOKE demo).');
  while i <= ParamCount do
  begin
    Arg := ParamStr(i);
    if Arg = '--class' then begin Inc(i); ClassId := StrToIntDef(ParamStr(i), ClassId); end
    else if Arg = '--temp' then begin Inc(i); Temperature := StrToFloatDef(ParamStr(i), Temperature); end
    else if Arg = '--seed' then begin Inc(i); SeedVal := StrToIntDef(ParamStr(i), SeedVal); end
    else if Arg = '--smoke' then SmokeMode := true
    else WriteLn('Ignoring unknown argument: ', Arg);
    Inc(i);
  end;

  RandSeed := SeedVal;
  VarNet := BuildVARFromSafeTensors(VarST, VarCfg, {pTrainable=}false, VarCfgPath);
  VqModel := BuildVqModelFromSafeTensors(VqST, VqCfg, {pTrainable=}false, VqCfgPath);
  Image := TNNetVolume.Create;
  try
    WriteLn(VARConfigToString(VarCfg));
    WriteLn(VqModelConfigToString(VqCfg));
    if VqCfg.LatentGrid <> VarCfg.PatchNums[VarCfg.NumScales - 1] then
      raise Exception.Create('VQ latent grid must equal VAR final patch_num.');
    if VqCfg.NumVqEmbeddings < VarCfg.VocabSize then
      raise Exception.Create('VQ codebook must be >= VAR vocab.');
    WriteLn('class=', ClassId, '  temperature=', Temperature:0:3,
      '  seed=', SeedVal);

    // The coarse-to-fine next-scale sampling loop.
    VARGenerate(VarNet, VarCfg, ClassId, Tokens, Temperature);
    WriteLn('Sampled ', VarCfg.SeqLen, ' multi-scale tokens:');
    PrintScaleMaps(VarCfg, Tokens);

    // Residual/discrete VQ decode of the final-scale token grid -> RGB.
    DecodeVARTokensToImage(VqModel, VarCfg, Tokens, Image);
    ImageOk := AllFinite(Image);
    WriteLn('Decoded image ', Image.SizeX, 'x', Image.SizeY, 'x', Image.Depth,
      '  finite=', BoolToStr(ImageOk, true));

    if SmokeMode then
    begin
      if (Length(Tokens) = VarCfg.SeqLen) and ImageOk then
      begin
        WriteLn('SMOKE OK');
        ExitCode := 0;
      end
      else
      begin
        WriteLn('SMOKE FAIL');
        ExitCode := 1;
      end;
    end
    else
    begin
      WriteColorPPM(Image, cOutFile);
      WriteLn('Wrote ', cOutFile, '.');
      WriteLn; WriteLn('decoded image:'); AsciiPreview(Image);
    end;
  finally
    Image.Free;
    VqModel.Free;
    VarNet.Free;
  end;
end.
