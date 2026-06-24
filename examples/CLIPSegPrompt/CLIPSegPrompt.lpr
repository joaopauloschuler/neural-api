program CLIPSegPrompt;
(*
CLIPSegPrompt: text-prompted ZERO-SHOT segmentation on the CPU end-to-end with
the repo's CLIPSeg importer (BuildCLIPSegFromSafeTensors, neuralpretrained.pas)
-- a "free-text prompt -> dense single-channel mask" import.
CLIPSeg (Lueddecke & Ecker 2022, "Image Segmentation Using Text and Image
Prompts", arXiv:2112.10003; CIDAS/clipseg-rd64-refined) runs an image through a
FROZEN CLIP ViT, taps a few intermediate encoder layers (config.extract_layers),
FiLM-modulates them with the CLIP TEXT embedding of an ARBITRARY prompt, refines
through a small post-norm transformer decoder and upsamples (a single
ConvTranspose2d) to an HxW logit map: the mask for "whatever the prompt names",
with NO fixed label set.

Heavy REUSE: the frozen CLIP vision tower and text tower are the same pre-LN CLIP
encoder blocks as BuildClipFromSafeTensors; the new code is only the
FiLM-conditioned decoder (TNNetFiLM + a post-norm CLIP-style block + a
DepthToSpace transposed-conv upsample).

NO NETWORK ACCESS / SELF-CONTAINED: the real CIDAS/clipseg-rd64-refined checkpoint
is large and not obtainable offline, so -- exactly like the repo's NAFNet/SwinIR
pico fixtures -- this falls back to the committed CONFIG-FAITHFUL random pico
CLIPSeg (tests/fixtures/tiny_clipseg.*, built by tools/clipseg_tiny_fixture.py
from the REAL HF CLIPSegForImageSegmentation and parity-checked < 1e-4 in
tests/TestNeuralPretrained.pas TestCLIPSegParity). The pico net is random (not
trained), so this is a wiring/throughput SMOKE demo: it builds the three nets,
runs the full image+prompt pipeline (RunCLIPSeg), thresholds the logit map at 0
and writes a binary-mask PPM. Pass a real .safetensors (+ config.json sibling)
and real prompt token ids to segment with your own trained checkpoint.

The prompt is supplied as TOKEN IDS (the pico fixture has no tokenizer); for a
real checkpoint feed the ids your CLIP tokenizer produces for the prompt text.

USAGE
  ./CLIPSegPrompt                 use the committed pico fixture
  ./CLIPSegPrompt model.safetensors [config.json]
                                  use a real checkpoint

OUTPUT
  Writes clipseg_image.ppm (the synthetic input) and clipseg_mask.ppm (the
  thresholded binary mask) and prints the config plus an ASCII preview of the
  image and the mask.

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
  cFixtureST  = '../../tests/fixtures/tiny_clipseg.safetensors';
  cFixtureCfg = '../../tests/fixtures/tiny_clipseg_config.json';
  cImageFile  = 'clipseg_image.ppm';
  cMaskFile   = 'clipseg_mask.ppm';

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

// Binary mask PPM: white where logit > 0, black elsewhere.
procedure WriteMaskPPM(Mask: TNNetVolume; const FileName: string);
var
  F: TextFile;
  Bin: TFileStream;
  X, Y: integer;
  V: byte;
begin
  AssignFile(F, FileName); Rewrite(F);
  WriteLn(F, 'P6'); WriteLn(F, Mask.SizeX, ' ', Mask.SizeY); WriteLn(F, 255);
  CloseFile(F);
  Bin := TFileStream.Create(FileName, fmOpenReadWrite);
  try
    Bin.Seek(0, soEnd);
    for Y := 0 to Mask.SizeY - 1 do
      for X := 0 to Mask.SizeX - 1 do
      begin
        if Mask[X, Y, 0] > 0 then V := 255 else V := 0;
        Bin.WriteByte(V); Bin.WriteByte(V); Bin.WriteByte(V);
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
      v := Img[X, Y, 0];
      if Img.Depth > 2 then v := (Img[X, Y, 0] + Img[X, Y, 1] + Img[X, Y, 2]) / 3;
      v := (v + 1) * 0.5; if v < 0 then v := 0; if v > 1 then v := 1;
      idx := Round(v * (Length(cRamp) - 1)) + 1;
      Write(cRamp[idx]);
    end;
    WriteLn;
  end;
end;

procedure AsciiMask(Mask: TNNetVolume);
var X, Y: integer;
begin
  for Y := 0 to Mask.SizeY - 1 do
  begin
    for X := 0 to Mask.SizeX - 1 do
      if Mask[X, Y, 0] > 0 then Write('#') else Write('.');
    WriteLn;
  end;
end;

// Deterministic synthetic CPU image in [-1,1] (a smooth blob center-right).
procedure FillImage(Img: TNNetVolume);
var X, Y, C: integer; v: TNeuralFloat;
begin
  for Y := 0 to Img.SizeY - 1 do
    for X := 0 to Img.SizeX - 1 do
      for C := 0 to Img.Depth - 1 do
      begin
        v := Sin((X + C) * 0.8) * 0.5 + Cos(Y * 0.7) * 0.4;
        if v < -1 then v := -1; if v > 1 then v := 1;
        Img[X, Y, C] := v;
      end;
end;

var
  VisionNet, TextNet, DecoderNet: TNNet;
  Config: TCLIPSegConfig;
  Image, TokenIds, Mask: TNNetVolume;
  STPath, CfgPath: string;
  i, SeqLen: integer;
  // Hand-typed prompt token ids for the pico fixture vocab (real checkpoints:
  // feed your CLIP tokenizer's ids; last id is the eot/argmax token).
  PromptIds: array[0..4] of integer = (0, 7, 11, 4, 29);
begin
  if ParamCount >= 1 then
  begin
    STPath := ParamStr(1);
    if ParamCount >= 2 then CfgPath := ParamStr(2)
    else CfgPath := ExtractFilePath(STPath) + 'config.json';
    WriteLn('Loading CLIPSeg from checkpoint: ', STPath);
  end
  else
  begin
    STPath := cFixtureST;
    CfgPath := cFixtureCfg;
    WriteLn('No checkpoint passed; using the committed pico fixture ' +
      '(random weights -- wiring SMOKE demo).');
  end;

  SeqLen := Length(PromptIds);
  // Build the text tower sized to the prompt length.
  BuildCLIPSegFromSafeTensors(STPath, VisionNet, TextNet, DecoderNet, Config,
    {TextSeqLen=}SeqLen, {pTrainable=}false, CfgPath);
  Image := TNNetVolume.Create(Config.ImageSize, Config.ImageSize,
    Config.NumChannels);
  TokenIds := TNNetVolume.Create(SeqLen, 1, 1);
  Mask := TNNetVolume.Create;
  try
    WriteLn(CLIPSegConfigToString(Config));
    FillImage(Image);
    for i := 0 to SeqLen - 1 do TokenIds.FData[i] := PromptIds[i];
    Write('Prompt token ids: ');
    for i := 0 to SeqLen - 1 do Write(PromptIds[i], ' ');
    WriteLn;

    RunCLIPSeg(VisionNet, TextNet, DecoderNet, Config, Image, TokenIds, Mask);
    WriteLn('Mask logits: ', Mask.SizeX, 'x', Mask.SizeY, 'x', Mask.Depth);

    WriteColorPPM(Image, cImageFile);
    WriteMaskPPM(Mask, cMaskFile);
    WriteLn('Wrote ', cImageFile, ' and ', cMaskFile, '.');

    WriteLn; WriteLn('image:'); AsciiPreview(Image);
    WriteLn; WriteLn('mask (logit > 0 = #):'); AsciiMask(Mask);
  finally
    Image.Free;
    TokenIds.Free;
    Mask.Free;
    VisionNet.Free;
    TextNet.Free;
    DecoderNet.Free;
  end;
end.
