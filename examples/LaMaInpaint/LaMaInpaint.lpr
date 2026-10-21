program LaMaInpaint;
(*
LaMaInpaint: removes the masked part of a photo in one forward pass with the
big-lama large-mask inpainting generator (Suvorov et al. 2021, "Resolution-robust
Large Mask Inpainting with Fourier Convolutions", arXiv:2109.07161), imported by
BuildLaMa (neuralpretrained.pas) from anyisalin/big-lama's big-lama.safetensors.

Pipeline (as lama's predict.py): load the RGB image in [0,1] and the mask (any
non-zero pixel = hole), pad both on the bottom/right by symmetric mirroring to a
multiple of 8, feed image*(1-mask) | mask, composite mask*pred + (1-mask)*image,
crop back and save a PNG.

USAGE
  ./LaMaInpaint --model big-lama.safetensors --image photo.png --mask mask.png
                --out result.png

Coded by Claude (AI).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuraldatasets, neuralsafetensors,
  neuralpretrained;

// Mirror index of numpy.pad(mode='symmetric') for a pad on the high side only.
function SymmetricIndex(Pos, Size: integer): integer;
begin
  Result := Pos;
  while Result >= Size do
  begin
    Result := 2 * Size - 1 - Result;
    if Result < 0 then Result := -Result - 1;
  end;
end;

// Builds the (PaddedW, PaddedH, 4) net input image*(1-mask) | mask from the
// (W, H, 3) image in [0,1] and the (W, H, 1) binary mask.
procedure BuildNetInput(Img, Mask, NetInput: TNNetVolume;
  PaddedW, PaddedH: integer);
var
  X, Y, SrcX, SrcY, C: integer;
  Hole: TNeuralFloat;
begin
  NetInput.ReSize(PaddedW, PaddedH, 4);
  for Y := 0 to PaddedH - 1 do
  begin
    SrcY := SymmetricIndex(Y, Img.SizeY);
    for X := 0 to PaddedW - 1 do
    begin
      SrcX := SymmetricIndex(X, Img.SizeX);
      Hole := Mask[SrcX, SrcY, 0];
      for C := 0 to 2 do
        NetInput[X, Y, C] := Img[SrcX, SrcY, C] * (1 - Hole);
      NetInput[X, Y, 3] := Hole;
    end;
  end;
end;

// OutImg (W, H, 3) in 0..255 := mask*Pred + (1-mask)*Img over the unpadded area.
procedure Composite(Img, Mask, Pred, OutImg: TNNetVolume);
var
  X, Y, C: integer;
  Hole: TNeuralFloat;
begin
  OutImg.ReSize(Img.SizeX, Img.SizeY, 3);
  for Y := 0 to Img.SizeY - 1 do
    for X := 0 to Img.SizeX - 1 do
    begin
      Hole := Mask[X, Y, 0];
      for C := 0 to 2 do
        OutImg[X, Y, C] := 255 *
          (Hole * Pred[X, Y, C] + (1 - Hole) * Img[X, Y, C]);
    end;
end;

procedure Usage();
begin
  WriteLn('Usage: LaMaInpaint --model big-lama.safetensors --image FILE ' +
    '--mask FILE --out FILE.png');
  WriteLn('  mask: same size as the image; any non-zero pixel is a hole.');
end;

var
  ModelFile, ImageFile, MaskFile, OutFile, Arg: string;
  ArgPos, Factor, X, Y: integer;
  Reader: TNNetSafeTensorsReader;
  Config: TLaMaConfig;
  NN: TNNet;
  Img, MaskRGB, Mask, NetInput, Inpainted: TNNetVolume;
  StartTime: double;
begin
  ModelFile := ''; ImageFile := ''; MaskFile := ''; OutFile := '';
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if (Arg = '--model') and (ArgPos < ParamCount) then begin Inc(ArgPos); ModelFile := ParamStr(ArgPos); end
    else if (Arg = '--image') and (ArgPos < ParamCount) then begin Inc(ArgPos); ImageFile := ParamStr(ArgPos); end
    else if (Arg = '--mask') and (ArgPos < ParamCount) then begin Inc(ArgPos); MaskFile := ParamStr(ArgPos); end
    else if (Arg = '--out') and (ArgPos < ParamCount) then begin Inc(ArgPos); OutFile := ParamStr(ArgPos); end
    else
    begin
      WriteLn('Unknown or incomplete argument: ', Arg);
      Usage();
      Halt(1);
    end;
    Inc(ArgPos);
  end;
  if (ModelFile = '') or (ImageFile = '') or (MaskFile = '') or (OutFile = '') then
  begin
    Usage();
    Halt(1);
  end;

  Img := TNNetVolume.Create;
  MaskRGB := TNNetVolume.Create;
  Mask := TNNetVolume.Create;
  NetInput := TNNetVolume.Create;
  Inpainted := TNNetVolume.Create;
  NN := nil;
  Reader := nil;
  try
    if not LoadImageFromFileIntoVolume(ImageFile, Img) then
      raise Exception.Create('could not load the image ' + ImageFile);
    if not LoadImageFromFileIntoVolume(MaskFile, MaskRGB) then
      raise Exception.Create('could not load the mask ' + MaskFile);
    if (MaskRGB.SizeX <> Img.SizeX) or (MaskRGB.SizeY <> Img.SizeY) then
      raise Exception.Create(Format('the mask is %dx%d but the image is %dx%d',
        [MaskRGB.SizeX, MaskRGB.SizeY, Img.SizeX, Img.SizeY]));
    Img.Divi(255);
    Mask.ReSize(Img.SizeX, Img.SizeY, 1);
    for Y := 0 to Img.SizeY - 1 do
      for X := 0 to Img.SizeX - 1 do
        if (MaskRGB[X, Y, 0] > 0) or (MaskRGB[X, Y, 1] > 0) or (MaskRGB[X, Y, 2] > 0)
          then Mask[X, Y, 0] := 1
          else Mask[X, Y, 0] := 0;

    Reader := CreatePretrainedTensorReader(ModelFile);
    Config := ReadLaMaConfigFromReader(Reader, 0, 0);
    Factor := 1 shl Config.DownsampleCount;
    // The builder needs at least 2 bottleneck cells per axis.
    Config.InputWidth := Max(2 * Factor, ((Img.SizeX + Factor - 1) div Factor) * Factor);
    Config.InputHeight := Max(2 * Factor, ((Img.SizeY + Factor - 1) div Factor) * Factor);
    WriteLn('Image ', Img.SizeX, 'x', Img.SizeY, ', padded to ',
      Config.InputWidth, 'x', Config.InputHeight, '.');
    WriteLn(LaMaConfigToString(Config));
    NN := BuildLaMa(Reader, Config, {pTrainable=}false);
    FreeAndNil(Reader);

    BuildNetInput(Img, Mask, NetInput, Config.InputWidth, Config.InputHeight);
    StartTime := Now();
    NN.Compute(NetInput);
    WriteLn('Forward pass: ', ((Now() - StartTime) * 86400):0:2, ' s.');
    Composite(Img, Mask, NN.GetLastLayer().Output, Inpainted);
    if not SaveImageFromVolumeIntoFile(Inpainted, OutFile) then
      raise Exception.Create('could not save ' + OutFile);
    WriteLn('Wrote ', OutFile, '.');
  finally
    Reader.Free;
    NN.Free;
    Inpainted.Free;
    NetInput.Free;
    Mask.Free;
    MaskRGB.Free;
    Img.Free;
  end;
end.
