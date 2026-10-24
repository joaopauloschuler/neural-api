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

Outpainting (--outpaint L,T,R,B instead of --mask): the image is placed on a
canvas L/T/R/B pixels larger per side, the new border filled by repeating the
edge pixels; the mask covers that border plus --outpaint-overlap pixels (default
8) into the image, so the seam is regenerated too (BuildOutpaintCanvas).

USAGE
  ./LaMaInpaint --model big-lama.safetensors --image photo.png --mask mask.png
                --out result.png
  ./LaMaInpaint --model big-lama.safetensors --image photo.png
                --outpaint 0,0,128,0 [--outpaint-overlap 8] --out wider.png

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
  neuralpretrained, neuralimageresize;

const
  csDefaultOutpaintOverlap = 8;

procedure Usage();
begin
  WriteLn('Usage: LaMaInpaint --model big-lama.safetensors --image FILE ' +
    '--mask FILE --out FILE.png');
  WriteLn('   or: LaMaInpaint --model big-lama.safetensors --image FILE ' +
    '--outpaint L,T,R,B [--outpaint-overlap N] --out FILE.png');
  WriteLn('  mask: same size as the image; any non-zero pixel is a hole.');
  WriteLn('  --outpaint L,T,R,B: pixels to add on the left, top, right and ' +
    'bottom; the new border');
  WriteLn('    and --outpaint-overlap pixels into the image (default ',
    csDefaultOutpaintOverlap, ') are generated.');
end;

var
  ModelFile, ImageFile, MaskFile, OutFile, OutpaintText, Arg, Problem: string;
  ArgPos, Factor, X, Y, OutpaintOverlap: integer;
  OutpaintLeft, OutpaintTop, OutpaintRight, OutpaintBottom: integer;
  Reader: TNNetSafeTensorsReader;
  Config: TLaMaConfig;
  NN: TNNet;
  Img, MaskRGB, Mask, NetInput, Inpainted, Original: TNNetVolume;
  StartTime: double;
begin
  ModelFile := ''; ImageFile := ''; MaskFile := ''; OutFile := '';
  OutpaintText := '';
  OutpaintOverlap := csDefaultOutpaintOverlap;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if (Arg = '--model') and (ArgPos < ParamCount) then begin Inc(ArgPos); ModelFile := ParamStr(ArgPos); end
    else if (Arg = '--image') and (ArgPos < ParamCount) then begin Inc(ArgPos); ImageFile := ParamStr(ArgPos); end
    else if (Arg = '--mask') and (ArgPos < ParamCount) then begin Inc(ArgPos); MaskFile := ParamStr(ArgPos); end
    else if (Arg = '--out') and (ArgPos < ParamCount) then begin Inc(ArgPos); OutFile := ParamStr(ArgPos); end
    else if (Arg = '--outpaint') and (ArgPos < ParamCount) then begin Inc(ArgPos); OutpaintText := ParamStr(ArgPos); end
    else if (Arg = '--outpaint-overlap') and (ArgPos < ParamCount) then
    begin
      Inc(ArgPos);
      OutpaintOverlap := StrToIntDef(ParamStr(ArgPos), -1);
      if OutpaintOverlap < 0 then
      begin
        WriteLn('--outpaint-overlap: "', ParamStr(ArgPos), '" is not a ',
          'pixel count >= 0.');
        Halt(1);
      end;
    end
    else
    begin
      WriteLn('Unknown or incomplete argument: ', Arg);
      Usage();
      Halt(1);
    end;
    Inc(ArgPos);
  end;
  if (ModelFile = '') or (ImageFile = '') or (OutFile = '') or
    ((MaskFile = '') = (OutpaintText = '')) then
  begin
    if (MaskFile <> '') and (OutpaintText <> '') then
      WriteLn('--mask and --outpaint are exclusive.');
    Usage();
    Halt(1);
  end;
  if OutpaintText <> '' then
  begin
    Problem := ParseOutpaintSides(OutpaintText, OutpaintLeft, OutpaintTop,
      OutpaintRight, OutpaintBottom);
    if Problem <> '' then
    begin
      WriteLn('--outpaint: ', Problem);
      Halt(1);
    end;
  end;

  Img := TNNetVolume.Create;
  MaskRGB := TNNetVolume.Create;
  Mask := TNNetVolume.Create;
  NetInput := TNNetVolume.Create;
  Inpainted := TNNetVolume.Create;
  Original := TNNetVolume.Create;
  NN := nil;
  Reader := nil;
  try
    if not LoadImageFromFileIntoVolume(ImageFile, Img) then
      raise Exception.Create('could not load the image ' + ImageFile);
    Img.Divi(255);
    if OutpaintText <> '' then
    begin
      Original.Copy(Img);
      BuildOutpaintCanvas(Original, OutpaintLeft, OutpaintTop, OutpaintRight,
        OutpaintBottom, OutpaintOverlap, Img, Mask);
      WriteLn('Outpaint ', OutpaintLeft, ',', OutpaintTop, ',', OutpaintRight,
        ',', OutpaintBottom, ' pixels (overlap ', OutpaintOverlap, '): ',
        Original.SizeX, 'x', Original.SizeY, ' -> ', Img.SizeX, 'x',
        Img.SizeY, '.');
    end
    else
    begin
      if not LoadImageFromFileIntoVolume(MaskFile, MaskRGB) then
        raise Exception.Create('could not load the mask ' + MaskFile);
      if (MaskRGB.SizeX <> Img.SizeX) or (MaskRGB.SizeY <> Img.SizeY) then
        raise Exception.Create(Format('the mask is %dx%d but the image is %dx%d',
          [MaskRGB.SizeX, MaskRGB.SizeY, Img.SizeX, Img.SizeY]));
      Mask.ReSize(Img.SizeX, Img.SizeY, 1);
      for Y := 0 to Img.SizeY - 1 do
        for X := 0 to Img.SizeX - 1 do
          if (MaskRGB[X, Y, 0] > 0) or (MaskRGB[X, Y, 1] > 0) or (MaskRGB[X, Y, 2] > 0)
            then Mask[X, Y, 0] := 1
            else Mask[X, Y, 0] := 0;
    end;

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

    LaMaBuildInput(Img, Mask, NetInput, Config.InputWidth, Config.InputHeight);
    StartTime := Now();
    NN.Compute(NetInput);
    WriteLn('Forward pass: ', ((Now() - StartTime) * 86400):0:2, ' s.');
    LaMaComposite(Img, Mask, NN.GetLastLayer().Output, Inpainted);
    if not SaveImageFromVolumeIntoFile(Inpainted, OutFile) then
      raise Exception.Create('could not save ' + OutFile);
    WriteLn('Wrote ', OutFile, '.');
  finally
    Reader.Free;
    NN.Free;
    Original.Free;
    Inpainted.Free;
    NetInput.Free;
    Mask.Free;
    MaskRGB.Free;
    Img.Free;
  end;
end.
