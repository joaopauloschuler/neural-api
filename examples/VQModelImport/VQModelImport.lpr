program VQModelImport;
(*
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

// VQModelImport -- imports a pretrained diffusers VQModel (VQGAN / discrete
// image tokenizer) into a CAI TNNetVqModel and ROUND-TRIPS a tiny image
// through the codebook: image -> grid of integer codebook token IDs -> image,
// reporting the reconstruction error. This is the discrete tokenizer used by
// autoregressive / masked image generators (MaskGIT, Parti, LlamaGen) to turn
// an image into a sequence of token IDs that a stock Llama/GPT LM can model.
//
// Usage:
//   VQModelImport [model.safetensors] [config.json]
//
// With no arguments it loads the committed pico fixture
// (tests/fixtures/tiny_vqmodel.*), a small random VQModel used by the parity
// tests, and round-trips its pinned synthetic image. Point it at a real
// diffusers VQModel checkpoint + config.json for the full thing.

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralpretrained;

var
  ModelFile, ConfigFile, IoFile: string;
  Vq: TNNetVqModel;
  Config: TVqModelConfig;
  Image, Recon: TNNetVolume;
  Ids: TNeuralIntegerArray;
  ImgGrid, Grid, y, x, c: integer;
  RefJson: TStringList;
  RefRoot: TJSONData;
  RootObj: TJSONObject;
  ImgArr, RowArr, ColArr: TJSONArray;
  MaxAbs, MeanAbs, Diff, Sum: double;
  N: integer;
  DistinctUsed: integer;
  Seen: array of boolean;

begin
  if ParamCount >= 1 then ModelFile := ParamStr(1)
  else ModelFile := '../../tests/fixtures/tiny_vqmodel.safetensors';
  if ParamCount >= 2 then ConfigFile := ParamStr(2)
  else ConfigFile := '../../tests/fixtures/tiny_vqmodel_config.json';
  IoFile := ChangeFileExt(ModelFile, '') + '_io.json';
  // The pico fixture stores its pinned image under tiny_vqmodel_io.json.
  if not FileExists(IoFile) then
    IoFile := ExtractFilePath(ModelFile) + 'tiny_vqmodel_io.json';

  WriteLn('VQModelImport: loading ', ModelFile);
  Vq := BuildVqModelFromSafeTensors(ModelFile, Config, {pInferenceOnly=}false,
    ConfigFile);
  try
    WriteLn(VqModelConfigToString(Config));
    Grid := Config.LatentGrid;
    ImgGrid := Grid shl (Config.NumBlockOut - 1);
    WriteLn(Format('  image %dx%dx%d -> %dx%d token grid (%d tokens), ' +
      'codebook %d x %d', [ImgGrid, ImgGrid, Config.InChannels, Grid, Grid,
      Grid * Grid, Config.NumVqEmbeddings, Config.VqEmbedDim]));

    Image := TNNetVolume.Create;
    Recon := TNNetVolume.Create;
    RefJson := TStringList.Create;
    RefRoot := nil;
    try
      // Load the pinned image from the io fixture if present; else a ramp.
      Image.ReSize(ImgGrid, ImgGrid, Config.InChannels);
      if FileExists(IoFile) then
      begin
        RefJson.LoadFromFile(IoFile);
        RefRoot := GetJSON(RefJson.Text);
        RootObj := TJSONObject(RefRoot);
        ImgArr := TJSONArray(RootObj.Find('image'));
        for c := 0 to Config.InChannels - 1 do
        begin
          RowArr := TJSONArray(ImgArr.Items[c]);
          for y := 0 to ImgGrid - 1 do
          begin
            ColArr := TJSONArray(RowArr.Items[y]);
            for x := 0 to ImgGrid - 1 do
              Image.FData[(y * ImgGrid + x) * Config.InChannels + c] :=
                ColArr.Items[x].AsFloat;
          end;
        end;
        WriteLn('  using pinned image from ', ExtractFileName(IoFile));
      end
      else
      begin
        for y := 0 to ImgGrid - 1 do
          for x := 0 to ImgGrid - 1 do
            for c := 0 to Config.InChannels - 1 do
              Image.FData[(y * ImgGrid + x) * Config.InChannels + c] :=
                (((c * 256 + y * ImgGrid + x) * 5) mod 13 - 6) / 8.0;
        WriteLn('  using synthetic ramp image');
      end;

      // ---- ENCODE: image -> token grid ----
      Vq.EncodeImageToTokenList(Image, Ids);
      WriteLn('Token grid (codebook IDs):');
      SetLength(Seen, Config.NumVqEmbeddings);
      for x := 0 to High(Seen) do Seen[x] := false;
      for y := 0 to Grid - 1 do
      begin
        Write('  ');
        for x := 0 to Grid - 1 do
        begin
          Write(Format('%3d', [Ids[y * Grid + x]]));
          if Ids[y * Grid + x] < Config.NumVqEmbeddings then
            Seen[Ids[y * Grid + x]] := true;
        end;
        WriteLn;
      end;
      DistinctUsed := 0;
      for x := 0 to High(Seen) do if Seen[x] then Inc(DistinctUsed);
      WriteLn(Format('  distinct codes used: %d of %d', [DistinctUsed,
        Config.NumVqEmbeddings]));

      // ---- DECODE: token grid -> image ----
      Vq.DecodeTokensToImage(Ids, Recon);

      // ---- reconstruction error vs the original ----
      MaxAbs := 0; Sum := 0; N := 0;
      for c := 0 to Config.OutChannels - 1 do
        for y := 0 to ImgGrid - 1 do
          for x := 0 to ImgGrid - 1 do
          begin
            Diff := Abs(
              Recon.FData[(y * ImgGrid + x) * Config.OutChannels + c] -
              Image.FData[(y * ImgGrid + x) * Config.InChannels + c]);
            if Diff > MaxAbs then MaxAbs := Diff;
            Sum := Sum + Diff;
            Inc(N);
          end;
      MeanAbs := Sum / N;
      WriteLn(Format('Round-trip reconstruction error: max|diff| = %.5f, ' +
        'mean|diff| = %.5f', [MaxAbs, MeanAbs]));
      WriteLn('(A random pico VQModel does not reconstruct well; a pretrained ' +
        'checkpoint does. The point is the discrete image -> tokens -> image ' +
        'pipeline runs end to end.)');
    finally
      RefRoot.Free;
      RefJson.Free;
      Recon.Free;
      Image.Free;
    end;
  finally
    Vq.Free;
  end;
end.
