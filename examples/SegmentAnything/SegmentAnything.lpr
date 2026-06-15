program SegmentAnything;
// Segment Anything (SAM) image-encoder smoke demo on the committed pico SAM
// fixture (or any facebook/sam-vit-* checkpoint passed as argument 1).
//
// SAM (Kirillov et al. 2023, "Segment Anything", https://arxiv.org/abs/2304.02643)
// is a PROMPTABLE-segmentation model: a heavy ViT-det image encoder produces a
// dense (Grid, Grid, 256) image embedding ONCE per image, and a lightweight
// prompt-conditioned mask decoder then turns a user CLICK (point) or BOX into a
// binary mask cheaply. This importer (BuildSAMFromSafeTensors) lands the IMAGE
// ENCODER with float64 parity (< 1e-4 vs HF); the two-way mask decoder is a
// documented follow-up (see tasklist.md).
//
// This demo therefore segments from one click using an ENCODER-ONLY proxy: it
// runs the encoder to get the per-patch embedding grid, takes the patch under
// the click as the "object prototype", and produces a mask by thresholding the
// cosine similarity of every patch to that prototype. This is NOT the trained
// SAM decoder - it is a transparent stand-in that exercises the real encoder
// end to end and writes a binary mask PPM you can open. When the decoder lands,
// this click->mask path swaps the cosine proxy for the real decoder.
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_sam.* pico fixture):
//   examples/SegmentAnything/SegmentAnything
//   examples/SegmentAnything/SegmentAnything /path/to/sam-vit-base/model.safetensors
//
// Writes sam_mask.ppm (a Grid x Grid binary mask, upscaled for visibility).
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  SysUtils, Math, Classes, neuralvolume, neuralnetwork, neuralpretrained;

function DefaultCheckpoint(): string;
begin
  Result := 'tests' + DirectorySeparator + 'fixtures' +
    DirectorySeparator + 'tiny_sam.safetensors';
  if not FileExists(Result) then
    Result := '..' + DirectorySeparator + '..' + DirectorySeparator +
      'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      'tiny_sam.safetensors';
end;

function DefaultConfig(const Ckpt: string): string;
begin
  Result := ExtractFilePath(Ckpt) + 'tiny_sam_config.json';
  if not FileExists(Result) then
    Result := ExtractFilePath(Ckpt) + 'config.json';
end;

var
  Ckpt, ConfigPath: string;
  NN: TNNet;
  Config: TSAMConfig;
  Img, Emb: TNNetVolume;
  Grid, OutCh, x, y, c, ClickX, ClickY, Upscale, py, px: integer;
  ProtoIdx, Dot, NA, NB, CosSim, Thr: TNeuralFloat;
  Mask: array of array of boolean;
  PPM: TStringList;
  Line: string;
  V: integer;
begin
  if ParamCount >= 1 then Ckpt := ParamStr(1) else Ckpt := DefaultCheckpoint();
  if not FileExists(Ckpt) then
  begin
    WriteLn('Checkpoint not found: ', Ckpt);
    WriteLn('Pass a sam-vit-* model.safetensors as argument 1, or run from the');
    WriteLn('repo root so the committed pico fixture is found.');
    Halt(1);
  end;
  ConfigPath := DefaultConfig(Ckpt);
  WriteLn('SAM image-encoder smoke demo');
  WriteLn('  checkpoint: ', Ckpt);
  WriteLn('  config    : ', ConfigPath);

  NN := BuildSAMFromSafeTensors(Ckpt, Config, {pInferenceOnly=}true, ConfigPath);
  try
    WriteLn('  ', SAMConfigToString(Config));
    Grid := Config.ImageSize div Config.PatchSize;
    OutCh := Config.OutputChannels;
    WriteLn('  embedding grid = ', Grid, 'x', Grid, ' x ', OutCh, ' channels');

    // Synthetic input: a smooth dyadic gradient (deterministic, no I/O needed).
    Img := TNNetVolume.Create();
    Img.ReSize(Config.ImageSize, Config.ImageSize, Config.NumChannels);
    for c := 0 to Config.NumChannels - 1 do
      for y := 0 to Config.ImageSize - 1 do
        for x := 0 to Config.ImageSize - 1 do
          Img.FData[(y * Config.ImageSize + x) * Config.NumChannels + c] :=
            (((c * 256 + y * 16 + x) * 5) mod 17 - 8) / 8.0;

    NN.Compute(Img);
    Emb := NN.GetLastLayer().Output;  // (Grid, Grid, OutCh), X=col Y=row

    // One CLICK at the grid centre -> the object prototype patch embedding.
    ClickX := Grid div 2;  // column
    ClickY := Grid div 2;  // row
    WriteLn('  click at grid (row=', ClickY, ', col=', ClickX, ')');

    // Cosine similarity of every patch to the clicked prototype, thresholded.
    SetLength(Mask, Grid, Grid);
    Thr := 0.5;
    for y := 0 to Grid - 1 do
      for x := 0 to Grid - 1 do
      begin
        Dot := 0; NA := 0; NB := 0;
        for c := 0 to OutCh - 1 do
        begin
          // Emb.GetRawPos(col=X, row=Y, c)
          ProtoIdx := Emb.FData[(Emb.SizeX * ClickY + ClickX) * OutCh + c];
          Dot := Dot + ProtoIdx * Emb.FData[(Emb.SizeX * y + x) * OutCh + c];
          NA := NA + Sqr(ProtoIdx);
          NB := NB + Sqr(Emb.FData[(Emb.SizeX * y + x) * OutCh + c]);
        end;
        if (NA > 0) and (NB > 0) then CosSim := Dot / (Sqrt(NA) * Sqrt(NB))
        else CosSim := 0;
        Mask[y][x] := CosSim >= Thr;
      end;

    // Write a binary mask PPM (P3), upscaled for visibility.
    Upscale := Max(1, 64 div Grid);
    PPM := TStringList.Create();
    try
      PPM.Add('P3');
      PPM.Add(IntToStr(Grid * Upscale) + ' ' + IntToStr(Grid * Upscale));
      PPM.Add('255');
      for y := 0 to Grid - 1 do
        for py := 0 to Upscale - 1 do
        begin
          Line := '';
          for x := 0 to Grid - 1 do
            for px := 0 to Upscale - 1 do
            begin
              if Mask[y][x] then V := 255 else V := 0;
              Line := Line + IntToStr(V) + ' ' + IntToStr(V) + ' ' +
                IntToStr(V) + ' ';
            end;
          PPM.Add(TrimRight(Line));
        end;
      PPM.SaveToFile('sam_mask.ppm');
      WriteLn('  wrote sam_mask.ppm (', Grid * Upscale, 'x', Grid * Upscale,
        ', cosine threshold ', Thr:0:2, ')');
    finally
      PPM.Free;
    end;
    Img.Free;
    WriteLn('Done. (Mask is an encoder-only cosine proxy; the trained SAM mask');
    WriteLn('decoder is a documented follow-up.)');
  finally
    NN.Free;
  end;
end.
