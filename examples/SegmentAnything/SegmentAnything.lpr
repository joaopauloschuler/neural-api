program SegmentAnything;
// Segment Anything (SAM) PROMPTABLE click->mask demo on the committed pico SAM
// fixture (or any facebook/sam-vit-* checkpoint passed as argument 1).
//
// SAM (Kirillov et al. 2023, "Segment Anything", https://arxiv.org/abs/2304.02643)
// is a PROMPTABLE-segmentation model: a heavy ViT-det image encoder produces a
// dense (Grid, Grid, OutCh) image embedding ONCE per image, and a lightweight
// prompt-conditioned mask decoder then turns a user CLICK (point) into a binary
// mask cheaply. Both stages are imported here with float64 parity (< 1e-4 vs
// HF): BuildSAMFromSafeTensors lands the image ENCODER, and RunSAMMaskDecoder
// lands the v1 single-point / single-mask DECODER (prompt encoder + two-way
// transformer + transposed-conv upscale + hypernetwork dot).
//
// This demo runs the REAL pipeline end to end: encode the image once, then feed
// one positive point click into the mask decoder and threshold the low-res mask
// logits at 0 (sigmoid 0.5) to a binary mask PPM you can open.
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_sam.* pico fixture):
//   examples/SegmentAnything/SegmentAnything
//   examples/SegmentAnything/SegmentAnything /path/to/sam-vit-base/model.safetensors [clickX clickY]
//
// Writes sam_mask.ppm (a 4*Grid x 4*Grid binary mask, upscaled for visibility).
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  SysUtils, Math, Classes, neuralvolume, neuralnetwork, neuralpretrained,
  neuralsafetensors;

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
  MDCfg: TSAMMaskDecoderConfig;
  Reader: TNNetSafeTensorsReader;
  Img, Emb, MaskLogits: TNNetVolume;
  Grid, OutCh, x, y, c, Upscale, py, px, MaskH, MaskW: integer;
  ClickX, ClickY: TNeuralFloat;
  PPM: TStringList;
  Line: string;
  V, OnCnt: integer;
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
  WriteLn('SAM promptable click->mask demo');
  WriteLn('  checkpoint: ', Ckpt);
  WriteLn('  config    : ', ConfigPath);

  NN := BuildSAMFromSafeTensors(Ckpt, Config, {pInferenceOnly=}true, ConfigPath);
  Reader := TNNetSafeTensorsReader.Create(Ckpt);
  Img := TNNetVolume.Create();
  MaskLogits := TNNetVolume.Create();
  try
    MDCfg := ReadSAMMaskDecoderConfig(ConfigPath, Config);
    WriteLn('  ', SAMConfigToString(Config));
    Grid := Config.ImageSize div Config.PatchSize;
    OutCh := Config.OutputChannels;
    WriteLn('  embedding grid = ', Grid, 'x', Grid, ' x ', OutCh, ' channels');

    // Synthetic input: a smooth dyadic gradient (deterministic, no I/O needed).
    Img.ReSize(Config.ImageSize, Config.ImageSize, Config.NumChannels);
    for c := 0 to Config.NumChannels - 1 do
      for y := 0 to Config.ImageSize - 1 do
        for x := 0 to Config.ImageSize - 1 do
          Img.FData[(y * Config.ImageSize + x) * Config.NumChannels + c] :=
            (((c * 256 + y * 16 + x) * 5) mod 17 - 8) / 8.0;

    NN.Compute(Img);
    Emb := NN.GetLastLayer().Output;  // (Grid, Grid, OutCh)

    // One positive CLICK, in ORIGINAL image PIXEL coordinates (x, y). Default to
    // the image centre; override via argv[2] argv[3].
    ClickX := Config.ImageSize / 2;
    ClickY := Config.ImageSize / 2;
    if ParamCount >= 3 then
    begin
      ClickX := StrToFloatDef(ParamStr(2), ClickX);
      ClickY := StrToFloatDef(ParamStr(3), ClickY);
    end;
    WriteLn('  positive click at pixel (x=', ClickX:0:1, ', y=', ClickY:0:1, ')');

    // Real SAM mask decoder: prompt encoder + two-way transformer + upscale +
    // hypernetwork dot -> low-res mask logits (4*Grid x 4*Grid).
    RunSAMMaskDecoder(Reader, MDCfg, Emb, ClickX, ClickY, {IsPositive=}true,
      MaskLogits);
    MaskH := MaskLogits.SizeX;
    MaskW := MaskLogits.SizeY;
    WriteLn('  mask logits = ', MaskH, 'x', MaskW,
      ' (threshold at 0 -> sigmoid 0.5)');

    // Write a binary mask PPM (P3), upscaled for visibility.
    Upscale := Max(1, 64 div MaskH);
    OnCnt := 0;
    PPM := TStringList.Create();
    try
      PPM.Add('P3');
      PPM.Add(IntToStr(MaskW * Upscale) + ' ' + IntToStr(MaskH * Upscale));
      PPM.Add('255');
      for y := 0 to MaskH - 1 do
        for py := 0 to Upscale - 1 do
        begin
          Line := '';
          for x := 0 to MaskW - 1 do
          begin
            if MaskLogits.FData[(y * MaskW + x) * 1 + 0] > 0 then
            begin
              V := 255;
              if py = 0 then Inc(OnCnt);
            end
            else V := 0;
            for px := 0 to Upscale - 1 do
              Line := Line + IntToStr(V) + ' ' + IntToStr(V) + ' ' +
                IntToStr(V) + ' ';
          end;
          PPM.Add(TrimRight(Line));
        end;
      PPM.SaveToFile('sam_mask.ppm');
      WriteLn('  wrote sam_mask.ppm (', MaskW * Upscale, 'x', MaskH * Upscale,
        '); ', OnCnt, '/', MaskH * MaskW, ' pixels foreground');
    finally
      PPM.Free;
    end;
    WriteLn('Done. (Real SAM image encoder + v1 single-point mask decoder, both');
    WriteLn('float64-parity importers.)');
  finally
    MaskLogits.Free;
    Img.Free;
    Reader.Free;
    NN.Free;
  end;
end.
