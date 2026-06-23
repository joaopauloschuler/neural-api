/// Dense semantic segmentation with an imported SegFormer (MiT) model.
///
/// SegFormer (Xie et al. 2021, "SegFormer: Simple and Efficient Design for
/// Semantic Segmentation with Transformers") is a hierarchical Mix-Transformer
/// encoder (overlap-patch embeddings + spatial-reduction efficient attention +
/// Mix-FFN, NO positional embedding) feeding a lightweight all-MLP decode head
/// that emits a per-pixel class map.
///
/// This example loads the committed PICO SegFormer parity fixture
/// (tests/fixtures/tiny_segformer.safetensors, a tiny MiT-b0-shaped model) with
/// BuildSegformerFromSafeTensors, runs it on a small synthetic CPU image, takes
/// the per-pixel argmax over the class logits, and renders the resulting label
/// map as a colored ASCII palette (one glyph per class) at the head resolution
/// (input/4). A real run would load nvidia/segformer-b0-finetuned-ade-512-512
/// the same way and colorize a photograph; the math is identical, only the
/// checkpoint and image differ. Kept tiny and CPU-only so it finishes in well
/// under a second.
///
/// Coded by Joao Paulo Schwarz Schuler with Claude (AI).
/// https://github.com/joaopauloschuler/neural-api
program SemanticSegmentation;

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuralpretrained;

const
  // Palette glyphs, one per class (the pico fixture has 5 classes).
  csPalette: array[0..7] of char = ('.', ':', '+', '#', '@', '*', 'o', '=');

function FixturePath(const FileName: string): string;
var
  Base: string;
begin
  // Run from the example dir or the repo root; probe both.
  Base := '../../tests/fixtures/' + FileName;
  if FileExists(Base) then Exit(Base);
  Base := 'tests/fixtures/' + FileName;
  if FileExists(Base) then Exit(Base);
  Result := FileName;
end;

var
  NN: TNNet;
  Config: TSegformerConfig;
  Img, Output: TNNetVolume;
  x, y, c, BestC, W, H, GW, GH, NLab: integer;
  BestVal: TNeuralFloat;
  Line: string;
  Counts: array of integer;
begin
  WriteLn('SegFormer semantic-segmentation example');
  WriteLn('---------------------------------------');
  NN := BuildSegformerFromSafeTensors(
    FixturePath('tiny_segformer.safetensors'), Config,
    {pTrainable=}false, FixturePath('tiny_segformer_config.json'));
  try
    WriteLn(SegformerConfigToString(Config));
    W := Config.ImageSize; H := Config.ImageSize;
    Img := TNNetVolume.Create;
    try
      // Synthetic test image: a smooth radial + diagonal RGB gradient so the
      // segmentation has spatial structure to show (any image works the same).
      Img.ReSize(W, H, Config.NumChannels);
      for y := 0 to H - 1 do
        for x := 0 to W - 1 do
        begin
          Img[x, y, 0] := Sin(x / 7.0) * 0.6;
          Img[x, y, 1] := Cos(y / 9.0) * 0.6;
          Img[x, y, 2] := (Sqrt(Sqr(x - W / 2) + Sqr(y - H / 2)) /
            (W / 2.0)) - 0.5;
        end;

      NN.Compute(Img);
      Output := NN.GetLastLayer().Output;
      GW := Output.SizeX; GH := Output.SizeY; NLab := Output.Depth;
      WriteLn(Format('input %dx%dx%d  ->  class map %dx%d over %d classes',
        [W, H, Config.NumChannels, GW, GH, NLab]));
      WriteLn;

      SetLength(Counts, NLab);
      for c := 0 to NLab - 1 do Counts[c] := 0;
      for y := 0 to GH - 1 do
      begin
        Line := '';
        for x := 0 to GW - 1 do
        begin
          // per-pixel argmax over the class logits.
          BestC := 0;
          BestVal := Output[x, y, 0];
          for c := 1 to NLab - 1 do
            if Output[x, y, c] > BestVal then
            begin
              BestVal := Output[x, y, c];
              BestC := c;
            end;
          Inc(Counts[BestC]);
          Line := Line + csPalette[BestC mod Length(csPalette)];
        end;
        WriteLn(Line);
      end;

      WriteLn;
      WriteLn('Per-class pixel counts (argmax label histogram):');
      for c := 0 to NLab - 1 do
        WriteLn(Format('  class %d  "%s"  %d px',
          [c, csPalette[c mod Length(csPalette)], Counts[c]]));
      WriteLn;
      WriteLn('Done. (A real run loads nvidia/segformer-b0-finetuned-ade the ' +
        'same way and colorizes a photo.)');
    finally
      Img.Free;
    end;
  finally
    NN.Free;
  end;
end.
