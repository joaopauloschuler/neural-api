program SegmentAnything;
// Segment Anything (SAM) PROMPTABLE click->mask demo on the committed pico SAM
// fixture (or any facebook/sam-vit-* checkpoint passed as argument 1).
//
// SAM (Kirillov et al. 2023, "Segment Anything", https://arxiv.org/abs/2304.02643)
// is a PROMPTABLE-segmentation model: a heavy ViT-det image encoder produces a
// dense (Grid, Grid, OutCh) image embedding ONCE per image, and a lightweight
// prompt-conditioned mask decoder then turns user CLICKS (points) or a box into
// a binary mask cheaply. TSAMPredictor (neuralpretrained) wraps the whole HF
// SamProcessor pipeline: longest-side resize + normalize + pad, prompt rescaling,
// encoder, decoder, and the mask upscale back to the photo's own resolution.
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_sam.* pico fixture):
//   examples/SegmentAnything/SegmentAnything
//   examples/SegmentAnything/SegmentAnything /path/to/sam-vit-base/model.safetensors \
//     --image photo.jpg --point 320,240 --point 100,80,0 [--box x0,y0,x1,y1]
//     [--multimask] [--out sam]
//
// Writes <out>_mask.png (white = mask, at the photo's resolution) and
// <out>_overlay.png (the photo with the mask tinted red).
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  SysUtils, Math, Classes, neuralvolume, neuralnetwork, neuralpretrained,
  neuraldatasets;

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

// Splits "a,b[,c...]" into floats; False when a field is not a number.
function ParseFloatList(const Text: string; out Values: TNeuralFloatDynArr): boolean;
var
  Fields: TStringList;
  FieldPos, MaxFieldPos: integer;
  FormatSettings: TFormatSettings;
begin
  FormatSettings := DefaultFormatSettings;
  FormatSettings.DecimalSeparator := '.';
  Fields := TStringList.Create;
  try
    Fields.Delimiter := ',';
    Fields.StrictDelimiter := true;
    Fields.DelimitedText := Text;
    SetLength(Values, Fields.Count);
    Result := Fields.Count > 0;
    MaxFieldPos := Fields.Count - 1;
    for FieldPos := 0 to MaxFieldPos do
      if not TryStrToFloat(Trim(Fields[FieldPos]), Values[FieldPos],
        FormatSettings) then Result := false;
  finally
    Fields.Free;
  end;
end;

// A 0..255 RGB gradient with a bright disc, so the default run needs no file.
procedure BuildSyntheticImage(Image: TNNetVolume; Width, Height: integer);
var
  X, Y, MaxX, MaxY: integer;
  InDisc: boolean;
begin
  Image.ReSize(Width, Height, 3);
  MaxX := Width - 1;
  MaxY := Height - 1;
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      InDisc := Sqr(X - Width div 2) + Sqr(Y - Height div 2) <
        Sqr(Min(Width, Height) div 4);
      if InDisc then
      begin
        Image[X, Y, 0] := 240; Image[X, Y, 1] := 200; Image[X, Y, 2] := 60;
      end
      else
      begin
        Image[X, Y, 0] := 40 + (X * 120) div Width;
        Image[X, Y, 1] := 60 + (Y * 120) div Height;
        Image[X, Y, 2] := 150;
      end;
    end;
end;

procedure Usage();
begin
  WriteLn('Usage: SegmentAnything [checkpoint.safetensors] [--config FILE]');
  WriteLn('  [--image FILE] [--point X,Y[,LABEL]]... [--box X0,Y0,X1,Y1]');
  WriteLn('  [--multimask] [--out PREFIX]');
  WriteLn('Coordinates are original-image pixels; LABEL 1 = foreground');
  WriteLn('(default), 0 = background. Without --image a synthetic 96x64 image');
  WriteLn('is segmented; without a prompt the click is the image centre.');
end;

var
  Ckpt, ConfigPath, ImagePath, OutPrefix, Arg: string;
  Predictor: TSAMPredictor;
  Image, Masks, IoUScores, MaskImage, Overlay: TNNetVolume;
  Points, Box, Fields: TNeuralFloatDynArr;
  Labels: array of integer;
  MultiMask: boolean;
  ArgPos, PointCount, BestMask, MaskPos, MaxMaskPos, X, Y, MaxX, MaxY: integer;
  ForegroundCount: integer;
  MaskValue: TNeuralFloat;
begin
  Ckpt := '';
  ConfigPath := '';
  ImagePath := '';
  OutPrefix := 'sam';
  MultiMask := false;
  SetLength(Points, 0);
  SetLength(Labels, 0);
  SetLength(Box, 0);
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if (Arg = '--help') or (Arg = '-h') then
    begin
      Usage();
      Halt(0);
    end
    else if (Arg = '--multimask') then MultiMask := true
    else if (Arg = '--image') or (Arg = '--point') or (Arg = '--box') or
      (Arg = '--out') or (Arg = '--config') then
    begin
      if ArgPos = ParamCount then
      begin
        WriteLn(Arg, ' needs a value.');
        Halt(2);
      end;
      Inc(ArgPos);
      if Arg = '--image' then ImagePath := ParamStr(ArgPos)
      else if Arg = '--out' then OutPrefix := ParamStr(ArgPos)
      else if Arg = '--config' then ConfigPath := ParamStr(ArgPos)
      else if Arg = '--point' then
      begin
        if not ParseFloatList(ParamStr(ArgPos), Fields) or
          (Length(Fields) < 2) or (Length(Fields) > 3) then
        begin
          WriteLn('--point expects X,Y or X,Y,LABEL, got "', ParamStr(ArgPos), '".');
          Halt(2);
        end;
        PointCount := Length(Labels);
        SetLength(Points, PointCount * 2 + 2);
        SetLength(Labels, PointCount + 1);
        Points[PointCount * 2] := Fields[0];
        Points[PointCount * 2 + 1] := Fields[1];
        if Length(Fields) = 3 then Labels[PointCount] := Round(Fields[2])
        else Labels[PointCount] := 1;
      end
      else
      begin
        if not ParseFloatList(ParamStr(ArgPos), Box) or (Length(Box) <> 4) then
        begin
          WriteLn('--box expects X0,Y0,X1,Y1, got "', ParamStr(ArgPos), '".');
          Halt(2);
        end;
      end;
    end
    else if (Copy(Arg, 1, 2) <> '--') and (Ckpt = '') then Ckpt := Arg
    else
    begin
      WriteLn('Unknown argument: ', Arg);
      Usage();
      Halt(2);
    end;
    Inc(ArgPos);
  end;

  if Ckpt = '' then Ckpt := DefaultCheckpoint();
  if not FileExists(Ckpt) then
  begin
    WriteLn('Checkpoint not found: ', Ckpt);
    WriteLn('Pass a sam-vit-* model.safetensors as argument 1, or run from the');
    WriteLn('repo root so the committed pico fixture is found.');
    Halt(1);
  end;
  if ConfigPath = '' then ConfigPath := DefaultConfig(Ckpt);
  WriteLn('SAM promptable segmentation demo');
  WriteLn('  checkpoint: ', Ckpt);
  WriteLn('  config    : ', ConfigPath);

  Image := TNNetVolume.Create;
  Masks := TNNetVolume.Create;
  IoUScores := TNNetVolume.Create;
  MaskImage := TNNetVolume.Create;
  Overlay := TNNetVolume.Create;
  Predictor := TSAMPredictor.Create(Ckpt, ConfigPath);
  try
    WriteLn('  ', SAMConfigToString(Predictor.Config));
    if ImagePath <> '' then
    begin
      if not LoadImageFromFileIntoVolume(ImagePath, Image) then
      begin
        WriteLn('Could not load image: ', ImagePath);
        Halt(1);
      end;
      WriteLn('  image     : ', ImagePath, ' (', Image.SizeX, 'x', Image.SizeY, ')');
    end
    else
    begin
      BuildSyntheticImage(Image, 96, 64);
      WriteLn('  image     : synthetic ', Image.SizeX, 'x', Image.SizeY);
    end;
    if (Length(Labels) = 0) and (Length(Box) = 0) then
    begin
      SetLength(Points, 2);
      SetLength(Labels, 1);
      Points[0] := Image.SizeX / 2;
      Points[1] := Image.SizeY / 2;
      Labels[0] := 1;
    end;
    for PointCount := 0 to Length(Labels) - 1 do
      WriteLn('  point     : (x=', Points[PointCount * 2]:0:1, ', y=',
        Points[PointCount * 2 + 1]:0:1, ') label ', Labels[PointCount]);
    if Length(Box) = 4 then
      WriteLn('  box       : (', Box[0]:0:1, ', ', Box[1]:0:1, ') - (',
        Box[2]:0:1, ', ', Box[3]:0:1, ')');

    // The encoder runs once here; every Predict call afterwards is decoder-only.
    Predictor.SetImage(Image);
    Predictor.Predict(Points, Labels, Box, MultiMask, Masks, IoUScores);

    BestMask := 0;
    MaxMaskPos := Masks.Depth - 1;
    for MaskPos := 0 to MaxMaskPos do
    begin
      WriteLn('  mask ', MaskPos, ': predicted IoU ', IoUScores.FData[MaskPos]:0:4);
      if IoUScores.FData[MaskPos] > IoUScores.FData[BestMask] then
        BestMask := MaskPos;
    end;
    if Masks.Depth > 1 then
      WriteLn('  keeping mask ', BestMask, ' (highest predicted IoU)');

    MaskImage.ReSize(Image.SizeX, Image.SizeY, 3);
    Overlay.Copy(Image);
    ForegroundCount := 0;
    MaxX := Image.SizeX - 1;
    MaxY := Image.SizeY - 1;
    for Y := 0 to MaxY do
      for X := 0 to MaxX do
      begin
        MaskValue := Masks[X, Y, BestMask];
        MaskImage[X, Y, 0] := 255 * MaskValue;
        MaskImage[X, Y, 1] := 255 * MaskValue;
        MaskImage[X, Y, 2] := 255 * MaskValue;
        if MaskValue > 0 then
        begin
          Inc(ForegroundCount);
          Overlay[X, Y, 0] := 0.5 * Overlay[X, Y, 0] + 127.5;
          Overlay[X, Y, 1] := 0.5 * Overlay[X, Y, 1];
          Overlay[X, Y, 2] := 0.5 * Overlay[X, Y, 2];
        end;
      end;
    SaveImageFromVolumeIntoFile(MaskImage, OutPrefix + '_mask.png');
    SaveImageFromVolumeIntoFile(Overlay, OutPrefix + '_overlay.png');
    WriteLn('  wrote ', OutPrefix, '_mask.png and ', OutPrefix, '_overlay.png (',
      Image.SizeX, 'x', Image.SizeY, '); ', ForegroundCount, '/',
      Image.SizeX * Image.SizeY, ' pixels foreground');
  finally
    Predictor.Free;
    Overlay.Free;
    MaskImage.Free;
    IoUScores.Free;
    Masks.Free;
    Image.Free;
  end;
end.
