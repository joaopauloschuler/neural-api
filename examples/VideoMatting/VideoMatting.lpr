program VideoMatting;
(*
VideoMatting: Robust Video Matting (Lin et al., WACV 2022) with the RVM
importer in neuralpretrained.pas (BuildRVMFromSafeTensors + TRVMMatting); the
MobileNetV3 or ResNet-50 backbone is detected from the checkpoint.

Two modes:
  * no arguments: a smoke of the committed random pico fixture
    tests/fixtures/tiny_rvm_mobilenetv3.pth: synthesize a few small frames
    (one of them of another size), matte them, write the outputs.
  * --model FILE --input DIR --output DIR: matte every PNG / JPEG frame of a
    folder (sorted by name, numbers compared by value) with the real
    rvm_mobilenetv3.pth or rvm_resnet50.pth.

RVM is recurrent: four ConvGRU states carry what the net saw in the previous
frames, so the frames must be consecutive frames of one clip. For each frame
the program writes <name>_rgba.png (foreground RGB + alpha) and
<name>_green.png (the foreground over a green screen), and optionally all the
green-screen frames as one animated GIF. With --segmentation it writes only
<name>_seg.png (16-bit grayscale), the sigmoid of RVM's segmentation output at
the downsampled size.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  {$IFDEF OpenCL}neuralopencl,{$ENDIF}
  neuralnetwork, neuralvolume, neuraldatasets, neuralpretrained, neuralgif;

const
  // Fixture committed with the RVM importer tests.
  FixturesDir = '../../tests/fixtures/';
  // RVM's default green-screen colour (inference.py: [120, 255, 155] / 255).
  csGreenScreen: array[0..2] of TNeuralFloat = (120, 255, 155);

type
  TVideoMattingOptions = record
    ModelFile, InputDir, OutputDir, GifFile: string;
    Ratio: double; // 0 = pick from the frame size
    Fps: integer;
    Refiner: string; // '' = from the checkpoint
    Segmentation: boolean;
    UseOpenCL: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: VideoMatting                          (pico fixture smoke)');
  WriteLn('       VideoMatting --model FILE --input DIR --output DIR [options]');
  WriteLn('  --model FILE         rvm_mobilenetv3.pth or rvm_resnet50.pth ',
    '(or a .safetensors of it)');
  WriteLn('  --input DIR          folder of consecutive frames (PNG / JPEG), ',
    'sorted by name');
  WriteLn('  --output DIR         writes <frame>_rgba.png and <frame>_green.png');
  WriteLn('  --ratio R            downsample ratio in (0, 1] (default: 1 up to ',
    '512 px, else 480 / longest side)');
  WriteLn('  --gif FILE           also write the green-screen frames as an ',
    'animated GIF');
  WriteLn('  --fps N              GIF frame rate (default 25)');
  WriteLn('  --refiner NAME       deep_guided_filter or fast_guided_filter ',
    '(default: deep if the checkpoint has its weights, else fast)');
  WriteLn('  --segmentation       write <frame>_seg.png (segmentation pass, ',
    'downsampled size) instead of the matte');
  WriteLn('  --gpu                run the net on OpenCL (needs a build with ',
    '-dOpenCL)');
  WriteLn('  --gpu-platform N     OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N       OpenCL device index (default 0)');
end;

function ParseOptions(out Opt: TVideoMattingOptions): boolean;
var
  ArgPos: integer;
  Arg: string;

  function NextArg(): string;
  begin
    Inc(ArgPos);
    if ArgPos > ParamCount then
      raise Exception.Create(Arg + ' needs a value');
    Result := ParamStr(ArgPos);
  end;

begin
  Opt.ModelFile := '';
  Opt.InputDir := '';
  Opt.OutputDir := '';
  Opt.GifFile := '';
  Opt.Ratio := 0;
  Opt.Fps := 25;
  Opt.Refiner := '';
  Opt.Segmentation := false;
  Opt.UseOpenCL := false;
  Opt.OpenCLPlatform := 0;
  Opt.OpenCLDevice := 0;
  Result := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--model' then Opt.ModelFile := NextArg()
    else if Arg = '--input' then Opt.InputDir := NextArg()
    else if Arg = '--output' then Opt.OutputDir := NextArg()
    else if Arg = '--gif' then Opt.GifFile := NextArg()
    else if Arg = '--ratio' then Opt.Ratio := StrToFloat(NextArg())
    else if Arg = '--fps' then Opt.Fps := StrToInt(NextArg())
    else if Arg = '--refiner' then Opt.Refiner := NextArg()
    else if Arg = '--segmentation' then Opt.Segmentation := true
    else if Arg = '--gpu' then Opt.UseOpenCL := true
    else if Arg = '--gpu-platform' then Opt.OpenCLPlatform := StrToInt(NextArg())
    else if Arg = '--gpu-device' then Opt.OpenCLDevice := StrToInt(NextArg())
    else if (Arg = '--help') or (Arg = '-h') then
    begin
      PrintUsage();
      Result := false;
      exit;
    end
    else
      raise Exception.Create('unknown argument: ' + Arg);
    Inc(ArgPos);
  end;
  if (Opt.ModelFile <> '') and ((Opt.InputDir = '') or (Opt.OutputDir = '')) then
    raise Exception.Create('--model needs --input and --output');
  if (Opt.Ratio < 0) or (Opt.Ratio > 1) then
    raise Exception.Create('--ratio must be in (0, 1]');
  if Opt.Fps < 1 then
    raise Exception.Create('--fps must be positive');
  if (Opt.Refiner <> '') and (Opt.Refiner <> csRVMDeepGuidedFilter) and
     (Opt.Refiner <> csRVMFastGuidedFilter) then
    raise Exception.Create('--refiner must be ' + csRVMDeepGuidedFilter +
      ' or ' + csRVMFastGuidedFilter);
end;

// The RVM README "portrait" column: 1 up to 512x512, 0.375 at 1280x720, 0.25
// at 1920x1080, 0.125 at 4K, i.e. the longest side becomes about 480 px.
function DefaultRatio(FrameWidth, FrameHeight: integer): double;
var
  LongestSide: integer;
begin
  LongestSide := Max(FrameWidth, FrameHeight);
  if LongestSide <= 512 then
    Result := 1
  else
    Result := 480 / LongestSide;
end;

// Compares file names with digit runs compared by value ("f2" < "f10").
function NaturalCompare(List: TStringList; Index1, Index2: integer): integer;
var
  NameA, NameB: string;
  PosA, PosB, StartA, StartB, LenA, LenB: integer;
  RunA, RunB: string;
begin
  NameA := LowerCase(List[Index1]);
  NameB := LowerCase(List[Index2]);
  PosA := 1;
  PosB := 1;
  while (PosA <= Length(NameA)) and (PosB <= Length(NameB)) do
  begin
    if (NameA[PosA] in ['0'..'9']) and (NameB[PosB] in ['0'..'9']) then
    begin
      StartA := PosA;
      StartB := PosB;
      while (PosA <= Length(NameA)) and (NameA[PosA] in ['0'..'9']) do Inc(PosA);
      while (PosB <= Length(NameB)) and (NameB[PosB] in ['0'..'9']) do Inc(PosB);
      // Leading zeros do not count: compare the significant digits.
      while (StartA < PosA - 1) and (NameA[StartA] = '0') do Inc(StartA);
      while (StartB < PosB - 1) and (NameB[StartB] = '0') do Inc(StartB);
      LenA := PosA - StartA;
      LenB := PosB - StartB;
      if LenA <> LenB then exit(LenA - LenB);
      RunA := Copy(NameA, StartA, LenA);
      RunB := Copy(NameB, StartB, LenB);
      Result := CompareStr(RunA, RunB);
      if Result <> 0 then exit;
    end
    else
    begin
      if NameA[PosA] <> NameB[PosB] then
        exit(Ord(NameA[PosA]) - Ord(NameB[PosB]));
      Inc(PosA);
      Inc(PosB);
    end;
  end;
  Result := (Length(NameA) - PosA) - (Length(NameB) - PosB);
end;

// Fills Frames with the PNG / JPEG file names of Dir, in frame order.
procedure ListFrames(const Dir: string; Frames: TStringList);
var
  Search: TSearchRec;
  Extension: string;
begin
  Frames.Clear;
  if FindFirst(IncludeTrailingPathDelimiter(Dir) + '*', faAnyFile, Search) = 0 then
  try
    repeat
      if (Search.Attr and faDirectory) = 0 then
      begin
        Extension := LowerCase(ExtractFileExt(Search.Name));
        if (Extension = '.png') or (Extension = '.jpg') or
           (Extension = '.jpeg') then
          Frames.Add(Search.Name);
      end;
    until FindNext(Search) <> 0;
  finally
    FindClose(Search);
  end;
  Frames.CustomSort(@NaturalCompare);
end;

// Matte (W, H, 4) in [0,1] -> RGBA 0..255 and the green-screen RGB 0..255.
procedure ComposeOutputs(Matte, RGBA, Green: TNNetVolume);
var
  PixelPos, MaxPixelPos, Channel: integer;
  Alpha: TNeuralFloat;
  Src, Dst: TNeuralFloatArrPtr;
begin
  RGBA.Copy(Matte);
  RGBA.Mul(255);
  Green.ReSize(Matte.SizeX, Matte.SizeY, 3);
  MaxPixelPos := Matte.SizeX * Matte.SizeY - 1;
  Src := RGBA.DataPtr;
  Dst := Green.DataPtr;
  for PixelPos := 0 to MaxPixelPos do
  begin
    Alpha := Src^[3] * (1 / 255);
    for Channel := 0 to 2 do
      Dst^[Channel] := Src^[Channel] * Alpha +
        csGreenScreen[Channel] * (1 - Alpha);
    Src := TNeuralFloatArrPtr(@Src^[4]);
    Dst := TNeuralFloatArrPtr(@Dst^[3]);
  end;
end;

// Seg logits (w, h, 1) -> the foreground probability in [0, 1].
procedure ComposeSegmentation(Seg, Mask: TNNetVolume);
begin
  Mask.ReSize(Seg);
  TNNetVolume.Sigmoid(Mask.DataPtr, Seg.DataPtr, Seg.Size);
end;

// Mattes every frame of Opt.InputDir into Opt.OutputDir.
procedure MatteFolder(const Opt: TVideoMattingOptions);
var
  Matting: TRVMMatting;
  Config: TRVMConfig;
  Frames: TStringList;
  Image, Frame, Matte, RGBA, Green: TNNetVolume;
  GifWriter: TNNetAnimatedGIFWriter;
  FramePos, MaxFramePos, NetWidth, NetHeight, GifWidth, GifHeight: integer;
  Ratio, AlphaMin, AlphaMax, AlphaSum, TotalSeconds, FrameSeconds: double;
  PixelPos, MaxPixelPos: integer;
  Stem, ComputeText: string;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  OpenCLProblem: string;
  OpenCLPlatform, OpenCLDevice: integer;
  {$ENDIF}

  // Builds the net for the current Image size and wraps it in Matting.
  procedure BuildNet();
  begin
    FreeAndNil(Matting);
    NetWidth := Image.SizeX;
    NetHeight := Image.SizeY;
    if Opt.Ratio > 0 then
      Ratio := Opt.Ratio
    else
      Ratio := DefaultRatio(NetWidth, NetHeight);
    StartTime := Now();
    Matting := TRVMMatting.Create(BuildRVMFromSafeTensors(Opt.ModelFile,
      NetWidth, NetHeight, Ratio, Config, {pTrainable=}false, Opt.Refiner,
      Opt.Segmentation));
    WriteLn(RVMConfigToString(Config));
    WriteLn(Format('  built in %.1f s', [(Now() - StartTime) * 86400.0]));
    {$IFDEF OpenCL}
    if OpenCLDevices <> nil then
      Matting.Net.EnableOpenCL(OpenCLDevices.PlatformIds[OpenCLPlatform],
        OpenCLDevices.Devices[OpenCLDevice]);
    {$ENDIF}
    Matting.ResetState();
  end;

begin
  Frames := TStringList.Create;
  Image := TNNetVolume.Create;
  Frame := TNNetVolume.Create;
  Matte := TNNetVolume.Create;
  RGBA := TNNetVolume.Create;
  Green := TNNetVolume.Create;
  Matting := nil;
  GifWriter := nil;
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
    ListFrames(Opt.InputDir, Frames);
    if Frames.Count = 0 then
      raise Exception.Create('no PNG / JPEG frames in ' + Opt.InputDir);
    ForceDirectories(Opt.OutputDir);
    ComputeText := 'CPU';
    if Opt.UseOpenCL then
    begin
      {$IFDEF OpenCL}
      OpenCLDevices := TEasyOpenCL.Create();
      OpenCLPlatform := Opt.OpenCLPlatform;
      OpenCLDevice := Opt.OpenCLDevice;
      if OpenCLDevices.SelectPlatformAndDevice(OpenCLPlatform, OpenCLDevice,
        OpenCLProblem) then
        ComputeText := 'OpenCL on ' +
          OpenCLDevices.PlatformNames[OpenCLPlatform] + ' / ' +
          OpenCLDevices.DeviceNames[OpenCLDevice]
      else
      begin
        ComputeText := 'CPU (' + OpenCLProblem + ')';
        FreeAndNil(OpenCLDevices);
      end;
      {$ELSE}
      ComputeText := 'CPU (built without -dOpenCL)';
      {$ENDIF}
    end;
    WriteLn(Format('Matting %d frames from %s on %s', [Frames.Count,
      Opt.InputDir, ComputeText]));
    TotalSeconds := 0;
    MaxFramePos := Frames.Count - 1;
    for FramePos := 0 to MaxFramePos do
    begin
      if not LoadImageFromFileIntoVolume(IncludeTrailingPathDelimiter(
        Opt.InputDir) + Frames[FramePos], Image) then
        raise Exception.Create('could not load ' + Frames[FramePos]);
      if Matting = nil then
        BuildNet()
      else if (Image.SizeX <> NetWidth) or (Image.SizeY <> NetHeight) then
      begin
        WriteLn(Format('Notice: %s is %dx%d, not %dx%d: rebuilding the net; ' +
          'the recurrent state starts over.', [Frames[FramePos], Image.SizeX,
          Image.SizeY, NetWidth, NetHeight]));
        BuildNet();
      end;
      Frame.Copy(Image);
      Frame.Mul(1 / 255);
      StartTime := Now();
      Matting.Step(Frame, Matte);
      FrameSeconds := (Now() - StartTime) * 86400.0;
      TotalSeconds := TotalSeconds + FrameSeconds;
      Stem := IncludeTrailingPathDelimiter(Opt.OutputDir) +
        ChangeFileExt(Frames[FramePos], '');
      if Opt.Segmentation then
      begin
        ComposeSegmentation(Matte, Green);
        if not SaveGray16PNGFromVolume(Green, Stem + '_seg.png') then
          raise Exception.Create('could not save ' + Stem + '_seg.png');
        WriteLn(Format('  %s %dx%d ratio %.4g: %.2f s, seg %dx%d, mean ' +
          'probability %.3f', [Frames[FramePos], Image.SizeX, Image.SizeY,
          Ratio, FrameSeconds, Green.SizeX, Green.SizeY,
          Green.GetAvg()]));
        continue;
      end;
      ComposeOutputs(Matte, RGBA, Green);
      if not SaveImageFromVolumeIntoFile(RGBA, Stem + '_rgba.png') then
        raise Exception.Create('could not save ' + Stem + '_rgba.png');
      if not SaveImageFromVolumeIntoFile(Green, Stem + '_green.png') then
        raise Exception.Create('could not save ' + Stem + '_green.png');
      AlphaMin := 1;
      AlphaMax := 0;
      AlphaSum := 0;
      MaxPixelPos := Matte.SizeX * Matte.SizeY - 1;
      for PixelPos := 0 to MaxPixelPos do
      begin
        AlphaMin := Min(AlphaMin, Matte.FData[PixelPos * 4 + 3]);
        AlphaMax := Max(AlphaMax, Matte.FData[PixelPos * 4 + 3]);
        AlphaSum := AlphaSum + Matte.FData[PixelPos * 4 + 3];
      end;
      WriteLn(Format('  %s %dx%d ratio %.4g: %.2f s, alpha min %.3f max %.3f ' +
        'mean %.3f', [Frames[FramePos], Image.SizeX, Image.SizeY, Ratio,
        FrameSeconds, AlphaMin, AlphaMax, AlphaSum / (MaxPixelPos + 1)]));
      if Opt.GifFile <> '' then
      begin
        if GifWriter = nil then
        begin
          GifWidth := Green.SizeX;
          GifHeight := Green.SizeY;
          GifWriter := TNNetAnimatedGIFWriter.Create(Opt.GifFile, GifWidth,
            GifHeight, Max(1, Round(100 / Opt.Fps)));
        end;
        // A GIF has one canvas size: frames of another size are left out.
        if (Green.SizeX = GifWidth) and (Green.SizeY = GifHeight) then
          GifWriter.AddFrame(Green, 0, 255)
        else
          WriteLn('  (', Frames[FramePos], ' left out of the GIF: its canvas ',
            'is ', GifWidth, 'x', GifHeight, ')');
      end;
    end;
    WriteLn(Format('Done: %d frames, %.2f s of net time (%.2f s/frame) -> %s',
      [Frames.Count, TotalSeconds, TotalSeconds / Frames.Count, Opt.OutputDir]));
    if GifWriter <> nil then
    begin
      GifWriter.Finish;
      WriteLn(Format('  %d-frame GIF -> %s', [GifWriter.FrameCount, Opt.GifFile]));
    end;
  finally
    GifWriter.Free;
    Matting.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    Green.Free;
    RGBA.Free;
    Matte.Free;
    Frame.Free;
    Image.Free;
    Frames.Free;
  end;
end;

// Writes Count SizeX x SizeY frames of a bright disc moving over a gradient.
procedure MakeSyntheticFrames(const Dir: string; FirstIndex, Count, SizeX,
  SizeY: integer);
var
  Img: TNNetVolume;
  FramePos, X, Y, CenterX, CenterY, Radius: integer;
begin
  ForceDirectories(Dir);
  Img := TNNetVolume.Create(SizeX, SizeY, 3);
  try
    Radius := Min(SizeX, SizeY) div 4;
    for FramePos := 0 to Count - 1 do
    begin
      CenterX := Radius + (FramePos * (SizeX - 2 * Radius)) div Max(1, Count);
      CenterY := SizeY div 2;
      for Y := 0 to SizeY - 1 do
        for X := 0 to SizeX - 1 do
          if Sqr(X - CenterX) + Sqr(Y - CenterY) <= Sqr(Radius) then
          begin
            Img[X, Y, 0] := 230;
            Img[X, Y, 1] := 190;
            Img[X, Y, 2] := 160;
          end
          else
          begin
            Img[X, Y, 0] := (X * 255) div (SizeX - 1);
            Img[X, Y, 1] := 60;
            Img[X, Y, 2] := (Y * 255) div (SizeY - 1);
          end;
      if not SaveImageFromVolumeIntoFile(Img, IncludeTrailingPathDelimiter(Dir) +
        'frame_' + IntToStr(FirstIndex + FramePos) + '.png') then
        raise Exception.Create('could not save a frame in ' + Dir);
    end;
  finally
    Img.Free;
  end;
end;

// Pico smoke: three 44x36 frames, then one 40x32 frame (rebuild notice).
procedure RunFixtureSmoke();
var
  Opt: TVideoMattingOptions;
begin
  MakeSyntheticFrames('rvm_smoke_frames', 1, 3, 44, 36);
  MakeSyntheticFrames('rvm_smoke_frames', 4, 1, 40, 32);
  ParseOptions(Opt);
  Opt.ModelFile := FixturesDir + 'tiny_rvm_mobilenetv3.pth';
  Opt.InputDir := 'rvm_smoke_frames';
  Opt.OutputDir := 'rvm_smoke_out';
  Opt.GifFile := 'rvm_smoke.gif';
  // Below 1 so the smoke also runs the Deep Guided Filter refiner.
  Opt.Ratio := 0.5;
  MatteFolder(Opt);
  WriteLn('Fast Guided Filter refiner, then the segmentation pass:');
  Opt.Refiner := csRVMFastGuidedFilter;
  Opt.OutputDir := 'rvm_smoke_out_fgf';
  Opt.GifFile := '';
  MatteFolder(Opt);
  Opt.Segmentation := true;
  Opt.OutputDir := 'rvm_smoke_out_seg';
  MatteFolder(Opt);
end;

var
  Options: TVideoMattingOptions;
begin
  try
    if not ParseOptions(Options) then exit;
    if Options.ModelFile <> '' then
      MatteFolder(Options)
    else
    begin
      WriteLn('Robust Video Matting smoke (random pico weights)');
      WriteLn('================================================');
      RunFixtureSmoke();
    end;
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
