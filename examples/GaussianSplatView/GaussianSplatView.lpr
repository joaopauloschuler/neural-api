program GaussianSplatView;
(*
GaussianSplatView: renders a trained 3D Gaussian Splatting scene (Kerbl et
al., SIGGRAPH 2023, arXiv:2308.04079) from a camera orbiting the scene
centre. TGaussianSplatScene (neuralgaussiansplat.pas) loads the INRIA
point_cloud.ply and TGaussianSplatRasterizer renders each view with the
reference tile rasterizer; --gpu runs the preprocess and blend kernels of
TGaussianSplatRasterizerCL in OpenCL.

The program writes frame_000.png, frame_001.png, ... and orbit.gif into the
output folder. --demo renders a random scene, so a run needs no download.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, fpjson, jsonparser,
  {$IFDEF OpenCL}neuralopencl,{$ENDIF}
  neuralvolume, neuraldatasets, neuralgif, neuralgaussiansplat;

const
  csDemoGaussianCount = 600;
  csDemoSHDegree = 3;
  csGifDelayCentiseconds = 8;
  // Without --radius and --cameras the orbit radius is this many median distances
  // of the Gaussians from the orbit target.
  csOrbitRadiusPerMedianDistance = 3;
  // Gaussians below this opacity do not vote for the orbit target.
  csTargetMinOpacity = 0.1;

type
  TGaussianSplatViewOptions = record
    PlyFile, CamerasFile, OutputDir: string;
    UseDemo, UseOpenCL: boolean;
    FrameCount, Width, Height: integer;
    FovDegrees, OrbitRadius: TNeuralFloat;
  end;

  TVector3 = array[0..2] of TNeuralFloat;

procedure PrintUsage();
begin
  WriteLn('Usage: GaussianSplatView (--ply FILE | --demo) [--orbit N] [--size WxH]');
  WriteLn('         [--fov DEG] [--cameras cameras.json] [--radius R] [--gpu] [--out DIR]');
  WriteLn('  --ply FILE      INRIA 3DGS point_cloud.ply');
  WriteLn('  --demo          render a random scene of ', csDemoGaussianCount, ' Gaussians instead');
  WriteLn('  --orbit N       frames in one orbit around the orbit target (default 36)');
  WriteLn('  --size WxH      image size (default 320x240)');
  WriteLn('  --fov DEG       horizontal field of view (default 60)');
  WriteLn('  --cameras FILE  INRIA cameras.json: camera 0 gives the focal lengths,');
  WriteLn('                  the up axis and the orbit start; the orbit target is');
  WriteLn('                  the point nearest to all camera view axes');
  WriteLn('  --radius R      orbit radius in scene units');
  WriteLn('  --gpu           preprocess and blend in OpenCL');
  WriteLn('  --out DIR       output folder (default gaussian_splat_frames)');
end;

function ParseOptions(out Opt: TGaussianSplatViewOptions): boolean;
var
  ArgPos, SeparatorPos: integer;
  Arg, SizeText: string;

  function NextArg(): string;
  begin
    Inc(ArgPos);
    if ArgPos > ParamCount then
      raise Exception.Create(Arg + ' needs a value');
    Result := ParamStr(ArgPos);
  end;

begin
  Opt.PlyFile := '';
  Opt.CamerasFile := '';
  Opt.OutputDir := 'gaussian_splat_frames';
  Opt.UseDemo := false;
  Opt.UseOpenCL := false;
  Opt.FrameCount := 36;
  Opt.Width := 320;
  Opt.Height := 240;
  Opt.FovDegrees := 60;
  Opt.OrbitRadius := 0;
  Result := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--ply' then Opt.PlyFile := NextArg()
    else if Arg = '--demo' then Opt.UseDemo := true
    else if Arg = '--orbit' then Opt.FrameCount := StrToInt(NextArg())
    else if Arg = '--size' then
    begin
      SizeText := LowerCase(NextArg());
      SeparatorPos := Pos('x', SizeText);
      if SeparatorPos = 0 then raise Exception.Create('--size needs WxH, e.g. 320x240');
      Opt.Width := StrToInt(Copy(SizeText, 1, SeparatorPos - 1));
      Opt.Height := StrToInt(Copy(SizeText, SeparatorPos + 1, MaxInt));
    end
    else if Arg = '--fov' then Opt.FovDegrees := StrToFloat(NextArg())
    else if Arg = '--cameras' then Opt.CamerasFile := NextArg()
    else if Arg = '--radius' then Opt.OrbitRadius := StrToFloat(NextArg())
    else if Arg = '--gpu' then Opt.UseOpenCL := true
    else if Arg = '--out' then Opt.OutputDir := NextArg()
    else if (Arg = '--help') or (Arg = '-h') then
    begin
      PrintUsage();
      Exit(false);
    end
    else
      raise Exception.Create('unknown argument: ' + Arg);
    Inc(ArgPos);
  end;
  if (Opt.PlyFile = '') = (not Opt.UseDemo) then
  begin
    PrintUsage();
    Exit(false);
  end;
  if Opt.FrameCount < 1 then raise Exception.Create('--orbit must be positive');
  if (Opt.Width < 1) or (Opt.Height < 1) then raise Exception.Create('--size must be positive');
  if (Opt.FovDegrees <= 0) or (Opt.FovDegrees >= 180) then
    raise Exception.Create('--fov must be in (0, 180)');
end;

// Reads an INRIA cameras.json: camera 0 gives the focal lengths scaled to the output
// size, Eye and Up (minus its y axis); every camera adds its centre and z axis to the rays.
procedure LoadInriaCameras(const FileName: string; var Camera: TGaussianSplatCamera;
  out Eye, Up: TVector3; out RayOrigins, RayDirections: TNeuralFloatDynArr;
  out RayCount: integer);
var
  FileStream: TFileStream;
  Root: TJSONData;
  Entry: TJSONObject;
  Position, Rotation: TJSONArray;
  AxisPos, CameraPos, MaxCameraPos: integer;
begin
  FileStream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    Root := GetJSON(FileStream);
  finally
    FileStream.Free;
  end;
  try
    if (not (Root is TJSONArray)) or (TJSONArray(Root).Count = 0) then
      raise Exception.Create(FileName + ' is not a non-empty JSON array of cameras');
    RayCount := TJSONArray(Root).Count;
    SetLength(RayOrigins, 3 * RayCount);
    SetLength(RayDirections, 3 * RayCount);
    MaxCameraPos := RayCount - 1;
    // "rotation" is camera-to-world: its columns are the camera axes in world space.
    for CameraPos := 0 to MaxCameraPos do
    begin
      Entry := TJSONArray(Root).Objects[CameraPos];
      Position := Entry.Arrays['position'];
      Rotation := Entry.Arrays['rotation'];
      for AxisPos := 0 to 2 do
      begin
        RayOrigins[3 * CameraPos + AxisPos] := Position.Floats[AxisPos];
        RayDirections[3 * CameraPos + AxisPos] := Rotation.Arrays[AxisPos].Floats[2];
      end;
    end;
    Entry := TJSONArray(Root).Objects[0];
    Camera.FocalX := Entry.Floats['fx'] * Camera.Width / Entry.Integers['width'];
    Camera.FocalY := Entry.Floats['fy'] * Camera.Height / Entry.Integers['height'];
    Position := Entry.Arrays['position'];
    Rotation := Entry.Arrays['rotation'];
    for AxisPos := 0 to 2 do
    begin
      Eye[AxisPos] := Position.Floats[AxisPos];
      Up[AxisPos] := -Rotation.Arrays[AxisPos].Floats[1];
    end;
    WriteLn('Camera 0 of ', FileName, ': ', Entry.Get('img_name', '?'),
      Format(', fx %.1f fy %.1f at %dx%d', [Entry.Floats['fx'], Entry.Floats['fy'],
      Entry.Integers['width'], Entry.Integers['height']]));
  finally
    Root.Free;
  end;
end;

// Offset turned by Angle radians about the unit axis Axis (Rodrigues).
procedure RotateAboutAxis(const Offset, Axis: TVector3; Angle: TNeuralFloat;
  out Rotated: TVector3);
var
  CosAngle, SinAngle, AxisDot: TNeuralFloat;
  Cross: TVector3;
  AxisPos: integer;
begin
  CosAngle := Cos(Angle);
  SinAngle := Sin(Angle);
  AxisDot := Axis[0] * Offset[0] + Axis[1] * Offset[1] + Axis[2] * Offset[2];
  Cross[0] := Axis[1] * Offset[2] - Axis[2] * Offset[1];
  Cross[1] := Axis[2] * Offset[0] - Axis[0] * Offset[2];
  Cross[2] := Axis[0] * Offset[1] - Axis[1] * Offset[0];
  for AxisPos := 0 to 2 do
    Rotated[AxisPos] := Offset[AxisPos] * CosAngle + Cross[AxisPos] * SinAngle +
      Axis[AxisPos] * AxisDot * (1 - CosAngle);
end;

// Returns the OpenCL rasterizer for --gpu, or the CPU one; Where names it for the log.
function CreateRasterizer(const Opt: TGaussianSplatViewOptions;
  out Where: string): TGaussianSplatRasterizer;
{$IFDEF OpenCL}
var
  Devices: TEasyOpenCL;
  OpenCLProblem: string;
  PlatformPos, DevicePos: integer;
{$ENDIF}
begin
  Result := nil;
  Where := 'CPU';
  if Opt.UseOpenCL then
  begin
    {$IFDEF OpenCL}
    Devices := TEasyOpenCL.Create();
    try
      PlatformPos := 0;
      DevicePos := 0;
      if Devices.SelectPlatformAndDevice(PlatformPos, DevicePos, OpenCLProblem) then
      begin
        Result := TGaussianSplatRasterizerCL.Create(Devices.PlatformIds[PlatformPos],
          Devices.Devices[DevicePos]);
        Where := 'OpenCL on ' + Devices.PlatformNames[PlatformPos] + ' / ' +
          Devices.DeviceNames[DevicePos];
      end
      else
        Where := 'CPU (' + OpenCLProblem + ')';
    finally
      Devices.Free;
    end;
    {$ELSE}
    Where := 'CPU (built without -dOpenCL)';
    {$ENDIF}
  end;
  if Result = nil then Result := TGaussianSplatRasterizer.Create;
end;

procedure RenderOrbit(const Opt: TGaussianSplatViewOptions);
var
  Scene: TGaussianSplatScene;
  Rasterizer: TGaussianSplatRasterizer;
  Camera: TGaussianSplatCamera;
  Rendered, Frame: TNNetVolume;
  GifWriter: TNNetAnimatedGIFWriter;
  Target, RayTarget, Eye, Up, Offset, Rotated: TVector3;
  MedianDistance, OffsetLength, UpLength: TNeuralFloat;
  RayOrigins, RayDirections: TNeuralFloatDynArr;
  RayCount: integer;
  TargetSource: string;
  StartTime, FrameSeconds, TotalSeconds: double;
  FramePos, MaxFramePos, AxisPos: integer;
  Where, FrameFile, GifFile: string;
begin
  Scene := TGaussianSplatScene.Create;
  Rendered := TNNetVolume.Create;
  Frame := TNNetVolume.Create;
  GifWriter := nil;
  Rasterizer := nil;
  try
    StartTime := Now();
    if Opt.UseDemo then
    begin
      RandSeed := 2023;
      Scene.FillRandom(csDemoGaussianCount, csDemoSHDegree, 1);
      WriteLn('Demo scene: ', Scene.Count, ' random Gaussians, SH degree ', Scene.SHDegree);
    end
    else
    begin
      Scene.LoadFromPLY(Opt.PlyFile);
      WriteLn(Opt.PlyFile, ': ', Scene.Count, ' Gaussians, SH degree ', Scene.SHDegree,
        Format(', loaded in %.2f s', [(Now() - StartTime) * 86400]));
    end;
    if Scene.Count = 0 then raise Exception.Create('the scene has no Gaussians');
    Scene.ComputeRobustCentre(csTargetMinOpacity, Target[0], Target[1], Target[2],
      MedianDistance);
    TargetSource := 'opacity-weighted median of the means';
    InitGaussianSplatCamera(Camera, Opt.Width, Opt.Height, DegToRad(Opt.FovDegrees));
    // COLMAP axes put image +y down, so world -y is the default up axis.
    Up[0] := 0; Up[1] := -1; Up[2] := 0;
    Eye[0] := Target[0]; Eye[1] := Target[1];
    Eye[2] := Target[2] - csOrbitRadiusPerMedianDistance * MedianDistance;
    if Opt.CamerasFile <> '' then
    begin
      LoadInriaCameras(Opt.CamerasFile, Camera, Eye, Up, RayOrigins, RayDirections, RayCount);
      if GaussianSplatNearestPointToRays(RayOrigins, RayDirections, RayCount,
        RayTarget[0], RayTarget[1], RayTarget[2]) then
      begin
        Target := RayTarget;
        TargetSource := Format('point nearest to the view axes of %d cameras', [RayCount]);
      end;
    end;
    UpLength := Sqrt(Sqr(Up[0]) + Sqr(Up[1]) + Sqr(Up[2]));
    for AxisPos := 0 to 2 do
    begin
      Up[AxisPos] := Up[AxisPos] / UpLength;
      Offset[AxisPos] := Eye[AxisPos] - Target[AxisPos];
    end;
    OffsetLength := Sqrt(Sqr(Offset[0]) + Sqr(Offset[1]) + Sqr(Offset[2]));
    if (Opt.OrbitRadius > 0) and (OffsetLength > 0) then
      for AxisPos := 0 to 2 do
        Offset[AxisPos] := Offset[AxisPos] * Opt.OrbitRadius / OffsetLength;
    WriteLn(Format('Orbit target (%.3f, %.3f, %.3f) = %s; median distance %.3f, orbit radius %.3f',
      [Target[0], Target[1], Target[2], TargetSource, MedianDistance,
      Sqrt(Sqr(Offset[0]) + Sqr(Offset[1]) + Sqr(Offset[2]))]));
    Rasterizer := CreateRasterizer(Opt, Where);
    WriteLn('Rendering ', Opt.FrameCount, ' frames of ', Opt.Width, 'x', Opt.Height,
      ' on ', Where);
    ForceDirectories(Opt.OutputDir);
    GifFile := IncludeTrailingPathDelimiter(Opt.OutputDir) + 'orbit.gif';
    GifWriter := TNNetAnimatedGIFWriter.Create(GifFile, Opt.Width, Opt.Height,
      csGifDelayCentiseconds);
    TotalSeconds := 0;
    MaxFramePos := Opt.FrameCount - 1;
    for FramePos := 0 to MaxFramePos do
    begin
      RotateAboutAxis(Offset, Up, 2 * Pi * FramePos / Opt.FrameCount, Rotated);
      SetGaussianSplatCameraLookAt(Camera, Target[0] + Rotated[0],
        Target[1] + Rotated[1], Target[2] + Rotated[2],
        Target[0], Target[1], Target[2], Up[0], Up[1], Up[2]);
      StartTime := Now();
      Rasterizer.Render(Scene, Camera, 0, 0, 0, Rendered);
      FrameSeconds := (Now() - StartTime) * 86400;
      TotalSeconds := TotalSeconds + FrameSeconds;
      GifWriter.AddFrame(Rendered, 0, 1);
      Frame.Copy(Rendered);
      Frame.Mul(255);
      FrameFile := IncludeTrailingPathDelimiter(Opt.OutputDir) +
        Format('frame_%.3d.png', [FramePos]);
      if not SaveImageFromVolumeIntoFile(Frame, FrameFile) then
        raise Exception.Create('could not write ' + FrameFile);
      WriteLn(Format('  frame %d: %d visible Gaussians, %d tile entries, %.3f s -> %s',
        [FramePos, Rasterizer.VisibleCount, Rasterizer.TileListLength, FrameSeconds,
        FrameFile]));
    end;
    GifWriter.Finish;
    WriteLn(Format('Wrote %s (%d frames); %.3f s per frame on average.',
      [GifFile, GifWriter.FrameCount, TotalSeconds / Opt.FrameCount]));
  finally
    GifWriter.Free;
    Rasterizer.Free;
    Frame.Free;
    Rendered.Free;
    Scene.Free;
  end;
end;

var
  Options: TGaussianSplatViewOptions;
begin
  try
    if ParseOptions(Options) then RenderOrbit(Options);
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
