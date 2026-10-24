unit TestNeuralGaussianSplat;
(*
Tests for neuralply.pas (PLY reader/writer) and neuralgaussiansplat.pas (3D
Gaussian Splatting scene + CPU tile rasterizer). Every expected value is
derived by hand from the 3DGS formulas in the test body.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralvolume, neuralply, neuralgaussiansplat
  {$IFDEF OpenCL}, cl, neuralopencl{$ENDIF};

type
  TTestNeuralGaussianSplat = class(TTestCase)
  published
    // Typed scalar properties survive ascii, little- and big-endian bodies.
    procedure TestPLYRoundTripAllFormats;
    // A face list after the vertices is skipped; its column holds the list length.
    procedure TestPLYReadsListProperty;
    // Activated values and channel-major f_rest survive SaveToPLY / LoadFromStream.
    procedure TestSceneRoundTripThroughPLY;
    // One isotropic Gaussian at z = 5: peak pixel, Gaussian falloff, 3-sigma radius.
    procedure TestIsotropicGaussianPeakAndFootprint;
    // Degree-0 colour, the clamp at 0 and a degree-1 view-dependent term.
    procedure TestSHColour;
    // Two Gaussians on the optical axis blend nearest first whatever the file order.
    procedure TestTwoGaussiansFrontToBack;
    // A quaternion turns an x-elongated Gaussian onto y and onto the diagonal.
    procedure TestQuaternionRotatedAnisotropicGaussian;
    procedure TestLookAtCamera;
    // TGaussianSplatRasterizerCL matches the CPU rasterizer on a random degree-3
    // scene from two cameras, and renders an empty scene as background.
    procedure TestOpenCLMatchesCPU;
  end;

implementation

const
  PixelTolerance = 1e-5;

procedure InitTestCamera(out Camera: TGaussianSplatCamera);
begin
  InitGaussianSplatCamera(Camera, 33, 33, Pi / 2);
  Camera.FocalX := 100;
  Camera.FocalY := 100;
  Camera.PrincipalX := 16;
  Camera.PrincipalY := 16;
end;

function PixelValue(Output: TNNetVolume; X, Y, Channel: integer): TNeuralFloat;
begin
  Result := Output.FData[Output.GetRawPos(X, Y, Channel)];
end;

procedure TTestNeuralGaussianSplat.TestPLYRoundTripAllFormats;
const
  RowCount = 3;
  ColumnCount = 7;
  Expected: array[0..RowCount - 1, 0..ColumnCount - 1] of TNeuralFloat = (
    (1.5, -0.25, 200, -70000, -300, -100, 65000),
    (-3.125e-7, 12345.5, 0, 2147483, 32767, 127, 0),
    (0, 1e10, 255, 0, -32768, -128, 1));
  ValueTypes: array[0..ColumnCount - 1] of TNNetPLYValueType = (plyFloat32,
    plyFloat64, plyUInt8, plyInt32, plyInt16, plyInt8, plyUInt16);
var
  PLYFormat: TNNetPLYFormat;
  Stream: TMemoryStream;
  Writer: TNNetPLYWriter;
  Reader: TNNetPLYReader;
  Row: array[0..ColumnCount - 1] of TNeuralFloat;
  RowPos, ColumnPos: integer;
begin
  for PLYFormat := Low(TNNetPLYFormat) to High(TNNetPLYFormat) do
  begin
    Stream := TMemoryStream.Create;
    try
      Writer := TNNetPLYWriter.CreateOnStream(Stream, PLYFormat);
      Writer.AddComment('round trip');
      Writer.AddElement('vertex', RowCount);
      for ColumnPos := 0 to ColumnCount - 1 do
        Writer.AddProperty('p' + IntToStr(ColumnPos), ValueTypes[ColumnPos]);
      Writer.WriteHeader;
      for RowPos := 0 to RowCount - 1 do
      begin
        for ColumnPos := 0 to ColumnCount - 1 do Row[ColumnPos] := Expected[RowPos, ColumnPos];
        Writer.WriteRow(Row);
      end;
      Writer.Free;
      Stream.Position := 0;
      Reader := TNNetPLYReader.CreateOnStream(Stream);
      try
        AssertTrue('format', Reader.Format = PLYFormat);
        AssertEquals('comment', 'round trip', Reader.Comments[0]);
        AssertEquals('element count', 1, Length(Reader.Elements));
        AssertEquals('row count', RowCount, Reader.Elements[0].Count);
        AssertEquals('p3 index', 3, Reader.PropertyIndex(0, 'p3'));
        for ColumnPos := 0 to ColumnCount - 1 do
          AssertTrue('type', Reader.Elements[0].Properties[ColumnPos].ValueType = ValueTypes[ColumnPos]);
        for RowPos := 0 to RowCount - 1 do
        begin
          Reader.ReadRow(Row);
          for ColumnPos := 0 to ColumnCount - 1 do
            AssertEquals(Format('format %d row %d column %d', [Ord(PLYFormat), RowPos, ColumnPos]),
              Expected[RowPos, ColumnPos], Row[ColumnPos],
              Abs(Expected[RowPos, ColumnPos]) * 1e-7);
        end;
        AssertEquals('all rows read', 1, Reader.CurrentElement);
      finally
        Reader.Free;
      end;
    finally
      Stream.Free;
    end;
  end;
end;

procedure TTestNeuralGaussianSplat.TestPLYReadsListProperty;
const
  Text = 'ply'#13#10'format ascii 1.0'#10'comment made by hand'#10'obj_info x'#10 +
    'element vertex 2'#10'property float x'#10'property uint8 red'#10 +
    'element face 1'#10'property list uchar int vertex_indices'#10 +
    'property float quality'#10'end_header'#10 +
    '0.5 10'#10'-2 255'#10'3 0 1 2 0.75'#10;
var
  Stream: TStringStream;
  Reader: TNNetPLYReader;
  Row: array[0..1] of TNeuralFloat;
begin
  Stream := TStringStream.Create(Text);
  Reader := TNNetPLYReader.CreateOnStream(Stream);
  try
    AssertEquals('elements', 2, Length(Reader.Elements));
    AssertTrue('list flag', Reader.Elements[1].Properties[0].IsList);
    AssertEquals('face element', 1, Reader.FindElement('face'));
    Reader.SkipToElement(1);
    Reader.ReadRow(Row);
    AssertEquals('list length', 3, Row[0], 0);
    AssertEquals('scalar after list', 0.75, Row[1], 0);
    AssertEquals('finished', 2, Reader.CurrentElement);
  finally
    Reader.Free;
    Stream.Free;
  end;
end;

procedure TTestNeuralGaussianSplat.TestSceneRoundTripThroughPLY;
var
  Scene, Loaded: TGaussianSplatScene;
  Stream: TMemoryStream;
  Reader: TNNetPLYReader;
  Row: TNeuralFloatDynArr;
  ValuePos, MaxValuePos: integer;
begin
  Scene := TGaussianSplatScene.Create;
  Loaded := TGaussianSplatScene.Create;
  Stream := TMemoryStream.Create;
  try
    Scene.Resize(2, 3);
    Scene.SetGaussian(0, 1, 2, 3, 0.1, 0.2, 0.3, 1, 2, 3, 4, 0.7, 0.9, 0.1, 0.4);
    Scene.SetGaussian(1, -1, 0, 5, 0.01, 1.5, 0.02, 0, 0, 0, 1, 0.05, 0.2, 0.3, 0.6);
    MaxValuePos := 3 * Scene.SHCoefCount - 1;
    for ValuePos := 3 to MaxValuePos do
      Scene.SHCoefs[3 * Scene.SHCoefCount + ValuePos] := 0.01 * ValuePos;
    // Gaussian 1, coefficient 2, green: channel-major f_rest index 15 + (2 - 1).
    Scene.SHCoefs[3 * Scene.SHCoefCount + 3 * 2 + 1] := 7;
    Scene.SaveToStream(Stream);
    Stream.Position := 0;
    Reader := TNNetPLYReader.CreateOnStream(Stream);
    try
      SetLength(Row, Length(Reader.Elements[0].Properties));
      Reader.ReadRow(Row);
      Reader.ReadRow(Row);
      AssertEquals('f_rest_16', 7, Row[Reader.PropertyIndex(0, 'f_rest_16')], 0);
      AssertEquals('raw opacity is a logit', Ln(0.05 / 0.95), Row[Reader.PropertyIndex(0, 'opacity')], 1e-5);
      AssertEquals('raw scale is a log', Ln(1.5), Row[Reader.PropertyIndex(0, 'scale_1')], 1e-6);
    finally
      Reader.Free;
    end;
    Stream.Position := 0;
    Loaded.LoadFromStream(Stream);
    AssertEquals('count', 2, Loaded.Count);
    AssertEquals('degree', 3, Loaded.SHDegree);
    for ValuePos := 0 to 5 do
    begin
      AssertEquals('position', Scene.Positions[ValuePos], Loaded.Positions[ValuePos], 0);
      AssertEquals('scale', Scene.Scales[ValuePos], Loaded.Scales[ValuePos], 1e-6 * Scene.Scales[ValuePos]);
    end;
    for ValuePos := 0 to 7 do
      AssertEquals('rotation', Scene.Rotations[ValuePos], Loaded.Rotations[ValuePos], 1e-7);
    AssertEquals('quaternion normalized', 1 / Sqrt(30), Loaded.Rotations[0], 1e-7);
    AssertEquals('opacity 0', 0.7, Loaded.Opacities[0], 1e-6);
    AssertEquals('opacity 1', 0.05, Loaded.Opacities[1], 1e-6);
    MaxValuePos := High(Scene.SHCoefs);
    for ValuePos := 0 to MaxValuePos do
      AssertEquals('sh', Scene.SHCoefs[ValuePos], Loaded.SHCoefs[ValuePos], 0);
  finally
    Stream.Free;
    Loaded.Free;
    Scene.Free;
  end;
end;

procedure TTestNeuralGaussianSplat.TestIsotropicGaussianPeakAndFootprint;
const
  Opacity = 0.5;
  // 3D sigma 0.1 at z = 5 with focal 100 gives a 2-pixel sigma: 4 + 0.3 dilation.
  PixelVariance = 4.3;
  Red = 0.8; Green = 0.4; Blue = 0.2;
  BackRed: TNeuralFloat = 0.1; BackGreen: TNeuralFloat = 0.2; BackBlue: TNeuralFloat = 0.3;
var
  Scene: TGaussianSplatScene;
  Rasterizer: TGaussianSplatRasterizer;
  Camera: TGaussianSplatCamera;
  Output: TNNetVolume;
  Alpha: TNeuralFloat;
  Offset: integer;
begin
  Scene := TGaussianSplatScene.Create;
  Rasterizer := TGaussianSplatRasterizer.Create;
  Output := TNNetVolume.Create;
  try
    InitTestCamera(Camera);
    Scene.Resize(1, 0);
    Scene.SetGaussian(0, 0, 0, 5, 0.1, 0.1, 0.1, 1, 0, 0, 0, Opacity, Red, Green, Blue);
    Rasterizer.Render(Scene, Camera, BackRed, BackGreen, BackBlue, Output);
    AssertEquals('width', 33, Output.SizeX);
    AssertEquals('height', 33, Output.SizeY);
    AssertEquals('depth', 3, Output.Depth);
    AssertEquals('mean x', 16, Rasterizer.MeansImage[0], 1e-5);
    AssertEquals('mean y', 16, Rasterizer.MeansImage[1], 1e-5);
    AssertEquals('depth', 5, Rasterizer.Depths[0], 1e-6);
    AssertEquals('conic a', 1 / PixelVariance, Rasterizer.ConicOpacity[0], 1e-6);
    AssertEquals('conic b', 0, Rasterizer.ConicOpacity[1], 1e-6);
    // Equal eigenvalues: lambda1 = 4.3 + sqrt(0.1); ceil(3 sqrt(4.616)) = 7.
    AssertEquals('radius', 7, Rasterizer.Radii[0]);
    AssertEquals('max tile x', 2, Rasterizer.TileRects[2]);
    AssertEquals('tile list length', 4, Rasterizer.TileListLength);
    AssertEquals('peak red', Opacity * Red + (1 - Opacity) * BackRed, PixelValue(Output, 16, 16, 0), PixelTolerance);
    AssertEquals('peak green', Opacity * Green + (1 - Opacity) * BackGreen, PixelValue(Output, 16, 16, 1), PixelTolerance);
    AssertEquals('peak blue', Opacity * Blue + (1 - Opacity) * BackBlue, PixelValue(Output, 16, 16, 2), PixelTolerance);
    // Offsets 1..6 keep alpha above 1/255; offset 7 drops below it.
    for Offset := 1 to 6 do
    begin
      Alpha := Opacity * Exp(-0.5 * Offset * Offset / PixelVariance);
      AssertEquals('falloff right', Alpha * Red + (1 - Alpha) * BackRed, PixelValue(Output, 16 + Offset, 16, 0), PixelTolerance);
      AssertEquals('falloff up', Alpha * Green + (1 - Alpha) * BackGreen, PixelValue(Output, 16, 16 - Offset, 1), PixelTolerance);
    end;
    AssertEquals('offset 7 is background', BackRed, PixelValue(Output, 23, 16, 0), 0);
    AssertEquals('diagonal 5,5 is background', BackBlue, PixelValue(Output, 21, 21, 2), 0);
    AssertEquals('unbinned tile is background', BackGreen, PixelValue(Output, 32, 32, 1), 0);
  finally
    Output.Free;
    Rasterizer.Free;
    Scene.Free;
  end;
end;

procedure TTestNeuralGaussianSplat.TestSHColour;
const
  DCRed = 0.3;
  DirZCoef = 0.4;
  DirXCoef = -0.2;
var
  Scene: TGaussianSplatScene;
  RGB: array[0..2] of TNeuralFloat;
begin
  Scene := TGaussianSplatScene.Create;
  try
    Scene.Resize(2, 0);
    Scene.SetGaussian(0, 0, 0, 5, 1, 1, 1, 1, 0, 0, 0, 0.5, 0.8, 0.4, 0.2);
    Scene.SHCoefs[3] := DCRed;
    Scene.SHCoefs[4] := -5;
    Scene.SHCoefs[5] := 0;
    Scene.ComputeColor(0, 0, 0, 0, RGB);
    AssertEquals('degree 0 from SetGaussian', 0.8, RGB[0], 1e-6);
    AssertEquals('degree 0 green', 0.4, RGB[1], 1e-6);
    Scene.ComputeColor(1, 0, 0, 0, RGB);
    AssertEquals('C0 * dc + 0.5', GaussianSplatSHC0 * DCRed + 0.5, RGB[0], 1e-7);
    AssertEquals('clamped at 0', 0, RGB[1], 0);
    AssertEquals('zero dc', 0.5, RGB[2], 0);
    // Degree 1 at (3, 0, 4) seen from the origin: direction (0.6, 0, 0.8).
    Scene.Resize(1, 1);
    Scene.Positions[0] := 3;
    Scene.Positions[2] := 4;
    Scene.SHCoefs[0] := DCRed;
    Scene.SHCoefs[3 * 1] := 9;
    Scene.SHCoefs[3 * 2] := DirZCoef;
    Scene.SHCoefs[3 * 3] := DirXCoef;
    Scene.ComputeColor(0, 0, 0, 0, RGB);
    AssertEquals('degree 1', GaussianSplatSHC0 * DCRed + 0.5 +
      GaussianSplatSHC1 * (0.8 * DirZCoef - 0.6 * DirXCoef), RGB[0], 1e-6);
    AssertEquals('degree 1 green', 0.5, RGB[1], 0);
  finally
    Scene.Free;
  end;
end;

procedure TTestNeuralGaussianSplat.TestTwoGaussiansFrontToBack;
var
  Scene: TGaussianSplatScene;
  Rasterizer: TGaussianSplatRasterizer;
  Camera: TGaussianSplatCamera;
  Output: TNNetVolume;
begin
  Scene := TGaussianSplatScene.Create;
  Rasterizer := TGaussianSplatRasterizer.Create;
  Output := TNNetVolume.Create;
  try
    InitTestCamera(Camera);
    Scene.Resize(2, 0);
    // Back blue Gaussian first in file order, front red one second.
    Scene.SetGaussian(0, 0, 0, 6, 0.1, 0.1, 0.1, 1, 0, 0, 0, 0.5, 0, 0, 1);
    Scene.SetGaussian(1, 0, 0, 4, 0.1, 0.1, 0.1, 1, 0, 0, 0, 0.5, 1, 0, 0);
    Rasterizer.Render(Scene, Camera, 0, 1, 0, Output);
    AssertEquals('nearest first in tile 0', 1, Rasterizer.TileGaussianList[Rasterizer.TileRanges[0]]);
    // Front alpha 0.5 at T = 1, back alpha 0.5 at T = 0.5, background at T = 0.25.
    AssertEquals('red', 0.5, PixelValue(Output, 16, 16, 0), PixelTolerance);
    AssertEquals('green', 0.25, PixelValue(Output, 16, 16, 1), PixelTolerance);
    AssertEquals('blue', 0.25, PixelValue(Output, 16, 16, 2), PixelTolerance);
    Scene.Positions[2] := 3;
    Rasterizer.Render(Scene, Camera, 0, 1, 0, Output);
    AssertEquals('swapped red', 0.25, PixelValue(Output, 16, 16, 0), PixelTolerance);
    AssertEquals('swapped blue', 0.5, PixelValue(Output, 16, 16, 2), PixelTolerance);
  finally
    Output.Free;
    Rasterizer.Free;
    Scene.Free;
  end;
end;

procedure TTestNeuralGaussianSplat.TestQuaternionRotatedAnisotropicGaussian;
var
  Scene: TGaussianSplatScene;
  Rasterizer: TGaussianSplatRasterizer;
  Camera: TGaussianSplatCamera;
  Output: TNNetVolume;
  Cov: array[0..5] of TNeuralFloat;
  ConicA, ConicB, ConicC, ConicDet, Power, Alpha: TNeuralFloat;
begin
  Scene := TGaussianSplatScene.Create;
  Rasterizer := TGaussianSplatRasterizer.Create;
  Output := TNNetVolume.Create;
  try
    InitTestCamera(Camera);
    Scene.Resize(1, 0);
    // 90 degrees about z sends the long x axis (sigma 0.3) onto y.
    Scene.SetGaussian(0, 0, 0, 5, 0.3, 0.05, 0.05, Cos(Pi / 4), 0, 0, Sin(Pi / 4), 0.5, 1, 1, 1);
    Scene.ComputeCovariance3D(0, Cov);
    AssertEquals('xx', 0.0025, Cov[0], 1e-7);
    AssertEquals('xy', 0, Cov[1], 1e-7);
    AssertEquals('yy', 0.09, Cov[3], 1e-7);
    AssertEquals('zz', 0.0025, Cov[5], 1e-7);
    Rasterizer.Render(Scene, Camera, 0, 0, 0, Output);
    // 2D covariance diag(1.3, 36.3): lambda1 = 36.3, ceil(3 sqrt(36.3)) = 19.
    AssertEquals('radius', 19, Rasterizer.Radii[0]);
    AssertEquals('conic a', 1 / 1.3, Rasterizer.ConicOpacity[0], 1e-5);
    AssertEquals('conic c', 1 / 36.3, Rasterizer.ConicOpacity[2], 1e-6);
    AssertEquals('along y', 0.5 * Exp(-0.5 * 25 / 36.3), PixelValue(Output, 16, 21, 0), PixelTolerance);
    AssertEquals('across x', 0, PixelValue(Output, 21, 16, 0), 0);
    // 45 degrees: xx = yy = (0.09 + 0.0025) / 2, xy = (0.09 - 0.0025) / 2.
    Scene.SetGaussian(0, 0, 0, 5, 0.3, 0.05, 0.05, Cos(Pi / 8), 0, 0, Sin(Pi / 8), 0.5, 1, 1, 1);
    Scene.ComputeCovariance3D(0, Cov);
    AssertEquals('xx 45', 0.04625, Cov[0], 1e-7);
    AssertEquals('xy 45', 0.04375, Cov[1], 1e-7);
    AssertEquals('yy 45', 0.04625, Cov[3], 1e-7);
    Rasterizer.Render(Scene, Camera, 0, 0, 0, Output);
    ConicA := Rasterizer.ConicOpacity[0];
    ConicB := Rasterizer.ConicOpacity[1];
    ConicC := Rasterizer.ConicOpacity[2];
    ConicDet := ConicA * ConicC - ConicB * ConicB;
    AssertEquals('cov2D a', 18.8, ConicC / ConicDet, 1e-3);
    AssertEquals('cov2D b', 17.5, -ConicB / ConicDet, 1e-3);
    AssertEquals('cov2D c', 18.8, ConicA / ConicDet, 1e-3);
    // Along the diagonal d = (-3, -3): power = -9 (18.8 - 17.5) / det.
    Power := -9 * (18.8 - 17.5) / (18.8 * 18.8 - 17.5 * 17.5);
    Alpha := 0.5 * Exp(Power);
    AssertEquals('diagonal', Alpha, PixelValue(Output, 19, 19, 1), 1e-4);
    AssertEquals('anti-diagonal', 0, PixelValue(Output, 19, 13, 1), 0);
  finally
    Output.Free;
    Rasterizer.Free;
    Scene.Free;
  end;
end;

procedure TTestNeuralGaussianSplat.TestLookAtCamera;
var
  Camera: TGaussianSplatCamera;
  X, Y, Z: TNeuralFloat;
  EntryPos: integer;
const
  Identity: array[0..11] of TNeuralFloat = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 5);
begin
  InitGaussianSplatCamera(Camera, 64, 48, Pi / 2);
  AssertEquals('focal from fov', 32, Camera.FocalX, 1e-4);
  AssertEquals('principal x', 31.5, Camera.PrincipalX, 0);
  SetGaussianSplatCameraLookAt(Camera, 0, 0, -5, 0, 0, 0, 0, -1, 0);
  for EntryPos := 0 to 11 do
    AssertEquals('identity look-at', Identity[EntryPos], Camera.WorldToView[EntryPos], 1e-6);
  GetGaussianSplatCameraPosition(Camera, X, Y, Z);
  AssertEquals('camera z', -5, Z, 1e-6);
  // From +x towards the origin: world +z is image right, world +y image down.
  SetGaussianSplatCameraLookAt(Camera, 5, 0, 0, 0, 0, 0, 0, -1, 0);
  AssertEquals('right is +z', 1, Camera.WorldToView[2], 1e-6);
  AssertEquals('down is +y', 1, Camera.WorldToView[5], 1e-6);
  AssertEquals('forward is -x', -1, Camera.WorldToView[8], 1e-6);
  AssertEquals('origin depth', 5, Camera.WorldToView[11], 1e-6);
  GetGaussianSplatCameraPosition(Camera, X, Y, Z);
  AssertEquals('camera x', 5, X, 1e-6);
end;


procedure TTestNeuralGaussianSplat.TestOpenCLMatchesCPU;
{$IFDEF OpenCL}
const
  ImageTolerance = 1e-4;
  Eyes: array[0..1, 0..2] of TNeuralFloat = ((0, 0, -4), (3, -1, -2.5));
var
  Devices: TEasyOpenCL;
  PlatformPos, DevicePos, EyePos, GaussianPos, MaxGaussianPos, RadiusMismatches: integer;
  ValuePos, MaxValuePos: integer;
  OpenCLProblem: string;
  Scene, EmptyScene: TGaussianSplatScene;
  RasterizerCPU: TGaussianSplatRasterizer;
  RasterizerCL: TGaussianSplatRasterizerCL;
  Camera: TGaussianSplatCamera;
  OutputCPU, OutputCL: TNNetVolume;
  MaxDiff: TNeuralFloat;
begin
  Devices := TEasyOpenCL.Create();
  PlatformPos := 0;
  DevicePos := 0;
  RasterizerCL := nil;
  Scene := TGaussianSplatScene.Create;
  EmptyScene := TGaussianSplatScene.Create;
  RasterizerCPU := TGaussianSplatRasterizer.Create;
  OutputCPU := TNNetVolume.Create;
  OutputCL := TNNetVolume.Create;
  try
    if not Devices.SelectPlatformAndDevice(PlatformPos, DevicePos, OpenCLProblem) then
    begin
      AssertTrue('no OpenCL device: SKIP', true);
      Exit;
    end;
    RasterizerCL := TGaussianSplatRasterizerCL.Create(Devices.PlatformIds[PlatformPos],
      Devices.Devices[DevicePos]);
    RandSeed := 20231023;
    Scene.FillRandom(400, 3, 1);
    // 70 x 45 leaves partial tiles on the right and bottom edges.
    InitGaussianSplatCamera(Camera, 70, 45, Pi / 3);
    for EyePos := 0 to High(Eyes) do
    begin
      SetGaussianSplatCameraLookAt(Camera, Eyes[EyePos, 0], Eyes[EyePos, 1],
        Eyes[EyePos, 2], 0, 0, 0, 0, -1, 0);
      RasterizerCPU.Render(Scene, Camera, 0.1, 0.2, 0.3, OutputCPU);
      RasterizerCL.Render(Scene, Camera, 0.1, 0.2, 0.3, OutputCL);
      AssertTrue('some Gaussians visible', RasterizerCPU.VisibleCount > 100);
      AssertEquals('visible count', RasterizerCPU.VisibleCount, RasterizerCL.VisibleCount);
      RadiusMismatches := 0;
      MaxGaussianPos := Scene.Count - 1;
      for GaussianPos := 0 to MaxGaussianPos do
      begin
        if RasterizerCPU.Radii[GaussianPos] <> RasterizerCL.Radii[GaussianPos] then
          Inc(RadiusMismatches)
        else if RasterizerCPU.Radii[GaussianPos] > 0 then
          AssertEquals('depth', RasterizerCPU.Depths[GaussianPos],
            RasterizerCL.Depths[GaussianPos], 1e-5);
      end;
      AssertEquals('radius mismatches', 0, RadiusMismatches);
      AssertEquals('output size', OutputCPU.Size, OutputCL.Size);
      MaxDiff := 0;
      MaxValuePos := OutputCPU.Size - 1;
      for ValuePos := 0 to MaxValuePos do
        MaxDiff := Max(MaxDiff, Abs(OutputCPU.FData[ValuePos] - OutputCL.FData[ValuePos]));
      AssertTrue('max |CPU - OpenCL| = ' + FloatToStr(MaxDiff), MaxDiff < ImageTolerance);
    end;
    EmptyScene.Resize(0, 0);
    RasterizerCL.Render(EmptyScene, Camera, 0.1, 0.2, 0.3, OutputCL);
    AssertEquals('empty scene red', 0.1, OutputCL.FData[OutputCL.GetRawPos(69, 44, 0)], 1e-7);
    AssertEquals('empty scene blue', 0.3, OutputCL.FData[OutputCL.GetRawPos(0, 0, 2)], 1e-7);
  finally
    OutputCL.Free;
    OutputCPU.Free;
    RasterizerCL.Free;
    RasterizerCPU.Free;
    EmptyScene.Free;
    Scene.Free;
    Devices.Free;
  end;
end;
{$ELSE}
begin
  AssertTrue('OpenCL not compiled in: SKIP', true);
end;
{$ENDIF}

initialization
  RegisterTest(TTestNeuralGaussianSplat);
end.
