unit neuralgaussiansplat;
{
  neuralgaussiansplat Free Pascal/Lazarus Library by Joao Paulo Schwarz Schuler
  Conscious Artificial Intelligence Project
  https://sourceforge.net/projects/cai/

  3D Gaussian Splatting viewer (Kerbl et al., SIGGRAPH 2023, arXiv:2308.04079):
  TGaussianSplatScene loads the INRIA point_cloud.ply layout and
  TGaussianSplatRasterizer renders it with the reference tile rasterizer and
  TGaussianSplatRasterizerCL (-dOpenCL) runs its preprocess and blend in
  OpenCL kernels. Coded by Claude (AI).
}

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, neuralvolume, neuralply
  {$IFDEF OpenCL}, cl, ctypes, neuralopencl{$ENDIF};

const
  GaussianSplatTileSize = 16;
  GaussianSplatMaxSHDegree = 3;
  GaussianSplatNearPlane = 0.2;
  GaussianSplatSHC0 = 0.28209479177387814;
  GaussianSplatSHC1 = 0.4886025119029199;
  GaussianSplatSHC2: array[0..4] of TNeuralFloat = (1.0925484305920792,
    -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
    0.5462742152960396);
  GaussianSplatSHC3: array[0..6] of TNeuralFloat = (-0.5900435899266435,
    2.890611442640554, -0.4570457994644658, 0.3731763325901154,
    -0.4570457994644658, 1.445305721320277, -0.5900435899266435);

type
  // WorldToView is the row-major 3x4 [R|t] (COLMAP axes: +x right, +y down,
  // +z forward); pixel (i,j) has its centre at image coordinate (i,j).
  TGaussianSplatCamera = record
    WorldToView: array[0..11] of TNeuralFloat;
    FocalX, FocalY, PrincipalX, PrincipalY: TNeuralFloat;
    Width, Height: integer;
  end;

  // Activated Gaussians: Scales after exp, Opacities after sigmoid, Rotations
  // as unit quaternions (w,x,y,z); SHCoefs is [Gaussian][coefficient][RGB].
  TGaussianSplatScene = class
  public
    Count: integer;
    SHDegree: integer;
    SHCoefCount: integer;
    Positions: TNeuralFloatDynArr;
    Scales: TNeuralFloatDynArr;
    Rotations: TNeuralFloatDynArr;
    Opacities: TNeuralFloatDynArr;
    SHCoefs: TNeuralFloatDynArr;
    // Sizes every array for pCount Gaussians and zeroes them.
    procedure Resize(pCount, pSHDegree: integer);
    // Sets activated values; the DC coefficients make the degree-0 colour RGB.
    procedure SetGaussian(Index: integer; X, Y, Z, ScaleX, ScaleY, ScaleZ,
      RotW, RotX, RotY, RotZ, Opacity, Red, Green, Blue: TNeuralFloat);
    // Random Gaussians inside the cube [-Extent, Extent]^3 from the global
    // Random generator (set RandSeed first for a repeatable scene).
    procedure FillRandom(pCount, pSHDegree: integer; Extent: TNeuralFloat);
    procedure LoadFromPLY(const FileName: string);
    procedure LoadFromStream(Stream: TStream);
    // Writes the INRIA layout (raw logit opacity, log scale), all float32.
    procedure SaveToPLY(const FileName: string; Format: TNNetPLYFormat = plyBinaryLittleEndian);
    procedure SaveToStream(Stream: TStream; Format: TNNetPLYFormat = plyBinaryLittleEndian);
    // Cov receives R S S^T R^T as xx, xy, xz, yy, yz, zz.
    procedure ComputeCovariance3D(Index: integer; var Cov: array of TNeuralFloat);
    // View-dependent SH colour seen from the camera centre, + 0.5, clamped at 0.
    procedure ComputeColor(Index: integer; CameraX, CameraY, CameraZ: TNeuralFloat;
      var RGB: array of TNeuralFloat);
    // Per-axis opacity-weighted median of the means with Opacity >= MinOpacity (all when
    // none qualify) and the opacity-weighted median distance of those means from it.
    procedure ComputeRobustCentre(MinOpacity: TNeuralFloat;
      out CentreX, CentreY, CentreZ, MedianDistance: TNeuralFloat);
  end;

  // Preprocess fills the per-Gaussian arrays, BinTiles the depth-sorted per-tile
  // lists (TileRanges[2t]..TileRanges[2t+1]-1 index TileGaussianList), Blend the image.
  TGaussianSplatRasterizer = class
  private
    FWidth, FHeight: integer;
  protected
    // Sets the image size, tile counts and Gaussian count of the next frame.
    procedure PrepareFrame(pGaussianCount: integer; const Camera: TGaussianSplatCamera);
  public
    GaussianCount: integer;
    VisibleCount: integer;
    TileCountX, TileCountY: integer;
    TileListLength: integer;
    MeansImage: TNeuralFloatDynArr;          // 2 per Gaussian: pixel x, y
    Depths: TNeuralFloatDynArr;              // view-space z
    Radii: TNeuralIntegerArray;              // 0 = culled
    ConicOpacity: TNeuralFloatDynArr;        // 4 per Gaussian: inverse 2D covariance a, b, c, opacity
    Colors: TNeuralFloatDynArr;              // 3 per Gaussian
    TileRects: TNeuralIntegerArray;          // 4 per Gaussian: min tile x, y, max (exclusive) x, y
    VisibleOrder: TNeuralIntegerArray;       // visible Gaussians, nearest first
    TileRanges: TNeuralIntegerArray;
    TileGaussianList: TNeuralIntegerArray;
    procedure Preprocess(Scene: TGaussianSplatScene; const Camera: TGaussianSplatCamera); virtual;
    procedure BinTiles;
    procedure BlendTile(TileIndex: integer; BackgroundRed, BackgroundGreen,
      BackgroundBlue: TNeuralFloat; Output: TNNetVolume);
    // Resizes Output to (Width, Height, 3) and blends every tile.
    procedure Blend(BackgroundRed, BackgroundGreen, BackgroundBlue: TNeuralFloat;
      Output: TNNetVolume); virtual;
    procedure Render(Scene: TGaussianSplatScene; const Camera: TGaussianSplatCamera;
      BackgroundRed, BackgroundGreen, BackgroundBlue: TNeuralFloat; Output: TNNetVolume);
    property Width: integer read FWidth;
    property Height: integer read FHeight;
  end;

  {$IFDEF OpenCL}
  // gsbCamera..gsbTileRects is the argument order of cai_gaussian_splat_preprocess.
  TGaussianSplatBufferKind = (gsbCamera, gsbPositions, gsbScales, gsbRotations,
    gsbOpacities, gsbSHCoefs, gsbMeansImage, gsbDepths, gsbRadii,
    gsbConicOpacity, gsbColors, gsbTileRects, gsbTileRanges,
    gsbTileGaussianList, gsbOutput);

  // Preprocess and Blend run in OpenCL; BinTiles stays on the host. Only Radii,
  // Depths and TileRects come back: MeansImage, ConicOpacity, Colors stay in OpenCL.
  TGaussianSplatRasterizerCL = class(TGaussianSplatRasterizer)
  private
    FPreprocessKernel, FBlendKernel: TNeuralKernel;
    FUploadedScene: TGaussianSplatScene;
    FSHDegree, FSHCoefCount: integer;
    FCameraParams: array[0..19] of TNeuralFloat;
    FBuffers: array[TGaussianSplatBufferKind] of cl_mem;
    FBufferBytes: array[TGaussianSplatBufferKind] of csize_t;
    function EnsureBuffer(Kind: TGaussianSplatBufferKind; Bytes: csize_t): cl_mem;
    procedure Upload(Kind: TGaussianSplatBufferKind; Bytes: csize_t; Source: Pointer;
      Blocking: cl_bool);
  public
    constructor Create(pPlatform: cl_platform_id; pDevice: cl_device_id);
    destructor Destroy; override;
    // Copies the scene to OpenCL memory; Preprocess calls it for a new Scene
    // object, the caller calls it again after editing a scene in place.
    procedure UploadScene(Scene: TGaussianSplatScene);
    procedure Preprocess(Scene: TGaussianSplatScene; const Camera: TGaussianSplatCamera); override;
    procedure Blend(BackgroundRed, BackgroundGreen, BackgroundBlue: TNeuralFloat;
      Output: TNNetVolume); override;
  end;
  {$ENDIF}

  // Identity view, square pixels from the horizontal field of view, centred principal point.
  procedure InitGaussianSplatCamera(out Camera: TGaussianSplatCamera;
    pWidth, pHeight: integer; FovXRadians: TNeuralFloat);
  // Points the camera from Eye at Target; image +y follows -Up.
  procedure SetGaussianSplatCameraLookAt(var Camera: TGaussianSplatCamera;
    EyeX, EyeY, EyeZ, TargetX, TargetY, TargetZ, UpX, UpY, UpZ: TNeuralFloat);
  procedure GetGaussianSplatCameraPosition(const Camera: TGaussianSplatCamera;
    out X, Y, Z: TNeuralFloat);
  // Least-squares point nearest to RayCount rays (Origins, Directions: 3 per ray); False
  // when the rays are near-parallel or the point lies behind at least half of them.
  function GaussianSplatNearestPointToRays(const Origins, Directions: array of TNeuralFloat;
    RayCount: integer; out X, Y, Z: TNeuralFloat): boolean;

implementation

procedure EnsureFloatCapacity(var Values: TNeuralFloatDynArr; Needed: integer);
begin
  if Length(Values) < Needed then SetLength(Values, Needed);
end;

function ClampTileIndex(TileCoordinate: TNeuralFloat; TileCount: integer): integer;
begin
  if TileCoordinate <= 0 then Exit(0);
  if TileCoordinate >= TileCount then Exit(TileCount);
  Result := Trunc(TileCoordinate);
end;

procedure EnsureIntegerCapacity(var Values: TNeuralIntegerArray; Needed: integer);
begin
  if Length(Values) < Needed then SetLength(Values, Needed);
end;

procedure InitGaussianSplatCamera(out Camera: TGaussianSplatCamera;
  pWidth, pHeight: integer; FovXRadians: TNeuralFloat);
var
  EntryPos: integer;
begin
  for EntryPos := 0 to 11 do Camera.WorldToView[EntryPos] := 0;
  Camera.WorldToView[0] := 1;
  Camera.WorldToView[5] := 1;
  Camera.WorldToView[10] := 1;
  Camera.Width := pWidth;
  Camera.Height := pHeight;
  Camera.FocalX := pWidth / (2 * Tan(FovXRadians * 0.5));
  Camera.FocalY := Camera.FocalX;
  Camera.PrincipalX := pWidth * 0.5 - 0.5;
  Camera.PrincipalY := pHeight * 0.5 - 0.5;
end;

procedure SetGaussianSplatCameraLookAt(var Camera: TGaussianSplatCamera;
  EyeX, EyeY, EyeZ, TargetX, TargetY, TargetZ, UpX, UpY, UpZ: TNeuralFloat);
var
  ForwardX, ForwardY, ForwardZ, DownX, DownY, DownZ: TNeuralFloat;
  RightX, RightY, RightZ, InvLength, UpAlongForward: TNeuralFloat;
begin
  ForwardX := TargetX - EyeX;
  ForwardY := TargetY - EyeY;
  ForwardZ := TargetZ - EyeZ;
  InvLength := 1 / Sqrt(ForwardX * ForwardX + ForwardY * ForwardY + ForwardZ * ForwardZ);
  ForwardX := ForwardX * InvLength;
  ForwardY := ForwardY * InvLength;
  ForwardZ := ForwardZ * InvLength;
  UpAlongForward := UpX * ForwardX + UpY * ForwardY + UpZ * ForwardZ;
  DownX := UpAlongForward * ForwardX - UpX;
  DownY := UpAlongForward * ForwardY - UpY;
  DownZ := UpAlongForward * ForwardZ - UpZ;
  InvLength := 1 / Sqrt(DownX * DownX + DownY * DownY + DownZ * DownZ);
  DownX := DownX * InvLength;
  DownY := DownY * InvLength;
  DownZ := DownZ * InvLength;
  // right = down x forward keeps (right, down, forward) right-handed.
  RightX := DownY * ForwardZ - DownZ * ForwardY;
  RightY := DownZ * ForwardX - DownX * ForwardZ;
  RightZ := DownX * ForwardY - DownY * ForwardX;
  with Camera do
  begin
    WorldToView[0] := RightX;   WorldToView[1] := RightY;   WorldToView[2] := RightZ;
    WorldToView[4] := DownX;    WorldToView[5] := DownY;    WorldToView[6] := DownZ;
    WorldToView[8] := ForwardX; WorldToView[9] := ForwardY; WorldToView[10] := ForwardZ;
    WorldToView[3] := -(RightX * EyeX + RightY * EyeY + RightZ * EyeZ);
    WorldToView[7] := -(DownX * EyeX + DownY * EyeY + DownZ * EyeZ);
    WorldToView[11] := -(ForwardX * EyeX + ForwardY * EyeY + ForwardZ * EyeZ);
  end;
end;

procedure GetGaussianSplatCameraPosition(const Camera: TGaussianSplatCamera;
  out X, Y, Z: TNeuralFloat);
begin
  with Camera do
  begin
    X := -(WorldToView[0] * WorldToView[3] + WorldToView[4] * WorldToView[7] + WorldToView[8] * WorldToView[11]);
    Y := -(WorldToView[1] * WorldToView[3] + WorldToView[5] * WorldToView[7] + WorldToView[9] * WorldToView[11]);
    Z := -(WorldToView[2] * WorldToView[3] + WorldToView[6] * WorldToView[7] + WorldToView[10] * WorldToView[11]);
  end;
end;

function GaussianSplatNearestPointToRays(const Origins, Directions: array of TNeuralFloat;
  RayCount: integer; out X, Y, Z: TNeuralFloat): boolean;
const
  // det(sum of I - d d^T) over RayCount^3: 2 orthogonal rays give 0.25, parallel rays 0.
  MinRelativeDeterminant = 1e-4;
var
  Normal: array[0..8] of TNeuralFloat;
  Rhs: array[0..2] of TNeuralFloat;
  Direction: array[0..2] of TNeuralFloat;
  RayPos, MaxRayPos, RowPos, ColPos, InFrontCount: integer;
  InvLength, Projector, Determinant, Along: TNeuralFloat;
begin
  Result := False;
  X := 0; Y := 0; Z := 0;
  if RayCount < 2 then Exit;
  FillChar(Normal, SizeOf(Normal), 0);
  FillChar(Rhs, SizeOf(Rhs), 0);
  MaxRayPos := RayCount - 1;
  // Normal equations: sum (I - d d^T) p = sum (I - d d^T) o.
  for RayPos := 0 to MaxRayPos do
  begin
    InvLength := Sqrt(Sqr(Directions[3 * RayPos]) + Sqr(Directions[3 * RayPos + 1]) +
      Sqr(Directions[3 * RayPos + 2]));
    if InvLength = 0 then Exit;
    InvLength := 1 / InvLength;
    for RowPos := 0 to 2 do Direction[RowPos] := Directions[3 * RayPos + RowPos] * InvLength;
    for RowPos := 0 to 2 do
      for ColPos := 0 to 2 do
      begin
        Projector := -Direction[RowPos] * Direction[ColPos];
        if RowPos = ColPos then Projector := Projector + 1;
        Normal[3 * RowPos + ColPos] := Normal[3 * RowPos + ColPos] + Projector;
        Rhs[RowPos] := Rhs[RowPos] + Projector * Origins[3 * RayPos + ColPos];
      end;
  end;
  Determinant :=
    Normal[0] * (Normal[4] * Normal[8] - Normal[5] * Normal[7]) -
    Normal[1] * (Normal[3] * Normal[8] - Normal[5] * Normal[6]) +
    Normal[2] * (Normal[3] * Normal[7] - Normal[4] * Normal[6]);
  if Determinant < MinRelativeDeterminant * RayCount * RayCount * RayCount then Exit;
  if not NeuralLinearSolve(Normal, Rhs, 3, 1) then Exit;
  InFrontCount := 0;
  for RayPos := 0 to MaxRayPos do
  begin
    Along := (Rhs[0] - Origins[3 * RayPos]) * Directions[3 * RayPos] +
      (Rhs[1] - Origins[3 * RayPos + 1]) * Directions[3 * RayPos + 1] +
      (Rhs[2] - Origins[3 * RayPos + 2]) * Directions[3 * RayPos + 2];
    if Along > 0 then Inc(InFrontCount);
  end;
  if 2 * InFrontCount <= RayCount then Exit;
  X := Rhs[0]; Y := Rhs[1]; Z := Rhs[2];
  Result := True;
end;

{ TGaussianSplatScene }

procedure TGaussianSplatScene.Resize(pCount, pSHDegree: integer);
begin
  if (pSHDegree < 0) or (pSHDegree > GaussianSplatMaxSHDegree) then
    raise ERangeError.CreateFmt('Gaussian splat SH degree %d is not in 0..3', [pSHDegree]);
  Count := pCount;
  SHDegree := pSHDegree;
  SHCoefCount := (pSHDegree + 1) * (pSHDegree + 1);
  Positions := nil;
  Scales := nil;
  Rotations := nil;
  Opacities := nil;
  SHCoefs := nil;
  SetLength(Positions, 3 * pCount);
  SetLength(Scales, 3 * pCount);
  SetLength(Rotations, 4 * pCount);
  SetLength(Opacities, pCount);
  SetLength(SHCoefs, 3 * SHCoefCount * pCount);
end;

procedure TGaussianSplatScene.SetGaussian(Index: integer; X, Y, Z, ScaleX,
  ScaleY, ScaleZ, RotW, RotX, RotY, RotZ, Opacity, Red, Green, Blue: TNeuralFloat);
var
  InvNorm: TNeuralFloat;
  SHBase: integer;
begin
  Positions[3 * Index] := X;
  Positions[3 * Index + 1] := Y;
  Positions[3 * Index + 2] := Z;
  Scales[3 * Index] := ScaleX;
  Scales[3 * Index + 1] := ScaleY;
  Scales[3 * Index + 2] := ScaleZ;
  InvNorm := 1 / Sqrt(RotW * RotW + RotX * RotX + RotY * RotY + RotZ * RotZ);
  Rotations[4 * Index] := RotW * InvNorm;
  Rotations[4 * Index + 1] := RotX * InvNorm;
  Rotations[4 * Index + 2] := RotY * InvNorm;
  Rotations[4 * Index + 3] := RotZ * InvNorm;
  Opacities[Index] := Opacity;
  SHBase := 3 * SHCoefCount * Index;
  SHCoefs[SHBase] := (Red - 0.5) / GaussianSplatSHC0;
  SHCoefs[SHBase + 1] := (Green - 0.5) / GaussianSplatSHC0;
  SHCoefs[SHBase + 2] := (Blue - 0.5) / GaussianSplatSHC0;
end;

procedure TGaussianSplatScene.FillRandom(pCount, pSHDegree: integer; Extent: TNeuralFloat);
const
  RestAmplitude = 0.4;
var
  GaussianPos, MaxGaussianPos, SHBase, RestPos, MaxRestPos: integer;
begin
  Resize(pCount, pSHDegree);
  MaxGaussianPos := pCount - 1;
  MaxRestPos := 3 * SHCoefCount - 1;
  for GaussianPos := 0 to MaxGaussianPos do
  begin
    SetGaussian(GaussianPos,
      Extent * (2 * Random - 1), Extent * (2 * Random - 1), Extent * (2 * Random - 1),
      Extent * (0.02 + 0.1 * Random), Extent * (0.02 + 0.1 * Random),
      Extent * (0.02 + 0.1 * Random),
      Random + 0.1, 2 * Random - 1, 2 * Random - 1, 2 * Random - 1,
      0.3 + 0.65 * Random, Random, Random, Random);
    SHBase := 3 * SHCoefCount * GaussianPos;
    for RestPos := 3 to MaxRestPos do
      SHCoefs[SHBase + RestPos] := RestAmplitude * (Random - 0.5);
  end;
end;

procedure TGaussianSplatScene.LoadFromPLY(const FileName: string);
var
  FileStream: TFileStream;
begin
  FileStream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    LoadFromStream(FileStream);
  finally
    FileStream.Free;
  end;
end;

procedure TGaussianSplatScene.LoadFromStream(Stream: TStream);
const
  ScalarNames: array[0..13] of string = ('x', 'y', 'z', 'f_dc_0', 'f_dc_1',
    'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1',
    'rot_2', 'rot_3');
var
  Reader: TNNetPLYReader;
  VertexElement, NamePos, RestCount, RestPos, MaxRestPos, Degree: integer;
  GaussianPos, MaxGaussianPos, RestPerChannel, SHBase, ChannelPos, CoefPos: integer;
  ScalarColumns: array[0..13] of integer;
  RestColumns: TNeuralIntegerArray;
  Row: TNeuralFloatDynArr;
  RotW, RotX, RotY, RotZ, QuaternionNorm: TNeuralFloat;
begin
  Reader := TNNetPLYReader.CreateOnStream(Stream);
  try
    VertexElement := Reader.FindElement('vertex');
    if VertexElement < 0 then raise EReadError.Create('Gaussian splat PLY has no vertex element');
    for NamePos := 0 to High(ScalarNames) do
    begin
      ScalarColumns[NamePos] := Reader.PropertyIndex(VertexElement, ScalarNames[NamePos]);
      if ScalarColumns[NamePos] < 0 then
        raise EReadError.CreateFmt('Gaussian splat PLY has no "%s" property', [ScalarNames[NamePos]]);
    end;
    RestCount := 0;
    while Reader.PropertyIndex(VertexElement, 'f_rest_' + IntToStr(RestCount)) >= 0 do
      Inc(RestCount);
    Degree := -1;
    case RestCount of
      0: Degree := 0;
      9: Degree := 1;
      24: Degree := 2;
      45: Degree := 3;
    end;
    if Degree < 0 then
      raise EReadError.CreateFmt('Gaussian splat PLY has %d f_rest values, expected 0, 9, 24 or 45', [RestCount]);
    SetLength(RestColumns, RestCount);
    MaxRestPos := RestCount - 1;
    for RestPos := 0 to MaxRestPos do
      RestColumns[RestPos] := Reader.PropertyIndex(VertexElement, 'f_rest_' + IntToStr(RestPos));
    Resize(Reader.Elements[VertexElement].Count, Degree);
    RestPerChannel := SHCoefCount - 1;
    SetLength(Row, Length(Reader.Elements[VertexElement].Properties));
    Reader.SkipToElement(VertexElement);
    MaxGaussianPos := Count - 1;
    for GaussianPos := 0 to MaxGaussianPos do
    begin
      Reader.ReadRow(Row);
      Positions[3 * GaussianPos] := Row[ScalarColumns[0]];
      Positions[3 * GaussianPos + 1] := Row[ScalarColumns[1]];
      Positions[3 * GaussianPos + 2] := Row[ScalarColumns[2]];
      SHBase := 3 * SHCoefCount * GaussianPos;
      SHCoefs[SHBase] := Row[ScalarColumns[3]];
      SHCoefs[SHBase + 1] := Row[ScalarColumns[4]];
      SHCoefs[SHBase + 2] := Row[ScalarColumns[5]];
      // f_rest is channel-major: all R coefficients 1.., then G, then B.
      for RestPos := 0 to MaxRestPos do
      begin
        ChannelPos := RestPos div RestPerChannel;
        CoefPos := 1 + RestPos - ChannelPos * RestPerChannel;
        SHCoefs[SHBase + 3 * CoefPos + ChannelPos] := Row[RestColumns[RestPos]];
      end;
      Opacities[GaussianPos] := Sigmoid(Row[ScalarColumns[6]]);
      Scales[3 * GaussianPos] := Exp(Row[ScalarColumns[7]]);
      Scales[3 * GaussianPos + 1] := Exp(Row[ScalarColumns[8]]);
      Scales[3 * GaussianPos + 2] := Exp(Row[ScalarColumns[9]]);
      RotW := Row[ScalarColumns[10]];
      RotX := Row[ScalarColumns[11]];
      RotY := Row[ScalarColumns[12]];
      RotZ := Row[ScalarColumns[13]];
      QuaternionNorm := Sqrt(RotW * RotW + RotX * RotX + RotY * RotY + RotZ * RotZ);
      if QuaternionNorm > 0 then
      begin
        QuaternionNorm := 1 / QuaternionNorm;
        RotW := RotW * QuaternionNorm;
        RotX := RotX * QuaternionNorm;
        RotY := RotY * QuaternionNorm;
        RotZ := RotZ * QuaternionNorm;
      end
      else
        RotW := 1;
      Rotations[4 * GaussianPos] := RotW;
      Rotations[4 * GaussianPos + 1] := RotX;
      Rotations[4 * GaussianPos + 2] := RotY;
      Rotations[4 * GaussianPos + 3] := RotZ;
    end;
  finally
    Reader.Free;
  end;
end;

procedure TGaussianSplatScene.SaveToPLY(const FileName: string; Format: TNNetPLYFormat);
var
  FileStream: TFileStream;
begin
  FileStream := TFileStream.Create(FileName, fmCreate);
  try
    SaveToStream(FileStream, Format);
  finally
    FileStream.Free;
  end;
end;

procedure TGaussianSplatScene.SaveToStream(Stream: TStream; Format: TNNetPLYFormat);
const
  OpacityLimit = 1e-6;
var
  Writer: TNNetPLYWriter;
  RestPerChannel, RestCount, RestPos, MaxRestPos, ChannelPos, CoefPos: integer;
  GaussianPos, MaxGaussianPos, SHBase, ColumnPos, AxisPos: integer;
  Row: TNeuralFloatDynArr;
  Opacity: TNeuralFloat;
begin
  RestPerChannel := SHCoefCount - 1;
  RestCount := 3 * RestPerChannel;
  MaxRestPos := RestCount - 1;
  Writer := TNNetPLYWriter.CreateOnStream(Stream, Format);
  try
    Writer.AddElement('vertex', Count);
    Writer.AddProperty('x', plyFloat32);
    Writer.AddProperty('y', plyFloat32);
    Writer.AddProperty('z', plyFloat32);
    Writer.AddProperty('nx', plyFloat32);
    Writer.AddProperty('ny', plyFloat32);
    Writer.AddProperty('nz', plyFloat32);
    Writer.AddProperty('f_dc_0', plyFloat32);
    Writer.AddProperty('f_dc_1', plyFloat32);
    Writer.AddProperty('f_dc_2', plyFloat32);
    for RestPos := 0 to MaxRestPos do
      Writer.AddProperty('f_rest_' + IntToStr(RestPos), plyFloat32);
    Writer.AddProperty('opacity', plyFloat32);
    Writer.AddProperty('scale_0', plyFloat32);
    Writer.AddProperty('scale_1', plyFloat32);
    Writer.AddProperty('scale_2', plyFloat32);
    Writer.AddProperty('rot_0', plyFloat32);
    Writer.AddProperty('rot_1', plyFloat32);
    Writer.AddProperty('rot_2', plyFloat32);
    Writer.AddProperty('rot_3', plyFloat32);
    Writer.WriteHeader;
    SetLength(Row, Length(Writer.Elements[0].Properties));
    MaxGaussianPos := Count - 1;
    for GaussianPos := 0 to MaxGaussianPos do
    begin
      Row[0] := Positions[3 * GaussianPos];
      Row[1] := Positions[3 * GaussianPos + 1];
      Row[2] := Positions[3 * GaussianPos + 2];
      Row[3] := 0;
      Row[4] := 0;
      Row[5] := 0;
      SHBase := 3 * SHCoefCount * GaussianPos;
      Row[6] := SHCoefs[SHBase];
      Row[7] := SHCoefs[SHBase + 1];
      Row[8] := SHCoefs[SHBase + 2];
      for RestPos := 0 to MaxRestPos do
      begin
        ChannelPos := RestPos div RestPerChannel;
        CoefPos := 1 + RestPos - ChannelPos * RestPerChannel;
        Row[9 + RestPos] := SHCoefs[SHBase + 3 * CoefPos + ChannelPos];
      end;
      ColumnPos := 9 + RestCount;
      Opacity := Min(Max(Opacities[GaussianPos], OpacityLimit), 1 - OpacityLimit);
      Row[ColumnPos] := Ln(Opacity / (1 - Opacity));
      for AxisPos := 0 to 2 do
        Row[ColumnPos + 1 + AxisPos] := Ln(Max(Scales[3 * GaussianPos + AxisPos], 1e-30));
      for AxisPos := 0 to 3 do
        Row[ColumnPos + 4 + AxisPos] := Rotations[4 * GaussianPos + AxisPos];
      Writer.WriteRow(Row);
    end;
  finally
    Writer.Free;
  end;
end;

procedure TGaussianSplatScene.ComputeCovariance3D(Index: integer; var Cov: array of TNeuralFloat);
var
  QuatBase, ScaleBase: integer;
  W, X, Y, Z: TNeuralFloat;
  R00, R01, R02, R10, R11, R12, R20, R21, R22: TNeuralFloat;
  VarX, VarY, VarZ: TNeuralFloat;
begin
  QuatBase := 4 * Index;
  ScaleBase := 3 * Index;
  W := Rotations[QuatBase];
  X := Rotations[QuatBase + 1];
  Y := Rotations[QuatBase + 2];
  Z := Rotations[QuatBase + 3];
  R00 := 1 - 2 * (Y * Y + Z * Z); R01 := 2 * (X * Y - W * Z);     R02 := 2 * (X * Z + W * Y);
  R10 := 2 * (X * Y + W * Z);     R11 := 1 - 2 * (X * X + Z * Z); R12 := 2 * (Y * Z - W * X);
  R20 := 2 * (X * Z - W * Y);     R21 := 2 * (Y * Z + W * X);     R22 := 1 - 2 * (X * X + Y * Y);
  VarX := Sqr(Scales[ScaleBase]);
  VarY := Sqr(Scales[ScaleBase + 1]);
  VarZ := Sqr(Scales[ScaleBase + 2]);
  Cov[0] := R00 * R00 * VarX + R01 * R01 * VarY + R02 * R02 * VarZ;
  Cov[1] := R00 * R10 * VarX + R01 * R11 * VarY + R02 * R12 * VarZ;
  Cov[2] := R00 * R20 * VarX + R01 * R21 * VarY + R02 * R22 * VarZ;
  Cov[3] := R10 * R10 * VarX + R11 * R11 * VarY + R12 * R12 * VarZ;
  Cov[4] := R10 * R20 * VarX + R11 * R21 * VarY + R12 * R22 * VarZ;
  Cov[5] := R20 * R20 * VarX + R21 * R21 * VarY + R22 * R22 * VarZ;
end;

procedure TGaussianSplatScene.ComputeColor(Index: integer; CameraX, CameraY,
  CameraZ: TNeuralFloat; var RGB: array of TNeuralFloat);
var
  SHBase, ChannelPos: integer;
  DirX, DirY, DirZ, InvLength: TNeuralFloat;
  XX, YY, ZZ, XY, YZ, XZ: TNeuralFloat;
  Basis: array[0..15] of TNeuralFloat;
  CoefPos, MaxCoefPos, CoefBase: integer;
  Sum: TNeuralFloat;
begin
  Basis[0] := GaussianSplatSHC0;
  if SHDegree > 0 then
  begin
    DirX := Positions[3 * Index] - CameraX;
    DirY := Positions[3 * Index + 1] - CameraY;
    DirZ := Positions[3 * Index + 2] - CameraZ;
    InvLength := DirX * DirX + DirY * DirY + DirZ * DirZ;
    if InvLength > 0 then InvLength := 1 / Sqrt(InvLength);
    DirX := DirX * InvLength;
    DirY := DirY * InvLength;
    DirZ := DirZ * InvLength;
    Basis[1] := -GaussianSplatSHC1 * DirY;
    Basis[2] := GaussianSplatSHC1 * DirZ;
    Basis[3] := -GaussianSplatSHC1 * DirX;
    if SHDegree > 1 then
    begin
      XX := DirX * DirX; YY := DirY * DirY; ZZ := DirZ * DirZ;
      XY := DirX * DirY; YZ := DirY * DirZ; XZ := DirX * DirZ;
      Basis[4] := GaussianSplatSHC2[0] * XY;
      Basis[5] := GaussianSplatSHC2[1] * YZ;
      Basis[6] := GaussianSplatSHC2[2] * (2 * ZZ - XX - YY);
      Basis[7] := GaussianSplatSHC2[3] * XZ;
      Basis[8] := GaussianSplatSHC2[4] * (XX - YY);
      if SHDegree > 2 then
      begin
        Basis[9] := GaussianSplatSHC3[0] * DirY * (3 * XX - YY);
        Basis[10] := GaussianSplatSHC3[1] * XY * DirZ;
        Basis[11] := GaussianSplatSHC3[2] * DirY * (4 * ZZ - XX - YY);
        Basis[12] := GaussianSplatSHC3[3] * DirZ * (2 * ZZ - 3 * XX - 3 * YY);
        Basis[13] := GaussianSplatSHC3[4] * DirX * (4 * ZZ - XX - YY);
        Basis[14] := GaussianSplatSHC3[5] * DirZ * (XX - YY);
        Basis[15] := GaussianSplatSHC3[6] * DirX * (XX - 3 * YY);
      end;
    end;
  end;
  SHBase := 3 * SHCoefCount * Index;
  MaxCoefPos := SHCoefCount - 1;
  for ChannelPos := 0 to 2 do
  begin
    Sum := 0.5;
    CoefBase := SHBase + ChannelPos;
    for CoefPos := 0 to MaxCoefPos do
    begin
      Sum := Sum + Basis[CoefPos] * SHCoefs[CoefBase];
      Inc(CoefBase, 3);
    end;
    if Sum < 0 then Sum := 0;
    RGB[ChannelPos] := Sum;
  end;
end;

procedure TGaussianSplatScene.ComputeRobustCentre(MinOpacity: TNeuralFloat;
  out CentreX, CentreY, CentreZ, MedianDistance: TNeuralFloat);
var
  Selected, Order: TNeuralIntegerArray;
  Values, Weights: TNeuralFloatDynArr;
  Centre: array[0..2] of TNeuralFloat;
  GaussianPos, MaxGaussianPos, SelectedCount, SelectedPos, MaxSelectedPos, AxisPos,
    PositionBase: integer;
begin
  CentreX := 0; CentreY := 0; CentreZ := 0; MedianDistance := 0;
  if Count <= 0 then Exit;
  SetLength(Selected, Count);
  MaxGaussianPos := Count - 1;
  SelectedCount := 0;
  for GaussianPos := 0 to MaxGaussianPos do
    if Opacities[GaussianPos] >= MinOpacity then
    begin
      Selected[SelectedCount] := GaussianPos;
      Inc(SelectedCount);
    end;
  if SelectedCount = 0 then
  begin
    for GaussianPos := 0 to MaxGaussianPos do Selected[GaussianPos] := GaussianPos;
    SelectedCount := Count;
  end;
  MaxSelectedPos := SelectedCount - 1;
  SetLength(Order, SelectedCount);
  SetLength(Values, SelectedCount);
  SetLength(Weights, SelectedCount);
  for SelectedPos := 0 to MaxSelectedPos do
    Weights[SelectedPos] := Opacities[Selected[SelectedPos]];
  for AxisPos := 0 to 2 do
  begin
    for SelectedPos := 0 to MaxSelectedPos do
      Values[SelectedPos] := Positions[3 * Selected[SelectedPos] + AxisPos];
    Centre[AxisPos] := NeuralWeightedMedian(Values, Weights, Order, SelectedCount);
  end;
  for SelectedPos := 0 to MaxSelectedPos do
  begin
    PositionBase := 3 * Selected[SelectedPos];
    Values[SelectedPos] := Sqrt(Sqr(Positions[PositionBase] - Centre[0]) +
      Sqr(Positions[PositionBase + 1] - Centre[1]) + Sqr(Positions[PositionBase + 2] - Centre[2]));
  end;
  MedianDistance := NeuralWeightedMedian(Values, Weights, Order, SelectedCount);
  CentreX := Centre[0]; CentreY := Centre[1]; CentreZ := Centre[2];
end;

{ TGaussianSplatRasterizer }

procedure TGaussianSplatRasterizer.PrepareFrame(pGaussianCount: integer;
  const Camera: TGaussianSplatCamera);
begin
  FWidth := Camera.Width;
  FHeight := Camera.Height;
  TileCountX := (FWidth + GaussianSplatTileSize - 1) div GaussianSplatTileSize;
  TileCountY := (FHeight + GaussianSplatTileSize - 1) div GaussianSplatTileSize;
  GaussianCount := pGaussianCount;
end;

procedure TGaussianSplatRasterizer.Preprocess(Scene: TGaussianSplatScene;
  const Camera: TGaussianSplatCamera);
var
  GaussianPos, MaxGaussianPos, PositionBase, Radius: integer;
  R00, R01, R02, R10, R11, R12, R20, R21, R22, T0, T1, T2: TNeuralFloat;
  FocalX, FocalY, PrincipalX, PrincipalY, LimitX, LimitY: TNeuralFloat;
  CameraX, CameraY, CameraZ: TNeuralFloat;
  ViewX, ViewY, ViewZ, InvZ, ClampedX, ClampedY: TNeuralFloat;
  J00, J02, J11, J12: TNeuralFloat;
  A0, A1, A2, B0, B1, B2: TNeuralFloat;
  CovA, CovB, CovC, Det, InvDet, MidValue, Lambda1: TNeuralFloat;
  MeanX, MeanY, TileExtent, InvTileSize: TNeuralFloat;
  MinTileX, MinTileY, MaxTileX, MaxTileY: integer;
  Cov3D: array[0..5] of TNeuralFloat;
  RGB: array[0..2] of TNeuralFloat;
begin
  PrepareFrame(Scene.Count, Camera);
  EnsureFloatCapacity(MeansImage, 2 * GaussianCount);
  EnsureFloatCapacity(Depths, GaussianCount);
  EnsureIntegerCapacity(Radii, GaussianCount);
  EnsureFloatCapacity(ConicOpacity, 4 * GaussianCount);
  EnsureFloatCapacity(Colors, 3 * GaussianCount);
  EnsureIntegerCapacity(TileRects, 4 * GaussianCount);
  with Camera do
  begin
    R00 := WorldToView[0]; R01 := WorldToView[1]; R02 := WorldToView[2];  T0 := WorldToView[3];
    R10 := WorldToView[4]; R11 := WorldToView[5]; R12 := WorldToView[6];  T1 := WorldToView[7];
    R20 := WorldToView[8]; R21 := WorldToView[9]; R22 := WorldToView[10]; T2 := WorldToView[11];
  end;
  FocalX := Camera.FocalX;
  FocalY := Camera.FocalY;
  PrincipalX := Camera.PrincipalX;
  PrincipalY := Camera.PrincipalY;
  // The reference clamps x/z and y/z to 1.3 x tan(half fov) before the Jacobian.
  LimitX := 1.3 * FWidth / (2 * FocalX);
  LimitY := 1.3 * FHeight / (2 * FocalY);
  GetGaussianSplatCameraPosition(Camera, CameraX, CameraY, CameraZ);
  TileExtent := GaussianSplatTileSize - 1;
  InvTileSize := 1 / GaussianSplatTileSize;
  MaxGaussianPos := GaussianCount - 1;
  for GaussianPos := 0 to MaxGaussianPos do
  begin
    Radii[GaussianPos] := 0;
    PositionBase := 3 * GaussianPos;
    ViewX := R00 * Scene.Positions[PositionBase] + R01 * Scene.Positions[PositionBase + 1] +
      R02 * Scene.Positions[PositionBase + 2] + T0;
    ViewY := R10 * Scene.Positions[PositionBase] + R11 * Scene.Positions[PositionBase + 1] +
      R12 * Scene.Positions[PositionBase + 2] + T1;
    ViewZ := R20 * Scene.Positions[PositionBase] + R21 * Scene.Positions[PositionBase + 1] +
      R22 * Scene.Positions[PositionBase + 2] + T2;
    if ViewZ <= GaussianSplatNearPlane then Continue;
    InvZ := 1 / ViewZ;
    ClampedX := Min(LimitX, Max(-LimitX, ViewX * InvZ)) * ViewZ;
    ClampedY := Min(LimitY, Max(-LimitY, ViewY * InvZ)) * ViewZ;
    J00 := FocalX * InvZ;
    J02 := -FocalX * ClampedX * InvZ * InvZ;
    J11 := FocalY * InvZ;
    J12 := -FocalY * ClampedY * InvZ * InvZ;
    // Rows of J W, the 2x3 map from world offsets to pixel offsets.
    A0 := J00 * R00 + J02 * R20;
    A1 := J00 * R01 + J02 * R21;
    A2 := J00 * R02 + J02 * R22;
    B0 := J11 * R10 + J12 * R20;
    B1 := J11 * R11 + J12 * R21;
    B2 := J11 * R12 + J12 * R22;
    Scene.ComputeCovariance3D(GaussianPos, Cov3D);
    CovA := A0 * (Cov3D[0] * A0 + Cov3D[1] * A1 + Cov3D[2] * A2) +
      A1 * (Cov3D[1] * A0 + Cov3D[3] * A1 + Cov3D[4] * A2) +
      A2 * (Cov3D[2] * A0 + Cov3D[4] * A1 + Cov3D[5] * A2) + 0.3;
    CovB := B0 * (Cov3D[0] * A0 + Cov3D[1] * A1 + Cov3D[2] * A2) +
      B1 * (Cov3D[1] * A0 + Cov3D[3] * A1 + Cov3D[4] * A2) +
      B2 * (Cov3D[2] * A0 + Cov3D[4] * A1 + Cov3D[5] * A2);
    CovC := B0 * (Cov3D[0] * B0 + Cov3D[1] * B1 + Cov3D[2] * B2) +
      B1 * (Cov3D[1] * B0 + Cov3D[3] * B1 + Cov3D[4] * B2) +
      B2 * (Cov3D[2] * B0 + Cov3D[4] * B1 + Cov3D[5] * B2) + 0.3;
    Det := CovA * CovC - CovB * CovB;
    if Det <= 0 then Continue;
    InvDet := 1 / Det;
    MidValue := 0.5 * (CovA + CovC);
    Lambda1 := MidValue + Sqrt(Max(0.1, MidValue * MidValue - Det));
    Radius := Ceil(3 * Sqrt(Lambda1));
    MeanX := FocalX * ViewX * InvZ + PrincipalX;
    MeanY := FocalY * ViewY * InvZ + PrincipalY;
    MinTileX := ClampTileIndex((MeanX - Radius) * InvTileSize, TileCountX);
    MinTileY := ClampTileIndex((MeanY - Radius) * InvTileSize, TileCountY);
    MaxTileX := ClampTileIndex((MeanX + Radius + TileExtent) * InvTileSize, TileCountX);
    MaxTileY := ClampTileIndex((MeanY + Radius + TileExtent) * InvTileSize, TileCountY);
    if (MaxTileX <= MinTileX) or (MaxTileY <= MinTileY) then Continue;
    Scene.ComputeColor(GaussianPos, CameraX, CameraY, CameraZ, RGB);
    Radii[GaussianPos] := Radius;
    Depths[GaussianPos] := ViewZ;
    MeansImage[2 * GaussianPos] := MeanX;
    MeansImage[2 * GaussianPos + 1] := MeanY;
    ConicOpacity[4 * GaussianPos] := CovC * InvDet;
    ConicOpacity[4 * GaussianPos + 1] := -CovB * InvDet;
    ConicOpacity[4 * GaussianPos + 2] := CovA * InvDet;
    ConicOpacity[4 * GaussianPos + 3] := Scene.Opacities[GaussianPos];
    Colors[PositionBase] := RGB[0];
    Colors[PositionBase + 1] := RGB[1];
    Colors[PositionBase + 2] := RGB[2];
    TileRects[4 * GaussianPos] := MinTileX;
    TileRects[4 * GaussianPos + 1] := MinTileY;
    TileRects[4 * GaussianPos + 2] := MaxTileX;
    TileRects[4 * GaussianPos + 3] := MaxTileY;
  end;
end;

procedure TGaussianSplatRasterizer.BinTiles;
var
  GaussianPos, MaxGaussianPos, OrderPos, MaxOrderPos, RectBase: integer;
  TileCount, TilePos, MaxTilePos, TileX, TileY, RowBase, Cursor: integer;
  MinTileX, MaxTileX, MaxTileY: integer;
begin
  EnsureIntegerCapacity(VisibleOrder, GaussianCount);
  VisibleCount := 0;
  MaxGaussianPos := GaussianCount - 1;
  for GaussianPos := 0 to MaxGaussianPos do
    if Radii[GaussianPos] > 0 then
    begin
      VisibleOrder[VisibleCount] := GaussianPos;
      Inc(VisibleCount);
    end;
  // One global depth sort followed by a stable bucket pass gives every tile its
  // Gaussians nearest first, the same order as sorting on (tile id, depth).
  if VisibleCount > 1 then QuickSortOrderByDist(VisibleOrder, Depths, 0, VisibleCount - 1);
  TileCount := TileCountX * TileCountY;
  EnsureIntegerCapacity(TileRanges, 2 * TileCount);
  MaxTilePos := TileCount - 1;
  for TilePos := 0 to MaxTilePos do TileRanges[2 * TilePos + 1] := 0;
  MaxOrderPos := VisibleCount - 1;
  for OrderPos := 0 to MaxOrderPos do
  begin
    RectBase := 4 * VisibleOrder[OrderPos];
    MinTileX := TileRects[RectBase];
    MaxTileX := TileRects[RectBase + 2] - 1;
    MaxTileY := TileRects[RectBase + 3] - 1;
    for TileY := TileRects[RectBase + 1] to MaxTileY do
    begin
      RowBase := TileY * TileCountX;
      for TileX := MinTileX to MaxTileX do
        Inc(TileRanges[2 * (RowBase + TileX) + 1]);
    end;
  end;
  Cursor := 0;
  for TilePos := 0 to MaxTilePos do
  begin
    TileRanges[2 * TilePos] := Cursor;
    Inc(Cursor, TileRanges[2 * TilePos + 1]);
    TileRanges[2 * TilePos + 1] := TileRanges[2 * TilePos];
  end;
  TileListLength := Cursor;
  EnsureIntegerCapacity(TileGaussianList, TileListLength);
  // TileRanges[2t+1] is the fill cursor here and ends as the exclusive end.
  for OrderPos := 0 to MaxOrderPos do
  begin
    GaussianPos := VisibleOrder[OrderPos];
    RectBase := 4 * GaussianPos;
    MinTileX := TileRects[RectBase];
    MaxTileX := TileRects[RectBase + 2] - 1;
    MaxTileY := TileRects[RectBase + 3] - 1;
    for TileY := TileRects[RectBase + 1] to MaxTileY do
    begin
      RowBase := TileY * TileCountX;
      for TileX := MinTileX to MaxTileX do
      begin
        TilePos := 2 * (RowBase + TileX) + 1;
        TileGaussianList[TileRanges[TilePos]] := GaussianPos;
        Inc(TileRanges[TilePos]);
      end;
    end;
  end;
end;

procedure TGaussianSplatRasterizer.BlendTile(TileIndex: integer; BackgroundRed,
  BackgroundGreen, BackgroundBlue: TNeuralFloat; Output: TNNetVolume);
const
  MinAlpha = 1 / 255;
  MaxAlpha = 0.99;
  MinTransmittance = 0.0001;
var
  FirstX, FirstY, MaxPixelX, MaxPixelY, PixelX, PixelY: integer;
  FirstListPos, MaxListPos, ListPos, GaussianPos, ConicBase, ColorBase, OutputPos: integer;
  Transmittance, NextTransmittance, Alpha, Weight, Power, DeltaX, DeltaY: TNeuralFloat;
  SumRed, SumGreen, SumBlue: TNeuralFloat;
begin
  FirstX := (TileIndex mod TileCountX) * GaussianSplatTileSize;
  FirstY := (TileIndex div TileCountX) * GaussianSplatTileSize;
  MaxPixelX := Min(FirstX + GaussianSplatTileSize, FWidth) - 1;
  MaxPixelY := Min(FirstY + GaussianSplatTileSize, FHeight) - 1;
  FirstListPos := TileRanges[2 * TileIndex];
  MaxListPos := TileRanges[2 * TileIndex + 1] - 1;
  for PixelY := FirstY to MaxPixelY do
  begin
    OutputPos := (PixelY * FWidth + FirstX) * 3;
    for PixelX := FirstX to MaxPixelX do
    begin
      Transmittance := 1;
      SumRed := 0;
      SumGreen := 0;
      SumBlue := 0;
      for ListPos := FirstListPos to MaxListPos do
      begin
        GaussianPos := TileGaussianList[ListPos];
        ConicBase := 4 * GaussianPos;
        DeltaX := MeansImage[2 * GaussianPos] - PixelX;
        DeltaY := MeansImage[2 * GaussianPos + 1] - PixelY;
        Power := -0.5 * (ConicOpacity[ConicBase] * DeltaX * DeltaX +
          ConicOpacity[ConicBase + 2] * DeltaY * DeltaY) -
          ConicOpacity[ConicBase + 1] * DeltaX * DeltaY;
        if Power > 0 then Continue;
        Alpha := ConicOpacity[ConicBase + 3] * NeuralExp(Power);
        if Alpha > MaxAlpha then Alpha := MaxAlpha;
        if Alpha < MinAlpha then Continue;
        NextTransmittance := Transmittance * (1 - Alpha);
        if NextTransmittance < MinTransmittance then Break;
        Weight := Alpha * Transmittance;
        ColorBase := 3 * GaussianPos;
        SumRed := SumRed + Colors[ColorBase] * Weight;
        SumGreen := SumGreen + Colors[ColorBase + 1] * Weight;
        SumBlue := SumBlue + Colors[ColorBase + 2] * Weight;
        Transmittance := NextTransmittance;
      end;
      Output.FData[OutputPos] := SumRed + Transmittance * BackgroundRed;
      Output.FData[OutputPos + 1] := SumGreen + Transmittance * BackgroundGreen;
      Output.FData[OutputPos + 2] := SumBlue + Transmittance * BackgroundBlue;
      Inc(OutputPos, 3);
    end;
  end;
end;

procedure TGaussianSplatRasterizer.Blend(BackgroundRed, BackgroundGreen,
  BackgroundBlue: TNeuralFloat; Output: TNNetVolume);
var
  TilePos, MaxTilePos: integer;
begin
  Output.ReSize(FWidth, FHeight, 3);
  MaxTilePos := TileCountX * TileCountY - 1;
  for TilePos := 0 to MaxTilePos do
    BlendTile(TilePos, BackgroundRed, BackgroundGreen, BackgroundBlue, Output);
end;

procedure TGaussianSplatRasterizer.Render(Scene: TGaussianSplatScene;
  const Camera: TGaussianSplatCamera; BackgroundRed, BackgroundGreen,
  BackgroundBlue: TNeuralFloat; Output: TNNetVolume);
begin
  Preprocess(Scene, Camera);
  BinTiles;
  Blend(BackgroundRed, BackgroundGreen, BackgroundBlue, Output);
end;

{$IFDEF OpenCL}
{ TGaussianSplatRasterizerCL }

constructor TGaussianSplatRasterizerCL.Create(pPlatform: cl_platform_id;
  pDevice: cl_device_id);
var
  BlendGroupSize: integer;
begin
  inherited Create;
  FPreprocessKernel := TNeuralKernel.Create(pPlatform, pDevice,
    'cai_gaussian_splat_preprocess', true);
  FBlendKernel := TNeuralKernel.CreateFromProgram(FPreprocessKernel,
    'cai_gaussian_splat_blend');
  BlendGroupSize := FBlendKernel.KernelMaxWorkGroupSize(FBlendKernel.Kernel);
  if BlendGroupSize < GaussianSplatTileSize * GaussianSplatTileSize then
    raise Exception.CreateFmt('cai_gaussian_splat_blend needs %d work-items per ' +
      'work-group; this OpenCL device allows %d',
      [GaussianSplatTileSize * GaussianSplatTileSize, BlendGroupSize]);
end;

destructor TGaussianSplatRasterizerCL.Destroy;
var
  Kind: TGaussianSplatBufferKind;
begin
  for Kind := Low(TGaussianSplatBufferKind) to High(TGaussianSplatBufferKind) do
    if Assigned(FBuffers[Kind]) then clReleaseMemObject(FBuffers[Kind]);
  FBlendKernel.Free;
  FPreprocessKernel.Free;
  inherited Destroy;
end;

function TGaussianSplatRasterizerCL.EnsureBuffer(Kind: TGaussianSplatBufferKind;
  Bytes: csize_t): cl_mem;
begin
  // OpenCL rejects 0-byte buffers; an empty scene or tile list binds 4 bytes.
  Result := FPreprocessKernel.EnsureBuffer(FBuffers[Kind], FBufferBytes[Kind],
    CL_MEM_READ_WRITE, Max(Bytes, 4));
end;

procedure TGaussianSplatRasterizerCL.Upload(Kind: TGaussianSplatBufferKind;
  Bytes: csize_t; Source: Pointer; Blocking: cl_bool);
begin
  EnsureBuffer(Kind, Bytes);
  if Bytes > 0 then FPreprocessKernel.WriteBuffer(FBuffers[Kind], Bytes, Source, Blocking);
end;

procedure TGaussianSplatRasterizerCL.UploadScene(Scene: TGaussianSplatScene);
var
  FloatCount: integer;
begin
  FloatCount := Scene.Count;
  // Blocking: the caller may free or edit the scene arrays right after.
  Upload(gsbPositions, 3 * FloatCount * csNeuralFloatSize, Pointer(Scene.Positions), CL_TRUE);
  Upload(gsbScales, 3 * FloatCount * csNeuralFloatSize, Pointer(Scene.Scales), CL_TRUE);
  Upload(gsbRotations, 4 * FloatCount * csNeuralFloatSize, Pointer(Scene.Rotations), CL_TRUE);
  Upload(gsbOpacities, FloatCount * csNeuralFloatSize, Pointer(Scene.Opacities), CL_TRUE);
  Upload(gsbSHCoefs, 3 * Scene.SHCoefCount * FloatCount * csNeuralFloatSize,
    Pointer(Scene.SHCoefs), CL_TRUE);
  FSHDegree := Scene.SHDegree;
  FSHCoefCount := Scene.SHCoefCount;
  FUploadedScene := Scene;
end;

procedure TGaussianSplatRasterizerCL.Preprocess(Scene: TGaussianSplatScene;
  const Camera: TGaussianSplatCamera);
var
  Kernel: cl_kernel;
  EntryPos, ArgPos: integer;
  WidthArg, HeightArg: integer;
  GaussianBytes: csize_t;
  Kind: TGaussianSplatBufferKind;
begin
  if Scene <> FUploadedScene then UploadScene(Scene);
  PrepareFrame(Scene.Count, Camera);
  EnsureFloatCapacity(Depths, GaussianCount);
  EnsureIntegerCapacity(Radii, GaussianCount);
  EnsureIntegerCapacity(TileRects, 4 * GaussianCount);
  for EntryPos := 0 to 11 do FCameraParams[EntryPos] := Camera.WorldToView[EntryPos];
  FCameraParams[12] := Camera.FocalX;
  FCameraParams[13] := Camera.FocalY;
  FCameraParams[14] := Camera.PrincipalX;
  FCameraParams[15] := Camera.PrincipalY;
  GetGaussianSplatCameraPosition(Camera, FCameraParams[16], FCameraParams[17],
    FCameraParams[18]);
  FCameraParams[19] := GaussianSplatNearPlane;
  Upload(gsbCamera, SizeOf(FCameraParams), @FCameraParams[0], CL_FALSE);
  GaussianBytes := csize_t(GaussianCount) * csNeuralFloatSize;
  EnsureBuffer(gsbMeansImage, 2 * GaussianBytes);
  EnsureBuffer(gsbDepths, GaussianBytes);
  EnsureBuffer(gsbRadii, GaussianBytes);
  EnsureBuffer(gsbConicOpacity, 4 * GaussianBytes);
  EnsureBuffer(gsbColors, 3 * GaussianBytes);
  EnsureBuffer(gsbTileRects, 4 * GaussianBytes);
  if GaussianCount = 0 then Exit;
  WidthArg := Width;
  HeightArg := Height;
  Kernel := FPreprocessKernel.Kernel;
  clSetKernelArg(Kernel, 0, csLongintSize, @GaussianCount);
  clSetKernelArg(Kernel, 1, csLongintSize, @FSHDegree);
  clSetKernelArg(Kernel, 2, csLongintSize, @FSHCoefCount);
  clSetKernelArg(Kernel, 3, csLongintSize, @WidthArg);
  clSetKernelArg(Kernel, 4, csLongintSize, @HeightArg);
  clSetKernelArg(Kernel, 5, csLongintSize, @TileCountX);
  clSetKernelArg(Kernel, 6, csLongintSize, @TileCountY);
  ArgPos := 7;
  for Kind := gsbCamera to gsbTileRects do
  begin
    clSetKernelArg(Kernel, ArgPos, csCLMemSize, @FBuffers[Kind]);
    Inc(ArgPos);
  end;
  FPreprocessKernel.RunKernel(Kernel, GaussianCount);
  FPreprocessKernel.ReadBuffer(FBuffers[gsbRadii], GaussianBytes, @Radii[0]);
  FPreprocessKernel.ReadBuffer(FBuffers[gsbDepths], GaussianBytes, @Depths[0]);
  FPreprocessKernel.ReadBuffer(FBuffers[gsbTileRects], 4 * GaussianBytes, @TileRects[0]);
end;

procedure TGaussianSplatRasterizerCL.Blend(BackgroundRed, BackgroundGreen,
  BackgroundBlue: TNeuralFloat; Output: TNNetVolume);
var
  Kernel: cl_kernel;
  WidthArg, HeightArg: integer;
  RedArg, GreenArg, BlueArg: single;
begin
  Output.ReSize(Width, Height, 3);
  // Fields, so the non-blocking uploads read memory that outlives this call.
  Upload(gsbTileRanges, 2 * TileCountX * TileCountY * csLongintSize,
    Pointer(TileRanges), CL_FALSE);
  Upload(gsbTileGaussianList, TileListLength * csLongintSize,
    Pointer(TileGaussianList), CL_FALSE);
  EnsureBuffer(gsbOutput, Output.GetMemSize());
  WidthArg := Width;
  HeightArg := Height;
  RedArg := BackgroundRed;
  GreenArg := BackgroundGreen;
  BlueArg := BackgroundBlue;
  Kernel := FBlendKernel.Kernel;
  clSetKernelArg(Kernel, 0, csLongintSize, @WidthArg);
  clSetKernelArg(Kernel, 1, csLongintSize, @HeightArg);
  clSetKernelArg(Kernel, 2, csLongintSize, @TileCountX);
  clSetKernelArg(Kernel, 3, csNeuralFloatSize, @RedArg);
  clSetKernelArg(Kernel, 4, csNeuralFloatSize, @GreenArg);
  clSetKernelArg(Kernel, 5, csNeuralFloatSize, @BlueArg);
  clSetKernelArg(Kernel, 6, csCLMemSize, @FBuffers[gsbTileRanges]);
  clSetKernelArg(Kernel, 7, csCLMemSize, @FBuffers[gsbTileGaussianList]);
  clSetKernelArg(Kernel, 8, csCLMemSize, @FBuffers[gsbMeansImage]);
  clSetKernelArg(Kernel, 9, csCLMemSize, @FBuffers[gsbConicOpacity]);
  clSetKernelArg(Kernel, 10, csCLMemSize, @FBuffers[gsbColors]);
  clSetKernelArg(Kernel, 11, csCLMemSize, @FBuffers[gsbOutput]);
  FBlendKernel.RunKernel2D(Kernel, TileCountX * GaussianSplatTileSize,
    TileCountY * GaussianSplatTileSize, GaussianSplatTileSize, GaussianSplatTileSize);
  FBlendKernel.ReadBuffer(FBuffers[gsbOutput], Output);
end;
{$ENDIF}

end.
