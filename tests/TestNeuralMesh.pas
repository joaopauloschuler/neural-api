unit TestNeuralMesh;
(*
Tests for neuralmesh.pas (TNNetTriangleMesh, MarchingCubes, PLY/OBJ export)
and the list-property path of neuralply.pas. Expected values are derived by
hand or from the analytic sphere field r - |p|.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralvolume, neuralply, neuralmesh;

type
  TTestNeuralMesh = class(TTestCase)
  published
    // Every case's triangles use exactly the edges whose corners straddle the level.
    procedure TestTriangleTableMatchesEdgeMask;
    // One corner below the level: one triangle at hand-derived points, facing it.
    procedure TestSingleCubeOneCorner;
    // Constant fields and grids thinner than one cell give no vertices or faces.
    procedure TestEmptyVolume;
    // Sphere r - |p|: vertices on the sphere, closed oriented mesh, Euler 2, volume.
    procedure TestSphereSurface;
    // A random field fenced by a below-level border gives a closed oriented mesh.
    procedure TestRandomFieldIsClosed;
    // The strided pointer overload on an x-fastest array matches the volume overload.
    procedure TestStridedFieldMatchesVolume;
    // Vertices, colours and faces survive SaveToPLYStream / LoadFromPLYStream.
    procedure TestPLYMeshRoundTripAllFormats;
    // WriteRow / ReadFlatRow list layout, the header line and a quad split in two.
    procedure TestPLYListPropertyText;
    procedure TestOBJText;
  end;

implementation

type
  TInt64DynArr = array of Int64;

procedure SortInt64(var Values: TInt64DynArr);
var
  Gap, Index, Back, MaxValuePos: integer;
  Value: Int64;
begin
  MaxValuePos := High(Values);
  Gap := Length(Values) div 2;
  while Gap > 0 do
  begin
    for Index := Gap to MaxValuePos do
    begin
      Value := Values[Index];
      Back := Index;
      while (Back >= Gap) and (Values[Back - Gap] > Value) do
      begin
        Values[Back] := Values[Back - Gap];
        Dec(Back, Gap);
      end;
      Values[Back] := Value;
    end;
    Gap := Gap div 2;
  end;
end;

function ContainsSorted(const Values: TInt64DynArr; Key: Int64): boolean;
var
  Lo, Hi, Mid: integer;
begin
  Lo := 0;
  Hi := High(Values);
  while Lo <= Hi do
  begin
    Mid := (Lo + Hi) div 2;
    if Values[Mid] = Key then Exit(true);
    if Values[Mid] < Key then Lo := Mid + 1 else Hi := Mid - 1;
  end;
  Result := false;
end;

// Closed and consistently oriented: each directed edge occurs once and its
// reverse occurs once. Returns the undirected edge count.
function AssertClosedOriented(Test: TTestCase; Mesh: TNNetTriangleMesh): integer;
var
  DirectedEdges: TInt64DynArr;
  FacePos, MaxFacePos, CornerPos, EdgePos, MaxEdgePos: integer;
  Corners: array[0..2] of integer;
  From, Into: Int64;
begin
  SetLength(DirectedEdges, Mesh.FaceCount * 3);
  MaxFacePos := Mesh.FaceCount - 1;
  for FacePos := 0 to MaxFacePos do
  begin
    Mesh.GetFace(FacePos, Corners[0], Corners[1], Corners[2]);
    Test.AssertTrue('face corners distinct', (Corners[0] <> Corners[1]) and
      (Corners[1] <> Corners[2]) and (Corners[0] <> Corners[2]));
    for CornerPos := 0 to 2 do
      DirectedEdges[FacePos * 3 + CornerPos] :=
        Int64(Corners[CornerPos]) * Mesh.VertexCount + Corners[(CornerPos + 1) mod 3];
  end;
  SortInt64(DirectedEdges);
  MaxEdgePos := High(DirectedEdges);
  for EdgePos := 1 to MaxEdgePos do
    if DirectedEdges[EdgePos] = DirectedEdges[EdgePos - 1] then
      Test.Fail('directed edge used twice (open or flipped face)');
  for EdgePos := 0 to MaxEdgePos do
  begin
    From := DirectedEdges[EdgePos] div Mesh.VertexCount;
    Into := DirectedEdges[EdgePos] mod Mesh.VertexCount;
    if not ContainsSorted(DirectedEdges, Into * Mesh.VertexCount + From) then
      Test.Fail('boundary edge: no face on the other side');
  end;
  Result := Length(DirectedEdges) div 2;
end;

// Signed volume enclosed by the mesh (divergence theorem).
function MeshSignedVolume(Mesh: TNNetTriangleMesh): Double;
var
  FacePos, MaxFacePos, A, B, C: integer;
  AX, AY, AZ, BX, BY, BZ, CX, CY, CZ: TNeuralFloat;
begin
  Result := 0;
  MaxFacePos := Mesh.FaceCount - 1;
  for FacePos := 0 to MaxFacePos do
  begin
    Mesh.GetFace(FacePos, A, B, C);
    Mesh.GetVertex(A, AX, AY, AZ);
    Mesh.GetVertex(B, BX, BY, BZ);
    Mesh.GetVertex(C, CX, CY, CZ);
    Result := Result + (AX * (BY * CZ - BZ * CY) - AY * (BX * CZ - BZ * CX) +
      AZ * (BX * CY - BY * CX)) / 6;
  end;
end;

procedure FillSphere(Field: TNNetVolume; GridSize: integer; HalfWidth, Radius: TNeuralFloat;
  out Origin, Spacing: TNNetPoint3D);
var
  X, Y, Z, MaxGridPos: integer;
  Step: TNeuralFloat;
begin
  Field.ReSize(GridSize, GridSize, GridSize);
  Step := 2 * HalfWidth / (GridSize - 1);
  Origin := NNetPoint3D(-HalfWidth, -HalfWidth, -HalfWidth);
  Spacing := NNetPoint3D(Step, Step, Step);
  MaxGridPos := GridSize - 1;
  for X := 0 to MaxGridPos do
    for Y := 0 to MaxGridPos do
      for Z := 0 to MaxGridPos do
        Field.FData[Field.GetRawPos(X, Y, Z)] := Radius -
          Sqrt(Sqr(Origin.X + X * Step) + Sqr(Origin.Y + Y * Step) + Sqr(Origin.Z + Z * Step));
end;

procedure TTestNeuralMesh.TestTriangleTableMatchesEdgeMask;
var
  CubeIndex, SlotPos, UsedEdges, EntryCount: integer;
begin
  for CubeIndex := 0 to 255 do
  begin
    UsedEdges := 0;
    EntryCount := 0;
    for SlotPos := 0 to 15 do
      if MarchingCubesTriangleTable[CubeIndex, SlotPos] >= 0 then
      begin
        AssertEquals('no gap in case ' + IntToStr(CubeIndex), SlotPos, EntryCount);
        UsedEdges := UsedEdges or (1 shl MarchingCubesTriangleTable[CubeIndex, SlotPos]);
        Inc(EntryCount);
      end;
    AssertEquals('whole triangles in case ' + IntToStr(CubeIndex), 0, EntryCount mod 3);
    AssertEquals('edges of case ' + IntToStr(CubeIndex),
      MarchingCubesEdgeMask[CubeIndex], UsedEdges);
  end;
  AssertEquals('case 1 mask', 1 or (1 shl 3) or (1 shl 8), MarchingCubesEdgeMask[1]);
end;

procedure TTestNeuralMesh.TestSingleCubeOneCorner;
var
  Field: TNNetVolume;
  Mesh: TNNetTriangleMesh;
  X, Y, Z: TNeuralFloat;
  A, B, C: integer;
begin
  Field := TNNetVolume.Create(2, 2, 2);
  Mesh := TNNetTriangleMesh.Create;
  try
    Field.Fill(1);
    Field.FData[Field.GetRawPos(0, 0, 0)] := 0;
    // Level 0.25 sits a quarter of the way from corner 0 along edges 0, 3 and 8.
    MarchingCubes(Field, 0.25, NNetPoint3D(10, 20, 30), NNetPoint3D(2, 3, 4), Mesh);
    AssertEquals('vertices', 3, Mesh.VertexCount);
    AssertEquals('faces', 1, Mesh.FaceCount);
    Mesh.GetVertex(0, X, Y, Z);
    AssertEquals('edge 0 x', 10.5, X, 1e-6);
    AssertEquals('edge 0 y', 20, Y, 1e-6);
    AssertEquals('edge 0 z', 30, Z, 1e-6);
    Mesh.GetVertex(1, X, Y, Z);
    AssertEquals('edge 3 x', 10, X, 1e-6);
    AssertEquals('edge 3 y', 20.75, Y, 1e-6);
    AssertEquals('edge 3 z', 30, Z, 1e-6);
    Mesh.GetVertex(2, X, Y, Z);
    AssertEquals('edge 8 x', 10, X, 1e-6);
    AssertEquals('edge 8 y', 20, Y, 1e-6);
    AssertEquals('edge 8 z', 31, Z, 1e-6);
    // Table row (0, 8, 3): counter-clockwise seen from corner 0, the low side.
    Mesh.GetFace(0, A, B, C);
    AssertEquals('face a', 0, A);
    AssertEquals('face b', 2, B);
    AssertEquals('face c', 1, C);
    // Flip the field: corner 0 is now the only corner above the level.
    Field.Fill(0);
    Field.FData[Field.GetRawPos(0, 0, 0)] := 1;
    MarchingCubes(Field, 0.75, NNetPoint3D(0, 0, 0), NNetPoint3D(1, 1, 1), Mesh);
    AssertEquals('flipped faces', 1, Mesh.FaceCount);
    AssertTrue('flipped volume sign', MeshSignedVolume(Mesh) > 0);
  finally
    Mesh.Free;
    Field.Free;
  end;
end;

procedure TTestNeuralMesh.TestEmptyVolume;
var
  Field: TNNetVolume;
  Mesh: TNNetTriangleMesh;
begin
  Field := TNNetVolume.Create(4, 5, 6);
  Mesh := TNNetTriangleMesh.Create;
  try
    Mesh.AddVertex(1, 2, 3);
    Field.Fill(2);
    MarchingCubes(Field, 0.5, NNetPoint3D(0, 0, 0), NNetPoint3D(1, 1, 1), Mesh);
    AssertEquals('above: vertices', 0, Mesh.VertexCount);
    AssertEquals('above: faces', 0, Mesh.FaceCount);
    Field.Fill(-2);
    MarchingCubes(Field, 0.5, NNetPoint3D(0, 0, 0), NNetPoint3D(1, 1, 1), Mesh);
    AssertEquals('below: faces', 0, Mesh.FaceCount);
    Field.ReSize(1, 5, 5);
    Field.FData[3] := 9;
    MarchingCubes(Field, 0.5, NNetPoint3D(0, 0, 0), NNetPoint3D(1, 1, 1), Mesh);
    AssertEquals('thin grid: faces', 0, Mesh.FaceCount);
  finally
    Mesh.Free;
    Field.Free;
  end;
end;

procedure TTestNeuralMesh.TestSphereSurface;
const
  GridSize = 24;
  HalfWidth = 1.2;
  Radius = 1.0;
var
  Field: TNNetVolume;
  Mesh: TNNetTriangleMesh;
  Origin, Spacing: TNNetPoint3D;
  VertexPos, MaxVertexPos, EdgeCount, ExpectedVertexCount: integer;
  X, Y, Z, WorstError: TNeuralFloat;
begin
  Field := TNNetVolume.Create;
  Mesh := TNNetTriangleMesh.Create;
  try
    FillSphere(Field, GridSize, HalfWidth, Radius, Origin, Spacing);
    MarchingCubes(Field, 0, Origin, Spacing, Mesh);
    // Roughly one vertex per crossing edge: about 1.5 * area / spacing^2.
    ExpectedVertexCount := Round(1.5 * 4 * Pi * Sqr(Radius) / Sqr(Spacing.X));
    AssertTrue('vertex count ' + IntToStr(Mesh.VertexCount),
      (Mesh.VertexCount > ExpectedVertexCount div 2) and
      (Mesh.VertexCount < ExpectedVertexCount * 2));
    WorstError := 0;
    MaxVertexPos := Mesh.VertexCount - 1;
    for VertexPos := 0 to MaxVertexPos do
    begin
      Mesh.GetVertex(VertexPos, X, Y, Z);
      WorstError := Max(WorstError, Abs(Sqrt(X * X + Y * Y + Z * Z) - Radius));
    end;
    AssertTrue('vertices on the sphere, worst ' + FloatToStr(WorstError),
      WorstError < 0.05 * Spacing.X);
    EdgeCount := AssertClosedOriented(Self, Mesh);
    AssertEquals('Euler characteristic', 2, Mesh.VertexCount - EdgeCount + Mesh.FaceCount);
    // Positive signed volume: normals point out, toward the field below the level.
    AssertEquals('enclosed volume', 4 / 3 * Pi * Power(Radius, 3), MeshSignedVolume(Mesh), 0.05);
  finally
    Mesh.Free;
    Field.Free;
  end;
end;

procedure TTestNeuralMesh.TestRandomFieldIsClosed;
const
  GridSize = 10;
var
  Field: TNNetVolume;
  Mesh: TNNetTriangleMesh;
  X, Y, Z, MaxGridPos, Trial: integer;
begin
  Field := TNNetVolume.Create(GridSize, GridSize, GridSize);
  Mesh := TNNetTriangleMesh.Create;
  try
    RandSeed := 1234;
    MaxGridPos := GridSize - 1;
    for Trial := 1 to 5 do
    begin
      for X := 0 to MaxGridPos do
        for Y := 0 to MaxGridPos do
          for Z := 0 to MaxGridPos do
            if (X = 0) or (Y = 0) or (Z = 0) or (X = MaxGridPos) or (Y = MaxGridPos) or
              (Z = MaxGridPos) then
              Field.FData[Field.GetRawPos(X, Y, Z)] := -1
            else
              Field.FData[Field.GetRawPos(X, Y, Z)] := Random - 0.5;
      MarchingCubes(Field, 0, NNetPoint3D(0, 0, 0), NNetPoint3D(1, 1, 1), Mesh);
      AssertTrue('faces', Mesh.FaceCount > 0);
      AssertClosedOriented(Self, Mesh);
      AssertTrue('volume sign', MeshSignedVolume(Mesh) > 0);
    end;
  finally
    Mesh.Free;
    Field.Free;
  end;
end;

procedure TTestNeuralMesh.TestStridedFieldMatchesVolume;
const
  GridSize = 12;
var
  Field: TNNetVolume;
  Flat: TNeuralFloatDynArr;
  VolumeMesh, FlatMesh: TNNetTriangleMesh;
  Origin, Spacing: TNNetPoint3D;
  X, Y, Z, MaxGridPos, SlotPos, MaxSlotPos: integer;
begin
  Field := TNNetVolume.Create;
  VolumeMesh := TNNetTriangleMesh.Create;
  FlatMesh := TNNetTriangleMesh.Create;
  try
    FillSphere(Field, GridSize, 1.1, 0.8, Origin, Spacing);
    SetLength(Flat, GridSize * GridSize * GridSize);
    MaxGridPos := GridSize - 1;
    for X := 0 to MaxGridPos do
      for Y := 0 to MaxGridPos do
        for Z := 0 to MaxGridPos do
          Flat[(Z * GridSize + Y) * GridSize + X] := Field.FData[Field.GetRawPos(X, Y, Z)];
    MarchingCubes(Field, 0, Origin, Spacing, VolumeMesh);
    MarchingCubes(@Flat[0], GridSize, GridSize, GridSize, 1, GridSize, GridSize * GridSize,
      0, Origin, Spacing, FlatMesh);
    AssertEquals('vertices', VolumeMesh.VertexCount, FlatMesh.VertexCount);
    AssertEquals('faces', VolumeMesh.FaceCount, FlatMesh.FaceCount);
    MaxSlotPos := VolumeMesh.VertexCount * 3 - 1;
    for SlotPos := 0 to MaxSlotPos do
      AssertEquals('vertex slot', VolumeMesh.Vertices[SlotPos], FlatMesh.Vertices[SlotPos], 0);
    MaxSlotPos := VolumeMesh.FaceCount * 3 - 1;
    for SlotPos := 0 to MaxSlotPos do
      AssertEquals('face slot', VolumeMesh.Faces[SlotPos], FlatMesh.Faces[SlotPos]);
  finally
    FlatMesh.Free;
    VolumeMesh.Free;
    Field.Free;
  end;
end;

procedure TTestNeuralMesh.TestPLYMeshRoundTripAllFormats;
var
  PLYFormat: TNNetPLYFormat;
  Mesh, Loaded: TNNetTriangleMesh;
  Stream: TMemoryStream;
  SlotPos, MaxSlotPos: integer;
begin
  Mesh := TNNetTriangleMesh.Create;
  Loaded := TNNetTriangleMesh.Create;
  Stream := TMemoryStream.Create;
  try
    Mesh.AddVertex(0, 0, 0);
    Mesh.AddVertex(1.5, -0.25, 3.125e-3);
    Mesh.AddVertex(-7, 2, 1e5);
    Mesh.AddVertex(0.5, 0.5, -0.5);
    Mesh.AddFace(0, 1, 2);
    Mesh.AddFace(0, 3, 1);
    Mesh.AddFace(3, 2, 1);
    Mesh.SetVertexColor(1, 1, 0.5, 0);
    Mesh.SetVertexColor(3, 0.2, 0.4, 1.3);
    for PLYFormat := Low(TNNetPLYFormat) to High(TNNetPLYFormat) do
    begin
      Stream.Clear;
      Mesh.SaveToPLYStream(Stream, PLYFormat);
      Stream.Position := 0;
      Loaded.LoadFromPLYStream(Stream);
      AssertEquals('vertices', Mesh.VertexCount, Loaded.VertexCount);
      AssertEquals('faces', Mesh.FaceCount, Loaded.FaceCount);
      AssertTrue('colours', Loaded.HasColors);
      MaxSlotPos := Mesh.VertexCount * 3 - 1;
      for SlotPos := 0 to MaxSlotPos do
      begin
        AssertEquals('vertex slot', Mesh.Vertices[SlotPos], Loaded.Vertices[SlotPos],
          Abs(Mesh.Vertices[SlotPos]) * 1e-7);
        AssertEquals('colour slot', Min(Mesh.Colors[SlotPos], 1), Loaded.Colors[SlotPos], 0.51 / 255);
      end;
      MaxSlotPos := Mesh.FaceCount * 3 - 1;
      for SlotPos := 0 to MaxSlotPos do
        AssertEquals('face slot', Mesh.Faces[SlotPos], Loaded.Faces[SlotPos]);
    end;
  finally
    Stream.Free;
    Loaded.Free;
    Mesh.Free;
  end;
end;

procedure TTestNeuralMesh.TestPLYListPropertyText;
const
  QuadText = 'ply'#10'format ascii 1.0'#10'element vertex 4'#10 +
    'property float x'#10'property float y'#10'property float z'#10 +
    'element face 1'#10'property uchar flags'#10'property list uchar int vertex_index'#10 +
    'end_header'#10'0 0 0'#10'1 0 0'#10'1 1 0'#10'0 1 0'#10'7 4 0 1 2 3'#10;
var
  Stream: TStringStream;
  Writer: TNNetPLYWriter;
  Reader: TNNetPLYReader;
  Mesh: TNNetTriangleMesh;
  Row: TNeuralFloatDynArr;
  A, B, C: integer;
begin
  Stream := TStringStream.Create('');
  Writer := TNNetPLYWriter.CreateOnStream(Stream, plyAscii);
  try
    Writer.AddElement('face', 2);
    Writer.AddListProperty('vertex_indices', plyUInt8, plyInt32);
    Writer.AddProperty('quality', plyFloat32);
    Writer.WriteHeader;
    Writer.WriteRow([3, 0, 1, 2, 0.5]);
    Writer.WriteRow([0, 2]);
  finally
    Writer.Free;
  end;
  try
    AssertEquals('ascii text', 'ply'#10'format ascii 1.0'#10'element face 2'#10 +
      'property list uchar int vertex_indices'#10'property float quality'#10 +
      'end_header'#10'3 0 1 2 0.5'#10'0 2'#10, Stream.DataString);
    Stream.Position := 0;
    Reader := TNNetPLYReader.CreateOnStream(Stream);
    try
      Row := nil;
      AssertEquals('first row slots', 5, Reader.ReadFlatRow(Row));
      AssertEquals('length', 3, Row[0], 0);
      AssertEquals('last index', 2, Row[3], 0);
      AssertEquals('scalar after list', 0.5, Row[4], 0);
      AssertEquals('empty list slots', 2, Reader.ReadFlatRow(Row));
      AssertEquals('empty list scalar', 2, Row[1], 0);
    finally
      Reader.Free;
    end;
  finally
    Stream.Free;
  end;
  Stream := TStringStream.Create(QuadText);
  Mesh := TNNetTriangleMesh.Create;
  try
    Mesh.LoadFromPLYStream(Stream);
    AssertEquals('quad vertices', 4, Mesh.VertexCount);
    AssertFalse('no colours', Mesh.HasColors);
    AssertEquals('quad fan faces', 2, Mesh.FaceCount);
    Mesh.GetFace(1, A, B, C);
    AssertEquals('fan a', 0, A);
    AssertEquals('fan b', 2, B);
    AssertEquals('fan c', 3, C);
  finally
    Mesh.Free;
    Stream.Free;
  end;
end;

procedure TTestNeuralMesh.TestOBJText;
var
  Mesh: TNNetTriangleMesh;
  Stream: TStringStream;
begin
  Mesh := TNNetTriangleMesh.Create;
  Stream := TStringStream.Create('');
  try
    Mesh.AddVertex(0, 0, 0);
    Mesh.AddVertex(1.5, 0, -2);
    Mesh.AddVertex(0, 0.25, 1);
    Mesh.AddFace(0, 1, 2);
    Mesh.SaveToOBJStream(Stream);
    AssertEquals('plain', 'v 0 0 0'#10'v 1.5 0 -2'#10'v 0 0.25 1'#10'f 1 2 3'#10,
      Stream.DataString);
    Mesh.SetVertexColor(2, 1, 0.5, 0);
    Stream.Size := 0;
    Mesh.SaveToOBJStream(Stream);
    AssertEquals('coloured', 'v 0 0 0 0 0 0'#10'v 1.5 0 -2 0 0 0'#10 +
      'v 0 0.25 1 1 0.5 0'#10'f 1 2 3'#10, Stream.DataString);
  finally
    Stream.Free;
    Mesh.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralMesh);
end.
