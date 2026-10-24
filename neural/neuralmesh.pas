unit neuralmesh;
{
  neuralmesh Free Pascal/Lazarus Library by Joao Paulo Schwarz Schuler
  Conscious Artificial Intelligence Project
  https://sourceforge.net/projects/cai/

  Indexed triangle meshes: TNNetTriangleMesh (vertices, optional per-vertex
  colours, triangle faces) with PLY and Wavefront OBJ export, and
  MarchingCubes (Lorensen & Cline 1987, Paul Bourke's 256-case tables), which
  extracts an iso surface from a density grid. Coded by Claude (AI).
}

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, bufstream, neuralvolume, neuralply;

type
  TNNetPoint3D = record
    X, Y, Z: TNeuralFloat;
  end;

  // Vertices and colours are stored xyz / rgb interleaved, faces as three
  // 0-based vertex indices; colours are 0..1 and exist once SetVertexColor ran.
  TNNetTriangleMesh = class
  private
    FVertices: TNeuralFloatDynArr;
    FColors: TNeuralFloatDynArr;
    FFaces: TNeuralIntegerArray;
    FVertexCount, FFaceCount: integer;
    FHasColors: boolean;
  public
    procedure Clear;
    function AddVertex(X, Y, Z: TNeuralFloat): integer;
    procedure AddFace(VertexA, VertexB, VertexC: integer);
    procedure SetVertexColor(VertexIndex: integer; Red, Green, Blue: TNeuralFloat);
    procedure GetVertex(VertexIndex: integer; out X, Y, Z: TNeuralFloat);
    procedure GetFace(FaceIndex: integer; out VertexA, VertexB, VertexC: integer);
    // Writes element vertex (x, y, z [, red, green, blue as uchar]) and element
    // face (list uchar int vertex_indices).
    procedure SaveToPLY(const FileName: string; pFormat: TNNetPLYFormat);
    procedure SaveToPLYStream(Stream: TStream; pFormat: TNNetPLYFormat);
    // Reads x/y/z, optional red/green/blue and vertex_indices (or vertex_index);
    // polygons with more than three corners are split into a triangle fan.
    procedure LoadFromPLY(const FileName: string);
    procedure LoadFromPLYStream(Stream: TStream);
    // Writes "v x y z [r g b]" and 1-based "f a b c" lines.
    procedure SaveToOBJ(const FileName: string);
    procedure SaveToOBJStream(Stream: TStream);
    property VertexCount: integer read FVertexCount;
    property FaceCount: integer read FFaceCount;
    property HasColors: boolean read FHasColors;
    property Vertices: TNeuralFloatDynArr read FVertices;
    property Colors: TNeuralFloatDynArr read FColors;
    property Faces: TNeuralIntegerArray read FFaces;
  end;

  function NNetPoint3D(X, Y, Z: TNeuralFloat): TNNetPoint3D;

  // Clears Mesh and fills it with the IsoLevel surface of the SizeX x SizeY x SizeZ
  // grid at Field (element offset X*StrideX + Y*StrideY + Z*StrideZ).
  procedure MarchingCubes(Field: TNeuralFloatPtr; SizeX, SizeY, SizeZ,
    StrideX, StrideY, StrideZ: integer; IsoLevel: TNeuralFloat;
    const Origin, Spacing: TNNetPoint3D; Mesh: TNNetTriangleMesh); overload;
  // Grid point (X, Y, D) of Field sits at Origin + (X, Y, D) * Spacing. Faces turn
  // counter-clockwise seen from the side where the field is below IsoLevel.
  procedure MarchingCubes(Field: TNNetVolume; IsoLevel: TNeuralFloat;
    const Origin, Spacing: TNNetPoint3D; Mesh: TNNetTriangleMesh); overload;

const
  // Corner c of a cell sits at offset (CubeCornerDX[c], CubeCornerDY[c], CubeCornerDZ[c]).
  CubeCornerDX: array[0..7] of integer = (0, 1, 1, 0, 0, 1, 1, 0);
  CubeCornerDY: array[0..7] of integer = (0, 0, 1, 1, 0, 0, 1, 1);
  CubeCornerDZ: array[0..7] of integer = (0, 0, 0, 0, 1, 1, 1, 1);
  // Edge e joins corner CubeEdgeLowCorner[e] to CubeEdgeHighCorner[e] along +axis.
  CubeEdgeLowCorner: array[0..11] of integer = (0, 1, 3, 0, 4, 5, 7, 4, 0, 1, 2, 3);
  CubeEdgeHighCorner: array[0..11] of integer = (1, 2, 2, 3, 5, 6, 6, 7, 4, 5, 6, 7);
  // Bit c of the case index is set when corner c is below the iso level; each row
  // lists edge triples, one per triangle, ended by -1.
  MarchingCubesTriangleTable: array[0..255, 0..15] of ShortInt = (
    ( -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1, -1),
    (  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1, -1),
    (  3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1, -1),
    (  3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1, -1),
    (  9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1),
    (  9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1),
    (  2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1, -1),
    (  8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    ( 11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1, -1),
    (  9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1),
    (  4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1, -1),
    (  3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1, -1),
    (  1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1, -1),
    (  4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1, -1),
    (  4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1, -1),
    (  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1, -1),
    (  1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1),
    (  5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1, -1),
    (  2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1, -1),
    (  9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1),
    (  0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1),
    (  2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1, -1),
    ( 10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1),
    (  4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1, -1),
    (  5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1, -1),
    (  5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1),
    (  9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1, -1),
    (  0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1),
    (  1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1, -1),
    ( 10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1, -1),
    (  8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1, -1),
    (  2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1),
    (  7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1),
    (  9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1, -1),
    (  2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1, -1),
    ( 11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1, -1),
    (  9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1, -1),
    (  5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0, -1),
    ( 11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0, -1),
    ( 11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    ( 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1),
    (  1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1, -1),
    (  9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1, -1),
    (  5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1, -1),
    (  2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    ( 11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1),
    (  0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1),
    (  5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1, -1),
    (  6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1, -1),
    (  0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1, -1),
    (  3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1, -1),
    (  6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1, -1),
    (  5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1),
    (  1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1),
    ( 10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1, -1),
    (  6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1),
    (  1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1, -1),
    (  8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1, -1),
    (  7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9, -1),
    (  3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1),
    (  5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1, -1),
    (  0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1),
    (  9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6, -1),
    (  8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1, -1),
    (  5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11, -1),
    (  0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7, -1),
    (  6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1, -1),
    ( 10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1),
    ( 10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1),
    (  8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1, -1),
    (  1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1, -1),
    (  3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1, -1),
    (  0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1),
    ( 10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1),
    (  0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1, -1),
    (  3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1, -1),
    (  6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1, -1),
    (  9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1, -1),
    (  8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1, -1),
    (  3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1, -1),
    (  6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1, -1),
    (  0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1, -1),
    ( 10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1, -1),
    ( 10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1),
    (  1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1, -1),
    (  2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9, -1),
    (  7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1, -1),
    (  7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1, -1),
    (  2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7, -1),
    (  1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11, -1),
    ( 11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1, -1),
    (  8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6, -1),
    (  0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1, -1),
    (  7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1),
    ( 10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1),
    (  2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1),
    (  6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1, -1),
    (  7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1, -1),
    (  2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1, -1),
    (  1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1, -1),
    ( 10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1, -1),
    ( 10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1, -1),
    (  0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1, -1),
    (  7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1, -1),
    (  6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1, -1),
    (  8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1, -1),
    (  9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1, -1),
    (  6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1, -1),
    (  4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1, -1),
    ( 10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3, -1),
    (  8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1),
    (  0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1, -1),
    (  1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1, -1),
    (  8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1, -1),
    ( 10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1, -1),
    (  4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3, -1),
    ( 10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1),
    (  5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1),
    ( 11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1, -1),
    (  9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1),
    (  6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1, -1),
    (  7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1, -1),
    (  3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6, -1),
    (  7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1, -1),
    (  9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1, -1),
    (  3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1, -1),
    (  6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8, -1),
    (  9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1, -1),
    (  1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4, -1),
    (  4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10, -1),
    (  7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1, -1),
    (  6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1, -1),
    (  3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1, -1),
    (  0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1, -1),
    (  6, 11,  3,  6,  3,  5,  5,  3,  1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1, -1),
    (  0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10, -1),
    ( 11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5, -1),
    (  6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1, -1),
    (  5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1, -1),
    (  9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1, -1),
    (  1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8, -1),
    (  1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6, -1),
    ( 10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, -1, -1, -1, -1),
    (  0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    ( 10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    ( 11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    ( 11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1, -1),
    (  5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1, -1),
    ( 10,  7,  5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1, -1),
    ( 11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1, -1),
    (  9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1, -1),
    (  7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2, -1),
    (  2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1),
    (  8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1, -1),
    (  9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, -1, -1, -1, -1),
    (  9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2, -1),
    (  1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1, -1),
    (  9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1, -1),
    (  9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1),
    (  5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1, -1),
    (  0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1, -1),
    ( 10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4, -1),
    (  2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1, -1),
    (  0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5,  1, 11, -1),
    (  0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5, -1),
    (  9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1, -1),
    (  5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1, -1),
    (  3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9, -1),
    (  5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, -1, -1, -1, -1),
    (  8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, -1, -1, -1, -1),
    (  9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1, -1),
    (  0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1, -1),
    (  1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1, -1),
    (  3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4, -1),
    (  4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1, -1),
    (  9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3, -1),
    ( 11,  7,  4, 11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1, -1),
    ( 11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1, -1),
    (  2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1, -1),
    (  9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7, -1),
    (  3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10, -1),
    (  1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  4,  9,  1,  4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1, -1),
    (  4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1, -1),
    (  4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  9, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1, -1),
    (  0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1, -1),
    (  3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1, -1),
    (  3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1, -1),
    (  0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1, -1),
    (  9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1, -1),
    (  1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (  0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    ( -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
  );

var
  // Bit e is set when edge e joins a corner below the iso level to one that is not.
  MarchingCubesEdgeMask: array[0..255] of integer;

implementation

const
  // 0/1: x edges on the low/high z slice, 2/3: y edges on the low/high slice, 4: z edges.
  CubeEdgeCache: array[0..11] of integer = (0, 2, 0, 2, 1, 3, 1, 3, 4, 4, 4, 4);
  CubeEdgeAxis: array[0..11] of integer = (0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2);
  MaxPLYIndexableVertexCount = 1 shl 24;

function NNetPoint3D(X, Y, Z: TNeuralFloat): TNNetPoint3D;
begin
  Result.X := X;
  Result.Y := Y;
  Result.Z := Z;
end;

{ TNNetTriangleMesh }

procedure TNNetTriangleMesh.Clear;
begin
  FVertexCount := 0;
  FFaceCount := 0;
  FHasColors := false;
end;

function TNNetTriangleMesh.AddVertex(X, Y, Z: TNeuralFloat): integer;
var
  Base: integer;
begin
  Base := FVertexCount * 3;
  if Length(FVertices) < Base + 3 then
  begin
    SetLength(FVertices, 2 * Base + 48);
    if FHasColors then SetLength(FColors, Length(FVertices));
  end;
  FVertices[Base] := X;
  FVertices[Base + 1] := Y;
  FVertices[Base + 2] := Z;
  if FHasColors then
  begin
    FColors[Base] := 0;
    FColors[Base + 1] := 0;
    FColors[Base + 2] := 0;
  end;
  Result := FVertexCount;
  Inc(FVertexCount);
end;

procedure TNNetTriangleMesh.AddFace(VertexA, VertexB, VertexC: integer);
var
  Base: integer;
begin
  Base := FFaceCount * 3;
  if Length(FFaces) < Base + 3 then SetLength(FFaces, 2 * Base + 48);
  FFaces[Base] := VertexA;
  FFaces[Base + 1] := VertexB;
  FFaces[Base + 2] := VertexC;
  Inc(FFaceCount);
end;

procedure TNNetTriangleMesh.SetVertexColor(VertexIndex: integer; Red, Green, Blue: TNeuralFloat);
var
  Base: integer;
begin
  if (VertexIndex < 0) or (VertexIndex >= FVertexCount) then
    raise ERangeError.CreateFmt('Mesh: vertex %d out of range', [VertexIndex]);
  if not FHasColors then
  begin
    SetLength(FColors, Length(FVertices));
    if FVertexCount > 0 then FillChar(FColors[0], FVertexCount * 3 * SizeOf(TNeuralFloat), 0);
    FHasColors := true;
  end;
  Base := VertexIndex * 3;
  FColors[Base] := Red;
  FColors[Base + 1] := Green;
  FColors[Base + 2] := Blue;
end;

procedure TNNetTriangleMesh.GetVertex(VertexIndex: integer; out X, Y, Z: TNeuralFloat);
var
  Base: integer;
begin
  Base := VertexIndex * 3;
  X := FVertices[Base];
  Y := FVertices[Base + 1];
  Z := FVertices[Base + 2];
end;

procedure TNNetTriangleMesh.GetFace(FaceIndex: integer; out VertexA, VertexB, VertexC: integer);
var
  Base: integer;
begin
  Base := FaceIndex * 3;
  VertexA := FFaces[Base];
  VertexB := FFaces[Base + 1];
  VertexC := FFaces[Base + 2];
end;

procedure TNNetTriangleMesh.SaveToPLY(const FileName: string; pFormat: TNNetPLYFormat);
var
  Stream: TFileStream;
begin
  Stream := TFileStream.Create(FileName, fmCreate);
  try
    SaveToPLYStream(Stream, pFormat);
  finally
    Stream.Free;
  end;
end;

function ColorToByte(Value: TNeuralFloat): integer;
begin
  Result := Round(Value * 255);
  if Result < 0 then Result := 0
  else if Result > 255 then Result := 255;
end;

procedure TNNetTriangleMesh.SaveToPLYStream(Stream: TStream; pFormat: TNNetPLYFormat);
var
  Writer: TNNetPLYWriter;
  VertexRow: array[0..5] of TNeuralFloat;
  FaceRow: array[0..3] of TNeuralFloat;
  VertexPos, MaxVertexPos, FacePos, MaxFacePos, Base: integer;
begin
  // WriteRow takes TNeuralFloat slots, which hold integers exactly up to 2^24.
  if FVertexCount > MaxPLYIndexableVertexCount then
    raise EWriteError.CreateFmt('Mesh: %d vertices exceed the PLY index limit', [FVertexCount]);
  Writer := TNNetPLYWriter.CreateOnStream(Stream, pFormat);
  try
    Writer.AddElement('vertex', FVertexCount);
    Writer.AddProperty('x', plyFloat32);
    Writer.AddProperty('y', plyFloat32);
    Writer.AddProperty('z', plyFloat32);
    if FHasColors then
    begin
      Writer.AddProperty('red', plyUInt8);
      Writer.AddProperty('green', plyUInt8);
      Writer.AddProperty('blue', plyUInt8);
    end;
    Writer.AddElement('face', FFaceCount);
    Writer.AddListProperty('vertex_indices', plyUInt8, plyInt32);
    Writer.WriteHeader;
    MaxVertexPos := FVertexCount - 1;
    for VertexPos := 0 to MaxVertexPos do
    begin
      Base := VertexPos * 3;
      VertexRow[0] := FVertices[Base];
      VertexRow[1] := FVertices[Base + 1];
      VertexRow[2] := FVertices[Base + 2];
      if FHasColors then
      begin
        VertexRow[3] := ColorToByte(FColors[Base]);
        VertexRow[4] := ColorToByte(FColors[Base + 1]);
        VertexRow[5] := ColorToByte(FColors[Base + 2]);
      end;
      Writer.WriteRow(VertexRow);
    end;
    FaceRow[0] := 3;
    MaxFacePos := FFaceCount - 1;
    for FacePos := 0 to MaxFacePos do
    begin
      Base := FacePos * 3;
      FaceRow[1] := FFaces[Base];
      FaceRow[2] := FFaces[Base + 1];
      FaceRow[3] := FFaces[Base + 2];
      Writer.WriteRow(FaceRow);
    end;
  finally
    Writer.Free;
  end;
end;

procedure TNNetTriangleMesh.LoadFromPLY(const FileName: string);
var
  Stream: TFileStream;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    LoadFromPLYStream(Stream);
  finally
    Stream.Free;
  end;
end;

procedure TNNetTriangleMesh.LoadFromPLYStream(Stream: TStream);
var
  Reader: TNNetPLYReader;
  Row: TNeuralFloatDynArr;
  ElementPos, RowPos, MaxRowPos, PropertyPos, MaxPropertyPos, SlotPos: integer;
  XSlot, YSlot, ZSlot, RedSlot, GreenSlot, BlueSlot, IndexSlot: integer;
  CornerCount, MaxFanPos, FanPos, NewVertex: integer;
  IsVertexElement, IsFaceElement: boolean;
begin
  Clear;
  Row := nil;
  Reader := TNNetPLYReader.CreateOnStream(Stream);
  try
    while Reader.CurrentElement < Length(Reader.Elements) do
    begin
      ElementPos := Reader.CurrentElement;
      IsVertexElement := Reader.Elements[ElementPos].Name = 'vertex';
      IsFaceElement := Reader.Elements[ElementPos].Name = 'face';
      // Slot positions hold only while no list precedes them; vertex rows have none.
      XSlot := Reader.PropertyIndex(ElementPos, 'x');
      YSlot := Reader.PropertyIndex(ElementPos, 'y');
      ZSlot := Reader.PropertyIndex(ElementPos, 'z');
      RedSlot := Reader.PropertyIndex(ElementPos, 'red');
      GreenSlot := Reader.PropertyIndex(ElementPos, 'green');
      BlueSlot := Reader.PropertyIndex(ElementPos, 'blue');
      if IsVertexElement and ((XSlot < 0) or (YSlot < 0) or (ZSlot < 0)) then
        raise EReadError.Create('Mesh PLY: vertex element lacks x, y or z');
      IsFaceElement := IsFaceElement and
        ((Reader.PropertyIndex(ElementPos, 'vertex_indices') >= 0) or
        (Reader.PropertyIndex(ElementPos, 'vertex_index') >= 0));
      MaxRowPos := Reader.Elements[ElementPos].Count - 1;
      MaxPropertyPos := High(Reader.Elements[ElementPos].Properties);
      for RowPos := 0 to MaxRowPos do
      begin
        Reader.ReadFlatRow(Row);
        if IsVertexElement then
        begin
          NewVertex := AddVertex(Row[XSlot], Row[YSlot], Row[ZSlot]);
          if (RedSlot >= 0) and (GreenSlot >= 0) and (BlueSlot >= 0) then
            SetVertexColor(NewVertex, Row[RedSlot] / 255, Row[GreenSlot] / 255,
              Row[BlueSlot] / 255);
        end
        else if IsFaceElement then
        begin
          SlotPos := 0;
          IndexSlot := -1;
          for PropertyPos := 0 to MaxPropertyPos do
            with Reader.Elements[ElementPos].Properties[PropertyPos] do
            begin
              if IsList and (IndexSlot < 0) and
                ((Name = 'vertex_indices') or (Name = 'vertex_index')) then
                IndexSlot := SlotPos;
              if IsList then Inc(SlotPos, 1 + Round(Row[SlotPos]))
              else Inc(SlotPos);
            end;
          CornerCount := Round(Row[IndexSlot]);
          MaxFanPos := CornerCount - 2;
          for FanPos := 1 to MaxFanPos do
            AddFace(Round(Row[IndexSlot + 1]), Round(Row[IndexSlot + 1 + FanPos]),
              Round(Row[IndexSlot + 2 + FanPos]));
        end;
      end;
    end;
  finally
    Reader.Free;
  end;
end;

procedure TNNetTriangleMesh.SaveToOBJ(const FileName: string);
var
  Stream: TFileStream;
begin
  Stream := TFileStream.Create(FileName, fmCreate);
  try
    SaveToOBJStream(Stream);
  finally
    Stream.Free;
  end;
end;

procedure TNNetTriangleMesh.SaveToOBJStream(Stream: TStream);
var
  Buffered: TWriteBufStream;
  FormatSettings: TFormatSettings;
  VertexPos, MaxVertexPos, FacePos, MaxFacePos, Base: integer;
  Line: string;

  function FloatText(Value: TNeuralFloat): string;
  begin
    Result := FloatToStrF(Value, ffGeneral, 9, 0, FormatSettings);
  end;

begin
  FormatSettings := DefaultFormatSettings;
  FormatSettings.DecimalSeparator := '.';
  FormatSettings.ThousandSeparator := #0;
  Buffered := TWriteBufStream.Create(Stream, 65536);
  try
    MaxVertexPos := FVertexCount - 1;
    for VertexPos := 0 to MaxVertexPos do
    begin
      Base := VertexPos * 3;
      Line := 'v ' + FloatText(FVertices[Base]) + ' ' + FloatText(FVertices[Base + 1]) +
        ' ' + FloatText(FVertices[Base + 2]);
      if FHasColors then
        Line := Line + ' ' + FloatText(FColors[Base]) + ' ' +
          FloatText(FColors[Base + 1]) + ' ' + FloatText(FColors[Base + 2]);
      Line := Line + #10;
      Buffered.WriteBuffer(Line[1], Length(Line));
    end;
    MaxFacePos := FFaceCount - 1;
    for FacePos := 0 to MaxFacePos do
    begin
      Base := FacePos * 3;
      Line := 'f ' + IntToStr(FFaces[Base] + 1) + ' ' + IntToStr(FFaces[Base + 1] + 1) +
        ' ' + IntToStr(FFaces[Base + 2] + 1) + #10;
      Buffered.WriteBuffer(Line[1], Length(Line));
    end;
  finally
    Buffered.Free;
  end;
end;

{ Marching cubes }

procedure MarchingCubes(Field: TNeuralFloatPtr; SizeX, SizeY, SizeZ,
  StrideX, StrideY, StrideZ: integer; IsoLevel: TNeuralFloat;
  const Origin, Spacing: TNNetPoint3D; Mesh: TNNetTriangleMesh);
var
  EdgeCaches: array[0..4] of TNeuralIntegerArray;
  SwapCache: TNeuralIntegerArray;
  CornerOffset: array[0..7] of integer;
  CornerValue: array[0..7] of TNeuralFloat;
  EdgeVertex: array[0..11] of integer;
  SliceCellCount, MaxCellX, MaxCellY, MaxCellZ, CellX, CellY, CellZ: integer;
  CachePos, SliceBase, CellBase, CornerPos, CubeIndex, EdgeBits, EdgePos: integer;
  CacheCell, LowCorner, HighCorner, TrianglePos: integer;
  Fraction, VertexX, VertexY, VertexZ: TNeuralFloat;
begin
  Mesh.Clear;
  MaxCellX := SizeX - 2;
  MaxCellY := SizeY - 2;
  MaxCellZ := SizeZ - 2;
  if (MaxCellX < 0) or (MaxCellY < 0) or (MaxCellZ < 0) then Exit;
  SliceCellCount := SizeX * SizeY;
  for CachePos := 0 to 4 do SetLength(EdgeCaches[CachePos], SliceCellCount);
  FillDWord(EdgeCaches[0][0], SliceCellCount, DWord(-1));
  FillDWord(EdgeCaches[2][0], SliceCellCount, DWord(-1));
  for CornerPos := 0 to 7 do
    CornerOffset[CornerPos] := CubeCornerDX[CornerPos] * StrideX +
      CubeCornerDY[CornerPos] * StrideY + CubeCornerDZ[CornerPos] * StrideZ;
  for CellZ := 0 to MaxCellZ do
  begin
    FillDWord(EdgeCaches[1][0], SliceCellCount, DWord(-1));
    FillDWord(EdgeCaches[3][0], SliceCellCount, DWord(-1));
    FillDWord(EdgeCaches[4][0], SliceCellCount, DWord(-1));
    SliceBase := CellZ * StrideZ;
    for CellY := 0 to MaxCellY do
    begin
      CellBase := SliceBase + CellY * StrideY;
      for CellX := 0 to MaxCellX do
      begin
        CubeIndex := 0;
        for CornerPos := 0 to 7 do
        begin
          CornerValue[CornerPos] := Field[CellBase + CornerOffset[CornerPos]];
          if CornerValue[CornerPos] < IsoLevel then
            CubeIndex := CubeIndex or (1 shl CornerPos);
        end;
        EdgeBits := MarchingCubesEdgeMask[CubeIndex];
        if EdgeBits <> 0 then
        begin
          for EdgePos := 0 to 11 do
            if (EdgeBits and (1 shl EdgePos)) <> 0 then
            begin
              LowCorner := CubeEdgeLowCorner[EdgePos];
              CacheCell := (CellY + CubeCornerDY[LowCorner]) * SizeX + CellX +
                CubeCornerDX[LowCorner];
              EdgeVertex[EdgePos] := EdgeCaches[CubeEdgeCache[EdgePos]][CacheCell];
              if EdgeVertex[EdgePos] < 0 then
              begin
                HighCorner := CubeEdgeHighCorner[EdgePos];
                Fraction := (IsoLevel - CornerValue[LowCorner]) /
                  (CornerValue[HighCorner] - CornerValue[LowCorner]);
                VertexX := CellX + CubeCornerDX[LowCorner];
                VertexY := CellY + CubeCornerDY[LowCorner];
                VertexZ := CellZ + CubeCornerDZ[LowCorner];
                case CubeEdgeAxis[EdgePos] of
                  0: VertexX := VertexX + Fraction;
                  1: VertexY := VertexY + Fraction;
                else
                  VertexZ := VertexZ + Fraction;
                end;
                EdgeVertex[EdgePos] := Mesh.AddVertex(Origin.X + VertexX * Spacing.X,
                  Origin.Y + VertexY * Spacing.Y, Origin.Z + VertexZ * Spacing.Z);
                EdgeCaches[CubeEdgeCache[EdgePos]][CacheCell] := EdgeVertex[EdgePos];
              end;
            end;
          TrianglePos := 0;
          while MarchingCubesTriangleTable[CubeIndex, TrianglePos] >= 0 do
          begin
            Mesh.AddFace(EdgeVertex[MarchingCubesTriangleTable[CubeIndex, TrianglePos]],
              EdgeVertex[MarchingCubesTriangleTable[CubeIndex, TrianglePos + 1]],
              EdgeVertex[MarchingCubesTriangleTable[CubeIndex, TrianglePos + 2]]);
            Inc(TrianglePos, 3);
          end;
        end;
        Inc(CellBase, StrideX);
      end;
    end;
    SwapCache := EdgeCaches[0];
    EdgeCaches[0] := EdgeCaches[1];
    EdgeCaches[1] := SwapCache;
    SwapCache := EdgeCaches[2];
    EdgeCaches[2] := EdgeCaches[3];
    EdgeCaches[3] := SwapCache;
  end;
end;

procedure MarchingCubes(Field: TNNetVolume; IsoLevel: TNeuralFloat;
  const Origin, Spacing: TNNetPoint3D; Mesh: TNNetTriangleMesh);
begin
  if Field.Size = 0 then
  begin
    Mesh.Clear;
    Exit;
  end;
  MarchingCubes(TNeuralFloatPtr(@Field.FData[0]), Field.SizeX, Field.SizeY, Field.Depth,
    Field.Depth, Field.SizeX * Field.Depth, 1, IsoLevel, Origin, Spacing, Mesh);
end;

procedure InitMarchingCubesEdgeMask;
var
  CubeIndex, EdgePos, LowBit, HighBit: integer;
begin
  for CubeIndex := 0 to 255 do
  begin
    MarchingCubesEdgeMask[CubeIndex] := 0;
    for EdgePos := 0 to 11 do
    begin
      LowBit := (CubeIndex shr CubeEdgeLowCorner[EdgePos]) and 1;
      HighBit := (CubeIndex shr CubeEdgeHighCorner[EdgePos]) and 1;
      if LowBit <> HighBit then
        MarchingCubesEdgeMask[CubeIndex] := MarchingCubesEdgeMask[CubeIndex] or (1 shl EdgePos);
    end;
  end;
end;

initialization
  InitMarchingCubesEdgeMask;
end.
