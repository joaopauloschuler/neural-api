unit TestNeuralVolume;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, neuralvolume;

type
  TTestNeuralVolume = class(TTestCase)
  published
    procedure TestVolumeCreation;
    procedure TestVolumeFill;
    procedure TestVolumeDotProduct;
    procedure TestVolumeAddSub;
    procedure TestVolumeCopy;
    procedure TestVolumeSaveLoad;
  end;

implementation

procedure TTestNeuralVolume.TestVolumeCreation;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(32, 32, 3);
  try
    AssertEquals('SizeX should be 32', 32, V.SizeX);
    AssertEquals('SizeY should be 32', 32, V.SizeY);
    AssertEquals('Depth should be 3', 3, V.Depth);
    AssertEquals('Total size should be 3072', 3072, V.Size);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeFill;
var
  V: TNNetVolume;
  I: integer;
begin
  V := TNNetVolume.Create(10, 10, 1);
  try
    V.Fill(5.0);
    for I := 0 to V.Size - 1 do
      AssertEquals('All values should be 5.0', 5.0, V.Raw[I], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeDotProduct;
var
  V1, V2: TNNetVolume;
  DotProd: TNeuralFloat;
begin
  V1 := TNNetVolume.Create(4, 1, 1);
  V2 := TNNetVolume.Create(4, 1, 1);
  try
    V1.Fill(2.0);
    V2.Fill(3.0);
    DotProd := V1.DotProduct(V2);
    AssertEquals('Dot product of [2,2,2,2] and [3,3,3,3] should be 24', 24.0, DotProd, 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeAddSub;
var
  V1, V2: TNNetVolume;
begin
  V1 := TNNetVolume.Create(4, 1, 1);
  V2 := TNNetVolume.Create(4, 1, 1);
  try
    V1.Fill(5.0);
    V2.Fill(3.0);
    V1.Add(V2);
    AssertEquals('After adding, values should be 8.0', 8.0, V1.Raw[0], 0.0001);
    V1.Sub(V2);
    AssertEquals('After subtracting, values should be 5.0', 5.0, V1.Raw[0], 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeCopy;
var
  V1, V2: TNNetVolume;
begin
  V1 := TNNetVolume.Create(10, 10, 3);
  V2 := TNNetVolume.Create(10, 10, 3);
  try
    V1.RandomizeGaussian();
    V2.Copy(V1);
    AssertEquals('Copied volume should match', 0.0, V1.SumDiff(V2), 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeSaveLoad;
var
  V1, V2: TNNetVolume;
  SavedStr: string;
begin
  V1 := TNNetVolume.Create(5, 5, 2);
  V2 := TNNetVolume.Create(1, 1, 1);
  try
    V1.RandomizeGaussian();
    SavedStr := V1.SaveToString();
    V2.LoadFromString(SavedStr);
    AssertEquals('Loaded volume should match saved', 0.0, V1.SumDiff(V2), 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralVolume);

end.
