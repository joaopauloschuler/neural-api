unit TestNeuralVolumePairs;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, neuralvolume;

type
  TTestNeuralVolumePairs = class(TTestCase)
  published
    // TNNetVolumePair tests
    procedure TestVolumePairCreation;
    procedure TestVolumePairCreationWithVolumes;
    procedure TestVolumePairCopying;
    procedure TestVolumePairProperties;
    
    // TNNetVolumeList tests
    procedure TestVolumeListCreation;
    procedure TestVolumeListAddAndCount;
    procedure TestVolumeListGetRandom;
    procedure TestVolumeListCopy;
    procedure TestVolumeListAddVolume;
    
    // TNNetVolumePairList tests
    procedure TestVolumePairListCreation;
    procedure TestVolumePairListAddAndCount;
    procedure TestVolumePairListGetRandom;

    // Mixup data augmentation tests
    procedure TestMixVolumesConvexCombination;
    procedure TestMixupLambdaOneReproducesFirst;
    procedure TestMixupLengthAndNoMutation;
    procedure TestRandomBetaUniformRange;

    // CutMix data augmentation tests
    procedure TestCutMixBoxGeometry;
    procedure TestCutMixPasteAndTargetMix;
    procedure TestCutMixLengthAndNoMutation;
  end;

implementation

procedure TTestNeuralVolumePairs.TestVolumePairCreation;
var
  Pair: TNNetVolumePair;
begin
  Pair := TNNetVolumePair.Create();
  try
    AssertTrue('Pair should be created', Pair <> nil);
    AssertTrue('Pair.A should exist', Pair.A <> nil);
    AssertTrue('Pair.B should exist', Pair.B <> nil);
  finally
    Pair.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestVolumePairCreationWithVolumes;
var
  Pair: TNNetVolumePair;
  VA, VB: TNNetVolume;
begin
  VA := TNNetVolume.Create(4, 4, 1);
  VB := TNNetVolume.Create(2, 1, 1);
  VA.Fill(1.0);
  VB.Fill(2.0);
  
  // Note: Create takes ownership of the volumes
  Pair := TNNetVolumePair.Create(VA, VB);
  try
    AssertTrue('Pair should be created', Pair <> nil);
    AssertEquals('Pair.A should have size 16', 16, Pair.A.Size);
    AssertEquals('Pair.B should have size 2', 2, Pair.B.Size);
    AssertEquals('Pair.A[0] should be 1.0', 1.0, Pair.A.Raw[0], 0.0001);
    AssertEquals('Pair.B[0] should be 2.0', 2.0, Pair.B.Raw[0], 0.0001);
  finally
    Pair.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestVolumePairCopying;
var
  Pair: TNNetVolumePair;
  VA, VB: TNNetVolume;
begin
  VA := TNNetVolume.Create(4, 1, 1);
  VB := TNNetVolume.Create(2, 1, 1);
  VA.Fill(3.0);
  VB.Fill(4.0);
  
  // CreateCopying creates copies, so we need to free the originals
  Pair := TNNetVolumePair.CreateCopying(VA, VB);
  try
    AssertEquals('Pair.A[0] should be 3.0', 3.0, Pair.A.Raw[0], 0.0001);
    AssertEquals('Pair.B[0] should be 4.0', 4.0, Pair.B.Raw[0], 0.0001);
    
    // Modify originals to confirm copies were made
    VA.Fill(5.0);
    VB.Fill(6.0);
    
    AssertEquals('Pair.A[0] should still be 3.0', 3.0, Pair.A.Raw[0], 0.0001);
    AssertEquals('Pair.B[0] should still be 4.0', 4.0, Pair.B.Raw[0], 0.0001);
  finally
    Pair.Free;
    VA.Free;
    VB.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestVolumePairProperties;
var
  Pair: TNNetVolumePair;
begin
  Pair := TNNetVolumePair.Create();
  try
    // I and O are aliases for A and B (Input/Output)
    AssertTrue('Pair.I should equal Pair.A', Pair.I = Pair.A);
    AssertTrue('Pair.O should equal Pair.B', Pair.O = Pair.B);
  finally
    Pair.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestVolumeListCreation;
var
  VL: TNNetVolumeList;
begin
  VL := TNNetVolumeList.Create();
  try
    AssertTrue('Volume list should be created', VL <> nil);
    AssertEquals('Volume list should be empty', 0, VL.Count);
  finally
    VL.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestVolumeListAddAndCount;
var
  VL: TNNetVolumeList;
  V: TNNetVolume;
begin
  VL := TNNetVolumeList.Create();
  try
    V := TNNetVolume.Create(4, 4, 1);
    V.Fill(1.0);
    VL.Add(V);
    
    AssertEquals('Volume list should have 1 item', 1, VL.Count);
    AssertEquals('Item value should be 1.0', 1.0, VL[0].Raw[0], 0.0001);
    
    V := TNNetVolume.Create(4, 4, 1);
    V.Fill(2.0);
    VL.Add(V);
    
    AssertEquals('Volume list should have 2 items', 2, VL.Count);
    AssertEquals('Second item value should be 2.0', 2.0, VL[1].Raw[0], 0.0001);
  finally
    VL.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestVolumeListGetRandom;
var
  VL: TNNetVolumeList;
  V: TNNetVolume;
  I: integer;
begin
  VL := TNNetVolumeList.Create();
  try
    // Add 5 volumes with distinct values
    for I := 0 to 4 do
    begin
      V := TNNetVolume.Create(1, 1, 1);
      V.Raw[0] := I * 1.0;
      VL.Add(V);
    end;
    
    // Get item by fixed index should work
    V := VL[2];
    AssertTrue('Volume at index 2 should exist', V <> nil);
    AssertEquals('Volume at index 2 should have value 2.0', 2.0, V.Raw[0], 0.0001);
    
    // Test that all items are accessible
    for I := 0 to 4 do
    begin
      V := VL[I];
      AssertEquals('Volume value should match index', I * 1.0, V.Raw[0], 0.0001);
    end;
  finally
    VL.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestVolumeListCopy;
var
  VL1, VL2: TNNetVolumeList;
  V: TNNetVolume;
  I: integer;
begin
  VL1 := TNNetVolumeList.Create();
  VL2 := TNNetVolumeList.Create();
  try
    // Add volumes to VL1
    for I := 0 to 2 do
    begin
      V := TNNetVolume.Create(2, 2, 1);
      V.Fill(I * 1.0);
      VL1.Add(V);
    end;
    
    // Copy volumes using AddCopy
    for I := 0 to VL1.Count - 1 do
    begin
      VL2.AddCopy(VL1[I]);
    end;
    
    AssertEquals('VL2 should have 3 items', 3, VL2.Count);
    AssertEquals('VL2[0] should have value 0.0', 0.0, VL2[0].Raw[0], 0.0001);
    AssertEquals('VL2[1] should have value 1.0', 1.0, VL2[1].Raw[0], 0.0001);
    AssertEquals('VL2[2] should have value 2.0', 2.0, VL2[2].Raw[0], 0.0001);
    
    // Modify VL1 to confirm copies were made
    VL1[0].Fill(100.0);
    AssertEquals('VL2[0] should still be 0.0', 0.0, VL2[0].Raw[0], 0.0001);
  finally
    VL1.Free;
    VL2.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestVolumeListAddVolume;
var
  VL: TNNetVolumeList;
  V: TNNetVolume;
begin
  VL := TNNetVolumeList.Create();
  V := TNNetVolume.Create(4, 4, 1);
  try
    V.Fill(5.0);
    
    // AddCopy should add a copy of the volume
    VL.AddCopy(V);
    
    AssertEquals('Volume list should have 1 item', 1, VL.Count);
    
    // Modify original to confirm copy was made
    V.Fill(10.0);
    AssertEquals('List item should still be 5.0', 5.0, VL[0].Raw[0], 0.0001);
  finally
    V.Free;
    VL.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestVolumePairListCreation;
var
  PL: TNNetVolumePairList;
begin
  PL := TNNetVolumePairList.Create();
  try
    AssertTrue('Pair list should be created', PL <> nil);
    AssertEquals('Pair list should be empty', 0, PL.Count);
  finally
    PL.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestVolumePairListAddAndCount;
var
  PL: TNNetVolumePairList;
  Pair: TNNetVolumePair;
begin
  PL := TNNetVolumePairList.Create();
  try
    Pair := TNNetVolumePair.Create();
    Pair.A.ReSize(4, 1, 1);
    Pair.B.ReSize(2, 1, 1);
    Pair.A.Fill(1.0);
    Pair.B.Fill(2.0);
    PL.Add(Pair);
    
    AssertEquals('Pair list should have 1 item', 1, PL.Count);
    AssertEquals('Pair.A[0] should be 1.0', 1.0, PL[0].A.Raw[0], 0.0001);
    AssertEquals('Pair.B[0] should be 2.0', 2.0, PL[0].B.Raw[0], 0.0001);
  finally
    PL.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestVolumePairListGetRandom;
var
  PL: TNNetVolumePairList;
  Pair: TNNetVolumePair;
  I, Idx: integer;
begin
  PL := TNNetVolumePairList.Create();
  try
    // Add 5 pairs
    for I := 0 to 4 do
    begin
      Pair := TNNetVolumePair.Create();
      Pair.A.ReSize(1, 1, 1);
      Pair.B.ReSize(1, 1, 1);
      Pair.A.Raw[0] := I * 1.0;
      Pair.B.Raw[0] := I * 2.0;
      PL.Add(Pair);
    end;
    
    // Get item by random index should work
    Idx := Random(5);
    Pair := PL[Idx];
    AssertTrue('Pair at random index should exist', Pair <> nil);
    AssertEquals('Pair A value should match index', Idx * 1.0, Pair.A.Raw[0], 0.0001);
  finally
    PL.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestMixVolumesConvexCombination;
var
  A, B, Output: TNNetVolume;
begin
  A := TNNetVolume.Create(3, 1, 1);
  B := TNNetVolume.Create(3, 1, 1);
  Output := TNNetVolume.Create();
  try
    A.Raw[0] := 1.0; A.Raw[1] := 2.0;  A.Raw[2] := 3.0;
    B.Raw[0] := 5.0; B.Raw[1] := 10.0; B.Raw[2] := 15.0;

    // Lambda = 1 reproduces A bit-for-bit.
    MixVolumes(Output, A, B, 1.0);
    AssertEquals('lambda=1 out[0]', 1.0, Output.Raw[0], 0.0);
    AssertEquals('lambda=1 out[1]', 2.0, Output.Raw[1], 0.0);
    AssertEquals('lambda=1 out[2]', 3.0, Output.Raw[2], 0.0);

    // Lambda = 0 reproduces B bit-for-bit.
    MixVolumes(Output, A, B, 0.0);
    AssertEquals('lambda=0 out[0]', 5.0,  Output.Raw[0], 0.0);
    AssertEquals('lambda=0 out[1]', 10.0, Output.Raw[1], 0.0);
    AssertEquals('lambda=0 out[2]', 15.0, Output.Raw[2], 0.0);

    // Known lambda: exact convex combination 0.25*A + 0.75*B.
    MixVolumes(Output, A, B, 0.25);
    AssertEquals('lambda=.25 out[0]', 0.25*1.0 + 0.75*5.0,  Output.Raw[0], 0.0001);
    AssertEquals('lambda=.25 out[1]', 0.25*2.0 + 0.75*10.0, Output.Raw[1], 0.0001);
    AssertEquals('lambda=.25 out[2]', 0.25*3.0 + 0.75*15.0, Output.Raw[2], 0.0001);
  finally
    A.Free; B.Free; Output.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestMixupLambdaOneReproducesFirst;
var
  PL, Mixed: TNNetVolumePairList;
  Pair: TNNetVolumePair;
  I: integer;
begin
  PL := TNNetVolumePairList.Create();
  try
    for I := 0 to 5 do
    begin
      Pair := TNNetVolumePair.Create();
      Pair.A.ReSize(2, 1, 1);
      Pair.B.ReSize(2, 1, 1);
      Pair.A.Raw[0] := I * 1.0;  Pair.A.Raw[1] := I * 1.0 + 0.5;
      Pair.B.Raw[0] := I * 10.0; Pair.B.Raw[1] := I * 10.0 + 0.5;
      PL.Add(Pair);
    end;

    // FixedLambda=1 must reproduce each original pair exactly, regardless
    // of the random partner permutation.
    Mixed := CreateMixedVolumePairList(PL, 1.0, 1.0);
    try
      AssertEquals('length preserved', PL.Count, Mixed.Count);
      for I := 0 to PL.Count - 1 do
      begin
        AssertEquals('A[0] reproduced', PL[I].A.Raw[0], Mixed[I].A.Raw[0], 0.0);
        AssertEquals('A[1] reproduced', PL[I].A.Raw[1], Mixed[I].A.Raw[1], 0.0);
        AssertEquals('B[0] reproduced', PL[I].B.Raw[0], Mixed[I].B.Raw[0], 0.0);
        AssertEquals('B[1] reproduced', PL[I].B.Raw[1], Mixed[I].B.Raw[1], 0.0);
      end;
    finally
      Mixed.Free;
    end;
  finally
    PL.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestMixupLengthAndNoMutation;
var
  PL, Mixed: TNNetVolumePairList;
  Pair: TNNetVolumePair;
  I: integer;
begin
  PL := TNNetVolumePairList.Create();
  try
    for I := 0 to 9 do
    begin
      Pair := TNNetVolumePair.Create();
      Pair.A.ReSize(4, 1, 1);
      Pair.B.ReSize(2, 1, 1);
      Pair.A.Fill(I * 1.0);
      Pair.B.Fill(I * 2.0);
      PL.Add(Pair);
    end;

    Mixed := CreateMixedVolumePairList(PL, 0.4);
    try
      // Output list has the expected length.
      AssertEquals('Mixed length', 10, Mixed.Count);
      // Inputs are not mutated: original values intact.
      for I := 0 to 9 do
      begin
        AssertEquals('orig A unchanged', I * 1.0, PL[I].A.Raw[0], 0.0);
        AssertEquals('orig B unchanged', I * 2.0, PL[I].B.Raw[0], 0.0);
      end;
      // Mixed volumes have the right shapes.
      AssertEquals('Mixed A size', 4, Mixed[0].A.Size);
      AssertEquals('Mixed B size', 2, Mixed[0].B.Size);
    finally
      Mixed.Free;
    end;
  finally
    PL.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestRandomBetaUniformRange;
var
  I: integer;
  V, MinV, MaxV: TNeuralFloat;
begin
  RandSeed := 424242;
  MinV := 1.0; MaxV := 0.0;
  // Beta(1,1) == Uniform(0,1); all draws must lie inside [0,1].
  for I := 0 to 999 do
  begin
    V := RandomBetaValue(1.0);
    AssertTrue('Beta in [0,1] lower', V >= 0.0);
    AssertTrue('Beta in [0,1] upper', V <= 1.0);
    if V < MinV then MinV := V;
    if V > MaxV then MaxV := V;
  end;
  // Over 1000 uniform draws we expect a wide spread.
  AssertTrue('Beta(1,1) spans low values', MinV < 0.1);
  AssertTrue('Beta(1,1) spans high values', MaxV > 0.9);

  // General alpha: a Beta(2,2) sample must still be a valid probability.
  for I := 0 to 99 do
  begin
    V := RandomBetaValue(2.0);
    AssertTrue('Beta(2,2) in [0,1] lower', V >= 0.0);
    AssertTrue('Beta(2,2) in [0,1] upper', V <= 1.0);
  end;
end;

procedure TTestNeuralVolumePairs.TestCutMixBoxGeometry;
var
  X0, Y0, BoxW, BoxH: integer;
begin
  // Lambda=1 -> cut ratio 0 -> empty box (pure first image).
  ComputeCutMixBox(8, 8, 1.0, 0.5, 0.5, X0, Y0, BoxW, BoxH);
  AssertEquals('lambda=1 -> zero box width', 0, BoxW);
  AssertEquals('lambda=1 -> zero box height', 0, BoxH);

  // Lambda=0 -> cut ratio 1 -> full image box.
  ComputeCutMixBox(8, 8, 0.0, 0.5, 0.5, X0, Y0, BoxW, BoxH);
  AssertEquals('lambda=0 -> full box width', 8, BoxW);
  AssertEquals('lambda=0 -> full box height', 8, BoxH);

  // Centered box of half the area (r = sqrt(0.75)): stays inside bounds.
  ComputeCutMixBox(8, 8, 0.25, 0.5, 0.5, X0, Y0, BoxW, BoxH);
  AssertTrue('box within X', (X0 >= 0) and (X0 + BoxW <= 8));
  AssertTrue('box within Y', (Y0 >= 0) and (Y0 + BoxH <= 8));
  AssertTrue('box has positive area', (BoxW > 0) and (BoxH > 0));

  // Center at a corner: box must still be clamped to the image.
  ComputeCutMixBox(8, 8, 0.0, 0.0, 0.0, X0, Y0, BoxW, BoxH);
  AssertTrue('corner box within X', (X0 >= 0) and (X0 + BoxW <= 8));
  AssertTrue('corner box within Y', (Y0 >= 0) and (Y0 + BoxH <= 8));
end;

procedure TTestNeuralVolumePairs.TestCutMixPasteAndTargetMix;
var
  PL, Cut: TNNetVolumePairList;
  Pair: TNNetVolumePair;
  I, X, Y, D, Inside, Outside, ExpectInside: integer;
  W, H, Depth: integer;
  WA, WB, LambdaAdj: TNeuralFloat;
begin
  W := 4; H := 4; Depth := 2;
  PL := TNNetVolumePairList.Create();
  try
    // Two solid-color image pairs with distinct, known fills.
    for I := 0 to 1 do
    begin
      Pair := TNNetVolumePair.Create();
      Pair.A.ReSize(W, H, Depth);
      Pair.B.ReSize(2, 1, 1);
      if I = 0 then
      begin
        Pair.A.Fill(1.0);                 // image A solid 1.0
        Pair.B.Raw[0] := 1.0; Pair.B.Raw[1] := 0.0;  // one-hot class 0
      end
      else
      begin
        Pair.A.Fill(7.0);                 // image B solid 7.0
        Pair.B.Raw[0] := 0.0; Pair.B.Raw[1] := 1.0;  // one-hot class 1
      end;
      PL.Add(Pair);
    end;

    // The two source images are SOLID colors, so regardless of where the
    // random box lands: every pasted pixel equals the partner's color and
    // every other pixel keeps the self color. With RandSeed:=9 the Fisher-
    // Yates step swaps the pair (out0's partner is image B = 7.0, class 1;
    // out1's partner is image A = 1.0, class 0) AND the box center lands near
    // the middle so a non-empty rectangle is actually pasted. The mixed target
    // must weight the partner by the TRUE pasted-area fraction we count from
    // the pixels themselves.
    RandSeed := 9;
    Cut := CreateCutMixVolumePairList(PL, 1.0, 0.5);
    try
      AssertEquals('length preserved', PL.Count, Cut.Count);

      // ---- output 0: self=image A (1.0/class0), partner=image B (7.0/class1)
      Inside := 0; Outside := 0;
      for X := 0 to W - 1 do
        for Y := 0 to H - 1 do
          for D := 0 to Depth - 1 do
            if Cut[0].A[X, Y, D] = 7.0 then Inc(Inside)         // pasted partner
            else if Cut[0].A[X, Y, D] = 1.0 then Inc(Outside)   // kept self
            else AssertTrue('out0 pixel is one of the two solid colors', False);
      AssertEquals('out0 every pixel accounted for', W * H * Depth, Inside + Outside);
      AssertTrue('out0 actually pasted a non-empty box', Inside > 0);
      AssertTrue('out0 kept some of its own pixels', Outside > 0);
      // True partner-area fraction from the pixels themselves.
      WB := Inside / (W * H * Depth);            // partner weight (class 1)
      WA := 1.0 - WB;                            // self weight   (class 0)
      AssertEquals('out0 target class0 = self area', WA, Cut[0].B.Raw[0], 0.0001);
      AssertEquals('out0 target class1 = partner area', WB, Cut[0].B.Raw[1], 0.0001);
      AssertEquals('out0 target weights sum to 1',
        1.0, Cut[0].B.Raw[0] + Cut[0].B.Raw[1], 0.0001);

      // ---- output 1: self=image B (7.0/class1), partner=image A (1.0/class0)
      Inside := 0; Outside := 0;
      for X := 0 to W - 1 do
        for Y := 0 to H - 1 do
          for D := 0 to Depth - 1 do
            if Cut[1].A[X, Y, D] = 1.0 then Inc(Inside)         // pasted partner
            else if Cut[1].A[X, Y, D] = 7.0 then Inc(Outside)   // kept self
            else AssertTrue('out1 pixel is one of the two solid colors', False);
      AssertEquals('out1 every pixel accounted for', W * H * Depth, Inside + Outside);
      WB := Inside / (W * H * Depth);            // partner weight (class 0)
      WA := 1.0 - WB;                            // self weight   (class 1)
      AssertEquals('out1 target class1 = self area', WA, Cut[1].B.Raw[1], 0.0001);
      AssertEquals('out1 target class0 = partner area', WB, Cut[1].B.Raw[0], 0.0001);
    finally
      Cut.Free;
    end;

    // Partial paste via the deterministic box helper: lambda=0.75 -> r=0.5 ->
    // 2x2 box centered in a 4x4 image, fully in-bounds. Pasted area = 4 of 16
    // pixels, so the area fraction (1 - LambdaAdj) = 4/16 = 0.25 exactly.
    ComputeCutMixBox(W, H, 0.75, 0.5, 0.5, X, Y, Inside, Outside);
    AssertEquals('2x2 box width', 2, Inside);
    AssertEquals('2x2 box height', 2, Outside);
    ExpectInside := Inside * Outside;  // box area in pixels
    LambdaAdj := 1.0 - ExpectInside / (W * H);
    AssertEquals('area-fraction LambdaAdj = 0.75', 0.75, LambdaAdj, 0.0001);
    WA := LambdaAdj;            // weight on self target
    WB := 1.0 - LambdaAdj;     // weight on partner target
    AssertEquals('partner-area weight = 0.25', 0.25, WB, 0.0001);
    AssertEquals('target weights sum to 1', 1.0, WA + WB, 0.0001);
  finally
    PL.Free;
  end;
end;

procedure TTestNeuralVolumePairs.TestCutMixLengthAndNoMutation;
var
  PL, Cut: TNNetVolumePairList;
  Pair: TNNetVolumePair;
  I: integer;
begin
  PL := TNNetVolumePairList.Create();
  try
    for I := 0 to 9 do
    begin
      Pair := TNNetVolumePair.Create();
      Pair.A.ReSize(4, 4, 3);
      Pair.B.ReSize(2, 1, 1);
      Pair.A.Fill(I * 1.0);
      Pair.B.Raw[0] := I * 1.0; Pair.B.Raw[1] := I * 2.0;
      PL.Add(Pair);
    end;

    RandSeed := 99;
    Cut := CreateCutMixVolumePairList(PL, 0.4);
    try
      // Output list has the expected length.
      AssertEquals('CutMix length', 10, Cut.Count);
      // Inputs are not mutated: original values intact (owns its own copies).
      for I := 0 to 9 do
      begin
        AssertEquals('orig A unchanged', I * 1.0, PL[I].A[0, 0, 0], 0.0);
        AssertEquals('orig B unchanged', I * 2.0, PL[I].B.Raw[1], 0.0);
      end;
      // Output shapes match the input geometry.
      AssertEquals('Cut A SizeX', 4, Cut[0].A.SizeX);
      AssertEquals('Cut A SizeY', 4, Cut[0].A.SizeY);
      AssertEquals('Cut A Depth', 3, Cut[0].A.Depth);
      AssertEquals('Cut B size', 2, Cut[0].B.Size);
      // Mutating the output must not touch the input copies.
      Cut[0].A.Fill(-123.0);
      AssertEquals('input still intact after output mutate',
        0.0, PL[0].A[0, 0, 0], 0.0);
    finally
      Cut.Free;
    end;
  finally
    PL.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralVolumePairs);

end.
