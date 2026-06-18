unit TestNeuralAugment;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralvolume, neuraldatasets;

type
  TTestNeuralAugment = class(TTestCase)
  private
    // Builds a small deterministic RGB test image in the neuronal [-2..2]
    // domain (a smooth gradient so geometric ops are observable).
    function MakeImage(W, H, Dep: integer): TNNetVolume;
    function VolumesEqual(A, B: TNNetVolume): boolean;
    function VolumesFinite(A: TNNetVolume): boolean;
  published
    // Op-bank shape / finiteness / identity invariants.
    procedure TestAllOpsFiniteAndShaped;
    procedure TestRotateZeroIsIdentity;
    procedure TestShearZeroIsIdentity;
    procedure TestTranslateZeroIsIdentity;
    procedure TestPosterize8BitsIsIdentity;
    procedure TestSolarizeIdentityWhenNoneAbove;
    procedure TestBrightnessFactorOneIsIdentity;
    procedure TestIdentityOpIsBitIdentical;
    procedure TestAutoContrastStretches;

    // RandAugment policy.
    procedure TestRandAugmentReproducible;
    procedure TestRandAugmentDiffersAcrossSeeds;

    // TrivialAugment policy.
    procedure TestTrivialAugmentReproducible;
    procedure TestTrivialAugmentChangesImage;

    // RandomErasing / Cutout.
    procedure TestRandomErasingReproducible;
    procedure TestRandomErasingFillsRectangle;
    procedure TestRandomErasingProbZeroNoOp;

    // TNeuralImageFit hook smoke (procedure-of-object signature).
    procedure TestPolicyHookSmoke;
  end;

implementation

function TTestNeuralAugment.MakeImage(W, H, Dep: integer): TNNetVolume;
var
  x, y, d: integer;
  px: TNeuralFloat;
begin
  Result := TNNetVolume.Create(W, H, Dep);
  for d := 0 to Dep - 1 do
    for y := 0 to H - 1 do
      for x := 0 to W - 1 do
      begin
        // pixel 0..255 gradient per channel, then to neuronal domain.
        px := ((x * 8 + y * 5 + d * 30) mod 256);
        Result[x, y, d] := (px - 128.0) / 64.0;
      end;
end;

function TTestNeuralAugment.VolumesEqual(A, B: TNNetVolume): boolean;
var
  I: integer;
begin
  Result := (A.Size = B.Size);
  if not Result then Exit;
  for I := 0 to A.Size - 1 do
    if A.FData[I] <> B.FData[I] then
    begin
      Result := false;
      Exit;
    end;
end;

function TTestNeuralAugment.VolumesFinite(A: TNNetVolume): boolean;
var
  I: integer;
  v: TNeuralFloat;
begin
  Result := true;
  for I := 0 to A.Size - 1 do
  begin
    v := A.FData[I];
    if IsNan(v) or IsInfinite(v) then
    begin
      Result := false;
      Exit;
    end;
    // Must remain in the neuronal clamp domain.
    if (v < -2.0001) or (v > 2.0001) then
    begin
      Result := false;
      Exit;
    end;
  end;
end;

procedure TTestNeuralAugment.TestAllOpsFiniteAndShaped;
var
  Op: TNeuralAugOp;
  V: TNNetVolume;
begin
  for Op := Low(TNeuralAugOp) to High(TNeuralAugOp) do
  begin
    RandSeed := 777;
    V := MakeImage(16, 16, 3);
    try
      NeuralAugApplyOp(V, Op, 20);
      AssertEquals('SizeX preserved', 16, V.SizeX);
      AssertEquals('SizeY preserved', 16, V.SizeY);
      AssertEquals('Depth preserved', 3, V.Depth);
      AssertTrue('Op output finite & in-domain for op ' + IntToStr(Ord(Op)),
        VolumesFinite(V));
    finally
      V.Free;
    end;
  end;
end;

procedure TTestNeuralAugment.TestRotateZeroIsIdentity;
var
  V, Orig: TNNetVolume;
begin
  V := MakeImage(12, 12, 3);
  Orig := TNNetVolume.Create;
  try
    Orig.Copy(V);
    NeuralAugApplyOp(V, csaRotate, 0);
    AssertTrue('rotate(M=0) is bit-identity', VolumesEqual(V, Orig));
  finally
    V.Free; Orig.Free;
  end;
end;

procedure TTestNeuralAugment.TestShearZeroIsIdentity;
var
  V, Orig: TNNetVolume;
begin
  V := MakeImage(12, 12, 3);
  Orig := TNNetVolume.Create;
  try
    Orig.Copy(V);
    NeuralAugApplyOp(V, csaShearX, 0);
    NeuralAugApplyOp(V, csaShearY, 0);
    AssertTrue('shear(M=0) is bit-identity', VolumesEqual(V, Orig));
  finally
    V.Free; Orig.Free;
  end;
end;

procedure TTestNeuralAugment.TestTranslateZeroIsIdentity;
var
  V, Orig: TNNetVolume;
begin
  V := MakeImage(12, 12, 3);
  Orig := TNNetVolume.Create;
  try
    Orig.Copy(V);
    NeuralAugApplyOp(V, csaTranslateX, 0);
    NeuralAugApplyOp(V, csaTranslateY, 0);
    AssertTrue('translate(M=0) is bit-identity', VolumesEqual(V, Orig));
  finally
    V.Free; Orig.Free;
  end;
end;

procedure TTestNeuralAugment.TestPosterize8BitsIsIdentity;
var
  V, Orig: TNNetVolume;
begin
  // M=0 maps posterize to 8 bits = identity.
  V := MakeImage(10, 10, 3);
  Orig := TNNetVolume.Create;
  try
    Orig.Copy(V);
    NeuralAugApplyOp(V, csaPosterize, 0);
    AssertTrue('posterize(8 bits) is identity', VolumesEqual(V, Orig));
  finally
    V.Free; Orig.Free;
  end;
end;

procedure TTestNeuralAugment.TestSolarizeIdentityWhenNoneAbove;
var
  V, Orig: TNNetVolume;
begin
  // M=0 maps solarize threshold to 255 -> only pixel 255 inverts; our
  // gradient maxes below 255 so this is (essentially) identity.
  V := MakeImage(10, 10, 3);
  Orig := TNNetVolume.Create;
  try
    Orig.Copy(V);
    NeuralAugApplyOp(V, csaSolarize, 0);
    AssertTrue('solarize(threshold=255) leaves sub-255 pixels',
      VolumesEqual(V, Orig));
  finally
    V.Free; Orig.Free;
  end;
end;

procedure TTestNeuralAugment.TestBrightnessFactorOneIsIdentity;
var
  V, Orig: TNNetVolume;
begin
  // M=0 -> factor 1.0 -> identity blend for brightness/contrast/color/sharp.
  V := MakeImage(10, 10, 3);
  Orig := TNNetVolume.Create;
  try
    Orig.Copy(V);
    NeuralAugApplyOp(V, csaBrightness, 0);
    AssertTrue('brightness(factor=1) ~ identity (1px tol)',
      VolumesEqual(V, Orig));
  finally
    V.Free; Orig.Free;
  end;
end;

procedure TTestNeuralAugment.TestIdentityOpIsBitIdentical;
var
  V, Orig: TNNetVolume;
begin
  V := MakeImage(8, 8, 3);
  Orig := TNNetVolume.Create;
  try
    Orig.Copy(V);
    NeuralAugApplyOp(V, csaIdentity, 25);
    AssertTrue('csaIdentity is bit-identity', VolumesEqual(V, Orig));
  finally
    V.Free; Orig.Free;
  end;
end;

procedure TTestNeuralAugment.TestAutoContrastStretches;
var
  V: TNNetVolume;
  mn, mx, v0: TNeuralFloat;
  I: integer;
begin
  // Build a low-contrast image (all pixels near gray) and check autocontrast
  // widens the per-channel range.
  V := TNNetVolume.Create(8, 8, 1);
  try
    for I := 0 to V.Size - 1 do
      V.FData[I] := (120 + (I mod 16) - 128.0) / 64.0; // pixels 120..135
    NeuralAugApplyOp(V, csaAutoContrast, 30);
    mn := V.FData[0]; mx := V.FData[0];
    for I := 0 to V.Size - 1 do
    begin
      v0 := V.FData[I];
      if v0 < mn then mn := v0;
      if v0 > mx then mx := v0;
    end;
    // After full stretch the range should span (near) the whole 0..255 domain.
    AssertTrue('autocontrast widens range', (mx - mn) > 3.0);
    AssertTrue('autocontrast finite', VolumesFinite(V));
  finally
    V.Free;
  end;
end;

procedure TTestNeuralAugment.TestRandAugmentReproducible;
var
  A, B: TNNetVolume;
begin
  A := MakeImage(16, 16, 3);
  B := MakeImage(16, 16, 3);
  try
    RandSeed := 12345;
    NeuralRandAugment(A, 3, 12);
    RandSeed := 12345;
    NeuralRandAugment(B, 3, 12);
    AssertTrue('RandAugment reproducible at fixed seed', VolumesEqual(A, B));
  finally
    A.Free; B.Free;
  end;
end;

procedure TTestNeuralAugment.TestRandAugmentDiffersAcrossSeeds;
var
  A, B: TNNetVolume;
  Diff: boolean;
  I: integer;
begin
  A := MakeImage(16, 16, 3);
  B := MakeImage(16, 16, 3);
  try
    RandSeed := 1;
    NeuralRandAugment(A, 3, 20);
    RandSeed := 999;
    NeuralRandAugment(B, 3, 20);
    Diff := false;
    for I := 0 to A.Size - 1 do
      if A.FData[I] <> B.FData[I] then Diff := true;
    AssertTrue('RandAugment differs across seeds', Diff);
  finally
    A.Free; B.Free;
  end;
end;

procedure TTestNeuralAugment.TestTrivialAugmentReproducible;
var
  A, B: TNNetVolume;
begin
  A := MakeImage(16, 16, 3);
  B := MakeImage(16, 16, 3);
  try
    RandSeed := 555;
    NeuralTrivialAugment(A);
    RandSeed := 555;
    NeuralTrivialAugment(B);
    AssertTrue('TrivialAugment reproducible at fixed seed', VolumesEqual(A, B));
  finally
    A.Free; B.Free;
  end;
end;

procedure TTestNeuralAugment.TestTrivialAugmentChangesImage;
var
  A, Orig: TNNetVolume;
  AnyChange, I: integer;
  Changed: boolean;
begin
  // Across several seeds at least one TrivialAugment draw changes the image
  // (it can draw csaIdentity, so loop a few seeds).
  AnyChange := 0;
  for I := 1 to 20 do
  begin
    A := MakeImage(16, 16, 3);
    Orig := TNNetVolume.Create;
    try
      Orig.Copy(A);
      RandSeed := I * 13 + 7;
      NeuralTrivialAugment(A);
      Changed := not VolumesEqual(A, Orig);
      if Changed then Inc(AnyChange);
      AssertTrue('TrivialAugment stays finite', VolumesFinite(A));
    finally
      A.Free; Orig.Free;
    end;
  end;
  AssertTrue('TrivialAugment changes image in some draws', AnyChange > 0);
end;

procedure TTestNeuralAugment.TestRandomErasingReproducible;
var
  A, B: TNNetVolume;
begin
  A := MakeImage(20, 20, 3);
  B := MakeImage(20, 20, 3);
  try
    RandSeed := 314;
    NeuralRandomErasing(A, 1.0);
    RandSeed := 314;
    NeuralRandomErasing(B, 1.0);
    AssertTrue('RandomErasing reproducible at fixed seed', VolumesEqual(A, B));
  finally
    A.Free; B.Free;
  end;
end;

procedure TTestNeuralAugment.TestRandomErasingFillsRectangle;
var
  V, Orig: TNNetVolume;
  x, y, d, filled, changed, untouched: integer;
  allFill: boolean;
begin
  V := MakeImage(24, 24, 3);
  Orig := TNNetVolume.Create;
  try
    Orig.Copy(V);
    RandSeed := 42;
    // prob=1, fill=0 (neutral gray). A contiguous rectangle becomes 0.
    NeuralRandomErasing(V, 1.0, 0.05, 0.2, 0.3, 0.0);
    filled := 0; changed := 0; untouched := 0;
    for y := 0 to V.SizeY - 1 do
      for x := 0 to V.SizeX - 1 do
      begin
        allFill := true;
        for d := 0 to V.Depth - 1 do
          if V[x, y, d] <> 0.0 then allFill := false;
        if allFill then Inc(filled);
        if V[x, y, 0] <> Orig[x, y, 0] then Inc(changed)
        else Inc(untouched);
      end;
    AssertTrue('RandomErasing filled some pixels', filled > 0);
    AssertTrue('RandomErasing left most of the image untouched',
      untouched > (V.SizeX * V.SizeY) div 2);
    AssertTrue('RandomErasing stays finite', VolumesFinite(V));
  finally
    V.Free; Orig.Free;
  end;
end;

procedure TTestNeuralAugment.TestRandomErasingProbZeroNoOp;
var
  V, Orig: TNNetVolume;
begin
  V := MakeImage(16, 16, 3);
  Orig := TNNetVolume.Create;
  try
    Orig.Copy(V);
    RandSeed := 99;
    NeuralRandomErasing(V, 0.0);
    AssertTrue('RandomErasing prob=0 is no-op', VolumesEqual(V, Orig));
  finally
    V.Free; Orig.Free;
  end;
end;

procedure TTestNeuralAugment.TestPolicyHookSmoke;
var
  Pol: TNeuralAugmentationPolicy;
  V: TNNetVolume;
  i: integer;
begin
  // Drive both policies through the procedure-of-object hook on a tiny batch.
  Pol := TNeuralAugmentationPolicy.Create(napRandAugment, 2, 9, 0.25);
  try
    for i := 0 to 7 do
    begin
      V := MakeImage(32, 32, 3);
      try
        RandSeed := 100 + i;
        Pol.Augment(V, {ThreadId=}0);
        AssertTrue('RandAugment hook keeps image finite', VolumesFinite(V));
        AssertEquals('shape preserved', 32, V.SizeX);
      finally
        V.Free;
      end;
    end;
    Pol.Policy := napTrivialAugment;
    for i := 0 to 7 do
    begin
      V := MakeImage(32, 32, 3);
      try
        RandSeed := 200 + i;
        Pol.Augment(V, 0);
        AssertTrue('TrivialAugment hook keeps image finite', VolumesFinite(V));
      finally
        V.Free;
      end;
    end;
  finally
    Pol.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralAugment);

end.
