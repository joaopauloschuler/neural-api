unit TestNeuralScheduler;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralvolume, neuralscheduler;

const
  TOL = 1e-5;

type
  TTestNeuralScheduler = class(TTestCase)
  published
    // TStepLR
    procedure TestStepLRBoundaries;
    procedure TestStepLRMonotonicNonIncreasing;
    procedure TestStepLRValidation;

    // TCosineAnnealingLR
    procedure TestCosineBoundaries;
    procedure TestCosineMidpoint;
    procedure TestCosineNonIncreasing;
    procedure TestCosineClampBeyondT;
    procedure TestCosineValidation;

    // TWarmupCosineLR
    procedure TestWarmupLinearRamp;
    procedure TestWarmupPeakAndEnd;
    procedure TestWarmupShape;
    procedure TestWarmupValidation;

    // TPolyLR
    procedure TestPolyLinearBoundaries;
    procedure TestPolyMidpoint;
    procedure TestPolyClampBeyondT;
    procedure TestPolyPower2;
    procedure TestPolyValidation;

    // General
    procedure TestEpochIgnored;
    procedure TestFiniteAcrossRange;
  end;

implementation

{ TStepLR }

procedure TTestNeuralScheduler.TestStepLRBoundaries;
var
  S: TStepLR;
begin
  // baseLR=1.0, stepSize=3, gamma=0.5
  S := TStepLR.Create(1.0, 3, 0.5);
  try
    AssertEquals('StepLR t=0', 1.0, S.NextLR(0, 0), TOL);
    AssertEquals('StepLR t=2 (still in first block)', 1.0, S.NextLR(0, 2), TOL);
    AssertEquals('StepLR t=3 (first drop)', 0.5, S.NextLR(0, 3), TOL);
    AssertEquals('StepLR t=5', 0.5, S.NextLR(0, 5), TOL);
    AssertEquals('StepLR t=6 (second drop)', 0.25, S.NextLR(0, 6), TOL);
    AssertEquals('StepLR t=9 (third drop)', 0.125, S.NextLR(0, 9), TOL);
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestStepLRMonotonicNonIncreasing;
var
  S: TStepLR;
  I: integer;
  Prev, Cur: TNeuralFloat;
begin
  S := TStepLR.Create(2.0, 4, 0.9);
  try
    Prev := S.NextLR(0, 0);
    for I := 1 to 40 do
    begin
      Cur := S.NextLR(0, I);
      AssertTrue('StepLR must be non-increasing at ' + IntToStr(I), Cur <= Prev + TOL);
      Prev := Cur;
    end;
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestStepLRValidation;
begin
  try
    TStepLR.Create(1.0, 0, 0.5).Free;
    Fail('StepLR stepSize<=0 must raise');
  except
    on E: Exception do ; // expected
  end;
  try
    TStepLR.Create(1.0, 3, 0.0).Free;
    Fail('StepLR gamma<=0 must raise');
  except
    on E: Exception do ; // expected
  end;
  try
    TStepLR.Create(1.0, 3, 1.5).Free;
    Fail('StepLR gamma>1 must raise');
  except
    on E: Exception do ; // expected
  end;
end;

{ TCosineAnnealingLR }

procedure TTestNeuralScheduler.TestCosineBoundaries;
var
  S: TCosineAnnealingLR;
begin
  // etaMax=1.0, etaMin=0.0, T=10
  S := TCosineAnnealingLR.Create(1.0, 0.0, 10);
  try
    AssertEquals('Cosine t=0 == etaMax', 1.0, S.NextLR(0, 0), TOL);
    AssertEquals('Cosine t=T == etaMin', 0.0, S.NextLR(0, 10), TOL);
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestCosineMidpoint;
var
  S: TCosineAnnealingLR;
begin
  // etaMax=1.0, etaMin=0.2, T=10 -> midpoint == (max+min)/2 = 0.6
  S := TCosineAnnealingLR.Create(1.0, 0.2, 10);
  try
    AssertEquals('Cosine t=T/2 == (etaMax+etaMin)/2', 0.6, S.NextLR(0, 5), TOL);
    AssertEquals('Cosine t=0 == etaMax', 1.0, S.NextLR(0, 0), TOL);
    AssertEquals('Cosine t=T == etaMin', 0.2, S.NextLR(0, 10), TOL);
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestCosineNonIncreasing;
var
  S: TCosineAnnealingLR;
  I: integer;
  Prev, Cur: TNeuralFloat;
begin
  S := TCosineAnnealingLR.Create(1.0, 0.0, 20);
  try
    Prev := S.NextLR(0, 0);
    for I := 1 to 20 do
    begin
      Cur := S.NextLR(0, I);
      AssertTrue('Cosine must be non-increasing at ' + IntToStr(I), Cur <= Prev + TOL);
      Prev := Cur;
    end;
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestCosineClampBeyondT;
var
  S: TCosineAnnealingLR;
begin
  S := TCosineAnnealingLR.Create(1.0, 0.0, 10);
  try
    AssertEquals('Cosine beyond T clamps to etaMin', 0.0, S.NextLR(0, 50), TOL);
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestCosineValidation;
begin
  try
    TCosineAnnealingLR.Create(1.0, 0.0, 0).Free;
    Fail('Cosine T<=0 must raise');
  except
    on E: Exception do ; // expected
  end;
end;

{ TWarmupCosineLR }

procedure TTestNeuralScheduler.TestWarmupLinearRamp;
var
  S: TWarmupCosineLR;
begin
  // etaMax=1.0, etaMin=0.0, warmup=4, T=12
  S := TWarmupCosineLR.Create(1.0, 0.0, 4, 12);
  try
    AssertEquals('Warmup t=0 == 0', 0.0, S.NextLR(0, 0), TOL);
    AssertEquals('Warmup t=1 linear', 0.25, S.NextLR(0, 1), TOL);
    AssertEquals('Warmup t=2 linear', 0.5, S.NextLR(0, 2), TOL);
    AssertEquals('Warmup t=3 linear', 0.75, S.NextLR(0, 3), TOL);
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestWarmupPeakAndEnd;
var
  S: TWarmupCosineLR;
begin
  // etaMax=1.0, etaMin=0.0, warmup=4, T=12 -> cosine over [4,12], midpoint t=8
  S := TWarmupCosineLR.Create(1.0, 0.0, 4, 12);
  try
    AssertEquals('Warmup t=warmup == etaMax', 1.0, S.NextLR(0, 4), TOL);
    AssertEquals('Warmup cosine midpoint t=8 == (max+min)/2', 0.5, S.NextLR(0, 8), TOL);
    AssertEquals('Warmup t=T == etaMin', 0.0, S.NextLR(0, 12), TOL);
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestWarmupShape;
var
  S: TWarmupCosineLR;
  I: integer;
  Prev, Cur: TNeuralFloat;
begin
  // Non-decreasing on [0,warmup], then non-increasing on [warmup,T].
  S := TWarmupCosineLR.Create(1.0, 0.1, 5, 25);
  try
    Prev := S.NextLR(0, 0);
    for I := 1 to 5 do
    begin
      Cur := S.NextLR(0, I);
      AssertTrue('Warmup ramp must be non-decreasing at ' + IntToStr(I), Cur >= Prev - TOL);
      Prev := Cur;
    end;
    Prev := S.NextLR(0, 5);
    for I := 6 to 25 do
    begin
      Cur := S.NextLR(0, I);
      AssertTrue('Warmup cosine must be non-increasing at ' + IntToStr(I), Cur <= Prev + TOL);
      Prev := Cur;
    end;
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestWarmupValidation;
begin
  try
    TWarmupCosineLR.Create(1.0, 0.0, 0, 10).Free;
    Fail('Warmup warmup<=0 must raise');
  except
    on E: Exception do ; // expected
  end;
  try
    TWarmupCosineLR.Create(1.0, 0.0, 10, 10).Free;
    Fail('Warmup T<=warmup must raise');
  except
    on E: Exception do ; // expected
  end;
end;

{ TPolyLR }

procedure TTestNeuralScheduler.TestPolyLinearBoundaries;
var
  S: TPolyLR;
begin
  // baseLR=1.0, T=10, power=1 -> linear decay
  S := TPolyLR.Create(1.0, 10, 1.0);
  try
    AssertEquals('Poly t=0 == baseLR', 1.0, S.NextLR(0, 0), TOL);
    AssertEquals('Poly t=T == 0', 0.0, S.NextLR(0, 10), TOL);
    AssertEquals('Poly power=1 t=2 linear', 0.8, S.NextLR(0, 2), TOL);
    AssertEquals('Poly power=1 t=7 linear', 0.3, S.NextLR(0, 7), TOL);
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestPolyMidpoint;
var
  S: TPolyLR;
begin
  S := TPolyLR.Create(2.0, 10, 1.0);
  try
    AssertEquals('Poly midpoint linear', 1.0, S.NextLR(0, 5), TOL);
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestPolyClampBeyondT;
var
  S: TPolyLR;
begin
  // power=0.5: must not take a negative base to a fractional power.
  S := TPolyLR.Create(1.0, 10, 0.5);
  try
    AssertEquals('Poly t>T clamps to 0', 0.0, S.NextLR(0, 50), TOL);
    AssertEquals('Poly t=T == 0', 0.0, S.NextLR(0, 10), TOL);
    AssertTrue('Poly fractional power finite', S.NextLR(0, 3) > 0);
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestPolyPower2;
var
  S: TPolyLR;
begin
  // baseLR=1.0, T=10, power=2 -> (1 - t/T)^2
  S := TPolyLR.Create(1.0, 10, 2.0);
  try
    AssertEquals('Poly power=2 t=0', 1.0, S.NextLR(0, 0), TOL);
    AssertEquals('Poly power=2 t=5 == 0.25', 0.25, S.NextLR(0, 5), TOL);
    AssertEquals('Poly power=2 t=10 == 0', 0.0, S.NextLR(0, 10), TOL);
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestPolyValidation;
begin
  try
    TPolyLR.Create(1.0, 0, 1.0).Free;
    Fail('Poly T<=0 must raise');
  except
    on E: Exception do ; // expected
  end;
  try
    TPolyLR.Create(1.0, 10, -1.0).Free;
    Fail('Poly power<0 must raise');
  except
    on E: Exception do ; // expected
  end;
end;

{ General }

procedure TTestNeuralScheduler.TestEpochIgnored;
var
  S: TCosineAnnealingLR;
begin
  // These schedulers key on Step; Epoch must not change the result.
  S := TCosineAnnealingLR.Create(1.0, 0.0, 10);
  try
    AssertEquals('Epoch must be ignored',
      S.NextLR(0, 5), S.NextLR(999, 5), TOL);
  finally
    S.Free;
  end;
end;

procedure TTestNeuralScheduler.TestFiniteAcrossRange;
var
  Schedulers: array[0..3] of TNeuralLRScheduler;
  I, Step: integer;
  V: TNeuralFloat;
begin
  Schedulers[0] := TStepLR.Create(0.1, 5, 0.7);
  Schedulers[1] := TCosineAnnealingLR.Create(0.1, 0.001, 30);
  Schedulers[2] := TWarmupCosineLR.Create(0.1, 0.001, 5, 30);
  Schedulers[3] := TPolyLR.Create(0.1, 30, 0.9);
  try
    for I := 0 to 3 do
      for Step := 0 to 40 do
      begin
        V := Schedulers[I].NextLR(0, Step);
        AssertTrue('LR must be finite (sched ' + IntToStr(I) + ' step ' + IntToStr(Step) + ')',
          not IsNan(V) and not IsInfinite(V));
        AssertTrue('LR must be non-negative (sched ' + IntToStr(I) + ' step ' + IntToStr(Step) + ')',
          V >= -TOL);
      end;
  finally
    for I := 0 to 3 do Schedulers[I].Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralScheduler);

end.
