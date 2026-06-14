unit TestNeuralImageMetrics;
(*
Tests for neuralimagemetrics.pas: FID (Frechet Inception Distance) and the
Inception Score.

FID + the matrix square-root are validated against a numpy float64 oracle on
synthetic Gaussian feature sets (tests/fixtures/image_metrics_oracle.json,
produced by tools/make_image_metrics_fixture.py). scipy is not available in
the project venv, so the oracle computes the matrix sqrt / cross-trace via
numpy eigendecomposition (the same algorithm the Pascal Jacobi solver
implements). FID is asserted to match the oracle within 1e-4 and FID of a set
against ITSELF is asserted ~0.

Inception Score is checked on controlled distributions whose value is
analytic: perfectly-confident + perfectly-balanced one-hot predictions over K
classes give IS == K; uniform predictions give IS == 1.

If the fixture is absent the FID/IS-vs-oracle tests Ignore() (the analytic IS
tests still run, as does the accumulator round-trip).
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, fpjson, jsonparser,
  neuralvolume, neuralimagemetrics;

type
  TTestNeuralImageMetrics = class(TTestCase)
  private
    function FixturePath: string;
    function LoadFixture(out Root: TJSONData): boolean;
    function JArrToDoubleArr(A: TJSONArray): TIMDoubleArray;
    function JArrToMatrix(A: TJSONArray): TIMDoubleMatrix;
  published
    // FID over both synthetic Gaussian cases matches the numpy oracle < 1e-4,
    // computed from the raw feature matrices (exercises the accumulator,
    // covariance, matrix sqrt and cross-trace end to end).
    procedure TestFIDvsOracle;
    // FID of a feature set against ITSELF is ~0.
    procedure TestFIDSelfIsZero;
    // The eigendecomposition matrix sqrt: M = sqrt(A); M*M == A.
    procedure TestMatrixSqrtReconstructs;
    // The accumulator's mean/covariance match the oracle's per-set stats.
    procedure TestAccumulatorStatistics;
    // Confident + balanced one-hot predictions over K classes -> IS == K.
    procedure TestISConfidentBalancedEqualsNumClasses;
    // Uniform predictions carry no information -> IS == 1.
    procedure TestISUniformIsOne;
    // IS over the mixed case (with splits) matches the numpy oracle.
    procedure TestISMixedVsOracle;
  end;

implementation

function TTestNeuralImageMetrics.FixturePath: string;
begin
  Result := ExtractFilePath(ParamStr(0)) + 'fixtures' + PathDelim +
    'image_metrics_oracle.json';
end;

function TTestNeuralImageMetrics.LoadFixture(out Root: TJSONData): boolean;
var
  SS: TStringStream;
  Parser: TJSONParser;
begin
  Root := nil;
  if not FileExists(FixturePath) then Exit(false);
  SS := TStringStream.Create('');
  try
    SS.LoadFromFile(FixturePath);
    Parser := TJSONParser.Create(SS.DataString, []);
    try
      Root := Parser.Parse;
    finally
      Parser.Free;
    end;
  finally
    SS.Free;
  end;
  Result := Root <> nil;
end;

function TTestNeuralImageMetrics.JArrToDoubleArr(A: TJSONArray): TIMDoubleArray;
var
  i: integer;
begin
  SetLength(Result, A.Count);
  for i := 0 to A.Count - 1 do
    Result[i] := A.Floats[i];
end;

function TTestNeuralImageMetrics.JArrToMatrix(A: TJSONArray): TIMDoubleMatrix;
var
  i: integer;
begin
  SetLength(Result, A.Count);
  for i := 0 to A.Count - 1 do
    Result[i] := JArrToDoubleArr(A.Arrays[i]);
end;

procedure TTestNeuralImageMetrics.TestFIDvsOracle;
var
  Root: TJSONData;
  cases: TJSONArray;
  c: TJSONObject;
  fr, fg: TIMDoubleMatrix;
  expected, got: Double;
  i: integer;
begin
  if not LoadFixture(Root) then
  begin
    Ignore('image_metrics_oracle.json absent; run ' +
      'tools/make_image_metrics_fixture.py');
    Exit;
  end;
  try
    cases := TJSONObject(Root).Arrays['fid_cases'];
    for i := 0 to cases.Count - 1 do
    begin
      c := cases.Objects[i];
      fr := JArrToMatrix(c.Arrays['featuresR']);
      fg := JArrToMatrix(c.Arrays['featuresG']);
      expected := c.Floats['fid'];
      got := ComputeFIDFromFeatures(fr, fg);
      AssertEquals('FID ' + c.Strings['name'], expected, got, 1e-4);
    end;
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestFIDSelfIsZero;
var
  Root: TJSONData;
  cases: TJSONArray;
  c: TJSONObject;
  fr: TIMDoubleMatrix;
  got: Double;
  i: integer;
begin
  if not LoadFixture(Root) then
  begin
    Ignore('fixture absent');
    Exit;
  end;
  try
    cases := TJSONObject(Root).Arrays['fid_cases'];
    for i := 0 to cases.Count - 1 do
    begin
      c := cases.Objects[i];
      fr := JArrToMatrix(c.Arrays['featuresR']);
      got := ComputeFIDFromFeatures(fr, fr);
      AssertEquals('FID self ' + c.Strings['name'], 0.0, got, 1e-6);
    end;
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestMatrixSqrtReconstructs;
var
  Root: TJSONData;
  c: TJSONObject;
  cov, sq, prod: TIMDoubleMatrix;
  n, i, j, k: integer;
  acc, maxErr: Double;
begin
  if not LoadFixture(Root) then
  begin
    Ignore('fixture absent');
    Exit;
  end;
  try
    c := TJSONObject(Root).Arrays['fid_cases'].Objects[0];
    cov := JArrToMatrix(c.Arrays['covR']);
    n := Length(cov);
    sq := MatrixSqrtSPD(cov);
    // prod = sq*sq should equal cov.
    SetLength(prod, n);
    maxErr := 0;
    for i := 0 to n - 1 do
    begin
      SetLength(prod[i], n);
      for j := 0 to n - 1 do
      begin
        acc := 0;
        for k := 0 to n - 1 do acc := acc + sq[i][k] * sq[k][j];
        prod[i][j] := acc;
        if Abs(acc - cov[i][j]) > maxErr then maxErr := Abs(acc - cov[i][j]);
      end;
    end;
    AssertTrue('sqrt(A)^2 reconstructs A (maxErr=' + FloatToStr(maxErr) + ')',
      maxErr < 1e-8);
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestAccumulatorStatistics;
var
  Root: TJSONData;
  c: TJSONObject;
  fr: TIMDoubleMatrix;
  emR: TIMDoubleArray;
  ecR: TIMDoubleMatrix;
  acc: TFIDFeatureAccumulator;
  mu: TIMDoubleArray;
  cov: TIMDoubleMatrix;
  i, j, dim: integer;
begin
  if not LoadFixture(Root) then
  begin
    Ignore('fixture absent');
    Exit;
  end;
  try
    c := TJSONObject(Root).Arrays['fid_cases'].Objects[0];
    fr := JArrToMatrix(c.Arrays['featuresR']);
    emR := JArrToDoubleArr(c.Arrays['meanR']);
    ecR := JArrToMatrix(c.Arrays['covR']);
    dim := Length(fr[0]);
    acc := TFIDFeatureAccumulator.Create(dim);
    try
      for i := 0 to Length(fr) - 1 do acc.Add(fr[i]);
      mu := acc.Mean;
      cov := acc.Covariance;
      for i := 0 to dim - 1 do
        AssertEquals('mean['+IntToStr(i)+']', emR[i], mu[i], 1e-6);
      for i := 0 to dim - 1 do
        for j := 0 to dim - 1 do
          AssertEquals('cov', ecR[i][j], cov[i][j], 1e-6);
    finally
      acc.Free;
    end;
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestISConfidentBalancedEqualsNumClasses;
const
  K = 5;
var
  probs: TIMDoubleMatrix;
  i, j: integer;
  s: Double;
begin
  // One sample per class, near-one-hot -> IS == K exactly.
  SetLength(probs, K);
  for i := 0 to K - 1 do
  begin
    SetLength(probs[i], K);
    for j := 0 to K - 1 do
      if i = j then probs[i][j] := 1.0 else probs[i][j] := 0.0;
  end;
  s := InceptionScore(probs, 1);
  AssertEquals('IS confident-balanced == K', Double(K), s, 1e-4);
end;

procedure TTestNeuralImageMetrics.TestISUniformIsOne;
var
  probs: TIMDoubleMatrix;
  i, j: integer;
  s: Double;
begin
  SetLength(probs, 20);
  for i := 0 to 19 do
  begin
    SetLength(probs[i], 4);
    for j := 0 to 3 do probs[i][j] := 0.25;
  end;
  s := InceptionScore(probs, 1);
  AssertEquals('IS uniform == 1', 1.0, s, 1e-6);
end;

procedure TTestNeuralImageMetrics.TestISMixedVsOracle;
var
  Root: TJSONData;
  cases: TJSONArray;
  c: TJSONObject;
  probs: TIMDoubleMatrix;
  expected, got: Double;
  i: integer;
begin
  if not LoadFixture(Root) then
  begin
    Ignore('fixture absent');
    Exit;
  end;
  try
    cases := TJSONObject(Root).Arrays['is_cases'];
    for i := 0 to cases.Count - 1 do
    begin
      c := cases.Objects[i];
      probs := JArrToMatrix(c.Arrays['probs']);
      expected := c.Floats['score'];
      got := InceptionScore(probs, c.Integers['splits']);
      AssertEquals('IS ' + c.Strings['name'], expected, got, 1e-4);
    end;
  finally
    Root.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralImageMetrics);
end.
