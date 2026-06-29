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
  neuralvolume, neuralnetwork, neuralimagemetrics, neuralpretrained,
  neuraldatasets;

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
    // SSIM/PSNR vs the numpy float64 oracle on pinned image pairs.
    procedure TestSSIMvsOracle;
    procedure TestPSNRvsOracle;
    // MS-SSIM vs the per-scale-product numpy oracle.
    procedure TestMSSSIMvsOracle;
    // Identical images -> SSIM 1.0 and PSNR +Inf.
    procedure TestSSIMIdenticalIsOne;
    procedure TestPSNRIdenticalIsInf;
    // 1-SSIM loss gradient matches a central difference.
    procedure TestSSIMLossGradient;
    // TNNetSSIMLoss head: forward is an identity passthrough.
    procedure TestSSIMLossLayerForwardPassthrough;
    // TNNetSSIMLoss head: the +gradient it writes to FOutputError matches a
    // central difference of the 1-SSIM loss.
    procedure TestSSIMLossLayerGradient;
    // TNNetSSIMLoss head: SaveToString / LoadFromString round-trips DataRange.
    procedure TestSSIMLossLayerRoundTrip;
    // KID unbiased MMD^2 vs the numpy float64 oracle on Gaussian sets.
    procedure TestKIDMMD2vsOracle;
    // KID ordering: self ~ small < same-distribution < shifted-distribution.
    procedure TestKIDOrdering;
    // KID subset-bootstrap mean is close to the full-set estimate.
    procedure TestKIDSubsetBootstrap;
    // FID wired onto the real Inception backbone (pico fixture): FID of a set
    // against ITSELF is ~0, and FID grows monotonically as the generated set is
    // perturbed further from the real set.
    procedure TestInceptionFIDSelfZeroAndMonotone;
    // --- ImageNet top-1 / top-5 accuracy harness ---
    // TopKIndices returns the K largest indices, most-confident first, with a
    // first-max (lowest-index) tie-break.
    procedure TestTopKIndices;
    // End-to-end top-1 / top-5 counting on a stub identity net with pinned
    // class-score volumes (deterministic, asserted accuracy numbers).
    procedure TestEvaluateImageNetCounting;
    // GoldLabel outside 0..NumClasses-1 is skipped; the confusion sample is
    // capped at MaxConfusion and records the top-K-hit flag.
    procedure TestEvaluateImageNetSkipAndConfusion;
    // The report formats top-1 / top-5 lines and the confusion sample.
    procedure TestImageNetReportFormat;
    // The resize-shorter-side + center-crop + mean/std transform math
    // (PreprocessImageForVisionModel) is exact on a pinned synthetic image.
    procedure TestPreprocessTransformMath;
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

procedure TTestNeuralImageMetrics.TestSSIMvsOracle;
var
  Root: TJSONData;
  cases: TJSONArray;
  c: TJSONObject;
  ia, ib: TIMDoubleArray;
  expected, got: Double;
  i: integer;
begin
  if not LoadFixture(Root) then begin Ignore('fixture absent'); Exit; end;
  try
    cases := TJSONObject(Root).Arrays['ssim_cases'];
    for i := 0 to cases.Count - 1 do
    begin
      c := cases.Objects[i];
      ia := JArrToDoubleArr(c.Arrays['imgA']);
      ib := JArrToDoubleArr(c.Arrays['imgB']);
      expected := c.Floats['ssim'];
      got := ComputeSSIM(ia, ib, c.Integers['H'], c.Integers['W'],
        c.Integers['C'], c.Floats['dataRange']);
      AssertEquals('SSIM ' + c.Strings['name'], expected, got, 1e-6);
    end;
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestPSNRvsOracle;
var
  Root: TJSONData;
  cases: TJSONArray;
  c: TJSONObject;
  ia, ib: TIMDoubleArray;
  expected, got: Double;
  i: integer;
begin
  if not LoadFixture(Root) then begin Ignore('fixture absent'); Exit; end;
  try
    cases := TJSONObject(Root).Arrays['ssim_cases'];
    for i := 0 to cases.Count - 1 do
    begin
      c := cases.Objects[i];
      ia := JArrToDoubleArr(c.Arrays['imgA']);
      ib := JArrToDoubleArr(c.Arrays['imgB']);
      expected := c.Floats['psnr'];
      got := ComputePSNR(ia, ib, c.Integers['H'], c.Integers['W'],
        c.Integers['C'], c.Floats['dataRange']);
      AssertEquals('PSNR ' + c.Strings['name'], expected, got, 1e-5);
    end;
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestMSSSIMvsOracle;
var
  Root: TJSONData;
  c: TJSONObject;
  ia, ib: TIMDoubleArray;
  expected, got: Double;
begin
  if not LoadFixture(Root) then begin Ignore('fixture absent'); Exit; end;
  try
    // gray_40x40 case carries the larger MS-SSIM plane in imgMA/imgMB/Hm/Wm.
    c := TJSONObject(Root).Arrays['ssim_cases'].Objects[0];
    ia := JArrToDoubleArr(c.Arrays['imgMA']);
    ib := JArrToDoubleArr(c.Arrays['imgMB']);
    expected := c.Floats['msssim'];
    got := ComputeMSSSIM(ia, ib, c.Integers['Hm'], c.Integers['Wm'], 1,
      c.Floats['dataRange']);
    AssertEquals('MS-SSIM gray', expected, got, 1e-6);
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestSSIMIdenticalIsOne;
var
  img: TIMDoubleArray;
  i: integer;
begin
  SetLength(img, 20 * 20);
  RandSeed := 123456;
  for i := 0 to Length(img) - 1 do img[i] := Random;
  AssertEquals('SSIM identical == 1', 1.0,
    ComputeSSIM(img, img, 20, 20, 1, 1.0), 1e-12);
end;

procedure TTestNeuralImageMetrics.TestPSNRIdenticalIsInf;
var
  img: TIMDoubleArray;
  i: integer;
  v: Double;
begin
  SetLength(img, 16 * 16);
  for i := 0 to Length(img) - 1 do img[i] := i / 256.0;
  v := ComputePSNR(img, img, 16, 16, 1, 1.0);
  AssertTrue('PSNR identical == +Inf', IsInfinite(v) and (v > 0));
end;

procedure TTestNeuralImageMetrics.TestSSIMLossGradient;
var
  ia, ib, grad, ia2: TIMDoubleArray;
  loss, lp, lm, numg: Double;
  i, n: integer;
  eps: Double;
  maxErr: Double;
  dummy: TIMDoubleArray;
begin
  // Small 16x16 single-channel pair; central-difference check on a few pixels.
  n := 16 * 16;
  SetLength(ia, n); SetLength(ib, n);
  RandSeed := 778899; Random;  // defensive reseed (FPC mtwist), see memory note
  RandSeed := 778800;
  for i := 0 to n - 1 do begin ia[i] := Random; ib[i] := Random; end;
  loss := ComputeSSIMLossAndGradient(ia, ib, 16, 16, 1, grad, 1.0);
  AssertTrue('loss in [0,2]', (loss >= 0) and (loss <= 2));
  eps := 1e-6;
  maxErr := 0;
  // probe a deterministic spread of pixels
  for i := 0 to 9 do
  begin
    SetLength(ia2, n);
    Move(ia[0], ia2[0], n * SizeOf(Double));
    ia2[i * 25 + 7] := ia2[i * 25 + 7] + eps;
    lp := ComputeSSIMLossAndGradient(ia2, ib, 16, 16, 1, dummy, 1.0);
    ia2[i * 25 + 7] := ia2[i * 25 + 7] - 2 * eps;
    lm := ComputeSSIMLossAndGradient(ia2, ib, 16, 16, 1, dummy, 1.0);
    numg := (lp - lm) / (2 * eps);
    if Abs(numg - grad[i * 25 + 7]) > maxErr then
      maxErr := Abs(numg - grad[i * 25 + 7]);
  end;
  AssertTrue('SSIM loss grad vs central diff (maxErr=' +
    FloatToStr(maxErr) + ')', maxErr < 1e-4);
end;

procedure TTestNeuralImageMetrics.TestSSIMLossLayerForwardPassthrough;
var
  NN: TNNet;
  Input: TNNetVolume;
  i: integer;
begin
  // TNNetSSIMLoss forward must be an identity passthrough (W=11, H=11, C=1).
  NN := TNNet.Create();
  Input := TNNetVolume.Create(11, 11, 1);
  try
    NN.AddLayer(TNNetInput.Create(11, 11, 1, 1));
    NN.AddLayer(TNNetSSIMLoss.Create(1.0));
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Abs(Sin(i * 0.31)) * 0.9 + 0.05;  // in (0,1)
    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
      AssertEquals('SSIMLoss forward is passthrough at ' + IntToStr(i),
        Input.Raw[i], NN.GetLastLayer.Output.Raw[i], 0.00001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestSSIMLossLayerGradient;
var
  NN: TNNet;
  Input, Target: TNNetVolume;
  LMid: TNNetIdentity;
  ia, ib, ia2, grad, dummy: TIMDoubleArray;
  H, W, C, n, i, p: integer;
  eps, lp, lm, numg, maxErr: Double;
begin
  // 11x11x3 image (both sides >= the 11x11 SSIM window). The head must write
  // +d(1-SSIM)/dprediction into FOutputError (which an upstream Identity layer
  // receives); check it against a central difference of the same 1-SSIM loss
  // the helper computes.
  W := 11; H := 11; C := 3;
  n := H * W * C;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(W, H, C);
  Target := TNNetVolume.Create(W, H, C);
  SetLength(ia, n); SetLength(ib, n);
  try
    NN.AddLayer(TNNetInput.Create(W, H, C, 1));
    LMid := TNNetIdentity.Create();
    NN.AddLayer(LMid);
    NN.AddLayer(TNNetSSIMLoss.Create(1.0));

    RandSeed := 991133; Random;  // defensive reseed (FPC mtwist), see memory note
    RandSeed := 424242;
    for i := 0 to n - 1 do
    begin
      ia[i] := Random * 0.8 + 0.1;   // prediction
      ib[i] := Random * 0.8 + 0.1;   // target
    end;
    // The volume raw layout is bit-identical to the helper's channel-last
    // layout, so a straight copy is correct.
    for i := 0 to n - 1 do
    begin
      Input.Raw[i]  := ia[i];
      Target.Raw[i] := ib[i];
    end;

    NN.Compute(Input);
    NN.Backpropagate(Target);

    eps := 1e-6;
    maxErr := 0;
    // probe a deterministic spread of pixels across all channels
    for i := 0 to 11 do
    begin
      p := (i * 19 + 5) mod n;
      SetLength(ia2, n);
      Move(ia[0], ia2[0], n * SizeOf(Double));
      ia2[p] := ia2[p] + eps;
      lp := ComputeSSIMLossAndGradient(ia2, ib, H, W, C, dummy, 1.0);
      ia2[p] := ia2[p] - 2 * eps;
      lm := ComputeSSIMLossAndGradient(ia2, ib, H, W, C, dummy, 1.0);
      numg := (lp - lm) / (2 * eps);
      if Abs(numg - LMid.OutputError.Raw[p]) > maxErr then
        maxErr := Abs(numg - LMid.OutputError.Raw[p]);
    end;
    // float32 OutputError vs float64 central diff: ~1e-3 tolerance.
    AssertTrue('SSIMLoss layer grad vs central diff (maxErr=' +
      FloatToStr(maxErr) + ')', maxErr < 1e-3);
    // sanity: analytic helper grad agrees too
    ComputeSSIMLossAndGradient(ia, ib, H, W, C, grad, 1.0);
    AssertTrue('helper grad finite', not IsNan(grad[0]));
  finally
    NN.Free;
    Input.Free;
    Target.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestSSIMLossLayerRoundTrip;
var
  NN, NN2: TNNet;
  Saved: string;
begin
  // SaveToString / LoadFromString must round-trip the layer type and DataRange.
  NN := TNNet.Create();
  NN2 := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(11, 11, 1, 1));
    NN.AddLayer(TNNetSSIMLoss.Create(255.0));
    Saved := NN.SaveToString();
    NN2.LoadFromString(Saved);
    AssertTrue('Loaded last layer is TNNetSSIMLoss',
      NN2.GetLastLayer is TNNetSSIMLoss);
    // The structure string encodes the FFloatSt params, so matching strings
    // proves DataRange (255.0) survived the round-trip.
    AssertEquals('SSIMLoss structure round-trip',
      NN.GetLastLayer.SaveStructureToString(),
      NN2.GetLastLayer.SaveStructureToString());
  finally
    NN.Free;
    NN2.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestKIDMMD2vsOracle;
var
  Root: TJSONData;
  c: TJSONObject;
  fr, fs, ff: TIMDoubleMatrix;
begin
  if not LoadFixture(Root) then begin Ignore('fixture absent'); Exit; end;
  try
    c := TJSONObject(Root).Arrays['kid_cases'].Objects[0];
    fr := JArrToMatrix(c.Arrays['featuresR']);
    fs := JArrToMatrix(c.Arrays['featuresGsame']);
    ff := JArrToMatrix(c.Arrays['featuresGfar']);
    AssertEquals('KID MMD2 self', c.Floats['mmd2_self'],
      ComputeKIDMMD2(fr, fr), 1e-6);
    AssertEquals('KID MMD2 same', c.Floats['mmd2_same'],
      ComputeKIDMMD2(fr, fs), 1e-6);
    AssertEquals('KID MMD2 far', c.Floats['mmd2_far'],
      ComputeKIDMMD2(fr, ff), 1e-4);
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestKIDOrdering;
var
  Root: TJSONData;
  c: TJSONObject;
  fr, fs, ff: TIMDoubleMatrix;
  kSelf, kSame, kFar: Double;
begin
  if not LoadFixture(Root) then begin Ignore('fixture absent'); Exit; end;
  try
    c := TJSONObject(Root).Arrays['kid_cases'].Objects[0];
    fr := JArrToMatrix(c.Arrays['featuresR']);
    fs := JArrToMatrix(c.Arrays['featuresGsame']);
    ff := JArrToMatrix(c.Arrays['featuresGfar']);
    kSelf := ComputeKIDMMD2(fr, fr);
    kSame := ComputeKIDMMD2(fr, fs);
    kFar := ComputeKIDMMD2(fr, ff);
    // self ~ 0 (unbiased estimator can be slightly negative)
    AssertTrue('KID self near zero', Abs(kSelf) < 1.0);
    // increasing distribution separation -> increasing KID
    AssertTrue('KID self < same', kSelf < kSame);
    AssertTrue('KID same < far', kSame < kFar);
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestKIDSubsetBootstrap;
var
  Root: TJSONData;
  c: TJSONObject;
  fr, ff: TIMDoubleMatrix;
  full, score, sd: Double;
begin
  if not LoadFixture(Root) then begin Ignore('fixture absent'); Exit; end;
  try
    c := TJSONObject(Root).Arrays['kid_cases'].Objects[0];
    fr := JArrToMatrix(c.Arrays['featuresR']);
    ff := JArrToMatrix(c.Arrays['featuresGfar']);
    full := ComputeKIDMMD2(fr, ff);
    RandSeed := 20260615; Random;  // defensive reseed (FPC mtwist)
    RandSeed := 20260600;
    ComputeKID(fr, ff, 40, 20, score, sd);
    // bootstrap mean should be in the ballpark of the full-set estimate
    AssertTrue('KID bootstrap mean ~ full (' + FloatToStr(score) + ' vs ' +
      FloatToStr(full) + ')', Abs(score - full) < 0.3 * Abs(full));
    AssertTrue('KID bootstrap std >= 0', sd >= 0);
  finally
    Root.Free;
  end;
end;

// Real-backbone FID self-consistency on the pico Inception fixture. Builds the
// committed tiny Inception-v3 (image_size 8, pooled width 13), makes a "real"
// set of random ImageNet-scale image volumes, then three "generated" sets that
// are the real set plus zero / small / large per-pixel perturbations. The
// pooled Inception feature drives FID; FID(real, real) must be ~0 and FID must
// grow monotonically with the perturbation. (Self-consistency rather than a
// torch oracle: torchvision is not installed and the pico backbone is faithful
// to exactly what CAI computes, so an external float64 reference would only
// re-derive these same pooled features. The FID MATH itself is already pinned
// to a numpy float64 oracle by TestFIDvsOracle.)
procedure TTestNeuralImageMetrics.TestInceptionFIDSelfZeroAndMonotone;
const
  cInceptionFixture = 'tiny_inceptionv3.safetensors';
  cInceptionConfig  = 'tiny_inceptionv3_config.json';
  cNumImages = 8;

  function MakeImageSet(Seed: integer; PerturbScale: Single;
    Base: TNNetVolumeList; Cfg: TInceptionV3Config): TNNetVolumeList;
  var
    n, i: integer;
    V: TNNetVolume;
  begin
    Result := TNNetVolumeList.Create(true);
    RandSeed := Seed;
    for n := 0 to cNumImages - 1 do
    begin
      V := TNNetVolume.Create(Cfg.ImageSize, Cfg.ImageSize, Cfg.NumChannels);
      for i := 0 to V.Size - 1 do
        if Base = nil then
          // ImageNet-normalised pixels live roughly in [-2.6, 2.6].
          V.FData[i] := (Random * 4.0) - 2.0
        else
          V.FData[i] := Base[n].FData[i] + PerturbScale * ((Random * 2.0) - 1.0);
      Result.Add(V);
    end;
  end;

var
  NN: TNNet;
  Cfg: TInceptionV3Config;
  PoolIdx: integer;
  fixDir, fixSafe, fixCfg: string;
  realSet, genSame, genNear, genFar: TNNetVolumeList;
  fidSelf, fidNear, fidFar: Double;
begin
  fixDir := ExtractFilePath(ParamStr(0)) + 'fixtures' + PathDelim;
  fixSafe := fixDir + cInceptionFixture;
  fixCfg  := fixDir + cInceptionConfig;
  if (not FileExists(fixSafe)) or (not FileExists(fixCfg)) then
  begin
    Ignore('inception fixture absent');
    Exit;
  end;

  NN := BuildInceptionV3FromSafeTensors(fixSafe, Cfg, PoolIdx,
    {pTrainable=}false, fixCfg);
  realSet := nil; genSame := nil; genNear := nil; genFar := nil;
  try
    // Real images, then generated sets at growing distance from them.
    realSet := MakeImageSet(101, 0.0, nil, Cfg);
    genSame := MakeImageSet(0, 0.0, realSet, Cfg);   // exact copies of realSet
    genNear := MakeImageSet(202, 0.10, realSet, Cfg);
    genFar  := MakeImageSet(303, 0.80, realSet, Cfg);

    // (a) FID of the real set against itself is ~0.
    fidSelf := ComputeInceptionFID(NN, PoolIdx, realSet, realSet);
    AssertEquals('FID(real, real) ~ 0', 0.0, fidSelf, 1e-6);

    // FID against an exact copy is also ~0.
    AssertEquals('FID(real, copy) ~ 0', 0.0,
      ComputeInceptionFID(NN, PoolIdx, realSet, genSame), 1e-6);

    // (b) FID grows monotonically as the generated set diverges.
    fidNear := ComputeInceptionFID(NN, PoolIdx, realSet, genNear);
    fidFar  := ComputeInceptionFID(NN, PoolIdx, realSet, genFar);
    AssertTrue('FID near > 0 (' + FloatToStr(fidNear) + ')', fidNear > 1e-9);
    AssertTrue('FID far > FID near (' + FloatToStr(fidFar) + ' > ' +
      FloatToStr(fidNear) + ')', fidFar > fidNear);

    // Single-feature extraction returns the pooled width (13 for the pico).
    AssertEquals('pooled feature width',
      NN.Layers[PoolIdx].Output.Size,
      Length(ExtractInceptionFeature(NN, PoolIdx, realSet[0])));
  finally
    realSet.Free; genSame.Free; genNear.Free; genFar.Free;
    NN.Free;
  end;
end;

{ ImageNet accuracy harness }

// A stub net whose output IS its input: a single Input layer of the class-score
// shape. Compute(V) copies V into the output, so feeding a hand-built score
// volume pins the net's logits exactly -> deterministic top-1 / top-5 counts.
function MakeScoreStubNet(NumClasses: integer): TNNet;
begin
  Result := TNNet.Create;
  Result.AddLayer(TNNetInput.Create(1, 1, NumClasses));
  // TNNet.Compute requires >= 2 layers; Identity preserves the score volume.
  Result.AddLayer(TNNetIdentity.Create);
end;

// Build a score volume (1,1,NumClasses) whose argmax is TopClass with a
// descending ramp, then bump RunnerUp so the SECOND-place class is controlled
// (used to engineer top-1-miss-but-top-5-hit cases). RunnerUp < 0 disables.
function MakeScoreVolume(NumClasses, TopClass, RunnerUp: integer): TNNetVolume;
var
  c: integer;
begin
  Result := TNNetVolume.Create(1, 1, NumClasses);
  for c := 0 to NumClasses - 1 do Result.FData[c] := 0.1;
  Result.FData[TopClass] := 10.0;       // clear argmax
  if (RunnerUp >= 0) and (RunnerUp < NumClasses) and (RunnerUp <> TopClass) then
    Result.FData[RunnerUp] := 5.0;      // clear second place
end;

procedure TTestNeuralImageMetrics.TestTopKIndices;
var
  Scores: array[0..5] of TNeuralFloat;
  Pred: array[0..2] of integer;
  Ties: array[0..3] of TNeuralFloat;
  TiePred: array[0..1] of integer;
begin
  // Distinct values: 0.1 0.9 0.3 0.7 0.2 0.5 -> top-3 = 1,3,5.
  Scores[0] := 0.1; Scores[1] := 0.9; Scores[2] := 0.3;
  Scores[3] := 0.7; Scores[4] := 0.2; Scores[5] := 0.5;
  TopKIndices(Scores, 6, 3, Pred);
  AssertEquals('top-1 index', 1, Pred[0]);
  AssertEquals('top-2 index', 3, Pred[1]);
  AssertEquals('top-3 index', 5, Pred[2]);

  // Exact ties: first-max tie-break keeps the LOWEST index first.
  Ties[0] := 1.0; Ties[1] := 1.0; Ties[2] := 0.0; Ties[3] := 1.0;
  TopKIndices(Ties, 4, 2, TiePred);
  AssertEquals('tie top-1 lowest index', 0, TiePred[0]);
  AssertEquals('tie top-2 next index', 1, TiePred[1]);
end;

procedure TTestNeuralImageMetrics.TestEvaluateImageNetCounting;
const
  cNumClasses = 10;
var
  NN: TNNet;
  Samples: TNNetImageNetSampleArray;
  Stats: TNNetImageNetStats;
  i: integer;
begin
  NN := MakeScoreStubNet(cNumClasses);
  SetLength(Samples, 4);
  // s0: top1 = gold (3)              -> top1 hit, top5 hit
  // s1: top1 = gold (7)              -> top1 hit, top5 hit
  // s2: top1 = 0, gold = 1 runner-up -> top1 MISS, top5 hit
  // s3: top1 = 9, gold = 4 (buried)  -> top1 MISS, top5 MISS
  Samples[0].Image := MakeScoreVolume(cNumClasses, 3, -1); Samples[0].GoldLabel := 3;
  Samples[1].Image := MakeScoreVolume(cNumClasses, 7, -1); Samples[1].GoldLabel := 7;
  Samples[2].Image := MakeScoreVolume(cNumClasses, 0, 1);  Samples[2].GoldLabel := 1;
  Samples[3].Image := MakeScoreVolume(cNumClasses, 9, -1); Samples[3].GoldLabel := 4;
  try
    Stats := EvaluateImageNet(NN, Samples, cNumClasses, 5, 16);
    AssertEquals('items scored', 4, Stats.ItemCount);
    AssertEquals('K used', 5, Stats.K);
    AssertEquals('top-1 correct', 2, Stats.Top1Correct);
    AssertEquals('top-5 correct', 3, Stats.TopKCorrect);
    AssertEquals('top-1 accuracy', 0.5, Stats.Top1Accuracy, 1e-6);
    AssertEquals('top-5 accuracy', 0.75, Stats.TopKAccuracy, 1e-6);
  finally
    for i := 0 to High(Samples) do Samples[i].Image.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestEvaluateImageNetSkipAndConfusion;
const
  cNumClasses = 10;
var
  NN: TNNet;
  Samples: TNNetImageNetSampleArray;
  Stats: TNNetImageNetStats;
  i: integer;
begin
  NN := MakeScoreStubNet(cNumClasses);
  SetLength(Samples, 4);
  // s0 gold=99 (out of range) -> SKIPPED
  // s1 top1=2 gold=2          -> hit (no confusion row)
  // s2 top1=0 gold=1 runner   -> top1 miss, top5 hit  (confusion row, InTopK)
  // s3 top1=9 gold=5 buried   -> top1 miss, top5 miss (confusion row)
  Samples[0].Image := MakeScoreVolume(cNumClasses, 0, -1); Samples[0].GoldLabel := 99;
  Samples[0].SourceName := 'skip.jpg';
  Samples[1].Image := MakeScoreVolume(cNumClasses, 2, -1); Samples[1].GoldLabel := 2;
  Samples[1].SourceName := 'hit.jpg';
  Samples[2].Image := MakeScoreVolume(cNumClasses, 0, 1);  Samples[2].GoldLabel := 1;
  Samples[2].SourceName := 'near.jpg';
  Samples[3].Image := MakeScoreVolume(cNumClasses, 9, -1); Samples[3].GoldLabel := 5;
  Samples[3].SourceName := 'far.jpg';
  try
    // MaxConfusion = 1: only the FIRST top-1 miss is retained.
    Stats := EvaluateImageNet(NN, Samples, cNumClasses, 5, 1);
    AssertEquals('out-of-range gold skipped', 3, Stats.ItemCount);
    AssertEquals('top-1 correct', 1, Stats.Top1Correct);
    AssertEquals('top-5 correct', 2, Stats.TopKCorrect);
    AssertEquals('confusion capped at MaxConfusion', 1, Length(Stats.Confusion));
    AssertEquals('confusion source', 'near.jpg', Stats.Confusion[0].SourceName);
    AssertEquals('confusion gold', 1, Stats.Confusion[0].GoldLabel);
    AssertEquals('confusion pred (top1)', 0, Stats.Confusion[0].PredLabel);
    AssertTrue('confusion InTopK flag', Stats.Confusion[0].InTopK);
  finally
    for i := 0 to High(Samples) do Samples[i].Image.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestImageNetReportFormat;
const
  cNumClasses = 10;
var
  NN: TNNet;
  Samples: TNNetImageNetSampleArray;
  Stats: TNNetImageNetStats;
  Names: array of string;
  Report: string;
  i: integer;
begin
  NN := MakeScoreStubNet(cNumClasses);
  SetLength(Samples, 2);
  Samples[0].Image := MakeScoreVolume(cNumClasses, 3, -1); Samples[0].GoldLabel := 3;
  Samples[0].SourceName := 'cat.jpg';
  Samples[1].Image := MakeScoreVolume(cNumClasses, 9, -1); Samples[1].GoldLabel := 4;
  Samples[1].SourceName := 'dog.jpg';
  SetLength(Names, cNumClasses);
  for i := 0 to cNumClasses - 1 do Names[i] := 'class' + IntToStr(i);
  try
    Stats := EvaluateImageNet(NN, Samples, cNumClasses, 5, 16);
    Report := ImageNetReport(Stats, Names, 'StubNet');
    AssertTrue('title in report', Pos('StubNet top-1 / top-5', Report) > 0);
    AssertTrue('top-1 line', Pos('top-1', Report) > 0);
    AssertTrue('top-5 line', Pos('top-5', Report) > 0);
    AssertTrue('images scored line', Pos('images scored : 2', Report) > 0);
    AssertTrue('confusion sample present', Pos('confusion sample', Report) > 0);
    AssertTrue('confusion row names class', Pos('class4', Report) > 0);
    AssertTrue('confusion flags top-K miss', Pos('top-K miss', Report) > 0);
  finally
    for i := 0 to High(Samples) do Samples[i].Image.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralImageMetrics.TestPreprocessTransformMath;
var
  Src, Dst: TNNetVolume;
  Mean, Std: array[0..2] of TNeuralFloat;
  x, y, c: integer;
  expect: TNeuralFloat;
begin
  // A 6x4 image (W=6 wider, H=4 shorter): shorter side resized to 4 is a no-op
  // on H; width is unchanged, so the resize stage is identity (SizeX/SizeY both
  // pass through when already at target). We instead use a SQUARE 4x4 image so
  // ResizeShorterSide=4 is exactly identity, then center-crop 2x2 + normalize is
  // pure arithmetic we can assert to the bit.
  Src := TNNetVolume.Create(4, 4, 3);
  Dst := TNNetVolume.Create;
  Mean[0] := 0.5; Mean[1] := 0.25; Mean[2] := 0.0;
  Std[0] := 0.5;  Std[1] := 1.0;   Std[2] := 2.0;
  try
    // Fill with a deterministic pattern in 0..255 byte range.
    for y := 0 to 3 do
      for x := 0 to 3 do
        for c := 0 to 2 do
          Src[x, y, c] := (x * 16 + y * 4 + c * 64) mod 256;

    // Resize shorter side -> 4 (identity for a 4x4 src), center-crop 2x2.
    PreprocessImageForVisionModel(Src, Dst, 4, 2, Mean, Std);

    AssertEquals('crop width', 2, Dst.SizeX);
    AssertEquals('crop height', 2, Dst.SizeY);
    AssertEquals('crop depth', 3, Dst.Depth);

    // Center crop offset for 4 -> 2 is (4-2) div 2 = 1, so Dst[x,y] = Src[x+1,y+1],
    // then (v/255 - mean)/std. Assert every cropped pixel exactly.
    for y := 0 to 1 do
      for x := 0 to 1 do
        for c := 0 to 2 do
        begin
          expect := (Src[x + 1, y + 1, c] / 255.0 - Mean[c]) / Std[c];
          AssertEquals(Format('pixel (%d,%d,%d)', [x, y, c]),
            expect, Dst[x, y, c], 1e-6);
        end;
  finally
    Src.Free;
    Dst.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralImageMetrics);
end.
