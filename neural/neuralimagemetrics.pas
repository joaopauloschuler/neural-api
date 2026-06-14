unit neuralimagemetrics;

(*
neuralimagemetrics
Generative-image quality metrics: the Frechet Inception Distance (FID,
Heusel et al. 2017) and the Inception Score (IS, Salimans et al. 2016).

These are the standard scalar yardsticks for comparing a set of GENERATED
images against a set of REAL images (FID) or for scoring the diversity +
confidence of a generator's own samples (IS). The repo already ships a deep
NLP-eval bench (perplexity / BLEU / ROUGE) and CV classification reports but
had ZERO image-generative metrics, so VisualGAN / DiffusionMNIST /
FlowMatching / VisualAutoencoder could only be compared by eye.

BACKBONE-AGNOSTIC BY DESIGN. The canonical FID/IS use an ImageNet-pretrained
Inception-v3 (the 2048-d "pool3" features for FID; the 1000-way softmax for
IS). That backbone is NOT yet available in this repo. Rather than hardcode a
network, every function here takes the already-extracted FEATURE vectors (for
FID) or the already-computed class-PROBABILITY vectors (for IS). The caller
chooses the backbone:

  * For the canonical metric, feed real Inception-v3 features once it lands.
  * As a DOCUMENTED PROXY (the task explicitly permits "a smaller landed
    CNN"), run any landed classifier CNN over the images, take a late hidden
    layer (e.g. the global-average-pooled features before the classification
    head) as the FID feature vector, and the final softmax as the IS prob
    vector. Absolute FID values are then only comparable WITHIN the same
    proxy backbone (as is true of FID generally - it is backbone-relative),
    but the ORDERING of generators is preserved, which is what FID is used
    for. examples/GenerativeImageMetrics demonstrates the proxy path with a
    small MNIST CNN.

NUMERICS. The feature dimension d for a proxy CNN is small (tens to a few
hundred); the covariance is d x d. Everything internal is done in DOUBLE
(float64) regardless of TNeuralFloat, because the Frechet distance subtracts
two large nearly-equal traces and the matrix square-root needs the extra
precision. The matrix square-root of the d x d SPD covariance is computed by
a SYMMETRIC JACOBI eigendecomposition (cyclic sweeps), sqrt of the
(clamped-nonnegative) eigenvalues, and reconstruction V*diag(sqrt(L))*V^T.
Jacobi is O(d^3) per sweep and converges in ~6-10 sweeps; it is exact for
symmetric matrices and needs no external LAPACK. Documented dim ceiling:
comfortable to a few thousand; beyond that prefer a divide-and-conquer
solver. For FID the cross term Tr(sqrt(Cr*Cg)) uses the identity
Tr(sqrt(A*B)) = Tr(sqrt(sqrt(A)*B*sqrt(A))) (the symmetrised product is SPD,
so its eigenvalues are real and the trace of the sqrt is the sum of their
sqrts), which keeps the whole computation inside the symmetric-eigensolver.

ACCUMULATOR. TFIDFeatureAccumulator keeps a running sum and outer-product
sum so features need not be held in RAM: Add() one feature vector at a time,
then Mean / Covariance give the unbiased (N-1) sample statistics, or feed two
accumulators straight to ComputeFIDFromAccumulators.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, neuralvolume;

type
  // Double-precision helpers (FID needs more than single precision).
  TIMDoubleArray   = array of Double;
  TIMDoubleMatrix  = array of TIMDoubleArray;  // row-major [row][col]

  { TFIDFeatureAccumulator }
  // Streaming mean + covariance over feature vectors of fixed dimension Dim.
  // Add() one vector at a time (no need to hold the whole feature matrix);
  // Mean and Covariance return the unbiased sample statistics.
  // Coded by Claude (AI).
  TFIDFeatureAccumulator = class
  private
    FDim: integer;
    FCount: Int64;
    FSum: TIMDoubleArray;            // sum_i x_i              (Dim)
    FOuter: TIMDoubleMatrix;         // sum_i x_i x_i^T        (Dim x Dim)
  public
    constructor Create(ADim: integer);
    procedure Clear;
    // Add one feature vector. Length must equal Dim.
    procedure Add(const AFeature: array of Double); overload;
    procedure Add(const AFeature: TNeuralFloatDynArr); overload;
    // Add the last-layer output (or any chosen layer) of a network as the
    // feature vector. Reads Volume.Raw[0..Dim-1].
    procedure AddVolume(AVolume: TNNetVolume);
    function Mean: TIMDoubleArray;
    // Unbiased (divide by Count-1) sample covariance, Dim x Dim.
    function Covariance: TIMDoubleMatrix;
    property Dim: integer read FDim;
    property Count: Int64 read FCount;
  end;

// --- Linear-algebra primitives (public so tests can hit them directly) ---

// Symmetric Jacobi eigendecomposition of the n x n symmetric matrix A.
// On return EigVals[0..n-1] holds the eigenvalues and EigVecs[*][k] the
// corresponding (column) eigenvector. A is not modified.
procedure SymmetricEigenJacobi(const A: TIMDoubleMatrix;
  out EigVals: TIMDoubleArray; out EigVecs: TIMDoubleMatrix;
  MaxSweeps: integer = 100);

// SPD matrix square-root via eigendecomposition: sqrt(A) =
// V * diag(sqrt(max(lambda,0))) * V^T. A must be symmetric.
function MatrixSqrtSPD(const A: TIMDoubleMatrix): TIMDoubleMatrix;

// Trace of sqrt(A*B) for symmetric SPD A,B, computed as
// sum sqrt(eig( sqrtA * B * sqrtA )) (the stable symmetrised form).
function TraceOfSqrtProductSPD(const A, B: TIMDoubleMatrix): Double;

// --- FID ---

// Core FID from the two sets' statistics (means + covariances):
//   ||muR - muG||^2 + Tr(Cr + Cg - 2*sqrt(Cr*Cg)).
function ComputeFID(const MeanR, MeanG: TIMDoubleArray;
  const CovR, CovG: TIMDoubleMatrix): Double;

// Convenience: FID directly from two filled accumulators.
function ComputeFIDFromAccumulators(AccR, AccG: TFIDFeatureAccumulator): Double;

// Convenience: FID from two raw feature matrices (Features[sample][dim]).
function ComputeFIDFromFeatures(const FeaturesR, FeaturesG: TIMDoubleMatrix): Double;

// --- Inception Score ---

// Inception Score over a set of softmax class-probability vectors
//   Probs[sample][class]: exp( E_x KL(p(y|x) || p(y)) ).
// Splits the samples into NumSplits chunks, scores each, and returns the
// mean (Score) and population std-dev (StdDev) across splits (the standard
// reporting convention). With NumSplits = 1 StdDev is 0.
procedure ComputeInceptionScore(const Probs: TIMDoubleMatrix;
  NumSplits: integer; out Score: Double; out StdDev: Double);
// Mean-only convenience.
function InceptionScore(const Probs: TIMDoubleMatrix;
  NumSplits: integer = 1): Double;

implementation

const
  cEPS = 1e-12;

{ TFIDFeatureAccumulator }

constructor TFIDFeatureAccumulator.Create(ADim: integer);
begin
  inherited Create;
  if ADim <= 0 then
    raise Exception.Create('TFIDFeatureAccumulator: Dim must be > 0');
  FDim := ADim;
  Clear;
end;

procedure TFIDFeatureAccumulator.Clear;
var
  i: integer;
begin
  FCount := 0;
  SetLength(FSum, FDim);
  SetLength(FOuter, FDim);
  for i := 0 to FDim - 1 do
  begin
    FSum[i] := 0;
    SetLength(FOuter[i], FDim);
    FillChar(FOuter[i][0], FDim * SizeOf(Double), 0);
  end;
end;

procedure TFIDFeatureAccumulator.Add(const AFeature: array of Double);
var
  i, j: integer;
  xi: Double;
begin
  if Length(AFeature) <> FDim then
    raise Exception.CreateFmt(
      'TFIDFeatureAccumulator.Add: feature length %d <> Dim %d',
      [Length(AFeature), FDim]);
  Inc(FCount);
  for i := 0 to FDim - 1 do
  begin
    xi := AFeature[i];
    FSum[i] := FSum[i] + xi;
    for j := 0 to FDim - 1 do
      FOuter[i][j] := FOuter[i][j] + xi * AFeature[j];
  end;
end;

procedure TFIDFeatureAccumulator.Add(const AFeature: TNeuralFloatDynArr);
var
  d: TIMDoubleArray;
  i: integer;
begin
  SetLength(d, Length(AFeature));
  for i := 0 to Length(AFeature) - 1 do
    d[i] := AFeature[i];
  Add(d);
end;

procedure TFIDFeatureAccumulator.AddVolume(AVolume: TNNetVolume);
var
  d: TIMDoubleArray;
  i: integer;
begin
  if AVolume.Size < FDim then
    raise Exception.CreateFmt(
      'TFIDFeatureAccumulator.AddVolume: volume size %d < Dim %d',
      [AVolume.Size, FDim]);
  SetLength(d, FDim);
  for i := 0 to FDim - 1 do
    d[i] := AVolume.FData[i];
  Add(d);
end;

function TFIDFeatureAccumulator.Mean: TIMDoubleArray;
var
  i: integer;
begin
  if FCount = 0 then
    raise Exception.Create('TFIDFeatureAccumulator.Mean: no samples');
  SetLength(Result, FDim);
  for i := 0 to FDim - 1 do
    Result[i] := FSum[i] / FCount;
end;

function TFIDFeatureAccumulator.Covariance: TIMDoubleMatrix;
var
  i, j: integer;
  mu: TIMDoubleArray;
  denom: Double;
begin
  if FCount < 2 then
    raise Exception.Create(
      'TFIDFeatureAccumulator.Covariance: need at least 2 samples');
  mu := Mean;
  denom := FCount - 1;  // unbiased
  SetLength(Result, FDim);
  for i := 0 to FDim - 1 do
  begin
    SetLength(Result[i], FDim);
    for j := 0 to FDim - 1 do
      // Cov[i][j] = (sum xi xj - N mu_i mu_j) / (N-1)
      Result[i][j] := (FOuter[i][j] - FCount * mu[i] * mu[j]) / denom;
  end;
end;

{ Linear algebra }

procedure SymmetricEigenJacobi(const A: TIMDoubleMatrix;
  out EigVals: TIMDoubleArray; out EigVecs: TIMDoubleMatrix;
  MaxSweeps: integer = 100);
var
  n, i, j, k, sweep: integer;
  M: TIMDoubleMatrix;
  offdiag, theta, t, c, s, tau, g, h, aij: Double;
  Mik, Mjk, Vik, Vjk: Double;
begin
  n := Length(A);
  // Working copy of A (symmetrised defensively).
  SetLength(M, n);
  for i := 0 to n - 1 do
  begin
    SetLength(M[i], n);
    for j := 0 to n - 1 do
      M[i][j] := 0.5 * (A[i][j] + A[j][i]);
  end;
  // V starts as identity.
  SetLength(EigVecs, n);
  for i := 0 to n - 1 do
  begin
    SetLength(EigVecs[i], n);
    for j := 0 to n - 1 do
      if i = j then EigVecs[i][j] := 1 else EigVecs[i][j] := 0;
  end;

  for sweep := 1 to MaxSweeps do
  begin
    // Sum of off-diagonal magnitudes; stop when negligible.
    offdiag := 0;
    for i := 0 to n - 1 do
      for j := i + 1 to n - 1 do
        offdiag := offdiag + Abs(M[i][j]);
    if offdiag <= cEPS then Break;

    for i := 0 to n - 2 do
      for j := i + 1 to n - 1 do
      begin
        aij := M[i][j];
        if Abs(aij) <= cEPS then Continue;
        // Jacobi rotation that zeros M[i][j].
        h := M[j][j] - M[i][i];
        if Abs(h) <= cEPS then
        begin
          theta := PI / 4;
          if aij < 0 then theta := -theta;
          t := Sin(theta) / Cos(theta);
        end
        else
        begin
          theta := 0.5 * h / aij;
          t := 1.0 / (Abs(theta) + Sqrt(theta * theta + 1.0));
          if theta < 0 then t := -t;
        end;
        c := 1.0 / Sqrt(t * t + 1.0);
        s := t * c;
        tau := s / (1.0 + c);

        // Update diagonals and the (i,j) entry.
        g := t * aij;
        M[i][i] := M[i][i] - g;
        M[j][j] := M[j][j] + g;
        M[i][j] := 0;
        M[j][i] := 0;

        // Update remaining entries in rows/cols i and j.
        for k := 0 to n - 1 do
          if (k <> i) and (k <> j) then
          begin
            Mik := M[i][k];
            Mjk := M[j][k];
            M[i][k] := Mik - s * (Mjk + tau * Mik);
            M[k][i] := M[i][k];
            M[j][k] := Mjk + s * (Mik - tau * Mjk);
            M[k][j] := M[j][k];
          end;

        // Accumulate eigenvectors.
        for k := 0 to n - 1 do
        begin
          Vik := EigVecs[k][i];
          Vjk := EigVecs[k][j];
          EigVecs[k][i] := Vik - s * (Vjk + tau * Vik);
          EigVecs[k][j] := Vjk + s * (Vik - tau * Vjk);
        end;
      end;
  end;

  SetLength(EigVals, n);
  for i := 0 to n - 1 do
    EigVals[i] := M[i][i];
end;

function MatrixSqrtSPD(const A: TIMDoubleMatrix): TIMDoubleMatrix;
var
  n, i, j, k: integer;
  vals: TIMDoubleArray;
  vecs: TIMDoubleMatrix;
  sl: TIMDoubleArray;
  acc: Double;
begin
  n := Length(A);
  SymmetricEigenJacobi(A, vals, vecs);
  SetLength(sl, n);
  for i := 0 to n - 1 do
    if vals[i] > 0 then sl[i] := Sqrt(vals[i]) else sl[i] := 0;
  // Result = V * diag(sl) * V^T
  SetLength(Result, n);
  for i := 0 to n - 1 do
  begin
    SetLength(Result[i], n);
    for j := 0 to n - 1 do
    begin
      acc := 0;
      for k := 0 to n - 1 do
        acc := acc + vecs[i][k] * sl[k] * vecs[j][k];
      Result[i][j] := acc;
    end;
  end;
end;

// C = A * B (n x n).
function MatMul(const A, B: TIMDoubleMatrix): TIMDoubleMatrix;
var
  n, i, j, k: integer;
  acc: Double;
begin
  n := Length(A);
  SetLength(Result, n);
  for i := 0 to n - 1 do
  begin
    SetLength(Result[i], n);
    for j := 0 to n - 1 do
    begin
      acc := 0;
      for k := 0 to n - 1 do
        acc := acc + A[i][k] * B[k][j];
      Result[i][j] := acc;
    end;
  end;
end;

function TraceOfSqrtProductSPD(const A, B: TIMDoubleMatrix): Double;
var
  sqrtA, prod, sym: TIMDoubleMatrix;
  vals: TIMDoubleArray;
  vecs: TIMDoubleMatrix;
  n, i, j: integer;
begin
  n := Length(A);
  // M = sqrtA * B * sqrtA is SPD and similar to A*B, so eig(M) = eig(A*B)
  // and Tr(sqrt(A*B)) = sum sqrt(eig(M)).
  sqrtA := MatrixSqrtSPD(A);
  prod := MatMul(MatMul(sqrtA, B), sqrtA);
  // Symmetrise to kill rounding asymmetry before the symmetric solver.
  SetLength(sym, n);
  for i := 0 to n - 1 do
  begin
    SetLength(sym[i], n);
    for j := 0 to n - 1 do
      sym[i][j] := 0.5 * (prod[i][j] + prod[j][i]);
  end;
  SymmetricEigenJacobi(sym, vals, vecs);
  Result := 0;
  for i := 0 to n - 1 do
    if vals[i] > 0 then Result := Result + Sqrt(vals[i]);
end;

{ FID }

function ComputeFID(const MeanR, MeanG: TIMDoubleArray;
  const CovR, CovG: TIMDoubleMatrix): Double;
var
  n, i: integer;
  diff, traceTerm, crossTr: Double;
begin
  n := Length(MeanR);
  if Length(MeanG) <> n then
    raise Exception.Create('ComputeFID: mean dimension mismatch');
  // ||muR - muG||^2
  diff := 0;
  for i := 0 to n - 1 do
    diff := diff + Sqr(MeanR[i] - MeanG[i]);
  // Tr(Cr) + Tr(Cg)
  traceTerm := 0;
  for i := 0 to n - 1 do
    traceTerm := traceTerm + CovR[i][i] + CovG[i][i];
  // 2 * Tr(sqrt(Cr*Cg))
  crossTr := TraceOfSqrtProductSPD(CovR, CovG);
  Result := diff + traceTerm - 2.0 * crossTr;
  // Tiny negative values are pure rounding; clamp at 0.
  if Result < 0 then Result := 0;
end;

function ComputeFIDFromAccumulators(AccR, AccG: TFIDFeatureAccumulator): Double;
begin
  if AccR.Dim <> AccG.Dim then
    raise Exception.Create('ComputeFIDFromAccumulators: dim mismatch');
  Result := ComputeFID(AccR.Mean, AccG.Mean, AccR.Covariance, AccG.Covariance);
end;

function ComputeFIDFromFeatures(const FeaturesR, FeaturesG: TIMDoubleMatrix): Double;
var
  accR, accG: TFIDFeatureAccumulator;
  dim, i: integer;
begin
  if (Length(FeaturesR) = 0) or (Length(FeaturesG) = 0) then
    raise Exception.Create('ComputeFIDFromFeatures: empty feature set');
  dim := Length(FeaturesR[0]);
  accR := TFIDFeatureAccumulator.Create(dim);
  accG := TFIDFeatureAccumulator.Create(dim);
  try
    for i := 0 to Length(FeaturesR) - 1 do accR.Add(FeaturesR[i]);
    for i := 0 to Length(FeaturesG) - 1 do accG.Add(FeaturesG[i]);
    Result := ComputeFIDFromAccumulators(accR, accG);
  finally
    accR.Free;
    accG.Free;
  end;
end;

{ Inception Score }

// IS over one block of probability rows: exp( mean_x KL(p(y|x) || pbar) ).
function ISBlock(const Probs: TIMDoubleMatrix; First, Last: integer): Double;
var
  numClass, i, c, n: integer;
  pbar: TIMDoubleArray;
  klSum, kl, p, q: Double;
begin
  numClass := Length(Probs[First]);
  n := Last - First + 1;
  SetLength(pbar, numClass);
  for c := 0 to numClass - 1 do pbar[c] := 0;
  for i := First to Last do
    for c := 0 to numClass - 1 do
      pbar[c] := pbar[c] + Probs[i][c];
  for c := 0 to numClass - 1 do
    pbar[c] := pbar[c] / n;

  klSum := 0;
  for i := First to Last do
  begin
    kl := 0;
    for c := 0 to numClass - 1 do
    begin
      p := Probs[i][c];
      if p > cEPS then
      begin
        q := pbar[c];
        if q < cEPS then q := cEPS;
        kl := kl + p * (Ln(p) - Ln(q));
      end;
    end;
    klSum := klSum + kl;
  end;
  Result := Exp(klSum / n);
end;

procedure ComputeInceptionScore(const Probs: TIMDoubleMatrix;
  NumSplits: integer; out Score: Double; out StdDev: Double);
var
  total, split, first, last, sz: integer;
  scores: TIMDoubleArray;
  mean, varSum: Double;
  i: integer;
begin
  total := Length(Probs);
  if total = 0 then
    raise Exception.Create('ComputeInceptionScore: no samples');
  if NumSplits < 1 then NumSplits := 1;
  if NumSplits > total then NumSplits := total;

  SetLength(scores, NumSplits);
  for split := 0 to NumSplits - 1 do
  begin
    // Contiguous split [first..last]; spread the remainder over early splits.
    first := (split * total) div NumSplits;
    last := ((split + 1) * total) div NumSplits - 1;
    sz := last - first + 1;
    if sz < 1 then
    begin
      scores[split] := 1.0;
      Continue;
    end;
    scores[split] := ISBlock(Probs, first, last);
  end;

  mean := 0;
  for i := 0 to NumSplits - 1 do mean := mean + scores[i];
  mean := mean / NumSplits;
  varSum := 0;
  for i := 0 to NumSplits - 1 do varSum := varSum + Sqr(scores[i] - mean);
  Score := mean;
  if NumSplits > 1 then
    StdDev := Sqrt(varSum / NumSplits)  // population std (standard IS report)
  else
    StdDev := 0;
end;

function InceptionScore(const Probs: TIMDoubleMatrix; NumSplits: integer): Double;
var
  sd: Double;
begin
  ComputeInceptionScore(Probs, NumSplits, Result, sd);
end;

end.
