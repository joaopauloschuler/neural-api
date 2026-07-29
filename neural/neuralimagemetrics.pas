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
  Classes, SysUtils, Math, neuralvolume, neuralnetwork;

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

// --- FID over the real Inception backbone ---
//
// The functions above are backbone-AGNOSTIC: the caller supplies feature
// vectors. These wire FID onto the actual Inception-v3 pooled feature, which
// is the canonical FID definition. The InceptionNet is built by
// neuralpretrained.BuildInceptionV3Full (full_arch=true), which returns the
// global-avg-pool layer index via its out PoolFeatureIdx parameter; that layer
// emits the 2048-d "pool3" activation FID is defined over (the width is smaller
// for a width_div-shrunk or pico net, but the API is identical).
//
// Each image must already be a network-ready volume: (ImageSize, ImageSize,
// NumChannels), ImageNet-normalised the same way the backbone was trained
// ((pixel/255 - csImageNetMean[c]) / csImageNetStd[c]) and resized/center-
// cropped to the backbone's ImageSize. neuraldatasets.LoadImageForVisionModel
// produces exactly such a volume.

// Runs one image volume through InceptionNet and returns the pooled feature at
// PoolFeatureIdx as a float64 vector (length = that layer's Output.Size).
function ExtractInceptionFeature(InceptionNet: TNNet;
  PoolFeatureIdx: integer; ImageVolume: TNNetVolume): TIMDoubleArray;

// Runs every image in Images through InceptionNet, accumulating the pooled
// PoolFeatureIdx feature of each into Acc (which must already be sized to the
// pooled width). Lets a caller stream large image sets without materialising
// the whole feature matrix. Images is a list of network-ready TNNetVolume.
procedure AccumulateInceptionFeatures(InceptionNet: TNNet;
  PoolFeatureIdx: integer; Images: TNNetVolumeList;
  Acc: TFIDFeatureAccumulator);

// The real-backbone FID: extracts the 2048-d Inception pooled feature for every
// real and generated image, then returns the Frechet distance between the two
// resulting Gaussians (reusing the mean/cov/matrix-sqrt math above). Both image
// lists must hold network-ready volumes (see preprocessing note); both must
// have >= 2 images so the per-set covariance is defined. FID(X, X) == 0.
function ComputeInceptionFID(InceptionNet: TNNet; PoolFeatureIdx: integer;
  RealImages, GenImages: TNNetVolumeList): Double;

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

// --- PSNR / SSIM / MS-SSIM (full-reference signal-statistics metrics) ---
//
// These are the classic PIXEL/STRUCTURE quality metrics (Wang et al. 2004,
// 2003), complementing the FEATURE-space FID/IS above. No backbone: they are
// pure statistics on the two images themselves. Images are passed as flat
// row-major Double arrays with explicit H, W, Channels (channel-last:
// index = (y*W + x)*Channels + c). DataRange (L) is the dynamic range of the
// signal (max-min): 1.0 for [0,1]-normalised images (the default convention
// in this repo), 255.0 for 8-bit. All math is float64.
// Coded by Claude (AI).

// Peak Signal-to-Noise Ratio: 10*log10(L^2/MSE). Identical images -> MSE 0 ->
// returns Math.Infinity (+Inf). Multi-channel MSE is over all pixels and
// channels jointly (the standard definition).
function ComputePSNR(const ImgA, ImgB: TIMDoubleArray;
  H, W, Channels: integer; DataRange: Double = 1.0): Double;

// Structural Similarity index, 11x11 Gaussian window (sigma 1.5),
// C1=(K1*L)^2, C2=(K2*L)^2 with K1=0.01, K2=0.03. Border handling matches
// skimage.metrics.structural_similarity(gaussian_weights=True) DEFAULT, i.e.
// 'valid'-style: the window slides only over fully-inside positions, so the
// SSIM map is (H-10) x (W-10) and the result is its mean. Multi-channel:
// SSIM is computed per channel and averaged (channel_axis behaviour).
function ComputeSSIM(const ImgA, ImgB: TIMDoubleArray;
  H, W, Channels: integer; DataRange: Double = 1.0): Double;

// Multi-Scale SSIM (Wang et al. 2003), 5 scales with the canonical weights
// [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]. At each scale the contrast (c)
// and structure (s) terms are formed from the same 11x11 Gaussian window;
// between scales the images are downsampled by a 2x2 average pool. The score
// is prod_{j=1..M-1} (cs_j)^w_j * (l_M * cs_M)^w_M, i.e. only the
// contrast-structure term is kept at every scale except the last, which also
// contributes the luminance (l) term. cs and l are the means over the valid
// window map at that scale. Multi-channel images are averaged over channels.
function ComputeMSSSIM(const ImgA, ImgB: TIMDoubleArray;
  H, W, Channels: integer; DataRange: Double = 1.0): Double;

// Differentiable 1 - SSIM training-loss helper. Fills GradA (same length as
// ImgA, channel-last) with d(1-SSIM)/d(ImgA) so a restoration trainer can use
// structural similarity as the objective instead of pixel MSE; returns the
// scalar loss 1 - mean(SSIM map). ImgA is the prediction, ImgB the target
// (treated as constant). Gradient is w.r.t. the single-channel-averaged SSIM
// with the same 'valid' window map as ComputeSSIM. Verified against a central
// difference. (A loss HELPER rather than a TNNet* layer: the windowed SSIM
// gradient is heavy and self-contained, and the task explicitly permits a
// documented ComputeSSIMLossAndGradient helper.)
function ComputeSSIMLossAndGradient(const ImgA, ImgB: TIMDoubleArray;
  H, W, Channels: integer; out GradA: TIMDoubleArray;
  DataRange: Double = 1.0): Double;

// --- KID (Kernel Inception Distance) ---
//
// The small-sample-UNBIASED complement to FID (Binkowski et al. 2018).
// Backbone-agnostic, same caller-supplies-features convention as ComputeFID:
// FeaturesR/FeaturesG are [sample][dim] matrices. Uses the cubic polynomial
// kernel k(x,y) = (x.y/d + 1)^3 (d = feature dimension) and the UNBIASED
// U-statistic MMD^2 estimator (NO diagonal self-terms):
//   MMD^2 = sum_{i<>j} k(x_i,x_j)/(m(m-1))
//         + sum_{i<>j} k(y_i,y_j)/(n(n-1))
//         - 2 sum_{i,j} k(x_i,y_j)/(m n).
// KID reports the mean +/- std of this estimator over NumSubsets random
// equal-size splits (subset size = SubsetSize, clipped to each set's count),
// mirroring ComputeInceptionScore's mean+std reporting. With NumSubsets=1 and
// SubsetSize>=both counts it is the single full-set unbiased estimate and
// StdDev=0. Coded by Claude (AI).

// Single unbiased MMD^2 estimate over the two full feature matrices (the math
// core; public so tests can hit it directly).
function ComputeKIDMMD2(const FeaturesR, FeaturesG: TIMDoubleMatrix): Double;

// Subset-bootstrap KID: mean + population std over NumSubsets random splits
// of SubsetSize samples drawn (without replacement) from each set.
procedure ComputeKID(const FeaturesR, FeaturesG: TIMDoubleMatrix;
  SubsetSize, NumSubsets: integer; out Score: Double; out StdDev: Double);

// ---------------------------------------------------------------------------
// ImageNet top-1 / top-5 accuracy harness (classifier IMPORT verification)
// ---------------------------------------------------------------------------
//
// EvaluateImageNet / ImageNetReport are to the imported vision backbones what
// EvaluateMMLU / MMLUReport (neuralnlpmetrics) are to the imported LLMs: an
// end-to-end ACCURACY backstop. Each classifier importer's parity test only
// compares raw logits on one or two tensors, which catches a transposed weight
// but NOT a wrong preprocessing pipeline (resize / center-crop / normalize) or
// a label permutation. Running a folder of labelled ImageNet-val images through
// the real preprocessing transform and the real net, then checking top-1 /
// top-5 against the published numbers, is the missing import-VERIFICATION.
//
// DESIGN. This core takes images that are ALREADY network-ready volumes (the
// (ImageSize, ImageSize, 3) ImageNet-normalised tensor that
// neuraldatasets.LoadImageForVisionModel / PreprocessImageForVisionModel
// produce). The harness therefore depends only on neuralvolume / neuralnetwork
// (NOT on the image codecs in neuraldatasets), exactly as EvaluateMMLU depends
// only on the net and pre-tokenized prompts: the resize/crop/normalize transform
// lives in the caller (the example wires neuraldatasets), and the test can inject
// deterministic synthetic volumes through a stub net so the accuracy numbers are
// pinned. The net's final layer must emit a class-score volume of length
// NumClasses (logits or post-softmax probabilities - argmax / top-k is identical
// either way). Label indices are the standard ImageNet class indices (0..999 for
// the full set); GoldLabel outside 0..NumClasses-1 marks the sample skipped.
//
type
  // One labelled, already-preprocessed evaluation image. Image is a
  // network-ready normalised volume (see DESIGN above); GoldLabel is the true
  // ImageNet class index. SourceName is an optional tag (e.g. the file name)
  // carried through into the confusion sample for human-readable reports.
  TNNetImageNetSample = record
    Image: TNNetVolume;
    GoldLabel: integer;
    SourceName: string;
  end;
  TNNetImageNetSampleArray = array of TNNetImageNetSample;

  // One misclassified item retained for the confusion sample in the report:
  // the true label, the predicted top-1 label, and whether the gold label was
  // anywhere in the top-K (a top-1 miss that is still a top-5 hit is the
  // common, benign case and is flagged so the report can separate the two).
  TNNetImageNetConfusion = record
    SourceName: string;
    GoldLabel: integer;
    PredLabel: integer;   // top-1 prediction
    InTopK: boolean;      // gold label within the top-K predictions
  end;
  TNNetImageNetConfusionArray = array of TNNetImageNetConfusion;

  // Aggregate ImageNet accuracy result. Top1/Top5 are the headline fractions
  // (Top1Correct/ItemCount, TopKCorrect/ItemCount); K is the K actually used
  // for the top-K column (the requested K clipped to NumClasses).
  TNNetImageNetStats = record
    ItemCount: integer;
    Top1Correct: integer;
    TopKCorrect: integer;
    Top1Accuracy: TNeuralFloat;
    TopKAccuracy: TNeuralFloat;
    K: integer;
    NumClasses: integer;
    Confusion: TNNetImageNetConfusionArray; // first MaxConfusion misses
  end;

// Returns, in Pred[0..K-1], the indices of the K largest values of Scores
// (Scores[0..Count-1]), most-confident first, with a deterministic first-max
// tie-break (a lower index wins an exact tie). K is clipped to Count. Public so
// the test can pin the top-K selection independently of any net.
procedure TopKIndices(const Scores: array of TNeuralFloat; Count, K: integer;
  out Pred: array of integer);

// The ImageNet accuracy harness. For every sample whose GoldLabel is a valid
// class index, the net is Compute()d on the (already-preprocessed) image, the
// final layer's NumClasses-length output is read, and the top-1 / top-K
// predictions are formed (TopKIndices). Tallies top-1 (argmax == gold) and
// top-K (gold anywhere in the top-K) accuracy. Up to MaxConfusion top-1 MISSES
// are retained in Stats.Confusion for a human-readable confusion sample. K is
// the top-K to score (5 is the ImageNet convention; clipped to NumClasses).
function EvaluateImageNet(NN: TNNet;
  const Samples: array of TNNetImageNetSample;
  NumClasses: integer; K: integer = 5;
  MaxConfusion: integer = 16): TNNetImageNetStats;

// Formats a TNNetImageNetStats into a small multi-line report (the *Report
// idiom). ClassNames, when supplied (length = NumClasses, or empty), labels the
// confusion-sample rows; otherwise classes are printed as bare indices. Title
// is the header line (e.g. "ResNet-50 ImageNet-val").
function ImageNetReport(const Stats: TNNetImageNetStats;
  const ClassNames: array of string; const Title: string = 'ImageNet'): string;

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
  i, DimM1: integer;
begin
  FCount := 0;
  SetLength(FSum, FDim);
  SetLength(FOuter, FDim);
  DimM1 := FDim - 1;
  for i := 0 to DimM1 do
  begin
    FSum[i] := 0;
    SetLength(FOuter[i], FDim);
    FillChar(FOuter[i][0], FDim * SizeOf(Double), 0);
  end;
end;

procedure TFIDFeatureAccumulator.Add(const AFeature: array of Double);
var
  i, j, DimM1: integer;
  xi: Double;
  row: TIMDoubleArray;   // bound FOuter[i] row (#7)
begin
  if Length(AFeature) <> FDim then
    raise Exception.CreateFmt(
      'TFIDFeatureAccumulator.Add: feature length %d <> Dim %d',
      [Length(AFeature), FDim]);
  Inc(FCount);
  DimM1 := FDim - 1;
  for i := 0 to DimM1 do
  begin
    xi := AFeature[i];
    FSum[i] := FSum[i] + xi;
    row := FOuter[i];   // #7: resolve the outer-product row once, not per j
    for j := 0 to DimM1 do
      row[j] := row[j] + xi * AFeature[j];
  end;
end;

procedure TFIDFeatureAccumulator.Add(const AFeature: TNeuralFloatDynArr);
var
  d: TIMDoubleArray;
  i, MaxIdx: integer;
begin
  SetLength(d, Length(AFeature));
  MaxIdx := Length(AFeature) - 1;
  for i := 0 to MaxIdx do
    d[i] := AFeature[i];
  Add(d);
end;

procedure TFIDFeatureAccumulator.AddVolume(AVolume: TNNetVolume);
var
  d: TIMDoubleArray;
  i, DimM1: integer;
begin
  if AVolume.Size < FDim then
    raise Exception.CreateFmt(
      'TFIDFeatureAccumulator.AddVolume: volume size %d < Dim %d',
      [AVolume.Size, FDim]);
  SetLength(d, FDim);
  DimM1 := FDim - 1;
  for i := 0 to DimM1 do
    d[i] := AVolume.FData[i];
  Add(d);
end;

function TFIDFeatureAccumulator.Mean: TIMDoubleArray;
var
  i, DimM1: integer;
begin
  if FCount = 0 then
    raise Exception.Create('TFIDFeatureAccumulator.Mean: no samples');
  SetLength(Result, FDim);
  DimM1 := FDim - 1;
  for i := 0 to DimM1 do
    Result[i] := FSum[i] / FCount;
end;

function TFIDFeatureAccumulator.Covariance: TIMDoubleMatrix;
var
  i, j, DimM1: integer;
  mu: TIMDoubleArray;
  denom: Double;
begin
  if FCount < 2 then
    raise Exception.Create(
      'TFIDFeatureAccumulator.Covariance: need at least 2 samples');
  mu := Mean;
  denom := FCount - 1;  // unbiased
  SetLength(Result, FDim);
  DimM1 := FDim - 1;
  for i := 0 to DimM1 do
  begin
    SetLength(Result[i], FDim);
    for j := 0 to DimM1 do
      // Cov[i][j] = (sum xi xj - N mu_i mu_j) / (N-1)
      Result[i][j] := (FOuter[i][j] - FCount * mu[i] * mu[j]) / denom;
  end;
end;

{ Linear algebra }

procedure SymmetricEigenJacobi(const A: TIMDoubleMatrix;
  out EigVals: TIMDoubleArray; out EigVecs: TIMDoubleMatrix;
  MaxSweeps: integer = 100);
var
  n, nM1, nM2, i, j, k, sweep, rotations: integer;
  M: TIMDoubleMatrix;
  theta, t, c, s, tau, g, h, aij: Double;
  Mik, Mjk, Vik, Vjk: Double;
  Mi, Mj, Mk, Vk: TIMDoubleArray;   // bound matrix rows (#7)
begin
  n := Length(A);
  nM1 := n - 1;
  nM2 := n - 2;
  // Working copy of A (symmetrised defensively).
  SetLength(M, n);
  for i := 0 to nM1 do
  begin
    SetLength(M[i], n);
    for j := 0 to nM1 do
      M[i][j] := 0.5 * (A[i][j] + A[j][i]);
  end;
  // V starts as identity.
  SetLength(EigVecs, n);
  for i := 0 to nM1 do
  begin
    SetLength(EigVecs[i], n);
    for j := 0 to nM1 do
      if i = j then EigVecs[i][j] := 1 else EigVecs[i][j] := 0;
  end;

  for sweep := 1 to MaxSweeps do
  begin
    // Convergence is decided by whether this sweep rotated anything, not by a
    // sum of off-diagonal magnitudes: the sum is O(n^2) entries each below
    // cEPS, so for a matrix of any size it stays above cEPS long after every
    // individual rotation has stopped qualifying, and the loop then burns the
    // whole MaxSweeps budget doing nothing. A sweep that rotates nothing leaves
    // M and EigVecs untouched, so every later sweep would too - the results are
    // identical to running the budget out.
    rotations := 0;

    for i := 0 to nM2 do
      for j := i + 1 to nM1 do
      begin
        aij := M[i][j];
        if Abs(aij) <= cEPS then Continue;
        Inc(rotations);
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
        Mi := M[i];   // #7: rows i and j are invariant across the k loop
        Mj := M[j];
        for k := 0 to nM1 do
          if (k <> i) and (k <> j) then
          begin
            Mik := Mi[k];
            Mjk := Mj[k];
            Mi[k] := Mik - s * (Mjk + tau * Mik);
            Mj[k] := Mjk + s * (Mik - tau * Mjk);
            Mk := M[k];   // #7: bind the M[k] row for the two symmetric writes
            Mk[i] := Mi[k];
            Mk[j] := Mj[k];
          end;

        // Accumulate eigenvectors.
        for k := 0 to nM1 do
        begin
          Vk := EigVecs[k];   // #7: bind the row once for four accesses
          Vik := Vk[i];
          Vjk := Vk[j];
          Vk[i] := Vik - s * (Vjk + tau * Vik);
          Vk[j] := Vjk + s * (Vik - tau * Vjk);
        end;
      end;

    if rotations = 0 then Break;
  end;

  SetLength(EigVals, n);
  for i := 0 to nM1 do
    EigVals[i] := M[i][i];
end;

function MatrixSqrtSPD(const A: TIMDoubleMatrix): TIMDoubleMatrix;
var
  n, nM1, i, j, k: integer;
  vals: TIMDoubleArray;
  vecs, Vs: TIMDoubleMatrix;
  sl, Vsi, Vj, Ri: TIMDoubleArray;
  acc: Double;
begin
  n := Length(A);
  nM1 := n - 1;
  SymmetricEigenJacobi(A, vals, vecs);
  SetLength(sl, n);
  for i := 0 to nM1 do
    if vals[i] > 0 then sl[i] := Sqrt(vals[i]) else sl[i] := 0;
  // Vs = V * diag(sl), built once in n^2 work, so the n^3 loop below is a plain
  // contiguous dot product of two rows instead of a scaled triple product.
  SetLength(Vs, n);
  for i := 0 to nM1 do
  begin
    SetLength(Vs[i], n);
    Vsi := Vs[i];
    Vj := vecs[i];
    for k := 0 to nM1 do Vsi[k] := Vj[k] * sl[k];
  end;
  // Result = V * diag(sl) * V^T = Vs * V^T
  SetLength(Result, n);
  for i := 0 to nM1 do
  begin
    SetLength(Result[i], n);
    Ri := Result[i];
    Vsi := Vs[i];
    for j := 0 to nM1 do
    begin
      Vj := vecs[j];
      acc := 0;
      for k := 0 to nM1 do
        acc := acc + Vsi[k] * Vj[k];
      Ri[j] := acc;
    end;
  end;
end;

// C = A * B (n x n).
function MatMul(const A, B: TIMDoubleMatrix): TIMDoubleMatrix;
var
  n, nM1, i, j, k: integer;
  aik: Double;
  Ai, Bk, Ri: TIMDoubleArray;
begin
  n := Length(A);
  nM1 := n - 1;
  SetLength(Result, n);
  // i-k-j order: B is walked along a contiguous row instead of down a column of
  // separately-allocated rows, and each row reference is resolved once per k.
  for i := 0 to nM1 do
  begin
    SetLength(Result[i], n);   // zero-filled, so it can accumulate
    Ai := A[i];
    Ri := Result[i];
    for k := 0 to nM1 do
    begin
      aik := Ai[k];
      Bk := B[k];
      for j := 0 to nM1 do
        Ri[j] := Ri[j] + aik * Bk[j];
    end;
  end;
end;

function TraceOfSqrtProductSPD(const A, B: TIMDoubleMatrix): Double;
var
  sqrtA, prod, sym: TIMDoubleMatrix;
  vals: TIMDoubleArray;
  vecs: TIMDoubleMatrix;
  n, nM1, i, j: integer;
begin
  n := Length(A);
  nM1 := n - 1;
  // M = sqrtA * B * sqrtA is SPD and similar to A*B, so eig(M) = eig(A*B)
  // and Tr(sqrt(A*B)) = sum sqrt(eig(M)).
  sqrtA := MatrixSqrtSPD(A);
  prod := MatMul(MatMul(sqrtA, B), sqrtA);
  // Symmetrise to kill rounding asymmetry before the symmetric solver.
  SetLength(sym, n);
  for i := 0 to nM1 do
  begin
    SetLength(sym[i], n);
    for j := 0 to nM1 do
      sym[i][j] := 0.5 * (prod[i][j] + prod[j][i]);
  end;
  SymmetricEigenJacobi(sym, vals, vecs);
  Result := 0;
  for i := 0 to nM1 do
    if vals[i] > 0 then Result := Result + Sqrt(vals[i]);
end;

{ FID }

function ComputeFID(const MeanR, MeanG: TIMDoubleArray;
  const CovR, CovG: TIMDoubleMatrix): Double;
var
  n, nM1, i: integer;
  diff, traceTerm, crossTr: Double;
begin
  n := Length(MeanR);
  if Length(MeanG) <> n then
    raise Exception.Create('ComputeFID: mean dimension mismatch');
  nM1 := n - 1;
  // ||muR - muG||^2
  diff := 0;
  for i := 0 to nM1 do
    diff := diff + Sqr(MeanR[i] - MeanG[i]);
  // Tr(Cr) + Tr(Cg)
  traceTerm := 0;
  for i := 0 to nM1 do
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
  dim, i, MaxR, MaxG: integer;
begin
  if (Length(FeaturesR) = 0) or (Length(FeaturesG) = 0) then
    raise Exception.Create('ComputeFIDFromFeatures: empty feature set');
  dim := Length(FeaturesR[0]);
  accR := TFIDFeatureAccumulator.Create(dim);
  accG := TFIDFeatureAccumulator.Create(dim);
  try
    MaxR := Length(FeaturesR) - 1;
    MaxG := Length(FeaturesG) - 1;
    for i := 0 to MaxR do accR.Add(FeaturesR[i]);
    for i := 0 to MaxG do accG.Add(FeaturesG[i]);
    Result := ComputeFIDFromAccumulators(accR, accG);
  finally
    accR.Free;
    accG.Free;
  end;
end;

{ FID over the real Inception backbone }

function ExtractInceptionFeature(InceptionNet: TNNet;
  PoolFeatureIdx: integer; ImageVolume: TNNetVolume): TIMDoubleArray;
var
  PoolOut: TNNetVolume;
  i, dim: integer;
  dimM1: integer;
begin
  if InceptionNet = nil then
    raise Exception.Create('ExtractInceptionFeature: nil net');
  if (PoolFeatureIdx < 0) or (PoolFeatureIdx >= InceptionNet.CountLayers) then
    raise Exception.CreateFmt(
      'ExtractInceptionFeature: PoolFeatureIdx %d out of range', [PoolFeatureIdx]);
  InceptionNet.Compute(ImageVolume);
  PoolOut := InceptionNet.Layers[PoolFeatureIdx].Output;
  dim := PoolOut.Size;
  SetLength(Result, dim);
  dimM1 := dim - 1;
  for i := 0 to dimM1 do
    Result[i] := PoolOut.FData[i];
end;

procedure AccumulateInceptionFeatures(InceptionNet: TNNet;
  PoolFeatureIdx: integer; Images: TNNetVolumeList;
  Acc: TFIDFeatureAccumulator);
var
  i, iMax: integer;
  feat: TIMDoubleArray;
begin
  if Images = nil then
    raise Exception.Create('AccumulateInceptionFeatures: nil image list');
  iMax := Images.Count - 1;
  for i := 0 to iMax do
  begin
    feat := ExtractInceptionFeature(InceptionNet, PoolFeatureIdx, Images[i]);
    Acc.Add(feat);
  end;
end;

function ComputeInceptionFID(InceptionNet: TNNet; PoolFeatureIdx: integer;
  RealImages, GenImages: TNNetVolumeList): Double;
var
  accR, accG: TFIDFeatureAccumulator;
  dim: integer;
begin
  if (RealImages = nil) or (GenImages = nil) then
    raise Exception.Create('ComputeInceptionFID: nil image list');
  if (RealImages.Count < 2) or (GenImages.Count < 2) then
    raise Exception.Create(
      'ComputeInceptionFID: each image set needs at least 2 images');
  if (PoolFeatureIdx < 0) or (PoolFeatureIdx >= InceptionNet.CountLayers) then
    raise Exception.CreateFmt(
      'ComputeInceptionFID: PoolFeatureIdx %d out of range', [PoolFeatureIdx]);
  // One forward pass primes the net so the pooled width is known.
  InceptionNet.Compute(RealImages[0]);
  dim := InceptionNet.Layers[PoolFeatureIdx].Output.Size;
  accR := TFIDFeatureAccumulator.Create(dim);
  accG := TFIDFeatureAccumulator.Create(dim);
  try
    AccumulateInceptionFeatures(InceptionNet, PoolFeatureIdx, RealImages, accR);
    AccumulateInceptionFeatures(InceptionNet, PoolFeatureIdx, GenImages, accG);
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
  numClass, numClassM1, i, c, n: integer;
  pbar: TIMDoubleArray;
  lnpbar: TIMDoubleArray; // Ln of the clamped marginal, one per class (#5)
  prow: TIMDoubleArray;   // bound Probs[i] row (#7)
  klSum, kl, p, q: Double;
begin
  numClass := Length(Probs[First]);
  numClassM1 := numClass - 1;
  n := Last - First + 1;
  SetLength(pbar, numClass);
  for c := 0 to numClassM1 do pbar[c] := 0;
  for i := First to Last do
  begin
    prow := Probs[i];   // #7: resolve the sample row once
    for c := 0 to numClassM1 do
      pbar[c] := pbar[c] + prow[c];
  end;
  // The marginal and its logarithm are invariant across the sample loop below,
  // so clamp and take Ln once per class instead of once per (sample, class) -
  // half of the KL term's transcendentals. ISBlock runs once per split, not on
  // a compute path, so the extra class-sized array is affordable.
  SetLength(lnpbar, numClass);
  for c := 0 to numClassM1 do
  begin
    q := pbar[c] / n;
    pbar[c] := q;
    if q < cEPS then q := cEPS;
    lnpbar[c] := Ln(q);
  end;

  klSum := 0;
  for i := First to Last do
  begin
    kl := 0;
    prow := Probs[i];   // #7: resolve the sample row once
    for c := 0 to numClassM1 do
    begin
      p := prow[c];
      if p > cEPS then
        kl := kl + p * (Ln(p) - lnpbar[c]);
    end;
    klSum := klSum + kl;
  end;
  Result := Exp(klSum / n);
end;

procedure ComputeInceptionScore(const Probs: TIMDoubleMatrix;
  NumSplits: integer; out Score: Double; out StdDev: Double);
var
  total, split, first, last, sz, NumSplitsM1: integer;
  scores: TIMDoubleArray;
  mean, varSum: Double;
  i: integer;
begin
  total := Length(Probs);
  if total = 0 then
    raise Exception.Create('ComputeInceptionScore: no samples');
  if NumSplits < 1 then NumSplits := 1;
  if NumSplits > total then NumSplits := total;
  NumSplitsM1 := NumSplits - 1;

  SetLength(scores, NumSplits);
  for split := 0 to NumSplitsM1 do
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
  for i := 0 to NumSplitsM1 do mean := mean + scores[i];
  mean := mean / NumSplits;
  varSum := 0;
  for i := 0 to NumSplitsM1 do varSum := varSum + Sqr(scores[i] - mean);
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

{ PSNR / SSIM / MS-SSIM }

const
  cSSIMWin   = 11;     // window side
  cSSIMSigma = 1.5;    // Gaussian sigma
  cSSIM_K1   = 0.01;
  cSSIM_K2   = 0.03;

// Build the normalised 1-D Gaussian window (sum = 1), length cSSIMWin.
//
// The 11x11 SSIM window is exp(-(x^2 + y^2) / s) / sum2d, which factors exactly:
//   exp(-(x^2 + y^2)/s) / sum2d = (exp(-y^2/s) / s1d) * (exp(-x^2/s) / s1d)
// because sum2d = s1d^2 for a separable Gaussian. So the normalisation folds
// into the 1-D taps once and the 2-D window never has to be materialised - both
// halves of every window pass run with this single vector.
procedure BuildGaussianWindow1D(out G: TIMDoubleArray);
var
  half, i, d, WinM1: integer;
  s, sum: Double;
begin
  SetLength(G, cSSIMWin);
  WinM1 := cSSIMWin - 1;
  half := cSSIMWin div 2;
  s := 2.0 * cSSIMSigma * cSSIMSigma;
  sum := 0;
  for i := 0 to WinM1 do
  begin
    d := i - half;
    G[i] := Exp(-(d * d) / s);
    sum := sum + G[i];
  end;
  for i := 0 to WinM1 do
    G[i] := G[i] / sum;
end;

var
  // The normalised 1-D Gaussian window, built once in this unit's initialization
  // section (#5/#8/§17) and read-only thereafter, so every SSIM caller
  // (SSIMPlaneMean / SSIMPlaneLCS / SSIMPlaneLossGrad) shares it without
  // re-running Exp + normalize per call and without a lazy-build race.
  GaussWin1D: TIMDoubleArray;

type
  // Reusable per-thread scratch for the separable window passes.
  //
  // THREAD SAFETY - the reason this is per-thread and not a unit-level array.
  // SSIMPlaneLossGrad is reached from TNNetSSIMLoss.Backpropagate through the
  // NeuralSSIMLossGradientHook, and TNeuralFit runs one network CLONE per worker
  // thread concurrently (neuralfit.pas RunNNThread, dispatched by
  // FProcs.StartProc), so several SSIM passes can be in flight at the same time.
  // Unlike GaussWin1D - which is built once and only ever read - these buffers
  // are mutable per-call state, so sharing them at unit level would be a data
  // race. Each thread lazily creates its own instance via ThreadSSIMScratch and
  // never touches another thread's; the instances are registered in
  // vSSIMScratchList purely so unit finalization can free them.
  //
  // The buffers are grow-only (rule #17): they are resized only when a larger
  // image arrives, so a training run allocates once and reuses forever.
  TSSIMScratch = class
  public
    // H x outW horizontal-pass accumulators, one per SSIM statistic. The
    // gradient scatter reuses the first three as its vertical-spread temps.
    HMx, HMy, HSxx, HSyy, HSxy: TIMDoubleArray;
    // outH x outW per-window gradient coefficients (constant / a / b terms).
    K0, K1, K2: TIMDoubleArray;
    procedure EnsureRows(n: integer);
    procedure EnsureWindows(n: integer);
  end;

procedure TSSIMScratch.EnsureRows(n: integer);
begin
  if Length(HMx) < n then
  begin
    SetLength(HMx, n);
    SetLength(HMy, n);
    SetLength(HSxx, n);
    SetLength(HSyy, n);
    SetLength(HSxy, n);
  end;
end;

procedure TSSIMScratch.EnsureWindows(n: integer);
begin
  if Length(K0) < n then
  begin
    SetLength(K0, n);
    SetLength(K1, n);
    SetLength(K2, n);
  end;
end;

threadvar
  // Pointer-sized (unmanaged) per-thread cache of this thread's scratch.
  vSSIMScratch: TSSIMScratch;

var
  // Owns every scratch ever created, so finalization frees them all. Guarded
  // because two threads can reach their first SSIM call simultaneously; the
  // lock is taken once per thread, never per call.
  vSSIMScratchList: TList;
  vSSIMScratchCS: TRTLCriticalSection;

// The calling thread's scratch, created on first use.
function ThreadSSIMScratch: TSSIMScratch;
begin
  Result := vSSIMScratch;
  if Result = nil then
  begin
    Result := TSSIMScratch.Create;
    vSSIMScratch := Result;
    EnterCriticalSection(vSSIMScratchCS);
    try
      vSSIMScratchList.Add(Result);
    finally
      LeaveCriticalSection(vSSIMScratchCS);
    end;
  end;
end;

// Horizontal half of the separable window pass. For every image row y and every
// valid window column ox, accumulates the cSSIMWin taps of the 1-D Gaussian into
// the five SSIM statistics at once. The products A*A, B*B and A*B are formed
// INLINE from the two source planes, so no product planes are materialised.
// Writes H * outW entries (row-major, stride outW) into the five scratch rows.
procedure SSIMHorizontalPass(const PA, PB, G: TIMDoubleArray;
  H, W, outW: integer;
  var HMx, HMy, HSxx, HSyy, HSxy: TIMDoubleArray);
var
  y, ox, wx, pix, rowB, idx, HM1, outWM1, WinM1: integer;
  mx, my, sxx, syy, sxy, gw, a, b, wa, wb: Double;
begin
  HM1 := H - 1;
  outWM1 := outW - 1;
  WinM1 := cSSIMWin - 1;
  idx := 0;                          // = y * outW + ox, carried (#6)
  for y := 0 to HM1 do
  begin
    rowB := y * W;                   // #11: row base, once per row
    for ox := 0 to outWM1 do
    begin
      mx := 0; my := 0; sxx := 0; syy := 0; sxy := 0;
      pix := rowB + ox;
      for wx := 0 to WinM1 do
      begin
        gw := G[wx];
        a := PA[pix]; b := PB[pix];
        wa := gw * a; wb := gw * b;                // #4: computed once
        mx := mx + wa; my := my + wb;
        sxx := sxx + wa * a; syy := syy + wb * b;  // (gw*a)*a, left-assoc
        sxy := sxy + wa * b;
        Inc(pix);
      end;
      HMx[idx] := mx; HMy[idx] := my;
      HSxx[idx] := sxx; HSyy[idx] := syy; HSxy[idx] := sxy;
      Inc(idx);
    end;
  end;
end;

// Extract one channel of a channel-last flat image into a dense H*W plane.
procedure ExtractChannel(const Img: TIMDoubleArray; H, W, Channels, C: integer;
  out Plane: TIMDoubleArray);
var
  y, x, HM1, WM1, dst, src: integer;
begin
  SetLength(Plane, H * W);
  HM1 := H - 1;
  WM1 := W - 1;
  for y := 0 to HM1 do
  begin
    dst := y * W;              // #11: row base, once per row
    src := dst * Channels + C; // channel-last source offset for x = 0
    for x := 0 to WM1 do
    begin
      Plane[dst + x] := Img[src];
      Inc(src, Channels);      // #6: strength-reduce the per-pixel multiply
    end;
  end;
end;

function ComputePSNR(const ImgA, ImgB: TIMDoubleArray;
  H, W, Channels: integer; DataRange: Double): Double;
var
  i, n, nM1: integer;
  mse, d: Double;
begin
  n := H * W * Channels;
  if (Length(ImgA) < n) or (Length(ImgB) < n) then
    raise Exception.Create('ComputePSNR: image array too small for H*W*Channels');
  nM1 := n - 1;
  mse := 0;
  for i := 0 to nM1 do
  begin
    d := ImgA[i] - ImgB[i];
    mse := mse + d * d;
  end;
  mse := mse / n;
  if mse <= 0 then
    Exit(Infinity);  // identical images
  Result := 10.0 * Log10((DataRange * DataRange) / mse);
end;

// Mean SSIM of a single H*W plane pair, 'valid' window map (returns the mean
// over the (H-win+1)*(W-win+1) fully-inside positions). Raises if too small.
function SSIMPlaneMean(const PA, PB: TIMDoubleArray; H, W: integer;
  C1, C2: Double): Double;
var
  G, HMx, HMy, HSxx, HSyy, HSxy: TIMDoubleArray;
  Sc: TSSIMScratch;
  oy, ox, wy, sIdx, rowB: integer;
  outH, outW, cnt, outHM1, outWM1, WinM1: integer;
  mx, my, sxx, syy, sxy, gw: Double;
  a1, a2, b1, b2, ssimSum: Double;
begin
  if (H < cSSIMWin) or (W < cSSIMWin) then
    raise Exception.CreateFmt(
      'SSIM: image %dx%d smaller than %dx%d window', [H, W, cSSIMWin, cSSIMWin]);
  G := GaussWin1D;               // #5: shared read-only 1-D window
  outH := H - cSSIMWin + 1;
  outW := W - cSSIMWin + 1;
  outHM1 := outH - 1;
  outWM1 := outW - 1;
  WinM1 := cSSIMWin - 1;
  // Separable window: the horizontal half runs once over the whole plane pair,
  // then each output window costs cSSIMWin taps instead of cSSIMWin^2.
  Sc := ThreadSSIMScratch;
  Sc.EnsureRows(H * outW);
  HMx := Sc.HMx; HMy := Sc.HMy;
  HSxx := Sc.HSxx; HSyy := Sc.HSyy; HSxy := Sc.HSxy;
  SSIMHorizontalPass(PA, PB, G, H, W, outW, HMx, HMy, HSxx, HSyy, HSxy);
  ssimSum := 0;
  cnt := outH * outW;
  for oy := 0 to outHM1 do
  begin
    rowB := oy * outW;             // #11: row base, once per oy
    for ox := 0 to outWM1 do
    begin
      mx := 0; my := 0; sxx := 0; syy := 0; sxy := 0;
      sIdx := rowB + ox;
      for wy := 0 to WinM1 do      // vertical half of the separable pass
      begin
        gw := G[wy];
        mx := mx + gw * HMx[sIdx];
        my := my + gw * HMy[sIdx];
        sxx := sxx + gw * HSxx[sIdx];
        syy := syy + gw * HSyy[sIdx];
        sxy := sxy + gw * HSxy[sIdx];
        Inc(sIdx, outW);           // #6: next image row of the same window
      end;
      sxx := sxx - mx * mx;
      syy := syy - my * my;
      sxy := sxy - mx * my;
      a1 := 2.0 * mx * my + C1;
      a2 := 2.0 * sxy + C2;
      b1 := mx * mx + my * my + C1;
      b2 := sxx + syy + C2;
      ssimSum := ssimSum + (a1 * a2) / (b1 * b2);
    end;
  end;
  Result := ssimSum / cnt;
end;

function ComputeSSIM(const ImgA, ImgB: TIMDoubleArray;
  H, W, Channels: integer; DataRange: Double): Double;
var
  c, ChannelsM1: integer;
  PA, PB: TIMDoubleArray;
  C1, C2, acc: Double;
begin
  if (Length(ImgA) < H * W * Channels) or (Length(ImgB) < H * W * Channels) then
    raise Exception.Create('ComputeSSIM: image array too small');
  C1 := Sqr(cSSIM_K1 * DataRange);
  C2 := Sqr(cSSIM_K2 * DataRange);
  ChannelsM1 := Channels - 1;
  acc := 0;
  for c := 0 to ChannelsM1 do
  begin
    ExtractChannel(ImgA, H, W, Channels, c, PA);
    ExtractChannel(ImgB, H, W, Channels, c, PB);
    acc := acc + SSIMPlaneMean(PA, PB, H, W, C1, C2);
  end;
  Result := acc / Channels;
end;

// 2x2 average-pool a single H*W plane (floor dims). Used between MS-SSIM scales.
procedure AvgPool2x2(const Plane: TIMDoubleArray; H, W: integer;
  out OutPlane: TIMDoubleArray; out OH, OW: integer);
var
  y, x, OHM1, OWM1, r0, r1, dst, c: integer;
begin
  OH := H div 2;
  OW := W div 2;
  OHM1 := OH - 1;
  OWM1 := OW - 1;
  SetLength(OutPlane, OH * OW);
  for y := 0 to OHM1 do
  begin
    r0 := (2 * y) * W;   // #11: row bases, once per y
    r1 := r0 + W;
    dst := y * OW;
    for x := 0 to OWM1 do
    begin
      c := 2 * x;        // #4: 2*x once
      OutPlane[dst] := 0.25 *
        (Plane[r0 + c] +
         Plane[r0 + c + 1] +
         Plane[r1 + c] +
         Plane[r1 + c + 1]);
      Inc(dst);          // #6
    end;
  end;
end;

// Per-scale luminance (l) and contrast-structure (cs) means over the valid
// window map for one plane pair.
procedure SSIMPlaneLCS(const PA, PB: TIMDoubleArray; H, W: integer;
  C1, C2: Double; out LMean, CSMean: Double);
var
  G, HMx, HMy, HSxx, HSyy, HSxy: TIMDoubleArray;
  Sc: TSSIMScratch;
  oy, ox, wy, sIdx, rowB: integer;
  outH, outW, cnt, outHM1, outWM1, WinM1: integer;
  mx, my, sxx, syy, sxy, gw: Double;
  lSum, csSum: Double;
begin
  if (H < cSSIMWin) or (W < cSSIMWin) then
    raise Exception.CreateFmt(
      'MS-SSIM: scale %dx%d smaller than %dx%d window', [H, W, cSSIMWin, cSSIMWin]);
  G := GaussWin1D;               // #5: shared read-only 1-D window
  outH := H - cSSIMWin + 1;
  outW := W - cSSIMWin + 1;
  outHM1 := outH - 1;
  outWM1 := outW - 1;
  WinM1 := cSSIMWin - 1;
  // Separable window - see SSIMPlaneMean.
  Sc := ThreadSSIMScratch;
  Sc.EnsureRows(H * outW);
  HMx := Sc.HMx; HMy := Sc.HMy;
  HSxx := Sc.HSxx; HSyy := Sc.HSyy; HSxy := Sc.HSxy;
  SSIMHorizontalPass(PA, PB, G, H, W, outW, HMx, HMy, HSxx, HSyy, HSxy);
  lSum := 0; csSum := 0;
  cnt := outH * outW;
  for oy := 0 to outHM1 do
  begin
    rowB := oy * outW;             // #11: row base, once per oy
    for ox := 0 to outWM1 do
    begin
      mx := 0; my := 0; sxx := 0; syy := 0; sxy := 0;
      sIdx := rowB + ox;
      for wy := 0 to WinM1 do      // vertical half of the separable pass
      begin
        gw := G[wy];
        mx := mx + gw * HMx[sIdx];
        my := my + gw * HMy[sIdx];
        sxx := sxx + gw * HSxx[sIdx];
        syy := syy + gw * HSyy[sIdx];
        sxy := sxy + gw * HSxy[sIdx];
        Inc(sIdx, outW);           // #6: next image row of the same window
      end;
      sxx := sxx - mx * mx;
      syy := syy - my * my;
      sxy := sxy - mx * my;
      lSum := lSum + (2.0 * mx * my + C1) / (mx * mx + my * my + C1);
      csSum := csSum + (2.0 * sxy + C2) / (sxx + syy + C2);
    end;
  end;
  LMean := lSum / cnt;
  CSMean := csSum / cnt;
end;

function MSSSIMPlane(const PA0, PB0: TIMDoubleArray; H0, W0: integer;
  C1, C2: Double): Double;
const
  Weights: array[0..4] of Double = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333);
var
  scale, H, W, OH, OW: integer;
  PA, PB, NA, NB: TIMDoubleArray;
  l, cs, prod: Double;
begin
  PA := Copy(PA0);
  PB := Copy(PB0);
  H := H0; W := W0;
  prod := 1.0;
  for scale := 0 to 4 do
  begin
    SSIMPlaneLCS(PA, PB, H, W, C1, C2, l, cs);
    if scale < 4 then
      prod := prod * Power(cs, Weights[scale])
    else
      prod := prod * Power(l * cs, Weights[scale]);
    if scale < 4 then
    begin
      AvgPool2x2(PA, H, W, NA, OH, OW);
      AvgPool2x2(PB, H, W, NB, OH, OW);
      PA := NA; PB := NB; H := OH; W := OW;
    end;
  end;
  Result := prod;
end;

function ComputeMSSSIM(const ImgA, ImgB: TIMDoubleArray;
  H, W, Channels: integer; DataRange: Double): Double;
var
  c, ChannelsM1: integer;
  PA, PB: TIMDoubleArray;
  C1, C2, acc: Double;
begin
  if (Length(ImgA) < H * W * Channels) or (Length(ImgB) < H * W * Channels) then
    raise Exception.Create('ComputeMSSSIM: image array too small');
  C1 := Sqr(cSSIM_K1 * DataRange);
  C2 := Sqr(cSSIM_K2 * DataRange);
  ChannelsM1 := Channels - 1;
  acc := 0;
  for c := 0 to ChannelsM1 do
  begin
    ExtractChannel(ImgA, H, W, Channels, c, PA);
    ExtractChannel(ImgB, H, W, Channels, c, PB);
    acc := acc + MSSSIMPlane(PA, PB, H, W, C1, C2);
  end;
  Result := acc / Channels;
end;

// 1 - SSIM loss + gradient w.r.t. PA (single channel). Accumulates the
// per-window SSIM gradient back to each contributing pixel of PA. The mean
// SSIM is (1/Nwin) sum_p S_p; for each window, with normalised weights w_i,
//   dS_p/dx_i = (dS_p/dmx) w_i + (dS_p/dsxx) 2 w_i (x_i - mx)
//             + (dS_p/dsxy) w_i (y_i - my)
// where S_p = (A1 A2)/(B1 B2), A1=2 mx my+C1, A2=2 sxy+C2,
// B1=mx^2+my^2+C1, B2=sxx+syy+C2, and
//   dS/dmx = [(2 my)A2 B1 B2 - A1 A2 (2 mx) B2] / (B1 B2)^2
//   dS/dsxy = (2 A1)/(B1 B2)
//   dS/dsxx = -(A1 A2)/(B1 B2^2)
// (note dmx, dsxx, dsxy each also pull mx through their definitions, already
// folded into the per-pixel chain above).
//
// BOTH halves are separable. Forward: mx, my and the raw sxx/syy/sxy moments are
// 2-D convolutions of A, B, A*A, B*B, A*B with the Gaussian, so they split into
// a horizontal and a vertical 1-D pass. Backward: expanding the per-pixel chain
//   dS_p/dx_i / w_i = (dS/dmx - 2 dS/dsxx mx - dS/dsxy my)
//                     + (2 dS/dsxx) a_i + (dS/dsxy) b_i
//                   = K0_p + K1_p a_i + K2_p b_i
// shows the pixel dependence is only through a_i and b_i, which are constant
// across the windows p that cover pixel i. So the scatter is
//   grad_i = -(1/N) [ (K0 * w)_i + a_i (K1 * w)_i + b_i (K2 * w)_i ]
// i.e. three transposed convolutions of the per-window coefficient planes K0,
// K1, K2 with the SAME separable Gaussian - a vertical spread followed by a
// horizontal one. Nothing here stays dense: cost drops from 121 taps x 5 (fwd)
// + 121 taps (bwd) per window to 11+11 taps x 5 and 11+11 taps x 3.
function SSIMPlaneLossGrad(const PA, PB: TIMDoubleArray; H, W: integer;
  C1, C2: Double; var GA: TIMDoubleArray; OffStride, Off: integer): Double;
var
  G, HMx, HMy, HSxx, HSyy, HSxy, K0, K1, K2: TIMDoubleArray;
  SK0, SK1, SK2: TIMDoubleArray;
  Sc: TSSIMScratch;
  oy, ox, x, y, wy, wx, sIdx, kIdx, vIdx, pix, gpix, rowB, rowV: integer;
  outH, outW, cnt, outHM1, outWM1, WinM1, HM1, WM1: integer;
  wLo, wHi: integer;
  mx, my, sxx, syy, sxy, gw, a, b: Double;
  a1, a2, b1, b2, bb, sp, ssimSum: Double;
  dSdmx, dSdsxx, dSdsxy, k1v, k2v: Double;
  acc0, acc1, acc2: Double;
begin
  G := GaussWin1D;               // #5: shared read-only 1-D window
  outH := H - cSSIMWin + 1;
  outW := W - cSSIMWin + 1;
  outHM1 := outH - 1;
  outWM1 := outW - 1;
  WinM1 := cSSIMWin - 1;
  HM1 := H - 1;
  WM1 := W - 1;
  cnt := outH * outW;
  Sc := ThreadSSIMScratch;
  Sc.EnsureRows(H * outW);
  Sc.EnsureWindows(cnt);
  HMx := Sc.HMx; HMy := Sc.HMy;
  HSxx := Sc.HSxx; HSyy := Sc.HSyy; HSxy := Sc.HSxy;
  K0 := Sc.K0; K1 := Sc.K1; K2 := Sc.K2;
  SSIMHorizontalPass(PA, PB, G, H, W, outW, HMx, HMy, HSxx, HSyy, HSxy);
  ssimSum := 0;
  for oy := 0 to outHM1 do
  begin
    rowB := oy * outW;             // #11: row base, once per oy
    for ox := 0 to outWM1 do
    begin
      mx := 0; my := 0; sxx := 0; syy := 0; sxy := 0;
      sIdx := rowB + ox;
      for wy := 0 to WinM1 do      // vertical half of the separable pass
      begin
        gw := G[wy];
        mx := mx + gw * HMx[sIdx];
        my := my + gw * HMy[sIdx];
        sxx := sxx + gw * HSxx[sIdx];
        syy := syy + gw * HSyy[sIdx];
        sxy := sxy + gw * HSxy[sIdx];
        Inc(sIdx, outW);           // #6: next image row of the same window
      end;
      sxx := sxx - mx * mx; syy := syy - my * my; sxy := sxy - mx * my;
      a1 := 2.0 * mx * my + C1;
      a2 := 2.0 * sxy + C2;
      b1 := mx * mx + my * my + C1;
      b2 := sxx + syy + C2;
      bb := b1 * b2;
      sp := (a1 * a2) / bb;
      ssimSum := ssimSum + sp;
      // partial derivatives of S_p w.r.t. the three moments (treating mx,sxx,
      // sxy as the independent aggregates; the mx coupling inside sxx/sxy is
      // handled by the per-pixel a and b terms below).
      dSdmx  := (2.0 * my * a2 * bb - a1 * a2 * (2.0 * mx) * b2) / (bb * bb);
      dSdsxy := (2.0 * a1) / bb;
      dSdsxx := -(a1 * a2) / (b1 * b2 * b2);
      // dmx/dx_i = w; dsxx/dx_i = 2 w (a - mx); dsxy/dx_i = w (b - my), so the
      // per-window scatter coefficient is K0 + K1*a_i + K2*b_i.
      k1v := 2.0 * dSdsxx;
      k2v := dSdsxy;
      kIdx := rowB + ox;
      K0[kIdx] := dSdmx - k1v * mx - k2v * my;
      K1[kIdx] := k1v;
      K2[kIdx] := k2v;
    end;
  end;
  // Transposed (scatter) convolution of K0/K1/K2 back onto the H*W pixel grid,
  // vertical half first. The horizontal-pass rows are dead by now, so they are
  // reused as the H x outW vertical-spread temporaries.
  SK0 := HMx; SK1 := HMy; SK2 := HSxx;
  for y := 0 to HM1 do
  begin
    // window rows wy with 0 <= y - wy <= outH-1 are the ones that reach row y
    wLo := y - outHM1;
    if wLo < 0 then wLo := 0;
    wHi := y;
    if wHi > WinM1 then wHi := WinM1;
    rowV := y * outW;              // #11: row base, once per y
    for ox := 0 to outWM1 do
    begin
      acc0 := 0; acc1 := 0; acc2 := 0;
      kIdx := (y - wLo) * outW + ox;
      for wy := wLo to wHi do
      begin
        gw := G[wy];
        acc0 := acc0 + gw * K0[kIdx];
        acc1 := acc1 + gw * K1[kIdx];
        acc2 := acc2 + gw * K2[kIdx];
        Dec(kIdx, outW);           // #6: previous window row
      end;
      vIdx := rowV + ox;
      SK0[vIdx] := acc0; SK1[vIdx] := acc1; SK2[vIdx] := acc2;
    end;
  end;
  // Horizontal half of the scatter, applied straight into GA.
  for y := 0 to HM1 do
  begin
    rowV := y * outW;              // #11: row bases, once per y
    rowB := y * W;
    gpix := rowB * OffStride + Off;  // #6: seed gpix for x = 0, then Inc
    for x := 0 to WM1 do
    begin
      wLo := x - outWM1;
      if wLo < 0 then wLo := 0;
      wHi := x;
      if wHi > WinM1 then wHi := WinM1;
      acc0 := 0; acc1 := 0; acc2 := 0;
      vIdx := rowV + x - wLo;
      for wx := wLo to wHi do
      begin
        gw := G[wx];
        acc0 := acc0 + gw * SK0[vIdx];
        acc1 := acc1 + gw * SK1[vIdx];
        acc2 := acc2 + gw * SK2[vIdx];
        Dec(vIdx);                 // #6: previous window column
      end;
      pix := rowB + x;
      a := PA[pix]; b := PB[pix];
      GA[gpix] := GA[gpix] - (acc0 + acc1 * a + acc2 * b) / cnt;  // d(1-SSIM) = -dSSIM
      Inc(gpix, OffStride);        // #6: advance by the channel stride
    end;
  end;
  Result := ssimSum / cnt;
end;

function ComputeSSIMLossAndGradient(const ImgA, ImgB: TIMDoubleArray;
  H, W, Channels: integer; out GradA: TIMDoubleArray;
  DataRange: Double): Double;
var
  c, i, n, nM1, ChannelsM1: integer;
  PA, PB: TIMDoubleArray;
  C1, C2, ssimAcc: Double;
begin
  n := H * W * Channels;
  if (Length(ImgA) < n) or (Length(ImgB) < n) then
    raise Exception.Create('ComputeSSIMLossAndGradient: image array too small');
  if (H < cSSIMWin) or (W < cSSIMWin) then
    raise Exception.CreateFmt(
      'ComputeSSIMLossAndGradient: image %dx%d smaller than window', [H, W]);
  C1 := Sqr(cSSIM_K1 * DataRange);
  C2 := Sqr(cSSIM_K2 * DataRange);
  nM1 := n - 1;
  ChannelsM1 := Channels - 1;
  SetLength(GradA, n);
  for i := 0 to nM1 do GradA[i] := 0;
  ssimAcc := 0;
  for c := 0 to ChannelsM1 do
  begin
    ExtractChannel(ImgA, H, W, Channels, c, PA);
    ExtractChannel(ImgB, H, W, Channels, c, PB);
    // SSIM averaged over channels, so each channel's loss & grad scale by
    // 1/Channels.
    ssimAcc := ssimAcc +
      SSIMPlaneLossGrad(PA, PB, H, W, C1, C2, GradA, Channels, c);
  end;
  // average over channels
  for i := 0 to nM1 do GradA[i] := GradA[i] / Channels;
  Result := 1.0 - ssimAcc / Channels;
end;

{ KID }

// Cubic polynomial kernel k(x,y) = (x.y/d + 1)^3.
function PolyKernel(const X, Y: TIMDoubleArray; d: integer): Double;
var
  i, dM1: integer;
  dot, t: Double;
begin
  dot := 0;
  dM1 := d - 1;
  for i := 0 to dM1 do dot := dot + X[i] * Y[i];
  t := dot / d + 1.0;
  Result := t * t * t;
end;

// Unbiased within-set term sum_{i<>j} k(x_i,x_j) / (m(m-1)).
function UnbiasedSelfTerm(const F: TIMDoubleMatrix; d: integer): Double;
var
  m, mM1, i, j: integer;
  s: Double;
  Fi: TIMDoubleArray;
begin
  m := Length(F);
  if m < 2 then
    raise Exception.Create('KID: each set needs at least 2 samples');
  mM1 := m - 1;
  s := 0;
  // k(x,y) is symmetric, so only the upper triangle is evaluated and doubled.
  for i := 0 to mM1 do
  begin
    Fi := F[i];
    for j := i + 1 to mM1 do
      s := s + PolyKernel(Fi, F[j], d);
  end;
  s := s * 2.0;
  Result := s / (m * (m - 1));
end;

function ComputeKIDMMD2(const FeaturesR, FeaturesG: TIMDoubleMatrix): Double;
var
  m, n, d, i, j, mM1, nM1: integer;
  termR, termG, cross: Double;
begin
  if (Length(FeaturesR) = 0) or (Length(FeaturesG) = 0) then
    raise Exception.Create('ComputeKIDMMD2: empty feature set');
  d := Length(FeaturesR[0]);
  if Length(FeaturesG[0]) <> d then
    raise Exception.Create('ComputeKIDMMD2: feature dimension mismatch');
  m := Length(FeaturesR);
  n := Length(FeaturesG);
  mM1 := m - 1;
  nM1 := n - 1;
  termR := UnbiasedSelfTerm(FeaturesR, d);
  termG := UnbiasedSelfTerm(FeaturesG, d);
  cross := 0;
  for i := 0 to mM1 do
    for j := 0 to nM1 do
      cross := cross + PolyKernel(FeaturesR[i], FeaturesG[j], d);
  cross := cross / (m * n);
  Result := termR + termG - 2.0 * cross;
end;

// Draw, without replacement, SubsetSize row indices from [0..Count-1].
procedure SampleIndices(Count, SubsetSize: integer; out Idx: array of integer);
var
  pool: array of integer;
  i, j, tmp, CountM1, SubsetM1: integer;
begin
  SetLength(pool, Count);
  CountM1 := Count - 1;
  SubsetM1 := SubsetSize - 1;
  for i := 0 to CountM1 do pool[i] := i;
  // partial Fisher-Yates
  for i := 0 to SubsetM1 do
  begin
    j := i + Random(Count - i);
    tmp := pool[i]; pool[i] := pool[j]; pool[j] := tmp;
    Idx[i] := pool[i];
  end;
end;

procedure ComputeKID(const FeaturesR, FeaturesG: TIMDoubleMatrix;
  SubsetSize, NumSubsets: integer; out Score: Double; out StdDev: Double);
var
  m, n, sub, s, i, subM1, NumSubsetsM1: integer;
  idxR, idxG: array of integer;
  subR, subG: TIMDoubleMatrix;
  scores: TIMDoubleArray;
  mean, varSum: Double;
begin
  if (Length(FeaturesR) = 0) or (Length(FeaturesG) = 0) then
    raise Exception.Create('ComputeKID: empty feature set');
  m := Length(FeaturesR);
  n := Length(FeaturesG);
  if NumSubsets < 1 then NumSubsets := 1;
  if SubsetSize < 2 then SubsetSize := 2;
  sub := SubsetSize;
  if sub > m then sub := m;
  if sub > n then sub := n;
  NumSubsetsM1 := NumSubsets - 1;
  subM1 := sub - 1;

  SetLength(scores, NumSubsets);
  SetLength(idxR, sub);
  SetLength(idxG, sub);
  SetLength(subR, sub);
  SetLength(subG, sub);
  for s := 0 to NumSubsetsM1 do
  begin
    SampleIndices(m, sub, idxR);
    SampleIndices(n, sub, idxG);
    for i := 0 to subM1 do
    begin
      subR[i] := FeaturesR[idxR[i]];
      subG[i] := FeaturesG[idxG[i]];
    end;
    scores[s] := ComputeKIDMMD2(subR, subG);
  end;

  mean := 0;
  for i := 0 to NumSubsetsM1 do mean := mean + scores[i];
  mean := mean / NumSubsets;
  varSum := 0;
  for i := 0 to NumSubsetsM1 do varSum := varSum + Sqr(scores[i] - mean);
  Score := mean;
  if NumSubsets > 1 then
    StdDev := Sqrt(varSum / NumSubsets)  // population std (mirrors IS report)
  else
    StdDev := 0;
end;

{ ImageNet top-1 / top-5 accuracy harness }

procedure TopKIndices(const Scores: array of TNeuralFloat; Count, K: integer;
  out Pred: array of integer);
var
  used: array of boolean;
  rank, i, best, CountM1, KM1: integer;
  bestVal: TNeuralFloat;
begin
  if Count > Length(Scores) then Count := Length(Scores);
  if K > Count then K := Count;
  if K < 0 then K := 0;
  if K = 0 then Exit;
  SetLength(used, Count);
  CountM1 := Count - 1;
  for i := 0 to CountM1 do used[i] := False;
  KM1 := K - 1;
  for rank := 0 to KM1 do
  begin
    best := -1;
    bestVal := 0;
    // First-max tie-break: scan ascending, take a STRICTLY greater value, so
    // the lowest index wins an exact tie (matches argmax / GetClass).
    for i := 0 to CountM1 do
      if not used[i] then
        if (best < 0) or (Scores[i] > bestVal) then
        begin
          bestVal := Scores[i];
          best := i;
        end;
    used[best] := True;
    Pred[rank] := best;
  end;
end;

function EvaluateImageNet(NN: TNNet;
  const Samples: array of TNNetImageNetSample;
  NumClasses: integer; K: integer;
  MaxConfusion: integer): TNNetImageNetStats;
var
  SamplesHi, SIdx, rank, KM1, ConfCnt: integer;
  NumClassesM1: integer;
  Gold, Top1: integer;
  OutVol: TNNetVolume;
  Pred: array of integer;
  Scores: array of TNeuralFloat;
  HitTopK: boolean;
begin
  Result.ItemCount := 0;
  Result.Top1Correct := 0;
  Result.TopKCorrect := 0;
  Result.Top1Accuracy := 0;
  Result.TopKAccuracy := 0;
  Result.NumClasses := NumClasses;
  SetLength(Result.Confusion, 0);
  if NumClasses <= 0 then
    raise Exception.Create('EvaluateImageNet: NumClasses must be > 0');
  if K < 1 then K := 1;
  if K > NumClasses then K := NumClasses;
  Result.K := K;
  if MaxConfusion < 0 then MaxConfusion := 0;
  if NN = nil then Exit;

  KM1 := K - 1;
  NumClassesM1 := NumClasses - 1;
  SetLength(Pred, K);
  SetLength(Scores, NumClasses);
  ConfCnt := 0;
  SamplesHi := High(Samples);
  for SIdx := 0 to SamplesHi do
  begin
    Gold := Samples[SIdx].GoldLabel;
    if (Gold < 0) or (Gold >= NumClasses) then continue;
    if Samples[SIdx].Image = nil then continue;
    NN.Compute(Samples[SIdx].Image);
    OutVol := NN.GetLastLayer().Output;
    if OutVol.Size < NumClasses then
      raise Exception.CreateFmt(
        'EvaluateImageNet: net output size %d < NumClasses %d',
        [OutVol.Size, NumClasses]);
    // Copy the first NumClasses scores (logits or probs - argmax is identical).
    for rank := 0 to NumClassesM1 do
      Scores[rank] := OutVol.FData[rank];
    TopKIndices(Scores, NumClasses, K, Pred);

    Top1 := Pred[0];
    HitTopK := False;
    for rank := 0 to KM1 do
      if Pred[rank] = Gold then
      begin
        HitTopK := True;
        Break;
      end;

    Inc(Result.ItemCount);
    if Top1 = Gold then Inc(Result.Top1Correct);
    if HitTopK then Inc(Result.TopKCorrect);

    // Retain top-1 misses for the confusion sample.
    if (Top1 <> Gold) and (ConfCnt < MaxConfusion) then
    begin
      SetLength(Result.Confusion, ConfCnt + 1);
      Result.Confusion[ConfCnt].SourceName := Samples[SIdx].SourceName;
      Result.Confusion[ConfCnt].GoldLabel := Gold;
      Result.Confusion[ConfCnt].PredLabel := Top1;
      Result.Confusion[ConfCnt].InTopK := HitTopK;
      Inc(ConfCnt);
    end;
  end;

  if Result.ItemCount > 0 then
  begin
    Result.Top1Accuracy := Result.Top1Correct / Result.ItemCount;
    Result.TopKAccuracy := Result.TopKCorrect / Result.ItemCount;
  end;
end;

function ImageNetReport(const Stats: TNNetImageNetStats;
  const ClassNames: array of string; const Title: string): string;
var
  SL: TStringList;
  i, ConfHi: integer;

  function LabelName(Idx: integer): string;
  begin
    if (Idx >= 0) and (Idx <= High(ClassNames)) then
      Result := Format('%d (%s)', [Idx, ClassNames[Idx]])
    else
      Result := IntToStr(Idx);
  end;

begin
  SL := TStringList.Create();
  try
    SL.Add(Format('%s top-1 / top-%d accuracy', [Title, Stats.K]));
    SL.Add(Format('  images scored : %d', [Stats.ItemCount]));
    SL.Add(Format('  classes       : %d', [Stats.NumClasses]));
    SL.Add(Format('  top-1         : %.4f  (%d / %d)',
      [Stats.Top1Accuracy, Stats.Top1Correct, Stats.ItemCount]));
    SL.Add(Format('  top-%-9d : %.4f  (%d / %d)',
      [Stats.K, Stats.TopKAccuracy, Stats.TopKCorrect, Stats.ItemCount]));
    ConfHi := High(Stats.Confusion);
    if ConfHi >= 0 then
    begin
      SL.Add(Format('  confusion sample (%d top-1 miss(es)):', [ConfHi + 1]));
      for i := 0 to ConfHi do
        SL.Add(Format('    %-24s gold %-20s pred %-20s %s',
          [Stats.Confusion[i].SourceName,
           LabelName(Stats.Confusion[i].GoldLabel),
           LabelName(Stats.Confusion[i].PredLabel),
           BoolToStr(Stats.Confusion[i].InTopK, 'top-K hit', 'top-K miss')]));
    end;
    Result := SL.Text;
  finally
    SL.Free;
  end;
end;

// Adapter wiring ComputeSSIMLossAndGradient (which takes TIMDoubleArray) into
// the open-array NeuralSSIMLossGradientHook that neuralnetwork.TNNetSSIMLoss
// calls. The open-array params are copied into local TIMDoubleArray buffers and
// the computed gradient copied back out. Assigned at unit init below so any
// program that 'uses neuralimagemetrics' enables the SSIM loss head.
// Coded by Claude (AI).
function SSIMLossGradientHookAdapter(const ImgA, ImgB: array of Double;
  H, W, Channels: integer; out GradA: array of Double; DataRange: Double): Double;
var
  A, B, G: TIMDoubleArray;
  i, n, nM1: integer;
begin
  n := Length(ImgA);
  SetLength(A, n);
  SetLength(B, n);
  nM1 := n - 1;
  for i := 0 to nM1 do
  begin
    A[i] := ImgA[i];
    B[i] := ImgB[i];
  end;
  Result := ComputeSSIMLossAndGradient(A, B, H, W, Channels, G, DataRange);
  nM1 := Length(G) - 1;
  for i := 0 to nM1 do
    GradA[i] := G[i];
end;

initialization
  NeuralSSIMLossGradientHook := @SSIMLossGradientHookAdapter;
  // Built eagerly (not lazily) so concurrent first calls cannot race on the
  // build; read-only from here on.
  BuildGaussianWindow1D(GaussWin1D);
  vSSIMScratchList := TList.Create;
  InitCriticalSection(vSSIMScratchCS);

finalization
  while vSSIMScratchList.Count > 0 do
  begin
    TSSIMScratch(vSSIMScratchList[vSSIMScratchList.Count - 1]).Free;
    vSSIMScratchList.Delete(vSSIMScratchList.Count - 1);
  end;
  vSSIMScratchList.Free;
  DoneCriticalSection(vSSIMScratchCS);

end.
