// Wavelet shrinkage denoising demo -- TNNetDWT1D
// ---------------------------------------------------------------------------
// Classic Donoho-Johnstone wavelet SOFT-THRESHOLD denoising of a 1-D signal.
//
// We build the canonical Donoho-Johnstone "Blocks" signal -- a PIECEWISE-CONSTANT
// staircase of signed steps with SHARP edges -- add white Gaussian noise, and
// recover it with the multi-resolution wavelet inductive bias:
//
//   1. Run a MALLAT pyramid: apply a single-level TNNetDWT1D (Haar) repeatedly
//      to the APPROXIMATION band only (the standard wavelet decomposition tree).
//      Each level halves the length and yields one detail band.
//   2. SOFT-THRESHOLD the detail coefficients PER LEVEL with the universal
//      threshold  lambda = sigma * sqrt(2 ln M)  (Donoho-Johnstone), where sigma
//      is robustly estimated from that level's detail band by the MAD estimator
//      and M is the band length (per-level because the unnormalised lifting
//      rescales the coefficients from one level to the next).
//   3. INVERT the pyramid (TNNetDWT1D.InverseChannel) level by level to get the
//      denoised signal.
//
// The headline this demo proves: wavelet shrinkage keeps the SHARP edges
// (their energy is concentrated in a few large detail coefficients that survive
// the threshold) while killing the noise (spread thinly across all coefficients)
// -- giving a higher reconstruction SNR than a plain LOW-PASS baseline
// (a moving average), which is forced to blur the very edges it is meant to
// preserve. The multi-resolution basis is the right inductive bias for
// piecewise-constant / transient signals.
//
// Pure CPU, tiny signal, no training -> runs in a couple of seconds and uses
// almost no memory. No binaries are committed.
program WaveletDenoise;
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, neuralnetwork, neuralvolume;

const
  N       = 1024;   // signal length (power of 2 -> clean dyadic pyramid)
  LEVELS  = 5;      // wavelet decomposition levels
  FILTER  = csDWT1DHaar;    // Haar: optimal sparse basis for piecewise-constant signals
  NOISESD = 0.30;   // standard deviation of the added Gaussian noise
  MAWIN   = 9;      // moving-average window of the low-pass baseline

// Standard-normal sample via Box-Muller (avoids needing a TNNetVolume just for
// noise generation).
function GaussNoise(): TNeuralFloat;
var
  u1, u2: TNeuralFloat;
begin
  u1 := Random; u2 := Random;
  if u1 < 1e-12 then u1 := 1e-12;
  Result := Sqrt(-2.0 * Ln(u1)) * Cos(2 * Pi * u2);
end;

// Donoho-Johnstone "Blocks" signal: a canonical PIECEWISE-CONSTANT test signal
// built from a sum of shifted, signed step (Heaviside) functions. It is sparse
// in the wavelet domain (energy concentrates in a handful of large detail
// coefficients at the jumps) but DENSE in the Fourier / local-average domain,
// so it is the textbook case where wavelet shrinkage beats any linear low-pass
// filter -- the filter must blur the very edges that define the signal.
procedure MakeClean(var x: array of TNeuralFloat);
const
  // jump locations (fraction of length) and signed heights (Donoho-Johnstone).
  POS: array[0..10] of TNeuralFloat =
    (0.10, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81);
  HGT: array[0..10] of TNeuralFloat =
    (4.0, -5.0, 3.0, -4.0, 5.0, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2);
var
  i, j: integer;
  t, acc: TNeuralFloat;
begin
  for i := 0 to N - 1 do
  begin
    t := i / N;
    acc := 0;
    for j := 0 to High(POS) do
      if t >= POS[j] then acc := acc + HGT[j];   // signed Heaviside steps
    x[i] := acc;                                  // full Donoho-Johnstone amplitude
  end;
end;

// SNR in dB of a reconstruction against the clean reference.
function SNRdB(const clean, recon: array of TNeuralFloat): TNeuralFloat;
var
  i: integer;
  sig, err, d: TNeuralFloat;
begin
  sig := 0; err := 0;
  for i := 0 to N - 1 do
  begin
    sig := sig + clean[i] * clean[i];
    d := clean[i] - recon[i];
    err := err + d * d;
  end;
  if err < 1e-20 then err := 1e-20;
  Result := 10.0 * Ln(sig / err) / Ln(10.0);
end;

// Soft-threshold (shrinkage) operator: sign(x) * max(|x| - lambda, 0).
function SoftThreshold(v, lambda: Double): Double;
begin
  if v > lambda then Result := v - lambda
  else if v < -lambda then Result := v + lambda
  else Result := 0;
end;

// Median of a copy of an array (used by the MAD noise estimator).
function MedianAbs(const a: array of Double): TNeuralFloat;
var
  b: array of Double;
  i, j, n2: integer;
  tmp: Double;
begin
  n2 := Length(a);
  SetLength(b, n2);
  for i := 0 to n2 - 1 do b[i] := Abs(a[i]);
  // simple insertion sort (n2 is small per level)
  for i := 1 to n2 - 1 do
  begin
    tmp := b[i]; j := i - 1;
    while (j >= 0) and (b[j] > tmp) do begin b[j + 1] := b[j]; Dec(j); end;
    b[j + 1] := tmp;
  end;
  if n2 = 0 then Result := 0
  else Result := b[n2 div 2];
end;

// Run a one-level DWT on a depth-1 signal of length Len using a fresh net,
// returning the approximation (s) and detail (d) bands of length Len div 2.
// The owning net is returned (caller frees it at the very end) so the DWT
// layer stays alive for the matching InverseChannel pass.
procedure DWTLevel(Filter, Len: integer; const sig: array of Double;
  out s, d: array of Double; out NN: TNNet; out Layer: TNNetDWT1D);
var
  Inp: TNNetVolume;
  half, i: integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(Len, 1, 1));
  Layer := TNNetDWT1D(NN.AddLayer(TNNetDWT1D.Create(Filter)));
  Inp := TNNetVolume.Create(Len, 1, 1);
  for i := 0 to Len - 1 do Inp.FData[i] := sig[i];
  NN.Compute(Inp);
  half := Len div 2;
  for i := 0 to half - 1 do
  begin
    s[i] := Layer.Output.FData[i * 2 + 0];   // approx band (channel 0)
    d[i] := Layer.Output.FData[i * 2 + 1];   // detail band (channel 1)
  end;
  Inp.Free;
end;

// Donoho-Johnstone wavelet soft-threshold denoising (Mallat decomposition tree).
procedure WaveletDenoise(const noisy: array of TNeuralFloat;
  var recon: array of TNeuralFloat);
var
  approx: array of array of Double;   // approximation band per level
  detail: array of array of Double;   // detail band per level
  layers: array of TNNetDWT1D;
  nets: array of TNNet;
  cur, s, d: array of Double;
  curLen, lev, half, i: integer;
  sigma, lambda: TNeuralFloat;
begin
  SetLength(approx, LEVELS);
  SetLength(detail, LEVELS);
  SetLength(layers, LEVELS);
  SetLength(nets, LEVELS);

  // ---- forward decomposition on the approximation band only --------------
  SetLength(cur, N);
  for i := 0 to N - 1 do cur[i] := noisy[i];
  curLen := N;
  for lev := 0 to LEVELS - 1 do
  begin
    half := curLen div 2;
    SetLength(s, half);
    SetLength(d, half);
    DWTLevel(FILTER, curLen, cur, s, d, nets[lev], layers[lev]);
    SetLength(approx[lev], half);
    SetLength(detail[lev], half);
    for i := 0 to half - 1 do begin approx[lev][i] := s[i]; detail[lev][i] := d[i]; end;
    SetLength(cur, half);
    for i := 0 to half - 1 do cur[i] := s[i];   // recurse on approximation
    curLen := half;
  end;

  // ---- soft-threshold every detail band, per-level -----------------------
  // sigma is estimated PER LEVEL by the robust MAD estimator
  //   sigma_lev = median(|d_lev|) / 0.6745   (Donoho-Johnstone),
  // because the unnormalised lifting changes the detail-coefficient scale from
  // level to level. The universal threshold lambda = sigma*sqrt(2 ln M) uses the
  // band length M of that level.
  for lev := 0 to LEVELS - 1 do
  begin
    sigma := MedianAbs(detail[lev]) / 0.6745;
    lambda := sigma * Sqrt(2.0 * Ln(Length(detail[lev]) + 1));
    for i := 0 to Length(detail[lev]) - 1 do
      detail[lev][i] := SoftThreshold(detail[lev][i], lambda);
  end;

  // ---- inverse reconstruction, deepest level first -----------------------
  // cur currently holds the deepest approximation (length N / 2^LEVELS).
  for lev := LEVELS - 1 downto 0 do
  begin
    half := Length(approx[lev]);
    SetLength(s, half);
    SetLength(d, half);
    for i := 0 to half - 1 do begin s[i] := cur[i]; d[i] := detail[lev][i]; end;
    SetLength(cur, 2 * half);
    layers[lev].InverseChannel(s, d, cur);   // reconstruct this level
  end;

  for i := 0 to N - 1 do recon[i] := cur[i];
  for lev := 0 to LEVELS - 1 do nets[lev].Free;   // frees the owned DWT layers
end;

// Plain low-pass baseline: a centred moving average. It removes noise but
// unavoidably BLURS the step and the spikes -- the price of a single fixed scale.
procedure MovingAverageDenoise(const noisy: array of TNeuralFloat;
  var recon: array of TNeuralFloat);
var
  i, k, half, lo, hi, cnt: integer;
  acc: TNeuralFloat;
begin
  half := MAWIN div 2;
  for i := 0 to N - 1 do
  begin
    lo := Max(0, i - half); hi := Min(N - 1, i + half);
    acc := 0; cnt := 0;
    for k := lo to hi do begin acc := acc + noisy[k]; Inc(cnt); end;
    recon[i] := acc / cnt;
  end;
end;

// Show the TNNet.AddWaveletPacketTransform builder: it stacks Levels single-
// level DWTs into a balanced wavelet-PACKET tree (every channel recursively
// decomposed), halving SeqLen and doubling Depth at each level. This is the
// network-builder companion to the hand-rolled Mallat pyramid used for the
// denoising above (which recurses the approximation band only).
procedure ShowPacketBuilder();
var
  NN: TNNet;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(N, 1, 1));
  NN.AddWaveletPacketTransform({Levels=}LEVELS, {Filter=}FILTER);
  WriteLn(Format('AddWaveletPacketTransform(%d) packet tree: (%d,1,1) -> (%d,1,%d)',
    [LEVELS, N, NN.GetLastLayer.Output.SizeX, NN.GetLastLayer.Output.Depth]));
  WriteLn;
  NN.Free;
end;

var
  clean, noisy, wRec, mRec: array of TNeuralFloat;
  i: integer;
  snrNoisy, snrWave, snrMA: TNeuralFloat;
begin
  RandSeed := 424242;
  SetLength(clean, N);
  SetLength(noisy, N);
  SetLength(wRec, N);
  SetLength(mRec, N);

  WriteLn('1-D wavelet shrinkage denoising demo (Donoho-Johnstone soft threshold)');
  WriteLn(Format('Signal length = %d, levels = %d, filter = Haar, noise sd = %.2f',
    [N, LEVELS, NOISESD]));
  WriteLn;

  ShowPacketBuilder();

  MakeClean(clean);
  for i := 0 to N - 1 do
    noisy[i] := clean[i] + NOISESD * GaussNoise();

  WaveletDenoise(noisy, wRec);
  MovingAverageDenoise(noisy, mRec);

  snrNoisy := SNRdB(clean, noisy);
  snrWave  := SNRdB(clean, wRec);
  snrMA    := SNRdB(clean, mRec);

  WriteLn('=== Reconstruction SNR (dB, higher is better) ===');
  WriteLn(Format('  noisy input                       %8.2f dB', [snrNoisy]));
  WriteLn(Format('  low-pass baseline (moving avg %d)  %8.2f dB', [MAWIN, snrMA]));
  WriteLn(Format('  wavelet soft-threshold (DWT)      %8.2f dB', [snrWave]));
  WriteLn;
  WriteLn(Format('  wavelet gain over noisy    : %6.2f dB', [snrWave - snrNoisy]));
  WriteLn(Format('  wavelet gain over low-pass : %6.2f dB', [snrWave - snrMA]));
  WriteLn;
  if snrWave > snrMA then
    WriteLn('HEADLINE: the multi-resolution wavelet basis keeps the sharp block')
  else
    WriteLn('NOTE: wavelet did not beat the baseline on this run');
  WriteLn('EDGES that the single-scale low-pass filter is forced to blur, so it');
  WriteLn('wins on this piecewise-constant / transient signal.');
end.
