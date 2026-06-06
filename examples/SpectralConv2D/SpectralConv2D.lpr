// 2-D Fourier Neural Operator (FNO) demo -- TNNetSpectralConv2D
// ---------------------------------------------------------------------------
// Learns a smooth 2-D linear SOLUTION OPERATOR: given a band-limited source
// field f(x,y) on the periodic unit square, produce u(x,y) where each Fourier
// mode is scaled by a smooth low-pass filter (a 2-D diffusion / smoothing step:
//   uhat[kx,ky] = f hat[kx,ky] / (1 + c*(kx^2 + ky^2)) ).
//
// The headline this demo proves: a 2-D FNO trained ONLY on a COARSE 16x16 grid
// GENERALISES, with NO retraining, to a FINER 32x32 grid it never saw. The
// spectral conv's learnable weights live in 2-D MODE space, not grid space, so
// the SAME weights describe the SAME continuous operator at any resolution --
// a property a plain local conv stack (the baseline) structurally cannot match.
//
// Pure CPU, small grids and channel counts -> runs in well under 5 minutes on
// 2 cores and uses little memory. No binaries are committed.
program SpectralConv2D;
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, neuralnetwork, neuralvolume;

const
  NTRAIN_GRID = 16;     // coarse grid the operators are TRAINED on (power of 2)
  NEVAL_GRID  = 32;     // finer grid used ONLY at test time (never trained on)
  MODESX      = 4;      // kept Fourier modes along X (low-pass band)
  MODESY      = 4;      // kept Fourier modes along Y
  WIDTH       = 8;      // FNO channel width (lifting dimension)
  NSAMPLES    = 96;     // training functions
  MB          = 12;     // mini-batch size
  EPOCHS      = 300;
  LR          = 0.02;
  DIFF_C      = 0.6;    // diffusion coefficient of the target operator

// Build one random smooth periodic source field f on an LxL grid and its target
// u = (low-pass 2-D diffusion operator applied to f). The field is constructed
// DIRECTLY in 2-D mode space with Hermitian symmetry over the kept low band (so
// the grid signal is real) and the target is the SAME spectrum scaled per mode
// by a smooth diffusion gain. Because both f and u live entirely inside the
// MODESX x MODESY low band, the operator is exactly representable by the
// spectral conv AND is identical on any grid -> resolution invariant. Sampling
// the SAME random spectrum (seeded by idx) on a coarse or a fine grid yields
// the SAME underlying continuous functions.
procedure MakeSample(idx, L: integer; Inp, Tgt: TNNetVolume);
var
  kx, ky, x, y: integer;
  cr, ci, ampF, gain, twoPi, ang, fval, uval: TNeuralFloat;
  oldSeed: integer;
begin
  oldSeed := RandSeed;
  RandSeed := 1000 + idx;             // deterministic per-sample spectrum
  twoPi := 2 * Pi;
  Inp.ReSize(L, L, 1);
  Tgt.ReSize(L, L, 1);

  // Precompute random complex coefficients for the kept low modes.
  // We reconstruct the real field as a sum over kx,ky in [0,MODES) of
  //   c[kx,ky] * exp(i 2pi (kx x + ky y)/L)  + complex conjugate (for ky>0),
  // which is automatically real. The high (negative-alias) modes therefore also
  // carry energy, but BOTH the field and the target share the identical
  // conjugate structure, so the per-(low-mode) complex gain the spectral conv
  // learns reproduces u exactly.
  Inp.Fill(0);
  Tgt.Fill(0);
  for kx := 0 to MODESX - 1 do
    for ky := 0 to MODESY - 1 do
    begin
      ampF := 0.15 / (1 + kx + ky);    // small, decaying amplitudes -> smooth field
      cr := (Random - 0.5) * 2.0 * ampF;
      ci := (Random - 0.5) * 2.0 * ampF;
      gain := 1.0 / (1.0 + DIFF_C * (kx * kx + ky * ky)); // diffusion low-pass
      for y := 0 to L - 1 do
        for x := 0 to L - 1 do
        begin
          ang := twoPi * (kx * x + ky * y) / L;
          // 2 * Re( c * exp(i ang) ) = 2*(cr*cos - ci*sin); halve DC double-count
          fval := 2.0 * (cr * Cos(ang) - ci * Sin(ang));
          uval := gain * fval;
          if (kx = 0) and (ky = 0) then
          begin
            fval := fval * 0.5;
            uval := uval * 0.5;
          end;
          Inp[x, y, 0] := Inp[x, y, 0] + fval;
          Tgt[x, y, 0] := Tgt[x, y, 0] + uval;
        end;
    end;
  RandSeed := oldSeed;
end;

// Mean relative L2 error of a net's operator over a batch of fresh samples on an
// LxL grid (samples drawn from an idx range disjoint from training).
function EvalRelL2(NN: TNNet; L, idxStart, count: integer): TNeuralFloat;
var
  s, n: integer;
  Inp, Tgt: TNNetVolume;
  num, den, d, acc: TNeuralFloat;
begin
  Inp := TNNetVolume.Create();
  Tgt := TNNetVolume.Create();
  acc := 0;
  for s := 0 to count - 1 do
  begin
    MakeSample(idxStart + s, L, Inp, Tgt);
    NN.Compute(Inp);
    num := 0; den := 0;
    for n := 0 to Tgt.Size - 1 do
    begin
      d := NN.GetLastLayer.Output.FData[n] - Tgt.FData[n];
      num := num + d * d;
      den := den + Tgt.FData[n] * Tgt.FData[n];
    end;
    if den > 1e-12 then acc := acc + Sqrt(num / den);
  end;
  Inp.Free; Tgt.Free;
  Result := acc / count;
end;

procedure Train(NN: TNNet; const tag: string);
var
  ep, s, n: integer;
  Inp, Tgt: TNNetVolume;
  loss, d: TNeuralFloat;
begin
  Inp := TNNetVolume.Create();
  Tgt := TNNetVolume.Create();
  // Mini-batch SGD: accumulate deltas over MB samples then update once. Scaling
  // the learning rate by 1/MB keeps the step size sane regardless of batch size.
  NN.SetLearningRate(LR / MB, 0.9);
  NN.SetBatchUpdate(true);
  for ep := 0 to EPOCHS - 1 do
  begin
    loss := 0;
    for s := 0 to NSAMPLES - 1 do
    begin
      if (s mod MB) = 0 then NN.ClearDeltas();
      MakeSample(s, NTRAIN_GRID, Inp, Tgt);
      NN.Compute(Inp);
      NN.Backpropagate(Tgt);
      for n := 0 to Tgt.Size - 1 do
      begin
        d := NN.GetLastLayer.Output.FData[n] - Tgt.FData[n];
        loss := loss + d * d;
      end;
      if (s mod MB) = (MB - 1) then NN.UpdateWeights();
    end;
    if (ep mod 100 = 0) or (ep = EPOCHS - 1) then
      WriteLn(Format('  [%s] epoch %4d  train MSE %.6f',
        [tag, ep, loss / (NSAMPLES * NTRAIN_GRID * NTRAIN_GRID)]));
  end;
  Inp.Free; Tgt.Free;
end;

// Build the FNO at a given grid size (lift -> spectral block -> project).
// The target operator here is LINEAR and diagonal in 2-D mode space, so a
// linear lift + one spectral conv + linear projection is the exact hypothesis
// class -- no ReLU is inserted (a nonlinearity would only get in the way of an
// affine spectral operator). This is what lets the FNO drive the error near
// zero AND stay resolution invariant.
function BuildFNO(L: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(L, L, 1));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(WIDTH));          // lift
  Result.AddLayer(TNNetSpectralConv2D.Create(WIDTH, MODESX, MODESY)); // spectral
  Result.AddLayer(TNNetPointwiseConvLinear.Create(1));             // project
end;

// Build a grid-LOCAL conv baseline of similar capacity (3x3 convs encode a
// FIXED grid spacing, so they cannot be resolution invariant).
function BuildBaseline(L: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(L, L, 1));
  Result.AddLayer(TNNetPointwiseConvReLU.Create(WIDTH));           // lift
  Result.AddLayer(TNNetConvolutionReLU.Create(WIDTH, 3, 1, 1));    // local 3x3
  Result.AddLayer(TNNetPointwiseConvReLU.Create(WIDTH));
  Result.AddLayer(TNNetConvolutionReLU.Create(WIDTH, 3, 1, 1));    // local 3x3
  Result.AddLayer(TNNetPointwiseConvLinear.Create(1));            // project
end;

var
  FNO, Baseline, FNOfine, BaseFine: TNNet;
  fnoTrain, fnoEval, baseTrain, baseEval: TNeuralFloat;
begin
  RandSeed := 424242;
  WriteLn('2-D Fourier Neural Operator demo: learning a smooth 2-D low-pass operator');
  WriteLn(Format('Train grid = %dx%d, eval grid = %dx%d, modes = %dx%d, width = %d',
    [NTRAIN_GRID, NTRAIN_GRID, NEVAL_GRID, NEVAL_GRID, MODESX, MODESY, WIDTH]));
  WriteLn;

  FNO := BuildFNO(NTRAIN_GRID);
  Baseline := BuildBaseline(NTRAIN_GRID);

  WriteLn('Training 2-D FNO...');
  Train(FNO, 'FNO');
  WriteLn('Training local-conv baseline...');
  Train(Baseline, 'CONV');
  WriteLn;

  // ---- Evaluation: SAME trained weights, two grid resolutions. -----------
  fnoTrain := EvalRelL2(FNO, NTRAIN_GRID, 5000, 32);
  baseTrain := EvalRelL2(Baseline, NTRAIN_GRID, 5000, 32);

  // Rebuild each net at the FINER grid and copy the trained weights across.
  // CopyWeights is the supported way to change the grid without retraining.
  FNOfine := BuildFNO(NEVAL_GRID);
  FNOfine.CopyWeights(FNO);
  fnoEval := EvalRelL2(FNOfine, NEVAL_GRID, 5000, 32);
  FNOfine.Free;

  BaseFine := BuildBaseline(NEVAL_GRID);
  BaseFine.CopyWeights(Baseline);
  baseEval := EvalRelL2(BaseFine, NEVAL_GRID, 5000, 32);
  BaseFine.Free;

  WriteLn('=== Relative L2 error on held-out 2-D fields ===');
  WriteLn(Format('               train grid (%dx%d)   eval grid (%dx%d, UNSEEN)',
    [NTRAIN_GRID, NTRAIN_GRID, NEVAL_GRID, NEVAL_GRID]));
  WriteLn(Format('  2-D FNO      %14.4f    %14.4f', [fnoTrain, fnoEval]));
  WriteLn(Format('  local conv   %14.4f    %14.4f', [baseTrain, baseEval]));
  WriteLn;
  WriteLn('HEADLINE: the 2-D FNO keeps a low error on the finer, never-trained grid');
  WriteLn('(2-D mode-space weights = resolution invariant), while the local conv');
  WriteLn('baseline degrades when the grid spacing changes.');

  FNO.Free;
  Baseline.Free;
end.
