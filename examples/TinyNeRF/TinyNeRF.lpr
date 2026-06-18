program TinyNeRF;
(*
TinyNeRF: a differentiable VOLUME RENDERER.

A tiny Neural Radiance Field (Mildenhall et al. 2020, "NeRF") learns an
implicit 3-D scene as a coordinate MLP

    F(x,y,z) -> (r, g, b, sigma)

and an image is produced by CASTING RAYS from a pinhole camera, SAMPLING points
along each ray, evaluating F at every sample, and ALPHA-COMPOSITING the samples
into a single pixel colour:

    C = sum_i  T_i * (1 - exp(-sigma_i * delta_i)) * c_i,
    T_i = exp( -sum_{j<i} sigma_j * delta_j )            (transmittance)

This compositing step -- and its hand-derived backward, so the WHOLE render is
trainable end-to-end -- is the new code here. It is implemented as
plain-array math in this driver around the MLP's per-sample outputs: the
composite gradient w.r.t. each sample's (r,g,b,sigma) is computed by hand and
fed to the MLP's last (linear) layer as its OutputError, then Backpropagate().

Self-contained and reproducible: there is NO dataset download. A synthetic
analytic scene -- a single coloured sphere floating in front of a graded
background -- is ray-marched with a KNOWN emission/density field to produce the
ground-truth posed views deterministically inside this program. The NeRF MLP
then learns to reproduce those views from a handful of training poses, and we
render a HELD-OUT pose it never saw during training.

Pipeline:
  Input(3) -> FourierFeatures(M,sigma)        (positional encoding, reused)
           -> FullConnectReLU(W) -> FullConnectReLU(W)
           -> FullConnectLinear(4)             (raw r,g,b,sigma per sample)
  rgb  = sigmoid(raw_rgb)   in (0,1)
  sig  = softplus(raw_s)    >= 0
  then the alpha-composite above collapses the per-ray samples into a pixel.

Outputs (written next to the binary, generated at runtime -- never committed):
  tinynerf_gt.ppm    ground-truth held-out view
  tinynerf_pred.ppm  the trained NeRF's render of the same held-out pose

It prints the held-out PSNR before vs after training to demonstrate the
renderer actually learns the scene (a flat/untrained output would mean the
compositing gradient is wrong).

SMOKE size by default: 24x24 render, 5 train poses, 8 ray samples, ~1000 steps,
finishes comfortably under a couple of minutes on CPU. To scale up, raise
ImgRes / NumTrainPoses / NumSamples / NumIters at the top of RunAlgo.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralfit;

type
  TVec3 = record X, Y, Z: TNeuralFloat; end;

  // One camera pose = position + an orthonormal (right, up, forward) frame.
  TPose = record
    Eye: TVec3;
    Right, Up, Fwd: TVec3;
  end;

const
  // ---- scene geometry (world space) ------------------------------------
  SphereR  = 0.5;             // sphere radius, centred at the origin
  NearT    = 1.5;             // ray sampling near bound
  FarT     = 4.5;             // ray sampling far bound
  CamDist  = 3.0;             // camera distance from origin
  FocalRel = 1.2;            // focal length relative to half-image (FOV)

// ----------------------------------------------------------------------------
// small vector helpers
// ----------------------------------------------------------------------------
function V3(X, Y, Z: TNeuralFloat): TVec3;
begin Result.X := X; Result.Y := Y; Result.Z := Z; end;

function VAdd(const A, B: TVec3): TVec3;
begin Result := V3(A.X + B.X, A.Y + B.Y, A.Z + B.Z); end;

function VSub(const A, B: TVec3): TVec3;
begin Result := V3(A.X - B.X, A.Y - B.Y, A.Z - B.Z); end;

function VScale(const A: TVec3; S: TNeuralFloat): TVec3;
begin Result := V3(A.X * S, A.Y * S, A.Z * S); end;

function VDot(const A, B: TVec3): TNeuralFloat;
begin Result := A.X * B.X + A.Y * B.Y + A.Z * B.Z; end;

function VLen(const A: TVec3): TNeuralFloat;
begin Result := Sqrt(VDot(A, A)); end;

function VNorm(const A: TVec3): TVec3;
var L: TNeuralFloat;
begin
  L := VLen(A);
  if L < 1e-12 then L := 1e-12;
  Result := VScale(A, 1.0 / L);
end;

function VCross(const A, B: TVec3): TVec3;
begin
  Result := V3(A.Y * B.Z - A.Z * B.Y,
               A.Z * B.X - A.X * B.Z,
               A.X * B.Y - A.Y * B.X);
end;

// ----------------------------------------------------------------------------
// camera: a pinhole pose looking at the origin from a given azimuth/elevation
// ----------------------------------------------------------------------------
function PoseLookingAtOrigin(AzimuthDeg, ElevationDeg: TNeuralFloat): TPose;
var
  Az, El: TNeuralFloat;
  WorldUp: TVec3;
begin
  Az := DegToRad(AzimuthDeg);
  El := DegToRad(ElevationDeg);
  Result.Eye := V3(CamDist * Cos(El) * Sin(Az),
                   CamDist * Sin(El),
                   CamDist * Cos(El) * Cos(Az));
  // forward points from the eye toward the origin
  Result.Fwd := VNorm(VSub(V3(0, 0, 0), Result.Eye));
  WorldUp := V3(0, 1, 0);
  Result.Right := VNorm(VCross(Result.Fwd, WorldUp));
  Result.Up := VNorm(VCross(Result.Right, Result.Fwd));
end;

// Ray direction (normalised) for pixel (px,py) in an ImgRes x ImgRes image.
function RayDir(const P: TPose; px, py, ImgRes: integer): TVec3;
var
  u, v: TNeuralFloat;
  D: TVec3;
begin
  // map pixel centre to [-1,1], y flipped so row 0 is the top
  u := (2.0 * (px + 0.5) / ImgRes - 1.0);
  v := -(2.0 * (py + 0.5) / ImgRes - 1.0);
  D := VAdd(VScale(P.Fwd, FocalRel),
            VAdd(VScale(P.Right, u), VScale(P.Up, v)));
  Result := VNorm(D);
end;

// ----------------------------------------------------------------------------
// ANALYTIC ground-truth radiance field (the scene the NeRF must learn).
// A single coloured sphere with a soft surface shell of high density, plus a
// faint graded background haze so empty space is not perfectly black.
//   returns emission colour (r,g,b in [0,1]) and a non-negative density.
// ----------------------------------------------------------------------------
procedure AnalyticField(const P: TVec3; out R, G, B, Sigma: TNeuralFloat);
var
  Dist, Shell: TNeuralFloat;
begin
  Dist := VLen(P);
  // soft shell: density peaks at the sphere surface, falls off smoothly
  Shell := Exp(-Sqr((Dist - SphereR) / 0.12));
  Sigma := 14.0 * Shell;
  // colour varies over the sphere surface so views differ by pose
  R := 0.5 + 0.45 * (P.X / SphereR);
  G := 0.5 + 0.45 * (P.Y / SphereR);
  B := 0.5 + 0.45 * (P.Z / SphereR);
  if R < 0 then R := 0; if R > 1 then R := 1;
  if G < 0 then G := 0; if G > 1 then G := 1;
  if B < 0 then B := 0; if B > 1 then B := 1;
  // faint background haze (very low density everywhere)
  Sigma := Sigma + 0.02;
end;

// ----------------------------------------------------------------------------
// activations used between the linear MLP output and the compositor
// ----------------------------------------------------------------------------
function Sigmoid(X: TNeuralFloat): TNeuralFloat;
begin
  if X >= 0 then Result := 1.0 / (1.0 + Exp(-X))
  else begin Result := Exp(X); Result := Result / (1.0 + Result); end;
end;

function Softplus(X: TNeuralFloat): TNeuralFloat;
begin
  if X > 20 then Result := X
  else Result := Ln(1.0 + Exp(X));
end;

var
  // global RNG-independent geometry shared across the program
  gNumSamples: integer;
  gDelta: TNeuralFloat;        // constant sample spacing along the ray

// ----------------------------------------------------------------------------
// ANALYTIC render: ray-march the known field to produce a ground-truth pixel.
// Uses the SAME compositing equation as the NeRF so the target is reachable.
// ----------------------------------------------------------------------------
procedure RenderPixelAnalytic(const P: TPose; px, py, ImgRes: integer;
  out CR, CG, CB: TNeuralFloat);
var
  D: TVec3;
  i: integer;
  t, alpha, w, Tr: TNeuralFloat;
  sr, sg, sb, ssig: TNeuralFloat;
  Pt: TVec3;
begin
  D := RayDir(P, px, py, ImgRes);
  CR := 0; CG := 0; CB := 0;
  Tr := 1.0;
  for i := 0 to gNumSamples - 1 do
  begin
    t := NearT + (i + 0.5) * gDelta;
    Pt := VAdd(P.Eye, VScale(D, t));
    AnalyticField(Pt, sr, sg, sb, ssig);
    alpha := 1.0 - Exp(-ssig * gDelta);
    w := Tr * alpha;
    CR := CR + w * sr;
    CG := CG + w * sg;
    CB := CB + w * sb;
    Tr := Tr * (1.0 - alpha);
  end;
end;

// ----------------------------------------------------------------------------
// Build the NeRF coordinate MLP.
// ----------------------------------------------------------------------------
function BuildNeRF(NumFeatures: integer; Sigma: TNeuralFloat;
  Width: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(3));
  Result.AddLayer(TNNetFourierFeatures.Create(NumFeatures, Sigma, {seed=}0));
  Result.AddLayer(TNNetFullConnectReLU.Create(Width));
  Result.AddLayer(TNNetFullConnectReLU.Create(Width));
  Result.AddLayer(TNNetFullConnectLinear.Create(4)); // raw r,g,b,sigma
end;

type
  // Per-ray scratch: the NeRF's forward outputs and intermediate composite
  // quantities, kept so the backward pass can reuse them.
  TRaySamples = record
    RawR, RawG, RawB, RawS: array of TNeuralFloat; // linear-layer outputs
    SigR, SigG, SigB: array of TNeuralFloat;        // sigmoid(raw rgb)
    Sig: array of TNeuralFloat;                      // softplus(raw s)
    Alpha, Trans, Weight: array of TNeuralFloat;     // composite terms
    Pt: array of TVec3;                              // world sample points
  end;

procedure InitRaySamples(var RS: TRaySamples; N: integer);
begin
  SetLength(RS.RawR, N); SetLength(RS.RawG, N);
  SetLength(RS.RawB, N); SetLength(RS.RawS, N);
  SetLength(RS.SigR, N); SetLength(RS.SigG, N); SetLength(RS.SigB, N);
  SetLength(RS.Sig, N);
  SetLength(RS.Alpha, N); SetLength(RS.Trans, N); SetLength(RS.Weight, N);
  SetLength(RS.Pt, N);
end;

// Forward render of a single ray with the NeRF. Fills RS and returns colour.
procedure RenderPixelNeRF(NN: TNNet; const P: TPose; px, py, ImgRes: integer;
  Input: TNNetVolume; var RS: TRaySamples; out CR, CG, CB: TNeuralFloat);
var
  D: TVec3;
  i: integer;
  t, Tr: TNeuralFloat;
  Outp: TNNetVolume;
begin
  D := RayDir(P, px, py, ImgRes);
  CR := 0; CG := 0; CB := 0;
  Tr := 1.0;
  for i := 0 to gNumSamples - 1 do
  begin
    t := NearT + (i + 0.5) * gDelta;
    RS.Pt[i] := VAdd(P.Eye, VScale(D, t));
    Input.FData[0] := RS.Pt[i].X;
    Input.FData[1] := RS.Pt[i].Y;
    Input.FData[2] := RS.Pt[i].Z;
    NN.Compute(Input);
    Outp := NN.GetLastLayer().Output;
    RS.RawR[i] := Outp.FData[0];
    RS.RawG[i] := Outp.FData[1];
    RS.RawB[i] := Outp.FData[2];
    RS.RawS[i] := Outp.FData[3];
    RS.SigR[i] := Sigmoid(RS.RawR[i]);
    RS.SigG[i] := Sigmoid(RS.RawG[i]);
    RS.SigB[i] := Sigmoid(RS.RawB[i]);
    RS.Sig[i]  := Softplus(RS.RawS[i]);

    RS.Trans[i] := Tr;
    RS.Alpha[i] := 1.0 - Exp(-RS.Sig[i] * gDelta);
    RS.Weight[i] := Tr * RS.Alpha[i];
    CR := CR + RS.Weight[i] * RS.SigR[i];
    CG := CG + RS.Weight[i] * RS.SigG[i];
    CB := CB + RS.Weight[i] * RS.SigB[i];
    Tr := Tr * (1.0 - RS.Alpha[i]);
  end;
end;

// ----------------------------------------------------------------------------
// Hand-derived backward of the compositor + activations.
//
// Loss for one ray: L = (CR-TR)^2 + (CG-TG)^2 + (CB-TB)^2.
// dL/dCc = 2*(Cc-Tc).  Composite for channel c:
//   Cc = sum_i W_i * cc_i,  W_i = T_i * alpha_i,  cc_i = sigmoid(raw_c_i).
//   T_i = prod_{j<i}(1-alpha_j).
//
// Gradient of every weight W_i w.r.t. alpha_k:
//   * for i = k:  dW_i/dalpha_i = T_i
//   * for i > k:  W_i carries the factor (1-alpha_k), so
//                 dW_i/dalpha_k = -W_i / (1-alpha_k)
// Accumulated over the composite, dL/dalpha_k =
//   gColor_k       := sum_c dL/dCc * cc_k                         (i = k term, *T_k)
//   minus the "behind" contribution = sum_{i>k} (dL/dCc * cc_i * W_i)/(1-alpha_k)
// We compute the running suffix S_c = sum_{i>k} dL/dCc * cc_i * W_i cheaply by
// walking k from far to near.
//
// Then alpha_k = 1 - exp(-sig_k*delta)  =>  dalpha_k/dsig_k = (1-alpha_k)*delta
//   and sig_k = softplus(raw_s_k)        =>  dsig/draw_s = sigmoid(raw_s_k).
// And cc_i = sigmoid(raw_c_i)            =>  dcc/draw_c = cc*(1-cc), scaled by
//   dL/dcc_i = dL/dCc * W_i.
// ----------------------------------------------------------------------------
procedure BackwardRay(NN: TNNet; Input: TNNetVolume; var RS: TRaySamples;
  CR, CG, CB, TR, TG, TB: TNeuralFloat);
var
  k: integer;
  dCR, dCG, dCB: TNeuralFloat;
  oneMinus, dAlpha, dSig, dRawS: TNeuralFloat;
  // running suffix sums of (dL/dCc * cc_i * W_i) for i = k+1 .. N-1
  SufR, SufG, SufB: TNeuralFloat;
  dRawR, dRawG, dRawB: TNeuralFloat;
  Err: TNNetVolume;
begin
  dCR := 2.0 * (CR - TR);
  dCG := 2.0 * (CG - TG);
  dCB := 2.0 * (CB - TB);

  // Walk samples from far (N-1) to near (0), maintaining suffix sums so the
  // "behind" attenuation term is O(N) overall.
  SufR := 0; SufG := 0; SufB := 0;
  for k := gNumSamples - 1 downto 0 do
  begin
    // ---- colour-channel gradients (per-sample, local) ----
    // dL/dcc_k = dL/dCc * W_k ; chain through sigmoid.
    dRawR := dCR * RS.Weight[k] * RS.SigR[k] * (1.0 - RS.SigR[k]);
    dRawG := dCG * RS.Weight[k] * RS.SigG[k] * (1.0 - RS.SigG[k]);
    dRawB := dCB * RS.Weight[k] * RS.SigB[k] * (1.0 - RS.SigB[k]);

    // ---- density gradient via alpha_k ----
    oneMinus := 1.0 - RS.Alpha[k];
    if oneMinus < 1e-8 then oneMinus := 1e-8;
    // dL/dalpha_k = (own weight term) - (behind attenuation term)
    //   own:   sum_c dL/dCc * cc_k * T_k         [ = dCc * cc_k * T_k ]
    //   behind: (1/(1-alpha_k)) * sum_{i>k} dCc * cc_i * W_i  = Suf/(1-alpha_k)
    dAlpha := (dCR * RS.SigR[k] + dCG * RS.SigG[k] + dCB * RS.SigB[k]) * RS.Trans[k]
              - (SufR + SufG + SufB) / oneMinus;

    // alpha_k = 1 - exp(-sig*delta)  => dalpha/dsig = (1-alpha)*delta
    dSig := dAlpha * oneMinus * gDelta;
    // sig = softplus(raw_s) => dsig/draw = sigmoid(raw_s)
    dRawS := dSig * Sigmoid(RS.RawS[k]);

    // ---- inject the 4-vector error into the linear layer & backprop ----
    Input.FData[0] := RS.Pt[k].X;
    Input.FData[1] := RS.Pt[k].Y;
    Input.FData[2] := RS.Pt[k].Z;
    NN.Compute(Input);                 // recompute forward for this sample
    NN.ResetBackpropCallCurrCnt();     // clears per-pass counters + OutputError
    Err := NN.GetLastLayer().OutputError;
    Err.FData[0] := dRawR;
    Err.FData[1] := dRawG;
    Err.FData[2] := dRawB;
    Err.FData[3] := dRawS;
    NN.GetLastLayer().Backpropagate();

    // ---- update suffix sums to include sample k for the next (k-1) ----
    SufR := SufR + dCR * RS.SigR[k] * RS.Weight[k];
    SufG := SufG + dCG * RS.SigG[k] * RS.Weight[k];
    SufB := SufB + dCB * RS.SigB[k] * RS.Weight[k];
  end;
end;

// ----------------------------------------------------------------------------
// PPM (P6) writer
// ----------------------------------------------------------------------------
procedure WritePPM(const FileName: string; const Img: array of TNeuralFloat;
  ImgRes: integer);
var
  F: TFileStream;
  Hdr: AnsiString;
  i: integer;
  b: byte;
  v: TNeuralFloat;
begin
  F := TFileStream.Create(FileName, fmCreate);
  try
    Hdr := 'P6'#10 + IntToStr(ImgRes) + ' ' + IntToStr(ImgRes) + #10'255'#10;
    F.WriteBuffer(Hdr[1], Length(Hdr));
    for i := 0 to ImgRes * ImgRes * 3 - 1 do
    begin
      v := Img[i];
      if v < 0 then v := 0; if v > 1 then v := 1;
      b := Round(v * 255);
      F.WriteBuffer(b, 1);
    end;
  finally
    F.Free;
  end;
end;

// PSNR (dB) between a render and the ground truth, both in [0,1].
function PSNR(const A, B: array of TNeuralFloat; N: integer): TNeuralFloat;
var
  i: integer;
  s, d: TNeuralFloat;
begin
  s := 0;
  for i := 0 to N - 1 do begin d := A[i] - B[i]; s := s + d * d; end;
  s := s / N;
  if s < 1e-12 then s := 1e-12;
  Result := 10.0 * Log10(1.0 / s);
end;

// ----------------------------------------------------------------------------
procedure RunAlgo();
const
  // ---- SMOKE-sized run (scale these up for a sharper result) ----
  ImgRes        = 24;
  NumTrainPoses = 5;
  NumFeatures   = 48;     // M; Fourier output Depth = 2*M
  FourierSigma  = 3.0;
  HiddenWidth   = 96;
  NumIters      = 1000;   // ray batches
  RaysPerBatch  = 64;
  LearningRate  = 0.004;
var
  NN: TNNet;
  Poses: array of TPose;
  HeldOut: TPose;
  Input: TNNetVolume;
  RS: TRaySamples;
  // ground-truth training pixels: [pose][py*ImgRes+px]*3 + channel
  GT: array of array of TNeuralFloat;
  HeldGT, HeldPred: array of TNeuralFloat;
  i, p, px, py, Iter, NPix, Ray: integer;
  CR, CG, CB, TRc, TGc, TBc: TNeuralFloat;
  Az: array[0..4] of TNeuralFloat;
  RayPose, RayPx, RayPy: integer;
  T0: TDateTime;
  BatchLoss, IntervalLoss: TNeuralFloat;
  IntervalCount: integer;
  PSNRbefore, PSNRafter: TNeuralFloat;

  procedure RenderHeldOut(var Dst: array of TNeuralFloat);
  var qx, qy: integer; rr, gg, bb: TNeuralFloat;
  begin
    for qy := 0 to ImgRes - 1 do
      for qx := 0 to ImgRes - 1 do
      begin
        RenderPixelNeRF(NN, HeldOut, qx, qy, ImgRes, Input, RS, rr, gg, bb);
        i := (qy * ImgRes + qx) * 3;
        Dst[i] := rr; Dst[i + 1] := gg; Dst[i + 2] := bb;
      end;
  end;

begin
  RandSeed := 42;
  gNumSamples := 8;
  gDelta := (FarT - NearT) / gNumSamples;
  NPix := ImgRes * ImgRes;

  WriteLn('TinyNeRF: a differentiable volume renderer (NeRF, Mildenhall 2020)');
  WriteLn('Scene: analytic coloured sphere, ray-marched ground truth.');
  WriteLn('Render ', ImgRes, 'x', ImgRes, '  train poses=', NumTrainPoses,
    '  samples/ray=', gNumSamples, '  iters=', NumIters,
    '  rays/batch=', RaysPerBatch);
  WriteLn('MLP: Input(3) -> FourierFeatures(M=', NumFeatures, ', sigma=',
    FourierSigma:0:1, ') -> ReLU(', HiddenWidth, ') -> ReLU(', HiddenWidth,
    ') -> Linear(4)');
  WriteLn;

  // ---- training camera poses, evenly spaced azimuths around the sphere ----
  Az[0] := 10;  Az[1] := 80;  Az[2] := 150; Az[3] := 220; Az[4] := 300;
  SetLength(Poses, NumTrainPoses);
  for p := 0 to NumTrainPoses - 1 do
    Poses[p] := PoseLookingAtOrigin(Az[p], 20.0);
  // held-out pose: a NEW azimuth + elevation not in the training set
  HeldOut := PoseLookingAtOrigin(45.0, -10.0);

  // ---- pre-render analytic ground truth for every training pose ----
  SetLength(GT, NumTrainPoses);
  for p := 0 to NumTrainPoses - 1 do
  begin
    SetLength(GT[p], NPix * 3);
    for py := 0 to ImgRes - 1 do
      for px := 0 to ImgRes - 1 do
      begin
        RenderPixelAnalytic(Poses[p], px, py, ImgRes, TRc, TGc, TBc);
        i := (py * ImgRes + px) * 3;
        GT[p][i] := TRc; GT[p][i + 1] := TGc; GT[p][i + 2] := TBc;
      end;
  end;
  // held-out ground truth
  SetLength(HeldGT, NPix * 3);
  SetLength(HeldPred, NPix * 3);
  for py := 0 to ImgRes - 1 do
    for px := 0 to ImgRes - 1 do
    begin
      RenderPixelAnalytic(HeldOut, px, py, ImgRes, TRc, TGc, TBc);
      i := (py * ImgRes + px) * 3;
      HeldGT[i] := TRc; HeldGT[i + 1] := TGc; HeldGT[i + 2] := TBc;
    end;
  WritePPM('tinynerf_gt.ppm', HeldGT, ImgRes);

  // ---- build the NeRF ----
  NN := BuildNeRF(NumFeatures, FourierSigma, HiddenWidth);
  NN.SetLearningRate(LearningRate, {Momentum=}0.9);
  NN.SetL2Decay(0.0);
  NN.GetLastLayer().IncDepartingBranchesCnt(); // we seed last-layer error by hand

  Input := TNNetVolume.Create(1, 1, 3);
  InitRaySamples(RS, gNumSamples);

  // ---- baseline (untrained) held-out PSNR ----
  RenderHeldOut(HeldPred);
  PSNRbefore := PSNR(HeldPred, HeldGT, NPix * 3);
  WriteLn('Held-out PSNR BEFORE training = ', PSNRbefore:0:2, ' dB');
  WriteLn('Training...');

  T0 := Now();
  IntervalLoss := 0;
  IntervalCount := 0;
  for Iter := 1 to NumIters do
  begin
    NN.ClearDeltas();
    BatchLoss := 0;
    for Ray := 0 to RaysPerBatch - 1 do
    begin
      RayPose := Random(NumTrainPoses);
      RayPx := Random(ImgRes);
      RayPy := Random(ImgRes);
      RenderPixelNeRF(NN, Poses[RayPose], RayPx, RayPy, ImgRes, Input, RS,
        CR, CG, CB);
      i := (RayPy * ImgRes + RayPx) * 3;
      TRc := GT[RayPose][i]; TGc := GT[RayPose][i + 1]; TBc := GT[RayPose][i + 2];
      BatchLoss := BatchLoss + Sqr(CR - TRc) + Sqr(CG - TGc) + Sqr(CB - TBc);
      BackwardRay(NN, Input, RS, CR, CG, CB, TRc, TGc, TBc);
    end;
    NN.UpdateWeights();
    IntervalLoss := IntervalLoss + BatchLoss / (RaysPerBatch * 3);
    Inc(IntervalCount);

    if (Iter = 1) or (Iter mod 250 = 0) or (Iter = NumIters) then
    begin
      WriteLn('  iter ', Iter:5, '  mean ray MSE = ',
        (IntervalLoss / IntervalCount):0:6);
      IntervalLoss := 0;
      IntervalCount := 0;
    end;
  end;

  WriteLn('Training done in ', FormatDateTime('nn:ss', Now() - T0));

  // ---- render the held-out view with the trained NeRF ----
  RenderHeldOut(HeldPred);
  WritePPM('tinynerf_pred.ppm', HeldPred, ImgRes);
  PSNRafter := PSNR(HeldPred, HeldGT, NPix * 3);

  WriteLn;
  WriteLn('=== HELD-OUT NOVEL VIEW (pose never seen during training) ===');
  WriteLn('  PSNR before training = ', PSNRbefore:0:2, ' dB');
  WriteLn('  PSNR after  training = ', PSNRafter:0:2, ' dB');
  WriteLn('  improvement          = ', (PSNRafter - PSNRbefore):0:2, ' dB');
  WriteLn('  wrote tinynerf_gt.ppm and tinynerf_pred.ppm (', ImgRes, 'x',
    ImgRes, ')');
  if PSNRafter > PSNRbefore + 2.0 then
    WriteLn('  OK: the volume renderer learned the scene.')
  else
    WriteLn('  WARNING: little improvement -- check the compositing gradient.');

  Input.Free;
  NN.Free;
end;

var
  Application: record Title: string; end;

begin
  Application.Title := 'TinyNeRF Example';
  RunAlgo();
end.
