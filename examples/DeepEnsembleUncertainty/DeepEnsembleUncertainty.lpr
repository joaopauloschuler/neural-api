program DeepEnsembleUncertainty;
(*
DeepEnsembleUncertainty: the gold-standard predictive-uncertainty baseline of
Lakshminarayanan, Pritzel & Blundell, "Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles" (NeurIPS 2017), on a pure-CPU toy
using ONLY existing layers (no new layer type).

It trains M (default 5) INDEPENDENT small softmax classifiers — identical
architecture but a DIFFERENT RandSeed per member — on the SAME synthetic
3-cluster 2D task that examples/MCDropoutUncertainty/ uses. The data generator
and the three probe groups (cluster cores / an out-of-distribution band in the
empty space BETWEEN clusters / a labelled validation split) are copied verbatim
from MCDropoutUncertainty so the two epistemic estimates are apples-to-apples.

At inference it averages the M per-member post-softmax probability vectors and
reports, per probe group:

  (a) the AVERAGE single-member top-1 accuracy and ECE/Brier vs the ENSEMBLE's
      accuracy and ECE/Brier, both computed through the already-landed
      neuralcalibration.ComputeCalibration / CalibrationReport (we do NOT
      hand-roll calibration). Headline: the ensemble improves BOTH accuracy and
      calibration over the average member.

  (b) the predictive-ENTROPY decomposition (nats):
          total H[mean_p]  =  aleatoric (mean_m H[p_m])  +  epistemic (MI),
      where the epistemic term is the mutual information / member disagreement
      I = H[mean_p] - mean_m H[p_m] >= 0 (Jensen). The epistemic term spikes on
      the OOD band and sits ~0 on the confident cluster cores.

  (c) an ASCII bar of per-group epistemic uncertainty.

BUILT-IN CORRECTNESS SIGNALS (asserted + printed):
  - M=1 reproduces a single member's predictions bit-for-bit (the ensemble of
    one == that one model);
  - the ensemble accuracy is >= the mean single-member accuracy on every group;
  - the epistemic (MI) term is >= 0 on every probe, and strictly higher on the
    OOD band than on the cluster cores.

HOW THIS DIFFERS FROM THE SIBLING EXAMPLES (read these for contrast):
  * examples/MCDropoutUncertainty/ estimates epistemic uncertainty by SAMPLING
    dropout masks inside ONE trained network (MC-dropout, Gal & Ghahramani
    2016). Deep ensembles instead use M genuinely INDEPENDENT networks. Both run
    on the SAME 3-cluster task here, so their epistemic estimates can be
    compared head-to-head: dropout samples one posterior mode, an ensemble
    explores several.
  * examples/KnowledgeDistillation/ treats an ensemble as the natural TEACHER
    to be compressed into one student. Here we KEEP the ensemble and quantify
    its uncertainty instead of distilling it away.
  * TestTimeAugmentation averages predictions over input TRANSFORMS of a SINGLE
    model; that captures input sensitivity, not the model-disagreement
    (epistemic) signal that M independent models give.

Pure CPU, SYNTHETIC data (no download), deterministic per seed list, well under
a minute. Single-threaded for reproducibility.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralcalibration;

const
  cClasses   = 3;
  cHidden    = 32;
  cEpochs    = 300;
  cBatch     = 48;
  cMembers   = 5;         // ensemble size M
  cTrainPerC = 200;       // training points per cluster
  cCoreProbe = 12;        // probe points per cluster core
  cOODProbe  = 24;        // probe points along the OOD band
  cValPerC   = 26;        // clean validation points per cluster (calibration)
  cValHard   = 6;         // hard boundary points per cluster (some misclassd)
  // Per-member RandSeeds. M=1 uses only the first; element 0 must match the
  // single-model baseline seed for the bit-for-bit check.
  cSeeds: array[0..cMembers - 1] of integer = (2026, 7, 99, 31337, 424242);

// Cluster centres: a wide triangle, leaving a large empty centre for the OOD
// band to sit in. (Verbatim from MCDropoutUncertainty for apples-to-apples.)
procedure ClusterCentre(C: integer; out cx, cy: TNeuralFloat);
begin
  case C of
    0: begin cx := -3.0; cy := -2.0; end;
    1: begin cx :=  3.0; cy := -2.0; end;
  else begin cx :=  0.0; cy :=  3.2; end;
  end;
end;

procedure MakeInput(out V: TNNetVolume; x, y: TNeuralFloat);
begin
  V := TNNetVolume.Create(2, 1, 1);
  V.Raw[0] := x;
  V.Raw[1] := y;
end;

procedure MakeClusterPoint(C: integer; out V: TNNetVolume);
var
  cx, cy: TNeuralFloat;
begin
  ClusterCentre(C, cx, cy);
  MakeInput(V, cx + (Random - 0.5) * 0.7, cy + (Random - 0.5) * 0.7);
end;

// A plain softmax classifier (no dropout): each ensemble member is one of
// these, trained from a different RandSeed.
procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(2, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(0.02, 0.9);
end;

procedure Train(NN: TNNet; Epochs: integer);
var
  Ep, B, C: integer;
  X, Yt: TNNetVolume;
begin
  for Ep := 1 to Epochs do
    for B := 1 to cBatch do
    begin
      C := Random(cClasses);
      MakeClusterPoint(C, X);
      Yt := TNNetVolume.Create(cClasses, 1, 1);
      Yt.Raw[C] := 1;
      try
        NN.Compute(X);
        NN.Backpropagate(Yt);
      finally
        X.Free;
        Yt.Free;
      end;
    end;
end;

// Shannon entropy (nats) of a probability vector.
function Entropy(const P: array of TNeuralFloat; N: integer): TNeuralFloat;
var
  I: integer;
begin
  Result := 0;
  for I := 0 to N - 1 do
    if P[I] > 1e-12 then
      Result := Result - P[I] * Ln(P[I]);
end;

type
  // Per-member cached probability vectors for one probe set:
  //   MemberProbs[m][i][k] = member m's P(class k | probe i).
  TProbCube = array of array of array of TNeuralFloat;

// Runs every member over every probe in Probes and caches the post-softmax
// probability vectors. M members x Probes.Count probes x cClasses.
procedure ForwardAll(Members: array of TNNet; Probes: TNNetVolumeList;
  MM: integer; out Cube: TProbCube);
var
  m, i, k: integer;
  Outp: TNNetVolume;
begin
  SetLength(Cube, MM);
  for m := 0 to MM - 1 do
  begin
    SetLength(Cube[m], Probes.Count);
    for i := 0 to Probes.Count - 1 do
    begin
      Members[m].Compute(Probes[i]);
      Outp := Members[m].GetLastLayer().Output;
      SetLength(Cube[m][i], cClasses);
      for k := 0 to cClasses - 1 do
        Cube[m][i][k] := Outp.Raw[k];
    end;
  end;
end;

// Builds a list of ensemble-MEAN probability vectors (one per probe). Each
// vector is wrapped as a TNNetVolume of size cClasses so it can be fed as the
// INPUT of a passthrough net into the existing ComputeCalibration (the net's
// output then equals the supplied mean-probability simplex).
function MeanProbVolumes(const Cube: TProbCube; MM, NumProbes: integer): TNNetVolumeList;
var
  i, k, m: integer;
  V: TNNetVolume;
begin
  // The plain average of M softmax simplices is itself a simplex (sums to ~1
  // up to float drift, well within ComputeCalibration's 1e-3 tolerance). We do
  // NOT renormalise: that keeps the M=1 case EXACTLY equal to member 0's own
  // probability vector (the ensemble-of-one == that one model, bit-for-bit).
  Result := TNNetVolumeList.Create(True);
  for i := 0 to NumProbes - 1 do
  begin
    V := TNNetVolume.Create(cClasses, 1, 1);
    for k := 0 to cClasses - 1 do
    begin
      for m := 0 to MM - 1 do
        V.Raw[k] := V.Raw[k] + Cube[m][i][k];
      V.Raw[k] := V.Raw[k] / MM;
    end;
    Result.Add(V);
  end;
end;

// A trivial passthrough net: Input(cClasses) -> Identity. Feeding a probability
// vector as its input makes its output equal that vector, so the forward-only
// neuralcalibration.ComputeCalibration sees the ensemble-mean probabilities as
// a softmax head. This lets us reuse ComputeCalibration unchanged for the
// ensemble (we do NOT re-implement ECE/Brier).
procedure BuildPassthrough(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cClasses, 1, 1));
  NN.AddLayer(TNNetIdentity.Create());
end;

// Entropy decomposition over a cached probe set.
//   TotalH    = mean_i H[mean_m p_m(i)]      (predictive / total entropy)
//   AleaH     = mean_i mean_m H[p_m(i)]       (aleatoric)
//   EpistMI   = TotalH - AleaH                (epistemic / mutual information)
//   MinMI/MaxMI = per-probe MI range (MinMI must be >= 0).
procedure EntropyDecomp(const Cube: TProbCube; MM, NumProbes: integer;
  out TotalH, AleaH, EpistMI, MinMI, MaxMI: TNeuralFloat);
var
  i, k, m: integer;
  MeanP: array of TNeuralFloat;
  hTot, hAleaI, mi: TNeuralFloat;
begin
  SetLength(MeanP, cClasses);
  TotalH := 0; AleaH := 0;
  MinMI := 1e30; MaxMI := -1e30;
  for i := 0 to NumProbes - 1 do
  begin
    for k := 0 to cClasses - 1 do MeanP[k] := 0;
    hAleaI := 0;
    for m := 0 to MM - 1 do
    begin
      for k := 0 to cClasses - 1 do MeanP[k] := MeanP[k] + Cube[m][i][k];
      hAleaI := hAleaI + Entropy(Cube[m][i], cClasses);
    end;
    for k := 0 to cClasses - 1 do MeanP[k] := MeanP[k] / MM;
    hAleaI := hAleaI / MM;
    hTot := Entropy(MeanP, cClasses);
    mi := hTot - hAleaI;            // per-probe mutual information
    TotalH := TotalH + hTot;
    AleaH  := AleaH + hAleaI;
    if mi < MinMI then MinMI := mi;
    if mi > MaxMI then MaxMI := mi;
  end;
  TotalH := TotalH / NumProbes;
  AleaH  := AleaH / NumProbes;
  EpistMI := TotalH - AleaH;
end;

// Mean single-member top-1 accuracy + mean single-member ECE/Brier over a
// labelled probe set, via per-member ComputeCalibration.
procedure MeanMemberCalib(Members: array of TNNet; Probes: TNNetVolumeList;
  const Labels: array of integer; MM: integer;
  out Acc, ECE, Brier: TNeuralFloat);
var
  m: integer;
  R: TNeuralCalibrationReport;
begin
  Acc := 0; ECE := 0; Brier := 0;
  for m := 0 to MM - 1 do
  begin
    R := ComputeCalibration(Members[m], Probes, Labels, 10);
    Acc := Acc + R.Accuracy;
    ECE := ECE + R.ECE;
    Brier := Brier + R.Brier;
  end;
  Acc := Acc / MM; ECE := ECE / MM; Brier := Brier / MM;
end;

procedure PrintEpistemicBar(const Name: string; MI, Scale: TNeuralFloat);
var
  n: integer;
begin
  n := Round((MI / Scale) * 50);
  if n < 0 then n := 0;
  if n > 50 then n := 50;
  WriteLn(Format('  %-14s |%s %.4f', [Name, StringOfChar('#', n), MI]));
end;

var
  Members: array[0..cMembers - 1] of TNNet;
  Cores, OOD, Val: TNNetVolumeList;
  ValLabels, CoresLabels: array of integer;
  EnsembleNet: TNNet;
  MeanVolCores, MeanVolOOD, MeanVolVal: TNNetVolumeList;
  CubeC, CubeO, CubeV: TProbCube;
  V: TNNetVolume;
  C, K, m, NM: integer;
  t0, t1: TDateTime;
  tx, cx, cy, nx, ny: TNeuralFloat;
  mTotC, mAlC, mMIC, mnMIC, mxMIC: TNeuralFloat;
  mTotO, mAlO, mMIO, mnMIO, mxMIO: TNeuralFloat;
  mTotV, mAlV, mMIV, mnMIV, mxMIV: TNeuralFloat;
  memAccC, memECEc, memBrierC: TNeuralFloat;
  memAccV, memECEv, memBrierV: TNeuralFloat;
  ensC, ensV: TNeuralCalibrationReport;
  Single0Cube: TProbCube;
  bitForBitOK: boolean;
  maxAbsDiff: TNeuralFloat;
  barScale: TNeuralFloat;
begin
  // Members train via the per-sample Compute/Backpropagate loop below (no
  // TNeuralFit thread pool), so determinism comes purely from RandSeed.
  NM := cMembers;
  t0 := Now();

  // ----- Build probe groups (verbatim layout from MCDropoutUncertainty) -----
  // Probe construction uses one fixed RNG seed so the probe coordinates are
  // identical regardless of M and identical to MCDropoutUncertainty.
  RandSeed := 2026;
  Cores := TNNetVolumeList.Create(True);
  OOD := TNNetVolumeList.Create(True);
  Val := TNNetVolumeList.Create(True);

  // (1) cluster cores: tight around each centre (in-distribution).
  SetLength(CoresLabels, cClasses * cCoreProbe);
  for C := 0 to cClasses - 1 do
    for K := 0 to cCoreProbe - 1 do
    begin
      MakeClusterPoint(C, V);
      CoresLabels[Cores.Count] := C;
      Cores.Add(V);
    end;

  // (2) OOD band: a horizontal sweep through the empty centre (0,0) area.
  for K := 0 to cOODProbe - 1 do
  begin
    tx := -2.0 + 4.0 * (K / (cOODProbe - 1));   // x in [-2, 2]
    MakeInput(V, tx, 0.4);                        // y=0.4, dead centre
    OOD.Add(V);
  end;

  // (3) validation split (labelled): clean cluster points + hard boundary pts.
  SetLength(ValLabels, cClasses * (cValPerC + cValHard));
  for C := 0 to cClasses - 1 do
  begin
    for K := 0 to cValPerC - 1 do
    begin
      MakeClusterPoint(C, V);
      ValLabels[Val.Count] := C;
      Val.Add(V);
    end;
    for K := 0 to cValHard - 1 do
    begin
      ClusterCentre(C, cx, cy);
      ClusterCentre((C + 1) mod cClasses, nx, ny);
      MakeInput(V, cx + 0.60 * (nx - cx) + (Random - 0.5) * 0.4,
                   cy + 0.60 * (ny - cy) + (Random - 0.5) * 0.4);
      ValLabels[Val.Count] := C;
      Val.Add(V);
    end;
  end;

  WriteLn('DeepEnsembleUncertainty: ', NM,
    ' INDEPENDENT softmax MLPs on a synthetic 3-cluster 2D task.');
  WriteLn('Training each member (different RandSeed) for ', cEpochs,
    ' epochs of batch ', cBatch, ' ...');

  // ----- Train M independent members, each from its own RandSeed -----
  for m := 0 to NM - 1 do
  begin
    RandSeed := cSeeds[m];           // independence comes from distinct seeds
    BuildNet(Members[m]);
    Train(Members[m], cEpochs);
    WriteLn(Format('  member %d (seed %d) trained.', [m, cSeeds[m]]));
  end;

  // ----- Forward every member over every probe group; cache probabilities ---
  ForwardAll(Members, Cores, NM, CubeC);
  ForwardAll(Members, OOD,   NM, CubeO);
  ForwardAll(Members, Val,   NM, CubeV);

  // ===================== (a) accuracy + calibration =========================
  WriteLn;
  WriteLn(StringOfChar('=', 78));
  WriteLn('(a) ACCURACY + CALIBRATION: average single member vs the ENSEMBLE');
  WriteLn('    (ECE/Brier via neuralcalibration.ComputeCalibration).');
  WriteLn(StringOfChar('=', 78));

  // ensemble-mean probability vectors, fed through a passthrough net.
  MeanVolCores := MeanProbVolumes(CubeC, NM, Cores.Count);
  MeanVolVal   := MeanProbVolumes(CubeV, NM, Val.Count);
  BuildPassthrough(EnsembleNet);
  ensC := ComputeCalibration(EnsembleNet, MeanVolCores, CoresLabels, 10);
  ensV := ComputeCalibration(EnsembleNet, MeanVolVal,   ValLabels, 10);

  MeanMemberCalib(Members, Cores, CoresLabels, NM, memAccC, memECEc, memBrierC);
  MeanMemberCalib(Members, Val,   ValLabels,   NM, memAccV, memECEv, memBrierV);

  WriteLn('  CLUSTER CORES (in-distribution):');
  WriteLn(Format('    mean member:  acc=%.4f  ECE=%.4f  Brier=%.4f',
    [memAccC, memECEc, memBrierC]));
  WriteLn(Format('    ENSEMBLE   :  acc=%.4f  ECE=%.4f  Brier=%.4f',
    [ensC.Accuracy, ensC.ECE, ensC.Brier]));
  WriteLn('  VALIDATION SPLIT (clean + hard boundary points):');
  WriteLn(Format('    mean member:  acc=%.4f  ECE=%.4f  Brier=%.4f',
    [memAccV, memECEv, memBrierV]));
  WriteLn(Format('    ENSEMBLE   :  acc=%.4f  ECE=%.4f  Brier=%.4f',
    [ensV.Accuracy, ensV.ECE, ensV.Brier]));
  WriteLn;
  WriteLn('  Full CalibrationReport for the ENSEMBLE on the validation split:');
  Write(CalibrationReport(EnsembleNet, MeanVolVal, ValLabels, 10));

  // ===================== (b) entropy decomposition ==========================
  EntropyDecomp(CubeC, NM, Cores.Count, mTotC, mAlC, mMIC, mnMIC, mxMIC);
  EntropyDecomp(CubeO, NM, OOD.Count,   mTotO, mAlO, mMIO, mnMIO, mxMIO);
  EntropyDecomp(CubeV, NM, Val.Count,   mTotV, mAlV, mMIV, mnMIV, mxMIV);

  WriteLn;
  WriteLn(StringOfChar('=', 78));
  WriteLn('(b) PREDICTIVE-ENTROPY DECOMPOSITION (nats): total = aleatoric + epistemic');
  WriteLn('    epistemic = mutual information I = H[mean_p] - mean_m H[p_m] >= 0');
  WriteLn(StringOfChar('=', 78));
  WriteLn('  group           total(H)   aleatoric   epistemic(MI)   MI range');
  WriteLn(Format('  cluster-cores   %8.4f   %8.4f   %11.4f    [%.4f, %.4f]',
    [mTotC, mAlC, mMIC, mnMIC, mxMIC]));
  WriteLn(Format('  OOD-band        %8.4f   %8.4f   %11.4f    [%.4f, %.4f]',
    [mTotO, mAlO, mMIO, mnMIO, mxMIO]));
  WriteLn(Format('  validation      %8.4f   %8.4f   %11.4f    [%.4f, %.4f]',
    [mTotV, mAlV, mMIV, mnMIV, mxMIV]));

  // ===================== (c) ASCII epistemic bar ============================
  WriteLn;
  WriteLn(StringOfChar('=', 78));
  WriteLn('(c) PER-GROUP EPISTEMIC UNCERTAINTY (mean mutual information, nats)');
  WriteLn(StringOfChar('=', 78));
  barScale := mMIO;
  if barScale < 1e-6 then barScale := 1e-6;
  PrintEpistemicBar('cluster-cores', mMIC, barScale);
  PrintEpistemicBar('OOD-band',      mMIO, barScale);
  PrintEpistemicBar('validation',    mMIV, barScale);
  WriteLn('  (bar full-scale = OOD-band mean MI)');

  // ===================== correctness signals ================================
  WriteLn;
  WriteLn(StringOfChar('=', 78));
  WriteLn('CORRECTNESS SIGNALS');
  WriteLn(StringOfChar('=', 78));

  // (1) M=1 reproduces member 0's predictions bit-for-bit: the ensemble-mean
  // over a single member is exactly that member's probability vector.
  ForwardAll(Members, OOD, 1, Single0Cube);
  MeanVolOOD := MeanProbVolumes(Single0Cube, 1, OOD.Count); // M=1 mean
  maxAbsDiff := 0;
  for K := 0 to OOD.Count - 1 do
    for C := 0 to cClasses - 1 do
      maxAbsDiff := Max(maxAbsDiff,
        Abs(MeanVolOOD[K].Raw[C] - CubeO[0][K][C]));
  bitForBitOK := maxAbsDiff = 0.0;
  WriteLn(Format('  [%s] M=1 reproduces member 0 bit-for-bit (max|diff|=%.3e).',
    [BoolToStr(bitForBitOK, 'PASS', 'FAIL'), maxAbsDiff]));
  if not bitForBitOK then
    raise Exception.Create('M=1 did not reproduce member 0 bit-for-bit.');

  // (2) ensemble accuracy >= mean single-member accuracy on every labelled grp.
  WriteLn(Format('  [%s] ensemble acc >= mean-member acc on cores (%.4f >= %.4f).',
    [BoolToStr(ensC.Accuracy >= memAccC - 1e-9, 'PASS', 'FAIL'),
     ensC.Accuracy, memAccC]));
  WriteLn(Format('  [%s] ensemble acc >= mean-member acc on val   (%.4f >= %.4f).',
    [BoolToStr(ensV.Accuracy >= memAccV - 1e-9, 'PASS', 'FAIL'),
     ensV.Accuracy, memAccV]));
  if (ensC.Accuracy < memAccC - 1e-9) or (ensV.Accuracy < memAccV - 1e-9) then
    raise Exception.Create('Ensemble accuracy below mean single-member accuracy.');

  // (3) epistemic MI >= 0 everywhere, strictly higher on OOD than cores.
  WriteLn(Format('  [%s] epistemic MI >= 0 everywhere (min over groups = %.4f).',
    [BoolToStr((mnMIC >= -1e-9) and (mnMIO >= -1e-9) and (mnMIV >= -1e-9),
       'PASS', 'FAIL'),
     Min(mnMIC, Min(mnMIO, mnMIV))]));
  WriteLn(Format('  [%s] OOD epistemic > cores epistemic (%.4f > %.4f).',
    [BoolToStr(mMIO > mMIC, 'PASS', 'FAIL'), mMIO, mMIC]));
  if (mnMIC < -1e-9) or (mnMIO < -1e-9) or (mnMIV < -1e-9) then
    raise Exception.Create('Epistemic MI went negative.');
  if not (mMIO > mMIC) then
    raise Exception.Create('OOD epistemic not strictly above cores epistemic.');

  WriteLn;
  WriteLn('Takeaway: averaging M independent nets improves BOTH accuracy and ',
    'calibration over the average member, and the mutual-information term ',
    'lights up on the OOD band while staying ~0 on the confident cluster ',
    'cores -- "the ensemble knows what it does not know".');

  // ----- cleanup -----
  MeanVolCores.Free;
  MeanVolOOD.Free;
  MeanVolVal.Free;
  EnsembleNet.Free;
  for m := 0 to NM - 1 do Members[m].Free;
  Val.Free;
  OOD.Free;
  Cores.Free;

  t1 := Now();
  WriteLn(Format('Total runtime: %.2f s.', [(t1 - t0) * 24 * 3600]));
end.
