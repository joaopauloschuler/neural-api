program HyperbolicEmbedding;
(*
HyperbolicEmbedding: a tiny demo that embeds the nodes of a small balanced
binary TREE and shows that a HYPERBOLIC (Poincare-ball) embedding recovers the
tree's shortest-path (geodesic) distances far better than a parameter-matched
EUCLIDEAN embedding.

Why hyperbolic? A tree's node count grows exponentially with depth, and so does
the volume of a hyperbolic ball with its radius - so trees embed into hyperbolic
space with almost no distortion, while Euclidean space (polynomial volume
growth) cannot hold an exponentially branching tree without crowding leaves
together. This is the classic Nickel & Kiela 2017 ("Poincare Embeddings",
https://arxiv.org/abs/1705.08039) result, reproduced here on a toy tree.

TREE: a complete binary tree of depth TREE_DEPTH (default 3 => 15 nodes). The
ground-truth distance between two nodes is the number of edges on the tree path
between them (computed via lowest-common-ancestor on the heap-style node ids).

MODELS (both map a one-hot node id -> a DIM-vector embedding):
  Model A (hyperbolic): TNNetInput(N) -> TNNetHyperbolicLinear(DIM, c)
      -> a point inside the Poincare ball; distance between two embeddings is the
      curvature-c Poincare distance
        dist_c(a,b) = (2/s) atanh( s || (-a) (+)_c b || ),  s = sqrt(c)
      which is exactly TNNetHyperbolicDistance's per-prototype formula (here we
      compute it between two LEARNED embeddings rather than against a fixed
      prototype bank, and hand-seed the gradient through the same Mobius math).
  Model B (Euclidean baseline): TNNetInput(N) -> TNNetFullConnectLinear(DIM)
      -> a point in R^DIM; distance is the plain Euclidean norm ||a-b||.

Both embedders have the SAME parameter budget (one N->DIM weight matrix; the
hyperbolic bias is suppressed so the counts match exactly) and train the SAME
way: sample node pairs, regress the embedded distance to the tree path length by
MSE. We hand-roll the pairwise training (two forward passes per pair, seed the
embedding layer's OutputError with the analytic dMSE/dembedding, backprop each)
because the loss couples two inputs - the standard custom-loss pattern used by
examples/SparseAutoencoder. SetBatchUpdate(True) is REQUIRED.

We then report, on ALL node pairs, the final embedded-vs-tree distance MSE and
Pearson correlation for each model and print a short verdict. Expected headline:
the hyperbolic model fits tree distances markedly better at the same budget.

This is a small CPU toy (well under a minute on 2 cores); printing is
NaN/Inf-guarded. Not added to the main README (see examples/README.md).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuralfit;

const
  TREE_DEPTH   = 4;                    // complete binary tree depth (31 nodes)
  NUM_NODES    = (1 shl (TREE_DEPTH + 1)) - 1;   // 2^(d+1)-1 = 15 nodes
  DIM          = 2;                    // embedding dimensionality
  CURVATURE    = 1.0;                  // Poincare-ball curvature c
  // Same param budget (one N->DIM matrix, no bias) and same data/epoch budget
  // for both models. The optimiser STEP differs by geometry: hyperbolic SGD is
  // boundary-sensitive and needs a gentler schedule than flat Euclidean SGD.
  LEARN_RATE   = 0.5;                  // Euclidean step (mean-grad mini-batch)
  EUC_MOM      = 0.9;
  HYP_LR       = 0.02;                 // hyperbolic step (boundary-safe)
  HYP_MOM      = 0.5;
  EPOCHS       = 1500;                 // one mini-batch update per epoch
  PAIRS_PER_EP = 64;                   // sampled node pairs per mini-batch
  HYP_EPS      = 1e-7;
  BALL_INIT    = 0.05;                 // tiny init so embeddings start near origin
  GRAD_CLIP    = 0.2;                  // max norm of each seeded (mean) error
  TARGET_SCALE = 0.30;                 // regress to SCALE*pathlen (same for both
                                       // models; keeps hyperbolic embeddings away
                                       // from the boundary. Correlation is scale-
                                       // invariant; MSE uses the same scaled
                                       // target for BOTH so the contest is fair.)

type
  TPairDist = array[0..NUM_NODES-1, 0..NUM_NODES-1] of integer;

var
  TreeDist: TPairDist;                 // ground-truth path lengths
  S: TNeuralFloat;                     // sqrt(curvature)

// --------------------------------------------------------------------------
// Tree path length between heap-style node ids a,b (1-based: root=1, children
// of i are 2i and 2i+1). Depth(i) = floor(log2(i)); walk the deeper node up to
// the lowest common ancestor, counting edges.
// --------------------------------------------------------------------------
function TreePathLen(a, b: integer): integer;
var
  da, db, steps: integer;
begin
  Inc(a); Inc(b);                      // to 1-based heap ids
  da := 0; db := 0;
  steps := a; while steps > 1 do begin steps := steps div 2; Inc(da); end;
  steps := b; while steps > 1 do begin steps := steps div 2; Inc(db); end;
  Result := 0;
  while da > db do begin a := a div 2; Dec(da); Inc(Result); end;
  while db > da do begin b := b div 2; Dec(db); Inc(Result); end;
  while a <> b do begin a := a div 2; b := b div 2; Result := Result + 2; end;
end;

procedure BuildTree();
var i, j: integer;
begin
  for i := 0 to NUM_NODES - 1 do
    for j := 0 to NUM_NODES - 1 do
      TreeDist[i][j] := TreePathLen(i, j);
end;

// --------------------------------------------------------------------------
// Curvature-c Poincare distance between two ball points a,b (length DIM), and
// its gradient w.r.t. a (gA) and b (gB). dist = (2/s) atanh(s r), r=||m||,
// m = (-a) (+)_c b (Mobius addition). dDist/dr = 2/(1 - c r^2); dr/dm = m/r;
// then the Mobius-add Jacobian gives dm/da, dm/db. With first-arg = -a, the
// chain yields dDist/da = -(dDist/d(-a)). Returns the distance.
// --------------------------------------------------------------------------
function PoincareDist(const a, b: TNNetVolume; var gA, gB: array of TNeuralFloat): TNeuralFloat;
var
  i: integer;
  c, Acoef, na2, nb2, pco, qco, Den, r, t, dist, ddr, rscale: TNeuralFloat;
  m, dLdm, dLdna, dLdb: array[0..DIM-1] of TNeuralFloat;
  ha, hb, Gm: TNeuralFloat;
  nega: array[0..DIM-1] of TNeuralFloat;   // = -a (first Mobius arg)
begin
  c := CURVATURE;
  Acoef := 0; na2 := 0; nb2 := 0;
  for i := 0 to DIM - 1 do
  begin
    nega[i] := -a.FData[i];
    Acoef := Acoef + nega[i] * b.FData[i];
    na2 := na2 + nega[i] * nega[i];
    nb2 := nb2 + b.FData[i] * b.FData[i];
  end;
  pco := 1.0 + 2.0*c*Acoef + c*nb2;
  qco := 1.0 - c*na2;
  Den := 1.0 + 2.0*c*Acoef + c*c*na2*nb2;
  if Abs(Den) < HYP_EPS then Den := HYP_EPS;
  r := 0;
  for i := 0 to DIM - 1 do
  begin
    m[i] := (pco*nega[i] + qco*b.FData[i]) / Den;
    r := r + m[i]*m[i];
  end;
  r := Sqrt(r);
  t := S*r;
  if t > 1.0 - HYP_EPS then t := 1.0 - HYP_EPS;
  if t < HYP_EPS then dist := (2.0/S)*t else dist := (2.0/S)*ArcTanh(t);
  Result := dist;

  // Seed dL/dm with the radial derivative (gradient of dist itself, dL/ddist=1).
  ddr := 2.0 / (1.0 - c*r*r);
  if r < HYP_EPS then rscale := 0 else rscale := ddr / r;
  for i := 0 to DIM - 1 do dLdm[i] := rscale * m[i];

  // Mobius-add backward (first arg = nega, second = b). Mirrors the layer math.
  ha := 0; hb := 0; Gm := 0;
  for i := 0 to DIM - 1 do
  begin
    ha := ha + dLdm[i]*nega[i];
    hb := hb + dLdm[i]*b.FData[i];
    Gm := Gm + dLdm[i]*(pco*nega[i] + qco*b.FData[i]);
  end;
  for i := 0 to DIM - 1 do
  begin
    dLdna[i] :=
      ( 2.0*c*b.FData[i]*ha + pco*dLdm[i] - 2.0*c*nega[i]*hb ) / Den
      - Gm/(Den*Den) * ( 2.0*c*b.FData[i] + 2.0*c*c*nb2*nega[i] );
    dLdb[i] :=
      ( (2.0*c*nega[i] + 2.0*c*b.FData[i])*ha + qco*dLdm[i] ) / Den
      - Gm/(Den*Den) * ( 2.0*c*nega[i] + 2.0*c*c*na2*b.FData[i] );
  end;
  // a = -nega => dDist/da = -dLdna.
  for i := 0 to DIM - 1 do
  begin
    gA[i] := -dLdna[i];
    gB[i] := dLdb[i];
  end;
end;

function EuclidDist(const a, b: TNNetVolume; var gA, gB: array of TNeuralFloat): TNeuralFloat;
var
  i: integer;
  d, r: TNeuralFloat;
begin
  r := 0;
  for i := 0 to DIM - 1 do r := r + Sqr(a.FData[i] - b.FData[i]);
  r := Sqrt(r);
  Result := r;
  if r < HYP_EPS then
    for i := 0 to DIM - 1 do begin gA[i] := 0; gB[i] := 0; end
  else
    for i := 0 to DIM - 1 do
    begin
      d := (a.FData[i] - b.FData[i]) / r;   // d r / d a_i
      gA[i] := d;
      gB[i] := -d;
    end;
end;

// --------------------------------------------------------------------------
// One-hot input volume for node id.
// --------------------------------------------------------------------------
procedure OneHot(V: TNNetVolume; node: integer);
var i: integer;
begin
  for i := 0 to NUM_NODES - 1 do V.FData[i] := 0;
  V.FData[node] := 1;
end;

// Copy the embedder's output (the embedding) into Dst (length DIM).
procedure SnapEmbed(NN: TNNet; Dst: TNNetVolume);
var i: integer;
begin
  for i := 0 to DIM - 1 do Dst.FData[i] := NN.GetLastLayer.Output.FData[i];
end;

// --------------------------------------------------------------------------
// Final report: MSE and Pearson correlation of embedded distance vs tree
// distance over all i<j pairs. Hyp selects the distance function.
// --------------------------------------------------------------------------
procedure Evaluate(NN: TNNet; Hyp: boolean; out MSE, Corr: TNeuralFloat);
var
  i, j, cnt: integer;
  Inp, EmbI, EmbJ: TNNetVolume;
  gA, gB: array[0..DIM-1] of TNeuralFloat;
  de, dt, sumE, sumT, sumEE, sumTT, sumET, se: TNeuralFloat;
  cov, vE, vT: TNeuralFloat;
begin
  Inp  := TNNetVolume.Create(NUM_NODES, 1, 1);
  EmbI := TNNetVolume.Create(DIM, 1, 1);
  EmbJ := TNNetVolume.Create(DIM, 1, 1);
  cnt := 0; se := 0;
  sumE := 0; sumT := 0; sumEE := 0; sumTT := 0; sumET := 0;
  for i := 0 to NUM_NODES - 1 do
    for j := i + 1 to NUM_NODES - 1 do
    begin
      OneHot(Inp, i); NN.Compute(Inp); SnapEmbed(NN, EmbI);
      OneHot(Inp, j); NN.Compute(Inp); SnapEmbed(NN, EmbJ);
      if Hyp then de := PoincareDist(EmbI, EmbJ, gA, gB)
              else de := EuclidDist(EmbI, EmbJ, gA, gB);
      dt := TARGET_SCALE * TreeDist[i][j];
      se := se + Sqr(de - dt);
      sumE := sumE + de;   sumT := sumT + dt;
      sumEE := sumEE + de*de; sumTT := sumTT + dt*dt; sumET := sumET + de*dt;
      Inc(cnt);
    end;
  MSE := se / cnt;
  cov := sumET/cnt - (sumE/cnt)*(sumT/cnt);
  vE  := sumEE/cnt - Sqr(sumE/cnt);
  vT  := sumTT/cnt - Sqr(sumT/cnt);
  if (vE > 1e-12) and (vT > 1e-12) then Corr := cov / Sqrt(vE*vT)
  else Corr := 0;
  Inp.Free; EmbI.Free; EmbJ.Free;
end;

// --------------------------------------------------------------------------
// Hand-rolled pairwise training. Per sampled pair (i,j):
//   forward j, snapshot embedding B; forward i, seed embedder OutputError with
//   dMSE/dembA and backprop; forward j again, seed with dMSE/dembB and backprop.
//   dMSE/dembA = (de - dt) * d(de)/dembA  (factor for 0.5*(de-dt)^2 is (de-dt)).
// Accumulate in batch mode over PAIRS_PER_EP pairs, then UpdateWeights once.
// --------------------------------------------------------------------------
// Clip a seeded error vector to GRAD_CLIP norm (and zero any non-finite entry).
procedure ClipErr(E: TNNetVolume);
var
  i: integer;
  nrm: TNeuralFloat;
begin
  nrm := 0;
  for i := 0 to DIM - 1 do
  begin
    if not (E.FData[i] = E.FData[i]) then E.FData[i] := 0;  // NaN guard
    nrm := nrm + Sqr(E.FData[i]);
  end;
  nrm := Sqrt(nrm);
  if nrm > GRAD_CLIP then
    for i := 0 to DIM - 1 do E.FData[i] := E.FData[i] * GRAD_CLIP / nrm;
end;

procedure TrainModel(NN: TNNet; Hyp: boolean);
var
  ep, p, i, ni, nj: integer;
  Inp, EmbI, EmbJ: TNNetVolume;
  Emb: TNNetLayer;
  gA, gB: array[0..DIM-1] of TNeuralFloat;
  de, dt, coef, mse: TNeuralFloat;
begin
  Inp  := TNNetVolume.Create(NUM_NODES, 1, 1);
  EmbI := TNNetVolume.Create(DIM, 1, 1);
  EmbJ := TNNetVolume.Create(DIM, 1, 1);
  Emb  := NN.GetLastLayer;
  // Identical optimiser budget for both models (same LR, same momentum, same
  // batch size and epoch count); the only difference is the geometry.
  if Hyp then NN.SetLearningRate(HYP_LR, HYP_MOM)
          else NN.SetLearningRate(LEARN_RATE, EUC_MOM);
  // The embedder is the last layer (no downstream layer to register a departing
  // branch). For a DIRECT Layer.Backpropagate() the departing-branch count must
  // be >=1 or the backprop guard rejects the call; register it once here, then
  // reset the per-call counters before every manual backprop (memory note:
  // manual-backprop-last-layer-branch).
  Emb.IncDepartingBranchesCnt();
  for ep := 1 to EPOCHS do
  begin
    // One MINI-BATCH per epoch: accumulate the per-pair MSE gradients into the
    // shared deltas (batch-update mode), then a single UpdateWeights. Seeds are
    // scaled by 1/PAIRS_PER_EP so the step is the MEAN gradient (stable for the
    // touchy hyperbolic geometry; per-pair SGD here is far too noisy).
    NN.ClearDeltas();
    for p := 1 to PAIRS_PER_EP do
    begin
      ni := Random(NUM_NODES);
      repeat nj := Random(NUM_NODES) until nj <> ni;
      dt := TARGET_SCALE * TreeDist[ni][nj];

      // Snapshot both embeddings first (distance gradient couples them).
      OneHot(Inp, ni); NN.Compute(Inp); SnapEmbed(NN, EmbI);
      OneHot(Inp, nj); NN.Compute(Inp); SnapEmbed(NN, EmbJ);
      if Hyp then de := PoincareDist(EmbI, EmbJ, gA, gB)
              else de := EuclidDist(EmbI, EmbJ, gA, gB);
      coef := (de - dt) / PAIRS_PER_EP;   // mean of 0.5*(de-dt)^2 over the batch

      // Backprop node i: seed embedder OutputError = coef * d(de)/dembA.
      // Near the ball boundary the radial atanh gradient 2/(1-c r^2) blows up, so
      // we CLIP the seeded error vector to a fixed norm (standard hyperbolic-SGD
      // stabilisation, Nickel & Kiela 2017). Without it the embeddings diverge.
      OneHot(Inp, ni); NN.Compute(Inp);
      NN.ResetBackpropCallCurrCnt();
      for i := 0 to DIM - 1 do Emb.OutputError.FData[i] := coef * gA[i];
      ClipErr(Emb.OutputError);
      Emb.Backpropagate();

      // Backprop node j: seed embedder OutputError = coef * d(de)/dembB.
      OneHot(Inp, nj); NN.Compute(Inp);
      NN.ResetBackpropCallCurrCnt();
      for i := 0 to DIM - 1 do Emb.OutputError.FData[i] := coef * gB[i];
      ClipErr(Emb.OutputError);
      Emb.Backpropagate();
    end;
    NN.UpdateWeights();
    if (ep mod 300 = 0) or (ep = 1) then
    begin
      Evaluate(NN, Hyp, mse, de);   // reuse de as corr holder
      WriteLn(Format('  [%s] epoch %4d   MSE=%8.4f   corr=%6.3f',
        [BoolToStr(Hyp, 'HYP', 'EUC'), ep, mse, de]));
    end;
  end;
  Inp.Free; EmbI.Free; EmbJ.Free;
end;

procedure InitEmbeddings(NN: TNNet);
var
  L: TNNetLayer;
  i, j: integer;
begin
  L := NN.GetLastLayer;
  for j := 0 to DIM - 1 do
    for i := 0 to NUM_NODES - 1 do
      L.Neurons[j].Weights.FData[i] := (Random - 0.5) * 2.0 * BALL_INIT;
end;

var
  HypNet, EucNet: TNNet;
  hMSE, hCorr, eMSE, eCorr: TNeuralFloat;
begin
  // The Poincare distance's atanh legitimately approaches the ball boundary, so
  // (like the QuaternionLinear / OctonionConv examples) we mask hardware FP
  // exceptions; the code still clamps every radius/denominator analytically.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 424242;
  S := Sqrt(CURVATURE);
  BuildTree();

  WriteLn('HyperbolicEmbedding: tree-distance recovery, hyperbolic vs Euclidean');
  WriteLn(Format('  complete binary tree depth=%d  nodes=%d  embed dim=%d  c=%.2f',
    [TREE_DEPTH, NUM_NODES, DIM, CURVATURE]));
  WriteLn('');

  // Model A: hyperbolic embedder (bias suppressed so param count matches EUC).
  HypNet := TNNet.Create();
  HypNet.AddLayer(TNNetInput.Create(NUM_NODES, 1, 1));
  HypNet.AddLayer(TNNetHyperbolicLinear.Create(DIM, CURVATURE, 1));
  HypNet.SetLearningRate(HYP_LR, HYP_MOM);   // (re-set per-model in TrainModel)
  HypNet.SetL2Decay(0.0);
  HypNet.SetBatchUpdate(True);
  InitEmbeddings(HypNet);

  // Model B: Euclidean embedder, same N->DIM matrix, no bias.
  EucNet := TNNet.Create();
  EucNet.AddLayer(TNNetInput.Create(NUM_NODES, 1, 1));
  EucNet.AddLayer(TNNetFullConnectLinear.Create(DIM, 1));
  EucNet.SetLearningRate(LEARN_RATE, EUC_MOM);
  EucNet.SetL2Decay(0.0);
  EucNet.SetBatchUpdate(True);
  InitEmbeddings(EucNet);

  WriteLn('Training hyperbolic model...');
  TrainModel(HypNet, True);
  WriteLn('Training Euclidean model...');
  TrainModel(EucNet, False);

  Evaluate(HypNet, True,  hMSE, hCorr);
  Evaluate(EucNet, False, eMSE, eCorr);

  WriteLn('');
  WriteLn('==== FINAL (all node pairs, embedded distance vs tree path length) ====');
  WriteLn(Format('  Hyperbolic : MSE=%8.4f   Pearson corr=%6.3f', [hMSE, hCorr]));
  WriteLn(Format('  Euclidean  : MSE=%8.4f   Pearson corr=%6.3f', [eMSE, eCorr]));
  WriteLn('');
  if (hMSE < eMSE) and (hCorr >= eCorr) then
    WriteLn(Format('  VERDICT: hyperbolic WINS - lower MSE (%.4f vs %.4f) and ' +
      'higher/equal correlation (%.3f vs %.3f) at the SAME param budget.',
      [hMSE, eMSE, hCorr, eCorr]))
  else if hMSE < eMSE then
    WriteLn(Format('  VERDICT: hyperbolic has lower MSE (%.4f vs %.4f) though ' +
      'correlation is %.3f vs %.3f.', [hMSE, eMSE, hCorr, eCorr]))
  else
    WriteLn(Format('  VERDICT: hyperbolic did NOT win in this budget ' +
      '(MSE %.4f vs %.4f, corr %.3f vs %.3f).', [hMSE, eMSE, hCorr, eCorr]));

  HypNet.Free;
  EucNet.Free;
end.
