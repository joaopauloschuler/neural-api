program TopKMoE;
(*
TopKMoE: hard top-k Mixture-of-Experts routing WITH vs WITHOUT a load-balancing
auxiliary loss, showing the aux loss prevents expert collapse.

BACKGROUND. A sparse Mixture-of-Experts feed-forward block (Shazeer et al. 2017;
Switch Transformer, Fedus et al. 2021) routes each TOKEN to only the k
highest-gated of NumExperts parallel expert MLPs (sparse dispatch), instead of
blending all of them. A small gating network produces a softmax over experts;
TNNet.AddTopKMixtureOfExperts keeps the top-k gate weights (TNNetTopKGate),
renormalizes the survivors to sum to 1, and combines only those experts.

THE COLLAPSE PROBLEM. With nothing stopping it, the router happily learns to
send (almost) every token to the SAME one or two experts: those experts get all
the gradient, improve fastest, attract even more tokens, and the rest of the
capacity is wasted. The fix is a LOAD-BALANCING auxiliary loss
(TNNetLoadBalanceLoss) added as a second output head on the gate distribution:

    L_aux = coeff * E * sum_i ( f_i * P_i )

where E = NumExperts, f_i = fraction of tokens whose top-k routing touches
expert i, and P_i = mean gate probability for expert i (over the tokens of the
sample). L_aux is minimized when the load is uniform, so its gradient pushes the
router to SPREAD tokens across all experts. (f_i is a non-differentiable hard
count, so it is treated as a stop-gradient constant; the gradient flows only
through P_i: dL_aux/dg_t[i] = coeff*E*f_i/T.)

THE TASK. A tiny synthetic per-token regression with NumExperts latent GROUPS.
Each "token" is a random vector in R^d_model carrying its group id in channel 0;
its target is a group-specific linear map GroupW[g] of the token. The FOUR
groups need FOUR DIFFERENT maps, so the natural solution dedicates one expert
per group and a HEALTHY router spreads its load roughly uniformly. We seed the
gate biased toward expert 0 so both runs START collapsed onto E0; only the aux
loss can push tokens onto the idle experts (which it MUST, to fit the other
groups).

THE COMPARISON. Two IDENTICAL networks are trained on the IDENTICAL data with
the IDENTICAL hand-rolled SGD loop, differing ONLY in the aux-loss weight:

  (A) WITHOUT aux loss  (coeff = 0)   -> router free to collapse
  (B) WITH    aux loss  (coeff > 0)   -> router pushed toward balance

At the end we feed a held-out batch through each model and print the per-expert
TOKEN LOAD (how many tokens picked each expert into its top-k set) plus a
load-imbalance ratio max/mean. HEADLINE: model (B)'s load is near-uniform while
model (A)'s is lopsided -- the aux loss prevented expert collapse.

Pure CPU, single thread, tiny dims (a few experts/tokens/dims), runtime well
under a minute.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cDModel    = 6;     // token feature dim (= d_model)
  cExperts   = 4;     // NumExperts
  cHidden    = 8;     // expert hidden width
  cTopCnt    = 1;     // hard top-k (route each token to its single best expert)
  cTokens    = 8;     // tokens (sequence positions) per sample
  cSteps     = 2500;  // SGD steps
  cBatch     = 16;    // samples per step
  cLR        = 0.02;
  cMomentum  = 0.9;
  cAuxCoeff  = 0.05;  // load-balancing weight for model (B)
  cSeed      = 424242;

const
  cGroups = cExperts;     // one latent group per expert

var
  // Group-specific target maps: GroupW[g] is a distinct (cDModel x cDModel)
  // linear map. The first feature channel of a token encodes its group g
  // (= sign/bucket of x0); the target is GroupW[g] * token. Because the FOUR
  // groups need FOUR DIFFERENT maps, a router that collapses onto one expert
  // can only fit ONE group -- so a healthy solution MUST spread tokens across
  // all experts (one per group). This is the regime where the load-balancing
  // aux loss both prevents collapse AND is required for low error.
  GroupW: array[0..cGroups-1, 0..cDModel-1, 0..cDModel-1] of TNeuralFloat;

// ---------------------------------------------------------------------------
// Synthetic data: a sample is cTokens tokens. Each token gets a random group g
// in 0..cGroups-1; its group is made decodable by writing g into channel 0
// (so the gate CAN learn to route by group), the rest of the vector is random,
// and the target is GroupW[g] * token.
// Input  tensor: (cTokens, 1, cDModel)
// Target tensor: (cTokens, 1, cDModel)
// ---------------------------------------------------------------------------
procedure DrawSample(Input, Target: TNNetVolume);
var
  t, g, i, j: integer;
  acc: TNeuralFloat;
begin
  for t := 0 to cTokens - 1 do
  begin
    g := Random(cGroups);
    for i := 0 to cDModel - 1 do
      Input[t, 0, i] := Random() * 2 - 1;
    // Encode the group in channel 0 so the gate has a learnable routing signal.
    Input[t, 0, 0] := (g + 0.5) / cGroups * 2 - 1;
    for i := 0 to cDModel - 1 do
    begin
      acc := 0;
      for j := 0 to cDModel - 1 do
        acc := acc + GroupW[g, i, j] * Input[t, 0, j];
      Target[t, 0, i] := acc;
    end;
  end;
end;

procedure InitGroupMaps();
var
  g, i, j: integer;
begin
  for g := 0 to cGroups - 1 do
    for i := 0 to cDModel - 1 do
      for j := 0 to cDModel - 1 do
        GroupW[g, i, j] := (Random() * 2 - 1) * 0.8;
end;

// ---------------------------------------------------------------------------
// Build a top-k MoE network. UseAux selects model (B) (coeff>0) vs (A) (0).
// Returns the net; outputs the block-output leaf, the aux head, and the hard
// TNNetTopKGate layer (for reading per-expert load).
// ---------------------------------------------------------------------------
procedure BuildNet(UseAux: boolean; out NN: TNNet;
  out MainLeaf, AuxHead, GateTopK: TNNetLayer);
var
  i: integer;
  Coeff: TNeuralFloat;
  GateConv: TNNetLayer;
begin
  if UseAux then Coeff := cAuxCoeff else Coeff := 0.0;
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cTokens, 1, cDModel, 1));
  MainLeaf := NN.AddTopKMixtureOfExperts(nil, cExperts, cHidden, cTopCnt,
    AuxHead, Coeff);
  // Locate the hard top-k gate layer to read per-expert load later.
  GateTopK := nil;
  for i := 0 to NN.CountLayers - 1 do
    if NN.Layers[i] is TNNetTopKGate then GateTopK := NN.Layers[i];
  // Seed a router that STARTS collapsed: bias the gate-logit projection
  // (TopKGate <- SoftMax <- gate conv) toward expert 0, so at init most tokens
  // route to E0. Because the shared task is solvable by E0 ALONE, an
  // unregularized router (model A) is happy to leave the load there; the aux
  // loss (model B) must actively pull tokens onto the idle experts. This makes
  // collapse-vs-balance the visible difference between the two runs.
  // Keep the random input weights (so the gate CAN learn to route by the
  // group signal in channel 0); only add a bias toward E0 to start collapsed.
  GateConv := GateTopK.PrevLayer.PrevLayer;   // SoftMax.PrevLayer = gate conv
  if (GateConv <> nil) and (GateConv.Neurons.Count = cExperts) then
    for i := 0 to cExperts - 1 do
      if i = 0 then GateConv.Neurons[i].BiasWeight := 3.0   // favour E0 at init
      else GateConv.Neurons[i].BiasWeight := 0.0;
  NN.SetLearningRate(cLR, {Momentum=}cMomentum);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);          // accumulate grads; UpdateWeights per batch
  // We hand-roll the two-leaf backward, so set BOTH leaves' departing-branch
  // counts once. Neither is reached as a normal child of a later layer, so the
  // framework would leave them at 0 and TestBackPropCallCurrCnt would mis-fire
  // ("Too many backprop calls ... Should be:0, Got:1").
  MainLeaf.IncDepartingBranchesCnt();
  AuxHead.IncDepartingBranchesCnt();
end;

// ---------------------------------------------------------------------------
// Read the per-expert TOKEN LOAD over NumProbe random samples: for each token,
// count which experts survive the hard top-k gate (nonzero renormalized weight).
// ---------------------------------------------------------------------------
procedure MeasureLoad(NN: TNNet; GateTopK: TNNetLayer; NumProbe: integer;
  out Load: array of integer);
var
  s, t, e: integer;
  Input, Target: TNNetVolume;
begin
  for e := 0 to cExperts - 1 do Load[e] := 0;
  Input := TNNetVolume.Create(cTokens, 1, cDModel);
  Target := TNNetVolume.Create(cTokens, 1, cDModel);
  try
    for s := 1 to NumProbe do
    begin
      DrawSample(Input, Target);
      NN.Compute(Input);
      for t := 0 to cTokens - 1 do
        for e := 0 to cExperts - 1 do
          if GateTopK.Output[t, 0, e] <> 0 then Inc(Load[e]);
    end;
  finally
    Input.Free;
    Target.Free;
  end;
end;

procedure ReportLoad(const Title: string; const Load: array of integer);
var
  e, Total, MaxL: integer;
  Mean, Ratio: TNeuralFloat;
  Line: string;
begin
  Total := 0;
  MaxL := 0;
  for e := 0 to cExperts - 1 do
  begin
    Total := Total + Load[e];
    if Load[e] > MaxL then MaxL := Load[e];
  end;
  if Total = 0 then Total := 1;
  Mean := Total / cExperts;
  if Mean = 0 then Mean := 1;
  Ratio := MaxL / Mean;
  Line := '';
  for e := 0 to cExperts - 1 do
    Line := Line + Format('  E%d=%4d (%5.1f%%)', [e, Load[e],
      100.0 * Load[e] / Total]);
  WriteLn(Title);
  WriteLn(Line);
  WriteLn(Format('    imbalance max/mean = %.2f  (1.00 = perfectly uniform)',
    [Ratio]));
  WriteLn('');
end;

// ---------------------------------------------------------------------------
// Train one model and report. Hand-rolled two-leaf backward inline (kept in a
// local procedure that closes over the concrete TNNet for clarity).
// ---------------------------------------------------------------------------
procedure RunModel(UseAux: boolean; const Tag: string);
var
  NN: TNNet;
  MainLeaf, AuxHead, GateTopK: TNNetLayer;
  Input, Target, Pseudo: TNNetVolume;
  Step, B, t, i: integer;
  Outp: TNNetVolume;
  Load: array[0..cExperts-1] of integer;
  Mse, Diff: TNeuralFloat;
begin
  RandSeed := cSeed;     // identical init + data stream for both models
  BuildNet(UseAux, NN, MainLeaf, AuxHead, GateTopK);

  Input  := TNNetVolume.Create(cTokens, 1, cDModel);
  Target := TNNetVolume.Create(cTokens, 1, cDModel);
  Pseudo := TNNetVolume.Create(cTokens, 1, cDModel);
  try
    for Step := 1 to cSteps do
    begin
      NN.ClearDeltas();
      for B := 1 to cBatch do
      begin
        DrawSample(Input, Target);
        NN.Compute(Input);
        Outp := MainLeaf.Output;
        // pseudo_i = out_i - (1/cBatch)*(out_i - target_i)
        //   -> stock error = (1/cBatch)*(out_i - target_i)  (mean MSE grad).
        for i := 0 to Outp.Size - 1 do
          Pseudo.FData[i] := Outp.FData[i] -
            (1.0 / cBatch) * (Outp.FData[i] - Target.FData[i]);

        // ---- two-leaf backward ----
        NN.ResetBackpropCallCurrCnt();
        MainLeaf.ComputeOutputErrorWith(Pseudo);
        MainLeaf.Backpropagate();   // gate receives 1 of its 2 branch calls
        AuxHead.Backpropagate();    // gate receives the 2nd -> propagates up
      end;
      NN.UpdateWeights();
    end;

    // Final retrieval MSE on a fresh batch.
    Mse := 0;
    for B := 1 to 64 do
    begin
      DrawSample(Input, Target);
      NN.Compute(Input);
      Outp := MainLeaf.Output;
      for t := 0 to Outp.Size - 1 do
      begin
        Diff := Outp.FData[t] - Target.FData[t];
        Mse := Mse + Diff * Diff;
      end;
    end;
    Mse := Mse / (64 * cTokens * cDModel);

    MeasureLoad(NN, GateTopK, {NumProbe=}200, Load);
    WriteLn('======================================================');
    WriteLn(Tag);
    WriteLn(Format('  final retrieval MSE = %.4f', [Mse]));
    ReportLoad('  per-expert token load (held-out 200 samples):', Load);
  finally
    Input.Free;
    Target.Free;
    Pseudo.Free;
    NN.Free;
  end;
end;

begin
  RandSeed := cSeed;
  InitGroupMaps();

  WriteLn('Hard top-k Mixture-of-Experts: load balancing prevents collapse');
  WriteLn(Format('NumExperts=%d  TopK=%d  d_model=%d  tokens/sample=%d  '
    + 'aux coeff(B)=%.3f', [cExperts, cTopCnt, cDModel, cTokens, cAuxCoeff]));
  WriteLn('');

  RunModel({UseAux=}False, '(A) WITHOUT load-balancing aux loss');
  RunModel({UseAux=}True,  '(B) WITH    load-balancing aux loss');

  WriteLn('Expected headline: (A) is lopsided (a few experts hog the tokens,');
  WriteLn('high max/mean), while (B) is near-uniform (max/mean close to 1).');
end.
