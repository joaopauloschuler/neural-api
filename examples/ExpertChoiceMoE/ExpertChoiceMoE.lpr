program ExpertChoiceMoE;
(*
ExpertChoiceMoE: Expert-Choice routing (Zhou et al. 2022, "Mixture-of-Experts
with Expert Choice Routing") contrasted with the token-choice top-k router, at
MATCHED capacity, showing that expert-choice gives a UNIFORM per-expert load BY
CONSTRUCTION while token-choice can be lopsided (and needs a load-balancing aux
loss to recover).

BACKGROUND. A sparse Mixture-of-Experts block routes tokens to a subset of
NumExperts parallel expert MLPs. There are two dual ways to pick the routing
from the (SeqLen x NumExperts) gate-score matrix:

  * TOKEN-CHOICE (Switch / Shazeer; TNNet.AddTopKMixtureOfExperts): each TOKEN
    keeps its top-k EXPERTS (argmax along the expert axis). Nothing constrains
    how many tokens land on a given expert, so the load can collapse onto a few
    experts -- the classic reason a Switch-style load-balancing AUX LOSS
    (TNNetLoadBalanceLoss) is bolted on.

  * EXPERT-CHOICE (Zhou et al. 2022; TNNet.AddExpertChoiceMixtureOfExperts):
    the TRANSPOSE -- each EXPERT keeps its top-Capacity TOKENS (argmax along the
    token axis). Every expert therefore processes EXACTLY Capacity tokens: load
    balance is STRUCTURAL, so NO aux loss is needed. A token may be picked by 0,
    1, or several experts.

THIS DEMO builds one shared gate-score matrix G (SeqLen tokens x NumExperts
experts, each row a per-token softmax over experts) and applies BOTH selection
rules at matched per-expert capacity:
  - token-choice top-1: each token -> its single best expert; we then COUNT how
    many tokens each expert received (this count is unconstrained).
  - expert-choice Capacity=SeqLen/NumExperts: each expert -> its Capacity best
    tokens; the count is Capacity for every expert, by construction.

We make the gate scores deliberately SKEWED (most tokens prefer expert 0) so the
token-choice load is visibly lopsided, then print the per-expert token counts
side by side. The expert-choice selection runs through the real
TNNetExpertChoiceGate layer (the same code path the builder uses), so the demo
also exercises the layer end-to-end. Pure CPU, finishes in well under a second.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork, neuralvolume;

const
  cSeqLen   = 12;          // tokens
  cExperts  = 3;           // experts
  cCapacity = cSeqLen div cExperts;   // 4 tokens per expert (expert-choice)

var
  Gate: TNNetVolume;       // (SeqLen, 1, NumExperts) per-token softmax gate
  TokenChoiceCnt: array[0..cExperts - 1] of integer;
  ExpertChoiceCnt: array[0..cExperts - 1] of integer;
  ECNet: TNNet;
  ECGateOut: TNNetVolume;
  t, e, BestE: integer;
  BestV, Sm: TNeuralFloat;
  Logit: array[0..cExperts - 1] of TNeuralFloat;

procedure PrintCounts(const Title: string; const Cnt: array of integer);
var
  i, Total, MinC, MaxC: integer;
begin
  WriteLn(Title);
  Write('    per-expert token count: [');
  Total := 0; MinC := MaxInt; MaxC := -1;
  for i := 0 to cExperts - 1 do
  begin
    Write(Cnt[i]:3);
    Total := Total + Cnt[i];
    if Cnt[i] < MinC then MinC := Cnt[i];
    if Cnt[i] > MaxC then MaxC := Cnt[i];
  end;
  WriteLn(' ]   total=', Total, '  min=', MinC, '  max=', MaxC,
          '  spread(max-min)=', MaxC - MinC);
end;

begin
  Randomize;
  RandSeed := 42;   // reproducible skewed gate

  // --- Build a deliberately SKEWED per-token softmax gate matrix -----------
  // Most tokens strongly prefer expert 0 (large logit on channel 0), so the
  // token-choice (argmax-over-experts) load will pile onto expert 0.
  Gate := TNNetVolume.Create(cSeqLen, 1, cExperts);
  for t := 0 to cSeqLen - 1 do
  begin
    Logit[0] := 2.0 + Random;            // expert 0 dominant for most tokens
    Logit[1] := Random;
    Logit[2] := Random;
    // A few tokens (every 5th) genuinely prefer experts 1/2 instead.
    if (t mod 5) = 1 then Logit[1] := 3.0 + Random;
    if (t mod 5) = 3 then Logit[2] := 3.0 + Random;
    // Softmax over experts -> per-token distribution (matches the upstream
    // SoftMax that AddExpertChoice/AddTopKMixtureOfExperts build).
    Sm := 0;
    for e := 0 to cExperts - 1 do
    begin
      Logit[e] := Exp(Logit[e]);
      Sm := Sm + Logit[e];
    end;
    for e := 0 to cExperts - 1 do
      Gate[t, 0, e] := Logit[e] / Sm;
  end;

  // --- TOKEN-CHOICE top-1: each token picks its single best expert ---------
  for e := 0 to cExperts - 1 do TokenChoiceCnt[e] := 0;
  for t := 0 to cSeqLen - 1 do
  begin
    BestE := 0; BestV := Gate[t, 0, 0];
    for e := 1 to cExperts - 1 do
      if Gate[t, 0, e] > BestV then begin BestV := Gate[t, 0, e]; BestE := e; end;
    Inc(TokenChoiceCnt[BestE]);
  end;

  // --- EXPERT-CHOICE Capacity: each expert picks its best Capacity tokens --
  // Run it through the real TNNetExpertChoiceGate layer so we exercise the
  // production code path; then count non-zero survivors per expert channel.
  ECNet := TNNet.Create();
  ECNet.AddLayer(TNNetInput.Create(cSeqLen, 1, cExperts, 1));
  ECNet.AddLayer(TNNetExpertChoiceGate.Create(cCapacity));
  ECNet.Compute(Gate);
  ECGateOut := ECNet.GetLastLayer.Output;
  for e := 0 to cExperts - 1 do ExpertChoiceCnt[e] := 0;
  for e := 0 to cExperts - 1 do
    for t := 0 to cSeqLen - 1 do
      if ECGateOut[t, 0, e] <> 0 then Inc(ExpertChoiceCnt[e]);

  // --- Report --------------------------------------------------------------
  WriteLn('Expert-Choice vs Token-Choice MoE load balance');
  WriteLn('  SeqLen=', cSeqLen, '  NumExperts=', cExperts,
          '  Capacity(expert-choice)=', cCapacity);
  WriteLn;
  PrintCounts('TOKEN-CHOICE top-1 (each token -> best expert):', TokenChoiceCnt);
  WriteLn('    -> load is UNCONSTRAINED and lopsided; needs a load-balance aux loss.');
  WriteLn;
  PrintCounts('EXPERT-CHOICE Capacity (each expert -> best Capacity tokens):',
              ExpertChoiceCnt);
  WriteLn('    -> every expert processes EXACTLY Capacity tokens, BY CONSTRUCTION.');
  WriteLn('       Uniform load with NO aux loss (spread = 0).');

  ECNet.Free;
  Gate.Free;
end.
