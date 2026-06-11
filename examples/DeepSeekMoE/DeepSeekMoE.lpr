program DeepSeekMoE;
(*
DeepSeekMoE: shared + fine-grained routed experts (Dai et al. 2024,
"DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts
Language Models", arXiv:2401.06066) with the DeepSeek-V3 AUXILIARY-LOSS-FREE
bias-based load balancing (Wang et al. 2024, "Auxiliary-Loss-Free Load
Balancing Strategy for Mixture-of-Experts", arXiv:2408.15664).

THE BLOCK (TNNet.AddDeepSeekMoE) adds two ideas on top of the vanilla top-k
MoE (TNNet.AddTopKMixtureOfExperts):

  * SHARED EXPERTS are always active: their output is added unconditionally
    with no routing weight, capturing common knowledge so the ROUTED experts
    can specialize.
  * FINE-GRAINED ROUTED EXPERTS: split each conventional expert of hidden
    width H into m smaller ones (hidden H/m) and route top-(k*m) over (N*m)
    experts -- the caller expresses the split by passing a large
    NumRoutedExperts with a small ExpertHiddenDim (same FLOP budget, far more
    expert combinations per token).

THE BALANCING (TNNetBiasBalancedTopKGate): a per-expert bias b_e is added to
the router affinity FOR TOP-K SELECTION ONLY -- the combine weights still use
the unbiased scores, so balancing never distorts how much a selected expert
contributes. Once per batch the training loop calls UpdateRoutingBias():
    b_e := b_e - speed * sign(load_e - mean_load)
so overloaded experts become less selectable and starved ones more. No
Switch-style auxiliary loss, hence no interference gradient on the LM loss.

THIS DEMO builds the SAME DeepSeekMoE block twice (identical weights, gate
conv biases hand-skewed so expert 0 hoards a disproportionate token share):
  net A: BalanceBiasSpeed = 0    (plain top-k gate, no balancing)
  net B: BalanceBiasSpeed = 0.02 (DeepSeek-V3 bias balancing)
then feeds the same stationary token batch to both for several "batches",
calling UpdateRoutingBias() on net B after each, and prints the per-expert
load histograms. Net A stays skewed forever; net B converges to an exactly
uniform load. Pure CPU, finishes in seconds.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork, neuralvolume;

const
  cTokens  = 32;     // tokens per batch (SeqLen)
  cDModel  = 8;      // model width
  cShared  = 1;      // always-active shared experts
  cRouted  = 4;      // fine-grained routed experts
  cActive  = 1;      // top-k over the routed experts
  cHidden  = 4;      // routed/shared expert hidden width
  cBatches = 60;

var
  NetA, NetB: TNNet;
  GateA: TNNetLayer;                    // plain TNNetTopKGate
  GateB: TNNetBiasBalancedTopKGate;     // bias-balanced gate
  Input: TNNetVolume;
  Batch, e, x: integer;
  LoadA, LoadB: array[0..cRouted - 1] of integer;
  FirstA, FirstB, LastA, LastB: array[0..cRouted - 1] of integer;
  MeanSpreadA, MeanSpreadB: double;

function SpreadOf(const Load: array of integer): integer;
var ee, MinL, MaxL: integer;
begin
  MinL := MaxInt; MaxL := -1;
  for ee := 0 to cRouted - 1 do
  begin
    if Load[ee] < MinL then MinL := Load[ee];
    if Load[ee] > MaxL then MaxL := Load[ee];
  end;
  Result := MaxL - MinL;
end;

procedure SkewRouter(NN: TNNet);
var n: integer;
begin
  // Layer order from AddDeepSeekMoE: 0=input, 1=gate conv, 2=softmax, 3=gate.
  // Hand-skew the router so expert 0 dominates the initial routing while the
  // amplified random weights keep PER-TOKEN affinity margins diverse (so the
  // balancing bias peels tokens off expert 0 one by one instead of flipping
  // the whole batch at once). Then refresh the packed weight caches.
  for n := 0 to NN.Layers[1].Neurons.Count - 1 do
  begin
    NN.Layers[1].Neurons[n].Weights.Mul(4);
    if n = 0
      then NN.Layers[1].Neurons[n].BiasWeight := 0.75
      else NN.Layers[1].Neurons[n].BiasWeight := 0.0;
  end;
  NN.UpdateWeights();
end;

procedure CountLoads(Gate: TNNetLayer; var Load: array of integer);
var tt, ee: integer;
begin
  // Per-batch per-expert load = number of tokens whose (non-zero) top-k gate
  // weight touches the expert channel.
  for ee := 0 to cRouted - 1 do Load[ee] := 0;
  for ee := 0 to cRouted - 1 do
    for tt := 0 to cTokens - 1 do
      if Gate.Output[tt, 0, ee] <> 0 then Inc(Load[ee]);
end;

procedure PrintHistogram(const Title: string; const Load: array of integer);
var ee, MinL, MaxL: integer;
begin
  Write(Title, ' [');
  MinL := MaxInt; MaxL := -1;
  for ee := 0 to cRouted - 1 do
  begin
    Write(Load[ee]:4);
    if Load[ee] < MinL then MinL := Load[ee];
    if Load[ee] > MaxL then MaxL := Load[ee];
  end;
  WriteLn(' ]  spread(max-min)=', MaxL - MinL);
end;

begin
  // Identical weights in both nets: same seed before each construction.
  RandSeed := 42;
  NetA := TNNet.Create();
  NetA.AddLayer(TNNetInput.Create(cTokens, 1, cDModel, 1));
  NetA.AddDeepSeekMoE(nil, cShared, cRouted, cActive, cHidden, 0.0);

  RandSeed := 42;
  NetB := TNNet.Create();
  NetB.AddLayer(TNNetInput.Create(cTokens, 1, cDModel, 1));
  NetB.AddDeepSeekMoE(nil, cShared, cRouted, cActive, cHidden, 0.02);

  SkewRouter(NetA);
  SkewRouter(NetB);

  GateA := NetA.Layers[3];
  GateB := TNNetBiasBalancedTopKGate(NetB.Layers[3]);
  WriteLn('DeepSeekMoE: ', cShared, ' shared + ', cRouted,
    ' fine-grained routed experts, top-', cActive, ' routing, ',
    cTokens, ' tokens/batch');
  WriteLn('net A: plain top-k gate (', GateA.ClassName, ')');
  WriteLn('net B: aux-loss-free bias balancing (', GateB.ClassName,
    '), speed=0.02');
  WriteLn;

  // One stationary token batch (the same data each step, as with a stationary
  // token distribution): the sign-based bias update then settles into a small
  // oscillation around balance instead of chasing batch noise.
  Input := TNNetVolume.Create(cTokens, 1, cDModel);
  for x := 0 to Input.Size - 1 do Input.Raw[x] := Random() - 0.5;
  MeanSpreadA := 0; MeanSpreadB := 0;
  for Batch := 1 to cBatches do
  begin
    NetA.Compute(Input);
    NetB.Compute(Input);
    CountLoads(GateA, LoadA);
    CountLoads(GateB, LoadB);
    if Batch = 1 then
      for e := 0 to cRouted - 1 do
      begin
        FirstA[e] := LoadA[e];
        FirstB[e] := LoadB[e];
      end;
    for e := 0 to cRouted - 1 do
    begin
      LastA[e] := LoadA[e];
      LastB[e] := LoadB[e];
    end;
    if Batch > cBatches - 10 then
    begin
      MeanSpreadA := MeanSpreadA + SpreadOf(LoadA) / 10;
      MeanSpreadB := MeanSpreadB + SpreadOf(LoadB) / 10;
    end;
    // THE per-batch balancing hook (a no-op for net A, which has no bias
    // gate): nudge each expert bias against its load violation.
    GateB.UpdateRoutingBias();
  end;

  WriteLn('Per-expert token loads (', cRouted, ' routed experts; the ',
    cShared, ' shared expert always processes all ' , cTokens, ' tokens):');
  PrintHistogram('  batch  1, no balancing  :', FirstA);
  PrintHistogram('  batch  1, bias balancing:', FirstB);
  PrintHistogram('  batch ' + IntToStr(cBatches) + ', no balancing  :', LastA);
  PrintHistogram('  batch ' + IntToStr(cBatches) + ', bias balancing:', LastB);
  WriteLn;
  WriteLn('Mean spread over the last 10 batches:  no balancing=',
    MeanSpreadA:0:1, '   bias balancing=', MeanSpreadB:0:1);
  WriteLn;
  WriteLn('-> Without balancing the skewed router keeps the load piled onto');
  WriteLn('   expert 0 forever. With DeepSeek-V3 bias balancing the same');
  WriteLn('   router converges to a uniform load -- and because the bias only');
  WriteLn('   affects SELECTION (combine weights stay unbiased), no auxiliary');
  WriteLn('   loss gradient ever touches the model.');

  Input.Free;
  NetB.Free;
  NetA.Free;
end.
