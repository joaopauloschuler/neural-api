// PonderNet example -- adaptive computation time with learned probabilistic halting
//
// PonderNet (Banino, Balaguer & Blundell 2021, https://arxiv.org/abs/2107.05407)
// learns HOW LONG to think per input. A weight-tied step function f is applied up
// to MaxSteps times; at each step a tiny halting head emits lambda_n in (0,1),
// giving the geometric halting distribution
//   p_n = lambda_n * prod_{k<n}(1-lambda_k),
// and the block output is the SMOOTH p_n-weighted sum of the per-step outputs (no
// hard argmax). The TNNetPonderCostLoss head adds a KL(p || geometric(prior))
// regularizer that prefers halting early -- so the model only spends extra steps
// where the task forces it.
//
// The toy task is PARITY of a variable-length bit string -- a textbook
// "harder = needs more sequential computation" problem. Each sample has a random
// number of ACTIVE leading bits L in [1..cMaxLen]; the rest are masked to zero.
// The label is the parity (XOR) of the active bits. Difficulty == L: a longer
// active prefix needs more iterative XOR steps to resolve, so an adaptive-compute
// model should ponder LONGER as L grows. (A fixed-depth net must pay the worst
// case on every input.)
//
// Network: Input -> TNNet.AddPonderNetBlock (weight-tied f x MaxSteps, shared
// halting head, running p_n accumulator) -> a 2-way SoftMax parity head on the
// p_n-weighted block output. Both losses are trained in ONE backward pass by
// DeepConcat-ing the two heads into a single output: [ parity-softmax (2 ch) |
// ponder-cost passthrough (MaxSteps ch) ]. The parity head uses cross-entropy
// (framework default), the halting branch uses TNNetPonderCostLoss.
//
// Headline (printed at the end and asserted): the network's EXPECTED number of
// ponder steps  E[n] = sum_n (n+1) * p_n  RISES monotonically with difficulty L.
//
// Inference note: this build always unrolls MaxSteps applications (static tensor
// shapes) and returns the p_n-weighted expectation; E[n] is the adaptive-depth
// signal. A true threshold-on-cumulative-p_n early-exit would need dynamic shapes
// the unrolled-graph API does not support, so it is intentionally not done -- the
// expected OUTPUT is identical, only compute is not saved.
//
// Pure CPU, tiny dims, finishes well under three minutes on 2 cores.
//
// Coded by Claude (AI).
program PonderNet;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cMaxLen     = 6;          // maximum active-bit prefix length (= difficulty)
  cInDim      = cMaxLen;    // one input channel per bit position
  cMaxSteps   = 6;          // PonderNet ponder budget
  cHidden     = 24;         // step-function hidden width
  cPrior      = 0.5;        // geometric halting prior (favours halting EARLY)
  cOutDim     = 2 + cMaxSteps;  // softmax parity (2) | halting distribution (cMaxSteps)
  cTrainSteps = 24000;
  cEvalPer    = 500;        // eval samples per difficulty bucket

// Build one parity sample: a random active-prefix length L in [1..cMaxLen],
// random bits in positions 0..L-1 (zero elsewhere). Returns L (the difficulty)
// and writes the input encoding plus the combined target.
//   Target[0..1]      = one-hot parity label for the softmax head.
//   Target[2..]       = ignored by the PonderCostLoss head (filled 0).
function MakeSample(Input, Target: TNNetVolume): integer;
var
  L, j, bit, par: integer;
begin
  Input.Fill(0);
  Target.Fill(0);
  L := 1 + Random(cMaxLen);
  par := 0;
  for j := 0 to L - 1 do
  begin
    bit := Random(2);
    // Encode bit as +1 / -1 on its own channel (0 elsewhere = "inactive").
    if bit = 1 then Input[0, 0, j] := 1.0 else Input[0, 0, j] := -1.0;
    par := par xor bit;
  end;
  Target.OneHotEncodingOnPixel(0, 0, par);  // one-hot parity (class 0 or 1)
  Result := L;
end;

// Build the PonderNet classifier. Returns the net; HaltingOut is the (1,1,cMaxSteps)
// halting distribution layer (already wired into the combined output).
function BuildNet(): TNNet;
var
  BlockOut, Halting, ParityHead, CostHead: TNNetLayer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(1, 1, cInDim));
  // Adaptive-compute core: f x cMaxSteps, weight-tied, smooth p_n-weighted output.
  BlockOut := Result.AddPonderNetBlock(nil, cMaxSteps, Halting, cHidden, cPrior);

  // Parity head on the p_n-weighted block output -> 2-way softmax.
  ParityHead := Result.AddLayerAfter(TNNetPointwiseConvLinear.Create(2), BlockOut);
  ParityHead := Result.AddLayerAfter(TNNetSoftMax.Create(), ParityHead);

  // Ponder-cost head on the halting branch (identity passthrough that rewrites its
  // gradient to the KL ponder-cost). Consuming Halting here also keeps the halting
  // branch on the backward path (no dangling branch).
  CostHead := Result.AddLayerAfter(TNNetPonderCostLoss.Create(cPrior), Halting);

  // One combined output so a SINGLE backward pass drives both losses:
  //   [ parity softmax (2) | ponder-cost passthrough (cMaxSteps) ].
  Result.AddLayer(TNNetDeepConcat.Create([ParityHead, CostHead]));
end;

// Expected number of ponder steps for the current forward pass:
//   E[n] = sum_n (n+1) * p_n   (1-based step count).
function ExpectedSteps(NN: TNNet; Halting: TNNetLayer): TNeuralFloat;
var n: integer;
begin
  Result := 0;
  for n := 0 to Halting.Output.Depth - 1 do
    Result := Result + (n + 1) * Halting.Output.Raw[n];
end;

procedure Train(NN: TNNet; Steps: integer; LR: TNeuralFloat);
var
  Input, Target: TNNetVolume;
  i: integer;
begin
  Input := TNNetVolume.Create(1, 1, cInDim);
  Target := TNNetVolume.Create(1, 1, cOutDim);
  NN.SetLearningRate(LR, 0.9);
  try
    for i := 0 to Steps - 1 do
    begin
      MakeSample(Input, Target);
      NN.Compute(Input);
      NN.Backpropagate(Target);
    end;
  finally
    Input.Free; Target.Free;
  end;
end;

var
  NN: TNNet;
  Halting: TNNetLayer;
  Input, Target: TNNetVolume;
  L, i, par, predClass, correct, total: integer;
  EStepsByL: array[1..cMaxLen] of TNeuralFloat;
  AccByL: array[1..cMaxLen] of TNeuralFloat;
  monotonic: boolean;
  prevE: TNeuralFloat;
begin
  RandSeed := 12345;

  WriteLn('=== PonderNet: adaptive computation time on variable-length parity ===');
  WriteLn('max active-bit length = ', cMaxLen, '   MaxSteps = ', cMaxSteps,
          '   hidden = ', cHidden, '   prior_lambda = ', cPrior:0:2);
  WriteLn;

  NN := BuildNet();
  // Find the halting layer reference (the cMaxSteps-wide DeepConcat feeding the
  // cost head); it is the only (1,1,cMaxSteps) layer with that exact width whose
  // class is TNNetDeepConcat sitting just before the PonderCostLoss head.
  Halting := nil;
  for i := 0 to NN.Layers.Count - 1 do
    if (NN.Layers[i] is TNNetPonderCostLoss) then
      Halting := NN.Layers[i];   // the cost head IS an identity passthrough of p
  if Halting = nil then begin WriteLn('internal: halting layer not found'); Halt(1); end;

  WriteLn('total params = ', NN.CountWeights(),
          '   (weight-tied: independent of MaxSteps)');
  WriteLn;
  WriteLn('training ', cTrainSteps, ' steps (task cross-entropy + KL ponder cost)...');
  Train(NN, cTrainSteps, 0.001);
  WriteLn('done.');
  WriteLn;

  // Evaluation: per difficulty L, measure parity accuracy and mean expected steps.
  Input := TNNetVolume.Create(1, 1, cInDim);
  Target := TNNetVolume.Create(1, 1, cOutDim);
  try
    for L := 1 to cMaxLen do
    begin
      EStepsByL[L] := 0;
      AccByL[L] := 0;
      correct := 0; total := 0;
      for i := 0 to cEvalPer - 1 do
      begin
        // Force this exact difficulty L with random bits.
        Input.Fill(0); Target.Fill(0);
        par := 0;
        for predClass := 0 to L - 1 do
        begin
          if Random(2) = 1 then begin Input[0,0,predClass] := 1.0; par := par xor 1; end
          else Input[0,0,predClass] := -1.0;
        end;
        NN.Compute(Input);
        EStepsByL[L] := EStepsByL[L] + ExpectedSteps(NN, Halting);
        // Parity head occupies output channels 0..1.
        if NN.GetLastLayer.Output.Raw[1] > NN.GetLastLayer.Output.Raw[0] then
          predClass := 1 else predClass := 0;
        if predClass = par then Inc(correct);
        Inc(total);
      end;
      EStepsByL[L] := EStepsByL[L] / cEvalPer;
      AccByL[L] := correct / total;
    end;
  finally
    Input.Free; Target.Free;
  end;

  WriteLn('difficulty L | parity acc | mean expected ponder steps E[n]');
  WriteLn('-------------+------------+-------------------------------');
  for L := 1 to cMaxLen do
    WriteLn('     ', L:2, '      |   ', (AccByL[L]*100):5:1, '%  |   ',
            EStepsByL[L]:0:3);
  WriteLn;

  // Headline check: E[n] should rise with difficulty (monotone non-decreasing,
  // with a clear net rise from the easiest to the hardest bucket).
  monotonic := true;
  prevE := EStepsByL[1];
  for L := 2 to cMaxLen do
  begin
    if EStepsByL[L] < prevE - 0.02 then monotonic := false;
    prevE := EStepsByL[L];
  end;

  WriteLn('E[n] at L=1 (easiest) = ', EStepsByL[1]:0:3,
          '   E[n] at L=', cMaxLen, ' (hardest) = ', EStepsByL[cMaxLen]:0:3);
  if monotonic and (EStepsByL[cMaxLen] > EStepsByL[1] + 0.1) then
    WriteLn('OK: expected ponder steps RISE with difficulty ',
            '(adaptive computation time).')
  else
  begin
    WriteLn('WARNING: expected steps did not rise monotonically with difficulty.');
    Halt(1);
  end;

  NN.Free;
end.
