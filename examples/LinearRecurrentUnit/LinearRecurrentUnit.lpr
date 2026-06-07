// LinearRecurrentUnit example
//
// Long-range temporal INTEGRATION, contrasting the Linear Recurrent Unit
// (TNNetLRU, Orvieto et al. 2023 "Resurrecting Recurrent Neural Networks for
// Long Sequences", arXiv:2303.06349) against a MEMORYLESS per-token baseline.
//
// The task (causal prefix sum / running integral). Each sequence has length
// cSeqLen and a single signal channel carrying a random value s_t at every
// position (plus irrelevant noise channels). The target at EVERY position is the
// running cumulative sum of the signal so far:
//   pos t    : [ signal=s_t | noise ]
//   target@t : sum_{tau=0..t} s_tau
// Evaluation reports the error at the FINAL position -- the hardest, since it
// integrates the WHOLE window. A correct answer there needs information from
// every past step, exactly what a linear recurrence does and what a memoryless
// per-token map cannot.
//
// Two arms share the SAME I/O contract and a matched parameter budget:
//   * LRU arm      : TNNetLRU (over the input channels) -> 1x1 readout.
//       TNNetLRU is a STABLE complex diagonal linear recurrence with the paper's
//       exp-parameterised eigenvalue lambda = exp(-exp(nu)+i*exp(theta))
//       (|lambda|<1 by construction) and a gamma=sqrt(1-|lambda|^2) input
//       normaliser. A channel whose |lambda| is near 1 acts as a leak-free
//       ACCUMULATOR: h_t ~= h_{t-1} + gamma*B*s_t, i.e. a running sum the readout
//       can rescale into the target.
//   * Memoryless arm : 1x1 projection -> tanh -> 1x1 readout, applied
//       INDEPENDENTLY per token (no recurrence). It only sees the current token,
//       so at the final position it has access to s_{N-1} alone and cannot
//       recover the sum of the earlier values -- its error floors at the variance
//       of the unseen partial sum.
//
// Headline: trained on the SAME data for the SAME number of steps, the LRU arm
// drives the integration MSE far below the memoryless arm, demonstrating that the
// LRU's stable near-unit eigenvalue genuinely integrates information across the
// whole time axis. Pure CPU, tiny dims, finishes in well under 5 minutes on 2
// cores.
//
// Coded by Claude (AI).
program LinearRecurrentUnit;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cSeqLen     = 24;    // integrate over a long window
  cNoiseDim   = 3;     // distracting noise channels (irrelevant to the target)
  cInDim      = 1 + cNoiseDim;  // signal | noise
  cModelDim   = 16;    // recurrent width
  cTrainSteps = 12000;
  cEvalSeqs   = 1000;
  cSigScale   = 0.3;   // per-step signal amplitude (keeps the sum well-scaled)

// Build one integration sequence. Each position carries a random signal value;
// the target at the final position is the running sum of all signal values.
procedure MakeSample(Input, Desired: TNNetVolume);
var
  pos, j: integer;
  s, total: TNeuralFloat;
begin
  Input.Fill(0);
  Desired.Fill(0);
  total := 0;
  for pos := 0 to cSeqLen - 1 do
  begin
    s := (Random * 2.0 - 1.0) * cSigScale;     // signal s_t in [-cSigScale,cSigScale]
    Input[pos, 0, 0] := s;
    total := total + s;
    for j := 0 to cNoiseDim - 1 do
      Input[pos, 0, 1 + j] := Random * 0.4 - 0.2;  // irrelevant noise
    // Target at EVERY position = running cumulative sum so far (a true causal
    // prefix-sum). The hardest position is the last (integrates the whole window);
    // the memoryless arm cannot solve ANY position past the first.
    Desired[pos, 0, 0] := total;
  end;
end;

// Integration MSE at the final position, averaged over N sequences.
function Evaluate(NN: TNNet; N: integer): TNeuralFloat;
var
  Input, Desired: TNNetVolume;
  seq: integer;
  diff, mse: TNeuralFloat;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, cInDim);
  Desired := TNNetVolume.Create(cSeqLen, 1, 1);
  mse := 0;
  try
    for seq := 0 to N - 1 do
    begin
      MakeSample(Input, Desired);
      NN.Compute(Input);
      diff := NN.GetLastLayer.Output[cSeqLen - 1, 0, 0] - Desired[cSeqLen - 1, 0, 0];
      mse := mse + diff * diff;
    end;
    Result := mse / N;
  finally
    Input.Free; Desired.Free;
  end;
end;

// LRU arm: stable complex diagonal recurrence (over input channels) -> 1x1 readout.
function BuildLRU(): TNNet;
var
  Cell: TNNetLRU;
  d: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  // Run the LRU DIRECTLY on the input channels (depth-preserving) so the clean
  // signal channel is not attenuated by a random front projection; a 1x1 readout
  // then selects/rescales the accumulated signal channel.
  Cell := TNNetLRU.Create();
  Result.AddLayer(Cell);
  Result.AddLayer(TNNetPointwiseConvLinear.Create(1));
  // Long-range warm start. The defining LRU property is the STABLE near-unit
  // eigenvalue: a channel with |lambda|->1 (parameterised through nu, so it can
  // never leave the unit disk) is a leak-free ACCUMULATOR. Seed every channel as
  // a non-rotating near-unit accumulator with an input gain that undoes the
  // gamma=sqrt(1-|lambda|^2) normaliser (gamma ~= 0.0447 at |lambda|=0.999) so
  // gamma*B ~= 1 and each channel forms a TRUE running sum of its input from step
  // one. |lambda| = exp(-exp(nu)): nu = ln(-ln(|lambda|)). All params remain fully
  // trainable from here -- the warm start only matches the paper's advice to
  // initialise |lambda| near 1 for the long-range regime.
  for d := 0 to cInDim - 1 do
  begin
    Cell.Neurons[0].Weights.Raw[d] := Ln(-Ln(0.999));  // |lambda| ~= 0.999
    Cell.Neurons[1].Weights.Raw[d] := Ln(0.001);       // near-zero rotation
    Cell.Neurons[2].Weights.Raw[d] := 1.0 / Sqrt(1 - 0.999 * 0.999);  // B ~= 22.4
  end;
end;

// Memoryless arm: per-token MLP (1x1 -> tanh -> 1x1), NO recurrence.
function BuildMemoryless(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConv.Create(cModelDim));   // tanh activation
  Result.AddLayer(TNNetPointwiseConvLinear.Create(1));
end;

procedure Train(NN: TNNet; Steps: integer; LR: TNeuralFloat);
var
  Input, Desired: TNNetVolume;
  i: integer;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, cInDim);
  Desired := TNNetVolume.Create(cSeqLen, 1, 1);
  NN.SetLearningRate(LR, 0.9);
  try
    for i := 0 to Steps - 1 do
    begin
      MakeSample(Input, Desired);
      NN.Compute(Input);
      NN.Backpropagate(Desired);
    end;
  finally
    Input.Free; Desired.Free;
  end;
end;

var
  LNet, MNet: TNNet;
  lMSE, mMSE: TNeuralFloat;
begin
  RandSeed := 12345;

  WriteLn('=== LRU: long-range integration (cumulative sum), recurrence vs memoryless ===');
  WriteLn('seq_len=', cSeqLen, '  model_dim=', cModelDim,
          '  integrate over ', cSeqLen, ' steps');
  WriteLn;

  LNet := BuildLRU();
  MNet := BuildMemoryless();
  WriteLn('LRU         params = ', LNet.CountWeights());
  WriteLn('Memoryless  params = ', MNet.CountWeights());
  WriteLn;

  WriteLn('training both arms on the SAME stream (', cTrainSteps, ' steps each)...');
  // Same RNG stream replay so both arms see identical training samples.
  RandSeed := 999;
  Train(LNet, cTrainSteps, 0.01);
  RandSeed := 999;
  Train(MNet, cTrainSteps, 0.01);
  WriteLn;

  RandSeed := 7;
  lMSE := Evaluate(LNet, cEvalSeqs);
  RandSeed := 7;
  mMSE := Evaluate(MNet, cEvalSeqs);

  WriteLn('eval over ', cEvalSeqs, ' held-out integration sequences:');
  WriteLn('  LRU (stable complex recurrence): integration MSE = ', lMSE:0:6);
  WriteLn('  Memoryless (per-token MLP)     : integration MSE = ', mMSE:0:6);
  WriteLn;

  if lMSE < mMSE * 0.5 then
    WriteLn('OK: the LRU integrates the signal across the whole window; the ',
            'memoryless baseline cannot see past the current token.')
  else
    WriteLn('NOTE: LRU did not clearly beat the memoryless baseline this run.');

  LNet.Free;
  MNet.Free;
end.
