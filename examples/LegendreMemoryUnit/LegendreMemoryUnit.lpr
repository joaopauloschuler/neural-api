// Legendre Memory Unit (LMU) example
//
// A continuous-delay (delay-line) reconstruction task that plays to the LMU's
// defining strength: the HiPPO-LegS Legendre Memory Unit keeps an order-N
// orthogonal-polynomial projection of a sliding WINDOW of each input channel,
// so reproducing the input from D steps ago is a fixed linear read-out of that
// window memory. A diagonal SSM (one scalar exponentially-decaying state per
// channel) has no notion of "the value exactly D steps ago" and can only
// approximate it with a blurred exponential trace -- it does markedly worse.
//
// The task. A smooth random 1-D signal x_t is streamed along the sequence axis;
// the target at every step is the signal delayed by cDelay steps:
//   target_t = x_{t - cDelay}        (0 for t < cDelay)
// Reconstructing a PURE DELAY is exactly what an orthogonal sliding-window
// memory makes trivial (read the Legendre coefficients at the corresponding
// lag) and what a leaky scalar accumulator cannot represent cleanly.
//
// Two arms share the SAME I/O contract:
//   * LMU arm : 1x1 projection to cModelDim -> TNNetLegendreMemoryUnit(order N)
//       -> 1x1 read-out. Each channel carries an N-coefficient Legendre window
//       memory; the read-out picks off the delayed value.
//   * DiagonalSSM arm : 1x1 projection to cModelDim*N (matched state budget)
//       -> TNNetDiagonalSSM (per-channel scalar leaky state) -> 1x1 read-out.
//       Same total number of recurrent state scalars as the LMU arm, but the
//       state is diagonal/scalar with no window structure.
//
// Headline: trained on the SAME data for the SAME number of steps, the LMU arm
// reaches markedly lower delayed-reconstruction MSE than the param/state-matched
// diagonal SSM. Pure CPU, tiny dims, finishes in well under a minute on 2 cores.
//
// Coded by Claude (AI).
program LegendreMemoryUnit;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cSeqLen     = 32;    // sequence length
  cDelay      = 6;     // delay (lag) to reconstruct
  cModelDim   = 4;     // memory channels
  cOrder      = 8;     // LMU memory order N (Legendre coefficients per channel)
  cTheta      = 16.0;  // LMU window length (>= delay so the lag is in-window)
  cTrainSteps = 4000;
  cEvalSeqs   = 400;

// Build one smooth random signal and its delayed-by-cDelay target.
procedure MakeSample(Input, Desired: TNNetVolume);
var
  t, h: integer;
  phase1, phase2, freq1, freq2, amp: TNeuralFloat;
begin
  Input.Fill(0);
  Desired.Fill(0);
  // A smooth signal = sum of two sinusoids with random phase/frequency.
  phase1 := Random * 2 * Pi;  phase2 := Random * 2 * Pi;
  freq1  := 0.15 + Random * 0.25;
  freq2  := 0.05 + Random * 0.15;
  amp    := 0.6 + Random * 0.3;
  for t := 0 to cSeqLen - 1 do
    Input[t, 0, 0] := amp * (Sin(freq1 * t + phase1) * 0.6 +
                             Sin(freq2 * t + phase2) * 0.4);
  // Target is the input delayed by cDelay (0 before the signal "arrives").
  for t := 0 to cSeqLen - 1 do
  begin
    h := t - cDelay;
    if h >= 0 then Desired[t, 0, 0] := Input[h, 0, 0];
  end;
end;

// Mean squared delayed-reconstruction error over the steps where the delayed
// target is defined (t >= cDelay), averaged over N sequences.
function Evaluate(NN: TNNet; N: integer): TNeuralFloat;
var
  Input, Desired: TNNetVolume;
  seq, t, cnt: integer;
  diff, mse: TNeuralFloat;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, 1);
  Desired := TNNetVolume.Create(cSeqLen, 1, 1);
  mse := 0; cnt := 0;
  try
    for seq := 0 to N - 1 do
    begin
      MakeSample(Input, Desired);
      NN.Compute(Input);
      for t := cDelay to cSeqLen - 1 do
      begin
        diff := NN.GetLastLayer.Output[t, 0, 0] - Desired[t, 0, 0];
        mse := mse + diff * diff;
        Inc(cnt);
      end;
    end;
    Result := mse / cnt;
  finally
    Input.Free; Desired.Free;
  end;
end;

// LMU arm: 1x1 projection -> Legendre window memory -> 1x1 read-out.
function BuildLMU(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim));
  Result.AddLayer(TNNetLegendreMemoryUnit.Create(cOrder, cTheta));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(1));
end;

// DiagonalSSM arm: 1x1 projection to cModelDim*cOrder (so the total number of
// recurrent state scalars matches the LMU arm) -> per-channel scalar leaky SSM
// -> 1x1 read-out.
function BuildDiagonalSSM(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cModelDim * cOrder));
  Result.AddLayer(TNNetDiagonalSSM.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(1));
end;

procedure Train(NN: TNNet; Steps: integer; LR: TNeuralFloat);
var
  Input, Desired: TNNetVolume;
  i: integer;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, 1);
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
  LNet, SNet: TNNet;
  lMSE, sMSE: TNeuralFloat;
begin
  RandSeed := 12345;

  WriteLn('=== Legendre Memory Unit: delayed-signal reconstruction ===');
  WriteLn('seq_len=', cSeqLen, '  delay=', cDelay, '  model_dim=', cModelDim,
          '  order=', cOrder, '  theta=', cTheta:0:1);
  WriteLn;

  LNet := BuildLMU();
  SNet := BuildDiagonalSSM();
  WriteLn('LMU         params = ', LNet.CountWeights());
  WriteLn('DiagonalSSM params = ', SNet.CountWeights());
  WriteLn;

  WriteLn('training both arms on the SAME delay stream (', cTrainSteps, ' steps each)...');
  RandSeed := 999;
  Train(LNet, cTrainSteps, 0.001);
  RandSeed := 999;
  Train(SNet, cTrainSteps, 0.001);
  WriteLn;

  RandSeed := 7;
  lMSE := Evaluate(LNet, cEvalSeqs);
  RandSeed := 7;
  sMSE := Evaluate(SNet, cEvalSeqs);

  WriteLn('eval over ', cEvalSeqs, ' held-out sequences (delayed-reconstruction MSE):');
  WriteLn('  LMU (Legendre window) : MSE = ', lMSE:0:6);
  WriteLn('  DiagonalSSM (scalar)  : MSE = ', sMSE:0:6);
  WriteLn;

  if lMSE < sMSE then
    WriteLn('OK: the Legendre window memory reconstructs the delay better than ',
            'the state-matched diagonal SSM.')
  else
    WriteLn('WARNING: LMU did not beat the diagonal SSM on the delay task.');

  LNet.Free; SNet.Free;
end.
