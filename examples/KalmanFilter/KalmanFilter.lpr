// KalmanFilter example
//
// Tracks a 1-D constant-velocity signal (a ramp) corrupted by additive
// Gaussian measurement noise. A TNNetKalmanFilterCell learns its per-channel
// transition a and process/measurement noises Q, R end-to-end (supervised on
// the clean signal) and recovers the latent track. We contrast it against a
// parameter-matched TNNetDiagonalSSM arm at the same parameter count (both
// 1-channel sequence mixers over the time axis).
//
// Headline: filtered MSE << noisy MSE, and the learned Kalman gain g = P/(P+R)
// gives the filter a principled, uncertainty-aware blend the LTI SSM cannot
// form (the SSM has no covariance, so its denoising is a fixed linear kernel).
//
// Runs in well under a minute on 2 CPU cores. Coded by Claude (AI).
program KalmanFilter;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cSeqLen = 96;     // length of the 1-D track
  cEpochs = 1500;   // supervised denoising epochs

var
  RngState: cardinal;

// Tiny deterministic LCG so the demo is reproducible across runs/platforms.
function NextRand(): TNeuralFloat;
begin
  RngState := (RngState * 1103515245 + 12345) and $7FFFFFFF;
  Result := RngState / 2147483647.0;  // in [0,1)
end;

function NextGauss(AStd: TNeuralFloat): TNeuralFloat;
begin
  // Box-Muller.
  Result := Sqrt(-2 * Ln(NextRand() + 1e-12)) * Cos(2 * Pi * NextRand()) * AStd;
end;

procedure FillSignals(Clean, Noisy: TNNetVolume; AStd: TNeuralFloat);
var
  i: integer;
  v: TNeuralFloat;
begin
  // Constant-velocity track: x_t = x0 + vel*t (a clean ramp), wrapped through a
  // gentle sine so it stays bounded and the filter has something to track.
  for i := 0 to cSeqLen - 1 do
  begin
    v := 0.9 * Sin(i * 0.07) + 0.012 * i - 0.5;
    Clean.Raw[i] := v;
    Noisy.Raw[i] := v + NextGauss(AStd);
  end;
end;

function MSE(A, B: TNNetVolume): TNeuralFloat;
var
  i: integer;
  d: TNeuralFloat;
begin
  Result := 0;
  for i := 0 to A.Size - 1 do
  begin
    d := A.Raw[i] - B.Raw[i];
    Result := Result + d * d;
  end;
  Result := Result / A.Size;
end;

procedure TrainArm(NN: TNNet; Clean, Noisy: TNNetVolume; ALR: TNeuralFloat);
var
  epoch: integer;
begin
  NN.SetLearningRate(ALR, 0.0);
  NN.SetBatchUpdate(false);
  for epoch := 0 to cEpochs - 1 do
  begin
    NN.Compute(Noisy);
    NN.Backpropagate(Clean);
  end;
end;

var
  Clean, Noisy, OutK, OutS: TNNetVolume;
  KNN, SNN: TNNet;
  LK: TNNetKalmanFilterCell;
  LS: TNNetDiagonalSSM;
  mseNoisy, mseK, mseS: TNeuralFloat;
  noiseStd: TNeuralFloat;
begin
  // Mask FP exceptions the way the test harness does (softplus/Exp and the
  // gain ratio can momentarily underflow/produce denormals during training).
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RngState := 424242;
  noiseStd := 0.45;

  Clean := TNNetVolume.Create(cSeqLen, 1, 1);
  Noisy := TNNetVolume.Create(cSeqLen, 1, 1);
  OutK  := TNNetVolume.Create(cSeqLen, 1, 1);
  OutS  := TNNetVolume.Create(cSeqLen, 1, 1);

  FillSignals(Clean, Noisy, noiseStd);

  // --- Arm 1: TNNetKalmanFilterCell (3 params: a_raw, Q_raw, R_raw) ----------
  KNN := TNNet.Create();
  KNN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1, 1));
  LK := TNNetKalmanFilterCell.Create();
  KNN.AddLayer(LK);

  // --- Arm 2: TNNetDiagonalSSM (4 params: a_raw,b,c,e) - param-matched ~O(1) -
  SNN := TNNet.Create();
  SNN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1, 1));
  LS := TNNetDiagonalSSM.Create();
  SNN.AddLayer(LS);

  WriteLn('=== KalmanFilter example: denoising a noisy 1-D track ===');
  WriteLn('SeqLen=', cSeqLen, '  noise std=', noiseStd:0:3,
    '  epochs=', cEpochs);
  WriteLn('Kalman params=', LK.CountWeights(),
    '   DiagonalSSM params=', LS.CountWeights(), '  (param-matched, O(1))');
  WriteLn;

  // The LTI DiagonalSSM (feedthrough e at init) is more LR-sensitive than the
  // bounded-gain Kalman cell, so it gets a smaller step.
  TrainArm(KNN, Clean, Noisy, 0.03);
  TrainArm(SNN, Clean, Noisy, 0.003);

  KNN.Compute(Noisy);  OutK.Copy(KNN.GetLastLayer.Output);
  SNN.Compute(Noisy);  OutS.Copy(SNN.GetLastLayer.Output);

  mseNoisy := MSE(Noisy, Clean);
  mseK := MSE(OutK, Clean);
  mseS := MSE(OutS, Clean);

  WriteLn('MSE(noisy   , clean) = ', mseNoisy:0:5, '   (the observation)');
  WriteLn('MSE(Kalman  , clean) = ', mseK:0:5,
    '   reduction ', (100 * (1 - mseK / mseNoisy)):0:1, '%');
  WriteLn('MSE(DiagSSM , clean) = ', mseS:0:5,
    '   reduction ', (100 * (1 - mseS / mseNoisy)):0:1, '%');
  WriteLn;

  // Report the LEARNED Kalman parameters and the resulting steady-state gain.
  WriteLn('Learned Kalman params (channel 0):');
  WriteLn('  a   = tanh(a_raw)     = ', TanH(LK.Neurons[0].Weights.Raw[0]):0:4);
  WriteLn('  Q   = softplus(Q_raw) = ',
    Ln(1 + Exp(LK.Neurons[1].Weights.Raw[0])):0:4, '  (process noise)');
  WriteLn('  R   = softplus(R_raw) = ',
    Ln(1 + Exp(LK.Neurons[2].Weights.Raw[0])):0:4, '  (measurement noise)');
  WriteLn;

  if (mseK < mseNoisy) then
    WriteLn('OK: the Kalman cell recovered the latent track from the noise.')
  else
    WriteLn('WARNING: Kalman cell did not beat the noisy baseline.');

  OutS.Free;
  OutK.Free;
  Noisy.Free;
  Clean.Free;
  SNN.Free;
  KNN.Free;
end.
