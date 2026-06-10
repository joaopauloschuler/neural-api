program SpikingMNIST;
(*
SpikingMNIST: a SPIKING neural network learns to classify rate-coded inputs
end-to-end through a SURROGATE GRADIENT, using the event-driven
leaky-integrate-and-fire layer TNNetLIFNeuron.

THE IDEA
--------
A leaky-integrate-and-fire (LIF) neuron integrates an input current into a
membrane potential over a TIME axis and emits a binary {0,1} SPIKE when the
potential crosses a threshold, then resets. The spike is a hard Heaviside, whose
derivative is zero almost everywhere, so a spiking net cannot be trained by plain
backprop. The SURROGATE GRADIENT trick (Neftci, Mostafa & Zenke 2019; Zenke &
Ganguli 2018, SuperSpike) replaces dS/dV in the backward pass with a smooth
fast-sigmoid kernel sigma'(V) = 1/(1 + alpha*|V - V_th|)^2 while keeping the
forward exactly binary. Gradients then flow backward-through-time across the
unrolled membrane recurrence and the net learns.

To stay inside a small pure-CPU budget this uses a TINY SYNTHETIC rate-coded task
(NOT real MNIST). There are NCLASS prototype patterns over DIN input features;
each prototype is a per-feature firing PROBABILITY. A sample is drawn by
rate-encoding its class prototype as T independent Bernoulli (Poisson-like) spike
trains -- input.FData[t,0,f] in {0,1}. The net must read the noisy spike trains
and recover the class.

THE NET
-------
  Input(T,1,DIN) spike trains
    -> AddSpikingBlock(HIDDEN, tau, V_th, alpha, LearnDynamics=true)
         which wires, in one call:
           PointwiseConvLinear(HIDDEN)  per-timestep linear projection (the input
                                        "synaptic current"); pointwise so each
                                        time step is projected independently (a
                                        FullConnect would flatten/mix the time axis)
           TNNetLIFNeuron(tau,V_th,alpha,LearnDynamics)  HIDDEN spiking neurons
                                        over time; LearnDynamics makes the
                                        per-channel threshold V_th and leak beta
                                        TRAINABLE parameters (opt-in)
           TNNetAvgChannel              rate readout: mean spikes over time
                                        -> (1,1,HIDDEN)
    -> FullConnectLinear(NCLASS) -> SoftMax

Trained with plain SoftMax + cross-entropy (NN.Backpropagate against a one-hot
target). The whole gradient passes THROUGH the LIF layer via the surrogate.

HEADLINE PAYOFF
---------------
We report TEST ACCURACY together with the mean SPIKE RATE (sparsity) of the
hidden layer -- the fraction of (time, neuron) sites that fired. A spiking net is
attractive precisely because that rate is LOW (event-driven, sparse compute), yet
the surrogate-trained net still classifies well above chance.

HONEST CAVEAT
-------------
The forward is the HARD reset + Heaviside spike, but the backward uses a SMOOTH
surrogate -- so the gradient is a BIASED estimator of the true (a.e. zero)
gradient: there is a real forward-vs-backward MISMATCH. In practice this means a
spiking net needs a gentler learning rate and/or more steps than an equivalent
ReLU MLP on the same task, and training is noisier. We pick a modest LR and a
small task accordingly; do not expect ReLU-MLP convergence speed.

Pure CPU, tiny dimensions and batches -> runs in well under 5 minutes on 2 cores
with modest memory. No binaries are committed.

LICENSE: GPL (same as the neural-api project).

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, neuralnetwork, neuralvolume;

const
  NCLASS   = 4;      // number of classes / prototypes
  DIN      = 12;     // input features
  T        = 20;     // time steps (spike-train length)
  HIDDEN   = 24;     // spiking hidden neurons
  EPOCHS   = 600;
  MB       = 24;     // mini-batch size
  LR       = 0.02;   // gentle LR (surrogate is a biased gradient estimator)
  MOMENTUM = 0.9;
  TAU      = 2.5;    // membrane time constant -> beta = exp(-1/tau)
  VTH      = 1.0;    // firing threshold
  ALPHA    = 2.0;    // surrogate sharpness
  NEVAL    = 800;    // test samples for accuracy / spike-rate

var
  NN: TNNet;
  LIFIdx: integer;                 // layer index of the LIF layer
  Proto: array[0..NCLASS - 1, 0..DIN - 1] of TNeuralFloat; // firing probabilities

// Build the NCLASS prototype firing-probability vectors (fixed, seeded).
procedure BuildPrototypes();
var
  c, f, oldSeed: integer;
begin
  oldSeed := RandSeed;
  RandSeed := 13579;
  for c := 0 to NCLASS - 1 do
    for f := 0 to DIN - 1 do
      // mostly-low base rate with a few class-specific high-rate features
      if (f mod NCLASS) = c
        then Proto[c, f] := 0.85
        else Proto[c, f] := 0.12;
  RandSeed := oldSeed;
end;

// Rate-encode class c into a (T,1,DIN) Bernoulli spike train. seed makes the
// draw reproducible per sample.
procedure MakeSample(c, seed: integer; Inp: TNNetVolume);
var
  ti, f, idx, oldSeed: integer;   // NOTE: loop var must NOT be named t -- FPC is
begin                             // case-insensitive so it would shadow const T.
  oldSeed := RandSeed;
  RandSeed := seed;
  Inp.ReSize(T, 1, DIN);
  for ti := 0 to T - 1 do
    for f := 0 to DIN - 1 do
    begin
      idx := ti * DIN + f;
      if Random < Proto[c, f]
        then Inp.FData[idx] := 1
        else Inp.FData[idx] := 0;
    end;
  RandSeed := oldSeed;
end;

// Evaluate test accuracy AND mean hidden spike rate over NEVAL fresh samples.
procedure Evaluate(out acc, spikeRate: TNeuralFloat);
var
  s, c, predicted, correct, k: integer;
  bestV, fired, total: TNeuralFloat;
  Inp: TNNetVolume;
  SpikeOut: TNNetVolume;
begin
  Inp := TNNetVolume.Create();
  correct := 0;
  fired := 0;
  total := 0;
  try
    for s := 0 to NEVAL - 1 do
    begin
      c := s mod NCLASS;
      MakeSample(c, 2000000 + s, Inp);   // disjoint from training seeds
      NN.Compute(Inp);

      // accumulate hidden spike rate from the LIF layer output (binary)
      SpikeOut := NN.Layers[LIFIdx].Output;
      for k := 0 to SpikeOut.Size - 1 do
        fired := fired + SpikeOut.FData[k];
      total := total + SpikeOut.Size;

      // arg-max over the softmax output
      predicted := 0;
      bestV := NN.GetLastLayer.Output.FData[0];
      for k := 1 to NCLASS - 1 do
        if NN.GetLastLayer.Output.FData[k] > bestV then
        begin
          bestV := NN.GetLastLayer.Output.FData[k];
          predicted := k;
        end;
      if predicted = c then Inc(correct);
    end;
  finally
    Inp.Free;
  end;
  acc := correct / NEVAL;
  spikeRate := fired / total;
end;

var
  epoch, s, c: integer;
  acc, spikeRate, accBefore: TNeuralFloat;
  Inp, Tgt: TNNetVolume;

begin
  Randomize;
  RandSeed := 424242;
  WriteLn('SpikingMNIST: a leaky-integrate-and-fire spiking net classifies ',
    'rate-coded inputs');
  WriteLn('             via a SURROGATE GRADIENT (', NCLASS, ' classes, T=', T,
    ' time steps, ', HIDDEN, ' LIF neurons).');
  WriteLn;

  BuildPrototypes();

  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(T, 1, DIN));
  // AddSpikingBlock wires the canonical PointwiseConvLinear -> TNNetLIFNeuron ->
  // TNNetAvgChannel (rate-readout) pipeline in one call. LearnDynamics=true makes
  // the per-channel firing threshold V_th and leak beta TRAINABLE (opt-in).
  NN.AddSpikingBlock(HIDDEN, TAU, VTH, ALPHA, {LearnDynamics=}true);
  // The LIF layer is the one BEFORE the AvgChannel rate readout (= last - 1).
  LIFIdx := NN.GetLastLayerIdx() - 1;
  NN.AddLayer(TNNetFullConnectLinear.Create(NCLASS));
  NN.AddLayer(TNNetSoftMax.Create());

  NN.SetLearningRate(LR, MOMENTUM);
  NN.SetBatchUpdate(true);   // accumulate per-sample deltas, step once per batch

  Inp := TNNetVolume.Create(T, 1, DIN);
  Tgt := TNNetVolume.Create(NCLASS, 1, 1);

  Evaluate(accBefore, spikeRate);
  WriteLn('Test accuracy BEFORE training: ', (accBefore * 100):0:1,
    '%  (chance = ', (100.0 / NCLASS):0:1, '%)   hidden spike rate: ',
    (spikeRate * 100):0:1, '%');
  WriteLn;

  try
    for epoch := 0 to EPOCHS - 1 do
    begin
      NN.ClearDeltas();
      for s := 0 to MB - 1 do
      begin
        c := (epoch * MB + s) mod NCLASS;
        MakeSample(c, epoch * MB + s, Inp);
        NN.Compute(Inp);
        // one-hot target for SoftMax + cross-entropy
        Tgt.Fill(0);
        Tgt.FData[c] := 1;
        NN.Backpropagate(Tgt);
      end;
      NN.UpdateWeights();

      if (epoch mod 100 = 0) or (epoch = EPOCHS - 1) then
      begin
        Evaluate(acc, spikeRate);
        WriteLn(Format('  epoch %4d  test-acc=%5.1f%%  hidden-spike-rate=%5.1f%%',
          [epoch, acc * 100, spikeRate * 100]));
      end;
    end;

    Evaluate(acc, spikeRate);
    WriteLn;
    WriteLn('Test accuracy AFTER training: ', (acc * 100):0:1, '%');
    WriteLn('Mean hidden SPIKE RATE: ', (spikeRate * 100):0:1,
      '%  (sparsity = ', ((1 - spikeRate) * 100):0:1, '% silent).');
    WriteLn('Headline: a SPIKING net learned the classification through the ',
      'surrogate gradient,');
    WriteLn('          climbing from ', (accBefore * 100):0:1, '% to ',
      (acc * 100):0:1, '% while firing on only ', (spikeRate * 100):0:1,
      '% of (time, neuron) sites.');
  finally
    Inp.Free;
    Tgt.Free;
    NN.Free;
  end;
end.
