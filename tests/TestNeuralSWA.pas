unit TestNeuralSWA;

// Tests for the Stochastic Weight Averaging callback (TNeuralFitSWA, a port of
// torch.optim.swa_utils): equal-weight running mean of end-of-epoch snapshots
// over the schedule tail, plus the averaged-weights swap-in/swap-out used for
// eval/save. Coded by Claude (AI).

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralnetwork, neuralvolume,
  neuralfit;

type
  // Minimal TNeuralFitBase carrier so OnEpochEnd(Sender) can read Sender.NN
  // without spinning up a full training loop. FNN is protected, so a
  // same-unit descendant can point it at our controllable net.
  TSenderStub = class(TNeuralFitBase)
  public
    procedure SetNetForTest(pNN: TNNet);
  end;

  TTestNeuralSWA = class(TTestCase)
  published
    procedure TestDefaultsAndEmptyState;
    procedure TestEqualWeightMeanMatchesArithmeticMean;
    procedure TestApplyRestoreSwapIsBitIdentical;
    procedure TestWindowSkipsEarlyEpochsAndCadence;
    procedure TestDisabledPathUnchanged;
  end;

implementation

procedure TSenderStub.SetNetForTest(pNN: TNNet);
begin
  FNN := pNN;
end;

// Builds a tiny 1-neuron linear net under a fixed seed.
function BuildSwaNet: TNNet;
begin
  RandSeed := 424242;
  Result := TNNet.Create();
  Result.AddLayer([
    TNNetInput.Create(2),
    TNNetFullConnectLinear.Create(3),
    TNNetFullConnectLinear.Create(1)
  ]);
  Result.SetLearningRate(0.01, 0.9);
end;

// Sets every trainable weight (and bias) of the net to a single value, so a
// snapshot's whole-net weight sum is exactly predictable.
procedure FillNetWeights(Net: TNNet; AValue: TNeuralFloat);
var
  L, N, W: integer;
  Neuron: TNNetNeuron;
begin
  for L := 1 to Net.Layers.Count - 1 do
    for N := 0 to Net.Layers[L].Neurons.Count - 1 do
    begin
      Neuron := Net.Layers[L].Neurons[N];
      for W := 0 to Neuron.Weights.Size - 1 do
        Neuron.Weights.FData[W] := AValue;
      Neuron.BiasWeight := AValue;
    end;
end;

// 4 tiny regression samples reused for both training and validation.
function BuildSwaPairs: TNNetVolumePairList;
var
  I: integer;
  Pair: TNNetVolumePair;
begin
  Result := TNNetVolumePairList.Create();
  for I := 0 to 3 do
  begin
    Pair := TNNetVolumePair.Create();
    Pair.A.ReSize(2, 1, 1);
    Pair.B.ReSize(1, 1, 1);
    Pair.A.Raw[0] := (I and 1);
    Pair.A.Raw[1] := ((I shr 1) and 1);
    Pair.B.Raw[0] := (I and 1) xor ((I shr 1) and 1);
    Result.Add(Pair);
  end;
end;

procedure TTestNeuralSWA.TestDefaultsAndEmptyState;
var
  Swa: TNeuralFitSWA;
begin
  Swa := TNeuralFitSWA.Create();
  try
    AssertEquals('StartEpoch defaults to 1', 1, Swa.StartEpoch);
    AssertEquals('SnapshotEveryEpochs defaults to 1', 1, Swa.SnapshotEveryEpochs);
    AssertEquals('No snapshots accumulated yet', 0, Swa.SnapshotCount());
    AssertFalse('Not swapped before any apply', Swa.Swapped);
    AssertNull('Shadow net is nil before the first snapshot', Swa.SWAShadowNet());
    // Apply with no snapshots is a no-op (cannot average nothing).
    AssertFalse('ApplySWAWeights with no snapshot returns false',
      Swa.ApplySWAWeights(nil));
  finally
    Swa.Free;
  end;
end;

procedure TTestNeuralSWA.TestEqualWeightMeanMatchesArithmeticMean;
var
  Net: TNNet;
  Swa: TNeuralFitSWA;
  Stub: TSenderStub;
  ExpectedMean, ShadowSum, PerNeuronW: TNeuralFloat;
const
  // Three snapshot values; their arithmetic mean is (1+4+7)/3 = 4.
  V: array[0..2] of TNeuralFloat = (1.0, 4.0, 7.0);
begin
  Net := BuildSwaNet();
  Stub := TSenderStub.Create();
  Stub.SetNetForTest(Net);
  Swa := TNeuralFitSWA.Create(1, 1); // snapshot every epoch from epoch 1
  try
    // nil Sender must be skipped (no net to snapshot).
    FillNetWeights(Net, V[0]);
    Swa.OnEpochEnd(nil, 1);
    AssertEquals('nil Sender does not snapshot', 0, Swa.SnapshotCount());

    // Drive three OnEpochEnd snapshots with a controllable net: set all
    // weights to V[i] before each snapshot. The wrapper must hold the uniform
    // (arithmetic) mean of the snapshots.
    FillNetWeights(Net, V[0]);
    Swa.OnEpochEnd(Stub, 1);
    FillNetWeights(Net, V[1]);
    Swa.OnEpochEnd(Stub, 2);
    FillNetWeights(Net, V[2]);
    Swa.OnEpochEnd(Stub, 3);

    AssertEquals('Three snapshots folded into the mean', 3, Swa.SnapshotCount());

    // Each averaged weight must equal the arithmetic mean of V.
    ExpectedMean := (V[0] + V[1] + V[2]) / 3;
    PerNeuronW := Swa.SWAShadowNet().Layers[1].Neurons[0].Weights.FData[0];
    AssertEquals('Averaged weight is the arithmetic mean of the snapshots',
      ExpectedMean, PerNeuronW, 1e-5);

    // Whole-net sum check: mean-of-all-weights net has every element = 4, so
    // its sum equals (count of weights) * 4.
    ShadowSum := Swa.SWAShadowNet().GetWeightSum();
    AssertEquals('Shadow weight sum equals mean-filled net sum',
      Swa.SWAShadowNet().CountWeights() * ExpectedMean, ShadowSum, 1e-3);
  finally
    Swa.Free;
    Stub.Free;
    Net.Free;
  end;
end;

procedure TTestNeuralSWA.TestApplyRestoreSwapIsBitIdentical;
var
  Net: TNNet;
  Swa: TNeuralFitSWA;
  Stub: TSenderStub;
  LiveSumBefore, LiveSumAfter, SwappedSum, ExpectedMean: TNeuralFloat;
const
  V: array[0..1] of TNeuralFloat = (2.0, 6.0); // mean = 4
begin
  Net := BuildSwaNet();
  Stub := TSenderStub.Create();
  Stub.SetNetForTest(Net);
  Swa := TNeuralFitSWA.Create(1, 1);
  try
    // Accumulate two snapshots (mean 4).
    FillNetWeights(Net, V[0]);
    Swa.OnEpochEnd(Stub, 1);
    FillNetWeights(Net, V[1]);
    Swa.OnEpochEnd(Stub, 2);

    // Now set the live net to a DISTINCT value so we can tell the swap apart.
    FillNetWeights(Net, 99.0);
    LiveSumBefore := Net.GetWeightSum();

    // Swap the averaged weights in. The live net must now read the mean.
    AssertTrue('ApplySWAWeights succeeds with snapshots present',
      Swa.ApplySWAWeights(Net));
    AssertTrue('Reports swapped while averaged weights are in', Swa.Swapped);
    ExpectedMean := (V[0] + V[1]) / 2;
    SwappedSum := Net.GetWeightSum();
    AssertEquals('Live net holds the averaged weights while swapped',
      Net.CountWeights() * ExpectedMean, SwappedSum, 1e-3);

    // A nested apply must be ignored (returns false, no further stash).
    AssertFalse('Nested ApplySWAWeights is ignored', Swa.ApplySWAWeights(Net));

    // Restore: live weights must be bit-identical to before the swap.
    Swa.RestoreLiveWeights(Net);
    AssertFalse('No longer swapped after restore', Swa.Swapped);
    LiveSumAfter := Net.GetWeightSum();
    AssertEquals('Live weights bit-identical after restore',
      LiveSumBefore, LiveSumAfter, 0.0);
  finally
    Swa.Free;
    Stub.Free;
    Net.Free;
  end;
end;

procedure TTestNeuralSWA.TestWindowSkipsEarlyEpochsAndCadence;
var
  Net: TNNet;
  Swa: TNeuralFitSWA;
  Stub: TSenderStub;
begin
  Net := BuildSwaNet();
  Stub := TSenderStub.Create();
  Stub.SetNetForTest(Net);
  // Start at epoch 3, snapshot every 2 epochs: window epochs are 3,5,7,...
  Swa := TNeuralFitSWA.Create(3, 2);
  try
    FillNetWeights(Net, 1.0);
    Swa.OnEpochEnd(Stub, 1); // before window => skip
    Swa.OnEpochEnd(Stub, 2); // before window => skip
    AssertEquals('No snapshots before StartEpoch', 0, Swa.SnapshotCount());

    Swa.OnEpochEnd(Stub, 3); // window start => snapshot
    AssertEquals('First in-window epoch snapshots', 1, Swa.SnapshotCount());

    Swa.OnEpochEnd(Stub, 4); // off-cadence => skip
    AssertEquals('Off-cadence epoch is skipped', 1, Swa.SnapshotCount());

    Swa.OnEpochEnd(Stub, 5); // on-cadence => snapshot
    AssertEquals('On-cadence epoch snapshots', 2, Swa.SnapshotCount());
  finally
    Swa.Free;
    Stub.Free;
    Net.Free;
  end;
end;

procedure TTestNeuralSWA.TestDisabledPathUnchanged;
var
  NetA, NetB: TNNet;
  Pairs, ValPairs: TNNetVolumePairList;
  FitA, FitB: TNeuralFit;
  Swa: TNeuralFitSWA;
const
  cEpochs = 3;
begin
  // Two identical fits: one with NO SWA callback, one with a SWA callback that
  // only OBSERVES (it never swaps weights into the live net during training).
  // SWA is a pure side-channel: it never writes the live weights or deltas, so
  // the optimization MATH is identical with SWA on or off. The ONLY difference
  // is the documented bounded RNG perturbation - the wrapper's lazy Clone()
  // draws from the global Mersenne-Twister when the first snapshot is taken,
  // which can shift the stochastic batch-shuffle order by a bounded amount
  // (same caveat as the EMA wiring). The trained live weights therefore match
  // within float tolerance, not necessarily bit-for-bit.
  NetA := BuildSwaNet();
  NetB := BuildSwaNet();
  Pairs := BuildSwaPairs();
  ValPairs := BuildSwaPairs();

  RandSeed := 100;
  FitA := TNeuralFit.Create;
  try
    FitA.HideMessages; FitA.Verbose := False; FitA.MaxThreadNum := 1;
    FitA.InitialLearningRate := 0.01; FitA.LearningRateDecay := 0;
    FitA.StaircaseEpochs := 1; FitA.LoadBestAtEnd := False;
    FitA.EnableRegressionComparison();
    FitA.Fit(NetA, Pairs, ValPairs, nil, 4, cEpochs);
  finally
    FitA.Free;
  end;

  RandSeed := 100;
  FitB := TNeuralFit.Create;
  Swa := TNeuralFitSWA.Create(2, 1);
  try
    FitB.HideMessages; FitB.Verbose := False; FitB.MaxThreadNum := 1;
    FitB.InitialLearningRate := 0.01; FitB.LearningRateDecay := 0;
    FitB.StaircaseEpochs := 1; FitB.LoadBestAtEnd := False;
    FitB.EnableRegressionComparison();
    FitB.OwnsCallbacks := True;
    FitB.AddCallback(Swa);
    FitB.Fit(NetB, Pairs, ValPairs, nil, 4, cEpochs);
    // SWA observed epochs 2 and 3 (StartEpoch=2).
    AssertEquals('SWA accumulated the tail epochs', cEpochs - 1,
      Swa.SnapshotCount());
  finally
    FitB.Free; // frees Swa (OwnsCallbacks)
  end;

  // Live training weights match (within the documented RNG-shuffle tolerance)
  // with/without the observing SWA callback.
  AssertEquals('SWA-observing fit leaves live weights unchanged (within RNG tol)',
    NetA.GetWeightSum(), NetB.GetWeightSum(), 1e-2);

  Pairs.Free;
  ValPairs.Free;
  NetA.Free;
  NetB.Free;
end;

initialization
  RegisterTest(TTestNeuralSWA);
end.
