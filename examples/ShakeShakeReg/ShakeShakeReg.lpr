program ShakeShakeReg;
(*
ShakeShakeReg: contrasts Shake-Shake regularization (AddShakeShakeBlock,
the stochastic alpha/beta merge of two residual branches) against a plain
DETERMINISTIC two-branch residual (y = skip + 0.5*B1 + 0.5*B2) on a small,
noisy, over-parameterised toy classification problem.

Both arms share an identical trunk and identical two-branch shapes; the ONLY
difference is the merge layer:
  * Shake-Shake arm: TNNetShakeShakeMerge -- forward samples alpha~U[0,1) and
    backward samples an INDEPENDENT beta~U[0,1) per pass (stochastic during
    training, deterministic 0.5/0.5 at eval).
  * Deterministic arm: each branch is scaled by 0.5 (TNNetMulByConstant) and
    summed with the skip (TNNetSum) -- the fixed 0.5/0.5 average, no noise.

The headline regularisation win is a SMALLER train-minus-val gap for the
shake-shake arm (better generalisation), in the spirit of the Mixup/SAM demos.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cInDim      = 40;    // 2 informative dims + 38 random nuisance dims
  cTrainCnt   = 64;    // few samples in a high-dim space -> easy to memorise
  cValCnt     = 600;   // large clean val set for a stable gap estimate
  cNoiseProb  = 0.25;  // 25% of TRAINING labels are flipped (noise)
  cHidden     = 64;    // over-parameterised trunk
  cBranchHid  = 96;    // wide branch -> plenty of capacity to overfit
  cEpochs     = 40;    // stop before full memorisation -> gap is informative

  // High-dim toy: the label depends ONLY on the first 2 (informative) dims;
  // the remaining cInDim-2 dims are pure Gaussian-ish nuisance noise. With few
  // training samples in this high-dim space an over-sized net can MEMORISE
  // individual (noisy) labels via the nuisance dims, so the deterministic arm
  // overfits while the stochastic Shake-Shake merge resists it.
  function CreateToyPairList(MaxCnt: integer; NoiseProb: TNeuralFloat): TNNetVolumePairList;
  var
    Cnt, Cls, Lbl, D: integer;
    cx, cy: TNeuralFloat;
    Inp: TNNetVolume;
  begin
    Result := TNNetVolumePairList.Create();
    for Cnt := 1 to MaxCnt do
    begin
      Cls := Random(2);
      if Cls = 0 then begin cx := -0.9; cy := -0.9; end
                 else begin cx :=  0.9; cy :=  0.9; end;
      Inp := TNNetVolume.Create(cInDim);
      // Two informative dims (overlapping blobs).
      Inp.Raw[0] := cx + (Random()-0.5) * 1.6;
      Inp.Raw[1] := cy + (Random()-0.5) * 1.6;
      // Nuisance dims: identically distributed for both classes (no signal).
      for D := 2 to cInDim - 1 do
        Inp.Raw[D] := (Random()-0.5) * 2.0;
      Lbl := Cls;
      // Inject label noise (only meaningful for the training set).
      if Random() < NoiseProb then Lbl := 1 - Lbl;
      if Lbl = 0
      then Result.Add(TNNetVolumePair.Create(Inp, TNNetVolume.Create([1.0, 0.0])))
      else Result.Add(TNNetVolumePair.Create(Inp, TNNetVolume.Create([0.0, 1.0])));
    end;
  end;

  function ArgMax(V: TNNetVolume): integer;
  begin
    if V.Raw[0] >= V.Raw[1] then Result := 0 else Result := 1;
  end;

  // Accuracy + mean cross-entropy loss over a pair list.
  procedure Evaluate(NN: TNNet; Pairs: TNNetVolumePairList;
    out Acc, Loss: TNeuralFloat);
  var
    Cnt, Hits, T: integer;
    Output: TNNetVolume;
    P, SumLoss: TNeuralFloat;
  begin
    Hits := 0;
    SumLoss := 0;
    Output := TNNetVolume.Create();
    for Cnt := 0 to Pairs.Count - 1 do
    begin
      NN.Compute(Pairs[Cnt].I);
      NN.GetOutput(Output);
      if ArgMax(Output) = ArgMax(Pairs[Cnt].O) then Inc(Hits);
      T := ArgMax(Pairs[Cnt].O);
      P := Output.Raw[T];
      if P < 1e-7 then P := 1e-7;
      SumLoss := SumLoss - Ln(P);
    end;
    Output.Free;
    Acc  := Hits / Pairs.Count;
    Loss := SumLoss / Pairs.Count;
  end;

  // Shared trunk: Input(2) -> over-sized FC -> pointwise-friendly (1,1,Depth).
  procedure AddTrunk(NN: TNNet);
  begin
    NN.AddLayer([
      TNNetInput.Create(cInDim),
      TNNetFullConnectReLU.Create(cHidden),
      TNNetFullConnectReLU.Create(cHidden)
    ]);
  end;

  // Arm 1: stochastic Shake-Shake block (builder).
  function BuildShakeNet(): TNNet;
  begin
    Result := TNNet.Create();
    AddTrunk(Result);
    Result.AddShakeShakeBlock(cBranchHid);  // y = x + alpha*B1 + (1-alpha)*B2
    Result.AddLayer([
      TNNetFullConnectLinear.Create(2),
      TNNetSoftMax.Create()
    ]);
  end;

  // Arm 2: deterministic two-branch residual: y = skip + 0.5*B1 + 0.5*B2.
  // Same branch shapes (PointwiseConvReLU(BranchHid) -> PointwiseConvLinear)
  // as AddShakeShakeBlock, only the merge differs (fixed 0.5/0.5, no noise).
  function BuildDetNet(): TNNet;
  var
    Skip, B1, B2: TNNetLayer;
    Depth: integer;
  begin
    Result := TNNet.Create();
    AddTrunk(Result);
    Skip  := Result.GetLastLayer();
    Depth := Skip.Output.Depth;
    // Branch 1.
    Result.AddLayerAfter(TNNetPointwiseConvReLU.Create(cBranchHid), Skip);
    Result.AddLayer(TNNetPointwiseConvLinear.Create(Depth));
    B1 := Result.AddLayer(TNNetMulByConstant.Create(0.5));
    // Branch 2.
    Result.AddLayerAfter(TNNetPointwiseConvReLU.Create(cBranchHid), Skip);
    Result.AddLayer(TNNetPointwiseConvLinear.Create(Depth));
    B2 := Result.AddLayer(TNNetMulByConstant.Create(0.5));
    // Deterministic merge: skip + 0.5*B1 + 0.5*B2.
    Result.AddLayer(TNNetSum.Create([B1, B2, Skip]));
    Result.AddLayer([
      TNNetFullConnectLinear.Create(2),
      TNNetSoftMax.Create()
    ]);
  end;

  procedure TrainArm(NN: TNNet; TrainPairs: TNNetVolumePairList);
  var
    Epoch, I, Idx: integer;
    Pair: TNNetVolumePair;
  begin
    NN.SetLearningRate(0.005, 0.9);
    for Epoch := 1 to cEpochs do
      for I := 0 to TrainPairs.Count - 1 do
      begin
        Idx := Random(TrainPairs.Count);
        Pair := TrainPairs[Idx];
        NN.Compute(Pair.I);
        NN.Backpropagate(Pair.O);
      end;
  end;

  procedure ReportArm(const Name: string; NN: TNNet;
    TrainPairs, ValPairs: TNNetVolumePairList);
  var
    TrAcc, TrLoss, VaAcc, VaLoss: TNeuralFloat;
  begin
    Evaluate(NN, TrainPairs, TrAcc, TrLoss);
    Evaluate(NN, ValPairs,   VaAcc, VaLoss);
    WriteLn(Format('%-14s | %7.2f %7.4f | %7.2f %7.4f | %7.2f %8.4f',
      [Name,
       TrAcc*100, TrLoss,
       VaAcc*100, VaLoss,
       (TrAcc - VaAcc)*100, (VaLoss - TrLoss)]));
  end;

  procedure RunAlgo();
  var
    TrainPairs, ValPairs: TNNetVolumePairList;
    ShakeNet, DetNet: TNNet;
  begin
    RandSeed := 20260607;
    // Noisy, small training set; clean, large validation set.
    TrainPairs := CreateToyPairList(cTrainCnt, cNoiseProb);
    ValPairs   := CreateToyPairList(cValCnt, 0.0);

    WriteLn('Shake-Shake regularization bake-off');
    WriteLn('  noisy ', cInDim, 'D 2-class toy (2 informative + ',
      cInDim-2, ' nuisance dims): ', cTrainCnt, ' train (',
      Round(cNoiseProb*100), '% label noise), ', cValCnt, ' clean val');
    WriteLn('  over-parameterised trunk(', cHidden,
      ') + two branch(', cBranchHid, ') residual, ', cEpochs, ' epochs');
    WriteLn('');

    // Deterministic baseline first (same RNG point as shake net would start).
    DetNet := BuildDetNet();
    TrainArm(DetNet, TrainPairs);

    ShakeNet := BuildShakeNet();
    TrainArm(ShakeNet, TrainPairs);

    WriteLn('=== Comparison (acc %, cross-entropy loss) ===');
    WriteLn('arm            |   trAcc  trLoss |   vaAcc  vaLoss |  accGAP lossGAP');
    WriteLn('---------------+-----------------+-----------------+----------------');
    ReportArm('Deterministic', DetNet,   TrainPairs, ValPairs);
    ReportArm('Shake-Shake',   ShakeNet, TrainPairs, ValPairs);
    WriteLn('');
    WriteLn('accGAP = trainAcc - valAcc (smaller = less overfitting).');
    WriteLn('lossGAP = valLoss - trainLoss (smaller = better generalisation).');

    ShakeNet.Free;
    DetNet.Free;
    ValPairs.Free;
    TrainPairs.Free;
  end;

var
  // Stops Lazarus errors
  Application: record Title:string; end;

begin
  Application.Title:='ShakeShakeReg Example';
  RunAlgo();
end.
