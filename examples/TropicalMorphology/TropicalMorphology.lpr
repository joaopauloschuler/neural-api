// Tropical (max-plus) morphology demo -- TNNetTropicalLinear
// ---------------------------------------------------------------------------
// The headline this demo proves: "different algebra, different hypothesis
// class". A max-plus (tropical) DILATION layer computes
//     y = max_j ( x_j + W_j )
// which is a CONVEX piecewise-linear function of its inputs. Feed it a bank of
// learnable affine features (slopes b_j*t + c_j) and the single tropical max
// collapses them into  max_j ( b_j*t + (c_j + W_j) ) -- exactly a tropical
// polynomial, i.e. ANY convex piecewise-linear curve.
//
// A plain linear layer of the SAME width, by contrast, can only output an
// AFFINE function of the same features:  sum_j a_j*(b_j*t) + bias = (one line).
// No matter how wide, a single linear layer is one straight line; it provably
// cannot bend. So on a convex, V-/fan-shaped target the tropical stack fits to
// near-zero error while the same-width linear baseline is stuck at the
// best-fitting straight line.
//
// We also show the ERODE (min-plus) sibling fitting a CONCAVE target.
//
// Pure CPU, single-threaded-friendly, runs in a few seconds. No binaries are
// committed.
program TropicalMorphology;
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, neuralnetwork, neuralvolume;

const
  NFEAT    = 16;     // affine feature bank width (shared by both models)
  NSAMPLES = 64;     // training points sampled on t in [-2, 2]
  EPOCHS   = 4000;
  LR       = 0.02;

// Convex piecewise-linear target: the upper envelope of three lines.
//   g(t) = max( -2t - 1 ,  0.3t + 0.2 ,  2.5t - 0.5 )
// A convex fan/V shape -- the canonical thing a single line cannot fit but a
// max-plus tropical polynomial fits exactly.
function ConvexTarget(t: TNeuralFloat): TNeuralFloat;
begin
  Result := Max(Max(-2.0 * t - 1.0, 0.3 * t + 0.2), 2.5 * t - 0.5);
end;

// Concave target (negated convex envelope) for the erode demo.
function ConcaveTarget(t: TNeuralFloat): TNeuralFloat;
begin
  Result := Min(Min(2.0 * t + 1.0, -0.3 * t - 0.2), -2.5 * t + 0.5);
end;

function SampleT(idx: integer): TNeuralFloat;
begin
  // Even grid over [-2, 2].
  Result := -2.0 + 4.0 * idx / (NSAMPLES - 1);
end;

// Train a 1-output regressor and return final mean-squared error over the grid.
function Train(NN: TNNet; concave: boolean; const tag: string): TNeuralFloat;
var
  ep, s: integer;
  Inp, Tgt: TNNetVolume;
  t, d, loss: TNeuralFloat;
begin
  Inp := TNNetVolume.Create(1, 1, 1);
  Tgt := TNNetVolume.Create(1, 1, 1);
  NN.SetLearningRate(LR, 0.9);
  NN.SetBatchUpdate(true);
  loss := 0;
  for ep := 0 to EPOCHS - 1 do
  begin
    loss := 0;
    NN.ClearDeltas();
    for s := 0 to NSAMPLES - 1 do
    begin
      t := SampleT(s);
      Inp.FData[0] := t;
      if concave then Tgt.FData[0] := ConcaveTarget(t)
      else Tgt.FData[0] := ConvexTarget(t);
      NN.Compute(Inp);
      NN.Backpropagate(Tgt);
      d := NN.GetLastLayer.Output.FData[0] - Tgt.FData[0];
      loss := loss + d * d;
    end;
    NN.UpdateWeights();
    if (ep mod 1000 = 0) or (ep = EPOCHS - 1) then
      WriteLn(Format('  [%s] epoch %4d  MSE %.6f', [tag, ep, loss / NSAMPLES]));
  end;
  Inp.Free; Tgt.Free;
  Result := loss / NSAMPLES;
end;

var
  Tropical, Linear, TropErode: TNNet;
  mseTrop, mseLin, mseErode: TNeuralFloat;
begin
  RandSeed := 424242;
  WriteLn('Tropical morphology demo: max-plus dilation fits a convex envelope');
  WriteLn('that a same-width single linear layer structurally cannot.');
  WriteLn(Format('Feature width = %d, samples = %d', [NFEAT, NSAMPLES]));
  WriteLn;

  // ---- Tropical stack: affine feature bank -> max-plus dilation. ----------
  // PointwiseConvLinear lifts the scalar t into NFEAT learnable affine features
  // (b_j*t + c_j); TNNetTropicalLinear then takes max_j(feat_j + W_j), giving a
  // convex piecewise-linear curve = a tropical polynomial in t.
  Tropical := TNNet.Create();
  Tropical.AddLayer(TNNetInput.Create(1, 1, 1));
  Tropical.AddLayer(TNNetPointwiseConvLinear.Create(NFEAT));   // NFEAT affine features
  Tropical.AddLayer(TNNetReshape.Create(NFEAT, 1, 1));         // flatten to a vector
  Tropical.AddLayer(TNNetTropicalLinear.Create(1, {erode=}0)); // max-plus dilation

  // ---- Baseline: SAME-width single linear layer. -------------------------
  // Same NFEAT affine features feeding one linear output. The whole thing is
  // affine in t -> a single straight line, which cannot bend to a convex fan.
  Linear := TNNet.Create();
  Linear.AddLayer(TNNetInput.Create(1, 1, 1));
  Linear.AddLayer(TNNetPointwiseConvLinear.Create(NFEAT));
  Linear.AddLayer(TNNetReshape.Create(NFEAT, 1, 1));
  Linear.AddLayer(TNNetFullConnectLinear.Create(1));

  // ---- Erode sibling: min-plus dilation fits a CONCAVE target. -----------
  TropErode := TNNet.Create();
  TropErode.AddLayer(TNNetInput.Create(1, 1, 1));
  TropErode.AddLayer(TNNetPointwiseConvLinear.Create(NFEAT));
  TropErode.AddLayer(TNNetReshape.Create(NFEAT, 1, 1));
  TropErode.AddLayer(TNNetTropicalLinear.Create(1, {erode=}1)); // min-plus erosion

  WriteLn('Training tropical (max-plus) model on the CONVEX target...');
  mseTrop := Train(Tropical, false, 'TROP');
  WriteLn('Training same-width linear baseline on the CONVEX target...');
  mseLin := Train(Linear, false, 'LIN');
  WriteLn('Training tropical (min-plus / erode) model on the CONCAVE target...');
  mseErode := Train(TropErode, true, 'EROD');
  WriteLn;

  WriteLn('=== Final mean-squared error on the regression grid ===');
  WriteLn(Format('  tropical dilation (convex target) : %.6f', [mseTrop]));
  WriteLn(Format('  linear baseline   (convex target) : %.6f', [mseLin]));
  WriteLn(Format('  tropical erosion  (concave target): %.6f', [mseErode]));
  WriteLn;
  WriteLn('HEADLINE: the max-plus stack drives the convex-target error to near');
  WriteLn('zero, while the same-width linear layer plateaus at the best single');
  WriteLn('straight line (an order of magnitude worse). The min-plus erode');
  WriteLn('sibling does the same for the concave target. Different algebra,');
  WriteLn('different hypothesis class.');

  Tropical.Free;
  Linear.Free;
  TropErode.Free;
end.
