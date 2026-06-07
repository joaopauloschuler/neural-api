// Growing Neural Cellular Automata (tiny, pure-CPU).
//
// A from-scratch reproduction of Mordvintsev, Randazzo, Niklasson & Levin (2020),
// "Growing Neural Cellular Automata" (https://distill.pub/2020/growing-ca/), shrunk
// to a 16x16 grid so it trains in a couple of minutes on two CPU cores with no
// image-library dependency (frames are printed as ASCII).
//
// THE MODEL. The world is a 16x16 grid of cells. Each cell carries Ch=12 channels:
// the first four are the VISIBLE state (R,G,B and an "alpha"/alive value), the
// remaining eight are HIDDEN scratch channels the rule may use freely. One CA
// "rule" step is a SHARED-WEIGHT residual conv stack applied to every cell in
// parallel and IN PLACE:
//   perceive : a learned 3x3 conv (padding 1) -> PDim perception channels
//   1x1 ReLU : TNNetPointwiseConvReLU  (Hid hidden units)
//   1x1 lin  : TNNetPointwiseConvLinear (Ch update channels, no bias)
//   update   : grid := clamp( grid + dgrid )      (residual + bounded leaky-ReLUL)
// The rule is applied T times. Crucially EVERY step after the first reuses the SAME
// weights via TNNetConvolutionSharedWeights -- the key enabler. Without weight
// tying each step would learn its own filters and the object would not be a genuine
// "growth rule"; with it the trainable parameter count is INDEPENDENT of T.
//
// TRAINING is plain backprop-through-time over the WHOLE unrolled graph: because the
// T steps are ordinary layers wired in sequence, TNNet.Backpropagate already walks
// the chain end to end and accumulates the shared layer's gradient across all T
// applications. We use the batch-update idiom (SetBatchUpdate(True) so per-sample
// backprop does not zero the shared Delta) and clip the global gradient norm; the
// L2 loss compares the final grid's RGBA to a fixed target glyph. The net starts as
// a near-identity map (the linear update head is zero-initialised, the NCA trick)
// so the seed survives step one, then learns to GROW the target from that single
// live seed pixel.
//
// WHAT FIT THE CPU/MEMORY BUDGET (honest notes; see README for detail):
//   * Full BPTT through ALL T=32 unrolled shared steps DID fit -- it is stable and
//     fast (the headline ran well under five minutes on two cores). We did NOT need
//     truncated BPTT.
//   * Stability needed three guards, all standard for NCA: zero-init update head,
//     a bounded leaky activation clamping the state to [-10,10] after each step, and
//     gradient-norm clipping. Without them the 32-deep residual recurrence overflows
//     to NaN within one update.
//   * The stochastic per-cell update mask and the sample-replacement POOL from the
//     paper were dropped to keep the demo single-sample and deterministic; the
//     "alive" alpha>0.1 masking is approximated by the learned alpha channel itself.
//     The persistence/regeneration experiments are noted as out-of-budget in the
//     README rather than half-implemented.
//
// Coded by Claude (AI).
program NeuralCellularAutomata;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cW        = 16;     // grid width
  cH        = 16;     // grid height
  cCh       = 12;     // channels: 4 visible RGBA + 8 hidden state
  cPDim     = 24;     // perception channels (learned 3x3 conv)
  cHid      = 48;     // 1x1 ReLU hidden width
  cSteps    = 32;     // CA rule applications unrolled (full BPTT through all of them)
  cIters    = 600;    // training iterations
  cClamp    = 10;     // state bound per step (leaky ReLUL limits, integer)
  cLR       = 0.0006; // learning rate
  cGradClip = 0.05;   // global gradient-norm clip
  cAlive    = 0.1;    // alpha threshold for "alive" rendering

type
  TFloatGrid = array[0..cW - 1, 0..cH - 1] of TNeuralFloat;

var
  NN: TNNet;
  Seed, Target: TNNetVolume;
  StepGridIdx: array[1..cSteps] of integer; // layer index of each step's bounded grid

// The target glyph rasterised into RGBA (white-on-black "A").  Returns 1.0 where the
// glyph is, 0.0 elsewhere; same value used for R,G,B and alpha so the net grows a
// solid white shape on a transparent background.
function TargetMask(x, y: integer): TNeuralFloat;
const
  // 16x16 bitmap of a chunky letter 'A' (1 = ink).
  cGlyph: array[0..cH - 1] of string = (
    '0000000000000000',
    '0000000000000000',
    '0000011111100000',
    '0000111111110000',
    '0001110000111000',
    '0001100000011000',
    '0011100000011100',
    '0011100000011100',
    '0011111111111100',
    '0011111111111100',
    '0011100000011100',
    '0011100000011100',
    '0011100000011100',
    '0011100000011100',
    '0000000000000000',
    '0000000000000000'
  );
begin
  if cGlyph[y][x + 1] = '1' then Result := 1.0 else Result := 0.0;
end;

// Build the unrolled, weight-tied NCA. Returns the net; records each step's bounded
// grid layer index in StepGridIdx so we can read intermediate frames.
function BuildNCA(): TNNet;
var
  PerL, ReluL, LinL, Grid: TNNetLayer;
  step, i: integer;
begin
  Result := TNNet.Create();
  Grid := Result.AddLayer(TNNetInput.Create(cW, cH, cCh));
  // --- the ONE shared rule (real trainable layers, step 1) ---
  PerL  := Result.AddLayer(TNNetConvolutionLinear.Create(cPDim, 3, 1, 1, 1)); // 3x3, pad 1, no bias
  ReluL := Result.AddLayer(TNNetPointwiseConvReLU.Create(cHid));
  LinL  := Result.AddLayer(TNNetPointwiseConvLinear.Create(cCh, {SuppressBias=}1));
  // zero-init the update head: the CA starts as an identity map so the seed is not
  // immediately destroyed (the standard Growing-NCA initialisation).
  for i := 0 to LinL.Neurons.Count - 1 do
    LinL.Neurons[i].Weights.Fill(0);
  Grid := Result.AddLayer(TNNetSum.Create([Grid, LinL]));              // residual update
  Grid := Result.AddLayer(TNNetReLUL.Create(-cClamp, cClamp, 1));      // bound state (leaky)
  StepGridIdx[1] := Grid.LayerIdx;
  // --- steps 2..T reuse the SAME weights via shared-weight convs ---
  for step := 2 to cSteps do
  begin
    Result.AddLayerAfter(TNNetConvolutionSharedWeights.Create(PerL), Grid);
    Result.AddLayer(TNNetConvolutionSharedWeights.Create(ReluL));
    LinL := Result.AddLayer(TNNetConvolutionSharedWeights.Create(LinL));
    Grid := Result.AddLayer(TNNetSum.Create([Grid, LinL]));
    Grid := Result.AddLayer(TNNetReLUL.Create(-cClamp, cClamp, 1));
    StepGridIdx[step] := Grid.LayerIdx;
  end;
end;

// Print an ASCII rendering of a grid layer's alpha channel (channel 3): a cell is
// drawn from its alpha "aliveness", so we watch the shape grow out of the seed.
procedure RenderLayer(LayerIdx: integer; const Title: string);
const
  cRamp = ' .:-=+*#%@';
var
  V: TNNetVolume;
  x, y, lvl: integer;
  a: TNeuralFloat;
  line: string;
begin
  V := NN.Layers[LayerIdx].Output;
  WriteLn(Title);
  for y := 0 to cH - 1 do
  begin
    line := '';
    for x := 0 to cW - 1 do
    begin
      a := V[x, y, 3];                 // alpha / alive channel
      if a < cAlive then line := line + ' '
      else
      begin
        lvl := Round(a * (Length(cRamp) - 1));
        if lvl < 1 then lvl := 1;
        if lvl > Length(cRamp) - 1 then lvl := Length(cRamp) - 1;
        line := line + cRamp[lvl + 1];
      end;
    end;
    WriteLn('  |', line, '|');
  end;
end;

// L2 loss over the final grid's RGBA versus the target (and writes it into Target so
// Backpropagate sees zero error on the 8 hidden channels -> they stay free scratch).
function FillDesiredAndLoss(out Loss: TNeuralFloat): TNNetVolume;
var
  V: TNNetVolume;
  x, y, c: integer;
  d: TNeuralFloat;
begin
  V := NN.GetLastLayer.Output;
  Loss := 0;
  for y := 0 to cH - 1 do
    for x := 0 to cW - 1 do
    begin
      // visible RGBA channels: error against the target glyph
      for c := 0 to 3 do
      begin
        d := V[x, y, c] - Target[x, y, c];
        Loss := Loss + d * d;
      end;
      // hidden channels: desired = current output -> zero error, no gradient pull
      for c := 4 to cCh - 1 do
        Target[x, y, c] := V[x, y, c];
    end;
  Loss := Loss / (cW * cH * 4);
  Result := Target;
end;

var
  i: integer;
  Loss, Best: TNeuralFloat;
  x, y, c: integer;
begin
  RandSeed := 20200;
  WriteLn('=== Growing Neural Cellular Automata (16x16, ', cCh, ' channels) ===');
  WriteLn('shared rule unrolled T=', cSteps, ' steps; full BPTT through all of them');
  WriteLn;

  // Seed: a single live pixel at the centre (alpha=1, hidden channels seeded too).
  Seed := TNNetVolume.Create(cW, cH, cCh);
  Seed.Fill(0);
  Seed[cW div 2, cH div 2, 3] := 1.0; // alpha alive
  for c := 4 to cCh - 1 do
    Seed[cW div 2, cH div 2, c] := 1.0;

  // Target glyph (RGBA).
  Target := TNNetVolume.Create(cW, cH, cCh);
  Target.Fill(0);
  for y := 0 to cH - 1 do
    for x := 0 to cW - 1 do
      for c := 0 to 3 do
        Target[x, y, c] := TargetMask(x, y);

  NN := BuildNCA();
  WriteLn('trainable params (shared, independent of T) = ', NN.CountWeights());
  WriteLn('total layers in unrolled graph              = ', NN.CountLayers());
  WriteLn;

  NN.SetLearningRate(cLR, 0.9);
  NN.SetBatchUpdate(True);
  // flush the zero-initialised update head into every (shared) layer's weight cache
  // so the very first forward pass already sees the identity-map initialisation.
  NN.ClearDeltas;
  NN.UpdateWeights;

  WriteLn('target glyph:');
  for y := 0 to cH - 1 do
  begin
    Write('  |');
    for x := 0 to cW - 1 do
      if Target[x, y, 3] > 0.5 then Write('#') else Write(' ');
    WriteLn('|');
  end;
  WriteLn;

  Best := 1e30;
  WriteLn('training ', cIters, ' iterations...');
  for i := 0 to cIters - 1 do
  begin
    NN.ClearDeltas;
    NN.Compute(Seed);
    FillDesiredAndLoss(Loss);     // sets hidden-channel desired = current output
    NN.Backpropagate(Target);
    NN.NormalizeMaxAbsoluteDelta(cGradClip);
    NN.UpdateWeights;
    if Loss < Best then Best := Loss;
    if (i mod 50 = 0) or (i = cIters - 1) then
      WriteLn('  iter ', i:4, '  L2 loss = ', Loss:0:6);
  end;
  WriteLn;
  WriteLn('best L2 loss = ', Best:0:6);
  WriteLn;

  // Final forward pass for rendering, then show growth at a few unrolled steps.
  NN.Compute(Seed);
  WriteLn('seed -> growth (alpha channel, ', cSteps, ' shared steps):');
  WriteLn;
  RenderLayer(StepGridIdx[4],          'step 04:');
  WriteLn;
  RenderLayer(StepGridIdx[cSteps div 2], 'step ' + IntToStr(cSteps div 2) + ':');
  WriteLn;
  RenderLayer(StepGridIdx[cSteps],     'step ' + IntToStr(cSteps) + ' (final):');
  WriteLn;

  if Best < 0.05 then
    WriteLn('OK: the shared-weight CA grew the target glyph from a single seed pixel.')
  else
    WriteLn('NOTE: loss plateaued above 0.05; the glyph is approximate (see README).');

  Seed.Free;
  Target.Free;
  NN.Free;
end.
