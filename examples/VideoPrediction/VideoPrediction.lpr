program VideoPrediction;
(*
VideoPrediction: the repo's first SPATIOTEMPORAL (video) example - next-frame
prediction in the Moving-MNIST style (Srivastava et al. 2015), but with a
self-contained, download-free synthetic dataset: a small bright BLOB that
translates across a grid at constant velocity, bouncing off the walls. The
model watches the first N frames of a clip and must predict the (N+1)-th frame.

This is the showcase for the new TNNetConvLSTMCell layer - a CONVOLUTIONAL
LSTM (Shi et al. 2015, "Convolutional LSTM Network", arXiv:1506.04214): the
spatial, image-state analogue of the dense recurrent cells
(TNNetMinLSTM / TNNetSLSTMCell / TNNetMLSTMCell). Instead of a vector hidden
state it carries (H,W,HiddenC) feature MAPS and replaces every gate
matrix-multiply with a K x K same-padding convolution over the channel
concatenation [x_t ; h_{t-1}]. That convolutional gating is exactly what lets
the recurrence track WHERE the blob is and WHERE it is heading - a dense LSTM
would have to flatten the frame and throw the spatial layout away.

THE DATA (synthetic, generated in code - no download)
-----------------------------------------------------
Each clip is N+1 frames of a single GRID x GRID grayscale image. A blob (a
small filled square, optionally feathered) starts at a random position with a
random integer velocity (vx,vy in {-1,0,+1}, not both zero) and moves one cell
per frame, reflecting off the borders. Pixels are encoded in [-1,1] (background
-1, blob centre +1) to match the model's Tanh output range. The first N frames
are the INPUT sequence; the final frame is the TARGET to predict.

THE MODEL (ConvLSTM encoder -> last-state decode head)
------------------------------------------------------
  Input(N*GRID, GRID, 1, 1)                 -- N frames packed along the X axis
    -> TNNetConvLSTMCell(N, HiddenC, 3)     -- per-step hidden maps (N*GRID,GRID,HiddenC)
    -> TNNetCrop((N-1)*GRID, 0, GRID, GRID) -- keep ONLY the last timestep's map
    -> TNNetConvolutionLinear(1, 3, 1, 1)   -- 3x3 conv head -> 1 channel
    -> TNNetHyperbolicTangent                -- predicted next frame in [-1,1]
The ConvLSTM packs the N input frames stacked on the X axis as (N*GRID,GRID,1);
its output is the N hidden maps stacked the same way, and we crop the LAST one
(the post-sequence summary) and project it to a single predicted frame.

TRAINING is plain supervised regression: Compute(inputFrames);
Backpropagate(nextFrame), MSE on the predicted vs. true next frame.

OUTPUT
------
Held-out reconstruction MSE/MAE over a fresh rollout set, plus an ASCII panel
(input frames | predicted | ground-truth) and a PPM dump of one held-out clip.

The default SMOKE run finishes well under five minutes on CPU; pass --full for
a longer run (more clips / epochs / hidden channels) and a sharper prediction.

  Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cGrid    = 12;   // frame is cGrid x cGrid grayscale
  cSeqLen  = 4;    // number of INPUT frames the model watches
  cBlobR   = 1;    // blob "radius" (half-side of the filled square)

var
  gHiddenC : integer = 8;    // ConvLSTM hidden channels
  gNumTrain: integer = 240;  // training clips
  gNumTest : integer = 40;   // held-out clips
  gEpochs  : integer = 6;    // passes over the training set
  gLR      : single  = 0.002;

// ---------------------------------------------------------------------------
// Synthetic moving-blob clip generator.
// A clip is (cSeqLen+1) frames; frame t is written into Dst at X-row offset
// t*cGrid (frames stacked on the X axis, the layout TNNetConvLSTMCell expects).
// ---------------------------------------------------------------------------
procedure MakeClip(Frames: TNNetVolume);
var
  px, py, vx, vy, t, dx, dy, x, y, fx: integer;
  v: single;
begin
  Frames.Fill(-1.0);  // background
  // Random start + non-zero integer velocity.
  px := cBlobR + Random(cGrid - 2 * cBlobR);
  py := cBlobR + Random(cGrid - 2 * cBlobR);
  repeat
    vx := Random(3) - 1;
    vy := Random(3) - 1;
  until (vx <> 0) or (vy <> 0);
  for t := 0 to cSeqLen do
  begin
    // Draw the blob at (px,py) into frame t.
    for dy := -cBlobR to cBlobR do
      for dx := -cBlobR to cBlobR do
      begin
        x := px + dx; y := py + dy;
        if (x >= 0) and (x < cGrid) and (y >= 0) and (y < cGrid) then
        begin
          if (dx = 0) and (dy = 0) then v := 1.0 else v := 0.4; // centre brighter
          fx := t * cGrid + x;
          if v > Frames[fx, y, 0] then Frames[fx, y, 0] := v;
        end;
      end;
    // Advance with wall reflection.
    px := px + vx; py := py + vy;
    if px < cBlobR then begin px := cBlobR; vx := -vx; end;
    if px > cGrid - 1 - cBlobR then begin px := cGrid - 1 - cBlobR; vx := -vx; end;
    if py < cBlobR then begin py := cBlobR; vy := -vy; end;
    if py > cGrid - 1 - cBlobR then begin py := cGrid - 1 - cBlobR; vy := -vy; end;
  end;
end;

// Split a clip volume into the N-frame input (Inp) and the next-frame target (Tgt).
procedure SplitClip(Clip, Inp, Tgt: TNNetVolume);
var x, y: integer;
begin
  Inp.Copy(Clip, cSeqLen * cGrid * cGrid); // first N frames (X rows 0..N*G-1)
  for y := 0 to cGrid - 1 do
    for x := 0 to cGrid - 1 do
      Tgt[x, y, 0] := Clip[cSeqLen * cGrid + x, y, 0];
end;

function BuildModel(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen * cGrid, cGrid, 1, 1));
  Result.AddLayer(TNNetConvLSTMCell.Create(cSeqLen, gHiddenC, 3));
  // Keep only the LAST timestep's hidden map (the post-sequence summary).
  Result.AddLayer(TNNetCrop.Create((cSeqLen - 1) * cGrid, 0, cGrid, cGrid));
  Result.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1)); // 3x3 -> 1 channel
  Result.AddLayer(TNNetHyperbolicTangent.Create());           // [-1,1]
end;

// ---------------------------------------------------------------------------
// Metrics + rendering.
// ---------------------------------------------------------------------------
procedure EvalSet(NN: TNNet; const Clips: array of TNNetVolume; Count: integer;
  out MSE, MAE: double);
var
  i, x, y: integer;
  Inp, Tgt, Pred: TNNetVolume;
  d, sse, sae: double;
begin
  Inp := TNNetVolume.Create(cSeqLen * cGrid, cGrid, 1);
  Tgt := TNNetVolume.Create(cGrid, cGrid, 1);
  sse := 0; sae := 0;
  for i := 0 to Count - 1 do
  begin
    SplitClip(Clips[i], Inp, Tgt);
    NN.Compute(Inp);
    Pred := NN.GetLastLayer.Output;
    for y := 0 to cGrid - 1 do
      for x := 0 to cGrid - 1 do
      begin
        d := Pred[x, y, 0] - Tgt[x, y, 0];
        sse := sse + d * d;
        sae := sae + Abs(d);
      end;
  end;
  MSE := sse / (Count * cGrid * cGrid);
  MAE := sae / (Count * cGrid * cGrid);
  Inp.Free; Tgt.Free;
end;

function Glyph(v: single): char;
begin
  // v in [-1,1] -> ramp.
  if v < -0.6 then Result := ' '
  else if v < -0.1 then Result := '.'
  else if v < 0.4 then Result := ':'
  else if v < 0.8 then Result := '+'
  else Result := '#';
end;

procedure RenderPanel(NN: TNNet; Clip: TNNetVolume);
var
  Inp, Tgt, Pred: TNNetVolume;
  x, y, t: integer;
  line: string;
begin
  Inp := TNNetVolume.Create(cSeqLen * cGrid, cGrid, 1);
  Tgt := TNNetVolume.Create(cGrid, cGrid, 1);
  SplitClip(Clip, Inp, Tgt);
  NN.Compute(Inp);
  Pred := NN.GetLastLayer.Output;
  WriteLn;
  WriteLn('ASCII panel: ', cSeqLen, ' input frames | PREDICTED | GROUND-TRUTH');
  for y := 0 to cGrid - 1 do
  begin
    line := '';
    for t := 0 to cSeqLen - 1 do
    begin
      for x := 0 to cGrid - 1 do line := line + Glyph(Inp[t * cGrid + x, y, 0]);
      line := line + ' ';
    end;
    line := line + '| ';
    for x := 0 to cGrid - 1 do line := line + Glyph(Pred[x, y, 0]);
    line := line + ' | ';
    for x := 0 to cGrid - 1 do line := line + Glyph(Tgt[x, y, 0]);
    WriteLn(line);
  end;
  Inp.Free; Tgt.Free;
end;

procedure DumpPPM(NN: TNNet; Clip: TNNetVolume; const FileName: string);
var
  Inp, Tgt, Pred: TNNetVolume;
  f: TextFile;
  panelW, x, y, t, gx: integer;
  v: single; b: integer;
begin
  Inp := TNNetVolume.Create(cSeqLen * cGrid, cGrid, 1);
  Tgt := TNNetVolume.Create(cGrid, cGrid, 1);
  SplitClip(Clip, Inp, Tgt);
  NN.Compute(Inp);
  Pred := NN.GetLastLayer.Output;
  // Layout: N input frames, then predicted, then ground-truth, side by side.
  panelW := (cSeqLen + 2) * cGrid;
  AssignFile(f, FileName); Rewrite(f);
  WriteLn(f, 'P3'); WriteLn(f, panelW, ' ', cGrid); WriteLn(f, '255');
  for y := 0 to cGrid - 1 do
    for gx := 0 to panelW - 1 do
    begin
      x := gx mod cGrid;
      if gx < cSeqLen * cGrid then
      begin
        t := gx div cGrid;
        v := Inp[t * cGrid + x, y, 0];
        b := Round((v + 1) * 0.5 * 255);
        WriteLn(f, b, ' ', b, ' ', b);          // grayscale inputs
      end
      else if gx < (cSeqLen + 1) * cGrid then
      begin
        v := Pred[x, y, 0];
        b := Round((v + 1) * 0.5 * 255);
        WriteLn(f, 0, ' ', b, ' ', 0);           // predicted -> green
      end
      else
      begin
        v := Tgt[x, y, 0];
        b := Round((v + 1) * 0.5 * 255);
        WriteLn(f, b, ' ', 0, ' ', 0);           // ground-truth -> red
      end;
    end;
  CloseFile(f);
  Inp.Free; Tgt.Free;
  WriteLn('Wrote ', FileName, ' (', panelW, 'x', cGrid,
    ' : inputs gray | prediction green | truth red)');
end;

// ---------------------------------------------------------------------------
var
  NN: TNNet;
  Train, Test: array of TNNetVolume;
  Inp, Tgt: TNNetVolume;
  i, ep, idx, order, tmp: integer;
  Perm: array of integer;
  MSE, MAE, MSE0: double;
  StartTime: double;
  a: integer;
begin
  Randomize; RandSeed := 20150604;  // arXiv id of the ConvLSTM paper :)

  for a := 1 to ParamCount do
    if ParamStr(a) = '--full' then
    begin
      gHiddenC  := 16;
      gNumTrain := 800;
      gNumTest  := 80;
      gEpochs   := 16;
      WriteLn('Running in --full mode (longer, sharper).');
    end;

  WriteLn('=== VideoPrediction: ConvLSTM next-frame prediction (Moving-blob) ===');
  WriteLn(Format('grid=%dx%d  seqLen=%d  hiddenC=%d  train=%d  test=%d  epochs=%d',
    [cGrid, cGrid, cSeqLen, gHiddenC, gNumTrain, gNumTest, gEpochs]));

  // Build datasets.
  SetLength(Train, gNumTrain);
  SetLength(Test, gNumTest);
  for i := 0 to gNumTrain - 1 do
  begin
    Train[i] := TNNetVolume.Create((cSeqLen + 1) * cGrid, cGrid, 1);
    MakeClip(Train[i]);
  end;
  for i := 0 to gNumTest - 1 do
  begin
    Test[i] := TNNetVolume.Create((cSeqLen + 1) * cGrid, cGrid, 1);
    MakeClip(Test[i]);
  end;

  NN := BuildModel();
  NN.SetLearningRate(gLR, 0.9);
  WriteLn('Model layers: ', NN.CountLayers);

  Inp := TNNetVolume.Create(cSeqLen * cGrid, cGrid, 1);
  Tgt := TNNetVolume.Create(cGrid, cGrid, 1);

  EvalSet(NN, Test, gNumTest, MSE0, MAE);
  WriteLn(Format('Held-out BEFORE training: MSE=%.5f  MAE=%.5f', [MSE0, MAE]));

  SetLength(Perm, gNumTrain);
  for i := 0 to gNumTrain - 1 do Perm[i] := i;

  StartTime := Now();
  for ep := 1 to gEpochs do
  begin
    // Shuffle.
    for i := gNumTrain - 1 downto 1 do
    begin
      order := Random(i + 1);
      tmp := Perm[i]; Perm[i] := Perm[order]; Perm[order] := tmp;
    end;
    for i := 0 to gNumTrain - 1 do
    begin
      idx := Perm[i];
      SplitClip(Train[idx], Inp, Tgt);
      NN.Compute(Inp);
      NN.Backpropagate(Tgt);   // MSE-style supervised regression
    end;
    EvalSet(NN, Test, gNumTest, MSE, MAE);
    WriteLn(Format('epoch %2d/%2d  held-out MSE=%.5f  MAE=%.5f',
      [ep, gEpochs, MSE, MAE]));
  end;
  WriteLn(Format('Training wall time: %.1f s', [(Now() - StartTime) * 86400]));

  EvalSet(NN, Test, gNumTest, MSE, MAE);
  WriteLn;
  WriteLn(Format('FINAL held-out: MSE=%.5f  MAE=%.5f  (MSE improved %.1f%% vs init)',
    [MSE, MAE, 100.0 * (MSE0 - MSE) / Max(MSE0, 1e-9)]));

  RenderPanel(NN, Test[0]);
  DumpPPM(NN, Test[0], 'videoprediction_sample.ppm');

  // Cleanup.
  Inp.Free; Tgt.Free; NN.Free;
  for i := 0 to gNumTrain - 1 do Train[i].Free;
  for i := 0 to gNumTest - 1 do Test[i].Free;
  WriteLn('Done.');
end.
