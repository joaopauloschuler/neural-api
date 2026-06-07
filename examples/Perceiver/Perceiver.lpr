program Perceiver;
(*
Perceiver: a LATENT-BOTTLENECK encoder classifies a DELIBERATELY LONG input
while the parameter count and per-step compute are governed by a small latent
tower, NOT by the input length.

THE IDEA
--------
A standard transformer encoder over an N-token sequence costs O(N^2) attention
and stacks blocks directly on the N tokens, so depth gets expensive fast on long
inputs. The Perceiver (Jaegle et al. 2021, "Perceiver: General Perception with
Iterative Attention", arXiv:2103.03206; Perceiver IO, arXiv:2107.14795)
decouples depth from length with a SMALL, FIXED learnable LATENT array Z of
NumLatents rows (NumLatents << N):

  1. CROSS-ATTENTION READ: the NumLatents latents cross-attend ONCE over the N
     input tokens, absorbing the whole sequence into a (NumLatents,1,d_latent)
     summary. This is the only step that touches the input; its cost is LINEAR
     in N (one NumLatents x N attention map).
  2. LATENT TOWER: a stack of Depth self-attention + FFN blocks refines the
     latents, acting ONLY over the NumLatents rows -- O(NumLatents^2) per block,
     completely independent of N.

So you can pour a 1000-token (or 50k-pixel) input through a DEEP tower whose
weight count and per-block compute never grow with the input length.

THIS EXAMPLE
------------
A tiny pure-CPU synthetic classification task makes the length-independence
visible. There are NCLASS prototype patterns living in only the FIRST few
"informative" channels of each token; the rest of the (long) sequence is noise.
Each sample is a (SEQLEN,1,DMODEL) tensor: a class-dependent signal scattered
across a LONG sequence plus Gaussian noise. The net is

  Input(SEQLEN,1,DMODEL)
    -> AddPerceiverEncoder(NumLatents=NLAT, d_latent=DLAT, Heads=HEADS,
                           Depth=DEPTH)
         which wires, in one call:
           (optional) PointwiseConvLinear(DLAT)   project input width -> d_latent
           AddAttentionPooling(NLAT, HEADS)        latent CROSS-ATTENTION read:
                                                   the learnable seed bank IS the
                                                   Perceiver latent array Z; it
                                                   collapses SEQLEN rows to NLAT
           AddTransformerEncoderBlock(...) x DEPTH latent SELF-attention tower
    -> [flatten NLAT latents] -> FullConnectLinear(NCLASS) -> SoftMax

HEADLINE PAYOFF
---------------
Before training we build the SAME net on TWO input lengths (SEQLEN and 2*SEQLEN)
and print the trainable-weight count of each: they are IDENTICAL. Doubling the
input length does NOT add a single weight -- the cost lives in the NLAT-row
latent tower, not the input. Then we train on the long input and report the
accuracy climb (well above chance) within a small CPU budget.

Pure CPU, tiny dimensions and batches -> runs in well under 5 minutes on 2 cores
with modest memory. No binaries are committed.

LICENSE: GPL (same as the neural-api project).

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, neuralnetwork, neuralvolume;

const
  NCLASS  = 4;     // number of classes / prototypes
  SEQLEN  = 256;   // DELIBERATELY LONG input sequence length (tokens)
  DMODEL  = 8;     // per-token feature width (input channels)
  NINFO   = 3;     // informative channels (rest are noise)
  NLAT    = 16;    // number of latents (NLAT << SEQLEN)
  DLAT    = 16;    // latent width
  HEADS   = 4;     // attention heads (must divide DLAT)
  DEPTH   = 2;     // latent self-attention blocks
  DFF     = 32;    // FFN hidden width inside each tower block
  NPLANT  = 24;    // how many sequence positions carry the class signal
  EPOCHS  = 400;
  MB      = 16;    // mini-batch size
  LR      = 0.004;
  MOMENTUM= 0.9;
  NEVAL   = 400;   // test samples for accuracy
  NOISE   = 0.4;   // Gaussian noise std on every channel

var
  NN: TNNet;
  Proto: array[0..NCLASS - 1, 0..NINFO - 1] of TNeuralFloat; // per-class signal

// Build NCLASS fixed prototype signal vectors (over the informative channels).
procedure BuildPrototypes();
var
  c, f, oldSeed: integer;
begin
  oldSeed := RandSeed;
  RandSeed := 13579;
  for c := 0 to NCLASS - 1 do
    for f := 0 to NINFO - 1 do
      Proto[c, f] := (Random - 0.5) * 4.0;
  RandSeed := oldSeed;
end;

// Box-Muller standard normal.
function RandNormal(): TNeuralFloat;
begin
  Result := Sqrt(-2.0 * Ln(Random + 1e-12)) * Cos(2.0 * Pi * Random);
end;

// Make a (pLen,1,DMODEL) sample of class c: the class signal is planted into the
// informative channels at a class-shifted position, everything else is noise.
procedure MakeSample(c, seed, pLen: integer; Inp: TNNetVolume);
var
  ti, f, pos, oldSeed: integer;
begin
  oldSeed := RandSeed;
  RandSeed := seed;
  Inp.ReSize(pLen, 1, DMODEL);
  for ti := 0 to pLen - 1 do
    for f := 0 to DMODEL - 1 do
      Inp[ti, 0, f] := RandNormal() * NOISE;
  // Plant the class signal into NINFO informative channels at NPLANT positions
  // SPREAD across the LONG sequence (not clustered at the front), so the latent
  // read has to gather class evidence from all over the input.
  for ti := 0 to NPLANT - 1 do
  begin
    pos := (c * 37 + 5 + ti * (pLen div NPLANT)) mod pLen;
    for f := 0 to NINFO - 1 do
      Inp[pos, 0, f] := Inp[pos, 0, f] + Proto[c, f];
  end;
  RandSeed := oldSeed;
end;

// Build a Perceiver classifier over a pLen-long input. Returns the net.
function BuildNet(pLen: integer): TNNet;
var
  N: TNNet;
begin
  N := TNNet.Create();
  N.AddLayer(TNNetInput.Create(pLen, 1, DMODEL));
  // The latent-bottleneck encoder: output is (NLAT,1,DLAT) REGARDLESS of pLen.
  N.AddPerceiverEncoder(NLAT, DLAT, HEADS, DEPTH, DFF);
  // Classification head over the flattened latents.
  N.AddLayer(TNNetFullConnectLinear.Create(NCLASS));
  N.AddLayer(TNNetSoftMax.Create());
  Result := N;
end;

// Evaluate test accuracy on freshly-drawn samples.
procedure Evaluate(out acc: TNeuralFloat);
var
  s, c, predicted, correct, k: integer;
  best: TNeuralFloat;
  Inp: TNNetVolume;
begin
  Inp := TNNetVolume.Create(SEQLEN, 1, DMODEL);
  correct := 0;
  try
    for s := 0 to NEVAL - 1 do
    begin
      c := s mod NCLASS;
      MakeSample(c, 900000 + s, SEQLEN, Inp);
      NN.Compute(Inp);
      predicted := 0; best := NN.GetLastLayer.Output.FData[0];
      for k := 1 to NCLASS - 1 do
        if NN.GetLastLayer.Output.FData[k] > best then
        begin best := NN.GetLastLayer.Output.FData[k]; predicted := k; end;
      if predicted = c then Inc(correct);
    end;
  finally
    Inp.Free;
  end;
  acc := correct / NEVAL;
end;

var
  NNshort, NNlong: TNNet;
  wShort, wLong: integer;
  Inp, Tgt: TNNetVolume;
  epoch, s, c: integer;
  acc, accBefore: TNeuralFloat;
begin
  WriteLn('Perceiver latent-bottleneck encoder');
  WriteLn('===================================');
  WriteLn(Format('Input: %d tokens x %d channels  ->  %d latents x %d width  ' +
    '(%d heads, %d tower blocks)', [SEQLEN, DMODEL, NLAT, DLAT, HEADS, DEPTH]));
  WriteLn;

  BuildPrototypes();

  // HEADLINE: weight count is INPUT-LENGTH INDEPENDENT.
  NNshort := BuildNet(SEQLEN);
  NNlong  := BuildNet(2 * SEQLEN);
  wShort := NNshort.CountWeights();
  wLong  := NNlong.CountWeights();
  WriteLn('Trainable weights @ SEQLEN=', SEQLEN, '   : ', wShort);
  WriteLn('Trainable weights @ SEQLEN=', 2 * SEQLEN, '   : ', wLong);
  if wShort = wLong then
    WriteLn('  -> DOUBLING the input length added ZERO weights: the cost lives ',
      'in the latent tower, not the input.')
  else
    WriteLn('  -> (unexpected) weight counts differ.');
  NNshort.Free;
  NNlong.Free;
  WriteLn;

  // Train the classifier on the LONG input.
  NN := BuildNet(SEQLEN);
  NN.SetLearningRate(LR, MOMENTUM);
  NN.DebugWeights();

  Inp := TNNetVolume.Create(SEQLEN, 1, DMODEL);
  Tgt := TNNetVolume.Create(NCLASS, 1, 1);
  try
    Evaluate(accBefore);
    WriteLn('Test accuracy BEFORE training: ', (accBefore * 100):0:1,
      '%  (chance = ', (100.0 / NCLASS):0:1, '%)');
    WriteLn;

    for epoch := 0 to EPOCHS - 1 do
    begin
      NN.ClearDeltas();
      for s := 0 to MB - 1 do
      begin
        c := (epoch * MB + s) mod NCLASS;
        MakeSample(c, epoch * MB + s, SEQLEN, Inp);
        NN.Compute(Inp);
        Tgt.Fill(0);
        Tgt.FData[c] := 1;
        NN.Backpropagate(Tgt);
      end;
      NN.UpdateWeights();

      if (epoch mod 50 = 0) or (epoch = EPOCHS - 1) then
      begin
        Evaluate(acc);
        WriteLn(Format('  epoch %4d  test-acc=%5.1f%%', [epoch, acc * 100]));
      end;
    end;

    Evaluate(acc);
    WriteLn;
    WriteLn('Test accuracy AFTER training: ', (acc * 100):0:1, '%');
    WriteLn('Headline: a Perceiver classified a ', SEQLEN, '-token input, ',
      'climbing from ', (accBefore * 100):0:1, '% to ', (acc * 100):0:1, '%,');
    WriteLn('          with ', wShort, ' weights that do NOT grow with input ',
      'length (the latent tower carries the depth).');
  finally
    Inp.Free;
    Tgt.Free;
    NN.Free;
  end;
end.
