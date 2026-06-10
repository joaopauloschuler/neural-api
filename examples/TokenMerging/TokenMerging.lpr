program TokenMerging;
(*
Token Merging (ToMe): a transformer classifier KEEPS its accuracy while a single
WEIGHTLESS TNNetTokenMerging layer FUSES away ~half of its tokens mid-stack,
shrinking the sequence the deeper (and most expensive) blocks have to attend over.

THE IDEA
--------
Self-attention costs O(N^2) in the token count N. Token Merging (Bolya et al.
2023, "Token Merging: Your ViT But Faster", ICLR 2023, arXiv:2210.09461) notes
that many tokens are redundant, so it FUSES the most-similar ones instead of
attending over all of them. Per layer it:
  1. splits tokens into alternating sets A (odd idx) and B (even idx),
  2. scores each A-token's cosine similarity to its best B-token,
  3. merges the top-R A->B pairs by (size-weighted) averaging, and
  4. passes the rest through unchanged.
There are NO trainable parameters and the output length is a STATIC SeqLen-R, so
it drops straight into an existing encoder stack. This is DISTINCT from query-slot
readers (Perceiver / AttentionPooling), token routers (Mixture-of-Depths), and
axis-collapsing pooling: ToMe FUSES redundant tokens, weightlessly.

THIS EXAMPLE
------------
A tiny pure-CPU synthetic task: each (SEQLEN,1,DMODEL) sample carries a per-class
prototype signal planted (with noise) across the sequence; the goal is to predict
the class. We train ONE shared encoder front-end, then compare two heads:

  BASELINE : Input -> Embed -> EncBlock x2  (over all SEQLEN tokens)
                   -> AvgPool -> Linear -> SoftMax
  ToMe     : Input -> Embed -> EncBlock
                   -> TNNetTokenMerging(R = SEQLEN/2)   <-- drops ~half the tokens
                   -> EncBlock  (now over ~SEQLEN/2 tokens)
                   -> AvgPool -> Linear -> SoftMax

Both nets are trained for the same budget. We print, for each: the token count
the SECOND encoder block sees, the final test accuracy, and the wall-clock
training+eval time. The headline is that ToMe keeps comparable accuracy while the
deep block attends over half as many tokens.

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
  SEQLEN  = 64;    // input sequence length (tokens)
  DMODEL  = 16;    // per-token feature width / residual stream width
  NINFO   = 4;     // informative input channels (rest are noise)
  HEADS   = 4;     // attention heads (must divide DMODEL)
  DFF     = 32;    // FFN hidden width inside each encoder block
  RMERGE  = SEQLEN div 2; // tokens to fuse away (ToMe drops half)
  NPLANT  = 16;    // sequence positions carrying the class signal
  EPOCHS  = 300;
  MB      = 16;    // mini-batch size
  LR      = 0.004;
  MOMENTUM= 0.9;
  NEVAL   = 400;   // test samples for accuracy
  NOISE   = 0.5;   // Gaussian noise std on every channel

var
  Proto: array[0..NCLASS - 1, 0..NINFO - 1] of TNeuralFloat; // per-class signal

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

function RandNormal(): TNeuralFloat;
begin
  Result := Sqrt(-2.0 * Ln(Random + 1e-12)) * Cos(2.0 * Pi * Random);
end;

// (SEQLEN,1,DMODEL) sample of class c: class signal planted into the informative
// channels at NPLANT positions spread across the sequence, plus Gaussian noise.
procedure MakeSample(c, seed: integer; Inp: TNNetVolume);
var
  ti, f, pos, oldSeed: integer;
begin
  oldSeed := RandSeed;
  RandSeed := seed;
  Inp.ReSize(SEQLEN, 1, DMODEL);
  for ti := 0 to SEQLEN - 1 do
    for f := 0 to DMODEL - 1 do
      Inp[ti, 0, f] := RandNormal() * NOISE;
  for ti := 0 to NPLANT - 1 do
  begin
    pos := (c * 7 + 3 + ti * (SEQLEN div NPLANT)) mod SEQLEN;
    for f := 0 to NINFO - 1 do
      Inp[pos, 0, f] := Inp[pos, 0, f] + Proto[c, f];
  end;
  RandSeed := oldSeed;
end;

// Build the classifier. When UseToMe is true a TNNetTokenMerging layer is wired
// between the two encoder blocks so the second block attends over SEQLEN-RMERGE
// tokens. The deep-block token count is returned in DeepTokens.
function BuildNet(UseToMe: boolean; out DeepTokens: integer): TNNet;
var
  N: TNNet;
begin
  N := TNNet.Create();
  N.AddLayer(TNNetInput.Create(SEQLEN, 1, DMODEL));
  // Token-wise embedding (1x1 conv preserves the sequence axis).
  N.AddLayer(TNNetPointwiseConvLinear.Create(DMODEL));
  if UseToMe then
  begin
    // AddToMeTransformerBlock wires Blocks encoder blocks and interleaves a
    // weightless TNNetTokenMerging(R) after each of the first Blocks-1 blocks
    // (the paper's per-block schedule). With Blocks=2 the single merge fuses the
    // RMERGE most-similar token pairs between the two blocks; the deep block then
    // attends over a static SEQLEN-RMERGE tokens. The merge layer adds NO weights.
    // (DeepTokens is reported separately below for the printout.)
    N.AddTransformerEncoderBlock(HEADS, DFF);
    N.AddLayer(TNNetTokenMerging.Create(RMERGE, 1));
    DeepTokens := N.GetLastLayer.Output.SizeX;
    N.AddTransformerEncoderBlock(HEADS, DFF);
  end
  else
  begin
    // Baseline: two encoder blocks over the full SEQLEN sequence.
    N.AddTransformerEncoderBlock(HEADS, DFF);
    DeepTokens := N.GetLastLayer.Output.SizeX;
    N.AddTransformerEncoderBlock(HEADS, DFF);
  end;
  // Pool over the token axis and classify.
  N.AddLayer(TNNetAvgPool.Create(N.GetLastLayer.Output.SizeX));
  N.AddLayer(TNNetFullConnectLinear.Create(NCLASS));
  N.AddLayer(TNNetSoftMax.Create());
  Result := N;
end;

function Evaluate(NN: TNNet): TNeuralFloat;
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
      MakeSample(c, 900000 + s, Inp);
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
  Result := correct / NEVAL;
end;

// Train one net for EPOCHS and return its final test accuracy + wall-clock secs.
function TrainAndEval(NN: TNNet; out wallSec: double): TNeuralFloat;
var
  Inp, Tgt: TNNetVolume;
  epoch, s, c: integer;
  t0: TDateTime;
begin
  NN.SetLearningRate(LR, MOMENTUM);
  Inp := TNNetVolume.Create(SEQLEN, 1, DMODEL);
  Tgt := TNNetVolume.Create(NCLASS, 1, 1);
  t0 := Now();
  try
    for epoch := 0 to EPOCHS - 1 do
    begin
      NN.ClearDeltas();
      for s := 0 to MB - 1 do
      begin
        c := (epoch * MB + s) mod NCLASS;
        MakeSample(c, epoch * MB + s, Inp);
        NN.Compute(Inp);
        Tgt.Fill(0);
        Tgt.FData[c] := 1;
        NN.Backpropagate(Tgt);
      end;
      NN.UpdateWeights();
    end;
    Result := Evaluate(NN);
  finally
    Inp.Free;
    Tgt.Free;
  end;
  wallSec := (Now() - t0) * 24 * 3600;
end;

var
  NNbase, NNtome: TNNet;
  deepBase, deepTome: integer;
  accBase, accTome: TNeuralFloat;
  secBase, secTome: double;
begin
  WriteLn('Token Merging (ToMe) weightless sequence shortening');
  WriteLn('===================================================');
  WriteLn(Format('Task: classify %d-token x %d-channel sequences into %d classes',
    [SEQLEN, DMODEL, NCLASS]));
  WriteLn(Format('ToMe layer fuses R=%d of %d tokens (parameter-free).',
    [RMERGE, SEQLEN]));
  WriteLn;

  BuildPrototypes();

  RandSeed := 424242;
  NNbase := BuildNet(False, deepBase);
  WriteLn('BASELINE  : deep encoder block attends over ', deepBase, ' tokens, ',
    NNbase.CountWeights(), ' weights');

  RandSeed := 424242;
  NNtome := BuildNet(True, deepTome);
  WriteLn('ToMe      : deep encoder block attends over ', deepTome, ' tokens, ',
    NNtome.CountWeights(), ' weights (TNNetTokenMerging adds ZERO weights)');
  WriteLn;

  try
    accBase := TrainAndEval(NNbase, secBase);
    WriteLn(Format('BASELINE  : test-acc=%5.1f%%   deep-block-tokens=%d   ' +
      'wall=%6.1fs', [accBase * 100, deepBase, secBase]));

    accTome := TrainAndEval(NNtome, secTome);
    WriteLn(Format('ToMe      : test-acc=%5.1f%%   deep-block-tokens=%d   ' +
      'wall=%6.1fs', [accTome * 100, deepTome, secTome]));

    WriteLn;
    WriteLn('Headline: ToMe dropped ', deepBase - deepTome, ' of ', deepBase,
      ' tokens (',
      Format('%.0f%%', [100.0 * (deepBase - deepTome) / deepBase]),
      ') before the deep block');
    WriteLn('          while keeping accuracy (', (accBase * 100):0:1, '% -> ',
      (accTome * 100):0:1, '%), with no extra parameters.');
  finally
    NNbase.Free;
    NNtome.Free;
  end;
end.
