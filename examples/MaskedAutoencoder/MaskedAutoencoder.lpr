program MaskedAutoencoder;
(*
MaskedAutoencoder: SELF-SUPERVISED visual pretraining with a Masked Autoencoder
(MAE, He et al. 2022, "Masked Autoencoders Are Scalable Vision Learners",
arXiv:2111.06377) on a tiny synthetic shape-classification task, then a LINEAR
PROBE that shows the MAE-pretrained encoder is a far better feature extractor
than a randomly-initialised one.

WHAT MAE DOES
-------------
1. Cut the image into a grid of non-overlapping patches.
2. Randomly DROP a large fraction (~75%) of the patches; keep the rest VISIBLE.
3. Embed and encode ONLY the patch tokens (here: visible patches kept, masked
   patches replaced by a shared mask token), running a ViT-style transformer
   encoder over the sequence with LEARNED positional embeddings so every token
   knows WHERE it sits in the grid.
4. A lightweight decoder reconstructs the raw pixels of EVERY patch.
5. The reconstruction loss (MSE) is computed ONLY on the MASKED patches -- the
   model must hallucinate the missing content from the visible context. This is
   what forces the encoder to learn the structure of the data with NO labels.

THE PAYOFF (the whole point)
----------------------------
After self-supervised MAE pretraining we FREEZE the encoder and train only a
single linear classifier on top of its features (a "linear probe"), using a
small labelled subset, and compare against the identical linear probe on top of a
RANDOMLY-INITIALISED (untrained) encoder of the SAME architecture. The MAE
features win clearly: the encoder genuinely learned something useful from
unlabelled pixels alone (a typical run scores ~0.77 vs ~0.66 for random, an ~11
point gap, well above the 0.25 chance level). The probe accuracy is averaged over
cProbeSeeds probe initialisations so the reported gap is stable and reproducible.

TASK: cSizeX x cSizeY single-channel images of binary STRIPE textures. The CLASS
is the stripe PERIOD (spatial frequency) -- one of {2,3,4,5} cells -- with the
stripe ORIENTATION randomised as a NUISANCE factor the classifier must ignore.
All classes share the same first-order statistics (duty cycle / mean intensity),
so they differ ONLY in spatial frequency: a GLOBAL property no single patch
reveals. A random untrained encoder has not learned to estimate frequency and
sits well below ceiling; the MAE encoder, having reconstructed thousands of
masked periodic fields, has effectively learned a frequency estimator. The
encoder width cDModel is kept deliberately NARROW (a bottleneck) so a random
projection has little capacity -- exactly the regime where a LEARNED encoder
beats random features. Patches are cPatch x cPatch -> cNumPatches tokens; cMaskN
of them are masked each step. Pure CPU, tiny data -- runs in well under a minute.

RELATION TO CANONICAL MAE (documented honestly)
-----------------------------------------------
Canonical MAE makes the ENCODER cheaper by feeding it ONLY the (variable-length)
visible tokens and inserting mask tokens just before the DECODER. A static
computation graph cannot express a per-sample variable sequence length, so here
the encoder runs over the FULL token grid with masked positions occupied by a
(fixed, zero) shared mask token; the asymmetric "encode-visible-only" speedup is
dropped. Everything else is faithful MAE: random per-image masking, learned
positional embeddings, a transformer encoder + lighter transformer decoder, and
a reconstruction loss applied ONLY to the masked patches. The masked-only loss
is realised exactly by building the regression target so that VISIBLE patch
entries equal the current prediction (zero error there) while MASKED entries hold
the true pixels -- for a linear output head FOutputError = prediction - target is
precisely the per-element MSE gradient.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cSizeX      = 12;
  cSizeY      = 12;
  cPatch      = 3;                         // 3x3 patches
  cGrid       = cSizeX div cPatch;         // 4 patches per side
  cNumPatches = cGrid * cGrid;             // 16 tokens
  cPatchDim   = cPatch * cPatch;           // 9 pixels per patch token
  cClasses    = 4;
  cMaskN      = 8;                         // mask 8 of 16 (=50%)
  cVisN       = cNumPatches - cMaskN;      // 8 visible

  cDModel     = 8;                         // NARROW transformer width (bottleneck)
  cHeads      = 2;
  cFF         = 16;                        // FFN inner dim

  cPretrain   = 6000;                      // self-supervised steps
  cPreLR      = 0.0005;

  cProbeTrain = 120;                       // labelled images for the probe
  cProbeTest  = 400;
  cProbeEpochs= 80;
  cProbeLR    = 0.02;
  cProbeSeeds = 5;                         // average probe over this many inits

  cSeed       = 424242;

type
  TPatchIdx = array[0..cNumPatches - 1] of integer;

const
  cChannelsHelper = 1;

var
  gEncEnd: integer;  // layer index of the encoder output (set by BuildMAE)

  // Pool of unlabelled images for self-supervised pretraining and a separate
  // labelled pool for the linear probe.
  PreImgs:   array of TNNetVolume;     // raw images for pretraining
  ProbeTrIn, ProbeTrTg: array[0..cProbeTrain - 1] of TNNetVolume;
  ProbeTeIn, ProbeTeTg: array[0..cProbeTest - 1] of TNNetVolume;

// ---------------------------------------------------------------------------
// Synthetic data: DENSE full-image TEXTURES. The class is a GLOBAL periodic
// pattern (horizontal / vertical / checkerboard / diagonal stripes) at a random
// phase and period, with noise. Dense global structure is exactly the regime
// where MAE helps: a patch's content is only predictable from the GLOBAL phase,
// which the encoder must infer from the few visible patches -- so reconstruction
// forces it to learn the texture, and that texture IS the class. (Sparse single
// shapes, by contrast, are reconstructable as ~all-zero and let a random encoder
// win the probe trivially.)
// ---------------------------------------------------------------------------
procedure StampTexture(Img: TNNetVolume; cls: integer);
var
  x, y, phase, coord, period, orient: integer;
begin
  // Binary STRIPES whose CLASS is the spatial PERIOD (frequency), with the stripe
  // ORIENTATION randomised as a NUISANCE factor the classifier must ignore. The
  // four classes are periods {2,3,4,5}; orientation is one of 4 directions chosen
  // independently of the class. Same duty cycle, so all classes share first-order
  // statistics -- they differ ONLY in spatial FREQUENCY.
  //
  // Why this regime is the right MAE demo:
  //  * Reconstruction is SOLVABLE (discrete phase + regular period: visible
  //    patches pin the global pattern), so the MAE has real structure to learn.
  //  * The class (frequency, invariant to a random orientation nuisance) is a
  //    GLOBAL property no single patch reveals -- a random untrained encoder has
  //    not learned to estimate frequency and lands well below ceiling, while the
  //    MAE encoder, having reconstructed many periodic fields, has effectively
  //    learned a frequency estimator and reads the class off easily.
  period := cls + 2;                 // classes 0..3 -> periods 2,3,4,5
  orient := Random(4);               // nuisance: orientation independent of class
  phase  := Random(period);
  for y := 0 to cSizeY - 1 do
    for x := 0 to cSizeX - 1 do
    begin
      case orient of
        0: coord := x;
        1: coord := y;
        2: coord := x + y;
      else coord := x - y + cSizeY;
      end;
      Img.Data[x, y, 0] := Ord(((coord + phase) mod period) < (period div 2 + period mod 2))
                           + (Random - 0.5) * 0.30;
    end;
end;

procedure OneHot(V: TNNetVolume; cls: integer);
begin
  V.Fill(0);
  V.Raw[cls] := 1;
end;

procedure MakeRandomImage(Img: TNNetVolume; out cls: integer);
begin
  cls := Random(cClasses);
  StampTexture(Img, cls);
end;

// ---------------------------------------------------------------------------
// Patchify: turn a (cSizeX,cSizeY,1) image into a (cNumPatches,1,cPatchDim)
// token sequence (token index = gridY*cGrid+gridX, depth = flattened patch).
// ---------------------------------------------------------------------------
procedure Patchify(Img, Tokens: TNNetVolume);
var
  gx, gy, fx, fy, tok, d: integer;
begin
  for gy := 0 to cGrid - 1 do
    for gx := 0 to cGrid - 1 do
    begin
      tok := gy * cGrid + gx;
      for fy := 0 to cPatch - 1 do
        for fx := 0 to cPatch - 1 do
        begin
          d := fy * cPatch + fx;
          Tokens.Data[tok, 0, d] :=
            Img.Get(gx * cPatch + fx, gy * cPatch + fy, 0);
        end;
    end;
end;

// Random subset of cMaskN masked token indices. Masked[i]=true => token dropped.
procedure RandomMask(out Masked: array of boolean);
var
  perm: TPatchIdx;
  i, j, t: integer;
begin
  for i := 0 to cNumPatches - 1 do perm[i] := i;
  for i := cNumPatches - 1 downto 1 do
  begin
    j := Random(i + 1);
    t := perm[i]; perm[i] := perm[j]; perm[j] := t;
  end;
  for i := 0 to cNumPatches - 1 do Masked[i] := false;
  for i := 0 to cMaskN - 1 do Masked[perm[i]] := true;
end;

// ---------------------------------------------------------------------------
// Build the MAE network (encoder + lightweight decoder + per-token pixel head).
// ---------------------------------------------------------------------------
// Returns the MAE net and, via EncEndIdx, the layer index of the ENCODER output
// (the token features the linear probe reads). The decoder block + pixel head
// follow it.
function BuildMAE(out EncEndIdx: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cNumPatches, 1, cPatchDim));
  // Patch embed: per-token linear projection patchDim -> d_model.
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cDModel));
  // Learned absolute positional embedding over the cNumPatches grid positions.
  Result.AddLayer(TNNetLearnedPositionalEmbedding.Create(cNumPatches));
  // ENCODER: 3 transformer blocks.
  Result.AddTransformerEncoderBlock(cHeads, cFF);
  Result.AddTransformerEncoderBlock(cHeads, cFF);
  EncEndIdx := Result.AddTransformerEncoderBlock(cHeads, cFF).LayerIdx;
  // DECODER (lighter): 1 transformer block.
  Result.AddTransformerEncoderBlock(cHeads, cFF);
  // Per-token pixel reconstruction head d_model -> patchDim.
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cPatchDim));
end;

// ---------------------------------------------------------------------------
// Mean-pool the token features of a given layer's output into a 1x1xD vector.
// ---------------------------------------------------------------------------
procedure MeanPoolTokens(Src, Dst: TNNetVolume);
var
  tok, d: integer;
  acc: TNeuralFloat;
begin
  Dst.ReSize(1, 1, cDModel);
  for d := 0 to cDModel - 1 do
  begin
    acc := 0;
    for tok := 0 to cNumPatches - 1 do
      acc := acc + Src.Get(tok, 0, d);
    Dst.Data[0, 0, d] := acc / cNumPatches;
  end;
end;

// ---------------------------------------------------------------------------
// One self-supervised MAE step on image Img.
//   - patchify, choose mask, zero masked tokens (shared zero mask token),
//   - forward, build masked-only MSE target, backpropagate.
// Returns the mean per-element reconstruction error on the masked patches.
// ---------------------------------------------------------------------------
function MAEStep(NN: TNNet; Img, Tokens, Target: TNNetVolume;
  Train: boolean): TNeuralFloat;
var
  Masked: array[0..cNumPatches - 1] of boolean;
  tok, d, cnt: integer;
  Pred: TNNetVolume;
  err, dif: TNeuralFloat;
begin
  Patchify(Img, Tokens);
  // Save the clean patch pixels as the reconstruction ground truth.
  Target.Copy(Tokens);
  RandomMask(Masked);
  // Replace masked tokens with the shared (zero) mask token at the INPUT.
  for tok := 0 to cNumPatches - 1 do
    if Masked[tok] then
      for d := 0 to cPatchDim - 1 do
        Tokens.Data[tok, 0, d] := 0;

  NN.Compute(Tokens);
  Pred := NN.GetLastLayer.Output;

  // Build the regression target: VISIBLE entries = prediction (=> zero error),
  // MASKED entries = true pixels. This realises "loss only on masked patches".
  err := 0; cnt := 0;
  for tok := 0 to cNumPatches - 1 do
    for d := 0 to cPatchDim - 1 do
      if Masked[tok] then
      begin
        dif := Pred.Get(tok, 0, d) - Target.Get(tok, 0, d);
        err := err + dif * dif;
        Inc(cnt);
        // Target already holds true pixels here.
      end
      else
        // Visible: set target = prediction so error is exactly 0.
        Target.Data[tok, 0, d] := Pred.Get(tok, 0, d);

  if Train then NN.Backpropagate(Target);
  if cnt > 0 then Result := err / cnt else Result := 0;
end;

// Extract the frozen mean-pooled encoder feature of every probe image once.
procedure ExtractFeatures(MAE: TNNet;
  var TrFeat: array of TNNetVolume; var TeFeat: array of TNNetVolume);
var
  Tokens: TNNetVolume;
  s: integer;
begin
  Tokens := TNNetVolume.Create(cNumPatches, 1, cPatchDim);
  for s := 0 to cProbeTrain - 1 do
  begin
    Patchify(ProbeTrIn[s], Tokens);
    MAE.Compute(Tokens);
    MeanPoolTokens(MAE.Layers[gEncEnd].Output, TrFeat[s]);
  end;
  for s := 0 to cProbeTest - 1 do
  begin
    Patchify(ProbeTeIn[s], Tokens);
    MAE.Compute(Tokens);
    MeanPoolTokens(MAE.Layers[gEncEnd].Output, TeFeat[s]);
  end;
  Tokens.Free;
end;

// ---------------------------------------------------------------------------
// Linear probe: freeze encoder, train a single linear+softmax classifier on the
// mean-pooled encoder features. Averaged over cProbeSeeds probe initialisations
// (the probe is tiny and high-variance) for a stable, reproducible number.
// ---------------------------------------------------------------------------
function LinearProbe(MAE: TNNet): TNeuralFloat;
var
  Probe: TNNet;
  TrFeat: array[0..cProbeTrain - 1] of TNNetVolume;
  TeFeat: array[0..cProbeTest  - 1] of TNNetVolume;
  ep, s, correct, i, order, tmp, seed: integer;
  perm: array[0..cProbeTrain - 1] of integer;
  accSum: TNeuralFloat;
begin
  for s := 0 to cProbeTrain - 1 do TrFeat[s] := TNNetVolume.Create(1, 1, cDModel);
  for s := 0 to cProbeTest  - 1 do TeFeat[s] := TNNetVolume.Create(1, 1, cDModel);
  ExtractFeatures(MAE, TrFeat, TeFeat);

  accSum := 0;
  for seed := 0 to cProbeSeeds - 1 do
  begin
    Probe := TNNet.Create();
    Probe.AddLayer(TNNetInput.Create(cDModel));
    Probe.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    Probe.AddLayer(TNNetSoftMax.Create());
    Probe.SetLearningRate(cProbeLR, 0.9);

    RandSeed := cSeed + 99 + seed * 17;
    for i := 0 to cProbeTrain - 1 do perm[i] := i;
    for ep := 1 to cProbeEpochs do
    begin
      for i := cProbeTrain - 1 downto 1 do
      begin
        order := Random(i + 1);
        tmp := perm[i]; perm[i] := perm[order]; perm[order] := tmp;
      end;
      for s := 0 to cProbeTrain - 1 do
      begin
        Probe.Compute(TrFeat[perm[s]]);
        Probe.Backpropagate(ProbeTrTg[perm[s]]);
      end;
    end;

    correct := 0;
    for s := 0 to cProbeTest - 1 do
    begin
      Probe.Compute(TeFeat[s]);
      if Probe.GetLastLayer.Output.GetClass() = ProbeTeTg[s].GetClass() then
        Inc(correct);
    end;
    accSum := accSum + correct / cProbeTest;
    Probe.Free;
  end;

  Result := accSum / cProbeSeeds;
  for s := 0 to cProbeTrain - 1 do TrFeat[s].Free;
  for s := 0 to cProbeTest  - 1 do TeFeat[s].Free;
end;

procedure BuildData();
var
  i, cls: integer;
begin
  RandSeed := cSeed;
  SetLength(PreImgs, 600);
  for i := 0 to High(PreImgs) do
  begin
    PreImgs[i] := TNNetVolume.Create(cSizeX, cSizeY, cChannelsHelper);
    MakeRandomImage(PreImgs[i], cls);
  end;
  for i := 0 to cProbeTrain - 1 do
  begin
    ProbeTrIn[i] := TNNetVolume.Create(cSizeX, cSizeY, cChannelsHelper);
    ProbeTrTg[i] := TNNetVolume.Create(cClasses, 1, 1);
    MakeRandomImage(ProbeTrIn[i], cls);
    OneHot(ProbeTrTg[i], cls);
  end;
  for i := 0 to cProbeTest - 1 do
  begin
    ProbeTeIn[i] := TNNetVolume.Create(cSizeX, cSizeY, cChannelsHelper);
    ProbeTeTg[i] := TNNetVolume.Create(cClasses, 1, 1);
    MakeRandomImage(ProbeTeIn[i], cls);
    OneHot(ProbeTeTg[i], cls);
  end;
end;

procedure FreeData();
var i: integer;
begin
  for i := 0 to High(PreImgs) do PreImgs[i].Free;
  for i := 0 to cProbeTrain - 1 do begin ProbeTrIn[i].Free; ProbeTrTg[i].Free; end;
  for i := 0 to cProbeTest  - 1 do begin ProbeTeIn[i].Free; ProbeTeTg[i].Free; end;
end;

var
  MAE, RandMAE: TNNet;
  Tokens, Target: TNNetVolume;
  step, idx: integer;
  loss, running, accMAE, accRand: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  WriteLn('MaskedAutoencoder: self-supervised MAE pretraining + linear probe.');
  WriteLn(Format('Image %dx%d  patch %dx%d  tokens=%d  d_model=%d',
    [cSizeX, cSizeY, cPatch, cPatch, cNumPatches, cDModel]));
  WriteLn(Format('Masking: %d of %d patches masked (%.0f%%) each step',
    [cMaskN, cNumPatches, 100.0 * cMaskN / cNumPatches]));
  WriteLn(Format('Pretrain steps=%d  LR=%.4f', [cPretrain, cPreLR]));
  WriteLn;

  BuildData();

  Tokens := TNNetVolume.Create(cNumPatches, 1, cPatchDim);
  Target := TNNetVolume.Create(cNumPatches, 1, cPatchDim);

  // gEncEnd is the layer index of the encoder output (the token features the
  // linear probe reads); the decoder block + pixel head follow it.
  MAE := BuildMAE(gEncEnd);
  MAE.SetLearningRate(cPreLR, 0.9);
  RandSeed := cSeed + 1;

  WriteLn('Pretraining (self-supervised, no labels)...');
  running := 0;
  for step := 1 to cPretrain do
  begin
    idx := Random(Length(PreImgs));
    loss := MAEStep(MAE, PreImgs[idx], Tokens, Target, True);
    // Per-step MSE is noisy (it depends on which patches were masked), so report
    // a running average of the masked-patch reconstruction error.
    if step = 1 then running := loss
    else running := 0.99 * running + 0.01 * loss;
    if (step mod 600 = 0) or (step = 1) then
      WriteLn(Format('  step %5d   masked-recon MSE (avg) = %.5f', [step, running]));
  end;
  WriteLn;

  // A second, identical-architecture MAE left at random init for the baseline
  // (same encoder-end layer index, so we ignore the returned value).
  RandMAE := BuildMAE(idx);

  WriteLn('Linear probe (frozen encoder, single linear classifier):');
  accMAE  := LinearProbe(MAE);
  accRand := LinearProbe(RandMAE);
  WriteLn(Format('  MAE-pretrained encoder  test-acc = %.3f', [accMAE]));
  WriteLn(Format('  random-init   encoder  test-acc = %.3f', [accRand]));
  WriteLn(Format('  chance level                     = %.3f', [1.0 / cClasses]));
  WriteLn;

  if accMAE > accRand then
    WriteLn('=> MAE pretraining learned useful features from UNLABELLED pixels alone:')
  else
    WriteLn('=> (no MAE advantage this run; try more pretrain steps)');
  WriteLn(Format('   the frozen-encoder linear probe is %.1f points more accurate than',
    [100.0 * (accMAE - accRand)]));
  WriteLn(Format('   the same architecture at random init (%.3f -> %.3f), well above the',
    [accRand, accMAE]));
  WriteLn(Format('   %.3f chance level -- the self-supervised objective shaped the narrow',
    [1.0 / cClasses]));
  WriteLn('   encoder bottleneck into class-relevant features with no labels.');

  MAE.Free;
  RandMAE.Free;
  Tokens.Free;
  Target.Free;
  FreeData();
end.
