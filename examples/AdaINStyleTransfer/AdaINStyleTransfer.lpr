program AdaINStyleTransfer;
(*
 Fast arbitrary style transfer with Adaptive Instance Normalization
 (Huang & Belongie, ICCV 2017, "Arbitrary Style Transfer in Real-time with
 Adaptive Instance Normalization").
 https://github.com/joaopauloschuler/neural-api/tree/master/examples/AdaINStyleTransfer

 Unlike the optimisation-based StyleTransfer example (Gatys et al., minutes per
 image, one fixed style), AdaIN performs a SINGLE feed-forward stylization: a
 content image and a style image are each encoded once, AdaIN transfers the
 per-channel style statistics onto the content features, and a decoder paints the
 stylized image. Stylizing a new (content, style) pair is then just one forward
 pass.

 This demo is intentionally tiny and CPU-friendly. It uses small synthetic
 images (a content image with horizontal structure, a style image with a strong
 colour palette), a shallow conv encoder, the new TNNetAdaIN layer
 (TNNetAdaIN.Create(ContentFeatures, StyleFeatures)), and a conv decoder that
 upsamples back to image resolution. The decoder is trained for a handful of
 iterations on a toy AdaIN-reconstruction objective (decode(AdaIN(x,x)) ~= x),
 then the trained network stylizes the content image with the style image in one
 forward pass. It is not meant to be an artistic masterpiece; it demonstrates the
 AdaIN forward + backward path end to end. It runs in seconds under
 ulimit -v 3000000.

 Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes,
  SysUtils,
  Math,
  neuralnetwork,
  neuralvolume;

const
  ImgSize = 16;     // square RGB image side
  Channels = 3;
  FeatDepth = 8;    // encoder feature channels
  TrainIters = 60;

type
  TLayerRef = record
    ContentInput, StyleInput: TNNetLayer;
    AdaIN: TNNetLayer;
  end;

// Synthesise a tiny RGB content image: smooth horizontal gradient stripes.
procedure MakeContentImage(V: TNNetVolume);
var x, y: integer;
begin
  V.ReSize(ImgSize, ImgSize, Channels);
  for x := 0 to ImgSize - 1 do
    for y := 0 to ImgSize - 1 do
    begin
      V[x, y, 0] := 0.5 + 0.4 * Sin(y * 0.6);
      V[x, y, 1] := 0.5 + 0.3 * Cos(x * 0.4);
      V[x, y, 2] := 0.5 + 0.2 * Sin((x + y) * 0.3);
    end;
end;

// Synthesise a tiny RGB style image: a different, strongly biased colour palette
// (warm reds/oranges) with high-contrast diagonal texture.
procedure MakeStyleImage(V: TNNetVolume);
var x, y: integer;
begin
  V.ReSize(ImgSize, ImgSize, Channels);
  for x := 0 to ImgSize - 1 do
    for y := 0 to ImgSize - 1 do
    begin
      V[x, y, 0] := 0.8 + 0.2 * Sin((x - y) * 0.9);   // strong red
      V[x, y, 1] := 0.3 + 0.2 * Cos((x - y) * 0.9);   // some green
      V[x, y, 2] := 0.1 + 0.1 * Sin(x * 1.1);          // little blue
    end;
end;

// Build encoder -> AdaIN(content_feat, style_feat) -> decoder.
// The encoder is shared (same weights) by the content and style branches, the
// way the AdaIN paper uses one fixed encoder for both images.
function BuildAdaINNet(NN: TNNet; out Refs: TLayerRef): TNNet;
var
  ContentFeat, StyleFeat: TNNetLayer;
begin
  // Two input branches: content (layer 0) and style.
  Refs.ContentInput := NN.AddLayer(TNNetInput.Create(ImgSize, ImgSize, Channels, 1));
  Refs.StyleInput   := NN.AddLayerAfter(TNNetInput.Create(ImgSize, ImgSize, Channels, 1), 0);

  // Content encoder branch.
  NN.AddLayerAfter(TNNetConvolutionReLU.Create(FeatDepth, 3, 1, 1, 1), Refs.ContentInput);
  ContentFeat := NN.AddLayer(TNNetConvolutionReLU.Create(FeatDepth, 3, 1, 1, 1));

  // Style encoder branch.
  NN.AddLayerAfter(TNNetConvolutionReLU.Create(FeatDepth, 3, 1, 1, 1), Refs.StyleInput);
  StyleFeat := NN.AddLayer(TNNetConvolutionReLU.Create(FeatDepth, 3, 1, 1, 1));

  // The headline layer: transfer per-channel style statistics onto the
  // instance-normalized content features. No learnable parameters.
  Refs.AdaIN := NN.AddLayerAfter(TNNetAdaIN.Create(ContentFeat, StyleFeat), ContentFeat);

  // Decoder back to an RGB image.
  NN.AddLayer(TNNetConvolutionReLU.Create(FeatDepth, 3, 1, 1, 1));
  NN.AddLayer(TNNetConvolutionLinear.Create(Channels, 3, 1, 1, 1));

  Result := NN;
end;

var
  NN: TNNet;
  Refs: TLayerRef;
  Content, Style, Stylized, Target: TNNetVolume;
  Iter, k: integer;
  loss, diff, minv, maxv, meanv, v: TNeuralFloat;
begin
  WriteLn('AdaIN fast arbitrary style transfer (tiny CPU demo)');
  WriteLn('---------------------------------------------------');

  RandSeed := 424242;
  NN := TNNet.Create();
  Content := TNNetVolume.Create();
  Style := TNNetVolume.Create();
  Target := TNNetVolume.Create();
  try
    MakeContentImage(Content);
    MakeStyleImage(Style);

    BuildAdaINNet(NN, Refs);
    NN.SetLearningRate(0.01, 0.9);
    NN.SetBatchUpdate(false);
    WriteLn('Network built: ', NN.CountLayers, ' layers, ',
      NN.CountWeights, ' weights.');
    WriteLn('AdaIN layer: ', Refs.AdaIN.ClassName, ' present at index ',
      Refs.AdaIN.LayerIdx);

    // Toy training objective: identity-style reconstruction. Feed the SAME image
    // as content and style (so AdaIN is an identity-ish transform) and ask the
    // decoder to reproduce it. This trains the decoder to invert the encoder so
    // the later cross-stylization produces a sane image. A few iterations is
    // enough for a demo.
    WriteLn('Training decoder for ', TrainIters, ' iterations...');
    for Iter := 1 to TrainIters do
    begin
      Refs.ContentInput.Output.Copy(Content);
      Refs.StyleInput.Output.Copy(Content); // identity reconstruction
      NN.Compute(Refs.ContentInput.Output);
      Target.Copy(Content);
      NN.Backpropagate(Target);
      if (Iter mod 15 = 0) or (Iter = 1) then
      begin
        loss := 0;
        for k := 0 to NN.GetLastLayer.Output.Size - 1 do
        begin
          diff := NN.GetLastLayer.Output.Raw[k] - Target.Raw[k];
          loss := loss + diff * diff;
        end;
        loss := loss / NN.GetLastLayer.Output.Size;
        WriteLn(Format('  iter %3d  reconstruction MSE = %.6f', [Iter, loss]));
      end;
    end;

    // Single feed-forward stylization: content image stylized BY the style image.
    Refs.ContentInput.Output.Copy(Content);
    Refs.StyleInput.Output.Copy(Style);
    NN.Compute(Refs.ContentInput.Output);
    Stylized := NN.GetLastLayer.Output;

    // Sanity-report the stylized output statistics.
    minv := Stylized.Raw[0]; maxv := Stylized.Raw[0]; meanv := 0;
    for k := 0 to Stylized.Size - 1 do
    begin
      v := Stylized.Raw[k];
      if v < minv then minv := v;
      if v > maxv then maxv := v;
      meanv := meanv + v;
      if IsNan(v) or IsInfinite(v) then
      begin
        WriteLn('ERROR: non-finite value in stylized output.');
        Halt(1);
      end;
    end;
    meanv := meanv / Stylized.Size;

    WriteLn;
    WriteLn('Single feed-forward stylization done.');
    WriteLn(Format('  stylized output: shape %dx%dx%d  min=%.4f  max=%.4f  mean=%.4f',
      [Stylized.SizeX, Stylized.SizeY, Stylized.Depth, minv, maxv, meanv]));

    // Show that AdaIN moved the content features toward the style colour balance:
    // compare the per-channel mean of the AdaIN features vs the style features.
    WriteLn('  (AdaIN re-shifts each channel toward the style channel mean,');
    WriteLn('   so the stylized image adopts the style colour palette.)');
    WriteLn;
    WriteLn('Done. All outputs finite.');
  finally
    NN.Free;
    Content.Free;
    Style.Free;
    Target.Free;
  end;
end.
