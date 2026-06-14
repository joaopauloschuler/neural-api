program StyleTransfer;
(*
 Neural style transfer (Gatys, Ecker & Bethge, CVPR 2016) command-line demo.
 https://github.com/joaopauloschuler/neural-api/tree/master/examples/StyleTransfer

 Optimises the pixels of a canvas so that its VGG content features match a
 content image and its VGG Gram-matrix style statistics match a style image.
 The VGG-16 feature extractor is imported with BuildVGGFromSafeTensors; the
 canvas is optimised by plain gradient descent on the INPUT volume (the same
 backprop-to-the-input machinery used by examples/GradientAscent).

 The example is self-contained: with no --content / --style it synthesises tiny
 images and uses the committed tiny VGG fixture so it runs on CPU in seconds and
 is CI-runnable. Point --vgg / --config at real torchvision VGG-16 safetensors
 (and pass real images) for genuine artistic results.

 Coded by Joao Paulo Schwarz Schuler.

 Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes,
  SysUtils,
  CustApp,
  Math,
  neuralnetwork,
  neuralvolume,
  neuraldatasets,
  neuralpretrained,
  FPImage,
  FPReadPNG, FPWritePNG,
  FPReadJPEG, FPWriteJPEG,
  FPReadBMP, FPWriteBMP;

type
  { ImageNet RGB normalisation constants (torchvision VGG). }
  TImageStats = record
    Mean, Std: array[0..2] of TNeuralFloat;
  end;

const
  // ImageNet mean/std on a 0..1 scale.
  cMean: array[0..2] of TNeuralFloat = (0.485, 0.456, 0.406);
  cStd:  array[0..2] of TNeuralFloat = (0.229, 0.224, 0.225);

type
  { TStyleTransferApp }
  TStyleTransferApp = class(TCustomApplication)
  protected
    procedure DoRun; override;
  public
    procedure WriteHelp; virtual;
  end;

// ---------------------------------------------------------------------------
// Gram-matrix helper (the genuinely new piece of code). Given a feature map
// V of shape (W, H, C) (depth-contiguous in V.FData), compute the normalised
// Gram matrix G[i,j] = (1/(C*H*W)) * sum_{x,y} V[x,y,i] * V[x,y,j].  G is
// returned as a (C, C, 1) volume (G.FData[i*C + j]).
// ---------------------------------------------------------------------------
procedure ComputeGram(V: TNNetVolume; G: TNNetVolume);
var
  C, HW, x, y, i, j, base: integer;
  norm, acc: TNeuralFloat;
begin
  C := V.Depth;
  HW := V.SizeX * V.SizeY;
  G.ReSize(C, C, 1);
  norm := 1.0 / (C * HW);
  for i := 0 to C - 1 do
    for j := i to C - 1 do
    begin
      acc := 0;
      for y := 0 to V.SizeY - 1 do
        for x := 0 to V.SizeX - 1 do
        begin
          base := (y * V.SizeX + x) * C;
          acc := acc + V.FData[base + i] * V.FData[base + j];
        end;
      acc := acc * norm;
      G.FData[i * C + j] := acc;
      G.FData[j * C + i] := acc; // symmetric
    end;
end;

// Accumulate the style-loss gradient w.r.t. the feature map V into Dst.
// For style loss L = ||Gram(V) - GramTarget||^2 the gradient is
//   dL/dV[x,y,i] = (4 / (C*H*W)) * sum_j (G - Gtarget)[i,j] * V[x,y,j].
// The diff (G - Gtarget) is precomputed in DiffGram. Returns the scalar loss.
function AddStyleGrad(V, DiffGram, Dst: TNNetVolume; Weight: TNeuralFloat): TNeuralFloat;
var
  C, HW, x, y, i, j, base: integer;
  norm, g, gv: TNeuralFloat;
begin
  C := V.Depth;
  HW := V.SizeX * V.SizeY;
  norm := Weight * 4.0 / (C * HW);
  Result := 0;
  for i := 0 to C - 1 do
    for j := 0 to C - 1 do
      Result := Result + Sqr(DiffGram.FData[i * C + j]);
  Result := Result * Weight;
  for y := 0 to V.SizeY - 1 do
    for x := 0 to V.SizeX - 1 do
    begin
      base := (y * V.SizeX + x) * C;
      for i := 0 to C - 1 do
      begin
        gv := 0;
        for j := 0 to C - 1 do
          gv := gv + DiffGram.FData[i * C + j] * V.FData[base + j];
        g := gv * norm;
        Dst.FData[base + i] := Dst.FData[base + i] + g;
      end;
    end;
end;

// Loads an image file into a 0..255 volume, or synthesises a tiny test image
// if FileName is empty. Pattern selects which synthetic image (0 = content
// gradient blobs, 1 = style stripes) so content and style differ.
procedure LoadOrSynth(const FileName: string; V: TNNetVolume;
  ImgSize, Pattern: integer);
var
  x, y: integer;
  tmp: TNNetVolume;
begin
  if FileName <> '' then
  begin
    if not LoadImageFromFileIntoVolume(FileName, V) then
      raise Exception.Create('Could not load image: ' + FileName);
    // Resize to the working resolution by simple area copy if needed.
    if (V.SizeX <> ImgSize) or (V.SizeY <> ImgSize) then
    begin
      tmp := TNNetVolume.Create(ImgSize, ImgSize, 3);
      tmp.CopyResizing(V, ImgSize, ImgSize);
      V.Copy(tmp);
      tmp.Free;
    end;
    exit;
  end;
  V.ReSize(ImgSize, ImgSize, 3);
  for y := 0 to ImgSize - 1 do
    for x := 0 to ImgSize - 1 do
    begin
      if Pattern = 0 then
      begin
        // smooth diagonal gradient (content)
        V.Store(x, y, 0, (x * 255) div ImgSize);
        V.Store(x, y, 1, (y * 255) div ImgSize);
        V.Store(x, y, 2, ((x + y) * 255) div (2 * ImgSize));
      end
      else
      begin
        // high-frequency stripes (style texture)
        if (x div 3) mod 2 = 0 then V.Store(x, y, 0, 220) else V.Store(x, y, 0, 30);
        if (y div 3) mod 2 = 0 then V.Store(x, y, 1, 200) else V.Store(x, y, 1, 50);
        V.Store(x, y, 2, ((x xor y) and 255));
      end;
    end;
end;

// 0..255 RGB -> ImageNet-normalised VGG input (in place).
procedure NormalizeForVGG(V: TNNetVolume);
var
  x, y, d, base: integer;
begin
  for y := 0 to V.SizeY - 1 do
    for x := 0 to V.SizeX - 1 do
    begin
      base := (y * V.SizeX + x) * V.Depth;
      for d := 0 to 2 do
        V.FData[base + d] := (V.FData[base + d] / 255.0 - cMean[d]) / cStd[d];
    end;
end;

// VGG-normalised -> 0..255 RGB (in place), clamped.
procedure DenormalizeFromVGG(V: TNNetVolume);
var
  x, y, d, base: integer;
  p: TNeuralFloat;
begin
  for y := 0 to V.SizeY - 1 do
    for x := 0 to V.SizeX - 1 do
    begin
      base := (y * V.SizeX + x) * V.Depth;
      for d := 0 to 2 do
      begin
        p := (V.FData[base + d] * cStd[d] + cMean[d]) * 255.0;
        if p < 0 then p := 0;
        if p > 255 then p := 255;
        V.FData[base + d] := p;
      end;
    end;
end;

procedure TStyleTransferApp.WriteHelp;
begin
  WriteLn('Neural Style Transfer (Gatys et al. 2016) demo');
  WriteLn('Usage: StyleTransfer [options]');
  WriteLn('  --vgg <file>      VGG-16 safetensors (default: tiny test fixture)');
  WriteLn('  --config <file>   VGG config JSON (default: matching fixture config)');
  WriteLn('  --content <file>  content image (default: synthetic gradient)');
  WriteLn('  --style <file>    style image   (default: synthetic stripes)');
  WriteLn('  --out <file>      output PNG (default: stylized.png)');
  WriteLn('  --size <n>        working square size (default: from VGG config)');
  WriteLn('  --iter <n>        optimisation steps (default: 40)');
  WriteLn('  --lr <f>          step size (default: 5.0)');
  WriteLn('  --styleweight <f> style/content balance (default: 1e3)');
end;

procedure TStyleTransferApp.DoRun;
var
  VGGFile, ConfigFile, ContentFile, StyleFile, OutFile: string;
  NN: TNNet;
  Config: TVGGConfig;
  TapIdx: array[0..4] of integer;
  ImgSize, Iter, MaxIter, ContentTap: integer;
  LR, StyleWeight, ContentWeight: TNeuralFloat;
  Content, Style, Canvas, Step: TNNetVolume;
  ContentTargets: array[0..4] of TNNetVolume;
  StyleGrams: array[0..4] of TNNetVolume;
  CurGram, DiffGram: TNNetVolume;
  TapW: array[0..4] of TNeuralFloat;
  k, LayerCnt: integer;
  TotalLoss, StyleLoss, ContentLoss, t0: double;
  TapV, ContentErr: TNNetVolume;
  cx, cy, cd, cbase: integer;
  cdiff: TNeuralFloat;
begin
  if HasOption('h', 'help') then begin WriteHelp; Terminate; exit; end;

  VGGFile     := GetOptionValue('vgg');
  ConfigFile  := GetOptionValue('config');
  ContentFile := GetOptionValue('content');
  StyleFile   := GetOptionValue('style');
  OutFile     := GetOptionValue('out');
  if OutFile = '' then OutFile := 'stylized.png';
  if VGGFile = '' then
  begin
    VGGFile := '../../tests/fixtures/tiny_vgg16.safetensors';
    if ConfigFile = '' then ConfigFile := '../../tests/fixtures/tiny_vgg16_config.json';
    WriteLn('No --vgg given: using committed tiny VGG fixture (pipeline demo,');
    WriteLn('not artistic). Pass real torchvision VGG-16 safetensors for art.');
  end;

  MaxIter := 40;
  if GetOptionValue('iter') <> '' then MaxIter := StrToInt(GetOptionValue('iter'));
  LR := 5.0;
  if GetOptionValue('lr') <> '' then LR := StrToFloat(GetOptionValue('lr'));
  StyleWeight := 1000.0;
  if GetOptionValue('styleweight') <> '' then StyleWeight := StrToFloat(GetOptionValue('styleweight'));
  ContentWeight := 1.0;

  WriteLn('Building VGG feature extractor from: ', VGGFile);
  // Read the config, then force FeatureTapStage = 5 so BuildVGG TRUNCATES the
  // net at relu5 (drops the AdaptiveAvgPool + FC classifier). All five per-stage
  // taps are still exposed via TapIdx and relu5 becomes the net's last layer, so
  // a manual Backpropagate() from the last layer carries every injected tap
  // gradient back to the input. pInferenceOnly = false so OutputError buffers
  // exist for backprop to the input pixels.
  if ConfigFile = '' then ConfigFile := ExtractFilePath(VGGFile) + 'config.json';
  Config := ReadVGGConfigFromJSONFile(ConfigFile);
  Config.FeatureTapStage := 5;
  NN := BuildVGGFromSafeTensorsEx(VGGFile, Config, TapIdx, {pInferenceOnly=}false);
  NN.EnableDropouts(false);
  // FREEZE the VGG: backprop must flow to the INPUT pixels only, never touch
  // the conv weights. SetBatchUpdate(true) makes Backpropagate ACCUMULATE weight
  // deltas instead of applying them per sample; since we never call
  // UpdateWeights(), the weights stay frozen while the error still propagates to
  // the input. (LR is kept nonzero: some backward kernels early-exit at LR=0.)
  NN.SetBatchUpdate(true);
  TNNetInput(NN.Layers[0]).EnableErrorCollection();
  LayerCnt := NN.CountLayers();
  WriteLn('VGG layers: ', LayerCnt, '  taps: ',
    TapIdx[0], ' ', TapIdx[1], ' ', TapIdx[2], ' ', TapIdx[3], ' ', TapIdx[4]);

  ImgSize := Config.ImageSize;
  if GetOptionValue('size') <> '' then ImgSize := StrToInt(GetOptionValue('size'));
  WriteLn('Working image size: ', ImgSize, 'x', ImgSize);

  // Style taps (relu1_2..relu5_x) all contribute; content uses relu4_3 (tap 3).
  for k := 0 to 4 do TapW[k] := 0.2;
  ContentTap := 3;

  Content := TNNetVolume.Create();
  Style   := TNNetVolume.Create();
  Canvas  := TNNetVolume.Create();
  Step    := TNNetVolume.Create();
  CurGram := TNNetVolume.Create();
  DiffGram := TNNetVolume.Create();
  for k := 0 to 4 do
  begin
    ContentTargets[k] := TNNetVolume.Create();
    StyleGrams[k] := TNNetVolume.Create();
  end;

  // Load / synthesise inputs.
  LoadOrSynth(ContentFile, Content, ImgSize, 0);
  LoadOrSynth(StyleFile, Style, ImgSize, 1);
  NormalizeForVGG(Content);
  NormalizeForVGG(Style);

  // Precompute content targets (just need the tap we use) and style Grams.
  NN.Compute(Content);
  ContentTargets[ContentTap].Copy(NN.Layers[TapIdx[ContentTap]].Output);

  NN.Compute(Style);
  for k := 0 to 4 do
    ComputeGram(NN.Layers[TapIdx[k]].Output, StyleGrams[k]);

  // Initialise the canvas with the content image.
  Canvas.Copy(Content);

  WriteLn('Optimising canvas for ', MaxIter, ' iterations (lr=', LR:0:3,
    ', styleweight=', StyleWeight:0:1, ')...');
  // The truncated feature net's last layer (relu5) has no consumer, so its
  // departing-branch count stays 0. Driving Backpropagate() manually would trip
  // the backward call-count guard; hoist the IncDepartingBranchesCnt above the
  // loop (ResetBackpropCallCurrCnt below clears the per-pass counter only, so
  // one increment here is correct for every iteration).
  NN.Layers[LayerCnt - 1].IncDepartingBranchesCnt();
  t0 := Now();
  for Iter := 1 to MaxIter do
  begin
    NN.Compute(Canvas);

    // Reset the per-pass backward counters AND zero every layer's OutputError
    // (ResetBackpropCallCurrCnt does both). We must do this BEFORE injecting tap
    // gradients, otherwise it would wipe the errors we preload. After injection
    // the standard backward accumulates them via FPrevLayer.OutputError.Add.
    NN.ResetBackpropCallCurrCnt();

    TotalLoss := 0; StyleLoss := 0; ContentLoss := 0;

    // Content loss gradient at relu4_3: dL/dF = 2*cw*(F - Ftarget).
    TapV := NN.Layers[TapIdx[ContentTap]].Output;
    ContentErr := NN.Layers[TapIdx[ContentTap]].OutputError;
    for cy := 0 to TapV.SizeY - 1 do
      for cx := 0 to TapV.SizeX - 1 do
      begin
        cbase := (cy * TapV.SizeX + cx) * TapV.Depth;
        for cd := 0 to TapV.Depth - 1 do
        begin
          cdiff := TapV.FData[cbase + cd] - ContentTargets[ContentTap].FData[cbase + cd];
          ContentLoss := ContentLoss + ContentWeight * Sqr(cdiff);
          ContentErr.FData[cbase + cd] := ContentErr.FData[cbase + cd] + 2.0 * ContentWeight * cdiff;
        end;
      end;

    // Style loss gradient at each tap.
    for k := 0 to 4 do
    begin
      TapV := NN.Layers[TapIdx[k]].Output;
      ComputeGram(TapV, CurGram);
      DiffGram.Copy(CurGram);
      DiffGram.Sub(StyleGrams[k]);
      StyleLoss := StyleLoss +
        AddStyleGrad(TapV, DiffGram, NN.Layers[TapIdx[k]].OutputError,
          StyleWeight * TapW[k]);
    end;

    TotalLoss := ContentLoss + StyleLoss;

    // Backward pass: drive from the last layer; ReLU/conv backward multiply &
    // ADD into earlier layers' OutputError, so our preloaded tap gradients ride
    // through to the input layer's OutputError. (Counters were reset above.)
    NN.Layers[LayerCnt - 1].Backpropagate();

    // Gradient-descent step on the input pixels.
    Step.Copy(NN.Layers[0].OutputError);
    Canvas.MulAdd(-LR, Step);

    if (Iter mod 5 = 0) or (Iter = 1) or (Iter = MaxIter) then
      WriteLn(Format('  iter %3d  total=%.5g  content=%.5g  style=%.5g  |dInput|=%.5g  |canvas|=%.5g',
        [Iter, TotalLoss, ContentLoss, StyleLoss, Step.GetSumAbs(), Canvas.GetSumAbs()]));
  end;
  WriteLn('Optimisation done in ', ((Now() - t0) * 86400):0:2, ' s.');

  DenormalizeFromVGG(Canvas);
  if SaveImageFromVolumeIntoFile(Canvas, OutFile) then
    WriteLn('Wrote stylized image: ', OutFile)
  else
    WriteLn('Failed to write: ', OutFile);

  for k := 0 to 4 do
  begin
    ContentTargets[k].Free;
    StyleGrams[k].Free;
  end;
  CurGram.Free; DiffGram.Free;
  Content.Free; Style.Free; Canvas.Free; Step.Free;
  NN.Free;
  Terminate;
end;

var
  App: TStyleTransferApp;
begin
  App := TStyleTransferApp.Create(nil);
  App.Title := 'StyleTransfer';
  App.Run;
  App.Free;
end.
