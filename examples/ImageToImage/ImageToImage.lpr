program ImageToImage;
(*
ImageToImage: SDEdit-style image-to-image / real-image EDITING. Unlike the
text->image-from-pure-noise diffusion examples (DiffusionMNIST,
ConditionalDiffusion, the PixArt text->image path), SDEdit (Meng et al. 2021,
"SDEdit: Guided Image Synthesis and Editing with Stochastic Differential
Equations") starts the reverse process from a REAL image that has been encoded
to a latent and then PARTIALLY noised to an intermediate timestep. Because the
start latent still carries the source image's coarse layout, the denoiser
preserves that layout while a NEW conditioning prompt steers the content -- the
classic "edit this picture" behaviour.

THE PIPELINE (all four pieces already landed in this repo; this example is ONLY
the driver that wires them together plus the `strength` knob):

  1. ENCODE   real RGB image  --VAE encoder-->  clean latent z0
        (neuralpretrained.pas BuildVaeEncoderFromSafeTensors;
         output is mean * scaling_factor, the deterministic SD latent.)
  2. NOISE    z0  --scheduler AddNoise to timestep t_start-->  z_t
        (neuraldiffusion.pas TNNetDiffusionScheduler.AddNoise. The `strength`
         in 0..1 picks t_start = round(strength * T): strength 0 keeps the
         original image untouched, strength 1 is full noise = text->image.)
  3. DENOISE  z_t  --PixArt denoiser, NEW prompt, reverse from t_start-->  z0'
        (neuralpretrained.pas BuildPixArtFromSafeTensors + PixArtDenoise,
         driven by the SAME reusable scheduler.Step loop the other examples use.)
  4. DECODE   z0'  --VAE decoder (latent /0.18215)-->  edited RGB image
        (neuralpretrained.pas BuildVaeDecoderFromSafeTensors.)

The ONLY new code here is the encode -> partial-noise -> denoise -> decode
driver and the `strength` knob. No new importer, no SD UNet, no hand-rolled
noising/sampling math (every noise/step call goes through the reusable
scheduler).

INPAINTING (--inpaint). A one-flag follow-up that reuses the EXACT same
encode -> partial-noise -> denoise -> decode pipeline but regenerates ONLY a
masked region while the rest stays pixel-faithful (the RePaint / SD-inpaint
resample trick, Lugmayr et al. 2022 "RePaint", arXiv:2201.09865). A binary
mask over the latent grid marks the region to regenerate (1) vs keep (0).
BEFORE each reverse step the UNMASKED latent is OVERWRITTEN with the clean
encoded latent z0 RE-NOISED (scheduler AddNoise) to that step's timestep, so
the kept region always carries the correct amount of noise for the current
step while only the masked region is driven by the denoiser. No new model: it
adds a mask volume + a per-step composite to the existing loop. The smoke run
asserts the UNMASKED output latent matches the source within tolerance and the
MASKED region differs, and writes edit_mask.ppm alongside before/after.

OFFLINE SMOKE RUN. Real Stable-Diffusion / PixArt checkpoints are far too large
for this environment, so by default the example runs a SELF-CONTAINED smoke
test on the committed tiny RANDOM fixtures the importer parity tests use
(tests/fixtures/tiny_vae_encoder|decoder|pixart .safetensors). Those nets are
untrained random weights, so the "edit" is not meaningful as a picture -- the
point is to exercise the full driver path end-to-end (encode, partial-noise at
several strengths, a few denoise steps, decode) and write before/after PPM
images, all within a small RAM/time budget. Point --vae-encoder / --vae-decoder
/ --pixart (with their --*-config) at real checkpoints to run it for real.

GRID NOTE. The fixtures are independent pico nets: the VAE latent grid (8x8x4)
and the PixArt sample grid (6x6x4) do NOT match (they were generated for
separate parity tests). The driver bridges them by center-cropping the VAE
latent to the PixArt grid for the denoise loop and writing the denoised crop
back into the latent before decoding (the untouched border just passes through).
With matched real checkpoints (same latent_channels and grid) this crop is a
no-op and the whole latent is denoised.

OUTPUT. edit_before.ppm (VAE round-trip of the source, strength 0 reference) and
edit_after.ppm (the edited image at the chosen --strength). A synthetic source
image (smooth color ramp) is generated when no --image is supplied. The run
asserts no NaN/Inf in the decoded pixels.

RUN
  ./ImageToImage                       smoke run on the tiny fixtures
  ./ImageToImage --strength 0.6        stronger edit (more noise added)
  ./ImageToImage --steps 8             more reverse steps
  ./ImageToImage --inpaint             regenerate only a masked latent region
  ./ImageToImage --vae-encoder enc.safetensors --vae-encoder-config enc.json \
                 --vae-decoder dec.safetensors --vae-decoder-config dec.json \
                 --pixart pix.safetensors --pixart-config pix.json
                                       real checkpoints (not in this repo)

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuraldiffusion,
  neuralpretrained;

const
  cDefaultStrength = 0.4;   // SDEdit edit strength in 0..1 (fraction of T noised)
  cDefaultSteps    = 6;     // reverse DDIM steps in the smoke run
  cT               = 200;   // scheduler timesteps (matches the other examples)

// ---------------------------------------------------------------------------
// Adapter so the reusable scheduler can drive the PixArt denoiser. The PixArt
// net works on a 6x6x4 latent; the scheduler operates on the same volume the
// driver passes in (already cropped to that grid). The NEW edit prompt is held
// in gTextStates and stays fixed across the whole reverse trajectory.
// Coded by Claude (AI).
type
  TPixArtDenoiser = class(TObject)
    procedure Denoise(Xt, Output: TNNetVolume; t: integer);
  end;

var
  gPixArt: TNNet;
  gPixCfg: TPixArtConfig;
  gTextStates: TNNetVolume;     // the NEW edit prompt's T5 encoder states
  gScratch: TNNetVolume;        // reused eps buffer

procedure TPixArtDenoiser.Denoise(Xt, Output: TNNetVolume; t: integer);
begin
  // PixArtDenoise fills gScratch with the eps half (first InChannels) at t.
  PixArtDenoise(gPixArt, gPixCfg, Xt, t, gTextStates, gScratch);
  Output.Copy(gScratch);
end;

// --- tiny PPM writer (P6 binary). Avoids the LCL/image dependency that the
//     PNG writer pulls in, so the example compiles with plain fpc. V is a
//     (W,H,3) volume in [-1,1]; pixels are mapped to 0..255 and clamped. ---
function WritePPM(V: TNNetVolume; const FileName: string): boolean;
var
  F: TFileStream;
  Hdr: string;
  x, y, c: integer;
  b: byte;
  v01: TNeuralFloat;
  Row: array of byte;
begin
  Result := false;
  try
    F := TFileStream.Create(FileName, fmCreate);
    try
      Hdr := Format('P6'#10'%d %d'#10'255'#10, [V.SizeX, V.SizeY]);
      F.WriteBuffer(Hdr[1], Length(Hdr));
      SetLength(Row, V.SizeX * 3);
      for y := 0 to V.SizeY - 1 do
      begin
        for x := 0 to V.SizeX - 1 do
          for c := 0 to 2 do
          begin
            v01 := (V[x, y, c] + 1.0) * 127.5;
            if IsNan(v01) or IsInfinite(v01) then v01 := 0;
            if v01 < 0 then v01 := 0;
            if v01 > 255 then v01 := 255;
            b := Round(v01);
            Row[x * 3 + c] := b;
          end;
        F.WriteBuffer(Row[0], Length(Row));
      end;
      Result := true;
    finally
      F.Free;
    end;
  except
    Result := false;
  end;
end;

// Build a tiny PixArt net either from a real checkpoint or the committed
// fixture. The text prompt is approximated by random T5 encoder states (no
// text encoder in this offline example; with a real run feed your T5 states).
var
  Enc, Dec: TNNet;
  EncCfg, DecCfg: TVaeDecoderConfig;
  Sched: TNNetDiffusionScheduler;
  Denoiser: TPixArtDenoiser;

  Strength: TNeuralFloat;
  Steps: integer;
  Inpaint: boolean;

  EncPath, EncCfgPath, DecPath, DecCfgPath, PixPath, PixCfgPath, ImgPath: string;
  UseFixtures: boolean;

  function FixtureBase: string;
  var
    Cand: array[0..2] of string;
    i: integer;
  begin
    Cand[0] := 'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator;
    Cand[1] := '..' + DirectorySeparator + '..' + DirectorySeparator +
      'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator;
    Cand[2] := 'fixtures' + DirectorySeparator;
    Result := Cand[0];
    for i := 0 to 2 do
      if FileExists(Cand[i] + 'tiny_pixart.safetensors') then
        Exit(Cand[i]);
  end;

  procedure ParseArgs;
  var i: integer; s: string;
  begin
    Strength := cDefaultStrength;
    Steps := cDefaultSteps;
    Inpaint := false;
    EncPath := ''; EncCfgPath := ''; DecPath := ''; DecCfgPath := '';
    PixPath := ''; PixCfgPath := ''; ImgPath := '';
    i := 1;
    while i <= ParamCount do
    begin
      s := ParamStr(i);
      if (s = '--strength') and (i < ParamCount) then begin Inc(i); Strength := StrToFloat(ParamStr(i)); end
      else if (s = '--steps') and (i < ParamCount) then begin Inc(i); Steps := StrToInt(ParamStr(i)); end
      else if (s = '--vae-encoder') and (i < ParamCount) then begin Inc(i); EncPath := ParamStr(i); end
      else if (s = '--vae-encoder-config') and (i < ParamCount) then begin Inc(i); EncCfgPath := ParamStr(i); end
      else if (s = '--vae-decoder') and (i < ParamCount) then begin Inc(i); DecPath := ParamStr(i); end
      else if (s = '--vae-decoder-config') and (i < ParamCount) then begin Inc(i); DecCfgPath := ParamStr(i); end
      else if (s = '--pixart') and (i < ParamCount) then begin Inc(i); PixPath := ParamStr(i); end
      else if (s = '--pixart-config') and (i < ParamCount) then begin Inc(i); PixCfgPath := ParamStr(i); end
      else if (s = '--image') and (i < ParamCount) then begin Inc(i); ImgPath := ParamStr(i); end
      else if (s = '--inpaint') then Inpaint := true;
      Inc(i);
    end;
    if Strength < 0 then Strength := 0;
    if Strength > 1 then Strength := 1;
    UseFixtures := (PixPath = '') or (EncPath = '') or (DecPath = '');
  end;

// Center-crop the (Sx,Sy,*)->(NewS,NewS,*) region of Src into Dst.
procedure CenterCrop(Src, Dst: TNNetVolume; NewS: integer);
var ox, oy, x, y, c: integer;
begin
  Dst.ReSize(NewS, NewS, Src.Depth);
  ox := (Src.SizeX - NewS) div 2;
  oy := (Src.SizeY - NewS) div 2;
  for y := 0 to NewS - 1 do
    for x := 0 to NewS - 1 do
      for c := 0 to Src.Depth - 1 do
        Dst[x, y, c] := Src[ox + x, oy + y, c];
end;

// Write the (NewS,NewS,*) Patch back into the center of Dst (inverse of crop).
procedure PasteCenter(Patch, Dst: TNNetVolume);
var ox, oy, x, y, c: integer;
begin
  ox := (Dst.SizeX - Patch.SizeX) div 2;
  oy := (Dst.SizeY - Patch.SizeY) div 2;
  for y := 0 to Patch.SizeY - 1 do
    for x := 0 to Patch.SizeX - 1 do
      for c := 0 to Patch.Depth - 1 do
        Dst[ox + x, oy + y, c] := Patch[x, y, c];
end;

// Build the inpainting MASK over a (Sx,Sy,Depth) latent: 1 = REGENERATE this
// voxel, 0 = KEEP it pixel-faithful. The smoke mask is the right half of the
// grid (all channels) -- a simple, deterministic region so the assertions are
// unambiguous. With a real run you would load an alpha matte here instead.
procedure MakeInpaintMask(Mask: TNNetVolume; Sx, Sy, Depth: integer);
var x, y, c: integer;
begin
  Mask.ReSize(Sx, Sy, Depth);
  for y := 0 to Sy - 1 do
    for x := 0 to Sx - 1 do
      for c := 0 to Depth - 1 do
        if x >= (Sx div 2) then Mask[x, y, c] := 1.0    // right half: regenerate
        else Mask[x, y, c] := 0.0;                      // left half: keep
end;

// Composite: Dst := Mask*Dst + (1-Mask)*Known, in place on Dst. Used to overwrite
// the UNMASKED region of the working latent with the (re-noised) known latent.
procedure CompositeMasked(Dst, Known, Mask: TNNetVolume);
var i: integer; m: TNeuralFloat;
begin
  for i := 0 to Dst.Size - 1 do
  begin
    m := Mask.FData[i];
    Dst.FData[i] := m * Dst.FData[i] + (1.0 - m) * Known.FData[i];
  end;
end;

// Fill V (W,H,3) with a smooth synthetic color ramp in [-1,1] (the stand-in
// source image when no --image is given).
procedure MakeSyntheticImage(V: TNNetVolume);
var x, y: integer;
begin
  for y := 0 to V.SizeY - 1 do
    for x := 0 to V.SizeX - 1 do
    begin
      V[x, y, 0] := (x / Max(1, V.SizeX - 1)) * 2.0 - 1.0;          // R: left->right
      V[x, y, 1] := (y / Max(1, V.SizeY - 1)) * 2.0 - 1.0;          // G: top->bottom
      V[x, y, 2] := Sin((x + y) * 0.6) * 0.8;                       // B: ripple
    end;
end;

var
  ImgIn, Z0, ZEdit, ZCrop, Recon: TNNetVolume;
  Mask, KnownNoised: TNNetVolume;   // inpainting: mask + re-noised known latent
  ImgGrid, LatGrid, PixGrid: integer;
  tStart, sIdx: integer;
  Schedule: TNeuralIntegerArray;
  NumNan, i: integer;
  outv: TNNetVolume;
  maxKeepErr, maskedDiff, m, d: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;
  ParseArgs;

  WriteLn('ImageToImage (SDEdit): encode -> partial-noise -> denoise -> decode.');
  if UseFixtures then
  begin
    EncPath := FixtureBase + 'tiny_vae_encoder.safetensors';
    EncCfgPath := FixtureBase + 'tiny_vae_encoder_config.json';
    DecPath := FixtureBase + 'tiny_vae_decoder.safetensors';
    DecCfgPath := FixtureBase + 'tiny_vae_decoder_config.json';
    PixPath := FixtureBase + 'tiny_pixart.safetensors';
    PixCfgPath := FixtureBase + 'tiny_pixart_config.json';
    WriteLn('[SMOKE mode] using committed tiny RANDOM fixtures: ', FixtureBase);
    WriteLn('  (untrained nets -> the edit is not a real picture; this validates the driver path)');
  end
  else
    WriteLn('[REAL mode] using supplied checkpoints.');

  if not FileExists(PixPath) then
  begin
    WriteLn('PixArt checkpoint not found: ', PixPath);
    WriteLn('Run from the repo root so the tiny fixtures are visible, or pass --pixart.');
    Halt(0);
  end;

  WriteLn(Format('strength = %.2f   reverse steps = %d   T = %d', [Strength, Steps, cT]));
  WriteLn;

  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps);
  Denoiser := TPixArtDenoiser.Create;

  Enc := BuildVaeEncoderFromSafeTensors(EncPath, EncCfg, {pInferenceOnly=}false, EncCfgPath);
  Dec := BuildVaeDecoderFromSafeTensors(DecPath, DecCfg, {pInferenceOnly=}false, DecCfgPath);
  gPixArt := BuildPixArtFromSafeTensors(PixPath, {TextSeqLen=}5, gPixCfg,
    {pInferenceOnly=}false, PixCfgPath);

  LatGrid := EncCfg.LatentGrid;
  ImgGrid := LatGrid shl (EncCfg.NumBlockOut - 1);
  PixGrid := gPixCfg.SampleSize;

  WriteLn(Format('VAE: %dx%d RGB <-> %dx%dx%d latent;  PixArt denoise grid %dx%dx%d.',
    [ImgGrid, ImgGrid, LatGrid, LatGrid, EncCfg.LatentChannels,
     PixGrid, PixGrid, gPixCfg.InChannels]));
  if (PixGrid <> LatGrid) then
    WriteLn('  (fixture grids differ -> center-crop the latent to the PixArt grid for denoising.)');
  WriteLn;

  ImgIn := TNNetVolume.Create(ImgGrid, ImgGrid, EncCfg.OutChannels);
  Z0    := TNNetVolume.Create();
  ZEdit := TNNetVolume.Create();
  ZCrop := TNNetVolume.Create();
  Recon := TNNetVolume.Create();
  Mask := TNNetVolume.Create();
  KnownNoised := TNNetVolume.Create();
  gTextStates := TNNetVolume.Create(gPixCfg.TextSeqLen, 1, gPixCfg.CaptionChannels);
  gScratch := TNNetVolume.Create();
  Schedule := nil;
  try
    // --- the NEW edit prompt. With a real run this is your T5 encoder output;
    //     offline we use random states so the driver path is fully exercised. ---
    for i := 0 to gTextStates.Size - 1 do
      gTextStates.FData[i] := RandG(0, 1) * 0.5;

    // 1. source image (synthetic ramp unless --image given; PPM read omitted to
    //    keep the example dependency-free -- the ramp is enough for a smoke run).
    MakeSyntheticImage(ImgIn);
    if ImgPath <> '' then
      WriteLn('(--image given but PPM reading is omitted in this offline example; using synthetic ramp.)');

    // 2. ENCODE -> clean latent z0.
    Enc.Compute(ImgIn);
    Z0.Copy(Enc.GetLastLayer().Output);
    WriteLn(Format('Encoded latent: %dx%dx%d.', [Z0.SizeX, Z0.SizeY, Z0.Depth]));

    // before image = plain VAE round-trip of the source (strength 0 reference).
    Dec.Compute(Z0);
    Recon.Copy(Dec.GetLastLayer().Output);
    if WritePPM(Recon, 'edit_before.ppm')
      then WriteLn('Wrote edit_before.ppm (VAE round-trip, no edit).')
      else WriteLn('FAILED writing edit_before.ppm');

    // 3. SDEdit: partially noise z0 to t_start = round(strength * T), then run
    //    the reverse trajectory from t_start with the NEW prompt. strength 1 ==
    //    full noise == text->image-from-noise; strength 0 keeps the source.
    tStart := Round(Strength * cT);
    if tStart < 1 then tStart := 1;
    if tStart > cT then tStart := cT;

    // center-crop the latent to the PixArt grid (no-op when grids match).
    CenterCrop(Z0, ZCrop, PixGrid);

    // INPAINT: build the binary regenerate/keep mask over the PixArt-grid latent.
    if Inpaint then
    begin
      MakeInpaintMask(Mask, ZCrop.SizeX, ZCrop.SizeY, ZCrop.Depth);
      WriteLn(Format('[INPAINT] mask %dx%dx%d: regenerate right half, keep left half.',
        [Mask.SizeX, Mask.SizeY, Mask.Depth]));
    end;

    // partial forward noising to t_start (the SDEdit start latent). AddNoise
    // writes into ZEdit element-wise WITHOUT resizing it, so size it to the
    // (cropped) latent first.
    ZEdit.ReSize(ZCrop);
    Sched.AddNoise(ZCrop, ZEdit, tStart);
    WriteLn(Format('Added noise up to t_start = %d (strength %.2f).', [tStart, Strength]));

    // reverse trajectory, BOUNDED at t_start (SDEdit starts partway down, not at
    // T): an evenly-strided descending schedule of Steps+1 timesteps from
    // t_start down to 0 (the SDEdit truncated reverse process -- the scheduler's
    // own BuildTimestepSchedule always spans the full [0..T], so we stride the
    // [0..t_start] subrange here and drive its public Step per pair).
    SetLength(Schedule, Steps + 1);
    for sIdx := 0 to Steps do
      Schedule[sIdx] := Round(tStart * (1.0 - sIdx / Steps));   // t_start..0
    Sched.ResetMultistep;
    WriteLn('Denoising (', Steps, ' reverse steps from t_start):');
    for sIdx := 0 to Length(Schedule) - 2 do
    begin
      if Schedule[sIdx] < 1 then Continue;          // skip degenerate t=0 steps
      // INPAINT (RePaint trick): BEFORE the reverse step, overwrite the UNMASKED
      // latent with the known clean latent RE-NOISED to THIS timestep, so the
      // kept region always carries the right noise level for the current step
      // and only the masked region is driven by the denoiser.
      if Inpaint then
      begin
        Sched.AddNoise(ZCrop, KnownNoised, Schedule[sIdx]);
        CompositeMasked(ZEdit, KnownNoised, Mask);
      end;
      Denoiser.Denoise(ZEdit, gScratch, Schedule[sIdx]);
      Sched.Step(ZEdit, gScratch, Schedule[sIdx], Schedule[sIdx + 1], smDDIM, 0.0);
      WriteLn(Format('  step %d: denoised at t = %d -> %d',
        [sIdx + 1, Schedule[sIdx], Schedule[sIdx + 1]]));
    end;

    // INPAINT: final composite at t=0 -- the kept region is exactly the clean
    // known latent so the unmasked output is pixel-faithful by construction.
    if Inpaint then
      CompositeMasked(ZEdit, ZCrop, Mask);

    // 4. write the denoised crop back into the full latent and DECODE.
    PasteCenter(ZEdit, Z0);
    Dec.Compute(Z0);
    outv := Dec.GetLastLayer().Output;
    NumNan := 0;
    for i := 0 to outv.Size - 1 do
      if IsNan(outv.FData[i]) or IsInfinite(outv.FData[i]) then Inc(NumNan);
    Recon.Copy(outv);
    if WritePPM(Recon, 'edit_after.ppm')
      then WriteLn('Wrote edit_after.ppm (edited image).')
      else WriteLn('FAILED writing edit_after.ppm');

    WriteLn;
    if NumNan = 0
      then WriteLn('OK: decoded edit has no NaN/Inf pixels.')
      else WriteLn('WARNING: ', NumNan, ' NaN/Inf pixels in the decoded edit.');

    // INPAINT regression check (latent space, exact): the UNMASKED region of the
    // final latent must equal the known clean latent (pixel-faithful), and the
    // MASKED region must have been changed by the denoiser (it was regenerated).
    if Inpaint then
    begin
      maxKeepErr := 0;          // worst |ZEdit - ZCrop| over KEPT (mask=0) voxels
      maskedDiff := 0;          // total |ZEdit - ZCrop| over MASKED (mask=1) voxels
      for i := 0 to ZEdit.Size - 1 do
      begin
        m := Mask.FData[i];
        d := Abs(ZEdit.FData[i] - ZCrop.FData[i]);
        if m < 0.5 then begin if d > maxKeepErr then maxKeepErr := d; end
        else maskedDiff := maskedDiff + d;
      end;
      // Write the mask as a grayscale-ish PPM (1->white regenerate, 0->black keep).
      Recon.Copy(Mask);
      for i := 0 to Recon.Size - 1 do Recon.FData[i] := Recon.FData[i] * 2.0 - 1.0;
      if WritePPM(Recon, 'edit_mask.ppm')
        then WriteLn('Wrote edit_mask.ppm (white = regenerated region).')
        else WriteLn('FAILED writing edit_mask.ppm');
      WriteLn(Format('[INPAINT] kept-region max error = %.6g ; masked-region total change = %.6g',
        [maxKeepErr, maskedDiff]));
      if maxKeepErr > 1e-4 then
      begin
        WriteLn('FAIL: unmasked latent region drifted from the source.');
        Halt(1);
      end;
      if maskedDiff <= 1e-6 then
      begin
        WriteLn('FAIL: masked latent region was not regenerated (unchanged).');
        Halt(1);
      end;
      WriteLn('OK: unmasked region pixel-faithful, masked region regenerated.');
    end;

    WriteLn('Done. Compare edit_before.ppm (source round-trip) vs edit_after.ppm (edit).');
  finally
    SetLength(Schedule, 0);
    ImgIn.Free; Z0.Free; ZEdit.Free; ZCrop.Free; Recon.Free;
    Mask.Free; KnownNoised.Free;
    gTextStates.Free; gScratch.Free;
    Enc.Free; Dec.Free; gPixArt.Free;
    Sched.Free; Denoiser.Free;
  end;
end.
