program MarigoldDepth;
(*
MarigoldDepth: monocular depth or surface normals with Marigold (Ke et al.
2024, arXiv:2312.02145) through TMarigoldPipeline in neuralpretrained.pas.

Two modes:
  * no --model: a smoke run on the committed random pico fixtures
    (tests/fixtures/tiny_sd_unet_linproj, tiny_vae_encoder, tiny_vae_decoder).
    The example assembles a diffusers-layout Marigold folder from them in the
    temp directory, synthesizes a 24x16 PNG, and writes a depth PNG and a
    normals PNG. The pico weights are random: this checks wiring and I/O only.
  * --model DIR --input IMG --output PNG: a diffusers Marigold folder
    (prs-eth/marigold-depth-v1-1 or prs-eth/marigold-normals-v1-1). The task
    (depth or normals) comes from model_index.json.

Depth is written as a 16-bit grayscale PNG (0 = near ... 65535 = far for the
depth checkpoints), normals as an RGB PNG of (n + 1) / 2.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, fpjson, jsonparser,
  {$IFDEF OpenCL}neuralopencl,{$ENDIF}
  neuralnetwork, neuralvolume, neuraldatasets, neuralsafetensors,
  neuralpretrained;

const
  FixturesDir = '../../tests/fixtures/';

type
  TMarigoldOptions = record
    ModelFolder, InputFile, OutputFile: string;
    // < 1 / < 0: the checkpoint's model_index.json defaults.
    Steps, ProcessingResolution: integer;
    EnsembleSize: integer;
    Seed: cardinal;
    QuantizeInt8: boolean;
    UseOpenCL: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: MarigoldDepth                          (pico fixture smoke)');
  WriteLn('       MarigoldDepth --model DIR --input IMG --output PNG [options]');
  WriteLn('  --model DIR          diffusers Marigold folder (model_index.json,');
  WriteLn('                       unet/, vae/, text_encoder/, scheduler/)');
  WriteLn('  --input IMG          image (PNG / JPEG / BMP)');
  WriteLn('  --output PNG         depth: 16-bit grayscale PNG; normals: RGB PNG');
  WriteLn('  --steps N            denoising steps (default: ',
    'default_denoising_steps, 4 for v1-1)');
  WriteLn('  --ensemble N         predictions from N noises, ensembled ',
    '(default 1)');
  WriteLn('  --processing-res N   long side the model sees; 0 = native ',
    '(default: default_processing_resolution, 768 for v1-1)');
  WriteLn('  --seed N             noise seed (default 0)');
  WriteLn('  --int8               int8 UNet weights');
  WriteLn('  --gpu                run the nets on OpenCL (needs a build with ',
    '-dOpenCL)');
  WriteLn('  --gpu-platform N     OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N       OpenCL device index (default 0)');
end;

function ParseOptions(out Opt: TMarigoldOptions): boolean;
var
  ArgPos: integer;
  Arg: string;

  function NextArg(): string;
  begin
    Inc(ArgPos);
    if ArgPos > ParamCount then
      raise Exception.Create(Arg + ' needs a value');
    Result := ParamStr(ArgPos);
  end;

begin
  Opt.ModelFolder := '';
  Opt.InputFile := '';
  Opt.OutputFile := '';
  Opt.Steps := 0;
  Opt.ProcessingResolution := -1;
  Opt.EnsembleSize := 1;
  Opt.Seed := 0;
  Opt.QuantizeInt8 := false;
  Opt.UseOpenCL := false;
  Opt.OpenCLPlatform := 0;
  Opt.OpenCLDevice := 0;
  Result := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--model' then Opt.ModelFolder := NextArg()
    else if Arg = '--input' then Opt.InputFile := NextArg()
    else if Arg = '--output' then Opt.OutputFile := NextArg()
    else if Arg = '--steps' then Opt.Steps := StrToInt(NextArg())
    else if Arg = '--ensemble' then Opt.EnsembleSize := StrToInt(NextArg())
    else if Arg = '--processing-res' then
      Opt.ProcessingResolution := StrToInt(NextArg())
    else if Arg = '--seed' then Opt.Seed := StrToInt64(NextArg())
    else if Arg = '--int8' then Opt.QuantizeInt8 := true
    else if Arg = '--gpu' then Opt.UseOpenCL := true
    else if Arg = '--gpu-platform' then Opt.OpenCLPlatform := StrToInt(NextArg())
    else if Arg = '--gpu-device' then Opt.OpenCLDevice := StrToInt(NextArg())
    else if (Arg = '--help') or (Arg = '-h') then
    begin
      PrintUsage();
      Result := false;
      exit;
    end
    else
      raise Exception.Create('unknown argument: ' + Arg);
    Inc(ArgPos);
  end;
  if (Opt.ModelFolder <> '') and
     ((Opt.InputFile = '') or (Opt.OutputFile = '')) then
    raise Exception.Create('--model needs --input and --output');
end;

// Depth (W,H,1) in [0,1] -> 16-bit gray PNG; normals (W,H,3) unit -> RGB PNG.
procedure SavePrediction(Prediction: TNNetVolume; PredictNormals: boolean;
  const FileName: string);
var
  Rgb: TNNetVolume;
begin
  if not PredictNormals then
  begin
    if not SaveGray16PNGFromVolume(Prediction, FileName) then
      raise Exception.Create('could not save ' + FileName);
    exit;
  end;
  Rgb := TNNetVolume.Create(Prediction);
  try
    Rgb.Add(1);
    Rgb.Mul(127.5);
    if not SaveImageFromVolumeIntoFile(Rgb, FileName) then
      raise Exception.Create('could not save ' + FileName);
  finally
    Rgb.Free;
  end;
end;

// Runs Pipeline on Opt.InputFile and writes Opt.OutputFile.
procedure PredictFile(Pipeline: TMarigoldPipeline; const Opt: TMarigoldOptions);
var
  Image, Prediction: TNNetVolume;
  ComputeText: string;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  OpenCLProblem: string;
  OpenCLPlatform, OpenCLDevice: integer;
  {$ENDIF}
begin
  Image := TNNetVolume.Create;
  Prediction := TNNetVolume.Create;
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
    if not LoadImageFromFileIntoVolume(Opt.InputFile, Image) then
      raise Exception.Create('could not load ' + Opt.InputFile);
    Image.Divi(255.0);
    Pipeline.QuantizeInt8 := Opt.QuantizeInt8;
    ComputeText := 'CPU';
    if Opt.UseOpenCL then
    begin
      {$IFDEF OpenCL}
      OpenCLDevices := TEasyOpenCL.Create();
      OpenCLPlatform := Opt.OpenCLPlatform;
      OpenCLDevice := Opt.OpenCLDevice;
      if OpenCLDevices.SelectPlatformAndDevice(OpenCLPlatform, OpenCLDevice,
        OpenCLProblem) then
      begin
        Pipeline.EnableOpenCL(OpenCLDevices.PlatformIds[OpenCLPlatform],
          OpenCLDevices.Devices[OpenCLDevice]);
        ComputeText := 'OpenCL on ' +
          OpenCLDevices.PlatformNames[OpenCLPlatform] + ' / ' +
          OpenCLDevices.DeviceNames[OpenCLDevice];
      end
      else
        ComputeText := 'CPU (' + OpenCLProblem + ')';
      {$ELSE}
      ComputeText := 'CPU (built without -dOpenCL)';
      {$ENDIF}
    end;
    if Opt.QuantizeInt8 then ComputeText := ComputeText + ', int8 UNet';
    WriteLn(Format('Marigold %s (prediction_type "%s") on %dx%d, %s',
      [BoolToStr(Pipeline.PredictNormals, 'normals', 'depth'),
       Pipeline.PredictionType, Image.SizeX, Image.SizeY, ComputeText]));
    StartTime := Now();
    Pipeline.Predict(Image, Opt.Steps, Opt.EnsembleSize,
      Opt.ProcessingResolution, Opt.Seed, Prediction);
    WriteLn(Format('  %.1f s, range %.4f .. %.4f', [(Now() - StartTime) * 86400.0,
      Prediction.GetMin(), Prediction.GetMax()]));
    SavePrediction(Prediction, Pipeline.PredictNormals, Opt.OutputFile);
    WriteLn('  wrote ', Opt.OutputFile);
  finally
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    Prediction.Free;
    Image.Free;
  end;
end;

// A diffusers-layout Marigold folder over the pico UNet (conv_in widened to
// 8 channels) and pico VAE; no text_encoder/ (the smoke sets the states).
procedure WritePicoFolder(const Folder, PipelineClass: string);
var
  Lines: TStringList;
  Root: TJSONData;
  PredictionType: string;
begin
  ForceDirectories(Folder + 'scheduler');
  ForceDirectories(Folder + 'unet');
  ForceDirectories(Folder + 'vae');
  Lines := TStringList.Create;
  Root := nil;
  try
    if PipelineClass = 'MarigoldNormalsPipeline' then PredictionType := 'normals'
    else PredictionType := 'depth';
    Lines.Text := '{"_class_name": "' + PipelineClass + '", ' +
      '"prediction_type": "' + PredictionType + '", "default_denoising_steps": 2, ' +
      '"default_processing_resolution": 16, "use_full_z_range": true}';
    Lines.SaveToFile(Folder + 'model_index.json');
    // prs-eth/marigold-depth-v1-1 scheduler settings.
    Lines.Text := '{"_class_name": "DDIMScheduler", "beta_end": 0.012, ' +
      '"beta_schedule": "scaled_linear", "beta_start": 0.00085, ' +
      '"clip_sample": false, "num_train_timesteps": 1000, ' +
      '"prediction_type": "v_prediction", "rescale_betas_zero_snr": true, ' +
      '"set_alpha_to_one": false, "timestep_spacing": "trailing"}';
    Lines.SaveToFile(Folder + 'scheduler' + DirectorySeparator +
      'scheduler_config.json');
    Lines.LoadFromFile(FixturesDir + 'tiny_sd_unet_linproj_config.json');
    Root := GetJSON(Lines.Text);
    TJSONObject(Root).Integers['in_channels'] := 8;
    Lines.Text := Root.FormatJSON();
    Lines.SaveToFile(Folder + 'unet' + DirectorySeparator + 'config.json');
    MarigoldWidenUNetConvIn(FixturesDir + 'tiny_sd_unet_linproj.safetensors',
      Folder + 'unet' + DirectorySeparator + 'diffusion_pytorch_model.safetensors');
    Lines.LoadFromFile(FixturesDir + 'tiny_vae_encoder_config.json');
    Lines.SaveToFile(Folder + 'vae' + DirectorySeparator + 'config.json');
    MergeSafeTensorsFiles([FixturesDir + 'tiny_vae_encoder.safetensors',
      FixturesDir + 'tiny_vae_decoder.safetensors'],
      Folder + 'vae' + DirectorySeparator + 'diffusion_pytorch_model.safetensors');
  finally
    Root.Free;
    Lines.Free;
  end;
end;

// Builds a 24x16 RGB gradient (0..255) so the smoke needs no input asset.
procedure MakeTinyImage(V: TNNetVolume);
var
  X, Y: integer;
begin
  V.ReSize(24, 16, 3);
  for X := 0 to 23 do
    for Y := 0 to 15 do
    begin
      V[X, Y, 0] := (X * 255) div 23;
      V[X, Y, 1] := (Y * 255) div 15;
      V[X, Y, 2] := ((X + Y) * 255) div 38;
    end;
end;

// Pico smoke: depth and normals from the random pico fixtures.
procedure RunPicoSmoke(BaseOpt: TMarigoldOptions);
const
  InPng = 'marigold_in.png';
var
  Img, PromptStates: TNNetVolume;
  Pipeline: TMarigoldPipeline;
  Folder: string;
  Opt: TMarigoldOptions;
  TaskPos: integer;
begin
  Img := TNNetVolume.Create;
  try
    MakeTinyImage(Img);
    if not SaveImageFromVolumeIntoFile(Img, InPng) then
      raise Exception.Create('could not save ' + InPng);
  finally
    Img.Free;
  end;
  Opt := BaseOpt;
  Opt.InputFile := InPng;
  for TaskPos := 0 to 1 do
  begin
    Folder := IncludeTrailingPathDelimiter(GetTempDir(false) +
      'cai_marigold_pico_example');
    if TaskPos = 0 then
    begin
      WritePicoFolder(Folder, 'MarigoldDepthPipeline');
      Opt.OutputFile := 'marigold_depth.png';
    end
    else
    begin
      WritePicoFolder(Folder, 'MarigoldNormalsPipeline');
      Opt.OutputFile := 'marigold_normals.png';
    end;
    Pipeline := TMarigoldPipeline.Create(Folder);
    PromptStates := TNNetVolume.Create(2, 1,
      Pipeline.UNetConfig.CrossAttentionDim);
    try
      // Stands in for text_encoder/: the pico CLIP is wider than the UNet's
      // cross_attention_dim.
      RandSeed := 1;
      PromptStates.RandomizeGaussian(0.5);
      Pipeline.SetEmptyPromptStates(PromptStates);
      PredictFile(Pipeline, Opt);
    finally
      PromptStates.Free;
      Pipeline.Free;
    end;
  end;
end;

var
  Options: TMarigoldOptions;
  Pipeline: TMarigoldPipeline;
begin
  try
    if not ParseOptions(Options) then exit;
    if Options.ModelFolder <> '' then
    begin
      Pipeline := TMarigoldPipeline.Create(Options.ModelFolder);
      try
        PredictFile(Pipeline, Options);
      finally
        Pipeline.Free;
      end;
    end
    else
    begin
      WriteLn('Marigold depth / normals smoke (random pico weights)');
      WriteLn('====================================================');
      if Options.EnsembleSize = 1 then Options.EnsembleSize := 3;
      RunPicoSmoke(Options);
      WriteLn('Done.');
    end;
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
