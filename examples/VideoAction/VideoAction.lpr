program VideoAction;
(*
VideoAction: classifies a short synthetic video CLIP with an IMPORTED VideoMAE
spatiotemporal transformer (BuildVideoMAEFromSafeTensors) - the repo's first
video-classification importer (a clip of T frames -> an action label). Pure CPU,
no external dataset, finishes in well under a second.

WHAT IT SHOWS
-------------
The committed pico checkpoint tests/fixtures/tiny_videomae.safetensors (a
randomly-initialized HF VideoMAEForVideoClassification, the SAME fixture the
parity test asserts float64-exact against) is loaded end-to-end:
  tubelet 3-D conv (TNNetConvolution3D) -> fixed sin-cos 3-D position table
  -> stock pre-LN transformer encoder (joint space-time attention)
  -> mean-pool over tokens -> fc_norm -> linear classifier -> action logits.

A clip is a (num_frames, H, W, C) volume packed for the importer as
(W, H, num_frames*C) (frame t's C channels contiguous at depth [t*C..t*C+C)).
The program synthesizes two clips - a bright blob sliding RIGHT and one sliding
DOWN - feeds each through the imported net, and prints the per-class action
logits + the argmax label. (The pico weights are random, so the predicted label
is not semantically meaningful; the point is the importer + forward path runs on
CPU and produces calibrated logits.)

Build (from the repo root):
  lazbuild examples/VideoAction/VideoAction.lpi
or with fpc:
  fpc -O3 -Mobjfpc -Sh -Funeural examples/VideoAction/VideoAction.lpr

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralpretrained;

const
  cFixtureDir = 'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator;

// Writes a synthetic clip into Clip (already sized (W, H, num_frames*C)): a
// bright 1x1 blob starting near a corner and sliding one pixel per frame in
// direction ADir (0 = right, 1 = down). Frame t's C channels live at depth
// [t*C .. t*C+C) - the TNNetConvolution3D frame-packing convention.
procedure MakeClip(Clip: TNNetVolume; const Config: TVideoMAEConfig;
  ADir: integer);
var
  t, c, bx, by, dx, dy: integer;
begin
  Clip.Fill(0);
  dx := 0; dy := 0;
  if ADir = 0 then dx := 1 else dy := 1;
  for t := 0 to Config.NumFrames - 1 do
  begin
    bx := 1 + dx * t;
    by := 1 + dy * t;
    if (bx >= 0) and (bx < Config.ImageSize) and
       (by >= 0) and (by < Config.ImageSize) then
      for c := 0 to Config.NumChannels - 1 do
        Clip[bx, by, t * Config.NumChannels + c] := 1.0;
  end;
end;

procedure ClassifyClip(Net: TNNet; const Config: TVideoMAEConfig;
  const Name: string; ADir: integer);
var
  Clip, Logits: TNNetVolume;
  i, best: integer;
  bestVal: TNeuralFloat;
  line: string;
begin
  Clip := TNNetVolume.Create(Config.ImageSize, Config.ImageSize,
    Config.NumFrames * Config.NumChannels);
  Logits := TNNetVolume.Create();
  try
    MakeClip(Clip, Config, ADir);
    RunVideoMAELogits(Net, Clip, Logits);
    best := 0;
    bestVal := Logits.FData[0];
    line := '';
    for i := 0 to Logits.Size - 1 do
    begin
      line := line + Format(' %7.4f', [Logits.FData[i]]);
      if Logits.FData[i] > bestVal then
      begin
        bestVal := Logits.FData[i];
        best := i;
      end;
    end;
    WriteLn(Format('  %-14s logits=[%s ]  -> action class %d',
      [Name, line, best]));
  finally
    Clip.Free;
    Logits.Free;
  end;
end;

procedure RunAlgo();
var
  Net: TNNet;
  Config: TVideoMAEConfig;
  FixturePath, ConfigPath: string;
begin
  FixturePath := cFixtureDir + 'tiny_videomae.safetensors';
  ConfigPath := cFixtureDir + 'tiny_videomae_config.json';
  if not FileExists(FixturePath) then
  begin
    WriteLn('Could not find ', FixturePath);
    WriteLn('Run this example from the repo root (the directory that holds ',
      '"tests/fixtures/").');
    Exit;
  end;

  WriteLn('VideoAction: imported VideoMAE video-classification transformer');
  Net := BuildVideoMAEFromSafeTensorsEx(FixturePath, Config,
    {pInferenceOnly=}true, ConfigPath);
  try
    WriteLn('  ', VideoMAEConfigToString(Config));
    WriteLn(Format('  clip = %d frames of %dx%d x %d channels, %d tubelet ' +
      'tokens, %d action classes',
      [Config.NumFrames, Config.ImageSize, Config.ImageSize,
       Config.NumChannels,
       (Config.NumFrames div Config.TubeletSize) *
       (Config.ImageSize div Config.PatchSize) *
       (Config.ImageSize div Config.PatchSize),
       Config.NumLabels]));
    WriteLn;
    WriteLn('Classifying two synthetic clips (pico random weights):');
    ClassifyClip(Net, Config, 'blob-right', 0);
    ClassifyClip(Net, Config, 'blob-down', 1);
    WriteLn;
    WriteLn('=> The imported tubelet-conv + transformer ran on CPU and ',
      'produced action logits.');
  finally
    Net.Free;
  end;
end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'VideoAction Example';
  RunAlgo();
end.
