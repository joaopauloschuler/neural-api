program EffectiveReceptiveField;
(*
EffectiveReceptiveField: SWEEPS a conv stem over several kernel-size /
stack-depth configurations and charts how the EMPIRICAL (gradient-measured)
effective receptive field grows against the THEORETICAL receptive field.

The Luo et al. 2016 headline ("Understanding the Effective Receptive Field in
Deep Convolutional Neural Networks"): a deep stack's effective RF grows only
SUB-LINEARLY in the theoretical RF — composing many small kernels concentrates
the centre-unit gradient into a tight, roughly Gaussian blob, so the unit
COULD see its whole theoretical window but only really WEIGHTS the middle of
it, and that gap WIDENS as the stack gets deeper.

For each stem config the program:
  * builds an untrained single-channel conv stem on a small synthetic grid,
  * runs TNNet.EffectiveReceptiveFieldReport over a synthetic probe batch,
  * writes the (radius, cumulative-mass-fraction) CSV side-output (the new
    optional CsvFile argument) so the growth curve can be plotted, and
  * extracts the effective-RF radius and the theoretical RF from the report,
then prints a compact table of (config, theoretical_RF, effective_RF_radius,
ratio) so the sub-linear growth is visible at a glance.

The configs are sized so the THEORETICAL RF roughly DOUBLES across the sweep
while the EFFECTIVE radius grows much more slowly — the sub-linear story.

Forward+backward only (the centre-unit gradient) on frozen, untrained nets:
no weight update, no dataset download. Tiny tensors (a 21x21x1 grid, a 16-probe
batch), so it runs in well under a second on 2 CPU cores.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cInputSize = 21;   // 21x21x1 synthetic input plane (room for the deeper RFs)
  cProbes    = 16;   // synthetic probe batch size
  cFeatures  = 4;    // channels per conv layer (kept tiny)

type
  TStemCfg = record
    Name: string;
    KSize: integer;    // square kernel side
    NLayers: integer;  // number of stacked stride-1 convs
  end;

// The sweep: kernel-size AND stack-depth variations whose theoretical RF
// climbs roughly geometrically. Theoretical RF along an axis for N stacked
// stride-1 convs of side K is  1 + N*(K-1).
const
  cConfigs: array[0..4] of TStemCfg = (
    (Name: '1x 3x3';  KSize: 3; NLayers: 1),   // theo RF = 3
    (Name: '2x 3x3';  KSize: 3; NLayers: 2),   // theo RF = 5
    (Name: '4x 3x3';  KSize: 3; NLayers: 4),   // theo RF = 9
    (Name: '1x 9x9';  KSize: 9; NLayers: 1),   // theo RF = 9
    (Name: '4x 5x5';  KSize: 5; NLayers: 4)    // theo RF = 17
  );

// Build an untrained single-channel conv stem from a config.
procedure BuildStem(const Cfg: TStemCfg; out NN: TNNet);
var
  L, Pad: integer;
begin
  Pad := Cfg.KSize div 2;   // 'same' padding keeps the map spatial
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cInputSize, cInputSize, 1));
  for L := 1 to Cfg.NLayers do
    NN.AddLayer(TNNetConvolutionReLU.Create(cFeatures, Cfg.KSize, Pad, 1));
  NN.InitWeights();
end;

// Deterministic synthetic probe batch (no dataset download).
procedure BuildProbes(out Probes: TNNetVolumeList);
var
  I, X, Y: integer;
  V: TNNetVolume;
begin
  Probes := TNNetVolumeList.Create();
  RandSeed := 20260607;
  for I := 1 to cProbes do
  begin
    V := TNNetVolume.Create(cInputSize, cInputSize, 1);
    for Y := 0 to cInputSize - 1 do
      for X := 0 to cInputSize - 1 do
        V[X, Y, 0] := Random - 0.5;
    Probes.Add(V);
  end;
end;

// Pull the integer that follows Marker (up to the next non-digit) out of S.
// Returns -1 if Marker is not present.
function ReadIntAfter(const S, Marker: string): integer;
var
  P, E: integer;
  Tok: string;
begin
  Result := -1;
  P := Pos(Marker, S);
  if P <= 0 then Exit;
  P := P + Length(Marker);
  while (P <= Length(S)) and (S[P] = ' ') do Inc(P);
  E := P;
  while (E <= Length(S)) and (S[E] in ['0'..'9']) do Inc(E);
  Tok := Copy(S, P, E - P);
  if Tok <> '' then Result := StrToIntDef(Tok, -1);
end;

// Pull the first float after Marker (a "<float> x" pattern) out of S.
function ReadTheoRF(const S: string): integer;
var
  P, E: integer;
  Tok: string;
begin
  Result := -1;
  P := Pos('Theoretical RF (analytical, ReceptiveFieldReport): ', S);
  if P <= 0 then Exit;
  P := P + Length('Theoretical RF (analytical, ReceptiveFieldReport): ');
  E := P;
  while (E <= Length(S)) and (S[E] in ['0'..'9']) do Inc(E);
  Tok := Copy(S, P, E - P);
  if Tok <> '' then Result := StrToIntDef(Tok, -1);
end;

var
  NN: TNNet;
  Probes: TNNetVolumeList;
  Report, CsvName: string;
  I, TheoRF, EffRad, EffDiam: integer;
  Ratio: TNeuralFloat;
begin
  BuildProbes(Probes);
  try
    WriteLn('=== Effective receptive-field growth sweep (Luo et al. 2016) ===');
    WriteLn('Input grid ', cInputSize, 'x', cInputSize,
      'x1, ', cProbes, ' synthetic probes, untrained frozen stems.');
    WriteLn;
    WriteLn(Format('%-10s %12s %14s %10s', ['config',
      'theo_RF', 'eff_RF_diam', 'eff/theo']));
    WriteLn(StringOfChar('-', 50));

    for I := Low(cConfigs) to High(cConfigs) do
    begin
      BuildStem(cConfigs[I], NN);
      try
        CsvName := GetTempDir(False) +
          Format('erf_sweep_%d.csv', [I]);
        Report := TNNet.EffectiveReceptiveFieldReport(NN, Probes, 0.9, 64,
          CsvName);

        TheoRF := ReadTheoRF(Report);
        // Effective per-axis diameter is reported on the "EFFECTIVE RF" line:
        //   "... per-axis half-width x=<hx> y=<hy> (diameter <dx>x<dy>)."
        EffRad := ReadIntAfter(Report, 'half-width x=');
        EffDiam := 2 * EffRad + 1;

        if (TheoRF > 0) and (EffDiam > 0) then
          Ratio := EffDiam / TheoRF
        else
          Ratio := 0;

        WriteLn(Format('%-10s %12d %14d %10.2f',
          [cConfigs[I].Name, TheoRF, EffDiam, Ratio]));
      finally
        NN.Free;
      end;
    end;

    WriteLn(StringOfChar('-', 50));
    WriteLn('Per-config cumulative-mass CSVs written to ', GetTempDir(False),
      'erf_sweep_<i>.csv');
    WriteLn;
    WriteLn('Read it as: as the THEORETICAL RF roughly doubles down the table,');
    WriteLn('the EFFECTIVE diameter grows much more slowly, so the eff/theo');
    WriteLn('ratio SHRINKS — the effective RF grows SUB-LINEARLY in the');
    WriteLn('theoretical receptive field (Luo et al. 2016).');
  finally
    Probes.Free;
  end;
end.
