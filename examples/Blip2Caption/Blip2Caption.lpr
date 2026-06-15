program Blip2Caption;
(*
Blip2Caption: the BLIP-2 Q-Former vision-language BRIDGE on the CPU, end to end
with the repo's importer (BuildBlip2FromSafeTensors / BuildBlip2QFormerFrom-
SafeTensors, neuralpretrained.pas).

BLIP-2 (Li et al. 2023, "BLIP-2: Bootstrapping Language-Image Pre-training with
Frozen Image Encoders and Large Language Models", arXiv:2301.12597; e.g.
Salesforce/blip2-flan-t5-xl) bridges a FROZEN vision encoder to a FROZEN LLM
with a small QUERYING TRANSFORMER (the Q-Former): a fixed set of LEARNED query
tokens (query_tokens, e.g. 32) is fed through a few BERT-style blocks that, in
each block, SELF-attend among themselves AND CROSS-attend into the frozen ViT
patch features, distilling the image into 32 query embeddings. A
language_projection linear then maps those query embeddings into the LLM token
space, where they are spliced ahead of the prompt and the LLM (FLAN-T5) decodes
a caption.

The genuinely NEW piece this demo exercises is the Q-Former: the interleaved
self/cross-attention querying transformer (the soft-prompt query tokens + the
two-source cross-attention into the ViT features + the BERT post-LN FFN). The
vision tower (BuildClipVisionTower) and the FLAN-T5 decode tail
(BuildT5FromSafeTensors, the T5EncoderStatesInput two-net convention) are
REUSE -- documented here, not built by this v1 demo.

NO NETWORK ACCESS / SELF-CONTAINED: the real Salesforce/blip2-* checkpoints are
large and not obtainable offline, so -- exactly like the repo's CLIPSeg /
LLaVA pico fixtures -- this falls back to the committed CONFIG-FAITHFUL random
pico BLIP-2 (tests/fixtures/tiny_blip2_full.*, built by
tools/blip2_qformer_tiny_fixture.py from the REAL HF Blip2QFormerModel float64
oracle, parity-checked < 1e-4 in tests/TestNeuralPretrained.pas
TestBlip2QFormerParity / TestBlip2FullBridgeParity). The pico net is random
(not trained), so this is a wiring/throughput SMOKE: it builds the Q-Former +
projection nets, feeds the learned query_tokens and a deterministic synthetic
set of ViT patch features, runs the bridge and prints the (NumQuery x
TextHidden) projected query embeddings that would be spliced into the LLM.

For a real caption: download a real blip2-flan-t5 checkpoint, run its EVA/CLIP
ViT (BuildClipVisionTower) on the preprocessed image to get the patch features,
feed them as the Q-Former's second input, project, then feed the projected
query embeddings to a BuildT5FromSafeTensors decoder via T5EncoderStatesInput.

USAGE
  ./Blip2Caption                 use the committed pico fixture
  ./Blip2Caption model.safetensors [config.json]
                                 use a real full-blip2 checkpoint

Coded by Claude (AI).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)
{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math,
  neuralnetwork, neuralvolume, neuralpretrained;

const
  cFixtureST  = '../../tests/fixtures/tiny_blip2_full.safetensors';
  cFixtureCfg = '../../tests/fixtures/tiny_blip2_full_config.json';

var
  STFile, CfgFile: string;
  QFormerNet, ProjectionNet: TNNet;
  QueryTokens, VisFeatures, QfOut, ProjOut: TNNetVolume;
  Config: TBlip2QFormerConfig;
  VisInput: TNNetLayer;
  NumPatches, i, j: integer;
begin
  if ParamCount >= 1 then STFile := ParamStr(1) else STFile := cFixtureST;
  if ParamCount >= 2 then CfgFile := ParamStr(2)
  else if ParamCount >= 1 then CfgFile := ''  // sibling config.json
  else CfgFile := cFixtureCfg;

  if not FileExists(STFile) then
  begin
    WriteLn('Checkpoint not found: ', STFile);
    WriteLn('Run from examples/Blip2Caption/ so the pico fixture resolves, ',
      'or pass a real .safetensors path.');
    Halt(1);
  end;

  // A deterministic synthetic "ViT patch grid": NumPatches feature vectors.
  NumPatches := 5;

  WriteLn('=== BLIP-2 Q-Former bridge (pico smoke) ===');
  WriteLn('Checkpoint : ', STFile);

  BuildBlip2FromSafeTensors(STFile, QFormerNet, ProjectionNet, QueryTokens,
    Config, NumPatches, {pInferenceOnly=}true, CfgFile);

  VisFeatures := TNNetVolume.Create;
  QfOut := TNNetVolume.Create;
  ProjOut := TNNetVolume.Create;
  try
    WriteLn('Config     : ', Blip2QFormerConfigToString(Config));
    WriteLn('Query tokens: ', Config.NumQueryTokens, ' x ', Config.HiddenSize);
    WriteLn('ViT patches : ', NumPatches, ' x ', Config.EncoderHiddenSize);

    // Synthetic frozen ViT features (a real demo runs BuildClipVisionTower
    // on the preprocessed image instead). Fill the Q-Former's SECOND input.
    VisInput := Blip2QFormerVisionInput(QFormerNet);
    VisFeatures.ReSize(NumPatches, 1, Config.EncoderHiddenSize);
    for i := 0 to NumPatches - 1 do
      for j := 0 to Config.EncoderHiddenSize - 1 do
        VisFeatures[i, 0, j] := Sin(0.3 * i + 0.1 * j) * 0.7;
    VisInput.Output.Copy(VisFeatures);

    // Q-Former: distil the patch grid into NumQuery query embeddings, then
    // project into the LLM token space.
    QFormerNet.Compute(QueryTokens);
    QFormerNet.GetOutput(QfOut);
    ProjectionNet.Compute(QfOut);
    ProjectionNet.GetOutput(ProjOut);

    WriteLn;
    WriteLn('Projected query embeddings (', ProjOut.SizeX, ' queries x ',
      ProjOut.Depth, ' LLM dims) - these splice into the FLAN-T5 input:');
    for i := 0 to ProjOut.SizeX - 1 do
    begin
      Write('  q', i, ':');
      for j := 0 to ProjOut.Depth - 1 do
        Write(' ', FormatFloat('0.000', ProjOut[i, 0, j]));
      WriteLn;
    end;
    WriteLn;
    WriteLn('(pico weights are random -> these are a wiring smoke, not a ',
      'trained caption; see the header for the real-checkpoint recipe.)');
  finally
    ProjOut.Free;
    QfOut.Free;
    VisFeatures.Free;
    QueryTokens.Free;
    ProjectionNet.Free;
    QFormerNet.Free;
  end;
end.
