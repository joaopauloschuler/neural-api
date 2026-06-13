program LlamaLogitsDump;
(*
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

// LlamaLogitsDump -- loads a Llama-architecture safetensors checkpoint with
// the CAI importer (neural/neuralpretrained.pas), runs one forward pass on
// the given token ids and dumps the logits rows of the real (non-padded)
// positions as JSON on stdout:
//   {"tokens": [t0, ...], "logits": [[row0...], [row1...], ...]}
// Row i is the full-vocab logits at position i (it depends only on tokens
// 0..i thanks to the causal mask, so trailing padding does not affect it).
// Companion of compare_hf_logits.py: the HF side runs the same tokens
// through transformers' LlamaForCausalLM and diffs every value.
//
// Usage:
//   LlamaLogitsDump <model.safetensors> <SeqLen> t0 [t1 ...]
// (SeqLen 0 = the full max_position_embeddings context - SLOW and
// memory-hungry on real checkpoints, pass something small like 16.
// config.json is read from the directory of the safetensors file.)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralnetwork,
  neuralsafetensors, neuralpretrained;

var
  NN: TNNet;
  Config: TLlamaConfig;
  Input, Output: TNNetVolume;
  Prompt: array of integer;
  SeqLen, ParamCnt, TokenCnt, PosCnt, TokCnt: integer;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.'; // JSON requires a dot whatever the locale
  if ParamCount < 3 then
  begin
    WriteLn(StdErr, 'Usage: LlamaLogitsDump <model.safetensors> <SeqLen> t0 [t1 ...]');
    Halt(1);
  end;
  SeqLen := StrToIntDef(ParamStr(2), 0);
  SetLength(Prompt, ParamCount - 2);
  for ParamCnt := 3 to ParamCount do
    Prompt[ParamCnt - 3] := StrToIntDef(ParamStr(ParamCnt), -1);

  // Dump-only tool: free training volumes during construction (~1/3 memory).
  NN := BuildLlamaFromSafeTensorsEx(ParamStr(1), Config, SeqLen,
    {pInferenceOnly=}true);
  try
    WriteLn(StdErr, LlamaConfigToString(Config));
    if SeqLen <= 0 then SeqLen := Config.MaxPositions;
    if Length(Prompt) > SeqLen then
    begin
      WriteLn(StdErr, 'Prompt is longer than the context window.');
      Halt(1);
    end;
    for TokenCnt := 0 to High(Prompt) do
      if (Prompt[TokenCnt] < 0) or (Prompt[TokenCnt] >= Config.VocabSize) then
      begin
        WriteLn(StdErr, 'Token ', Prompt[TokenCnt], ' is outside the vocabulary.');
        Halt(1);
      end;
    Input := TNNetVolume.Create(SeqLen, 1, 1);
    Output := TNNetVolume.Create;
    try
      Input.Fill(0);
      for TokenCnt := 0 to High(Prompt) do
        Input.FData[TokenCnt] := Prompt[TokenCnt];
      NN.Compute(Input);
      NN.GetOutput(Output);

      Write('{"tokens": [');
      for TokenCnt := 0 to High(Prompt) do
      begin
        if TokenCnt > 0 then Write(', ');
        Write(Prompt[TokenCnt]);
      end;
      WriteLn('],');
      WriteLn('"logits": [');
      for PosCnt := 0 to High(Prompt) do
      begin
        Write('[');
        for TokCnt := 0 to Config.VocabSize - 1 do
        begin
          if TokCnt > 0 then Write(', ');
          // 9 significant digits round-trip a single exactly.
          Write(FloatToStrF(Output.FData[PosCnt * Config.VocabSize + TokCnt],
            ffExponent, 9, 2, FS));
        end;
        if PosCnt < High(Prompt) then WriteLn('],') else WriteLn(']');
      end;
      WriteLn(']}');
    finally
      Output.Free;
      Input.Free;
    end;
  finally
    NN.Free;
  end;
end.
