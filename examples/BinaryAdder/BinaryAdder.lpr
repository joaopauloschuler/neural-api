program BinaryAdder;
(*
BinaryAdder: tiny "learns to add two binary numbers" demo.

A small MLP learns to add two N-bit binary numbers (N=4) and produce an
(N+1)-bit binary sum (the extra bit accommodates carry-out). All 256
input combinations are enumerated and used for training. After training,
the program reports final loss, prints a handful of sample additions in
binary, and the exact-match accuracy over the full 256-case set.

Architecture:

  Input(8) -> FullConnectReLU(64) -> FullConnectReLU(64)
           -> FullConnect(5, Sigmoid)

The output is a 5-bit one-bit-per-channel sigmoid head, so we frame the
problem as 5 independent per-bit binary regressions trained with MSE.
Predictions are obtained by rounding each output channel at 0.5.

Designed to finish in a few seconds on a single CPU and to act as a
deterministic smoke test for the library.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume,
  neuralfit;

const
  NBits     = 4;            // bits per operand
  InSize    = 2 * NBits;    // 8 input bits
  OutSize   = NBits + 1;    // 5 output bits (includes carry)
  NumCases  = 1 shl InSize; // 256 total input combinations
  NumEpochs = 400;
  BatchSize = 16;
  LearningRate = 0.01;

  // Writes the low NumBits bits of Value into Vol (MSB at index 0).
  procedure EncodeBits(Value, NumBits: integer; Vol: TNNetVolume;
    Offset: integer);
  var
    I: integer;
  begin
    for I := 0 to NumBits - 1 do
      Vol.FData[Offset + I] := (Value shr (NumBits - 1 - I)) and 1;
  end;

  // Decodes a {0/1}-valued (or near-valued) bit vector back into an int.
  function DecodeBits(Vol: TNNetVolume; Offset, NumBits: integer): integer;
  var
    I, Bit: integer;
  begin
    Result := 0;
    for I := 0 to NumBits - 1 do
    begin
      if Vol.FData[Offset + I] >= 0.5 then Bit := 1 else Bit := 0;
      Result := (Result shl 1) or Bit;
    end;
  end;

  function BitsToStr(Vol: TNNetVolume; Offset, NumBits: integer;
    Round: boolean): string;
  var
    I, Bit: integer;
  begin
    Result := '';
    for I := 0 to NumBits - 1 do
    begin
      if Round then
      begin
        if Vol.FData[Offset + I] >= 0.5 then Bit := 1 else Bit := 0;
      end
      else
        Bit := Trunc(Vol.FData[Offset + I] + 0.0001);
      Result := Result + IntToStr(Bit);
    end;
  end;

  function CreateAdderPairList(): TNNetVolumePairList;
  var
    A, B, Sum, Idx: integer;
    Inp, Outp: TNNetVolume;
  begin
    Result := TNNetVolumePairList.Create();
    for Idx := 0 to NumCases - 1 do
    begin
      A := (Idx shr NBits) and ((1 shl NBits) - 1);
      B := Idx and ((1 shl NBits) - 1);
      Sum := A + B;
      Inp  := TNNetVolume.Create(InSize, 1, 1);
      Outp := TNNetVolume.Create(OutSize, 1, 1);
      EncodeBits(A,   NBits,   Inp,  0);
      EncodeBits(B,   NBits,   Inp,  NBits);
      EncodeBits(Sum, OutSize, Outp, 0);
      Result.Add(TNNetVolumePair.Create(Inp, Outp));
    end;
  end;

  // Sum of squared errors averaged over (samples * output_dim).
  function MeanSquaredError(NN: TNNet;
    Pairs: TNNetVolumePairList): TNeuralFloat;
  var
    I, J: integer;
    Diff, Sum: TNeuralFloat;
    Output: TNNetVolume;
  begin
    Output := TNNetVolume.Create(OutSize, 1, 1);
    Sum := 0;
    for I := 0 to Pairs.Count - 1 do
    begin
      NN.Compute(Pairs[I].I);
      NN.GetOutput(Output);
      for J := 0 to OutSize - 1 do
      begin
        Diff := Output.FData[J] - Pairs[I].O.FData[J];
        Sum := Sum + Diff * Diff;
      end;
    end;
    Output.Free;
    Result := Sum / (Pairs.Count * OutSize);
  end;

  // Fraction of inputs whose rounded prediction matches the target exactly.
  function ExactAccuracy(NN: TNNet;
    Pairs: TNNetVolumePairList): TNeuralFloat;
  var
    I: integer;
    Hits, Predicted, Target: integer;
    Output: TNNetVolume;
  begin
    Output := TNNetVolume.Create(OutSize, 1, 1);
    Hits := 0;
    for I := 0 to Pairs.Count - 1 do
    begin
      NN.Compute(Pairs[I].I);
      NN.GetOutput(Output);
      Predicted := DecodeBits(Output,    0, OutSize);
      Target    := DecodeBits(Pairs[I].O, 0, OutSize);
      if Predicted = Target then Inc(Hits);
    end;
    Output.Free;
    Result := Hits / Pairs.Count;
  end;

  procedure ShuffleIndices(var Idx: array of integer);
  var
    I, J, Tmp: integer;
  begin
    for I := High(Idx) downto 1 do
    begin
      J := Random(I + 1);
      Tmp := Idx[I];
      Idx[I] := Idx[J];
      Idx[J] := Tmp;
    end;
  end;

  procedure RunAlgo();
  var
    NN: TNNet;
    Pairs: TNNetVolumePairList;
    Epoch, Step, I, Cnt, ShowIdx: integer;
    Pair: TNNetVolumePair;
    Output: TNNetVolume;
    Order: array of integer;
    Loss, Acc: TNeuralFloat;
    A, B, Sum, Predicted: integer;
  begin
    RandSeed := 42;
    NN := TNNet.Create();
    NN.AddLayer([
      TNNetInput.Create(InSize),
      TNNetFullConnectReLU.Create(64),
      TNNetFullConnectReLU.Create(64),
      TNNetFullConnect.Create(OutSize)  // sigmoid activation by default
    ]);

    Pairs := CreateAdderPairList();
    SetLength(Order, Pairs.Count);
    for I := 0 to High(Order) do Order[I] := I;

    NN.SetLearningRate(LearningRate, {Momentum=}0.9);
    NN.SetL2Decay(0.0);

    WriteLn('BinaryAdder: learning ', NBits, '-bit + ', NBits,
      '-bit addition (', NumCases, ' cases).');
    WriteLn('Training for ', NumEpochs, ' epochs (batch=', BatchSize,
      ', lr=', LearningRate:0:4, ')...');

    for Epoch := 1 to NumEpochs do
    begin
      ShuffleIndices(Order);
      Step := 0;
      NN.ClearDeltas();
      for I := 0 to Pairs.Count - 1 do
      begin
        Pair := Pairs[Order[I]];
        NN.Compute(Pair.I);
        NN.Backpropagate(Pair.O);
        Inc(Step);
        if Step = BatchSize then
        begin
          NN.UpdateWeights();
          NN.ClearDeltas();
          Step := 0;
        end;
      end;
      if Step > 0 then
      begin
        NN.UpdateWeights();
        NN.ClearDeltas();
      end;

      if (Epoch = 1) or (Epoch mod 50 = 0) or (Epoch = NumEpochs) then
      begin
        Loss := MeanSquaredError(NN, Pairs);
        Acc  := ExactAccuracy(NN, Pairs);
        WriteLn('  Epoch ', Epoch:4, '  MSE = ', Loss:0:6,
          '  exact-acc = ', (Acc * 100):6:2, '%');
      end;
    end;

    Loss := MeanSquaredError(NN, Pairs);
    Acc  := ExactAccuracy(NN, Pairs);
    WriteLn;
    WriteLn('Final MSE         : ', Loss:0:6);
    WriteLn('Final exact-acc   : ', (Acc * 100):6:2, '% over all ',
      NumCases, ' (a,b) pairs');
    WriteLn;
    WriteLn('Sample additions (A + B = Sum  predicted: Pred):');

    Output := TNNetVolume.Create(OutSize, 1, 1);
    // Deterministic spread of 12 cases.
    for Cnt := 0 to 11 do
    begin
      ShowIdx := (Cnt * (NumCases div 12)) mod NumCases;
      A := (ShowIdx shr NBits) and ((1 shl NBits) - 1);
      B := ShowIdx and ((1 shl NBits) - 1);
      Sum := A + B;
      NN.Compute(Pairs[ShowIdx].I);
      NN.GetOutput(Output);
      Predicted := DecodeBits(Output, 0, OutSize);
      WriteLn('  ',
        BitsToStr(Pairs[ShowIdx].I, 0,     NBits,  False), ' + ',
        BitsToStr(Pairs[ShowIdx].I, NBits, NBits,  False), ' = ',
        BitsToStr(Pairs[ShowIdx].O, 0,     OutSize, False),
        '  predicted: ',
        BitsToStr(Output,           0,     OutSize, True),
        '   (', A:2, ' + ', B:2, ' = ', Sum:2,
        ', got ', Predicted:2, ')');
    end;
    Output.Free;

    Pairs.Free;
    NN.Free;
  end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'Binary Adder Example';
  RunAlgo();
end.
