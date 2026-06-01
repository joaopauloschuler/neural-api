program SoftCappingStability;
(*
SoftCappingStability: a logit-stability micro-experiment for TNNetSoftCapping.

TNNetSoftCapping(c) maps every pre-softmax logit through
    y = c * tanh(x / c)
which squashes the logit smoothly into the open interval (-c, +c). Gemma-style
"soft capping" uses exactly this to stop classification logits from running away,
where a large logit makes exp(logit) inside the SoftMax overflow float32 and the
forward/backward pass can turn into Inf/NaN.

This program demonstrates that effect directly. It trains the SAME tiny
multi-class classifier TWICE on the SAME small synthetic 2D Gaussian-blob task,
with the SAME fixed RandSeed (424242), the same data, the same init order and
the same DELIBERATELY-AGGRESSIVE learning rate. The two arms differ in exactly
one layer:

  NoCap  arm : Input(2) -> Head(HIDDEN) -> ReLU -> Head(K) ->                    SoftMax
  Capped arm : Input(2) -> Head(HIDDEN) -> ReLU -> Head(K) -> SoftCapping(CAP) -> SoftMax

After every weight update the program inspects the logits feeding the SoftMax.
A "SoftMax-overflow event" is defined exactly: a logit whose magnitude exceeds
EXP_OVERFLOW (= 88.7, the float32 threshold above which exp(logit) is no longer
finite) -- i.e. a logit the SoftMax cannot exponentiate without overflowing -- OR
an already non-finite logit/weight (checked with IsNan / IsInfinite as a hard
backstop). Any epoch containing such an event is counted as a "blow-up epoch".
The run prints, per arm, the aggressive LR, the number of blow-up epochs, the
epoch of FIRST onset, the largest logit magnitude reached, and the final
loss / accuracy.

(The framework's SoftMax is internally hardened -- it subtracts the row max and
clamps its input range -- and the cross-entropy gradient is bounded, so the
weights themselves rarely reach a literal float32 Inf in a few dozen epochs. The
physically meaningful blow-up is therefore the logit growing past the point
where a naive exp(logit) overflows; the bounded SoftMax is precisely the band-aid
that soft-capping makes unnecessary by fixing the cause.)

The headline claim: at an LR aggressive enough to drive the uncapped head's
logits into the exp-overflow regime, inserting a single TNNetSoftCapping layer
keeps every logit bounded in (-CAP, +CAP), CAP << 88.7, so no logit can ever
overflow the SoftMax. The run ends with a self-checking PASS/FAIL gate: the
Capped arm must have STRICTLY FEWER blow-up epochs than the NoCap arm, the NoCap
arm must actually blow up, and the Capped arm must stay completely clean and
remain a useful classifier.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  RAND_SEED    = 424242;
  NUM_CLASSES  = 4;
  HIDDEN_UNITS = 24;
  TRAIN_SIZE   = 1000;
  TEST_SIZE    = 400;
  NUM_EPOCHS   = 60;
  BATCH_SIZE   = 25;
  // Deliberately-too-high learning rate (50x a sane 0.1), chosen to drive the
  // uncapped head's logits into the exp-overflow regime while still leaving the
  // capped head trainable.
  AGGRESSIVE_LR = 5.0;
  MOMENTUM      = 0.9;
  // SoftCapping cap constant c: logits are squashed into (-CAP, +CAP). CAP is
  // chosen well below EXP_OVERFLOW so a capped logit can never overflow exp().
  CAP           = 8.0;
  // float32 exp() overflows above ~88.72 (ln(MaxSingle)); a pre-softmax logit
  // past this point makes exp(logit) non-finite -> a SoftMax-overflow event.
  EXP_OVERFLOW  = 88.7;

type
  TArmResult = record
    Name        : string;
    UseCap      : boolean;
    LR          : TNeuralFloat;
    BlowUps     : integer;     // # epochs containing a SoftMax-overflow event
    FirstOnset  : integer;     // epoch index of first blow-up (0 = never)
    MaxLogit    : TNeuralFloat;// largest |logit| seen during training
    Loss        : TNeuralFloat;// final mean cross-entropy on the test set
    Acc         : TNeuralFloat;// final argmax accuracy on the test set
  end;

// Build the SAME small synthetic 4-class 2D Gaussian-blob task. Each class is a
// Gaussian blob around one of the four corners; the hidden layer is needed
// because the four-corner layout is not linearly separable into one class.
function CreateBlobPairList(MaxCnt: integer): TNNetVolumePairList;
var
  Cnt, Cls: integer;
  Centers: array[0..NUM_CLASSES-1, 0..1] of TNeuralFloat;
  Px, Py: TNeuralFloat;
  Inp, Outp: TNNetVolume;
begin
  Centers[0][0] := -1.0; Centers[0][1] := -1.0;
  Centers[1][0] :=  1.0; Centers[1][1] := -1.0;
  Centers[2][0] := -1.0; Centers[2][1] :=  1.0;
  Centers[3][0] :=  1.0; Centers[3][1] :=  1.0;

  Result := TNNetVolumePairList.Create();
  for Cnt := 0 to MaxCnt - 1 do
  begin
    Cls := Cnt mod NUM_CLASSES;
    Px := Centers[Cls][0] + 0.45 * ((Random + Random) - 1.0);
    Py := Centers[Cls][1] + 0.45 * ((Random + Random) - 1.0);

    Inp := TNNetVolume.Create([Px, Py]);
    Outp := TNNetVolume.Create(NUM_CLASSES);
    Outp.SetClassForSoftMax(Cls);
    Result.Add(TNNetVolumePair.Create(Inp, Outp));
  end;
end;

// True when V is non-finite (NaN or +/- Inf).
function NonFinite(V: TNeuralFloat): boolean;
begin
  Result := IsNan(V) or IsInfinite(V);
end;

// Largest absolute element of a volume (the worst-case logit magnitude).
function MaxAbsElem(V: TNNetVolume): TNeuralFloat;
var
  I: integer;
  A: TNeuralFloat;
begin
  Result := 0;
  for I := 0 to V.Size - 1 do
  begin
    A := Abs(V.FData[I]);
    if A > Result then Result := A;
  end;
end;

// Mean cross-entropy + argmax accuracy of NN over a pair list. A non-finite
// SoftMax probability contributes a large-but-finite penalty so the reported
// loss stays printable even after a blow-up.
procedure EvalArm(NN: TNNet; Pairs: TNNetVolumePairList;
  out Loss, Acc: TNeuralFloat);
var
  I, Hits: integer;
  P, SumLoss: TNeuralFloat;
  Outp: TNNetVolume;
begin
  Hits := 0;
  SumLoss := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    Outp := NN.GetLastLayer().Output;
    if Outp.GetClass() = Pairs[I].O.GetClass() then Inc(Hits);
    P := Outp.FData[Pairs[I].O.GetClass()];
    if NonFinite(P) or (P <= 0) then
      SumLoss := SumLoss + 30.0          // saturated penalty for a dead output
    else
      SumLoss := SumLoss - Ln(Max(P, 1e-12));
  end;
  if Pairs.Count > 0 then
  begin
    Loss := SumLoss / Pairs.Count;
    Acc  := Hits / Pairs.Count;
  end
  else begin Loss := 0; Acc := 0; end;
end;

// Build the fixed classifier. The only difference between arms is whether a
// TNNetSoftCapping layer sits between the final linear head and the SoftMax.
// We keep a handle on the layer feeding the SoftMax (the logit layer) so the
// training loop can inspect its activations for overflow.
procedure BuildNet(out NN: TNNet; UseCap: boolean; out LogitLayer: TNNetLayer);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(2));
  NN.AddLayer(TNNetFullConnectLinear.Create(HIDDEN_UNITS));
  NN.AddLayer(TNNetReLU.Create());
  NN.AddLayer(TNNetFullConnectLinear.Create(NUM_CLASSES));
  if UseCap then
    LogitLayer := NN.AddLayer(TNNetSoftCapping.Create(CAP))
  else
    LogitLayer := NN.GetLastLayer();   // the FullConnectLinear head
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(AGGRESSIVE_LR, MOMENTUM);
  NN.SetL2Decay(0.0);                  // no implicit logit regularisation
  NN.SetBatchUpdate(True);
end;

// Train one arm with manual mini-batch SGD so we can probe the logits after
// every weight update. Returns the blow-up counters and final test metrics.
function RunArm(const Name: string; UseCap: boolean): TArmResult;
var
  NN: TNNet;
  LogitLayer: TNNetLayer;
  HeadLayer: TNNetLayer;
  Train, Test: TNNetVolumePairList;
  Epoch, I, J, B, Tmp, InBatch: integer;
  Order: array of integer;
  EpochBad: boolean;
  LogitMag, WeightMag: TNeuralFloat;
begin
  Result.Name       := Name;
  Result.UseCap     := UseCap;
  Result.LR         := AGGRESSIVE_LR;
  Result.BlowUps    := 0;
  Result.FirstOnset := 0;
  Result.MaxLogit   := 0;

  // Identical data for both arms: reseed before building.
  RandSeed := RAND_SEED;
  Train := CreateBlobPairList(TRAIN_SIZE);
  Test  := CreateBlobPairList(TEST_SIZE);
  // Reseed again so the weight init draws are identical across arms.
  RandSeed := RAND_SEED + 7;
  BuildNet(NN, UseCap, LogitLayer);
  // The trainable head feeding the logits, used as a representative weight
  // probe (it is the same FullConnectLinear layer in both arms).
  HeadLayer := NN.Layers[3];

  SetLength(Order, Train.Count);
  for I := 0 to High(Order) do Order[I] := I;
  try
    for Epoch := 1 to NUM_EPOCHS do
    begin
      // Fisher-Yates shuffle of the presentation order.
      for I := High(Order) downto 1 do
      begin
        J := Random(I + 1);
        Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
      end;

      EpochBad := False;
      InBatch := 0;
      NN.ClearDeltas();
      for B := 0 to High(Order) do
      begin
        NN.Compute(Train[Order[B]].I);
        NN.Backpropagate(Train[Order[B]].O);
        Inc(InBatch);
        if (InBatch >= BATCH_SIZE) or (B = High(Order)) then
        begin
          NN.UpdateWeights();
          NN.ClearDeltas();
          InBatch := 0;
          // Probe for instability right after the update: the worst-case logit
          // magnitude feeding the SoftMax, plus a representative head weight.
          LogitMag  := MaxAbsElem(LogitLayer.Output);
          WeightMag := HeadLayer.Neurons[0].Weights.GetSumAbs();
          if (not NonFinite(LogitMag)) and (LogitMag > Result.MaxLogit) then
            Result.MaxLogit := LogitMag;
          // SoftMax-overflow event: a logit exp() can no longer exponentiate
          // finitely, or an already non-finite logit/weight (hard backstop).
          if (LogitMag > EXP_OVERFLOW)
             or NonFinite(LogitMag) or NonFinite(WeightMag) then
            EpochBad := True;
        end;
      end;

      if EpochBad then
      begin
        Inc(Result.BlowUps);
        if Result.FirstOnset = 0 then Result.FirstOnset := Epoch;
      end;
    end;

    EvalArm(NN, Test, Result.Loss, Result.Acc);
  finally
    NN.Free;
    Train.Free;
    Test.Free;
  end;
end;

procedure RunExperiment();
var
  NoCap, Capped: TArmResult;
  StartTime, EndTime: TDateTime;
  PassFewer, PassNoCapBlewUp, PassCappedClean, PassCappedUseful, PassAll: boolean;
const
  // The capped arm must still classify the blobs well (the four corners are
  // trivially separable for a finite network, so this bar is generous).
  MIN_CAPPED_ACC = 0.85;

  function OnsetStr(R: TArmResult): string;
  begin
    if R.FirstOnset = 0 then Result := '   --'
    else Result := Format('%5d', [R.FirstOnset]);
  end;

begin
  WriteLn('SoftCapping logit-stability micro-experiment.');
  WriteLn('Task: ', NUM_CLASSES, '-class 2D Gaussian-blob classification.');
  WriteLn('NoCap  net: Input(2) -> Head(', HIDDEN_UNITS, ') -> ReLU -> Head(',
          NUM_CLASSES, ') ->                  SoftMax');
  WriteLn('Capped net: Input(2) -> Head(', HIDDEN_UNITS, ') -> ReLU -> Head(',
          NUM_CLASSES, ') -> SoftCapping(', CAP:0:1, ') -> SoftMax');
  WriteLn('Same net/seed/data/epochs; only the SoftCapping layer differs.');
  WriteLn(Format('%d epochs, %d train / %d test, batch=%d, AGGRESSIVE LR=%.2f, momentum=%.2f.',
          [NUM_EPOCHS, TRAIN_SIZE, TEST_SIZE, BATCH_SIZE, AGGRESSIVE_LR, MOMENTUM]));
  WriteLn(Format('A blow-up = a logit with |logit| > %.1f (exp() overflows float32) '
    + 'or a non-finite logit/weight.', [EXP_OVERFLOW]));
  WriteLn;

  StartTime := Now;
  Write('Training NoCap  arm (no SoftCapping)  ... ');
  NoCap := RunArm('NoCap', False);
  WriteLn('done.');
  Write('Training Capped arm (SoftCapping(', CAP:0:1, ')) ... ');
  Capped := RunArm('Capped', True);
  WriteLn('done.');
  EndTime := Now;
  WriteLn;

  WriteLn('=== Numerical-stability summary ===');
  WriteLn('arm     LR     blow-up epochs  first onset   max |logit|   final loss   final acc');
  WriteLn(Format('%-6s  %5.2f  %8d %-6s  %11s   %11.2f   %10.4f   %7.2f%%',
    [NoCap.Name, NoCap.LR, NoCap.BlowUps,
     Format('/%d', [NUM_EPOCHS]), OnsetStr(NoCap),
     NoCap.MaxLogit, NoCap.Loss, NoCap.Acc * 100]));
  WriteLn(Format('%-6s  %5.2f  %8d %-6s  %11s   %11.2f   %10.4f   %7.2f%%',
    [Capped.Name, Capped.LR, Capped.BlowUps,
     Format('/%d', [NUM_EPOCHS]), OnsetStr(Capped),
     Capped.MaxLogit, Capped.Loss, Capped.Acc * 100]));
  WriteLn;
  WriteLn(Format('Blow-up epochs: NoCap=%d  Capped=%d  (lower is better).',
    [NoCap.BlowUps, Capped.BlowUps]));
  WriteLn(Format('Largest |logit| reached: NoCap=%.2f  Capped=%.2f  (cap=%.1f, exp-overflow=%.1f).',
    [NoCap.MaxLogit, Capped.MaxLogit, CAP, EXP_OVERFLOW]));
  WriteLn;

  WriteLn('=== Stability gate ===');
  PassNoCapBlewUp := NoCap.BlowUps > 0;
  WriteLn(Format('[%s] NoCap arm blew up: %d blow-up epoch(s) (must be > 0).',
    [BoolToStr(PassNoCapBlewUp, 'PASS', 'FAIL'), NoCap.BlowUps]));
  PassFewer := Capped.BlowUps < NoCap.BlowUps;
  WriteLn(Format('[%s] Capped arm has STRICTLY fewer blow-up epochs: %d < %d.',
    [BoolToStr(PassFewer, 'PASS', 'FAIL'), Capped.BlowUps, NoCap.BlowUps]));
  PassCappedClean := Capped.BlowUps = 0;
  WriteLn(Format('[%s] Capped arm stayed completely clean: %d blow-up epoch(s).',
    [BoolToStr(PassCappedClean, 'PASS', 'FAIL'), Capped.BlowUps]));
  PassCappedUseful := Capped.Acc >= MIN_CAPPED_ACC;
  WriteLn(Format('[%s] Capped arm still useful: test acc %.2f%% (must be >= %.0f%%).',
    [BoolToStr(PassCappedUseful, 'PASS', 'FAIL'), Capped.Acc * 100, MIN_CAPPED_ACC * 100]));
  WriteLn;
  WriteLn('TAKEAWAY: at a learning rate aggressive enough to drive the uncapped head''s');
  WriteLn('logits past the exp() overflow point, a single TNNetSoftCapping(c) layer');
  WriteLn('bounds every logit to (-c, c) so no SoftMax-overflow event can occur -- and');
  WriteLn('the classifier still trains.');
  WriteLn;

  PassAll := PassNoCapBlewUp and PassFewer and PassCappedClean and PassCappedUseful;
  if PassAll then
    WriteLn('GATE: PASS -- SoftCapping tamed the logit blow-up.')
  else
    WriteLn('GATE: FAIL -- stability claim NOT met on this run.');

  WriteLn;
  WriteLn('Total wall time: ', FormatFloat('0.00', (EndTime - StartTime) * 86400), ' s');

  if not PassAll then Halt(1);
end;

begin
  RandSeed := RAND_SEED;
  RunExperiment();
end.
