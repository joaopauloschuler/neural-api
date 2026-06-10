program SlidingWindowBakeoff;
(*
SlidingWindowBakeoff: train the SAME tiny attention-based next-token model
THREE times, differing ONLY in the causal-masking scheme, and print a
side-by-side comparison of final loss vs the per-query key count each arm
must inspect.

The three arms (all causal — a query never sees the future):
  (a) W=2    - SLIDING WINDOW of width 2 (TNNetSlidingWindowMaskedFill(2)):
               each query attends only to itself and the 1 key before it.
  (b) W=4    - SLIDING WINDOW of width 4 (TNNetSlidingWindowMaskedFill(4)):
               each query attends to itself and the 3 keys before it.
  (c) FULL   - FULL causal attention (TNNetMaskedFill): each query attends
               to ALL keys at or before it (count grows with position).

Task (ANSWER LIVES INSIDE A SHORT WINDOW): a causal content-gated COPY rule
that depends ONLY on the last two tokens:

    if input[i] is EVEN -> target[i] = input[i]      (copy current token)
    if input[i] is ODD  -> target[i] = input[i-1]    (copy previous token)
    position 0           -> target[0] = input[0]      (no predecessor)

Because the rule reads at most the current token and its immediate
predecessor, the necessary context is fully contained in a width-2 window.
It is a retrieval/copy task (which single-head attention handles well),
gated by the current token's parity.
So BOTH sliding arms (W=2 and W=4) have everything they need, and the FULL
arm has the same information PLUS a lot of irrelevant far-past keys. The
teaching point: the sliding window solves the task at the SAME quality as
full causal attention while inspecting FEWER keys per query — exactly the
long-context cost saving the layer exists for.

Per-query key count (the cost knob, charted alongside loss):
  - FULL causal: query at position p attends to p+1 keys; averaged over a
    length-L sequence the mean is (L+1)/2 and the LAST query inspects L
    keys. Cost grows with sequence length.
  - SLIDING W:   query at position p attends to min(p+1, W) keys; bounded
    by W regardless of how long the sequence gets.

Shared stack (one BuildNet switches ONLY the mask layer):
  TNNetInput(SeqLen, 1, 1)                 { token IDs along X }
  -> TNNetEmbedding(Vocab, d_model)        { learned token vectors }
  -> TNNetAddPositionalEmbedding           { sin/cos position (fixed) }
  -> hand-rolled single-head CAUSAL attention (same wiring as the in-tree
     TNNet.AddSingleHeadSelfAttention helper, and as PositionEncodingBakeoff):
       Q | K | V via three TNNetSplitChannels on a packed projection
       ValueT = TransposeXD(V)
       scores = DotProducts(Q, K) / sqrt(d_k)   { (key, 1, query) }
       reshape -> (key, query, 1)
       -> MASK: TNNetSlidingWindowMaskedFill(W)  { arms W=2, W=4 }
              or TNNetMaskedFill                 { arm FULL }
       reshape -> (key, 1, query) -> ReLUL -> softmax (over depth)
       -> DotProducts(ValueT, W)                 { weighted sum of V }
  -> TNNetPointwiseConvLinear(Vocab)        { per-position logits }
  -> TNNetPointwiseSoftMax(1)               { softmax across depth }

The only difference between arms is the single mask layer, so the
comparison is apples-to-apples (shared seed, steps, learning rate, data).
Pure CPU, no dataset download; all three arms finish in well under a minute
on a single thread.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

type
  TArm = (armW2, armW4, armFull);

const
  cVocab   = 8;     // vocabulary size (token IDs 0..7)
  cSeqLen  = 12;    // sequence length
  cDModel  = 32;    // embedding dim (even, required by positional layer)
  cDk      = 32;    // attention head dim (== d_model here)
  cSteps   = 350;   // training steps per arm
  cBatch   = 32;    // sequences per step
  cLR      = 0.02;  // learning rate
  cInertia = 0.9;
  cSeed    = 2026;  // shared across arms for apples-to-apples comparison
  cValSeed = 7777;  // separate seed for a held-out validation stream

  // Window width per arm; armFull uses the full-causal TNNetMaskedFill.
  cArmWindow: array[TArm] of integer = (2, 4, cSeqLen);

  cArmName: array[TArm] of string =
    ('SLIDING W=2', 'SLIDING W=4', 'FULL causal');

  // Build the identical attention model, switching ONLY the mask layer.
  function BuildNet(Arm: TArm): TNNet;
  var
    NN: TNNet;
    EmbeddedLayer, QKV, Query, Key, ValueT: TNNetLayer;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
    // Fixed sin/cos positional info so attention can address by position
    // (identical across all three arms).
    NN.AddLayer(TNNetAddPositionalEmbedding.Create());
    EmbeddedLayer := NN.GetLastLayer;

    // --- single-head packed Q|K|V projection ---
    QKV := NN.AddLayerAfter(TNNetPointwiseConvLinear.Create(3 * cDk),
             EmbeddedLayer);
    Query := NN.AddLayerAfter(TNNetSplitChannels.Create(0, cDk), QKV);
    Key   := NN.AddLayerAfter(TNNetSplitChannels.Create(cDk, cDk), QKV);
    NN.AddLayerAfter(TNNetSplitChannels.Create(2 * cDk, cDk), QKV);
    ValueT := NN.AddLayer(TNNetTransposeXD.Create());

    // --- scores: (key, 1, query) = Q . K^T, scaled by 1/sqrt(d_k) ---
    NN.AddLayer(TNNetDotProducts.Create(Query, Key, False));
    NN.AddLayer(TNNetMulByConstant.Create(1.0 / Sqrt(cDk)));

    // Reshape to (key, query, 1): X=key, Y=query — the layout the mask
    // layers expect (mask X>Y as future; sliding also masks X<Y-W+1).
    NN.AddLayer(TNNetReshape.Create(cSeqLen, cSeqLen, 1));

    // --- THE ONLY DIFFERENCE BETWEEN ARMS: the causal mask layer ---
    case Arm of
      armW2:   NN.AddLayer(TNNetSlidingWindowMaskedFill.Create(cArmWindow[armW2]));
      armW4:   NN.AddLayer(TNNetSlidingWindowMaskedFill.Create(cArmWindow[armW4]));
      armFull: NN.AddLayer(TNNetMaskedFill.Create());
    end;

    NN.AddLayer(TNNetReshape.Create(cSeqLen, 1, cSeqLen));
    NN.AddLayer(TNNetReLUL.Create(-500, +500, 0));
    NN.AddLayer(TNNetPointwiseSoftMax.Create(0));

    NN.AddLayer(TNNetDotProducts.Create(ValueT, NN.GetLastLayer, False));

    NN.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
    NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
    NN.SetLearningRate(cLR, cInertia);
    Result := NN;
  end;

  procedure RandomTokens(var Tokens: array of integer);
  var
    I: integer;
  begin
    for I := 0 to High(Tokens) do
      Tokens[I] := Random(cVocab);
  end;

  // The in-window rule (content-gated copy from the last two tokens):
  //   if input[i] is EVEN -> copy the CURRENT token  (input[i])
  //   if input[i] is ODD  -> copy the PREVIOUS token (input[i-1])
  //   position 0 (no predecessor) always copies the current token.
  // Depends only on input[i] and input[i-1], so the answer lives fully
  // inside a width-2 window — a retrieval/copy task attention handles well,
  // unlike an arithmetic rule a single linear-value head cannot compute.
  function RuleTarget(const Tokens: array of integer; I: integer): integer;
  begin
    if (I = 0) or ((Tokens[I] and 1) = 0) then
      Result := Tokens[I]
    else
      Result := Tokens[I - 1];
  end;

  procedure MakePair(const Tokens: array of integer;
    InputV, TargetV: TNNetVolume);
  var
    I: integer;
  begin
    TargetV.Fill(0);
    for I := 0 to cSeqLen - 1 do
    begin
      InputV.FData[I] := Tokens[I];
      TargetV[I, 0, RuleTarget(Tokens, I)] := 1.0;
    end;
  end;

  function ArgMaxDepth(V: TNNetVolume; Pos: integer): integer;
  var
    D, Best: integer;
    BestVal, Cur: TNeuralFloat;
  begin
    Best := 0;
    BestVal := V[Pos, 0, 0];
    for D := 1 to cVocab - 1 do
    begin
      Cur := V[Pos, 0, D];
      if Cur > BestVal then begin BestVal := Cur; Best := D; end;
    end;
    Result := Best;
  end;

  function CrossEntropy(Output, Target: TNNetVolume): TNeuralFloat;
  var
    I: integer;
    P: TNeuralFloat;
  begin
    Result := 0;
    for I := 0 to Output.Size - 1 do
      if Target.FData[I] > 0 then
      begin
        P := Output.FData[I];
        if P < 1e-12 then P := 1e-12;
        Result := Result - Target.FData[I] * Ln(P);
      end;
    Result := Result / cSeqLen;
  end;

  // Mean per-query key count under each masking scheme over a length-L
  // sequence. FULL causal = (L+1)/2; sliding = mean of min(p+1, W).
  function MeanKeyCount(Arm: TArm): TNeuralFloat;
  var
    P, W, Cnt, Sum: integer;
  begin
    W := cArmWindow[Arm];
    Sum := 0;
    for P := 0 to cSeqLen - 1 do
    begin
      Cnt := P + 1;
      if (Arm <> armFull) and (Cnt > W) then Cnt := W;
      Sum := Sum + Cnt;
    end;
    Result := Sum / cSeqLen;
  end;

  // Key count the LAST query inspects (worst case / long-context cost).
  function LastKeyCount(Arm: TArm): integer;
  begin
    if Arm = armFull then Result := cSeqLen
    else Result := Min(cSeqLen, cArmWindow[Arm]);
  end;

  // Window label for the results table ('full' or the integer width).
  function WinLabel(Arm: TArm): string;
  begin
    if Arm = armFull then Result := 'full'
    else Result := IntToStr(cArmWindow[Arm]);
  end;

  // Evaluate mean CE + token accuracy over a fresh batch drawn from Seed.
  procedure Evaluate(NN: TNNet; Seed, NSeqs: integer;
    out CE: TNeuralFloat; out Acc: TNeuralFloat);
  var
    S, I, Correct, Total: integer;
    InputV, TargetV: TNNetVolume;
    Tokens: array[0..cSeqLen - 1] of integer;
    SumLoss: TNeuralFloat;
  begin
    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    SumLoss := 0; Correct := 0; Total := 0;
    RandSeed := Seed;
    try
      for S := 1 to NSeqs do
      begin
        RandomTokens(Tokens);
        MakePair(Tokens, InputV, TargetV);
        NN.Compute(InputV);
        SumLoss := SumLoss + CrossEntropy(NN.GetLastLayer.Output, TargetV);
        for I := 0 to cSeqLen - 1 do
        begin
          if ArgMaxDepth(NN.GetLastLayer.Output, I) = RuleTarget(Tokens, I) then
            Inc(Correct);
          Inc(Total);
        end;
      end;
      CE := SumLoss / NSeqs;
      Acc := Correct / Total;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

  function Train(NN: TNNet; const Name: string): TNeuralFloat;
  var
    Step, B: integer;
    InputV, TargetV: TNNetVolume;
    Tokens: array[0..cSeqLen - 1] of integer;
    SumLoss: TNeuralFloat;
    StartTime, Elapsed: double;
  begin
    Result := 0;
    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    RandSeed := cSeed;
    try
      WriteLn('  --- training arm: ', Name, ' ---');
      StartTime := Now();
      for Step := 1 to cSteps do
      begin
        SumLoss := 0;
        for B := 1 to cBatch do
        begin
          RandomTokens(Tokens);
          MakePair(Tokens, InputV, TargetV);
          NN.Compute(InputV);
          SumLoss := SumLoss + CrossEntropy(NN.GetLastLayer.Output, TargetV);
          NN.Backpropagate(TargetV);
        end;
        Result := SumLoss / cBatch;
        if (Step = 1) or (Step mod 50 = 0) or (Step = cSteps) then
        begin
          Elapsed := (Now() - StartTime) * 86400.0;
          WriteLn(Format('    step %4d / %4d   mean-CE=%.5f   elapsed=%.1fs',
            [Step, cSteps, Result, Elapsed]));
        end;
      end;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

var
  Arm: TArm;
  NN: TNNet;
  FinalLoss, ValLoss, ValAcc: array[TArm] of TNeuralFloat;
  GTotalStart, GElapsed: double;
  FullCE: TNeuralFloat;
  Pass: boolean;
  Slack: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  WriteLn('SlidingWindowBakeoff: same tiny CAUSAL attention model, three ',
    'masking schemes.');
  WriteLn('Mask layer differs only: TNNetSlidingWindowMaskedFill(2), (4), ',
    'and full-causal TNNetMaskedFill.');
  WriteLn('Task: target[i] = input[i] if input[i] even else input[i-1]   ',
    '(content-gated copy; answer lives in a width-2 window);');
  WriteLn('vocab=', cVocab, ', SeqLen=', cSeqLen, ', d_model=', cDModel, '.');
  WriteLn('Both sliding arms already have the needed context, so they should ',
    'match FULL while inspecting fewer keys per query.');
  WriteLn;

  GTotalStart := Now();
  for Arm := Low(TArm) to High(TArm) do
  begin
    NN := BuildNet(Arm);
    try
      if Arm = armW2 then
      begin
        WriteLn('Architecture (W=2 arm shown; other arms swap only the ',
          'mask layer):');
        NN.PrintSummary();
        WriteLn;
      end;
      FinalLoss[Arm] := Train(NN, cArmName[Arm]);
      Evaluate(NN, cValSeed, 256, ValLoss[Arm], ValAcc[Arm]);
    finally
      NN.Free;
    end;
    WriteLn;
  end;
  GElapsed := (Now() - GTotalStart) * 86400.0;

  WriteLn(StringOfChar('=', 78));
  WriteLn('RESULTS: loss vs per-query key count (the long-context cost/quality trade)');
  WriteLn(StringOfChar('=', 78));
  WriteLn(Format('  %-12s  %4s  %10s  %10s  %7s  %9s  %8s',
    ['arm', 'W', 'train-CE', 'val-CE', 'val-acc', 'mean-keys', 'last-key']));
  WriteLn('  ', StringOfChar('-', 74));
  for Arm := Low(TArm) to High(TArm) do
    WriteLn(Format('  %-12s  %4s  %10.5f  %10.5f  %6.1f%%  %9.2f  %8d',
      [cArmName[Arm],
       WinLabel(Arm),
       FinalLoss[Arm], ValLoss[Arm], 100.0 * ValAcc[Arm],
       MeanKeyCount(Arm), LastKeyCount(Arm)]));
  WriteLn;
  WriteLn('  mean-keys = average keys inspected per query over the sequence; ',
    'last-key = keys the final query sees.');
  WriteLn('  FULL grows with position (last query reads all ', cSeqLen,
    ' keys); sliding is bounded by W.');
  WriteLn;

  // ---- Grading: sliding arms must MATCH full causal (answer is in-window) ----
  FullCE := ValLoss[armFull];
  Slack := 0.05 + 0.5 * FullCE;   // absolute + relative tolerance vs FULL
  Pass := True;
  WriteLn(StringOfChar('=', 78));
  WriteLn('GRADING (sliding window must not hurt — the answer is inside the window):');
  WriteLn(StringOfChar('=', 78));
  for Arm := armW2 to armW4 do
  begin
    if ValLoss[Arm] <= FullCE + Slack then
      WriteLn(Format('  [PASS] %-12s val-CE=%.5f  within %.5f of FULL (%.5f)',
        [cArmName[Arm], ValLoss[Arm], Slack, FullCE]))
    else
    begin
      WriteLn(Format('  [FAIL] %-12s val-CE=%.5f  exceeds FULL (%.5f) by > %.5f',
        [cArmName[Arm], ValLoss[Arm], FullCE, Slack]));
      Pass := False;
    end;
  end;
  // Also require all arms to actually learn the task (well below uniform CE).
  for Arm := Low(TArm) to High(TArm) do
    if ValLoss[Arm] > 0.5 then
    begin
      WriteLn(Format('  [FAIL] %-12s did not learn the task (val-CE=%.5f > 0.5)',
        [cArmName[Arm], ValLoss[Arm]]));
      Pass := False;
    end;
  WriteLn;

  WriteLn(StringOfChar('=', 78));
  WriteLn('Total runtime for all three arms: ', Format('%.1fs', [GElapsed]));
  WriteLn('Takeaway: because the rule reads only the last two tokens, the W=2 ',
    'and W=4 sliding');
  WriteLn('windows reach the SAME loss as FULL causal attention while each ',
    'query inspects far');
  WriteLn('fewer keys (mean ', Format('%.2f / %.2f', [MeanKeyCount(armW2),
    MeanKeyCount(armFull)]), ', last ', LastKeyCount(armW2), ' / ',
    LastKeyCount(armFull), ') — the long-context cost saving the layer enables.');
  if Pass then
    WriteLn('RESULT: PASS')
  else
    WriteLn('RESULT: FAIL');
end.
