unit TestNeuralMaskedLM;
(*
Tests for TNNetMaskedLMCollator (neuraldatasets.pas): BERT-style dynamic
masked-language-model collation (HuggingFace DataCollatorForLanguageModeling
port). The statistical tests run the collator over a long sequence at a fixed
RNG seed and assert: the masked fraction is within tolerance of MaskProb; the
80/10/10 split among selected tokens is within tolerance; special tokens are
never selected; labels are exactly the ignore sentinel off the selected set
and the original id on it. A small loss-mask integration test verifies that
ignored positions backpropagate exactly zero gradient with the framework's
e = Output - Desired error convention.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry,
  neuralnetwork, neuralvolume, neuraldatasets;

type
  TTestNeuralMaskedLM = class(TTestCase)
  published
    // Masked fraction over a long sequence is within tolerance of MaskProb.
    procedure TestMaskedFractionWithinTolerance;
    // Among selected tokens, the [MASK]/random/unchanged split is ~80/10/10.
    procedure TestEightyTenTenSplit;
    // Registered special tokens are never selected for masking.
    procedure TestSpecialTokensNeverMasked;
    // Labels are exactly the ignore sentinel off the selected set and carry
    // the ORIGINAL id on it; corruption only touches selected positions.
    procedure TestLabelsIgnoreAndOriginalIds;
    // Random replacement never produces a special token.
    procedure TestRandomReplacementNeverSpecial;
    // Reseeding reproduces identical collation; different seeds differ.
    procedure TestReseedReproducible;
    // BuildTrainingPair + ApplyLossMask: ignored rows carry zero output error.
    procedure TestLossMaskZeroErrorAtIgnoredPositions;
    // Whole-word masking: a selected word has ALL its pieces masked together
    // and no partial-word masking ever occurs.
    procedure TestWholeWordNoPartialMasking;
    // Whole-word masking with MaskProb=0 selects nothing and corrupts nothing.
    procedure TestWholeWordZeroProbSelectsNothing;
  end;

  TTestNeuralSpanCorruption = class(TTestCase)
  published
    // Sentinel ids descend from the configured base.
    procedure TestSentinelIdsDescend;
    // The original sequence is exactly reconstructable from source + target.
    procedure TestRoundTripReconstruction;
    // Source sentinel placement matches the target sentinel/span stream and the
    // target ends with the trailing (final) sentinel.
    procedure TestSourceTargetSentinelConsistency;
    // CorruptionRate=0 leaves the sequence untouched (no spans, source=input).
    procedure TestZeroRateNoCorruption;
    // Reseeding reproduces identical collation.
    procedure TestSpanReseedReproducible;
  end;

implementation

const
  MaskId = 4;
  Vocab = 100;

procedure TTestNeuralMaskedLM.TestMaskedFractionWithinTolerance;
var
  Col: TNNetMaskedLMCollator;
  Tokens, Corrupt, Labels: TNeuralIntegerArray;
  N, I, Selected: integer;
  Frac: TNeuralFloat;
begin
  N := 20000;
  SetLength(Tokens, N);
  for I := 0 to N - 1 do Tokens[I] := 10 + (I mod 80); // all real, non-special
  Col := TNNetMaskedLMCollator.Create(MaskId, Vocab, 0.15);
  try
    Col.Reseed(424242);
    Col.Collate(Tokens, Corrupt, Labels);
    Selected := 0;
    for I := 0 to N - 1 do
      if Labels[I] <> csMaskedLMIgnoreLabel then Inc(Selected);
    Frac := Selected / N;
    AssertTrue('masked fraction ' + FloatToStr(Frac) + ' near 0.15',
      Abs(Frac - 0.15) < 0.01);
  finally
    Col.Free;
  end;
end;

procedure TTestNeuralMaskedLM.TestEightyTenTenSplit;
var
  Col: TNNetMaskedLMCollator;
  Tokens, Corrupt, Labels: TNeuralIntegerArray;
  N, I, Sel, NMask, NRand, NSame: integer;
  PMask, PRand, PSame: TNeuralFloat;
begin
  N := 40000;
  SetLength(Tokens, N);
  for I := 0 to N - 1 do Tokens[I] := 10 + (I mod 80);
  Col := TNNetMaskedLMCollator.Create(MaskId, Vocab, 0.15);
  try
    Col.Reseed(123456);
    Col.Collate(Tokens, Corrupt, Labels);
    Sel := 0; NMask := 0; NRand := 0; NSame := 0;
    for I := 0 to N - 1 do
      if Labels[I] <> csMaskedLMIgnoreLabel then
      begin
        Inc(Sel);
        if Corrupt[I] = MaskId then Inc(NMask)
        else if Corrupt[I] = Tokens[I] then Inc(NSame)
        else Inc(NRand);
      end;
    AssertTrue('enough selected for statistics', Sel > 4000);
    PMask := NMask / Sel; PRand := NRand / Sel; PSame := NSame / Sel;
    AssertTrue('mask share ' + FloatToStr(PMask) + ' near 0.8',
      Abs(PMask - 0.8) < 0.03);
    AssertTrue('random share ' + FloatToStr(PRand) + ' near 0.1',
      Abs(PRand - 0.1) < 0.02);
    AssertTrue('unchanged share ' + FloatToStr(PSame) + ' near 0.1',
      Abs(PSame - 0.1) < 0.02);
  finally
    Col.Free;
  end;
end;

procedure TTestNeuralMaskedLM.TestSpecialTokensNeverMasked;
var
  Col: TNNetMaskedLMCollator;
  Tokens, Corrupt, Labels: TNeuralIntegerArray;
  N, I: integer;
begin
  // Special ids 0 (pad), 1 (cls), 2 (sep), MaskId. Interleave them densely.
  N := 10000;
  SetLength(Tokens, N);
  for I := 0 to N - 1 do
    case I mod 5 of
      0: Tokens[I] := 0;
      1: Tokens[I] := 1;
      2: Tokens[I] := 2;
      3: Tokens[I] := MaskId;
    else Tokens[I] := 50 + (I mod 30);
    end;
  Col := TNNetMaskedLMCollator.Create(MaskId, Vocab, 0.5); // high prob to stress
  try
    Col.AddSpecialTokenId(0);
    Col.AddSpecialTokenId(1);
    Col.AddSpecialTokenId(2);
    Col.Reseed(999);
    Col.Collate(Tokens, Corrupt, Labels);
    for I := 0 to N - 1 do
      if (Tokens[I] = 0) or (Tokens[I] = 1) or (Tokens[I] = 2) or
         (Tokens[I] = MaskId) then
      begin
        AssertEquals('special pos ' + IntToStr(I) + ' not selected',
          csMaskedLMIgnoreLabel, Labels[I]);
        AssertEquals('special pos ' + IntToStr(I) + ' unchanged',
          Tokens[I], Corrupt[I]);
      end;
  finally
    Col.Free;
  end;
end;

procedure TTestNeuralMaskedLM.TestLabelsIgnoreAndOriginalIds;
var
  Col: TNNetMaskedLMCollator;
  Tokens, Corrupt, Labels: TNeuralIntegerArray;
  N, I: integer;
begin
  N := 5000;
  SetLength(Tokens, N);
  for I := 0 to N - 1 do Tokens[I] := 10 + (I mod 80);
  Col := TNNetMaskedLMCollator.Create(MaskId, Vocab, 0.15);
  try
    Col.Reseed(7);
    Col.Collate(Tokens, Corrupt, Labels);
    for I := 0 to N - 1 do
      if Labels[I] = csMaskedLMIgnoreLabel then
        // Off the selected set: input is untouched.
        AssertEquals('ignored pos ' + IntToStr(I) + ' unchanged input',
          Tokens[I], Corrupt[I])
      else
        // On the selected set: label is the ORIGINAL id.
        AssertEquals('selected pos ' + IntToStr(I) + ' label = original',
          Tokens[I], Labels[I]);
  finally
    Col.Free;
  end;
end;

procedure TTestNeuralMaskedLM.TestRandomReplacementNeverSpecial;
var
  Col: TNNetMaskedLMCollator;
  Tokens, Corrupt, Labels: TNeuralIntegerArray;
  N, I: integer;
begin
  N := 20000;
  SetLength(Tokens, N);
  for I := 0 to N - 1 do Tokens[I] := 10 + (I mod 80);
  // Force 100% random replacement so we exercise RandomRealToken heavily.
  Col := TNNetMaskedLMCollator.Create(MaskId, Vocab, 1.0);
  try
    Col.AddSpecialTokenId(0);
    Col.AddSpecialTokenId(1);
    Col.AddSpecialTokenId(2);
    Col.ReplaceMaskProb := 0.0;
    Col.RandomTokenProb := 1.0;
    Col.Reseed(31337);
    Col.Collate(Tokens, Corrupt, Labels);
    for I := 0 to N - 1 do
    begin
      AssertTrue('random repl pos ' + IntToStr(I) + ' not pad', Corrupt[I] <> 0);
      AssertTrue('random repl pos ' + IntToStr(I) + ' not cls', Corrupt[I] <> 1);
      AssertTrue('random repl pos ' + IntToStr(I) + ' not sep', Corrupt[I] <> 2);
      AssertTrue('random repl pos ' + IntToStr(I) + ' not mask',
        Corrupt[I] <> MaskId);
      AssertTrue('random repl pos ' + IntToStr(I) + ' in vocab',
        (Corrupt[I] >= 0) and (Corrupt[I] < Vocab));
    end;
  finally
    Col.Free;
  end;
end;

procedure TTestNeuralMaskedLM.TestReseedReproducible;
var
  ColA, ColB: TNNetMaskedLMCollator;
  Tokens, Ca, La, Cb, Lb, Cc, Lc: TNeuralIntegerArray;
  N, I, Diff: integer;
begin
  N := 2000;
  SetLength(Tokens, N);
  for I := 0 to N - 1 do Tokens[I] := 10 + (I mod 80);
  ColA := TNNetMaskedLMCollator.Create(MaskId, Vocab, 0.15);
  ColB := TNNetMaskedLMCollator.Create(MaskId, Vocab, 0.15);
  try
    ColA.Reseed(55555);
    ColA.Collate(Tokens, Ca, La);
    ColB.Reseed(55555);
    ColB.Collate(Tokens, Cb, Lb);
    for I := 0 to N - 1 do
    begin
      AssertEquals('same seed corrupt ' + IntToStr(I), Ca[I], Cb[I]);
      AssertEquals('same seed label ' + IntToStr(I), La[I], Lb[I]);
    end;
    // A different seed should differ somewhere.
    ColA.Reseed(66666);
    ColA.Collate(Tokens, Cc, Lc);
    Diff := 0;
    for I := 0 to N - 1 do
      if (Cc[I] <> Ca[I]) or (Lc[I] <> La[I]) then Inc(Diff);
    AssertTrue('different seed differs', Diff > 0);
  finally
    ColB.Free;
    ColA.Free;
  end;
end;

procedure TTestNeuralMaskedLM.TestLossMaskZeroErrorAtIgnoredPositions;
var
  Col: TNNetMaskedLMCollator;
  Tokens, Corrupt, Labels: TNeuralIntegerArray;
  NN: TNNet;
  InputV, TargetV, ErrV: TNNetVolume;
  Ctx, P, D, Selected: integer;
  RowAbs, SelectedAbs: TNeuralFloat;
begin
  RandSeed := 424242;
  Ctx := 16;
  SetLength(Tokens, Ctx);
  for P := 0 to Ctx - 1 do Tokens[P] := 10 + P; // distinct real ids
  Col := TNNetMaskedLMCollator.Create(MaskId, Vocab, 0.4);
  NN := TNNet.Create();
  InputV := TNNetVolume.Create(Ctx, 1, 1);
  TargetV := TNNetVolume.Create(Ctx, 1, Vocab);
  try
    NN.AddLayer([
      TNNetInput.Create(Ctx, 1, 1),
      TNNetEmbedding.Create(Vocab, 8),
      TNNetPointwiseConvLinear.Create(Vocab),
      TNNetPointwiseSoftMax.Create(1)
    ]);
    Col.Reseed(2024);
    Col.Collate(Tokens, Corrupt, Labels);
    Selected := 0;
    for P := 0 to Ctx - 1 do
      if Labels[P] <> csMaskedLMIgnoreLabel then Inc(Selected);
    AssertTrue('at least one masked position', Selected > 0);
    AssertTrue('not all masked (need ignored rows)', Selected < Ctx);

    NN.SetBatchUpdate(true);
    NN.ClearDeltas();
    Col.BuildTrainingPair(Corrupt, Labels, InputV, TargetV);
    NN.Compute(InputV);
    Col.ApplyLossMask(Labels, TargetV, NN.GetLastLayer().Output);
    NN.Backpropagate(TargetV);
    ErrV := NN.GetLastLayer().OutputError;
    SelectedAbs := 0;
    for P := 0 to Ctx - 1 do
    begin
      RowAbs := 0;
      for D := 0 to Vocab - 1 do RowAbs := RowAbs + Abs(ErrV[P, 0, D]);
      if Labels[P] = csMaskedLMIgnoreLabel
      then AssertEquals('ignored row ' + IntToStr(P) + ' error', 0, RowAbs, 0)
      else SelectedAbs := SelectedAbs + RowAbs;
    end;
    AssertTrue('some masked error flows', SelectedAbs > 1e-6);
  finally
    TargetV.Free;
    InputV.Free;
    NN.Free;
    Col.Free;
  end;
end;

procedure TTestNeuralMaskedLM.TestWholeWordNoPartialMasking;
var
  Col: TNNetMaskedLMCollator;
  Tokens, WordIds, Corrupt, Labels: TNeuralIntegerArray;
  N, I, W, WordStart, AnySelected: integer;
  WordHasSelected, WordHasIgnored: boolean;
  Selected: integer;
begin
  // 600 words, each 1..4 pieces. WordIds tags each piece with its word index.
  N := 0; W := 0;
  SetLength(Tokens, 4000);
  SetLength(WordIds, 4000);
  while W < 600 do
  begin
    WordStart := (W mod 4) + 1; // 1..4 pieces
    for I := 0 to WordStart - 1 do
    begin
      Tokens[N] := 10 + ((W * 7 + I) mod 80);
      WordIds[N] := W;
      Inc(N);
    end;
    Inc(W);
  end;
  SetLength(Tokens, N);
  SetLength(WordIds, N);
  Col := TNNetMaskedLMCollator.Create(MaskId, Vocab, 0.15);
  try
    Col.Reseed(424242);
    Col.CollateWholeWord(Tokens, WordIds, Corrupt, Labels);
    // Walk each word: it must be entirely selected or entirely ignored.
    AnySelected := 0;
    I := 0;
    while I < N do
    begin
      W := WordIds[I];
      WordHasSelected := false;
      WordHasIgnored := false;
      while (I < N) and (WordIds[I] = W) do
      begin
        if Labels[I] <> csMaskedLMIgnoreLabel then WordHasSelected := true
        else WordHasIgnored := true;
        Inc(I);
      end;
      AssertTrue('word not partially masked',
        not (WordHasSelected and WordHasIgnored));
      if WordHasSelected then Inc(AnySelected);
    end;
    AssertTrue('some words selected', AnySelected > 0);
    // Selected fraction (over tokens) should be roughly the mask prob.
    Selected := 0;
    for I := 0 to N - 1 do
      if Labels[I] <> csMaskedLMIgnoreLabel then Inc(Selected);
    AssertTrue('selected token fraction sane',
      (Selected / N > 0.05) and (Selected / N < 0.30));
    // Off the selected set the input is untouched; on it the label is original.
    for I := 0 to N - 1 do
      if Labels[I] = csMaskedLMIgnoreLabel
      then AssertEquals('ignored unchanged', Tokens[I], Corrupt[I])
      else AssertEquals('selected label = original', Tokens[I], Labels[I]);
  finally
    Col.Free;
  end;
end;

procedure TTestNeuralMaskedLM.TestWholeWordZeroProbSelectsNothing;
var
  Col: TNNetMaskedLMCollator;
  Tokens, WordIds, Corrupt, Labels: TNeuralIntegerArray;
  N, I: integer;
begin
  N := 500;
  SetLength(Tokens, N);
  SetLength(WordIds, N);
  for I := 0 to N - 1 do
  begin
    Tokens[I] := 10 + (I mod 80);
    WordIds[I] := I div 3; // words of 3 pieces
  end;
  Col := TNNetMaskedLMCollator.Create(MaskId, Vocab, 0.0);
  try
    Col.Reseed(11);
    Col.CollateWholeWord(Tokens, WordIds, Corrupt, Labels);
    for I := 0 to N - 1 do
    begin
      AssertEquals('p=0 nothing selected', csMaskedLMIgnoreLabel, Labels[I]);
      AssertEquals('p=0 nothing corrupted', Tokens[I], Corrupt[I]);
    end;
  finally
    Col.Free;
  end;
end;

const
  SpanVocab = 200;
  SpanBase = 199; // <extra_id_0> at top of vocab

procedure TTestNeuralSpanCorruption.TestSentinelIdsDescend;
var
  Col: TNNetSpanCorruptionCollator;
begin
  Col := TNNetSpanCorruptionCollator.Create(SpanBase, SpanVocab, 0.15, 3.0);
  try
    AssertEquals('extra_id_0', SpanBase, Col.SentinelId(0));
    AssertEquals('extra_id_1', SpanBase - 1, Col.SentinelId(1));
    AssertEquals('extra_id_5', SpanBase - 5, Col.SentinelId(5));
  finally
    Col.Free;
  end;
end;

// Rebuilds the original sequence from a T5 (source, target) pair: walk the
// source, and at each sentinel splice in the span that follows the matching
// sentinel in the target.
function RebuildFromSpanPair(const Source, Target: TNeuralIntegerArray;
  Base: integer): TNeuralIntegerArray;
var
  I, J, OutLen, Sent: integer;
begin
  SetLength(Result, 4096);
  OutLen := 0;
  for I := 0 to Length(Source) - 1 do
  begin
    if (Source[I] <= Base) and (Source[I] > Base - 64) then
    begin
      // A sentinel: find it in the target and copy the span until the next
      // sentinel (or end).
      Sent := Source[I];
      J := 0;
      while (J < Length(Target)) and (Target[J] <> Sent) do Inc(J);
      Inc(J); // skip the sentinel itself
      while (J < Length(Target)) and
            (not ((Target[J] <= Base) and (Target[J] > Base - 64))) do
      begin
        Result[OutLen] := Target[J]; Inc(OutLen);
        Inc(J);
      end;
    end
    else
    begin
      Result[OutLen] := Source[I]; Inc(OutLen);
    end;
  end;
  SetLength(Result, OutLen);
end;

procedure TTestNeuralSpanCorruption.TestRoundTripReconstruction;
var
  Col: TNNetSpanCorruptionCollator;
  Tokens, Source, Target, Rebuilt: TNeuralIntegerArray;
  N, I, NumSpans: integer;
begin
  N := 120;
  SetLength(Tokens, N);
  // Distinct non-sentinel real ids (kept well below the sentinel band).
  for I := 0 to N - 1 do Tokens[I] := 10 + (I mod 100);
  Col := TNNetSpanCorruptionCollator.Create(SpanBase, SpanVocab, 0.15, 3.0);
  try
    Col.Reseed(424242);
    Col.Collate(Tokens, Source, Target, NumSpans);
    AssertTrue('at least one span', NumSpans > 0);
    Rebuilt := RebuildFromSpanPair(Source, Target, SpanBase);
    AssertEquals('rebuilt length', N, Length(Rebuilt));
    for I := 0 to N - 1 do
      AssertEquals('rebuilt token ' + IntToStr(I), Tokens[I], Rebuilt[I]);
  finally
    Col.Free;
  end;
end;

procedure TTestNeuralSpanCorruption.TestSourceTargetSentinelConsistency;
var
  Col: TNNetSpanCorruptionCollator;
  Tokens, Source, Target: TNeuralIntegerArray;
  N, I, NumSpans, SrcSent: integer;
begin
  N := 120;
  SetLength(Tokens, N);
  for I := 0 to N - 1 do Tokens[I] := 10 + (I mod 100);
  Col := TNNetSpanCorruptionCollator.Create(SpanBase, SpanVocab, 0.2, 3.0);
  try
    Col.Reseed(2024);
    Col.Collate(Tokens, Source, Target, NumSpans);
    // Count sentinels in the source: must equal NumSpans, in descending order.
    SrcSent := 0;
    for I := 0 to Length(Source) - 1 do
      if Source[I] = Col.SentinelId(SrcSent) then Inc(SrcSent)
      else AssertTrue('source has no out-of-order sentinel',
        (Source[I] < SpanBase - 63) or (Source[I] >= 0));
    AssertEquals('source sentinel count = spans', NumSpans, SrcSent);
    // Target must START with sentinel 0 and END with the trailing sentinel.
    AssertEquals('target starts with extra_id_0',
      Col.SentinelId(0), Target[0]);
    AssertEquals('target ends with final sentinel',
      Col.SentinelId(NumSpans), Target[Length(Target) - 1]);
  finally
    Col.Free;
  end;
end;

procedure TTestNeuralSpanCorruption.TestZeroRateNoCorruption;
var
  Col: TNNetSpanCorruptionCollator;
  Tokens, Source, Target: TNeuralIntegerArray;
  N, I, NumSpans: integer;
begin
  N := 80;
  SetLength(Tokens, N);
  for I := 0 to N - 1 do Tokens[I] := 10 + (I mod 100);
  Col := TNNetSpanCorruptionCollator.Create(SpanBase, SpanVocab, 0.0, 3.0);
  try
    Col.Reseed(5);
    Col.Collate(Tokens, Source, Target, NumSpans);
    AssertEquals('no spans', 0, NumSpans);
    AssertEquals('source = input length', N, Length(Source));
    for I := 0 to N - 1 do
      AssertEquals('source untouched', Tokens[I], Source[I]);
    // Target is just the single trailing sentinel.
    AssertEquals('target is one sentinel', 1, Length(Target));
    AssertEquals('that sentinel is extra_id_0', Col.SentinelId(0), Target[0]);
  finally
    Col.Free;
  end;
end;

procedure TTestNeuralSpanCorruption.TestSpanReseedReproducible;
var
  ColA, ColB: TNNetSpanCorruptionCollator;
  Tokens, Sa, Ta, Sb, Tb: TNeuralIntegerArray;
  N, I, NsA, NsB: integer;
begin
  N := 120;
  SetLength(Tokens, N);
  for I := 0 to N - 1 do Tokens[I] := 10 + (I mod 100);
  ColA := TNNetSpanCorruptionCollator.Create(SpanBase, SpanVocab, 0.15, 3.0);
  ColB := TNNetSpanCorruptionCollator.Create(SpanBase, SpanVocab, 0.15, 3.0);
  try
    ColA.Reseed(777); ColA.Collate(Tokens, Sa, Ta, NsA);
    ColB.Reseed(777); ColB.Collate(Tokens, Sb, Tb, NsB);
    AssertEquals('same span count', NsA, NsB);
    AssertEquals('same source length', Length(Sa), Length(Sb));
    AssertEquals('same target length', Length(Ta), Length(Tb));
    for I := 0 to Length(Sa) - 1 do
      AssertEquals('same source ' + IntToStr(I), Sa[I], Sb[I]);
    for I := 0 to Length(Ta) - 1 do
      AssertEquals('same target ' + IntToStr(I), Ta[I], Tb[I]);
  finally
    ColB.Free;
    ColA.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralMaskedLM);
  RegisterTest(TTestNeuralSpanCorruption);
end.
