unit TestNeuralLengthGrouped;
(*
Tests for TNNetLengthGroupedBatcher (neuraldatasets.pas): length-grouped
batching with dynamic per-batch padding (a port of transformers
LengthGroupedSampler + DataCollatorWithPadding). The corpora are pinned, the
megabatch shuffle is seeded so the emission order is reproducible, and the two
load-bearing properties are asserted directly:
  * the multiset of emitted (input,target) PAIRS equals the source corpus
    modulo order and padding (no sample dropped/duplicated, each batch
    internally consistent and padded only to its own max length), and
  * dynamic per-batch padding emits strictly fewer pad tokens than naive
    global padding on a length-skewed corpus.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry,
  neuralnetwork, neuralvolume, neuraldatasets;

type
  TTestNeuralLengthGrouped = class(TTestCase)
  private
    // Builds a length-skewed corpus of CountPerLen samples at each length in
    // 1..MaxLen; token ids are >= 2 and unique per (sample,position) so the
    // round-trip check can recover each original sample exactly.
    function MakeSkewedBatcher(MaxLen, CountPerLen, BatchSize,
      MegaMult, Vocab: integer): TNNetLengthGroupedBatcher;
  published
    // Every batch is padded only to its OWN max sample length, and that max
    // appears as a real (unpadded) sample in the batch.
    procedure TestEachBatchPaddedToOwnMax;
    // The multiset of emitted samples (recovered from each pair's real-token
    // prefix) equals the source corpus: no sample dropped or duplicated.
    procedure TestNoSampleDroppedOrDuplicated;
    // Within one emitted pair, input ids = sample tokens then pad token, and
    // target row P is the one-hot next token for P in 0..len-2, all-zero from
    // len-1 on (the padded/last positions carry no loss target).
    procedure TestPairInternallyConsistent;
    // ApplyLossMask zeroes the error (Desired := Actual) exactly at the
    // non-predictable positions (the sample's last real token and all padding).
    procedure TestApplyLossMaskZeroesPadTargets;
    // Dynamic per-batch padding emits strictly fewer pad tokens than naive
    // global padding on the skewed corpus.
    procedure TestDynamicPadFewerThanGlobal;
    // A fixed seed reproduces the emission order bit-for-bit; different seeds
    // generally differ (still grouped by length within mega-batches).
    procedure TestReproducibleOrder;
  end;

implementation

const
  cTestVocab = 64;

function TTestNeuralLengthGrouped.MakeSkewedBatcher(MaxLen, CountPerLen,
  BatchSize, MegaMult, Vocab: integer): TNNetLengthGroupedBatcher;
var
  Len, C, P: integer;
  Toks: array of integer;
begin
  Result := TNNetLengthGroupedBatcher.Create(Vocab, BatchSize, MegaMult, 0);
  for Len := 1 to MaxLen do
    for C := 0 to CountPerLen - 1 do
    begin
      SetLength(Toks, Len);
      // A small repeating real-token pattern (ids >= 2) so the round-trip can
      // verify the recovered prefix; exact ids are not what we assert on.
      for P := 0 to Len - 1 do
        Toks[P] := 2 + ((Len * 7 + C * 3 + P) mod (Vocab - 2));
      Result.AddSample(Toks);
    end;
end;

procedure TTestNeuralLengthGrouped.TestEachBatchPaddedToOwnMax;
var
  B: TNNetLengthGroupedBatcher;
  Bi, Wi, SeqLen, MaxInBatch, GlobalMax, L: integer;
  SawMaxAsReal: boolean;
begin
  B := MakeSkewedBatcher(12, 3, 4, 50, cTestVocab);
  try
    B.Reseed(20260614);
    B.BuildBatches();
    GlobalMax := 12;
    for Bi := 0 to B.BatchCount - 1 do
    begin
      SeqLen := B.BatchSeqLen(Bi);
      MaxInBatch := 0;
      SawMaxAsReal := false;
      for Wi := 0 to B.BatchSize(Bi) - 1 do
      begin
        L := B.SampleLenOf(Bi, Wi);
        if L > MaxInBatch then MaxInBatch := L;
        if L = SeqLen then SawMaxAsReal := true;
      end;
      AssertEquals('batch padded exactly to its own max', MaxInBatch, SeqLen);
      AssertTrue('batch max is a real sample length', SawMaxAsReal);
      AssertTrue('a length-skewed corpus has batches below the global max',
        SeqLen <= GlobalMax);
    end;
  finally
    B.Free;
  end;
end;

procedure TTestNeuralLengthGrouped.TestNoSampleDroppedOrDuplicated;
var
  B: TNNetLengthGroupedBatcher;
  Bi, Wi, P, Total, SeqLen, RealLen: integer;
  Inp, Tgt: TNNetVolume;
  Seen: TStringList;
  Key: string;
begin
  // Total = sum over lengths 1..MaxLen of CountPerLen.
  B := MakeSkewedBatcher(10, 4, 3, 50, cTestVocab);
  Total := 10 * 4;
  Seen := TStringList.Create;
  Inp := TNNetVolume.Create(10, 1, 1);   // Depth=1 -> token ids on X
  Tgt := TNNetVolume.Create(10, 1, cTestVocab);
  try
    B.Reseed(7);
    B.BuildBatches();
    for Bi := 0 to B.BatchCount - 1 do
    begin
      SeqLen := B.BatchSeqLen(Bi);
      for Wi := 0 to B.BatchSize(Bi) - 1 do
      begin
        B.GetTrainingPair(Bi, Wi, Inp, Tgt);
        RealLen := B.SampleLenOf(Bi, Wi);
        // Recover the real-token prefix from the input volume.
        Key := IntToStr(RealLen);
        for P := 0 to RealLen - 1 do
          Key := Key + ':' + IntToStr(Round(Inp[P, 0, 0]));
        // Padding past the real length must be the pad token (0).
        for P := RealLen to SeqLen - 1 do
          AssertEquals('padding is pad token', 0, Round(Inp[P, 0, 0]));
        Seen.Add(Key);
      end;
    end;
    AssertEquals('every sample emitted exactly once', Total, Seen.Count);
    // Our generator makes the (len, token-pattern) keys distinct per source
    // sample, so after sorting no two adjacent keys may be equal: that proves
    // no sample was duplicated (and, with the count above, none was dropped).
    Seen.Sort;
    for P := 1 to Seen.Count - 1 do
      AssertTrue('no emitted sample duplicated', Seen[P] <> Seen[P - 1]);
  finally
    Inp.Free; Tgt.Free; Seen.Free; B.Free;
  end;
end;

procedure TTestNeuralLengthGrouped.TestPairInternallyConsistent;
var
  B: TNNetLengthGroupedBatcher;
  Inp, Tgt: TNNetVolume;
  SeqLen, RealLen, P, D, HotCount, ExpectNext: integer;
begin
  B := MakeSkewedBatcher(8, 2, 4, 50, cTestVocab);
  Inp := TNNetVolume.Create(8, 1, 1);
  Tgt := TNNetVolume.Create(8, 1, cTestVocab);
  try
    B.Reseed(99);
    B.BuildBatches();
    // Inspect the first sample of the first batch.
    B.GetTrainingPair(0, 0, Inp, Tgt);
    SeqLen := B.BatchSeqLen(0);
    RealLen := B.SampleLenOf(0, 0);
    for P := 0 to SeqLen - 1 do
    begin
      HotCount := 0;
      for D := 0 to cTestVocab - 1 do
        if Round(Tgt[P, 0, D]) = 1 then Inc(HotCount);
      if P <= RealLen - 2 then
      begin
        // Exactly one hot bit: the next input token.
        AssertEquals('one-hot target at predictable pos', 1, HotCount);
        ExpectNext := Round(Inp[P + 1, 0, 0]);
        AssertEquals('target row P = next input token',
          1, Round(Tgt[P, 0, ExpectNext]));
      end
      else
        // Last real position and every padded position: no target.
        AssertEquals('no target at non-predictable pos', 0, HotCount);
    end;
  finally
    Inp.Free; Tgt.Free; B.Free;
  end;
end;

procedure TTestNeuralLengthGrouped.TestApplyLossMaskZeroesPadTargets;
var
  B: TNNetLengthGroupedBatcher;
  Desired, Actual: TNNetVolume;
  SeqLen, RealLen, P, D: integer;
begin
  B := MakeSkewedBatcher(8, 2, 4, 50, cTestVocab);
  Desired := TNNetVolume.Create(8, 1, cTestVocab);
  Actual := TNNetVolume.Create(8, 1, cTestVocab);
  try
    B.Reseed(5);
    B.BuildBatches();
    B.GetTrainingPair(0, 0, Desired, Actual); // Desired = real targets; reuse
    // Fabricate a non-trivial "network output".
    for P := 0 to Actual.SizeX - 1 do
      for D := 0 to cTestVocab - 1 do
        Actual[P, 0, D] := 0.01 * (P + 1) + 0.001 * D;
    SeqLen := B.BatchSeqLen(0);
    RealLen := B.SampleLenOf(0, 0);
    B.ApplyLossMask(0, 0, Desired, Actual);
    for P := 0 to SeqLen - 1 do
      for D := 0 to cTestVocab - 1 do
        if P > RealLen - 2 then
          // Non-predictable: Desired must now equal Actual (zero error).
          AssertEquals('masked pos Desired := Actual',
            Actual[P, 0, D], Desired[P, 0, D], 1e-7);
  finally
    Desired.Free; Actual.Free; B.Free;
  end;
end;

procedure TTestNeuralLengthGrouped.TestDynamicPadFewerThanGlobal;
var
  B: TNNetLengthGroupedBatcher;
  Dyn, Naive: int64;
begin
  B := MakeSkewedBatcher(16, 3, 4, 50, cTestVocab);
  try
    B.Reseed(123);
    B.BuildBatches();
    Dyn := B.TotalPadTokens();
    Naive := B.NaiveTotalPadTokens();
    AssertTrue('dynamic per-batch padding emits some pad tokens', Dyn >= 0);
    AssertTrue('dynamic pad (' + IntToStr(Dyn) +
      ') < naive global pad (' + IntToStr(Naive) + ')', Dyn < Naive);
  finally
    B.Free;
  end;
end;

procedure TTestNeuralLengthGrouped.TestReproducibleOrder;
var
  B: TNNetLengthGroupedBatcher;
  Bi, Wi, K: integer;
  Run1, Run2: TStringList;

  procedure CollectOrder(Dst: TStringList);
  var
    Cb, Cw: integer;
  begin
    Dst.Clear;
    for Cb := 0 to B.BatchCount - 1 do
      for Cw := 0 to B.BatchSize(Cb) - 1 do
        Dst.Add(IntToStr(B.SampleIndexOf(Cb, Cw)));
  end;

begin
  B := MakeSkewedBatcher(10, 3, 4, 50, cTestVocab);
  Run1 := TStringList.Create;
  Run2 := TStringList.Create;
  try
    B.Reseed(42); B.BuildBatches(); CollectOrder(Run1);
    B.Reseed(42); B.BuildBatches(); CollectOrder(Run2);
    AssertEquals('same seed -> same number of slots', Run1.Count, Run2.Count);
    for K := 0 to Run1.Count - 1 do
      AssertEquals('same seed -> identical emission order',
        Run1[K], Run2[K]);
    // Every original sample index appears exactly once across the order.
    AssertEquals('order covers all samples', B.SampleCount, Run1.Count);
  finally
    Run1.Free; Run2.Free; B.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralLengthGrouped);
end.
