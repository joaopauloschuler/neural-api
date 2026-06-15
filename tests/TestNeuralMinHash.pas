unit TestNeuralMinHash;
(*
Tests for neuralminhash.pas: MinHash signatures, LSH banding and near-duplicate
corpus deduplication. The Jaccard-estimate tolerance is set from the estimator's
standard error ~ 1/sqrt(numHashes): with numHashes = 256 that is ~0.0625, so we
allow 0.10 absolute error on the constructed shingle sets (comfortably above the
stderr, well below any value that would let a wrong estimate pass).
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralminhash;

type
  TTestNeuralMinHash = class(TTestCase)
  private
    // Count kept docs in a mask.
    function CountKept(const Mask: TNeuralBooleanArray): integer;
    // Are docs I and J in the same cluster? (both representatives of the same
    // root is detectable via the mask + estimate, but we check via a fresh
    // hasher's estimated Jaccard being high; here we expose cluster membership
    // by comparing which single doc each maps to is not available, so tests use
    // KeepMask + counts which fully pin clustering for the small fixtures.)
    function BuildBase: string;
  published
    // Planted near-duplicates (one-word edits / small insertions) land in one
    // cluster: all but one representative are dropped.
    procedure TestPlantedNearDuplicatesClustered;
    // Clearly distinct documents are never merged: every doc is kept.
    procedure TestDistinctDocumentsAllKept;
    // Mixed corpus: one duplicate group + distinct docs -> exactly the right
    // kept count and a single cluster of the right size.
    procedure TestMixedCorpusStats;
    // MinHash estimates the true Jaccard within the stderr-derived tolerance.
    procedure TestJaccardEstimateAccuracy;
    // Identical documents always estimate Jaccard = 1.0 exactly.
    procedure TestIdenticalJaccardIsOne;
    // Determinism: same seed -> identical signatures, masks and stats.
    procedure TestDeterminismSameSeed;
    // Bands * Rows = NumHashes invariant and the divisor guard.
    procedure TestBandingInvariant;
    // Short documents (fewer tokens than N) still get a stable signature and
    // identical short docs deduplicate.
    procedure TestShortDocumentsDeduplicate;
  end;

implementation

const
  csHashes = 256; // stderr ~ 1/sqrt(256) = 0.0625
  csTol = 0.10;   // Jaccard tolerance: > stderr, < any cross-doc gap we test
  csN = 5;        // word 5-grams
  csSeed = 20260615;

function TTestNeuralMinHash.CountKept(const Mask: TNeuralBooleanArray): integer;
var I: integer;
begin
  Result := 0;
  for I := 0 to High(Mask) do if Mask[I] then Inc(Result);
end;

function TTestNeuralMinHash.BuildBase: string;
begin
  // A long, content-rich base document so 5-gram shingle sets are large and a
  // one-word edit changes only a few shingles (high Jaccard between variants).
  Result := 'the quick brown fox jumps over the lazy dog while the curious ' +
    'cat watches from the old wooden fence near the river in the early ' +
    'morning light as birds sing softly in the tall green trees above';
end;

procedure TTestNeuralMinHash.TestPlantedNearDuplicatesClustered;
var
  Docs: TStringList;
  Mask: TNeuralBooleanArray;
  Stats: TNeuralDedupStats;
  Base: string;
begin
  Base := BuildBase;
  Docs := TStringList.Create;
  try
    Docs.Add(Base);                                   // 0: original
    Docs.Add(StringReplace(Base, 'lazy', 'sleepy', [])); // 1: one-word edit
    Docs.Add(StringReplace(Base, 'quick', 'swift', [])); // 2: one-word edit
    Docs.Add('a totally inserted clause and then ' + Base); // 3: small insertion
    DeduplicateCorpus(Docs, csHashes, csN, 0.7, csSeed, Mask, Stats);
    AssertEquals('four docs in', 4, Stats.DocCount);
    // All four are near-duplicates of the base -> one cluster, keep exactly 1.
    AssertEquals('exactly one representative kept', 1, Stats.KeptCount);
    AssertEquals('three dropped as near-duplicates', 3, Stats.RemovedCount);
    AssertEquals('one near-duplicate cluster', 1, Stats.ClusterCount);
    AssertEquals('cluster spans all four docs', 4, Stats.LargestClusterSize);
    // The kept representative is the lowest index (the original).
    AssertTrue('doc 0 (original) is the representative', Mask[0]);
    AssertEquals('kept count matches mask', 1, CountKept(Mask));
  finally
    Docs.Free;
  end;
end;

procedure TTestNeuralMinHash.TestDistinctDocumentsAllKept;
var
  Docs: TStringList;
  Mask: TNeuralBooleanArray;
  Stats: TNeuralDedupStats;
begin
  Docs := TStringList.Create;
  try
    Docs.Add('the quick brown fox jumps over the lazy dog in the meadow today');
    Docs.Add('economic policy and inflation targets dominated the central bank meeting');
    Docs.Add('photosynthesis converts sunlight water and carbon dioxide into glucose and oxygen');
    Docs.Add('the spacecraft entered orbit around the distant icy moon after years');
    DeduplicateCorpus(Docs, csHashes, csN, 0.7, csSeed, Mask, Stats);
    AssertEquals('four distinct docs in', 4, Stats.DocCount);
    AssertEquals('all four kept', 4, Stats.KeptCount);
    AssertEquals('none removed', 0, Stats.RemovedCount);
    AssertEquals('no clusters', 0, Stats.ClusterCount);
    AssertEquals('mask keeps all', 4, CountKept(Mask));
  finally
    Docs.Free;
  end;
end;

procedure TTestNeuralMinHash.TestMixedCorpusStats;
var
  Docs: TStringList;
  Mask: TNeuralBooleanArray;
  Stats: TNeuralDedupStats;
  Base: string;
begin
  Base := BuildBase;
  Docs := TStringList.Create;
  try
    Docs.Add('economic policy and inflation targets dominated the central bank meeting today');
    Docs.Add(Base);                                   // 1: dup group anchor
    Docs.Add('photosynthesis converts sunlight water and carbon dioxide into glucose');
    Docs.Add(StringReplace(Base, 'lazy', 'sleepy', [])); // 3: near-dup of 1
    Docs.Add(StringReplace(Base, 'morning', 'evening', [])); // 4: near-dup of 1
    DeduplicateCorpus(Docs, csHashes, csN, 0.7, csSeed, Mask, Stats);
    AssertEquals('five docs in', 5, Stats.DocCount);
    // Docs 1,3,4 form one cluster (keep 1); docs 0,2 are distinct (kept).
    AssertEquals('one duplicate cluster', 1, Stats.ClusterCount);
    AssertEquals('cluster of three', 3, Stats.LargestClusterSize);
    AssertEquals('three kept (2 distinct + 1 representative)', 3, Stats.KeptCount);
    AssertEquals('two removed', 2, Stats.RemovedCount);
    AssertTrue('distinct doc 0 kept', Mask[0]);
    AssertTrue('dup-group anchor doc 1 kept', Mask[1]);
    AssertTrue('distinct doc 2 kept', Mask[2]);
    AssertTrue('near-dup doc 3 dropped', not Mask[3]);
    AssertTrue('near-dup doc 4 dropped', not Mask[4]);
  finally
    Docs.Free;
  end;
end;

procedure TTestNeuralMinHash.TestJaccardEstimateAccuracy;
var
  Hasher: TNeuralMinHasher;
  ShA, ShB: TStringList;
  SigA, SigB: TNeuralMinHashSignature;
  DocA, DocB: string;
  Truth, Est: double;
begin
  Hasher := TNeuralMinHasher.Create(csHashes, csN, csSeed);
  try
    DocA := BuildBase;
    // A two-word edit of the base: a handful of 5-grams change, the rest stay.
    DocB := StringReplace(StringReplace(BuildBase, 'quick', 'swift', []),
      'lazy', 'sleepy', []);
    ShA := Hasher.Shingle(DocA);
    ShB := Hasher.Shingle(DocB);
    try
      Truth := TrueJaccardOfSets(ShA, ShB);
      // Sanity: the two-word edit should leave a high (but < 1) true Jaccard.
      AssertTrue('true Jaccard is a high near-dup value', (Truth > 0.6) and (Truth < 1.0));
    finally
      ShA.Free; ShB.Free;
    end;
    SigA := Hasher.ComputeSignature(DocA);
    SigB := Hasher.ComputeSignature(DocB);
    Est := Hasher.EstimateJaccard(SigA, SigB);
    AssertEquals('MinHash estimate close to true Jaccard', Truth, Est, csTol);

    // A second, lower-overlap pair: disjoint topics share almost no 5-grams.
    DocA := 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu';
    DocB := 'red orange yellow green blue indigo violet white black gray brown pink';
    ShA := Hasher.Shingle(DocA);
    ShB := Hasher.Shingle(DocB);
    try
      Truth := TrueJaccardOfSets(ShA, ShB);
    finally
      ShA.Free; ShB.Free;
    end;
    SigA := Hasher.ComputeSignature(DocA);
    SigB := Hasher.ComputeSignature(DocB);
    Est := Hasher.EstimateJaccard(SigA, SigB);
    AssertEquals('disjoint pair true Jaccard ~ 0', 0.0, Truth, 1e-9);
    AssertEquals('disjoint pair estimate ~ 0', Truth, Est, csTol);
  finally
    Hasher.Free;
  end;
end;

procedure TTestNeuralMinHash.TestIdenticalJaccardIsOne;
var
  Hasher: TNeuralMinHasher;
  Sig: TNeuralMinHashSignature;
begin
  Hasher := TNeuralMinHasher.Create(csHashes, csN, csSeed);
  try
    Sig := Hasher.ComputeSignature(BuildBase);
    AssertEquals('identical signatures estimate Jaccard 1.0', 1.0,
      Hasher.EstimateJaccard(Sig, Sig), 1e-12);
  finally
    Hasher.Free;
  end;
end;

procedure TTestNeuralMinHash.TestDeterminismSameSeed;
var
  H1, H2: TNeuralMinHasher;
  Sig1, Sig2: TNeuralMinHashSignature;
  Docs: TStringList;
  Mask1, Mask2: TNeuralBooleanArray;
  Stats1, Stats2: TNeuralDedupStats;
  I: integer;
begin
  H1 := TNeuralMinHasher.Create(csHashes, csN, csSeed);
  H2 := TNeuralMinHasher.Create(csHashes, csN, csSeed);
  try
    Sig1 := H1.ComputeSignature(BuildBase);
    Sig2 := H2.ComputeSignature(BuildBase);
    AssertEquals('same signature length', Length(Sig1), Length(Sig2));
    for I := 0 to High(Sig1) do
      AssertTrue('signature element ' + IntToStr(I) + ' identical',
        Sig1[I] = Sig2[I]);
  finally
    H1.Free; H2.Free;
  end;
  Docs := TStringList.Create;
  try
    Docs.Add(BuildBase);
    Docs.Add(StringReplace(BuildBase, 'lazy', 'sleepy', []));
    Docs.Add('a completely different unrelated sentence about quantum field theory');
    DeduplicateCorpus(Docs, csHashes, csN, 0.7, csSeed, Mask1, Stats1);
    DeduplicateCorpus(Docs, csHashes, csN, 0.7, csSeed, Mask2, Stats2);
    AssertEquals('same kept count', Stats1.KeptCount, Stats2.KeptCount);
    AssertEquals('same cluster count', Stats1.ClusterCount, Stats2.ClusterCount);
    for I := 0 to High(Mask1) do
      AssertTrue('mask element ' + IntToStr(I) + ' identical',
        Mask1[I] = Mask2[I]);
  finally
    Docs.Free;
  end;
end;

procedure TTestNeuralMinHash.TestBandingInvariant;
var
  Hasher: TNeuralMinHasher;
  Raised: boolean;
begin
  Hasher := TNeuralMinHasher.Create(120, csN, csSeed);
  try
    AssertEquals('bands * rows = numHashes', 120,
      Hasher.Bands * Hasher.Rows);
    Hasher.Bands := 20; // 20 * 6 = 120
    AssertEquals('rows follow bands', 6, Hasher.Rows);
    Hasher.Rows := 8;   // 15 * 8 = 120
    AssertEquals('bands follow rows', 15, Hasher.Bands);
    // A non-divisor must raise.
    Raised := false;
    try
      Hasher.Bands := 7; // 120 mod 7 <> 0
    except
      on EArgumentException do Raised := true;
    end;
    AssertTrue('non-divisor band count raises', Raised);
  finally
    Hasher.Free;
  end;
end;

procedure TTestNeuralMinHash.TestShortDocumentsDeduplicate;
var
  Docs: TStringList;
  Mask: TNeuralBooleanArray;
  Stats: TNeuralDedupStats;
  Hasher: TNeuralMinHasher;
  Sig: TNeuralMinHashSignature;
  I: integer;
  AllMax: boolean;
begin
  // Documents shorter than N (=5) tokens still produce a stable, non-empty
  // signature (the whole-token-list fallback), and identical short docs dedup.
  Hasher := TNeuralMinHasher.Create(csHashes, csN, csSeed);
  try
    Sig := Hasher.ComputeSignature('two words');
    AllMax := true;
    for I := 0 to High(Sig) do
      if Sig[I] <> High(UInt64) then AllMax := false;
    AssertTrue('short-doc signature is populated (not all sentinel)', not AllMax);
  finally
    Hasher.Free;
  end;
  Docs := TStringList.Create;
  try
    Docs.Add('short header line');
    Docs.Add('short header line');   // exact dup of 0
    Docs.Add('a different short line');
    DeduplicateCorpus(Docs, csHashes, csN, 0.7, csSeed, Mask, Stats);
    AssertEquals('exact short dup removed', 1, Stats.RemovedCount);
    AssertEquals('two kept', 2, Stats.KeptCount);
    AssertTrue('first short doc kept', Mask[0]);
    AssertTrue('exact dup dropped', not Mask[1]);
    AssertTrue('distinct short doc kept', Mask[2]);
  finally
    Docs.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralMinHash);
end.
