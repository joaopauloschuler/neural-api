unit neuralminhash;

(*
neuralminhash
MinHash + LSH near-duplicate corpus deduplication: the C4 / Pile data-hygiene
step. Given a set of documents (strings) it finds near-duplicate clusters in
roughly linear time and keeps one representative per cluster, dropping the rest.

PIPELINE (Broder 1997 MinHash; Indyk & Motwani LSH banding):

  1. SHINGLING. Each document is lower-cased (optional, default on), split into
     whitespace tokens, and turned into the SET of word N-grams ("shingles",
     default N = 5 - the de-facto C4/Pile standard). Documents with fewer than
     N tokens fall back to a single shingle of the whole token list, so very
     short documents still get a (degenerate but stable) signature. Duplicate
     shingles within a document collapse to a set (MinHash is a set sketch).

  2. MINHASH SIGNATURE. Each shingle string is hashed to a 64-bit base value
     with FNV-1a. A family of numHashes universal permutations
        h_i(x) = (a_i * x + b_i) mod p        (p = 2^61 - 1, a Mersenne prime)
     is applied (a_i in [1,p), b_i in [0,p), from a deterministic seeded RNG).
     The signature's i-th element is the MINIMUM of h_i over the document's
     shingles. The estimated Jaccard similarity of two documents is the FRACTION
     of signature positions where the two signatures agree - an unbiased
     estimator with standard error ~ 1/sqrt(numHashes) (numHashes = 128 gives
     ~0.088 stderr).

  3. LSH BANDING. The numHashes-long signature is split into b BANDS of r ROWS
     (numHashes = b*r; choose them so the S-curve threshold ~ (1/b)^(1/r) sits
     near your target similarity). Two documents are CANDIDATE near-duplicates
     if any band's r-tuple of signature values is identical (same band bucket).
     This is the sub-quadratic filter: documents that share no band are never
     compared. A higher b (more bands) lowers the threshold (more recall); a
     higher r (longer bands) raises it (more precision).

  4. CLUSTER + KEEP ONE. Candidate pairs are unioned with union-find. By default
     a candidate pair is CONFIRMED only if its estimated Jaccard >= Threshold
     (default 0.8) - LSH over-generates, so this prunes the false positives the
     banding admits. Threshold <= 0 keeps every banded candidate (pure LSH). One
     representative per cluster is kept: the LOWEST original index (stable,
     order-preserving). KeepMask[i] = true means "keep document i".

PUBLIC API:

  TNeuralMinHasher - the configurable sketcher. Construct with
    Create(NumHashes, NGramSize, Seed). Properties Bands/Rows (must multiply to
    NumHashes; defaults factor NumHashes as evenly as possible) and Lowercase.
    ComputeSignature(Doc): TNeuralMinHashSignature (= array of UInt64).
    Shingle(Doc): TStringList of the document's shingles (caller frees).
    EstimateJaccard(SigA, SigB): fraction of matching signature positions.
    BandKey(Sig, BandIdx): the string bucket key of one band.

  TrueJaccardOfSets(A, B): the exact Jaccard of two shingle TStringLists.

  DeduplicateCorpus(Docs, NumHashes, NGramSize, Threshold, Seed, out KeepMask,
    out Stats) - the top-level dedup. Docs is a TStringList (one document per
    line) or an open array of string. KeepMask[i] = keep document i. Stats
    reports DocCount, KeptCount, RemovedCount, ClusterCount (clusters of size
    > 1), LargestClusterSize and DuplicateClusterCount.

  Hashing is DETERMINISTIC given Seed: same Seed -> same signatures, candidates
  and clusters (reproducible tests / runs).

Dependency-light: only Classes, SysUtils, Math. No layer/training dependencies.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math;

type
  // A MinHash signature: one 64-bit minimum per hash permutation.
  TNeuralMinHashSignature = array of UInt64;

  // Keep/drop mask returned by DeduplicateCorpus (locally defined to keep the
  // unit dependency-light - identical layout to types.TNeuralBooleanArray).
  TNeuralBooleanArray = array of boolean;

  // Statistics returned by DeduplicateCorpus.
  TNeuralDedupStats = record
    DocCount: integer;             // total documents in
    KeptCount: integer;            // documents kept (representatives + singletons)
    RemovedCount: integer;         // documents dropped as near-duplicates
    ClusterCount: integer;         // number of clusters of size > 1
    DuplicateClusterCount: integer;// alias of ClusterCount (clusters with dups)
    LargestClusterSize: integer;   // size of the biggest near-duplicate cluster
  end;

  { TNeuralMinHasher }

  TNeuralMinHasher = class(TObject)
  private
    FNumHashes: integer;
    FNGramSize: integer;
    FSeed: longword;
    FBands: integer;
    FRows: integer;
    FLowercase: boolean;
    FA: array of UInt64; // per-permutation multiplier a_i in [1, p)
    FB: array of UInt64; // per-permutation offset b_i in [0, p)
    procedure InitPermutations;
    procedure SetBands(AValue: integer);
    procedure SetRows(AValue: integer);
  public
    // NumHashes signature length; NGramSize word-shingle size; Seed determinism.
    constructor Create(NumHashes: integer = 128; NGramSize: integer = 5;
      Seed: longword = 1234567);
    // The set of word N-gram shingles of Doc (caller owns the returned list).
    function Shingle(const Doc: string): TStringList;
    // The MinHash signature of Doc (length = NumHashes).
    function ComputeSignature(const Doc: string): TNeuralMinHashSignature;
    // Estimated Jaccard = fraction of matching signature positions in [0,1].
    function EstimateJaccard(const SigA, SigB: TNeuralMinHashSignature): double;
    // The string bucket key of band BandIdx (0..Bands-1) of a signature.
    function BandKey(const Sig: TNeuralMinHashSignature;
      BandIdx: integer): string;
    property NumHashes: integer read FNumHashes;
    property NGramSize: integer read FNGramSize;
    property Seed: longword read FSeed;
    property Bands: integer read FBands write SetBands;
    property Rows: integer read FRows write SetRows;
    property Lowercase: boolean read FLowercase write FLowercase;
  end;

// 64-bit FNV-1a hash of a string (the base shingle hash; exposed for reuse).
function FNV1a64(const S: string): UInt64;

// True Jaccard |A intersect B| / |A union B| of two shingle lists (as sets).
function TrueJaccardOfSets(A, B: TStringList): double;

// Deduplicate Docs (one document per line). KeepMask[i] = keep document i.
// Threshold <= 0 confirms every banded candidate (pure LSH); otherwise a
// candidate pair is confirmed only when its estimated Jaccard >= Threshold.
procedure DeduplicateCorpus(Docs: TStringList; NumHashes, NGramSize: integer;
  Threshold: double; Seed: longword;
  out KeepMask: TNeuralBooleanArray; out Stats: TNeuralDedupStats);

// Open-array overload (no TStringList needed).
procedure DeduplicateCorpusArr(const Docs: array of string;
  NumHashes, NGramSize: integer; Threshold: double; Seed: longword;
  out KeepMask: TNeuralBooleanArray; out Stats: TNeuralDedupStats);

implementation

const
  // Mersenne prime modulus for the universal hash family (2^61 - 1).
  csP: UInt64 = 2305843009213693951;

{ ---- base hashing ---- }

function FNV1a64(const S: string): UInt64;
const
  csOffset: UInt64 = 14695981039346656037;
  csPrime: UInt64 = 1099511628211;
var
  I, Len: integer;
begin
  Result := csOffset;
  Len := Length(S);
  for I := 1 to Len do
  begin
    Result := Result xor UInt64(Ord(S[I]));
    Result := Result * csPrime;
  end;
end;

// (a + b) mod p, with a,b already < p (< 2^61), so a+b < 2^62 never overflows.
function AddModP(a, b: UInt64): UInt64; inline;
begin
  Result := a + b;
  if Result >= csP then Result := Result - csP;
end;

// (a * b) mod p, overflow-safe russian-peasant multiply (a,b < p < 2^62).
function MulModP(a, b: UInt64): UInt64;
var
  res: UInt64;
begin
  res := 0;
  a := a mod csP;
  while b > 0 do
  begin
    if (b and 1) = 1 then
      res := AddModP(res, a);
    a := AddModP(a, a); // a = (2a) mod p
    b := b shr 1;
  end;
  Result := res;
end;

{ ---- TNeuralMinHasher ---- }

constructor TNeuralMinHasher.Create(NumHashes: integer; NGramSize: integer;
  Seed: longword);
var
  B, R: integer;
  Mid, Delta, BestDelta: double;
begin
  inherited Create;
  if NumHashes < 1 then NumHashes := 1;
  if NGramSize < 1 then NGramSize := 1;
  FNumHashes := NumHashes;
  FNGramSize := NGramSize;
  FSeed := Seed;
  FLowercase := true;
  // Default banding: over all divisor splits NumHashes = b*r, pick the one whose
  // LSH S-curve midpoint (1/b)^(1/r) is closest to a target similarity of 0.7.
  // This makes the banding RECALL near-duplicates around the documented default
  // confirm Threshold (0.8): a balanced b=r split (e.g. 16x16 at 256 hashes) has
  // a far-too-steep curve that misses genuine 0.75-Jaccard pairs entirely. The
  // confirm-by-estimated-Jaccard step then prunes the extra candidates this
  // looser banding admits.
  FBands := 1;
  BestDelta := 1e30;
  for B := 1 to NumHashes do
    if (NumHashes mod B) = 0 then
    begin
      R := NumHashes div B;
      Mid := Power(1.0 / B, 1.0 / R); // S-curve threshold for this (b,r)
      Delta := Abs(Mid - 0.7);
      if Delta < BestDelta then
      begin
        BestDelta := Delta;
        FBands := B;
      end;
    end;
  FRows := FNumHashes div FBands;
  InitPermutations;
end;

procedure TNeuralMinHasher.InitPermutations;
var
  I, NumHashesM1: integer;
  R: UInt64;
  State: UInt64;
begin
  SetLength(FA, FNumHashes);
  SetLength(FB, FNumHashes);
  // Deterministic splitmix64-style generator seeded by FSeed (independent of
  // the global RandSeed so results never depend on RNG call ordering).
  State := UInt64(FSeed) + 14695981039346656037;
  NumHashesM1 := FNumHashes - 1;
  for I := 0 to NumHashesM1 do
  begin
    // a_i in [1, p)
    State := State + 11400714819323198485;
    R := State;
    R := (R xor (R shr 30)) * 13787848793156543929;
    R := (R xor (R shr 27)) * 10723151780598845931;
    R := R xor (R shr 31);
    FA[I] := (R mod (csP - 1)) + 1;
    // b_i in [0, p)
    State := State + 11400714819323198485;
    R := State;
    R := (R xor (R shr 30)) * 13787848793156543929;
    R := (R xor (R shr 27)) * 10723151780598845931;
    R := R xor (R shr 31);
    FB[I] := R mod csP;
  end;
end;

procedure TNeuralMinHasher.SetBands(AValue: integer);
begin
  if (AValue < 1) or ((FNumHashes mod AValue) <> 0) then
    raise EArgumentException.CreateFmt(
      'Bands (%d) must be >= 1 and divide NumHashes (%d).',
      [AValue, FNumHashes]);
  FBands := AValue;
  FRows := FNumHashes div FBands;
end;

procedure TNeuralMinHasher.SetRows(AValue: integer);
begin
  if (AValue < 1) or ((FNumHashes mod AValue) <> 0) then
    raise EArgumentException.CreateFmt(
      'Rows (%d) must be >= 1 and divide NumHashes (%d).',
      [AValue, FNumHashes]);
  FRows := AValue;
  FBands := FNumHashes div FRows;
end;

function TNeuralMinHasher.Shingle(const Doc: string): TStringList;
var
  Tokens: TStringList;
  Work: string;
  I, J, TokenCount, TokenCountM1, LastStart, NGramM1: integer;
  Sh: string;
begin
  Result := TStringList.Create;
  Result.Sorted := true;
  Result.Duplicates := dupIgnore; // collapse to a SET
  Result.CaseSensitive := true;   // already lowercased below if requested
  if FLowercase then Work := SysUtils.LowerCase(Doc) else Work := Doc;
  Tokens := TStringList.Create;
  try
    Tokens.Delimiter := ' ';
    Tokens.StrictDelimiter := false; // any whitespace run splits (and trims)
    Tokens.DelimitedText := Work;
    // Drop empty tokens that StrictDelimiter=false can still leave on edges.
    for I := Tokens.Count - 1 downto 0 do
      if Trim(Tokens[I]) = '' then Tokens.Delete(I);
    TokenCount := Tokens.Count;
    if Tokens.Count = 0 then
    begin
      Result.Add(''); // empty-doc shingle: stable degenerate signature
      Exit;
    end;
    if Tokens.Count < FNGramSize then
    begin
      // Fewer tokens than N: one shingle of the whole (short) token list.
      Sh := '';
      TokenCountM1 := TokenCount - 1;
      for J := 0 to TokenCountM1 do
      begin
        if J > 0 then Sh := Sh + #31; // unit-separator joins tokens unambiguously
        Sh := Sh + Tokens[J];
      end;
      Result.Add(Sh);
      Exit;
    end;
    LastStart := TokenCount - FNGramSize;
    NGramM1 := FNGramSize - 1;
    for I := 0 to LastStart do
    begin
      Sh := '';
      for J := 0 to NGramM1 do
      begin
        if J > 0 then Sh := Sh + #31;
        Sh := Sh + Tokens[I + J];
      end;
      Result.Add(Sh);
    end;
  finally
    Tokens.Free;
  end;
end;

function TNeuralMinHasher.ComputeSignature(
  const Doc: string): TNeuralMinHashSignature;
var
  Shingles: TStringList;
  I, S, ShingleCount, ShingleCountM1, NumHashesM1: integer;
  Base, HVal: UInt64;
begin
  SetLength(Result, FNumHashes);
  NumHashesM1 := FNumHashes - 1;
  for I := 0 to NumHashesM1 do Result[I] := High(UInt64);
  Shingles := Shingle(Doc);
  try
    ShingleCount := Shingles.Count;
    ShingleCountM1 := ShingleCount - 1;
    for S := 0 to ShingleCountM1 do
    begin
      Base := FNV1a64(Shingles[S]) mod csP; // reduce into the field
      for I := 0 to NumHashesM1 do
      begin
        HVal := AddModP(MulModP(FA[I], Base), FB[I]);
        if HVal < Result[I] then Result[I] := HVal;
      end;
    end;
  finally
    Shingles.Free;
  end;
end;

function TNeuralMinHasher.EstimateJaccard(
  const SigA, SigB: TNeuralMinHashSignature): double;
var
  I, Match, N, NM1: integer;
begin
  N := Min(Length(SigA), Length(SigB));
  if N = 0 then Exit(0.0);
  Match := 0;
  NM1 := N - 1;
  for I := 0 to NM1 do
    if SigA[I] = SigB[I] then Inc(Match);
  Result := Match / N;
end;

function TNeuralMinHasher.BandKey(const Sig: TNeuralMinHashSignature;
  BandIdx: integer): string;
var
  R, Start, RowsM1: integer;
begin
  Start := BandIdx * FRows;
  Result := '';
  RowsM1 := FRows - 1;
  for R := 0 to RowsM1 do
    Result := Result + IntToHex(Sig[Start + R], 16);
end;

{ ---- set Jaccard ---- }

function TrueJaccardOfSets(A, B: TStringList): double;
var
  Inter, UnionCnt, I: integer;
  SA, SB: TStringList;
  Idx, ACount, BCount, SACount, ACountM1, BCountM1, SACountM1: integer;
begin
  // Work on sorted SET copies so |A| / |B| / intersection are exact.
  SA := TStringList.Create;
  SB := TStringList.Create;
  try
    SA.Sorted := true; SA.Duplicates := dupIgnore; SA.CaseSensitive := true;
    SB.Sorted := true; SB.Duplicates := dupIgnore; SB.CaseSensitive := true;
    ACount := A.Count;
    BCount := B.Count;
    ACountM1 := ACount - 1;
    BCountM1 := BCount - 1;
    for I := 0 to ACountM1 do SA.Add(A[I]);
    for I := 0 to BCountM1 do SB.Add(B[I]);
    if (SA.Count = 0) and (SB.Count = 0) then Exit(1.0);
    Inter := 0;
    SACount := SA.Count;
    SACountM1 := SACount - 1;
    for I := 0 to SACountM1 do
      if SB.Find(SA[I], Idx) then Inc(Inter);
    UnionCnt := SACount + SB.Count - Inter;
    if UnionCnt = 0 then Exit(0.0);
    Result := Inter / UnionCnt;
  finally
    SA.Free;
    SB.Free;
  end;
end;

{ ---- union-find dedup ---- }

type
  TIntArr = array of integer;

function UFFind(var Parent: TIntArr; X: integer): integer;
begin
  while Parent[X] <> X do
  begin
    Parent[X] := Parent[Parent[X]]; // path halving
    X := Parent[X];
  end;
  Result := X;
end;

procedure UFUnion(var Parent: TIntArr; X, Y: integer);
var
  Rx, Ry: integer;
begin
  Rx := UFFind(Parent, X);
  Ry := UFFind(Parent, Y);
  if Rx = Ry then Exit;
  // Always attach the higher root to the lower so the cluster root is its
  // smallest member -> the representative kept is the lowest original index.
  if Rx < Ry then Parent[Ry] := Rx else Parent[Rx] := Ry;
end;

procedure DeduplicateCorpusArr(const Docs: array of string;
  NumHashes, NGramSize: integer; Threshold: double; Seed: longword;
  out KeepMask: TNeuralBooleanArray; out Stats: TNeuralDedupStats);
var
  Hasher: TNeuralMinHasher;
  N, I, B, D, Root, NM1, BandsM1: integer;
  Sigs: array of TNeuralMinHashSignature;
  Buckets: TStringList; // band-bucket key -> first doc index seen (as Object)
  Parent: TIntArr;
  Key: string;
  Idx, FirstDoc: integer;
  ClusterSize: TIntArr;
begin
  N := Length(Docs);
  SetLength(KeepMask, N);
  FillChar(Stats, SizeOf(Stats), 0);
  Stats.DocCount := N;
  if N = 0 then Exit;
  NM1 := N - 1;

  Hasher := TNeuralMinHasher.Create(NumHashes, NGramSize, Seed);
  try
    SetLength(Sigs, N);
    for I := 0 to NM1 do
      Sigs[I] := Hasher.ComputeSignature(Docs[I]);

    SetLength(Parent, N);
    for I := 0 to NM1 do Parent[I] := I;

    // LSH banding: for each band, group docs sharing a band bucket; union the
    // confirmed candidate pairs. One bucket map per band (cleared between bands)
    // so only same-band collisions are candidates.
    BandsM1 := Hasher.Bands - 1;
    for B := 0 to BandsM1 do
    begin
      Buckets := TStringList.Create;
      try
        Buckets.Sorted := true;
        Buckets.CaseSensitive := true;
        Buckets.Duplicates := dupError; // we manage first-seen ourselves
        for D := 0 to NM1 do
        begin
          Key := Hasher.BandKey(Sigs[D], B);
          if Buckets.Find(Key, Idx) then
          begin
            FirstDoc := PtrInt(Buckets.Objects[Idx]);
            // Candidate pair (FirstDoc, D): confirm by estimated Jaccard.
            if (Threshold <= 0) or
               (Hasher.EstimateJaccard(Sigs[FirstDoc], Sigs[D]) >= Threshold)
            then
              UFUnion(Parent, FirstDoc, D);
            // Transitivity is handled by union-find: D joins FirstDoc's whole
            // cluster, and the lowest-index root keeps the cluster intact.
          end
          else
            Buckets.AddObject(Key, TObject(PtrInt(D)));
        end;
      finally
        Buckets.Free;
      end;
    end;

    // KeepMask: keep a doc iff it IS its cluster root (lowest index).
    SetLength(ClusterSize, N);
    for I := 0 to NM1 do ClusterSize[I] := 0;
    for I := 0 to NM1 do
    begin
      Root := UFFind(Parent, I);
      Inc(ClusterSize[Root]);
      KeepMask[I] := (Root = I);
    end;

    // Stats.
    Stats.KeptCount := 0;
    Stats.RemovedCount := 0;
    Stats.ClusterCount := 0;
    Stats.LargestClusterSize := 0;
    for I := 0 to NM1 do
    begin
      if KeepMask[I] then Inc(Stats.KeptCount) else Inc(Stats.RemovedCount);
      if ClusterSize[I] > 1 then
      begin
        Inc(Stats.ClusterCount);
        if ClusterSize[I] > Stats.LargestClusterSize then
          Stats.LargestClusterSize := ClusterSize[I];
      end;
    end;
    Stats.DuplicateClusterCount := Stats.ClusterCount;
  finally
    Hasher.Free;
  end;
end;

procedure DeduplicateCorpus(Docs: TStringList; NumHashes, NGramSize: integer;
  Threshold: double; Seed: longword;
  out KeepMask: TNeuralBooleanArray; out Stats: TNeuralDedupStats);
var
  Arr: array of string;
  I, DocCount, DocCountM1: integer;
begin
  DocCount := Docs.Count;
  SetLength(Arr, DocCount);
  DocCountM1 := DocCount - 1;
  for I := 0 to DocCountM1 do Arr[I] := Docs[I];
  DeduplicateCorpusArr(Arr, NumHashes, NGramSize, Threshold, Seed,
    KeepMask, Stats);
end;

end.
