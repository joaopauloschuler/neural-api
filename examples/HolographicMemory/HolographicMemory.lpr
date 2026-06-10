program HolographicMemory;
(*
HolographicMemory: a cleanup-memory demo of TNNetHolographicBinding, the
Holographic Reduced Representation (HRR) vector-symbolic BINDING layer.

HRR in one paragraph. Fix a dimension n. Atomic symbols are random unit-ish
n-vectors. The BIND operator is the circular convolution  c = a (conv) b
(c[k] = sum_j a[j]*b[(k-j) mod n]); it produces a new n-vector that is
dissimilar to both a and b, so a role-filler pair (key (conv) value) can be
SUPERPOSED (just added) with many others into ONE fixed-width trace
    t = sum_i  key_i (conv) value_i.
To query the trace by a key, UNBIND with the circular correlation
    q = key (corr) t      (the approximate inverse),
which returns a NOISY copy of the bound value plus crosstalk from the other
pairs. A "cleanup memory" then snaps q to the nearest clean codebook vector.
Both operations are exactly TNNetHolographicBinding (Unbind=0 / Unbind=1) on a
(1,1,2n) input whose first n channels are the left vector and last n the right.

This program:
  1. samples a codebook of random unit keys and values (dimension n),
  2. for a growing number P of stored pairs, binds P key->value pairs into ONE
     superposed trace, unbinds by each stored key, cleans up against the value
     codebook (nearest cosine), and measures recall accuracy,
  3. prints the HRR CAPACITY CURVE: recall stays ~perfect while P is small and
     degrades GRACEFULLY (not catastrophically) as the trace is overloaded.
It also sanity-checks the algebra on a single pair: unbinding the bound trace by
the correct key returns a vector whose nearest codebook entry is the right value
and whose cosine similarity to it is high.

Pure CPU, tiny data, no training (HRR binding is weightless) -- runs in a couple
of seconds and a few MB.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF} Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  N        = 256;   // HRR vector dimension
  CodeSize = 24;    // number of distinct key/value atoms in the codebook
  Trials   = 40;    // random codebooks averaged per capacity point

var
  Keys, Vals: array of TNNetVolume;          // codebook atoms (1,1,N)
  BindNet, UnbindNet: TNNet;
  TmpL, TmpR, Trace, Query: TNNetVolume;

// Fill V with a random zero-mean unit-norm n-vector (the standard HRR atom).
procedure SampleAtom(V: TNNetVolume);
var i: integer; norm: TNeuralFloat;
begin
  for i := 0 to N - 1 do V.Raw[i] := V.RandomGaussianValue();
  norm := 0;
  for i := 0 to N - 1 do norm := norm + V.Raw[i] * V.Raw[i];
  norm := Sqrt(norm);
  if norm < 1e-12 then norm := 1e-12;
  for i := 0 to N - 1 do V.Raw[i] := V.Raw[i] / norm;
end;

// Pack left|right into a (1,1,2N) volume and run the given weightless net.
function RunHRR(Net: TNNet; ALeft, ARight: TNNetVolume): TNNetVolume;
var i: integer; Inp: TNNetVolume;
begin
  Inp := TNNetVolume.Create(1, 1, 2 * N);
  try
    for i := 0 to N - 1 do
    begin
      Inp.Raw[i]     := ALeft.Raw[i];
      Inp.Raw[N + i] := ARight.Raw[i];
    end;
    Net.Compute(Inp);
    Result := TNNetVolume.Create(1, 1, N);
    Result.Copy(Net.GetLastLayer.Output);
  finally
    Inp.Free;
  end;
end;

function Cosine(A, B: TNNetVolume): TNeuralFloat;
var i: integer; dot, na, nb: TNeuralFloat;
begin
  dot := 0; na := 0; nb := 0;
  for i := 0 to N - 1 do
  begin
    dot := dot + A.Raw[i] * B.Raw[i];
    na  := na  + A.Raw[i] * A.Raw[i];
    nb  := nb  + B.Raw[i] * B.Raw[i];
  end;
  if (na < 1e-12) or (nb < 1e-12) then Result := 0
  else Result := dot / (Sqrt(na) * Sqrt(nb));
end;

// Nearest value-codebook index by cosine similarity (the cleanup memory).
function CleanupNearestVal(Q: TNNetVolume): integer;
var i, best: integer; bestSim, sim: TNeuralFloat;
begin
  best := 0; bestSim := -2;
  for i := 0 to CodeSize - 1 do
  begin
    sim := Cosine(Q, Vals[i]);
    if sim > bestSim then begin bestSim := sim; best := i; end;
  end;
  Result := best;
end;

procedure BuildNets;
begin
  BindNet := TNNet.Create();
  BindNet.AddLayer(TNNetInput.Create(1, 1, 2 * N));
  BindNet.AddLayer(TNNetHolographicBinding.Create(0)); // bind = circular convolution

  UnbindNet := TNNet.Create();
  UnbindNet.AddLayer(TNNetInput.Create(1, 1, 2 * N));
  UnbindNet.AddLayer(TNNetHolographicBinding.Create(1)); // unbind = circular correlation
end;

// Average recall accuracy when P key->value pairs are superposed into one trace.
function CapacityAt(P: integer): TNeuralFloat;
var
  trial, i, k, queryKey, recovered, hits, total: integer;
  perm: array of integer;
  bound: TNNetVolume;
begin
  hits := 0; total := 0;
  SetLength(perm, CodeSize);
  for trial := 1 to Trials do
  begin
    // fresh random codebook each trial
    for i := 0 to CodeSize - 1 do
    begin
      SampleAtom(Keys[i]);
      SampleAtom(Vals[i]);
      perm[i] := i;
    end;
    // choose P distinct pairs (use the first P indices of the codebook)
    Trace.Fill(0);
    for k := 0 to P - 1 do
    begin
      bound := RunHRR(BindNet, Keys[k], Vals[k]);
      try
        Trace.Add(bound);          // superpose key_k (*) val_k
      finally
        bound.Free;
      end;
    end;
    // query the trace by each stored key and clean up
    for queryKey := 0 to P - 1 do
    begin
      Query.Free;
      // unbind(a=trace, b=key) = trace (conv) involution(key) = inv(key) (conv) trace,
      // the HRR query that recovers an approximate copy of the bound value.
      Query := RunHRR(UnbindNet, Trace, Keys[queryKey]);
      recovered := CleanupNearestVal(Query);
      if recovered = queryKey then Inc(hits);
      Inc(total);
    end;
  end;
  if total = 0 then Result := 0 else Result := hits / total;
end;

procedure SinglePairSanity;
var bound, q: TNNetVolume; nearest: integer; sim: TNeuralFloat;
begin
  SampleAtom(Keys[0]); SampleAtom(Vals[0]);
  SampleAtom(Keys[1]); SampleAtom(Vals[1]);
  bound := RunHRR(BindNet, Keys[0], Vals[0]);   // t = key0 (*) val0
  try
    WriteLn('Single-pair sanity (1 pair stored):');
    WriteLn('  cos(bind(key0,val0), val0)      = ', Cosine(bound, Vals[0]):0:4,
            '   (bound trace is DISSIMILAR to its filler -- by design)');
    q := RunHRR(UnbindNet, bound, Keys[0]);     // unbind by correct key
    try
      sim := Cosine(q, Vals[0]);
      nearest := CleanupNearestVal(q);
      WriteLn('  cos(unbind(key0, t), val0)      = ', sim:0:4,
              '   (recovers an APPROXIMATE copy of val0)');
      WriteLn('  cleanup nearest value index     = ', nearest,
              '   (expected 0)  ', BoolToStr(nearest = 0, 'OK', 'MISS'));
    finally
      q.Free;
    end;
  finally
    bound.Free;
  end;
end;

var
  i, P: integer;
  acc: TNeuralFloat;
begin
  RandSeed := 20260607;
  WriteLn('=== Holographic Reduced Representation (HRR) cleanup memory ===');
  WriteLn('  vector dim n = ', N, ',  codebook atoms = ', CodeSize,
          ',  trials/point = ', Trials);
  WriteLn('  bind   = TNNetHolographicBinding(Unbind=0)  (circular convolution)');
  WriteLn('  unbind = TNNetHolographicBinding(Unbind=1)  (circular correlation)');
  WriteLn;

  SetLength(Keys, CodeSize);
  SetLength(Vals, CodeSize);
  for i := 0 to CodeSize - 1 do
  begin
    Keys[i] := TNNetVolume.Create(1, 1, N);
    Vals[i] := TNNetVolume.Create(1, 1, N);
  end;
  TmpL  := TNNetVolume.Create(1, 1, N);
  TmpR  := TNNetVolume.Create(1, 1, N);
  Trace := TNNetVolume.Create(1, 1, N);
  Query := TNNetVolume.Create(1, 1, N);

  BuildNets;
  try
    SinglePairSanity;
    WriteLn;
    WriteLn('HRR capacity curve -- bind P pairs into ONE trace, unbind+cleanup:');
    WriteLn('   pairs P |  recall accuracy');
    WriteLn('   --------+-----------------');
    for P := 1 to CodeSize do
    begin
      acc := CapacityAt(P);
      Write('   ', P:7, ' |   ', (acc * 100):6:2, '%   ');
      for i := 1 to Round(acc * 30) do Write('#');
      WriteLn;
    end;
    WriteLn;
    WriteLn('Recall stays high for small P and degrades GRACEFULLY as the single');
    WriteLn('fixed-width trace is overloaded -- the hallmark HRR capacity tradeoff.');
  finally
    BindNet.Free;
    UnbindNet.Free;
    for i := 0 to CodeSize - 1 do
    begin
      Keys[i].Free;
      Vals[i].Free;
    end;
    TmpL.Free; TmpR.Free; Trace.Free; Query.Free;
  end;
end.
