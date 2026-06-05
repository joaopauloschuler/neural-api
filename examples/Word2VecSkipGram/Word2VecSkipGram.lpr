program Word2VecSkipGram;
(*
Word2VecSkipGram: a self-contained, from-scratch implementation of the classic
word2vec SKIP-GRAM WITH NEGATIVE SAMPLING (SGNS, Mikolov et al. 2013) word
embedding model, trained on a tiny BUILT-IN corpus (no download, pure CPU,
finishes in a few seconds). It exercises TNNetEmbedding in a NON-transformer,
UNSUPERVISED distributional-semantics setting -- distinct from SimpleNLP (a
char-level next-token language model) and CharTokenizer (tokenisation only).

WHAT SKIP-GRAM WITH NEGATIVE SAMPLING IS
----------------------------------------
Skip-gram learns, for every word, a dense vector such that a word predicts the
words that appear AROUND it. For a center word c and a context word w inside a
+/-window, the positive objective wants the dot product v_c . u_w to be large.
Computing a full softmax over the vocabulary is expensive, so NEGATIVE SAMPLING
replaces it with a set of cheap binary logistic-regression problems: for each
positive (c, w) pair we also draw K "noise" words from a unigram^0.75 table and
ask the model to score them as NOT-a-context. The per-pair loss is the
binary-cross-entropy of a sigmoid head over the dot product:

    L = -log sigma( v_c . u_w )  -  sum_{j=1..K} log sigma( -v_c . u_neg_j )

so positive dot products are pushed UP and negative dot products DOWN. The
distributional hypothesis ("words in similar contexts have similar meaning")
then makes the learned vectors cluster semantically, which is why simple cosine
nearest-neighbour lookups and the textbook analogy arithmetic
    king - man + woman ~= queen
emerge purely from co-occurrence statistics.

HOW IT IS WIRED ON TNNetEmbedding (no transformer, native backprop)
-------------------------------------------------------------------
Canonical word2vec keeps TWO embedding matrices: an INPUT ("center") matrix W
and an OUTPUT ("context") matrix W'. We realise each as its own
Input + TNNetEmbedding(vocab, d) network (one (1,1,d) weight row per vocabulary
id):

    center net : Input(1, 1, 1)        -> Embedding -> v_c   (1, 1, d)
    context net: Input(1 + K, 1, 1)    -> Embedding -> (1+K, 1, d):
        slab 0    : u_w   (positive context vector)
        slab 1..K : u_neg (K negative / noise vectors)

The SGNS loss and its analytic gradient are computed in Pascal from the two
nets' output slabs, in the style of examples/InfoNCEContrastive and the repo's
loss-layer pattern. We seed the gradient via the standard "target = output -
grad" trick: TNNet.Backpropagate(Target) sets the layer OutputError to
(output - target) = grad, and TNNetEmbedding.Backpropagate then applies that
per-slab gradient straight onto the corresponding word-vector rows.

The gradients (sg = sigmoid):
    dL/dv_c    = (sg(v_c.u_w) - 1) * u_w  +  sum_j sg(v_c.u_neg_j) * u_neg_j
    dL/du_w    = (sg(v_c.u_w) - 1) * v_c
    dL/du_neg  =  sg(v_c.u_neg_j)      * v_c
Using two matrices (rather than a single shared one) avoids the
self-interference that otherwise drives every cosine toward zero, and matches
the textbook SGNS formulation. Nearest neighbours and analogies are read from
the INPUT matrix W (the conventional choice).

EncodeZero=1 is required so that vocabulary id 0 also trains (TNNetEmbedding
skips id 0 by default, treating it as a padding token).

CONTRAST WITH THE OTHER NLP EXAMPLES
------------------------------------
- examples/SimpleNLP trains a CHAR-LEVEL next-token language model (a small
  transformer/decoder). It is SUPERVISED next-token prediction; its embeddings
  are a by-product, not the goal.
- examples/CharTokenizer is about TOKENISATION (turning text into ids), not
  representation learning.
This example is the missing UNSUPERVISED word-embedding demo: the vectors ARE
the deliverable, evaluated by cosine nearest neighbours and analogy arithmetic.

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

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cEmbedDim  = 24;      // d: word-vector dimensionality
  cWindow    = 2;       // +/- context window radius
  cNumNeg    = 8;       // K negatives drawn per positive pair
  cEpochs    = 1500;    // passes over all positive pairs
  cLearnRate = 0.15;    // initial SGD step on the embedding rows
  cMinRate   = 0.002;   // floor for the linearly decayed learning rate
  cSeed      = 424242;
  cTopK      = 6;       // nearest-neighbour list length

// A small but STRUCTURED corpus. Word families with consistent co-occurrence:
//   royalty x gender x age:  king/queen, prince/princess, man/woman, boy/girl
//   number: one/two/three (with singular/plural cue words)
//   animals: dog/cat/puppy/kitten + bark/meow
//   food:    bread/cheese/apple + eat/eats
// The repeated frames are what let nearest-neighbours and the gender/number
// analogies emerge from pure co-occurrence.
const
  cCorpus =
    'the king is a royal man . ' +
    'the queen is a royal woman . ' +
    'the prince is a young royal man . ' +
    'the princess is a young royal woman . ' +
    'a man is an adult boy . ' +
    'a woman is an adult girl . ' +
    'the boy is a young man . ' +
    'the girl is a young woman . ' +
    'the king and the queen rule the kingdom . ' +
    'the prince and the princess live in the kingdom . ' +
    'the king loves the queen . ' +
    'the prince loves the princess . ' +
    'a man loves a woman . ' +
    'a boy loves a girl . ' +
    'the king is the father of the prince . ' +
    'the queen is the mother of the princess . ' +
    'the man is the father of the boy . ' +
    'the woman is the mother of the girl . ' +
    'a king wears a crown . ' +
    'a queen wears a crown . ' +
    'the royal king sits on the throne . ' +
    'the royal queen sits on the throne . ' +
    'one dog is small . ' +
    'two dogs are small . ' +
    'one cat is small . ' +
    'two cats are small . ' +
    'the dog can bark . ' +
    'the cat can meow . ' +
    'a puppy is a young dog . ' +
    'a kitten is a young cat . ' +
    'the puppy likes to bark . ' +
    'the kitten likes to meow . ' +
    'the dog and the cat are animals . ' +
    'the puppy and the kitten are animals . ' +
    'one apple is sweet . ' +
    'two apples are sweet . ' +
    'the bread is fresh . ' +
    'the cheese is fresh . ' +
    'a man eats bread . ' +
    'a woman eats cheese . ' +
    'the boy eats an apple . ' +
    'the girl eats an apple . ' +
    'people eat bread and cheese . ' +
    'people eat apple and bread . ' +
    'one boy and two girls . ' +
    'one dog and two cats . ' +
    'one king and two queens . ' +
    'one man and two women . ' +
    'the man and the woman are people . ' +
    'the boy and the girl are children . ' +
    'a young king becomes an old king . ' +
    'a young queen becomes an old queen . ' +
    'the old man tells a story . ' +
    'the old woman tells a story . ' +
    'the dog runs fast . ' +
    'the cat runs fast . ' +
    'the king rules over the people . ' +
    'the queen rules over the people . ' +
    'a man and a woman walk together . ' +
    'a boy and a girl play together . ';

type
  TWordArray = array of string;
  TIntArray = array of integer;

var
  Vocab: TWordArray;       // id -> word
  Tokens: TIntArray;       // corpus as a stream of ids
  PairCenter: TIntArray;   // positive pairs: center ids
  PairContext: TIntArray;  // positive pairs: context ids
  NegTable: TIntArray;     // unigram^0.75 sampling table

// ---- tokenisation / vocabulary ------------------------------------------

function VocabIndexOf(const W: string): integer;
var I: integer;
begin
  for I := 0 to High(Vocab) do
    if Vocab[I] = W then Exit(I);
  Result := -1;
end;

procedure SplitWords(const S: string; out Words: TWordArray);
var
  I, Cnt: integer;
  Cur: string;
begin
  SetLength(Words, 0);
  Cnt := 0;
  Cur := '';
  for I := 1 to Length(S) do
  begin
    if S[I] = ' ' then
    begin
      if Cur <> '' then
      begin
        SetLength(Words, Cnt + 1);
        Words[Cnt] := Cur;
        Inc(Cnt);
        Cur := '';
      end;
    end
    else
      Cur := Cur + S[I];
  end;
  if Cur <> '' then
  begin
    SetLength(Words, Cnt + 1);
    Words[Cnt] := Cur;
  end;
end;

procedure BuildVocabAndTokens();
var
  Words: TWordArray;
  I, Idx, Cnt: integer;
begin
  SplitWords(cCorpus, Words);
  SetLength(Vocab, 0);
  SetLength(Tokens, Length(Words));
  Cnt := 0;
  for I := 0 to High(Words) do
  begin
    if Words[I] = '.' then        // sentence boundary: keep as a real token
    begin
      // '.' is a genuine token so the window does not cross sentences much;
      // it acts as a soft separator and a high-frequency neutral word.
    end;
    Idx := VocabIndexOf(Words[I]);
    if Idx < 0 then
    begin
      SetLength(Vocab, Length(Vocab) + 1);
      Idx := High(Vocab);
      Vocab[Idx] := Words[I];
    end;
    Tokens[Cnt] := Idx;
    Inc(Cnt);
  end;
end;

// ---- positive pairs + negative sampling table ---------------------------

function IsBoundary(Id: integer): boolean;
begin
  Result := Vocab[Id] = '.';
end;

// High-frequency, content-free function words. They stay in the vocabulary
// (the embedding still allocates a row), but they are never used as a center,
// a context, or a negative -- exactly the role word2vec's frequent-word
// subsampling plays. Removing them sharply cleans the co-occurrence signal so
// the content words (king/queen/dog/...) cluster meaningfully at this scale.
function IsStop(Id: integer): boolean;
var W: string;
begin
  W := Vocab[Id];
  Result := (W = '.') or (W = 'the') or (W = 'a') or (W = 'an') or
            (W = 'is') or (W = 'are') or (W = 'of') or (W = 'in') or
            (W = 'on') or (W = 'to') or (W = 'and') or (W = 'over') or
            (W = 'can') or (W = 'be');
end;

// Builds positive (center, context) pairs WITHIN each sentence (segments
// separated by '.'), skipping stop words on both sides.
procedure BuildPositivePairs();
var
  I, J, N, SegStart: integer;
begin
  N := 0;
  SetLength(PairCenter, 0);
  SetLength(PairContext, 0);
  SegStart := 0;
  for I := 0 to High(Tokens) do
  begin
    if IsBoundary(Tokens[I]) then
    begin
      SegStart := I + 1;   // next sentence starts after the period
      Continue;
    end;
    if IsStop(Tokens[I]) then Continue;
    for J := I - cWindow to I + cWindow do
    begin
      if (J < SegStart) or (J > High(Tokens)) or (J = I) then Continue;
      if IsBoundary(Tokens[J]) then Break;  // do not cross the sentence end
      if IsStop(Tokens[J]) then Continue;
      SetLength(PairCenter, N + 1);
      SetLength(PairContext, N + 1);
      PairCenter[N] := Tokens[I];
      PairContext[N] := Tokens[J];
      Inc(N);
    end;
  end;
end;

// Unigram^0.75 negative-sampling table (Mikolov et al. 2013).
procedure BuildNegTable();
var
  Counts: array of integer;
  Weights: array of double;
  Total: double;
  I, J, Slots, Filled, Take: integer;
const
  cTableSize = 20000;
begin
  SetLength(Counts, Length(Vocab));
  for I := 0 to High(Counts) do Counts[I] := 0;
  for I := 0 to High(Tokens) do Inc(Counts[Tokens[I]]);
  SetLength(Weights, Length(Vocab));
  Total := 0;
  for I := 0 to High(Vocab) do
  begin
    if IsStop(I) then
      Weights[I] := 0           // never sample a stop word as a negative
    else
      Weights[I] := Power(Counts[I], 0.75);
    Total := Total + Weights[I];
  end;
  SetLength(NegTable, cTableSize);
  Filled := 0;
  for I := 0 to High(Vocab) do
  begin
    Slots := Round((Weights[I] / Total) * cTableSize);
    Take := Slots;
    if Filled + Take > cTableSize then Take := cTableSize - Filled;
    for J := 0 to Take - 1 do
      NegTable[Filled + J] := I;
    Inc(Filled, Take);
  end;
  // pad any remainder with the last non-boundary word
  while Filled < cTableSize do
  begin
    NegTable[Filled] := PairCenter[0];
    Inc(Filled);
  end;
end;

function SampleNegative(AvoidCenter, AvoidContext: integer): integer;
begin
  repeat
    Result := NegTable[Random(Length(NegTable))];
  until (Result <> AvoidCenter) and (Result <> AvoidContext);
end;

// ---- vector access on the embedding weight matrix -----------------------

function EmbedLayer(NN: TNNet): TNNetLayer;
begin
  Result := NN.Layers[1];  // [0]=Input, [1]=TNNetEmbedding
end;

// Reads word-vector row Id from the embedding's weight matrix.
procedure ReadVec(NN: TNNet; Id: integer; out V: array of TNeuralFloat);
var
  W: TNNetVolume;
  D: integer;
begin
  W := EmbedLayer(NN).Neurons[0].Weights;
  for D := 0 to cEmbedDim - 1 do
    V[D] := W.FData[Id * cEmbedDim + D];
end;

function Sigmoid(X: TNeuralFloat): TNeuralFloat;
begin
  if X >= 0 then Result := 1.0 / (1.0 + Exp(-X))
  else begin Result := Exp(X); Result := Result / (1.0 + Result); end;
end;

function DotN(const A, B: array of TNeuralFloat): TNeuralFloat;
var D: integer;
begin
  Result := 0;
  for D := 0 to cEmbedDim - 1 do Result := Result + A[D] * B[D];
end;

function CosineN(const A, B: array of TNeuralFloat): TNeuralFloat;
var D: integer; na, nb: TNeuralFloat;
begin
  na := 0; nb := 0;
  for D := 0 to cEmbedDim - 1 do begin na := na + A[D]*A[D]; nb := nb + B[D]*B[D]; end;
  na := Sqrt(na); nb := Sqrt(nb);
  if (na = 0) or (nb = 0) then Exit(0);
  Result := DotN(A, B) / (na * nb);
end;

// ---- training -----------------------------------------------------------

// Word2vec uses TWO embedding matrices: an INPUT ("center") matrix W and an
// OUTPUT ("context") matrix W'. We realise them as two independent
// Input+TNNetEmbedding networks. The center net embeds one token (the center
// word); the context net embeds the positive + K negatives at once. Using two
// matrices (rather than a single shared one) avoids the self-interference that
// otherwise drives all cosines toward zero, and matches the canonical SGNS
// formulation. Nearest neighbours / analogies are read from the INPUT matrix.
function BuildCenterNet(VocabSize: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(1, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(VocabSize, cEmbedDim, {EncodeZero=}1));
  Result.InitWeights();
end;

function BuildContextNet(VocabSize: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(1 + cNumNeg, 1, 1)); // positive + K negatives
  Result.AddLayer(TNNetEmbedding.Create(VocabSize, cEmbedDim, {EncodeZero=}1));
  Result.InitWeights();
end;

// One SGD step on a single positive pair with K fresh negatives.
// Returns the SGNS loss for this step. The gradient is seeded through the
// "target = output - grad" trick so each TNNetEmbedding applies it to its rows.
function TrainStep(NetC, NetO: TNNet; InC, InO, TgtC, TgtO: TNNetVolume;
  Center, Context: integer): TNeuralFloat;
var
  OutC, OutO: TNNetVolume;
  Negs: array[0..cNumNeg - 1] of integer;
  Vc, Uw, Uj: array[0..cEmbedDim - 1] of TNeuralFloat;
  GradC: array[0..cEmbedDim - 1] of TNeuralFloat;
  J, D: integer;
  Score, P, Coef: TNeuralFloat;
begin
  // center net: just the center word
  InC.FData[0] := Center;
  // context net: x=0 positive, x=1..K negatives
  InO.FData[0] := Context;
  for J := 0 to cNumNeg - 1 do
  begin
    Negs[J] := SampleNegative(Center, Context);
    InO.FData[1 + J] := Negs[J];
  end;

  NetC.Compute(InC);
  NetO.Compute(InO);
  OutC := NetC.GetLastLayer().Output;   // (1, 1, d)
  OutO := NetO.GetLastLayer().Output;   // (1+K, 1, d)

  for D := 0 to cEmbedDim - 1 do
  begin
    Vc[D] := OutC.FData[D];
    Uw[D] := OutO.FData[D];             // slab 0 = positive context
  end;

  // start each target as a COPY of the corresponding output, then subtract the
  // gradient per slab so that OutputError = output - target = grad.
  TgtC.CopyNoChecks(OutC);
  TgtO.CopyNoChecks(OutO);
  for D := 0 to cEmbedDim - 1 do GradC[D] := 0;

  // positive term: L += -log sigma(v_c . u_w)
  Score := DotN(Vc, Uw);
  P := Sigmoid(Score);
  Result := -Ln(P + 1e-12);
  Coef := P - 1.0;                       // dL/dscore for the positive
  for D := 0 to cEmbedDim - 1 do
  begin
    GradC[D] := GradC[D] + Coef * Uw[D];
    TgtO.FData[D] := OutO.FData[D] - (Coef * Vc[D]);  // grad on positive (slab 0)
  end;

  // negative terms: L += -log sigma(-v_c . u_neg_j)
  for J := 0 to cNumNeg - 1 do
  begin
    for D := 0 to cEmbedDim - 1 do Uj[D] := OutO.FData[(1 + J) * cEmbedDim + D];
    Score := DotN(Vc, Uj);
    P := Sigmoid(Score);
    Result := Result - Ln((1.0 - P) + 1e-12);
    Coef := P;                           // dL/dscore for a negative
    for D := 0 to cEmbedDim - 1 do
    begin
      GradC[D] := GradC[D] + Coef * Uj[D];
      TgtO.FData[(1 + J) * cEmbedDim + D] :=
        OutO.FData[(1 + J) * cEmbedDim + D] - (Coef * Vc[D]);
    end;
  end;

  // center grad (the only slab of the center net)
  for D := 0 to cEmbedDim - 1 do
    TgtC.FData[D] := OutC.FData[D] - GradC[D];

  NetC.Backpropagate(TgtC);
  NetO.Backpropagate(TgtO);
end;

// ---- evaluation: nearest neighbours + analogy ---------------------------

procedure PrintNearest(NN: TNNet; const Query: string);
var
  Qid, I, BestI, Used: integer;
  Q, V: array[0..cEmbedDim - 1] of TNeuralFloat;
  Sim: TNeuralFloat;
  Done: array of boolean;
  BestSim: TNeuralFloat;
begin
  Qid := VocabIndexOf(Query);
  if Qid < 0 then begin WriteLn('  (unknown word: ', Query, ')'); Exit; end;
  ReadVec(NN, Qid, Q);
  SetLength(Done, Length(Vocab));
  for I := 0 to High(Done) do Done[I] := IsStop(I) or (I = Qid);
  Write(Format('  %-10s ->', [Query]));
  for Used := 1 to cTopK do
  begin
    BestI := -1; BestSim := -2;
    for I := 0 to High(Vocab) do
    begin
      if Done[I] then Continue;
      ReadVec(NN, I, V);
      Sim := CosineN(Q, V);
      if Sim > BestSim then begin BestSim := Sim; BestI := I; end;
    end;
    if BestI < 0 then Break;
    Done[BestI] := True;
    Write(Format('  %s(%.2f)', [Vocab[BestI], BestSim]));
  end;
  WriteLn;
end;

// Solves A - B + C ~= ? and prints the top candidates by cosine to the target
// vector, excluding the three query words.
procedure PrintAnalogy(NN: TNNet; const A, B, C: string);
var
  Ia, Ib, Ic, I, Used, BestI: integer;
  Va, Vb, Vc, T, V: array[0..cEmbedDim - 1] of TNeuralFloat;
  D: integer;
  Sim, BestSim: TNeuralFloat;
  Done: array of boolean;
begin
  Ia := VocabIndexOf(A); Ib := VocabIndexOf(B); Ic := VocabIndexOf(C);
  if (Ia < 0) or (Ib < 0) or (Ic < 0) then
  begin WriteLn('  (analogy has an unknown word)'); Exit; end;
  ReadVec(NN, Ia, Va); ReadVec(NN, Ib, Vb); ReadVec(NN, Ic, Vc);
  for D := 0 to cEmbedDim - 1 do T[D] := Va[D] - Vb[D] + Vc[D];
  SetLength(Done, Length(Vocab));
  for I := 0 to High(Done) do
    Done[I] := IsStop(I) or (I = Ia) or (I = Ib) or (I = Ic);
  Write(Format('  %s - %s + %s  ->', [A, B, C]));
  for Used := 1 to 3 do
  begin
    BestI := -1; BestSim := -2;
    for I := 0 to High(Vocab) do
    begin
      if Done[I] then Continue;
      ReadVec(NN, I, V);
      Sim := CosineN(T, V);
      if Sim > BestSim then begin BestSim := Sim; BestI := I; end;
    end;
    if BestI < 0 then Break;
    Done[BestI] := True;
    Write(Format('  %s(%.2f)', [Vocab[BestI], BestSim]));
  end;
  WriteLn;
end;

// ---- main ---------------------------------------------------------------

procedure RunAlgo();
var
  NetC, NetO: TNNet;
  InC, InO, TgtC, TgtO: TNNetVolume;
  Epoch, P, NumPairs: integer;
  Order: TIntArray;
  TmpI, Swap, I: integer;
  TotalLoss, Rate: TNeuralFloat;
begin
  RandSeed := cSeed;
  WriteLn('Word2VecSkipGram: skip-gram word embeddings with negative sampling');
  WriteLn('================================================================');

  BuildVocabAndTokens();
  BuildPositivePairs();
  BuildNegTable();
  NumPairs := Length(PairCenter);

  WriteLn(Format('corpus tokens: %d   vocabulary: %d   positive pairs: %d',
    [Length(Tokens), Length(Vocab), NumPairs]));
  WriteLn(Format('embed_dim: %d   window: +/-%d   negatives K: %d   epochs: %d   lr: %.3f',
    [cEmbedDim, cWindow, cNumNeg, cEpochs, cLearnRate]));
  WriteLn;

  NetC := BuildCenterNet(Length(Vocab));
  NetO := BuildContextNet(Length(Vocab));
  InC  := TNNetVolume.Create(1, 1, 1);
  InO  := TNNetVolume.Create(1 + cNumNeg, 1, 1);
  TgtC := TNNetVolume.Create(1, 1, cEmbedDim);
  TgtO := TNNetVolume.Create(1 + cNumNeg, 1, cEmbedDim);

  // shuffled pass order
  SetLength(Order, NumPairs);
  for I := 0 to NumPairs - 1 do Order[I] := I;

  try
    WriteLn('Training (two embedding matrices: input/center + output/context)...');
    for Epoch := 1 to cEpochs do
    begin
      // linearly decay the learning rate from cLearnRate down to cMinRate
      Rate := cLearnRate + (cMinRate - cLearnRate) * ((Epoch - 1) / (cEpochs - 1));
      NetC.SetLearningRate(Rate, 0.0);
      NetO.SetLearningRate(Rate, 0.0);
      // Fisher-Yates shuffle each epoch
      for I := NumPairs - 1 downto 1 do
      begin
        Swap := Random(I + 1);
        TmpI := Order[I]; Order[I] := Order[Swap]; Order[Swap] := TmpI;
      end;
      TotalLoss := 0;
      for P := 0 to NumPairs - 1 do
        TotalLoss := TotalLoss + TrainStep(NetC, NetO, InC, InO, TgtC, TgtO,
          PairCenter[Order[P]], PairContext[Order[P]]);
      if (Epoch = 1) or (Epoch mod 100 = 0) then
        WriteLn(Format('  epoch %4d   mean SGNS loss = %8.5f',
          [Epoch, TotalLoss / NumPairs]));
    end;
    WriteLn;

    // Nearest neighbours / analogies use the INPUT (center) matrix, NetC.
    WriteLn('Nearest neighbours by cosine similarity (input embedding matrix):');
    PrintNearest(NetC, 'king');
    PrintNearest(NetC, 'queen');
    PrintNearest(NetC, 'man');
    PrintNearest(NetC, 'woman');
    PrintNearest(NetC, 'boy');
    PrintNearest(NetC, 'dog');
    PrintNearest(NetC, 'cat');
    PrintNearest(NetC, 'bread');
    PrintNearest(NetC, 'puppy');
    WriteLn;

    WriteLn('Analogy arithmetic (A - B + C ~= ?), top-3 candidates:');
    PrintAnalogy(NetC, 'king', 'man', 'woman');     // hope: queen
    PrintAnalogy(NetC, 'prince', 'man', 'woman');   // hope: princess
    PrintAnalogy(NetC, 'puppy', 'dog', 'cat');      // hope: kitten
    PrintAnalogy(NetC, 'queen', 'woman', 'man');    // hope: king
    PrintAnalogy(NetC, 'king', 'prince', 'princess'); // hope: queen
    WriteLn;
    WriteLn('(See README.md for an honest note on what lands at this tiny scale.)');
  finally
    TgtO.Free;
    TgtC.Free;
    InO.Free;
    InC.Free;
    NetO.Free;
    NetC.Free;
  end;
end;

var
  Application: record Title: string; end;

begin
  Application.Title := 'Word2VecSkipGram Example';
  RunAlgo();
end.
