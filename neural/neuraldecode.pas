unit neuraldecode;

(*
neuraldecode
Deterministic, sequence-level decoding strategies for char-level next-token
models built with neural-api. The flagship routine is DecodeBeamSearch, the
missing counterpart to the per-token stochastic TNNetSamplerBase family
(Greedy / TopK / TopP).

Beam search keeps the B highest CUMULATIVE-log-probability partial sequences,
expands each by every candidate next token, then re-prunes to the top B. Unlike
a per-token argmax, it can RECOVER from a locally-greedy first-token mistake
that a single argmax would lock in forever. Because it scores whole sequences
rather than one token, it does NOT fit the GetToken(Origin) interface and is a
standalone routine rather than a TNNetSamplerBeam subclass.

Honest v1 scope notes ("what did NOT fit"):
  (a) All scoring is in LOG space and log-probs are SUMMED (never multiply raw
      probabilities -> underflow). Model softmax outputs are converted to
      log-probs with a numerically-safe log of a clamped probability.
  (b) Wu et al. 2016 length penalty:  score = sum_logp / ((5+L)/6)^alpha .
      alpha = 0 reproduces the raw, short-sequence-biased sum; alpha > 0 lifts
      longer beams. See LengthPenaltyDenominator.
  (c) v1 RE-ENCODES each candidate prefix on every step (O(L^2) total forward
      passes). The KV-cache incremental-decode plumbing now lives in
      TNNetStreamingDecoder below; wiring beam search onto it is the remaining
      follow-up.
  (d) A beam that emits the EOS token is moved to a finished pool and ranked
      there against still-growing beams; growth stops once enough finished
      beams dominate or MaxLen is reached.

The forward-pass / encoding convention matches GenerateStringFromChars in
neuraldatasets: the prompt+generated text is one-hot encoded right-aligned via
OneHotEncodingReversed, NN.Compute produces a single SoftMax distribution over
the vocabulary, and EOS is token 1 (chr(1)), terminating like the
"NextTokenInt < 2" rule used elsewhere in the codebase.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralvolume, neuralnetwork;

type
  // A scored decode candidate: the generated text (excluding the prompt) and
  // its cumulative log-probability.
  TNNetDecodeResult = record
    Text: string;        // generated continuation (prompt NOT included)
    SumLogProb: TNeuralFloat; // sum of per-step log-probabilities
    Score: TNeuralFloat; // length-penalised score actually ranked on
    Finished: boolean;   // True if the beam emitted EOS
  end;

  TNNetDecodeResultArray = array of TNNetDecodeResult;

  // TNNetStreamingDecoder: a reusable incremental-decode "streaming session"
  // over a causal next-token net, replacing the hand-rolled step-net plumbing
  // every streaming example repeats (build a short-width twin, CopyWeights,
  // scan layers by class, switch caches/state into incremental mode, set RoPE
  // offsets before every forward).
  //
  // OWNERSHIP. The session does NOT build and does NOT own the net. The
  // caller typically builds the SAME architecture at a short input width
  // (1 for plain token-at-a-time decode, K+1 for a speculative verify
  // window), calls ShortNet.CopyWeights(TrainedNet) - every parameter shape
  // in a streamable model is sequence-length independent, so the layer-by-
  // layer copy is exact - and hands the twin to Create. Destroy switches the
  // collected layers back out of incremental mode but never frees the net.
  //
  // WHAT IS COLLECTED. Create scans pNet.Layers once and collects
  //   - every TNNetScaledDotProductAttention: BeginIncrementalDecode(
  //     pMaxCacheLen) switches it onto the KV-cache path (pMaxCacheLen must
  //     cover the worst transient load: committed context + one whole
  //     window);
  //   - every TNNetDiagonalSSM: BeginIncrementalDecode() switches it onto the
  //     O(1)-per-step persisted-state path (no preallocation budget - the
  //     entire past is one Depth-long state vector h);
  //   - every TNNetRotaryEmbedding: kept so PositionOffset can be advanced
  //     before each forward (below).
  // A net may contain any mix (attention-only, SSM-only, hybrid); the counts
  // are exposed for diagnostics.
  //
  // RoPE EXACTNESS CONTRACT. A streamed window has SizeX = window width, so a
  // rope layer would otherwise always rotate it starting at position 0.
  // StepForward therefore sets PositionOffset := AbsPos on EVERY collected
  // rope layer before EVERY pNet.Compute, where AbsPos is the ABSOLUTE
  // position of the FIRST token in the window (the running committed length).
  // This single rule makes width-1 decode steps and width-K speculative
  // verify windows rotate exactly as the full forward would; skip it and the
  // cached path silently diverges from the full forward.
  //
  // WHICH NORM LAYERS ARE STREAMABLE. TNNetDyT is per-element (tanh of a
  // scaled activation, no cross-token statistics), so cached/streamed decode
  // is exact. TNNetLayerNorm and TNNetRMSNorm normalize over the WHOLE sample
  // INCLUDING the sequence axis: a width-1 window sees different statistics
  // than the same token inside a full-width forward, breaking full-vs-
  // incremental exactness. Build streaming models with TNNetDyT (e.g.
  // AddTransformerEncoderBlock(..., NormClass=TNNetDyT)).
  //
  // TYPICAL LOOP (greedy decode):
  //   Session := TNNetStreamingDecoder.Create(ShortNet, ContextLen + Width);
  //   Session.Reset();
  //   for t := 0 to PromptLen - 2 do            // prefill
  //     begin InV.FData[0] := Toks[t]; Session.StepForward(InV, t); end;
  //   while generating do
  //   begin
  //     InV.FData[0] := Toks[Pos - 1];
  //     Session.StepForward(InV, Pos - 1);
  //     Toks[Pos] := Session.Output().GetClassOnPixel(0, 0); Inc(Pos);
  //   end;
  // Speculative decoding additionally calls TruncateTo(CommittedLen) to roll
  // the KV caches back past rejected draft tokens (pad/draft K/V appended by
  // a verify window is discarded the same way). SSM state cannot be rolled
  // back (it is a folded summary, not a list), so TruncateTo only touches the
  // attention caches - speculative decoding is an attention-family feature.
  // Coded by Claude (AI).
  TNNetStreamingDecoder = class(TObject)
  private
    FNet: TNNet;
    FSDPAs: array of TNNetScaledDotProductAttention;
    FSSMs: array of TNNetDiagonalSSM;
    FRopes: array of TNNetRotaryEmbedding;
    function GetSDPACount(): integer;
    function GetSSMCount(): integer;
    function GetRopeCount(): integer;
  public
    constructor Create(pNet: TNNet; pMaxCacheLen: integer);
    destructor Destroy(); override;
    // Start a fresh sequence: ResetCache on every attention layer, ResetState
    // on every SSM layer. Call before the first prefill token of a sequence.
    procedure Reset();
    // One streamed forward of the window in InV. AbsPos is the absolute
    // position of the FIRST token in the window; it is written to every rope
    // layer's PositionOffset before pNet.Compute (see the exactness contract
    // above).
    procedure StepForward(InV: TNNetVolume; AbsPos: integer);
    // Speculative-decode rollback: TruncateCache(CommittedLen) on every
    // attention layer, discarding the K/V of rejected/pad tokens. No-op when
    // the net has no attention layers.
    procedure TruncateTo(CommittedLen: integer);
    // Convenience: the net's last layer output (e.g. the softmax row(s) of
    // the window just computed).
    function Output(): TNNetVolume;
    property Net: TNNet read FNet;
    property SDPACount: integer read GetSDPACount;
    property SSMCount: integer read GetSSMCount;
    property RopeCount: integer read GetRopeCount;
  end;

const
  csDecodeEOSToken = 1; // chr(1), the codebase end-of-sequence marker.

// Wu et al. 2016 length-penalty denominator ((5+L)/6)^alpha. With alpha=0 this
// is exactly 1.0 (no penalty -> raw sum-log-prob ranking, short-biased).
function LengthPenaltyDenominator(L: integer; Alpha: TNeuralFloat): TNeuralFloat;

// Numerically-safe natural log of a probability (clamps tiny / zero probs so a
// dead-but-not-impossible token never produces -Inf and poisons the sum).
function SafeLogProb(P: TNeuralFloat): TNeuralFloat;

// Deterministic greedy argmax decode, in the same forward-pass / encoding
// convention as DecodeBeamSearch. Returned as a single-element result so its
// SumLogProb is directly comparable to a beam result.
function DecodeGreedy(NN: TNNet; const Prompt: string;
  MaxLen: integer): TNNetDecodeResult;

// Beam search. Keeps BeamWidth partial sequences ranked by length-penalised
// cumulative log-prob. Returns the single best (highest Score) result.
//   MaxLen        : maximum number of generated tokens (excludes the prompt).
//   BeamWidth     : B; B=1 with LengthPenalty=0 is exactly greedy argmax.
//   LengthPenalty : alpha in the Wu et al. formula.
function DecodeBeamSearch(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer;
  LengthPenalty: TNeuralFloat): TNNetDecodeResult;

// Full beam search returning the entire final ranked beam (finished + best
// surviving), so callers can inspect the runners-up. Sorted best-first.
function DecodeBeamSearchAll(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer;
  LengthPenalty: TNeuralFloat): TNNetDecodeResultArray;

implementation

uses
  Math;

function LengthPenaltyDenominator(L: integer; Alpha: TNeuralFloat): TNeuralFloat;
begin
  if Alpha = 0 then
    Result := 1.0
  else
    Result := Power((5.0 + L) / 6.0, Alpha);
end;

function SafeLogProb(P: TNeuralFloat): TNeuralFloat;
const
  csTinyProb = 1e-30;
begin
  if P < csTinyProb then P := csTinyProb;
  Result := Ln(P);
end;

{ TNNetStreamingDecoder }

constructor TNNetStreamingDecoder.Create(pNet: TNNet; pMaxCacheLen: integer);
var
  i, n: integer;
  Layer: TNNetLayer;
begin
  inherited Create();
  FNet := pNet;
  SetLength(FSDPAs, 0);
  SetLength(FSSMs, 0);
  SetLength(FRopes, 0);
  // One class-based scan collects every streamable state holder; the scan is
  // builder-agnostic, so any mix of attention/SSM/hybrid models works.
  for i := 0 to FNet.Layers.Count - 1 do
  begin
    Layer := FNet.Layers[i];
    if Layer is TNNetScaledDotProductAttention then
    begin
      n := Length(FSDPAs);
      SetLength(FSDPAs, n + 1);
      FSDPAs[n] := TNNetScaledDotProductAttention(Layer);
      FSDPAs[n].BeginIncrementalDecode(pMaxCacheLen);
    end;
    if Layer is TNNetDiagonalSSM then
    begin
      n := Length(FSSMs);
      SetLength(FSSMs, n + 1);
      FSSMs[n] := TNNetDiagonalSSM(Layer);
      FSSMs[n].BeginIncrementalDecode();
    end;
    if Layer is TNNetRotaryEmbedding then
    begin
      n := Length(FRopes);
      SetLength(FRopes, n + 1);
      FRopes[n] := TNNetRotaryEmbedding(Layer);
    end;
  end;
end;

destructor TNNetStreamingDecoder.Destroy();
var
  i: integer;
begin
  // Switch the collected layers back onto the normal full-sequence path and
  // restore the default rope offset; the net itself is NOT owned/freed.
  for i := 0 to High(FSDPAs) do FSDPAs[i].EndIncrementalDecode();
  for i := 0 to High(FSSMs) do FSSMs[i].EndIncrementalDecode();
  for i := 0 to High(FRopes) do FRopes[i].PositionOffset := 0;
  SetLength(FSDPAs, 0);
  SetLength(FSSMs, 0);
  SetLength(FRopes, 0);
  inherited Destroy();
end;

procedure TNNetStreamingDecoder.Reset();
var
  i: integer;
begin
  for i := 0 to High(FSDPAs) do FSDPAs[i].ResetCache();
  for i := 0 to High(FSSMs) do FSSMs[i].ResetState();
end;

procedure TNNetStreamingDecoder.StepForward(InV: TNNetVolume; AbsPos: integer);
var
  i: integer;
begin
  // The exactness contract: every rope layer is shifted to the window's
  // ABSOLUTE start position before every forward, so a width-1 step and a
  // width-K speculative verify window both rotate exactly like the full
  // forward.
  for i := 0 to High(FRopes) do FRopes[i].PositionOffset := AbsPos;
  FNet.Compute(InV);
end;

procedure TNNetStreamingDecoder.TruncateTo(CommittedLen: integer);
var
  i: integer;
begin
  for i := 0 to High(FSDPAs) do FSDPAs[i].TruncateCache(CommittedLen);
end;

function TNNetStreamingDecoder.Output(): TNNetVolume;
begin
  Result := FNet.GetLastLayer().Output;
end;

function TNNetStreamingDecoder.GetSDPACount(): integer;
begin
  Result := Length(FSDPAs);
end;

function TNNetStreamingDecoder.GetSSMCount(): integer;
begin
  Result := Length(FSSMs);
end;

function TNNetStreamingDecoder.GetRopeCount(): integer;
begin
  Result := Length(FRopes);
end;

// Forward pass: encode (Prompt + Generated) and return the next-token
// distribution as log-probabilities in LogProbs (length = vocab size).
// The output volume is treated as a probability distribution (SoftMax head);
// if it does not sum to ~1 it is re-normalised defensively before the log.
procedure NextLogProbs(NN: TNNet; const Context: string;
  InputVolume, OutputVolume: TNNetVolume; var LogProbs: array of TNeuralFloat);
var
  I: integer;
  Total: TNeuralFloat;
begin
  InputVolume.OneHotEncodingReversed(Context);
  NN.Compute(InputVolume, OutputVolume);
  Total := OutputVolume.GetSum();
  if (Total <= 0) then Total := 1.0;
  for I := 0 to OutputVolume.Size - 1 do
    LogProbs[I] := SafeLogProb(OutputVolume.Raw[I] / Total);
end;

function DecodeGreedy(NN: TNNet; const Prompt: string;
  MaxLen: integer): TNNetDecodeResult;
var
  InputVolume, OutputVolume: TNNetVolume;
  LogProbs: array of TNeuralFloat;
  VocabSize, Step, Best, I: integer;
  Context: string;
begin
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  VocabSize := OutputVolume.Size;
  SetLength(LogProbs, VocabSize);
  Result.Text := '';
  Result.SumLogProb := 0;
  Result.Finished := False;
  Context := Prompt;
  try
    for Step := 1 to MaxLen do
    begin
      NextLogProbs(NN, Context, InputVolume, OutputVolume, LogProbs);
      Best := 0;
      for I := 1 to VocabSize - 1 do
        if LogProbs[I] > LogProbs[Best] then Best := I;
      Result.SumLogProb := Result.SumLogProb + LogProbs[Best];
      if Best = csDecodeEOSToken then
      begin
        Result.Finished := True;
        Break;
      end;
      Result.Text := Result.Text + Chr(Best);
      Context := Context + Chr(Best);
    end;
  finally
    InputVolume.Free;
    OutputVolume.Free;
  end;
  Result.Score := Result.SumLogProb /
    LengthPenaltyDenominator(Length(Result.Text), 0);
end;

type
  TBeam = record
    Text: string;
    SumLogProb: TNeuralFloat;
    Score: TNeuralFloat;
    Finished: boolean;
  end;
  TBeamArray = array of TBeam;

// Insertion sort beams in DESCENDING Score (small arrays, B is tiny).
procedure SortBeamsByScore(var Beams: TBeamArray);
var
  I, J: integer;
  Tmp: TBeam;
begin
  for I := 1 to High(Beams) do
  begin
    Tmp := Beams[I];
    J := I - 1;
    while (J >= 0) and (Beams[J].Score < Tmp.Score) do
    begin
      Beams[J + 1] := Beams[J];
      Dec(J);
    end;
    Beams[J + 1] := Tmp;
  end;
end;

function DecodeBeamSearchAll(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer;
  LengthPenalty: TNeuralFloat): TNNetDecodeResultArray;
var
  InputVolume, OutputVolume: TNNetVolume;
  LogProbs: array of TNeuralFloat;
  VocabSize, Step, I, T, B: integer;
  Live: TBeamArray;      // still-growing beams
  Finished: TBeamArray;  // beams that emitted EOS
  Cand: TBeamArray;      // expansion candidates for this step
  NewBeam: TBeam;
  CutScore: TNeuralFloat;
  AllDominated: boolean;
begin
  if BeamWidth < 1 then BeamWidth := 1;
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  VocabSize := OutputVolume.Size;
  SetLength(LogProbs, VocabSize);
  try
    SetLength(Live, 1);
    Live[0].Text := '';
    Live[0].SumLogProb := 0;
    Live[0].Score := 0;
    Live[0].Finished := False;
    SetLength(Finished, 0);

    for Step := 1 to MaxLen do
    begin
      if Length(Live) = 0 then Break;

      // (d) Early stop: if we already have BeamWidth finished beams and the
      // best live beam cannot beat the worst kept finished one, stop growing.
      if Length(Finished) >= BeamWidth then
      begin
        SortBeamsByScore(Finished);
        CutScore := Finished[BeamWidth - 1].Score;
        AllDominated := True;
        for I := 0 to High(Live) do
          // A growing beam's sum-log-prob only decreases; its best possible
          // future score is bounded above by its current score (adding more
          // negative log-probs and a >=1 penalty denominator can only lower
          // it once alpha>=0 and tokens are < prob 1). Use current score as
          // an admissible upper bound for the prune decision.
          if Live[I].Score > CutScore then AllDominated := False;
        if AllDominated then Break;
      end;

      // Expand every live beam by every vocabulary token.
      SetLength(Cand, 0);
      for B := 0 to High(Live) do
      begin
        NextLogProbs(NN, Prompt + Live[B].Text,
          InputVolume, OutputVolume, LogProbs);
        for T := 0 to VocabSize - 1 do
        begin
          NewBeam.SumLogProb := Live[B].SumLogProb + LogProbs[T];
          if T = csDecodeEOSToken then
          begin
            NewBeam.Text := Live[B].Text;
            NewBeam.Finished := True;
            NewBeam.Score := NewBeam.SumLogProb /
              LengthPenaltyDenominator(Length(NewBeam.Text), LengthPenalty);
            SetLength(Finished, Length(Finished) + 1);
            Finished[High(Finished)] := NewBeam;
          end
          else
          begin
            NewBeam.Text := Live[B].Text + Chr(T);
            NewBeam.Finished := False;
            NewBeam.Score := NewBeam.SumLogProb /
              LengthPenaltyDenominator(Length(NewBeam.Text), LengthPenalty);
            SetLength(Cand, Length(Cand) + 1);
            Cand[High(Cand)] := NewBeam;
          end;
        end;
      end;

      // Re-prune the survivors to the top BeamWidth by length-penalised score.
      SortBeamsByScore(Cand);
      if Length(Cand) > BeamWidth then SetLength(Cand, BeamWidth);
      Live := Copy(Cand, 0, Length(Cand));
    end;

    // Merge any remaining live beams into the finished pool (MaxLen reached).
    for B := 0 to High(Live) do
    begin
      SetLength(Finished, Length(Finished) + 1);
      Finished[High(Finished)] := Live[B];
    end;

    SortBeamsByScore(Finished);
    SetLength(Result, Length(Finished));
    for I := 0 to High(Finished) do
    begin
      Result[I].Text := Finished[I].Text;
      Result[I].SumLogProb := Finished[I].SumLogProb;
      Result[I].Score := Finished[I].Score;
      Result[I].Finished := Finished[I].Finished;
    end;
  finally
    InputVolume.Free;
    OutputVolume.Free;
  end;
end;

function DecodeBeamSearch(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer;
  LengthPenalty: TNeuralFloat): TNNetDecodeResult;
var
  All: TNNetDecodeResultArray;
begin
  All := DecodeBeamSearchAll(NN, Prompt, MaxLen, BeamWidth, LengthPenalty);
  if Length(All) > 0 then
    Result := All[0]
  else
  begin
    Result.Text := '';
    Result.SumLogProb := 0;
    Result.Score := 0;
    Result.Finished := False;
  end;
end;

end.
