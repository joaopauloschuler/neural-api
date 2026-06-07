// Multi-Token Prediction (MTP) example
//
// Demonstrates that adding AUXILIARY future-token prediction heads (Gloeckle et
// al. 2024; DeepSeek-V3) speeds up convergence of the PRIMARY next-token (t+1)
// head, compared to a trunk-matched baseline that predicts only t+1.
//
// THE TOY RULE (deterministic, multi-step predictable). Each sequence is an
// arithmetic progression over a small vocabulary:
//     token[t] = (start + t*step) mod V
// with a random start and a random step drawn from a small set. After seeing the
// first couple of tokens the rule is fully determined, so a CAUSAL trunk at
// position t can predict not only t+1 but t+2, t+3, ... This is exactly the
// regime where MTP's denser training signal helps: every position supervises
// NumFuture future tokens instead of one.
//
// TWO ARMS (identical trunk, identical training stream):
//   * MTP arm      : shared trunk -> TNNet.AddMultiTokenPrediction(NumFuture=3)
//                    -> 3 parallel per-token softmax heads (t+1, t+2, t+3),
//                    DeepConcat'd into one (SeqLen,1,3*V) output. Trained against
//                    a (SeqLen,1,3*V) target whose slab h is the one-hot of the
//                    token at t+1+h. The EXTRA heads only add training signal --
//                    at inference we read just the t+1 slab (the other heads are
//                    reusable for self-speculative decoding, see the README).
//   * Baseline arm : same trunk -> a single next-token head (NumFuture=1), i.e.
//                    ordinary t+1-only language-model training.
//
// HEADLINE: at an EARLY training checkpoint (a fixed, deliberately short step
// budget) the MTP arm reaches HIGHER next-token (t+1) accuracy than the t+1-only
// baseline -- the auxiliary future losses accelerate the primary head. Because a
// single run is noisy, the comparison is averaged over several independent seeds;
// the MTP arm wins on the mean and in most individual seeds. (Train both arms to
// convergence and this easy rule saturates, erasing the gap -- the win is about
// convergence SPEED.) Pure CPU, tiny dims, finishes in well under five minutes on
// two cores.
//
// Coded by Claude (AI).
program MultiTokenPrediction;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils,
  neuralnetwork, neuralvolume;

const
  cVocab      = 11;    // vocabulary size (arithmetic progression modulus)
  cSeqLen     = 12;    // sequence length
  cDModel     = 24;    // trunk residual-stream width
  cHeads      = 2;     // attention heads in the causal trunk block
  cDFF        = 48;    // feed-forward width in the trunk block
  cNumFuture  = 3;     // MTP arm forecasts t+1, t+2, t+3
  cTrainSteps = 700;  // training steps per arm (kept short so the faster
                       // convergence of the MTP arm is visible -- with a long
                       // budget both arms eventually saturate this easy rule)
  cEvalSeqs   = 600;   // held-out sequences for the accuracy report
  // The step of the arithmetic progression is drawn from this set (each value is
  // coprime-ish with cVocab so the progressions are varied).
  cSteps: array[0..3] of integer = (1, 2, 3, 4);

// Generate one arithmetic-progression sequence into Tokens[0..cSeqLen-1].
procedure MakeTokens(out Tokens: array of integer);
var
  start, step, t: integer;
begin
  start := Random(cVocab);
  step  := cSteps[Random(Length(cSteps))];
  for t := 0 to cSeqLen - 1 do
    Tokens[t] := (start + t * step) mod cVocab;
end;

// Build the (SeqLen,1,cVocab) one-hot INPUT from a token sequence.
procedure FillInput(Input: TNNetVolume; const Tokens: array of integer);
var t: integer;
begin
  Input.Fill(0);
  for t := 0 to cSeqLen - 1 do
    Input[t, 0, Tokens[t]] := 1.0;
end;

// Build the (SeqLen,1,NumFuture*cVocab) MTP target: at position t, slab h holds
// the one-hot of token[t+1+h]. Positions whose future token falls past the end
// of the sequence are left zero (no supervision there).
procedure FillTarget(Desired: TNNetVolume; const Tokens: array of integer;
  NumFuture: integer);
var t, h, fut: integer;
begin
  Desired.Fill(0);
  for t := 0 to cSeqLen - 1 do
    for h := 0 to NumFuture - 1 do
    begin
      fut := t + 1 + h;
      if fut <= cSeqLen - 1 then
        Desired[t, 0, h * cVocab + Tokens[fut]] := 1.0;
    end;
end;

// Shared causal trunk: one-hot -> per-token projection to d_model -> absolute
// positional embedding -> one CAUSAL transformer block. (A FullConnect head is
// deliberately avoided so the per-token sequence axis is preserved end to end.)
procedure AddTrunk(NN: TNNet);
begin
  NN.AddLayer([
    TNNetInput.Create(cSeqLen, 1, cVocab),
    TNNetPointwiseConvLinear.Create(cDModel),
    TNNetAddPositionalEmbedding.Create(10000)
  ]);
  NN.AddTransformerEncoderBlock({Heads=}cHeads, {d_ff=}cDFF,
    {PreNorm=}true, {CausalMask=}true, {UseRoPE=}false, {NormClass=}nil);
end;

function BuildNet(NumFuture: integer): TNNet;
begin
  Result := TNNet.Create();
  AddTrunk(Result);
  // NumFuture=1 -> a single ordinary next-token head; NumFuture>1 -> MTP.
  Result.AddMultiTokenPrediction(NumFuture, cVocab);
end;

procedure Train(NN: TNNet; Steps: integer; LR: TNeuralFloat; NumFuture: integer);
var
  Input, Desired: TNNetVolume;
  Tokens: array[0..cSeqLen - 1] of integer;
  i: integer;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, cVocab);
  Desired := TNNetVolume.Create(cSeqLen, 1, NumFuture * cVocab);
  NN.SetLearningRate(LR, 0.9);
  try
    for i := 0 to Steps - 1 do
    begin
      MakeTokens(Tokens);
      FillInput(Input, Tokens);
      FillTarget(Desired, Tokens, NumFuture);
      NN.Compute(Input);
      NN.Backpropagate(Desired);
    end;
  finally
    Input.Free; Desired.Free;
  end;
end;

// Next-token (t+1) accuracy: argmax of the t+1 slab vs the true next token,
// averaged over all supervised positions of N held-out sequences. The t+1 slab
// is depth [0 .. cVocab-1] for BOTH arms (it is head h=0).
function NextTokenAccuracy(NN: TNNet; N: integer): TNeuralFloat;
var
  Input: TNNetVolume;
  Tokens: array[0..cSeqLen - 1 ] of integer;
  seq, t, v, bestV, correct, total: integer;
  bestP, p: TNeuralFloat;
begin
  Input := TNNetVolume.Create(cSeqLen, 1, cVocab);
  correct := 0; total := 0;
  try
    for seq := 0 to N - 1 do
    begin
      MakeTokens(Tokens);
      FillInput(Input, Tokens);
      NN.Compute(Input);
      // Every position t with a valid t+1 target.
      for t := 0 to cSeqLen - 2 do
      begin
        bestV := 0; bestP := -1;
        for v := 0 to cVocab - 1 do
        begin
          p := NN.GetLastLayer.Output[t, 0, v];   // t+1 slab
          if p > bestP then begin bestP := p; bestV := v; end;
        end;
        if bestV = Tokens[t + 1] then Inc(correct);
        Inc(total);
      end;
    end;
    Result := correct / total;
  finally
    Input.Free;
  end;
end;

// Run one trial (one weight-init / data seed): build both arms, train them on the
// SAME data stream for Steps steps, and return the EARLY-checkpoint t+1 accuracy
// of each. Both arms share the trunk architecture and the training RNG stream, so
// any gap is attributable to the auxiliary future heads, not luck.
procedure RunTrial(Seed, Steps: integer; out MtpAcc, BaseAcc: TNeuralFloat);
var
  MtpNet, BaseNet: TNNet;
begin
  RandSeed := Seed;
  MtpNet  := BuildNet(cNumFuture);   // predicts t+1, t+2, t+3
  RandSeed := Seed;
  BaseNet := BuildNet(1);            // predicts t+1 only (identical trunk init)

  RandSeed := Seed * 31 + 1;
  Train(MtpNet, Steps, 0.01, cNumFuture);
  RandSeed := Seed * 31 + 1;         // identical training stream for both arms
  Train(BaseNet, Steps, 0.01, 1);

  RandSeed := Seed * 7 + 5;
  MtpAcc := NextTokenAccuracy(MtpNet, cEvalSeqs);
  RandSeed := Seed * 7 + 5;          // identical eval stream for both arms
  BaseAcc := NextTokenAccuracy(BaseNet, cEvalSeqs);

  MtpNet.Free; BaseNet.Free;
end;

const
  cTrials: array[0..7] of integer =
    (11, 23, 37, 53, 71, 89, 101, 131);  // independent seeds

var
  trial: integer;
  mtpAcc, baseAcc, mtpSum, baseSum: TNeuralFloat;
  mtpWins: integer;
begin
  WriteLn('=== Multi-Token Prediction: auxiliary future heads speed t+1 convergence ===');
  WriteLn('vocab=', cVocab, '  seq_len=', cSeqLen, '  d_model=', cDModel,
          '  heads=', cHeads, '  NumFuture(MTP)=', cNumFuture);
  WriteLn('rule: token[t] = (start + t*step) mod ', cVocab,
          '   (step in {1,2,3,4})');
  WriteLn('Comparing the t+1 accuracy at an EARLY checkpoint (', cTrainSteps,
          ' steps) over ', Length(cTrials), ' seeds.');
  WriteLn('Both arms share the SAME trunk init and the SAME data/eval streams per ',
          'seed,');
  WriteLn('so the only difference is the MTP arm''s extra t+2/t+3 future heads.');
  WriteLn;

  mtpSum := 0; baseSum := 0; mtpWins := 0;
  WriteLn(' seed | MTP t+1 acc | baseline t+1 acc | delta');
  WriteLn(' -----+-------------+------------------+-------');
  for trial := 0 to Length(cTrials) - 1 do
  begin
    RunTrial(cTrials[trial], cTrainSteps, mtpAcc, baseAcc);
    mtpSum := mtpSum + mtpAcc; baseSum := baseSum + baseAcc;
    if mtpAcc > baseAcc then Inc(mtpWins);
    WriteLn(Format(' %4d |   %6.1f%%   |     %6.1f%%      | %6.1f',
      [cTrials[trial], mtpAcc * 100, baseAcc * 100, (mtpAcc - baseAcc) * 100]));
  end;
  mtpSum := mtpSum / Length(cTrials);
  baseSum := baseSum / Length(cTrials);
  WriteLn(' -----+-------------+------------------+-------');
  WriteLn(Format(' mean |   %6.1f%%   |     %6.1f%%      | %6.1f',
    [mtpSum * 100, baseSum * 100, (mtpSum - baseSum) * 100]));
  WriteLn;
  WriteLn('MTP arm ahead in ', mtpWins, ' / ', Length(cTrials), ' seeds.');
  WriteLn;

  if mtpSum > baseSum then
    WriteLn('OK: averaged over seeds, the auxiliary future-token losses give the ',
            't+1 head a head start (faster convergence).')
  else
    WriteLn('WARNING: MTP did not beat the t+1-only baseline on average this run.');
end.
