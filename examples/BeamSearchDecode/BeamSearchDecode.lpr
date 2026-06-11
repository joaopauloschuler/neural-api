program BeamSearchDecode;
(*
BeamSearchDecode - a decoding-strategy bake-off.

Headline experiment: a tiny char-level next-token model on a SYNTHETIC corpus
deliberately built so that GREEDY argmax dead-ends. After the prompt the
locally-likeliest first token leads into a globally WORSE (lower total
log-prob) continuation, while a slightly-less-likely first token opens a
strongly-deterministic, high-probability long continuation. Beam search keeps
both branches alive and recovers the higher TOTAL log-prob sequence.

We then print:
  * a Greedy vs Beam(B=2,4,8) table (total log-prob + length-penalised score),
  * the alpha=0 (short-biased) vs alpha>0 (Wu et al. 2016) length-penalty
    contrast,
  * a diversity contrast against the existing TopK / TopP stochastic samplers
    (beam = sharp / repetitive, sampling = diverse / noisier).

Trains in-process on CPU in well under a minute. No binaries are committed.

Copyright (C) 2024 Joao Paulo Schwarz Schuler / neural-api contributors.
GNU General Public License v2 or later (see neural-api LICENSE).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralfit,
  neuralthread,
  neuraldatasets,
  neuraldecode;

const
  csContextLen = 8;    // characters of context the model sees
  csVocabSize  = 128;  // char-based vocabulary
  csEOS        = 1;    // chr(1) end-of-sequence

// --- Synthetic corpus -------------------------------------------------------
// EOS = chr(1). The greedy dead-end is built from TWO branches off prompt 'a':
//
//   'a' 'b' ...  : 'b' is locally the SLIGHTLY-most-likely first move, BUT the
//                  'ab' branch is HIGH-ENTROPY: it forks into many different,
//                  roughly-equiprobable third characters (k,l,m,n,o,p). Every
//                  'ab?' continuation therefore costs ~ln(1/6) of log-prob, so
//                  no 'ab...' sequence can reach a high CUMULATIVE log-prob.
//
//   'a' 'c' ...  : 'c' is locally just-below 'b' (so a single argmax skips it),
//                  but 'ac' opens a long, NEAR-DETERMINISTIC tail 'cdefgh' where
//                  every step has prob ~1.0 (~0 log-prob cost). Its cumulative
//                  log-prob is far higher than any high-entropy 'ab?' branch.
//
// Greedy commits to 'b' and then pays the entropy tax forever (dead-end).
// Beam(B>=2) keeps the 'c' branch alive and recovers the globally better tail.
//
// The b:c first-token ratio is kept CLOSE (8:7) so both survive into the beam;
// it is the downstream entropy-vs-determinism gap that decides the winner.
function BuildCorpus: TStringList;
var
  I: integer;
  ForkChars: string;
  C: integer;
begin
  Result := TStringList.Create;
  ForkChars := 'kl'; // 2 equiprobable continuations after 'ab' (genuine entropy)
  // Trap branch: 'ab' forks 2 ways then ends. Equal copies -> p(k|ab)=p(l|ab)=0.5
  // is the UNIQUE cross-entropy optimum (same input, two equiprobable targets),
  // so a converged net cannot collapse it. 24 copies each fork -> 'b' has 48
  // first-token occurrences after 'a'.
  for C := 1 to Length(ForkChars) do
    for I := 1 to 24 do
      Result.Add('ab' + ForkChars[C] + Chr(csEOS));
  // Reward branch: 'ac' -> deterministic 'cdefgh'. 42 copies so the FIRST token
  // 'c' (count 42) stays just under 'b' (count 48) -> a single argmax picks 'b'.
  for I := 1 to 42 do
    Result.Add('acdefgh' + Chr(csEOS));
end;

type
  { TBeamDemo }
  TBeamDemo = class
  private
    FDataset: TStringList;
    FSamples: TStringList;   // (context -> next char) flattened training pairs
    FNN: TNNet;
    procedure BuildSamples;
  public
    constructor Create;
    destructor Destroy; override;
    procedure GetTrainingPair(Idx, ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure Run;
  end;

constructor TBeamDemo.Create;
begin
  FDataset := BuildCorpus;
  FSamples := TStringList.Create;
  BuildSamples;
  FNN := TNNet.Create;
end;

destructor TBeamDemo.Destroy;
begin
  FNN.Free;
  FSamples.Free;
  FDataset.Free;
  inherited Destroy;
end;

// Flatten every (prefix, next-char) pair from every corpus line. The Objects
// slot carries the target character so a training pair is fully described by
// one TStringList entry.
procedure TBeamDemo.BuildSamples;
var
  Row, Pos: integer;
  Line, Prefix: string;
begin
  for Row := 0 to FDataset.Count - 1 do
  begin
    Line := FDataset[Row];
    for Pos := 1 to Length(Line) do
    begin
      Prefix := Copy(Line, 1, Pos - 1);
      // Predict Line[Pos] from Prefix (Prefix may be empty -> start token).
      FSamples.AddObject(Prefix, TObject(PtrInt(Ord(Line[Pos]))));
    end;
  end;
end;

procedure TBeamDemo.GetTrainingPair(Idx, ThreadId: integer;
  pInput, pOutput: TNNetVolume);
var
  Prefix: string;
  Target: integer;
begin
  Idx := Idx mod FSamples.Count;
  Prefix := FSamples[Idx];
  Target := PtrInt(FSamples.Objects[Idx]);
  pInput.ReSize(csContextLen, 1, csVocabSize);
  pInput.OneHotEncodingReversed(Prefix);
  pOutput.ReSize(1, 1, csVocabSize);
  pOutput.SetClassForSoftMax(Target); // one-hot at Target
  pOutput.Tag := Target;              // required by DefaultLossFn / class compare
end;

procedure TBeamDemo.Run;
var
  NFit: TNeuralFit;
  Prompt: string;
  Greedy, Beam2, Beam4, Beam8, Beam4NoPen: TNNetDecodeResult;
  AllBeams: TNNetDecodeResultArray;
  TopK: TNNetSamplerTopK;
  TopP: TNNetSamplerTopP;
  I: integer;
  GenStr: string;
  ProbeIn, ProbeOut: TNNetVolume;

  function Show(const S: string): string;
  var C: integer; R: string;
  begin
    R := '';
    for C := 1 to Length(S) do
      if Ord(S[C]) = csEOS then R := R + '<EOS>'
      else if Ord(S[C]) < 32 then R := R + '?'
      else R := R + S[C];
    Result := R;
  end;

begin
  RandSeed := 424242;
  Prompt := 'a';

  FNN.AddLayer([
    TNNetInput.Create(csContextLen, 1, csVocabSize),
    TNNetFullConnectReLU.Create(48),
    TNNetFullConnectReLU.Create(48),
    TNNetFullConnectLinear.Create(csVocabSize),
    TNNetSoftMax.Create()
  ]);
  FNN.DebugStructure;

  NFit := TNeuralFit.Create;
  try
    NFit.InitialLearningRate := 0.005;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.MaxThreadNum := 4;
    NFit.EnableClassComparison();
    NFit.EnableDefaultLoss();
    NFit.AvgWeightEpochCount := 1;
    NFit.Verbose := False;
    // Many minibatches per epoch (corpus replicated), few epochs -> low
    // per-epoch autosave/validation overhead. Converges in well under a minute.
    WriteLn('Training tiny char model on the synthetic dead-end corpus...');
    NFit.FitLoading(FNN,
      {TrainingCnt=}FSamples.Count * 30,
      {ValidationCnt=}0, {TestCnt=}0,
      {BatchSize=}8, {Epochs=}90,
      @GetTrainingPair, nil, nil);
  finally
    NFit.Free;
  end;

  WriteLn;
  WriteLn('====================================================================');
  WriteLn(' Prompt: "', Prompt, '"   (corpus: "ab{k..p}<EOS>" high-entropy trap');
  WriteLn('                         vs "acdefgh<EOS>" deterministic reward)');
  WriteLn('====================================================================');
  WriteLn;
  WriteLn('Right after "a" the model has SEEN "b" more often than "c", so a');
  WriteLn('single greedy argmax commits to "b" and dead-ends. Beam search keeps');
  WriteLn('the "c" branch alive and recovers the globally higher log-prob tail.');
  WriteLn;

  // Probe: what does the trained model predict right after the prompt "a"?
  ProbeIn := TNNetVolume.Create(FNN.GetFirstLayer.Output);
  ProbeOut := TNNetVolume.Create(FNN.GetLastLayer.Output);
  ProbeIn.OneHotEncodingReversed(Prompt);
  FNN.Compute(ProbeIn, ProbeOut);
  WriteLn(Format('Probe P(next|"a"):  b=%.4f  c=%.4f  <EOS>=%.4f  argmax=%s',
    [ProbeOut.Raw[Ord('b')], ProbeOut.Raw[Ord('c')], ProbeOut.Raw[csEOS],
     Show(Chr(ProbeOut.GetClass()))]));
  ProbeIn.OneHotEncodingReversed('ab');
  FNN.Compute(ProbeIn, ProbeOut);
  WriteLn(Format('Probe P(next|"ab"): k=%.4f l=%.4f m=%.4f (high-entropy fork)',
    [ProbeOut.Raw[Ord('k')], ProbeOut.Raw[Ord('l')], ProbeOut.Raw[Ord('m')]]));
  ProbeIn.OneHotEncodingReversed('ac');
  FNN.Compute(ProbeIn, ProbeOut);
  WriteLn(Format('Probe P(next|"ac"): d=%.4f (deterministic tail; should be ~1)',
    [ProbeOut.Raw[Ord('d')]]));
  WriteLn;
  ProbeIn.Free; ProbeOut.Free;

  Greedy     := DecodeGreedy(FNN, Prompt, 12);
  Beam2      := DecodeBeamSearch(FNN, Prompt, 12, 2, 0.7);
  Beam4      := DecodeBeamSearch(FNN, Prompt, 12, 4, 0.7);
  Beam8      := DecodeBeamSearch(FNN, Prompt, 12, 8, 0.7);

  WriteLn('--- Greedy vs Beam (length penalty alpha=0.7) ----------------------');
  WriteLn(Format('%-12s %-22s %12s %10s', ['strategy', 'output', 'sum_logp', 'score']));
  WriteLn(Format('%-12s %-22s %12.4f %10.4f',
    ['greedy',     Show(Prompt + Greedy.Text), Greedy.SumLogProb, Greedy.Score]));
  WriteLn(Format('%-12s %-22s %12.4f %10.4f',
    ['beam B=2',   Show(Prompt + Beam2.Text),  Beam2.SumLogProb,  Beam2.Score]));
  WriteLn(Format('%-12s %-22s %12.4f %10.4f',
    ['beam B=4',   Show(Prompt + Beam4.Text),  Beam4.SumLogProb,  Beam4.Score]));
  WriteLn(Format('%-12s %-22s %12.4f %10.4f',
    ['beam B=8',   Show(Prompt + Beam8.Text),  Beam8.SumLogProb,  Beam8.Score]));
  WriteLn;

  if Beam4.SumLogProb > Greedy.SumLogProb + 1e-4 then
    WriteLn('RESULT: beam recovered a HIGHER total log-prob sequence than greedy.')
  else
    WriteLn('NOTE: beam did not beat greedy on this run (corpus/training too easy).');
  WriteLn;

  // alpha contrast: raw sum (short-biased) vs Wu et al. penalised.
  Beam4NoPen := DecodeBeamSearch(FNN, Prompt, 12, 4, 0.0);
  WriteLn('--- Length-penalty contrast (B=4) ----------------------------------');
  WriteLn(Format('%-14s %-22s %12s %10s', ['alpha', 'output', 'sum_logp', 'score']));
  WriteLn(Format('%-14s %-22s %12.4f %10.4f',
    ['alpha=0 (raw)', Show(Prompt + Beam4NoPen.Text), Beam4NoPen.SumLogProb, Beam4NoPen.Score]));
  WriteLn(Format('%-14s %-22s %12.4f %10.4f',
    ['alpha=0.7',     Show(Prompt + Beam4.Text), Beam4.SumLogProb, Beam4.Score]));
  WriteLn('(alpha=0 ranks on the raw sum and favours SHORTER beams; alpha>0');
  WriteLn(' divides by ((5+L)/6)^alpha, lifting longer, content-bearing tails.)');
  WriteLn;

  // Show the final ranked beam (finished pool + survivors).
  AllBeams := DecodeBeamSearchAll(FNN, Prompt, 12, 4, 0.7);
  WriteLn('--- Final ranked beam (B=4, alpha=0.7) -----------------------------');
  for I := 0 to Min(High(AllBeams), 3) do
    WriteLn(Format('  #%d %-20s sum_logp=%9.4f score=%9.4f %s',
      [I + 1, Show(Prompt + AllBeams[I].Text), AllBeams[I].SumLogProb,
       AllBeams[I].Score, BoolToStr(AllBeams[I].Finished, '[EOS]', '')]));
  WriteLn;

  // Diversity contrast vs the stochastic per-token samplers.
  WriteLn('--- Diversity contrast vs stochastic samplers ----------------------');
  WriteLn('Beam is sharp/deterministic: repeating it gives the SAME sequence.');
  WriteLn('TopK / TopP re-sample per token and wander. 6 draws each:');
  RandSeed := 1234;
  TopK := TNNetSamplerTopK.Create(3);
  TopP := TNNetSamplerTopP.Create(0.9);
  try
    Write('  beam B=4 : ');
    for I := 1 to 6 do Write('"', Show(Prompt + DecodeBeamSearch(FNN, Prompt, 12, 4, 0.7).Text), '" ');
    WriteLn;
    Write('  TopK(3)  : ');
    for I := 1 to 6 do
    begin
      GenStr := GenerateStringFromChars(FNN, Prompt, TopK);
      Write('"', Show(GenStr), '" ');
    end;
    WriteLn;
    Write('  TopP(.9) : ');
    for I := 1 to 6 do
    begin
      GenStr := GenerateStringFromChars(FNN, Prompt, TopP);
      Write('"', Show(GenStr), '" ');
    end;
    WriteLn;
  finally
    TopK.Free;
    TopP.Free;
  end;
  WriteLn;
  WriteLn('Done.');
end;

var
  Demo: TBeamDemo;
begin
  Demo := TBeamDemo.Create;
  try
    Demo.Run;
  finally
    Demo.Free;
  end;
end.
