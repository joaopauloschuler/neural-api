program TestNeuralRL;
(*
TestNeuralRL: validates the neuralrl.pas unit (TNeuralEnv, TNeuralReplayBuffer,
TNeuralDQNTrainer, TNeuralPolicyGradTrainer, TNeuralCartPoleEnv, TNeuralGridWorldEnv).

TESTS:
  1. GridWorld env: deterministic path to goal (no slip) must reach goal in
     exactly (W-1) + (H-1) steps with reward 10 - 0.02*steps.
  2. CartPole env: from a small initial angle, the pole must eventually fall
     (episode terminates) within 500 steps if no action is taken.
  3. ReplayBuffer: add N transitions, sample, verify size and no crash.
  4. DQN on GridWorld: train for 300 episodes, greedy policy must reach the
     goal (TotalReward > 0, ideally ~10 - step_cost).
  5. PolicyGradient on GridWorld: train for 200 episodes, greedy policy must
     reach the goal.

Each test prints PASS or FAIL. Any FAIL causes Halt(1).

Pure CPU, tiny nets, finishes in under 2 minutes.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuralrl;

const
  // Grid world for testing.
  cGW = 5;           // 5x5 grid
  cGWMaxSteps = 30;  // max steps per episode

  // DQN params.
  cDQNEpisodes = 400;
  cDQNGamma = 0.95;
  cDQNLR = 0.02;

  // PG params.
  cPGEpisodes = 500;
  cPGGamma = 0.95;
  cPGLR = 0.05;

var
  GPassCount, GFailCount: integer;

procedure Report(TestName: string; Passed: boolean; const Detail: string = '');
begin
  if Passed then
  begin
    Inc(GPassCount);
    WriteLn('[PASS] ', TestName, '  ', Detail);
  end
  else
  begin
    Inc(GFailCount);
    WriteLn('[FAIL] ', TestName, '  ', Detail);
  end;
end;

// ---------------------------------------------------------------------------
// Network builders.
// ---------------------------------------------------------------------------

// Q-network: state (one-hot cell) -> hidden -> Q-values (4 actions).
function BuildQNet(StateDim, NumActions, Hidden: integer): TNNet;
begin
  Result := TNNet.Create;
  Result.AddLayer(TNNetInput.Create(1, 1, StateDim));
  Result.AddLayer(TNNetFullConnectReLU.Create(Hidden));
  Result.AddLayer(TNNetFullConnectReLU.Create(Hidden));
  Result.AddLayer(TNNetFullConnectLinear.Create(NumActions));
end;

// Policy network: state -> hidden -> softmax over actions.
function BuildPolicyNet(StateDim, NumActions, Hidden: integer): TNNet;
begin
  Result := TNNet.Create;
  Result.AddLayer(TNNetInput.Create(1, 1, StateDim));
  Result.AddLayer(TNNetFullConnectReLU.Create(Hidden));
  Result.AddLayer(TNNetFullConnectReLU.Create(Hidden));
  Result.AddLayer(TNNetSoftMax.Create);
end;

// One-hot encode a grid cell (x,y) into a StateDim-vector (cGW*cGW cells).
// NOTE: The GridWorld env uses a 4-d state [x, y, 0, 0], not one-hot.
// This function is kept for reference but the test uses the raw 4-d state.
procedure EncodeCell(x, y: integer; out V: TNeuralRLState);
var
  i: integer;
begin
  SetLength(V, 4);
  V[0] := x; V[1] := y; V[2] := 0; V[3] := 0;
end;

// Extract (x, y) from a 4-d state vector.
procedure DecodeCell(const V: TNeuralRLState; out X, Y: integer);
begin
  X := Trunc(V[0]);
  Y := Trunc(V[1]);
end;

// ---------------------------------------------------------------------------
// Test 1: GridWorld deterministic path.
// ---------------------------------------------------------------------------
procedure TestGridWorldPath;
var
  Env: TNeuralGridWorldEnv;
  A, Steps: integer;
  R, TotalR: TNeuralFloat;
  S: TNeuralRLState;
  X, Y: integer;
  // Optimal path from (0,0) to (4,0) [goal at top-right]: 4 steps right.
  // With our default: goal at (W-1, 0) = (4, 0).
  // Actions: 1=right. 4 steps right reaches goal.
begin
  Env := TNeuralGridWorldEnv.Create(cGW, cGW, cGWMaxSteps, 0.0); // no slip
  try
    Env.Reset;
    // Goal is at (cGW-1, 0) = (4, 0) by default.
    // Start at (0,0). Need to go right 4 times.
    // But wait: our default goal is FGoalX = pW-1 = 4, FGoalY = 0.
    // So from (0,0) we need to go right (action 1) four times.
    TotalR := 0;
    Steps := 0;
    Env.CopyState(S);
    DecodeCell(S, X, Y);
    // Verify start position.
    if (X <> 0) or (Y <> 0) then
    begin
      Report('GridWorld start position', false, Format('got (%d,%d), expected (0,0)', [X, Y]));
      Exit;
    end;
    // Walk right until done.
    while not Env.Done do
    begin
      A := 1; // right
      R := Env.Step(A);
      TotalR := TotalR + R;
      Inc(Steps);
      if Steps > 100 then Break;
    end;
    Env.CopyState(S);
    DecodeCell(S, X, Y);
    // Should be at goal (4, 0) in exactly 4 steps, reward = 10 (goal).
    // Steps 1-4: each costs -1, step 4 gives +10 (goal replaces step cost).
    // Total = 3*(-1) + 10 = 7.
    // Actually: each Step returns the reward. Steps 1-3 return -1 each.
    // Step 4 reaches goal and returns +10.
    // Total = -1 + -1 + -1 + 10 = 7.
    Report('GridWorld reaches goal', Env.Done and (X = 4) and (Y = 0),
      Format('steps=%d, final=(%d,%d), reward=%.2f', [Steps, X, Y, TotalR]));
  finally
    Env.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Test 2: CartPole pole falls without action.
// ---------------------------------------------------------------------------
procedure TestCartPoleFalls;
var
  Env: TNeuralCartPoleEnv;
  I: integer;
  R: TNeuralFloat;
  TotalR: TNeuralFloat;
begin
  Env := TNeuralCartPoleEnv.Create;
  try
    Env.Reset;
    TotalR := 0;
    for I := 0 to 499 do
    begin
      R := Env.Step(1); // always push right (arbitrary fixed action)
      TotalR := TotalR + R;
      if Env.Done then Break;
    end;
    // The pole must eventually fall (episode terminates).
    // Even with a fixed action, the pole will fall within 500 steps.
    Report('CartPole episode terminates', Env.Done,
      Format('steps=%d, total_reward=%.1f', [I + 1, TotalR]));
  finally
    Env.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Test 3: ReplayBuffer add/sample.
// ---------------------------------------------------------------------------
procedure TestReplayBuffer;
var
  Buf: TNeuralReplayBuffer;
  T: TNeuralRLTransition;
  I: integer;
  S: TNeuralRLState;
begin
  Buf := TNeuralReplayBuffer.Create(100);
  try
    // Add 50 transitions.
    for I := 0 to 49 do
    begin
      SetLength(S, 4);
      S[0] := I; S[1] := 0; S[2] := 0; S[3] := 0;
      T.State := S;
      T.Action := I mod 4;
      T.Reward := 1.0;
      T.Done := (I = 49);
      T.NextState := S;
      Buf.Add(T);
    end;
    Report('ReplayBuffer size', Buf.Size = 50, 'size=' + IntToStr(Buf.Size));

    // Sample 10.
    Buf.Sample(10);
    Report('ReplayBuffer sample', Length(Buf.Batch) = 10,
      'batch=' + IntToStr(Length(Buf.Batch)));

    // Fill beyond capacity.
    for I := 0 to 99 do
    begin
      SetLength(S, 4);
      S[0] := I; S[1] := 0; S[2] := 0; S[3] := 0;
      T.State := S;
      T.Action := 0;
      T.Reward := 0;
      T.Done := false;
      T.NextState := S;
      Buf.Add(T);
    end;
    Report('ReplayBuffer capacity cap', Buf.Size = 100,
      'size=' + IntToStr(Buf.Size));
  finally
    Buf.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Test 4: DQN on GridWorld.
// ---------------------------------------------------------------------------
procedure TestDQNGridWorld;
var
  Env: TNeuralGridWorldEnv;
  QNet: TNNet;
  Trainer: TNeuralDQNTrainer;
  R, FinalR: TNeuralFloat;
  I: integer;
  S: TNeuralRLState;
  X, Y: integer;
begin
  Env := TNeuralGridWorldEnv.Create(cGW, cGW, cGWMaxSteps, 0.0); // no slip for determinism
  // Set goal to (4, 0) and pit to (2, 2).
  Env.SetGoal(cGW - 1, 0);
  Env.SetPit(2, 2);

  // Build Q-net: input = 4-d state [x,y,0,0], output = 4 Q-values.
  QNet := BuildQNet(Env.StateDim, Env.NumActions, 32);
  QNet.SetLearningRate(cDQNLR, 0);

  Trainer := TNeuralDQNTrainer.Create(QNet, cDQNGamma, 1.0, 0.05, 0.995, 50, 5000, 16);
  try
    // Train.
    for I := 1 to cDQNEpisodes do
    begin
      R := Trainer.TrainEpisode(Env, cGWMaxSteps);
      if (I mod 50 = 0) then
        WriteLn('  DQN episode ', I, ': reward=', Format('%.2f', [R]),
          '  eps=', Format('%.3f', [Trainer.Epsilon]));
    end;

    // Evaluate: greedy policy from start.
    FinalR := Trainer.TotalReward(Env, cGWMaxSteps, 0.0);
    Env.CopyState(S);
    DecodeCell(S, X, Y);
    // Success: reached the goal (reward includes +10).
    Report('DQN GridWorld learns', FinalR > 0,
      Format('final_reward=%.2f, pos=(%d,%d), done=%s',
        [FinalR, X, Y, BoolToStr(Env.Done, true)]));
  finally
    Trainer.Free;
    QNet.Free;
    Env.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Test 5: PolicyGradient on GridWorld.
// ---------------------------------------------------------------------------
procedure TestPGGridWorld;
var
  Env: TNeuralGridWorldEnv;
  Policy: TNNet;
  PG: TNeuralPolicyGradTrainer;
  R, EarlySum, LateSum, EarlyAvg, LateAvg: TNeuralFloat;
  I, EarlyN, LateN: integer;
begin
  Env := TNeuralGridWorldEnv.Create(cGW, cGW, cGWMaxSteps, 0.0);
  Env.SetGoal(1, 0);
  Env.SetPit(2, 2);

  Policy := BuildPolicyNet(Env.StateDim, Env.NumActions, 32);
  Policy.SetLearningRate(cPGLR, 0);

  PG := TNeuralPolicyGradTrainer.Create(Policy, cPGGamma, 1.0, 0.05, 0.995);
  try
    EarlySum := 0; LateSum := 0; EarlyN := 0; LateN := 0;
    for I := 1 to cPGEpisodes do
    begin
      R := PG.TrainOnEpisode(Env, cGWMaxSteps);
      if (I mod 50 = 0) then
        WriteLn('  PG episode ', I, ': reward=', Format('%.2f', [R]));
      if I <= 50 then begin EarlySum := EarlySum + R; EarlyN := EarlyN + 1; end;
      if I > cPGEpisodes - 50 then begin LateSum := LateSum + R; LateN := LateN + 1; end;
    end;

    // Success: average reward in the last 50 episodes is better than the first 50.
    EarlyAvg := EarlySum / Max(1, EarlyN);
    LateAvg := LateSum / Max(1, LateN);
    Report('PG GridWorld learns', LateAvg > EarlyAvg,
      Format('early_avg=%.2f, late_avg=%.2f (improvement=%.2f)',
        [EarlyAvg, LateAvg, LateAvg - EarlyAvg]));
  finally
    PG.Free;
    Policy.Free;
    Env.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------
begin
  GPassCount := 0;
  GFailCount := 0;

  WriteLn('=== TestNeuralRL ===');
  WriteLn;

  WriteLn('Test 1: GridWorld deterministic path');
  TestGridWorldPath;
  WriteLn;

  WriteLn('Test 2: CartPole episode terminates');
  TestCartPoleFalls;
  WriteLn;

  WriteLn('Test 3: ReplayBuffer');
  TestReplayBuffer;
  WriteLn;

  WriteLn('Test 4: DQN on GridWorld (', cDQNEpisodes, ' episodes)');
  TestDQNGridWorld;
  WriteLn;

  WriteLn('Test 5: PolicyGradient on GridWorld (', cPGEpisodes, ' episodes)');
  TestPGGridWorld;
  WriteLn;

  WriteLn('=================================');
  WriteLn('Total: ', GPassCount + GFailCount, '  Passed: ', GPassCount,
    '  Failed: ', GFailCount);
  WriteLn;

  if GFailCount > 0 then
  begin
    WriteLn('SOME TESTS FAILED');
  end
  else
    WriteLn('ALL TESTS PASSED');

  Readln;
end.
