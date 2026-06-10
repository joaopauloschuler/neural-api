// DeepQLearning example -- the suite's first REINFORCEMENT-LEARNING demo.
//
// A minimal Deep Q-Network (DQN; Mnih et al. 2015, "Human-level control through
// deep reinforcement learning") that learns an optimal navigation policy on a
// TINY, fully self-contained, deterministic grid-world. No external data, no
// physics integrator, no new layer classes -- the agent, the environment and the
// replay machinery all live in this single .lpr and the Q-network is composed
// from existing dense layers.
//
// THE ENVIRONMENT (a 5x5 grid-world):
//   * The agent starts at the top-left corner (0,0).
//   * The goal is the bottom-right corner (4,4): reward +1.0 and the episode ends.
//   * Two fixed PIT cells terminate the episode with reward -1.0 (obstacles to
//     route around).
//   * Every non-terminal step costs -0.02 (a small living penalty that pressures
//     the agent toward the SHORTEST path, not just any path).
//   * Four actions: 0=up 1=down 2=left 3=right. Moving into a wall is a no-op
//     move (the agent stays put but still pays the step penalty).
//   * Episodes are capped at cMaxSteps to bound runtime.
// The state is fed to the network as a 25-d ONE-HOT of the agent's cell, so the
// inputs are already normalised (exactly one 1.0, the rest 0.0) -- important
// because the manual UpdateWeights path BYPASSES gradient clipping, so we lean on
// well-conditioned inputs + a small learning rate for stability.
//
// THE AGENT (textbook DQN):
//   * Q-network: 25 -> FullConnectReLU(64) -> FullConnectReLU(64) ->
//     FullConnectLinear(4). One linear output per action = Q(s, .).
//   * Experience replay: a ring buffer of (s, a, r, s', done) transitions; each
//     learning step samples a random minibatch, decorrelating the updates.
//   * Target network: a frozen copy of the online net, re-synced every
//     cTargetSync steps via CopyWeights (NOT LoadFromFile), used to form a stable
//     TD target.
//   * Epsilon-greedy exploration with exponential epsilon decay (explore early,
//     exploit late).
//   * TD update: y = r + gamma * max_a' Q_target(s', a') for non-terminal s',
//     else y = r. We regress Q(s,a) toward y for the TAKEN action ONLY: the
//     training target vector is set equal to the current online Q output, then its
//     taken-action component is overwritten with y, so the squared-error gradient
//     is exactly zero on the untaken actions (standard DQN single-action update).
//   * Minibatch gradients are ACCUMULATED with SetBatchUpdate(True) (the per-sample
//     default would zero each neuron's delta) and applied with one UpdateWeights +
//     ClearDeltas per batch.
//
// HEADLINE: a moving-average learning curve shows the agent climbing from random
// (negative return, hitting pits / timing out) toward the optimal return; then we
// roll out the GREEDY (epsilon=0) policy from the fixed start and print the ASCII
// trajectory reaching the goal, plus the greedy success rate over all start cells.
//
// Pure CPU, tiny net + modest replay buffer, finishes in well under a minute.
//
// Coded by Claude (AI).
program DeepQLearning;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cGrid       = 5;                 // 5x5 grid-world
  cNumCells   = cGrid * cGrid;     // 25 states (one-hot encoded)
  cNumActions = 4;                 // up / down / left / right
  cStartX     = 0;  cStartY = 0;   // fixed start (top-left)
  cGoalX      = 4;  cGoalY = 4;    // fixed goal  (bottom-right)
  cPit1X      = 1;  cPit1Y = 3;    // pit / obstacle #1
  cPit2X      = 3;  cPit2Y = 1;    // pit / obstacle #2

  cStepPenalty= -0.02;             // living cost per non-terminal step
  cGoalReward = 1.0;
  cPitReward  = -1.0;
  cMaxSteps   = 50;                // per-episode step cap

  cEpisodes   = 1500;              // training episodes
  cGamma      = 0.95;              // discount
  cLR         = 0.02;              // small LR (manual path: no grad clipping)
  cBufCap     = 4000;              // replay ring-buffer capacity
  cBatch      = 32;                // minibatch size
  cWarmup     = 200;               // transitions before learning starts
  cTargetSync = 250;              // env-steps between target-net resyncs

  cEpsStart   = 1.0;
  cEpsEnd     = 0.05;
  cEpsDecay   = 0.995;             // per-episode multiplicative decay

  cMAWindow   = 50;                // learning-curve moving-average window

type
  TTransition = record
    S:    array[0..cNumCells - 1] of TNeuralFloat; // one-hot state
    A:    integer;                                 // action taken
    R:    TNeuralFloat;                            // reward
    SN:   array[0..cNumCells - 1] of TNeuralFloat; // one-hot next state
    Done: boolean;                                 // terminal flag
  end;

var
  // Replay ring buffer.
  Buf: array[0..cBufCap - 1] of TTransition;
  BufCount, BufHead: integer;

// ---------------------------------------------------------------------------
// Environment helpers.
// ---------------------------------------------------------------------------

function IsPit(x, y: integer): boolean;
begin
  Result := ((x = cPit1X) and (y = cPit1Y)) or ((x = cPit2X) and (y = cPit2Y));
end;

function IsGoal(x, y: integer): boolean;
begin
  Result := (x = cGoalX) and (y = cGoalY);
end;

function IsTerminalCell(x, y: integer): boolean;
begin
  Result := IsGoal(x, y) or IsPit(x, y);
end;

// One-hot encode (x,y) into a 25-d state vector.
procedure Encode(x, y: integer; var V: array of TNeuralFloat);
var i: integer;
begin
  for i := 0 to cNumCells - 1 do V[i] := 0.0;
  V[y * cGrid + x] := 1.0;
end;

// One-hot encode (x,y) directly into a volume's Raw buffer.
procedure EncodeVol(x, y: integer; V: TNNetVolume);
var i: integer;
begin
  for i := 0 to cNumCells - 1 do V.Raw[i] := 0.0;
  V.Raw[y * cGrid + x] := 1.0;
end;

// Apply action a in cell (x,y); returns reward and (via var) the next cell + done.
function Step(x, y, a: integer; out nx, ny: integer; out done: boolean): TNeuralFloat;
begin
  nx := x; ny := y;
  case a of
    0: if y > 0          then ny := y - 1;  // up
    1: if y < cGrid - 1  then ny := y + 1;  // down
    2: if x > 0          then nx := x - 1;  // left
    3: if x < cGrid - 1  then nx := x + 1;  // right
  end;
  if IsGoal(nx, ny) then
  begin
    Result := cGoalReward; done := True;
  end
  else if IsPit(nx, ny) then
  begin
    Result := cPitReward; done := True;
  end
  else
  begin
    Result := cStepPenalty; done := False;
  end;
end;

// ---------------------------------------------------------------------------
// Replay buffer.
// ---------------------------------------------------------------------------

procedure StoreTransition(const S, SN: array of TNeuralFloat; A: integer;
  R: TNeuralFloat; Done: boolean);
var i: integer;
begin
  for i := 0 to cNumCells - 1 do
  begin
    Buf[BufHead].S[i]  := S[i];
    Buf[BufHead].SN[i] := SN[i];
  end;
  Buf[BufHead].A    := A;
  Buf[BufHead].R    := R;
  Buf[BufHead].Done := Done;
  BufHead := (BufHead + 1) mod cBufCap;
  if BufCount < cBufCap then Inc(BufCount);
end;

// ---------------------------------------------------------------------------
// Q-network helpers.
// ---------------------------------------------------------------------------

function BuildQNet(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cNumCells));
  Result.AddLayer(TNNetFullConnectReLU.Create(64));
  Result.AddLayer(TNNetFullConnectReLU.Create(64));
  Result.AddLayer(TNNetFullConnectLinear.Create(cNumActions));
end;

// Greedy action = argmax_a Q(s,a) under network NN (writes Q row into QOut).
function GreedyAction(NN: TNNet; Inp: TNNetVolume; out QMax: TNeuralFloat): integer;
var a, best: integer; q: TNeuralFloat;
begin
  NN.Compute(Inp);
  best := 0; QMax := NN.GetLastLayer.Output.Raw[0];
  for a := 1 to cNumActions - 1 do
  begin
    q := NN.GetLastLayer.Output.Raw[a];
    if q > QMax then begin QMax := q; best := a; end;
  end;
  Result := best;
end;

// ---------------------------------------------------------------------------
// One DQN learning step over a random minibatch.
// ---------------------------------------------------------------------------

procedure LearnBatch(Online, Target: TNNet; Inp, NextInp, Tgt: TNNetVolume);
var
  b, idx, a, j: integer;
  qmax, y: TNeuralFloat;
begin
  Online.SetBatchUpdate(True);
  Online.ClearDeltas();
  for b := 0 to cBatch - 1 do
  begin
    idx := Random(BufCount);
    // Online Q(s,.) -> used as the regression target template.
    for j := 0 to cNumCells - 1 do Inp.Raw[j] := Buf[idx].S[j];
    Online.Compute(Inp);
    Tgt.Copy(Online.GetLastLayer.Output);  // zero error on untaken actions
    // TD target for the taken action.
    if Buf[idx].Done then
      y := Buf[idx].R
    else
    begin
      for j := 0 to cNumCells - 1 do NextInp.Raw[j] := Buf[idx].SN[j];
      a := GreedyAction(Target, NextInp, qmax);
      y := Buf[idx].R + cGamma * qmax;
    end;
    Tgt.Raw[Buf[idx].A] := y;
    // Backprop against the SAME forward pass that produced the template
    // (Online.Output still holds Q(s,.) from the Compute above), so the error
    // is exactly zero on the untaken actions and (Q(s,a) - y) on the taken one.
    Online.Backpropagate(Tgt);
  end;
  // Apply the averaged minibatch gradient once.
  Online.UpdateWeights();
  Online.ClearDeltas();
end;

// ---------------------------------------------------------------------------
// Greedy rollout (epsilon = 0) -> ASCII trajectory + success flag.
// ---------------------------------------------------------------------------

function GreedyRollout(NN: TNNet; sx, sy: integer; PrintTraj: boolean): boolean;
var
  Inp: TNNetVolume;
  x, y, nx, ny, a, steps, gx, gy: integer;
  qmax, r: TNeuralFloat;
  done: boolean;
  visited: array[0..cGrid - 1, 0..cGrid - 1] of boolean;
  line: string;
begin
  Inp := TNNetVolume.Create(cNumCells);
  for gx := 0 to cGrid - 1 do for gy := 0 to cGrid - 1 do visited[gx, gy] := False;
  try
    x := sx; y := sy; steps := 0; Result := False;
    visited[x, y] := True;
    while (steps < cMaxSteps) do
    begin
      EncodeVol(x, y, Inp);
      a := GreedyAction(NN, Inp, qmax);
      r := Step(x, y, a, nx, ny, done);
      x := nx; y := ny; Inc(steps);
      visited[x, y] := True;
      if done then
      begin
        Result := IsGoal(x, y);
        Break;
      end;
    end;
    if PrintTraj then
    begin
      WriteLn('  greedy trajectory from start (', sx, ',', sy, '):  ',
              'reached goal = ', Result, '  steps = ', steps);
      for gy := 0 to cGrid - 1 do
      begin
        line := '    ';
        for gx := 0 to cGrid - 1 do
        begin
          if IsGoal(gx, gy)       then line := line + ' G'
          else if IsPit(gx, gy)   then line := line + ' X'
          else if (gx = sx) and (gy = sy) then line := line + ' S'
          else if visited[gx, gy] then line := line + ' *'
          else                         line := line + ' .';
        end;
        WriteLn(line);
      end;
    end;
  finally
    Inp.Free;
  end;
end;

// Greedy success rate over all non-terminal start cells.
function GreedySuccessRate(NN: TNNet): TNeuralFloat;
var sx, sy, total, ok: integer;
begin
  total := 0; ok := 0;
  for sy := 0 to cGrid - 1 do
    for sx := 0 to cGrid - 1 do
      if not IsTerminalCell(sx, sy) then
      begin
        Inc(total);
        if GreedyRollout(NN, sx, sy, False) then Inc(ok);
      end;
  Result := ok / total;
end;

// ---------------------------------------------------------------------------
// Main training loop.
// ---------------------------------------------------------------------------
var
  Online, Target: TNNet;
  Inp, NextInp, Tgt: TNNetVolume;
  encS, encSN: array[0..cNumCells - 1] of TNeuralFloat;
  ep, x, y, nx, ny, a, steps, envSteps: integer;
  eps, epRet, r, qmax: TNeuralFloat;
  done: boolean;
  retHist: array[0..cEpisodes - 1] of TNeuralFloat;
  stepHist: array[0..cEpisodes - 1] of integer;
  i, k0: integer;
  maRet: TNeuralFloat; maSteps: TNeuralFloat;
begin
  RandSeed := 20260607;

  WriteLn('=== Deep Q-Learning on a ', cGrid, 'x', cGrid, ' grid-world (DQN) ===');
  WriteLn('start=(', cStartX, ',', cStartY, ')  goal=(', cGoalX, ',', cGoalY,
          ')  pits=(', cPit1X, ',', cPit1Y, '),(', cPit2X, ',', cPit2Y, ')');
  WriteLn('step penalty=', cStepPenalty:0:2, '  gamma=', cGamma:0:2,
          '  lr=', cLR:0:3, '  replay=', cBufCap, '  batch=', cBatch,
          '  target-sync=', cTargetSync);
  WriteLn;

  Online := BuildQNet();
  Target := BuildQNet();
  Target.CopyWeights(Online);            // start synced
  WriteLn('Q-net: 25 -> ReLU(64) -> ReLU(64) -> Linear(4)   params = ',
          Online.CountWeights());
  WriteLn;

  Online.SetLearningRate(cLR, 0.9);

  Inp     := TNNetVolume.Create(cNumCells);
  NextInp := TNNetVolume.Create(cNumCells);
  Tgt     := TNNetVolume.Create(cNumActions);

  BufCount := 0; BufHead := 0; envSteps := 0;
  eps := cEpsStart;

  WriteLn('training ', cEpisodes, ' episodes...');
  WriteLn('  episode | epsilon | MA(', cMAWindow, ') return | MA(', cMAWindow, ') steps');

  try
    for ep := 0 to cEpisodes - 1 do
    begin
      x := cStartX; y := cStartY;
      epRet := 0.0; steps := 0; done := False;
      while (not done) and (steps < cMaxSteps) do
      begin
        Encode(x, y, encS);
        EncodeVol(x, y, Inp);
        // Epsilon-greedy action selection.
        if Random < eps then
          a := Random(cNumActions)
        else
          a := GreedyAction(Online, Inp, qmax);
        r := Step(x, y, a, nx, ny, done);
        Encode(nx, ny, encSN);
        StoreTransition(encS, encSN, a, r, done);
        epRet := epRet + r;
        x := nx; y := ny;
        Inc(steps); Inc(envSteps);
        // Learn once warmed up.
        if BufCount >= cWarmup then
          LearnBatch(Online, Target, Inp, NextInp, Tgt);
        // Periodic target-network resync (CopyWeights, not LoadFromFile).
        if (envSteps mod cTargetSync) = 0 then
          Target.CopyWeights(Online);
      end;
      retHist[ep]  := epRet;
      stepHist[ep] := steps;
      // Decay exploration.
      if eps > cEpsEnd then eps := eps * cEpsDecay;
      if eps < cEpsEnd then eps := cEpsEnd;
      // Periodic learning-curve log (moving averages).
      if ((ep + 1) mod 100 = 0) or (ep = 0) then
      begin
        k0 := ep - cMAWindow + 1; if k0 < 0 then k0 := 0;
        maRet := 0; maSteps := 0;
        for i := k0 to ep do
        begin
          maRet := maRet + retHist[i];
          maSteps := maSteps + stepHist[i];
        end;
        maRet := maRet / (ep - k0 + 1);
        maSteps := maSteps / (ep - k0 + 1);
        WriteLn('  ', (ep + 1):7, ' | ', eps:7:3, ' | ', maRet:14:3, ' | ', maSteps:13:1);
      end;
    end;

    WriteLn;
    WriteLn('optimal shortest path start->goal avoiding pits = 8 steps ',
            '(return ~ ', (cGoalReward + 7 * cStepPenalty):0:2, ')');
    WriteLn;

    // Final greedy rollout from the fixed start -> ASCII proof it learned.
    WriteLn('=== final GREEDY policy rollout ===');
    GreedyRollout(Online, cStartX, cStartY, True);
    WriteLn;
    WriteLn('greedy success rate over all ', (cNumCells - 4),
            ' non-terminal start cells = ',
            (GreedySuccessRate(Online) * 100):0:1, '%');
  finally
    Inp.Free; NextInp.Free; Tgt.Free;
    Online.Free; Target.Free;
  end;
end.
