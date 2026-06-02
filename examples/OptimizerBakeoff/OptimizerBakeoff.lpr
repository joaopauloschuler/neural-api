program OptimizerBakeoff;
(*
OptimizerBakeoff: trains the SAME tiny MLP on the SAME fixed toy dataset four
times, changing ONLY the optimizer, and prints a loss-vs-epoch table.

Task    : hypotenuse regression, y = sqrt(x0^2 + x1^2), x0,x1 ~ U(0,1).
Network : 2 -> 16 -> 16 -> 1 (ReLU hidden, Linear head).

Optimizer arms (all other knobs held fixed: seed, data, architecture,
mini-batch order, learning rate and number of epochs):
  1. SGD            - plain stochastic gradient descent (momentum = 0).
  2. SGD+momentum   - classic heavy-ball momentum (inertia = 0.9).
  3. Adam           - InitAdam(beta1=0.9, beta2=0.999, eps=1e-8) + CalcAdamDelta/UpdateWeightsAdam.
  4. RMSProp        - InitAdam(beta1=0.0, beta2=0.999, eps=1e-8): with beta1=0 the
                      first-moment (momentum) term collapses to the raw gradient,
                      so the Adam update lr*m/(sqrt(v)+eps) becomes lr*g/(sqrt(v)+eps),
                      i.e. textbook RMSProp.  This re-uses the library's Adam code
                      path (the framework has no separate RMSProp optimizer).

How the optimizer is selected (library API, see neural/neuralnetwork.pas):
  - SGD / SGD+momentum : NN.SetLearningRate(LR, Inertia) then NN.UpdateWeights();
      UpdateWeights uses a velocity buffer (FBackInertia) when Inertia>0.
  - Adam / RMSProp     : per layer NN.Layers[i].InitAdam(Beta1, Beta2, Epsilon),
      then NN.CalcAdamDelta() + NN.UpdateWeightsAdam().
  NN.SetBatchUpdate(True) makes the deltas accumulate across the whole
  mini-batch so every arm sees identical mini-batch gradients.

NOTE on fairness: the "best" learning rate differs per optimizer.  Here we hold
LR FIXED across all four arms (LR_SGD for the two SGD variants, LR_ADAPTIVE for
Adam/RMSProp which prefer a smaller step).  This isolates the update rule but is
NOT a tuned-LR shoot-out; see README.md.

This is a CPU-only, single-threaded demo; it finishes in a few seconds.

Copyright (C) 2026 Joao Paulo Schwarz Schuler
Released under the GNU General Public License v2 or later.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes,
  SysUtils,
  Math,
  neuralnetwork,
  neuralvolume;

const
  SEED          = 424242;
  NUM_TRAIN     = 512;
  NUM_VAL       = 256;
  BATCH_SIZE    = 32;
  NUM_EPOCHS    = 200;
  // Fixed learning rates (held constant within each family across arms).
  LR_SGD        = 0.05;   // plain SGD and SGD+momentum
  LR_ADAPTIVE   = 0.01;   // Adam and RMSProp
  MOMENTUM      = 0.9;
  ADAM_BETA1    = 0.9;
  RMS_BETA1     = 0.0;    // beta1 = 0 turns Adam into RMSProp
  BETA2         = 0.999;
  EPSILON       = 1e-8;
  CONVERGE_LOSS = 0.001;  // "epochs-to-converge" = first epoch val-loss < this
  NUM_ARMS      = 4;

type
  TArm = (armSGD, armMomentum, armAdam, armRMSProp);

var
  TrainX, TrainY: array of array of TNeuralFloat;  // fixed dataset, shared by all arms
  ValX, ValY:     array of array of TNeuralFloat;
  // loss-vs-epoch history per arm
  History: array[0..NUM_ARMS-1] of array of TNeuralFloat;
  ArmNames: array[0..NUM_ARMS-1] of string;
  FinalLoss: array[0..NUM_ARMS-1] of TNeuralFloat;
  ConvergeEpoch: array[0..NUM_ARMS-1] of integer;

procedure BuildDataset();
var
  i: integer;
  x0, x1: TNeuralFloat;
begin
  RandSeed := SEED;
  SetLength(TrainX, NUM_TRAIN, 2);
  SetLength(TrainY, NUM_TRAIN, 1);
  for i := 0 to NUM_TRAIN - 1 do
  begin
    x0 := Random; x1 := Random;
    TrainX[i][0] := x0; TrainX[i][1] := x1;
    TrainY[i][0] := Sqrt(x0*x0 + x1*x1);
  end;
  SetLength(ValX, NUM_VAL, 2);
  SetLength(ValY, NUM_VAL, 1);
  for i := 0 to NUM_VAL - 1 do
  begin
    x0 := Random; x1 := Random;
    ValX[i][0] := x0; ValX[i][1] := x1;
    ValY[i][0] := Sqrt(x0*x0 + x1*x1);
  end;
end;

// Mean squared error over the validation set.
function ValidationLoss(NN: TNNet; Input, Output: TNNetVolume): TNeuralFloat;
var
  i: integer;
  Diff, Sum: TNeuralFloat;
begin
  Sum := 0;
  for i := 0 to NUM_VAL - 1 do
  begin
    Input.FData[0] := ValX[i][0];
    Input.FData[1] := ValX[i][1];
    NN.Compute(Input);
    NN.GetOutput(Output);
    Diff := Output.FData[0] - ValY[i][0];
    Sum := Sum + Diff * Diff;
  end;
  Result := Sum / NUM_VAL;
end;

procedure TrainArm(Arm: TArm; ArmIdx: integer);
var
  NN: TNNet;
  Input, Target, Output: TNNetVolume;
  Epoch, B, i, LayerCnt: integer;
  Order: array of integer;
  Tmp, Pos: integer;
  UseAdamPath: boolean;
  LR, Beta1: TNeuralFloat;
  ValLoss: TNeuralFloat;
begin
  // Identical network, identical initial weights for every arm.
  RandSeed := SEED;
  NN := TNNet.Create();
  NN.AddLayer([
    TNNetInput.Create(2),
    TNNetFullConnectReLU.Create(16),
    TNNetFullConnectReLU.Create(16),
    TNNetFullConnectLinear.Create(1)
  ]);
  NN.SetL2Decay(0.0);
  // Accumulate deltas across the whole mini-batch (true batch gradient).
  NN.SetBatchUpdate(True);

  UseAdamPath := Arm in [armAdam, armRMSProp];
  case Arm of
    armSGD:      begin LR := LR_SGD;      NN.SetLearningRate(LR, 0.0);      end;
    armMomentum: begin LR := LR_SGD;      NN.SetLearningRate(LR, MOMENTUM); end;
    armAdam:     LR := LR_ADAPTIVE;
    armRMSProp:  LR := LR_ADAPTIVE;
  else
    LR := LR_SGD;
  end;

  if UseAdamPath then
  begin
    if Arm = armAdam then Beta1 := ADAM_BETA1 else Beta1 := RMS_BETA1;
    NN.SetLearningRate(LR, 0.0);
    for LayerCnt := 0 to NN.GetLastLayerIdx() do
      NN.Layers[LayerCnt].InitAdam(Beta1, BETA2, EPSILON);
  end;

  Input  := TNNetVolume.Create(2, 1, 1);
  Target := TNNetVolume.Create(1, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);

  // Fixed mini-batch order, reshuffled identically each epoch via the shared
  // RNG (which is reseeded to SEED above, so every arm shuffles the same way).
  SetLength(Order, NUM_TRAIN);
  for i := 0 to NUM_TRAIN - 1 do Order[i] := i;

  SetLength(History[ArmIdx], NUM_EPOCHS);
  ConvergeEpoch[ArmIdx] := -1;

  for Epoch := 1 to NUM_EPOCHS do
  begin
    // Fisher-Yates shuffle (identical across arms thanks to shared RNG state).
    for i := NUM_TRAIN - 1 downto 1 do
    begin
      Pos := Random(i + 1);
      Tmp := Order[i]; Order[i] := Order[Pos]; Order[Pos] := Tmp;
    end;

    i := 0;
    while i < NUM_TRAIN do
    begin
      NN.ClearDeltas();
      B := 0;
      while (B < BATCH_SIZE) and (i < NUM_TRAIN) do
      begin
        Input.FData[0]  := TrainX[Order[i]][0];
        Input.FData[1]  := TrainX[Order[i]][1];
        Target.FData[0] := TrainY[Order[i]][0];
        NN.Compute(Input);
        NN.Backpropagate(Target);
        Inc(B); Inc(i);
      end;
      if UseAdamPath then
      begin
        NN.CalcAdamDelta();
        NN.UpdateWeightsAdam();
      end
      else
        NN.UpdateWeights();
    end;

    ValLoss := ValidationLoss(NN, Input, Output);
    History[ArmIdx][Epoch - 1] := ValLoss;
    if (ConvergeEpoch[ArmIdx] < 0) and (ValLoss < CONVERGE_LOSS) then
      ConvergeEpoch[ArmIdx] := Epoch;
  end;

  FinalLoss[ArmIdx] := History[ArmIdx][NUM_EPOCHS - 1];

  Input.Free;
  Target.Free;
  Output.Free;
  NN.Free;
end;

procedure PrintTable();
const
  Checkpoints: array[0..6] of integer = (1, 5, 10, 25, 50, 100, 200);
var
  Arm, c, Ep: integer;
  Line: string;
begin
  WriteLn;
  WriteLn('Validation loss (MSE) vs epoch');
  WriteLn('================================================================================');
  Write(Format('%-14s', ['epoch ->']));
  for c := 0 to High(Checkpoints) do
    Write(Format('%10d', [Checkpoints[c]]));
  WriteLn;
  WriteLn('--------------------------------------------------------------------------------');
  for Arm := 0 to NUM_ARMS - 1 do
  begin
    Write(Format('%-14s', [ArmNames[Arm]]));
    for c := 0 to High(Checkpoints) do
    begin
      Ep := Checkpoints[c];
      if Ep <= NUM_EPOCHS then
        Write(Format('%10.5f', [History[Arm][Ep - 1]]))
      else
        Write(Format('%10s', ['-']));
    end;
    WriteLn;
  end;
  WriteLn('--------------------------------------------------------------------------------');

  WriteLn;
  WriteLn('Summary');
  WriteLn('--------------------------------------------------------------------------------');
  WriteLn(Format('%-14s%14s%24s', ['optimizer', 'final loss', 'epochs-to-converge']));
  WriteLn(Format('%-14s%14s%24s', ['', '(epoch 200)', '(val MSE < ' + FormatFloat('0.###', CONVERGE_LOSS) + ')']));
  for Arm := 0 to NUM_ARMS - 1 do
  begin
    if ConvergeEpoch[Arm] >= 0 then
      Line := IntToStr(ConvergeEpoch[Arm])
    else
      Line := 'not reached';
    WriteLn(Format('%-14s%14.6f%24s', [ArmNames[Arm], FinalLoss[Arm], Line]));
  end;
  WriteLn('--------------------------------------------------------------------------------');
end;

procedure PrintChart();
const
  CHART_WIDTH = 56;
  Marks: array[0..6] of integer = (1, 5, 10, 25, 50, 100, 200);
var
  Arm, c, Col, Ep: integer;
  MinL, MaxL, v, Frac: TNeuralFloat;
  ChartLine: string;
  Glyphs: array[0..NUM_ARMS-1] of char;
begin
  Glyphs[0] := 'S'; // SGD
  Glyphs[1] := 'M'; // momentum
  Glyphs[2] := 'A'; // Adam
  Glyphs[3] := 'R'; // RMSProp

  // log10 loss range across all arms / marked epochs.
  MinL :=  MaxSingle; MaxL := -MaxSingle;
  for Arm := 0 to NUM_ARMS - 1 do
    for c := 0 to High(Marks) do
    begin
      Ep := Marks[c];
      if Ep > NUM_EPOCHS then Continue;
      v := History[Arm][Ep - 1];
      if v <= 0 then Continue;
      v := Log10(v);
      if v < MinL then MinL := v;
      if v > MaxL then MaxL := v;
    end;
  if MaxL <= MinL then MaxL := MinL + 1e-6;

  WriteLn;
  WriteLn('ASCII chart  (x = log10 val-loss, low is better; rows = epoch checkpoints)');
  WriteLn(Format('  log10 loss range: %.3f .. %.3f', [MinL, MaxL]));
  WriteLn('  ', StringOfChar('-', CHART_WIDTH + 14));
  for c := 0 to High(Marks) do
  begin
    Ep := Marks[c];
    if Ep > NUM_EPOCHS then Continue;
    SetLength(ChartLine, CHART_WIDTH);
    FillChar(ChartLine[1], CHART_WIDTH, ' ');
    for Arm := 0 to NUM_ARMS - 1 do
    begin
      v := History[Arm][Ep - 1];
      if v <= 0 then v := 1e-9;
      Frac := (Log10(v) - MinL) / (MaxL - MinL);
      if Frac < 0 then Frac := 0;
      if Frac > 1 then Frac := 1;
      Col := Round(Frac * (CHART_WIDTH - 1));
      ChartLine[Col + 1] := Glyphs[Arm];
    end;
    WriteLn(Format('  ep %3d |%s|', [Ep, ChartLine]));
  end;
  WriteLn('  ', StringOfChar('-', CHART_WIDTH + 14));
  WriteLn('  legend: S=SGD  M=SGD+momentum  A=Adam  R=RMSProp   (left = lower loss)');
end;

begin
  // Reproducible: every arm reseeds RandSeed to SEED before building its net.
  ArmNames[0] := 'SGD';
  ArmNames[1] := 'SGD+momentum';
  ArmNames[2] := 'Adam';
  ArmNames[3] := 'RMSProp';

  WriteLn('Optimizer Bake-off');
  WriteLn('  task    : y = sqrt(x0^2 + x1^2), x0,x1 ~ U(0,1)');
  WriteLn('  network : 2 -> 16 -> 16 -> 1 (ReLU hidden, Linear head)');
  WriteLn(Format('  data    : %d train / %d val pairs, batch %d, %d epochs',
    [NUM_TRAIN, NUM_VAL, BATCH_SIZE, NUM_EPOCHS]));
  WriteLn(Format('  LR      : SGD family = %.3f, Adam/RMSProp = %.3f (HELD FIXED across arms)',
    [LR_SGD, LR_ADAPTIVE]));
  WriteLn(Format('  momentum: %.2f   adam betas: (%.1f, %.3f)   rmsprop beta1: %.1f',
    [MOMENTUM, ADAM_BETA1, BETA2, RMS_BETA1]));
  WriteLn;

  BuildDataset();

  WriteLn('Training arm 1/4: SGD ...');        TrainArm(armSGD,      0);
  WriteLn('Training arm 2/4: SGD+momentum ...');TrainArm(armMomentum, 1);
  WriteLn('Training arm 3/4: Adam ...');        TrainArm(armAdam,     2);
  WriteLn('Training arm 4/4: RMSProp ...');     TrainArm(armRMSProp,  3);

  PrintTable();
  PrintChart();
  WriteLn;
end.
