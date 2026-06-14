{
neuralscheduler - learning-rate schedulers for the CAI Neural API.

This unit provides a small family of learning-rate (LR) schedulers built on a
common abstract base. A scheduler maps a training position (Epoch, Step) to a
scalar learning rate via NextLR.

t-axis convention
-----------------
Every concrete scheduler is configured with a total horizon T (and, where
relevant, sub-horizons such as stepSize or warmup) expressed in the SAME unit
as the value the scheduler keys on. All schedulers in this unit key on the
STEP argument (the global step counter), i.e. t := Step. Epoch is accepted for
interface compatibility but ignored by these implementations. t is clamped to
[0, T] (and, for PolyLR, t > T yields exactly 0) so NextLR always returns a
finite value.

Note: these classes only compute the schedule. Wiring them into a training
loop (TNeuralFit) is intentionally out of scope for this unit.

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
}
unit neuralscheduler;

{$IFDEF FPC}
{$mode objfpc}{$H+}
{$ENDIF}

interface

uses
  SysUtils, Math, neuralvolume;

type
  { TNeuralLRScheduler: abstract base for all LR schedulers. }
  // Coded by Claude (AI).
  TNeuralLRScheduler = class(TObject)
  public
    { Returns the learning rate for the given training position.
      Implementations key on Step (t := Step); Epoch is ignored. }
    function NextLR(Epoch, Step: integer): TNeuralFloat; virtual; abstract;
    { Optional hook to feed a monitored metric (e.g. validation loss) into a
      stateful, metric-driven scheduler such as TReduceLROnPlateau. The base
      implementation is a no-op so that the training loop can call it
      unconditionally for every scheduler. }
    procedure ReportMetric(metric: TNeuralFloat); virtual;
  end;

  { TStepLR: step decay. lr = baseLR * gamma^floor(t/stepSize). }
  // Coded by Claude (AI).
  TStepLR = class(TNeuralLRScheduler)
  private
    FBaseLR: TNeuralFloat;
    FStepSize: integer;
    FGamma: TNeuralFloat;
  public
    constructor Create(baseLR: TNeuralFloat; stepSize: integer; gamma: TNeuralFloat);
    function NextLR(Epoch, Step: integer): TNeuralFloat; override;
  end;

  { TCosineAnnealingLR: cosine anneal from etaMax (t=0) to etaMin (t=T).
    lr = etaMin + (etaMax-etaMin)*0.5*(1+cos(pi*t/T)). }
  // Coded by Claude (AI).
  TCosineAnnealingLR = class(TNeuralLRScheduler)
  private
    FEtaMax: TNeuralFloat;
    FEtaMin: TNeuralFloat;
    FT: integer;
  public
    constructor Create(etaMax, etaMin: TNeuralFloat; T: integer);
    function NextLR(Epoch, Step: integer): TNeuralFloat; override;
  end;

  { TWarmupCosineLR: linear warmup from 0 to etaMax over [0,warmup), then
    cosine anneal from etaMax to etaMin over [warmup,T]. }
  // Coded by Claude (AI).
  TWarmupCosineLR = class(TNeuralLRScheduler)
  private
    FEtaMax: TNeuralFloat;
    FEtaMin: TNeuralFloat;
    FWarmup: integer;
    FT: integer;
  public
    constructor Create(etaMax, etaMin: TNeuralFloat; warmup, T: integer);
    function NextLR(Epoch, Step: integer): TNeuralFloat; override;
  end;

  { TPolyLR: polynomial decay. lr = baseLR * (1 - t/T)^power, clamped to 0
    for t >= T. }
  // Coded by Claude (AI).
  TPolyLR = class(TNeuralLRScheduler)
  private
    FBaseLR: TNeuralFloat;
    FT: integer;
    FPower: TNeuralFloat;
  public
    constructor Create(baseLR: TNeuralFloat; T: integer; power: TNeuralFloat);
    function NextLR(Epoch, Step: integer): TNeuralFloat; override;
  end;

  { TPlateauMode: whether the monitored metric should be minimized (e.g.
    validation loss) or maximized (e.g. accuracy). }
  TPlateauMode = (pmMin, pmMax);
  { TPlateauThresholdMode: relative or absolute improvement threshold, as in
    PyTorch ReduceLROnPlateau (threshold_mode 'rel' / 'abs'). }
  TPlateauThresholdMode = (ptRel, ptAbs);

  { TReduceLROnPlateau: metric-driven decay, a port of PyTorch's
    torch.optim.lr_scheduler.ReduceLROnPlateau. Unlike the other schedulers in
    this unit it is NOT a pure function of the step: it tracks the best metric
    seen so far and, when no improvement beyond a threshold is observed for
    `patience` epochs, multiplies the current LR by `factor` (clamped to
    min_lr), then ignores the next `cooldown` epochs.

    Feed the monitored metric (typically validation loss) via ReportMetric
    BEFORE calling NextLR for the corresponding epoch. NextLR then returns the
    current (possibly just-reduced) LR; its Epoch/Step arguments are ignored.
    Mirrors torch state: best, num_bad_epochs, cooldown_counter. }
  // Coded by Claude (AI).
  TReduceLROnPlateau = class(TNeuralLRScheduler)
  private
    FCurrentLR: TNeuralFloat;
    FMode: TPlateauMode;
    FFactor: TNeuralFloat;
    FPatience: integer;
    FThreshold: TNeuralFloat;
    FThresholdMode: TPlateauThresholdMode;
    FCooldown: integer;
    FMinLR: TNeuralFloat;
    FBest: TNeuralFloat;
    FNumBadEpochs: integer;
    FCooldownCounter: integer;
    function IsBetter(a, best: TNeuralFloat): boolean;
  public
    constructor Create(baseLR: TNeuralFloat; mode: TPlateauMode;
      factor: TNeuralFloat; patience: integer;
      threshold: TNeuralFloat; thresholdMode: TPlateauThresholdMode;
      cooldown: integer; minLR: TNeuralFloat);
    { Feed one monitored metric; updates internal state and may reduce the LR. }
    procedure ReportMetric(metric: TNeuralFloat); override;
    { Returns the current LR (Epoch/Step ignored). }
    function NextLR(Epoch, Step: integer): TNeuralFloat; override;
  end;

  { TOneCycleLR: port of PyTorch OneCycleLR with anneal_strategy='cos'. LR
    rises from maxLR/divFactor at t=0 to maxLR over the first pct_start of the
    horizon, then cosine-anneals down to maxLR/(divFactor*finalDivFactor) at
    t=T-1. Keys on Step in [0, totalSteps-1]. }
  // Coded by Claude (AI).
  TOneCycleLR = class(TNeuralLRScheduler)
  private
    FMaxLR: TNeuralFloat;
    FInitialLR: TNeuralFloat;
    FMinLR: TNeuralFloat;
    FTotalSteps: integer;
    FUpEnd: TNeuralFloat;  // step_size_up = pct_start*total_steps - 1
  public
    constructor Create(maxLR: TNeuralFloat; totalSteps: integer;
      pctStart, divFactor, finalDivFactor: TNeuralFloat);
    function NextLR(Epoch, Step: integer): TNeuralFloat; override;
  end;

  { TCyclicLRMode: triangular keeps a fixed amplitude; triangular2 halves the
    amplitude each full cycle. (exp_range is not ported.) }
  TCyclicLRMode = (clTriangular, clTriangular2);

  { TCyclicLR: port of PyTorch CyclicLR triangular family (cycle_momentum
    handled by the optimizer, not here). LR oscillates between baseLR and maxLR
    with a half-cycle of stepSizeUp up and stepSizeDown down. }
  // Coded by Claude (AI).
  TCyclicLR = class(TNeuralLRScheduler)
  private
    FBaseLR: TNeuralFloat;
    FMaxLR: TNeuralFloat;
    FStepSizeUp: integer;
    FStepSizeDown: integer;
    FMode: TCyclicLRMode;
  public
    constructor Create(baseLR, maxLR: TNeuralFloat;
      stepSizeUp, stepSizeDown: integer; mode: TCyclicLRMode);
    function NextLR(Epoch, Step: integer): TNeuralFloat; override;
  end;

implementation

{ TNeuralLRScheduler }

procedure TNeuralLRScheduler.ReportMetric(metric: TNeuralFloat);
begin
  // Default: stateless schedulers ignore the metric.
end;

{ TStepLR }

constructor TStepLR.Create(baseLR: TNeuralFloat; stepSize: integer; gamma: TNeuralFloat);
begin
  inherited Create();
  if stepSize <= 0 then
    raise Exception.Create('TStepLR: stepSize must be > 0');
  if (gamma <= 0) or (gamma > 1) then
    raise Exception.Create('TStepLR: gamma must be in (0, 1]');
  FBaseLR := baseLR;
  FStepSize := stepSize;
  FGamma := gamma;
end;

function TStepLR.NextLR(Epoch, Step: integer): TNeuralFloat;
var
  T: integer;
begin
  T := Step;
  if T < 0 then T := 0;
  Result := FBaseLR * Power(FGamma, T div FStepSize);
end;

{ TCosineAnnealingLR }

constructor TCosineAnnealingLR.Create(etaMax, etaMin: TNeuralFloat; T: integer);
begin
  inherited Create();
  if T <= 0 then
    raise Exception.Create('TCosineAnnealingLR: T must be > 0');
  FEtaMax := etaMax;
  FEtaMin := etaMin;
  FT := T;
end;

function TCosineAnnealingLR.NextLR(Epoch, Step: integer): TNeuralFloat;
var
  T: integer;
begin
  T := Step;
  if T < 0 then T := 0;
  if T > FT then T := FT;
  Result := FEtaMin + (FEtaMax - FEtaMin) * 0.5 * (1 + Cos(Pi * T / FT));
end;

{ TWarmupCosineLR }

constructor TWarmupCosineLR.Create(etaMax, etaMin: TNeuralFloat; warmup, T: integer);
begin
  inherited Create();
  if warmup <= 0 then
    raise Exception.Create('TWarmupCosineLR: warmup must be > 0');
  if T <= warmup then
    raise Exception.Create('TWarmupCosineLR: T must be > warmup');
  FEtaMax := etaMax;
  FEtaMin := etaMin;
  FWarmup := warmup;
  FT := T;
end;

function TWarmupCosineLR.NextLR(Epoch, Step: integer): TNeuralFloat;
var
  T: integer;
begin
  T := Step;
  if T < 0 then T := 0;
  if T > FT then T := FT;
  if T < FWarmup then
    Result := FEtaMax * T / FWarmup
  else
    Result := FEtaMin + (FEtaMax - FEtaMin) * 0.5 *
      (1 + Cos(Pi * (T - FWarmup) / (FT - FWarmup)));
end;

{ TPolyLR }

constructor TPolyLR.Create(baseLR: TNeuralFloat; T: integer; power: TNeuralFloat);
begin
  inherited Create();
  if T <= 0 then
    raise Exception.Create('TPolyLR: T must be > 0');
  if power < 0 then
    raise Exception.Create('TPolyLR: power must be >= 0');
  FBaseLR := baseLR;
  FT := T;
  FPower := power;
end;

function TPolyLR.NextLR(Epoch, Step: integer): TNeuralFloat;
var
  T: integer;
  Base: TNeuralFloat;
begin
  T := Step;
  if T < 0 then T := 0;
  if T >= FT then
  begin
    Result := 0;
    Exit;
  end;
  Base := 1 - T / FT;
  Result := FBaseLR * Power(Base, FPower);
end;

{ TReduceLROnPlateau }

constructor TReduceLROnPlateau.Create(baseLR: TNeuralFloat; mode: TPlateauMode;
  factor: TNeuralFloat; patience: integer;
  threshold: TNeuralFloat; thresholdMode: TPlateauThresholdMode;
  cooldown: integer; minLR: TNeuralFloat);
begin
  inherited Create();
  if (factor <= 0) or (factor >= 1) then
    raise Exception.Create('TReduceLROnPlateau: factor must be in (0, 1)');
  if patience < 0 then
    raise Exception.Create('TReduceLROnPlateau: patience must be >= 0');
  if cooldown < 0 then
    raise Exception.Create('TReduceLROnPlateau: cooldown must be >= 0');
  FCurrentLR := baseLR;
  FMode := mode;
  FFactor := factor;
  FPatience := patience;
  FThreshold := threshold;
  FThresholdMode := thresholdMode;
  FCooldown := cooldown;
  FMinLR := minLR;
  // torch: best initialised to +inf (min) / -inf (max).
  if FMode = pmMin then
    FBest := Infinity
  else
    FBest := NegInfinity;
  FNumBadEpochs := 0;
  FCooldownCounter := 0;
end;

function TReduceLROnPlateau.IsBetter(a, best: TNeuralFloat): boolean;
begin
  // Mirrors torch ReduceLROnPlateau._is_better.
  case FMode of
    pmMin:
      if FThresholdMode = ptRel then
        Result := a < best * (1 - FThreshold)
      else
        Result := a < best - FThreshold;
    else // pmMax
      if FThresholdMode = ptRel then
        Result := a > best * (1 + FThreshold)
      else
        Result := a > best + FThreshold;
  end;
end;

procedure TReduceLROnPlateau.ReportMetric(metric: TNeuralFloat);
begin
  if IsBetter(metric, FBest) then
  begin
    FBest := metric;
    FNumBadEpochs := 0;
  end
  else
    Inc(FNumBadEpochs);

  // In cooldown: ignore bad epochs (torch zeroes num_bad_epochs in cooldown).
  if FCooldownCounter > 0 then
  begin
    Dec(FCooldownCounter);
    FNumBadEpochs := 0;
  end;

  if FNumBadEpochs > FPatience then
  begin
    if FCurrentLR > FMinLR then
    begin
      FCurrentLR := FCurrentLR * FFactor;
      if FCurrentLR < FMinLR then FCurrentLR := FMinLR;
    end;
    FCooldownCounter := FCooldown;
    FNumBadEpochs := 0;
  end;
end;

function TReduceLROnPlateau.NextLR(Epoch, Step: integer): TNeuralFloat;
begin
  Result := FCurrentLR;
end;

{ TOneCycleLR }

constructor TOneCycleLR.Create(maxLR: TNeuralFloat; totalSteps: integer;
  pctStart, divFactor, finalDivFactor: TNeuralFloat);
begin
  inherited Create();
  if totalSteps <= 1 then
    raise Exception.Create('TOneCycleLR: totalSteps must be > 1');
  if (pctStart <= 0) or (pctStart >= 1) then
    raise Exception.Create('TOneCycleLR: pctStart must be in (0, 1)');
  if divFactor <= 0 then
    raise Exception.Create('TOneCycleLR: divFactor must be > 0');
  if finalDivFactor <= 0 then
    raise Exception.Create('TOneCycleLR: finalDivFactor must be > 0');
  FMaxLR := maxLR;
  FInitialLR := maxLR / divFactor;
  FMinLR := FInitialLR / finalDivFactor;
  FTotalSteps := totalSteps;
  // torch: step_size_up = float(pct_start * total_steps) - 1
  FUpEnd := pctStart * totalSteps - 1;
end;

function TOneCycleLR.NextLR(Epoch, Step: integer): TNeuralFloat;
var
  T: integer;
  pct: TNeuralFloat;

  function Anneal(startV, endV, p: TNeuralFloat): TNeuralFloat;
  begin
    Result := endV + (startV - endV) / 2.0 * (Cos(Pi * p) + 1);
  end;

begin
  T := Step;
  if T < 0 then T := 0;
  if T > FTotalSteps - 1 then T := FTotalSteps - 1;
  if T <= FUpEnd then
  begin
    if FUpEnd > 0 then pct := T / FUpEnd else pct := 0;
    Result := Anneal(FInitialLR, FMaxLR, pct);
  end
  else
  begin
    // Down phase spans (FUpEnd, FTotalSteps-1].
    pct := (T - FUpEnd) / ((FTotalSteps - 1) - FUpEnd);
    Result := Anneal(FMaxLR, FMinLR, pct);
  end;
end;

{ TCyclicLR }

constructor TCyclicLR.Create(baseLR, maxLR: TNeuralFloat;
  stepSizeUp, stepSizeDown: integer; mode: TCyclicLRMode);
begin
  inherited Create();
  if stepSizeUp <= 0 then
    raise Exception.Create('TCyclicLR: stepSizeUp must be > 0');
  if stepSizeDown < 0 then
    raise Exception.Create('TCyclicLR: stepSizeDown must be >= 0');
  FBaseLR := baseLR;
  FMaxLR := maxLR;
  FStepSizeUp := stepSizeUp;
  if stepSizeDown = 0 then
    FStepSizeDown := stepSizeUp
  else
    FStepSizeDown := stepSizeDown;
  FMode := mode;
end;

function TCyclicLR.NextLR(Epoch, Step: integer): TNeuralFloat;
var
  T, totalSize, cycle: integer;
  stepRatio, x, scaleFactor, baseHeight, amp: TNeuralFloat;
begin
  T := Step;
  if T < 0 then T := 0;
  totalSize := FStepSizeUp + FStepSizeDown;
  stepRatio := FStepSizeUp / totalSize;
  cycle := Floor(1 + T / totalSize);
  x := 1.0 + T / totalSize - cycle;
  if x <= stepRatio then
    scaleFactor := x / stepRatio
  else
    scaleFactor := (x - 1) / (stepRatio - 1);
  if scaleFactor < 0 then scaleFactor := 0;
  amp := FMaxLR - FBaseLR;
  if FMode = clTriangular2 then
    amp := amp / Power(2.0, cycle - 1);
  baseHeight := amp * scaleFactor;
  Result := FBaseLR + baseHeight;
end;

end.
