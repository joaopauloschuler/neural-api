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

implementation

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

end.
