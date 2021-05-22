(*
neuralgeneric
Copyright (C) 2013 Joao Paulo Schwarz Schuler

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
*)
unit neuralgeneric;

{$IFDEF FPC}
{$mode objfpc}
{$ENDIF}

{$H+}

interface

type
  // class used for increment/decrement/counting with maximuns and minimuns.
  TIncDec = object
  private
    X, Min, Max, Step: longint;
  public
    constructor Init(PMin, PMax, PStep, Start: longint);
    destructor Done;
    procedure Inc;
    procedure Dec;
    function Read: longint;
    procedure Define(PX: longint);
  end;

type
  { saves and restores RandSeed }
  TRandom = object
  private
    FRandSeed: longint;
  public
    procedure Save;
    procedure Restore;
  end;

function MaxSingle(x, y: single): single;
function GetMaxDivisor(x, acceptableMax: integer): integer;

/// This is an inefficient max acceptable common divisor implementation to be improved.
//  # Arguments
//     a: is an integer.
//     b: is an integer.
//     max_acceptable: maximum acceptable common divisor.
// This implementation has been ported from:
// https://github.com/joaopauloschuler/k-neural-api/blob/master/cai/util.py
function GetMaxAcceptableCommonDivisor(a, b: integer;
  max_acceptable: integer=1000000):integer;

implementation
uses math;

function GetMaxDivisor(x, acceptableMax: integer): integer;
begin
  Result := acceptableMax;
  if x <= acceptableMax then
  begin
    Result := x;
  end
  else
  begin
    while (Result > 1) do
    begin
      if x mod Result = 0 then break;
      Dec(Result);
    end;
  end;
end;

function GetMaxAcceptableCommonDivisor(a, b: integer;
  max_acceptable: integer=1000000):integer;
var
  divisor: integer;
begin
 divisor := Max(1, Min(Min(a, b), max_acceptable));
 while (divisor > 0) do
 begin
   if (a mod divisor=0) and (b mod divisor=0) then
   begin
     break;
   end;
   Dec(divisor);
 end;
 Result := divisor;
end;

constructor TIncDec.Init(PMin, PMax, PStep, Start: longint);
begin
  Min := PMin;
  Max := PMax;
  Step := PStep;
  X := Start;
end;

destructor TIncDec.Done;
begin
end;

procedure TIncDec.Inc;
begin
  X := X + Step;
  if X > Max then
    X := Max;
end;

procedure TIncDec.Dec;
begin
  X := X - Step;
  if X < Step then
    X := Min;
end;

function TIncDec.Read: longint;
begin
  Read := X;
end;

procedure TIncDec.Define(PX: longint);
begin
  X := PX;
end;

procedure TRandom.Save;
begin
  FRandSeed := RandSeed;
end;

procedure TRandom.Restore;
begin
  RandSeed := FRandSeed;
end;

function MaxSingle(x, y: single): single;
begin
  if x > y then
    MaxSingle := x
  else
    MaxSingle := Y;
end;


end.
