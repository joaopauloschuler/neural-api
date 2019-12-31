(*
neuralevolutionary
Copyright (C) 2017 Joao Paulo Schwarz Schuler

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

unit neuralevolutionary;
(*
// Generic Evolutionary Algorithm by Joao Paulo Schwarz Schuler
// https://sourceforge.net/p/cai/

// Usage Example:
// This example implements the maximization of a simple problem:
// MAX f(x) = -Abs(5000-X)

type
  // Element to be optimized. Make this object as complex as you would like.
  TEvElement = object
    Data: single;
  end;

  // Tell how your elements are mutated and evaluated.
  TMyEvExample = class( specialize TEvolutionary<TEvElement> )
  protected
    procedure Mutate(var Element: TEvElement); override;
    function Evaluation(var Element: TEvElement): double; override;
  end;

procedure TMyEvExample.Mutate(var Element: TEvElement);
begin
  Element.data := Element.data + (Random(100)-50);
end;

function TMyEvExample.Evaluation(var Element: TEvElement): double;
begin
  Result := -Abs(5000-Element.Data);
end;

var
  Ev: TMyEvExample;
  Element: TEvElement;
begin
  Element.Data := 0;
  Ev := TMyEvExample.Create();
  Element := Ev.Evolve(Element,10000);
  WriteLn('Solution:', Element.Data:6:2);
  Ev.Free;
end;
*)

{$IFDEF FPC} {$mode objfpc}{$H+} {$ENDIF}

interface

type

  { TEvolutionary }
  {$IFDEF FPC}
  generic TEvolutionary<T> = class(TObject)
  {$ELSE}
  TEvolutionary<T> = class(TObject)
  {$ENDIF}
  protected
    FAlwaysGetBest: boolean;
    FKidsPerFather: integer;
    FLastEval: double;
    function Clone(var Element: T): T; virtual;
    procedure Mutate(var Element: T); virtual; abstract;
    function Evaluation(var Element: T): double; virtual; abstract;
    function GetBestKid(var Element: T): double;
  public
    function Evolve(Element: T; RunCnt: integer): T;
    constructor Create(pAlwaysGetBest: boolean = False; pKidsPerFather: integer = 10);
    destructor Destroy; override;
  published
    property AlwaysGetBest: boolean read FAlwaysGetBest write FAlwaysGetBest;
    property KidsPerFather: integer read FKidsPerFather write FKidsPerFather;
    property LastEval: double read FLastEval;
  end;

implementation

{$IFDEF FPC}
function TEvolutionary.Clone(var Element: T): T;
{$ELSE}
function TEvolutionary<T>.Clone(var Element: T): T;
{$ENDIF}
begin
  // This solution works well for records and "objects" declared as "object".
  Result := Element;
end;

{$IFDEF FPC}
function TEvolutionary.GetBestKid(var Element: T): double;
{$ELSE}
function TEvolutionary<T>.GetBestKid(var Element: T): double;
{$ENDIF}
var
  KidsCnt: integer;
  NewElement: T;
  NewEval: double;
begin
  for KidsCnt := 1 to FKidsPerFather do
  begin
    NewElement := Clone(Element);
    Mutate(NewElement);
    NewEval := Evaluation(NewElement);

    if (KidsCnt = 1) then
    begin
      Result := NewEval;
      Element := Clone(NewElement);
    end
    else
    begin
      if (NewEval > Result) then
      begin
        Result := NewEval;
        Element := Clone(NewElement);
      end; // end of if (NewEval > Result)
    end; // end of if (KidsCnt = 1)
  end; // end of KidsCnt
end;

{$IFDEF FPC}
function TEvolutionary.Evolve(Element: T; RunCnt: integer): T;
{$ELSE}
function TEvolutionary<T>.Evolve(Element: T; RunCnt: integer): T;
{$ENDIF}
var
  GlobalCnt: integer;
  CurrentEval: double;
  NewElement: T;
  NewEval: double;
begin
  Result := Clone(Element);
  CurrentEval := Evaluation(Result);

  for GlobalCnt := 1 to RunCnt do
  begin
    NewElement := Clone(Result);
    NewEval := GetBestKid(NewElement);

    if (FAlwaysGetBest or (NewEval >= CurrentEval)) then
    begin
      CurrentEval := NewEval;
      FLastEval := CurrentEval;
      Result := Clone(NewElement);
    end;
  end;
end;

{$IFDEF FPC}
constructor TEvolutionary.Create(pAlwaysGetBest: boolean = False; pKidsPerFather: integer = 10);
{$ELSE}
constructor TEvolutionary<T>.Create(pAlwaysGetBest: boolean = False; pKidsPerFather: integer = 10);
{$ENDIF}
begin
  inherited Create;
  FAlwaysGetBest := pAlwaysGetBest;
  FKidsPerFather := pKidsPerFather;
end;

{$IFDEF FPC}
destructor TEvolutionary.Destroy;
{$ELSE}
destructor TEvolutionary<T>.Destroy;
{$ENDIF}
begin
  inherited Destroy;
end;

end.
