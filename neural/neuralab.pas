unit neuralab;
{
A stands for Array. B stands for Bytes.
This unit contains "array of bytes" functions.
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
}
{$IFDEF FPC}{$MODE Delphi}{$ENDIF}

interface

uses SysUtils;

type

  // This class represents a long array of bits.
  //   FLenght is the array lenght in bits.
  //   "Include" method calculates a key (position) for a given input array and sets 1 to the bit in the array.
  //   "Test" method returs TRUE if the bit is 1 for the position of the key of the S array.
  TABHash = object
    A: array of byte;
    FLength: longint; // number of bits

    // creates memory structure with PLength number of bits.
    constructor Init(PLength: longint{number of bits});

    // frees the memory.
    destructor Done;

    // Clears (all positions are now zeroes).
    procedure Clear;

    // Adds 1 to a position calculated from the array of bytes.
    procedure Include(var S: array of byte);

    // Checks is the position corresponding to the array of byte is 1. Returns true if it's 1.
    function Test(var S: array of byte): boolean;
  end;

// creates a Key from the array.
function ABKey(S: array of byte; Divisor: longint): longint;

// AB Compare
function ABCmp(var X, Y: array of byte): boolean;

// returns the next 1 position from ST
function ABGetNext1(var AB: array of byte; ST: word): word;

// counts the number of diff bytes
function ABCountDif(var X, Y: array of byte): longint;

// counts the number of non zero bytes
function ABCountDifZero(var X: array of byte): longint;

// A <- A and B
procedure ABAnd(var A, B: array of byte);

// returns the number of equal (matching) bytes
function ABGetEqual(var Equal, X, Y: array of byte): longint;

// Shift Logical Left
procedure ABShiftLogicalLeft(var X: array of byte);

// Shift Logical Right
procedure ABShiftLogicalRight(var X: array of byte);

// returns the number of non matching bytes
function ABGetDif(var Dif, X, Y: array of byte): longint;

// returns a string from the array (valid numbers are 0 and 1).
function ABToString(var AB: array of byte): string;

// returns a string from the array
function ABToStringR(var AB: array of byte): string;

// clears array (fills with zeros)
procedure ABClear(var AB: array of byte);

// fills with 1 the array.
procedure ABFull(var AB: array of byte);

// fills with zeros and place "1" at POS.
procedure ABBitOnPos(var AB: array of byte; POS: longint);

{ 000000 but AB[Start+X]=1 from Start with Length}
procedure ABBitOnPosAtPos(var AB: array of byte; X, Start, Len: longint);

{inverse of ABBitOnPosAtPos }
function ABReadBitOnPosAtPos(var AB: array of single; Start, Len: longint): longint;

// A <- B
procedure ABCopy(var A, B: array of byte);

{ gera passo do triangulo de pascal: A:=B[I-1] xor B[I+1]; }
procedure ABTriPascal(var A, B: array of byte);

const
  ByteToBool: array [0..1] of boolean = (False, True);
  BoolToByte: array [False..True] of byte = (0, 1);
  BoolToChar: array [False..True] of char = ('0', '1');
  CharToBool: array ['0'..'1'] of boolean = (False, True);

procedure ABSet(var A: array of byte; B: array of byte);

implementation

uses neuralbit;

function ABKey(S: array of byte; Divisor: longint): longint;
var
  I: longint;
  SumKey: longint;
begin
  SumKey := 203;
{$IFDEF FPC}
{$PUSH}
{$ENDIF}
{$OVERFLOWCHECKS OFF}

  for I := Low(S) to High(S) do
  begin
    SumKey :=
      (((SumKey * (S[I] + 11)) mod Divisor) *
      (S[(I + 1) mod (High(S) + 1)] + 17) + 3) mod Divisor;
  end;
  ABKey := abs(SumKey) mod Divisor;
{$OVERFLOWCHECKS ON}
{$IFDEF FPC}
{$POP}
{$ENDIF}
end;

function ABCmp(var X, Y: array of byte): boolean;
begin
  if Length(X) <> Length(Y) then
  begin
    ABCmp := False;
    exit;
  end;
  ABCmp := CompareMem(addr(X), addr(Y), Length(X));
end;

// Example: ABSet(X,[1,2]);
procedure ABSet(var A: array of byte; B: array of byte);
var
  I: longint;
begin
  for I := Low(A) to High(A) do
    A[I] := B[I];
end;

procedure ABTriPascal(var A, B: array of byte);
var
  I: longint;
begin
  for I := Low(A) + 1 to High(A) - 1 do
    A[I] := B[I - 1] xor B[I + 1];
end;

procedure ABCopy(var A, B: array of byte);
var
  MinLen: word;
begin
  if length(A) < length(B) then
    MinLen := length(A)
  else
    MinLen := length(B);
  Move(B, A, MinLen);
end;

{returns the next 1 position from ST }
function ABGetNext1(var AB: array of byte; ST: word): word;
var
  I: integer;
  Found: boolean;
  L: word;
begin
  Found := False;
  I := -1;
  L := Length(AB);
  while not (Found) and (I < L) do
  begin
    Inc(I);
    Found := (AB[(I + ST) mod L] <> 0);
  end;
  ABGetNext1 := (I + ST) mod L;
end;

procedure ABClear(var AB: array of byte);
var
  I: longint;
begin
  for I := Low(AB) to High(AB) do
    AB[I] := 0;
end;

procedure ABFull(var AB: array of byte);
var
  I: longint;
begin
  for I := Low(AB) to High(AB) do
    AB[I] := 1;
end;

procedure ABAnd(var A, B: array of byte);
var
  I: longint;
begin
  for I := Low(A) to High(A) do
    A[I] := A[I] and B[I];
end;

function ABCountDif(var X, Y: array of byte): longint;
var
  I: longint;
  R: longint;
begin
  R := 0;
  for I := Low(X) to High(X) do
    if X[I] <> Y[I] then
      Inc(R);
  ABCountDif := R;
end;

function ABCountDifZero(var X: array of byte): longint;
var
  I: longint;
  R: longint;
begin
  R := 0;
  for I := Low(X) to High(X) do
    if X[I] <> 0 then
      Inc(R);
  ABCountDifZero := R;
end;

function ABGetEqual(var Equal, X, Y: array of byte): longint;
var
  I: longint;
  R: longint;
begin
  R := 0;
  for I := Low(X) to High(X) do
    if X[I] = Y[I] then
    begin
      Equal[I] := 1;
      Inc(R);
    end
    else
      Equal[I] := 0;
  ABGetEqual := R;
end;

procedure ABShiftLogicalLeft(var X: array of byte);
var
  I: longint;
begin
  if High(X) > Low(X) then
    for I := Low(X) to High(X) - 1 do
      X[I] := X[I + 1];
  X[High(X)] := 0;
end;

procedure ABShiftLogicalRight(var X: array of byte);
var
  I: longint;
begin
  if High(X) > Low(X) then
    for I := High(X) downto Low(X) + 1 do
      X[I] := X[I - 1];
  X[Low(X)] := 0;
end;


function ABGetDif(var Dif, X, Y: array of byte): longint;
var
  I: longint;
  R: longint;
begin
  R := 0;
  for I := Low(X) to High(X) do
    if X[I] <> Y[I] then
    begin
      Dif[I] := 1;
      Inc(R);
    end
    else
      Dif[I] := 0;
  ABGetDif := R;
end;

procedure ABBitOnPos(var AB: array of byte; POS: longint);
begin
  ABClear(AB);
  AB[POS] := 1;
end;

procedure ABBitOnPosAtPos(var AB: array of byte; X, Start, Len: longint);
var
  I: longint;
begin
  for I := Start to Start + Len - 1 do
    AB[I] := 0;
  if (X < 0) or (X > Len) then
  begin
    writeln('Error: ABBit: STL: ', Start: 10, X: 10, Len: 10);
  end
  else
    AB[Start + X] := 1;
end;

function ABReadBitOnPosAtPos(var AB: array of single; Start, Len: longint): longint;
var
  I: longint;
  Max: single;
  MaxPos: longint;
begin
  MaxPos := Start;
  Max := AB[Start];
  for I := Start + 1 to Start + Len - 1 do
  begin
    if AB[I] > Max then
    begin
      MAX := AB[I];
      MaxPos := I;
    end;
  end;
  ABReadBitOnPosAtPos := MaxPos - Start;
end;

function ABToString(var AB: array of byte): string;
var
  I: longint;
  R: string;
begin
  R := '';
  for I := Low(AB) to High(AB) do
    case AB[I] of
      0: R := R + '0';
      1: R := R + '1';
      else
        R := R + 'X';
    end; // of case
  ABToString := R;
end;

function ABToStringR(var AB: array of byte): string;
var
  I: longint;
  R: string;
begin
  R := '';
  for I := Low(AB) to High(AB) do
    if AB[I] <> 0 then
      R := R + IntToStr(AB[I])
    else
      R := R + ' ';
  ABToStringR := R;
end;

constructor TABHash.Init(PLength: longint);
begin
  SetLength(A, (Plength div 8) + 1);
  FLength := Plength;
  Clear;
end;

destructor TABHash.Done;
begin
  SetLength(A, 0);
end;

procedure TABHash.Clear;
begin
  ABClear(A);
end;

procedure TABHash.Include(var S: array of byte);
begin
  BAWrite(A, ABKey(S, FLength), 1);
end;

function TABHash.Test(var S: array of byte): boolean;
begin
  Test := BATest(A, ABKey(S, FLength));
end;

(* procedure Test;
var H:TABHash;
    X,Y:array[0..1] of byte;
begin
H.Init(8000);
Writeln(H.Test(X));   //false
Writeln(H.Test(Y));   //false
ABSet(X,[1,2]);
ABSet(Y,[100,200]);
H.Include(X);
H.Include(Y);
Writeln(H.Test(X));   //true
Writeln(H.Test(Y));   //true
ABSet(X,[1,1]);
ABSet(Y,[200,100]);
Writeln(H.Test(X));  //false
Writeln(H.Test(Y));  //false
H.Done;
end; *)

end.
