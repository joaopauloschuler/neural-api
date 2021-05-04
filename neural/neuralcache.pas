{
uarraycache - ultra fast byte array cache.

This unit is used to store a memory structure where the key (index) is an
array of bytes and the data is an array of bytes.
The class TCacheMem is designed to be used as CACHE memory provider.

The index is an array of bytes and is sometimes called STATE or just ST.

Copyright (C) 2001 Joao Paulo Schwarz Schuler

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
unit neuralcache; // this unit used to be called UCashST2 (not sure why)

interface

// type of key and data in the memory.
type
  TState = array of byte;

type
  TProcPred = procedure(var ST: array of byte; Acao: byte) of object;

type
  // This class represents an array such as "Memory[array of byte] maps into array of byte".
  TCacheMem = object
  protected
    // maximum number of states to be cached
    FMaxStates: integer;

    // indicates that the position is filled with data.
    FFilledStatePosition: array of boolean;

    // Keys in the memory structure.
    FKeyStates: array of TState;

    // Data stored in the memory structure.
    DataA: array of TState;

    // number of hits in the cache memory.
    NHit: longint;

    // number of misses in the cache memory.
    NMiss: longint;
    NOver: longint;

  public
    procedure Init(StateLength, DataLength: longint; CacheSize: integer = 1000);
    procedure DeInit;

    // clears the memory
    procedure Clear;

    // Includes data into the memory.
    procedure Include(var ST{memory index}, DTA{memory data}: array of byte);

    // This function returns -1 when ST does not exist. When exists, returns the position.
    // Returns DTA when ST exists.
    function Read(var {input}ST{memory index},
    {output}DTA{memory data}: array of byte): longint;

    // Returns TRUE if the state/index exists on the array.
    function ValidEntry(var ST{memory index}: array of byte): boolean;

    // This function returns an indication of how much of the memory structure is in use.
    function Used: extended;

    // This function returns the frequency of hits.
    function HitsOverAll: extended;
  end;

implementation

uses neuralab;

function TCacheMem.Used: extended;
var
  I: integer;
  S: extended;
begin
  S := 0;
  for I := 0 to FMaxStates - 1 do
    if FFilledStatePosition[I] then
      S := S + 1;
  Used := S / FMaxStates;
end;

function TCacheMem.HitsOverAll: extended;
var
  T: extended;
begin
  T := NHit + NMiss;
  HitsOverAll := NHit / (T + 0.01);
end;

procedure TCacheMem.Init(StateLength, DataLength: longint; CacheSize: integer);
var
  I: integer;
begin
  Clear;
  NHit := 0;
  NMiss := 0;
  NOver := 0;
  FMaxStates := CacheSize;
  SetLength(FKeyStates, FMaxStates);
  SetLength(DataA, FMaxStates);
  SetLength(FFilledStatePosition, FMaxStates);
  for I := 0 to FMaxStates - 1 do
  begin
    SetLength(FKeyStates[I], StateLength);
    SetLength(DataA[I], DataLength);
    FFilledStatePosition[I] := False;
  end;
end;

procedure TCacheMem.DeInit;
var
  I: integer;
begin
  for I := 0 to FMaxStates - 1 do
  begin
    SetLength(FKeyStates[I], 0);
    SetLength(DataA[I], 0);
  end;
  SetLength(FKeyStates, 0);
  SetLength(DataA, 0);
  SetLength(FFilledStatePosition, 0);
end;

procedure TCacheMem.Clear;
var
  I: integer;
begin
  for I := 0 to FMaxStates - 1 do
    FFilledStatePosition[I] := False;
end;

procedure TCacheMem.Include(var ST, DTA: array of byte);

  procedure PInclude(POS: longint);
  begin
    FFilledStatePosition[POS] := True;
    ABCopy(FKeyStates[POS], ST);
    ABCopy(DataA[POS], DTA);
  end;

var
  POS, POS1: longint;
begin
  POS := ABKey(ST, FMaxStates);
  POS1 := (POS + 1) mod FMaxStates;

  // is it valid entry with wrong/other index?
  if FFilledStatePosition[POS] and not (ABCmp(FKeyStates[POS], ST)){ or
     not(ABCmp(DataA[POS],DTA)) ) } then
  begin
    Inc(NOver);
    // POS1 not valid/filled or POS1 has correct index?
    if not (FFilledStatePosition[POS1]) or (FFilledStatePosition[POS1] and
      ABCmp(FKeyStates[POS1], ST)) then
      PInclude(POS1)
    else
      PInclude(POS);
  end
  else
  begin
    PInclude(POS);
  end;
end;

function TCacheMem.ValidEntry(var ST: array of byte): boolean;
var
  POS, POS1: longint;
begin
  POS := ABKey(ST, FMaxStates);
  POS1 := (POS + 1) mod FMaxStates;
  ValidEntry := (FFilledStatePosition[POS] and ABCmp(ST, FKeyStates[POS])) or
    (FFilledStatePosition[POS1] and ABCmp(ST, FKeyStates[POS1]));
end;

function TCacheMem.Read(var ST, DTA: array of byte): longint;

  procedure PRead(POS: longint);
  begin
    Read := POS;
    ABCopy(DTA, DataA[POS]);
    Inc(NHit);
  end;

var
  POS: longint;
begin
  POS := ABKey(ST, FMaxStates);
  if FFilledStatePosition[POS] and ABCmp(ST, FKeyStates[POS]) then
  begin
    PRead(POS);
  end
  else
  begin
    POS := (POS + 1) mod FMaxStates;
    if FFilledStatePosition[POS] and ABCmp(ST, FKeyStates[POS]) then
      PRead(POS)
    else
    begin
      Read := -1;
      Inc(NMiss);
    end;
  end;
end;

end.
