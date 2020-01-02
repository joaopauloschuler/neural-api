unit neuralbit;
{ v:0.3 - this unit is deprecated - do not use it}
{
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

interface
{$include neuralnetwork.inc}

type
  TArrOf2Bytes = array[0..1] of byte;
  TArrOf3Bytes = array[0..2] of byte;
  TArrOf4Bytes = array[0..3] of byte;
  TArrOf2BytesPtr = ^TArrOf2Bytes;
  TArrOf3BytesPtr = ^TArrOf3Bytes;
  TArrOf4BytesPtr = ^TArrOf4Bytes;
  TLongByteArray = array[0..1000000000] of byte;
  TArrBytePtr = ^TLongByteArray;

function POT(numero, elevado: extended): extended;
{*  Eleva o numero a sua potencia.
     exemplo : 3^4= pot(3,4) *}

{ Bit Array }
function LongintBitTest(Data: longint; P: longint): boolean;
function LongintBitFlip(Data: longint; P: longint): longint;
{ Flip no Bit P }
procedure BAClear(var VARS: array of byte);
{ limpa o valor }
procedure BAMake1(var VARS: array of byte);
{ VARS  <- 1.0 }
function BARead(var A: array of byte; P: longint): byte;
{ Bit Array Read }
procedure BAFlip(var A: array of byte; P: longint);
{ Flip no bit P }
procedure BAWrite(var A: array of byte; P: longint; Data: byte);
{ Bit Array Write }
function BATest(var A: array of byte; P: longint): boolean;
{ Bit Array Test }
procedure BASum(var x, y: array of byte);
{ soma registradores: x=x+y }
procedure BASub(var x, y: array of byte);
{ subtrai registradores x:=x - y }
procedure BAIncPos(var x: array of byte; POS: longint);
{ Incrementa na Poscicao }
procedure BADecPos(var x: array of byte; POS: longint);
{ Decrementa na Poscicao }
procedure BAInc(var x: array of byte);
{ Incrementa }
procedure BADec(var x: array of byte);
{ Decrementa }
function BAToString(VARS: array of byte): string;
{ transforma um BA em string }
function BAToFloat(var VARS: array of byte): extended;
{ transforma um BA em float }
procedure PFloatToBA(var VARS: array of byte; Valor: extended);
{ transforma um real em BA }
procedure BANot(var VARS: array of byte);
{ NOT }
procedure BAAnd(var r, x, y: array of byte);
{ r := x AND y }
procedure BAOr(var r, x, y: array of byte);
{ r := x OR y }
procedure BAXOr(var r, x, y: array of byte);
{ r := x XOR y }
function BAGrater(var x, y: array of byte): boolean;
{ x>y ?}
function BALower(var x, y: array of byte): boolean;
{ x<y? }
function BAEqual(var x, y: array of byte): boolean;
{ x=y? }
procedure BAPMul(var r, x, y: array of byte);
{ r:=r + x*y }

{ Real Boolean Functions }
function RAnd(A, B: extended): extended;
function ROr(A, B: extended): extended;
function RNot(A: extended): extended;
function RXor(A, B: extended): extended;
function REqual(A, B: extended): extended;

procedure RSum(x, y, z: extended; var R, C: extended);
{ SOMA: entrada: x,y,z; saida: Resultado e Carrier }
procedure RegSum(var x, y: array of extended);
{ soma registradores: x=x+y }
function RegEqual(var x, y: array of extended): extended;
{ devolve nivel de igualdade entre 2 registradores }
function RegOrdEqual(var x, y: array of extended): extended;
{ devolve nivel de igualdade entre 2 registradores usando relacao de ordem}
function RegToString(var VARS: array of extended): string;
{ transforma um registrador em string }

function ROrer(var VARS: array of extended): extended;
{ calcula OR de todos os operandos }
function RAnder(var VARS: array of extended): extended;
{ calcula AND de todos os operandos }
function RCNot(X: extended; var VARS: array of extended): extended;
{ Not Controlado }

function ROrMaxTerm(var VARS: array of extended; NumMaxTerm: longint): extended;
{ termo maximo do Or }
function ROrMaxTermStr(NumVars: longint; NumMaxTerm: longint): string;
{ termo maximo do Or Str }
{ string do termo maximo do or }
function RSatFunc(var VARS: array of extended; NumFunc: longint): extended;
{ funcao Sat }
function RSatFuncStr(NumVars: longint; NumFunc: longint): string;
{ funcao Sat Str}
procedure RRegen(var VARS: array of extended);
{ regenera o valor }
procedure RDegen(var VARS: array of extended);
{ degenera o valor }
procedure RDegenP(var VARS: array of extended; P: extended);
{ degenera o valor usando parametro }
procedure Clear(var VARS: array of extended);
{ limpa o valor }

{ Real Boolean Functions using Bit Arrays }
procedure BARAnd(var R, A, B: array of byte);
procedure BAROr(var R, AUX, A, B: array of byte);
procedure BARNot(var R, A: array of byte);

const
  POW2: array[0..7] of byte = (1, 2, 4, 8, 16, 32, 64, 128);
  WPOW2: array[0..15] of word = (1, 2, 4, 8, 16, 32, 64, 128, 256,
    512, 1024, 2048, 4096, 8192, 16384, 32768);

implementation

function POT(numero, elevado: extended): extended;
{*  Eleva o numero a sua potencia.
     exemplo : 3^4= pot(3,4) *}
begin
  if Numero = 0 then
    Pot := 0
  else
    Pot := exp(elevado * ln(numero));
end;

function BARead(var A: array of byte; P: longint): byte;
var
  BytePos, BitPos: longint;
begin
  BytePos := P div 8;
  BitPos := P mod 8;
  if BytePos > High(A) then
    Result := 0
  else
    Result := A[BytePos] and POW2[BitPos] shr BitPos;
end;

procedure BAFlip(var A: array of byte; P: longint);
{ Flip no bit P }
var
  BytePos, BitPos: longint;
begin
  BytePos := P div 8;
  BitPos := P mod 8;
  if BytePos <= High(A) then
    A[BytePos] := A[BytePos] xor POW2[BitPos];
end;

function BATest(var A: array of byte; P: longint): boolean;
begin
  if BARead(A, P) = 0 then
    Result := False
  else
    Result := True;
end;

function LongintBitTest(Data: longint; P: longint): boolean;
type
  TLongintToArray = array[0..3] of byte;
var
  A: ^TLongintToArray;
begin
  A := addr(Data);
  LongintBitTest := BATest(A^, P);
end;

function LongintBitFlip(Data: longint; P: longint): longint;
  { Flip no Bit P }
type
  TLongintToArray = array[0..3] of byte;
var
  A: ^TLongintToArray;
begin
  A := addr(Data);
  BAFlip(A^, P);
  LongintBitFlip := Data;
end;

procedure BAWrite(var A: array of byte; P: longint; Data: byte);
var
  BytePos, BitPos: longint;
begin
  BytePos := P div 8;
  BitPos := P mod 8;
  if Data = 0 then
    A[BytePos] := A[BytePos] and (255 - POW2[BitPos])
  else
    A[BytePos] := A[BytePos] or POW2[BitPos];
end;

procedure BAXOr(var r, x, y: array of byte);
{ r := x XOR y }
var
  Cont: longint;
begin
  for Cont := Low(r) to High(r) do
    r[Cont] := x[Cont] xor y[Cont];
end;

procedure BAOr(var r, x, y: array of byte);
{ r := x OR y }
var
  Cont: longint;
begin
  for Cont := Low(r) to High(r) do
    r[Cont] := x[Cont] or y[Cont];
end;

procedure BAAnd(var r, x, y: array of byte);
{ r := x AND y }
var
  Cont: longint;
begin
  for Cont := Low(r) to High(r) do
    r[Cont] := x[Cont] and y[Cont];
end;

procedure BANot(var VARS: array of byte);
{ NOT }
var
  Cont: longint;
begin
  for Cont := Low(VARS) to High(VARS) do
    VARS[Cont] := not (VARS[Cont]);
end;

procedure BAClear(var VARS: array of byte);
{ limpa o valor }
var
  Cont: longint;
begin
  for Cont := Low(VARS) to High(VARS) do
    VARS[Cont] := 0;
end;

procedure BAMake1(var VARS: array of byte);
{ VARS  <- 1.0 }
begin
(*BAClear(VARS);
VARS[High(VARS)]:=128;*)
  PFloatToBA(VARS, 1);
end;


procedure BSum(x, y, z{operands}: byte; var R{esult}, C{arrier}: byte); {bit sum}
var
  S: byte; {soma}
begin
  S := x + y + z;
  R := S and 1;
  C := (S shr 1);
end;

procedure BASumWordPos(var X: array of byte; POS: longint; DADO: word);
{ x[POS]:=x[POS]+DADO }
var
  NumBytes: longint;
  SOMA: word;
begin
  NumBytes := High(X) + 1;
  SOMA := X[POS] + DADO;
  X[POS] := SOMA and 255;

  if (SOMA shr 8 > 0) and (POS + 1 < NumBytes) then
    BASumWordPos(X, POS + 1, SOMA shr 8);
end; { of procedure }

procedure BASum(var x, y: array of byte);
{ soma registradores: x=x+y }
var
  NumBytes: longint;
  Cont: longint;
begin
  NumBytes := High(y) + 1;
  for Cont := 0 to NumBytes - 1 do
  begin
    BASumWordPos(X, Cont, Y[Cont]);
  end;
end; { of procedure }

procedure BASubBytePos(var X: array of byte; POS: longint; DADO: byte);
{ x[POS]:=x[POS]-DADO }
var
  NumBytes: longint;
  SUB: word;
begin
  NumBytes := High(X) + 1;
  SUB := X[POS];
  if DADO > X[POS] then
  begin
    if (POS + 1 < NumBytes) then
      BASubBytePos(X, POS + 1, 1);
    SUB := SUB + 256;
  end;

  SUB := SUB - DADO;
  X[POS] := SUB;
end; { of procedure }

procedure BASub(var x, y: array of byte);
{ soma registradores: x=x+y }
var
  NumBytes: longint;
  Cont: longint;
begin
  NumBytes := High(y) + 1;
  for Cont := 0 to NumBytes - 1 do
  begin
    BASubBytePos(X, Cont, Y[Cont]);
  end;
end; { of procedure }

(* procedure BASum(var x,y:array of byte);
{ soma registradores: x=x+y }
var NumBits:Longint;
    C,NX:byte;
    Cont:Longint;
begin
C:=0;
NumBits:=(High(y)+1)*8;
for Cont:= 0 to NumBits-1
    do begin
       BSum(BARead(x,Cont) ,BARead(y,Cont),C,NX,C);
       BAWrite(x,Cont,NX);
       end;
end; { of procedure } *)

procedure BAIncPos(var x: array of byte; POS: longint);
{ Incrementa na Poscicao }
var
  Cont: longint;
  NumBits: longint;
begin
  NumBits := (High(x) + 1) * 8;
  Cont := POS;
  while (Cont < NumBits) and (BARead(x, Cont) = 1) do
  begin
    BAWrite(x, Cont, 0);
    Cont := Cont + 1;
  end;

  if (Cont < NumBits) and (BARead(x, Cont) = 0) then
    BAWrite(x, Cont, 1);
end;

procedure BAInc(var x: array of byte);
{ Incrementa }
begin
  {BAIncPos(x,0);}
  BASumWordPos(x, 0, 1);
end;

procedure BADecPos(var x: array of byte; POS: longint);
{ Decrementa na Poscicao }
var
  Cont: longint;
  NumBits: longint;
begin
  NumBits := (High(x) + 1) * 8;
  Cont := POS;
  while (Cont < NumBits) and (BARead(x, Cont) = 0) do
  begin
    BAWrite(x, Cont, 1);
    Cont := Cont + 1;
  end;

  if (Cont < NumBits) and (BARead(x, Cont) = 1) then
    BAWrite(x, Cont, 0);
end;

procedure BADec(var x: array of byte);
{ Decrementa }
begin
  {BADecPos(x,0);}
  BASubBytePos(x, 0, 1);
end;

(*procedure BASub(var x,y:array of byte);
{ x:=x - y }
var NumBits:Longint;
    Cont:Longint;
begin
NumBits:=(High(x)+1)*8;
for Cont:=0 to NumBits-1
    do begin
       if BATest(y,Cont)
          then BADecPos(x,Cont);
       end;
end;*)

function BAGrater(var x, y: array of byte): boolean;
  { x>y ?}
var
  NumBits: longint;
  Cont: longint;
begin
  BAGrater := False;
  NumBits := (High(x) + 1) * 8;
  Cont := NumBits;
  while (Cont >= 0) and (BARead(x, Cont) = BARead(y, Cont)) do
    Cont := Cont - 1;

  if (Cont >= 0) then
    BAGrater := (BARead(x, Cont) > BARead(y, Cont));
end;

function BALower(var x, y: array of byte): boolean;
  { x<y? }
begin
  BALower := BAGrater(y, x);
end;

function BAEqual(var x, y: array of byte): boolean;
  { x=y? }
var
  NumBytes: longint;
  Cont: longint;
begin
  NumBytes := (High(x) + 1);
  Cont := 0;
  while (Cont < NumBytes) and (x[Cont] = y[Cont]) do
    Cont := Cont + 1;

  BAEqual := (Cont >= NumBytes);
end;

procedure BANumFirstLast(var x: array of byte; var Num, First, Last: longint);
{ return the Number of 1s; the first 1; the last 1 }
var
  NumBits: longint;
  ContX: longint;
begin
  NumBits := (High(x) + 1) * 8;
  First := NumBits - 1;
  Last := 0;
  Num := 0;
  for ContX := 0 to NumBits - 1 do
  begin
    if BATest(x, ContX) then
    begin
      Num := Num + 1;
      if ContX < First then
        First := ContX;
      if ContX > Last then
        Last := ContX;
    end;
  end;
end;

(* NEW procedure BAPMul(var r,x,y:array of byte);
{ r:=r + x*y }
var NumBits:Longint;
    ContX,ContY:Longint;
    FirstX,LastX,FirstY,LastY,NumX,NumY:longint;
begin
BANumFirstLast(x,NumX,FirstX,LastX);
BANumFirstLast(y,NumY,FirstY,LastY);
NumBits:=(High(x)+1)*8;
if (NumX>0) and (NumY>0)
   then begin
        for ContX:=LastX downto FirstX
            do begin
               ContY:=LastY;
               while (ContX+ContY<NumBits) and
                     BATest(x,ContX) and
                     (ContY>=FirstY) do
                     begin
                     if BATest(y,ContY)
                        then BAIncPos(r,ContX+ContY);
                     ContY:=ContY-1;
                     end;
               end;
        end;
end; { of procedure } *)

(* OLD procedure BAPMul(var r,x,y:array of byte);
{ r:=r + x*y }
var NumBits:Longint;
    ContX,ContY:Longint;
begin
NumBits:=(High(x)+1)*8;
for ContX:=0 to NumBits-1
    do begin
       ContY:=0;
       while (ContX+ContY<NumBits) do
             begin
             if (BARead(x,ContX) and BARead(y,ContY) > 0)
                then BAIncPos(r,ContX+ContY);
             ContY:=ContY+1;
             end;
       end;
end;  *)

(*procedure BAPMul(var r,x,y:array of byte);
{ r:=r + x*y }
var NumBits:Longint;
    ContX,ContY,ICX,ICY:Longint;
begin
NumBits:=(High(x)+1)*8;
for ContX:=NumBits-1 downto 0
    do begin
       ContY:=NumBits-1;
       ICX:=NumBits-ContX-1; { inverso de ContX }
       ICY:=NumBits-ContY-1; { inverso de ContY }

       while (ICX+ICY<NumBits) and
             BATest(x,ContX) and
             (ContY>=0)
             do
             begin
             if BATest(y,ContY)
                then BAIncPos(r,NumBits-(ICX+ICY)-1);
             ContY:=ContY-1;
             ICY:=NumBits-ContY-1; { inverso de ContY }
             end;
       end;
end; *)

procedure BAPMul(var r, x, y: array of byte);
{ r:=r + x*y }
var
  NumBytes: longint;
  ContX, ContY, ICX, ICY, POS: longint;
  Produto: word;
begin
  NumBytes := (High(x) + 1);
  for ContX := NumBytes - 1 downto 0 do
  begin
    ContY := NumBytes - 1;
    ICX := NumBytes - ContX - 1; { inverso de ContX }
    ICY := NumBytes - ContY - 1; { inverso de ContY }

    while (ICX + ICY < NumBytes) and (X[ContX] > 0){BATest(x,ContX)} and
      (ContY >= 0) do
    begin
      Produto := X[ContX] * Y[ContY];
      POS := NumBytes - (ICX + ICY) - 1;
      BASumWordPos(r, POS, Produto);
      ContY := ContY - 1;
      ICY := NumBytes - ContY - 1; { inverso de ContY }
    end;
  end;
end;


procedure BARAnd(var R, A, B: array of byte);
begin
  BAClear(R);
  BAPMul(R, A, B);
end;

procedure BAROr(var R, AUX, A, B: array of byte);
begin
  BAClear(R);
  BAClear(AUX);
  BARAnd(AUX, A, B); { R2=A*B }

  BASum(R, A); { R1=A+B }
  BASum(R, B);

  BASub(R, AUX); { R1 = A+B - A*B }
end;

procedure BARNot(var R, A: array of byte);
begin
  BAClear(R);
  R[0] := 1;
  BASub(R, A);
end;

function RAnd(A, B: extended): extended;
begin
  RAND := A * B;
end;

function ROr(A, B: extended): extended;
begin
  ROR := A + B - RAND(A, B);
end;

function RNot(A: extended): extended;
begin
  RNOT := 1 - A;
end;

function RXor(A, B: extended): extended;
begin
  RXOR := ROR(RAND(RNOT(A), B), RAND(A, RNOT(B)));
end;

function REqual(A, B: extended): extended;
begin
  REqual := RNot(RXor(A, B));
end;

function ROrer(var VARS: array of extended): extended;
  { calcula OR de todos os operandos }
var
  NumVars: longint;
  R: extended;
  Cont: longint;
begin
  R := 0; {elemento neutro do OR}
  NumVars := High(VARS) + 1;
  for Cont := 0 to NumVars - 1 do
    R := ROr(R, VARS[Cont]);
  ROrer := R;
end; { of function }

function RAnder(var VARS: array of extended): extended;
  { calcula AND de todos os operandos }
var
  NumVars: longint;
  R: extended;
  Cont: longint;
begin
  R := 1; {elemento neutro do AND}
  NumVars := High(VARS) + 1;
  for Cont := 0 to NumVars - 1 do
    R := RAnd(R, VARS[Cont]);
  RAnder := R;
end; { of function }

procedure RSum(x, y, z: extended; var R, C: extended);
{ SOMA: entrada: x,y,z; saida: Resultado e Carrier }
var
  AOR: array [1..4] of extended; { aux for OR }
begin
  AOR[1] := (1 - x) * (1 - y) * z;
  AOR[2] := (1 - x) * y * (1 - z);
  AOR[3] := x * (1 - y) * (1 - z);
  AOR[4] := x * y * z;
  C := ROr(ROr(x * y, y * z), x * z);
  R := ROrer(AOR);
end;

procedure RegSum(var x, y: array of extended);
{ soma registradores: x=x+y }
var
  NumVars: longint;
  C: extended;
  Cont: longint;
begin
  C := 0;
  NumVars := High(y) + 1;
  for Cont := NumVars - 1 downto 0 do
    RSum(x[Cont], y[Cont], C, x[cont], C);
  if High(x) > High(y) then
    RSum(x[NumVars], 0, C, x[NumVars], C);
end; { of procedure }

function RegEqual(var x, y: array of extended): extended;
  { devolve nivel de igualdade entre 2 registradores }
var
  NumVars: longint;
  C: extended;
  Cont: longint;
begin
  C := 1;
  NumVars := High(y) + 1;
  for Cont := NumVars - 1 downto 0 do
    C := RAnd(C, REqual(x[Cont], y[Cont]));
  RegEqual := C;
end; { of procedure }

function RegOrdEqual(var x, y: array of extended): extended;
  { devolve nivel de igualdade entre 2 registradores usando relacao de ordem}
var
  NumVars: longint;
  O1, O2: extended;
  Cont: longint;
begin
  O1 := 0;
  O2 := 0;
  NumVars := High(y) + 1;
  for Cont := 0 to NumVars - 1 do
  begin
    O1 := O1 + x[Cont] * WPOW2[NumVars - Cont - 1];
    O2 := O2 + y[Cont] * WPOW2[NumVars - Cont - 1];
  end;

  RegOrdEqual := 1 - (abs(O1 - O2) / (WPOW2[NumVars] - 1));
end; { of procedure }

function BAToString(VARS: array of byte): string;
  { transforma um BA em string }
var
  NumBits: longint;
  R: string;
  Cont: longint;
begin
  R := ''; {elemento neutro da string}
  NumBits := (High(VARS) + 1) * 8;
  for Cont := NumBits - 1 downto 0 do
  begin
    if BARead(VARS, Cont) > 0 then
      R := R + '1'
    else
      R := R + '0';
    if (Cont = NumBits - 8) then
      R := R + '.';
  end;
  BAToString := R;
end; { of function }

function BAToFloat(var VARS: array of byte): extended;
  { transforma um BA em float }
var
  NumBits: longint;
  R, Fator: extended;
  Cont: longint;
begin
  Fator := 1;
  R := 0; {elemento neutro da string}
  NumBits := (High(VARS) + 1) * 8;
  for Cont := NumBits - 1 downto 0 do
  begin
    R := R + Fator * BARead(VARS, Cont);
    Fator := Fator / 2;
  end;
  BAToFloat := R;
end; { of function }

procedure PFloatToBA(var VARS: array of byte; Valor: extended);
{ transforma um real em BA }
var
  NumBits: longint;
  Fator: extended;
  Cont: longint;
begin
  Fator := 128;
  NumBits := (High(VARS) + 1) * 8;
  BAClear(VARS);
  for Cont := NumBits - 1 downto 0 do
  begin
    if Valor >= Fator then
    begin
      Valor := Valor - Fator;
      BAWrite(VARS, Cont, 1);
    end;
    Fator := Fator / 2;
  end;
end; { of procedure }

function RegToString(var VARS: array of extended): string;
  { transforma um registrador em string }
var
  NumVars: longint;
  R: string;
  Cont: longint;
begin
  R := ''; {elemento neutro da string}
  NumVars := High(VARS) + 1;
  for Cont := 0 to NumVars - 1 do
    if VARS[Cont] > 0.55 then
      R := R + '1'
    else if VARS[Cont] < 0.45 then
      R := R + '0'
    else
      R := R + 'X';
  RegToString := R;
end; { of function }


function RCNot(X: extended; var VARS: array of extended): extended;
  { Not Controlado }
begin
  RCNot := RXor(X, RAnder(VARS));
end;


function ROrMaxTerm(var VARS: array of extended; NumMaxTerm: longint): extended;
  { termo maximo do Or }
var
  NumVars: longint;
  R: extended;
  Cont: longint;
begin
  R := 0; {elemento neutro do OR}
  NumVars := High(VARS) + 1;
  for Cont := 0 to NumVars - 1 do
  begin
    if LongintBitTest(NumMaxTerm, Cont) then
      R := ROr(R, VARS[Cont])
    else
      R := ROr(R, RNot(VARS[Cont]));
  end;
  ROrMaxTerm := R;
end; { of function }

function ROrMaxTermStr(NumVars: longint; NumMaxTerm: longint): string;
  { termo maximo do Or }
  { string do termo maximo do or }
var
  R: string;
  Cont: longint;
begin
  R := ''; {elemento neutro da + em String}
  for Cont := 0 to NumVars - 1 do
  begin
    if Cont > 0 then
      R := R + ' ';
    if LongintBitTest(NumMaxTerm, Cont) then
      R := R + ' ' + chr(Ord('A') + Cont)
    else
      R := R + '-' + chr(Ord('A') + Cont);
  end;
  ROrMaxTermStr := R;
end; { of procedure }

function RSatFunc(var VARS: array of extended; NumFunc: longint): extended;
  { funcao Sat }
var
  NumVars: longint;
  R: extended;
  Cont: longint;
  MaxMaxTerm: longint;
begin
  R := 1; {elemento neutro do AND}
  NumVars := High(VARS) + 1;
  MaxMaxTerm := Round(POT(2, NumVars));
{ Neste caso, o algoritmo eh exponencial porque
esta montando a funcao de indice NumFunc.
  A complexidade dessa funcao nao deve ser computada
juntamente com a complexidade do algoritmo de otimizacao.}
  for Cont := MaxMaxTerm - 1 downto 0 do
  begin
    if LongintBitTest(NumFunc, Cont) then
      R := RAnd(R, ROrMaxTerm(VARS, Cont));
  end;
  RSatFunc := R;
end; { of procedure }

function RSatFuncStr(NumVars: longint; NumFunc: longint): string;
  { funcao Sat }
var
  R: string;
  Cont: longint;
  MaxMaxTerm: longint;
begin
  R := ''; {elemento neutro da + em String}
  MaxMaxTerm := Round(POT(2, NumVars));
  for Cont := MaxMaxTerm - 1 downto 0 do
  begin
    if LongintBitTest(NumFunc, Cont) then
      R := R + '(' + ROrMaxTermStr(Numvars, Cont) + ')';
  end;
  RSatFuncStr := R;
end; { of procedure }

procedure RRegen(var VARS: array of extended);
{ regenera o valor }
var
  Cont: longint;
begin
  for Cont := Low(VARS) to High(VARS) do
  begin
    if VARS[Cont] > 0.5 then
      VARS[Cont] := 1
    else
      VARS[Cont] := 0;
  end;
end;

procedure RDegen(var VARS: array of extended);
{ degenera o valor }
var
  Cont: longint;
begin
  for Cont := Low(VARS) to High(VARS) do
  begin
    if VARS[Cont] > 0.5 then
      VARS[Cont] := 0.8
    else
      VARS[Cont] := 0.2;
  end;
end;

procedure RDegenP(var VARS: array of extended; P: extended);
{ degenera o valor usando parametro }
var
  Cont: longint;
begin
  for Cont := Low(VARS) to High(VARS) do
  begin
    if VARS[Cont] > 0.5 then
      VARS[Cont] := P
    else
      VARS[Cont] := 1 - P;
  end;
end;

procedure Clear(var VARS: array of extended);
{ limpa o valor }
var
  Cont: longint;
begin
  for Cont := Low(VARS) to High(VARS) do
    VARS[Cont] := 0;
end;

end. { of unit }
