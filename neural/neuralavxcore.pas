unit neuralavxcore;

interface

{$IFDEF CPUX64}
{$DEFINE x64}
{$ENDIF}
{$IFDEF cpux86_64}
{$DEFINE x64}
{$ENDIF}

// #################################################
// #### Exported AVX functions that use the "best" AVX cpu extension available

function AVXDotProd( x : PSingle; y : PSingle; N : integer ) : single;  inline;
procedure AVXMulAdd( x : PSingle; y : PSingle; N : integer; const fact : single );  inline;

procedure TestAVX;

implementation

uses SysUtils, CPUFeatures, Math, {$IFDEF x64} NeuralAVXx64 {$ELSE} NeuralAVX {$ENDIF};

function AVXDotProd( x : PSingle; y : PSingle; N : integer ) : single;
begin
     if IsAVX512Present and (N >=32)
     then
         Result := AVX512DotProd(x, y, N)
     else
         Result := AVX2DotProd(x, y, N);
end;

procedure AVXMulAdd( x : PSingle; y : PSingle; N : integer; const fact : single );
begin
     if IsAVX512Present and (N >=32)
     then
         AVX512MulAdd(x, y, N, fact)
     else
         AVX2MulAdd(x, y, N, fact);
end;

type
  TFloatArr = Array[0..127] of single;
  PFloatArr = ^TFloatArr;

procedure TestAVX;
   function DotProd(PtrA, PtrB: PFloatArr; NumElements: integer) : single;
   var i: Integer;
       x : double;
   begin
        x := 0;

        for i := 0 to NumElements - 1 do
            x := x + ptrA^[i]*ptrB^[i];

        Result := x;
   end;

   procedure MulAdd( PtrA, PtrB: PFloatArr; NumElements: integer; fact : single );
   var i : integer;
   begin
        for i := 0 to NumElements - 1 do
            ptra^[i] := ptra^[i] + fact*PtrB^[i];
   end;
var a, b : Array[0..127] of single;
   i: Integer;
   c1, c2 : Array[0..127] of single;
   j: Integer;
   r1, r2 : single;
begin
    // test the avx implementation for convolution and muldadd
    for i := 0 to High(a) do
    begin
         a[i] := i/10;
         b[i] := (i + 0.1)/5;
    end;

    Write('   AVX Dot Prod');
    for i := 1 to Length(a) do
    begin
         r1 := DotProd( @a[0], @b[0], i);
         r2 := AVX2DotProd( @a[0], @b[0], i );
         if not SameValue( r1, r2, 1e-3) then
         begin
              Writeln;
              Writeln('Dot product failed @ index ' + IntToStr(i) );
              exit;
         end;
    end;
    Writeln('... passed');

    Write('   AVX MulAdd');
    for i := 1 to Length(a) do
    begin
         Move( a[0], c1[0], sizeof(c1));
         Move( a[0], c2[0], sizeof(c2));

         MulAdd( @c1[0], @b[0], i, 0.2 );
         AVX2MulAdd( @c2[0], @b[0], i, 0.2);

         for j := 0 to i - 1 do
         begin
              if not SameValue( c1[j], c2[j], 1e-5) then
              begin
                   Writeln;
                   Writeln('MulAdd failed @ index ' + IntToStr(i) + ',' + IntToStr(j) );
                   exit;
              end;
         end;
    end;
    Writeln('... passed');

    if IsAVX512Present then
    begin
         Write('   AVX512 Dot Prod');
         for i := 1 to Length(a) do
         begin
              r1 := DotProd( @a[0], @b[0], i);
              r2 := AVX512DotProd( @a[0], @b[0], i );
              if not SameValue( r1, r2, 1e-3) then
              begin
                   Writeln;
                   Writeln('512bit Dot product failed @ index ' + IntToStr(i) );
                   exit;
              end;
         end;
         Writeln('... passed');

         Write('   AVX512 MulAdd');
         for i := 1 to Length(a) do
         begin
              Move( a[0], c1[0], sizeof(c1));
              Move( a[0], c2[0], sizeof(c2));

              MulAdd( @c1[0], @b[0], i, 0.2 );
              AVX512MulAdd( @c2[0], @b[0], i, 0.2);

              for j := 0 to i - 1 do
              begin
                   if not SameValue( c1[j], c2[j], 1e-4) then
                   begin
                        Writeln;
                        Writeln('MulAdd failed @ index ' + IntToStr(i) + ',' + IntToStr(j), ' ', c1[j], '/', c2[j] );
                        exit;
                   end;
              end;
         end;
         Writeln('... passed');
    end
    else
        Writeln('AVX 512 test skipped - no CPU support detected');
end;

end.

