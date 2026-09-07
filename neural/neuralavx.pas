unit neuralavx;

interface

{$include neuralnetwork.inc}

{$IFDEF FPC}
   This unit is only for Delphi
{$ENDIF}

procedure _AVXFillMem( dst : PSingle; FillOp : Single; NumElements : Integer ); inline;
procedure _AVXCopyRelu( dst : PSingle; src : PSingle; N : Integer ); inline;
procedure _AVXMulAdd( dst : PSingle; src : PSingle; z : PSingle; N : Integer ); inline;
procedure _AVXMulAddF( dst : PSingle; src : PSingle; N : Integer; const fact : Single ); inline;
procedure _AVXMulMulAdd( dst : PSingle; src : PSingle; N : Integer; const mulOp1, mulOp2 : Single ); inline;
procedure _AVXMulF( dst : PSingle; N : Integer; const factor : Single ); inline;
procedure _AVXMul( dst : PSingle; src : PSingle; N : Integer ); inline;
procedure _AVXAdd( dst : PSingle; src : PSingle; N : Integer ); inline;
procedure _AVXMax( dst : PSingle; src : PSingle; N : Integer ); inline;
function _AVXSumDiff( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;
function _AVXDistanceSqr( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;
procedure _AVXSub( dst : PSingle; src : PSingle; N : Integer ); inline;
function _AVXGetSum( src : PSingle; N : Integer ) : Single; inline;
function _AVXGetSumSqr( src : PSingle; N : Integer ) : Single; inline;
procedure _AVXExp( dst : PSingle; src : PSingle; N : Integer ); inline;
function _AVXDotProd( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;
function _AVXDotProdInt8( dst : PShortInt; src : PSingle; N : Integer ) : Single; inline;
procedure _AVXMulAddInt8Scalar( dst : PSingle; codes : PShortInt; W : Single; N : Integer ); inline;
procedure _AVXMulAddInt8( dst : PSingle; src : PSingle; codes : PShortInt; N : Integer ); inline;
function _AVXMaxAbsFinite( src : PSingle; N : Integer ) : Single; inline;
procedure _AVXQuantizeInt8( dst : PShortInt; src : PSingle; N : Integer; const MaxAbs : Single ); inline;
procedure _AVXDequantizeInt8( dst : PSingle; src : PShortInt; N : Integer; const Scale : Single ); inline;
procedure _AVXDecodeBF16( dst : PSingle; src : PWord; N : Integer ); inline;
procedure _AVXReluGateMask( dst : PSingle; src : PSingle; N : Integer ); inline;
procedure _AVXLeakyRelu( dst : PSingle; src : PSingle; N : Integer; const Slope : Single ); inline;
procedure _AVXDecodeF16( dst : PSingle; src : PWord; N : Integer ); inline;
function _AVXSumSqrCentered( src : PSingle; Mean : Single; N : Integer ) : Single; inline;
procedure _AVXAdamDelta( PtrDelta, PtrM, PtrV : PSingle;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR : Single;
  NumElements : Integer ); inline;
procedure _AVXAdafactorDelta( PtrDelta, PtrV : PSingle; Beta2, k, c, Epsilon : Single; NumElements : Integer ); inline;
procedure _AVXClampAbs( PtrA : PSingle; Value : Single; NumElements : Integer ); inline;
procedure _AVXLionDelta( PtrDelta, PtrM: PSingle; Beta1, k1, Beta2, k2, NegLR, PosLR : Single; NumElements : Integer ); inline;
function _AVXGetMaxPos( PtrA : PSingle; NumElements : integer; out Position : integer ) : single; inline;
function _AVXGetMinPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;
function _AVXGetMaxAbsPos( PtrA : PSingle; NumElements : integer; out Position : integer ) : single; inline;
procedure _AVXAddScalar( PtrA : PSingle; Value : single; N : integer ); inline;
function _AVXExpShiftSum( dst : PSingle; src : PSingle; Shift : single; N : integer ) : single; inline;
procedure _AVXLn( dst : PSingle; src : PSingle; N : integer ); inline;
procedure _AVXSinCos( dst : PSingle; src : PSingle; N : integer; DoCos : integer ); inline;
procedure _AVXEncodeBF16( dst: PSingle; src : PSingle; N : integer ); inline;
procedure _AVXReluL(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;
procedure _AVXEncodeF16(dst, src: Pointer; N: integer); inline;
procedure _AVXReluLGateMask(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;


implementation

uses
  SysUtils, Math,
  {$IFDEF CPU64}
  neuralavx64w
  {$ELSE}
  neuralavx32w
  {$ENDIF};

{-----------------------------------------------------------------------------
  _AVXFillMem: dst[i] := FillOp for i = 0..NumElements-1.
  Handles tail (N mod 4) in Pascal.
-----------------------------------------------------------------------------}
procedure _AVXFillMem( dst : PSingle; FillOp : Single; NumElements : Integer );
var
  localNumElements, MissedElements: Integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;

  if localNumElements > 0 then
  begin
    {$IFDEF AVX64}
    _AVX512FillMem(dst, localNumElements, FillOp);
    {$ELSE}
    _AVX2FillMem(dst, localNumElements, FillOp);
    {$ENDIF}
  end;

  {$PUSHOPT}
    {$POINTERMATH ON}
  if MissedElements > 0 then
  begin
    dst[localNumElements] := FillOp;
    if MissedElements > 1 then
    begin
      dst[localNumElements + 1] := FillOp;
      if MissedElements > 2 then
        dst[localNumElements + 2] := FillOp;
    end;
  end;
  {$POPOPT}
end;

procedure _AVXMulMulAdd( dst : PSingle; src : PSingle; N : Integer; const mulOp1, mulOp2 : Single );
begin
  {$IFDEF AVX64}
  _AVX512MulMulAdd(dst, src, N, mulOp1, mulOp2);
  {$ELSE}
  _AVX2MulMulAdd(dst, src, N, mulOp1, mulOp2);
  {$ENDIF}
end;

procedure _AVXMulAdd( dst : PSingle; src : PSingle; z : PSingle; N : Integer );
begin
  {$IFDEF AVX64}
  _AVX512MulAdd(dst, src, z, N);
  {$ELSE}
  _AVX2MulAdd(dst, src, z, N);
  {$ENDIF}
end;

procedure _AVXCopyRelu( dst : PSingle; src : PSingle; N : Integer );
begin
  {$IFDEF AVX64}
  _AVX512CopyRelu(dst, src, N);
  {$ELSE}
  _AVX2CopyRelu(dst, src, N);
  {$ENDIF}
end;

procedure _AVXMulF( dst : PSingle; N : Integer; const factor : Single );
begin
  {$IFDEF AVX64}
  _AVX512MulF(dst, N, factor);
  {$ELSE}
  _AVX2MulF(dst, N, factor);
  {$ENDIF}
end;

procedure _AVXMul( dst : PSingle; src : PSingle; N : Integer );
begin
  {$IFDEF AVX64}
  _AVX512Mul(dst, src, N);
  {$ELSE}
  _AVX2Mul(dst, src, N);
  {$ENDIF}
end;

procedure _AVXAdd( dst : PSingle; src : PSingle; N : Integer );
begin
  {$IFDEF AVX64}
  _AVX512Add(dst, src, N);
  {$ELSE}
  _AVX2Add(dst, src, N);
  {$ENDIF}
end;

procedure _AVXMax( dst : PSingle; src : PSingle; N : Integer );
begin
  {$IFDEF AVX64}
  _AVX512Max(dst, src, N);
  {$ELSE}
  _AVX2Max(dst, src, N);
  {$ENDIF}
end;

function _AVXSumDiff( dst : PSingle; src : PSingle; N : Integer ) : Single;
begin
  {$IFDEF AVX64}
  Result := _AVX512SumDiff(dst, src, N);
  {$ELSE}
  Result := _AVX2SumDiff(dst, src, N);
  {$ENDIF}
end;

function _AVXDistanceSqr( dst : PSingle; src : PSingle; N : Integer ) : Single;
begin
  {$IFDEF AVX64}
  Result := _AVX512DistanceSqr(dst, src, N);
  {$ELSE}
  Result := _AVX2DistanceSqr(dst, src, N);
  {$ENDIF}
end;

procedure _AVXSub( dst : PSingle; src : PSingle; N : Integer );
begin
  {$IFDEF AVX64}
  _AVX512SubPair(dst, src, N);
  {$ELSE}
  _AVX2Sub(dst, src, N);
  {$ENDIF}
end;

function _AVXGetSum( src : PSingle; N : Integer ) : Single;
begin
  {$IFDEF AVX64}
  Result := _AVX512GetSum(src, N);
  {$ELSE}
  Result := _AVX2GetSum(src, N);
  {$ENDIF}
end;

function _AVXGetSumSqr( src : PSingle; N : Integer ) : Single;
begin
  {$IFDEF AVX64}
  Result := _AVX512GetSumSqr(src, N);
  {$ELSE}
  Result := _AVX2GetSumSqr(src, N);
  {$ENDIF}
end;

procedure _AVXExp( dst : PSingle; src : PSingle; N : Integer );
begin
  {$IFDEF AVX64}
  _AVX512Exp(dst, src, N);
  {$ELSE}
  _AVX2Exp(dst, src, N);
  {$ENDIF}
end;

function _AVXDotProd( dst : PSingle; src : PSingle; N : Integer ) : Single;
begin
  {$IFDEF AVX64}
  Result := _AVX512DotProd(dst, src, N);
  {$ELSE}
  Result := _AVX2DotProd(dst, src, N);
  {$ENDIF}
end;

procedure _AVXMulAddF( dst : PSingle; src : PSingle; N : Integer; const fact : Single );
begin
  {$IFDEF AVX64}
  _AVX512MulAddF(dst, src, N, fact);
  {$ELSE}
  _AVX2MulAddF(dst, src, N, fact);
  {$ENDIF}
end;

function _AVXDotProdInt8( dst : PShortInt; src : PSingle; N : Integer ) : Single;
begin
  {$IFDEF AVX64}
  Result := _AVX512DotProdInt8(dst, src, N);
  {$ELSE}
  Result := _AVX2DotProdInt8(dst, src, N);
  {$ENDIF}
end;

procedure _AVXMulAddInt8Scalar( dst : PSingle; codes : PShortInt; W : Single; N : Integer );
begin
  {$IFDEF AVX64}
  _AVX512MulAddInt8Scalar(dst, codes, W, N);
  {$ELSE}
  _AVX2MulAddInt8Scalar(dst, codes, W, N);
  {$ENDIF}
end;

procedure _AVXMulAddInt8( dst : PSingle; src : PSingle; codes : PShortInt; N : Integer );
begin
  {$IFDEF AVX64}
  _AVX512MulAddInt8(dst, src, codes, N);
  {$ELSE}
  _AVX2MulAddInt8(dst, src, codes, N);
  {$ENDIF}
end;

function _AVXMaxAbsFinite( src : PSingle; N : Integer ) : Single;
begin
  {$IFDEF AVX64}
  Result := _AVX512MaxAbsFinite(src, N);
  {$ELSE}
  Result := _AVX2MaxAbsFinite(src, N);
  {$ENDIF}
end;

procedure _AVXQuantizeInt8( dst : PShortInt; src : PSingle; N : Integer; const MaxAbs : Single );
begin
  {$IFDEF AVX64}
  _AVX512QuantizeInt8(dst, src, N, MaxAbs);
  {$ELSE}
  _AVX2QuantizeInt8(dst, src, N, MaxAbs);
  {$ENDIF}
end;

procedure _AVXDequantizeInt8( dst : PSingle; src : PShortInt; N : Integer; const Scale : Single );
begin
  {$IFDEF AVX64}
  _AVX512DequantizeInt8(dst, src, N, Scale);
  {$ELSE}
  _AVX2DequantizeInt8(dst, src, N, Scale);
  {$ENDIF}
end;

procedure _AVXDecodeBF16( dst : PSingle; src : PWord; N : Integer );
begin
  {$IFDEF AVX64}
  _AVX512DecodeBF16(dst, src, N);
  {$ELSE}
  _AVX2DecodeBF16(dst, src, N);
  {$ENDIF}
end;

procedure _AVXReluGateMask( dst : PSingle; src : PSingle; N : Integer ); inline;
begin
  {$IFDEF AVX64}
  _AVX512ReluGateMask(dst, src, N);
  {$ELSE}
  _AVX2ReluGateMask(dst, src, N);
  {$ENDIF}
end;

procedure _AVXLeakyRelu( dst : PSingle; src : PSingle; N : Integer; const Slope : Single ); inline;
begin
  {$IFDEF AVX64}
  _AVX512LeakyRelu(dst, src, N, Slope);
  {$ELSE}
  _AVX2LeakyRelu(dst, src, N, Slope);
  {$ENDIF}
end;

procedure _AVXDecodeF16( dst : PSingle; src : PWord; N : Integer ); inline;
begin
  {$IFDEF AVX64}
  _AVX512DecodeF16(dst, src, N);
  {$ELSE}
  _AVX2DecodeF16(dst, src, N);
  {$ENDIF}
end;

function _AVXSumSqrCentered( src : PSingle; Mean : Single; N : Integer ) : Single; inline;
begin
  {$IFDEF AVX64}
  Result := _AVX512SumSqrCentered(src, Mean, N);
  {$ELSE}
  Result := _AVX2SumSqrCentered(src, Mean, N);
  {$ENDIF}
end;

procedure _AVXAdamDelta( PtrDelta, PtrM, PtrV : PSingle;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR : Single;
  NumElements : Integer ); inline;
begin
  {$IFDEF AVX64}
  _AVX512AdamDelta(PtrDelta, PtrM, PtrV, Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR, NumElements);
  {$ELSE}
  _AVX2AdamDelta(PSingle(PtrDelta), PSingle(PtrM), PSingle(PtrV), Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR, NumElements);
  {$ENDIF}
end;

procedure _AVXAdafactorDelta( PtrDelta, PtrV : PSingle; Beta2, k, c, Epsilon : Single; NumElements : Integer ); inline;
begin
  {$IFDEF AVX64}
  _AVX512AdafactorDelta(PtrDelta, PtrV, Beta2, k, c, Epsilon, NumElements);
  {$ELSE}
  _AVX2AdafactorDelta(PtrDelta, PtrV, Beta2, k, c, Epsilon, NumElements);
  {$ENDIF}
end;

procedure _AVXClampAbs( PtrA : PSingle; Value : Single; NumElements : Integer ); inline;
begin
  {$IFDEF AVX64}
  _AVX512ClampAbs(PtrA, Value, NumElements);
  {$ELSE}
  _AVX2ClampAbs(PtrA, Value, NumElements);
  {$ENDIF}
end;

procedure _AVXLionDelta( PtrDelta, PtrM: PSingle; Beta1, k1, Beta2, k2, NegLR, PosLR : Single; NumElements : Integer );
begin
  {$IFDEF AVX64}
  _AVX512LionDelta(PtrDelta, PtrM, Beta1, k1, Beta2, k2, NegLR, PosLR, NumElements);
  {$ELSE}
  _AVX2LionDelta(PtrDelta, PtrM, Beta1, k1, Beta2, k2, NegLR, PosLR, NumElements);
  {$ENDIF}
end;

function _AVXGetMaxPos( PtrA : PSingle; NumElements : Integer; out Position : integer ) : Single;
begin
  {$IFDEF AVX64}
  Result := _AVX512GetMaxPos(PtrA, NumElements, Position);
  {$ELSE}
  Result := _AVX2GetMaxPos(PtrA, NumElements, Position);
  {$ENDIF}
end;

function _AVXGetMinPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;
begin
  {$IFDEF AVX64}
  Result := _AVX512GetMinPos(PtrA, NumElements, Position);
  {$ELSE}
  Result := _AVX2GetMinPos(PtrA, NumElements, Position);
  {$ENDIF}
end;

function _AVXGetMaxAbsPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;
begin
  {$IFDEF AVX64}
  Result := _AVX512GetMaxAbsPos(PtrA, NumElements, Position);
  {$ELSE}
  Result := _AVX2GetMaxAbsPos(PtrA, NumElements, Position);
  {$ENDIF}
end;

procedure _AVXAddScalar( PtrA : PSingle; Value : single; N : integer ); inline;
begin
  {$IFDEF AVX64}
  _AVX512AddScalar(PtrA, Value, N);
  {$ELSE}
  _AVX2AddScalar(PtrA, Value, N);
  {$ENDIF}
end;

function _AVXExpShiftSum( dst : PSingle; src : PSingle; Shift : single; N : integer ) : single; inline;
begin
  {$IFDEF AVX64}
  Result := _AVX512ExpShiftSum(dst, src, Shift, N);
  {$ELSE}
  Result := _AVX2ExpShiftSum(dst, src, Shift, N);
  {$ENDIF}
end;

procedure _AVXLn( dst : PSingle; src : PSingle; N : integer ); inline;
begin
  {$IFDEF AVX64}
  _AVX512Ln(dst, src, N);
  {$ELSE}
  _AVX2Ln(dst, src, N);
  {$ENDIF}
end;

procedure _AVXSinCos( dst : PSingle; src : PSingle; N : integer; DoCos : integer ); inline;
begin
  {$IFDEF AVX64}
  _AVX512SinCos(dst, src, N, DoCos);
  {$ELSE}
  _AVX2SinCos(dst, src, N, DoCos);
  {$ENDIF}
end;

procedure _AVXEncodeBF16( dst: PSingle; src : PSingle; N : integer ); inline;
begin
  {$IFDEF AVX64}
  _AVX512EncodeBF16(dst, src, N);
  {$ELSE}
  _AVX2EncodeBF16(dst, src, N);
  {$ENDIF}
end;

procedure _AVXReluL(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;
begin
  {$IFDEF AVX64}
  _AVX512ReluL(dst, src, LowLimit, HighLimit, Slope, N);
  {$ELSE}
  _AVX2ReluL(dst, src, LowLimit, HighLimit, Slope, N);
  {$ENDIF}
end;

procedure _AVXEncodeF16(dst, src: Pointer; N: integer); inline;
begin
  {$IFDEF AVX64}
  _AVX512EncodeF16(dst, src, N);
  {$ELSE}
  _AVX2EncodeF16(dst, src, N);
  {$ENDIF}
end;

procedure _AVXReluLGateMask(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;
begin
  {$IFDEF AVX64}
  _AVX512ReluLGateMask(dst, src, LowLimit, HighLimit, Slope, N);
  {$ELSE}
  _AVX2ReluLGateMask(dst, src, LowLimit, HighLimit, Slope, N);
  {$ENDIF}
end;

end.
