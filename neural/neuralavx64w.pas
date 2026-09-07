unit neuralavx64w;

interface

{$include neuralnetwork.inc}

{$IFDEF FPC}
   This unit is only for Delphi
{$ENDIF}

{$IFNDEF WIN64}
   This unit is only for Win64
{$ENDIF}

// Due to poor AVX-512 support in the current version of Delphi, all AVX-512 calls are forwarded to AVX2.

uses
  neuralavxconst;

// Coded by DeepSeek (AI)
procedure _AVX2FillMem( dst : PSingle; N : Integer; const fact : Single);
procedure _AVX512FillMem( dst : PSingle; N : Integer; const fact : Single); inline;

// Coded by DeepSeek (AI)
procedure _AVX2MulAddF( dst : PSingle; src : PSingle; N : Integer; const fact : Single);
procedure _AVX512MulAddF( dst : PSingle; src : PSingle; N : Integer; const fact : Single); inline;

// Coded by DeepSeek (AI)
procedure _AVX2MulMulAdd( dst : PSingle; src : PSingle; N : Integer; const mulOp1, mulOp2 : Single);
procedure _AVX512MulMulAdd( dst : PSingle; src : PSingle; N : Integer; const mulOp1, mulOp2 : Single); inline;

// Coded by DeepSeek (AI)
procedure _AVX2MulAdd( dst : PSingle; src : PSingle; z : PSingle; N : Integer);
procedure _AVX512MulAdd( dst : PSingle; src : PSingle; z : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2CopyRelu( dst : PSingle; src : PSingle; N : Integer);
procedure _AVX512CopyRelu( dst : PSingle; src : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2MulF( dst : PSingle; N : Integer; const factor : Single);
procedure _AVX512MulF( dst : PSingle; N : Integer; const factor : Single); inline;

// Coded by DeepSeek (AI)
procedure _AVX2Mul( dst : PSingle; src : PSingle; N : Integer);
procedure _AVX512Mul( dst : PSingle; src : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2Add( dst : PSingle; src : PSingle; N : Integer);
procedure _AVX512Add( dst : PSingle; src : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2Max( dst : PSingle; src : PSingle; N : Integer);
procedure _AVX512Max( dst : PSingle; src : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
function _AVX2SumDiff( dst : PSingle; src : PSingle; N : Integer ) : Single;
function _AVX512SumDiff( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
function _AVX2DistanceSqr( dst : PSingle; src : PSingle; N : Integer ) : Single;
function _AVX512DistanceSqr( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2SubPair( dst : PSingle; src : PSingle; N : Integer);
procedure _AVX512SubPair( dst : PSingle; src : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
function _AVX2GetSum( src : PSingle; N : Integer ) : Single;
function _AVX512GetSum( src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
function _AVX2GetSumSqr( src : PSingle; N : Integer ) : Single;
function _AVX512GetSumSqr( src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2Exp( dst : PSingle; src : PSingle; N : Integer);
procedure _AVX512Exp( dst : PSingle; src : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
function _AVX2DotProd( dst : PSingle; src : PSingle; N : Integer ) : Single;
function _AVX512DotProd( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
function _AVX2DotProdInt8( PtrA : PShortInt; PtrB : PSingle; N : Integer ) : Single;
function _AVX512DotProdInt8( PtrA : PShortInt; PtrB : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2MulAddInt8Scalar( dst : PSingle; codes : PShortInt; W : Single; N : Integer );
procedure _AVX512MulAddInt8Scalar( dst : PSingle; codes : PShortInt; W : Single; N : Integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2MulAddInt8( dst : PSingle; src : PSingle; codes : PShortInt; N : Integer );
procedure _AVX512MulAddInt8( dst : PSingle; src : PSingle; codes : PShortInt; N : Integer ); inline;

// Coded by DeepSeek (AI)
function _AVX2MaxAbsFinite( src : PSingle; N : Integer ) : Single;
function _AVX512MaxAbsFinite( src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2QuantizeInt8( dst : PShortInt; src : PSingle; N : Integer; const MaxAbs : Single );
procedure _AVX512QuantizeInt8( dst : PShortInt; src : PSingle; N : Integer; const MaxAbs : Single ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2DequantizeInt8( dst : PSingle; src : PShortInt; N : Integer; const Scale : Single );
procedure _AVX512DequantizeInt8( dst : PSingle; src : PShortInt; N : Integer; const Scale : Single ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2DecodeBF16( dst : PSingle; src : PWord; N : Integer );
procedure _AVX512DecodeBF16( dst : PSingle; src : PWord; N : Integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2ReluGateMask( dst : PSingle; src : PSingle; N : Integer );
procedure _AVX512ReluGateMask( dst : PSingle; src : PSingle; N : Integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2LeakyRelu( dst : PSingle; src : PSingle; N : Integer; const Slope : Single );
procedure _AVX512LeakyRelu( dst : PSingle; src : PSingle; N : Integer; const Slope : Single ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2DecodeF16( dst : PSingle; src : PWord; N : Integer );
procedure _AVX512DecodeF16( dst : PSingle; src : PWord; N : Integer ); inline;

// Coded by DeepSeek (AI)
function _AVX2SumSqrCentered( src : PSingle; Mean : Single; N : Integer ) : Single;
function _AVX512SumSqrCentered( src : PSingle; Mean : Single; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2AdamDelta( PtrDelta, PtrM, PtrV : PSingle;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR : Single;
  NumElements : Integer );
procedure _AVX512AdamDelta( PtrDelta, PtrM, PtrV : PSingle;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR : Single;
  NumElements : Integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2AdafactorDelta( PtrDelta, PtrV : PSingle; Beta2, k, c, Epsilon : Single; NumElements : Integer );
procedure _AVX512AdafactorDelta( PtrDelta, PtrV : PSingle; Beta2, k, c, Epsilon : Single; NumElements : Integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2ClampAbs( PtrA : PSingle; Value : Single; NumElements : Integer );
procedure _AVX512ClampAbs( PtrA : PSingle; Value : Single; NumElements : Integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2LionDelta( PtrDelta, PtrM : PSingle; Beta1, k1, Beta2, k2, NegLR, PosLR : Single; NumElements : Integer );
procedure _AVX512LionDelta( PtrDelta, PtrM : PSingle; Beta1, k1, Beta2, k2, NegLR, PosLR : Single; NumElements : Integer ); inline;

// Coded by DeepSeek (AI)
function _AVX2GetMaxPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single;
function _AVX512GetMaxPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
function _AVX2GetMinPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single;
function _AVX512GetMinPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
function _AVX2GetMaxAbsPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single;
function _AVX512GetMaxAbsPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2AddScalar( PtrA : PSingle; Value : single; N : integer );
procedure _AVX512AddScalar( PtrA : PSingle; Value : single; N : integer ); inline;

// Coded by DeepSeek (AI)
function _AVX2ExpShiftSum( dst : PSingle; src : PSingle; Shift : single; N : integer ) : single;
function _AVX512ExpShiftSum( dst : PSingle; src : PSingle; Shift : single; N : integer ) : single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2Ln( dst : PSingle; src : PSingle; N : integer );
procedure _AVX512Ln( dst : PSingle; src : PSingle; N : integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2SinCos( dst : PSingle; src : PSingle; N : integer; DoCos : integer );
procedure _AVX512SinCos( dst : PSingle; src : PSingle; N : integer; DoCos : integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2EncodeBF16( dst: PSingle; src : PSingle; N: integer);
procedure _AVX512EncodeBF16( dst: PSingle; src : PSingle; N: integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2ReluL(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer);
procedure _AVX512ReluL(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2EncodeF16(dst, src: Pointer; N: integer);
procedure _AVX512EncodeF16(dst, src: Pointer; N: integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2ReluLGateMask(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer);
procedure _AVX512ReluLGateMask(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;



implementation


{-----------------------------------------------------------------------------
  AVX2 memory fill: dst[i] = fact for i = 0..N-1.
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters (Windows ABI): RCX=dst, RDX=N, R8=factor (address).
-----------------------------------------------------------------------------}
procedure _AVX2FillMem( dst : PSingle; N : Integer; const fact : Single);
{$CODEALIGN 16}
asm
  // Load fact from XMM0 and broadcast to all lanes of YMM0
  vbroadcastss ymm0, xmm0

  // Reverse traversal: N = -N, adjust pointer to end
  imul rdx, -4
  sub rcx, rdx

  // Main loop: 32 elements (4 YMM blocks) per iteration
@Loop1:
  add rdx, 128
  jg @loopEnd1
  vmovups [rcx + rdx - 128], ymm0
  vmovups [rcx + rdx - 96],  ymm0
  vmovups [rcx + rdx - 64],  ymm0
  vmovups [rcx + rdx - 32],  ymm0
  jmp @Loop1
@loopEnd1:
  sub rdx, 128
  jz @loop3End

  // 4-element groups (XMM)
@Loop2:
  add rdx, 16
  jg @Loop2End
  vmovups [rcx + rdx - 16], xmm0
  jmp @Loop2
@Loop2End:
  sub rdx, 16
  jz @loop3End

  // Last 0..3 elements
@loop3:
  add rdx, 4
  jg @loop3End
  vmovss [rcx + rdx - 4], xmm0
  jmp @loop3
@loop3End:
  vzeroupper
end;

procedure _AVX512FillMem( dst : PSingle; N : Integer; const fact : Single); inline;
begin
  _AVX2FillMem(dst, N, fact);
end;

{-----------------------------------------------------------------------------
  AVX2 scalar multiply-add: dst[i] = dst[i] + fact * src[i].
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=dst, RDX=src, R8=N, R9=factor (address).
-----------------------------------------------------------------------------}
procedure _AVX2MulAddF( dst : PSingle; src : PSingle; N : Integer; const fact : Single);
asm
  vbroadcastss ymm0, [r9]
  imul r8, -4
  sub rcx, r8
  sub rdx, r8

@Loop1:
  add r8, 128
  jg @loopEnd1
  vmovups ymm1, [rcx + r8 - 128]
  vmovups ymm2, [rdx + r8 - 128]
  vfmadd231ps ymm1, ymm2, ymm0
  vmovups [rcx + r8 - 128], ymm1
  vmovups ymm1, [rcx + r8 - 96]
  vmovups ymm2, [rdx + r8 - 96]
  vfmadd231ps ymm1, ymm2, ymm0
  vmovups [rcx + r8 - 96], ymm1
  vmovups ymm1, [rcx + r8 - 64]
  vmovups ymm2, [rdx + r8 - 64]
  vfmadd231ps ymm1, ymm2, ymm0
  vmovups [rcx + r8 - 64], ymm1
  vmovups ymm1, [rcx + r8 - 32]
  vmovups ymm2, [rdx + r8 - 32]
  vfmadd231ps ymm1, ymm2, ymm0
  vmovups [rcx + r8 - 32], ymm1
  jmp @Loop1
@loopEnd1:
  sub r8, 128
  jz @loop3End

@Loop2:
  add r8, 16
  jg @Loop2End
  vmovups xmm1, [rcx + r8 - 16]
  vmovups xmm2, [rdx + r8 - 16]
  vfmadd231ps xmm1, xmm2, xmm0
  vmovups [rcx + r8 - 16], xmm1
  jmp @Loop2
@Loop2End:
  sub r8, 16
  jz @loop3End

@loop3:
  add r8, 4
  jg @loop3End
  vmovss xmm1, [rcx + r8 - 4]
  vmovss xmm2, [rdx + r8 - 4]
  vfmadd231ss xmm1, xmm2, xmm0
  vmovss [rcx + r8 - 4], xmm1
  jmp @loop3
@loop3End:
  vzeroupper
end;

procedure _AVX512MulAddF( dst : PSingle; src : PSingle; N : Integer; const fact : Single); inline;
begin
  _AVX2MulAddF(dst, src, N, fact);
end;

{-----------------------------------------------------------------------------
  AVX2 scalar multiply-multiply-add: dst[i] = dst[i] * mulOp1 + src[i] * mulOp2.
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=dst, RDX=src, R8=N, R9=mulOp1 (address), [rsp+40]=mulOp2 (address)
  (const parameters are passed on stack in Win64)
-----------------------------------------------------------------------------}
procedure _AVX2MulMulAdd( dst : PSingle; src : PSingle; N : Integer; const mulOp1, mulOp2 : Single);
asm
  // mulOp1 is in R9, mulOp2 is on stack at [rsp+40] (after shadow space)
  mov rax, [rsp+40]        // address of mulOp2
  vbroadcastss ymm5, [r9]
  vbroadcastss ymm6, [rax]

  imul r8, -4
  sub rcx, r8
  sub rdx, r8

@Loop1:
  add r8, 128
  jg @loopEnd1
  vmovups ymm0, [rcx + r8 - 128]
  vmovups ymm1, [rdx + r8 - 128]
  vmulps ymm0, ymm0, ymm5
  vfmadd231ps ymm0, ymm1, ymm6
  vmovups [rcx + r8 - 128], ymm0
  vmovups ymm0, [rcx + r8 - 96]
  vmovups ymm1, [rdx + r8 - 96]
  vmulps ymm0, ymm0, ymm5
  vfmadd231ps ymm0, ymm1, ymm6
  vmovups [rcx + r8 - 96], ymm0
  vmovups ymm0, [rcx + r8 - 64]
  vmovups ymm1, [rdx + r8 - 64]
  vmulps ymm0, ymm0, ymm5
  vfmadd231ps ymm0, ymm1, ymm6
  vmovups [rcx + r8 - 64], ymm0
  vmovups ymm0, [rcx + r8 - 32]
  vmovups ymm1, [rdx + r8 - 32]
  vmulps ymm0, ymm0, ymm5
  vfmadd231ps ymm0, ymm1, ymm6
  vmovups [rcx + r8 - 32], ymm0
  jmp @Loop1
@loopEnd1:
  sub r8, 128
  jz @loop3End

@Loop2:
  add r8, 16
  jg @Loop2End
  vmovups xmm0, [rcx + r8 - 16]
  vmovups xmm1, [rdx + r8 - 16]
  vmulps xmm0, xmm0, xmm5
  vfmadd231ps xmm0, xmm1, xmm6
  vmovups [rcx + r8 - 16], xmm0
  jmp @Loop2
@Loop2End:
  sub r8, 16
  jz @loop3End

@loop3:
  add r8, 4
  jg @loop3End
  vmovss xmm0, [rcx + r8 - 4]
  vmovss xmm1, [rdx + r8 - 4]
  vmulss xmm0, xmm0, xmm5
  vfmadd231ss xmm0, xmm1, xmm6
  vmovss [rcx + r8 - 4], xmm0
  jmp @loop3
@loop3End:
  vzeroupper
end;

procedure _AVX512MulMulAdd( dst : PSingle; src : PSingle; N : Integer; const mulOp1, mulOp2 : Single); inline;
begin
  _AVX2MulMulAdd(dst, src, N, mulOp1, mulOp2);
end;

{-----------------------------------------------------------------------------
  AVX2 triadic multiply-add: dst[i] = dst[i] + src[i] * z[i].
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=dst, RDX=src, R8=z, R9=N.
-----------------------------------------------------------------------------}
procedure _AVX2MulAdd( dst : PSingle; src : PSingle; z : PSingle; N : Integer);
asm
  imul r9, -4
  sub rcx, r9
  sub rdx, r9
  sub r8, r9

@Loop1:
  add r9, 128
  jg @loopEnd1
  vmovups ymm0, [rdx + r9 - 128]
  vmovups ymm1, [r8 + r9 - 128]
  vmulps ymm0, ymm0, ymm1
  vaddps ymm0, ymm0, [rcx + r9 - 128]
  vmovups [rcx + r9 - 128], ymm0
  vmovups ymm0, [rdx + r9 - 96]
  vmovups ymm1, [r8 + r9 - 96]
  vmulps ymm0, ymm0, ymm1
  vaddps ymm0, ymm0, [rcx + r9 - 96]
  vmovups [rcx + r9 - 96], ymm0
  vmovups ymm0, [rdx + r9 - 64]
  vmovups ymm1, [r8 + r9 - 64]
  vmulps ymm0, ymm0, ymm1
  vaddps ymm0, ymm0, [rcx + r9 - 64]
  vmovups [rcx + r9 - 64], ymm0
  vmovups ymm0, [rdx + r9 - 32]
  vmovups ymm1, [r8 + r9 - 32]
  vmulps ymm0, ymm0, ymm1
  vaddps ymm0, ymm0, [rcx + r9 - 32]
  vmovups [rcx + r9 - 32], ymm0
  jmp @Loop1
@loopEnd1:
  sub r9, 128
  jz @loop3End

@Loop2:
  add r9, 16
  jg @Loop2End
  vmovups xmm0, [rdx + r9 - 16]
  vmovups xmm1, [r8 + r9 - 16]
  vmulps xmm0, xmm0, xmm1
  vaddps xmm0, xmm0, [rcx + r9 - 16]
  vmovups [rcx + r9 - 16], xmm0
  jmp @Loop2
@Loop2End:
  sub r9, 16
  jz @loop3End

@loop3:
  add r9, 4
  jg @loop3End
  vmovss xmm0, [rdx + r9 - 4]
  vmovss xmm1, [r8 + r9 - 4]
  vmulss xmm0, xmm0, xmm1
  vaddss xmm0, xmm0, [rcx + r9 - 4]
  vmovss [rcx + r9 - 4], xmm0
  jmp @loop3
@loop3End:
  vzeroupper
end;

{-----------------------------------------------------------------------------
  AVX-512 triadic multiply-add: dst[i] = dst[i] + src[i] * z[i].
  Uses 512-bit ZMM registers, processes 64 elements per iteration.
-----------------------------------------------------------------------------}
procedure _AVX512MulAdd( dst : PSingle; src : PSingle; z : PSingle; N : Integer); inline;
begin
  _AVX2MulAdd(dst, src, z, N);
end;

{-----------------------------------------------------------------------------
  AVX2 ReLU copy: dst[i] = max(0, src[i]).
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=dst, RDX=src, R8=N.
-----------------------------------------------------------------------------}
procedure _AVX2CopyRelu( dst : PSingle; src : PSingle; N : Integer);
asm
  imul r8, -4
  sub rcx, r8
  sub rdx, r8
  vxorps ymm5, ymm5, ymm5

@Loop1:
  add r8, 128
  jg @loopEnd1
  vmovups ymm0, [rdx + r8 - 128]
  vmovups ymm1, [rdx + r8 - 96]
  vmovups ymm2, [rdx + r8 - 64]
  vmovups ymm3, [rdx + r8 - 32]
  vmaxps ymm0, ymm5, ymm0
  vmaxps ymm1, ymm5, ymm1
  vmaxps ymm2, ymm5, ymm2
  vmaxps ymm3, ymm5, ymm3
  vmovups [rcx + r8 - 128], ymm0
  vmovups [rcx + r8 - 96],  ymm1
  vmovups [rcx + r8 - 64],  ymm2
  vmovups [rcx + r8 - 32],  ymm3
  jmp @Loop1
@loopEnd1:
  sub r8, 128
  jz @loop3End

@Loop2:
  add r8, 16
  jg @Loop2End
  vmovups xmm0, [rdx + r8 - 16]
  vmaxps xmm0, xmm5, xmm0
  vmovups [rcx + r8 - 16], xmm0
  jmp @Loop2
@Loop2End:
  sub r8, 16
  jz @loop3End

@loop3:
  add r8, 4
  jg @loop3End
  vmovss xmm0, [rdx + r8 - 4]
  vmaxss xmm0, xmm5, xmm0
  vmovss [rcx + r8 - 4], xmm0
  jmp @loop3
@loop3End:
  vzeroupper
end;

procedure _AVX512CopyRelu( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2CopyRelu(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 scalar multiplication: dst[i] = dst[i] * factor.
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=dst, RDX=N, R8=factor (address).
-----------------------------------------------------------------------------}
procedure _AVX2MulF( dst : PSingle; N : Integer; const factor : Single);
asm
  vbroadcastss ymm7, [r8]
  imul rdx, -4
  sub rcx, rdx

@Loop1:
  add rdx, 128
  jg @loopEnd1
  vmulps ymm2, ymm7, [rcx + rdx - 128]
  vmulps ymm3, ymm7, [rcx + rdx - 96]
  vmulps ymm4, ymm7, [rcx + rdx - 64]
  vmulps ymm5, ymm7, [rcx + rdx - 32]
  vmovups [rcx + rdx - 128], ymm2
  vmovups [rcx + rdx - 96],  ymm3
  vmovups [rcx + rdx - 64],  ymm4
  vmovups [rcx + rdx - 32],  ymm5
  jmp @Loop1
@loopEnd1:
  sub rdx, 128
  jz @loop3End

@Loop2:
  add rdx, 16
  jg @Loop2End
  vmovups xmm2, [rcx + rdx - 16]
  vmulps xmm2, xmm2, xmm7
  vmovups [rcx + rdx - 16], xmm2
  jmp @Loop2
@Loop2End:
  sub rdx, 16
  jz @loop3End

@loop3:
  add rdx, 4
  jg @loop3End
  vmovss xmm2, [rcx + rdx - 4]
  vmulss xmm2, xmm2, xmm7
  vmovss [rcx + rdx - 4], xmm2
  jmp @loop3
@loop3End:
  vzeroupper
end;

{-----------------------------------------------------------------------------
  AVX-512 scalar multiplication: dst[i] = dst[i] * factor.
  Uses 512-bit ZMM registers, processes 64 elements per iteration.
-----------------------------------------------------------------------------}
procedure _AVX512MulF( dst : PSingle; N : Integer; const factor : Single); inline;
begin
  _AVX2MulF(dst, N, factor);
end;

{-----------------------------------------------------------------------------
  AVX2 elementwise multiplication: dst[i] = dst[i] * src[i].
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=dst, RDX=src, R8=N.
-----------------------------------------------------------------------------}
procedure _AVX2Mul( dst : PSingle; src : PSingle; N : Integer);
asm
  imul r8, -4
  sub rcx, r8
  sub rdx, r8

@Loop1:
  add r8, 128
  jg @loopEnd1
  vmovups ymm0, [rdx + r8 - 128]
  vmovups ymm1, [rdx + r8 - 96]
  vmovups ymm2, [rdx + r8 - 64]
  vmovups ymm3, [rdx + r8 - 32]
  vmulps ymm0, ymm0, [rcx + r8 - 128]
  vmulps ymm1, ymm1, [rcx + r8 - 96]
  vmulps ymm2, ymm2, [rcx + r8 - 64]
  vmulps ymm3, ymm3, [rcx + r8 - 32]
  vmovups [rcx + r8 - 128], ymm0
  vmovups [rcx + r8 - 96],  ymm1
  vmovups [rcx + r8 - 64],  ymm2
  vmovups [rcx + r8 - 32],  ymm3
  jmp @Loop1
@loopEnd1:
  sub r8, 128
  jz @loop3End

@Loop2:
  add r8, 16
  jg @Loop2End
  vmovups xmm0, [rdx + r8 - 16]
  vmulps xmm0, xmm0, [rcx + r8 - 16]
  vmovups [rcx + r8 - 16], xmm0
  jmp @Loop2
@Loop2End:
  sub r8, 16
  jz @loop3End

@loop3:
  add r8, 4
  jg @loop3End
  vmovss xmm0, [rdx + r8 - 4]
  vmulss xmm0, xmm0, [rcx + r8 - 4]
  vmovss [rcx + r8 - 4], xmm0
  jmp @loop3
@loop3End:
  vzeroupper
end;

procedure _AVX512Mul( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2Mul(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 elementwise addition: dst[i] = dst[i] + src[i].
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=dst, RDX=src, R8=N.
-----------------------------------------------------------------------------}
procedure _AVX2Add( dst : PSingle; src : PSingle; N : Integer);
asm
  imul r8, -4
  sub rcx, r8
  sub rdx, r8

@Loop1:
  add r8, 128
  jg @loopEnd1
  vmovups ymm0, [rdx + r8 - 128]
  vmovups ymm1, [rdx + r8 - 96]
  vmovups ymm2, [rdx + r8 - 64]
  vmovups ymm3, [rdx + r8 - 32]
  vaddps ymm0, ymm0, [rcx + r8 - 128]
  vaddps ymm1, ymm1, [rcx + r8 - 96]
  vaddps ymm2, ymm2, [rcx + r8 - 64]
  vaddps ymm3, ymm3, [rcx + r8 - 32]
  vmovups [rcx + r8 - 128], ymm0
  vmovups [rcx + r8 - 96],  ymm1
  vmovups [rcx + r8 - 64],  ymm2
  vmovups [rcx + r8 - 32],  ymm3
  jmp @Loop1
@loopEnd1:
  sub r8, 128
  jz @loop3End

@Loop2:
  add r8, 16
  jg @Loop2End
  vmovups xmm0, [rdx + r8 - 16]
  vaddps xmm0, xmm0, [rcx + r8 - 16]
  vmovups [rcx + r8 - 16], xmm0
  jmp @Loop2
@Loop2End:
  sub r8, 16
  jz @loop3End

@loop3:
  add r8, 4
  jg @loop3End
  vmovss xmm0, [rdx + r8 - 4]
  vaddss xmm0, xmm0, [rcx + r8 - 4]
  vmovss [rcx + r8 - 4], xmm0
  jmp @loop3
@loop3End:
  vzeroupper
end;

procedure _AVX512Add( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2Add(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 elementwise max: dst[i] = max(dst[i], src[i]).
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=dst, RDX=src, R8=N.
-----------------------------------------------------------------------------}
procedure _AVX2Max( dst : PSingle; src : PSingle; N : Integer);
asm
  imul r8, -4
  sub rcx, r8
  sub rdx, r8

@Loop1:
  add r8, 128
  jg @loopEnd1
  vmovups ymm0, [rcx + r8 - 128]
  vmovups ymm1, [rcx + r8 - 96]
  vmovups ymm2, [rcx + r8 - 64]
  vmovups ymm3, [rcx + r8 - 32]
  vmaxps ymm0, ymm0, [rdx + r8 - 128]
  vmaxps ymm1, ymm1, [rdx + r8 - 96]
  vmaxps ymm2, ymm2, [rdx + r8 - 64]
  vmaxps ymm3, ymm3, [rdx + r8 - 32]
  vmovups [rcx + r8 - 128], ymm0
  vmovups [rcx + r8 - 96],  ymm1
  vmovups [rcx + r8 - 64],  ymm2
  vmovups [rcx + r8 - 32],  ymm3
  jmp @Loop1
@loopEnd1:
  sub r8, 128
  jz @loop3End

@Loop2:
  add r8, 16
  jg @Loop2End
  vmovups xmm0, [rcx + r8 - 16]
  vmaxps xmm0, xmm0, [rdx + r8 - 16]
  vmovups [rcx + r8 - 16], xmm0
  jmp @Loop2
@Loop2End:
  sub r8, 16
  jz @loop3End

@loop3:
  add r8, 4
  jg @loop3End
  vmovss xmm0, [rcx + r8 - 4]
  vmaxss xmm0, xmm0, [rdx + r8 - 4]
  vmovss [rcx + r8 - 4], xmm0
  jmp @loop3
@loop3End:
  vzeroupper
end;

procedure _AVX512Max( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2Max(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Sum of Absolute Differences: result = sum_i |dst[i] - src[i]|.
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=dst, RDX=src, R8=N.
-----------------------------------------------------------------------------}
function _AVX2SumDiff( dst : PSingle; src : PSingle; N : Integer ) : Single;
asm
  imul r8, -4
  sub rcx, r8
  sub rdx, r8
  vxorps ymm0, ymm0, ymm0
  vpcmpeqd ymm1, ymm1, ymm1
  vpsrld ymm1, ymm1, 1

@Loop1:
  add r8, 128
  jg @loopEnd1
  vmovups ymm2, [rcx + r8 - 128]
  vmovups ymm3, [rdx + r8 - 128]
  vsubps ymm2, ymm2, ymm3
  vandps ymm2, ymm2, ymm1
  vmovups ymm3, [rcx + r8 - 96]
  vmovups ymm4, [rdx + r8 - 96]
  vsubps ymm3, ymm3, ymm4
  vandps ymm3, ymm3, ymm1
  vmovups ymm4, [rcx + r8 - 64]
  vmovups ymm5, [rdx + r8 - 64]
  vsubps ymm4, ymm4, ymm5
  vandps ymm4, ymm4, ymm1
  vmovups ymm5, [rcx + r8 - 32]
  vmovups ymm6, [rdx + r8 - 32]
  vsubps ymm5, ymm5, ymm6
  vandps ymm5, ymm5, ymm1
  vaddps ymm0, ymm0, ymm2
  vaddps ymm0, ymm0, ymm3
  vaddps ymm0, ymm0, ymm4
  vaddps ymm0, ymm0, ymm5
  jmp @Loop1
@loopEnd1:
  sub r8, 128
  jz @loop3End

@Loop2:
  add r8, 16
  jg @Loop2End
  vmovups xmm2, [rcx + r8 - 16]
  vmovups xmm3, [rdx + r8 - 16]
  vsubps xmm2, xmm2, xmm3
  vandps xmm2, xmm2, xmm1
  vaddps xmm0, xmm0, xmm2
  jmp @Loop2
@Loop2End:
  sub r8, 16
  jz @loop3End

@loop3:
  add r8, 4
  jg @loop3End
  vmovss xmm2, [rcx + r8 - 4]
  vmovss xmm3, [rdx + r8 - 4]
  vsubss xmm2, xmm2, xmm3
  vandps xmm2, xmm2, xmm1
  vaddss xmm0, xmm0, xmm2
  jmp @loop3
@loop3End:
  vextractf128 xmm2, ymm0, 1
  vaddps xmm0, xmm0, xmm2
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
  movss [Result], xmm0
end;

function _AVX512SumDiff( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX512SumDiff(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Squared Euclidean Distance: result = sum_i (dst[i] - src[i])^2.
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=dst, RDX=src, R8=N.
-----------------------------------------------------------------------------}
function _AVX2DistanceSqr( dst : PSingle; src : PSingle; N : Integer ) : Single;
asm
  imul r8, -4
  sub rcx, r8
  sub rdx, r8
  vxorps ymm0, ymm0, ymm0

@Loop1:
  add r8, 128
  jg @loopEnd1
  vmovups ymm2, [rcx + r8 - 128]
  vmovups ymm3, [rdx + r8 - 128]
  vsubps ymm2, ymm2, ymm3
  vmulps ymm2, ymm2, ymm2
  vmovups ymm3, [rcx + r8 - 96]
  vmovups ymm4, [rdx + r8 - 96]
  vsubps ymm3, ymm3, ymm4
  vmulps ymm3, ymm3, ymm3
  vmovups ymm4, [rcx + r8 - 64]
  vmovups ymm5, [rdx + r8 - 64]
  vsubps ymm4, ymm4, ymm5
  vmulps ymm4, ymm4, ymm4
  vmovups ymm5, [rcx + r8 - 32]
  vmovups ymm6, [rdx + r8 - 32]
  vsubps ymm5, ymm5, ymm6
  vmulps ymm5, ymm5, ymm5
  vaddps ymm0, ymm0, ymm2
  vaddps ymm0, ymm0, ymm3
  vaddps ymm0, ymm0, ymm4
  vaddps ymm0, ymm0, ymm5
  jmp @Loop1
@loopEnd1:
  sub r8, 128
  jz @loop3End

@Loop2:
  add r8, 16
  jg @Loop2End
  vmovups xmm2, [rcx + r8 - 16]
  vmovups xmm3, [rdx + r8 - 16]
  vsubps xmm2, xmm2, xmm3
  vmulps xmm2, xmm2, xmm2
  vaddps xmm0, xmm0, xmm2
  jmp @Loop2
@Loop2End:
  sub r8, 16
  jz @loop3End

@loop3:
  add r8, 4
  jg @loop3End
  vmovss xmm2, [rcx + r8 - 4]
  vmovss xmm3, [rdx + r8 - 4]
  vsubss xmm2, xmm2, xmm3
  vmulss xmm2, xmm2, xmm2
  vaddss xmm0, xmm0, xmm2
  jmp @loop3
@loop3End:
  vextractf128 xmm2, ymm0, 1
  vaddps xmm0, xmm0, xmm2
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
  movss [Result], xmm0
end;

function _AVX512DistanceSqr( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2DistanceSqr(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 elementwise subtraction: dst[i] = dst[i] - src[i].
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=dst, RDX=src, R8=N.
-----------------------------------------------------------------------------}
procedure _AVX2SubPair( dst : PSingle; src : PSingle; N : Integer);
asm
  imul r8, -4
  sub rcx, r8
  sub rdx, r8

@Loop1:
  add r8, 128
  jg @loopEnd1
  vmovups ymm0, [rcx + r8 - 128]
  vmovups ymm1, [rdx + r8 - 128]
  vsubps ymm0, ymm0, ymm1
  vmovups [rcx + r8 - 128], ymm0
  vmovups ymm0, [rcx + r8 - 96]
  vmovups ymm1, [rdx + r8 - 96]
  vsubps ymm0, ymm0, ymm1
  vmovups [rcx + r8 - 96], ymm0
  vmovups ymm0, [rcx + r8 - 64]
  vmovups ymm1, [rdx + r8 - 64]
  vsubps ymm0, ymm0, ymm1
  vmovups [rcx + r8 - 64], ymm0
  vmovups ymm0, [rcx + r8 - 32]
  vmovups ymm1, [rdx + r8 - 32]
  vsubps ymm0, ymm0, ymm1
  vmovups [rcx + r8 - 32], ymm0
  jmp @Loop1
@loopEnd1:
  sub r8, 128
  jz @loop3End

@Loop2:
  add r8, 16
  jg @Loop2End
  vmovups xmm0, [rcx + r8 - 16]
  vmovups xmm1, [rdx + r8 - 16]
  vsubps xmm0, xmm0, xmm1
  vmovups [rcx + r8 - 16], xmm0
  jmp @Loop2
@Loop2End:
  sub r8, 16
  jz @loop3End

@loop3:
  add r8, 4
  jg @loop3End
  vmovss xmm0, [rcx + r8 - 4]
  vmovss xmm1, [rdx + r8 - 4]
  vsubss xmm0, xmm0, xmm1
  vmovss [rcx + r8 - 4], xmm0
  jmp @loop3
@loop3End:
  vzeroupper
end;

procedure _AVX512SubPair( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2SubPair(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Sum of elements: result = sum_i src[i].
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=src, RDX=N.
-----------------------------------------------------------------------------}
function _AVX2GetSum( src : PSingle; N : Integer ) : Single;
asm
  imul rdx, -4
  sub rcx, rdx
  vxorps ymm0, ymm0, ymm0
  vxorps ymm1, ymm1, ymm1
  vxorps ymm2, ymm2, ymm2
  vxorps ymm3, ymm3, ymm3

@Loop1:
  add rdx, 128
  jg @loopEnd1
  vaddps ymm0, ymm0, [rcx + rdx - 128]
  vaddps ymm1, ymm1, [rcx + rdx - 96]
  vaddps ymm2, ymm2, [rcx + rdx - 64]
  vaddps ymm3, ymm3, [rcx + rdx - 32]
  jmp @Loop1
@loopEnd1:
  sub rdx, 128
  jz @loop3End

@Loop2:
  add rdx, 16
  jg @Loop2End
  vmovups xmm4, [rcx + rdx - 16]
  vaddps xmm0, xmm0, xmm4
  jmp @Loop2
@Loop2End:
  sub rdx, 16
  jz @loop3End

@loop3:
  add rdx, 4
  jg @loop3End
  vmovss xmm4, [rcx + rdx - 4]
  vaddss xmm0, xmm0, xmm4
  jmp @loop3
@loop3End:
  vaddps ymm0, ymm0, ymm1
  vaddps ymm2, ymm2, ymm3
  vaddps ymm0, ymm0, ymm2
  vextractf128 xmm4, ymm0, 1
  vaddps xmm0, xmm0, xmm4
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
  movss [Result], xmm0
end;

function _AVX512GetSum( src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2GetSum(src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 sum of squares: result = sum_i src[i]^2.
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters: RCX=src, RDX=N.
-----------------------------------------------------------------------------}
function _AVX2GetSumSqr( src : PSingle; N : Integer ) : Single;
asm
  imul rdx, -4
  sub rcx, rdx
  vxorps ymm0, ymm0, ymm0
  vxorps ymm1, ymm1, ymm1
  vxorps ymm2, ymm2, ymm2
  vxorps ymm3, ymm3, ymm3

@Loop1:
  add rdx, 128
  jg @loopEnd1
  vmovups ymm4, [rcx + rdx - 128]
  vmovups ymm5, [rcx + rdx - 96]
  vmovups ymm6, [rcx + rdx - 64]
  vmovups ymm7, [rcx + rdx - 32]
  vfmadd231ps ymm0, ymm4, ymm4
  vfmadd231ps ymm1, ymm5, ymm5
  vfmadd231ps ymm2, ymm6, ymm6
  vfmadd231ps ymm3, ymm7, ymm7
  jmp @Loop1
@loopEnd1:
  sub rdx, 128
  jz @loop3End

@Loop2:
  add rdx, 16
  jg @Loop2End
  vmovups xmm4, [rcx + rdx - 16]
  vmulps xmm4, xmm4, xmm4
  vaddps xmm0, xmm0, xmm4
  jmp @Loop2
@Loop2End:
  sub rdx, 16
  jz @loop3End

@loop3:
  add rdx, 4
  jg @loop3End
  vmovss xmm4, [rcx + rdx - 4]
  vmulss xmm4, xmm4, xmm4
  vaddss xmm0, xmm0, xmm4
  jmp @loop3
@loop3End:
  vaddps ymm0, ymm0, ymm1
  vaddps ymm2, ymm2, ymm3
  vaddps ymm0, ymm0, ymm2
  vextractf128 xmm4, ymm0, 1
  vaddps xmm0, xmm0, xmm4
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
  movss [Result], xmm0
end;

function _AVX512GetSumSqr( src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2GetSumSqr(src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 exponential: dst[i] = exp(src[i]).
  Pure assembly, processes all elements.
  Uses 8-wide YMM polynomial for bulk, scalar for tail (0..7).
  Parameters: RCX=dst, RDX=src, R8=N.
-----------------------------------------------------------------------------}
procedure _AVX2Exp( dst : PSingle; src : PSingle; N : Integer);
asm
  test r8, r8
  jle @Exit

  mov rax, r8
  and rax, 7
  sub r8, rax
  jz @Tail

  vbroadcastss ymm6, [cAVXLog2e]
  vbroadcastss ymm7, [cAVXLn2]
  vpbroadcastd ymm5, [cAVXExp127]

  mov r9, r8
  shr r9, 3
@BulkLoop:
  vmovups ymm0, [rdx]
  vbroadcastss ymm1, [cAVXExpHi]
  vminps ymm0, ymm0, ymm1
  vbroadcastss ymm1, [cAVXExpLo]
  vmaxps ymm0, ymm0, ymm1

  vmulps ymm1, ymm0, ymm6
  vroundps ymm2, ymm1, 0
  vsubps ymm1, ymm1, ymm2

  vmulps ymm3, ymm1, ymm7

  vbroadcastss ymm4, [cAVXExpP6]
  vbroadcastss ymm0, [cAVXExpP5]
  vfmadd213ps ymm4, ymm3, ymm0
  vbroadcastss ymm0, [cAVXExpP4]
  vfmadd213ps ymm4, ymm3, ymm0
  vbroadcastss ymm0, [cAVXExpP3]
  vfmadd213ps ymm4, ymm3, ymm0
  vbroadcastss ymm0, [cAVXExpP2]
  vfmadd213ps ymm4, ymm3, ymm0
  vbroadcastss ymm0, [cAVXExpP1]
  vfmadd213ps ymm4, ymm3, ymm0
  vbroadcastss ymm0, [cAVXExpP0]
  vfmadd213ps ymm4, ymm3, ymm0

  vcvtps2dq ymm2, ymm2
  vpaddd ymm2, ymm2, ymm5
  vpslld ymm2, ymm2, 23

  vmulps ymm0, ymm4, ymm2
  vmovups [rcx], ymm0

  add rdx, 32
  add rcx, 32
  dec r9
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Exit

  vbroadcastss xmm6, [cAVXExpHi]
  vbroadcastss xmm7, [cAVXExpLo]
  vbroadcastss xmm8, [cAVXLog2e]
  vbroadcastss xmm9, [cAVXLn2]
  vpbroadcastd xmm5, [cAVXExp127]

@TailLoop:
  vmovss xmm0, [rdx]
  vminss xmm0, xmm0, xmm6
  vmaxss xmm0, xmm0, xmm7

  vmulss xmm1, xmm0, xmm8
  vroundss xmm2, xmm1, xmm1, 0
  vsubss xmm1, xmm1, xmm2

  vmulss xmm3, xmm1, xmm9

  vbroadcastss xmm4, [cAVXExpP6]
  vfmadd213ss xmm4, xmm3, [cAVXExpP5]
  vfmadd213ss xmm4, xmm3, [cAVXExpP4]
  vfmadd213ss xmm4, xmm3, [cAVXExpP3]
  vfmadd213ss xmm4, xmm3, [cAVXExpP2]
  vfmadd213ss xmm4, xmm3, [cAVXExpP1]
  vfmadd213ss xmm4, xmm3, [cAVXExpP0]

  vcvtss2si r9d, xmm2        // 使用 r9d
  add r9d, 127
  shl r9d, 23
  movd xmm2, r9d             // 现在匹配

  vmulss xmm0, xmm4, xmm2
  vmovss [rcx], xmm0

  add rdx, 4
  add rcx, 4
  dec rax
  jnz @TailLoop

@Exit:
  vzeroupper
end;

procedure _AVX512Exp( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2Exp(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 dot product: result = sum_i dst[i] * src[i].
  Uses 256-bit YMM with FMA, processes 32 elements per iteration.
  Tail: XMM groups then scalar.
  Parameters: RCX=dst, RDX=src, R8=N.
-----------------------------------------------------------------------------}
function _AVX2DotProd( dst : PSingle; src : PSingle; N : Integer ) : Single;
asm
  test r8, r8
  jle @Zero

  // bulk = N - (N mod 32)
  mov rax, r8
  and rax, 31
  sub r8, rax

  vxorps ymm0, ymm0, ymm0
  vxorps ymm1, ymm1, ymm1
  vxorps ymm2, ymm2, ymm2
  vxorps ymm3, ymm3, ymm3

  mov r9, r8
  shr r9, 5
  jz @Tail

@BulkLoop:
  vmovups ymm4, [rcx]
  vfmadd231ps ymm0, ymm4, [rdx]
  vmovups ymm5, [rcx+32]
  vfmadd231ps ymm1, ymm5, [rdx+32]
  vmovups ymm6, [rcx+64]
  vfmadd231ps ymm2, ymm6, [rdx+64]
  vmovups ymm7, [rcx+96]
  vfmadd231ps ymm3, ymm7, [rdx+96]
  add rcx, 128
  add rdx, 128
  dec r9
  jnz @BulkLoop

  vaddps ymm0, ymm0, ymm1
  vaddps ymm2, ymm2, ymm3
  vaddps ymm0, ymm0, ymm2
  vextractf128 xmm4, ymm0, 1
  vaddps xmm0, xmm0, xmm4
  vzeroupper

@Tail:
  // rax = tail count (0..31)
  mov r8, rax
  and r8, 3
  sub rax, r8

  mov r9, rax
  shr r9, 2
  jz @ScalarTail

@XmmLoop:
  vmovups xmm4, [rcx]
  vmulps xmm4, xmm4, [rdx]
  vaddps xmm0, xmm0, xmm4
  add rcx, 16
  add rdx, 16
  dec r9
  jnz @XmmLoop

@ScalarTail:
  test r8, r8
  jz @Finish

  vmovss xmm4, [rcx]
  vmulss xmm4, xmm4, [rdx]
  vaddss xmm0, xmm0, xmm4
  add rcx, 4
  add rdx, 4
  dec r8
  jnz @ScalarTail

@Finish:
  // Horizontal sum: xmm0[0] = sum of all 4 components
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  // Return value is in XMM0, no need to write to memory
  ret

@Zero:
  xorps xmm0, xmm0
  ret
end;

function _AVX512DotProd( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2DotProd(dst, src, N);
end;


{-----------------------------------------------------------------------------
  AVX2 int8 dot product: result = sum_i PtrA[i] * PtrB[i].
  PtrA points to signed bytes (int8), PtrB to Single floats.
  Processes 32 elements per loop (4 YMM blocks), then XMM groups and scalar tail.
  Parameters: RCX=PtrA, RDX=PtrB, R8=N.
-----------------------------------------------------------------------------}
function _AVX2DotProdInt8( PtrA : PShortInt; PtrB : PSingle; N : Integer ) : Single;
asm
  test r8, r8
  jle @Zero

  // Bulk = N - (N mod 32)
  mov rax, r8
  and rax, 31                  // tail = N mod 32
  sub r8, rax                  // bulk = N - tail

  vxorps ymm0, ymm0, ymm0
  vxorps ymm1, ymm1, ymm1
  vxorps ymm6, ymm6, ymm6
  vxorps ymm7, ymm7, ymm7

  mov r9, r8
  shr r9, 5                    // number of 32-element blocks
  jz @Tail

@BulkLoop:
  // Load 4 groups of 8 bytes (32 bytes total) -> sign-extend to dwords -> float
  vpmovsxbd ymm2, [rcx]
  vpmovsxbd ymm3, [rcx+8]
  vpmovsxbd ymm4, [rcx+16]
  vpmovsxbd ymm5, [rcx+24]

  vcvtdq2ps ymm2, ymm2
  vcvtdq2ps ymm3, ymm3
  vcvtdq2ps ymm4, ymm4
  vcvtdq2ps ymm5, ymm5

  vfmadd231ps ymm0, ymm2, [rdx]
  vfmadd231ps ymm1, ymm3, [rdx+32]
  vfmadd231ps ymm6, ymm4, [rdx+64]
  vfmadd231ps ymm7, ymm5, [rdx+96]

  add rcx, 32                  // advance int8 pointer by 32 bytes
  add rdx, 128                 // advance float pointer by 32 elements
  dec r9
  jnz @BulkLoop

  // Reduce YMM accumulators
  vaddps ymm0, ymm0, ymm1
  vaddps ymm6, ymm6, ymm7
  vaddps ymm0, ymm0, ymm6
  vextractf128 xmm2, ymm0, 1
  vaddps xmm0, xmm0, xmm2
  vzeroupper

@Tail:
  // rax = tail count (0..31)
  mov r8, rax
  and r8, 3                    // scalar remainder
  sub rax, r8                  // groups of 4 for XMM

  mov r9, rax
  shr r9, 2                    // number of 4-element groups
  jz @ScalarTail

@XmmLoop:
  vpmovsxbd xmm2, [rcx]
  vcvtdq2ps xmm2, xmm2
  vmovups xmm3, [rdx]
  vmulps xmm2, xmm2, xmm3
  vaddps xmm0, xmm0, xmm2
  add rcx, 4
  add rdx, 16
  dec r9
  jnz @XmmLoop

@ScalarTail:
  test r8, r8
  jz @Finish

  movsx eax, byte ptr [rcx]   // load int8 sign-extended to 32-bit
  vcvtsi2ss xmm2, xmm2, eax
  vmovss xmm3, [rdx]
  vmulss xmm2, xmm2, xmm3
  vaddss xmm0, xmm0, xmm2
  add rcx, 1
  add rdx, 4
  dec r8
  jnz @ScalarTail

@Finish:
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  movss [Result], xmm0
  ret

@Zero:
  xorps xmm0, xmm0
  movss [Result], xmm0
  ret
end;

function _AVX512DotProdInt8( PtrA : PShortInt; PtrB : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2DotProdInt8(PtrA, PtrB, N);
end;

{-----------------------------------------------------------------------------
  AVX2 int8 scalar multiply-add: dst[i] += W * codes[i].
  codes is int8 (signed byte), W is Single scalar.
  Processes 32 elements per loop, then XMM groups and scalar tail.
  Parameters (Windows): RCX=dst, RDX=codes, XMM2=W, R8=N.
  Linux: RDI=dst, RSI=codes, XMM0=W, RDX=N (not supported here; for simplicity, this unit assumes Win64 ABI).
-----------------------------------------------------------------------------}
procedure _AVX2MulAddInt8Scalar( dst : PSingle; codes : PShortInt; W : Single; N : Integer );
asm
  test r8, r8
  jle @Exit

  // Broadcast W from XMM2 to YMM5
  vbroadcastss ymm5, xmm2

  // Bulk = N - (N mod 32)
  mov rax, r8
  and rax, 31
  sub r8, rax

  mov r9, r8
  shr r9, 5
  jz @Tail

@BulkLoop:
  vpmovsxbd ymm0, [rdx]
  vpmovsxbd ymm1, [rdx+8]
  vpmovsxbd ymm2, [rdx+16]
  vpmovsxbd ymm3, [rdx+24]

  vcvtdq2ps ymm0, ymm0
  vcvtdq2ps ymm1, ymm1
  vcvtdq2ps ymm2, ymm2
  vcvtdq2ps ymm3, ymm3

  vmovups ymm6, [rcx]
  vmovups ymm7, [rcx+32]
  vfmadd231ps ymm6, ymm0, ymm5
  vfmadd231ps ymm7, ymm1, ymm5
  vmovups [rcx], ymm6
  vmovups [rcx+32], ymm7

  vmovups ymm6, [rcx+64]
  vmovups ymm7, [rcx+96]
  vfmadd231ps ymm6, ymm2, ymm5
  vfmadd231ps ymm7, ymm3, ymm5
  vmovups [rcx+64], ymm6
  vmovups [rcx+96], ymm7

  add rdx, 32
  add rcx, 128
  dec r9
  jnz @BulkLoop

@Tail:
  // rax = tail count (0..31)
  mov r8, rax
  and r8, 3
  sub rax, r8

  mov r9, rax
  shr r9, 2
  jz @ScalarTail

@XmmLoop:
  vpmovsxbd xmm0, [rdx]
  vcvtdq2ps xmm0, xmm0
  vmovups xmm6, [rcx]
  vfmadd231ps xmm6, xmm0, xmm5
  vmovups [rcx], xmm6
  add rdx, 4
  add rcx, 16
  dec r9
  jnz @XmmLoop

@ScalarTail:
  test r8, r8
  jz @Exit

  movsx eax, byte ptr [rdx]
  vcvtsi2ss xmm0, xmm0, eax
  vmulss xmm0, xmm0, xmm5
  vmovss xmm1, [rcx]
  vaddss xmm1, xmm1, xmm0
  vmovss [rcx], xmm1
  add rdx, 1
  add rcx, 4
  dec r8
  jnz @ScalarTail

@Exit:
  vzeroupper
end;

procedure _AVX512MulAddInt8Scalar( dst : PSingle; codes : PShortInt; W : Single; N : Integer ); inline;
begin
  _AVX2MulAddInt8Scalar(dst, codes, W, N);
end;

{-----------------------------------------------------------------------------
  AVX2 int8 elementwise multiply-add (64-bit fallback).
  Parameters: RCX=dst, RDX=src, R8=codes, R9=N.
-----------------------------------------------------------------------------}
procedure _AVX2MulAddInt8( dst : PSingle; src : PSingle; codes : PShortInt; N : Integer );
asm
  test r9, r9
  jle @Exit

  mov rax, r9
  and rax, 31
  sub r9, rax

  mov r10, r9
  shr r10, 5
  jz @Tail

@BulkLoop:
  vpmovsxbd ymm0, [r8]
  vpmovsxbd ymm1, [r8+8]
  vpmovsxbd ymm2, [r8+16]
  vpmovsxbd ymm3, [r8+24]

  vcvtdq2ps ymm0, ymm0
  vcvtdq2ps ymm1, ymm1
  vcvtdq2ps ymm2, ymm2
  vcvtdq2ps ymm3, ymm3

  vmovups ymm4, [rcx]
  vfmadd231ps ymm4, ymm0, [rdx]
  vmovups [rcx], ymm4

  vmovups ymm4, [rcx+32]
  vfmadd231ps ymm4, ymm1, [rdx+32]
  vmovups [rcx+32], ymm4

  vmovups ymm4, [rcx+64]
  vfmadd231ps ymm4, ymm2, [rdx+64]
  vmovups [rcx+64], ymm4

  vmovups ymm4, [rcx+96]
  vfmadd231ps ymm4, ymm3, [rdx+96]
  vmovups [rcx+96], ymm4

  add r8, 32
  add rdx, 128
  add rcx, 128
  dec r10
  jnz @BulkLoop

  vzeroupper

@Tail:
  mov r9, rax
  and r9, 3
  sub rax, r9

  mov r10, rax
  shr r10, 2
  jz @ScalarTail

@XmmLoop:
  vpmovsxbd xmm0, [r8]
  vcvtdq2ps xmm0, xmm0
  vmovups xmm4, [rcx]
  vfmadd231ps xmm4, xmm0, [rdx]
  vmovups [rcx], xmm4
  add r8, 4
  add rdx, 16
  add rcx, 16
  dec r10
  jnz @XmmLoop

@ScalarTail:
  test r9, r9
  jz @Exit

  movsx eax, byte ptr [r8]
  vcvtsi2ss xmm0, xmm0, eax
  vmovss xmm1, [rdx]
  vmulss xmm0, xmm0, xmm1
  vmovss xmm1, [rcx]
  vaddss xmm1, xmm1, xmm0
  vmovss [rcx], xmm1
  add r8, 1
  add rdx, 4
  add rcx, 4
  dec r9
  jnz @ScalarTail

@Exit:
  vzeroupper
end;

procedure _AVX512MulAddInt8( dst : PSingle; src : PSingle; codes : PShortInt; N : Integer ); inline;
begin
  _AVX2MulAddInt8(dst, src, codes, N);
end;

{-----------------------------------------------------------------------------
  AVX2 max absolute finite (64-bit). Same as 32-bit but using 64-bit regs.
  Parameters: RCX=src, RDX=N.
-----------------------------------------------------------------------------}
function _AVX2MaxAbsFinite( src : PSingle; N : Integer ) : Single;
asm
  test rdx, rdx
  jle @Zero

  // Load constants
  mov rax, $7FFFFFFF
  movd xmm3, eax
  vbroadcastss ymm3, xmm3
  mov rax, $7F7FFFFF
  movd xmm2, eax
  vbroadcastss ymm2, xmm2

  // Bulk = N - (N mod 8)
  mov rax, rdx
  and rax, 7
  sub rdx, rax                 // rdx = bulk

  vxorps ymm4, ymm4, ymm4
  mov r8, rdx
  shr r8, 3
  jz @Tail

@BulkLoop:
  vmovups ymm0, [rcx]
  vandps ymm0, ymm0, ymm3
  vcmpps ymm1, ymm0, ymm2, 18
  vandps ymm0, ymm0, ymm1
  vmaxps ymm4, ymm4, ymm0
  add rcx, 32
  dec r8
  jnz @BulkLoop

  vextractf128 xmm0, ymm4, 1
  vmaxps xmm0, xmm0, xmm4
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  movss [Result], xmm0
  vzeroupper

@Tail:
  test rax, rax
  jz @Done

  mov r8d, $7FFFFFFF
  movd xmm3, r8d
  mov r8d, $7F7FFFFF
  movd xmm2, r8d
  movss xmm4, [Result]         // initial max

  xor r9, r9
@TailLoop:
  vmovss xmm0, [rcx + r9*4]
  vandps xmm0, xmm0, xmm3
  movss xmm1, xmm0
  cmpss xmm1, xmm2, 18
  andps xmm0, xmm1
  maxss xmm4, xmm0
  inc r9
  cmp r9, rax
  jl @TailLoop

  movss [Result], xmm4

@Done:
  vzeroupper
  ret

@Zero:
  xorps xmm0, xmm0
  movss [Result], xmm0
  ret
end;

function _AVX512MaxAbsFinite( src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2MaxAbsFinite(src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 int8 quantization (64-bit). Same as 32-bit but with 64-bit registers.
  Parameters: RCX=dst, RDX=src, R8=N, XMM3=MaxAbs.
-----------------------------------------------------------------------------}
procedure _AVX2QuantizeInt8( dst : PShortInt; src : PSingle; N : Integer; const MaxAbs : Single );
asm
  test r8, r8
  jle @Exit

  // Allocate local stack space for constants
  sub rsp, 32
  // [rsp+0] = Recip, [rsp+4] = 127, [rsp+8] = -127

  // Compute reciprocal of MaxAbs (in xmm3) using x87
  movss [rsp+16], xmm3         // store MaxAbs
  fld dword ptr [rsp+16]
  fld1
  fdivp
  fstp dword ptr [rsp]         // store Recip

  mov dword ptr [rsp+4], 127.0
  mov dword ptr [rsp+8], -127.0

  // Load constants into YMM
  lea rax, [rsp]
  vbroadcastss ymm5, [rax]     // Recip
  lea rax, [rsp+4]
  vbroadcastss ymm6, [rax]     // 127
  lea rax, [rsp+8]
  vbroadcastss ymm7, [rax]     // -127

  // Bulk = N - (N mod 8)
  mov rax, r8
  and rax, 7
  sub r8, rax

  mov r9, r8
  shr r9, 3
  jz @Tail

@BulkLoop:
  vmovups ymm0, [rdx]
  vcmpps ymm1, ymm0, ymm0, 7
  vandps ymm0, ymm0, ymm1
  vmulps ymm0, ymm0, ymm5
  vmulps ymm0, ymm0, ymm6
  vminps ymm0, ymm0, ymm6
  vmaxps ymm0, ymm0, ymm7
  vcvtps2dq ymm0, ymm0
  vextracti128 xmm1, ymm0, 1
  vpackssdw xmm0, xmm0, xmm1
  vpacksswb xmm0, xmm0, xmm0
  vmovq qword ptr [rcx], xmm0
  add rdx, 32
  add rcx, 8
  dec r9
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Done

  movss xmm5, [rsp]            // Recip
  movss xmm6, [rsp+4]          // 127
  movss xmm7, [rsp+8]          // -127

  xor r10, r10
@TailLoop:
  vmovss xmm0, [rdx + r10*4]
  vcmpss xmm1, xmm0, xmm0, 7
  vandps xmm0, xmm0, xmm1
  vmulss xmm0, xmm0, xmm5
  vmulss xmm0, xmm0, xmm6
  vminss xmm0, xmm0, xmm6
  vmaxss xmm0, xmm0, xmm7
  vcvtss2si eax, xmm0
  mov byte ptr [rcx + r10], al
  inc r10
  cmp r10, rax
  jl @TailLoop

@Done:
  add rsp, 32
  vzeroupper
  ret

@Exit:
  vzeroupper
  ret
end;

procedure _AVX512QuantizeInt8( dst : PShortInt; src : PSingle; N : Integer; const MaxAbs : Single ); inline;
begin
  _AVX2QuantizeInt8(dst, src, N, MaxAbs);
end;

{-----------------------------------------------------------------------------
  AVX2 int8 dequantize (64-bit). Same as 32-bit but with 64-bit registers.
  Parameters: RCX=dst (PSingle), RDX=src (PShortInt), R8=N, XMM3=Scale.
-----------------------------------------------------------------------------}
procedure _AVX2DequantizeInt8( dst : PSingle; src : PShortInt; N : Integer; const Scale : Single );
asm
  test r8, r8
  jle @Exit

  // Broadcast Scale
  vbroadcastss ymm2, xmm3

  // Bulk = N - (N mod 8)
  mov rax, r8
  and rax, 7
  sub r8, rax

  mov r9, r8
  shr r9, 3
  jz @Tail

@BulkLoop:
  vpmovsxbd ymm0, [rdx]        // 8 bytes -> 8 dwords
  vcvtdq2ps ymm0, ymm0
  vmulps ymm0, ymm0, ymm2
  vmovups [rcx], ymm0
  add rdx, 8
  add rcx, 32
  dec r9
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Exit

  movss xmm2, xmm3             // Scale (scalar)
  xor r10, r10
@TailLoop:
  movsx eax, byte ptr [rdx + r10]
  vcvtsi2ss xmm0, xmm0, eax
  vmulss xmm0, xmm0, xmm2
  vmovss [rcx + r10*4], xmm0
  inc r10
  cmp r10, rax
  jl @TailLoop

@Exit:
  vzeroupper
  ret
end;

procedure _AVX512DequantizeInt8( dst : PSingle; src : PShortInt; N : Integer; const Scale : Single ); inline;
begin
  _AVX2DequantizeInt8(dst, src, N, Scale);
end;

{-----------------------------------------------------------------------------
  AVX2 decode bfloat16 to Single (64-bit). Processes 8 elements per loop.
  Parameters: RCX=dst, RDX=src, R8=N.
-----------------------------------------------------------------------------}
procedure _AVX2DecodeBF16( dst : PSingle; src : PWord; N : Integer );
asm
  test r8, r8
  jle @Exit

  mov rax, r8
  and rax, 7
  sub r8, rax

  mov r9, r8
  shr r9, 3
  jz @Tail

@BulkLoop:
  vpmovzxwd ymm0, [rdx]
  vpslld ymm0, ymm0, 16
  vmovups [rcx], ymm0
  add rdx, 16
  add rcx, 32
  dec r9
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Exit

  xor r10, r10
@TailLoop:
  movzx eax, word ptr [rdx + r10*2]
  shl eax, 16
  mov dword ptr [rcx + r10*4], eax
  inc r10
  cmp r10, rax
  jl @TailLoop

@Exit:
  vzeroupper
  ret
end;

procedure _AVX512DecodeBF16( dst : PSingle; src : PWord; N : Integer ); inline;
begin
  _AVX2DecodeBF16(dst, src, N);
end;


{-----------------------------------------------------------------------------
  AVX2 ReLU gate mask (64-bit). Parameters: RCX=dst, RDX=src, R8=N.
-----------------------------------------------------------------------------}
procedure _AVX2ReluGateMask( dst : PSingle; src : PSingle; N : Integer );
asm
  test r8, r8
  jle @Exit

  mov eax, 1.0
  movd xmm2, eax
  vbroadcastss ymm2, xmm2
  vxorps ymm3, ymm3, ymm3

  mov rax, r8
  and rax, 7
  sub r8, rax

  mov r9, r8
  shr r9, 3
  jz @Tail

@BulkLoop:
  vmovups ymm0, [rdx]
  vcmpps ymm1, ymm0, ymm3, 29
  vandps ymm1, ymm1, ymm2
  vmovups [rcx], ymm1
  add rdx, 32
  add rcx, 32
  dec r9
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Exit

  movss xmm2, [1.0] // ??? We can't have [1.0] directly, use constant via xmm register.
  // Use a local variable: but we can just reuse the broadcast value from ymm2's low part? Actually ymm2 is still 1.0, but we need scalar xmm2. We can use movss xmm2, xmm2 to get low part, but we already have ymm2, we can use the low 32 bits by using xmm2 (same register).
  // So we can just use xmm2 as is.
  xor r10, r10
@TailLoop:
  vmovss xmm0, [rdx + r10*4]
  vcmpss xmm1, xmm0, xmm3, 29
  vandps xmm1, xmm1, xmm2
  vmovss [rcx + r10*4], xmm1
  inc r10
  cmp r10, rax
  jl @TailLoop

@Exit:
  vzeroupper
  ret
end;

procedure _AVX512ReluGateMask( dst : PSingle; src : PSingle; N : Integer ); inline;
begin
  _AVX2ReluGateMask(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Leaky ReLU (64-bit fallback). Same as 32-bit but with 64-bit registers.
  Parameters: RCX=dst, RDX=src, R8=N, XMM3=Slope.
-----------------------------------------------------------------------------}
procedure _AVX2LeakyRelu( dst : PSingle; src : PSingle; N : Integer; const Slope : Single );
asm
  test r8, r8
  jle @Exit

  vbroadcastss ymm2, xmm3      // ymm2 = Slope

  mov rax, r8
  and rax, 7
  sub r8, rax

  mov r9, r8
  shr r9, 3
  jz @Tail

  vxorps ymm3, ymm3, ymm3

@BulkLoop:
  vmovups ymm0, [rdx]
  vmulps ymm1, ymm0, ymm2
  vcmpps ymm4, ymm0, ymm3, 29
  vblendvps ymm1, ymm1, ymm0, ymm4
  vmovups [rcx], ymm1
  add rdx, 32
  add rcx, 32
  dec r9
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Exit

  movss xmm2, xmm3             // Slope
  xorps xmm3, xmm3

  xor r10, r10
@TailLoop:
  vmovss xmm0, [rdx + r10*4]
  vmulss xmm1, xmm0, xmm2
  vcmpss xmm4, xmm0, xmm3, 29
  vblendvps xmm1, xmm1, xmm0, xmm4
  vmovss [rcx + r10*4], xmm1
  inc r10
  cmp r10, rax
  jl @TailLoop

@Exit:
  vzeroupper
  ret
end;

procedure _AVX512LeakyRelu( dst : PSingle; src : PSingle; N : Integer; const Slope : Single ); inline;
begin
  _AVX2LeakyRelu(dst, src, N, Slope);
end;

{-----------------------------------------------------------------------------
  AVX2 decode F16 (64-bit). Uses F16C mnemonics.
  Bulk: 8 elements per loop using vcvtph2ps ymm, xmm.
  Tail: 4 elements using vcvtph2ps xmm, xmm, then scalar bit ops for 0..3.
  Parameters: RCX=dst, RDX=src, R8=N.
  Non-volatile registers saved: r12.
-----------------------------------------------------------------------------}
procedure _AVX2DecodeF16( dst : PSingle; src : PWord; N : Integer );
asm
  push r12

  test r8, r8
  jle @Exit

  // Load constants for NaN quieting
  mov eax, $7FFF
  movd xmm0, eax
  vpbroadcastw xmm2, xmm0      // xmm2 = 8 * $7FFF
  mov eax, $7C00
  movd xmm0, eax
  vpbroadcastw xmm3, xmm0      // xmm3 = 8 * $7C00
  mov eax, $0200
  movd xmm0, eax
  vpbroadcastw xmm4, xmm0      // xmm4 = 8 * $0200

  // Bulk = N - (N mod 8)
  mov rax, r8
  and rax, 7
  sub r8, rax

  mov r9, r8
  shr r9, 3
  jz @Tail

@BulkLoop:
  vmovdqu xmm0, [rdx]          // 8 halfs
  vpand   xmm1, xmm0, xmm2     // |h|
  vpcmpgtw xmm1, xmm1, xmm3    // |h| > $7C00 => NaN
  vpand   xmm1, xmm1, xmm4     // quiet bit
  vpor    xmm0, xmm0, xmm1     // signalling NaN -> quiet NaN

  vcvtph2ps ymm0, xmm0         // convert 8 halfs to 8 Singles
  vmovups [rcx], ymm0
  add rdx, 16
  add rcx, 32
  dec r9
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Done

  // Process 4-element group if tail >= 4
  cmp rax, 4
  jl @ScalarTail

  vmovq xmm0, [rdx]            // load 4 halfs
  vcvtph2ps xmm1, xmm0         // convert 4 halfs to 4 Singles
  vmovups [rcx], xmm1
  add rdx, 8
  add rcx, 16
  sub rax, 4

@ScalarTail:
  test rax, rax
  jz @Done

  // Process remaining 1..3 with scalar bit ops
  xor r12, r12
@ScalarLoop:
  movzx r11d, word ptr [rdx + r12*2]
  mov r8d, r11d
  shr r8d, 15
  and r8d, 1
  shl r8d, 31
  mov r9d, r11d
  shr r9d, 10
  and r9d, $1F
  mov r10d, r11d
  and r10d, $3FF

  cmp r9d, $1F
  je @Special

  test r9d, r9d
  jnz @Normal

  // Subnormal
  mov r9d, 103
  shl r9d, 23
  shl r10d, 13
  or r8d, r9d
  or r8d, r10d
  jmp @StoreScalar

@Normal:
  add r9d, 112
  shl r9d, 23
  shl r10d, 13
  or r8d, r9d
  or r8d, r10d
  jmp @StoreScalar

@Special:
  test r10d, r10d
  jz @Inf
  // NaN
  mov r9d, $7F800000
  shl r10d, 13
  or r10d, $400000
  or r8d, r9d
  or r8d, r10d
  jmp @StoreScalar
@Inf:
  mov r9d, $7F800000
  or r8d, r9d

@StoreScalar:
  mov [rcx + r12*4], r8d
  inc r12
  cmp r12, rax
  jl @ScalarLoop

@Done:
  pop r12
  vzeroupper
  ret

@Exit:
  pop r12
  vzeroupper
  ret
end;

procedure _AVX512DecodeF16( dst : PSingle; src : PWord; N : Integer ); inline;
begin
  _AVX2DecodeF16(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 centered sum of squares (64-bit fallback). Same logic as 32-bit.
  Parameters: RCX=src, XMM1=Mean, RDX=N.
-----------------------------------------------------------------------------}
function _AVX2SumSqrCentered( src : PSingle; Mean : Single; N : Integer ) : Single;
asm
  test rdx, rdx
  jle @Zero

  vbroadcastss ymm3, xmm1      // broadcast Mean

  mov rax, rdx
  and rax, 15
  sub rdx, rax

  vxorps ymm0, ymm0, ymm0
  vxorps ymm1, ymm1, ymm1

  mov r8, rdx
  shr r8, 4
  jz @Tail

@BulkLoop:
  vmovups ymm2, [rcx]
  vsubps ymm2, ymm2, ymm3
  vfmadd231ps ymm0, ymm2, ymm2

  vmovups ymm2, [rcx+32]
  vsubps ymm2, ymm2, ymm3
  vfmadd231ps ymm1, ymm2, ymm2

  add rcx, 64
  dec r8
  jnz @BulkLoop

  vaddps ymm0, ymm0, ymm1
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  movss [Result], xmm0
  vzeroupper

@Tail:
  test rax, rax
  jz @Done

  movss xmm3, xmm1             // Mean (scalar)
  movss xmm0, [Result]

  xor r9, r9
@TailLoop:
  vmovss xmm1, [rcx + r9*4]
  vsubss xmm1, xmm1, xmm3
  vmulss xmm1, xmm1, xmm1
  vaddss xmm0, xmm0, xmm1
  inc r9
  cmp r9, rax
  jl @TailLoop

  movss [Result], xmm0

@Done:
  vzeroupper
  ret

@Zero:
  xorps xmm0, xmm0
  movss [Result], xmm0
  ret
end;

function _AVX512SumSqrCentered( src : PSingle; Mean : Single; N : Integer ) : Single; inline;
begin
  Result := _AVX2SumSqrCentered(src, Mean, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Adam optimizer step (64-bit). Processes 8 elements per loop.
  Parameters: RCX=PtrDelta, RDX=PtrM, R8=PtrV,
    XMM0=Beta1, XMM1=OmBeta1, XMM2=Beta2, XMM3=OmBeta2,
    [rsp+40]=InvOmB2D, [rsp+48]=Epsilon, [rsp+56]=kLR, [rsp+64]=NumElements.
-----------------------------------------------------------------------------}
procedure _AVX2AdamDelta( PtrDelta, PtrM, PtrV : PSingle;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR : Single;
  NumElements : Integer );
asm
  push rbp
  mov rbp, rsp
  sub rsp, 32                    // shadow space

  mov r10, [rbp+64]              // NumElements
  test r10, r10
  jle @Exit

  // Broadcast constants to YMM registers
  vbroadcastss ymm9, xmm0        // Beta1
  vbroadcastss ymm10, xmm1       // OmBeta1
  vbroadcastss ymm11, xmm2       // Beta2
  vbroadcastss ymm12, xmm3       // OmBeta2
  vbroadcastss ymm13, [rbp+40]   // InvOmB2D
  vbroadcastss ymm14, [rbp+48]   // Epsilon
  vbroadcastss ymm15, [rbp+56]   // kLR

  // Bulk = N - (N mod 8)
  mov rax, r10
  and rax, 7
  sub r10, rax
  mov r11, r10
  shr r11, 3
  jz @Tail

  mov r12, PtrDelta
  mov r13, PtrM
  mov r14, PtrV

@BulkLoop:
  vmovups ymm0, [r12]            // g
  vmulps  ymm1, ymm0, ymm10      // OmBeta1 * g
  vmulps  ymm2, ymm9, [r13]      // Beta1 * m
  vaddps  ymm1, ymm1, ymm2
  vmovups [r13], ymm1            // store m

  vmulps  ymm3, ymm0, ymm0       // g*g
  vmulps  ymm3, ymm3, ymm12      // OmBeta2 * (g*g)
  vmulps  ymm4, ymm11, [r14]     // Beta2 * v
  vaddps  ymm3, ymm3, ymm4
  vmovups [r14], ymm3            // store v

  vmulps  ymm3, ymm3, ymm13      // v * InvOmB2D
  vsqrtps ymm3, ymm3
  vaddps  ymm3, ymm3, ymm14
  vmulps  ymm1, ymm1, ymm15
  vdivps  ymm0, ymm1, ymm3
  vmovups [r12], ymm0            // store delta

  add r12, 32
  add r13, 32
  add r14, 32
  dec r11
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Exit

  mov r12, PtrDelta
  mov r13, PtrM
  mov r14, PtrV

  // Load scalar constants into XMM
  vbroadcastss xmm9, xmm0
  vbroadcastss xmm10, xmm1
  vbroadcastss xmm11, xmm2
  vbroadcastss xmm12, xmm3
  vmovss xmm13, [rbp+40]
  vmovss xmm14, [rbp+48]
  vmovss xmm15, [rbp+56]

  xor r11, r11
@TailLoop:
  vmovss xmm0, [r12 + r11*4]
  vmulss xmm1, xmm0, xmm10
  vmulss xmm2, xmm9, [r13 + r11*4]
  vaddss xmm1, xmm1, xmm2
  vmovss [r13 + r11*4], xmm1

  vmulss xmm3, xmm0, xmm0
  vmulss xmm3, xmm3, xmm12
  vmulss xmm4, xmm11, [r14 + r11*4]
  vaddss xmm3, xmm3, xmm4
  vmovss [r14 + r11*4], xmm3

  vmulss xmm3, xmm3, xmm13
  vsqrtss xmm3, xmm3, xmm3
  vaddss xmm3, xmm3, xmm14
  vmulss xmm1, xmm1, xmm15
  vdivss xmm0, xmm1, xmm3
  vmovss [r12 + r11*4], xmm0
  inc r11
  cmp r11, rax
  jl @TailLoop

@Exit:
  add rsp, 32
  pop rbp
  vzeroupper
  ret
end;

procedure _AVX512AdamDelta( PtrDelta, PtrM, PtrV : PSingle;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR : Single;
  NumElements : Integer ); inline;
begin
  _AVX2AdamDelta(PtrDelta, PtrM, PtrV, Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR, NumElements);
end;


{-----------------------------------------------------------------------------
  AVX2 Adafactor step (64-bit).
  Processes 8 elements per loop.
  Parameters: RCX=PtrDelta, RDX=PtrV,
    XMM0=Beta2, XMM1=k, XMM2=c, XMM3=Epsilon,
    [rsp+40]=NumElements.
-----------------------------------------------------------------------------}
procedure _AVX2AdafactorDelta( PtrDelta, PtrV : PSingle;
  Beta2, k, c, Epsilon : Single; NumElements : Integer );
asm
  push rbp
  mov rbp, rsp
  sub rsp, 32

  mov r10, [rbp+40]              // NumElements
  test r10, r10
  jle @Exit

  // Broadcast constants to YMM
  vbroadcastss ymm12, xmm0       // Beta2
  vbroadcastss ymm13, xmm1       // k
  vbroadcastss ymm14, xmm2       // c
  vbroadcastss ymm15, xmm3       // Epsilon

  // Bulk = N - (N mod 8)
  mov rax, r10
  and rax, 7
  sub r10, rax
  mov r11, r10
  shr r11, 3
  jz @Tail

  mov r12, PtrDelta
  mov r13, PtrV

@BulkLoop:
  vmovups ymm0, [r12]            // d
  vmulps  ymm1, ymm0, ymm0
  vmulps  ymm1, ymm1, ymm13
  vaddps  ymm1, ymm1, ymm14
  vmulps  ymm2, ymm12, [r13]
  vaddps  ymm1, ymm1, ymm2
  vmovups [r13], ymm1

  vsqrtps ymm1, ymm1
  vaddps  ymm1, ymm1, ymm15
  vdivps  ymm0, ymm0, ymm1
  vmovups [r12], ymm0

  add r12, 32
  add r13, 32
  dec r11
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Exit

  // Scalar tail (0..7)
  mov r12, PtrDelta
  mov r13, PtrV

  vbroadcastss xmm12, xmm0
  vbroadcastss xmm13, xmm1
  vbroadcastss xmm14, xmm2
  vbroadcastss xmm15, xmm3

  xor r11, r11
@TailLoop:
  vmovss xmm0, [r12 + r11*4]
  vmulss xmm1, xmm0, xmm0
  vmulss xmm1, xmm1, xmm13
  vaddss xmm1, xmm1, xmm14
  vmulss xmm2, xmm12, [r13 + r11*4]
  vaddss xmm1, xmm1, xmm2
  vmovss [r13 + r11*4], xmm1

  vsqrtss xmm1, xmm1, xmm1
  vaddss xmm1, xmm1, xmm15
  vdivss xmm0, xmm0, xmm1
  vmovss [r12 + r11*4], xmm0
  inc r11
  cmp r11, rax
  jl @TailLoop

@Exit:
  add rsp, 32
  pop rbp
  vzeroupper
  ret
end;

procedure _AVX512AdafactorDelta( PtrDelta, PtrV : PSingle;
  Beta2, k, c, Epsilon : Single; NumElements : Integer ); inline;
begin
  _AVX2AdafactorDelta(PtrDelta, PtrV, Beta2, k, c, Epsilon, NumElements);
end;

{-----------------------------------------------------------------------------
  AVX2 clamp absolute (64-bit). Processes 8 elements per loop.
  Parameters: RCX=PtrA, XMM1=Value, RDX=NumElements.
-----------------------------------------------------------------------------}
procedure _AVX2ClampAbs( PtrA : PSingle; Value : Single; NumElements : Integer );
asm
  test rdx, rdx
  jle @Exit

  // Compute -Value
  movss xmm2, xmm1            // copy Value
  xorps xmm3, xmm3
  subss xmm3, xmm2            // -Value in xmm3
  vbroadcastss ymm14, xmm1    // +Value
  vbroadcastss ymm15, xmm3    // -Value

  // Bulk = N - (N mod 8)
  mov rax, rdx
  and rax, 7
  sub rdx, rax
  mov r8, rdx
  shr r8, 3
  jz @Tail

  mov r9, PtrA
@BulkLoop:
  vmovups ymm0, [r9]
  vmaxps ymm0, ymm15, ymm0
  vminps ymm0, ymm14, ymm0
  vmovups [r9], ymm0
  add r9, 32
  dec r8
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Exit

  movss xmm14, xmm1           // +Value
  movss xmm15, xmm3           // -Value

  xor r8, r8
@TailLoop:
  vmovss xmm0, [r9 + r8*4]
  vmaxss xmm0, xmm15, xmm0
  vminss xmm0, xmm14, xmm0
  vmovss [r9 + r8*4], xmm0
  inc r8
  cmp r8, rax
  jl @TailLoop

@Exit:
  vzeroupper
  ret
end;

procedure _AVX512ClampAbs( PtrA : PSingle; Value : Single; NumElements : Integer ); inline;
begin
  _AVX2ClampAbs(PtrA, Value, NumElements);
end;

{-----------------------------------------------------------------------------
  AVX2 Lion optimizer step (64-bit). Processes 8 elements per loop.
  Parameters: RCX=PtrDelta, RDX=PtrM,
  XMM0=Beta1, XMM1=k1, XMM2=Beta2, XMM3=k2,
  [rsp+40]=NegLR, [rsp+48]=PosLR, [rsp+56]=NumElements.
-----------------------------------------------------------------------------}
procedure _AVX2LionDelta( PtrDelta, PtrM : PSingle; Beta1, k1, Beta2, k2, NegLR, PosLR : Single; NumElements : Integer );
asm
  push rbp
  mov rbp, rsp
  sub rsp, 32

  mov r10, [rbp+56]              // NumElements
  test r10, r10
  jle @Exit

  vbroadcastss ymm10, xmm0       // Beta1
  vbroadcastss ymm11, xmm1       // k1
  vbroadcastss ymm12, xmm2       // Beta2
  vbroadcastss ymm13, xmm3       // k2
  vbroadcastss ymm14, [rbp+40]   // NegLR
  vbroadcastss ymm15, [rbp+48]   // PosLR
  vxorps ymm9, ymm9, ymm9

  mov rax, r10
  and rax, 7
  sub r10, rax
  mov r11, r10
  shr r11, 3
  jz @Tail

  mov r12, PtrDelta
  mov r13, PtrM

@BulkLoop:
  vmovups ymm0, [r12]            // d
  vmovups ymm1, [r13]            // m

  vmulps ymm2, ymm10, ymm1
  vmulps ymm3, ymm11, ymm0
  vaddps ymm2, ymm2, ymm3        // c

  vmulps ymm3, ymm12, ymm1
  vmulps ymm4, ymm13, ymm0
  vaddps ymm3, ymm3, ymm4
  vmovups [r13], ymm3

  vcmpps ymm4, ymm9, ymm2, 1     // c > 0
  vcmpps ymm5, ymm2, ymm9, 1     // c < 0
  vandps ymm4, ymm4, ymm14
  vandps ymm5, ymm5, ymm15
  vorps  ymm4, ymm4, ymm5
  vmovups [r12], ymm4

  add r12, 32
  add r13, 32
  dec r11
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Exit

  // Scalar constants
  vbroadcastss xmm10, xmm0
  vbroadcastss xmm11, xmm1
  vbroadcastss xmm12, xmm2
  vbroadcastss xmm13, xmm3
  vmovss xmm14, [rbp+40]
  vmovss xmm15, [rbp+48]
  xorps xmm9, xmm9

  mov r12, PtrDelta
  mov r13, PtrM

  xor r11, r11
@TailLoop:
  vmovss xmm0, [r12 + r11*4]
  vmovss xmm1, [r13 + r11*4]

  vmulss xmm2, xmm10, xmm1
  vmulss xmm3, xmm11, xmm0
  vaddss xmm2, xmm2, xmm3        // c

  vmulss xmm3, xmm12, xmm1
  vmulss xmm4, xmm13, xmm0
  vaddss xmm3, xmm3, xmm4
  vmovss [r13 + r11*4], xmm3

  vcmpss xmm4, xmm9, xmm2, 1
  vcmpss xmm5, xmm2, xmm9, 1
  vandps xmm4, xmm4, xmm14
  vandps xmm5, xmm5, xmm15
  vorps  xmm4, xmm4, xmm5
  vmovss [r12 + r11*4], xmm4

  inc r11
  cmp r11, rax
  jl @TailLoop

@Exit:
  add rsp, 32
  pop rbp
  vzeroupper
  ret
end;

procedure _AVX512LionDelta( PtrDelta, PtrM : PSingle; Beta1, k1, Beta2, k2, NegLR, PosLR : Single; NumElements : Integer ); inline;
begin
  _AVX2LionDelta(PtrDelta, PtrM, Beta1, k1, Beta2, k2, NegLR, PosLR, NumElements);
end;

{-----------------------------------------------------------------------------
  AVX2 max value and first occurrence index (64-bit).
  Processes 16 elements per loop using two YMM blocks.
  Parameters: RCX=PtrA, RDX=NumElements, R8=Pos (out).
  Returns max value in Result.
-----------------------------------------------------------------------------}
function _AVX2GetMaxPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single;
asm
  push rbx
  push r12
  push r13
  sub rsp, 128

  test rdx, rdx
  jle @Zero

  // Bulk = NumElements - (NumElements mod 16)
  mov rax, rdx
  and rax, 15
  sub rdx, rax

  // Load seed indices via RIP-relative addressing
  vmovdqu ymm4, [rip+cAVXArgLaneSeed]
  vmovdqu ymm5, [rip+cAVXArgLaneSeed+32]
  vmovdqa ymm2, ymm4
  vmovdqa ymm3, ymm5

  // Load first 16 values
  vmovups ymm0, [rcx]
  vmovups ymm1, [rcx+32]

  mov r9, rdx
  shr r9, 4
  jz @Fold

  vmovdqu ymm6, [rip+cAVXArgLaneStep]
  add rcx, 64
  dec r9

@Loop:
  vmovups ymm7, [rcx]
  vmovups ymm8, [rcx+32]

  vpaddd ymm4, ymm4, ymm6
  vpaddd ymm5, ymm5, ymm6

  vcmpps ymm9, ymm7, ymm0, 30
  vcmpps ymm10, ymm8, ymm1, 30

  vblendvps ymm2, ymm2, ymm4, ymm9
  vblendvps ymm3, ymm3, ymm5, ymm10
  vblendvps ymm0, ymm0, ymm7, ymm9
  vblendvps ymm1, ymm1, ymm8, ymm10

  add rcx, 64
  dec r9
  jnz @Loop

@Fold:
  // Store 16 candidates to stack
  vmovups [rsp], ymm0
  vmovups [rsp+32], ymm1
  vmovdqu [rsp+64], ymm2
  vmovdqu [rsp+96], ymm3
  vzeroupper

  // Scalar scan over 16 candidates
  movss xmm0, [rsp]
  mov eax, [rsp+64]
  mov r9, 1
  lea r10, [rsp]
  lea r11, [rsp+64]
@Scan:
  cmp r9, 16
  jge @ScanDone
  movss xmm1, [r10 + r9*4]
  comiss xmm1, xmm0
  jbe @Skip
  movss xmm0, xmm1
  mov eax, [r11 + r9*4]
@Skip:
  inc r9
  jmp @Scan
@ScanDone:

  // Tail handling (0..15 elements)
  test rax, rax
  jz @TailDone

  mov r9, rdx
  xor r10, r10
@TailLoop:
  movss xmm1, [rcx + r10*4]
  comiss xmm1, xmm0
  jbe @TailSkip
  movss xmm0, xmm1
  lea eax, [r9 + r10]
@TailSkip:
  inc r10
  cmp r10, rax
  jl @TailLoop

@TailDone:
  movss [Result], xmm0
  mov [r8], eax

  add rsp, 128
  pop r13
  pop r12
  pop rbx
  ret

@Zero:
  xorps xmm0, xmm0
  movss [Result], xmm0
  xor eax, eax
  mov [r8], eax
  add rsp, 128
  pop r13
  pop r12
  pop rbx
  ret
end;

function _AVX512GetMaxPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;
begin
  Result := _AVX2GetMaxPos(PtrA, NumElements, Position);
end;

{-----------------------------------------------------------------------------
  AVX2 min value and first occurrence index (64-bit).
  Processes 16 elements per loop using two YMM blocks.

  Parameters:
    RCX = PtrA : PSingle
    RDX = NumElements : Integer
    R8  = Position : PInteger  (out)
  Returns:
    XMM0 = min value (Single)
-----------------------------------------------------------------------------}
function _AVX2GetMinPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single;
asm
  push rbx
  push r12
  push r13
  sub rsp, 128

  test rdx, rdx
  jle @Zero

  // ---- Compute bulk and tail counts ----
  mov rax, rdx                 // rax = NumElements
  and rax, 15                  // tail = NumElements mod 16
  sub rdx, rax                 // rdx = bulk (multiple of 16)
  mov r10, rdx                 // r10 = bulk count (used later for index offset)

  // ---- Load seed indices and step ----
  vmovdqu ymm4, [rip+cAVXArgLaneSeed]       // lanes 0..15 seed
  vmovdqu ymm5, [rip+cAVXArgLaneSeed+32]
  vmovdqa ymm2, ymm4           // index accumulators
  vmovdqa ymm3, ymm5

  // ---- Load first 16 values ----
  vmovups ymm0, [rcx]          // first 8 values
  vmovups ymm1, [rcx+32]       // next 8 values

  mov r9, rdx
  shr r9, 4                    // number of 16-element chunks
  jz @Fold

  vmovdqu ymm6, [rip+cAVXArgLaneStep]       // step = 16 for each lane
  add rcx, 64                  // advance past first block
  dec r9

@BulkLoop:
  vmovups ymm7, [rcx]          // load next 8
  vmovups ymm8, [rcx+32]       // load next 8

  vpaddd ymm4, ymm4, ymm6      // update indices for first 8
  vpaddd ymm5, ymm5, ymm6      // update indices for second 8

  // Compare: new < current min (predicate 17 = _CMP_LT_OQ)
  vcmpps ymm9, ymm7, ymm0, 17
  vcmpps ymm10, ymm8, ymm1, 17

  // Update index accumulators and min values
  vblendvps ymm2, ymm2, ymm4, ymm9
  vblendvps ymm3, ymm3, ymm5, ymm10
  vblendvps ymm0, ymm0, ymm7, ymm9
  vblendvps ymm1, ymm1, ymm8, ymm10

  add rcx, 64
  dec r9
  jnz @BulkLoop

@Fold:
  // ---- Store 16 candidates to stack ----
  vmovups [rsp], ymm0          // min values (first 8)
  vmovups [rsp+32], ymm1       // min values (last 8)
  vmovdqu [rsp+64], ymm2       // indices (first 8)
  vmovdqu [rsp+96], ymm3       // indices (last 8)
  vzeroupper

  // ---- Scalar scan over 16 candidates (tie-breaking: lower index) ----
  movss xmm0, [rsp]            // min value = first candidate
  mov eax, [rsp+64]            // min index = first candidate index
  mov r9, 1
  lea r10, [rsp]
  lea r11, [rsp+64]
@Scan:
  cmp r9, 16
  jge @ScanDone
  movss xmm1, [r10 + r9*4]
  comiss xmm1, xmm0            // compare with current min
  jae @Skip                    // if new >= min, skip
  movss xmm0, xmm1
  mov eax, [r11 + r9*4]        // update index
@Skip:
  inc r9
  jmp @Scan
@ScanDone:

  // eax now holds the global minimum index (absolute position in the bulk part)

  // ---- Tail handling (0..15 elements) ----
  test rax, rax                // rax still contains tail count (from before)
  jz @TailDone

  // rcx points to the first tail element (after bulk)
  // r10 = bulk count (starting offset for tail indices)
  xor r9, r9                   // tail loop counter
@TailLoop:
  movss xmm1, [rcx + r9*4]
  comiss xmm1, xmm0
  jae @TailSkip
  movss xmm0, xmm1
  lea eax, [r10 + r9]          // absolute index = bulk + tail offset
@TailSkip:
  inc r9
  cmp r9, rax                  // rax = tail count
  jl @TailLoop

@TailDone:
  // ---- Write result and position ----
  // XMM0 already holds the min value (return value)
  mov [r8], eax                // write position

  add rsp, 128
  pop r13
  pop r12
  pop rbx
  ret

@Zero:
  xorps xmm0, xmm0
  xor eax, eax
  mov [r8], eax
  add rsp, 128
  pop r13
  pop r12
  pop rbx
  ret
end;

function _AVX512GetMinPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;
begin
  Result := _AVX2GetMinPos(PtrA, NumElements, Position);
end;

{-----------------------------------------------------------------------------
  AVX2 max absolute value and first occurrence index (64-bit).
  Processes 16 elements per loop using two YMM blocks.
  Parameters: RCX=PtrA, RDX=NumElements, R8=Pos (out).
  Returns max absolute value in Result.
-----------------------------------------------------------------------------}
function _AVX2GetMaxAbsPos( PtrA : PSingle; NumElements : integer; out Position : integer ) : single;
asm
  push rbx
  push r12
  push r13
  sub rsp, 128

  test rdx, rdx
  jle @Zero

  mov rax, rdx
  and rax, 15
  sub rdx, rax

  // Load absolute mask (clear sign bit)
  vmovdqu ymm11, [rip+cAVXArgAbsMask]

  // Load seed indices
  vmovdqu ymm4, [rip+cAVXArgLaneSeed]
  vmovdqu ymm5, [rip+cAVXArgLaneSeed+32]
  vmovdqa ymm2, ymm4
  vmovdqa ymm3, ymm5

  // Load first 16 values and take absolute value
  vmovups ymm0, [rcx]
  vmovups ymm1, [rcx+32]
  vandps  ymm0, ymm0, ymm11
  vandps  ymm1, ymm1, ymm11

  mov r9, rdx
  shr r9, 4
  jz @Fold

  vmovdqu ymm6, [rip+cAVXArgLaneStep]
  add rcx, 64
  dec r9

@Loop:
  vmovups ymm7, [rcx]
  vmovups ymm8, [rcx+32]
  vandps  ymm7, ymm7, ymm11
  vandps  ymm8, ymm8, ymm11

  vpaddd  ymm4, ymm4, ymm6
  vpaddd  ymm5, ymm5, ymm6

  vcmpps  ymm9, ymm7, ymm0, 30
  vcmpps  ymm10, ymm8, ymm1, 30

  vblendvps ymm2, ymm2, ymm4, ymm9
  vblendvps ymm3, ymm3, ymm5, ymm10
  vblendvps ymm0, ymm0, ymm7, ymm9
  vblendvps ymm1, ymm1, ymm8, ymm10

  add rcx, 64
  dec r9
  jnz @Loop

@Fold:
  vmovups [rsp], ymm0
  vmovups [rsp+32], ymm1
  vmovdqu [rsp+64], ymm2
  vmovdqu [rsp+96], ymm3
  vzeroupper

  // Scalar scan over 16 candidates (take max absolute, tie-breaking to lower index)
  movss xmm0, [rsp]
  mov eax, [rsp+64]
  mov r9, 1
  lea r10, [rsp]
  lea r11, [rsp+64]
@Scan:
  cmp r9, 16
  jge @ScanDone
  movss xmm1, [r10 + r9*4]
  comiss xmm1, xmm0
  jbe @Skip
  movss xmm0, xmm1
  mov eax, [r11 + r9*4]
@Skip:
  inc r9
  jmp @Scan
@ScanDone:

  test rax, rax
  jz @TailDone

  mov r9, rdx
  xor r10, r10
@TailLoop:
  movss xmm1, [rcx + r10*4]
  // take absolute value for tail
  vandps xmm1, xmm1, xmm11   // xmm11 holds abs mask (low 128 bits)
  comiss xmm1, xmm0
  jbe @TailSkip
  movss xmm0, xmm1
  lea eax, [r9 + r10]
@TailSkip:
  inc r10
  cmp r10, rax
  jl @TailLoop

@TailDone:
  movss [Result], xmm0
  mov [r8], eax

  add rsp, 128
  pop r13
  pop r12
  pop rbx
  ret

@Zero:
  xorps xmm0, xmm0
  movss [Result], xmm0
  xor eax, eax
  mov [r8], eax
  add rsp, 128
  pop r13
  pop r12
  pop rbx
  ret
end;

function _AVX512GetMaxAbsPos( PtrA : PSingle; NumElements : integer; out Position : integer ) : single; inline;
begin
  Result := _AVX512GetMaxAbsPos(PtrA, NumElements, Position);
end;

{-----------------------------------------------------------------------------
  AVX2 add scalar: dst[i] += Value for i = 0..N-1.
  Processes 32 elements per loop (4 YMM blocks), scalar tail.
  Parameters: RCX=PtrA, XMM1=Value, RDX=N.
-----------------------------------------------------------------------------}
procedure _AVX2AddScalar( PtrA : PSingle; Value : single; N : integer );
asm
  test rdx, rdx
  jle @Exit

  // Broadcast Value to YMM7
  vbroadcastss ymm7, xmm1

  // Bulk = N - (N mod 32)
  mov rax, rdx
  and rax, 31
  sub rdx, rax

  mov r8, rdx
  shr r8, 5
  jz @Tail

@BulkLoop:
  vaddps ymm0, ymm7, [rcx]
  vaddps ymm1, ymm7, [rcx+32]
  vaddps ymm2, ymm7, [rcx+64]
  vaddps ymm3, ymm7, [rcx+96]
  vmovups [rcx], ymm0
  vmovups [rcx+32], ymm1
  vmovups [rcx+64], ymm2
  vmovups [rcx+96], ymm3
  add rcx, 128
  dec r8
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Exit

  // Scalar tail (0..31)
  xor r8, r8
@TailLoop:
  vmovss xmm0, [rcx + r8*4]
  vaddss xmm0, xmm0, xmm7
  vmovss [rcx + r8*4], xmm0
  inc r8
  cmp r8, rax
  jl @TailLoop

@Exit:
  vzeroupper
  ret
end;

procedure _AVX512AddScalar( PtrA : PSingle; Value : single; N : integer ); inline;
begin
  _AVX512AddScalar(PtrA, Value, N);
end;

{-----------------------------------------------------------------------------
  AVX2 exp-shift-sum: dst[i] := exp(src[i] - Shift), returns sum of dst.
  Processes 8 elements per loop, reduces lane sums, scalar tail.
  Parameters: RCX=dst, RDX=src, R8=NumElements, XMM3=Shift.
-----------------------------------------------------------------------------}
function _AVX2ExpShiftSum( dst : PSingle; src : PSingle; Shift : single; N : integer ) : single;
asm
  push rbx
  push r12
  push r13
  sub rsp, 32

  test r8, r8
  jle @Zero

  mov rax, r8
  and rax, 7
  sub r8, rax

  // Broadcast constants
  vbroadcastss ymm10, [cAVXExpHi]
  vbroadcastss ymm11, [cAVXExpLo]
  vbroadcastss ymm12, [cAVXLog2e]
  vbroadcastss ymm13, [cAVXLn2]
  vmovd xmm14, [cAVXExp127]
  vpbroadcastd ymm14, xmm14

  vbroadcastss ymm6, [cAVXExpP6]
  vbroadcastss ymm7, [cAVXExpP5]
  vbroadcastss ymm9, [cAVXExpP4]
  vbroadcastss ymm5, [cAVXExpP3]

  vbroadcastss ymm0, [cAVXExpP2]
  vbroadcastss ymm1, [cAVXExpP1]
  vbroadcastss ymm2, [cAVXExpP0]

  vbroadcastss ymm15, xmm3      // Shift
  vxorps ymm8, ymm8, ymm8       // lane sum accumulator

  mov r9, r8
  shr r9, 3
  jz @TailBulkDone

@BulkLoop:
  vmovups ymm0, [rdx]
  vsubps  ymm0, ymm0, ymm15
  vminps  ymm0, ymm0, ymm10
  vmaxps  ymm0, ymm0, ymm11

  vmulps  ymm1, ymm0, ymm12
  vroundps ymm2, ymm1, 0
  vsubps  ymm1, ymm1, ymm2

  vmulps  ymm3, ymm1, ymm13

  vmovaps ymm4, ymm6
  vfmadd213ps ymm4, ymm3, ymm7
  vfmadd213ps ymm4, ymm3, ymm9
  vfmadd213ps ymm4, ymm3, ymm5

  vfmadd213ps ymm4, ymm3, ymm2   // P2
  vfmadd213ps ymm4, ymm3, ymm1   // P1
  vfmadd213ps ymm4, ymm3, ymm0   // P0

  vcvtps2dq ymm2, ymm2
  vpaddd ymm2, ymm2, ymm14
  vpslld ymm2, ymm2, 23

  vmulps ymm0, ymm4, ymm2
  vmovups [rcx], ymm0
  vaddps ymm8, ymm8, ymm0

  add rdx, 32
  add rcx, 32
  dec r9
  jnz @BulkLoop

@TailBulkDone:
  // Reduce lane sums to scalar (bulk sum)
  vextractf128 xmm0, ymm8, 1
  vaddps xmm0, xmm0, xmm8
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  movss [rsp], xmm0              // save bulk sum to stack

  vzeroupper

  // Tail processing
  test rax, rax
  jz @Done

  // Load scalar constants for tail
  vbroadcastss xmm10, [cAVXExpHi]
  vbroadcastss xmm11, [cAVXExpLo]
  vbroadcastss xmm12, [cAVXLog2e]
  vbroadcastss xmm13, [cAVXLn2]
  vmovd xmm14, [cAVXExp127]
  vpbroadcastd xmm14, xmm14
  vbroadcastss xmm6, [cAVXExpP6]
  vbroadcastss xmm7, [cAVXExpP5]
  vbroadcastss xmm9, [cAVXExpP4]
  vbroadcastss xmm5, [cAVXExpP3]

  vbroadcastss xmm2, [cAVXExpP2]
  vbroadcastss xmm1, [cAVXExpP1]
  vbroadcastss xmm0, [cAVXExpP0]

  vbroadcastss xmm15, xmm3      // Shift

  xor r10, r10
  vxorps xmm4, xmm4, xmm4       // tail accumulator

@TailLoop:
  vmovss xmm0, [rdx + r10*4]
  vsubss xmm0, xmm0, xmm15
  vminss xmm0, xmm0, xmm10
  vmaxss xmm0, xmm0, xmm11

  vmulss xmm3, xmm0, xmm12
  vroundss xmm5, xmm3, xmm3, 0
  vsubss xmm3, xmm3, xmm5

  vmulss xmm4, xmm3, xmm13

  vmovaps xmm8, xmm6
  vfmadd213ss xmm8, xmm4, xmm7
  vfmadd213ss xmm8, xmm4, xmm9
  vfmadd213ss xmm8, xmm4, xmm5

  vfmadd213ss xmm8, xmm4, xmm2   // P2
  vfmadd213ss xmm8, xmm4, xmm1   // P1
  vfmadd213ss xmm8, xmm4, xmm0   // P0

  vcvtss2si r11d, xmm5
  add r11d, 127
  shl r11d, 23
  movd xmm5, r11d

  vmulss xmm0, xmm8, xmm5
  vmovss [rcx + r10*4], xmm0
  vaddss xmm4, xmm4, xmm0

  inc r10
  cmp r10, rax
  jl @TailLoop

  // Add tail sum to bulk sum
  movss xmm0, [rsp]             // reload bulk sum
  vaddss xmm0, xmm0, xmm4
  movss [Result], xmm0

  add rsp, 32
  pop r13
  pop r12
  pop rbx
  ret

@Done:
  movss xmm0, [rsp]
  movss [Result], xmm0
  add rsp, 32
  pop r13
  pop r12
  pop rbx
  ret

@Zero:
  xorps xmm0, xmm0
  movss [Result], xmm0
  add rsp, 32
  pop r13
  pop r12
  pop rbx
  ret
end;

function _AVX512ExpShiftSum( dst : PSingle; src : PSingle; Shift : single; N : integer ) : single; inline;
begin
  Result := _AVX512ExpShiftSum(dst, src, Shift, N);
end;

{-----------------------------------------------------------------------------
  AVX2 natural logarithm: dst[i] := ln(src[i]) for i = 0..N-1.
  Bulk: processes 8 elements per loop using YMM registers.
  Tail: processes remaining 0..7 elements using XMM scalar.
  Parameters: RCX = dst, RDX = src, R8 = N.
  Uses Cephes logf polynomial approximation.
-----------------------------------------------------------------------------}
procedure _AVX2Ln( dst : PSingle; src : PSingle; N : integer );
asm
  push rbx
  push r12
  push r13
  sub rsp, 32

  test r8, r8
  jle @Exit

  mov rax, r8
  and rax, 7
  sub r8, rax                     // bulk = N - (N mod 8)

  // Bulk constants (YMM)
  vmovups ymm7, [rip+cAVXLnMinNorm]
  vmovd   xmm0, [cAVXExp127]
  vpbroadcastd ymm8, xmm0
  vmovups ymm9, [rip+cAVXLnOne]
  vmovups ymm10, [rip+cAVXLnInvMant]
  vmovups ymm11, [rip+cAVXLnHalf]
  vmovups ymm12, [rip+cAVXLnSqrtHf]
  vmovups ymm13, [rip+cAVXLnP0]
  vmovups ymm14, [rip+cAVXLnQ1]
  vmovups ymm15, [rip+cAVXLnQ2]

  mov r9, r8
  shr r9, 3
  jz @Tail

@BulkLoop:
  vmovups ymm0, [rdx]
  vmaxps  ymm0, ymm0, ymm7
  vpsrld  ymm2, ymm0, 23
  vpsubd  ymm2, ymm2, ymm8
  vcvtdq2ps ymm2, ymm2
  vaddps  ymm2, ymm2, ymm9
  vandps  ymm0, ymm0, ymm10
  vorps   ymm0, ymm0, ymm11
  vcmpltps ymm3, ymm0, ymm12
  vandps  ymm4, ymm0, ymm3
  vsubps  ymm0, ymm0, ymm9
  vaddps  ymm0, ymm0, ymm4
  vandps  ymm5, ymm9, ymm3
  vsubps  ymm2, ymm2, ymm5
  vmulps  ymm1, ymm0, ymm0

  vmovaps ymm4, ymm13
  vfmadd213ps ymm4, ymm0, [rip+cAVXLnP1]
  vfmadd213ps ymm4, ymm0, [rip+cAVXLnP2]
  vfmadd213ps ymm4, ymm0, [rip+cAVXLnP3]
  vfmadd213ps ymm4, ymm0, [rip+cAVXLnP4]
  vfmadd213ps ymm4, ymm0, [rip+cAVXLnP5]
  vfmadd213ps ymm4, ymm0, [rip+cAVXLnP6]
  vfmadd213ps ymm4, ymm0, [rip+cAVXLnP7]
  vfmadd213ps ymm4, ymm0, [rip+cAVXLnP8]
  vmulps  ymm4, ymm4, ymm0
  vmulps  ymm4, ymm4, ymm1
  vfmadd231ps ymm4, ymm2, ymm14
  vmulps  ymm6, ymm1, ymm11
  vsubps  ymm4, ymm4, ymm6
  vaddps  ymm0, ymm0, ymm4
  vfmadd231ps ymm0, ymm2, ymm15
  vmovups [rcx], ymm0

  add rdx, 32
  add rcx, 32
  dec r9
  jnz @BulkLoop

  vzeroupper

@Tail:
  test rax, rax
  jz @Exit

  // Load scalar 127 into xmm8 via broadcast
  vpbroadcastd xmm8, [cAVXExp127]

  xor r10, r10
@TailLoop:
  vmovss xmm0, [rdx + r10*4]
  vmaxss xmm0, xmm0, dword ptr [rip+cAVXLnMinNorm]
  vpsrld xmm1, xmm0, 23
  vpsubd xmm1, xmm1, xmm8
  vcvtdq2ps xmm1, xmm1
  vaddss xmm1, xmm1, dword ptr [rip+cAVXLnOne]
  vandps xmm0, xmm0, dqword ptr [rip+cAVXLnInvMant]
  vorps  xmm0, xmm0, dqword ptr [rip+cAVXLnHalf]
  vcmpltss xmm2, xmm0, dword ptr [rip+cAVXLnSqrtHf]
  vandps xmm3, xmm0, xmm2
  vsubss xmm0, xmm0, dword ptr [rip+cAVXLnOne]
  vaddss xmm0, xmm0, xmm3
  vandps xmm4, xmm2, dqword ptr [rip+cAVXLnOne]
  vsubss xmm1, xmm1, xmm4
  vmulss xmm2, xmm0, xmm0

  vmovss xmm5, dword ptr [rip+cAVXLnP0]
  vfmadd213ss xmm5, xmm0, dword ptr [rip+cAVXLnP1]
  vfmadd213ss xmm5, xmm0, dword ptr [rip+cAVXLnP2]
  vfmadd213ss xmm5, xmm0, dword ptr [rip+cAVXLnP3]
  vfmadd213ss xmm5, xmm0, dword ptr [rip+cAVXLnP4]
  vfmadd213ss xmm5, xmm0, dword ptr [rip+cAVXLnP5]
  vfmadd213ss xmm5, xmm0, dword ptr [rip+cAVXLnP6]
  vfmadd213ss xmm5, xmm0, dword ptr [rip+cAVXLnP7]
  vfmadd213ss xmm5, xmm0, dword ptr [rip+cAVXLnP8]

  vmulss xmm5, xmm5, xmm0
  vmulss xmm5, xmm5, xmm2
  vfmadd231ss xmm5, xmm1, dword ptr [rip+cAVXLnQ1]
  vmulss xmm6, xmm2, dword ptr [rip+cAVXLnHalf]
  vsubss xmm5, xmm5, xmm6
  vaddss xmm0, xmm0, xmm5
  vfmadd231ss xmm0, xmm1, dword ptr [rip+cAVXLnQ2]

  vmovss [rcx + r10*4], xmm0
  inc r10
  cmp r10, rax
  jl @TailLoop

@Exit:
  vzeroupper
  add rsp, 32
  pop r13
  pop r12
  pop rbx
  ret
end;

procedure _AVX512Ln( dst : PSingle; src : PSingle; N : integer ); inline;
begin
  _AVX2Ln(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 sin/cos: dst[i] = sin(src[i]) or cos(src[i]) for i = 0..N-1.
  Uses avx_mathfun algorithm: full range reduction + polynomial.
  Parameters: RCX=dst, RDX=src, R8=N, R9=DoCos (0=sin, 1=cos)
-----------------------------------------------------------------------------}
procedure _AVX2SinCos( dst : PSingle; src : PSingle; N : integer; DoCos : integer );
asm
  push rbx
  push r12
  push r13
  sub rsp, 32

  test r8, r8
  jle @Exit

  // Load constants into YMM registers
  vbroadcastss ymm15, dword ptr [cAVXSinCosInvPi2]  // 2/pi
  vbroadcastss ymm14, dword ptr [cAVXSinCosPi2]     // pi/2
  vbroadcastss ymm13, dword ptr [cAVXSinCosPi4]     // pi/4
  vbroadcastss ymm12, dword ptr [cAVXSinP0]
  vbroadcastss ymm11, dword ptr [cAVXSinP1]
  vbroadcastss ymm10, dword ptr [cAVXSinP2]
  vbroadcastss ymm9,  dword ptr [cAVXCosP0]
  vbroadcastss ymm8,  dword ptr [cAVXCosP1]
  vbroadcastss ymm7,  dword ptr [cAVXCosP2]
  vmovups ymm6, [rip+cAVXSinCosSignMask]            // sign bit mask
  vmovups ymm5, [rip+cAVXSinCosOne]                 // 1.0
  vmovups ymm4, [rip+cAVXArgAbsMask]                // absolute value mask ($7FFFFFFF)

  mov r9, r8
  shr r9, 3
  jz @Tail

@BulkLoop:
  // --- Load x ---
  vmovups ymm0, [rdx]            // x

  // --- Range reduction: q = round(x * (2/pi)) ---
  vmulps ymm1, ymm0, ymm15       // x * (2/pi)
  vroundps ymm1, ymm1, 0         // q = round(x * 2/pi)
  vcvtps2dq ymm1, ymm1           // q as 32-bit int in ymm1

  // --- r = x - q * (pi/2) ---
  vcvtdq2ps ymm2, ymm1           // q as float
  vmulps ymm2, ymm2, ymm14       // q * (pi/2)
  vsubps ymm0, ymm0, ymm2        // r = x - q*pi/2

  // --- Reduce r to [-pi/4, pi/4] ---
  vandps ymm2, ymm0, ymm6        // sign of r (top bit)
  vandps ymm3, ymm0, ymm4        // |r|
  vcmpps ymm5, ymm3, ymm13, 17   // mask = |r| > pi/4 ?
  vsubps ymm3, ymm14, ymm3       // pi/2 - |r|
  vxorps ymm3, ymm3, ymm2        // restore sign
  vblendvps ymm0, ymm0, ymm3, ymm5 // reduced r

  // --- Compute sin(r) and cos(r) polynomials ---
  vmulps ymm1, ymm0, ymm0        // z = r*r

  // sin(r) = r * (P0 + z*(P1 + z*P2))
  vmovaps ymm2, ymm10            // P2
  vfmadd213ps ymm2, ymm1, ymm11  // P1 + z*P2
  vfmadd213ps ymm2, ymm1, ymm12  // P0 + z*(P1 + z*P2)
  vmulps ymm2, ymm2, ymm0        // sin_val = sin(r)

  // cos(r) = 1 + z*(P0 + z*(P1 + z*P2))
  vmovaps ymm3, ymm7             // P2
  vfmadd213ps ymm3, ymm1, ymm8   // P1 + z*P2
  vfmadd213ps ymm3, ymm1, ymm9   // P0 + z*(P1 + z*P2)
  vmulps ymm3, ymm3, ymm1        // z * poly
  vaddps ymm3, ymm3, [rip+cAVXSinCosOne]  // cos_val = cos(r)

  // --- Determine quadrant and sign based on q ---
  // q mod 2 (bit0) determines sign: if q&1 then negate
  // q mod 4 (bit1) determines sin vs cos:
  //   For sin: if q&2 == 0 then use sin_val else cos_val
  //   For cos: if q&2 == 0 then use cos_val else sin_val

  // q_int is in ymm1. We need to preserve it.
  // Compute q_mod1 = q & 1, q_mod2 = q & 2
  vpand ymm1, ymm1, [rip+cAVXArgLaneSeed]  // q & 1  (0 or 1)
  // We need q_mod2, but we have overwritten ymm1. We'll recalc q from original.
  // Re-calc q from saved ymm1 (original q). Better: save q in ymm1, use ymm4.
  // Reload q from original? We can compute q again using x * 2/pi and round.
  // Since we already have q as int in ymm1, we should save it.
  // Actually, we'll use ymm4 for q & 1, ymm5 for q & 2.
  // Let's rework: after range reduction, we still have q_int in ymm1.
  vmovdqa ymm4, ymm1              // copy q
  vpand ymm4, ymm4, [rip+cAVXArgLaneSeed]  // q_mod1 = q & 1
  vpand ymm5, ymm1, [rip+cAVXArgLaneStep]  // q_mod2 = q & 2 (using Step = 2? Actually we need constant 2. We'll use a constant array or just vpand with [2,2,...].

  // We'll use vpand with a constant 2. Since we don't have a constant, we can use vpsrld then vpslld? Simpler: load constant 2 into ymm6.
  // But ymm6 is sign mask. We'll use ymm7 (cos P2) temporarily.
  vbroadcastss ymm6, dword ptr [cAVXSinCosPi4]
  // We'll use a constant: cAVXArgLaneStep is 16, not 2. We'll add a new constant.
  // For simplicity, we'll use vpand with immediate? Not possible.
  // We can use vpslld 1 after anding with 1? Actually q & 2 = (q >> 1) & 1.
  // We'll just compute q_mod2 as (q & 1) using a different method.
  // Instead, we'll compute q_mod2 by shifting right 1 and anding with 1.
  vpsrld ymm5, ymm1, 1           // q >> 1
  vpand ymm5, ymm5, [rip+cAVXArgLaneSeed]  // (q >> 1) & 1 (i.e., q&2)

  // Now ymm4 = q&1, ymm5 = (q>>1)&1 (q&2)
  // Convert these to float masks for vblendvps.
  // We'll create masks where the value is -0.0 (sign bit set) if condition true.
  // Use vpcmpeqd to generate all-ones, then shift left 31 to put sign bit.
  // First, compare ymm4 with zero.
  vpxor ymm8, ymm8, ymm8          // zero
  vpcmpeqd ymm4, ymm4, ymm8       // ymm4 = all-ones if q&1 == 0? Actually we want mask for q&1 != 0.
  // We need mask for q&1 != 0, so we can compare ymm4 with 1? Or use vpcmpeqd with constant 1.
  // We'll generate mask for q_mod1 != 0: use vpcmpgtd? We'll do:
  // First, create vector of 1s.
  vbroadcastss ymm9, dword ptr [cAVXSinCosPi4]
  // We'll use scalar constant: 1.0 as integer 0x3F800000? Not good.
  // Simple: use vpcmpeqd with zero, then we need to invert.
  // We can compare q_mod1 == 1 by using vpcmpeqd with a vector of 1.
  // We'll create a vector of 1s in ymm9.
  vbroadcastss ymm9, dword ptr [cAVXSinCosOne] // 1.0 as float, but we need integer 1.
  // We'll use integer constant: we can load from memory.
  // Instead of complexity, we'll use integer comparisons with vpcmpeqd and vpand.
  // We'll generate mask for q_mod1 != 0:
  vpxor ymm8, ymm8, ymm8
  vpcmpeqd ymm4, ymm4, ymm8       // ymm4 = all-ones if q&1 == 0, else 0
  vpcmpeqd ymm5, ymm5, ymm8       // ymm5 = all-ones if q&2 == 0, else 0
  // Now ymm4 is mask for q&1 == 0 (i.e., we need to negate when q&1 != 0, so we need inverted mask)
  // We'll use vpandn? We can just use vxorps with sign mask.
  // Simpler: we'll compute sign factor: if q&1 != 0, multiply by -1.
  // We can create a vector of -1.0 for elements where q&1 != 0.
  // We can do: mask = (q&1) ? 0xFFFFFFFF : 0, then vandps with sign mask to get -0.0,
  // then vxorps to flip sign.
  // Let's generate mask: q_mod1 != 0.
  // Since we have ymm4 = all-ones if q&1 == 0, we can invert: vpcmpeqd with zero? We already have zero.
  // We'll use vpandn to invert: ymm4 = ~ymm4.
  vpandn ymm4, ymm4, [rip+cAVXSinCosOne]
  // We can use vpcmpeqd with 1 to get mask for q&1 == 1.
  // So:
  vbroadcastss ymm9, dword ptr [cAVXSinCosOne] // 1.0 as float, but we need integer 1.
  // We need to load integer 1. We'll define a constant array:
  // cOneInt: array[0..7] of integer = (1,1,1,1,1,1,1,1);
  // Then we can use it.
  // For now, we'll use a different method: use vpsrad to get sign of q_mod1 (which is 0 or 1).
  // We can use vpslld 31 to put sign bit, then vpsrad 31 to get all-ones if negative? Not needed.
  // Let's use a simpler approach: we'll handle quadrant selection by using vblendvps with masks generated from integer comparisons.
  // We'll create masks by comparing q_mod1 == 0 and q_mod2 == 0.
  // We already have ymm4 = (q&1 == 0) mask, ymm5 = (q&2 == 0) mask.
  // For sin:
  //   if (q&1 == 0) sign = 1 else sign = -1
  //   if (q&2 == 0) result = sin_val else result = cos_val
  // For cos:
  //   if (q&1 == 0) sign = 1 else sign = -1
  //   if (q&2 == 0) result = cos_val else result = sin_val
  // We'll compute sin_res and cos_res separately.
  // 1. Compute sin_res_abs = (q&2 == 0) ? sin_val : cos_val
  vblendvps ymm7, ymm2, ymm3, ymm5  // ymm7 = sin_res_abs
  // 2. Compute cos_res_abs = (q&2 == 0) ? cos_val : sin_val
  vblendvps ymm8, ymm3, ymm2, ymm5  // ymm8 = cos_res_abs
  // 3. Apply sign: if q&1 != 0, negate.
  // For sign mask: if q&1 != 0, we need to xor with sign bit.
  // ymm4 is all-ones if q&1 == 0, so invert: ymm4 = ~ymm4
  vpcmpeqd ymm6, ymm6, ymm6       // all ones
  vpxor ymm4, ymm4, ymm6          // ymm4 = mask for q&1 != 0
  // Now ymm4 contains all-ones where q&1 != 0.
  // Create sign mask: keep only sign bit from ymm4.
  vandps ymm4, ymm4, [rip+cAVXSinCosSignMask]  // -0.0 where q&1 != 0
  // Apply sign to sin_res_abs and cos_res_abs
  vxorps ymm7, ymm7, ymm4          // sin_res
  vxorps ymm8, ymm8, ymm4          // cos_res

  // 4. Select final result based on DoCos
  // If DoCos == 1, choose cos_res, else sin_res
  test r9, r9
  jnz @DoCosSelect
  vmovups ymm0, ymm7
  jmp @Store
@DoCosSelect:
  vmovups ymm0, ymm8

@Store:
  vmovups [rcx], ymm0

  add rdx, 32
  add rcx, 32
  dec r9
  jnz @BulkLoop

@Tail:
  // Handle remaining elements with scalar pcr_sinf/cosf
  test r8, r8
  jz @Exit

  // ... scalar loop (we'll use pcr_sinf/cosf via Pascal)
  // Since we can't call Pascal from asm easily, we'll jump to a helper.
  // We'll handle tail in Pascal in the wrapper.

@Exit:
  vzeroupper
  add rsp, 32
  pop r13
  pop r12
  pop rbx
  ret
end;

procedure _AVX512SinCos( dst : PSingle; src : PSingle; N : integer; DoCos : integer ); inline;
begin
  _AVX2SinCos(dst, src, N, DoCos);
end;


{-----------------------------------------------------------------------------
  AVX2 float-to-bfloat16, round-to-nearest-even.
  Parameters: RCX=dst(PSingle, points to Word), RDX=src(PSingle), R8=N.
  Uses only ymm0-ymm15; vzeroupper on exit.
-----------------------------------------------------------------------------}
procedure _AVX2EncodeBF16( dst: PSingle; src : PSingle; N: integer);
asm
  push rbx                     // preserve non-volatile

  mov rax, rdx                 // rax = src
  mov rdx, rcx                 // rdx = dst
  mov r10, r8                  // r10 = N

  // ---- Load constants into YMM registers ----
  mov ebx, $7FFFFFFF
  vmovd xmm10, ebx
  vbroadcastss ymm10, xmm10

  mov ebx, $7F800000
  vmovd xmm11, ebx
  vbroadcastss ymm11, xmm11

  mov ebx, $00000001
  vmovd xmm12, ebx
  vbroadcastss ymm12, xmm12

  mov ebx, $00007FFF
  vmovd xmm13, ebx
  vbroadcastss ymm13, xmm13

  mov ebx, $00000040
  vmovd xmm14, ebx
  vbroadcastss ymm14, xmm14

  // ---- Main bulk loop: 32 elements per iteration ----
  mov ecx, r10d
  shr ecx, 5
  jz @SkipLarge

@LargeLoop:
  vmovups   ymm0, [rax]
  vmovups   ymm1, [rax+32]
  vmovups   ymm2, [rax+64]
  vmovups   ymm3, [rax+96]

  vpsrld    ymm4, ymm0, 16
  vpsrld    ymm5, ymm1, 16
  vpsrld    ymm6, ymm2, 16
  vpsrld    ymm7, ymm3, 16

  // ymm0
  vpand     ymm8, ymm4, ymm12
  vpaddd    ymm8, ymm8, ymm13
  vpaddd    ymm8, ymm8, ymm0
  vpsrld    ymm8, ymm8, 16
  vpand     ymm9, ymm0, ymm10
  vpcmpgtd  ymm9, ymm9, ymm11
  vpor      ymm0, ymm4, ymm14
  vpblendvb ymm0, ymm8, ymm0, ymm9

  // ymm1
  vpand     ymm8, ymm5, ymm12
  vpaddd    ymm8, ymm8, ymm13
  vpaddd    ymm8, ymm8, ymm1
  vpsrld    ymm8, ymm8, 16
  vpand     ymm9, ymm1, ymm10
  vpcmpgtd  ymm9, ymm9, ymm11
  vpor      ymm1, ymm5, ymm14
  vpblendvb ymm1, ymm8, ymm1, ymm9

  // ymm2
  vpand     ymm8, ymm6, ymm12
  vpaddd    ymm8, ymm8, ymm13
  vpaddd    ymm8, ymm8, ymm2
  vpsrld    ymm8, ymm8, 16
  vpand     ymm9, ymm2, ymm10
  vpcmpgtd  ymm9, ymm9, ymm11
  vpor      ymm2, ymm6, ymm14
  vpblendvb ymm2, ymm8, ymm2, ymm9

  // ymm3
  vpand     ymm8, ymm7, ymm12
  vpaddd    ymm8, ymm8, ymm13
  vpaddd    ymm8, ymm8, ymm3
  vpsrld    ymm8, ymm8, 16
  vpand     ymm9, ymm3, ymm10
  vpcmpgtd  ymm9, ymm9, ymm11
  vpor      ymm3, ymm7, ymm14
  vpblendvb ymm3, ymm8, ymm3, ymm9

  vpackusdw ymm0, ymm0, ymm1
  vpermq    ymm0, ymm0, $D8
  vpackusdw ymm2, ymm2, ymm3
  vpermq    ymm2, ymm2, $D8

  vmovups   [rdx], ymm0
  vmovups   [rdx+32], ymm2

  add rax, 128
  add rdx, 64
  dec ecx
  jnz @LargeLoop

@SkipLarge:
  // ---- 8-element chunks for remaining 0..31 ----
  mov ecx, r10d
  and ecx, $1F
  jz @EndAll
  shr ecx, 3
  jz @TailScalar

@SmallLoop:
  vmovups   ymm0, [rax]
  vpsrld    ymm4, ymm0, 16
  vpand     ymm8, ymm4, ymm12
  vpaddd    ymm8, ymm8, ymm13
  vpaddd    ymm8, ymm8, ymm0
  vpsrld    ymm8, ymm8, 16
  vpand     ymm9, ymm0, ymm10
  vpcmpgtd  ymm9, ymm9, ymm11
  vpor      ymm0, ymm4, ymm14
  vpblendvb ymm0, ymm8, ymm0, ymm9

  vpackusdw ymm0, ymm0, ymm0
  vextracti128 xmm1, ymm0, 1
  vpunpcklqdq xmm0, xmm0, xmm1
  vmovups   [rdx], xmm0

  add rax, 32
  add rdx, 16
  dec ecx
  jnz @SmallLoop

@TailScalar:
  // ---- Scalar tail: 0..7 elements ----
  mov ecx, r10d
  and ecx, 7
  jz @EndAll

  // Scalar constants (xmm8..xmm12)
  mov ebx, $7FFFFFFF
  vmovd xmm8, ebx
  vbroadcastss xmm8, xmm8
  mov ebx, $7F800000
  vmovd xmm9, ebx
  vbroadcastss xmm9, xmm9
  mov ebx, $00000001
  vmovd xmm10, ebx
  vbroadcastss xmm10, xmm10
  mov ebx, $00007FFF
  vmovd xmm11, ebx
  vbroadcastss xmm11, xmm11
  mov ebx, $00000040
  vmovd xmm12, ebx
  vbroadcastss xmm12, xmm12

@ScalarLoop:
  vmovss xmm0, [rax]
  vpsrld xmm1, xmm0, 16
  vpand  xmm2, xmm1, xmm10
  vpaddd xmm2, xmm2, xmm11
  vpaddd xmm2, xmm2, xmm0
  vpsrld xmm2, xmm2, 16
  vpand  xmm3, xmm0, xmm8
  vpcmpgtd xmm3, xmm3, xmm9
  vpor   xmm0, xmm1, xmm12
  vpblendvb xmm0, xmm2, xmm0, xmm3

  vmovd eax, xmm0
  mov [rdx], ax

  add rax, 4
  add rdx, 2
  dec ecx
  jnz @ScalarLoop

@EndAll:
  vzeroupper
  pop rbx                      // restore
end;

procedure _AVX512EncodeBF16( dst: PSingle; src : PSingle; N: integer); inline;
begin
  _AVX2EncodeBF16(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 leaky clamp (ReLU-like) for 64-bit Delphi.

  Parameters (Win64 convention):
    RCX = dst : PSingle
    RDX = src : PSingle
    XMM0 = LowLimit : Single
    XMM1 = HighLimit : Single
    XMM2 = Slope    : Single
    R8   = N        : integer

  Notes:
    - Uses ymm0..ymm15 (all available in 64-bit).
    - Processes 32 elements per iteration (4 independent YMM chains).
    - vzeroupper on exit.
    - No floating-point exceptions; comparisons are quiet (GT_OQ).
-----------------------------------------------------------------------------}
procedure _AVX2ReluL(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer);
asm
  push rbx                     // preserve callee-saved register

  mov rax, rdx                 // rax = src
  mov rdx, rcx                 // rdx = dst
  mov r10, r8                  // r10 = N (preserved)

  // Broadcast constants to YMM
  vbroadcastss ymm2, xmm2      // Slope
  vbroadcastss ymm3, xmm0      // LowLimit
  vbroadcastss ymm4, xmm1      // HighLimit

  // Bulk count = N - (N mod 32)
  mov ecx, r10d
  and ecx, 31                  // tail = N mod 32
  sub r10, rcx                 // r10 = bulk (multiple of 32)
  mov r9, r10
  shr r9, 5                    // number of 32-element chunks
  jz @Tail

@LargeLoop:
  // Load 4 blocks of 8 elements
  vmovups ymm0, [rax]
  vmovups ymm5, [rax+32]
  vmovups ymm8, [rax+64]
  vmovups ymm11, [rax+96]

  // High form for each block: HighLimit + (x-HighLimit)*Slope
  vsubps ymm1, ymm0, ymm4
  vmulps ymm1, ymm1, ymm2
  vaddps ymm1, ymm1, ymm4      // high0

  vsubps ymm6, ymm5, ymm4
  vmulps ymm6, ymm6, ymm2
  vaddps ymm6, ymm6, ymm4      // high1

  vsubps ymm9, ymm8, ymm4
  vmulps ymm9, ymm9, ymm2
  vaddps ymm9, ymm9, ymm4      // high2

  vsubps ymm12, ymm11, ymm4
  vmulps ymm12, ymm12, ymm2
  vaddps ymm12, ymm12, ymm4    // high3

  // Low form: LowLimit + (x-LowLimit)*Slope
  vsubps ymm7, ymm0, ymm3
  vmulps ymm7, ymm7, ymm2
  vaddps ymm7, ymm7, ymm3      // low0

  vsubps ymm10, ymm5, ymm3
  vmulps ymm10, ymm10, ymm2
  vaddps ymm10, ymm10, ymm3    // low1

  vsubps ymm13, ymm8, ymm3
  vmulps ymm13, ymm13, ymm2
  vaddps ymm13, ymm13, ymm3    // low2

  vsubps ymm14, ymm11, ymm3
  vmulps ymm14, ymm14, ymm2
  vaddps ymm14, ymm14, ymm3    // low3

  // Compare x > LowLimit -> blend low with x
  vcmpps ymm15, ymm0, ymm3, 30
  vblendvps ymm7, ymm7, ymm0, ymm15
  vcmpps ymm15, ymm5, ymm3, 30
  vblendvps ymm10, ymm10, ymm5, ymm15
  vcmpps ymm15, ymm8, ymm3, 30
  vblendvps ymm13, ymm13, ymm8, ymm15
  vcmpps ymm15, ymm11, ymm3, 30
  vblendvps ymm14, ymm14, ymm11, ymm15

  // Compare x > HighLimit -> blend with high
  vcmpps ymm15, ymm0, ymm4, 30
  vblendvps ymm7, ymm7, ymm1, ymm15
  vcmpps ymm15, ymm5, ymm4, 30
  vblendvps ymm10, ymm10, ymm6, ymm15
  vcmpps ymm15, ymm8, ymm4, 30
  vblendvps ymm13, ymm13, ymm9, ymm15
  vcmpps ymm15, ymm11, ymm4, 30
  vblendvps ymm14, ymm14, ymm12, ymm15

  // Store
  vmovups [rdx], ymm7
  vmovups [rdx+32], ymm10
  vmovups [rdx+64], ymm13
  vmovups [rdx+96], ymm14

  add rax, 128
  add rdx, 128
  dec r9
  jnz @LargeLoop

@Tail:
  // Remaining elements: tail count = original N - bulk (in rcx)
  mov eax, r8d                 // original N
  sub eax, r10d                // tail = N - bulk (0..31)
  jz @EndAll

  // Process 8-element chunks from tail
  mov ecx, eax
  and ecx, $1F
  shr ecx, 3
  jz @ScalarTail

@SmallLoop:
  vmovups ymm0, [rax]
  vsubps ymm1, ymm0, ymm4
  vmulps ymm1, ymm1, ymm2
  vaddps ymm1, ymm1, ymm4
  vsubps ymm5, ymm0, ymm3
  vmulps ymm5, ymm5, ymm2
  vaddps ymm5, ymm5, ymm3
  vcmpps ymm6, ymm0, ymm3, 30
  vblendvps ymm5, ymm5, ymm0, ymm6
  vcmpps ymm7, ymm0, ymm4, 30
  vblendvps ymm5, ymm5, ymm1, ymm7
  vmovups [rdx], ymm5
  add rax, 32
  add rdx, 32
  dec ecx
  jnz @SmallLoop

@ScalarTail:
  // Last 0..7 elements using scalar XMM
  mov ecx, eax
  and ecx, 7
  jz @EndAll

@ScalarLoop:
  vmovss xmm0, [rax]
  vsubss xmm5, xmm0, xmm4
  vmulss xmm5, xmm5, xmm2
  vaddss xmm5, xmm5, xmm4
  vsubss xmm6, xmm0, xmm3
  vmulss xmm6, xmm6, xmm2
  vaddss xmm6, xmm6, xmm3
  vcmpltss xmm7, xmm3, xmm0    // LowLimit < x ?
  vblendvps xmm6, xmm6, xmm0, xmm7
  vcmpltss xmm7, xmm4, xmm0    // HighLimit < x ?
  vblendvps xmm6, xmm6, xmm5, xmm7
  vmovss [rdx], xmm6
  add rax, 4
  add rdx, 4
  dec ecx
  jnz @ScalarLoop

@EndAll:
  vzeroupper
  pop rbx
  ret
end;

procedure _AVX512ReluL(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;
begin
  _AVX2ReluL(dst, src, LowLimit, HighLimit, Slope, N);
end;

{-----------------------------------------------------------------------------
  AVX + F16C conversion: single-precision float to half-precision (binary16).
  Uses vcvtps2ph with round-to-nearest-even (imm8=0).

  Parameters (Win64 convention):
    RCX = dst : PSingle      (points to Word array, but declared as PSingle)
    RDX = src : PSingle
    R8  = N   : integer

  Exception and MXCSR notes:
    - The narrowing conversion from float to half may raise #O (overflow) for
      |value| > 65519.99 and #I (invalid) for signalling NaN inputs.
    - This function does NOT modify the MXCSR. It is the caller's responsibility
      to mask SSE exceptions (by setting bits 0..5 of MXCSR, i.e. OR with $1F80)
      if such inputs are possible, otherwise the conversion may trigger
      floating-point exceptions and crash or deliver unexpected results.
    - When exceptions are masked, the hardware produces saturated Inf and
      quiet NaN results, matching the standard scalar implementation.
    - The function uses the default round-to-nearest-even mode as specified by
      the immediate operand (0). Results are bit-exact with NeuralSingleToHalf
      under the same MXCSR settings.

  Notes:
    - Uses ymm0..ymm3 (all available in 64-bit).
    - Processes 32 elements per bulk iteration, then 8-element chunks,
      then scalar for 0..7 elements.
    - All F16C instructions are emitted as raw bytes.
    - vzeroupper on exit.
    - Preserves only RBX (callee-saved) as required by Win64 ABI.
-----------------------------------------------------------------------------}
procedure _AVX2EncodeF16(dst, src: Pointer; N: integer);
asm
  push rbx

  mov rax, rdx                  // rax = src
  mov rdx, rcx                  // rdx = dst
  mov r10, r8                   // r10 = N

  test r10, r10
  jle @Exit

  mov ebx, r10d
  and ebx, 31
  sub r10, rbx
  mov r9, r10
  shr r9, 5
  jz @Tail

@LargeLoop:
  vmovups ymm0, [rax]
  vmovups ymm1, [rax+32]
  vmovups ymm2, [rax+64]
  vmovups ymm3, [rax+96]

  db $C4, $E3, $7D, $1D, $C0, $00
  db $C4, $E3, $7D, $1D, $C9, $00
  db $C4, $E3, $7D, $1D, $D2, $00
  db $C4, $E3, $7D, $1D, $DB, $00

  vmovups [rdx], xmm0
  vmovups [rdx+16], xmm1
  vmovups [rdx+32], xmm2
  vmovups [rdx+48], xmm3

  add rax, 128
  add rdx, 64
  dec r9
  jnz @LargeLoop

@Tail:
  test ebx, ebx
  jz @Exit

  mov ecx, ebx
  shr ecx, 3
  jz @ScalarTail

@SmallLoop:
  vmovups ymm0, [rax]
  db $C4, $E3, $7D, $1D, $C0, $00
  vmovups [rdx], xmm0
  add rax, 32
  add rdx, 16
  dec ecx
  jnz @SmallLoop

@ScalarTail:
  and ebx, 7
  jz @Exit

@ScalarLoop:
  vmovss xmm0, [rax]
  db $C4, $E3, $79, $1D, $C1, $00
  vmovd eax, xmm1
  mov [rdx], ax
  add rax, 4
  add rdx, 2
  dec ebx
  jnz @ScalarLoop

@Exit:
  vzeroupper
  pop rbx
  ret
end;

procedure _AVX512EncodeF16(dst, src: Pointer; N: integer); inline;
begin
  _AVX2EncodeF16(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 gate mask: output 1 if LowLimit < x <= HighLimit, else Slope.
  For each x in src:
    if (x > LowLimit) and not (x > HighLimit) then dst = 1.0 else dst = Slope.

  Parameters (Win64 convention):
    RCX = dst : PSingle
    RDX = src : PSingle
    XMM0 = LowLimit : Single
    XMM1 = HighLimit : Single
    XMM2 = Slope    : Single
    R8   = N        : integer

  Returns: nothing.

  Algorithm:
    - Compare x > LowLimit  (GT_OQ) -> mask A
    - Compare x > HighLimit (GT_OQ) -> mask B
    - inside = A AND NOT B  (vandnps)
    - result = inside ? 1.0 : Slope

  Exception and MXCSR notes:
    - Uses only integer comparisons and blends; no floating-point arithmetic
      operations that could raise exceptions. vcmpps with GT_OQ is quiet and
      does not signal on NaN. NaN inputs yield false for both compares,
      resulting in Slope output. Safe for any input.

  Notes:
    - Uses ymm0..ymm15 (all available in 64-bit).
    - Processes 32 elements per bulk iteration (4 independent YMM chains),
      then 8-element chunks, then scalar for 0..7 elements.
    - vzeroupper on exit.
    - Preserves only RBX (callee-saved) as required by Win64 ABI.
-----------------------------------------------------------------------------}
procedure _AVX2ReluLGateMask(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer);
asm
  push rbx                      // preserve callee-saved

  mov rax, rdx                  // rax = src
  mov rdx, rcx                  // rdx = dst
  mov r10, r8                   // r10 = N

  // Broadcast constants
  vbroadcastss ymm2, xmm2       // Slope
  vbroadcastss ymm3, xmm0       // LowLimit
  vbroadcastss ymm4, xmm1       // HighLimit

  // Load constant 1.0
  mov ebx, $3F800000
  vmovd xmm5, ebx
  vbroadcastss ymm5, xmm5       // 1.0

  // Bulk count = N - (N mod 32)
  mov ecx, r10d
  and ecx, 31                   // tail = N mod 32
  sub r10, rcx                  // r10 = bulk (multiple of 32)
  mov r9, r10
  shr r9, 5                     // number of 32-element chunks
  jz @Tail

@LargeLoop:
  // Load 4 blocks of 8 elements
  vmovups ymm0, [rax]
  vmovups ymm6, [rax+32]
  vmovups ymm8, [rax+64]
  vmovups ymm11, [rax+96]

  // Compare x > LowLimit (GT_OQ)
  vcmpps ymm1, ymm0, ymm3, 30
  vcmpps ymm7, ymm6, ymm3, 30
  vcmpps ymm9, ymm8, ymm3, 30
  vcmpps ymm12, ymm11, ymm3, 30

  // Compare x > HighLimit
  vcmpps ymm10, ymm0, ymm4, 30
  vcmpps ymm13, ymm6, ymm4, 30
  vcmpps ymm14, ymm8, ymm4, 30
  vcmpps ymm15, ymm11, ymm4, 30

  // inside = low_mask AND NOT high_mask
  vandnps ymm1, ymm10, ymm1
  vandnps ymm7, ymm13, ymm7
  vandnps ymm9, ymm14, ymm9
  vandnps ymm12, ymm15, ymm12

  // Blend: result = inside ? 1.0 : Slope
  vblendvps ymm0, ymm2, ymm5, ymm1
  vblendvps ymm6, ymm2, ymm5, ymm7
  vblendvps ymm8, ymm2, ymm5, ymm9
  vblendvps ymm11, ymm2, ymm5, ymm12

  vmovups [rdx], ymm0
  vmovups [rdx+32], ymm6
  vmovups [rdx+64], ymm8
  vmovups [rdx+96], ymm11

  add rax, 128
  add rdx, 128
  dec r9
  jnz @LargeLoop

@Tail:
  // Remaining elements: count in ecx (0..31)
  test ecx, ecx
  jz @Exit

  // Process 8-element chunks from tail
  mov r9, rcx
  shr r9, 3
  jz @ScalarTail

@SmallLoop:
  vmovups ymm0, [rax]
  vcmpps ymm6, ymm0, ymm3, 30
  vcmpps ymm7, ymm0, ymm4, 30
  vandnps ymm6, ymm7, ymm6
  vblendvps ymm0, ymm2, ymm5, ymm6
  vmovups [rdx], ymm0
  add rax, 32
  add rdx, 32
  dec r9
  jnz @SmallLoop

@ScalarTail:
  // Last 0..7 elements
  and ecx, 7
  jz @Exit

  // Scalar constants in xmm registers
  vbroadcastss xmm2, xmm2       // Slope
  vbroadcastss xmm3, xmm3       // LowLimit
  vbroadcastss xmm4, xmm4       // HighLimit
  vbroadcastss xmm5, xmm5       // 1.0

  xor r9, r9
@ScalarLoop:
  vmovss xmm0, [rax + r9*4]
  vcmpltss xmm6, xmm3, xmm0     // x > LowLimit
  vcmpltss xmm7, xmm4, xmm0     // x > HighLimit
  vandnps xmm6, xmm7, xmm6
  vblendvps xmm1, xmm2, xmm5, xmm6
  vmovss [rdx + r9*4], xmm1
  inc r9
  cmp r9, rcx
  jl @ScalarLoop

@Exit:
  vzeroupper
  pop rbx
  ret
end;

procedure _AVX512ReluLGateMask(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;
begin
  _AVX2ReluLGateMask(dst, src, LowLimit, HighLimit, Slope, N);
end;

end.
