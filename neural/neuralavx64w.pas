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
procedure _AVX2FillMem( dst : PSingle; N : Integer; const fact : Single); register; assembler;
procedure _AVX512FillMem( dst : PSingle; N : Integer; const fact : Single); inline;

// Coded by DeepSeek (AI)
procedure _AVX2MulAddF( dst : PSingle; src : PSingle; N : Integer; const fact : Single); register; assembler;
procedure _AVX512MulAddF( dst : PSingle; src : PSingle; N : Integer; const fact : Single); inline;

// Coded by DeepSeek (AI)
procedure _AVX2MulMulAdd( dst : PSingle; src : PSingle; N : Integer; const mulOp1, mulOp2 : Single); register; assembler;
procedure _AVX512MulMulAdd( dst : PSingle; src : PSingle; N : Integer; const mulOp1, mulOp2 : Single); inline;

// Coded by DeepSeek (AI)
procedure _AVX2MulAdd( dst : PSingle; src : PSingle; z : PSingle; N : Integer); register; assembler;
procedure _AVX512MulAdd( dst : PSingle; src : PSingle; z : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2CopyRelu( dst : PSingle; src : PSingle; N : Integer); register; assembler;
procedure _AVX512CopyRelu( dst : PSingle; src : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2MulF( dst : PSingle; N : Integer; const factor : Single); register; assembler;
procedure _AVX512MulF( dst : PSingle; N : Integer; const factor : Single); inline;

// Coded by DeepSeek (AI)
procedure _AVX2Mul( dst : PSingle; src : PSingle; N : Integer); register; assembler;
procedure _AVX512Mul( dst : PSingle; src : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2Add( dst : PSingle; src : PSingle; N : Integer); register; assembler;
procedure _AVX512Add( dst : PSingle; src : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2Max( dst : PSingle; src : PSingle; N : Integer); register; assembler;
procedure _AVX512Max( dst : PSingle; src : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
function _AVX2SumDiff( dst : PSingle; src : PSingle; N : Integer ) : Single; register; assembler;
function _AVX512SumDiff( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
function _AVX2DistanceSqr( dst : PSingle; src : PSingle; N : Integer ) : Single; register; assembler;
function _AVX512DistanceSqr( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2Sub( dst : PSingle; src : PSingle; N : Integer); register; assembler;
procedure _AVX512Sub( dst : PSingle; src : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
function _AVX2GetSum( src : PSingle; N : Integer ) : Single; register; assembler;
function _AVX512GetSum( src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
function _AVX2GetSumSqr( src : PSingle; N : Integer ) : Single; register; assembler;
function _AVX512GetSumSqr( src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2Exp( dst : PSingle; src : PSingle; N : Integer); register; assembler;
procedure _AVX512Exp( dst : PSingle; src : PSingle; N : Integer); inline;

// Coded by DeepSeek (AI)
function _AVX2DotProd( dst : PSingle; src : PSingle; N : Integer ) : Single; register; assembler;
function _AVX512DotProd( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
function _AVX2DotProdInt8( dst : PShortInt; src : PSingle; N : Integer ) : Single; register; assembler;
function _AVX512DotProdInt8( dst : PShortInt; src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2MulAddInt8Scalar( dst : PSingle; codes : PShortInt; W : Single; N : Integer ); register; assembler;
procedure _AVX512MulAddInt8Scalar( dst : PSingle; codes : PShortInt; W : Single; N : Integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2MulAddInt8( dst : PSingle; src : PSingle; codes : PShortInt; N : Integer ); register; assembler;
procedure _AVX512MulAddInt8( dst : PSingle; src : PSingle; codes : PShortInt; N : Integer ); inline;

// Coded by DeepSeek (AI)
function _AVX2MaxAbsFinite( src : PSingle; N : Integer ) : Single; register; assembler;
function _AVX512MaxAbsFinite( src : PSingle; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2QuantizeInt8( dst : PShortInt; src : PSingle; N : Integer; const MaxAbs : Single ); register; assembler;
procedure _AVX512QuantizeInt8( dst : PShortInt; src : PSingle; N : Integer; const MaxAbs : Single ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2DequantizeInt8( dst : PSingle; src : PShortInt; N : Integer; const Scale : Single ); register; assembler;
procedure _AVX512DequantizeInt8( dst : PSingle; src : PShortInt; N : Integer; const Scale : Single ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2DecodeBF16( dst : PSingle; src : PWord; N : Integer ); register; assembler;
procedure _AVX512DecodeBF16( dst : PSingle; src : PWord; N : Integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2ReluGateMask( dst : PSingle; src : PSingle; N : Integer ); register; assembler;
procedure _AVX512ReluGateMask( dst : PSingle; src : PSingle; N : Integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2LeakyRelu( dst : PSingle; src : PSingle; N : Integer; const Slope : Single ); register; assembler;
procedure _AVX512LeakyRelu( dst : PSingle; src : PSingle; N : Integer; const Slope : Single ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2DecodeF16( dst : PSingle; src : PWord; N : Integer ); register; assembler;
procedure _AVX512DecodeF16( dst : PSingle; src : PWord; N : Integer ); inline;

// Coded by DeepSeek (AI)
function _AVX2SumSqrCentered( src : PSingle; Mean : Single; N : Integer ) : Single; register; assembler;
function _AVX512SumSqrCentered( src : PSingle; Mean : Single; N : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2AdamDelta( PtrDelta, PtrM, PtrV : PSingle;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR : Single;
  NumElements : Integer ); register; assembler;
procedure _AVX512AdamDelta( PtrDelta, PtrM, PtrV : PSingle;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR : Single;
  NumElements : Integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2AdafactorDelta( PtrDelta, PtrV : PSingle;
  Beta2, k, c, Epsilon : Single; NumElements : Integer ); register; assembler;
procedure _AVX512AdafactorDelta( PtrDelta, PtrV : PSingle;
  Beta2, k, c, Epsilon : Single; NumElements : Integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2ClampAbs( PtrA : PSingle; Value : Single; NumElements : Integer ); register; assembler;
procedure _AVX512ClampAbs( PtrA : PSingle; Value : Single; NumElements : Integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2LionDelta( PtrDelta, PtrM : PSingle; Beta1, k1, Beta2, k2, NegLR, PosLR : Single; NumElements : Integer ); register; assembler;
procedure _AVX512LionDelta( PtrDelta, PtrM : PSingle; Beta1, k1, Beta2, k2, NegLR, PosLR : Single; NumElements : Integer ); inline;

// Coded by DeepSeek (AI)
function _AVX2GetMaxPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; register; assembler;
function _AVX512GetMaxPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
function _AVX2GetMinPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; register; assembler;
function _AVX512GetMinPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;

// Coded by DeepSeek (AI)
function _AVX2GetMaxAbsPos( PtrA : PSingle; NumElements : integer; out Position : integer ) : single; register; assembler;
function _AVX512GetMaxAbsPos( PtrA : PSingle; NumElements : integer; out Position : integer ) : single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2AddScalar( PtrA : PSingle; Value : single; N : integer ); register; assembler;
procedure _AVX512AddScalar( PtrA : PSingle; Value : single; N : integer ); inline;

// Coded by DeepSeek (AI)
function _AVX2ExpShiftSum( dst : PSingle; src : PSingle; Shift : single; N : integer ) : single; register; assembler;
function _AVX512ExpShiftSum( dst : PSingle; src : PSingle; Shift : single; N : integer ) : single; inline;

// Coded by DeepSeek (AI)
procedure _AVX2Ln( dst : PSingle; src : PSingle; N : integer ); register; assembler;
procedure _AVX512Ln( dst : PSingle; src : PSingle; N : integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2SinCos( dst : PSingle; src : PSingle; N : integer; DoCos : integer ); register; assembler;
procedure _AVX512SinCos( dst : PSingle; src : PSingle; N : integer; DoCos : integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2EncodeBF16( dst: PSingle; src : PSingle; N : integer ); register; assembler;
procedure _AVX512EncodeBF16( dst: PSingle; src : PSingle; N : integer ); inline;

// Coded by DeepSeek (AI)
procedure _AVX2ReluL(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); register; assembler;
procedure _AVX512ReluL(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2EncodeF16(dst, src: Pointer; N: integer); register; assembler;
procedure _AVX512EncodeF16(dst, src: Pointer; N: integer); inline;

// Coded by DeepSeek (AI)
procedure _AVX2ReluLGateMask(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); register; assembler;
procedure _AVX512ReluLGateMask(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;


implementation


{-----------------------------------------------------------------------------
  AVX2 scalar multiply-add: dst[i] = dst[i] + fact * src[i].
  Optimized for Win64: processes 16 elements per iteration using 2 YMM blocks.
  Parameters:
    RCX = dst (PSingle)
    RDX = src (PSingle)
    R8  = N (Integer)
    XMM3 = fact (Single)
  Uses FMA instructions (requires AVX2+FMA capable CPU).
  All YMM registers used are volatile (YMM0-YMM5), no prologue/epilogue needed.
-----------------------------------------------------------------------------}
procedure _AVX2MulAddF(dst: PSingle; src: PSingle; N: Integer; const fact: Single);
asm
  // Load parameters into volatile registers
  mov r10, rcx                // r10 = dst
  mov r11, rdx                // r11 = src
  mov eax, r8d                // eax = N (32-bit)

  test eax, eax
  jle @Exit

  // Broadcast fact (in XMM3) to all lanes of YMM0
  vbroadcastss ymm0, xmm3     // ymm0 = fact

  // Compute bulk (multiple of 16) and tail (0..15)
  mov ecx, eax
  and ecx, 15                 // ecx = tail (N mod 16)
  mov edx, ecx                // save tail in edx
  sub eax, ecx                // eax = bulk count
  shr eax, 4                  // eax = number of 16-element chunks
  jz @Tail

  // ---- Main loop: 16 elements per iteration (2 YMM blocks) ----
@BulkLoop:
  // Block 0: elements 0..7
  vmovups ymm1, [r11]         // src[0..7]
  vmovups ymm2, [r10]         // dst[0..7]
  vfmadd231ps ymm2, ymm1, ymm0   // dst = dst + src * fact
  vmovups [r10], ymm2

  // Block 1: elements 8..15
  vmovups ymm3, [r11+32]      // src[8..15]
  vmovups ymm4, [r10+32]      // dst[8..15]
  vfmadd231ps ymm4, ymm3, ymm0
  vmovups [r10+32], ymm4

  add r10, 64                 // advance by 16*4 = 64 bytes
  add r11, 64
  dec eax
  jnz @BulkLoop

  vzeroupper

@Tail:
  mov eax, edx                // restore tail count
  test eax, eax
  jz @Exit

  // Process 8-element chunk if tail >= 8
  cmp eax, 8
  jl @ScalarTail

  vmovups ymm1, [r11]         // src[0..7]
  vmovups ymm2, [r10]         // dst[0..7]
  vfmadd231ps ymm2, ymm1, ymm0
  vmovups [r10], ymm2
  add r10, 32
  add r11, 32
  sub eax, 8
  vzeroupper

@ScalarTail:
  // Last 0..7 elements handled one by one
  test eax, eax
  jz @Exit
  xor ecx, ecx
@ScalarLoop:
  vmovss xmm1, [r11 + rcx*4]  // src[i]
  vmovss xmm2, [r10 + rcx*4]  // dst[i]
  vfmadd231ss xmm2, xmm1, xmm0  // scalar FMA (uses low 32-bit of ymm0)
  vmovss [r10 + rcx*4], xmm2
  inc ecx
  cmp ecx, eax
  jl @ScalarLoop

@Exit:
  vzeroupper
end;

procedure _AVX512MulAddF( dst : PSingle; src : PSingle; N : Integer; const fact : Single); inline;
begin
  _AVX2MulAddF(dst, src, N, fact);
end;

{-----------------------------------------------------------------------------
  AVX2 memory fill: dst[i] = fact for i = 0..N-1.
  Optimized for Win64: processes 32 elements per iteration using 4 YMM stores.
  Parameters:
    RCX = dst (PSingle)
    EDX = N   (Integer)
    XMM2 = fact (Single)  // observed from call site
  Uses only volatile registers; no non-volatile saving.
-----------------------------------------------------------------------------}
procedure _AVX2FillMem(dst: PSingle; N: Integer; const fact: Single);
asm
  // Broadcast fact from XMM2 to all lanes of YMM0
  vbroadcastss ymm0, xmm2

  mov rax, rcx                // rax = dst
  mov ecx, edx                // ecx = N

  // ---- Main loop: 32 elements per iteration ----
  shr ecx, 5                  // number of 32-element blocks
  jz @Tail32
@Loop32:
  vmovups [rax], ymm0
  vmovups [rax+32], ymm0
  vmovups [rax+64], ymm0
  vmovups [rax+96], ymm0
  add rax, 128
  dec ecx
  jnz @Loop32

@Tail32:
  // ---- Process remaining 0..31 elements ----
  mov ecx, edx
  and ecx, 31                 // remaining count
  jz @Done

  // Process 4-element groups using XMM
  mov edx, ecx
  shr edx, 2                  // number of 4-element groups
  jz @Scalar
@Loop4:
  vmovups [rax], xmm0
  add rax, 16
  dec edx
  jnz @Loop4

@Scalar:
  // Process last 0..3 elements
  mov edx, ecx
  and edx, 3
  jz @Done
@Loop1:
  vmovss [rax], xmm0
  add rax, 4
  dec edx
  jnz @Loop1

@Done:
  vzeroupper
end;

procedure _AVX512FillMem( dst : PSingle; N : Integer; const fact : Single); inline;
begin
  _AVX2FillMem(dst, N, fact);
end;

{-----------------------------------------------------------------------------
  AVX2 scalar multiply-multiply-add:
    dst[i] = dst[i] * mulOp1 + src[i] * mulOp2.

  Optimized for Win64: processes 16 elements per iteration using 2 YMM blocks.
  Uses FMA for accuracy and performance.

  Win64 calling convention (observed offsets):
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : Integer
    XMM3 = mulOp1 : Single
    [RBP+48] = mulOp2 : Single

  Uses vmovdqu for unaligned memory access.
  R12 is non-volatile and saved/restored.
  vzeroupper is called before exit.
-----------------------------------------------------------------------------}
procedure _AVX2MulMulAdd(dst: PSingle; src: PSingle; N: Integer;
  const mulOp1, mulOp2: Single);
asm
  push r12                     // save non-volatile R12
  mov r10, rcx                 // r10 = dst
  mov r12, rdx                 // r12 = src
  mov eax, r8d                 // eax = N

  test eax, eax
  jle @Exit

  // Broadcast constants
  vbroadcastss ymm0, xmm3      // ymm0 = mulOp1
  vbroadcastss ymm1, [rbp+48]  // ymm1 = mulOp2

  // Split bulk (multiple of 16) and tail (0..15)
  mov ecx, eax
  and ecx, 15
  mov edx, ecx                 // tail count
  sub eax, ecx
  shr eax, 4                   // number of 16-element chunks
  jz @Tail

@BulkLoop:
  // Block 0: elements 0..7
  vmovdqu ymm2, [r10]          // load dst
  vmovdqu ymm3, [r12]          // load src
  vmulps  ymm2, ymm2, ymm0     // dst * mulOp1
  vfmadd231ps ymm2, ymm3, ymm1 // + src * mulOp2
  vmovdqu [r10], ymm2          // store result

  // Block 1: elements 8..15
  vmovdqu ymm4, [r10+32]
  vmovdqu ymm5, [r12+32]
  vmulps  ymm4, ymm4, ymm0
  vfmadd231ps ymm4, ymm5, ymm1
  vmovdqu [r10+32], ymm4

  add r10, 64                  // advance by 16 elements
  add r12, 64
  dec eax
  jnz @BulkLoop

  vzeroupper

@Tail:
  mov eax, edx                 // tail count
  test eax, eax
  jz @Exit

  // Process 8-element chunk if tail >= 8
  cmp eax, 8
  jl @ScalarTail

  vmovdqu ymm2, [r10]
  vmovdqu ymm3, [r12]
  vmulps  ymm2, ymm2, ymm0
  vfmadd231ps ymm2, ymm3, ymm1
  vmovdqu [r10], ymm2
  add r10, 32
  add r12, 32
  sub eax, 8

@ScalarTail:
  test eax, eax
  jz @Exit

  // Scalar tail for remaining 0..7 elements
  xor ecx, ecx
@ScalarLoop:
  vmovss xmm2, [r10 + rcx*4]
  vmovss xmm3, [r12 + rcx*4]
  vmulss xmm2, xmm2, xmm0
  vfmadd231ss xmm2, xmm3, xmm1
  vmovss [r10 + rcx*4], xmm2
  inc ecx
  cmp ecx, eax
  jl @ScalarLoop

@Exit:
  vzeroupper
  pop r12
end;

procedure _AVX512MulMulAdd( dst : PSingle; src : PSingle; N : Integer; const mulOp1, mulOp2 : Single); inline;
begin
  _AVX2MulMulAdd(dst, src, N, mulOp1, mulOp2);
end;

{-----------------------------------------------------------------------------
  AVX2 triadic multiply-add: dst[i] = dst[i] + src[i] * z[i].
  Optimized for Win64: processes 32 elements per iteration using 4 YMM blocks.

  Parameters (Win64 calling convention):
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = z   : PSingle
    R9  = N   : Integer

  All registers used are volatile (RCX, RDX, R8, R9, RAX, YMM0-YMM5),
  no prologue/epilogue needed.
  vzeroupper is called before exit to avoid AVX-SSE transition penalties.
-----------------------------------------------------------------------------}
procedure _AVX2MulAdd(dst: PSingle; src: PSingle; z: PSingle; N: Integer);
asm
  // Load parameters into volatile registers (they are already in RCX, RDX, R8, R9)
  // but we can use them directly; we'll move to R10/R11/R12 for clarity and to
  // preserve original values if needed (not necessary but okay).
  mov r10, rcx                // r10 = dst
  mov r11, rdx                // r11 = src
  mov r12, r8                 // r12 = z
  mov eax, r9d                // eax = N (32-bit)

  test eax, eax
  jle @Exit

  // Compute bulk (multiple of 32) and tail (0..31)
  mov ecx, eax
  and ecx, 31                 // ecx = tail (N mod 32)
  mov edx, ecx                // save tail in edx
  sub eax, ecx                // eax = bulk count
  shr eax, 5                  // eax = number of 32-element chunks
  jz @Tail

  // ---- Main loop: 32 elements per iteration (4 YMM blocks) ----
@BulkLoop:
  // Block 0: elements 0..7
  vmovups ymm0, [r11]
  vmulps  ymm0, ymm0, [r12]
  vaddps  ymm0, ymm0, [r10]
  vmovups [r10], ymm0

  // Block 1: elements 8..15
  vmovups ymm1, [r11+32]
  vmulps  ymm1, ymm1, [r12+32]
  vaddps  ymm1, ymm1, [r10+32]
  vmovups [r10+32], ymm1

  // Block 2: elements 16..23
  vmovups ymm2, [r11+64]
  vmulps  ymm2, ymm2, [r12+64]
  vaddps  ymm2, ymm2, [r10+64]
  vmovups [r10+64], ymm2

  // Block 3: elements 24..31
  vmovups ymm3, [r11+96]
  vmulps  ymm3, ymm3, [r12+96]
  vaddps  ymm3, ymm3, [r10+96]
  vmovups [r10+96], ymm3

  add r10, 128               // advance by 32*4 = 128 bytes
  add r11, 128
  add r12, 128
  dec eax
  jnz @BulkLoop

  vzeroupper

@Tail:
  mov eax, edx               // restore tail count
  test eax, eax
  jz @Exit

  // Process remaining elements in 8-element chunks
  mov ecx, eax
  and ecx, 7                 // leftover < 8
  sub eax, ecx               // multiple of 8
  shr eax, 3                 // number of 8-element chunks
  jz @ScalarTail

@SmallLoop:
  vmovups ymm0, [r11]
  vmulps  ymm0, ymm0, [r12]
  vaddps  ymm0, ymm0, [r10]
  vmovups [r10], ymm0
  add r10, 32
  add r11, 32
  add r12, 32
  dec eax
  jnz @SmallLoop

  vzeroupper

@ScalarTail:
  // Last 0..7 elements processed one by one
  mov eax, ecx               // remaining count
  test eax, eax
  jz @Exit
  xor ecx, ecx
@ScalarLoop:
  vmovss xmm0, [r11 + rcx*4]
  vmulss xmm0, xmm0, [r12 + rcx*4]
  vaddss xmm0, xmm0, [r10 + rcx*4]
  vmovss [r10 + rcx*4], xmm0
  inc ecx
  cmp ecx, eax
  jl @ScalarLoop

@Exit:
  vzeroupper
end;

procedure _AVX512MulAdd( dst : PSingle; src : PSingle; z : PSingle; N : Integer); inline;
begin
  _AVX2MulAdd(dst, src, z, N);
end;

{-----------------------------------------------------------------------------
  AVX2 ReLU copy: dst[i] = max(0, src[i]), with NaN -> 0.
  Optimized for Win64: processes 32 elements per iteration (4 YMM blocks),
  then XMM (4 elements) and scalar tail.
  Reverse traversal for compatibility with legacy code.
  Parameters:
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : Integer
  Uses only volatile YMM registers (YMM0-YMM5).
-----------------------------------------------------------------------------}
procedure _AVX2CopyRelu(dst: PSingle; src: PSingle; N: Integer);
asm
  // Prepare reverse traversal
  mov rax, rcx
  mov rdx, rdx
  mov rcx, r8
  imul rcx, -4
  sub rax, rcx
  sub rdx, rcx

  vxorps ymm4, ymm4, ymm4      // zero

@Loop1:
  add rcx, 128
  jg @Loop1End

  // Block 0
  vmovups ymm0, [rdx + rcx - 128]
  vcmpps ymm5, ymm0, ymm0, 7   // NaN check (ORD_Q)
  vandps ymm0, ymm0, ymm5      // NaN -> 0
  vmaxps ymm0, ymm4, ymm0

  // Block 1
  vmovups ymm1, [rdx + rcx - 96]
  vcmpps ymm5, ymm1, ymm1, 7
  vandps ymm1, ymm1, ymm5
  vmaxps ymm1, ymm4, ymm1

  // Block 2
  vmovups ymm2, [rdx + rcx - 64]
  vcmpps ymm5, ymm2, ymm2, 7
  vandps ymm2, ymm2, ymm5
  vmaxps ymm2, ymm4, ymm2

  // Block 3
  vmovups ymm3, [rdx + rcx - 32]
  vcmpps ymm5, ymm3, ymm3, 7
  vandps ymm3, ymm3, ymm5
  vmaxps ymm3, ymm4, ymm3

  vmovups [rax + rcx - 128], ymm0
  vmovups [rax + rcx - 96],  ymm1
  vmovups [rax + rcx - 64],  ymm2
  vmovups [rax + rcx - 32],  ymm3

  jmp @Loop1

@Loop1End:
  sub rcx, 128
  jz @Done

@Loop2:
  add rcx, 16
  jg @Loop2End

  vmovups xmm0, [rdx + rcx - 16]
  vcmpps xmm5, xmm0, xmm0, 7
  vandps xmm0, xmm0, xmm5
  vmaxps xmm0, xmm4, xmm0
  vmovups [rax + rcx - 16], xmm0
  jmp @Loop2

@Loop2End:
  sub rcx, 16
  jz @Done

@Loop3:
  add rcx, 4
  jg @Done

  vmovss xmm0, [rdx + rcx - 4]
  vcmpss xmm5, xmm0, xmm0, 7
  vandps xmm0, xmm0, xmm5
  vmaxss xmm0, xmm4, xmm0
  vmovss [rax + rcx - 4], xmm0
  jmp @Loop3

@Done:
  vzeroupper
end;

procedure _AVX512CopyRelu( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2CopyRelu(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 scalar multiplication: dst[i] = dst[i] * factor.
  Optimized for Win64: processes 32 elements per iteration using 4 YMM stores.
  Reverse traversal (from end to start) for compatibility.
  Parameters (Win64 calling convention):
    RCX = dst : PSingle
    RDX = N   : Integer
    XMM2 = factor : Single
  Uses YMM registers; all used registers are volatile (YMM0-YMM5, RAX, RCX, RDX).
  No prologue/epilogue needed.
  vzeroupper is called before exit.
-----------------------------------------------------------------------------}
procedure _AVX2MulF(dst: PSingle; N: Integer; const factor: Single);
asm
  // Load factor and broadcast to all lanes of YMM0
  vbroadcastss ymm0, xmm2     // ymm0 = factor

  // Prepare reverse traversal: RDX = -N*4 (byte offset), RAX = end pointer
  imul rdx, -4                // rdx = -N * 4
  mov rax, rcx                // rax = dst
  sub rax, rdx                // rax = dst + N*4 (end of array)

  // ---- Main loop: 32 elements (4 YMM blocks) per iteration ----
@Loop1:
  add rdx, 128                // move forward by 32 elements (128 bytes)
  jg @Loop1End                // if rdx > 0, passed the start

  // Load, multiply, and store 4 blocks of 8 elements each
  vmulps ymm1, ymm0, [rax + rdx - 128]
  vmulps ymm2, ymm0, [rax + rdx - 96]
  vmulps ymm3, ymm0, [rax + rdx - 64]
  vmulps ymm4, ymm0, [rax + rdx - 32]

  vmovups [rax + rdx - 128], ymm1
  vmovups [rax + rdx - 96],  ymm2
  vmovups [rax + rdx - 64],  ymm3
  vmovups [rax + rdx - 32],  ymm4

  jmp @Loop1

@Loop1End:
  sub rdx, 128                // restore rdx to remaining offset
  jz @Done

  // ---- Process 4-element groups using XMM ----
@Loop2:
  add rdx, 16                 // move forward by 4 elements (16 bytes)
  jg @Loop2End

  vmovups xmm1, [rax + rdx - 16]
  vmulps xmm1, xmm1, xmm0
  vmovups [rax + rdx - 16], xmm1
  jmp @Loop2

@Loop2End:
  sub rdx, 16
  jz @Done

  // ---- Process last 0..3 elements using scalar ----
@Loop3:
  add rdx, 4                  // move forward by 1 element (4 bytes)
  jg @Done

  vmovss xmm1, [rax + rdx - 4]
  vmulss xmm1, xmm1, xmm0
  vmovss [rax + rdx - 4], xmm1
  jmp @Loop3

@Done:
  vzeroupper
end;

procedure _AVX512MulF( dst : PSingle; N : Integer; const factor : Single); inline;
begin
  _AVX2MulF(dst, N, factor);
end;

{-----------------------------------------------------------------------------
  AVX2 elementwise multiplication: dst[i] = dst[i] * src[i].
  Optimized for Win64: uses 256-bit YMM registers, processes 32 elements per
  iteration (4 YMM blocks), then XMM (4 elements) and scalar tail.
  Reverse traversal for compatibility.

  Win64 calling convention:
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : Integer

  All registers used are volatile (RAX, RCX, RDX, R8, R9, YMM0-YMM3).
  vzeroupper before exit.
-----------------------------------------------------------------------------}
procedure _AVX2Mul(dst: PSingle; src: PSingle; N: Integer);
asm
  // ---- Prepare reverse traversal ----
  mov rax, rcx                // rax = dst
  mov rdx, rdx                // rdx = src
  mov rcx, r8                 // rcx = N (64-bit)
  imul rcx, -4                // rcx = -N*4 (byte offset)
  sub rax, rcx                // rax = dst + N*4 (end pointer)
  sub rdx, rcx                // rdx = src + N*4 (end pointer)
  // Now rcx is the negative offset counter, rax/rdx point to end of arrays

  // ---- Main loop: 32 elements (4 YMM blocks) per iteration ----
@Loop1:
  add rcx, 128                // move forward by 32 elements (128 bytes)
  jg @Loop1End                // if rcx > 0, we've passed the start

  // Load src blocks
  vmovups ymm0, [rdx + rcx - 128]
  vmovups ymm1, [rdx + rcx - 96]
  vmovups ymm2, [rdx + rcx - 64]
  vmovups ymm3, [rdx + rcx - 32]

  // Multiply with dst (load, multiply, store)
  vmulps ymm0, ymm0, [rax + rcx - 128]
  vmulps ymm1, ymm1, [rax + rcx - 96]
  vmulps ymm2, ymm2, [rax + rcx - 64]
  vmulps ymm3, ymm3, [rax + rcx - 32]

  vmovups [rax + rcx - 128], ymm0
  vmovups [rax + rcx - 96],  ymm1
  vmovups [rax + rcx - 64],  ymm2
  vmovups [rax + rcx - 32],  ymm3

  jmp @Loop1

@Loop1End:
  sub rcx, 128                // restore rcx to remaining offset
  jz @Done

  // ---- Tail: 4-element groups using XMM ----
@Loop2:
  add rcx, 16                 // move forward by 4 elements (16 bytes)
  jg @Loop2End

  vmovups xmm0, [rdx + rcx - 16]
  vmulps xmm0, xmm0, [rax + rcx - 16]
  vmovups [rax + rcx - 16], xmm0
  jmp @Loop2

@Loop2End:
  sub rcx, 16
  jz @Done

  // ---- Final scalar tail (0..3 elements) ----
@Loop3:
  add rcx, 4                  // move forward by 1 element (4 bytes)
  jg @Done

  vmovss xmm0, [rdx + rcx - 4]
  vmulss xmm0, xmm0, [rax + rcx - 4]
  vmovss [rax + rcx - 4], xmm0
  jmp @Loop3

@Done:
  vzeroupper
end;

procedure _AVX512Mul( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2Mul(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 elementwise addition: dst[i] = dst[i] + src[i].
  Optimized for Win64: uses 256-bit YMM registers, processes 32 elements per
  iteration (4 YMM blocks), then XMM (4 elements) and scalar tail.
  Reverse traversal for compatibility.

  Win64 calling convention:
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : Integer

  All registers used are volatile (RAX, RCX, RDX, R8, R9, YMM0-YMM3).
  vzeroupper before exit.
-----------------------------------------------------------------------------}
procedure _AVX2Add(dst: PSingle; src: PSingle; N: Integer);
asm
  // ---- Prepare reverse traversal ----
  mov rax, rcx                // rax = dst
  mov rdx, rdx                // rdx = src
  mov rcx, r8                 // rcx = N (64-bit)
  imul rcx, -4                // rcx = -N*4 (byte offset)
  sub rax, rcx                // rax = dst + N*4 (end pointer)
  sub rdx, rcx                // rdx = src + N*4 (end pointer)
  // Now rcx is the negative offset counter, rax/rdx point to end of arrays

  // ---- Main loop: 32 elements (4 YMM blocks) per iteration ----
@Loop1:
  add rcx, 128                // move forward by 32 elements (128 bytes)
  jg @Loop1End                // if rcx > 0, we've passed the start

  // Load src blocks
  vmovups ymm0, [rdx + rcx - 128]
  vmovups ymm1, [rdx + rcx - 96]
  vmovups ymm2, [rdx + rcx - 64]
  vmovups ymm3, [rdx + rcx - 32]

  // Add with dst (load, add, store)
  vaddps ymm0, ymm0, [rax + rcx - 128]
  vaddps ymm1, ymm1, [rax + rcx - 96]
  vaddps ymm2, ymm2, [rax + rcx - 64]
  vaddps ymm3, ymm3, [rax + rcx - 32]

  vmovups [rax + rcx - 128], ymm0
  vmovups [rax + rcx - 96],  ymm1
  vmovups [rax + rcx - 64],  ymm2
  vmovups [rax + rcx - 32],  ymm3

  jmp @Loop1

@Loop1End:
  sub rcx, 128                // restore rcx to remaining offset
  jz @Done

  // ---- Tail: 4-element groups using XMM ----
@Loop2:
  add rcx, 16                 // move forward by 4 elements (16 bytes)
  jg @Loop2End

  vmovups xmm0, [rdx + rcx - 16]
  vaddps xmm0, xmm0, [rax + rcx - 16]
  vmovups [rax + rcx - 16], xmm0
  jmp @Loop2

@Loop2End:
  sub rcx, 16
  jz @Done

  // ---- Final scalar tail (0..3 elements) ----
@Loop3:
  add rcx, 4                  // move forward by 1 element (4 bytes)
  jg @Done

  vmovss xmm0, [rdx + rcx - 4]
  vaddss xmm0, xmm0, [rax + rcx - 4]
  vmovss [rax + rcx - 4], xmm0
  jmp @Loop3

@Done:
  vzeroupper
end;

procedure _AVX512Add( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2Add(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 elementwise max: dst[i] = max(dst[i], src[i]).
  Optimized for Win64: uses 256-bit YMM registers, processes 32 elements per
  iteration (4 YMM blocks), then XMM (4 elements) and scalar tail.
  Reverse traversal for compatibility.

  Win64 calling convention:
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : Integer

  All registers used are volatile (RAX, RCX, RDX, R8, R9, YMM0-YMM5).
  vzeroupper before exit.
-----------------------------------------------------------------------------}
procedure _AVX2Max(dst: PSingle; src: PSingle; N: Integer);
asm
  // ---- Prepare reverse traversal ----
  mov rax, rcx                // rax = dst
  mov rdx, rdx                // rdx = src
  mov rcx, r8                 // rcx = N (64-bit)
  imul rcx, -4                // rcx = -N*4 (byte offset)
  sub rax, rcx                // rax = dst + N*4 (end pointer)
  sub rdx, rcx                // rdx = src + N*4 (end pointer)
  // Now rcx is the negative offset counter, rax/rdx point to end of arrays

  // ---- Main loop: 32 elements (4 YMM blocks) per iteration ----
@Loop1:
  add rcx, 128                // move forward by 32 elements (128 bytes)
  jg @Loop1End                // if rcx > 0, we've passed the start

  // Load dst blocks
  vmovups ymm0, [rax + rcx - 128]
  vmovups ymm1, [rax + rcx - 96]
  vmovups ymm2, [rax + rcx - 64]
  vmovups ymm3, [rax + rcx - 32]

  // max with src
  vmaxps ymm0, ymm0, [rdx + rcx - 128]
  vmaxps ymm1, ymm1, [rdx + rcx - 96]
  vmaxps ymm2, ymm2, [rdx + rcx - 64]
  vmaxps ymm3, ymm3, [rdx + rcx - 32]

  // Store back
  vmovups [rax + rcx - 128], ymm0
  vmovups [rax + rcx - 96],  ymm1
  vmovups [rax + rcx - 64],  ymm2
  vmovups [rax + rcx - 32],  ymm3

  jmp @Loop1

@Loop1End:
  sub rcx, 128                // restore rcx to remaining offset
  jz @Done

  // ---- Tail: 4-element groups using XMM ----
@Loop2:
  add rcx, 16                 // move forward by 4 elements (16 bytes)
  jg @Loop2End

  vmovups xmm0, [rax + rcx - 16]
  vmaxps xmm0, xmm0, [rdx + rcx - 16]
  vmovups [rax + rcx - 16], xmm0
  jmp @Loop2

@Loop2End:
  sub rcx, 16
  jz @Done

  // ---- Final scalar tail (0..3 elements) ----
@Loop3:
  add rcx, 4                  // move forward by 1 element (4 bytes)
  jg @Done

  vmovss xmm0, [rax + rcx - 4]
  vmaxss xmm0, xmm0, [rdx + rcx - 4]
  vmovss [rax + rcx - 4], xmm0
  jmp @Loop3

@Done:
  vzeroupper
end;

procedure _AVX512Max( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2Max(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Sum of Absolute Differences: result = sum_i |dst[i] - src[i]|.
  Optimized for Win64: processes 8 elements per iteration using YMM.
  Tail (0..7) accumulated in XMM4, added after horizontal sum.
  Parameters (Win64 calling convention):
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : Integer
  Returns Single result in XMM0 (no FPU stack used).
  All registers used are volatile (YMM0-YMM4, RAX, RDX, RCX, R8).
  vzeroupper is called before exit.
-----------------------------------------------------------------------------}
function _AVX2SumDiff(dst: PSingle; src: PSingle; N: Integer): Single;
asm
  // Load parameters into volatile registers
  mov rax, rcx                // rax = dst
  mov rdx, rdx                // rdx = src (already)
  mov ecx, r8d                // ecx = N (32-bit)

  vxorps ymm0, ymm0, ymm0     // ymm0 = main accumulator (8 lanes)

  // Generate absolute mask: 0x7FFFFFFF in each lane (clear sign bit)
  vpcmpeqd ymm1, ymm1, ymm1   // ymm1 = all ones
  vpsrld   ymm1, ymm1, 1      // ymm1 = 0x7FFFFFFF (abs mask)

  // ---- Bulk: process 8 elements per iteration ----
@Loop8:
  cmp ecx, 8
  jl @Tail

  vmovups ymm2, [rax]         // dst[i..i+7]
  vmovups ymm3, [rdx]         // src[i..i+7]
  vsubps  ymm2, ymm2, ymm3    // difference
  vandps  ymm2, ymm2, ymm1    // absolute value
  vaddps  ymm0, ymm0, ymm2    // accumulate

  add rax, 32
  add rdx, 32
  sub ecx, 8
  jmp @Loop8

@Tail:
  test ecx, ecx
  jz @Done

  // ---- Tail: accumulate in XMM4 (scalar) ----
  vxorps xmm4, xmm4, xmm4
  xor r8, r8                  // r8 = index
@ScalarLoop:
  vmovss xmm2, [rax + r8*4]
  vmovss xmm3, [rdx + r8*4]
  vsubss xmm2, xmm2, xmm3
  vandps xmm2, xmm2, xmm1     // absolute (xmm1 low part is mask)
  vaddss xmm4, xmm4, xmm2
  inc r8
  cmp r8, rcx
  jl @ScalarLoop

@Done:
  // ---- Horizontal sum of ymm0 (8 lanes) into xmm0 ----
  vextractf128 xmm2, ymm0, 1  // high 4 lanes
  vaddps xmm0, xmm0, xmm2     // sum of low and high
  vhaddps xmm0, xmm0, xmm0    // pairwise horizontal add (4->2)
  vhaddps xmm0, xmm0, xmm0    // 2->1 (now xmm0[0] = sum of all 8)

  // ---- Add tail sum (in xmm4[0]) ----
  vaddss xmm0, xmm0, xmm4

  // ---- Return result in XMM0 (Single) ----
  vzeroupper
end;

function _AVX512SumDiff( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2SumDiff(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Squared Euclidean Distance: result = sum_i (dst[i] - src[i])^2.
  Optimized for Win64: processes 32 elements per iteration (4 YMM blocks).
  Tail (0..31) handled by XMM (4 elements) and scalar (0..3).
  Parameters (Win64 calling convention):
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : Integer
  Returns Single result in XMM0 (no FPU stack used).
  All registers used are volatile (YMM0-YMM4, RAX, RDX, RCX, R8).
  vzeroupper is called before exit.
-----------------------------------------------------------------------------}
function _AVX2DistanceSqr(dst: PSingle; src: PSingle; N: Integer): Single;
asm
  // Load parameters into volatile registers
  mov rax, rcx                // rax = dst
  mov rdx, rdx                // rdx = src (already)
  mov ecx, r8d                // ecx = N (32-bit)

  vxorps ymm0, ymm0, ymm0     // ymm0 = main accumulator (8 lanes)

  // ---- Main loop: process 32 elements per iteration (4 YMM blocks) ----
@Loop32:
  cmp ecx, 32
  jl @Loop4

  // Block 0: elements 0..7
  vmovups ymm2, [rax]
  vmovups ymm3, [rdx]
  vsubps  ymm2, ymm2, ymm3
  vmulps  ymm2, ymm2, ymm2
  vaddps  ymm0, ymm0, ymm2

  // Block 1: elements 8..15
  vmovups ymm2, [rax+32]
  vmovups ymm3, [rdx+32]
  vsubps  ymm2, ymm2, ymm3
  vmulps  ymm2, ymm2, ymm2
  vaddps  ymm0, ymm0, ymm2

  // Block 2: elements 16..23
  vmovups ymm2, [rax+64]
  vmovups ymm3, [rdx+64]
  vsubps  ymm2, ymm2, ymm3
  vmulps  ymm2, ymm2, ymm2
  vaddps  ymm0, ymm0, ymm2

  // Block 3: elements 24..31
  vmovups ymm2, [rax+96]
  vmovups ymm3, [rdx+96]
  vsubps  ymm2, ymm2, ymm3
  vmulps  ymm2, ymm2, ymm2
  vaddps  ymm0, ymm0, ymm2

  add rax, 128                // advance by 32*4 = 128 bytes
  add rdx, 128
  sub ecx, 32
  jmp @Loop32

@Loop4:
  cmp ecx, 4
  jl @Tail

  // Process 4 elements using XMM
  vmovups xmm2, [rax]
  vmovups xmm3, [rdx]
  vsubps  xmm2, xmm2, xmm3
  vmulps  xmm2, xmm2, xmm2
  vaddps  xmm0, xmm0, xmm2   // xmm0 is part of ymm0; accumulating here is fine,
                               // but ymm0 high part remains unchanged, we'll handle later.

  add rax, 16
  add rdx, 16
  sub ecx, 4
  jmp @Loop4

@Tail:
  test ecx, ecx
  jz @Done

  // ---- Scalar tail (0..3 elements) accumulated in XMM4 ----
  vxorps xmm4, xmm4, xmm4
  xor r8, r8                  // r8 = index
@ScalarLoop:
  vmovss xmm2, [rax + r8*4]
  vmovss xmm3, [rdx + r8*4]
  vsubss xmm2, xmm2, xmm3
  vmulss xmm2, xmm2, xmm2
  vaddss xmm4, xmm4, xmm2
  inc r8
  cmp r8, rcx
  jl @ScalarLoop

@Done:
  // ---- Horizontal sum of ymm0 (8 lanes) into xmm0 ----
  vextractf128 xmm2, ymm0, 1  // high 4 lanes
  vaddps xmm0, xmm0, xmm2     // sum low+high
  vhaddps xmm0, xmm0, xmm0    // pairwise horizontal add (4->2)
  vhaddps xmm0, xmm0, xmm0    // 2->1 (now xmm0[0] = sum of all 8 lanes)

  // ---- Add tail sum (in xmm4[0]) ----
  vaddss xmm0, xmm0, xmm4

  // ---- Return result in XMM0 ----
  vzeroupper
end;

function _AVX512DistanceSqr( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2DistanceSqr(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 elementwise subtraction: dst[i] = dst[i] - src[i].
  Optimized for Win64: uses 256-bit YMM registers, processes 32 elements per
  iteration (4 YMM blocks), then XMM (4 elements) and scalar tail.
  Reverse traversal for compatibility.

  Win64 calling convention:
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : Integer

  All registers used are volatile (RAX, RCX, RDX, R8, R9, YMM0-YMM1).
  vzeroupper before exit.
-----------------------------------------------------------------------------}
procedure _AVX2Sub(dst: PSingle; src: PSingle; N: Integer);
asm
  // ---- Prepare reverse traversal ----
  mov rax, rcx                // rax = dst
  mov rdx, rdx                // rdx = src
  mov rcx, r8                 // rcx = N (64-bit)
  imul rcx, -4                // rcx = -N*4 (byte offset)
  sub rax, rcx                // rax = dst + N*4 (end pointer)
  sub rdx, rcx                // rdx = src + N*4 (end pointer)
  // Now rcx is the negative offset counter, rax/rdx point to end of arrays

  // ---- Main loop: 32 elements (4 YMM blocks) per iteration ----
@Loop1:
  add rcx, 128                // move forward by 32 elements (128 bytes)
  jg @Loop1End                // if rcx > 0, we've passed the start

  // Block 0: elements 0..7
  vmovups ymm0, [rax + rcx - 128]
  vmovups ymm1, [rdx + rcx - 128]
  vsubps  ymm0, ymm0, ymm1
  vmovups [rax + rcx - 128], ymm0

  // Block 1: elements 8..15
  vmovups ymm0, [rax + rcx - 96]
  vmovups ymm1, [rdx + rcx - 96]
  vsubps  ymm0, ymm0, ymm1
  vmovups [rax + rcx - 96], ymm0

  // Block 2: elements 16..23
  vmovups ymm0, [rax + rcx - 64]
  vmovups ymm1, [rdx + rcx - 64]
  vsubps  ymm0, ymm0, ymm1
  vmovups [rax + rcx - 64], ymm0

  // Block 3: elements 24..31
  vmovups ymm0, [rax + rcx - 32]
  vmovups ymm1, [rdx + rcx - 32]
  vsubps  ymm0, ymm0, ymm1
  vmovups [rax + rcx - 32], ymm0

  jmp @Loop1

@Loop1End:
  sub rcx, 128                // restore rcx to remaining offset
  jz @Done

  // ---- Tail: 4-element groups using XMM ----
@Loop2:
  add rcx, 16                 // move forward by 4 elements (16 bytes)
  jg @Loop2End

  vmovups xmm0, [rax + rcx - 16]
  vmovups xmm1, [rdx + rcx - 16]
  vsubps  xmm0, xmm0, xmm1
  vmovups [rax + rcx - 16], xmm0
  jmp @Loop2

@Loop2End:
  sub rcx, 16
  jz @Done

  // ---- Final scalar tail (0..3 elements) ----
@Loop3:
  add rcx, 4                  // move forward by 1 element (4 bytes)
  jg @Done

  vmovss xmm0, [rax + rcx - 4]
  vmovss xmm1, [rdx + rcx - 4]
  vsubss  xmm0, xmm0, xmm1
  vmovss [rax + rcx - 4], xmm0
  jmp @Loop3

@Done:
  vzeroupper
end;

procedure _AVX512Sub( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2Sub(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Sum of elements: result = sum_i src[i].
  Optimized for Win64: processes 32 elements per iteration (4 YMM loads).
  Tail (0..31) handled by XMM (4 elements) and scalar (0..3).
  Parameters (Win64 calling convention):
    RCX = src : PSingle
    RDX = N   : Integer
  Returns Single result in XMM0.
  All registers used are volatile (YMM0-YMM2, RAX, RCX, RDX).
  vzeroupper is called before exit.
-----------------------------------------------------------------------------}
function _AVX2GetSum(src: PSingle; N: Integer): Single;
asm
  // Load parameters
  mov rax, rcx                // rax = src
  mov ecx, edx                // ecx = N (32-bit)

  vxorps ymm0, ymm0, ymm0     // ymm0 = main accumulator (8 lanes)

  // ---- Main loop: process 32 elements per iteration (4 YMM loads) ----
@Loop32:
  cmp ecx, 32
  jl @Loop4

  vaddps ymm0, ymm0, [rax]    // elements 0..7
  vaddps ymm0, ymm0, [rax+32] // elements 8..15
  vaddps ymm0, ymm0, [rax+64] // elements 16..23
  vaddps ymm0, ymm0, [rax+96] // elements 24..31

  add rax, 128                // advance by 32*4 = 128 bytes
  sub ecx, 32
  jmp @Loop32

@Loop4:
  cmp ecx, 4
  jl @Tail

  // Process 4 elements using XMM
  vaddps xmm0, xmm0, [rax]    // accumulate into low 128 bits
  add rax, 16
  sub ecx, 4
  jmp @Loop4

@Tail:
  test ecx, ecx
  jz @Done

  // ---- Scalar tail (0..3 elements) accumulated in XMM1 ----
  vxorps xmm1, xmm1, xmm1
  xor r8, r8                  // r8 = index
@ScalarLoop:
  vaddss xmm1, xmm1, [rax + r8*4]
  inc r8
  cmp r8, rcx
  jl @ScalarLoop

@Done:
  // ---- Horizontal sum of ymm0 (8 lanes) into xmm0 ----
  vextractf128 xmm2, ymm0, 1  // high 4 lanes
  vaddps xmm0, xmm0, xmm2     // sum low+high
  vhaddps xmm0, xmm0, xmm0    // pairwise horizontal add (4->2)
  vhaddps xmm0, xmm0, xmm0    // 2->1 (now xmm0[0] = sum of all 8 lanes)

  // ---- Add tail sum (in xmm1[0]) ----
  vaddss xmm0, xmm0, xmm1

  // ---- Return result in XMM0 ----
  vzeroupper
end;

function _AVX512GetSum( src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2GetSum(src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 sum of squares: result = sum_i src[i]^2.
  Optimized for Win64: processes 8 elements per iteration using YMM.
  Tail (0..7) handled by scalar; XMM1 is zero if no tail.
  Parameters (Win64 calling convention):
    RCX = src : PSingle
    RDX = N   : Integer
  Returns Single result in XMM0.
  All registers used are volatile (YMM0-YMM2, RAX, RCX, RDX, R8).
  vzeroupper is called before exit.
-----------------------------------------------------------------------------}
function _AVX2GetSumSqr(src: PSingle; N: Integer): Single;
asm
  // Load parameters
  mov rax, rcx                // rax = src
  mov ecx, edx                // ecx = N (32-bit)

  vxorps ymm0, ymm0, ymm0     // ymm0 = main accumulator (8 lanes)

  // ---- Main loop: process 8 elements per iteration ----
@Loop8:
  cmp ecx, 8
  jl @Tail

  vmovups ymm1, [rax]         // load 8 elements
  vmulps  ymm1, ymm1, ymm1    // square
  vaddps  ymm0, ymm0, ymm1    // accumulate

  add rax, 32                 // advance by 8*4 = 32 bytes
  sub ecx, 8
  jmp @Loop8

@Tail:
  // Clear tail accumulator
  vxorps xmm1, xmm1, xmm1
  test ecx, ecx
  jz @Merge

  // ---- Scalar tail (0..7 elements) ----
  xor r8, r8                  // r8 = index
@ScalarLoop:
  vmovss xmm2, [rax + r8*4]
  vmulss xmm2, xmm2, xmm2
  vaddss xmm1, xmm1, xmm2
  inc r8
  cmp r8, rcx
  jl @ScalarLoop

@Merge:
  // ---- Horizontal sum of ymm0 (8 lanes) into xmm0 ----
  vextractf128 xmm2, ymm0, 1  // high 4 lanes
  vaddps xmm0, xmm0, xmm2     // sum low+high
  vhaddps xmm0, xmm0, xmm0    // pairwise horizontal add (4->2)
  vhaddps xmm0, xmm0, xmm0    // 2->1 (now xmm0[0] = sum of all 8 lanes)

  // ---- Add tail sum (xmm1[0]) ----
  vaddss xmm0, xmm0, xmm1

  // ---- Return result in XMM0 ----
  vzeroupper
end;

function _AVX512GetSumSqr( src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2GetSumSqr(src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 exponential: dst[i] = exp(src[i]).
  Optimized for Win64: uses 8-wide YMM polynomial for bulk (ymm0..ymm7),
  scalar polynomial for tail.
  Parameters (Win64 calling convention):
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : Integer
  All registers used are volatile (RAX, RCX, RDX, R8, R9, R10, R11, YMM0-YMM7,
  and optionally YMM8-YMM15 can be used for constants).
  vzeroupper is called before exit.
  Requires FMA instructions (vfmadd213ps/ss) for polynomial evaluation.
-----------------------------------------------------------------------------}
procedure _AVX2Exp(dst: PSingle; src: PSingle; N: Integer);
asm
  // Load parameters
  mov rax, rcx                // rax = dst
  mov rdx, rdx                // rdx = src
  mov ecx, r8d                // ecx = N (32-bit)

  test ecx, ecx
  jle @Exit

  // tail count = N mod 8
  mov r8d, ecx                // r8d = N
  and r8d, 7                  // tail = N & 7
  sub ecx, r8d                // ecx = bulk count (multiple of 8)
  jz @Tail

  // Bulk: process 8 at a time
  mov r9d, ecx
  shr r9d, 3                  // r9d = number of 8-element blocks

  // Load constants into registers for bulk loop
  vbroadcastss ymm6, [rip + cAVXLog2e]
  vbroadcastss ymm7, [rip + cAVXLn2]
  vpbroadcastd ymm5, [rip + cAVXExp127]

@BulkLoop:
  vmovups ymm0, [rdx]         // load 8 src

  // Clamp to [-88.376, 88.376]
  vbroadcastss ymm1, [rip + cAVXExpHi]
  vminps ymm0, ymm0, ymm1
  vbroadcastss ymm1, [rip + cAVXExpLo]
  vmaxps ymm0, ymm0, ymm1

  // t = x * log2e
  vmulps ymm1, ymm0, ymm6
  vroundps ymm2, ymm1, 0      // k = round(t)
  vsubps ymm1, ymm1, ymm2     // f = t - k

  // g = f * ln2
  vmulps ymm3, ymm1, ymm7

  // Horner polynomial for 2^f: ymm4 = P6
  vbroadcastss ymm4, [rip + cAVXExpP6]
  vbroadcastss ymm0, [rip + cAVXExpP5]   // reuse ymm0 as scratch
  vfmadd213ps ymm4, ymm3, ymm0
  vbroadcastss ymm0, [rip + cAVXExpP4]
  vfmadd213ps ymm4, ymm3, ymm0
  vbroadcastss ymm0, [rip + cAVXExpP3]
  vfmadd213ps ymm4, ymm3, ymm0
  vbroadcastss ymm0, [rip + cAVXExpP2]
  vfmadd213ps ymm4, ymm3, ymm0
  vbroadcastss ymm0, [rip + cAVXExpP1]
  vfmadd213ps ymm4, ymm3, ymm0
  vbroadcastss ymm0, [rip + cAVXExpP0]
  vfmadd213ps ymm4, ymm3, ymm0    // ymm4 = 2^f

  // 2^k
  vcvtps2dq ymm2, ymm2
  vpaddd ymm2, ymm2, ymm5         // add 127
  vpslld ymm2, ymm2, 23           // shift to exponent bits

  // result = 2^f * 2^k
  vmulps ymm0, ymm4, ymm2
  vmovups [rax], ymm0

  add rdx, 32
  add rax, 32
  dec r9d
  jnz @BulkLoop

  vzeroupper

@Tail:
  mov r10d, r8d               // r10d = tail count (0..7)
  test r10d, r10d
  jz @Exit

  // Load scalar constants once, outside the loop
  vbroadcastss xmm6, [rip + cAVXExpHi]
  vbroadcastss xmm7, [rip + cAVXExpLo]
  vpbroadcastd xmm5, [rip + cAVXExp127]

  xor r9d, r9d                // index
@TailLoop:
  vmovss xmm0, [rdx + r9*4]
  vminss xmm0, xmm0, xmm6
  vmaxss xmm0, xmm0, xmm7

  vmulss xmm1, xmm0, [rip + cAVXLog2e]
  vroundss xmm2, xmm1, xmm1, 0
  vsubss xmm1, xmm1, xmm2

  vmulss xmm3, xmm1, [rip + cAVXLn2]

  // Scalar polynomial: xmm4 = P6
  vbroadcastss xmm4, [rip + cAVXExpP6]
  vfmadd213ss xmm4, xmm3, [rip + cAVXExpP5]
  vfmadd213ss xmm4, xmm3, [rip + cAVXExpP4]
  vfmadd213ss xmm4, xmm3, [rip + cAVXExpP3]
  vfmadd213ss xmm4, xmm3, [rip + cAVXExpP2]
  vfmadd213ss xmm4, xmm3, [rip + cAVXExpP1]
  vfmadd213ss xmm4, xmm3, [rip + cAVXExpP0]

  // 2^k (scalar)
  vcvtss2si r11, xmm2         // use r11d as temporary
  add r11, 127
  shl r11, 23
  movd xmm2, r11d

  vmulss xmm0, xmm4, xmm2
  vmovss [rax + r9*4], xmm0

  inc r9d
  cmp r9d, r10d
  jl @TailLoop

@Exit:
  vzeroupper
end;

procedure _AVX512Exp( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2Exp(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 dot product: result = sum_i dst[i] * src[i].
  Optimized for Win64: processes 32 elements per iteration using 4 YMM blocks.
  Tail handling: 4-element XMM groups then scalar remainder.
  Parameters (Win64 calling convention):
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : Integer
  Returns Single result in XMM0.
  Uses FMA instructions (requires AVX2+FMA capable CPU).
  All registers used are volatile (YMM0-YMM7, RAX, RDX, RCX, R8, R9, R10).
  vzeroupper is called before exit.
-----------------------------------------------------------------------------}
function _AVX2DotProd(dst: PSingle; src: PSingle; N: Integer): Single;
asm
  // Load parameters into volatile registers
  mov rax, rcx                // rax = dst
  mov rdx, rdx                // rdx = src (already)
  mov ecx, r8d                // ecx = N (32-bit)

  test ecx, ecx
  jle @ZeroResult

  // bulk = N - (N mod 32)
  mov r8d, ecx                // r8d = N
  and r8d, 31                 // tail = N & 31
  sub ecx, r8d                // ecx = bulk (multiple of 32)
  jz @Tail

  // Bulk: process 32 elements at a time (4 YMM blocks)
  vxorps ymm0, ymm0, ymm0     // accumulator for blocks 0..7
  vxorps ymm1, ymm1, ymm1     // accumulator for blocks 8..15
  vxorps ymm2, ymm2, ymm2     // accumulator for blocks 16..23
  vxorps ymm3, ymm3, ymm3     // accumulator for blocks 24..31

  mov r9d, ecx
  shr r9d, 5                  // r9d = number of 32-element chunks
  jz @Tail

@BulkLoop:
  vmovups ymm4, [rax]
  vfmadd231ps ymm0, ymm4, [rdx]     // block 0
  vmovups ymm5, [rax+32]
  vfmadd231ps ymm1, ymm5, [rdx+32]  // block 1
  vmovups ymm6, [rax+64]
  vfmadd231ps ymm2, ymm6, [rdx+64]  // block 2
  vmovups ymm7, [rax+96]
  vfmadd231ps ymm3, ymm7, [rdx+96]  // block 3

  add rax, 128
  add rdx, 128
  dec r9d
  jnz @BulkLoop

  // Reduce YMM accumulators to XMM
  vaddps ymm0, ymm0, ymm1
  vaddps ymm2, ymm2, ymm3
  vaddps ymm0, ymm0, ymm2
  vextractf128 xmm4, ymm0, 1
  vaddps xmm0, xmm0, xmm4          // xmm0 holds partial sum

  vzeroupper

@Tail:
  // r8d = tail count (0..31)
  mov ecx, r8d
  and ecx, 3                   // scalar remainder (0..3)
  sub r8d, ecx                 // r8d = tail multiple of 4
  jz @ScalarTail

  // Process tail in groups of 4 with XMM
  mov r9d, r8d
  shr r9d, 2                   // number of 4-element blocks
@XmmLoop:
  vmovups xmm4, [rax]
  vmulps xmm4, xmm4, [rdx]
  vaddps xmm0, xmm0, xmm4
  add rax, 16
  add rdx, 16
  dec r9d
  jnz @XmmLoop

@ScalarTail:
  // ecx = remaining 0..3 elements
  test ecx, ecx
  jz @Finish
  xor r10d, r10d
@ScalarLoop:
  vmovss xmm4, [rax + r10*4]
  vmulss xmm4, xmm4, [rdx + r10*4]
  vaddss xmm0, xmm0, xmm4
  inc r10d
  cmp r10d, ecx
  jl @ScalarLoop

@Finish:
  // Horizontal sum xmm0 -> xmm0[0] = sum
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  jmp @Exit

@ZeroResult:
  vxorps xmm0, xmm0, xmm0

@Exit:
  vzeroupper
end;

function _AVX512DotProd( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2DotProd(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 int8 dot product: result = sum_i dst[i] * src[i].
  dst points to signed bytes (int8), src to Single floats.
  Processes 32 elements per loop (4 YMM blocks), then XMM groups and scalar tail.
  Win64 calling convention:
    RCX = dst : PShortInt
    RDX = src : PSingle
    R8  = N   : Integer
  Returns Single in XMM0.
  All registers used are volatile (YMM0-YMM7, RAX, RDX, RCX, R8, R9, R10, R11).
-----------------------------------------------------------------------------}
function _AVX2DotProdInt8(dst: PShortInt; src: PSingle; N: Integer): Single;
asm
  // Load parameters into volatile registers
  mov rax, rcx                // rax = dst
  mov rdx, rdx                // rdx = src (already)
  mov ecx, r8d                // ecx = N (32-bit)

  test ecx, ecx
  jle @Zero

  // Bulk = N - (N mod 32)
  mov r8d, ecx                // r8d = N
  and r8d, 31                 // tail = N & 31
  sub ecx, r8d                // ecx = bulk (multiple of 32)
  jz @Tail

  // Initialize accumulators
  vxorps ymm0, ymm0, ymm0     // blocks 0..7
  vxorps ymm1, ymm1, ymm1     // blocks 8..15
  vxorps ymm6, ymm6, ymm6     // blocks 16..23
  vxorps ymm7, ymm7, ymm7     // blocks 24..31

  mov r9d, ecx
  shr r9d, 5                  // number of 32-element chunks
  jz @Tail

@BulkLoop:
  // Load 8 int8 values and sign-extend to int32
  vpmovsxbd ymm2, [rax]       // bytes 0..7
  vpmovsxbd ymm3, [rax+8]     // bytes 8..15
  vpmovsxbd ymm4, [rax+16]    // bytes 16..23
  vpmovsxbd ymm5, [rax+24]    // bytes 24..31

  // Convert int32 to float
  vcvtdq2ps ymm2, ymm2
  vcvtdq2ps ymm3, ymm3
  vcvtdq2ps ymm4, ymm4
  vcvtdq2ps ymm5, ymm5

  // Multiply-add with corresponding float values (FMA)
  vfmadd231ps ymm0, ymm2, [rdx]      // block 0
  vfmadd231ps ymm1, ymm3, [rdx+32]   // block 1
  vfmadd231ps ymm6, ymm4, [rdx+64]   // block 2
  vfmadd231ps ymm7, ymm5, [rdx+96]   // block 3

  add rax, 32                 // advance by 32 bytes (32 int8)
  add rdx, 128                // advance by 128 bytes (32 floats)
  dec r9d
  jnz @BulkLoop

  // Reduce YMM accumulators to XMM
  vaddps ymm0, ymm0, ymm1
  vaddps ymm6, ymm6, ymm7
  vaddps ymm0, ymm0, ymm6
  vextractf128 xmm2, ymm0, 1
  vaddps xmm0, xmm0, xmm2
  vzeroupper

@Tail:
  // r8d = tail count (0..31)
  // Use r9d as working register for tail processing to preserve rax
  mov r9d, r8d                // r9d = tail
  and r9d, 3                  // scalar remainder (0..3)
  sub r8d, r9d                // r8d = multiple of 4 for XMM

  // Process tail in groups of 4 using XMM
  test r8d, r8d
  jz @ScalarTail

  mov r10d, r8d
  shr r10d, 2                 // number of 4-element groups
@XmmLoop:
  vpmovsxbd xmm2, [rax]       // load 4 int8
  vcvtdq2ps xmm2, xmm2
  vmovups xmm3, [rdx]
  vmulps xmm2, xmm2, xmm3
  vaddps xmm0, xmm0, xmm2
  add rax, 4
  add rdx, 16
  dec r10d
  jnz @XmmLoop

@ScalarTail:
  // r9d = remaining 0..3 elements
  test r9d, r9d
  jz @Finish

  xor r11d, r11d              // index
@ScalarLoop:
  movsx r10d, byte ptr [rax + r11]   // load int8 sign-extended (use r10d)
  vcvtsi2ss xmm2, xmm2, r10d
  vmovss xmm3, [rdx + r11*4]
  vmulss xmm2, xmm2, xmm3
  vaddss xmm0, xmm0, xmm2
  inc r11d
  cmp r11d, r9d
  jl @ScalarLoop

@Finish:
  // Horizontal sum xmm0 -> scalar
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  jmp @Done

@Zero:
  vxorps xmm0, xmm0, xmm0

@Done:
  vzeroupper
end;

function _AVX512DotProdInt8( dst : PShortInt; src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2DotProdInt8(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 int8 scalar multiply-add: dst[i] += W * codes[i].
  codes is int8 (signed byte), W is Single scalar.
  Processes 32 elements per loop (4 YMM blocks), then XMM groups and scalar tail.
  Win64 calling convention:
    RCX = dst : PSingle
    RDX = codes : PShortInt
    XMM2 = W : Single
    R9  = N : Integer
  All other registers used are volatile (RAX, RDX, RCX, R8, R9, R10, R11,
  YMM0-YMM5). YMM6-YMM7 are saved/restored.
-----------------------------------------------------------------------------}
procedure _AVX2MulAddInt8Scalar(dst: PSingle; codes: PShortInt; W: Single; N: Integer);
asm
  // Save non-volatile YMM6 and YMM7
  sub rsp, 64
  vmovups [rsp], ymm6
  vmovups [rsp+32], ymm7

  // Load parameters into volatile registers
  mov rax, rcx                // rax = dst
  mov rdx, rdx                // rdx = codes (already)
  mov ecx, r9d                // ecx = N (32-bit)

  test ecx, ecx
  jle @Exit

  // Broadcast W to all lanes of YMM5 (from XMM2)
  vbroadcastss ymm5, xmm2     // ymm5 = W

  // Bulk = N - (N mod 32)
  mov r8d, ecx                // r8d = N
  and r8d, 31                 // tail = N & 31
  sub ecx, r8d                // ecx = bulk (multiple of 32)
  jz @Tail

  mov r9d, ecx
  shr r9d, 5                  // r9d = number of 32-element chunks
  jz @Tail

@BulkLoop:
  // Load 8 int8 values and sign-extend to int32 (4 blocks of 8)
  vpmovsxbd ymm0, [rdx]       // bytes 0..7
  vpmovsxbd ymm1, [rdx+8]     // bytes 8..15
  vpmovsxbd ymm2, [rdx+16]    // bytes 16..23
  vpmovsxbd ymm3, [rdx+24]    // bytes 24..31

  // Convert int32 to float
  vcvtdq2ps ymm0, ymm0
  vcvtdq2ps ymm1, ymm1
  vcvtdq2ps ymm2, ymm2
  vcvtdq2ps ymm3, ymm3

  // Load first two dst blocks (16 elements) and update with FMA
  vmovups ymm6, [rax]         // dst[0..7]
  vmovups ymm7, [rax+32]      // dst[8..15]
  vfmadd231ps ymm6, ymm0, ymm5
  vfmadd231ps ymm7, ymm1, ymm5
  vmovups [rax], ymm6
  vmovups [rax+32], ymm7

  // Load second two dst blocks (16 elements) and update
  vmovups ymm6, [rax+64]      // dst[16..23]
  vmovups ymm7, [rax+96]      // dst[24..31]
  vfmadd231ps ymm6, ymm2, ymm5
  vfmadd231ps ymm7, ymm3, ymm5
  vmovups [rax+64], ymm6
  vmovups [rax+96], ymm7

  add rdx, 32                 // advance codes by 32 bytes
  add rax, 128                // advance dst by 128 bytes
  dec r9d
  jnz @BulkLoop

  vzeroupper

@Tail:
  // r8d = tail count (0..31)
  // Use r9d as working register to preserve rax
  mov r9d, r8d                // r9d = tail
  and r9d, 3                  // scalar remainder (0..3)
  sub r8d, r9d                // r8d = multiple of 4 for XMM

  // Process tail in groups of 4 using XMM
  test r8d, r8d
  jz @ScalarTail

  mov r10d, r8d
  shr r10d, 2                 // number of 4-element groups
@XmmLoop:
  vpmovsxbd xmm0, [rdx]       // load 4 int8
  vcvtdq2ps xmm0, xmm0
  vmovups xmm6, [rax]
  vfmadd231ps xmm6, xmm0, xmm5
  vmovups [rax], xmm6
  add rdx, 4
  add rax, 16
  dec r10d
  jnz @XmmLoop

@ScalarTail:
  // r9d = remaining 0..3 elements
  test r9d, r9d
  jz @Exit

  xor r10d, r10d              // index
@ScalarLoop:
  movsx r11d, byte ptr [rdx + r10]   // load int8 sign-extended
  vcvtsi2ss xmm0, xmm0, r11d
  vmulss xmm0, xmm0, xmm5             // W * code
  vmovss xmm1, [rax + r10*4]          // load dst
  vaddss xmm1, xmm1, xmm0
  vmovss [rax + r10*4], xmm1
  inc r10d
  cmp r10d, r9d
  jl @ScalarLoop

@Exit:
  vzeroupper
  // Restore non-volatile YMM6 and YMM7
  vmovups ymm6, [rsp]
  vmovups ymm7, [rsp+32]
  add rsp, 64
end;

procedure _AVX512MulAddInt8Scalar( dst : PSingle; codes : PShortInt; W : Single; N : Integer ); inline;
begin
  _AVX2MulAddInt8Scalar(dst, codes, W, N);
end;

{-----------------------------------------------------------------------------
  AVX2 int8 elementwise multiply-add: dst[i] += codes[i] * src[i].
  codes is int8, src is float.
  Processes 8 elements per loop using one YMM register.
  Tail (0..7) handled by scalar.
  Win64 calling convention:
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = codes : PShortInt
    R9  = N : Integer
  All registers used are volatile (RCX, RDX, R8, R9, RAX, R10, R11,
  YMM0-YMM3).
-----------------------------------------------------------------------------}
procedure _AVX2MulAddInt8(dst: PSingle; src: PSingle; codes: PShortInt; N: Integer);
asm
  // Load parameters
  mov rax, rcx                // rax = dst
  mov r10, rdx                // r10 = src (bulk pointer)
  mov r11, r8                 // r11 = codes
  mov ecx, r9d                // ecx = N (32-bit)

  test ecx, ecx
  jle @Exit

  // Split bulk (multiples of 8) and tail (0..7)
  mov r8d, ecx
  and r8d, 7                  // tail = N & 7
  sub ecx, r8d                // ecx = bulk (multiple of 8)
  jz @Tail

  mov r9d, ecx
  shr r9d, 3                  // r9d = number of 8-element chunks
  jz @Tail

@Loop8:
  // Load 8 int8 codes -> float
  vpmovsxbd ymm0, [r11]       // codes[0..7]
  vcvtdq2ps ymm0, ymm0

  // Multiply by src[0..7]
  vmovups ymm1, [r10]         // src[0..7]
  vmulps  ymm0, ymm0, ymm1

  // Add to dst[0..7]
  vmovups ymm2, [rax]         // dst[0..7]
  vaddps  ymm0, ymm0, ymm2
  vmovups [rax], ymm0

  add r11, 8                  // codes += 8 bytes
  add r10, 32                 // src += 8 floats
  add rax, 32                 // dst += 8 floats
  dec r9d
  jnz @Loop8

  vzeroupper

@Tail:
  test r8d, r8d
  jz @Exit

  xor r9d, r9d                // index
  // r11 points to codes[bulk], rax to dst[bulk], r10 to src[bulk]
@ScalarLoop:
  movsx r12d, byte ptr [r11 + r9]   // load code sign-extended
  vcvtsi2ss xmm0, xmm0, r12d
  vmovss xmm1, [r10 + r9*4]          // src[bulk + r9]  <-- corrected pointer
  vmulss xmm0, xmm0, xmm1
  vmovss xmm2, [rax + r9*4]          // dst[bulk + r9]
  vaddss xmm2, xmm2, xmm0
  vmovss [rax + r9*4], xmm2
  inc r9d
  cmp r9d, r8d
  jl @ScalarLoop

@Exit:
  vzeroupper
end;

procedure _AVX512MulAddInt8( dst : PSingle; src : PSingle; codes : PShortInt; N : Integer ); inline;
begin
  _AVX2MulAddInt8(dst, src, codes, N);
end;

{-----------------------------------------------------------------------------
  AVX2 max absolute finite: result = max(|src[i]|) for i = 0..N-1,
  ignoring NaN and Inf. Processes 8 elements per loop (YMM).
  Returns the maximum absolute finite value.
  Win64 calling convention:
    RCX = src : PSingle
    RDX = N   : Integer
  Returns Single in XMM0.
  All registers used are volatile (RAX, RCX, RDX, R8, R9, R10, R11,
  YMM0-YMM4).
-----------------------------------------------------------------------------}
function _AVX2MaxAbsFinite(src: PSingle; N: Integer): Single;
asm
  // Load parameters
  mov r10, rcx                // r10 = src
  mov ecx, edx                // ecx = N (32-bit)
  test ecx, ecx
  jle @Zero

  // Constants for absolute value and finite check
  mov eax, $7FFFFFFF          // abs mask
  vmovd xmm3, eax
  vbroadcastss ymm3, xmm3     // ymm3 = {absmask}

  mov eax, $7F7FFFFF          // max finite single
  vmovd xmm2, eax
  vbroadcastss ymm2, xmm2     // ymm2 = {maxfinite}

  // Bulk = N - (N mod 8)
  mov r8d, ecx                // r8d = N
  and r8d, 7                  // tail = N & 7
  sub ecx, r8d                // ecx = bulk (multiple of 8)
  jz @Tail

  mov r9d, ecx
  shr r9d, 3                  // number of 8-element blocks
  jz @Tail

  vxorps ymm4, ymm4, ymm4     // accumulator = 0

@BulkLoop:
  vmovups ymm0, [r10]         // load 8 floats
  vandps ymm0, ymm0, ymm3     // |x|
  vcmpps ymm1, ymm0, ymm2, 18 // LE_OQ: x <= maxfinite ?
  vandps ymm0, ymm0, ymm1     // non-finite -> 0
  vmaxps ymm4, ymm4, ymm0     // update max
  add r10, 32
  dec r9d
  jnz @BulkLoop

  // Reduce ymm4 to scalar in xmm0
  vextractf128 xmm0, ymm4, 1
  vmaxps xmm0, xmm0, xmm4
  vpshufd xmm1, xmm0, $55
  vmaxss xmm0, xmm0, xmm1
  vpshufd xmm1, xmm0, $AA
  vmaxss xmm0, xmm0, xmm1
  vpshufd xmm1, xmm0, $FF
  vmaxss xmm0, xmm0, xmm1
  vzeroupper

@Tail:
  // r8d = tail count (0..7)
  test r8d, r8d
  jz @Return

  // Scalar constants (reload)
  mov eax, $7FFFFFFF
  vmovd xmm3, eax
  mov eax, $7F7FFFFF
  vmovd xmm2, eax
  xor r9d, r9d                // index
@TailLoop:
  vmovss xmm1, [r10 + r9*4]
  vandps xmm1, xmm1, xmm3     // |x|
  vcmpltss xmm4, xmm1, xmm2   // x < maxfinite ?
  vandps xmm1, xmm1, xmm4     // non-finite -> 0
  vmaxss xmm0, xmm0, xmm1     // update max
  inc r9d
  cmp r9d, r8d
  jl @TailLoop

@Return:
  vzeroupper
  ret

@Zero:
  vxorps xmm0, xmm0, xmm0
  ret
end;

function _AVX512MaxAbsFinite( src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2MaxAbsFinite(src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 symmetric int8 quantization:
    dst[i] = clamp(round(src[i] * 127/MaxAbs), -127, 127)
  NaN -> 0, Inf -> +/-127. Processes 8 elements per loop.
  Win64 calling convention:
    RCX = dst : PShortInt
    RDX = src : PSingle
    R8  = N   : Integer
    XMM3 = MaxAbs : Single
  All registers used are volatile (RAX, RCX, RDX, R8, R10, R11,
  YMM0-YMM7).
-----------------------------------------------------------------------------}
procedure _AVX2QuantizeInt8(dst: PShortInt; src: PSingle; N: Integer; const MaxAbs: Single);
asm
  // Load parameters into volatile registers
  mov r10, rcx                // r10 = dst
  mov r11, rdx                // r11 = src
  mov eax, r8d                // eax = N (32-bit)

  test eax, eax
  jle @Exit

  // ---- Compute factor = 127 / MaxAbs (XMM3 = MaxAbs) ----
  mov ecx, 127
  vmovd xmm0, ecx             // xmm0 = 127 (integer)
  vcvtdq2ps xmm0, xmm0        // xmm0 = 127.0f
  vdivss xmm0, xmm0, xmm3     // xmm0 = 127 / MaxAbs
  vbroadcastss ymm5, xmm0     // ymm5 = factor (all lanes)

  // ---- Load +127 and -127 as broadcast constants ----
  mov ecx, $42FE0000          // +127.0
  vmovd xmm6, ecx
  vbroadcastss ymm6, xmm6     // ymm6 = +127

  mov ecx, $C2FE0000          // -127.0
  vmovd xmm7, ecx
  vbroadcastss ymm7, xmm7     // ymm7 = -127

  // ---- Bulk = N - (N mod 8) ----
  mov edx, eax                // edx = N
  and edx, 7                  // tail = N & 7
  sub eax, edx                // eax = bulk (multiple of 8)
  jz @Tail

  mov ecx, eax
  shr ecx, 3                  // ecx = number of 8-element blocks
  jz @Tail

@BulkLoop:
  vmovups ymm0, [r11]         // load 8 src floats
  vcmpps ymm1, ymm0, ymm0, 7  // ORD_Q (false for NaN)
  vandps ymm0, ymm0, ymm1     // NaN -> 0
  vmulps ymm0, ymm0, ymm5     // * factor
  vminps ymm0, ymm0, ymm6     // clip to +127
  vmaxps ymm0, ymm0, ymm7     // clip to -127
  vcvtps2dq ymm0, ymm0        // round to nearest (banker's)
  vextracti128 xmm1, ymm0, 1  // high 4 integers
  vpackssdw xmm0, xmm0, xmm1  // dwords -> words (saturated)
  vpxor xmm2, xmm2, xmm2
  vpacksswb xmm0, xmm0, xmm2  // words -> bytes (saturated)
  vmovq qword ptr [r10], xmm0 // store 8 bytes
  add r11, 32                 // src += 8 (floats)
  add r10, 8                  // dst += 8 (bytes)
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  // edx = tail count (0..7)
  test edx, edx
  jz @Exit

  // Scalar versions of constants (low parts of YMM registers)
  // xmm5 = factor, xmm6 = +127, xmm7 = -127
  xor ecx, ecx                // index
@TailLoop:
  vmovss xmm0, [r11 + rcx*4]  // load src[i]
  vcmpss xmm1, xmm0, xmm0, 7  // ORD_Q (NaN check)
  vandps xmm0, xmm0, xmm1     // NaN -> 0
  vmulss xmm0, xmm0, xmm5     // * factor
  vminss xmm0, xmm0, xmm6     // clip to +127
  vmaxss xmm0, xmm0, xmm7     // clip to -127
  vcvtss2si eax, xmm0         // convert to int (rounding)
  mov byte ptr [r10 + rcx], al // store as signed byte
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
end;

procedure _AVX512QuantizeInt8( dst : PShortInt; src : PSingle; N : Integer; const MaxAbs : Single );
begin
  _AVX2QuantizeInt8(dst, src, N, MaxAbs);
end;

{-----------------------------------------------------------------------------
  AVX2 int8 dequantize: dst[i] = Scale * src[i] for i = 0..N-1.
  Processes 8 elements per loop (YMM), scalar tail.
  Win64 calling convention:
    RCX = dst : PSingle
    RDX = src : PShortInt
    R8  = N   : Integer
    XMM3 = Scale : Single
  All registers used are volatile (RAX, RCX, RDX, R8, R9, R10, R11,
  YMM0-YMM2).
-----------------------------------------------------------------------------}
procedure _AVX2DequantizeInt8(dst: PSingle; src: PShortInt; N: Integer; const Scale: Single);
asm
  // Load parameters into volatile registers
  mov r10, rcx                // r10 = dst
  mov r11, rdx                // r11 = src
  mov eax, r8d                // eax = N (32-bit)

  test eax, eax
  jle @Exit

  // Broadcast Scale from XMM3 to all lanes of YMM2
  vbroadcastss ymm2, xmm3     // ymm2 = Scale

  // Bulk = N - (N mod 8)
  mov edx, eax                // edx = N
  and edx, 7                  // tail = N & 7
  sub eax, edx                // eax = bulk (multiple of 8)
  jz @Tail

  mov ecx, eax
  shr ecx, 3                  // ecx = number of 8-element blocks
  jz @Tail

@BulkLoop:
  // Load 8 int8 values, sign-extend to int32, convert to float
  vpmovsxbd ymm0, [r11]       // 8 bytes -> 8 dwords (sign-extended)
  vcvtdq2ps ymm0, ymm0
  vmulps ymm0, ymm0, ymm2     // * Scale
  vmovups [r10], ymm0

  add r11, 8                  // src += 8 bytes
  add r10, 32                 // dst += 8 floats
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  // edx = tail count (0..7)
  test edx, edx
  jz @Exit

  xor ecx, ecx                // index
  // xmm2 contains Scale in low lane
@TailLoop:
  movsx eax, byte ptr [r11 + rcx]   // load int8 sign-extended
  vcvtsi2ss xmm0, xmm0, eax
  vmulss xmm0, xmm0, xmm2           // * Scale (low lane)
  vmovss [r10 + rcx*4], xmm0
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
end;

procedure _AVX512DequantizeInt8( dst : PSingle; src : PShortInt; N : Integer; const Scale : Single ); inline;
begin
  _AVX2DequantizeInt8(dst, src, N, Scale);
end;

{-----------------------------------------------------------------------------
  AVX2 decode bfloat16 to Single: dst[i] = (float)bfloat16(src[i]).
  Processes 8 elements per loop (YMM).
  Win64 calling convention:
    RCX = dst : PSingle
    RDX = src : PWord
    R8  = N   : Integer
  All registers used are volatile (RAX, RCX, RDX, R8, R9, R10, R11,
  YMM0).
-----------------------------------------------------------------------------}
procedure _AVX2DecodeBF16(dst: PSingle; src: PWord; N: Integer);
asm
  // Load parameters into volatile registers
  mov r10, rcx                // r10 = dst
  mov r11, rdx                // r11 = src
  mov eax, r8d                // eax = N (32-bit)

  test eax, eax
  jle @Exit

  // Bulk = N - (N mod 8)
  mov edx, eax                // edx = N
  and edx, 7                  // tail = N & 7
  sub eax, edx                // eax = bulk (multiple of 8)
  jz @Tail

  mov ecx, eax
  shr ecx, 3                  // ecx = number of 8-element blocks
  jz @Tail

@BulkLoop:
  // Load 8 16-bit bfloat16 values, zero-extend to 32-bit
  vpmovzxwd ymm0, [r11]       // 8 words -> 8 dwords (low 16 bits each)
  vpslld ymm0, ymm0, 16       // shift left by 16 => single-precision bit pattern
  vmovups [r10], ymm0         // store 8 singles

  add r11, 16                 // advance 8 words (16 bytes)
  add r10, 32                 // advance 8 floats (32 bytes)
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  // edx = tail count (0..7)
  test edx, edx
  jz @Exit

  xor ecx, ecx                // index
@TailLoop:
  movzx eax, word ptr [r11 + rcx*2]   // load bfloat16
  shl eax, 16                         // shift to high 16 bits
  mov dword ptr [r10 + rcx*4], eax    // store as single (bit pattern)
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
end;

procedure _AVX512DecodeBF16( dst : PSingle; src : PWord; N : Integer ); inline;
begin
  _AVX2DecodeBF16(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 ReLU gate mask: dst[i] = 1.0 if src[i] >= 0 else 0.0.
  Processes 8 elements per loop.
  Win64 calling convention:
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : Integer
  All registers used are volatile (RAX, RCX, RDX, R8, R10, R11,
  YMM0-YMM3).
-----------------------------------------------------------------------------}
procedure _AVX2ReluGateMask(dst: PSingle; src: PSingle; N: Integer);
asm
  // Load parameters into volatile registers
  mov rax, rcx                // rax = dst
  mov r10, rdx                // r10 = src
  mov ecx, r8d                // ecx = N (32-bit)

  test ecx, ecx
  jle @Exit

  // Load 1.0 constant into YMM2
  mov r11d, $3F800000         // 1.0 bit pattern
  vmovd xmm2, r11d
  vbroadcastss ymm2, xmm2     // ymm2 = 1.0 in all lanes

  vxorps ymm3, ymm3, ymm3     // ymm3 = 0.0

  // Bulk = N - (N mod 8)
  mov edx, ecx                // edx = N
  and edx, 7                  // tail = N & 7
  sub ecx, edx                // ecx = bulk (multiple of 8)
  jz @Tail

  mov r8d, ecx
  shr r8d, 3                  // r8d = number of 8-element blocks
  jz @Tail

@BulkLoop:
  vmovups ymm0, [r10]         // load 8 src values
  vcmpps ymm1, ymm0, ymm3, 29 // GE_OQ: src >= 0 ?
  vandps ymm1, ymm1, ymm2     // mask * 1.0
  vmovups [rax], ymm1         // store mask
  add r10, 32
  add rax, 32
  dec r8d
  jnz @BulkLoop

  vzeroupper

@Tail:
  // edx = tail count (0..7)
  test edx, edx
  jz @Exit

  // xmm2 contains 1.0, xmm3 contains 0.0
  xor r8d, r8d                // index
@TailLoop:
  vmovss xmm0, [r10 + r8*4]
  vcmpss xmm1, xmm0, xmm3, 29 // src >= 0 ?
  vandps xmm1, xmm1, xmm2     // mask * 1.0
  vmovss [rax + r8*4], xmm1
  inc r8d
  cmp r8d, edx
  jl @TailLoop

@Exit:
  vzeroupper
end;

procedure _AVX512ReluGateMask( dst : PSingle; src : PSingle; N : Integer ); inline;
begin
  _AVX2ReluGateMask(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Leaky ReLU: dst[i] = src[i] if src[i] >= 0, else Slope * src[i].
  Uses 256-bit YMM, processes 8 elements per iteration.
  Win64 calling convention:
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : Integer
    XMM3 = Slope : Single
  All registers used are volatile (RAX, RCX, RDX, R8, R9, R10, R11,
  YMM0-YMM4).
-----------------------------------------------------------------------------}
procedure _AVX2LeakyRelu(dst: PSingle; src: PSingle; N: Integer; const Slope: Single);
asm
  // Load parameters into volatile registers
  mov rax, rcx                // rax = dst
  mov r10, rdx                // r10 = src
  mov ecx, r8d                // ecx = N (32-bit)

  test ecx, ecx
  jle @Exit

  // Broadcast Slope from XMM3 to all lanes of YMM2
  vbroadcastss ymm2, xmm3     // ymm2 = Slope

  // Zero constant for comparison
  vxorps ymm3, ymm3, ymm3     // ymm3 = 0.0

  // Bulk = N - (N mod 8)
  mov edx, ecx                // edx = N
  and edx, 7                  // tail = N & 7
  sub ecx, edx                // ecx = bulk (multiple of 8)
  jz @Tail

  mov r8d, ecx
  shr r8d, 3                  // r8d = number of 8-element blocks
  jz @Tail

@BulkLoop:
  vmovups ymm0, [r10]         // load 8 src values
  vmulps ymm1, ymm0, ymm2     // ymm1 = Slope * src
  vcmpps ymm4, ymm0, ymm3, 29 // GE_OQ: src >= 0
  vblendvps ymm1, ymm1, ymm0, ymm4 // select src if >=0 else Slope*src
  vmovups [rax], ymm1
  add r10, 32
  add rax, 32
  dec r8d
  jnz @BulkLoop

  vzeroupper

@Tail:
  // edx = tail count (0..7)
  test edx, edx
  jz @Exit

  // xmm2 contains Slope (low part), xmm3 contains 0.0
  xor r8d, r8d                // index
@TailLoop:
  vmovss xmm0, [r10 + r8*4]   // src[i]
  vmulss xmm1, xmm0, xmm2     // Slope * src
  vcmpss xmm4, xmm0, xmm3, 29 // src >= 0 ?
  vblendvps xmm1, xmm1, xmm0, xmm4 // select
  vmovss [rax + r8*4], xmm1
  inc r8d
  cmp r8d, edx
  jl @TailLoop

@Exit:
  vzeroupper
end;

procedure _AVX512LeakyRelu( dst : PSingle; src : PSingle; N : Integer; const Slope : Single ); inline;
begin
  _AVX2LeakyRelu(dst, src, N, Slope);
end;

{-----------------------------------------------------------------------------
  AVX2 decode F16: dst[i] = half_to_single(src[i]) using F16C (vcvtph2ps).
  Processes 8 elements per loop. Tail (0..7) handled via temporary stack buffer.
  Win64 calling convention:
    RCX = dst : PSingle
    RDX = src : PWord
    R8  = N   : Integer
  All registers used are volatile (RAX, RCX, RDX, R8, R9, R10, R11,
  XMM0, YMM0).
-----------------------------------------------------------------------------}
procedure _AVX2DecodeF16(dst: PSingle; src: PWord; N: Integer);
asm
  // Load parameters into volatile registers
  mov r10, rcx                // r10 = dst
  mov r11, rdx                // r11 = src
  mov eax, r8d                // eax = N (32-bit)

  test eax, eax
  jle @Exit

  // Bulk = N - (N mod 8)
  mov edx, eax                // edx = N
  and edx, 7                  // tail = N & 7
  sub eax, edx                // eax = bulk (multiple of 8)
  jz @TailOnly

  mov ecx, eax
  shr ecx, 3                  // ecx = number of 8-element blocks
  jz @TailOnly

@BulkLoop:
  vmovups xmm0, [r11]         // load 8 halfs (16 bytes)
  vcvtph2ps ymm0, xmm0        // convert to 8 floats
  vmovups [r10], ymm0
  add r11, 16                 // advance 8 halfs (16 bytes)
  add r10, 32                 // advance 8 floats (32 bytes)
  dec ecx
  jnz @BulkLoop

  vzeroupper

@TailOnly:
  // edx = tail count (0..7)
  test edx, edx
  jz @Exit

  // Allocate 16-byte temporary buffer on stack for 8 halfs
  sub rsp, 16

  // Clear buffer (optional, but safe)
  vpxor xmm0, xmm0, xmm0
  vmovups [rsp], xmm0

  // Copy tail halfs to buffer
  xor ecx, ecx
@CopyTail:
  movzx eax, word ptr [r11 + rcx*2]   // load half
  mov word ptr [rsp + rcx*2], ax      // store to buffer
  inc ecx
  cmp ecx, edx
  jl @CopyTail

  // Convert 8 halfs from buffer to floats (ymm0)
  vmovups xmm0, [rsp]
  vcvtph2ps ymm0, xmm0

  // Store only tail elements to dst (sequential singles)
  xor ecx, ecx
@StoreTail:
  vmovss [r10 + rcx*4], xmm0   // store low float
  vpsrldq xmm0, xmm0, 4        // shift right by 4 bytes to get next float
  inc ecx
  cmp ecx, edx
  jl @StoreTail

  // Free temporary buffer
  add rsp, 16

@Exit:
  vzeroupper
end;

procedure _AVX512DecodeF16( dst : PSingle; src : PWord; N : Integer ); inline;
begin
  _AVX2DecodeF16(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 centered sum of squares: result = sum_i (src[i] - Mean)^2.
  Processes 16 elements per loop (2 YMM blocks), then scalar tail.

  Win64 calling convention:
    RCX = src : PSingle
    XMM1 = Mean : Single
    R8  = N : Integer

  Returns Single result in XMM0.
  All YMM registers used are volatile (YMM0-YMM3), no non-volatile saving.
  vzeroupper is called before exit.
-----------------------------------------------------------------------------}
function _AVX2SumSqrCentered(src: PSingle; Mean: Single; N: Integer): Single;
asm
  // Load parameters into volatile registers
  mov r10, rcx                 // r10 = src
  mov r11d, r8d                // r11d = N

  // Broadcast Mean to all lanes of YMM3; keep scalar Mean in XMM4 for tail
  vbroadcastss ymm3, xmm1      // ymm3 = {Mean, Mean, ...}
  vmovaps xmm4, xmm1           // xmm4 = Mean (scalar)

  test r11d, r11d
  jle @Zero

  // ---- Split bulk (multiple of 16) and tail (0..15) ----
  mov r8d, r11d
  and r8d, 15                  // tail = N & 15
  sub r11d, r8d                // bulk = N - tail
  jz @Tail

  mov r9d, r11d
  shr r9d, 4                   // number of 16-element blocks
  jz @Tail

  // ---- Bulk accumulators ----
  vxorps ymm0, ymm0, ymm0      // accumulator for block 0
  vxorps ymm1, ymm1, ymm1      // accumulator for block 1

@BulkLoop:
  // Block 0: elements 0..7
  vmovups ymm2, [r10]
  vsubps  ymm2, ymm2, ymm3
  vfmadd231ps ymm0, ymm2, ymm2

  // Block 1: elements 8..15
  vmovups ymm2, [r10+32]
  vsubps  ymm2, ymm2, ymm3
  vfmadd231ps ymm1, ymm2, ymm2

  add r10, 64                  // advance 16 elements (64 bytes)
  dec r9d
  jnz @BulkLoop

  // ---- Reduce bulk sum to scalar in XMM0 ----
  vaddps ymm0, ymm0, ymm1
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0     // xmm0[0] = sum of bulk
  vzeroupper

@Tail:
  // ---- Handle remaining 0..15 elements ----
  test r8d, r8d
  jz @Return

  xor r9d, r9d                 // index
@TailLoop:
  vmovss xmm2, [r10 + r9*4]
  vsubss xmm2, xmm2, xmm4      // src - Mean
  vmulss xmm2, xmm2, xmm2      // square
  vaddss xmm0, xmm0, xmm2      // accumulate
  inc r9d
  cmp r9d, r8d
  jl @TailLoop

@Return:
  vzeroupper
  ret

@Zero:
  vxorps xmm0, xmm0, xmm0
  ret
end;

function _AVX512SumSqrCentered( src : PSingle; Mean : Single; N : Integer ) : Single; inline;
begin
  Result := _AVX2SumSqrCentered(src, Mean, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Adam optimizer step (32-bit).
  Updates per element:
    m := Beta1*m + OmBeta1*g
    v := Beta2*v + OmBeta2*(g*g)
    delta := (kLR*m) / (sqrt(v) + Epsilon)

  Processes 8 elements per loop (YMM), scalar tail.

  Win64 calling convention:
    RCX = PtrDelta
    RDX = PtrM
    R8  = PtrV
    XMM3 = Beta1
    [RBP+48] = OmBeta1  (compiler-established frame)
    [RBP+56] = Beta2
    [RBP+64] = OmBeta2
    [RBP+80] = Epsilon
    [RBP+88] = kLR
    [RBP+96] = NumElements

  All YMM registers used are volatile (YMM0-YMM5, YMM7).
  No manual stack frame manipulation.
-----------------------------------------------------------------------------}
procedure _AVX2AdamDelta(PtrDelta, PtrM, PtrV: PSingle;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR: Single;
  NumElements: Integer);
asm
  // ---- Load pointer parameters into volatile registers ----
  mov r10, rcx                // r10 = PtrDelta
  mov r11, rdx                // r11 = PtrM
  mov r9,  r8                 // r9  = PtrV

  // ---- Broadcast constants (using compiler's RBP) ----
  vbroadcastss ymm0, xmm3                    // Beta1
  vbroadcastss ymm1, dword ptr [rbp+48]      // OmBeta1
  vbroadcastss ymm2, dword ptr [rbp+56]      // Beta2
  vbroadcastss ymm3, dword ptr [rbp+64]      // OmBeta2
  vbroadcastss ymm4, dword ptr [rbp+80]      // Epsilon
  vbroadcastss ymm5, dword ptr [rbp+88]      // kLR

  // ---- Load NumElements ----
  mov eax, dword ptr [rbp+96]
  test eax, eax
  jle @Exit

  // ---- Split bulk (multiple of 8) and tail (0..7) ----
  mov edx, eax
  and edx, 7
  sub eax, edx
  jz @Tail

  mov ecx, eax
  shr ecx, 3                  // number of 8-element blocks
  jz @Tail

@BulkLoop:
  // ---- m = Beta1*m + OmBeta1*g ----
  vmovups ymm7, [r10]         // g
  vmulps ymm7, ymm7, ymm1     // OmBeta1 * g
  vmovups ymm6, [r11]         // old m
  vfmadd231ps ymm7, ymm6, ymm0 // m = Beta1*m + OmBeta1*g
  vmovups [r11], ymm7

  // ---- v = Beta2*v + OmBeta2*(g*g) ----
  vmovups ymm7, [r10]         // g (reload)
  vmulps ymm7, ymm7, ymm7     // g^2
  vmulps ymm7, ymm7, ymm3     // OmBeta2 * g^2
  vmovups ymm6, [r9]          // old v
  vfmadd231ps ymm7, ymm6, ymm2 // v = Beta2*v + OmBeta2*(g*g)
  vmovups [r9], ymm7

  // ---- delta = (kLR*m) / (sqrt(v) + Epsilon) ----
  vmovups ymm6, [r11]         // m
  vmulps ymm6, ymm6, ymm5     // kLR * m
  vsqrtps ymm7, [r9]          // sqrt(v)
  vaddps ymm7, ymm7, ymm4     // sqrt(v) + Epsilon
  vdivps ymm6, ymm6, ymm7     // delta
  vmovups [r10], ymm6

  add r10, 32
  add r11, 32
  add r9,  32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  xor ecx, ecx
@TailLoop:
  // ---- m = Beta1*m + OmBeta1*g ----
  vmovss xmm7, [r10 + rcx*4]  // g
  vmulss xmm7, xmm7, xmm1
  vmovss xmm6, [r11 + rcx*4]  // old m
  vmulss xmm6, xmm6, xmm0
  vaddss xmm7, xmm7, xmm6
  vmovss [r11 + rcx*4], xmm7

  // ---- v = Beta2*v + OmBeta2*(g*g) ----
  vmovss xmm7, [r10 + rcx*4]  // g
  vmulss xmm7, xmm7, xmm7
  vmulss xmm7, xmm7, xmm3
  vmovss xmm6, [r9 + rcx*4]   // old v
  vmulss xmm6, xmm6, xmm2
  vaddss xmm7, xmm7, xmm6
  vmovss [r9 + rcx*4], xmm7

  // ---- delta = (kLR*m) / (sqrt(v) + Epsilon) ----
  vmovss xmm6, [r11 + rcx*4]  // m
  vmulss xmm6, xmm6, xmm5
  vsqrtss xmm7, xmm7, [r9 + rcx*4]  // sqrt(v)
  vaddss xmm7, xmm7, xmm4
  vdivss xmm6, xmm6, xmm7
  vmovss [r10 + rcx*4], xmm6

  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper

end;

procedure _AVX512AdamDelta( PtrDelta, PtrM, PtrV : PSingle;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR : Single;
  NumElements : Integer ); inline;
begin
  _AVX2AdamDelta(PtrDelta, PtrM, PtrV, Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR, NumElements);
end;

{-----------------------------------------------------------------------------
  AVX2 Adafactor step (32-bit, Win64 calling convention).
  Updates per element:
    v := Beta2 * v + (k * d * d + c)
    d := (k * d) / (sqrt(v) + Epsilon)

  Win64 calling convention (observed):
    RCX = PtrDelta
    RDX = PtrV
    XMM2 = Beta2
    XMM3 = k
    [RBP+48] = c
    [RBP+56] = Epsilon
    [RBP+64] = NumElements
-----------------------------------------------------------------------------}
procedure _AVX2AdafactorDelta(PtrDelta, PtrV: PSingle;
  Beta2, k, c, Epsilon: Single; NumElements: Integer);
asm
  mov r10, rcx
  mov r11, rdx

  vbroadcastss ymm0, xmm2     // Beta2
  vbroadcastss ymm1, xmm3     // k
  vbroadcastss ymm2, [rbp+48] // c
  vbroadcastss ymm3, [rbp+56] // Epsilon
  mov eax, [rbp+64]           // NumElements

  test eax, eax
  jle @Exit

  mov edx, eax
  and edx, 7
  sub eax, edx
  jz @Tail

  mov ecx, eax
  shr ecx, 3
  jz @Tail

@BulkLoop:
  vmovups ymm7, [r10]         // d
  vmulps ymm6, ymm7, ymm7
  vmulps ymm6, ymm6, ymm1     // k
  vaddps ymm6, ymm6, ymm2     // + c
  vmovups ymm5, [r11]
  vmulps ymm5, ymm5, ymm0     // Beta2 * v
  vaddps ymm6, ymm6, ymm5
  vmovups [r11], ymm6

  vsqrtps ymm6, ymm6
  vaddps ymm6, ymm6, ymm3     // + Epsilon
  vmulps ymm7, ymm7, ymm1     // k * d
  vdivps ymm7, ymm7, ymm6
  vmovups [r10], ymm7

  add r10, 32
  add r11, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  vmovaps xmm4, xmm0          // Beta2
  vmovaps xmm5, xmm1          // k
  vmovss xmm6, [rbp+48]       // c
  vmovss xmm7, [rbp+56]       // Epsilon

  xor ecx, ecx
@TailLoop:
  vmovss xmm0, [r10 + rcx*4]  // d
  vmulss xmm1, xmm0, xmm0
  vmulss xmm1, xmm1, xmm5     // k
  vaddss xmm1, xmm1, xmm6     // + c
  vmovss xmm2, [r11 + rcx*4]
  vmulss xmm2, xmm2, xmm4     // Beta2 * v
  vaddss xmm1, xmm1, xmm2
  vmovss [r11 + rcx*4], xmm1

  vsqrtss xmm2, xmm1, xmm1
  vaddss xmm2, xmm2, xmm7     // + Epsilon
  vmulss xmm0, xmm0, xmm5     // k * d
  vdivss xmm0, xmm0, xmm2
  vmovss [r10 + rcx*4], xmm0

  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
end;

procedure _AVX512AdafactorDelta( PtrDelta, PtrV : PSingle;
  Beta2, k, c, Epsilon : Single; NumElements : Integer ); inline;
begin
  _AVX2AdafactorDelta(PtrDelta, PtrV, Beta2, k, c, Epsilon, NumElements);
end;

{-----------------------------------------------------------------------------
  AVX2 clamp absolute: dst[i] := clamp(dst[i], -Value, +Value).
  Uses 256-bit YMM, processes 8 elements per loop.
  Win64 calling convention:
    RCX = PtrA : PSingle
    XMM1 = Value : Single
    R8  = NumElements : Integer
  All registers used are volatile (RAX, RCX, RDX, R8, R9, R10, R11,
  XMM0-XMM4, YMM0-YMM2). No prologue/epilogue needed.
-----------------------------------------------------------------------------}
procedure _AVX2ClampAbs(PtrA: PSingle; Value: Single; NumElements: Integer);
asm
  // Load parameters into volatile registers
  mov rax, rcx                // rax = PtrA
  mov ecx, r8d                // ecx = NumElements (32-bit)

  test ecx, ecx
  jle @Exit

  // Broadcast +Value to YMM0, -Value to YMM1
  vbroadcastss ymm0, xmm1     // ymm0 = +Value
  vxorps ymm1, ymm1, ymm1
  vsubps ymm1, ymm1, ymm0     // ymm1 = -Value

  // Bulk = N - (N mod 8)
  mov edx, ecx                // edx = N
  and edx, 7                  // tail = N & 7
  sub ecx, edx                // ecx = bulk (multiple of 8)
  jz @Tail

  mov r8d, ecx
  shr r8d, 3                  // r8d = number of 8-element blocks
  jz @Tail

@BulkLoop:
  vmovups ymm2, [rax]         // load 8 elements
  vmaxps ymm2, ymm1, ymm2     // clamp lower bound
  vminps ymm2, ymm0, ymm2     // clamp upper bound
  vmovups [rax], ymm2         // store back
  add rax, 32
  dec r8d
  jnz @BulkLoop

  vzeroupper

@Tail:
  // edx = tail count (0..7)
  test edx, edx
  jz @Exit

  // Scalar tail using VEX-encoded instructions
  // xmm0 = +Value, xmm1 = -Value (low parts)
  xor r8d, r8d                // index
@TailLoop:
  vmovss xmm2, [rax + r8*4]   // load element
  vmaxss xmm2, xmm1, xmm2     // clamp lower bound
  vminss xmm2, xmm0, xmm2     // clamp upper bound
  vmovss [rax + r8*4], xmm2   // store back
  inc r8d
  cmp r8d, edx
  jl @TailLoop

@Exit:
  vzeroupper
end;

procedure _AVX512ClampAbs( PtrA : PSingle; Value : Single; NumElements : Integer ); inline;
begin
  _AVX2ClampAbs(PtrA, Value, NumElements);
end;

{-----------------------------------------------------------------------------
  AVX2 Lion optimizer step (32-bit) - Win64 version.
  Updates per element:
    c = Beta1 * m + k1 * g
    m_new = Beta2 * m + k2 * g
    delta = -LR if c > 0, +LR if c < 0, else 0
  Processes 8 elements per loop (YMM), scalar tail.

  Win64 calling convention (observed):
    RCX = PtrDelta
    RDX = PtrM
    XMM2 = Beta1
    XMM3 = k1
    [RBP+48] = Beta2
    [RBP+56] = k2
    [RBP+64] = NegLR
    [RBP+72] = PosLR
    [RBP+80] = NumElements

  Uses volatile high XMM registers (XMM8-XMM13) to preserve constants.
  Non-volatile YMM6 and YMM7 are saved/restored.
-----------------------------------------------------------------------------}
{-----------------------------------------------------------------------------
  AVX2 Lion optimizer step (32-bit) - Win64 version.
  Updates per element:
    c = Beta1 * m + k1 * g
    m_new = Beta2 * m + k2 * g
    delta = -LR if c > 0, +LR if c < 0, else 0
  Processes 8 elements per loop (YMM), scalar tail.

  Win64 calling convention (observed):
    RCX = PtrDelta
    RDX = PtrM
    XMM2 = Beta1
    XMM3 = k1
    [RBP+48] = Beta2
    [RBP+56] = k2
    [RBP+64] = NegLR
    [RBP+72] = PosLR
    [RBP+80] = NumElements

  Uses volatile high XMM registers (XMM8-XMM13) to preserve constants.
  Non-volatile YMM6 and YMM7 are saved/restored.
-----------------------------------------------------------------------------}
{-----------------------------------------------------------------------------
  AVX2 Lion optimizer step (32-bit) - Win64 version.
  Updates per element:
    c = Beta1 * m + k1 * g
    m_new = Beta2 * m + k2 * g
    delta = -LR if c > 0, +LR if c < 0, else 0
  Processes 8 elements per loop (YMM), scalar tail.

  Win64 calling convention (observed):
    RCX = PtrDelta
    RDX = PtrM
    XMM2 = Beta1
    XMM3 = k1
    [RBP+48] = Beta2
    [RBP+56] = k2
    [RBP+64] = NegLR
    [RBP+72] = PosLR
    [RBP+80] = NumElements

  Uses volatile high XMM registers (XMM8-XMM13) to preserve constants.
  Non-volatile YMM6 and YMM7 are saved/restored.
-----------------------------------------------------------------------------}
{-----------------------------------------------------------------------------
  AVX2 Lion optimizer step (32-bit) - Win64 version.
  Updates per element:
    c = Beta1 * m + k1 * g
    m_new = Beta2 * m + k2 * g
    delta = -LR if c > 0, +LR if c < 0, else 0
  Processes 8 elements per loop (YMM), scalar tail.

  Win64 calling convention (observed):
    RCX = PtrDelta
    RDX = PtrM
    XMM2 = Beta1
    XMM3 = k1
    [RBP+48] = Beta2
    [RBP+56] = k2
    [RBP+64] = NegLR
    [RBP+72] = PosLR
    [RBP+80] = NumElements

  Uses volatile high XMM registers (XMM8-XMM13) to preserve constants.
  Non-volatile YMM6 and YMM7 are saved/restored.
-----------------------------------------------------------------------------}
procedure _AVX2LionDelta(PtrDelta, PtrM: PSingle;
  Beta1, k1, Beta2, k2, NegLR, PosLR: Single;
  NumElements: Integer);
asm
  sub rsp, 64
  vmovups [rsp], ymm6
  vmovups [rsp+32], ymm7

  vmovaps xmm8, xmm2          // Beta1
  vmovaps xmm9, xmm3          // k1
  vmovss xmm10, [rbp+48]      // Beta2
  vmovss xmm11, [rbp+56]      // k2
  vmovss xmm12, [rbp+64]      // NegLR
  vmovss xmm13, [rbp+72]      // PosLR

  mov r10, rcx
  mov r11, rdx
  mov eax, [rbp+80]           // NumElements

  test eax, eax
  jle @Exit

  mov edx, eax
  and edx, 7
  sub eax, edx
  jz @Tail

  mov ecx, eax
  shr ecx, 3
  jz @Tail

  vbroadcastss ymm2, xmm8     // Beta1
  vbroadcastss ymm9, xmm9     // k1 (avoid ymm3 confusion)
  vbroadcastss ymm4, xmm10    // Beta2
  vbroadcastss ymm5, xmm11    // k2

@BulkLoop:
  vmovups ymm0, [r10]         // g
  vmovups ymm1, [r11]         // m

  // c = Beta1*m + k1*g
  vmulps ymm6, ymm1, ymm2
  vfmadd231ps ymm6, ymm0, ymm9

  // m_new = Beta2*m + k2*g
  vmovaps ymm7, ymm1
  vmulps ymm7, ymm7, ymm4
  vfmadd231ps ymm7, ymm0, ymm5
  vmovups [r11], ymm7

  // delta selection
  vxorps ymm7, ymm7, ymm7
  vcmpps ymm0, ymm6, ymm7, 6  // c > 0
  vcmpps ymm1, ymm6, ymm7, 1  // c < 0

  vbroadcastss ymm6, xmm12    // NegLR
  vbroadcastss ymm7, xmm13    // PosLR

  vandps ymm0, ymm0, ymm6
  vandps ymm1, ymm1, ymm7
  vorps ymm0, ymm0, ymm1
  vmovups [r10], ymm0

  add r10, 32
  add r11, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  // 重新加载尾部常量（确保正确性）
  vmovss xmm14, [rbp+48]      // Beta2
  vmovss xmm15, [rbp+56]      // k2

  xor ecx, ecx
@TailLoop:
  vmovss xmm0, [r10 + rcx*4]  // g
  vmovss xmm1, [r11 + rcx*4]  // m

  // 保存 g 到 xmm2 (使用 vmovaps 复制)
  vmovaps xmm2, xmm0

  // c = Beta1*m + k1*g
  vmulss xmm1, xmm1, xmm8
  vmulss xmm0, xmm0, xmm9
  vaddss xmm0, xmm0, xmm1

  // delta selection
  vxorps xmm1, xmm1, xmm1
  comiss xmm0, xmm1
  jg @Greater
  jl @Less
  xorps xmm0, xmm0
  jmp @StoreDelta
@Greater:
  vmovaps xmm0, xmm12
  jmp @StoreDelta
@Less:
  vmovaps xmm0, xmm13
@StoreDelta:
  vmovss [r10 + rcx*4], xmm0

  // m_new = Beta2*m + k2*g (使用保存的 g 在 xmm2)
  vmovss xmm0, [r11 + rcx*4]  // m (reload)
  vmulss xmm0, xmm0, xmm14
  vmulss xmm2, xmm2, xmm15
  vaddss xmm0, xmm0, xmm2
  vmovss [r11 + rcx*4], xmm0

  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
  vmovups ymm6, [rsp]
  vmovups ymm7, [rsp+32]
  add rsp, 64
end;

procedure _AVX512LionDelta( PtrDelta, PtrM : PSingle; Beta1, k1, Beta2, k2, NegLR, PosLR : Single; NumElements : Integer ); inline;
begin
  _AVX2LionDelta(PtrDelta, PtrM, Beta1, k1, Beta2, k2, NegLR, PosLR, NumElements);
end;

{-----------------------------------------------------------------------------
  AVX2 max value and first occurrence index (32-bit, 8 elements per loop).
  Win64 calling convention:
    RCX = PtrA        : PSingle
    RDX = NumElements : Integer
    R8  = Position    : PInteger (out)
  Returns max value in XMM0 (Single).
  Uses only volatile registers YMM0-YMM5, no non-volatile saving.
  Handles any N (including N<8 via scalar fallback).
-----------------------------------------------------------------------------}
function _AVX2GetMaxPos(PtrA: PSingle; NumElements: Integer; out Position: Integer): Single;
asm
  // ---- Load parameters ----
  mov r10, rcx                // r10 = PtrA
  mov r11, rdx                // r11 = NumElements
  mov r9, r8                  // r9 = @Position

  test r11d, r11d
  jle @Zero

  // ---- If N < 8, use scalar fallback ----
  cmp r11d, 8
  jl @ScalarFallback

  // ---- YMM path: bulk (multiple of 8) + tail (0..7) ----
  mov eax, r11d
  mov edx, eax
  and edx, 7                  // tail = N & 7
  sub eax, edx                // bulk = N - tail (multiple of 8)

  // Save bulk count and tail count to scratch stack space
  mov [rsp+8], rax            // bulk count (64-bit, but we'll use 32-bit later)
  mov [rsp+16], rdx           // tail count

  // ---- Load lane indices 0..7 ----
  vmovdqu ymm1, yword ptr [cAVXArgLaneSeed]

  // ---- Generate step vector of 8's ----
  mov dword ptr [rsp+24], 8
  vpbroadcastd ymm2, [rsp+24]

  // ---- Load first 8 elements as initial candidates ----
  vmovups ymm0, [r10]         // ymm0 = max values
  add r10, 32

  mov ecx, eax
  shr ecx, 3                  // number of full blocks
  dec ecx                     // first block already processed
  jz @Fold

@BulkLoop:
  vmovups ymm3, [r10]         // next 8 values
  vpaddd ymm4, ymm1, ymm2     // new indices = current + 8
  vcmpps ymm5, ymm3, ymm0, 22 // ymm3 > ymm0 ?
  vblendvps ymm0, ymm0, ymm3, ymm5
  vblendvps ymm1, ymm1, ymm4, ymm5
  add r10, 32
  dec ecx
  jnz @BulkLoop

@Fold:
  // ---- Reduce 8 lanes to scalar max + index (using only xmm0..xmm5) ----
  vextractf128 xmm2, ymm0, 1   // high 4 values
  vextracti128 xmm3, ymm1, 1   // high 4 indices
  vcmpps xmm5, xmm0, xmm2, 22  // low > high ?
  vblendvps xmm4, xmm3, xmm1, xmm5  // selected index
  vmaxps xmm0, xmm0, xmm2      // max values (low/high combined)

  // Reduce 4 -> 2 lanes
  vpshufd xmm2, xmm0, $55
  vpshufd xmm3, xmm4, $55
  vcmpps xmm5, xmm0, xmm2, 22
  vblendvps xmm4, xmm3, xmm4, xmm5
  vmaxss xmm0, xmm0, xmm2

  // Reduce 2 -> 1 lane
  vpshufd xmm2, xmm0, $AA
  vpshufd xmm3, xmm4, $AA
  vcmpps xmm5, xmm0, xmm2, 22
  vblendvps xmm4, xmm3, xmm4, xmm5
  vmaxss xmm0, xmm0, xmm2

  // Final reduction (1 lane)
  vpshufd xmm2, xmm0, $FF
  vpshufd xmm3, xmm4, $FF
  vcmpps xmm5, xmm0, xmm2, 22
  vblendvps xmm4, xmm3, xmm4, xmm5
  vmaxss xmm0, xmm0, xmm2

  // xmm0[0] = max, xmm4[0] = index
  vmovd [rsp+32], xmm4        // store index temporarily
  vzeroupper
  mov eax, [rsp+32]           // eax = bulk index

  // ---- Tail handling (0..7 elements) ----
  mov edx, [rsp+16]           // tail count (32-bit)
  test edx, edx
  jz @TailDone

  // r10 already points to the start of tail (after bulk loop)
  // Base index for tail = bulk count (32-bit)
  mov edi, [rsp+8]            // edi = bulk count (32-bit)
  xor ecx, ecx                // ecx = tail index (32-bit)
@TailLoop:
  vmovss xmm1, [r10 + rcx*4]
  comiss xmm1, xmm0
  jbe @TailSkip
  movaps xmm0, xmm1           // update max
  lea eax, [edi + ecx]        // new position = bulk_count + tail_index (both 32-bit)
@TailSkip:
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@TailDone:
  mov [r9], eax               // store final position (r9 is 64-bit pointer)
  jmp @Exit

  // ---- Scalar fallback for N < 8 ----
@ScalarFallback:
  vmovss xmm0, [r10]          // init with first element
  xor eax, eax                // index = 0
  mov ecx, 1                  // ecx = 1
@ScalarLoop:
  cmp ecx, r11d
  jge @ScalarDone
  vmovss xmm1, [r10 + rcx*4]
  comiss xmm1, xmm0
  jbe @ScalarSkip
  movaps xmm0, xmm1
  mov eax, ecx
@ScalarSkip:
  inc ecx
  jmp @ScalarLoop
@ScalarDone:
  mov [r9], eax
  jmp @Exit

@Zero:
  xorps xmm0, xmm0
  mov dword ptr [r9], 0

@Exit:
  vzeroupper
  ret                         // result already in xmm0
end;

function _AVX512GetMaxPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;
begin
  Result := _AVX2GetMaxPos(PtrA, NumElements, Position);
end;

{-----------------------------------------------------------------------------
  AVX2 min value and first occurrence index (32-bit, 8 elements per loop).
  Win64 calling convention:
    RCX = PtrA        : PSingle
    RDX = NumElements : Integer
    R8  = Position    : PInteger (out)
  Returns min value in XMM0 (Single).
  Uses only volatile registers YMM0-YMM5, no non-volatile saving.
  Handles any N (including N<8 via scalar fallback).
-----------------------------------------------------------------------------}
function _AVX2GetMinPos(PtrA: PSingle; NumElements: Integer; out Position: Integer): Single;
asm
  // ---- Load parameters ----
  mov r10, rcx                // r10 = PtrA
  mov r11, rdx                // r11 = NumElements
  mov r9, r8                  // r9 = @Position

  test r11d, r11d
  jle @Zero

  // ---- If N < 8, use scalar fallback ----
  cmp r11d, 8
  jl @ScalarFallback

  // ---- YMM path: bulk (multiple of 8) + tail (0..7) ----
  mov eax, r11d
  mov edx, eax
  and edx, 7                  // tail = N & 7
  sub eax, edx                // bulk = N - tail (multiple of 8)

  // Save bulk count and tail count to scratch stack space
  mov [rsp+8], rax            // bulk count (64-bit, but we'll use 32-bit later)
  mov [rsp+16], rdx           // tail count (32-bit)

  // ---- Load lane indices 0..7 ----
  vmovdqu ymm1, yword ptr [cAVXArgLaneSeed]

  // ---- Generate step vector of 8's ----
  mov dword ptr [rsp+24], 8
  vpbroadcastd ymm2, [rsp+24]

  // ---- Load first 8 elements as initial candidates ----
  vmovups ymm0, [r10]         // ymm0 = min values
  add r10, 32

  mov ecx, eax
  shr ecx, 3                  // number of full blocks
  dec ecx                     // first block already processed
  jz @Fold

@BulkLoop:
  vmovups ymm3, [r10]         // next 8 values
  vpaddd ymm4, ymm1, ymm2     // new indices = current + 8
  vcmpps ymm5, ymm3, ymm0, 17 // ymm3 < ymm0 ?
  vblendvps ymm0, ymm0, ymm3, ymm5
  vblendvps ymm1, ymm1, ymm4, ymm5
  add r10, 32
  dec ecx
  jnz @BulkLoop

@Fold:
  // ---- Reduce 8 lanes to scalar min + index (using only xmm0..xmm5) ----
  vextractf128 xmm2, ymm0, 1   // high 4 values
  vextracti128 xmm3, ymm1, 1   // high 4 indices
  vcmpps xmm5, xmm0, xmm2, 17  // low < high ?
  vblendvps xmm4, xmm3, xmm1, xmm5  // selected index
  vminps xmm0, xmm0, xmm2      // min values (low/high combined)

  // Reduce 4 -> 2 lanes
  vpshufd xmm2, xmm0, $55
  vpshufd xmm3, xmm4, $55
  vcmpps xmm5, xmm0, xmm2, 17
  vblendvps xmm4, xmm3, xmm4, xmm5
  vminss xmm0, xmm0, xmm2

  // Reduce 2 -> 1 lane
  vpshufd xmm2, xmm0, $AA
  vpshufd xmm3, xmm4, $AA
  vcmpps xmm5, xmm0, xmm2, 17
  vblendvps xmm4, xmm3, xmm4, xmm5
  vminss xmm0, xmm0, xmm2

  // Final reduction (1 lane)
  vpshufd xmm2, xmm0, $FF
  vpshufd xmm3, xmm4, $FF
  vcmpps xmm5, xmm0, xmm2, 17
  vblendvps xmm4, xmm3, xmm4, xmm5
  vminss xmm0, xmm0, xmm2

  // xmm0[0] = min, xmm4[0] = index
  vmovd [rsp+32], xmm4        // store index temporarily
  vzeroupper
  mov eax, [rsp+32]           // eax = bulk index

  // ---- Tail handling (0..7 elements) ----
  mov edx, [rsp+16]           // tail count (32-bit)
  test edx, edx
  jz @TailDone

  // r10 already points to the start of tail (after bulk loop)
  // Base index for tail = bulk count (32-bit)
  mov edi, [rsp+8]            // edi = bulk count (32-bit)
  xor ecx, ecx                // ecx = tail index (32-bit)
@TailLoop:
  vmovss xmm1, [r10 + rcx*4]
  comiss xmm1, xmm0
  jae @TailSkip               // if >=, skip (keep first occurrence)
  movaps xmm0, xmm1           // update min
  lea eax, [edi + ecx]        // new position = bulk_count + tail_index (both 32-bit)
@TailSkip:
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@TailDone:
  mov [r9], eax               // store final position (r9 is 64-bit pointer)
  jmp @Exit

  // ---- Scalar fallback for N < 8 ----
@ScalarFallback:
  vmovss xmm0, [r10]          // init with first element
  xor eax, eax                // index = 0
  mov ecx, 1                  // ecx = 1
@ScalarLoop:
  cmp ecx, r11d
  jge @ScalarDone
  vmovss xmm1, [r10 + rcx*4]
  comiss xmm1, xmm0
  jae @ScalarSkip             // if >=, keep first
  movaps xmm0, xmm1
  mov eax, ecx
@ScalarSkip:
  inc ecx
  jmp @ScalarLoop
@ScalarDone:
  mov [r9], eax
  jmp @Exit

@Zero:
  xorps xmm0, xmm0
  mov dword ptr [r9], 0

@Exit:
  vzeroupper
  ret                         // result already in xmm0
end;

function _AVX512GetMinPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;
begin
  Result := _AVX2GetMinPos(PtrA, NumElements, Position);
end;

{-----------------------------------------------------------------------------
  AVX2 max absolute value and first occurrence index (32-bit, 8 elements/loop).
  Win64 calling convention:
    RCX = PtrA        : PSingle
    RDX = NumElements : Integer
    R8  = Position    : PInteger (out)
  Returns max absolute value in XMM0 (Single).
  Saves and restores non-volatile YMM6, YMM7.
  Handles any N (including N<8 via scalar fallback).
  All pointer arithmetic uses 64-bit registers.

  Fixes:
  1. Reload abs mask into xmm2 before tail loop (was clobbered during fold).
  2. Save/restore non-volatile YMM6, YMM7 (Win64 ABI requirement).
-----------------------------------------------------------------------------}
function _AVX2GetMaxAbsPos(PtrA: PSingle; NumElements: Integer; out Position: Integer): Single;
asm
  // ---- Save non-volatile YMM6, YMM7 ----
  // Allocate 64 bytes, 16-byte aligned (RSP is 8 mod 16 on entry, sub 64 keeps 8 mod 16)
  sub rsp, 64
  vmovdqu [rsp], ymm6
  vmovdqu [rsp+32], ymm7

  // ---- Load parameters ----
  mov r10, rcx                // r10 = PtrA (64-bit)
  mov r11, rdx                // r11 = NumElements (64-bit, use 32-bit part)
  mov r9, r8                  // r9 = @Position (64-bit)

  test r11d, r11d
  jle @Zero

  // ---- If N < 8, use scalar fallback ----
  cmp r11d, 8
  jl @ScalarFallback

  // ---- YMM path: bulk (multiple of 8) + tail (0..7) ----
  mov eax, r11d
  mov edx, eax
  and edx, 7                  // tail = N & 7
  sub eax, edx                // bulk = N - tail (multiple of 8)

  // Save bulk count and tail count to scratch stack (shadow space)
  // Note: after sub rsp, 64, old [rsp+8] becomes [rsp+72]
  mov [rsp+72], rax           // bulk count (64-bit)
  mov [rsp+80], rdx           // tail count (64-bit)

  // ---- Load absolute mask (clear sign bit) into ymm2 ----
  vmovdqu ymm2, yword ptr [cAVXArgAbsMask]   // ymm2 = {0x7FFFFFFF,...}

  // ---- Load lane indices 0..7 ----
  vmovdqu ymm1, yword ptr [cAVXArgLaneSeed]  // ymm1 = {0,1,2,3,4,5,6,7}

  // ---- Generate step vector of 8's in ymm3 ----
  mov dword ptr [rsp+88], 8
  vpbroadcastd ymm3, [rsp+88]   // ymm3 = {8,8,8,8,8,8,8,8}

  // ---- Seed with first 8 elements, take absolute values ----
  vmovups ymm0, [r10]          // ymm0 = original values
  vandps ymm0, ymm0, ymm2      // ymm0 = |values|
  add r10, 32

  // ---- Initialize base counter (ymm4) to 8 (first block processed) ----
  mov dword ptr [rsp+92], 8
  vpbroadcastd ymm4, [rsp+92]  // ymm4 = {8,8,8,8,8,8,8,8}

  mov ecx, eax
  shr ecx, 3                   // number of full blocks
  dec ecx                      // first block already processed
  jz @Fold

@BulkLoop:
  vmovups ymm5, [r10]          // load next 8 values
  vandps ymm5, ymm5, ymm2      // absolute values

  // ---- Compute new indices: base + lane offsets ----
  vmovdqu ymm6, yword ptr [cAVXArgLaneSeed]   // ymm6 = lane offsets (0..7)
  vpaddd ymm6, ymm4, ymm6      // ymm6 = base + offset

  vcmpps ymm7, ymm5, ymm0, 6   // ymm5 > ymm0 ? (condition 6 = greater-than, ordered)
  vblendvps ymm0, ymm0, ymm5, ymm7   // update max absolute values
  vblendvps ymm1, ymm1, ymm6, ymm7   // update indices

  vpaddd ymm4, ymm4, ymm3      // base += 8 for next block

  add r10, 32
  dec ecx
  jnz @BulkLoop

@Fold:
  // ---- Reduce 8 lanes to scalar max + index (using xmm0..xmm5) ----
  vextractf128 xmm2, ymm0, 1   // high 4 values
  vextracti128 xmm3, ymm1, 1   // high 4 indices
  vcmpps xmm5, xmm0, xmm2, 6   // low > high ?
  vblendvps xmm4, xmm3, xmm1, xmm5  // selected index
  vmaxps xmm0, xmm0, xmm2      // max values (low/high combined)

  // Reduce 4 -> 2 lanes
  vpshufd xmm2, xmm0, $55
  vpshufd xmm3, xmm4, $55
  vcmpps xmm5, xmm0, xmm2, 6
  vblendvps xmm4, xmm3, xmm4, xmm5
  vmaxss xmm0, xmm0, xmm2

  // Reduce 2 -> 1 lane
  vpshufd xmm2, xmm0, $AA
  vpshufd xmm3, xmm4, $AA
  vcmpps xmm5, xmm0, xmm2, 6
  vblendvps xmm4, xmm3, xmm4, xmm5
  vmaxss xmm0, xmm0, xmm2

  // Final reduction (1 lane)
  vpshufd xmm2, xmm0, $FF
  vpshufd xmm3, xmm4, $FF
  vcmpps xmm5, xmm0, xmm2, 6
  vblendvps xmm4, xmm3, xmm4, xmm5
  vmaxss xmm0, xmm0, xmm2

  // xmm0[0] = max, xmm4[0] = index
  vmovd [rsp+96], xmm4        // store index temporarily
  vzeroupper
  mov eax, [rsp+96]           // eax = bulk index (32-bit)

  // ---- Tail handling (0..7 elements) ----
  mov edx, [rsp+80]           // tail count (32-bit)
  test edx, edx
  jz @TailDone

  // r10 already points to start of tail (after bulk loop)
  // Base index for tail = bulk count (32-bit)
  mov edi, [rsp+72]           // edi = bulk count (32-bit)
  xor ecx, ecx                // ecx = tail index (32-bit)

  // Reload absolute mask into xmm2 (low 128-bit)
  vmovdqu xmm2, oword ptr [cAVXArgAbsMask]

@TailLoop:
  vmovss xmm1, [r10 + rcx*4]   // load element (64-bit pointer + 32-bit index scaled)
  vandps xmm1, xmm1, xmm2      // absolute value (mask now correct)
  comiss xmm1, xmm0
  jbe @TailSkip                // if <=, keep first occurrence
  movaps xmm0, xmm1            // update max
  lea eax, [edi + ecx]         // new position = bulk_count + tail_index
@TailSkip:
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@TailDone:
  mov [r9], eax                // store final position (r9 is 64-bit pointer)
  jmp @Exit

  // ---- Scalar fallback for N < 8 ----
@ScalarFallback:
  vmovdqu xmm2, oword ptr [cAVXArgAbsMask]   // 16-byte mask
  vmovss xmm0, [r10]           // first element
  vandps xmm0, xmm0, xmm2      // absolute
  xor eax, eax                 // index = 0
  mov ecx, 1
@ScalarLoop:
  cmp ecx, r11d
  jge @ScalarDone
  vmovss xmm1, [r10 + rcx*4]
  vandps xmm1, xmm1, xmm2      // absolute
  comiss xmm1, xmm0
  jbe @ScalarSkip              // if <=, keep first
  movaps xmm0, xmm1
  mov eax, ecx
@ScalarSkip:
  inc ecx
  jmp @ScalarLoop
@ScalarDone:
  mov [r9], eax
  jmp @Exit

@Zero:
  xorps xmm0, xmm0
  mov dword ptr [r9], 0

@Exit:
  vzeroupper
  // ---- Restore non-volatile YMM6, YMM7 ----
  vmovdqu ymm6, [rsp]
  vmovdqu ymm7, [rsp+32]
  add rsp, 64
  ret                         // result already in xmm0
end;

function _AVX512GetMaxAbsPos( PtrA : PSingle; NumElements : integer; out Position : integer ) : single; inline;
begin
  Result := _AVX2GetMaxAbsPos(PtrA, NumElements, Position);
end;

{-----------------------------------------------------------------------------
  AVX2 add scalar (32-bit): dst[i] := dst[i] + Value.
  Processes 32 elements per loop (4 YMM blocks), scalar tail.

  Win64 calling convention:
    RCX = PtrA  : PSingle
    XMM1 = Value : Single
    R8  = N     : Integer

  All registers used are volatile (R10, R11, RAX, RCX, RDX, R8, R9,
  YMM0-YMM4, XMM0-XMM4). No prologue/epilogue needed.
-----------------------------------------------------------------------------}
procedure _AVX2AddScalar(PtrA: PSingle; Value: Single; N: integer);
asm
  // ---- Load parameters into volatile registers ----
  mov r10, rcx                // r10 = PtrA (64-bit)
  mov r11d, r8d               // r11d = N (32-bit)

  test r11d, r11d
  jle @Exit

  // ---- Broadcast Value to all lanes of YMM0 ----
  vbroadcastss ymm0, xmm1     // ymm0 = {Value, Value, ...}

  // ---- Split bulk (multiple of 32) and tail (0..31) ----
  mov eax, r11d
  mov edx, eax
  and edx, 31                  // tail = N & 31
  sub eax, edx                 // eax = bulk (multiple of 32)
  jz @Tail

  mov ecx, eax
  shr ecx, 5                   // ecx = number of 32-element blocks
  jz @Tail

@BulkLoop:
  // Block 0: elements 0..7
  vmovups ymm1, [r10]
  vaddps ymm1, ymm1, ymm0
  vmovups [r10], ymm1

  // Block 1: elements 8..15
  vmovups ymm2, [r10+32]
  vaddps ymm2, ymm2, ymm0
  vmovups [r10+32], ymm2

  // Block 2: elements 16..23
  vmovups ymm3, [r10+64]
  vaddps ymm3, ymm3, ymm0
  vmovups [r10+64], ymm3

  // Block 3: elements 24..31
  vmovups ymm4, [r10+96]
  vaddps ymm4, ymm4, ymm0
  vmovups [r10+96], ymm4

  add r10, 128                 // advance by 32 elements (128 bytes)
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  // edx = tail count (0..31)
  test edx, edx
  jz @Exit

  xor ecx, ecx                 // index (32-bit)

  // xmm0 holds Value (low lane of ymm0)
@TailLoop:
  vmovss xmm1, [r10 + rcx*4]   // load dst[i]
  vaddss xmm1, xmm1, xmm0      // dst[i] + Value
  vmovss [r10 + rcx*4], xmm1   // store back
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
end;

procedure _AVX512AddScalar( PtrA : PSingle; Value : single; N : integer ); inline;
begin
  _AVX2AddScalar(PtrA, Value, N);
end;

{-----------------------------------------------------------------------------
  AVX2 exp-shift-sum: dst[i] = exp(src[i] - Shift), returns sum of dst.
  Bulk: 8 elements per loop (YMM). Tail: scalar.
  Win64 calling convention (actual):
    RCX = dst : PSingle
    RDX = src : PSingle
    XMM2 = Shift : Single
    R9  = N : Integer
  Returns sum in XMM0 (Single).
  Saves and restores non-volatile YMM6, YMM7.
-----------------------------------------------------------------------------}
function _AVX2ExpShiftSum(dst: PSingle; src: PSingle; Shift: Single; N: Integer): Single;
asm
  // ---- Save non-volatile YMM6, YMM7 ----
  sub rsp, 64
  vmovdqu [rsp], ymm6
  vmovdqu [rsp+32], ymm7

  // ---- Load parameters ----
  mov r10, rcx                // r10 = dst (64-bit)
  mov r11, rdx                // r11 = src  (64-bit)
  mov eax, r9d                // eax = N (32-bit, in R9)

  test eax, eax
  jle @Zero

  // ---- Split bulk (multiple of 8) and tail (0..7) ----
  mov edx, eax
  and edx, 7                  // tail = N & 7
  sub eax, edx                // eax = bulk count (multiple of 8)
  jz @Tail

  // ---- Broadcast Shift to ymm5 (Shift is in XMM2) ----
  vbroadcastss ymm5, xmm2     // ymm5 = {Shift, ...}

  // ---- Initialize accumulator ymm4 = 0 ----
  vxorps ymm4, ymm4, ymm4

  // ---- Bulk loop: process 8 elements per iteration ----
  mov ecx, eax
  shr ecx, 3                  // ecx = number of full blocks
  jz @TailBulkDone

@BulkLoop:
  // ---- Reload constants (overwritten each iteration) ----
  vbroadcastss ymm0, [rip + cAVXExpHi]
  vbroadcastss ymm1, [rip + cAVXExpLo]
  vbroadcastss ymm2, [rip + cAVXLog2e]
  vbroadcastss ymm3, [rip + cAVXLn2]
  vpbroadcastd ymm6, [rip + cAVXExp127]

  // ---- Load src, subtract Shift, clamp ----
  vmovups ymm7, [r11]          // src
  vsubps  ymm7, ymm7, ymm5     // src - Shift
  vminps  ymm7, ymm7, ymm0
  vmaxps  ymm7, ymm7, ymm1

  // ---- Compute t = x * log2e ----
  vmulps  ymm0, ymm7, ymm2
  vroundps ymm1, ymm0, 0
  vsubps  ymm0, ymm0, ymm1
  vmulps  ymm2, ymm0, ymm3     // g = f * ln2

  // ---- Polynomial 2^f (use vmulps/vaddps instead of FMA) ----
  vbroadcastss ymm3, [rip + cAVXExpP6]
  vbroadcastss ymm0, [rip + cAVXExpP5]
  vmulps  ymm3, ymm3, ymm2
  vaddps  ymm3, ymm3, ymm0
  vbroadcastss ymm0, [rip + cAVXExpP4]
  vmulps  ymm3, ymm3, ymm2
  vaddps  ymm3, ymm3, ymm0
  vbroadcastss ymm0, [rip + cAVXExpP3]
  vmulps  ymm3, ymm3, ymm2
  vaddps  ymm3, ymm3, ymm0
  vbroadcastss ymm0, [rip + cAVXExpP2]
  vmulps  ymm3, ymm3, ymm2
  vaddps  ymm3, ymm3, ymm0
  vbroadcastss ymm0, [rip + cAVXExpP1]
  vmulps  ymm3, ymm3, ymm2
  vaddps  ymm3, ymm3, ymm0
  vbroadcastss ymm0, [rip + cAVXExpP0]
  vmulps  ymm3, ymm3, ymm2
  vaddps  ymm3, ymm3, ymm0

  // ---- 2^k ----
  vcvtps2dq ymm1, ymm1
  vpaddd ymm1, ymm1, ymm6
  vpslld ymm1, ymm1, 23

  vmulps ymm7, ymm3, ymm1
  vmovups [r10], ymm7

  vaddps ymm4, ymm4, ymm7

  add r11, 32
  add r10, 32
  dec ecx
  jnz @BulkLoop

@TailBulkDone:
  // ---- Reduce bulk sum (ymm4) to scalar ----
  vextractf128 xmm0, ymm4, 1   // xmm0 = high 128 bits
  vaddps xmm0, xmm0, xmm4      // xmm0 = high + low (xmm4)
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  vzeroupper
  movss [rsp+72], xmm0          // store bulk sum

  // ---- Tail (0..7) scalar ----
@Tail:
  test edx, edx
  jz @Done

  // ---- Load scalar constants ----
  vbroadcastss xmm0, [rip + cAVXExpHi]
  vbroadcastss xmm1, [rip + cAVXExpLo]
  vbroadcastss xmm2, [rip + cAVXLog2e]
  vbroadcastss xmm3, [rip + cAVXLn2]
  vpbroadcastd xmm6, [rip + cAVXExp127]

  // Shift is in XMM5 (low lane of YMM5, broadcast from XMM2)
  // XMM5 still holds Shift.

  vxorps xmm7, xmm7, xmm7       // tail sum = 0
  xor ecx, ecx

@TailLoop:
  vmovss xmm4, [r11 + rcx*4]    // src[i]
  vsubss xmm4, xmm4, xmm5       // src - Shift
  vminss xmm4, xmm4, xmm0
  vmaxss xmm4, xmm4, xmm1

  vmulss xmm0, xmm4, xmm2       // t
  vroundss xmm1, xmm0, xmm0, 0  // k
  vsubss xmm0, xmm0, xmm1       // f
  vmulss xmm2, xmm0, xmm3       // g

  // ---- Scalar polynomial (use vmulss/vaddss instead of FMA) ----
  vmovss xmm3, [rip + cAVXExpP6]
  vmulss xmm3, xmm3, xmm2
  vaddss xmm3, xmm3, [rip + cAVXExpP5]
  vmulss xmm3, xmm3, xmm2
  vaddss xmm3, xmm3, [rip + cAVXExpP4]
  vmulss xmm3, xmm3, xmm2
  vaddss xmm3, xmm3, [rip + cAVXExpP3]
  vmulss xmm3, xmm3, xmm2
  vaddss xmm3, xmm3, [rip + cAVXExpP2]
  vmulss xmm3, xmm3, xmm2
  vaddss xmm3, xmm3, [rip + cAVXExpP1]
  vmulss xmm3, xmm3, xmm2
  vaddss xmm3, xmm3, [rip + cAVXExpP0]

  // ---- 2^k scalar ----
  vcvtss2si eax, xmm1
  add eax, 127
  shl eax, 23
  movd xmm1, eax

  vmulss xmm4, xmm3, xmm1
  vmovss [r10 + rcx*4], xmm4
  vaddss xmm7, xmm7, xmm4

  inc ecx
  cmp ecx, edx
  jl @TailLoop

  // ---- Combine bulk and tail sums ----
  movss xmm0, [rsp+72]
  vaddss xmm0, xmm0, xmm7
  jmp @Exit

@Done:
  movss xmm0, [rsp+72]
  jmp @Exit

@Zero:
  xorps xmm0, xmm0

@Exit:
  vzeroupper
  // ---- Restore non-volatile YMM6, YMM7 ----
  vmovdqu ymm6, [rsp]
  vmovdqu ymm7, [rsp+32]
  add rsp, 64
  ret
end;

function _AVX512ExpShiftSum( dst : PSingle; src : PSingle; Shift : single; N : integer ) : single;
begin
  Result := _AVX2ExpShiftSum(dst, src, Shift, N);
end;

{-----------------------------------------------------------------------------
  AVX2 natural logarithm (32-bit).
  Uses ymm0-ymm7, saves/restores non-volatile ymm6/ymm7.
  Win64 calling convention:
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : integer
  Algorithm: extract exponent and mantissa, scale mantissa to [0.5,1),
  use polynomial approximation for ln(m), then result = ln(m) + e*ln(2).
  Processes 8 floats per iteration. Tail scalar.
  Constants are reloaded each iteration to avoid register corruption.
-----------------------------------------------------------------------------}
procedure _AVX2Ln(dst: PSingle; src: PSingle; N: integer);
asm
  // ---- Save non-volatile YMM6 and YMM7 ----
  sub rsp, 64
  vmovups [rsp], ymm6
  vmovups [rsp+32], ymm7

  // ---- Load parameters ----
  mov r10, rcx                // r10 = dst
  mov r11, rdx                // r11 = src
  mov eax, r8d                // eax = N (32-bit)

  test eax, eax
  jle @Exit

  // ---- Split bulk (multiple of 8) and tail (0..7) ----
  mov edx, eax
  and edx, 7                  // tail = N & 7
  sub eax, edx                // bulk = N - tail
  jz @Tail

  mov ecx, eax
  shr ecx, 3                  // ecx = number of 8-element blocks
  jz @Tail

@BulkLoop:
  // ---- Reload constants (overwritten during loop) ----
  vbroadcastss ymm0, [rip + cAVXLnMinNorm]
  vpbroadcastd ymm1, [rip + cAVXExp127]
  vbroadcastss ymm2, [rip + cAVX8SSOne]
  vpbroadcastd ymm3, [rip + cAVXLnInvMant]
  vbroadcastss ymm4, [rip + cAVXLnHalf]
  vbroadcastss ymm5, [rip + cAVXLnSqrtHf]

  // ---- Load x, clamp to min norm ----
  vmovups ymm7, [r11]         // ymm7 = src[i..i+7]
  vmaxps ymm0, ymm7, ymm0     // ymm0 = max(x, MinNorm)

  // ---- Extract exponent ----
  vpsrld ymm6, ymm0, 23       // ymm6 = (bits >> 23)
  vpsubd ymm6, ymm6, ymm1     // e = e - 127
  vcvtdq2ps ymm6, ymm6
  vaddps ymm6, ymm6, ymm2     // e = (e - 127) + 1

  // ---- Extract mantissa ----
  vandps ymm7, ymm0, ymm3     // ymm7 = bits & invmant
  vorps  ymm7, ymm7, ymm4     // mantissa = (bits & invmant) | half

  // ---- Mask for sqrt(0.5) adjustment ----
  vcmpltps ymm0, ymm7, ymm5   // ymm0 = mask (mantissa < sqrt(0.5))

  // ---- Adjust mantissa: if mask, x = 2*x - 1 ----
  vandps ymm1, ymm7, ymm0
  vsubps ymm7, ymm7, ymm2
  vaddps ymm7, ymm7, ymm1

  // ---- Adjust exponent: if mask, e = e - 1 ----
  vandps ymm1, ymm2, ymm0
  vsubps ymm6, ymm6, ymm1

  // ---- z = x*x ----
  vmulps ymm1, ymm7, ymm7

  // ---- Polynomial: ln(m) = x * z * P(x) ----
  vmovaps ymm0, ymm7
  vmovups ymm3, [rip + cAVXLnP0]
  vfmadd213ps ymm3, ymm0, [rip + cAVXLnP1]
  vfmadd213ps ymm3, ymm0, [rip + cAVXLnP2]
  vfmadd213ps ymm3, ymm0, [rip + cAVXLnP3]
  vfmadd213ps ymm3, ymm0, [rip + cAVXLnP4]
  vfmadd213ps ymm3, ymm0, [rip + cAVXLnP5]
  vfmadd213ps ymm3, ymm0, [rip + cAVXLnP6]
  vfmadd213ps ymm3, ymm0, [rip + cAVXLnP7]
  vfmadd213ps ymm3, ymm0, [rip + cAVXLnP8]   // poly = P(x)

  vmulps ymm3, ymm3, ymm0     // poly * x
  vmulps ymm3, ymm3, ymm1     // poly * x * z

  // ---- Correction terms ----
  vfmadd231ps ymm3, ymm6, [rip + cAVXLnQ1]
  vmulps ymm4, ymm1, ymm4     // ymm4 = z * 0.5 (original half constant)
  vsubps ymm3, ymm3, ymm4
  vaddps ymm0, ymm0, ymm3
  vfmadd231ps ymm0, ymm6, [rip + cAVXLnQ2]

  // ---- Store result ----
  vmovups [r10], ymm0

  add r11, 32
  add r10, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  // ---- Tail handling (0..7 elements) ----
  test edx, edx
  jz @Exit

  xor ecx, ecx
@TailLoop:
  // ---- Load scalar constants ----
  vpbroadcastd xmm1, [rip + cAVXExp127]
  vpbroadcastd xmm3, [rip + cAVXLnInvMant]
  vbroadcastss xmm2, [rip + cAVX8SSOne]
  vbroadcastss xmm4, [rip + cAVXLnHalf]
  vbroadcastss xmm5, [rip + cAVXLnSqrtHf]
  vbroadcastss xmm6, [rip + cAVXLnMinNorm]   // xmm6 used as temp

  // ---- Load x, clamp ----
  vmovss xmm0, [r11 + rcx*4]
  vmaxss xmm0, xmm0, xmm6

  // ---- Extract exponent ----
  vpsrld xmm6, xmm0, 23
  vpsubd xmm6, xmm6, xmm1
  vcvtdq2ps xmm6, xmm6
  vaddss xmm6, xmm6, xmm2

  // ---- Extract mantissa ----
  vandps xmm7, xmm0, xmm3
  vorps  xmm7, xmm7, xmm4

  // ---- Adjust for sqrt(0.5) ----
  vcmpltss xmm0, xmm7, xmm5
  vandps xmm1, xmm7, xmm0
  vsubss xmm7, xmm7, xmm2
  vaddss xmm7, xmm7, xmm1

  vandps xmm1, xmm2, xmm0
  vsubss xmm6, xmm6, xmm1

  // ---- z = x*x ----
  vmulss xmm1, xmm7, xmm7

  // ---- Polynomial ----
  vmovaps xmm0, xmm7
  vmovss xmm3, [rip + cAVXLnP0]
  vfmadd213ss xmm3, xmm0, [rip + cAVXLnP1]
  vfmadd213ss xmm3, xmm0, [rip + cAVXLnP2]
  vfmadd213ss xmm3, xmm0, [rip + cAVXLnP3]
  vfmadd213ss xmm3, xmm0, [rip + cAVXLnP4]
  vfmadd213ss xmm3, xmm0, [rip + cAVXLnP5]
  vfmadd213ss xmm3, xmm0, [rip + cAVXLnP6]
  vfmadd213ss xmm3, xmm0, [rip + cAVXLnP7]
  vfmadd213ss xmm3, xmm0, [rip + cAVXLnP8]

  vmulss xmm3, xmm3, xmm0
  vmulss xmm3, xmm3, xmm1
  vfmadd231ss xmm3, xmm6, [rip + cAVXLnQ1]
  vmulss xmm4, xmm1, xmm4
  vsubss xmm3, xmm3, xmm4
  vaddss xmm0, xmm0, xmm3
  vfmadd231ss xmm0, xmm6, [rip + cAVXLnQ2]

  // ---- Store result ----
  vmovss [r10 + rcx*4], xmm0

  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
  // ---- Restore non-volatile YMM6 and YMM7 ----
  vmovups ymm6, [rsp]
  vmovups ymm7, [rsp+32]
  add rsp, 64
end;

procedure _AVX512Ln( dst : PSingle; src : PSingle; N : integer ); inline;
begin
  _AVX2Ln(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 sin/cos (32-bit): dst[i] = sin(src[i]) or cos(src[i]) for i=0..N-1.
  Win64 calling convention:
    RCX = dst : PSingle
    RDX = src : PSingle
    R8  = N   : integer
    R9  = DoCos : integer (0 = sin, non-zero = cos)
  Algorithm: range reduction q=round(x*2/pi), reduce r to [-pi/4, pi/4],
  compute sin_abs and cos_abs via degree-3 polynomials, reconstruct based on q.
  Uses only volatile registers (YMM0-YMM7), no non-volatile saving.
  vzeroupper before exit.

  Fixes:
  1. Range reduction: use abs_r - pi/2 instead of pi/2 - abs_r, then xor sign.
  2. Q adjustment in bulk loop: when r_new is negative, q must be incremented.
  3. vblendvps operand order for sin/cos reconstruction.
-----------------------------------------------------------------------------}
procedure _AVX2SinCos(dst: PSingle; src: PSingle; N: integer; DoCos: integer);
asm
  sub rsp, 128
  mov [rsp+4], r9d            // save DoCos

  mov r10, rcx                // dst
  mov r11, rdx                // src
  mov eax, r8d                // N

  test eax, eax
  jle @Exit

  mov edx, eax
  and edx, 7                  // tail
  mov [rsp+16], edx
  sub eax, edx
  mov ecx, eax
  shr ecx, 3
  mov [rsp+12], ecx
  jz @Tail

  vpbroadcastd ymm0, [rip + cOneInt]

@BulkLoop:
  vbroadcastss ymm4, [rip + cAVXSinCosInvPi2]
  vbroadcastss ymm5, [rip + cAVXSinCosPi2]
  vbroadcastss ymm6, [rip + cAVXSinCosPi4]
  vbroadcastss ymm7, [rip + cAVX8SSOne]

  // ---- Range reduction ----
  vmovups ymm0, [r11]
  vmulps  ymm1, ymm0, ymm4
  vroundps ymm1, ymm1, 0
  vcvtps2dq ymm1, ymm1
  vmovups [rsp+32], ymm1

  vcvtdq2ps ymm2, ymm1
  vmulps  ymm2, ymm2, ymm5
  vsubps  ymm0, ymm0, ymm2
  vmovups [rsp+64], ymm0

  // ---- Reduce r to [-pi/4, pi/4] ----
  vmovaps ymm3, ymm0
  vandps ymm0, ymm0, [rip + cAVXArgAbsMask]    // ymm0 = |r|
  vandps ymm3, ymm3, [rip + cAVXSinCosSignMask] // ymm3 = sign(r)
  vcmpgtps ymm2, ymm0, ymm6                     // ymm2 = |r| > pi/4 ?
  vsubps ymm0, ymm0, ymm5                       // ymm0 = |r| - pi/2  (fix)
  vxorps ymm0, ymm0, ymm3                       // ymm0 = sign(r)*(|r|-pi/2) = r_new
  vmovups ymm3, [rsp+64]                        // ymm3 = original r
  vblendvps ymm0, ymm3, ymm0, ymm2              // if |r|>pi/4 use r_new else original r

  // ---- Adjust q based on reduction ----
  vmovups ymm3, [rsp+32]                        // q_initial
  vpbroadcastd ymm1, [rip + cOneInt]
  vpaddd ymm4, ymm3, ymm1                       // q+1
  vpsubd ymm5, ymm3, ymm1                       // q-1
  vandps ymm6, ymm0, [rip + cAVXSinCosSignMask] // sign of r_new
  vpsrad ymm6, ymm6, 31                         // mask: negative -> all ones
  // when r_new is negative, we need q+1. vblendvps picks src2 when mask is set.
  vblendvps ymm4, ymm5, ymm4, ymm6              // src1=q-1, src2=q+1
  vblendvps ymm1, ymm3, ymm4, ymm2              // if reduced, use new q else old q

  // ---- Compute sin_abs and cos_abs ----
  vmulps ymm2, ymm0, ymm0      // z = r^2

  vbroadcastss ymm3, [rip + cAVXSinP3]
  vbroadcastss ymm4, [rip + cAVXSinP2]
  vfmadd213ps ymm3, ymm2, ymm4
  vbroadcastss ymm4, [rip + cAVXSinP1]
  vfmadd213ps ymm3, ymm2, ymm4
  vbroadcastss ymm4, [rip + cAVXSinP0]
  vfmadd213ps ymm3, ymm2, ymm4
  vmulps ymm3, ymm3, ymm0      // ymm3 = sin_abs

  vbroadcastss ymm4, [rip + cAVXCosQ2]
  vbroadcastss ymm5, [rip + cAVXCosQ1]
  vfmadd213ps ymm4, ymm2, ymm5
  vbroadcastss ymm5, [rip + cAVXCosQ0]
  vfmadd213ps ymm4, ymm2, ymm5
  vmulps ymm4, ymm4, ymm2
  vaddps ymm4, ymm4, ymm7      // ymm4 = cos_abs

  // ---- Reconstruct sin and cos from q_final (ymm1) ----
  vpbroadcastd ymm0, [rip + cOneInt]
  vpand ymm5, ymm1, ymm0       // q & 1
  vpsrld ymm6, ymm1, 1
  vpand ymm6, ymm6, ymm0       // (q>>1) & 1
  vpxor ymm7, ymm7, ymm7
  vpcmpeqd ymm7, ymm5, ymm7    // mask1 = (q&1)==0
  vpcmpeqd ymm6, ymm6, ymm0    // mask2 = ((q>>1)&1)==1

  // sin = (q&1==0) ? sin_abs : cos_abs, then flip sign if (q>>1)&1
  // vblendvps picks src2 when mask set. For mask1 set (q&1==0), choose sin_abs (ymm3).
  // So src1 = cos_abs (ymm4), src2 = sin_abs (ymm3).
  vblendvps ymm2, ymm4, ymm3, ymm7   // sin_candidate
  vxorps ymm5, ymm2, [rip + cAVXSinCosSignMask]
  vblendvps ymm0, ymm2, ymm5, ymm6   // ymm0 = sin

  // cos = (q&1==0) ? cos_abs : sin_abs, then flip sign if (q&1) XOR ((q>>1)&1)
  // For mask1 set (q&1==0), choose cos_abs (ymm4). So src1 = sin_abs (ymm3), src2 = cos_abs (ymm4).
  vblendvps ymm2, ymm3, ymm4, ymm7   // cos_candidate

  // Compute cos sign mask: (q&1) XOR ((q>>1)&1) == 1
  vpbroadcastd ymm3, [rip + cOneInt]
  vpand ymm5, ymm1, ymm3
  vpsrld ymm6, ymm1, 1
  vpand ymm6, ymm6, ymm3
  vpxor ymm7, ymm5, ymm6
  vpcmpeqd ymm7, ymm7, ymm3
  vxorps ymm5, ymm2, [rip + cAVXSinCosSignMask]
  vblendvps ymm1, ymm2, ymm5, ymm7   // ymm1 = cos

  // ---- Select sin or cos ----
  cmp dword ptr [rsp+4], 0
  jnz @DoCosSelect
  vmovaps ymm0, ymm0
  jmp @Store
@DoCosSelect:
  vmovaps ymm0, ymm1

@Store:
  vmovups [r10], ymm0

  add r11, 32
  add r10, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  mov edx, [rsp+16]
  test edx, edx
  jz @Exit

  xor ecx, ecx
  mov r8d, edx

  vbroadcastss xmm4, [rip + cAVXSinCosInvPi2]
  vbroadcastss xmm5, [rip + cAVXSinCosPi2]
  vbroadcastss xmm6, [rip + cAVXSinCosPi4]
  vbroadcastss xmm7, [rip + cAVX8SSOne]

@TailLoop:
  vmovss xmm0, [r11 + rcx*4]
  vmulss xmm1, xmm0, xmm4
  vroundss xmm1, xmm1, xmm1, 0
  vcvtss2si eax, xmm1
  vcvtsi2ss xmm2, xmm2, eax
  vmulss xmm2, xmm2, xmm5
  vsubss xmm0, xmm0, xmm2

  vandps xmm2, xmm0, [rip + cAVXSinCosSignMask]
  vandps xmm3, xmm0, [rip + cAVXArgAbsMask]
  vcomiss xmm3, xmm6
  jbe @NoReduce
    vsubss xmm3, xmm3, xmm5    // |r| - pi/2  (fix)
    vxorps xmm3, xmm3, xmm2    // r_new
    vmovss xmm0, xmm3, xmm3
    vpsrad xmm2, xmm2, 31
    vmovmskps edx, xmm2
    test edx, 1
    jnz @NegQ
      inc eax
      jmp @QAdjusted
    @NegQ:
      dec eax
    @QAdjusted:
  @NoReduce:

  vmulss xmm1, xmm0, xmm0

  vmovss xmm2, [rip + cAVXSinP3]
  vmovss xmm3, [rip + cAVXSinP2]
  vfmadd213ss xmm2, xmm1, xmm3
  vmovss xmm3, [rip + cAVXSinP1]
  vfmadd213ss xmm2, xmm1, xmm3
  vmovss xmm3, [rip + cAVXSinP0]
  vfmadd213ss xmm2, xmm1, xmm3
  vmulss xmm2, xmm2, xmm0

  vmovss xmm3, [rip + cAVXCosQ2]
  vmovss xmm4, [rip + cAVXCosQ1]
  vfmadd213ss xmm3, xmm1, xmm4
  vmovss xmm4, [rip + cAVXCosQ0]
  vfmadd213ss xmm3, xmm1, xmm4
  vmulss xmm3, xmm3, xmm1
  vaddss xmm3, xmm3, xmm7

  mov edx, eax
  and edx, 1
  shr eax, 1
  and eax, 1

  test edx, edx
  jnz @SinSelCos
    vmovaps xmm4, xmm2
    jmp @SinSign
@SinSelCos:
    vmovaps xmm4, xmm3
@SinSign:
  test eax, eax
  jz @SinNoFlip
    vxorps xmm4, xmm4, [rip + cAVXSinCosSignMask]
@SinNoFlip:

  test edx, edx
  jnz @CosSelSin
    vmovaps xmm5, xmm3
    jmp @CosSign
@CosSelSin:
    vmovaps xmm5, xmm2
@CosSign:
  xor edx, eax
  test edx, 1
  jz @CosNoFlip
    vxorps xmm5, xmm5, [rip + cAVXSinCosSignMask]
@CosNoFlip:

  cmp dword ptr [rsp+4], 0
  jnz @TailDoCos
  vmovaps xmm0, xmm4
  jmp @TailStore
@TailDoCos:
  vmovaps xmm0, xmm5
@TailStore:
  vmovss [r10 + rcx*4], xmm0

  inc ecx
  cmp ecx, r8d
  jl @TailLoop

@Exit:
  vzeroupper
  add rsp, 128
end;

procedure _AVX512SinCos( dst : PSingle; src : PSingle; N : integer; DoCos : integer ); inline;
begin
  _AVX2SinCos(dst, src, N, DoCos);
end;

{-----------------------------------------------------------------------------
  AVX2 conversion: single-precision float to bfloat16 (round-to-nearest-even).
  Win64 calling convention:
    RCX = dst : PSingle (actually points to Word array)
    RDX = src : PSingle
    R8  = N   : integer
  Processes 8 elements per loop (YMM), scalar tail.
  Uses integer operations exclusively; no MXCSR dependency.
  All registers used are volatile (YMM0-YMM5, XMM0-XMM5).
  vzeroupper called before exit.
-----------------------------------------------------------------------------}
procedure _AVX2EncodeBF16(dst: PSingle; src: PSingle; N: integer);
asm
  // ---- Load parameters ----
  mov r10, rcx                // r10 = dst (Word pointer)
  mov r11, rdx                // r11 = src
  mov eax, r8d                // eax = N

  test eax, eax
  jle @Exit

  // ---- Bulk: process 8 elements at a time ----
  mov ecx, eax
  shr ecx, 3                  // ecx = number of 8-element blocks
  jz @Tail

  // ---- Load constants into YMM registers ----
  // ymm0 = sign mask (0x7FFFFFFF)
  mov ebx, $7FFFFFFF
  vmovd xmm0, ebx
  vbroadcastss ymm0, xmm0

  // ymm1 = inf/NaN threshold (0x7F800000)
  mov ebx, $7F800000
  vmovd xmm1, ebx
  vbroadcastss ymm1, xmm1

  // ymm2 = round low bit (0x00000001)
  mov ebx, $00000001
  vmovd xmm2, ebx
  vbroadcastss ymm2, xmm2

  // ymm3 = half ULP (0x00007FFF)
  mov ebx, $00007FFF
  vmovd xmm3, ebx
  vbroadcastss ymm3, xmm3

  // ymm4 = quiet NaN bit (0x00000040)
  mov ebx, $00000040
  vmovd xmm4, ebx
  vbroadcastss ymm4, xmm4

@Loop8:
  vmovups   ymm5, [r11]        // ymm5 = src[0..7]

  // ---- Round to nearest even ----
  vpsrld    ymm6, ymm5, 16      // kept = bits >> 16
  vpand     ymm7, ymm6, ymm2    // kept & 1
  vpaddd    ymm7, ymm7, ymm3    // + half_ulp
  vpaddd    ymm7, ymm7, ymm5    // + original
  vpsrld    ymm7, ymm7, 16      // rounded = (original + half_ulp + (kept&1)) >> 16

  // ---- Check for NaN/Inf ----
  vpand     ymm5, ymm5, ymm0    // |bits|
  vpcmpgtd  ymm5, ymm5, ymm1    // mask = |bits| > 0x7F800000 ? (NaN/Inf)
  vpor      ymm6, ymm6, ymm4    // quiet NaN bit
  vpblendvb ymm5, ymm7, ymm6, ymm5   // blend: if NaN/Inf, use quiet NaN, else rounded

  // ---- Pack and store ----
  vpackusdw ymm5, ymm5, ymm5    // pack 32-bit to 16-bit (low 8 words)
  vextracti128 xmm6, ymm5, 1    // high 8 words
  vpunpcklqdq xmm5, xmm5, xmm6  // combine into 16-byte (8 words)
  vmovups [r10], xmm5           // store 8 words

  add r11, 32                   // src += 8*4
  add r10, 16                   // dst += 8*2
  dec ecx
  jnz @Loop8

@Tail:
  // ---- Tail: 0..7 elements (scalar) ----
  mov ecx, eax
  and ecx, 7
  jz @Exit

  // ---- Load scalar constants ----
  // xmm0 = sign mask
  mov ebx, $7FFFFFFF
  vmovd xmm0, ebx
  vbroadcastss xmm0, xmm0

  // xmm1 = inf/NaN threshold
  mov ebx, $7F800000
  vmovd xmm1, ebx
  vbroadcastss xmm1, xmm1

  // xmm2 = round low bit
  mov ebx, $00000001
  vmovd xmm2, ebx
  vbroadcastss xmm2, xmm2

  // xmm3 = half ULP
  mov ebx, $00007FFF
  vmovd xmm3, ebx
  vbroadcastss xmm3, xmm3

  // xmm4 = quiet NaN bit
  mov ebx, $00000040
  vmovd xmm4, ebx
  vbroadcastss xmm4, xmm4

  xor edx, edx                 // index

@ScalarLoop:
  vmovss xmm5, [r11 + rdx*4]   // load float

  // Round to nearest even
  vpsrld xmm6, xmm5, 16        // kept
  vpand  xmm7, xmm6, xmm2      // kept & 1
  vpaddd xmm7, xmm7, xmm3      // + half_ulp
  vpaddd xmm7, xmm7, xmm5      // + original
  vpsrld xmm7, xmm7, 16        // rounded

  // Check NaN/Inf
  vpand  xmm5, xmm5, xmm0      // |bits|
  vpcmpgtd xmm5, xmm5, xmm1    // mask
  vpor   xmm6, xmm6, xmm4      // quiet NaN
  vpblendvb xmm5, xmm7, xmm6, xmm5  // blend

  vmovd eax, xmm5              // extract 32-bit
  mov [r10 + rdx*2], ax        // store 16-bit word

  inc edx
  cmp edx, ecx
  jl @ScalarLoop

@Exit:
  vzeroupper
end;

procedure _AVX512EncodeBF16( dst: PSingle; src : PSingle; N : integer ); inline;
begin
  _AVX2EncodeBF16(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 leaky clamp (ReLU-like) for 64-bit Delphi.
  For each x in src:
    if x > HighLimit then
      dst = HighLimit + (x - HighLimit) * Slope
    else if x > LowLimit then
      dst = x
    else
      dst = LowLimit + (x - LowLimit) * Slope

  Win64 calling convention (observed):
    RCX = dst
    RDX = src
    XMM2 = LowLimit
    XMM3 = HighLimit
    [RBP+48] = Slope (Single)
    [RBP+56] = N (integer)

  Saves/restores non-volatile YMM6/YMM7.
-----------------------------------------------------------------------------}
procedure _AVX2ReluL(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer);
asm
  sub rsp, 64
  vmovups [rsp], ymm6
  vmovups [rsp+32], ymm7

  mov r10, rcx
  mov r11, rdx
  mov eax, [rbp+56]                        // eax = N (from stack)
  test eax, eax
  jle @Exit

  // Broadcast HighLimit FIRST (from XMM3), then LowLimit (from XMM2).
  // Order matters: broadcasting LowLimit into YMM3 would overwrite XMM3.
  vbroadcastss ymm4, xmm3                  // ymm4 = HighLimit
  vbroadcastss ymm3, xmm2                  // ymm3 = LowLimit
  vbroadcastss ymm2, dword ptr [rbp+48]    // ymm2 = Slope

  mov edx, eax
  and edx, 7                               // tail count (0..7)
  sub eax, edx                             // bulk count (multiple of 8)
  mov ecx, eax
  shr ecx, 3                               // number of full blocks
  jz @Tail

@Loop8:
  vmovups ymm0, [r11]                      // x

  // High branch: HL + (x - HL) * S
  vsubps ymm1, ymm0, ymm4
  vmulps ymm1, ymm1, ymm2
  vaddps ymm1, ymm1, ymm4

  // Low branch: LL + (x - LL) * S
  vsubps ymm5, ymm0, ymm3
  vmulps ymm5, ymm5, ymm2
  vaddps ymm5, ymm5, ymm3

  // if x > LowLimit, use x
  vcmpltps ymm6, ymm3, ymm0                // ymm6 = (LowLimit < x)
  vblendvps ymm5, ymm5, ymm0, ymm6

  // if x > HighLimit, use high branch
  vcmpltps ymm7, ymm4, ymm0                // ymm7 = (HighLimit < x)
  vblendvps ymm5, ymm5, ymm1, ymm7

  vmovups [r10], ymm5
  add r11, 32
  add r10, 32
  dec ecx
  jnz @Loop8

@Tail:
  test edx, edx
  jz @Exit

  xor ecx, ecx
@ScalarLoop:
  vmovss xmm0, [r11 + rcx*4]

  // High branch
  vsubss xmm6, xmm0, xmm4
  vmulss xmm6, xmm6, xmm2
  vaddss xmm6, xmm6, xmm4

  // Low branch
  vsubss xmm1, xmm0, xmm3
  vmulss xmm1, xmm1, xmm2
  vaddss xmm1, xmm1, xmm3

  // Select
  vcmpltss xmm7, xmm3, xmm0                // LowLimit < x
  vblendvps xmm1, xmm1, xmm0, xmm7
  vcmpltss xmm7, xmm4, xmm0                // HighLimit < x
  vblendvps xmm1, xmm1, xmm6, xmm7

  vmovss [r10 + rcx*4], xmm1
  inc ecx
  cmp ecx, edx
  jl @ScalarLoop

@Exit:
  vzeroupper
  vmovups ymm6, [rsp]
  vmovups ymm7, [rsp+32]
  add rsp, 64
end;

procedure _AVX512ReluL(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;
begin
  _AVX2ReluL(dst, src, LowLimit, HighLimit, Slope, N);
end;

{-----------------------------------------------------------------------------
  AVX + F16C conversion: single-precision float to half-precision (binary16).
  Uses vcvtps2ph with round-to-nearest-even (imm8=0).

  Win64 calling convention:
    RCX = dst : Pointer      (points to Word array for half-precision output)
    RDX = src : PSingle
    R8  = N   : integer

  Exception and MXCSR notes:
    - The narrowing conversion may raise #O or #I for overflow/NaN inputs.
    - This function does NOT modify MXCSR; caller must mask exceptions if needed.
    - Uses default round-to-nearest-even mode.

  Uses only YMM registers for vcvtps2ph (no XMM source).
  vzeroupper on exit.
-----------------------------------------------------------------------------}
procedure _AVX2EncodeF16(dst: Pointer; src: Pointer; N: integer);
asm
  // ---- Load parameters ----
  mov r10, rcx                // r10 = dst (Word pointer)
  mov r11, rdx                // r11 = src
  mov eax, r8d                // eax = N

  test eax, eax
  jle @Exit

  // ---- Allocate 16 bytes for scalar conversion temporary (aligned) ----
  sub rsp, 16                 // 16-byte stack buffer

  // ---- Split bulk (multiple of 32) and tail (0..31) ----
  mov r8d, eax                // r8d = N
  and r8d, 31                 // tail = N & 31
  sub eax, r8d                // bulk = N - tail
  mov ecx, eax
  shr ecx, 5                  // ecx = number of 32-element chunks
  jz @Tail

@LargeLoop:
  vmovups ymm0, [r11]         // load 8 floats
  vmovups ymm1, [r11+32]
  vmovups ymm2, [r11+64]
  vmovups ymm3, [r11+96]

  vcvtps2ph [r10], ymm0, 0    // store 8 halfs (16 bytes)
  vcvtps2ph [r10+16], ymm1, 0
  vcvtps2ph [r10+32], ymm2, 0
  vcvtps2ph [r10+48], ymm3, 0

  add r11, 128                // src += 32*4
  add r10, 64                 // dst += 32*2
  dec ecx
  jnz @LargeLoop

@Tail:
  // ---- Tail: 0..31 elements ----
  mov eax, r8d                // eax = remaining count
  test eax, eax
  jz @Cleanup

  // ---- Process 8-element chunks ----
  mov ecx, eax
  shr ecx, 3                  // number of 8-element chunks
  jz @ScalarTail

@SmallLoop:
  vmovups ymm0, [r11]
  vcvtps2ph [r10], ymm0, 0
  add r11, 32
  add r10, 16
  dec ecx
  jnz @SmallLoop

@ScalarTail:
  // ---- Scalar 0..7 elements (use YMM broadcast) ----
  and eax, 7                  // remaining count
  jz @Cleanup

  xor ecx, ecx                // index
@ScalarLoop:
  vpbroadcastd ymm0, [r11 + rcx*4]   // load float and broadcast to all 8 lanes
  vcvtps2ph [rsp], ymm0, 0            // convert all 8 lanes, write 16 bytes to [rsp]
  mov ax, [rsp]                       // read low 16 bits (first half)
  mov [r10 + rcx*2], ax               // store half to destination

  inc ecx
  cmp ecx, eax
  jl @ScalarLoop

@Cleanup:
  // ---- Free temporary stack ----
  add rsp, 16

@Exit:
  vzeroupper
end;

procedure _AVX512EncodeF16(dst, src: Pointer; N: integer); inline;
begin
  _AVX2EncodeF16(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 gate mask: output 1 if LowLimit < x <= HighLimit, else Slope.
  For each x in src:
    if (x > LowLimit) and not (x > HighLimit) then dst = 1.0 else dst = Slope.

  Win64 calling convention (observed):
    RCX = dst
    RDX = src
    XMM2 = LowLimit
    XMM3 = HighLimit
    [RBP+48] = Slope (Single)
    [RBP+56] = N (integer)

  Uses only volatile YMM registers (YMM0-YMM5), no non-volatile saving.
-----------------------------------------------------------------------------}
procedure _AVX2ReluLGateMask(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer);
asm
  // ---- Load parameters ----
  mov r10, rcx                // r10 = dst
  mov r11, rdx                // r11 = src
  mov eax, [rbp+56]           // eax = N (from stack)
  test eax, eax
  jle @Exit

  // ---- Broadcast constants ----
  // Broadcast HighLimit BEFORE LowLimit: writing ymm3 would overwrite XMM3,
  // which is the source of HighLimit.
  vbroadcastss ymm4, xmm3                        // ymm4 = HighLimit
  vbroadcastss ymm3, xmm2                        // ymm3 = LowLimit
  vbroadcastss ymm2, dword ptr [rbp+48]          // ymm2 = Slope
  vbroadcastss ymm5, [rip + cAVX8SSOne]          // ymm5 = 1.0
  // Now: xmm2=Slope, xmm3=LowLimit, xmm4=HighLimit, xmm5=1.0

  // ---- Split bulk (multiple of 8) and tail (0..7) ----
  mov edx, eax
  and edx, 7                  // tail = N & 7
  sub eax, edx                // bulk = N - tail
  mov ecx, eax
  shr ecx, 3                  // number of 8-element blocks
  jz @Tail

@Loop8:
  vmovups ymm0, [r11]          // x
  vcmpltps ymm1, ymm3, ymm0    // mask1 = (LowLimit < x)
  vcmpltps ymm0, ymm4, ymm0    // mask2 = (HighLimit < x)
  vandnps ymm1, ymm0, ymm1     // inside = mask1 AND NOT mask2
  vblendvps ymm0, ymm2, ymm5, ymm1   // inside ? 1.0 : Slope
  vmovups [r10], ymm0
  add r11, 32
  add r10, 32
  dec ecx
  jnz @Loop8

@Tail:
  test edx, edx
  jz @Exit

  xor ecx, ecx
@ScalarLoop:
  vmovss xmm0, [r11 + rcx*4]   // x
  vcmpltss xmm1, xmm3, xmm0    // mask1 = (LowLimit < x)
  vcmpltss xmm0, xmm4, xmm0    // mask2 = (HighLimit < x); x no longer needed
  vandnps xmm1, xmm0, xmm1     // inside
  vblendvps xmm0, xmm2, xmm5, xmm1   // inside ? 1.0 : Slope
  vmovss [r10 + rcx*4], xmm0
  inc ecx
  cmp ecx, edx
  jl @ScalarLoop

@Exit:
  vzeroupper
end;

procedure _AVX512ReluLGateMask(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;
begin
  _AVX2ReluLGateMask(dst, src, LowLimit, HighLimit, Slope, N);
end;


end.
