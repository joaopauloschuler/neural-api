unit neuralavx32w;

interface

{$include neuralnetwork.inc}

{$IFDEF FPC}
   This unit is only for Delphi
{$ENDIF}

{$IFNDEF WIN32}
   This unit is only for Win32
{$ENDIF}

// AVX-512 is unavailable in 32-bit mode. To maintain code consistency,
// AVX-512 calls are still provided but simply forwarded to AVX2.


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
  Optimized for Win32: processes 16 elements per iteration using 2 YMM blocks
  due to register constraints (ymm0-ymm7 only).
  Parameters: EAX=dst, EDX=src, ECX=N, fact at [EBP+8] (stack parameter).
  Uses FMA instructions (requires AVX2+FMA capable CPU).
-----------------------------------------------------------------------------}
procedure _AVX2MulAddF( dst : PSingle; src : PSingle; N : Integer; const fact : Single);
asm
  push esi
  push edi
  push ebx

  mov edi, eax            // dst pointer
  mov esi, edx            // src pointer
  mov ecx, ecx            // N (already in ECX)
  test ecx, ecx
  jle @Exit

  // Broadcast factor to all lanes
  vbroadcastss ymm0, [ebp+8]   // ymm0 = fact (fact is the 4th parameter, at [EBP+8])

  // Compute bulk count (multiple of 16) and tail (0..15)
  mov eax, ecx
  and eax, 15             // tail = N mod 16
  mov ebx, eax            // save tail
  sub ecx, eax            // bulk = N - tail
  mov eax, ecx
  shr eax, 4              // number of 16-element chunks
  jz @Tail

  // ---- Main loop: 16 elements per iteration (2 YMM blocks) ----
@BulkLoop:
  // Block 0: elements 0..7
  vmovups ymm1, [esi]       // src
  vmovups ymm2, [edi]       // dst
  vfmadd231ps ymm2, ymm1, ymm0   // dst = dst + src * fact
  vmovups [edi], ymm2

  // Block 1: elements 8..15
  vmovups ymm3, [esi+32]
  vmovups ymm4, [edi+32]
  vfmadd231ps ymm4, ymm3, ymm0
  vmovups [edi+32], ymm4

  add edi, 64             // advance by 16*4 = 64 bytes
  add esi, 64
  dec eax
  jnz @BulkLoop

  vzeroupper

@Tail:
  mov ecx, ebx            // restore tail count (0..15)
  test ecx, ecx
  jz @Exit

  // Process 8-element chunks if tail >= 8
  cmp ecx, 8
  jl @ScalarTail

  vmovups ymm1, [esi]
  vmovups ymm2, [edi]
  vfmadd231ps ymm2, ymm1, ymm0
  vmovups [edi], ymm2
  add edi, 32
  add esi, 32
  sub ecx, 8
  vzeroupper

@ScalarTail:
  // Last 0..7 elements handled one by one
  test ecx, ecx
  jz @Exit
  xor eax, eax
@ScalarLoop:
  vmovss xmm1, [esi + eax*4]
  vmovss xmm2, [edi + eax*4]
  vfmadd231ss xmm2, xmm1, xmm0   // scalar FMA
  vmovss [edi + eax*4], xmm2
  inc eax
  cmp eax, ecx
  jl @ScalarLoop

@Exit:
  vzeroupper
  pop ebx
  pop edi
  pop esi
end;

procedure _AVX512MulAddF( dst : PSingle; src : PSingle; N : Integer; const fact : Single); inline;
begin
  _AVX2MulAddF(dst, src, N, fact);
end;

{-----------------------------------------------------------------------------
  AVX2 memory fill: dst[i] = fact for i = 0..N-1.
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Reverse traversal for compatibility with original CAI code.
  Parameters (register): EAX=dst, EDX=N.
-----------------------------------------------------------------------------}
procedure _AVX2FillMem( dst : PSingle; N : Integer; const fact : Single);
asm
  // Load factor and broadcast to all lanes
  lea ecx, fact                // ECX = address of factor
  vbroadcastss ymm0, [ecx]     // ymm0 = {fact, fact, ...}

  // Prepare reverse traversal: N = -N (in elements), adjust pointer
  imul edx, -4                 // EDX = -N*4 (byte offset)
  sub eax, edx                 // dst points to end of array

  // Main loop: process 32 elements (4 YMM blocks) per iteration
@Loop1:
  add edx, 128                 // Move forward by 32 elements (128 bytes)
  jg @loopEnd1                 // If EDX > 0, we've passed the start, exit

  // Store 4 YMM blocks (32 elements)
  vmovups [eax + edx - 128], ymm0
  vmovups [eax + edx - 96],  ymm0
  vmovups [eax + edx - 64],  ymm0
  vmovups [eax + edx - 32],  ymm0
  jmp @Loop1

@loopEnd1:
  sub edx, 128                 // Restore EDX to actual remaining elements
  jz @loop3End                 // If exactly multiple of 32, skip tail

  // Process remaining groups of 4 elements using XMM
@Loop2:
  add edx, 16                  // Move forward by 4 elements (16 bytes)
  jg @Loop2End                 // If EDX > 0, we've passed the start

  vmovups [eax + edx - 16], xmm0
  jmp @Loop2

@Loop2End:
  sub edx, 16                  // Restore EDX to remaining scalar count
  jz @loop3End

  // Process last 0..3 elements using scalar operations
@loop3:
  add edx, 4                   // Move forward by 1 element (4 bytes)
  jg @loop3End

  vmovss [eax + edx - 4], xmm0
  jmp @loop3

@loop3End:
  vzeroupper                   // Clear upper YMM state
end;

procedure _AVX512FillMem( dst : PSingle; N : Integer; const fact : Single); inline;
begin
  _AVX2FillMem(dst, N, fact);
end;

{-----------------------------------------------------------------------------
  AVX2 scalar multiply-multiply-add:
    dst[i] = dst[i] * mulOp1 + src[i] * mulOp2.

  Optimized for Win32: processes 16 elements per iteration using 2 YMM blocks.
  Uses FMA for accuracy and performance.

  Parameters (register convention):
    EAX = dst : PSingle
    EDX = src : PSingle
    ECX = N   : Integer
    [EBP+12]  = mulOp1 : Single (const, stack parameter)
    [EBP+8] = mulOp2 : Single (const, stack parameter)

  vzeroupper is called before exit to avoid AVX-SSE transition penalties.
-----------------------------------------------------------------------------}
procedure _AVX2MulMulAdd( dst : PSingle; src : PSingle; N : Integer;
  const mulOp1, mulOp2 : Single);
asm
  push esi
  push edi
  push ebx

  mov edi, eax            // dst pointer
  mov esi, edx            // src pointer
  mov ecx, ecx            // N (already in ECX)
  test ecx, ecx
  jle @Exit

  vbroadcastss ymm0, [ebp+12]   // ymm0 = mulOp1
  vbroadcastss ymm1, [ebp+8]  // ymm1 = mulOp2

  // Compute bulk count (multiple of 16) and tail (0..15)
  mov eax, ecx
  and eax, 15             // tail = N mod 16
  mov ebx, eax            // save tail
  sub ecx, eax            // bulk = N - tail
  mov eax, ecx
  shr eax, 4              // number of 16-element chunks
  jz @Tail

  // ---- Main loop: 16 elements per iteration (2 YMM blocks) ----
@BulkLoop:
  // Block 0: elements 0..7
  vmovups ymm2, [edi]       // load dst[0..7]
  vmovups ymm3, [esi]       // load src[0..7]
  vmulps  ymm2, ymm2, ymm0  // dst * mulOp1
  vfmadd231ps ymm2, ymm3, ymm1   // + src * mulOp2
  vmovups [edi], ymm2       // store back

  // Block 1: elements 8..15
  vmovups ymm4, [edi+32]
  vmovups ymm5, [esi+32]
  vmulps  ymm4, ymm4, ymm0
  vfmadd231ps ymm4, ymm5, ymm1
  vmovups [edi+32], ymm4

  add edi, 64             // advance by 16*4 = 64 bytes
  add esi, 64
  dec eax
  jnz @BulkLoop

  vzeroupper

@Tail:
  mov ecx, ebx            // restore tail count (0..15)
  test ecx, ecx
  jz @Exit

  // If tail >= 8, process one 8-element chunk
  cmp ecx, 8
  jl @ScalarTail

  vmovups ymm2, [edi]
  vmovups ymm3, [esi]
  vmulps  ymm2, ymm2, ymm0
  vfmadd231ps ymm2, ymm3, ymm1
  vmovups [edi], ymm2
  add edi, 32
  add esi, 32
  sub ecx, 8
  vzeroupper

@ScalarTail:
  // Last 0..7 elements handled one by one
  test ecx, ecx
  jz @Exit
  xor eax, eax
@ScalarLoop:
  vmovss xmm2, [edi + eax*4]
  vmovss xmm3, [esi + eax*4]
  vmulss xmm2, xmm2, xmm0
  vfmadd231ss xmm2, xmm3, xmm1
  vmovss [edi + eax*4], xmm2
  inc eax
  cmp eax, ecx
  jl @ScalarLoop

@Exit:
  vzeroupper
  pop ebx
  pop edi
  pop esi
end;

procedure _AVX512MulMulAdd( dst : PSingle; src : PSingle; N : Integer; const mulOp1, mulOp2 : Single); inline;
begin
  _AVX2MulMulAdd(dst, src, N, mulOp1, mulOp2);
end;

{-----------------------------------------------------------------------------
  AVX2 triadic multiply-add: dst[i] = dst[i] + src[i] * z[i].
  Optimized for Win32: processes 32 elements per iteration using 4 YMM blocks.

  Parameters (register convention):
    EAX = dst : PSingle
    EDX = src : PSingle
    ECX = z   : PSingle
    [EBP+8] = N : Integer

  The compiler automatically generates the standard stack frame
  (push ebp; mov ebp, esp) because the function accesses a stack parameter.
  Do NOT write push ebp / pop ebp / ret manually inside the asm block.

  Registers saved/restored: ESI, EDI, EBX (callee-saved).
  vzeroupper is called before exit to avoid AVX-SSE transition penalties.
-----------------------------------------------------------------------------}
procedure _AVX2MulAdd( dst : PSingle; src : PSingle; z : PSingle; N : Integer);
asm
  push esi
  push edi
  push ebx

  mov edi, eax            // dst pointer
  mov esi, edx            // src pointer
  mov edx, ecx            // z pointer
  mov ecx, [ebp+8]        // N (stack parameter)

  test ecx, ecx
  jle @Exit

  // Compute bulk count (multiple of 32) and tail (0..31)
  mov eax, ecx
  and eax, 31             // tail = N mod 32
  mov ebx, eax            // save tail
  sub ecx, eax            // bulk = N - tail
  mov eax, ecx
  shr eax, 5              // number of 32-element chunks
  jz @Tail

  // ---- Main loop: 32 elements per iteration (4 YMM blocks) ----
@BulkLoop:
  // Block 0: elements 0..7
  vmovups ymm0, [esi]
  vmulps  ymm0, ymm0, [edx]
  vaddps  ymm0, ymm0, [edi]
  vmovups [edi], ymm0

  // Block 1: elements 8..15
  vmovups ymm1, [esi+32]
  vmulps  ymm1, ymm1, [edx+32]
  vaddps  ymm1, ymm1, [edi+32]
  vmovups [edi+32], ymm1

  // Block 2: elements 16..23
  vmovups ymm2, [esi+64]
  vmulps  ymm2, ymm2, [edx+64]
  vaddps  ymm2, ymm2, [edi+64]
  vmovups [edi+64], ymm2

  // Block 3: elements 24..31
  vmovups ymm3, [esi+96]
  vmulps  ymm3, ymm3, [edx+96]
  vaddps  ymm3, ymm3, [edi+96]
  vmovups [edi+96], ymm3

  add edi, 128            // advance dst by 32*4 = 128 bytes
  add esi, 128
  add edx, 128
  dec eax
  jnz @BulkLoop

  vzeroupper

@Tail:
  mov ecx, ebx            // restore tail count (0..31)
  test ecx, ecx
  jz @Exit

  // Process remaining elements in 8-element chunks
  mov eax, ecx
  and eax, 7              // leftover < 8
  sub ecx, eax            // multiple of 8
  shr ecx, 3              // number of 8-element chunks
  jz @ScalarTail

@SmallLoop:
  vmovups ymm0, [esi]
  vmulps  ymm0, ymm0, [edx]
  vaddps  ymm0, ymm0, [edi]
  vmovups [edi], ymm0
  add edi, 32
  add esi, 32
  add edx, 32
  dec ecx
  jnz @SmallLoop

  vzeroupper

@ScalarTail:
  // Last 0..7 elements processed one by one
  mov ecx, eax            // remaining count
  test ecx, ecx
  jz @Exit
  xor eax, eax
@ScalarLoop:
  vmovss xmm0, [esi + eax*4]
  vmulss xmm0, xmm0, [edx + eax*4]
  vaddss xmm0, xmm0, [edi + eax*4]
  vmovss [edi + eax*4], xmm0
  inc eax
  cmp eax, ecx
  jl @ScalarLoop

@Exit:
  vzeroupper
  pop ebx
  pop edi
  pop esi
  // Compiler automatically emits pop ebp and ret; do not add them manually.
end;

procedure _AVX512MulAdd( dst : PSingle; src : PSingle; z : PSingle; N : Integer); inline;
begin
  _AVX2MulAdd(dst, src, z, N);
end;

{-----------------------------------------------------------------------------
  AVX2 ReLU copy: dst[i] = max(0, src[i]).
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Win32 register calling convention:
    EAX = dst, EDX = src, ECX = N.
  ReLU semantics match: if x > 0 then x else 0.
  NaN and -0.0 both map to 0.
-----------------------------------------------------------------------------}
procedure _AVX2CopyRelu(dst: PSingle; src: PSingle; N: Integer);
asm
  // ecx = -4 * N for reverse traversal
  neg ecx
  shl ecx, 2

  // Adjust pointers to end of arrays
  sub eax, ecx
  sub edx, ecx

  vxorps ymm5, ymm5, ymm5      // ymm5 = 0

@Loop1:
  add ecx, 128
  jg @loopEnd1

  vmovups ymm0, [edx + ecx - 128]
  vmovups ymm1, [edx + ecx - 96]
  vmovups ymm2, [edx + ecx - 64]
  vmovups ymm3, [edx + ecx - 32]

  // ReLU: if x > 0 then x else 0
  // mask = (0 < x); blend picks x when mask set, else 0.
  vcmpltps ymm4, ymm5, ymm0
  vblendvps ymm0, ymm5, ymm0, ymm4

  vcmpltps ymm4, ymm5, ymm1
  vblendvps ymm1, ymm5, ymm1, ymm4

  vcmpltps ymm4, ymm5, ymm2
  vblendvps ymm2, ymm5, ymm2, ymm4

  vcmpltps ymm4, ymm5, ymm3
  vblendvps ymm3, ymm5, ymm3, ymm4

  vmovups [eax + ecx - 128], ymm0
  vmovups [eax + ecx - 96],  ymm1
  vmovups [eax + ecx - 64],  ymm2
  vmovups [eax + ecx - 32],  ymm3

  jmp @Loop1

@loopEnd1:
  sub ecx, 128
  jz @loop3End

@Loop2:
  add ecx, 16
  jg @Loop2End

  vmovups xmm0, [edx + ecx - 16]
  vcmpltps xmm4, xmm5, xmm0
  vblendvps xmm0, xmm5, xmm0, xmm4
  vmovups [eax + ecx - 16], xmm0
  jmp @Loop2

@Loop2End:
  sub ecx, 16
  jz @loop3End

@loop3:
  add ecx, 4
  jg @loop3End

  vmovss xmm0, [edx + ecx - 4]
  vcmpltss xmm4, xmm5, xmm0
  vblendvps xmm0, xmm5, xmm0, xmm4
  vmovss [eax + ecx - 4], xmm0
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
  Parameters (register): EAX=dst, EDX=N.
-----------------------------------------------------------------------------}
procedure _AVX2MulF( dst : PSingle; N : Integer; const factor : Single);
asm
  push esi                     // Save ESI

  // Load factor and broadcast to all lanes
  lea esi, factor
  vbroadcastss ymm7, [esi]

  // Reverse traversal: N = -N
  mov ecx, N
  imul ecx, -4
  sub eax, ecx

  // Main loop: 32 elements (4 YMM blocks)
@Loop1:
  add ecx, 128
  jg @loopEnd1

  vmulps ymm2, ymm7, [eax + ecx - 128]
  vmulps ymm3, ymm7, [eax + ecx - 96]
  vmulps ymm4, ymm7, [eax + ecx - 64]
  vmulps ymm5, ymm7, [eax + ecx - 32]

  vmovups [eax + ecx - 128], ymm2
  vmovups [eax + ecx - 96],  ymm3
  vmovups [eax + ecx - 64],  ymm4
  vmovups [eax + ecx - 32],  ymm5

  jmp @Loop1

@loopEnd1:
  sub ecx, 128
  jz @loop3End

  // Process 4-element groups (XMM)
@Loop2:
  add ecx, 16
  jg @Loop2End

  vmovups xmm2, [eax + ecx - 16]
  vmulps xmm2, xmm2, xmm7
  vmovups [eax + ecx - 16], xmm2
  jmp @Loop2

@Loop2End:
  sub ecx, 16
  jz @loop3End

  // Handle last 0..3 elements (scalar)
@loop3:
  add ecx, 4
  jg @loop3End

  vmovss xmm2, [eax + ecx - 4]
  vmulss xmm2, xmm2, xmm7
  vmovss [eax + ecx - 4], xmm2
  jmp @loop3

@loop3End:
  vzeroupper
  pop esi
end;

procedure _AVX512MulF( dst : PSingle; N : Integer; const factor : Single); inline;
begin
  _AVX2MulF(dst, N, factor);
end;

{-----------------------------------------------------------------------------
  AVX2 elementwise multiplication: dst[i] = dst[i] * src[i].
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters (register): EAX=dst, EDX=src, ECX=N.
-----------------------------------------------------------------------------}
procedure _AVX2Mul( dst : PSingle; src : PSingle; N : Integer);
asm
  imul ecx, -4                 // N := -N for reverse traversal
  sub eax, ecx                 // Adjust dst pointer
  sub edx, ecx                 // Adjust src pointer

  // Main loop: 32 elements (4 YMM blocks)
@Loop1:
  add ecx, 128
  jg @loopEnd1

  // Load src values
  vmovups ymm0, [edx + ecx - 128]
  vmovups ymm1, [edx + ecx - 96]
  vmovups ymm2, [edx + ecx - 64]
  vmovups ymm3, [edx + ecx - 32]

  // Multiply with dst and store back
  vmulps ymm0, ymm0, [eax + ecx - 128]
  vmulps ymm1, ymm1, [eax + ecx - 96]
  vmulps ymm2, ymm2, [eax + ecx - 64]
  vmulps ymm3, ymm3, [eax + ecx - 32]

  vmovups [eax + ecx - 128], ymm0
  vmovups [eax + ecx - 96],  ymm1
  vmovups [eax + ecx - 64],  ymm2
  vmovups [eax + ecx - 32],  ymm3

  jmp @Loop1

@loopEnd1:
  sub ecx, 128
  jz @loop3End

  // Handle remaining groups of 4 elements (XMM)
@Loop2:
  add ecx, 16
  jg @Loop2End

  vmovups xmm0, [edx + ecx - 16]
  vmulps xmm0, xmm0, [eax + ecx - 16]
  vmovups [eax + ecx - 16], xmm0
  jmp @Loop2

@Loop2End:
  sub ecx, 16
  jz @loop3End

  // Handle last 0..3 elements (scalar)
@loop3:
  add ecx, 4
  jg @loop3End

  vmovss xmm0, [edx + ecx - 4]
  vmulss xmm0, xmm0, [eax + ecx - 4]
  vmovss [eax + ecx - 4], xmm0
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
  Parameters (register): EAX=dst, EDX=src, ECX=N.
-----------------------------------------------------------------------------}
procedure _AVX2Add( dst : PSingle; src : PSingle; N : Integer);
asm
  imul ecx, -4                 // N := -N for reverse traversal
  sub eax, ecx                 // Adjust dst pointer
  sub edx, ecx                 // Adjust src pointer

  // Main loop: 32 elements (4 YMM blocks)
@Loop1:
  add ecx, 128
  jg @loopEnd1

  vmovups ymm0, [edx + ecx - 128]
  vmovups ymm1, [edx + ecx - 96]
  vmovups ymm2, [edx + ecx - 64]
  vmovups ymm3, [edx + ecx - 32]

  vaddps ymm0, ymm0, [eax + ecx - 128]
  vaddps ymm1, ymm1, [eax + ecx - 96]
  vaddps ymm2, ymm2, [eax + ecx - 64]
  vaddps ymm3, ymm3, [eax + ecx - 32]

  vmovups [eax + ecx - 128], ymm0
  vmovups [eax + ecx - 96],  ymm1
  vmovups [eax + ecx - 64],  ymm2
  vmovups [eax + ecx - 32],  ymm3

  jmp @Loop1

@loopEnd1:
  sub ecx, 128
  jz @loop3End

  // Handle remaining groups of 4 elements (XMM)
@Loop2:
  add ecx, 16
  jg @Loop2End

  vmovups xmm0, [edx + ecx - 16]
  vaddps xmm0, xmm0, [eax + ecx - 16]
  vmovups [eax + ecx - 16], xmm0
  jmp @Loop2

@Loop2End:
  sub ecx, 16
  jz @loop3End

  // Handle last 0..3 elements (scalar)
@loop3:
  add ecx, 4
  jg @loop3End

  vmovss xmm0, [edx + ecx - 4]
  vaddss xmm0, xmm0, [eax + ecx - 4]
  vmovss [eax + ecx - 4], xmm0
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
  Parameters (register): EAX=dst, EDX=src, ECX=N.
-----------------------------------------------------------------------------}
procedure _AVX2Max( dst : PSingle; src : PSingle; N : Integer);
asm
  imul ecx, -4                 // N := -N for reverse traversal
  sub eax, ecx                 // Adjust dst pointer
  sub edx, ecx                 // Adjust src pointer

  // Main loop: 32 elements (4 YMM blocks)
@Loop1:
  add ecx, 128
  jg @loopEnd1

  vmovups ymm2, [eax + ecx - 128]
  vmovups ymm3, [eax + ecx - 96]
  vmovups ymm4, [eax + ecx - 64]
  vmovups ymm5, [eax + ecx - 32]

  vmaxps ymm2, ymm2, [edx + ecx - 128]
  vmaxps ymm3, ymm3, [edx + ecx - 96]
  vmaxps ymm4, ymm4, [edx + ecx - 64]
  vmaxps ymm5, ymm5, [edx + ecx - 32]

  vmovups [eax + ecx - 128], ymm2
  vmovups [eax + ecx - 96],  ymm3
  vmovups [eax + ecx - 64],  ymm4
  vmovups [eax + ecx - 32],  ymm5

  jmp @Loop1

@loopEnd1:
  sub ecx, 128
  jz @loop3End

  // Handle 4-element groups (XMM)
@Loop2:
  add ecx, 16
  jg @Loop2End

  vmovups xmm2, [eax + ecx - 16]
  vmaxps xmm2, xmm2, [edx + ecx - 16]
  vmovups [eax + ecx - 16], xmm2
  jmp @Loop2

@Loop2End:
  sub ecx, 16
  jz @loop3End

  // Handle last 0..3 elements (scalar)
@loop3:
  add ecx, 4
  jg @loop3End

  vmovss xmm2, [eax + ecx - 4]
  vmaxss xmm2, xmm2, [edx + ecx - 4]
  vmovss [eax + ecx - 4], xmm2
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
  Processes 8 elements per iteration using YMM.
  Tail (0..7) accumulated in XMM4, added after horizontal sum.
  Parameters (register): EAX=dst, EDX=src, ECX=N.
  Returns result via ST(0).
-----------------------------------------------------------------------------}
function _AVX2SumDiff(dst: PSingle; src: PSingle; N: Integer): Single;
asm
  push ebp
  mov ebp, esp
  sub esp, 4

  mov eax, dst
  mov edx, src
  mov ecx, N

  vxorps ymm0, ymm0, ymm0    // main accumulator

  // generate absolute mask: 0x7FFFFFFF
  vpcmpeqd ymm1, ymm1, ymm1
  vpsrld   ymm1, ymm1, 1

  // ---- Bulk: process 8 elements per iteration ----
@Loop8:
  cmp ecx, 8
  jl @Tail

  vmovups ymm2, [eax]
  vmovups ymm3, [edx]
  vsubps  ymm2, ymm2, ymm3
  vandps  ymm2, ymm2, ymm1
  vaddps  ymm0, ymm0, ymm2

  add eax, 32
  add edx, 32
  sub ecx, 8
  jmp @Loop8

@Tail:
  test ecx, ecx
  jz @Done

  // ---- Tail: accumulate in XMM4 ----
  vxorps xmm4, xmm4, xmm4
@ScalarLoop:
  vmovss xmm2, [eax]
  vmovss xmm3, [edx]
  vsubss xmm2, xmm2, xmm3
  vandps xmm2, xmm2, xmm1
  vaddss xmm4, xmm4, xmm2
  add eax, 4
  add edx, 4
  sub ecx, 1
  jnz @ScalarLoop

@Done:
  // ---- Horizontal sum of ymm0 (8 lanes) into xmm0 ----
  vextractf128 xmm2, ymm0, 1
  vaddps xmm0, xmm0, xmm2
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0

  // ---- Add tail sum ----
  vaddss xmm0, xmm0, xmm4

  // ---- Return via ST(0) ----
  movss dword ptr [esp], xmm0
  fld dword ptr [esp]

  vzeroupper
  add esp, 4
  pop ebp
  ret
end;

function _AVX512SumDiff( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2SumDiff(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Squared Euclidean Distance: result = sum_i (dst[i] - src[i])^2.
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Tail (0..31) handled by XMM and scalar.
  Parameters (register): EAX=dst, EDX=src, ECX=N.
  Returns result via ST(0).
-----------------------------------------------------------------------------}
function _AVX2DistanceSqr(dst: PSingle; src: PSingle; N: Integer): Single;
asm
  push ebp
  mov ebp, esp
  sub esp, 4

  mov eax, dst
  mov edx, src
  mov ecx, N

  vxorps ymm0, ymm0, ymm0    // accumulator

  // ---- Main loop: process 32 elements per iteration ----
@Loop32:
  cmp ecx, 32
  jl @Loop4

  vmovups ymm2, [eax]
  vmovups ymm3, [edx]
  vsubps  ymm2, ymm2, ymm3
  vmulps  ymm2, ymm2, ymm2
  vaddps  ymm0, ymm0, ymm2

  vmovups ymm2, [eax+32]
  vmovups ymm3, [edx+32]
  vsubps  ymm2, ymm2, ymm3
  vmulps  ymm2, ymm2, ymm2
  vaddps  ymm0, ymm0, ymm2

  vmovups ymm2, [eax+64]
  vmovups ymm3, [edx+64]
  vsubps  ymm2, ymm2, ymm3
  vmulps  ymm2, ymm2, ymm2
  vaddps  ymm0, ymm0, ymm2

  vmovups ymm2, [eax+96]
  vmovups ymm3, [edx+96]
  vsubps  ymm2, ymm2, ymm3
  vmulps  ymm2, ymm2, ymm2
  vaddps  ymm0, ymm0, ymm2

  add eax, 128
  add edx, 128
  sub ecx, 32
  jmp @Loop32

@Loop4:
  cmp ecx, 4
  jl @Tail

  vmovups xmm2, [eax]
  vmovups xmm3, [edx]
  vsubps  xmm2, xmm2, xmm3
  vmulps  xmm2, xmm2, xmm2
  vaddps  xmm0, xmm0, xmm2

  add eax, 16
  add edx, 16
  sub ecx, 4
  jmp @Loop4

@Tail:
  test ecx, ecx
  jz @Done

  vxorps xmm4, xmm4, xmm4
@ScalarLoop:
  vmovss xmm2, [eax]
  vmovss xmm3, [edx]
  vsubss xmm2, xmm2, xmm3
  vmulss xmm2, xmm2, xmm2
  vaddss xmm4, xmm4, xmm2
  add eax, 4
  add edx, 4
  sub ecx, 1
  jnz @ScalarLoop

@Done:
  // Horizontal sum of ymm0 (8 lanes) into xmm0
  vextractf128 xmm2, ymm0, 1
  vaddps xmm0, xmm0, xmm2
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0

  // Add tail sum
  vaddss xmm0, xmm0, xmm4

  // Return via ST(0)
  movss dword ptr [esp], xmm0
  fld dword ptr [esp]

  vzeroupper
  add esp, 4
  pop ebp
  ret
end;

function _AVX512DistanceSqr( dst : PSingle; src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2DistanceSqr(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 elementwise subtraction: dst[i] = dst[i] - src[i].
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Parameters (register): EAX=dst, EDX=src, ECX=N.
-----------------------------------------------------------------------------}
procedure _AVX2Sub( dst : PSingle; src : PSingle; N : Integer);
asm
  imul ecx, -4                 // N := -N for reverse traversal
  sub eax, ecx                 // Adjust dst pointer
  sub edx, ecx                 // Adjust src pointer

  // Main loop: 32 elements (4 YMM blocks)
@Loop1:
  add ecx, 128
  jg @loopEnd1

  vmovups ymm2, [eax + ecx - 128]
  vmovups ymm3, [edx + ecx - 128]
  vsubps  ymm2, ymm2, ymm3
  vmovups [eax + ecx - 128], ymm2

  vmovups ymm2, [eax + ecx - 96]
  vmovups ymm3, [edx + ecx - 96]
  vsubps  ymm2, ymm2, ymm3
  vmovups [eax + ecx - 96], ymm2

  vmovups ymm2, [eax + ecx - 64]
  vmovups ymm3, [edx + ecx - 64]
  vsubps  ymm2, ymm2, ymm3
  vmovups [eax + ecx - 64], ymm2

  vmovups ymm2, [eax + ecx - 32]
  vmovups ymm3, [edx + ecx - 32]
  vsubps  ymm2, ymm2, ymm3
  vmovups [eax + ecx - 32], ymm2

  jmp @Loop1

@loopEnd1:
  sub ecx, 128
  jz @loop3End

  // Handle 4-element groups (XMM)
@Loop2:
  add ecx, 16
  jg @Loop2End

  vmovups xmm2, [eax + ecx - 16]
  vmovups xmm3, [edx + ecx - 16]
  vsubps  xmm2, xmm2, xmm3
  vmovups [eax + ecx - 16], xmm2
  jmp @Loop2

@Loop2End:
  sub ecx, 16
  jz @loop3End

  // Handle last 0..3 elements (scalar)
@loop3:
  add ecx, 4
  jg @loop3End

  vmovss xmm2, [eax + ecx - 4]
  vmovss xmm3, [edx + ecx - 4]
  vsubss xmm2, xmm2, xmm3
  vmovss [eax + ecx - 4], xmm2
  jmp @loop3

@loop3End:
  vzeroupper
end;

procedure _AVX512Sub( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2Sub(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Sum of elements: result = sum_i src[i].
  Uses 256-bit YMM registers, processes 32 elements per iteration.
  Tail handled by XMM and scalar.
  Parameters (register): EAX=src, EDX=N.
  Returns result via ST(0).
-----------------------------------------------------------------------------}
function _AVX2GetSum(src: PSingle; N: Integer): Single;
asm
  push ebp
  mov ebp, esp
  sub esp, 4

  mov eax, src
  mov ecx, N

  vxorps ymm0, ymm0, ymm0    // accumulator (8 lanes)

  // ---- Main loop: process 32 elements per iteration ----
@Loop32:
  cmp ecx, 32
  jl @Loop4

  vaddps ymm0, ymm0, [eax]
  vaddps ymm0, ymm0, [eax+32]
  vaddps ymm0, ymm0, [eax+64]
  vaddps ymm0, ymm0, [eax+96]

  add eax, 128
  sub ecx, 32
  jmp @Loop32

@Loop4:
  cmp ecx, 4
  jl @Tail

  vaddps xmm0, xmm0, [eax]
  add eax, 16
  sub ecx, 4
  jmp @Loop4

@Tail:
  test ecx, ecx
  jz @Done

  // Tail accumulator in xmm1
  vxorps xmm1, xmm1, xmm1
@ScalarLoop:
  vaddss xmm1, xmm1, [eax]
  add eax, 4
  sub ecx, 1
  jnz @ScalarLoop

@Done:
  // ---- Horizontal sum of ymm0 (8 lanes) into xmm0 ----
  vextractf128 xmm2, ymm0, 1
  vaddps xmm0, xmm0, xmm2
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0

  // ---- Add tail sum ----
  vaddss xmm0, xmm0, xmm1

  // ---- Return via ST(0) ----
  movss dword ptr [esp], xmm0
  fld dword ptr [esp]

  vzeroupper
  add esp, 4
  pop ebp
  ret
end;

function _AVX512GetSum( src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2GetSum(src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 sum of squares: result = sum_i src[i]^2.
  Processes 8 elements per iteration.
  Tail (0..7) handled by scalar; xmm1 is zero if no tail.
  Parameters (register): EAX=src, EDX=N.
  Returns result via ST(0).
-----------------------------------------------------------------------------}
function _AVX2GetSumSqr(src: PSingle; N: Integer): Single;
asm
  push ebp
  mov ebp, esp
  sub esp, 4

  mov eax, src
  mov ecx, N

  vxorps ymm0, ymm0, ymm0

  // ---- Process 8 elements per iteration ----
@Loop8:
  cmp ecx, 8
  jl @Tail

  vmovups ymm1, [eax]
  vmulps  ymm1, ymm1, ymm1
  vaddps  ymm0, ymm0, ymm1

  add eax, 32
  sub ecx, 8
  jmp @Loop8

@Tail:
  // Clear tail accumulator (xmm1 will be added only if non-zero)
  vxorps xmm1, xmm1, xmm1
  test ecx, ecx
  jz @Merge

@ScalarLoop:
  vmovss xmm2, [eax]
  vmulss xmm2, xmm2, xmm2
  vaddss xmm1, xmm1, xmm2
  add eax, 4
  sub ecx, 1
  jnz @ScalarLoop

@Merge:
  // ---- Horizontal sum of ymm0 (8 lanes) into xmm0 ----
  vextractf128 xmm2, ymm0, 1
  vaddps xmm0, xmm0, xmm2
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0

  // ---- Add tail sum (xmm1 is zero if no tail) ----
  vaddss xmm0, xmm0, xmm1

  // ---- Return via ST(0) ----
  movss dword ptr [esp], xmm0
  fld dword ptr [esp]

  vzeroupper
  add esp, 4
  pop ebp
  ret
end;

function _AVX512GetSumSqr( src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2GetSumSqr(src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 exponential: dst[i] = exp(src[i]).
  Uses 8-wide YMM polynomial for bulk (ymm0..ymm7 only), scalar polynomial for tail.
  Parameters (register): EAX=dst, EDX=src, ECX=N.
-----------------------------------------------------------------------------}
procedure _AVX2Exp( dst : PSingle; src : PSingle; N : Integer);
asm
  push ebx
  push esi
  push edi

  mov esi, edx                 // src
  mov edi, eax                 // dst
  mov ebx, ecx                 // N

  test ebx, ebx
  jle @Exit

  // tail count = N mod 8
  mov ecx, ebx
  and ecx, 7
  sub ebx, ecx                 // ebx = bulk count (multiple of 8)

  jz @Tail

  // bulk: process 8 at a time
  mov eax, ebx
  shr eax, 3                   // number of 8-element blocks

  // Load constants into registers for bulk loop
  // Use ymm6 = log2e, ymm7 = ln2, ymm5 = 127
  vbroadcastss ymm6, [cAVXLog2e]
  vbroadcastss ymm7, [cAVXLn2]
  vpbroadcastd ymm5, [cAVXExp127]

@BulkLoop:
  vmovups ymm0, [esi]

  // Clamp to [-88.376, 88.376] using ymm1 as scratch
  vbroadcastss ymm1, [cAVXExpHi]   // ymm1 = Hi
  vminps ymm0, ymm0, ymm1
  vbroadcastss ymm1, [cAVXExpLo]   // ymm1 = Lo
  vmaxps ymm0, ymm0, ymm1

  // t = x * log2e (ymm6 holds log2e)
  vmulps ymm1, ymm0, ymm6
  vroundps ymm2, ymm1, 0
  vsubps ymm1, ymm1, ymm2         // f = t - k

  // g = f * ln2 (ymm7 holds ln2)
  vmulps ymm3, ymm1, ymm7

  // Horner polynomial for 2^f: ymm4 = P6
  vbroadcastss ymm4, [cAVXExpP6]
  vbroadcastss ymm0, [cAVXExpP5]   // reuse ymm0 as scratch
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
  vfmadd213ps ymm4, ymm3, ymm0    // ymm4 = 2^f

  // 2^k
  vcvtps2dq ymm2, ymm2
  vpaddd ymm2, ymm2, ymm5         // ymm5 holds 127
  vpslld ymm2, ymm2, 23

  // result = 2^f * 2^k
  vmulps ymm0, ymm4, ymm2
  vmovups [edi], ymm0

  add esi, 32
  add edi, 32
  dec eax
  jnz @BulkLoop

  vzeroupper

@Tail:
  mov eax, ecx                 // tail count (0..7)
  test eax, eax
  jz @Exit

  // Load scalar constants once, outside the loop
  vbroadcastss xmm6, [cAVXExpHi]
  vbroadcastss xmm7, [cAVXExpLo]
  vpbroadcastd xmm5, [cAVXExp127]

@TailLoop:
  vmovss xmm0, [esi]
  vminss xmm0, xmm0, xmm6
  vmaxss xmm0, xmm0, xmm7

  vmulss xmm1, xmm0, [cAVXLog2e]
  vroundss xmm2, xmm1, xmm1, 0
  vsubss xmm1, xmm1, xmm2

  vmulss xmm3, xmm1, [cAVXLn2]

  // Scalar polynomial: xmm4 = P6
  vbroadcastss xmm4, [cAVXExpP6]
  vfmadd213ss xmm4, xmm3, [cAVXExpP5]
  vfmadd213ss xmm4, xmm3, [cAVXExpP4]
  vfmadd213ss xmm4, xmm3, [cAVXExpP3]
  vfmadd213ss xmm4, xmm3, [cAVXExpP2]
  vfmadd213ss xmm4, xmm3, [cAVXExpP1]
  vfmadd213ss xmm4, xmm3, [cAVXExpP0]

  // 2^k (scalar)
  vcvtss2si ebx, xmm2
  add ebx, 127
  shl ebx, 23
  movd xmm2, ebx

  vmulss xmm0, xmm4, xmm2
  vmovss [edi], xmm0

  add esi, 4
  add edi, 4
  dec eax
  jnz @TailLoop

@Exit:
  vzeroupper
  pop edi
  pop esi
  pop ebx
end;

procedure _AVX512Exp( dst : PSingle; src : PSingle; N : Integer); inline;
begin
  _AVX2Exp(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 dot product: result = sum_i dst[i] * src[i].
  Uses 256-bit YMM registers with FMA, processes 32 elements per iteration.
  Tail handling: 4-element XMM groups then scalar remainder.
  Parameters (register): EAX=dst, EDX=src, ECX=N.
  Returns: ST(0) = Single (as per Delphi 32-bit floating-point return convention)
-----------------------------------------------------------------------------}
function _AVX2DotProd(dst: PSingle; src: PSingle; N: Integer): Single;
asm
  push ebx
  push esi
  push edi

  mov esi, edx                 // src
  mov edi, eax                 // dst
  mov ebx, ecx                 // N

  test ebx, ebx
  jle @ZeroResult

  // bulk = N - (N mod 32)
  mov eax, ebx
  and eax, 31                  // tail = N mod 32
  sub ebx, eax                 // bulk = N - tail

  // Bulk: process 32 elements at a time (4 YMM)
  vxorps ymm0, ymm0, ymm0
  vxorps ymm1, ymm1, ymm1
  vxorps ymm2, ymm2, ymm2
  vxorps ymm3, ymm3, ymm3

  mov ecx, ebx
  shr ecx, 5                   // number of 32-element blocks
  jz @Tail

@BulkLoop:
  vmovups ymm4, [edi]
  vfmadd231ps ymm0, ymm4, [esi]   // ymm0 += dst * src
  vmovups ymm5, [edi+32]
  vfmadd231ps ymm1, ymm5, [esi+32]
  vmovups ymm6, [edi+64]
  vfmadd231ps ymm2, ymm6, [esi+64]
  vmovups ymm7, [edi+96]
  vfmadd231ps ymm3, ymm7, [esi+96]
  add edi, 128
  add esi, 128
  dec ecx
  jnz @BulkLoop

  // Reduce YMM accumulators to XMM
  vaddps ymm0, ymm0, ymm1
  vaddps ymm2, ymm2, ymm3
  vaddps ymm0, ymm0, ymm2
  vextractf128 xmm4, ymm0, 1
  vaddps xmm0, xmm0, xmm4          // xmm0 holds partial sum

  vzeroupper

@Tail:
  // eax = tail count (0..31)
  mov ecx, eax
  and ecx, 3                   // tail mod 4 for scalar
  sub eax, ecx                 // tail4 = tail - scalar (multiple of 4)

  // Process tail in groups of 4 with XMM
  mov edx, eax
  shr edx, 2                   // number of 4-element blocks
  jz @ScalarTail

@XmmLoop:
  vmovups xmm4, [edi]
  vmulps xmm4, xmm4, [esi]
  vaddps xmm0, xmm0, xmm4
  add edi, 16
  add esi, 16
  dec edx
  jnz @XmmLoop

@ScalarTail:
  // ecx = remaining 0..3 elements
  test ecx, ecx
  jz @Finish

@ScalarLoop:
  vmovss xmm4, [edi]
  vmulss xmm4, xmm4, [esi]
  vaddss xmm0, xmm0, xmm4
  add edi, 4
  add esi, 4
  dec ecx
  jnz @ScalarLoop

@Finish:
  // Horizontal sum xmm0 -> xmm0[0] = sum
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  jmp @Exit

@ZeroResult:
  vxorps xmm0, xmm0, xmm0
  // fall through to @Exit

@Exit:
  // Convert XMM0 to ST(0) for return
  sub esp, 4
  movss [esp], xmm0
  fld dword ptr [esp]
  add esp, 4

  pop edi
  pop esi
  pop ebx
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
  Parameters (register): EAX=PtrA, EDX=PtrB, ECX=N.
-----------------------------------------------------------------------------}
function _AVX2DotProdInt8( dst : PShortInt; src : PSingle; N : Integer ) : Single;
asm
  push esi                     // save ESI
  push edi                     // save EDI

  mov esi, eax                 // ESI = PtrA
  mov edi, edx                 // EDI = PtrB
  mov eax, ecx                 // EAX = N

  test eax, eax
  jle @Zero

  // Bulk = N - (N mod 32)
  mov edx, eax
  and edx, 31                  // tail = N mod 32
  sub eax, edx                 // bulk = N - tail

  vxorps ymm0, ymm0, ymm0
  vxorps ymm1, ymm1, ymm1
  vxorps ymm6, ymm6, ymm6
  vxorps ymm7, ymm7, ymm7

  mov ecx, eax
  shr ecx, 5                    // number of 32-element blocks
  jz @Tail

@BulkLoop:
  vpmovsxbd ymm2, [esi]
  vpmovsxbd ymm3, [esi+8]
  vpmovsxbd ymm4, [esi+16]
  vpmovsxbd ymm5, [esi+24]

  vcvtdq2ps ymm2, ymm2
  vcvtdq2ps ymm3, ymm3
  vcvtdq2ps ymm4, ymm4
  vcvtdq2ps ymm5, ymm5

  vfmadd231ps ymm0, ymm2, [edi]
  vfmadd231ps ymm1, ymm3, [edi+32]
  vfmadd231ps ymm6, ymm4, [edi+64]
  vfmadd231ps ymm7, ymm5, [edi+96]

  add esi, 32
  add edi, 128
  dec ecx
  jnz @BulkLoop

  vaddps ymm0, ymm0, ymm1
  vaddps ymm6, ymm6, ymm7
  vaddps ymm0, ymm0, ymm6
  vextractf128 xmm2, ymm0, 1
  vaddps xmm0, xmm0, xmm2
  vzeroupper

@Tail:
  // edx = tail count (0..31)
  mov eax, edx
  and eax, 3                    // scalar remainder
  sub edx, eax                  // groups of 4 for XMM

  mov ecx, edx
  shr ecx, 2                    // number of 4-element groups
  jz @ScalarTail

@XmmLoop:
  vpmovsxbd xmm2, [esi]
  vcvtdq2ps xmm2, xmm2
  vmovups xmm3, [edi]
  vmulps xmm2, xmm2, xmm3
  vaddps xmm0, xmm0, xmm2
  add esi, 4
  add edi, 16
  dec ecx
  jnz @XmmLoop

@ScalarTail:
  test eax, eax
  jz @Finish

  movsx ecx, byte ptr [esi]     // load int8 sign-extended to 32-bit
  vcvtsi2ss xmm2, xmm2, ecx
  vmovss xmm3, [edi]
  vmulss xmm2, xmm2, xmm3
  vaddss xmm0, xmm0, xmm2
  add esi, 1
  add edi, 4
  dec eax
  jnz @ScalarTail

@Finish:
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  movss [Result], xmm0
  jmp @Done

@Zero:
  xorps xmm0, xmm0
  movss [Result], xmm0

@Done:
  vzeroupper
  pop edi
  pop esi
end;

function _AVX512DotProdInt8( dst : PShortInt; src : PSingle; N : Integer ) : Single; inline;
begin
  Result := _AVX2DotProdInt8(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 int8 scalar multiply-add: dst[i] += W * codes[i].
  codes is int8 (signed byte), W is Single scalar.
  Processes 32 elements per loop, then XMM groups and scalar tail.
  Parameters (register): EAX=dst, EDX=codes, ECX=N, [EBP+8]=W (const)
-----------------------------------------------------------------------------}
procedure _AVX2MulAddInt8Scalar( dst : PSingle; codes : PShortInt; W : Single; N : Integer );
asm
  push esi
  push edi

  mov esi, edx                 // ESI = codes
  mov edi, eax                 // EDI = dst
  mov eax, ecx                 // EAX = N

  test eax, eax
  jle @Exit

  // Load W and broadcast
  lea edx, W                   // [ebp+8] is W
  vbroadcastss ymm5, [edx]     // ymm5 = W

  // Bulk = N - (N mod 32)
  mov edx, eax
  and edx, 31                  // tail = N mod 32
  sub eax, edx                 // bulk = N - tail

  mov ecx, eax
  shr ecx, 5                   // number of 32-element blocks
  jz @Tail

@BulkLoop:
  vpmovsxbd ymm0, [esi]
  vpmovsxbd ymm1, [esi+8]
  vpmovsxbd ymm2, [esi+16]
  vpmovsxbd ymm3, [esi+24]

  vcvtdq2ps ymm0, ymm0
  vcvtdq2ps ymm1, ymm1
  vcvtdq2ps ymm2, ymm2
  vcvtdq2ps ymm3, ymm3

  vmovups ymm6, [edi]
  vmovups ymm7, [edi+32]
  vfmadd231ps ymm6, ymm0, ymm5
  vfmadd231ps ymm7, ymm1, ymm5
  vmovups [edi], ymm6
  vmovups [edi+32], ymm7

  vmovups ymm6, [edi+64]
  vmovups ymm7, [edi+96]
  vfmadd231ps ymm6, ymm2, ymm5
  vfmadd231ps ymm7, ymm3, ymm5
  vmovups [edi+64], ymm6
  vmovups [edi+96], ymm7

  add esi, 32
  add edi, 128
  dec ecx
  jnz @BulkLoop

@Tail:
  // edx = tail count (0..31)
  mov eax, edx
  and eax, 3                    // scalar remainder
  sub edx, eax                  // groups of 4 for XMM

  mov ecx, edx
  shr ecx, 2                    // number of 4-element groups
  jz @ScalarTail

@XmmLoop:
  vpmovsxbd xmm0, [esi]
  vcvtdq2ps xmm0, xmm0
  vmovups xmm6, [edi]
  vfmadd231ps xmm6, xmm0, xmm5
  vmovups [edi], xmm6
  add esi, 4
  add edi, 16
  dec ecx
  jnz @XmmLoop

@ScalarTail:
  test eax, eax
  jz @Exit

  movsx ecx, byte ptr [esi]     // load int8 sign-extended
  vcvtsi2ss xmm0, xmm0, ecx
  vmulss xmm0, xmm0, xmm5       // W * code
  vmovss xmm1, [edi]            // load dst
  vaddss xmm1, xmm1, xmm0
  vmovss [edi], xmm1
  add esi, 1
  add edi, 4
  dec eax
  jnz @ScalarTail

@Exit:
  vzeroupper
  pop edi
  pop esi
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
  Parameters: EAX=dst, EDX=src, ECX=codes, [EBP+8]=N.
-----------------------------------------------------------------------------}
procedure _AVX2MulAddInt8(dst: PSingle; src: PSingle; codes: PShortInt; N: Integer);
asm
  push esi
  push edi
  push ebx

  sub esp, 4

  mov edi, eax                 // edi = dst
  mov esi, ecx                 // esi = codes
  mov ebx, edx                 // ebx = src
  mov eax, [ebp+8]             // eax = N

  test eax, eax
  jle @Exit

  // Bulk = N - (N mod 8)
  mov edx, eax
  and edx, 7                   // tail = N mod 8
  sub eax, edx                 // bulk = N - tail
  mov ecx, eax
  shr ecx, 3                   // number of 8-element blocks
  jz @Tail

@Loop8:
  vpmovsxbd ymm0, [esi]        // load 8 int8 codes, sign-extend to int32
  vcvtdq2ps ymm0, ymm0         // convert int32 to float
  vmovups ymm1, [ebx]          // load 8 src floats
  vmulps  ymm0, ymm0, ymm1     // codes * src
  vmovups ymm1, [edi]          // load 8 dst floats
  vaddps  ymm0, ymm0, ymm1     // dst += codes * src
  vmovups [edi], ymm0

  add esi, 8
  add ebx, 32
  add edi, 32
  dec ecx
  jnz @Loop8

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  xor ecx, ecx
@ScalarLoop:
  movsx eax, byte ptr [esi + ecx]   // load int8 code
  vcvtsi2ss xmm0, xmm0, eax
  vmovss xmm1, [ebx + ecx*4]
  vmulss xmm0, xmm0, xmm1
  vmovss xmm1, [edi + ecx*4]
  vaddss xmm1, xmm1, xmm0
  vmovss [edi + ecx*4], xmm1
  inc ecx
  cmp ecx, edx
  jl @ScalarLoop

@Exit:
  vzeroupper
  add esp, 4

  pop ebx
  pop edi
  pop esi
end;

procedure _AVX512MulAddInt8( dst : PSingle; src : PSingle; codes : PShortInt; N : Integer ); inline;
begin
  _AVX2MulAddInt8(dst, src, codes, N);
end;

{-----------------------------------------------------------------------------
  AVX2 max absolute finite: result = max(|src[i]|) for i = 0..N-1,
  ignoring NaN and Inf. Processes 8 elements per loop (YMM).
  Returns the maximum absolute finite value.
-----------------------------------------------------------------------------}
function _AVX2MaxAbsFinite( src : PSingle; N : Integer ) : Single;
asm
  push ebx
  push esi
  push edi

  mov esi, eax
  mov eax, edx
  test eax, eax
  jle @Zero

  // Initialize result to 0
  vxorps xmm0, xmm0, xmm0

  // Load constants using AVX instructions
  mov edi, $7FFFFFFF
  vmovd xmm3, edi                 // xmm3 = abs mask
  vbroadcastss ymm3, xmm3         // ymm3 = {absmask, ...}

  mov edi, $7F7FFFFF              // max finite single
  vmovd xmm2, edi
  vbroadcastss ymm2, xmm2         // ymm2 = {maxfinite, ...}

  // Bulk count = N - (N mod 8)
  mov edx, eax
  and edx, 7
  sub eax, edx
  mov ecx, eax
  shr ecx, 3
  jz @Tail

  vxorps ymm4, ymm4, ymm4         // accumulator = 0

@BulkLoop:
  vmovups ymm0, [esi]
  vandps ymm0, ymm0, ymm3         // |x|
  vcmpps ymm1, ymm0, ymm2, 18     // LE_OQ (finite)
  vandps ymm0, ymm0, ymm1         // non-finite -> 0
  vmaxps ymm4, ymm4, ymm0
  add esi, 32
  dec ecx
  jnz @BulkLoop

  // Reduce ymm4 to scalar in xmm0
  vextractf128 xmm0, ymm4, 1
  vmaxps xmm0, xmm0, xmm4         // xmm0 = max of two halves
  vpshufd xmm1, xmm0, $55         // xmm1[0] = xmm0[1]
  vmaxss xmm0, xmm0, xmm1
  vpshufd xmm1, xmm0, $AA         // xmm1[0] = original xmm0[2] (after previous)
  vmaxss xmm0, xmm0, xmm1
  vpshufd xmm1, xmm0, $FF         // xmm1[0] = original xmm0[3]
  vmaxss xmm0, xmm0, xmm1
  vzeroupper

@Tail:
  test edx, edx
  jz @Return

  // Scalar constants for tail
  mov edi, $7FFFFFFF
  vmovd xmm3, edi
  mov edi, $7F7FFFFF
  vmovd xmm2, edi
  xor ecx, ecx
@TailLoop:
  vmovss xmm1, [esi + ecx*4]
  vandps xmm1, xmm1, xmm3         // |x|
  vcmpltss xmm4, xmm1, xmm2       // finite?
  vandps xmm1, xmm1, xmm4         // non-finite -> 0
  vmaxss xmm0, xmm0, xmm1         // update max
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Return:
  // Return single in ST(0)
  sub esp, 4
  vmovss [esp], xmm0
  fld dword ptr [esp]
  add esp, 4
  pop edi
  pop esi
  pop ebx
  ret

@Zero:
  vxorps xmm0, xmm0, xmm0
  sub esp, 4
  vmovss [esp], xmm0
  fld dword ptr [esp]
  add esp, 4
  pop edi
  pop esi
  pop ebx
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
-----------------------------------------------------------------------------}
procedure _AVX2QuantizeInt8( dst : PShortInt; src : PSingle; N : Integer; const MaxAbs : Single );
asm
  sub esp, 16                     // allocate local stack space
  push esi
  push edi

  mov edi, eax                    // dst
  mov esi, edx                    // src
  mov eax, ecx                    // N

  test eax, eax
  jle @Exit

  // ---- Compute factor = 127 / MaxAbs using SSE (no x87) ----
  // Load MaxAbs from stack (compiler has generated prologue, so ebp is valid)
  movss xmm0, [ebp+8]             // xmm0 = MaxAbs
  // Load 127 as integer -> convert to float
  mov edx, 127
  vmovd xmm1, edx                 // xmm1 = 127 (as integer)
  vcvtdq2ps xmm1, xmm1            // xmm1 = 127.0f
  // Compute factor = 127 / MaxAbs
  vdivss xmm0, xmm1, xmm0         // xmm0 = 127 / MaxAbs
  // Store factor to local stack (for tail)
  vmovss [esp], xmm0
  // Broadcast to YMM5
  vbroadcastss ymm5, xmm0         // ymm5 = factor in all lanes

  // ---- Load +127 and -127 as broadcast constants ----
  mov edx, $42FE0000              // +127.0
  vmovd xmm6, edx
  vbroadcastss ymm6, xmm6         // ymm6 = +127

  mov edx, $C2FE0000              // -127.0
  vmovd xmm7, edx
  vbroadcastss ymm7, xmm7         // ymm7 = -127

  // ---- Bulk processing ----
  mov edx, eax
  and edx, 7                      // tail
  sub eax, edx                    // bulk
  mov ecx, eax
  shr ecx, 3
  jz @Tail

@BulkLoop:
  vmovups ymm0, [esi]             // load 8 floats
  vcmpps ymm1, ymm0, ymm0, 7      // ORD_Q (false for NaN)
  vandps ymm0, ymm0, ymm1         // NaN -> 0
  vmulps ymm0, ymm0, ymm5         // * factor
  vminps ymm0, ymm0, ymm6         // clip to +127
  vmaxps ymm0, ymm0, ymm7         // clip to -127
  vcvtps2dq ymm0, ymm0            // round to nearest (banker's)
  vextracti128 xmm1, ymm0, 1
  vpackssdw xmm0, xmm0, xmm1      // dwords -> words
  vpxor xmm2, xmm2, xmm2
  vpacksswb xmm0, xmm0, xmm2      // words -> bytes
  vmovq qword ptr [edi], xmm0     // store 8 bytes
  add esi, 32
  add edi, 8
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  // ---- Scalar tail ----
  movss xmm5, [esp]               // reload factor
  // xmm6 and xmm7 already contain +127 and -127 as scalars
  xor ecx, ecx
@TailLoop:
  vmovss xmm0, [esi + ecx*4]
  vcmpss xmm1, xmm0, xmm0, 7
  vandps xmm0, xmm0, xmm1
  vmulss xmm0, xmm0, xmm5
  vminss xmm0, xmm0, xmm6
  vmaxss xmm0, xmm0, xmm7
  vcvtss2si eax, xmm0
  mov byte ptr [edi + ecx], al
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
  pop edi
  pop esi
  add esp, 16

end;

procedure _AVX512QuantizeInt8( dst : PShortInt; src : PSingle; N : Integer; const MaxAbs : Single );
begin
  _AVX2QuantizeInt8(dst, src, N, MaxAbs);
end;

{-----------------------------------------------------------------------------
  AVX2 int8 dequantize: dst[i] = Scale * src[i] for i = 0..N-1.
  Processes 8 elements per loop (YMM), scalar tail.
-----------------------------------------------------------------------------}
procedure _AVX2DequantizeInt8( dst : PSingle; src : PShortInt; N : Integer; const Scale : Single );
asm
  push esi
  push edi

  mov edi, eax                     // dst
  mov esi, edx                     // src
  mov eax, ecx                     // N

  test eax, eax
  jle @Exit

  // ---- Load Scale via GPR -> XMM -> broadcast (avoid memory operand issues) ----
  mov edx, [ebp+8]                 // Scale bit pattern
  vmovd xmm2, edx
  vbroadcastss ymm2, xmm2          // ymm2 = Scale in all lanes

  // Bulk = N - (N mod 8)
  mov edx, eax
  and edx, 7                       // tail
  sub eax, edx                     // bulk
  mov ecx, eax
  shr ecx, 3                       // number of 8-element blocks
  jz @Tail

@BulkLoop:
  vpmovsxbd ymm0, [esi]            // 8 bytes -> 8 dwords (sign-extended)
  vcvtdq2ps ymm0, ymm0
  vmulps ymm0, ymm0, ymm2
  vmovups [edi], ymm0
  add esi, 8
  add edi, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  // xmm2 already contains Scale in its low lane
  xor ecx, ecx
@TailLoop:
  movsx eax, byte ptr [esi + ecx]   // load int8, sign-extend
  vcvtsi2ss xmm0, xmm0, eax
  vmulss xmm0, xmm0, xmm2           // xmm2 low lane is Scale
  vmovss [edi + ecx*4], xmm0
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
  pop edi
  pop esi
end;

procedure _AVX512DequantizeInt8( dst : PSingle; src : PShortInt; N : Integer; const Scale : Single ); inline;
begin
  _AVX512DequantizeInt8(dst, src, N, Scale);
end;

{-----------------------------------------------------------------------------
  AVX2 decode bfloat16 to Single: dst[i] = (float)bfloat16(src[i]).
  Processes 8 elements per loop (YMM).
  Parameters: EAX=dst, EDX=src, ECX=N.
-----------------------------------------------------------------------------}
procedure _AVX2DecodeBF16( dst : PSingle; src : PWord; N : Integer );
asm
  push esi
  push edi

  mov edi, eax                 // EDI = dst
  mov esi, edx                 // ESI = src
  mov eax, ecx                 // EAX = N

  test eax, eax
  jle @Exit

  // Bulk = N - (N mod 8)
  mov edx, eax
  and edx, 7
  sub eax, edx                 // eax = bulk

  mov ecx, eax
  shr ecx, 3                   // number of 8-element blocks
  jz @Tail

@BulkLoop:
  vpmovzxwd ymm0, [esi]        // 8 words -> 8 dwords (low 16 bits each)
  vpslld ymm0, ymm0, 16        // shift to high 16 bits
  vmovups [edi], ymm0
  add esi, 16                  // advance 8 words (16 bytes)
  add edi, 32                  // advance 8 floats (32 bytes)
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  xor ecx, ecx
@TailLoop:
  movzx eax, word ptr [esi + ecx*2]
  shl eax, 16
  mov dword ptr [edi + ecx*4], eax
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
  pop edi
  pop esi
  ret
end;

procedure _AVX512DecodeBF16( dst : PSingle; src : PWord; N : Integer ); inline;
begin
  _AVX2DecodeBF16(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 ReLU gate mask: dst[i] = 1.0 if src[i] >= 0 else 0.0.
  Processes 8 elements per loop.
  Parameters: EAX=dst, EDX=src, ECX=N.
-----------------------------------------------------------------------------}
procedure _AVX2ReluGateMask(dst: PSingle; src: PSingle; N: Integer);
asm
  push esi
  push edi

  mov edi, eax                 // dst
  mov esi, edx                 // src
  mov eax, ecx                 // N

  test eax, eax
  jle @Exit

  // Constants: 1.0 and 0.0 in YMM
  vbroadcastss ymm2, dword ptr [cAVX8SSOne]   // ymm2 = 1.0
  vxorps ymm3, ymm3, ymm3                     // ymm3 = 0.0

  // Bulk count = N - (N mod 8)
  mov edx, eax
  and edx, 7                   // tail
  sub eax, edx                 // bulk
  mov ecx, eax
  shr ecx, 3                   // number of 8-element chunks
  jz @Tail

@BulkLoop:
  vmovups ymm0, [esi]
  vcmpps ymm1, ymm0, ymm3, 29 // GE_OQ (>= 0)
  vandps ymm1, ymm1, ymm2
  vmovups [edi], ymm1
  add esi, 32
  add edi, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  // Reload 1.0 and 0.0 into XMM for scalar tail
  vbroadcastss xmm2, dword ptr [cAVX8SSOne]   // 1.0
  vxorps xmm3, xmm3, xmm3                     // 0.0 (already zero, but safe)

  xor ecx, ecx
@TailLoop:
  vmovss xmm0, [esi + ecx*4]
  vcmpss xmm1, xmm0, xmm3, 29 // x >= 0 ?
  vandps xmm1, xmm1, xmm2
  vmovss [edi + ecx*4], xmm1
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
  pop edi
  pop esi
  ret
end;

procedure _AVX512ReluGateMask( dst : PSingle; src : PSingle; N : Integer ); inline;
begin
  _AVX2ReluGateMask(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 Leaky ReLU: dst[i] = src[i] if src[i] >= 0, else Slope * src[i].
  Uses 256-bit YMM, processes 8 elements per iteration.
  Parameters (register): EAX=dst, EDX=src, ECX=N, [EBP+8]=const Slope pointer.
-----------------------------------------------------------------------------}
procedure _AVX2LeakyRelu(dst: PSingle; src: PSingle; N: Integer; const Slope: Single);
asm
  push esi
  push edi

  mov edi, eax                 // EDI = dst
  mov esi, edx                 // ESI = src
  mov eax, ecx                 // EAX = N

  test eax, eax
  jle @Exit

  // Load Slope pointer from stack and broadcast
  lea edx, Slope               // or mov edx, [ebp+8] (pointer to Slope)
  vbroadcastss ymm2, [edx]     // ymm2 = Slope (broadcast to all lanes)

  // Bulk = N - (N mod 8)
  mov edx, eax
  and edx, 7
  sub eax, edx                 // eax = bulk

  mov ecx, eax
  shr ecx, 3                   // number of 8-element blocks
  jz @Tail

  vxorps ymm3, ymm3, ymm3      // zero for comparison

@BulkLoop:
  vmovups ymm0, [esi]          // load 8 src
  vmulps ymm1, ymm0, ymm2      // ymm1 = Slope * src
  vcmpps ymm4, ymm0, ymm3, 29  // GE_OQ: src >= 0
  vblendvps ymm1, ymm1, ymm0, ymm4  // select src if >=0 else Slope*src
  vmovups [edi], ymm1
  add esi, 32
  add edi, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  // Load Slope pointer from stack for scalar tail
  mov eax, [ebp+8]             // eax = pointer to Slope
  vbroadcastss xmm2, [eax]     // xmm2 = Slope (broadcast to all lanes for simplicity)
  vxorps xmm3, xmm3, xmm3      // zero

  xor ecx, ecx
@TailLoop:
  vmovss xmm0, [esi + ecx*4]
  vmulss xmm1, xmm0, xmm2      // Slope * src
  vcmpss xmm4, xmm0, xmm3, 29  // src >= 0
  vblendvps xmm1, xmm1, xmm0, xmm4  // select
  vmovss [edi + ecx*4], xmm1
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
  pop edi
  pop esi
end;

procedure _AVX512LeakyRelu( dst : PSingle; src : PSingle; N : Integer; const Slope : Single ); inline;
begin
  _AVX2LeakyRelu(dst, src, N, Slope);
end;

{-----------------------------------------------------------------------------
  AVX2 decode F16: dst[i] = half_to_single(src[i]) using F16C (vcvtph2ps).
  Processes 8 elements per loop. Tail (0..7) handled via temporary stack buffer.
  Parameters: EAX=dst, EDX=src, ECX=N.
-----------------------------------------------------------------------------}
procedure _AVX2DecodeF16(dst: PSingle; src: PWord; N: Integer);
asm
  push esi
  push edi
  push ebx

  mov edi, eax                 // EDI = dst
  mov esi, edx                 // ESI = src
  mov eax, ecx                 // EAX = N

  test eax, eax
  jle @Exit

  // Bulk = N - (N mod 8)
  mov edx, eax
  and edx, 7                   // tail = N mod 8
  sub eax, edx                 // bulk = N - tail
  mov ecx, eax
  shr ecx, 3                   // number of 8-element chunks
  jz @TailOnly

@BulkLoop:
  vmovups xmm0, [esi]          // load 8 halfs (16 bytes)
  vcvtph2ps ymm0, xmm0         // convert to 8 floats
  vmovups [edi], ymm0
  add esi, 16
  add edi, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@TailOnly:
  test edx, edx
  jz @Exit

  // ---- Tail processing: 0..7 elements ----
  sub esp, 16                  // allocate 16 bytes for temporary half buffer

  // Clear buffer
  vpxor xmm0, xmm0, xmm0
  vmovups [esp], xmm0

  // Copy tail halfs to buffer
  xor ecx, ecx
@CopyTail:
  movzx ebx, word ptr [esi + ecx*2]   // load half
  mov word ptr [esp + ecx*2], bx      // store to buffer
  inc ecx
  cmp ecx, edx
  jl @CopyTail

  // Convert 8 halfs from buffer to floats
  vmovups xmm0, [esp]
  vcvtph2ps ymm0, xmm0

  // Store only tail elements to dst (sequential single floats)
  xor ecx, ecx
@StoreTail:
  vmovss [edi + ecx*4], xmm0   // store low float
  vpsrldq xmm0, xmm0, 4        // shift right by 4 bytes to get next float
  inc ecx
  cmp ecx, edx
  jl @StoreTail

  add esp, 16                  // release temporary buffer

@Exit:
  vzeroupper
  pop ebx
  pop edi
  pop esi
  ret
end;

procedure _AVX512DecodeF16( dst : PSingle; src : PWord; N : Integer ); inline;
begin
  _AVX2DecodeF16(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 centered sum of squares: result = sum_i (src[i] - Mean)^2.
  Processes 16 elements per loop (2 YMM blocks), scalar tail.

  Parameter passing (Delphi register convention):
    EAX = src (PSingle)
    EDX = N   (Integer)   // Mean is floating-point, so it's on stack
    Mean at [EBP+8]       // Single, passed on stack (compiler generates prologue)

  Return value via ST(0) as caller expects (fstp).
-----------------------------------------------------------------------------}
function _AVX2SumSqrCentered(src: PSingle; Mean: Single; N: Integer): Single;
asm
  // No manual prologue; compiler generates push ebp; mov ebp, esp

  // Allocate temporary 4-byte stack space for broadcasting Mean
  sub esp, 4

  mov esi, eax                 // src
  mov eax, edx                 // N (from EDX)

  movss xmm3, [ebp+8]          // Mean (scalar)
  movss xmm4, [ebp+8]          // keep scalar for tail

  // Broadcast Mean to all lanes using memory operand
  movss [esp], xmm3            // store Mean to temp
  vbroadcastss ymm3, [esp]     // ymm3 = {Mean, Mean, ...}
  add esp, 4                   // release temp

  // Save registers after using them for parameters and temp stack
  push esi
  push edi
  push ebx

  test eax, eax
  jle @Zero

  // Bulk = N - (N mod 16)
  mov edx, eax
  and edx, 15                  // tail
  sub eax, edx                 // bulk
  mov ecx, eax
  shr ecx, 4                   // number of 16-element blocks
  jz @Tail

  vxorps ymm0, ymm0, ymm0
  vxorps ymm1, ymm1, ymm1

@BulkLoop:
  vmovups ymm2, [esi]          // first 8
  vsubps  ymm2, ymm2, ymm3
  vfmadd231ps ymm0, ymm2, ymm2

  vmovups ymm2, [esi+32]       // second 8
  vsubps  ymm2, ymm2, ymm3
  vfmadd231ps ymm1, ymm2, ymm2

  add esi, 64
  dec ecx
  jnz @BulkLoop

  // Combine and reduce to scalar
  vaddps ymm0, ymm0, ymm1
  vextractf128 xmm1, ymm0, 1
  vaddps xmm0, xmm0, xmm1
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0     // xmm0[0] = sum of bulk

@Tail:
  test edx, edx
  jz @Return

  // Scalar tail
  movss xmm3, xmm4             // reload Mean
  xor ecx, ecx
@TailLoop:
  vmovss xmm1, [esi + ecx*4]
  vsubss xmm1, xmm1, xmm3
  vmulss xmm1, xmm1, xmm1
  vaddss xmm0, xmm0, xmm1
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Return:
  vzeroupper
  // Store result to FPU stack
  sub esp, 4
  vmovss [esp], xmm0
  fld dword ptr [esp]
  add esp, 4
  pop ebx
  pop edi
  pop esi
  jmp @Exit

@Zero:
  xorps xmm0, xmm0
  vzeroupper
  sub esp, 4
  vmovss [esp], xmm0
  fld dword ptr [esp]
  add esp, 4
  pop ebx
  pop edi
  pop esi

@Exit:

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
-----------------------------------------------------------------------------}
procedure _AVX2AdamDelta(PtrDelta, PtrM, PtrV: PSingle;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR: Single;
  NumElements: Integer);
asm
  push esi
  push edi
  push ebx
  sub esp, 4

  mov edi, eax                 // edi = PtrDelta
  mov ebx, edx                 // ebx = PtrM
  mov esi, ecx                 // esi = PtrV
  mov eax, [ebp+8]             // eax = NumElements

  test eax, eax
  jle @Exit

  // Broadcast constants (InvOmB2D is not used in this implementation)
  vbroadcastss ymm0, dword ptr [ebp+36]  // Beta1
  vbroadcastss ymm1, dword ptr [ebp+32]  // OmBeta1
  vbroadcastss ymm2, dword ptr [ebp+28]  // Beta2
  vbroadcastss ymm3, dword ptr [ebp+24]  // OmBeta2
  vbroadcastss ymm5, dword ptr [ebp+16]  // Epsilon
  vbroadcastss ymm6, dword ptr [ebp+12]  // kLR

  mov edx, eax
  and edx, 7
  sub eax, edx
  mov ecx, eax
  shr ecx, 3
  jz @Tail

@BulkLoop:
  // Reload Beta1 (ymm0 is overwritten later)
  vbroadcastss ymm0, dword ptr [ebp+36]

  // ---- m = Beta1*m + OmBeta1*g ----
  vmovups ymm7, [edi]                     // g
  vmulps ymm7, ymm7, ymm1
  vfmadd231ps ymm7, ymm0, [ebx]
  vmovups [ebx], ymm7

  // ---- v = Beta2*v + OmBeta2*(g*g) ----
  vmovups ymm7, [edi]                     // reload g
  vmulps ymm7, ymm7, ymm7
  vmulps ymm7, ymm7, ymm3
  vfmadd231ps ymm7, ymm2, [esi]
  vmovups [esi], ymm7

  // ---- delta = (kLR*m) / (sqrt(v) + Epsilon) ----
  vmovups ymm7, [esi]                     // v
  vsqrtps ymm7, ymm7
  vaddps ymm7, ymm7, ymm5

  vmovups ymm0, [ebx]                     // m
  vmulps ymm0, ymm0, ymm6                 // numerator
  vdivps ymm0, ymm0, ymm7
  vmovups [edi], ymm0

  add edi, 32
  add ebx, 32
  add esi, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  mov ecx, edx                 // tail count
  xor edx, edx                 // index

@TailLoop:
  // ---- m = Beta1*m + OmBeta1*g ----
  vmovss xmm7, [edi + edx*4]            // g
  vmulss xmm7, xmm7, [ebp+32]           // OmBeta1
  vmovss xmm6, [ebx + edx*4]            // old m
  vmulss xmm6, xmm6, [ebp+36]           // Beta1
  vaddss xmm7, xmm7, xmm6
  vmovss [ebx + edx*4], xmm7

  // ---- v = Beta2*v + OmBeta2*(g*g) ----
  vmovss xmm7, [edi + edx*4]            // g
  vmulss xmm7, xmm7, xmm7               // g^2
  vmulss xmm7, xmm7, [ebp+24]           // OmBeta2
  vmovss xmm6, [esi + edx*4]            // old v
  vmulss xmm6, xmm6, [ebp+28]           // Beta2
  vaddss xmm7, xmm7, xmm6
  vmovss [esi + edx*4], xmm7

  // ---- delta = (kLR*m) / (sqrt(v) + Epsilon) ----
  vmovss xmm7, [esi + edx*4]            // v
  vsqrtss xmm7, xmm7, xmm7
  vaddss xmm7, xmm7, [ebp+16]           // Epsilon
  vmovss xmm6, [ebx + edx*4]            // m
  vmulss xmm6, xmm6, [ebp+12]           // kLR
  vdivss xmm6, xmm6, xmm7
  vmovss [edi + edx*4], xmm6

  inc edx
  cmp edx, ecx
  jl @TailLoop

@Exit:
  vzeroupper
  add esp, 4
  pop ebx
  pop edi
  pop esi
end;

procedure _AVX512AdamDelta( PtrDelta, PtrM, PtrV : PSingle;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR : Single;
  NumElements : Integer ); inline;
begin
  _AVX2AdamDelta(PtrDelta, PtrM, PtrV, Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR, NumElements);
end;

{-----------------------------------------------------------------------------
  AVX2 Adafactor step (32-bit, register calling convention).
  Updates per element:
    v := Beta2 * v + (k * d * d + c)
    d := d / (sqrt(v) + Epsilon)
  Processes 8 elements per loop (YMM), scalar tail.
  Win32 BASM limitation: only ymm0..ymm7 are available.
-----------------------------------------------------------------------------}
procedure _AVX2AdafactorDelta( PtrDelta, PtrV : PSingle;
  Beta2, k, c, Epsilon : Single; NumElements : Integer );
asm
  push esi
  push edi
  push ebx

  mov edi, eax                     // PtrDelta
  mov ebx, edx                     // PtrV
  mov esi, ecx                     // NumElements (in ECX)

  test esi, esi
  jle @Exit

  // ---- Broadcast constants from stack ----
  vbroadcastss ymm0, dword ptr [ebp+20]   // Beta2
  vbroadcastss ymm1, dword ptr [ebp+16]   // k
  vbroadcastss ymm2, dword ptr [ebp+12]   // c
  vbroadcastss ymm3, dword ptr [ebp+8]    // Epsilon

  // ---- Bulk count ----
  mov eax, esi
  and eax, 7                    // tail = NumElements mod 8
  sub esi, eax                  // bulk = NumElements - tail
  mov ecx, esi
  shr ecx, 3                    // number of 8-element blocks
  jz @Tail

@BulkLoop:
  vmovups ymm7, [edi]           // d

  vmulps ymm6, ymm7, ymm7       // (d*d)
  vmulps ymm6, ymm6, ymm1       // (k*d*d)
  vaddps ymm6, ymm6, ymm2       // (+c)
  vmovups ymm5, [ebx]           // old v

  vmulps ymm5, ymm5, ymm0       // (Beta2*v)
  vaddps ymm6, ymm6, ymm5       // (new v)
  vmovups [ebx], ymm6
  vsqrtps ymm6, ymm6            // sqrt(v)
  vaddps ymm6, ymm6, ymm3       // (+Epsilon)
  vmulps ymm7, ymm7, ymm1       //(k*d)
  vdivps ymm7, ymm7, ymm6       //(k*d / denominator)
  vmovups [edi], ymm7

  add edi, 32
  add ebx, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test eax, eax
  jz @Exit

  // ---- 标量尾部（与主循环偏移一致） ----
  movss xmm0, dword ptr [ebp+20]   // Beta2
  movss xmm1, dword ptr [ebp+16]   // k
  movss xmm2, dword ptr [ebp+12]   // c
  movss xmm3, dword ptr [ebp+8]    // Epsilon

  xor ecx, ecx
@TailLoop:
  movss xmm7, [edi + ecx*4]       // d
  movss xmm6, xmm7                // d
  mulss xmm6, xmm7                // d*d
  mulss xmm6, xmm1                // k * d*d
  addss xmm6, xmm2                // + c
  movss xmm5, [ebx + ecx*4]       // old v
  mulss xmm5, xmm0                // Beta2 * v
  addss xmm6, xmm5                // new v
  movss [ebx + ecx*4], xmm6

  movss xmm5, xmm6                // copy v
  sqrtss xmm5, xmm5               // sqrt(v)
  addss xmm5, xmm3                // + Epsilon

  movss xmm6, xmm7                // copy d
  mulss xmm6, xmm1                // k * d
  divss xmm6, xmm5                // k*d / denominator
  movss [edi + ecx*4], xmm6

  inc ecx
  cmp ecx, eax
  jl @TailLoop

@Exit:
  vzeroupper
  pop ebx
  pop edi
  pop esi
end;

procedure _AVX512AdafactorDelta( PtrDelta, PtrV : PSingle;
  Beta2, k, c, Epsilon : Single; NumElements : Integer ); inline;
begin
  _AVX2AdafactorDelta(PtrDelta, PtrV, Beta2, k, c, Epsilon, NumElements);
end;

{-----------------------------------------------------------------------------
  AVX2 clamp absolute: dst[i] := clamp(dst[i], -Value, +Value).
  Uses 256-bit YMM, processes 8 elements per loop.
  Parameters: EAX=PtrA, EDX=NumElements, Value at [EBP+8].
-----------------------------------------------------------------------------}
procedure _AVX2ClampAbs( PtrA : PSingle; Value : Single; NumElements : Integer );
asm
  push esi
  push edi

  mov esi, eax                // PtrA
  mov eax, edx                // NumElements (in EDX)

  test eax, eax
  jle @Exit

  // Broadcast Value from stack
  vbroadcastss ymm0, [ebp+8]  // ymm0 = +Value
  vxorps ymm1, ymm1, ymm1
  vsubps ymm1, ymm1, ymm0     // ymm1 = -Value

  // Bulk count
  mov edx, eax
  and edx, 7
  sub eax, edx
  mov ecx, eax
  shr ecx, 3
  jz @Tail

@BulkLoop:
  vmovups ymm2, [esi]
  vmaxps ymm2, ymm1, ymm2
  vminps ymm2, ymm0, ymm2
  vmovups [esi], ymm2
  add esi, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  // Scalar tail using SSE (non-VEX) instructions
  movss xmm2, [ebp+8]         // +Value
  movss xmm3, [ebp+8]         // copy
  xorps xmm3, xmm3
  subss xmm3, xmm2            // -Value

  xor ecx, ecx
@TailLoop:
  movss xmm4, [esi + ecx*4]
  maxss xmm4, xmm3
  minss xmm4, xmm2
  movss [esi + ecx*4], xmm4
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
  pop edi
  pop esi
end;

procedure _AVX512ClampAbs( PtrA : PSingle; Value : Single; NumElements : Integer ); inline;
begin
  _AVX2ClampAbs(PtrA, Value, NumElements);
end;

{-----------------------------------------------------------------------------
  AVX2 Lion optimizer step (32-bit) - Standard syntax, no db.
  All instructions use standard mnemonics. vcmpps predicates: 6 = NLE (c > 0),
  1 = LT (c < 0). Constants reloaded each iteration.
-----------------------------------------------------------------------------}
procedure _AVX2LionDelta(PtrDelta, PtrM: PSingle;
  Beta1, k1, Beta2, k2, NegLR, PosLR: Single;
  NumElements: Integer);
asm
  push edi
  push ebx

  // Allocate 64 bytes for saving non-volatile YMM6 and YMM7
  sub esp, 64

  // Save non-volatile YMM6 and YMM7
  vmovups [esp], ymm6
  vmovups [esp+32], ymm7

  // Working pointers
  mov edi, eax                  // edi = PtrDelta
  mov ebx, edx                  // ebx = PtrM
  mov eax, ecx                  // eax = NumElements

  test eax, eax
  jle @Exit

  // Split into bulk (8-element blocks) and tail
  mov edx, eax
  and edx, 7                    // tail count
  sub eax, edx
  shr eax, 3                    // number of full blocks
  jz @Tail
  mov ecx, eax                  // loop counter

  // Broadcast constants (using correct stack offsets)
  vbroadcastss ymm2, [ebp+28]   // Beta1
  vbroadcastss ymm3, [ebp+24]   // k1
  vbroadcastss ymm4, [ebp+20]   // Beta2
  vbroadcastss ymm5, [ebp+16]   // k2

@BulkLoop:
  vmovups ymm0, [edi]           // g
  vmovups ymm1, [ebx]           // m

  // c = Beta1*m + k1*g  -> ymm6
  vmulps ymm6, ymm1, ymm2
  vfmadd231ps ymm6, ymm0, ymm3

  // m_new = Beta2*m + k2*g  -> ymm7
  vmovups ymm7, ymm1
  vmulps ymm7, ymm7, ymm4
  vfmadd231ps ymm7, ymm0, ymm5
  vmovups [ebx], ymm7

  // Build delta based on sign of c
  vxorps ymm7, ymm7, ymm7
  vcmpps ymm0, ymm6, ymm7, 6    // c > 0
  vcmpps ymm1, ymm6, ymm7, 1    // c < 0

  vbroadcastss ymm6, [ebp+12]   // NegLR
  vbroadcastss ymm7, [ebp+8]    // PosLR

  vandps ymm0, ymm0, ymm6
  vandps ymm1, ymm1, ymm7
  vorps ymm0, ymm0, ymm1
  vmovups [edi], ymm0

  add edi, 32
  add ebx, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  xor ecx, ecx                  // index

@TailLoop:
  vmovss xmm0, [edi + ecx*4]    // g
  vmovss xmm1, [ebx + ecx*4]    // m

  // c = Beta1*m + k1*g
  vmovss xmm2, [ebp+28]         // Beta1
  vmulss xmm2, xmm2, xmm1
  vmovss xmm3, [ebp+24]         // k1
  vmulss xmm3, xmm3, xmm0
  vaddss xmm2, xmm2, xmm3

  // m_new = Beta2*m + k2*g
  vmovss xmm3, [ebp+20]         // Beta2
  vmulss xmm3, xmm3, xmm1
  vmovss xmm4, [ebp+16]         // k2
  vmulss xmm4, xmm4, xmm0
  vaddss xmm3, xmm3, xmm4
  vmovss [ebx + ecx*4], xmm3

  // delta selection
  vxorps xmm4, xmm4, xmm4
  comiss xmm2, xmm4
  jg @Greater
  jl @Less
  xorps xmm0, xmm0
  jmp @Store
@Greater:
  vmovss xmm0, [ebp+12]         // NegLR
  jmp @Store
@Less:
  vmovss xmm0, [ebp+8]          // PosLR
@Store:
  vmovss [edi + ecx*4], xmm0

  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
  // Restore YMM6 and YMM7
  vmovups ymm6, [esp]
  vmovups ymm7, [esp+32]
  add esp, 64
  pop ebx
  pop edi
end;

procedure _AVX512LionDelta( PtrDelta, PtrM : PSingle; Beta1, k1, Beta2, k2, NegLR, PosLR : Single; NumElements : Integer ); inline;
begin
  _AVX2LionDelta(PtrDelta, PtrM, Beta1, k1, Beta2, k2, NegLR, PosLR, NumElements);
end;

{-----------------------------------------------------------------------------
  AVX2 max value and first occurrence index (32-bit, 8 elements per loop).
  Parameters: EAX=PtrA, EDX=NumElements, ECX=Pos (out pointer).
  Returns max value via ST(0).
-----------------------------------------------------------------------------}
function _AVX2GetMaxPos(PtrA: PSingle; NumElements: Integer; out Position: Integer): Single;
asm
  push ebx
  push esi
  push edi
  sub esp, 128

  mov esi, eax                // esi = PtrA
  mov ebx, ecx                // ebx = @Position (out parameter)
  mov eax, edx                // eax = NumElements
  mov [esp+124], esi          // save original PtrA

  test eax, eax
  jle @Zero

  // Split into bulk (multiples of 8) and tail (0..7)
  mov edx, eax
  and edx, 7                  // tail count
  sub eax, edx                // bulk element count
  mov [esp+120], eax          // store bulk count (in elements)

  // Load lane indices 0..7 (seed)
  vmovdqu ymm1, yword ptr [cAVXArgLaneSeed]

  // Generate step vector of 8's
  mov [esp+64], 8
  vpbroadcastd ymm2, [esp+64]

  // Load first 8 elements as initial candidates
  vmovups ymm0, [esi]         // ymm0 = max values
  add esi, 32

  mov ecx, [esp+120]
  shr ecx, 3                  // number of full 8-element blocks
  dec ecx                     // we already processed the first block
  jz @Fold                    // if only one block, skip bulk loop

@BulkLoop:
  vmovups ymm3, [esi]         // load next 8 values
  vpaddd ymm4, ymm1, ymm2     // new indices = current + 8
  vcmpps ymm5, ymm3, ymm0, 22 // ymm3 > ymm0 ?  (condition code 22 = greater-than)
  vblendvps ymm0, ymm0, ymm3, ymm5   // update max values
  vblendvps ymm1, ymm1, ymm4, ymm5   // update corresponding indices
  add esi, 32
  dec ecx
  jnz @BulkLoop

@Fold:
  // Reduce 8 lanes to scalar while preserving the index of the maximum
  vextractf128 xmm2, ymm0, 1   // high 4 values
  vextracti128 xmm3, ymm1, 1   // high 4 indices

  vcmpps xmm5, xmm0, xmm2, 22  // low > high ?
  vblendvps xmm6, xmm3, xmm1, xmm5 // select indices
  vmaxps xmm0, xmm0, xmm2      // max values (low/high combined)

  // Reduce to 2 lanes
  vpshufd xmm2, xmm0, $55      // shuffle: lane 1 broadcast
  vpshufd xmm3, xmm6, $55
  vcmpps xmm5, xmm0, xmm2, 22
  vblendvps xmm4, xmm3, xmm6, xmm5
  vmaxss xmm0, xmm0, xmm2
  vmovdqa xmm6, xmm4

  // Reduce to 1 lane
  vpshufd xmm2, xmm0, $AA
  vpshufd xmm3, xmm6, $AA
  vcmpps xmm5, xmm0, xmm2, 22
  vblendvps xmm4, xmm3, xmm6, xmm5
  vmaxss xmm0, xmm0, xmm2
  vmovdqa xmm6, xmm4

  vpshufd xmm2, xmm0, $FF
  vpshufd xmm3, xmm6, $FF
  vcmpps xmm5, xmm0, xmm2, 22
  vblendvps xmm4, xmm3, xmm6, xmm5
  vmaxss xmm0, xmm0, xmm2
  vmovdqa xmm6, xmm4

  // Now xmm0[0] = maximum value, xmm6[0] = corresponding index
  vmovss [esp], xmm0
  vmovd [esp+96], xmm6

  vzeroupper

  mov eax, [esp+96]           // load index of maximum from bulk

  // ---- Tail handling (0..7 elements) ----
  test edx, edx
  jz @TailDone

  // FIX: pointer offset must be in bytes: bulk_count * 4
  mov esi, [esp+124]          // restore original PtrA
  mov edi, [esp+120]          // bulk element count (base index)
  lea esi, [esi + edi*4]      // point to the start of tail elements

  xor ecx, ecx
@TailLoop:
  vmovss xmm1, [esi + ecx*4]
  comiss xmm1, xmm0           // compare current with current maximum
  jbe @TailSkip               // if <=, skip (keep first occurrence)
  movss xmm0, xmm1
  lea eax, [edi + ecx]        // new position = bulk_count + tail_index
@TailSkip:
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@TailDone:
  mov [ebx], eax              // store final position
  jmp @Exit

@Zero:
  xorps xmm0, xmm0
  mov dword ptr [ebx], 0

@Exit:
  vmovss [esp], xmm0
  fld dword ptr [esp]         // return Single via ST(0)
  add esp, 128
  pop edi
  pop esi
  pop ebx
end;

function _AVX512GetMaxPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;
begin
  Result := _AVX2GetMaxPos(PtrA, NumElements, Position);
end;

{-----------------------------------------------------------------------------
  AVX2 min value and first occurrence index (32-bit).
  Processes 8 elements per loop (same as GetMaxPos).
  Parameters: EAX=PtrA, EDX=NumElements, ECX=Position (out).
  Returns: min value via ST(0).
-----------------------------------------------------------------------------}
function _AVX2GetMinPos(PtrA: PSingle; NumElements: Integer; out Position: Integer): Single;
asm
  push ebx
  push esi
  push edi
  sub esp, 128

  mov esi, eax                // esi = PtrA
  mov ebx, ecx                // ebx = @Position (out parameter)
  mov eax, edx                // eax = NumElements
  mov [esp+124], esi          // save original PtrA

  test eax, eax
  jle @Zero

  // Split into bulk (multiples of 8) and tail (0..7)
  mov edx, eax
  and edx, 7                  // tail count
  sub eax, edx                // bulk element count
  mov [esp+120], eax          // store bulk count (in elements)

  // Load lane indices 0..7 (seed)
  vmovdqu ymm1, yword ptr [cAVXArgLaneSeed]

  // Generate step vector of 8's
  mov [esp+64], 8
  vpbroadcastd ymm2, [esp+64]

  // Load first 8 elements as initial candidates
  vmovups ymm0, [esi]         // ymm0 = min values
  add esi, 32

  mov ecx, [esp+120]
  shr ecx, 3                  // number of full 8-element blocks
  dec ecx                     // we already processed the first block
  jz @Fold                    // if only one block, skip bulk loop

@BulkLoop:
  vmovups ymm3, [esi]         // load next 8 values
  vpaddd ymm4, ymm1, ymm2     // new indices = current + 8
  vcmpps ymm5, ymm3, ymm0, 17 // ymm3 < ymm0 ?
  vblendvps ymm0, ymm0, ymm3, ymm5   // update min values
  vblendvps ymm1, ymm1, ymm4, ymm5   // update corresponding indices
  add esi, 32
  dec ecx
  jnz @BulkLoop

@Fold:
  // Reduce 8 lanes to scalar while preserving the index of the minimum
  vextractf128 xmm2, ymm0, 1   // high 4 values
  vextracti128 xmm3, ymm1, 1   // high 4 indices

  vcmpps xmm5, xmm0, xmm2, 17  // low < high ?
  vblendvps xmm6, xmm3, xmm1, xmm5 // select indices
  vminps xmm0, xmm0, xmm2      // min values (low/high combined)

  // Reduce to 2 lanes
  vpshufd xmm2, xmm0, $55      // shuffle: lane 1 broadcast
  vpshufd xmm3, xmm6, $55
  vcmpps xmm5, xmm0, xmm2, 17
  vblendvps xmm4, xmm3, xmm6, xmm5
  vminss xmm0, xmm0, xmm2
  vmovdqa xmm6, xmm4

  // Reduce to 1 lane
  vpshufd xmm2, xmm0, $AA
  vpshufd xmm3, xmm6, $AA
  vcmpps xmm5, xmm0, xmm2, 17
  vblendvps xmm4, xmm3, xmm6, xmm5
  vminss xmm0, xmm0, xmm2
  vmovdqa xmm6, xmm4

  vpshufd xmm2, xmm0, $FF
  vpshufd xmm3, xmm6, $FF
  vcmpps xmm5, xmm0, xmm2, 17
  vblendvps xmm4, xmm3, xmm6, xmm5
  vminss xmm0, xmm0, xmm2
  vmovdqa xmm6, xmm4

  // Now xmm0[0] = minimum value, xmm6[0] = corresponding index
  vmovss [esp], xmm0
  vmovd [esp+96], xmm6

  vzeroupper

  mov eax, [esp+96]           // load index of minimum from bulk

  // ---- Tail handling (0..7 elements) ----
  test edx, edx
  jz @TailDone

  // FIX: pointer offset must be in bytes: bulk_count * 4
  mov esi, [esp+124]          // restore original PtrA
  mov edi, [esp+120]          // bulk element count (base index)
  lea esi, [esi + edi*4]      // point to the start of tail elements

  xor ecx, ecx
@TailLoop:
  vmovss xmm1, [esi + ecx*4]
  comiss xmm1, xmm0           // compare current with current minimum
  jae @TailSkip               // if >=, skip (keep first occurrence)
  movss xmm0, xmm1
  lea eax, [edi + ecx]        // new position = bulk_count + tail_index
@TailSkip:
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@TailDone:
  mov [ebx], eax              // store final position
  jmp @Exit

@Zero:
  xorps xmm0, xmm0
  mov dword ptr [ebx], 0

@Exit:
  vmovss [esp], xmm0
  fld dword ptr [esp]         // return Single via ST(0)
  add esp, 128
  pop edi
  pop esi
  pop ebx
end;

function _AVX512GetMinPos( PtrA : PSingle; NumElements : Integer; out Position : Integer ) : Single; inline;
begin
  Result := _AVX2GetMinPos(PtrA, NumElements, Position);
end;

{-----------------------------------------------------------------------------
  AVX2 max absolute value and first occurrence index (32-bit, 8 elements/loop).
  Parameters: EAX = PtrA, EDX = NumElements, ECX = Position (out pointer).
  Returns: max absolute value via ST(0).
  This is _AVX2GetMaxPos with absolute value applied to loaded elements.
-----------------------------------------------------------------------------}
function _AVX2GetMaxAbsPos(PtrA: PSingle; NumElements: Integer; out Position: Integer): Single;
asm
  push ebx
  push esi
  push edi
  sub esp, 128

  mov esi, eax                // esi = PtrA
  mov ebx, ecx                // ebx = @Position (out parameter)
  mov eax, edx                // eax = NumElements
  mov [esp+124], esi          // save original PtrA

  test eax, eax
  jle @Zero

  // Split into bulk (multiples of 8) and tail (0..7)
  mov edx, eax
  and edx, 7                  // tail count
  sub eax, edx                // bulk element count
  mov [esp+120], eax          // store bulk count (in elements)

  // Load absolute mask into ymm7 (clears sign bit)
  vmovdqu ymm7, yword ptr [cAVXArgAbsMask]

  // Load lane offsets 0..7
  vmovdqu ymm1, yword ptr [cAVXArgLaneSeed]

  // Generate step vector of 8's
  mov [esp+64], 8
  vpbroadcastd ymm2, [esp+64]

  vmovdqa ymm6, ymm2          // ymm6 = {8,8,8,8,8,8,8,8} (base for indices)

  // Seed with first 8 elements, take absolute values
  vmovups ymm0, [esi]
  vandps ymm0, ymm0, ymm7     // abs

  add esi, 32
  mov ecx, [esp+120]
  shr ecx, 3                  // number of full blocks
  dec ecx                     // first block already processed
  jz @Fold

@BulkLoop:
  vmovups ymm3, [esi]
  vandps ymm3, ymm3, ymm7     // abs

  vmovdqu ymm4, yword ptr [cAVXArgLaneSeed]   // lane offsets
  vpaddd ymm4, ymm6, ymm4                     // absolute indices = base + offset
  vpaddd ymm6, ymm6, ymm2                     // base += 8 for next batch

  vcmpps ymm5, ymm3, ymm0, 6   // ymm3 > ymm0 ?
  vblendvps ymm0, ymm0, ymm3, ymm5
  vblendvps ymm1, ymm1, ymm4, ymm5

  add esi, 32
  dec ecx
  jnz @BulkLoop

@Fold:
  // Reduce 8 lanes to scalar max and corresponding index
  vextractf128 xmm2, ymm0, 1
  vextracti128 xmm3, ymm1, 1

  vcmpps xmm5, xmm0, xmm2, 6
  vblendvps xmm6, xmm3, xmm1, xmm5
  vmaxps xmm0, xmm0, xmm2

  vpshufd xmm2, xmm0, $55
  vpshufd xmm3, xmm6, $55
  vcmpps xmm5, xmm0, xmm2, 6
  vblendvps xmm4, xmm3, xmm6, xmm5
  vmaxss xmm0, xmm0, xmm2
  vmovdqa xmm6, xmm4

  vpshufd xmm2, xmm0, $AA
  vpshufd xmm3, xmm6, $AA
  vcmpps xmm5, xmm0, xmm2, 6
  vblendvps xmm4, xmm3, xmm6, xmm5
  vmaxss xmm0, xmm0, xmm2
  vmovdqa xmm6, xmm4

  vpshufd xmm2, xmm0, $FF
  vpshufd xmm3, xmm6, $FF
  vcmpps xmm5, xmm0, xmm2, 6
  vblendvps xmm4, xmm3, xmm6, xmm5
  vmaxss xmm0, xmm0, xmm2
  vmovdqa xmm6, xmm4

  // Now xmm0[0] = maximum absolute value, xmm6[0] = corresponding index
  vmovss [esp], xmm0
  vmovd [esp+96], xmm6

  vzeroupper

  mov eax, [esp+96]           // index from bulk

  // ---- Tail handling (0..7 elements) ----
  test edx, edx
  jz @TailDone

  // FIX: pointer offset must be in bytes: bulk_count * 4
  mov esi, [esp+124]          // restore original PtrA
  mov edi, [esp+120]          // bulk element count (base index)
  lea esi, [esi + edi*4]      // point to start of tail elements

  xor ecx, ecx
@TailLoop:
  vmovss xmm1, [esi + ecx*4]
  vandps xmm1, xmm1, xmm7     // absolute value (xmm7 is low part of ymm7)
  comiss xmm1, xmm0           // compare with current max
  jbe @TailSkip               // if <=, keep first occurrence
  movss xmm0, xmm1
  lea eax, [edi + ecx]        // new position = bulk_count + tail_index
@TailSkip:
  inc ecx
  cmp ecx, edx
  jl @TailLoop

@TailDone:
  mov [ebx], eax              // store final position
  jmp @Exit

@Zero:
  xorps xmm0, xmm0
  mov dword ptr [ebx], 0

@Exit:
  vmovss [esp], xmm0
  fld dword ptr [esp]         // return Single via ST(0)
  add esp, 128
  pop edi
  pop esi
  pop ebx
end;

function _AVX512GetMaxAbsPos( PtrA : PSingle; NumElements : integer; out Position : integer ) : single; inline;
begin
  Result := _AVX2GetMaxAbsPos(PtrA, NumElements, Position);
end;

{-----------------------------------------------------------------------------
  AVX2 add scalar (32-bit): dst[i] := dst[i] + Value.
  Processes 32 elements per loop (4 YMM blocks), scalar tail.

  Calling convention: register (Delphi default).
  Parameter passing:
    EAX = PtrA  (pointer to array of Single)
    Value = passed on the stack (since it's a Single, not in integer registers)
    EDX = N     (integer, because float doesn't use EDX)

  The compiler automatically sets up EBP as frame pointer, so [ebp+8] points to Value.
-----------------------------------------------------------------------------}
procedure _AVX2AddScalar(PtrA: PSingle; Value: Single; N: integer);
asm
  push esi
  push edi

  mov esi, eax                 // esi = PtrA
  mov edi, edx                 // edi = N (N is in EDX)

  test edi, edi
  jle @Exit

  // Load Value from stack (at [ebp+8]) and broadcast to YMM7
  vbroadcastss ymm7, [ebp+8]

  // Bulk processing: 32 elements per loop (4 YMM registers × 8 lanes)
  mov eax, edi
  and eax, 31                  // tail = N mod 32
  sub edi, eax                 // bulk = N - tail (multiple of 32)

  mov ecx, edi
  shr ecx, 5                   // number of 32-element iterations
  jz @Tail

@BulkLoop:
  vaddps ymm0, ymm7, [esi]
  vaddps ymm1, ymm7, [esi+32]
  vaddps ymm2, ymm7, [esi+64]
  vaddps ymm3, ymm7, [esi+96]
  vmovups [esi], ymm0
  vmovups [esi+32], ymm1
  vmovups [esi+64], ymm2
  vmovups [esi+96], ymm3
  add esi, 128
  dec ecx
  jnz @BulkLoop

  vzeroupper                    // avoid AVX-SSE transition penalty

@Tail:
  test eax, eax
  jz @Exit

  xor ecx, ecx
@TailLoop:
  vmovss xmm0, [esi + ecx*4]
  vaddss xmm0, xmm0, xmm7      // xmm7 holds broadcasted scalar (low lane)
  vmovss [esi + ecx*4], xmm0
  inc ecx
  cmp ecx, eax
  jl @TailLoop

@Exit:
  vzeroupper
  pop edi
  pop esi
end;

procedure _AVX512AddScalar( PtrA : PSingle; Value : single; N : integer ); inline;
begin
  _AVX2AddScalar(PtrA, Value, N);
end;

{-----------------------------------------------------------------------------
  AVX2 exp-shift-sum: dst[i] = exp(src[i] - Shift), returns sum of dst.
  Bulk: 8 elements per loop. Tail: scalar.
  Parameters: EAX=dst, EDX=src, ECX=N, [EBP+8]=Shift (compiler-provided EBP).
-----------------------------------------------------------------------------}
function _AVX2ExpShiftSum(dst: PSingle; src: PSingle; Shift: Single; N: Integer): Single;
asm
  push esi
  push edi
  push ebx

  sub esp, 32

  mov esi, edx                 // src
  mov edi, eax                 // dst
  mov eax, ecx                 // N

  test eax, eax
  jle @Zero

  mov edx, eax
  and edx, 7                   // tail count
  sub eax, edx                 // bulk count

  // Load Shift (used throughout, never modified)
  lea ebx, [ebp+8]
  vbroadcastss ymm5, [ebx]

  // Accumulator ymm6 is preserved across loop iterations
  vxorps ymm6, ymm6, ymm6

  mov ecx, eax
  shr ecx, 3
  jz @TailBulkDone

@BulkLoop:
  // ---------- Reload constants destroyed in the loop ----------
  vbroadcastss ymm0, dword ptr [cAVXExpHi]    // for min/max
  vbroadcastss ymm1, dword ptr [cAVXExpLo]
  vbroadcastss ymm2, dword ptr [cAVXLog2e]    // log2(e)
  vbroadcastss ymm3, dword ptr [cAVXLn2]      // ln(2)
  vpbroadcastd ymm4, dword ptr [cAVXExp127]   // 127
  vbroadcastss ymm5, dword ptr [ebp+8]   // reload Shift

  // Load source and subtract shift
  vmovups ymm7, [esi]
  vsubps  ymm7, ymm7, ymm5
  vminps  ymm7, ymm7, ymm0
  vmaxps  ymm7, ymm7, ymm1

  // t = x * log2e  (use ymm0 as temp, not ymm6)
  vmulps  ymm0, ymm7, ymm2
  vroundps ymm1, ymm0, 0         // ymm1 = k = round(t)
  vsubps  ymm0, ymm0, ymm1       // ymm0 = f = t - k
  vmulps  ymm2, ymm0, ymm3       // ymm2 = g = f * ln2

  // Polynomial: 2^f = P6*g^6 + ... + P0
  // Use ymm3 as accumulator, ymm0 as temp
  vbroadcastss ymm3, dword ptr [cAVXExpP6]
  vbroadcastss ymm0, dword ptr [cAVXExpP5]
  vfmadd213ps ymm3, ymm2, ymm0
  vbroadcastss ymm0, dword ptr [cAVXExpP4]
  vfmadd213ps ymm3, ymm2, ymm0
  vbroadcastss ymm0, dword ptr [cAVXExpP3]
  vfmadd213ps ymm3, ymm2, ymm0
  vbroadcastss ymm0, dword ptr [cAVXExpP2]
  vfmadd213ps ymm3, ymm2, ymm0
  vbroadcastss ymm0, dword ptr [cAVXExpP1]
  vfmadd213ps ymm3, ymm2, ymm0
  vbroadcastss ymm0, dword ptr [cAVXExpP0]
  vfmadd213ps ymm3, ymm2, ymm0   // ymm3 = 2^f

  // 2^k from integer k (ymm1 still holds k as float)
  vcvtps2dq ymm1, ymm1
  vpaddd ymm1, ymm1, ymm4
  vpslld ymm1, ymm1, 23

  vmulps ymm7, ymm3, ymm1
  vmovups [edi], ymm7

  // Accumulate into ymm6 (ymm6 is never overwritten in the loop)
  vaddps ymm6, ymm6, ymm7

  add esi, 32
  add edi, 32
  dec ecx

  jnz @BulkLoop

@TailBulkDone:
  // Reduce bulk sum to scalar
  vextractf128 xmm0, ymm6, 1
  vaddps xmm0, xmm0, xmm6
  vhaddps xmm0, xmm0, xmm0
  vhaddps xmm0, xmm0, xmm0
  movss [esp], xmm0            // bulk sum at [esp]

  vzeroupper

  test edx, edx
  jz @Done

  // ---------- Tail (0..7) scalar ----------
  vbroadcastss xmm0, dword ptr [cAVXExpHi]
  vbroadcastss xmm1, dword ptr [cAVXExpLo]
  vbroadcastss xmm2, dword ptr [cAVXLog2e]
  vbroadcastss xmm3, dword ptr [cAVXLn2]
  vmovd xmm4, [cAVXExp127]          // use xmm4 for scalar 127
  vpbroadcastd xmm4, xmm4           // broadcast to all lanes of xmm4 (for consistency)

  lea ebx, [ebp+8]
  vbroadcastss xmm5, [ebx]

  xor ecx, ecx
  vxorps xmm6, xmm6, xmm6      // tail sum

@TailLoop:
  vmovss xmm7, [esi + ecx*4]
  vsubss xmm7, xmm7, xmm5
  vminss xmm7, xmm7, xmm0
  vmaxss xmm7, xmm7, xmm1

  vmulss xmm0, xmm7, xmm2
  vroundss xmm1, xmm0, xmm0, 0
  vsubss xmm0, xmm0, xmm1

  vmulss xmm2, xmm0, xmm3

  vmovss xmm3, dword ptr [cAVXExpP6]
  vfmadd213ss xmm3, xmm2, dword ptr [cAVXExpP5]
  vfmadd213ss xmm3, xmm2, dword ptr [cAVXExpP4]
  vfmadd213ss xmm3, xmm2, dword ptr [cAVXExpP3]
  vfmadd213ss xmm3, xmm2, dword ptr [cAVXExpP2]
  vfmadd213ss xmm3, xmm2, dword ptr [cAVXExpP1]
  vfmadd213ss xmm3, xmm2, dword ptr [cAVXExpP0]

  vcvtss2si ebx, xmm1
  add ebx, 127
  shl ebx, 23
  movd xmm1, ebx

  vmulss xmm7, xmm3, xmm1
  vmovss [edi + ecx*4], xmm7
  vaddss xmm6, xmm6, xmm7

  inc ecx
  cmp ecx, edx
  jl @TailLoop

  // Combine bulk + tail
  movss xmm0, [esp]            // bulk sum
  vaddss xmm0, xmm0, xmm6
  movss [esp+8], xmm0
  jmp @Exit

@Done:
  movss xmm0, [esp]            // bulk sum only
  jmp @Exit

@Zero:
  xorps xmm0, xmm0

@Exit:
  movss [esp+8], xmm0
  fld dword ptr [esp+8]        // return via ST(0)

  add esp, 32
  pop ebx
  pop edi
  pop esi

end;

function _AVX512ExpShiftSum( dst : PSingle; src : PSingle; Shift : single; N : integer ) : single;
begin
  Result := _AVX2ExpShiftSum(dst, src, Shift, N);
end;

{-----------------------------------------------------------------------------
  AVX2 natural logarithm (32-bit). Uses ymm0-ymm7 only.

  Parameters:
    EAX = dst : PSingle
    EDX = src : PSingle
    ECX = N   : integer

  Algorithm: extract exponent and mantissa, scale mantissa to [0.5,1),
  use polynomial approximation for ln(m), then result = ln(m) + e*ln(2).

  Notes:
    - Requires AVX2 support.
    - Processes 8 floats per iteration.
    - All constants that are modified inside the loop are reloaded each iteration.
-----------------------------------------------------------------------------}
procedure _AVX2Ln(dst: PSingle; src: PSingle; N: integer);
asm
  push ebx
  push esi
  push edi
  sub esp, 32

  mov esi, edx
  mov edi, eax
  mov eax, ecx

  test eax, eax
  jle @Exit

  mov edx, eax
  and edx, 7
  sub eax, edx

  mov ecx, eax
  shr ecx, 3
  jz @Tail

@BulkLoop:
  // Reload constants destroyed in the loop
  vbroadcastss ymm0, dword ptr [cAVXLnMinNorm]
  vpbroadcastd ymm1, dword ptr [cAVXExp127]          // 127
  vbroadcastss ymm2, dword ptr [cAVX8SSOne]           // 1.0
  vpbroadcastd ymm3, dword ptr [cAVXLnInvMant]       // $007FFFFF
  vbroadcastss ymm4, dword ptr [cAVXLnHalf]          // 0.5
  vbroadcastss ymm5, dword ptr [cAVXLnSqrtHf]        // sqrt(0.5)

  // Load x and clamp to min norm
  vmovups ymm7, [esi]
  vmaxps ymm0, ymm7, ymm0

  // Extract exponent: e = ((bits >> 23) & 0xff) - 127 + 1
  vpsrld ymm6, ymm0, 23
  vpsubd ymm6, ymm6, ymm1
  vcvtdq2ps ymm6, ymm6
  vaddps ymm6, ymm6, ymm2

  // Mantissa: (bits & invmant) | half
  vandps ymm7, ymm0, ymm3
  vorps  ymm7, ymm7, ymm4

  // mask = (x < sqrt(0.5))
  vcmpltps ymm0, ymm7, ymm5

  // if mask: x = 2*x - 1
  vandps ymm1, ymm7, ymm0
  vsubps ymm7, ymm7, ymm2
  vaddps ymm7, ymm7, ymm1

  // e = e - (mask ? 1.0 : 0.0)
  vandps ymm1, ymm2, ymm0
  vsubps ymm6, ymm6, ymm1

  // z = x*x
  vmulps ymm1, ymm7, ymm7

  // Polynomial: ln(m) = x * z * (P0 + x*(P1 + x*(P2 + ...)))
  vmovaps ymm0, ymm7
  vmovups ymm3, yword ptr [cAVXLnP0]
  vfmadd213ps ymm3, ymm0, yword ptr [cAVXLnP1]
  vfmadd213ps ymm3, ymm0, yword ptr [cAVXLnP2]
  vfmadd213ps ymm3, ymm0, yword ptr [cAVXLnP3]
  vfmadd213ps ymm3, ymm0, yword ptr [cAVXLnP4]
  vfmadd213ps ymm3, ymm0, yword ptr [cAVXLnP5]
  vfmadd213ps ymm3, ymm0, yword ptr [cAVXLnP6]
  vfmadd213ps ymm3, ymm0, yword ptr [cAVXLnP7]
  vfmadd213ps ymm3, ymm0, yword ptr [cAVXLnP8]   // poly = P(x)

  vmulps ymm3, ymm3, ymm0
  vmulps ymm3, ymm3, ymm1

  // Add correction terms
  vfmadd231ps ymm3, ymm6, yword ptr [cAVXLnQ1]
  vmulps ymm4, ymm1, ymm4
  vsubps ymm3, ymm3, ymm4
  vaddps ymm0, ymm0, ymm3
  vfmadd231ps ymm0, ymm6, yword ptr [cAVXLnQ2]

  vmovups [edi], ymm0

  add esi, 32
  add edi, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  test edx, edx
  jz @Exit

  xor ecx, ecx
@TailLoop:
  // Reload constants for each scalar iteration
  vpbroadcastd xmm1, [cAVXExp127]
  vpbroadcastd xmm3, dword ptr [cAVXLnInvMant]
  vbroadcastss xmm2, dword ptr [cAVX8SSOne]
  vbroadcastss xmm4, dword ptr [cAVXLnHalf]
  vbroadcastss xmm5, dword ptr [cAVXLnSqrtHf]

  vmovss xmm0, [esi + ecx*4]
  vbroadcastss xmm6, dword ptr [cAVXLnMinNorm]
  vmaxss xmm0, xmm0, xmm6

  vpsrld xmm6, xmm0, 23
  vpsubd xmm6, xmm6, xmm1
  vcvtdq2ps xmm6, xmm6
  vaddss xmm6, xmm6, xmm2

  vandps xmm7, xmm0, xmm3
  vorps  xmm7, xmm7, xmm4
  vcmpltss xmm0, xmm7, xmm5
  vandps xmm1, xmm7, xmm0
  vsubss xmm7, xmm7, xmm2
  vaddss xmm7, xmm7, xmm1

  vandps xmm1, xmm2, xmm0
  vsubss xmm6, xmm6, xmm1

  vmulss xmm1, xmm7, xmm7

  vmovaps xmm0, xmm7
  vmovss xmm3, dword ptr [cAVXLnP0]
  vfmadd213ss xmm3, xmm0, dword ptr [cAVXLnP1]
  vfmadd213ss xmm3, xmm0, dword ptr [cAVXLnP2]
  vfmadd213ss xmm3, xmm0, dword ptr [cAVXLnP3]
  vfmadd213ss xmm3, xmm0, dword ptr [cAVXLnP4]
  vfmadd213ss xmm3, xmm0, dword ptr [cAVXLnP5]
  vfmadd213ss xmm3, xmm0, dword ptr [cAVXLnP6]
  vfmadd213ss xmm3, xmm0, dword ptr [cAVXLnP7]
  vfmadd213ss xmm3, xmm0, dword ptr [cAVXLnP8]

  vmulss xmm3, xmm3, xmm0
  vmulss xmm3, xmm3, xmm1
  vfmadd231ss xmm3, xmm6, dword ptr [cAVXLnQ1]
  vmulss xmm4, xmm1, xmm4
  vsubss xmm3, xmm3, xmm4
  vaddss xmm0, xmm0, xmm3
  vfmadd231ss xmm0, xmm6, dword ptr [cAVXLnQ2]

  vmovss [edi + ecx*4], xmm0

  inc ecx
  cmp ecx, edx
  jl @TailLoop

@Exit:
  vzeroupper
  add esp, 32
  pop edi
  pop esi
  pop ebx
  ret
end;

procedure _AVX512Ln( dst : PSingle; src : PSingle; N : integer ); inline;
begin
  _AVX2Ln(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 sin/cos (32-bit): dst[i] = sin(src[i]) or cos(src[i]) for i=0..N-1.
  Parameters: EAX=dst, EDX=src, ECX=N, DoCos on stack at [EBP+8].
  Algorithm: range reduction q=round(x*2/pi), reduce r to [-pi/4, pi/4],
  compute sin_abs and cos_abs via degree-3 polynomials, reconstruct based on q.
  Notes: Win32 AVX2 only (ymm0-ymm7), vzeroupper before exit.
-----------------------------------------------------------------------------}
procedure _AVX2SinCos(dst: PSingle; src: PSingle; N: integer; DoCos: integer);
asm
  push esi
  push edi
  push ebx

  sub esp, 128

  mov [esp+4], eax
  mov [esp+8], edx

  mov esi, edx
  mov edi, eax
  mov eax, ecx

  test eax, eax
  jle @Exit

  mov edx, eax
  and edx, 7
  mov [esp+16], edx
  sub eax, edx

  mov ecx, eax
  shr ecx, 3
  mov [esp+12], ecx

  jz @Tail

  vpbroadcastd ymm0, dword ptr [cOneInt]

@BulkLoop:
  vbroadcastss ymm4, dword ptr [cAVXSinCosInvPi2]
  vbroadcastss ymm5, dword ptr [cAVXSinCosPi2]
  vbroadcastss ymm6, dword ptr [cAVXSinCosPi4]
  vbroadcastss ymm7, dword ptr [cAVX8SSOne]

  vmovups ymm0, [esi]
  vmulps  ymm1, ymm0, ymm4
  vroundps ymm1, ymm1, $00
  vcvtps2dq ymm1, ymm1
  vmovups [esp+32], ymm1

  vcvtdq2ps ymm2, ymm1
  vmulps  ymm2, ymm2, ymm5
  vsubps  ymm0, ymm0, ymm2
  vmovups [esp+64], ymm0

  vmovaps ymm3, ymm0
  vandps ymm0, ymm0, yword ptr [cAVXArgAbsMask]
  vandps ymm3, ymm3, yword ptr [cAVXSinCosSignMask]
  vcmpgtps ymm2, ymm0, ymm6
  vsubps ymm0, ymm5, ymm0
  vxorps ymm0, ymm0, ymm3

  vmovups ymm3, [esp+64]
  vblendvps ymm0, ymm3, ymm0, ymm2

  vmovups ymm3, [esp+32]
  vpbroadcastd ymm1, dword ptr [cOneInt]

  vpaddd ymm4, ymm3, ymm1
  vpsubd ymm5, ymm3, ymm1

  vandps ymm6, ymm0, yword ptr [cAVXSinCosSignMask]
  vpsrad ymm6, ymm6, 31
  vblendvps ymm4, ymm4, ymm5, ymm6
  vblendvps ymm1, ymm3, ymm4, ymm2
  vmulps ymm2, ymm0, ymm0
  vbroadcastss ymm3, dword ptr [cAVXSinP3]
  vbroadcastss ymm4, dword ptr [cAVXSinP2]
  vfmadd213ps ymm3, ymm2, ymm4
  vbroadcastss ymm4, dword ptr [cAVXSinP1]
  vfmadd213ps ymm3, ymm2, ymm4
  vbroadcastss ymm4, dword ptr [cAVXSinP0]
  vfmadd213ps ymm3, ymm2, ymm4
  vmulps ymm3, ymm3, ymm0
  vbroadcastss ymm4, dword ptr [cAVXCosQ2]
  vbroadcastss ymm5, dword ptr [cAVXCosQ1]
  vfmadd213ps ymm4, ymm2, ymm5
  vbroadcastss ymm5, dword ptr [cAVXCosQ0]
  vfmadd213ps ymm4, ymm2, ymm5
  vmulps ymm4, ymm4, ymm2
  vaddps ymm4, ymm4, ymm7

  vpbroadcastd ymm0, dword ptr [cOneInt]

  vpand ymm5, ymm1, ymm0
  vpsrld ymm6, ymm1, 1
  vpand ymm6, ymm6, ymm0
  vpxor ymm7, ymm7, ymm7
  vpcmpeqd ymm7, ymm5, ymm7
  vpcmpeqd ymm6, ymm6, ymm0
  vpand ymm5, ymm1, ymm0
  vpsrld ymm1, ymm1, 1
  vpand ymm1, ymm1, ymm0
  vpxor ymm5, ymm5, ymm1
  vpcmpeqd ymm5, ymm5, ymm0
  vblendvps ymm0, ymm4, ymm3, ymm7
  vxorps ymm2, ymm0, yword ptr [cAVXSinCosSignMask]
  vblendvps ymm0, ymm0, ymm2, ymm6
  vblendvps ymm1, ymm3, ymm4, ymm7
  vxorps ymm2, ymm1, yword ptr [cAVXSinCosSignMask]
  vblendvps ymm1, ymm1, ymm2, ymm5


  cmp [ebp+8], 0
  jnz @DoCosSelect
  jmp @Store
@DoCosSelect:
  vmovaps ymm0, ymm1

@Store:
  mov eax, [esp+4]
  mov edx, [esp+12]
  sub edx, ecx
  shl edx, 5
  add eax, edx

  vmovups [edi], ymm0

  add esi, 32
  add edi, 32
  dec ecx
  jnz @BulkLoop

  vzeroupper

@Tail:
  mov edx, [esp+16]
  test edx, edx
  jz @Exit

  mov ebx, edx

  vbroadcastss xmm4, dword ptr [cAVXSinCosInvPi2]
  vbroadcastss xmm5, dword ptr [cAVXSinCosPi2]
  vbroadcastss xmm6, dword ptr [cAVXSinCosPi4]
  vbroadcastss xmm7, dword ptr [cAVX8SSOne]

  xor ecx, ecx
@TailLoop:
  vmovss xmm0, [esi + ecx*4]
  vmulss xmm1, xmm0, xmm4
  vroundss xmm1, xmm1, xmm1, $00
  vcvtss2si eax, xmm1
  vcvtsi2ss xmm2, xmm2, eax
  vmulss xmm2, xmm2, xmm5
  vsubss xmm0, xmm0, xmm2

  vandps xmm2, xmm0, oword ptr [cAVXSinCosSignMask]
  vandps xmm3, xmm0, oword ptr [cAVXArgAbsMask]
  vcomiss xmm3, xmm6
  jbe @NoReduce
    vsubss xmm3, xmm5, xmm3
    vxorps xmm3, xmm3, xmm2
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

  vmovss xmm2, dword ptr [cAVXSinP3]
  vmovss xmm3, dword ptr [cAVXSinP2]
  vfmadd213ss xmm2, xmm1, xmm3
  vmovss xmm3, dword ptr [cAVXSinP1]
  vfmadd213ss xmm2, xmm1, xmm3
  vmovss xmm3, dword ptr [cAVXSinP0]
  vfmadd213ss xmm2, xmm1, xmm3
  vmulss xmm2, xmm2, xmm0

  vmovss xmm3, dword ptr [cAVXCosQ2]
  vmovss xmm4, dword ptr [cAVXCosQ1]
  vfmadd213ss xmm3, xmm1, xmm4
  vmovss xmm4, dword ptr [cAVXCosQ0]
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
    vxorps xmm4, xmm4, oword ptr [cAVXSinCosSignMask]
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
    vxorps xmm5, xmm5, oword ptr [cAVXSinCosSignMask]
@CosNoFlip:

  cmp dword ptr [ebp+8], 0
  jnz @TailDoCos
  vmovss xmm0, xmm0, xmm4
  jmp @TailStore
@TailDoCos:
  vmovss xmm0, xmm0, xmm5
@TailStore:
  vmovss [edi + ecx*4], xmm0

  inc ecx
  cmp ecx, ebx
  jl @TailLoop

@Exit:
  vzeroupper
  add esp, 128

  pop ebx
  pop edi
  pop esi
end;

procedure _AVX512SinCos( dst : PSingle; src : PSingle; N : integer; DoCos : integer ); inline;
begin
  _AVX2SinCos(dst, src, N, DoCos);
end;

{-----------------------------------------------------------------------------
  AVX2 conversion: single-precision float to bfloat16 (round-to-nearest-even).

  Parameters:
    EAX = dst : PSingle     - pointer to destination buffer (will store 16-bit values)
                              (Note: declared as PSingle for simplicity, but actually points to Word array)
    EDX = src : PSingle     - pointer to source single-precision array
    ECX = N   : integer     - number of elements to convert

  Notes:
    - AVX2 required (ymm0-ymm7 only in 32-bit mode).
    - Processes 32 elements per bulk iteration (4 YMM blocks), then 8-element chunks, then scalar tail for 0..7.
    - Uses integer operations exclusively; no MXCSR dependency.
    - vzeroupper called before exit to avoid AVX-SSE transition penalty.
-----------------------------------------------------------------------------}
procedure _AVX2EncodeBF16( dst: PSingle; src: PSingle; N: integer );
asm
  push ebx
  push esi
  push edi

  mov esi, edx
  mov edi, eax
  mov eax, ecx

  // Constants in ymm2..ymm6
  mov ebx, $7FFFFFFF
  //vmovd xmm0, ebx
  //vbroadcastss ymm2, xmm0   // sign mask
  vpbroadcastd ymm2, eax

  mov ebx, $7F800000
  vmovd xmm0, ebx
  vbroadcastss ymm3, xmm0   // inf/NaN threshold

  mov ebx, $00000001
  vmovd xmm0, ebx
  vbroadcastss ymm4, xmm0   // round low bit

  mov ebx, $00007FFF
  vmovd xmm0, ebx
  vbroadcastss ymm5, xmm0   // half ULP

  mov ebx, $00000040
  vmovd xmm0, ebx
  vbroadcastss ymm6, xmm0   // quiet NaN bit

  // Bulk: 8 elements per iteration
  mov ecx, eax
  shr ecx, 3
  jz @Tail

@Loop8:
  vmovups   ymm0, [esi]

  vpsrld    ymm7, ymm0, 16          // kept
  vpand     ymm1, ymm7, ymm4        // (kept & 1)
  vpaddd    ymm1, ymm1, ymm5        // + half_ulp
  vpaddd    ymm1, ymm1, ymm0        // + original
  vpsrld    ymm1, ymm1, 16          // rounded

  vpand     ymm0, ymm0, ymm2        // |bits|
  vpcmpgtd  ymm0, ymm0, ymm3        // mask (NaN?)
  vpor      ymm7, ymm7, ymm6        // quiet NaN
  vpblendvb ymm0, ymm1, ymm7, ymm0  // blend

  vpackusdw ymm0, ymm0, ymm0
  vextracti128 xmm1, ymm0, 1
  vpunpcklqdq xmm0, xmm0, xmm1
  vmovups   [edi], xmm0

  add esi, 32
  add edi, 16
  dec ecx
  jnz @Loop8

@Tail:
  mov ecx, eax
  and ecx, 7
  jz @Done

  // Scalar constants in xmm0..xmm4
  mov ebx, $7FFFFFFF
  vmovd xmm0, ebx
  vbroadcastss xmm0, xmm0
  mov ebx, $7F800000
  vmovd xmm1, ebx
  vbroadcastss xmm1, xmm1
  mov ebx, $00000001
  vmovd xmm2, ebx
  vbroadcastss xmm2, xmm2
  mov ebx, $00007FFF
  vmovd xmm3, ebx
  vbroadcastss xmm3, xmm3
  mov ebx, $00000040
  vmovd xmm4, ebx
  vbroadcastss xmm4, xmm4

@ScalarLoop:
  vmovss xmm5, [esi]
  vpsrld xmm6, xmm5, 16
  vpand  xmm7, xmm6, xmm2
  vpaddd xmm7, xmm7, xmm3
  vpaddd xmm7, xmm7, xmm5
  vpsrld xmm7, xmm7, 16
  vpand  xmm5, xmm5, xmm0
  vpcmpgtd xmm5, xmm5, xmm1
  vpor   xmm6, xmm6, xmm4
  vpblendvb xmm5, xmm7, xmm6, xmm5
  vmovd eax, xmm5
  mov [edi], ax

  add esi, 4
  add edi, 2
  dec ecx
  jnz @ScalarLoop

@Done:
  vzeroupper
  pop edi
  pop esi
  pop ebx
end;

procedure _AVX512EncodeBF16( dst: PSingle; src : PSingle; N : integer ); inline;
begin
  _AVX2EncodeBF16(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 leaky clamp (ReLU-like) for 32-bit Delphi.
  For each x in src:
    if x > HighLimit then
      dst = HighLimit + (x - HighLimit) * Slope
    else if x > LowLimit then
      dst = x
    else
      dst = LowLimit + (x - LowLimit) * Slope

  Parameters (register convention):
    EAX = dst : PSingle
    EDX = src : PSingle
    ECX = N   : integer
    LowLimit  : Single (stack, [ebp+16])
    HighLimit : Single (stack, [ebp+12])
    Slope     : Single (stack, [ebp+8])

  Notes:
    - Uses only ymm0..ymm7 (safe for 32-bit).
    - Processes 8 elements per iteration (one YMM block).
    - vzeroupper on exit.
    - No floating-point exceptions; comparisons are quiet (GT_OQ).
-----------------------------------------------------------------------------}
procedure _AVX2ReluL(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer);
asm
  push esi
  push edi
  push ebx

  mov edi, eax                 // edi = dst
  mov esi, edx                 // esi = src
  mov edx, ecx                 // edx = N

  test edx, edx
  jle @Exit

  vbroadcastss ymm3, dword ptr [ebp+16]  // ymm3 = LowLimit
  vbroadcastss ymm4, dword ptr [ebp+12]  // ymm4 = HighLimit
  vbroadcastss ymm2, dword ptr [ebp+8]   // ymm2 = Slope

  mov eax, edx
  and eax, 7
  sub edx, eax
  mov ecx, edx
  shr ecx, 3
  jz @Tail

@Loop8:
  vmovups ymm0, [esi]          // x

  // High
  vsubps ymm1, ymm0, ymm4
  vmulps ymm1, ymm1, ymm2
  vaddps ymm1, ymm1, ymm4

  // Low
  vsubps ymm5, ymm0, ymm3
  vmulps ymm5, ymm5, ymm2
  vaddps ymm5, ymm5, ymm3

  // x > LowLimit
  vcmpps ymm6, ymm0, ymm3, 30
  vblendvps ymm5, ymm5, ymm0, ymm6

  // x > HighLimit
  vcmpps ymm7, ymm0, ymm4, 30
  vblendvps ymm5, ymm5, ymm1, ymm7

  vmovups [edi], ymm5

  add esi, 32
  add edi, 32
  dec ecx
  jnz @Loop8

@Tail:
  test eax, eax
  jz @Exit

  vbroadcastss xmm2, dword ptr [ebp+8]   // Slope
  vbroadcastss xmm3, dword ptr [ebp+16]  // LowLimit
  vbroadcastss xmm4, dword ptr [ebp+12]  // HighLimit

  xor ecx, ecx
@ScalarLoop:
  vmovss xmm0, [esi + ecx*4]

  // high
  vsubss xmm5, xmm0, xmm4
  vmulss xmm5, xmm5, xmm2
  vaddss xmm5, xmm5, xmm4

  // low
  vsubss xmm6, xmm0, xmm3
  vmulss xmm6, xmm6, xmm2
  vaddss xmm6, xmm6, xmm3

  vcmpltss xmm7, xmm3, xmm0
  vblendvps xmm6, xmm6, xmm0, xmm7

  vcmpltss xmm7, xmm4, xmm0
  vblendvps xmm6, xmm6, xmm5, xmm7

  vmovss [edi + ecx*4], xmm6
  inc ecx
  cmp ecx, eax
  jl @ScalarLoop

@Exit:
  vzeroupper

  pop ebx
  pop edi
  pop esi
end;

procedure _AVX512ReluL(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;
begin
  _AVX2ReluL(dst, src, LowLimit, HighLimit, Slope, N);
end;

{-----------------------------------------------------------------------------
  AVX + F16C conversion: single-precision float to half-precision (binary16).
  Uses vcvtps2ph with round-to-nearest-even (imm8=0).

  Parameters:
    EAX = dst : Pointer      (points to Word array for half-precision output)
    EDX = src : PSingle
    ECX = N   : integer

  Exception and MXCSR notes:
    - The narrowing conversion may raise #O or #I for overflow/NaN inputs.
    - This function does NOT modify MXCSR; caller must mask exceptions if needed.
    - Uses default round-to-nearest-even mode.

  Notes:
    - Uses only ymm0..ymm3 (safe for 32-bit).
    - Processes 32 elements per bulk iteration, then 8-element chunks,
      then scalar for 0..7 elements.
    - vzeroupper on exit.
-----------------------------------------------------------------------------}
procedure _AVX2EncodeF16(dst: Pointer; src: Pointer; N: integer);
asm
  push ebx
  push esi
  push edi

  mov esi, edx                 // esi = src
  mov edi, eax                 // edi = dst
  mov eax, ecx                 // eax = N

  test eax, eax
  jle @Exit

  // Bulk count = N - (N mod 32)
  mov ebx, eax
  and ebx, 31                  // tail = N mod 32
  sub eax, ebx                 // bulk = N - tail
  mov ecx, eax
  shr ecx, 5                   // number of 32-element chunks
  jz @Tail

@LargeLoop:
  vmovups ymm0, [esi]
  vmovups ymm1, [esi+32]
  vmovups ymm2, [esi+64]
  vmovups ymm3, [esi+96]

  vcvtps2ph [edi], ymm0, 0
  vcvtps2ph [edi+16], ymm1, 0
  vcvtps2ph [edi+32], ymm2, 0
  vcvtps2ph [edi+48], ymm3, 0

  add esi, 128
  add edi, 64
  dec ecx
  jnz @LargeLoop

@Tail:
  test ebx, ebx
  jz @Exit

  mov ecx, ebx
  shr ecx, 3
  jz @ScalarTail

@SmallLoop:
  vmovups ymm0, [esi]
  vcvtps2ph [edi], ymm0, 0
  add esi, 32
  add edi, 16
  dec ecx
  jnz @SmallLoop

@ScalarTail:
  and ebx, 7
  jz @Exit

@ScalarLoop:
  vmovss xmm0, [esi]           // load 1 float

  //vcvtps2ph xmm1, xmm0, 0     // convert to half, result in low word of xmm1
  db $c4, $e3, $79, $1d, $c1, $00
  vmovd eax, xmm1              // move low 32 bits (contains half in low 16 bits)

  mov [edi], ax                // store 16-bit half

  add esi, 4
  add edi, 2
  dec ebx
  jnz @ScalarLoop

@Exit:
  vzeroupper

  pop edi
  pop esi
  pop ebx
end;

procedure _AVX512EncodeF16(dst, src: Pointer; N: integer); inline;
begin
  _AVX2EncodeF16(dst, src, N);
end;

{-----------------------------------------------------------------------------
  AVX2 gate mask: output 1 if LowLimit < x <= HighLimit, else Slope.
  For each x in src:
    if (x > LowLimit) and not (x > HighLimit) then dst = 1.0 else dst = Slope.

  Parameters (register convention):
    EAX = dst : PSingle
    EDX = src : PSingle
    ECX = N   : integer
    LowLimit  : Single (stack, [EBP+16])
    HighLimit : Single (stack, [EBP+12])
    Slope     : Single (stack, [EBP+8])

  Returns: nothing.

  Notes:
    - Uses only ymm0..ymm7 (safe for 32-bit).
    - Processes 8 elements per iteration (one YMM block).
    - vzeroupper on exit.
    - No floating-point exceptions; comparisons are quiet (GT_OQ).
-----------------------------------------------------------------------------}
procedure _AVX2ReluLGateMask(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer);
asm
  push esi
  push edi
  push ebx

  mov edi, eax                 // edi = dst
  mov esi, edx                 // esi = src
  mov edx, ecx                 // edx = N (save it)

  test edx, edx
  jle @Exit

  vbroadcastss ymm2, dword ptr [ebp+8]    // ymm2 = Slope
  vbroadcastss ymm4, dword ptr [ebp+12]   // ymm4 = HighLimit
  vbroadcastss ymm3, dword ptr [ebp+16]   // ymm3 = LowLimit

  // Load 1.0 from constant array
  vbroadcastss ymm5, dword ptr [cAVX8SSOne]   // ymm5 = 1.0

  // Bulk count = N - (N mod 8)
  mov eax, edx
  and eax, 7                   // tail = N mod 8
  sub edx, eax                 // bulk = N - tail
  mov ecx, edx
  shr ecx, 3                   // number of 8-element chunks
  jz @Tail

@Loop8:
  vmovups ymm0, [esi]          // x

  // low_mask = (x > LowLimit) ? all_ones : all_zeros
  vcmpps ymm6, ymm0, ymm3, 30  // GT_OQ

  // high_mask = (x > HighLimit) ? all_ones : all_zeros
  vcmpps ymm7, ymm0, ymm4, 30  // GT_OQ

  // inside = low_mask AND NOT high_mask
  vandnps ymm6, ymm7, ymm6

  // result = inside ? 1.0 : Slope
  vblendvps ymm1, ymm2, ymm5, ymm6
  vmovups [edi], ymm1

  add esi, 32
  add edi, 32
  dec ecx
  jnz @Loop8

@Tail:
  test eax, eax
  jz @Exit

  // Reload constants from stack for scalar tail with correct offsets
  vbroadcastss xmm2, dword ptr [ebp+8]    // Slope
  vbroadcastss xmm3, dword ptr [ebp+16]   // LowLimit
  vbroadcastss xmm4, dword ptr [ebp+12]   // HighLimit
  vbroadcastss xmm5, dword ptr [cAVX8SSOne]   // 1.0

  xor ecx, ecx
@ScalarLoop:
  vmovss xmm0, [esi + ecx*4]   // x

  // x > LowLimit  (using LT with swapped operands)
  vcmpltss xmm6, xmm3, xmm0
  // x > HighLimit
  vcmpltss xmm7, xmm4, xmm0

  // inside = low AND NOT high
  vandnps xmm6, xmm7, xmm6

  // result = inside ? 1.0 : Slope
  vblendvps xmm1, xmm2, xmm5, xmm6
  vmovss [edi + ecx*4], xmm1

  inc ecx
  cmp ecx, eax
  jl @ScalarLoop

@Exit:
  vzeroupper
  pop ebx
  pop edi
  pop esi
end;

procedure _AVX512ReluLGateMask(dst, src: Pointer; LowLimit, HighLimit, Slope: Single; N: integer); inline;
begin
  _AVX2ReluLGateMask(dst, src, LowLimit, HighLimit, Slope, N);
end;


end.
