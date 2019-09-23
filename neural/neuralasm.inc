{$IFDEF AVXANY}
{$ASMMODE intel}
  {$IFDEF AVX32}
  {$define asm_dword_copy :=
  asm
    mov     esi, SourceRawPos
    mov     edi, DestRawPos
    mov     ecx, RowSize
    rep     movsd
  end ['esi','edi','ecx'];
  }
  {$ELSE}
  {$define asm_dword_copy :=
  begin
    asm
      mov     rsi, SourceRawPos
      mov     rdi, DestRawPos
      mov     rcx, 0
      mov     ecx, RowSize
      rep     movsd
    end ['rsi','rdi','rcx'];
  end;
  }
  {$ENDIF}
{$ENDIF}

{$IFDEF AVX64}
{$define asm_avx64_mulladd_ptra_ptrb_ptrc_num :=
if localNumElements > 0 then
begin
asm
mov ecx, localNumElements
mov rdx, PtrA
mov rax, PtrB
mov rbx, PtrC

push rcx
shr ecx,5  // number of large iterations = number of elements / 32
jz @SkipLargeAddLoop

@LargeAddLoop:
{$IFDEF AVX512}
  vmovups zmm4, [rbx]
  vmovups zmm5, [rbx+64]

  vmulps  zmm4, zmm4, [rax]
  vmulps  zmm5, zmm5, [rax+64]

  vaddps  zmm0, zmm4, [rdx]
  vaddps  zmm1, zmm5, [rdx+64]

  vmovups [rdx],    zmm0
  vmovups [rdx+64], zmm1
{$ELSE}
  vmovups ymm4, [rbx]
  vmovups ymm5, [rbx+32]

  vmulps  ymm0, ymm4, [rax]
  vmulps  ymm1, ymm5, [rax+32]

  vaddps  ymm0, ymm0, [rdx]
  vaddps  ymm1, ymm1, [rdx+32]

  vmovups [rdx],    ymm0
  vmovups [rdx+32], ymm1

  vmovups ymm4, [rbx+64]
  vmovups ymm5, [rbx+96]

  vmulps  ymm2, ymm4, [rax+64]
  vmulps  ymm3, ymm5, [rax+96]

  vaddps  ymm2, ymm2, [rdx+64]
  vaddps  ymm3, ymm3, [rdx+96]

  vmovups [rdx+64], ymm2
  vmovups [rdx+96], ymm3
{$ENDIF}

add rax, 128
add rdx, 128
add rbx, 128

dec ecx
jnz @LargeAddLoop

@SkipLargeAddLoop:
vzeroupper

pop rcx
and ecx,$0000001F
jz @EndAdd
shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

movups  xmm2, [rax]
movups  xmm5, [rbx]
movups  xmm4, [rdx]

mulps   xmm2, xmm5
addps   xmm4, xmm2

movups  [rdx], xmm4

add rax, 16
add rbx, 16
add rdx, 16

dec ecx
jnz @SmallAddLoop

@EndAdd:
end  [
  'RAX', 'RBX', 'RCX', 'RDX',
  'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
  {$IFDEF AVX512},'zmm0','zmm1','zmm4','zmm5'{$ENDIF}
];
end; // of if

if MissedElements>0 then
begin
  PtrA^[localNumElements] += PtrB^[localNumElements]*PtrC^[localNumElements];
  if MissedElements>1 then
  begin
    PtrA^[localNumElements+1] += PtrB^[localNumElements+1]*PtrC^[localNumElements+1];
    if MissedElements>2 then PtrA^[localNumElements+2] += PtrB^[localNumElements+2]*PtrC^[localNumElements+2];
  end;
end;
}
{$ENDIF}

{$IFDEF AVX64}
{$define asm_avx64_prev_backprop :=
if PrevNumElements > 0  then
asm
mov ecx, PrevNumElements
mov rax, PrevPtrB
mov rdx, SmoothLocalOutputErrorDerivPtr

{$IFDEF AVX512}
VBROADCASTSS zmm5, [rdx]
{$ELSE}
VBROADCASTSS ymm5, [rdx]
{$ENDIF}

mov rdx, PrevPtrA

push rcx
shr ecx,5  // number of large iterations = number of elements / 32
jz @SkipLargeAddLoop

@LargeAddLoop:
{$IFDEF AVX512}
vmulps  zmm0, zmm5, [rax]
vmulps  zmm1, zmm5, [rax+64]

vaddps  zmm0, zmm0, [rdx]
vaddps  zmm1, zmm1, [rdx+64]

vmovups [rdx],    zmm0
vmovups [rdx+64], zmm1
{$ELSE}
  {$IFDEF AVX2}
  vmovups ymm0, [rdx]
  vmovups ymm1, [rdx+32]
  vmovups ymm2, [rdx+64]
  vmovups ymm3, [rdx+96]

  vfmadd231ps ymm0, ymm5, [rax]
  vfmadd231ps ymm1, ymm5, [rax+32]
  vfmadd231ps ymm2, ymm5, [rax+64]
  vfmadd231ps ymm3, ymm5, [rax+96]
  {$ELSE}
  vmulps  ymm0, ymm5, [rax]
  vmulps  ymm1, ymm5, [rax+32]
  vmulps  ymm2, ymm5, [rax+64]
  vmulps  ymm3, ymm5, [rax+96]

  vaddps  ymm0, ymm0, [rdx]
  vaddps  ymm1, ymm1, [rdx+32]
  vaddps  ymm2, ymm2, [rdx+64]
  vaddps  ymm3, ymm3, [rdx+96]
  {$ENDIF}

  vmovups [rdx],    ymm0
  vmovups [rdx+32], ymm1
  vmovups [rdx+64], ymm2
  vmovups [rdx+96], ymm3
{$ENDIF}

add rax, 128
add rdx, 128
dec ecx
jnz @LargeAddLoop

@SkipLargeAddLoop:
vzeroupper

pop rcx
and ecx,$0000001F
jz @EndAdd
shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

movups  xmm2, [rax]
movups  xmm4, [rdx]

mulps   xmm2, xmm5
addps   xmm4, xmm2

movups  [rdx], xmm4

add rax, 16
add rdx, 16

dec ecx
jnz @SmallAddLoop

@EndAdd:
end  [
  'RAX', 'RCX', 'RDX',
  'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
  {$IFDEF AVX512},'zmm0','zmm1','zmm5'{$ENDIF}
];

if PrevMissedElements>0 then
begin
  PrevPtrA^[PrevNumElements] += SmoothLocalOutputErrorDeriv*PrevPtrB^[PrevNumElements];
  if PrevMissedElements>1 then
  begin
    PrevPtrA^[PrevNumElements+1] += SmoothLocalOutputErrorDeriv*PrevPtrB^[PrevNumElements+1];
    if PrevMissedElements>2 then PrevPtrA^[PrevNumElements+2] += SmoothLocalOutputErrorDeriv*PrevPtrB^[PrevNumElements+2];
  end;
end;
}
{$ENDIF}
