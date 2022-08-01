unit Neural.AVX;

// ###########################################
// #### 32 bit intel avx functions
// ###########################################

interface

{$IFDEF CPUX64}
{$DEFINE x64}
{$ENDIF}
{$IFDEF cpux86_64}
{$DEFINE x64}
{$ENDIF}
{$IFNDEF x64}

function AVXDotProd( x : PSingle; y : PSingle; N : integer ) : single; {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}

{$ENDIF}

implementation

{$IFNDEF x64}

{$IFDEF FPC} {$ASMMODE intel} {$S-} {$ENDIF}

function AVXDotProd( x : PSingle; y : PSingle; N : integer ) : single;
// eax = x, edx = y, ecx = N
asm
   // iters
   imul ecx, -4;

   // helper registers for the mt1, mt2 and dest pointers
   sub eax, ecx;
   sub edx, ecx;

   {$IFDEF FPC}vxorpd ymm0, ymm0, ymm0;{$ELSE}db $C5,$FD,$57,$C0;{$ENDIF}

   // unrolled loop
   @Loop1:
       add ecx, 128;
       jg @loopEnd1;

       {$IFDEF FPC}vmovupd ymm1, [eax + ecx - 128];{$ELSE}db $C5,$FD,$10,$4C,$08,$80;{$ENDIF}
       {$IFDEF FPC}vmovupd ymm2, [edx + ecx - 128];{$ELSE}db $C5,$FD,$10,$54,$0A,$80;{$ENDIF}
       {$IFDEF FPC}vmulps ymm1, ymm1, ymm2;{$ELSE}db $C5,$F4,$59,$CA;{$ENDIF}
       {$IFDEF FPC}vaddps ymm0, ymm0, ymm1;{$ELSE}db $C5,$FC,$58,$C1;{$ENDIF}

       {$IFDEF FPC}vmovupd ymm3, [eax + ecx - 96];{$ELSE}db $C5,$FD,$10,$5C,$08,$A0;{$ENDIF}
       {$IFDEF FPC}vmovupd ymm4, [edx + ecx - 96];{$ELSE}db $C5,$FD,$10,$64,$0A,$A0;{$ENDIF}
       {$IFDEF FPC}vmulps ymm3, ymm3, ymm4;{$ELSE}db $C5,$E4,$59,$DC;{$ENDIF}
       {$IFDEF FPC}vaddps ymm0, ymm0, ymm3;{$ELSE}db $C5,$FC,$58,$C3;{$ENDIF}

       {$IFDEF FPC}vmovupd ymm1, [eax + ecx - 64];{$ELSE}db $C5,$FD,$10,$4C,$08,$C0;{$ENDIF}
       {$IFDEF FPC}vmovupd ymm2, [edx + ecx - 64];{$ELSE}db $C5,$FD,$10,$54,$0A,$C0;{$ENDIF}
       {$IFDEF FPC}vmulps ymm1, ymm1, ymm2;{$ELSE}db $C5,$F4,$59,$CA;{$ENDIF}
       {$IFDEF FPC}vaddps ymm0, ymm0, ymm1;{$ELSE}db $C5,$FC,$58,$C1;{$ENDIF}

       {$IFDEF FPC}vmovupd ymm3, [eax + ecx - 32];{$ELSE}db $C5,$FD,$10,$5C,$08,$E0;{$ENDIF}
       {$IFDEF FPC}vmovupd ymm4, [edx + ecx - 32];{$ELSE}db $C5,$FD,$10,$64,$0A,$E0;{$ENDIF}
       {$IFDEF FPC}vmulps ymm3, ymm3, ymm4;{$ELSE}db $C5,$E4,$59,$DC;{$ENDIF}
       {$IFDEF FPC}vaddps ymm0, ymm0, ymm3;{$ELSE}db $C5,$FC,$58,$C3;{$ENDIF}

   jmp @Loop1;

   @loopEnd1:

   {$IFDEF FPC}vextractf128 xmm2, ymm0, 1;{$ELSE}db $C4,$E3,$7D,$19,$C2,$01;{$ENDIF}
   {$IFDEF FPC}vhaddps xmm0, xmm0, xmm2;{$ELSE}db $C5,$FB,$7C,$C2;{$ENDIF}

   sub ecx, 128;
   jz @loop2End;

   // loop to get all fitting into an array of 4
   @Loop2:
      add ecx, 16;
      jg @Loop2End;

      {$IFDEF FPC}vmovupd xmm3, [eax + ecx - 16];{$ELSE}db $C5,$F9,$10,$5C,$08,$F0;{$ENDIF}
      {$IFDEF FPC}vmovupd xmm4, [edx + ecx - 16];{$ELSE}db $C5,$F9,$10,$64,$0A,$F0;{$ENDIF}
      {$IFDEF FPC}vmulps xmm3, xmm3, xmm4;{$ELSE}db $C5,$E0,$59,$DC;{$ENDIF}
      {$IFDEF FPC}vaddps xmm0, xmm0, xmm3;{$ELSE}db $C5,$F8,$58,$C3;{$ENDIF}
   jmp @Loop2;

   @Loop2End:

   // handle last 2 elements
   sub ecx, 16;
   jz @loop3End;

   @loop3:
     add ecx, 4;
     jg @loop3End;

     {$IFDEF FPC}vmovss xmm3, [eax + ecx - 4];{$ELSE}db $C5,$FA,$10,$5C,$08,$FC;{$ENDIF}
     {$IFDEF FPC}vmovss xmm4, [edx + ecx - 4];{$ELSE}db $C5,$FA,$10,$64,$0A,$FC;{$ENDIF}
     {$IFDEF FPC}vmulss xmm3, xmm3, xmm4;{$ELSE}db $C5,$E2,$59,$DC;{$ENDIF}
     {$IFDEF FPC}vaddss xmm0, xmm0, xmm3;{$ELSE}db $C5,$FA,$58,$C3;{$ENDIF}

   jmp @loop3;
   @loop3End:

   // build result
   {$IFDEF FPC}vhaddps xmm0, xmm0, xmm0;{$ELSE}db $C5,$FB,$7C,$C0;{$ENDIF}
   {$IFDEF FPC}vhaddps xmm0, xmm0, xmm0;{$ELSE}db $C5,$FB,$7C,$C0;{$ENDIF}
   {$IFDEF FPC}vzeroupper;{$ELSE}db $C5,$F8,$77;{$ENDIF}
   movss Result, xmm0;
end;

{$ENDIF}

end.
