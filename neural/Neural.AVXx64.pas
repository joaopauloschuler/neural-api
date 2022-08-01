unit Neural.AVXx64;

// ###########################################
// #### 64 bit intel avx functions
// ###########################################

interface

{$IFDEF CPUX64}
{$DEFINE x64}
{$ENDIF}
{$IFDEF cpux86_64}
{$DEFINE x64}
{$ENDIF}
{$IFDEF x64}

function AVXDotProd( x : PSingle; y : PSingle; N : integer ) : single;  {$IFDEF FPC}assembler;{$ENDIF}

{$ENDIF}

implementation

{$IFDEF x64}

{$IFDEF FPC} {$ASMMODE intel} {$S-} {$ENDIF}

function AVXDotProd( x : PSingle; y : PSingle; N : integer ) : single;
asm
   {$IFDEF UNIX}
   // Linux uses a diffrent ABI -> copy over the registers so they meet with winABI
   // The parameters are passed in the following order:
   // RDI, RSI, RDX, RCX, r8, r9 -> mov to RCX, RDX, R8, r9
   mov r8, rdx;
   mov rdx, rsi;
   mov rcx, rdi;
   {$ENDIF}

   // iters
   imul r8, -4;

   // helper registers for the mt1, mt2 and dest pointers
   sub rcx, r8;
   sub rdx, r8;

   {$IFDEF FPC}vxorpd ymm0, ymm0, ymm0;{$ELSE}db $C5,$FD,$57,$C0;{$ENDIF}

   // unrolled loop
   @Loop1:
       add r8, 128;
       jg @loopEnd1;

       {$IFDEF FPC}vmovupd ymm1, [rcx + r8 - 128];{$ELSE}db $C4,$A1,$7D,$10,$4C,$01,$80;{$ENDIF}
       {$IFDEF FPC}vmovupd ymm2, [rdx + r8 - 128];{$ELSE}db $C4,$A1,$7D,$10,$54,$02,$80;{$ENDIF}
       {$IFDEF FPC}vmulps ymm1, ymm1, ymm2;{$ELSE}db $C5,$F4,$59,$CA;{$ENDIF}
       {$IFDEF FPC}vaddps ymm0, ymm0, ymm1;{$ELSE}db $C5,$FC,$58,$C1;{$ENDIF}

       {$IFDEF FPC}vmovupd ymm3, [rcx + r8 - 96];{$ELSE}db $C4,$A1,$7D,$10,$5C,$01,$A0;{$ENDIF}
       {$IFDEF FPC}vmovupd ymm4, [rdx + r8 - 96];{$ELSE}db $C4,$A1,$7D,$10,$64,$02,$A0;{$ENDIF}
       {$IFDEF FPC}vmulps ymm3, ymm3, ymm4;{$ELSE}db $C5,$E4,$59,$DC;{$ENDIF}
       {$IFDEF FPC}vaddps ymm0, ymm0, ymm3;{$ELSE}db $C5,$FC,$58,$C3;{$ENDIF}

       {$IFDEF FPC}vmovupd ymm1, [rcx + r8 - 64];{$ELSE}db $C4,$A1,$7D,$10,$4C,$01,$C0;{$ENDIF}
       {$IFDEF FPC}vmovupd ymm2, [rdx + r8 - 64];{$ELSE}db $C4,$A1,$7D,$10,$54,$02,$C0;{$ENDIF}
       {$IFDEF FPC}vmulps ymm1, ymm1, ymm2;{$ELSE}db $C5,$F4,$59,$CA;{$ENDIF}
       {$IFDEF FPC}vaddps ymm0, ymm0, ymm1;{$ELSE}db $C5,$FC,$58,$C1;{$ENDIF}

       {$IFDEF FPC}vmovupd ymm3, [rcx + r8 - 32];{$ELSE}db $C4,$A1,$7D,$10,$5C,$01,$E0;{$ENDIF}
       {$IFDEF FPC}vmovupd ymm4, [rdx + r8 - 32];{$ELSE}db $C4,$A1,$7D,$10,$64,$02,$E0;{$ENDIF}
       {$IFDEF FPC}vmulps ymm3, ymm3, ymm4;{$ELSE}db $C5,$E4,$59,$DC;{$ENDIF}
       {$IFDEF FPC}vaddps ymm0, ymm0, ymm3;{$ELSE}db $C5,$FC,$58,$C3;{$ENDIF}

   jmp @Loop1;

   @loopEnd1:

   {$IFDEF FPC}vextractf128 xmm2, ymm0, 1;{$ELSE}db $C4,$E3,$7D,$19,$C2,$01;{$ENDIF}
   {$IFDEF FPC}vhaddps xmm0, xmm0, xmm2;{$ELSE}db $C5,$FB,$7C,$C2;{$ENDIF}

   sub r8, 128;
   jz @loop2End;

   // loop to get all fitting into an array of 4
   @Loop2:
      add r8, 16;
      jg @Loop2End;

      {$IFDEF FPC}vmovupd xmm3, [rcx + r8 - 16];{$ELSE}db $C4,$A1,$79,$10,$5C,$01,$F0;{$ENDIF}
      {$IFDEF FPC}vmovupd xmm4, [rdx + r8 - 16];{$ELSE}db $C4,$A1,$79,$10,$64,$02,$F0;{$ENDIF}
      {$IFDEF FPC}vmulps xmm3, xmm3, xmm4;{$ELSE}db $C5,$E0,$59,$DC;{$ENDIF}
      {$IFDEF FPC}vaddps xmm0, xmm0, xmm3;{$ELSE}db $C5,$F8,$58,$C3;{$ENDIF}
   jmp @Loop2;

   @Loop2End:

   // handle last 2 elements
   sub r8, 16;
   jz @loop3End;

   @loop3:
     add r8, 4;
     jg @loop3End;

     {$IFDEF FPC}vmovss xmm3, [rcx + r8 - 4];{$ELSE}db $C4,$A1,$7A,$10,$5C,$01,$FC;{$ENDIF}
     {$IFDEF FPC}vmovss xmm4, [rdx + r8 - 4];{$ELSE}db $C4,$A1,$7A,$10,$64,$02,$FC;{$ENDIF}
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
