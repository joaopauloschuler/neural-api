unit NeuralAVX;

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

{$DEFINE AVXSUP}  // assembler support for AVX/FMA built in
{$IFNDEF FPC}
{$IF CompilerVersion<35}
{$UNDEF AVXSUP}
{$IFEND}
{$ENDIF}


// performs Result = sum(x[i]*y[i]);
function AVXDotProd( x : PSingle; y : PSingle; N : integer ) : single; {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}

// performs x[i] = x[i] + fact*y[i];
procedure AVXMulAdd( x : PSingle; y : PSingle; N : integer; const fact : single); {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}

{$ENDIF}

implementation

{$IFNDEF x64}

{$IFDEF FPC} {$ASMMODE intel} {$S-} {$ENDIF}

function AVXDotProd( x : PSingle; y : PSingle; N : integer ) : single;
// eax = x, edx = y, ecx = N
asm
   // iters
   imul ecx, -4;

   // helper registers for the x, y pointers
   sub eax, ecx;
   sub edx, ecx;

   {$IFDEF AVXSUP}vxorpd ymm0, ymm0, ymm0;                            {$ELSE}db $C5,$FD,$57,$C0;{$ENDIF} 

   // unrolled loop
   @Loop1:
       add ecx, 128;
       jg @loopEnd1;

       {$IFDEF AVXSUP}vmovupd ymm1, [eax + ecx - 128];                {$ELSE}db $C5,$FD,$10,$4C,$08,$80;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm2, [edx + ecx - 128];                {$ELSE}db $C5,$FD,$10,$54,$0A,$80;{$ENDIF} 
       {$IFDEF AVXSUP}vmulps ymm1, ymm1, ymm2;                        {$ELSE}db $C5,$F4,$59,$CA;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm0, ymm0, ymm1;                        {$ELSE}db $C5,$FC,$58,$C1;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd ymm3, [eax + ecx - 96];                 {$ELSE}db $C5,$FD,$10,$5C,$08,$A0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm4, [edx + ecx - 96];                 {$ELSE}db $C5,$FD,$10,$64,$0A,$A0;{$ENDIF} 
       {$IFDEF AVXSUP}vmulps ymm3, ymm3, ymm4;                        {$ELSE}db $C5,$E4,$59,$DC;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm0, ymm0, ymm3;                        {$ELSE}db $C5,$FC,$58,$C3;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd ymm1, [eax + ecx - 64];                 {$ELSE}db $C5,$FD,$10,$4C,$08,$C0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm2, [edx + ecx - 64];                 {$ELSE}db $C5,$FD,$10,$54,$0A,$C0;{$ENDIF} 
       {$IFDEF AVXSUP}vmulps ymm1, ymm1, ymm2;                        {$ELSE}db $C5,$F4,$59,$CA;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm0, ymm0, ymm1;                        {$ELSE}db $C5,$FC,$58,$C1;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd ymm3, [eax + ecx - 32];                 {$ELSE}db $C5,$FD,$10,$5C,$08,$E0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm4, [edx + ecx - 32];                 {$ELSE}db $C5,$FD,$10,$64,$0A,$E0;{$ENDIF} 
       {$IFDEF AVXSUP}vmulps ymm3, ymm3, ymm4;                        {$ELSE}db $C5,$E4,$59,$DC;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm0, ymm0, ymm3;                        {$ELSE}db $C5,$FC,$58,$C3;{$ENDIF} 

   jmp @Loop1;

   @loopEnd1:

       {$IFDEF AVXSUP}vextractf128 xmm2, ymm0, 1;                     {$ELSE}db $C4,$E3,$7D,$19,$C2,$01;{$ENDIF} 
       {$IFDEF AVXSUP}vhaddps xmm0, xmm0, xmm2;                       {$ELSE}db $C5,$FB,$7C,$C2;{$ENDIF} 

   sub ecx, 128;
   jz @loop3End;

   // loop to get all fitting into an array of 4
   @Loop2:
      add ecx, 16;
      jg @Loop2End;

      {$IFDEF AVXSUP}vmovupd xmm3, [eax + ecx - 16];                  {$ELSE}db $C5,$F9,$10,$5C,$08,$F0;{$ENDIF} 
      {$IFDEF AVXSUP}vmovupd xmm4, [edx + ecx - 16];                  {$ELSE}db $C5,$F9,$10,$64,$0A,$F0;{$ENDIF} 
      {$IFDEF AVXSUP}vmulps xmm3, xmm3, xmm4;                         {$ELSE}db $C5,$E0,$59,$DC;{$ENDIF} 
      {$IFDEF AVXSUP}vaddps xmm0, xmm0, xmm3;                         {$ELSE}db $C5,$F8,$58,$C3;{$ENDIF} 
   jmp @Loop2;

   @Loop2End:

   // handle last 2 elements
   sub ecx, 16;
   jz @loop3End;

   @loop3:
     add ecx, 4;
     jg @loop3End;

     {$IFDEF AVXSUP}vmovss xmm3, [eax + ecx - 4];                     {$ELSE}db $C5,$FA,$10,$5C,$08,$FC;{$ENDIF} 
     {$IFDEF AVXSUP}vmovss xmm4, [edx + ecx - 4];                     {$ELSE}db $C5,$FA,$10,$64,$0A,$FC;{$ENDIF} 
     {$IFDEF AVXSUP}vmulss xmm3, xmm3, xmm4;                          {$ELSE}db $C5,$E2,$59,$DC;{$ENDIF} 
     {$IFDEF AVXSUP}vaddss xmm0, xmm0, xmm3;                          {$ELSE}db $C5,$FA,$58,$C3;{$ENDIF} 

   jmp @loop3;
   @loop3End:

   // build result
  {$IFDEF AVXSUP}vhaddps xmm0, xmm0, xmm0;                            {$ELSE}db $C5,$FB,$7C,$C0;{$ENDIF} 
  {$IFDEF AVXSUP}vhaddps xmm0, xmm0, xmm0;                            {$ELSE}db $C5,$FB,$7C,$C0;{$ENDIF} 
  {$IFDEF AVXSUP}vzeroupper;                                          {$ELSE}db $C5,$F8,$77;{$ENDIF} 
   movss Result, xmm0;
end;

procedure AVXMulAdd( x : PSingle; y : PSingle; N : integer; const fact : single); {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}
// eax = x, edx = y, ecx = N
asm
   push esi;

   // broadcast factor to ymm0
   lea esi, fact;
   {$IFDEF AVXSUP}vbroadcastss ymm0, fact;                            {$ELSE}db $C4,$E2,$7D,$18,$45,$08;{$ENDIF} 

   // iters
   imul ecx, -4;

   // helper registers for the x, y
   sub eax, ecx;
   sub edx, ecx;

   // unrolled loop
   @Loop1:
       add ecx, 128;
       jg @loopEnd1;

       {$IFDEF AVXSUP}vmovupd ymm1, [eax + ecx - 128];                {$ELSE}db $C5,$FD,$10,$4C,$08,$80;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm2, [edx + ecx - 128];                {$ELSE}db $C5,$FD,$10,$54,$0A,$80;{$ENDIF} 

       {$IFDEF AVXSUP}vmulps ymm2, ymm2, ymm0;                        {$ELSE}db $C5,$EC,$59,$D0;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm1, ymm1, ymm2;                        {$ELSE}db $C5,$F4,$58,$CA;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [eax + ecx - 128], ymm1;                {$ELSE}db $C5,$FD,$11,$4C,$08,$80;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd ymm1, [eax + ecx - 96];                 {$ELSE}db $C5,$FD,$10,$4C,$08,$A0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm2, [edx + ecx - 96];                 {$ELSE}db $C5,$FD,$10,$54,$0A,$A0;{$ENDIF} 

       {$IFDEF AVXSUP}vmulps ymm2, ymm2, ymm0;                        {$ELSE}db $C5,$EC,$59,$D0;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm1, ymm1, ymm2;                        {$ELSE}db $C5,$F4,$58,$CA;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [eax + ecx - 96], ymm1;                 {$ELSE}db $C5,$FD,$11,$4C,$08,$A0;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd ymm1, [eax + ecx - 64];                 {$ELSE}db $C5,$FD,$10,$4C,$08,$C0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm2, [edx + ecx - 64];                 {$ELSE}db $C5,$FD,$10,$54,$0A,$C0;{$ENDIF} 

       {$IFDEF AVXSUP}vmulps ymm2, ymm2, ymm0;                        {$ELSE}db $C5,$EC,$59,$D0;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm1, ymm1, ymm2;                        {$ELSE}db $C5,$F4,$58,$CA;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [eax + ecx - 64], ymm1;                 {$ELSE}db $C5,$FD,$11,$4C,$08,$C0;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd ymm1, [eax + ecx - 32];                 {$ELSE}db $C5,$FD,$10,$4C,$08,$E0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm2, [edx + ecx - 32];                 {$ELSE}db $C5,$FD,$10,$54,$0A,$E0;{$ENDIF} 

       {$IFDEF AVXSUP}vmulps ymm2, ymm2, ymm0;                        {$ELSE}db $C5,$EC,$59,$D0;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm1, ymm1, ymm2;                        {$ELSE}db $C5,$F4,$58,$CA;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [eax + ecx - 32], ymm1;                 {$ELSE}db $C5,$FD,$11,$4C,$08,$E0;{$ENDIF} 
   jmp @Loop1;

   @loopEnd1:

   sub ecx, 128;
   jz @loop3End;

   // loop to get all fitting into an array of 4
   @Loop2:
      add ecx, 16;
      jg @Loop2End;

      {$IFDEF AVXSUP}vmovupd xmm1, [eax + ecx - 16];                  {$ELSE}db $C5,$F9,$10,$4C,$08,$F0;{$ENDIF} 
      {$IFDEF AVXSUP}vmovupd xmm2, [edx + ecx - 16];                  {$ELSE}db $C5,$F9,$10,$54,$0A,$F0;{$ENDIF} 

      {$IFDEF AVXSUP}vmulps xmm2, xmm2, xmm0;                         {$ELSE}db $C5,$E8,$59,$D0;{$ENDIF} 
      {$IFDEF AVXSUP}vaddps xmm1, xmm1, xmm2;                         {$ELSE}db $C5,$F0,$58,$CA;{$ENDIF} 

      {$IFDEF AVXSUP}vmovupd [eax + ecx - 16], xmm1;                  {$ELSE}db $C5,$F9,$11,$4C,$08,$F0;{$ENDIF} 
   jmp @Loop2;

   @Loop2End:

   // handle last 2 elements
   sub ecx, 16;
   jz @loop3End;

   @loop3:
     add ecx, 4;
     jg @loop3End;

     {$IFDEF AVXSUP}vmovss xmm1, [eax + ecx - 4];                     {$ELSE}db $C5,$FA,$10,$4C,$08,$FC;{$ENDIF} 
     {$IFDEF AVXSUP}vmovss xmm2, [edx + ecx - 4];                     {$ELSE}db $C5,$FA,$10,$54,$0A,$FC;{$ENDIF} 

     {$IFDEF AVXSUP}vmulss xmm2, xmm2, xmm0;                          {$ELSE}db $C5,$EA,$59,$D0;{$ENDIF} 
     {$IFDEF AVXSUP}vaddss xmm1, xmm1, xmm2;                          {$ELSE}db $C5,$F2,$58,$CA;{$ENDIF} 

     {$IFDEF AVXSUP}vmovss [eax + ecx - 4], xmm1;                     {$ELSE}db $C5,$FA,$11,$4C,$08,$FC;{$ENDIF} 

   jmp @loop3;
   @loop3End:

               {$IFDEF AVXSUP}vzeroupper;                             {$ELSE}db $C5,$F8,$77;{$ENDIF} 
   pop esi;
end;

{$ENDIF}

end.
