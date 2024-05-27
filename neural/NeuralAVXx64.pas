unit NeuralAVXx64;

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

{$DEFINE AVXSUP}  // assembler support for AVX/FMA built in
{$IFNDEF FPC}
{$IF CompilerVersion<35}
{$UNDEF AVXSUP}
{$IFEND}
{$ENDIF}

// performs Result = sum(x[i]*y[i]);
function AVXDotProd( x : PSingle; y : PSingle; N : integer ) : single;  {$IFDEF FPC}assembler;{$ENDIF}

// performs x[i] = x[i] + fact*y[i];
procedure AVXMulAdd( x : PSingle; y : PSingle; N : integer; const fact : single); {$IFDEF FPC} assembler; {$ENDIF}


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

               {$IFDEF AVXSUP}vxorpd ymm0, ymm0, ymm0;                {$ELSE}db $C5,$FD,$57,$C0;{$ENDIF} 

   // unrolled loop
   @Loop1:
       add r8, 128;
       jg @loopEnd1;

                   {$IFDEF AVXSUP}vmovupd ymm1, [rcx + r8 - 128];     {$ELSE}db $C4,$A1,$7D,$10,$4C,$01,$80;{$ENDIF} 
                   {$IFDEF AVXSUP}vmovupd ymm2, [rdx + r8 - 128];     {$ELSE}db $C4,$A1,$7D,$10,$54,$02,$80;{$ENDIF} 
                   {$IFDEF AVXSUP}vmulps ymm1, ymm1, ymm2;            {$ELSE}db $C5,$F4,$59,$CA;{$ENDIF} 
                   {$IFDEF AVXSUP}vaddps ymm0, ymm0, ymm1;            {$ELSE}db $C5,$FC,$58,$C1;{$ENDIF} 

                   {$IFDEF AVXSUP}vmovupd ymm3, [rcx + r8 - 96];      {$ELSE}db $C4,$A1,$7D,$10,$5C,$01,$A0;{$ENDIF} 
                   {$IFDEF AVXSUP}vmovupd ymm4, [rdx + r8 - 96];      {$ELSE}db $C4,$A1,$7D,$10,$64,$02,$A0;{$ENDIF} 
                   {$IFDEF AVXSUP}vmulps ymm3, ymm3, ymm4;            {$ELSE}db $C5,$E4,$59,$DC;{$ENDIF} 
                   {$IFDEF AVXSUP}vaddps ymm0, ymm0, ymm3;            {$ELSE}db $C5,$FC,$58,$C3;{$ENDIF} 

                   {$IFDEF AVXSUP}vmovupd ymm1, [rcx + r8 - 64];      {$ELSE}db $C4,$A1,$7D,$10,$4C,$01,$C0;{$ENDIF} 
                   {$IFDEF AVXSUP}vmovupd ymm2, [rdx + r8 - 64];      {$ELSE}db $C4,$A1,$7D,$10,$54,$02,$C0;{$ENDIF} 
                   {$IFDEF AVXSUP}vmulps ymm1, ymm1, ymm2;            {$ELSE}db $C5,$F4,$59,$CA;{$ENDIF} 
                   {$IFDEF AVXSUP}vaddps ymm0, ymm0, ymm1;            {$ELSE}db $C5,$FC,$58,$C1;{$ENDIF} 

                   {$IFDEF AVXSUP}vmovupd ymm3, [rcx + r8 - 32];      {$ELSE}db $C4,$A1,$7D,$10,$5C,$01,$E0;{$ENDIF} 
                   {$IFDEF AVXSUP}vmovupd ymm4, [rdx + r8 - 32];      {$ELSE}db $C4,$A1,$7D,$10,$64,$02,$E0;{$ENDIF} 
                   {$IFDEF AVXSUP}vmulps ymm3, ymm3, ymm4;            {$ELSE}db $C5,$E4,$59,$DC;{$ENDIF} 
                   {$IFDEF AVXSUP}vaddps ymm0, ymm0, ymm3;            {$ELSE}db $C5,$FC,$58,$C3;{$ENDIF} 

   jmp @Loop1;

   @loopEnd1:

               {$IFDEF AVXSUP}vextractf128 xmm2, ymm0, 1;             {$ELSE}db $C4,$E3,$7D,$19,$C2,$01;{$ENDIF} 
               {$IFDEF AVXSUP}vhaddps xmm0, xmm0, xmm2;               {$ELSE}db $C5,$FB,$7C,$C2;{$ENDIF} 

   sub r8, 128;
   jz @loop3End;

   // loop to get all fitting into an array of 4
   @Loop2:
      add r8, 16;
      jg @Loop2End;

                  {$IFDEF AVXSUP}vmovupd xmm3, [rcx + r8 - 16];       {$ELSE}db $C4,$A1,$79,$10,$5C,$01,$F0;{$ENDIF} 
                  {$IFDEF AVXSUP}vmovupd xmm4, [rdx + r8 - 16];       {$ELSE}db $C4,$A1,$79,$10,$64,$02,$F0;{$ENDIF} 
                  {$IFDEF AVXSUP}vmulps xmm3, xmm3, xmm4;             {$ELSE}db $C5,$E0,$59,$DC;{$ENDIF} 
                  {$IFDEF AVXSUP}vaddps xmm0, xmm0, xmm3;             {$ELSE}db $C5,$F8,$58,$C3;{$ENDIF} 
   jmp @Loop2;

   @Loop2End:

   // handle last 2 elements
   sub r8, 16;
   jz @loop3End;

   @loop3:
     add r8, 4;
     jg @loop3End;

                 {$IFDEF AVXSUP}vmovss xmm3, [rcx + r8 - 4];          {$ELSE}db $C4,$A1,$7A,$10,$5C,$01,$FC;{$ENDIF} 
                 {$IFDEF AVXSUP}vmovss xmm4, [rdx + r8 - 4];          {$ELSE}db $C4,$A1,$7A,$10,$64,$02,$FC;{$ENDIF} 
                 {$IFDEF AVXSUP}vmulss xmm3, xmm3, xmm4;              {$ELSE}db $C5,$E2,$59,$DC;{$ENDIF} 
                 {$IFDEF AVXSUP}vaddss xmm0, xmm0, xmm3;              {$ELSE}db $C5,$FA,$58,$C3;{$ENDIF} 

   jmp @loop3;
   @loop3End:

   // build result
               {$IFDEF AVXSUP}vhaddps xmm0, xmm0, xmm0;               {$ELSE}db $C5,$FB,$7C,$C0;{$ENDIF} 
               {$IFDEF AVXSUP}vhaddps xmm0, xmm0, xmm0;               {$ELSE}db $C5,$FB,$7C,$C0;{$ENDIF} 
               {$IFDEF AVXSUP}vzeroupper;                             {$ELSE}db $C5,$F8,$77;{$ENDIF} 
   movss Result, xmm0;
end;

procedure AVXMulAdd( x : PSingle; y : PSingle; N : integer; const fact : single); {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}
// eax = x, edx = y, ecx = N
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

   // broadcast factor to ymm0
   {$IFDEF AVXSUP}vbroadcastss ymm0, xmm3;                            {$ELSE}db $C4,$E2,$7D,$18,$C3;{$ENDIF} 

   // unrolled loop
   @Loop1:
       add r8, 128;
       jg @loopEnd1;

       {$IFDEF AVXSUP}vmovupd ymm1, [rcx + r8 - 128];                 {$ELSE}db $C4,$A1,$7D,$10,$4C,$01,$80;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm2, [rdx + r8 - 128];                 {$ELSE}db $C4,$A1,$7D,$10,$54,$02,$80;{$ENDIF} 

       {$IFDEF AVXSUP}vmulps ymm2, ymm2, ymm0;                        {$ELSE}db $C5,$EC,$59,$D0;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm1, ymm1, ymm2;                        {$ELSE}db $C5,$F4,$58,$CA;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [rcx + r8 - 128], ymm1;                 {$ELSE}db $C4,$A1,$7D,$11,$4C,$01,$80;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd ymm1, [rcx + r8 - 96];                  {$ELSE}db $C4,$A1,$7D,$10,$4C,$01,$A0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm2, [rdx + r8 - 96];                  {$ELSE}db $C4,$A1,$7D,$10,$54,$02,$A0;{$ENDIF} 

       {$IFDEF AVXSUP}vmulps ymm2, ymm2, ymm0;                        {$ELSE}db $C5,$EC,$59,$D0;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm1, ymm1, ymm2;                        {$ELSE}db $C5,$F4,$58,$CA;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [rcx + r8 - 96], ymm1;                  {$ELSE}db $C4,$A1,$7D,$11,$4C,$01,$A0;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd ymm1, [rcx + r8 - 64];                  {$ELSE}db $C4,$A1,$7D,$10,$4C,$01,$C0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm2, [rdx + r8 - 64];                  {$ELSE}db $C4,$A1,$7D,$10,$54,$02,$C0;{$ENDIF} 

       {$IFDEF AVXSUP}vmulps ymm2, ymm2, ymm0;                        {$ELSE}db $C5,$EC,$59,$D0;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm1, ymm1, ymm2;                        {$ELSE}db $C5,$F4,$58,$CA;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [rcx + r8 - 64], ymm1;                  {$ELSE}db $C4,$A1,$7D,$11,$4C,$01,$C0;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd ymm1, [rcx + r8 - 32];                  {$ELSE}db $C4,$A1,$7D,$10,$4C,$01,$E0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm2, [rdx + r8 - 32];                  {$ELSE}db $C4,$A1,$7D,$10,$54,$02,$E0;{$ENDIF} 

       {$IFDEF AVXSUP}vmulps ymm2, ymm2, ymm0;                        {$ELSE}db $C5,$EC,$59,$D0;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm1, ymm1, ymm2;                        {$ELSE}db $C5,$F4,$58,$CA;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [rcx + r8 - 32], ymm1;                  {$ELSE}db $C4,$A1,$7D,$11,$4C,$01,$E0;{$ENDIF} 
   jmp @Loop1;

   @loopEnd1:

   sub r8, 128;
   jz @loop3End;

   // loop to get all fitting into an array of 4
   @Loop2:
      add r8, 16;
      jg @Loop2End;

      {$IFDEF AVXSUP}vmovupd xmm1, [rcx + r8 - 16];                   {$ELSE}db $C4,$A1,$79,$10,$4C,$01,$F0;{$ENDIF} 
      {$IFDEF AVXSUP}vmovupd xmm2, [rdx + r8 - 16];                   {$ELSE}db $C4,$A1,$79,$10,$54,$02,$F0;{$ENDIF} 

      {$IFDEF AVXSUP}vmulps xmm2, xmm2, xmm0;                         {$ELSE}db $C5,$E8,$59,$D0;{$ENDIF} 
      {$IFDEF AVXSUP}vaddps xmm1, xmm1, xmm2;                         {$ELSE}db $C5,$F0,$58,$CA;{$ENDIF} 

      {$IFDEF AVXSUP}vmovupd [rcx + r8 - 16], xmm1;                   {$ELSE}db $C4,$A1,$79,$11,$4C,$01,$F0;{$ENDIF} 
   jmp @Loop2;

   @Loop2End:

   // handle last 2 elements
   sub r8, 16;
   jz @loop3End;

   @loop3:
     add r8, 4;
     jg @loop3End;

     {$IFDEF AVXSUP}vmovss xmm1, [rcx + r8 - 4];                      {$ELSE}db $C4,$A1,$7A,$10,$4C,$01,$FC;{$ENDIF} 
     {$IFDEF AVXSUP}vmovss xmm2, [rdx + r8 - 4];                      {$ELSE}db $C4,$A1,$7A,$10,$54,$02,$FC;{$ENDIF} 

     {$IFDEF AVXSUP}vmulss xmm2, xmm2, xmm0;                          {$ELSE}db $C5,$EA,$59,$D0;{$ENDIF} 
     {$IFDEF AVXSUP}vaddss xmm1, xmm1, xmm2;                          {$ELSE}db $C5,$F2,$58,$CA;{$ENDIF} 

     {$IFDEF AVXSUP}vmovss [rcx + r8 - 4], xmm1;                      {$ELSE}db $C4,$A1,$7A,$11,$4C,$01,$FC;{$ENDIF} 

   jmp @loop3;
   @loop3End:

               {$IFDEF AVXSUP}vzeroupper;                             {$ELSE}db $C5,$F8,$77;{$ENDIF} 
end;

{$ENDIF}

end.
