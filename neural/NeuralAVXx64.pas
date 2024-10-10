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
{$IF CompilerVersion<135}       // delhi compiler bug prevents on AVX512 -> use a very future compiler version...
{$UNDEF AVXSUP}
{$IFEND}
{$ENDIF}

// performs Result = sum(x[i]*y[i]);
function AVX2DotProd( x : PSingle; y : PSingle; N : integer ) : single;  {$IFDEF FPC}assembler;{$ENDIF}
function AVX512DotProd( x : PSingle; y : PSingle; N : integer ) : single; {$IFDEF FPC} assembler; {$ENDIF}

// performs x[i] = x[i] + fact*y[i];
procedure AVX2MulAdd( x : PSingle; y : PSingle; N : integer; const fact : single); {$IFDEF FPC} assembler; {$ENDIF}
procedure AVX512MulAdd( x : PSingle; y : PSingle; N : integer; const fact : single); {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}


{$ENDIF}

implementation

{$IFDEF x64}

{$IFDEF FPC} {$ASMMODE intel} {$S-} {$ENDIF}

function AVX2DotProd( x : PSingle; y : PSingle; N : integer ) : single;
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

   {$IFDEF AVXSUP}vxorpd ymm0, ymm0, ymm0;                            {$ELSE}db $C5,$FD,$57,$C0;{$ENDIF} 

   // unrolled loop
   @Loop1:
       add r8, 128;
       jg @loopEnd1;

       {$IFDEF AVXSUP}vmovupd ymm1, [rcx + r8 - 128];                 {$ELSE}db $C4,$A1,$7D,$10,$4C,$01,$80;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm2, [rdx + r8 - 128];                 {$ELSE}db $C4,$A1,$7D,$10,$54,$02,$80;{$ENDIF} 
       {$IFDEF AVXSUP}vmulps ymm1, ymm1, ymm2;                        {$ELSE}db $C5,$F4,$59,$CA;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm0, ymm0, ymm1;                        {$ELSE}db $C5,$FC,$58,$C1;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd ymm3, [rcx + r8 - 96];                  {$ELSE}db $C4,$A1,$7D,$10,$5C,$01,$A0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm4, [rdx + r8 - 96];                  {$ELSE}db $C4,$A1,$7D,$10,$64,$02,$A0;{$ENDIF} 
       {$IFDEF AVXSUP}vmulps ymm3, ymm3, ymm4;                        {$ELSE}db $C5,$E4,$59,$DC;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm0, ymm0, ymm3;                        {$ELSE}db $C5,$FC,$58,$C3;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd ymm1, [rcx + r8 - 64];                  {$ELSE}db $C4,$A1,$7D,$10,$4C,$01,$C0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm2, [rdx + r8 - 64];                  {$ELSE}db $C4,$A1,$7D,$10,$54,$02,$C0;{$ENDIF} 
       {$IFDEF AVXSUP}vmulps ymm1, ymm1, ymm2;                        {$ELSE}db $C5,$F4,$59,$CA;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm0, ymm0, ymm1;                        {$ELSE}db $C5,$FC,$58,$C1;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd ymm3, [rcx + r8 - 32];                  {$ELSE}db $C4,$A1,$7D,$10,$5C,$01,$E0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd ymm4, [rdx + r8 - 32];                  {$ELSE}db $C4,$A1,$7D,$10,$64,$02,$E0;{$ENDIF} 
       {$IFDEF AVXSUP}vmulps ymm3, ymm3, ymm4;                        {$ELSE}db $C5,$E4,$59,$DC;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps ymm0, ymm0, ymm3;                        {$ELSE}db $C5,$FC,$58,$C3;{$ENDIF} 

   jmp @Loop1;

   @loopEnd1:

       {$IFDEF AVXSUP}vextractf128 xmm2, ymm0, 1;                     {$ELSE}db $C4,$E3,$7D,$19,$C2,$01;{$ENDIF} 
       {$IFDEF AVXSUP}vhaddps xmm0, xmm0, xmm2;                       {$ELSE}db $C5,$FB,$7C,$C2;{$ENDIF} 

   sub r8, 128;
   jz @loop3End;

   // loop to get all fitting into an array of 4
   @Loop2:
      add r8, 16;
      jg @Loop2End;

       {$IFDEF AVXSUP}vmovupd xmm3, [rcx + r8 - 16];                  {$ELSE}db $C4,$A1,$79,$10,$5C,$01,$F0;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd xmm4, [rdx + r8 - 16];                  {$ELSE}db $C4,$A1,$79,$10,$64,$02,$F0;{$ENDIF} 
       {$IFDEF AVXSUP}vmulps xmm3, xmm3, xmm4;                        {$ELSE}db $C5,$E0,$59,$DC;{$ENDIF} 
       {$IFDEF AVXSUP}vaddps xmm0, xmm0, xmm3;                        {$ELSE}db $C5,$F8,$58,$C3;{$ENDIF} 
   jmp @Loop2;

   @Loop2End:

   // handle last 2 elements
   sub r8, 16;
   jz @loop3End;

   @loop3:
     add r8, 4;
     jg @loop3End;

       {$IFDEF AVXSUP}vmovss xmm3, [rcx + r8 - 4];                    {$ELSE}db $C4,$A1,$7A,$10,$5C,$01,$FC;{$ENDIF} 
       {$IFDEF AVXSUP}vmovss xmm4, [rdx + r8 - 4];                    {$ELSE}db $C4,$A1,$7A,$10,$64,$02,$FC;{$ENDIF} 
       {$IFDEF AVXSUP}vmulss xmm3, xmm3, xmm4;                        {$ELSE}db $C5,$E2,$59,$DC;{$ENDIF} 
       {$IFDEF AVXSUP}vaddss xmm0, xmm0, xmm3;                        {$ELSE}db $C5,$FA,$58,$C3;{$ENDIF} 

   jmp @loop3;
   @loop3End:

   // build result
   {$IFDEF AVXSUP}vhaddps xmm0, xmm0, xmm0;                           {$ELSE}db $C5,$FB,$7C,$C0;{$ENDIF} 
   {$IFDEF AVXSUP}vhaddps xmm0, xmm0, xmm0;                           {$ELSE}db $C5,$FB,$7C,$C0;{$ENDIF} 
   {$IFDEF AVXSUP}vzeroupper;                                         {$ELSE}db $C5,$F8,$77;{$ENDIF} 
   movss Result, xmm0;
end;


function AVX512DotProd( x : PSingle; y : PSingle; N : integer ) : single; {$IFDEF FPC} assembler; {$ENDIF}
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

   // adjust pointers for reverse array access
   sub rcx, r8;
   sub rdx, r8;

   {$IFDEF AVXSUP}vxorpd ymm0, ymm0, ymm0;                            {$ELSE}db $C5,$FD,$57,$C0;{$ENDIF} 
   {$IFDEF AVXSUP}vxorps zmm5, zmm5, zmm5;                            {$ELSE}db $62,$F1,$54,$48,$57,$ED;{$ENDIF} 

   // unrolled loop
   @Loop1:
       add r8, 256;
       jg @loopEnd1;

       {$IFDEF AVXSUP}vmovupd zmm1, [rcx + r8 - 256];                 {$ELSE}db $62,$B1,$FD,$48,$10,$4C,$01,$FC;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [rdx + r8 - 256];                 {$ELSE}db $62,$B1,$FD,$48,$10,$54,$02,$FC;{$ENDIF} 
       {$IFDEF AVXSUP}vfmadd231ps zmm5, zmm1, zmm2;                   {$ELSE}db $62,$F2,$75,$48,$B8,$EA;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd zmm3, [rcx + r8 - 192];                 {$ELSE}db $62,$B1,$FD,$48,$10,$5C,$01,$FD;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm4, [rdx + r8 - 192];                 {$ELSE}db $62,$B1,$FD,$48,$10,$64,$02,$FD;{$ENDIF} 
       {$IFDEF AVXSUP}vfmadd231ps zmm5, zmm3, zmm4;                   {$ELSE}db $62,$F2,$65,$48,$B8,$EC;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd zmm1, [rcx + r8 - 128];                 {$ELSE}db $62,$B1,$FD,$48,$10,$4C,$01,$FE;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [rdx + r8 - 128];                 {$ELSE}db $62,$B1,$FD,$48,$10,$54,$02,$FE;{$ENDIF} 
       {$IFDEF AVXSUP}vfmadd231ps zmm5, zmm1, zmm2;                   {$ELSE}db $62,$F2,$75,$48,$B8,$EA;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd zmm3, [rcx + r8 - 64];                  {$ELSE}db $62,$B1,$FD,$48,$10,$5C,$01,$FF;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm4, [rdx + r8 - 64];                  {$ELSE}db $62,$B1,$FD,$48,$10,$64,$02,$FF;{$ENDIF} 
       {$IFDEF AVXSUP}vfmadd231ps zmm5, zmm3, zmm4;                   {$ELSE}db $62,$F2,$65,$48,$B8,$EC;{$ENDIF} 

   jmp @Loop1;

   @loopEnd1:

   sub r8, 256;
   jz @buildRes;

   @Loop2:
       add r8, 64;
       jg @loopEnd2;

       {$IFDEF AVXSUP}vmovupd zmm1, [rcx + r8 - 64];                  {$ELSE}db $62,$B1,$FD,$48,$10,$4C,$01,$FF;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [rdx + r8 - 64];                  {$ELSE}db $62,$B1,$FD,$48,$10,$54,$02,$FF;{$ENDIF} 
       {$IFDEF AVXSUP}vfmadd231ps zmm5, zmm1, zmm2;                   {$ELSE}db $62,$F2,$75,$48,$B8,$EA;{$ENDIF} 
   jmp @Loop2;
   @loopEnd2:

   sub r8, 64;
   jz @buildRes;

   // loop to get all fitting into an array of 4
   @Loop3:
      add r8, 16;
      jg @Loop3End;

         {$IFDEF AVXSUP}vmovupd xmm3, [rcx + r8 - 16];                {$ELSE}db $C4,$A1,$79,$10,$5C,$01,$F0;{$ENDIF} 
         {$IFDEF AVXSUP}vmovupd xmm4, [rdx + r8 - 16];                {$ELSE}db $C4,$A1,$79,$10,$64,$02,$F0;{$ENDIF} 
         {$IFDEF AVXSUP}vmulps xmm3, xmm3, xmm4;                      {$ELSE}db $C5,$E0,$59,$DC;{$ENDIF} 
         {$IFDEF AVXSUP}vaddps xmm0, xmm0, xmm3;                      {$ELSE}db $C5,$F8,$58,$C3;{$ENDIF} 
   jmp @Loop3;
   @Loop3End:

   sub r8, 16;
   jz @buildRes;

   // handle last elements
   @loop4:
     add r8, 4;
     jg @loop4End;

         {$IFDEF AVXSUP}vmovss xmm3, [rcx + r8 - 4];                  {$ELSE}db $C4,$A1,$7A,$10,$5C,$01,$FC;{$ENDIF} 
         {$IFDEF AVXSUP}vmovss xmm4, [rdx + r8 - 4];                  {$ELSE}db $C4,$A1,$7A,$10,$64,$02,$FC;{$ENDIF} 
         {$IFDEF AVXSUP}vmulss xmm3, xmm3, xmm4;                      {$ELSE}db $C5,$E2,$59,$DC;{$ENDIF} 
         {$IFDEF AVXSUP}vaddss xmm0, xmm0, xmm3;                      {$ELSE}db $C5,$FA,$58,$C3;{$ENDIF} 

   jmp @loop4;
   @loop4End:
   @buildRes:

   // add result from the zmm register to xmm
   {$IFDEF AVXSUP}VEXTRACTF32x4 xmm1, zmm5, 0;                        {$ELSE}db $62,$F3,$7D,$48,$19,$E9,$00;{$ENDIF} 
   {$IFDEF AVXSUP}VEXTRACTF32x4 xmm2, zmm5, 1;                        {$ELSE}db $62,$F3,$7D,$48,$19,$EA,$01;{$ENDIF} 
   {$IFDEF AVXSUP}VEXTRACTF32x4 xmm3, zmm5, 2;                        {$ELSE}db $62,$F3,$7D,$48,$19,$EB,$02;{$ENDIF} 
   {$IFDEF AVXSUP}VEXTRACTF32x4 xmm4, zmm5, 3;                        {$ELSE}db $62,$F3,$7D,$48,$19,$EC,$03;{$ENDIF} 

   {$IFDEF AVXSUP}vaddps xmm1, xmm1, xmm2;                            {$ELSE}db $C5,$F0,$58,$CA;{$ENDIF} 
   {$IFDEF AVXSUP}vaddps xmm3, xmm3, xmm4;                            {$ELSE}db $C5,$E0,$58,$DC;{$ENDIF} 
   {$IFDEF AVXSUP}vaddps xmm5, xmm1, xmm3;                            {$ELSE}db $C5,$F0,$58,$EB;{$ENDIF} 
   {$IFDEF AVXSUP}vaddps xmm0, xmm0, xmm5;                            {$ELSE}db $C5,$F8,$58,$C5;{$ENDIF} 

   // build result
   {$IFDEF AVXSUP}vhaddps xmm0, xmm0, xmm0;                           {$ELSE}db $C5,$FB,$7C,$C0;{$ENDIF} 
   {$IFDEF AVXSUP}vhaddps xmm0, xmm0, xmm0;                           {$ELSE}db $C5,$FB,$7C,$C0;{$ENDIF} 
   {$IFDEF AVXSUP}vzeroupper;                                         {$ELSE}db $C5,$F8,$77;{$ENDIF} 
   movss Result, xmm0;
end;


procedure AVX2MulAdd( x : PSingle; y : PSingle; N : integer; const fact : single); {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}
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

procedure AVX512MulAdd( x : PSingle; y : PSingle; N : integer; const fact : single); {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}
// eax = x, edx = y, ecx = N
var r : Array[0..15] of single;
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

   // broadcast factor to zmm0
   {$IFDEF AVXSUP}vbroadcastss zmm0, xmm3;                            {$ELSE}db $62,$F2,$7D,$48,$18,$C3;{$ENDIF} 

   // unrolled loop
   @Loop1:
       add r8, 256;
       jg @loopEnd1;

       {$IFDEF AVXSUP}vmovupd zmm1, [rcx + r8 - 256];                 {$ELSE}db $62,$B1,$FD,$48,$10,$4C,$01,$FC;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [rdx + r8 - 256];                 {$ELSE}db $62,$B1,$FD,$48,$10,$54,$02,$FC;{$ENDIF} 
       {$IFDEF AVXSUP}vfmadd231ps zmm1, zmm2, zmm0;                   {$ELSE}db $62,$F2,$6D,$48,$B8,$C8;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [rcx + r8 - 256], zmm1;                 {$ELSE}db $62,$B1,$FD,$48,$11,$4C,$01,$FC;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd zmm1, [rcx + r8 - 192];                 {$ELSE}db $62,$B1,$FD,$48,$10,$4C,$01,$FD;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [rdx + r8 - 192];                 {$ELSE}db $62,$B1,$FD,$48,$10,$54,$02,$FD;{$ENDIF} 

       {$IFDEF AVXSUP}vfmadd231ps zmm1, zmm2, zmm0;                   {$ELSE}db $62,$F2,$6D,$48,$B8,$C8;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [rcx + r8 - 192], zmm1;                 {$ELSE}db $62,$B1,$FD,$48,$11,$4C,$01,$FD;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd zmm1, [rcx + r8 - 128];                 {$ELSE}db $62,$B1,$FD,$48,$10,$4C,$01,$FE;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [rdx + r8 - 128];                 {$ELSE}db $62,$B1,$FD,$48,$10,$54,$02,$FE;{$ENDIF} 

       {$IFDEF AVXSUP}vfmadd231ps zmm1, zmm2, zmm0;                   {$ELSE}db $62,$F2,$6D,$48,$B8,$C8;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [rcx + r8 - 128], zmm1;                 {$ELSE}db $62,$B1,$FD,$48,$11,$4C,$01,$FE;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd zmm1, [rcx + r8 - 64];                  {$ELSE}db $62,$B1,$FD,$48,$10,$4C,$01,$FF;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [rdx + r8 - 64];                  {$ELSE}db $62,$B1,$FD,$48,$10,$54,$02,$FF;{$ENDIF} 

       {$IFDEF AVXSUP}vfmadd231ps zmm1, zmm2, zmm0;                   {$ELSE}db $62,$F2,$6D,$48,$B8,$C8;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [rcx + r8 - 64], zmm1;                  {$ELSE}db $62,$B1,$FD,$48,$11,$4C,$01,$FF;{$ENDIF} 
   jmp @Loop1;

   @loopEnd1:

   sub r8, 256;
   jz @exitProc;

   @Loop2:
      add r8, 64;
      jg @loopEnd2;

      {$IFDEF AVXSUP}vmovupd zmm1, [rcx + r8 - 64];                   {$ELSE}db $62,$B1,$FD,$48,$10,$4C,$01,$FF;{$ENDIF} 
      {$IFDEF AVXSUP}vmovupd zmm2, [rdx + r8 - 64];                   {$ELSE}db $62,$B1,$FD,$48,$10,$54,$02,$FF;{$ENDIF} 

      {$IFDEF AVXSUP}vfmadd231ps zmm1, zmm2, zmm0;                    {$ELSE}db $62,$F2,$6D,$48,$B8,$C8;{$ENDIF} 

      {$IFDEF AVXSUP}vmovupd [rcx + r8 - 64], zmm1;                   {$ELSE}db $62,$B1,$FD,$48,$11,$4C,$01,$FF;{$ENDIF} 
   jmp @loop2;

   @loopEnd2:

   sub r8, 64;
   jz @exitProc;

   // loop to get all fitting into an array of 4
   @Loop3:
      add r8, 16;
      jg @Loop3End;

      {$IFDEF AVXSUP}vmovupd xmm1, [rcx + r8 - 16];                   {$ELSE}db $C4,$A1,$79,$10,$4C,$01,$F0;{$ENDIF} 
      {$IFDEF AVXSUP}vmovupd xmm2, [rdx + r8 - 16];                   {$ELSE}db $C4,$A1,$79,$10,$54,$02,$F0;{$ENDIF} 

      {$IFDEF AVXSUP}vmulps xmm2, xmm2, xmm0;                         {$ELSE}db $C5,$E8,$59,$D0;{$ENDIF} 
      {$IFDEF AVXSUP}vaddps xmm1, xmm1, xmm2;                         {$ELSE}db $C5,$F0,$58,$CA;{$ENDIF} 

      {$IFDEF AVXSUP}vmovupd [rcx + r8 - 16], xmm1;                   {$ELSE}db $C4,$A1,$79,$11,$4C,$01,$F0;{$ENDIF} 
   jmp @Loop3;
   @Loop3End:

   // handle last elements
   sub r8, 16;
   jz @exitProc;

   @loop4:
     add r8, 4;
     jg @loop4End;

     {$IFDEF AVXSUP}vmovss xmm1, [rcx + r8 - 4];                      {$ELSE}db $C4,$A1,$7A,$10,$4C,$01,$FC;{$ENDIF} 
     {$IFDEF AVXSUP}vmovss xmm2, [rdx + r8 - 4];                      {$ELSE}db $C4,$A1,$7A,$10,$54,$02,$FC;{$ENDIF} 

     {$IFDEF AVXSUP}vmulss xmm2, xmm2, xmm0;                          {$ELSE}db $C5,$EA,$59,$D0;{$ENDIF} 
     {$IFDEF AVXSUP}vaddss xmm1, xmm1, xmm2;                          {$ELSE}db $C5,$F2,$58,$CA;{$ENDIF} 

     {$IFDEF AVXSUP}vmovss [rcx + r8 - 4], xmm1;                      {$ELSE}db $C4,$A1,$7A,$11,$4C,$01,$FC;{$ENDIF} 

   jmp @loop4;
   @loop4End:

   @exitProc:
   {$IFDEF AVXSUP}vzeroupper;                                         {$ELSE}db $C5,$F8,$77;{$ENDIF} 
end;

{$ENDIF}

end.
