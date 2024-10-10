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
{$IF CompilerVersion<135}       // delhi compiler bug prevents on AVX512 -> use a very future compiler version...
{$UNDEF AVXSUP}
{$IFEND}
{$ENDIF}


// performs Result = sum(x[i]*y[i]);
function AVX2DotProd( x : PSingle; y : PSingle; N : integer ) : single; {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}
function AVX512DotProd( x : PSingle; y : PSingle; N : integer ) : single; {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}

// performs x[i] = x[i] + fact*y[i];
procedure AVX2MulAdd( x : PSingle; y : PSingle; N : integer; const fact : single); {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}
procedure AVX512MulAdd( x : PSingle; y : PSingle; N : integer; const fact : single); {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}

{$ENDIF}

implementation

{$IFNDEF x64}

{$IFDEF FPC} {$ASMMODE intel} {$S-} {$ENDIF}

function AVX2DotProd( x : PSingle; y : PSingle; N : integer ) : single;
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

function AVX512DotProd( x : PSingle; y : PSingle; N : integer ) : single; {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}
// eax = x, edx = y, ecx = N
asm
   // iters
   imul ecx, -4;

   // helper registers for the x, y pointers
   sub eax, ecx;
   sub edx, ecx;

   {$IFDEF AVXSUP}vxorps zmm5, zmm5, zmm5;                            {$ELSE}db $62,$F1,$54,$48,$57,$ED;{$ENDIF} 
   {$IFDEF AVXSUP}vxorps ymm0, ymm0, ymm0;                            {$ELSE}db $C5,$FC,$57,$C0;{$ENDIF} 

   // unrolled loop
   @Loop1:
       add ecx, 256;
       jg @loopEnd1;

       {$IFDEF AVXSUP}vmovupd zmm1, [eax + ecx - 256];                {$ELSE}db $62,$F1,$FD,$48,$10,$4C,$08,$FC;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [edx + ecx - 256];                {$ELSE}db $62,$F1,$FD,$48,$10,$54,$0A,$FC;{$ENDIF} 
       {$IFDEF AVXSUP}vfmadd231ps zmm5, zmm1, zmm2;                   {$ELSE}db $62,$F2,$75,$48,$B8,$EA;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd zmm3, [eax + ecx - 192];                {$ELSE}db $62,$F1,$FD,$48,$10,$5C,$08,$FD;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm4, [edx + ecx - 192];                {$ELSE}db $62,$F1,$FD,$48,$10,$64,$0A,$FD;{$ENDIF} 
       {$IFDEF AVXSUP}vfmadd231ps zmm5, zmm3, zmm4;                   {$ELSE}db $62,$F2,$65,$48,$B8,$EC;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd zmm1, [eax + ecx - 128];                {$ELSE}db $62,$F1,$FD,$48,$10,$4C,$08,$FE;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [edx + ecx - 128];                {$ELSE}db $62,$F1,$FD,$48,$10,$54,$0A,$FE;{$ENDIF} 
       {$IFDEF AVXSUP}vfmadd231ps zmm5, zmm1, zmm2;                   {$ELSE}db $62,$F2,$75,$48,$B8,$EA;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd zmm3, [eax + ecx - 64];                 {$ELSE}db $62,$F1,$FD,$48,$10,$5C,$08,$FF;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm4, [edx + ecx - 64];                 {$ELSE}db $62,$F1,$FD,$48,$10,$64,$0A,$FF;{$ENDIF} 
       {$IFDEF AVXSUP}vfmadd231ps zmm5, zmm3, zmm4;                   {$ELSE}db $62,$F2,$65,$48,$B8,$EC;{$ENDIF} 

   jmp @Loop1;

   @loopEnd1:

   sub ecx, 256;
   jz @buildRes;

   @Loop2:
      add ecx, 64;
      jg @Loop2End;

      {$IFDEF AVXSUP}vmovupd zmm1, [eax + ecx - 64];                  {$ELSE}db $62,$F1,$FD,$48,$10,$4C,$08,$FF;{$ENDIF} 
      {$IFDEF AVXSUP}vmovupd zmm2, [edx + ecx - 64];                  {$ELSE}db $62,$F1,$FD,$48,$10,$54,$0A,$FF;{$ENDIF} 
      {$IFDEF AVXSUP}vfmadd231ps zmm5, zmm1, zmm2;                    {$ELSE}db $62,$F2,$75,$48,$B8,$EA;{$ENDIF} 
   jmp @Loop2;

   @Loop2End:

   sub ecx, 64;
   jz @buildRes;

   // loop to get all fitting into an array of 4
   @Loop3:
      add ecx, 16;
      jg @Loop3End;

      {$IFDEF AVXSUP}vmovupd xmm3, [eax + ecx - 16];                  {$ELSE}db $C5,$F9,$10,$5C,$08,$F0;{$ENDIF} 
      {$IFDEF AVXSUP}vmovupd xmm4, [edx + ecx - 16];                  {$ELSE}db $C5,$F9,$10,$64,$0A,$F0;{$ENDIF} 
      {$IFDEF AVXSUP}vmulps xmm3, xmm3, xmm4;                         {$ELSE}db $C5,$E0,$59,$DC;{$ENDIF} 
      {$IFDEF AVXSUP}vaddps xmm0, xmm0, xmm3;                         {$ELSE}db $C5,$F8,$58,$C3;{$ENDIF} 
   jmp @Loop3;

   @Loop3End:

   // handle last 2 elements
   sub ecx, 16;
   jz @buildRes;

   @loop4:
     add ecx, 4;
     jg @loop4End;

     {$IFDEF AVXSUP}vmovss xmm3, [eax + ecx - 4];                     {$ELSE}db $C5,$FA,$10,$5C,$08,$FC;{$ENDIF} 
     {$IFDEF AVXSUP}vmovss xmm4, [edx + ecx - 4];                     {$ELSE}db $C5,$FA,$10,$64,$0A,$FC;{$ENDIF} 
     {$IFDEF AVXSUP}vmulss xmm3, xmm3, xmm4;                          {$ELSE}db $C5,$E2,$59,$DC;{$ENDIF} 
     {$IFDEF AVXSUP}vaddss xmm0, xmm0, xmm3;                          {$ELSE}db $C5,$FA,$58,$C3;{$ENDIF} 

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

procedure AVX512MulAdd( x : PSingle; y : PSingle; N : integer; const fact : single); {$IFDEF FPC} assembler; {$ELSE} register; {$ENDIF}
// eax = x, edx = y, ecx = N
asm
   push esi;

   // broadcast factor to zmm0
   lea esi, fact;
   {$IFDEF AVXSUP}vbroadcastss zmm0, fact;                            {$ELSE}db $62,$F2,$7D,$48,$18,$45,$02;{$ENDIF} 

   // iters
   imul ecx, -4;

   // helper registers for the x, y
   sub eax, ecx;
   sub edx, ecx;

   // unrolled loop
   @Loop1:
       add ecx, 256;
       jg @loopEnd1;

       {$IFDEF AVXSUP}vmovupd zmm1, [eax + ecx - 256];                {$ELSE}db $62,$F1,$FD,$48,$10,$4C,$08,$FC;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [edx + ecx - 256];                {$ELSE}db $62,$F1,$FD,$48,$10,$54,$0A,$FC;{$ENDIF} 

       {$IFDEF AVXSUP}vfmadd231ps zmm1, zmm2, zmm0;                   {$ELSE}db $62,$F2,$6D,$48,$B8,$C8;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [eax + ecx - 256], zmm1;                {$ELSE}db $62,$F1,$FD,$48,$11,$4C,$08,$FC;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd zmm1, [eax + ecx - 192];                {$ELSE}db $62,$F1,$FD,$48,$10,$4C,$08,$FD;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [edx + ecx - 192];                {$ELSE}db $62,$F1,$FD,$48,$10,$54,$0A,$FD;{$ENDIF} 

       {$IFDEF AVXSUP}vfmadd231ps zmm1, zmm2, zmm0;                   {$ELSE}db $62,$F2,$6D,$48,$B8,$C8;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [eax + ecx - 192], zmm1;                {$ELSE}db $62,$F1,$FD,$48,$11,$4C,$08,$FD;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd zmm1, [eax + ecx - 128];                {$ELSE}db $62,$F1,$FD,$48,$10,$4C,$08,$FE;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [edx + ecx - 128];                {$ELSE}db $62,$F1,$FD,$48,$10,$54,$0A,$FE;{$ENDIF} 

       {$IFDEF AVXSUP}vfmadd231ps zmm1, zmm2, zmm0;                   {$ELSE}db $62,$F2,$6D,$48,$B8,$C8;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [eax + ecx - 128], zmm1;                {$ELSE}db $62,$F1,$FD,$48,$11,$4C,$08,$FE;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd zmm1, [eax + ecx - 64];                 {$ELSE}db $62,$F1,$FD,$48,$10,$4C,$08,$FF;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [edx + ecx - 64];                 {$ELSE}db $62,$F1,$FD,$48,$10,$54,$0A,$FF;{$ENDIF} 

       {$IFDEF AVXSUP}vfmadd231ps zmm1, zmm2, zmm0;                   {$ELSE}db $62,$F2,$6D,$48,$B8,$C8;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [eax + ecx - 64], zmm1;                 {$ELSE}db $62,$F1,$FD,$48,$11,$4C,$08,$FF;{$ENDIF} 
   jmp @Loop1;

   @loopEnd1:

   sub ecx, 256;
   jz @exitProc;

   @Loop2:
       add ecx, 64;
       jg @loopEnd2;

       {$IFDEF AVXSUP}vmovupd zmm1, [eax + ecx - 64];                 {$ELSE}db $62,$F1,$FD,$48,$10,$4C,$08,$FF;{$ENDIF} 
       {$IFDEF AVXSUP}vmovupd zmm2, [edx + ecx - 64];                 {$ELSE}db $62,$F1,$FD,$48,$10,$54,$0A,$FF;{$ENDIF} 

       {$IFDEF AVXSUP}vfmadd231ps zmm1, zmm2, zmm0;                   {$ELSE}db $62,$F2,$6D,$48,$B8,$C8;{$ENDIF} 

       {$IFDEF AVXSUP}vmovupd [eax + ecx - 64], zmm1;                 {$ELSE}db $62,$F1,$FD,$48,$11,$4C,$08,$FF;{$ENDIF} 
   jmp @Loop2;

   @LoopEnd2:
   sub ecx, 64;
   jz @exitProc;

   // loop to get all fitting into an array of 4
   @Loop3:
      add ecx, 16;
      jg @Loop3End;

      {$IFDEF AVXSUP}vmovupd xmm1, [eax + ecx - 16];                  {$ELSE}db $C5,$F9,$10,$4C,$08,$F0;{$ENDIF} 
      {$IFDEF AVXSUP}vmovupd xmm2, [edx + ecx - 16];                  {$ELSE}db $C5,$F9,$10,$54,$0A,$F0;{$ENDIF} 

      {$IFDEF AVXSUP}vmulps xmm2, xmm2, xmm0;                         {$ELSE}db $C5,$E8,$59,$D0;{$ENDIF} 
      {$IFDEF AVXSUP}vaddps xmm1, xmm1, xmm2;                         {$ELSE}db $C5,$F0,$58,$CA;{$ENDIF} 

      {$IFDEF AVXSUP}vmovupd [eax + ecx - 16], xmm1;                  {$ELSE}db $C5,$F9,$11,$4C,$08,$F0;{$ENDIF} 
   jmp @Loop3;

   @Loop3End:

   // handle last 2 elements
   sub ecx, 16;
   jz @exitProc;

   @loop4:
     add ecx, 4;
     jg @loop4End;

     {$IFDEF AVXSUP}vmovss xmm1, [eax + ecx - 4];                     {$ELSE}db $C5,$FA,$10,$4C,$08,$FC;{$ENDIF} 
     {$IFDEF AVXSUP}vmovss xmm2, [edx + ecx - 4];                     {$ELSE}db $C5,$FA,$10,$54,$0A,$FC;{$ENDIF} 

     {$IFDEF AVXSUP}vmulss xmm2, xmm2, xmm0;                          {$ELSE}db $C5,$EA,$59,$D0;{$ENDIF} 
     {$IFDEF AVXSUP}vaddss xmm1, xmm1, xmm2;                          {$ELSE}db $C5,$F2,$58,$CA;{$ENDIF} 

     {$IFDEF AVXSUP}vmovss [eax + ecx - 4], xmm1;                     {$ELSE}db $C5,$FA,$11,$4C,$08,$FC;{$ENDIF} 

   jmp @loop4;
   @loop4End:
   @exitProc:

   {$IFDEF AVXSUP}vzeroupper;                                         {$ELSE}db $C5,$F8,$77;{$ENDIF} 
   pop esi;
end;

{$ENDIF}

end.
