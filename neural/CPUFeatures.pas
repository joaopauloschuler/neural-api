// ###################################################################
// #### This file is part of the mathematics library project, and is
// #### offered under the licence agreement described on
// #### http://www.mrsoft.org/
// ####
// #### Copyright:(c) 2011, Michael R. . All rights reserved.
// ####
// #### Unless required by applicable law or agreed to in writing, software
// #### distributed under the License is distributed on an "AS IS" BASIS,
// #### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// #### See the License for the specific language governing permissions and
// #### limitations under the License.
// ###################################################################


unit CPUFeatures;

// unit to determine some cpu features

interface

function IsSSE3Present : boolean;
function IsAVXPresent : boolean;
function IsAVX512Present : boolean;
function IsFMAPresent : boolean;
function IsHardwareRNDSupport : boolean;
function IsHardwareRDSeed : boolean;

function GetCurrentProcessorNumber : LongWord; register;

implementation

// ###########################################
// #### Global constants for features:


// base idea from https://stackoverflow.com/questions/6121792/how-to-check-if-a-cpu-supports-the-sse3-instruction-set
// misc
var HW_MMX: boolean = False;
    HW_x64: boolean = False;
    HW_ABM: boolean = False;      // Advanced Bit Manipulation
    HW_RDRAND: boolean = False;
    HW_RDSEED: boolean = False;
    HW_BMI1: boolean = False;
    HW_BMI2: boolean = False;
    HW_ADX: boolean = False;
    HW_PREFETCHWT1: boolean = False;

    // SIMD: 128-bit
    HW_SSE: boolean = False;
    HW_SSE2: boolean = False;
    HW_SSE3: boolean = False;
    HW_SSSE3: boolean = False;
    HW_SSE41: boolean = False;
    HW_SSE42: boolean = False;
    HW_SSE4a: boolean = False;
    HW_AES: boolean = False;
    HW_SHA: boolean = False;

    // SIMD: 256-bit
    HW_AVX: boolean = False;
    HW_XOP: boolean = False;
    HW_FMA3: boolean = False;
    HW_FMA4: boolean = False;
    HW_AVX2: boolean = False;

    // SIMD: 512-bit
    HW_AVX512F: boolean = False;    //  AVX512 Foundation
    HW_AVX512CD: boolean = False;   //  AVX512 Conflict Detection
    HW_AVX512PF: boolean = False;   //  AVX512 Prefetch
    HW_AVX512ER: boolean = False;   //  AVX512 Exponential + Reciprocal
    HW_AVX512VL: boolean = False;   //  AVX512 Vector Length Extensions
    HW_AVX512BW: boolean = False;   //  AVX512 Byte + Word
    HW_AVX512DQ: boolean = False;   //  AVX512 Doubleword + Quadword
    HW_AVX512IFMA: boolean = False; //  AVX512 Integer 52-bit Fused Multiply-Add
    HW_AVX512VBMI: boolean = False; //  AVX512 Vector Byte Manipulation Instructions

    AVX_OS_SUPPORT : boolean = False;     // 256bit AVX supported in context switch
    AVX512_OS_SUPPORT : boolean = False;  // 512bit AVX supported in context switch

// ##############################################################
// #### feature detection code
// ##############################################################

type
  TRegisters = record
    EAX,
    EBX,
    ECX,
    EDX: Cardinal;
  end;

{$IFDEF FPC} {$ASMMODE intel} {$S-} {$ENDIF}

{$IFDEF CPUX64}
{$DEFINE x64}
{$ENDIF}
{$IFDEF cpux86_64}
{$DEFINE x64}
{$ENDIF}
{$IFDEF x64}

function IsCPUID_Available : boolean;
begin
     Result := true;
end;

procedure GetCPUID(Param: Cardinal; out Registers: TRegisters);
var iRBX, iRDI : int64;
{$IFDEF FPC}
begin
{$ENDIF}
asm
   mov iRBX, rbx;
   mov iRDI, rdi;

//   .pushnv rbx;                        {save affected registers}
//   .pushnv rdi;

   MOV     RDI, Registers
   MOV     EAX, Param;
   XOR     RBX, RBX                    {clear EBX register}
   XOR     RCX, RCX                    {clear ECX register}
   XOR     RDX, RDX                    {clear EDX register}
   DB $0F, $A2                         {CPUID opcode}
   MOV     TRegisters(RDI).&EAX, EAX   {save EAX register}
   MOV     TRegisters(RDI).&EBX, EBX   {save EBX register}
   MOV     TRegisters(RDI).&ECX, ECX   {save ECX register}
   MOV     TRegisters(RDI).&EDX, EDX   {save EDX register}

   // epilog
   mov rbx, iRBX;
   mov rdi, IRDI;
{$IFDEF FPC}
end;
{$ENDIF}
end;

{$ELSE}

function IsCPUID_Available: Boolean; register;
{$IFDEF FPC} begin {$ENDIF}
asm
   PUSHFD                 {save EFLAGS to stack}
   POP     EAX            {store EFLAGS in EAX}
   MOV     EDX, EAX       {save in EDX for later testing}
   XOR     EAX, $200000;  {flip ID bit in EFLAGS}
   PUSH    EAX            {save new EFLAGS value on stack}
   POPFD                  {replace current EFLAGS value}
   PUSHFD                 {get new EFLAGS}
   POP     EAX            {store new EFLAGS in EAX}
   XOR     EAX, EDX       {check if ID bit changed}
   JZ      @exit          {no, CPUID not available}
   MOV     EAX, True      {yes, CPUID is available}
@exit:
end;
{$IFDEF FPC} end; {$ENDIF}

procedure GetCPUID(Param: Cardinal; var Registers: TRegisters);
{$IFDEF FPC} begin {$ENDIF}
asm
   PUSH    EBX                         {save affected registers}
   PUSH    EDI
   MOV     EDI, Registers
   XOR     EBX, EBX                    {clear EBX register}
   XOR     ECX, ECX                    {clear ECX register}
   XOR     EDX, EDX                    {clear EDX register}
   DB $0F, $A2                         {CPUID opcode}
   MOV     TRegisters(EDI).&EAX, EAX   {save EAX register}
   MOV     TRegisters(EDI).&EBX, EBX   {save EBX register}
   MOV     TRegisters(EDI).&ECX, ECX   {save ECX register}
   MOV     TRegisters(EDI).&EDX, EDX   {save EDX register}
   POP     EDI                         {restore registers}
   POP     EBX
end;
{$IFDEF FPC} end; {$ENDIF}

{$ENDIF}


// ###########################################
// #### Local check for AVX support according to
// from https://software.intel.com/en-us/blogs/2011/04/14/is-avx-enabled
// and // from https://software.intel.com/content/www/us/en/develop/articles/how-to-detect-knl-instruction-support.html
procedure InitAVXOSSupportFlags; {$IFDEF FPC}assembler;{$ENDIF}
asm
   {$IFDEF x64}
   push rbx;
   {$ELSE}
   push ebx;
   {$ENDIF}

   xor eax, eax;
   cpuid;
   cmp eax, 1;
   jb @@endProc;

   mov eax, 1;
   cpuid;

   and ecx, $018000000; // check 27 bit (OS uses XSAVE/XRSTOR)
   cmp ecx, $018000000; // and 28 (AVX supported by CPU)
   jne @@endProc;

   xor ecx, ecx ; // XFEATURE_ENABLED_MASK/XCR0 register number = 0
   db $0F, $01, $D0; //xgetbv ; // XFEATURE_ENABLED_MASK register is in edx:eax
   and eax, $E6; //110b
   cmp eax, $E6; //1110 0011 = zmm_ymm_xmm = (7 << 5) | (1 << 2) | (1 << 1);
   jne @@not_supported;
   {$IFDEF x64}
   mov [rip + AVX512_OS_SUPPORT], 1;
   {$ELSE}
   mov AVX512_OS_SUPPORT, 1;
   {$ENDIF}
   @@not_supported:

   and eax, $6; //110b
   cmp eax, $6; //1110 0011 = check for AVX os support (256bit) in a context switch
   jne @@endProc;
   {$IFDEF x64}
   mov [rip + AVX_OS_SUPPORT], 1;
   {$ELSE}
   mov AVX_OS_SUPPORT, 1;
   {$ENDIF}

   @@endProc:

   {$IFDEF x64}
   pop rbx;
   {$ELSE}
   pop ebx;
   {$ENDIF}
end;

function GetCurrentProcessorNumber : LongWord; register; // stdcall; external 'Kernel32.dll';
{$IFDEF FPC}
begin
{$ENDIF}
asm
   mov eax, 1;
   DB $0F, $A2;  //cpuid;
   shr ebx, 24;
   mov eax, ebx;
{$IFDEF FPC}
end;
{$ENDIF}
end;

procedure InitFlags;
var nIds : LongWord;
    reg : TRegisters;
begin
     if IsCPUID_Available then
     begin
          GetCPUID(0, reg);
          nids := reg.EAX;


          if nids >= 1 then
          begin
               GetCPUID(1, reg);

               HW_MMX    := (reg.EDX and (1 shl 23)) <> 0;
               HW_SSE    := (reg.EDX and (1 shl 25)) <> 0;
               HW_SSE2   := (reg.EDX and (1 shl 26)) <> 0;
               HW_SSE3   := (reg.EDX and (1 shl 0)) <> 0;

               HW_SSSE3  := (reg.ECX and (1 shl 9)) <> 0;
               HW_SSE41  := (reg.ECX and (1 shl 19)) <> 0;
               HW_SSE42  := (reg.ECX and (1 shl 20)) <> 0;
               HW_AES    := (reg.ECX and (1 shl 25)) <> 0;

               HW_AVX    := (reg.ECX and (1 shl 28)) <> 0;
               HW_FMA3   := (reg.ECX and (1 shl 12)) <> 0;

               HW_RDRAND := (reg.ECX and (1 shl 30)) <> 0;
          end;

          if nids >= 7 then
          begin
               GetCPUID($7, reg);
               HW_AVX2        := (reg.EBX and (1 shl 5)) <> 0;

               HW_BMI1        := (reg.EBX and (1 shl  3)) <> 0;
               HW_BMI2        := (reg.EBX and (1 shl  8)) <> 0;
               HW_ADX         := (reg.EBX and (1 shl 19)) <> 0;
               HW_SHA         := (reg.EBX and (1 shl 29)) <> 0;
               HW_PREFETCHWT1 := (reg.EBX and (1 shl  0)) <> 0;
               HW_RDSEED      := (reg.EBX and (1 shl  18)) <> 0;

               HW_AVX512F     := (reg.EBX and (1 shl 16)) <> 0;
               HW_AVX512CD    := (reg.EBX and (1 shl 28)) <> 0;
               HW_AVX512PF    := (reg.EBX and (1 shl 26)) <> 0;
               HW_AVX512ER    := (reg.EBX and (1 shl 27)) <> 0;
               HW_AVX512VL    := (reg.EBX and (1 shl 31)) <> 0;
               HW_AVX512BW    := (reg.EBX and (1 shl 30)) <> 0;
               HW_AVX512DQ    := (reg.EBX and (1 shl 17)) <> 0;
               HW_AVX512IFMA  := (reg.EBX and (1 shl 21)) <> 0;
               HW_AVX512VBMI  := (reg.ECX and (1 shl  1)) <> 0;
          end;

          GetCPUID($80000000, reg);

          if reg.EAX >= $80000001 then
          begin
               GetCPUID($80000001, reg);

               HW_x64   := (reg.EDX and (1 shl 29)) <> 0;
               HW_ABM   := (reg.ECX and (1 shl  5)) <> 0;
               HW_SSE4a := (reg.ECX and (1 shl  6)) <> 0;
               HW_FMA4  := (reg.ECX and (1 shl 16)) <> 0;
               HW_XOP   := (reg.ECX and (1 shl 11)) <> 0;
          end;

          // now check the os support
          if (HW_AVX) or (HW_AVX2) then
             InitAVXOSSupportFlags;
     end;
end;

function IsSSE3Present : boolean;
begin
     Result := HW_SSE3;
end;

function IsAVXPresent : boolean;
begin
     Result := HW_AVX2 and AVX_OS_SUPPORT;
end;

function IsAVX512Present : boolean;
begin
     Result := HW_AVX512F and AVX512_OS_SUPPORT;
end;

function IsFMAPresent : boolean;
begin
     Result := AVX_OS_SUPPORT and HW_FMA3;
end;

function IsHardwareRNDSupport : boolean;
begin
     Result := HW_RDRAND;
end;

function IsHardwareRDSeed : boolean;
begin
     Result :=  HW_RDSEED;
end;

initialization
  InitFlags;

end.
