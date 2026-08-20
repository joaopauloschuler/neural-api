// This file is part of the Conscious Artificial Intelligence project
// located at:
// https://sourceforge.net/projects/cai/

// CAI Dot Product
// A vectors (A1, A2, A3, ...) are operated with a number of
// B vectors (B1, B2, B3, ...) via dot product. There is a resulting vector
// R with all dot products A1.B1 .. A1.B2 .. A2.B1 .. AN.BN .
// A vectors are sometimes interleaved.
__kernel void cai_dot_product
(
  const int FThreadCount,
  const int FNumAs,
  const int FNumBs,
  const int FSize,
  int ActFN,
  __global float* FInputBufferAs,
  __global float* FInputBufferBs,
  __global float* FResultBuffer,
  // Optional fused bias: when UseBias != 0, FBiasOutput[b_id*FNumAs + a_id] is
  // added to the reduced dot product BEFORE the activation, so an inference
  // forward computes act(W.x + b) entirely on the device (no host bias-add +
  // activation sweep). FBiasOutput carries the host FBiasOutput volume verbatim
  // (bias replicated per output position, same [pos][feature] layout as the
  // result), so the index matches the result write exactly. When UseBias == 0
  // the pointer is unread and may be NULL. Coded by Claude (AI).
  const int UseBias,
  __global const float* FBiasOutput
)
{
  const int a_id = get_global_id(0);
  const int b_id = get_global_id(1);

  if ( (a_id < FNumAs) && (b_id < FNumBs) )
  {
    const int VectBPos = b_id * FSize;

    float DotProductResult = 0;
    int i = 0;

    const int FSizeMinus8  = FSize -  8;
    const int FSizeMinus32 = FSize - 32;

    while (i < FSizeMinus32)
    {
      const int startBPos = i + VectBPos;

      DotProductResult =
        mad(FInputBufferAs[a_id + (i+ 0)*FNumAs], FInputBufferBs[startBPos +  0],
        mad(FInputBufferAs[a_id + (i+ 1)*FNumAs], FInputBufferBs[startBPos +  1],
        mad(FInputBufferAs[a_id + (i+ 2)*FNumAs], FInputBufferBs[startBPos +  2],
        mad(FInputBufferAs[a_id + (i+ 3)*FNumAs], FInputBufferBs[startBPos +  3],
        mad(FInputBufferAs[a_id + (i+ 4)*FNumAs], FInputBufferBs[startBPos +  4],
        mad(FInputBufferAs[a_id + (i+ 5)*FNumAs], FInputBufferBs[startBPos +  5],
        mad(FInputBufferAs[a_id + (i+ 6)*FNumAs], FInputBufferBs[startBPos +  6],
        mad(FInputBufferAs[a_id + (i+ 7)*FNumAs], FInputBufferBs[startBPos +  7],
        mad(FInputBufferAs[a_id + (i+ 8)*FNumAs], FInputBufferBs[startBPos +  8],
        mad(FInputBufferAs[a_id + (i+ 9)*FNumAs], FInputBufferBs[startBPos +  9],
        mad(FInputBufferAs[a_id + (i+10)*FNumAs], FInputBufferBs[startBPos + 10],
        mad(FInputBufferAs[a_id + (i+11)*FNumAs], FInputBufferBs[startBPos + 11],
        mad(FInputBufferAs[a_id + (i+12)*FNumAs], FInputBufferBs[startBPos + 12],
        mad(FInputBufferAs[a_id + (i+13)*FNumAs], FInputBufferBs[startBPos + 13],
        mad(FInputBufferAs[a_id + (i+14)*FNumAs], FInputBufferBs[startBPos + 14],
        mad(FInputBufferAs[a_id + (i+15)*FNumAs], FInputBufferBs[startBPos + 15],
        mad(FInputBufferAs[a_id + (i+16)*FNumAs], FInputBufferBs[startBPos + 16],
        mad(FInputBufferAs[a_id + (i+17)*FNumAs], FInputBufferBs[startBPos + 17],
        mad(FInputBufferAs[a_id + (i+18)*FNumAs], FInputBufferBs[startBPos + 18],
        mad(FInputBufferAs[a_id + (i+19)*FNumAs], FInputBufferBs[startBPos + 19],
        mad(FInputBufferAs[a_id + (i+20)*FNumAs], FInputBufferBs[startBPos + 20],
        mad(FInputBufferAs[a_id + (i+21)*FNumAs], FInputBufferBs[startBPos + 21],
        mad(FInputBufferAs[a_id + (i+22)*FNumAs], FInputBufferBs[startBPos + 22],
        mad(FInputBufferAs[a_id + (i+23)*FNumAs], FInputBufferBs[startBPos + 23],
        mad(FInputBufferAs[a_id + (i+24)*FNumAs], FInputBufferBs[startBPos + 24],
        mad(FInputBufferAs[a_id + (i+25)*FNumAs], FInputBufferBs[startBPos + 25],
        mad(FInputBufferAs[a_id + (i+26)*FNumAs], FInputBufferBs[startBPos + 26],
        mad(FInputBufferAs[a_id + (i+27)*FNumAs], FInputBufferBs[startBPos + 27],
        mad(FInputBufferAs[a_id + (i+28)*FNumAs], FInputBufferBs[startBPos + 28],
        mad(FInputBufferAs[a_id + (i+29)*FNumAs], FInputBufferBs[startBPos + 29],
        mad(FInputBufferAs[a_id + (i+30)*FNumAs], FInputBufferBs[startBPos + 30],
        mad(FInputBufferAs[a_id + (i+31)*FNumAs], FInputBufferBs[startBPos + 31],
        DotProductResult
        ))))))))
        ))))))))
        ))))))))
        ))))))));

      i += 32;
    }

    while (i < FSizeMinus8)
    {
      const int startBPos = i + VectBPos;

      DotProductResult =
        mad(FInputBufferAs[a_id + (i+0)*FNumAs], FInputBufferBs[startBPos + 0],
        mad(FInputBufferAs[a_id + (i+1)*FNumAs], FInputBufferBs[startBPos + 1],
        mad(FInputBufferAs[a_id + (i+2)*FNumAs], FInputBufferBs[startBPos + 2],
        mad(FInputBufferAs[a_id + (i+3)*FNumAs], FInputBufferBs[startBPos + 3],
        mad(FInputBufferAs[a_id + (i+4)*FNumAs], FInputBufferBs[startBPos + 4],
        mad(FInputBufferAs[a_id + (i+5)*FNumAs], FInputBufferBs[startBPos + 5],
        mad(FInputBufferAs[a_id + (i+6)*FNumAs], FInputBufferBs[startBPos + 6],
        mad(FInputBufferAs[a_id + (i+7)*FNumAs], FInputBufferBs[startBPos + 7],
        DotProductResult))))))));
      i += 8;
    }

    while (i < FSize)
    {
      DotProductResult =
        mad(FInputBufferAs[a_id + i*FNumAs], FInputBufferBs[i + VectBPos], DotProductResult);
        i += 1;
    }

    // Fused bias-add (see the FBiasOutput arg comment): act must see W.x + b.
    if (UseBias != 0) DotProductResult += FBiasOutput[b_id * FNumAs + a_id];

    // Optional fused activation, applied in-register to the reduced dot product
    // before it is written back. Opcodes match the csAct* constants (and the
    // cai_activation switch): 1 = ReLU, 2 = Sigmoid, 3 = HyperbolicTangent;
    // 0/other = pass-through. This lets an inference forward skip the host-side
    // bias-add + activation sweep over the whole output volume. The sigmoid/tanh
    // math mirrors cai_activation exactly so device and host agree to ~1e-6.
    // Coded by Claude (AI).
    if (ActFN == 1)
    {
      if (DotProductResult < 0.0f) { DotProductResult = 0.0f; }
    }
    else if (ActFN == 2) // Sigmoid: numerically-stable two-branch 1/(1+exp(-x))
    {
      if (DotProductResult > 0.0f)
        DotProductResult = 1.0f / (1.0f + exp(-DotProductResult));
      else
      {
        const float s = exp(DotProductResult);
        DotProductResult = s / (1.0f + s);
      }
    }
    else if (ActFN == 3) // HyperbolicTangent: clamp [-10,10], (1-e)/(1+e), e=exp(-2x)
    {
      float xc = DotProductResult;
      if (xc > 10.0f) xc = 10.0f; else if (xc < -10.0f) xc = -10.0f;
      const float e = exp(-2.0f * xc);
      DotProductResult = (1.0f - e) / (1.0f + e);
    }

    FResultBuffer[b_id * FNumAs + a_id] = DotProductResult;
  }
} // end of kernel

// Int8-weight twin of cai_dot_product: the A operand is per-output-row
// symmetric int8 codes (dequantized value = code * FScales[a_id]) stored in
// the SAME interleaved layout ([a_id + i*FNumAs], one byte per element), so
// adjacent work-items read adjacent bytes - 4 lanes per 32-bit transaction,
// 1/4 of the FP32 weight traffic. Codes convert to float in-register
// (convert_float on OpenCL's signed char); the per-row scale is applied ONCE
// to the reduced raw code sum, before the fused bias-add, mirroring the host
// fused kernel (TNNetVolume.DotProductInt8 + deferred scale) so device and
// host agree to normal float tolerance. B, the result layout, and the fused
// bias/activation tail (args 8/9, opcodes 1=ReLU 2=Sigmoid 3=Tanh) are
// identical to cai_dot_product; FScales rides as arg 10. Coded by Claude (AI).
__kernel void cai_dot_product_int8
(
  const int FThreadCount,
  const int FNumAs,
  const int FNumBs,
  const int FSize,
  int ActFN,
  __global const char* FInputBufferAs,
  __global float* FInputBufferBs,
  __global float* FResultBuffer,
  const int UseBias,
  __global const float* FBiasOutput,
  __global const float* FScales
)
{
  const int a_id = get_global_id(0);
  const int b_id = get_global_id(1);

  if ( (a_id < FNumAs) && (b_id < FNumBs) )
  {
    const int VectBPos = b_id * FSize;

    float DotProductResult = 0;
    int i = 0;

    const int FSizeMinus8  = FSize -  8;
    const int FSizeMinus32 = FSize - 32;

    while (i < FSizeMinus32)
    {
      const int startBPos = i + VectBPos;

      DotProductResult =
        mad(convert_float(FInputBufferAs[a_id + (i+ 0)*FNumAs]), FInputBufferBs[startBPos +  0],
        mad(convert_float(FInputBufferAs[a_id + (i+ 1)*FNumAs]), FInputBufferBs[startBPos +  1],
        mad(convert_float(FInputBufferAs[a_id + (i+ 2)*FNumAs]), FInputBufferBs[startBPos +  2],
        mad(convert_float(FInputBufferAs[a_id + (i+ 3)*FNumAs]), FInputBufferBs[startBPos +  3],
        mad(convert_float(FInputBufferAs[a_id + (i+ 4)*FNumAs]), FInputBufferBs[startBPos +  4],
        mad(convert_float(FInputBufferAs[a_id + (i+ 5)*FNumAs]), FInputBufferBs[startBPos +  5],
        mad(convert_float(FInputBufferAs[a_id + (i+ 6)*FNumAs]), FInputBufferBs[startBPos +  6],
        mad(convert_float(FInputBufferAs[a_id + (i+ 7)*FNumAs]), FInputBufferBs[startBPos +  7],
        mad(convert_float(FInputBufferAs[a_id + (i+ 8)*FNumAs]), FInputBufferBs[startBPos +  8],
        mad(convert_float(FInputBufferAs[a_id + (i+ 9)*FNumAs]), FInputBufferBs[startBPos +  9],
        mad(convert_float(FInputBufferAs[a_id + (i+10)*FNumAs]), FInputBufferBs[startBPos + 10],
        mad(convert_float(FInputBufferAs[a_id + (i+11)*FNumAs]), FInputBufferBs[startBPos + 11],
        mad(convert_float(FInputBufferAs[a_id + (i+12)*FNumAs]), FInputBufferBs[startBPos + 12],
        mad(convert_float(FInputBufferAs[a_id + (i+13)*FNumAs]), FInputBufferBs[startBPos + 13],
        mad(convert_float(FInputBufferAs[a_id + (i+14)*FNumAs]), FInputBufferBs[startBPos + 14],
        mad(convert_float(FInputBufferAs[a_id + (i+15)*FNumAs]), FInputBufferBs[startBPos + 15],
        mad(convert_float(FInputBufferAs[a_id + (i+16)*FNumAs]), FInputBufferBs[startBPos + 16],
        mad(convert_float(FInputBufferAs[a_id + (i+17)*FNumAs]), FInputBufferBs[startBPos + 17],
        mad(convert_float(FInputBufferAs[a_id + (i+18)*FNumAs]), FInputBufferBs[startBPos + 18],
        mad(convert_float(FInputBufferAs[a_id + (i+19)*FNumAs]), FInputBufferBs[startBPos + 19],
        mad(convert_float(FInputBufferAs[a_id + (i+20)*FNumAs]), FInputBufferBs[startBPos + 20],
        mad(convert_float(FInputBufferAs[a_id + (i+21)*FNumAs]), FInputBufferBs[startBPos + 21],
        mad(convert_float(FInputBufferAs[a_id + (i+22)*FNumAs]), FInputBufferBs[startBPos + 22],
        mad(convert_float(FInputBufferAs[a_id + (i+23)*FNumAs]), FInputBufferBs[startBPos + 23],
        mad(convert_float(FInputBufferAs[a_id + (i+24)*FNumAs]), FInputBufferBs[startBPos + 24],
        mad(convert_float(FInputBufferAs[a_id + (i+25)*FNumAs]), FInputBufferBs[startBPos + 25],
        mad(convert_float(FInputBufferAs[a_id + (i+26)*FNumAs]), FInputBufferBs[startBPos + 26],
        mad(convert_float(FInputBufferAs[a_id + (i+27)*FNumAs]), FInputBufferBs[startBPos + 27],
        mad(convert_float(FInputBufferAs[a_id + (i+28)*FNumAs]), FInputBufferBs[startBPos + 28],
        mad(convert_float(FInputBufferAs[a_id + (i+29)*FNumAs]), FInputBufferBs[startBPos + 29],
        mad(convert_float(FInputBufferAs[a_id + (i+30)*FNumAs]), FInputBufferBs[startBPos + 30],
        mad(convert_float(FInputBufferAs[a_id + (i+31)*FNumAs]), FInputBufferBs[startBPos + 31],
        DotProductResult
        ))))))))
        ))))))))
        ))))))))
        ))))))));

      i += 32;
    }

    while (i < FSizeMinus8)
    {
      const int startBPos = i + VectBPos;

      DotProductResult =
        mad(convert_float(FInputBufferAs[a_id + (i+0)*FNumAs]), FInputBufferBs[startBPos + 0],
        mad(convert_float(FInputBufferAs[a_id + (i+1)*FNumAs]), FInputBufferBs[startBPos + 1],
        mad(convert_float(FInputBufferAs[a_id + (i+2)*FNumAs]), FInputBufferBs[startBPos + 2],
        mad(convert_float(FInputBufferAs[a_id + (i+3)*FNumAs]), FInputBufferBs[startBPos + 3],
        mad(convert_float(FInputBufferAs[a_id + (i+4)*FNumAs]), FInputBufferBs[startBPos + 4],
        mad(convert_float(FInputBufferAs[a_id + (i+5)*FNumAs]), FInputBufferBs[startBPos + 5],
        mad(convert_float(FInputBufferAs[a_id + (i+6)*FNumAs]), FInputBufferBs[startBPos + 6],
        mad(convert_float(FInputBufferAs[a_id + (i+7)*FNumAs]), FInputBufferBs[startBPos + 7],
        DotProductResult))))))));
      i += 8;
    }

    while (i < FSize)
    {
      DotProductResult =
        mad(convert_float(FInputBufferAs[a_id + i*FNumAs]), FInputBufferBs[i + VectBPos], DotProductResult);
        i += 1;
    }

    // Deferred per-row dequantization scale: applied ONCE to the raw code sum,
    // BEFORE the (FP32, unscaled) bias - same order as the host fused path.
    DotProductResult *= FScales[a_id];

    // Fused bias-add (see cai_dot_product): act must see W.x + b.
    if (UseBias != 0) DotProductResult += FBiasOutput[b_id * FNumAs + a_id];

    // Fused activation, identical opcode set and math as cai_dot_product.
    if (ActFN == 1)
    {
      if (DotProductResult < 0.0f) { DotProductResult = 0.0f; }
    }
    else if (ActFN == 2) // Sigmoid: numerically-stable two-branch 1/(1+exp(-x))
    {
      if (DotProductResult > 0.0f)
        DotProductResult = 1.0f / (1.0f + exp(-DotProductResult));
      else
      {
        const float s = exp(DotProductResult);
        DotProductResult = s / (1.0f + s);
      }
    }
    else if (ActFN == 3) // HyperbolicTangent: clamp [-10,10], (1-e)/(1+e), e=exp(-2x)
    {
      float xc = DotProductResult;
      if (xc > 10.0f) xc = 10.0f; else if (xc < -10.0f) xc = -10.0f;
      const float e = exp(-2.0f * xc);
      DotProductResult = (1.0f - e) / (1.0f + e);
    }

    FResultBuffer[b_id * FNumAs + a_id] = DotProductResult;
  }
} // end of kernel

// Fused bias/activation tail shared by the split-K reduce kernel: the opcode
// set and the math are identical to cai_dot_product's inline tail.
static inline float cai_fused_act(float v, const int ActFN)
{
  if (ActFN == 1)
  {
    return (v < 0.0f) ? 0.0f : v;
  }
  else if (ActFN == 2) // Sigmoid: numerically-stable two-branch 1/(1+exp(-x))
  {
    if (v > 0.0f) return 1.0f / (1.0f + exp(-v));
    const float s = exp(v);
    return s / (1.0f + s);
  }
  else if (ActFN == 3) // HyperbolicTangent: clamp [-10,10], (1-e)/(1+e), e=exp(-2x)
  {
    float xc = v;
    if (xc > 10.0f) xc = 10.0f; else if (xc < -10.0f) xc = -10.0f;
    const float e = exp(-2.0f * xc);
    return (1.0f - e) / (1.0f + e);
  }
  return v;
}

// SPLIT-K PASS 1. cai_dot_product_int8 gives one work-item per (output row,
// sample), so a decode GEMV (FNumBs=1) launches only FNumAs work-items and
// leaves a large device mostly idle. This kernel adds a third grid axis over
// the reduction: work-item (a_id, b_id, s) sums the slab
// [s*KChunk, (s+1)*KChunk) of row a_id and writes its RAW code sum (no scale,
// no bias, no activation) to FPartialBuffer. The A operand keeps the
// codes[a + i*FNumAs] layout, so consecutive a_id lanes still read consecutive
// bytes; splitting across work-GROUPS rather than lanes is what preserves that.
// cai_dot_product_int8_splitk_reduce finishes the job. Coded by Claude (AI).
__kernel void cai_dot_product_int8_splitk
(
  const int FNumAs,
  const int FNumBs,
  const int FSize,
  const int KSplits,
  __global const char* FInputBufferAs,
  __global const float* FInputBufferBs,
  __global float* FPartialBuffer
)
{
  const int a_id = get_global_id(0);
  const int b_id = get_global_id(1);
  const int s    = get_global_id(2);

  if ( (a_id < FNumAs) && (b_id < FNumBs) && (s < KSplits) )
  {
    const int KChunk = (FSize + KSplits - 1) / KSplits;
    const int kStart = s * KChunk;
    int kEnd = kStart + KChunk;
    if (kEnd > FSize) kEnd = FSize;

    float PartialResult = 0;
    int i = kStart;

    if (i < kEnd)
    {
      const int VectBPos = b_id * FSize;
      const int kEndMinus8 = kEnd - 8;

      while (i < kEndMinus8)
      {
        const int startBPos = i + VectBPos;

        PartialResult =
          mad(convert_float(FInputBufferAs[a_id + (i+0)*FNumAs]), FInputBufferBs[startBPos + 0],
          mad(convert_float(FInputBufferAs[a_id + (i+1)*FNumAs]), FInputBufferBs[startBPos + 1],
          mad(convert_float(FInputBufferAs[a_id + (i+2)*FNumAs]), FInputBufferBs[startBPos + 2],
          mad(convert_float(FInputBufferAs[a_id + (i+3)*FNumAs]), FInputBufferBs[startBPos + 3],
          mad(convert_float(FInputBufferAs[a_id + (i+4)*FNumAs]), FInputBufferBs[startBPos + 4],
          mad(convert_float(FInputBufferAs[a_id + (i+5)*FNumAs]), FInputBufferBs[startBPos + 5],
          mad(convert_float(FInputBufferAs[a_id + (i+6)*FNumAs]), FInputBufferBs[startBPos + 6],
          mad(convert_float(FInputBufferAs[a_id + (i+7)*FNumAs]), FInputBufferBs[startBPos + 7],
          PartialResult))))))));
        i += 8;
      }

      while (i < kEnd)
      {
        PartialResult =
          mad(convert_float(FInputBufferAs[a_id + i*FNumAs]), FInputBufferBs[i + VectBPos], PartialResult);
        i += 1;
      }
    }

    // Slab-major layout: pass 1 writes and pass 2 reads with consecutive a_id
    // lanes hitting consecutive floats, so both passes stay coalesced.
    FPartialBuffer[s * FNumAs * FNumBs + b_id * FNumAs + a_id] = PartialResult;
  }
} // end of kernel

// SPLIT-K PASS 2. Sums the KSplits raw partials of one (a_id, b_id), then
// applies the deferred per-row scale, the fused bias and the fused activation
// in cai_dot_product_int8's order, and writes the final result. One work-item
// per output element. Coded by Claude (AI).
__kernel void cai_dot_product_int8_splitk_reduce
(
  const int FNumAs,
  const int FNumBs,
  const int KSplits,
  const int ActFN,
  __global const float* FPartialBuffer,
  __global float* FResultBuffer,
  const int UseBias,
  __global const float* FBiasOutput,
  __global const float* FScales
)
{
  const int a_id = get_global_id(0);
  const int b_id = get_global_id(1);

  if ( (a_id < FNumAs) && (b_id < FNumBs) )
  {
    const int RowStride = FNumAs * FNumBs;
    const int BasePos = b_id * FNumAs + a_id;

    float DotProductResult = 0;
    for (int s = 0; s < KSplits; s++)
    {
      DotProductResult += FPartialBuffer[s * RowStride + BasePos];
    }

    // Deferred per-row dequantization scale, then the (FP32, unscaled) bias -
    // same order as cai_dot_product_int8 and as the host fused path.
    DotProductResult *= FScales[a_id];
    if (UseBias != 0) DotProductResult += FBiasOutput[BasePos];

    FResultBuffer[BasePos] = cai_fused_act(DotProductResult, ActFN);
  }
} // end of kernel

__kernel void cai_dot_product2
(
  const int FThreadCount,
  const int FNumAs,
  const int FNumBs,
  const int FSize,
  int ActFN,
  __global float* FInputBufferAs,
  __global float* FInputBufferBs,
  __global float* FResultBuffer
)
{
  const int a_id = get_global_id(0);
  const int b_id = get_global_id(1);

  if ( (a_id < FNumAs) && (b_id < FNumBs) )
  {
    const int VectBPos = b_id * FSize;

    float DotProductResult = 0;
    int i = 0;

    const int FSizeMinus8  = FSize -  8;
    const int FSizeMinus32 = FSize - 32;

    const int a0 =  0*FNumAs;
    const int a1 =  1*FNumAs;
    const int a2 =  2*FNumAs;
    const int a3 =  3*FNumAs;
    const int a4 =  4*FNumAs;
    const int a5 =  5*FNumAs;
    const int a6 =  6*FNumAs;
    const int a7 =  7*FNumAs;
    const int a8 =  8*FNumAs;
    const int a9 =  9*FNumAs;
    const int a10 = 10*FNumAs;
    const int a11 = 11*FNumAs;
    const int a12 = 12*FNumAs;
    const int a13 = 13*FNumAs;
    const int a14 = 14*FNumAs;
    const int a15 = 15*FNumAs;
    const int a16 = 16*FNumAs;
    const int a17 = 17*FNumAs;
    const int a18 = 18*FNumAs;
    const int a19 = 19*FNumAs;
    const int a20 = 20*FNumAs;
    const int a21 = 21*FNumAs;
    const int a22 = 22*FNumAs;
    const int a23 = 23*FNumAs;
    const int a24 = 24*FNumAs;
    const int a25 = 25*FNumAs;
    const int a26 = 26*FNumAs;
    const int a27 = 27*FNumAs;
    const int a28 = 28*FNumAs;
    const int a29 = 29*FNumAs;
    const int a30 = 30*FNumAs;
    const int a31 = 31*FNumAs;

    while (i < FSizeMinus32)
    {
      const int startBPos = i + VectBPos;

      //a_id + (i+31)*FNumAs -> a_id + i*FNumAs + FNumAs * 31 -> ai + a31

      const int ai =  a_id + i*FNumAs;

      DotProductResult =
        mad(FInputBufferAs[ai +  a0], FInputBufferBs[startBPos +  0],
        mad(FInputBufferAs[ai +  a1], FInputBufferBs[startBPos +  1],
        mad(FInputBufferAs[ai +  a2], FInputBufferBs[startBPos +  2],
        mad(FInputBufferAs[ai +  a3], FInputBufferBs[startBPos +  3],
        mad(FInputBufferAs[ai +  a4], FInputBufferBs[startBPos +  4],
        mad(FInputBufferAs[ai +  a5], FInputBufferBs[startBPos +  5],
        mad(FInputBufferAs[ai +  a6], FInputBufferBs[startBPos +  6],
        mad(FInputBufferAs[ai +  a7], FInputBufferBs[startBPos +  7],
        mad(FInputBufferAs[ai +  a8], FInputBufferBs[startBPos +  8],
        mad(FInputBufferAs[ai +  a9], FInputBufferBs[startBPos +  9],
        mad(FInputBufferAs[ai + a10], FInputBufferBs[startBPos + 10],
        mad(FInputBufferAs[ai + a11], FInputBufferBs[startBPos + 11],
        mad(FInputBufferAs[ai + a12], FInputBufferBs[startBPos + 12],
        mad(FInputBufferAs[ai + a13], FInputBufferBs[startBPos + 13],
        mad(FInputBufferAs[ai + a14], FInputBufferBs[startBPos + 14],
        mad(FInputBufferAs[ai + a15], FInputBufferBs[startBPos + 15],
        mad(FInputBufferAs[ai + a16], FInputBufferBs[startBPos + 16],
        mad(FInputBufferAs[ai + a17], FInputBufferBs[startBPos + 17],
        mad(FInputBufferAs[ai + a18], FInputBufferBs[startBPos + 18],
        mad(FInputBufferAs[ai + a19], FInputBufferBs[startBPos + 19],
        mad(FInputBufferAs[ai + a20], FInputBufferBs[startBPos + 20],
        mad(FInputBufferAs[ai + a21], FInputBufferBs[startBPos + 21],
        mad(FInputBufferAs[ai + a22], FInputBufferBs[startBPos + 22],
        mad(FInputBufferAs[ai + a23], FInputBufferBs[startBPos + 23],
        mad(FInputBufferAs[ai + a24], FInputBufferBs[startBPos + 24],
        mad(FInputBufferAs[ai + a25], FInputBufferBs[startBPos + 25],
        mad(FInputBufferAs[ai + a26], FInputBufferBs[startBPos + 26],
        mad(FInputBufferAs[ai + a27], FInputBufferBs[startBPos + 27],
        mad(FInputBufferAs[ai + a28], FInputBufferBs[startBPos + 28],
        mad(FInputBufferAs[ai + a29], FInputBufferBs[startBPos + 29],
        mad(FInputBufferAs[ai + a30], FInputBufferBs[startBPos + 30],
        mad(FInputBufferAs[ai + a31], FInputBufferBs[startBPos + 31],
        DotProductResult
        ))))))))
        ))))))))
        ))))))))
        ))))))));
      i += 32;
    }

    while (i < FSizeMinus8)
    {
      const int startBPos = i + VectBPos;
      const int ai =  a_id + i*FNumAs;

      DotProductResult =
        mad(FInputBufferAs[ai +  a0], FInputBufferBs[startBPos +  0],
        mad(FInputBufferAs[ai +  a1], FInputBufferBs[startBPos +  1],
        mad(FInputBufferAs[ai +  a2], FInputBufferBs[startBPos +  2],
        mad(FInputBufferAs[ai +  a3], FInputBufferBs[startBPos +  3],
        mad(FInputBufferAs[ai +  a4], FInputBufferBs[startBPos +  4],
        mad(FInputBufferAs[ai +  a5], FInputBufferBs[startBPos +  5],
        mad(FInputBufferAs[ai +  a6], FInputBufferBs[startBPos +  6],
        mad(FInputBufferAs[ai +  a7], FInputBufferBs[startBPos +  7],
        DotProductResult))))))));
      i += 8;
    }

    while (i < FSize)
    {
      DotProductResult =
        mad(FInputBufferAs[a_id + i*FNumAs], FInputBufferBs[i + VectBPos], DotProductResult);
        i += 1;
    }

    FResultBuffer[b_id * FNumAs + a_id] = DotProductResult;
  }
} // end of kernel

#define TS 16 // The tile-size
__kernel void simpleGEMMT(
  const int FThreadCount,
  const int M, const int N, const int K,
  int ActFN,
  __global float* A,
  __global float* B,
  __global float* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // 0..M
    const int globalCol = TS*get_group_id(1) + col; // 0..N

    // Local memory to fit a tile of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation registers
    float acc;
    acc = 0.0f;

    // Loop over all tiles
    int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        int tiledIndex = TS*t;
        //int indexA = globalRow*K + tiledIndex + col; // not interleaved: a_id*K + i // a_id = globalRow; i = tileIndex + col
        int indexA = globalRow + (tiledIndex + col)*M; // interleaved: a_id + (i+0)*FNumAs
        int indexB = globalCol*K + tiledIndex + row;
        Asub[row][col] = A[indexA];
        Bsub[row][col] = B[indexB];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
          acc += Asub[row][k] * Bsub[k][col];
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[globalCol * M + globalRow] = acc;
}

__kernel void cai_dot_product_simple
(
  const int FThreadCount,
  const int FNumAs,
  const int FNumBs,
  const int FSize,
  int ActFN,
  __global   float16* FInputBufferAs,
  __global   float16* FInputBufferBs,
  __global   float* FResultBuffer
)
{
  const int a_id = get_global_id(0);
  const int b_id = get_global_id(1);
  const int SizeDiv16 = FSize / 16;

  if ( (a_id < FNumAs) && (b_id < FNumBs) )
  {
    const int VectAPos = a_id * SizeDiv16;
    const int VectBPos = b_id * SizeDiv16;

    float16 DotProductResult = 0.0f;
    int i = 0;

    while (i < SizeDiv16)
    {
      DotProductResult =
        mad(FInputBufferAs[VectAPos + i], FInputBufferBs[VectBPos + i], DotProductResult);
        i += 1;
    }

    float8 Final8 = DotProductResult.lo + DotProductResult.hi;
    float4 Final4 = Final8.lo + Final8.hi;
    float2 Final2 = Final4.lo + Final4.hi;
    float FinalResult   = Final2.lo + Final2.hi;

    if (ActFN == 1)
    {
      if (FinalResult < 0.0f) { FinalResult = 0.0f; }
    }

    FResultBuffer[b_id * FNumAs + a_id] = FinalResult;//b_id + a_id;
  }
} // end of kernel

// myGEMM5 adapted from https://cnugteren.github.io/tutorial/pages/page7.html
#define TSM 16                 // The tile-size in dimension M
#define TSN 16                 // The tile-size in dimension N
#define TSK 16                 // The tile-size in dimension K
#define WPTM 1                 // The work-per-thread in dimension N
#define WPTN 1                 // The work-per-thread in dimension N
#define RTSM (TSM/WPTN)        // The reduced tile-size in dimension M
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPT ((TSK*TSM)/(RTSM*RTSN)) // The loads-per-thread for a tile
__kernel void myGEMM5(
  const int FThreadCount,
  const int M, const int N, const int K,
  int ActFN,
  __global float* A,
  __global float* B,
  __global float* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TSM)
    const int col = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int globalRow = TSM*get_group_id(0) + row; // 0..M
    const int globalCol = TSN*get_group_id(1) + col; // 0..N

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK];

    // Initialise the accumulation registers
    float acc[WPTN];
    for (int w=0; w<WPTN; w++) {
        acc[w] = 0.0f;
    }

    // Loop over all tiles
    int numTiles = K/TSK;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int l=0; l<LPT; l++) {
            int tiledIndex = TSK*t + col + l*RTSN;
            int indexA = tiledIndex*M + TSM*get_group_id(0) + row;
            int indexB = tiledIndex*N + TSN*get_group_id(1) + row;
            Asub[col + l*RTSN][row] = A[indexA];
            Bsub[row][col + l*RTSN] = B[indexB];
       }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TSK; k++) {
            for (int w=0; w<WPTN; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w*RTSN][k];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w=0; w<WPTN; w++)
    {
        if (ActFN == 1)
        {
          if (acc[w] < 0.0f) { acc[w] = 0.0f; }
        }

        C[(globalCol + w*RTSN)*M + globalRow] = acc[w];
    }
}

#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B

// myGEMM6 adapted from https://cnugteren.github.io/tutorial/pages/page8.html
// Use 2D register blocking (further increase in work per thread)
__kernel void myGEMM6(
  const int FThreadCount,
  const int M, const int N, const int K,
  int ActFN,
  __global float* A,
  __global float* B,
  __global float* C) {
    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK+2];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    for (int wm=0; wm<WPTM; wm++) {
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    // Loop over all tiles
    int numTiles = K/TSK;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = id % TSM;
            int col = id / TSM;
            int tiledIndex = TSK*t + col;
            Asub[col][row] = A[tiledIndex*M + offsetM + row];
            Bsub[row][col] = B[tiledIndex*N + offsetN + row];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[col][k];
            }

            // Perform the computation
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            if (ActFN == 1)
            {
               if (acc[wm][wn] < 0.0f) { acc[wm][wn] = 0.0f; }
            }

            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}

// this function is under development - do not use it.
__kernel void volume_operations
(
  const int OpID,
  const int FSize,
  __global float* FAs,
  __global float* FBs,
  __global float* FCs,
  const float FA,
  const float FB,
  const float FC
)
{
  const int g_id = get_global_id(0);

  // MulAdd
  if (OpID == 1)
  {
     FAs[g_id] = FAs[g_id] + FBs[g_id] * FB;
  }

  // MulMulAdd
  if (OpID == 2)
  {
     FAs[g_id] = FAs[g_id] * FA + FBs[g_id] * FB;
  }
}

// CAI Bilinear Gather
// Embarrassingly-parallel depth-blend resampler shared by the bilinear-gather
// sampler layers (TNNetFlowWarp / TNNetBackwardWarp / TNNetAffineGridSample /
// TNNetBilinearUpsample). The 4 source corner indices and the 4 bilinear blend
// weights for every output pixel are precomputed ON THE CPU (exact floor /
// border-clamp / zero-pad logic, byte-identical to the scalar forward) and
// uploaded; this kernel only does the heavy memory-bound blend of the four
// Depth-long source columns into the output column.
//   FSrc      : source feature map, raw [(y*W + x)*Depth + d]    (Depth = FDepth)
//   FCorners  : 4 source linear pixel offsets per output pixel (stored as float,
//               exact for any realistic pixel count), [outpix*4 + corner], each =
//               (y*W + x) for an in-bounds corner or -1 for a masked (zero-pad /
//               out-of-range) corner.
//   FWeights  : 4 blend weights per output pixel, [outpix*4 + corner].
//   FDst      : output feature map, raw [outpix*Depth + d].
// One work-item per (outpix, d): global size = FNumOut * FDepth (dim 0).
__kernel void cai_bilinear_gather
(
  const int FNumOut,
  const int FDepth,
  __global const float* FCorners,
  __global const float* FWeights,
  __global const float* FSrc,
  __global float* FDst
)
{
  const int g_id = get_global_id(0);
  if (g_id >= FNumOut * FDepth) return;
  const int outpix = g_id / FDepth;
  const int d      = g_id - outpix * FDepth;
  const int cbase  = outpix * 4;
  float acc = 0.0f;
  for (int c = 0; c < 4; c++)
  {
    const int corner = (int)FCorners[cbase + c];
    if (corner >= 0)
      acc += FWeights[cbase + c] * FSrc[corner * FDepth + d];
  }
  FDst[g_id] = acc;
}

// CAI Pixel Shuffle (depth-to-space gather forward)
// Coded by Claude (AI).
// Device forward for TNNetPixelShuffle: a pure depth->space gather with NO
// arithmetic (each output element is a verbatim copy of one source element).
// The caller precomputes, per output element, the source LINEAR offset into the
// raw source feature map [(y*W + x)*Depth + d] (stored as float, exact for any
// realistic element count) and uploads it; this kernel only performs the copy.
//   FSrcIdx : source linear element offset per output element, [g_id].
//   FSrc    : source feature map, raw.
//   FDst    : output feature map, raw, [g_id].
// One work-item per OUTPUT element: global size = FNumOut (dim 0).
__kernel void cai_pixel_shuffle
(
  const int FNumOut,
  __global const float* FSrcIdx,
  __global const float* FSrc,
  __global float* FDst
)
{
  const int g_id = get_global_id(0);
  if (g_id >= FNumOut) return;
  FDst[g_id] = FSrc[(int)FSrcIdx[g_id]];
}

// CAI Bicubic Gather (separable 4x4 weighted gather forward)
// Coded by Claude (AI).
// Device forward for TNNetBicubicUpsample, the 16-corner sibling of
// cai_bilinear_gather. The caller precomputes, per output pixel, the 16 source
// linear pixel offsets (the 4x4 clamped neighbourhood) and the 16 separable
// cubic weights (wy[r]*wx[c]) ON THE CPU (byte-identical to the scalar forward)
// and uploads them; this kernel only does the memory-bound blend of the 16
// Depth-long source columns into the output column.
//   FCorners : 16 source linear pixel offsets per output pixel (stored as float,
//              exact), [outpix*16 + corner], each = (y*W + x) (all in-bounds:
//              bicubic clamps replicate the edge, so there is no masking).
//   FWeights : 16 blend weights per output pixel, [outpix*16 + corner].
//   FSrc     : source feature map, raw [(y*W + x)*Depth + d].
//   FDst     : output feature map, raw [outpix*Depth + d].
// One work-item per (outpix, d): global size = FNumOut * FDepth (dim 0).
__kernel void cai_bicubic_gather
(
  const int FNumOut,
  const int FDepth,
  __global const float* FCorners,
  __global const float* FWeights,
  __global const float* FSrc,
  __global float* FDst
)
{
  const int g_id = get_global_id(0);
  if (g_id >= FNumOut * FDepth) return;
  const int outpix = g_id / FDepth;
  const int d      = g_id - outpix * FDepth;
  const int cbase  = outpix * 16;
  float acc = 0.0f;
  for (int c = 0; c < 16; c++)
  {
    const int corner = (int)FCorners[cbase + c];
    acc = mad(FWeights[cbase + c], FSrc[corner * FDepth + d], acc);
  }
  FDst[g_id] = acc;
}

// CAI Pixel Shuffle Scatter (depth-to-space backward)
// Coded by Claude (AI).
// Device backward for TNNetPixelShuffle. The forward shuffle is a bijection
// (a pure permutation: each output element copies exactly one source element),
// so the backward gradient scatter is collision-free: every source element
// receives exactly one output gradient. We REUSE the SAME index buffer the
// forward built (output element -> source linear offset) and write in the OTHER
// direction. One work-item per OUTPUT element, no atomics needed.
//   FSrcIdx : source linear element offset per output element, [g_id] (same
//             buffer as the forward cai_pixel_shuffle).
//   FSrc    : output gradient, raw, [g_id].
//   FDst    : scattered source gradient, raw (zero-init by the host; this
//             permutation fully covers it with one write per element).
// One work-item per OUTPUT element: global size = FNumOut (dim 0).
__kernel void cai_pixel_shuffle_scatter
(
  const int FNumOut,
  __global const float* FSrcIdx,
  __global const float* FSrc,
  __global float* FDst
)
{
  const int g_id = get_global_id(0);
  if (g_id >= FNumOut) return;
  FDst[(int)FSrcIdx[g_id]] = FSrc[g_id];
}

// CAI Bicubic Scatter (separable 4x4 weighted backward)
// Coded by Claude (AI).
// Device backward for TNNetBicubicUpsample, the transpose of cai_bicubic_gather.
// The forward reads a 4x4 clamped source neighbourhood per output pixel; the
// backward scatters each output pixel's gradient into those same 16 corners
// with the same wy[r]*wx[c] weights. Because border clamping makes several
// output pixels write the SAME source pixel, a naive per-output scatter would
// race; to match the codebase's atomic-free gather style we instead run ONE
// work-item per (SOURCE pixel, depth) and gather every output contribution that
// lands on it from a CPU-built CSR contribution table (offsets + flat
// (outpix,weight) entries, byte-identical weights to the scalar backward).
//   FRowOff  : CSR row offsets per source pixel, [srcpix] and [srcpix+1],
//              stored as float (exact). Length FNumSrc+1.
//   FOutIdx  : flat output-pixel index per CSR entry, [entry] (stored float).
//   FWeights : flat blend weight per CSR entry, [entry].
//   FSrc     : output gradient, raw [outpix*FDepth + d].
//   FDst     : scattered source gradient, raw [srcpix*FDepth + d].
// One work-item per (srcpix, d): global size = FNumSrc * FDepth (dim 0).
__kernel void cai_bicubic_scatter
(
  const int FNumSrc,
  const int FDepth,
  __global const float* FRowOff,
  __global const float* FOutIdx,
  __global const float* FWeights,
  __global const float* FSrc,
  __global float* FDst
)
{
  const int g_id = get_global_id(0);
  if (g_id >= FNumSrc * FDepth) return;
  const int srcpix = g_id / FDepth;
  const int d      = g_id - srcpix * FDepth;
  const int e0     = (int)FRowOff[srcpix];
  const int e1     = (int)FRowOff[srcpix + 1];
  float acc = 0.0f;
  for (int e = e0; e < e1; e++)
  {
    const int outpix = (int)FOutIdx[e];
    acc = mad(FWeights[e], FSrc[outpix * FDepth + d], acc);
  }
  FDst[g_id] = acc;
}

// CAI Per-Token Norm (RMSNorm / LayerNorm forward)
// Coded by Claude (AI).
// Device forward for the per-TOKEN depth-axis normalization layers
// TNNetTokenRMSNorm and TNNetTokenLayerNorm. The input is a sequence of tokens
// laid out with the feature vector CONTIGUOUS on the Depth axis: token t occupies
// FX[t*FDepth .. t*FDepth + FDepth-1]. Each token is normalized INDEPENDENTLY
// over its FDepth elements, then a per-channel gain (and, for LayerNorm, a bias)
// is applied. This reproduces the exact scalar arithmetic of the CPU Compute():
//   FUseMean == 0 (RMSNorm, no mean subtraction):
//     ms      = mean(x^2)
//     invStd  = 1/sqrt(ms + FEps)
//     y[c]    = FGain[c] * (x[c] * invStd)
//   FUseMean == 1 (LayerNorm):
//     mean    = mean(x)
//     var     = mean((x-mean)^2)
//     invStd  = 1/sqrt(var + FEps)
//     y[c]    = FGain[c] * ((x[c]-mean) * invStd) + FBias[c]
//   FGain : per-channel gain weights, FDepth long.
//   FBias : per-channel bias weights, FDepth long (ignored when FUseMean == 0).
//   FX    : input tokens, raw [t*FDepth + c].
//   FY    : output tokens, raw [t*FDepth + c].
// One work-item per TOKEN: global size = FNumTokens (dim 0). Keeping the whole
// per-token reduction inside a single work-item keeps the depth-axis sum order
// close to the scalar path (parity well under 1e-4).
__kernel void cai_token_norm
(
  const int FNumTokens,
  const int FDepth,
  const int FUseMean,
  const float FEps,
  __global const float* FGain,
  __global const float* FBias,
  __global const float* FX,
  __global float* FY
)
{
  const int t = get_global_id(0);
  if (t >= FNumTokens) return;
  const int base = t * FDepth;
  float mean = 0.0f;
  if (FUseMean != 0)
  {
    float s = 0.0f;
    for (int c = 0; c < FDepth; c++) s += FX[base + c];
    mean = s / (float)FDepth;
  }
  // reduction: sum of squares (RMS) or sum of centered squares (LayerNorm var)
  float ss = 0.0f;
  for (int c = 0; c < FDepth; c++)
  {
    const float v = FX[base + c] - mean;
    ss = mad(v, v, ss);
  }
  const float invStd = 1.0f / sqrt(ss / (float)FDepth + FEps);
  for (int c = 0; c < FDepth; c++)
  {
    const float xhat = (FX[base + c] - mean) * invStd;
    if (FUseMean != 0)
      FY[base + c] = mad(FGain[c], xhat, FBias[c]);
    else
      FY[base + c] = FGain[c] * xhat;
  }
}

// Whole-volume normalization (TNNetRMSNorm with FUseMean=0 / TNNetLayerNorm with
// FUseMean=1). The WHOLE sample is a single reduction over FSize elements, so the
// per-token kernel above -- invoked with one token of width FSize -- would put
// the entire FSize-element reduction AND all FSize output writes on ONE
// work-item (a pathological serialization: ~100x slower than the parallel token
// path on a large volume). Instead this kernel runs ONE work-group of get_local_
// size(0) work-items that cooperatively reduce mean/variance through local
// memory, then apply the per-ELEMENT gain/bias in parallel via a grid-stride
// loop. Gain/Bias are FSize long (per element, NOT per channel), matching the
// scalar TNNetRMSNorm/TNNetLayerNorm which scale the flattened sample. Launch
// with global size == local size (a single work-group) and a power-of-two local
// size; pass FScratch as get_local_size(0) floats of __local memory. The CPU
// reference reduces with an 8-wide AVX accumulator, so this tree reduction --
// also order-independent in spirit -- stays within the <1e-4 parity bound.
__kernel void cai_volume_norm
(
  const int FSize,
  const int FUseMean,
  const float FEps,
  __global const float* FGain,
  __global const float* FBias,
  __global const float* FX,
  __global float* FY,
  __local float* FScratch
)
{
  const int lid = get_local_id(0);
  const int lsize = get_local_size(0);
  int s;

  // ---- mean (LayerNorm only); RMSNorm leaves mean = 0 ----
  float mean = 0.0f;
  if (FUseMean != 0)
  {
    float partial = 0.0f;
    for (int i = lid; i < FSize; i += lsize) partial += FX[i];
    FScratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (s = lsize >> 1; s > 0; s >>= 1)
    {
      if (lid < s) FScratch[lid] += FScratch[lid + s];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    mean = FScratch[0] / (float)FSize;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // ---- sum of (centered) squares ----
  float ss = 0.0f;
  for (int i = lid; i < FSize; i += lsize)
  {
    const float v = FX[i] - mean;
    ss = mad(v, v, ss);
  }
  FScratch[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (s = lsize >> 1; s > 0; s >>= 1)
  {
    if (lid < s) FScratch[lid] += FScratch[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float invStd = 1.0f / sqrt(FScratch[0] / (float)FSize + FEps);

  // ---- apply per-element gain/bias in parallel ----
  for (int i = lid; i < FSize; i += lsize)
  {
    const float xhat = (FX[i] - mean) * invStd;
    if (FUseMean != 0)
      FY[i] = mad(FGain[i], xhat, FBias[i]);
    else
      FY[i] = FGain[i] * xhat;
  }
}

// Per-depth-column L2 normalization forward (TNNetL2Normalize axis-0 / PixelNorm
// per-position mode, and TNNetPixelNorm). For each position p of FNumPositions
// (= SizeX*SizeY), the FDepth contiguous channels at base = p*FDepth are scaled
// to unit (or 1/sqrt(mean)) L2 norm over the depth axis -- NO mean subtraction,
// NO gain/bias:
//   ss     = sum_c x[c]^2
//   invN   = 1 / sqrt(ss * FInvScale + FEps)
//   y[c]   = x[c] * invN
// FInvScale selects the variant: 1.0 = plain L2 (TNNetL2Normalize, invN =
// rsqrt(sum + eps)); 1/FDepth = RMS-style (TNNetPixelNorm, invN =
// rsqrt(mean(x^2) + eps)). One work-item per POSITION: global size =
// FNumPositions (dim 0). Keeping the whole per-position reduction in one
// work-item matches the scalar/AVX depth-sum order (parity well under 1e-4).
// Forward-only; training stays on the CPU.
__kernel void cai_l2norm_perdepth
(
  const int FNumPositions,
  const int FDepth,
  const float FInvScale,
  const float FEps,
  __global const float* FX,
  __global float* FY
)
{
  const int p = get_global_id(0);
  if (p >= FNumPositions) return;
  const int base = p * FDepth;
  float ss = 0.0f;
  for (int c = 0; c < FDepth; c++)
  {
    const float v = FX[base + c];
    ss = mad(v, v, ss);
  }
  const float invN = 1.0f / sqrt(ss * FInvScale + FEps);
  for (int c = 0; c < FDepth; c++)
    FY[base + c] = FX[base + c] * invN;
}

// Shared GLU-family gated feed-forward activation. The input tensor is laid out
// as FNumTokens rows of (2*FHalfDepth) contiguous channels; each row splits into
// two contiguous depth-halves A = [0 .. FHalfDepth) and B = [FHalfDepth ..
// 2*FHalfDepth). The output is A * act(B) where act is selected by FActFlag:
//   0 = sigmoid     -> GLU       (A * sigmoid(B))
//   1 = swish       -> SwiGLU    (A * B * sigmoid(B))
//   2 = gelu-tanh   -> GEGLU     (A * B * 0.5*(1+tanh(sqrt(2/pi)*(B+0.044715*B^3))))
//   3 = gelu-erf    -> GEGLUErf  (A * B * 0.5*(1+erf(B/sqrt(2))))
// One work-item per (token, output-channel). Forward-only; the formulas are the
// exact analytic forms used by the scalar CPU Compute() so parity is < 1e-4.
// The sigmoid and tanh here are cai_activation's stable forms, not the direct
// ones: PoCL runs kernels on the host CPU, where the host process leaves
// floating-point exceptions unmasked, so an exp that overflows inside the
// library kills the process instead of saturating. Coded by Claude (AI).
__kernel void cai_glu_gate
(
  const int FNumTokens,
  const int FHalfDepth,
  const int FActFlag,
  __global const float* FX,
  __global float* FY
)
{
  const int gid = get_global_id(0);
  const int total = FNumTokens * FHalfDepth;
  if (gid >= total) return;
  const int t = gid / FHalfDepth;
  const int d = gid - t * FHalfDepth;
  const int inBase = t * (2 * FHalfDepth);
  const float a = FX[inBase + d];
  const float b = FX[inBase + FHalfDepth + d];
  float gated;
  if (FActFlag <= 1)            // GLU: sigmoid(B) / SwiGLU: swish(B) = B*sigmoid(B)
  {
    // Two-branch sigmoid: the negative side evaluates exp(b), which UNDERFLOWS
    // to zero instead of overflowing, so no clamp is needed on either side.
    float sig;
    if (b > 0.0f)
      sig = 1.0f / (1.0f + exp(-b));
    else
    {
      const float s = exp(b);
      sig = s / (1.0f + s);
    }
    if (FActFlag == 0) gated = sig; else gated = b * sig;
  }
  else if (FActFlag == 2)       // GEGLU: gelu_tanh(B)
  {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float GELU_CONST = 0.044715f;
    // The cubic term drives arg past 100 by |B| ~ 15, and tanh is already 1.0f
    // in single precision by |arg| ~ 9, so clamping to cai_activation's [-10,10]
    // changes no representable result and keeps exp(-2*arg) at exp(20).
    float arg = SQRT_2_OVER_PI * (b + GELU_CONST * b * b * b);
    if (arg > 10.0f) arg = 10.0f; else if (arg < -10.0f) arg = -10.0f;
    const float e = exp(-2.0f * arg);
    gated = b * 0.5f * (1.0f + (1.0f - e) / (1.0f + e));
  }
  else                          // GEGLUErf: gelu_erf(B)
  {
    const float INV_SQRT_2 = 0.7071067811865476f;
    gated = b * 0.5f * (1.0f + erf(b * INV_SQRT_2));
  }
  FY[gid] = a * gated;
}

// Group / Instance normalization forward (TNNetGroupNorm, and its Groups=Depth
// limit TNNetInstanceNorm). A single sample's volume is laid out depth-axis
// contiguous: element (x,y,c) lives at FX[(x*FSizeY + y)*FDepth + c]. The Depth
// channels are partitioned into FGroups contiguous groups of FChannelsPerGroup =
// FDepth/FGroups channels; group g owns channels [g*FChannelsPerGroup ..
// +FChannelsPerGroup). Each group is normalized to zero mean / unit variance
// over its (FSizeX * FSizeY * FChannelsPerGroup) elements, then an affine
// gamma/beta is applied. FAffineMode selects the gamma/beta layout:
//   0 = per-channel : FGain/FBias are FDepth long, indexed by absolute channel c
//   1 = per-element : FGain/FBias are FSizeX*FSizeY*FDepth long, indexed by the
//                     full element offset (legacy affine)
// One work-item per GROUP: global size = FGroups (dim 0). Keeping the whole
// per-group reduction in a single work-item matches the scalar accumulation
// order closely (parity well under 1e-4). Forward-only; training stays on CPU.
__kernel void cai_group_norm
(
  const int FSizeX,
  const int FSizeY,
  const int FDepth,
  const int FGroups,
  const int FChannelsPerGroup,
  const int FAffineMode,
  const float FEps,
  __global const float* FGain,
  __global const float* FBias,
  __global const float* FX,
  __global float* FY
)
{
  const int g = get_global_id(0);
  if (g >= FGroups) return;
  const int dStart = g * FChannelsPerGroup;
  const int groupSize = FSizeX * FSizeY * FChannelsPerGroup;
  const int rowStride = FSizeY * FDepth; // stride between successive x
  // Mean over the group.
  float s = 0.0f;
  for (int x = 0; x < FSizeX; x++)
    for (int y = 0; y < FSizeY; y++)
    {
      const int base = x * rowStride + y * FDepth + dStart;
      for (int c = 0; c < FChannelsPerGroup; c++)
        s += FX[base + c];
    }
  const float mean = s / (float)groupSize;
  // Variance = mean( (x-mean)^2 ) over the group.
  float ss = 0.0f;
  for (int x = 0; x < FSizeX; x++)
    for (int y = 0; y < FSizeY; y++)
    {
      const int base = x * rowStride + y * FDepth + dStart;
      for (int c = 0; c < FChannelsPerGroup; c++)
      {
        const float v = FX[base + c] - mean;
        ss = mad(v, v, ss);
      }
    }
  const float variance = ss / (float)groupSize;
  const float invStd = 1.0f / sqrt(variance + FEps);
  // Normalize then apply the learnable scale (gamma) and bias (beta).
  for (int x = 0; x < FSizeX; x++)
    for (int y = 0; y < FSizeY; y++)
    {
      const int base = x * rowStride + y * FDepth + dStart;
      for (int c = 0; c < FChannelsPerGroup; c++)
      {
        const int idx = base + c;
        const float xhat = (FX[idx] - mean) * invStd;
        int wIdx;
        if (FAffineMode == 0) wIdx = dStart + c; // per-channel
        else                  wIdx = idx;        // per-element
        FY[idx] = mad(FGain[wIdx], xhat, FBias[wIdx]);
      }
    }
}

// Windowed 2-D pooling forward (TNNetMaxPool / TNNetAvgPool). The input volume is
// laid out depth-axis contiguous in the TVolume convention: element (x,y,d) lives
// at FX[((FInW * y) + x) * FDepth + d], with row stride FInW*FDepth and x stride
// FDepth. One work-item per OUTPUT (x,y,d) cell: it reduces its pooling window
//   ix in [ox*FStride .. min(ox*FStride + FPoolSize - 1, FInW - 1)]
//   iy in [oy*FStride .. min(oy*FStride + FPoolSize - 1, FInH - 1)]
// (window edges are clipped to the input, exactly like the scalar loops). FReduce
// selects the reduction:
//   0 = MAX  : maximum over the actual (clipped) window cells. For TNNetMaxPool
//              the host passes the post-CopyPadding input when padding>0 / strided
//              so the zero-padded border cells are real window members, matching
//              the scalar FInputCopy path. FStride is the layer stride.
//   1 = AVG  : sum over the window divided by FDivisor (TNNetAvgPool divides by
//              the FULL FPoolSize*FPoolSize, NOT the clipped cell count; the host
//              passes FStride = FPoolSize and FDivisor = FPoolSize*FPoolSize).
// Global size = FOutW * FOutH * FDepth (dim 0). Forward-only; training stays on CPU.
__kernel void cai_pool2d
(
  const int FInW,
  const int FInH,
  const int FDepth,
  const int FOutW,
  const int FOutH,
  const int FPoolSize,
  const int FStride,
  const int FReduce,
  const float FDivisor,
  __global const float* FX,
  __global float* FY
)
{
  const int gid = get_global_id(0);
  const int total = FOutW * FOutH * FDepth;
  if (gid >= total) return;
  // Decode the output (ox, oy, d) from the flat work-item id (depth-contiguous).
  const int d  = gid % FDepth;
  const int t  = gid / FDepth;
  const int ox = t % FOutW;
  const int oy = t / FOutW;

  const int ix0 = ox * FStride;
  const int iy0 = oy * FStride;
  int ixMax = ix0 + FPoolSize - 1; if (ixMax > FInW - 1) ixMax = FInW - 1;
  int iyMax = iy0 + FPoolSize - 1; if (iyMax > FInH - 1) iyMax = FInH - 1;

  const int rowStride = FInW * FDepth;
  if (FReduce == 0)
  {
    float m = -1e30f;
    for (int iy = iy0; iy <= iyMax; iy++)
    {
      const int rowBase = iy * rowStride + d;
      for (int ix = ix0; ix <= ixMax; ix++)
      {
        const float v = FX[rowBase + ix * FDepth];
        if (v > m) m = v;
      }
    }
    FY[gid] = m;
  }
  else
  {
    float s = 0.0f;
    for (int iy = iy0; iy <= iyMax; iy++)
    {
      const int rowBase = iy * rowStride + d;
      for (int ix = ix0; ix <= ixMax; ix++)
        s += FX[rowBase + ix * FDepth];
    }
    FY[gid] = s / FDivisor;
  }
}

// Token-gather embedding forward (TNNetEmbedding). The weight table FW is laid out
// row-per-token, depth-contiguous (row t, embedding index e at FW[t*FEmbeddingSize + e]),
// exactly FNeurons[0].Weights in the TVolume convention. FTokenRows holds one resolved
// source row per output token: FTokenRows[c] = the vocab row to copy into output token c,
// or -1 to leave that output token zero (the scalar path's zero-padding case -- token 0
// when EncodeZero is false). One work-item per (output token, depth) pair copies a single
// scalar:
//   FY[c*FEmbeddingSize + e] = (FTokenRows[c] < 0) ? 0 : FW[FTokenRows[c]*FEmbeddingSize + e]
// Global size = FNumTokens * FEmbeddingSize (dim 0). Forward-only; training stays on CPU.
__kernel void cai_embedding_gather
(
  const int FNumTokens,
  const int FEmbeddingSize,
  __global const int* FTokenRows,
  __global const float* FW,
  __global float* FY
)
{
  const int gid = get_global_id(0);
  const int total = FNumTokens * FEmbeddingSize;
  if (gid >= total) return;
  const int e = gid % FEmbeddingSize;
  const int c = gid / FEmbeddingSize;
  const int row = FTokenRows[c];
  if (row < 0)
    FY[gid] = 0.0f;
  else
    FY[gid] = FW[row * FEmbeddingSize + e];
}

// Int8 twin of cai_embedding_gather (TNNetEmbedding under --int8). FCodes holds
// the same row-major table as FW above with one symmetric int8 code per element,
// and FScales one FP32 scale per vocab row: the dequantized value of element e of
// row t is FCodes[t*FEmbeddingSize + e] * FScales[t], exactly what the host
// TNNetVolumeQuant8.DequantizeRowTo computes. Argument order keeps the FP32
// prefix (FScales last, as cai_moe_expert_down_int8 does) so one host call site
// serves both entry points. Coded by Claude (AI).
__kernel void cai_embedding_gather_int8
(
  const int FNumTokens,
  const int FEmbeddingSize,
  __global const int* FTokenRows,
  __global const char* FCodes,
  __global float* FY,
  __global const float* FScales
)
{
  const int gid = get_global_id(0);
  const int total = FNumTokens * FEmbeddingSize;
  if (gid >= total) return;
  const int e = gid % FEmbeddingSize;
  const int c = gid / FEmbeddingSize;
  const int row = FTokenRows[c];
  if (row < 0)
    FY[gid] = 0.0f;
  else
    FY[gid] = convert_float(FCodes[row * FEmbeddingSize + e]) * FScales[row];
}

// Device-side im2col: builds the convolution's FInputPrepared column matrix
// straight into device memory, so only the small (padded) input crosses the bus
// instead of the ~FeatureSizeX*FeatureSizeY-times-larger column matrix, and the
// host im2col gather (PrepareInputForConvolutionFast) is skipped entirely. Pure
// gather - every output element FCols[gid] is a copy of one FInput element, index
// computed from the closed-form conv geometry. It is bit-faithful to the CPU
// PrepareInputForConvolutionFast (same TVolume layout ((SizeX*y)+x)*Depth+d), so
// the GEMM that follows reads an identical B operand. FInput is the ALREADY
// (zero-)padded input (FInputCopy), hence no bounds checks. Only used on the
// inference-only, non-pointwise, non-Winograd forward path. One work-item per
// column-matrix element. Coded by Claude (AI).
__kernel void cai_im2col
(
  const int N,          // total elements = OutSizeX*OutSizeY*ColDepth
  const int OutSizeX,   // FOutput.SizeX
  const int ColDepth,   // FInputPrepared depth = InDepth*FeatX*FeatY
  const int RowSpan,    // one feature-row width = InDepth*FeatX
  const int InSizeX,    // FInputCopy.SizeX (padded)
  const int InDepth,    // FInputCopy.Depth
  const int Stride,
  __global const float* FInput,
  __global float* FCols
)
{
  const int gid = get_global_id(0);
  if (gid >= N) return;
  const int col_elem = gid % ColDepth;
  const int pos      = gid / ColDepth;
  const int ox       = pos % OutSizeX;
  const int oy       = pos / OutSizeX;
  const int yCount   = col_elem / RowSpan;
  const int rem      = col_elem % RowSpan;
  const int src = ((InSizeX * (oy * Stride + yCount)) + ox * Stride) * InDepth + rem;
  FCols[gid] = FInput[src];
}

// Rotary positional embedding (RoPE) forward. The input is FSeqLen tokens of
// FDepth depth-contiguous channels [t*FDepth + c]; FDepth is even and the
// rotation operates on the interleaved channel-pairs (2k, 2k+1). FTheta holds
// the FHalfDepth (= FDepth/2) precomputed per-pair effective frequencies (with
// any NTK/YaRN/Llama3/PI/LongRoPE scaling ALREADY folded in on the host, so the
// device only does the plain rotation). The rotation angle for pair k of token t
// is (t + FPositionOffset) * FTheta[k]; FOutScale is the YaRN/LongRoPE output
// multiplier (1.0 on the default path). One work-item per (token, channel-pair).
// Bit-faithful to the scalar CPU Compute() so parity is < 1e-4.
__kernel void cai_rope
(
  const int FSeqLen,
  const int FDepth,
  const int FHalfDepth,
  const int FPositionOffset,
  const float FOutScale,
  __global const float* FTheta,
  __global const float* FX,
  __global float* FY
)
{
  const int gid = get_global_id(0);
  const int total = FSeqLen * FHalfDepth;
  if (gid >= total) return;
  const int k = gid % FHalfDepth;
  const int pos = gid / FHalfDepth;
  const float angle = (float)(pos + FPositionOffset) * FTheta[k];
  const float s = sin(angle);
  const float c = cos(angle);
  const int base = pos * FDepth + 2 * k;
  const float x0 = FX[base];
  const float x1 = FX[base + 1];
  FY[base]     = FOutScale * (c * x0 - s * x1);
  FY[base + 1] = FOutScale * (s * x0 + c * x1);
}

// Multimodal rotary forward (M-RoPE, TNNetMRotaryEmbedding). Same interleaved
// (2k, 2k+1) pair rotation as cai_rope, but the per-(token, pair) ANGLE is
// resolved on the HOST (which 3-D section position each pair uses, plus the
// FTheta frequency and any RoPE scaling) and uploaded verbatim as the
// FAngle[token*FHalfDepth + k] table. The device only applies the pure
// rotation; FOutScale is the YaRN/LongRoPE output multiplier (1.0 default).
// One work-item per (token, channel-pair). Bit-faithful to the scalar
// TNNetMRotaryEmbedding.Compute() so parity is < 1e-4.
// Coded by Claude (AI).
__kernel void cai_mrope
(
  const int FSeqLen,
  const int FDepth,
  const int FHalfDepth,
  const float FOutScale,
  __global const float* FAngle,
  __global const float* FX,
  __global float* FY
)
{
  const int gid = get_global_id(0);
  const int total = FSeqLen * FHalfDepth;
  if (gid >= total) return;
  const int k = gid % FHalfDepth;
  const int pos = gid / FHalfDepth;
  const float angle = FAngle[gid];
  const float s = sin(angle);
  const float c = cos(angle);
  const int base = pos * FDepth + 2 * k;
  const float x0 = FX[base];
  const float x1 = FX[base + 1];
  FY[base]     = FOutScale * (c * x0 - s * x1);
  FY[base + 1] = FOutScale * (s * x0 + c * x1);
}

// Numerically-stable softmax forward for the softmax head layers
// (TNNetPointwiseSoftMax, TNNetSoftMax). The volume is tiled into contiguous
// normalization groups of FGroupLen elements; group g owns FX[g*FGroupLen ..
// +FGroupLen). One work-item per group does the standard max-subtract -> exp ->
// sum -> divide:
//   m = max(x);  e_i = exp(clamp(x_i - m, 4000));  y_i = e_i / sum(e)
// FApplyMinScale selects the variant:
//   0 = TNNetPointwiseSoftMax (FOutput.PointwiseSoftMax, GroupLen = Depth): no
//       low-end rescaling.
//   1 = TNNetSoftMax (TVolume.SoftMax, GroupLen = FOutput.Size): mirrors the
//       scalar path which, after the max-subtract, multiplies the whole group by
//       (-1000 / minValue) when minValue < -1000 (and leaves the group UNCHANGED
//       -- TotalSum := 0 -- in the degenerate minValue == 0 all-equal case).
// The per-group reduction stays inside one work-item to match the scalar
// accumulation order (parity < 1e-4). Forward-only; training stays on the CPU.
__kernel void cai_softmax
(
  const int FNumGroups,
  const int FGroupLen,
  const int FApplyMinScale,
  __global const float* FX,
  __global float* FY
)
{
  const int g = get_global_id(0);
  if (g >= FNumGroups) return;
  const int base = g * FGroupLen;
  // Per-group max (for the stable shift) and min (for the whole-volume rescale).
  float maxv = FX[base];
  float minv = FX[base];
  for (int c = 1; c < FGroupLen; c++)
  {
    const float v = FX[base + c];
    if (v > maxv) maxv = v;
    if (v < minv) minv = v;
  }
  // Shift by the max (skipped when max == 0, matching the scalar Sub guard).
  const float shift = (maxv != 0.0f) ? maxv : 0.0f;
  // Whole-volume variant: after the shift, minValue := min - shift. When that
  // shifted minimum is < -1000 the scalar path rescales the whole group by
  // (-1000 / shiftedMin); when it is exactly 0 (all elements equal) the scalar
  // path returns without normalizing.
  float scale = 1.0f;
  if (FApplyMinScale != 0)
  {
    const float shiftedMin = minv - shift;
    if (shiftedMin == 0.0f)
    {
      // Degenerate all-equal group: scalar SoftMax leaves data unchanged.
      for (int c = 0; c < FGroupLen; c++) FY[base + c] = FX[base + c];
      return;
    }
    if (shiftedMin < -1000.0f) scale = -1000.0f / shiftedMin;
  }
  float total = 0.0f;
  for (int c = 0; c < FGroupLen; c++)
  {
    float a = (FX[base + c] - shift) * scale;
    if (a > 4000.0f) a = 4000.0f; else if (a < -4000.0f) a = -4000.0f;
    const float e = exp(a);
    FY[base + c] = e;
    total += e;
  }
  if (total > 0.0f)
  {
    const float inv = 1.0f / total;
    for (int c = 0; c < FGroupLen; c++) FY[base + c] *= inv;
  }
}

// CAI shared elementwise activation forward. One work-item per element applies
// the function selected by FOpcode (kept in sync with the csAct* constants in
// neuralnetwork.pas): 1 = ReLU, 2 = Sigmoid, 3 = HyperbolicTangent, 4 = Swish,
// 5 = GELU, 6 = GELUErf, 7 = HardSwish, 8 = HardSigmoid, 9 = ELU, 10 = SELU,
// 11..23 = the branch-and-arithmetic activations (Abs through BentIdentity),
// 24 = ReLUL.
// This single kernel backs every opting-in TNNetIdentity activation descendant,
// so new elementwise activations only add a case here plus an opcode. FParamA,
// FParamB and FParamC carry the per-layer constants of the parameterized
// activations (a slope, a lambda, a pair of limits and their leak); cases that
// take none ignore them.
// Forward-only: the host keeps the backward pass (and, for ReLU, the
// derivative gate mask). The sigmoid/tanh math mirrors the scalar CPU forms -
// the two-branch stable sigmoid and the [-10,10]-clamped tanh - so the device
// result tracks the host to ~1e-6 (exp here is more accurate than the CPU
// polynomial pcr_expf, not less). Coded by Claude (AI).
__kernel void cai_activation
(
  const int FSize,
  const int FOpcode,
  const float FParamA,
  const float FParamB,
  const float FParamC,
  __global const float* FX,
  __global float* FY
)
{
  const int i = get_global_id(0);
  if (i >= FSize) return;
  const float x = FX[i];
  float y;
  switch (FOpcode)
  {
    case 1: // ReLU: max(x, 0)
      y = (x > 0.0f) ? x : 0.0f;
      break;
    case 2: // Sigmoid: numerically-stable two-branch 1/(1+exp(-x))
      if (x > 0.0f)
        y = 1.0f / (1.0f + exp(-x));
      else
      {
        const float s = exp(x);
        y = s / (1.0f + s);
      }
      break;
    case 3: // HyperbolicTangent: clamp to [-10,10], (1-exp(-2x))/(1+exp(-2x))
    {
      float xc = x;
      if (xc > 10.0f) xc = 10.0f; else if (xc < -10.0f) xc = -10.0f;
      const float e = exp(-2.0f * xc);
      y = (1.0f - e) / (1.0f + e);
      break;
    }
    case 4: // Swish / SiLU: x * sigmoid(x), sigmoid in the same two-branch form
      if (x > 0.0f)
        y = x / (1.0f + exp(-x));
      else
      {
        const float s = exp(x);
        y = x * s / (1.0f + s);
      }
      break;
    case 5: // GELU (tanh approximation): x * 0.5 * (1 + tanh(arg))
    {
      const float SQRT_2_OVER_PI = 0.7978845608f;
      const float GELU_CONST = 0.044715f;
      // The cubic term drives arg past 100 by |x| ~ 15, and tanh is already 1.0f
      // in single precision by |arg| ~ 9, so the [-10,10] clamp changes no
      // representable result and keeps exp(-2*arg) at exp(20).
      float arg = SQRT_2_OVER_PI * (x + GELU_CONST * x * x * x);
      if (arg > 10.0f) arg = 10.0f; else if (arg < -10.0f) arg = -10.0f;
      const float e = exp(-2.0f * arg);
      y = x * 0.5f * (1.0f + (1.0f - e) / (1.0f + e));
      break;
    }
    case 6: // GELUErf (exact form): x * 0.5 * (1 + erf(x/sqrt(2)))
    {
      const float INV_SQRT_2 = 0.7071067811865476f;
      y = x * 0.5f * (1.0f + erf(x * INV_SQRT_2));
      break;
    }
    case 7: // HardSwish: x for x > 3, 0 for x < -3, else x*(x+3)/6
      if (x > 3.0f) y = x;
      else if (x < -3.0f) y = 0.0f;
      else y = x * (x + 3.0f) / 6.0f;
      break;
    case 8: // HardSigmoid: 1 for x > 3, 0 for x < -3, else (x+3)/6
      if (x > 3.0f) y = 1.0f;
      else if (x < -3.0f) y = 0.0f;
      else y = (x + 3.0f) / 6.0f;
      break;
    case 9: // ELU: x for x > 0, else alpha*(exp(x)-1). FParamA = alpha.
      // exp is evaluated only on the negative branch, where it underflows
      // towards zero rather than overflowing, so no clamp is needed.
      if (x > 0.0f) y = x; else y = FParamA * (exp(x) - 1.0f);
      break;
    case 10: // SELU: scale*x for x > 0, else scale*alpha*exp(x) - scale*alpha.
      // FParamA = scale*alpha and FParamB = scale, both passed in from the layer
      // so the device uses the very floats the host multiplied.
      if (x > 0.0f) y = FParamB * x; else y = FParamA * exp(x) - FParamA;
      break;
    case 11: // Abs
      y = fabs(x);
      break;
    case 12: // Sign: +1 above zero, -1 below, 0 at exactly zero
      if (x > 0.0f) y = 1.0f; else if (x < 0.0f) y = -1.0f; else y = 0.0f;
      break;
    case 13: // Square
      y = x * x;
      break;
    case 14: // SquaredReLU: x*x for x > 0, else 0
      y = (x > 0.0f) ? x * x : 0.0f;
      break;
    case 15: // LeakyReLU: x for x > 0, else slope*x. FParamA = slope.
      y = (x > 0.0f) ? x : FParamA * x;
      break;
    case 16: // ShiftedReLU: max(x, -1)
      y = (x > -1.0f) ? x : -1.0f;
      break;
    case 17: // HardTanh: clamp to [-1,1]
      if (x > 1.0f) y = 1.0f; else if (x < -1.0f) y = -1.0f; else y = x;
      break;
    case 18: // HardShrink: x outside [-lambda,lambda], else 0. FParamA = lambda.
      y = ((x > FParamA) || (x < -FParamA)) ? x : 0.0f;
      break;
    case 19: // SoftShrink: shrink towards zero by lambda. FParamA = lambda.
      if (x > FParamA) y = x - FParamA;
      else if (x < -FParamA) y = x + FParamA;
      else y = 0.0f;
      break;
    case 20: // Threshold: x above theta, else a fixed value.
      y = (x > FParamA) ? x : FParamB;   // FParamA = theta, FParamB = value
      break;
    case 21: // Clamp to [FParamA, FParamB]
      if (x <= FParamA) y = FParamA;
      else if (x >= FParamB) y = FParamB;
      else y = x;
      break;
    case 22: // SoftSign: x / (1 + |x|)
      y = x / (1.0f + fabs(x));
      break;
    case 23: // BentIdentity: (sqrt(x^2 + 1) - 1)/2 + x
      y = (sqrt(x * x + 1.0f) - 1.0f) * 0.5f + x;
      break;
    case 24: // ReLUL: leaky clamp into [FParamA, FParamB]. FParamC = slope.
      if (x > FParamB) y = FParamB + (x - FParamB) * FParamC;
      else if (x > FParamA) y = x;
      else y = FParamA + (x - FParamA) * FParamC;
      break;
    default: // csActNone / unknown: pass through
      y = x;
  }
  FY[i] = y;
}

// CAI multi-source elementwise sum (TNNetSum forward). One work-item per element
// adds FCount (1..4) same-sized sources into FDst; with FAccumulate the sources
// are added to what FDst already holds, so more than 4 sources finish in
// ceil(sources/4) launches. TNNetSum only dispatches this when EVERY source
// output is ALREADY resident on the device, so nothing is uploaded here and the
// result stays on the device until a host reader asks for it. Slots the launch
// does not use repeat the first source (never a NULL argument) and are kept out
// of the sum by FCount. Coded by Claude (AI).
__kernel void cai_volume_sum
(
  const int FSize,
  const int FCount,
  const int FAccumulate,
  __global const float* FA,
  __global const float* FB,
  __global const float* FC,
  __global const float* FD,
  __global float* FDst
)
{
  const int i = get_global_id(0);
  if (i >= FSize) return;
  float total = (FAccumulate != 0) ? FDst[i] : 0.0f;
  total += FA[i];
  if (FCount > 1) total += FB[i];
  if (FCount > 2) total += FC[i];
  if (FCount > 3) total += FD[i];
  FDst[i] = total;
}

// CAI two-source elementwise product, one work-item per element. FB is either
// the same length as FA (FBSize = FSize: TNNetCellMulByCell, a plain cellwise
// product) or one value per channel (FBSize = Depth: TNNetChannelMulByLayer,
// broadcast over the (X,Y) positions - a volume is depth-contiguous, so channel
// i % FBSize). The comparison below is uniform across the work-items, so the
// cellwise case never evaluates the modulo. Both layers only dispatch this when
// BOTH source outputs are ALREADY resident on the device, so nothing is uploaded
// here and the product stays on the device until a host reader asks for it.
// Coded by Claude (AI).
__kernel void cai_cell_mul
(
  const int FSize,
  const int FBSize,
  __global const float* FA,
  __global const float* FB,
  __global float* FDst
)
{
  const int i = get_global_id(0);
  if (i >= FSize) return;
  const int j = (FBSize == FSize) ? i : (i % FBSize);
  FDst[i] = FA[i] * FB[j];
}

// CAI channel gather (TNNetSplitChannels forward). Output element (pos, d) is
// source element (pos, FChannelIdx[d]), so one kernel covers both a contiguous
// channel run and the arbitrary channel list of TNNetSplitChannelEvery. A
// position is an (X,Y) site: both volumes share SizeX/SizeY, so pos indexes the
// same site in each and only the depth stride differs. TNNetSplitChannels only
// dispatches this when the source output is ALREADY resident on the device, so
// nothing is uploaded here and the slice stays on the device until a host reader
// asks for it. Launched 2-D: dim 0 = position, dim 1 = output channel.
// Coded by Claude (AI).
__kernel void cai_split_channels
(
  const int FPositionCount,
  const int FOutDepth,
  const int FInDepth,
  __global const int* FChannelIdx,
  __global const float* FSrc,
  __global float* FDst
)
{
  const int pos = get_global_id(0);
  const int d = get_global_id(1);
  if ((pos >= FPositionCount) || (d >= FOutDepth)) return;
  FDst[pos * FOutDepth + d] = FSrc[pos * FInDepth + FChannelIdx[d]];
}

// CAI depth-axis scatter (TNNetDeepConcat forward). Writes one source into the
// output channels [FDestChannel .. FDestChannel+get_global_size(1)-1], reading
// source channel (d % FInDepth). Two ways to launch it: dim 1 = FInDepth per
// source scatters that source's own contiguous block (the modulo is then an
// identity), and dim 1 = FOutDepth with FDestChannel = 0 tiles ONE source across
// the whole output depth - the broadcast a same-layer source list asks for, in a
// single launch instead of one per replica. TNNetDeepConcat only dispatches this
// when EVERY source output is ALREADY resident on the device, so nothing is
// uploaded here and the result stays on the device until a host reader asks for
// it. Dim 0 is the (X,Y) position: all sources share the output's SizeX/SizeY,
// so pos indexes the same site in each. Coded by Claude (AI).
__kernel void cai_deep_concat
(
  const int FPositionCount,
  const int FOutDepth,
  const int FInDepth,
  const int FDestChannel,
  __global const float* FSrc,
  __global float* FDst
)
{
  const int pos = get_global_id(0);
  const int d = get_global_id(1);
  if (pos >= FPositionCount) return;
  const int outChannel = FDestChannel + d;
  if (outChannel >= FOutDepth) return;
  FDst[pos * FOutDepth + outChannel] = FSrc[pos * FInDepth + (d % FInDepth)];
}

// CAI depth-axis concat fused over ALL sources (TNNetDeepConcat forward with 2,
// 3 or 4 sources). One launch writes every output channel exactly once: the
// source is chosen per channel from the FDepth* split points, which replaces
// cai_deep_concat's one-launch-per-source loop. The source depths sum to
// FOutDepth, so the last source's depth is derived and needs no argument.
// Grid is (FOutDepth, FPositionCount) -- dim 0 is the CHANNEL here, not the
// position as in cai_deep_concat. Dim 0 varies fastest across work-items and
// the channel axis is the contiguous one, so both the read and the write
// coalesce. The nested ?: chains compile to selects, so every work-item follows
// one instruction stream and no wavefront diverges at a split point. A source
// list whose entries are all the same layer needs no special case: the same
// buffer simply binds to every FSrc slot. TNNetDeepConcat only dispatches these
// when EVERY source output is ALREADY resident on the device, so nothing is
// uploaded here and the result stays on the device until a host reader asks for
// it. Coded by Claude (AI).
__kernel void cai_deep_concat2
(
  const int FPositionCount,
  const int FOutDepth,
  const int FDepth0,
  __global const float* FSrc0,
  __global const float* FSrc1,
  __global float* FDst
)
{
  const int d = get_global_id(0);
  const int pos = get_global_id(1);
  if ((d >= FOutDepth) || (pos >= FPositionCount)) return;
  const bool inFirst = (d < FDepth0);
  __global const float* src = inFirst ? FSrc0 : FSrc1;
  const int base = inFirst ? 0 : FDepth0;
  const int inDepth = inFirst ? FDepth0 : (FOutDepth - FDepth0);
  FDst[pos * FOutDepth + d] = src[pos * inDepth + (d - base)];
}

// Three-source form of cai_deep_concat2.
__kernel void cai_deep_concat3
(
  const int FPositionCount,
  const int FOutDepth,
  const int FDepth0,
  const int FDepth1,
  __global const float* FSrc0,
  __global const float* FSrc1,
  __global const float* FSrc2,
  __global float* FDst
)
{
  const int d = get_global_id(0);
  const int pos = get_global_id(1);
  if ((d >= FOutDepth) || (pos >= FPositionCount)) return;
  const int split1 = FDepth0 + FDepth1;
  const bool inFirst = (d < FDepth0);
  const bool inSecond = (d < split1);
  __global const float* src = inFirst ? FSrc0 : (inSecond ? FSrc1 : FSrc2);
  const int base = inFirst ? 0 : (inSecond ? FDepth0 : split1);
  const int inDepth =
    inFirst ? FDepth0 : (inSecond ? FDepth1 : (FOutDepth - split1));
  FDst[pos * FOutDepth + d] = src[pos * inDepth + (d - base)];
}

// Four-source form of cai_deep_concat2.
__kernel void cai_deep_concat4
(
  const int FPositionCount,
  const int FOutDepth,
  const int FDepth0,
  const int FDepth1,
  const int FDepth2,
  __global const float* FSrc0,
  __global const float* FSrc1,
  __global const float* FSrc2,
  __global const float* FSrc3,
  __global float* FDst
)
{
  const int d = get_global_id(0);
  const int pos = get_global_id(1);
  if ((d >= FOutDepth) || (pos >= FPositionCount)) return;
  const int split1 = FDepth0 + FDepth1;
  const int split2 = split1 + FDepth2;
  const bool inFirst = (d < FDepth0);
  const bool inSecond = (d < split1);
  const bool inThird = (d < split2);
  __global const float* src =
    inFirst ? FSrc0 : (inSecond ? FSrc1 : (inThird ? FSrc2 : FSrc3));
  const int base =
    inFirst ? 0 : (inSecond ? FDepth0 : (inThird ? split1 : split2));
  const int inDepth =
    inFirst ? FDepth0 :
      (inSecond ? FDepth1 : (inThird ? FDepth2 : (FOutDepth - split2)));
  FDst[pos * FOutDepth + d] = src[pos * inDepth + (d - base)];
}

// CAI Depthwise Convolution 2-D forward (TNNetDepthwiseConv).
// Coded by Claude (AI).
// A TRUE per-channel convolution: output channel (n*FInDepth + d) reduces ONLY
// over input channel d's FFx*FFy spatial taps (depthwise -- NO cross-channel
// mixing). This replaces the earlier dense-GEMV mapping that computed the full
// (Mult*InDepth) x (NumPos*InDepth) product and read back only the depth-diagonal
// -- an InDepth-fold compute overspend that made the device path far slower than
// the CPU. Here each work-item does exactly its own FFx*FFy MACs, zero waste.
//   Raw[ox,oy, n*InDepth + d] =
//       sum_{cy,cx} FW[n][cy,cx,d] * FX[ox*S+cx, oy*S+cy, d]
// FX is the host-padded input copy (SizeX = FInW, depth = FInDepth), so no bounds
// checks are needed (output extents were sized to fit). FW is the Mult neurons'
// weight volumes concatenated, each in its native depth-contiguous raw layout
// (tap (cx,cy), depth d at [(FFx*cy + cx)*FInDepth + d]); neuron n starts at
// n*FFx*FFy*FInDepth. The result is written PRE-activation into FY (the raw
// output); the host applies the activation function afterwards (the depthwise
// conv adds no bias), exactly matching the scalar path. One work-item per output
// element: global size = FOutW * FOutH * (FMult * FInDepth) (dim 0).
__kernel void cai_depthwise_conv2d
(
  const int FOutW,
  const int FOutH,
  const int FInW,
  const int FInDepth,
  const int FMult,
  const int FFx,
  const int FFy,
  const int FStride,
  __global const float* FW,
  __global const float* FX,
  __global float* FY
)
{
  const int gid = get_global_id(0);
  const int FOutDepth = FMult * FInDepth;
  const int total = FOutW * FOutH * FOutDepth;
  if (gid >= total) return;
  const int outd = gid % FOutDepth;
  const int t    = gid / FOutDepth;
  const int ox   = t % FOutW;
  const int oy   = t / FOutW;
  const int n = outd / FInDepth;
  const int d = outd - n * FInDepth;
  const int ix0 = ox * FStride;
  const int iy0 = oy * FStride;
  const int wBase = n * (FFx * FFy) * FInDepth + d; // + tap*FInDepth below
  float acc = 0.0f;
  for (int cy = 0; cy < FFy; cy++)
  {
    const int iy = iy0 + cy;
    const int xRow = ((FInW * iy) + ix0) * FInDepth + d;
    const int wRow = wBase + (FFx * cy) * FInDepth;
    for (int cx = 0; cx < FFx; cx++)
      acc = mad(FW[wRow + cx * FInDepth], FX[xRow + cx * FInDepth], acc);
  }
  FY[gid] = acc;
}

// CAI Depthwise Convolution 1-D forward (TNNetDepthwiseConv1D).
// Coded by Claude (AI).
// The 1-D sibling of cai_depthwise_conv2d: a per-channel causal/SAME temporal
// convolution along the sequence (SizeX = time, depth = channel; SizeY = 1).
// Output (t, c) reduces ONLY over channel c's own length-FKsize kernel -- again
// replacing the dense-GEMV-and-discard mapping with one work-item doing exactly
// its FKsize MACs.
//   out[t,c] = (FSuppressBias ? 0 : FBias[c])
//              + sum_kk FW[c*FKsize + kk] * x[t - FOff + kk, c]   (OOB tap -> 0)
// FOff = FKsize-1 for causal, FKsize/2 for centred SAME (resolved on the host).
// FX is the previous layer's output, raw [t*FChannels + c]; OOB taps are skipped
// here (zero-pad) so -- unlike the 2-D path -- NO host pre-padding is needed.
// FW is the C neurons' length-K kernels concatenated [c*FKsize + kk]; FBias holds
// the C per-channel biases. Linear, no activation, matching the scalar path. One
// work-item per output element: global size = FSeqLen * FChannels (dim 0).
__kernel void cai_depthwise_conv1d
(
  const int FSeqLen,
  const int FChannels,
  const int FKsize,
  const int FOff,
  const int FSuppressBias,
  __global const float* FW,
  __global const float* FBias,
  __global const float* FX,
  __global float* FY
)
{
  const int gid = get_global_id(0);
  const int total = FSeqLen * FChannels;
  if (gid >= total) return;
  const int c = gid % FChannels;
  const int t = gid / FChannels;
  float acc = (FSuppressBias == 0) ? FBias[c] : 0.0f;
  const int wBase = c * FKsize;
  for (int kk = 0; kk < FKsize; kk++)
  {
    const int srcT = t - FOff + kk;
    if (srcT < 0 || srcT >= FSeqLen) continue; // zero pad
    acc = mad(FW[wBase + kk], FX[srcT * FChannels + c], acc);
  }
  FY[gid] = acc;
}

// CAI DEPTHWISE 1-D CONVOLUTION, INCREMENTAL DECODE (TNNetDepthwiseConv1D
// inside a decode session). The same causal per-channel sweep as
// cai_depthwise_conv1d, except that the K-1 rows preceding the window are read
// from FHistIn instead of being zero-padded, and this kernel ALSO writes the
// K-1 rows that follow the window into FHistOut -- so the history never travels
// back to the host between tokens.
// Concatenated rows: 0..K-2 are FHistIn, K-1..K-2+FSeqLen are FX. Then
//   out[t,c] = (FSuppressBias ? 0 : FBias[c])
//              + sum_kk FW[c*FKsize + kk] * row(t + kk)[c]
// which is the causal read [t-(K-1) .. t] with FHistIn filling srcT < 0 --
// matching TNNetDepthwiseConv1D.ComputeDecodeCPURange tap for tap.
// FHistOut MUST be a different buffer from FHistIn (the host ping-pongs the
// two): work-items write it while others are still reading FHistIn.
// One work-item per (row, channel) over max(FSeqLen, K-1) rows; a work-item
// past both row counts writes nothing, so no explicit total guard is needed.
__kernel void cai_depthwise_conv1d_decode
(
  const int FSeqLen,
  const int FChannels,
  const int FKsize,
  const int FSuppressBias,
  __global const float* FW,
  __global const float* FBias,
  __global const float* FHistIn,
  __global const float* FX,
  __global float* FY,
  __global float* FHistOut
)
{
  const int gid = get_global_id(0);
  const int c = gid % FChannels;
  const int t = gid / FChannels;
  const int HistLen = FKsize - 1;
  if (t < FSeqLen)
  {
    float acc = (FSuppressBias == 0) ? FBias[c] : 0.0f;
    const int wBase = c * FKsize;
    for (int kk = 0; kk < FKsize; kk++)
    {
      const int srcT = t - HistLen + kk;
      const float xv = (srcT >= 0) ? FX[srcT * FChannels + c]
                                   : FHistIn[(HistLen + srcT) * FChannels + c];
      acc = mad(FW[wBase + kk], xv, acc);
    }
    FY[t * FChannels + c] = acc;
  }
  if (t < HistLen)
  {
    // New history row t is concatenated row FSeqLen + t.
    const int r = FSeqLen + t;
    FHistOut[gid] = (r < HistLen) ? FHistIn[r * FChannels + c]
                                  : FX[(r - HistLen) * FChannels + c];
  }
}

// GATED DELTA-RULE RECURRENCE (TNNetGatedDeltaNet), the token mixer of the
// Qwen3.5 / Qwen3-Next "linear_attention" blocks. ONE WORK-GROUP PER VALUE
// HEAD: the left-to-right scan is sequential in t but strictly per head, so a
// head runs its WHOLE sequence inside one work-group and no cross-work-group
// synchronization is ever needed. Launch 2-D with global (LocalSize,
// FNumVHeads) and local (LocalSize, 1); LocalSize must be a power of two (the
// tree reductions halve it). FScratch is LocalSize + 2*FHeadDimK + FHeadDimV
// floats of __local memory.
//
// Input row t is [ q (Hk*Dk) | k (Hk*Dk) | v (Hv*Dv) | z (Hv*Dv) | b (Hv) |
// a (Hv) ]; the six channel offsets arrive as arguments so this kernel never
// restates the layout. Per t, value head h (key head h / FRep):
//   qn = q * rsqrt(sum(q^2) + eps) * FScale;  kn = k * rsqrt(sum(k^2) + eps)
//   beta  = sigmoid(b);  decay = exp(-exp(min(A_log,30)) * softplus(a+dt_bias))
//   err   = v - decay * (S^T kn)
//   S     = decay * S + kn (x) beta*err
//   o     = S^T qn
//   out   = o * rsqrt(mean(o^2) + eps) * w * silu(z)
// err is folded with the decay so the state is decayed and rewritten in ONE
// pass, and the read-out accumulates from the value just written, so a token
// touches each state element exactly twice: once to read, once to rewrite.
// Lane e owns column e of the (Dk,Dv) state for the whole token, which keeps
// err and o private and makes adjacent lanes read adjacent floats.
//
// FResetState = 1 starts the scan from S = 0 (the full-sequence forward);
// FResetState = 0 carries the resident state bank in, which is what lets an
// incremental decode session step without touching RAM. Forward-only: no per-t
// cache is written, so Backpropagate has nothing to read after this kernel.
// Coded by Claude (AI).
__kernel void cai_gated_delta_net
(
  const int FSeqLen,
  const int FNumVHeads,
  const int FHeadDimK,
  const int FHeadDimV,
  const int FRep,
  const int FInDepth,
  const int FQOff,
  const int FKOff,
  const int FVOff,
  const int FZOff,
  const int FBOff,
  const int FAOff,
  const int FResetState,
  const float FEps,
  const float FScale,
  __global const float* FALog,
  __global const float* FDtBias,
  __global const float* FNormW,
  __global const float* FX,
  __global float* FS,
  __global float* FY,
  __local float* FScratch
)
{
  const int h = get_group_id(1);
  const int lid = get_local_id(0);
  const int lsize = get_local_size(0);
  const int Dk = FHeadDimK;
  const int Dv = FHeadDimV;
  const int kh = h / FRep;
  int s, d, e, i, t;

  __local float* kn = FScratch + lsize;
  __local float* qn = kn + Dk;
  __local float* oloc = qn + Dk;

  __global float* S = FS + (h * Dk) * Dv;
  // ea is invariant across t.
  const float ea = exp(fmin(FALog[h], 30.0f));
  const int qBase = FQOff + kh * Dk;
  const int kBase = FKOff + kh * Dk;
  const int vBase = FVOff + h * Dv;
  const int zBase = FZOff + h * Dv;
  const int yHead = h * Dv;
  const int yStride = FNumVHeads * Dv;

  for (t = 0; t < FSeqLen; t++)
  {
    const int xRow = t * FInDepth;
    float partial;

    // ---- q/k per-head L2 norm; eps INSIDE the squared sum, as HF does it ----
    partial = 0.0f;
    for (i = lid; i < Dk; i += lsize)
    {
      const float qv = FX[xRow + qBase + i];
      partial = mad(qv, qv, partial);
    }
    FScratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (s = lsize >> 1; s > 0; s >>= 1)
    {
      if (lid < s) FScratch[lid] += FScratch[lid + s];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float qinv = 1.0f / sqrt(FScratch[0] + FEps);
    barrier(CLK_LOCAL_MEM_FENCE);

    partial = 0.0f;
    for (i = lid; i < Dk; i += lsize)
    {
      const float kv = FX[xRow + kBase + i];
      partial = mad(kv, kv, partial);
    }
    FScratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (s = lsize >> 1; s > 0; s >>= 1)
    {
      if (lid < s) FScratch[lid] += FScratch[lid + s];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float kinv = 1.0f / sqrt(FScratch[0] + FEps);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (i = lid; i < Dk; i += lsize)
    {
      qn[i] = FX[xRow + qBase + i] * qinv * FScale;
      kn[i] = FX[xRow + kBase + i] * kinv;
    }

    // ---- per-head scalar gates: every lane computes them, so no reduction ----
    const float bv = FX[xRow + FBOff + h];
    float beta;
    if (bv > 0.0f) beta = 1.0f / (1.0f + exp(-bv));
    else { const float sb = exp(bv); beta = sb / (1.0f + sb); }
    const float pre = FX[xRow + FAOff + h] + FDtBias[h];
    float sp;
    if (pre > 30.0f) sp = pre;
    else if (pre < -30.0f) sp = exp(pre);
    else sp = log(1.0f + exp(pre));
    const float decay = exp(-ea * sp);
    // S_{-1} = 0 only at the very first step of a scan that starts from zero.
    const int hasPrev = (t > 0) || (FResetState == 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- the delta rule, one state column per lane ----
    for (e = lid; e < Dv; e += lsize)
    {
      float acc = 0.0f;
      if (hasPrev)
        for (d = 0; d < Dk; d++) acc = mad(S[d * Dv + e], kn[d], acc);
      const float bk = beta * (FX[xRow + vBase + e] - decay * acc);
      float o = 0.0f;
      for (d = 0; d < Dk; d++)
      {
        const int idx = d * Dv + e;
        const float sn = hasPrev ? mad(decay, S[idx], kn[d] * bk) : (kn[d] * bk);
        S[idx] = sn;
        o = mad(qn[d], sn, o);
      }
      oloc[e] = o;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- gated RMSNorm read-out over the head's Dv columns ----
    partial = 0.0f;
    for (e = lid; e < Dv; e += lsize) partial = mad(oloc[e], oloc[e], partial);
    FScratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (s = lsize >> 1; s > 0; s >>= 1)
    {
      if (lid < s) FScratch[lid] += FScratch[lid + s];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float rinv = 1.0f / sqrt(FScratch[0] / (float)Dv + FEps);
    const int yRow = t * yStride + yHead;
    for (e = lid; e < Dv; e += lsize)
    {
      const float zv = FX[xRow + zBase + e];
      FY[yRow + e] = oloc[e] * rinv * FNormW[e] * (zv / (1.0f + exp(-zv)));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// KV-CACHE APPEND FOR THE FUSED MULTI-HEAD ATTENTION DECODE (TNNetFusedSDPA).
// Writes the current token's K and V rows, one per KV head, into cache slot
// FCacheSlot. ONE WORK-GROUP PER KV HEAD; the lanes split the head dimension.
//
// The cache is HEAD-MAJOR: head g's rows are the contiguous block starting at
// g*FCacheMax*FDk, so slot (g*FCacheMax + position) addresses one row and the
// decode kernel below reads each head's key stream contiguously. This is the
// layout TNNetFusedSDPA.AppendRow already writes on the host side.
//
// The token row is [ Q (FQW) | K (FKW) | V (FKW) ], so head g's key slice
// starts at FQW + g*FDk and its value slice at FQW + FKW + g*FDk.
// This kernel and cai_sdpa_decode share one command queue and are enqueued in
// that order, so the in-order queue is what makes the appended row visible -
// there is no cross-work-group synchronization and none is needed.
// Coded by Claude (AI).
__kernel void cai_sdpa_append_kv
(
  const int FKVHeads,
  const int FDk,
  const int FCacheMax,
  const int FCacheSlot,
  const int FQW,
  const int FKW,
  __global const float* FX,
  __global float* FKCache,
  __global float* FVCache
)
{
  const int g = get_group_id(1);
  const int lid = get_local_id(0);
  const int lsize = get_local_size(0);
  if (g >= FKVHeads) return;
  const int dst = (g * FCacheMax + FCacheSlot) * FDk;
  const int kSrc = FQW + g * FDk;
  const int vSrc = FQW + FKW + g * FDk;
  for (int d = lid; d < FDk; d += lsize)
  {
    FKCache[dst + d] = FX[kSrc + d];
    FVCache[dst + d] = FX[vSrc + d];
  }
}

// CACHED-DECODE SCALED DOT-PRODUCT ATTENTION (TNNetFusedSDPA), one token over
// the resident KV cache. ONE WORK-GROUP PER QUERY HEAD: the head's score band
// is private to its work-group, so every synchronization this kernel needs is
// an intra-work-group barrier and no cross-work-group ordering is ever
// required. Launch 2-D with global (LocalSize, FQHeads) and local (LocalSize,
// 1); LocalSize must be a power of two (the tree reductions halve it).
// FScratch is LocalSize + FDk floats of __local memory.
//
// Query head h reads KV head h/FGroupSize (grouped-query attention) and runs
// three phases over the live cache [jStart..FCacheLen-1]:
//   1. lanes split the key axis: score j = dot(q, K[j]) * FInvSqrtDk, the
//      Gemma-2 soft-cap when FScoreSoftCap > 0, then a tree max;
//   2. the same partition exponentiates in place and tree-sums the normalizer;
//   3. lanes split the head dimension: out[d] = sum_j P[j] * V[j][d], each lane
//      accumulating over the whole key range and dividing once at the end.
// The score band lives in global memory (FScores, FQHeads*FCacheMax floats)
// rather than __local because a long context does not fit in a work-group's
// local memory; it is written and read only by the one work-group that owns
// it, so the barrier between phases 2 and 3 carries a global memory fence.
//
// FWindow > 0 is the sliding-window mask: jStart = FCacheLen - FWindow. The
// causal mask needs no code at all - the cache holds only committed tokens, so
// every live row is attendable. Forward-only, and the caller restricts it to a
// single-token step: prefill, eviction, segment masking and the int8 cache all
// stay on the host path. Coded by Claude (AI).
__kernel void cai_sdpa_decode
(
  const int FQHeads,
  const int FGroupSize,
  const int FDk,
  const int FCacheMax,
  const int FCacheLen,
  const int FWindow,
  const float FInvSqrtDk,
  const float FScoreSoftCap,
  const float FInvScoreSoftCap,
  __global const float* FX,
  __global const float* FKCache,
  __global const float* FVCache,
  __global float* FScores,
  __global float* FY,
  __local float* FScratch
)
{
  const int h = get_group_id(1);
  const int lid = get_local_id(0);
  const int lsize = get_local_size(0);
  int s, d, j;
  if (h >= FQHeads) return;

  __local float* qloc = FScratch + lsize;

  const int g = h / FGroupSize;
  const int qBase = h * FDk;
  const int plane = g * FCacheMax * FDk;
  const int scoreBase = h * FCacheMax;
  const int jStart = ((FWindow > 0) && (FCacheLen > FWindow))
                     ? (FCacheLen - FWindow) : 0;

  for (d = lid; d < FDk; d += lsize) qloc[d] = FX[qBase + d];
  barrier(CLK_LOCAL_MEM_FENCE);

  // ---- phase 1: scores over the live cache, then the row max ----
  float m = -1e30f;
  for (j = jStart + lid; j < FCacheLen; j += lsize)
  {
    __global const float* krow = FKCache + plane + j * FDk;
    float acc = 0.0f;
    for (d = 0; d < FDk; d++) acc = mad(qloc[d], krow[d], acc);
    float sc = acc * FInvSqrtDk;
    if (FScoreSoftCap > 0.0f)
      sc = FScoreSoftCap * tanh(sc * FInvScoreSoftCap);
    FScores[scoreBase + j] = sc;
    m = fmax(m, sc);
  }
  FScratch[lid] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (s = lsize >> 1; s > 0; s >>= 1)
  {
    if (lid < s) FScratch[lid] = fmax(FScratch[lid], FScratch[lid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float MaxScore = FScratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  // ---- phase 2: shifted exp in place (same lane owns the same j), then the
  // normalizer ----
  float partial = 0.0f;
  for (j = jStart + lid; j < FCacheLen; j += lsize)
  {
    const float e = exp(FScores[scoreBase + j] - MaxScore);
    FScores[scoreBase + j] = e;
    partial += e;
  }
  FScratch[lid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (s = lsize >> 1; s > 0; s >>= 1)
  {
    if (lid < s) FScratch[lid] += FScratch[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float SumExp = FScratch[0];
  // Phase 3 reads score entries written by OTHER lanes, so the fence spans
  // global memory too.
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // ---- phase 3: the value sum, one output dimension per lane ----
  // SumExp = 0 cannot arise here (the live range is never empty and exp of the
  // shifted max is 1), but the host path zeroes the row in that case and this
  // matches it.
  const float InvSumExp = (SumExp > 0.0f) ? (1.0f / SumExp) : 0.0f;
  for (d = lid; d < FDk; d += lsize)
  {
    float acc = 0.0f;
    for (j = jStart; j < FCacheLen; j++)
      acc = mad(FScores[scoreBase + j], FVCache[plane + j * FDk + d], acc);
    FY[qBase + d] = acc * InvSumExp;
  }
}

// INT8 KV-CACHE APPEND (TNNetFusedSDPA, int8 cache). Quantizes the token's K
// and V slices into the resident int8 cache at slot FCacheSlot, one work-group
// per KV head, lanes splitting FDk. The format is the host's, unchanged: one
// symmetric FP32 scale per row, scale = maxabs/127, codes in [-127,127], laid
// out exactly as TNNetScaledDotProductAttention.QuantizeCacheRow writes them,
// so the cache stays byte-comparable with FKCacheQ/FVCacheQ.
//
// Three deliberate choices:
//   - rint, not round: OpenCL's round is half-away-from-zero while FPC's Round
//     is half-to-even, and rint is half-to-even under the default rounding mode.
//   - multiply by 1/maxabs FIRST and by 127 second, like TNNetVolume.QuantizeInt8:
//     forming 127/maxabs overflows single precision for a tiny row.
//   - a denormal row maximum (below MinSingle) emits zero codes and unit scale.
//     The host scales in double there; a kernel cannot, and the values involved
//     are below 1e-38.
// Finiteness is tested on the bit pattern rather than with isnan, because
// -cl-fast-relaxed-math implies -cl-finite-math-only and may fold isnan away.
// FScratch is LocalSize floats. Coded by Claude (AI).
__kernel void cai_sdpa_append_kv_int8
(
  const int FKVHeads,
  const int FDk,
  const int FCacheMax,
  const int FCacheSlot,
  const int FQW,
  const int FKW,
  __global const float* FX,
  __global char* FKCodes,
  __global float* FKScales,
  __global char* FVCodes,
  __global float* FVScales,
  __local float* FScratch
)
{
  const int g = get_group_id(1);
  const int lid = get_local_id(0);
  const int lsize = get_local_size(0);
  int s, d, part;
  if (g >= FKVHeads) return;
  const int slot = g * FCacheMax + FCacheSlot;
  const int dst = slot * FDk;

  for (part = 0; part < 2; part++)
  {
    const int base = (part == 0) ? (FQW + g * FDk) : (FQW + FKW + g * FDk);
    __global char* codes = (part == 0) ? FKCodes : FVCodes;
    __global float* scales = (part == 0) ? FKScales : FVScales;

    // ---- the row maximum over finite magnitudes, MaxAbsFinite's rule ----
    float m = 0.0f;
    for (d = lid; d < FDk; d += lsize)
    {
      const float a = fabs(FX[base + d]);
      if (as_uint(a) <= 0x7F7FFFFFu) m = fmax(m, a);
    }
    FScratch[lid] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (s = lsize >> 1; s > 0; s >>= 1)
    {
      if (lid < s) FScratch[lid] = fmax(FScratch[lid], FScratch[lid + s]);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float MaxAbs = FScratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    const int Usable = (as_uint(MaxAbs) >= 0x00800000u);
    const float RowScale = Usable ? (MaxAbs / 127.0f) : 1.0f;
    const float Recip = Usable ? (1.0f / MaxAbs) : 0.0f;
    if (lid == 0) scales[slot] = RowScale;
    for (d = lid; d < FDk; d += lsize)
    {
      const float v = FX[base + d];
      float scaled = v * Recip * 127.0f;
      // NaN maps to code 0 and an infinity clamps to +/-127, both matching the
      // host quantizer.
      if (as_uint(fabs(v)) > 0x7F800000u) scaled = 0.0f;
      codes[dst + d] = (char)clamp(rint(scaled), -127.0f, 127.0f);
    }
  }
}

// CACHED-DECODE SCALED DOT-PRODUCT ATTENTION OVER AN INT8 KV CACHE
// (TNNetFusedSDPA). Phase for phase the same kernel as cai_sdpa_decode above -
// one work-group per query head, the score band in global memory, a
// CLK_GLOBAL_MEM_FENCE barrier between phases 2 and 3 - and only the loads
// differ: the codes stream straight into the accumulator and the row scale is
// folded in as one scalar OUTSIDE the element loop, so the cache is never
// dequantized into memory. That is what makes an int8 cache a bandwidth saving
// rather than a bandwidth cost, and it mirrors what
// TNNetFusedSDPA.ComputeCachedToken already does on the host.
// char is signed in OpenCL C, so (float)code sign-extends with no mask.
// FScratch is LocalSize + FDk floats. Coded by Claude (AI).
__kernel void cai_sdpa_decode_int8
(
  const int FQHeads,
  const int FGroupSize,
  const int FDk,
  const int FCacheMax,
  const int FCacheLen,
  const int FWindow,
  const float FInvSqrtDk,
  const float FScoreSoftCap,
  const float FInvScoreSoftCap,
  __global const float* FX,
  __global const char* FKCodes,
  __global const float* FKScales,
  __global const char* FVCodes,
  __global const float* FVScales,
  __global float* FScores,
  __global float* FY,
  __local float* FScratch
)
{
  const int h = get_group_id(1);
  const int lid = get_local_id(0);
  const int lsize = get_local_size(0);
  int s, d, j;
  if (h >= FQHeads) return;

  __local float* qloc = FScratch + lsize;

  const int g = h / FGroupSize;
  const int qBase = h * FDk;
  const int scalePlane = g * FCacheMax;
  const int plane = scalePlane * FDk;
  const int scoreBase = h * FCacheMax;
  const int jStart = ((FWindow > 0) && (FCacheLen > FWindow))
                     ? (FCacheLen - FWindow) : 0;

  for (d = lid; d < FDk; d += lsize) qloc[d] = FX[qBase + d];
  barrier(CLK_LOCAL_MEM_FENCE);

  // ---- phase 1: scores over the live cache, then the row max ----
  float m = -1e30f;
  for (j = jStart + lid; j < FCacheLen; j += lsize)
  {
    __global const char* krow = FKCodes + plane + j * FDk;
    float acc = 0.0f;
    for (d = 0; d < FDk; d++) acc = mad(qloc[d], (float)krow[d], acc);
    float sc = acc * (FKScales[scalePlane + j] * FInvSqrtDk);
    if (FScoreSoftCap > 0.0f)
      sc = FScoreSoftCap * tanh(sc * FInvScoreSoftCap);
    FScores[scoreBase + j] = sc;
    m = fmax(m, sc);
  }
  FScratch[lid] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (s = lsize >> 1; s > 0; s >>= 1)
  {
    if (lid < s) FScratch[lid] = fmax(FScratch[lid], FScratch[lid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float MaxScore = FScratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  // ---- phase 2: shifted exp in place (same lane owns the same j), then the
  // normalizer ----
  float partial = 0.0f;
  for (j = jStart + lid; j < FCacheLen; j += lsize)
  {
    const float e = exp(FScores[scoreBase + j] - MaxScore);
    FScores[scoreBase + j] = e;
    partial += e;
  }
  FScratch[lid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (s = lsize >> 1; s > 0; s >>= 1)
  {
    if (lid < s) FScratch[lid] += FScratch[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float SumExp = FScratch[0];
  // Phase 3 reads score entries written by OTHER lanes, so the fence spans
  // global memory too.
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // ---- phase 3: the value sum, one output dimension per lane ----
  const float InvSumExp = (SumExp > 0.0f) ? (1.0f / SumExp) : 0.0f;
  for (d = lid; d < FDk; d += lsize)
  {
    float acc = 0.0f;
    for (j = jStart; j < FCacheLen; j++)
      acc = mad(FScores[scoreBase + j] * FVScales[scalePlane + j],
                (float)FVCodes[plane + j * FDk + d], acc);
    FY[qBase + d] = acc * InvSumExp;
  }
}

// FUSED MIXTURE-OF-EXPERTS DOWN PROJECTION (TNNetMoEExpertBankDown).
// Computes the whole gate-weighted expert mixture of one MoE block in a single
// launch:
//
//   Out[t,h] = SUM over the slots j of token t:
//                SlotGate[t,j] * ( W_down[SlotExpert[t,j]][h] . Hidden[t,j][:]
//                                  + Bias[SlotExpert[t,j]][h] )
//
// The routing decision is made on the host (the gate|up bank's slot map) and
// rides in as three small arrays, so this kernel never scans the router. Slots
// a token did not fill are simply not iterated (SlotCount[t] <= TopK), and a
// token with no slot at all yields exactly 0.
// FUnitCombine = 1 is Llama-4 routing, where the gate already scaled the expert
// INPUT: the mixture then combines the slots with weight 1.
// The bias is added INSIDE the slot loop and is therefore gate-weighted, once
// per slot - matching TNNetMoEExpertBankDown.ComputeMixture exactly.
// One work-item per output element; global size = (FHiddenSize, FTokenCnt).
// Deliberately unoptimized (no local memory, no tiling, no manual work-group
// size) so the runtime picks its own configuration on every device.
// Coded by Claude (AI).
__kernel void cai_moe_expert_down
(
  const int FTokenCnt,
  const int FTopK,
  const int FHiddenSize,
  const int FExpertWidth,
  const int FUnitCombine,
  const int FUseBias,
  __global const int*   FSlotExpert,
  __global const float* FSlotGate,
  __global const int*   FSlotCount,
  __global const float* FHidden,
  __global const float* FWeights,
  __global const float* FBias,
  __global float* FOut
)
{
  const int h = get_global_id(0);
  const int t = get_global_id(1);
  if ((h >= FHiddenSize) || (t >= FTokenCnt)) return;

  const int PairBase = t * FTopK;
  const int SlotCnt = FSlotCount[t];
  float acc = 0.0f;

  for (int j = 0; j < SlotCnt; j++)
  {
    const int e = FSlotExpert[PairBase + j];
    const float g = (FUnitCombine != 0) ? 1.0f : FSlotGate[PairBase + j];
    // Row-major weights: the layer's int8/FP32 row order, uploaded verbatim.
    const int row = e * FHiddenSize + h;
    const int wBase = row * FExpertWidth;
    const int hBase = (PairBase + j) * FExpertWidth;
    float d = 0.0f;
    for (int i = 0; i < FExpertWidth; i++)
    {
      d = mad(FWeights[wBase + i], FHidden[hBase + i], d);
    }
    if (FUseBias != 0) d += FBias[row];
    acc = mad(g, d, acc);
  }

  FOut[t * FHiddenSize + h] = acc;
} // end of kernel

// Int8-weight twin of cai_moe_expert_down. The weights are the layer's
// per-row symmetric int8 codes (dequantized value = code * FScales[row]) in
// the SAME row-major order, so the quantization table uploads verbatim. The
// per-row scale is applied ONCE to the reduced raw code sum and BEFORE the
// (FP32, unscaled) bias, mirroring the host fused kernel
// (TNNetMoEExpertBankDown.ComputeMixtureQuantizedInt8) and cai_dot_product_int8
// so device and host agree to normal float tolerance. Coded by Claude (AI).
__kernel void cai_moe_expert_down_int8
(
  const int FTokenCnt,
  const int FTopK,
  const int FHiddenSize,
  const int FExpertWidth,
  const int FUnitCombine,
  const int FUseBias,
  __global const int*   FSlotExpert,
  __global const float* FSlotGate,
  __global const int*   FSlotCount,
  __global const float* FHidden,
  __global const char*  FCodes,
  __global const float* FBias,
  __global float* FOut,
  __global const float* FScales
)
{
  const int h = get_global_id(0);
  const int t = get_global_id(1);
  if ((h >= FHiddenSize) || (t >= FTokenCnt)) return;

  const int PairBase = t * FTopK;
  const int SlotCnt = FSlotCount[t];
  float acc = 0.0f;

  for (int j = 0; j < SlotCnt; j++)
  {
    const int e = FSlotExpert[PairBase + j];
    const float g = (FUnitCombine != 0) ? 1.0f : FSlotGate[PairBase + j];
    const int row = e * FHiddenSize + h;
    const int wBase = row * FExpertWidth;
    const int hBase = (PairBase + j) * FExpertWidth;
    float d = 0.0f;
    for (int i = 0; i < FExpertWidth; i++)
    {
      d = mad(convert_float(FCodes[wBase + i]), FHidden[hBase + i], d);
    }
    // Deferred per-row dequantization scale: once on the raw code sum, BEFORE
    // the bias - the same order as the host fused path.
    d *= FScales[row];
    if (FUseBias != 0) d += FBias[row];
    acc = mad(g, d, acc);
  }

  FOut[t * FHiddenSize + h] = acc;
} // end of kernel
