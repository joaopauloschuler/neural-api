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

    if (ActFN == 1)
    {
      if (DotProductResult < 0.0f) { DotProductResult = 0.0f; }
    }

    FResultBuffer[b_id * FNumAs + a_id] = DotProductResult;
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
  if (FActFlag == 0)            // GLU: sigmoid(B)
    gated = 1.0f / (1.0f + exp(-b));
  else if (FActFlag == 1)       // SwiGLU: swish(B) = B*sigmoid(B)
    gated = b * (1.0f / (1.0f + exp(-b)));
  else if (FActFlag == 2)       // GEGLU: gelu_tanh(B)
  {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float GELU_CONST = 0.044715f;
    const float arg = SQRT_2_OVER_PI * (b + GELU_CONST * b * b * b);
    gated = b * 0.5f * (1.0f + tanh(arg));
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
