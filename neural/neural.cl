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
  __global __read_only float* FInputBufferAs,
  __global __read_only float* FInputBufferBs,
  __global   float* FResultBuffer
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
  __global __read_only float* FInputBufferAs,
  __global __read_only float* FInputBufferBs,
  __global   float* FResultBuffer
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
  __global __read_only float* A,
  __global __read_only float* B,
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
  __global __read_only float16* FInputBufferAs,
  __global __read_only float16* FInputBufferBs,
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
  __global __read_only float* A,
  __global __read_only float* B,
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
  __global __read_only float* A,
  __global __read_only float* B,
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
