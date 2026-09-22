unit neuralavxconst;

interface

type
  T8Single = array[0..7] of Single;
  P8Single = ^T8Single;

  T8Integer = array[0..7] of Integer;
  P8Integer = ^T8Integer;

// Constants for AVX2/AVX-512 exponential approximation
// exp(x) = 2^(x*log2e). Split t=x*log2e into k=round(t) and f=t-k in [-0.5,0.5];
// 2^k from exponent bits, 2^f via degree-6 minimax polynomial.
const
  cAVXExpHi:  Single = 88.3762626647949;
  cAVXExpLo:  Single = -88.3762626647949;
  cAVXLog2e:  Single = 1.44269504088896341;
  cAVXLn2:    Single = 0.6931471805599453;
  cAVXExpP0:  Single = 1.0;
  cAVXExpP1:  Single = 1.0;
  cAVXExpP2:  Single = 0.5;
  cAVXExpP3:  Single = 0.16666666666666666;
  cAVXExpP4:  Single = 0.041666666666666664;
  cAVXExpP5:  Single = 0.008333333333333333;
  cAVXExpP6:  Single = 0.001388888888888889;
  cAVXExp127: Integer = 127;

const
  cAVX8SSOne: array[0..7] of Single = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

const
  cAVXArgLaneSeed: array[0..15] of longint = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
  cAVXArgLaneStep: array[0..15] of longint = (16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16);
  cAVXArgAbsMask: array[0..7] of longint = ($7FFFFFFF, $7FFFFFFF, $7FFFFFFF, $7FFFFFFF,
                                            $7FFFFFFF, $7FFFFFFF, $7FFFFFFF, $7FFFFFFF);

// AVX2 logarithm constants (Cephes logf)
const
  cAVXLnMinNorm: array[0..7] of Single = (1.1754943508222875e-38, 1.1754943508222875e-38, 1.1754943508222875e-38, 1.1754943508222875e-38,
                                          1.1754943508222875e-38, 1.1754943508222875e-38, 1.1754943508222875e-38, 1.1754943508222875e-38);
  cAVXLnInvMant: array[0..7] of Integer = ($007FFFFF, $007FFFFF, $007FFFFF, $007FFFFF,
                                           $007FFFFF, $007FFFFF, $007FFFFF, $007FFFFF);
  cAVXLnHalf: array[0..7] of Single = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
  cAVXLnSqrtHf: array[0..7] of Single = (0.707106781186547524, 0.707106781186547524, 0.707106781186547524, 0.707106781186547524,
                                         0.707106781186547524, 0.707106781186547524, 0.707106781186547524, 0.707106781186547524);

  cAVXLnP0: array[0..7] of Single = (7.0376836292E-2, 7.0376836292E-2, 7.0376836292E-2, 7.0376836292E-2,
                                     7.0376836292E-2, 7.0376836292E-2, 7.0376836292E-2, 7.0376836292E-2);
  cAVXLnP1: array[0..7] of Single = (-1.1514610310E-1, -1.1514610310E-1, -1.1514610310E-1, -1.1514610310E-1,
                                     -1.1514610310E-1, -1.1514610310E-1, -1.1514610310E-1, -1.1514610310E-1);
  cAVXLnP2: array[0..7] of Single = (1.1676998740E-1, 1.1676998740E-1, 1.1676998740E-1, 1.1676998740E-1,
                                     1.1676998740E-1, 1.1676998740E-1, 1.1676998740E-1, 1.1676998740E-1);
  cAVXLnP3: array[0..7] of Single = (-1.2420140846E-1, -1.2420140846E-1, -1.2420140846E-1, -1.2420140846E-1,
                                     -1.2420140846E-1, -1.2420140846E-1, -1.2420140846E-1, -1.2420140846E-1);
  cAVXLnP4: array[0..7] of Single = (1.4249322787E-1, 1.4249322787E-1, 1.4249322787E-1, 1.4249322787E-1,
                                     1.4249322787E-1, 1.4249322787E-1, 1.4249322787E-1, 1.4249322787E-1);
  cAVXLnP5: array[0..7] of Single = (-1.6668057665E-1, -1.6668057665E-1, -1.6668057665E-1, -1.6668057665E-1,
                                     -1.6668057665E-1, -1.6668057665E-1, -1.6668057665E-1, -1.6668057665E-1);
  cAVXLnP6: array[0..7] of Single = (2.0000714765E-1, 2.0000714765E-1, 2.0000714765E-1, 2.0000714765E-1,
                                     2.0000714765E-1, 2.0000714765E-1, 2.0000714765E-1, 2.0000714765E-1);
  cAVXLnP7: array[0..7] of Single = (-2.4999993993E-1, -2.4999993993E-1, -2.4999993993E-1, -2.4999993993E-1,
                                     -2.4999993993E-1, -2.4999993993E-1, -2.4999993993E-1, -2.4999993993E-1);
  cAVXLnP8: array[0..7] of Single = (3.3333331174E-1, 3.3333331174E-1, 3.3333331174E-1, 3.3333331174E-1,
                                     3.3333331174E-1, 3.3333331174E-1, 3.3333331174E-1, 3.3333331174E-1);
  cAVXLnQ1: array[0..7] of Single = (-2.12194440E-4, -2.12194440E-4, -2.12194440E-4, -2.12194440E-4,
                                     -2.12194440E-4, -2.12194440E-4, -2.12194440E-4, -2.12194440E-4);
  cAVXLnQ2: array[0..7] of Single = (0.693359375, 0.693359375, 0.693359375, 0.693359375,
                                     0.693359375, 0.693359375, 0.693359375, 0.693359375);


// ---------------------------------------------------------------------------
// AVX2 Sin/Cos constants (avx_mathfun style)
// ---------------------------------------------------------------------------

const
  cOneInt: Integer = 1;   // used for broadcasting via vpbroadcastd

  // sin(r) = r * (P0 + z*(P1 + z*(P2 + z*P3)))
  cAVXSinP0: Single =  1.0;
  cAVXSinP1: Single = -1.6666654611E-1;
  cAVXSinP2: Single =  8.3321608736E-3;
  cAVXSinP3: Single = -1.9515295891E-4;

  // cos(r) = 1 + z*(Q0 + z*(Q1 + z*Q2))
  cAVXCosQ0: Single = -0.5;
  cAVXCosQ1: Single =  4.166664568298827E-2;
  cAVXCosQ2: Single = -1.388731625493765E-3;

  cAVXSinCosPi4: Single = 0.7853981633974483;   // pi/4
  cAVXSinCosPi2: Single = 1.5707963267948966;   // pi/2
  cAVXSinCosInvPi2: Single = 0.6366197723675813; // 2/pi

  cAVXSinCosSignMask: array[0..7] of Cardinal = ($80000000, $80000000, $80000000, $80000000,
                                                $80000000, $80000000, $80000000, $80000000);


// ---- Constants for AVX2 Sin/Cos Both ----

// j = (trunc(|x|*4/pi) + 1) & ~1
const
  cAVXSC_FOPI: array[0..7] of Single =
    (1.2732395447351627, 1.2732395447351627, 1.2732395447351627, 1.2732395447351627,
     1.2732395447351627, 1.2732395447351627, 1.2732395447351627, 1.2732395447351627);

  cAVXSC_Half: array[0..7] of Single =
    (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);

  // r = |x| + y*DP1 + y*DP2 + y*DP3  (avx_mathfun reduction coefficients)
  cAVXSC_DP1: array[0..7] of Single =
    (-0.78515625, -0.78515625, -0.78515625, -0.78515625,
     -0.78515625, -0.78515625, -0.78515625, -0.78515625);

  cAVXSC_DP2: array[0..7] of Single =
    (-2.4187564849853515625e-4, -2.4187564849853515625e-4,
     -2.4187564849853515625e-4, -2.4187564849853515625e-4,
     -2.4187564849853515625e-4, -2.4187564849853515625e-4,
     -2.4187564849853515625e-4, -2.4187564849853515625e-4);

  cAVXSC_DP3: array[0..7] of Single =
    (-3.77489497744594108e-8, -3.77489497744594108e-8,
     -3.77489497744594108e-8, -3.77489497744594108e-8,
     -3.77489497744594108e-8, -3.77489497744594108e-8,
     -3.77489497744594108e-8, -3.77489497744594108e-8);

  // Highest-power cos coefficient (the remaining ones reuse cAVXCosQ2, cAVXCosQ1)
  cAVXSC_CosP0: array[0..7] of Single =
    (2.443315711809948e-5, 2.443315711809948e-5,
     2.443315711809948e-5, 2.443315711809948e-5,
     2.443315711809948e-5, 2.443315711809948e-5,
     2.443315711809948e-5, 2.443315711809948e-5);

  // Integer vectors used by the bit-twiddling reductions
  cAVXSC_1i: array[0..7] of Integer = (1, 1, 1, 1, 1, 1, 1, 1);
  cAVXSC_2i: array[0..7] of Integer = (2, 2, 2, 2, 2, 2, 2, 2);
  cAVXSC_4i: array[0..7] of Integer = (4, 4, 4, 4, 4, 4, 4, 4);
  cAVXSC_NOT1i: array[0..7] of UInt32 =
    ($FFFFFFFE, $FFFFFFFE, $FFFFFFFE, $FFFFFFFE,
     $FFFFFFFE, $FFFFFFFE, $FFFFFFFE, $FFFFFFFE);

implementation

end.
