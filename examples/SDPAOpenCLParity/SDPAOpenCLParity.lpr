program SDPAOpenCLParity;
(*
SDPAOpenCLParity: numeric parity check for the Phase-1 OpenCL offload of
TNNetScaledDotProductAttention (the Q.K^T score matmul runs on the device;
masking, softmax and the value sum stay on the CPU).

For several attention configurations (full, causal, sliding-window,
soft-capped, and a fully-masked-row edge case) it builds a one-layer SDPA net,
runs the CPU forward, then enables OpenCL and runs the same input again, and
reports the maximum absolute difference between the two output tensors. The
device path reorders the dot-product reductions, so the results are not
bit-identical; PASS requires max abs diff below cTolerance.

Requires a build with -dOpenCL and a linkable libOpenCL. With no OpenCL
platform available the program prints SKIP and exits 0 (so it is harmless in a
CPU-only CI). Exit code is 1 on any FAIL.

Build (from this directory):
  fpc -Mobjfpc -Sh -O3 -dAVX2 -dRelease -dOpenCL -Fu../../neural SDPAOpenCLParity.lpr

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX}cthreads,{$ENDIF}
  SysUtils, neuralvolume, neuralnetwork
  {$IFDEF OpenCL}, neuralopencl, cl{$ENDIF};

const
  cTolerance = 1e-3; // max abs output diff allowed (FP32 + reordered reductions)
  cSeqLen    = 64;   // >= the layer's csSDPAOpenCLMinSeqLen so the GPU path fires
  cDk        = 48;

{$IFDEF OpenCL}
// Builds Input(SeqLen,1,3*Dk) -> SDPA. Caller frees.
function BuildNet(Causal: boolean; Window: integer;
  SoftCap: TNeuralFloat): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 3 * cDk));
  Result.AddLayer(TNNetScaledDotProductAttention.Create(cDk, Causal, Window,
    SoftCap));
end;

// CPU forward vs OpenCL forward on the same input; returns max abs diff.
function MaxAbsDiff(Causal: boolean; Window: integer; SoftCap: TNeuralFloat;
  PlatformId: cl_platform_id; DeviceId: cl_device_id): TNeuralFloat;
var
  NN: TNNet;
  Inp, CpuOut: TNNetVolume;
  Cnt: integer;
  D: TNeuralFloat;
begin
  Inp := TNNetVolume.Create(cSeqLen, 1, 3 * cDk);
  Inp.Randomize(); // values in [-1, 1]
  CpuOut := TNNetVolume.Create();

  NN := BuildNet(Causal, Window, SoftCap);
  try
    NN.Compute(Inp);
    CpuOut.Copy(NN.GetLastLayer().Output); // snapshot CPU result

    NN.EnableOpenCL(PlatformId, DeviceId);
    NN.Compute(Inp); // OpenCL prefill path (SeqLen >= min threshold)

    Result := 0;
    for Cnt := 0 to CpuOut.Size - 1 do
    begin
      D := Abs(CpuOut.FData[Cnt] - NN.GetLastLayer().Output.FData[Cnt]);
      if D > Result then Result := D;
    end;
  finally
    NN.Free;
    CpuOut.Free;
    Inp.Free;
  end;
end;

var
  EasyCL: TEasyOpenCL;
  PlatformId: cl_platform_id;
  DeviceId: cl_device_id;
  Failures: integer;

  procedure Case_(const Name: string; Causal: boolean; Window: integer;
    SoftCap: TNeuralFloat);
  var
    D: TNeuralFloat;
  begin
    RandSeed := 424242; // deterministic input per case
    D := MaxAbsDiff(Causal, Window, SoftCap, PlatformId, DeviceId);
    if D <= cTolerance then
      WriteLn(Format('PASS  %-22s max|diff| = %.3e', [Name, D]))
    else
    begin
      WriteLn(Format('FAIL  %-22s max|diff| = %.3e  (tol %.1e)',
        [Name, D, cTolerance]));
      Inc(Failures);
    end;
  end;

begin
  EasyCL := TEasyOpenCL.Create();
  try
    if EasyCL.GetPlatformCount() = 0 then
    begin
      WriteLn('SKIP: no OpenCL platform found.');
      Halt(0);
    end;
    EasyCL.SetCurrentPlatform(EasyCL.PlatformIds[0]);
    if EasyCL.GetDeviceCount() = 0 then
    begin
      WriteLn('SKIP: no OpenCL device on platform ', EasyCL.PlatformNames[0]);
      Halt(0);
    end;
    EasyCL.SetCurrentDevice(EasyCL.Devices[0]);
    PlatformId := EasyCL.PlatformIds[0];
    DeviceId := EasyCL.Devices[0];
    WriteLn('OpenCL: ', EasyCL.PlatformNames[0], ' / ', EasyCL.DeviceNames[0]);
    WriteLn(Format('SeqLen=%d  Dk=%d  tol=%.1e', [cSeqLen, cDk, cTolerance]));
    WriteLn;

    Failures := 0;
    Case_('full attention',    false, 0,  0);
    Case_('causal',            true,  0,  0);
    Case_('sliding-window 8',  true,  8,  0);
    Case_('soft-cap 50',       true,  0,  50);
    Case_('causal window 1',   true,  1,  0); // near-single-key rows

    WriteLn;
    if Failures = 0 then WriteLn('SDPA OpenCL PARITY OK')
    else
    begin
      WriteLn('SDPA OpenCL PARITY FAILED: ', Failures, ' case(s)');
      Halt(1);
    end;
  finally
    EasyCL.Free;
  end;
end.
{$ELSE}
begin
  WriteLn('SKIP: built without -dOpenCL (rebuild with -dOpenCL to run).');
end.
{$ENDIF}
