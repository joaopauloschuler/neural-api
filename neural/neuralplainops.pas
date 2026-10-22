unit neuralplainops;

(*
neuralplainops
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).
*)

// neuralplainops -- inference ops on plain TNNetVolume buffers, without a
// TNNet: 1D conv, pointwise conv, dense and depthwise transposed 1D conv,
// nearest upsample, reflect pad, instance norm with a channel affine, Snake,
// a norm + Snake + conv chunked over time, LeakyReLU, channel concat and a
// bidirectional LSTM. A caller holds the weights once and
// reuses a few activation volumes, so memory does not grow with per-layer
// buffers (PyTorch eager style).
//
// Sequences use the framework layout: SizeX = time, SizeY = 1,
// Depth = channels, element (x, c) at x * Depth + c.
//
// TNNetPlainOps owns the per-thread scratch (tile im2col patch and tile
// result) and splits the time axis over the shared pool fNTL. fNTL is not
// re-entrant: never call these ops from inside a TNNet parallel compute or
// from another fNTL job; construct with pThreaded = false there.

{$include neuralnetwork.inc}

interface

uses
  Classes, SysUtils, Math, neuralvolume, neuralnetwork, neuralthread;

type
  // Row layout of TNNetPlainConvWeights.Rows, one per op family; the pclConv
  // and pclTransposed rows are packed by TNNetVolume.PackDotProductRows.
  TNNetPlainConvLayout = (
    pclConv,                 // [Cout][K][Cin]: Conv1D, PointwiseConv
    pclTransposed,           // [K][Cout][Cin]: ConvTranspose1D
    pclDepthwiseTransposed); // [K][C]: DepthwiseConvTranspose1D

  // Weights and bias of one 1D conv in the layout its TNNetPlainOps op reads.
  TNNetPlainConvWeights = class
  private
    FRows, FBias: TNNetVolume;
    FLayout: TNNetPlainConvLayout;
    FOutChannels, FInChannels, FKernelSize: integer;
    FHasBias: boolean;
    procedure SetShape(pLayout: TNNetPlainConvLayout;
      pOutChannels, pInChannels, pKernelSize: integer);
    procedure LoadLayerBias(Layer: TNNetLayer; pHasBias: boolean);
    procedure LoadTorchBias(Bias: TNNetVolume);
    // Packs the filled pclConv / pclTransposed rows for DotProductsPacked.
    procedure PackRows();
  public
    constructor Create();
    destructor Destroy(); override;
    // Rows = the neurons' (K, 1, Cin) weights back to back, as in
    // TNNetConv1DLinear / TNNetPointwiseConvLinear. pHasBias reads BiasWeight.
    procedure LoadConv1DLayer(Conv: TNNetLayer; pHasBias: boolean);
    // torch Conv1d weight [Cout][Cin][K]; Bias nil = no bias.
    procedure LoadTorchConv1D(Weight, Bias: TNNetVolume;
      pOutChannels, pInChannels, pKernelSize: integer);
    // A dense TNNetDeconv1DLinear (groups 1) into the [K][Cout][Cin] rows.
    procedure LoadDeconv1DLayer(Deconv: TNNetLayer; pHasBias: boolean);
    // torch ConvTranspose1d weight [Cin][Cout][K] (groups 1); Bias nil = none.
    procedure LoadTorchConvTranspose1D(Weight, Bias: TNNetVolume;
      pInChannels, pOutChannels, pKernelSize: integer);
    // A TNNetDeconv1DLinear with groups = channels into the [K][C] taps.
    procedure LoadDepthwiseDeconv1DLayer(Deconv: TNNetLayer; pHasBias: boolean);
    // torch ConvTranspose1d weight [C][1][K] with groups = C; Bias nil = none.
    procedure LoadTorchDepthwiseConvTranspose1D(Weight, Bias: TNNetVolume;
      pChannels, pKernelSize: integer);
    property Rows: TNNetVolume read FRows;
    property Bias: TNNetVolume read FBias;
    property Layout: TNNetPlainConvLayout read FLayout;
    property OutChannels: integer read FOutChannels;
    property InChannels: integer read FInChannels;
    property KernelSize: integer read FKernelSize;
    property HasBias: boolean read FHasBias;
  end;

  // One LSTM direction: gate rows in torch order i, f, g, o.
  TNNetPlainLSTMWeights = class
  private
    FInputRows, FRecurrentRows, FBias: TNNetVolume;
    FHiddenSize, FInChannels: integer;
  public
    constructor Create();
    destructor Destroy(); override;
    // TNNetLSTMCell neurons [0..3] input, [4..7] recurrent, [8..11] biases.
    procedure LoadLSTMCell(Cell: TNNetLayer);
    // torch weight_ih [4H][In], weight_hh [4H][H], bias = bias_ih + bias_hh.
    procedure LoadTorchLSTM(WeightIH, WeightHH, BiasIH, BiasHH: TNNetVolume);
    property InputRows: TNNetVolume read FInputRows;         // [4H][In]
    property RecurrentRows: TNNetVolume read FRecurrentRows; // [4H][H]
    property Bias: TNNetVolume read FBias;                   // [4H]
    property HiddenSize: integer read FHiddenSize;
    property InChannels: integer read FInChannels;
  end;

  // Runs the ops and owns their reusable scratch. Scratch grows at op entry
  // (never inside the per-tile work) and is kept for the next call.
  TNNetPlainOps = class
  private
    FThreaded: boolean;
    FTilePositions, FMinPositionsToThread: integer;
    // Per pool thread: tile im2col patch / input rows, and the tile result.
    FPatch, FTileOut: array of TNNetVolume;
    // Per LSTM direction: input projection of every step, gates, h and c.
    FProjection, FGates, FHidden, FCell: array[0..1] of TNNetVolume;
    // Per channel: the prepared InstanceNorm mean, inv_std * Scale and Shift.
    FMean, FInvStd, FSum, FSumSq, FChannelScale, FNormShift: TNNetVolume;
    FInvAlpha: TNNetVolume;
    // NormSnakeConv1D: one chunk's input window and its conv output.
    FWindow, FChunkOutput: TNNetVolume;
    FMaxThreads: integer;
    // Arguments of the running job.
    FJobInput, FJobOutput, FJobAlpha: TNNetVolume;
    FJobConv: TNNetPlainConvWeights;
    FJobLSTM: array[0..1] of TNNetPlainLSTMWeights;
    FJobPositionCount, FJobStride, FJobDilation, FJobPadding: integer;
    FJobThreadCount: integer;
    FCappedJob: TNeuralProc;
    function ScratchSlotCount(): integer;
    // Pool threads a job uses: ScratchSlotCount capped by MaxThreads.
    function JobThreadCount(): integer;
    procedure EnsureTileScratch(PatchFloats, TileOutFloats: integer);
    function JobBiasPtr(): TNeuralFloatArrPtr;
    procedure RunJob(Job: TNeuralProc; PositionCount: integer);
    // Runs Job on JobThreadCount pool threads (Job sees that thread count).
    procedure RunOnPool(Job: TNeuralProc);
    procedure CappedJob(Index, ThreadNum: integer);
    procedure AddBiasToRows(Output: TNNetVolume; Bias: TNNetVolume;
      FirstPos, LastPos: integer);
    procedure Conv1DJob(Index, ThreadNum: integer);
    procedure PointwiseConvJob(Index, ThreadNum: integer);
    procedure ConvTranspose1DJob(Index, ThreadNum: integer);
    procedure DepthwiseConvTranspose1DJob(Index, ThreadNum: integer);
    procedure SnakeJob(Index, ThreadNum: integer);
    procedure LSTMProjectionJob(Index, ThreadNum: integer);
    procedure LSTMRecurrenceJob(Index, ThreadNum: integer);
    procedure RunLSTMDirection(Direction: integer);
    class procedure CheckSequence(V: TNNetVolume; Channels: integer;
      const OpName: string);
  public
    // pTilePositions time positions per tile (per-thread scratch scales with
    // it); pThreaded = false runs every op on the calling thread.
    constructor Create(pThreaded: boolean = true; pTilePositions: integer = 36);
    destructor Destroy(); override;
    // Grows V to at least FloatCount floats and never shrinks it.
    class procedure GrowScratch(V: TNNetVolume; FloatCount: integer);
    // torch Conv1d (zero padding, dilation); PaddingRight only sets the length.
    procedure Conv1D(Input, Output: TNNetVolume; Weights: TNNetPlainConvWeights;
      Stride, Dilation, PaddingLeft, PaddingRight: integer); overload;
    procedure Conv1D(Input, Output: TNNetVolume; Weights: TNNetPlainConvWeights;
      Stride, Dilation, Padding: integer); overload;
    // Kernel-1 Conv1d (pclConv weights with K = 1), no gather.
    procedure PointwiseConv(Input, Output: TNNetVolume;
      Weights: TNNetPlainConvWeights);
    // torch ConvTranspose1d, groups 1: length (T-1)*Stride - 2*Padding + K + OutputPadding.
    procedure ConvTranspose1D(Input, Output: TNNetVolume;
      Weights: TNNetPlainConvWeights; Stride, Padding, OutputPadding: integer);
    // torch ConvTranspose1d with groups = channels (one tap vector per k).
    procedure DepthwiseConvTranspose1D(Input, Output: TNNetVolume;
      Weights: TNNetPlainConvWeights; Stride, Padding, OutputPadding: integer);
    // Output row x = Input row x div Factor (torch nearest upsample).
    procedure UpsampleNearest(Input, Output: TNNetVolume; Factor: integer);
    // torch ReflectionPad1d((PaddingLeft, PaddingRight)); pads must be < length.
    procedure ReflectPad(Input, Output: TNNetVolume;
      PaddingLeft, PaddingRight: integer);
    // In place: per channel (x - mean) * inv_std (biased variance) * Scale + Shift.
    procedure InstanceNorm(V, Scale, Shift: TNNetVolume; Eps: TNeuralFloat = 1e-5);
    // Stores V's per-channel mean and inv_std * Scale, and Shift, for
    // ApplyInstanceNorm (InstanceNorm = PrepareInstanceNorm + ApplyInstanceNorm).
    procedure PrepareInstanceNorm(V, Scale, Shift: TNNetVolume;
      Eps: TNeuralFloat = 1e-5);
    // In place on any rows of the prepared depth: the prepared InstanceNorm.
    procedure ApplyInstanceNorm(V: TNNetVolume);
    // Conv1D(Snake(prepared InstanceNorm(Input))), stride 1, zero padding
    // Dilation*(K-1)/2, chunked over time; Accumulate adds it to Output.
    procedure NormSnakeConv1D(Input, Output: TNNetVolume;
      Weights: TNNetPlainConvWeights; Alpha: TNNetVolume; Dilation: integer;
      Accumulate: boolean; ChunkPositions: integer = 1024);
    // In place: x + 1/alpha * sin(alpha * x)^2 with one alpha per channel.
    procedure Snake(V, Alpha: TNNetVolume);
    // Output may be Input.
    procedure LeakyReLU(Input, Output: TNNetVolume; Slope: TNeuralFloat);
    // Output (T, 1, sum of depths): per position the Sources' channels in order
    // (torch.cat on dim 1). Output must not be a source.
    procedure ConcatChannels(const Sources: array of TNNetVolume;
      Output: TNNetVolume);
    // Output (T, 1, 2H) = [forward h_t ; backward h_t], zero initial state.
    procedure BiLSTM(Input, Output: TNNetVolume;
      ForwardWeights, BackwardWeights: TNNetPlainLSTMWeights);
    // Bytes of scratch this object holds now.
    function ScratchBytes(): int64;
    property Threaded: boolean read FThreaded;
    property TilePositions: integer read FTilePositions;
    // Ops on fewer positions run on the calling thread.
    property MinPositionsToThread: integer read FMinPositionsToThread
      write FMinPositionsToThread;
    // Most pool threads one op uses; 0 = every pool thread.
    property MaxThreads: integer read FMaxThreads write FMaxThreads;
  end;

// AdaIN1d / AdaLayerNorm channel affine for InstanceNorm, as
// WriteStyleIntoChannelAffine computes it: Scale = 1 + gamma, Shift = beta.
procedure ComputeStyleChannelAffine(FcWeight, FcBias: TNNetVolume;
  const Style: array of TNeuralFloat; Scale, Shift: TNNetVolume;
  NormWeight: TNNetVolume = nil; NormBias: TNNetVolume = nil);

implementation

const
  // A rows per DotProductsTiled tile (the LSTM input projection).
  csPlainOpsTileRows = 16;

procedure ComputeStyleChannelAffine(FcWeight, FcBias: TNNetVolume;
  const Style: array of TNeuralFloat; Scale, Shift: TNNetVolume;
  NormWeight: TNNetVolume = nil; NormBias: TNNetVolume = nil);
var
  ChannelCount, ChannelPos, MaxChannelPos, StyleCount: integer;
  GammaRowBase, BetaRowBase: integer;
  Gamma, Beta, ChannelScale, ChannelShift: TNeuralFloat;
  HasNormAffine: boolean;
begin
  StyleCount := Length(Style);
  ChannelCount := FcBias.Size div 2;
  if FcWeight.Size <> 2 * ChannelCount * StyleCount then
    raise Exception.Create('ComputeStyleChannelAffine: fc weight has ' +
      IntToStr(FcWeight.Size) + ' elements, expected ' +
      IntToStr(2 * ChannelCount * StyleCount) + '.');
  HasNormAffine := Assigned(NormWeight);
  Scale.ReSize(1, 1, ChannelCount);
  Shift.ReSize(1, 1, ChannelCount);
  MaxChannelPos := ChannelCount - 1;
  GammaRowBase := 0;
  BetaRowBase := ChannelCount * StyleCount;
  for ChannelPos := 0 to MaxChannelPos do
  begin
    Gamma := FcBias.FData[ChannelPos] + TNNetVolume.DotProduct(
      @FcWeight.FData[GammaRowBase], @Style[0], StyleCount);
    Beta := FcBias.FData[ChannelCount + ChannelPos] + TNNetVolume.DotProduct(
      @FcWeight.FData[BetaRowBase], @Style[0], StyleCount);
    ChannelScale := 1 + Gamma;
    ChannelShift := Beta;
    if HasNormAffine then
    begin
      ChannelShift := ChannelScale * NormBias.FData[ChannelPos] + Beta;
      ChannelScale := ChannelScale * NormWeight.FData[ChannelPos];
    end;
    Scale.FData[ChannelPos] := ChannelScale;
    Shift.FData[ChannelPos] := ChannelShift;
    Inc(GammaRowBase, StyleCount);
    Inc(BetaRowBase, StyleCount);
  end;
end;

{ TNNetPlainConvWeights }

constructor TNNetPlainConvWeights.Create();
begin
  inherited Create();
  FRows := TNNetVolume.Create();
  FBias := TNNetVolume.Create();
end;

destructor TNNetPlainConvWeights.Destroy();
begin
  FBias.Free;
  FRows.Free;
  inherited Destroy();
end;

procedure TNNetPlainConvWeights.SetShape(pLayout: TNNetPlainConvLayout;
  pOutChannels, pInChannels, pKernelSize: integer);
begin
  if (pOutChannels < 1) or (pInChannels < 1) or (pKernelSize < 1) then
    raise Exception.Create('TNNetPlainConvWeights: bad shape ' +
      IntToStr(pOutChannels) + ' x ' + IntToStr(pInChannels) + ' x ' +
      IntToStr(pKernelSize) + '.');
  FLayout := pLayout;
  FOutChannels := pOutChannels;
  FInChannels := pInChannels;
  FKernelSize := pKernelSize;
  if pLayout = pclDepthwiseTransposed
    then FRows.ReSize(pKernelSize, 1, pOutChannels)
    else FRows.ReSize(pOutChannels * pKernelSize, 1, pInChannels);
  FBias.ReSize(1, 1, pOutChannels);
  FBias.Fill(0);
  FHasBias := false;
end;

procedure TNNetPlainConvWeights.LoadLayerBias(Layer: TNNetLayer;
  pHasBias: boolean);
var
  NeuronPos, MaxNeuronPos: integer;
begin
  FHasBias := pHasBias;
  if not pHasBias then exit;
  MaxNeuronPos := FOutChannels - 1;
  for NeuronPos := 0 to MaxNeuronPos do
    FBias.FData[NeuronPos] := Layer.Neurons[NeuronPos].BiasWeight;
end;

procedure TNNetPlainConvWeights.LoadTorchBias(Bias: TNNetVolume);
begin
  FHasBias := Assigned(Bias);
  if not FHasBias then exit;
  if Bias.Size <> FOutChannels then
    raise Exception.Create('TNNetPlainConvWeights: bias has ' +
      IntToStr(Bias.Size) + ' elements, expected ' +
      IntToStr(FOutChannels) + '.');
  FBias.CopyNoChecks(Bias);
end;

procedure TNNetPlainConvWeights.PackRows();
var
  Unpacked: TNNetVolume;
  RowCount, VectorSize: integer;
begin
  if FLayout = pclConv then
  begin
    RowCount := FOutChannels;
    VectorSize := FKernelSize * FInChannels;
  end
  else
  begin
    RowCount := FKernelSize * FOutChannels;
    VectorSize := FInChannels;
  end;
  Unpacked := TNNetVolume.Create();
  try
    Unpacked.Copy(FRows);
    FRows.ReSize(TNNetVolume.PackedDotProductFloats(RowCount, VectorSize), 1, 1);
    TNNetVolume.PackDotProductRows(FRows.DataPtr, Unpacked.DataPtr, RowCount,
      VectorSize);
  finally
    Unpacked.Free;
  end;
end;

procedure TNNetPlainConvWeights.LoadConv1DLayer(Conv: TNNetLayer;
  pHasBias: boolean);
var
  NeuronPos, MaxNeuronPos, RowFloats: integer;
  NeuronWeights: TNNetVolume;
begin
  NeuronWeights := Conv.Neurons[0].Weights;
  if NeuronWeights.SizeY <> 1 then
    raise Exception.Create('LoadConv1DLayer: ' + Conv.ClassName +
      ' has a 2D kernel.');
  SetShape(pclConv, Conv.Neurons.Count, NeuronWeights.Depth,
    NeuronWeights.SizeX);
  RowFloats := FKernelSize * FInChannels;
  MaxNeuronPos := FOutChannels - 1;
  for NeuronPos := 0 to MaxNeuronPos do
    Move(Conv.Neurons[NeuronPos].Weights.FData[0],
      FRows.FData[NeuronPos * RowFloats], RowFloats * csNeuralFloatSize);
  PackRows();
  LoadLayerBias(Conv, pHasBias);
end;

procedure TNNetPlainConvWeights.LoadTorchConv1D(Weight, Bias: TNNetVolume;
  pOutChannels, pInChannels, pKernelSize: integer);
var
  OutPos, InPos, TapPos, SourcePos: integer;
  MaxOutPos, MaxInPos, MaxTapPos: integer;
begin
  if Weight.Size <> pOutChannels * pInChannels * pKernelSize then
    raise Exception.Create('LoadTorchConv1D: weight size does not fit.');
  SetShape(pclConv, pOutChannels, pInChannels, pKernelSize);
  MaxOutPos := pOutChannels - 1;
  MaxInPos := pInChannels - 1;
  MaxTapPos := pKernelSize - 1;
  SourcePos := 0;
  for OutPos := 0 to MaxOutPos do
    for InPos := 0 to MaxInPos do
      for TapPos := 0 to MaxTapPos do
      begin
        FRows.FData[(OutPos * pKernelSize + TapPos) * pInChannels + InPos] :=
          Weight.FData[SourcePos];
        Inc(SourcePos);
      end;
  PackRows();
  LoadTorchBias(Bias);
end;

procedure TNNetPlainConvWeights.LoadDeconv1DLayer(Deconv: TNNetLayer;
  pHasBias: boolean);
var
  NeuronPos, MaxNeuronPos, TapPos, MaxTapPos, RowBytes: integer;
  NeuronWeights: TNNetVolume;
begin
  if not (Deconv is TNNetDeconvolution) or
     (TNNetDeconvolution(Deconv).Groups <> 1) then
    raise Exception.Create('LoadDeconv1DLayer: needs a dense ' +
      'TNNetDeconv1DLinear, got ' + Deconv.ClassName + '.');
  NeuronWeights := Deconv.Neurons[0].Weights;
  SetShape(pclTransposed, Deconv.Neurons.Count, NeuronWeights.Depth,
    NeuronWeights.SizeX);
  RowBytes := FInChannels * csNeuralFloatSize;
  MaxNeuronPos := FOutChannels - 1;
  MaxTapPos := FKernelSize - 1;
  for NeuronPos := 0 to MaxNeuronPos do
  begin
    NeuronWeights := Deconv.Neurons[NeuronPos].Weights;
    for TapPos := 0 to MaxTapPos do
      Move(NeuronWeights.FData[TapPos * FInChannels],
        FRows.FData[(TapPos * FOutChannels + NeuronPos) * FInChannels],
        RowBytes);
  end;
  PackRows();
  LoadLayerBias(Deconv, pHasBias);
end;

procedure TNNetPlainConvWeights.LoadTorchConvTranspose1D(Weight,
  Bias: TNNetVolume; pInChannels, pOutChannels, pKernelSize: integer);
var
  OutPos, InPos, TapPos, SourcePos: integer;
  MaxOutPos, MaxInPos, MaxTapPos: integer;
begin
  if Weight.Size <> pOutChannels * pInChannels * pKernelSize then
    raise Exception.Create('LoadTorchConvTranspose1D: weight size does not fit.');
  SetShape(pclTransposed, pOutChannels, pInChannels, pKernelSize);
  MaxOutPos := pOutChannels - 1;
  MaxInPos := pInChannels - 1;
  MaxTapPos := pKernelSize - 1;
  SourcePos := 0;
  for InPos := 0 to MaxInPos do
    for OutPos := 0 to MaxOutPos do
      for TapPos := 0 to MaxTapPos do
      begin
        FRows.FData[(TapPos * pOutChannels + OutPos) * pInChannels + InPos] :=
          Weight.FData[SourcePos];
        Inc(SourcePos);
      end;
  PackRows();
  LoadTorchBias(Bias);
end;

procedure TNNetPlainConvWeights.LoadDepthwiseDeconv1DLayer(Deconv: TNNetLayer;
  pHasBias: boolean);
var
  ChannelPos, MaxChannelPos, TapPos, MaxTapPos: integer;
  NeuronWeights: TNNetVolume;
begin
  if not (Deconv is TNNetDeconvolution) or
     (TNNetDeconvolution(Deconv).Groups <> Deconv.Neurons.Count) or
     (Deconv.Neurons[0].Weights.Depth <> 1) then
    raise Exception.Create('LoadDepthwiseDeconv1DLayer: needs a ' +
      'TNNetDeconv1DLinear with groups = channels, got ' +
      Deconv.ClassName + '.');
  SetShape(pclDepthwiseTransposed, Deconv.Neurons.Count, Deconv.Neurons.Count,
    Deconv.Neurons[0].Weights.SizeX);
  MaxChannelPos := FOutChannels - 1;
  MaxTapPos := FKernelSize - 1;
  for ChannelPos := 0 to MaxChannelPos do
  begin
    NeuronWeights := Deconv.Neurons[ChannelPos].Weights;
    for TapPos := 0 to MaxTapPos do
      FRows.FData[TapPos * FOutChannels + ChannelPos] :=
        NeuronWeights.FData[TapPos];
  end;
  LoadLayerBias(Deconv, pHasBias);
end;

procedure TNNetPlainConvWeights.LoadTorchDepthwiseConvTranspose1D(Weight,
  Bias: TNNetVolume; pChannels, pKernelSize: integer);
var
  ChannelPos, MaxChannelPos, TapPos, MaxTapPos, SourcePos: integer;
begin
  if Weight.Size <> pChannels * pKernelSize then
    raise Exception.Create('LoadTorchDepthwiseConvTranspose1D: weight size ' +
      'does not fit.');
  SetShape(pclDepthwiseTransposed, pChannels, pChannels, pKernelSize);
  MaxChannelPos := pChannels - 1;
  MaxTapPos := pKernelSize - 1;
  SourcePos := 0;
  for ChannelPos := 0 to MaxChannelPos do
    for TapPos := 0 to MaxTapPos do
    begin
      FRows.FData[TapPos * pChannels + ChannelPos] := Weight.FData[SourcePos];
      Inc(SourcePos);
    end;
  LoadTorchBias(Bias);
end;

{ TNNetPlainLSTMWeights }

constructor TNNetPlainLSTMWeights.Create();
begin
  inherited Create();
  FInputRows := TNNetVolume.Create();
  FRecurrentRows := TNNetVolume.Create();
  FBias := TNNetVolume.Create();
end;

destructor TNNetPlainLSTMWeights.Destroy();
begin
  FBias.Free;
  FRecurrentRows.Free;
  FInputRows.Free;
  inherited Destroy();
end;

procedure TNNetPlainLSTMWeights.LoadLSTMCell(Cell: TNNetLayer);
var
  GatePos, InputFloats, RecurrentFloats: integer;
begin
  if not (Cell is TNNetLSTMCell) then
    raise Exception.Create('LoadLSTMCell: needs a TNNetLSTMCell, got ' +
      Cell.ClassName + '.');
  FHiddenSize := Cell.Neurons[8].Weights.Size;
  FInChannels := Cell.Neurons[0].Weights.Depth;
  InputFloats := FHiddenSize * FInChannels;
  RecurrentFloats := FHiddenSize * FHiddenSize;
  FInputRows.ReSize(4 * FHiddenSize, 1, FInChannels);
  FRecurrentRows.ReSize(4 * FHiddenSize, 1, FHiddenSize);
  FBias.ReSize(1, 1, 4 * FHiddenSize);
  for GatePos := 0 to 3 do
  begin
    Move(Cell.Neurons[GatePos].Weights.FData[0],
      FInputRows.FData[GatePos * InputFloats], InputFloats * csNeuralFloatSize);
    Move(Cell.Neurons[4 + GatePos].Weights.FData[0],
      FRecurrentRows.FData[GatePos * RecurrentFloats],
      RecurrentFloats * csNeuralFloatSize);
    Move(Cell.Neurons[8 + GatePos].Weights.FData[0],
      FBias.FData[GatePos * FHiddenSize], FHiddenSize * csNeuralFloatSize);
  end;
end;

procedure TNNetPlainLSTMWeights.LoadTorchLSTM(WeightIH, WeightHH, BiasIH,
  BiasHH: TNNetVolume);
begin
  FHiddenSize := BiasIH.Size div 4;
  if (FHiddenSize < 1) or (BiasHH.Size <> BiasIH.Size) or
     (WeightHH.Size <> 4 * FHiddenSize * FHiddenSize) or
     (WeightIH.Size mod (4 * FHiddenSize) <> 0) then
    raise Exception.Create('LoadTorchLSTM: tensor sizes do not fit one LSTM.');
  FInChannels := WeightIH.Size div (4 * FHiddenSize);
  FInputRows.ReSize(4 * FHiddenSize, 1, FInChannels);
  FRecurrentRows.ReSize(4 * FHiddenSize, 1, FHiddenSize);
  FBias.ReSize(1, 1, 4 * FHiddenSize);
  FInputRows.CopyNoChecks(WeightIH);
  FRecurrentRows.CopyNoChecks(WeightHH);
  FBias.CopyNoChecks(BiasIH);
  FBias.Add(BiasHH);
end;

{ TNNetPlainOps }

constructor TNNetPlainOps.Create(pThreaded: boolean; pTilePositions: integer);
var
  Direction: integer;
begin
  inherited Create();
  FThreaded := pThreaded;
  FTilePositions := Max(pTilePositions, 1);
  FMinPositionsToThread := 2 * FTilePositions;
  if FThreaded then CreateNeuralThreadListIfRequired();
  for Direction := 0 to 1 do
  begin
    FProjection[Direction] := TNNetVolume.Create();
    FGates[Direction] := TNNetVolume.Create();
    FHidden[Direction] := TNNetVolume.Create();
    FCell[Direction] := TNNetVolume.Create();
  end;
  FMean := TNNetVolume.Create();
  FInvStd := TNNetVolume.Create();
  FSum := TNNetVolume.Create();
  FSumSq := TNNetVolume.Create();
  FChannelScale := TNNetVolume.Create();
  FNormShift := TNNetVolume.Create();
  FInvAlpha := TNNetVolume.Create();
  FWindow := TNNetVolume.Create();
  FChunkOutput := TNNetVolume.Create();
end;

destructor TNNetPlainOps.Destroy();
var
  SlotPos, Direction: integer;
begin
  for SlotPos := 0 to High(FPatch) do
  begin
    FPatch[SlotPos].Free;
    FTileOut[SlotPos].Free;
  end;
  for Direction := 0 to 1 do
  begin
    FProjection[Direction].Free;
    FGates[Direction].Free;
    FHidden[Direction].Free;
    FCell[Direction].Free;
  end;
  FMean.Free;
  FInvStd.Free;
  FSum.Free;
  FSumSq.Free;
  FChannelScale.Free;
  FNormShift.Free;
  FInvAlpha.Free;
  FWindow.Free;
  FChunkOutput.Free;
  inherited Destroy();
end;

function TNNetPlainOps.ScratchSlotCount(): integer;
begin
  if FThreaded then
  begin
    // The pool may have been freed or rebuilt since Create.
    CreateNeuralThreadListIfRequired();
    Result := Max(fNTL.Count, 1);
  end
  else Result := 1;
end;

class procedure TNNetPlainOps.GrowScratch(V: TNNetVolume; FloatCount: integer);
begin
  if V.Size < FloatCount then V.ReSize(FloatCount, 1, 1);
end;

class procedure TNNetPlainOps.CheckSequence(V: TNNetVolume; Channels: integer;
  const OpName: string);
begin
  if (V.SizeY <> 1) or (V.SizeX < 1) or (V.Depth <> Channels) then
    raise Exception.Create(OpName + ': input is ' + IntToStr(V.SizeX) + 'x' +
      IntToStr(V.SizeY) + 'x' + IntToStr(V.Depth) + ', expected (T, 1, ' +
      IntToStr(Channels) + ').');
end;

procedure TNNetPlainOps.EnsureTileScratch(PatchFloats, TileOutFloats: integer);
var
  SlotCount, SlotPos, OldCount: integer;
begin
  SlotCount := ScratchSlotCount();
  OldCount := Length(FPatch);
  if OldCount < SlotCount then
  begin
    SetLength(FPatch, SlotCount);
    SetLength(FTileOut, SlotCount);
    for SlotPos := OldCount to SlotCount - 1 do
    begin
      FPatch[SlotPos] := TNNetVolume.Create();
      FTileOut[SlotPos] := TNNetVolume.Create();
    end;
  end;
  for SlotPos := 0 to SlotCount - 1 do
  begin
    GrowScratch(FPatch[SlotPos], PatchFloats);
    GrowScratch(FTileOut[SlotPos], TileOutFloats);
  end;
end;

function TNNetPlainOps.JobThreadCount(): integer;
begin
  Result := ScratchSlotCount();
  if (FMaxThreads > 0) and (FMaxThreads < Result) then Result := FMaxThreads;
end;

procedure TNNetPlainOps.CappedJob(Index, ThreadNum: integer);
begin
  if Index < FJobThreadCount then FCappedJob(Index, FJobThreadCount);
end;

procedure TNNetPlainOps.RunOnPool(Job: TNeuralProc);
begin
  FJobThreadCount := JobThreadCount();
  if FJobThreadCount <= 1 then Job(0, 1)
  else if FJobThreadCount = fNTL.Count then fNTL.StartProc(Job)
  else
  begin
    // The pool wakes every thread; the ones past the cap return at once.
    FCappedJob := Job;
    fNTL.StartProc(@CappedJob);
  end;
end;

procedure TNNetPlainOps.RunJob(Job: TNeuralProc; PositionCount: integer);
begin
  FJobPositionCount := PositionCount;
  if FThreaded and (PositionCount >= FMinPositionsToThread)
    then RunOnPool(Job)
    else Job(0, 1);
end;

procedure TNNetPlainOps.AddBiasToRows(Output: TNNetVolume; Bias: TNNetVolume;
  FirstPos, LastPos: integer);
var
  Position, Channels: integer;
  BiasPtr: TNeuralFloatArrPtr;
begin
  Channels := Output.Depth;
  BiasPtr := Bias.DataPtr;
  for Position := FirstPos to LastPos do
    TNNetVolume.Add(Output.GetRawPtr(Position * Channels), BiasPtr, Channels);
end;

function TNNetPlainOps.JobBiasPtr(): TNeuralFloatArrPtr;
begin
  if FJobConv.HasBias then Result := FJobConv.Bias.DataPtr
  else Result := nil;
end;

procedure TNNetPlainOps.Conv1DJob(Index, ThreadNum: integer);
var
  StartPos, FinishPos, TileStart, TileCount, MaxTilePos: integer;
  OutPos, PatchPos, TapPos, FirstInPos, LastInOffset, FirstTap, LastTap: integer;
  InChannels, OutChannels, KernelSize, Stride, Dilation, InLength: integer;
  TapBytes, PatchRowFloats, InPos: integer;
  Patch, Input: TNNetVolume;
  BiasPtr: TNeuralFloatArrPtr;
begin
  TNeuralThreadList.CalculateWorkingRange(Index, ThreadNum, FJobPositionCount,
    StartPos, FinishPos);
  Patch := FPatch[Index];
  Input := FJobInput;
  BiasPtr := JobBiasPtr();
  InChannels := FJobConv.InChannels;
  OutChannels := FJobConv.OutChannels;
  KernelSize := FJobConv.KernelSize;
  Stride := FJobStride;
  Dilation := FJobDilation;
  InLength := Input.SizeX;
  TapBytes := InChannels * csNeuralFloatSize;
  PatchRowFloats := KernelSize * InChannels;
  TileStart := StartPos;
  while TileStart <= FinishPos do
  begin
    TileCount := Min(FTilePositions, FinishPos - TileStart + 1);
    MaxTilePos := TileStart + TileCount - 1;
    // Tile im2col: row j holds the K taps of output position TileStart + j;
    // taps outside the input are zero.
    PatchPos := 0;
    for OutPos := TileStart to MaxTilePos do
    begin
      FirstInPos := OutPos * Stride - FJobPadding;
      if FirstInPos >= 0 then FirstTap := 0
      else FirstTap := (Dilation - 1 - FirstInPos) div Dilation;
      LastInOffset := InLength - 1 - FirstInPos;
      if LastInOffset < 0 then LastTap := -1
      else LastTap := Min(KernelSize - 1, LastInOffset div Dilation);
      if FirstTap > LastTap then
        FillChar(Patch.FData[PatchPos], PatchRowFloats * csNeuralFloatSize, 0)
      else
      begin
        if FirstTap > 0 then
          FillChar(Patch.FData[PatchPos], FirstTap * TapBytes, 0);
        if Dilation = 1 then
          Move(Input.FData[(FirstInPos + FirstTap) * InChannels],
            Patch.FData[PatchPos + FirstTap * InChannels],
            (LastTap - FirstTap + 1) * TapBytes)
        else
        begin
          InPos := FirstInPos + FirstTap * Dilation;
          for TapPos := FirstTap to LastTap do
          begin
            Move(Input.FData[InPos * InChannels],
              Patch.FData[PatchPos + TapPos * InChannels], TapBytes);
            Inc(InPos, Dilation);
          end;
        end;
        if LastTap < KernelSize - 1 then
          FillChar(Patch.FData[PatchPos + (LastTap + 1) * InChannels],
            (KernelSize - 1 - LastTap) * TapBytes, 0);
      end;
      Inc(PatchPos, PatchRowFloats);
    end;
    TNNetVolume.DotProductsPacked(FJobOutput.GetRawPtr(TileStart * OutChannels),
      OutChannels, FJobConv.Rows.DataPtr, OutChannels, Patch.DataPtr,
      PatchRowFloats, TileCount, PatchRowFloats, BiasPtr);
    Inc(TileStart, TileCount);
  end;
end;

procedure TNNetPlainOps.Conv1D(Input, Output: TNNetVolume;
  Weights: TNNetPlainConvWeights; Stride, Dilation, PaddingLeft,
  PaddingRight: integer);
var
  OutLength: integer;
begin
  if Weights.Layout <> pclConv then
    raise Exception.Create('Conv1D: weights are not in the conv layout.');
  CheckSequence(Input, Weights.InChannels, 'Conv1D');
  if Input = Output then
    raise Exception.Create('Conv1D: Output must not be Input.');
  if (Stride < 1) or (Dilation < 1) or (PaddingLeft < 0) or (PaddingRight < 0) then
    raise Exception.Create('Conv1D: bad stride / dilation / padding.');
  OutLength := (Input.SizeX + PaddingLeft + PaddingRight -
    Dilation * (Weights.KernelSize - 1) - 1) div Stride + 1;
  if OutLength < 1 then
    raise Exception.Create('Conv1D: input of ' + IntToStr(Input.SizeX) +
      ' positions is shorter than the kernel.');
  Output.ReSize(OutLength, 1, Weights.OutChannels);
  EnsureTileScratch(FTilePositions * Weights.KernelSize * Weights.InChannels, 0);
  FJobInput := Input;
  FJobOutput := Output;
  FJobConv := Weights;
  FJobStride := Stride;
  FJobDilation := Dilation;
  FJobPadding := PaddingLeft;
  RunJob(@Conv1DJob, OutLength);
end;

procedure TNNetPlainOps.Conv1D(Input, Output: TNNetVolume;
  Weights: TNNetPlainConvWeights; Stride, Dilation, Padding: integer);
begin
  Conv1D(Input, Output, Weights, Stride, Dilation, Padding, Padding);
end;

procedure TNNetPlainOps.PointwiseConvJob(Index, ThreadNum: integer);
var
  StartPos, FinishPos, OutChannels, InChannels: integer;
begin
  TNeuralThreadList.CalculateWorkingRange(Index, ThreadNum, FJobPositionCount,
    StartPos, FinishPos);
  if StartPos > FinishPos then exit;
  OutChannels := FJobConv.OutChannels;
  InChannels := FJobConv.InChannels;
  TNNetVolume.DotProductsPacked(FJobOutput.GetRawPtr(StartPos * OutChannels),
    OutChannels, FJobConv.Rows.DataPtr, OutChannels,
    FJobInput.GetRawPtr(StartPos * InChannels), InChannels,
    FinishPos - StartPos + 1, InChannels, JobBiasPtr());
end;

procedure TNNetPlainOps.PointwiseConv(Input, Output: TNNetVolume;
  Weights: TNNetPlainConvWeights);
begin
  if (Weights.Layout <> pclConv) or (Weights.KernelSize <> 1) then
    raise Exception.Create('PointwiseConv: needs conv weights with K = 1.');
  CheckSequence(Input, Weights.InChannels, 'PointwiseConv');
  if Input = Output then
    raise Exception.Create('PointwiseConv: Output must not be Input.');
  Output.ReSize(Input.SizeX, 1, Weights.OutChannels);
  FJobInput := Input;
  FJobOutput := Output;
  FJobConv := Weights;
  RunJob(@PointwiseConvJob, Input.SizeX);
end;

procedure TNNetPlainOps.ConvTranspose1DJob(Index, ThreadNum: integer);
var
  FirstOutPos, LastOutPos, FirstInPos, LastInPos, TileStart, TileCount: integer;
  InChannels, OutChannels, KernelSize, ProductCount, Stride, LowestInput: integer;
  TilePos, MaxTilePos, FirstOut, FirstTap, LastTap, TapPos: integer;
  ProductPos, OutPos: integer;
  TileOut, Input, Output: TNNetVolume;
begin
  TNeuralThreadList.CalculateWorkingRange(Index, ThreadNum, FJobPositionCount,
    FirstOutPos, LastOutPos);
  if FirstOutPos > LastOutPos then exit;
  TileOut := FTileOut[Index];
  Input := FJobInput;
  Output := FJobOutput;
  InChannels := FJobConv.InChannels;
  OutChannels := FJobConv.OutChannels;
  KernelSize := FJobConv.KernelSize;
  ProductCount := KernelSize * OutChannels;
  Stride := FJobStride;
  FillChar(Output.FData[FirstOutPos * OutChannels],
    (LastOutPos - FirstOutPos + 1) * OutChannels * csNeuralFloatSize, 0);
  // Input positions with a tap landing in [FirstOutPos, LastOutPos].
  LowestInput := FirstOutPos + FJobPadding - KernelSize + 1;
  if LowestInput <= 0 then FirstInPos := 0
  else FirstInPos := (LowestInput + Stride - 1) div Stride;
  LastInPos := Min(Input.SizeX - 1, (LastOutPos + FJobPadding) div Stride);
  TileStart := FirstInPos;
  while TileStart <= LastInPos do
  begin
    TileCount := Min(FTilePositions, LastInPos - TileStart + 1);
    MaxTilePos := TileCount - 1;
    // Product (j, k, oc) at j*K*Cout + k*Cout + oc: one contiguous Cout run per tap.
    TNNetVolume.DotProductsPacked(TileOut.DataPtr, ProductCount,
      FJobConv.Rows.DataPtr, ProductCount, Input.GetRawPtr(TileStart * InChannels),
      InChannels, TileCount, InChannels, nil);
    // Scatter in ascending input position, as the TNNetDeconvolution forwards.
    for TilePos := 0 to MaxTilePos do
    begin
      FirstOut := (TileStart + TilePos) * Stride - FJobPadding;
      FirstTap := Max(0, FirstOutPos - FirstOut);
      LastTap := Min(KernelSize - 1, LastOutPos - FirstOut);
      ProductPos := TilePos * ProductCount + FirstTap * OutChannels;
      OutPos := (FirstOut + FirstTap) * OutChannels;
      for TapPos := FirstTap to LastTap do
      begin
        TNNetVolume.Add(Output.GetRawPtr(OutPos), TileOut.GetRawPtr(ProductPos),
          OutChannels);
        Inc(ProductPos, OutChannels);
        Inc(OutPos, OutChannels);
      end;
    end;
    Inc(TileStart, TileCount);
  end;
  if FJobConv.HasBias then
    AddBiasToRows(Output, FJobConv.Bias, FirstOutPos, LastOutPos);
end;

procedure TNNetPlainOps.ConvTranspose1D(Input, Output: TNNetVolume;
  Weights: TNNetPlainConvWeights; Stride, Padding, OutputPadding: integer);
var
  OutLength: integer;
begin
  if Weights.Layout <> pclTransposed then
    raise Exception.Create('ConvTranspose1D: weights are not in the ' +
      'transposed layout.');
  CheckSequence(Input, Weights.InChannels, 'ConvTranspose1D');
  if Input = Output then
    raise Exception.Create('ConvTranspose1D: Output must not be Input.');
  if (Stride < 1) or (Padding < 0) or (OutputPadding < 0) then
    raise Exception.Create('ConvTranspose1D: bad stride / padding.');
  OutLength := (Input.SizeX - 1) * Stride - 2 * Padding + Weights.KernelSize +
    OutputPadding;
  if OutLength < 1 then
    raise Exception.Create('ConvTranspose1D: empty output.');
  Output.ReSize(OutLength, 1, Weights.OutChannels);
  EnsureTileScratch(0, FTilePositions * Weights.KernelSize * Weights.OutChannels);
  FJobInput := Input;
  FJobOutput := Output;
  FJobConv := Weights;
  FJobStride := Stride;
  FJobPadding := Padding;
  RunJob(@ConvTranspose1DJob, OutLength);
end;

procedure TNNetPlainOps.DepthwiseConvTranspose1DJob(Index, ThreadNum: integer);
var
  FirstOutPos, LastOutPos, OutPos, Channels, KernelSize, Stride: integer;
  InLength, Residue, TopTap, TapPos, InPos: integer;
  OutPtr, BiasPtr: TNeuralFloatArrPtr;
  HasBias: boolean;
begin
  TNeuralThreadList.CalculateWorkingRange(Index, ThreadNum, FJobPositionCount,
    FirstOutPos, LastOutPos);
  if FirstOutPos > LastOutPos then exit;
  Channels := FJobConv.OutChannels;
  KernelSize := FJobConv.KernelSize;
  Stride := FJobStride;
  InLength := FJobInput.SizeX;
  HasBias := FJobConv.HasBias;
  BiasPtr := FJobConv.Bias.DataPtr;
  for OutPos := FirstOutPos to LastOutPos do
  begin
    OutPtr := FJobOutput.GetRawPtr(OutPos * Channels);
    FillChar(OutPtr^, Channels * csNeuralFloatSize, 0);
    // Taps k = Residue + n*Stride land here from input (OutPos + P - k) / S;
    // descending k is ascending input position, the layer's summation order.
    Residue := (OutPos + FJobPadding) mod Stride;
    if Residue < KernelSize then
    begin
      TopTap := Residue + ((KernelSize - 1 - Residue) div Stride) * Stride;
      TapPos := TopTap;
      while TapPos >= Residue do
      begin
        InPos := (OutPos + FJobPadding - TapPos) div Stride;
        if (InPos >= 0) and (InPos < InLength) then
          TNNetVolume.MulAdd(OutPtr, FJobInput.GetRawPtr(InPos * Channels),
            FJobConv.Rows.GetRawPtr(TapPos * Channels), Channels);
        Dec(TapPos, Stride);
      end;
    end;
    if HasBias then TNNetVolume.Add(OutPtr, BiasPtr, Channels);
  end;
end;

procedure TNNetPlainOps.DepthwiseConvTranspose1D(Input, Output: TNNetVolume;
  Weights: TNNetPlainConvWeights; Stride, Padding, OutputPadding: integer);
var
  OutLength: integer;
begin
  if Weights.Layout <> pclDepthwiseTransposed then
    raise Exception.Create('DepthwiseConvTranspose1D: weights are not in ' +
      'the depthwise transposed layout.');
  CheckSequence(Input, Weights.OutChannels, 'DepthwiseConvTranspose1D');
  if Input = Output then
    raise Exception.Create('DepthwiseConvTranspose1D: Output must not be Input.');
  if (Stride < 1) or (Padding < 0) or (OutputPadding < 0) then
    raise Exception.Create('DepthwiseConvTranspose1D: bad stride / padding.');
  OutLength := (Input.SizeX - 1) * Stride - 2 * Padding + Weights.KernelSize +
    OutputPadding;
  if OutLength < 1 then
    raise Exception.Create('DepthwiseConvTranspose1D: empty output.');
  Output.ReSize(OutLength, 1, Weights.OutChannels);
  FJobInput := Input;
  FJobOutput := Output;
  FJobConv := Weights;
  FJobStride := Stride;
  FJobPadding := Padding;
  RunJob(@DepthwiseConvTranspose1DJob, OutLength);
end;

procedure TNNetPlainOps.UpsampleNearest(Input, Output: TNNetVolume;
  Factor: integer);
var
  InPos, MaxInPos, RepeatPos, MaxRepeatPos, OutPos, Channels, RowBytes: integer;
begin
  CheckSequence(Input, Input.Depth, 'UpsampleNearest');
  if (Factor < 1) or (Input = Output) then
    raise Exception.Create('UpsampleNearest: bad factor or Output = Input.');
  Channels := Input.Depth;
  Output.ReSize(Input.SizeX * Factor, 1, Channels);
  RowBytes := Channels * csNeuralFloatSize;
  MaxInPos := Input.SizeX - 1;
  MaxRepeatPos := Factor - 1;
  OutPos := 0;
  for InPos := 0 to MaxInPos do
    for RepeatPos := 0 to MaxRepeatPos do
    begin
      Move(Input.FData[InPos * Channels], Output.FData[OutPos], RowBytes);
      Inc(OutPos, Channels);
    end;
end;

procedure TNNetPlainOps.ReflectPad(Input, Output: TNNetVolume;
  PaddingLeft, PaddingRight: integer);
var
  InLength, OutPos, MaxOutPos, SourcePos, Channels, RowBytes: integer;
begin
  CheckSequence(Input, Input.Depth, 'ReflectPad');
  InLength := Input.SizeX;
  if (PaddingLeft < 0) or (PaddingRight < 0) or (PaddingLeft >= InLength) or
     (PaddingRight >= InLength) or (Input = Output) then
    raise Exception.Create('ReflectPad: pads must be in [0, ' +
      IntToStr(InLength - 1) + '] and Output must not be Input.');
  Channels := Input.Depth;
  Output.ReSize(InLength + PaddingLeft + PaddingRight, 1, Channels);
  RowBytes := Channels * csNeuralFloatSize;
  MaxOutPos := Output.SizeX - 1;
  for OutPos := 0 to MaxOutPos do
  begin
    SourcePos := Abs(OutPos - PaddingLeft);
    if SourcePos >= InLength then SourcePos := 2 * (InLength - 1) - SourcePos;
    Move(Input.FData[SourcePos * Channels], Output.FData[OutPos * Channels],
      RowBytes);
  end;
end;

procedure TNNetPlainOps.InstanceNorm(V, Scale, Shift: TNNetVolume;
  Eps: TNeuralFloat);
begin
  PrepareInstanceNorm(V, Scale, Shift, Eps);
  ApplyInstanceNorm(V);
end;

procedure TNNetPlainOps.PrepareInstanceNorm(V, Scale, Shift: TNNetVolume;
  Eps: TNeuralFloat);
var
  Channels, ChannelPos, MaxChannelPos: integer;
begin
  Channels := V.Depth;
  if (Scale.Size <> Channels) or (Shift.Size <> Channels) then
    raise Exception.Create('InstanceNorm: Scale / Shift need ' +
      IntToStr(Channels) + ' elements.');
  NeuralPerChannelMeanInvStd(V, FMean, FInvStd, FSum, FSumSq, Eps);
  // inv_std and the affine scale fold into one per-channel multiplier.
  FChannelScale.ReSize(1, 1, Channels);
  MaxChannelPos := Channels - 1;
  for ChannelPos := 0 to MaxChannelPos do
    FChannelScale.FData[ChannelPos] :=
      FInvStd.FData[ChannelPos] * Scale.FData[ChannelPos];
  FNormShift.ReSize(1, 1, Channels);
  FNormShift.CopyNoChecks(Shift);
end;

procedure TNNetPlainOps.ApplyInstanceNorm(V: TNNetVolume);
var
  Channels, Position, MaxPosition: integer;
  RowPtr, MeanPtr, ScalePtr, ShiftPtr: TNeuralFloatArrPtr;
begin
  Channels := V.Depth;
  if FChannelScale.Size <> Channels then
    raise Exception.Create('ApplyInstanceNorm: prepared for ' +
      IntToStr(FChannelScale.Size) + ' channels, got ' +
      IntToStr(Channels) + '.');
  MeanPtr := FMean.DataPtr;
  ScalePtr := FChannelScale.DataPtr;
  ShiftPtr := FNormShift.DataPtr;
  MaxPosition := V.SizeX * V.SizeY - 1;
  for Position := 0 to MaxPosition do
  begin
    RowPtr := V.GetRawPtr(Position * Channels);
    TNNetVolume.MulAdd(RowPtr, MeanPtr, -1.0, Channels);
    TNNetVolume.Mul(RowPtr, ScalePtr, Channels);
    TNNetVolume.Add(RowPtr, ShiftPtr, Channels);
  end;
end;

procedure TNNetPlainOps.SnakeJob(Index, ThreadNum: integer);
var
  StartPos, FinishPos, Position, Channels: integer;
  RowPtr, ScratchPtr, AlphaPtr, InvAlphaPtr: TNeuralFloatArrPtr;
begin
  TNeuralThreadList.CalculateWorkingRange(Index, ThreadNum, FJobPositionCount,
    StartPos, FinishPos);
  Channels := FJobOutput.Depth;
  ScratchPtr := FPatch[Index].DataPtr;
  AlphaPtr := FJobAlpha.DataPtr;
  InvAlphaPtr := FInvAlpha.DataPtr;
  // The TNNetSnake.ComputePerChannel inference sequence, one row at a time.
  for Position := StartPos to FinishPos do
  begin
    RowPtr := FJobOutput.GetRawPtr(Position * Channels);
    Move(RowPtr^, ScratchPtr^, Channels * csNeuralFloatSize);
    TNNetVolume.Mul(ScratchPtr, AlphaPtr, Channels);
    TNNetVolume.Sin(ScratchPtr, ScratchPtr, Channels);
    TNNetVolume.Mul(ScratchPtr, ScratchPtr, Channels);
    TNNetVolume.Mul(ScratchPtr, InvAlphaPtr, Channels);
    TNNetVolume.Add(RowPtr, ScratchPtr, Channels);
  end;
end;

procedure TNNetPlainOps.Snake(V, Alpha: TNNetVolume);
var
  Channels, ChannelPos, MaxChannelPos: integer;
begin
  Channels := V.Depth;
  if Alpha.Size <> Channels then
    raise Exception.Create('Snake: Alpha needs ' + IntToStr(Channels) +
      ' elements.');
  FInvAlpha.ReSize(1, 1, Channels);
  MaxChannelPos := Channels - 1;
  for ChannelPos := 0 to MaxChannelPos do
    FInvAlpha.FData[ChannelPos] := 1 / Alpha.FData[ChannelPos];
  EnsureTileScratch(Channels, 0);
  FJobOutput := V;
  FJobAlpha := Alpha;
  RunJob(@SnakeJob, V.SizeX * V.SizeY);
end;

procedure TNNetPlainOps.NormSnakeConv1D(Input, Output: TNNetVolume;
  Weights: TNNetPlainConvWeights; Alpha: TNNetVolume; Dilation: integer;
  Accumulate: boolean; ChunkPositions: integer);
var
  InLength, InChannels, OutChannels, Padding, ChunkCount, WindowRows: integer;
  ChunkPos, MaxChunkPos, ChunkStart, FirstInPos, FirstValidPos, LastValidPos: integer;
  LeadingRows, TrailingRows, FirstWritePos, WriteRows, WrittenEnd: integer;
  RowBytes: integer;

  procedure ZeroPaddingRows();
  begin
    if LeadingRows > 0 then
      FillChar(FWindow.FData[0], LeadingRows * RowBytes, 0);
    if TrailingRows > 0 then
      FillChar(FWindow.FData[(WindowRows - TrailingRows) * InChannels],
        TrailingRows * RowBytes, 0);
  end;

begin
  if Weights.Layout <> pclConv then
    raise Exception.Create('NormSnakeConv1D: weights are not in the conv layout.');
  CheckSequence(Input, Weights.InChannels, 'NormSnakeConv1D');
  if (Input = Output) or (Dilation < 1) or (ChunkPositions < 1) or
     not Odd(Weights.KernelSize) then
    raise Exception.Create('NormSnakeConv1D: Output = Input, bad dilation or ' +
      'chunk, or an even kernel.');
  InLength := Input.SizeX;
  InChannels := Input.Depth;
  OutChannels := Weights.OutChannels;
  Padding := Dilation * (Weights.KernelSize - 1) div 2;
  if Accumulate then
  begin
    if (Output.SizeX <> InLength) or (Output.SizeY <> 1) or
       (Output.Depth <> OutChannels) then
      raise Exception.Create('NormSnakeConv1D: Output to accumulate into is ' +
        IntToStr(Output.SizeX) + 'x' + IntToStr(Output.SizeY) + 'x' +
        IntToStr(Output.Depth) + ', expected (' + IntToStr(InLength) +
        ', 1, ' + IntToStr(OutChannels) + ').');
  end
  else Output.ReSize(InLength, 1, OutChannels);
  // Equal chunks, the last one moved back to end at InLength (its first rows
  // repeat the previous chunk and are not written), keep the window size fixed.
  ChunkCount := (InLength + ChunkPositions - 1) div ChunkPositions;
  ChunkPositions := (InLength + ChunkCount - 1) div ChunkCount;
  WindowRows := ChunkPositions + 2 * Padding;
  FWindow.ReSize(WindowRows, 1, InChannels);
  RowBytes := InChannels * csNeuralFloatSize;
  WrittenEnd := 0;
  MaxChunkPos := ChunkCount - 1;
  for ChunkPos := 0 to MaxChunkPos do
  begin
    ChunkStart := Min(ChunkPos * ChunkPositions, InLength - ChunkPositions);
    FirstInPos := ChunkStart - Padding;
    FirstValidPos := Max(0, FirstInPos);
    LastValidPos := Min(InLength - 1, FirstInPos + WindowRows - 1);
    LeadingRows := FirstValidPos - FirstInPos;
    TrailingRows := FirstInPos + WindowRows - 1 - LastValidPos;
    Move(Input.FData[FirstValidPos * InChannels],
      FWindow.FData[LeadingRows * InChannels],
      (LastValidPos - FirstValidPos + 1) * RowBytes);
    // The padding rows are zeroed before the transform too, so it never
    // reads stale values.
    ZeroPaddingRows();
    ApplyInstanceNorm(FWindow);
    Snake(FWindow, Alpha);
    // torch pads the transformed input with zeros.
    ZeroPaddingRows();
    Conv1D(FWindow, FChunkOutput, Weights, 1, Dilation, 0);
    FirstWritePos := Max(ChunkStart, WrittenEnd);
    WriteRows := ChunkStart + ChunkPositions - FirstWritePos;
    if Accumulate then
      TNNetVolume.Add(Output.GetRawPtr(FirstWritePos * OutChannels),
        FChunkOutput.GetRawPtr((FirstWritePos - ChunkStart) * OutChannels),
        WriteRows * OutChannels)
    else
      Move(FChunkOutput.FData[(FirstWritePos - ChunkStart) * OutChannels],
        Output.FData[FirstWritePos * OutChannels],
        WriteRows * OutChannels * csNeuralFloatSize);
    WrittenEnd := ChunkStart + ChunkPositions;
  end;
end;

procedure TNNetPlainOps.LeakyReLU(Input, Output: TNNetVolume;
  Slope: TNeuralFloat);
begin
  Output.ReSize(Input);
  TNNetVolume.LeakyRelu(Output.DataPtr, Input.DataPtr, Slope, Input.Size);
end;

procedure TNNetPlainOps.ConcatChannels(const Sources: array of TNNetVolume;
  Output: TNNetVolume);
var
  SourcePos, Position, MaxPosition, PositionCount: integer;
  OutChannels, SourceChannels, ChannelOffset, RowBytes: integer;
  Source: TNNetVolume;
begin
  if Length(Sources) = 0 then
    raise Exception.Create('ConcatChannels: no sources.');
  PositionCount := Sources[0].SizeX;
  OutChannels := 0;
  for SourcePos := 0 to High(Sources) do
  begin
    Source := Sources[SourcePos];
    CheckSequence(Source, Source.Depth, 'ConcatChannels');
    if (Source.SizeX <> PositionCount) or (Source = Output) then
      raise Exception.Create('ConcatChannels: source ' + IntToStr(SourcePos) +
        ' has ' + IntToStr(Source.SizeX) + ' positions (expected ' +
        IntToStr(PositionCount) + ') or is the output.');
    Inc(OutChannels, Source.Depth);
  end;
  Output.ReSize(PositionCount, 1, OutChannels);
  MaxPosition := PositionCount - 1;
  ChannelOffset := 0;
  for SourcePos := 0 to High(Sources) do
  begin
    Source := Sources[SourcePos];
    SourceChannels := Source.Depth;
    RowBytes := SourceChannels * csNeuralFloatSize;
    for Position := 0 to MaxPosition do
      Move(Source.FData[Position * SourceChannels],
        Output.FData[Position * OutChannels + ChannelOffset], RowBytes);
    Inc(ChannelOffset, SourceChannels);
  end;
end;

procedure TNNetPlainOps.LSTMProjectionJob(Index, ThreadNum: integer);
var
  StartPos, FinishPos, Direction, GateCount: integer;
  Weights: TNNetPlainLSTMWeights;
  Position: integer;
  BiasPtr: TNeuralFloatArrPtr;
begin
  TNeuralThreadList.CalculateWorkingRange(Index, ThreadNum, FJobPositionCount,
    StartPos, FinishPos);
  if StartPos > FinishPos then exit;
  for Direction := 0 to 1 do
  begin
    Weights := FJobLSTM[Direction];
    GateCount := 4 * Weights.HiddenSize;
    FProjection[Direction].DotProductsTiled(GateCount, StartPos, FinishPos,
      Weights.InChannels, Weights.InputRows, FJobInput,
      Min(csPlainOpsTileRows, GateCount), FTilePositions);
    BiasPtr := Weights.Bias.DataPtr;
    for Position := StartPos to FinishPos do
      TNNetVolume.Add(FProjection[Direction].GetRawPtr(Position * GateCount),
        BiasPtr, GateCount);
  end;
end;

procedure TNNetPlainOps.RunLSTMDirection(Direction: integer);
var
  Weights: TNNetPlainLSTMWeights;
  Hidden, GateCount, OutChannels, OutOffset: integer;
  StepPos, MaxStepPos, Position: integer;
  Gates, HiddenState: TNNetVolume;
  GatesPtr, InputGatePtr, ForgetGatePtr, CellGatePtr, OutputGatePtr: TNeuralFloatArrPtr;
  HiddenPtr, CellPtr: TNeuralFloatArrPtr;
  IsBackward: boolean;
begin
  Weights := FJobLSTM[Direction];
  Hidden := Weights.HiddenSize;
  GateCount := 4 * Hidden;
  OutChannels := FJobOutput.Depth;
  OutOffset := Direction * Hidden;
  IsBackward := Direction = 1;
  Gates := FGates[Direction];
  HiddenState := FHidden[Direction];
  GatesPtr := Gates.DataPtr;
  InputGatePtr := GatesPtr;
  ForgetGatePtr := Gates.GetRawPtr(Hidden);
  CellGatePtr := Gates.GetRawPtr(2 * Hidden);
  OutputGatePtr := Gates.GetRawPtr(3 * Hidden);
  HiddenPtr := HiddenState.DataPtr;
  CellPtr := FCell[Direction].DataPtr;
  HiddenState.Fill(0);
  FCell[Direction].Fill(0);
  MaxStepPos := FJobPositionCount - 1;
  for StepPos := 0 to MaxStepPos do
  begin
    if IsBackward then Position := MaxStepPos - StepPos else Position := StepPos;
    // Gates read the h of the previous step; h is rewritten only below.
    Gates.DotProducts(GateCount, 0, 0, Hidden, Weights.RecurrentRows,
      HiddenState);
    TNNetVolume.Add(GatesPtr,
      FProjection[Direction].GetRawPtr(Position * GateCount), GateCount);
    TNNetVolume.Sigmoid(InputGatePtr, InputGatePtr, 2 * Hidden);
    TNNetVolume.Tanh(CellGatePtr, CellGatePtr, Hidden);
    TNNetVolume.Sigmoid(OutputGatePtr, OutputGatePtr, Hidden);
    // c = f * c + i * g; h = o * tanh(c).
    TNNetVolume.Mul(CellPtr, ForgetGatePtr, Hidden);
    TNNetVolume.MulAdd(CellPtr, InputGatePtr, CellGatePtr, Hidden);
    TNNetVolume.Tanh(HiddenPtr, CellPtr, Hidden);
    TNNetVolume.Mul(HiddenPtr, OutputGatePtr, Hidden);
    Move(HiddenPtr^, FJobOutput.FData[Position * OutChannels + OutOffset],
      Hidden * csNeuralFloatSize);
  end;
end;

procedure TNNetPlainOps.LSTMRecurrenceJob(Index, ThreadNum: integer);
var
  Direction: integer;
begin
  Direction := Index;
  while Direction <= 1 do
  begin
    RunLSTMDirection(Direction);
    Inc(Direction, ThreadNum);
  end;
end;

procedure TNNetPlainOps.BiLSTM(Input, Output: TNNetVolume;
  ForwardWeights, BackwardWeights: TNNetPlainLSTMWeights);
var
  Hidden, StepCount, Direction: integer;
begin
  Hidden := ForwardWeights.HiddenSize;
  if (BackwardWeights.HiddenSize <> Hidden) or
     (BackwardWeights.InChannels <> ForwardWeights.InChannels) then
    raise Exception.Create('BiLSTM: the two directions differ in shape.');
  CheckSequence(Input, ForwardWeights.InChannels, 'BiLSTM');
  if Input = Output then
    raise Exception.Create('BiLSTM: Output must not be Input.');
  StepCount := Input.SizeX;
  Output.ReSize(StepCount, 1, 2 * Hidden);
  for Direction := 0 to 1 do
  begin
    GrowScratch(FProjection[Direction], StepCount * 4 * Hidden);
    FGates[Direction].ReSize(1, 1, 4 * Hidden);
    FHidden[Direction].ReSize(1, 1, Hidden);
    FCell[Direction].ReSize(1, 1, Hidden);
  end;
  FJobInput := Input;
  FJobOutput := Output;
  FJobLSTM[0] := ForwardWeights;
  FJobLSTM[1] := BackwardWeights;
  RunJob(@LSTMProjectionJob, StepCount);
  // One pool thread per direction; the steps of a direction are sequential.
  FJobPositionCount := StepCount;
  if FThreaded then RunOnPool(@LSTMRecurrenceJob)
  else LSTMRecurrenceJob(0, 1);
end;

function TNNetPlainOps.ScratchBytes(): int64;
var
  SlotPos, Direction: integer;
begin
  Result := int64(FMean.Size) + FInvStd.Size + FSum.Size + FSumSq.Size +
    FChannelScale.Size + FNormShift.Size + FInvAlpha.Size + FWindow.Size +
    FChunkOutput.Size;
  for SlotPos := 0 to High(FPatch) do
    Result := Result + FPatch[SlotPos].Size + FTileOut[SlotPos].Size;
  for Direction := 0 to 1 do
    Result := Result + FProjection[Direction].Size + FGates[Direction].Size +
      FHidden[Direction].Size + FCell[Direction].Size;
  Result := Result * csNeuralFloatSize;
end;

end.
