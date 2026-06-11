unit neuralpretrained;
// Pretrained-checkpoint importers: builds CAI TNNet networks from
// HuggingFace safetensors checkpoints and loads the real weights into them.
//
// Currently implemented: GPT-2 (the openai-community/gpt2 family). The
// architecture is rebuilt from existing CAI layers plus the import-oriented
// TNNetTokenLayerNorm / TNNetLearnedPositionalEmbedding layers:
//
//   Input(SeqLen tokens) -> TNNetEmbedding (wte)
//     -> TNNetLearnedPositionalEmbedding (wpe)
//     -> n_layer x [ pre-LN attention residual + pre-LN GELU-MLP residual ]
//     -> TNNetTokenLayerNorm (ln_f)
//     -> TNNetPointwiseConvLinear(vocab) LM head (wte^T COPIED, see below)
//
// Per block (HF GPT2Block, pre-LN):
//   x := x + c_proj( MHA( c_attn( ln_1(x) ) ) )           [c_attn: d -> 3d]
//   x := x + mlp.c_proj( gelu_new( mlp.c_fc( ln_2(x) ) ) ) [c_fc: d -> 4d]
// where MHA is TNNet.AddMultiHeadSelfAttention (causal): the fused Q|K|V
// slab is split per head exactly like HF does (head h of Q = channels
// h*d_head..(h+1)*d_head-1 of the Q third, same for K and V), each head is
// a TNNetScaledDotProductAttention (softmax(QK^T/sqrt(d_head)) with causal
// mask), heads are depth-concatenated and out-projected by the builder's
// TNNetPointwiseConvLinear(d) which receives c_proj.
// TNNetGELU implements the tanh approximation = HF's gelu_new. Layer
// normalization uses TNNetTokenLayerNorm (per-token over d_model with
// learnable gain+bias, eps=1e-5) - NOT TNNetLayerNorm, which normalizes
// over the whole (SeqLen,1,d) volume.
//
// HF GPT-2 stores its dense weights in Conv1D layout [in, out] (TRANSPOSED
// relative to nn.Linear); this importer transposes them into the
// TNNetPointwiseConvLinear convention (neuron j holds the weight column of
// output channel j).
//
// WEIGHT TYING NOTE: GPT-2 ties the LM head to wte (logits = h . wte^T).
// CAI has no shared-storage tying layer yet, so this importer COPIES wte
// into the LM head's weights (duplicating vocab*d_model floats). A future
// weight-tying layer can remove the duplication; for pure inference the
// copy is exact.
//
// The importer FAILS LOUDLY (EPretrainedImportError) on missing tensors,
// unexpected tensors and shape mismatches.
//
// This unit is coded by Claude (AI).
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralvolume, neuralnetwork, neuralsafetensors;

type
  EPretrainedImportError = class(Exception);

  TGPT2Config = record
    NLayers: integer;    // number of transformer blocks (n_layer)
    NHeads: integer;     // attention heads per block (n_head)
    NEmbd: integer;      // model width d_model (n_embd)
    NCtx: integer;       // maximum context length (n_positions / n_ctx)
    VocabSize: integer;  // token vocabulary size
    Prefix: string;      // tensor-name prefix ('' or 'transformer.')
  end;

// Infers the GPT-2 config from the tensor shapes of an open safetensors
// reader (GPT-2 checkpoints carry no config JSON inside the safetensors):
// vocab/d_model from wte [vocab, d], n_ctx from wpe [n_ctx, d], n_layer by
// counting h.N blocks. n_head is NOT recoverable from shapes; pass it in
// pNumHeads, or pass 0 to apply the GPT-2 family rule n_head = n_embd/64
// (gpt2 768->12, medium 1024->16, large 1280->20, xl 1600->25) - an error
// is raised if n_embd is not divisible by 64 in that case.
function ReadGPT2Config(Reader: TNNetSafeTensorsReader;
  pNumHeads: integer = 0): TGPT2Config;

function GPT2ConfigToString(const Config: TGPT2Config): string;

// Builds a TNNet with the GPT-2 architecture described by the checkpoint at
// FileName and loads every weight. The returned net takes a (SeqLen,1,1)
// volume of token ids (as floats) and outputs (SeqLen,1,vocab) logits, one
// row per input position. pSeqLen = 0 uses the full n_ctx context;
// pNumHeads = 0 applies the n_embd/64 rule (see ReadGPT2Config).
function BuildGPT2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pNumHeads: integer = 0): TNNet;

// Same, also returning the inferred config.
function BuildGPT2FromSafeTensorsEx(const FileName: string;
  out Config: TGPT2Config; pSeqLen: integer = 0;
  pNumHeads: integer = 0): TNNet;

implementation

procedure ImportError(const Msg: string);
begin
  raise EPretrainedImportError.Create(Msg);
end;

function ReadGPT2Config(Reader: TNNetSafeTensorsReader;
  pNumHeads: integer = 0): TGPT2Config;
var
  BlockCnt: integer;
begin
  // The original openai-community/gpt2 safetensors stores tensors without a
  // prefix ("wte.weight"); checkpoints exported from GPT2LMHeadModel carry
  // a "transformer." prefix. Support both.
  if Reader.HasTensor('wte.weight') then Result.Prefix := ''
  else if Reader.HasTensor('transformer.wte.weight') then
    Result.Prefix := 'transformer.'
  else
    ImportError('GPT-2 import: neither "wte.weight" nor ' +
      '"transformer.wte.weight" found in ' + Reader.FileName +
      ' - not a GPT-2 checkpoint?');
  if Reader.DimCount(Result.Prefix + 'wte.weight') <> 2 then
    ImportError('GPT-2 import: wte.weight must be 2-D, got shape ' +
      Reader.ShapeAsString(Result.Prefix + 'wte.weight'));
  Result.VocabSize := Reader.DimSize(Result.Prefix + 'wte.weight', 0);
  Result.NEmbd := Reader.DimSize(Result.Prefix + 'wte.weight', 1);
  if not Reader.HasTensor(Result.Prefix + 'wpe.weight') then
    ImportError('GPT-2 import: missing tensor "' + Result.Prefix +
      'wpe.weight" in ' + Reader.FileName);
  Result.NCtx := Reader.DimSize(Result.Prefix + 'wpe.weight', 0);
  if Reader.DimSize(Result.Prefix + 'wpe.weight', 1) <> Result.NEmbd then
    ImportError('GPT-2 import: wpe.weight width ' +
      IntToStr(Reader.DimSize(Result.Prefix + 'wpe.weight', 1)) +
      ' does not match wte.weight width ' + IntToStr(Result.NEmbd));
  BlockCnt := 0;
  while Reader.HasTensor(Result.Prefix + 'h.' + IntToStr(BlockCnt) +
    '.ln_1.weight') do Inc(BlockCnt);
  if BlockCnt = 0 then
    ImportError('GPT-2 import: no transformer blocks found (missing "' +
      Result.Prefix + 'h.0.ln_1.weight") in ' + Reader.FileName);
  Result.NLayers := BlockCnt;
  if pNumHeads > 0 then
    Result.NHeads := pNumHeads
  else
  begin
    if (Result.NEmbd mod 64) <> 0 then
      ImportError('GPT-2 import: cannot infer n_head (n_embd=' +
        IntToStr(Result.NEmbd) + ' is not divisible by 64). The head count ' +
        'is not stored in the checkpoint - pass pNumHeads explicitly.');
    Result.NHeads := Result.NEmbd div 64;
  end;
  if (Result.NEmbd mod Result.NHeads) <> 0 then
    ImportError('GPT-2 import: n_embd=' + IntToStr(Result.NEmbd) +
      ' is not divisible by n_head=' + IntToStr(Result.NHeads) + '.');
end;

function GPT2ConfigToString(const Config: TGPT2Config): string;
begin
  Result := 'GPT-2 config: n_layer=' + IntToStr(Config.NLayers) +
    ', n_head=' + IntToStr(Config.NHeads) +
    ', n_embd=' + IntToStr(Config.NEmbd) +
    ', n_ctx=' + IntToStr(Config.NCtx) +
    ', vocab=' + IntToStr(Config.VocabSize);
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

type
  TGPT2BlockLayers = record
    LN1, CAttn, AttnProj, LN2, CFc, MlpProj: TNNetLayer;
  end;

// Loads a HF LayerNorm weight/bias pair into a TNNetTokenLayerNorm.
procedure LoadLayerNormWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName, BName: string; d_model: integer);
var
  Tmp: TNNetVolume;
  i: integer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('GPT-2 import: missing tensor "' + WName + '".');
  if not Reader.HasTensor(BName) then
    ImportError('GPT-2 import: missing tensor "' + BName + '".');
  if (Reader.DimCount(WName) <> 1) or (Reader.DimSize(WName, 0) <> d_model) then
    ImportError('GPT-2 import: "' + WName + '" must have shape [' +
      IntToStr(d_model) + '], got ' + Reader.ShapeAsString(WName));
  if (Reader.DimCount(BName) <> 1) or (Reader.DimSize(BName, 0) <> d_model) then
    ImportError('GPT-2 import: "' + BName + '" must have shape [' +
      IntToStr(d_model) + '], got ' + Reader.ShapeAsString(BName));
  Tmp := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, Tmp);
    for i := 0 to d_model - 1 do
      Layer.Neurons[0].Weights.FData[i] := Tmp.FData[i]; // gamma
    Reader.LoadTensorFlat(BName, Tmp);
    for i := 0 to d_model - 1 do
      Layer.Neurons[1].Weights.FData[i] := Tmp.FData[i]; // beta
  finally
    Tmp.Free;
  end;
  Layer.FlushWeightCache();
end;

// Loads a HF Conv1D weight [in, out] (+ bias [out]) pair into a
// TNNetPointwiseConvLinear with OutDim neurons of InDim weights each.
// HF Conv1D computes y = x @ W + b with W[in][out] (row-major), so output
// channel j gets weight column j: Neuron[j].Weights[i] = W[i*out + j].
procedure LoadConv1DWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName, BName: string; InDim, OutDim: integer);
var
  W, B: TNNetVolume;
  i, j: integer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('GPT-2 import: missing tensor "' + WName + '".');
  if not Reader.HasTensor(BName) then
    ImportError('GPT-2 import: missing tensor "' + BName + '".');
  if (Reader.DimCount(WName) <> 2) or
     (Reader.DimSize(WName, 0) <> InDim) or
     (Reader.DimSize(WName, 1) <> OutDim) then
    ImportError('GPT-2 import: "' + WName + '" must have shape [' +
      IntToStr(InDim) + ', ' + IntToStr(OutDim) + '] (HF Conv1D stores ' +
      '[in, out]), got ' + Reader.ShapeAsString(WName));
  if (Reader.DimCount(BName) <> 1) or (Reader.DimSize(BName, 0) <> OutDim) then
    ImportError('GPT-2 import: "' + BName + '" must have shape [' +
      IntToStr(OutDim) + '], got ' + Reader.ShapeAsString(BName));
  if Layer.Neurons.Count <> OutDim then
    ImportError('GPT-2 import: internal error - layer for "' + WName +
      '" has ' + IntToStr(Layer.Neurons.Count) + ' neurons, expected ' +
      IntToStr(OutDim) + '.');
  W := TNNetVolume.Create;
  B := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, W);
    Reader.LoadTensorFlat(BName, B);
    for j := 0 to OutDim - 1 do
    begin
      if Layer.Neurons[j].Weights.Size <> InDim then
        ImportError('GPT-2 import: internal error - neuron ' + IntToStr(j) +
          ' for "' + WName + '" has ' +
          IntToStr(Layer.Neurons[j].Weights.Size) + ' weights, expected ' +
          IntToStr(InDim) + '.');
      for i := 0 to InDim - 1 do
        Layer.Neurons[j].Weights.FData[i] := W.FData[i * OutDim + j];
      Layer.Neurons[j].BiasWeight := B.FData[j];
    end;
  finally
    B.Free;
    W.Free;
  end;
  Layer.FlushWeightCache();
end;

function BuildGPT2FromSafeTensorsEx(const FileName: string;
  out Config: TGPT2Config; pSeqLen: integer = 0;
  pNumHeads: integer = 0): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TGPT2BlockLayers;
  EmbeddingLayer, PosLayer, FinalLN, LMHead: TNNetLayer;
  BranchInput: TNNetLayer;
  BlockCnt, SeqLen, i, j: integer;
  Tmp: TNNetVolume;
  BlockPrefix, TensorNameStr: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

begin
  Reader := TNNetSafeTensorsReader.Create(FileName);
  NN := nil;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      Config := ReadGPT2Config(Reader, pNumHeads);
      if pSeqLen <= 0 then SeqLen := Config.NCtx else SeqLen := pSeqLen;
      if SeqLen > Config.NCtx then
        ImportError('GPT-2 import: requested SeqLen=' + IntToStr(SeqLen) +
          ' exceeds the checkpoint context n_ctx=' + IntToStr(Config.NCtx) + '.');

      // ---------------- Architecture ----------------
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      // EncodeZero=1: token id 0 is a real BPE token ("!"), not padding.
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.NEmbd, {EncodeZero=}1) );
      PosLayer := NN.AddLayer(
        TNNetLearnedPositionalEmbedding.Create(Config.NCtx) );
      SetLength(Blocks, Config.NLayers);
      for BlockCnt := 0 to Config.NLayers - 1 do
      begin
        // Attention sub-block: x := x + c_proj(MHA(c_attn(ln_1(x)))).
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].LN1 := NN.AddLayer( TNNetTokenLayerNorm.Create(1e-5) );
        Blocks[BlockCnt].CAttn := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(3 * Config.NEmbd) );
        // Splits the fused Q|K|V slab per head, runs one causal SDPA per
        // head, concats heads and out-projects with PointwiseConvLinear(d)
        // - the returned layer IS the out-projection and receives c_proj.
        Blocks[BlockCnt].AttnProj := NN.AddMultiHeadSelfAttention(
          Config.NHeads, {CausalMask=}true);
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        // MLP sub-block: x := x + c_proj(gelu_new(c_fc(ln_2(x)))).
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].LN2 := NN.AddLayer( TNNetTokenLayerNorm.Create(1e-5) );
        Blocks[BlockCnt].CFc := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(4 * Config.NEmbd) );
        NN.AddLayer( TNNetGELU.Create() ); // tanh approximation = gelu_new
        Blocks[BlockCnt].MlpProj := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.NEmbd) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
      end;
      FinalLN := NN.AddLayer( TNNetTokenLayerNorm.Create(1e-5) );
      // LM head tied to wte: logits = h . wte^T. Implemented as an untied
      // PointwiseConvLinear(vocab) whose weights are a COPY of wte (see the
      // unit header). Bias-free in GPT-2: biases stay 0.
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        // wte -> embedding table (vocab rows of d floats, row-major both
        // in the checkpoint and in TNNetEmbedding's neuron volume).
        Reader.LoadTensorFlat(Config.Prefix + 'wte.weight', Tmp);
        if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('GPT-2 import: wte.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table size ' +
            IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
        EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
        EmbeddingLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'wte.weight');
        // wte -> LM head (tied weights copied; row t of wte = neuron t).
        for j := 0 to Config.VocabSize - 1 do
        begin
          for i := 0 to Config.NEmbd - 1 do
            LMHead.Neurons[j].Weights.FData[i] :=
              Tmp.FData[j * Config.NEmbd + i];
          LMHead.Neurons[j].BiasWeight := 0;
        end;
        LMHead.FlushWeightCache();
        // wpe -> learned positional table (n_ctx rows of d floats).
        Reader.LoadTensorFlat(Config.Prefix + 'wpe.weight', Tmp);
        if PosLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('GPT-2 import: wpe.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the positional table size ' +
            IntToStr(PosLayer.Neurons[0].Weights.Size) + '.');
        PosLayer.Neurons[0].Weights.Copy(Tmp);
        PosLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'wpe.weight');
      finally
        Tmp.Free;
      end;
      for BlockCnt := 0 to Config.NLayers - 1 do
      begin
        BlockPrefix := Config.Prefix + 'h.' + IntToStr(BlockCnt) + '.';
        LoadLayerNormWeights(Reader, Blocks[BlockCnt].LN1,
          BlockPrefix + 'ln_1.weight', BlockPrefix + 'ln_1.bias',
          Config.NEmbd);
        MarkConsumed(BlockPrefix + 'ln_1.weight');
        MarkConsumed(BlockPrefix + 'ln_1.bias');
        LoadConv1DWeights(Reader, Blocks[BlockCnt].CAttn,
          BlockPrefix + 'attn.c_attn.weight', BlockPrefix + 'attn.c_attn.bias',
          Config.NEmbd, 3 * Config.NEmbd);
        MarkConsumed(BlockPrefix + 'attn.c_attn.weight');
        MarkConsumed(BlockPrefix + 'attn.c_attn.bias');
        LoadConv1DWeights(Reader, Blocks[BlockCnt].AttnProj,
          BlockPrefix + 'attn.c_proj.weight', BlockPrefix + 'attn.c_proj.bias',
          Config.NEmbd, Config.NEmbd);
        MarkConsumed(BlockPrefix + 'attn.c_proj.weight');
        MarkConsumed(BlockPrefix + 'attn.c_proj.bias');
        LoadLayerNormWeights(Reader, Blocks[BlockCnt].LN2,
          BlockPrefix + 'ln_2.weight', BlockPrefix + 'ln_2.bias',
          Config.NEmbd);
        MarkConsumed(BlockPrefix + 'ln_2.weight');
        MarkConsumed(BlockPrefix + 'ln_2.bias');
        LoadConv1DWeights(Reader, Blocks[BlockCnt].CFc,
          BlockPrefix + 'mlp.c_fc.weight', BlockPrefix + 'mlp.c_fc.bias',
          Config.NEmbd, 4 * Config.NEmbd);
        MarkConsumed(BlockPrefix + 'mlp.c_fc.weight');
        MarkConsumed(BlockPrefix + 'mlp.c_fc.bias');
        LoadConv1DWeights(Reader, Blocks[BlockCnt].MlpProj,
          BlockPrefix + 'mlp.c_proj.weight', BlockPrefix + 'mlp.c_proj.bias',
          4 * Config.NEmbd, Config.NEmbd);
        MarkConsumed(BlockPrefix + 'mlp.c_proj.weight');
        MarkConsumed(BlockPrefix + 'mlp.c_proj.bias');
      end;
      LoadLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'ln_f.weight', Config.Prefix + 'ln_f.bias',
        Config.NEmbd);
      MarkConsumed(Config.Prefix + 'ln_f.weight');
      MarkConsumed(Config.Prefix + 'ln_f.bias');

      // ---------------- Unexpected-tensor check ----------------
      // Every tensor must be consumed or be a known ignorable buffer:
      //  - h.N.attn.bias / h.N.attn.masked_bias: the causal-mask buffers
      //    PyTorch serializes (the mask is structural here);
      //  - lm_head.weight: the tied LM head (identical to wte by definition).
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        if (Pos('.attn.bias', TensorNameStr) > 0) or
           (Pos('.attn.masked_bias', TensorNameStr) > 0) or
           (TensorNameStr = 'lm_head.weight') then continue;
        ImportError('GPT-2 import: unexpected tensor "' + TensorNameStr +
          '" (shape ' + Reader.ShapeAsString(TensorNameStr) +
          ') in ' + FileName + ' - refusing a partial import.');
      end;
      Result := NN;
      NN := nil; // ownership transferred to the caller
    except
      on E: ESafeTensorsError do
        raise EPretrainedImportError.Create(E.Message);
    end;
  finally
    NN.Free; // non-nil only if an exception unwound the build
    Consumed.Free;
    Reader.Free;
  end;
end;

function BuildGPT2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pNumHeads: integer = 0): TNNet;
var
  IgnoredConfig: TGPT2Config;
begin
  Result := BuildGPT2FromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pNumHeads);
end;

end.
