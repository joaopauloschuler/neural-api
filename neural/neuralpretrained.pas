unit neuralpretrained;
// Pretrained-checkpoint importers: builds CAI TNNet networks from
// HuggingFace safetensors checkpoints and loads the real weights into them.
//
// Currently implemented:
//   - GPT-2 (the openai-community/gpt2 family) - BuildGPT2FromSafeTensors.
//   - Llama-architecture decoders (Llama-2 / TinyLlama / Llama-3-style:
//     RMSNorm + rotary GQA + SwiGLU MLP, no biases) -
//     BuildLlamaFromSafeTensors. See the LLAMA IMPORT section below.
//
// ============================ GPT-2 IMPORT =================================
// The architecture is rebuilt from existing CAI layers plus the
// import-oriented TNNetTokenLayerNorm / TNNetLearnedPositionalEmbedding
// layers:
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
// ============================ LLAMA IMPORT =================================
// BuildLlamaFromSafeTensors rebuilds the standard Llama-2/TinyLlama decoder:
//
//   Input(SeqLen tokens) -> TNNetEmbedding (model.embed_tokens)
//     -> num_hidden_layers x [ pre-RMSNorm rotary-GQA residual
//                              + pre-RMSNorm SwiGLU-MLP residual ]
//     -> TNNetTokenRMSNorm (model.norm)
//     -> TNNetPointwiseConvLinear(vocab) LM head (lm_head.weight; COPIED
//        from embed_tokens when config says tie_word_embeddings).
//
// Per block (HF LlamaDecoderLayer, pre-norm, NO biases anywhere):
//   x := x + o_proj( GQA( RoPE applied per head to q_proj/k_proj slices ) )
//   x := x + down_proj( silu(gate_proj(h)) * up_proj(h) ), h = RMSNorm(x)
//
// The rotary GQA is wired from primitives exactly like
// TNNet.AddMultiHeadGroupedQueryAttention, with one TNNetRotaryEmbedding
// (base = rope_theta) inserted after each per-head Q slice and each
// per-KV-head K slice (depth = head_dim, so the RoPE frequency schedule
// matches HF's per-head inv_freq). Positional information comes ONLY from
// RoPE - there is no positional-embedding table.
//
// ROTATE_HALF CONVENTION: HF Llama rotates pairs (i, i + head_dim/2)
// (first-half/second-half split), while TNNetRotaryEmbedding rotates
// consecutive pairs (2k, 2k+1) - the interleaved GPT-NeoX-vs-GPT-J layout
// difference. Both use the same angle theta_k = rope_theta^(-2k/head_dim)
// for pair k, so the importer PERMUTES the q_proj/k_proj output rows
// within each head while loading:
//   CAI channel (h*hd + 2k)     <- HF row (h*hd + k)
//   CAI channel (h*hd + 2k + 1) <- HF row (h*hd + k + hd/2)
// after which the interleaved rotation is numerically identical to HF's
// rotate_half. Q and K are permuted IDENTICALLY, so the attention logits
// q.k are unchanged; V and o_proj are NOT permuted (V is never rotated and
// the SDPA output is a V-weighted average, untouched by the Q/K relabel).
//
// SwiGLU mapping: TNNetSwiGLU computes A * Swish(B) where A is the FIRST
// depth half and B the SECOND, so the fused gate/up projection loads
// up_proj into neurons 0..I-1 and gate_proj into neurons I..2I-1, giving
// HF's down_proj( silu(gate) * up ) exactly.
//
// HF Llama uses nn.Linear ([out, in] storage, y = x.W^T) - the OPPOSITE of
// GPT-2's transposed Conv1D - so rows load straight into
// TNNetPointwiseConvLinear neurons with NO transpose. All Linear layers
// are bias-free; the CAI biases are zeroed.
//
// The config (hidden_size, intermediate_size, num_hidden_layers,
// num_attention_heads, num_key_value_heads, rms_norm_eps, rope_theta,
// vocab_size, max_position_embeddings, tie_word_embeddings) is read from
// the HF config.json sitting next to the safetensors file (or from an
// explicit path / record). Tokenizer loading (SentencePiece /
// tokenizer.json) is OUT OF SCOPE here: drive the net with raw token ids.
//
// Both importers FAIL LOUDLY (EPretrainedImportError) on missing tensors,
// unexpected tensors and shape mismatches.
//
// This unit is coded by Claude (AI).
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralsafetensors;

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
// pInferenceOnly = True frees every layer's Delta/BackInertia training
// volumes DURING construction (TNNet.MakeInferenceOnly after each block),
// cutting peak and resident memory to ~1/3 - the returned net can only
// Compute(), never train. The full 124M GPT-2 then fits in well under 2 GB.
function BuildGPT2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pNumHeads: integer = 0;
  pInferenceOnly: boolean = false): TNNet;

// Same, also returning the inferred config.
function BuildGPT2FromSafeTensorsEx(const FileName: string;
  out Config: TGPT2Config; pSeqLen: integer = 0;
  pNumHeads: integer = 0; pInferenceOnly: boolean = false): TNNet;

type
  TLlamaConfig = record
    HiddenSize: integer;       // d_model (hidden_size)
    IntermediateSize: integer; // SwiGLU MLP width (intermediate_size)
    NumLayers: integer;        // decoder blocks (num_hidden_layers)
    NumHeads: integer;         // query heads (num_attention_heads)
    NumKVHeads: integer;       // K/V heads (num_key_value_heads; GQA/MQA)
    VocabSize: integer;        // vocab_size
    MaxPositions: integer;     // max_position_embeddings (context length)
    RmsNormEps: TNeuralFloat;  // rms_norm_eps
    RopeTheta: TNeuralFloat;   // rope_theta (RoPE base)
    TieWordEmbeddings: boolean;// tie_word_embeddings (lm_head := embed)
    Prefix: string;            // tensor-name prefix ('model.' or '')
  end;

// Reads a HF Llama config.json. Required: hidden_size, intermediate_size,
// num_hidden_layers, num_attention_heads, vocab_size,
// max_position_embeddings. Defaults: num_key_value_heads =
// num_attention_heads, rms_norm_eps = 1e-6, rope_theta = 10000,
// tie_word_embeddings = false. Fails on a non-"llama" model_type or on a
// non-null rope_scaling (long-context scaling is not wired here yet).
// Prefix is left '' - the builders detect it from the checkpoint.
function ReadLlamaConfigFromJSONFile(const FileName: string): TLlamaConfig;

function LlamaConfigToString(const Config: TLlamaConfig): string;

// Builds a TNNet with the Llama architecture described by Config and loads
// every weight from the safetensors checkpoint at FileName (see the unit
// header for the exact mapping). The net takes a (SeqLen,1,1) volume of
// token ids (as floats) and outputs (SeqLen,1,vocab) logits. pSeqLen = 0
// uses the full max_position_embeddings context. pInferenceOnly = True
// frees training volumes during construction (Compute()-only afterwards).
// Config.Prefix is detected from the checkpoint and written back.
function BuildLlamaFromSafeTensorsWithConfig(const FileName: string;
  var Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildLlamaFromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;

function BuildLlamaFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;

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
  pNumHeads: integer = 0; pInferenceOnly: boolean = false): TNNet;
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
      // pInferenceOnly: shrink training volumes as soon as each chunk of
      // layers exists - peak memory then carries the training volumes of at
      // most one block (plus the LM head's, briefly) instead of the whole
      // net's. MakeInferenceOnly is idempotent, so re-sweeping all layers
      // each time is cheap and keeps this code simple.
      if pInferenceOnly then NN.MakeInferenceOnly();
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
        if pInferenceOnly then NN.MakeInferenceOnly();
      end;
      FinalLN := NN.AddLayer( TNNetTokenLayerNorm.Create(1e-5) );
      // LM head tied to wte: logits = h . wte^T. Implemented as an untied
      // PointwiseConvLinear(vocab) whose weights are a COPY of wte (see the
      // unit header). Bias-free in GPT-2: biases stay 0.
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();

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
  pSeqLen: integer = 0; pNumHeads: integer = 0;
  pInferenceOnly: boolean = false): TNNet;
var
  IgnoredConfig: TGPT2Config;
begin
  Result := BuildGPT2FromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pNumHeads, pInferenceOnly);
end;

// ============================ LLAMA IMPORT =================================

function ReadLlamaConfigFromJSONFile(const FileName: string): TLlamaConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType: string;
  RopeScaling: TJSONData;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('Llama import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('Llama import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

begin
  if not FileExists(FileName) then
    ImportError('Llama import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('Llama import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('Llama import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'llama');
    if ModelType <> 'llama' then
      ImportError('Llama import: config model_type is "' + ModelType +
        '" - only "llama" is supported (Mistral sliding-window etc. are ' +
        'different architectures).');
    RopeScaling := Obj.Find('rope_scaling');
    if (RopeScaling <> nil) and not RopeScaling.IsNull then
      ImportError('Llama import: config carries a non-null "rope_scaling" - ' +
        'long-context RoPE scaling is not wired into this importer yet.');
    Result.HiddenSize := RequiredInt('hidden_size');
    Result.IntermediateSize := RequiredInt('intermediate_size');
    Result.NumLayers := RequiredInt('num_hidden_layers');
    Result.NumHeads := RequiredInt('num_attention_heads');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.MaxPositions := RequiredInt('max_position_embeddings');
    Result.NumKVHeads := Obj.Get('num_key_value_heads', Result.NumHeads);
    Result.RmsNormEps := Obj.Get('rms_norm_eps', 1.0e-6);
    Result.RopeTheta := Obj.Get('rope_theta', 10000.0);
    Result.TieWordEmbeddings := Obj.Get('tie_word_embeddings', False);
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function LlamaConfigToString(const Config: TLlamaConfig): string;
begin
  Result := 'Llama config: layers=' + IntToStr(Config.NumLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', kv_heads=' + IntToStr(Config.NumKVHeads) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', max_pos=' + IntToStr(Config.MaxPositions) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', rms_eps=' + FloatToStr(Config.RmsNormEps) +
    ', rope_theta=' + FloatToStr(Config.RopeTheta) +
    ', tied=' + BoolToStr(Config.TieWordEmbeddings, true);
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

// Loads a HF RMSNorm gain vector [d_model] into a TNNetTokenRMSNorm.
procedure LoadLlamaRMSNormWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName: string; d_model: integer);
var
  Tmp: TNNetVolume;
  i: integer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('Llama import: missing tensor "' + WName + '".');
  if (Reader.DimCount(WName) <> 1) or (Reader.DimSize(WName, 0) <> d_model) then
    ImportError('Llama import: "' + WName + '" must have shape [' +
      IntToStr(d_model) + '], got ' + Reader.ShapeAsString(WName));
  Tmp := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, Tmp);
    for i := 0 to d_model - 1 do
      Layer.Neurons[0].Weights.FData[i] := Tmp.FData[i]; // gain
  finally
    Tmp.Free;
  end;
  Layer.FlushWeightCache();
end;

// Loads a HF nn.Linear weight [out, in] (bias-free, the Llama convention)
// into a TNNetPointwiseConvLinear: HF computes y = x . W^T, so output
// channel j IS row j: Neuron[NeuronBase + j].Weights[i] = W[j*in + i].
// (NO transpose - the opposite of GPT-2's Conv1D [in, out] storage.)
// NeuronBase lets the fused SwiGLU gate/up layer load two HF tensors into
// disjoint neuron ranges. ExpectedNeurons guards the layer width.
// RotaryHeadDim > 0 additionally PERMUTES the output rows within each
// RotaryHeadDim-wide head to convert HF's rotate_half (first-half /
// second-half) pair layout into TNNetRotaryEmbedding's interleaved
// (even/odd) pair layout (see the unit header):
//   target channel (h*hd + 2k)     <- HF row (h*hd + k)
//   target channel (h*hd + 2k + 1) <- HF row (h*hd + k + hd/2)
procedure LoadLlamaLinearWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName: string; InDim, OutDim: integer;
  NeuronBase: integer = 0; ExpectedNeurons: integer = -1;
  RotaryHeadDim: integer = 0);
var
  W: TNNetVolume;
  i, j, TargetIdx, HeadIdx, RowInHead, HalfDim: integer;
begin
  if ExpectedNeurons < 0 then ExpectedNeurons := OutDim;
  if not Reader.HasTensor(WName) then
    ImportError('Llama import: missing tensor "' + WName + '".');
  if (Reader.DimCount(WName) <> 2) or
     (Reader.DimSize(WName, 0) <> OutDim) or
     (Reader.DimSize(WName, 1) <> InDim) then
    ImportError('Llama import: "' + WName + '" must have shape [' +
      IntToStr(OutDim) + ', ' + IntToStr(InDim) + '] (nn.Linear stores ' +
      '[out, in]), got ' + Reader.ShapeAsString(WName));
  if Layer.Neurons.Count <> ExpectedNeurons then
    ImportError('Llama import: internal error - layer for "' + WName +
      '" has ' + IntToStr(Layer.Neurons.Count) + ' neurons, expected ' +
      IntToStr(ExpectedNeurons) + '.');
  if (RotaryHeadDim > 0) and
     (((OutDim mod RotaryHeadDim) <> 0) or Odd(RotaryHeadDim)) then
    ImportError('Llama import: internal error - "' + WName + '" rows (' +
      IntToStr(OutDim) + ') are not a multiple of the even head_dim ' +
      IntToStr(RotaryHeadDim) + '.');
  HalfDim := RotaryHeadDim div 2;
  W := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, W);
    for j := 0 to OutDim - 1 do
    begin
      if RotaryHeadDim > 0 then
      begin
        HeadIdx := j div RotaryHeadDim;
        RowInHead := j mod RotaryHeadDim;
        if RowInHead < HalfDim then
          TargetIdx := HeadIdx * RotaryHeadDim + 2 * RowInHead
        else
          TargetIdx := HeadIdx * RotaryHeadDim + 2 * (RowInHead - HalfDim) + 1;
      end
      else
        TargetIdx := j;
      TargetIdx := TargetIdx + NeuronBase;
      if Layer.Neurons[TargetIdx].Weights.Size <> InDim then
        ImportError('Llama import: internal error - neuron ' +
          IntToStr(TargetIdx) + ' for "' + WName + '" has ' +
          IntToStr(Layer.Neurons[TargetIdx].Weights.Size) +
          ' weights, expected ' + IntToStr(InDim) + '.');
      for i := 0 to InDim - 1 do
        Layer.Neurons[TargetIdx].Weights.FData[i] := W.FData[j * InDim + i];
      Layer.Neurons[TargetIdx].BiasWeight := 0; // Llama Linears are bias-free
    end;
  finally
    W.Free;
  end;
  Layer.FlushWeightCache();
end;

type
  TLlamaBlockLayers = record
    AttnNorm, QProj, KProj, VProj, OProj, MlpNorm, GateUp, Down: TNNetLayer;
  end;

function BuildLlamaFromSafeTensorsWithConfig(const FileName: string;
  var Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TLlamaBlockLayers;
  EmbeddingLayer, FinalNorm, LMHead: TNNetLayer;
  BranchInput, NormedSource: TNNetLayer;
  QSlice, KSlice, VSlice, HeadPack: TNNetLayer;
  KRotated, VSlices, HeadOutputs: array of TNNetLayer;
  SliceChannels: array of integer;
  BlockCnt, SeqLen, HeadCnt, KVHeadCnt, KVGroup, GroupSize: integer;
  HeadDim, KVWidth, i, j, d: integer;
  Tmp: TNNetVolume;
  BlockPrefix, TensorNameStr, LMHeadName: string;
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
      // ---------------- Config validation & prefix detection -------------
      if Config.NumHeads < 1 then
        ImportError('Llama import: num_attention_heads must be >= 1.');
      if Config.NumKVHeads < 1 then
        ImportError('Llama import: num_key_value_heads must be >= 1.');
      if (Config.HiddenSize mod Config.NumHeads) <> 0 then
        ImportError('Llama import: hidden_size=' +
          IntToStr(Config.HiddenSize) + ' is not divisible by ' +
          'num_attention_heads=' + IntToStr(Config.NumHeads) + '.');
      if (Config.NumHeads mod Config.NumKVHeads) <> 0 then
        ImportError('Llama import: num_attention_heads=' +
          IntToStr(Config.NumHeads) + ' is not divisible by ' +
          'num_key_value_heads=' + IntToStr(Config.NumKVHeads) + '.');
      HeadDim := Config.HiddenSize div Config.NumHeads;
      if Odd(HeadDim) then
        ImportError('Llama import: head_dim=' + IntToStr(HeadDim) +
          ' must be even (RoPE rotates channel pairs).');
      KVWidth := Config.NumKVHeads * HeadDim;
      GroupSize := Config.NumHeads div Config.NumKVHeads;
      if Reader.HasTensor('model.embed_tokens.weight') then
        Config.Prefix := 'model.'
      else if Reader.HasTensor('embed_tokens.weight') then
        Config.Prefix := ''
      else
        ImportError('Llama import: neither "model.embed_tokens.weight" nor ' +
          '"embed_tokens.weight" found in ' + Reader.FileName +
          ' - not a Llama checkpoint?');
      if (Reader.DimCount(Config.Prefix + 'embed_tokens.weight') <> 2) or
         (Reader.DimSize(Config.Prefix + 'embed_tokens.weight', 0) <>
          Config.VocabSize) or
         (Reader.DimSize(Config.Prefix + 'embed_tokens.weight', 1) <>
          Config.HiddenSize) then
        ImportError('Llama import: embed_tokens.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.HiddenSize) +
          '], got ' +
          Reader.ShapeAsString(Config.Prefix + 'embed_tokens.weight'));
      LMHeadName := 'lm_head.weight';
      if (not Config.TieWordEmbeddings) and
         (not Reader.HasTensor(LMHeadName)) then
        ImportError('Llama import: config says tie_word_embeddings=false ' +
          'but "' + LMHeadName + '" is missing from ' + Reader.FileName + '.');
      if pSeqLen <= 0 then SeqLen := Config.MaxPositions
      else SeqLen := pSeqLen;
      if SeqLen > Config.MaxPositions then
        ImportError('Llama import: requested SeqLen=' + IntToStr(SeqLen) +
          ' exceeds max_position_embeddings=' +
          IntToStr(Config.MaxPositions) + '.');

      // ---------------- Architecture ----------------
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      // EncodeZero=1: token id 0 is a real token (<unk> in the Llama vocab),
      // not padding.
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      SetLength(Blocks, Config.NumLayers);
      SetLength(KRotated, Config.NumKVHeads);
      SetLength(VSlices, Config.NumKVHeads);
      SetLength(HeadOutputs, Config.NumHeads);
      SetLength(SliceChannels, HeadDim);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // Attention sub-block: x := x + o_proj(rotary-GQA(RMSNorm(x))).
        // Wired from primitives exactly like AddMultiHeadGroupedQueryAttention
        // with TNNetRotaryEmbedding(rope_theta) on each per-head Q/K slice
        // (depth = head_dim so the frequency schedule matches HF).
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].AttnNorm :=
          NN.AddLayer( TNNetTokenRMSNorm.Create(Config.RmsNormEps) );
        NormedSource := NN.GetLastLayer();
        Blocks[BlockCnt].QProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize), NormedSource);
        Blocks[BlockCnt].KProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(KVWidth), NormedSource);
        Blocks[BlockCnt].VProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(KVWidth), NormedSource);
        // K is rotated ONCE per KV head; V is never rotated.
        for KVHeadCnt := 0 to Config.NumKVHeads - 1 do
        begin
          for d := 0 to HeadDim - 1 do
            SliceChannels[d] := KVHeadCnt * HeadDim + d;
          KSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(SliceChannels), Blocks[BlockCnt].KProj);
          KRotated[KVHeadCnt] := NN.AddLayerAfter(
            TNNetRotaryEmbedding.Create(Config.RopeTheta), KSlice);
          VSlices[KVHeadCnt] := NN.AddLayerAfter(
            TNNetSplitChannels.Create(SliceChannels), Blocks[BlockCnt].VProj);
        end;
        for HeadCnt := 0 to Config.NumHeads - 1 do
        begin
          KVGroup := HeadCnt div GroupSize;
          for d := 0 to HeadDim - 1 do
            SliceChannels[d] := HeadCnt * HeadDim + d;
          QSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(SliceChannels), Blocks[BlockCnt].QProj);
          QSlice := NN.AddLayerAfter(
            TNNetRotaryEmbedding.Create(Config.RopeTheta), QSlice);
          // Pack [Q_h | K_group | V_group] (width 3*head_dim) for SDPA.
          HeadPack := NN.AddLayer( TNNetDeepConcat.Create(
            [QSlice, KRotated[KVGroup], VSlices[KVGroup]]) );
          HeadOutputs[HeadCnt] := NN.AddLayerAfter(
            TNNetScaledDotProductAttention.Create(HeadDim, {CausalMask=}true),
            HeadPack);
        end;
        NN.AddLayer( TNNetDeepConcat.Create(HeadOutputs) );
        Blocks[BlockCnt].OProj := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        // MLP sub-block: x := x + down(silu(gate(h)) * up(h)), h = RMSNorm(x).
        // TNNetSwiGLU computes FIRSTHALF * Swish(SECONDHALF), so the fused
        // projection holds up_proj in neurons 0..I-1 and gate_proj in
        // neurons I..2I-1 (see LoadLlamaLinearWeights calls below).
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].MlpNorm :=
          NN.AddLayer( TNNetTokenRMSNorm.Create(Config.RmsNormEps) );
        Blocks[BlockCnt].GateUp := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(2 * Config.IntermediateSize) );
        NN.AddLayer( TNNetSwiGLU.Create() );
        Blocks[BlockCnt].Down := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
      end;
      FinalNorm := NN.AddLayer(
        TNNetTokenRMSNorm.Create(Config.RmsNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        // embed_tokens -> embedding table (vocab rows of d floats,
        // row-major both in the checkpoint and in TNNetEmbedding).
        Reader.LoadTensorFlat(Config.Prefix + 'embed_tokens.weight', Tmp);
        if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('Llama import: embed_tokens.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table size ' +
            IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
        EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
        EmbeddingLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'embed_tokens.weight');
        if Config.TieWordEmbeddings then
        begin
          // Tied LM head: logits = h . embed^T (rows copied, bias-free).
          for j := 0 to Config.VocabSize - 1 do
          begin
            for i := 0 to Config.HiddenSize - 1 do
              LMHead.Neurons[j].Weights.FData[i] :=
                Tmp.FData[j * Config.HiddenSize + i];
            LMHead.Neurons[j].BiasWeight := 0;
          end;
          LMHead.FlushWeightCache();
          // A redundant serialized lm_head.weight is ignorable when tied.
          if Reader.HasTensor(LMHeadName) then MarkConsumed(LMHeadName);
        end
        else
        begin
          LoadLlamaLinearWeights(Reader, LMHead, LMHeadName,
            Config.HiddenSize, Config.VocabSize);
          MarkConsumed(LMHeadName);
        end;
      finally
        Tmp.Free;
      end;
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        BlockPrefix := Config.Prefix + 'layers.' + IntToStr(BlockCnt) + '.';
        LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].AttnNorm,
          BlockPrefix + 'input_layernorm.weight', Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'input_layernorm.weight');
        // q_proj/k_proj rows are PERMUTED per head for the rotate_half
        // convention (RotaryHeadDim); v_proj/o_proj load straight.
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].QProj,
          BlockPrefix + 'self_attn.q_proj.weight',
          Config.HiddenSize, Config.HiddenSize, 0, -1, HeadDim);
        MarkConsumed(BlockPrefix + 'self_attn.q_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].KProj,
          BlockPrefix + 'self_attn.k_proj.weight',
          Config.HiddenSize, KVWidth, 0, -1, HeadDim);
        MarkConsumed(BlockPrefix + 'self_attn.k_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].VProj,
          BlockPrefix + 'self_attn.v_proj.weight',
          Config.HiddenSize, KVWidth);
        MarkConsumed(BlockPrefix + 'self_attn.v_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].OProj,
          BlockPrefix + 'self_attn.o_proj.weight',
          Config.HiddenSize, Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'self_attn.o_proj.weight');
        LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].MlpNorm,
          BlockPrefix + 'post_attention_layernorm.weight', Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'post_attention_layernorm.weight');
        // Fused gate/up: up_proj -> neurons 0..I-1 (SwiGLU's linear half),
        // gate_proj -> neurons I..2I-1 (SwiGLU's Swish-gated half).
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].GateUp,
          BlockPrefix + 'mlp.up_proj.weight',
          Config.HiddenSize, Config.IntermediateSize,
          0, 2 * Config.IntermediateSize);
        MarkConsumed(BlockPrefix + 'mlp.up_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].GateUp,
          BlockPrefix + 'mlp.gate_proj.weight',
          Config.HiddenSize, Config.IntermediateSize,
          Config.IntermediateSize, 2 * Config.IntermediateSize);
        MarkConsumed(BlockPrefix + 'mlp.gate_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Down,
          BlockPrefix + 'mlp.down_proj.weight',
          Config.IntermediateSize, Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'mlp.down_proj.weight');
      end;
      LoadLlamaRMSNormWeights(Reader, FinalNorm,
        Config.Prefix + 'norm.weight', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'norm.weight');

      // ---------------- Unexpected-tensor check ----------------
      // Every tensor must be consumed or be a known ignorable buffer:
      // older HF exports serialize the per-layer "rotary_emb.inv_freq"
      // buffers (RoPE is structural here, rebuilt from rope_theta).
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        if Pos('rotary_emb.inv_freq', TensorNameStr) > 0 then continue;
        ImportError('Llama import: unexpected tensor "' + TensorNameStr +
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

function BuildLlamaFromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadLlamaConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildLlamaFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly);
end;

function BuildLlamaFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildLlamaFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly);
end;

end.
