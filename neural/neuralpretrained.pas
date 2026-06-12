unit neuralpretrained;
// Pretrained-checkpoint importers: builds CAI TNNet networks from
// HuggingFace safetensors checkpoints and loads the real weights into them.
//
// Currently implemented:
//   - GPT-2 (the openai-community/gpt2 family) - BuildGPT2FromSafeTensors.
//   - Llama-architecture decoders (Llama-2 / TinyLlama / Llama-3-style:
//     RMSNorm + rotary GQA + SwiGLU MLP, no biases) -
//     BuildLlamaFromSafeTensors. See the LLAMA IMPORT section below.
//   - Mistral (the Llama skeleton + optional sliding-window attention from
//     config sliding_window; null = full attention) -
//     BuildMistralFromSafeTensors.
//   - Qwen2 (the Llama skeleton + q/k/v projection biases; o_proj and the
//     MLP stay bias-free) - BuildQwen2FromSafeTensors.
//   - Qwen3 (the Llama skeleton WITHOUT q/k/v biases + per-head RMSNorm on
//     q and k applied AFTER projection and BEFORE RoPE, gain shape
//     [head_dim] shared across heads, plus an optionally DECOUPLED
//     head_dim where num_heads*head_dim <> hidden_size) -
//     BuildQwen3FromSafeTensors.
//   - Gemma 1 (the Llama skeleton with three LOAD-TIME deltas: GeGLU MLP
//     - gated tanh-GELU, TNNetGEGLU instead of TNNetSwiGLU; zero-centered
//     RMSNorm - gains stored as 1+w; embedding output scaled by
//     sqrt(hidden_size) - folded into the embedding rows, NOT the tied LM
//     head; always-tied head; 2B is MQA via num_key_value_heads=1, 7B uses
//     the decoupled head_dim=256) - BuildGemmaFromSafeTensors.
//   - GPT-Neo (EleutherAI's GPT-3 replica; the roneneldan/TinyStories
//     checkpoints share this architecture) - BuildGPTNeoFromSafeTensors.
//     See the GPT-NEO IMPORT section below.
//   - GPT-NeoX (EleutherAI's Pythia 70M..12B science suite; parallel
//     two-LN residual + PARTIAL rotary + fused per-head-interleaved
//     query_key_value + untied embed_in/embed_out) -
//     BuildGPTNeoXFromSafeTensors. See the GPT-NEOX IMPORT section below.
//   - GPT-J (EleutherAI's GPT-J-6B; SHARED-LN parallel residual +
//     partial INTERLEAVED rotary + separate bias-free q/k/v + untied
//     lm_head WITH bias) - BuildGPTJFromSafeTensors. See the GPT-J
//     IMPORT section below.
//   - Phi (microsoft/phi-1, phi-1_5, phi-2; SHARED-LN parallel residual
//     like GPT-J but with BIASED separate q/k/v/dense + partial rotary in
//     the NeoX rotate_half layout + untied lm_head WITH bias) -
//     BuildPhiFromSafeTensors. See the PHI IMPORT section below.
//   - BERT (vanilla encoder family, model_type "bert": bert-base/tiny,
//     sentence-transformers MiniLM, ...) - BuildBertFromSafeTensors.
//     The FIRST ENCODER here: outputs hidden states, not logits. See the
//     BERT IMPORT section below.
//   - DistilBERT (model_type "distilbert": the same post-LN encoder math
//     as BERT with different tensor names, NO token-type embeddings and
//     NO pooler) - the SAME BuildBertFromSafeTensors entry points; the
//     config's model_type selects the family. See the BERT IMPORT section.
//   - RoBERTa (model_type "roberta": the BERT skeleton with position ids
//     OFFSET by padding_idx+1 = 2 and a degenerate 1-row token-type
//     table) - also the SAME BuildBertFromSafeTensors entry points. See
//     the BERT IMPORT section.
//   - T5 / Flan-T5 (model_type "t5") - the FIRST ENCODER-DECODER import:
//     BuildT5FromSafeTensors returns TWO nets (encoder + two-input decoder
//     whose cross-attentions read the encoder hidden states from a second
//     TNNetInput); RunT5 runs the pair end-to-end. Supports both the
//     original v1.0 recipe (ReLU FFN, tied + d_model^-0.5-scaled head) and
//     the v1.1/Flan recipe (gated-GELU FFN, untied head). See the T5
//     IMPORT section below.
//   - Fine-tuned classifier checkpoints: BertForSequenceClassification
//     ([CLS]-pooled) and GPT2ForSequenceClassification (last-token-pooled)
//     - BuildBertForSequenceClassificationFromSafeTensors /
//     BuildGPT2ForSequenceClassificationFromSafeTensors, with id2label
//     support. See the SEQUENCE CLASSIFICATION IMPORT section.
//   - AutoModel-style dispatch on config.json's model_type (and, for the
//     classifier heads, its "architectures" array) -
//     BuildFromPretrained (directory or .safetensors/.index.json path).
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
// ============================ GPT-NEO IMPORT ===============================
// BuildGPTNeoFromSafeTensors rebuilds EleutherAI's GPT-Neo decoder - the
// GPT-2 skeleton (learned absolute wpe positions, pre-LN blocks, gelu_new
// MLP, final ln_f, tied LM head) with three architectural quirks:
//
// 1. ALTERNATING ATTENTION: config attention_layers (or the packed
//    attention_types form) marks each layer "global" (full causal) or
//    "local" (causally banded). HF's local mask
//    bias ^ tril(bias, -window_size) keeps keys j with i - j < window_size
//    (exactly window_size keys including the query itself) - the SAME band
//    TNNetScaledDotProductAttention's pWindow applies (it masks
//    i - j >= Window), so local layers simply pass Window = window_size to
//    AddMultiHeadSelfAttention and global layers pass 0.
//
// 2. UNSCALED ATTENTION: GPT-Neo does NOT divide the scores by
//    sqrt(d_head). Rather than a new layer flag, the importer FOLDS the
//    compensating factor into the query projection at load time:
//    W_q (and a q bias, were one present - GPT-Neo's q_proj is bias-free)
//    are multiplied by sqrt(d_head), so the standard scaled SDPA computes
//    softmax((sqrt(d)q).k / sqrt(d)) = softmax(q.k) - identical scores
//    with zero layer changes.
//
// 3. WEIGHT ORIENTATION: plain nn.Linear [out, in] storage (like Llama,
//    NOT GPT-2's transposed Conv1D), with SEPARATE q/k/v projections.
//    q/k/v carry NO bias; out_proj and the two MLP linears do. The
//    separate projections are loaded into the fused Q|K|V slab the
//    multi-head builder expects (q -> neurons 0..d-1, k -> d..2d-1,
//    v -> 2d..3d-1; per-head slicing then matches HF's _split_heads).
//
// intermediate_size is null in every GPT-Neo config observed in the wild:
// the HF default 4 * hidden_size applies. The LM head is tied to wte
// (copied, like the GPT-2 path). The roneneldan/TinyStories-1M..33M
// reference checkpoints are GPT-Neo and load through this path unchanged.
//
// =========================== GPT-NEOX IMPORT ===============================
// BuildGPTNeoXFromSafeTensors rebuilds EleutherAI's GPT-NeoX decoder
// (model_type "gpt_neox") - the architecture of the Pythia science suite
// (70M..12B). Two ingredients distinguish it from everything above:
//
// 1. PARALLEL residual with TWO LayerNorms (config use_parallel_residual,
//    default true and used by every Pythia size):
//      x := x + Attn(LN_1(x)) + MLP(LN_2(x))
//    - both branches read the SAME block input through their own norms
//    (input_layernorm / post_attention_layernorm) and the residual is summed
//    ONCE (a 3-input TNNetSum). use_parallel_residual=false falls back to
//    the sequential pre-LN form x := x + Attn(LN_1(x));
//    x := x + MLP(LN_2(x)) - also wired here.
//
// 2. PARTIAL ROTARY: only the FIRST rotary_ndims = int(rotary_pct*head_dim)
//    channels of each Q/K head get RoPE (rotary_pct = 0.25 for Pythia); the
//    remaining channels pass through unrotated. Wired per head as a channel
//    split: SplitChannels(first d_rot) -> TNNetRotaryEmbedding(theta) and
//    SplitChannels(rest), re-concatenated before the SDPA. The RoPE
//    frequency schedule matches HF (theta_k = base^(-2k/d_rot): the layer
//    derives the schedule from its input depth, which IS d_rot here).
//    GPT-NeoX applies rotate_half WITHIN the rotary slice (the same
//    first-half/second-half layout as Llama), so the loader permutes the
//    first d_rot rows of every head exactly like the Llama importer does
//    over the full head (see the ROTATE_HALF CONVENTION note above).
//
// The attention projection is ONE fused query_key_value tensor
// [3*hidden, hidden] with PER-HEAD interleaving: head h occupies rows
// h*3*head_dim..(h+1)*3*head_dim-1, the first head_dim of them being q,
// then k, then v (HF view(..., num_heads, 3*head_dim) then thirds) -
// DIFFERENT from GPT-2's q|k|v thirds layout. The loader de-interleaves
// into the Q|K|V slab the per-head slicing expects, composing the rotary
// permutation on the way. query_key_value, attention.dense and both MLP
// linears (mlp.dense_h_to_4h / dense_4h_to_h) are plain nn.Linear
// [out, in] WITH biases. Attention is standard scaled (1/sqrt(head_dim)).
// hidden_act is "gelu" = the EXACT erf form in every Pythia config -
// composed from existing layers exactly like the BERT path (gelu_new /
// gelu_pytorch_tanh configs get the single TNNetGELU instead).
// Embeddings are UNTIED by default (embed_in / embed_out are separate
// tensors; tie_word_embeddings=false in every Pythia config); a tied
// config copies embed_in like the GPT-2 path. final_layer_norm caps the
// stack. Positions come ONLY from RoPE (no wpe table).
//
// ============================ GPT-J IMPORT =================================
// BuildGPTJFromSafeTensors rebuilds EleutherAI's GPT-J decoder (model_type
// "gptj", the architecture of GPT-J-6B). Two ingredients distinguish it
// from the GPT-NeoX path above:
//
// 1. SHARED-LN parallel residual: ONE LayerNorm (ln_1) feeds BOTH branches,
//      x := x + Attn(LN(x)) + MLP(LN(x))
//    - structurally GPT-NeoX's parallel form with LN_2 = LN_1, so the MLP
//    branch simply reads the SAME TNNetTokenLayerNorm output the attention
//    branch does (no sequential variant: GPT-J is always parallel).
//
// 2. PARTIAL rotary with the INTERLEAVED (GPT-J) pair layout: only the
//    FIRST rotary_dim channels of each Q/K head get RoPE (64 of head_dim
//    256 for GPT-J-6B), wired per head as the same SplitChannels ->
//    TNNetRotaryEmbedding -> re-concat split the GPT-NeoX path uses. But
//    GPT-J pairs channels (0,1),(2,3),... - rotate_every_two, exactly
//    TNNetRotaryEmbedding's native even/odd layout - so unlike
//    gpt_neox/llama there is NO rotate_half row permutation at weight-load
//    time: q_proj/k_proj rows load STRAIGHT. (The frequency schedule
//    matches HF: pair (2k, 2k+1) rotates at base^(-2k/rotary_dim), which
//    the layer derives from its input depth = rotary_dim.)
//
// Attention has SEPARATE bias-free q/k/v projections plus a bias-free
// out_proj (plain nn.Linear [out, in], head h = rows h*head_dim..); the
// MLP linears (mlp.fc_in / fc_out) carry biases. Attention is standard
// scaled (1/sqrt(head_dim)). activation_function is "gelu_new" (the tanh
// TNNetGELU) in every GPT-J config; the exact erf "gelu" composition is
// also wired. The LM head is UNTIED and - uniquely among the decoders
// here - carries a BIAS (lm_head.weight + lm_head.bias). ln_f caps the
// stack. Positions come ONLY from RoPE (no wpe table).
//
// ============================ PHI IMPORT ===================================
// BuildPhiFromSafeTensors rebuilds Microsoft's Phi decoder (model_type
// "phi": phi-1, phi-1_5, phi-2). Structurally it is GPT-J's SHARED-LN
// parallel residual,
//      x := x + Attn(LN(x)) + MLP(LN(x))
// with ONE input_layernorm (a LayerNorm WITH bias, not RMSNorm) feeding
// both branches - but every linear carries a BIAS (separate q/k/v +
// attention dense, mlp fc1/fc2, and the untied lm_head), and the partial
// rotary uses the NeoX rotate_half pair layout instead of GPT-J's
// interleaved one:
//
//   PARTIAL rotary, rotate_half layout: only the first
//   rotary_ndims = int(head_dim * partial_rotary_factor) channels of each
//   q/k head get RoPE (0.5 for phi-1/phi-1_5, 0.4 for phi-2); the rest
//   pass through. HF pairs channel k with k + rotary_ndims/2 (rotate_half
//   restricted to the rotary slice), so at WEIGHT-LOAD time the first
//   rotary_ndims rows of every q/k head are PERMUTED into
//   TNNetRotaryEmbedding's interleaved even/odd layout - the same
//   permutation the gpt_neox path composes into its fused slab, here
//   applied to the separate q_proj/k_proj matrices (and their biases) by
//   LoadPhiPartialRotaryLinearWeights. The rotary slice feeds a
//   depth-rotary_ndims TNNetRotaryEmbedding, so the frequency schedule
//   theta^(-2k/rotary_ndims) matches HF's inv_freq exactly.
//
// Attention is standard scaled (1/sqrt(head_dim)). hidden_act is
// "gelu_new" (the tanh TNNetGELU) in every released Phi config; the exact
// erf "gelu" composition is also wired. The LM head is UNTIED with a BIAS
// (lm_head.weight + lm_head.bias, like GPT-J). model.final_layernorm caps
// the stack. Positions come ONLY from RoPE (no wpe table). The
// qk_layernorm config flag (false in every released checkpoint) and GQA
// (num_key_value_heads <> num_attention_heads) are REJECTED explicitly
// rather than silently mis-loaded.
//
// ============================ BERT IMPORT ==================================
// BuildBertFromSafeTensors rebuilds the vanilla BERT ENCODER (HF BertModel,
// model_type "bert") - the first non-decoder in this unit. Differences from
// the decoder importers, all from existing CAI layers:
//
//   Input(SeqLen,1,2)            [ch 0 = token ids, ch 1 = token-type ids]
//     -> word + learned-position + token-type embeddings, SUMMED
//     -> TNNetTokenLayerNorm (embeddings.LayerNorm, eps 1e-12)
//     -> num_hidden_layers x POST-LN blocks (below)
//   output: (SeqLen,1,hidden_size) final hidden states - NO LM head.
//
// Per block (HF BertLayer, POST-LN - norms sit AFTER each residual sum,
// not before the sublayer like every decoder above):
//   x := LN_attn( x + dense( MHA(query|key|value (x)) ) )
//   x := LN_out ( x + output.dense( gelu( intermediate.dense(x) ) ) )
// MHA is TNNet.AddMultiHeadSelfAttention with CausalMask=FALSE -
// BIDIRECTIONAL attention, every position attends to all SeqLen keys.
// There is NO padding mask: feed full-length token rows (pad-token
// attention is the caller's problem; for parity work, avoid padding).
// Every linear is nn.Linear [out, in] WITH a bias (q/k/v, attention
// output.dense, both FFN linears) - LoadLlamaLinearWeights with BiasName.
// The separate q/k/v load into the fused Q|K|V slab like the GPT-Neo path.
//
// EXACT GELU: BERT's hidden_act "gelu" is the exact erf form
// 0.5*x*(1+erf(x/sqrt(2))), NOT TNNetGELU's tanh approximation (the two
// differ by ~3e-4 around |x|=2). The exact form is COMPOSED from existing
// layers: Phi(x) = MulByConstant(1/sqrt(2)) -> TNNetErf -> AddConstant(1)
// -> MulByConstant(0.5), then DeepConcat([Phi, x]) -> TNNetReGLU. ReGLU
// outputs ReLU(A)*B with A the FIRST half, and Phi is strictly in (0,1),
// so ReLU(Phi) = Phi and the product is exactly Phi(x)*x. Configs saying
// gelu_new / gelu_pytorch_tanh get the single TNNetGELU layer instead.
//
// TOKEN-TYPE (segment) EMBEDDINGS: input channel 1 is sliced off
// (TNNetSplitChannels), looked up in its own TNNetEmbedding
// (type_vocab_size rows) and SUMMED with the word+position branch before
// the embedding LayerNorm. Callers with single-segment inputs simply feed
// zeros in channel 1.
//
// POOLER (optional, pIncludePooler): HF's pooler_output =
// tanh(pooler.dense(h[CLS])). Wired as a per-token PointwiseConvLinear +
// TNNetHyperbolicTangent, so the output stays (SeqLen,1,hidden) and ROW 0
// (the [CLS] position) equals HF's pooler_output. When the pooler is NOT
// requested, pooler.dense.* tensors in the checkpoint are ignored (they
// are pinned weights of a head this net does not carry, like a tied
// lm_head.weight).
//
// DISTILBERT (model_type "distilbert") shares this exact encoder math and
// loads through the SAME builders; only the surface differs:
//   - config keys: n_layers, n_heads, dim, hidden_dim (instead of
//     num_hidden_layers, num_attention_heads, hidden_size,
//     intermediate_size); activation (default "gelu", the exact erf form;
//     modeling_distilbert hardcodes LayerNorm eps 1e-12);
//   - tensor names: embeddings.* as in BERT but NO token_type_embeddings,
//     blocks at transformer.layer.N. with attention.q_lin / k_lin / v_lin /
//     out_lin, sa_layer_norm, ffn.lin1 / lin2, output_layer_norm
//     (prefix '' for DistilBertModel, 'distilbert.' for DistilBertFor*);
//   - NO token-type embeddings: the input stays (SeqLen,1,2) for API
//     uniformity but channel 1 is IGNORED (feed zeros);
//   - NO pooler: pIncludePooler = True is rejected.
//
// ROBERTA (model_type "roberta") is the BERT skeleton with identical
// tensor names modulo the optional 'roberta.' prefix; the math differs in
// ONE place: position ids. RoBERTa's create_position_ids_from_input_ids
// starts real-token positions at padding_idx+1 = pad_token_id+1 = 2, so
// row p of the structural position table here is row p+2 of the
// checkpoint's position_embeddings - rows 0 and 1 (the padding-position
// rows) are NEVER read - and the usable context is
// max_position_embeddings - 2 (the famous 514-rows-for-512-positions
// layout). The importer loads the table with that row offset and caps
// SeqLen accordingly. This assumes NO padding in the input (positions are
// consecutive from 2), consistent with the no-padding-mask convention of
// this importer. type_vocab_size is 1 in real RoBERTa checkpoints: the
// token-type branch degenerates to a constant row added to every
// position (feed zeros in channel 1). RobertaModel exports carry a pooler
// (same as BERT, optional via pIncludePooler); lm_head-less RobertaModel
// is the supported export, like BertModel.
//
// =================== SEQUENCE CLASSIFICATION IMPORT =======================
//
// BuildBertForSequenceClassificationFromSafeTensors and
// BuildGPT2ForSequenceClassificationFromSafeTensors load FINE-TUNED
// classifier checkpoints (sentiment / NLI / toxicity heads), not just base
// LMs. Both stack a tiny head on the landed trunks; the POOLING convention
// differs by family and is the whole point of documenting them together:
//
//   - BertForSequenceClassification pools the FIRST token: HF computes
//     logits = classifier(dropout(pooler(h[:, 0]))), i.e. the dense+tanh
//     pooler over the [CLS] hidden state at POSITION 0, then the
//     classifier dense (classifier.{weight,bias}, [num_labels, hidden]).
//     Wired per token here (pooler + classifier as PointwiseConvLinear),
//     so the net outputs (SeqLen,1,num_labels) and ROW 0 carries HF's
//     class logits (the other rows have no HF meaning).
//     DistilBertForSequenceClassification and
//     RobertaForSequenceClassification pool the FIRST token too but use
//     DIFFERENT stacks (and NO pooler): distilbert computes
//     classifier(ReLU(pre_classifier(h[:, 0]))), roberta computes
//     classifier.out_proj(tanh(classifier.dense(h[:, 0]))) over the <s>
//     position. Same per-token wiring and ROW-0 convention here.
//   - GPT2ForSequenceClassification pools the LAST non-pad token: the
//     causal trunk only lets information flow left-to-right, so only the
//     final position has seen the whole text. HF applies score.weight
//     ([num_labels, n_embd], NO bias) to every hidden state and returns
//     the logits at the last non-pad position. Here the net outputs
//     per-position logits (SeqLen,1,num_labels); select row
//     RealTokens - 1 (= SeqLen-1 for full-length input) - the
//     SelectTokenLogits helper does exactly that.
//
// num_labels is inferred from the classifier/score weight shape.
// id2label: config.json's {"0": "neg", ...} map is parsed into an
// index-ordered TStringList (ReadId2LabelFromJSONFile, or the Id2Label
// argument of the ...Ex builders); ClassIndexToLabel maps an argmax class
// index to its label string ("LABEL_i" fallback, the HF default naming).
// BuildFromPretrained dispatches on config.json's "architectures" array:
// "BertForSequenceClassification" / "DistilBertForSequenceClassification"
// / "RobertaForSequenceClassification" / "GPT2ForSequenceClassification"
// route to these builders, anything else keeps the base-LM/encoder
// default.
//
// All importers FAIL LOUDLY (EPretrainedImportError) on missing tensors,
// unexpected tensors and shape mismatches.
//
// This unit is coded by Claude (AI).
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralsafetensors, neuraltorchbin,
  neuralhftokenizer;

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

// Opens the checkpoint at FileName with the reader matching its format:
// a ".bin" extension gets the restricted torch.save reader
// (TNNetTorchBinReader, neuraltorchbin.pas - pytorch_model.bin); anything
// else gets TNNetSafeTensorsReader (which itself dispatches ".json" to the
// sharded-index path). TNNetTorchBinReader subclasses the safetensors
// reader, so every Build*FromSafeTensors* importer in this unit accepts a
// pytorch_model.bin path transparently through this helper.
function CreatePretrainedTensorReader(
  const FileName: string): TNNetSafeTensorsReader;

// Builds a TNNet with the GPT-2 architecture described by the checkpoint at
// FileName and loads every weight. The returned net takes a (SeqLen,1,1)
// volume of token ids (as floats) and outputs (SeqLen,1,vocab) logits, one
// row per input position. pSeqLen = 0 uses the full n_ctx context;
// pNumHeads = 0 applies the n_embd/64 rule (see ReadGPT2Config).
// pInferenceOnly = True frees every layer's Delta/BackInertia training
// volumes DURING construction (TNNet.MakeInferenceOnly after each block),
// cutting peak and resident memory to ~1/3 - the returned net can only
// Compute(), never train. The full 124M GPT-2 then fits in well under 2 GB.
// pExactGelu = True swaps the MLP's gelu_new (the tanh approximation, the
// OpenAI GPT-2 activation_function) for the EXACT erf "gelu" composition -
// the Cerebras-GPT family (model_type "gpt2", the open GPT-3 reproduction)
// trains with activation_function "gelu" and needs it for logit parity.
function BuildGPT2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pNumHeads: integer = 0;
  pInferenceOnly: boolean = false; pExactGelu: boolean = false): TNNet;

// Same, also returning the inferred config.
// pSeqClsHead = True builds the GPT2ForSequenceClassification variant:
// the LM head is replaced by the score head (score.weight,
// [num_labels, n_embd], NO bias - HF defines no score bias) and the net
// outputs per-position class logits (SeqLen,1,num_labels); HF's logits
// are the LAST non-pad row (see the SEQUENCE CLASSIFICATION IMPORT
// section and SelectTokenLogits). num_labels comes from score.weight.
function BuildGPT2FromSafeTensorsEx(const FileName: string;
  out Config: TGPT2Config; pSeqLen: integer = 0;
  pNumHeads: integer = 0; pInferenceOnly: boolean = false;
  pSeqClsHead: boolean = false; pExactGelu: boolean = false): TNNet;

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
    ModelType: string;         // 'llama', 'mistral', 'qwen2' or 'qwen3'
    SlidingWindow: integer;    // sliding-window width; 0 = full attention
    QKVBias: boolean;          // q/k/v projections carry biases (Qwen2)
    HeadDim: integer;          // per-head width; 0 = hidden_size/num_heads
    QKNorm: boolean;           // per-head RMSNorm on q/k pre-RoPE (Qwen3)
    GegluFFN: boolean;         // gated-GELU (tanh) MLP instead of SwiGLU
                               // (Gemma: hidden_act gelu_pytorch_tanh)
    RMSNormAddOne: boolean;    // zero-centered RMSNorm: gain = 1 + w (Gemma)
    EmbedScale: TNeuralFloat;  // embedding-output scale folded into the
                               // embedding rows at load (Gemma: sqrt(d));
                               // 0 or 1 = off
    Prefix: string;            // tensor-name prefix ('model.' or '')
  end;

// Reads a HF Llama-family config.json (model_type 'llama', 'mistral',
// 'qwen2' or 'qwen3'). Required: hidden_size, intermediate_size,
// num_hidden_layers, num_attention_heads, vocab_size,
// max_position_embeddings. Defaults: num_key_value_heads =
// num_attention_heads, rms_norm_eps = 1e-6, rope_theta = 10000,
// tie_word_embeddings = false. An explicit head_dim is honored even when
// num_heads*head_dim <> hidden_size (Qwen3-0.6B: head_dim=128 with
// hidden=1024, heads=16); absent/null falls back to
// hidden_size/num_attention_heads (HeadDim stays 0). Family deltas:
//   mistral: SlidingWindow := sliding_window (null/absent = 0 = full
//            attention - many Mistral configs disable it);
//   qwen2:   QKVBias := true (q/k/v carry biases; o_proj does not); a
//            use_sliding_window=true config is rejected (Qwen2 windows
//            only its FIRST max_window_layers layers - not wired here);
//   qwen3:   QKNorm := true (per-head RMSNorm on q/k BEFORE RoPE, gain
//            shape [head_dim] shared across heads); QKVBias :=
//            attention_bias (default false - Qwen3 dropped the biases);
//            use_sliding_window=true is rejected like Qwen2;
//   gemma:   GegluFFN := true (gated-GELU MLP - hidden_act/hidden_activation
//            must be gelu / gelu_new / gelu_pytorch_tanh; older HF configs
//            say "gelu" but Gemma means the tanh approximation, transformers
//            special-cases this and so does this reader); RMSNormAddOne :=
//            true (Gemma applies (1+w)*xhat - the builder stores 1+w into
//            the TNNetTokenRMSNorm gains); EmbedScale := sqrt(hidden_size)
//            (Gemma scales the embedding OUTPUT - folded into the embedding
//            rows at load, NOT into the tied LM head); tie_word_embeddings
//            defaults TRUE (Gemma always ties); QKVBias := attention_bias
//            (default false);
//   llama:   QKVBias := attention_bias (default false).
// Fails on an unsupported model_type and on a non-null rope_scaling
// (long-context scaling is not wired here yet).
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

// Mistral: the Llama skeleton + optional sliding-window attention (config
// sliding_window; null/absent = full attention). Thin wrappers over the
// Llama path that ASSERT the config's model_type is 'mistral'.
function BuildMistralFromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;

function BuildMistralFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;

// Qwen2: the Llama skeleton + biases on the q/k/v projections (o_proj and
// the MLP stay bias-free). Thin wrappers over the Llama path that ASSERT
// the config's model_type is 'qwen2'.
function BuildQwen2FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;

function BuildQwen2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;

// Qwen3: the Llama skeleton WITHOUT q/k/v biases, with per-head RMSNorm on
// q and k (applied after the projection and BEFORE RoPE, gain [head_dim]
// shared across heads - q_norm.weight/k_norm.weight) and an optionally
// decoupled head_dim. Thin wrappers over the Llama path that ASSERT the
// config's model_type is 'qwen3'.
function BuildQwen3FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;

function BuildQwen3FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;

// Gemma 1: the Llama skeleton with three load-time deltas (no new layers):
// (a) the embedding OUTPUT is scaled by sqrt(hidden_size) - folded into the
//     embedding rows at load (the tied LM head keeps the UNSCALED rows, as
//     in HF GemmaForCausalLM);
// (b) zero-centered RMSNorm - Gemma computes (1 + w) * xhat, so 1 + w is
//     stored into every TNNetTokenRMSNorm gain at load;
// (c) gated-GELU MLP (GeGLU, hidden_act gelu_pytorch_tanh) - TNNetGEGLU
//     replaces TNNetSwiGLU in the FFN (same fused up|gate packing).
// Gemma-2B's multi-query attention (num_key_value_heads=1) and Gemma-7B's
// decoupled head_dim=256 ride the landed GQA/head_dim paths unchanged.
// Thin wrappers over the Llama path that ASSERT model_type is 'gemma'.
function BuildGemmaFromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;

function BuildGemmaFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;

type
  TGPTNeoConfig = record
    HiddenSize: integer;       // d_model (hidden_size)
    IntermediateSize: integer; // MLP width (intermediate_size; null = 4*d)
    NumLayers: integer;        // decoder blocks (num_layers)
    NumHeads: integer;         // attention heads (num_heads)
    VocabSize: integer;        // vocab_size
    MaxPositions: integer;     // max_position_embeddings (wpe rows)
    WindowSize: integer;       // local-attention band width (window_size)
    LayerNormEps: TNeuralFloat;// layer_norm_epsilon
    TieWordEmbeddings: boolean;// tie_word_embeddings (lm_head := wte)
    LayerIsLocal: array of boolean; // per-layer attention type (local=true)
    Prefix: string;            // tensor-name prefix ('transformer.' or '')
  end;

// Reads a HF GPT-Neo config.json (model_type 'gpt_neo'). Required:
// hidden_size, num_layers, num_heads, vocab_size, max_position_embeddings.
// Defaults: window_size = 256, layer_norm_epsilon = 1e-5,
// intermediate_size = 4 * hidden_size (when null/absent),
// tie_word_embeddings = true (the GPT-Neo convention). The per-layer
// global/local pattern is read from "attention_layers" (a flat list of
// 'global'/'local' strings) when present, else expanded from the packed
// "attention_types" form [[[types...], count], ...]; its length must equal
// num_layers. Prefix is left '' - the builder detects it.
function ReadGPTNeoConfigFromJSONFile(const FileName: string): TGPTNeoConfig;

function GPTNeoConfigToString(const Config: TGPTNeoConfig): string;

// Builds a TNNet with the GPT-Neo architecture described by the config and
// loads every weight from the safetensors checkpoint at FileName (see the
// GPT-NEO IMPORT section of the unit header). The net takes a (SeqLen,1,1)
// volume of token ids and outputs (SeqLen,1,vocab) logits. pSeqLen = 0 uses
// the full max_position_embeddings context. pInferenceOnly = True frees
// training volumes during construction (Compute()-only afterwards).
// Config.Prefix is detected from the checkpoint and written back.
function BuildGPTNeoFromSafeTensorsWithConfig(const FileName: string;
  var Config: TGPTNeoConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildGPTNeoFromSafeTensorsEx(const FileName: string;
  out Config: TGPTNeoConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;

function BuildGPTNeoFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;

type
  TGPTNeoXConfig = record
    HiddenSize: integer;        // d_model (hidden_size)
    IntermediateSize: integer;  // MLP width (intermediate_size)
    NumLayers: integer;         // decoder blocks (num_hidden_layers)
    NumHeads: integer;          // attention heads (num_attention_heads)
    VocabSize: integer;         // vocab_size
    MaxPositions: integer;      // max_position_embeddings (context length)
    LayerNormEps: TNeuralFloat; // layer_norm_eps
    RotaryPct: TNeuralFloat;    // rotary_pct (0.25 for Pythia; 1.0 = full)
    RopeTheta: TNeuralFloat;    // rotary_emb_base / rope_theta (RoPE base)
    UseParallelResidual: boolean; // use_parallel_residual (Pythia: true)
    TieWordEmbeddings: boolean; // tie_word_embeddings (Pythia: FALSE)
    HiddenActTanh: boolean;     // true = gelu_new/gelu_pytorch_tanh; false =
                                // exact erf "gelu" (the Pythia default)
    Prefix: string;             // tensor-name prefix ('gpt_neox.' or '')
  end;

// Reads a HF GPT-NeoX config.json (model_type 'gpt_neox'). Required:
// hidden_size, intermediate_size, num_hidden_layers, num_attention_heads,
// vocab_size, max_position_embeddings. Defaults: rotary_pct = 0.25 (also
// accepts the newer partial_rotary_factor spelling), rotary_emb_base =
// 10000 (rope_theta overrides when present), layer_norm_eps = 1e-5,
// use_parallel_residual = true, tie_word_embeddings = false (the GPT-NeoX
// convention - Pythia is UNTIED), hidden_act = 'gelu' (the exact erf form;
// gelu_new/gelu_pytorch_tanh select the tanh approximation, anything else
// is rejected). Prefix is left '' - the builder detects it.
function ReadGPTNeoXConfigFromJSONFile(const FileName: string): TGPTNeoXConfig;

function GPTNeoXConfigToString(const Config: TGPTNeoXConfig): string;

// Builds a TNNet with the GPT-NeoX architecture described by the config and
// loads every weight from the safetensors checkpoint at FileName (see the
// GPT-NEOX IMPORT section of the unit header). The net takes a (SeqLen,1,1)
// volume of token ids and outputs (SeqLen,1,vocab) logits. pSeqLen = 0 uses
// the full max_position_embeddings context. pInferenceOnly = True frees
// training volumes during construction (Compute()-only afterwards).
// Config.Prefix is detected from the checkpoint and written back.
function BuildGPTNeoXFromSafeTensorsWithConfig(const FileName: string;
  var Config: TGPTNeoXConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildGPTNeoXFromSafeTensorsEx(const FileName: string;
  out Config: TGPTNeoXConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;

function BuildGPTNeoXFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;

type
  TGPTJConfig = record
    HiddenSize: integer;        // d_model (n_embd)
    IntermediateSize: integer;  // MLP width (n_inner; null = 4*n_embd)
    NumLayers: integer;         // decoder blocks (n_layer)
    NumHeads: integer;          // attention heads (n_head)
    VocabSize: integer;         // vocab_size
    MaxPositions: integer;      // n_positions (context length)
    LayerNormEps: TNeuralFloat; // layer_norm_epsilon
    RotaryDim: integer;         // rotary_dim (64 of head_dim 256 for 6B)
    RopeTheta: TNeuralFloat;    // RoPE base (GPT-J hardcodes 10000)
    TieWordEmbeddings: boolean; // tie_word_embeddings (GPT-J: FALSE)
    HiddenActTanh: boolean;     // true = gelu_new/gelu_pytorch_tanh (the
                                // GPT-J default); false = exact erf "gelu"
    Prefix: string;             // tensor-name prefix ('transformer.' or '')
  end;

// Reads a HF GPT-J config.json (model_type 'gptj'). Required: n_embd,
// n_layer, n_head, vocab_size, n_positions. Defaults: n_inner = null ->
// 4*n_embd (the GPT-J-6B case), rotary_dim = 64 (the HF GPTJConfig
// default), layer_norm_epsilon = 1e-5, tie_word_embeddings = false,
// activation_function = 'gelu_new' (the tanh approximation, GPT-J's
// default; 'gelu' selects the exact erf form, anything else is rejected).
// Prefix is left '' - the builder detects it.
function ReadGPTJConfigFromJSONFile(const FileName: string): TGPTJConfig;

function GPTJConfigToString(const Config: TGPTJConfig): string;

// Builds a TNNet with the GPT-J architecture described by the config and
// loads every weight from the safetensors checkpoint at FileName (see the
// GPT-J IMPORT section of the unit header). The net takes a (SeqLen,1,1)
// volume of token ids and outputs (SeqLen,1,vocab) logits. pSeqLen = 0 uses
// the full n_positions context. pInferenceOnly = True frees training
// volumes during construction (Compute()-only afterwards).
// Config.Prefix is detected from the checkpoint and written back.
function BuildGPTJFromSafeTensorsWithConfig(const FileName: string;
  var Config: TGPTJConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildGPTJFromSafeTensorsEx(const FileName: string;
  out Config: TGPTJConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;

function BuildGPTJFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;

type
  TPhiConfig = record
    HiddenSize: integer;        // d_model (hidden_size)
    IntermediateSize: integer;  // MLP width (intermediate_size)
    NumLayers: integer;         // decoder blocks (num_hidden_layers)
    NumHeads: integer;          // attention heads (num_attention_heads)
    VocabSize: integer;         // vocab_size
    MaxPositions: integer;      // max_position_embeddings
    LayerNormEps: TNeuralFloat; // layer_norm_eps
    PartialRotaryFactor: TNeuralFloat; // partial_rotary_factor (0.5 for
                                // phi-1/phi-1_5, 0.4 for phi-2)
    RopeTheta: TNeuralFloat;    // rope_theta (RoPE base)
    TieWordEmbeddings: boolean; // tie_word_embeddings (Phi: FALSE)
    HiddenActTanh: boolean;     // true = gelu_new/gelu_pytorch_tanh (the
                                // Phi default); false = exact erf "gelu"
    Prefix: string;             // tensor-name prefix ('model.' or '')
  end;

// Reads a HF Phi config.json (model_type 'phi'). Required: hidden_size,
// intermediate_size, num_hidden_layers, num_attention_heads, vocab_size,
// max_position_embeddings. Defaults: partial_rotary_factor = 0.5 (the HF
// PhiConfig default), rope_theta = 10000, layer_norm_eps = 1e-5,
// tie_word_embeddings = false, hidden_act = 'gelu_new' (the tanh
// approximation, Phi's default; 'gelu' selects the exact erf form,
// anything else is rejected). qk_layernorm = true and GQA
// (num_key_value_heads <> num_attention_heads) are rejected. Prefix is
// left '' - the builder detects it.
function ReadPhiConfigFromJSONFile(const FileName: string): TPhiConfig;

function PhiConfigToString(const Config: TPhiConfig): string;

// Builds a TNNet with the Phi architecture described by the config and
// loads every weight from the safetensors checkpoint at FileName (see the
// PHI IMPORT section of the unit header). The net takes a (SeqLen,1,1)
// volume of token ids and outputs (SeqLen,1,vocab) logits. pSeqLen = 0 uses
// the full max_position_embeddings context. pInferenceOnly = True frees
// training volumes during construction (Compute()-only afterwards).
// Config.Prefix is detected from the checkpoint and written back.
function BuildPhiFromSafeTensorsWithConfig(const FileName: string;
  var Config: TPhiConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildPhiFromSafeTensorsEx(const FileName: string;
  out Config: TPhiConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;

function BuildPhiFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;

type
  // The encoder families sharing the BERT skeleton (selected by
  // config.json's model_type; see the BERT IMPORT section of the header).
  TBertFamily = (bfBert, bfDistilBert, bfRoberta);

  TBertConfig = record
    Family: TBertFamily;       // bert / distilbert / roberta (model_type)
    HiddenSize: integer;       // d_model (hidden_size; distilbert: dim)
    IntermediateSize: integer; // FFN width (intermediate_size; hidden_dim)
    NumLayers: integer;        // encoder blocks (num_hidden_layers; n_layers)
    NumHeads: integer;         // attention heads (num_attention_heads;
                               // distilbert: n_heads)
    VocabSize: integer;        // vocab_size
    MaxPositions: integer;     // max_position_embeddings (position table rows)
    TypeVocabSize: integer;    // token-type (segment) vocabulary (usually 2;
                               // 0 = NO token-type branch, the distilbert case)
    LayerNormEps: TNeuralFloat;// layer_norm_eps (BERT default 1e-12;
                               // distilbert hardcodes 1e-12)
    HiddenActTanh: boolean;    // true = gelu_new/gelu_pytorch_tanh (tanh
                               // approximation); false = exact erf "gelu"
    PositionOffset: integer;   // first position-table row actually used:
                               // 0 for bert/distilbert; pad_token_id+1 = 2
                               // for roberta (rows 0..1 are NEVER read and
                               // usable SeqLen is MaxPositions - 2)
    Prefix: string;            // tensor-name prefix ('' or 'bert.' /
                               // 'distilbert.' / 'roberta.')
  end;

// Reads a HF BERT-family config.json (model_type 'bert' or 'distilbert').
// bert - required: hidden_size, intermediate_size, num_hidden_layers,
// num_attention_heads, vocab_size, max_position_embeddings. Defaults:
// type_vocab_size = 2, layer_norm_eps = 1e-12, hidden_act = 'gelu' (the
// EXACT erf form; 'gelu_new'/'gelu_pytorch_tanh' select the tanh
// approximation, anything else is rejected).
// distilbert - required: dim, hidden_dim, n_layers, n_heads, vocab_size,
// max_position_embeddings; activation defaults to 'gelu'; TypeVocabSize is
// forced to 0 (no token-type embeddings) and LayerNormEps to 1e-12 (the
// value modeling_distilbert hardcodes).
// roberta - the bert keys, plus PositionOffset := pad_token_id (default 1)
// + 1: the first usable position-table row (see the header).
// Prefix is left '' - the builder detects it.
function ReadBertConfigFromJSONFile(const FileName: string): TBertConfig;

function BertConfigToString(const Config: TBertConfig): string;

// Builds a TNNet with the BERT ENCODER architecture described by Config and
// loads every weight from the safetensors checkpoint at FileName (see the
// BERT IMPORT section of the unit header). UNLIKE the decoder importers,
// the net takes a (SeqLen,1,2) volume - channel 0 = token ids, channel 1 =
// token-type (segment) ids, both as floats - and outputs the final HIDDEN
// STATES, a (SeqLen,1,hidden_size) volume (NOT logits; there is no LM
// head). Attention is BIDIRECTIONAL (non-causal) and every position
// attends to all SeqLen positions: there is no padding mask, so feed
// full-length sequences (or slice the net shorter via pSeqLen).
// pIncludePooler = True appends the BERT pooler head (dense + tanh,
// applied per token): the output stays (SeqLen,1,hidden_size) and ROW 0
// (the [CLS] position) equals HF's pooler_output; the other rows are
// tanh(dense(h_i)) and carry no HF meaning. pSeqLen = 0 uses the full
// max_position_embeddings context. pInferenceOnly = True frees training
// volumes during construction (Compute()-only afterwards).
// Config.Prefix is detected from the checkpoint and written back.
// Config.Family = bfDistilBert keeps the same math and input/output shapes
// but maps the DistilBERT tensor names, skips the token-type branch
// (channel 1 of the input is IGNORED - feed zeros) and rejects
// pIncludePooler (DistilBERT has no pooler).
// Config.Family = bfRoberta keeps the BERT names but loads the position
// table with Config.PositionOffset (= 2) rows skipped and caps SeqLen at
// MaxPositions - PositionOffset; checkpoint position rows 0..1 are NEVER
// read (see the ROBERTA paragraph of the unit header).
// pSeqClsHead = True builds the *ForSequenceClassification variant of the
// family: bert forces the pooler on and stacks the classifier dense
// (classifier.{weight,bias}, [num_labels, hidden]); distilbert stacks
// pre_classifier + ReLU + classifier (no pooler); roberta stacks
// classifier.dense + tanh + classifier.out_proj (no pooler). All wired
// per token, so the output is (SeqLen,1,num_labels) and ROW 0 (the
// [CLS] / <s> position) carries HF's class logits (see the SEQUENCE
// CLASSIFICATION IMPORT section). num_labels is inferred from the final
// classifier weight shape.
function BuildBertFromSafeTensorsWithConfig(const FileName: string;
  var Config: TBertConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pIncludePooler: boolean = false;
  pSeqClsHead: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildBertFromSafeTensorsEx(const FileName: string;
  out Config: TBertConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pIncludePooler: boolean = false;
  const ConfigFileName: string = ''): TNNet;

function BuildBertFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pIncludePooler: boolean = false): TNNet;

// ======================= SENTENCE EMBEDDINGS ===============================
// sentence-transformers style sentence vectors on top of the BERT encoder
// (e.g. sentence-transformers/all-MiniLM-L6-v2): tokenize, run the net,
// MEAN-pool the final hidden states over the REAL tokens only, L2-normalize.

// Tokenizes Text for the BERT encoder: WordPiece content ids wrapped in
// [CLS] ... [SEP] (matching HF encode(add_special_tokens=True) for a single
// segment). MaxTokens > 0 truncates the CONTENT so the result is at most
// MaxTokens ids ([CLS]/[SEP] always kept), like HF truncation. Raises
// EPretrainedImportError when the tokenizer has no [CLS]/[SEP] tokens.
function BertTokenizeSentence(Tokenizer: TNeuralHFTokenizer;
  const Text: string; MaxTokens: integer = 0): TNeuralIntegerArray;

// Attention-mask-aware pooling: Embedding := L2normalize(
// mean(HiddenStates rows 0..RealTokens-1) ), exactly the
// sentence-transformers "mean pooling + normalize" head. HiddenStates is
// the (SeqLen,1,hidden) output of a BuildBertFromSafeTensors net;
// RealTokens counts the non-[PAD] positions (incl. [CLS]/[SEP]), so [PAD]
// rows contribute NOTHING (deliberately NOT TNNetAvgChannel, which would
// average all SeqLen rows including pads). Embedding becomes (1,1,hidden),
// unit L2 norm (cosine similarity = plain dot product).
procedure BertPoolSentenceEmbedding(HiddenStates: TNNetVolume;
  RealTokens: integer; Embedding: TNNetVolume);

// Full pipeline: BertTokenizeSentence -> pad with [PAD] to the net's
// SeqLen (token-type channel all zeros) -> Net.Compute -> mean pooling
// over the real tokens -> L2 normalize into Embedding (1,1,hidden).
// IMPORTANT: the imported encoder carries NO attention padding mask, so
// real tokens DO attend to [PAD] rows when the sentence is shorter than
// the net. For exact sentence-transformers parity build the net with
// pSeqLen equal to the token count of THIS sentence (see
// examples/SemanticSearch, which caches one net per sentence length);
// padded calls return an approximation.
procedure BertEncodeSentence(Net: TNNet; Tokenizer: TNeuralHFTokenizer;
  const Text: string; Embedding: TNNetVolume);

// =================== SEQUENCE CLASSIFICATION IMPORT =======================
// Fine-tuned classifier checkpoints (see the unit-header section of the
// same name for the [CLS]-first vs last-token pooling difference).

// Reads config.json's id2label map ({"0": "neg", "1": "neu", ...}) into a
// NEW index-ordered TStringList (Result[i] = label of class i; caller
// frees). A missing id2label yields an EMPTY list (use ClassIndexToLabel
// for the LABEL_i fallback). Non-contiguous or duplicate ids fail loudly.
function ReadId2LabelFromJSONFile(const FileName: string): TStringList;

// Maps an argmax class index to its label string: Id2Label[ClassIndex]
// when available, else the HF default name 'LABEL_<index>' (also used when
// Id2Label is nil or empty).
function ClassIndexToLabel(Id2Label: TStringList;
  ClassIndex: integer): string;

// Copies the depth column at position TokenPos of a per-position output
// (SeqLen,1,Depth, SizeY must be 1) into Logits (resized to (1,1,Depth)).
// For BertForSequenceClassification pass TokenPos = 0 (the [CLS] row); for
// GPT2ForSequenceClassification pass the LAST non-pad position,
// RealTokens - 1 (= SeqLen - 1 for full-length input).
procedure SelectTokenLogits(NetOutput: TNNetVolume; TokenPos: integer;
  Logits: TNNetVolume);

// Builds the FULL *ForSequenceClassification stack of the BERT encoder
// family (model_type 'bert', 'distilbert' or 'roberta'): the encoder
// trunk plus the family's classifier head (bert: pooler dense+tanh +
// classifier; distilbert: pre_classifier + ReLU + classifier; roberta:
// classifier.dense + tanh + classifier.out_proj), loading every weight.
// Output (SeqLen,1,num_labels); ROW 0 (the [CLS] / <s> position) carries
// HF's class logits.
// Id2Label, when non-nil, is CLEARED and filled with config.json's
// id2label map (index-ordered; empty when the config has none).
function BuildBertForSequenceClassificationFromSafeTensorsEx(
  const FileName: string; out Config: TBertConfig; Id2Label: TStringList;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;

function BuildBertForSequenceClassificationFromSafeTensors(
  const FileName: string; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false): TNNet;

// Builds the FULL GPT2ForSequenceClassification stack: GPT-2 trunk + the
// bias-free score head instead of the LM head. Output is per-position
// class logits (SeqLen,1,num_labels); HF's logits are the LAST non-pad row
// - use SelectTokenLogits(Output, RealTokens - 1, ...). Id2Label, when
// non-nil, is CLEARED and filled from ConfigFileName ('' = "config.json"
// next to FileName; the file is only required when Id2Label <> nil - the
// trunk config itself is inferred from the tensors, see ReadGPT2Config).
function BuildGPT2ForSequenceClassificationFromSafeTensorsEx(
  const FileName: string; out Config: TGPT2Config; Id2Label: TStringList;
  pSeqLen: integer = 0; pNumHeads: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;

function BuildGPT2ForSequenceClassificationFromSafeTensors(
  const FileName: string; pSeqLen: integer = 0; pNumHeads: integer = 0;
  pInferenceOnly: boolean = false): TNNet;

// ---------------------------------------------------------------------------
// T5 / FLAN-T5 IMPORT (model_type "t5") - the FIRST ENCODER-DECODER import.
// ---------------------------------------------------------------------------
// A TNNet is a DAG grown from one primary input, so the seq2seq stack is
// built as TWO nets returned together by BuildT5FromSafeTensors:
//   - the ENCODER net: (EncSeqLen,1,1) token ids in ->
//     (EncSeqLen,1,d_model) final hidden states out (the bidirectional T5
//     encoder stack including its final RMSNorm);
//   - the DECODER net with TWO TNNetInput layers: Layers[0] takes the
//     (DecSeqLen,1,1) decoder token ids (rows usually start with the
//     decoder_start_token_id 0) and a SECOND TNNetInput (locate it with
//     T5EncoderStatesInput; fill its Output manually before Compute) takes
//     the (EncSeqLen,1,d_model) ENCODER hidden states, consumed by every
//     decoder block's cross-attention. Output: (DecSeqLen,1,vocab) logits.
// RunT5 runs the pair end-to-end (encoder Compute -> copy the hidden
// states into the decoder's second input -> decoder Compute).
//
// Per encoder block (pre-norm, residual adds AFTER the sublayer, NO biases
// in any linear):
//   x := x + o( SelfAttn_relpos( RMSNorm(x) ) )
//   x := x + FFN( RMSNorm(x) )
// Per decoder block:
//   x := x + o( CausalSelfAttn_relpos( RMSNorm(x) ) )
//   x := x + o( CrossAttn( RMSNorm(x), EncoderStates ) )
//   x := x + FFN( RMSNorm(x) )
// T5LayerNorm is scale-only RMSNorm (x/sqrt(mean(x^2)+eps)*w, no mean
// subtraction, no bias) = TNNetTokenRMSNorm, exactly.
//
// T5 quirks handled here:
//   - NO positional embedding; position enters ONLY through the BUCKETED
//     relative-position bias (TNNetT5RelPosBiasAttention, an exact port of
//     HF's relative_position_bucket). The bias table lives in block 0 of
//     each stack ([num_buckets, num_heads]) and is SHARED by all the
//     stack's layers: the loader copies column h of the stack's single
//     table into head h of EVERY layer (bidirectional buckets in the
//     encoder, causal buckets in the decoder; the decoder's
//     cross-attention has NO bias).
//   - attention WITHOUT the 1/sqrt(d_k) scaling (T5 folds it into the
//     pretraining init). TNNetT5RelPosBiasAttention / TNNetCrossAttention
//     scale by 1/sqrt(d_k), so the loader multiplies every q.weight by
//     sqrt(d_kv) to compensate (the GPT-Neo trick, in reverse).
//   - d_kv may be DECOUPLED from d_model/num_heads (inner_dim =
//     num_heads*d_kv need not equal d_model).
//   - FFN: feed_forward_proj "gated-gelu" (T5 v1.1 / Flan-T5) is
//     act(wi_0(x)) * wi_1(x) with the gelu_new TANH approximation - the
//     fused TNNetGEGLU projection holds wi_1 in neurons 0..d_ff-1 (linear
//     half) and wi_0 in neurons d_ff..2*d_ff-1 (gated half);
//     feed_forward_proj "relu" (the original T5 v1.0) is wo(relu(wi(x))).
//   - LM head: T5 v1.0 TIES lm_head to the shared embedding and scales
//     the decoder output by d_model^-0.5 before the head - the loader
//     folds the scale into the copied head rows. Flan-T5 / v1.1 does NOT
//     tie (config tie_word_embeddings=false) and loads lm_head.weight.
//   - encoder.embed_tokens / decoder.embed_tokens always alias
//     "shared.weight" (HF drops the duplicates from saved checkpoints;
//     legacy exports that still carry them are accepted and ignored).
// BuildFromPretrained does NOT dispatch "t5" (it returns ONE net); it
// raises an error pointing here instead. TOKENIZER NOTE: T5 ships a
// SentencePiece UNIGRAM tokenizer, which neuralhftokenizer.pas does not
// cover yet - end-to-end text use needs the Unigram tokenizer task; the
// importer itself is tokenizer-independent (token ids in, logits out).

type
  TT5Config = record
    DModel: integer;            // d_model (hidden width)
    DKV: integer;               // d_kv (per-head width; may be decoupled)
    DFF: integer;               // d_ff (FFN inner width)
    NumLayers: integer;         // num_layers (encoder blocks)
    NumDecoderLayers: integer;  // num_decoder_layers
    NumHeads: integer;          // num_heads
    VocabSize: integer;         // vocab_size
    RelPosNumBuckets: integer;  // relative_attention_num_buckets
    RelPosMaxDistance: integer; // relative_attention_max_distance
    LayerNormEps: TNeuralFloat; // layer_norm_epsilon
    GatedFFN: boolean;          // feed_forward_proj "gated-gelu" (Flan/v1.1)
    TieWordEmbeddings: boolean; // tie_word_embeddings (v1.0 ties + scales)
    ModelType: string;          // 't5'
  end;

// Reads a HF T5 config.json (model_type "t5"). Required: d_model, d_kv,
// d_ff, num_layers, num_heads, vocab_size. Defaults: num_decoder_layers =
// num_layers, relative_attention_num_buckets = 32,
// relative_attention_max_distance = 128, layer_norm_epsilon = 1e-6,
// feed_forward_proj = "relu", tie_word_embeddings = TRUE (the HF default -
// the original T5 ties; Flan-T5/v1.1 configs say false). Only
// feed_forward_proj "relu" and "gated-gelu" are supported (the two shipped
// T5 recipes).
function ReadT5ConfigFromJSONFile(const FileName: string): TT5Config;

function T5ConfigToString(const Config: TT5Config): string;

// Builds the T5 ENCODER and DECODER nets described by Config and loads
// every weight from the safetensors checkpoint at FileName (see the T5
// IMPORT section above for shapes and the two-input decoder convention).
// EncSeqLen/DecSeqLen fix the two sequence lengths at build time (T5 has
// no positional table, so any positive lengths are valid). Both nets are
// owned by the caller. pInferenceOnly = True frees training volumes during
// construction (Compute()-only afterwards).
procedure BuildT5FromSafeTensorsWithConfig(const FileName: string;
  var Config: TT5Config; out EncoderNet, DecoderNet: TNNet;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false);

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
procedure BuildT5FromSafeTensors(const FileName: string;
  out EncoderNet, DecoderNet: TNNet; out Config: TT5Config;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  const ConfigFileName: string = '');

// Returns the decoder net's SECOND TNNetInput - the (EncSeqLen,1,d_model)
// encoder-hidden-states input every cross-attention reads. Fill its
// Output with the encoder net's final output before DecoderNet.Compute
// (RunT5 does exactly that).
function T5EncoderStatesInput(DecoderNet: TNNet): TNNetLayer;

// Runs the encoder-decoder pair end-to-end: EncoderNet.Compute on the
// (EncSeqLen,1,1) EncoderTokens ids, copies the (EncSeqLen,1,d_model)
// hidden states into DecoderNet's second input, DecoderNet.Compute on the
// (DecSeqLen,1,1) DecoderTokens ids, and copies the (DecSeqLen,1,vocab)
// logits into Logits (resized as needed).
procedure RunT5(EncoderNet, DecoderNet: TNNet;
  EncoderTokens, DecoderTokens: TNNetVolume; Logits: TNNetVolume);

// AutoModel-style dispatch: reads config.json's model_type and routes to
// the right Build*FromSafeTensors builder. Path may be
//   - a checkpoint DIRECTORY (uses Path/config.json and
//     Path/model.safetensors.index.json if present, else
//     Path/model.safetensors), or
//   - a .safetensors / .safetensors.index.json FILE (config.json is read
//     from the same directory unless ConfigFileName overrides it).
// Supported model_types: gpt2 (n_head read from the config when present),
// gpt_neo, gpt_neox, gptj, phi, llama, mistral, qwen2, qwen3, bert,
// distilbert, roberta. Anything else
// raises EPretrainedImportError listing the supported types. The explicit
// builders stay public for callers that want a compile-time architecture
// choice. OUTPUT SEMANTICS DIFFER BY model_type: the decoder families
// return causal-LM nets ((SeqLen,1,1) token ids in, (SeqLen,1,vocab)
// logits out), while the bert family returns an ENCODER ((SeqLen,1,2)
// token|token-type ids in - channel 1 IGNORED for distilbert, zeros for
// roberta - (SeqLen,1,hidden_size) final hidden states out, pooler NOT
// included - call BuildBertFromSafeTensors directly for the pooler head;
// bert/roberta only). FINE-TUNED CLASSIFIERS: when config.json's
// "architectures" array names BertForSequenceClassification,
// DistilBertForSequenceClassification, RobertaForSequenceClassification
// (model_types bert/distilbert/roberta) or
// GPT2ForSequenceClassification (model_type gpt2), the dispatch
// routes to the classifier builders instead and the net outputs
// (SeqLen,1,num_labels) class logits (row 0 / last non-pad row, see the
// SEQUENCE CLASSIFICATION IMPORT section); the LM/encoder stays the
// default for every other architectures value.
function BuildFromPretrained(const Path: string; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;

// ---------------------------------------------------------------------------
// CLASSICAL PRETRAINED WORD EMBEDDINGS (GloVe / word2vec / fastText text
// format) - the classical-NLP counterpart of the checkpoint importers above:
// examples/Word2VecSkipGram TRAINS embeddings, this loads PUBLISHED ones.
// ---------------------------------------------------------------------------
// LoadPretrainedEmbedding parses the standard text format - one
// "word v1 v2 ... vD" line per word, whitespace-separated - including the
// optional "count dim" header line of word2vec / fastText .vec files (a
// first line of exactly two integers; the dim must match the layer), and
// copies every vector whose word is in the tokenizer's vocabulary into the
// matching TNNetEmbedding row (token id = TStringListInt.WordToIndex, so
// the tokenizer must be SORTED). Vocabulary words MISSING from the file are
// initialized to the MEAN of the matched vectors (a better prior than the
// random init they would otherwise keep; with zero matches every row stays
// untouched). Matching is CASE-SENSITIVE: GloVe ships lowercase vectors -
// lowercase your vocabulary to match - while word2vec / fastText publish
// cased entries. Vector width must equal Embedding.EmbeddingSize.
// Returns the number of vocabulary rows filled from the file.
// FreezeEmbedding = True sets the layer's LearningRate to 0 (the per-layer
// freeze convention of examples/LoRAFineTune). NOTE: TNNet.SetLearningRate
// sweeps EVERY layer, so re-apply the freeze after each call to it.
function LoadPretrainedEmbedding(const FileName: string;
  Embedding: TNNetEmbedding; Tokenizer: TStringListInt;
  FreezeEmbedding: boolean = false): integer;

implementation

procedure ImportError(const Msg: string);
begin
  raise EPretrainedImportError.Create(Msg);
end;

function CreatePretrainedTensorReader(
  const FileName: string): TNNetSafeTensorsReader;
begin
  if LowerCase(ExtractFileExt(FileName)) = '.bin' then
    Result := TNNetTorchBinReader.Create(FileName)
  else
    Result := TNNetSafeTensorsReader.Create(FileName);
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
  pNumHeads: integer = 0; pInferenceOnly: boolean = false;
  pSeqClsHead: boolean = false; pExactGelu: boolean = false): TNNet;
var
  GELUSource, PhiBranch: TNNetLayer;
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TGPT2BlockLayers;
  EmbeddingLayer, PosLayer, FinalLN, LMHead: TNNetLayer;
  BranchInput: TNNetLayer;
  BlockCnt, SeqLen, NumLabels, i, j: integer;
  Tmp: TNNetVolume;
  BlockPrefix, TensorNameStr: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

begin
  Reader := CreatePretrainedTensorReader(FileName);
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
        // MLP sub-block: x := x + c_proj(act(c_fc(ln_2(x)))), where act is
        // gelu_new (OpenAI GPT-2) or, with pExactGelu, the EXACT erf gelu
        // composed from existing layers like the BERT/GPT-NeoX paths
        // (Cerebras-GPT's activation_function "gelu").
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].LN2 := NN.AddLayer( TNNetTokenLayerNorm.Create(1e-5) );
        Blocks[BlockCnt].CFc := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(4 * Config.NEmbd) );
        if pExactGelu then
        begin
          GELUSource := NN.GetLastLayer();
          NN.AddLayerAfter(
            TNNetMulByConstant.Create(0.7071067811865476), GELUSource);
          NN.AddLayer( TNNetErf.Create() );
          NN.AddLayer( TNNetAddConstant.Create(1.0) );
          PhiBranch := NN.AddLayer( TNNetMulByConstant.Create(0.5) );
          NN.AddLayer( TNNetDeepConcat.Create([PhiBranch, GELUSource]) );
          NN.AddLayer( TNNetReGLU.Create() );
        end
        else
          NN.AddLayer( TNNetGELU.Create() ); // tanh approximation = gelu_new
        Blocks[BlockCnt].MlpProj := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.NEmbd) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
      end;
      FinalLN := NN.AddLayer( TNNetTokenLayerNorm.Create(1e-5) );
      NumLabels := 0;
      if pSeqClsHead then
      begin
        // GPT2ForSequenceClassification: the score head (bias-free
        // nn.Linear [num_labels, n_embd]) replaces the LM head; the net
        // outputs per-position class logits and HF's logits are the LAST
        // non-pad row (see the SEQUENCE CLASSIFICATION IMPORT section).
        if not Reader.HasTensor('score.weight') then
          ImportError('GPT-2 import: missing tensor "score.weight" - not ' +
            'a GPT2ForSequenceClassification checkpoint?');
        if (Reader.DimCount('score.weight') <> 2) or
           (Reader.DimSize('score.weight', 1) <> Config.NEmbd) then
          ImportError('GPT-2 import: "score.weight" must have shape ' +
            '[num_labels, ' + IntToStr(Config.NEmbd) + '] (nn.Linear ' +
            'stores [out, in]), got ' +
            Reader.ShapeAsString('score.weight'));
        NumLabels := Reader.DimSize('score.weight', 0);
        LMHead := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(NumLabels) );
      end
      else
        // LM head tied to wte: logits = h . wte^T. Implemented as an untied
        // PointwiseConvLinear(vocab) whose weights are a COPY of wte (see
        // the unit header). Bias-free in GPT-2: biases stay 0.
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
        if not pSeqClsHead then
        begin
          // wte -> LM head (tied weights copied; row t of wte = neuron t).
          for j := 0 to Config.VocabSize - 1 do
          begin
            for i := 0 to Config.NEmbd - 1 do
              LMHead.Neurons[j].Weights.FData[i] :=
                Tmp.FData[j * Config.NEmbd + i];
            LMHead.Neurons[j].BiasWeight := 0;
          end;
          LMHead.FlushWeightCache();
        end;
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
      if pSeqClsHead then
      begin
        // nn.Linear [out, in] row-major: row j -> neuron j. Bias-free by
        // HF definition (GPT2ForSequenceClassification's score has none).
        Tmp := TNNetVolume.Create;
        try
          Reader.LoadTensorFlat('score.weight', Tmp);
          for j := 0 to NumLabels - 1 do
          begin
            for i := 0 to Config.NEmbd - 1 do
              LMHead.Neurons[j].Weights.FData[i] :=
                Tmp.FData[j * Config.NEmbd + i];
            LMHead.Neurons[j].BiasWeight := 0;
          end;
          LMHead.FlushWeightCache();
        finally
          Tmp.Free;
        end;
        MarkConsumed('score.weight');
      end;

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
  pInferenceOnly: boolean = false; pExactGelu: boolean = false): TNNet;
var
  IgnoredConfig: TGPT2Config;
begin
  Result := BuildGPT2FromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pNumHeads, pInferenceOnly, {pSeqClsHead=}false, pExactGelu);
end;

// ============================ LLAMA IMPORT =================================

function ReadLlamaConfigFromJSONFile(const FileName: string): TLlamaConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, HiddenAct: string;
  RopeScaling, HeadDimField, SlidingWindowField: TJSONData;

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
    if (ModelType <> 'llama') and (ModelType <> 'mistral') and
       (ModelType <> 'qwen2') and (ModelType <> 'qwen3') and
       (ModelType <> 'gemma') then
      ImportError('Llama import: config model_type is "' + ModelType +
        '" - only "llama", "mistral", "qwen2", "qwen3" and "gemma" are ' +
        'supported here (see BuildFromPretrained for the full dispatch).');
    RopeScaling := Obj.Find('rope_scaling');
    if (RopeScaling <> nil) and not RopeScaling.IsNull then
      ImportError('Llama import: config carries a non-null "rope_scaling" - ' +
        'long-context RoPE scaling is not wired into this importer yet.');
    Result.ModelType := ModelType;
    Result.HiddenSize := RequiredInt('hidden_size');
    Result.IntermediateSize := RequiredInt('intermediate_size');
    Result.NumLayers := RequiredInt('num_hidden_layers');
    Result.NumHeads := RequiredInt('num_attention_heads');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.MaxPositions := RequiredInt('max_position_embeddings');
    Result.NumKVHeads := Obj.Get('num_key_value_heads', Result.NumHeads);
    Result.RmsNormEps := Obj.Get('rms_norm_eps', 1.0e-6);
    Result.RopeTheta := Obj.Get('rope_theta', 10000.0);
    // Gemma ALWAYS ties the LM head to the embedding (the HF GemmaConfig
    // default is tie_word_embeddings=true); the other families default off.
    Result.TieWordEmbeddings :=
      Obj.Get('tie_word_embeddings', ModelType = 'gemma');
    // An explicit head_dim is honored even when it is DECOUPLED from
    // hidden_size/num_attention_heads (Qwen3-0.6B: head_dim=128 with
    // hidden=1024, heads=16). Absent/null leaves HeadDim=0 and the builder
    // falls back to hidden_size div num_attention_heads.
    Result.HeadDim := 0;
    HeadDimField := Obj.Find('head_dim');
    if (HeadDimField <> nil) and not HeadDimField.IsNull then
    begin
      Result.HeadDim := HeadDimField.AsInteger;
      if Result.HeadDim < 1 then
        ImportError('Llama import: config head_dim must be a positive ' +
          'integer or null, got ' + HeadDimField.AsJSON + '.');
    end;
    // ---- family deltas ----
    Result.SlidingWindow := 0;
    Result.QKVBias := False;
    Result.QKNorm := False;
    Result.GegluFFN := False;
    Result.RMSNormAddOne := False;
    Result.EmbedScale := 1.0;
    if ModelType = 'mistral' then
    begin
      // Many Mistral configs ship sliding_window=null (full attention).
      SlidingWindowField := Obj.Find('sliding_window');
      if (SlidingWindowField <> nil) and not SlidingWindowField.IsNull then
      begin
        Result.SlidingWindow := SlidingWindowField.AsInteger;
        if Result.SlidingWindow < 1 then
          ImportError('Llama import: config sliding_window must be a ' +
            'positive integer or null, got ' + SlidingWindowField.AsJSON + '.');
      end;
    end
    else if ModelType = 'qwen2' then
    begin
      Result.QKVBias := True; // Qwen2 q/k/v carry biases; o_proj does not
      if Obj.Get('use_sliding_window', False) then
        ImportError('Llama import: Qwen2 use_sliding_window=true (per-layer ' +
          'max_window_layers windowing) is not wired into this importer yet.');
    end
    else if ModelType = 'qwen3' then
    begin
      // Qwen3 dropped the Qwen2 q/k/v biases (attention_bias defaults to
      // false) and added per-head RMSNorm on q/k before RoPE.
      Result.QKVBias := Obj.Get('attention_bias', False);
      Result.QKNorm := True;
      if Obj.Get('use_sliding_window', False) then
        ImportError('Llama import: Qwen3 use_sliding_window=true (per-layer ' +
          'max_window_layers windowing) is not wired into this importer yet.');
    end
    else if ModelType = 'gemma' then
    begin
      // Gemma-1 deltas (all load-time; see the BuildGemmaFromSafeTensors
      // interface comment): gated-GELU MLP, zero-centered RMSNorm and the
      // sqrt(d_model) embedding-output scale.
      Result.GegluFFN := True;
      Result.RMSNormAddOne := True;
      Result.EmbedScale := Sqrt(Result.HiddenSize);
      Result.QKVBias := Obj.Get('attention_bias', False);
      // The newer "hidden_activation" key wins over the legacy "hidden_act".
      // Older HF Gemma configs say "gelu" but Gemma means the TANH
      // approximation (gelu_pytorch_tanh) - transformers special-cases this
      // and so does this reader: every accepted spelling maps to TNNetGEGLU
      // (tanh). Anything non-GELU is rejected.
      HiddenAct := Obj.Get('hidden_activation', '');
      if HiddenAct = '' then
        HiddenAct := Obj.Get('hidden_act', 'gelu_pytorch_tanh');
      if (HiddenAct <> 'gelu') and (HiddenAct <> 'gelu_new') and
         (HiddenAct <> 'gelu_pytorch_tanh') then
        ImportError('Llama import: Gemma hidden activation "' + HiddenAct +
          '" is not supported - expected gelu / gelu_new / ' +
          'gelu_pytorch_tanh (Gemma''s MLP is gated GELU).');
    end
    else // llama
      Result.QKVBias := Obj.Get('attention_bias', False);
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function LlamaConfigToString(const Config: TLlamaConfig): string;
begin
  if Config.ModelType = '' then Result := 'llama' else Result := Config.ModelType;
  Result := Result + ' config: layers=' + IntToStr(Config.NumLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', kv_heads=' + IntToStr(Config.NumKVHeads) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', max_pos=' + IntToStr(Config.MaxPositions) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', rms_eps=' + FloatToStr(Config.RmsNormEps) +
    ', rope_theta=' + FloatToStr(Config.RopeTheta) +
    ', tied=' + BoolToStr(Config.TieWordEmbeddings, true);
  if Config.SlidingWindow > 0 then
    Result := Result + ', sliding_window=' + IntToStr(Config.SlidingWindow);
  if Config.HeadDim > 0 then
    Result := Result + ', head_dim=' + IntToStr(Config.HeadDim);
  if Config.QKVBias then
    Result := Result + ', qkv_bias=true';
  if Config.QKNorm then
    Result := Result + ', qk_norm=true';
  if Config.GegluFFN then
    Result := Result + ', geglu_ffn=true';
  if Config.RMSNormAddOne then
    Result := Result + ', rmsnorm_add_one=true';
  if (Config.EmbedScale <> 0) and (Config.EmbedScale <> 1.0) then
    Result := Result + ', embed_scale=' + FloatToStr(Config.EmbedScale);
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

// Loads a HF RMSNorm gain vector [d_model] into a TNNetTokenRMSNorm.
// GainOffset is added to every stored gain: Gemma's zero-centered RMSNorm
// computes (1 + w) * xhat, so the Gemma path folds the +1 in at load
// (GainOffset = 1) with zero layer changes.
procedure LoadLlamaRMSNormWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName: string; d_model: integer;
  GainOffset: TNeuralFloat = 0);
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
      Layer.Neurons[0].Weights.FData[i] := GainOffset + Tmp.FData[i]; // gain
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
// BiasName <> '' loads a [OutDim] bias vector too (the Qwen2 q/k/v case);
// the bias follows the SAME row permutation as the weights, so a permuted
// q/k projection keeps bias j attached to its output channel.
// RotaryHeadDim > 0 additionally PERMUTES the output rows within each
// RotaryHeadDim-wide head to convert HF's rotate_half (first-half /
// second-half) pair layout into TNNetRotaryEmbedding's interleaved
// (even/odd) pair layout (see the unit header):
//   target channel (h*hd + 2k)     <- HF row (h*hd + k)
//   target channel (h*hd + 2k + 1) <- HF row (h*hd + k + hd/2)
// Scale multiplies every loaded weight (and bias): the GPT-Neo importer
// folds the missing 1/sqrt(d_head) attention scaling into W_q with it.
procedure LoadLlamaLinearWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName: string; InDim, OutDim: integer;
  NeuronBase: integer = 0; ExpectedNeurons: integer = -1;
  RotaryHeadDim: integer = 0; const BiasName: string = '';
  Scale: TNeuralFloat = 1.0);
var
  W, B: TNNetVolume;
  i, j, TargetIdx, HeadIdx, RowInHead, HalfDim: integer;
begin
  if ExpectedNeurons < 0 then ExpectedNeurons := OutDim;
  if not Reader.HasTensor(WName) then
    ImportError('Llama import: missing tensor "' + WName + '".');
  if BiasName <> '' then
  begin
    if not Reader.HasTensor(BiasName) then
      ImportError('Llama import: missing tensor "' + BiasName + '".');
    if (Reader.DimCount(BiasName) <> 1) or
       (Reader.DimSize(BiasName, 0) <> OutDim) then
      ImportError('Llama import: "' + BiasName + '" must have shape [' +
        IntToStr(OutDim) + '], got ' + Reader.ShapeAsString(BiasName));
  end;
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
  B := nil;
  try
    Reader.LoadTensorFlat(WName, W);
    if BiasName <> '' then
    begin
      B := TNNetVolume.Create;
      Reader.LoadTensorFlat(BiasName, B);
    end;
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
        Layer.Neurons[TargetIdx].Weights.FData[i] :=
          Scale * W.FData[j * InDim + i];
      if B <> nil then
        Layer.Neurons[TargetIdx].BiasWeight := Scale * B.FData[j]
      else
        Layer.Neurons[TargetIdx].BiasWeight := 0; // bias-free Linear
    end;
  finally
    B.Free;
    W.Free;
  end;
  Layer.FlushWeightCache();
end;

// Loads a Qwen3-style SHARED per-head RMSNorm gain vector [HeadDim] into
// every per-head TNNetTokenRMSNorm copy in NormLayers (q_norm/k_norm are
// stored ONCE in the checkpoint but applied to each head's q/k slice).
// The per-head q/k slices live in the rotate_half-PERMUTED channel order
// produced by LoadLlamaLinearWeights (RotaryHeadDim), so the gain is
// permuted the same way: target channel 2k <- HF position k, target
// channel 2k+1 <- HF position k + HeadDim/2. The RMS denominator itself is
// permutation-invariant (mean of squares over the whole head).
procedure LoadLlamaHeadRMSNormWeights(Reader: TNNetSafeTensorsReader;
  const NormLayers: array of TNNetLayer; const WName: string;
  HeadDim: integer);
var
  Tmp: TNNetVolume;
  HeadCnt, j, TargetIdx, HalfDim: integer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('Llama import: missing tensor "' + WName + '".');
  if (Reader.DimCount(WName) <> 1) or
     (Reader.DimSize(WName, 0) <> HeadDim) then
    ImportError('Llama import: "' + WName + '" must have shape [' +
      IntToStr(HeadDim) + '], got ' + Reader.ShapeAsString(WName));
  HalfDim := HeadDim div 2;
  Tmp := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, Tmp);
    for HeadCnt := Low(NormLayers) to High(NormLayers) do
    begin
      for j := 0 to HeadDim - 1 do
      begin
        if j < HalfDim then
          TargetIdx := 2 * j
        else
          TargetIdx := 2 * (j - HalfDim) + 1;
        NormLayers[HeadCnt].Neurons[0].Weights.FData[TargetIdx] :=
          Tmp.FData[j]; // gain
      end;
      NormLayers[HeadCnt].FlushWeightCache();
    end;
  finally
    Tmp.Free;
  end;
end;

type
  TLlamaBlockLayers = record
    AttnNorm, QProj, KProj, VProj, OProj, MlpNorm, GateUp, Down: TNNetLayer;
    // Per-head q/k RMSNorm copies (Qwen3 QKNorm); empty otherwise.
    QNorms, KNorms: array of TNNetLayer;
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
  HeadDim, QWidth, KVWidth, i, j, d: integer;
  NormGainOffset: TNeuralFloat;
  Tmp: TNNetVolume;
  BlockPrefix, TensorNameStr, LMHeadName: string;
  QBiasName, KBiasName, VBiasName: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

begin
  Reader := CreatePretrainedTensorReader(FileName);
  NN := nil;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      // ---------------- Config validation & prefix detection -------------
      if Config.SlidingWindow < 0 then
        ImportError('Llama import: SlidingWindow must be >= 0 ' +
          '(0 = full attention).');
      if Config.NumHeads < 1 then
        ImportError('Llama import: num_attention_heads must be >= 1.');
      if Config.NumKVHeads < 1 then
        ImportError('Llama import: num_key_value_heads must be >= 1.');
      if (Config.NumHeads mod Config.NumKVHeads) <> 0 then
        ImportError('Llama import: num_attention_heads=' +
          IntToStr(Config.NumHeads) + ' is not divisible by ' +
          'num_key_value_heads=' + IntToStr(Config.NumKVHeads) + '.');
      // An explicit config head_dim wins (it may be DECOUPLED from
      // hidden_size/num_attention_heads, e.g. Qwen3-0.6B); otherwise the
      // classic hidden_size div num_attention_heads applies.
      if Config.HeadDim > 0 then
        HeadDim := Config.HeadDim
      else
      begin
        if (Config.HiddenSize mod Config.NumHeads) <> 0 then
          ImportError('Llama import: hidden_size=' +
            IntToStr(Config.HiddenSize) + ' is not divisible by ' +
            'num_attention_heads=' + IntToStr(Config.NumHeads) + '.');
        HeadDim := Config.HiddenSize div Config.NumHeads;
      end;
      if Odd(HeadDim) then
        ImportError('Llama import: head_dim=' + IntToStr(HeadDim) +
          ' must be even (RoPE rotates channel pairs).');
      QWidth := Config.NumHeads * HeadDim;
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
          TNNetPointwiseConvLinear.Create(QWidth), NormedSource);
        Blocks[BlockCnt].KProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(KVWidth), NormedSource);
        Blocks[BlockCnt].VProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(KVWidth), NormedSource);
        // Config.QKNorm (Qwen3): per-head RMSNorm on each q/k slice AFTER
        // the projection and BEFORE RoPE (the HF modeling_qwen3 ordering:
        // q_norm(q_proj(x)) then apply_rotary_pos_emb). One
        // TNNetTokenRMSNorm copy per head; the shared [head_dim] gain is
        // loaded into every copy below.
        if Config.QKNorm then
        begin
          SetLength(Blocks[BlockCnt].QNorms, Config.NumHeads);
          SetLength(Blocks[BlockCnt].KNorms, Config.NumKVHeads);
        end;
        // K is rotated ONCE per KV head; V is never rotated.
        for KVHeadCnt := 0 to Config.NumKVHeads - 1 do
        begin
          for d := 0 to HeadDim - 1 do
            SliceChannels[d] := KVHeadCnt * HeadDim + d;
          KSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(SliceChannels), Blocks[BlockCnt].KProj);
          if Config.QKNorm then
          begin
            Blocks[BlockCnt].KNorms[KVHeadCnt] := NN.AddLayerAfter(
              TNNetTokenRMSNorm.Create(Config.RmsNormEps), KSlice);
            KSlice := Blocks[BlockCnt].KNorms[KVHeadCnt];
          end;
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
          if Config.QKNorm then
          begin
            Blocks[BlockCnt].QNorms[HeadCnt] := NN.AddLayerAfter(
              TNNetTokenRMSNorm.Create(Config.RmsNormEps), QSlice);
            QSlice := Blocks[BlockCnt].QNorms[HeadCnt];
          end;
          QSlice := NN.AddLayerAfter(
            TNNetRotaryEmbedding.Create(Config.RopeTheta), QSlice);
          // Pack [Q_h | K_group | V_group] (width 3*head_dim) for SDPA.
          HeadPack := NN.AddLayer( TNNetDeepConcat.Create(
            [QSlice, KRotated[KVGroup], VSlices[KVGroup]]) );
          // Config.SlidingWindow > 0 (Mistral) applies the same banded
          // causal sliding-window mask HF uses: query i attends keys j with
          // i - j < W (and j <= i from the causal mask). 0 = full attention.
          HeadOutputs[HeadCnt] := NN.AddLayerAfter(
            TNNetScaledDotProductAttention.Create(HeadDim, {CausalMask=}true,
              {pWindow=}Config.SlidingWindow),
            HeadPack);
        end;
        NN.AddLayer( TNNetDeepConcat.Create(HeadOutputs) );
        Blocks[BlockCnt].OProj := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        // MLP sub-block: x := x + down(act(gate(h)) * up(h)), h = RMSNorm(x),
        // where act is SiLU (SwiGLU, the Llama default) or tanh-GELU (GeGLU,
        // Gemma's gelu_pytorch_tanh - Config.GegluFFN). Both TNNetSwiGLU and
        // TNNetGEGLU compute FIRSTHALF * act(SECONDHALF), so the fused
        // projection holds up_proj in neurons 0..I-1 and gate_proj in
        // neurons I..2I-1 (see LoadLlamaLinearWeights calls below) for
        // either activation.
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].MlpNorm :=
          NN.AddLayer( TNNetTokenRMSNorm.Create(Config.RmsNormEps) );
        Blocks[BlockCnt].GateUp := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(2 * Config.IntermediateSize) );
        if Config.GegluFFN then
          NN.AddLayer( TNNetGEGLU.Create() )
        else
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
      // Config.RMSNormAddOne (Gemma): every RMSNorm gain is stored as 1 + w
      // (zero-centered checkpoint weights) - folded in by the loader.
      if Config.RMSNormAddOne then NormGainOffset := 1.0
      else NormGainOffset := 0;
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
        // Config.EmbedScale (Gemma: sqrt(hidden_size)) scales the embedding
        // OUTPUT; TNNetEmbedding's ScaleEmbedding is init-only, so the scale
        // is folded into the embedding ROWS instead. ONLY the embedding copy
        // is scaled - the tied LM head below reads the UNSCALED Tmp rows,
        // matching HF GemmaForCausalLM (the head ties to the raw matrix).
        if (Config.EmbedScale <> 0) and (Config.EmbedScale <> 1.0) then
          EmbeddingLayer.Neurons[0].Weights.Mul(Config.EmbedScale);
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
          BlockPrefix + 'input_layernorm.weight', Config.HiddenSize,
          NormGainOffset);
        MarkConsumed(BlockPrefix + 'input_layernorm.weight');
        // q_proj/k_proj rows are PERMUTED per head for the rotate_half
        // convention (RotaryHeadDim); v_proj/o_proj load straight.
        // Config.QKVBias (Qwen2) loads q/k/v biases too - permuted along
        // with their rows for q/k; o_proj stays bias-free in all families.
        if Config.QKVBias then
        begin
          QBiasName := BlockPrefix + 'self_attn.q_proj.bias';
          KBiasName := BlockPrefix + 'self_attn.k_proj.bias';
          VBiasName := BlockPrefix + 'self_attn.v_proj.bias';
        end
        else
        begin
          QBiasName := '';
          KBiasName := '';
          VBiasName := '';
        end;
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].QProj,
          BlockPrefix + 'self_attn.q_proj.weight',
          Config.HiddenSize, QWidth, 0, -1, HeadDim, QBiasName);
        MarkConsumed(BlockPrefix + 'self_attn.q_proj.weight');
        if QBiasName <> '' then MarkConsumed(QBiasName);
        // Qwen3 per-head q/k RMSNorm: one shared [head_dim] gain per block,
        // copied into every per-head norm (rotate_half-permuted to match
        // the permuted q/k channel order).
        if Config.QKNorm then
        begin
          LoadLlamaHeadRMSNormWeights(Reader, Blocks[BlockCnt].QNorms,
            BlockPrefix + 'self_attn.q_norm.weight', HeadDim);
          MarkConsumed(BlockPrefix + 'self_attn.q_norm.weight');
          LoadLlamaHeadRMSNormWeights(Reader, Blocks[BlockCnt].KNorms,
            BlockPrefix + 'self_attn.k_norm.weight', HeadDim);
          MarkConsumed(BlockPrefix + 'self_attn.k_norm.weight');
        end;
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].KProj,
          BlockPrefix + 'self_attn.k_proj.weight',
          Config.HiddenSize, KVWidth, 0, -1, HeadDim, KBiasName);
        MarkConsumed(BlockPrefix + 'self_attn.k_proj.weight');
        if KBiasName <> '' then MarkConsumed(KBiasName);
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].VProj,
          BlockPrefix + 'self_attn.v_proj.weight',
          Config.HiddenSize, KVWidth, 0, -1, 0, VBiasName);
        MarkConsumed(BlockPrefix + 'self_attn.v_proj.weight');
        if VBiasName <> '' then MarkConsumed(VBiasName);
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].OProj,
          BlockPrefix + 'self_attn.o_proj.weight',
          QWidth, Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'self_attn.o_proj.weight');
        LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].MlpNorm,
          BlockPrefix + 'post_attention_layernorm.weight', Config.HiddenSize,
          NormGainOffset);
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
        Config.Prefix + 'norm.weight', Config.HiddenSize, NormGainOffset);
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

// ============================ GPT-NEO IMPORT ===============================

function ReadGPTNeoConfigFromJSONFile(const FileName: string): TGPTNeoConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, TypeName: string;
  Field, TypesField: TJSONData;
  LayersArr, TypesArr, PairArr, NamesArr: TJSONArray;
  i, i2, j, RepeatCnt, LayerCnt: integer;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('GPT-Neo import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('GPT-Neo import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

  procedure AddLayerType(const AName: string);
  begin
    if (AName <> 'global') and (AName <> 'local') then
      ImportError('GPT-Neo import: config attention layer type "' + AName +
        '" is not supported - only "global" and "local" exist.');
    if LayerCnt >= Result.NumLayers then
      ImportError('GPT-Neo import: config lists more attention layer ' +
        'types than num_layers=' + IntToStr(Result.NumLayers) + '.');
    Result.LayerIsLocal[LayerCnt] := (AName = 'local');
    Inc(LayerCnt);
  end;

begin
  if not FileExists(FileName) then
    ImportError('GPT-Neo import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('GPT-Neo import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('GPT-Neo import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'gpt_neo');
    if ModelType <> 'gpt_neo' then
      ImportError('GPT-Neo import: config model_type is "' + ModelType +
        '" - expected "gpt_neo" (see BuildFromPretrained for the full ' +
        'dispatch).');
    Result.HiddenSize := RequiredInt('hidden_size');
    Result.NumLayers := RequiredInt('num_layers');
    Result.NumHeads := RequiredInt('num_heads');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.MaxPositions := RequiredInt('max_position_embeddings');
    Result.WindowSize := Obj.Get('window_size', 256);
    if Result.WindowSize < 1 then
      ImportError('GPT-Neo import: config window_size must be >= 1, got ' +
        IntToStr(Result.WindowSize) + '.');
    Result.LayerNormEps := Obj.Get('layer_norm_epsilon', 1.0e-5);
    Result.TieWordEmbeddings := Obj.Get('tie_word_embeddings', True);
    // intermediate_size is null in every GPT-Neo config in the wild: the
    // HF default 4 * hidden_size applies.
    Field := Obj.Find('intermediate_size');
    if (Field = nil) or Field.IsNull then
      Result.IntermediateSize := 4 * Result.HiddenSize
    else
      Result.IntermediateSize := RequiredInt('intermediate_size');
    // Per-layer global/local pattern: the flat "attention_layers" list
    // when present, else the packed "attention_types" [[[t..], n], ...].
    SetLength(Result.LayerIsLocal, Result.NumLayers);
    LayerCnt := 0;
    Field := Obj.Find('attention_layers');
    if (Field <> nil) and (Field is TJSONArray) then
    begin
      LayersArr := TJSONArray(Field);
      for i := 0 to LayersArr.Count - 1 do
        AddLayerType(LayersArr.Items[i].AsString);
    end
    else
    begin
      TypesField := Obj.Find('attention_types');
      if (TypesField = nil) or not (TypesField is TJSONArray) then
        ImportError('GPT-Neo import: config "' + FileName + '" carries ' +
          'neither "attention_layers" nor "attention_types".');
      TypesArr := TJSONArray(TypesField);
      for i := 0 to TypesArr.Count - 1 do
      begin
        if not (TypesArr.Items[i] is TJSONArray) or
           (TJSONArray(TypesArr.Items[i]).Count <> 2) then
          ImportError('GPT-Neo import: malformed attention_types entry ' +
            TypesArr.Items[i].AsJSON + ' (expected [[types...], count]).');
        PairArr := TJSONArray(TypesArr.Items[i]);
        if not (PairArr.Items[0] is TJSONArray) then
          ImportError('GPT-Neo import: malformed attention_types entry ' +
            PairArr.AsJSON + ' (expected [[types...], count]).');
        NamesArr := TJSONArray(PairArr.Items[0]);
        RepeatCnt := PairArr.Items[1].AsInteger;
        if RepeatCnt < 1 then
          ImportError('GPT-Neo import: attention_types repeat count must ' +
            'be >= 1, got ' + PairArr.Items[1].AsJSON + '.');
        for i2 := 1 to RepeatCnt do
          for j := 0 to NamesArr.Count - 1 do
            AddLayerType(NamesArr.Items[j].AsString);
      end;
    end;
    if LayerCnt <> Result.NumLayers then
      ImportError('GPT-Neo import: config lists ' + IntToStr(LayerCnt) +
        ' attention layer types but num_layers=' +
        IntToStr(Result.NumLayers) + '.');
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function GPTNeoConfigToString(const Config: TGPTNeoConfig): string;
var
  i: integer;
begin
  Result := 'gpt_neo config: layers=' + IntToStr(Config.NumLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', max_pos=' + IntToStr(Config.MaxPositions) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', window=' + IntToStr(Config.WindowSize) +
    ', ln_eps=' + FloatToStr(Config.LayerNormEps) +
    ', tied=' + BoolToStr(Config.TieWordEmbeddings, true) +
    ', attention=[';
  for i := 0 to High(Config.LayerIsLocal) do
  begin
    if i > 0 then Result := Result + ',';
    if Config.LayerIsLocal[i] then Result := Result + 'local'
    else Result := Result + 'global';
  end;
  Result := Result + ']';
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

type
  TGPTNeoBlockLayers = record
    LN1, QKV, AttnProj, LN2, CFc, MlpProj: TNNetLayer;
  end;

function BuildGPTNeoFromSafeTensorsWithConfig(const FileName: string;
  var Config: TGPTNeoConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TGPTNeoBlockLayers;
  EmbeddingLayer, PosLayer, FinalLN, LMHead: TNNetLayer;
  BranchInput: TNNetLayer;
  BlockCnt, SeqLen, HeadDim, Window, i, j: integer;
  QScale: TNeuralFloat;
  Tmp: TNNetVolume;
  BlockPrefix, AttnPrefix, TensorNameStr: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

begin
  Reader := CreatePretrainedTensorReader(FileName);
  NN := nil;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      // ---------------- Config validation & prefix detection -------------
      if Config.NumHeads < 1 then
        ImportError('GPT-Neo import: num_heads must be >= 1.');
      if (Config.HiddenSize mod Config.NumHeads) <> 0 then
        ImportError('GPT-Neo import: hidden_size=' +
          IntToStr(Config.HiddenSize) + ' is not divisible by num_heads=' +
          IntToStr(Config.NumHeads) + '.');
      if Length(Config.LayerIsLocal) <> Config.NumLayers then
        ImportError('GPT-Neo import: LayerIsLocal lists ' +
          IntToStr(Length(Config.LayerIsLocal)) +
          ' attention types but num_layers=' +
          IntToStr(Config.NumLayers) + '.');
      if Config.WindowSize < 1 then
        ImportError('GPT-Neo import: window_size must be >= 1.');
      HeadDim := Config.HiddenSize div Config.NumHeads;
      // The 1/sqrt(d_head) the scaled SDPA applies is folded back out of
      // the scores by pre-scaling W_q (see the unit header).
      QScale := Sqrt(HeadDim);
      if Reader.HasTensor('transformer.wte.weight') then
        Config.Prefix := 'transformer.'
      else if Reader.HasTensor('wte.weight') then
        Config.Prefix := ''
      else
        ImportError('GPT-Neo import: neither "transformer.wte.weight" nor ' +
          '"wte.weight" found in ' + Reader.FileName +
          ' - not a GPT-Neo checkpoint?');
      if (Reader.DimCount(Config.Prefix + 'wte.weight') <> 2) or
         (Reader.DimSize(Config.Prefix + 'wte.weight', 0) <>
          Config.VocabSize) or
         (Reader.DimSize(Config.Prefix + 'wte.weight', 1) <>
          Config.HiddenSize) then
        ImportError('GPT-Neo import: wte.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.HiddenSize) +
          '], got ' + Reader.ShapeAsString(Config.Prefix + 'wte.weight'));
      if (not Config.TieWordEmbeddings) and
         (not Reader.HasTensor('lm_head.weight')) then
        ImportError('GPT-Neo import: config says tie_word_embeddings=false ' +
          'but "lm_head.weight" is missing from ' + Reader.FileName + '.');
      if pSeqLen <= 0 then SeqLen := Config.MaxPositions
      else SeqLen := pSeqLen;
      if SeqLen > Config.MaxPositions then
        ImportError('GPT-Neo import: requested SeqLen=' + IntToStr(SeqLen) +
          ' exceeds max_position_embeddings=' +
          IntToStr(Config.MaxPositions) + '.');

      // ---------------- Architecture ----------------
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      // EncodeZero=1: token id 0 is a real BPE token, not padding.
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      PosLayer := NN.AddLayer(
        TNNetLearnedPositionalEmbedding.Create(Config.MaxPositions) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      SetLength(Blocks, Config.NumLayers);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // Attention sub-block: x := x + out_proj(MHA(qkv(ln_1(x)))).
        // The separate HF q/k/v projections are fused into one Q|K|V slab;
        // local layers pass the GPT-Neo band as the SDPA sliding window
        // (identical mask, see the unit header), global layers pass 0.
        if Config.LayerIsLocal[BlockCnt] then Window := Config.WindowSize
        else Window := 0;
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].LN1 := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        Blocks[BlockCnt].QKV := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(3 * Config.HiddenSize) );
        Blocks[BlockCnt].AttnProj := NN.AddMultiHeadSelfAttention(
          Config.NumHeads, {CausalMask=}true, {UseRoPE=}false,
          {Variant=}avSDPA, {NumSinks=}1, Window);
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        // MLP sub-block: x := x + c_proj(gelu_new(c_fc(ln_2(x)))).
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].LN2 := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        Blocks[BlockCnt].CFc := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.IntermediateSize) );
        NN.AddLayer( TNNetGELU.Create() ); // tanh approximation = gelu_new
        Blocks[BlockCnt].MlpProj := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
      end;
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        // wte -> embedding table (vocab rows of d floats).
        Reader.LoadTensorFlat(Config.Prefix + 'wte.weight', Tmp);
        if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('GPT-Neo import: wte.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table ' +
            'size ' + IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
        EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
        EmbeddingLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'wte.weight');
        if Config.TieWordEmbeddings then
        begin
          // Tied LM head: logits = h . wte^T (rows copied, bias-free).
          for j := 0 to Config.VocabSize - 1 do
          begin
            for i := 0 to Config.HiddenSize - 1 do
              LMHead.Neurons[j].Weights.FData[i] :=
                Tmp.FData[j * Config.HiddenSize + i];
            LMHead.Neurons[j].BiasWeight := 0;
          end;
          LMHead.FlushWeightCache();
          // A redundant serialized lm_head.weight is ignorable when tied.
          if Reader.HasTensor('lm_head.weight') then
            MarkConsumed('lm_head.weight');
        end
        else
        begin
          LoadLlamaLinearWeights(Reader, LMHead, 'lm_head.weight',
            Config.HiddenSize, Config.VocabSize);
          MarkConsumed('lm_head.weight');
        end;
        // wpe -> learned positional table (max_pos rows of d floats).
        if not Reader.HasTensor(Config.Prefix + 'wpe.weight') then
          ImportError('GPT-Neo import: missing tensor "' + Config.Prefix +
            'wpe.weight" in ' + Reader.FileName);
        Reader.LoadTensorFlat(Config.Prefix + 'wpe.weight', Tmp);
        if PosLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('GPT-Neo import: wpe.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the positional table ' +
            'size ' + IntToStr(PosLayer.Neurons[0].Weights.Size) + '.');
        PosLayer.Neurons[0].Weights.Copy(Tmp);
        PosLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'wpe.weight');
      finally
        Tmp.Free;
      end;
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        BlockPrefix := Config.Prefix + 'h.' + IntToStr(BlockCnt) + '.';
        AttnPrefix := BlockPrefix + 'attn.attention.';
        LoadLayerNormWeights(Reader, Blocks[BlockCnt].LN1,
          BlockPrefix + 'ln_1.weight', BlockPrefix + 'ln_1.bias',
          Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'ln_1.weight');
        MarkConsumed(BlockPrefix + 'ln_1.bias');
        // Separate bias-free nn.Linear q/k/v -> fused Q|K|V slab. W_q is
        // multiplied by sqrt(d_head) to cancel the SDPA's 1/sqrt(d_head)
        // (GPT-Neo attention is UNSCALED; q_proj has no bias to scale).
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].QKV,
          AttnPrefix + 'q_proj.weight', Config.HiddenSize, Config.HiddenSize,
          0, 3 * Config.HiddenSize, 0, '', QScale);
        MarkConsumed(AttnPrefix + 'q_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].QKV,
          AttnPrefix + 'k_proj.weight', Config.HiddenSize, Config.HiddenSize,
          Config.HiddenSize, 3 * Config.HiddenSize);
        MarkConsumed(AttnPrefix + 'k_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].QKV,
          AttnPrefix + 'v_proj.weight', Config.HiddenSize, Config.HiddenSize,
          2 * Config.HiddenSize, 3 * Config.HiddenSize);
        MarkConsumed(AttnPrefix + 'v_proj.weight');
        // out_proj carries a bias (the only biased attention linear).
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].AttnProj,
          AttnPrefix + 'out_proj.weight', Config.HiddenSize,
          Config.HiddenSize, 0, -1, 0, AttnPrefix + 'out_proj.bias');
        MarkConsumed(AttnPrefix + 'out_proj.weight');
        MarkConsumed(AttnPrefix + 'out_proj.bias');
        LoadLayerNormWeights(Reader, Blocks[BlockCnt].LN2,
          BlockPrefix + 'ln_2.weight', BlockPrefix + 'ln_2.bias',
          Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'ln_2.weight');
        MarkConsumed(BlockPrefix + 'ln_2.bias');
        // MLP: plain nn.Linear [out, in] WITH biases (NOT GPT-2's
        // transposed Conv1D).
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].CFc,
          BlockPrefix + 'mlp.c_fc.weight', Config.HiddenSize,
          Config.IntermediateSize, 0, -1, 0, BlockPrefix + 'mlp.c_fc.bias');
        MarkConsumed(BlockPrefix + 'mlp.c_fc.weight');
        MarkConsumed(BlockPrefix + 'mlp.c_fc.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].MlpProj,
          BlockPrefix + 'mlp.c_proj.weight', Config.IntermediateSize,
          Config.HiddenSize, 0, -1, 0, BlockPrefix + 'mlp.c_proj.bias');
        MarkConsumed(BlockPrefix + 'mlp.c_proj.weight');
        MarkConsumed(BlockPrefix + 'mlp.c_proj.bias');
      end;
      LoadLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'ln_f.weight', Config.Prefix + 'ln_f.bias',
        Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'ln_f.weight');
      MarkConsumed(Config.Prefix + 'ln_f.bias');

      // ---------------- Unexpected-tensor check ----------------
      // Every tensor must be consumed or be a known ignorable buffer:
      // h.N.attn.attention.bias / .masked_bias are the causal/banded mask
      // buffers older PyTorch exports serialize (structural here).
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        if (Pos('.attn.attention.bias', TensorNameStr) > 0) or
           (Pos('.attn.attention.masked_bias', TensorNameStr) > 0) then
          continue;
        ImportError('GPT-Neo import: unexpected tensor "' + TensorNameStr +
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

function BuildGPTNeoFromSafeTensorsEx(const FileName: string;
  out Config: TGPTNeoConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadGPTNeoConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildGPTNeoFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly);
end;

function BuildGPTNeoFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;
var
  IgnoredConfig: TGPTNeoConfig;
begin
  Result := BuildGPTNeoFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly);
end;

// =========================== GPT-NEOX IMPORT ===============================

function ReadGPTNeoXConfigFromJSONFile(const FileName: string): TGPTNeoXConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, HiddenAct: string;
  Field: TJSONData;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('GPT-NeoX import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('GPT-NeoX import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

begin
  if not FileExists(FileName) then
    ImportError('GPT-NeoX import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('GPT-NeoX import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('GPT-NeoX import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'gpt_neox');
    if ModelType <> 'gpt_neox' then
      ImportError('GPT-NeoX import: config model_type is "' + ModelType +
        '" - expected "gpt_neox" (see BuildFromPretrained for the full ' +
        'dispatch).');
    Result.HiddenSize := RequiredInt('hidden_size');
    Result.IntermediateSize := RequiredInt('intermediate_size');
    Result.NumLayers := RequiredInt('num_hidden_layers');
    Result.NumHeads := RequiredInt('num_attention_heads');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.MaxPositions := RequiredInt('max_position_embeddings');
    Result.LayerNormEps := Obj.Get('layer_norm_eps', 1.0e-5);
    // rotary_pct is the classic GPTNeoXConfig key; newer transformers
    // versions serialize the same fraction as partial_rotary_factor.
    Result.RotaryPct := Obj.Get('rotary_pct',
      Obj.Get('partial_rotary_factor', 0.25));
    if (Result.RotaryPct <= 0) or (Result.RotaryPct > 1) then
      ImportError('GPT-NeoX import: config rotary_pct must be in (0, 1], ' +
        'got ' + FloatToStr(Result.RotaryPct) + '.');
    // rotary_emb_base is the classic key; rope_theta the newer spelling.
    Result.RopeTheta := Obj.Get('rope_theta',
      Obj.Get('rotary_emb_base', 10000.0));
    Field := Obj.Find('rope_scaling');
    if (Field <> nil) and not Field.IsNull then
      ImportError('GPT-NeoX import: config carries a non-null ' +
        '"rope_scaling" - long-context RoPE scaling is not wired into ' +
        'this importer yet.');
    Result.UseParallelResidual := Obj.Get('use_parallel_residual', True);
    Result.TieWordEmbeddings := Obj.Get('tie_word_embeddings', False);
    HiddenAct := Obj.Get('hidden_act', 'gelu');
    if HiddenAct = 'gelu' then
      Result.HiddenActTanh := False // exact erf form (the Pythia default)
    else if (HiddenAct = 'gelu_new') or
            (HiddenAct = 'gelu_pytorch_tanh') then
      Result.HiddenActTanh := True
    else
      ImportError('GPT-NeoX import: config hidden_act "' + HiddenAct +
        '" is not supported - only "gelu" (exact), "gelu_new" and ' +
        '"gelu_pytorch_tanh" are wired here.');
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function GPTNeoXConfigToString(const Config: TGPTNeoXConfig): string;
begin
  Result := 'gpt_neox config: layers=' + IntToStr(Config.NumLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', max_pos=' + IntToStr(Config.MaxPositions) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', ln_eps=' + FloatToStr(Config.LayerNormEps) +
    ', rotary_pct=' + FloatToStr(Config.RotaryPct) +
    ', rope_theta=' + FloatToStr(Config.RopeTheta) +
    ', parallel=' + BoolToStr(Config.UseParallelResidual, true) +
    ', tied=' + BoolToStr(Config.TieWordEmbeddings, true);
  if Config.HiddenActTanh then
    Result := Result + ', act=gelu_tanh'
  else
    Result := Result + ', act=gelu_exact';
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

// Loads the fused GPT-NeoX query_key_value nn.Linear ([3*hidden, hidden]
// weight + [3*hidden] bias) into a TNNetPointwiseConvLinear Q|K|V slab
// (q -> neurons 0..hidden-1, k -> hidden..2*hidden-1, v -> 2*hidden..).
// The checkpoint interleaves PER HEAD: source row r belongs to head
// h = r div (3*head_dim), within which the first head_dim rows are q, the
// next head_dim k, the last head_dim v (HF view(.., heads, 3*head_dim)
// then thirds). On top of the de-interleave, the first RotaryDims rows of
// every q and k head are PERMUTED from HF's rotate_half (first-half /
// second-half within the rotary slice) into TNNetRotaryEmbedding's
// interleaved (even/odd) pair layout - the Llama permutation restricted to
// the rotary slice; v rows and the non-rotary tail load straight.
procedure LoadGPTNeoXQKVWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName, BName: string;
  Hidden, Heads, HeadDim, RotaryDims: integer);
var
  W, B: TNNetVolume;
  r, i, HeadIdx, Third, RowInHead, RotHalf, TargetRow, TargetIdx: integer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('GPT-NeoX import: missing tensor "' + WName + '".');
  if not Reader.HasTensor(BName) then
    ImportError('GPT-NeoX import: missing tensor "' + BName + '".');
  if (Reader.DimCount(WName) <> 2) or
     (Reader.DimSize(WName, 0) <> 3 * Hidden) or
     (Reader.DimSize(WName, 1) <> Hidden) then
    ImportError('GPT-NeoX import: "' + WName + '" must have shape [' +
      IntToStr(3 * Hidden) + ', ' + IntToStr(Hidden) + '] (nn.Linear ' +
      'stores [out, in]), got ' + Reader.ShapeAsString(WName));
  if (Reader.DimCount(BName) <> 1) or
     (Reader.DimSize(BName, 0) <> 3 * Hidden) then
    ImportError('GPT-NeoX import: "' + BName + '" must have shape [' +
      IntToStr(3 * Hidden) + '], got ' + Reader.ShapeAsString(BName));
  if Layer.Neurons.Count <> 3 * Hidden then
    ImportError('GPT-NeoX import: internal error - layer for "' + WName +
      '" has ' + IntToStr(Layer.Neurons.Count) + ' neurons, expected ' +
      IntToStr(3 * Hidden) + '.');
  RotHalf := RotaryDims div 2;
  W := TNNetVolume.Create;
  B := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, W);
    Reader.LoadTensorFlat(BName, B);
    for r := 0 to 3 * Hidden - 1 do
    begin
      HeadIdx := r div (3 * HeadDim);
      Third := (r mod (3 * HeadDim)) div HeadDim; // 0=q, 1=k, 2=v
      RowInHead := r mod HeadDim;
      // rotate_half -> interleaved permutation, RESTRICTED to the first
      // RotaryDims rows of q and k heads (the partial-rotary slice).
      TargetRow := RowInHead;
      if (Third < 2) and (RowInHead < RotaryDims) then
      begin
        if RowInHead < RotHalf then
          TargetRow := 2 * RowInHead
        else
          TargetRow := 2 * (RowInHead - RotHalf) + 1;
      end;
      TargetIdx := Third * Hidden + HeadIdx * HeadDim + TargetRow;
      if Layer.Neurons[TargetIdx].Weights.Size <> Hidden then
        ImportError('GPT-NeoX import: internal error - neuron ' +
          IntToStr(TargetIdx) + ' for "' + WName + '" has ' +
          IntToStr(Layer.Neurons[TargetIdx].Weights.Size) +
          ' weights, expected ' + IntToStr(Hidden) + '.');
      for i := 0 to Hidden - 1 do
        Layer.Neurons[TargetIdx].Weights.FData[i] := W.FData[r * Hidden + i];
      Layer.Neurons[TargetIdx].BiasWeight := B.FData[r];
    end;
  finally
    B.Free;
    W.Free;
  end;
  Layer.FlushWeightCache();
end;

type
  TGPTNeoXBlockLayers = record
    LN1, QKV, AttnDense, LN2, HTo4H, FourHToH: TNNetLayer;
  end;

function BuildGPTNeoXFromSafeTensorsWithConfig(const FileName: string;
  var Config: TGPTNeoXConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TGPTNeoXBlockLayers;
  EmbeddingLayer, FinalLN, LMHead: TNNetLayer;
  BranchInput, AttnOut, MlpOut, QKVLayer: TNNetLayer;
  RotSlice, PassSlice, QHead, KHead, VHead, HeadPack: TNNetLayer;
  GELUSource, PhiBranch: TNNetLayer;
  HeadOutputs: array of TNNetLayer;
  RotChannels, PassChannels, VChannels: array of integer;
  BlockCnt, SeqLen, HeadCnt, HeadDim, RotaryDims, i, j, d: integer;
  Tmp: TNNetVolume;
  BlockPrefix, AttnPrefix, TensorNameStr: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

  // x*Phi(x), the exact erf GELU, composed from existing layers exactly
  // like the BERT path (see the BERT IMPORT section of the unit header).
  procedure AddExactOrTanhGELU;
  begin
    if Config.HiddenActTanh then
      NN.AddLayer( TNNetGELU.Create() ) // tanh approximation
    else
    begin
      GELUSource := NN.GetLastLayer();
      NN.AddLayerAfter(
        TNNetMulByConstant.Create(0.7071067811865476), GELUSource);
      NN.AddLayer( TNNetErf.Create() );
      NN.AddLayer( TNNetAddConstant.Create(1.0) );
      PhiBranch := NN.AddLayer( TNNetMulByConstant.Create(0.5) );
      NN.AddLayer( TNNetDeepConcat.Create([PhiBranch, GELUSource]) );
      NN.AddLayer( TNNetReGLU.Create() );
    end;
  end;

begin
  Reader := CreatePretrainedTensorReader(FileName);
  NN := nil;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      // ---------------- Config validation & prefix detection -------------
      if Config.NumHeads < 1 then
        ImportError('GPT-NeoX import: num_attention_heads must be >= 1.');
      if (Config.HiddenSize mod Config.NumHeads) <> 0 then
        ImportError('GPT-NeoX import: hidden_size=' +
          IntToStr(Config.HiddenSize) + ' is not divisible by ' +
          'num_attention_heads=' + IntToStr(Config.NumHeads) + '.');
      HeadDim := Config.HiddenSize div Config.NumHeads;
      if (Config.RotaryPct <= 0) or (Config.RotaryPct > 1) then
        ImportError('GPT-NeoX import: rotary_pct must be in (0, 1].');
      // HF: rotary_ndims = int(head_size * rotary_pct) - truncation.
      RotaryDims := Trunc(HeadDim * Config.RotaryPct);
      if (RotaryDims < 2) or Odd(RotaryDims) then
        ImportError('GPT-NeoX import: rotary_ndims = int(head_dim * ' +
          'rotary_pct) = ' + IntToStr(RotaryDims) + ' must be an even ' +
          'number >= 2 (RoPE rotates channel pairs).');
      if Reader.HasTensor('gpt_neox.embed_in.weight') then
        Config.Prefix := 'gpt_neox.'
      else if Reader.HasTensor('embed_in.weight') then
        Config.Prefix := ''
      else
        ImportError('GPT-NeoX import: neither "gpt_neox.embed_in.weight" ' +
          'nor "embed_in.weight" found in ' + Reader.FileName +
          ' - not a GPT-NeoX checkpoint?');
      if (Reader.DimCount(Config.Prefix + 'embed_in.weight') <> 2) or
         (Reader.DimSize(Config.Prefix + 'embed_in.weight', 0) <>
          Config.VocabSize) or
         (Reader.DimSize(Config.Prefix + 'embed_in.weight', 1) <>
          Config.HiddenSize) then
        ImportError('GPT-NeoX import: embed_in.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.HiddenSize) +
          '], got ' +
          Reader.ShapeAsString(Config.Prefix + 'embed_in.weight'));
      // embed_out (the untied LM head) sits OUTSIDE the gpt_neox. prefix in
      // GPTNeoXForCausalLM exports (like lm_head.weight in the Llama path).
      if (not Config.TieWordEmbeddings) and
         (not Reader.HasTensor('embed_out.weight')) then
        ImportError('GPT-NeoX import: config says tie_word_embeddings=' +
          'false (the GPT-NeoX/Pythia convention) but "embed_out.weight" ' +
          'is missing from ' + Reader.FileName + ' - a bare GPTNeoXModel ' +
          'export carries no LM head.');
      if pSeqLen <= 0 then SeqLen := Config.MaxPositions
      else SeqLen := pSeqLen;
      if SeqLen > Config.MaxPositions then
        ImportError('GPT-NeoX import: requested SeqLen=' + IntToStr(SeqLen) +
          ' exceeds max_position_embeddings=' +
          IntToStr(Config.MaxPositions) + '.');

      // ---------------- Architecture ----------------
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      // EncodeZero=1: token id 0 is a real BPE token, not padding.
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      SetLength(Blocks, Config.NumLayers);
      SetLength(HeadOutputs, Config.NumHeads);
      SetLength(RotChannels, RotaryDims);
      SetLength(PassChannels, HeadDim - RotaryDims);
      SetLength(VChannels, HeadDim);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // Attention branch: dense(MHA-with-partial-RoPE(LN_1(x))).
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].LN1 := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        QKVLayer := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(3 * Config.HiddenSize) );
        Blocks[BlockCnt].QKV := QKVLayer;
        for HeadCnt := 0 to Config.NumHeads - 1 do
        begin
          // PARTIAL ROTARY per head: RoPE on the first RotaryDims channels
          // of the Q and K slices, pass-through on the rest. The rotary
          // slice feeds a depth-RotaryDims TNNetRotaryEmbedding, so the
          // layer's frequency schedule theta^(-2k/RotaryDims) matches HF's
          // inv_freq over rotary_ndims exactly.
          for d := 0 to RotaryDims - 1 do
            RotChannels[d] := HeadCnt * HeadDim + d;
          for d := 0 to HeadDim - RotaryDims - 1 do
            PassChannels[d] := HeadCnt * HeadDim + RotaryDims + d;
          // Q slice of head HeadCnt (slab channels h*hd..h*hd+hd-1).
          RotSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(RotChannels), QKVLayer);
          RotSlice := NN.AddLayerAfter(
            TNNetRotaryEmbedding.Create(Config.RopeTheta), RotSlice);
          if HeadDim > RotaryDims then
          begin
            PassSlice := NN.AddLayerAfter(
              TNNetSplitChannels.Create(PassChannels), QKVLayer);
            QHead := NN.AddLayer(
              TNNetDeepConcat.Create([RotSlice, PassSlice]) );
          end
          else
            QHead := RotSlice;
          // K slice (slab channels hidden + h*hd...).
          for d := 0 to RotaryDims - 1 do
            RotChannels[d] := Config.HiddenSize + HeadCnt * HeadDim + d;
          for d := 0 to HeadDim - RotaryDims - 1 do
            PassChannels[d] :=
              Config.HiddenSize + HeadCnt * HeadDim + RotaryDims + d;
          RotSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(RotChannels), QKVLayer);
          RotSlice := NN.AddLayerAfter(
            TNNetRotaryEmbedding.Create(Config.RopeTheta), RotSlice);
          if HeadDim > RotaryDims then
          begin
            PassSlice := NN.AddLayerAfter(
              TNNetSplitChannels.Create(PassChannels), QKVLayer);
            KHead := NN.AddLayer(
              TNNetDeepConcat.Create([RotSlice, PassSlice]) );
          end
          else
            KHead := RotSlice;
          // V slice (slab channels 2*hidden + h*hd..., one contiguous
          // head_dim-wide slice; V is never rotated).
          for d := 0 to HeadDim - 1 do
            VChannels[d] := 2 * Config.HiddenSize + HeadCnt * HeadDim + d;
          VHead := NN.AddLayerAfter(
            TNNetSplitChannels.Create(VChannels), QKVLayer);
          // Pack [Q_h | K_h | V_h] (width 3*head_dim) for the scaled SDPA
          // (GPT-NeoX uses the standard 1/sqrt(head_dim) scaling).
          HeadPack := NN.AddLayer(
            TNNetDeepConcat.Create([QHead, KHead, VHead]) );
          HeadOutputs[HeadCnt] := NN.AddLayerAfter(
            TNNetScaledDotProductAttention.Create(HeadDim,
              {CausalMask=}true), HeadPack);
        end;
        NN.AddLayer( TNNetDeepConcat.Create(HeadOutputs) );
        Blocks[BlockCnt].AttnDense := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        AttnOut := NN.GetLastLayer();
        if Config.UseParallelResidual then
          // PARALLEL residual: the MLP branch reads the BLOCK INPUT through
          // its own LayerNorm; one fused 3-input sum closes the block:
          //   x := x + Attn(LN_1(x)) + MLP(LN_2(x))
          Blocks[BlockCnt].LN2 := NN.AddLayerAfter(
            TNNetTokenLayerNorm.Create(Config.LayerNormEps), BranchInput)
        else
        begin
          // SEQUENTIAL (use_parallel_residual=false): close the attention
          // residual first, then the MLP reads the attention output.
          AttnOut := NN.AddLayer(
            TNNetSum.Create([AttnOut, BranchInput]) );
          Blocks[BlockCnt].LN2 := NN.AddLayer(
            TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        end;
        Blocks[BlockCnt].HTo4H := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.IntermediateSize) );
        AddExactOrTanhGELU;
        Blocks[BlockCnt].FourHToH := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        MlpOut := NN.GetLastLayer();
        if Config.UseParallelResidual then
          NN.AddLayer( TNNetSum.Create([BranchInput, AttnOut, MlpOut]) )
        else
          NN.AddLayer( TNNetSum.Create([MlpOut, AttnOut]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
      end;
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        // embed_in -> embedding table (vocab rows of d floats).
        Reader.LoadTensorFlat(Config.Prefix + 'embed_in.weight', Tmp);
        if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('GPT-NeoX import: embed_in.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table ' +
            'size ' + IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
        EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
        EmbeddingLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'embed_in.weight');
        if Config.TieWordEmbeddings then
        begin
          // Tied LM head (NOT the Pythia convention, but cheap to support):
          // logits = h . embed_in^T, rows copied, bias-free.
          for j := 0 to Config.VocabSize - 1 do
          begin
            for i := 0 to Config.HiddenSize - 1 do
              LMHead.Neurons[j].Weights.FData[i] :=
                Tmp.FData[j * Config.HiddenSize + i];
            LMHead.Neurons[j].BiasWeight := 0;
          end;
          LMHead.FlushWeightCache();
          if Reader.HasTensor('embed_out.weight') then
            MarkConsumed('embed_out.weight');
        end
        else
        begin
          // UNTIED head (Pythia): embed_out is its own [vocab, hidden]
          // nn.Linear weight, bias-free.
          LoadLlamaLinearWeights(Reader, LMHead, 'embed_out.weight',
            Config.HiddenSize, Config.VocabSize);
          MarkConsumed('embed_out.weight');
        end;
      finally
        Tmp.Free;
      end;
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        BlockPrefix := Config.Prefix + 'layers.' + IntToStr(BlockCnt) + '.';
        AttnPrefix := BlockPrefix + 'attention.';
        LoadLayerNormWeights(Reader, Blocks[BlockCnt].LN1,
          BlockPrefix + 'input_layernorm.weight',
          BlockPrefix + 'input_layernorm.bias', Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'input_layernorm.weight');
        MarkConsumed(BlockPrefix + 'input_layernorm.bias');
        LoadLayerNormWeights(Reader, Blocks[BlockCnt].LN2,
          BlockPrefix + 'post_attention_layernorm.weight',
          BlockPrefix + 'post_attention_layernorm.bias', Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'post_attention_layernorm.weight');
        MarkConsumed(BlockPrefix + 'post_attention_layernorm.bias');
        // Fused per-head-interleaved query_key_value -> Q|K|V slab with the
        // partial-rotary rotate_half permutation composed in.
        LoadGPTNeoXQKVWeights(Reader, Blocks[BlockCnt].QKV,
          AttnPrefix + 'query_key_value.weight',
          AttnPrefix + 'query_key_value.bias',
          Config.HiddenSize, Config.NumHeads, HeadDim, RotaryDims);
        MarkConsumed(AttnPrefix + 'query_key_value.weight');
        MarkConsumed(AttnPrefix + 'query_key_value.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].AttnDense,
          AttnPrefix + 'dense.weight', Config.HiddenSize,
          Config.HiddenSize, 0, -1, 0, AttnPrefix + 'dense.bias');
        MarkConsumed(AttnPrefix + 'dense.weight');
        MarkConsumed(AttnPrefix + 'dense.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].HTo4H,
          BlockPrefix + 'mlp.dense_h_to_4h.weight', Config.HiddenSize,
          Config.IntermediateSize, 0, -1, 0,
          BlockPrefix + 'mlp.dense_h_to_4h.bias');
        MarkConsumed(BlockPrefix + 'mlp.dense_h_to_4h.weight');
        MarkConsumed(BlockPrefix + 'mlp.dense_h_to_4h.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].FourHToH,
          BlockPrefix + 'mlp.dense_4h_to_h.weight', Config.IntermediateSize,
          Config.HiddenSize, 0, -1, 0,
          BlockPrefix + 'mlp.dense_4h_to_h.bias');
        MarkConsumed(BlockPrefix + 'mlp.dense_4h_to_h.weight');
        MarkConsumed(BlockPrefix + 'mlp.dense_4h_to_h.bias');
      end;
      LoadLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'final_layer_norm.weight',
        Config.Prefix + 'final_layer_norm.bias', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'final_layer_norm.weight');
      MarkConsumed(Config.Prefix + 'final_layer_norm.bias');

      // ---------------- Unexpected-tensor check ----------------
      // Every tensor must be consumed or be a known ignorable buffer:
      //  - layers.N.attention.bias / .masked_bias: the causal-mask buffers
      //    older PyTorch exports serialize (the mask is structural here);
      //  - rotary_emb.inv_freq: RoPE buffers (structural, rebuilt from the
      //    config's base).
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        if (Pos('.attention.bias', TensorNameStr) > 0) or
           (Pos('.attention.masked_bias', TensorNameStr) > 0) or
           (Pos('rotary_emb.inv_freq', TensorNameStr) > 0) then continue;
        ImportError('GPT-NeoX import: unexpected tensor "' + TensorNameStr +
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

function BuildGPTNeoXFromSafeTensorsEx(const FileName: string;
  out Config: TGPTNeoXConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadGPTNeoXConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildGPTNeoXFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly);
end;

function BuildGPTNeoXFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;
var
  IgnoredConfig: TGPTNeoXConfig;
begin
  Result := BuildGPTNeoXFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly);
end;

// ============================ GPT-J IMPORT =================================

function ReadGPTJConfigFromJSONFile(const FileName: string): TGPTJConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, HiddenAct: string;
  Field: TJSONData;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('GPT-J import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('GPT-J import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

begin
  if not FileExists(FileName) then
    ImportError('GPT-J import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('GPT-J import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('GPT-J import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'gptj');
    if ModelType <> 'gptj' then
      ImportError('GPT-J import: config model_type is "' + ModelType +
        '" - expected "gptj" (see BuildFromPretrained for the full ' +
        'dispatch).');
    Result.HiddenSize := RequiredInt('n_embd');
    Result.NumLayers := RequiredInt('n_layer');
    Result.NumHeads := RequiredInt('n_head');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.MaxPositions := RequiredInt('n_positions');
    // n_inner is null in the GPT-J-6B config: the HF default 4*n_embd
    // applies (GPTJMLP: inner_dim = n_inner if n_inner is not None else
    // 4 * n_embd).
    Field := Obj.Find('n_inner');
    if (Field = nil) or Field.IsNull then
      Result.IntermediateSize := 4 * Result.HiddenSize
    else
      Result.IntermediateSize := RequiredInt('n_inner');
    Result.LayerNormEps := Obj.Get('layer_norm_epsilon', 1.0e-5);
    // rotary_dim defaults to 64 in HF's GPTJConfig (the GPT-J-6B value).
    Result.RotaryDim := Obj.Get('rotary_dim', 64);
    if Result.RotaryDim <= 0 then
      ImportError('GPT-J import: config rotary_dim must be a positive ' +
        'integer, got ' + Obj.Find('rotary_dim').AsJSON + '.');
    // GPT-J predates configurable RoPE bases: modeling_gptj hardcodes
    // 10000. Read rope_theta anyway so a hypothetical override works.
    Result.RopeTheta := Obj.Get('rope_theta', 10000.0);
    Result.TieWordEmbeddings := Obj.Get('tie_word_embeddings', False);
    HiddenAct := Obj.Get('activation_function', 'gelu_new');
    if HiddenAct = 'gelu' then
      Result.HiddenActTanh := False // exact erf form
    else if (HiddenAct = 'gelu_new') or
            (HiddenAct = 'gelu_pytorch_tanh') then
      Result.HiddenActTanh := True // the GPT-J default
    else
      ImportError('GPT-J import: config activation_function "' + HiddenAct +
        '" is not supported - only "gelu" (exact), "gelu_new" and ' +
        '"gelu_pytorch_tanh" are wired here.');
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function GPTJConfigToString(const Config: TGPTJConfig): string;
begin
  Result := 'gptj config: layers=' + IntToStr(Config.NumLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', max_pos=' + IntToStr(Config.MaxPositions) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', ln_eps=' + FloatToStr(Config.LayerNormEps) +
    ', rotary_dim=' + IntToStr(Config.RotaryDim) +
    ', rope_theta=' + FloatToStr(Config.RopeTheta) +
    ', tied=' + BoolToStr(Config.TieWordEmbeddings, true);
  if Config.HiddenActTanh then
    Result := Result + ', act=gelu_tanh'
  else
    Result := Result + ', act=gelu_exact';
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

type
  TGPTJBlockLayers = record
    LN1, QProj, KProj, VProj, OutProj, FcIn, FcOut: TNNetLayer;
  end;

function BuildGPTJFromSafeTensorsWithConfig(const FileName: string;
  var Config: TGPTJConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TGPTJBlockLayers;
  EmbeddingLayer, FinalLN, LMHead: TNNetLayer;
  BranchInput, SharedLN, AttnOut, MlpOut: TNNetLayer;
  RotSlice, PassSlice, QHead, KHead, VHead, HeadPack: TNNetLayer;
  GELUSource, PhiBranch: TNNetLayer;
  HeadOutputs: array of TNNetLayer;
  RotChannels, PassChannels, VChannels: array of integer;
  BlockCnt, SeqLen, HeadCnt, HeadDim, i, d: integer;
  Tmp: TNNetVolume;
  BlockPrefix, AttnPrefix, TensorNameStr: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

  // x*Phi(x), the exact erf GELU, composed from existing layers exactly
  // like the BERT path (see the BERT IMPORT section of the unit header).
  // GPT-J configs use gelu_new (the tanh TNNetGELU) in practice.
  procedure AddExactOrTanhGELU;
  begin
    if Config.HiddenActTanh then
      NN.AddLayer( TNNetGELU.Create() ) // tanh approximation
    else
    begin
      GELUSource := NN.GetLastLayer();
      NN.AddLayerAfter(
        TNNetMulByConstant.Create(0.7071067811865476), GELUSource);
      NN.AddLayer( TNNetErf.Create() );
      NN.AddLayer( TNNetAddConstant.Create(1.0) );
      PhiBranch := NN.AddLayer( TNNetMulByConstant.Create(0.5) );
      NN.AddLayer( TNNetDeepConcat.Create([PhiBranch, GELUSource]) );
      NN.AddLayer( TNNetReGLU.Create() );
    end;
  end;

begin
  Reader := CreatePretrainedTensorReader(FileName);
  NN := nil;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      // ---------------- Config validation & prefix detection -------------
      if Config.NumHeads < 1 then
        ImportError('GPT-J import: n_head must be >= 1.');
      if (Config.HiddenSize mod Config.NumHeads) <> 0 then
        ImportError('GPT-J import: n_embd=' +
          IntToStr(Config.HiddenSize) + ' is not divisible by ' +
          'n_head=' + IntToStr(Config.NumHeads) + '.');
      HeadDim := Config.HiddenSize div Config.NumHeads;
      if (Config.RotaryDim < 2) or Odd(Config.RotaryDim) or
         (Config.RotaryDim > HeadDim) then
        ImportError('GPT-J import: rotary_dim = ' +
          IntToStr(Config.RotaryDim) + ' must be an even number in ' +
          '[2, head_dim=' + IntToStr(HeadDim) + '] (RoPE rotates ' +
          'channel pairs).');
      if Config.TieWordEmbeddings then
        ImportError('GPT-J import: tie_word_embeddings=true is not ' +
          'supported - GPT-J carries an UNTIED lm_head with its own ' +
          'bias (every published GPT-J config is untied).');
      if Reader.HasTensor('transformer.wte.weight') then
        Config.Prefix := 'transformer.'
      else if Reader.HasTensor('wte.weight') then
        Config.Prefix := ''
      else
        ImportError('GPT-J import: neither "transformer.wte.weight" ' +
          'nor "wte.weight" found in ' + Reader.FileName +
          ' - not a GPT-J checkpoint?');
      if (Reader.DimCount(Config.Prefix + 'wte.weight') <> 2) or
         (Reader.DimSize(Config.Prefix + 'wte.weight', 0) <>
          Config.VocabSize) or
         (Reader.DimSize(Config.Prefix + 'wte.weight', 1) <>
          Config.HiddenSize) then
        ImportError('GPT-J import: wte.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.HiddenSize) +
          '], got ' +
          Reader.ShapeAsString(Config.Prefix + 'wte.weight'));
      // lm_head sits OUTSIDE the transformer. prefix in GPTJForCausalLM
      // exports (like embed_out in the GPT-NeoX path) and has a BIAS.
      if not Reader.HasTensor('lm_head.weight') then
        ImportError('GPT-J import: "lm_head.weight" is missing from ' +
          Reader.FileName + ' - a bare GPTJModel export carries no LM ' +
          'head.');
      if pSeqLen <= 0 then SeqLen := Config.MaxPositions
      else SeqLen := pSeqLen;
      if SeqLen > Config.MaxPositions then
        ImportError('GPT-J import: requested SeqLen=' + IntToStr(SeqLen) +
          ' exceeds n_positions=' + IntToStr(Config.MaxPositions) + '.');

      // ---------------- Architecture ----------------
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      // EncodeZero=1: token id 0 is a real BPE token, not padding.
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      SetLength(Blocks, Config.NumLayers);
      SetLength(HeadOutputs, Config.NumHeads);
      SetLength(RotChannels, Config.RotaryDim);
      SetLength(PassChannels, HeadDim - Config.RotaryDim);
      SetLength(VChannels, HeadDim);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // SHARED-LN PARALLEL residual: ONE LayerNorm (ln_1) feeds BOTH the
        // attention and the MLP branch; one fused 3-input sum closes the
        // block:  x := x + Attn(LN(x)) + MLP(LN(x))
        BranchInput := NN.GetLastLayer();
        SharedLN := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        Blocks[BlockCnt].LN1 := SharedLN;
        // SEPARATE bias-free q/k/v projections (plain nn.Linear), all
        // reading the shared LN output.
        Blocks[BlockCnt].QProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize), SharedLN);
        Blocks[BlockCnt].KProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize), SharedLN);
        Blocks[BlockCnt].VProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize), SharedLN);
        for HeadCnt := 0 to Config.NumHeads - 1 do
        begin
          // PARTIAL ROTARY per head: RoPE on the first rotary_dim channels
          // of the Q and K head, pass-through on the rest. GPT-J pairs the
          // rotary channels INTERLEAVED - (0,1),(2,3),... - which is
          // TNNetRotaryEmbedding's native layout, so the projection rows
          // were loaded STRAIGHT (no rotate_half permutation; see the
          // unit header). The rotary slice feeds a depth-rotary_dim
          // TNNetRotaryEmbedding, so the layer's frequency schedule
          // theta^(-2k/rotary_dim) matches HF's inv_freq exactly.
          for d := 0 to Config.RotaryDim - 1 do
            RotChannels[d] := HeadCnt * HeadDim + d;
          for d := 0 to HeadDim - Config.RotaryDim - 1 do
            PassChannels[d] := HeadCnt * HeadDim + Config.RotaryDim + d;
          RotSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(RotChannels),
            Blocks[BlockCnt].QProj);
          RotSlice := NN.AddLayerAfter(
            TNNetRotaryEmbedding.Create(Config.RopeTheta), RotSlice);
          if HeadDim > Config.RotaryDim then
          begin
            PassSlice := NN.AddLayerAfter(
              TNNetSplitChannels.Create(PassChannels),
              Blocks[BlockCnt].QProj);
            QHead := NN.AddLayer(
              TNNetDeepConcat.Create([RotSlice, PassSlice]) );
          end
          else
            QHead := RotSlice;
          RotSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(RotChannels),
            Blocks[BlockCnt].KProj);
          RotSlice := NN.AddLayerAfter(
            TNNetRotaryEmbedding.Create(Config.RopeTheta), RotSlice);
          if HeadDim > Config.RotaryDim then
          begin
            PassSlice := NN.AddLayerAfter(
              TNNetSplitChannels.Create(PassChannels),
              Blocks[BlockCnt].KProj);
            KHead := NN.AddLayer(
              TNNetDeepConcat.Create([RotSlice, PassSlice]) );
          end
          else
            KHead := RotSlice;
          // V head (one contiguous head_dim-wide slice; never rotated).
          for d := 0 to HeadDim - 1 do
            VChannels[d] := HeadCnt * HeadDim + d;
          VHead := NN.AddLayerAfter(
            TNNetSplitChannels.Create(VChannels), Blocks[BlockCnt].VProj);
          // Pack [Q_h | K_h | V_h] (width 3*head_dim) for the scaled SDPA
          // (GPT-J uses the standard 1/sqrt(head_dim) scaling).
          HeadPack := NN.AddLayer(
            TNNetDeepConcat.Create([QHead, KHead, VHead]) );
          HeadOutputs[HeadCnt] := NN.AddLayerAfter(
            TNNetScaledDotProductAttention.Create(HeadDim,
              {CausalMask=}true), HeadPack);
        end;
        NN.AddLayer( TNNetDeepConcat.Create(HeadOutputs) );
        Blocks[BlockCnt].OutProj := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        AttnOut := NN.GetLastLayer();
        // MLP branch: reads the SAME shared LN output (NOT a second norm,
        // NOT the attention output - GPT-J is always parallel).
        Blocks[BlockCnt].FcIn := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.IntermediateSize),
          SharedLN);
        AddExactOrTanhGELU;
        Blocks[BlockCnt].FcOut := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        MlpOut := NN.GetLastLayer();
        NN.AddLayer( TNNetSum.Create([BranchInput, AttnOut, MlpOut]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
      end;
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        // wte -> embedding table (vocab rows of d floats).
        Reader.LoadTensorFlat(Config.Prefix + 'wte.weight', Tmp);
        if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('GPT-J import: wte.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table ' +
            'size ' + IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
        EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
        EmbeddingLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'wte.weight');
      finally
        Tmp.Free;
      end;
      // UNTIED head with a BIAS (unique among the decoder importers).
      LoadLlamaLinearWeights(Reader, LMHead, 'lm_head.weight',
        Config.HiddenSize, Config.VocabSize, 0, -1, 0, 'lm_head.bias');
      MarkConsumed('lm_head.weight');
      MarkConsumed('lm_head.bias');
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        BlockPrefix := Config.Prefix + 'h.' + IntToStr(BlockCnt) + '.';
        AttnPrefix := BlockPrefix + 'attn.';
        LoadLayerNormWeights(Reader, Blocks[BlockCnt].LN1,
          BlockPrefix + 'ln_1.weight',
          BlockPrefix + 'ln_1.bias', Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'ln_1.weight');
        MarkConsumed(BlockPrefix + 'ln_1.bias');
        // Bias-free q/k/v, rows loaded STRAIGHT (RotaryHeadDim=0: GPT-J's
        // interleaved RoPE pairing is already the layer's layout).
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].QProj,
          AttnPrefix + 'q_proj.weight', Config.HiddenSize,
          Config.HiddenSize);
        MarkConsumed(AttnPrefix + 'q_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].KProj,
          AttnPrefix + 'k_proj.weight', Config.HiddenSize,
          Config.HiddenSize);
        MarkConsumed(AttnPrefix + 'k_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].VProj,
          AttnPrefix + 'v_proj.weight', Config.HiddenSize,
          Config.HiddenSize);
        MarkConsumed(AttnPrefix + 'v_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].OutProj,
          AttnPrefix + 'out_proj.weight', Config.HiddenSize,
          Config.HiddenSize);
        MarkConsumed(AttnPrefix + 'out_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].FcIn,
          BlockPrefix + 'mlp.fc_in.weight', Config.HiddenSize,
          Config.IntermediateSize, 0, -1, 0,
          BlockPrefix + 'mlp.fc_in.bias');
        MarkConsumed(BlockPrefix + 'mlp.fc_in.weight');
        MarkConsumed(BlockPrefix + 'mlp.fc_in.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].FcOut,
          BlockPrefix + 'mlp.fc_out.weight', Config.IntermediateSize,
          Config.HiddenSize, 0, -1, 0,
          BlockPrefix + 'mlp.fc_out.bias');
        MarkConsumed(BlockPrefix + 'mlp.fc_out.weight');
        MarkConsumed(BlockPrefix + 'mlp.fc_out.bias');
      end;
      LoadLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'ln_f.weight',
        Config.Prefix + 'ln_f.bias', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'ln_f.weight');
      MarkConsumed(Config.Prefix + 'ln_f.bias');

      // ---------------- Unexpected-tensor check ----------------
      // Every tensor must be consumed or be a known ignorable buffer:
      //  - h.N.attn.bias / .masked_bias: the causal-mask buffers older
      //    PyTorch exports serialize (the mask is structural here);
      //  - attn.embed_positions: the sinusoidal RoPE table some
      //    transformers versions persist (structural, rebuilt from
      //    rotary_dim + the base).
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        if (Pos('.attn.bias', TensorNameStr) > 0) or
           (Pos('.attn.masked_bias', TensorNameStr) > 0) or
           (Pos('.attn.embed_positions', TensorNameStr) > 0) then continue;
        ImportError('GPT-J import: unexpected tensor "' + TensorNameStr +
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

function BuildGPTJFromSafeTensorsEx(const FileName: string;
  out Config: TGPTJConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadGPTJConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildGPTJFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly);
end;

function BuildGPTJFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;
var
  IgnoredConfig: TGPTJConfig;
begin
  Result := BuildGPTJFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly);
end;

// ============================ PHI IMPORT ===================================

function ReadPhiConfigFromJSONFile(const FileName: string): TPhiConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, HiddenAct: string;
  Field: TJSONData;
  NumKVHeads: integer;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('Phi import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('Phi import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

begin
  if not FileExists(FileName) then
    ImportError('Phi import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('Phi import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('Phi import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'phi');
    if ModelType <> 'phi' then
      ImportError('Phi import: config model_type is "' + ModelType +
        '" - expected "phi" (see BuildFromPretrained for the full ' +
        'dispatch).');
    Result.HiddenSize := RequiredInt('hidden_size');
    Result.IntermediateSize := RequiredInt('intermediate_size');
    Result.NumLayers := RequiredInt('num_hidden_layers');
    Result.NumHeads := RequiredInt('num_attention_heads');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.MaxPositions := RequiredInt('max_position_embeddings');
    // Every released Phi checkpoint is plain MHA; a GQA config would need
    // K/V head broadcasting this builder does not wire.
    NumKVHeads := Obj.Get('num_key_value_heads', Result.NumHeads);
    if NumKVHeads <> Result.NumHeads then
      ImportError('Phi import: num_key_value_heads=' +
        IntToStr(NumKVHeads) + ' <> num_attention_heads=' +
        IntToStr(Result.NumHeads) + ' (GQA) is not supported - every ' +
        'released Phi checkpoint uses plain MHA.');
    // qk_layernorm is false in every released checkpoint; loading one
    // with the flag set would silently skip the q/k norms.
    if Obj.Get('qk_layernorm', False) then
      ImportError('Phi import: qk_layernorm=true is not supported - ' +
        'every released Phi checkpoint has qk_layernorm=false.');
    Result.LayerNormEps := Obj.Get('layer_norm_eps', 1.0e-5);
    Result.PartialRotaryFactor :=
      Obj.Get('partial_rotary_factor', 0.5);
    if (Result.PartialRotaryFactor <= 0) or
       (Result.PartialRotaryFactor > 1) then
      ImportError('Phi import: config partial_rotary_factor must be in ' +
        '(0, 1], got ' + FloatToStr(Result.PartialRotaryFactor) + '.');
    Result.RopeTheta := Obj.Get('rope_theta', 10000.0);
    Field := Obj.Find('rope_scaling');
    if (Field <> nil) and not Field.IsNull then
      ImportError('Phi import: config carries a non-null ' +
        '"rope_scaling" - long-context RoPE scaling is not wired into ' +
        'this importer yet.');
    Result.TieWordEmbeddings := Obj.Get('tie_word_embeddings', False);
    HiddenAct := Obj.Get('hidden_act', 'gelu_new');
    if HiddenAct = 'gelu' then
      Result.HiddenActTanh := False // exact erf form
    else if (HiddenAct = 'gelu_new') or
            (HiddenAct = 'gelu_pytorch_tanh') then
      Result.HiddenActTanh := True // the Phi default
    else
      ImportError('Phi import: config hidden_act "' + HiddenAct +
        '" is not supported - only "gelu" (exact), "gelu_new" and ' +
        '"gelu_pytorch_tanh" are wired here.');
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function PhiConfigToString(const Config: TPhiConfig): string;
begin
  Result := 'phi config: layers=' + IntToStr(Config.NumLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', max_pos=' + IntToStr(Config.MaxPositions) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', ln_eps=' + FloatToStr(Config.LayerNormEps) +
    ', partial_rotary=' + FloatToStr(Config.PartialRotaryFactor) +
    ', rope_theta=' + FloatToStr(Config.RopeTheta) +
    ', tied=' + BoolToStr(Config.TieWordEmbeddings, true);
  if Config.HiddenActTanh then
    Result := Result + ', act=gelu_tanh'
  else
    Result := Result + ', act=gelu_exact';
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

// Loads a Phi q_proj/k_proj nn.Linear ([Hidden, Hidden] weight + [Hidden]
// bias) whose first RotaryDims rows PER HEAD are PERMUTED from HF's
// rotate_half layout (first-half / second-half within the rotary slice)
// into TNNetRotaryEmbedding's interleaved (even/odd) pair layout - the
// same permutation LoadGPTNeoXQKVWeights composes into its fused slab,
// here applied to a separate projection. Rows beyond the rotary slice
// (the pass-through tail) and the bias entries follow the same row map.
procedure LoadPhiPartialRotaryLinearWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName, BName: string;
  Hidden, HeadDim, RotaryDims: integer);
var
  W, B: TNNetVolume;
  r, i, HeadIdx, RowInHead, RotHalf, TargetRow, TargetIdx: integer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('Phi import: missing tensor "' + WName + '".');
  if not Reader.HasTensor(BName) then
    ImportError('Phi import: missing tensor "' + BName + '".');
  if (Reader.DimCount(WName) <> 2) or
     (Reader.DimSize(WName, 0) <> Hidden) or
     (Reader.DimSize(WName, 1) <> Hidden) then
    ImportError('Phi import: "' + WName + '" must have shape [' +
      IntToStr(Hidden) + ', ' + IntToStr(Hidden) + '] (nn.Linear ' +
      'stores [out, in]), got ' + Reader.ShapeAsString(WName));
  if (Reader.DimCount(BName) <> 1) or
     (Reader.DimSize(BName, 0) <> Hidden) then
    ImportError('Phi import: "' + BName + '" must have shape [' +
      IntToStr(Hidden) + '], got ' + Reader.ShapeAsString(BName));
  if Layer.Neurons.Count <> Hidden then
    ImportError('Phi import: internal error - layer for "' + WName +
      '" has ' + IntToStr(Layer.Neurons.Count) + ' neurons, expected ' +
      IntToStr(Hidden) + '.');
  RotHalf := RotaryDims div 2;
  W := TNNetVolume.Create;
  B := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, W);
    Reader.LoadTensorFlat(BName, B);
    for r := 0 to Hidden - 1 do
    begin
      HeadIdx := r div HeadDim;
      RowInHead := r mod HeadDim;
      // rotate_half -> interleaved permutation, RESTRICTED to the first
      // RotaryDims rows of the head (the partial-rotary slice).
      TargetRow := RowInHead;
      if RowInHead < RotaryDims then
      begin
        if RowInHead < RotHalf then
          TargetRow := 2 * RowInHead
        else
          TargetRow := 2 * (RowInHead - RotHalf) + 1;
      end;
      TargetIdx := HeadIdx * HeadDim + TargetRow;
      if Layer.Neurons[TargetIdx].Weights.Size <> Hidden then
        ImportError('Phi import: internal error - neuron ' +
          IntToStr(TargetIdx) + ' for "' + WName + '" has ' +
          IntToStr(Layer.Neurons[TargetIdx].Weights.Size) +
          ' weights, expected ' + IntToStr(Hidden) + '.');
      for i := 0 to Hidden - 1 do
        Layer.Neurons[TargetIdx].Weights.FData[i] := W.FData[r * Hidden + i];
      Layer.Neurons[TargetIdx].BiasWeight := B.FData[r];
    end;
  finally
    B.Free;
    W.Free;
  end;
  Layer.FlushWeightCache();
end;

type
  TPhiBlockLayers = record
    LN1, QProj, KProj, VProj, AttnDense, Fc1, Fc2: TNNetLayer;
  end;

function BuildPhiFromSafeTensorsWithConfig(const FileName: string;
  var Config: TPhiConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TPhiBlockLayers;
  EmbeddingLayer, FinalLN, LMHead: TNNetLayer;
  BranchInput, SharedLN, AttnOut, MlpOut: TNNetLayer;
  RotSlice, PassSlice, QHead, KHead, VHead, HeadPack: TNNetLayer;
  GELUSource, PhiBranch: TNNetLayer;
  HeadOutputs: array of TNNetLayer;
  RotChannels, PassChannels, VChannels: array of integer;
  BlockCnt, SeqLen, HeadCnt, HeadDim, RotaryDims, i, d: integer;
  Tmp: TNNetVolume;
  BlockPrefix, AttnPrefix, TensorNameStr: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

  // x*Phi(x), the exact erf GELU, composed from existing layers exactly
  // like the BERT path (see the BERT IMPORT section of the unit header).
  // Phi configs use gelu_new (the tanh TNNetGELU) in practice.
  procedure AddExactOrTanhGELU;
  begin
    if Config.HiddenActTanh then
      NN.AddLayer( TNNetGELU.Create() ) // tanh approximation
    else
    begin
      GELUSource := NN.GetLastLayer();
      NN.AddLayerAfter(
        TNNetMulByConstant.Create(0.7071067811865476), GELUSource);
      NN.AddLayer( TNNetErf.Create() );
      NN.AddLayer( TNNetAddConstant.Create(1.0) );
      PhiBranch := NN.AddLayer( TNNetMulByConstant.Create(0.5) );
      NN.AddLayer( TNNetDeepConcat.Create([PhiBranch, GELUSource]) );
      NN.AddLayer( TNNetReGLU.Create() );
    end;
  end;

begin
  Reader := CreatePretrainedTensorReader(FileName);
  NN := nil;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      // ---------------- Config validation & prefix detection -------------
      if Config.NumHeads < 1 then
        ImportError('Phi import: num_attention_heads must be >= 1.');
      if (Config.HiddenSize mod Config.NumHeads) <> 0 then
        ImportError('Phi import: hidden_size=' +
          IntToStr(Config.HiddenSize) + ' is not divisible by ' +
          'num_attention_heads=' + IntToStr(Config.NumHeads) + '.');
      HeadDim := Config.HiddenSize div Config.NumHeads;
      if (Config.PartialRotaryFactor <= 0) or
         (Config.PartialRotaryFactor > 1) then
        ImportError('Phi import: partial_rotary_factor must be in (0, 1].');
      // HF: rotary_ndims = int(head_dim * partial_rotary_factor).
      RotaryDims := Trunc(HeadDim * Config.PartialRotaryFactor);
      if (RotaryDims < 2) or Odd(RotaryDims) then
        ImportError('Phi import: rotary_ndims = int(head_dim * ' +
          'partial_rotary_factor) = ' + IntToStr(RotaryDims) +
          ' must be an even number >= 2 (RoPE rotates channel pairs).');
      if Config.TieWordEmbeddings then
        ImportError('Phi import: tie_word_embeddings=true is not ' +
          'supported - Phi carries an UNTIED lm_head with its own ' +
          'bias (every published Phi config is untied).');
      if Reader.HasTensor('model.embed_tokens.weight') then
        Config.Prefix := 'model.'
      else if Reader.HasTensor('embed_tokens.weight') then
        Config.Prefix := ''
      else
        ImportError('Phi import: neither "model.embed_tokens.weight" ' +
          'nor "embed_tokens.weight" found in ' + Reader.FileName +
          ' - not a Phi checkpoint?');
      if (Reader.DimCount(Config.Prefix + 'embed_tokens.weight') <> 2) or
         (Reader.DimSize(Config.Prefix + 'embed_tokens.weight', 0) <>
          Config.VocabSize) or
         (Reader.DimSize(Config.Prefix + 'embed_tokens.weight', 1) <>
          Config.HiddenSize) then
        ImportError('Phi import: embed_tokens.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.HiddenSize) +
          '], got ' +
          Reader.ShapeAsString(Config.Prefix + 'embed_tokens.weight'));
      // lm_head sits OUTSIDE the model. prefix in PhiForCausalLM exports
      // (like the GPT-J path) and has a BIAS.
      if not Reader.HasTensor('lm_head.weight') then
        ImportError('Phi import: "lm_head.weight" is missing from ' +
          Reader.FileName + ' - a bare PhiModel export carries no LM ' +
          'head.');
      if pSeqLen <= 0 then SeqLen := Config.MaxPositions
      else SeqLen := pSeqLen;
      if SeqLen > Config.MaxPositions then
        ImportError('Phi import: requested SeqLen=' + IntToStr(SeqLen) +
          ' exceeds max_position_embeddings=' +
          IntToStr(Config.MaxPositions) + '.');

      // ---------------- Architecture ----------------
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      // EncodeZero=1: token id 0 is a real BPE token, not padding.
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      SetLength(Blocks, Config.NumLayers);
      SetLength(HeadOutputs, Config.NumHeads);
      SetLength(RotChannels, RotaryDims);
      SetLength(PassChannels, HeadDim - RotaryDims);
      SetLength(VChannels, HeadDim);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // SHARED-LN PARALLEL residual: ONE LayerNorm (input_layernorm)
        // feeds BOTH the attention and the MLP branch; one fused 3-input
        // sum closes the block:  x := x + Attn(LN(x)) + MLP(LN(x))
        BranchInput := NN.GetLastLayer();
        SharedLN := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        Blocks[BlockCnt].LN1 := SharedLN;
        // SEPARATE BIASED q/k/v projections (plain nn.Linear), all
        // reading the shared LN output.
        Blocks[BlockCnt].QProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize), SharedLN);
        Blocks[BlockCnt].KProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize), SharedLN);
        Blocks[BlockCnt].VProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize), SharedLN);
        for HeadCnt := 0 to Config.NumHeads - 1 do
        begin
          // PARTIAL ROTARY per head: RoPE on the first RotaryDims channels
          // of the Q and K head, pass-through on the rest. Phi pairs the
          // rotary channels rotate_half-style (k with k + RotaryDims/2),
          // so the q/k projection rows were PERMUTED at load time into
          // TNNetRotaryEmbedding's interleaved layout (see
          // LoadPhiPartialRotaryLinearWeights). The rotary slice feeds a
          // depth-RotaryDims TNNetRotaryEmbedding, so the layer's
          // frequency schedule theta^(-2k/RotaryDims) matches HF's
          // inv_freq over rotary_ndims exactly.
          for d := 0 to RotaryDims - 1 do
            RotChannels[d] := HeadCnt * HeadDim + d;
          for d := 0 to HeadDim - RotaryDims - 1 do
            PassChannels[d] := HeadCnt * HeadDim + RotaryDims + d;
          RotSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(RotChannels),
            Blocks[BlockCnt].QProj);
          RotSlice := NN.AddLayerAfter(
            TNNetRotaryEmbedding.Create(Config.RopeTheta), RotSlice);
          if HeadDim > RotaryDims then
          begin
            PassSlice := NN.AddLayerAfter(
              TNNetSplitChannels.Create(PassChannels),
              Blocks[BlockCnt].QProj);
            QHead := NN.AddLayer(
              TNNetDeepConcat.Create([RotSlice, PassSlice]) );
          end
          else
            QHead := RotSlice;
          RotSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(RotChannels),
            Blocks[BlockCnt].KProj);
          RotSlice := NN.AddLayerAfter(
            TNNetRotaryEmbedding.Create(Config.RopeTheta), RotSlice);
          if HeadDim > RotaryDims then
          begin
            PassSlice := NN.AddLayerAfter(
              TNNetSplitChannels.Create(PassChannels),
              Blocks[BlockCnt].KProj);
            KHead := NN.AddLayer(
              TNNetDeepConcat.Create([RotSlice, PassSlice]) );
          end
          else
            KHead := RotSlice;
          // V head (one contiguous head_dim-wide slice; never rotated).
          for d := 0 to HeadDim - 1 do
            VChannels[d] := HeadCnt * HeadDim + d;
          VHead := NN.AddLayerAfter(
            TNNetSplitChannels.Create(VChannels), Blocks[BlockCnt].VProj);
          // Pack [Q_h | K_h | V_h] (width 3*head_dim) for the scaled SDPA
          // (Phi uses the standard 1/sqrt(head_dim) scaling).
          HeadPack := NN.AddLayer(
            TNNetDeepConcat.Create([QHead, KHead, VHead]) );
          HeadOutputs[HeadCnt] := NN.AddLayerAfter(
            TNNetScaledDotProductAttention.Create(HeadDim,
              {CausalMask=}true), HeadPack);
        end;
        NN.AddLayer( TNNetDeepConcat.Create(HeadOutputs) );
        Blocks[BlockCnt].AttnDense := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        AttnOut := NN.GetLastLayer();
        // MLP branch: reads the SAME shared LN output (NOT a second norm,
        // NOT the attention output - Phi is always parallel).
        Blocks[BlockCnt].Fc1 := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.IntermediateSize),
          SharedLN);
        AddExactOrTanhGELU;
        Blocks[BlockCnt].Fc2 := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        MlpOut := NN.GetLastLayer();
        NN.AddLayer( TNNetSum.Create([BranchInput, AttnOut, MlpOut]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
      end;
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        // embed_tokens -> embedding table (vocab rows of d floats).
        Reader.LoadTensorFlat(Config.Prefix + 'embed_tokens.weight', Tmp);
        if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('Phi import: embed_tokens.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table ' +
            'size ' + IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
        EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
        EmbeddingLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'embed_tokens.weight');
      finally
        Tmp.Free;
      end;
      // UNTIED head WITH a bias (like the GPT-J path).
      LoadLlamaLinearWeights(Reader, LMHead, 'lm_head.weight',
        Config.HiddenSize, Config.VocabSize, 0, -1, 0, 'lm_head.bias');
      MarkConsumed('lm_head.weight');
      MarkConsumed('lm_head.bias');
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        BlockPrefix := Config.Prefix + 'layers.' + IntToStr(BlockCnt) + '.';
        AttnPrefix := BlockPrefix + 'self_attn.';
        LoadLayerNormWeights(Reader, Blocks[BlockCnt].LN1,
          BlockPrefix + 'input_layernorm.weight',
          BlockPrefix + 'input_layernorm.bias', Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'input_layernorm.weight');
        MarkConsumed(BlockPrefix + 'input_layernorm.bias');
        // Biased q/k with the partial-rotary rotate_half permutation
        // composed in; v and dense load straight (with their biases).
        LoadPhiPartialRotaryLinearWeights(Reader, Blocks[BlockCnt].QProj,
          AttnPrefix + 'q_proj.weight', AttnPrefix + 'q_proj.bias',
          Config.HiddenSize, HeadDim, RotaryDims);
        MarkConsumed(AttnPrefix + 'q_proj.weight');
        MarkConsumed(AttnPrefix + 'q_proj.bias');
        LoadPhiPartialRotaryLinearWeights(Reader, Blocks[BlockCnt].KProj,
          AttnPrefix + 'k_proj.weight', AttnPrefix + 'k_proj.bias',
          Config.HiddenSize, HeadDim, RotaryDims);
        MarkConsumed(AttnPrefix + 'k_proj.weight');
        MarkConsumed(AttnPrefix + 'k_proj.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].VProj,
          AttnPrefix + 'v_proj.weight', Config.HiddenSize,
          Config.HiddenSize, 0, -1, 0, AttnPrefix + 'v_proj.bias');
        MarkConsumed(AttnPrefix + 'v_proj.weight');
        MarkConsumed(AttnPrefix + 'v_proj.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].AttnDense,
          AttnPrefix + 'dense.weight', Config.HiddenSize,
          Config.HiddenSize, 0, -1, 0, AttnPrefix + 'dense.bias');
        MarkConsumed(AttnPrefix + 'dense.weight');
        MarkConsumed(AttnPrefix + 'dense.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Fc1,
          BlockPrefix + 'mlp.fc1.weight', Config.HiddenSize,
          Config.IntermediateSize, 0, -1, 0,
          BlockPrefix + 'mlp.fc1.bias');
        MarkConsumed(BlockPrefix + 'mlp.fc1.weight');
        MarkConsumed(BlockPrefix + 'mlp.fc1.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Fc2,
          BlockPrefix + 'mlp.fc2.weight', Config.IntermediateSize,
          Config.HiddenSize, 0, -1, 0,
          BlockPrefix + 'mlp.fc2.bias');
        MarkConsumed(BlockPrefix + 'mlp.fc2.weight');
        MarkConsumed(BlockPrefix + 'mlp.fc2.bias');
      end;
      LoadLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'final_layernorm.weight',
        Config.Prefix + 'final_layernorm.bias', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'final_layernorm.weight');
      MarkConsumed(Config.Prefix + 'final_layernorm.bias');

      // ---------------- Unexpected-tensor check ----------------
      // Every tensor must be consumed or be a known ignorable buffer:
      //  - rotary_emb.inv_freq: RoPE buffers some transformers versions
      //    persist (structural, rebuilt from the config's base).
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        if Pos('rotary_emb.inv_freq', TensorNameStr) > 0 then continue;
        ImportError('Phi import: unexpected tensor "' + TensorNameStr +
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

function BuildPhiFromSafeTensorsEx(const FileName: string;
  out Config: TPhiConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadPhiConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildPhiFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly);
end;

function BuildPhiFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;
var
  IgnoredConfig: TPhiConfig;
begin
  Result := BuildPhiFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly);
end;

// ============================ BERT IMPORT ==================================

function ReadBertConfigFromJSONFile(const FileName: string): TBertConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, HiddenAct: string;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('BERT import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('BERT import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

begin
  if not FileExists(FileName) then
    ImportError('BERT import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('BERT import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('BERT import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'bert');
    if ModelType = 'bert' then
      Result.Family := bfBert
    else if ModelType = 'distilbert' then
      Result.Family := bfDistilBert
    else if ModelType = 'roberta' then
      Result.Family := bfRoberta
    else
      ImportError('BERT import: config model_type is "' + ModelType +
        '" - expected "bert", "distilbert" or "roberta" (see ' +
        'BuildFromPretrained for the full dispatch).');
    if Result.Family = bfDistilBert then
    begin
      // DistilBertConfig spells everything differently and hardcodes the
      // LayerNorm eps (no layer_norm_eps key exists).
      Result.HiddenSize := RequiredInt('dim');
      Result.IntermediateSize := RequiredInt('hidden_dim');
      Result.NumLayers := RequiredInt('n_layers');
      Result.NumHeads := RequiredInt('n_heads');
      Result.VocabSize := RequiredInt('vocab_size');
      Result.MaxPositions := RequiredInt('max_position_embeddings');
      Result.TypeVocabSize := 0; // no token-type embeddings in DistilBERT
      Result.LayerNormEps := 1.0e-12;
      HiddenAct := Obj.Get('activation', 'gelu');
    end
    else
    begin
      Result.HiddenSize := RequiredInt('hidden_size');
      Result.IntermediateSize := RequiredInt('intermediate_size');
      Result.NumLayers := RequiredInt('num_hidden_layers');
      Result.NumHeads := RequiredInt('num_attention_heads');
      Result.VocabSize := RequiredInt('vocab_size');
      Result.MaxPositions := RequiredInt('max_position_embeddings');
      Result.TypeVocabSize := Obj.Get('type_vocab_size', 2);
      if Result.TypeVocabSize < 1 then
        ImportError('BERT import: config type_vocab_size must be >= 1, got ' +
          IntToStr(Result.TypeVocabSize) + '.');
      Result.LayerNormEps := Obj.Get('layer_norm_eps', 1.0e-12);
      HiddenAct := Obj.Get('hidden_act', 'gelu');
    end;
    // RoBERTa: real-token position ids start at padding_idx+1 (HF
    // create_position_ids_from_input_ids), so the first usable row of the
    // checkpoint's position table is pad_token_id+1 = 2 for every
    // published RoBERTa config (pad_token_id 1).
    if Result.Family = bfRoberta then
    begin
      Result.PositionOffset := Obj.Get('pad_token_id', 1) + 1;
      if (Result.PositionOffset < 1) or
         (Result.PositionOffset >= Result.MaxPositions) then
        ImportError('BERT import: roberta pad_token_id ' +
          IntToStr(Result.PositionOffset - 1) + ' leaves no usable ' +
          'positions (max_position_embeddings=' +
          IntToStr(Result.MaxPositions) + ').');
    end
    else
      Result.PositionOffset := 0;
    if HiddenAct = 'gelu' then
      Result.HiddenActTanh := False // exact erf form (the BERT default)
    else if (HiddenAct = 'gelu_new') or
            (HiddenAct = 'gelu_pytorch_tanh') then
      Result.HiddenActTanh := True
    else
      ImportError('BERT import: config hidden_act "' + HiddenAct +
        '" is not supported - only "gelu" (exact), "gelu_new" and ' +
        '"gelu_pytorch_tanh" are wired here.');
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function BertConfigToString(const Config: TBertConfig): string;
begin
  if Config.Family = bfDistilBert then
    Result := 'distilbert config: layers='
  else if Config.Family = bfRoberta then
    Result := 'roberta config: layers='
  else
    Result := 'bert config: layers=';
  Result := Result + IntToStr(Config.NumLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', max_pos=' + IntToStr(Config.MaxPositions) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', type_vocab=' + IntToStr(Config.TypeVocabSize) +
    ', ln_eps=' + FloatToStr(Config.LayerNormEps);
  if Config.HiddenActTanh then
    Result := Result + ', act=gelu_tanh'
  else
    Result := Result + ', act=gelu_exact';
  if Config.PositionOffset > 0 then
    Result := Result + ', pos_offset=' + IntToStr(Config.PositionOffset);
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

type
  TBertBlockLayers = record
    QKV, AttnDense, AttnLN, Inter, OutDense, OutLN: TNNetLayer;
  end;

function BuildBertFromSafeTensorsWithConfig(const FileName: string;
  var Config: TBertConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pIncludePooler: boolean = false;
  pSeqClsHead: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TBertBlockLayers;
  InputLayer, WordEmb, PosEmb, TypeEmb, EmbLN: TNNetLayer;
  PoolerDense, ClassifierDense, ClsHeadDense: TNNetLayer;
  ClsHeadName, ClsOutName: string;
  IncludePooler: boolean;
  BranchInput, SliceLayer, HiddenAct, PhiBranch: TNNetLayer;
  BlockCnt, SeqLen, UsablePositions, NumLabels, i: integer;
  BlockPrefix, TensorNameStr, FamilyPrefix: string;
  QName, KName, VName, AttnDenseName, AttnLNName: string;
  InterName, OutDenseName, OutLNName: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

  // Loads a [Rows, Cols] embedding table straight into the single-neuron
  // weight volume of a TNNetEmbedding / TNNetLearnedPositionalEmbedding
  // (row-major in both the checkpoint and the layer). SrcRowOffset > 0
  // SKIPS the first SrcRowOffset checkpoint rows (the layer table then has
  // Rows - SrcRowOffset rows): the RoBERTa position-table case, where rows
  // 0..PositionOffset-1 are padding-position rows that are NEVER read.
  procedure LoadEmbeddingTable(Layer: TNNetLayer; const TName: string;
    Rows, Cols: integer; SrcRowOffset: integer = 0);
  var
    Tmp: TNNetVolume;
    ElementCnt, UsedSize: integer;
  begin
    if not Reader.HasTensor(TName) then
      ImportError('BERT import: missing tensor "' + TName + '".');
    if (Reader.DimCount(TName) <> 2) or
       (Reader.DimSize(TName, 0) <> Rows) or
       (Reader.DimSize(TName, 1) <> Cols) then
      ImportError('BERT import: "' + TName + '" must have shape [' +
        IntToStr(Rows) + ', ' + IntToStr(Cols) + '], got ' +
        Reader.ShapeAsString(TName));
    Tmp := TNNetVolume.Create;
    try
      Reader.LoadTensorFlat(TName, Tmp);
      UsedSize := (Rows - SrcRowOffset) * Cols;
      if Layer.Neurons[0].Weights.Size <> UsedSize then
        ImportError('BERT import: "' + TName + '" usable element count ' +
          IntToStr(UsedSize) + ' does not match the table size ' +
          IntToStr(Layer.Neurons[0].Weights.Size) + '.');
      if SrcRowOffset = 0 then
        Layer.Neurons[0].Weights.Copy(Tmp)
      else
        for ElementCnt := 0 to UsedSize - 1 do
          Layer.Neurons[0].Weights.FData[ElementCnt] :=
            Tmp.FData[SrcRowOffset * Cols + ElementCnt];
      Layer.FlushWeightCache();
    finally
      Tmp.Free;
    end;
    MarkConsumed(TName);
  end;

begin
  Reader := CreatePretrainedTensorReader(FileName);
  NN := nil;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      // ---------------- Config validation & prefix detection -------------
      if Config.NumHeads < 1 then
        ImportError('BERT import: num_attention_heads must be >= 1.');
      if (Config.HiddenSize mod Config.NumHeads) <> 0 then
        ImportError('BERT import: hidden_size=' +
          IntToStr(Config.HiddenSize) + ' is not divisible by ' +
          'num_attention_heads=' + IntToStr(Config.NumHeads) + '.');
      if (Config.Family = bfBert) and (Config.TypeVocabSize < 1) then
        ImportError('BERT import: type_vocab_size must be >= 1.');
      if (Config.Family = bfDistilBert) and pIncludePooler then
        ImportError('BERT import: DistilBERT has no pooler head - ' +
          'pIncludePooler is not available for model_type "distilbert".');
      // Only the bert classifier sits on tanh(pooler(h)) - that head
      // forces the pooler. DistilBERT (pre_classifier + ReLU) and RoBERTa
      // (classifier.dense + tanh + classifier.out_proj) classify the raw
      // [CLS] / <s> hidden state: no pooler exists in those checkpoints.
      IncludePooler := pIncludePooler or
        (pSeqClsHead and (Config.Family = bfBert));
      // The plain *Model classes serialize without a prefix; *For* exports
      // carry "bert." / "distilbert." / "roberta.". Support both.
      if Config.Family = bfDistilBert then
        FamilyPrefix := 'distilbert.'
      else if Config.Family = bfRoberta then
        FamilyPrefix := 'roberta.'
      else
        FamilyPrefix := 'bert.';
      if Reader.HasTensor('embeddings.word_embeddings.weight') then
        Config.Prefix := ''
      else if Reader.HasTensor(
        FamilyPrefix + 'embeddings.word_embeddings.weight') then
        Config.Prefix := FamilyPrefix
      else
        ImportError('BERT import: neither ' +
          '"embeddings.word_embeddings.weight" nor "' +
          FamilyPrefix + 'embeddings.word_embeddings.weight" found in ' +
          Reader.FileName + ' - not a checkpoint of this family?');
      if (Config.PositionOffset < 0) or
         (Config.PositionOffset >= Config.MaxPositions) then
        ImportError('BERT import: PositionOffset=' +
          IntToStr(Config.PositionOffset) + ' leaves no usable positions ' +
          '(max_position_embeddings=' + IntToStr(Config.MaxPositions) +
          ').');
      // RoBERTa burns the first PositionOffset (= pad_token_id+1 = 2) rows
      // of the position table on padding positions: the usable context is
      // SHORTER than the table (the 514-rows-for-512-positions layout).
      UsablePositions := Config.MaxPositions - Config.PositionOffset;
      if pSeqLen <= 0 then SeqLen := UsablePositions
      else SeqLen := pSeqLen;
      if SeqLen > UsablePositions then
        ImportError('BERT import: requested SeqLen=' + IntToStr(SeqLen) +
          ' exceeds the usable context ' + IntToStr(UsablePositions) +
          ' (max_position_embeddings=' + IntToStr(Config.MaxPositions) +
          ' minus position offset ' + IntToStr(Config.PositionOffset) +
          ').');

      // ---------------- Architecture ----------------
      NN := TNNet.Create();
      // Channel 0 = token ids, channel 1 = token-type (segment) ids
      // (IGNORED when the family has no token-type embeddings - the
      // distilbert case; the shape stays (SeqLen,1,2) for API uniformity).
      InputLayer := NN.AddLayer( TNNetInput.Create(SeqLen, 1, 2) );
      // Word branch: tokens -> embedding -> + learned absolute positions.
      // EncodeZero=1: token id 0 is [PAD], a REAL row in the BERT vocab.
      NN.AddLayer( TNNetSplitChannels.Create([0]) );
      WordEmb := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      // The structural table only carries the USABLE rows: row p is
      // checkpoint row p + PositionOffset (0 for bert/distilbert).
      PosEmb := NN.AddLayer(
        TNNetLearnedPositionalEmbedding.Create(UsablePositions) );
      TypeEmb := nil;
      if Config.TypeVocabSize > 0 then
      begin
        // Token-type branch: segment ids -> their own embedding table,
        // then word + position + token-type summed.
        SliceLayer := NN.AddLayerAfter(
          TNNetSplitChannels.Create([1]), InputLayer);
        TypeEmb := NN.AddLayerAfter( TNNetEmbedding.Create(
          Config.TypeVocabSize, Config.HiddenSize, {EncodeZero=}1),
          SliceLayer);
        NN.AddLayer( TNNetSum.Create([PosEmb, TypeEmb]) );
      end;
      // ... then the embedding LayerNorm.
      EmbLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      SetLength(Blocks, Config.NumLayers);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // POST-LN attention sub-block:
        //   x := LN(x + dense(BIDIRECTIONAL-MHA(q|k|v(x)))).
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].QKV := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(3 * Config.HiddenSize) );
        // CausalMask=false: every position attends to all SeqLen keys.
        Blocks[BlockCnt].AttnDense := NN.AddMultiHeadSelfAttention(
          Config.NumHeads, {CausalMask=}false);
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        Blocks[BlockCnt].AttnLN := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        // POST-LN FFN sub-block:
        //   x := LN(x + output.dense(gelu(intermediate.dense(x)))).
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].Inter := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.IntermediateSize) );
        if Config.HiddenActTanh then
          NN.AddLayer( TNNetGELU.Create() ) // tanh approximation
        else
        begin
          // EXACT erf GELU x*Phi(x) composed from existing layers (see the
          // unit header): Phi = 0.5*(1 + erf(x/sqrt(2))) on a side branch,
          // then ReGLU(Phi|x) = ReLU(Phi)*x = Phi*x since Phi is in (0,1).
          // (TNNetReGLU applies the ReLU to the FIRST depth half, the
          // Shazeer max(0, xW) (x) xV convention - Phi must come first.)
          HiddenAct := NN.GetLastLayer();
          NN.AddLayerAfter(
            TNNetMulByConstant.Create(0.7071067811865476), HiddenAct);
          NN.AddLayer( TNNetErf.Create() );
          NN.AddLayer( TNNetAddConstant.Create(1.0) );
          PhiBranch := NN.AddLayer( TNNetMulByConstant.Create(0.5) );
          NN.AddLayer( TNNetDeepConcat.Create([PhiBranch, HiddenAct]) );
          NN.AddLayer( TNNetReGLU.Create() );
        end;
        Blocks[BlockCnt].OutDense := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        Blocks[BlockCnt].OutLN := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        if pInferenceOnly then NN.MakeInferenceOnly();
      end;
      // No final norm and NO LM head: the encoder output IS the last
      // block's LN output, (SeqLen,1,hidden) final hidden states.
      PoolerDense := nil;
      if IncludePooler then
      begin
        // Per-token dense+tanh; row 0 ([CLS]) equals HF's pooler_output.
        PoolerDense := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        NN.AddLayer( TNNetHyperbolicTangent.Create() );
      end;
      ClassifierDense := nil;
      ClsHeadDense := nil;
      NumLabels := 0;
      if pSeqClsHead then
      begin
        // Family-specific *ForSequenceClassification stacks, all wired PER
        // TOKEN so row 0 (the [CLS] / <s> position) carries HF's logits:
        //   bert:       classifier(tanh(pooler(h)))         (pooler above)
        //   distilbert: classifier(ReLU(pre_classifier(h))) (no pooler)
        //   roberta:    classifier.out_proj(tanh(classifier.dense(h)))
        if Config.Family = bfDistilBert then
        begin
          ClsHeadName := 'pre_classifier';
          ClsOutName := 'classifier';
        end
        else if Config.Family = bfRoberta then
        begin
          ClsHeadName := 'classifier.dense';
          ClsOutName := 'classifier.out_proj';
        end
        else
        begin
          ClsHeadName := ''; // bert: the pooler already played this role
          ClsOutName := 'classifier';
        end;
        if ClsHeadName <> '' then
        begin
          if not Reader.HasTensor(ClsHeadName + '.weight') then
            ImportError('BERT import: missing tensor "' + ClsHeadName +
              '.weight" - not a *ForSequenceClassification checkpoint of ' +
              'this family?');
          ClsHeadDense := NN.AddLayer(
            TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
          if Config.Family = bfDistilBert then
            NN.AddLayer( TNNetReLU.Create() )
          else
            NN.AddLayer( TNNetHyperbolicTangent.Create() );
        end;
        if not Reader.HasTensor(ClsOutName + '.weight') then
          ImportError('BERT import: missing tensor "' + ClsOutName +
            '.weight" - not a *ForSequenceClassification checkpoint?');
        if (Reader.DimCount(ClsOutName + '.weight') <> 2) or
           (Reader.DimSize(ClsOutName + '.weight', 1) <>
            Config.HiddenSize) then
          ImportError('BERT import: "' + ClsOutName + '.weight" must ' +
            'have shape [num_labels, ' + IntToStr(Config.HiddenSize) +
            '] (nn.Linear stores [out, in]), got ' +
            Reader.ShapeAsString(ClsOutName + '.weight'));
        NumLabels := Reader.DimSize(ClsOutName + '.weight', 0);
        ClassifierDense := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(NumLabels) );
      end;
      if pInferenceOnly then NN.MakeInferenceOnly();

      // ---------------- Weights ----------------
      LoadEmbeddingTable(WordEmb,
        Config.Prefix + 'embeddings.word_embeddings.weight',
        Config.VocabSize, Config.HiddenSize);
      // PositionOffset > 0 (roberta): checkpoint rows 0..PositionOffset-1
      // are padding-position rows and are NEVER read.
      LoadEmbeddingTable(PosEmb,
        Config.Prefix + 'embeddings.position_embeddings.weight',
        Config.MaxPositions, Config.HiddenSize, Config.PositionOffset);
      if Config.TypeVocabSize > 0 then
        LoadEmbeddingTable(TypeEmb,
          Config.Prefix + 'embeddings.token_type_embeddings.weight',
          Config.TypeVocabSize, Config.HiddenSize);
      LoadLayerNormWeights(Reader, EmbLN,
        Config.Prefix + 'embeddings.LayerNorm.weight',
        Config.Prefix + 'embeddings.LayerNorm.bias', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'embeddings.LayerNorm.weight');
      MarkConsumed(Config.Prefix + 'embeddings.LayerNorm.bias');
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // Same six weight slots in both families - only the names differ.
        if Config.Family = bfDistilBert then
        begin
          BlockPrefix := Config.Prefix + 'transformer.layer.' +
            IntToStr(BlockCnt) + '.';
          QName := BlockPrefix + 'attention.q_lin';
          KName := BlockPrefix + 'attention.k_lin';
          VName := BlockPrefix + 'attention.v_lin';
          AttnDenseName := BlockPrefix + 'attention.out_lin';
          AttnLNName := BlockPrefix + 'sa_layer_norm';
          InterName := BlockPrefix + 'ffn.lin1';
          OutDenseName := BlockPrefix + 'ffn.lin2';
          OutLNName := BlockPrefix + 'output_layer_norm';
        end
        else
        begin
          BlockPrefix := Config.Prefix + 'encoder.layer.' +
            IntToStr(BlockCnt) + '.';
          QName := BlockPrefix + 'attention.self.query';
          KName := BlockPrefix + 'attention.self.key';
          VName := BlockPrefix + 'attention.self.value';
          AttnDenseName := BlockPrefix + 'attention.output.dense';
          AttnLNName := BlockPrefix + 'attention.output.LayerNorm';
          InterName := BlockPrefix + 'intermediate.dense';
          OutDenseName := BlockPrefix + 'output.dense';
          OutLNName := BlockPrefix + 'output.LayerNorm';
        end;
        // Separate biased nn.Linear q/k/v -> the fused Q|K|V slab (query ->
        // neurons 0..d-1, key -> d..2d-1, value -> 2d..3d-1; the builder's
        // per-head slicing then matches HF's transpose_for_scores).
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].QKV,
          QName + '.weight',
          Config.HiddenSize, Config.HiddenSize, 0, 3 * Config.HiddenSize,
          0, QName + '.bias');
        MarkConsumed(QName + '.weight');
        MarkConsumed(QName + '.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].QKV,
          KName + '.weight',
          Config.HiddenSize, Config.HiddenSize, Config.HiddenSize,
          3 * Config.HiddenSize, 0, KName + '.bias');
        MarkConsumed(KName + '.weight');
        MarkConsumed(KName + '.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].QKV,
          VName + '.weight',
          Config.HiddenSize, Config.HiddenSize, 2 * Config.HiddenSize,
          3 * Config.HiddenSize, 0,
          VName + '.bias');
        MarkConsumed(VName + '.weight');
        MarkConsumed(VName + '.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].AttnDense,
          AttnDenseName + '.weight',
          Config.HiddenSize, Config.HiddenSize, 0, -1, 0,
          AttnDenseName + '.bias');
        MarkConsumed(AttnDenseName + '.weight');
        MarkConsumed(AttnDenseName + '.bias');
        LoadLayerNormWeights(Reader, Blocks[BlockCnt].AttnLN,
          AttnLNName + '.weight',
          AttnLNName + '.bias',
          Config.HiddenSize);
        MarkConsumed(AttnLNName + '.weight');
        MarkConsumed(AttnLNName + '.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Inter,
          InterName + '.weight',
          Config.HiddenSize, Config.IntermediateSize, 0, -1, 0,
          InterName + '.bias');
        MarkConsumed(InterName + '.weight');
        MarkConsumed(InterName + '.bias');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].OutDense,
          OutDenseName + '.weight',
          Config.IntermediateSize, Config.HiddenSize, 0, -1, 0,
          OutDenseName + '.bias');
        MarkConsumed(OutDenseName + '.weight');
        MarkConsumed(OutDenseName + '.bias');
        LoadLayerNormWeights(Reader, Blocks[BlockCnt].OutLN,
          OutLNName + '.weight',
          OutLNName + '.bias', Config.HiddenSize);
        MarkConsumed(OutLNName + '.weight');
        MarkConsumed(OutLNName + '.bias');
      end;
      if IncludePooler then
      begin
        LoadLlamaLinearWeights(Reader, PoolerDense,
          Config.Prefix + 'pooler.dense.weight',
          Config.HiddenSize, Config.HiddenSize, 0, -1, 0,
          Config.Prefix + 'pooler.dense.bias');
        MarkConsumed(Config.Prefix + 'pooler.dense.weight');
        MarkConsumed(Config.Prefix + 'pooler.dense.bias');
      end;
      if pSeqClsHead then
      begin
        // The classifier tensors sit at the top level (no family prefix).
        if ClsHeadDense <> nil then
        begin
          LoadLlamaLinearWeights(Reader, ClsHeadDense,
            ClsHeadName + '.weight', Config.HiddenSize, Config.HiddenSize,
            0, -1, 0, ClsHeadName + '.bias');
          MarkConsumed(ClsHeadName + '.weight');
          MarkConsumed(ClsHeadName + '.bias');
        end;
        LoadLlamaLinearWeights(Reader, ClassifierDense,
          ClsOutName + '.weight', Config.HiddenSize, NumLabels, 0, -1, 0,
          ClsOutName + '.bias');
        MarkConsumed(ClsOutName + '.weight');
        MarkConsumed(ClsOutName + '.bias');
      end;

      // ---------------- Unexpected-tensor check ----------------
      // Every tensor must be consumed or be a known ignorable buffer:
      //  - embeddings.position_ids: the positional-index buffer older
      //    transformers versions serialize (positions are structural here);
      //  - pooler.dense.*: the pooler head when pIncludePooler=false (a
      //    real weight this net deliberately does not carry).
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        if Pos('embeddings.position_ids', TensorNameStr) > 0 then continue;
        if (Config.Family in [bfBert, bfRoberta]) and
           (not IncludePooler) and
           (Pos('pooler.dense.', TensorNameStr) > 0) then continue;
        ImportError('BERT import: unexpected tensor "' + TensorNameStr +
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

function BuildBertFromSafeTensorsEx(const FileName: string;
  out Config: TBertConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pIncludePooler: boolean = false;
  const ConfigFileName: string = ''): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadBertConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildBertFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, pIncludePooler);
end;

function BuildBertFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pIncludePooler: boolean = false): TNNet;
var
  IgnoredConfig: TBertConfig;
begin
  Result := BuildBertFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, pIncludePooler);
end;

// ======================= SENTENCE EMBEDDINGS ===============================

function BertTokenizeSentence(Tokenizer: TNeuralHFTokenizer;
  const Text: string; MaxTokens: integer = 0): TNeuralIntegerArray;
var
  ContentIds: TNeuralIntegerArray;
  ClsId, SepId, ContentLen, Cnt: integer;
begin
  ClsId := Tokenizer.TokenToId('[CLS]');
  SepId := Tokenizer.TokenToId('[SEP]');
  if (ClsId < 0) or (SepId < 0) then
    ImportError('BertTokenizeSentence: tokenizer has no [CLS]/[SEP] ' +
      'special tokens (not a BERT-family tokenizer.json?).');
  if MaxTokens = 1 then
    ImportError('BertTokenizeSentence: MaxTokens must be 0 or >= 2 ' +
      '([CLS] and [SEP] alone need two positions).');
  ContentIds := Tokenizer.Encode(Text);
  ContentLen := Length(ContentIds);
  if (MaxTokens > 0) and (ContentLen > MaxTokens - 2) then
    ContentLen := MaxTokens - 2; // truncate, keep [CLS]/[SEP]
  SetLength(Result, ContentLen + 2);
  Result[0] := ClsId;
  for Cnt := 0 to ContentLen - 1 do Result[Cnt + 1] := ContentIds[Cnt];
  Result[ContentLen + 1] := SepId;
end;

procedure BertPoolSentenceEmbedding(HiddenStates: TNNetVolume;
  RealTokens: integer; Embedding: TNNetVolume);
var
  SeqLen, Depth, PosCnt, ChanCnt: integer;
  Sum, Norm: TNeuralFloat;
begin
  SeqLen := HiddenStates.SizeX;
  Depth := HiddenStates.Depth;
  if (HiddenStates.SizeY <> 1) or (SeqLen < 1) or (Depth < 1) then
    ImportError('BertPoolSentenceEmbedding: expected a (SeqLen,1,hidden) ' +
      'volume, got ' + IntToStr(SeqLen) + 'x' +
      IntToStr(HiddenStates.SizeY) + 'x' + IntToStr(Depth) + '.');
  if (RealTokens < 1) or (RealTokens > SeqLen) then
    ImportError('BertPoolSentenceEmbedding: RealTokens ' +
      IntToStr(RealTokens) + ' out of range 1..' + IntToStr(SeqLen) + '.');
  Embedding.ReSize(1, 1, Depth);
  // manual sum/count over the REAL rows only (never TNNetAvgChannel: it
  // averages all SeqLen rows, pads included)
  for ChanCnt := 0 to Depth - 1 do
  begin
    Sum := 0;
    for PosCnt := 0 to RealTokens - 1 do
      Sum := Sum + HiddenStates.FData[PosCnt * Depth + ChanCnt];
    Embedding.FData[ChanCnt] := Sum / RealTokens;
  end;
  Norm := 0;
  for ChanCnt := 0 to Depth - 1 do
    Norm := Norm + Embedding.FData[ChanCnt] * Embedding.FData[ChanCnt];
  Norm := Sqrt(Norm);
  if Norm > 0 then
    for ChanCnt := 0 to Depth - 1 do
      Embedding.FData[ChanCnt] := Embedding.FData[ChanCnt] / Norm;
end;

procedure BertEncodeSentence(Net: TNNet; Tokenizer: TNeuralHFTokenizer;
  const Text: string; Embedding: TNNetVolume);
var
  TokenIds: TNeuralIntegerArray;
  Input, Hidden: TNNetVolume;
  SeqLen, PadId, PosCnt: integer;
begin
  SeqLen := Net.Layers[0].Output.SizeX;
  if Net.Layers[0].Output.Depth <> 2 then
    ImportError('BertEncodeSentence: net input is not (SeqLen,1,2) - not ' +
      'a BuildBertFromSafeTensors encoder?');
  TokenIds := BertTokenizeSentence(Tokenizer, Text, SeqLen);
  PadId := Tokenizer.TokenToId('[PAD]');
  if PadId < 0 then PadId := 0; // BERT-family [PAD] is id 0 anyway
  Input := TNNetVolume.Create();
  Hidden := TNNetVolume.Create();
  try
    Input.ReSize(SeqLen, 1, 2);
    Input.Fill(0); // token-type channel: single segment = all zeros
    for PosCnt := 0 to SeqLen - 1 do
      if PosCnt < Length(TokenIds) then
        Input.FData[PosCnt * 2] := TokenIds[PosCnt]
      else
        Input.FData[PosCnt * 2] := PadId;
    Net.Compute(Input);
    Net.GetOutput(Hidden);
    BertPoolSentenceEmbedding(Hidden, Length(TokenIds), Embedding);
  finally
    Hidden.Free;
    Input.Free;
  end;
end;

// =================== SEQUENCE CLASSIFICATION IMPORT =======================

function ReadId2LabelFromJSONFile(const FileName: string): TStringList;
var
  JsonText: TStringList;
  Root, MapData: TJSONData;
  MapObj: TJSONObject;
  ClassId, EntryCnt: integer;
begin
  if not FileExists(FileName) then
    ImportError('id2label import: config file not found: ' + FileName);
  Result := TStringList.Create;
  JsonText := TStringList.Create;
  Root := nil;
  try
    try
      JsonText.LoadFromFile(FileName);
      try
        Root := GetJSON(JsonText.Text);
      except
        on E: Exception do
          ImportError('id2label import: config "' + FileName +
            '" is not valid JSON (' + E.Message + ').');
      end;
      if not (Root is TJSONObject) then
        ImportError('id2label import: config "' + FileName +
          '" is not a JSON object.');
      MapData := TJSONObject(Root).Find('id2label');
      if MapData = nil then exit; // no map: empty list (LABEL_i fallback)
      if not (MapData is TJSONObject) then
        ImportError('id2label import: "id2label" in ' + FileName +
          ' is not a JSON object.');
      MapObj := TJSONObject(MapData);
      // Pre-size, then place each label at its integer key: the JSON map
      // {"0": ..} must cover exactly 0..Count-1 (HF's contiguous ids).
      for EntryCnt := 0 to MapObj.Count - 1 do Result.Add('');
      for EntryCnt := 0 to MapObj.Count - 1 do
      begin
        ClassId := StrToIntDef(MapObj.Names[EntryCnt], -1);
        if (ClassId < 0) or (ClassId >= MapObj.Count) then
          ImportError('id2label import: id "' + MapObj.Names[EntryCnt] +
            '" in ' + FileName + ' is not in 0..' +
            IntToStr(MapObj.Count - 1) + ' (non-contiguous map).');
        if Result[ClassId] <> '' then
          ImportError('id2label import: duplicate id ' +
            IntToStr(ClassId) + ' in ' + FileName + '.');
        Result[ClassId] := MapObj.Items[EntryCnt].AsString;
      end;
    except
      Result.Free;
      raise;
    end;
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function ClassIndexToLabel(Id2Label: TStringList;
  ClassIndex: integer): string;
begin
  if (Id2Label <> nil) and (ClassIndex >= 0) and
     (ClassIndex < Id2Label.Count) then
    Result := Id2Label[ClassIndex]
  else
    Result := 'LABEL_' + IntToStr(ClassIndex); // the HF default naming
end;

procedure SelectTokenLogits(NetOutput: TNNetVolume; TokenPos: integer;
  Logits: TNNetVolume);
var
  ChanCnt, LabelCnt: integer;
begin
  if NetOutput.SizeY <> 1 then
    ImportError('SelectTokenLogits: expected a (SeqLen,1,num_labels) ' +
      'volume, got SizeY=' + IntToStr(NetOutput.SizeY) + '.');
  if (TokenPos < 0) or (TokenPos >= NetOutput.SizeX) then
    ImportError('SelectTokenLogits: TokenPos ' + IntToStr(TokenPos) +
      ' is outside 0..' + IntToStr(NetOutput.SizeX - 1) + '.');
  LabelCnt := NetOutput.Depth;
  Logits.ReSize(1, 1, LabelCnt);
  for ChanCnt := 0 to LabelCnt - 1 do
    Logits.FData[ChanCnt] := NetOutput.FData[TokenPos * LabelCnt + ChanCnt];
end;

function BuildBertForSequenceClassificationFromSafeTensorsEx(
  const FileName: string; out Config: TBertConfig; Id2Label: TStringList;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;
var
  ConfigPath: string;
  Labels: TStringList;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadBertConfigFromJSONFile(ConfigPath);
  // Labels first: a label-parse failure must not leak a built net.
  if Id2Label <> nil then
  begin
    Labels := ReadId2LabelFromJSONFile(ConfigPath);
    try
      Id2Label.Assign(Labels);
    finally
      Labels.Free;
    end;
  end;
  // pIncludePooler stays false: the builder forces the pooler on for the
  // families whose head needs it (bert only - distilbert/roberta
  // classifier checkpoints carry no pooler tensors).
  Result := BuildBertFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, {pIncludePooler=}false, {pSeqClsHead=}true);
end;

function BuildBertForSequenceClassificationFromSafeTensors(
  const FileName: string; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false): TNNet;
var
  IgnoredConfig: TBertConfig;
begin
  Result := BuildBertForSequenceClassificationFromSafeTensorsEx(FileName,
    IgnoredConfig, nil, pSeqLen, pInferenceOnly);
end;

function BuildGPT2ForSequenceClassificationFromSafeTensorsEx(
  const FileName: string; out Config: TGPT2Config; Id2Label: TStringList;
  pSeqLen: integer = 0; pNumHeads: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;
var
  ConfigPath: string;
  Labels: TStringList;
begin
  // Labels first (same leak-avoidance order as the BERT variant). The
  // config.json is ONLY touched for id2label: the trunk config is
  // inferred from the checkpoint tensors (ReadGPT2Config).
  if Id2Label <> nil then
  begin
    if ConfigFileName <> '' then ConfigPath := ConfigFileName
    else ConfigPath := ExtractFilePath(FileName) + 'config.json';
    Labels := ReadId2LabelFromJSONFile(ConfigPath);
    try
      Id2Label.Assign(Labels);
    finally
      Labels.Free;
    end;
  end;
  Result := BuildGPT2FromSafeTensorsEx(FileName, Config, pSeqLen,
    pNumHeads, pInferenceOnly, {pSeqClsHead=}true);
end;

function BuildGPT2ForSequenceClassificationFromSafeTensors(
  const FileName: string; pSeqLen: integer = 0; pNumHeads: integer = 0;
  pInferenceOnly: boolean = false): TNNet;
var
  IgnoredConfig: TGPT2Config;
begin
  Result := BuildGPT2ForSequenceClassificationFromSafeTensorsEx(FileName,
    IgnoredConfig, nil, pSeqLen, pNumHeads, pInferenceOnly);
end;

// ============= MISTRAL / QWEN2 / QWEN3 / GEMMA / DISPATCH ==================

// Shared wrapper: builds via the Llama path and asserts the config's
// model_type is the expected family.
function BuildLlamaFamilyFromSafeTensors(const FileName: string;
  const ExpectedModelType: string; out Config: TLlamaConfig;
  pSeqLen: integer; pInferenceOnly: boolean;
  const ConfigFileName: string): TNNet;
begin
  Result := BuildLlamaFromSafeTensorsEx(FileName, Config, pSeqLen,
    pInferenceOnly, ConfigFileName);
  if Config.ModelType <> ExpectedModelType then
  begin
    Result.Free;
    Result := nil;
    ImportError(ExpectedModelType + ' import: config model_type is "' +
      Config.ModelType + '", expected "' + ExpectedModelType +
      '" - use the matching builder or BuildFromPretrained.');
  end;
end;

function BuildMistralFromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'mistral', Config,
    pSeqLen, pInferenceOnly, ConfigFileName);
end;

function BuildMistralFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildMistralFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly);
end;

function BuildQwen2FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'qwen2', Config,
    pSeqLen, pInferenceOnly, ConfigFileName);
end;

function BuildQwen2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildQwen2FromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly);
end;

function BuildQwen3FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'qwen3', Config,
    pSeqLen, pInferenceOnly, ConfigFileName);
end;

function BuildQwen3FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildQwen3FromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly);
end;

function BuildGemmaFromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'gemma', Config,
    pSeqLen, pInferenceOnly, ConfigFileName);
end;

function BuildGemmaFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildGemmaFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly);
end;

function LoadPretrainedEmbedding(const FileName: string;
  Embedding: TNNetEmbedding; Tokenizer: TStringListInt;
  FreezeEmbedding: boolean = false): integer;
var
  F: TextFile;
  Weights: TNNetVolume;
  Matched: array of boolean;
  Row: array of TNeuralFloat;
  Line, Field: string;
  FS: TFormatSettings;
  LineNo, EmbDim, VocabRows, TokenIdx, DimCnt, RowCnt: integer;
  MatchedCnt, HeaderCount, HeaderDim: integer;
  MeanVal: TNeuralFloat;
  FileOpen: boolean;

  // Returns the next whitespace-separated field of Line starting at P
  // (1-based); '' once the line is exhausted. Spaces and tabs separate.
  function NextField(var P: integer): string;
  begin
    while (P <= Length(Line)) and (Line[P] in [' ', #9]) do Inc(P);
    Result := '';
    while (P <= Length(Line)) and not (Line[P] in [' ', #9]) do
    begin
      Result := Result + Line[P];
      Inc(P);
    end;
  end;

  // Parses the D vector floats following the word at position P and, when
  // the word is a vocabulary token, copies them into its embedding row.
  procedure ParseDataLine(P: integer; const WordStr: string);
  var
    DimCnt: integer;
  begin
    for DimCnt := 0 to EmbDim - 1 do
    begin
      Field := NextField(P);
      if Field = '' then
        ImportError('Embedding import: line ' + IntToStr(LineNo) + ' of ' +
          FileName + ' has fewer than ' + IntToStr(EmbDim) +
          ' vector values (word "' + WordStr + '").');
      try
        Row[DimCnt] := StrToFloat(Field, FS);
      except
        on E: Exception do
          ImportError('Embedding import: line ' + IntToStr(LineNo) + ' of ' +
            FileName + ': "' + Field + '" is not a number (word "' + WordStr +
            '").');
      end;
    end;
    if NextField(P) <> '' then
      ImportError('Embedding import: line ' + IntToStr(LineNo) + ' of ' +
        FileName + ' has more than ' + IntToStr(EmbDim) +
        ' vector values (word "' + WordStr + '") - wrong embedding width?');
    TokenIdx := Tokenizer.WordToIndex(WordStr);
    if (TokenIdx >= 0) and (TokenIdx < VocabRows) then
    begin
      if not Matched[TokenIdx] then Inc(MatchedCnt);
      Matched[TokenIdx] := true;
      for DimCnt := 0 to EmbDim - 1 do
        Weights.FData[TokenIdx * EmbDim + DimCnt] := Row[DimCnt];
    end;
  end;

  // True when S is a plain unsigned integer (a "count dim" header field).
  function IsPlainInteger(const S: string): boolean;
  var
    CharPos: integer;
  begin
    Result := S <> '';
    for CharPos := 1 to Length(S) do
      if not (S[CharPos] in ['0'..'9']) then Result := false;
  end;

var
  P: integer;
  WordStr, SecondField, ThirdField: string;
begin
  Result := 0;
  if Embedding = nil then
    ImportError('Embedding import: Embedding is nil.');
  if Tokenizer = nil then
    ImportError('Embedding import: Tokenizer is nil.');
  if not FileExists(FileName) then
    ImportError('Embedding import: file not found: ' + FileName);
  EmbDim := Embedding.EmbeddingSize;
  VocabRows := Embedding.VocabSize;
  Weights := Embedding.Neurons[0].Weights;
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.'; // the format is locale-independent
  SetLength(Matched, VocabRows);
  for RowCnt := 0 to VocabRows - 1 do Matched[RowCnt] := false;
  SetLength(Row, EmbDim);
  MatchedCnt := 0;
  LineNo := 0;
  AssignFile(F, FileName);
  Reset(F);
  FileOpen := true;
  try
    while not Eof(F) do
    begin
      ReadLn(F, Line);
      Inc(LineNo);
      P := 1;
      WordStr := NextField(P);
      if WordStr = '' then continue; // blank line
      if LineNo = 1 then
      begin
        // Optional .vec header: exactly two integer fields "count dim".
        SecondField := NextField(P);
        ThirdField := NextField(P);
        if (ThirdField = '') and IsPlainInteger(WordStr) and
           IsPlainInteger(SecondField) then
        begin
          HeaderCount := StrToInt(WordStr);
          HeaderDim := StrToInt(SecondField);
          if HeaderDim <> EmbDim then
            ImportError('Embedding import: ' + FileName + ' declares ' +
              IntToStr(HeaderDim) + '-dim vectors (header "' + WordStr + ' ' +
              SecondField + '") but the embedding layer is ' +
              IntToStr(EmbDim) + '-dim.');
          if HeaderCount <= 0 then
            ImportError('Embedding import: ' + FileName +
              ' declares a non-positive word count in its header.');
          continue;
        end;
        P := 1;
        WordStr := NextField(P); // not a header: reparse as a data line
      end;
      ParseDataLine(P, WordStr);
    end;
    CloseFile(F);
    FileOpen := false;
    // Mean-init the vocabulary rows the file did not cover: with at least
    // one match, the mean vector is a better prior than the random init.
    if (MatchedCnt > 0) and (MatchedCnt < VocabRows) then
    begin
      for DimCnt := 0 to EmbDim - 1 do
      begin
        MeanVal := 0;
        for RowCnt := 0 to VocabRows - 1 do
          if Matched[RowCnt] then
            MeanVal := MeanVal + Weights.FData[RowCnt * EmbDim + DimCnt];
        MeanVal := MeanVal / MatchedCnt;
        for RowCnt := 0 to VocabRows - 1 do
          if not Matched[RowCnt] then
            Weights.FData[RowCnt * EmbDim + DimCnt] := MeanVal;
      end;
    end;
    Embedding.FlushWeightCache();
    if FreezeEmbedding then Embedding.LearningRate := 0;
    Result := MatchedCnt;
  finally
    if FileOpen then CloseFile(F);
  end;
end;

// ---------------------------------------------------------------------------
// T5 / FLAN-T5 IMPORT implementation (see the interface section).
// ---------------------------------------------------------------------------

function ReadT5ConfigFromJSONFile(const FileName: string): TT5Config;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, FFProj: string;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('T5 import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('T5 import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

begin
  if not FileExists(FileName) then
    ImportError('T5 import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('T5 import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('T5 import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 't5');
    if ModelType <> 't5' then
      ImportError('T5 import: config model_type is "' + ModelType +
        '" - only "t5" is supported here (see BuildFromPretrained for ' +
        'the full dispatch).');
    Result.ModelType := ModelType;
    Result.DModel := RequiredInt('d_model');
    Result.DKV := RequiredInt('d_kv');
    Result.DFF := RequiredInt('d_ff');
    Result.NumLayers := RequiredInt('num_layers');
    Result.NumHeads := RequiredInt('num_heads');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.NumDecoderLayers :=
      Obj.Get('num_decoder_layers', Result.NumLayers);
    Result.RelPosNumBuckets :=
      Obj.Get('relative_attention_num_buckets', 32);
    Result.RelPosMaxDistance :=
      Obj.Get('relative_attention_max_distance', 128);
    Result.LayerNormEps := Obj.Get('layer_norm_epsilon', 1.0e-6);
    // The HF T5Config default is TRUE: the original T5 ties the head (and
    // scales the decoder output by d_model^-0.5); Flan-T5/v1.1 say false.
    Result.TieWordEmbeddings := Obj.Get('tie_word_embeddings', True);
    // Only the two shipped T5 FFN recipes are supported: "relu" (v1.0,
    // wo(relu(wi(x)))) and "gated-gelu" (v1.1/Flan, gelu_new(wi_0)*wi_1).
    // HF's plain "gelu" would be the EXACT erf form (dense_act_fn stays
    // "gelu"), which no released T5 uses - rejected to avoid a silent
    // erf-vs-tanh mismatch.
    FFProj := Obj.Get('feed_forward_proj', 'relu');
    if FFProj = 'relu' then
      Result.GatedFFN := False
    else if FFProj = 'gated-gelu' then
      Result.GatedFFN := True
    else
      ImportError('T5 import: feed_forward_proj "' + FFProj +
        '" is not supported - expected "relu" (T5 v1.0) or "gated-gelu" ' +
        '(T5 v1.1 / Flan-T5).');
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function T5ConfigToString(const Config: TT5Config): string;
begin
  if Config.ModelType = '' then Result := 't5' else Result := Config.ModelType;
  Result := Result + ' config: enc_layers=' + IntToStr(Config.NumLayers) +
    ', dec_layers=' + IntToStr(Config.NumDecoderLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', d_model=' + IntToStr(Config.DModel) +
    ', d_kv=' + IntToStr(Config.DKV) +
    ', d_ff=' + IntToStr(Config.DFF) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', rel_buckets=' + IntToStr(Config.RelPosNumBuckets) +
    ', rel_max_dist=' + IntToStr(Config.RelPosMaxDistance) +
    ', ln_eps=' + FloatToStr(Config.LayerNormEps) +
    ', tied=' + BoolToStr(Config.TieWordEmbeddings, true);
  if Config.GatedFFN then
    Result := Result + ', gated_ffn=true';
end;

type
  TT5AttnLayers = record
    Norm, QProj, KProj, VProj, OProj: TNNetLayer;
    Heads: array of TNNetLayer; // per-head attention leaves
  end;
  TT5BlockLayers = record
    SelfAttn: TT5AttnLayers;
    CrossAttn: TT5AttnLayers;   // decoder blocks only
    FFNNorm, Wi, Wo: TNNetLayer;
  end;
  TT5BlockArray = array of TT5BlockLayers;

// Grows one T5 stack (encoder or decoder) onto NN after the embedding.
// IsDecoder adds the causal mask on self-attention plus the cross-attention
// sub-block reading Keys|Values from EncStates (the decoder net's second
// TNNetInput). Layer handles for the weight loader are returned in Blocks.
procedure BuildT5StackBlocks(NN: TNNet; const Config: TT5Config;
  NumBlocks: integer; IsDecoder: boolean; EncStates: TNNetLayer;
  var Blocks: TT5BlockArray; pInferenceOnly: boolean);
var
  InnerDim, BlockCnt, HeadCnt, d: integer;
  BranchInput, NormedSource: TNNetLayer;
  QSlice, KSlice, VSlice, HeadPack, KVPack: TNNetLayer;
  SliceChannels: array of integer;
begin
  InnerDim := Config.NumHeads * Config.DKV;
  SetLength(Blocks, NumBlocks);
  SetLength(SliceChannels, Config.DKV);
  for BlockCnt := 0 to NumBlocks - 1 do
  begin
    // ---- self-attention sub-block (pre-norm, residual after) ----
    // Wired from primitives like the Llama importer; the per-head leaf is
    // TNNetT5RelPosBiasAttention so every head carries its own copy of the
    // stack's SHARED bias table column (loaded below). The decoder is
    // causal (= HF bidirectional=False bucketing); the encoder is not.
    BranchInput := NN.GetLastLayer();
    Blocks[BlockCnt].SelfAttn.Norm :=
      NN.AddLayer( TNNetTokenRMSNorm.Create(Config.LayerNormEps) );
    NormedSource := NN.GetLastLayer();
    Blocks[BlockCnt].SelfAttn.QProj := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(InnerDim), NormedSource);
    Blocks[BlockCnt].SelfAttn.KProj := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(InnerDim), NormedSource);
    Blocks[BlockCnt].SelfAttn.VProj := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(InnerDim), NormedSource);
    SetLength(Blocks[BlockCnt].SelfAttn.Heads, Config.NumHeads);
    for HeadCnt := 0 to Config.NumHeads - 1 do
    begin
      for d := 0 to Config.DKV - 1 do
        SliceChannels[d] := HeadCnt * Config.DKV + d;
      QSlice := NN.AddLayerAfter(
        TNNetSplitChannels.Create(SliceChannels),
        Blocks[BlockCnt].SelfAttn.QProj);
      KSlice := NN.AddLayerAfter(
        TNNetSplitChannels.Create(SliceChannels),
        Blocks[BlockCnt].SelfAttn.KProj);
      VSlice := NN.AddLayerAfter(
        TNNetSplitChannels.Create(SliceChannels),
        Blocks[BlockCnt].SelfAttn.VProj);
      HeadPack := NN.AddLayer(
        TNNetDeepConcat.Create([QSlice, KSlice, VSlice]) );
      Blocks[BlockCnt].SelfAttn.Heads[HeadCnt] := NN.AddLayerAfter(
        TNNetT5RelPosBiasAttention.Create(Config.DKV,
          {pCausalMask=}IsDecoder, Config.RelPosNumBuckets,
          Config.RelPosMaxDistance), HeadPack);
    end;
    NN.AddLayer( TNNetDeepConcat.Create(Blocks[BlockCnt].SelfAttn.Heads) );
    Blocks[BlockCnt].SelfAttn.OProj := NN.AddLayer(
      TNNetPointwiseConvLinear.Create(Config.DModel) );
    NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );

    // ---- cross-attention sub-block (decoder only) ----
    // Queries from the (normed) decoder stream; Keys|Values projected from
    // EncStates - the two grids may have DIFFERENT lengths, so the per-head
    // leaf is TNNetCrossAttention (rectangular DecSeqLen x EncSeqLen
    // scores; a Q|K|V DeepConcat would be illegal across unequal SizeX).
    // NO relative-position bias here (the T5 cross-attention has none).
    if IsDecoder then
    begin
      BranchInput := NN.GetLastLayer();
      Blocks[BlockCnt].CrossAttn.Norm :=
        NN.AddLayer( TNNetTokenRMSNorm.Create(Config.LayerNormEps) );
      NormedSource := NN.GetLastLayer();
      Blocks[BlockCnt].CrossAttn.QProj := NN.AddLayerAfter(
        TNNetPointwiseConvLinear.Create(InnerDim), NormedSource);
      Blocks[BlockCnt].CrossAttn.KProj := NN.AddLayerAfter(
        TNNetPointwiseConvLinear.Create(InnerDim), EncStates);
      Blocks[BlockCnt].CrossAttn.VProj := NN.AddLayerAfter(
        TNNetPointwiseConvLinear.Create(InnerDim), EncStates);
      SetLength(Blocks[BlockCnt].CrossAttn.Heads, Config.NumHeads);
      for HeadCnt := 0 to Config.NumHeads - 1 do
      begin
        for d := 0 to Config.DKV - 1 do
          SliceChannels[d] := HeadCnt * Config.DKV + d;
        QSlice := NN.AddLayerAfter(
          TNNetSplitChannels.Create(SliceChannels),
          Blocks[BlockCnt].CrossAttn.QProj);
        KSlice := NN.AddLayerAfter(
          TNNetSplitChannels.Create(SliceChannels),
          Blocks[BlockCnt].CrossAttn.KProj);
        VSlice := NN.AddLayerAfter(
          TNNetSplitChannels.Create(SliceChannels),
          Blocks[BlockCnt].CrossAttn.VProj);
        KVPack := NN.AddLayer( TNNetDeepConcat.Create([KSlice, VSlice]) );
        Blocks[BlockCnt].CrossAttn.Heads[HeadCnt] := NN.AddLayerAfter(
          TNNetCrossAttention.Create(Config.DKV, {CausalMask=}false,
            KVPack), QSlice);
      end;
      NN.AddLayer( TNNetDeepConcat.Create(Blocks[BlockCnt].CrossAttn.Heads) );
      Blocks[BlockCnt].CrossAttn.OProj := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.DModel) );
      NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
    end;

    // ---- FFN sub-block ----
    // "gated-gelu": h = gelu_new(wi_0(x)) * wi_1(x) - TNNetGEGLU computes
    // FIRSTHALF * GELU(SECONDHALF) with the tanh approximation (= HF
    // gelu_new), so the fused projection holds wi_1 in neurons 0..d_ff-1
    // and wi_0 in neurons d_ff..2*d_ff-1 (see the weight loader).
    // "relu": h = relu(wi(x)). Both end with wo back to d_model.
    BranchInput := NN.GetLastLayer();
    Blocks[BlockCnt].FFNNorm :=
      NN.AddLayer( TNNetTokenRMSNorm.Create(Config.LayerNormEps) );
    if Config.GatedFFN then
    begin
      Blocks[BlockCnt].Wi := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(2 * Config.DFF) );
      NN.AddLayer( TNNetGEGLU.Create() );
    end
    else
    begin
      Blocks[BlockCnt].Wi := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.DFF) );
      NN.AddLayer( TNNetReLU.Create() );
    end;
    Blocks[BlockCnt].Wo := NN.AddLayer(
      TNNetPointwiseConvLinear.Create(Config.DModel) );
    NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
    if pInferenceOnly then NN.MakeInferenceOnly();
  end;
  SetLength(SliceChannels, 0);
end;

// Copies the stack's SHARED relative-position bias table
// ([NumBuckets, NumHeads], stored only in block 0 in HF checkpoints) into
// head h of EVERY layer of the stack: bias[bucket] = T[bucket, h]. The
// per-layer-per-head copies strictly generalize T5's sharing, so loading
// the same column everywhere reproduces it exactly.
procedure LoadT5RelPosBias(Reader: TNNetSafeTensorsReader;
  const Blocks: TT5BlockArray; const WName: string; const Config: TT5Config);
var
  Tmp: TNNetVolume;
  BlockCnt, HeadCnt, BucketCnt: integer;
  HeadLayer: TNNetLayer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('T5 import: missing tensor "' + WName + '".');
  if (Reader.DimCount(WName) <> 2) or
     (Reader.DimSize(WName, 0) <> Config.RelPosNumBuckets) or
     (Reader.DimSize(WName, 1) <> Config.NumHeads) then
    ImportError('T5 import: "' + WName + '" must have shape [' +
      IntToStr(Config.RelPosNumBuckets) + ', ' +
      IntToStr(Config.NumHeads) + '], got ' + Reader.ShapeAsString(WName));
  Tmp := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, Tmp);
    for BlockCnt := 0 to High(Blocks) do
      for HeadCnt := 0 to Config.NumHeads - 1 do
      begin
        HeadLayer := Blocks[BlockCnt].SelfAttn.Heads[HeadCnt];
        if HeadLayer.Neurons[0].Weights.Size <> Config.RelPosNumBuckets then
          ImportError('T5 import: internal error - relpos head neuron for "' +
            WName + '" has ' +
            IntToStr(HeadLayer.Neurons[0].Weights.Size) +
            ' weights, expected ' + IntToStr(Config.RelPosNumBuckets) + '.');
        for BucketCnt := 0 to Config.RelPosNumBuckets - 1 do
          HeadLayer.Neurons[0].Weights.FData[BucketCnt] :=
            Tmp.FData[BucketCnt * Config.NumHeads + HeadCnt];
        HeadLayer.FlushWeightCache();
      end;
  finally
    Tmp.Free;
  end;
end;

// Loads one attention sub-block's q/k/v/o (HF nn.Linear [out, in],
// bias-free). q gets the sqrt(d_kv) compensation folded in: T5 computes
// UNSCALED q.k scores while the CAI attention leaves scale by 1/sqrt(d_k).
procedure LoadT5AttnLinears(Reader: TNNetSafeTensorsReader;
  const Attn: TT5AttnLayers; const AttnPrefix: string;
  const Config: TT5Config; Consumed: TStringList);
var
  InnerDim: integer;
begin
  InnerDim := Config.NumHeads * Config.DKV;
  LoadLlamaLinearWeights(Reader, Attn.QProj, AttnPrefix + 'q.weight',
    Config.DModel, InnerDim, 0, -1, 0, '', Sqrt(Config.DKV));
  Consumed.Add(AttnPrefix + 'q.weight');
  LoadLlamaLinearWeights(Reader, Attn.KProj, AttnPrefix + 'k.weight',
    Config.DModel, InnerDim);
  Consumed.Add(AttnPrefix + 'k.weight');
  LoadLlamaLinearWeights(Reader, Attn.VProj, AttnPrefix + 'v.weight',
    Config.DModel, InnerDim);
  Consumed.Add(AttnPrefix + 'v.weight');
  LoadLlamaLinearWeights(Reader, Attn.OProj, AttnPrefix + 'o.weight',
    InnerDim, Config.DModel);
  Consumed.Add(AttnPrefix + 'o.weight');
end;

// Loads one full T5 stack: per-block sublayer norms + attention linears +
// FFN linears, plus the stack's single shared relative-position bias table.
// The decoder's sublayer indices are 0=self-attn, 1=cross-attn, 2=FFN; the
// encoder's are 0=self-attn, 1=FFN.
procedure LoadT5Stack(Reader: TNNetSafeTensorsReader;
  const Blocks: TT5BlockArray; const StackPrefix: string;
  const Config: TT5Config; IsDecoder: boolean; Consumed: TStringList);
var
  BlockCnt, FFNIdx: integer;
  BP, FFNPrefix: string;
begin
  if IsDecoder then FFNIdx := 2 else FFNIdx := 1;
  for BlockCnt := 0 to High(Blocks) do
  begin
    BP := StackPrefix + 'block.' + IntToStr(BlockCnt) + '.';
    LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].SelfAttn.Norm,
      BP + 'layer.0.layer_norm.weight', Config.DModel);
    Consumed.Add(BP + 'layer.0.layer_norm.weight');
    LoadT5AttnLinears(Reader, Blocks[BlockCnt].SelfAttn,
      BP + 'layer.0.SelfAttention.', Config, Consumed);
    if IsDecoder then
    begin
      LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].CrossAttn.Norm,
        BP + 'layer.1.layer_norm.weight', Config.DModel);
      Consumed.Add(BP + 'layer.1.layer_norm.weight');
      LoadT5AttnLinears(Reader, Blocks[BlockCnt].CrossAttn,
        BP + 'layer.1.EncDecAttention.', Config, Consumed);
    end;
    FFNPrefix := BP + 'layer.' + IntToStr(FFNIdx) + '.';
    LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].FFNNorm,
      FFNPrefix + 'layer_norm.weight', Config.DModel);
    Consumed.Add(FFNPrefix + 'layer_norm.weight');
    if Config.GatedFFN then
    begin
      // TNNetGEGLU = FIRSTHALF * GELU(SECONDHALF); HF gated-gelu =
      // gelu_new(wi_0) * wi_1 -> wi_1 into neurons 0..d_ff-1 (linear
      // half), wi_0 into neurons d_ff..2*d_ff-1 (gated half).
      LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Wi,
        FFNPrefix + 'DenseReluDense.wi_1.weight',
        Config.DModel, Config.DFF, 0, 2 * Config.DFF);
      Consumed.Add(FFNPrefix + 'DenseReluDense.wi_1.weight');
      LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Wi,
        FFNPrefix + 'DenseReluDense.wi_0.weight',
        Config.DModel, Config.DFF, Config.DFF, 2 * Config.DFF);
      Consumed.Add(FFNPrefix + 'DenseReluDense.wi_0.weight');
    end
    else
    begin
      LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Wi,
        FFNPrefix + 'DenseReluDense.wi.weight', Config.DModel, Config.DFF);
      Consumed.Add(FFNPrefix + 'DenseReluDense.wi.weight');
    end;
    LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Wo,
      FFNPrefix + 'DenseReluDense.wo.weight', Config.DFF, Config.DModel);
    Consumed.Add(FFNPrefix + 'DenseReluDense.wo.weight');
  end;
  // The stack's ONE shared bias table (block 0 only in the checkpoint),
  // copied into every layer's heads.
  LoadT5RelPosBias(Reader, Blocks,
    StackPrefix + 'block.0.layer.0.SelfAttention.' +
    'relative_attention_bias.weight', Config);
  Consumed.Add(StackPrefix + 'block.0.layer.0.SelfAttention.' +
    'relative_attention_bias.weight');
end;

procedure BuildT5FromSafeTensorsWithConfig(const FileName: string;
  var Config: TT5Config; out EncoderNet, DecoderNet: TNNet;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false);
var
  Reader: TNNetSafeTensorsReader;
  Enc, Dec: TNNet;
  EncBlocks, DecBlocks: TT5BlockArray;
  EncEmbed, DecEmbed, EncFinalNorm, DecFinalNorm: TNNetLayer;
  DecTokenInput, EncStates, LMHead: TNNetLayer;
  Consumed: TStringList;
  Tmp: TNNetVolume;
  i, j: integer;
  TieScale: TNeuralFloat;
  TensorNameStr: string;
begin
  EncoderNet := nil;
  DecoderNet := nil;
  // ---------------- Config validation ----------------
  if EncSeqLen < 1 then
    ImportError('T5 import: EncSeqLen must be >= 1, got ' +
      IntToStr(EncSeqLen) + '.');
  if DecSeqLen < 1 then
    ImportError('T5 import: DecSeqLen must be >= 1, got ' +
      IntToStr(DecSeqLen) + '.');
  if Config.NumHeads < 1 then
    ImportError('T5 import: num_heads must be >= 1.');
  if Config.DKV < 1 then
    ImportError('T5 import: d_kv must be >= 1.');
  if (Config.RelPosNumBuckets < 4) or Odd(Config.RelPosNumBuckets) then
    ImportError('T5 import: relative_attention_num_buckets must be an ' +
      'EVEN integer >= 4, got ' + IntToStr(Config.RelPosNumBuckets) + '.');
  if Config.RelPosMaxDistance <= Config.RelPosNumBuckets div 2 then
    ImportError('T5 import: relative_attention_max_distance must exceed ' +
      'num_buckets/2, got ' + IntToStr(Config.RelPosMaxDistance) + '.');
  Reader := CreatePretrainedTensorReader(FileName);
  Enc := nil;
  Dec := nil;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      if not Reader.HasTensor('shared.weight') then
        ImportError('T5 import: "shared.weight" not found in ' +
          Reader.FileName + ' - not a T5 checkpoint?');
      if (Reader.DimCount('shared.weight') <> 2) or
         (Reader.DimSize('shared.weight', 0) <> Config.VocabSize) or
         (Reader.DimSize('shared.weight', 1) <> Config.DModel) then
        ImportError('T5 import: shared.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.DModel) +
          '], got ' + Reader.ShapeAsString('shared.weight'));
      if (not Config.TieWordEmbeddings) and
         (not Reader.HasTensor('lm_head.weight')) then
        ImportError('T5 import: config says tie_word_embeddings=false ' +
          'but "lm_head.weight" is missing from ' + Reader.FileName + '.');

      // ---------------- Encoder architecture ----------------
      Enc := TNNet.Create();
      Enc.AddLayer( TNNetInput.Create(EncSeqLen) );
      // EncodeZero=1: token id 0 (<pad>, also the decoder start token) is
      // a real embedding row.
      EncEmbed := Enc.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.DModel, {EncodeZero=}1) );
      if pInferenceOnly then Enc.MakeInferenceOnly();
      BuildT5StackBlocks(Enc, Config, Config.NumLayers, {IsDecoder=}false,
        nil, EncBlocks, pInferenceOnly);
      EncFinalNorm := Enc.AddLayer(
        TNNetTokenRMSNorm.Create(Config.LayerNormEps) );
      if pInferenceOnly then Enc.MakeInferenceOnly();

      // ---------------- Decoder architecture ----------------
      // TWO inputs: Layers[0] = decoder token ids (what Compute feeds);
      // the second TNNetInput holds the encoder hidden states and is
      // filled MANUALLY before Compute (the TNNetCrossAttention two-source
      // convention - see TestMultiHeadCrossAttentionGradientCheck).
      Dec := TNNet.Create();
      DecTokenInput := Dec.AddLayer( TNNetInput.Create(DecSeqLen) );
      EncStates := Dec.AddLayerAfter(
        TNNetInput.Create(EncSeqLen, 1, Config.DModel, 1), 0);
      DecEmbed := Dec.AddLayerAfter( TNNetEmbedding.Create(
        Config.VocabSize, Config.DModel, {EncodeZero=}1), DecTokenInput);
      if pInferenceOnly then Dec.MakeInferenceOnly();
      BuildT5StackBlocks(Dec, Config, Config.NumDecoderLayers,
        {IsDecoder=}true, EncStates, DecBlocks, pInferenceOnly);
      DecFinalNorm := Dec.AddLayer(
        TNNetTokenRMSNorm.Create(Config.LayerNormEps) );
      LMHead := Dec.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then Dec.MakeInferenceOnly();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        Reader.LoadTensorFlat('shared.weight', Tmp);
        if EncEmbed.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('T5 import: shared.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table ' +
            'size ' + IntToStr(EncEmbed.Neurons[0].Weights.Size) + '.');
        EncEmbed.Neurons[0].Weights.Copy(Tmp);
        EncEmbed.FlushWeightCache();
        DecEmbed.Neurons[0].Weights.Copy(Tmp);
        DecEmbed.FlushWeightCache();
        Consumed.Add('shared.weight');
        // Legacy exports may still carry the tied embed_tokens aliases.
        if Reader.HasTensor('encoder.embed_tokens.weight') then
          Consumed.Add('encoder.embed_tokens.weight');
        if Reader.HasTensor('decoder.embed_tokens.weight') then
          Consumed.Add('decoder.embed_tokens.weight');
        if Config.TieWordEmbeddings then
        begin
          // T5 v1.0 tied head: logits = (h * d_model^-0.5) . shared^T -
          // the scale is folded into the copied rows (bias-free).
          TieScale := 1.0 / Sqrt(Config.DModel);
          for j := 0 to Config.VocabSize - 1 do
          begin
            for i := 0 to Config.DModel - 1 do
              LMHead.Neurons[j].Weights.FData[i] :=
                TieScale * Tmp.FData[j * Config.DModel + i];
            LMHead.Neurons[j].BiasWeight := 0;
          end;
          LMHead.FlushWeightCache();
          // A redundant serialized lm_head.weight is ignorable when tied.
          if Reader.HasTensor('lm_head.weight') then
            Consumed.Add('lm_head.weight');
        end
        else
        begin
          // Flan-T5 / v1.1: separate head, NO d_model^-0.5 scaling.
          LoadLlamaLinearWeights(Reader, LMHead, 'lm_head.weight',
            Config.DModel, Config.VocabSize);
          Consumed.Add('lm_head.weight');
        end;
      finally
        Tmp.Free;
      end;
      LoadT5Stack(Reader, EncBlocks, 'encoder.', Config,
        {IsDecoder=}false, Consumed);
      LoadT5Stack(Reader, DecBlocks, 'decoder.', Config,
        {IsDecoder=}true, Consumed);
      LoadLlamaRMSNormWeights(Reader, EncFinalNorm,
        'encoder.final_layer_norm.weight', Config.DModel);
      Consumed.Add('encoder.final_layer_norm.weight');
      LoadLlamaRMSNormWeights(Reader, DecFinalNorm,
        'decoder.final_layer_norm.weight', Config.DModel);
      Consumed.Add('decoder.final_layer_norm.weight');

      // ---------------- Unexpected-tensor check ----------------
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        ImportError('T5 import: unexpected tensor "' + TensorNameStr +
          '" (shape ' + Reader.ShapeAsString(TensorNameStr) +
          ') in ' + FileName + ' - refusing a partial import.');
      end;
      EncoderNet := Enc;
      DecoderNet := Dec;
      Enc := nil; // ownership transferred to the caller
      Dec := nil;
    except
      on E: ESafeTensorsError do
        raise EPretrainedImportError.Create(E.Message);
    end;
  finally
    Enc.Free; // non-nil only if an exception unwound the build
    Dec.Free;
    Consumed.Free;
    Reader.Free;
  end;
end;

procedure BuildT5FromSafeTensors(const FileName: string;
  out EncoderNet, DecoderNet: TNNet; out Config: TT5Config;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  const ConfigFileName: string = '');
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadT5ConfigFromJSONFile(ConfigPath);
  BuildT5FromSafeTensorsWithConfig(FileName, Config, EncoderNet,
    DecoderNet, EncSeqLen, DecSeqLen, pInferenceOnly);
end;

function T5EncoderStatesInput(DecoderNet: TNNet): TNNetLayer;
var
  LayerCnt, InputCnt: integer;
begin
  Result := nil;
  InputCnt := 0;
  for LayerCnt := 0 to DecoderNet.Layers.Count - 1 do
    if DecoderNet.Layers[LayerCnt] is TNNetInput then
    begin
      Inc(InputCnt);
      if InputCnt = 2 then
      begin
        Result := DecoderNet.Layers[LayerCnt];
        exit;
      end;
    end;
  ImportError('T5EncoderStatesInput: the net has no second TNNetInput - ' +
    'not a BuildT5FromSafeTensors decoder?');
end;

procedure RunT5(EncoderNet, DecoderNet: TNNet;
  EncoderTokens, DecoderTokens: TNNetVolume; Logits: TNNetVolume);
var
  EncStates: TNNetLayer;
begin
  EncStates := T5EncoderStatesInput(DecoderNet);
  EncoderNet.Compute(EncoderTokens);
  if EncoderNet.GetLastLayer().Output.Size <> EncStates.Output.Size then
    ImportError('RunT5: encoder output size ' +
      IntToStr(EncoderNet.GetLastLayer().Output.Size) +
      ' does not match the decoder''s encoder-states input size ' +
      IntToStr(EncStates.Output.Size) + ' (EncSeqLen/d_model mismatch?).');
  EncStates.Output.Copy(EncoderNet.GetLastLayer().Output);
  DecoderNet.Compute(DecoderTokens);
  DecoderNet.GetOutput(Logits);
end;

function BuildFromPretrained(const Path: string; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''): TNNet;
var
  ConfigPath, WeightsPath, ModelType, ArchName: string;
  JsonText: TStringList;
  Root, ArchData: TJSONData;
  NumHeads, ArchCnt: integer;
  BertSeqCls, GPT2SeqCls, GPT2ExactGelu: boolean;
  IgnoredGPT2Config: TGPT2Config;
  IgnoredId2Label: TStringList;
  IgnoredLlamaConfig: TLlamaConfig;
  IgnoredGPTNeoConfig: TGPTNeoConfig;
  IgnoredGPTNeoXConfig: TGPTNeoXConfig;
  IgnoredGPTJConfig: TGPTJConfig;
  IgnoredPhiConfig: TPhiConfig;
  IgnoredBertConfig: TBertConfig;
begin
  // ---- resolve the weights file and the config.json ----
  if DirectoryExists(Path) then
  begin
    ConfigPath := IncludeTrailingPathDelimiter(Path) + 'config.json';
    WeightsPath := IncludeTrailingPathDelimiter(Path) +
      'model.safetensors.index.json';
    if not FileExists(WeightsPath) then
      WeightsPath := IncludeTrailingPathDelimiter(Path) + 'model.safetensors';
    if not FileExists(WeightsPath) then
      WeightsPath := IncludeTrailingPathDelimiter(Path) +
        'pytorch_model.bin'; // torch.save fallback (TNNetTorchBinReader)
    if not FileExists(WeightsPath) then
      ImportError('BuildFromPretrained: none of "model.safetensors", ' +
        '"model.safetensors.index.json" or "pytorch_model.bin" found in ' +
        'directory ' + Path + '.');
  end
  else if FileExists(Path) then
  begin
    WeightsPath := Path;
    ConfigPath := ExtractFilePath(Path) + 'config.json';
  end
  else
    ImportError('BuildFromPretrained: path not found: ' + Path);
  if ConfigFileName <> '' then ConfigPath := ConfigFileName;
  if not FileExists(ConfigPath) then
    ImportError('BuildFromPretrained: config file not found: ' + ConfigPath);

  // ---- read model_type (and the GPT-2 head count) from the config ----
  ModelType := '';
  NumHeads := 0;
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(ConfigPath);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('BuildFromPretrained: config "' + ConfigPath +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('BuildFromPretrained: config "' + ConfigPath +
        '" is not a JSON object.');
    ModelType := TJSONObject(Root).Get('model_type', '');
    NumHeads := TJSONObject(Root).Get('n_head', 0); // GPT-2 family key
    // GPT-2 family activation: OpenAI checkpoints say "gelu_new" (the tanh
    // approximation, the builder default); Cerebras-GPT - model_type "gpt2",
    // the open GPT-3 reproduction - says "gelu", the EXACT erf form.
    GPT2ExactGelu :=
      TJSONObject(Root).Get('activation_function', 'gelu_new') = 'gelu';
    // "architectures" disambiguates a fine-tuned classifier head from the
    // base LM/encoder of the same model_type (the LM stays the default).
    BertSeqCls := false;
    GPT2SeqCls := false;
    ArchData := TJSONObject(Root).Find('architectures');
    if (ArchData <> nil) and (ArchData is TJSONArray) then
      for ArchCnt := 0 to TJSONArray(ArchData).Count - 1 do
      begin
        ArchName := TJSONArray(ArchData).Items[ArchCnt].AsString;
        if (ArchName = 'BertForSequenceClassification') or
           (ArchName = 'DistilBertForSequenceClassification') or
           (ArchName = 'RobertaForSequenceClassification') then
          BertSeqCls := true
        else if ArchName = 'GPT2ForSequenceClassification' then
          GPT2SeqCls := true;
      end;
  finally
    Root.Free;
    JsonText.Free;
  end;

  // ---- dispatch ----
  if (ModelType = 'gpt2') and GPT2SeqCls then
  begin
    IgnoredId2Label := nil;
    Result := BuildGPT2ForSequenceClassificationFromSafeTensorsEx(
      WeightsPath, IgnoredGPT2Config, IgnoredId2Label, pSeqLen, NumHeads,
      pInferenceOnly, ConfigPath);
  end
  else if ModelType = 'gpt2' then
    Result := BuildGPT2FromSafeTensors(WeightsPath, pSeqLen, NumHeads,
      pInferenceOnly, GPT2ExactGelu)
  else if ((ModelType = 'bert') or (ModelType = 'distilbert') or
           (ModelType = 'roberta')) and BertSeqCls then
  begin
    IgnoredId2Label := nil;
    Result := BuildBertForSequenceClassificationFromSafeTensorsEx(
      WeightsPath, IgnoredBertConfig, IgnoredId2Label, pSeqLen,
      pInferenceOnly, ConfigPath);
  end
  else if ModelType = 'gpt_neo' then
    Result := BuildGPTNeoFromSafeTensorsEx(WeightsPath, IgnoredGPTNeoConfig,
      pSeqLen, pInferenceOnly, ConfigPath)
  else if ModelType = 'gpt_neox' then
    Result := BuildGPTNeoXFromSafeTensorsEx(WeightsPath, IgnoredGPTNeoXConfig,
      pSeqLen, pInferenceOnly, ConfigPath)
  else if ModelType = 'gptj' then
    Result := BuildGPTJFromSafeTensorsEx(WeightsPath, IgnoredGPTJConfig,
      pSeqLen, pInferenceOnly, ConfigPath)
  else if ModelType = 'phi' then
    Result := BuildPhiFromSafeTensorsEx(WeightsPath, IgnoredPhiConfig,
      pSeqLen, pInferenceOnly, ConfigPath)
  else if (ModelType = 'llama') or (ModelType = 'mistral') or
          (ModelType = 'qwen2') or (ModelType = 'qwen3') or
          (ModelType = 'gemma') then
    // 'gemma' (architectures ["GemmaForCausalLM"]) rides the same path: the
    // config reader raises the Gemma flags (GeGLU FFN, zero-centered
    // RMSNorm, sqrt(d) embedding scale) from model_type alone.
    Result := BuildLlamaFromSafeTensorsEx(WeightsPath, IgnoredLlamaConfig,
      pSeqLen, pInferenceOnly, ConfigPath)
  else if (ModelType = 'bert') or (ModelType = 'distilbert') or
          (ModelType = 'roberta') then
    // ENCODER route: input (SeqLen,1,2) token|token-type ids (channel 1
    // ignored for distilbert, zeros for roberta), output (SeqLen,1,hidden)
    // final hidden states - NOT causal-LM logits (see the interface
    // comment). Pooler excluded; call BuildBertFromSafeTensors directly to
    // include it (bert/roberta - distilbert has none).
    Result := BuildBertFromSafeTensorsEx(WeightsPath, IgnoredBertConfig,
      pSeqLen, pInferenceOnly, {pIncludePooler=}false, ConfigPath)
  else if ModelType = 't5' then
  begin
    // T5 is an encoder-decoder: the import builds TWO nets, which this
    // single-net dispatch cannot return.
    Result := nil;
    ImportError('BuildFromPretrained: model_type "t5" builds an ' +
      'ENCODER-DECODER pair - call BuildT5FromSafeTensors (returns both ' +
      'nets; run them with RunT5) instead of this single-net dispatch.');
  end
  else
  begin
    Result := nil;
    ImportError('BuildFromPretrained: model_type "' + ModelType +
      '" (config ' + ConfigPath + ') is not supported. Supported ' +
      'model_types: gpt2, gpt_neo, gpt_neox, gptj, phi, llama, mistral, ' +
      'qwen2, qwen3, gemma, bert, distilbert, roberta.');
  end;
end;

end.
