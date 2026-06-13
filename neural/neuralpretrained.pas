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
//   - OLMo-2 (model_type "olmo2", the fully-open AllenAI family; the Llama
//     tensor layout with REORDERED post-norm - RMSNorm on the sublayer
//     OUTPUT before the residual add, x + Norm(Attn(x)), NO input_layernorm
//     - and q/k RMSNorm over the FULL flattened projection width BEFORE the
//     head split + RoPE, unlike Qwen3's per-head placement) -
//     BuildOlmo2FromSafeTensors.
//   - Gemma 1 (the Llama skeleton with three LOAD-TIME deltas: GeGLU MLP
//     - gated tanh-GELU, TNNetGEGLU instead of TNNetSwiGLU; zero-centered
//     RMSNorm - gains stored as 1+w; embedding output scaled by
//     sqrt(hidden_size) - folded into the embedding rows, NOT the tied LM
//     head; always-tied head; 2B is MQA via num_key_value_heads=1, 7B uses
//     the decoupled head_dim=256) - BuildGemmaFromSafeTensors.
//   - Gemma 2 (the Gemma-1 skeleton + alternating local/global attention
//     via the SDPA sliding window on even layers; query_pre_attn_scalar
//     folded into W_q; attention-logit soft-capping inside SDPA;
//     final-logit soft-capping via TNNetSoftCapping; sandwich norms - 4
//     RMSNorms per block with the post-norms inside the residual
//     branches) - BuildGemma2FromSafeTensors.
//   - Gemma 3 TEXT-ONLY (model_type "gemma3_text", e.g. gemma-3-1b-it; the
//     Gemma-2 skeleton with the soft-caps REPLACED by a per-head
//     learnable-scale RMSNorm on q/k BEFORE RoPE - the Qwen3 QKNorm
//     composition with Gemma's 1+w gains; a 5:1 local:global layer ratio
//     (every 6th layer attends globally) and PER-LAYER-TYPE RoPE theta:
//     rope_local_base_freq (10k) on sliding layers, rope_theta (1M) on
//     global layers. The 4B+ multimodal vision tower is OUT OF SCOPE) -
//     BuildGemma3FromSafeTensors.
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
//   - Phi-3 / Phi-4-mini (model_type "phi3": microsoft/Phi-3-mini-4k-
//     instruct, Phi-4-mini-instruct and siblings; the LLAMA skeleton -
//     pre-norm RMSNorm, sequential residual, SwiGLU, GQA - with FUSED
//     bias-free qkv_proj (q|k|v packed rows) and gate_up_proj (gate|up)
//     slabs plus optional PARTIAL rotary (partial_rotary_factor, 0.75 on
//     Phi-4-mini) in the rotate_half layout) - BuildPhi3FromSafeTensors,
//     a thin wrapper over the Llama path (longrope rope_scaling - the
//     128k variants - is REJECTED, not mis-loaded). See the LLAMA IMPORT
//     section below.
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
//   - Marian / OPUS-MT (model_type "marian": the Helsinki-NLP/opus-mt-*
//     translation pairs) - BuildMarianFromSafeTensors, the second
//     encoder-decoder import (POST-norm blocks, static sinusoidal
//     positions, swish FFN, tied head + final_logits_bias). Runs through
//     the same RunT5 two-net convention. See the MARIAN IMPORT section.
//   - Whisper (model_type "whisper": the openai/whisper-* speech-to-text
//     checkpoints) - BuildWhisperFromSafeTensors, the FIRST SPEECH import
//     and the third encoder-decoder: log-mel spectrogram input (computed
//     by neural/neuralaudio.pas), conv1d+GELU frontend with a stride-2
//     halving, fixed sinusoidal encoder positions, learned decoder
//     positions, PRE-norm blocks with final stack norms, bias-free
//     k_proj, exact erf GELU, tied head. Runs through the same RunT5
//     two-net convention. See the WHISPER IMPORT section.
//   - RWKV-4 (model_type "rwkv": RWKV/rwkv-4-169m-pile and siblings) -
//     BuildRWKVFromSafeTensors, the FIRST NON-TRANSFORMER import: a
//     recurrent WKV mixer (TNNetWKV + TNNetTokenShift) with constant
//     decode memory and no KV cache - token-shift lerps, squared-ReLU
//     channel mix, biased LayerNorms incl. the block-0 pre_ln, and the
//     exp/softplus decay-convention bridge. See the RWKV-4 IMPORT section.
//   - Mamba (model_type "mamba": state-spaces/mamba-130m-hf and siblings)
//     - BuildMambaFromSafeTensors, the SECOND NON-TRANSFORMER import: a
//     selective-SSM (S6) mixer (TNNetSelectiveSSM with d_state states per
//     channel) with constant decode memory and no KV cache - depthwise
//     causal conv1d + SiLU, the low-rank dt_proj folded into the scan's
//     delta projection, A_raw = A_log raw (identical discretizations),
//     SiLU(z) gating and per-token RMSNorms. See the MAMBA IMPORT section.
//   - BLOOM (model_type "bloom": bigscience/bloom-560m..bloom and the
//     instruction-tuned bloomz family) - BuildBloomFromSafeTensors, the
//     FIRST ALiBi importer: NO positional embeddings at all, per-head
//     FIXED linear attention biases (TNNetALiBiAttention with the
//     geometric HF build_alibi_tensor slopes); embedding LayerNorm right
//     after the word embeddings, sequential pre-LN GELU(tanh) blocks,
//     fused per-head [q|k|v] query_key_value, ALWAYS-tied LM head + ln_f.
//     See the BLOOM IMPORT section below.
//   - ModernBERT (model_type "modernbert": answerdotai/ModernBERT-base /
//     -large) - BuildModernBertFromSafeTensors, the SECOND encoder family
//     and the first with RoPE: NO position table at all, BIDIRECTIONAL
//     attention with rotary q/k, alternating local/global layers (every
//     3rd global; local layers carry a SYMMETRIC |i-j| <= local_attention/2
//     sliding window with its own RoPE theta), pre-LN bias-free blocks
//     (layer 0 skips attn_norm), exact-erf GeGLU MLPs (TNNetGEGLUErf) and
//     a final_norm. Outputs hidden states like the BERT family. See the
//     MODERNBERT IMPORT section below.
//   - DeepSeek-V2 (model_type "deepseek_v2"; DeepSeek-V2-Lite is the
//     reference checkpoint) - BuildDeepSeekV2FromSafeTensors, the first
//     family with Multi-head Latent Attention (low-rank compressed KV
//     latent + decoupled RoPE keys shared across heads) and DeepSeekMoE
//     blocks (shared + routed SwiGLU experts, per-token softmax top-k
//     gating; layers below first_k_dense_replace keep a dense MLP).
//     Causal-LM contract. See the DEEPSEEK-V2 IMPORT section below.
//   - CLIP (model_type "clip": openai/clip-vit-base-patch32 and siblings)
//     - BuildClipFromSafeTensors, the FIRST VISION-LANGUAGE importer and
//     the first ViT: returns TWO independent nets (the contrastive
//     dual-encoder - a CAUSAL pre-LN quick_gelu TEXT tower with learned
//     positions, eot-pooled + text_projection, and a BIDIRECTIONAL ViT
//     VISION tower - bias-free patch conv, class token folded into the
//     position table, pre_layrnorm/post_layernorm + visual_projection,
//     factored as the reusable BuildClipVisionTower). L2-normalized
//     cosine scoring via ClipExtractEmbedding/ClipSimilarity +
//     exp(logit_scale). See the CLIP IMPORT section below.
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
  neuralgguf, neuralhftokenizer;

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
// a ".bin" extension - or a ".bin.index.json" sharded-checkpoint index
// (pytorch_model.bin.index.json) - gets the restricted torch.save reader
// (TNNetTorchBinReader, neuraltorchbin.pas - pytorch_model.bin); anything
// else gets TNNetSafeTensorsReader (which itself dispatches ".json" to the
// sharded-index path). Both readers handle their sharded index
// transparently, and TNNetTorchBinReader subclasses the safetensors
// reader, so every Build*FromSafeTensors* importer in this unit accepts a
// single-file or sharded checkpoint of either format through this helper.
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
// pQuantizeInt8 = True stores the weights of every PointwiseConvLinear in
// per-output-channel symmetric int8 (TNNet.QuantizeWeightsInt8), quantizing
// block by block DURING construction and load so peak RAM carries the FP32
// weights of at most one block: the loaded net holds ~1/4 of the FP32
// weight bytes (embedding/positional tables stay FP32) and is
// inference-only. Logit drift measures ~2e-2 relative on the pico parity
// fixture (rows of 8-32 columns; much smaller at real checkpoint widths) -
// see TestInt8QuantizedGPT2LogitDrift.
function BuildGPT2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pNumHeads: integer = 0;
  pInferenceOnly: boolean = false; pExactGelu: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

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
  pSeqClsHead: boolean = false; pExactGelu: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

type
  // Parsed HF config.json "rope_scaling" (long-context extension), mapped
  // onto TNNetRotaryEmbedding's scaling-mode constructor arguments:
  //   "linear"          -> rsmPositionInterpolation (factor)
  //   "dynamic"/"ntk"   -> rsmNTKAware (factor; static approximation of the
  //                        dynamic rule at the full config factor)
  //   "yarn"            -> rsmYaRN (factor, original_max_position_embeddings,
  //                        YarnAlpha=beta_slow, YarnBeta=beta_fast)
  //   "llama3"          -> rsmLlama3 (factor, original_max_position_embeddings,
  //                        YarnAlpha=low_freq_factor, YarnBeta=high_freq_factor)
  //   "longrope"/"su"   -> rsmLongRoPE (Phi-3): per-frequency LongFactors table
  //                        (long_factor, length head_dim/2) divides each
  //                        inverse frequency, LongAttnFactor scales the output.
  // null/absent rope_scaling (and rope_type "default") parse to Mode=rsmNone,
  // which keeps the import bit-identical to the unscaled path.
  TRoPEScalingConfig = record
    Mode: TNNetRoPEScalingMode;  // rsmNone when rope_scaling is null/absent
    Factor: TNeuralFloat;        // "factor" (>= 1); 1.0 when Mode=rsmNone
    OriginalContextLen: integer; // pretraining context (YaRN/llama3 ramp)
    YarnAlpha: TNeuralFloat;     // beta_slow (yarn) / low_freq_factor (llama3)
    YarnBeta: TNeuralFloat;      // beta_fast (yarn) / high_freq_factor (llama3)
    // LongRoPE (rsmLongRoPE) only: the verbatim per-frequency long_factor
    // table (length head_dim/2) and the output attention scaling (long_mscale,
    // or sqrt(1+ln(max/orig)/ln(orig)) when no explicit mscale is given).
    LongFactors: TNeuralFloatDynArr;
    LongAttnFactor: TNeuralFloat;
  end;

// The Mode=rsmNone record (Factor=1, OriginalContextLen=0, alpha=1, beta=32).
function DefaultRoPEScaling(): TRoPEScalingConfig;

// 'none' for rsmNone, else e.g. 'linear x2', 'yarn x4 orig=128 ...'.
function RoPEScalingToString(const S: TRoPEScalingConfig): string;

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
    // ---- Gemma-2 deltas (all default off/0 for the other families) ----
    QueryPreAttnScalar: TNeuralFloat; // attention scores scaled by
                               // 1/sqrt(query_pre_attn_scalar) instead of
                               // 1/sqrt(head_dim); folded into W_q at load
                               // (0 = off)
    AttnLogitSoftCap: TNeuralFloat;  // attn_logit_softcapping: pre-softmax
                               // scores squashed to cap*tanh(s/cap) inside
                               // every SDPA head (0 = off)
    FinalLogitSoftCap: TNeuralFloat; // final_logit_softcapping: LM-head
                               // logits squashed to cap*tanh(l/cap) by a
                               // TNNetSoftCapping layer (0 = off)
    SandwichNorm: boolean;     // sandwich norms: post-attention and
                               // post-feedforward RMSNorms INSIDE each
                               // residual branch (4 norms per block)
    AltSlidingWindow: boolean; // alternating local/global attention: the
                               // sliding window applies to EVEN layers
                               // (0, 2, ...) only; odd layers attend fully
    // ---- Gemma-3 deltas (all default off/0 for the other families) ----
    SlidingWindowPattern: integer; // every Nth layer ((i+1) mod N = 0)
                               // attends GLOBALLY, the rest carry the
                               // sliding window (Gemma-3: 6, a 5:1
                               // local:global ratio); 0/1 = off (takes
                               // precedence over AltSlidingWindow when set)
    RopeLocalTheta: TNeuralFloat; // RoPE base for the SLIDING (local)
                               // layers (rope_local_base_freq, Gemma-3
                               // default 10000); global layers keep
                               // RopeTheta (1e6); 0 = single theta
    RopeScaling: TRoPEScalingConfig; // parsed "rope_scaling" (Mode=rsmNone
                               // when null/absent). With RopeLocalTheta > 0
                               // (Gemma-3) it applies to the GLOBAL layers
                               // only, matching HF (rope_local is unscaled)
    // ---- Phi-3 deltas (all default off/full for the other families) ----
    FusedQKVGateUp: boolean;   // FUSED bias-free projections (Phi-3):
                               // self_attn.qkv_proj.weight packs q|k|v rows
                               // and mlp.gate_up_proj.weight packs gate|up -
                               // sliced into the same layers at load
    PartialRotaryFactor: TNeuralFloat; // partial_rotary_factor: RoPE rotates
                               // only the first int(head_dim*factor)
                               // channels of each q/k head, the tail passes
                               // through (Phi-4-mini: 0.75); 0 or 1 = full
    // ---- OLMo-2 deltas (all default off for the other families) ----
    PostNormReordered: boolean;// REORDERED post-norm: RMSNorm on the SUBLAYER
                               // OUTPUT before the residual add -
                               // x + Norm(Attn(x)) - with NO input_layernorm
                               // and NO pre-FFN norm (HF keys
                               // post_attention_layernorm /
                               // post_feedforward_layernorm)
    QKNormFullWidth: boolean;  // RMSNorm over the FULL flattened q/k
                               // projection width (num_heads*head_dim /
                               // num_kv_heads*head_dim) AFTER the projection
                               // and BEFORE the head split + RoPE - distinct
                               // from the Qwen3/Gemma-3 PER-HEAD placement
                               // (QKNorm)
    // ---- Mixtral deltas (all default off for the other families) ----
    IsMoE: boolean;            // every FFN is a block_sparse_moe: a router
                               // (mlp gate linear -> num_local_experts
                               // logits) + N independent SwiGLU experts,
                               // softmax over ALL experts then top-k
                               // routing with the top-k subset RENORMALIZED
                               // (Mixtral; the standard top-2 sparse MoE)
    NumLocalExperts: integer;  // num_local_experts (Mixtral: 8); experts
                               // are block_sparse_moe.experts.{i}.w1/w2/w3
    MoEExpertsPerTok: integer; // num_experts_per_tok routed per token
                               // (Mixtral: 2); the renormalized top-k
    Prefix: string;            // tensor-name prefix ('model.' or '')
  end;

// Reads a HF Llama-family config.json (model_type 'llama', 'mistral',
// 'qwen2', 'qwen3', 'gemma', 'gemma2', 'gemma3_text', 'phi3' or 'olmo2').
// Required: hidden_size, intermediate_size,
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
//   gemma2:  the gemma deltas + SandwichNorm, AltSlidingWindow (1:1
//            local:global), SlidingWindow (default 4096),
//            QueryPreAttnScalar (default 256) and the two soft-caps
//            (defaults 50/30);
//   gemma3_text: the gemma2 deltas with the soft-caps defaulting OFF
//            (None in Gemma3TextConfig), QKNorm := true (per-head 1+w
//            RMSNorm on q/k BEFORE RoPE), SlidingWindowPattern
//            (sliding_window_pattern, default 6 - every 6th layer global),
//            RopeTheta default 1e6 and RopeLocalTheta
//            (rope_local_base_freq, default 10000) for the sliding layers;
//   phi3:    FusedQKVGateUp := true (fused bias-free qkv_proj / gate_up_proj
//            slabs); PartialRotaryFactor := partial_rotary_factor (default
//            1.0 = full rotary; Phi-4-mini ships 0.75); rms_norm_eps
//            defaults 1e-5 (the Phi3Config default); SlidingWindow :=
//            sliding_window like mistral (null/absent = full attention -
//            HF Phi3 windows EVERY layer through the same mask util as
//            Mistral, so the Mistral convention applies unchanged); a
//            "longrope" (or Phi-3-spelled "su"/"yarn") rope_scaling - the
//            128k checkpoints - is REJECTED with a clear error (the
//            per-frequency long/short factor tables are not standard YaRN
//            and are not wired into TNNetRotaryEmbedding);
//   llama:   QKVBias := attention_bias (default false).
// Fails on an unsupported model_type. A non-null "rope_scaling" is parsed
// into Result.RopeScaling (see TRoPEScalingConfig for the supported types:
// linear, dynamic/ntk, yarn, llama3; anything else is rejected).
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
// pQuantizeInt8 = True stores every projection/MLP/LM-head weight matrix in
// per-output-channel symmetric int8 (TNNet.QuantizeWeightsInt8), quantizing
// block by block DURING construction and load so peak RAM carries the FP32
// weights of at most one block plus one streamed tensor: the loaded net
// holds ~1/4 of the FP32 weight bytes (the embedding table stays FP32) and
// is inference-only. This is what lets a TinyLlama-1.1B-class checkpoint
// (~4.4 GB of FP32 weights) run on a 3 GB-class machine. Expect logit
// drift measures ~2e-2 relative on the pico parity fixture (much smaller
// at real checkpoint widths) - see TestInt8QuantizedLlamaLogitDrift.
// Every importer that rides this path (Mistral/Qwen/Gemma/...) inherits
// the behaviour by passing pQuantizeInt8 through.
function BuildLlamaFromSafeTensorsWithConfig(const FileName: string;
  var Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildLlamaFromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = '';
  pQuantizeInt8: boolean = false): TNNet;

function BuildLlamaFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

// GGUF IMPORT (llama.cpp checkpoints): builds the SAME Llama net as
// BuildLlamaFromSafeTensors from a .gguf file (general.architecture
// "llama"). No config.json is needed - the config travels inside the file
// as GGUF metadata, mapped onto TLlamaConfig:
//   llama.embedding_length -> hidden_size       (required)
//   llama.feed_forward_length -> intermediate_size (required)
//   llama.block_count -> num_hidden_layers      (required)
//   llama.attention.head_count -> num_attention_heads (required)
//   llama.attention.head_count_kv -> num_key_value_heads (default
//     head_count)
//   llama.context_length -> max_position_embeddings (required)
//   llama.attention.layer_norm_rms_epsilon -> rms_norm_eps (default 1e-5)
//   llama.rope.freq_base -> rope_theta (default 10000)
//   llama.attention.key_length -> head_dim (default hidden/heads)
//   llama.vocab_size -> vocab_size (default: the token_embd.weight rows,
//     itself defaulting to the tokenizer.ggml.tokens count)
//   llama.rope.scaling.type "linear" -> rope_scaling linear with
//     llama.rope.scaling.factor; absent/"none" = unscaled; anything else
//     (yarn ...) is rejected.
// tie_word_embeddings is inferred: tied iff the file has no
// "output.weight" tensor (llama.cpp omits the LM head when tied).
// The ggml tensor names (token_embd.weight, blk.N.attn_q.weight,
// output_norm.weight, ...) are renamed to their HF equivalents and the
// import reuses the exact BuildLlamaFromSafeTensorsWithConfig path - with
// ONE data fix-up: llama.cpp's convert script permutes the q/k projection
// rows from HF's rotate_half layout into the interleaved-pair rotary
// layout, so the reader de-interleaves those rows back to HF order at
// load (TNNetGGUFReader.RegisterRowDeinterleave).
// Tensor dtypes F32/F16/Q8_0 are decoded (Q8_0 dequantizes to FP32 at
// load); other ggml quantizations are rejected with a clear error.
// pInferenceOnly/pQuantizeInt8 behave exactly as in
// BuildLlamaFromSafeTensorsWithConfig (pQuantizeInt8 re-quantizes the
// dequantized weights into the int8 weight-only storage).
function BuildLlamaFromGGUFEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

function BuildLlamaFromGGUF(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

// Mistral: the Llama skeleton + optional sliding-window attention (config
// sliding_window; null/absent = full attention). Thin wrappers over the
// Llama path that ASSERT the config's model_type is 'mistral'.
function BuildMistralFromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildMistralFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

// Qwen2: the Llama skeleton + biases on the q/k/v projections (o_proj and
// the MLP stay bias-free). Thin wrappers over the Llama path that ASSERT
// the config's model_type is 'qwen2'.
function BuildQwen2FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildQwen2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

// Qwen3: the Llama skeleton WITHOUT q/k/v biases, with per-head RMSNorm on
// q and k (applied after the projection and BEFORE RoPE, gain [head_dim]
// shared across heads - q_norm.weight/k_norm.weight) and an optionally
// decoupled head_dim. Thin wrappers over the Llama path that ASSERT the
// config's model_type is 'qwen3'.
function BuildQwen3FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildQwen3FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

// OLMo-2 (model_type "olmo2", e.g. allenai/OLMo-2-0425-1B - the fully-open
// weights+data+code family): the Llama tensor layout (SwiGLU, RoPE, no
// biases) with TWO architectural deltas:
// (a) REORDERED post-norm - RMSNorm applied to the SUBLAYER OUTPUT before
//     the residual add, x + Norm(Attn(x)) / x + Norm(MLP(x)) (HF keys
//     post_attention_layernorm / post_feedforward_layernorm; there is NO
//     input_layernorm and no pre-FFN norm) - a third placement beside the
//     pre-norm (Llama) and sandwich-norm (Gemma-2) conventions (see
//     TNNet.AddOutputNormResidual for the builder form);
// (b) q/k RMSNorm over the FULL flattened projection width
//     (num_heads*head_dim / num_kv_heads*head_dim) AFTER the projection and
//     BEFORE the head split + RoPE (HF keys q_norm / k_norm) - distinct
//     from the Qwen3/Gemma-3 PER-HEAD placement whose RMS statistic spans
//     head_dim channels only.
// The final norm is the standard model.norm and the tokenizer is a stock
// GPT-NeoX-style BPE tokenizer.json (TNeuralHFTokenizer). Thin wrappers
// over the Llama path that ASSERT the config's model_type is 'olmo2'.
function BuildOlmo2FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildOlmo2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

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
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildGemmaFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

// Gemma 2: the Gemma-1 skeleton (GeGLU MLP, zero-centered RMSNorm,
// sqrt(d) embedding scale, tied head) with five further deltas:
// (a) ALTERNATING local/global attention - config sliding_window (HF
//     default 4096) masks EVEN layers (0, 2, ...) with the banded causal
//     sliding window; odd layers attend over the full context (the
//     transformers Gemma2Config layer_types default);
// (b) query_pre_attn_scalar (e.g. 224 on 27B; HF default 256) replaces
//     head_dim in the attention score scaling - folded into W_q at load;
// (c) attention-logit soft-capping (attn_logit_softcapping, typically
//     50.0): pre-softmax scores squashed to cap*tanh(s/cap) inside every
//     SDPA head (the TNNetScaledDotProductAttention ScoreSoftCap hook);
// (d) final-logit soft-capping (final_logit_softcapping, typically 30.0):
//     a plain TNNetSoftCapping on the LM-head logits;
// (e) sandwich norms - 4 RMSNorms per block: input_layernorm /
//     post_attention_layernorm around attention and
//     pre_feedforward_layernorm / post_feedforward_layernorm around the
//     FFN, with each post-norm INSIDE its residual branch.
// Thin wrappers over the Llama path that ASSERT model_type is 'gemma2'.
function BuildGemma2FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildGemma2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

// Gemma 3 TEXT-ONLY (model_type "gemma3_text", architectures
// ["Gemma3ForCausalLM"], e.g. google/gemma-3-1b-it; the 4B+ multimodal
// vision tower is OUT OF SCOPE): the Gemma-2 skeleton (GeGLU MLP,
// zero-centered RMSNorm, sqrt(d) embedding scale, tied head, sandwich
// norms, query_pre_attn_scalar) with three deltas:
// (a) the attention/final logit soft-caps are REPLACED by a per-head
//     learnable-scale RMSNorm on q and k - applied AFTER the projection and
//     BEFORE RoPE, the HF modeling_gemma3 ordering (q_norm(q_proj(x)) then
//     apply_rotary_pos_emb - the SAME placement as Qwen3); the shared
//     [head_dim] q_norm/k_norm gains are zero-centered (1+w) like every
//     Gemma norm. Because an RMSNorm erases any scale folded into W_q, the
//     query_pre_attn_scalar fold moves into the q_norm GAINS here;
// (b) a 5:1 local:global layer ratio - layer_types: every 6th layer
//     ((i+1) mod sliding_window_pattern = 0, HF default pattern 6) attends
//     globally, the rest carry the sliding window (default 4096);
// (c) PER-LAYER-TYPE RoPE theta - sliding layers use rope_local_base_freq
//     (default 10000), global layers use rope_theta (default 1e6).
// Soft-capping configs are still honored if non-null (Gemma3TextConfig
// keeps the fields, defaulting to None). Thin wrappers over the Llama path
// that ASSERT model_type is 'gemma3_text'.
function BuildGemma3FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildGemma3FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

// Phi-3 / Phi-4-mini (model_type "phi3": microsoft/Phi-3-mini-4k-instruct,
// Phi-4-mini-instruct and siblings - NOT phi-1/phi-2, which are model_type
// "phi" and architecturally different; see BuildPhiFromSafeTensors): the
// Llama skeleton (pre-norm RMSNorm, sequential residual, SwiGLU MLP, GQA,
// model.norm + lm_head - tied on Phi-4-mini, untied on Phi-3-mini) with
// three load-time deltas:
// (a) FUSED bias-free projections - self_attn.qkv_proj.weight packs the
//     q|k|v rows (q = num_heads*head_dim, then k and v =
//     num_key_value_heads*head_dim each) and mlp.gate_up_proj.weight packs
//     gate|up; both are sliced row-block-wise into the same Llama layers,
//     with the rotate_half->interleaved q/k permutation applied AFTER
//     slicing;
// (b) PARTIAL rotary (partial_rotary_factor, 0.75 on Phi-4-mini; 1.0 -
//     full - on Phi-3-mini): RoPE rotates only the first
//     int(head_dim*factor) channels of each q/k head, the tail passes
//     through (the Phi/NeoX rotate_half slice layout);
// (c) Mistral-style sliding window (sliding_window, EVERY layer; HF Phi3
//     routes it through the same mask util as Mistral, so the landed SDPA
//     window convention applies unchanged; null/absent = full attention).
// The 128k "longrope" rope_scaling variants are REJECTED with a clear
// error (not silently mis-loaded). Thin wrappers over the Llama path that
// ASSERT model_type is 'phi3'.
function BuildPhi3FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildPhi3FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

// Imports a HF Mixtral checkpoint (model_type "mixtral", e.g.
// mistralai/Mixtral-8x7B-v0.1): a stock Mistral/Llama decoder (full
// MHA/GQA, RoPE, optional sliding window) whose every FFN is replaced by
// block_sparse_moe = a router gate linear (num_local_experts logits) + N
// independent SwiGLU experts (experts.{i}.w1/w2/w3), softmax over ALL
// experts then top-k (num_experts_per_tok, default 2) routing with the
// top-k subset RENORMALIZED (HF normalize_topk_prob). No shared expert, no
// MLA - distinct from BuildDeepSeekV2FromSafeTensors. Runs on the shared
// Llama path (TLlamaConfig.IsMoE).
function BuildMixtralFromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildMixtralFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

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
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildGPTNeoFromSafeTensorsEx(const FileName: string;
  out Config: TGPTNeoConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildGPTNeoFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

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
    RopeScaling: TRoPEScalingConfig; // parsed "rope_scaling" (Mode=rsmNone
                                // when null/absent)
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
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildGPTNeoXFromSafeTensorsEx(const FileName: string;
  out Config: TGPTNeoXConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildGPTNeoXFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

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
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildGPTJFromSafeTensorsEx(const FileName: string;
  out Config: TGPTJConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildGPTJFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

type
  TCohereConfig = record
    HiddenSize: integer;        // d_model (hidden_size)
    IntermediateSize: integer;  // SwiGLU MLP width (intermediate_size)
    NumLayers: integer;         // decoder blocks (num_hidden_layers)
    NumHeads: integer;          // query heads (num_attention_heads)
    NumKVHeads: integer;        // K/V heads (num_key_value_heads; GQA/MQA)
    VocabSize: integer;         // vocab_size
    MaxPositions: integer;      // max_position_embeddings (context length)
    HeadDim: integer;           // per-head width; 0 = hidden_size/num_heads
    LayerNormEps: TNeuralFloat; // layer_norm_eps
    RopeTheta: TNeuralFloat;    // rope_theta (RoPE base)
    LogitScale: TNeuralFloat;   // logit_scale (final logits *= it; folded
                                // into the tied LM-head rows at load)
    TieWordEmbeddings: boolean; // tie_word_embeddings (Cohere: TRUE)
    UseQKNorm: boolean;         // per-head q/k LayerNorm pre-RoPE (cohere
                                // only; CohereLayerNorm = mean-subtracting)
    ModelType: string;          // 'cohere' or 'cohere2'
    SlidingWindow: integer;     // cohere2 sliding-window width (0 = cohere)
    SlidingWindowPattern: integer; // cohere2: every Nth layer global, the
                                // rest local AND RoPE only on local layers
                                // (global = NoPE); 0 = cohere (full + RoPE)
    Prefix: string;             // tensor-name prefix ('model.' or '')
  end;

// Reads a HF Cohere config.json (model_type 'cohere' or 'cohere2').
// Required: hidden_size, intermediate_size, num_hidden_layers,
// num_attention_heads, vocab_size, max_position_embeddings. Defaults:
// num_key_value_heads = num_attention_heads, layer_norm_eps = 1e-5,
// rope_theta = 10000, logit_scale = 0.0625, tie_word_embeddings = true,
// use_qk_norm = false. cohere2 adds sliding_window (4096) and
// sliding_window_pattern (4); attention_bias=true and any non-silu
// hidden_act are rejected. See the COHERE IMPORT section.
function ReadCohereConfigFromJSONFile(const FileName: string): TCohereConfig;

function CohereConfigToString(const Config: TCohereConfig): string;

// Builds the Cohere Command-R / Aya (model_type 'cohere') or Command-R7B
// ('cohere2') decoder and loads every weight from FileName. The net takes a
// (SeqLen,1,1) volume of token ids and outputs (SeqLen,1,vocab) logits
// already scaled by logit_scale (folded into the tied LM head). See the
// COHERE IMPORT section for the architecture mapping. pSeqLen/pInferenceOnly/
// pQuantizeInt8 behave as on the Llama path.
function BuildCohereFromSafeTensorsWithConfig(const FileName: string;
  var Config: TCohereConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildCohereFromSafeTensorsEx(const FileName: string;
  out Config: TCohereConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildCohereFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

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
    RopeScaling: TRoPEScalingConfig; // parsed "rope_scaling" (Mode=rsmNone
                                // when null/absent)
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
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildPhiFromSafeTensorsEx(const FileName: string;
  out Config: TPhiConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildPhiFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

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
  pSeqClsHead: boolean = false; pQuantizeInt8: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildBertFromSafeTensorsEx(const FileName: string;
  out Config: TBertConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pIncludePooler: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildBertFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pIncludePooler: boolean = false; pQuantizeInt8: boolean = false): TNNet;

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
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildBertForSequenceClassificationFromSafeTensors(
  const FileName: string; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

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
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildGPT2ForSequenceClassificationFromSafeTensors(
  const FileName: string; pSeqLen: integer = 0; pNumHeads: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

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
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false);

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
procedure BuildT5FromSafeTensors(const FileName: string;
  out EncoderNet, DecoderNet: TNNet; out Config: TT5Config;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false);

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

// ---------------------------------------------------------------------------
// MARIAN / OPUS-MT IMPORT (model_type "marian") - the Helsinki-NLP/opus-mt-*
// translation family (1000+ published language pairs, ~77M params each, the
// de-facto open machine-translation checkpoints). The SECOND encoder-decoder
// import, sharing T5's two-net convention but architecturally distinct:
//   - POST-norm blocks: LayerNorm AFTER each residual add
//     (self_attn_layer_norm / encoder_attn_layer_norm / final_layer_norm
//     per block) using plain BIASED nn.LayerNorm (TNNetTokenLayerNorm,
//     eps 1e-5), and NO final stack norm - vs T5's pre-RMSNorm + final norm;
//   - STATIC sinusoidal positions in HF Marian's HALF-SPLIT layout:
//     table[pos, c] = sin(pos / 10000^(2c/d)) for c < d/2 and the matching
//     cos in column d/2 + c. This is NOT the interleaved Vaswani layout of
//     TNNetSinusoidalPositionalEmbedding (even/odd columns), so the
//     importer fills a TNNetLearnedPositionalEmbedding table with the
//     exact half-split formula instead. Real checkpoints do NOT store the
//     tables (HF _keys_to_ignore_on_save) - they are regenerated here;
//     legacy exports that still carry them are accepted and ignored;
//   - swish/SiLU FFN: fc2(silu(fc1(x))) with EVERY linear biased (the
//     q/k/v/out projections too - the opposite of T5's bias-free stack);
//   - STANDARD 1/sqrt(head_dim) attention scaling (exactly what
//     TNNetScaledDotProductAttention / TNNetCrossAttention compute, so no
//     q-weight compensation is needed);
//   - scale_embedding (true in every published opus-mt config): token
//     embeddings are multiplied by sqrt(d_model) BEFORE the positions are
//     added - folded into the copied embedding rows, while the TIED
//     lm_head keeps the UNSCALED shared matrix;
//   - the lm_head is TIED to the shared source/target embedding matrix
//     PLUS Marian's final_logits_bias vector ([1, vocab_size] buffer),
//     folded into the head's per-neuron bias;
//   - decoder_start_token_id = pad_token_id (decoder rows start with it).
// BuildMarianFromSafeTensors returns the same pair as the T5 importer: the
// ENCODER ((EncSeqLen,1,1) token ids -> (EncSeqLen,1,d_model) final hidden
// states) and the two-input DECODER ((DecSeqLen,1,1) token ids +
// (EncSeqLen,1,d_model) encoder-states second TNNetInput ->
// (DecSeqLen,1,vocab) logits). T5EncoderStatesInput and RunT5 are GENERIC
// over this two-net convention - use them to run a Marian pair as well.
// BuildFromPretrained does NOT dispatch "marian" (it returns ONE net); it
// raises an error pointing here instead. TOKENIZER NOTE: opus-mt pairs ship
// source.spm/target.spm SentencePiece UNIGRAM files without tokenizer.json,
// which neuralhftokenizer.pas does not cover yet - end-to-end text
// translation needs the Unigram tokenizer task; the importer itself is
// tokenizer-independent (token ids in, logits out).

type
  TMarianConfig = record
    DModel: integer;            // d_model (hidden width; must be EVEN)
    EncoderLayers: integer;     // encoder_layers
    DecoderLayers: integer;     // decoder_layers
    EncoderHeads: integer;      // encoder_attention_heads
    DecoderHeads: integer;      // decoder_attention_heads
    EncoderFFNDim: integer;     // encoder_ffn_dim
    DecoderFFNDim: integer;     // decoder_ffn_dim
    VocabSize: integer;         // vocab_size
    MaxPositionEmbeddings: integer; // max_position_embeddings
    PadTokenId: integer;        // pad_token_id
    DecoderStartTokenId: integer; // decoder_start_token_id (= pad in Marian)
    ScaleEmbedding: boolean;    // scale_embedding (sqrt(d_model) on embeds)
    SwishFFN: boolean;          // activation_function swish/silu vs relu
    ModelType: string;          // 'marian'
  end;

// Reads a HF Marian config.json (model_type "marian"). Required: d_model,
// encoder_layers, decoder_layers, encoder_attention_heads,
// decoder_attention_heads, encoder_ffn_dim, decoder_ffn_dim, vocab_size.
// Defaults follow MarianConfig: max_position_embeddings = 1024,
// pad_token_id = 58100, decoder_start_token_id = pad_token_id,
// scale_embedding = false (every published opus-mt config pins true),
// activation_function = "swish" (only "swish"/"silu" and "relu" are
// supported). share_encoder_decoder_embeddings = false and
// tie_word_embeddings = false are REJECTED (the untied variants need a
// second embedding matrix this v1 does not wire).
function ReadMarianConfigFromJSONFile(const FileName: string): TMarianConfig;

function MarianConfigToString(const Config: TMarianConfig): string;

// Builds the Marian ENCODER and DECODER nets described by Config and loads
// every weight from the safetensors checkpoint at FileName (see the MARIAN
// IMPORT section above for shapes and the two-input decoder convention).
// EncSeqLen/DecSeqLen fix the two sequence lengths at build time and must
// not exceed max_position_embeddings (the static sinusoidal tables are
// generated for exactly these lengths). Both nets are owned by the caller.
// pInferenceOnly = True frees training volumes during construction
// (Compute()-only afterwards). Run the pair with RunT5.
procedure BuildMarianFromSafeTensorsWithConfig(const FileName: string;
  var Config: TMarianConfig; out EncoderNet, DecoderNet: TNNet;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false);

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
procedure BuildMarianFromSafeTensors(const FileName: string;
  out EncoderNet, DecoderNet: TNNet; out Config: TMarianConfig;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false);

// ---------------------------------------------------------------------------
// BART IMPORT (model_type "bart": the facebook/bart-large-cnn and
// sshleifer/distilbart-cnn-* abstractive-summarization checkpoints;
// architectures ["BartForConditionalGeneration"]) - the dominant pretrained
// encoder-decoder for SUMMARIZATION and the SECOND learned-position text
// encoder-decoder after Marian (same two-net + RunT5 convention). Lewis et
// al. 2019, arXiv:1910.13461. BART = a bidirectional BERT-style encoder + a
// GPT-2-style causal decoder with cross-attention. Differences from the
// Marian importer (which shares the same POST-norm block skeleton):
//   - LEARNED ABSOLUTE position embeddings (embed_positions, an
//     (max_position_embeddings + 2, d_model) table) with BART's +2 padding
//     offset: token position p reads table row p + 2 (rows 0/1 are the
//     padding-idx slots and are never used). The importer copies rows
//     2..EncSeqLen+1 / 2..DecSeqLen+1 into a TNNetLearnedPositionalEmbedding;
//   - a layernorm_embedding LayerNorm applied AFTER token+position
//     embeddings in BOTH stacks (Marian has none);
//   - EXACT-erf GELU FFN (HF activation_function "gelu"), composed from the
//     same Phi side-branch + ReGLU as the BERT importer;
//   - scale_embedding defaults FALSE (no sqrt(d_model) on the embeddings);
//   - decoder_start_token_id defaults to eos_token_id (BART's shift), not
//     pad as in Marian.
// Everything else matches Marian: shared source/target embedding TIED to the
// lm_head plus a final_logits_bias row, all q/k/v/out/fc Linears biased,
// standard 1/sqrt(head_dim) attention, post-residual biased LayerNorms (eps
// 1e-5), no final stack norm. tie_word_embeddings=false is REJECTED (the
// untied head needs a second matrix this v1 does not wire). The GPT-2
// byte-level BPE tokenizer is already covered by TNeuralHFTokenizer, so
// end-to-end summarization works (see examples/Summarize). BuildFromPretrained
// does NOT dispatch "bart" (it returns ONE net); it raises an error pointing
// here instead.

type
  TBartConfig = record
    DModel: integer;            // d_model (hidden width)
    EncoderLayers: integer;     // encoder_layers
    DecoderLayers: integer;     // decoder_layers
    EncoderHeads: integer;      // encoder_attention_heads
    DecoderHeads: integer;      // decoder_attention_heads
    EncoderFFNDim: integer;     // encoder_ffn_dim
    DecoderFFNDim: integer;     // decoder_ffn_dim
    VocabSize: integer;         // vocab_size
    MaxPositionEmbeddings: integer; // max_position_embeddings
    PadTokenId: integer;        // pad_token_id (default 1)
    BosTokenId: integer;        // bos_token_id (default 0)
    EosTokenId: integer;        // eos_token_id (default 2)
    DecoderStartTokenId: integer; // decoder_start_token_id (= eos in BART)
    ScaleEmbedding: boolean;    // scale_embedding (sqrt(d_model); default off)
    ModelType: string;          // 'bart'
  end;

// Reads a HF BART config.json (model_type "bart"). Required: d_model,
// encoder_layers, decoder_layers, encoder_attention_heads,
// decoder_attention_heads, encoder_ffn_dim, decoder_ffn_dim, vocab_size.
// Defaults follow BartConfig: max_position_embeddings = 1024, pad_token_id
// = 1, bos_token_id = 0, eos_token_id = 2, decoder_start_token_id =
// eos_token_id, scale_embedding = false. activation_function must be "gelu"
// (the exact erf form; "gelu_new"/"relu"/"swish" are rejected to avoid a
// silent activation mismatch - every published BART pins "gelu").
// tie_word_embeddings = false is REJECTED.
function ReadBartConfigFromJSONFile(const FileName: string): TBartConfig;

function BartConfigToString(const Config: TBartConfig): string;

// Builds the BART ENCODER and DECODER nets described by Config and loads
// every weight from the safetensors checkpoint at FileName (see the BART
// IMPORT section above). EncSeqLen/DecSeqLen fix the two sequence lengths at
// build time and must not exceed max_position_embeddings. Both nets are
// owned by the caller. Run the pair with RunT5 (the two-net convention is
// shared with the T5/Marian importers); the DECODER's second TNNetInput
// holds the encoder hidden states (T5EncoderStatesInput).
procedure BuildBartFromSafeTensorsWithConfig(const FileName: string;
  var Config: TBartConfig; out EncoderNet, DecoderNet: TNNet;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false);

// Same, but takes the already-read Config by value (the ...Ex naming the
// single-net importers use); kept for symmetry with the dispatch helpers.
procedure BuildBartFromSafeTensorsEx(const FileName: string;
  Config: TBartConfig; out EncoderNet, DecoderNet: TNNet;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false);

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
procedure BuildBartFromSafeTensors(const FileName: string;
  out EncoderNet, DecoderNet: TNNet; out Config: TBartConfig;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false);

// ---------------------------------------------------------------------------
// WHISPER IMPORT (model_type "whisper": the openai/whisper-* speech-to-text
// checkpoints; architectures ["WhisperForConditionalGeneration"]) - the
// FIRST SPEECH importer and the THIRD encoder-decoder import after T5 and
// Marian (same two-net + RunT5 convention). Radford et al. 2022,
// arXiv:2212.04356. Differences from the text encoder-decoders:
//   - the ENCODER input is NOT token ids but a LOG-MEL SPECTROGRAM volume
//     (2*max_source_positions frames, 1, num_mel_bins) - produced by
//     ComputeWhisperLogMel / WhisperLogMelFromWavFile in
//     neural/neuralaudio.pas (the exact HF WhisperFeatureExtractor recipe:
//     400-pt periodic-hann STFT, hop 160, slaney mel bank, log10,
//     global max-8 clamp, (x+4)/4); 30 s of 16 kHz audio = 3000 frames;
//   - conv frontend instead of an embedding: Conv1d(num_mel_bins ->
//     d_model, k=3, s=1, p=1) + EXACT erf GELU, then Conv1d(d_model ->
//     d_model, k=3, s=2, p=1) + GELU - the stride-2 conv HALVES the frames
//     to max_source_positions (1500 for the published checkpoints), the
//     encoder grid the decoder cross-attends to. Each conv is built as
//     TNNetPadXY(1,0) + TNNetConvolutionLinear (kernel Y clamps to the
//     sequence's SizeY=1, so the (3,1) kernel + X-only pad reproduce
//     nn.Conv1d's SAME padding exactly);
//   - FIXED sinusoidal encoder positions in the WHISPER layout - concat
//     sin|cos halves with timescale exponent c/(d/2 - 1), NOT the
//     interleaved Vaswani layout and NOT Marian's 2c/d exponent - added
//     AFTER the conv frontend. Real checkpoints SAVE the table
//     (model.encoder.embed_positions.weight) and it is loaded when
//     present; otherwise it is regenerated from the formula. Decoder
//     positions are LEARNED (model.decoder.embed_positions.weight) like
//     GPT-2's wpe;
//   - PRE-norm blocks (biased nn.LayerNorm eps 1e-5 BEFORE each sublayer,
//     residual adds the raw stream) with a FINAL stack LayerNorm in BOTH
//     stacks - vs Marian's post-norm;
//   - attention: q/v/out projections biased, k_proj BIAS-FREE, standard
//     1/sqrt(head_dim) scaling; decoder self-attention causal;
//     cross-attention reads the encoder states through TNNetCrossAttention
//     (rectangular DecSeqLen x max_source_positions scores);
//   - EXACT erf GELU FFNs (activation_function "gelu" in every published
//     Whisper config) - composed from existing layers like the BERT path;
//   - tied LM head (proj_out = decoder.embed_tokens, bias-free; the
//     proj_out.weight tensor is absent from real checkpoints).
// Tensor names: model.encoder.{conv1,conv2}.{weight,bias},
// model.{encoder,decoder}.embed_positions.weight (encoder's optional),
// model.decoder.embed_tokens.weight, model.{encoder,decoder}.layers.N.
// {self_attn,encoder_attn (decoder only)}.{q,k,v}_proj/out_proj,
// ...{self_attn,encoder_attn,final}_layer_norm.{weight,bias},
// ...{fc1,fc2}.{weight,bias}, model.{encoder,decoder}.layer_norm.
// BuildFromPretrained does NOT dispatch "whisper" (it returns ONE net); it
// raises an error pointing here instead. Decoding starts from the prologue
// <|startoftranscript|>[<|lang|>]<|transcribe|><|notimestamps|> (token ids
// from generation_config.json / the GPT-2-style byte-level BPE
// tokenizer.json, which neuralhftokenizer.pas reads); see
// examples/WhisperTranscribe.

type
  TWhisperConfig = record
    DModel: integer;            // d_model (must be EVEN and >= 4)
    EncoderLayers: integer;     // encoder_layers
    DecoderLayers: integer;     // decoder_layers
    EncoderHeads: integer;      // encoder_attention_heads
    DecoderHeads: integer;      // decoder_attention_heads
    EncoderFFNDim: integer;     // encoder_ffn_dim
    DecoderFFNDim: integer;     // decoder_ffn_dim
    VocabSize: integer;         // vocab_size (51865 in whisper-tiny)
    NumMelBins: integer;        // num_mel_bins (80; the encoder input depth)
    MaxSourcePositions: integer; // max_source_positions (1500): the FIXED
                                // encoder length; mel input = 2x frames
    MaxTargetPositions: integer; // max_target_positions (448)
    PadTokenId: integer;        // pad_token_id
    EosTokenId: integer;        // eos_token_id (<|endoftext|>)
    DecoderStartTokenId: integer; // decoder_start_token_id
                                // (<|startoftranscript|>)
    ScaleEmbedding: boolean;    // scale_embedding (false in every release)
    ModelType: string;          // 'whisper'
  end;

// Reads a HF Whisper config.json (model_type "whisper"). Required: d_model,
// encoder_layers, decoder_layers, encoder_attention_heads,
// decoder_attention_heads, encoder_ffn_dim, decoder_ffn_dim, vocab_size,
// num_mel_bins, max_source_positions, max_target_positions. Defaults follow
// WhisperConfig: pad 50256, eos 50256, decoder_start 50257,
// scale_embedding false. activation_function must be "gelu" (the exact erf
// form; every published Whisper pins it).
function ReadWhisperConfigFromJSONFile(const FileName: string): TWhisperConfig;

function WhisperConfigToString(const Config: TWhisperConfig): string;

// Builds the Whisper ENCODER ((2*max_source_positions,1,num_mel_bins)
// log-mel volume -> (max_source_positions,1,d_model) final hidden states)
// and the two-input DECODER ((DecSeqLen,1,1) token ids +
// (max_source_positions,1,d_model) encoder-states second TNNetInput ->
// (DecSeqLen,1,vocab) logits) and loads every weight from the checkpoint
// at FileName (.safetensors, sharded index or pytorch_model.bin). The
// encoder length is FIXED by the config (like HF, which rejects any other
// mel length); only DecSeqLen is chosen at build time (1..
// max_target_positions). Both nets are owned by the caller. pInferenceOnly
// = True frees training volumes during construction. Run the pair with
// RunT5 (the generic two-net runner).
procedure BuildWhisperFromSafeTensorsWithConfig(const FileName: string;
  var Config: TWhisperConfig; out EncoderNet, DecoderNet: TNNet;
  DecSeqLen: integer; pInferenceOnly: boolean = false);

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
procedure BuildWhisperFromSafeTensors(const FileName: string;
  out EncoderNet, DecoderNet: TNNet; out Config: TWhisperConfig;
  DecSeqLen: integer; pInferenceOnly: boolean = false;
  const ConfigFileName: string = '');

// ---------------------------------------------------------------------------
// RWKV-4 IMPORT (model_type "rwkv": RWKV/rwkv-4-169m-pile and siblings,
// architectures ["RwkvForCausalLM"]) - the FIRST NON-TRANSFORMER importer:
// a recurrent WKV mixer (Peng et al. 2023, arXiv:2305.13048) that decodes
// with CONSTANT memory and no KV cache. Differences from the decoder
// families above:
//   - NO positional embedding and NO attention: each block is a pre-LN
//     TIME-MIX residual (token-shift lerped k/v/r streams -> bias-free
//     key/value/receptance projections -> the TNNetWKV exponential-decay
//     numerator/denominator recurrence with per-channel decay w and
//     current-token bonus u -> sigmoid(receptance) gate -> output proj)
//     plus a pre-LN CHANNEL-MIX residual (token-shift lerped k/r streams,
//     key = SQUARED ReLU of a D->intermediate proj, out =
//     sigmoid(receptance proj) (*) value proj back to D);
//   - token shift: per-channel lerp x*mix + shift(x)*(1-mix) with SEPARATE
//     learned time_mix_{key,value,receptance} vectors per stream
//     (TNNetTokenShift, previous token zero-padded at t=0);
//   - DECAY CONVENTION: the checkpoint stores a RAW time_decay vector
//     applied as a per-step decay factor exp(-exp(time_decay)); TNNetWKV
//     parameterizes the same factor as exp(-softplus(w_raw)), so the
//     import sets w_raw = invsoftplus(exp(time_decay)) - EXACT for every
//     real value of time_decay because exp() is positive and softplus is a
//     bijection onto the positives (w_raw = x for x > 30 matches the
//     layer's own softplus shortcut bit-for-bit). time_first maps to the
//     bonus u unchanged;
//   - plain BIASED LayerNorm (TNNetTokenLayerNorm, NOT RMSNorm): ln1/ln2
//     per block, an EXTRA pre_ln ("ln0") after the embedding in block 0
//     only, and a final ln_out before the head;
//   - every Linear is bias-free; the head is UNTIED by default
//     (tie_word_embeddings false in the published configs);
//   - config rescale_every (inference-time halving of the hidden stream
//     every N blocks with matching weight division, a float16 trick) is
//     read but IGNORED: LayerNorm scale-invariance makes it a mathematical
//     identity, so loading the raw weights is exact.
// Tensor names: rwkv.embeddings.weight, rwkv.blocks.N.{pre_ln (N=0 only),
// ln1,ln2}.{weight,bias}, rwkv.blocks.N.attention.{time_decay,time_first,
// time_mix_key,time_mix_value,time_mix_receptance,key,value,receptance,
// output}, rwkv.blocks.N.feed_forward.{time_mix_key,time_mix_receptance,
// key,receptance,value}, rwkv.ln_out.{weight,bias}, head.weight.
// BuildFromPretrained dispatches model_type "rwkv" here. pSeqLen <= 0
// defaults to the config's context_length (RWKV itself has no positional
// limit; the net is built at a fixed sequence length like every importer).

type
  TRWKVConfig = record
    HiddenSize: integer;          // hidden_size (d_model)
    NumLayers: integer;           // num_hidden_layers
    VocabSize: integer;           // vocab_size
    AttentionHiddenSize: integer; // attention_hidden_size (= hidden_size)
    IntermediateSize: integer;    // intermediate_size (default 4*hidden)
    ContextLength: integer;       // context_length (default 1024)
    LayerNormEps: double;         // layer_norm_epsilon (default 1e-5)
    RescaleEvery: integer;        // rescale_every (read but a no-op; see above)
    TieWordEmbeddings: boolean;   // tie_word_embeddings (default false)
    ModelType: string;            // 'rwkv'
    Prefix: string;               // 'rwkv.' or '' (detected from the file)
  end;

// Reads a HF RWKV config.json (model_type "rwkv"). Required: hidden_size,
// num_hidden_layers, vocab_size. Defaults follow RwkvConfig:
// attention_hidden_size = hidden_size, intermediate_size = 4*hidden_size
// (null in the published configs), context_length = 1024,
// layer_norm_epsilon = 1e-5, rescale_every = 6, tie_word_embeddings false.
function ReadRWKVConfigFromJSONFile(const FileName: string): TRWKVConfig;

function RWKVConfigToString(const Config: TRWKVConfig): string;

// Builds the RWKV-4 causal-LM net described by Config ((SeqLen,1,1) token
// ids in, (SeqLen,1,vocab) logits out, like the decoder families) and loads
// every weight from the checkpoint at FileName (see the RWKV-4 IMPORT
// section above). pSeqLen <= 0 uses Config.ContextLength. pInferenceOnly =
// True frees training volumes during construction (Compute()-only).
function BuildRWKVFromSafeTensorsWithConfig(const FileName: string;
  var Config: TRWKVConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildRWKVFromSafeTensorsEx(const FileName: string;
  out Config: TRWKVConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildRWKVFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

// ============================ MAMBA IMPORT =================================
// Mamba (model_type "mamba": state-spaces/mamba-130m-hf and siblings,
// architectures ["MambaForCausalLM"]) - the SECOND NON-TRANSFORMER importer
// after RWKV-4: a selective-SSM (S6, Gu & Dao 2023) mixer that decodes with
// CONSTANT memory (a (d_inner, d_state) recurrent state per block, no KV
// cache). Rebuilt from existing CAI layers:
//
//   Input(SeqLen) -> TNNetEmbedding (backbone.embeddings)
//     -> num_hidden_layers x [ x := x + Mixer(TokenRMSNorm(x)) ]
//     -> TNNetTokenRMSNorm (backbone.norm_f)
//     -> TNNetPointwiseConvLinear(vocab) LM head (tied to the embedding
//        rows when lm_head.weight is absent - the published checkpoints)
//
// Per mixer (HF MambaMixer slow path, d_inner = expand * hidden_size):
//   x|z   = split(in_proj(h))                  [2*d_inner; x first]
//   x     = SiLU(causal depthwise conv1d_k(x)) [k = conv_kernel, with bias]
//   y     = SelectiveScan(x)                   [TNNetSelectiveSSM(d_state)]
//   out   = out_proj( y (*) SiLU(z) )
// The scan leaf owns its delta/B/C projections, so the checkpoint maps:
//   - W_d (Neurons[0]) <- dt_proj.weight @ x_proj.weight[0:dt_rank] - the
//     LOW-RANK delta path folded EXACTLY into one d_inner x d_inner matrix
//     (double accumulation; mathematically identical, dt_rank =
//     time_step_rank, "auto" = ceil(hidden_size/16));
//   - W_B/W_C (Neurons[1]/[2]) <- x_proj.weight rows [dt_rank | d_state |
//     d_state] (B_t/C_t in R^d_state SHARED across channels);
//   - b_d (Neurons[3]) <- dt_proj.bias (delta = softplus(W_d.x + b_d));
//   - A_raw (Neurons[4]) <- A_log RAW: HF discretizes abar = exp(delta*A)
//     with A = -exp(A_log) and the layer computes a = exp(-delta*
//     exp(A_raw)) - the SAME formula, no convention bridge needed (the
//     fixture generator asserts this bit-tight);
//   - e (Neurons[5]) <- D (the y += D*x skip).
// Tensor names: backbone.embeddings.weight, backbone.layers.N.norm.weight,
// backbone.layers.N.mixer.{A_log, D, conv1d.weight, conv1d.bias,
// in_proj.weight, x_proj.weight, dt_proj.weight, dt_proj.bias,
// out_proj.weight}, backbone.norm_f.weight (+ in_proj.bias/out_proj.bias
// when use_bias, lm_head.weight when untied). BuildFromPretrained
// dispatches model_type "mamba" here. Mamba has NO positional limit (and
// no context_length config field): pSeqLen <= 0 defaults to 1024.

type
  TMambaConfig = record
    HiddenSize: integer;          // hidden_size (d_model)
    NumLayers: integer;           // num_hidden_layers
    StateSize: integer;           // state_size (d_state, default 16)
    TimeStepRank: integer;        // time_step_rank ("auto" = ceil(hidden/16))
    VocabSize: integer;           // vocab_size
    Expand: integer;              // expand (default 2)
    DInner: integer;              // intermediate_size (default expand*hidden)
    ConvKernel: integer;          // conv_kernel (default 4)
    UseConvBias: boolean;         // use_conv_bias (default true)
    UseBias: boolean;             // use_bias - in/out_proj (default false)
    LayerNormEps: double;         // layer_norm_epsilon (default 1e-5)
    TieWordEmbeddings: boolean;   // tie_word_embeddings (default true)
    ModelType: string;            // 'mamba'
    Prefix: string;               // 'backbone.' or '' (detected from the file)
  end;

// Reads a HF Mamba config.json (model_type "mamba"). Required: hidden_size,
// num_hidden_layers, vocab_size. Defaults follow MambaConfig: state_size =
// 16, time_step_rank = "auto" = ceil(hidden_size/16), expand = 2,
// intermediate_size = expand*hidden_size, conv_kernel = 4, use_conv_bias
// true, use_bias false, layer_norm_epsilon = 1e-5, tie_word_embeddings true.
function ReadMambaConfigFromJSONFile(const FileName: string): TMambaConfig;

function MambaConfigToString(const Config: TMambaConfig): string;

// Builds the Mamba causal-LM net described by Config ((SeqLen,1,1) token
// ids in, (SeqLen,1,vocab) logits out, like the decoder families) and loads
// every weight from the checkpoint at FileName (see the MAMBA IMPORT
// section above). pSeqLen <= 0 uses 1024 (Mamba has no positional limit).
// pInferenceOnly = True frees training volumes during construction.
function BuildMambaFromSafeTensorsWithConfig(const FileName: string;
  var Config: TMambaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildMambaFromSafeTensorsEx(const FileName: string;
  out Config: TMambaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildMambaFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

// ============================ BLOOM IMPORT =================================
// BuildBloomFromSafeTensors rebuilds BigScience's BLOOM decoder (model_type
// "bloom": bigscience/bloom-560m .. bloom-176B and the instruction-tuned
// bloomz family) - the FIRST ALiBi-positional importer (GPT-2/BERT use
// learned positions, the Llama/NeoX/Qwen/Gemma families RoPE, T5 a bucketed
// relative bias): BLOOM has NO positional embeddings AT ALL; position
// enters ONLY as a fixed per-head linear bias slope_h*(j-i) on the
// pre-softmax attention logits (TNNetALiBiAttention, slopes from the exact
// HF build_alibi_tensor geometric recipe - see TNNetALiBiAttention.
// ALiBiSlope). Architecture per checkpoint:
//
//   Input(SeqLen) -> TNNetEmbedding (word_embeddings)
//     -> TNNetTokenLayerNorm (word_embeddings_layernorm - BLOOM
//        normalises the embeddings BEFORE the first block, a Megatron
//        bf16-stability artifact no other family here carries)
//     -> n_layer x [ x := x + Dense(MHA_ALiBi(LN_1(x)));
//                    x := x + MLP(LN_2(x)) ]      (sequential pre-LN)
//     -> TNNetTokenLayerNorm (ln_f)
//     -> TNNetPointwiseConvLinear(vocab) LM head, ALWAYS tied to the
//        word_embeddings rows (BLOOM ships no separate lm_head tensor)
//
// The attention projection is ONE fused query_key_value nn.Linear
// [3*hidden, hidden] WITH bias. Its row layout interleaves PER HEAD: head h
// occupies rows h*3*head_dim..(h+1)*3*head_dim-1, the first head_dim being
// q, then k, then v (HF view(.., heads, 3, head_dim) then index the middle
// axis). NOTE the contrast with the siblings: GPT-2's c_attn is q|k|v WHOLE
// thirds (no per-head interleave); GPT-NeoX uses the SAME h-major per-head
// [q|k|v] interleave as BLOOM (HF view(.., heads, 3*head_dim) then thirds -
// identical bytes, different idiom) but additionally needs the rotate_half
// RoPE row permutation composed into the de-interleave. BLOOM has no rotary
// at all, so its rows load STRAIGHT - the de-interleave into the Q|K|V slab
// is the whole job. attention.dense and both MLP linears (mlp.dense_h_to_4h
// / dense_4h_to_h, intermediate = 4*hidden) are plain nn.Linear [out, in]
// WITH biases. Attention is standard scaled (1/sqrt(head_dim)) + the ALiBi
// bias; the MLP activation is BLOOM's Megatron GELU
// x*0.5*(1+tanh(0.79788456*x*(1+0.044715*x^2))) = the tanh GELU
// (TNNetGELU) exactly.
// config.json uses the LEGACY GPT-2-style names n_head / n_layer and
// hidden_size (older exports spell it n_embed); layer_norm_epsilon;
// intermediate is n_inner (null = 4*hidden in every released checkpoint).
// There is NO max_position_embeddings (ALiBi extrapolates): pSeqLen <= 0
// uses the config's seq_length when present, else 2048.
// apply_residual_connection_post_layernorm=true (a pretraining-only
// Megatron variant no released HF checkpoint uses) is REJECTED explicitly
// rather than silently mis-wired.
// The byte-level BPE tokenizer (250880-entry multilingual vocab, 46
// languages) is the existing TNeuralHFTokenizer tokenizer.json path - no
// new tokenizer machinery.

type
  TBloomConfig = record
    HiddenSize: integer;          // hidden_size (legacy alias n_embed)
    NumLayers: integer;           // n_layer
    NumHeads: integer;            // n_head
    IntermediateSize: integer;    // n_inner (null/absent = 4*hidden)
    VocabSize: integer;           // vocab_size
    SeqLength: integer;           // seq_length (informational; 0 if absent)
    LayerNormEps: double;         // layer_norm_epsilon
    ModelType: string;            // 'bloom'
    Prefix: string;               // 'transformer.' or '' (detected)
  end;

// Reads a HF BLOOM config.json (model_type "bloom"). Required: n_head,
// n_layer, vocab_size and hidden_size (or its legacy alias n_embed).
// Defaults follow BloomConfig: n_inner null = 4*hidden_size,
// layer_norm_epsilon = 1e-5. apply_residual_connection_post_layernorm=true
// is rejected (see the BLOOM IMPORT section above).
function ReadBloomConfigFromJSONFile(const FileName: string): TBloomConfig;

function BloomConfigToString(const Config: TBloomConfig): string;

// Builds the BLOOM causal-LM net described by Config ((SeqLen,1,1) token
// ids in, (SeqLen,1,vocab) logits out, like the other decoder families) and
// loads every weight from the checkpoint at FileName (see the BLOOM IMPORT
// section above). pSeqLen <= 0 uses Config.SeqLength (2048 when the config
// does not carry seq_length; ALiBi imposes no hard positional limit).
// pInferenceOnly = True frees training volumes during construction.
function BuildBloomFromSafeTensorsWithConfig(const FileName: string;
  var Config: TBloomConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildBloomFromSafeTensorsEx(const FileName: string;
  out Config: TBloomConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildBloomFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

// ============================= FALCON IMPORT ===============================
// BuildFalconFromSafeTensors rebuilds the Falcon decoder family (model_type
// "falcon"; legacy "RefinedWebModel" = falcon-7b, "RefinedWeb" = falcon-40b)
// - the closest cousin of the GPT-NeoX path (fused query_key_value, RoPE,
// parallel attention+MLP residual) with two Falcon-specific twists:
//   - FUSED MULTI-QUERY / GQA query_key_value: a single [qkv_out, hidden]
//     slab. For multi_query=true & new_decoder_architecture=false
//     (falcon-7b, falcon-rw): num_heads query heads + ONE shared K head +
//     ONE shared V head (view(num_heads+2, head_dim)); the single cached
//     K/V fans out across every query head. For new_decoder_architecture=
//     true (falcon-40b): num_kv_heads GQA groups, each
//     (num_heads/num_kv_heads query heads + 1 K + 1 V) INTERLEAVED per group
//     (view(-1, num_heads//num_kv_heads + 2, head_dim)). multi_query=false &
//     new_decoder_architecture=false is plain per-head [q|k|v] (the GPT-NeoX
//     layout). RoPE is FULL-head Llama rotate_half (the loader composes the
//     rotate_half->interleaved permutation into Q and K rows).
//   - PARALLEL attention+MLP residual x := x + Attn(ln(x)) + MLP(ln(x)) with
//     a SINGLE shared LayerNorm (parallel_attn / num_ln_in_parallel_attn=1,
//     falcon-7b: input_layernorm) OR TWO separate norms (new arch /
//     num_ln_in_parallel_attn=2, falcon-40b: ln_attn for attention,
//     ln_mlp for the MLP). parallel_attn=false is the rare SEQUENTIAL pre-LN
//     fallback (input_layernorm + post_attention_layernorm).
// Plain BIASED LayerNorm (NOT RMSNorm), GELU MLP, NO biases on any Linear
// (bias=false). Causal-LM contract like the other decoder families
// ((SeqLen,1,1) ids in, (SeqLen,1,vocab) logits out).
type
  TFalconConfig = record
    HiddenSize: integer;        // hidden_size (d_model)
    IntermediateSize: integer;  // ffn_hidden_size (null/absent = 4*hidden)
    NumLayers: integer;         // num_hidden_layers (legacy n_layer)
    NumHeads: integer;          // num_attention_heads (legacy n_head)
    NumKVHeads: integer;        // EFFECTIVE K/V heads: num_kv_heads (new
                                // arch), 1 (multi_query), or NumHeads (MHA)
    VocabSize: integer;         // vocab_size
    MaxPositions: integer;      // max_position_embeddings
    LayerNormEps: TNeuralFloat; // layer_norm_epsilon
    RopeTheta: TNeuralFloat;    // rope_theta (RoPE base, default 10000)
    NewDecoderArchitecture: boolean; // new_decoder_architecture (falcon-40b)
    ParallelAttn: boolean;      // parallel_attn (single LN, falcon-7b)
    TwoLayerNorms: boolean;     // ln_attn/ln_mlp split (new arch /
                                // num_ln_in_parallel_attn=2)
    TieWordEmbeddings: boolean; // tie_word_embeddings
    RopeScaling: TRoPEScalingConfig; // parsed "rope_scaling"/"rope_parameters"
    Prefix: string;             // tensor-name prefix ('transformer.' or '')
  end;

// Reads a HF Falcon config.json (model_type "falcon"; the legacy
// "RefinedWebModel"/"RefinedWeb" spellings are accepted). Required: n_head /
// num_attention_heads, n_layer / num_hidden_layers, vocab_size, hidden_size.
// Defaults follow FalconConfig: ffn_hidden_size null = 4*hidden_size,
// num_kv_heads = num_attention_heads, new_decoder_architecture = false,
// multi_query = true, parallel_attn = true, layer_norm_epsilon = 1e-5,
// rope_theta = 10000, max_position_embeddings = 2048, tie_word_embeddings =
// true. alibi=true is rejected (use the BLOOM-style path, not wired for
// Falcon). bias=true is rejected (no released Falcon uses biased Linears).
// Prefix is left '' - the builder detects it.
function ReadFalconConfigFromJSONFile(const FileName: string): TFalconConfig;

function FalconConfigToString(const Config: TFalconConfig): string;

// Builds the Falcon causal-LM net described by Config and loads every weight
// from the checkpoint at FileName (see the FALCON IMPORT section above).
// pSeqLen <= 0 uses max_position_embeddings. pInferenceOnly = True frees
// training volumes during construction. Config.Prefix is detected from the
// checkpoint and written back.
function BuildFalconFromSafeTensorsWithConfig(const FileName: string;
  var Config: TFalconConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildFalconFromSafeTensorsEx(const FileName: string;
  out Config: TFalconConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildFalconFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

// ========================= MODERNBERT IMPORT ===============================
// BuildModernBertFromSafeTensors rebuilds answer.ai's ModernBERT encoder
// (model_type "modernbert": answerdotai/ModernBERT-base / -large) - the
// SECOND encoder family here (after BERT/DistilBERT/RoBERTa) and the first
// with RoPE: ModernBERT has NO learned position table at all; position
// enters ONLY through rotary embeddings on q/k, applied with BIDIRECTIONAL
// (non-causal) attention. Architecture (verified against HF transformers
// modeling_modernbert):
//
//   Input(SeqLen,1,1 token ids) -> TNNetEmbedding (embeddings.tok_embeddings)
//     -> TNNetTokenLayerNorm (embeddings.norm; norm_bias=false -> beta=0)
//     -> num_hidden_layers x PRE-LN blocks
//          x := x + Wo(MHA_RoPE(attn_norm(x)))   [layer 0: attn_norm is
//                                                 nn.Identity - SKIPPED]
//          x := x + mlp.Wo(GeGLU(mlp.Wi(mlp_norm(x))))
//     -> TNNetTokenLayerNorm (final_norm)
//   Output: (SeqLen,1,hidden_size) final hidden states (HF ModernBertModel
//   last_hidden_state), like BuildBertFromSafeTensors - NO LM/pooler head.
//
// ALTERNATING LOCAL/GLOBAL attention: layer i attends GLOBALLY iff
// i mod global_attn_every_n_layers = 0 (HF: "sliding_attention" iff
// bool(i % n); default n = 3 - layers 0, 3, 6, ... are global). Local
// layers carry a BIDIRECTIONAL SYMMETRIC sliding window: query i attends
// keys j with |i - j| <= local_attention div 2 (config local_attention =
// 128 is the TOTAL window; HF masks abs(q_idx - kv_idx) > sliding_window
// with sliding_window = local_attention // 2) - the new
// pBidirectionalWindow mode of TNNetScaledDotProductAttention (W =
// local_attention div 2 + 1, the |i-j| < W convention).
// PER-LAYER-TYPE ROPE THETA (the Gemma-3 delta on a bidirectional
// encoder): global layers rotate with global_rope_theta (default 160000),
// local layers with local_rope_theta (default 10000).
// The fused attn.Wqkv nn.Linear [3*hidden, hidden] packs q|k|v as WHOLE
// thirds (HF view(.., 3, heads, head_dim): NO per-head interleave - the
// GPT-2 layout, NOT the BLOOM/NeoX one); the q and k thirds additionally
// need the rotate_half row permutation (TNNetRotaryEmbedding rotates
// interleaved pairs; HF rotates half-split lanes - same trick as the Llama
// family). The GeGLU MLP packs mlp.Wi [2*intermediate, hidden] as
// input|gate (HF chunk(2): FIRST half is the ACTIVATED input, second half
// the linear gate - out = Wo(act(input) * gate)); the CAI TNNetGEGLU*
// layers compute FIRSTHALF * act(SECONDHALF), so the loader swaps the
// halves (gate rows -> neurons 0..I-1, input rows -> I..2I-1).
// hidden_activation "gelu" (the ModernBERT default) is the EXACT erf GELU
// - TNNetGEGLUErf; "gelu_pytorch_tanh" selects the tanh TNNetGEGLU.
// attention_bias / mlp_bias / norm_bias (all default FALSE) are honored:
// true loads the corresponding .bias tensors, false zeroes biases/betas.
// The classifier/MLM heads are OUT OF SCOPE (head-bearing checkpoints are
// rejected by the unexpected-tensor check) - the base hidden states feed
// the same downstream heads as the BERT importer.

type
  TModernBertConfig = record
    HiddenSize: integer;       // hidden_size
    IntermediateSize: integer; // intermediate_size (Wi is 2x this wide)
    NumLayers: integer;        // num_hidden_layers
    NumHeads: integer;         // num_attention_heads
    VocabSize: integer;        // vocab_size
    MaxPositions: integer;     // max_position_embeddings (RoPE soft limit)
    NormEps: double;           // norm_eps (HF default 1e-5)
    LocalAttention: integer;   // local_attention TOTAL window (default 128)
    GlobalEveryN: integer;     // global_attn_every_n_layers (default 3)
    GlobalRopeTheta: double;   // global_rope_theta (default 160000)
    LocalRopeTheta: double;    // local_rope_theta (default 10000)
    HiddenActTanh: boolean;    // true = gelu_pytorch_tanh; false = exact
                               // erf "gelu" (the ModernBERT default)
    AttentionBias: boolean;    // attention_bias (Wqkv + attn Wo biases)
    MlpBias: boolean;          // mlp_bias (Wi + mlp Wo biases)
    NormBias: boolean;         // norm_bias (LayerNorm betas)
    ModelType: string;         // 'modernbert'
    Prefix: string;            // 'model.' or '' (detected by the builder)
  end;

// Reads a HF ModernBERT config.json (model_type "modernbert"). Required:
// hidden_size, intermediate_size, num_hidden_layers, num_attention_heads,
// vocab_size, max_position_embeddings. Defaults follow ModernBertConfig:
// norm_eps = 1e-5, local_attention = 128, global_attn_every_n_layers = 3,
// global_rope_theta = 160000, local_rope_theta = 10000, hidden_activation
// = "gelu" (EXACT erf; "gelu_pytorch_tanh" selects the tanh form, anything
// else is rejected), attention_bias / mlp_bias / norm_bias = false. An
// explicit "layer_types" array is verified against the
// global_attn_every_n_layers pattern (a mismatch is rejected rather than
// silently mis-wired). Prefix is left '' - the builder detects it.
function ReadModernBertConfigFromJSONFile(
  const FileName: string): TModernBertConfig;

function ModernBertConfigToString(const Config: TModernBertConfig): string;

// Builds the ModernBERT ENCODER described by Config ((SeqLen,1,1) token ids
// in, (SeqLen,1,hidden_size) final hidden states out) and loads every
// weight from the checkpoint at FileName (see the MODERNBERT IMPORT section
// above). pSeqLen = 0 uses the full max_position_embeddings context (8192
// on the released checkpoints - pass an explicit pSeqLen for CPU work).
// pInferenceOnly = True frees training volumes during construction.
function BuildModernBertFromSafeTensorsWithConfig(const FileName: string;
  var Config: TModernBertConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
function BuildModernBertFromSafeTensorsEx(const FileName: string;
  out Config: TModernBertConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

function BuildModernBertFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

// ---------------------------------------------------------------------------
// DEEPSEEK-V2 IMPORT (model_type "deepseek_v2"; DeepSeek-V2-Lite is the
// reference checkpoint) - the first importer to carry the two DeepSeek-V2
// signature blocks: Multi-head Latent Attention (MLA: a low-rank compressed
// KV latent c_KV plus a DECOUPLED RoPE key slice shared by all heads; the
// AddMultiHeadLatentAttention wiring, extended with the checkpoint's
// kv_a_layernorm RMSNorm on the latent) and DeepSeekMoE (shared + routed
// experts with softmax top-k gating; the AddDeepSeekMoE wiring with SwiGLU
// expert MLPs and TNNetTopKGate, raw weights when norm_topk_prob=false).
//
// Per layer: x := x + o_proj(MLA(RMSNorm(x))); x := x + MLP(RMSNorm(x)).
// MLA from primitives (all token-wise 1x1 convs, mirroring the
// AddMultiHeadLatentAttention builder; dn := qk_nope_head_dim,
// dr := qk_rope_head_dim, dv := v_head_dim, r := kv_lora_rank):
//   q       := q_proj(x), per head packed [q_nope(dn) | q_rope(dr)] - split
//              at LOAD time into a content projection (Heads*dn) and a
//              per-head rope projection (Heads*dr);
//   kv_a_proj_with_mqa(x) packs [c_KV(r) | k_rope(dr)] - split at load into
//              the latent down-projection (r) and ONE shared rope-K
//              projection (dr);
//   c_KV    := RMSNorm_{kv_a_layernorm}(c_KV)  - the latent norm;
//   kv_b_proj(c_KV) packs per head [k_nope(dn) | v(dv)] - split at load
//              into K (Heads*dn) and V (Heads*dv) up-projections;
//   per head: SDPA over [q_nope|RoPE(q_rope_h)] . [k_nope|RoPE(k_rope)]
//              with scale 1/sqrt(dn+dr); the rope-K slice is rotated ONCE
//              and shared by every head; V is zero-padded by dr channels
//              (the exact x+(-x) block of the MLA builder) and the head
//              output sliced back to its dv content channels.
// RoPE convention note: DeepSeek's checkpoints store the rope slices in
// INTERLEAVED (even/odd) pair order - the HF remote code de-interleaves
// before rotate_half; transformers' builtin port multiplies consecutive
// pairs as complex numbers - which is exactly TNNetRotaryEmbedding's
// layout, so unlike the Llama path the rope rows load with NO permutation.
// MLP: layers with index < first_k_dense_replace (layer 0 in -Lite) keep a
// DENSE SwiGLU MLP of intermediate_size; the rest are DeepSeekMoE blocks:
//   gate logits -> per-token softmax -> TNNetTopKGate(num_experts_per_tok,
//   pRenormalize=norm_topk_prob); per routed expert a SwiGLU MLP of
//   moe_intermediate_size combined with its gate weight; ONE fused shared-
//   expert SwiGLU MLP (width n_shared_experts*moe_intermediate_size) added
//   UNGATED; routed_scaling_factor is folded into every routed expert's
//   down_proj at load (it scales only the routed sum, never the shared
//   branch - linear, so the fold is exact).
// LIMITATIONS (all checked, with clear errors): q_lora_rank must be null
// (the -Lite case; the full 236B V2 uses a low-rank W_q, not wired),
// qk_nope_head_dim must equal v_head_dim (true for -Lite AND full V2; the
// zero-pad trick reuses the dr-wide rope-K block), rope_scaling must be
// null (DeepSeek's YaRN carries mscale/mscale_all_dim attention-scale
// overrides that the rope-scaling support deliberately rejects),
// topk_method must be "greedy" (group_limited_greedy routing is a full-V2
// feature), scoring_func "softmax", moe_layer_freq 1, attention_bias and
// mlp_bias false.
// Causal-LM contract like the other decoder families: input (SeqLen,1,1)
// token ids, output (SeqLen,1,vocab_size) logits.
// ---------------------------------------------------------------------------

type
  TDeepSeekV2Config = record
    VocabSize: integer;
    HiddenSize: integer;        // d_model
    NumLayers: integer;         // num_hidden_layers
    NumHeads: integer;          // num_attention_heads
    IntermediateSize: integer;  // DENSE MLP width (first_k_dense_replace)
    MoEIntermediateSize: integer; // per routed/shared expert MLP width
    KVLoraRank: integer;        // r: latent c_KV width
    QKNopeHeadDim: integer;     // dn: per-head content q/k width
    QKRopeHeadDim: integer;     // dr: decoupled rope width (shared k_rope)
    VHeadDim: integer;          // dv: per-head value width (= dn, checked)
    NSharedExperts: integer;    // 0 = no shared-expert branch
    NRoutedExperts: integer;
    NumExpertsPerTok: integer;  // top-k over the routed experts
    FirstKDenseReplace: integer; // layers [0..k) keep a dense MLP
    NormTopKProb: boolean;      // false in -Lite: RAW survivor weights
    RoutedScalingFactor: TNeuralFloat; // folded into routed down_proj
    MaxPositions: integer;      // max_position_embeddings
    RmsNormEps: TNeuralFloat;
    RopeTheta: TNeuralFloat;
    TieWordEmbeddings: boolean;
    ModelType: string;          // 'deepseek_v2'
    Prefix: string;             // detected: 'model.' or ''
  end;

// Reads a HF DeepSeek-V2 config.json (model_type "deepseek_v2"). Required:
// hidden_size, intermediate_size, moe_intermediate_size, num_hidden_layers,
// num_attention_heads, vocab_size, max_position_embeddings, kv_lora_rank,
// qk_nope_head_dim, qk_rope_head_dim, v_head_dim, n_routed_experts,
// num_experts_per_tok. Optional: n_shared_experts (null/absent = 0),
// first_k_dense_replace (default 0), norm_topk_prob (default false),
// routed_scaling_factor (default 1.0), rms_norm_eps (1e-6), rope_theta
// (10000), tie_word_embeddings (false). Fails with a clear message on the
// unsupported variants listed in the section header (q_lora_rank,
// rope_scaling, group_limited_greedy, ...).
function ReadDeepSeekV2ConfigFromJSONFile(
  const FileName: string): TDeepSeekV2Config;
// One-line human-readable summary (mirrors LlamaConfigToString).
function DeepSeekV2ConfigToString(const Config: TDeepSeekV2Config): string;

// Core DeepSeek-V2 builder. FileName accepts .safetensors,
// .safetensors.index.json (sharded), pytorch_model.bin or
// pytorch_model.bin.index.json (CreatePretrainedTensorReader dispatch, like
// every other family). pSeqLen <= 0 uses max_position_embeddings.
// Config.Prefix is detected from the checkpoint (var parameter).
function BuildDeepSeekV2FromSafeTensorsWithConfig(const FileName: string;
  var Config: TDeepSeekV2Config; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
// Reads ConfigFileName (default: config.json next to FileName), then builds.
function BuildDeepSeekV2FromSafeTensorsEx(const FileName: string;
  out Config: TDeepSeekV2Config; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
function BuildDeepSeekV2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;

// ---------------------------------------------------------------------------
// CLIP IMPORT (model_type "clip": openai/clip-vit-base-patch32/-patch16,
// -large-patch14 and the laion CLIP family) - the FIRST VISION-LANGUAGE
// importer and the FIRST ViT (vision transformer) in this unit. Radford et
// al. 2021, arXiv:2103.00020. CLIP is a contrastive DUAL-ENCODER: two
// independent towers projected into one shared embedding space, so the
// import returns TWO nets (the T5/Marian two-net convention - but here the
// nets are independent peers, not an encoder feeding a decoder; there is
// no cross-attention and RunT5 does not apply):
//   - the TEXT net: (SeqLen,1,1) token ids in -> (SeqLen,1,projection_dim)
//     PROJECTED per-token features out. Architecture: token embedding +
//     LEARNED absolute positions (GPT-2-style wpe), num_hidden_layers x
//     CAUSAL pre-LN blocks [x += out(MHA(q|k|v(LN1(x)))); x +=
//     fc2(quick_gelu(fc1(LN2(x))))] with biased q/k/v/out projections and
//     standard 1/sqrt(head_dim) scaling, final_layer_norm, then the
//     bias-free text_projection applied PER TOKEN. HF's text_embeds is the
//     ROW AT THE EOT POSITION (ClipTextEosPosition + ClipExtractEmbedding
//     below);
//   - the VISION net: (image_size,image_size,num_channels) pixel values in
//     (CLIP-preprocessed floats: resized/cropped, scaled to [0,1], then
//     channel-normalized with the published CLIP mean/std - preprocessing
//     itself is out of scope) -> (num_patches+1,1,projection_dim)
//     PROJECTED per-token features out. Architecture: bias-FREE
//     patch_embedding conv (TNNetConvolutionLinear, kernel = stride =
//     patch_size) -> flatten to a (num_patches,1,hidden) row-major patch
//     sequence -> the learned class_embedding PREPENDED as token 0 ->
//     learned absolute positions -> pre_layrnorm (the historical HF
//     spelling; "pre_layernorm" is accepted too) -> BIDIRECTIONAL pre-LN
//     blocks (same shape as the text blocks) -> post_layernorm ->
//     bias-free visual_projection, both applied PER TOKEN. HF applies
//     post_layernorm + projection only to the pooled CLASS token; per-token
//     application is exact for ROW 0, which is HF's image_embeds
//     (ClipExtractEmbedding(Output, 0, ...)).
// Import notes:
//   - quick_gelu (x * sigmoid(1.702 x), HF QuickGELUActivation) is built
//     as TNNetSwishLearnable(beta = 1.702) - y = x*sigmoid(beta*x) exactly;
//     hidden_act "gelu" (exact erf, the laion/SigLIP-adjacent configs) and
//     "gelu_new"/"gelu_pytorch_tanh" are also accepted;
//   - the CLASS token is wired WITHOUT a dedicated layer: the patch
//     sequence is left-padded with one ZERO row (TNNetPadXY + TNNetCrop)
//     and class_embedding is FOLDED into row 0 of the learned position
//     table (exact: the class slot's pre-position content is identically
//     zero, so row0 = class_embedding + position_row0);
//   - EOS pooling follows modeling_clip exactly: eos_token_id == 2 (the
//     legacy pre-#24773 id every published OpenAI CLIP config carries)
//     pools at ARGMAX(input_ids) - the first occurrence of the HIGHEST
//     token id, the eot token in the CLIP vocab - while any other
//     eos_token_id pools at the FIRST position equal to it
//     (ClipTextEosPosition implements both branches);
//   - logit_scale (a 0-dim learned scalar) is loaded into
//     Config.LogitScale; HF's logits_per_image = exp(logit_scale) *
//     cosine(image, text) - see ClipSimilarity;
//   - the VISION tower is factored as the reusable BuildClipVisionTower
//     (a plain pre-LN ViT-with-class-token: a future ViT/DINO/SigLIP
//     import can reuse it with different prefixes/projection).
// Tensor names: text_model.embeddings.{token,position}_embedding.weight,
// text_model.encoder.layers.N.{layer_norm1,layer_norm2}.{weight,bias},
// ...self_attn.{q,k,v,out}_proj.{weight,bias}, ...mlp.{fc1,fc2}.{weight,
// bias}, text_model.final_layer_norm.{weight,bias}, text_projection.weight,
// vision_model.embeddings.{class_embedding,patch_embedding.weight,
// position_embedding.weight}, vision_model.{pre_layrnorm,post_layernorm}.
// {weight,bias}, vision_model.encoder.layers.N.* (text names),
// visual_projection.weight, logit_scale.
// BuildFromPretrained does NOT dispatch "clip" (it returns ONE net); it
// raises an error pointing here instead. TOKENIZER NOTE: CLIP ships a
// byte-level BPE tokenizer.json (lowercasing + </w> word markers) readable
// by neuralhftokenizer.pas; the importer itself is tokenizer-independent.
// See examples/ClipZeroShot for the zero-shot classification recipe.

type
  // hidden_act of one CLIP tower: quick_gelu (x*sigmoid(1.702x), every
  // published OpenAI CLIP), the exact erf gelu, or the tanh approximation.
  TClipHiddenAct = (chaQuickGelu, chaGeluExact, chaGeluTanh);

  // One CLIP transformer tower (the text/vision halves share this shape).
  TClipTowerConfig = record
    HiddenSize: integer;        // hidden_size
    IntermediateSize: integer;  // intermediate_size (fc1 width)
    NumLayers: integer;         // num_hidden_layers
    NumHeads: integer;          // num_attention_heads
    LayerNormEps: TNeuralFloat; // layer_norm_eps (1e-5 in published CLIPs)
    HiddenAct: TClipHiddenAct;  // hidden_act
  end;

  TClipConfig = record
    Text: TClipTowerConfig;     // text_config
    Vision: TClipTowerConfig;   // vision_config
    TextVocabSize: integer;     // text_config.vocab_size (49408)
    TextMaxPositions: integer;  // text_config.max_position_embeddings (77)
    EosTokenId: integer;        // text_config.eos_token_id (2 = legacy
                                // ARGMAX pooling; see ClipTextEosPosition)
    ImageSize: integer;         // vision_config.image_size (224)
    PatchSize: integer;         // vision_config.patch_size (32/16/14)
    NumChannels: integer;       // vision_config.num_channels (3)
    ProjectionDim: integer;     // projection_dim (the shared space, 512)
    LogitScale: TNeuralFloat;   // logit_scale_init_value from the config,
                                // REPLACED by the checkpoint's learned
                                // logit_scale tensor when present (RAW,
                                // i.e. pre-exp)
    ModelType: string;          // 'clip'
  end;

// Reads a HF CLIP config.json (model_type "clip"). Required per tower
// (text_config / vision_config sub-objects): hidden_size,
// intermediate_size, num_hidden_layers, num_attention_heads, plus
// vocab_size + max_position_embeddings (text) and image_size + patch_size
// (vision). Defaults follow CLIPTextConfig/CLIPVisionConfig: hidden_act =
// "quick_gelu", layer_norm_eps = 1e-5, eos_token_id = 2, num_channels = 3,
// projection_dim = 512, logit_scale_init_value = 2.6592. hidden_act other
// than quick_gelu / gelu / gelu_new / gelu_pytorch_tanh is rejected.
function ReadClipConfigFromJSONFile(const FileName: string): TClipConfig;

function ClipConfigToString(const Config: TClipConfig): string;

// Builds the CLIP TEXT and VISION nets described by Config and loads every
// weight from the checkpoint at FileName (.safetensors / sharded index /
// pytorch_model.bin via CreatePretrainedTensorReader; see the CLIP IMPORT
// section above for shapes). TextSeqLen <= 0 uses the full
// text max_position_embeddings context. Both nets are owned by the caller.
// pInferenceOnly = True frees training volumes during construction
// (Compute()-only afterwards). Config.LogitScale is updated from the
// checkpoint's logit_scale tensor.
procedure BuildClipFromSafeTensorsWithConfig(const FileName: string;
  var Config: TClipConfig; out TextNet, VisionNet: TNNet;
  TextSeqLen: integer = 0; pInferenceOnly: boolean = false);

// Same, reading the config from ConfigFileName ('' = "config.json" in the
// directory of FileName) and returning it in Config.
procedure BuildClipFromSafeTensors(const FileName: string;
  out TextNet, VisionNet: TNNet; out Config: TClipConfig;
  TextSeqLen: integer = 0; pInferenceOnly: boolean = false;
  const ConfigFileName: string = '');

// Builds + loads the CLIP ViT VISION tower alone (the reusable half: a
// plain pre-LN ViT with a class token). Prefix is the tensor-name prefix
// ('vision_model.' for CLIP checkpoints); ProjectionTensorName names the
// bias-free [ProjectionDim, hidden] projection applied per token after the
// post LayerNorm ('visual_projection.weight' for CLIP; '' skips the
// projection and the net outputs (num_patches+1,1,hidden) post-LN hidden
// states - the plain ViT contract). The CLASS-token row is row 0.
function BuildClipVisionTower(Reader: TNNetSafeTensorsReader;
  const Tower: TClipTowerConfig; ImageSize, PatchSize, NumChannels,
  ProjectionDim: integer; const Prefix: string;
  const ProjectionTensorName: string = '';
  pInferenceOnly: boolean = false): TNNet;

// The text-pooling position of modeling_clip: EosTokenId = 2 (the legacy
// id of every published OpenAI CLIP) returns the position of the FIRST
// occurrence of the HIGHEST id in TokenIds (HF input_ids.argmax(-1) - the
// eot token, which carries the top id in the CLIP vocab); any other
// EosTokenId returns the first position whose id equals it (fails loudly
// when absent). TokenIds is the (SeqLen,1,1) text-net input volume.
function ClipTextEosPosition(TokenIds: TNNetVolume;
  EosTokenId: integer): integer;

// Copies the depth column at TokenPos of a (SeqLen,1,Depth) net output
// into Embedding ((1,1,Depth)) and L2-normalizes it - the CLIP embedding
// head (mirrors BertPoolSentenceEmbedding's normalize-for-cosine
// convention). TEXT net: TokenPos = ClipTextEosPosition(...); VISION net:
// TokenPos = 0 (the class token). After normalization cosine similarity is
// a plain dot product (ClipSimilarity); HF's logits_per_image =
// exp(Config.LogitScale) * ClipSimilarity(ImageEmb, TextEmb).
procedure ClipExtractEmbedding(NetOutput: TNNetVolume; TokenPos: integer;
  Embedding: TNNetVolume);

// Dot product of two same-size embedding volumes = cosine similarity of
// ClipExtractEmbedding outputs (both are unit-L2).
function ClipSimilarity(EmbA, EmbB: TNNetVolume): TNeuralFloat;

// AutoModel-style dispatch: reads config.json's model_type and routes to
// the right Build*FromSafeTensors builder. Path may be
//   - a checkpoint DIRECTORY (uses Path/config.json and
//     Path/model.safetensors.index.json if present, else
//     Path/model.safetensors), or
//   - a .safetensors / .safetensors.index.json FILE (config.json is read
//     from the same directory unless ConfigFileName overrides it).
// Supported model_types: gpt2 (n_head read from the config when present),
// gpt_neo, gpt_neox, gptj, phi, llama, mistral, qwen2, qwen3, gemma,
// gemma2, gemma3_text, rwkv, mamba, bloom, falcon (legacy RefinedWebModel /
// RefinedWeb spellings accepted), bert, distilbert, roberta,
// modernbert, deepseek_v2.
// Anything else
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
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;

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

// ----------------------------- HF PEFT LoRA -------------------------------
// Loads a Hugging Face PEFT LoRA adapter (an adapter_model.safetensors plus
// an optional adapter_config.json) onto the A=down / B=up pointwise
// projections built by TNNet.AddLoRAAdapter. PEFT stores, per wrapped module
// <m>, two nn.Linear weight matrices:
//   <prefix><m>.lora_A.weight : [r,     d_in ]   (down, A)
//   <prefix><m>.lora_B.weight : [d_out, r    ]   (up,   B)
// where <prefix> is usually "base_model.model." (and the matrices often sit
// under a ".default." adapter-name segment). The effective bypass applied at
// inference is (lora_alpha / r) * B*A; PEFT's own scaling lives in the config,
// NOT in the stored weights, which exactly matches AddLoRAAdapter's
// MulByConstant(Alpha/Rank). Both [out, in] matrices follow nn.Linear layout,
// so they map row-for-row onto the pointwise neurons (neuron j, weight i).
type
  // alpha (lora_alpha) and r read from an adapter_config.json. RankFound is
  // false when no config was read (caller keeps the rank it built with).
  TPEFTAdapterConfig = record
    Alpha: TNeuralFloat;
    Rank: integer;
    RankFound: boolean;
  end;

// Loads a bias-free nn.Linear [OutDim, InDim] matrix straight onto a pointwise
// layer's neurons (neuron j gets row j, weight i = W[j*InDim+i]), each weight
// multiplied by Scale. The pointwise/per-token layout the AddLoRAAdapter A/B
// projections use, so it is reused to drop the base proj and the lora_A/lora_B
// matrices onto their layers. Raises EPretrainedImportError on a shape or
// neuron-count mismatch.
procedure LoadPlainLinearWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName: string; InDim, OutDim: integer;
  Scale: TNeuralFloat = 1.0);

// Reads lora_alpha and r from a PEFT adapter_config.json. Missing fields
// default to Alpha=Rank (scale 1.0) so a config-less adapter still loads.
function ReadPEFTAdapterConfig(const FileName: string): TPEFTAdapterConfig;

// Loads ONE wrapped module's LoRA pair from an already-open reader onto the
// AddLoRAAdapter A (down) and B (up) layers. ModuleName is the PEFT module
// path (e.g. "model.layers.0.self_attn.q_proj"); the loader tries the common
// "base_model.model." prefix and ".default."/"" adapter-name segments until
// it finds "<...>.lora_A.weight" and "<...>.lora_B.weight". Scale is the
// effective alpha/r multiplier folded into B at load time (so the built
// MulByConstant(1.0) net still applies the right PEFT scaling). Returns the
// resolved tensor base name (the "<...>" before ".lora_A.weight").
function LoadPEFTLoRAModule(Reader: TNNetSafeTensorsReader;
  ADown, BUp: TNNetLayer; const ModuleName: string;
  Scale: TNeuralFloat = 1.0): string;

implementation

procedure ImportError(const Msg: string);
begin
  raise EPretrainedImportError.Create(Msg);
end;

function CreatePretrainedTensorReader(
  const FileName: string): TNNetSafeTensorsReader;
const
  BinIndexSuffix = '.bin.index.json';
var
  LowerName: string;
begin
  LowerName := LowerCase(FileName);
  if (ExtractFileExt(LowerName) = '.bin') or
     (Copy(LowerName, Length(LowerName) - Length(BinIndexSuffix) + 1,
       Length(BinIndexSuffix)) = BinIndexSuffix) then
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
// With pQuantizeInt8 the importers quantize layers DURING construction (to
// bound peak RAM); a loader about to write checkpoint weights into such a
// layer must first restore writable FP32 neuron volumes. The importer
// re-quantizes (TNNet.QuantizeWeightsInt8 sweep) after the layer/block is
// loaded. Requantizing an untouched dequantized row is EXACT: the row max
// is a +-127 code, so the recomputed scale and codes are identical.
// Coded by Claude (AI).
procedure EnsureWritableImportWeights(Layer: TNNetLayer);
begin
  if (Layer is TNNetLayerConcatedWeights) and
     TNNetLayerConcatedWeights(Layer).WeightsQuantizedInt8 then
    TNNetLayerConcatedWeights(Layer).DequantizeWeightsInt8();
end;

procedure LoadConv1DWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName, BName: string; InDim, OutDim: integer);
var
  W, B: TNNetVolume;
  i, j: integer;
begin
  EnsureWritableImportWeights(Layer);
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
  pSeqClsHead: boolean = false; pExactGelu: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
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
      // pQuantizeInt8 follows the same idempotent-sweep pattern: each sweep
      // converts the freshly built block's weight matrices to int8 so peak
      // memory carries at most one block of FP32 weights; the load phase
      // below refills each layer (DequantizeWeightsInt8 inside the loaders)
      // and re-sweeps after every block.
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

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
          EnsureWritableImportWeights(LMHead);
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
      // Re-quantize the refilled LM head before streaming the blocks.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
        // Re-quantize the block just refilled with checkpoint weights.
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
          EnsureWritableImportWeights(LMHead);
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
      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

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
  pInferenceOnly: boolean = false; pExactGelu: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TGPT2Config;
begin
  Result := BuildGPT2FromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pNumHeads, pInferenceOnly, {pSeqClsHead=}false, pExactGelu,
    pQuantizeInt8);
end;

// ===================== ROPE SCALING (config.json) ==========================

function DefaultRoPEScaling(): TRoPEScalingConfig;
begin
  Result.Mode := rsmNone;
  Result.Factor := 1.0;
  Result.OriginalContextLen := 0;
  Result.YarnAlpha := 1.0;
  Result.YarnBeta := 32.0;
  SetLength(Result.LongFactors, 0);
  Result.LongAttnFactor := 1.0;
end;

function RoPEScalingToString(const S: TRoPEScalingConfig): string;
begin
  case S.Mode of
    rsmPositionInterpolation: Result := 'linear x' + FloatToStr(S.Factor);
    rsmNTKAware: Result := 'ntk x' + FloatToStr(S.Factor);
    rsmYaRN: Result := 'yarn x' + FloatToStr(S.Factor) +
      ' orig=' + IntToStr(S.OriginalContextLen) +
      ' beta_slow=' + FloatToStr(S.YarnAlpha) +
      ' beta_fast=' + FloatToStr(S.YarnBeta);
    rsmLlama3: Result := 'llama3 x' + FloatToStr(S.Factor) +
      ' orig=' + IntToStr(S.OriginalContextLen) +
      ' low=' + FloatToStr(S.YarnAlpha) +
      ' high=' + FloatToStr(S.YarnBeta);
    rsmLongRoPE: Result := 'longrope orig=' +
      IntToStr(S.OriginalContextLen) +
      ' long_factors=' + IntToStr(Length(S.LongFactors)) +
      ' attn=' + FloatToStr(S.LongAttnFactor);
    else Result := 'none';
  end;
end;

// Maps a HF config.json "rope_scaling" object (when present and non-null)
// onto TNNetRotaryEmbedding's scaling-mode constructor arguments - see
// TRoPEScalingConfig for the type mapping. Notes:
//  - "rope_type" is the modern spelling; legacy configs say "type".
//  - "dynamic" (dynamic NTK) rescales the base from the CURRENT sequence
//    length at inference in HF; this importer applies the static NTK-aware
//    base change at the full config "factor" instead.
//  - yarn: original_max_position_embeddings is read from rope_scaling, then
//    the top level (Phi-3 style), then falls back to max_position_embeddings
//    (the HF fallback chain). DeepSeek-style "mscale"/"mscale_all_dim"
//    overrides and a custom "attention_factor" (anything other than the
//    YaRN default 0.1*ln(factor)+1, which the layer hardcodes) are rejected.
//    The layer implements HF's DEFAULT truncate=true band edges; a config
//    with an explicit "truncate": false would deviate slightly on the two
//    boundary pairs (the key is ignored).
//  - "longrope" (Phi-3) and unknown types are rejected.
// Coded by Claude (AI).
function ReadRoPEScalingFromJSONObject(Obj: TJSONObject;
  MaxPositions: integer; const ImporterName: string): TRoPEScalingConfig;
var
  Field, Sub: TJSONData;
  Scaling: TJSONObject;
  TypeName: string;
  DefaultAttn: TNeuralFloat;

  function ReadOriginalContextLen(): integer;
  var
    F: TJSONData;
  begin
    F := Scaling.Find('original_max_position_embeddings');
    if (F = nil) or F.IsNull then
      F := Obj.Find('original_max_position_embeddings');
    if (F = nil) or F.IsNull then
      Result := MaxPositions
    else
      Result := F.AsInteger;
    if Result <= 0 then
      ImportError(ImporterName + ': rope_scaling ' +
        'original_max_position_embeddings must be positive, got ' +
        IntToStr(Result) + '.');
  end;

  procedure RejectKey(const KeyName: string);
  var
    F: TJSONData;
  begin
    F := Scaling.Find(KeyName);
    if (F <> nil) and not F.IsNull then
      ImportError(ImporterName + ': rope_scaling "' + KeyName + '" (' +
        F.AsJSON + ') is not wired into TNNetRotaryEmbedding (it changes ' +
        'the attention factor away from the YaRN default the layer ' +
        'hardcodes).');
  end;

  // Reads a LongRoPE per-frequency factor array (long_factor / short_factor):
  // a non-empty JSON array of positive numbers.
  function ReadFactorArray(const KeyName: string): TNeuralFloatDynArr;
  var
    F: TJSONData;
    Arr: TJSONArray;
    I: integer;
    V: TNeuralFloat;
  begin
    F := Scaling.Find(KeyName);
    if (F = nil) or F.IsNull or not (F is TJSONArray) then
      ImportError(ImporterName + ': rope_scaling longrope requires a "' +
        KeyName + '" array.');
    Arr := TJSONArray(F);
    if Arr.Count < 1 then
      ImportError(ImporterName + ': rope_scaling longrope "' + KeyName +
        '" array is empty.');
    SetLength(Result, Arr.Count);
    for I := 0 to Arr.Count - 1 do
    begin
      V := Arr.Items[I].AsFloat;
      if V <= 0 then
        ImportError(ImporterName + ': rope_scaling longrope "' + KeyName +
          '" entries must be positive, got ' + FloatToStr(V) +
          ' at index ' + IntToStr(I) + '.');
      Result[I] := V;
    end;
  end;

begin
  Result := DefaultRoPEScaling();
  Field := Obj.Find('rope_scaling');
  if (Field = nil) or Field.IsNull then Exit; // unscaled: unchanged path
  if not (Field is TJSONObject) then
    ImportError(ImporterName + ': config "rope_scaling" must be a JSON ' +
      'object or null, got ' + Field.AsJSON + '.');
  Scaling := TJSONObject(Field);
  TypeName := Scaling.Get('rope_type', Scaling.Get('type', ''));
  if TypeName = '' then
    ImportError(ImporterName + ': rope_scaling carries neither "rope_type" ' +
      'nor "type": ' + Scaling.AsJSON + '.');
  if TypeName = 'default' then Exit; // explicit no-op scaling
  // HF's Phi3Config remaps rope_scaling type "su"/"yarn" to "longrope" when a
  // per-frequency long_factor table is present; detect that FIRST (before the
  // scalar "factor" guard, which longrope does not require) so a Phi-3 "yarn"
  // (carrying long_factor, NOT beta_slow/beta_fast) is treated as LongRoPE
  // rather than mis-parsed as standard YaRN.
  if (TypeName = 'longrope') or (TypeName = 'su') or
     ((TypeName = 'yarn') and (Scaling.Find('long_factor') <> nil) and
      not Scaling.Find('long_factor').IsNull) then
  begin
    // Phi-3 LongRoPE: per-frequency long_factor table (used for the long
    // context) divides each inverse frequency, plus an attention scaling.
    Result.Mode := rsmLongRoPE;
    Result.Factor := 1.0;
    Result.OriginalContextLen := ReadOriginalContextLen();
    Result.LongFactors := ReadFactorArray('long_factor');
    // attention_factor (long_mscale): explicit value wins; otherwise HF's
    // sqrt(1 + ln(max_pos/orig)/ln(orig)) when max_pos > orig, else 1.0.
    Sub := Scaling.Find('long_mscale');
    if (Sub = nil) or Sub.IsNull then Sub := Scaling.Find('attention_factor');
    if (Sub <> nil) and not Sub.IsNull then
      Result.LongAttnFactor := Sub.AsFloat
    else if MaxPositions > Result.OriginalContextLen then
      Result.LongAttnFactor :=
        Sqrt(1.0 + Ln(MaxPositions / Result.OriginalContextLen) /
          Ln(Result.OriginalContextLen))
    else
      Result.LongAttnFactor := 1.0;
    Exit;
  end;
  Sub := Scaling.Find('factor');
  if (Sub = nil) or Sub.IsNull then
    ImportError(ImporterName + ': rope_scaling type "' + TypeName +
      '" is missing the required "factor".');
  Result.Factor := Sub.AsFloat;
  if Result.Factor < 1 then
    ImportError(ImporterName + ': rope_scaling factor must be >= 1, got ' +
      FloatToStr(Result.Factor) + '.');
  if TypeName = 'linear' then
    Result.Mode := rsmPositionInterpolation
  else if (TypeName = 'dynamic') or (TypeName = 'ntk') then
    Result.Mode := rsmNTKAware
  else if TypeName = 'yarn' then
  begin
    Result.Mode := rsmYaRN;
    Result.OriginalContextLen := ReadOriginalContextLen();
    Result.YarnAlpha := Scaling.Get('beta_slow', 1.0);
    Result.YarnBeta := Scaling.Get('beta_fast', 32.0);
    if Result.YarnBeta <= Result.YarnAlpha then
      ImportError(ImporterName + ': rope_scaling yarn requires beta_fast > ' +
        'beta_slow, got beta_fast=' + FloatToStr(Result.YarnBeta) +
        ' beta_slow=' + FloatToStr(Result.YarnAlpha) + '.');
    RejectKey('mscale');
    RejectKey('mscale_all_dim');
    Sub := Scaling.Find('attention_factor');
    if (Sub <> nil) and not Sub.IsNull then
    begin
      if Result.Factor > 1 then
        DefaultAttn := 0.1 * Ln(Result.Factor) + 1.0
      else
        DefaultAttn := 1.0;
      if Abs(Sub.AsFloat - DefaultAttn) > 1e-3 then
        ImportError(ImporterName + ': rope_scaling attention_factor=' +
          FloatToStr(Sub.AsFloat) + ' is not wired - TNNetRotaryEmbedding ' +
          'hardcodes the YaRN default 0.1*ln(factor)+1 = ' +
          FloatToStr(DefaultAttn) + '.');
    end;
  end
  else if TypeName = 'llama3' then
  begin
    Result.Mode := rsmLlama3;
    Result.OriginalContextLen := ReadOriginalContextLen();
    Result.YarnAlpha := Scaling.Get('low_freq_factor', 1.0);
    Result.YarnBeta := Scaling.Get('high_freq_factor', 4.0);
    if Result.YarnBeta <= Result.YarnAlpha then
      ImportError(ImporterName + ': rope_scaling llama3 requires ' +
        'high_freq_factor > low_freq_factor, got high=' +
        FloatToStr(Result.YarnBeta) + ' low=' +
        FloatToStr(Result.YarnAlpha) + '.');
  end
  else
    ImportError(ImporterName + ': rope_scaling type "' + TypeName +
      '" is not supported - only "linear" (Position Interpolation), ' +
      '"dynamic"/"ntk" (NTK-aware), "yarn", "llama3" and ' +
      '"longrope"/"su" (Phi-3 LongRoPE) are wired onto ' +
      'TNNetRotaryEmbedding.');
end;

// Constructs a TNNetRotaryEmbedding from a base and a parsed rope_scaling
// (Mode=rsmNone passes only the base - bit-identical to the unscaled
// constructor). Coded by Claude (AI).
function CreateRoPEFromScaling(Base: TNeuralFloat;
  const S: TRoPEScalingConfig): TNNetRotaryEmbedding;
begin
  if S.Mode = rsmLongRoPE then
    Result := TNNetRotaryEmbedding.CreateLongRoPE(Base, S.LongFactors,
      S.LongAttnFactor)
  else
    Result := TNNetRotaryEmbedding.Create(Base, S.Mode, S.Factor,
      S.OriginalContextLen, S.YarnAlpha, S.YarnBeta);
end;

// ============================ LLAMA IMPORT =================================

function ReadLlamaConfigFromJSONFile(const FileName: string): TLlamaConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, HiddenAct: string;
  HeadDimField, SlidingWindowField, FloatField: TJSONData;

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
       (ModelType <> 'gemma') and (ModelType <> 'gemma2') and
       (ModelType <> 'gemma3_text') and (ModelType <> 'phi3') and
       (ModelType <> 'olmo2') and (ModelType <> 'mixtral') then
      ImportError('Llama import: config model_type is "' + ModelType +
        '" - only "llama", "mistral", "qwen2", "qwen3", "gemma", ' +
        '"gemma2", "gemma3_text", "phi3", "olmo2" and "mixtral" are ' +
        'supported here ' +
        '(see BuildFromPretrained for the full dispatch; multimodal ' +
        '"gemma3" configs are out of scope - use a TEXT-ONLY gemma3_text ' +
        'checkpoint).');
    Result.ModelType := ModelType;
    Result.HiddenSize := RequiredInt('hidden_size');
    Result.IntermediateSize := RequiredInt('intermediate_size');
    Result.NumLayers := RequiredInt('num_hidden_layers');
    Result.NumHeads := RequiredInt('num_attention_heads');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.MaxPositions := RequiredInt('max_position_embeddings');
    Result.RopeScaling := ReadRoPEScalingFromJSONObject(Obj,
      Result.MaxPositions, 'Llama import');
    Result.NumKVHeads := Obj.Get('num_key_value_heads', Result.NumHeads);
    Result.RmsNormEps := Obj.Get('rms_norm_eps', 1.0e-6);
    Result.RopeTheta := Obj.Get('rope_theta', 10000.0);
    // Gemma ALWAYS ties the LM head to the embedding (the HF GemmaConfig /
    // Gemma2Config default is tie_word_embeddings=true); the other families
    // default off.
    Result.TieWordEmbeddings := Obj.Get('tie_word_embeddings',
      (ModelType = 'gemma') or (ModelType = 'gemma2') or
      (ModelType = 'gemma3_text'));
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
    Result.QueryPreAttnScalar := 0;
    Result.AttnLogitSoftCap := 0;
    Result.FinalLogitSoftCap := 0;
    Result.SandwichNorm := False;
    Result.AltSlidingWindow := False;
    Result.SlidingWindowPattern := 0;
    Result.RopeLocalTheta := 0;
    Result.FusedQKVGateUp := False;
    Result.PartialRotaryFactor := 1.0;
    Result.PostNormReordered := False;
    Result.QKNormFullWidth := False;
    Result.IsMoE := False;
    Result.NumLocalExperts := 0;
    Result.MoEExpertsPerTok := 0;
    if ModelType = 'mixtral' then
    begin
      // Mixtral: a stock Mistral decoder (full MHA/GQA, RoPE, sliding
      // window) whose every FFN is replaced by block_sparse_moe = a gate
      // linear (num_local_experts logits) + N independent SwiGLU experts,
      // softmax over ALL experts then top-k (num_experts_per_tok, default
      // 2) routing, output = sum of selected experts weighted by the
      // RENORMALIZED top-k softmax probs (HF normalizes the top-k subset).
      // No shared expert, no MLA - the distinction from DeepSeek-V2.
      Result.IsMoE := True;
      Result.NumLocalExperts := Obj.Get('num_local_experts', 8);
      Result.MoEExpertsPerTok := Obj.Get('num_experts_per_tok', 2);
      if Result.NumLocalExperts < 1 then
        ImportError('Llama import: Mixtral num_local_experts must be >= 1, ' +
          'got ' + IntToStr(Result.NumLocalExperts) + '.');
      if (Result.MoEExpertsPerTok < 1) or
         (Result.MoEExpertsPerTok > Result.NumLocalExperts) then
        ImportError('Llama import: Mixtral num_experts_per_tok=' +
          IntToStr(Result.MoEExpertsPerTok) + ' must be in [1, ' +
          'num_local_experts=' + IntToStr(Result.NumLocalExperts) + '].');
      HiddenAct := Obj.Get('hidden_act', 'silu');
      if HiddenAct <> 'silu' then
        ImportError('Llama import: Mixtral hidden_act "' + HiddenAct +
          '" is not supported - every released mixtral checkpoint uses ' +
          '"silu" (SwiGLU experts).');
      Result.QKVBias := Obj.Get('attention_bias', False);
      // Mixtral configs ship sliding_window (8x7B: null in newer revisions,
      // 4096 in the original); null/absent = full attention.
      SlidingWindowField := Obj.Find('sliding_window');
      if (SlidingWindowField <> nil) and not SlidingWindowField.IsNull then
      begin
        Result.SlidingWindow := SlidingWindowField.AsInteger;
        if Result.SlidingWindow < 1 then
          ImportError('Llama import: config sliding_window must be a ' +
            'positive integer or null, got ' + SlidingWindowField.AsJSON + '.');
      end;
    end
    else if ModelType = 'mistral' then
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
    else if (ModelType = 'gemma') or (ModelType = 'gemma2') or
            (ModelType = 'gemma3_text') then
    begin
      // Gemma-1 deltas (all load-time; see the BuildGemmaFromSafeTensors
      // interface comment): gated-GELU MLP, zero-centered RMSNorm and the
      // sqrt(d_model) embedding-output scale. Gemma-2 keeps all three and
      // adds its own deltas below.
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
      if (ModelType = 'gemma2') or (ModelType = 'gemma3_text') then
      begin
        // Gemma-2 deltas on top of the Gemma-1 skeleton:
        // (a) alternating local/global attention - sliding_window (HF
        //     default 4096) applies to EVEN layers only (transformers'
        //     Gemma2Config layer_types default: "sliding_attention" for
        //     bool((i+1) % 2), i.e. layers 0, 2, ...);
        // (b) query_pre_attn_scalar (HF default 256) replaces head_dim in
        //     the attention score scaling - folded into W_q at load;
        // (c) attention-logit soft-capping (attn_logit_softcapping, HF
        //     default 50.0; null = off) inside every SDPA head;
        // (d) final-logit soft-capping (final_logit_softcapping, HF
        //     default 30.0; null = off) on the LM-head logits;
        // (e) sandwich norms - post_attention_layernorm and
        //     post_feedforward_layernorm INSIDE the residual branches.
        // Gemma-3 (gemma3_text) keeps (b), (e) and the sliding window but
        // REPLACES the soft-caps (c)/(d) with the per-head q/k RMSNorm
        // (QKNorm, defaults below the shared block) and changes the
        // local/global pattern from 1:1 to 5:1 with per-layer-type RoPE
        // theta.
        Result.SandwichNorm := True;
        Result.AltSlidingWindow := (ModelType = 'gemma2');
        Result.SlidingWindow := 4096;
        SlidingWindowField := Obj.Find('sliding_window');
        if (SlidingWindowField <> nil) and not SlidingWindowField.IsNull then
        begin
          Result.SlidingWindow := SlidingWindowField.AsInteger;
          if Result.SlidingWindow < 1 then
            ImportError('Llama import: config sliding_window must be a ' +
              'positive integer or null, got ' +
              SlidingWindowField.AsJSON + '.');
        end;
        Result.QueryPreAttnScalar := 256;
        FloatField := Obj.Find('query_pre_attn_scalar');
        if (FloatField <> nil) and not FloatField.IsNull then
        begin
          Result.QueryPreAttnScalar := FloatField.AsFloat;
          if Result.QueryPreAttnScalar <= 0 then
            ImportError('Llama import: config query_pre_attn_scalar must ' +
              'be positive, got ' + FloatField.AsJSON + '.');
        end;
        // Soft-cap defaults: Gemma-2 ships 50/30; Gemma3TextConfig keeps
        // the fields but defaults BOTH to None (the QK-norm replaces them).
        // A non-null Gemma-3 value is still honored.
        if ModelType = 'gemma2' then
          Result.AttnLogitSoftCap := 50.0
        else
          Result.AttnLogitSoftCap := 0;
        FloatField := Obj.Find('attn_logit_softcapping');
        if FloatField <> nil then
        begin
          if FloatField.IsNull then Result.AttnLogitSoftCap := 0
          else Result.AttnLogitSoftCap := FloatField.AsFloat;
          if Result.AttnLogitSoftCap < 0 then
            ImportError('Llama import: config attn_logit_softcapping must ' +
              'be positive or null, got ' + FloatField.AsJSON + '.');
        end;
        if ModelType = 'gemma2' then
          Result.FinalLogitSoftCap := 30.0
        else
          Result.FinalLogitSoftCap := 0;
        FloatField := Obj.Find('final_logit_softcapping');
        if FloatField <> nil then
        begin
          if FloatField.IsNull then Result.FinalLogitSoftCap := 0
          else Result.FinalLogitSoftCap := FloatField.AsFloat;
          if Result.FinalLogitSoftCap < 0 then
            ImportError('Llama import: config final_logit_softcapping ' +
              'must be positive or null, got ' + FloatField.AsJSON + '.');
        end;
        if ModelType = 'gemma3_text' then
        begin
          // Gemma-3 deltas on top of the shared Gemma-2 machinery:
          // (a) per-head learnable-scale RMSNorm on q/k AFTER the
          //     projection and BEFORE RoPE (HF modeling_gemma3:
          //     q_norm(q_proj(x)) then apply_rotary_pos_emb - the same
          //     placement as Qwen3); the [head_dim] gains are
          //     zero-centered (1+w) like every Gemma norm;
          // (b) layer_types pattern: every Nth layer ((i+1) mod N = 0,
          //     sliding_window_pattern, HF default 6) attends GLOBALLY,
          //     the rest carry the sliding window (5:1 local:global);
          // (c) per-layer-type RoPE theta: rope_theta (default 1e6) on
          //     global layers, rope_local_base_freq (default 10000) on
          //     sliding layers.
          Result.QKNorm := True;
          Result.RopeTheta := Obj.Get('rope_theta', 1000000.0);
          Result.RopeLocalTheta := Obj.Get('rope_local_base_freq', 10000.0);
          Result.SlidingWindowPattern :=
            Obj.Get('sliding_window_pattern', 6);
          if Result.SlidingWindowPattern < 1 then
            ImportError('Llama import: config sliding_window_pattern must ' +
              'be a positive integer, got ' +
              IntToStr(Result.SlidingWindowPattern) + '.');
        end;
      end;
    end
    else if ModelType = 'phi3' then
    begin
      // Phi-3 / Phi-4-mini deltas (see BuildPhi3FromSafeTensors): fused
      // bias-free qkv_proj / gate_up_proj slabs, optional partial rotary
      // and a Mistral-style sliding window (HF Phi3 routes
      // config.sliding_window through the SAME mask util as Mistral, on
      // every layer; null/absent = full attention).
      Result.FusedQKVGateUp := True;
      Result.RmsNormEps := Obj.Get('rms_norm_eps', 1.0e-5); // Phi3 default
      HiddenAct := Obj.Get('hidden_act', 'silu');
      if HiddenAct <> 'silu' then
        ImportError('Llama import: Phi-3 hidden_act "' + HiddenAct +
          '" is not supported - every released phi3 checkpoint uses ' +
          '"silu" (SwiGLU).');
      Result.PartialRotaryFactor :=
        Obj.Get('partial_rotary_factor', 1.0);
      if (Result.PartialRotaryFactor <= 0) or
         (Result.PartialRotaryFactor > 1) then
        ImportError('Llama import: config partial_rotary_factor must be ' +
          'in (0, 1], got ' + FloatToStr(Result.PartialRotaryFactor) + '.');
      SlidingWindowField := Obj.Find('sliding_window');
      if (SlidingWindowField <> nil) and not SlidingWindowField.IsNull then
      begin
        Result.SlidingWindow := SlidingWindowField.AsInteger;
        if Result.SlidingWindow < 1 then
          ImportError('Llama import: config sliding_window must be a ' +
            'positive integer or null, got ' + SlidingWindowField.AsJSON + '.');
      end;
      // The 128k Phi-3 variants ship rope_scaling type "longrope" (older
      // configs spell it "su" or even "yarn" - HF's Phi3Config remaps all
      // three to longrope): per-frequency long/short factor tables. The
      // generic parse above (ReadRoPEScalingFromJSONObject) now maps these to
      // rsmLongRoPE (using the long_factor table and the long attention
      // scaling for the static long-context import).
    end
    else if ModelType = 'olmo2' then
    begin
      // OLMo-2 deltas (see BuildOlmo2FromSafeTensors): reordered post-norm
      // (x + Norm(Attn(x)), NO input_layernorm) and full-width q/k RMSNorm
      // before the head split + RoPE. rms_norm_eps defaults to 1e-5 (the
      // HF Olmo2Config default). attention_bias=true would put biases on
      // ALL FOUR projections including o_proj (unlike Qwen2's q/k/v-only
      // QKVBias) - no released OLMo-2 checkpoint uses it, so it is
      // rejected rather than half-wired.
      Result.PostNormReordered := True;
      Result.QKNormFullWidth := True;
      Result.RmsNormEps := Obj.Get('rms_norm_eps', 1.0e-5);
      if Obj.Get('attention_bias', False) then
        ImportError('Llama import: OLMo-2 attention_bias=true (biases on ' +
          'q/k/v AND o_proj) is not wired into this importer - every ' +
          'released OLMo-2 checkpoint is bias-free.');
      HiddenAct := Obj.Get('hidden_act', 'silu');
      if HiddenAct <> 'silu' then
        ImportError('Llama import: OLMo-2 hidden_act "' + HiddenAct +
          '" is not supported - every released olmo2 checkpoint uses ' +
          '"silu" (SwiGLU).');
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
  if Config.QueryPreAttnScalar > 0 then
    Result := Result + ', query_pre_attn_scalar=' +
      FloatToStr(Config.QueryPreAttnScalar);
  if Config.AttnLogitSoftCap > 0 then
    Result := Result + ', attn_logit_softcapping=' +
      FloatToStr(Config.AttnLogitSoftCap);
  if Config.FinalLogitSoftCap > 0 then
    Result := Result + ', final_logit_softcapping=' +
      FloatToStr(Config.FinalLogitSoftCap);
  if Config.SandwichNorm then
    Result := Result + ', sandwich_norm=true';
  if Config.AltSlidingWindow then
    Result := Result + ', alt_sliding_window=true';
  if Config.RopeScaling.Mode <> rsmNone then
    Result := Result + ', rope_scaling=' +
      RoPEScalingToString(Config.RopeScaling);
  if Config.SlidingWindowPattern > 0 then
    Result := Result + ', sliding_window_pattern=' +
      IntToStr(Config.SlidingWindowPattern);
  if Config.RopeLocalTheta > 0 then
    Result := Result + ', rope_local_theta=' +
      FloatToStr(Config.RopeLocalTheta);
  if Config.FusedQKVGateUp then
    Result := Result + ', fused_qkv_gate_up=true';
  if Config.PostNormReordered then
    Result := Result + ', post_norm_reordered=true';
  if Config.QKNormFullWidth then
    Result := Result + ', qk_norm_full_width=true';
  if (Config.PartialRotaryFactor > 0) and
     (Config.PartialRotaryFactor < 1) then
    Result := Result + ', partial_rotary=' +
      FloatToStr(Config.PartialRotaryFactor);
  if Config.IsMoE then
    Result := Result + ', moe=' + IntToStr(Config.NumLocalExperts) +
      'x top-' + IntToStr(Config.MoEExpertsPerTok);
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
// RotaryDims > 0 RESTRICTS the rotate_half permutation to the first
// RotaryDims rows of each RotaryHeadDim-wide head (the PARTIAL-rotary
// layout of Phi-3; rows beyond the rotary slice keep their position) -
// 0 permutes the full head (RotaryDims = RotaryHeadDim).
// SrcRowBase/SrcRows slice a ROW BLOCK out of a larger FUSED slab (the
// Phi-3 qkv_proj / gate_up_proj packing): the checkpoint tensor is
// expected to have SrcRows rows (0 = OutDim, the whole tensor) and rows
// SrcRowBase..SrcRowBase+OutDim-1 are loaded - the q/k rotate_half
// permutation applies AFTER the slicing, within the destination block.
procedure LoadLlamaLinearWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName: string; InDim, OutDim: integer;
  NeuronBase: integer = 0; ExpectedNeurons: integer = -1;
  RotaryHeadDim: integer = 0; const BiasName: string = '';
  Scale: TNeuralFloat = 1.0; RotaryDims: integer = 0;
  SrcRowBase: integer = 0; SrcRows: integer = 0);
var
  W, B: TNNetVolume;
  i, j, TargetIdx, HeadIdx, RowInHead, TargetRow, HalfDim, SrcRow: integer;
begin
  EnsureWritableImportWeights(Layer);
  if ExpectedNeurons < 0 then ExpectedNeurons := OutDim;
  if SrcRows <= 0 then SrcRows := OutDim;
  if SrcRowBase + OutDim > SrcRows then
    ImportError('Llama import: internal error - row block ' +
      IntToStr(SrcRowBase) + '..' + IntToStr(SrcRowBase + OutDim - 1) +
      ' for "' + WName + '" exceeds the slab rows ' + IntToStr(SrcRows) + '.');
  if not Reader.HasTensor(WName) then
    ImportError('Llama import: missing tensor "' + WName + '".');
  if BiasName <> '' then
  begin
    if not Reader.HasTensor(BiasName) then
      ImportError('Llama import: missing tensor "' + BiasName + '".');
    if (Reader.DimCount(BiasName) <> 1) or
       (Reader.DimSize(BiasName, 0) <> SrcRows) then
      ImportError('Llama import: "' + BiasName + '" must have shape [' +
        IntToStr(SrcRows) + '], got ' + Reader.ShapeAsString(BiasName));
  end;
  if (Reader.DimCount(WName) <> 2) or
     (Reader.DimSize(WName, 0) <> SrcRows) or
     (Reader.DimSize(WName, 1) <> InDim) then
    ImportError('Llama import: "' + WName + '" must have shape [' +
      IntToStr(SrcRows) + ', ' + IntToStr(InDim) + '] (nn.Linear stores ' +
      '[out, in]), got ' + Reader.ShapeAsString(WName));
  if Layer.Neurons.Count <> ExpectedNeurons then
    ImportError('Llama import: internal error - layer for "' + WName +
      '" has ' + IntToStr(Layer.Neurons.Count) + ' neurons, expected ' +
      IntToStr(ExpectedNeurons) + '.');
  if (RotaryHeadDim > 0) and (RotaryDims <= 0) then
    RotaryDims := RotaryHeadDim; // full-head rotary (the Llama default)
  if (RotaryHeadDim > 0) and
     (((OutDim mod RotaryHeadDim) <> 0) or Odd(RotaryDims) or
      (RotaryDims > RotaryHeadDim)) then
    ImportError('Llama import: internal error - "' + WName + '" rows (' +
      IntToStr(OutDim) + ') are not a multiple of head_dim ' +
      IntToStr(RotaryHeadDim) + ' with an even rotary slice ' +
      IntToStr(RotaryDims) + '.');
  HalfDim := RotaryDims div 2;
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
        // rotate_half -> interleaved, restricted to the rotary slice
        // (RowInHead >= RotaryDims is the partial-rotary pass-through
        // tail and keeps its position).
        TargetRow := RowInHead;
        if RowInHead < RotaryDims then
        begin
          if RowInHead < HalfDim then
            TargetRow := 2 * RowInHead
          else
            TargetRow := 2 * (RowInHead - HalfDim) + 1;
        end;
        TargetIdx := HeadIdx * RotaryHeadDim + TargetRow;
      end
      else
        TargetIdx := j;
      TargetIdx := TargetIdx + NeuronBase;
      SrcRow := SrcRowBase + j;
      if Layer.Neurons[TargetIdx].Weights.Size <> InDim then
        ImportError('Llama import: internal error - neuron ' +
          IntToStr(TargetIdx) + ' for "' + WName + '" has ' +
          IntToStr(Layer.Neurons[TargetIdx].Weights.Size) +
          ' weights, expected ' + IntToStr(InDim) + '.');
      for i := 0 to InDim - 1 do
        Layer.Neurons[TargetIdx].Weights.FData[i] :=
          Scale * W.FData[SrcRow * InDim + i];
      if B <> nil then
        Layer.Neurons[TargetIdx].BiasWeight := Scale * B.FData[SrcRow]
      else
        Layer.Neurons[TargetIdx].BiasWeight := 0; // bias-free Linear
    end;
  finally
    B.Free;
    W.Free;
  end;
  Layer.FlushWeightCache();
end;

// Loads a bias-free nn.Linear [OutDim, InDim] matrix straight onto a pointwise
// layer's neurons (neuron j gets row j, weight i = W[j*InDim+i]), each weight
// multiplied by Scale. Used by the PEFT LoRA loader to drop lora_A/lora_B onto
// the AddLoRAAdapter A/B projections (which carry exactly this layout).
procedure LoadPlainLinearWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName: string; InDim, OutDim: integer;
  Scale: TNeuralFloat = 1.0);
var
  W: TNNetVolume;
  i, j: integer;
begin
  EnsureWritableImportWeights(Layer);
  if not Reader.HasTensor(WName) then
    ImportError('PEFT import: missing tensor "' + WName + '".');
  if (Reader.DimCount(WName) <> 2) or
     (Reader.DimSize(WName, 0) <> OutDim) or
     (Reader.DimSize(WName, 1) <> InDim) then
    ImportError('PEFT import: "' + WName + '" must have shape [' +
      IntToStr(OutDim) + ', ' + IntToStr(InDim) + '] (nn.Linear stores ' +
      '[out, in]), got ' + Reader.ShapeAsString(WName));
  if Layer.Neurons.Count <> OutDim then
    ImportError('PEFT import: layer for "' + WName + '" has ' +
      IntToStr(Layer.Neurons.Count) + ' neurons, expected ' +
      IntToStr(OutDim) + '.');
  W := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, W);
    for j := 0 to OutDim - 1 do
    begin
      if Layer.Neurons[j].Weights.Size <> InDim then
        ImportError('PEFT import: neuron ' + IntToStr(j) + ' for "' + WName +
          '" has ' + IntToStr(Layer.Neurons[j].Weights.Size) +
          ' weights, expected ' + IntToStr(InDim) + '.');
      for i := 0 to InDim - 1 do
        Layer.Neurons[j].Weights.FData[i] := Scale * W.FData[j * InDim + i];
      Layer.Neurons[j].BiasWeight := 0; // LoRA projections are bias-free
    end;
  finally
    W.Free;
  end;
  Layer.FlushWeightCache();
end;

function ReadPEFTAdapterConfig(const FileName: string): TPEFTAdapterConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  Field: TJSONData;
begin
  Result.Alpha := 1.0;
  Result.Rank := 1;
  Result.RankFound := False;
  if not FileExists(FileName) then
    ImportError('PEFT import: adapter_config.json not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('PEFT import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('PEFT import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    Field := Obj.Find('r');
    if (Field <> nil) and not Field.IsNull then
    begin
      Result.Rank := Field.AsInteger;
      Result.RankFound := True;
      if Result.Rank < 1 then
        ImportError('PEFT import: config "r" must be a positive integer, got '
          + Field.AsJSON + '.');
    end;
    // lora_alpha defaults to r (PEFT scale 1.0) when absent.
    Result.Alpha := Obj.Get('lora_alpha', TNeuralFloat(Result.Rank));
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function LoadPEFTLoRAModule(Reader: TNNetSafeTensorsReader;
  ADown, BUp: TNNetLayer; const ModuleName: string;
  Scale: TNeuralFloat): string;
var
  Rank, d_in, d_out, p, s: integer;
  Prefixes, Segments: array of string;
  Base, AName, BName: string;
  Found: boolean;
begin
  if (ADown = nil) or (BUp = nil) then
    ImportError('PEFT import: LoadPEFTLoRAModule requires non-nil A and B ' +
      'layers (module "' + ModuleName + '").');
  Rank := ADown.Neurons.Count;
  d_out := BUp.Neurons.Count;
  if Rank < 1 then
    ImportError('PEFT import: A layer for "' + ModuleName +
      '" has no neurons.');
  d_in := ADown.Neurons[0].Weights.Size;
  // PEFT names are "<prefix><module>.lora_A<seg>weight" where <prefix> is
  // usually "base_model.model." and <seg> is ".default." (the adapter name)
  // or "." (no adapter-name segment). Try the common combinations.
  SetLength(Prefixes, 2);
  Prefixes[0] := 'base_model.model.';
  Prefixes[1] := '';
  SetLength(Segments, 2);
  Segments[0] := '.default.';
  Segments[1] := '.';
  Found := False;
  Base := '';
  AName := '';
  BName := '';
  for p := 0 to High(Prefixes) do
  begin
    for s := 0 to High(Segments) do
    begin
      Base := Prefixes[p] + ModuleName;
      AName := Base + '.lora_A' + Segments[s] + 'weight';
      BName := Base + '.lora_B' + Segments[s] + 'weight';
      if Reader.HasTensor(AName) and Reader.HasTensor(BName) then
      begin
        Found := True;
        break;
      end;
    end;
    if Found then break;
  end;
  if not Found then
    ImportError('PEFT import: could not find lora_A/lora_B tensors for module '
      + '"' + ModuleName + '" (tried prefixes "base_model.model." and "", ' +
      'adapter segments ".default." and ".").');
  // A: [Rank, d_in] -> A layer (Rank neurons, d_in weights). The PEFT scaling
  // alpha/r is folded into B, so A loads unscaled.
  LoadPlainLinearWeights(Reader, ADown, AName, d_in, Rank, 1.0);
  // B: [d_out, Rank] -> B layer (d_out neurons, Rank weights), scaled.
  LoadPlainLinearWeights(Reader, BUp, BName, Rank, d_out, Scale);
  Result := Base;
end;

// Loads a Qwen3-style SHARED per-head RMSNorm gain vector [HeadDim] into
// every per-head TNNetTokenRMSNorm copy in NormLayers (q_norm/k_norm are
// stored ONCE in the checkpoint but applied to each head's q/k slice).
// The per-head q/k slices live in the rotate_half-PERMUTED channel order
// produced by LoadLlamaLinearWeights (RotaryHeadDim), so the gain is
// permuted the same way: target channel 2k <- HF position k, target
// channel 2k+1 <- HF position k + HeadDim/2. The RMS denominator itself is
// permutation-invariant (mean of squares over the whole head).
// GainOffset is added to every gain BEFORE scaling (Gemma-3's zero-centered
// norms store w with gain 1+w, so GainOffset=1 there; Qwen3 uses 0).
// Scale multiplies the resulting gain: the Gemma-3 query_pre_attn_scalar
// fold lands HERE (stored gain = Scale*(GainOffset+w)) because a scale
// folded into W_q would be ERASED by the q-side RMSNorm; the scalar
// commutes with RoPE (a rotation), so scaling the norm gain is exact.
procedure LoadLlamaHeadRMSNormWeights(Reader: TNNetSafeTensorsReader;
  const NormLayers: array of TNNetLayer; const WName: string;
  HeadDim: integer; GainOffset: TNeuralFloat = 0;
  Scale: TNeuralFloat = 1.0);
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
          Scale * (GainOffset + Tmp.FData[j]); // gain
      end;
      NormLayers[HeadCnt].FlushWeightCache();
    end;
  finally
    Tmp.Free;
  end;
end;

// Loads a HF FULL-WIDTH q/k RMSNorm gain [num_heads*head_dim] (OLMo-2
// QKNormFullWidth: ONE norm over the whole flattened projection output,
// applied BEFORE the head split + RoPE) into a single TNNetTokenRMSNorm.
// The q/k projection ROWS were loaded with the per-head rotate_half ->
// interleaved-pair permutation, so each head's HeadDim-wide gain slice is
// permuted the same way (the RMS statistic itself is permutation-invariant,
// only the per-channel gains must follow their channels).
procedure LoadLlamaFullWidthQKNormWeights(Reader: TNNetSafeTensorsReader;
  NormLayer: TNNetLayer; const WName: string; Width, HeadDim: integer;
  GainOffset: TNeuralFloat = 0);
var
  Tmp: TNNetVolume;
  HeadCnt, j, TargetIdx, HalfDim: integer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('Llama import: missing tensor "' + WName + '".');
  if (Reader.DimCount(WName) <> 1) or
     (Reader.DimSize(WName, 0) <> Width) then
    ImportError('Llama import: "' + WName + '" must have shape [' +
      IntToStr(Width) + '], got ' + Reader.ShapeAsString(WName));
  HalfDim := HeadDim div 2;
  Tmp := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, Tmp);
    for HeadCnt := 0 to (Width div HeadDim) - 1 do
    begin
      for j := 0 to HeadDim - 1 do
      begin
        if j < HalfDim then
          TargetIdx := 2 * j
        else
          TargetIdx := 2 * (j - HalfDim) + 1;
        NormLayer.Neurons[0].Weights.FData[HeadCnt * HeadDim + TargetIdx] :=
          GainOffset + Tmp.FData[HeadCnt * HeadDim + j]; // gain
      end;
    end;
    NormLayer.FlushWeightCache();
  finally
    Tmp.Free;
  end;
end;

type
  TLlamaBlockLayers = record
    AttnNorm, QProj, KProj, VProj, OProj, MlpNorm, GateUp, Down: TNNetLayer;
    // Sandwich norms (Gemma-2 SandwichNorm): post-attention and
    // post-feedforward RMSNorms INSIDE the residual branches; nil otherwise.
    // OLMo-2's PostNormReordered REUSES the two fields (its
    // post_attention_layernorm / post_feedforward_layernorm sit in the same
    // place - the sublayer output, before the residual add - it just drops
    // the entry norms: AttnNorm and MlpNorm stay nil).
    PostAttnNorm, PostMlpNorm: TNNetLayer;
    // Per-head q/k RMSNorm copies (Qwen3 QKNorm); empty otherwise.
    QNorms, KNorms: array of TNNetLayer;
    // FULL-WIDTH q/k RMSNorm (OLMo-2 QKNormFullWidth); nil otherwise.
    QNormFull, KNormFull: TNNetLayer;
    // Mixtral block_sparse_moe (Config.IsMoE): the router gate conv plus
    // per-expert fused gate|up and down projections; empty otherwise.
    GateConv: TNNetLayer;
    ExpertGateUp, ExpertDown: array of TNNetLayer;
  end;

// Wires Mixtral's block_sparse_moe FFN from primitives onto MoESource (the
// pre-FFN-normalized residual stream): a token-wise router (gate linear ->
// num_local_experts logits -> PER-TOKEN softmax -> hard top-k gate with the
// surviving probs RENORMALIZED, HF's normalize_topk_prob), N independent
// SwiGLU experts, and y = Sum_e gTopK[e] * Expert_e(x). This is the
// AddTopKMixtureOfExperts combine specialized to SwiGLU experts so the
// importer can drop the checkpoint's per-expert w1/w2/w3 onto named layers.
// The gate's PER-TOKEN softmax matches HF (the whole-volume TNNetSoftMax
// would couple tokens); pRenormalize=true is Mixtral's top-k renorm, the
// one routing knob that distinguishes it from DeepSeek-V2 (norm=false).
// Leaves the combined sum as the last layer (the caller closes the residual).
procedure BuildMixtralMoEBranch(NN: TNNet; var Block: TLlamaBlockLayers;
  MoESource: TNNetLayer; const Config: TLlamaConfig);
var
  GateTopK, ExpertOut, GateE, GateEBroadcast: TNNetLayer;
  MoEBranches: array of TNNetLayer;
  ExpertCnt: integer;
begin
  SetLength(Block.ExpertGateUp, Config.NumLocalExperts);
  SetLength(Block.ExpertDown, Config.NumLocalExperts);
  SetLength(MoEBranches, Config.NumLocalExperts);
  // Router: token-wise linear -> PER-TOKEN softmax (over ALL experts) ->
  // hard top-k gate with the top-k subset renormalized (pRenormalize=true).
  Block.GateConv := NN.AddLayerAfter(
    TNNetPointwiseConvLinear.Create(Config.NumLocalExperts), MoESource);
  NN.AddLayer( TNNetPointwiseSoftMax.Create() );
  GateTopK := NN.AddLayer( TNNetTopKGate.Create(
    Config.MoEExpertsPerTok, {pRenormalize=}true) );
  for ExpertCnt := 0 to Config.NumLocalExperts - 1 do
  begin
    // Per-expert SwiGLU MLP. TNNetSwiGLU computes FIRSTHALF * silu(SECONDHALF):
    // the fused projection holds w3 (up) in neurons 0..I-1 and w1 (gate) in
    // I..2I-1; w2 is the down projection (loaded below).
    Block.ExpertGateUp[ExpertCnt] := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(2 * Config.IntermediateSize), MoESource);
    NN.AddLayer( TNNetSwiGLU.Create() );
    Block.ExpertDown[ExpertCnt] := NN.AddLayer(
      TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
    ExpertOut := NN.GetLastLayer();
    // Slice this expert's renormalized top-k gate weight g[e], broadcast it
    // across d_model and cell-multiply it in (zero for non-selected experts).
    GateE := NN.AddLayerAfter(
      TNNetSplitChannels.Create(ExpertCnt, 1), GateTopK);
    GateEBroadcast := NN.AddLayer(
      TNNetDeepConcat.Replicate(Config.HiddenSize, GateE) );
    MoEBranches[ExpertCnt] := NN.AddLayer(
      TNNetCellMulByCell.Create(ExpertOut, GateEBroadcast) );
  end;
  // y = Sum_e gTopK[e] * Expert_e(x). (NumExpertsPerTok>=2 routing implies
  // NumLocalExperts>=2; a single-expert sum is still valid for safety.)
  NN.AddLayer( TNNetSum.Create(MoEBranches) );
  SetLength(MoEBranches, 0);
end;

// The Llama builder core: builds the net and loads every weight from the
// ALREADY-OPEN pReader (whose tensor table must use the HF tensor names).
// Takes OWNERSHIP of pReader (frees it on every path). FileName is used in
// error messages only. BuildLlamaFromSafeTensorsWithConfig wraps this with
// CreatePretrainedTensorReader; BuildLlamaFromGGUFEx wraps it with a
// TNNetGGUFReader whose tensors were renamed to the HF names.
function BuildLlamaFromTensorReaderWithConfig(
  pReader: TNNetSafeTensorsReader; const FileName: string;
  var Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TLlamaBlockLayers;
  EmbeddingLayer, FinalNorm, LMHead: TNNetLayer;
  BranchInput, NormedSource: TNNetLayer;
  QSource, KSource: TNNetLayer;
  QSlice, KSlice, VSlice, HeadPack, RotSlice, PassSlice: TNNetLayer;
  KRotated, VSlices, HeadOutputs: array of TNNetLayer;
  SliceChannels, RotChannels, PassChannels: array of integer;
  BlockCnt, SeqLen, HeadCnt, KVHeadCnt, KVGroup, GroupSize: integer;
  HeadDim, QWidth, KVWidth, RotaryDims, LayerWindow, i, j, d: integer;
  LayerIsLocal: boolean;
  NormGainOffset, QScale, QProjScale, LayerTheta: TNeuralFloat;
  LayerRoPEScaling: TRoPEScalingConfig;
  Tmp: TNNetVolume;
  BlockPrefix, TensorNameStr, LMHeadName, FusedName: string;
  QBiasName, KBiasName, VBiasName: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

begin
  Reader := pReader; // ownership taken: freed in the finally below
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
      // Partial rotary (Phi-3): RoPE rotates only the first
      // int(head_dim * partial_rotary_factor) channels of each q/k head
      // (HF: rotary_dim = int(head_dim * partial_rotary_factor)); the
      // tail passes through unrotated.
      if (Config.PartialRotaryFactor > 0) and
         (Config.PartialRotaryFactor < 1) then
        RotaryDims := Trunc(HeadDim * Config.PartialRotaryFactor)
      else
        RotaryDims := HeadDim;
      if (RotaryDims < 2) or Odd(RotaryDims) then
        ImportError('Llama import: rotary_dim = int(head_dim * ' +
          'partial_rotary_factor) = ' + IntToStr(RotaryDims) +
          ' must be an even number >= 2 (RoPE rotates channel pairs).');
      // LongRoPE per-frequency table must have exactly RotaryDims/2 entries
      // (one per rotated channel pair) - HF's long_factor is rotary_ndims/2
      // long. Validate here for a clear message before the layer build.
      if (Config.RopeScaling.Mode = rsmLongRoPE) and
         (Length(Config.RopeScaling.LongFactors) <> (RotaryDims div 2)) then
        ImportError('Llama import: rope_scaling longrope long_factor has ' +
          IntToStr(Length(Config.RopeScaling.LongFactors)) + ' entries but ' +
          'rotary_dim/2 = ' + IntToStr(RotaryDims div 2) +
          ' are required (one factor per rotated channel pair).');
      if Config.QKNorm and (RotaryDims < HeadDim) then
        ImportError('Llama import: internal error - partial rotary ' +
          'combined with per-head q/k RMSNorm is not wired (no family ' +
          'uses both).');
      if Config.QKNormFullWidth and (RotaryDims < HeadDim) then
        ImportError('Llama import: internal error - partial rotary ' +
          'combined with full-width q/k RMSNorm is not wired (no family ' +
          'uses both).');
      if Config.QKNormFullWidth and Config.QKNorm then
        ImportError('Llama import: internal error - per-head (QKNorm) and ' +
          'full-width (QKNormFullWidth) q/k RMSNorm are mutually ' +
          'exclusive.');
      if Config.PostNormReordered and Config.SandwichNorm then
        ImportError('Llama import: internal error - PostNormReordered ' +
          '(OLMo-2, no entry norms) and SandwichNorm (Gemma-2, entry + ' +
          'exit norms) are mutually exclusive.');
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
      // pQuantizeInt8: idempotent block-by-block sweeps (same pattern as
      // MakeInferenceOnly) keep peak RAM at quantized-net + one FP32 block
      // during BOTH construction and the weight-load phase below (the
      // loaders call DequantizeWeightsInt8 before refilling a layer).
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      SetLength(Blocks, Config.NumLayers);
      SetLength(KRotated, Config.NumKVHeads);
      SetLength(VSlices, Config.NumKVHeads);
      SetLength(HeadOutputs, Config.NumHeads);
      SetLength(SliceChannels, HeadDim);
      SetLength(RotChannels, RotaryDims);
      SetLength(PassChannels, HeadDim - RotaryDims);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // Attention sub-block: x := x + o_proj(rotary-GQA(RMSNorm(x))).
        // Wired from primitives exactly like AddMultiHeadGroupedQueryAttention
        // with TNNetRotaryEmbedding(rope_theta) on each per-head Q/K slice
        // (depth = head_dim so the frequency schedule matches HF).
        // Per-layer attention type:
        //  - Config.SlidingWindowPattern > 1 (Gemma-3): every Nth layer
        //    ((BlockCnt+1) mod N = 0, transformers' Gemma3TextConfig
        //    layer_types default) attends GLOBALLY; the rest carry the
        //    sliding window (5:1 local:global at the default pattern 6);
        //  - Config.AltSlidingWindow (Gemma-2): the window applies to EVEN
        //    layers (transformers' layer_types default: "sliding_attention"
        //    for layers 0, 2, ...); odd layers attend over the full context;
        //  - otherwise (Mistral) every layer is local iff SlidingWindow > 0.
        // Config.RopeLocalTheta > 0 (Gemma-3) selects a PER-LAYER-TYPE RoPE
        // base: sliding layers rotate with rope_local_base_freq (10k),
        // global layers with rope_theta (1e6).
        if Config.SlidingWindowPattern > 1 then
          LayerIsLocal := ((BlockCnt + 1) mod Config.SlidingWindowPattern) <> 0
        else if Config.AltSlidingWindow then
          LayerIsLocal := not Odd(BlockCnt)
        else
          LayerIsLocal := Config.SlidingWindow > 0;
        if LayerIsLocal then
          LayerWindow := Config.SlidingWindow
        else
          LayerWindow := 0;
        // rope_scaling: HF Gemma-3 applies it ONLY to the GLOBAL rope
        // (rope_local_base_freq layers keep the default unscaled
        // parameters), so sliding layers with a dedicated local theta stay
        // unscaled; every other family scales all layers.
        if LayerIsLocal and (Config.RopeLocalTheta > 0) then
        begin
          LayerTheta := Config.RopeLocalTheta;
          LayerRoPEScaling := DefaultRoPEScaling();
        end
        else
        begin
          LayerTheta := Config.RopeTheta;
          LayerRoPEScaling := Config.RopeScaling;
        end;
        BranchInput := NN.GetLastLayer();
        // Config.PostNormReordered (OLMo-2): NO input_layernorm - the
        // attention sub-block reads the raw residual stream and the RMSNorm
        // moves to the SUBLAYER OUTPUT (PostAttnNorm below):
        // x := x + Norm(Attn(x)).
        if Config.PostNormReordered then
          NormedSource := BranchInput
        else
        begin
          Blocks[BlockCnt].AttnNorm :=
            NN.AddLayer( TNNetTokenRMSNorm.Create(Config.RmsNormEps) );
          NormedSource := NN.GetLastLayer();
        end;
        Blocks[BlockCnt].QProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(QWidth), NormedSource);
        Blocks[BlockCnt].KProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(KVWidth), NormedSource);
        Blocks[BlockCnt].VProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(KVWidth), NormedSource);
        // Config.QKNormFullWidth (OLMo-2): ONE RMSNorm over the FULL
        // flattened q/k projection width (num_heads*head_dim /
        // num_kv_heads*head_dim) AFTER the projection and BEFORE the head
        // split + RoPE (HF modeling_olmo2: q_norm(q_proj(x)) over the whole
        // width, then view(heads, head_dim) and apply_rotary_pos_emb) -
        // distinct from the Qwen3/Gemma-3 PER-HEAD QKNorm whose RMS
        // statistic spans head_dim channels only.
        QSource := Blocks[BlockCnt].QProj;
        KSource := Blocks[BlockCnt].KProj;
        if Config.QKNormFullWidth then
        begin
          Blocks[BlockCnt].QNormFull := NN.AddLayerAfter(
            TNNetTokenRMSNorm.Create(Config.RmsNormEps),
            Blocks[BlockCnt].QProj);
          QSource := Blocks[BlockCnt].QNormFull;
          Blocks[BlockCnt].KNormFull := NN.AddLayerAfter(
            TNNetTokenRMSNorm.Create(Config.RmsNormEps),
            Blocks[BlockCnt].KProj);
          KSource := Blocks[BlockCnt].KNormFull;
        end;
        // Config.QKNorm (Qwen3, Gemma-3): per-head RMSNorm on each q/k
        // slice AFTER the projection and BEFORE RoPE (the HF
        // modeling_qwen3 AND modeling_gemma3 ordering, both verified:
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
          if RotaryDims < HeadDim then
          begin
            // PARTIAL rotary (Phi-3, Config.PartialRotaryFactor < 1):
            // RoPE on the first RotaryDims channels of the head only -
            // both slices are cut straight off the K projection and
            // re-concatenated [rotated | pass-through]. The rotary slice
            // feeds a depth-RotaryDims TNNetRotaryEmbedding, so the
            // frequency schedule theta^(-2k/RotaryDims) matches HF's
            // inv_freq over rotary_dim exactly. (QKNorm + partial rotary
            // is rejected up front - no family combines them.)
            for d := 0 to RotaryDims - 1 do
              RotChannels[d] := KVHeadCnt * HeadDim + d;
            for d := 0 to HeadDim - RotaryDims - 1 do
              PassChannels[d] := KVHeadCnt * HeadDim + RotaryDims + d;
            RotSlice := NN.AddLayerAfter(
              TNNetSplitChannels.Create(RotChannels), KSource);
            RotSlice := NN.AddLayerAfter(
              CreateRoPEFromScaling(LayerTheta, LayerRoPEScaling), RotSlice);
            PassSlice := NN.AddLayerAfter(
              TNNetSplitChannels.Create(PassChannels), KSource);
            KRotated[KVHeadCnt] := NN.AddLayer(
              TNNetDeepConcat.Create([RotSlice, PassSlice]) );
          end
          else
          begin
            KSlice := NN.AddLayerAfter(
              TNNetSplitChannels.Create(SliceChannels), KSource);
            if Config.QKNorm then
            begin
              Blocks[BlockCnt].KNorms[KVHeadCnt] := NN.AddLayerAfter(
                TNNetTokenRMSNorm.Create(Config.RmsNormEps), KSlice);
              KSlice := Blocks[BlockCnt].KNorms[KVHeadCnt];
            end;
            KRotated[KVHeadCnt] := NN.AddLayerAfter(
              CreateRoPEFromScaling(LayerTheta, LayerRoPEScaling), KSlice);
          end;
          VSlices[KVHeadCnt] := NN.AddLayerAfter(
            TNNetSplitChannels.Create(SliceChannels), Blocks[BlockCnt].VProj);
        end;
        for HeadCnt := 0 to Config.NumHeads - 1 do
        begin
          KVGroup := HeadCnt div GroupSize;
          for d := 0 to HeadDim - 1 do
            SliceChannels[d] := HeadCnt * HeadDim + d;
          if RotaryDims < HeadDim then
          begin
            // Partial rotary on the Q head - same wiring as the K path.
            for d := 0 to RotaryDims - 1 do
              RotChannels[d] := HeadCnt * HeadDim + d;
            for d := 0 to HeadDim - RotaryDims - 1 do
              PassChannels[d] := HeadCnt * HeadDim + RotaryDims + d;
            RotSlice := NN.AddLayerAfter(
              TNNetSplitChannels.Create(RotChannels), QSource);
            RotSlice := NN.AddLayerAfter(
              CreateRoPEFromScaling(LayerTheta, LayerRoPEScaling), RotSlice);
            PassSlice := NN.AddLayerAfter(
              TNNetSplitChannels.Create(PassChannels), QSource);
            QSlice := NN.AddLayer(
              TNNetDeepConcat.Create([RotSlice, PassSlice]) );
          end
          else
          begin
            QSlice := NN.AddLayerAfter(
              TNNetSplitChannels.Create(SliceChannels), QSource);
            if Config.QKNorm then
            begin
              Blocks[BlockCnt].QNorms[HeadCnt] := NN.AddLayerAfter(
                TNNetTokenRMSNorm.Create(Config.RmsNormEps), QSlice);
              QSlice := Blocks[BlockCnt].QNorms[HeadCnt];
            end;
            QSlice := NN.AddLayerAfter(
              CreateRoPEFromScaling(LayerTheta, LayerRoPEScaling), QSlice);
          end;
          // Pack [Q_h | K_group | V_group] (width 3*head_dim) for SDPA.
          HeadPack := NN.AddLayer( TNNetDeepConcat.Create(
            [QSlice, KRotated[KVGroup], VSlices[KVGroup]]) );
          // LayerWindow > 0 applies the same banded causal sliding-window
          // mask HF uses: query i attends keys j with i - j < W (and j <= i
          // from the causal mask); 0 = full attention. The per-layer value
          // is computed at the top of the block loop (Mistral: every layer;
          // Gemma-2: even layers; Gemma-3: all but every Nth layer).
          // Config.AttnLogitSoftCap > 0 (Gemma-2) enables the pre-softmax
          // attention-logit soft-cap cap*tanh(s/cap) inside every head.
          HeadOutputs[HeadCnt] := NN.AddLayerAfter(
            TNNetScaledDotProductAttention.Create(HeadDim, {CausalMask=}true,
              {pWindow=}LayerWindow,
              {pScoreSoftCap=}Config.AttnLogitSoftCap),
            HeadPack);
        end;
        NN.AddLayer( TNNetDeepConcat.Create(HeadOutputs) );
        Blocks[BlockCnt].OProj := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        // Config.SandwichNorm (Gemma-2) / Config.PostNormReordered (OLMo-2):
        // post-attention RMSNorm INSIDE the residual branch (HF
        // post_attention_layernorm normalizes the attention output BEFORE
        // the residual add). OLMo-2 drops the entry norm above, making this
        // the block's ONLY attention norm: x + Norm(Attn(x)).
        if Config.SandwichNorm or Config.PostNormReordered then
          Blocks[BlockCnt].PostAttnNorm :=
            NN.AddLayer( TNNetTokenRMSNorm.Create(Config.RmsNormEps) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        // MLP sub-block: x := x + down(act(gate(h)) * up(h)), h = RMSNorm(x),
        // where act is SiLU (SwiGLU, the Llama default) or tanh-GELU (GeGLU,
        // Gemma's gelu_pytorch_tanh - Config.GegluFFN). Both TNNetSwiGLU and
        // TNNetGEGLU compute FIRSTHALF * act(SECONDHALF), so the fused
        // projection holds up_proj in neurons 0..I-1 and gate_proj in
        // neurons I..2I-1 (see LoadLlamaLinearWeights calls below) for
        // either activation.
        BranchInput := NN.GetLastLayer();
        // Config.PostNormReordered (OLMo-2): no pre-FFN norm either - the
        // MLP reads the raw residual stream and PostMlpNorm (below) closes
        // the branch: x := x + Norm(MLP(x)).
        if not Config.PostNormReordered then
          Blocks[BlockCnt].MlpNorm :=
            NN.AddLayer( TNNetTokenRMSNorm.Create(Config.RmsNormEps) );
        if Config.IsMoE then
          BuildMixtralMoEBranch(NN, Blocks[BlockCnt], NN.GetLastLayer(),
            Config)
        else
        begin
          Blocks[BlockCnt].GateUp := NN.AddLayer(
            TNNetPointwiseConvLinear.Create(2 * Config.IntermediateSize) );
          if Config.GegluFFN then
            NN.AddLayer( TNNetGEGLU.Create() )
          else
            NN.AddLayer( TNNetSwiGLU.Create() );
          Blocks[BlockCnt].Down := NN.AddLayer(
            TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        end;
        // Config.SandwichNorm (Gemma-2) / Config.PostNormReordered (OLMo-2):
        // post-feedforward RMSNorm INSIDE the residual branch (HF
        // post_feedforward_layernorm).
        if Config.SandwichNorm or Config.PostNormReordered then
          Blocks[BlockCnt].PostMlpNorm :=
            NN.AddLayer( TNNetTokenRMSNorm.Create(Config.RmsNormEps) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      FinalNorm := NN.AddLayer(
        TNNetTokenRMSNorm.Create(Config.RmsNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      // Config.FinalLogitSoftCap (Gemma-2): the LM-head logits are squashed
      // to cap*tanh(logits/cap) - a plain TNNetSoftCapping after the head
      // (HF applies it to the logits before any sampling softmax).
      if Config.FinalLogitSoftCap > 0 then
        NN.AddLayer( TNNetSoftCapping.Create(Config.FinalLogitSoftCap) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

      // ---------------- Weights ----------------
      // Config.RMSNormAddOne (Gemma): every RMSNorm gain is stored as 1 + w
      // (zero-centered checkpoint weights) - folded in by the loader.
      if Config.RMSNormAddOne then NormGainOffset := 1.0
      else NormGainOffset := 0;
      // Config.QueryPreAttnScalar (Gemma-2): HF scales the attention scores
      // by 1/sqrt(query_pre_attn_scalar) instead of SDPA's structural
      // 1/sqrt(head_dim). Folded into W_q at load (the GPT-Neo unscaled-
      // attention trick): scaling every q row by sqrt(head_dim/scalar)
      // turns SDPA's 1/sqrt(head_dim) into the desired 1/sqrt(scalar).
      // The fold commutes with the per-head rotate_half permutation and
      // with RoPE (a rotation); it would NOT commute with a q-side RMSNorm
      // (the norm ERASES any scale folded into W_q), so when QKNorm is on
      // (Gemma-3 combines it with QueryPreAttnScalar) the fold moves into
      // the per-head q_norm GAINS instead (still ahead of RoPE, which
      // commutes with a scalar).
      if Config.QueryPreAttnScalar > 0 then
        QScale := Sqrt(HeadDim / Config.QueryPreAttnScalar)
      else
        QScale := 1.0;
      if Config.QKNorm or Config.QKNormFullWidth then
        QProjScale := 1.0
      else
        QProjScale := QScale;
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
          EnsureWritableImportWeights(LMHead);
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
      // Re-quantize the refilled LM head before streaming the blocks.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        BlockPrefix := Config.Prefix + 'layers.' + IntToStr(BlockCnt) + '.';
        // Config.PostNormReordered (OLMo-2): there is NO input_layernorm -
        // the block's two norms load into the post-norms below.
        if not Config.PostNormReordered then
        begin
          LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].AttnNorm,
            BlockPrefix + 'input_layernorm.weight', Config.HiddenSize,
            NormGainOffset);
          MarkConsumed(BlockPrefix + 'input_layernorm.weight');
        end;
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
        if Config.FusedQKVGateUp then
        begin
          // Phi-3 fused qkv_proj: q|k|v packed rows
          // (q = num_heads*head_dim, then k and v = kv_heads*head_dim
          // each), no biases. Each block of rows is sliced into its
          // separate projection layer; the rotate_half->interleaved
          // permutation applies AFTER the slicing, restricted to the
          // RotaryDims-wide rotary slice of each head (partial rotary).
          FusedName := BlockPrefix + 'self_attn.qkv_proj.weight';
          LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].QProj, FusedName,
            Config.HiddenSize, QWidth, 0, -1, HeadDim, '', QProjScale,
            RotaryDims, 0, QWidth + 2 * KVWidth);
          LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].KProj, FusedName,
            Config.HiddenSize, KVWidth, 0, -1, HeadDim, '', 1.0,
            RotaryDims, QWidth, QWidth + 2 * KVWidth);
          LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].VProj, FusedName,
            Config.HiddenSize, KVWidth, 0, -1, 0, '', 1.0,
            0, QWidth + KVWidth, QWidth + 2 * KVWidth);
          MarkConsumed(FusedName);
        end
        else
        begin
          LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].QProj,
            BlockPrefix + 'self_attn.q_proj.weight',
            Config.HiddenSize, QWidth, 0, -1, HeadDim, QBiasName, QProjScale);
          MarkConsumed(BlockPrefix + 'self_attn.q_proj.weight');
          if QBiasName <> '' then MarkConsumed(QBiasName);
          // Qwen3/Gemma-3 per-head q/k RMSNorm: one shared [head_dim] gain
          // per block, copied into every per-head norm (rotate_half-permuted
          // to match the permuted q/k channel order). Gemma-3's zero-centered
          // gains get the +1 offset (NormGainOffset), and the
          // query_pre_attn_scalar fold rides the q-side gains (QScale; see
          // above - a W_q fold would be erased by the norm).
          if Config.QKNorm then
          begin
            LoadLlamaHeadRMSNormWeights(Reader, Blocks[BlockCnt].QNorms,
              BlockPrefix + 'self_attn.q_norm.weight', HeadDim,
              NormGainOffset, QScale);
            MarkConsumed(BlockPrefix + 'self_attn.q_norm.weight');
            LoadLlamaHeadRMSNormWeights(Reader, Blocks[BlockCnt].KNorms,
              BlockPrefix + 'self_attn.k_norm.weight', HeadDim,
              NormGainOffset);
            MarkConsumed(BlockPrefix + 'self_attn.k_norm.weight');
          end;
          // OLMo-2 full-width q/k RMSNorm: ONE [num_heads*head_dim] /
          // [num_kv_heads*head_dim] gain per block, loaded with the same
          // per-head rotate_half permutation as the q/k projection ROWS
          // (the RMS statistic is permutation-invariant; the per-channel
          // gains must follow their permuted channels).
          if Config.QKNormFullWidth then
          begin
            LoadLlamaFullWidthQKNormWeights(Reader,
              Blocks[BlockCnt].QNormFull,
              BlockPrefix + 'self_attn.q_norm.weight', QWidth, HeadDim,
              NormGainOffset);
            MarkConsumed(BlockPrefix + 'self_attn.q_norm.weight');
            LoadLlamaFullWidthQKNormWeights(Reader,
              Blocks[BlockCnt].KNormFull,
              BlockPrefix + 'self_attn.k_norm.weight', KVWidth, HeadDim,
              NormGainOffset);
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
        end;
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].OProj,
          BlockPrefix + 'self_attn.o_proj.weight',
          QWidth, Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'self_attn.o_proj.weight');
        if Config.SandwichNorm then
        begin
          // Gemma-2 sandwich norms: post_attention_layernorm is the TRUE
          // post-attention norm (inside the residual branch) and the FFN's
          // pre-norm is pre_feedforward_layernorm; post_feedforward_layernorm
          // closes the FFN branch.
          LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].PostAttnNorm,
            BlockPrefix + 'post_attention_layernorm.weight',
            Config.HiddenSize, NormGainOffset);
          MarkConsumed(BlockPrefix + 'post_attention_layernorm.weight');
          LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].MlpNorm,
            BlockPrefix + 'pre_feedforward_layernorm.weight',
            Config.HiddenSize, NormGainOffset);
          MarkConsumed(BlockPrefix + 'pre_feedforward_layernorm.weight');
          LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].PostMlpNorm,
            BlockPrefix + 'post_feedforward_layernorm.weight',
            Config.HiddenSize, NormGainOffset);
          MarkConsumed(BlockPrefix + 'post_feedforward_layernorm.weight');
        end
        else if Config.PostNormReordered then
        begin
          // OLMo-2 reordered post-norms: post_attention_layernorm closes
          // the attention branch and post_feedforward_layernorm the FFN
          // branch (both INSIDE the residual, before the add); there are
          // no entry norms.
          LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].PostAttnNorm,
            BlockPrefix + 'post_attention_layernorm.weight',
            Config.HiddenSize, NormGainOffset);
          MarkConsumed(BlockPrefix + 'post_attention_layernorm.weight');
          LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].PostMlpNorm,
            BlockPrefix + 'post_feedforward_layernorm.weight',
            Config.HiddenSize, NormGainOffset);
          MarkConsumed(BlockPrefix + 'post_feedforward_layernorm.weight');
        end
        else
        begin
          // Llama naming: post_attention_layernorm is the FFN's pre-norm.
          LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].MlpNorm,
            BlockPrefix + 'post_attention_layernorm.weight',
            Config.HiddenSize, NormGainOffset);
          MarkConsumed(BlockPrefix + 'post_attention_layernorm.weight');
        end;
        if Config.IsMoE then
        begin
          // Mixtral block_sparse_moe: router gate + per-expert SwiGLU.
          // Router gate: bias-free [num_local_experts, hidden] nn.Linear.
          LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].GateConv,
            BlockPrefix + 'block_sparse_moe.gate.weight',
            Config.HiddenSize, Config.NumLocalExperts);
          MarkConsumed(BlockPrefix + 'block_sparse_moe.gate.weight');
          // Per-expert SwiGLU (HF: w1=gate_proj, w3=up_proj, w2=down_proj;
          // y = w2(silu(w1 x) * w3 x)). TNNetSwiGLU = FIRSTHALF*silu(SECOND),
          // so w3 (up) -> neurons 0..I-1 and w1 (gate) -> I..2I-1.
          for d := 0 to Config.NumLocalExperts - 1 do
          begin
            TensorNameStr := BlockPrefix + 'block_sparse_moe.experts.' +
              IntToStr(d) + '.';
            LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].ExpertGateUp[d],
              TensorNameStr + 'w3.weight',
              Config.HiddenSize, Config.IntermediateSize,
              0, 2 * Config.IntermediateSize);
            MarkConsumed(TensorNameStr + 'w3.weight');
            LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].ExpertGateUp[d],
              TensorNameStr + 'w1.weight',
              Config.HiddenSize, Config.IntermediateSize,
              Config.IntermediateSize, 2 * Config.IntermediateSize);
            MarkConsumed(TensorNameStr + 'w1.weight');
            LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].ExpertDown[d],
              TensorNameStr + 'w2.weight',
              Config.IntermediateSize, Config.HiddenSize);
            MarkConsumed(TensorNameStr + 'w2.weight');
          end;
          if pQuantizeInt8 then NN.QuantizeWeightsInt8();
          continue;
        end;
        // Fused gate/up: up_proj -> neurons 0..I-1 (SwiGLU's linear half),
        // gate_proj -> neurons I..2I-1 (SwiGLU's Swish-gated half).
        if Config.FusedQKVGateUp then
        begin
          // Phi-3 fused mlp.gate_up_proj packs gate (rows 0..I-1) THEN up
          // (rows I..2I-1) - HF chunks the output in that order - so the
          // UP half lands in neurons 0..I-1 and the GATE half in
          // neurons I..2I-1 (TNNetSwiGLU computes FIRSTHALF *
          // silu(SECONDHALF)).
          FusedName := BlockPrefix + 'mlp.gate_up_proj.weight';
          LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].GateUp, FusedName,
            Config.HiddenSize, Config.IntermediateSize,
            0, 2 * Config.IntermediateSize, 0, '', 1.0, 0,
            Config.IntermediateSize, 2 * Config.IntermediateSize);
          LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].GateUp, FusedName,
            Config.HiddenSize, Config.IntermediateSize,
            Config.IntermediateSize, 2 * Config.IntermediateSize,
            0, '', 1.0, 0, 0, 2 * Config.IntermediateSize);
          MarkConsumed(FusedName);
        end
        else
        begin
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
        end;
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Down,
          BlockPrefix + 'mlp.down_proj.weight',
          Config.IntermediateSize, Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'mlp.down_proj.weight');
        // Re-quantize the block just refilled with checkpoint weights.
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      LoadLlamaRMSNormWeights(Reader, FinalNorm,
        Config.Prefix + 'norm.weight', Config.HiddenSize, NormGainOffset);
      MarkConsumed(Config.Prefix + 'norm.weight');

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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

function BuildLlamaFromSafeTensorsWithConfig(const FileName: string;
  var Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
begin
  Result := BuildLlamaFromTensorReaderWithConfig(
    CreatePretrainedTensorReader(FileName), FileName, Config, pSeqLen,
    pInferenceOnly, pQuantizeInt8);
end;

function BuildLlamaFromGGUFEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
var
  Reader: TNNetGGUFReader;
  Arch, ScalingType, GGUFPrefix, HFPrefix: string;
  BlockCnt, HeadDim: integer;
  OK: boolean;

  function RequiredMetaInt(const Key: string): integer;
  var
    V: Int64;
  begin
    V := Reader.GetMetaInt(Key, -1);
    if V <= 0 then
      ImportError('Llama GGUF import: metadata key "' + Key +
        '" is missing or not a positive integer in ' + FileName + '.');
    Result := integer(V);
  end;

  // Renames a ggml tensor name to its HF equivalent; Optional=true makes
  // a missing tensor a no-op (the core builder reports missing REQUIRED
  // tensors itself, with the HF name).
  procedure Rename(const GGUFName, HFName: string; Optional: boolean);
  begin
    if Reader.HasTensor(GGUFName) then
      Reader.RenameTensor(GGUFName, HFName)
    else if not Optional then
      ImportError('Llama GGUF import: missing tensor "' + GGUFName +
        '" in ' + FileName + '.');
  end;

begin
  Reader := TNNetGGUFReader.Create(FileName);
  OK := false;
  try
    Arch := Reader.GetMetaString('general.architecture', '');
    if Arch <> 'llama' then
      ImportError('Llama GGUF import: general.architecture is "' + Arch +
        '" - only "llama" is supported here: ' + FileName);
    Config := Default(TLlamaConfig);
    Config.ModelType := 'llama';
    Config.HiddenSize := RequiredMetaInt('llama.embedding_length');
    Config.IntermediateSize := RequiredMetaInt('llama.feed_forward_length');
    Config.NumLayers := RequiredMetaInt('llama.block_count');
    Config.NumHeads := RequiredMetaInt('llama.attention.head_count');
    Config.NumKVHeads := integer(Reader.GetMetaInt(
      'llama.attention.head_count_kv', Config.NumHeads));
    Config.MaxPositions := RequiredMetaInt('llama.context_length');
    Config.RmsNormEps := Reader.GetMetaFloat(
      'llama.attention.layer_norm_rms_epsilon', 1e-5);
    Config.RopeTheta := Reader.GetMetaFloat('llama.rope.freq_base', 10000.0);
    Config.HeadDim := integer(Reader.GetMetaInt(
      'llama.attention.key_length', 0));
    Config.PartialRotaryFactor := 1.0;
    Config.RopeScaling := DefaultRoPEScaling();
    ScalingType := Reader.GetMetaString('llama.rope.scaling.type', 'none');
    if ScalingType = 'linear' then
    begin
      Config.RopeScaling.Mode := rsmPositionInterpolation;
      Config.RopeScaling.Factor := Reader.GetMetaFloat(
        'llama.rope.scaling.factor', 1.0);
      if Config.RopeScaling.Factor < 1 then
        ImportError('Llama GGUF import: llama.rope.scaling.factor must ' +
          'be >= 1, got ' + FloatToStr(Config.RopeScaling.Factor) + '.');
      if Config.RopeScaling.Factor = 1.0 then
        Config.RopeScaling.Mode := rsmNone;
    end
    else if ScalingType <> 'none' then
      ImportError('Llama GGUF import: llama.rope.scaling.type "' +
        ScalingType + '" is not supported (supported: none, linear): ' +
        FileName);
    // vocab_size: explicit key, else the embedding rows (the GGUF dims are
    // served reversed, so dim 0 IS the vocab axis), else the token list.
    Config.VocabSize := integer(Reader.GetMetaInt('llama.vocab_size', 0));
    if Config.VocabSize <= 0 then
    begin
      if Reader.HasTensor('token_embd.weight') and
         (Reader.DimCount('token_embd.weight') = 2) then
        Config.VocabSize := integer(Reader.DimSize('token_embd.weight', 0))
      else
        Config.VocabSize := integer(Reader.GetMetaArrayCount(
          'tokenizer.ggml.tokens'));
    end;
    if Config.VocabSize <= 0 then
      ImportError('Llama GGUF import: cannot determine the vocab size ' +
        '(no llama.vocab_size, no 2-D token_embd.weight, no ' +
        'tokenizer.ggml.tokens) in ' + FileName + '.');
    // llama.cpp omits "output.weight" when the LM head ties to the
    // embedding table.
    Config.TieWordEmbeddings := not Reader.HasTensor('output.weight');

    // ---- ggml -> HF tensor names (the names the core builder expects) ----
    Rename('token_embd.weight', 'model.embed_tokens.weight', false);
    Rename('output_norm.weight', 'model.norm.weight', false);
    Rename('output.weight', 'lm_head.weight', true);
    for BlockCnt := 0 to Config.NumLayers - 1 do
    begin
      GGUFPrefix := 'blk.' + IntToStr(BlockCnt) + '.';
      HFPrefix := 'model.layers.' + IntToStr(BlockCnt) + '.';
      Rename(GGUFPrefix + 'attn_norm.weight',
        HFPrefix + 'input_layernorm.weight', false);
      Rename(GGUFPrefix + 'attn_q.weight',
        HFPrefix + 'self_attn.q_proj.weight', false);
      Rename(GGUFPrefix + 'attn_k.weight',
        HFPrefix + 'self_attn.k_proj.weight', false);
      Rename(GGUFPrefix + 'attn_v.weight',
        HFPrefix + 'self_attn.v_proj.weight', false);
      Rename(GGUFPrefix + 'attn_output.weight',
        HFPrefix + 'self_attn.o_proj.weight', false);
      Rename(GGUFPrefix + 'ffn_norm.weight',
        HFPrefix + 'post_attention_layernorm.weight', false);
      Rename(GGUFPrefix + 'ffn_gate.weight',
        HFPrefix + 'mlp.gate_proj.weight', false);
      Rename(GGUFPrefix + 'ffn_up.weight',
        HFPrefix + 'mlp.up_proj.weight', false);
      Rename(GGUFPrefix + 'ffn_down.weight',
        HFPrefix + 'mlp.down_proj.weight', false);
    end;
    // llama.cpp's convert script permutes the q/k rows from HF rotate_half
    // into the interleaved rotary layout; register the inverse so the core
    // builder (which applies the rotate_half->interleaved permutation
    // itself) receives HF-order rows.
    if Config.HeadDim > 0 then
      HeadDim := Config.HeadDim
    else
    begin
      if (Config.HiddenSize mod Config.NumHeads) <> 0 then
        ImportError('Llama GGUF import: llama.embedding_length=' +
          IntToStr(Config.HiddenSize) + ' is not divisible by ' +
          'llama.attention.head_count=' + IntToStr(Config.NumHeads) + '.');
      HeadDim := Config.HiddenSize div Config.NumHeads;
    end;
    for BlockCnt := 0 to Config.NumLayers - 1 do
    begin
      HFPrefix := 'model.layers.' + IntToStr(BlockCnt) + '.';
      Reader.RegisterRowDeinterleave(
        HFPrefix + 'self_attn.q_proj.weight', HeadDim);
      Reader.RegisterRowDeinterleave(
        HFPrefix + 'self_attn.k_proj.weight', HeadDim);
    end;
    OK := true;
  finally
    // The core builder takes ownership of the reader only when reached.
    if not OK then Reader.Free;
  end;
  Result := BuildLlamaFromTensorReaderWithConfig(Reader, FileName, Config,
    pSeqLen, pInferenceOnly, pQuantizeInt8);
end;

function BuildLlamaFromGGUF(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildLlamaFromGGUFEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, pQuantizeInt8);
end;

function BuildLlamaFromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = '';
  pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadLlamaConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildLlamaFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, pQuantizeInt8);
end;

function BuildLlamaFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildLlamaFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
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
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
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
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

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
          EnsureWritableImportWeights(LMHead);
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
        // Re-quantize the block just refilled with checkpoint weights.
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      LoadLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'ln_f.weight', Config.Prefix + 'ln_f.bias',
        Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'ln_f.weight');
      MarkConsumed(Config.Prefix + 'ln_f.bias');

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadGPTNeoConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildGPTNeoFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, pQuantizeInt8);
end;

function BuildGPTNeoFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TGPTNeoConfig;
begin
  Result := BuildGPTNeoFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
end;

// =========================== GPT-NEOX IMPORT ===============================

function ReadGPTNeoXConfigFromJSONFile(const FileName: string): TGPTNeoXConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, HiddenAct: string;

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
    Result.RopeScaling := ReadRoPEScalingFromJSONObject(Obj,
      Result.MaxPositions, 'GPT-NeoX import');
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
  if Config.RopeScaling.Mode <> rsmNone then
    Result := Result + ', rope_scaling=' +
      RoPEScalingToString(Config.RopeScaling);
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
  EnsureWritableImportWeights(Layer);
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
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
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
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
            CreateRoPEFromScaling(Config.RopeTheta, Config.RopeScaling), RotSlice);
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
            CreateRoPEFromScaling(Config.RopeTheta, Config.RopeScaling), RotSlice);
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
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

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
          EnsureWritableImportWeights(LMHead);
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
        // Re-quantize the block just refilled with checkpoint weights.
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      LoadLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'final_layer_norm.weight',
        Config.Prefix + 'final_layer_norm.bias', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'final_layer_norm.weight');
      MarkConsumed(Config.Prefix + 'final_layer_norm.bias');

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadGPTNeoXConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildGPTNeoXFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, pQuantizeInt8);
end;

function BuildGPTNeoXFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TGPTNeoXConfig;
begin
  Result := BuildGPTNeoXFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
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
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
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
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

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
        // Re-quantize the block just refilled with checkpoint weights.
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      LoadLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'ln_f.weight',
        Config.Prefix + 'ln_f.bias', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'ln_f.weight');
      MarkConsumed(Config.Prefix + 'ln_f.bias');

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadGPTJConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildGPTJFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, pQuantizeInt8);
end;

function BuildGPTJFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TGPTJConfig;
begin
  Result := BuildGPTJFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
end;

// ============================ COHERE IMPORT ================================
//
// BuildCohereFromSafeTensors rebuilds Cohere's Command-R / Aya decoder
// (model_type "cohere") and the Command-R7B variant (model_type "cohere2").
// This is the leading open MULTILINGUAL instruct family (C4AI Command-R,
// Aya-Expanse-8B, Command-R7B). It is NOT a Llama clone - the genuinely new
// pieces, all verified against HF modeling_cohere / modeling_cohere2:
//
//  (a) PARALLEL residual: ONE input LayerNorm feeds BOTH the attention and
//      the MLP branch, and a single fused 3-input sum closes the block:
//         x := x + Attn(LN(x)) + MLP(LN(x))
//      exactly the GPT-J/GPT-NeoX parallel-residual path reused here.
//  (b) Cohere's norm is a TRUE LayerNorm (mean-subtracting), bias-free and
//      weight-only (CohereLayerNorm: (x-mean)/sqrt(var+eps)*weight, no beta).
//      -> TNNetTokenLayerNorm with the beta neuron left at zero (only the
//      gamma vector is loaded). This is distinct from the Llama families'
//      TNNetTokenRMSNorm (no mean subtraction).
//  (c) a scalar logit_scale multiplies the final logits before softmax
//      (HF: logits = (h . embed^T) * logit_scale). The embeddings are tied,
//      so logit_scale is folded into every LM-head weight row at load (the
//      same idiom as Gemma's query_pre_attn_scalar / embed-scale folding) -
//      the head alone reproduces the scaled logits.
//  (d) tied input/output embeddings, NO attention/MLP biases (attention_bias
//      defaults false; the MLP and o_proj are plain bias-free nn.Linear).
//  (e) RoPE is FULL-head and uses the GPT-J INTERLEAVED pair layout
//      (rotate_half = stack(-x[1::2], x[0::2]); repeat_interleave(freqs,2)),
//      which is TNNetRotaryEmbedding's NATIVE (even/odd) layout - so the
//      q/k projection rows load STRAIGHT (RotaryHeadDim=0, NO rotate_half
//      row permutation, opposite of the Llama/NeoX half-split path).
//  (f) GQA (num_key_value_heads <= num_attention_heads): each KV head is
//      projected and rotated once, then shared across its query-head group.
//  (g) cohere ONLY (Config.UseQKNorm): per-head q_norm/k_norm are
//      CohereLayerNorm over head_dim with PER-HEAD DISTINCT gains (weight
//      shape [num_heads, head_dim] / [num_kv_heads, head_dim]), applied
//      AFTER the projection and BEFORE RoPE (HF: "main diff from Llama").
//      Mean-subtracting -> a per-head TNNetTokenLayerNorm (NOT TokenRMSNorm).
//  (h) cohere2 ONLY: an alternating sliding-window / global attention
//      pattern (Config.SlidingWindowPattern, every Nth layer global, the
//      rest local) AND - crucially - RoPE is applied ONLY on the SLIDING
//      layers; the GLOBAL layers use NoPE (no positional embedding at all,
//      HF: `if self.sliding_window is not None: apply_rotary_pos_emb`).
//      cohere2 has NO qk_norm.
//
// The MLP is the standard SwiGLU (down(silu(gate(x)) * up(x))). TNNetSwiGLU
// computes FIRSTHALF * Swish(SECONDHALF), so the fused gate/up projection
// holds up_proj in neurons 0..I-1 and gate_proj in I..2I-1.
//
// The net takes a (SeqLen,1,1) volume of token ids and outputs
// (SeqLen,1,vocab) logit_scale-scaled logits, matching the other decoder
// importers' causal-LM contract. .bin checkpoints dispatch through the same
// TNNetTorchBinReader path (CreatePretrainedTensorReader) and multi-shard
// .index.json is handled by the reader, like every importer in this unit.
// Coded by Claude (AI).

function ReadCohereConfigFromJSONFile(const FileName: string): TCohereConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, HiddenAct: string;
  Field: TJSONData;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('Cohere import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('Cohere import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

begin
  if not FileExists(FileName) then
    ImportError('Cohere import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('Cohere import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('Cohere import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'cohere');
    if (ModelType <> 'cohere') and (ModelType <> 'cohere2') then
      ImportError('Cohere import: config model_type is "' + ModelType +
        '" - expected "cohere" or "cohere2".');
    Result.ModelType := ModelType;
    Result.HiddenSize := RequiredInt('hidden_size');
    Result.IntermediateSize := RequiredInt('intermediate_size');
    Result.NumLayers := RequiredInt('num_hidden_layers');
    Result.NumHeads := RequiredInt('num_attention_heads');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.MaxPositions := RequiredInt('max_position_embeddings');
    // num_key_value_heads defaults to num_attention_heads (no GQA).
    Field := Obj.Find('num_key_value_heads');
    if (Field = nil) or Field.IsNull then
      Result.NumKVHeads := Result.NumHeads
    else
      Result.NumKVHeads := RequiredInt('num_key_value_heads');
    // head_dim is implicit in every published Cohere config.
    Field := Obj.Find('head_dim');
    if (Field = nil) or Field.IsNull then
      Result.HeadDim := 0
    else
      Result.HeadDim := RequiredInt('head_dim');
    Result.LayerNormEps := Obj.Get('layer_norm_eps', 1.0e-5);
    Result.RopeTheta := Obj.Get('rope_theta', 10000.0);
    // logit_scale: CohereConfig default 0.0625 (Command-R) - always present
    // in the published configs, but keep the documented default.
    Result.LogitScale := Obj.Get('logit_scale', 0.0625);
    // tie_word_embeddings defaults TRUE for Cohere (the published configs
    // tie; there is no separate lm_head tensor).
    Result.TieWordEmbeddings := Obj.Get('tie_word_embeddings', True);
    // attention_bias / MLP bias are false in every Cohere config; a true
    // value would mean tensors this importer does not wire.
    if Obj.Get('attention_bias', False) then
      ImportError('Cohere import: attention_bias=true is not supported ' +
        '(every published Cohere config is bias-free).');
    HiddenAct := Obj.Get('hidden_act', 'silu');
    if HiddenAct <> 'silu' then
      ImportError('Cohere import: hidden_act "' + HiddenAct +
        '" is not supported - Cohere uses "silu" (SwiGLU).');
    // use_qk_norm: cohere only (CohereConfig default false; Aya-Expanse and
    // some Command-R checkpoints enable it). cohere2 has no qk_norm.
    Result.UseQKNorm := Obj.Get('use_qk_norm', False);
    if ModelType = 'cohere2' then
    begin
      if Result.UseQKNorm then
        ImportError('Cohere import: cohere2 does not define use_qk_norm.');
      // sliding_window + sliding_window_pattern: every Nth layer is global,
      // the rest carry the sliding window (Command-R7B: window 4096,
      // pattern 4 - 3 local : 1 global). order_of_interleaved_layers in
      // older configs maps to the same period.
      Result.SlidingWindow := Obj.Get('sliding_window', 4096);
      Result.SlidingWindowPattern := Obj.Get('sliding_window_pattern', 4);
      if Result.SlidingWindowPattern < 1 then
        ImportError('Cohere import: sliding_window_pattern must be >= 1.');
    end
    else
    begin
      Result.SlidingWindow := 0;
      Result.SlidingWindowPattern := 0;
    end;
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function CohereConfigToString(const Config: TCohereConfig): string;
begin
  Result := Config.ModelType + ' config: layers=' +
    IntToStr(Config.NumLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', kv_heads=' + IntToStr(Config.NumKVHeads) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', max_pos=' + IntToStr(Config.MaxPositions) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', ln_eps=' + FloatToStr(Config.LayerNormEps) +
    ', rope_theta=' + FloatToStr(Config.RopeTheta) +
    ', logit_scale=' + FloatToStr(Config.LogitScale) +
    ', qk_norm=' + BoolToStr(Config.UseQKNorm, true) +
    ', tied=' + BoolToStr(Config.TieWordEmbeddings, true);
  if Config.SlidingWindowPattern > 0 then
    Result := Result + ', window=' + IntToStr(Config.SlidingWindow) +
      ', pattern=' + IntToStr(Config.SlidingWindowPattern);
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

// Loads a bias-free, weight-only LayerNorm gain [d] (CohereLayerNorm) into a
// TNNetTokenLayerNorm: only the gamma neuron (Neurons[0]) is written; the
// beta neuron (Neurons[1]) is forced to zero (Cohere has no learned shift).
// Coded by Claude (AI).
procedure LoadCohereLayerNormWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName: string; d_model: integer);
var
  Tmp: TNNetVolume;
  i: integer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('Cohere import: missing tensor "' + WName + '".');
  if (Reader.DimCount(WName) <> 1) or (Reader.DimSize(WName, 0) <> d_model) then
    ImportError('Cohere import: "' + WName + '" must have shape [' +
      IntToStr(d_model) + '], got ' + Reader.ShapeAsString(WName));
  Tmp := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, Tmp);
    for i := 0 to d_model - 1 do
    begin
      Layer.Neurons[0].Weights.FData[i] := Tmp.FData[i]; // gamma
      Layer.Neurons[1].Weights.FData[i] := 0;            // beta (bias-free)
    end;
  finally
    Tmp.Free;
  end;
  Layer.FlushWeightCache();
end;

// Loads the per-head CohereLayerNorm q_norm/k_norm gains (weight shape
// [num_heads, head_dim], PER-HEAD DISTINCT) into the list of per-head
// TNNetTokenLayerNorm layers (one gamma row each, beta zero). Row h of the
// flat [num_heads*head_dim] tensor is head h's gain. Loaded STRAIGHT - the
// q/k projection rows are NOT rotate_half-permuted on the Cohere path.
// Coded by Claude (AI).
procedure LoadCohereHeadLayerNormWeights(Reader: TNNetSafeTensorsReader;
  NormLayers: array of TNNetLayer; const WName: string;
  NumHeadsExpected, HeadDim: integer);
var
  Tmp: TNNetVolume;
  h, i: integer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('Cohere import: missing tensor "' + WName + '".');
  if Reader.ElementCount(WName) <> NumHeadsExpected * HeadDim then
    ImportError('Cohere import: "' + WName + '" must have ' +
      IntToStr(NumHeadsExpected * HeadDim) + ' elements ([' +
      IntToStr(NumHeadsExpected) + ', ' + IntToStr(HeadDim) + ']), got ' +
      Reader.ShapeAsString(WName));
  Tmp := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, Tmp);
    for h := 0 to NumHeadsExpected - 1 do
    begin
      for i := 0 to HeadDim - 1 do
      begin
        NormLayers[h].Neurons[0].Weights.FData[i] :=
          Tmp.FData[h * HeadDim + i];           // gamma row h
        NormLayers[h].Neurons[1].Weights.FData[i] := 0; // beta
      end;
      NormLayers[h].FlushWeightCache();
    end;
  finally
    Tmp.Free;
  end;
end;

type
  TCohereBlockLayers = record
    LN1, QProj, KProj, VProj, OProj, GateUp, Down: TNNetLayer;
    QNorms, KNorms: array of TNNetLayer; // per-head q/k LayerNorm (cohere)
  end;

function BuildCohereFromSafeTensorsWithConfig(const FileName: string;
  var Config: TCohereConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TCohereBlockLayers;
  EmbeddingLayer, FinalLN, LMHead: TNNetLayer;
  BranchInput, SharedLN, AttnOut, MlpOut: TNNetLayer;
  QSlice, KSlice, VSlice, HeadPack: TNNetLayer;
  KRotated, VSlices: array of TNNetLayer;
  HeadOutputs: array of TNNetLayer;
  SliceChannels: array of integer;
  BlockCnt, SeqLen, HeadCnt, KVHeadCnt, KVGroup, GroupSize: integer;
  HeadDim, QWidth, KVWidth, i, j, d: integer;
  LayerIsLocal: boolean;
  LayerWindow: integer;
  Tmp: TNNetVolume;
  BlockPrefix, AttnPrefix, LMHeadName, TensorNameStr: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

  // Builds one q or k head: optional per-head LayerNorm (cohere qk_norm),
  // then RoPE iff this layer is rotary (full head, interleaved native
  // layout). Returns the rotated/normed head slice layer.
  function BuildQKHead(ProjLayer: TNNetLayer; HeadIdx: integer;
    IsQuery, ApplyRoPE: boolean): TNNetLayer;
  var
    s: TNNetLayer;
    cc: integer;
  begin
    for cc := 0 to HeadDim - 1 do
      SliceChannels[cc] := HeadIdx * HeadDim + cc;
    s := NN.AddLayerAfter(TNNetSplitChannels.Create(SliceChannels), ProjLayer);
    if Config.UseQKNorm then
    begin
      s := NN.AddLayerAfter(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps), s);
      if IsQuery then Blocks[BlockCnt].QNorms[HeadIdx] := s
      else Blocks[BlockCnt].KNorms[HeadIdx] := s;
    end;
    if ApplyRoPE then
      s := NN.AddLayerAfter(
        TNNetRotaryEmbedding.Create(Config.RopeTheta), s);
    Result := s;
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
        ImportError('Cohere import: num_attention_heads must be >= 1.');
      if Config.NumKVHeads < 1 then
        ImportError('Cohere import: num_key_value_heads must be >= 1.');
      if (Config.NumHeads mod Config.NumKVHeads) <> 0 then
        ImportError('Cohere import: num_attention_heads=' +
          IntToStr(Config.NumHeads) + ' is not a multiple of ' +
          'num_key_value_heads=' + IntToStr(Config.NumKVHeads) + '.');
      if Config.HeadDim > 0 then HeadDim := Config.HeadDim
      else
      begin
        if (Config.HiddenSize mod Config.NumHeads) <> 0 then
          ImportError('Cohere import: hidden_size=' +
            IntToStr(Config.HiddenSize) + ' is not divisible by ' +
            'num_attention_heads=' + IntToStr(Config.NumHeads) + '.');
        HeadDim := Config.HiddenSize div Config.NumHeads;
      end;
      if Odd(HeadDim) then
        ImportError('Cohere import: head_dim=' + IntToStr(HeadDim) +
          ' must be even (RoPE rotates channel pairs).');
      GroupSize := Config.NumHeads div Config.NumKVHeads;
      QWidth := Config.NumHeads * HeadDim;
      KVWidth := Config.NumKVHeads * HeadDim;
      if Reader.HasTensor('model.embed_tokens.weight') then
        Config.Prefix := 'model.'
      else if Reader.HasTensor('embed_tokens.weight') then
        Config.Prefix := ''
      else
        ImportError('Cohere import: neither "model.embed_tokens.weight" ' +
          'nor "embed_tokens.weight" found in ' + Reader.FileName +
          ' - not a Cohere checkpoint?');
      if (Reader.DimCount(Config.Prefix + 'embed_tokens.weight') <> 2) or
         (Reader.DimSize(Config.Prefix + 'embed_tokens.weight', 0) <>
          Config.VocabSize) or
         (Reader.DimSize(Config.Prefix + 'embed_tokens.weight', 1) <>
          Config.HiddenSize) then
        ImportError('Cohere import: embed_tokens.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.HiddenSize) +
          '], got ' +
          Reader.ShapeAsString(Config.Prefix + 'embed_tokens.weight'));
      LMHeadName := 'lm_head.weight';
      if (not Config.TieWordEmbeddings) and (not Reader.HasTensor(LMHeadName)) then
        ImportError('Cohere import: tie_word_embeddings=false but ' +
          '"lm_head.weight" is missing from ' + Reader.FileName + '.');
      if pSeqLen <= 0 then SeqLen := Config.MaxPositions
      else SeqLen := pSeqLen;
      if SeqLen > Config.MaxPositions then
        ImportError('Cohere import: requested SeqLen=' + IntToStr(SeqLen) +
          ' exceeds max_position_embeddings=' +
          IntToStr(Config.MaxPositions) + '.');

      // ---------------- Architecture ----------------
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      SetLength(Blocks, Config.NumLayers);
      SetLength(HeadOutputs, Config.NumHeads);
      SetLength(KRotated, Config.NumKVHeads);
      SetLength(VSlices, Config.NumKVHeads);
      SetLength(SliceChannels, HeadDim);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // cohere2 alternating attention: every Nth layer (1-indexed) is
        // GLOBAL, the rest carry the sliding window. AND RoPE is applied
        // only on the SLIDING layers - global layers are NoPE (HF
        // Cohere2Attention: rotary only when self.sliding_window is not
        // None). cohere (pattern 0) is full attention + RoPE everywhere.
        if Config.SlidingWindowPattern > 1 then
          LayerIsLocal := ((BlockCnt + 1) mod Config.SlidingWindowPattern) <> 0
        else
          LayerIsLocal := false;
        if (Config.SlidingWindowPattern > 0) then
        begin
          // cohere2: window on local layers, NoPE on global layers.
          if LayerIsLocal then LayerWindow := Config.SlidingWindow
          else LayerWindow := 0;
        end
        else
          LayerWindow := 0; // cohere: full attention everywhere
        // SHARED-LN PARALLEL residual: one input_layernorm feeds BOTH
        // branches; one fused 3-input sum closes the block.
        BranchInput := NN.GetLastLayer();
        SharedLN := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        Blocks[BlockCnt].LN1 := SharedLN;
        // bias-free q/k/v projections (GQA: kv width may be narrower).
        Blocks[BlockCnt].QProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(QWidth), SharedLN);
        Blocks[BlockCnt].KProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(KVWidth), SharedLN);
        Blocks[BlockCnt].VProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(KVWidth), SharedLN);
        if Config.UseQKNorm then
        begin
          SetLength(Blocks[BlockCnt].QNorms, Config.NumHeads);
          SetLength(Blocks[BlockCnt].KNorms, Config.NumKVHeads);
        end;
        // RoPE iff this layer is rotary: cohere = always, cohere2 = only
        // the sliding layers (global layers are NoPE).
        // Build each KV head once (rotated/normed), shared across its group.
        for KVHeadCnt := 0 to Config.NumKVHeads - 1 do
        begin
          KRotated[KVHeadCnt] := BuildQKHead(Blocks[BlockCnt].KProj,
            KVHeadCnt, {IsQuery=}false,
            {ApplyRoPE=}(Config.SlidingWindowPattern = 0) or LayerIsLocal);
          for d := 0 to HeadDim - 1 do
            SliceChannels[d] := KVHeadCnt * HeadDim + d;
          VSlices[KVHeadCnt] := NN.AddLayerAfter(
            TNNetSplitChannels.Create(SliceChannels), Blocks[BlockCnt].VProj);
        end;
        for HeadCnt := 0 to Config.NumHeads - 1 do
        begin
          KVGroup := HeadCnt div GroupSize;
          QSlice := BuildQKHead(Blocks[BlockCnt].QProj, HeadCnt,
            {IsQuery=}true,
            {ApplyRoPE=}(Config.SlidingWindowPattern = 0) or LayerIsLocal);
          // Pack [Q_h | K_group | V_group] for SDPA (standard
          // 1/sqrt(head_dim) scaling). LayerWindow>0 bands the causal mask.
          HeadPack := NN.AddLayer( TNNetDeepConcat.Create(
            [QSlice, KRotated[KVGroup], VSlices[KVGroup]]) );
          HeadOutputs[HeadCnt] := NN.AddLayerAfter(
            TNNetScaledDotProductAttention.Create(HeadDim, {CausalMask=}true,
              {pWindow=}LayerWindow), HeadPack);
        end;
        NN.AddLayer( TNNetDeepConcat.Create(HeadOutputs) );
        Blocks[BlockCnt].OProj := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        AttnOut := NN.GetLastLayer();
        // MLP branch reads the SAME shared LN (parallel residual).
        Blocks[BlockCnt].GateUp := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(2 * Config.IntermediateSize),
          SharedLN);
        NN.AddLayer( TNNetSwiGLU.Create() );
        Blocks[BlockCnt].Down := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        MlpOut := NN.GetLastLayer();
        NN.AddLayer( TNNetSum.Create([BranchInput, AttnOut, MlpOut]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        Reader.LoadTensorFlat(Config.Prefix + 'embed_tokens.weight', Tmp);
        if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('Cohere import: embed_tokens.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table size ' +
            IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
        EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
        EmbeddingLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'embed_tokens.weight');
        // LM head: logits = (h . embed^T) * logit_scale. With tied
        // embeddings the scale is folded into every head weight row (the
        // head alone reproduces the scaled logits, bias-free). An untied
        // lm_head loads via LoadLlamaLinearWeights with the same Scale fold.
        if Config.TieWordEmbeddings then
        begin
          EnsureWritableImportWeights(LMHead);
          for j := 0 to Config.VocabSize - 1 do
          begin
            for i := 0 to Config.HiddenSize - 1 do
              LMHead.Neurons[j].Weights.FData[i] :=
                Config.LogitScale * Tmp.FData[j * Config.HiddenSize + i];
            LMHead.Neurons[j].BiasWeight := 0;
          end;
          LMHead.FlushWeightCache();
          if Reader.HasTensor(LMHeadName) then MarkConsumed(LMHeadName);
        end
        else
        begin
          LoadLlamaLinearWeights(Reader, LMHead, LMHeadName,
            Config.HiddenSize, Config.VocabSize, 0, -1, 0, '',
            Config.LogitScale);
          MarkConsumed(LMHeadName);
        end;
      finally
        Tmp.Free;
      end;
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        BlockPrefix := Config.Prefix + 'layers.' + IntToStr(BlockCnt) + '.';
        AttnPrefix := BlockPrefix + 'self_attn.';
        LoadCohereLayerNormWeights(Reader, Blocks[BlockCnt].LN1,
          BlockPrefix + 'input_layernorm.weight', Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'input_layernorm.weight');
        // bias-free q/k/v, rows loaded STRAIGHT (RotaryHeadDim=0: Cohere's
        // interleaved RoPE pairing is already the layer's native layout).
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].QProj,
          AttnPrefix + 'q_proj.weight', Config.HiddenSize, QWidth);
        MarkConsumed(AttnPrefix + 'q_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].KProj,
          AttnPrefix + 'k_proj.weight', Config.HiddenSize, KVWidth);
        MarkConsumed(AttnPrefix + 'k_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].VProj,
          AttnPrefix + 'v_proj.weight', Config.HiddenSize, KVWidth);
        MarkConsumed(AttnPrefix + 'v_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].OProj,
          AttnPrefix + 'o_proj.weight', QWidth, Config.HiddenSize);
        MarkConsumed(AttnPrefix + 'o_proj.weight');
        if Config.UseQKNorm then
        begin
          LoadCohereHeadLayerNormWeights(Reader, Blocks[BlockCnt].QNorms,
            AttnPrefix + 'q_norm.weight', Config.NumHeads, HeadDim);
          MarkConsumed(AttnPrefix + 'q_norm.weight');
          LoadCohereHeadLayerNormWeights(Reader, Blocks[BlockCnt].KNorms,
            AttnPrefix + 'k_norm.weight', Config.NumKVHeads, HeadDim);
          MarkConsumed(AttnPrefix + 'k_norm.weight');
        end;
        // SwiGLU: TNNetSwiGLU = FIRSTHALF * Swish(SECONDHALF); Cohere's MLP
        // is down(silu(gate) * up), so up_proj -> neurons 0..I-1 and
        // gate_proj -> I..2I-1.
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].GateUp,
          BlockPrefix + 'mlp.up_proj.weight', Config.HiddenSize,
          Config.IntermediateSize, 0, 2 * Config.IntermediateSize);
        MarkConsumed(BlockPrefix + 'mlp.up_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].GateUp,
          BlockPrefix + 'mlp.gate_proj.weight', Config.HiddenSize,
          Config.IntermediateSize, Config.IntermediateSize,
          2 * Config.IntermediateSize);
        MarkConsumed(BlockPrefix + 'mlp.gate_proj.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Down,
          BlockPrefix + 'mlp.down_proj.weight', Config.IntermediateSize,
          Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'mlp.down_proj.weight');
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      LoadCohereLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'norm.weight', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'norm.weight');

      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      // ---------------- Unexpected-tensor check ----------------
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        // rotary_emb.inv_freq: the persisted RoPE buffer (structural).
        if Pos('.rotary_emb.inv_freq', TensorNameStr) > 0 then continue;
        ImportError('Cohere import: unexpected tensor "' + TensorNameStr +
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
    NN.Free;
    Consumed.Free;
    Reader.Free;
  end;
end;

function BuildCohereFromSafeTensorsEx(const FileName: string;
  out Config: TCohereConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadCohereConfigFromJSONFile(ConfigPath);
  Result := BuildCohereFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, pQuantizeInt8);
end;

function BuildCohereFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TCohereConfig;
begin
  Result := BuildCohereFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
end;

// ============================ PHI IMPORT ===================================

function ReadPhiConfigFromJSONFile(const FileName: string): TPhiConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, HiddenAct: string;
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
    Result.RopeScaling := ReadRoPEScalingFromJSONObject(Obj,
      Result.MaxPositions, 'Phi import');
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
  if Config.RopeScaling.Mode <> rsmNone then
    Result := Result + ', rope_scaling=' +
      RoPEScalingToString(Config.RopeScaling);
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
  EnsureWritableImportWeights(Layer);
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
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
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
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
            CreateRoPEFromScaling(Config.RopeTheta, Config.RopeScaling), RotSlice);
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
            CreateRoPEFromScaling(Config.RopeTheta, Config.RopeScaling), RotSlice);
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
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

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
        // Re-quantize the block just refilled with checkpoint weights.
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      LoadLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'final_layernorm.weight',
        Config.Prefix + 'final_layernorm.bias', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'final_layernorm.weight');
      MarkConsumed(Config.Prefix + 'final_layernorm.bias');

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadPhiConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildPhiFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, pQuantizeInt8);
end;

function BuildPhiFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TPhiConfig;
begin
  Result := BuildPhiFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
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
  pSeqClsHead: boolean = false; pQuantizeInt8: boolean = false): TNNet;
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
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

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
        // Re-quantize the block just refilled with checkpoint weights.
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadBertConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildBertFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, pIncludePooler, {pSeqClsHead=}false, pQuantizeInt8);
end;

function BuildBertFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pIncludePooler: boolean = false; pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TBertConfig;
begin
  Result := BuildBertFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, pIncludePooler, '', pQuantizeInt8);
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
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
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
    pInferenceOnly, {pIncludePooler=}false, {pSeqClsHead=}true,
    pQuantizeInt8);
end;

function BuildBertForSequenceClassificationFromSafeTensors(
  const FileName: string; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TBertConfig;
begin
  Result := BuildBertForSequenceClassificationFromSafeTensorsEx(FileName,
    IgnoredConfig, nil, pSeqLen, pInferenceOnly, '', pQuantizeInt8);
end;

function BuildGPT2ForSequenceClassificationFromSafeTensorsEx(
  const FileName: string; out Config: TGPT2Config; Id2Label: TStringList;
  pSeqLen: integer = 0; pNumHeads: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
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
    pNumHeads, pInferenceOnly, {pSeqClsHead=}true, {pExactGelu=}false,
    pQuantizeInt8);
end;

function BuildGPT2ForSequenceClassificationFromSafeTensors(
  const FileName: string; pSeqLen: integer = 0; pNumHeads: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TGPT2Config;
begin
  Result := BuildGPT2ForSequenceClassificationFromSafeTensorsEx(FileName,
    IgnoredConfig, nil, pSeqLen, pNumHeads, pInferenceOnly, '',
    pQuantizeInt8);
end;

// ============= MISTRAL / QWEN2 / QWEN3 / GEMMA / DISPATCH ==================

// Shared wrapper: builds via the Llama path and asserts the config's
// model_type is the expected family.
function BuildLlamaFamilyFromSafeTensors(const FileName: string;
  const ExpectedModelType: string; out Config: TLlamaConfig;
  pSeqLen: integer; pInferenceOnly: boolean;
  const ConfigFileName: string; pQuantizeInt8: boolean = false): TNNet;
begin
  Result := BuildLlamaFromSafeTensorsEx(FileName, Config, pSeqLen,
    pInferenceOnly, ConfigFileName, pQuantizeInt8);
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
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'mistral', Config,
    pSeqLen, pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

function BuildMistralFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildMistralFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
end;

function BuildQwen2FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'qwen2', Config,
    pSeqLen, pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

function BuildQwen2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildQwen2FromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
end;

function BuildQwen3FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'qwen3', Config,
    pSeqLen, pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

function BuildQwen3FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildQwen3FromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
end;

function BuildOlmo2FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'olmo2', Config,
    pSeqLen, pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

function BuildOlmo2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildOlmo2FromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
end;

function BuildGemmaFromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'gemma', Config,
    pSeqLen, pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

function BuildGemmaFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildGemmaFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
end;

function BuildGemma2FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'gemma2', Config,
    pSeqLen, pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

function BuildGemma2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildGemma2FromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
end;

function BuildGemma3FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'gemma3_text', Config,
    pSeqLen, pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

function BuildGemma3FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildGemma3FromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
end;

function BuildPhi3FromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'phi3', Config,
    pSeqLen, pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

function BuildPhi3FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildPhi3FromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
end;

function BuildMixtralFromSafeTensorsEx(const FileName: string;
  out Config: TLlamaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
begin
  Result := BuildLlamaFamilyFromSafeTensors(FileName, 'mixtral', Config,
    pSeqLen, pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

function BuildMixtralFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TLlamaConfig;
begin
  Result := BuildMixtralFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
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
  var Blocks: TT5BlockArray; pInferenceOnly: boolean;
  pQuantizeInt8: boolean = false);
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
    if pQuantizeInt8 then NN.QuantizeWeightsInt8();
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
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false);
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
      if pQuantizeInt8 then Enc.QuantizeWeightsInt8();
      BuildT5StackBlocks(Enc, Config, Config.NumLayers, {IsDecoder=}false,
        nil, EncBlocks, pInferenceOnly, pQuantizeInt8);
      EncFinalNorm := Enc.AddLayer(
        TNNetTokenRMSNorm.Create(Config.LayerNormEps) );
      if pInferenceOnly then Enc.MakeInferenceOnly();
      if pQuantizeInt8 then Enc.QuantizeWeightsInt8();

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
      if pQuantizeInt8 then Dec.QuantizeWeightsInt8();
      BuildT5StackBlocks(Dec, Config, Config.NumDecoderLayers,
        {IsDecoder=}true, EncStates, DecBlocks, pInferenceOnly,
        pQuantizeInt8);
      DecFinalNorm := Dec.AddLayer(
        TNNetTokenRMSNorm.Create(Config.LayerNormEps) );
      LMHead := Dec.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then Dec.MakeInferenceOnly();
      if pQuantizeInt8 then Dec.QuantizeWeightsInt8();

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
          EnsureWritableImportWeights(LMHead);
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
      // Re-quantize the stack just refilled with checkpoint weights.
      if pQuantizeInt8 then Enc.QuantizeWeightsInt8();
      LoadT5Stack(Reader, DecBlocks, 'decoder.', Config,
        {IsDecoder=}true, Consumed);
      // Re-quantize the stack just refilled with checkpoint weights.
      if pQuantizeInt8 then Dec.QuantizeWeightsInt8();
      LoadLlamaRMSNormWeights(Reader, EncFinalNorm,
        'encoder.final_layer_norm.weight', Config.DModel);
      Consumed.Add('encoder.final_layer_norm.weight');
      LoadLlamaRMSNormWeights(Reader, DecFinalNorm,
        'decoder.final_layer_norm.weight', Config.DModel);
      Consumed.Add('decoder.final_layer_norm.weight');

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then Enc.QuantizeWeightsInt8();
      if pQuantizeInt8 then Dec.QuantizeWeightsInt8();
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
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false);
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadT5ConfigFromJSONFile(ConfigPath);
  BuildT5FromSafeTensorsWithConfig(FileName, Config, EncoderNet,
    DecoderNet, EncSeqLen, DecSeqLen, pInferenceOnly, pQuantizeInt8);
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

function ReadMarianConfigFromJSONFile(const FileName: string): TMarianConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, ActFn: string;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('Marian import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('Marian import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

begin
  if not FileExists(FileName) then
    ImportError('Marian import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('Marian import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('Marian import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'marian');
    if ModelType <> 'marian' then
      ImportError('Marian import: config model_type is "' + ModelType +
        '" - only "marian" is supported here (see BuildFromPretrained ' +
        'for the full dispatch).');
    Result.ModelType := ModelType;
    Result.DModel := RequiredInt('d_model');
    Result.EncoderLayers := RequiredInt('encoder_layers');
    Result.DecoderLayers := RequiredInt('decoder_layers');
    Result.EncoderHeads := RequiredInt('encoder_attention_heads');
    Result.DecoderHeads := RequiredInt('decoder_attention_heads');
    Result.EncoderFFNDim := RequiredInt('encoder_ffn_dim');
    Result.DecoderFFNDim := RequiredInt('decoder_ffn_dim');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.MaxPositionEmbeddings :=
      Obj.Get('max_position_embeddings', 1024);
    // The HF MarianConfig defaults (every opus-mt pair pins pad/start to
    // vocab_size-1 = 58100 explicitly; decoder start = pad is the Marian
    // shift_tokens_right convention).
    Result.PadTokenId := Obj.Get('pad_token_id', 58100);
    Result.DecoderStartTokenId :=
      Obj.Get('decoder_start_token_id', Result.PadTokenId);
    Result.ScaleEmbedding := Obj.Get('scale_embedding', False);
    // Only the published Marian recipes are supported: "swish"/"silu"
    // (every opus-mt pair) and "relu". HF's default "gelu" (the exact erf
    // form, unused by any released Marian) is rejected to avoid a silent
    // activation mismatch.
    ActFn := Obj.Get('activation_function', 'swish');
    if (ActFn = 'swish') or (ActFn = 'silu') then
      Result.SwishFFN := True
    else if ActFn = 'relu' then
      Result.SwishFFN := False
    else
      ImportError('Marian import: activation_function "' + ActFn +
        '" is not supported - expected "swish"/"silu" (opus-mt) or ' +
        '"relu".');
    // The untied variants need a SECOND embedding matrix this v1 does not
    // wire (share_encoder_decoder_embeddings=false ships separate
    // encoder/decoder vocabularies; tie_word_embeddings=false a separate
    // lm_head).
    if not Obj.Get('share_encoder_decoder_embeddings', True) then
      ImportError('Marian import: share_encoder_decoder_embeddings=false ' +
        'is not supported - only the shared-vocabulary opus-mt layout.');
    if not Obj.Get('tie_word_embeddings', True) then
      ImportError('Marian import: tie_word_embeddings=false is not ' +
        'supported - Marian ties the lm_head to the shared embedding.');
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function MarianConfigToString(const Config: TMarianConfig): string;
begin
  if Config.ModelType = '' then Result := 'marian'
  else Result := Config.ModelType;
  Result := Result + ' config: enc_layers=' + IntToStr(Config.EncoderLayers) +
    ', dec_layers=' + IntToStr(Config.DecoderLayers) +
    ', enc_heads=' + IntToStr(Config.EncoderHeads) +
    ', dec_heads=' + IntToStr(Config.DecoderHeads) +
    ', d_model=' + IntToStr(Config.DModel) +
    ', enc_ffn=' + IntToStr(Config.EncoderFFNDim) +
    ', dec_ffn=' + IntToStr(Config.DecoderFFNDim) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', max_pos=' + IntToStr(Config.MaxPositionEmbeddings) +
    ', pad=' + IntToStr(Config.PadTokenId) +
    ', dec_start=' + IntToStr(Config.DecoderStartTokenId) +
    ', scale_embedding=' + BoolToStr(Config.ScaleEmbedding, true);
  if Config.SwishFFN then
    Result := Result + ', ffn=swish'
  else
    Result := Result + ', ffn=relu';
end;

const
  // nn.LayerNorm's default eps - Marian never overrides it.
  MarianLayerNormEps = 1e-5;

type
  TMarianAttnLayers = record
    QProj, KProj, VProj, OProj, Norm: TNNetLayer;
  end;
  TMarianBlockLayers = record
    SelfAttn: TMarianAttnLayers;
    CrossAttn: TMarianAttnLayers; // decoder blocks only
    Fc1, Fc2, FFNNorm: TNNetLayer;
  end;
  TMarianBlockArray = array of TMarianBlockLayers;

// Grows one Marian stack (encoder or decoder) onto NN after the embedding +
// positional layers. POST-norm wiring: every sublayer reads the RAW stream,
// adds its residual, THEN applies the biased LayerNorm (the opposite of
// T5's pre-norm); there is NO final stack norm. IsDecoder adds the causal
// mask on self-attention plus the cross-attention sub-block reading
// Keys|Values from EncStates (the decoder net's second TNNetInput). Layer
// handles for the weight loader are returned in Blocks.
procedure BuildMarianStackBlocks(NN: TNNet; const Config: TMarianConfig;
  NumBlocks, NumHeads, FFNDim: integer; IsDecoder: boolean;
  EncStates: TNNetLayer; var Blocks: TMarianBlockArray;
  pInferenceOnly: boolean; pQuantizeInt8: boolean = false);
var
  HeadDim, BlockCnt, HeadCnt, d: integer;
  BranchInput: TNNetLayer;
  QSlice, KSlice, VSlice, HeadPack, KVPack: TNNetLayer;
  Heads: array of TNNetLayer;
  SliceChannels: array of integer;
begin
  HeadDim := Config.DModel div NumHeads;
  SetLength(Blocks, NumBlocks);
  SetLength(SliceChannels, HeadDim);
  SetLength(Heads, NumHeads);
  for BlockCnt := 0 to NumBlocks - 1 do
  begin
    // ---- self-attention sub-block (residual add, THEN LayerNorm) ----
    // Standard softmax(QK^T/sqrt(head_dim)) per head - exactly what
    // TNNetScaledDotProductAttention computes, so the q/k/v weights load
    // unmodified (no T5-style scale compensation).
    BranchInput := NN.GetLastLayer();
    Blocks[BlockCnt].SelfAttn.QProj := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(Config.DModel), BranchInput);
    Blocks[BlockCnt].SelfAttn.KProj := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(Config.DModel), BranchInput);
    Blocks[BlockCnt].SelfAttn.VProj := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(Config.DModel), BranchInput);
    for HeadCnt := 0 to NumHeads - 1 do
    begin
      for d := 0 to HeadDim - 1 do
        SliceChannels[d] := HeadCnt * HeadDim + d;
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
      Heads[HeadCnt] := NN.AddLayerAfter(
        TNNetScaledDotProductAttention.Create(HeadDim,
          {CausalMask=}IsDecoder), HeadPack);
    end;
    NN.AddLayer( TNNetDeepConcat.Create(Heads) );
    Blocks[BlockCnt].SelfAttn.OProj := NN.AddLayer(
      TNNetPointwiseConvLinear.Create(Config.DModel) );
    NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
    Blocks[BlockCnt].SelfAttn.Norm := NN.AddLayer(
      TNNetTokenLayerNorm.Create(MarianLayerNormEps) );

    // ---- cross-attention sub-block (decoder only) ----
    // Queries from the post-normed decoder stream; Keys|Values projected
    // from EncStates - the two grids may have DIFFERENT lengths, so the
    // per-head leaf is TNNetCrossAttention (rectangular DecSeqLen x
    // EncSeqLen scores; a Q|K|V DeepConcat would be illegal across unequal
    // SizeX). Residual add THEN encoder_attn_layer_norm.
    if IsDecoder then
    begin
      BranchInput := NN.GetLastLayer();
      Blocks[BlockCnt].CrossAttn.QProj := NN.AddLayerAfter(
        TNNetPointwiseConvLinear.Create(Config.DModel), BranchInput);
      Blocks[BlockCnt].CrossAttn.KProj := NN.AddLayerAfter(
        TNNetPointwiseConvLinear.Create(Config.DModel), EncStates);
      Blocks[BlockCnt].CrossAttn.VProj := NN.AddLayerAfter(
        TNNetPointwiseConvLinear.Create(Config.DModel), EncStates);
      for HeadCnt := 0 to NumHeads - 1 do
      begin
        for d := 0 to HeadDim - 1 do
          SliceChannels[d] := HeadCnt * HeadDim + d;
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
        Heads[HeadCnt] := NN.AddLayerAfter(
          TNNetCrossAttention.Create(HeadDim, {CausalMask=}false,
            KVPack), QSlice);
      end;
      NN.AddLayer( TNNetDeepConcat.Create(Heads) );
      Blocks[BlockCnt].CrossAttn.OProj := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.DModel) );
      NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
      Blocks[BlockCnt].CrossAttn.Norm := NN.AddLayer(
        TNNetTokenLayerNorm.Create(MarianLayerNormEps) );
    end;

    // ---- FFN sub-block: fc2(act(fc1(x))), then final_layer_norm ----
    BranchInput := NN.GetLastLayer();
    Blocks[BlockCnt].Fc1 := NN.AddLayer(
      TNNetPointwiseConvLinear.Create(FFNDim) );
    if Config.SwishFFN then
      NN.AddLayer( TNNetSwish.Create() )
    else
      NN.AddLayer( TNNetReLU.Create() );
    Blocks[BlockCnt].Fc2 := NN.AddLayer(
      TNNetPointwiseConvLinear.Create(Config.DModel) );
    NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
    Blocks[BlockCnt].FFNNorm := NN.AddLayer(
      TNNetTokenLayerNorm.Create(MarianLayerNormEps) );
    if pInferenceOnly then NN.MakeInferenceOnly();
    if pQuantizeInt8 then NN.QuantizeWeightsInt8();
  end;
  SetLength(SliceChannels, 0);
  SetLength(Heads, 0);
end;

// Fills a TNNetLearnedPositionalEmbedding table (SeqLen rows of DModel)
// with HF Marian's STATIC half-split sinusoids:
//   table[pos, c]         = sin(pos / 10000^(2c/DModel))   for c < DModel/2
//   table[pos, DModel/2+c] = cos(pos / 10000^(2c/DModel))
// (MarianSinusoidalPositionalEmbedding.create_weight - "features are not
// interleaved. The cos features are in the 2nd half of the vector"). Real
// checkpoints never store the table (_keys_to_ignore_on_save), so the
// importer regenerates it; HF builds it in float32, and these doubles
// round to the same float32 values.
procedure FillMarianSinusoidalPositions(Layer: TNNetLayer;
  SeqLen, DModel: integer);
var
  PosCnt, ChCnt, Half: integer;
  Angle: double;
begin
  Half := DModel div 2;
  for PosCnt := 0 to SeqLen - 1 do
    for ChCnt := 0 to Half - 1 do
    begin
      Angle := PosCnt / Exp(Ln(10000.0) * (2 * ChCnt) / DModel);
      Layer.Neurons[0].Weights.FData[PosCnt * DModel + ChCnt] := Sin(Angle);
      Layer.Neurons[0].Weights.FData[PosCnt * DModel + Half + ChCnt] :=
        Cos(Angle);
    end;
  Layer.FlushWeightCache();
end;

// Loads one Marian attention sub-block's q/k/v/out projections (HF
// nn.Linear [d_model, d_model], ALL biased) plus its post-residual
// LayerNorm (NormPrefix, weight+bias).
procedure LoadMarianAttn(Reader: TNNetSafeTensorsReader;
  const Attn: TMarianAttnLayers; const AttnPrefix, NormPrefix: string;
  const Config: TMarianConfig; Consumed: TStringList);
begin
  LoadLlamaLinearWeights(Reader, Attn.QProj, AttnPrefix + 'q_proj.weight',
    Config.DModel, Config.DModel, 0, -1, 0, AttnPrefix + 'q_proj.bias');
  Consumed.Add(AttnPrefix + 'q_proj.weight');
  Consumed.Add(AttnPrefix + 'q_proj.bias');
  LoadLlamaLinearWeights(Reader, Attn.KProj, AttnPrefix + 'k_proj.weight',
    Config.DModel, Config.DModel, 0, -1, 0, AttnPrefix + 'k_proj.bias');
  Consumed.Add(AttnPrefix + 'k_proj.weight');
  Consumed.Add(AttnPrefix + 'k_proj.bias');
  LoadLlamaLinearWeights(Reader, Attn.VProj, AttnPrefix + 'v_proj.weight',
    Config.DModel, Config.DModel, 0, -1, 0, AttnPrefix + 'v_proj.bias');
  Consumed.Add(AttnPrefix + 'v_proj.weight');
  Consumed.Add(AttnPrefix + 'v_proj.bias');
  LoadLlamaLinearWeights(Reader, Attn.OProj, AttnPrefix + 'out_proj.weight',
    Config.DModel, Config.DModel, 0, -1, 0, AttnPrefix + 'out_proj.bias');
  Consumed.Add(AttnPrefix + 'out_proj.weight');
  Consumed.Add(AttnPrefix + 'out_proj.bias');
  LoadLayerNormWeights(Reader, Attn.Norm, NormPrefix + '.weight',
    NormPrefix + '.bias', Config.DModel);
  Consumed.Add(NormPrefix + '.weight');
  Consumed.Add(NormPrefix + '.bias');
end;

// Loads one full Marian stack: per-block self-attention (+ cross-attention
// in the decoder) with their post-residual norms, plus fc1/fc2 and the
// block's final_layer_norm.
procedure LoadMarianStack(Reader: TNNetSafeTensorsReader;
  const Blocks: TMarianBlockArray; const StackPrefix: string;
  const Config: TMarianConfig; FFNDim: integer; IsDecoder: boolean;
  Consumed: TStringList);
var
  BlockCnt: integer;
  BP: string;
begin
  for BlockCnt := 0 to High(Blocks) do
  begin
    BP := StackPrefix + IntToStr(BlockCnt) + '.';
    LoadMarianAttn(Reader, Blocks[BlockCnt].SelfAttn, BP + 'self_attn.',
      BP + 'self_attn_layer_norm', Config, Consumed);
    if IsDecoder then
      LoadMarianAttn(Reader, Blocks[BlockCnt].CrossAttn,
        BP + 'encoder_attn.', BP + 'encoder_attn_layer_norm', Config,
        Consumed);
    LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Fc1,
      BP + 'fc1.weight', Config.DModel, FFNDim, 0, -1, 0, BP + 'fc1.bias');
    Consumed.Add(BP + 'fc1.weight');
    Consumed.Add(BP + 'fc1.bias');
    LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Fc2,
      BP + 'fc2.weight', FFNDim, Config.DModel, 0, -1, 0, BP + 'fc2.bias');
    Consumed.Add(BP + 'fc2.weight');
    Consumed.Add(BP + 'fc2.bias');
    LoadLayerNormWeights(Reader, Blocks[BlockCnt].FFNNorm,
      BP + 'final_layer_norm.weight', BP + 'final_layer_norm.bias',
      Config.DModel);
    Consumed.Add(BP + 'final_layer_norm.weight');
    Consumed.Add(BP + 'final_layer_norm.bias');
  end;
end;

procedure BuildMarianFromSafeTensorsWithConfig(const FileName: string;
  var Config: TMarianConfig; out EncoderNet, DecoderNet: TNNet;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false);
var
  Reader: TNNetSafeTensorsReader;
  Enc, Dec: TNNet;
  EncBlocks, DecBlocks: TMarianBlockArray;
  EncEmbed, DecEmbed, EncPos, DecPos: TNNetLayer;
  DecTokenInput, EncStates, LMHead: TNNetLayer;
  Consumed: TStringList;
  Tmp: TNNetVolume;
  i, j: integer;
  EmbedScale: TNeuralFloat;
  TensorNameStr: string;
begin
  EncoderNet := nil;
  DecoderNet := nil;
  // ---------------- Config validation ----------------
  if EncSeqLen < 1 then
    ImportError('Marian import: EncSeqLen must be >= 1, got ' +
      IntToStr(EncSeqLen) + '.');
  if DecSeqLen < 1 then
    ImportError('Marian import: DecSeqLen must be >= 1, got ' +
      IntToStr(DecSeqLen) + '.');
  if (EncSeqLen > Config.MaxPositionEmbeddings) or
     (DecSeqLen > Config.MaxPositionEmbeddings) then
    ImportError('Marian import: EncSeqLen/DecSeqLen must not exceed ' +
      'max_position_embeddings = ' +
      IntToStr(Config.MaxPositionEmbeddings) + '.');
  if Odd(Config.DModel) then
    ImportError('Marian import: d_model must be EVEN (the half-split ' +
      'sinusoidal table), got ' + IntToStr(Config.DModel) + '.');
  if (Config.EncoderHeads < 1) or
     ((Config.DModel mod Config.EncoderHeads) <> 0) then
    ImportError('Marian import: encoder_attention_heads must divide ' +
      'd_model, got ' + IntToStr(Config.EncoderHeads) + ' heads for ' +
      'd_model ' + IntToStr(Config.DModel) + '.');
  if (Config.DecoderHeads < 1) or
     ((Config.DModel mod Config.DecoderHeads) <> 0) then
    ImportError('Marian import: decoder_attention_heads must divide ' +
      'd_model, got ' + IntToStr(Config.DecoderHeads) + ' heads for ' +
      'd_model ' + IntToStr(Config.DModel) + '.');
  Reader := CreatePretrainedTensorReader(FileName);
  Enc := nil;
  Dec := nil;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      if not Reader.HasTensor('model.shared.weight') then
        ImportError('Marian import: "model.shared.weight" not found in ' +
          Reader.FileName + ' - not a Marian checkpoint?');
      if (Reader.DimCount('model.shared.weight') <> 2) or
         (Reader.DimSize('model.shared.weight', 0) <> Config.VocabSize) or
         (Reader.DimSize('model.shared.weight', 1) <> Config.DModel) then
        ImportError('Marian import: model.shared.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.DModel) +
          '], got ' + Reader.ShapeAsString('model.shared.weight'));
      if not Reader.HasTensor('final_logits_bias') then
        ImportError('Marian import: "final_logits_bias" not found in ' +
          Reader.FileName + ' - not a MarianMTModel checkpoint?');
      if (Reader.DimCount('final_logits_bias') <> 2) or
         (Reader.DimSize('final_logits_bias', 0) <> 1) or
         (Reader.DimSize('final_logits_bias', 1) <> Config.VocabSize) then
        ImportError('Marian import: final_logits_bias must have shape ' +
          '[1, ' + IntToStr(Config.VocabSize) + '], got ' +
          Reader.ShapeAsString('final_logits_bias'));

      // ---------------- Encoder architecture ----------------
      Enc := TNNet.Create();
      Enc.AddLayer( TNNetInput.Create(EncSeqLen) );
      // EncodeZero=1: token id 0 (eos in opus-mt) is a real embedding row.
      EncEmbed := Enc.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.DModel, {EncodeZero=}1) );
      // Static sinusoidal positions, added AFTER the (scaled) token
      // embedding; the table is filled below (Marian half-split layout).
      EncPos := Enc.AddLayer(
        TNNetLearnedPositionalEmbedding.Create(EncSeqLen) );
      if pInferenceOnly then Enc.MakeInferenceOnly();
      if pQuantizeInt8 then Enc.QuantizeWeightsInt8();
      BuildMarianStackBlocks(Enc, Config, Config.EncoderLayers,
        Config.EncoderHeads, Config.EncoderFFNDim, {IsDecoder=}false,
        nil, EncBlocks, pInferenceOnly, pQuantizeInt8);
      // POST-norm stack: NO final norm (the last block ends in one).

      // ---------------- Decoder architecture ----------------
      // TWO inputs: Layers[0] = decoder token ids (what Compute feeds);
      // the second TNNetInput holds the encoder hidden states and is
      // filled MANUALLY before Compute (RunT5 does it - the convention is
      // shared with the T5 importer).
      Dec := TNNet.Create();
      DecTokenInput := Dec.AddLayer( TNNetInput.Create(DecSeqLen) );
      EncStates := Dec.AddLayerAfter(
        TNNetInput.Create(EncSeqLen, 1, Config.DModel, 1), 0);
      DecEmbed := Dec.AddLayerAfter( TNNetEmbedding.Create(
        Config.VocabSize, Config.DModel, {EncodeZero=}1), DecTokenInput);
      DecPos := Dec.AddLayer(
        TNNetLearnedPositionalEmbedding.Create(DecSeqLen) );
      if pInferenceOnly then Dec.MakeInferenceOnly();
      if pQuantizeInt8 then Dec.QuantizeWeightsInt8();
      BuildMarianStackBlocks(Dec, Config, Config.DecoderLayers,
        Config.DecoderHeads, Config.DecoderFFNDim, {IsDecoder=}true,
        EncStates, DecBlocks, pInferenceOnly, pQuantizeInt8);
      LMHead := Dec.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then Dec.MakeInferenceOnly();
      if pQuantizeInt8 then Dec.QuantizeWeightsInt8();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        Reader.LoadTensorFlat('model.shared.weight', Tmp);
        if EncEmbed.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('Marian import: model.shared.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table ' +
            'size ' + IntToStr(EncEmbed.Neurons[0].Weights.Size) + '.');
        // scale_embedding folds sqrt(d_model) into the embedding rows
        // (token embeds are scaled BEFORE positions are added); the TIED
        // head below copies the UNSCALED rows.
        if Config.ScaleEmbedding then
          EmbedScale := Sqrt(Config.DModel)
        else
          EmbedScale := 1.0;
        for i := 0 to Tmp.Size - 1 do
        begin
          EncEmbed.Neurons[0].Weights.FData[i] := EmbedScale * Tmp.FData[i];
          DecEmbed.Neurons[0].Weights.FData[i] := EmbedScale * Tmp.FData[i];
        end;
        EncEmbed.FlushWeightCache();
        DecEmbed.FlushWeightCache();
        Consumed.Add('model.shared.weight');
        // Tied head: logits = h . shared^T + final_logits_bias.
        EnsureWritableImportWeights(LMHead);
        for j := 0 to Config.VocabSize - 1 do
          for i := 0 to Config.DModel - 1 do
            LMHead.Neurons[j].Weights.FData[i] :=
              Tmp.FData[j * Config.DModel + i];
        Reader.LoadTensorFlat('final_logits_bias', Tmp);
        for j := 0 to Config.VocabSize - 1 do
          LMHead.Neurons[j].BiasWeight := Tmp.FData[j];
        LMHead.FlushWeightCache();
        Consumed.Add('final_logits_bias');
        // Legacy exports may still carry the tied/static aliases (HF drops
        // embed_tokens via _tied_weights_keys and the SINUSOIDAL
        // embed_positions via _keys_to_ignore_on_save).
        if Reader.HasTensor('lm_head.weight') then
          Consumed.Add('lm_head.weight');
        if Reader.HasTensor('model.encoder.embed_tokens.weight') then
          Consumed.Add('model.encoder.embed_tokens.weight');
        if Reader.HasTensor('model.decoder.embed_tokens.weight') then
          Consumed.Add('model.decoder.embed_tokens.weight');
        if Reader.HasTensor('model.encoder.embed_positions.weight') then
          Consumed.Add('model.encoder.embed_positions.weight');
        if Reader.HasTensor('model.decoder.embed_positions.weight') then
          Consumed.Add('model.decoder.embed_positions.weight');
      finally
        Tmp.Free;
      end;
      FillMarianSinusoidalPositions(EncPos, EncSeqLen, Config.DModel);
      FillMarianSinusoidalPositions(DecPos, DecSeqLen, Config.DModel);
      LoadMarianStack(Reader, EncBlocks, 'model.encoder.layers.', Config,
        Config.EncoderFFNDim, {IsDecoder=}false, Consumed);
      // Re-quantize the stack just refilled with checkpoint weights.
      if pQuantizeInt8 then Enc.QuantizeWeightsInt8();
      LoadMarianStack(Reader, DecBlocks, 'model.decoder.layers.', Config,
        Config.DecoderFFNDim, {IsDecoder=}true, Consumed);
      // Re-quantize the stack just refilled with checkpoint weights.
      if pQuantizeInt8 then Dec.QuantizeWeightsInt8();

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then Enc.QuantizeWeightsInt8();
      if pQuantizeInt8 then Dec.QuantizeWeightsInt8();
      // ---------------- Unexpected-tensor check ----------------
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        ImportError('Marian import: unexpected tensor "' + TensorNameStr +
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

procedure BuildMarianFromSafeTensors(const FileName: string;
  out EncoderNet, DecoderNet: TNNet; out Config: TMarianConfig;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false);
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadMarianConfigFromJSONFile(ConfigPath);
  BuildMarianFromSafeTensorsWithConfig(FileName, Config, EncoderNet,
    DecoderNet, EncSeqLen, DecSeqLen, pInferenceOnly, pQuantizeInt8);
end;

// ===========================================================================
// BART IMPORT
// ===========================================================================

const
  // nn.LayerNorm's default eps - BART never overrides it.
  BartLayerNormEps = 1e-5;
  // BartLearnedPositionalEmbedding's padding offset: token position p reads
  // table row p + 2 (rows 0/1 are the padding-idx slots).
  BartPositionOffset = 2;

function ReadBartConfigFromJSONFile(const FileName: string): TBartConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, ActFn: string;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('BART import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('BART import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

begin
  if not FileExists(FileName) then
    ImportError('BART import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('BART import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('BART import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'bart');
    if ModelType <> 'bart' then
      ImportError('BART import: config model_type is "' + ModelType +
        '" - only "bart" is supported here (see BuildFromPretrained for ' +
        'the full dispatch).');
    Result.ModelType := ModelType;
    Result.DModel := RequiredInt('d_model');
    Result.EncoderLayers := RequiredInt('encoder_layers');
    Result.DecoderLayers := RequiredInt('decoder_layers');
    Result.EncoderHeads := RequiredInt('encoder_attention_heads');
    Result.DecoderHeads := RequiredInt('decoder_attention_heads');
    Result.EncoderFFNDim := RequiredInt('encoder_ffn_dim');
    Result.DecoderFFNDim := RequiredInt('decoder_ffn_dim');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.MaxPositionEmbeddings :=
      Obj.Get('max_position_embeddings', 1024);
    // HF BartConfig defaults.
    Result.PadTokenId := Obj.Get('pad_token_id', 1);
    Result.BosTokenId := Obj.Get('bos_token_id', 0);
    Result.EosTokenId := Obj.Get('eos_token_id', 2);
    Result.DecoderStartTokenId :=
      Obj.Get('decoder_start_token_id', Result.EosTokenId);
    Result.ScaleEmbedding := Obj.Get('scale_embedding', False);
    // Only the published BART recipe ("gelu", the exact erf form) is
    // supported. gelu_new / relu / swish are rejected to avoid a silent
    // activation mismatch.
    ActFn := Obj.Get('activation_function', 'gelu');
    if ActFn <> 'gelu' then
      ImportError('BART import: activation_function "' + ActFn +
        '" is not supported - expected "gelu" (the exact erf form every ' +
        'published BART pins).');
    if not Obj.Get('tie_word_embeddings', True) then
      ImportError('BART import: tie_word_embeddings=false is not ' +
        'supported - BART ties the lm_head to the shared embedding.');
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function BartConfigToString(const Config: TBartConfig): string;
begin
  if Config.ModelType = '' then Result := 'bart'
  else Result := Config.ModelType;
  Result := Result + ' config: enc_layers=' +
    IntToStr(Config.EncoderLayers) +
    ', dec_layers=' + IntToStr(Config.DecoderLayers) +
    ', enc_heads=' + IntToStr(Config.EncoderHeads) +
    ', dec_heads=' + IntToStr(Config.DecoderHeads) +
    ', d_model=' + IntToStr(Config.DModel) +
    ', enc_ffn=' + IntToStr(Config.EncoderFFNDim) +
    ', dec_ffn=' + IntToStr(Config.DecoderFFNDim) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', max_pos=' + IntToStr(Config.MaxPositionEmbeddings) +
    ', pad=' + IntToStr(Config.PadTokenId) +
    ', eos=' + IntToStr(Config.EosTokenId) +
    ', dec_start=' + IntToStr(Config.DecoderStartTokenId) +
    ', scale_embedding=' + BoolToStr(Config.ScaleEmbedding, true) +
    ', ffn=gelu';
end;

// Grows one BART stack (encoder or decoder) onto NN after the embedding +
// positional + layernorm_embedding layers. POST-norm wiring identical to
// Marian (residual add THEN biased LayerNorm; no final stack norm), but the
// FFN uses the EXACT-erf GELU (the BERT composition) instead of swish/relu.
// IsDecoder adds the causal self-attention mask plus the cross-attention
// sub-block reading Keys|Values from EncStates. The layer handles reuse the
// Marian block record (TMarianBlockArray); .CrossAttn is filled only for
// decoder blocks.
procedure BuildBartStackBlocks(NN: TNNet; const Config: TBartConfig;
  NumBlocks, NumHeads, FFNDim: integer; IsDecoder: boolean;
  EncStates: TNNetLayer; var Blocks: TMarianBlockArray;
  pInferenceOnly: boolean; pQuantizeInt8: boolean = false);
var
  HeadDim, BlockCnt, HeadCnt, d: integer;
  BranchInput, HiddenAct, PhiBranch: TNNetLayer;
  QSlice, KSlice, VSlice, HeadPack, KVPack: TNNetLayer;
  Heads: array of TNNetLayer;
  SliceChannels: array of integer;
begin
  HeadDim := Config.DModel div NumHeads;
  SetLength(Blocks, NumBlocks);
  SetLength(SliceChannels, HeadDim);
  SetLength(Heads, NumHeads);
  for BlockCnt := 0 to NumBlocks - 1 do
  begin
    // ---- self-attention sub-block (residual add, THEN LayerNorm) ----
    BranchInput := NN.GetLastLayer();
    Blocks[BlockCnt].SelfAttn.QProj := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(Config.DModel), BranchInput);
    Blocks[BlockCnt].SelfAttn.KProj := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(Config.DModel), BranchInput);
    Blocks[BlockCnt].SelfAttn.VProj := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(Config.DModel), BranchInput);
    for HeadCnt := 0 to NumHeads - 1 do
    begin
      for d := 0 to HeadDim - 1 do
        SliceChannels[d] := HeadCnt * HeadDim + d;
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
      Heads[HeadCnt] := NN.AddLayerAfter(
        TNNetScaledDotProductAttention.Create(HeadDim,
          {CausalMask=}IsDecoder), HeadPack);
    end;
    NN.AddLayer( TNNetDeepConcat.Create(Heads) );
    Blocks[BlockCnt].SelfAttn.OProj := NN.AddLayer(
      TNNetPointwiseConvLinear.Create(Config.DModel) );
    NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
    Blocks[BlockCnt].SelfAttn.Norm := NN.AddLayer(
      TNNetTokenLayerNorm.Create(BartLayerNormEps) );

    // ---- cross-attention sub-block (decoder only) ----
    if IsDecoder then
    begin
      BranchInput := NN.GetLastLayer();
      Blocks[BlockCnt].CrossAttn.QProj := NN.AddLayerAfter(
        TNNetPointwiseConvLinear.Create(Config.DModel), BranchInput);
      Blocks[BlockCnt].CrossAttn.KProj := NN.AddLayerAfter(
        TNNetPointwiseConvLinear.Create(Config.DModel), EncStates);
      Blocks[BlockCnt].CrossAttn.VProj := NN.AddLayerAfter(
        TNNetPointwiseConvLinear.Create(Config.DModel), EncStates);
      for HeadCnt := 0 to NumHeads - 1 do
      begin
        for d := 0 to HeadDim - 1 do
          SliceChannels[d] := HeadCnt * HeadDim + d;
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
        Heads[HeadCnt] := NN.AddLayerAfter(
          TNNetCrossAttention.Create(HeadDim, {CausalMask=}false,
            KVPack), QSlice);
      end;
      NN.AddLayer( TNNetDeepConcat.Create(Heads) );
      Blocks[BlockCnt].CrossAttn.OProj := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.DModel) );
      NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
      Blocks[BlockCnt].CrossAttn.Norm := NN.AddLayer(
        TNNetTokenLayerNorm.Create(BartLayerNormEps) );
    end;

    // ---- FFN sub-block: fc2(gelu(fc1(x))), then final_layer_norm ----
    BranchInput := NN.GetLastLayer();
    Blocks[BlockCnt].Fc1 := NN.AddLayer(
      TNNetPointwiseConvLinear.Create(FFNDim) );
    // EXACT erf GELU x*Phi(x) composed from existing layers (the BERT
    // recipe): Phi = 0.5*(1 + erf(x/sqrt(2))) on a side branch, then
    // ReGLU(Phi|x) = ReLU(Phi)*x = Phi*x since Phi is in (0, 1). TNNetReGLU
    // applies the ReLU to the FIRST depth half, so Phi must come first.
    HiddenAct := NN.GetLastLayer();
    NN.AddLayerAfter(
      TNNetMulByConstant.Create(0.7071067811865476), HiddenAct);
    NN.AddLayer( TNNetErf.Create() );
    NN.AddLayer( TNNetAddConstant.Create(1.0) );
    PhiBranch := NN.AddLayer( TNNetMulByConstant.Create(0.5) );
    NN.AddLayer( TNNetDeepConcat.Create([PhiBranch, HiddenAct]) );
    NN.AddLayer( TNNetReGLU.Create() );
    Blocks[BlockCnt].Fc2 := NN.AddLayer(
      TNNetPointwiseConvLinear.Create(Config.DModel) );
    NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
    Blocks[BlockCnt].FFNNorm := NN.AddLayer(
      TNNetTokenLayerNorm.Create(BartLayerNormEps) );
    if pInferenceOnly then NN.MakeInferenceOnly();
    if pQuantizeInt8 then NN.QuantizeWeightsInt8();
  end;
  SetLength(SliceChannels, 0);
  SetLength(Heads, 0);
end;

procedure BuildBartFromSafeTensorsWithConfig(const FileName: string;
  var Config: TBartConfig; out EncoderNet, DecoderNet: TNNet;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false);
var
  Reader: TNNetSafeTensorsReader;
  Enc, Dec: TNNet;
  EncBlocks, DecBlocks: TMarianBlockArray;
  EncEmbed, DecEmbed, EncPos, DecPos, EncEmbLN, DecEmbLN: TNNetLayer;
  DecTokenInput, EncStates, LMHead: TNNetLayer;
  Consumed: TStringList;
  Tmp: TNNetVolume;
  i, j: integer;
  EmbedScale: TNeuralFloat;
  TensorNameStr: string;
  // LoadMarianStack/LoadMarianAttn only read .DModel from the config; this
  // shim carries it so the shared per-block loader works unchanged.
  MarianShim: TMarianConfig;

  // Loads embed_positions rows BartPositionOffset..SeqLen+offset-1 into the
  // SeqLen-row TNNetLearnedPositionalEmbedding table (token position p reads
  // checkpoint row p + 2).
  procedure LoadBartPositions(PosLayer: TNNetLayer; const TName: string;
    SeqLen: integer);
  var
    PosCnt, ElementCnt: integer;
  begin
    if not Reader.HasTensor(TName) then
      ImportError('BART import: missing tensor "' + TName + '".');
    if (Reader.DimCount(TName) <> 2) or
       (Reader.DimSize(TName, 0) <>
          Config.MaxPositionEmbeddings + BartPositionOffset) or
       (Reader.DimSize(TName, 1) <> Config.DModel) then
      ImportError('BART import: "' + TName + '" must have shape [' +
        IntToStr(Config.MaxPositionEmbeddings + BartPositionOffset) + ', ' +
        IntToStr(Config.DModel) + '], got ' + Reader.ShapeAsString(TName));
    Reader.LoadTensorFlat(TName, Tmp);
    for PosCnt := 0 to SeqLen - 1 do
      for ElementCnt := 0 to Config.DModel - 1 do
        PosLayer.Neurons[0].Weights.FData[PosCnt * Config.DModel +
          ElementCnt] := Tmp.FData[(PosCnt + BartPositionOffset) *
            Config.DModel + ElementCnt];
    PosLayer.FlushWeightCache();
    Consumed.Add(TName);
  end;

begin
  EncoderNet := nil;
  DecoderNet := nil;
  // ---------------- Config validation ----------------
  if EncSeqLen < 1 then
    ImportError('BART import: EncSeqLen must be >= 1, got ' +
      IntToStr(EncSeqLen) + '.');
  if DecSeqLen < 1 then
    ImportError('BART import: DecSeqLen must be >= 1, got ' +
      IntToStr(DecSeqLen) + '.');
  if (EncSeqLen > Config.MaxPositionEmbeddings) or
     (DecSeqLen > Config.MaxPositionEmbeddings) then
    ImportError('BART import: EncSeqLen/DecSeqLen must not exceed ' +
      'max_position_embeddings = ' +
      IntToStr(Config.MaxPositionEmbeddings) + '.');
  if (Config.EncoderHeads < 1) or
     ((Config.DModel mod Config.EncoderHeads) <> 0) then
    ImportError('BART import: encoder_attention_heads must divide ' +
      'd_model, got ' + IntToStr(Config.EncoderHeads) + ' heads for ' +
      'd_model ' + IntToStr(Config.DModel) + '.');
  if (Config.DecoderHeads < 1) or
     ((Config.DModel mod Config.DecoderHeads) <> 0) then
    ImportError('BART import: decoder_attention_heads must divide ' +
      'd_model, got ' + IntToStr(Config.DecoderHeads) + ' heads for ' +
      'd_model ' + IntToStr(Config.DModel) + '.');
  Reader := CreatePretrainedTensorReader(FileName);
  Enc := nil;
  Dec := nil;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      if not Reader.HasTensor('model.shared.weight') then
        ImportError('BART import: "model.shared.weight" not found in ' +
          Reader.FileName + ' - not a BART checkpoint?');
      if (Reader.DimCount('model.shared.weight') <> 2) or
         (Reader.DimSize('model.shared.weight', 0) <> Config.VocabSize) or
         (Reader.DimSize('model.shared.weight', 1) <> Config.DModel) then
        ImportError('BART import: model.shared.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.DModel) +
          '], got ' + Reader.ShapeAsString('model.shared.weight'));
      if not Reader.HasTensor('final_logits_bias') then
        ImportError('BART import: "final_logits_bias" not found in ' +
          Reader.FileName +
          ' - not a BartForConditionalGeneration checkpoint?');
      if (Reader.DimCount('final_logits_bias') <> 2) or
         (Reader.DimSize('final_logits_bias', 0) <> 1) or
         (Reader.DimSize('final_logits_bias', 1) <> Config.VocabSize) then
        ImportError('BART import: final_logits_bias must have shape ' +
          '[1, ' + IntToStr(Config.VocabSize) + '], got ' +
          Reader.ShapeAsString('final_logits_bias'));

      // ---------------- Encoder architecture ----------------
      Enc := TNNet.Create();
      Enc.AddLayer( TNNetInput.Create(EncSeqLen) );
      EncEmbed := Enc.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.DModel, {EncodeZero=}1) );
      EncPos := Enc.AddLayer(
        TNNetLearnedPositionalEmbedding.Create(EncSeqLen) );
      EncEmbLN := Enc.AddLayer(
        TNNetTokenLayerNorm.Create(BartLayerNormEps) );
      if pInferenceOnly then Enc.MakeInferenceOnly();
      if pQuantizeInt8 then Enc.QuantizeWeightsInt8();
      BuildBartStackBlocks(Enc, Config, Config.EncoderLayers,
        Config.EncoderHeads, Config.EncoderFFNDim, {IsDecoder=}false,
        nil, EncBlocks, pInferenceOnly, pQuantizeInt8);

      // ---------------- Decoder architecture ----------------
      Dec := TNNet.Create();
      DecTokenInput := Dec.AddLayer( TNNetInput.Create(DecSeqLen) );
      EncStates := Dec.AddLayerAfter(
        TNNetInput.Create(EncSeqLen, 1, Config.DModel, 1), 0);
      DecEmbed := Dec.AddLayerAfter( TNNetEmbedding.Create(
        Config.VocabSize, Config.DModel, {EncodeZero=}1), DecTokenInput);
      DecPos := Dec.AddLayer(
        TNNetLearnedPositionalEmbedding.Create(DecSeqLen) );
      DecEmbLN := Dec.AddLayer(
        TNNetTokenLayerNorm.Create(BartLayerNormEps) );
      if pInferenceOnly then Dec.MakeInferenceOnly();
      if pQuantizeInt8 then Dec.QuantizeWeightsInt8();
      BuildBartStackBlocks(Dec, Config, Config.DecoderLayers,
        Config.DecoderHeads, Config.DecoderFFNDim, {IsDecoder=}true,
        EncStates, DecBlocks, pInferenceOnly, pQuantizeInt8);
      LMHead := Dec.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then Dec.MakeInferenceOnly();
      if pQuantizeInt8 then Dec.QuantizeWeightsInt8();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        Reader.LoadTensorFlat('model.shared.weight', Tmp);
        if EncEmbed.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('BART import: model.shared.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table ' +
            'size ' + IntToStr(EncEmbed.Neurons[0].Weights.Size) + '.');
        if Config.ScaleEmbedding then
          EmbedScale := Sqrt(Config.DModel)
        else
          EmbedScale := 1.0;
        for i := 0 to Tmp.Size - 1 do
        begin
          EncEmbed.Neurons[0].Weights.FData[i] := EmbedScale * Tmp.FData[i];
          DecEmbed.Neurons[0].Weights.FData[i] := EmbedScale * Tmp.FData[i];
        end;
        EncEmbed.FlushWeightCache();
        DecEmbed.FlushWeightCache();
        Consumed.Add('model.shared.weight');
        // Tied head: logits = h . shared^T + final_logits_bias.
        EnsureWritableImportWeights(LMHead);
        for j := 0 to Config.VocabSize - 1 do
          for i := 0 to Config.DModel - 1 do
            LMHead.Neurons[j].Weights.FData[i] :=
              Tmp.FData[j * Config.DModel + i];
        Reader.LoadTensorFlat('final_logits_bias', Tmp);
        for j := 0 to Config.VocabSize - 1 do
          LMHead.Neurons[j].BiasWeight := Tmp.FData[j];
        LMHead.FlushWeightCache();
        Consumed.Add('final_logits_bias');
        // Legacy exports may still carry the tied aliases (HF drops
        // embed_tokens and lm_head via _tied_weights_keys).
        if Reader.HasTensor('lm_head.weight') then
          Consumed.Add('lm_head.weight');
        if Reader.HasTensor('model.encoder.embed_tokens.weight') then
          Consumed.Add('model.encoder.embed_tokens.weight');
        if Reader.HasTensor('model.decoder.embed_tokens.weight') then
          Consumed.Add('model.decoder.embed_tokens.weight');
        // Learned positions (the +2-offset window).
        LoadBartPositions(EncPos,
          'model.encoder.embed_positions.weight', EncSeqLen);
        LoadBartPositions(DecPos,
          'model.decoder.embed_positions.weight', DecSeqLen);
      finally
        Tmp.Free;
      end;
      // layernorm_embedding (after token+position embeddings).
      LoadLayerNormWeights(Reader, EncEmbLN,
        'model.encoder.layernorm_embedding.weight',
        'model.encoder.layernorm_embedding.bias', Config.DModel);
      Consumed.Add('model.encoder.layernorm_embedding.weight');
      Consumed.Add('model.encoder.layernorm_embedding.bias');
      LoadLayerNormWeights(Reader, DecEmbLN,
        'model.decoder.layernorm_embedding.weight',
        'model.decoder.layernorm_embedding.bias', Config.DModel);
      Consumed.Add('model.decoder.layernorm_embedding.weight');
      Consumed.Add('model.decoder.layernorm_embedding.bias');
      // Per-block weights (same names/shapes as Marian; the loader only
      // needs d_model from the shim config).
      MarianShim.DModel := Config.DModel;
      LoadMarianStack(Reader, EncBlocks, 'model.encoder.layers.',
        MarianShim, Config.EncoderFFNDim, {IsDecoder=}false,
        Consumed);
      if pQuantizeInt8 then Enc.QuantizeWeightsInt8();
      LoadMarianStack(Reader, DecBlocks, 'model.decoder.layers.',
        MarianShim, Config.DecoderFFNDim, {IsDecoder=}true,
        Consumed);
      if pQuantizeInt8 then Dec.QuantizeWeightsInt8();
      if pQuantizeInt8 then Enc.QuantizeWeightsInt8();
      if pQuantizeInt8 then Dec.QuantizeWeightsInt8();
      // ---------------- Unexpected-tensor check ----------------
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        ImportError('BART import: unexpected tensor "' + TensorNameStr +
          '" (shape ' + Reader.ShapeAsString(TensorNameStr) +
          ') in ' + FileName + ' - refusing a partial import.');
      end;
      EncoderNet := Enc;
      DecoderNet := Dec;
      Enc := nil;
      Dec := nil;
    except
      on E: ESafeTensorsError do
        raise EPretrainedImportError.Create(E.Message);
    end;
  finally
    Enc.Free;
    Dec.Free;
    Consumed.Free;
    Reader.Free;
  end;
end;

procedure BuildBartFromSafeTensorsEx(const FileName: string;
  Config: TBartConfig; out EncoderNet, DecoderNet: TNNet;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false);
begin
  BuildBartFromSafeTensorsWithConfig(FileName, Config, EncoderNet,
    DecoderNet, EncSeqLen, DecSeqLen, pInferenceOnly, pQuantizeInt8);
end;

procedure BuildBartFromSafeTensors(const FileName: string;
  out EncoderNet, DecoderNet: TNNet; out Config: TBartConfig;
  EncSeqLen, DecSeqLen: integer; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false);
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadBartConfigFromJSONFile(ConfigPath);
  BuildBartFromSafeTensorsWithConfig(FileName, Config, EncoderNet,
    DecoderNet, EncSeqLen, DecSeqLen, pInferenceOnly, pQuantizeInt8);
end;

function ReadWhisperConfigFromJSONFile(const FileName: string): TWhisperConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType, ActFn: string;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('Whisper import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('Whisper import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

begin
  if not FileExists(FileName) then
    ImportError('Whisper import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('Whisper import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('Whisper import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'whisper');
    if ModelType <> 'whisper' then
      ImportError('Whisper import: config model_type is "' + ModelType +
        '" - only "whisper" is supported here (see BuildFromPretrained ' +
        'for the full dispatch).');
    Result.ModelType := ModelType;
    Result.DModel := RequiredInt('d_model');
    Result.EncoderLayers := RequiredInt('encoder_layers');
    Result.DecoderLayers := RequiredInt('decoder_layers');
    Result.EncoderHeads := RequiredInt('encoder_attention_heads');
    Result.DecoderHeads := RequiredInt('decoder_attention_heads');
    Result.EncoderFFNDim := RequiredInt('encoder_ffn_dim');
    Result.DecoderFFNDim := RequiredInt('decoder_ffn_dim');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.NumMelBins := RequiredInt('num_mel_bins');
    Result.MaxSourcePositions := RequiredInt('max_source_positions');
    Result.MaxTargetPositions := RequiredInt('max_target_positions');
    // The HF WhisperConfig defaults (the released checkpoints pin their
    // own special-token ids explicitly - tiny: pad/eos 50257-region ids).
    Result.PadTokenId := Obj.Get('pad_token_id', 50256);
    Result.EosTokenId := Obj.Get('eos_token_id', 50256);
    Result.DecoderStartTokenId := Obj.Get('decoder_start_token_id', 50257);
    Result.ScaleEmbedding := Obj.Get('scale_embedding', False);
    // Every published Whisper recipe uses "gelu" - the EXACT erf form.
    // Anything else is rejected to avoid a silent activation mismatch.
    ActFn := Obj.Get('activation_function', 'gelu');
    if ActFn <> 'gelu' then
      ImportError('Whisper import: activation_function "' + ActFn +
        '" is not supported - every published Whisper uses "gelu".');
    if not Obj.Get('tie_word_embeddings', True) then
      ImportError('Whisper import: tie_word_embeddings=false is not ' +
        'supported - Whisper ties proj_out to the decoder embedding.');
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function WhisperConfigToString(const Config: TWhisperConfig): string;
begin
  if Config.ModelType = '' then Result := 'whisper'
  else Result := Config.ModelType;
  Result := Result + ' config: enc_layers=' + IntToStr(Config.EncoderLayers) +
    ', dec_layers=' + IntToStr(Config.DecoderLayers) +
    ', enc_heads=' + IntToStr(Config.EncoderHeads) +
    ', dec_heads=' + IntToStr(Config.DecoderHeads) +
    ', d_model=' + IntToStr(Config.DModel) +
    ', enc_ffn=' + IntToStr(Config.EncoderFFNDim) +
    ', dec_ffn=' + IntToStr(Config.DecoderFFNDim) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', mel_bins=' + IntToStr(Config.NumMelBins) +
    ', max_src=' + IntToStr(Config.MaxSourcePositions) +
    ', max_tgt=' + IntToStr(Config.MaxTargetPositions) +
    ', pad=' + IntToStr(Config.PadTokenId) +
    ', eos=' + IntToStr(Config.EosTokenId) +
    ', dec_start=' + IntToStr(Config.DecoderStartTokenId) +
    ', scale_embedding=' + BoolToStr(Config.ScaleEmbedding, true);
end;

// Appends the EXACT erf GELU 0.5*x*(1+erf(x/sqrt(2))) after the current
// last layer - the same composition the BERT/GPT-NeoX importers use:
// Phi(x) = MulByConstant(1/sqrt(2)) -> Erf -> AddConstant(1) ->
// MulByConstant(0.5), then DeepConcat([Phi, x]) -> ReGLU multiplies the
// two halves (Phi is always positive, so the ReLU inside ReGLU is a
// pass-through on it).
procedure AddWhisperExactGelu(NN: TNNet);
var
  GELUSource, PhiBranch: TNNetLayer;
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

// Grows one Whisper stack (encoder or decoder) onto NN after the frontend
// (conv+positions / embedding+positions). PRE-norm wiring: every sublayer
// norms FIRST (self_attn_layer_norm / encoder_attn_layer_norm /
// final_layer_norm), the residual adds the RAW stream - the opposite of
// Marian's post-norm; the caller appends the FINAL stack LayerNorm.
// IsDecoder adds the causal mask on self-attention plus the
// cross-attention sub-block reading Keys|Values from EncStates (the
// decoder net's second TNNetInput) - TNNetCrossAttention forms the
// rectangular DecSeqLen x EncSeqLen scores. The layer-handle records are
// shared with the Marian loader types; .Norm holds each sublayer's PRE
// norm here.
procedure BuildWhisperStackBlocks(NN: TNNet; const Config: TWhisperConfig;
  NumBlocks, NumHeads, FFNDim: integer; IsDecoder: boolean;
  EncStates: TNNetLayer; var Blocks: TMarianBlockArray;
  pInferenceOnly: boolean);
var
  HeadDim, BlockCnt, HeadCnt, d: integer;
  BranchInput, NormOut: TNNetLayer;
  QSlice, KSlice, VSlice, HeadPack, KVPack: TNNetLayer;
  Heads: array of TNNetLayer;
  SliceChannels: array of integer;
begin
  HeadDim := Config.DModel div NumHeads;
  SetLength(Blocks, NumBlocks);
  SetLength(SliceChannels, HeadDim);
  SetLength(Heads, NumHeads);
  for BlockCnt := 0 to NumBlocks - 1 do
  begin
    // ---- self-attention sub-block (norm FIRST, then residual add) ----
    // Standard softmax(QK^T/sqrt(head_dim)) per head - exactly what
    // TNNetScaledDotProductAttention computes, so the q/k/v weights load
    // unmodified.
    BranchInput := NN.GetLastLayer();
    Blocks[BlockCnt].SelfAttn.Norm := NN.AddLayer(
      TNNetTokenLayerNorm.Create(MarianLayerNormEps) );
    NormOut := NN.GetLastLayer();
    Blocks[BlockCnt].SelfAttn.QProj := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(Config.DModel), NormOut);
    Blocks[BlockCnt].SelfAttn.KProj := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(Config.DModel), NormOut);
    Blocks[BlockCnt].SelfAttn.VProj := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(Config.DModel), NormOut);
    for HeadCnt := 0 to NumHeads - 1 do
    begin
      for d := 0 to HeadDim - 1 do
        SliceChannels[d] := HeadCnt * HeadDim + d;
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
      Heads[HeadCnt] := NN.AddLayerAfter(
        TNNetScaledDotProductAttention.Create(HeadDim,
          {CausalMask=}IsDecoder), HeadPack);
    end;
    NN.AddLayer( TNNetDeepConcat.Create(Heads) );
    Blocks[BlockCnt].SelfAttn.OProj := NN.AddLayer(
      TNNetPointwiseConvLinear.Create(Config.DModel) );
    NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );

    // ---- cross-attention sub-block (decoder only) ----
    // Queries from the pre-normed decoder stream; Keys|Values projected
    // from EncStates - the two grids have DIFFERENT lengths, so the
    // per-head leaf is TNNetCrossAttention (rectangular DecSeqLen x
    // EncSeqLen scores; a Q|K|V DeepConcat would be illegal across
    // unequal SizeX).
    if IsDecoder then
    begin
      BranchInput := NN.GetLastLayer();
      Blocks[BlockCnt].CrossAttn.Norm := NN.AddLayer(
        TNNetTokenLayerNorm.Create(MarianLayerNormEps) );
      NormOut := NN.GetLastLayer();
      Blocks[BlockCnt].CrossAttn.QProj := NN.AddLayerAfter(
        TNNetPointwiseConvLinear.Create(Config.DModel), NormOut);
      Blocks[BlockCnt].CrossAttn.KProj := NN.AddLayerAfter(
        TNNetPointwiseConvLinear.Create(Config.DModel), EncStates);
      Blocks[BlockCnt].CrossAttn.VProj := NN.AddLayerAfter(
        TNNetPointwiseConvLinear.Create(Config.DModel), EncStates);
      for HeadCnt := 0 to NumHeads - 1 do
      begin
        for d := 0 to HeadDim - 1 do
          SliceChannels[d] := HeadCnt * HeadDim + d;
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
        Heads[HeadCnt] := NN.AddLayerAfter(
          TNNetCrossAttention.Create(HeadDim, {CausalMask=}false,
            KVPack), QSlice);
      end;
      NN.AddLayer( TNNetDeepConcat.Create(Heads) );
      Blocks[BlockCnt].CrossAttn.OProj := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.DModel) );
      NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
    end;

    // ---- FFN sub-block: x := x + fc2(gelu(fc1(final_layer_norm(x)))) ----
    BranchInput := NN.GetLastLayer();
    Blocks[BlockCnt].FFNNorm := NN.AddLayer(
      TNNetTokenLayerNorm.Create(MarianLayerNormEps) );
    Blocks[BlockCnt].Fc1 := NN.AddLayer(
      TNNetPointwiseConvLinear.Create(FFNDim) );
    AddWhisperExactGelu(NN);
    Blocks[BlockCnt].Fc2 := NN.AddLayer(
      TNNetPointwiseConvLinear.Create(Config.DModel) );
    NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
    if pInferenceOnly then NN.MakeInferenceOnly();
  end;
  SetLength(SliceChannels, 0);
  SetLength(Heads, 0);
end;

// Fills a TNNetLearnedPositionalEmbedding table (SeqLen rows of DModel)
// with the WHISPER sinusoids (modeling_whisper.sinusoids - the original
// OpenAI formula):
//   inv_timescale[c]      = exp(-ln(10000) * c / (DModel/2 - 1))
//   table[pos, c]          = sin(pos * inv_timescale[c])  for c < DModel/2
//   table[pos, DModel/2+c] = cos(pos * inv_timescale[c])
// Same concat-sin|cos layout as Marian but a DIFFERENT timescale exponent:
// c/(half-1) (the LAST column pair reaches 1/10000 exactly) vs Marian's
// 2c/DModel. Real checkpoints carry the tensor; this is the fallback when
// it is absent. HF builds it in float32; these doubles round to the same
// float32 values.
procedure FillWhisperSinusoidalPositions(Layer: TNNetLayer;
  SeqLen, DModel: integer);
var
  PosCnt, ChCnt, Half: integer;
  Angle: double;
begin
  Half := DModel div 2;
  for PosCnt := 0 to SeqLen - 1 do
    for ChCnt := 0 to Half - 1 do
    begin
      Angle := PosCnt * Exp(-Ln(10000.0) * ChCnt / (Half - 1));
      Layer.Neurons[0].Weights.FData[PosCnt * DModel + ChCnt] := Sin(Angle);
      Layer.Neurons[0].Weights.FData[PosCnt * DModel + Half + ChCnt] :=
        Cos(Angle);
    end;
  Layer.FlushWeightCache();
end;

// Loads one Whisper attention sub-block's q/k/v/out projections (HF
// nn.Linear [d_model, d_model]; q/v/out BIASED, k_proj BIAS-FREE) plus its
// PRE-LayerNorm (NormPrefix, weight+bias).
procedure LoadWhisperAttn(Reader: TNNetSafeTensorsReader;
  const Attn: TMarianAttnLayers; const AttnPrefix, NormPrefix: string;
  const Config: TWhisperConfig; Consumed: TStringList);
begin
  LoadLlamaLinearWeights(Reader, Attn.QProj, AttnPrefix + 'q_proj.weight',
    Config.DModel, Config.DModel, 0, -1, 0, AttnPrefix + 'q_proj.bias');
  Consumed.Add(AttnPrefix + 'q_proj.weight');
  Consumed.Add(AttnPrefix + 'q_proj.bias');
  // k_proj has NO bias in Whisper (the loader zeroes the layer's bias).
  LoadLlamaLinearWeights(Reader, Attn.KProj, AttnPrefix + 'k_proj.weight',
    Config.DModel, Config.DModel, 0, -1, 0, '');
  Consumed.Add(AttnPrefix + 'k_proj.weight');
  LoadLlamaLinearWeights(Reader, Attn.VProj, AttnPrefix + 'v_proj.weight',
    Config.DModel, Config.DModel, 0, -1, 0, AttnPrefix + 'v_proj.bias');
  Consumed.Add(AttnPrefix + 'v_proj.weight');
  Consumed.Add(AttnPrefix + 'v_proj.bias');
  LoadLlamaLinearWeights(Reader, Attn.OProj, AttnPrefix + 'out_proj.weight',
    Config.DModel, Config.DModel, 0, -1, 0, AttnPrefix + 'out_proj.bias');
  Consumed.Add(AttnPrefix + 'out_proj.weight');
  Consumed.Add(AttnPrefix + 'out_proj.bias');
  LoadLayerNormWeights(Reader, Attn.Norm, NormPrefix + '.weight',
    NormPrefix + '.bias', Config.DModel);
  Consumed.Add(NormPrefix + '.weight');
  Consumed.Add(NormPrefix + '.bias');
end;

// Loads one full Whisper stack: per-block self-attention (+
// cross-attention in the decoder) with their pre-norms, plus fc1/fc2 and
// the block's final_layer_norm (the FFN's pre-norm).
procedure LoadWhisperStack(Reader: TNNetSafeTensorsReader;
  const Blocks: TMarianBlockArray; const StackPrefix: string;
  const Config: TWhisperConfig; FFNDim: integer; IsDecoder: boolean;
  Consumed: TStringList);
var
  BlockCnt: integer;
  BP: string;
begin
  for BlockCnt := 0 to High(Blocks) do
  begin
    BP := StackPrefix + IntToStr(BlockCnt) + '.';
    LoadWhisperAttn(Reader, Blocks[BlockCnt].SelfAttn, BP + 'self_attn.',
      BP + 'self_attn_layer_norm', Config, Consumed);
    if IsDecoder then
      LoadWhisperAttn(Reader, Blocks[BlockCnt].CrossAttn,
        BP + 'encoder_attn.', BP + 'encoder_attn_layer_norm', Config,
        Consumed);
    LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Fc1,
      BP + 'fc1.weight', Config.DModel, FFNDim, 0, -1, 0, BP + 'fc1.bias');
    Consumed.Add(BP + 'fc1.weight');
    Consumed.Add(BP + 'fc1.bias');
    LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].Fc2,
      BP + 'fc2.weight', FFNDim, Config.DModel, 0, -1, 0, BP + 'fc2.bias');
    Consumed.Add(BP + 'fc2.weight');
    Consumed.Add(BP + 'fc2.bias');
    LoadLayerNormWeights(Reader, Blocks[BlockCnt].FFNNorm,
      BP + 'final_layer_norm.weight', BP + 'final_layer_norm.bias',
      Config.DModel);
    Consumed.Add(BP + 'final_layer_norm.weight');
    Consumed.Add(BP + 'final_layer_norm.bias');
  end;
end;

// Loads one Whisper frontend conv (HF nn.Conv1d, weight [OutDim, InDim, 3],
// biased) into a TNNetConvolutionLinear built with a (3,1) kernel on the
// (SeqLen, 1, InDim) sequence grid. Tap k of HF's kernel multiplies the
// SAME padded input frame as tap x=k of the (3,1) kernel, so the layout
// map is Neurons[o].Weights[x*InDim + i] = W[o, i, x].
procedure LoadWhisperConv1D(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName, BName: string; InDim, OutDim: integer;
  Consumed: TStringList);
var
  W, B: TNNetVolume;
  o, i, kk: integer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('Whisper import: missing tensor "' + WName + '".');
  if not Reader.HasTensor(BName) then
    ImportError('Whisper import: missing tensor "' + BName + '".');
  if (Reader.DimCount(WName) <> 3) or
     (Reader.DimSize(WName, 0) <> OutDim) or
     (Reader.DimSize(WName, 1) <> InDim) or
     (Reader.DimSize(WName, 2) <> 3) then
    ImportError('Whisper import: "' + WName + '" must have shape [' +
      IntToStr(OutDim) + ', ' + IntToStr(InDim) + ', 3] (nn.Conv1d ' +
      'stores [out, in, kernel]), got ' + Reader.ShapeAsString(WName));
  if (Reader.DimCount(BName) <> 1) or
     (Reader.DimSize(BName, 0) <> OutDim) then
    ImportError('Whisper import: "' + BName + '" must have shape [' +
      IntToStr(OutDim) + '], got ' + Reader.ShapeAsString(BName));
  if Layer.Neurons.Count <> OutDim then
    ImportError('Whisper import: internal error - conv layer for "' +
      WName + '" has ' + IntToStr(Layer.Neurons.Count) +
      ' neurons, expected ' + IntToStr(OutDim) + '.');
  if Layer.Neurons[0].Weights.Size <> 3 * InDim then
    ImportError('Whisper import: internal error - conv neuron for "' +
      WName + '" has ' + IntToStr(Layer.Neurons[0].Weights.Size) +
      ' weights, expected ' + IntToStr(3 * InDim) + '.');
  W := TNNetVolume.Create;
  B := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, W);
    Reader.LoadTensorFlat(BName, B);
    for o := 0 to OutDim - 1 do
    begin
      for kk := 0 to 2 do
        for i := 0 to InDim - 1 do
          Layer.Neurons[o].Weights.FData[kk * InDim + i] :=
            W.FData[(o * InDim + i) * 3 + kk];
      Layer.Neurons[o].BiasWeight := B.FData[o];
    end;
  finally
    B.Free;
    W.Free;
  end;
  Layer.FlushWeightCache();
  Consumed.Add(WName);
  Consumed.Add(BName);
end;

procedure BuildWhisperFromSafeTensorsWithConfig(const FileName: string;
  var Config: TWhisperConfig; out EncoderNet, DecoderNet: TNNet;
  DecSeqLen: integer; pInferenceOnly: boolean = false);
var
  Reader: TNNetSafeTensorsReader;
  Enc, Dec: TNNet;
  EncBlocks, DecBlocks: TMarianBlockArray;
  DecEmbed, EncPos, DecPos: TNNetLayer;
  Conv1, Conv2: TNNetLayer;
  EncFinalNorm, DecFinalNorm: TNNetLayer;
  DecTokenInput, EncStates, LMHead: TNNetLayer;
  Consumed: TStringList;
  Tmp: TNNetVolume;
  i, j, NumInputFrames: integer;
  EmbedScale: TNeuralFloat;
  TensorNameStr: string;
begin
  EncoderNet := nil;
  DecoderNet := nil;
  // ---------------- Config validation ----------------
  if DecSeqLen < 1 then
    ImportError('Whisper import: DecSeqLen must be >= 1, got ' +
      IntToStr(DecSeqLen) + '.');
  if DecSeqLen > Config.MaxTargetPositions then
    ImportError('Whisper import: DecSeqLen must not exceed ' +
      'max_target_positions = ' + IntToStr(Config.MaxTargetPositions) + '.');
  if Odd(Config.DModel) or (Config.DModel < 4) then
    ImportError('Whisper import: d_model must be EVEN and >= 4 (the ' +
      'concat sin|cos sinusoidal table), got ' +
      IntToStr(Config.DModel) + '.');
  if (Config.EncoderHeads < 1) or
     ((Config.DModel mod Config.EncoderHeads) <> 0) then
    ImportError('Whisper import: encoder_attention_heads must divide ' +
      'd_model, got ' + IntToStr(Config.EncoderHeads) + ' heads for ' +
      'd_model ' + IntToStr(Config.DModel) + '.');
  if (Config.DecoderHeads < 1) or
     ((Config.DModel mod Config.DecoderHeads) <> 0) then
    ImportError('Whisper import: decoder_attention_heads must divide ' +
      'd_model, got ' + IntToStr(Config.DecoderHeads) + ' heads for ' +
      'd_model ' + IntToStr(Config.DModel) + '.');
  if Config.MaxSourcePositions < 2 then
    ImportError('Whisper import: max_source_positions must be >= 2, got ' +
      IntToStr(Config.MaxSourcePositions) + '.');
  NumInputFrames := 2 * Config.MaxSourcePositions;
  Reader := CreatePretrainedTensorReader(FileName);
  Enc := nil;
  Dec := nil;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      if not Reader.HasTensor('model.decoder.embed_tokens.weight') then
        ImportError('Whisper import: "model.decoder.embed_tokens.weight" ' +
          'not found in ' + Reader.FileName + ' - not a Whisper checkpoint?');
      if (Reader.DimCount('model.decoder.embed_tokens.weight') <> 2) or
         (Reader.DimSize('model.decoder.embed_tokens.weight', 0) <>
          Config.VocabSize) or
         (Reader.DimSize('model.decoder.embed_tokens.weight', 1) <>
          Config.DModel) then
        ImportError('Whisper import: model.decoder.embed_tokens.weight ' +
          'must have shape [' + IntToStr(Config.VocabSize) + ', ' +
          IntToStr(Config.DModel) + '], got ' +
          Reader.ShapeAsString('model.decoder.embed_tokens.weight'));
      if not Reader.HasTensor('model.decoder.embed_positions.weight') then
        ImportError('Whisper import: ' +
          '"model.decoder.embed_positions.weight" (the LEARNED decoder ' +
          'positions) not found in ' + Reader.FileName + '.');
      if (Reader.DimCount('model.decoder.embed_positions.weight') <> 2) or
         (Reader.DimSize('model.decoder.embed_positions.weight', 0) <>
          Config.MaxTargetPositions) or
         (Reader.DimSize('model.decoder.embed_positions.weight', 1) <>
          Config.DModel) then
        ImportError('Whisper import: model.decoder.embed_positions.weight ' +
          'must have shape [' + IntToStr(Config.MaxTargetPositions) + ', ' +
          IntToStr(Config.DModel) + '], got ' +
          Reader.ShapeAsString('model.decoder.embed_positions.weight'));

      // ---------------- Encoder architecture ----------------
      // Input: the log-mel volume (2*max_source_positions frames along
      // SizeX, num_mel_bins along Depth) from ComputeWhisperLogMel. The
      // conv frontend (SAME padding via X-only TNNetPadXY; the (3,1)
      // kernel clamps to the sequence's SizeY=1) halves the frames with
      // conv2's stride 2, then the FIXED sinusoidal positions are added.
      Enc := TNNet.Create();
      Enc.AddLayer( TNNetInput.Create(NumInputFrames, 1,
        Config.NumMelBins, 1) );
      Enc.AddLayer( TNNetPadXY.Create(1, 0) );
      Conv1 := Enc.AddLayer( TNNetConvolutionLinear.Create(
        Config.DModel, 3, {Padding=}0, {Stride=}1) );
      AddWhisperExactGelu(Enc);
      Enc.AddLayer( TNNetPadXY.Create(1, 0) );
      Conv2 := Enc.AddLayer( TNNetConvolutionLinear.Create(
        Config.DModel, 3, {Padding=}0, {Stride=}2) );
      AddWhisperExactGelu(Enc);
      EncPos := Enc.AddLayer(
        TNNetLearnedPositionalEmbedding.Create(Config.MaxSourcePositions) );
      if pInferenceOnly then Enc.MakeInferenceOnly();
      BuildWhisperStackBlocks(Enc, Config, Config.EncoderLayers,
        Config.EncoderHeads, Config.EncoderFFNDim, {IsDecoder=}false,
        nil, EncBlocks, pInferenceOnly);
      // PRE-norm stack: the FINAL LayerNorm closes the encoder.
      EncFinalNorm := Enc.AddLayer(
        TNNetTokenLayerNorm.Create(MarianLayerNormEps) );
      if pInferenceOnly then Enc.MakeInferenceOnly();

      // ---------------- Decoder architecture ----------------
      // TWO inputs: Layers[0] = decoder token ids (what Compute feeds);
      // the second TNNetInput holds the encoder hidden states and is
      // filled MANUALLY before Compute (RunT5 does it - the convention is
      // shared with the T5/Marian importers).
      Dec := TNNet.Create();
      DecTokenInput := Dec.AddLayer( TNNetInput.Create(DecSeqLen) );
      EncStates := Dec.AddLayerAfter(
        TNNetInput.Create(Config.MaxSourcePositions, 1, Config.DModel, 1),
        0);
      DecEmbed := Dec.AddLayerAfter( TNNetEmbedding.Create(
        Config.VocabSize, Config.DModel, {EncodeZero=}1), DecTokenInput);
      DecPos := Dec.AddLayer(
        TNNetLearnedPositionalEmbedding.Create(DecSeqLen) );
      if pInferenceOnly then Dec.MakeInferenceOnly();
      BuildWhisperStackBlocks(Dec, Config, Config.DecoderLayers,
        Config.DecoderHeads, Config.DecoderFFNDim, {IsDecoder=}true,
        EncStates, DecBlocks, pInferenceOnly);
      DecFinalNorm := Dec.AddLayer(
        TNNetTokenLayerNorm.Create(MarianLayerNormEps) );
      LMHead := Dec.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then Dec.MakeInferenceOnly();

      // ---------------- Weights ----------------
      LoadWhisperConv1D(Reader, Conv1, 'model.encoder.conv1.weight',
        'model.encoder.conv1.bias', Config.NumMelBins, Config.DModel,
        Consumed);
      LoadWhisperConv1D(Reader, Conv2, 'model.encoder.conv2.weight',
        'model.encoder.conv2.bias', Config.DModel, Config.DModel,
        Consumed);
      Tmp := TNNetVolume.Create;
      try
        // Token embedding (+ the tied, UNSCALED head). scale_embedding
        // folds sqrt(d_model) into the embedding rows like Marian; every
        // released Whisper has it false.
        Reader.LoadTensorFlat('model.decoder.embed_tokens.weight', Tmp);
        if DecEmbed.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('Whisper import: embed_tokens element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table ' +
            'size ' + IntToStr(DecEmbed.Neurons[0].Weights.Size) + '.');
        if Config.ScaleEmbedding then
          EmbedScale := Sqrt(Config.DModel)
        else
          EmbedScale := 1.0;
        for i := 0 to Tmp.Size - 1 do
          DecEmbed.Neurons[0].Weights.FData[i] := EmbedScale * Tmp.FData[i];
        DecEmbed.FlushWeightCache();
        Consumed.Add('model.decoder.embed_tokens.weight');
        // Tied head: logits = h . embed_tokens^T (proj_out is bias-free
        // and its tensor is absent from real checkpoints).
        for j := 0 to Config.VocabSize - 1 do
        begin
          for i := 0 to Config.DModel - 1 do
            LMHead.Neurons[j].Weights.FData[i] :=
              Tmp.FData[j * Config.DModel + i];
          LMHead.Neurons[j].BiasWeight := 0;
        end;
        LMHead.FlushWeightCache();
        if Reader.HasTensor('proj_out.weight') then
          Consumed.Add('proj_out.weight');
        // LEARNED decoder positions: the first DecSeqLen rows of the
        // [max_target_positions, d_model] table.
        Reader.LoadTensorFlat('model.decoder.embed_positions.weight', Tmp);
        for i := 0 to DecSeqLen * Config.DModel - 1 do
          DecPos.Neurons[0].Weights.FData[i] := Tmp.FData[i];
        DecPos.FlushWeightCache();
        Consumed.Add('model.decoder.embed_positions.weight');
        // FIXED sinusoidal encoder positions: real checkpoints save the
        // table - load it when present, regenerate it otherwise.
        if Reader.HasTensor('model.encoder.embed_positions.weight') then
        begin
          if (Reader.DimCount('model.encoder.embed_positions.weight') <> 2)
             or (Reader.DimSize('model.encoder.embed_positions.weight', 0)
                 <> Config.MaxSourcePositions)
             or (Reader.DimSize('model.encoder.embed_positions.weight', 1)
                 <> Config.DModel) then
            ImportError('Whisper import: ' +
              'model.encoder.embed_positions.weight must have shape [' +
              IntToStr(Config.MaxSourcePositions) + ', ' +
              IntToStr(Config.DModel) + '], got ' + Reader.ShapeAsString(
              'model.encoder.embed_positions.weight'));
          Reader.LoadTensorFlat('model.encoder.embed_positions.weight',
            Tmp);
          for i := 0 to Config.MaxSourcePositions * Config.DModel - 1 do
            EncPos.Neurons[0].Weights.FData[i] := Tmp.FData[i];
          EncPos.FlushWeightCache();
          Consumed.Add('model.encoder.embed_positions.weight');
        end
        else
          FillWhisperSinusoidalPositions(EncPos,
            Config.MaxSourcePositions, Config.DModel);
      finally
        Tmp.Free;
      end;
      LoadWhisperStack(Reader, EncBlocks, 'model.encoder.layers.', Config,
        Config.EncoderFFNDim, {IsDecoder=}false, Consumed);
      LoadWhisperStack(Reader, DecBlocks, 'model.decoder.layers.', Config,
        Config.DecoderFFNDim, {IsDecoder=}true, Consumed);
      LoadLayerNormWeights(Reader, EncFinalNorm,
        'model.encoder.layer_norm.weight', 'model.encoder.layer_norm.bias',
        Config.DModel);
      Consumed.Add('model.encoder.layer_norm.weight');
      Consumed.Add('model.encoder.layer_norm.bias');
      LoadLayerNormWeights(Reader, DecFinalNorm,
        'model.decoder.layer_norm.weight', 'model.decoder.layer_norm.bias',
        Config.DModel);
      Consumed.Add('model.decoder.layer_norm.weight');
      Consumed.Add('model.decoder.layer_norm.bias');

      // ---------------- Unexpected-tensor check ----------------
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        ImportError('Whisper import: unexpected tensor "' + TensorNameStr +
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

procedure BuildWhisperFromSafeTensors(const FileName: string;
  out EncoderNet, DecoderNet: TNNet; out Config: TWhisperConfig;
  DecSeqLen: integer; pInferenceOnly: boolean = false;
  const ConfigFileName: string = '');
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadWhisperConfigFromJSONFile(ConfigPath);
  BuildWhisperFromSafeTensorsWithConfig(FileName, Config, EncoderNet,
    DecoderNet, DecSeqLen, pInferenceOnly);
end;

function ReadRWKVConfigFromJSONFile(const FileName: string): TRWKVConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType: string;
  Field: TJSONData;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('RWKV import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('RWKV import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

begin
  if not FileExists(FileName) then
    ImportError('RWKV import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('RWKV import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('RWKV import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'rwkv');
    if ModelType <> 'rwkv' then
      ImportError('RWKV import: config model_type is "' + ModelType +
        '" - expected "rwkv" (see BuildFromPretrained for the full ' +
        'dispatch).');
    Result.ModelType := ModelType;
    Result.HiddenSize := RequiredInt('hidden_size');
    Result.NumLayers := RequiredInt('num_hidden_layers');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.AttentionHiddenSize :=
      Obj.Get('attention_hidden_size', Result.HiddenSize);
    if Result.AttentionHiddenSize <= 0 then
      Result.AttentionHiddenSize := Result.HiddenSize;
    // intermediate_size is null in the published RWKV-4 configs: the HF
    // default 4 * hidden_size applies.
    Field := Obj.Find('intermediate_size');
    if (Field = nil) or Field.IsNull then
      Result.IntermediateSize := 4 * Result.HiddenSize
    else
      Result.IntermediateSize := RequiredInt('intermediate_size');
    Result.ContextLength := Obj.Get('context_length', 1024);
    if Result.ContextLength < 1 then
      ImportError('RWKV import: config context_length must be >= 1, got ' +
        IntToStr(Result.ContextLength) + '.');
    Result.LayerNormEps := Obj.Get('layer_norm_epsilon', 1.0e-5);
    // rescale_every is an inference-time float16 trick that LayerNorm
    // scale-invariance makes a mathematical identity - read, then ignored
    // (raw weights are exact). See the RWKV-4 IMPORT section.
    Result.RescaleEvery := Obj.Get('rescale_every', 6);
    Result.TieWordEmbeddings := Obj.Get('tie_word_embeddings', False);
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function RWKVConfigToString(const Config: TRWKVConfig): string;
begin
  Result := 'rwkv config: layers=' + IntToStr(Config.NumLayers) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', attention_hidden=' + IntToStr(Config.AttentionHiddenSize) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', context=' + IntToStr(Config.ContextLength) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', ln_eps=' + FloatToStr(Config.LayerNormEps) +
    ', rescale_every=' + IntToStr(Config.RescaleEvery) +
    ', tied=' + BoolToStr(Config.TieWordEmbeddings, true);
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

type
  TRWKVBlockLayers = record
    LN1, AttShiftK, AttShiftV, AttShiftR: TNNetLayer;
    AttKey, AttValue, AttRecept, WKV, AttOut: TNNetLayer;
    LN2, FFShiftK, FFShiftR, FFKey, FFRecept, FFValue: TNNetLayer;
  end;

function BuildRWKVFromSafeTensorsWithConfig(const FileName: string;
  var Config: TRWKVConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TRWKVBlockLayers;
  EmbeddingLayer, PreLN, FinalLN, LMHead: TNNetLayer;
  BranchInput, KV, RSig, Gated: TNNetLayer;
  BlockCnt, SeqLen, i, j: integer;
  Tmp: TNNetVolume;
  BlockPrefix, AttPrefix, FFPrefix, TensorNameStr: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

  // Loads a per-channel vector tensor (any dim layout, e.g. the [1,1,C]
  // time_mix vectors or the [C] time_decay/time_first vectors) into neuron
  // NeuronIdx of Layer, optionally transformed element-wise.
  procedure LoadChannelVector(Layer: TNNetLayer; NeuronIdx: integer;
    const TName: string; Channels: integer);
  var
    d: integer;
  begin
    if not Reader.HasTensor(TName) then
      ImportError('RWKV import: missing tensor "' + TName + '" in ' +
        Reader.FileName);
    Reader.LoadTensorFlat(TName, Tmp);
    if Tmp.Size <> Channels then
      ImportError('RWKV import: "' + TName + '" must carry ' +
        IntToStr(Channels) + ' elements, got ' +
        Reader.ShapeAsString(TName));
    for d := 0 to Channels - 1 do
      Layer.Neurons[NeuronIdx].Weights.FData[d] := Tmp.FData[d];
    Layer.FlushWeightCache();
    MarkConsumed(TName);
  end;

  // time_decay -> TNNetWKV w_raw: the checkpoint's per-step decay factor is
  // exp(-exp(time_decay)) while TNNetWKV computes exp(-softplus(w_raw)), so
  // w_raw = invsoftplus(exp(time_decay)) = ln(exp(exp(time_decay)) - 1).
  // EXACT for every real time_decay (exp() is positive and softplus is a
  // bijection onto the positives); for x > 30 w_raw = x matches the layer's
  // own softplus large-input shortcut bit-for-bit, and for tiny x the
  // ln(x) limit avoids the exp(x)-1 float underflow.
  procedure LoadWKVDecay(Layer: TNNetLayer; const TName: string;
    Channels: integer);
  var
    d: integer;
    x, wraw: double;
  begin
    if not Reader.HasTensor(TName) then
      ImportError('RWKV import: missing tensor "' + TName + '" in ' +
        Reader.FileName);
    Reader.LoadTensorFlat(TName, Tmp);
    if Tmp.Size <> Channels then
      ImportError('RWKV import: "' + TName + '" must carry ' +
        IntToStr(Channels) + ' elements, got ' +
        Reader.ShapeAsString(TName));
    for d := 0 to Channels - 1 do
    begin
      x := Exp(Tmp.FData[d]);            // the checkpoint's effective w > 0
      if x > 30 then wraw := x
      else if x < 1e-7 then wraw := Ln(x)
      else wraw := Ln(Exp(x) - 1.0);
      Layer.Neurons[0].Weights.FData[d] := wraw;
    end;
    Layer.FlushWeightCache();
    MarkConsumed(TName);
  end;

begin
  Reader := CreatePretrainedTensorReader(FileName);
  NN := nil;
  Tmp := TNNetVolume.Create;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      // ---------------- Config validation & prefix detection -------------
      if Config.NumLayers < 1 then
        ImportError('RWKV import: num_hidden_layers must be >= 1.');
      if Reader.HasTensor('rwkv.embeddings.weight') then
        Config.Prefix := 'rwkv.'
      else if Reader.HasTensor('embeddings.weight') then
        Config.Prefix := ''
      else
        ImportError('RWKV import: neither "rwkv.embeddings.weight" nor ' +
          '"embeddings.weight" found in ' + Reader.FileName +
          ' - not an RWKV checkpoint?');
      if (Reader.DimCount(Config.Prefix + 'embeddings.weight') <> 2) or
         (Reader.DimSize(Config.Prefix + 'embeddings.weight', 0) <>
          Config.VocabSize) or
         (Reader.DimSize(Config.Prefix + 'embeddings.weight', 1) <>
          Config.HiddenSize) then
        ImportError('RWKV import: embeddings.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.HiddenSize) +
          '], got ' +
          Reader.ShapeAsString(Config.Prefix + 'embeddings.weight'));
      if (not Config.TieWordEmbeddings) and
         (not Reader.HasTensor('head.weight')) then
        ImportError('RWKV import: config says tie_word_embeddings=false ' +
          'but "head.weight" is missing from ' + Reader.FileName + '.');
      if pSeqLen <= 0 then SeqLen := Config.ContextLength
      else SeqLen := pSeqLen;

      // ---------------- Architecture ----------------
      // No positional embedding: RWKV's only notion of order is the
      // recurrence itself (token shift + the WKV decay scan).
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      // EncodeZero=1: token id 0 is a real token, not padding.
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      // Block 0's extra embedding LayerNorm ("ln0" / HF pre_ln).
      PreLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      SetLength(Blocks, Config.NumLayers);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // ---- Time-mix sub-block: x := x + Att(ln1(x)). The importer
        // wires the block INLINE (not via AddRWKVTimeMix/AddRWKVBlock)
        // because the checkpoint carries SEPARATE time_mix vectors per
        // k/v/r stream - one TNNetTokenShift each.
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].LN1 := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        Blocks[BlockCnt].AttShiftK := NN.AddLayer( TNNetTokenShift.Create() );
        Blocks[BlockCnt].AttKey := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.AttentionHiddenSize, 1) );
        Blocks[BlockCnt].AttShiftV := NN.AddLayerAfter(
          TNNetTokenShift.Create(), Blocks[BlockCnt].LN1 );
        Blocks[BlockCnt].AttValue := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.AttentionHiddenSize, 1) );
        Blocks[BlockCnt].AttShiftR := NN.AddLayerAfter(
          TNNetTokenShift.Create(), Blocks[BlockCnt].LN1 );
        Blocks[BlockCnt].AttRecept := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.AttentionHiddenSize, 1) );
        RSig := NN.AddLayer( TNNetSigmoid.Create() );
        // k|v concatenated on the Depth axis -> the 2*C tensor TNNetWKV
        // splits (first C channels key, next C value).
        KV := NN.AddLayer( TNNetDeepConcat.Create(
          [Blocks[BlockCnt].AttKey, Blocks[BlockCnt].AttValue]) );
        Blocks[BlockCnt].WKV := NN.AddLayerAfter( TNNetWKV.Create(), KV );
        Gated := NN.AddLayer( TNNetCellMulByCell.Create(
          RSig, Blocks[BlockCnt].WKV) );
        Blocks[BlockCnt].AttOut := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize, 1), Gated );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        // ---- Channel-mix sub-block: x := x + FFN(ln2(x)) with squared-
        // ReLU keying and sigmoid receptance gating.
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].LN2 := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        Blocks[BlockCnt].FFShiftK := NN.AddLayer( TNNetTokenShift.Create() );
        Blocks[BlockCnt].FFKey := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.IntermediateSize, 1) );
        NN.AddLayer( TNNetSquaredReLU.Create() );
        Blocks[BlockCnt].FFValue := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize, 1) );
        Blocks[BlockCnt].FFShiftR := NN.AddLayerAfter(
          TNNetTokenShift.Create(), Blocks[BlockCnt].LN2 );
        Blocks[BlockCnt].FFRecept := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize, 1) );
        RSig := NN.AddLayer( TNNetSigmoid.Create() );
        Gated := NN.AddLayer( TNNetCellMulByCell.Create(
          RSig, Blocks[BlockCnt].FFValue) );
        NN.AddLayer( TNNetSum.Create([Gated, BranchInput]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

      // ---------------- Weights ----------------
      Reader.LoadTensorFlat(Config.Prefix + 'embeddings.weight', Tmp);
      if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
        ImportError('RWKV import: embeddings.weight element count ' +
          IntToStr(Tmp.Size) + ' does not match the embedding table ' +
          'size ' + IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
      EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
      EmbeddingLayer.FlushWeightCache();
      MarkConsumed(Config.Prefix + 'embeddings.weight');
      if Config.TieWordEmbeddings then
      begin
        // Tied LM head: logits = h . E^T (rows copied, bias-free).
        EnsureWritableImportWeights(LMHead);
        for j := 0 to Config.VocabSize - 1 do
        begin
          for i := 0 to Config.HiddenSize - 1 do
            LMHead.Neurons[j].Weights.FData[i] :=
              Tmp.FData[j * Config.HiddenSize + i];
          LMHead.Neurons[j].BiasWeight := 0;
        end;
        LMHead.FlushWeightCache();
        if Reader.HasTensor('head.weight') then
          MarkConsumed('head.weight');
      end
      else
      begin
        LoadLlamaLinearWeights(Reader, LMHead, 'head.weight',
          Config.HiddenSize, Config.VocabSize);
        MarkConsumed('head.weight');
      end;
      LoadLayerNormWeights(Reader, PreLN,
        Config.Prefix + 'blocks.0.pre_ln.weight',
        Config.Prefix + 'blocks.0.pre_ln.bias', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'blocks.0.pre_ln.weight');
      MarkConsumed(Config.Prefix + 'blocks.0.pre_ln.bias');
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        BlockPrefix := Config.Prefix + 'blocks.' + IntToStr(BlockCnt) + '.';
        AttPrefix := BlockPrefix + 'attention.';
        FFPrefix := BlockPrefix + 'feed_forward.';
        LoadLayerNormWeights(Reader, Blocks[BlockCnt].LN1,
          BlockPrefix + 'ln1.weight', BlockPrefix + 'ln1.bias',
          Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'ln1.weight');
        MarkConsumed(BlockPrefix + 'ln1.bias');
        LoadLayerNormWeights(Reader, Blocks[BlockCnt].LN2,
          BlockPrefix + 'ln2.weight', BlockPrefix + 'ln2.bias',
          Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'ln2.weight');
        MarkConsumed(BlockPrefix + 'ln2.bias');
        // Token-shift lerp vectors (HF stores them [1,1,hidden]).
        LoadChannelVector(Blocks[BlockCnt].AttShiftK, 0,
          AttPrefix + 'time_mix_key', Config.HiddenSize);
        LoadChannelVector(Blocks[BlockCnt].AttShiftV, 0,
          AttPrefix + 'time_mix_value', Config.HiddenSize);
        LoadChannelVector(Blocks[BlockCnt].AttShiftR, 0,
          AttPrefix + 'time_mix_receptance', Config.HiddenSize);
        LoadChannelVector(Blocks[BlockCnt].FFShiftK, 0,
          FFPrefix + 'time_mix_key', Config.HiddenSize);
        LoadChannelVector(Blocks[BlockCnt].FFShiftR, 0,
          FFPrefix + 'time_mix_receptance', Config.HiddenSize);
        // WKV decay (softplus-inverted, see LoadWKVDecay) + bonus u.
        LoadWKVDecay(Blocks[BlockCnt].WKV, AttPrefix + 'time_decay',
          Config.AttentionHiddenSize);
        LoadChannelVector(Blocks[BlockCnt].WKV, 1,
          AttPrefix + 'time_first', Config.AttentionHiddenSize);
        // Bias-free nn.Linear projections.
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].AttKey,
          AttPrefix + 'key.weight', Config.HiddenSize,
          Config.AttentionHiddenSize);
        MarkConsumed(AttPrefix + 'key.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].AttValue,
          AttPrefix + 'value.weight', Config.HiddenSize,
          Config.AttentionHiddenSize);
        MarkConsumed(AttPrefix + 'value.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].AttRecept,
          AttPrefix + 'receptance.weight', Config.HiddenSize,
          Config.AttentionHiddenSize);
        MarkConsumed(AttPrefix + 'receptance.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].AttOut,
          AttPrefix + 'output.weight', Config.AttentionHiddenSize,
          Config.HiddenSize);
        MarkConsumed(AttPrefix + 'output.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].FFKey,
          FFPrefix + 'key.weight', Config.HiddenSize,
          Config.IntermediateSize);
        MarkConsumed(FFPrefix + 'key.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].FFRecept,
          FFPrefix + 'receptance.weight', Config.HiddenSize,
          Config.HiddenSize);
        MarkConsumed(FFPrefix + 'receptance.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].FFValue,
          FFPrefix + 'value.weight', Config.IntermediateSize,
          Config.HiddenSize);
        MarkConsumed(FFPrefix + 'value.weight');
        // Re-quantize the block just refilled with checkpoint weights.
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      LoadLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'ln_out.weight', Config.Prefix + 'ln_out.bias',
        Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'ln_out.weight');
      MarkConsumed(Config.Prefix + 'ln_out.bias');

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      // ---------------- Unexpected-tensor check ----------------
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        ImportError('RWKV import: unexpected tensor "' + TensorNameStr +
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
    Tmp.Free;
    Reader.Free;
  end;
end;

function BuildRWKVFromSafeTensorsEx(const FileName: string;
  out Config: TRWKVConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadRWKVConfigFromJSONFile(ConfigPath);
  Result := BuildRWKVFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, pQuantizeInt8);
end;

function BuildRWKVFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TRWKVConfig;
begin
  Result := BuildRWKVFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

function ReadMambaConfigFromJSONFile(const FileName: string): TMambaConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ModelType: string;
  Field: TJSONData;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('Mamba import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('Mamba import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

begin
  if not FileExists(FileName) then
    ImportError('Mamba import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('Mamba import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('Mamba import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'mamba');
    if ModelType <> 'mamba' then
      ImportError('Mamba import: config model_type is "' + ModelType +
        '" - expected "mamba" (see BuildFromPretrained for the full ' +
        'dispatch).');
    Result.ModelType := ModelType;
    Result.HiddenSize := RequiredInt('hidden_size');
    Result.NumLayers := RequiredInt('num_hidden_layers');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.StateSize := Obj.Get('state_size', 16);
    if Result.StateSize < 1 then
      ImportError('Mamba import: config state_size must be >= 1, got ' +
        IntToStr(Result.StateSize) + '.');
    // time_step_rank is "auto" (= ceil(hidden_size/16), the HF default and
    // the published-checkpoint value) or an explicit integer.
    Field := Obj.Find('time_step_rank');
    if (Field = nil) or Field.IsNull or
       ((Field is TJSONString) and (Field.AsString = 'auto')) then
      Result.TimeStepRank := (Result.HiddenSize + 15) div 16
    else
      Result.TimeStepRank := RequiredInt('time_step_rank');
    Result.Expand := Obj.Get('expand', 2);
    if Result.Expand < 1 then
      ImportError('Mamba import: config expand must be >= 1, got ' +
        IntToStr(Result.Expand) + '.');
    // intermediate_size (= d_inner) is derived in MambaConfig; honor an
    // explicit value when present.
    Field := Obj.Find('intermediate_size');
    if (Field = nil) or Field.IsNull then
      Result.DInner := Result.Expand * Result.HiddenSize
    else
      Result.DInner := RequiredInt('intermediate_size');
    Result.ConvKernel := Obj.Get('conv_kernel', 4);
    if Result.ConvKernel < 1 then
      ImportError('Mamba import: config conv_kernel must be >= 1, got ' +
        IntToStr(Result.ConvKernel) + '.');
    Result.UseConvBias := Obj.Get('use_conv_bias', True);
    Result.UseBias := Obj.Get('use_bias', False);
    Result.LayerNormEps := Obj.Get('layer_norm_epsilon', 1.0e-5);
    Result.TieWordEmbeddings := Obj.Get('tie_word_embeddings', True);
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function MambaConfigToString(const Config: TMambaConfig): string;
begin
  Result := 'mamba config: layers=' + IntToStr(Config.NumLayers) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', d_inner=' + IntToStr(Config.DInner) +
    ', d_state=' + IntToStr(Config.StateSize) +
    ', dt_rank=' + IntToStr(Config.TimeStepRank) +
    ', conv_kernel=' + IntToStr(Config.ConvKernel) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', ln_eps=' + FloatToStr(Config.LayerNormEps) +
    ', conv_bias=' + BoolToStr(Config.UseConvBias, true) +
    ', bias=' + BoolToStr(Config.UseBias, true) +
    ', tied=' + BoolToStr(Config.TieWordEmbeddings, true);
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

type
  TMambaBlockLayers = record
    Norm, InProj, Conv, Scan, OutProj: TNNetLayer;
  end;

function BuildMambaFromSafeTensorsWithConfig(const FileName: string;
  var Config: TMambaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TMambaBlockLayers;
  EmbeddingLayer, FinalNorm, LMHead: TNNetLayer;
  BranchInput, ScanL, ZGate, Gated: TNNetLayer;
  BlockCnt, SeqLen, i, j: integer;
  Tmp: TNNetVolume;
  MixPrefix, TensorNameStr: string;
  Consumed: TStringList;
  ConvBiasSuppress: integer;
  InBias, OutBias: string;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

  procedure RequireShape2(const TName: string; Rows, Cols: integer);
  begin
    if not Reader.HasTensor(TName) then
      ImportError('Mamba import: missing tensor "' + TName + '" in ' +
        Reader.FileName);
    if (Reader.DimCount(TName) <> 2) or
       (Reader.DimSize(TName, 0) <> Rows) or
       (Reader.DimSize(TName, 1) <> Cols) then
      ImportError('Mamba import: "' + TName + '" must have shape [' +
        IntToStr(Rows) + ', ' + IntToStr(Cols) + '], got ' +
        Reader.ShapeAsString(TName));
  end;

  // Loads a per-channel vector tensor into neuron NeuronIdx of Layer.
  procedure LoadChannelVector(Layer: TNNetLayer; NeuronIdx: integer;
    const TName: string; Channels: integer);
  var
    d: integer;
  begin
    if not Reader.HasTensor(TName) then
      ImportError('Mamba import: missing tensor "' + TName + '" in ' +
        Reader.FileName);
    Reader.LoadTensorFlat(TName, Tmp);
    if Tmp.Size <> Channels then
      ImportError('Mamba import: "' + TName + '" must carry ' +
        IntToStr(Channels) + ' elements, got ' +
        Reader.ShapeAsString(TName));
    for d := 0 to Channels - 1 do
      Layer.Neurons[NeuronIdx].Weights.FData[d] := Tmp.FData[d];
    Layer.FlushWeightCache();
    MarkConsumed(TName);
  end;

  // conv1d.weight [d_inner, 1, k] (depthwise: one k-tap filter per
  // channel) + optional conv1d.bias [d_inner] into TNNetDepthwiseConv1D.
  // Tap order maps DIRECTLY: HF pads k-1 zeros on the left and truncates
  // to seq_len, so output[t] = sum_kk w[kk]*x[t-(k-1)+kk] - exactly the
  // layer's causal read.
  procedure LoadDepthwiseConv(Layer: TNNetLayer; const WName, BName: string);
  var
    d, kk: integer;
  begin
    if not Reader.HasTensor(WName) then
      ImportError('Mamba import: missing tensor "' + WName + '" in ' +
        Reader.FileName);
    if (Reader.DimCount(WName) <> 3) or
       (Reader.DimSize(WName, 0) <> Config.DInner) or
       (Reader.DimSize(WName, 1) <> 1) or
       (Reader.DimSize(WName, 2) <> Config.ConvKernel) then
      ImportError('Mamba import: "' + WName + '" must have shape [' +
        IntToStr(Config.DInner) + ', 1, ' + IntToStr(Config.ConvKernel) +
        '], got ' + Reader.ShapeAsString(WName));
    Reader.LoadTensorFlat(WName, Tmp);
    for d := 0 to Config.DInner - 1 do
      for kk := 0 to Config.ConvKernel - 1 do
        Layer.Neurons[d].Weights.FData[kk] :=
          Tmp.FData[d * Config.ConvKernel + kk];
    MarkConsumed(WName);
    if Config.UseConvBias then
    begin
      if not Reader.HasTensor(BName) then
        ImportError('Mamba import: config says use_conv_bias=true but "' +
          BName + '" is missing from ' + Reader.FileName + '.');
      Reader.LoadTensorFlat(BName, Tmp);
      if Tmp.Size <> Config.DInner then
        ImportError('Mamba import: "' + BName + '" must carry ' +
          IntToStr(Config.DInner) + ' elements, got ' +
          Reader.ShapeAsString(BName));
      for d := 0 to Config.DInner - 1 do
        Layer.Neurons[d].BiasWeight := Tmp.FData[d];
      MarkConsumed(BName);
    end;
    Layer.FlushWeightCache();
  end;

  // The selective-scan parameter bundle: x_proj.weight rows split
  // [dt_rank | d_state | d_state] into the folded W_d (with dt_proj),
  // W_B and W_C; dt_proj.bias -> b_d; A_log -> A_raw RAW (identical
  // discretizations, see the MAMBA IMPORT section); D -> e.
  procedure LoadScanWeights(Layer: TNNetLayer; const MixP: string);
  var
    XW, DtW: TNNetVolume;
    d, s, r, j, NS, DI, RK: integer;
    Acc: double;
  begin
    NS := Config.StateSize;
    DI := Config.DInner;
    RK := Config.TimeStepRank;
    RequireShape2(MixP + 'x_proj.weight', RK + 2 * NS, DI);
    RequireShape2(MixP + 'dt_proj.weight', DI, RK);
    XW := TNNetVolume.Create;
    DtW := TNNetVolume.Create;
    try
      Reader.LoadTensorFlat(MixP + 'x_proj.weight', XW);
      Reader.LoadTensorFlat(MixP + 'dt_proj.weight', DtW);
      // W_d = dt_proj.weight @ x_proj.weight[0:dt_rank] - the low-rank
      // delta path folded exactly (double accumulation).
      for d := 0 to DI - 1 do
        for j := 0 to DI - 1 do
        begin
          Acc := 0;
          for r := 0 to RK - 1 do
            Acc := Acc + DtW.FData[d * RK + r] * XW.FData[r * DI + j];
          Layer.Neurons[0].Weights.FData[d * DI + j] := Acc;
        end;
      // W_B / W_C: the next d_state + d_state x_proj rows (shared across
      // channels - TNNetSelectiveSSM's (NS,1,Depth) projections).
      for s := 0 to NS - 1 do
        for j := 0 to DI - 1 do
        begin
          Layer.Neurons[1].Weights.FData[s * DI + j] :=
            XW.FData[(RK + s) * DI + j];
          Layer.Neurons[2].Weights.FData[s * DI + j] :=
            XW.FData[(RK + NS + s) * DI + j];
        end;
    finally
      DtW.Free;
      XW.Free;
    end;
    MarkConsumed(MixP + 'x_proj.weight');
    MarkConsumed(MixP + 'dt_proj.weight');
    LoadChannelVector(Layer, 3, MixP + 'dt_proj.bias', DI);
    // A_log [d_inner, d_state] row-major matches A_raw's (Depth,1,NS)
    // [d*NS + s] layout - a raw copy.
    LoadChannelVector(Layer, 4, MixP + 'A_log', DI * NS);
    LoadChannelVector(Layer, 5, MixP + 'D', DI);
  end;

begin
  Reader := CreatePretrainedTensorReader(FileName);
  NN := nil;
  Tmp := TNNetVolume.Create;
  Consumed := TStringList.Create;
  Consumed.Sorted := True;
  Consumed.Duplicates := dupIgnore;
  try
    try
      // ---------------- Config validation & prefix detection -------------
      if Config.NumLayers < 1 then
        ImportError('Mamba import: num_hidden_layers must be >= 1.');
      if Reader.HasTensor('backbone.embeddings.weight') then
        Config.Prefix := 'backbone.'
      else if Reader.HasTensor('embeddings.weight') then
        Config.Prefix := ''
      else
        ImportError('Mamba import: neither "backbone.embeddings.weight" ' +
          'nor "embeddings.weight" found in ' + Reader.FileName +
          ' - not a Mamba checkpoint?');
      if (Reader.DimCount(Config.Prefix + 'embeddings.weight') <> 2) or
         (Reader.DimSize(Config.Prefix + 'embeddings.weight', 0) <>
          Config.VocabSize) or
         (Reader.DimSize(Config.Prefix + 'embeddings.weight', 1) <>
          Config.HiddenSize) then
        ImportError('Mamba import: embeddings.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.HiddenSize) +
          '], got ' +
          Reader.ShapeAsString(Config.Prefix + 'embeddings.weight'));
      if (not Config.TieWordEmbeddings) and
         (not Reader.HasTensor('lm_head.weight')) then
        ImportError('Mamba import: config says tie_word_embeddings=false ' +
          'but "lm_head.weight" is missing from ' + Reader.FileName + '.');
      // Mamba has no positional limit (and no context_length field).
      if pSeqLen <= 0 then SeqLen := 1024
      else SeqLen := pSeqLen;
      if Config.UseConvBias then ConvBiasSuppress := 0
      else ConvBiasSuppress := 1;

      // ---------------- Architecture ----------------
      // No positional embedding: order enters only through the causal
      // conv1d and the selective scan recurrence.
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      // EncodeZero=1: token id 0 is a real token, not padding.
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      SetLength(Blocks, Config.NumLayers);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // x := x + Mixer(TokenRMSNorm(x)) - the AddMambaBlock wiring,
        // inlined with the checkpoint's PER-TOKEN RMSNorm (HF
        // MambaRMSNorm; the builder's TNNetRMSNorm normalizes the whole
        // volume) and the use_bias/use_conv_bias flags.
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].Norm := NN.AddLayer(
          TNNetTokenRMSNorm.Create(Config.LayerNormEps) );
        // (1) in_proj to 2*d_inner, then the x|z split (x first).
        Blocks[BlockCnt].InProj := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(2 * Config.DInner, 1) );
        NN.AddLayerAfter( TNNetSplitChannels.Create(0, Config.DInner),
          Blocks[BlockCnt].InProj );
        // (2) x path: depthwise causal conv1d + SiLU.
        Blocks[BlockCnt].Conv := NN.AddLayer( TNNetDepthwiseConv1D.Create(
          Config.ConvKernel, {pCausal=}true, ConvBiasSuppress) );
        NN.AddLayer( TNNetSiLU.Create() );
        // (3) the selective scan (owns the delta/B/C projections + D skip).
        Blocks[BlockCnt].Scan := NN.AddLayer(
          TNNetSelectiveSSM.Create(Config.StateSize) );
        ScanL := Blocks[BlockCnt].Scan;
        // (4) z path: SiLU gate, cell-multiplied into the scan output.
        NN.AddLayerAfter(
          TNNetSplitChannels.Create(Config.DInner, Config.DInner),
          Blocks[BlockCnt].InProj );
        ZGate := NN.AddLayer( TNNetSiLU.Create() );
        Gated := NN.AddLayer( TNNetCellMulByCell.Create(ScanL, ZGate) );
        // (5) out_proj back to hidden + residual sum.
        Blocks[BlockCnt].OutProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize, 1), Gated );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      FinalNorm := NN.AddLayer(
        TNNetTokenRMSNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

      // ---------------- Weights ----------------
      Reader.LoadTensorFlat(Config.Prefix + 'embeddings.weight', Tmp);
      if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
        ImportError('Mamba import: embeddings.weight element count ' +
          IntToStr(Tmp.Size) + ' does not match the embedding table ' +
          'size ' + IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
      EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
      EmbeddingLayer.FlushWeightCache();
      MarkConsumed(Config.Prefix + 'embeddings.weight');
      if Config.TieWordEmbeddings then
      begin
        // Tied LM head: logits = h . E^T (rows copied, bias-free) - the
        // published Mamba checkpoints carry no lm_head.weight tensor.
        EnsureWritableImportWeights(LMHead);
        for j := 0 to Config.VocabSize - 1 do
        begin
          for i := 0 to Config.HiddenSize - 1 do
            LMHead.Neurons[j].Weights.FData[i] :=
              Tmp.FData[j * Config.HiddenSize + i];
          LMHead.Neurons[j].BiasWeight := 0;
        end;
        LMHead.FlushWeightCache();
        if Reader.HasTensor('lm_head.weight') then
          MarkConsumed('lm_head.weight');
      end
      else
      begin
        LoadLlamaLinearWeights(Reader, LMHead, 'lm_head.weight',
          Config.HiddenSize, Config.VocabSize);
        MarkConsumed('lm_head.weight');
      end;
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        MixPrefix := Config.Prefix + 'layers.' + IntToStr(BlockCnt) +
          '.mixer.';
        LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].Norm,
          Config.Prefix + 'layers.' + IntToStr(BlockCnt) + '.norm.weight',
          Config.HiddenSize);
        MarkConsumed(Config.Prefix + 'layers.' + IntToStr(BlockCnt) +
          '.norm.weight');
        if Config.UseBias then InBias := MixPrefix + 'in_proj.bias'
        else InBias := '';
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].InProj,
          MixPrefix + 'in_proj.weight', Config.HiddenSize,
          2 * Config.DInner, 0, -1, 0, InBias);
        MarkConsumed(MixPrefix + 'in_proj.weight');
        if InBias <> '' then MarkConsumed(InBias);
        LoadDepthwiseConv(Blocks[BlockCnt].Conv,
          MixPrefix + 'conv1d.weight', MixPrefix + 'conv1d.bias');
        LoadScanWeights(Blocks[BlockCnt].Scan, MixPrefix);
        if Config.UseBias then OutBias := MixPrefix + 'out_proj.bias'
        else OutBias := '';
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].OutProj,
          MixPrefix + 'out_proj.weight', Config.DInner,
          Config.HiddenSize, 0, -1, 0, OutBias);
        MarkConsumed(MixPrefix + 'out_proj.weight');
        if OutBias <> '' then MarkConsumed(OutBias);
        // Re-quantize the block just refilled with checkpoint weights.
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      LoadLlamaRMSNormWeights(Reader, FinalNorm,
        Config.Prefix + 'norm_f.weight', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'norm_f.weight');

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      // ---------------- Unexpected-tensor check ----------------
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        ImportError('Mamba import: unexpected tensor "' + TensorNameStr +
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
    Tmp.Free;
    Reader.Free;
  end;
end;

function BuildMambaFromSafeTensorsEx(const FileName: string;
  out Config: TMambaConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadMambaConfigFromJSONFile(ConfigPath);
  Result := BuildMambaFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, pQuantizeInt8);
end;

function BuildMambaFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TMambaConfig;
begin
  Result := BuildMambaFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

// ============================ BLOOM IMPORT =================================

function ReadBloomConfigFromJSONFile(const FileName: string): TBloomConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  Field: TJSONData;

  function RequiredInt(const Key: string): integer;
  var
    Data: TJSONData;
  begin
    Data := Obj.Find(Key);
    if (Data = nil) or Data.IsNull then
      ImportError('BLOOM import: config "' + FileName +
        '" is missing required key "' + Key + '".');
    Result := Data.AsInteger;
  end;

begin
  Result := Default(TBloomConfig);
  if not FileExists(FileName) then
    ImportError('BLOOM import: config file "' + FileName + '" not found.');
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('BLOOM import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('BLOOM import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    Result.ModelType := Obj.Get('model_type', 'bloom');
    if Result.ModelType <> 'bloom' then
      ImportError('BLOOM import: config model_type is "' + Result.ModelType +
        '" - expected "bloom" (see BuildFromPretrained for the full ' +
        'dispatch).');
    // hidden_size is the modern key; the original BigScience exports spell
    // it n_embed (the legacy GPT-2-style name). Likewise n_head /
    // num_attention_heads and n_layer / num_hidden_layers (HF BloomConfig's
    // attribute_map accepts both spellings; bloom-560m's config.json ships
    // n_embed + num_attention_heads + n_layer).
    Field := Obj.Find('hidden_size');
    if (Field <> nil) and not Field.IsNull then
      Result.HiddenSize := Field.AsInteger
    else
      Result.HiddenSize := RequiredInt('n_embed');
    Field := Obj.Find('n_layer');
    if (Field <> nil) and not Field.IsNull then
      Result.NumLayers := Field.AsInteger
    else
      Result.NumLayers := RequiredInt('num_hidden_layers');
    Field := Obj.Find('n_head');
    if (Field <> nil) and not Field.IsNull then
      Result.NumHeads := Field.AsInteger
    else
      Result.NumHeads := RequiredInt('num_attention_heads');
    Result.VocabSize := RequiredInt('vocab_size');
    // n_inner is null in every released checkpoint = 4*hidden (HF
    // BloomMLP hardcodes dense_h_to_4h to 4*hidden_size).
    Field := Obj.Find('n_inner');
    if (Field <> nil) and not Field.IsNull then
      Result.IntermediateSize := Field.AsInteger
    else
      Result.IntermediateSize := 4 * Result.HiddenSize;
    Result.SeqLength := Obj.Get('seq_length', 0);
    Result.LayerNormEps := Obj.Get('layer_norm_epsilon', 1.0e-5);
    if Obj.Get('apply_residual_connection_post_layernorm', False) then
      ImportError('BLOOM import: config has ' +
        'apply_residual_connection_post_layernorm=true (a pretraining-only ' +
        'Megatron variant) - this importer only wires the released ' +
        'pre-LN residual form.');
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function BloomConfigToString(const Config: TBloomConfig): string;
begin
  Result := 'bloom config: layers=' + IntToStr(Config.NumLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', ln_eps=' + FloatToStr(Config.LayerNormEps);
  if Config.SeqLength > 0 then
    Result := Result + ', seq_length=' + IntToStr(Config.SeqLength);
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

// Loads the fused BLOOM query_key_value nn.Linear ([3*hidden, hidden]
// weight + [3*hidden] bias) into a TNNetPointwiseConvLinear Q|K|V slab
// (q -> neurons 0..hidden-1, k -> hidden..2*hidden-1, v -> 2*hidden..).
// LAYOUT CONTRAST (the per-head [q|k|v] interleave families):
//   - GPT-2 c_attn: q|k|v WHOLE thirds of the 3*hidden axis, NO per-head
//     interleave at all (row r of the q third is q channel r).
//   - GPT-NeoX query_key_value: PER-HEAD interleave - head h owns rows
//     h*3*head_dim..(h+1)*3*head_dim-1, the first head_dim of them q, then
//     k, then v (HF view(.., heads, 3*head_dim) then chunk into thirds) -
//     AND the loader must compose the rotate_half RoPE row permutation
//     into the de-interleave (LoadGPTNeoXQKVWeights).
//   - BLOOM query_key_value: the SAME h-major per-head [q|k|v] byte layout
//     (HF spells it view(.., heads, 3, head_dim) and indexes the middle
//     axis - a different idiom for identical bytes), but BLOOM has NO
//     rotary at all (position lives in the ALiBi score bias), so every row
//     loads STRAIGHT: the de-interleave into the slab is the whole job.
procedure LoadBloomQKVWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName, BName: string;
  Hidden, HeadDim: integer);
var
  W, B: TNNetVolume;
  r, i, HeadIdx, Third, RowInHead, TargetIdx: integer;
begin
  EnsureWritableImportWeights(Layer);
  if not Reader.HasTensor(WName) then
    ImportError('BLOOM import: missing tensor "' + WName + '".');
  if not Reader.HasTensor(BName) then
    ImportError('BLOOM import: missing tensor "' + BName + '".');
  if (Reader.DimCount(WName) <> 2) or
     (Reader.DimSize(WName, 0) <> 3 * Hidden) or
     (Reader.DimSize(WName, 1) <> Hidden) then
    ImportError('BLOOM import: "' + WName + '" must have shape [' +
      IntToStr(3 * Hidden) + ', ' + IntToStr(Hidden) + '] (nn.Linear ' +
      'stores [out, in]), got ' + Reader.ShapeAsString(WName));
  if (Reader.DimCount(BName) <> 1) or
     (Reader.DimSize(BName, 0) <> 3 * Hidden) then
    ImportError('BLOOM import: "' + BName + '" must have shape [' +
      IntToStr(3 * Hidden) + '], got ' + Reader.ShapeAsString(BName));
  if Layer.Neurons.Count <> 3 * Hidden then
    ImportError('BLOOM import: internal error - layer for "' + WName +
      '" has ' + IntToStr(Layer.Neurons.Count) + ' neurons, expected ' +
      IntToStr(3 * Hidden) + '.');
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
      TargetIdx := Third * Hidden + HeadIdx * HeadDim + RowInHead;
      if Layer.Neurons[TargetIdx].Weights.Size <> Hidden then
        ImportError('BLOOM import: internal error - neuron ' +
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
  TBloomBlockLayers = record
    LN1, QKV, AttnDense, LN2, HTo4H, FourHToH: TNNetLayer;
  end;

function BuildBloomFromSafeTensorsWithConfig(const FileName: string;
  var Config: TBloomConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TBloomBlockLayers;
  EmbeddingLayer, EmbeddingLN, FinalLN, LMHead: TNNetLayer;
  BranchInput, AttnOut, QKVLayer, HeadPack: TNNetLayer;
  HeadOutputs: array of TNNetLayer;
  Channels: array of integer;
  BlockCnt, SeqLen, HeadCnt, HeadDim, i, j, d: integer;
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
        ImportError('BLOOM import: n_head must be >= 1.');
      if (Config.HiddenSize mod Config.NumHeads) <> 0 then
        ImportError('BLOOM import: hidden_size=' +
          IntToStr(Config.HiddenSize) + ' is not divisible by n_head=' +
          IntToStr(Config.NumHeads) + '.');
      HeadDim := Config.HiddenSize div Config.NumHeads;
      if Reader.HasTensor('transformer.word_embeddings.weight') then
        Config.Prefix := 'transformer.'
      else if Reader.HasTensor('word_embeddings.weight') then
        Config.Prefix := ''
      else
        ImportError('BLOOM import: neither ' +
          '"transformer.word_embeddings.weight" nor ' +
          '"word_embeddings.weight" found in ' + Reader.FileName +
          ' - not a BLOOM checkpoint?');
      if (Reader.DimCount(Config.Prefix + 'word_embeddings.weight') <> 2) or
         (Reader.DimSize(Config.Prefix + 'word_embeddings.weight', 0) <>
          Config.VocabSize) or
         (Reader.DimSize(Config.Prefix + 'word_embeddings.weight', 1) <>
          Config.HiddenSize) then
        ImportError('BLOOM import: word_embeddings.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.HiddenSize) +
          '], got ' +
          Reader.ShapeAsString(Config.Prefix + 'word_embeddings.weight'));
      // ALiBi imposes no positional table limit; default to the config's
      // (informational) pretraining seq_length, else 2048.
      if pSeqLen > 0 then SeqLen := pSeqLen
      else if Config.SeqLength > 0 then SeqLen := Config.SeqLength
      else SeqLen := 2048;

      // ---------------- Architecture ----------------
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      // EncodeZero=1: token id 0 is a real BPE token (<unk>), not padding.
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      // word_embeddings_layernorm: BLOOM normalises the embedding output
      // BEFORE the first block (no positional embedding is added - ALiBi
      // is the only position signal, inside the attention scores).
      EmbeddingLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      SetLength(Blocks, Config.NumLayers);
      SetLength(HeadOutputs, Config.NumHeads);
      SetLength(Channels, 3 * HeadDim);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // Attention branch: dense(MHA_ALiBi(LN_1(x))).
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].LN1 := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        QKVLayer := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(3 * Config.HiddenSize) );
        Blocks[BlockCnt].QKV := QKVLayer;
        for HeadCnt := 0 to Config.NumHeads - 1 do
        begin
          // Pack [Q_h | K_h | V_h] (width 3*head_dim) straight off the slab
          // (no rotary: BLOOM's only position signal is the ALiBi bias).
          for d := 0 to HeadDim - 1 do
          begin
            Channels[d]               := HeadCnt * HeadDim + d;
            Channels[HeadDim + d]     :=
              Config.HiddenSize + HeadCnt * HeadDim + d;
            Channels[2 * HeadDim + d] :=
              2 * Config.HiddenSize + HeadCnt * HeadDim + d;
          end;
          HeadPack := NN.AddLayerAfter(
            TNNetSplitChannels.Create(Channels), QKVLayer);
          // Standard 1/sqrt(head_dim) scaling + the per-head geometric
          // ALiBi slope (exact HF build_alibi_tensor recipe; HF adds
          // slope*j, this layer slope*(j-i) - identical softmax, see
          // TNNetALiBiAttention).
          HeadOutputs[HeadCnt] := NN.AddLayer(
            TNNetALiBiAttention.Create(HeadDim, {CausalMask=}true,
              TNNetALiBiAttention.ALiBiSlope(HeadCnt, Config.NumHeads)) );
        end;
        NN.AddLayer( TNNetDeepConcat.Create(HeadOutputs) );
        Blocks[BlockCnt].AttnDense := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        // Sequential pre-LN residuals (apply_residual_connection_post_
        // layernorm=false, the released form):
        //   x := x + Attn(LN_1(x)); x := x + MLP(LN_2(x))
        AttnOut := NN.AddLayer(
          TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        Blocks[BlockCnt].LN2 := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
        Blocks[BlockCnt].HTo4H := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.IntermediateSize) );
        // BLOOM's Megatron GELU is the tanh approximation
        // x*0.5*(1+tanh(0.79788456*x*(1+0.044715*x^2))) = TNNetGELU.
        NN.AddLayer( TNNetGELU.Create() );
        Blocks[BlockCnt].FourHToH := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), AttnOut]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        // word_embeddings -> embedding table AND the ALWAYS-tied LM head
        // (BloomForCausalLM ships no separate lm_head tensor; logits =
        // h . word_embeddings^T, bias-free).
        Reader.LoadTensorFlat(Config.Prefix + 'word_embeddings.weight', Tmp);
        if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('BLOOM import: word_embeddings.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table ' +
            'size ' + IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
        EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
        EmbeddingLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'word_embeddings.weight');
        EnsureWritableImportWeights(LMHead);
        for j := 0 to Config.VocabSize - 1 do
        begin
          for i := 0 to Config.HiddenSize - 1 do
            LMHead.Neurons[j].Weights.FData[i] :=
              Tmp.FData[j * Config.HiddenSize + i];
          LMHead.Neurons[j].BiasWeight := 0;
        end;
        LMHead.FlushWeightCache();
        if Reader.HasTensor('lm_head.weight') then
          MarkConsumed('lm_head.weight'); // redundant tied copy, if present
      finally
        Tmp.Free;
      end;
      LoadLayerNormWeights(Reader, EmbeddingLN,
        Config.Prefix + 'word_embeddings_layernorm.weight',
        Config.Prefix + 'word_embeddings_layernorm.bias', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'word_embeddings_layernorm.weight');
      MarkConsumed(Config.Prefix + 'word_embeddings_layernorm.bias');
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        BlockPrefix := Config.Prefix + 'h.' + IntToStr(BlockCnt) + '.';
        AttnPrefix := BlockPrefix + 'self_attention.';
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
        // Fused per-head-interleaved query_key_value -> Q|K|V slab (rows
        // load straight - no rotary permutation, see LoadBloomQKVWeights).
        LoadBloomQKVWeights(Reader, Blocks[BlockCnt].QKV,
          AttnPrefix + 'query_key_value.weight',
          AttnPrefix + 'query_key_value.bias',
          Config.HiddenSize, HeadDim);
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
        // Re-quantize the block just refilled with checkpoint weights.
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      LoadLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'ln_f.weight',
        Config.Prefix + 'ln_f.bias', Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'ln_f.weight');
      MarkConsumed(Config.Prefix + 'ln_f.bias');

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      // ---------------- Unexpected-tensor check ----------------
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        ImportError('BLOOM import: unexpected tensor "' + TensorNameStr +
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

function BuildBloomFromSafeTensorsEx(const FileName: string;
  out Config: TBloomConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadBloomConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildBloomFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, pQuantizeInt8);
end;

function BuildBloomFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TBloomConfig;
begin
  Result := BuildBloomFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

// ============================= FALCON IMPORT ===============================

function ReadFalconConfigFromJSONFile(const FileName: string): TFalconConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj, RopeObj: TJSONObject;
  Field: TJSONData;
  ModelType: string;
  MultiQuery: boolean;
  NumLnInParallel: integer;

  function RequiredInt(const Key, AltKey: string): integer;
  var
    Data: TJSONData;
  begin
    Data := Obj.Find(Key);
    if (Data = nil) or Data.IsNull then
      Data := Obj.Find(AltKey);
    if (Data = nil) or Data.IsNull then
      ImportError('Falcon import: config "' + FileName +
        '" is missing required key "' + Key + '" (or "' + AltKey + '").');
    Result := Data.AsInteger;
  end;

begin
  Result := Default(TFalconConfig);
  if not FileExists(FileName) then
    ImportError('Falcon import: config file "' + FileName + '" not found.');
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('Falcon import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('Falcon import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    // model_type "falcon"; the original tiiuae exports used the legacy
    // "RefinedWebModel" (falcon-7b) and "RefinedWeb" (falcon-40b) spellings.
    ModelType := Obj.Get('model_type', 'falcon');
    if (ModelType <> 'falcon') and (ModelType <> 'RefinedWebModel') and
       (ModelType <> 'RefinedWeb') then
      ImportError('Falcon import: config model_type is "' + ModelType +
        '" - expected "falcon", "RefinedWebModel" or "RefinedWeb" (see ' +
        'BuildFromPretrained for the full dispatch).');
    Result.HiddenSize := RequiredInt('hidden_size', 'hidden_size');
    Result.NumLayers := RequiredInt('num_hidden_layers', 'n_layer');
    Result.NumHeads := RequiredInt('num_attention_heads', 'n_head');
    Result.VocabSize := RequiredInt('vocab_size', 'vocab_size');
    Result.MaxPositions := Obj.Get('max_position_embeddings', 2048);
    Result.LayerNormEps := Obj.Get('layer_norm_epsilon', 1.0e-5);
    Result.TieWordEmbeddings := Obj.Get('tie_word_embeddings', True);
    // ffn_hidden_size null/absent = 4*hidden_size (the FalconConfig default).
    Field := Obj.Find('ffn_hidden_size');
    if (Field <> nil) and not Field.IsNull then
      Result.IntermediateSize := Field.AsInteger
    else
      Result.IntermediateSize := 4 * Result.HiddenSize;
    // alibi=true and bias=true are not wired for Falcon (no released Falcon
    // uses either - alibi was an early experiment, all releases use RoPE).
    if Obj.Get('alibi', False) then
      ImportError('Falcon import: config has alibi=true - this importer ' +
        'only wires the RoPE form (every released Falcon uses RoPE).');
    if Obj.Get('bias', False) then
      ImportError('Falcon import: config has bias=true - this importer ' +
        'only wires the bias-free form (no released Falcon uses biased ' +
        'Linears).');
    Result.NewDecoderArchitecture := Obj.Get('new_decoder_architecture', False);
    Result.ParallelAttn := Obj.Get('parallel_attn', True);
    MultiQuery := Obj.Get('multi_query', True);
    // EFFECTIVE K/V head count and the QKV slab layout depend on the branch:
    //   - new arch: num_kv_heads GQA groups (default = num_attention_heads);
    //   - multi_query (old arch): ONE shared K/V head;
    //   - neither: plain per-head [q|k|v] MHA (num_kv_heads = num_heads).
    if Result.NewDecoderArchitecture then
    begin
      Field := Obj.Find('num_kv_heads');
      if (Field <> nil) and not Field.IsNull then
        Result.NumKVHeads := Field.AsInteger
      else
        Result.NumKVHeads := Result.NumHeads;
    end
    else if MultiQuery then
      Result.NumKVHeads := 1
    else
      Result.NumKVHeads := Result.NumHeads;
    // LayerNorm topology: new arch (or num_ln_in_parallel_attn=2) splits the
    // single input_layernorm into ln_attn + ln_mlp. parallel_attn=false is
    // the rare sequential pre-LN fallback (input_layernorm +
    // post_attention_layernorm), single norm at the block entry.
    NumLnInParallel := Obj.Get('num_ln_in_parallel_attn', 0); // 0 = absent
    if Result.NewDecoderArchitecture and (NumLnInParallel = 0) then
      NumLnInParallel := 2; // FalconDecoderLayer default for the new arch
    Result.TwoLayerNorms :=
      Result.ParallelAttn and (NumLnInParallel = 2);
    // rope_theta: the classic key, or the newer rope_parameters.rope_theta
    // nesting; default 10000.
    Result.RopeTheta := Obj.Get('rope_theta', 10000.0);
    Field := Obj.Find('rope_parameters');
    if (Field <> nil) and (Field is TJSONObject) then
    begin
      RopeObj := TJSONObject(Field);
      Result.RopeTheta := RopeObj.Get('rope_theta', Result.RopeTheta);
      Result.RopeScaling := ReadRoPEScalingFromJSONObject(RopeObj,
        Result.MaxPositions, 'Falcon import');
    end
    else
      Result.RopeScaling := ReadRoPEScalingFromJSONObject(Obj,
        Result.MaxPositions, 'Falcon import');
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function FalconConfigToString(const Config: TFalconConfig): string;
begin
  Result := 'falcon config: layers=' + IntToStr(Config.NumLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', kv_heads=' + IntToStr(Config.NumKVHeads) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', max_pos=' + IntToStr(Config.MaxPositions) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', ln_eps=' + FloatToStr(Config.LayerNormEps) +
    ', rope_theta=' + FloatToStr(Config.RopeTheta) +
    ', new_arch=' + BoolToStr(Config.NewDecoderArchitecture, true) +
    ', parallel=' + BoolToStr(Config.ParallelAttn, true) +
    ', two_ln=' + BoolToStr(Config.TwoLayerNorms, true) +
    ', tied=' + BoolToStr(Config.TieWordEmbeddings, true);
  if Config.RopeScaling.Mode <> rsmNone then
    Result := Result + ', rope_scaling=' +
      RoPEScalingToString(Config.RopeScaling);
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

// Loads the fused Falcon query_key_value nn.Linear ([qkv_out, hidden]
// weight, bias-free) into a TNNetPointwiseConvLinear Q|K|V slab laid out
// q -> neurons 0..QWidth-1, k -> QWidth..QWidth+KVWidth-1,
// v -> QWidth+KVWidth..QWidth+2*KVWidth-1 (the same de-interleaved layout
// the GQA head wiring below expects, identical to the Llama q/k/v split).
// The source slab is INTERLEAVED PER GQA GROUP: group g packs GroupSize
// query heads, then ONE K head, then ONE V head, each head_dim rows wide
// (HF view(-1, GroupSize + 2, head_dim)); multi_query is the special case
// num_kv_heads=1 (one group). The plain-MHA layout (NOT new arch, NOT
// multi_query) is per-head [q|k|v] thirds (view(num_heads, 3, head_dim)) and
// is handled by PerHeadThirds=true. On top of the de-interleave, the q and k
// rows are PERMUTED from HF's rotate_half (first-half / second-half) into
// TNNetRotaryEmbedding's interleaved (even/odd) pair layout (the Llama
// permutation); v rows load straight.
procedure LoadFalconQKVWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName: string;
  Hidden, Heads, KVHeads, HeadDim: integer; PerHeadThirds: boolean);
var
  W: TNNetVolume;
  GroupSize, QkvOut, QWidth, KVWidth, r, i: integer;
  HeadIdx, Third, RowInHead, RotHalf, TargetRow, TargetIdx: integer;
  Group, SubInGroup, GroupStride: integer;
begin
  EnsureWritableImportWeights(Layer);
  GroupSize := Heads div KVHeads;
  QWidth := Heads * HeadDim;
  KVWidth := KVHeads * HeadDim;
  QkvOut := QWidth + 2 * KVWidth;
  if not Reader.HasTensor(WName) then
    ImportError('Falcon import: missing tensor "' + WName + '".');
  if (Reader.DimCount(WName) <> 2) or
     (Reader.DimSize(WName, 0) <> QkvOut) or
     (Reader.DimSize(WName, 1) <> Hidden) then
    ImportError('Falcon import: "' + WName + '" must have shape [' +
      IntToStr(QkvOut) + ', ' + IntToStr(Hidden) + '] (nn.Linear stores ' +
      '[out, in]), got ' + Reader.ShapeAsString(WName));
  if Layer.Neurons.Count <> QkvOut then
    ImportError('Falcon import: internal error - layer for "' + WName +
      '" has ' + IntToStr(Layer.Neurons.Count) + ' neurons, expected ' +
      IntToStr(QkvOut) + '.');
  RotHalf := HeadDim div 2;
  GroupStride := (GroupSize + 2) * HeadDim;
  W := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, W);
    for r := 0 to QkvOut - 1 do
    begin
      // Decode source row r -> (Third in {0=q,1=k,2=v}, HeadIdx, RowInHead).
      if PerHeadThirds then
      begin
        // view(num_heads, 3, head_dim): head h, third t, dim d at
        // ((h*3 + t)*head_dim + d).
        HeadIdx := r div (3 * HeadDim);
        Third := (r mod (3 * HeadDim)) div HeadDim;
        RowInHead := r mod HeadDim;
      end
      else
      begin
        // view(num_kv_heads, GroupSize + 2, head_dim): group g, sub s, dim d
        // at ((g*(GroupSize+2) + s)*head_dim + d). s in 0..GroupSize-1 are
        // query heads (global head g*GroupSize + s), s=GroupSize is the K
        // head, s=GroupSize+1 the V head (both shared across the group).
        Group := r div GroupStride;
        SubInGroup := (r mod GroupStride) div HeadDim;
        RowInHead := r mod HeadDim;
        if SubInGroup < GroupSize then
        begin
          Third := 0;
          HeadIdx := Group * GroupSize + SubInGroup;
        end
        else if SubInGroup = GroupSize then
        begin
          Third := 1;
          HeadIdx := Group;
        end
        else
        begin
          Third := 2;
          HeadIdx := Group;
        end;
      end;
      // rotate_half -> interleaved permutation on q and k rows (full head).
      TargetRow := RowInHead;
      if Third < 2 then
      begin
        if RowInHead < RotHalf then
          TargetRow := 2 * RowInHead
        else
          TargetRow := 2 * (RowInHead - RotHalf) + 1;
      end;
      // Destination index in the q|k|v slab.
      case Third of
        0: TargetIdx := HeadIdx * HeadDim + TargetRow;
        1: TargetIdx := QWidth + HeadIdx * HeadDim + TargetRow;
      else
        TargetIdx := QWidth + KVWidth + HeadIdx * HeadDim + TargetRow;
      end;
      if Layer.Neurons[TargetIdx].Weights.Size <> Hidden then
        ImportError('Falcon import: internal error - neuron ' +
          IntToStr(TargetIdx) + ' for "' + WName + '" has ' +
          IntToStr(Layer.Neurons[TargetIdx].Weights.Size) +
          ' weights, expected ' + IntToStr(Hidden) + '.');
      for i := 0 to Hidden - 1 do
        Layer.Neurons[TargetIdx].Weights.FData[i] := W.FData[r * Hidden + i];
      Layer.Neurons[TargetIdx].BiasWeight := 0;
    end;
  finally
    W.Free;
  end;
  Layer.FlushWeightCache();
end;

type
  TFalconBlockLayers = record
    LNAttn, LNMlp, LNPostAttn, QKV, AttnDense, HTo4H, FourHToH: TNNetLayer;
  end;

function BuildFalconFromSafeTensorsWithConfig(const FileName: string;
  var Config: TFalconConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TFalconBlockLayers;
  EmbeddingLayer, FinalLN, LMHead: TNNetLayer;
  BranchInput, AttnSource, MlpSource, AttnOut, MlpOut, QKVLayer: TNNetLayer;
  QSource, KSource, QSlice, KSlice, HeadPack: TNNetLayer;
  KRotated, VSlices, HeadOutputs: array of TNNetLayer;
  SliceChannels: array of integer;
  BlockCnt, SeqLen, HeadCnt, KVHeadCnt, KVGroup, GroupSize: integer;
  HeadDim, QWidth, KVWidth, i, j, d: integer;
  PerHeadThirds: boolean;
  Tmp: TNNetVolume;
  BlockPrefix, AttnPrefix, TensorNameStr: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

  // x*Phi(x), the exact erf GELU (Falcon's activation is "gelu" = exact),
  // composed from existing layers exactly like the GPT-NeoX/BERT path.
  procedure AddExactGELU;
  var
    GELUSource, PhiBranch: TNNetLayer;
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
        ImportError('Falcon import: num_attention_heads must be >= 1.');
      if Config.NumKVHeads < 1 then
        ImportError('Falcon import: effective num_kv_heads must be >= 1.');
      if (Config.NumHeads mod Config.NumKVHeads) <> 0 then
        ImportError('Falcon import: num_attention_heads=' +
          IntToStr(Config.NumHeads) + ' is not divisible by the effective ' +
          'num_kv_heads=' + IntToStr(Config.NumKVHeads) + '.');
      if (Config.HiddenSize mod Config.NumHeads) <> 0 then
        ImportError('Falcon import: hidden_size=' +
          IntToStr(Config.HiddenSize) + ' is not divisible by ' +
          'num_attention_heads=' + IntToStr(Config.NumHeads) + '.');
      HeadDim := Config.HiddenSize div Config.NumHeads;
      if Odd(HeadDim) then
        ImportError('Falcon import: head_dim=' + IntToStr(HeadDim) +
          ' must be even (RoPE rotates channel pairs).');
      QWidth := Config.NumHeads * HeadDim;
      KVWidth := Config.NumKVHeads * HeadDim;
      GroupSize := Config.NumHeads div Config.NumKVHeads;
      // The plain-MHA QKV layout (per-head [q|k|v] thirds) is used ONLY when
      // neither the new arch nor multi_query applies: that is exactly when
      // the effective num_kv_heads equals num_heads AND the new arch is off.
      PerHeadThirds := (not Config.NewDecoderArchitecture) and
        (Config.NumKVHeads = Config.NumHeads);
      if Reader.HasTensor('transformer.word_embeddings.weight') then
        Config.Prefix := 'transformer.'
      else if Reader.HasTensor('word_embeddings.weight') then
        Config.Prefix := ''
      else
        ImportError('Falcon import: neither ' +
          '"transformer.word_embeddings.weight" nor ' +
          '"word_embeddings.weight" found in ' + Reader.FileName +
          ' - not a Falcon checkpoint?');
      if (Reader.DimCount(Config.Prefix + 'word_embeddings.weight') <> 2) or
         (Reader.DimSize(Config.Prefix + 'word_embeddings.weight', 0) <>
          Config.VocabSize) or
         (Reader.DimSize(Config.Prefix + 'word_embeddings.weight', 1) <>
          Config.HiddenSize) then
        ImportError('Falcon import: word_embeddings.weight must have shape [' +
          IntToStr(Config.VocabSize) + ', ' + IntToStr(Config.HiddenSize) +
          '], got ' +
          Reader.ShapeAsString(Config.Prefix + 'word_embeddings.weight'));
      if (not Config.TieWordEmbeddings) and
         (not Reader.HasTensor('lm_head.weight')) then
        ImportError('Falcon import: config says tie_word_embeddings=false ' +
          'but "lm_head.weight" is missing from ' + Reader.FileName + '.');
      if pSeqLen <= 0 then SeqLen := Config.MaxPositions
      else SeqLen := pSeqLen;
      if SeqLen > Config.MaxPositions then
        ImportError('Falcon import: requested SeqLen=' + IntToStr(SeqLen) +
          ' exceeds max_position_embeddings=' +
          IntToStr(Config.MaxPositions) + '.');

      // ---------------- Architecture ----------------
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      SetLength(Blocks, Config.NumLayers);
      SetLength(KRotated, Config.NumKVHeads);
      SetLength(VSlices, Config.NumKVHeads);
      SetLength(HeadOutputs, Config.NumHeads);
      SetLength(SliceChannels, HeadDim);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        BranchInput := NN.GetLastLayer();
        // LayerNorm topology:
        //  - TwoLayerNorms (new arch / num_ln_in_parallel_attn=2): ln_attn
        //    feeds attention, ln_mlp feeds the MLP, BOTH read the block input;
        //  - otherwise a single LayerNorm (input_layernorm) feeds both the
        //    attention branch and (parallel_attn) the MLP branch.
        if Config.TwoLayerNorms then
        begin
          Blocks[BlockCnt].LNAttn := NN.AddLayerAfter(
            TNNetTokenLayerNorm.Create(Config.LayerNormEps), BranchInput);
          AttnSource := Blocks[BlockCnt].LNAttn;
          Blocks[BlockCnt].LNMlp := NN.AddLayerAfter(
            TNNetTokenLayerNorm.Create(Config.LayerNormEps), BranchInput);
          MlpSource := Blocks[BlockCnt].LNMlp;
        end
        else
        begin
          Blocks[BlockCnt].LNAttn := NN.AddLayerAfter(
            TNNetTokenLayerNorm.Create(Config.LayerNormEps), BranchInput);
          AttnSource := Blocks[BlockCnt].LNAttn;
          MlpSource := Blocks[BlockCnt].LNAttn; // shared (set below if seq.)
        end;
        // ---- attention branch: dense(GQA-RoPE(ln(x))) ----
        QKVLayer := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(QWidth + 2 * KVWidth), AttnSource);
        Blocks[BlockCnt].QKV := QKVLayer;
        QSource := QKVLayer;
        KSource := QKVLayer;
        // K is rotated ONCE per KV head; V (slab tail) is never rotated.
        for KVHeadCnt := 0 to Config.NumKVHeads - 1 do
        begin
          for d := 0 to HeadDim - 1 do
            SliceChannels[d] := QWidth + KVHeadCnt * HeadDim + d;
          KSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(SliceChannels), KSource);
          KRotated[KVHeadCnt] := NN.AddLayerAfter(
            CreateRoPEFromScaling(Config.RopeTheta, Config.RopeScaling),
            KSlice);
          for d := 0 to HeadDim - 1 do
            SliceChannels[d] := QWidth + KVWidth + KVHeadCnt * HeadDim + d;
          VSlices[KVHeadCnt] := NN.AddLayerAfter(
            TNNetSplitChannels.Create(SliceChannels), QKVLayer);
        end;
        for HeadCnt := 0 to Config.NumHeads - 1 do
        begin
          KVGroup := HeadCnt div GroupSize;
          for d := 0 to HeadDim - 1 do
            SliceChannels[d] := HeadCnt * HeadDim + d;
          QSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(SliceChannels), QSource);
          QSlice := NN.AddLayerAfter(
            CreateRoPEFromScaling(Config.RopeTheta, Config.RopeScaling),
            QSlice);
          HeadPack := NN.AddLayer( TNNetDeepConcat.Create(
            [QSlice, KRotated[KVGroup], VSlices[KVGroup]]) );
          HeadOutputs[HeadCnt] := NN.AddLayerAfter(
            TNNetScaledDotProductAttention.Create(HeadDim,
              {CausalMask=}true), HeadPack);
        end;
        NN.AddLayer( TNNetDeepConcat.Create(HeadOutputs) );
        Blocks[BlockCnt].AttnDense := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        AttnOut := NN.GetLastLayer();
        if not Config.ParallelAttn then
        begin
          // SEQUENTIAL pre-LN (parallel_attn=false): close the attention
          // residual first, then the MLP reads post_attention_layernorm of
          // the updated stream. x := x + Attn(ln1(x));
          // x := x + MLP(ln2(x)).
          AttnOut := NN.AddLayer( TNNetSum.Create([AttnOut, BranchInput]) );
          Blocks[BlockCnt].LNPostAttn := NN.AddLayer(
            TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
          MlpSource := Blocks[BlockCnt].LNPostAttn;
        end;
        // ---- MLP branch: dense_4h_to_h(gelu(dense_h_to_4h(ln(x)))) ----
        Blocks[BlockCnt].HTo4H := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.IntermediateSize), MlpSource);
        AddExactGELU;
        Blocks[BlockCnt].FourHToH := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        MlpOut := NN.GetLastLayer();
        if Config.ParallelAttn then
          // PARALLEL: x := x + Attn(ln(x)) + MLP(ln(x)). One fused 3-input
          // sum (HF computes mlp_output += attention_output then
          // residual + mlp_output).
          NN.AddLayer( TNNetSum.Create([BranchInput, AttnOut, MlpOut]) )
        else
          NN.AddLayer( TNNetSum.Create([MlpOut, AttnOut]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.LayerNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        Reader.LoadTensorFlat(Config.Prefix + 'word_embeddings.weight', Tmp);
        if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('Falcon import: word_embeddings.weight element count ' +
            IntToStr(Tmp.Size) + ' does not match the embedding table size ' +
            IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
        EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
        EmbeddingLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'word_embeddings.weight');
        if Config.TieWordEmbeddings then
        begin
          EnsureWritableImportWeights(LMHead);
          for j := 0 to Config.VocabSize - 1 do
          begin
            for i := 0 to Config.HiddenSize - 1 do
              LMHead.Neurons[j].Weights.FData[i] :=
                Tmp.FData[j * Config.HiddenSize + i];
            LMHead.Neurons[j].BiasWeight := 0;
          end;
          LMHead.FlushWeightCache();
          if Reader.HasTensor('lm_head.weight') then
            MarkConsumed('lm_head.weight');
        end
        else
        begin
          LoadLlamaLinearWeights(Reader, LMHead, 'lm_head.weight',
            Config.HiddenSize, Config.VocabSize);
          MarkConsumed('lm_head.weight');
        end;
      finally
        Tmp.Free;
      end;
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        BlockPrefix := Config.Prefix + 'h.' + IntToStr(BlockCnt) + '.';
        AttnPrefix := BlockPrefix + 'self_attention.';
        if Config.TwoLayerNorms then
        begin
          LoadLayerNormWeights(Reader, Blocks[BlockCnt].LNAttn,
            BlockPrefix + 'ln_attn.weight', BlockPrefix + 'ln_attn.bias',
            Config.HiddenSize);
          MarkConsumed(BlockPrefix + 'ln_attn.weight');
          MarkConsumed(BlockPrefix + 'ln_attn.bias');
          LoadLayerNormWeights(Reader, Blocks[BlockCnt].LNMlp,
            BlockPrefix + 'ln_mlp.weight', BlockPrefix + 'ln_mlp.bias',
            Config.HiddenSize);
          MarkConsumed(BlockPrefix + 'ln_mlp.weight');
          MarkConsumed(BlockPrefix + 'ln_mlp.bias');
        end
        else
        begin
          LoadLayerNormWeights(Reader, Blocks[BlockCnt].LNAttn,
            BlockPrefix + 'input_layernorm.weight',
            BlockPrefix + 'input_layernorm.bias', Config.HiddenSize);
          MarkConsumed(BlockPrefix + 'input_layernorm.weight');
          MarkConsumed(BlockPrefix + 'input_layernorm.bias');
          if not Config.ParallelAttn then
          begin
            LoadLayerNormWeights(Reader, Blocks[BlockCnt].LNPostAttn,
              BlockPrefix + 'post_attention_layernorm.weight',
              BlockPrefix + 'post_attention_layernorm.bias',
              Config.HiddenSize);
            MarkConsumed(BlockPrefix + 'post_attention_layernorm.weight');
            MarkConsumed(BlockPrefix + 'post_attention_layernorm.bias');
          end;
        end;
        LoadFalconQKVWeights(Reader, Blocks[BlockCnt].QKV,
          AttnPrefix + 'query_key_value.weight',
          Config.HiddenSize, Config.NumHeads, Config.NumKVHeads, HeadDim,
          PerHeadThirds);
        MarkConsumed(AttnPrefix + 'query_key_value.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].AttnDense,
          AttnPrefix + 'dense.weight', Config.HiddenSize, Config.HiddenSize);
        MarkConsumed(AttnPrefix + 'dense.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].HTo4H,
          BlockPrefix + 'mlp.dense_h_to_4h.weight', Config.HiddenSize,
          Config.IntermediateSize);
        MarkConsumed(BlockPrefix + 'mlp.dense_h_to_4h.weight');
        LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].FourHToH,
          BlockPrefix + 'mlp.dense_4h_to_h.weight', Config.IntermediateSize,
          Config.HiddenSize);
        MarkConsumed(BlockPrefix + 'mlp.dense_4h_to_h.weight');
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      LoadLayerNormWeights(Reader, FinalLN,
        Config.Prefix + 'ln_f.weight', Config.Prefix + 'ln_f.bias',
        Config.HiddenSize);
      MarkConsumed(Config.Prefix + 'ln_f.weight');
      MarkConsumed(Config.Prefix + 'ln_f.bias');

      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      // ---------------- Unexpected-tensor check ----------------
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        // rotary_emb.inv_freq: RoPE buffers (structural, rebuilt from base).
        if Pos('rotary_emb.inv_freq', TensorNameStr) > 0 then continue;
        ImportError('Falcon import: unexpected tensor "' + TensorNameStr +
          '" (shape ' + Reader.ShapeAsString(TensorNameStr) + ') in ' +
          FileName + ' - refusing a partial import.');
      end;
      Result := NN;
      NN := nil; // ownership transferred to the caller
    except
      on E: ESafeTensorsError do
        raise EPretrainedImportError.Create(E.Message);
    end;
  finally
    NN.Free;
    Consumed.Free;
    Reader.Free;
  end;
end;

function BuildFalconFromSafeTensorsEx(const FileName: string;
  out Config: TFalconConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadFalconConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildFalconFromSafeTensorsWithConfig(FileName, Config, pSeqLen,
    pInferenceOnly, pQuantizeInt8);
end;

function BuildFalconFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TFalconConfig;
begin
  Result := BuildFalconFromSafeTensorsEx(FileName, IgnoredConfig, pSeqLen,
    pInferenceOnly, '', pQuantizeInt8);
end;

function ReadModernBertConfigFromJSONFile(
  const FileName: string): TModernBertConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  LayerTypes: TJSONData;
  HiddenAct, LayerTypeStr, ExpectedTypeStr: string;
  LayerCnt: integer;

  function RequiredInt(const Key: string): integer;
  var
    Data: TJSONData;
  begin
    Data := Obj.Find(Key);
    if (Data = nil) or Data.IsNull then
      ImportError('ModernBERT import: config "' + FileName +
        '" is missing required key "' + Key + '".');
    Result := Data.AsInteger;
  end;

begin
  Result := Default(TModernBertConfig);
  if not FileExists(FileName) then
    ImportError('ModernBERT import: config file "' + FileName +
      '" not found.');
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('ModernBERT import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('ModernBERT import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    Result.ModelType := Obj.Get('model_type', 'modernbert');
    if Result.ModelType <> 'modernbert' then
      ImportError('ModernBERT import: config model_type is "' +
        Result.ModelType + '" - expected "modernbert" (see ' +
        'BuildFromPretrained for the full dispatch).');
    Result.HiddenSize := RequiredInt('hidden_size');
    Result.IntermediateSize := RequiredInt('intermediate_size');
    Result.NumLayers := RequiredInt('num_hidden_layers');
    Result.NumHeads := RequiredInt('num_attention_heads');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.MaxPositions := RequiredInt('max_position_embeddings');
    Result.NormEps := Obj.Get('norm_eps', 1.0e-5);
    Result.LocalAttention := Obj.Get('local_attention', 128);
    Result.GlobalEveryN := Obj.Get('global_attn_every_n_layers', 3);
    if Result.GlobalEveryN < 1 then
      ImportError('ModernBERT import: global_attn_every_n_layers must be ' +
        '>= 1, got ' + IntToStr(Result.GlobalEveryN) + '.');
    if Result.LocalAttention < 2 then
      ImportError('ModernBERT import: local_attention must be >= 2 (it is ' +
        'the TOTAL window; HF halves it), got ' +
        IntToStr(Result.LocalAttention) + '.');
    Result.GlobalRopeTheta := Obj.Get('global_rope_theta', 160000.0);
    Result.LocalRopeTheta := Obj.Get('local_rope_theta', 10000.0);
    HiddenAct := Obj.Get('hidden_activation', 'gelu');
    if HiddenAct = 'gelu' then
      Result.HiddenActTanh := false
    else if (HiddenAct = 'gelu_pytorch_tanh') or (HiddenAct = 'gelu_new') then
      Result.HiddenActTanh := true
    else
      ImportError('ModernBERT import: hidden_activation "' + HiddenAct +
        '" is not supported (gelu / gelu_pytorch_tanh / gelu_new).');
    Result.AttentionBias := Obj.Get('attention_bias', False);
    Result.MlpBias := Obj.Get('mlp_bias', False);
    Result.NormBias := Obj.Get('norm_bias', False);
    // An explicit layer_types array must MATCH the
    // global_attn_every_n_layers pattern the builder wires (HF derives
    // exactly this when layer_types is absent: "sliding_attention" iff
    // bool(i % n)) - a divergent hand-written pattern is rejected rather
    // than silently mis-wired.
    LayerTypes := Obj.Find('layer_types');
    if (LayerTypes <> nil) and not LayerTypes.IsNull then
    begin
      if (not (LayerTypes is TJSONArray)) or
         (TJSONArray(LayerTypes).Count <> Result.NumLayers) then
        ImportError('ModernBERT import: layer_types must be an array of ' +
          IntToStr(Result.NumLayers) + ' entries.');
      for LayerCnt := 0 to Result.NumLayers - 1 do
      begin
        LayerTypeStr := TJSONArray(LayerTypes).Items[LayerCnt].AsString;
        if (LayerCnt mod Result.GlobalEveryN) = 0 then
          ExpectedTypeStr := 'full_attention'
        else
          ExpectedTypeStr := 'sliding_attention';
        if LayerTypeStr <> ExpectedTypeStr then
          ImportError('ModernBERT import: layer_types[' +
            IntToStr(LayerCnt) + '] = "' + LayerTypeStr + '" does not ' +
            'match the global_attn_every_n_layers=' +
            IntToStr(Result.GlobalEveryN) + ' pattern ("' +
            ExpectedTypeStr + '" expected).');
      end;
    end;
    Result.Prefix := ''; // detected from the checkpoint by the builder
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function ModernBertConfigToString(const Config: TModernBertConfig): string;
begin
  Result := 'modernbert config: layers=' + IntToStr(Config.NumLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', hidden=' + IntToStr(Config.HiddenSize) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', max_pos=' + IntToStr(Config.MaxPositions) +
    ', local_attention=' + IntToStr(Config.LocalAttention) +
    ', global_every=' + IntToStr(Config.GlobalEveryN) +
    ', rope_theta=' + FloatToStr(Config.GlobalRopeTheta) + '/' +
    FloatToStr(Config.LocalRopeTheta) +
    ', norm_eps=' + FloatToStr(Config.NormEps);
  if Config.HiddenActTanh then
    Result := Result + ', act=gelu_tanh'
  else
    Result := Result + ', act=gelu_erf';
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

// Loads the fused ModernBERT attn.Wqkv nn.Linear ([3*hidden, hidden] weight
// + optional [3*hidden] bias) into a TNNetPointwiseConvLinear Q|K|V slab
// (q -> neurons 0..hidden-1, k -> hidden..2*hidden-1, v -> 2*hidden..).
// LAYOUT: WHOLE thirds, NO per-head interleave (HF view(.., 3, heads,
// head_dim): the q rows of ALL heads come first - the GPT-2 c_attn layout,
// NOT the BLOOM/NeoX h-major interleave). The q and k thirds additionally
// get the rotate_half ROW PERMUTATION inside each head (TNNetRotaryEmbedding
// rotates interleaved channel pairs (2c, 2c+1); HF rotates the half-split
// lanes (c, c+head_dim/2) - permuting the projection rows makes the two
// conventions bit-equal, exactly as in the Llama-family loaders). The v
// third loads straight. BiasName = '' zeroes the biases (attention_bias =
// false, the released-checkpoint default).
procedure LoadModernBertQKVWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName, BName: string; Hidden, HeadDim: integer);
var
  W, B: TNNetVolume;
  r, i, Third, RowInThird, HeadIdx, RowInHead, HalfDim, TargetIdx: integer;
begin
  EnsureWritableImportWeights(Layer);
  if not Reader.HasTensor(WName) then
    ImportError('ModernBERT import: missing tensor "' + WName + '".');
  if (Reader.DimCount(WName) <> 2) or
     (Reader.DimSize(WName, 0) <> 3 * Hidden) or
     (Reader.DimSize(WName, 1) <> Hidden) then
    ImportError('ModernBERT import: "' + WName + '" must have shape [' +
      IntToStr(3 * Hidden) + ', ' + IntToStr(Hidden) + '] (nn.Linear ' +
      'stores [out, in]), got ' + Reader.ShapeAsString(WName));
  if BName <> '' then
  begin
    if not Reader.HasTensor(BName) then
      ImportError('ModernBERT import: missing tensor "' + BName + '".');
    if (Reader.DimCount(BName) <> 1) or
       (Reader.DimSize(BName, 0) <> 3 * Hidden) then
      ImportError('ModernBERT import: "' + BName + '" must have shape [' +
        IntToStr(3 * Hidden) + '], got ' + Reader.ShapeAsString(BName));
  end;
  if Layer.Neurons.Count <> 3 * Hidden then
    ImportError('ModernBERT import: internal error - layer for "' + WName +
      '" has ' + IntToStr(Layer.Neurons.Count) + ' neurons, expected ' +
      IntToStr(3 * Hidden) + '.');
  HalfDim := HeadDim div 2;
  W := TNNetVolume.Create;
  B := nil;
  try
    Reader.LoadTensorFlat(WName, W);
    if BName <> '' then
    begin
      B := TNNetVolume.Create;
      Reader.LoadTensorFlat(BName, B);
    end;
    for r := 0 to 3 * Hidden - 1 do
    begin
      Third := r div Hidden; // 0=q, 1=k, 2=v (whole thirds)
      RowInThird := r mod Hidden;
      if Third < 2 then
      begin
        // q and k: rotate_half permutation within each head.
        HeadIdx := RowInThird div HeadDim;
        RowInHead := RowInThird mod HeadDim;
        if RowInHead < HalfDim then
          TargetIdx := Third * Hidden + HeadIdx * HeadDim + 2 * RowInHead
        else
          TargetIdx := Third * Hidden + HeadIdx * HeadDim +
            2 * (RowInHead - HalfDim) + 1;
      end
      else
        TargetIdx := r; // v: straight
      if Layer.Neurons[TargetIdx].Weights.Size <> Hidden then
        ImportError('ModernBERT import: internal error - neuron ' +
          IntToStr(TargetIdx) + ' for "' + WName + '" has ' +
          IntToStr(Layer.Neurons[TargetIdx].Weights.Size) +
          ' weights, expected ' + IntToStr(Hidden) + '.');
      for i := 0 to Hidden - 1 do
        Layer.Neurons[TargetIdx].Weights.FData[i] := W.FData[r * Hidden + i];
      if B <> nil then
        Layer.Neurons[TargetIdx].BiasWeight := B.FData[r]
      else
        Layer.Neurons[TargetIdx].BiasWeight := 0;
    end;
  finally
    B.Free;
    W.Free;
  end;
  Layer.FlushWeightCache();
end;

// Loads the fused ModernBERT mlp.Wi nn.Linear ([2*intermediate, hidden]
// weight + optional bias) into the fused GeGLU projection with the halves
// SWAPPED: HF packs input|gate and computes Wo(act(input) * gate) (chunk(2)
// - the FIRST half is the ACTIVATED one); the CAI TNNetGEGLU/TNNetGEGLUErf
// layers compute FIRSTHALF * act(SECONDHALF), so the HF gate rows (I..2I-1)
// land on neurons 0..I-1 and the HF input rows (0..I-1) on neurons
// I..2I-1. BiasName = '' zeroes the biases (mlp_bias = false).
procedure LoadModernBertWiWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName, BName: string;
  Hidden, Intermediate: integer);
var
  W, B: TNNetVolume;
  r, i, TargetIdx: integer;
begin
  EnsureWritableImportWeights(Layer);
  if not Reader.HasTensor(WName) then
    ImportError('ModernBERT import: missing tensor "' + WName + '".');
  if (Reader.DimCount(WName) <> 2) or
     (Reader.DimSize(WName, 0) <> 2 * Intermediate) or
     (Reader.DimSize(WName, 1) <> Hidden) then
    ImportError('ModernBERT import: "' + WName + '" must have shape [' +
      IntToStr(2 * Intermediate) + ', ' + IntToStr(Hidden) +
      '] (nn.Linear stores [out, in]), got ' + Reader.ShapeAsString(WName));
  if BName <> '' then
  begin
    if not Reader.HasTensor(BName) then
      ImportError('ModernBERT import: missing tensor "' + BName + '".');
    if (Reader.DimCount(BName) <> 1) or
       (Reader.DimSize(BName, 0) <> 2 * Intermediate) then
      ImportError('ModernBERT import: "' + BName + '" must have shape [' +
        IntToStr(2 * Intermediate) + '], got ' +
        Reader.ShapeAsString(BName));
  end;
  if Layer.Neurons.Count <> 2 * Intermediate then
    ImportError('ModernBERT import: internal error - layer for "' + WName +
      '" has ' + IntToStr(Layer.Neurons.Count) + ' neurons, expected ' +
      IntToStr(2 * Intermediate) + '.');
  W := TNNetVolume.Create;
  B := nil;
  try
    Reader.LoadTensorFlat(WName, W);
    if BName <> '' then
    begin
      B := TNNetVolume.Create;
      Reader.LoadTensorFlat(BName, B);
    end;
    for r := 0 to 2 * Intermediate - 1 do
    begin
      // Swap the halves: HF input rows (r < I) -> neurons I + r (the
      // activated SECOND half); HF gate rows (r >= I) -> neurons r - I.
      if r < Intermediate then
        TargetIdx := Intermediate + r
      else
        TargetIdx := r - Intermediate;
      if Layer.Neurons[TargetIdx].Weights.Size <> Hidden then
        ImportError('ModernBERT import: internal error - neuron ' +
          IntToStr(TargetIdx) + ' for "' + WName + '" has ' +
          IntToStr(Layer.Neurons[TargetIdx].Weights.Size) +
          ' weights, expected ' + IntToStr(Hidden) + '.');
      for i := 0 to Hidden - 1 do
        Layer.Neurons[TargetIdx].Weights.FData[i] := W.FData[r * Hidden + i];
      if B <> nil then
        Layer.Neurons[TargetIdx].BiasWeight := B.FData[r]
      else
        Layer.Neurons[TargetIdx].BiasWeight := 0;
    end;
  finally
    B.Free;
    W.Free;
  end;
  Layer.FlushWeightCache();
end;

// Loads a ModernBERT LayerNorm into a TNNetTokenLayerNorm: gamma from
// WName; beta from BName when HasBias (norm_bias=true), else zeroed (the
// released checkpoints ship BIAS-FREE norms - HF nn.LayerNorm with
// bias=False has no .bias tensor at all).
procedure LoadModernBertNormWeights(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName, BName: string; d_model: integer;
  HasBias: boolean);
var
  Tmp: TNNetVolume;
  i: integer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('ModernBERT import: missing tensor "' + WName + '".');
  if (Reader.DimCount(WName) <> 1) or
     (Reader.DimSize(WName, 0) <> d_model) then
    ImportError('ModernBERT import: "' + WName + '" must have shape [' +
      IntToStr(d_model) + '], got ' + Reader.ShapeAsString(WName));
  Tmp := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, Tmp);
    for i := 0 to d_model - 1 do
      Layer.Neurons[0].Weights.FData[i] := Tmp.FData[i]; // gamma
    if HasBias then
    begin
      if not Reader.HasTensor(BName) then
        ImportError('ModernBERT import: missing tensor "' + BName + '".');
      if (Reader.DimCount(BName) <> 1) or
         (Reader.DimSize(BName, 0) <> d_model) then
        ImportError('ModernBERT import: "' + BName + '" must have shape [' +
          IntToStr(d_model) + '], got ' + Reader.ShapeAsString(BName));
      Reader.LoadTensorFlat(BName, Tmp);
      for i := 0 to d_model - 1 do
        Layer.Neurons[1].Weights.FData[i] := Tmp.FData[i]; // beta
    end
    else
      for i := 0 to d_model - 1 do
        Layer.Neurons[1].Weights.FData[i] := 0; // bias-free norm
  finally
    Tmp.Free;
  end;
  Layer.FlushWeightCache();
end;

type
  TModernBertBlockLayers = record
    AttnNorm, QKV, AttnWo, MlpNorm, Wi, MlpWo: TNNetLayer;
  end;

function BuildModernBertFromSafeTensorsWithConfig(const FileName: string;
  var Config: TModernBertConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TModernBertBlockLayers;
  EmbeddingLayer, EmbeddingNorm, FinalNorm: TNNetLayer;
  BranchInput, NormedSource, QSlice, KSlice, VSlice, HeadPack: TNNetLayer;
  HeadOutputs: array of TNNetLayer;
  Channels: array of integer;
  BlockCnt, SeqLen, HeadCnt, HeadDim, LayerWindow, i, d: integer;
  LayerIsGlobal: boolean;
  LayerTheta: TNeuralFloat;
  BlockPrefix, TensorNameStr: string;
  Tmp: TNNetVolume;
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
        ImportError('ModernBERT import: num_attention_heads must be >= 1.');
      if (Config.HiddenSize mod Config.NumHeads) <> 0 then
        ImportError('ModernBERT import: hidden_size=' +
          IntToStr(Config.HiddenSize) + ' is not divisible by ' +
          'num_attention_heads=' + IntToStr(Config.NumHeads) + '.');
      HeadDim := Config.HiddenSize div Config.NumHeads;
      if Odd(HeadDim) then
        ImportError('ModernBERT import: head_dim=' + IntToStr(HeadDim) +
          ' must be even (RoPE rotates channel pairs).');
      if Config.GlobalEveryN < 1 then
        ImportError('ModernBERT import: global_attn_every_n_layers must ' +
          'be >= 1.');
      // The plain ModernBertModel serializes without a prefix; the
      // head-bearing *For* exports carry "model.".
      if Reader.HasTensor('embeddings.tok_embeddings.weight') then
        Config.Prefix := ''
      else if Reader.HasTensor('model.embeddings.tok_embeddings.weight') then
        Config.Prefix := 'model.'
      else
        ImportError('ModernBERT import: neither ' +
          '"embeddings.tok_embeddings.weight" nor ' +
          '"model.embeddings.tok_embeddings.weight" found in ' +
          Reader.FileName + ' - not a ModernBERT checkpoint?');
      if (Reader.DimCount(Config.Prefix +
            'embeddings.tok_embeddings.weight') <> 2) or
         (Reader.DimSize(Config.Prefix +
            'embeddings.tok_embeddings.weight', 0) <> Config.VocabSize) or
         (Reader.DimSize(Config.Prefix +
            'embeddings.tok_embeddings.weight', 1) <> Config.HiddenSize) then
        ImportError('ModernBERT import: tok_embeddings.weight must have ' +
          'shape [' + IntToStr(Config.VocabSize) + ', ' +
          IntToStr(Config.HiddenSize) + '], got ' +
          Reader.ShapeAsString(Config.Prefix +
            'embeddings.tok_embeddings.weight'));
      if pSeqLen <= 0 then SeqLen := Config.MaxPositions
      else SeqLen := pSeqLen;
      if SeqLen > Config.MaxPositions then
        ImportError('ModernBERT import: requested SeqLen=' +
          IntToStr(SeqLen) + ' exceeds max_position_embeddings=' +
          IntToStr(Config.MaxPositions) + '.');

      // ---------------- Architecture ----------------
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      // EncodeZero=1: token id 0 is a real BPE token, not padding (the
      // ModernBERT [PAD] sits at id 50283).
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      // embeddings.norm right after the table - ModernBERT has NO position
      // embeddings at all (RoPE inside attention is the only position
      // signal).
      EmbeddingNorm := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.NormEps) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      SetLength(Blocks, Config.NumLayers);
      SetLength(HeadOutputs, Config.NumHeads);
      SetLength(Channels, HeadDim);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        // Per-layer attention type: layer i attends GLOBALLY iff
        // i mod global_attn_every_n_layers = 0 (HF ModernBertConfig:
        // "sliding_attention" iff bool(i % n) - layer 0 is ALWAYS global).
        // NOTE the contrast with Gemma-3 ((i+1) mod n = 0). Local layers
        // carry the BIDIRECTIONAL window |i-j| <= local_attention div 2
        // (W = half + 1 in the |i-j| < W convention) and rotate with
        // local_rope_theta; global layers attend fully and rotate with
        // global_rope_theta.
        LayerIsGlobal := (BlockCnt mod Config.GlobalEveryN) = 0;
        if LayerIsGlobal then
        begin
          LayerWindow := 0;
          LayerTheta := Config.GlobalRopeTheta;
        end
        else
        begin
          LayerWindow := (Config.LocalAttention div 2) + 1;
          LayerTheta := Config.LocalRopeTheta;
        end;
        BranchInput := NN.GetLastLayer();
        // PRE-LN attention residual; HF builds layers[0].attn_norm as
        // nn.Identity (the embedding norm directly above already
        // normalized the stream), so block 0 SKIPS the norm.
        if BlockCnt = 0 then
        begin
          Blocks[BlockCnt].AttnNorm := nil;
          NormedSource := BranchInput;
        end
        else
        begin
          Blocks[BlockCnt].AttnNorm := NN.AddLayer(
            TNNetTokenLayerNorm.Create(Config.NormEps) );
          NormedSource := NN.GetLastLayer();
        end;
        // ONE fused Wqkv slab (q|k|v whole thirds after the load-time
        // rotate_half permutation; see LoadModernBertQKVWeights).
        Blocks[BlockCnt].QKV := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(3 * Config.HiddenSize),
          NormedSource);
        for HeadCnt := 0 to Config.NumHeads - 1 do
        begin
          // q and k slices rotate with the layer's RoPE theta; v never.
          for d := 0 to HeadDim - 1 do
            Channels[d] := HeadCnt * HeadDim + d;
          QSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(Channels), Blocks[BlockCnt].QKV);
          QSlice := NN.AddLayerAfter(
            TNNetRotaryEmbedding.Create(LayerTheta), QSlice);
          for d := 0 to HeadDim - 1 do
            Channels[d] := Config.HiddenSize + HeadCnt * HeadDim + d;
          KSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(Channels), Blocks[BlockCnt].QKV);
          KSlice := NN.AddLayerAfter(
            TNNetRotaryEmbedding.Create(LayerTheta), KSlice);
          for d := 0 to HeadDim - 1 do
            Channels[d] := 2 * Config.HiddenSize + HeadCnt * HeadDim + d;
          VSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(Channels), Blocks[BlockCnt].QKV);
          HeadPack := NN.AddLayer(
            TNNetDeepConcat.Create([QSlice, KSlice, VSlice]) );
          // BIDIRECTIONAL attention (CausalMask=false - this is an
          // encoder); local layers add the symmetric sliding window.
          HeadOutputs[HeadCnt] := NN.AddLayerAfter(
            TNNetScaledDotProductAttention.Create(HeadDim,
              {CausalMask=}false, {pWindow=}LayerWindow,
              {pScoreSoftCap=}0,
              {pBidirectionalWindow=}LayerWindow > 0),
            HeadPack);
        end;
        NN.AddLayer( TNNetDeepConcat.Create(HeadOutputs) );
        Blocks[BlockCnt].AttnWo := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        // PRE-LN GeGLU MLP residual: x := x + Wo(act(input) * gate),
        // input|gate = Wi(mlp_norm(x)).
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].MlpNorm := NN.AddLayer(
          TNNetTokenLayerNorm.Create(Config.NormEps) );
        Blocks[BlockCnt].Wi := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(2 * Config.IntermediateSize) );
        // hidden_activation "gelu" is the EXACT erf GELU (TNNetGEGLUErf);
        // "gelu_pytorch_tanh" the tanh approximation (TNNetGEGLU). Both
        // compute FIRSTHALF * act(SECONDHALF) - the loader swapped the HF
        // input|gate halves accordingly (see LoadModernBertWiWeights).
        if Config.HiddenActTanh then
          NN.AddLayer( TNNetGEGLU.Create() )
        else
          NN.AddLayer( TNNetGEGLUErf.Create() );
        Blocks[BlockCnt].MlpWo := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      // final_norm; the output IS the (SeqLen,1,hidden) hidden states (HF
      // ModernBertModel last_hidden_state) - no LM/pooler head.
      FinalNorm := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.NormEps) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        Reader.LoadTensorFlat(
          Config.Prefix + 'embeddings.tok_embeddings.weight', Tmp);
        if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('ModernBERT import: tok_embeddings.weight element ' +
            'count ' + IntToStr(Tmp.Size) + ' does not match the ' +
            'embedding table size ' +
            IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
        EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
        EmbeddingLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'embeddings.tok_embeddings.weight');
      finally
        Tmp.Free;
      end;
      LoadModernBertNormWeights(Reader, EmbeddingNorm,
        Config.Prefix + 'embeddings.norm.weight',
        Config.Prefix + 'embeddings.norm.bias', Config.HiddenSize,
        Config.NormBias);
      MarkConsumed(Config.Prefix + 'embeddings.norm.weight');
      if Config.NormBias then
        MarkConsumed(Config.Prefix + 'embeddings.norm.bias');
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        BlockPrefix := Config.Prefix + 'layers.' + IntToStr(BlockCnt) + '.';
        // Layer 0's attn_norm is nn.Identity - NO tensors exist for it.
        if BlockCnt > 0 then
        begin
          LoadModernBertNormWeights(Reader, Blocks[BlockCnt].AttnNorm,
            BlockPrefix + 'attn_norm.weight',
            BlockPrefix + 'attn_norm.bias', Config.HiddenSize,
            Config.NormBias);
          MarkConsumed(BlockPrefix + 'attn_norm.weight');
          if Config.NormBias then
            MarkConsumed(BlockPrefix + 'attn_norm.bias');
        end;
        if Config.AttentionBias then
        begin
          LoadModernBertQKVWeights(Reader, Blocks[BlockCnt].QKV,
            BlockPrefix + 'attn.Wqkv.weight',
            BlockPrefix + 'attn.Wqkv.bias', Config.HiddenSize, HeadDim);
          MarkConsumed(BlockPrefix + 'attn.Wqkv.bias');
          LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].AttnWo,
            BlockPrefix + 'attn.Wo.weight', Config.HiddenSize,
            Config.HiddenSize, 0, -1, 0, BlockPrefix + 'attn.Wo.bias');
          MarkConsumed(BlockPrefix + 'attn.Wo.bias');
        end
        else
        begin
          LoadModernBertQKVWeights(Reader, Blocks[BlockCnt].QKV,
            BlockPrefix + 'attn.Wqkv.weight', '',
            Config.HiddenSize, HeadDim);
          LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].AttnWo,
            BlockPrefix + 'attn.Wo.weight', Config.HiddenSize,
            Config.HiddenSize);
        end;
        MarkConsumed(BlockPrefix + 'attn.Wqkv.weight');
        MarkConsumed(BlockPrefix + 'attn.Wo.weight');
        LoadModernBertNormWeights(Reader, Blocks[BlockCnt].MlpNorm,
          BlockPrefix + 'mlp_norm.weight',
          BlockPrefix + 'mlp_norm.bias', Config.HiddenSize,
          Config.NormBias);
        MarkConsumed(BlockPrefix + 'mlp_norm.weight');
        if Config.NormBias then
          MarkConsumed(BlockPrefix + 'mlp_norm.bias');
        if Config.MlpBias then
        begin
          LoadModernBertWiWeights(Reader, Blocks[BlockCnt].Wi,
            BlockPrefix + 'mlp.Wi.weight', BlockPrefix + 'mlp.Wi.bias',
            Config.HiddenSize, Config.IntermediateSize);
          MarkConsumed(BlockPrefix + 'mlp.Wi.bias');
          LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].MlpWo,
            BlockPrefix + 'mlp.Wo.weight', Config.IntermediateSize,
            Config.HiddenSize, 0, -1, 0, BlockPrefix + 'mlp.Wo.bias');
          MarkConsumed(BlockPrefix + 'mlp.Wo.bias');
        end
        else
        begin
          LoadModernBertWiWeights(Reader, Blocks[BlockCnt].Wi,
            BlockPrefix + 'mlp.Wi.weight', '',
            Config.HiddenSize, Config.IntermediateSize);
          LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].MlpWo,
            BlockPrefix + 'mlp.Wo.weight', Config.IntermediateSize,
            Config.HiddenSize);
        end;
        MarkConsumed(BlockPrefix + 'mlp.Wi.weight');
        MarkConsumed(BlockPrefix + 'mlp.Wo.weight');
        // Re-quantize the block just refilled with checkpoint weights.
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      LoadModernBertNormWeights(Reader, FinalNorm,
        Config.Prefix + 'final_norm.weight',
        Config.Prefix + 'final_norm.bias', Config.HiddenSize,
        Config.NormBias);
      MarkConsumed(Config.Prefix + 'final_norm.weight');
      if Config.NormBias then
        MarkConsumed(Config.Prefix + 'final_norm.bias');

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      // ---------------- Unexpected-tensor check ----------------
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        ImportError('ModernBERT import: unexpected tensor "' +
          TensorNameStr + '" (shape ' +
          Reader.ShapeAsString(TensorNameStr) + ') in ' + FileName +
          ' - refusing a partial import.');
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

function BuildModernBertFromSafeTensorsEx(const FileName: string;
  out Config: TModernBertConfig; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadModernBertConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildModernBertFromSafeTensorsWithConfig(FileName, Config,
    pSeqLen, pInferenceOnly, pQuantizeInt8);
end;

function BuildModernBertFromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TModernBertConfig;
begin
  Result := BuildModernBertFromSafeTensorsEx(FileName, IgnoredConfig,
    pSeqLen, pInferenceOnly, ConfigFileName, pQuantizeInt8);
end;

// ========================= DEEPSEEK-V2 IMPORT ==============================
// (See the interface section comment for the architecture map.)

function ReadDeepSeekV2ConfigFromJSONFile(
  const FileName: string): TDeepSeekV2Config;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  Field: TJSONData;
  ModelType, StrVal: string;

  function RequiredInt(const FieldName: string): integer;
  begin
    if Obj.IndexOfName(FieldName) < 0 then
      ImportError('DeepSeek-V2 import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := Obj.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('DeepSeek-V2 import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        Obj.Find(FieldName).AsJSON + '.');
  end;

  // True when the field is present AND non-null (HF configs ship many
  // explicit nulls, e.g. q_lora_rank: null in DeepSeek-V2-Lite).
  function HasNonNull(const FieldName: string): boolean;
  begin
    Field := Obj.Find(FieldName);
    Result := (Field <> nil) and (not Field.IsNull);
  end;

begin
  if not FileExists(FileName) then
    ImportError('DeepSeek-V2 import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('DeepSeek-V2 import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('DeepSeek-V2 import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'deepseek_v2');
    if ModelType <> 'deepseek_v2' then
      ImportError('DeepSeek-V2 import: config model_type is "' + ModelType +
        '" - only "deepseek_v2" is supported here (see BuildFromPretrained ' +
        'for the full dispatch).');
    Result.ModelType := ModelType;
    Result.HiddenSize := RequiredInt('hidden_size');
    Result.IntermediateSize := RequiredInt('intermediate_size');
    Result.MoEIntermediateSize := RequiredInt('moe_intermediate_size');
    Result.NumLayers := RequiredInt('num_hidden_layers');
    Result.NumHeads := RequiredInt('num_attention_heads');
    Result.VocabSize := RequiredInt('vocab_size');
    Result.MaxPositions := RequiredInt('max_position_embeddings');
    Result.KVLoraRank := RequiredInt('kv_lora_rank');
    Result.QKNopeHeadDim := RequiredInt('qk_nope_head_dim');
    Result.QKRopeHeadDim := RequiredInt('qk_rope_head_dim');
    Result.VHeadDim := RequiredInt('v_head_dim');
    Result.NRoutedExperts := RequiredInt('n_routed_experts');
    Result.NumExpertsPerTok := RequiredInt('num_experts_per_tok');
    Result.NSharedExperts := 0;
    if HasNonNull('n_shared_experts') then
      Result.NSharedExperts := Obj.Get('n_shared_experts', 0);
    if Result.NSharedExperts < 0 then
      ImportError('DeepSeek-V2 import: config n_shared_experts must be ' +
        '>= 0 or null, got ' + IntToStr(Result.NSharedExperts) + '.');
    Result.FirstKDenseReplace := Obj.Get('first_k_dense_replace', 0);
    if Result.FirstKDenseReplace < 0 then
      ImportError('DeepSeek-V2 import: config first_k_dense_replace must ' +
        'be >= 0, got ' + IntToStr(Result.FirstKDenseReplace) + '.');
    Result.NormTopKProb := Obj.Get('norm_topk_prob', False);
    Result.RoutedScalingFactor := Obj.Get('routed_scaling_factor', 1.0);
    Result.RmsNormEps := Obj.Get('rms_norm_eps', 1.0e-6);
    Result.RopeTheta := Obj.Get('rope_theta', 10000.0);
    Result.TieWordEmbeddings := Obj.Get('tie_word_embeddings', False);
    Result.Prefix := 'model.'; // detected from the checkpoint by the builder
    // ---- structural checks ----
    if Odd(Result.QKRopeHeadDim) then
      ImportError('DeepSeek-V2 import: qk_rope_head_dim must be even ' +
        '(RoPE rotates channel pairs), got ' +
        IntToStr(Result.QKRopeHeadDim) + '.');
    if Result.VHeadDim <> Result.QKNopeHeadDim then
      ImportError('DeepSeek-V2 import: v_head_dim=' +
        IntToStr(Result.VHeadDim) + ' must equal qk_nope_head_dim=' +
        IntToStr(Result.QKNopeHeadDim) + ' (true for DeepSeek-V2/-Lite; ' +
        'the per-head V zero-padding reuses the qk_rope_head_dim-wide ' +
        'rope-K block, see the section header).');
    if Result.NumExpertsPerTok > Result.NRoutedExperts then
      ImportError('DeepSeek-V2 import: num_experts_per_tok=' +
        IntToStr(Result.NumExpertsPerTok) + ' exceeds n_routed_experts=' +
        IntToStr(Result.NRoutedExperts) + '.');
    // ---- unsupported variants (fail loudly, never silently) ----
    if HasNonNull('q_lora_rank') then
      ImportError('DeepSeek-V2 import: q_lora_rank=' + Field.AsJSON +
        ' is not supported - only the full-W_q variant (q_lora_rank: ' +
        'null, the DeepSeek-V2-Lite case) is wired. The low-rank ' +
        'q_a_proj/q_a_layernorm/q_b_proj query path of the full 236B V2 ' +
        'is not implemented.');
    if HasNonNull('rope_scaling') then
      ImportError('DeepSeek-V2 import: rope_scaling is not supported - ' +
        'DeepSeek''s YaRN configs carry mscale/mscale_all_dim ' +
        'attention-scale overrides that the rope-scaling support ' +
        'deliberately rejects. Use a checkpoint with rope_scaling: null.');
    StrVal := Obj.Get('topk_method', 'greedy');
    if StrVal <> 'greedy' then
      ImportError('DeepSeek-V2 import: topk_method "' + StrVal +
        '" is not supported - only "greedy" (group_limited_greedy ' +
        'device-grouped routing is a full-V2 feature).');
    StrVal := Obj.Get('scoring_func', 'softmax');
    if StrVal <> 'softmax' then
      ImportError('DeepSeek-V2 import: scoring_func "' + StrVal +
        '" is not supported - only "softmax" (V2 gating; sigmoid is V3).');
    if Obj.Get('moe_layer_freq', 1) <> 1 then
      ImportError('DeepSeek-V2 import: moe_layer_freq must be 1 (every ' +
        'layer past first_k_dense_replace is MoE), got ' +
        IntToStr(Obj.Get('moe_layer_freq', 1)) + '.');
    if Obj.Get('attention_bias', False) then
      ImportError('DeepSeek-V2 import: attention_bias=true is not wired ' +
        '(DeepSeek-V2 checkpoints are bias-free).');
    if Obj.Get('mlp_bias', False) then
      ImportError('DeepSeek-V2 import: mlp_bias=true is not wired ' +
        '(DeepSeek-V2 checkpoints are bias-free).');
    StrVal := Obj.Get('hidden_act', 'silu');
    if StrVal <> 'silu' then
      ImportError('DeepSeek-V2 import: hidden_act "' + StrVal +
        '" is not supported - only "silu" (SwiGLU).');
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function DeepSeekV2ConfigToString(const Config: TDeepSeekV2Config): string;
begin
  Result := 'model_type=' + Config.ModelType +
    ', hidden_size=' + IntToStr(Config.HiddenSize) +
    ', layers=' + IntToStr(Config.NumLayers) +
    ', heads=' + IntToStr(Config.NumHeads) +
    ', kv_lora_rank=' + IntToStr(Config.KVLoraRank) +
    ', qk_nope_head_dim=' + IntToStr(Config.QKNopeHeadDim) +
    ', qk_rope_head_dim=' + IntToStr(Config.QKRopeHeadDim) +
    ', v_head_dim=' + IntToStr(Config.VHeadDim) +
    ', intermediate=' + IntToStr(Config.IntermediateSize) +
    ', moe_intermediate=' + IntToStr(Config.MoEIntermediateSize) +
    ', shared_experts=' + IntToStr(Config.NSharedExperts) +
    ', routed_experts=' + IntToStr(Config.NRoutedExperts) +
    ', experts_per_tok=' + IntToStr(Config.NumExpertsPerTok) +
    ', first_k_dense=' + IntToStr(Config.FirstKDenseReplace) +
    ', vocab=' + IntToStr(Config.VocabSize) +
    ', max_pos=' + IntToStr(Config.MaxPositions) +
    ', rope_theta=' + FloatToStr(Config.RopeTheta) +
    ', rms_eps=' + FloatToStr(Config.RmsNormEps);
  if Config.NormTopKProb then
    Result := Result + ', norm_topk_prob=true';
  if Config.RoutedScalingFactor <> 1.0 then
    Result := Result + ', routed_scaling_factor=' +
      FloatToStr(Config.RoutedScalingFactor);
  if Config.TieWordEmbeddings then
    Result := Result + ', tied_embeddings=true';
  if Config.Prefix <> '' then
    Result := Result + ', prefix="' + Config.Prefix + '"';
end;

type
  TDeepSeekV2BlockLayers = record
    AttnNorm: TNNetLayer;       // input_layernorm
    QContent: TNNetLayer;       // q_proj content rows (Heads*dn)
    QRope: TNNetLayer;          // q_proj rope rows (Heads*dr)
    CKV: TNNetLayer;            // kv_a_proj_with_mqa latent rows (r)
    KRope: TNNetLayer;          // kv_a_proj_with_mqa rope rows (dr, shared)
    LatentNorm: TNNetLayer;     // kv_a_layernorm (RMSNorm on c_KV)
    KContent: TNNetLayer;       // kv_b_proj k_nope rows (Heads*dn)
    VProj: TNNetLayer;          // kv_b_proj v rows (Heads*dv)
    OProj: TNNetLayer;          // o_proj
    MlpNorm: TNNetLayer;        // post_attention_layernorm
    // dense MLP path (BlockCnt < first_k_dense_replace)
    GateUp, Down: TNNetLayer;
    // MoE path
    GateConv: TNNetLayer;       // router (mlp.gate.weight)
    ExpertGateUp, ExpertDown: array of TNNetLayer;
    SharedGateUp, SharedDown: TNNetLayer;
  end;

function BuildDeepSeekV2FromSafeTensorsWithConfig(const FileName: string;
  var Config: TDeepSeekV2Config; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false; pQuantizeInt8: boolean = false): TNNet;
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  Blocks: array of TDeepSeekV2BlockLayers;
  EmbeddingLayer, FinalNorm, LMHead: TNNetLayer;
  BranchInput, NormedSource, MoESource: TNNetLayer;
  KRopeRot, NegRopeK, VZeroPad: TNNetLayer;
  QSlice, KSlice, VSlice, QRopeSlice, HeadPack, HeadAttn: TNNetLayer;
  GateTopK, ExpertOut, GateE, GateEBroadcast: TNNetLayer;
  MoEBranches, HeadOutputs: array of TNNetLayer;
  BlockCnt, SeqLen, HeadCnt, ExpertCnt, i, j: integer;
  dn, dr, dv, LatentRank, SharedWidth: integer;
  LayerIsMoE: boolean;
  Tmp: TNNetVolume;
  BlockPrefix, TensorNameStr, LMHeadName, AttnPrefix: string;
  Consumed: TStringList;

  procedure MarkConsumed(const TName: string);
  begin
    Consumed.Add(TName);
  end;

  // Shape-checks a [OutDim, InDim] nn.Linear tensor (HF stores [out, in]).
  procedure CheckLinearShape(const WName: string; OutDim, InDim: integer);
  begin
    if not Reader.HasTensor(WName) then
      ImportError('DeepSeek-V2 import: missing tensor "' + WName + '".');
    if (Reader.DimCount(WName) <> 2) or
       (Reader.DimSize(WName, 0) <> OutDim) or
       (Reader.DimSize(WName, 1) <> InDim) then
      ImportError('DeepSeek-V2 import: "' + WName + '" must have shape [' +
        IntToStr(OutDim) + ', ' + IntToStr(InDim) + '] (nn.Linear stores ' +
        '[out, in]), got ' + Reader.ShapeAsString(WName));
  end;

  // Copies RowCount consecutive rows of a flat-loaded [*, InDim] tensor W
  // (starting at SrcRowBase) into Layer neurons DstBase..DstBase+RowCount-1
  // (bias-free). The caller flushes the layer's weight cache afterwards.
  // Used to UNPACK DeepSeek's fused projections: q_proj's per-head
  // [q_nope|q_rope], kv_a_proj_with_mqa's [c_KV|k_rope] and kv_b_proj's
  // per-head [k_nope|v] row blocks each land in their own token-wise conv.
  // NO rotate_half permutation anywhere: the checkpoint rope rows are
  // already in TNNetRotaryEmbedding's interleaved pair order (see the
  // interface section header).
  procedure LoadRows(Layer: TNNetLayer; const W: TNNetVolume;
    InDim, SrcRowBase, RowCount, DstBase: integer);
  var
    RowCnt, ColCnt: integer;
  begin
    EnsureWritableImportWeights(Layer);
    for RowCnt := 0 to RowCount - 1 do
    begin
      if Layer.Neurons[DstBase + RowCnt].Weights.Size <> InDim then
        ImportError('DeepSeek-V2 import: internal error - neuron ' +
          IntToStr(DstBase + RowCnt) + ' has ' +
          IntToStr(Layer.Neurons[DstBase + RowCnt].Weights.Size) +
          ' weights, expected ' + IntToStr(InDim) + '.');
      for ColCnt := 0 to InDim - 1 do
        Layer.Neurons[DstBase + RowCnt].Weights.FData[ColCnt] :=
          W.FData[(SrcRowBase + RowCnt) * InDim + ColCnt];
      Layer.Neurons[DstBase + RowCnt].BiasWeight := 0;
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
      // ---------------- Checkpoint validation & prefix detection ---------
      dn := Config.QKNopeHeadDim;
      dr := Config.QKRopeHeadDim;
      dv := Config.VHeadDim;
      LatentRank := Config.KVLoraRank;
      if Reader.HasTensor('model.embed_tokens.weight') then
        Config.Prefix := 'model.'
      else if Reader.HasTensor('embed_tokens.weight') then
        Config.Prefix := ''
      else
        ImportError('DeepSeek-V2 import: neither ' +
          '"model.embed_tokens.weight" nor "embed_tokens.weight" found in ' +
          Reader.FileName + ' - not a DeepSeek-V2 checkpoint?');
      if (Reader.DimCount(Config.Prefix + 'embed_tokens.weight') <> 2) or
         (Reader.DimSize(Config.Prefix + 'embed_tokens.weight', 0) <>
          Config.VocabSize) or
         (Reader.DimSize(Config.Prefix + 'embed_tokens.weight', 1) <>
          Config.HiddenSize) then
        ImportError('DeepSeek-V2 import: embed_tokens.weight must have ' +
          'shape [' + IntToStr(Config.VocabSize) + ', ' +
          IntToStr(Config.HiddenSize) + '], got ' +
          Reader.ShapeAsString(Config.Prefix + 'embed_tokens.weight'));
      LMHeadName := 'lm_head.weight';
      if (not Config.TieWordEmbeddings) and
         (not Reader.HasTensor(LMHeadName)) then
        ImportError('DeepSeek-V2 import: config says ' +
          'tie_word_embeddings=false but "' + LMHeadName +
          '" is missing from ' + Reader.FileName + '.');
      if pSeqLen <= 0 then SeqLen := Config.MaxPositions
      else SeqLen := pSeqLen;
      if SeqLen > Config.MaxPositions then
        ImportError('DeepSeek-V2 import: requested SeqLen=' +
          IntToStr(SeqLen) + ' exceeds max_position_embeddings=' +
          IntToStr(Config.MaxPositions) + '.');

      // ---------------- Architecture ----------------
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen) );
      // EncodeZero=1: token id 0 is a real token, not padding.
      EmbeddingLayer := NN.AddLayer( TNNetEmbedding.Create(
        Config.VocabSize, Config.HiddenSize, {EncodeZero=}1) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      SetLength(Blocks, Config.NumLayers);
      SetLength(HeadOutputs, Config.NumHeads);
      for BlockCnt := 0 to Config.NumLayers - 1 do
      begin
        LayerIsMoE := BlockCnt >= Config.FirstKDenseReplace;
        // ---- MLA sub-block: x := x + o_proj(MLA(RMSNorm(x))). Wired ----
        // from primitives exactly like AddMultiHeadLatentAttention with the
        // checkpoint's kv_a_layernorm RMSNorm inserted on the latent c_KV.
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].AttnNorm :=
          NN.AddLayer( TNNetTokenRMSNorm.Create(Config.RmsNormEps) );
        NormedSource := NN.GetLastLayer();
        // q_proj content/rope split (the per-head [q_nope|q_rope] row
        // packing is undone at LOAD time; see LoadRows).
        Blocks[BlockCnt].QContent := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.NumHeads * dn),
          NormedSource);
        Blocks[BlockCnt].QRope := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.NumHeads * dr),
          NormedSource);
        // kv_a_proj_with_mqa = [c_KV(r) | k_rope(dr)]: the latent down-
        // projection + ONE shared rope-K projection, both from x.
        Blocks[BlockCnt].CKV := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(LatentRank), NormedSource);
        // kv_a_layernorm: the latent RMSNorm (only c_KV is normalized,
        // never the rope slice - the HF ordering). NOTE the eps: HF
        // constructs DeepseekV2RMSNorm(kv_lora_rank) WITHOUT an eps
        // argument, so the latent norm uses the class default 1e-6 - NOT
        // config.rms_norm_eps like every other norm (a real, measurable
        // parity difference at rms_norm_eps=1e-5).
        Blocks[BlockCnt].LatentNorm := NN.AddLayerAfter(
          TNNetTokenRMSNorm.Create(1.0e-6), Blocks[BlockCnt].CKV);
        // kv_b_proj = per-head [k_nope|v] up-projections from the latent.
        Blocks[BlockCnt].KContent := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.NumHeads * dn),
          Blocks[BlockCnt].LatentNorm);
        Blocks[BlockCnt].VProj := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(Config.NumHeads * dv),
          Blocks[BlockCnt].LatentNorm);
        // Shared rope-K: projected from x, rotated ONCE, shared by every
        // head (the paper's w_KR; per-token decode state grows by only dr).
        Blocks[BlockCnt].KRope := NN.AddLayerAfter(
          TNNetPointwiseConvLinear.Create(dr), NormedSource);
        KRopeRot := NN.AddLayerAfter(
          CreateRoPEFromScaling(Config.RopeTheta, DefaultRoPEScaling()),
          Blocks[BlockCnt].KRope);
        // SDPA packs equal-width Q|K|V; V (width dv = dn) has no rope
        // slice, so it is padded with an exact dr-wide zero block built as
        // ropeK + (-ropeK) - forward is exactly 0 and the two backward
        // branches cancel (the MLA builder's trick).
        NegRopeK := NN.AddLayerAfter( TNNetNegate.Create(), KRopeRot );
        VZeroPad := NN.AddLayer( TNNetSum.Create([KRopeRot, NegRopeK]) );
        for HeadCnt := 0 to Config.NumHeads - 1 do
        begin
          QSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(HeadCnt * dn, dn),
            Blocks[BlockCnt].QContent);
          KSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(HeadCnt * dn, dn),
            Blocks[BlockCnt].KContent);
          VSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(HeadCnt * dv, dv),
            Blocks[BlockCnt].VProj);
          // This head's rope-Q slice, rotated AFTER slicing so the RoPE
          // frequency schedule runs over dr channels (matching rope-K).
          QRopeSlice := NN.AddLayerAfter(
            TNNetSplitChannels.Create(HeadCnt * dr, dr),
            Blocks[BlockCnt].QRope);
          QRopeSlice := NN.AddLayerAfter(
            CreateRoPEFromScaling(Config.RopeTheta, DefaultRoPEScaling()),
            QRopeSlice);
          // Pack [Q_h|ropeQ_h | K_h|ropeK | V_h|0] (width 3*(dn+dr)).
          HeadPack := NN.AddLayer( TNNetDeepConcat.Create(
            [QSlice, QRopeSlice, KSlice, KRopeRot, VSlice, VZeroPad]) );
          // Scale 1/sqrt(dn+dr) = HF's qk_head_dim^-0.5.
          HeadAttn := NN.AddLayerAfter(
            TNNetScaledDotProductAttention.Create(dn + dr,
              {CausalMask=}true), HeadPack);
          // The trailing dr output channels are attention-weighted zeros
          // (from the zero-padded V); keep the dv content channels.
          HeadOutputs[HeadCnt] := NN.AddLayerAfter(
            TNNetSplitChannels.Create(0, dv), HeadAttn);
        end;
        NN.AddLayer( TNNetDeepConcat.Create(HeadOutputs) );
        Blocks[BlockCnt].OProj := NN.AddLayer(
          TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        // ---- MLP sub-block: dense SwiGLU (BlockCnt < ----
        // first_k_dense_replace) or DeepSeekMoE (the AddDeepSeekMoE wiring
        // with SwiGLU experts; shared experts bypass the router).
        BranchInput := NN.GetLastLayer();
        Blocks[BlockCnt].MlpNorm :=
          NN.AddLayer( TNNetTokenRMSNorm.Create(Config.RmsNormEps) );
        if not LayerIsMoE then
        begin
          // Dense path: TNNetSwiGLU computes FIRSTHALF * act(SECONDHALF),
          // so the fused projection holds up_proj in neurons 0..I-1 and
          // gate_proj in neurons I..2I-1 (the Llama-path convention).
          Blocks[BlockCnt].GateUp := NN.AddLayer(
            TNNetPointwiseConvLinear.Create(2 * Config.IntermediateSize) );
          NN.AddLayer( TNNetSwiGLU.Create() );
          Blocks[BlockCnt].Down := NN.AddLayer(
            TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
        end
        else
        begin
          MoESource := NN.GetLastLayer(); // the normalized x
          SetLength(Blocks[BlockCnt].ExpertGateUp, Config.NRoutedExperts);
          SetLength(Blocks[BlockCnt].ExpertDown, Config.NRoutedExperts);
          if Config.NSharedExperts > 0 then
            SetLength(MoEBranches, Config.NRoutedExperts + 1)
          else
            SetLength(MoEBranches, Config.NRoutedExperts);
          // Router: token-wise linear -> PER-TOKEN softmax (the whole-
          // volume TNNetSoftMax would couple tokens) -> hard top-k gate.
          // norm_topk_prob=false (DeepSeek-V2/-Lite) keeps the RAW softmax
          // weights of the survivors (pRenormalize=false).
          Blocks[BlockCnt].GateConv := NN.AddLayerAfter(
            TNNetPointwiseConvLinear.Create(Config.NRoutedExperts),
            MoESource);
          NN.AddLayer( TNNetPointwiseSoftMax.Create() );
          GateTopK := NN.AddLayer( TNNetTopKGate.Create(
            Config.NumExpertsPerTok,
            {pRenormalize=}Config.NormTopKProb) );
          for ExpertCnt := 0 to Config.NRoutedExperts - 1 do
          begin
            Blocks[BlockCnt].ExpertGateUp[ExpertCnt] := NN.AddLayerAfter(
              TNNetPointwiseConvLinear.Create(
                2 * Config.MoEIntermediateSize), MoESource);
            NN.AddLayer( TNNetSwiGLU.Create() );
            Blocks[BlockCnt].ExpertDown[ExpertCnt] := NN.AddLayer(
              TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
            ExpertOut := NN.GetLastLayer();
            // Slice this expert's gate weight g[e], broadcast across
            // d_model and cell-multiply in (the AddDeepSeekMoE combine).
            GateE := NN.AddLayerAfter(
              TNNetSplitChannels.Create(ExpertCnt, 1), GateTopK);
            GateEBroadcast := NN.AddLayer(
              TNNetDeepConcat.Replicate(Config.HiddenSize, GateE) );
            MoEBranches[ExpertCnt] := NN.AddLayer(
              TNNetCellMulByCell.Create(ExpertOut, GateEBroadcast) );
          end;
          if Config.NSharedExperts > 0 then
          begin
            // ONE fused shared-expert MLP (HF stores mlp.shared_experts as
            // a single SwiGLU MLP of width n_shared*moe_intermediate_size),
            // added UNGATED to the routed sum.
            SharedWidth :=
              Config.NSharedExperts * Config.MoEIntermediateSize;
            Blocks[BlockCnt].SharedGateUp := NN.AddLayerAfter(
              TNNetPointwiseConvLinear.Create(2 * SharedWidth), MoESource);
            NN.AddLayer( TNNetSwiGLU.Create() );
            Blocks[BlockCnt].SharedDown := NN.AddLayer(
              TNNetPointwiseConvLinear.Create(Config.HiddenSize) );
            MoEBranches[Config.NRoutedExperts] := NN.GetLastLayer();
          end;
          // y = Sum_s Shared_s(x) + Sum_e g[e] * Routed_e(x).
          if Length(MoEBranches) > 1 then
            NN.AddLayer( TNNetSum.Create(MoEBranches) );
        end;
        NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
        if pInferenceOnly then NN.MakeInferenceOnly();
        if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      end;
      FinalNorm := NN.AddLayer(
        TNNetTokenRMSNorm.Create(Config.RmsNormEps) );
      LMHead := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.VocabSize) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();

      // ---------------- Weights ----------------
      Tmp := TNNetVolume.Create;
      try
        Reader.LoadTensorFlat(Config.Prefix + 'embed_tokens.weight', Tmp);
        if EmbeddingLayer.Neurons[0].Weights.Size <> Tmp.Size then
          ImportError('DeepSeek-V2 import: embed_tokens.weight element ' +
            'count ' + IntToStr(Tmp.Size) + ' does not match the ' +
            'embedding table size ' +
            IntToStr(EmbeddingLayer.Neurons[0].Weights.Size) + '.');
        EmbeddingLayer.Neurons[0].Weights.Copy(Tmp);
        EmbeddingLayer.FlushWeightCache();
        MarkConsumed(Config.Prefix + 'embed_tokens.weight');
        if Config.TieWordEmbeddings then
        begin
          // Tied LM head: logits = h . embed^T (rows copied, bias-free).
          EnsureWritableImportWeights(LMHead);
          for j := 0 to Config.VocabSize - 1 do
          begin
            for i := 0 to Config.HiddenSize - 1 do
              LMHead.Neurons[j].Weights.FData[i] :=
                Tmp.FData[j * Config.HiddenSize + i];
            LMHead.Neurons[j].BiasWeight := 0;
          end;
          LMHead.FlushWeightCache();
          if Reader.HasTensor(LMHeadName) then MarkConsumed(LMHeadName);
        end
        else
        begin
          LoadLlamaLinearWeights(Reader, LMHead, LMHeadName,
            Config.HiddenSize, Config.VocabSize);
          MarkConsumed(LMHeadName);
        end;
        for BlockCnt := 0 to Config.NumLayers - 1 do
        begin
          LayerIsMoE := BlockCnt >= Config.FirstKDenseReplace;
          BlockPrefix := Config.Prefix + 'layers.' +
            IntToStr(BlockCnt) + '.';
          AttnPrefix := BlockPrefix + 'self_attn.';
          LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].AttnNorm,
            BlockPrefix + 'input_layernorm.weight', Config.HiddenSize);
          MarkConsumed(BlockPrefix + 'input_layernorm.weight');
          // q_proj: per-head [q_nope(dn)|q_rope(dr)] rows split into the
          // content and rope projections.
          CheckLinearShape(AttnPrefix + 'q_proj.weight',
            Config.NumHeads * (dn + dr), Config.HiddenSize);
          Reader.LoadTensorFlat(AttnPrefix + 'q_proj.weight', Tmp);
          for HeadCnt := 0 to Config.NumHeads - 1 do
          begin
            LoadRows(Blocks[BlockCnt].QContent, Tmp, Config.HiddenSize,
              HeadCnt * (dn + dr), dn, HeadCnt * dn);
            LoadRows(Blocks[BlockCnt].QRope, Tmp, Config.HiddenSize,
              HeadCnt * (dn + dr) + dn, dr, HeadCnt * dr);
          end;
          Blocks[BlockCnt].QContent.FlushWeightCache();
          Blocks[BlockCnt].QRope.FlushWeightCache();
          MarkConsumed(AttnPrefix + 'q_proj.weight');
          // kv_a_proj_with_mqa: [c_KV(r) | k_rope(dr)] rows.
          CheckLinearShape(AttnPrefix + 'kv_a_proj_with_mqa.weight',
            LatentRank + dr, Config.HiddenSize);
          Reader.LoadTensorFlat(AttnPrefix + 'kv_a_proj_with_mqa.weight',
            Tmp);
          LoadRows(Blocks[BlockCnt].CKV, Tmp, Config.HiddenSize,
            0, LatentRank, 0);
          LoadRows(Blocks[BlockCnt].KRope, Tmp, Config.HiddenSize,
            LatentRank, dr, 0);
          Blocks[BlockCnt].CKV.FlushWeightCache();
          Blocks[BlockCnt].KRope.FlushWeightCache();
          MarkConsumed(AttnPrefix + 'kv_a_proj_with_mqa.weight');
          // kv_a_layernorm -> the latent RMSNorm (gain width r).
          LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].LatentNorm,
            AttnPrefix + 'kv_a_layernorm.weight', LatentRank);
          MarkConsumed(AttnPrefix + 'kv_a_layernorm.weight');
          // kv_b_proj: per-head [k_nope(dn)|v(dv)] rows split into the K
          // and V up-projections.
          CheckLinearShape(AttnPrefix + 'kv_b_proj.weight',
            Config.NumHeads * (dn + dv), LatentRank);
          Reader.LoadTensorFlat(AttnPrefix + 'kv_b_proj.weight', Tmp);
          for HeadCnt := 0 to Config.NumHeads - 1 do
          begin
            LoadRows(Blocks[BlockCnt].KContent, Tmp, LatentRank,
              HeadCnt * (dn + dv), dn, HeadCnt * dn);
            LoadRows(Blocks[BlockCnt].VProj, Tmp, LatentRank,
              HeadCnt * (dn + dv) + dn, dv, HeadCnt * dv);
          end;
          Blocks[BlockCnt].KContent.FlushWeightCache();
          Blocks[BlockCnt].VProj.FlushWeightCache();
          MarkConsumed(AttnPrefix + 'kv_b_proj.weight');
          LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].OProj,
            AttnPrefix + 'o_proj.weight',
            Config.NumHeads * dv, Config.HiddenSize);
          MarkConsumed(AttnPrefix + 'o_proj.weight');
          LoadLlamaRMSNormWeights(Reader, Blocks[BlockCnt].MlpNorm,
            BlockPrefix + 'post_attention_layernorm.weight',
            Config.HiddenSize);
          MarkConsumed(BlockPrefix + 'post_attention_layernorm.weight');
          if not LayerIsMoE then
          begin
            // Dense MLP: fused up (neurons 0..I-1) | gate (I..2I-1).
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
          end
          else
          begin
            // Router gate (bias-free [n_routed, hidden] nn.Linear).
            LoadLlamaLinearWeights(Reader, Blocks[BlockCnt].GateConv,
              BlockPrefix + 'mlp.gate.weight',
              Config.HiddenSize, Config.NRoutedExperts);
            MarkConsumed(BlockPrefix + 'mlp.gate.weight');
            // Routed experts: per-expert SwiGLU MLPs.
            // routed_scaling_factor is folded into down_proj - exact,
            // since the scale is linear, it multiplies ONLY the routed
            // contributions, and the shared branch bypasses it. (HF
            // multiplies the top-k gate weights instead; same product.)
            for ExpertCnt := 0 to Config.NRoutedExperts - 1 do
            begin
              TensorNameStr := BlockPrefix + 'mlp.experts.' +
                IntToStr(ExpertCnt) + '.';
              LoadLlamaLinearWeights(Reader,
                Blocks[BlockCnt].ExpertGateUp[ExpertCnt],
                TensorNameStr + 'up_proj.weight',
                Config.HiddenSize, Config.MoEIntermediateSize,
                0, 2 * Config.MoEIntermediateSize);
              MarkConsumed(TensorNameStr + 'up_proj.weight');
              LoadLlamaLinearWeights(Reader,
                Blocks[BlockCnt].ExpertGateUp[ExpertCnt],
                TensorNameStr + 'gate_proj.weight',
                Config.HiddenSize, Config.MoEIntermediateSize,
                Config.MoEIntermediateSize,
                2 * Config.MoEIntermediateSize);
              MarkConsumed(TensorNameStr + 'gate_proj.weight');
              LoadLlamaLinearWeights(Reader,
                Blocks[BlockCnt].ExpertDown[ExpertCnt],
                TensorNameStr + 'down_proj.weight',
                Config.MoEIntermediateSize, Config.HiddenSize,
                0, -1, 0, '', Config.RoutedScalingFactor);
              MarkConsumed(TensorNameStr + 'down_proj.weight');
            end;
            if Config.NSharedExperts > 0 then
            begin
              SharedWidth :=
                Config.NSharedExperts * Config.MoEIntermediateSize;
              LoadLlamaLinearWeights(Reader,
                Blocks[BlockCnt].SharedGateUp,
                BlockPrefix + 'mlp.shared_experts.up_proj.weight',
                Config.HiddenSize, SharedWidth, 0, 2 * SharedWidth);
              MarkConsumed(
                BlockPrefix + 'mlp.shared_experts.up_proj.weight');
              LoadLlamaLinearWeights(Reader,
                Blocks[BlockCnt].SharedGateUp,
                BlockPrefix + 'mlp.shared_experts.gate_proj.weight',
                Config.HiddenSize, SharedWidth, SharedWidth,
                2 * SharedWidth);
              MarkConsumed(
                BlockPrefix + 'mlp.shared_experts.gate_proj.weight');
              LoadLlamaLinearWeights(Reader,
                Blocks[BlockCnt].SharedDown,
                BlockPrefix + 'mlp.shared_experts.down_proj.weight',
                SharedWidth, Config.HiddenSize);
              MarkConsumed(
                BlockPrefix + 'mlp.shared_experts.down_proj.weight');
            end;
          end;
          // Re-quantize the block just refilled with checkpoint weights.
          if pQuantizeInt8 then NN.QuantizeWeightsInt8();
        end;
        LoadLlamaRMSNormWeights(Reader, FinalNorm,
          Config.Prefix + 'norm.weight', Config.HiddenSize);
        MarkConsumed(Config.Prefix + 'norm.weight');
      finally
        Tmp.Free;
      end;

      // Final sweep: everything refilled above ends int8-quantized.
      if pQuantizeInt8 then NN.QuantizeWeightsInt8();
      // ---------------- Unexpected-tensor check ----------------
      for i := 0 to Reader.Count - 1 do
      begin
        TensorNameStr := Reader.TensorName(i);
        if Consumed.IndexOf(TensorNameStr) >= 0 then continue;
        if Pos('rotary_emb.inv_freq', TensorNameStr) > 0 then continue;
        ImportError('DeepSeek-V2 import: unexpected tensor "' +
          TensorNameStr + '" (shape ' +
          Reader.ShapeAsString(TensorNameStr) + ') in ' + FileName +
          ' - refusing a partial import.');
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

function BuildDeepSeekV2FromSafeTensorsEx(const FileName: string;
  out Config: TDeepSeekV2Config; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadDeepSeekV2ConfigFromJSONFile(ConfigPath);
  // The builder detects Config.Prefix from the checkpoint (var parameter).
  Result := BuildDeepSeekV2FromSafeTensorsWithConfig(FileName, Config,
    pSeqLen, pInferenceOnly, pQuantizeInt8);
end;

function BuildDeepSeekV2FromSafeTensors(const FileName: string;
  pSeqLen: integer = 0; pInferenceOnly: boolean = false;
  pQuantizeInt8: boolean = false): TNNet;
var
  IgnoredConfig: TDeepSeekV2Config;
begin
  Result := BuildDeepSeekV2FromSafeTensorsEx(FileName, IgnoredConfig,
    pSeqLen, pInferenceOnly, '', pQuantizeInt8);
end;

// ============================ CLIP IMPORT ==================================

type
  // The per-block layer handles of one CLIP pre-LN encoder block (shared
  // by the text and vision towers; mirrors TBertBlockLayers).
  TClipBlockLayers = record
    LN1, QKV, AttnDense, LN2, Inter, OutDense: TNNetLayer;
  end;
  TClipBlockLayersArray = array of TClipBlockLayers;

function ClipHiddenActFromString(const ActStr: string): TClipHiddenAct;
begin
  Result := chaQuickGelu;
  if ActStr = 'quick_gelu' then Result := chaQuickGelu
  else if ActStr = 'gelu' then Result := chaGeluExact
  else if (ActStr = 'gelu_new') or (ActStr = 'gelu_pytorch_tanh') then
    Result := chaGeluTanh
  else
    ImportError('CLIP import: hidden_act "' + ActStr + '" is not ' +
      'supported - expected "quick_gelu" (every published OpenAI CLIP), ' +
      '"gelu", "gelu_new" or "gelu_pytorch_tanh".');
end;

function ClipHiddenActToString(HiddenAct: TClipHiddenAct): string;
begin
  case HiddenAct of
    chaQuickGelu: Result := 'quick_gelu';
    chaGeluExact: Result := 'gelu';
    else Result := 'gelu_tanh';
  end;
end;

// Appends Config's hidden activation to the net:
//   quick_gelu: x * sigmoid(1.702 x) = TNNetSwishLearnable(beta=1.702)
//     (y = x*sigmoid(beta*x) exactly; beta initialized to 1.702);
//   gelu (exact erf): the side-branch Phi composition of the BERT path;
//   gelu_new / gelu_pytorch_tanh: TNNetGELU (the tanh approximation).
procedure AddClipHiddenAct(NN: TNNet; HiddenAct: TClipHiddenAct);
var
  HiddenIn, PhiBranch: TNNetLayer;
begin
  case HiddenAct of
    chaQuickGelu:
      NN.AddLayer( TNNetSwishLearnable.Create(1.702) );
    chaGeluTanh:
      NN.AddLayer( TNNetGELU.Create() );
    else
    begin
      // EXACT erf GELU x*Phi(x) composed from existing layers (see the
      // BERT path): Phi = 0.5*(1 + erf(x/sqrt(2))) on a side branch, then
      // ReGLU(Phi|x) = ReLU(Phi)*x = Phi*x since Phi is in (0,1).
      HiddenIn := NN.GetLastLayer();
      NN.AddLayerAfter(
        TNNetMulByConstant.Create(0.7071067811865476), HiddenIn);
      NN.AddLayer( TNNetErf.Create() );
      NN.AddLayer( TNNetAddConstant.Create(1.0) );
      PhiBranch := NN.AddLayer( TNNetMulByConstant.Create(0.5) );
      NN.AddLayer( TNNetDeepConcat.Create([PhiBranch, HiddenIn]) );
      NN.AddLayer( TNNetReGLU.Create() );
    end;
  end;
end;

// One CLIP pre-LN encoder block (HF CLIPEncoderLayer):
//   x := x + out_proj( MHA( q|k|v( layer_norm1(x) ) ) )
//   x := x + fc2( act( fc1( layer_norm2(x) ) ) )
// Standard 1/sqrt(head_dim)-scaled SDPA, causal only in the text tower.
procedure AddClipEncoderBlock(NN: TNNet; const Tower: TClipTowerConfig;
  CausalMask: boolean; var Block: TClipBlockLayers;
  pInferenceOnly: boolean);
var
  BranchInput: TNNetLayer;
begin
  BranchInput := NN.GetLastLayer();
  Block.LN1 := NN.AddLayer(
    TNNetTokenLayerNorm.Create(Tower.LayerNormEps) );
  Block.QKV := NN.AddLayer(
    TNNetPointwiseConvLinear.Create(3 * Tower.HiddenSize) );
  Block.AttnDense := NN.AddMultiHeadSelfAttention(Tower.NumHeads,
    CausalMask);
  NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
  BranchInput := NN.GetLastLayer();
  Block.LN2 := NN.AddLayer(
    TNNetTokenLayerNorm.Create(Tower.LayerNormEps) );
  Block.Inter := NN.AddLayer(
    TNNetPointwiseConvLinear.Create(Tower.IntermediateSize) );
  AddClipHiddenAct(NN, Tower.HiddenAct);
  Block.OutDense := NN.AddLayer(
    TNNetPointwiseConvLinear.Create(Tower.HiddenSize) );
  NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
  if pInferenceOnly then NN.MakeInferenceOnly();
end;

// Loads one CLIP encoder block: separate biased nn.Linear q/k/v into the
// fused Q|K|V slab (query -> neurons 0..d-1, key -> d..2d-1, value ->
// 2d..3d-1; the builder's per-head slicing then matches HF's per-head
// reshape), biased out_proj/fc1/fc2 and the two biased LayerNorms.
procedure LoadClipEncoderBlockWeights(Reader: TNNetSafeTensorsReader;
  const Block: TClipBlockLayers; const BlockPrefix: string;
  const Tower: TClipTowerConfig);
var
  d: integer;
begin
  d := Tower.HiddenSize;
  LoadLayerNormWeights(Reader, Block.LN1,
    BlockPrefix + 'layer_norm1.weight',
    BlockPrefix + 'layer_norm1.bias', d);
  LoadLlamaLinearWeights(Reader, Block.QKV,
    BlockPrefix + 'self_attn.q_proj.weight', d, d, 0, 3 * d, 0,
    BlockPrefix + 'self_attn.q_proj.bias');
  LoadLlamaLinearWeights(Reader, Block.QKV,
    BlockPrefix + 'self_attn.k_proj.weight', d, d, d, 3 * d, 0,
    BlockPrefix + 'self_attn.k_proj.bias');
  LoadLlamaLinearWeights(Reader, Block.QKV,
    BlockPrefix + 'self_attn.v_proj.weight', d, d, 2 * d, 3 * d, 0,
    BlockPrefix + 'self_attn.v_proj.bias');
  LoadLlamaLinearWeights(Reader, Block.AttnDense,
    BlockPrefix + 'self_attn.out_proj.weight', d, d, 0, -1, 0,
    BlockPrefix + 'self_attn.out_proj.bias');
  LoadLayerNormWeights(Reader, Block.LN2,
    BlockPrefix + 'layer_norm2.weight',
    BlockPrefix + 'layer_norm2.bias', d);
  LoadLlamaLinearWeights(Reader, Block.Inter,
    BlockPrefix + 'mlp.fc1.weight', d, Tower.IntermediateSize, 0, -1, 0,
    BlockPrefix + 'mlp.fc1.bias');
  LoadLlamaLinearWeights(Reader, Block.OutDense,
    BlockPrefix + 'mlp.fc2.weight', Tower.IntermediateSize, d, 0, -1, 0,
    BlockPrefix + 'mlp.fc2.bias');
end;

// Loads a [Rows, Cols] embedding table straight into the single-neuron
// weight volume of a TNNetEmbedding / TNNetLearnedPositionalEmbedding
// (row-major in both the checkpoint and the layer) - the BERT path's
// loader without the row offset.
procedure LoadClipEmbeddingTable(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const TName: string; Rows, Cols: integer);
var
  Tmp: TNNetVolume;
begin
  if not Reader.HasTensor(TName) then
    ImportError('CLIP import: missing tensor "' + TName + '".');
  if (Reader.DimCount(TName) <> 2) or
     (Reader.DimSize(TName, 0) <> Rows) or
     (Reader.DimSize(TName, 1) <> Cols) then
    ImportError('CLIP import: "' + TName + '" must have shape [' +
      IntToStr(Rows) + ', ' + IntToStr(Cols) + '], got ' +
      Reader.ShapeAsString(TName));
  Tmp := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(TName, Tmp);
    if Layer.Neurons[0].Weights.Size <> Rows * Cols then
      ImportError('CLIP import: "' + TName + '" element count ' +
        IntToStr(Rows * Cols) + ' does not match the table size ' +
        IntToStr(Layer.Neurons[0].Weights.Size) + '.');
    Layer.Neurons[0].Weights.Copy(Tmp);
    Layer.FlushWeightCache();
  finally
    Tmp.Free;
  end;
end;

// Loads the bias-FREE CLIP patch embedding (HF nn.Conv2d, weight
// [Hidden, NumChannels, Patch, Patch], stride = kernel = Patch) into a
// TNNetConvolutionLinear built with a (Patch,Patch) kernel on the
// (ImageSize, ImageSize, NumChannels) pixel grid. CAI neuron weights are
// (x, y, depth)-indexed ((y*Patch + x)*NumChannels + c), HF kernels
// [o, c, ky, kx]-indexed.
procedure LoadClipPatchConv(Reader: TNNetSafeTensorsReader;
  Layer: TNNetLayer; const WName: string;
  NumChannels, Patch, Hidden: integer);
var
  W: TNNetVolume;
  o, c, ky, kx: integer;
begin
  if not Reader.HasTensor(WName) then
    ImportError('CLIP import: missing tensor "' + WName + '".');
  if (Reader.DimCount(WName) <> 4) or
     (Reader.DimSize(WName, 0) <> Hidden) or
     (Reader.DimSize(WName, 1) <> NumChannels) or
     (Reader.DimSize(WName, 2) <> Patch) or
     (Reader.DimSize(WName, 3) <> Patch) then
    ImportError('CLIP import: "' + WName + '" must have shape [' +
      IntToStr(Hidden) + ', ' + IntToStr(NumChannels) + ', ' +
      IntToStr(Patch) + ', ' + IntToStr(Patch) + '] (nn.Conv2d stores ' +
      '[out, in, kh, kw]), got ' + Reader.ShapeAsString(WName));
  if Layer.Neurons.Count <> Hidden then
    ImportError('CLIP import: internal error - patch conv has ' +
      IntToStr(Layer.Neurons.Count) + ' neurons, expected ' +
      IntToStr(Hidden) + '.');
  if Layer.Neurons[0].Weights.Size <> Patch * Patch * NumChannels then
    ImportError('CLIP import: internal error - patch conv neuron has ' +
      IntToStr(Layer.Neurons[0].Weights.Size) + ' weights, expected ' +
      IntToStr(Patch * Patch * NumChannels) + '.');
  W := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat(WName, W);
    for o := 0 to Hidden - 1 do
    begin
      for ky := 0 to Patch - 1 do
        for kx := 0 to Patch - 1 do
          for c := 0 to NumChannels - 1 do
            Layer.Neurons[o].Weights.FData[
              (ky * Patch + kx) * NumChannels + c] :=
              W.FData[((o * NumChannels + c) * Patch + ky) * Patch + kx];
      Layer.Neurons[o].BiasWeight := 0; // bias=False in CLIP's patch conv
    end;
  finally
    W.Free;
  end;
  Layer.FlushWeightCache();
end;

function BuildClipVisionTower(Reader: TNNetSafeTensorsReader;
  const Tower: TClipTowerConfig; ImageSize, PatchSize, NumChannels,
  ProjectionDim: integer; const Prefix: string;
  const ProjectionTensorName: string = '';
  pInferenceOnly: boolean = false): TNNet;
var
  NN: TNNet;
  PatchConv, PosEmb, PreLN, PostLN, Proj: TNNetLayer;
  Blocks: TClipBlockLayersArray;
  Tbl: TNNetVolume;
  Grid, NumPatches, BlockCnt, ChanCnt: integer;
  PreLNName: string;
begin
  if (Tower.NumHeads < 1) or
     ((Tower.HiddenSize mod Tower.NumHeads) <> 0) then
    ImportError('CLIP import: vision hidden_size=' +
      IntToStr(Tower.HiddenSize) + ' is not divisible by ' +
      'num_attention_heads=' + IntToStr(Tower.NumHeads) + '.');
  if (PatchSize < 1) or ((ImageSize mod PatchSize) <> 0) then
    ImportError('CLIP import: image_size=' + IntToStr(ImageSize) +
      ' is not a multiple of patch_size=' + IntToStr(PatchSize) + '.');
  Grid := ImageSize div PatchSize;
  NumPatches := Grid * Grid;
  NN := TNNet.Create();
  try
    // ---------------- Architecture ----------------
    NN.AddLayer( TNNetInput.Create(ImageSize, ImageSize, NumChannels) );
    // Bias-free patch embedding: kernel = stride = patch_size, no padding
    // -> a (Grid, Grid, hidden) patch grid ...
    PatchConv := NN.AddLayer( TNNetConvolutionLinear.Create(
      Tower.HiddenSize, PatchSize, {pInputPadding=}0, {pStride=}PatchSize,
      {pSuppressBias=}1) );
    // ... flattened row-major (the volume layout IS row-major over (y,x),
    // exactly HF's flatten(2).transpose(1,2) patch order) ...
    NN.AddLayer( TNNetReshape.Create(NumPatches, 1, Tower.HiddenSize) );
    // ... with one ZERO row prepended as the class-token slot (PadXY pads
    // both ends; Crop drops the right pad). class_embedding itself is
    // folded into row 0 of the position table below - exact, since this
    // slot's pre-position content is identically zero.
    NN.AddLayer( TNNetPadXY.Create(1, 0) );
    NN.AddLayer( TNNetCrop.Create(0, 0, NumPatches + 1, 1) );
    PosEmb := NN.AddLayer(
      TNNetLearnedPositionalEmbedding.Create(NumPatches + 1) );
    PreLN := NN.AddLayer(
      TNNetTokenLayerNorm.Create(Tower.LayerNormEps) );
    if pInferenceOnly then NN.MakeInferenceOnly();
    SetLength(Blocks, Tower.NumLayers);
    for BlockCnt := 0 to Tower.NumLayers - 1 do
      AddClipEncoderBlock(NN, Tower, {CausalMask=}false,
        Blocks[BlockCnt], pInferenceOnly);
    // HF applies post_layernorm (and the projection) only to the pooled
    // class token; applying them PER TOKEN is exact for row 0.
    PostLN := NN.AddLayer(
      TNNetTokenLayerNorm.Create(Tower.LayerNormEps) );
    Proj := nil;
    if ProjectionTensorName <> '' then
      Proj := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(ProjectionDim) );
    if pInferenceOnly then NN.MakeInferenceOnly();

    // ---------------- Weights ----------------
    LoadClipPatchConv(Reader, PatchConv,
      Prefix + 'embeddings.patch_embedding.weight',
      NumChannels, PatchSize, Tower.HiddenSize);
    LoadClipEmbeddingTable(Reader, PosEmb,
      Prefix + 'embeddings.position_embedding.weight',
      NumPatches + 1, Tower.HiddenSize);
    // class_embedding [hidden] folded into position row 0 (see above).
    if not Reader.HasTensor(Prefix + 'embeddings.class_embedding') then
      ImportError('CLIP import: missing tensor "' + Prefix +
        'embeddings.class_embedding".');
    Tbl := TNNetVolume.Create;
    try
      Reader.LoadTensorFlat(Prefix + 'embeddings.class_embedding', Tbl);
      if Tbl.Size <> Tower.HiddenSize then
        ImportError('CLIP import: "' + Prefix +
          'embeddings.class_embedding" must have ' +
          IntToStr(Tower.HiddenSize) + ' elements, got ' +
          IntToStr(Tbl.Size) + '.');
      for ChanCnt := 0 to Tower.HiddenSize - 1 do
        PosEmb.Neurons[0].Weights.FData[ChanCnt] :=
          PosEmb.Neurons[0].Weights.FData[ChanCnt] + Tbl.FData[ChanCnt];
      PosEmb.FlushWeightCache();
    finally
      Tbl.Free;
    end;
    // modeling_clip's historical "pre_layrnorm" spelling; accept the
    // fixed spelling too in case an exporter normalizes it.
    PreLNName := Prefix + 'pre_layrnorm';
    if not Reader.HasTensor(PreLNName + '.weight') then
      PreLNName := Prefix + 'pre_layernorm';
    LoadLayerNormWeights(Reader, PreLN, PreLNName + '.weight',
      PreLNName + '.bias', Tower.HiddenSize);
    for BlockCnt := 0 to Tower.NumLayers - 1 do
      LoadClipEncoderBlockWeights(Reader, Blocks[BlockCnt],
        Prefix + 'encoder.layers.' + IntToStr(BlockCnt) + '.', Tower);
    LoadLayerNormWeights(Reader, PostLN,
      Prefix + 'post_layernorm.weight',
      Prefix + 'post_layernorm.bias', Tower.HiddenSize);
    if Proj <> nil then
      LoadLlamaLinearWeights(Reader, Proj, ProjectionTensorName,
        Tower.HiddenSize, ProjectionDim); // bias-free: CAI biases zeroed
    Result := NN;
  except
    NN.Free;
    raise;
  end;
end;

procedure BuildClipFromSafeTensorsWithConfig(const FileName: string;
  var Config: TClipConfig; out TextNet, VisionNet: TNNet;
  TextSeqLen: integer = 0; pInferenceOnly: boolean = false);
var
  Reader: TNNetSafeTensorsReader;
  NN: TNNet;
  TokEmb, PosEmb, FinalLN, Proj: TNNetLayer;
  Blocks: TClipBlockLayersArray;
  Tmp: TNNetVolume;
  SeqLen, BlockCnt: integer;
begin
  TextNet := nil;
  VisionNet := nil;
  Reader := CreatePretrainedTensorReader(FileName);
  NN := nil;
  try
    try
      if (Config.Text.NumHeads < 1) or
         ((Config.Text.HiddenSize mod Config.Text.NumHeads) <> 0) then
        ImportError('CLIP import: text hidden_size=' +
          IntToStr(Config.Text.HiddenSize) + ' is not divisible by ' +
          'num_attention_heads=' + IntToStr(Config.Text.NumHeads) + '.');
      if TextSeqLen <= 0 then SeqLen := Config.TextMaxPositions
      else SeqLen := TextSeqLen;
      if SeqLen > Config.TextMaxPositions then
        ImportError('CLIP import: requested TextSeqLen=' +
          IntToStr(SeqLen) + ' exceeds max_position_embeddings=' +
          IntToStr(Config.TextMaxPositions) + '.');

      // ---------------- TEXT tower architecture ----------------
      NN := TNNet.Create();
      NN.AddLayer( TNNetInput.Create(SeqLen, 1, 1) );
      // EncodeZero=1: token id 0 (<|startoftext|>) is a REAL vocab row.
      TokEmb := NN.AddLayer( TNNetEmbedding.Create(
        Config.TextVocabSize, Config.Text.HiddenSize, {EncodeZero=}1) );
      PosEmb := NN.AddLayer(
        TNNetLearnedPositionalEmbedding.Create(Config.TextMaxPositions) );
      if pInferenceOnly then NN.MakeInferenceOnly();
      SetLength(Blocks, Config.Text.NumLayers);
      for BlockCnt := 0 to Config.Text.NumLayers - 1 do
        AddClipEncoderBlock(NN, Config.Text, {CausalMask=}true,
          Blocks[BlockCnt], pInferenceOnly);
      FinalLN := NN.AddLayer(
        TNNetTokenLayerNorm.Create(Config.Text.LayerNormEps) );
      // text_projection applied PER TOKEN; HF's text_embeds is the row at
      // the eot position (ClipTextEosPosition).
      Proj := NN.AddLayer(
        TNNetPointwiseConvLinear.Create(Config.ProjectionDim) );
      if pInferenceOnly then NN.MakeInferenceOnly();

      // ---------------- TEXT tower weights ----------------
      LoadClipEmbeddingTable(Reader, TokEmb,
        'text_model.embeddings.token_embedding.weight',
        Config.TextVocabSize, Config.Text.HiddenSize);
      LoadClipEmbeddingTable(Reader, PosEmb,
        'text_model.embeddings.position_embedding.weight',
        Config.TextMaxPositions, Config.Text.HiddenSize);
      for BlockCnt := 0 to Config.Text.NumLayers - 1 do
        LoadClipEncoderBlockWeights(Reader, Blocks[BlockCnt],
          'text_model.encoder.layers.' + IntToStr(BlockCnt) + '.',
          Config.Text);
      LoadLayerNormWeights(Reader, FinalLN,
        'text_model.final_layer_norm.weight',
        'text_model.final_layer_norm.bias', Config.Text.HiddenSize);
      LoadLlamaLinearWeights(Reader, Proj, 'text_projection.weight',
        Config.Text.HiddenSize, Config.ProjectionDim); // bias-free
      TextNet := NN;
      NN := nil;

      // ---------------- VISION tower ----------------
      VisionNet := BuildClipVisionTower(Reader, Config.Vision,
        Config.ImageSize, Config.PatchSize, Config.NumChannels,
        Config.ProjectionDim, 'vision_model.',
        'visual_projection.weight', pInferenceOnly);

      // ---------------- logit_scale ----------------
      if Reader.HasTensor('logit_scale') then
      begin
        Tmp := TNNetVolume.Create;
        try
          Reader.LoadTensorFlat('logit_scale', Tmp);
          if Tmp.Size >= 1 then Config.LogitScale := Tmp.FData[0];
        finally
          Tmp.Free;
        end;
      end;
    except
      NN.Free;
      TextNet.Free;
      TextNet := nil;
      VisionNet.Free;
      VisionNet := nil;
      raise;
    end;
  finally
    Reader.Free;
  end;
end;

procedure BuildClipFromSafeTensors(const FileName: string;
  out TextNet, VisionNet: TNNet; out Config: TClipConfig;
  TextSeqLen: integer = 0; pInferenceOnly: boolean = false;
  const ConfigFileName: string = '');
var
  ConfigPath: string;
begin
  if ConfigFileName <> '' then ConfigPath := ConfigFileName
  else ConfigPath := ExtractFilePath(FileName) + 'config.json';
  Config := ReadClipConfigFromJSONFile(ConfigPath);
  BuildClipFromSafeTensorsWithConfig(FileName, Config, TextNet, VisionNet,
    TextSeqLen, pInferenceOnly);
end;

function ReadClipConfigFromJSONFile(const FileName: string): TClipConfig;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj, TowerObj: TJSONObject;
  ModelType: string;

  function RequiredSubObject(const FieldName: string): TJSONObject;
  var
    Data: TJSONData;
  begin
    Data := Obj.Find(FieldName);
    if (Data = nil) or not (Data is TJSONObject) then
      ImportError('CLIP import: config "' + FileName +
        '" is missing the required sub-object "' + FieldName + '".');
    Result := TJSONObject(Data);
  end;

  function RequiredInt(O: TJSONObject; const FieldName: string): integer;
  begin
    if O.IndexOfName(FieldName) < 0 then
      ImportError('CLIP import: config "' + FileName +
        '" is missing the required field "' + FieldName + '".');
    Result := O.Get(FieldName, 0);
    if Result <= 0 then
      ImportError('CLIP import: config field "' + FieldName +
        '" must be a positive integer, got ' +
        O.Find(FieldName).AsJSON + '.');
  end;

  procedure ReadTower(O: TJSONObject; var Tower: TClipTowerConfig);
  begin
    Tower.HiddenSize := RequiredInt(O, 'hidden_size');
    Tower.IntermediateSize := RequiredInt(O, 'intermediate_size');
    Tower.NumLayers := RequiredInt(O, 'num_hidden_layers');
    Tower.NumHeads := RequiredInt(O, 'num_attention_heads');
    Tower.LayerNormEps := O.Get('layer_norm_eps', 0.00001);
    Tower.HiddenAct := ClipHiddenActFromString(
      O.Get('hidden_act', 'quick_gelu'));
  end;

begin
  if not FileExists(FileName) then
    ImportError('CLIP import: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        ImportError('CLIP import: config "' + FileName +
          '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      ImportError('CLIP import: config "' + FileName +
        '" is not a JSON object.');
    Obj := TJSONObject(Root);
    ModelType := Obj.Get('model_type', 'clip');
    if ModelType <> 'clip' then
      ImportError('CLIP import: config model_type is "' + ModelType +
        '" - only "clip" is supported here (see BuildFromPretrained ' +
        'for the full dispatch).');
    Result.ModelType := ModelType;
    TowerObj := RequiredSubObject('text_config');
    ReadTower(TowerObj, Result.Text);
    Result.TextVocabSize := RequiredInt(TowerObj, 'vocab_size');
    Result.TextMaxPositions :=
      RequiredInt(TowerObj, 'max_position_embeddings');
    // eos_token_id = 2 is the legacy pre-#24773 id every published OpenAI
    // CLIP config carries; it selects the ARGMAX pooling branch (see
    // ClipTextEosPosition).
    Result.EosTokenId := TowerObj.Get('eos_token_id', 2);
    TowerObj := RequiredSubObject('vision_config');
    ReadTower(TowerObj, Result.Vision);
    Result.ImageSize := RequiredInt(TowerObj, 'image_size');
    Result.PatchSize := RequiredInt(TowerObj, 'patch_size');
    Result.NumChannels := TowerObj.Get('num_channels', 3);
    Result.ProjectionDim := Obj.Get('projection_dim', 512);
    Result.LogitScale := Obj.Get('logit_scale_init_value', 2.6592);
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function ClipConfigToString(const Config: TClipConfig): string;
begin
  if Config.ModelType = '' then Result := 'clip'
  else Result := Config.ModelType;
  Result := Result + ' config: text(d=' + IntToStr(Config.Text.HiddenSize) +
    ', layers=' + IntToStr(Config.Text.NumLayers) +
    ', heads=' + IntToStr(Config.Text.NumHeads) +
    ', ffn=' + IntToStr(Config.Text.IntermediateSize) +
    ', vocab=' + IntToStr(Config.TextVocabSize) +
    ', max_pos=' + IntToStr(Config.TextMaxPositions) +
    ', act=' + ClipHiddenActToString(Config.Text.HiddenAct) +
    ', eos=' + IntToStr(Config.EosTokenId) +
    '), vision(d=' + IntToStr(Config.Vision.HiddenSize) +
    ', layers=' + IntToStr(Config.Vision.NumLayers) +
    ', heads=' + IntToStr(Config.Vision.NumHeads) +
    ', ffn=' + IntToStr(Config.Vision.IntermediateSize) +
    ', image=' + IntToStr(Config.ImageSize) +
    ', patch=' + IntToStr(Config.PatchSize) +
    ', act=' + ClipHiddenActToString(Config.Vision.HiddenAct) +
    '), proj_dim=' + IntToStr(Config.ProjectionDim) +
    ', logit_scale=' + FloatToStrF(Config.LogitScale, ffGeneral, 6, 0);
end;

function ClipTextEosPosition(TokenIds: TNNetVolume;
  EosTokenId: integer): integer;
var
  PosCnt, TokenId, BestId: integer;
begin
  Result := 0;
  if (TokenIds.SizeY <> 1) or (TokenIds.Depth <> 1) or
     (TokenIds.SizeX < 1) then
    ImportError('ClipTextEosPosition: expected a (SeqLen,1,1) token-id ' +
      'volume, got ' + IntToStr(TokenIds.SizeX) + 'x' +
      IntToStr(TokenIds.SizeY) + 'x' + IntToStr(TokenIds.Depth) + '.');
  if EosTokenId = 2 then
  begin
    // The legacy pre-#24773 branch of modeling_clip: pool at
    // input_ids.argmax(-1) - the FIRST occurrence of the highest id
    // (the eot token carries the top id in the CLIP vocab).
    BestId := Round(TokenIds.FData[0]);
    for PosCnt := 1 to TokenIds.SizeX - 1 do
    begin
      TokenId := Round(TokenIds.FData[PosCnt]);
      if TokenId > BestId then
      begin
        BestId := TokenId;
        Result := PosCnt;
      end;
    end;
  end
  else
  begin
    // The fixed branch: the FIRST position equal to eos_token_id.
    Result := -1;
    for PosCnt := 0 to TokenIds.SizeX - 1 do
      if Round(TokenIds.FData[PosCnt]) = EosTokenId then
      begin
        Result := PosCnt;
        break;
      end;
    if Result < 0 then
      ImportError('ClipTextEosPosition: no eos token (id ' +
        IntToStr(EosTokenId) + ') in the sequence - CLIP text pooling ' +
        'requires one (append it like the CLIP tokenizer does).');
  end;
end;

procedure ClipExtractEmbedding(NetOutput: TNNetVolume; TokenPos: integer;
  Embedding: TNNetVolume);
var
  Depth, ChanCnt: integer;
  Norm: TNeuralFloat;
begin
  Depth := NetOutput.Depth;
  if (NetOutput.SizeY <> 1) or (NetOutput.SizeX < 1) or (Depth < 1) then
    ImportError('ClipExtractEmbedding: expected a (SeqLen,1,depth) ' +
      'volume, got ' + IntToStr(NetOutput.SizeX) + 'x' +
      IntToStr(NetOutput.SizeY) + 'x' + IntToStr(Depth) + '.');
  if (TokenPos < 0) or (TokenPos >= NetOutput.SizeX) then
    ImportError('ClipExtractEmbedding: TokenPos ' + IntToStr(TokenPos) +
      ' out of range 0..' + IntToStr(NetOutput.SizeX - 1) + '.');
  Embedding.ReSize(1, 1, Depth);
  Norm := 0;
  for ChanCnt := 0 to Depth - 1 do
  begin
    Embedding.FData[ChanCnt] := NetOutput.FData[TokenPos * Depth + ChanCnt];
    Norm := Norm + Embedding.FData[ChanCnt] * Embedding.FData[ChanCnt];
  end;
  Norm := Sqrt(Norm);
  if Norm > 0 then
    for ChanCnt := 0 to Depth - 1 do
      Embedding.FData[ChanCnt] := Embedding.FData[ChanCnt] / Norm;
end;

function ClipSimilarity(EmbA, EmbB: TNNetVolume): TNeuralFloat;
var
  ChanCnt: integer;
begin
  Result := 0;
  if EmbA.Size <> EmbB.Size then
    ImportError('ClipSimilarity: embedding sizes differ (' +
      IntToStr(EmbA.Size) + ' vs ' + IntToStr(EmbB.Size) + ').');
  for ChanCnt := 0 to EmbA.Size - 1 do
    Result := Result + EmbA.FData[ChanCnt] * EmbB.FData[ChanCnt];
end;

function BuildFromPretrained(const Path: string; pSeqLen: integer = 0;
  pInferenceOnly: boolean = false;
  const ConfigFileName: string = ''; pQuantizeInt8: boolean = false): TNNet;
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
  IgnoredCohereConfig: TCohereConfig;
  IgnoredPhiConfig: TPhiConfig;
  IgnoredBertConfig: TBertConfig;
  IgnoredRWKVConfig: TRWKVConfig;
  IgnoredMambaConfig: TMambaConfig;
  IgnoredBloomConfig: TBloomConfig;
  IgnoredFalconConfig: TFalconConfig;
  IgnoredModernBertConfig: TModernBertConfig;
  IgnoredDeepSeekV2Config: TDeepSeekV2Config;
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
      WeightsPath := IncludeTrailingPathDelimiter(Path) +
        'pytorch_model.bin.index.json'; // sharded torch.save fallback
    if not FileExists(WeightsPath) then
      ImportError('BuildFromPretrained: none of "model.safetensors", ' +
        '"model.safetensors.index.json", "pytorch_model.bin" or ' +
        '"pytorch_model.bin.index.json" found in directory ' + Path + '.');
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
      pInferenceOnly, ConfigPath, pQuantizeInt8);
  end
  else if ModelType = 'gpt2' then
    Result := BuildGPT2FromSafeTensors(WeightsPath, pSeqLen, NumHeads,
      pInferenceOnly, GPT2ExactGelu, pQuantizeInt8)
  else if ((ModelType = 'bert') or (ModelType = 'distilbert') or
           (ModelType = 'roberta')) and BertSeqCls then
  begin
    IgnoredId2Label := nil;
    Result := BuildBertForSequenceClassificationFromSafeTensorsEx(
      WeightsPath, IgnoredBertConfig, IgnoredId2Label, pSeqLen,
      pInferenceOnly, ConfigPath, pQuantizeInt8);
  end
  else if ModelType = 'gpt_neo' then
    Result := BuildGPTNeoFromSafeTensorsEx(WeightsPath, IgnoredGPTNeoConfig,
      pSeqLen, pInferenceOnly, ConfigPath, pQuantizeInt8)
  else if ModelType = 'gpt_neox' then
    Result := BuildGPTNeoXFromSafeTensorsEx(WeightsPath, IgnoredGPTNeoXConfig,
      pSeqLen, pInferenceOnly, ConfigPath, pQuantizeInt8)
  else if ModelType = 'gptj' then
    Result := BuildGPTJFromSafeTensorsEx(WeightsPath, IgnoredGPTJConfig,
      pSeqLen, pInferenceOnly, ConfigPath, pQuantizeInt8)
  else if (ModelType = 'cohere') or (ModelType = 'cohere2') then
    // Cohere Command-R / Aya (architectures ["CohereForCausalLM"]) and the
    // Command-R7B variant (["Cohere2ForCausalLM"]): parallel residual,
    // mean-subtracting weight-only LayerNorm, interleaved RoPE, tied
    // embeddings with logit_scale folded into the head. cohere2 adds the
    // alternating sliding/global pattern with NoPE on global layers. See
    // the COHERE IMPORT section.
    Result := BuildCohereFromSafeTensorsEx(WeightsPath, IgnoredCohereConfig,
      pSeqLen, pInferenceOnly, ConfigPath, pQuantizeInt8)
  else if ModelType = 'phi' then
    Result := BuildPhiFromSafeTensorsEx(WeightsPath, IgnoredPhiConfig,
      pSeqLen, pInferenceOnly, ConfigPath, pQuantizeInt8)
  else if (ModelType = 'llama') or (ModelType = 'mistral') or
          (ModelType = 'qwen2') or (ModelType = 'qwen3') or
          (ModelType = 'gemma') or (ModelType = 'gemma2') or
          (ModelType = 'gemma3_text') or (ModelType = 'phi3') or
          (ModelType = 'olmo2') or (ModelType = 'mixtral') then
    // 'gemma' (architectures ["GemmaForCausalLM"]) rides the same path: the
    // config reader raises the Gemma flags (GeGLU FFN, zero-centered
    // RMSNorm, sqrt(d) embedding scale) from model_type alone. 'gemma2'
    // (["Gemma2ForCausalLM"]) adds the Gemma-2 deltas (alternating
    // local/global attention, query_pre_attn_scalar, attention- and
    // final-logit soft-capping, sandwich norms) the same way.
    // 'gemma3_text' (["Gemma3ForCausalLM"], TEXT-ONLY) swaps the soft-caps
    // for the per-head q/k RMSNorm and adds the 5:1 local:global pattern
    // with per-layer-type RoPE theta. 'phi3' (["Phi3ForCausalLM"],
    // Phi-3-mini / Phi-4-mini) raises the fused qkv_proj/gate_up_proj and
    // partial-rotary flags the same way (the 128k longrope variants are
    // rejected by the config reader). 'olmo2' (["Olmo2ForCausalLM"]) raises
    // the reordered post-norm (x + Norm(Attn(x)), no input_layernorm) and
    // full-width q/k RMSNorm flags. 'mixtral' (["MixtralForCausalLM"])
    // raises the block_sparse_moe flags (per-FFN router gate +
    // num_local_experts SwiGLU experts, renormalized top-k routing).
    Result := BuildLlamaFromSafeTensorsEx(WeightsPath, IgnoredLlamaConfig,
      pSeqLen, pInferenceOnly, ConfigPath, pQuantizeInt8)
  else if (ModelType = 'bert') or (ModelType = 'distilbert') or
          (ModelType = 'roberta') then
    // ENCODER route: input (SeqLen,1,2) token|token-type ids (channel 1
    // ignored for distilbert, zeros for roberta), output (SeqLen,1,hidden)
    // final hidden states - NOT causal-LM logits (see the interface
    // comment). Pooler excluded; call BuildBertFromSafeTensors directly to
    // include it (bert/roberta - distilbert has none).
    Result := BuildBertFromSafeTensorsEx(WeightsPath, IgnoredBertConfig,
      pSeqLen, pInferenceOnly, {pIncludePooler=}false, ConfigPath,
      pQuantizeInt8)
  else if ModelType = 'rwkv' then
    // The first NON-TRANSFORMER route: RWKV-4 (architectures
    // ["RwkvForCausalLM"]), a recurrent WKV mixer - causal-LM contract
    // like the decoder families ((SeqLen,1,1) ids in, (SeqLen,1,vocab)
    // logits out). See the RWKV-4 IMPORT section.
    Result := BuildRWKVFromSafeTensorsEx(WeightsPath, IgnoredRWKVConfig,
      pSeqLen, pInferenceOnly, ConfigPath, pQuantizeInt8)
  else if ModelType = 'mamba' then
    // The second non-transformer route: Mamba (architectures
    // ["MambaForCausalLM"]), a selective-SSM mixer - causal-LM contract
    // like the decoder families ((SeqLen,1,1) ids in, (SeqLen,1,vocab)
    // logits out). See the MAMBA IMPORT section.
    Result := BuildMambaFromSafeTensorsEx(WeightsPath, IgnoredMambaConfig,
      pSeqLen, pInferenceOnly, ConfigPath, pQuantizeInt8)
  else if ModelType = 'bloom' then
    // The ALiBi route: BLOOM (architectures ["BloomForCausalLM"]), no
    // positional embeddings - per-head fixed linear attention biases
    // (TNNetALiBiAttention). Causal-LM contract like the other decoder
    // families. See the BLOOM IMPORT section.
    Result := BuildBloomFromSafeTensorsEx(WeightsPath, IgnoredBloomConfig,
      pSeqLen, pInferenceOnly, ConfigPath, pQuantizeInt8)
  else if (ModelType = 'falcon') or (ModelType = 'RefinedWebModel') or
          (ModelType = 'RefinedWeb') then
    // Falcon (architectures ["FalconForCausalLM"]; the original tiiuae
    // exports used model_type "RefinedWebModel" = falcon-7b and "RefinedWeb"
    // = falcon-40b): fused multi-query/GQA query_key_value, RoPE, parallel
    // attention+MLP residual with one (parallel_attn) or two (new arch,
    // ln_attn/ln_mlp) LayerNorms. Causal-LM contract like the other decoder
    // families. See the FALCON IMPORT section.
    Result := BuildFalconFromSafeTensorsEx(WeightsPath, IgnoredFalconConfig,
      pSeqLen, pInferenceOnly, ConfigPath, pQuantizeInt8)
  else if ModelType = 'modernbert' then
    // The second ENCODER route: ModernBERT (architectures
    // ["ModernBertModel"]; head-bearing exports load their base weights
    // under the "model." prefix) - input (SeqLen,1,1) token ids (NO
    // token-type channel, unlike the bert family), output (SeqLen,1,hidden)
    // final hidden states. See the MODERNBERT IMPORT section.
    Result := BuildModernBertFromSafeTensorsEx(WeightsPath,
      IgnoredModernBertConfig, pSeqLen, pInferenceOnly, ConfigPath,
      pQuantizeInt8)
  else if ModelType = 'deepseek_v2' then
    // DeepSeek-V2 (architectures ["DeepseekV2ForCausalLM"];
    // DeepSeek-V2-Lite is the reference checkpoint): Multi-head Latent
    // Attention (compressed KV latent + decoupled RoPE keys) and
    // DeepSeekMoE (shared + routed experts, softmax top-k gating) layers.
    // Causal-LM contract like the other decoder families. See the
    // DEEPSEEK-V2 IMPORT section for the wiring and the unsupported
    // full-V2 variants (q_lora_rank, YaRN-with-mscale rope_scaling,
    // group_limited_greedy).
    Result := BuildDeepSeekV2FromSafeTensorsEx(WeightsPath,
      IgnoredDeepSeekV2Config, pSeqLen, pInferenceOnly, ConfigPath,
      pQuantizeInt8)
  else if ModelType = 't5' then
  begin
    // T5 is an encoder-decoder: the import builds TWO nets, which this
    // single-net dispatch cannot return.
    Result := nil;
    ImportError('BuildFromPretrained: model_type "t5" builds an ' +
      'ENCODER-DECODER pair - call BuildT5FromSafeTensors (returns both ' +
      'nets; run them with RunT5) instead of this single-net dispatch.');
  end
  else if ModelType = 'marian' then
  begin
    // Marian / OPUS-MT is an encoder-decoder: the import builds TWO nets,
    // which this single-net dispatch cannot return.
    Result := nil;
    ImportError('BuildFromPretrained: model_type "marian" builds an ' +
      'ENCODER-DECODER pair - call BuildMarianFromSafeTensors (returns ' +
      'both nets; run them with RunT5) instead of this single-net ' +
      'dispatch.');
  end
  else if ModelType = 'bart' then
  begin
    // BART is an encoder-decoder: the import builds TWO nets, which this
    // single-net dispatch cannot return.
    Result := nil;
    ImportError('BuildFromPretrained: model_type "bart" builds an ' +
      'ENCODER-DECODER pair - call BuildBartFromSafeTensors (returns ' +
      'both nets; run them with RunT5) instead of this single-net ' +
      'dispatch.');
  end
  else if ModelType = 'whisper' then
  begin
    // Whisper is an encoder-decoder: the import builds TWO nets, which
    // this single-net dispatch cannot return.
    Result := nil;
    ImportError('BuildFromPretrained: model_type "whisper" builds an ' +
      'ENCODER-DECODER pair - call BuildWhisperFromSafeTensors (returns ' +
      'both nets; run them with RunT5, mel input from ' +
      'neuralaudio.ComputeWhisperLogMel) instead of this single-net ' +
      'dispatch.');
  end
  else if ModelType = 'clip' then
  begin
    // CLIP is a dual encoder: the import builds TWO independent nets
    // (text + vision), which this single-net dispatch cannot return.
    Result := nil;
    ImportError('BuildFromPretrained: model_type "clip" builds a ' +
      'TWO-net dual encoder (text + vision) - call ' +
      'BuildClipFromSafeTensors (returns both nets; pool/score with ' +
      'ClipTextEosPosition, ClipExtractEmbedding and ClipSimilarity) ' +
      'instead of this single-net dispatch.');
  end
  else
  begin
    Result := nil;
    ImportError('BuildFromPretrained: model_type "' + ModelType +
      '" (config ' + ConfigPath + ') is not supported. Supported ' +
      'model_types: gpt2, gpt_neo, gpt_neox, gptj, phi, phi3, llama, ' +
      'mistral, mixtral, qwen2, qwen3, gemma, gemma2, gemma3_text, rwkv, ' +
      'mamba, bloom, falcon, bert, distilbert, roberta, modernbert, ' +
      'deepseek_v2, olmo2.');
  end;
end;

end.
