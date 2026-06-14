# Task List — Feature & Enhancement Ideas

## Authorship convention (AI-coded classes)

Every class that was newly added to `neural/neuralnetwork.pas` by Claude
(i.e. not present in the upstream `../neural-master` baseline) carries an
attribution comment as the **last comment line directly above the class
declaration**, written exactly as:

```
  // Coded by Claude (AI).
```

Rules:
- One attribution per **class** (not per method), placed immediately above
  the `TNNet... = class(...)` line, after any `///`/`//` doc comment.
- Use the literal text `// Coded by Claude (AI).` (plain `//`, not `///`,
  trailing period) so it can be audited with
  `grep -c "Coded by Claude" neural/neuralnetwork.pas`.
- Applies only to genuinely **new** classes. Do NOT retrofit it onto
  pre-existing upstream classes that were merely edited.
- Human-authored hand-coding of new classes is no longer the norm here;
  new classes are Claude-authored and should be marked as such.

### Example programs (`examples/**/*.lpr`)

Every example program newly added by Claude (i.e. not present in the
`../neural-master` baseline) carries the attribution `Coded by Claude (AI).`
inserted with a blank-line separator immediately **before the closing `*)`**
of the file's header `(* ... *)` license comment block.

Rules:
- One attribution per file, inside the header comment block (so it never
  affects compilation).
- Applies only to genuinely **new** example `.lpr` files; skip stray
  `backup/` copies. Identify "new" by diffing `find examples -name '*.lpr'`
  against the `../neural-master` baseline.
- Audit with `grep -rl "Coded by Claude" examples --include='*.lpr' | wc -l`.

## DO NOT REINTRODUCE — removed by intent

The following layer types were intentionally removed because they
duplicated the forward pass of existing layers:

- `TNNetBias` — duplicate of `TNNetChannelBias` (forward pass).
- `TNNetLayerScale` (and its alias `TNNetLearnableScale`) — duplicate of
  `TNNetChannelMul` (forward pass).
- `TNNetNeg` — duplicate of `TNNetNegate` (which is itself just
  `TNNetMulByConstant.Create(-1)`). Use `TNNetNegate` for `y = -x`.
- `TNNetGlobalAvgPool` — empty-body subclass of `TNNetAvgChannel`. Use
  `TNNetAvgChannel` for global average pooling.
- `TNNetGlobalMaxPool` — overlapped `TNNetMaxChannel`. Use
  `TNNetMaxChannel` for global max pooling. Note: `TNNetMaxChannel`
  currently assumes square feature maps (`SizeX == SizeY`); the deleted
  `TNNetGlobalMaxPool` had a direct (X, Y) loop that also worked on
  rectangular inputs. If you ever need global max on a non-square
  tensor, fix `TNNetMaxChannel` rather than reintroducing this class.
- `TNNetGlobalMinPool` — overlapped `TNNetMinChannel`. Use
  `TNNetMinChannel` for global min pooling. Same square-only caveat as
  `TNNetMaxChannel` above.
- `TNNetThresholdedReLU` — strict subset of `TNNetThreshold`.
  `ThresholdedReLU(x; θ)` is `TNNetThreshold.Create(theta=θ, value=0)`.
  Use `TNNetThreshold` directly.

Do NOT add them back under any name. The minor differences they had
(true-sum vs spatial-mean weight-gradient scaling; constructor-
configurable initial scale) are not worth the code duplication. If a
true-sum gradient or a configurable initial multiplier is genuinely
needed, extend `TNNetChannelBias` / `TNNetChannelMul` with an option
flag instead of forking a new layer class. Any task entry below that
references these removed layers is obsolete and should be ignored
rather than acted on.


## Bugs

- [ ] `TNNetFlipX.Backpropagate` (and likely `TNNetFlipY`) range-check
      overflow when the NEXT layer is a padded convolution: the flip layer's
      `OutputError` is sized exactly to its output, but a padded conv writes a
      larger (padded) error region into it, overflowing. Surfaced while wiring
      an `Input -> FlipX -> Conv -> ...` flip-invariant net for
      EquivarianceReport (worked around by using a global-avg construction
      instead). Add a numerical-gradient / forward+backward regression test
      for `FlipX -> padded Conv` and fix the unpad sizing.
- [ ] FFT-path FPU denormal/invalid-op traps in TNNetSpectralConv2D needed an
      example-side SetExceptionMask workaround — consider masking/guarding the
      denormals inside the layer's FFT so callers don't have to.
- [ ] ulimit -v 3000000 ChatTerminal /path/to/Qwen2.5-0.5B-Instruct crashes using all RAM at loading the model
      Tested with !git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct q2
- [ ] SentencePiece `.model` `precompiled_charsmap` parsing — the opaque
      per-model normalization trie (NOT standard NFKC) is still not
      parsed/applied. A tokenizer.json that declares a standard NFKC/NFKD
      normalizer already works in full (`UnicodeNormalize` + `AddNormalizer`,
      landed); only the embedded charsmap is unhandled.

## Infrastructure / dev experience

- [ ] Gradient checkpointing for training deeper nets in less memory
- [ ] ONNX import
- [ ] Gemma 4 import
- [ ] Qwen 3.5 import
- [ ] IBM Granite 3.x importer follow-ups (`BuildGraniteFromSafeTensors[Ex]`
      LANDED, model_type `granite`/`granitemoe`; four multipliers folded at
      load on the Llama path — embedding/residual/attention multipliers + a
      DIVIDING logits_scaling into the LM head, wired through the existing
      QScale/W_q slot not hardcoded; granitemoe reuses the Mixtral MoE WIRING
      with a dedicated LoadGraniteMoEExperts loader for its fused 3-D slabs;
      pico parity TestGranite{Config,Logit,MoeLogit}Parity < 1e-4 vs float64
      HF transformers):
  - [ ] real-checkpoint slicer (make_pico_*_fixture.py reuse) to parity-check
        a sliced ibm-granite/granite-3.1-* against the random pico fixture.
- [ ] OLMoE real-checkpoint slicer follow-up (BuildOlmoeFromSafeTensors[Ex],
      model_type `olmoe`, LANDED: OLMo backbone with QK-norm full-width +
      original-OLMo pre-norm placement × Qwen3-MoE/Mixtral top-k MoE wiring,
      uniformly all-MoE; pico parity TestOlmoeLogitParity < 1e-4 vs float64 HF
      OlmoeForCausalLM): the pico fixture is random-init only — add a real
      allenai/OLMoE-1B-0924 slicer (slice_llama.py reuse) to parity-check a
      sliced real checkpoint against the random pico fixture.
- [X] BitNet b1.58 importer (`BuildBitNetFromSafeTensors[Ex]` LANDED,
      model_type "bitnet" / the released `microsoft/bitnet-b1.58-2B-4T`) — the
      ternary-weight LLM family, mapped onto the Llama backbone (RMSNorm, RoPE)
      with the BitNet deltas wired onto the existing builder slots:
      (a) the SubLN "norm-before-quantized-linear" RMSNorm — attn_sub_norm on
      the concatenated head outputs BEFORE o_proj and ffn_sub_norm on the gated
      activation BEFORE down_proj (Config.BitNetSubLN, distinct from the
      Gemma-2/OLMo-2 post-norms; no new layer class, reuses TNNetTokenRMSNorm);
      (b) the relu2 gated FFN (hidden_act "relu2") via the new
      TNNetReGLUSquared activation (up*ReLU(gate)^2) with SEPARATE
      gate_proj/up_proj on the standard separate-projection loader path; and
      (c) rope_theta read from the transformers-5.x rope_parameters dict
      (default 500000). The ternary weights ride the standard de-quant-at-load
      convention (the GGUF Q8_0 / MXFP4 precedent): the HF transformers
      BitNetForCausalLM checkpoint is the FP "shadow" form storing the
      already-ternary effective weights (scale*{-1,0,+1}) and runs them through
      plain nn.Linear, so loading them straight is bit-for-bit (the absmean
      ternarize is a no-op round-trip on already-ternary weights). Pico parity
      fixture tools/bitnet_tiny_fixture.py re-randomizes + ternarizes the
      projections and ASSERTS the ternarization moves the HF logits
      (non-vacuous); TestBitNet{Config,Logit}Parity < 1e-4 vs float64 HF
      BitNetForCausalLM, both passing. Headline: a 2B model that resides in
      well under 1GB, dovetailing with the int8/GGUF "commodity RAM" theme.
  - [ ] native I2_S packed-ternary de-quant-at-load (2-bit unpack: 4 ternary
        values per byte + a separate weight_scale tensor). The shadow-weight
        (FP effective-weights) path is covered; the GGUF/bitnet.cpp-style I2_S
        packed format is NOT yet read (the HF transformers checkpoint ships
        the shadow form, so the parity fixture exercises only that path).
  - [ ] real-checkpoint slicer (slice_llama.py reuse) to parity-check a sliced
        microsoft/bitnet-b1.58-2B-4T against the random pico fixture — SKIPPED
        (network/RAM-gated: requires downloading the 2B checkpoint).
- [ ] M2M100/NLLB translate demo + real-vocab check (follow-up to the landed
      BuildM2M100FromSafeTensors, commit cb21550): an examples/NLLBTranslate
      seq2seq demo that loads a real (small) NLLB/M2M100 checkpoint and
      round-trips a translation on CPU via DecodeSeq2SeqGreedy/BeamSearch, plus
      verifying the importer against a downloaded sentencepiece.bpe.model vocab
      (the parity fixture today uses raw token ids only). Blocked partly on the
      BPE-in-.model tokenizer follow-up for NLLB-BPE variants (see the
      neuralhftokenizer Tokenizer follow-up (b) entry).
- [ ] GPT-OSS importer follow-ups (BuildGptOssFromSafeTensors[Ex] LANDED,
      model_type "gpt_oss", with TNNetGptOssSinkAttention per-head scalar sink +
      alternating sliding/full window + YaRN truncate flag + top-k MoE +
      TNNetGptOssGatedSwiGLU clamped experts + MXFP4 dequant-at-load; pico parity
      TestGptOssLogitParity / TestGptOssMXFP4LogitParity <1e-4): (a) MXFP4 -> int8
      weight-only STORAGE path — pQuantizeInt8 currently re-quantizes the
      dequantized FP32 rather than a direct 4->8 transcode; (b) no real-weight
      20B/120B smoke run yet (RAM-gated; pico parity only).
- [ ] Jamba importer follow-ups (BuildJambaFromSafeTensors[Ex] LANDED, model_type
      "jamba", first hybrid Mamba+attention+MoE import; TNNetSelectiveSSM
      CreateJambaInner inner-norm mode; pico parity TestJambaLogitParity <1e-4):
      (a) mamba_proj_bias=true (in/out_proj biases) is REJECTED — wire the bias
      path if a checkpoint needs it; (b) training through the Jamba inner-norm SSM
      is forward-only — add backward through the dt/B/C RMSNorms for fine-tuning;
      (c) no Jamba-specific tokenizer demo (raw ids work); (d) no real-checkpoint
      slicer fixture (parity uses a random pico model).
- [ ] LLaVA-style GENERATIVE vision-language import — image-conditioned text
      generation, the capability step past the landed CLIP dual encoder
      (which only scores image/text similarity and cannot generate).
      Target a small open checkpoint on the classic LLaVA recipe (e.g.
      llava-hf/llava-interleave-qwen-0.5b-hf or a SmolVLM-class model —
      whichever config maps cleanest onto existing paths): ViT vision
      tower -> 2-layer MLP projector (gelu) -> visual tokens SPLICED into
      the decoder's token-embedding sequence at the <image> placeholder
      position, then ordinary causal decoding. Nearly everything exists:
      BuildClipVisionTower is the reusable ViT (LLaVA uses the
      penultimate-layer patch tokens WITHOUT the CLS row and WITHOUT the
      projection head — needs a select-hidden-layer/skip-pooling mode),
      the language side is the stock Llama/Qwen2 path, and decode/chat
      infra (KV cache, samplers, ApplyChatTemplate) is landed. The genuinely
      new pieces: (a) the projector import + a prompt-assembly helper that
      runs the vision tower once and concatenates [text-embeds | projected
      image tokens | text-embeds] as a TNNetInput-fed embedding sequence
      (in-repo precedent: T5EncoderStatesInput external-states convention),
      (b) image preprocessing to the processor's normalized RGB tensor
      (CLIP-style resize/center-crop, mean/std from preprocessor_config.json
      — also unblocks the ClipZeroShot real-image follow-up), (c) the
      multimodal chat template ("USER: <image>\n...ASSISTANT:" or
      ChatML-with-image variant) wired into neuralchat.pas. Deliverables:
      BuildLlavaFromSafeTensors[Ex] (two nets + projector, multi-shard
      index.json support), pico parity fixture via the make_pico_*_fixture.py
      recipe asserting projected visual tokens AND next-token logits for a
      mixed image+text prompt vs HF float64, and an examples/LlavaDescribe
      demo that captions a small image on CPU. First image-in/text-out
      model in the repo; opens the door to Qwen-VL/PaliGemma later.
- [ ] Cohere real-checkpoint slicer follow-up (BuildCohereFromSafeTensors[Ex]
      for cohere + cohere2 LANDED on a dedicated parallel-residual builder,
      parity 3.96e-7/2.15e-7 vs HF float64 against SYNTHETIC config-faithful
      pico fixtures): add a real-weight slicer for Aya-Expanse-8B /
      Command-R7B (like slice_llama.py) so parity is checked against real
      Cohere weights on top of the synthetic fixture. Also: order_of_interleaved_layers
      (legacy cohere2 spelling) maps to sliding_window_pattern but was not seen
      in a published config — wire it if one surfaces.
- [ ] ONNX (or simpler JSON) export path — minimal viable: dump a
      forward-only graph for the currently-supported subset of layers,
      enough to run inference in onnxruntime. Doc which layers are
      out-of-scope for v1.
- [ ] Quantized inference follow-up (int8 weight-only v1 is landed): int8
      activation quantization + true
      int8 matmul kernels (AVX2 maddubs / dot-product paths) so quantized
      layers stop paying the per-forward dequantize cost; today the int8
      win is RAM only — compute still runs the FP32 kernels.
- [ ] Quantized inference follow-up: GPTQ/AWQ-style calibrated quantization
      (error-compensating rounding / activation-aware scale search) and
      4-bit (int4 pairs packed per byte, group-wise scales); also quantized
      EMBEDDING table storage (the one remaining FP32 heavyweight — vocab x
      d_model stays FP32 in v1, e.g. ~262MB for TinyLlama) and FP16/BF16
      weight storage as a zero-drift middle rung. GGUF dovetails (reader
      landed in neural/neuralgguf.pas): load Q8_0 blocks straight into the
      int8 weight-only storage instead of dequantize-then-requantize (the
      k-quant Q4_K/Q6_K/Q5_K/Q2_K dequant-at-load READ path has landed in
      neural/neuralgguf.pas).
- [ ] Quantized inference follow-up: upstream fix for TVolume.GetMaxAbs
      (seeds the running max with the SIGNED first element, so a negative
      max-magnitude element 0 is missed; csErrorOverflowBackpropProtection
      and friends consume it — audit callers before fixing).
- [ ] Tokenizer follow-ups for neuralhftokenizer.pas:
      (b) DONE — raw SentencePiece .model protobuf path landed
      (LoadSentencePieceModel; hand-decoded ModelProto wire format, no
      vendored proto lib; auto-dispatched from LoadFromFile by the '.model'
      extension or a non-'{' first byte). Populates the SAME Unigram
      structures as the tokenizer.json path; pico fixture + spm-oracle parity
      test (tools/make_pico_spm_fixture.py, tests/fixtures/tiny_spm.model,
      TestSentencePieceModelParity). UNIGRAM model_type only; a BPE/WORD/CHAR
      ModelProto raises EHFTokenizerError ("use tokenizer.json"). OPEN
      follow-ups: BPE-in-.model (NLLB/mBART-BPE exports — needs the merges
      decoded from the ModelProto pieces or via tokenizer.json). BYTE
      (type=6) byte-fallback is now DONE: LoadSentencePieceModel sets
      FByteFallback when any type-6 piece is present, so (1) encode routes
      unknown chars through the <0xNN> BYTE pieces — the Unigram Viterbi
      unk node now decomposes the char into its UTF-8 bytes via byte
      pieces (EmitByteFallback) instead of one fused <unk>; and (2) decode
      emits the RAW byte for each <0xNN> piece (DecodeToken), so a run of
      byte pieces reassembles into the original multi-byte UTF-8 string
      (the metaspace U+2581 handling is unaffected). Round-trip parity test
      TestSentencePieceByteFallbackRoundTrip + tiny_spm_bytefallback.model
      fixture (tools/make_pico_spm_fixture.py byte_fallback variant; cases
      read via TJSONParser(s,[]) to keep their UTF-8 bytes intact). This
      partially UNBLOCKS the mBART/NLLB BART-family follow-up (b) and the
      DeBERTa-v3 import (both ship raw sentencepiece .model and are
      Unigram). (c) exact full-Unicode \p{L}/\p{N} tables (current
      classifier covers Latin/Greek/Cyrillic/Armenian/Hebrew/Arabic/
      Devanagari/Kana/CJK/Hangul; exotic scripts fall into the
      punctuation class of the GPT-2 regex); (d) DONE — build a tokenizer from
      GGUF tokenizer.ggml.* metadata (TNeuralHFTokenizer.LoadFromGGUF: opens
      via TNNetGGUFReader, reads tokenizer.ggml.model + tokens/scores/
      token_type [+ merges] and the bos/eos/unknown_token_id scalars, and
      populates the SAME internal structures the tokenizer.json/.model loaders
      use), so a single .gguf is self-contained for generation. SUPPORTED
      tokenizer.ggml.model values: "llama" (SentencePiece-with-scores →
      Unigram/Viterbi path: metaspace U+2581, <0xNN> byte fallback, CONTROL/
      UNKNOWN/USER_DEFINED pieces exposed as added tokens) and "gpt2" (byte-
      level BPE → GPT-2 byte table + cl100k Split pre-tokenizer + merges).
      DEFERRED: "bert" (WordPiece) and "no_vocab" raise EHFTokenizerError;
      gpt2 path assumes the cl100k Split pre-tok (GGUF carries no pre-tok
      config). Oracle = write-then-read round-trip: a tokenizer.json fixture's
      vocab/scores(+merges) is emitted to a temp .gguf via TNNetGGUFWriter and
      read back, then Encode/Decode are asserted id-identical to the
      tokenizer.json source (TestLoadFromGGUF{Llama,Gpt2}RoundTrip,
      TestLoadFromGGUFRejectsUnsupportedModel in TestNeuralHFTokenizer.pas).
- [ ] neuralhftokenizer.pas pre_tokenizer leftovers from the Split/Metaspace
      batch: ~~(a) a STANDALONE ByteLevel pre_tokenizer with use_regex=false
      silently applies the GPT-2 regex anyway (the flag is only honored
      implicitly inside the Sequence[Split, ByteLevel] path, which bypasses
      ByteLevelPieces; parse use_regex and skip the regex split when false)~~
      DONE: FByteLevelUseRegex parsed; standalone ByteLevel(use_regex=false)
      now feeds the whole segment as ONE chunk to the byte alphabet (no GPT-2
      regex). Test TestByteLevelNoRegexParityWithHF + bytelevel_noregex
      fixture group.
      (b) only the Qwen2 and Llama-3/cl100k Split pattern literals are
      recognized — the o200k (GPT-4o-family) and DeepSeek pattern strings
      raise EHFTokenizerError. ~~the DeepSeek family is now importable
      (BuildDeepSeekV2FromSafeTensors), so its pattern is the live gap~~ DONE
      for DeepSeek-V2/V2-Lite: the multi-Split+Digits Sequence is detected as
      a WHOLE (MatchesDeepSeekSequence) and dispatched to SplitDeepSeekPieces
      (a distinct splitter: \s?-prefixed letter/punct runs, separate CJK run,
      individual digits, \s+$ trailing). Test TestSplitDeepSeekParityWithHF +
      split_deepseek fixture. REMAINING sub-tasks: (i) DONE — o200k_base
      (GPT-4o family: gpt-4o/gpt-4o-mini/o1/o3) Split pattern recognized
      verbatim (csO200kSplitPattern) and dispatched to a dedicated
      case-aware SplitO200kPieces splitter (two letter alternations
      [\p{Lu}...]*[\p{Ll}...]+ and [\p{Lu}...]+[\p{Ll}...]*, \p{N}{1,3}
      digits, the [\r\n/]* punct trailing run that adds '/'). The Lu/Ll
      split uses the ASCII case tables; non-ASCII letters fall into the
      Ll/Lo class, so byte parity is guaranteed over ASCII letters (incl
      case transitions like HelloWorld->Hello+World, iPhone->i+Phone),
      digits, punct (incl '/'), and whitespace — the SAME approximation
      stance as the cl100k/DeepSeek splitters. Test
      TestSplitO200kParityWithHF + split_o200k fixture group
      (tools/hf_pretok_fixture.py build_o200k_split). RESIDUAL GAP: non-ASCII
      Latin AND CJK parity for SplitO200kPieces (would need the exact o200k
      Lu/Ll/Lo class tables ported to Pascal); (ii) DeepSeek-V3
      pre-tokenizer (different shape: \p{N}{1,3} + packed punct+letter Split,
      NO Digits stage) — not matched by MatchesDeepSeekSequence, falls through
      and raises; add a V3 variant when V3 is imported; (iii) non-ASCII Latin
      parity for SplitDeepSeekPieces — the HF onig engine isolates some
      precomposed Latin-1 letters that the \p{L} table groups, so exact byte
      parity is only guaranteed over ASCII/digit/punct/whitespace/CJK (the
      same approximation stance as the cl100k splitter; would need the exact
      DeepSeek letter-class range table ported to Pascal). Test: per-pattern
      parity fixtures like tools/hf_pretok_fixture.py.
- [ ] Parameter groups for the optimizer (PyTorch param_groups port):
      per-group learning-rate multipliers and weight-decay exclusion for
      norm/bias parameters (AdamW currently decays everything uniformly).
- [ ] HF-names safetensors exporter — per-importer name maps follow-up
      (GPT-2 SaveGPT2ToSafeTensors and Llama SaveLlamaToSafeTensors landed as the
      exact inverses of their importers — Llama via the shared
      BuildLlamaMemTensorReader walk that also backs the GGUF exporter, undoing
      the q/k rotate_half de-permute + SwiGLU gate|up un-fuse + Gemma RMSNorm
      gain offset; round-trip gated by TestLlamaSafeTensorsRoundTrip; QWEN3
      SaveQwen3ToSafeTensors LANDED via the dedicated BuildQwen3MemTensorReader
      walk — same q/k rotate_half de-permute + SwiGLU gate|up un-fuse PLUS the
      per-head q/k RMSNorm gain un-permute, round-trip gated by
      TestQwen3SafeTensorsRoundTrip; BERT SaveBertToSafeTensors LANDED as the
      exact inverse of BuildBertFromSafeTensors — walks the encoder, un-fuses the
      Q|K|V slab into the three nn.Linear tensors and dumps straight [out,in]
      linears (+bias) / LayerNorm gamma-beta / word|type|position embeddings
      under the family name map (bert/distilbert/roberta prefixes), round-trip
      gated by TestBertSafeTensorsRoundTrip; GPT-NeoX SaveGPTNeoXToSafeTensors
      LANDED as the exact inverse of BuildGPTNeoXFromSafeTensors — collects the
      decoder's typed layers in build order and RE-FUSES the per-head Q|K|V slab
      back into the single interleaved query_key_value tensor (inverting the
      partial rotate_half permutation of LoadGPTNeoXQKVWeights) plus straight
      [out,in] dense/dense_h_to_4h/dense_4h_to_h linears, LayerNorm gamma|beta,
      embed_in / embed_out under the gpt_neox.* name map, round-trip gated by
      TestGPTNeoXSafeTensorsRoundTrip over BOTH parallel and sequential
      configs; BLOOM SaveBloomToSafeTensors LANDED as the exact inverse of
      BuildBloomFromSafeTensors — collects the decoder's typed layers in build
      order and RE-FUSES the per-head Q|K|V slab back into the single
      query_key_value tensor (NO rotate_half — BLOOM is ALiBi, not rotary) plus
      straight [out,in] dense/dense_h_to_4h/dense_4h_to_h linears, LayerNorm
      gamma|beta INCLUDING the word_embeddings_layernorm, tied word_embeddings
      (no separate lm_head tensor) and ln_f under the transformer.* name map,
      round-trip gated by TestBloomSafeTensorsRoundTrip; Mamba
      SaveMambaToSafeTensors LANDED as the exact inverse of
      BuildMambaFromSafeTensors — the first NON-TRANSFORMER exporter: walks the
      typed layers (Embedding / per-block TokenRMSNorm+in_proj PointwiseConvLinear
      +DepthwiseConv1D+SelectiveSSM+out_proj / norm_f / LM head) in build order
      and emits the backbone.* name map; A_log/D/dt_proj.bias and the B|C x_proj
      rows round-trip RAW, conv1d/in_proj/out_proj are straight dumps, and the
      one non-trivial inversion — the importer's FOLD of dt_proj.weight @
      x_proj.weight[0:dt_rank] into the single rank-<=dt_rank [d_inner,d_inner]
      W_d — is re-factored EXACTLY at export by Gaussian elimination (emit
      dt_proj.weight=L, x_proj rows[0:dt_rank]=U with L@U=W_d); tied
      tie_word_embeddings emits no lm_head tensor; round-trip gated by
      TestMambaSafeTensorsRoundTrip, max |logit diff| = 4.4e-6 (single-precision
      rounding of the factor product, well under 1e-5); RWKV
      SaveRWKVToSafeTensors LANDED as the exact inverse of
      BuildRWKVFromSafeTensors — the second NON-TRANSFORMER exporter: walks the
      typed layers (Embedding / ln0 pre_ln / per-block ln1+ln2 TokenLayerNorm,
      5 TokenShift lerp vectors, 7 PointwiseConvLinear projections, WKV / ln_out
      / tied|untied head) in build order and emits the rwkv.* name map; every
      tensor round-trips RAW except time_decay, which is reconstructed by the
      FORWARD softplus time_decay=ln(softplus(w_raw)) — the EXACT inverse of
      LoadWKVDecay's invsoftplus (same w_raw>30 and exp-limit branch bounds), so
      the round-trip is BIT-EXACT; LayerNorms dump gamma|beta, projections are
      straight bias-free [out,in] dumps, time_first bonus and token-shift lerps
      round-trip raw, tied tie_word_embeddings emits no head.weight tensor;
      round-trip gated by TestRWKVSafeTensorsRoundTrip, max |logit diff| = 0
      (bit-exact)): add a layer->HF-name + transpose inverse map for the
      REMAINING architectures — only the ENCODER-DECODER importers (T5/Marian,
      each its own two-net map) now remain on this entry.
- [ ] GGUF writer follow-up: write Q8_0 STRAIGHT from the int8 weight-only
      storage ([[int8-quantized-inference]]) instead of quantizing-on-write
      from F32 (avoids the dequantize-then-requantize round trip when the
      source layers already hold int8 blocks).
- [ ] GGUF writer follow-up: byte-level-BPE end-to-end model export
      (SaveTokenizerToGGUF gpt2/llama tokenizer block + verify_gguf_writer.py
      llama-cpp-python logit-parity hook LANDED): SaveLlamaToGGUFEx itself still
      only takes a plain `Tokens` array (SP "llama" model) — route the gpt2
      tokenizer block through SaveTokenizerToGGUF from the MODEL exporter (not
      just the tokenizer unit test). Also: the llama-cpp-python parity arm is
      wired but UNVERIFIED end-to-end (lib not installed here; pico demo GGUF may
      be too small for llama.cpp) — confirm argmax/ranking agree on a real
      checkpoint once the lib is available.
- [ ] Stochastic Weight Averaging (torch.optim.swa_utils port): equal-weight
      running average of checkpoints over the schedule tail + a constant or
      cyclic SWA learning rate phase; swap averaged weights in for eval/save.
      Distinct from the landed EMA shadow-weights wiring (running decay
      average) but should share the shadow-weights machinery. NOTE: the EMA
      task landed its trainer wiring
      (TNeuralFitBase.EnableEMA + ApplyEMAWeights/RestoreLiveWeights store-and-
      restore swap) on TNNetEMAWrapper; TNNetSWAWrapper already exists too, so
      SWA is mostly a matter of wiring that wrapper into TNeuralFitBase reusing
      the same Apply/Restore swap plumbing.
- [ ] Optimizer zoo expansion (only SGD/Adam/AdamW exist today):
      Adafactor (factored second-moment estimate, drastically less optimizer
      state — pairs with the "run big imported models on commodity RAM"
      quantization theme), Lion (sign-based update, single momentum buffer,
      half of Adam's state), and optionally Muon for 2-D weight matrices
      (a hand-rolled Muon gradient-surgery demo already exists in
      examples/MuonOptimizer; the optimizer-class port is what's missing).
      Each is a small TNeuralOptimizer subclass in neuralfit.pas.
- [ ] ReduceLROnPlateau + OneCycle / cyclical LR schedulers
      (neural/neuralscheduler.pas has Step/CosineAnnealing/WarmupCosine/Poly):
      plateau-driven decay needs a hook feeding the validation metric into
      NextLR — that wiring is the interesting part; OneCycle/CyclicLR are
      straightforward NextLR formulas.
- [ ] Trainer callbacks API (transformers TrainerCallback port): a
      TNeuralFitCallback with OnEpochBegin/End, OnStepEnd, OnEvaluate hooks
      registered on TNeuralFitBase. Early stopping, custom logging, and the
      EMA/SWA tasks become small callbacks instead of ever more
      TNeuralFitBase fields.
- [ ] CutMix training augmentation (torchvision transforms-v2 staple;
      Mixup itself is landed: CreateMixedVolumePairList in neuralvolume +
      examples/Mixup): patch a random rectangle from a second sample into
      the input and mix the targets by area fraction (Beta-distributed
      lambda). The CIFAR image-classification examples give an instant
      bake-off harness.
- [ ] Batched generation with left-padding in neural/neuraldecode.pas:
      generate for N prompts in one forward pass per step (today's decode
      paths look single-sample). Makes evaluation sweeps cheap and is a
      prerequisite for an efficient speculative-decoding verify step.
- [ ] Chat templates v2 (v1 is landed in neural/neuralchat.pas —
      ApplyChatTemplate with seven hardcoded formats + DetectChatFormat
      fingerprinting + EncodeChat): ~~a mini-Jinja subset interpreter for
      unrecognized chat_template strings (must pass ground truth for all
      bundled templates and raise cleanly on unsupported constructs)~~
      [DONE: TJInterp / RenderChatTemplate + ApplyChatTemplateString
      fallback path; reproduces all 9 bundled templates byte-exact incl.
      whitespace control, raises EChatTemplateError on unsupported
      constructs; tests in TestNeuralHFTokenizer.pas]; more
      formats (~~DeepSeek~~ [DONE: cfDeepSeek], ~~Phi-4-mini's tool-aware
      ChatML variant~~ [DONE: cfPhi4Mini], Qwen's default-system injection);
      ~~read the separate chat_template.jinja file newer transformers exports
      alongside tokenizer_config.json~~ [DONE: LoadChatTemplateString sibling
      fallback]; continue_final_message / return_assistant_tokens_mask
      equivalents.
- [ ] Per-layer profiler report (torch.profiler lite): TNNet.ProfileReport
      with forward/backward wall-time and parameter/activation memory per
      layer (introspection-report pattern). Directly serves the open
      chunked-forward throughput tasks by showing where time actually goes.
- [ ] FlashAttention-style tiled online-softmax SDPA forward: opt-in fast
      mode — not for GPU speed but for O(L*d) vs O(L^2) attention-score
      MEMORY on long sequences; gate behind an exact-vs-naive equivalence
      assert, same pattern as the chunked-forward recurrence family.
- [ ] longrope short-factor / dynamic switching follow-up (static long-context
      import landed): the import statically picks the long_factor table + long
      attention scaling. HF switches to short_factor when seq_len <=
      original_max_position_embeddings. Add a decode-time mode that selects the
      short table for short sequences (or document that the static long import
      is intentional for the 128k use case).
- [ ] CFG follow-up: a full-width-net -> width-1 unconditional-twin auto-clone
      so MakeUnconditionalTwin (commit 48e2fd2) works from a single imported
      model with no hand-built width-1 net. A SaveToString->LoadFromString
      round-trip preserves the ORIGINAL input width, so this needs either a
      rebuild-architecture-at-width-1 + CopyWeights walk (per-importer, or a
      generic layer-shape-independent rebuilder) or a width-rewrite on the
      serialized string. Assert the twin's logits match the source on a pinned
      input (the existing TestMakeUnconditionalTwinMatchesSourceLogits is the
      template).
- [ ] KV-cache eviction for unbounded streaming: attention sinks + rolling
      window (StreamingLLM; transformers SinkCache) in TNNetStreamingDecoder
      — today the per-SDPA-layer cache grows without bound. Keep the first
      NumSinks tokens plus a fixed window, evicting the middle, so streamed
      generation runs forever in constant memory. Note TNNetSinkAttention is
      the TRAINING-side cousin (learnable sink logits); this is the
      decode-side cache policy and needs care with RoPE positions after
      eviction (re-rotate or cache pre-RoPE K). Assert: while output length
      <= window the streamed tokens are bit-identical to the unbounded
      cache, and memory stays flat past it.
- [ ] KV-cache quantization (int8 cache with per-channel scales): the
      landed int8 quantized inference covers WEIGHT storage; for long-context
      decode the KV cache dominates memory instead. Quantize cached K/V
      blocks to int8 on append, dequantize on read inside
      TNNetStreamingDecoder; assert logit drift vs the FP32 cache stays
      within a documented tolerance on the pico-Llama parity fixture.
- [ ] Preference-optimization follow-ups on the landed DPO/GRPO trainers
      (TNeuralGRPOTrainer in neural/neuraldpo.pas LANDED: group-relative
      advantages + PG + DeepSeek-k3 per-token KL reusing the DPO softmax-backward
      plumbing, tests in tests/TestNeuralGRPO.pas): ORPO / SimPO / KTO
      loss-formula deltas on the landed DPO trainer, and a Bradley-Terry pairwise
      reward-model trainer to feed GRPO real (learned) rewards.
- [ ] Offset-mapping follow-up: EncodeWithOffsets (commit 1e90b8a) is a
      post-hoc surface-match heuristic (each token's DecodeToken surface
      located forward at the running cursor), so it leaves tokens unmapped
      when the decoded surface can't be found at the cursor (added/special
      tokens, byte-fallback fragments). Add a byte-exact trace through the
      byte-level-BPE merge state for a guaranteed alignment, and a test over
      a pinned string with a byte-fallback / multi-byte-UTF8 token that the
      heuristic currently leaves unmapped.
- [ ] Length-grouped batching + dynamic padding collator (transformers
      LengthGroupedSampler + DataCollatorWithPadding port) in neuralfit:
      sort/bucket variable-length text by length, batch neighbors, pad each
      batch only to its own max (not the global max) — a large real-world
      throughput win. Complements (distinct from) the per-sample-attention-
      mask and left-padded-generation tasks: this is the TRAINING data-side
      half. Test: identical loss trajectory vs naive padding at fixed seed
      modulo batch order, plus a padded-token-count reduction assert.
- [ ] Sequence-length warmup curriculum in neuralfit.pas: train at short
      context first and grow SeqLen on a schedule (the rebuild-same-
      architecture-at-a-new-width idiom this list already notes near the
      variable-context trick) — large early-epoch throughput win for LM
      pretraining. Needs CopyWeights across width rebuilds + a schedule
      hook; the Trainer-callbacks task above is the natural home. Test:
      weights survive a width hop bit-for-bit, loss continuous across the
      hop.
- [ ] Streaming/lazy tensor materialization with load-time quantization:
      the import path materializes full FP32 tensor buffers before copying
      into layers, so PEAK import memory, not steady-state, can be the gate
      on commodity RAM (the TinyLlama ~4.4GB case). Read one tensor at a
      time from the (seekable) safetensors stream straight into the
      destination layer — or straight into the landed int8 storage —
      keeping only one tensor-sized scratch buffer. Assert peak RSS during import
      stays within tensor-size + model-size on the parity fixture.
- [ ] NumPy .npz follow-up (neural/neuralnumpy.pas reader/writer landed; WRITER
      is STORED-only): DEFLATE-compressed .npz WRITER (savez_compressed) +
      zip64 / >4GB archive support (reader currently rejects zip64).
- [ ] TinyStories reference-vs-from-scratch perplexity bake-off (follow-up
      to the landed, parity-verified roneneldan/TinyStories-1M import on
      the GPT-Neo route; the published pytorch_model.bin-only checkpoints
      now load DIRECTLY via the landed TNNetTorchBinReader — no Python
      conversion step needed, see examples/GPT2Import/README.md): the
      repo's NLP examples already train
      on the TinyStories dataset — compare neuralnlpmetrics perplexity of
      a from-scratch CAI training run against the imported
      roneneldan/TinyStories-1M..33M reference at matched vocab/context;
      at 1M-33M params the comparison runs against FULL checkpoints
      instead of sliced fixtures, the only importer family where that is
      true.
- [ ] RWKV-4 decode-side demo: flat-memory recurrent decoding vs a
      transformer of equal size (constant-memory headline of the landed
      BuildRWKVFromSafeTensors importer; needs an incremental TNNetWKV
      state-carry path).
- [ ] Mamba decode-side demo: tokens/sec flat in context length where a
      transformer of equal size slows (constant-memory headline of the
      landed BuildMambaFromSafeTensors importer; needs an incremental
      TNNetSelectiveSSM state-carry path, the sibling of the RWKV-4
      decode demo task above).
- [ ] Forced-prefix seq2seq decode + KV cache for Whisper-style decoders:
      DecodeSeq2SeqGreedy/Sampled assume a text encoder input and a
      single BOS start token, so examples/WhisperTranscribe hand-rolls
      its decode loop (mel-volume encoder input + the 4-token
      <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
      prologue, full prefix re-encoded every step). Add a seq2seq decode
      variant taking an arbitrary forced token prologue and a
      pre-computed encoder-states volume, plus decoder-side KV caching
      (the cross-attention K/V are fixed after encoding — cache them
      once). Assert greedy output matches the re-encoding loop
      bit-identically; would cut WhisperTranscribe's ~5 min CPU decode
      substantially. Note: the WhisperTranscribe example needs ~4 GB
      VIRTUAL memory (ulimit -v 4000000; the 3 GB test cap aborts during
      build).
- [ ] KV-cache beam search (cache forking): DecodeBeamSearch takes a plain
      TNNet and RE-ENCODES the whole prefix every step — the streaming-
      decode docs explicitly note only greedy/sampled streamed generation
      is exact today. Add a fork/clone primitive to TNNetStreamingDecoder
      (copy, or copy-on-write, of per-layer cache state per surviving
      hypothesis) and a cache-backed DecodeBeamSearch variant on top.
      Turns beam from O(L^2) per hypothesis into O(L); the fork primitive
      is also what best-of-N and the speculative-decoding verify step
      want. Assert ranked beam output is identical to the re-encoding
      implementation.
- [ ] Prefix/session cache reuse in TNNetStreamingDecoder: no way today to
      save/restore or fork a session, so a shared system prompt is
      re-prefilled for every generation. Add snapshot-after-prefill +
      clone-per-request (and optional save/load to disk for a persistent
      system-prompt cache). The single biggest practical speedup for the
      landed chat-templates use case; shares the fork primitive with the
      KV-cache beam task above. Assert a forked session's continuation is
      bit-identical to a fresh prefill. Related consumer: the landed
      examples/ChatTerminal decodes with one full fixed-width forward per
      token because importers build at full context width — an importer
      option to build a width-1 decode twin (or build-twice +
      CopyWeights) would let chat ride TNNetStreamingDecoder.
- [ ] Early-exit / self-speculative decoding (LayerSkip / CALM): decode
      easy tokens from an intermediate layer through the LM head, fall
      back to full depth when confidence is low — the model becomes its
      OWN draft model, no second checkpoint. Distinct from the landed
      examples/SelfSpeculativeDecoding, which drafts from MTP heads, not an
      intermediate-layer exit. The repo is unusually well positioned: the
      LogitLens/TunedLens frozen-body splice idiom already implements "read
      logits at layer k", and examples/SpeculativeDecoding implements the
      accept/verify rule. v1:
      static exit layer + confidence threshold; follow-up: per-token
      adaptive exit. Report tokens/sec vs full-depth at matched output
      quality.
- [ ] Token healing follow-ups (v1 is landed: TNNetTokenHealingConstraint +
      PrepareTokenHealing + TGenerationConfig.TokenHealing):
      (a) PrepareTokenHealing is TStringListInt-only — a TNeuralHFTokenizer
      byte-level-BPE variant needs a vocab prefix-scan helper there;
      (b) guidance-style multi-token rollback (back up over more than the
      single last prompt token when the boundary artifact spans merges).
- [ ] Streaming corpus loader with shuffle buffer: the landed packing
      pipeline materializes the whole token stream in RAM (neuraldatasets
      builds one concatenated Stream array). Read large text/token files
      in chunks through a fixed-size shuffle buffer (the datasets-library
      streaming pattern) feeding the existing packer windows — the only
      way to pretrain on corpora bigger than RAM. Assert: same model
      quality on a small corpus vs the in-memory path at matched
      examples-seen, and bounded RSS on a corpus larger than the buffer.
- [ ] MinHash near-duplicate corpus dedup tool: the C4/Pile hygiene step —
      shingle each document, MinHash signatures, LSH banding to find
      near-duplicate clusters, keep one representative. Small standalone
      unit (or scripts/ tool) pairing with the streaming-corpus-loader
      task above; report duplicate-cluster stats. Test: planted
      near-duplicates (one-word edits) are found, distinct documents are
      not merged.
- [ ] ColBERT late-interaction retrieval follow-ups (core import +
      BuildColBERTFromSafeTensors[Ex] + MaxSim scorer + ColBERTRetrievalReport
      + parity fixture + examples/ColBERTSearch all LANDED a3, commit 442af2c):
      (a) attention-padding-mask support so a document shorter than the net's
      SeqLen is encoded exactly — today real tokens attend to the [PAD] rows
      (the same approximation examples/SemanticSearch documents); this also
      unblocks faithful batch-encoding of mixed-length docs in one net; (b) a
      library-side end-to-end "encode corpus -> cache doc matrices -> score
      query" helper in neuralpretrained.pas (today only the examples/ColBERTSearch
      demo wires this; ColBERTEmbedTokens + ColBERTMaxSimScore exist but the
      caching loop lives in the example); (c) pQuantizeInt8 for the ColBERT path
      — the projection head is always f32 while the BERT backbone already
      supports int8.

## Layer follow-ups that fix real limitations

(The sub-quadratic / chunked-forward family below is one coherent systems effort:
every recurrence currently trains as a strict per-token left-to-right scan.)

- [ ] TNNetDeltaNet chunked/parallel forward (the paper's WY-matrix
      reformulation of Yang et al. 2024, arXiv:2406.06484) so training is
      sub-quadratic instead of the current strict per-token left-to-right scan in
      TNNetDeltaNet; gate it behind an exact-vs-chunked equivalence assert.
- [ ] TNNetWKV chunked/parallel forward (RWKV-5/6 style) so training is not a
      strict per-token left-to-right scan; gate behind an exact-vs-chunked
      equivalence assert (mirrors the open TNNetDeltaNet chunked-forward task).
- [ ] TNNetGatedLinearAttention chunked/parallel hardware-efficient forward (the
      paper's main systems contribution; v1 ships the exact per-token scan only) —
      gate behind an exact-vs-chunked equivalence assert (mirrors open
      DeltaNet/WKV chunked tasks).
- [ ] TNNetMinGRU / TNNetMinLSTM follow-up (landed 2026-06-07, commit 69f8d53):
      Parallel-prefix-scan forward (the paper's main systems win) so training is
      not a strict per-token left-to-right loop; gate behind an exact-vs-parallel
      equivalence assert (mirrors the open DeltaNet/WKV chunked-forward tasks).
- [ ] TNNetLRU parallel/associative-scan forward (LTI recurrence → parallelizable)
      gated behind an exact-vs-scan equivalence assert.
- [ ] TNNetRetention chunkwise-recurrent hybrid form (a throughput optimisation
      skipped in v1 — the parallel and naive-recurrent forms both landed).
- [ ] Mini-batch / chunked Test-Time Training — follow-up to the landed
      TNNetTestTimeTraining (both TTT-Linear and TTT-MLP arms, with exact
      second-order BPTT through the inner update, shipped with tests + the
      examples/TestTimeTraining parity-binding recall demo). Apply ONE inner
      gradient-descent step per CHUNK of b tokens (mini-batch the inner SGD)
      instead of per token, for the same sub-quadratic-training motive as the
      open DeltaNet/WKV chunked-forward tasks. Optional second follow-up: a
      learnable PER-CHANNEL inner LR eta (currently a single learnable per-layer
      scalar via softplus(eta_raw)).
- [ ] `TNNetTitansMemory` follow-up — a **gated-DeltaNet-style chunked parallel
      scan** forward for `TNNetTitansMemory`, replacing the sequential O(SeqLen)
      inner-gradient scan with a chunked associative/parallel recurrence (the
      hardware-efficient training path Titans/Gated-DeltaNet use); must keep the
      exact second-order BPTT semantics and pass the existing gradient checks.
- [ ] TNNetImplicitLongConv / AddHyenaOperator FFT-based O(L log L)
      forward/backward path as an opt-in fast mode for long sequences — the
      current forward is the direct O(L^2) causal time-domain sum, fine for small
      sizes but quadratic in SeqLen. Gate it so the exact time-domain path stays
      the default and assert FFT-vs-direct equivalence to <1e-5.
- [ ] TNNetHyperLinear CHUNKED weight generation so the main layer can be larger
      than the generator's output width (generate W in tiles) — the landed layer
      generates the whole Din*Dout matrix in one shot, which caps main-layer
      size; document the memory/param trade-off.
- [ ] TNNet.AddDeepEquilibriumBlock follow-up: (a) the EXACT
      implicit-function-theorem gradient (inverse-Jacobian solve via a second
      fixed-point iteration) vs the phantom approximation; (b) spectral /
      contraction constraints so convergence is guaranteed at arbitrary init
      (v1 uses damped Picard + output bounding, not guaranteed).
- [ ] TNNetReversibleBlock follow-up: the MEMORY-SAVING recompute path (the
      actual point of RevNet — discard activations in forward, RECONSTRUCT them in
      backward via the analytic inverse instead of storing them). The landed
      builder demonstrates the inverse FORMULA and trains via ordinary stored-
      activation backprop; the O(1)-activation-memory training mode is still open
      and needs a custom backward that recomputes x1,x2 from y1,y2. Pairs with the
      open "Gradient checkpointing" infrastructure task.
- [ ] TNNetReversibleBlock follow-up: stack N reversible blocks into a deep net
      and show constant activation memory vs a plain residual stack of equal depth
      (the headline RevNet scaling claim) — depends on the recompute path above.
- [ ] TNNetSpectralNormConv follow-up: the landed wrapper normalizes by sigma_1
      of the FLATTENED kernel matrix (out_channels x in*kx*ky), a single scalar
      that BOUNDS but does not equal the true conv-OPERATOR spectral norm. Add a
      true-operator variant (Sedghi et al. 2019 FFT-based conv spectral norm, or
      a per-output-channel sigma) plus a small conv-SN bake-off / example vs the
      flattened-matrix version.
- [ ] `TNNet.AddTitansMemory` builder — a **MAC residual builder** wrapping
      `TNNetTitansMemory` (token-shift + per-token k/v/q projections + the neural
      memory leaf + residual/out-projection), the drop-in Memory-as-Context block,
      mirroring `AddGatedLinearAttention` / `AddRWKVTimeMix`.
- [ ] SDPA-level KV-cache aliasing for GQA decode (follow-up to the GQA
      verification batch, commit c1f8c8a): `AddMultiHeadGroupedQueryAttention`'s
      PARAMETER saving is real, but at decode time `TNNetStreamingDecoder` keeps
      one KV cache per SDPA layer, so the QueryHeads/KVHeads heads in a group
      each cache an identical K/V copy. Let grouped SDPA layers share a single
      KVHeads-sized cache (cache aliasing keyed by the shared K/V projection
      layers) so the GQA memory win materializes at inference; assert streamed
      output stays bit-identical to the unaliased path.
- [ ] Segment-mask MHA-builder wiring follow-up (the SDPA-layer + packer half
      landed above): thread an optional segment-id source through the
      multi-head attention BUILDERS (AddMultiHeadSelfAttention and friends) so
      packed-window training masks cross-document attention end-to-end, not just
      at the bare TNNetScaledDotProductAttention layer. Each per-head SDPA in the
      builder takes the same shared `pSegmentSource`; assert a packed two-doc MHA
      stack matches the concatenation of independent per-document MHA runs (the
      builder-level analogue of TestSegmentMaskMatchesUnpackedBaseline). KV-cache
      incremental decode stays intentionally unmasked (single-stream = one doc).

## Tests / numerical-gradient audit

- [ ] Shared `LayerInputAndWeightGradientCheck(layer, inputShape)` helper
      in tests/TestNeuralNumerical.pas. Three-line tests instead of
      copy-pasted blocks. Should handle both input and weight central-
      difference checks with a `Tolerance` parameter (default 1e-2), and
      promote DeMaxPoolFamilyGradientCheck's Double-precision SSE
      accumulator into it (sum the SSE in Double; eps and tolerance stay
      TNeuralFloat) so it can be opted into per-test. DATA POINT:
      TNNetAdaptiveMaxPool's gradient check hit the same float32
      subtractive-cancellation issue (a single cell carrying the whole
      window error, num=1.2588 vs ana=1.2709) and had to be loosened to
      tol 0.02 with an in-code comment — verified NOT a layer bug
      (double-precision central difference matches analytic exactly); a
      strong candidate to convert once this helper lands.
- [ ] Property-based gradient harness v0: randomize input shape (keeping
      layer type fixed) for the 6 most recently landed layers. Catches
      shape-edge bugs hand-written tests miss.
- [ ] Continue upsampling/deconvolution audit: TNNetDeconvolution input
      AND weight gradients (Upsample / DeMaxPool / DeAvgPool already
      covered). Likely benefits from the Double-precision SSE accumulator
      helper above.
- [ ] Recurrent-style layer audit: TNNetEmbedding's weight-gradient path
      (sparse-update pattern — easy place for a silent broadcast/reduction
      bug), TNNetTokenAndPositionalEmbedding, etc.
- [ ] Layer-registry round-trip audit — for every concrete TNNet* in the
      LoadFromString/CreateLayer dispatch table, instantiate with defaults,
      save, load, save again, assert bit-for-bit string equality. Highest-
      leverage single test for the "added a layer but forgot to register
      it" bug.
- [ ] Shape-inference smoke test — instantiate every concrete layer at a
      small canonical input shape, assert declared output shape matches
      actual.
- [ ] TestExtensions check: every layer that declares `FStruct[k]`
      constructor parameters should be tested for LoadFromString round-trip
      with NON-default values.
- [ ] Find-or-falsify pass: scan neuralnetwork.pas for any Backpropagate
      override whose body is just `inherited;` plus a tiny tweak — flag
      candidates for gradient-check coverage. The exact-softmax-Jacobian
      story teaches us how silent the diagonal-only bug class can be.
- [ ] Audit any remaining TNNet* layers that compute a softmax-like
      normalization (search for "Exp(" near a normalization loop) to
      confirm none still ship the diagonal-only approximation.
- [ ] Numerical-gradient stress test for TNNetSoftMax / TNNetPointwiseSoftMax
      across SeqLen / Depth / SizeX combinations.
- [ ] Backward audit for TNNetPointwiseNorm — its backward is the scalar-
      only `Mul(1/n)` approximation; TNNetL2Normalize now implements the
      exact Jacobian. Either replace or add a deprecation comment.
- [ ] Random-architecture forward/backward fuzz — generate ~50 random
      stacks, seed-controlled, assert no NaN/Inf in forward, backward,
      or parameter gradients.
- [ ] Cross-layer composition gradient test: build a 3-layer stack
      (LayerNorm → SwiGLU → Dense) and run a single end-to-end central-
      difference check on the input.
- [ ] Activation derivative-cache invariants sweep: for each cache-using
      activation, run `Compute(A); Compute(B);` and assert FOutputErrorDeriv
      matches a fresh recomputation against B. Catches the Sigmoid-class
      stale-cache bug pattern.
- [ ] Activation golden-values regression test — for every registered
      activation, evaluate forward/backward on a pinned input at fixed
      seed and assert against pinned outputs within 1e-5.
- [ ] Per-activation derivative-sign sanity test — for each strictly
      monotone activation, assert FOutputErrorDeriv has the expected sign
      on a grid of inputs.
- [ ] Saturation-safety tests for TNNetTanhExp / TNNetSmish at ±extreme
      inputs, mirroring the HardTanh/SoftCapping pattern.
- [ ] TNNetDigital forward-equality test — pin threshold and output for
      three inputs straddling it (non-differentiable, so forward-only).
- [ ] TNNetMaxPoolWithPosition correctness check — the auxiliary "position
      channels" should round-trip through TNNetDeMaxPool to exactly
      reconstruct the upsample pattern.
- [ ] TNNetAddPositionalEmbedding scale-factor backward check on
      rectangular (X≠Y) shapes (square inputs can hide off-by-one bugs).
- [ ] Gradient-flow regression test — train a 12-layer ReLU MLP one epoch
      with and without a single TNNetLayerNorm/RMSNorm at the midpoint;
      assert per-layer gradient norms with the norm layer are uniformly
      bounded above the no-norm case.
- [ ] TNNetDotProducts numerical-gradient test — standalone class still
      ships, weight-gradient path looks like the kind of place a silent
      bug could live.
- [ ] TNNetLocalConnect / TNNetDeLocalConnect input + weight gradient tests.
- [ ] Kink-region test parametric helper: with Clamp / HardShrink /
      SoftShrink / Threshold / ShiftedReLU / HardTanh all in tree, the
      "no-central-difference, hand-picked kink convention" pattern
      repeats. Capture as `AssertKinkDerivative(layer, x_kink, expected_dydx)`.
- [ ] TNNetHardShrink / TNNetSoftShrink kink-region tests at hand-picked
      inputs (no central differences).
- [ ] FP-exception robustness for TNNetSoftSign / TNNetESwish at truly extreme
      inputs (surfaced 2026-05-31 while writing the saturation tests above):
      both raise a HARDWARE FP exception rather than returning a finite value
      at far-extreme magnitudes — TNNetSoftSign's closed-form derivative
      `1/(1+|x|)^2` overflows float32 around |x|~1e30 (EInvalidOp), and
      TNNetESwish's `Exp(-beta*x)` overflows the RTL `Exp` around beta*x~-570
      (EOverflow). The landed tests stay inside the safe-but-saturating band
      (SoftSign ±1e6, ESwish beta*x up to ±625) and document the limit. Decide
      a policy: either clamp the offending intermediate (saturate the derivative
      to 0 / the sigmoid to its asymptote) so the layers stay finite at any
      input like HardTanh/SoftCapping do, or document the input-range contract.
      Then extend the tests to the far-extreme range under the chosen policy.
- [ ] LiSHT / BentIdentity gradient-magnitude sanity at large |x| — both
      grow unboundedly, finite-difference eps must scale with input
      magnitude.
- [ ] Shape-edge test for TNNetTokenShift: assert SetPrevLayer raises the
      documented error when SizeY > 1.
- [ ] Two-layer TokenShift composition test (catches subtle double-pass
      bugs in the t-1 / t+1 input-gradient scatter).
- [ ] TNNetStraightThroughEstimator `step ≤ 0` guard test.
- [ ] Audit TNNetSigmoid and TNNetHardSigmoid for negative-x / positive-x
      symmetric-stability (same question as SoftPlus).
- [ ] Add a "FP32 SSE accumulator warning" comment near LayerInputGradientCheck
      pointing future audits at the DeMaxPool case and the Double-precision
      workaround.
- [ ] TNNetPointwiseSoftMax: now that the exact Jacobian lives in
      Backpropagate, opt cross-entropy training paths into the cheap
      (y - target) shortcut explicitly, and add a regression test that
      checks the shortcut and the full-Jacobian path agree to 1e-5.
- [ ] TNNetSoftmaxTemperature refactor attempt: extract a shared softmax-
      Jacobian helper so SoftMax / PointwiseSoftMax / SoftmaxTemperature
      reduce to one Backpropagate body parameterised by axis +
      inv-temperature. Pure refactor, gradient tests pin behavior.
- [ ] Cross-entropy regression-style check: confirm classification
      examples (SimpleImage CIFAR) converge to the same loss curve they
      did before the TNNetSoftMax.Backpropagate exact-Jacobian change.
- [ ] Re-validate examples that use TNNetDeMaxPool / TNNetDeAvgPool after
      the gradient fix: DenseNet helper, VisualGAN, SuperResolution. The
      fix increases backward magnitude by `PoolSize` (=2 in practice), so
      existing learning rates may be off by 2x.
- [ ] Loss-layer gradient-check helper — parameterized helper that takes
      (LossLayer, BatchSize, Shape) and runs a single central-difference
      check.
- [ ] Backward-pass sign-correlation test — for every layer that overrides
      Backpropagate, perturb input by ±ε, assert gradient direction agrees
      with loss-difference direction >90% of the time across a small grid.
- [ ] Coverage matrix at the top of TestNeuralNumerical.pas: per-class
      `[grad] [serialize]` block, written by a small script.
