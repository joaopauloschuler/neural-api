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

## Infrastructure / dev experience

- [ ] Gradient checkpointing for training deeper nets in less memory
- [ ] ONNX import
- [ ] Gemma 4 import
- [ ] Qwen 3.5 import
- [ ] Import NFC tokenizer
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
      int8 weight-only storage instead of dequantize-then-requantize, and
      decode the common ggml 4-bit types (Q4_0/Q4_K/Q6_K) so real-world
      quantized checkpoints import.
- [ ] Quantized inference follow-up: upstream fix for TVolume.GetMaxAbs
      (seeds the running max with the SIGNED first element, so a negative
      max-magnitude element 0 is missed; csErrorOverflowBackpropProtection
      and friends consume it — audit callers before fixing).
- [ ] Tokenizer follow-ups for neuralhftokenizer.pas: (a) Unigram model
      support (model.type "Unigram", Viterbi segmentation) -- needed only
      for tokenizers not yet converted to BPE format; (b) the raw
      SentencePiece .model protobuf path (parse the protobuf wire format
      or vendor a minimal decoder) for checkpoints that ship without a
      tokenizer.json; (c) exact full-Unicode \p{L}/\p{N} tables (current
      classifier covers Latin/Greek/Cyrillic/Armenian/Hebrew/Arabic/
      Devanagari/Kana/CJK/Hangul; exotic scripts fall into the
      punctuation class of the GPT-2 regex); (d) build a tokenizer from
      GGUF tokenizer.ggml.* metadata (tokens/merges/scores arrays are
      already readable via TNNetGGUFReader) so a single .gguf file is
      fully self-contained for generation without tokenizer.json.
- [ ] neuralhftokenizer.pas pre_tokenizer leftovers from the Split/Metaspace
      batch: (a) a STANDALONE ByteLevel pre_tokenizer with use_regex=false
      silently applies the GPT-2 regex anyway (the flag is only honored
      implicitly inside the Sequence[Split, ByteLevel] path, which bypasses
      ByteLevelPieces; parse use_regex and skip the regex split when false);
      (b) only the Qwen2 and Llama-3/cl100k Split pattern literals are
      recognized — the o200k (GPT-4o-family) and DeepSeek pattern strings
      raise EHFTokenizerError; add them to the verbatim-match table with a
      hand-written splitter variant — the DeepSeek family is now importable
      (BuildDeepSeekV2FromSafeTensors), so its pattern is the live gap; add
      o200k when a GPT-4o-family checkpoint matters. Test: per-pattern
      parity fixtures like tools/hf_pretok_fixture.py.
- [ ] Gradient accumulation in neuralfit.pas: accumulate deltas over N
      micro-batches before the optimizer step (large effective batch
      without the memory), scaling the loss/deltas by 1/N so results match
      a true big batch.
- [ ] EMA (exponential moving average) shadow weights in neuralfit.pas:
      maintain decay-averaged copies of all weights during training and
      allow swapping them in for eval/save; classic free eval-quality win.
- [ ] Parameter groups for the optimizer (PyTorch param_groups port):
      per-group learning-rate multipliers and weight-decay exclusion for
      norm/bias parameters (AdamW currently decays everything uniformly).
- [ ] safetensors writer F16/BF16 output: TNNetSafeTensorsWriter only
      emits F32; add encode-on-write halves (EncodeF16/EncodeBF16 mirroring
      the existing decoders) for smaller exported checkpoints.
- [ ] HF-names safetensors exporter: export an imported pico-GPT-2 back to
      HF tensor names (wte.weight, h.N.attn.c_attn.weight, ...), reload via
      BuildGPT2FromSafeTensors and compare logits; generalize per-importer
      name maps later. The generic writer landed; only the naming/transpose
      mapping (Pascal neuron-major vs HF [in,out] Conv1D) is missing.
- [ ] GGUF WRITER / export path (neural/neuralgguf.pas currently READS only:
      TNNetGGUFReader F32/F16/Q8_0 + BuildLlamaFromGGUF). Add a
      TNNetGGUFWriter / SaveNNetToGGUF that emits a llama.cpp-loadable .gguf
      for an imported-or-trained Llama-family decoder, so CAI models can run
      in the dominant local-inference ecosystem (llama.cpp / ollama / LM
      Studio) — the export mirror of the landed reader and the natural
      interop sibling of the safetensors writer and the separate ONNX-export
      task (different runtime, F16/Q8_0 quantized output is the headline GGUF
      can do that ONNX/safetensors do not). The genuinely new work: (a) the
      ggml binary container — magic/version, the typed key/value metadata
      block (general.architecture="llama", the llama.* hyperparameters
      block_count/embedding_length/attention.head_count[_kv]/rope.* /
      context_length, plus the tokenizer.ggml.* model/tokens/merges/scores/
      token-type arrays so a single .gguf is self-contained, reusing the
      vocab already carried by the importers), and (b) the tensor section
      with ggml's REVERSED dimension order and 32-byte alignment, writing
      back the HF-canonical names llama.cpp expects (token_embd, blk.N.attn_q/
      k/v/output, blk.N.ffn_gate/up/down, *_norm, output_norm, output) and
      UNDOING the importer's rotate_half q/k permutation + SwiGLU up|gate
      split so a llama.cpp round-trip reproduces logits. v1: F32 + F16 +
      Q8_0 output (Q8_0 dovetails with the landed int8 weight-only storage —
      write its blocks straight out instead of re-quantizing). Deliverables:
      SaveNNetToGGUF[Ex], a tools/verify_gguf_writer.py cross-check that
      loads the emitted file with the gguf python package (and, when present,
      llama-cpp-python) and asserts dimensions/metadata and next-token logits
      vs the Pascal model within tolerance, and a round-trip test that writes
      a pico-Llama then re-imports it via the landed BuildLlamaFromGGUF and
      compares logits bit-for-bit. Scope v1 to the Llama family (the
      best-covered importer path); document other architectures as
      out-of-scope, same as the ONNX-export task.
- [ ] Stochastic Weight Averaging (torch.optim.swa_utils port): equal-weight
      running average of checkpoints over the schedule tail + a constant or
      cyclic SWA learning rate phase; swap averaged weights in for eval/save.
      Distinct from the EMA task above (running decay average) but should
      share the shadow-weights machinery — consider landing both on one
      shadow-copy mechanism.
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
- [ ] NEFTune noisy embedding fine-tuning: uniform noise scaled by
      alpha/sqrt(L*d) added to embedding-layer outputs during TRAINING only
      (off at eval); famously a ~5-line instruction-tuning quality win.
      Trivially testable: assert eval forward is noise-free and train
      forward differs.
- [ ] CutMix training augmentation (torchvision transforms-v2 staple;
      Mixup itself is landed: CreateMixedVolumePairList in neuralvolume +
      examples/Mixup): patch a random rectangle from a second sample into
      the input and mix the targets by area fraction (Beta-distributed
      lambda). The CIFAR image-classification examples give an instant
      bake-off harness.
- [ ] Contrastive search decoding (transformers penalty_alpha): degeneration
      penalty re-ranking each candidate token by max cosine similarity
      between its hidden state and all previous tokens' hidden states;
      needs hidden-state capture during decode — a different beast from the
      landed sampling/logits-processor chain and strong for greedy-quality
      open-ended text.
- [ ] Diverse beam search (Hamming-diversity groups) + constrained beam
      search (force_words_ids: force given phrases to APPEAR anywhere in the
      output — stronger than the existing TNNetTokenConstraint prefix/mask
      machinery) in neural/neuraldecode.pas as DecodeBeamSearch variants.
- [ ] Batched generation with left-padding in neural/neuraldecode.pas:
      generate for N prompts in one forward pass per step (today's decode
      paths look single-sample). Makes evaluation sweeps cheap and is a
      prerequisite for an efficient speculative-decoding verify step.
- [ ] Chat templates v2 (v1 is landed in neural/neuralchat.pas —
      ApplyChatTemplate with seven hardcoded formats + DetectChatFormat
      fingerprinting + EncodeChat): a mini-Jinja subset interpreter for unrecognized
      chat_template strings (must pass ground truth for all bundled
      templates and raise cleanly on unsupported constructs); more
      formats (DeepSeek, Phi-4-mini's tool-aware ChatML variant, Qwen's
      default-system injection); read the separate chat_template.jinja
      file newer transformers exports alongside tokenizer_config.json;
      continue_final_message / return_assistant_tokens_mask equivalents.
- [ ] Magnitude pruning (torch.nn.utils.prune port): PERSISTENT global or
      per-layer magnitude masks applied during training/inference — the
      diagnostics half is landed (TNNet.MagnitudePruningReport +
      examples/MagnitudePruning prune-and-restore sweep); still open are
      masks that stay applied (instead of restoring weights after the
      sweep) and a fine-tune-after-prune example showing accuracy recovery.
- [ ] Anomaly detection mode (autograd.set_detect_anomaly port): a debug
      flag that checks every layer's Output/OutputError for NaN/Inf during
      forward AND backward and names the FIRST offending layer + phase.
      The random-architecture NaN fuzz task elsewhere in this list finds
      such failures; this is the runtime tool that makes them diagnosable.
      Probably the highest value-to-effort diagnostic on this list.
- [ ] Per-layer profiler report (torch.profiler lite): TNNet.ProfileReport
      with forward/backward wall-time and parameter/activation memory per
      layer (introspection-report pattern). Directly serves the open
      chunked-forward throughput tasks by showing where time actually goes.
- [ ] FlashAttention-style tiled online-softmax SDPA forward: opt-in fast
      mode — not for GPU speed but for O(L*d) vs O(L^2) attention-score
      MEMORY on long sequences; gate behind an exact-vs-naive equivalence
      assert, same pattern as the chunked-forward recurrence family.
- [ ] rope_scaling follow-ups to the landed wiring (7e74fee): (a) DeepSeek-style YaRN `mscale` /
      `mscale_all_dim` overrides are rejected — needed for full
      DeepSeek-V2 checkpoints (the -Lite config carries them); (b) yarn
      `"truncate": false` configs are silently treated as truncate=true —
      honor the flag or reject loudly.
      [longrope (Phi-3) DONE: rsmLongRoPE mode wired into TNNetRotaryEmbedding
      + ReadRoPEScalingFromJSONObject parses longrope/su/yarn-with-long_factor;
      parity fixture tiny_phi3_longrope verified vs HF float64.]
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
- [ ] GRPO trainer (DeepSeekMath/R1-style group-relative policy
      optimization) in neural/neuraldpo.pas or a sibling unit: sample N
      completions per prompt, advantage = (reward - group mean)/group std,
      policy-gradient step with a KL penalty against the reference — no
      value network, so it is the one RL-from-feedback method that fits this
      framework. The DPO trainer already holds policy+reference and computes
      per-sequence logprobs, and sampled streamed generation already exists
      in neuraldecode. Cheap follow-ups on the same plumbing: ORPO / SimPO / KTO
      (loss-formula deltas on the landed DPO), and a Bradley-Terry pairwise
      reward-model trainer to feed GRPO real rewards.
- [ ] Seq2seq translation/summarization EXAMPLE on the landed beam:
      DecodeSeq2SeqBeamSearch + the BLEU/ROUGE metrics in
      neuralnlpmetrics.pas are both waiting on the Unigram/SentencePiece
      tokenizer for real Marian/T5 checkpoints - wire an examples/ entry
      (and an examples/README.md mention) once that lands.
- [ ] Masked-LM collator follow-up (TNNetMaskedLMCollator is landed in
      neuraldatasets.pas): whole-word masking (group subword pieces of one
      word and mask them together).
- [ ] Masked-LM collator follow-up: T5 span corruption with sentinel tokens
      (contiguous-span masking, sentinel id stream).
- [ ] Prompt tuning / P-tuning soft prompts (PEFT beyond the LoRA task
      above): K learnable virtual-token embeddings prepended to the
      embedding-layer output, base model frozen — K*d_model trainable
      params, the cheapest fine-tune of an imported checkpoint. Mostly
      composes existing pieces (a learnable bank + sequence-axis concat);
      decode side must skip the K virtual positions when detokenizing.
      Test: base weights bit-unchanged after a training step, soft-prompt
      gradient nonzero, eval forward deterministic.
- [ ] Token-classification head + entity-level metrics (transformers
      ForTokenClassification + seqeval port): per-token PointwiseConvLinear
      head over the sequence axis plus span-aware precision/recall/F1
      (BIO/IOB2 decoding, entity-level not token-level) in
      neuralnlpmetrics.pas. Needs tokenizer offset-mapping / word-id
      alignment (subword → word labels, return_offsets_mapping equivalent in
      neuraltokenizer.pas) — that alignment utility is reusable beyond NER.
      Test: pinned BIO sequences with known entity P/R/F1, including the
      classic boundary-error cases.
- [ ] QA span-extraction head (transformers ForQuestionAnswering port):
      two per-token logit heads (start/end) over the sequence axis +
      SQuAD-style postprocessing (top-k start×end pairs, end>=start, max
      answer length, n-best list). Pure composition over existing per-token
      projections; pairs with the offset-mapping utility from the
      token-classification task to map spans back to text. Test: pinned
      logits → pinned extracted span.
- [ ] Strided sliding-window perplexity in neural/neuralnlpmetrics.pas
      (the HF-docs-standard evaluation): for corpora longer than the model
      context, slide a window with stride < window and score only the
      non-overlapping tail tokens, so every token gets (bounded) left
      context instead of the chop-into-disjoint-windows underestimate the
      landed Perplexity() implies. Test: stride = window reproduces the
      disjoint baseline exactly; smaller stride gives <= NLL on a model
      with real long-range structure.
- [ ] Length-grouped batching + dynamic padding collator (transformers
      LengthGroupedSampler + DataCollatorWithPadding port) in neuralfit:
      sort/bucket variable-length text by length, batch neighbors, pad each
      batch only to its own max (not the global max) — a large real-world
      throughput win. Complements (distinct from) the per-sample-attention-
      mask and left-padded-generation tasks: this is the TRAINING data-side
      half. Test: identical loss trajectory vs naive padding at fixed seed
      modulo batch order, plus a padded-token-count reduction assert.
- [X] Classifier-free guidance for text generation (transformers
      UnbatchedClassifierFreeGuidanceLogitsProcessor port): run the model
      with and without the prompt (or with a negative prompt), combine
      l_uncond + g*(l_cond - l_uncond) before sampling. Two forward passes
      per step, no training; slots into the landed logits-processor chain
      as just another processor. Test: g=1 is bit-identical to normal
      decoding; g=0 ignores the prompt.
      LANDED: TNNetCFGProcessor in neuraldecode.pas owns a SECOND width-1
      TNNetStreamingDecoder for the unconditional/negative branch, steps it
      each ProcessRow, combines in log-prob space (the per-branch softmax
      constant cancels in the final softmax, so g=1 leaves the cond row
      untouched bit-for-bit and skips the extra forward). 3 tests:
      g=1==plain greedy, g=0 prompt-independent, arg validation.
- [ ] Best-of-N / self-consistency reranking utility in
      neural/neuraldecode.pas: sample N completions, rerank by
      length-normalized sequence logprob (LengthPenaltyDenominator already
      exists) or by an external scorer callback — the standard
      test-time-compute baseline, and the natural consumer of the
      Bradley-Terry reward model from the GRPO task. Self-consistency
      variant: majority-vote over extracted answers. Sampled generation is
      already landed, so this is mostly harness work; worth its own entry as
      the canonical harness.
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
- [ ] DoLa decoding (Decoding by Contrasting Layers, Chuang et al. 2023;
      transformers `generate(dola_layers=...)`): improve FACTUALITY (not
      speed — distinct from the early-exit task above and from contrastive
      search, which contrasts against context tokens) by reading logits at
      a premature layer via the same LogitLens "logits at layer k" splice,
      then scoring next tokens by log p_final - log p_premature over an
      adaptive head-candidate set (tokens above an alpha fraction of the
      final layer's max prob); pick the premature layer per step as the
      one with max Jensen-Shannon divergence from the final distribution
      over a small candidate bucket. Pure decode-time composition of
      landed primitives — no new weights, no training. v1 in
      neural/neuraldecode.pas with fixed candidate-layer bucket; test:
      on a toy LM with a planted shallow-layer bias, DoLa flips the
      biased completion while greedy full-depth decode does not, and
      alpha=0/empty-bucket degrades exactly to standard greedy.
- [ ] Grammar/regex-constrained decoding (GBNF-style): generalize the
      landed hand-written JSON state machine — TNNetTokenConstraint's
      Reset/MaskAllowed/Commit interface (plus the copy-on-fork support
      the JSON machine already has) is exactly the right plug. v1: a
      llama.cpp-GBNF-subset grammar compiled to a pushdown machine over
      CHARACTERS, with the existing char-by-char token-feasibility walk;
      alternatively (or additionally) regex -> DFA. Test: a small
      arithmetic-expression grammar accepts only valid strings across
      greedy/sampled decoding, and forked beams keep independent states.
- [ ] Token healing follow-ups (v1 is landed: TNNetTokenHealingConstraint +
      PrepareTokenHealing + TGenerationConfig.TokenHealing):
      (a) PrepareTokenHealing is TStringListInt-only — a TNeuralHFTokenizer
      byte-level-BPE variant needs a vocab prefix-scan helper there;
      (b) guidance-style multi-token rollback (back up over more than the
      single last prompt token when the boundary artifact spans merges).
- [ ] HellaSwag-style eval example on an imported checkpoint: a small
      example program that loads a real imported model (e.g. SmolLM2 /
      pythia via the safetensors importers), tokenizes a handful of
      multiple-choice items with TNeuralHFTokenizer and reports acc /
      acc_norm through EvaluateMultipleChoice (neuralnlpmetrics.pas) —
      the end-to-end "imported model scores X" demo the landed scoring
      API was built for. Follow-ups: batch candidates sharing
      a context prefix; optional last-window scoring for over-context
      sequences (v1 raises).
- [ ] Needle-in-a-haystack long-context eval harness: place a fact at
      varying depths in a synthetic long context, measure retrieval
      accuracy vs (depth, context length) as a small grid report. The
      landed RoPE-scaling wiring and the open KV-cache-eviction task both
      NEED this to demonstrate they work — neither has an eval. Works
      with the char-level/TinyStories-scale models the repo can actually
      run, not just imported LLMs.
- [ ] Streaming corpus loader with shuffle buffer: the landed packing
      pipeline materializes the whole token stream in RAM (neuraldatasets
      builds one concatenated Stream array). Read large text/token files
      in chunks through a fixed-size shuffle buffer (the datasets-library
      streaming pattern) feeding the existing packer windows — the only
      way to pretrain on corpora bigger than RAM. Assert: same model
      quality on a small corpus vs the in-memory path at matched
      examples-seen, and bounded RSS on a corpus larger than the buffer.
- [ ] BPE-dropout subword regularization (Provilkov et al. 2020): during
      TRAINING tokenization randomly skip each applicable merge with
      p~0.1 so the model sees alternative segmentations of the same text;
      well-known robustness/quality win, ~20 lines in the existing
      neuraltokenizer BPE merge loop, train-time only (eval tokenization
      unchanged). Test: p=0 is bit-identical to the deterministic
      tokenizer; p>0 at fixed seed yields a pinned alternative
      segmentation.
- [ ] MinHash near-duplicate corpus dedup tool: the C4/Pile hygiene step —
      shingle each document, MinHash signatures, LSH banding to find
      near-duplicate clusters, keep one representative. Small standalone
      unit (or scripts/ tool) pairing with the streaming-corpus-loader
      task above; report duplicate-cluster stats. Test: planted
      near-duplicates (one-word edits) are found, distinct documents are
      not merged.

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
- [ ] Per-sample / dynamic attention masks in TNNetScaledDotProductAttention
      (follow-up to TNNetSequencePacker, commit 52c5ca0): SDPA only supports
      the static causal flag + static sliding window, so packed training
      windows cannot mask attention across document boundaries (GPT-2/3-style
      cross-doc attention is what ships today). Add an optional per-sample
      block-diagonal/document-id mask input (or a segment-ids side channel) and
      wire `TNNetSequencePacker` to emit it; verify with a test that attention
      weights across a separator are exactly zero and gradients match an
      unpacked per-document baseline.

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
