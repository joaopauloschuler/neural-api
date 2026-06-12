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

## Infrastructure / dev experience

- [ ] Gradient checkpointing for training deeper nets in less memory
- [ ] ONNX import
- [ ] ONNX (or simpler JSON) export path — minimal viable: dump a
      forward-only graph for the currently-supported subset of layers,
      enough to run inference in onnxruntime. Doc which layers are
      out-of-scope for v1.
- [ ] Quantized inference — int8 (and/or FP16 storage) weight compression so
      real imported checkpoints fit on commodity RAM: TinyLlama-1.1B in FP32 is
      ~4.4GB of weights, beyond a 3GB-class machine even with MakeInferenceOnly,
      so this is what separates "imports Llama" (landed, abe7573) from "runs
      Llama". v1: per-output-channel symmetric int8 weight storage for the
      weight-heavy layers (PointwiseConvLinear / FullConnect / Conv — the
      layers the introspection report already flags as quantization
      candidates), dequantize-on-the-fly into the existing FP32 forward (no
      int8 matmul kernels needed yet); a `QuantizeWeights` /
      inference-only-load path wired into BuildGPT2/BuildLlamaFromSafeTensors;
      assert logit drift vs FP32 stays within a documented tolerance on the
      pico-Llama parity fixture. FP16 weight STORAGE (half the RAM, ~zero
      accuracy risk) is an acceptable stepping stone if int8 proves invasive.
      Follow-ups: int8 activation/matmul path, GPTQ/AWQ-style calibrated
      quantization, 4-bit.
- [ ] Tokenizer follow-ups for neuralhftokenizer.pas: (a) Unigram model
      support (model.type "Unigram", Viterbi segmentation) -- needed only
      for tokenizers not yet converted to BPE format; (b) the raw
      SentencePiece .model protobuf path (parse the protobuf wire format
      or vendor a minimal decoder) for checkpoints that ship without a
      tokenizer.json; (c) exact full-Unicode \p{L}/\p{N} tables (current
      classifier covers Latin/Greek/Cyrillic/Armenian/Hebrew/Arabic/
      Devanagari/Kana/CJK/Hangul; exotic scripts fall into the
      punctuation class of the GPT-2 regex).
- [X] `Split`-regex + `Metaspace` pre_tokenizer support in
      (DONE: AddPreTokenizer now accepts Metaspace — replacement /
      prepend_scheme always|first|never / legacy add_prefix_space / split,
      routed onto the metaspace BPE machinery with HF first-segment-only
      prepend semantics — and Split, matching the shipped pattern VERBATIM
      against the Qwen2 (\p{N}) and Llama-3/cl100k (\p{N}{1,3}) literals
      and dispatching to a hand-written SplitCl100kPieces ordered-
      alternation splitter (case-insensitive contractions, optional
      non-letter lead char, capped digit runs, punct+[\r\n]*, \s*[\r\n]+,
      \s+(?!\S) trailing-space rule); unknown patterns/behaviors still
      raise EHFTokenizerError. Metaspace decoder branch maps onto
      Replace(▁→' ')+strip-left. 3 new tiny fixtures + parity batteries
      (tools/hf_pretok_fixture.py, exact id+decode parity vs HF tokenizers
      0.22) + unknown-pattern rejection test in
      tests/TestNeuralHFTokenizer.pas.)
      neural/neuralhftokenizer.pas: AddPreTokenizer hard-rejects every type
      except Sequence/ByteLevel/BertPreTokenizer, so the tokenizer.json
      shipped with the ALREADY-IMPORTABLE Llama-family checkpoints fails to
      load — Qwen2/Qwen3 and Llama-3-style tokenizers use
      Sequence[Split(pattern=cl100k-style regex, behavior=isolated),
      ByteLevel(use_regex=false)], and Llama-2/Mistral-v0.1
      SentencePiece-BPE tokenizers use Metaspace (U+2581 prefix-space
      replacement). Net effect: BuildQwen2/BuildMistralFromSafeTensors load
      weights that nothing in the repo can tokenize for end-to-end. v1:
      (a) Metaspace — space→▁ replacement with prepend/add_prefix_space
      handling plus the matching Metaspace decoder branch; (b) Split —
      interpret the shipped pattern against the handful of regex idioms
      these tokenizers actually use rather than vendoring a full regex
      engine (SplitGPT2 already implements the contraction/\p{L}/\p{N}
      machinery; generalize for the cl100k deltas: case-insensitive
      contractions, \p{N}{1,3} digit-run capping, optional non-letter
      leading char), rejecting genuinely unknown patterns loudly as today.
      Test: per-family parity fixtures (tiny tokenizer.json + reference
      token ids from transformers AutoTokenizer covering numbers,
      punctuation, CJK, newlines, leading spaces), same pattern as the
      existing GPT-2/BERT tokenizer tests. Distinct from the
      Unigram/.model-protobuf follow-ups task above: these checkpoints DO
      ship a tokenizer.json — only the pre_tokenizer kind is unsupported.
- [ ] neuralhftokenizer.pas pre_tokenizer leftovers from the Split/Metaspace
      batch: (a) a STANDALONE ByteLevel pre_tokenizer with use_regex=false
      silently applies the GPT-2 regex anyway (the flag is only honored
      implicitly inside the Sequence[Split, ByteLevel] path, which bypasses
      ByteLevelPieces; parse use_regex and skip the regex split when false);
      (b) only the Qwen2 and Llama-3/cl100k Split pattern literals are
      recognized — the o200k (GPT-4o-family) and DeepSeek pattern strings
      raise EHFTokenizerError; add them to the verbatim-match table with a
      hand-written splitter variant when those checkpoint families become
      importable. Test: per-pattern parity fixtures like
      tools/hf_pretok_fixture.py.
- [ ] Logits-processor chain + generation config in neural/neuraldecode.pas
      (the remaining half of the transformers GenerationMixin port):
      top-k/top-p/min-p sampling, repetition/frequency/presence penalties
      (TNNetTokenHistoryPenalty) and stop sequences/strings are all landed —
      still missing are a temperature knob and a chainable logits-processor
      abstraction (generalize TNNetTokenConstraint from "mask allowed
      tokens" into processors that transform the logit vector in order) so
      penalties, temperature and the existing JSON/forced-sequence
      constraints compose in one pipeline. Add a small TGenerationConfig
      record bundling sampler + penalties + stopping criteria (EOS-id list,
      stop strings, max new tokens).
- [ ] LoRA follow-ups (the adapter itself is landed: TNNet.AddLoRAAdapter
      low-rank B·A bypass + examples/LoRAFineTune, commit 34511c0):
      (a) MergeLoRA — fold the trained B·A (scaled by alpha/r) into the
      frozen base layer's W for zero-overhead inference, asserting merged
      forward equals adapter forward; (b) load HF PEFT adapter safetensors
      files onto an imported base model (name-map lora_A/lora_B tensors
      onto the AddLoRAAdapter layers).
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
- [ ] safetensors WRITER (neural/neuralsafetensors.pas has only the
      reader): export named tensors so Pascal-trained models round-trip
      into PyTorch/transformers. Doubles as a cross-framework correctness
      check for the GPT-2/Llama importers (export → reload → compare).
- [ ] Knowledge distillation trainer (transformers DistillationTrainer /
      classic Hinton KD): temperature-softened KL between a frozen teacher's
      logits and the student's, blended with the hard-label loss
      (L = alpha*CE(hard) + (1-alpha)*T^2*KL(soft)). The DPO trainer already
      holds two TNNets simultaneously, so the two-model plumbing exists.
      Killer combo with the Llama/GPT-2 importers: distill an imported
      pretrained teacher into a small Pascal-trained student.
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
      sampling/logits-processor chain task above and strong for
      greedy-quality open-ended text.
- [ ] Diverse beam search (Hamming-diversity groups) + constrained beam
      search (force_words_ids: force given phrases to APPEAR anywhere in the
      output — stronger than the existing TNNetTokenConstraint prefix/mask
      machinery) in neural/neuraldecode.pas as DecodeBeamSearch variants.
- [ ] Batched generation with left-padding in neural/neuraldecode.pas:
      generate for N prompts in one forward pass per step (today's decode
      paths look single-sample). Makes evaluation sweeps cheap and is a
      prerequisite for an efficient speculative-decoding verify step.
- [ ] Chat templates (transformers apply_chat_template): follow-up of the
      landed tokenizer.json loader (neural/neuralhftokenizer.pas) — parse
      the chat template from
      tokenizer_config.json (or hardcode the common Llama/ChatML/Zephyr
      formats) so an imported instruct checkpoint can be prompted with a
      correctly formatted conversation end to end.
- [ ] resize_token_embeddings equivalent: grow/shrink the token embedding +
      LM-head vocab of an imported model (mean-init new rows, keep tied
      heads consistent, update FStruct vocab sizes). Needed the moment
      anyone adds special tokens to fine-tune on top of the Llama/GPT-2
      importers.
- [ ] GGUF reader (sibling of neural/neuralsafetensors.pas): the other
      de-facto checkpoint format, and it ships PRE-quantized weights —
      dovetails with the int8 quantized-inference task (read Q8_0 blocks
      directly instead of quantizing FP32 yourself).
- [ ] Magnitude pruning (torch.nn.utils.prune port): global or per-layer
      magnitude masks with a sparsity report (the introspection-report
      pattern covers the diagnostics half); optional fine-tune-after-prune
      example showing accuracy recovery.
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
- [ ] RoPE scaling config wiring for context extension: the RoPE layer
      already implements Position Interpolation, NTK-aware and YaRN
      (TNNetRoPEScalingMode, including YaRN's per-band interpolation +
      attention-temperature factor), but neural/neuralpretrained.pas still
      PARSES the rope_scaling config key and then rejects it ("long-context
      scaling is not wired here yet", ~line 172). Map the config (type +
      factor + YaRN params) onto the layer's scaling-mode constructor
      arguments so imported Llama-family checkpoints run past their trained
      context. Verify: HF parity fixture with a rope_scaling config
      (transformers applies the same remap), plus a sanity check that an
      unscaled config stays bit-identical to the landed path.
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
      quantized-inference task above covers WEIGHT storage; for long-context
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
- [ ] Seq2seq (encoder-decoder) generation harness in
      neural/neuraldecode.pas: every decode path today is decoder-only.
      TNNetCrossAttention and TNNetCrossWKV (asymmetric mode) already exist
      as layers — add a T5/BART-style loop that encodes the source ONCE,
      caches the encoder output, feeds it as the K|V side of the cross
      layers, and autoregresses the target with the usual greedy/beam/sample
      machinery. Unlocks translation/summarization examples that the landed
      BLEU/ROUGE metrics in neuralnlpmetrics.pas are waiting for.
- [ ] Masked-LM data collator (transformers DataCollatorForLanguageModeling
      port): BERT-style dynamic masking — pick 15% of tokens, replace 80%
      with [MASK] / 10% random / 10% unchanged, loss only on masked
      positions — plus whole-word masking and, as a stretch, T5 span
      corruption (sentinel tokens). Everything in the current NLP stack is
      causal-LM; one collator unit unlocks encoder pretraining with the
      existing AddTransformerEncoderBlock, no new layers. Test: masked
      fraction and 80/10/10 split within tolerance at fixed seed; loss
      ignores unmasked positions exactly.
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
- [ ] chrF metric in neural/neuralnlpmetrics.pas: character n-gram F-score
      (Popović 2015) — tokenizer-independent so it sidesteps the BLEU
      tokenization sensitivity, ~100 lines beside the landed BLEU/ROUGE.
      Optional chrF++ (adds word unigrams+bigrams). Test against pinned
      values from sacrebleu on a couple of sentence pairs.
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
- [ ] Classifier-free guidance for text generation (transformers
      UnbatchedClassifierFreeGuidanceLogitsProcessor port): run the model
      with and without the prompt (or with a negative prompt), combine
      l_uncond + g*(l_cond - l_uncond) before sampling. Two forward passes
      per step, no training; slots into the logits-processor chain task
      above as just another processor once that lands. Test: g=1 is
      bit-identical to normal decoding; g=0 ignores the prompt.
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
- [X] PyTorch pytorch_model.bin loader: a RESTRICTED unpickler for the
      torch.save zip format — the long tail of older/fine-tuned checkpoints
      never got converted to safetensors. State_dicts use a small pickle
      subset (a dozen opcodes + persistent-id storage references into the
      zip's per-tensor data files); implement exactly that whitelist and
      reject everything else by design, which makes the Pascal loader
      immune to the arbitrary-code-execution hazard that motivated
      safetensors. Expose the same named-tensor API as
      TNNetSafeTensorsReader so the GPT-2/Llama builders take either
      format. Test: torch.save a pico state_dict in Python, assert every
      tensor matches its safetensors twin bit-for-bit. Concrete consumers
      verified 2026-06-12: cerebras/Cerebras-GPT-* and the
      roneneldan/TinyStories-* checkpoints ship pytorch_model.bin ONLY
      (the Cerebras parity work converted via Python as a stopgap).
      DONE 2026-06-12: TNNetTorchBinReader in neural/neuraltorchbin.pas —
      own zip central-directory parser (incl. zip64, STORED entries read
      at absolute offsets) + restricted protocol-2 unpickler (whitelisted
      GLOBALs: collections OrderedDict, torch._utils
      _rebuild_tensor_v2/_rebuild_parameter, torch *Storage dtypes; any
      other GLOBAL/opcode/REDUCE hard-fails, e.g. the fixture's `posix
      system` payload). Subclasses TNNetSafeTensorsReader and fills the
      same tensor table, so EVERY importer takes .bin transparently via
      CreatePretrainedTensorReader extension dispatch (all 7 Build*
      families + BuildFromPretrained directory probe of
      pytorch_model.bin). Tests: bit-for-bit twin parity
      (tools/torch_bin_fixture.py f32/f16/bf16/i64 + 3-D + scalar),
      malicious-pickle rejection, GPT-2 logit parity through tiny_gpt2.bin.
- [ ] Sharded pytorch_model.bin checkpoints: support
      pytorch_model.bin.index.json (same weight_map shape as the landed
      safetensors index) by opening every referenced .bin shard behind the
      TNNetTorchBinReader API — mirrors TNNetSafeTensorsReader's
      OpenFromIndex, mostly plumbing since per-shard absolute offsets
      already work. Also out of scope from the landed v1, if ever needed:
      the pre-torch-1.6 non-zip legacy format, DEFLATE-compressed zip
      entries, and non-contiguous (stride-permuted) state_dict tensors —
      all currently rejected with descriptive ETorchBinError messages.
- [ ] T5/Flan-T5 (encoder-decoder) importer: the natural companion to the
      seq2seq generation harness task above. T5's relative-position bias
      buckets are landed (TNNetT5RelPosBiasAttention / avT5RelPosBias), and
      RMSNorm-style scale-only norm and gated FFN map onto landed blocks, so
      the work is the importer weight mapping (bias shared across layers,
      separate enc/dec). Flan-T5-small is 80M params — genuinely CPU-friendly and
      instruction-tuned, so it doubles as the first imported model the
      BLEU/ROUGE metrics can score out of the box. Same HF-parity fixture
      verification as GPT-2/Llama.
- [X] DistilBERT / RoBERTa ForSequenceClassification head deltas: the
      landed BuildBertForSequenceClassificationFromSafeTensors (4f0e2c1)
      explicitly rejects non-bfBert families because their classifier
      stacks differ — DistilBERT uses pre_classifier dense + ReLU on the
      [CLS] hidden (no pooler), RoBERTa uses classifier.dense + tanh +
      classifier.out_proj on the <s> token (also no pooler). Both trunks
      are already landed as deltas on the BERT encoder builder, so each is
      a small head-mapping case on the same pSeqClsHead option. Same
      fixture/parity-test pattern as tiny_bert_seqcls (id2label included).
      DONE: pSeqClsHead now maps the family-specific head (bert keeps the
      forced pooler; distilbert = pre_classifier+ReLU+classifier; roberta
      = classifier.dense+tanh+classifier.out_proj, no pooler in either);
      BuildFromPretrained dispatches DistilBert/Roberta-
      ForSequenceClassification architectures. Fixtures
      tiny_{distilbert,roberta}_seqcls (tools/*_seqcls_tiny_fixture.py,
      self-checked oracle paths: ReLU clips / tanh bends / offset
      positions) + parity tests TestDistilBertSeqClsLogitParity,
      TestRobertaSeqClsLogitParity + both dispatch routes.
- [ ] Streaming/lazy tensor materialization with load-time quantization:
      the import path materializes full FP32 tensor buffers before copying
      into layers, so PEAK import memory, not steady-state, can be the gate
      on commodity RAM (the TinyLlama ~4.4GB case the quantized-inference
      task documents). Read one tensor at a time from the (seekable)
      safetensors stream straight into the destination layer — or straight
      into int8 storage once the quantized-inference task lands — keeping
      only one tensor-sized scratch buffer. Assert peak RSS during import
      stays within tensor-size + model-size on the parity fixture.
- [ ] NumPy .npy/.npz reader + writer: the universal interchange escape
      hatch — ANY framework can np.savez(**state_dict), and the format is
      trivial (magic + header dict + raw bytes; npz is a zip of npy).
      Reader gives a generic named-tensor import path for frameworks with
      no safetensors export; WRITER doubles as fixture tooling for parity
      tests (dump Pascal tensors, compare in Python) and is arguably an
      easier first Pascal→Python round-trip than the listed safetensors
      writer. Support F32/F64/F16 + int dtypes, C-order only, reject
      Fortran-order/pickled-object arrays explicitly.
- [ ] Gemma 1 - GeGLU FFN wiring: Gemma's MLP is gated GELU, not SwiGLU.
      TNNetGELU already implements the exact tanh approximation Gemma
      specifies (gelu_pytorch_tanh), so this is letting the importer's
      gated-FFN construction take the gate activation as a parameter
      (GELU vs SiLU) rather than any new layer. Test: gated-FFN forward
      parity against transformers' GemmaMLP on a pinned tensor.
- [ ] Gemma 1 - BuildGemmaFromSafeTensors importer (load-time weight
      folding on the landed Llama path; depends on the GeGLU task above —
      the decoupled head_dim config override Gemma-7B needs is already
      landed on the Llama config reader via the Qwen3 work): (a) embedding output scaled by sqrt(d_model) — fold
      sqrt(d) into the embedding ROWS at load since TNNetEmbedding's
      ScaleEmbedding is init-only, and scale ONLY the embedding copy, not
      the tied LM-head copy (Gemma always ties); (b) zero-centered RMSNorm
      — Gemma applies (1+w)*xhat, so store 1+w into TNNetTokenRMSNorm
      weights at load, zero layer changes; (c) MQA on 2B via the landed
      GQA builder at KVHeads=1. Verify with the HF-parity fixture tooling
      (sliced pico-Gemma) against GemmaForCausalLM, tied-head case.
- [ ] Gemma 2 - SDPA attention-logit soft-cap hook: the one real LAYER
      change in the Gemma track — Gemma-2 applies cap*tanh(scores/cap)
      (cap 50) to attention logits PRE-softmax, but scores live inside
      TNNetScaledDotProductAttention, so the standalone TNNetSoftCapping
      cannot reach them. Add an optional score-softcap parameter to SDPA
      (same opt-in pattern as the causal/window flags; default off =
      bit-identical), with the tanh' factor in backward. Gradient-check
      with the cap on, and assert cap=0/off matches the landed path
      exactly.
- [ ] Gemma 2 - sandwich-norm block builder: pre AND post RMSNorm around
      both the attention and FFN sublayers (4 norms per block, vs the
      landed pre-norm-only residual helpers). Pure composition — a builder
      variant beside AddPreNormResidual/AddRMSNormResidual; the post-norm
      sits INSIDE the residual branch (normalize sublayer output before
      the add). Reusable beyond Gemma (several recent models adopt
      sandwich norms).
- [ ] Gemma 2 - BuildGemma2FromSafeTensors importer (depends on the Gemma-1
      importer + the two Gemma-2 tasks above): alternating local(4096)/
      global attention via the landed sliding-window SDPA (same alternating
      pattern as the landed GPT-Neo importer); query_pre_attn_scalar (e.g. 224 on 27B)
      folded into W_q at load (same trick as GPT-Neo's unscaled attention);
      final-logit soft-capping as a plain TNNetSoftCapping.Create(30)
      before the softmax head — zero new code. Note the 256k vocab makes
      embedding+head the FP32 memory hot spot (quantized-inference task
      dependency for the larger sizes). Parity fixture vs
      Gemma2ForCausalLM.
- [ ] Gemma 3 - per-head QK-norm composition: Gemma-3 replaces soft-capping
      with RMSNorm applied per-head to q and k before attention (distinct
      from the landed avQKNorm / TNNetQKNormAttention, which L2-normalises
      q/k with a learnable temperature — Gemma needs a learnable-scale
      RMSNorm). Likely pure composition: the multi-head builders split q/k/v into explicit
      per-head layers before SDPA, so insert TNNetTokenRMSNorm on the q and
      k branches — verify the insertion point sits AFTER RoPE the way
      transformers' Gemma3 does (norm placement relative to RoPE changed
      across implementations; pin it against the HF reference, do not
      guess). Gradient check + a wired-into-MHA-builder flag.
      NOTE 2026-06-12: the reusable composition landed with the Qwen3
      importer — per-head TNNetTokenRMSNorm copies on the q/k slices with a
      shared rotate_half-permuted [head_dim] gain
      (LoadLlamaHeadRMSNormWeights in neural/neuralpretrained.pas), wired
      BEFORE RoPE there (the verified Qwen3 ordering). Gemma-3 still needs
      its own placement verification vs HF Gemma3 (norm-vs-RoPE order
      differs across implementations) + the MHA-builder flag.
- [ ] Gemma 3 - BuildGemma3FromSafeTensors importer, TEXT-ONLY (depends on
      the QK-norm task + the Gemma-2 importer): 5:1 local:global layer
      ratio (config wiring over the same alternating machinery) and
      PER-LAYER-TYPE RoPE theta — 10k for local layers, 1M for global
      layers, so rope_theta becomes per-layer instead of one global config
      value (plumbing exists from the Llama config reader, needs a
      per-layer override). The 4B+ multimodal vision tower is explicitly
      OUT OF SCOPE (separate project). Parity fixture vs
      Gemma3ForCausalLM on a sliced text-only checkpoint.
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
- [ ] RWKV-4 checkpoint importer (RWKV-4-Pile-169M/430M) — the first
      NON-TRANSFORMER importer, and the most distinctive: recurrent
      inference with CONSTANT memory and no KV cache, exactly the right
      architecture for CPU deployment. TNNetWKV (RWKV-4 time-mixing) and
      TNNetTokenShift are landed; the gaps are the channel-mix block
      (token-shift + squared-ReLU gating — existing primitives, builder
      work) and the weight mapping (per-layer time/channel mix params;
      map the checkpoint's time_decay/time_first vectors onto TNNetWKV's
      softplus-decay w and bonus u parameterization — check the exp/log
      convention against the reference carefully). Verify logit parity vs
      HF's RwkvForCausalLM on a sliced fixture; a decode-side demo
      showing flat memory vs a transformer of equal size is the headline.
- [ ] Mamba checkpoint importer (state-spaces/mamba-130m-hf) — the
      selective-SSM sibling of the RWKV-4 importer task above and the
      second non-transformer family; like RWKV it decodes with CONSTANT
      memory (no KV cache). TNNetSelectiveSSM is landed but is the N=1
      special case: it keeps ONE state scalar per channel and emits
      per-channel b_t/c_t, while the HF MambaMixer carries d_state=16
      states per channel with B,C in R^d_state shared across channels
      (x_proj -> [delta_rank|2*d_state] split, low-rank dt_proj with the
      inv-softplus dt bias init, per-(channel,state) A_log, D skip).
      Work: (1) generalize TNNetSelectiveSSM with a DState argument
      (FStruct, default 1 = today's behavior bit-for-bit, gradient check
      for DState>1); (2) AddMambaBlock builder — in_proj to 2*d_inner
      (x|z gate split via TNNetSplitChannels), depthwise
      TNNetCausalConv1D(k=4)+bias, SiLU, selective scan, SiLU(z) gate
      mul, out_proj, wrapped in AddRMSNormResidual; (3) the
      BuildMambaFromSafeTensors weight mapping in
      neural/neuralpretrained.pas (config.json: d_model/n_layer/
      time_step_rank; fold conv1d/x_proj/dt_proj layouts; tied LM head).
      Verify logit parity vs HF MambaForCausalLM on a sliced pico fixture
      (same make_pico script recipe as GPT-2/Llama); headline demo:
      tokens/sec flat in context length where the transformer slows.
- [ ] ModernBERT importer (answer.ai, 139M) — the encoder worth targeting
      BEYOND vanilla BERT now that BuildBertFromSafeTensors has landed:
      RoPE instead of learned positions, GeGLU, alternating local/global
      attention — every ingredient is already landed or tasked (Gemma /
      GPT-Neo machinery). Best current retrieval/classification encoder
      at CPU-friendly size; feeds the same token-classification / QA /
      sentence-embedding heads as the landed BERT importer. Parity fixture vs
      ModernBertModel hidden states.
- [ ] Whisper-tiny importer (openai/whisper-tiny, 39M) — the FIRST speech
      model and the first real ENCODER-DECODER checkpoint import (the T5
      task above is text-to-text; this one exercises cross-attention from
      a non-text modality). Every hard ingredient is already landed:
      TNNetCrossAttention asymmetric mode (QSeqLen != KVSeqLen) for the
      decoder reading the 1500-frame encoder output, the GPT-2 byte-level
      BPE tokenizer.json path for Whisper's vocabulary, and FFT machinery
      (FourierMixFFT) for the frontend. Work: (1) log-mel spectrogram
      frontend — 80 mel bins, 400-sample STFT window / 160 hop, the fixed
      HF mel filterbank — as a preprocessing helper (plus a ~40-line
      16-bit PCM WAV reader) in neuraldatasets.pas or a small
      neuralaudio.pas; (2) encoder builder: Conv1D(k=3,s=1)+GELU,
      Conv1D(k=3,s=2)+GELU, FIXED sinusoidal positions, pre-norm
      transformer blocks; (3) decoder builder: learned positions, causal
      self-attention + cross-attention per block, tied LM head, and the
      <|startoftranscript|><|en|><|transcribe|><|notimestamps|> decode
      prologue; (4) BuildWhisperFromSafeTensors weight map in
      neural/neuralpretrained.pas (config.json: d_model/encoder_layers/
      decoder_layers/num_mel_bins). Pairs naturally with the seq2seq
      generation-harness task above (encode once, autoregress the
      decoder). Verify encoder hidden states + decoder logits vs
      transformers WhisperModel on a sliced pico fixture with a pinned
      mel input; headline demo: examples/WhisperTranscribe transcribing a
      short WAV to text on CPU.
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
      chat-template task's use case; shares the fork primitive with the
      KV-cache beam task above. Assert a forked session's continuation is
      bit-identical to a fresh prefill.
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
- [ ] Token healing (guidance-style): back up over the LAST prompt token
      and constrain the first generated token to extensions of its text,
      fixing the classic BPE boundary artifact ("http:" never continuing
      to "//" because the prompt split mid-merge). ~30 lines on top of the
      constraint machinery + tokenizer vocab prefix lookup;
      disproportionate quality win for completion-style prompts. Test: a
      pinned vocab where the healed and unhealed first-token distributions
      provably differ.
- [ ] Token-level logprob scoring + mini lm-eval harness: ScoreSequence
      (NN, tokens) returning per-token logprobs (one forward, no
      generation), then multiple-choice evaluation by length-normalized
      answer logprob — the HellaSwag/ARC/PIQA pattern. NOT the existing
      forced-sequence constraint (that FORCES generation; this SCORES
      candidates). This is what makes the importer program measurable:
      "imported SmolLM2 scores X on HellaSwag" is the end-to-end proof.
      Reuses the perplexity NLL plumbing in neuralnlpmetrics.pas; ship
      with a tiny pinned multiple-choice fixture for the harness itself.
- [ ] Generation-quality / degeneration metrics in neuralnlpmetrics.pas:
      distinct-n (Li et al. 2016), self-BLEU (reuses the landed
      CorpusBLEU), and repetition rate — the standard degeneration suite.
      The contrastive-search and sampling tasks elsewhere in this list
      have no way to demonstrate their benefit without these. Pinned
      hand-computable fixtures (e.g. "a a a a" distinct-1 = 1/4).
- [ ] Needle-in-a-haystack long-context eval harness: place a fact at
      varying depths in a synthetic long context, measure retrieval
      accuracy vs (depth, context length) as a small grid report. The
      RoPE-scaling and KV-cache-eviction tasks elsewhere in this list both
      NEED this to demonstrate they work — neither lists an eval. Works
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
