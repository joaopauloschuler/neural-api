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
- [x] Llama-architecture safetensors importer (sibling of the landed GPT-2
      HuggingFace import in neural/neuralpretrained.pas): RMSNorm + SwiGLU FFN +
      RoPE + GQA are all available as building blocks, so a TinyLlama/Llama-style
      checkpoint loader is mostly weight-mapping work (untied embeddings, no
      biases, per-layer q/k/v/o + gate/up/down proj names). Reuse the GPT-2
      parity tooling (slicer + logit dump + compare, commit aff96f5) to verify
      logit parity against HF transformers on a sliced tiny checkpoint.
      DONE: BuildLlamaFromSafeTensors[Ex/WithConfig] + TNNetTokenRMSNorm +
      ReadLlamaConfigFromJSONFile (rotate_half q/k row permutation, tied or
      untied LM head); verified vs transformers' LlamaForCausalLM at ~2e-7
      max |logit diff| (untied / tied / sliced) via examples/LlamaImport;
      committed pure-Python-oracle fixture test TestLlamaLogitParity.
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
- [ ] SentencePiece / tokenizer.json tokenizer loading for the Llama importer
      (examples/LlamaImport currently drives the net with raw token ids):
      parse HF tokenizer.json (BPE/Unigram vocab + merges + byte-fallback) or
      the sentencepiece .model protobuf into a TNeuralTokenizer-compatible
      encoder/decoder so LlamaImport can take text prompts end to end.
- [ ] Sampling decoders + logits-processor chain in neural/neuraldecode.pas
      (transformers GenerationMixin port): temperature, top-k, top-p, min-p,
      repetition/frequency/presence penalties as a `DecodeSample` sibling of
      DecodeGreedy/DecodeBeamSearch. Generalize TNNetTokenConstraint from
      "mask allowed tokens" into a chainable logits-processor abstraction
      (transform the logit vector in order) so penalties, temperature and
      the existing JSON/forced-sequence constraints compose in one pipeline.
      Add a small TGenerationConfig record plus stopping criteria (EOS-id
      list, stop strings, max new tokens).
- [ ] LoRA adapters (PEFT port): freeze a base PointwiseConvLinear /
      FullConnect layer's weights and add a parallel low-rank B·A path
      (rank r, scaling alpha/r, B zero-init so the start is a no-op) with
      its own learning rate; provide MergeLoRA to fold B·A into W for
      inference and a builder that wraps existing projection layers.
      Killer combo with the Llama safetensors importer: cheap fine-tuning
      of imported pretrained models. Follow-up: load HF PEFT adapter
      safetensors files onto an imported base model.
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
- [ ] Speculative decoding in neural/neuraldecode.pas: small draft model
      proposes K tokens greedily, target model verifies them in one
      forward pass with the standard accept/resample rule; output
      distribution provably identical to target-only sampling. Both nets
      are plain TNNets (the DPO trainer already holds two models), and
      TNNetStreamingDecoder gives the harness.
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
      half of Adam's state), and optionally Muon for 2-D weight matrices.
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
- [ ] Mixup / CutMix training augmentation (torchvision transforms-v2
      staples): sample-pair convex interpolation of inputs AND targets
      (Beta-distributed lambda) in the fit loop; CutMix patches a rectangle
      instead. The CIFAR image-classification examples give an instant
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
      tokenizer.json task above — parse the chat template from
      tokenizer_config.json (or hardcode the common Llama/ChatML/Zephyr
      formats) so an imported instruct checkpoint can be prompted with a
      correctly formatted conversation end to end.
- [ ] resize_token_embeddings equivalent: grow/shrink the token embedding +
      LM-head vocab of an imported model (mean-init new rows, keep tied
      heads consistent, update FStruct vocab sizes). Needed the moment
      anyone adds special tokens to fine-tune on top of the Llama/GPT-2
      importers.
- [ ] More importer architectures: Mistral / Qwen2 are ~weight-mapping
      deltas on the landed Llama path (sliding-window attention and QKV
      biases already exist as building blocks); Phi slightly more work.
      Each reuses the HF-parity fixture tooling (slicer + logit dump
      + compare). Gemma is broken out into its own per-generation task
      track below (gap analysis 2026-06-11: Gemma-1 is ~80% load-time
      weight folding on the Llama path). Qwen3 (0.6B is the most-used
      small instruct model) is a small delta ON the Qwen2 path: dropped
      QKV biases + per-head QK-norm — the SAME ingredient as the
      "Gemma 3 - per-head QK-norm" task below, so land that once and both
      importers consume it.
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
- [ ] RoPE scaling for context extension (linear / NTK-aware / YaRN /
      dynamic-NTK): neural/neuralpretrained.pas already PARSES the
      rope_scaling config key and then ignores it ("long-context scaling is
      not wired here yet", ~line 172). Wire a scaling mode + factor into the
      RoPE layer so imported Llama-family checkpoints run past their trained
      context. Linear ("position interpolation") and NTK-aware are pure
      frequency-remap formulas; YaRN adds per-band interpolation + an
      attention-temperature factor. Verify: HF parity fixture with a
      rope_scaling config (transformers applies the same remap), plus a
      sanity check that factor=1 stays bit-identical to the landed path.
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
      per-sequence logprobs, and the sampling-decoder task above provides
      generation. Cheap follow-ups on the same plumbing: ORPO / SimPO / KTO
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
      variant: majority-vote over extracted answers. Trivial once the
      sampling-decoder task lands; worth its own entry as the canonical
      harness.
- [ ] Sequence-length warmup curriculum in neuralfit.pas: train at short
      context first and grow SeqLen on a schedule (the rebuild-same-
      architecture-at-a-new-width idiom this list already notes near the
      variable-context trick) — large early-epoch throughput win for LM
      pretraining. Needs CopyWeights across width rebuilds + a schedule
      hook; the Trainer-callbacks task above is the natural home. Test:
      weights survive a width hop bit-for-bit, loss continuous across the
      hop.
- [ ] Sharded safetensors support (model.safetensors.index.json):
      TNNetSafeTensorsReader opens exactly ONE TFileStream, but anything
      above ~2B params (and many smaller repos) ships as
      model-00001-of-000NN.safetensors + an index file. Parse the index
      JSON, map tensor name → shard file, keep a stream pool behind the
      existing GetDType/read API so BuildGPT2/BuildLlamaFromSafeTensors
      (and the listed Mistral/Qwen2 importer task, which silently depends
      on this) need no changes. Test: split the pico-Llama parity fixture
      into two shards + index, assert logits bit-identical to the
      single-file load.
- [ ] HF Hub download helper: there is no HTTP anywhere in the import path —
      users must hand-download checkpoint files. A small fphttpclient-based
      resolver (https://huggingface.co/{repo}/resolve/{rev}/{file}, local
      cache dir, skip-if-present, optional token header for gated repos) so
      BuildLlamaFromSafeTensors('TinyLlama/TinyLlama-1.1B-Chat-v1.0') works
      end to end. Pairs with the sharded-safetensors task (the index file
      says which shards to fetch) and the tokenizer.json / chat-template
      tasks (same repo, same fetch). Keep it a separate opt-in unit so the
      core importers stay offline-only.
- [ ] PyTorch pytorch_model.bin loader: a RESTRICTED unpickler for the
      torch.save zip format — the long tail of older/fine-tuned checkpoints
      never got converted to safetensors. State_dicts use a small pickle
      subset (a dozen opcodes + persistent-id storage references into the
      zip's per-tensor data files); implement exactly that whitelist and
      reject everything else by design, which makes the Pascal loader
      immune to the arbitrary-code-execution hazard that motivated
      safetensors. Expose the same named-tensor API as
      TNNetSafeTensorsReader so the GPT-2/Llama builders take either
      format. Test: torch.save a pico state_dict in Python, assert every
      tensor matches its safetensors twin bit-for-bit.
- [ ] BERT/encoder-family importer (BERT / RoBERTa / DistilBERT / MiniLM):
      neural/neuralpretrained.pas is decoder-only and the importer task
      above (Mistral/Qwen2/Phi/Gemma) is also all-decoder. Encoders are a
      different skeleton: learned absolute + token-type (segment)
      embeddings, post-LN blocks, GELU, optional pooler head — all existing
      building blocks. Highest-leverage importer on this list: it feeds the
      token-classification head, the QA span head, and (via
      sentence-transformers MiniLM) real pretrained sentence embeddings for
      the InfoNCE/retrieval side. Verify with the HF-parity fixture tooling
      (slicer + hidden-state dump + compare) against BertModel.
- [ ] T5/Flan-T5 (encoder-decoder) importer: the natural companion to the
      seq2seq generation harness task above. New ingredient is T5's
      relative-position bias buckets (shared across layers, separate
      enc/dec); RMSNorm-style scale-only norm and gated FFN map onto landed
      blocks. Flan-T5-small is 80M params — genuinely CPU-friendly and
      instruction-tuned, so it doubles as the first imported model the
      BLEU/ROUGE metrics can score out of the box. Same HF-parity fixture
      verification as GPT-2/Llama.
- [ ] ForSequenceClassification checkpoint import: load FINE-TUNED
      classifier checkpoints (sentiment / NLI / toxicity), not just base
      LMs — classifier-head weight mapping plus id2label from config.json
      so predictions come out as label strings. Cheap delta on the
      BERT-family importer above; also applies to decoder classifiers
      (GPT2ForSequenceClassification uses the LAST non-pad token's hidden
      state — document the pooling difference). Test: parity fixture
      asserting class logits match transformers.
- [ ] CLIP importer (dual text/image encoder): the image side is a
      patch-embedding conv + the same pre-LN transformer blocks; text side
      is a small causal encoder; both end in a learned projection to a
      shared space + temperature. Flagship demo potential: ZERO-SHOT CIFAR
      classification via the repo's existing image pipeline with no
      training at all (encode class-name prompts, cosine-match image
      embeddings). Verify embedding parity per tower against transformers'
      CLIPModel on the sliced-fixture pattern.
- [ ] AutoModel-style dispatch (BuildFromPretrained): read config.json's
      model_type and route to the right Build*FromSafeTensors builder —
      two exist today (gpt2, llama) and the importer tasks above add more;
      one entry point stops every example from hardcoding the family, and
      is the natural place for a clear "model_type X unsupported, supported
      are: ..." error. Trivial once a third importer lands; keep the
      explicit builders public for callers that want compile-time choice.
- [ ] Streaming/lazy tensor materialization with load-time quantization:
      the import path materializes full FP32 tensor buffers before copying
      into layers, so PEAK import memory, not steady-state, can be the gate
      on commodity RAM (the TinyLlama ~4.4GB case the quantized-inference
      task documents). Read one tensor at a time from the (seekable)
      safetensors stream straight into the destination layer — or straight
      into int8 storage once the quantized-inference task lands — keeping
      only one tensor-sized scratch buffer. Assert peak RSS during import
      stays within tensor-size + model-size on the parity fixture.
- [ ] GloVe / word2vec / fastText pretrained-embedding loader into
      TNNetEmbedding: the classical-NLP counterpart of the checkpoint
      importers — examples/Word2VecSkipGram TRAINS embeddings, loading
      published pretrained ones is the missing half. Parse the standard
      text format ("word v1 v2 ... vD" lines, optional count/dim header for
      .vec), match rows against a TNeuralTokenizer vocab (mean-init
      misses), optional freeze flag. ~50 lines of parser that instantly
      upgrades every small-model NLP example. Test: pinned 3-word file
      round-trips into embedding rows exactly.
- [ ] NumPy .npy/.npz reader + writer: the universal interchange escape
      hatch — ANY framework can np.savez(**state_dict), and the format is
      trivial (magic + header dict + raw bytes; npz is a zip of npy).
      Reader gives a generic named-tensor import path for frameworks with
      no safetensors export; WRITER doubles as fixture tooling for parity
      tests (dump Pascal tensors, compare in Python) and is arguably an
      easier first Pascal→Python round-trip than the listed safetensors
      writer. Support F32/F64/F16 + int dtypes, C-order only, reject
      Fortran-order/pickled-object arrays explicitly.
- [ ] Cerebras-GPT parity verification (possibly ZERO-code "GPT-3 import"):
      Cerebras-GPT is the truest open GPT-3 reproduction (exact GPT-3
      recipe — dense attention, learned absolute positions, GPT-2 BPE,
      Chinchilla-scaled) and its HF checkpoints ship as model_type "gpt2"
      in GPT2LMHeadModel format, so BuildGPT2FromSafeTensors may load
      cerebras/Cerebras-GPT-111M TODAY. Task: run the existing GPT-2
      parity tooling (slicer + logit dump + compare) against it, fix
      whatever config tolerance breaks (n_positions=2048 etc.), and pin a
      sliced parity fixture. The cheapest credible answer to "import a
      trained GPT-3-class model"; doc cross-link target for
      ../gpt-3-for-pascal.
- [ ] GPT-Neo importer (EleutherAI's GPT-3 replica — the open
      implementation of GPT-3's ALTERNATING dense / locally-banded sparse
      attention): odd layers use a 256-key local window, which the landed
      sliding-window SDPA (FWindow / TNNetSlidingWindowMaskedFill) already
      covers. Two quirks: attention is UNSCALED (no 1/sqrt(d) — fold the
      compensating factor into W_q at load time, zero layer changes), and
      weights are plain Linear orientation rather than GPT-2's transposed
      Conv1D. Adds the "gpt_neo" route to the AutoModel-dispatch task.
      Verify with the HF-parity fixture tooling against GPTNeoForCausalLM
      (EleutherAI/gpt-neo-125m is CPU-sized).
- [ ] GPT-J / GPT-NeoX (Pythia) importer — the workhorse open GPT-3-class
      science suite (Pythia: 70M..12B with many training-step snapshots,
      untied embeddings; the untied-head path landed with the Llama
      importer). Two new ingredients: the PARALLEL residual block
      (x + Attn(LN(x)) + FFN(LN(x)) — one residual add of both branches, a
      builder variant; gptj uses one shared LN, gpt_neox two, gated by
      use_parallel_residual) and PARTIAL rotary (rotary_pct: only the
      first d_rot dims of each head get RoPE; the landed RoPE rotates full
      head dim). Adds "gptj"/"gpt_neox" AutoModel routes; Pythia-70M/160M
      are CPU-sized parity targets, larger ones depend on the
      sharded-safetensors task above. Verify against GPTNeoXForCausalLM
      with the parity fixture tooling.
- [ ] Gemma 1 - head_dim config override in the Llama import path: the ONE
      structural gap for Gemma-1 — neural/neuralpretrained.pas (~line 771)
      hardcodes HeadDim := HiddenSize div NumHeads, but Gemma-7B has
      head_dim=256 != 3072/16=192. Read head_dim from config.json when
      present, size the q/o projections as NumHeads*HeadDim != hidden_size
      (o_proj maps back to hidden_size), and keep the div fallback so all
      landed Llama fixtures stay bit-identical. Prerequisite for every
      Gemma task below; also future-proofs other head_dim-decoupled
      architectures.
- [ ] Gemma 1 - GeGLU FFN wiring: Gemma's MLP is gated GELU, not SwiGLU.
      TNNetGELU already implements the exact tanh approximation Gemma
      specifies (gelu_pytorch_tanh), so this is letting the importer's
      gated-FFN construction take the gate activation as a parameter
      (GELU vs SiLU) rather than any new layer. Test: gated-FFN forward
      parity against transformers' GemmaMLP on a pinned tensor.
- [ ] Gemma 1 - BuildGemmaFromSafeTensors importer (load-time weight
      folding on the landed Llama path; depends on the head_dim + GeGLU
      tasks above): (a) embedding output scaled by sqrt(d_model) — fold
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
      pattern as the GPT-Neo task); query_pre_attn_scalar (e.g. 224 on 27B)
      folded into W_q at load (same trick as GPT-Neo's unscaled attention);
      final-logit soft-capping as a plain TNNetSoftCapping.Create(30)
      before the softmax head — zero new code. Note the 256k vocab makes
      embedding+head the FP32 memory hot spot (quantized-inference task
      dependency for the larger sizes). Parity fixture vs
      Gemma2ForCausalLM.
- [ ] Gemma 3 - per-head QK-norm composition: Gemma-3 replaces soft-capping
      with RMSNorm applied per-head to q and k before attention. Likely
      pure composition: the multi-head builders split q/k/v into explicit
      per-head layers before SDPA, so insert TNNetTokenRMSNorm on the q and
      k branches — verify the insertion point sits AFTER RoPE the way
      transformers' Gemma3 does (norm placement relative to RoPE changed
      across implementations; pin it against the HF reference, do not
      guess). Gradient check + a wired-into-MHA-builder flag.
- [ ] Gemma 3 - BuildGemma3FromSafeTensors importer, TEXT-ONLY (depends on
      the QK-norm task + the Gemma-2 importer): 5:1 local:global layer
      ratio (config wiring over the same alternating machinery) and
      PER-LAYER-TYPE RoPE theta — 10k for local layers, 1M for global
      layers, so rope_theta becomes per-layer instead of one global config
      value (plumbing exists from the Llama config reader, needs a
      per-layer override). The 4B+ multimodal vision tower is explicitly
      OUT OF SCOPE (separate project). Parity fixture vs
      Gemma3ForCausalLM on a sliced text-only checkpoint.
- [ ] SmolLM2 parity verification (possibly ZERO-code, the
      Cerebras-GPT-pattern task for the Llama path): HuggingFace's
      SmolLM2 family (135M/360M/1.7B) ships as LlamaForCausalLM with tied
      embeddings — the landed BuildLlamaFromSafeTensors may load
      SmolLM2-135M TODAY. Run the parity tooling against it, fix whatever
      config tolerance breaks, pin a fixture. A 135M instruct-tuned
      2024-era model is arguably the best end-to-end CPU demo this
      framework could have; natural base model for the LoRA /
      distillation / GRPO tasks elsewhere in this list.
- [ ] TinyStories reference-checkpoint import (rider on the GPT-Neo
      importer task above): roneneldan/TinyStories-1M..33M are GPT-NEO
      architecture, so they load with zero extra work once that importer
      lands. Special fit: the repo's NLP examples already train on the
      TinyStories dataset, so this gives a direct "our from-scratch
      training vs the published reference checkpoint" bake-off (compare
      with neuralnlpmetrics perplexity at matched vocab), and at 1M-33M
      params the parity tests run against FULL checkpoints instead of
      sliced fixtures — the only importer family where that is true.
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
- [ ] ModernBERT importer (answer.ai, 139M) — the encoder worth targeting
      BEYOND vanilla BERT once the BERT-family importer task above lands:
      RoPE instead of learned positions, GeGLU, alternating local/global
      attention — every ingredient is already landed or tasked (Gemma /
      GPT-Neo machinery). Best current retrieval/classification encoder
      at CPU-friendly size; feeds the same token-classification / QA /
      sentence-embedding heads as the BERT task. Parity fixture vs
      ModernBertModel hidden states.
- [ ] distilgpt2 fixture (ZERO-code claim): 82M-param distilled GPT-2
      almost certainly loads with the landed BuildGPT2FromSafeTensors
      as-is (same architecture, 6 layers). Add it to the GPT-2 parity
      fixtures to claim the smallest practical pretrained English LM and
      a natural teacher/student pair (gpt2 -> distilgpt2) for the
      knowledge-distillation task elsewhere in this list.
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
      OWN draft model, no second checkpoint. The repo is unusually well
      positioned: the LogitLens/TunedLens frozen-body splice idiom already
      implements "read logits at layer k", and the speculative-decoding
      task elsewhere in this list provides the accept/verify rule. v1:
      static exit layer + confidence threshold; follow-up: per-token
      adaptive exit. Report tokens/sec vs full-depth at matched output
      quality.
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
- [ ] Calibration: ECE report + temperature scaling — expected calibration
      error as a TNNet.*Report batch diagnostic (introspection-report
      pattern: per-bin confidence/accuracy table + ECE scalar) plus a
      one-parameter temperature fit on a validation set (closed loop:
      report, fit T, report again, ECE drops). General beyond NLP
      (any softmax classifier); complements the evidential heads, which
      tackle the same problem by architecture instead of post-hoc.
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
      difference checks with a `Tolerance` parameter (default 1e-2) so
      the DeMaxPool-style Double-precision SSE accumulator can be opted
      into per-test.
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
- [ ] Promote DeMaxPoolFamilyGradientCheck's Double-precision SSE
      accumulator into the shared LayerInputGradientCheck (and weight-grad
      variant). Sum the SSE in Double; eps and tolerance stay TNeuralFloat.
      NEW DATA POINT: TNNetAdaptiveMaxPool's gradient check hit the same
      float32 subtractive-cancellation issue (a single cell carrying the
      whole window error, num=1.2588 vs ana=1.2709) and had to be loosened
      to tol 0.02 with an in-code comment — verified NOT a layer bug
      (double-precision central difference matches analytic exactly). A
      strong candidate to convert once the Double accumulator helper lands.
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
- [ ] Scheduler unit tests — given seed and schedule parameters, NextLR
      must produce a deterministic, finite, monotonically-correct sequence.
- [ ] Backward-pass sign-correlation test — for every layer that overrides
      Backpropagate, perturb input by ±ε, assert gradient direction agrees
      with loss-difference direction >90% of the time across a small grid.
- [ ] Coverage matrix at the top of TestNeuralNumerical.pas: per-class
      `[grad] [serialize]` block, written by a small script.
- [ ] LoadFromString round-trip for the entire activation menagerie — one
      parameterised test walking every TNNetReLUBase descendant.
