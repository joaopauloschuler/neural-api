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
  `TNNetMaxChannel` for global max pooling. `TNNetMaxChannel` now does a
  true (x, y)-over-all-positions reduction per depth channel, so it is
  correct on RECTANGULAR (`SizeX <> SizeY`) inputs too — no reason to
  reintroduce this class.
- `TNNetGlobalMinPool` — overlapped `TNNetMinChannel`. Use
  `TNNetMinChannel` for global min pooling. `TNNetMinChannel` is now
  rectangular-correct as well (see `TNNetMaxChannel` above).
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

- [ ] Possible defect (not fixed, awaits user decision): dense `TNNetSpectralNorm`
      (neural/neuralnetwork.pas ~13905) keeps the inherited FC weight delta
      (gradient w.r.t. w_eff) without the 1/sigma factor its conv sibling now
      applies; its declaration comment calls this intended, but dL/dW =
      (1/sigma) dL/dw_eff under the same constant-sigma rule.
- [ ] FFT-path FPU denormal/invalid-op traps in TNNetSpectralConv2D needed an
      example-side SetExceptionMask workaround — consider masking/guarding the
      denormals inside the layer's FFT so callers don't have to.
- [ ] SentencePiece `.model` `precompiled_charsmap` parsing — the opaque
      per-model normalization trie (NOT standard NFKC) is still not
      parsed/applied. A tokenizer.json that declares a standard NFKC/NFKD
      normalizer already works in full (`UnicodeNormalize` + `AddNormalizer`,
      landed); only the embedded charsmap is unhandled.

## Hardware Support
  - [ ] CUDA/HIP Support
    - [ ] Software Spec 
  - [ ] ARM SIMD: NEON → SVE → SME kernels behind the TNNetVolume dispatch (requires ARM processor) 
    - [ ] Software Spec

## Infrastructure / dev experience

- [ ] OpenCL test suite under 2 minutes (user goal and authorization
      2026-10-05; today ~600 s on PoCL). Diagnosis (single-test runs): idle
      scheduler workers spin hot against PoCL (one VAE test: OpenCL + parallel
      pass 95 s vs 3.3 s serial, 415k clock syscalls); neural.cl is built again
      for every EnableOpenCL; a few heavy test shapes. Serial tasks, each with a
      fresh read-only Opus reviewer, the three suites and a commit:
      S1-S5 (hot worker 0 only, neural.cl built once per process, smaller test
      shapes, fewer work-group sizes, opt-in slow set) are done; the goal is
      not yet met.

- [ ] ChatServer: multiple chat sessions in cache, so parallel chats do not
      re-prefill (design agreed with the user 2026-10-02). Conversation slots
      (`--kv-slots`, `TConversationSlot`, T1-T5) are coded; open items below.
      Compute stays serialized (one request at a time). Process: one Opus coder per task, then
      an independent Opus review, commit after review + green suites
      (lazbuild -B; default AND -dAVX2 suites; the PoCL suite when OpenCL code is
      touched; 3 GB ulimit). Each task ships its own tests.
  - [ ] T7. User acceptance on the GPU box: TTFT of two alternating real
        conversations on a real hybrid (qwen3_5) before and after.
  - [ ] Follow-up (not in this task): shared paged KV blocks, so conversations
        share the blocks of their common prefix instead of one full KV copy per
        slot (vLLM-style); a large KV-cache rewrite.
- [ ] Gradient checkpointing for training deeper nets in less memory
- [ ] GGUF import beyond Llama — open follow-ups (core `BuildFromGGUF`/`BuildFromGGUFEx`
      arch dispatch with llama/qwen2/gemma2 LANDED & verified):
  - [ ] cross-tool llama.cpp gemma2 GGUF parity (DEFERRED: needs the llama.cpp
        shared lib / CLI — network + native build — unavailable here; the Llama
        drift test does not shell out either).
  - [ ] starcoder2 GGUF (out of scope for v1): needs LayerNorm-not-RMSNorm + full
        per-projection biases + non-gated GELU on the GGUF import path.
  - [ ] Extend dispatch to more Llama-backbone GGUF archs (phi3, qwen3, gemma3,
        gpt2/neox, MoE families) once each has an offline parity oracle.
- [ ] ONNX import
- [ ] Gemma 4 import
- [ ] MiniCPM importer follow-ups (`BuildMiniCPMFromSafeTensors[Ex]` LANDED,
      model_type `minicpm`; OpenBMB μP-style scale_emb / per-sublayer
      scale_depth / dim_model_base logits folds on the core Llama path +
      SentencePiece `.model` tokenizer; pico parity TestMiniCPM{Config,Logit}Parity
      < 1e-4 vs float64 HF):
  - [ ] MiniCPM real-checkpoint slicer follow-up (slice_llama.py reuse) to
        parity-check a sliced openbmb/MiniCPM-1.2B/2B against the random pico
        fixture (network/RAM-gated; the landed parity is the hand-built pico
        oracle only).
  - [ ] MiniCPM4 / MiniCPM-MoE importers (separate model families; MiniCPM4
        adds its own long-context/sparse-attention pieces, MiniCPM-MoE adds the
        Mixtral-style MoE FFN — neither is the dense-1.2B/2B path landed here).
- [ ] InternLM2 / InternLM2.5 importer follow-up (`BuildInternLM2FromSafeTensors[Ex]`,
      model_type "internlm2"; internlm/internlm2_5-1_8b/7b/20b) — LANDED.
      A plain Llama-backbone family (RMSNorm + RoPE + SwiGLU, GQA) whose ONLY
      genuinely new piece is the checkpoint LAYOUT, resolved by an
      InternLM2->HF translating reader (`TNNetInternLM2Reader`, a
      `TNNetSafeTensorsReader` subclass) that rides the existing core Llama
      builder with NO new layer/config flags. It (a) unpacks the fused
      `attention.wqkv` (reshaped `[num_kv_heads, q_per_kv+2, head_dim, hidden]`,
      per-group q slices then single K then single V) into standard separate
      W_q/W_k/W_v with the group-interleaved row reorder (all Q heads
      group-major, then all K, then all V — HF rotate_half order, so the core
      builder's own rotate_half→interleaved q/k de-permute for RoPE applies
      unchanged), and (b) renames the InternLM2 tensors (tok_embeddings /
      wo / feed_forward.w1=gate|w2=down|w3=up / attention_norm / ffn_norm /
      output) to the HF Llama names the core builder reads. Tokenizer is a
      SentencePiece `.model` (already supported); `bias=true` rejected.
      Verified by a HAND-BUILT pico fixture (tools/internlm2_tiny_fixture.py —
      InternLM2 is NOT in transformers, needs trust_remote_code offline, so the
      reference is a self-contained numpy float64 forward) asserting next-token
      logits < 1e-4: tests `TestInternLM2ConfigParity` / `TestInternLM2LogitParity`.
  - [ ] real-checkpoint slicer not exercised: the translating reader's wqkv
        unpack rematerializes each q/k/v slice in RAM per tensor (fine at pico
        and 1.8B/7B widths); a streaming/int8 path for the 20B checkpoint and a
        live trust_remote_code parity run against a downloaded internlm2_5-* are
        left open.
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
- [ ] BitNet b1.58 importer follow-ups (`BuildBitNetFromSafeTensors[Ex]` LANDED,
      model_type "bitnet" / the released `microsoft/bitnet-b1.58-2B-4T`; the
      ternary-weight LLM mapped onto the Llama backbone with the SubLN
      norm-before-quantized-linear RMSNorms (attn_sub_norm before o_proj,
      ffn_sub_norm before down_proj via Config.BitNetSubLN, reusing
      TNNetTokenRMSNorm), the relu2 gated FFN via the new TNNetReGLUSquared
      activation (up*ReLU(gate)^2) on the separate-projection loader, and
      rope_theta from the transformers-5.x rope_parameters dict; the FP "shadow"
      checkpoint loads bit-for-bit since the absmean ternarize is a no-op on
      already-ternary weights; pico parity TestBitNet{Config,Logit}Parity < 1e-4
      vs float64 HF BitNetForCausalLM):
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
      (the parity fixture today uses raw token ids only).
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
- [ ] Nemotron-H-MoE importer follow-ups ('E' MoE block LANDED on the landed
      BuildNemotronHFromSafeTensors: DeepSeek-V3-style sparse non-gated relu2
      experts + always-on shared expert via the landed
      TNNetBiasBalancedTopKGate/TNNetTopKGate + TNNetSigmoid + TNNetMulByConstant
      (NO new layer); TestNemotronHMoELogitParity, schedule "M*E-", < 1e-4 vs a
      float64 HF oracle):
  - [ ] Nemotron-H-MoE grouped-routing follow-up: the importer REJECTS
        n_group>1 and moe_latent_size with a clear message. Wire the
        group-limited / latent-MoE routing path if a real Nemotron-H-MoE
        checkpoint needs it (DeepSeek-V2-style grouped topk gate is landed —
        reuse it).
- [ ] Zamba2 importer (model_type "zamba2") — the sibling Mamba-2 hybrid that
      additionally shares a small set of transformer blocks across depth with a
      per-invocation LoRA adapter; needs a shared-block + LoRA wiring helper
      (larger than Nemotron-H, distinct from the per-layer-independent schedule).
- [ ] SigLIP follow-up: NaFlex / variable-resolution siglip2 (the landed
      BuildSigLIPFromSafeTensors scopes to the FIXED-resolution siglip/siglip2
      base configs: square image_size, learned position table over a fixed
      num_patches grid). NaFlex siglip2 supports per-image patch counts
      (aspect-ratio-preserving resize to a variable token grid) with attention
      pooling over the variable grid and a padding/attention mask — needs a
      variable-SeqLen vision input, an interpolated/resampled position table,
      and key-padding-masked MAP pooling. Not yet wired.
- [ ] examples/TCNClassifier follow-up (needs user OK): a faster
      `TNNetCausalConv1D` forward/backward for narrow InputDepth (one AVX call
      per tap is overhead-bound at e.g. 5 features). The example's wall clock
      varies widely (8:08 and 33:34 measured), cause not yet diagnosed.

### Computer vision & generative models

- [ ] FastSAM real-time segment-anything importer (`BuildFastSAMFromSafeTensors[Ex]`,
      ultralytics FastSAM-s / FastSAM-x). Architecturally distinct from the landed
      ViT-encoder SAM path: FastSAM reframes "segment everything" as a YOLOv8
      instance-segmentation backbone (CSP/C2f conv stages + a prototype-mask head)
      followed by a CLIP-text or point/box prompt-selection step over the emitted
      masks — no heavy image encoder, so it runs interactively on CPU. v1 scope:
      the YOLOv8-seg backbone + proto-mask head producing the full mask set on a
      fixed input, reusing the landed conv/SPPF building blocks; the CLIP-prompt
      mask ranking as a follow-up. Pico-fixture parity < 1e-4 on the proto-mask
      logits vs a float64 reference.

- [ ] LLaVA-NeXT (llava-v1.6, model_type `llava_next`) dynamic-resolution VLM
      importer follow-up to the landed `BuildLlavaFromSafeTensors`. The v1.6
      difference is the "anyres" image processing: the input is split into a grid
      of native-aspect tiles plus a global thumbnail, each tile is CLIP-encoded
      separately, and the per-tile patch grids are spatially unpadded and
      concatenated into one visual-token stream (a large accuracy gain on dense
      OCR/chart images over the v1.5 single-square crop). v1 scope: the anyres
      tile-split + per-tile encode + token-merge connector reusing the landed
      CLIP vision tower and Llama/Vicuna decoder; `examples/LlavaNextDescribe`
      caption. Pico-fixture parity < 1e-4 on the merged visual-token features.

- [ ] OpenCLIP / timm vision-tower real-checkpoint round-trip follow-up
      (`BuildOpenClipVisionTower` LANDED, commit 794a10ce, neuralpretrained.pas —
      `TNNetOpenClipReader` maps flat `visual.*` open_clip keys onto the HF
      `vision_model.*` graph `BuildClipVisionTower` reads; pico parity
      TestOpenClipVisionTowerParity < 1e-4 on the already-parity-verified CLIP
      vision graph): verify against the actual `open_clip` reference (laion
      ViT-L/14, ViT-g/14) — open_clip lib + LAION weights unavailable offline.

- [ ] Janus-Pro unified multimodal importer (`BuildJanusProFromSafeTensors[Ex]`,
      deepseek-ai/Janus-Pro-1B and Janus-Pro-7B). Architecturally distinct from
      every landed VLM (LLaVA / BLIP-2 / Qwen2-VL / PaliGemma / Florence-2) and
      from the diffusion image-generation stacks: Janus-Pro is a SINGLE
      autoregressive Llama-backbone transformer with TWO decoupled vision paths
      sharing one token stream — a SigLIP "understanding" encoder (reuse
      `BuildClipVisionTower` / SigLIP path) projected in for image→text, and a
      VQ "generation" path (`gen_vision_model` VQ tokenizer + `gen_embed` +
      `gen_head`) that emits DISCRETE image tokens decoded back to pixels.
      v1 scope: the understanding path (image-conditioned text, an
      `examples/JanusDescribe` caption) reusing the landed Llama decoder + VQ
      codebook layers. Image-GENERATION head (sample image tokens, VQ-decode to
      an RGB grid) as the noted follow-up. Pico-fixture parity < 1e-4 vs a
      float64 transformers oracle on the understanding logits.

- [ ] VMamba / Vision Mamba (Vim) state-space vision-backbone importer
      (`BuildVMambaFromSafeTensors[Ex]`, state-spaces / MzeroMiko VMamba and
      hustvl/Vim-* checkpoints). Ports the 2-D selective-scan vision backbone on
      top of the already-landed `TNNetSelectiveSSM` / `AddMambaBlock`: patch-embed
      stem → stacked SS2D blocks that run the selective scan over FOUR directional
      flattenings of the (H,W) grid (the cross-scan) and merge them, with the
      hierarchical down-sampling stages of a classification backbone. The reusable
      new piece is the 2-D cross-scan wrapper (four directional traversals of a
      depth-contiguous feature map feeding the existing 1-D SSM, then summed);
      everything downstream reuses the landed Mamba layer + classifier head.
      Pico-fixture top-1 parity vs a float64 oracle, plus reuse of the tracked
      ImageNet-val eval harness on a sliced real checkpoint.

- [ ] HunyuanVideo / Mochi-1 text-to-VIDEO DiT importer
      (`BuildHunyuanVideoFromSafeTensors[Ex]`, tencent/HunyuanVideo and the
      genmo/mochi-1 sibling). Distinct from the landed CogVideoX and the planned
      Wan / SVD / AnimateDiff video stacks: a dual-stream → single-stream MMDiT (separate
      image- and text-stream blocks that later fuse into joint blocks, like
      FLUX's MMDiT but over a spatio-temporal latent) with 3-D RoPE over the
      (T,H,W) token grid and a 3-D causal-VAE latent space. v1 scope: the
      transformer denoiser fed pre-computed text embeddings + a random latent,
      reusing the landed diffusion schedulers (`TNNetDiffusionScheduler` flow /
      DPM paths) for the sampling loop; the 3-D causal VAE decoder as a noted
      follow-up (a few latent frames → pixel frames). Pico-fixture parity < 1e-4
      vs a float64 transformers/diffusers oracle on one denoise step.

- [ ] BiRefNet / RMBG-2.0 background-removal (dichotomous high-resolution image
      segmentation) importer (`BuildBiRefNetFromSafeTensors[Ex]`,
      ZhengPeng7/BiRefNet and briaai/RMBG-2.0, which ships the same BiRefNet
      architecture). Distinct from the landed semantic/instance/panoptic
      segmentation paths (SegFormer / Mask2Former / Mask-R-CNN): BiRefNet predicts
      a SINGLE high-resolution foreground alpha matte via a bilateral-reference
      decoder — a Swin-L backbone (reuse `BuildSwinFromSafeTensors`) feeding a
      localization module + a reconstruction decoder whose blocks each take a
      gradient/inward-reference and an outward-reference feature, upsampled to a
      full-resolution sigmoid mask. The reusable new pieces are the
      bilateral-reference decoder block and the multi-scale supervision head;
      reuse the landed bilinear upsample (`TNNetBilinearUpsample`) and Swin tower.
      Pico-fixture parity < 1e-4 vs a float64 transformers oracle, plus an
      `examples/BackgroundRemoval` cutting a subject out of a tiny CPU image
      (sigmoid mask -> RGBA composite).

- [ ] Hunyuan-DiT bilingual text-to-image importer
      (`BuildHunyuanDiTFromSafeTensors[Ex]`, Tencent `Hunyuan-DiT`). A distinct
      DiT from the landed PixArt/MMDiT/Sana stacks: a cross-attention DiT
      conditioned on BOTH a bilingual CLIP and an mT5 text encoder (two separate
      cross-attn streams concatenated), 2-D rotary position embedding on the
      image tokens, and a "skip-connection" (U-Net-style long skips between the
      first and second half of the transformer blocks, fused by a Linear). The
      reusable new pieces are the dual-encoder cross-attention wiring and the
      DiT-level long-skip Linear merge; reuse the landed VAE decoder + scheduler
      and scope a small parity fixture.
- [ ] LTX-Video text-to-video DiT importer (`BuildLTXVideoFromSafeTensors[Ex]`,
      Lightricks `LTX-Video`). Distinct from the landed CogVideoX and the planned
      Wan / SVD video importers: a single-stream MMDiT-style video DiT operating on tokens
      from a high-compression causal video VAE (patchified space-time latents with
      3-D rotary position embedding), conditioned on a T5 text encoder (landed),
      with rectified-flow sampling. High-value because it reuses the landed
      T5 tower + rectified-flow path and only the causal video-VAE patch
      tokenizer + 3-D RoPE wiring are new; scope a pico latent-space parity
      fixture (the full VAE decode can be a follow-up like the other video
      importers).
- [ ] Pixtral vision-language importer (`BuildPixtralFromSafeTensors[Ex]`,
      mistralai `Pixtral-12B`, model_type "pixtral"). Distinct from the landed
      LLaVA / Qwen2-VL / PaliGemma VLMs: a native-resolution ViT vision encoder
      with 2-D RoPE and a block-diagonal attention mask over a variable number of
      image patches, a GELU MLP connector, and a Mistral decoder (importer already
      landed) where each image is spliced in as a flat run of patch tokens framed
      by `[IMG]` / `[IMG_BREAK]` / `[IMG_END]` markers. The reusable new piece is
      the 2-D-RoPE variable-resolution vision tower + the patch-token splicing
      position builder; verify the tower against a transformers reference and the
      decoder against the landed Mistral parity fixture.
- [ ] Sana text-to-image importer (`BuildSanaFromSafeTensors[Ex]`, NVIDIA /
      Efficient-Large-Model `Sana_*`). A genuinely different image-generation
      stack from the landed PixArt/MMDiT path: a LINEAR-attention DiT denoiser
      (Sana's "linear DiT" replaces softmax SDPA with the landed
      `TNNetLinearAttention` plus a Mix-FFN), a Gemma decoder-only text encoder
      (reuse the landed Gemma importer), and a deep-compression autoencoder
      (DC-AE, f32 spatial compression) in place of the SD VAE. High-value because
      it exercises the linear-attention path end-to-end in a real generative model
      and reuses three already-landed subsystems; scope a small `Sana-0.6B` /
      `Sana-Sprint` parity fixture like the other importers.
- [ ] InternVL vision-language importer (`BuildInternVLFromSafeTensors[Ex]`,
      model_type "internvl_chat", e.g. OpenGVLab/InternVL2-1B). Distinct from the
      landed LLaVA / PaliGemma / Qwen2-VL VLMs: an InternViT vision tower + a
      PIXEL-UNSHUFFLE downsampler (`TNNetSpaceToDepth`) feeding an MLP projector
      into a Qwen2 / InternLM2 decoder (both importers already landed). The reusable
      new piece is the pixel-unshuffle token reducer + dynamic-tiling image
      preprocessor; verify on the 1B checkpoint against a transformers reference.
- [ ] DINOv3 ViT backbone importer follow-ups (`BuildDINOv3FromSafeTensors[Ex/WithConfig]`
      LANDED, HF model_type "dinov3_vit", facebook/dinov3-*; pre-LN ViT + LayerScale +
      2-D AXIAL RoPE on patch tokens (TNNetVisionRoPE2D + AddMultiHeadVisionRoPE2DAttention),
      register tokens prepended after CLS, BERT-unfused q/k/v/o_proj with UNbiased key;
      parity < 1e-4 vs a float64 HF DINOv3ViTModel oracle, scalar + AVX2 green):
  - [ ] use_gated_mlp (SwiGLU giant) variant rejected (only plain
        up_proj/down_proj+gelu wired).
  - [ ] non-square / non-native image_size (dynamic RoPE grid) not yet plumbed —
        TNNetVisionRoPE2D is fixed to the build-time grid. Still unblocks
        DINOv3-backed dense-prediction once the "register-token DINOv2 backbones
        rejected by BuildDPT" follow-up lands.

- [ ] Kandinsky 2.2 unCLIP text-to-image importer
      (`BuildKandinskyPriorFromSafeTensors[Ex]` + `BuildKandinskyDecoderFromSafeTensors[Ex]`,
      kandinsky-community `kandinsky-2-2-prior` + `kandinsky-2-2-decoder`). Distinct
      from every landed diffusion stack (SD UNet / PixArt / DiT / MMDiT block), which
      denoise in a VAE LATENT conditioned on a TEXT SEQUENCE: Kandinsky is a two-stage
      unCLIP pipeline. Stage 1 is a DIFFUSION PRIOR — a small transformer
      (`PriorTransformer`) that denoises a single CLIP IMAGE embedding vector
      conditioned on the CLIP TEXT embedding + pooled text (the genuinely-new piece:
      the model's "tokens" are [text-emb, pooled-text, time-emb, noised-image-emb,
      learnable-final] and the target is the clean image embedding, NOT a spatial
      latent). Stage 2 is a UNet decoder conditioned on that predicted image embedding
      (via an `image_embeds` add-embedding, no cross-attn text stream) denoising in a
      MoVQ (VQ-VAE) latent. Reuse: the landed CLIP text/image towers (image-emb
      conditioning), `BuildSDUNetFromSafeTensors` for the decoder UNet trunk (add the
      image-embeds projection path behind a config flag), `BuildVqModelFromSafeTensors`
      / VQ decode for MoVQ, and an existing DDIM/UniPC scheduler for both diffusion
      loops. The reusable new code is the `PriorTransformer` (a sequence-of-embeddings
      denoiser) + the image-embedding add-conditioning on the decoder UNet. Scope a
      pico parity fixture for each stage against a diffusers float64 oracle and an
      `examples/KandinskyGenerate` end-to-end ("a red fox in snow" -> tiny CPU image).

- [ ] Llama 3.2 Vision (mllama) cross-attention VLM importer
      (BuildMllamaFromSafeTensors[Ex], model_type "mllama", e.g.
      meta-llama/Llama-3.2-11B-Vision-Instruct). DISTINCT from the landed
      Qwen2-VL / Florence-2 / LLaVA paths, which fuse vision by PREPENDING visual
      tokens into the text sequence: mllama interleaves dedicated CROSS-ATTENTION
      decoder layers (cross_attn_layers) that let the text stream attend to a frozen
      ViT vision tower's patch features, gated by a learnable tanh gate
      (cross_attn_attn_gate / cross_attn_mlp_gate, zero-init so the base LM is
      unchanged at start). Genuinely-new code: the vision tower (a ViT with a
      pre/post tile-position embedding + global+local transformer stack over image
      TILES) feeding the cross-attention layers, the tanh-gated cross-attn residual
      injection (reuse TNNetCrossAttention + a learnable scalar gate), and the
      per-layer self-attn-vs-cross-attn schedule; the text self-attention decoder is
      the landed Llama backbone. v1 may take the vision tower's tile features as a
      precomputed TNNetInput (the Qwen2-VL "merged tokens as input v1" stance), then
      drop the shortcut. Pico parity < 1e-4 vs a float64 HF MllamaForConditionalGeneration
      oracle + an examples/LlamaVisionDescribe captioning a tiny CPU image.

- [ ] Grounding DINO open-vocabulary detection importer
      (`BuildGroundingDINOFromSafeTensors[Ex]`, model_type "grounding-dino",
      e.g. IDEA-Research/grounding-dino-tiny). Distinct from the landed OWL-ViT
      path: GDINO is a text-conditioned DETR with a Swin backbone, a BERT text
      encoder, a cross-modality feature enhancer (deformable image self-attn +
      bi-directional text<->image cross-attn) and a cross-modality decoder whose
      queries attend to BOTH towers. Reuse BuildSwinFromSafeTensors (image),
      BuildBertFromSafeTensors (text) and TNNetCrossAttention for the fusion;
      output = boxes + per-query token-similarity logits scored against the
      prompt's token spans. Pico-fixture parity + an examples/GroundingDINODetect
      ("detect: a cat. a remote." -> boxes) demo.

- [ ] SAM 2 promptable image + VIDEO segmentation importer
      (`BuildSAM2FromSafeTensors[Ex]`, e.g. facebook/sam2-hiera-tiny). The landed
      SAM v1 path covers the single-image ViT encoder + mask decoder; SAM 2 adds
      the pieces that make it a distinct model: a Hiera hierarchical image
      encoder, a memory-attention block that conditions the current frame on a
      memory bank of past-frame embeddings + object pointers, and a memory
      encoder that writes mask predictions back into the bank. v1 scope = single
      image (skip the bank); follow-up = the streaming video loop. Reuse the SAM
      prompt encoder + two-source TNNetCrossAttention. examples/SAM2Segment
      (point/box prompt on one frame, then carry the mask across a short clip).

- [ ] Depth Anything V2 importer follow-ups (BuildDepthAnythingV2FromSafeTensors[Ex]
      LANDED, commit 51036f9, depth_anything model_type, e.g.
      depth-anything/Depth-Anything-V2-Small-hf — DINOv2 ViT backbone -> DPT
      reassemble+fusion neck + 3-conv depth head, TDPTConfig.OutIndices block
      selection, pico parity TestDepthAnythingV2Parity max|diff|<1e-4 vs
      transformers float64 DepthAnythingForDepthEstimation; examples/DepthAnythingV2):
  - [ ] real-checkpoint parity not run (no weights downloaded).
  - [ ] metric-depth Depth-Anything-V2 (indoor/outdoor, depth_estimation_type
        "metric") untested on a real ckpt.
  - [ ] register-token / SwiGLU DINOv2 backbones still rejected by BuildDPT.

- [ ] Stable Diffusion VAE decoder importer follow-ups (BuildVaeDecoderFromSafeTensors[Ex]
      LANDED, commit a7870a1: diffusers AutoencoderKL decoder — post_quant_conv ->
      mid block single self-attention over HxW -> up blocks of ResNet groups +
      nearest upsample -> conv_out, latent scaled by 1/0.18215; reuse-only (nearest
      upsample = TNNetDeMaxPool(2) FSpacing=0; mid attention flattens (H,W,C)->
      (H*W,1,C) per-token SDPA; added GroupNormEpsilon to TNNetGroupNorm to pin
      diffusers eps 1e-6); TestVaeDecoderParity <1e-4 vs a numpy float64 oracle):
  - [ ] real-checkpoint (stabilityai/sd-vae-ft-mse) parity (diffusers is in the
        ~/x venv; not run yet); SDXL VAE uses different group counts / multi-head attention.
  - [ ] SD UNet importer LANDED (BuildSDUNetFromSafeTensors[Ex] + SDUNetDenoise,
        UNet2DConditionModel v1; TestSDUNetParity 2.75e-6 < 1e-4). Open follow-ups:
    - [ ] Real Stable Diffusion text-to-image from a diffusers folder (audit
          2026-10-02). No example loads a real SD checkpoint today: every SD
          example defaults to the tests/fixtures pico nets, and all of them fill
          the UNet text input with random/synthetic states instead of CLIP over a
          prompt (DiffusionInpainting.lpr:260, FillText in ControlNetCanny and
          T2IAdapterSketch). ControlNetCanny/T2IAdapterSketch run ONE denoise step
          and write no image. Missing pieces:
      - [ ] `TStableDiffusionPipeline` in neuralpretrained.pas with the same
            public shape as `TQwenImage21Pipeline` (Create(ModelFolder) over
            model_index.json + text_encoder/ tokenizer/ unet/ vae/ scheduler/,
            TokenizePrompt, EncodePrompt, Denoise, DecodeLatents,
            GenerateFromEmbeds, OnPhase/OnStep, load-use-free one component at a
            time). Scheduler from scheduler/scheduler_config.json onto
            TNNetDiffusionScheduler (dsScaledLinear; dpEps for SD 1.x, dpV for
            SD 2.x 768). Latent (H/8,W/8,4) scaled by 1/0.18215 (SDXL 0.13025).
      - [ ] CFG inside the SD loop: a cond + an empty-prompt uncond UNet pass per
            step (twice the per-step cost).
      - [ ] CLIP text encoder for SD: BuildClipFromSafeTensors loads a full CLIP
            (text + vision); check / add a text-only path for the
            CLIPTextModel-only text_encoder/ folder (unverified). SD 1.x feeds the
            last hidden state (77 tokens, padded); SD 2.x the penultimate layer;
            SDXL two encoders (CLIP-L + OpenCLIP-bigG) concatenated + the pooled
            vector into SDUNetDenoiseSDXL.
      - [ ] CLIP tokenizer from vocab.json + merges.txt: the SD 1.5 tokenizer/
            folder is believed to ship no tokenizer.json, which is what
            neuralhftokenizer.pas reads (unverified; else convert once).
      - [ ] F16 weights: the safetensors reader decodes F16, but the SD UNet / VAE
            loaders are unverified on the F16 (and *.fp16.safetensors variant)
            files most SD repos ship.
      - [ ] Resolution: the UNet TNNetInput is sized from Config.LatentGrid at
            build, so each image size needs a rebuild (or a resizable input).
      - [ ] Memory/speed: FP32 SD 1.5 UNet ~3.4 GB (over the 3 GB cap on the dev
            box, so the first real run is on the user's box); int8/int4 UNet
            weights (the int8 conv path exists, untried on this UNet), UNet
            OpenCL arming, per-prompt caching of the text cross-attention K/V.
      - [ ] CLI: either teach examples/QwenImage to pick the pipeline from
            model_index.json `_class_name` (QwenImage21Pipeline vs
            StableDiffusionPipeline) so the argument parsing / reporter / OpenCL
            selection are shared, or a separate examples/StableDiffusion over the
            same pipeline class. Then wire DiffusionInpainting / ControlNetCanny /
            T2IAdapterSketch to the real CLIP text states and a full sampling loop.
      - [ ] Per-component parity against diffusers on real weights (the ~/x venv
            oracle recipe from the Qwen-Image project), SD 1.5 first, then SD 2.x,
            then SDXL.
- [ ] CogVLM / CogVLM2 vision-language importer (`BuildCogVLMFromSafeTensors[Ex]`,
      model_type "cogvlm"/"cogvlm2"). Architecturally distinct from the
      shared-trunk VLMs already imported (LLaVA / Florence2 / Blip2):
      CogVLM routes IMAGE tokens and TEXT tokens through SEPARATE per-block
      weights — a dedicated "visual expert" with its own QKV and FFN matrices
      running in parallel with the language QKV/FFN, selected per token by
      modality. The importer needs a token-type-routed block builder (text
      experts vs vision experts sharing the attention score space), the EVA-CLIP
      ViT image encoder + MLP adapter, and a pico safetensors parity fixture
      (`make_pico_cogvlm_fixture.py`, reusing the slice-real-checkpoint pattern)
      checked < 1e-4 vs a transformers float64 oracle. Add a CV captioning /
      VQA smoke under examples/ once a small checkpoint is obtainable offline.
- [ ] Mask2Former universal-segmentation importer LANDED
      (BuildMask2FormerFromSafeTensors[Ex], model_type "mask2former";
      masked-attention decoder + per-query mask/class heads, RunMask2FormerSemantic +
      DecodeMask2FormerSemantic, TestMask2FormerParity ~1.2e-7 < 1e-4,
      examples/UniversalSegmentation). Deferred follow-ups:
      (a) wire the full Swin backbone + FPN pixel decoder (mask_features + 3 multi-scale
          memory levels with level_embed + sine pos) into ONE forward — v1 takes the
          pixel-decoder outputs as PRECOMPUTED TNNetInput feature maps (mirroring Mask
          R-CNN v1's FPN-input stance); the masked-attention decoder + heads are complete;
      (b) instance + panoptic post-processing (v1 is SEMANTIC argmax only);
      (c) multi-head cross-attention beyond the pico 1-head fixture is wired generically
          (Heads loop) but only parity-verified at Heads=1;
      (d) a real-checkpoint smoke once a swin-tiny-semantic checkpoint is obtainable
          offline (the importer is config-driven and ready).
- [ ] NAFNet image-restoration importer follow-ups (BuildNAFNetFromSafeTensors[Ex]
      + TNAFNetConfig LANDED — TNNetSimpleGate gate layer + Simplified Channel
      Attention + U-Net of NAFBlocks with LayerNorm2d / depthwise 3x3 / PixelShuffle;
      pico parity TestNAFNetParity < 1e-4 vs a numpy float64 oracle;
      examples/ImageRestoration end-to-end CPU denoise):
  - [ ] Real official NAFNet denoise/deblur checkpoint parity (offline env — the
        weights are large / not redistributable; importer uses the canonical
        NAFNet state_dict key scheme so a real checkpoint should drop in).
  - [ ] PNG input/output (the example uses deterministic synthetic images + PPM;
        no PNG decoder in-tree).
  - [ ] NAFBlock dropout + the wider official width/block configs are wired by
        config but only the small pico shape is parity-pinned.
- [ ] SwinIR transformer image-restoration importer follow-ups
      (BuildSwinIRFromSafeTensors[Ex] + TSwinIRConfig LANDED — RSTB of Swin
      window/shifted-window attention + 3x3 conv residual + pixel-shuffle upsample
      tail, reusing the landed Swin attention blocks, NO new leaf layers; pico parity
      TestSwinIRParity / TestSwinIRConfigFromJSONFile < 1e-4 vs a float64 numpy
      oracle; examples/SwinIRRestore 2x SR smoke):
  - [ ] Pinned the classical-SR (pixel-shuffle upscale) variant; the
        same-resolution denoise tail (upscale=1, single conv reconstruction) is
        wired by config but only the SR shape is parity-pinned.
  - [ ] Real-checkpoint parity deferred (official SwinIR .safetensors are large /
        not obtainable offline); the importer accepts a real checkpoint path.
  - [ ] Lightweight-SR (no conv_after_body / single shared upsample) and the
        nearest+conv "real-world" SR upsampler variants not yet wired.
- [ ] RIFE video frame-interpolation importer follow-ups (BuildRIFEFromSafeTensors[Ex]
      + TRIFEConfig + TNNetBackwardWarp LANDED — IFNet intermediate-frame synthesis
      with accumulated flow residuals + sigmoid fusion-mask blend; pico parity
      TestRIFEParity < 1e-4 vs a float64 numpy oracle;
      examples/VideoFrameInterpolation one middle frame t=0.5, inference-only, on CPU):
  - [ ] Real Practical-RIFE checkpoint parity (offline-download deferred): the real
          IFNet is MULTI-SCALE (stride-2 IFBlocks downsample then bilinearly upsample
          the predicted flow) and RE-FEEDS the warped frames + previous flow/mask into
          each successive block; v1 refines from the SAME [frame0|frame1] input at full
          resolution. Wire the scale schedule + warped-frame re-feeding and verify
          against an exported hzwer/Practical-RIFE checkpoint.
  - [ ] Arbitrary-t interpolation (t != 0.5) and the recursive 2x->4x multi-frame
        schedule (RIFE's timestep encoding + recursion).
  - [ ] Privileged-distillation teacher (the training-time IFNet that sees the GT
        middle frame); inference-only path is landed.
- [ ] GFPGAN blind face-restoration importer (BuildGFPGANFromSafeTensors[Ex],
      TencentARC/GFPGAN) — leverages the LANDED StyleGAN2 generator
      (BuildStyleGAN2GeneratorFromSafeTensors) as a fixed facial prior: a U-Net
      degradation-removal encoder maps a low-quality face to the StyleGAN2 latent/
      noise inputs, and per-resolution encoder features are spatially modulated into
      the generator via CS-SFT (channel-split spatial feature transform) layers. The
      only genuinely new code is the small U-Net encoder + the CS-SFT scale/shift
      injection (an affine FiLM-like modulation — reuse the landed TNNetFiLM); the
      synthesis backbone is reuse. Distinct from NAFNet/SwinIR (generic restoration
      with no generative face prior). Pico parity vs a float64 oracle on the SFT
      injection for a fixed latent + examples/FaceRestoration on a tiny CPU image.
- [ ] LPIPS follow-ups — the metric LANDED (ComputeLPIPSDistance /
      LPIPSStageDistance / LPIPSUnitNormalize in neuralpretrained.pas, reusing the
      VGG importer's 5 relu taps; unit-normalize -> squared-diff -> per-stage lin
      head -> spatial-mean -> sum; parity test TestLPIPSDistanceParity vs a numpy
      float64 oracle, LPIPS(x,x)=0). The lin head is a LOADABLE per-stage weight
      vector; nil = the unweighted (1/C channel-mean) lpips lin_layers=False
      baseline (what the oracle pins). OPEN: (a) ship/import the OFFICIAL
      richzhang/PerceptualSimilarity lin{0..4}.model 1x1-conv weights + parity vs
      the reference calibrated LPIPS (needs the `lpips` weights, unobtainable
      offline here). (b) wire ComputePerceptualLossAndGradient (neuralpretrained.pas,
      the Johnson-2016 perceptual loss + d(Loss)/d(pixels) gradient through the
      frozen VGG; calibrated unit-normalize is a follow-up) into an SR example as
      an opt-in objective.
- [ ] ControlNet spatial-conditioning importer LANDED
      (BuildControlNetFromSafeTensors[Ex], lllyasviel/sd-controlnet-canny;
      hint stem + per-resolution zero-conv residual taps + ControlNetResiduals driver,
      TestControlNetParity 2.89e-6 < 1e-4). First conditioning-by-feature-injection model.
      DEFERRED follow-ups: (a) real-checkpoint slicer (slice sd-controlnet-canny like
      slice_llama.py) for parity vs real weights on top of the synthetic fixture;
      (b) the diffusers per-residual conditioning_scale multiplier (constant 1.0
      here); (c) T2I-Adapter (BuildT2IAdapterFromSafeTensors[Ex], landed)
      follow-ups: real-checkpoint slicer (TencentARC/t2iadapter_*_sd15v2),
      light_adapter variant, multi-adapter blending.
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
- [ ] Quantized inference follow-up: GPTQ/AWQ-style calibrated quantization
      (error-compensating rounding / activation-aware scale search) and
      FP16/BF16 weight storage as a zero-drift middle rung. GGUF dovetails (reader
      landed in neural/neuralgguf.pas): load Q8_0 blocks straight into the
      int8 weight-only storage instead of dequantize-then-requantize (the
      k-quant Q4_K/Q6_K/Q5_K/Q2_K dequant-at-load READ path has landed in
      neural/neuralgguf.pas).
- [ ] FP16 (half-precision) OpenCL compute path for the dot-product/matmul
      offload. PARTLY LANDED: `--gpu-fp16` narrows the activation (B operand)
      and the split-K partials to half for the int8/int4 GEMV/GEMM
      (`cai_dot_product_int8_h`, `cai_f32_to_half`, `cai_im2col_h`); user-measured
      +10.2% decode on the 27B. What remains below is the FP32-weight path
      (half weight buffers for non-quantized nets). ENV NOTE (2026-06-27): the only OpenCL device available here is
      PoCL on the host CPU, which does NOT advertise `cl_khr_fp16` (clinfo shows
      "Half-precision Floating-point support (n/a)"). The detection + opt-in
      buffer/kernel path can be written and the FP32-fallback verified, but the
      actual half-precision kernel CANNOT be exercised/parity-tested here — defer
      until a `cl_khr_fp16` device is available, or scope a host-side
      half<->float conversion test only and mark the GPU kernel unverified.
      Every GEMM kernel in `neural/neural.cl` (cai_dot_product,
      simpleGEMMT, myGEMM5/6) is FP32 today, and the landed FP16/BF16 weight
      storage is RAM-only — compute still runs the FP32 kernels (tasklist
      note above is explicit). For GPU LLM decode the binding constraint is
      device memory bandwidth, so a `cl_khr_fp16`-guarded half-precision
      kernel variant (FP16 device-side weight/input buffers, FP32 accumulate)
      would roughly halve buffer traffic and lift throughput on GPUs that
      advertise the extension. Scope: detect `cl_khr_fp16` at device init,
      add an opt-in FP16 buffer/kernel path in neuralopencl.pas wired through
      the existing TDotProductSharedKernel offload (reuse the
      NewVAs/weight-residency machinery), keep the FP32 kernel as the
      fallback when the extension is absent, and add a numerical test that
      checks the FP16 result matches the FP32 path within a half-precision
      tolerance on a fixed fixture. Distinct from the int8 work (this is a
      GPU compute path, not CPU storage) and from FP16 weight storage (this
      is the missing compute half).
- [ ] Tokenizer follow-ups for neuralhftokenizer.pas: GGUF tokenizer
      (LoadFromGGUF) DEFERRED: "bert" (WordPiece) and "no_vocab" raise EHFTokenizerError;
      gpt2 path assumes the cl100k Split pre-tok (GGUF carries no pre-tok
      config).
- [ ] neuralhftokenizer.pas pre_tokenizer leftovers from the Split/Metaspace
      batch (o200k_base and DeepSeek-V2/V2-Lite Split patterns are recognized).
      REMAINING sub-tasks: (i) RESIDUAL GAP: non-ASCII
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
- [ ] GGUF writer follow-up: the llama-cpp-python parity arm is
      wired but UNVERIFIED end-to-end (lib not installed here; pico demo GGUF may
      be too small for llama.cpp) — confirm argmax/ranking agree on a real
      checkpoint once the lib is available.
- [ ] Optimizer zoo expansion (SGD/Adam/AdamW + Lion + Adafactor exist now):
      REMAINING:
  - [ ] optional Adafactor follow-up: the omitted first-moment beta1 EMA + update
        RMS clipping + internal relative-step LR rule if a real fine-tune needs them.
- [ ] True single-kernel batched forward (real batch axis + attention padding
      mask for left-pad) — the lockstep DecodeBatchGreedy orchestration landed,
      but NN.Compute has no SIMD batch axis on the char path, so each step still
      pays one forward per running row. The vectorized batch is the actual
      prerequisite for an efficient speculative-decoding verify step.
- [ ] FlashAttention tiled online-softmax SDPA follow-ups (opt-in
      EnableTiledForward/ComputeTiled v1 LANDED — STANDARD-softmax forward only,
      forward-only, parity-gated TestSDPATiledOnlineSoftmaxParity < 1e-5): the
      exotic attention subclasses (differential/sink/ALiBi/cosine-sim/T5/soft-cap/
      segment/prefix-LM/block-causal/KV-cache) and the tiled (recompute-scores)
      BACKWARD are still open.
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
- [ ] Preference-optimization follow-ups on the landed DPO/GRPO trainers
      (TNeuralGRPOTrainer in neural/neuraldpo.pas LANDED: group-relative
      advantages + PG + DeepSeek-k3 per-token KL reusing the DPO softmax-backward
      plumbing, tests in tests/TestNeuralGRPO.pas). ORPO / SimPO / KTO
      loss-formula deltas LANDED on TNeuralDPOTrainer as a TNeuralPreferenceLossMode
      (plmSimPO/plmORPO are reference-free via CreateReferenceFree; plmKTO uses a
      reference, paired-batch simplification documented in the unit header), tests
      in tests/TestNeuralPreference.pas; Bradley-Terry TNeuralRewardModelTrainer
      also exists. REMAINING: a full
      unpaired-batch KTO variant (current KTO uses the paired pair-mean KL point).
- [ ] Sequence-length warmup curriculum in neuralfit.pas: train at short
      context first and grow SeqLen on a schedule (the rebuild-same-
      architecture-at-a-new-width idiom this list already notes near the
      variable-context trick) — large early-epoch throughput win for LM
      pretraining. Needs CopyWeights across width rebuilds + a schedule
      hook; the Trainer-callbacks task above is the natural home. Test:
      weights survive a width hop bit-for-bit, loss continuous across the
      hop.
- [ ] NumPy .npz follow-up (neural/neuralnumpy.pas reader/writer landed; WRITER
      is STORED-only): DEFLATE-compressed .npz WRITER (savez_compressed) +
      zip64 / >4GB archive support (reader currently rejects zip64).
- [ ] MMLU few-shot eval follow-ups (EvaluateMMLU + MMLUReport single-token
      A/B/C/D answer-letter scoring, per-subject + macro + micro, 0/5-shot, in
      neuralnlpmetrics.pas; TestNeuralNLPMetrics MMLU tests; examples/MMLUEval
      tiny embedded smoke subset — all LANDED):
  - [ ] examples/MMLUEval --full <path> hook: dump cais/mmlu (hendrycks_test)
        dev+test splits to a small text file via the venv-x datasets package and
        feed the questions through the same FormatQuestion/BuildPrompt builder
        + a real subword tokenizer (the smoke build hard-codes its questions).
  - [ ] EvaluateMMLU large-model path: reuse the left-padded DecodeBatchGreedy /
        KV-cache batch scoring instead of the per-question per-letter
        ScoreCompletion loop (acceptable for the smoke subset, slow at the full
        14k-question x 57-subject scale).
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
- [ ] RWKV/recurrent decode follow-ups (O(1) incremental decode for TNNetWKV +
      TNNetTokenShift + net-wide TNNet.BeginIncrementalDecode driver over
      TokenShift/WKV/SelectiveSSM/DiagonalSSM LANDED, commits 62165c1/cc1bfcb):
  - [ ] TNNetCrossWKV incremental path (two-source + asymmetric modes + receptance
        gate — non-trivial, deferred).
- [ ] Refactor examples/WhisperTranscribe to USE the landed cached forced-prefix
      seq2seq decode helper (`DecodeSeq2SeqForcedPrefixCached`, neural/neuraldecode.pas).
      NOT a clean drop-in today: the word-timestamp cross-attention alignment step
      needs a full-width forward over the wide decoder, so the width-1 cached
      decode would have to either skip alignment or run a second full pass.
      (The example still re-encodes the full prefix every step; needs ~4 GB
      virtual memory, ulimit -v 4000000.)
- [ ] Streaming corpus loader with shuffle buffer: the landed packing
      pipeline materializes the whole token stream in RAM (neuraldatasets
      builds one concatenated Stream array). Read large text/token files
      in chunks through a fixed-size shuffle buffer (the datasets-library
      streaming pattern) feeding the existing packer windows — the only
      way to pretrain on corpora bigger than RAM. Assert: same model
      quality on a small corpus vs the in-memory path at matched
      examples-seen, and bounded RSS on a corpus larger than the buffer.
- [ ] Wav2Vec2 -large / robust LayerNorm variant + pretraining (follow-up to the
      landed Wav2Vec2/HuBERT CTC importer, which supports ONLY the wav2vec2-base
      "group" feat_extract_norm + post-norm encoder; ReadWav2Vec2ConfigFromJSONFile
      rejects feat_extract_norm="layer" and do_stable_layer_norm=true loudly). The
      -large/robust path needs: LayerNorm on EVERY conv feature-extractor layer
      (not just GroupNorm on conv 0), a PRE-norm transformer encoder
      (do_stable_layer_norm), and the final encoder LayerNorm placement that
      pre-norm implies. Then wav2vec2 SELF-SUPERVISED pretraining (the quantizer /
      contrastive masked-prediction heads currently dropped as ignorable tensors).
- [ ] HiFi-GAN neural vocoder importer follow-ups (`BuildHiFiGANFromSafeTensors[Ex]`
      LANDED + `ReadHiFiGANConfigFromJSONFile`/`HiFiGANConfigToString` +
      `THiFiGANConfig` + the `TNNetHiFiGAN` channel-major holder
      (Synthesize / SynthesizeVolume); mel-spectrogram -> waveform, the reusable
      synthesis backend shared by Tacotron2 / FastSpeech2 / SpeechT5 / Bark /
      VITS-style models). The generator is implemented as a self-contained
      convolutional holder (like TEnCodecModel) doing the conv_pre ->
      ConvTranspose1d-upsample + MRF (AVERAGE of num_kernels dilated ResBlocks) ->
      conv_post -> tanh math directly on channel-major arrays; conv weights fold
      weight_norm g/v at import (or load plain folded `.weight` as SpeechT5HifiGan
      ships). Discriminators skipped (training-only). Parity-gated < 1e-4 against an
      HF SpeechT5HifiGan float64 oracle (tools/make_pico_hifigan_fixture.py ->
      tests/fixtures/tiny_hifigan*, TestHiFiGANSynthesisParity). Open follow-ups:
  - [ ] real-checkpoint smoke: resynthesize a clip with a downloaded `hifigan` /
        SpeechT5HifiGan generator (weight_norm fold path) and write it via
        SaveVolumeToWav16 (offline + RAM-gated here, so deferred).
- [ ] AVX (and optional OpenCL) acceleration for the channel-major conv1d inner
      loops in the self-contained audio holders — `TEnCodecModel`, `TNNetHiFiGAN`,
      `TNNetVits`, `TNNetMimi`, and the MusicGen EnCodec decode path. These holders
      deliberately do the conv1d / ConvTranspose1d math DIRECTLY on channel-major
      arrays (not via TNNet layers).
      Forward conv1d (commit 8e15052) and ConvTranspose1d (commit 2873a1f; Mimi
      30c2342, DAC 8277d25) are AVX-vectorized in all of these holders. REMAINING:
  - [ ] OpenCL offload of the Mimi (`RunMimiConv`) / DAC (`RunDACConv`) holders —
        blocked on a Double-precision shared dot-product kernel (their oracle gate
        is `< 1e-4` against a float64 reference; a Float kernel would not hold it).
        (Forward conv1d + ConvTranspose1d OpenCL offload for EnCodec/HiFiGAN/Vits
        has landed via the `EnableConvOpenCL`/`SetConvOpenCLMinWork` gated path.)

- [ ] Parler-TTS importer end-to-end follow-up (`BuildParlerTTSFromSafeTensors[Ex]`
      + `TParlerConfig` + `TParlerTTSModel` holder + `examples/ParlerTTS` LANDED,
      model_type `parler_tts`; (By)T5 description encoder cross-attention + shared
      `BuildMusicGenDecoderNet` codec-LM decoder over delay-patterned DAC codes +
      SDPA KV-cache width-1-twin AR decode; pico parity TestParlerTTSParity < 1e-4):
      pair the decoder with the real (By)T5 encoder + DAC decoder to a WAVEFORM
      end-to-end (all three importable in-tree); the real parler-tts-mini-v1
      checkpoint key mapping + tokenizers; classifier-free guidance; sampling controls.
- [ ] VITS / MMS-TTS end-to-end text-to-speech importer (`BuildVitsFromSafeTensors[Ex]`
      LANDED + `ReadVitsConfigFromJSONFile`/`VitsConfigToString` + the `TVitsConfig`
      record + the `TNNetVits` channel-major holder (Analyze / ExpandPrior /
      FlowReverse / Synthesize) + examples/TextToSpeech) — the FIRST text-to-speech
      model in the library (model_type `vits`: facebook/mms-tts-* and
      kakao-enterprise/vits-ljs; Kim et al. 2021). Like the HiFi-GAN vocoder it is a
      self-contained channel-major holder doing the INFERENCE math directly (conv1d /
      relative-position attention / WaveNet residual stacks). All four stages landed
      and parity-gated `< 1e-4` vs the HF `VitsModel` float64 oracle
      (`TestVitsSynthesisParity`, `tools/make_pico_vits_fixture.py` ->
      `tests/fixtures/tiny_vits*`): (1) the conditional-prior normalizing FLOW run in
      reverse — VITS's residual coupling is RealNVP/Glow ADDITIVE coupling
      (`log_stddev≡0`, so `second_half -= mean(first_half)` with the mean from a
      `conv_pre`->WaveNet->`conv_post` stack, weight_norm g/v folded) — note this
      REUSES the WaveNet/coupling math directly rather than the generic
      `TNNetAffineCoupling`/`TNNetInvertible1x1Conv` Glow layers (VITS's coupling is
      its own WaveNet-conditioned variant, not a layer-graph fit); (2) the HiFi-GAN
      DECODER reused from `TNNetHiFiGAN` via a shared `BuildVitsDecoderInto` (loaded
      under the `decoder.` key prefix, bias-free `conv_post`); (3) the
      relative-position text encoder + the DETERMINISTIC duration predictor + the
      monotonic length-regulator expansion; (4) the full text->waveform graph (the
      prior noise `z` is an EXPLICIT input so the parity test is deterministic). The
      posterior encoder (training-only) is dropped from the committed fixture and
      never required. Example writes a smoke WAV via `SaveVolumeToWav16`. Open
      follow-ups:
  - [ ] MULTI-SPEAKER models (`num_speakers>1` / `speaker_embedding_size!=0`, the
        global-conditioning `cond` convs into the WaveNet/duration/decoder) —
        rejected loudly.
  - [ ] real-checkpoint smoke: synthesize a sentence with a downloaded
        `facebook/mms-tts-eng` / `kakao-enterprise/vits-ljs` and write it via
        `SaveVolumeToWav16` (offline + RAM-gated here, so deferred).
- [ ] Kokoro-82M TTS, real checkpoint (`hexgrad/Kokoro-82M`, Apache-2.0, StyleTTS2 +
      iSTFTNet). Plan agreed 2026-10-12. Target: load the HF snapshot folder
      (`config.json`, `kokoro-v1_0.pth`, `voices/*.pt`) and write a 24 kHz WAV.
      Reference: the `kokoro` 0.9.4 package in the `x` venv (`kokoro/model.py`,
      `modules.py`, `istftnet.py`). One Opus agent per task, run serially, test +
      commit after each.
      Decisions:
      - REPLACE the pico `TNNetKokoro` with the real architecture. Its AdaIN used
        plain `gamma`; the checkpoint needs `(1+gamma)`. Removed in commit 403d17f2
        (`TNNetKokoro`, its config/AdaIN records and builders,
        `tools/make_pico_kokoro_fixture.py`, the `tiny_kokoro*` pico fixtures,
        `TestKokoroSynthesisParity`, examples/KokoroTTS); the names are free for
        the real model.
      - Noise: zero noise by default (repeatable output, simple parity), random noise
        as an option.
      - Code structure: `TNNet` stages run by `TNNet.Compute`, so OpenCL comes with
        the layers; as many `TNNet`s as needed, host steps between them (duration
        rounding + frame repeat, sine source + forward STFT).
      - English text goes through the G2P in `neuralg2p.pas`; a phoneme string
        is also accepted as input.
      Checkpoint facts (from the cached snapshot and the package source):
      - 81.7M parameters, 327 MB FP32; fits under the 3 GB ulimit, so real-checkpoint
        runs are possible on the dev box.
      - `kokoro-v1_0.pth`: torch zip, entries stored (no DEFLATE), root = dict of 5
        state dicts `bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`,
        every key prefixed `module.`. Weight norm stored as legacy `weight_g`/`weight_v`.
      - `voices/<name>.pt`: torch zip whose root is a BARE tensor [510,1,256]. Row =
        phoneme count - 1. Columns 0..127 = decoder style, 128..255 = prosody style.
        54 voices, 9 languages; only `af_heart` is downloaded here.
      - PL-BERT = ALBERT: word embedding 178x128, `embedding_hidden_mapping_in`
        128->768, ONE shared layer (`albert_layer_groups.0.albert_layers.0`) run 12
        times, 12 heads, FFN 2048, max 512 positions. Then `bert_encoder` 768->512.
      - AdaIN1d: InstanceNorm WITHOUT affine weights (none in the checkpoint), then
        `(1+gamma)*x + beta` with gamma/beta from Linear(style 128 -> 2C).
      - frame rate: decoder upsamples x2, generator x10 x6, iSTFT hop 5 = 600 samples
        per predicted frame = 40 frames/s at 24 kHz.
  Phase 0 — oracle and structure:
  - [ ] 0.4 Layers for the `TNNet` route:
        - [ ] `ShouldUseInterleavedDotProduct` picks `InterleavedDotProduct` for
          every conv with a multiple of 32 neurons; on 1D convs it is slower than
          `DotProductsTiled`: same conv at 250 ch (tiled) 199 ms serial / 89 ms
          threaded vs 256 ch (interleaved) 260 / 161-189 ms. Kokoro stage A's
          128/256/512 channel convs take the interleaved path. Measure on 2D convs
          too before changing the heuristic.
        - [ ] OpenCL weight borrowing (`BuildWeightOwner` + `EnableOpenCLInContextOf`)
          is not tested for `TNNetConv1DLinear` (the CPU borrowing build is).
        - [ ] The unused backward variants (`BackpropagateCPU`, `BackpropagateFastCPU`,
          `BackpropagateFastCPUDev`, `BackpropagateOpenCL`) ignore `FPaddingY` and
          `FDilationX`; only `BackpropagateFastTiledCPU` is called.
        - [ ] `TNNetDeconvolution` is single-threaded (`ChunkEligible` false); a
          neuron-axis `ComputeRange` over the CPU GEMM needs per-thread chunk
          scratch.
        - [ ] OpenCL: the scatter GEMM uploads the input, downloads all L*K*N
          products and scatters on the host; no resident source/output, and no
          borrowing (`EnableOpenCLInContextOf`) test for the deconv.
        - [ ] Base `SetPrevLayer` still builds `FConcatedWInter` (unused by the
          deconv) for 32-multiple channel counts: one extra weight copy (dense
          deconvs only now).
        - [ ] `TNNetSnake` scalar-alpha inference still runs the scalar
          `pcr_sinf` loop; the per-channel path is vectorized.
        - [ ] The new `cai_group_norm` and the OpenCL residency of the norm /
          activation chain are unmeasured on a GPU (PoCL parity only).
  - [ ] 0.5 Stage-net holder for Kokoro (stage A: `TNNetKokoro.PrepareStageA`;
        stages B/C run as plain code). Open:
        - [ ] `TNNetEmbedding` sizes its table in its constructor (before the net
          is known), so a borrowing build allocates and frees each table: 3
          tables, 114,176 floats on the real checkpoint. Other layers size none.
        - [ ] CPU profile of the real stage A: `TNNetLSTMCell` ~51% of layer time
          (the CPU input projection is per-step dot products, no GEMM; the OpenCL
          path batches it only at Depth >= 32), `TNNetPointwiseConvLinear` ~44%.
        - [ ] Real-checkpoint stage A on a GPU: not run (PoCL parity on the tiny
          fixture only). The AdaLayerNorms after the bi-LSTMs run on the CPU
          unless forced, because the LSTM output is on RAM.
  Phase 2 — components, in graph order:
  - [x] 2.1 ALBERT: extend `BuildBertFromSafeTensors` with the embedding projection
        and one shared layer (do not clone it); then `bert_encoder` 768->512.
        - [ ] `bfAlbert` through `BuildBertFromSafeTensorsWithConfig` (HF
          `albert-*` checkpoints) is untested; no config reader for model_type
          "albert"; pooler / classifier / int8 refused.
  - [x] 2.2 TextEncoder: embedding, 3x (weight-norm Conv1d k5, channel LayerNorm,
        LeakyReLU 0.2), bi-LSTM 512 -> 2x256.
  - [x] 2.6 `AdainResBlk1d`: two weight-norm convs + shortcut (nearest x2 upsample,
        1x1 conv without bias), times rsqrt(2). The x2 block uses a DEPTHWISE
        ConvTranspose1d (k3, s2, groups=C, output_padding 1): add groups to
        `THiFiGANConv` (or the `TNNet` equivalent).
        - [ ] Grouped `TNNetDeconvolution` stays on the CPU under OpenCL: each
          grouped deconvolution downloads its source and re-uploads its output.
        - [ ] A Y factor other than the X factor on SizeY > 1 runs
          `TNNetDeMaxPool` on the CPU (`cai_upsample_gather` has one factor).
  - [x] 2.9 Public forward complex STFT (n_fft 20, hop 5, Hann, centre reflect pad)
        in neuralaudio.pas; only power-spectrum and private FFT routines exist.
        - [ ] `TNNetForwardSTFT` is O(NFFT * bins) per frame; a long NFFT would
          want `RealFramePowerSpectrum`'s two-factor path with complex output.
        - [ ] The harmonic source runs scalar `pcr_sinf` (arguments up to ~4e4
          rad need its exact range reduction; the vector `Sin` was not checked
          at that range) on one thread; 41 ms total is small next to the nets.
  - [x] 2.11 Generator: noise convs (k12 s6 p3; k1), `noise_res`, ConvTranspose1d
        ups, reflection pad (1,0) on the last stage, MRF average of 3 resblocks,
        `conv_post` 128->22, exp (magnitude) / sin (phase), iSTFT via
        `ISTFTOverlapAdd` with the centre trim and the phase-sign flip.
        - [ ] `ISTFTCenteredMagnitudePhase` and `ISTFTOverlapAddReIm` allocate
          their scratch per call (once per utterance).
  Later (user 2026-10-13): MP3 in pure Pascal — decoder (minimp3 port, CC0)
  and/or encoder (shine port, LGPL); not scheduled.
  Not in scope: `hexgrad/Kokoro-82M-v1.1-zh` (different repo and vocab); int8.
- [ ] Mimi streaming neural-codec importer (`BuildMimiFromSafeTensors[Ex]`,
      model_type "mimi", e.g. `kyutai/mimi`) — LANDED (conv encoder/decoder +
      RoPE transformer bottleneck + semantic/acoustic split-VQ; codes match the
      HF `MimiModel` float64 oracle exactly, round-trip waveform max|diff| ~1e-12,
      `TestMimiParity`; examples/MimiCodec round-trip smoke). Open follow-ups:
  - [ ] Mimi REAL-checkpoint parity vs the downloaded `kyutai/mimi` (encode codes +
        decode waveform) — the pico fixture pins the wiring; a real run pins widths,
        the GELU/erf path at scale, and the F32-storage budget on a trained model.
  - [ ] Mimi STREAMING chunk-at-a-time encode/decode (HF `MimiConv1dPaddingCache`
        per-conv padding cache + KV-cache transformer decode via the landed SDPA
        Begin/EndIncrementalDecode) for O(1) per-frame Moshi-style inference.
- [ ] F5-TTS flow-matching TTS follow-ups (`BuildF5TTSFromSafeTensors[Ex]` + `TF5Config`
      + `ReadF5ConfigFromJSONFile` + `examples/F5TTS` LANDED, model `SWivid/F5-TTS`;
      NON-autoregressive, NON-GAN voice cloner regressing a mel via a conditional-flow-
      matching ODE through an adaLN-zero RoPE-SDPA DiT trunk conditioned in-context on a
      masked reference mel + a ConvNeXt-V2 char embed, reusing the landed DiT/FiLM/RoPE-
      SDPA blocks, no new leaf; pico parity TestF5TTSParity ~1.4e-5 < 1e-4 on the DiT
      velocity field): real `SWivid/F5-TTS` checkpoint key-mapping parity (offline/RAM-
      gated); the mel->waveform Vocos/HiFi-GAN vocoder pairing (v1 emits mel); non-default
      `rope_theta` (the SDPA RoPE base is fixed at 10000, rejected loudly); the E2-TTS
      flat-UNet variant (same flow-matching objective, simpler trunk); classifier-free-
      guidance strength sweep on the cond/uncond DiT pass.
- [ ] CLAP audio-text contrastive importer (`BuildClapFromSafeTensors[Ex]` +
      `BuildClapFromSafeTensorsWithConfig`, `TClapConfig`/`ReadClapConfigFromJSONFile`)
      + examples/ZeroShotAudioTag — LANDED. Audio-domain analogue of the CLIP
      dual-encoder (two nets, shared L2-normalized space, `exp(logit_scale_a)*cosine`),
      but genuinely distinct: the audio tower is an HTS-AT Swin transformer over a
      log-mel spectrogram (PURE REUSE of the landed Swin window machinery —
      `TNNetWindowAttention`/`TNNetGatherTokens`/`SwinBuildWindowLayout`/
      `SwinSetWindowBias`/patch-merge — under the `clap_audio_model` key spelling, plus
      a Conv2d patch-embed + token mean-pool + 2-layer `ClapProjectionLayer`), and the
      text tower is RoBERTa (built inline like `BuildBertFromSafeTensors` + BERT pooler
      + 2-layer projection). The HF `BatchNorm2d`(mel) + `reshape_mel2img` freq<->time
      transpose are folded into `ClapBatchNormMelImage` (caller supplies the
      `(time,mel,1)` image). NO new leaf layers. Reuses `ClipExtractEmbedding`/
      `ClipSimilarity` + new `ClapSimilarityMatrix`. Pico parity `< 1e-4` on BOTH
      embeddings vs the float64 HF `ClapModel` oracle (`TestClapParity`, generator
      `tools/clap_tiny_fixture.py`, committed `tests/fixtures/tiny_clap.*`).
      Follow-ups:
  - [ ] "fused" CLAP (`enable_fusion = true`, clap-htsat-FUSED): the local/global
        mel-fusion patch-embed (`mel_conv2d` + the attention-feature-fusion block) is
        rejected. Distinct windowing — own importer branch + fixture.
  - [ ] Real laion checkpoint parity + the log-mel frontend wiring: `ClapFeatureExtractor`
        produces the 1024x64 mel (depends on freq_ratio=4); wire `neuralaudio.pas`
        log-mel -> `ClapBatchNormMelImage` end-to-end and verify against a downloaded
        `clap-htsat-unfused`. (The RoBERTa BPE tokenizer is already importable.)
- [ ] AudioLDM 2 / Stable Audio latent text-to-audio (music & sound) capstone —
      text-prompt -> audio via LATENT diffusion, the audio analogue of the landed
      LatentTextToImage (PixArt/DiT + VAE) pipeline and the natural home for the
      music priority. Reuses the landed `TNNetDiffusionScheduler` samplers
      (DDIM / DPM-Solver++(2M) / Euler) and a VAE/EnCodec latent codec; the new piece
      is the importer for the audio U-Net/DiT denoiser conditioned on a text encoder
      (CLAP or T5/FLAN-T5, both already importable) plus the mel-VAE decode ->
      HiFi-GAN vocode -> WAV tail. Scope v1 as: denoiser + scheduler loop producing a
      latent, decoded to a short (~5 s) clip written via the WAV writer; defer
      classifier-free-guidance tuning and long-form generation to a follow-up.
      The WAV writer + HiFi-GAN vocoder it builds on have landed; the open
      piece is the audio U-Net/DiT denoiser importer.
- [ ] MusicGen MELODY decoder follow-ups (BuildMusicGenMelodyFromSafeTensors[Ex],
      facebook/musicgen-melody, model_type "musicgen_melody" LANDED — decoder-only
      chroma+text prepended conditioning, ComputeMusicgenMelodyChroma chromagram,
      pico parity TestMusicGenMelodyParity, examples/MusicGenMelody smoke):
  - [ ] a --download real-checkpoint mode for facebook/musicgen-melody (the example
        is pico-only) + a real-melody-conditioned smoke clip; stereo; classifier-free
        guidance / KV-cache for the melody decoder (text-MusicGen has them).
- [ ] SeamlessM4T-v2 follow-ups deferred from the landed S2TT v1:
      (1) the text-to-speech (T2ST) unit vocoder path (TextToUnit decoder +
      HiFi-GAN-style unit vocoder). (2) the UnitY2 two-pass decoding. (3) a real
      downloaded facebook/seamless-m4t-v2-large checkpoint + real SentencePiece
      tokenizer + a runnable examples/SeamlessTranslate S2TT end-to-end demo.
- [ ] examples/SpeechCommands keyword-spotting trainer — the audio analogue of
      SimpleImageClassifier and the first FROM-SCRATCH (no-import) audio training
      example. Feed the existing log-mel frontend (neuralaudio.pas) into a small
      conv stack to classify the Google Speech Commands v2 spoken-word set
      ("yes"/"no"/"up"/"down"/...). Pure CPU, ships a tiny downloader + a
      reproducible accuracy number; demonstrates that the audio frontend is usable
      for ordinary supervised training, not only for pretrained-model import.
      LANDED (examples/SpeechCommands, .lpr + .lpi + README + scripts/download_speech_commands.sh):
      default no-network SMOKE generates a deterministic synthetic 6-keyword set
      (fixed RandSeed; low/mid/high tones + up/down chirps + band-limited noise)
      and runs it through the REAL ComputeWhisperLogMel ((100,1,40) log-mel) into a
      Conv(24,5)/MaxPool4 -> Conv(32,3)/MaxPool4 -> Conv(48,3)/MaxPool2 ->
      FullConnectReLU(64) -> Dropout -> FullConnectLinear -> SoftMax stack trained
      with TNeuralFit. Smoke validation climbs 83.5% -> 100% over 2 epochs and the
      held-out test set scores 93.89% (chance 16.7%), well under the 280 s /
      ulimit -v 3000000 budget. An optional `--full <dir>` loads real Speech
      Commands WAVs via LoadWav16ToVolume (same frontend), documented but NOT
      exercised by the smoke. Open follow-ups:
  - [ ] run the `--full` path on real downloaded Speech Commands v2 and record a
        real-data accuracy number (the synthetic smoke is trivially separable —
        validation saturates at epoch 2 — so it proves the path, not a hard
        benchmark).
- [ ] Whisper word-timestamp follow-ups (v1 landed, scoped to one 30 s
      greedy window). REMAINING:
  - [ ] (b) multi-window stitching for clips > 30 s (carry the running time offset
        and merge the per-window word lists).
  - [ ] (c rest) read `generation_config.alignment_heads` from arbitrary model
        configs when present (v1 still hardcodes the curated tiny/base/small/medium/
        large head lists and falls back to all-heads otherwise; no config-plumbing
        + fixture yet).
- [ ] Speaker diarization importer (`BuildPyannoteSegmentationFromSafeTensors[Ex]` +
      `TPyannoteConfig`/`ReadPyannoteConfigFromJSONFile`, model_type `pyannote`) — LANDED.
      New leaf layer `TNNetSincConv1D` (SincNet band-pass, kernels materialized from two
      scalars (low_freq, band) per filter, Hamming-windowed; full forward+BPTT, input &
      weight numerical-gradient tests `TestSincConv1D*`, README layer row). Importer
      pipeline: SincConv front-end → abs/MaxPool/TokenLayerNorm → Conv+ReLU/MaxPool/
      TokenLayerNorm → bidirectional `TNNetMinLSTM` trunk (forward + time-reversed concat)
      → linear → per-frame 7-class **powerset** head + `PyannotePowersetDecode` to a
      per-speaker binary activity matrix. Parity-gated `< 1e-4` (observed ~2e-7) vs a
      hand-written numpy float64 forward oracle on a pico fixture
      (`tools/make_pico_pyannote_fixture.py` → `tests/fixtures/tiny_pyannote*`,
      `TestPyannoteParity`) — the `pyannote.audio` python package is NOT installed here, so
      the oracle reimplements the exact forward math on a re-randomized fixture.
      `examples/SpeakerDiarization` synthesizes a two-tone clip, prints a speaker-activity
      timeline + RTTM lines, saves a WAV via `SaveVolumeToWav16`. Open follow-ups:
  - [ ] Real `pyannote/segmentation-3.0` checkpoint key-mapping (the on-disk tensor
        names + the actual conv-stack depths/strides) instead of the pico fixture's
        re-keyed names, verified against a real `pyannote.audio` float64 oracle once the
        package is available.
  - [ ] Sliding-window inference + overlap stitching for clips longer than the model's
        receptive field, and turning the per-frame activity matrix into final diarized
        speaker turns (clustering across windows).
- [ ] ECAPA-TDNN speaker-verification real-checkpoint follow-up
      (`BuildEcapaTdnnFromSafeTensors[Ex]` + `TEcapaTdnnConfig` + `EcapaCosineScore`,
      new leaves `TNNetTDNNConv1D`/`TNNetAttentiveStatsPooling`, builder
      `TNNet.AddSERes2Block`, examples/SpeakerVerification, pico parity TestEcapaParity
      <1e-4 LANDED): real speechbrain/spkrec-ecapa-voxceleb checkpoint key-mapping
      (BatchNorm1d eval-mode stats, SpeechBrain TDNN naming — the pico SE block uses
      the simplified `/N²` AvgChannel scaling) + a VoxCeleb EER smoke (network/RAM-gated).
- [ ] SAM mask decoder as a real TNNet layer graph (TNNetCrossAttention two-source
      wiring) instead of the plain-array RunSAMMaskDecoder forward, so the decoder is
      trainable / fine-tunable end-to-end (v1 is inference-only). Needs a builder that
      threads the per-step query/key positional-embedding additions through the
      asymmetric token<->image cross-attentions.
- [ ] SAM full end-to-end importer + processor parity: a single BuildSAMModel that wires
      encoder + decoder + the HF SamProcessor coordinate transform (longest-side resize
      to image_size, point rescaling) so a real sam-vit-base checkpoint segments a real
      image from pixel-space clicks (v1 assumes input_image_size == raw pixel coords).
- [ ] Medusa / EAGLE tree-attention speculative decoding — a follow-up that is
      genuinely distinct from the landed SEQUENTIAL self-speculative paths
      (MTP-draft SelfSpeculativeDecoding + LayerSkip/CALM EarlyExitSelfSpeculative,
      which verify ONE linear draft sequence per step). Tree drafting proposes a
      TREE of candidate continuations (multiple top-k branches per draft head) and
      verifies them all in ONE forward pass using a block-diagonal TREE-ATTENTION
      mask so every root-to-node path is scored simultaneously, committing the
      longest accepted path. New pieces: the draft-tree builder (Medusa multi-head
      or EAGLE feature-autoregressive draft), the tree-attention mask construction,
      and the path-acceptance/commit walk. Assert greedy output is bit-identical to
      plain greedy on the target while issuing fewer target forwards; an
      examples/TreeSpeculativeDecoding demo reporting the accepted-tokens/forward
      speedup vs the linear self-speculative baseline.

- [ ] ImageNet top-1 / top-5 parity eval harness LANDED (EvaluateImageNet +
      ImageNetReport + TopKIndices + TNNetImageNetSample/Stats in
      neuralimagemetrics.pas, examples/ImageNetEval; synthetic smoke + --full <dir>
      real-val hook, TestNeuralImageMetrics green). Open follow-up:
  - [ ] Run against REAL ImageNet-val: wire one landed classifier importer
        (BuildResNetFromSafeTensors resnet18/50 etc.) over a real checkpoint + the
        50k-image ImageNet-val set and compare top-1/top-5 to the published
        torchvision numbers (the smoke is synthetic-only by design; torchvision
        not installed here). Mind the multi-GB RAM budget.

- [ ] End-to-end latent text-to-image generation example (examples/LatentTextToImage)
      that finally CHAINS the imported generative pieces into one pipeline on CPU.
      NOW UNBLOCKED via the landed PixArt path (no SD-UNet needed): T5 text encoder
      (BuildT5FromSafeTensors) -> PixArt latent denoiser (BuildPixArtFromSafeTensors,
      LANDED, parity < 1e-4) -> VAE decoder (BuildVaeDecoderFromSafeTensors, LANDED)
      -> RGB, driven by the existing TNNetDiffusionScheduler DDIM/DPM-Solver++ loop
      with classifier-free guidance (PixArt uses a null/empty-caption uncond branch).
      Steps 1 & 2 LANDED: examples/LatentTextToImage runs a CFG DDIM / DPM-Solver++(2M)
      loop over the pico PixArt (regression TestLatentTextToImageSmoke) and decodes the
      (6,6,4) latent through a matched pico VAE decoder to a (12,12,3) RGB P6 PPM.
      Remaining steps:
  - [ ] Step 3 — add the real T5 tower (BuildT5FromSafeTensors) + CFG (cond vs
        empty-caption uncond) and a hard-coded prompt; ulimit-bounded demo that
        generates one small image. The CV-generative-stack-composes capstone.
        (Steps 1+2 supply only DETERMINISTIC SYNTHETIC T5 states + pico fixtures;
        Step 3 swaps in a real T5 encoder + real PixArt/VAE checkpoint.)
  - [ ] Follow-up — the pico PixArt + matched pico VAE are RANDOM (smoke only);
        once a real checkpoint is wired (Step 3) re-verify the chain produces a
        sensible image, and consider a Karras-spaced / Euler-ancestral variant.
      Edit examples/README.md. Mind the 5-min/ulimit budget — default to a smoke run.

- [ ] Mask R-CNN instance-segmentation importer follow-ups (`BuildMaskRCNNFromSafeTensors[Ex]`
      + `BuildMaskRCNN` + `RunMaskRCNN` + `TMaskRCNNConfig` + `examples/InstanceSegmentation`
      LANDED, the FIRST instance-segmentation vertical — per-OBJECT binary masks; FPN
      top-down pyramid + `TNNetRoIAlign` of an externally supplied proposal + box head
      (fc6/fc7 -> cls_score + bbox_pred) + mask head (4x conv -> ConvTranspose2d -> 1x1
      per-class mask logits); pico parity TestMaskRCNNParity 1.8e-6 < 1e-4 vs a numpy
      float64 oracle). Deferred follow-ups (v1 scope was bounded to the tested core):
  - [ ] RPN / anchor generation + proposal NMS (v1 takes EXTERNALLY supplied proposal
        boxes; the RPN head + anchor grid + objectness/box-delta decode + top-k/NMS
        proposal selection are the missing front end for an end-to-end detector).
  - [ ] wire the real ResNet-50 + FPN BACKBONE into one forward (v1 feeds the FPN-input
        feature maps directly as TNNetInput levels; a full run should tap C2..C5 from
        BuildResNetFromSafeTensors and feed all FPN levels, then route each proposal to
        its FPN level by box area like torchvision's MultiScaleRoIAlign).
  - [ ] real torchvision maskrcnn_resnet50_fpn checkpoint parity (the landed parity is
        the hand-built pico numpy oracle only); + box-delta decode/clip + mask paste-to-
        image + per-class NMS for a meaningful instance overlay on a real photo.

- [ ] TrOCR optical-character-recognition importer follow-up (BuildTrOCRFromSafeTensors
      LANDED, commit 2000f69; DeiT encoder + Bart decoder on the T5EncoderStates two-net
      convention, TestTrOCRParity decoder logits < 1e-4 + examples/OCRTranscribe):
  - [ ] tensor-name mapping targets the transformers 5.11 layout
        (encoder.layers.N.attention.q_proj, decoder.model.decoder.*); add the published
        checkpoints' older DeiT naming if it differs (the fixture is the parity contract).
- [ ] EfficientNet (timm / torchvision efficientnet_b0..b7) importer follow-up
      (BuildEfficientNetFromSafeTensors LANDED, commit d35970c; AddEfficientNetMBConv
      reusing MobileNetV3 MBConv + conv-BN fold, b0..b7 share one JSON-driven builder,
      TestEfficientNetImageClassificationParity max|diff| < 1e-4 vs a numpy float64 oracle):
  - [ ] real torchvision/timm efficientnet_b0 checkpoint parity + top-1 via the tracked
        ImageNet eval harness (the landed parity is the hand-built pico oracle only).

- [ ] StyleGAN2 generator importer follow-ups (BuildStyleGAN2Generator + NEW leaf layer
      TNNetModulatedConv2D LANDED, commit 8d72b95; mapping MLP z->w + synthesis tower of
      modulated conv + ReZero noise + LeakyReLU + summed toRGB skips, inference-only,
      TestStyleGAN2GeneratorParity < 1e-4 vs a numpy float64 oracle + examples/StyleGAN2Generate):
  - [ ] per-pixel 1-channel BROADCAST random noise (v1 stores full-depth fixed noise maps
        in the safetensors for deterministic oracle-exact synthesis — wire the standard
        per-pixel broadcast noise for real stochastic synthesis).
  - [ ] real stylegan2 checkpoint (NVIDIA .pkl / a rosinality safetensors) parity once
        obtainable; the training path (discriminator + path-length reg) and StyleGAN3.
- [ ] MMDiT (Stable Diffusion 3 / FLUX.1) full text-to-image importer + sampler
      (dual-stream joint-attention BLOCK v1 LANDED — AddMMDiTJointBlock +
      BuildMMDiTBlockFromSafeTensors + ReadMMDiTConfigFromJSONFile): full
      BuildMMDiTFromSafeTensors stack (patch embed + combined CLIP+T5 prompt tower +
      pooled-projection conditioning + N joint blocks + final adaLN/unpatchify); the
      CONTEXT-FREE final block (context_pre_only=True, text stream dropped); the
      end-to-end rectified-flow Euler sampler (reuse examples/FlowMatching) into the
      LatentTextToImage capstone with NO SD UNet; real stable-diffusion-3-medium
      (or Flux-schnell) checkpoint parity; SD3.5 RMSNorm QK-norm support.
- [ ] AnimateDiff text-to-VIDEO motion-module importer (BuildAnimateDiffFromSafeTensors,
      e.g. guoyww/animatediff-motion-adapter-v1-5-2) — the FIRST video-GENERATIVE
      importer (a sequence of frames from a text prompt), a brand-new generative
      modality vs every landed image generator (DiT/PixArt diffusion, VisualGAN/
      StyleGAN2, MaskGIT). AnimateDiff inserts a TEMPORAL motion module after each
      spatial block of a frozen SD UNet: the per-frame token grids are transposed so
      attention runs along the TIME axis (each spatial location attends across frames)
      with a sinusoidal temporal position embedding, learning motion while the spatial
      weights stay fixed. The genuinely new code is that temporal-axis attention block
      (reuse the landed SDPA over a (NumFrames,1,C) reshape per spatial cell, exactly
      the VideoMAE space<->time transpose trick) + the zero-initialised residual
      injection into the frozen UNet (same wiring as the tracked ControlNet zero-conv).
      Builds on the landed SD UNet importer (BuildSDUNetFromSafeTensors); track as its
      natural successor alongside ControlNet. Pico parity vs a diffusers
      float64 oracle on one motion-module output for a fixed multi-frame latent; an
      examples/TextToVideo that writes a short animated GIF/PPM sequence on CPU. Note:
      a cheaper no-UNet route to video is bolting the same temporal block onto the
      landed PixArt DiT.
- [ ] CogVideoX text-to-VIDEO DiT real-checkpoint follow-up
      (BuildCogVideoXFromSafeTensors[Ex] + TCogVideoXConfig + DecodeCogVideoXVae +
      examples/TextToVideo LANDED, reusing TNNetMRotaryEmbedding 3D RoPE +
      TNNetCausalConv1D temporal VAE, no new leaf; pico parity TestCogVideoXParity
      < 1e-4): real-checkpoint parity (THUDM/CogVideoX-2b) vs a sliced real checkpoint
      (the make_pico_*_fixture slicer pattern) once weights are available, incl. the
      real diffusers 3D RoPE frequency layout / temporal VAE up/down blocks and a real
      T5 encoder over a tokenized prompt feeding BuildT5FromSafeTensors instead of the
      synthetic text states. Further gaps (audit 2026-10-02), which together mean
      the repo has NO usable video generator today:
  - [ ] Full AutoencoderKLCogVideoX decoder: DecodeCogVideoXVae is only the decode
        TAIL (one causal temporal conv + SiLU + pointwise conv per spatial cell),
        not the real up/down-block VAE, so even real denoiser weights cannot
        produce real frames.
  - [ ] A `TCogVideoXPipeline` over a diffusers model_index.json folder (the
        TQwenImage21Pipeline shape: tokenizer + T5 -> transformer -> VAE, one
        component in memory at a time, OnPhase/OnStep) and a CLI that loads it;
        examples/TextToVideo takes only a single .safetensors today.
  - [ ] CFG in the TextToVideo loop with a real empty-prompt uncond branch.
  - [ ] Output: frames are written as separate P6 PPMs only; add an animated
        GIF / PNG-sequence or a video-container writer.
- [ ] Wan 2.1 text-to-VIDEO DiT importer (`BuildWanFromSafeTensors[Ex]` +
      `TWanConfig`, model_type "wan", e.g. Wan-AI/Wan2.1-T2V-1.3B) — the current
      most-downloaded open text-to-video model, a flow-matching MMDiT distinct from
      the landed CogVideoX (Wan uses a 3D causal-VAE latent + a different DiT block:
      self-attn + cross-attn to a T5/umT5 text encoder + a per-block time-modulated
      AdaLN, no QK-norm). Reuse the landed MMDiT/DiT cross-attention + AdaLN
      modulation blocks and the 3D-RoPE / TNNetCausalConv1D temporal-VAE primitives
      from the CogVideoX path; add only the Wan-specific patch-embed (1 x 2 x 2
      patchify), the flow-matching sampler, and the Wan VAE decoder key mapping.
      Pico parity TestWanParity < 1e-4 vs a float64 HF WanTransformer3DModel; defer
      real Wan-AI/Wan2.1 weights (network/RAM-gated) to a slicer follow-up, plus an
      examples/TextToVideo --wan mode. Note: distinct enough from CogVideoX that the
      MMDiT-video path is reused, not duplicated.
- [ ] Stable Video Diffusion IMAGE-to-video importer (`BuildSVDFromSafeTensors[Ex]`
      + `TSVDConfig`, e.g. stabilityai/stable-video-diffusion-img2vid) — the
      image-CONDITIONED video generator that complements the landed CogVideoX
      TEXT-to-video DiT (different modality: an input frame, not a prompt, drives a
      short clip). A spatio-temporal U-Net denoiser: reuse the landed SD/diffusion
      2D conv-resnet + cross-attention blocks for the SPATIAL layers and add the
      TEMPORAL mixing (per-pixel attention/conv across the frame axis + the
      learnable per-block temporal mix factor) interleaved between them; CLIP-image
      embedding conditioning (reuse `BuildClipVisionTower`) plus the
      fps/motion-bucket/noise-aug micro-conditioning embeddings; EDM-preconditioned
      denoiser + the temporal VAE decode (reuse the landed VAE decoder path). Pico
      parity `< 1e-4` on one denoiser step + one VAE decode vs a first-principles
      float64 oracle on a committed random pico config (the CogVideoX fixture
      pattern), `tools/make_pico_svd_fixture.py` / `TestSVDParity`,
      `examples/ImageToVideo` driving the pico denoiser through EDM sampling and
      writing a per-frame PPM sequence. Real-checkpoint parity is the network/RAM-
      gated follow-up.
- [ ] VAR (Visual AutoRegressive, next-scale prediction) image-generation follow-ups
      (class-conditional v1 LANDED — BuildVARFromSafeTensors[Ex] + ReadVARConfigFromJSONFile
      + the TNNetScaledDotProductAttention.BlockCausalSegments scale-mask flag):
  - [ ] VAR real-checkpoint parity (FoundationVision/var) — v1 parity is vs a
        first-principles float64 oracle on a random pico config; verify against a sliced
        real checkpoint once weights are available (the make_pico_*_fixture slicer
        pattern).
- [ ] Structured-vision accuracy eval harness for the imported DETECTION and DENSE-
      prediction backbones (EvaluateDetectionMAP / EvaluateSegmentationMIoU + reports in
      neuralimagemetrics.pas, plus examples/VisionEval) — the verification backstop that
      the tracked ImageNet top-1 harness is for CLASSIFIERS, but for the importers whose
      output is NOT a class vector: boxes (DETR / YOLO / OWL-ViT) and dense maps
      (SegFormer / DPT-Depth). Each detection importer's parity test only compares raw
      head logits on one image, which catches a transposed weight but NOT a wrong box
      decode (cxcywh<->xyxy, sigmoid placement), NMS, or label permutation. Add: (a) a
      COCO-style mean-Average-Precision scorer (per-class precision/recall over IoU
      thresholds 0.50:0.95, the standard 101-point interpolation) over a small folder of
      labelled boxes; (b) a semantic-segmentation mean-IoU / pixel-accuracy scorer over a
      small folder of label-map PNGs. Pure CPU post-process on the landed importers'
      outputs; reports a per-class table + a few visual overlays. Distinct from the
      logit-parity tests (those pin the math; this pins the end-to-end pipeline incl.
      preprocessing/decode). The missing import-VERIFICATION mirror of MMLUEval for vision.
- [ ] Florence-2 unified vision importer follow-ups (BuildFlorence2FromSafeTensors
      [WithConfig] + TFlorence2Config + RunFlorence2Logits/RunFlorence2Projector +
      Florence2Quantize/DequantizeCoord location tokens + examples/Florence2 LANDED —
      visual-prefix BART encoder + BART seq2seq decoder, pure builder reuse, no new leaf;
      pico parity TestFlorence2Parity < 1e-4 + TestFlorence2LocationTokens vs the real HF
      Florence2ForConditionalGeneration float64 oracle):
  - [ ] The DaViT vision tower itself is the DEFERRED gap: v1 takes the tower's
        last_hidden_state feature map as a PRECOMPUTED input (like the tracked Qwen2-VL
        "merged visual tokens as input v1"). Build the real DaViT: a 4-stage conv-embed
        (kernel/stride/pad per stage, pre/post LayerNorm), per-block depthwise 3x3 convs
        (TNNetDepthwiseConv), WINDOW attention (partition H*W into windows; reuse
        TNNetWindowAttention), and the genuinely-new GROUPED CHANNEL attention (attention
        ACROSS channels within groups, scale = num_tokens^-0.5 — NO existing Pascal layer,
        needs a new leaf or a transposed-SDPA builder). Then drop the precomputed-feature
        shortcut and add the patch_embed/window-partition reshape plumbing.
  - [ ] Real-checkpoint parity (microsoft/Florence-2-base) + the real DaViT image
        preprocessing + the Florence-2 tokenizer (post-processing the <loc_>/box token
        stream into pixel boxes per task) + autoregressive DecodeFlorence2Greedy with a
        proper caption/OD parse; v1 runs a single greedy step + the location-token math.
  - [ ] <OCR>, <REFERRING_EXPRESSION_SEGMENTATION> (polygon <loc_> streams) and the other
        task tokens beyond <CAPTION>/<OD>; the polygon (de)quantization reuses the same
        per-coordinate <loc_> helpers but needs the polygon-point grouping parse.
- [ ] Qwen2-VL / Qwen2.5-VL vision-language importer follow-ups (M-RoPE v1 LANDED —
      BuildQwen2VLFromSafeTensors[Ex] + TNNetMRotaryEmbedding + Qwen2VLRunLogits):
  - [ ] The native-dynamic-resolution VISION TOWER (Conv3d patch embed + window
        attention over a variable patch grid + the 2x2 spatial patch-merger MLP) — v1
        takes the MERGED visual tokens as input (precomputed). Reuse the landed
        BuildClipVisionTower/SigLIP ViT path; build the variable-grid window-attention
        mask + the patch merger. Then drop the precomputed-embeds shortcut.
  - [ ] Qwen2.5-VL specifics (the temporal M-RoPE section for VIDEO, the per-frame
        time_interval, the windowed full-attention layer pattern, MRoPE rope_deltas for
        incremental decode) — the M-RoPE index builder already has the temporal slot.
  - [ ] Multi-image / multi-modality prompts (the v1 position builder assumes ONE
        contiguous still-image block); generalize the get_rope_index grouping.
  - [ ] Real-checkpoint parity (Qwen/Qwen2-VL-2B-Instruct) + an examples/Qwen2VLDescribe
        that captions a tiny CPU image (ulimit-bounded).
- [ ] CLIPSeg text-prompted zero-shot segmentation importer follow-ups
      (BuildCLIPSegFromSafeTensors[WithConfig] + TCLIPSegConfig + RunCLIPSeg LANDED —
      frozen CLIP dual encoder + a FiLM-conditioned transposed-conv decoder emitting a
      single-channel mask for a free-text prompt; pico parity TestCLIPSegParity < 1e-4
      vs the REAL HF CLIPSegForImageSegmentation float64 oracle; examples/CLIPSegPrompt
      writes a binary-mask PPM; v1 = single text prompt -> one mask, inference-only,
      use_complex_transposed_convolution = false):
  - [ ] Real-checkpoint parity (CIDAS/clipseg-rd64-refined) — verify against the REAL
        HF checkpoint logits (the offline env could not download it; the pico parity
        already pins the math vs the real HF classes). Needs the ~600 MB checkpoint +
        CLIP tokenizer ids; also exercises the default extract_layers [3,6,9] and
        reduce_dim 64 (the pico uses [0,1]/6).
  - [ ] Image-prompt (visual) conditioning — CLIPSeg can also condition on a PROMPT
        IMAGE (conditional_pixel_values -> clip.get_image_features pooled embedding)
        instead of text; v1 does text only. Add a RunCLIPSegImagePrompt path reusing
        the vision tower's pooled class-token embedding as the conditional vector.
- [ ] BEiT / data2vec-vision ViT importer follow-ups (BuildBeitFromSafeTensors[Ex]
      /WithConfig + TBeitConfig + ReadBeitConfigFromJSONFile/BeitConfigToString
      LANDED, e.g. microsoft/beit-base-patch16-224, facebook/data2vec-vision-base —
      full global attention with a per-LAYER learned cls-aware relative_position_bias
      reusing TNNetWindowAttention (no new leaf layer) + LayerScale (TNNetChannelMul)
      on both branches, learnable cls token, query/value biased + KEY bias-free, no
      absolute positions; BeitBuildRelPosIndex matches HF generate_relative_position_index
      exactly; pico parity TestBeitParity/TestBeitConfigFromJSONFile < 1e-4 vs HF
      float64 oracle via tools/beit_tiny_fixture.py + tests/fixtures/tiny_beit.*):
  - [ ] use_shared_relative_position_bias=true (one model-level table shared by
        all layers) is rejected; only the per-layer table is wired.
  - [ ] use_absolute_position_embeddings=true and use_relative_position_bias=
        false variants rejected (the published checkpoints don't use them).
  - [ ] BEiTv2 (vector-quantized) not validated.

- [ ] OPT decoder importer (BuildOPTFromSafeTensors[Ex], model_type "opt", e.g.
      facebook/opt-125m..2.7b) — LANDED (learned-absolute +2-offset positions,
      LayerNorm + ReLU FFN, pre-/post-LN per do_layer_norm_before, optional
      final_layer_norm, BuildFromPretrained dispatch; TestOPTNextTokenLogitsParity
      < 1e-4 vs float64 HF OPTForCausalLM). Open follow-ups:
  - [ ] Wire OPT as the blip2-opt decode tail (the original motivation): feed
        the Q-Former query/text states into BuildOPTFromSafeTensors and verify
        a BLIP-2-OPT generation path.

- [ ] DepthPro (Apple ml-depth-pro) sharp metric monocular-depth importer
      (BuildDepthProFromSafeTensors[Ex], apple/DepthPro). DISTINCT from the landed
      DPT and Depth-Anything importers, which produce RELATIVE/affine-invariant
      depth at a single patch scale: DepthPro is a MULTI-SCALE patch-ViT that runs a
      shared ViT (reuse the landed DINOv2/ViT tower) over the image at several
      resolutions, splits each into overlapping 384px patches, then fuses the
      per-scale patch features through a DPT-style convolutional decoder to emit a
      high-resolution METRIC depth map (plus a focal-length head for true scale).
      The genuinely new code is the image->patch tiling + per-scale feature
      stitching/merge and the metric (not normalized) depth head; the ViT encoder
      and the DPT fusion-block decoder are already landed and reusable. Real value:
      first sharp, boundary-accurate, metric-depth model in the tree. Pico-fixture
      smoke + a real-checkpoint parity follow-up.

- [ ] RT-DETR real-time detection-transformer importer
      (BuildRtDetrFromSafeTensors[Ex], e.g. PekingU/rtdetr_r50vd). A DISTINCT
      detection model family from the landed DETR (ResNet+sinusoidal+vanilla
      transformer decoder, the slow NMS-free baseline): RT-DETR is the real-time
      redesign — a CNN backbone (reuse the landed ResNet) feeding an EFFICIENT
      HYBRID ENCODER (intra-scale single-layer self-attention "AIFI" on the top
      feature map only + a CNN cross-scale fusion / PANet-style neck) and an
      IoU-AWARE / uncertainty-minimal QUERY SELECTION that seeds the decoder from
      the top-K encoder proposals instead of fixed learned object queries. The new
      code is the hybrid-encoder neck (AIFI block + CCFM up/down conv fusion) and
      the top-K query-selection head; the transformer decoder, box/class heads, and
      the DETR-style post-process can be reused from the landed DETR. Real value:
      the first real-time / production-grade detector importer (DETR is reference,
      not deployable speed). Pico-fixture smoke + a real-checkpoint top-1/box
      parity follow-up via the structured-vision detection eval harness.

- [ ] Fuyu decoder-only multimodal importer (BuildFuyuFromSafeTensors[Ex],
      model_type "fuyu", e.g. adept/fuyu-8b). Architecturally DISTINCT from every
      landed vision-language importer (CLIP/SigLIP-ViT tower + projector feeding a
      decoder — LLaVA, Qwen2-VL, PaliGemma, BLIP-2, Florence-2): Fuyu has NO
      separate vision encoder at all. Raw image patches are unfolded
      (patch_size x patch_size x channels), linearly projected by a single
      `vision_embed_tokens` dense layer straight into the decoder's token stream,
      interleaved with text tokens via `|SPEAKER|`/`|NEWLINE|` image-newline
      markers. The decoder is a Persimmon stack (standard RoPE attention with a
      QK-LayerNorm, square-ReLU MLP, separate qkv proj). New code is: the
      patchify+linear-projection input adapter (reuse TNNetReshape/SpaceToDepth for
      the unfold + a FullConnect for the projection) and the Persimmon block wiring
      (QK-LayerNorm before RoPE, ReLU^2 FFN — TNNetReGLUSquared already exists). Real
      value: covers the "patches-as-tokens" multimodal family (Fuyu, and the recipe
      generalizes to UI/document screenshots) that the encoder-tower importers
      cannot express. Pico-fixture parity < 1e-4 vs a float64 HF
      FuyuForCausalLM.forward, then a real adept/fuyu-8b caption smoke.

- [ ] Donut / Nougat OCR-free document-understanding importer
      (BuildDonutFromSafeTensors[Ex], model_type "vision-encoder-decoder" with a
      Donut-Swin encoder, e.g. naver-clova-ix/donut-base / facebook/nougat-base).
      A DISTINCT task family from the landed TrOCR (ViT encoder + text decoder for
      single-line OCR) and the caption VLMs: Donut/Nougat read a FULL document image
      and autoregressively emit STRUCTURED output (Donut: a `<s_...>` key-value
      sequence for receipts/forms; Nougat: Markdown+LaTeX for scientific PDFs) with
      NO external OCR step. New code is mostly wiring: the encoder is a Swin
      transformer (reuse the landed Swin/SwinIR window-attention blocks) and the
      decoder is an MBart cross-attention stack (reuse the landed mBART importer);
      the importer maps the `encoder.`/`decoder.` HF key namespaces and the
      shifted-token-id generation prompt. The genuinely new pieces are the
      Donut-Swin patch-embed/relative-position-index loading and the
      task-prompt-conditioned `RunSeq2Seq`-style greedy/beam decode over the
      document. Real value: first document-AI (forms, receipts, scientific-PDF
      transcription) importer; reuses Swin + mBART that already ship. Pico-fixture
      cross-entropy parity < 1e-4 vs a float64 HF VisionEncoderDecoderModel.

- [~] Stable Diffusion img2img + inpainting pipelines (VAE encoder
      BuildVaeEncoderFromSafeTensors, examples/ImageToImage SDEdit loop and blended
      SDUNetDenoiseInpaint / examples/DiffusionInpainting are in place).
      STILL OPEN: (1) the non-deterministic VAE reparameterize head
      (mean+exp(0.5*logvar)*noise); (2) the 9-channel inpaint-specialized UNet
      (conv_in widened to 4 latent + 1 mask + 4 masked-latent) importer support;
      (3) a fixed-seed img2img latent trajectory matching diffusers
      `StableDiffusionImg2ImgPipeline` step-for-step.
- [ ] Kosmos-2 grounded vision-language importer (BuildKosmos2FromSafeTensors[Ex],
      model_type "kosmos-2", e.g. microsoft/kosmos-2-patch14-224). DISTINCT from the
      landed caption VLMs (LLaVA/Blip2/Florence-2 fuse image features as prepended
      tokens for free-form text): Kosmos-2 emits GROUNDED text where phrases are
      tied to image regions via special `<grounding>`/`<phrase>`/`<object>` tokens
      whose `<patch_index_NNN>` ids decode to bounding boxes over the patch grid.
      It pairs a CLIP ViT image encoder (landed BuildClipVisionTower) with a Magneto
      transformer decoder (a Llama-ish stack with a sub-LayerNorm / scaled residual
      variant) and an image-to-text projection. New code: the Magneto decoder block
      (the extra normalization placement vs the landed Llama backbone) and the
      grounding token-id <-> patch-index <-> bbox decode helper + an
      examples/Kosmos2Ground demo that prints phrase->box pairs. Real value: first
      grounded-captioning / phrase-localization importer, a different capability
      from the landed open-vocabulary DETECTION and free-form caption paths.
      Pico-fixture parity < 1e-4 vs a float64 HF Kosmos2ForConditionalGeneration.
- [ ] SmolVLM2 image-AND-video VLM importer (`BuildSmolVLM2FromSafeTensors[Ex]`,
      model_type "smolvlm"/Idefics3-family, e.g. HuggingFaceTB/SmolVLM2-2.2B-Instruct
      and the 256M/500M tiers). Pairs a SigLIP vision tower (landed
      `BuildClipVisionTower`/SigLIP path) with a **pixel-shuffle space-to-depth
      connector** (reuse the landed `TNNetSpaceToDepth`/`TNNetPixelShuffle` to fold a
      `scale_factor²` spatial block into the channel axis, shrinking the per-tile
      token count) feeding a SmolLM2 (Llama-path) text decoder. New code is the
      image-splitting / tiling preprocessor and the **multi-frame VIDEO token
      packing** (sample N frames, encode each through the shared vision tower, and
      interleave the per-frame token grids with frame/`<row>_<col>` separators) —
      the rest is wiring landed layers. Pico fixture + `TestSmolVLM2LogitParity`
      < 1e-4 vs a float64 HF `SmolVLMForConditionalGeneration`, plus an
      examples/SmolVLM2Describe demo (one image, then a short clip). Real value:
      the landed caption VLMs (LLaVA/Blip2/Florence-2/PaliGemma) are image-only;
      this is the efficient pixel-shuffle-connector + native short-VIDEO path.
- [ ] RegNet importer follow-ups (`BuildRegNetFromSafeTensors[Ex]` LANDED,
      transformers model_type `regnet`, e.g. facebook/regnet-y-040 / regnet-x-040;
      conv+BN fold at load, grouped 3x3 via per-group SplitChannels/DeepConcat,
      regnet_y SE gate, NO new leaf layers; pico parity
      `TestRegNet{Config,XImageClassification,YImageClassification}Parity` < 1e-4
      vs a REAL float64 HF RegNetForImageClassification):
  - [ ] real-checkpoint top-1 sweep (facebook/regnet-y-040 / regnet-x-040 needs
        the HF image-processor preprocessing path).
  - [ ] timm naming variant.
  - [ ] downsample-in-first-stage=true real configs (code handles it, untested
        on a real ckpt).
- [ ] Qwen-Image-2.1 text-to-image and image editing (Qwen/Qwen-Image-2.1, diffusers
      `QwenImage21Pipeline`, Qwen Research License). Target: Phase A (text-to-image)
      first, 1024x1024 on the OpenCL GPU box, branch a19; B and C are recorded, not
      scheduled. One Opus agent per task, run serially, test + commit after each.
      Reference: diffusers `transformer_qwenimage21.py`, `autoencoder_kl_qwenimage21.py`,
      `pipelines/qwenimage21/pipeline_qwenimage21.py` (merged 2026-09-18).
      Checkpoint facts (from the configs and safetensors headers):
      - text encoder `text_encoder/`: Qwen3-VL-8B (`qwen3_vl`). Text part = Qwen3 dense
        decoder, 36 layers, hidden 4096, 32 q / 8 kv heads, head_dim 128, per-head QK
        RMSNorm, rope_theta 5e6, interleaved M-RoPE [24,20,20]. For a text-only prompt
        the three M-RoPE positions are equal, so it reduces to plain 1D RoPE. The
        transformer reads the LAST decoder layer's output BEFORE the final RMSNorm
        (diffusers hooks the norm away), minus the system-prompt tokens (drop_idx =
        tokenized length of the system message). Prompt template:
        `<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n
        <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.
      - transformer `transformer/`: 7B, BF16, 2 shards, NO biases anywhere. 32
        SINGLE-stream blocks, dim 4096, 32 heads x 128, SwiGLU 12288 (`gate_layer`,
        `proj`, `out`). Block: non-affine LayerNorm -> x(1+scale) -> attention (QK
        RMSNorm, 3-axis RoPE) -> residual with tanh(gate); same again for the MLP.
        ONE shared `modulation` Linear 4096->16384 (scale1, gate1, scale2, gate2) feeds
        every block. `txt_in` = zero-centred RMSNorm (stored weight-1) -> Linear ->
        GELU-tanh -> Linear. `img_in` Linear 64->4096. `norm_out` = non-affine LayerNorm
        x (1+scale), no shift. `proj_out` 4096->64. Timestep embedding: t in [0,1] x
        1000, cos half FIRST then sin, dim 256 -> Linear -> SiLU -> Linear.
      - attention mask: block-causal, `(q >= kv) or same_image_block`. With
        `causal_condition`, text and condition-image tokens take modulation from t=0,
        so their K/V are step-independent: the pipeline computes them once (extract)
        and later steps run only the target-image tokens against the cached prefix
        (cached). RoPE: axes (frame, h, w) = (16, 56, 56) dims, theta 10000,
        consecutive-pair (complex) layout; text tokens get position p on all 3 axes;
        an image block gets frame = position after the preceding text and an (h, w)
        grid centred on zero (negative indices allowed).
      - VAE `vae/` (`AutoencoderKLQwenImage21`, F32): 2D for images (all conv weights
        [out,in,3,3]) despite the Wan-style names. RGBA in/out, z_dim 64, 16x spatial.
        Encoder base 96, decoder base 144 (up to 1152 channels). Channel RMSNorm
        (F.normalize x sqrt(C) x gamma), SiLU, residual AvgDown/DupUp shortcuts,
        nearest-exact upsample + conv, single-head mid-block attention. Latents are
        de-normalised with 64 `latents_mean` / `latents_std` values before decode.
      - scheduler: `FlowMatchEulerDiscreteScheduler`, sigmas = linspace(1, 1/N, N),
        exponential dynamic shift, mu linear in image token count (256 -> 0.5,
        8192 -> 0.9), `shift_terminal` 0.02; x += (sigma_next - sigma) * v. Guidance
        `true_cfg_scale` defaults to 1.0 (off): one transformer pass per step. Default
        40 steps.
      - cost: 1024x1024 = 4096 image tokens, ~66 TFLOP per step, ~2.6 PFLOP for 40
        steps (2048x2048 ~6x more). Compute-bound GEMM, not decode GEMV. int8:
        transformer ~7 GB, text encoder ~8 GB, loaded one after the other.
  Phase A — text-to-image, correctness first:
  - [ ] A9 follow-up: OpenCL errors only print (`FErrorProc`); nothing lets the pipeline
        detect a failed allocation, so a REPL image can be wrong yet
        reported as written. Needs a counting error hook on the armed nets.
  - [ ] A9 follow-up: `--steps 1` raises EInvalidOp in
        `TNNetFlowMatchEulerScheduler.SetSigmas` (shift_terminal stretch
        divides 0/0; diffusers yields NaN). One-shot too.
  Phase B — speed (after a first real measurement on the GPU box):
  - [ ] B1. OpenCL for the transformer step pass. Int4 is required; int8 comes along
        with it (the tiled OpenCL kernels share one layout). The text encoder and
        the VAE stay on the CPU in this task. Target GPUs have 24-100 GB of memory.
        Runs BEFORE B0 and A9 (user decision, 2026-09-25).
        Already in the library: `cai_dot_product_int4_tiled` / `_int8_tiled`
        (many-token projections, used by --prefill-window); resident-codes sharing
        (`EnableOpenCLInContextOf`, `BorrowOwnerOpenCLCodes`); OpenCL paths for
        TokenLayerNorm, ChannelMulByLayer, HeadRMSNorm (via TokenRMSNorm), axial
        RoPE (via M-RoPE), DeepConcat, Sum, SwiGLU.
        Estimate (not measured): 1024x1024 step ~57 TFLOP of projections + ~18
        TFLOP of attention, limited by arithmetic, so int4 saves GPU memory
        (~4.4 GB of weights vs ~7 GB int8), not time.
        First GPU run (user, 2026-09-30, NVIDIA L4, tip dd2c2971, before B1d):
        256x256 --int4, 40 steps: 2.2 s/step (CPU int4 16.4 s/step), denoise
        91.7 s; load transformer 240.3 s (CPU int4 load 148 s); load text
        encoder 39.8 s; VAE decode 33.3 s (CPU); peak RSS 8.8 GB. 2.2 s/step is
        ~1.6 TFLOPS of projections, far below the L4, so host layers,
        transfers and synchronisation dominate at 256 tokens.
        Second GPU run (user, 2026-09-30, L4): 1024x1024 --int4, 10 steps:
        36.3 s/step (16.5x the 256x256 step for 16x the tokens: linear, so
        attention does not dominate; ~75 TFLOP/step, ~2 TFLOPS achieved),
        denoise 370.4 s, VAE decode 528.2 s (CPU, now the largest cost), load
        transformer 245.2 s, peak RSS 10.9 GB. B1f runs before B1d2 to tell
        the int4 projection kernel, the host layers and the transfers apart.
    - [ ] B1d1 follow-up: Non-blocking row write + a queue Finish at the end of
          TNNet.Compute when the last layer ends on the host (worth it
          now that B1d2 has landed: these writes are the only mid-block waits).
    - [ ] B1d1 follow-up: Shared "bind or upload the operand" helper for
          TNNetChannelMulByLayer and TNNetCellMulByCell.
    - [ ] B1d1 follow-up: SE / CBAM / DiT importers now take this OpenCL path with no
          OpenCL parity test.
    - [ ] B1d2 follow-up: Shared parent class for the four norm layers' identical
          FTokenNormCL / EnableOpenCL / WillOpenCL / output overrides.
    - [ ] B1d2 follow-up: Forced-OpenCL TRAINING through TokenLayerNorm/TokenRMSNorm
          never fills FNormalized/FInvStd (FInvRMS): backprop reads
          stale values (pre-existing).
    - [ ] B1d3 follow-up: Qwen2-VL/Qwen3-VL decode: M-RoPE is per head there and the
          positions change every token, so the blocking table write
          waits ~900 times per token (no worse than before; a
          non-blocking write would be safe in current graphs).
      Risks from B1a (not measured): the tiled int4/int8 OpenCL kernel streams
      the whole weight matrix once per 16-token tile (~35 GB of int4 codes per
      block at 4096 tokens unless L2 absorbs it; a wider token tile is the likely
      fix); if a device rejects the tiled kernel, int4 falls back to split-K
      (correct, very slow); kernel int32 indices overflow past ~87k tokens; the
      GateUp result buffer (403 MB at 4096 tokens) can hit
      CL_DEVICE_MAX_MEM_ALLOC_SIZE at larger images.
  - [ ] B0. CPU many-token int8/int4 projections. Evidence: at 256x256 a step is
        34.3 s int8 and 16.4 s int4 (~2.1x, the weight-byte ratio), i.e. ~105 GFLOPS
        and ~53 GB/s: the projections stream all ~7 GB of block weights once per
        TOKEN instead of reusing each weight row across the 256 tokens. Step 1
        (read-only): which kernel a many-token `TNNetPointwiseConvLinear` int8/int4
        forward runs in inference-only low-memory mode, and whether the
        `--prefill-window` batched path (27B TTFT 341 -> 53.9 s) can be reused; report
        + proposed fix. Step 2: the fix, after approval. At 1024x1024 (16x the tokens)
        today's rate is ~9 min per step.
  - [ ] B2. VAE decode speed: 3x3 convolutions at 1152 channels.
        Estimate (not measured): a CPU VAE decode at 1024x1024 is ~8 min (16x the
        31 s measured at 256x256), so it is the next bottleneck.
    - [ ] B2b follow-up: Drop the armed nets' host FConcatedWeights/FConcatedWInter
          copies (~2 GB) after the upload.
    - [ ] B2b follow-up: Parallel scheduler + OpenCL on PoCL is 20-90x slower than
          --serial (CPU oversubscription with PoCL's threads; worker
          count 4 -> 2 gives 32 s -> 3.3 s). Measure --serial vs default
          on the L4; cap the workers for CPU OpenCL devices.
    - [ ] B2f. Big tiles / untiled decode. Plan (2026-10-03, computed from the
          code and the real config, not measured; dims 1152/1152/1152/576/288/
          144, 1.06 TFLOP per 256-px tile; 1024x1024 = 29.2 TFLOP at 256/192,
          26.5 at 512/384, 17.0 untiled). OpenCL bytes per sized net at a
          256 tile / 512 / untiled 1024: today 10.4 / 38.5 / 151 GB (im2col
          6.1 GB at 256); after B2f1 4.2 / 13.6 / 51.4; after B2f2 3.5 / 11.0 /
          41.0; after B2f3 ~1.3 / ~1.9 / ~4.7. Host today 9.9 + 2.0 GB at 256.
          Corrections to earlier notes: the 1 GB FP32 upload is inside the net
          (first forward) and small; blending is milliseconds; the ~22 s outside
          the nets is most likely host-side arming per tile shape
          (`TNNetLayerConcatedWeights.EnableOpenCL` calls `AfterWeightUpdate`
          twice: 1 GB concat + scalar transpose + output-sized bias fill per
          call) plus ~10 GB of zero-filled host volumes per build. Untiled 1024
          also overflows on the host (`TVolume.ReSize` integer size;
          `FInputPrepared` 2.72e9 elements). In the REPL the transformer stays
          resident, so the VAE has ~10-14 GB on a 24 GB L4.
          - [ ] B2f5. Nearest-2x + 3x3 conv fold as four 2x2 phase convs on
                OpenCL (-17% of decode FLOPs); CPU path unchanged.
          - [ ] B2f6. Untiled path: memory estimator vs the device limits,
                `--vae-tile 0`; L4 runs 256/192 -> 512/384 -> untiled in the REPL;
                then the default decision with the user.
          Side note: `FInputBufferBs` (READ_ONLY) is written by cai_im2col and
          `FResultBuffer` (WRITE_ONLY) is read by consumer kernels; works on
          NVIDIA/PoCL, undefined per spec; pool buffers must be READ_WRITE.
  - [ ] B3. Optional guidance (negative prompt) with its own prefix cache.
  Phase C — editing and reference images (plan from source reading of
  diffusers 0.41.0.dev0 `pipeline_qwenimage21.py` / `transformer_qwenimage21.py`
  / `autoencoder_kl_qwenimage21.py` and transformers 5.17.0 `modeling_qwen3_vl.py`;
  order C2 -> C2b -> C1a -> C1b -> C3a -> C3b, C4 decided after C3a). diffusers has
  one pipeline for text-to-image and editing (`image=`): no strength, no mask, no
  second-image input; masks and circles are extra input images. Condition images
  are resized with our own resize, not PIL Lanczos (accepted divergence).
  - [ ] C4. Prefix pass on OpenCL, int8 prefix K/V and a windowed text-encoder
        prefill. Estimates at 1024^2: prefix ~4.1k tokens / ~4.3 GB FP32 K/V for one
        reference image, ~41k tokens / ~43 GB for ten; text-encoder scores ~13.6 GB
        unwindowed at ten. Decide scope after C3a with an L4 measurement.
- [~] Qwen-Image inpainting: `--init-image FILE --mask FILE` in examples/QwenImage
      (and `/mask FILE|off` in the REPL) regenerates only the white region of a
      mask image and keeps the rest of the init image. Port of diffusers
      `QwenImageInpaintPipeline` (`pipelines/qwenimage/pipeline_qwenimage_inpaint.py`
      in the ~/x venv), the no-retrain blended mode; the img2img path it extends
      is already in `TQwenImage21Pipeline` (`EncodeImage`, `Img2ImgStartStep`,
      `ScaleNoise` in `DenoiseAt`).
      Status (2026-10-20): CODED and parity-tested on the pico checkpoint.
      `QwenImage21MaskToLatents` / `TQwenImage21Pipeline.EncodeMask`, a
      `MaskLatents` argument on `Denoise` / `GenerateFromEmbeds`, the shared
      `InpaintBlend` (neuraldiffusion.pas, also used by `SDUNetDenoiseInpaint`),
      `--mask` / `/mask` and the README section. Tests
      `TestQwenImage21InpaintMaskLatents`, `...InpaintKeepsKnown`,
      `...InpaintParity` (oracle `tiny_qwenimage21_inpaint_io.json` from
      `tools/make_pico_qwenimage21_img2img_fixture.py --inpaint-only`).
      Note: 2.1 latents are unpatched (VAE factor 16,
      diffusers 2.1 `_pack_latents` is a plain flatten), so there is no 2x2
      packing; each token's channels all take that token's mask value.
      - [ ] Real 1024x1024 inpaint on the GPU box (user run), then the README
            example output.
      - [ ] `--mask-blur N` (diffusers `mask_processor.blur`).
      - [ ] `padding_mask_crop`: crop to the mask's bounding box plus padding,
            generate at full resolution, paste back (`apply_overlay`); makes a
            small hole in a large photo cheap.
- [ ] Image-generator LoRA adapters: `--lora FILE[:SCALE]` (repeatable) in
      examples/QwenImage, merged into the transformer weights at load. Today
      `TNNetLoRAMergeReader` merges only HF PEFT adapters into the Llama-builder
      text models (`pLoRAAdapterDir` of `BuildFromPretrained`); the image side has
      no adapter path at all, yet community LoRAs (styles, characters, and
      step-distilled "Lightning"-type adapters that cut 40 steps to 4-8) are how
      people use diffusion checkpoints. The transformer reads every tensor through
      one `TNNetSafeTensorsReader` in `TQwenImage21Transformer.Create`, so wrapping
      that reader is the whole hook; int8/int4 quantization then sees the merged
      weights, as on the LLM side. What `TNNetLoRAMergeReader` needs (extend it,
      do not fork a second reader):
      - Key prefixes besides `base_model.model.`: diffusers saves
        `transformer.<m>.lora_A.weight`, ComfyUI-style files use
        `diffusion_model.<m>`; replace the text-decoder prefix probe with a
        caller-supplied list of strip prefixes.
      - The kohya layout: `lora_unet_<m with dots as underscores>.lora_down.weight`
        / `.lora_up.weight` plus a per-module `.alpha` scalar (scale = alpha /
        rank, no adapter_config.json). Resolve the flattened name against the
        base reader's tensor names (dots -> underscores index built once).
      - Per-adapter user scale multiplied into `Scale`; several adapters stack by
        chaining merge readers (each one is a `TNNetSafeTensorsReader` base).
      - Fail loudly on an adapter key with no base tensor (wrong model family)
        and on LoRA-on-conv tensors the reader cannot merge yet.
      Tests: a pico adapter on `tests/fixtures/tiny_qwenimage21` in both the
      diffusers and the kohya key layouts, oracle from the ~/x venv
      (`pipe.load_lora_weights` + `fuse_lora`, latent after one step), the test
      itself Pascal; plus an unchanged-tensor pass-through check. README section in
      examples/QwenImage. Before promising a step cut, check that a step-distilled
      LoRA exists for Qwen-Image-2.1 (the known Lightning adapters target earlier
      Qwen-Image releases). The SD UNet path (`BuildSDUNetFromSafeTensors`) can
      take the same reader once the real Stable Diffusion pipeline item above lands.
- [ ] Jev-style typed decision models ("System One": a state plus typed questions in,
      calibrated option probabilities out, no text generated). Plan agreed 2026-10-18,
      branch a19. Scope: stage 0 (shared decision layer + `/v1/systemone`), stage 1
      (AnyJev L0 readout on any loaded decoder), stage 2 (Kev and Open-Jev: Qwen3.5 +
      LoRA + a small head). Laya (ModernBERT encoder + decision head) is LATER, not in
      this plan. One Opus agent per task, run serially, test + commit after each.
      Decisions:
      - Wire format = TypeSafe's `POST /v1/systemone` (OpenAPI 0.2.0 at
        api.typesafe.ai/openapi.json). Request `{model, state, questions:{id:{type,
        instructions, criteria}}}`; `state`/`instructions`/descriptions may be a
        string, an object or an array. `choice.criteria` = name -> description map
        (1..255, key order = option order, null description allowed);
        `score.criteria` = ordered array of level descriptions (1..10, levels 0..n-1);
        `noul.criteria` = optional `{true, false}`. Response `{model, answers, usage:
        {input_tokens, output_tokens}}`; choice `{type, choice, probabilities (by
        name), confidence}`; score `{type, score (= sum i*p_i), legend, probabilities
        (keys "0".."n-1"), confidence}`; noul `{type, noul}` only. Validation errors
        = HTTP 422 `{"detail":[{loc, msg, type}]}`; other errors `{"detail": str}`.
        `box` questions and image `assets` are rejected (422).
      - Confidence = Jev's published formulas on EVERY scorer, computed from the
        probabilities only: choice `(p_max-1/n)/(1-1/n)`; score
        `max(0, 1 - sum p_i|i-mode| / D)`, `D = (1/n) sum |i-(n-1)/2|`, mode = first
        argmax; both 1.0 when n = 1; all-zero p -> uniform. Noul has no confidence.
        Checks: score (0,0.57,0.43) -> 0.355; (0,0.95,0.05) -> 0.925.
      - Rendering of object/array values (state, instructions, descriptions): Kev's
        `api.render` (str/number/bool as text, list -> "- item" lines, dict ->
        "key: value" lines, nested values indented). One renderer for all scorers.
      - Special-token strings in caller text: rewrite `<|x|>` -> `<¦x¦>` before
        tokenizing (Kev's rule), on every scorer. A general "specials as plain text"
        tokenizer flag is a separate follow-up.
      - ChatServer always registers `/v1/systemone`. Without `--decision-adapter` it
        answers with AnyJev L0 on the loaded chat model (nothing extra is loaded).
        With `--decision-adapter <dir>` the server merges the adapter, reads the head
        and becomes decision-only: the chat endpoints return an error naming the
        reason. ChatTerminal does not change.
      - LoRA is merged in HF TENSOR space before the build (base + scale*B@A in FP32,
        rounded once), so the importer's row permutations (rotate_half q/k, the
        Qwen3.5 q/gate split, fused gate+up, fused z/b/a) apply unchanged and
        int8/int4 quantization follows as usual. No live (unmerged) LoRA path.
      - `head.pt` is read by `neuraltorchbin.pas` (extended for top-level scalar /
        string metadata). No Python conversion step for users.
      - Multi-question and multi-candidate scoring: prefill the shared prefix once,
        then per branch `TruncateTo(PrefixLen)` + `RestoreStateFrom(checkpoint)`,
        then feed the branch. Exact on attention-only and GatedDeltaNet hybrids; no
        new attention masks, no per-token RoPE positions.
      - AnyJev label prior: the content-free prior (states "N/A", "" and "[MASK]",
        alpha 1.0) is the default. The running-mean batch prior (stateful across
        requests) is a follow-up. AnyJev L1/L2 are out of scope (removed upstream in
        0.3.0, and both need labelled data).
      - GPT-AGI/OpenJev is not ported: AnyJev L0 covers it, and its first-token-only
        label scoring is a correctness bug.
      - Tests: pico fixtures with an HF/PyTorch float64 oracle made by
        `tools/*_fixture.py` scripts in the `~/x` venv under the 2 GB cap, plus
        model-free unit tests. Real-checkpoint checks run on the user's box.
      Reference facts (research 2026-10-18):
      - AnyJev L0 (`anyjev` 0.3.0 `readout.py`, `calibrate/permute.py`,
        `calibrate/contextual.py`): system prompt "You are a decision function. You
        will be given a state and one question. Reply with the answer label only: no
        words, no punctuation, no explanation."; user message lines "State:", state
        (or "(empty)"), "", "Question: <q>", then per kind: choice "Options:" + "A. opt"
        lines + "Answer with the letter only."; score "Pick the level that applies (the
        levels are ordered):" + "1. lvl" lines + "Answer with the number only."; noul
        "Answer Yes or No." / "Answer No or Yes.". Chat template with
        add_generation_prompt and thinking OFF; no template -> "{system}\n\n{user}\n
        Answer:". Encoded without special tokens. Labels: choice A..Z (max 26), score
        "1".."K" (letters if a digit is not one token), noul Yes/No. Each label = the
        first of `lab`, `" "+lab` that is ONE token, else error. Readout = full-vocab
        log_softmax at the last position, then softmax over the label logits.
        Rotations: choice cyclic shifts `perm[j] = (j+s) mod K`, noul both orders,
        score none. Per rotation p = p_raw / prior^alpha, clip 1e-8, renormalise; then
        map positions back to options and combine by log-mean (mean of log(clip
        1e-12), subtract max, exp, normalise). Content-free prior = mean over the 3
        empty-state probes per rotation, clip 1e-8, renormalise. Model-free oracle:
        raw (0.62,0.38), prior (0.70,0.30), alpha 1 -> 0.4115.
      - Kev (`jaredpalmer/kev`, HF `jaredpalmer/kev-0.8b|4b|9b|27b`): bases
        `Qwen/Qwen3.5-{0.8B,4B,9B}-Base` (frozen, LoRA r16 alpha32 on q/k/v/o,
        gate/up/down, in_proj_qkv/z/a/b, out_proj) and `Qwen/Qwen3.8-27B` (27B = full
        fine-tune, `Qwen3_5TextModel` keys without `model.` prefix, no lm_head).
        Adapter keys `base_model.model.layers.N.<module>.lora_{A,B}.weight`, F32.
        Tokens reused: state `<|fim_prefix|>`, question `<|fim_middle|>`, option open
        `<|box_start|>`, option close `<|box_end|>`, decide `<|fim_suffix|>`. Row =
        `[<|fim_prefix|>] + state` then per question `[<|fim_middle|>] + instructions
        + (<|box_start|> opt <|box_end|>)* + [<|fim_suffix|>]`; no BOS, no chat
        template. Options: noul `["no[: false desc]", "yes[: true desc]"]` -> answer
        p[1]; choice "name" or "name: desc"; score rendered level text. Head reads the
        FINAL hidden state (after the final RMSNorm): `z = (K h_</opt>) . (Q h_decide)
        / sqrt(256) / T`, Q and K = Linear(d -> 256) with bias. `head.pt` = torch.save
        dict: `head` (OrderedDict q.weight/q.bias/k.weight/k.bias, fp32), `temperature`
        (0.8B 2.3510958), `head_dim` 256, `base`, `base_revision`, `lora`, plus nested
        training metadata. State limit 65536 tokens (422 above it).
      - Open-Jev (`Zefan-Cai/Open-Jev`, MIT code; HF `ZefanCai/Open-Jev-2B|9B|27B-v1.1`,
        Apache-2.0): bases `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.8-27B`
        (post-trained). Package `package/checkpoint/{adapter/, head.pt, model.json,
        temperature.json}`. LoRA r8 alpha16 on q/k/v/o, in_proj_qkv, out_proj, same key
        naming as Kev. Head = plain state_dict `weight [1,d]`, `bias [1]` (storage
        location may be 'cuda:0'). One chat-template prompt per candidate (user turn
        only, thinking off): "Context:\n{state}\n\nQuestion: {q}\n" + "Proposed
        answer: {option}\nIs this proposed answer correct? Answer Yes or No."; noul =
        ONE prompt "...Is the answer to this question yes? Answer Yes or No." with
        "\nYes means: ..."/"\nNo means: ..." appended to the question. Score s = head(
        final hidden state of the last token); choice/score p = softmax(s/T), noul
        p = sigmoid(s/T). Non-string values rendered with json.dumps(sort_keys).
  Stage 1 — AnyJev L0 readout (no trained weights):
  - [ ] D1.3 user check of the coded AnyJev L0 default scorer on a real checkpoint (this VM cannot hold one):
        ```
        git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct q25
        lazbuild -B neural-api/examples/ChatTerminal/ChatServer.lpi
        neural-api/bin/x86_64-linux/bin/ChatServer q25/ --fp32 --ctx 4096 --port 8080 &
        pip install anyjev==0.3.0
        pip install --no-deps git+https://github.com/jaredpalmer/kev@1f04cfa28e6a708485b28a9585010b9e436b776e  # kev pins torch<2.9
        python3 neural-api/tools/anyjev_real_check.py q25/ --server http://127.0.0.1:8080
        kill %1
        ```
        PASS = max abs difference <= 5e-3 (exit 0). `--fp32` is required: int8
        weights move the probabilities far more than FP32 rounding.
  Stage 2 — Kev and Open-Jev:
  - [ ] D2.7 User-box real checks: Kev-0.8B (Kev's own test record) and
        Open-Jev-2B against their Python servers; commands written by the agents.
        Commands (ChatServer.md *Trained decision checkpoints*); PASS = max abs
        difference <= 5e-3 (exit 0); `--fp32` is required:
        ```
        git clone https://huggingface.co/Qwen/Qwen3.5-0.8B-Base q08b
        git -C q08b checkout dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
        git clone https://huggingface.co/jaredpalmer/kev-0.8b
        lazbuild -B neural-api/examples/ChatTerminal/ChatServer.lpi
        neural-api/bin/x86_64-linux/bin/ChatServer q08b/ --decision-adapter kev-0.8b/ --fp32 --ctx 4096 --port 8080 &
        pip install --no-deps git+https://github.com/jaredpalmer/kev@1f04cfa28e6a708485b28a9585010b9e436b776e
        python3 neural-api/tools/decision_real_check.py q08b/ kev-0.8b/ --server http://127.0.0.1:8080
        kill %1
        ```
        ```
        git clone https://huggingface.co/Qwen/Qwen3.5-2B q2b
        git -C q2b checkout 15852e8c16360a2fea060d615a32b45270f8a8fc
        git clone https://huggingface.co/ZefanCai/Open-Jev-2B
        neural-api/bin/x86_64-linux/bin/ChatServer q2b/ --decision-adapter Open-Jev-2B/ --fp32 --ctx 4096 --port 8080 &
        pip install --no-deps git+https://github.com/Zefan-Cai/Open-Jev@bd4118882f733574a3250a4b65fe4d884130c08b
        python3 neural-api/tools/decision_real_check.py q2b/ Open-Jev-2B/ --server http://127.0.0.1:8080
        kill %1
        ```
        Kev-27B (int8 on the GPU against kev in bfloat16, ~55 GB host RAM; a
        sanity bound, not FP32 parity: read the table, same argmax expected;
        add `--int4` to the server when the int8 weights do not fit):
        ```
        git clone https://huggingface.co/jaredpalmer/kev-27b
        neural-api/bin/x86_64-linux/bin/ChatServer kev-27b/ --decision-adapter kev-27b/ --gpu --ctx 4096 --port 8080 &
        python3 neural-api/tools/decision_real_check.py kev-27b/ kev-27b/ --dtype bfloat16 --threshold 5e-2 --server http://127.0.0.1:8080
        kill %1
        ```
        - Open-Jev-2B: sanity check done on the user box 2026-10-19 (`--gpu --int4`, now SUPPORTED); its `--fp32` parity run above is still open.
        - [ ] Kev-0.8B (and the other Kev sizes, Kev-27B above).
        - [ ] AnyJev L0: ran on a 0.5B chat checkpoint, but the yes/no answer was
              questionable; stays planned (coded).
  Follow-ups (not in this plan): Laya; AnyJev batch prior, L1, L2; Kev
  `KEV_DATE_FACTS`; `/v1/systemone/permute` and `/separate`; per-question
  `TruncateTo` downloads the resident KV cache on OpenCL (measure first); tokenizer
  "specials as plain text" flag for chat requests; live (unmerged) LoRA.
- [ ] Flow-matching sampler clean-ups surfaced by `TNNetFlowMatchEulerScheduler`
      (fd99162f):
  - [ ] `examples/FlowMatching/FlowMatching.lpr` (~296-312): optionally drive the
        loop with `TNNetFlowMatchEulerScheduler`. The example integrates t from 0
        (noise) to 1 (data) with dt = 1/N and no shift, so it needs time = 1 - sigma
        and a reversed velocity sign; the gain is small (the loop is already one
        `MulAdd` per step).

## OpenCL forward coverage — layers still on the host

Forward pass only; a device backward is a separate project (nothing in the tree
has one today). 52 of the 487 `TNNet*` classes in `neuralnetwork.pas` declare a
`ComputeOpenCL`, and `cai_activation` covers 72 elementwise opcodes on top of
that. `csActivationOpenCLMinSize` is 64 MB, so a small layer reaches the device
only when its source is ALREADY resident — which is why the ranking below is by
"does this layer evict a resident chain to RAM", not by FLOPs.

Every item: guard behind the existing `WillOpenCL()`/`FShouldOpenCL` verdict,
bind the resident source instead of downloading it (`ShouldBindPrevOutputOnOpenCL`)
and leave the result in OpenCL memory, keep the host path as the fallback, and
pin an exact-vs-CPU parity test that skips cleanly with no device. PoCL on the
dev box proves parity; throughput numbers have to come from the GPU box.

### Residency breakers on the decode path (small kernels, largest payoff)

- [ ] `TNNetMoEExpertBankGateUp` single-launch device forward. The clearest
      single gap: `TNNetMoEExpertBankDown` got one (commit d105047c) but its
      sibling did not, so every MoE block downloads the expert activations
      between the two banks. Mirror the down-bank kernel (one launch over the
      selected experts, resident int8 or FP32 weight slab) so gate/up -> down
      stays device-side end to end. Hits Qwen3.5-MoE, Nemotron-H-MoE, Mixtral,
      OLMoE and GPT-OSS.
- [ ] `TNNetDeepConcat` with one host-only source falls back to the host and
      downloads every resident source. `TNNetConcat.ComputeOpenCL` uploads a
      host-only source straight into its slice of the output; DeepConcat's
      channel-interleaved layout needs an upload buffer per such source (or,
      when SizeX*SizeY = 1, the same flat `WriteBufferAt` into the output).
- [ ] `TNNetSum` repeats the `TNNetConcatOpenCLBase` output-buffer lifecycle
      (`FSumKernel`/`FSumBuffer`, release, adopt, share). Deriving it from that
      base with `ConcatKernelName = 'cai_volume_sum'` would delete the copy;
      the field names would then need a neutral rename (`FOutputKernel`/...).
- [ ] Follow-ups of the `cai_gather_copy` glue: `cai_upsample_gather`
      (DeMaxPool, Upsample, PixelShuffle) is expressible as a `cai_gather_copy`
      plan too (DeMaxPool's random spacing would need a per-forward plan, so
      it stays out). The gather-copy layers do not take part in OpenCL output
      sharing by liveness (`CanShareOpenCLOutput` is False).
- [ ] `TNNetTopKGate` / `TNNetBiasBalancedTopKGate` device forward. Tiny work
      per token, but the router sits between two resident MoE banks, so running
      it on the host costs two transfers per token. Only worth landing together
      with the gate/up bank item above.
- [ ] `--gpu-fp16`: narrow each resident FP32 source to half ONCE per forward,
      not once per consumer. `TDotProductSharedKernel.PrepareInt8BOperand` calls
      `CastBOperandToFP16` (`cai_f32_to_half`) into the layer's own
      `FInputBufferBsFP16` on every forward, so consumers sharing one source
      each re-cast it: in a Qwen3.5/3.8 block the attention norm feeds Q/K/V and
      the post-attention norm feeds gate/up, which is five casts (and five half
      buffers) for two distinct sources, up to ~337 extra launches per token on
      the 27B. The user's 27B A/B (2026-08-24) showed host enqueue is now the
      limit (+12% enqueue in FP16 mode, `TNNetPointwiseConvLinear` +31%), so
      launch count is what pays. Proposed: a net-level cache of the narrowed
      copy keyed by source `cl_mem` plus a version counter the producing layer
      bumps when it rewrites its output; consumers bind the cached half buffer
      instead of casting. Rejected alternative: producers emitting a half
      output directly (larger, and non-int8 consumers still need the FP32
      copy). Traps: the per-layer queues of `--no-gpu-shared-kernel` need an
      event or the shared queue between the cast and every reader; output
      sharing by liveness (`CanShareOpenCLOutput`) reuses a `cl_mem` across
      producers, so the key must not be the buffer alone. Test: PoCL parity of
      a two-consumer net with the cache on vs off, plus a launch-count check;
      the throughput number comes from the GPU box.

### Real device-side work still running on the host

- [ ] `TNNetSelectiveSSM` device forward. The Mamba-1/2, Jamba, Nemotron-H and
      falcon_mamba trunks are majority-SSM and run entirely on the CPU.
      `TNNetGatedDeltaNet` already proved the recipe (commit 63047d0f: one
      work-group per head, the whole scan in OpenCL memory, state carried on the
      device across decode steps); `TNNetSelectiveSSM` is the same shape. Cover
      both the prefill scan and the `TNNetRecurrentDecodeBase` one-token path.
- [ ] `TNNetWKV`, `TNNetTokenShift` and `TNNetDiagonalSSM` device forward — the
      RWKV trunk, same per-head-recurrence recipe as `TNNetSelectiveSSM`. Lower
      priority only because fewer tested checkpoints use it.
- [ ] Attention subclasses that override `Compute` and so lose
      `TNNetScaledDotProductAttention`'s device path:
      `TNNetGptOssSinkAttention` (GPT-OSS is a live importer, and the sink is
      `TNNetFusedSDPA` plus one per-head scalar, so it should reuse the fused
      decode kernels) and `TNNetDifferentialAttention` (`TNNetWindowAttention`
      is in the vision attention item below). Take the sink first.
- [ ] `TNNetDeltaNet`, `TNNetGatedLinearAttention`, `TNNetRetention` and the
      CAUSAL half of `TNNetLinearAttention`. COUPLED, not independent: each is a
      strict per-token scan today, so there is no GEMM-shaped kernel to offload
      until the chunked/parallel forward tasks in "Layer follow-ups that fix real
      limitations" land. Sequence them after those, or fold the device path into
      each chunked rewrite.
- [ ] `TNNetTokenAndPositionalEmbedding` device gather. `TNNetEmbedding` got the
      int8 device gather (`cai_embedding_gather_int8`) and leaves its rows
      resident; its sibling still gathers on the host. Reuse the same entry
      point, adding the positional table as a second resident source.


### Vision and audio layers with no device forward

Companion to the two groups above, same contract (forward only, bind the
resident source, leave the result resident, host fallback, exact-vs-CPU parity
test). The ranking differs though: a vision or audio forward runs over large
tensors, so here the FLOPs matter alongside the transfers, and a kernel can pay
for itself even when it sits between two host layers.

Derived by walking the class graph: a layer counts as covered if it declares
`ComputeOpenCL`, sets an `FActivationOpcode`, dispatches to OpenCL inside its
own `Compute`, or inherits one of those without overriding `Compute`. 310 of
the 487 classes fail that test; the ones below are the subset the vision and
audio builders actually instantiate. Note the third case: a class that
overrides `Compute` with no device branch silently loses its parent's kernel
(`TNNetMaxPoolPortable` lost `TNNetMaxPool`'s that way; fixed in commit abd639f0).

Training-only layers (`TNNetDropout`, `TNNetDropPath`) are out of scope.

- [X] `TNNetCausalConv1D` (the CogVideoX VAE decoder) device forward. Done on
      `cai_conv_implicit` with no kernel change: a K x 1 kernel, PaddingX =
      Dilation*(K-1), PaddingY 0, DilationX = Dilation and OutSizeX = SeqLen
      (the kernel's bounds check is the missing right pad). Covered by
      `TestCausalConv1DOpenCLParity` (5 shapes, tiled and untiled, a weight update).
- [X] `TNNetConvolution3D` (VideoMAE tubelet embedding, 3-D VAEs) device
      forward on `cai_conv_implicit`, no new kernel: output frame ot reads ONE
      contiguous depth window [ot*C, ot*C + FeatureSizeT*C) per source pixel,
      so `cai_conv_gather` gained a source pixel stride and an output-frame
      axis folded into the column (b = pixel*OutT + ot; the result
      [b*NumFeatures + f] is the packed output as it is). Covered by
      `TestConvolution3DOpenCLParity` (5 shapes incl. the tubelet stride).
- [ ] `TNNetModulatedConv2D` device forward keeps the host in the loop: the
      style source is downloaded, the w'' slab (Cout*Cin*K*K) is built and
      re-laid out k-major on the host and uploaded every forward before
      `cai_conv_implicit` runs. To keep the base kernel resident instead,
      fold s[i] into the gather and d[o] into the epilogue (a per-column and a
      per-row scale on `cai_conv_implicit`, with d[o] from a resident
      sum_t w[o,t,i]^2 table times s^2). Measure the upload share on the GPU
      box first; at 1024x1024 the contraction should dominate.
- [X] `TNNetWindowAttention` (Swin / MaxViT / BEiT / SwinIR / the CLAP audio
      tower) and `TNNetT5RelPosBiasAttention` device forward on
      `cai_sdpa_flash`: the kernel gained an additive score bias
      `FScoreBias[FBiasBase + t*FBiasRowStride + key]` (FGroupSize 1 only; a
      biased score <= -1e8 counts as masked, as on the host). Window attention
      passes its SeqLen x SeqLen bias matrix (row stride SeqLen), T5 a per-offset
      table of 2*SeqLen-1 bucket biases (row stride -1); both re-upload after
      `AfterWeightUpdate`. T5's cached decode stays on the host. Covered by
      `WindowAttentionOpenCLParity` (2-3 windows x 2-3 heads, -1e9 and -100
      shift masks, sentinels, a bias update) and `T5RelPosBiasAttentionOpenCLParity`
      (non-causal, causal, causal + window). The extra uniform branch per score
      in the decode flash kernel is not timed on a GPU yet.
- [ ] The rest of the vision attention subclasses that override `Compute` and
      so lose `TNNetScaledDotProductAttention`'s device path:
      - `TNNetSwinV2WindowAttention`: cosine attention, so Q and K rows must be
        L2-normalized before the dot and the scale is the per-head
        logit_scale, not 1/sqrt(d_k). Route: a row-normalize pre-pass into a
        packed [Qn|Kn|V] buffer (allocated in EnableOpenCL), then
        `ComputeFlashPacked` with InvSqrtDk = logit_scale and the bias matrix
        exactly as `TNNetWindowAttention` passes it.
      - `TNNetSAMVisionAttention` (SAM): not an SDPA subclass; it fuses the qkv
        projection, the padded window partition, the out projection and a
        QUERY-DEPENDENT decomposed rel-pos bias (Q . rel_pos_h + Q . rel_pos_w),
        so no static bias table fits. Route: the projections on the existing
        dot-product kernels, a small kernel writing the per-window bias into a
        SeqLen x SeqLen buffer each forward, then the flash bias argument.
      - `TNNetDotProducts`, the raw contraction behind the CAI self-attention
        builders.
      `TNNetSinkAttention` and `TNNetDifferentialAttention` are named in the
      decode-path group above; landing them there covers the vision builders
      that also use them.
- [ ] The long tail of one-model vision layers: `TNNetRoIAlign` (Mask R-CNN),
      `TNNetConvGRUCell` (RAFT), `TNNetSpectralConv2D` (FNO - coordinate with
      the open FFT denormal-trap item, since the device path would move that
      math off the host FPU entirely) and `TNNetLocalResponseNormDepth`
      (AlexNet, minor). Rank these by which importer gets used, not by size.
- [X] `TNNetGRN` device forward (ConvNeXt, RIFE, StyleGAN2, SAM), the last
      conv-stack member of the per-channel glue. It needs a per-channel
      spatial L2 reduction (Gx[c] = ||X[:,:,c]||) and the channel mean of Gx
      before an elementwise tail, so it is not a `cai_activation` opcode. First
      check whether `cai_group_norm`'s per-group statistics (groups = Depth)
      can supply the per-channel sum of squares.
      DONE: `cai_group_norm` cannot (it keeps its statistics in local memory and
      writes only the normalized output). A new one-work-group `cai_grn_table`
      writes the coef|beta table on `cai_activation`'s queue, and the existing
      ChannelAffine opcode 46 does the tail; inference only, resident in and out.
      Follow-up: the table kernel is one work-group, so on a big map (56x56x384)
      it runs on one compute unit; split positions across groups if it shows up.
- [X] `TNNetLearnedPositionalEmbedding` device forward: cai_activation's
      AddChannel opcode over a SizeX*Depth table (new virtual
      `ActivationChannelCount`), the table resident and re-uploaded on a weight
      change; a non-zero PositionOffset copies the used rows into the queue
      scratch buffer (`CopyBuffer` gains a source offset). Inference only.
      Covered by `TestLearnedPositionalEmbeddingOpenCLParity`.
- [X] `TNNetSoftPrompt` device forward: a `cai_gather_copy` plan writes the
      source rows after K zero rows, then one `CopyBuffer` lands the resident
      bank (re-uploaded on a weight change) in rows 0..K-1. Inference only.
      Covered by `TestSoftPromptOpenCLParity`.
- [X] `TNNetSinusoidalTimeEmbedding`: SKIPPED, no transfer to save. Its source
      is the 1x1x1 timestep the sampler writes on the host each step
      (`TimeIn.Output.FData[0] := t` in the diffusion examples), and its output
      is one EmbeddingSize vector. Host compute costs one upload of that vector
      to the next layer; a kernel would cost one upload of t plus a launch.
- [ ] `TNNetSincConv1D` and `TNNetTDNNConv1D` device forward — the pyannote
      segmentation frontend and the ECAPA-TDNN trunk. Both are convolutions
      whose kernel is generated (sinc band-pass) or dilated, so the existing
      conv offload applies once the kernel slab is materialized; the closest
      thing to a free win on the audio side.
- [X] `TNNetFlipX`, the one host hop inside an otherwise device-capable
      bidirectional stack. Done: its forward is a `cai_gather_copy` plan (the X
      axis table reversed), covered by `TestGatherCopyOpenCLParity` and the
      resident chain of `TestGatherCopyResidentOpenCLParity`.
- [X] `TNNetFlipY` and `TNNetReverseXY` on the same `cai_gather_copy` plan
      (reversed Y table / reversed position table), in both gather-copy tests.
- [X] `TNNetReverseChannels` and `TNNetRoll` on the same `cai_gather_copy`
      plan (reversed depth table / cyclic table on the rolled X, Y or depth
      axis), in both gather-copy tests.
- [ ] `TNNetAttentiveStatsPooling` (ECAPA-TDNN). Not a gather copy: it needs a
      new kernel (per-channel softmax over time, then the two weighted moments)
      reading two sources, the features and `FLogitLayer`. It runs once per
      utterance and reduces T x C to 2C, so its host hop is one T x C download per source.

The channel-major audio holders (`TEnCodecModel`, `TNNetHiFiGAN`, `TNNetVits`,
`TNNetMimi`, `TNNetDAC`, `TNNetDemucs`, `TNNetKokoro`) are NOT TNNet layers and
are deliberately absent from this list: they run conv1d directly on
channel-major arrays through `RunEnCodecConv` / `RunHiFiGANConv` /
`RunMimiConv` / `RunDACConv`. EnCodec and HiFiGAN already have the
`EnableConvOpenCL` GEMM route and Demucs and Kokoro inherit it by calling them;
Mimi and DAC are the remaining gap and are already tracked in the audio section
above (blocked on a Double-precision shared dot-product kernel).


### Activation coverage — the fused GEMM tail and the standalone kernel

Two separate mechanisms, easy to confuse. (1) The FUSED tail inside the matmul
kernels: `cai_dot_product` and its int8/split-K siblings apply `ActFN`
in-register after the bias add through `cai_fused_act`, which implements the
eight parameterless opcodes 1-8. `TNNetLayer.IsActivationFunctionInOpenCL` maps
five host activation functions onto them (ReLU, Sigmoid, Tanh, Swish,
HardSwish) and returns false for everything else. (2) The STANDALONE
`cai_activation` kernel, which has 72 opcodes, each wired to its layer classes
through `ComputeActivationOnOpenCL()`. Every elementwise `TNNetReLUBase`
activation now has one (`TNNetRReLU` reuses 15 = LeakyReLU, device path only
while `Enabled = false`, i.e. at its fixed inference slope).

Both are forward-only by construction: neither produces `FOutputRaw` or the
derivative mask, so `IsActivationFunctionInOpenCL` requires `not FIsTrainable`
and the standalone path requires it too. `csActivationOpenCLMinSize` (64 MB)
keeps the standalone kernel off the device unless the source is already
resident or `ForceOpenCL` is set — deliberate, since a 1-flop-per-word kernel
cannot pay for its own transfer.

- [X] `TNNetArcSinh` (host and opcode 56) used ln(x + sqrt(x^2 + 1)), which
      cancels in float32 for x << 0. Done: both now compute
      sign(x) * ln(|x| + sqrt(x^2 + 1)), exact to |x| = 2^125 and saturating past
      it; the host derivative no longer overflows x^2; pinned by
      `TestArcSinhLargeMagnitude`, and the 1e30 extreme-input parity now compares values.
- [ ] `TNNetMetaAconC`, the one per-channel learnable activation still with
      no device forward. Its beta is sigmoid(gamma[c] * mean_c(x) + delta[c]),
      so it needs a per-sample spatial channel mean before the elementwise
      ACON map; the map itself is opcode 52 (`TNNetAconC`) with a third table
      computed on the device. The other ten (PReLU, PReLUChannel, SReLU, APL,
      AconC, Spline, Swish/Mish/SoftPlusBeta learnable, Polynomial) run on
      opcodes 15, 40 and 48-55: the scalar ones pass their weight as a kernel
      argument, the per-channel ones read their neurons packed into one table
      (`TNNetChannelTransformBase.PackActivationChannelTables`).
- [ ] Extend the `cai_activation` trap audit to the other kernels. The
      kernels build with `-cl-fast-relaxed-math` and PoCL runs them with FP
      traps unmasked; `TestActivationOpenCLExtremeInputs` showed PoCL's
      `exp` trapping below ~-88 (not underflowing), `erf`/`sin`/`cos`/`log`
      trapping on huge arguments, both sides of data-dependent branches
      running, and `x * tanh` reassociated into an overflowing `x * (1 - e)`.
      `cai_activation`, `cai_fused_act` and `cai_glu_gate` now go through the
      `cai_exp_floored`/`cai_erf`/`cai_sin`/`cai_log`/`cai_tanh` helpers at the
      top of `neural.cl`. Still raw: `cai_gated_delta_net` (two-branch sigmoid
      and softplus on raw `exp`, `exp(-zv)` in the SiLU gate), `cai_softmax`
      (`exp` of an argument clamped only to +-4000), the SDPA row softmax
      `exp(row[r] - RowMax)` without the -80 floor its siblings use, and the
      RoPE `sin`/`cos` of an unclamped angle. Each needs an extreme-input
      device test before and after.

## Layer follow-ups that fix real limitations

- [~] Bidirectional + multi-layer stacking for `TNNetLSTMCell` / `TNNetGRUCell`
      (and a real `nn.LSTM`/`nn.GRU` num_layers>1 / bidirectional=True checkpoint
      importer). The single-cell `TNNetLSTMCell`/`TNNetGRUCell` (torch fused
      `weight_ih`/`weight_hh` gate layout, full forward + exact BPTT) have landed.
      The `AddBidirectionalLSTM`/`AddBidirectionalGRU` builders exist too.
  - [ ] Faithful STACKED-BIDIRECTIONAL torch import follow-up, GRU half and
        the multi-layer fixture: the LSTM half exists (`CellsReadInputWidth`
        builds projection-free `TNNetLSTMCell.Create(Hidden)` cells that
        `LoadTorchLSTMInto` loads), but `AddBidirectionalRecurrentStack` still
        inserts a learned `TNNetPointwiseConvLinear` projection on every
        incoming Depth≠Hidden feed for GRU cells, which torch nn.GRU does not
        have, so `LoadTorchGRUInto` rejects those stacks. Give `TNNetGRUCell`
        the same any-input-width form, then add a 2-layer bidirectional
        pico-oracle parity fixture for LSTM and GRU (the generator already
        supports the shape — `tools/torch_rnn_tiny_fixture.py`).

- [ ] OpenCL forward offload for `TNNetCausalLinearAttention` (the non-causal global
      `TNNetLinearAttention` is now DONE — `ComputeOpenCL` two-GEMM offload behind
      `FShouldOpenCL` + `LinearAttentionOpenCLParity` exact-vs-CPU test, PoCL-verified
      max|diff| ≈ 1.5e-8). The causal sibling was deliberately LEFT CPU-only: its
      forward is a left-to-right prefix-sum scan (`S_t = S_{t-1} + ϕ(K_t)⊗V_t`,
      per-query `Out_t = ϕ(Q_t)·S_t / ϕ(Q_t)·Z_t`), NOT a pair of dense GEMMs, so it
      does not map onto the `FDotCL` matmul kernel without a chunked-scan rewrite.
      Tackle it with the chunked-forward family below (intra-chunk dense GEMM +
      inter-chunk running state), then add a causal parity test alongside.

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
- [ ] Segment-mask MHA-builder wiring follow-up (the SDPA-layer + packer half
      landed above): thread an optional segment-id source through the
      multi-head attention BUILDERS (AddMultiHeadSelfAttention and friends) so
      packed-window training masks cross-document attention end-to-end, not just
      at the bare TNNetScaledDotProductAttention layer. Each per-head SDPA in the
      builder takes the same shared `pSegmentSource`; assert a packed two-doc MHA
      stack matches the concatenation of independent per-document MHA runs (the
      builder-level analogue of TestSegmentMaskMatchesUnpackedBaseline). KV-cache
      incremental decode stays intentionally unmasked (single-stream = one doc).

- [ ] Residual scalar backward loops left by the AVX/OpenCL vectorization batch
      (the forward + cleanly-mappable backward paths are done; these are the
      strided-on-one-operand remainders that a single DotProduct/MulAdd cannot cover
      without an extra gather):
  - [ ] TNNetTestTimeTraining / TNNetTitansMemory backward "undo" loops (interleaved
        scalar etaGrad/dEta/dTheta accumulation) — the per-token forward rank-1 writes
        are vectorized; this is the lower-value remainder.

## Tests / numerical-gradient audit

- [ ] Property-based gradient harness v0: randomize input shape (keeping
      layer type fixed) for the 6 most recently landed layers. Catches
      shape-edge bugs hand-written tests miss.
- [ ] Continue upsampling/deconvolution audit: TNNetDeconvolution input-
      gradient and full weight-gradient finite-difference checks (one-weight
      check in TestDeconvolutionGradient only; Upsample / DeMaxPool / DeAvgPool
      already covered). Likely benefits from the Double-precision accumulator
      of DeMaxPoolFamilyGradientCheck (tests/TestNeuralNumerical.pas).
- [ ] Recurrent-style layer audit: TNNetEmbedding's weight-gradient path
      (sparse-update pattern — easy place for a silent broadcast/reduction
      bug), TNNetTokenAndPositionalEmbedding, etc.
- [ ] Multi-branch layer-registry round-trip audit: extend the landed
      single-source registry round-trip (tests/TestNeuralRegistry.pas) with a
      multi-branch builder to cover the skipped two-source layers
      (Concat/Sum/CrossAttention/FiLM/HyperLinear/etc.).
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
      repeats. Capture as `AssertKinkDerivative(layer, x_kink, expected_dydx)`
      and use it first for the missing TNNetHardShrink / TNNetSoftShrink
      kink-region tests at hand-picked inputs (no central differences).
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
      The device forward (opcodes 22, 70) is already finite at any finite input
      (`TestActivationOpenCLExtremeInputs`); this item is the host path only.
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

## Vision / generative & accelerator batch 2026-06-27b

(Verified absent in source before listing. An NF4 dequant path did not exist
then — only int8 + MXFP4-dequant did.)

- [ ] NF4 (bitsandbytes 4-bit) dequantizing import path. PARTIALLY LANDED
      (commit 90b6c931): the `neural/neuralnf4.pas` dequant unit ships — the exact
      16-level NF4 codebook (`NF4Code`) + `DequantizeNF4(Codes, Absmax, N, Dest,
      BlockSize=64)` with HIGH-nibble-first packing and per-block FP32 absmax
      (`dequant[i] = NF4_CODE[nibble] * absmax[i div BlockSize]`); single-quant
      only. Cross-checked to <1e-5 against a pure-numpy reference
      (`tools/make_pico_nf4_fixture.py` → `tests/fixtures/pico_nf4.json`;
      `bitsandbytes` is NOT importable under `venv x` — CPU torch, no CUDA bnb
      build) via `TestNF4DequantFixtureParity` + `TestNF4DequantHandBlock`.
      OPEN FOLLOW-UPS:
  - [ ] Double-quantized absmax: handle `*.nested_absmax` / `*.nested_quant_map`
        / `*.quant_state.*` (the secondary int8 quant + nested absmax + offset).
        DEFERRED. `LoadNF4QuantizedTensorFlat` now RAISES a clear
        `EPretrainedImportError` on these nested/double-quant markers (and on a
        non-FP32 absmax sibling) rather than feeding quantized absmax into
        DequantizeNF4 — no silent garbage path. `TestNF4ImporterRejectsDoubleQuant`
        covers the rejection (fixture `pico_nf4_doublequant.safetensors`). To
        finish: expand the int8-quantized absmax (nested absmax + offset +
        quant_map) to FP32 upstream, then drop the guard. A real `bnb-4bit`
        Llama/Qwen slicer fixture is RAM/network-gated; today's parity is the
        random pico numpy oracle only.

## Lucky-day batch 2026-06-28f (follow-ups surfaced by 28e landings)

(All three 28e items LANDED: Heun sampler [891e7537], Min-SNR-gamma loss
weighting [f8e1823c], tiled VAE decode [2e28a44b]. Genuinely-novel follow-ups
surfaced while landing them:)

- [ ] Fully-convolutional VAE-decoder Compute path so `TiledVaeDecode`
      (neuralpretrained.pas) can reuse ONE decoder net across tiles instead of
      rebuilding a tile-sized decoder per tile-grid. Today `TNNet.Compute` errors
      on a latent whose size differs from the fixed `TNNetInput`, so the tiled
      path rebuilds the decoder at the tile latent size. A size-agnostic forward
      (re-derive layer shapes from the actual input volume on each Compute, or a
      Resize-input hook) would let a single resident net decode arbitrary tile
      sizes — also unblocks variable-resolution inference generally. Scope first:
      audit which layers in the VAE decoder graph hard-bind to TNNetInput size.
- [ ] Tiled VAE decode is an APPROXIMATION because the decoder MID block has
      GLOBAL self-attention (same caveat as diffusers): per-tile attention can't
      see across seams. Follow-up: a windowed/overlapped-attention tiled mode, or
      document+measure the approximation error vs whole-image decode on a real
      (trained) checkpoint so users know the quality trade-off. Pure host /
      analysis; no new layer.

## Lucky-day batch 2026-06-28j (verified-novel accelerator / generative / torch port) — ALL THREE LANDED

(All three 28j items LANDED and removed: AVX per-channel + OpenCL per-depth
forward for `TNNetL2Normalize` (`cai_l2norm_perdepth` kernel + `TNNetL2NormCL`,
commit b01eab21); OpenCL per-pixel RMS forward for `TNNetPixelNorm` reusing that
kernel with InvScale=1/Depth + Eps (commit d50725e5); `TNNetResize2D` unified
arbitrary-size resize — nearest/bilinear/bicubic + align_corners, size-based
source maps for down/non-integer ratios, exact-transpose backward, FStruct[0..3]
round-trip (commit e37d07bb). Open follow-ups surfaced by the Resize2D landing:)

- [ ] `TNNetResize2D` `antialias=True` box-prefilter for downscale was SCOPED OUT
      of v1 (it changes the kernel support to scale-dependent, breaking the fixed
      2-tap / 4-tap separable gather; torch's antialias widens the kernel and
      renormalises). Add it as an opt-in 5th flag (FStruct[4]) when needed.
