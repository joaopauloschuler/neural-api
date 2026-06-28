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

- [ ] `TestSetTrainableKeepsOutputs` fails ONLY on `-dAVX2` builds (~1e-7 logit
      drift) — pre-existing, unrelated to audio/Mimi (confirmed by `git stash`
      during the RunMimiConv AVX work, commit 30c2342). The scalar build passes.
      Likely an AVX-vs-scalar reassociation crossing the test's exact-equality
      tolerance; either loosen that one assertion to a small epsilon or pin the
      reassociation. Whole suite is otherwise 0/0 on both builds.
- [ ] FFT-path FPU denormal/invalid-op traps in TNNetSpectralConv2D needed an
      example-side SetExceptionMask workaround — consider masking/guarding the
      denormals inside the layer's FFT so callers don't have to.
- [ ] SentencePiece `.model` `precompiled_charsmap` parsing — the opaque
      per-model normalization trie (NOT standard NFKC) is still not
      parsed/applied. A tokenizer.json that declares a standard NFKC/NFKD
      normalizer already works in full (`UnicodeNormalize` + `AddNormalizer`,
      landed); only the embedded charsmap is unhandled.

## Infrastructure / dev experience

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
- [ ] Qwen 3.5 import
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
      genmo/mochi-1 sibling). Distinct from the landed CogVideoX / Wan / SVD /
      AnimateDiff video stacks: a dual-stream → single-stream MMDiT (separate
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

- [ ] SDXL residual: real stabilityai/stable-diffusion-xl-base-1.0 checkpoint
      parity still deferred (only the pico oracle is verified). The 2048 cross
      width is already just CrossAttentionDim, and the highest-res "no attn"
      block layout is handled by down_block_types=DownBlock2D (DownHasAttn=false
      skips the Transformer2DModel) — transformer_layers_per_block list (e.g.
      [1,2,10]) already supported; no depth=0 special case needed. The dual
      text-encoder concatenation lives in the (still-deferred) SDXL pipeline
      driver, not the UNet importer.
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
      Lightricks `LTX-Video`). Distinct from the landed CogVideoX / Wan / SVD
      video importers: a single-stream MMDiT-style video DiT operating on tokens
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

- [ ] ResNet importer follow-ups (BuildResNetFromSafeTensors[Ex] LANDED, commit
      317a19c: torchvision resnet18/50 state_dict, conv-BN fold at load, config
      reader/ToString, resnet18 parity 1.0e-6 vs a numpy float64 oracle —
      torchvision not installed; ConvNeXt LayerScale+GRN+depthwise-7x7 is the
      modern-CNN stretch goal on the same path):
  - [ ] real downloaded torchvision resnet18/50 top-1 over ImageNet val
        (torchvision not installed here); ConvNeXt modern-CNN stretch goal. (The
        torchvision .pth load path + PyTorch stem maxpool reconciliation have
        landed.)
- [ ] Stable Diffusion VAE decoder importer follow-ups (BuildVaeDecoderFromSafeTensors[Ex]
      LANDED, commit a7870a1: diffusers AutoencoderKL decoder — post_quant_conv ->
      mid block single self-attention over HxW -> up blocks of ResNet groups +
      nearest upsample -> conv_out, latent scaled by 1/0.18215; reuse-only (nearest
      upsample = TNNetDeMaxPool(2) FSpacing=0; mid attention flattens (H,W,C)->
      (H*W,1,C) per-token SDPA; added GroupNormEpsilon to TNNetGroupNorm to pin
      diffusers eps 1e-6); TestVaeDecoderParity <1e-4 vs a numpy float64 oracle):
  - [ ] real-checkpoint (stabilityai/sd-vae-ft-mse) parity once diffusers is
        installable; SDXL VAE uses different group counts / multi-head attention.
  - [ ] SD UNet importer LANDED (BuildSDUNetFromSafeTensors[Ex] + SDUNetDenoise,
        UNet2DConditionModel v1; TestSDUNetParity 2.75e-6 < 1e-4). Open follow-ups:
    - [ ] real-checkpoint parity (runwayml/stable-diffusion-v1-5 / SDXL unet)
          once diffusers is installable. (Per-block attention-head arrays —
          SDXL-style attention_head_dim list via Config.HeadsPerBlock — have
          landed, parity < 1e-4.)
    - [ ] the end-to-end LatentTextToImage capstone (CLIP text -> this UNet ->
          scheduler loop -> VAE decoder), incl. the SDXL dual-text-encoder pooled
          embedding + real-checkpoint parity.
- [ ] CogVLM / CogVLM2 vision-language importer (`BuildCogVLMFromSafeTensors[Ex]`,
      model_type "cogvlm"/"cogvlm2"). Architecturally distinct from the
      shared-trunk VLMs already imported (LLaVA / Florence2 / Blip2 / Pixtral):
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
      offline here). (b) DONE: ComputePerceptualLossAndGradient (neuralpretrained.pas)
      exposes the Johnson-2016 deep-feature perceptual loss WITH the d(Loss)/d(pixels)
      gradient (per-stage feature L2, plain — calibrated unit-normalize is a follow-up),
      driving a multi-tap backward through the frozen VGG to the input layer; tests
      TestPerceptualLossGradientCheck (FD max rel err 7.1e-3) + TestPerceptualLossSelfZero.
      Added public TNNetLayer.GetDepartingBranchesCnt for the idempotent intermediate-tap
      IncDepartingBranchesCnt guard. OPEN: wire it into an SR example as an opt-in objective.
- [ ] ControlNet spatial-conditioning importer LANDED
      (BuildControlNetFromSafeTensors[Ex], lllyasviel/sd-controlnet-canny;
      hint stem + per-resolution zero-conv residual taps + ControlNetResiduals driver,
      TestControlNetParity 2.89e-6 < 1e-4). First conditioning-by-feature-injection model.
      DEFERRED follow-ups: (a) end-to-end base-UNet-plus-ControlNet generation
      LANDED — BuildSDUNet(..., pWithControl=true) splices a zero-default
      TNNetInput+TNNetSum onto every down-path skip + the mid output (encoder
      forward unperturbed, plain SDUNetDenoise stays bit-identical), and the new
      SDUNetDenoiseWithControl driver ADDS the ControlNetResiduals into those
      skips exactly as diffusers (down += controlnet_down, mid += controlnet_mid);
      combined forward parity-checked < 1e-4 in TestControlNetCombinedParity
      (tools/controlnet_combined_fixture.py reuses the two config-compatible pico
      fixtures) + examples/ControlNetCanny smoke. (b) real-checkpoint slicer (slice sd-controlnet-canny like
      slice_llama.py) for parity vs real weights on top of the synthetic fixture;
      (c) the diffusers per-residual conditioning_scale multiplier (constant 1.0
      here); (d) T2I-Adapter (lighter feature-injection sibling) LANDED —
      BuildT2IAdapterFromSafeTensors[Ex] imports the diffusers T2IAdapter
      full_adapter (PixelUnshuffle hint stem via TNNetSpaceToDepth with a conv_in
      input-channel permutation, an AvgPool ResNet-ish ladder of block1 3x3 ->
      ReLU -> block2 1x1 -> +identity ResnetBlocks, 1x1 in_conv channel changes);
      the T2IAdapterFeatures driver produces the per-stage feature pyramid and
      BuildSDUNet(..., pWithAdapter=true) + SDUNetDenoiseWithAdapter ADD those
      features into the SD UNet down-block hidden state (diffusers
      down_intrablock_additional_residuals); reuses existing layers (no new
      class). Feature parity < 1e-4 in TestT2IAdapterParity vs a numpy float64
      oracle (tools/t2i_adapter_tiny_fixture.py, diffusers not installed).
      Follow-ups: real-checkpoint slicer (TencentARC/t2iadapter_*_sd15v2),
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
- [ ] Quantization-aware training (QAT) — port torch.ao.quantization's
      fake-quantize-during-training so accuracy lost to post-training int8 can be
      recovered (the landed int8 path is post-training, inference-only). Add a
      `TNNetFakeQuantize` layer that, in the FORWARD pass, simulates int8
      round-to-nearest + clamp at the current (running min/max or learned) scale
      so the network sees quantization error while training, and in the BACKWARD
      pass passes the gradient straight through inside the representable range
      (reuse the landed `TNNetStraightThroughEstimator` STE clamp; zero gradient
      outside the clamp). Reuse the existing int8 scale/storage machinery so a
      QAT-trained net exports cleanly into the landed int8 weight-only format.
      Scope v1: per-tensor symmetric fake-quant on weights + activations,
      observer-driven scale; an `examples/QATFineTune` that PTQ-quantizes a small
      CIFAR net, measures the accuracy drop, then QAT-fine-tunes it back. Pin a
      numerical test that the STE gradient matches finite differences inside the
      clamp band and that the fake-quant forward equals dequant(quant(x)).
- [ ] FP16 (half-precision) OpenCL compute path for the dot-product/matmul
      offload. ENV NOTE (2026-06-27): the only OpenCL device available here is
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
- [ ] Keep activations resident on the OpenCL device across consecutive offloaded
      matmul/SDPA layers (eliminate the per-layer host round-trip during decode).
      Today each offloaded layer copies its input host->device, runs the kernel, and
      reads the result device->host (`FInputPrepared` in, `FOutput` out); the next
      offloaded layer immediately re-uploads that same buffer. For an LLM decode step
      a single block is qkv-proj -> SDPA -> o-proj -> FFN-up -> FFN-down, i.e. ~5
      back-to-back GEMM/attention layers whose intermediate tensors never need to
      touch the host. The weight buffers are already device-resident (the
      TDotProductSharedKernel weight-residency machinery the FP16 note references);
      this is the missing ACTIVATION half. Scope: add an opt-in "device output
      handle" so a layer can leave its result in a `cl_mem` buffer and the next
      offloaded layer can consume that buffer directly, with an automatic
      read-back only when the consumer is a CPU layer (norm, softmax, activation)
      or when the caller reads `.Output`. The non-offloaded layers in between
      (RMSNorm, RoPE, residual add) still run on host, so a first cut can fuse just
      the adjacent GEMM pairs (qkv->split, up->down) and measure; a full
      block-resident path is the stretch goal. Real value: device<->host traffic,
      not FLOPs, dominates the GPU decode wall-clock for ChatTerminal-scale models,
      so removing the redundant copies should lift tokens/sec materially. Guard
      everything behind the existing `FShouldOpenCL`, keep the host round-trip as the
      fallback, and pin parity with the SDPAOpenCLParity-style exact-vs-CPU test.
- [ ] Tokenizer follow-ups for neuralhftokenizer.pas:
      (b) DONE — raw SentencePiece .model protobuf path landed
      (LoadSentencePieceModel; hand-decoded ModelProto wire format, no
      vendored proto lib; auto-dispatched from LoadFromFile by the '.model'
      extension or a non-'{' first byte). Populates the SAME Unigram
      structures as the tokenizer.json path; pico fixture + spm-oracle parity
      test (tools/make_pico_spm_fixture.py, tests/fixtures/tiny_spm.model,
      TestSentencePieceModelParity). UNIGRAM model_type only; a BPE/WORD/CHAR
      ModelProto raises EHFTokenizerError ("use tokenizer.json").
      BPE-in-.model is now DONE — model_type=BPE (NLLB/mBART-BPE/DeBERTa-v3
      sentencepiece.bpe.model exports): LoadSentencePieceModel reconstructs the
      merges from the scored pieces (ReconstructBPEMerges, byte-for-byte the
      HF transformers generate_merges algorithm — per piece, every codepoint
      split whose both halves are in the vocab becomes a candidate ordered by
      (id_l,id_r), then a stable sort by (piece_id,len_l,len_r) gives the merge
      rank), sets FUnigram=false and routes encode/decode through the SAME
      metaspace + byte-level-BPE (BPEWord/EmitTokenOrFallback) machinery the
      tokenizer.json/GGUF-gpt2 BPE paths use; byte_fallback (type-6 pieces)
      covers non-ASCII via <0xNN>. Pico fixture tiny_spm_bpe.model
      (tools/make_pico_spm_fixture.py BPE variant) + TestSentencePieceBPEModelParity.
      Parity is FULL (exact), NOT an approximation: verified id-identical to the
      sentencepiece encoder over ASCII *and* non-ASCII (café/CJK/emoji/control-
      byte) inputs, and that encoder is itself id-identical to a tokenizers BPE
      model built from these same reconstructed merges. WORD/CHAR model_types
      still raise EHFTokenizerError. This UNBLOCKED the mBART/NLLB BART-family
      import (BuildMBartFromSafeTensors / BuildM2M100FromSafeTensors, LANDED)
      and the DeBERTa-v3 import (BuildDebertaV2FromSafeTensors, LANDED). No
      residual non-ASCII parity gap. BYTE
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
      UNBLOCKED the mBART/NLLB BART-family import and the DeBERTa-v3 import
      (both ship raw sentencepiece .model and are Unigram; all LANDED).
      (c) exact full-Unicode \p{L}/\p{N} tables (current
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
- [ ] GGUF writer follow-up: byte-level-BPE end-to-end model export
      (SaveTokenizerToGGUF gpt2/llama tokenizer block + verify_gguf_writer.py
      llama-cpp-python logit-parity hook LANDED): SaveLlamaToGGUFEx itself still
      only takes a plain `Tokens` array (SP "llama" model) — route the gpt2
      tokenizer block through SaveTokenizerToGGUF from the MODEL exporter (not
      just the tokenizer unit test). DONE — SaveLlamaToGGUFEx gained an optional
      `Tokenizer: TNeuralHFTokenizer` arg (+ SaveLlamaToGGUFExWithTokenizer
      convenience overload) that, when non-nil, routes the FULL block through
      TNeuralHFTokenizer.SaveTokenizerToGGUF (gpt2 vocab+merges+ids / llama
      Unigram scores), taking precedence over the plain Tokens SP path (which is
      unchanged); covered by TestGGUFWriterGpt2TokenizerRoundTrip (one-call
      weights+gpt2-tokenizer export, read back self-contained: model logits
      < 1e-5 + Encode/Decode id-identical via LoadFromGGUF on the SAME file).
      Also: the llama-cpp-python parity arm is
      wired but UNVERIFIED end-to-end (lib not installed here; pico demo GGUF may
      be too small for llama.cpp) — confirm argmax/ranking agree on a real
      checkpoint once the lib is available.
- [ ] Optimizer zoo expansion (SGD/Adam/AdamW + Lion + Adafactor exist now):
      Lion + Adafactor DONE (commit 89da1f9). TNeuralOptimizerLion (Chen et al.
      2023, sign-based update with ONE momentum buffer vs Adam's two; decoupled
      AdamW-style weight decay via TNNet.ApplyDecoupledWeightDecay) and
      TNeuralOptimizerAdafactor (Shazeer & Stern 2018, factored R+C second-moment
      vectors vs Adam's R*C — 12<27 buffers proven for a 3x3x3 kernel; full
      per-element fallback for 1-D/non-factorable params) added to neuralfit.pas,
      with the per-neuron math in neuralnetwork.pas mirroring the
      CalcAdamDelta/UpdateWeightsAdam chain. Both require SetBatchUpdate(True)
      (TNeuralFit already sets it). Tests TestLionOptimizer / TestAdafactorOptimizer
      / TestAdafactorUsesFewerBuffersThanAdam (convergence + buffer count); full
      suite 0/0. Adafactor knobs deliberately omitted (documented in the class
      header): optional first-moment beta1 EMA + update RMS clipping; LR comes from
      the host fit schedule, not Adafactor's internal relative-step rule.
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
      in tests/TestNeuralPreference.pas. Bradley-Terry pairwise reward-model
      trainer DONE: TNeuralRewardModelTrainer in neural/neuraldpo.pas learns a
      scalar reward head r(x)=RewardNet(prompt+response) from (chosen,rejected)
      pairs with loss=-ln sigmoid(r_w-r_l) and the sigmoid-margin gradient
      (dL/dr_w=-(1-sigmoid(delta)), dL/dr_l=+(...)) via the DPO batch-update +
      direct-output-error plumbing; plugs into GRPO/Best-of-N as a learned
      reward; tests in tests/TestNeuralRewardModel.pas. REMAINING: a full
      unpaired-batch KTO variant (current KTO uses the paired pair-mean KL point).
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
  - [ ] Wire the net-wide recurrent driver INTO TNNetStreamingDecoder so GenerateTokens*
        drives TokenShift/WKV/SSM uniformly alongside the SDPA KV-cache (decoder
        currently only collects SDPA + TNNetDiagonalSSM, not TokenShift/WKV/SelectiveSSM).
- [ ] Mamba decode follow-up (O(1) incremental TNNetSelectiveSSM state-carry decode
      LANDED, commit 27ba256):
  - [ ] Full Mamba-BLOCK token-by-token decode: causal DepthwiseConv1D must carry
        its (kernel-1)-token ring buffer + in/out projections driven one token at a
        time, then wire into TNNetStreamingDecoder (mirrors the RWKV TokenShift
        block-integration follow-up).
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
      arrays (not via TNNet layers), and those inner loops are currently scalar
      Pascal. After the EnCodec LSTM was AVX'd (~12.9x, real-T4 profiled), the 1D
      convolutions became the dominant cost of audio decode, so this is the next
      profiled bottleneck rather than a speculative optimization.
      FORWARD conv1d DONE (commit 8e15052): EnCodec (RunEnCodecConv serial +
      threaded TEnCodecConvWorker — covers TEnCodecModel + MusicGen EnCodec decode)
      and HiFiGAN/Vits (RunHiFiGANConv — covers TNNetHiFiGAN and the nested Vits
      decoder) now gather an im2col receptive-field patch (size InCh*K) per output
      position and run each output channel as one contiguous TNNetVolume.DotProduct;
      all 17 audio parity tests stay < 1e-4 on both scalar-fallback and real -dAVX2
      builds.
      CONVTRANSPOSE1D (upsample) DONE (commit 2873a1f for EnCodec + HiFiGAN/Vits;
      Mimi grouped transpose in 30c2342 and DAC ungrouped transpose in 8277d25 were
      vectorized inline when those holders landed): the overlap-add scatter is now a
      transposed-im2col — InT packs InSig as [t*InCh+i] (each output column
      InCh-contiguous) and WT repacks W as [(o*K+k2)*InCh+i] (each (o,k2) tap
      InCh-contiguous), so Full[o][t*Stride+k2] += DotProduct(WT_{o,k2}, InT_t, InCh)
      dispatches to AVX (Mimi/DAC use the Double-precision MimiDotProductD per group).
      EnCodec's threaded TEnCodecConvWorker.RunTranspose splits over output channels.
      The (t,k2) overlap-add order matches the original scatter; only the inner
      in-channel sum reassociates (parity < 1e-4). Re-verified all 17 audio parity
      tests (EnCodec round-trips, TestMimiParity, TestDACRoundTripParity,
      TestHiFiGANSynthesisParity, TestVitsSynthesisParity, TestMusicGen*DecoderParity)
      pass on BOTH scalar-fallback and real -dAVX2 builds. REMAINING:
  - [ ] OpenCL offload of the Mimi (`RunMimiConv`) / DAC (`RunDACConv`) holders —
        blocked on a Double-precision shared dot-product kernel (their oracle gate
        is `< 1e-4` against a float64 reference; a Float kernel would not hold it).
        (Forward conv1d + ConvTranspose1d OpenCL offload for EnCodec/HiFiGAN/Vits
        has landed via the `EnableConvOpenCL`/`SetConvOpenCLMinWork` gated path.)
      Design note (the landed forward path follows this): channel-major layout
      means the contraction (sum over in-channels for a fixed kernel tap) is
      depth-axis-contiguous — exactly the case `TNNetVolume.DotProduct` / `MulAdd`
      already vectorize — so the win is reusing the existing AVX volume primitives
      (or a small im2col-into-DotProduct reshape) for the per-tap accumulation
      instead of triple-nested scalar loops. Gate any remaining rewrite on the
      existing parity tests (TestHiFiGANSynthesisParity / TestVitsSynthesisParity /
      EnCodec round-trip) staying `< 1e-4`, and re-profile decode wall-clock
      before/after.

- [ ] Single-sample `TNNetFullConnect` forward threading follow-ups (opt-in
      `EnableFullConnectThreading` multi-core `ComputeCPU` v1 LANDED for
      FullConnect/FullConnectLinear/FullConnectReLU — off by default, bit-identical
      to serial, work-thresholded via `SetFullConnectThreadingMinWork`):
      (a) the per-token projection HOLDERS (ChatTerminal / MusicGenText /
      `RunLlama`-style decoders) still need to call
      `EnableFullConnectThreading(true)` to opt in — wire it into the imported-model
      run paths; (b) re-profile one-token decode wall-clock before/after on a real
      imported decoder to tune the default threshold; (c) the int8-quantized
      `ComputeQuantizedInt8CPU` and `TNNetFullConnectSigmoid`/other activation
      siblings are still serial.
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
- [ ] Kokoro StyleTTS2 TTS real-checkpoint follow-up (`BuildKokoroFromSafeTensors[Ex]`
      + `TNNetAdaIN` leaf + `TNNetKokoro` holder + examples/KokoroTTS LANDED, pico
      parity TestKokoroSynthesisParity < 1e-4): real `hexgrad/Kokoro-82M` checkpoint +
      voice-pack `.pt` style-tensor load (offline/RAM-gated here); also the
      grapheme->phoneme (espeak/misaki) frontend dropped from v1 (v1 takes
      pre-phonemized integer ids).
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
      greedy window). DONE in commit d0b27b9: (a)(e)(d) + partial (c) —
      (a) median-filter smoothing (WhisperMedianFilterRow, odd-kernel reflect-
      padded median along the audio-frame axis; optional MedianKernel param on
      WhisperCollectCrossAttention + WhisperWordTimestamps, default 0 = disabled =
      bit-identical to v1, 7 = openai-whisper default); (e) per-word confidence
      (Confidence: TNeuralFloat field on TWhisperWordTimestamp = mean DTW-path
      attention over the word's frame segment); (d) examples/WhisperTranscribe
      `--word-timestamps` flag printing `start - end [confidence] word`, documented
      in both READMEs; (c partial) baked-in alignment heads added for medium
      (24L/16H) and large (32L/20H) shapes. Tests in TestWhisperWordTimestamps
      (kernel-1 bit-identical to no-smoothing, kernel-7 monotonic non-decreasing
      boundaries + valid prob rows, confidence in [0,1.0001], non-empty
      medium/large head lists); suite 2273/0/0; example compiles (not run e2e — no
      checkpoint). REMAINING:
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
        (BuildResNetFromSafeTensors etc.) over a real checkpoint + the 50k-image
        ImageNet-val set and compare top-1/top-5 to the published torchvision numbers
        (the smoke is synthetic-only by design). Mind the multi-GB RAM budget.

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
      synthetic text states.
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

- [~] Stable Diffusion img2img + inpainting pipelines. DONE (commit 2191b80): the
      VAE ENCODER gap is closed — BuildVaeEncoderFromSafeTensors[Ex] +
      ReadVaeEncoderConfigFromJSONFile land the diffusers AutoencoderKL encoder
      (conv_in -> down blocks with stride-2 downsample convs reproducing the
      asymmetric (0,1,0,1) pad -> mid self-attn -> conv_norm_out/conv_out ->
      quant_conv), reusing every decoder helper (TVaeDecoderConfig,
      AddVaeResnetBlock, AddVaeAttention, LoadVaeConv). The net's final layer crops
      to mean and applies the 0.18215 scaling_factor = the deterministic img2img
      init-latent setup. Pico fixture (tools/vae_encoder_tiny_fixture.py ->
      tests/fixtures/tiny_vae_encoder*) + TestVaeEncoderParity (numpy float64
      oracle, max|diff| < 1e-4) + TestVaeRoundTrip (encode->decode end to end).
      DONE (img2img sampler loop): examples/ImageToImage drives the full
      encode->partial-noise->denoise->decode SDEdit trajectory (strength knob).
      DONE (blended inpainting): SDUNetDenoiseInpaint (neuralpretrained.pas) runs a
      complete reverse loop on the standard 4-channel SD UNet and at EVERY step
      blends the known region back in (latents = mask*denoised + (1-mask)*AddNoise(
      z0,t)); examples/DiffusionInpainting smoke-runs it on the pico VAE+SD-UNet
      fixtures (right-half hole), writing masked_input.ppm/inpainted.ppm, and
      TestDiffusionInpaintSmoke asserts the kept latent region equals z0 exactly
      while the masked hole changes. This is the no-retrain blended mode.
      STILL OPEN: (1) the non-deterministic VAE reparameterize head
      (mean+exp(0.5*logvar)*noise); (2) the 9-channel inpaint-specialized UNet
      (conv_in widened to 4 latent + 1 mask + 4 masked-latent) importer support.
      ORIGINAL NOTE — The landed SD path
      (BuildVaeDecoderFromSafeTensors + BuildSDUNetFromSafeTensors + SDUNetDenoise +
      the neuraldiffusion sampler zoo) does text-to-image only — it starts denoising
      from pure Gaussian latents. The two highest-value missing modes reuse ALL of
      that and add only the latent setup/blend: (a) img2img — VAE-ENCODE an init
      image to latents (the VAE decoder importer's encoder twin is the gap; add
      BuildVaeEncoderFromSafeTensors for the down-blocks + the
      `mean,logvar`->reparameterize sampling head), then start the existing sampler
      at timestep `int(strength * num_steps)` with noise added to those latents
      (`scheduler.add_noise` — already have the alphas_cumprod schedule); (b)
      inpainting — keep img2img's noised init but at every denoise step BLEND the
      known region back in (`latents = mask*denoised + (1-mask)*noised_known`), and
      support the 9-channel inpaint UNet (4 latent + 1 mask + 4 masked-latent in
      `conv_in`) by widening the importer's first-conv loader. Real value: img2img
      and inpainting are the two most-used SD modes after txt2img and unlock the
      ImageToImage/Inpainting example stubs. Pico parity: VAE-encode round-trip <
      1e-4 vs a float64 HF AutoencoderKL.encode, and a fixed-seed img2img latent
      trajectory matching diffusers `StableDiffusionImg2ImgPipeline` step-for-step.
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
- [ ] Qwen-Image text-to-image MMDiT importer (`BuildQwenImageFromSafeTensors[Ex]`,
      Qwen/Qwen-Image, model_type `qwen_image`). The 2025 flagship open text-to-image
      model: a Qwen2.5-VL text encoder (LANDED — `BuildQwen2VL`-family) feeds an
      MMDiT-style double-stream diffusion transformer over packed VAE latents, with a
      16-channel VAE decoder (the landed `BuildVaeDecoder` path) producing the image.
      Almost entirely a wiring job over already-landed pieces: the MMDiT joint-attention
      blocks (the SD3/FLUX `MMDiT` task's building blocks), the landed
      `TNNetDiffusionScheduler` (flow-matching/Euler sampling already supported), the
      VAE decoder and the Qwen2.5-VL prompt encoder. New work is the config-driven
      block stacking + the modulation (AdaLN-Zero) timestep/text conditioning wiring and
      the 2-D RoPE over latent patches. Pico fixture + `TestQwenImage*Parity` < 1e-4 vs a
      float64 HF `QwenImagePipeline` transformer forward on a fixed latent/text pair.
      Real value: a current state-of-the-art open text-to-image generator runnable as a
      single native binary, reusing the diffusion + VAE + Qwen-VL infrastructure already
      in the repo.

## Layer follow-ups that fix real limitations

- [~] Bidirectional + multi-layer stacking for `TNNetLSTMCell` / `TNNetGRUCell`
      (and a real `nn.LSTM`/`nn.GRU` num_layers>1 / bidirectional=True checkpoint
      importer). The single-cell `TNNetLSTMCell`/`TNNetGRUCell` (torch fused
      `weight_ih`/`weight_hh` gate layout, full forward + exact BPTT) have landed.
      DONE: `TNNet.AddBidirectionalLSTM(Hidden,NumLayers,Bidirectional)` and
      `AddBidirectionalGRU(...)` builders (shared `AddBidirectionalRecurrentStack`
      engine) — pure composition of cells + `TNNetFlipX` reversal + `TNNetDeepConcat`
      ([forward;backward] torch concat order); multi-layer feeds layer k's output
      into k+1, each direction gets a `TNNetPointwiseConvLinear` projection to
      Hidden when incoming Depth≠Hidden (covers layer 0 and the 2*Hidden→Hidden
      feed above a bidirectional layer, matching nn.LSTM input_size). Tests:
      builder wiring/shape/SaveToString round-trip (LSTM+GRU), unidirectional
      1-layer == bare cell equivalence, bidirectional forward-half == forward cell
      alone (pins concat order), and a 2-layer bidirectional stack input-gradient
      check (all in TestNeuralNumerical, passing; full suite green 2376/0/0). This
      was applied to the pyannote `segmentation-3.0` trunk, which now uses true
      nn.LSTM cells.
  - [ ] Faithful STACKED-BIDIRECTIONAL torch import follow-up: the
        `AddBidirectionalRecurrentStack` engine inserts a learned
        `TNNetPointwiseConvLinear` projection on every incoming Depth≠Hidden feed
        (layer-0 input_size≠Hidden, and the 2*Hidden→Hidden feed above each
        bidirectional layer). torch nn.LSTM/nn.GRU has NO such projection — layer
        k's `weight_ih_l{k}` is a plain `[gates*Hidden, 2*Hidden]` slab consuming
        the concatenated [forward;backward] hidden directly. To import a real
        multi-layer bidirectional checkpoint, add an opt-in builder mode that
        feeds the 2*Hidden concat straight into the next layer's cell (cell
        input_size = 2*Hidden, no projection) so the torch `weight_ih_l{k}` rows
        map 1:1. Then extend `LoadTorchLSTMInto`/`LoadTorchGRUInto` to accept that
        projection-free stack and drop the loud rejection; add a 2-layer
        bidirectional pico-oracle parity fixture (the generator already supports
        the shape — `tools/torch_rnn_tiny_fixture.py`). Unblocks the pyannote
        `segmentation-3.0` real bidirectional-LSTM trunk drop-in.

- [ ] OpenCL forward offload for `TNNetCausalLinearAttention` (the non-causal global
      `TNNetLinearAttention` is now DONE — `ComputeOpenCL` two-GEMM offload behind
      `FShouldOpenCL` + `LinearAttentionOpenCLParity` exact-vs-CPU test, PoCL-verified
      max|diff| ≈ 1.5e-8). The causal sibling was deliberately LEFT CPU-only: its
      forward is a left-to-right prefix-sum scan (`S_t = S_{t-1} + ϕ(K_t)⊗V_t`,
      per-query `Out_t = ϕ(Q_t)·S_t / ϕ(Q_t)·Z_t`), NOT a pair of dense GEMMs, so it
      does not map onto the `FDotCL` matmul kernel without a chunked-scan rewrite.
      Tackle it with the chunked-forward family below (intra-chunk dense GEMM +
      inter-chunk running state), then add a causal parity test alongside.

- [ ] AVX-vectorize the remaining trig/inverse-hyperbolic activation FORWARD
      loops left scalar: `TNNetSin` / `TNNetCos` / `TNNetArcSinh`.
      `TNNetSinhAct` is now DONE — `sinh(x) = (exp(x) - exp(-x))/2` reduces to two
      `VectorExp` passes, so `TNNetVolume.VectorSinh` was added (alias-safe, AVX2
      via `VectorExp`) and `TNNetSinhAct.Compute` converted to the two-pass scheme;
      parity test `TestVectorSinhScalarParity` (< 1e-4, body+tail+aliased) green on
      both builds. The remaining three CANNOT ride `VectorExp`: `TNNetArcSinh`
      needs a vectorized natural log (`ln(x+sqrt(x^2+1))`) and `TNNetSin`/`TNNetCos`
      need a Cephes-style range-reduced polynomial sin/cos. None of those have a
      `VectorExp`-derived identity, and the AVX path is asm-only (`AVXExp`), so a
      pure-Pascal "primitive" would gain nothing — they need dedicated 8-wide
      `VectorLn` / `VectorSin` / `VectorCos` asm/intrinsic kernels held to < 1e-4
      vs the RTL, which is the actual remaining work. Earlier tanh/erf batch:
      `TNNetGELU`, `TNNetGELUErf`, `TNNetMish`, `TNNetSerf`, `TNNetSmish`,
      `TNNetPhish`, `TNNetLogCoshActivation`, `TNNetLisht` vectorized via
      `VectorTanh`/`VectorErf` (built on `VectorExp`); parity tests
      `TestVectorTanhScalarParity` / `TestVectorErfScalarParity` (< 1e-4) green.

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
- [ ] Continue upsampling/deconvolution audit: TNNetDeconvolution input
      AND weight gradients (Upsample / DeMaxPool / DeAvgPool already
      covered). Likely benefits from the Double-precision SSE accumulator
      helper above.
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
  - [ ] Wire detection into `BuildLlamaFromSafeTensors` family — detect a
        `weight` + `weight.absmax` bnb-4bit pair, expand to F32 at load (mirrors
        how MXFP4 dequant is consumed; both are currently helper-level, the
        gpt-oss MXFP4 build-path consumption is itself documented-but-deferred).
  - [ ] Double-quantized absmax: handle `*.nested_absmax` / `*.nested_quant_map`
        / `*.quant_state.*` (the secondary int8 quant + nested absmax + offset).
        The unit consumes already-FP32 absmax and documents that the importer
        must expand or raise rather than feed quantized absmax in (no silent
        garbage path). A real `bnb-4bit` Llama/Qwen slicer fixture is RAM/network
        -gated; today's parity is the random pico numpy oracle only.

## Accelerator & dedup batch 2026-06-27f

(All LANDED and removed: OpenCL softmax-head offload — new `cai_softmax` kernel +
`TNNetSoftMaxCL` host helper for `TNNetPointwiseSoftMax`/`TNNetSoftMax`, commit
e8042aa0; AVX `TNNetReLU` forward via `AVXCopyRelu`, commit 10433bfd; AVX
SoftPlus/Gaussian/SoftExponential forwards via `AVXExp` — note `AVXExp`'s scalar
remainder tail has NO internal clamp so extreme inputs must be pre-clamped to
[-88,88] before the call, commit 3e5e649d. Open follow-ups surfaced by this batch:)

- [ ] Keep the softmax-head activation resident on the OpenCL device across the
      SDPA score-matrix producer -> `cai_softmax` -> consumer chain. The new
      `cai_softmax` offload (e8042aa0) still uploads/downloads the volume per call;
      when the producer is already device-resident this is a wasted round-trip.
      Tie into the existing "keep activations resident across consecutive offloaded
      layers" follow-up in the vision/generative section so attention blocks chain
      device-side. Forward-only, parity-tested, skip-clean when no device.

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

## Lucky-day batch 2026-06-28g (follow-ups surfaced by 28g landings)

(All five 28g items LANDED and removed: SDPA attention-variant OpenCL safety
guard for all four variants + correct ALiBi device offload [9957892f]; AVX
`TNNetSoftPool` [6e373ae4]; AVX `TNNetCumSum` depth-axis [ff593de0];
`TNNetFiniteScalarQuant`/FSQ [dfaa9c41]; `TNNetLookupFreeQuant`/LFQ [8cfaea3b].
The open OpenCL follow-up LANDED and removed: device offload for the remaining
three SDPA variants (cosine-sim / disentangled / conformer) via per-variant
EnableOpenCL + ComputeOpenCL, gathered position dots stay on the CPU gap, new
CosineSimilarity/Disentangled/ConformerRelPosAttentionOpenCLParity tests RAN on
PoCL at max|diff| ~1e-7, suite 2493/2493 on both builds [914cbdc9].)

## Lucky-day batch 2026-06-28h (verified-novel AVX / dedup) — ALL LANDED

(All three 28h items LANDED and removed: AVX-vectorized SDPA `ComputeTiled`
online-softmax accumulation via `TNNetVolume.Mul`/`MulAdd` depth-contiguous ops,
tiled-vs-naive parity green on scalar + -dAVX2 [02d11b09]; fused-QKV unpackers
unified into `LoadFusedQKVWeights` + `TQKVPackLayout` enum, three importer
wrappers preserved, pico fixtures bit-identical [6083a4d0]; GPT-2/Llama/Cohere/
Bark affine norm-weight loaders unified into `LoadAffineNormWeights`, four thin
wrappers preserved, all parity fixtures bit-identical [dd5373fc].)

## Lucky-day batch 2026-06-28i (verified-novel vision dedup / AVX / torch port)

- [ ] Add a reusable `TNNetGridSample` layer (port of torch `F.grid_sample`):
      two-source layer that warps a feature map by an explicit per-pixel sampling
      grid (second input supplies normalized (x, y) flow in [-1, 1] over the
      output positions), with `bilinear` and `nearest` interpolation modes,
      `zeros`/`border` padding modes and an `align_corners` flag. The bilinear
      forward already exists TWICE in scalar/partly-AVX form — in
      `TNNetAffineGridSample.Compute` (depth-contiguous `MulAdd` blend, AVX) and
      in `TNNetDeformableConv.SampleBilinear` (scalar per-channel) — so factor a
      single shared depth-column bilinear-gather helper and have all three sites
      use it. `TNNetAffineGridSample` then becomes the affine special case that
      generates its grid from a 2x3 theta and delegates to the shared gather.
      Numerical-gradient test the grid (flow) input path; OpenCL is optional
      (the existing `TNNetAffineGridSample` gather kernel is the template).

- [ ] AVX-vectorize the `TNNetDeformableConv` bilinear gather. The offsets are
      per-tap (shared across all Depth channels of a given sampled position), so
      the four corner taps can be blended as Depth-long contiguous `MulAdd`
      accumulations exactly like `TNNetAffineGridSample.Compute` already does,
      instead of the current scalar per-channel `PrevOut.Get(x, y, ci)` loop in
      `SampleBilinear`. Route through the shared depth-column gather helper from
      the `TNNetGridSample` item above (or stand alone if that lands later).
      Forward-and-backward parity-test against the current scalar path on a small
      net; the GEMM stage already offloads, this removes the remaining CPU-scalar
      gather bottleneck. Pairs with the existing DCNv2 modulated mask path.

- [ ] Add a reusable `TNNet.AddPatchEmbedding` builder method (patchify +
      linear projection to tokens, optional learnable class token and learnable
      positional embedding) for ViT-style stacks. Today every patch-tokenizing
      example (e.g. `MaskedAutoencoder`, `Perceiver`, MLP-Mixer / ViT demos)
      hand-rolls the conv-stride-`PatchSize` → flatten-to-(SeqLen,1,EmbedDim)
      sequence reshape inline. One builder removes that duplication and makes
      from-scratch ViT examples a few lines. Mirror the existing
      `Add*Block`/`Add*Encoder` builder conventions; convert at least one
      example to use it as the regression check.
