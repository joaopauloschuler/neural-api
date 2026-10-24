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

- [ ] Possible trap (not changed, awaits user decision): `TVolume.Copy`
      (neural/neuralvolume.pas) calls `ReSize` only when `Size` differs,
      so copying a 2048x512x1 volume into a 1024x1024x1 one keeps the
      destination's 1024x1024 shape. `PasteInpaintCrop` (ecd33dfb) resizes its
      destination first to avoid it. Options: compare SizeX/SizeY/Depth there
      (cheap, but a hot call), or a separate `CopyWithShape`; audit callers that
      rely on the current behaviour before changing it.

- [ ] Possible trap (not changed, awaits user decision; found during fcfbfa66):
      FPC's `genrand_MT19937` reseeds when `RandSeed <> OldRandSeed`, then sets
      `RandSeed := not RandSeed` and stores that as `OldRandSeed`. Code that saves
      `RandSeed` and restores it afterwards (about 40 sites in neural/) restores a value equal to `OldRandSeed`, so the next draw
      CONTINUES the old Mersenne Twister state instead of replaying the seed;
      "same seed" noise can then differ by call order (it broke
      `TestQwenImage21Img2ImgKeepLoaded` once). Options: restore with `RandSeed :=
      Saved` followed by an explicit reseed, or a private seeded generator for
      latent noise. Audit the call sites first. `RestartRandSeed(Seed)`
      (neuralvolume.pas, 37b3922a) forces the reseed and
      `MakeInitialLatents` uses it; the other save/restore sites in neural/
      are unchanged and could switch to it.
- [ ] Possible defect (not fixed, awaits user decision): dense `TNNetSpectralNorm`
      (neural/neuralnetwork.pas) keeps the inherited FC weight delta
      (gradient w.r.t. w_eff) without the 1/sigma factor its conv sibling now
      applies; its declaration comment calls this intended, but dL/dW =
      (1/sigma) dL/dw_eff under the same constant-sigma rule.
- [ ] SentencePiece `.model` `precompiled_charsmap` parsing — the opaque
      per-model normalization trie (NOT standard NFKC) is still not
      parsed/applied. A tokenizer.json that declares a standard NFKC/NFKD
      normalizer already works in full (`UnicodeNormalize` + `AddNormalizer`);
      only the embedded charsmap is unhandled.

## Hardware Support
  - [ ] CUDA/HIP Support
    - [ ] Software Spec 
  - [ ] ARM SIMD: NEON → SVE → SME kernels behind the TNNetVolume dispatch (requires ARM processor) 
    - [ ] Software Spec

## Infrastructure / dev experience

- [ ] OpenCL test suite under 2 minutes (user goal and authorization
      2026-10-05; ~600 s on PoCL before S1-S5). Diagnosis (single-test runs): idle
      scheduler workers spin hot against PoCL (one VAE test: OpenCL + parallel
      pass 95 s vs 3.3 s serial, 415k clock syscalls); neural.cl is built again
      for every EnableOpenCL; a few heavy test shapes. Serial tasks, each with a
      fresh read-only Opus reviewer, the three suites and a commit:
      S1-S5 (hot worker 0 only, neural.cl built once per process, smaller test
      shapes, fewer work-group sizes, opt-in slow set) are done; the goal is
      not yet met.

- [ ] ChatServer: multiple chat sessions in cache, so parallel chats do not
      re-prefill (design agreed with the user 2026-10-02). Conversation slots
      (`--kv-slots`, `TConversationSlot`, T1-T6) are coded; open items below.
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
- [ ] InternLM2 / InternLM2.5 importer follow-ups (`BuildInternLM2FromSafeTensors[Ex]`,
      model_type "internlm2", `TNNetInternLM2Reader` wqkv unpack + rename onto the
      core Llama builder; pico parity `TestInternLM2ConfigParity` /
      `TestInternLM2LogitParity` < 1e-4 vs a numpy float64 forward from
      tools/internlm2_tiny_fixture.py):
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
        checkpoint needs it (no grouped top-k gate exists yet; the DeepSeek
        importer also rejects group_limited_greedy, so both would share one
        implementation).
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

- [ ] FastSAM follow-ups (ultralytics FastSAM-s / FastSAM-x on the YOLOv8-seg
      importer: segment head, `ComputeYoloMasks` / `DecodeYoloSegmentation`
      with retina masks and `MaxDetections`, box/point prompts, native
      retina masks at the original image size (`ComputeYoloMasksNative`), the
      CLIP text prompt (`FastSAMSelectByText`) and examples/FastSAM are done;
      pico parity `TestYoloSegParity`, `TestFastSAMPromptSelection`,
      `TestYoloSegNativeMasks`, `TestFastSAMTextSelection`,
      `TestYoloSegOpenCLParity`):
      - [ ] Real-checkpoint run: FastSAM-s.pt / FastSAM-x.pt converted to
            safetensors on the user's box (config rows in
            examples/FastSAM/README.md); check masks against ultralytics.
      - [ ] Real run of `--text` with openai/clip-vit-base-patch32 (real
            tokenizer.json + towers) against ultralytics' selection.

- [ ] Marigold diffusion depth / surface normals follow-ups (`TMarigoldPipeline`
      in neuralpretrained.pas + `examples/MarigoldDepth`, dc2979d4; checkpoints
      `prs-eth/marigold-depth-v1-1` / `prs-eth/marigold-normals-v1-1`, Ke et al.
      2024, arXiv:2312.02145). The SD2 `CLIPTextModel` 1024-wide config was
      checked against the real SD 2.1 `text_encoder/` (719fb4d2). Open:
      - [ ] Pico-fixture parity vs diffusers in the ~/x venv (tiny UNet + VAE,
            like the existing tools/sd_unet_*_fixture.py; ask the user before
            adding a new Python tool), < 1e-4 on the final depth map with fixed
            noise.
      - [ ] Real-checkpoint runs of examples/MarigoldDepth (FP32, `--int8`,
            `--gpu`) on the user's GPU box: the UNet is ~865M params (3.4 GB
            FP32), above the 2 GB cap on the dev box. SD UNet / VAE under
            OpenCL are covered only by pico tests on PoCL.

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

- [ ] BiRefNet / RMBG-2.0 follow-up (`BuildBiRefNetFromSafeTensors[Ex]` +
      examples/BackgroundRemoval landed; pico parity and OpenCL parity only):
      run ZhengPeng7/BiRefNet and briaai/RMBG-2.0 at 1024x1024 on the user's GPU
      box (`--gpu`, also CPU) and compare the mask with the official
      `transformers` remote-code output before README calls it supported.

- [ ] ViTMatte trimap-based image matting follow-ups (Yao et al. 2023,
      arXiv:2305.15272; `BuildViTMatteFromSafeTensors[Ex]`, `ViTMatteEstimateAlpha`,
      `TrimapFromMask` and examples/ImageMatting are done; pico parity
      `TestViTMatteParity` plus OpenCL parity):
  - [ ] Real hustvl/vitmatte-small-composition-1k run on the GPU box (`--gpu`,
        also CPU) compared with the transformers output before README calls
        it supported.

- [ ] Robust Video Matting (Lin et al. WACV 2022, arXiv:2108.11515,
      PeterL1n/RobustVideoMatting; `rvm_mobilenetv3.pth` is GPL-3.0, so load it
      at run time and never ship it). The MobileNetV3 importer
      (`BuildRVM[FromSafeTensors[Ex]]`, `TRVMMatting`) and examples/VideoMatting
      are done; open:
  - [ ] Real-checkpoint run on the GPU box (`--gpu`) before README calls it
        supported.
  - [ ] ResNet-50 real checkpoint: `rvm_resnet50.pth` was not on the dev box;
        run `tools/rvm_tiny_fixture.py --variant resnet50 --real FILE` and
        compare a 128x96 clip, then a GPU run, before README calls it tested.
  - [ ] Memory: the real `rvm_mobilenetv3.pth` at 640x480, ratio 0.75 peaked at
        1.7 GB RSS on CPU (dev box). Find which activations dominate (the
        full-res refiner maps?) before 720p/1080p runs; check the low-memory
        mode and whether the full-res layers can drop their outputs early.
  - [ ] Real-checkpoint check of `--segmentation` and `--refiner
        fast_guided_filter` with `rvm_mobilenetv3.pth` against the official
        model (`tools/rvm_tiny_fixture.py --real` only runs the DGF matte).

- [ ] Hunyuan-DiT bilingual text-to-image importer
      (`BuildHunyuanDiTFromSafeTensors[Ex]`, Tencent `Hunyuan-DiT`). A distinct
      DiT from the landed PixArt/MMDiT stacks: a cross-attention DiT
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
        TNNetVisionRoPE2D is fixed to the build-time grid. DINOv3-backed
        dense prediction also needs the BuildDPT register-token item under
        Depth Anything V2 below.

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

- [ ] YOLO-World v2 importer follow-ups (BuildYoloWorld[FromSafeTensors[Ex]] LANDED: folds CLIP prompt
      embeddings into the C2fAttn attention + BNContrastiveHead convs of the shared
      YOLOv8 builder (TestYoloWorldParity, TestYoloWorldClipTextRoute);
      examples/YoloDetect takes --prompts/--clip/--image):
  - [ ] Run a real yolov8s-worldv2 checkpoint + openai/clip-vit-base-patch32 on a
        photo and compare boxes/scores with ultralytics (only the random pico
        fixture is tested; the module semantics come from memory of the
        ultralytics source, which is not installed here).
  - [ ] Checkpoint route: ultralytics ships yolov8s-worldv2.pt as a pickled
        model object (AGPL-3.0); read it through neuraltorchbin.pas or document a
        Pascal-only conversion to a safetensors state_dict + config.json.
  - [ ] Re-prompt without reloading: keep the gl / cv3.i.2 / cv4.i tensors and
        rewrite the folded convs for a same-count prompt list (a new count still
        changes the conv output sizes).
  - [ ] YOLO-World v1 (yolov8s-world: ImagePoolingAttn + L2 ContrastiveHead,
        head at model.23) is rejected; add it only if a user needs it.
  - [ ] Measure decode cost vs plain YOLOv8 at 640 with 80 prompts (the
        attention logit conv is dense over heads: Channels * prompts * heads).

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
  - [ ] real keep_aspect_ratio run (518, ensure_multiple_of 14) with
        DPTPreprocessImage on a photo; the example has no image input yet.
- [ ] Video Depth Anything importer (temporally consistent depth for long videos;
      BLOCKED on user approval: the pico parity needs a NEW Python oracle script in
      tools/ (reference DPTHeadTemporal forward in the ~/x venv); ask first.
      Pascal-only pieces that can go ahead without it: the AlignDepthScaleShift
      least-squares routine (hand-derived tests) and the frame-folder reader.
      DepthAnything/Video-Depth-Anything, video_depth_anything_vits.pth, Apache-2.0
      Small). It is the landed Depth Anything V2 stack (DINOv2 encoder per frame ->
      DPT neck) whose head gains temporal motion modules: self-attention along the
      FRAME axis per spatial cell with a sinusoidal temporal position embedding,
      inserted on the coarse reassembled levels and the fusion path (check the exact
      placement against the reference DPTHeadTemporal). Reuse: BuildDPT /
      BuildDepthAnythingV2 for the per-frame trunk, neuraltorchbin.pas to read the
      .pth directly, the landed SDPA over a (NumFrames,1,C) per-cell reshape for the
      temporal block (share that block with the AnimateDiff motion-module task below
      instead of coding two). Inference driver: 32-frame windows with overlap and
      keyframes, per-window least-squares scale+shift alignment of the overlapping
      frames, then linear blend (one reusable AlignDepthScaleShift routine; factor
      the closed-form s*a+t solve out of MarigoldEnsembleDepth). Pico parity vs the
      reference float64 forward on a 4-frame clip; examples/VideoDepth reads a folder
      of frames (same input convention as examples/VideoMatting / ObjectTracking) and
      writes per-frame colour depth PPMs + an animated GIF via the landed GIF writer,
      with a README comparing the flicker of per-frame examples/DepthAnythingV2
      output against the temporal model.

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
          2026-10-02). SD 1.x pipeline + CLI LANDED 2026-10-21 (8abce32d,
          d7d07607, d55b7bfc, b764df66; pico parity vs a diffusers float64 run
          on tests/fixtures/tiny_sd15 from tools/make_pico_sd_pipeline_fixture.py).
          Open pieces:
      - [ ] Real-checkpoint run of SD 1.5 (runwayml/stable-diffusion-v1-5 F32
            safetensors) on the user's GPU box with examples/StableDiffusion
            --gpu; FP32 UNet ~3.4 GB does not fit the dev box. Then
            per-component parity against diffusers on the real weights (the ~/x
            venv oracle recipe).
      - [ ] Sampler gaps behind SamplerNotice (DDIM/PNDM/Euler/Euler a/DPM++ 2M
            are mapped with diffusers parity): UniPCMultistepScheduler (smUniPC
            has only a numpy oracle; diffusers takes N of N+1 points like DPM),
            PNDM PRK warm-up, DDPMScheduler (smDDPM ignores the stride).
            Continuous sigmas for the 4-channel inpaint blend (AfterStep takes
            an integer timestep, so those runs keep integer timesteps), Euler
            sigma_min / sigma_max overrides and final_sigmas_type "sigma_min"
            (noticed, not done).
      - [ ] Latent Consistency Models follow-ups (LCM-LoRA sampling
            `--sampler lcm` on TNNetLCMScheduler, 1a31f710; LCM-distilled
            UNets via time_cond_proj_dim, 0bfbb06b):
        - [ ] Real LCM-LoRA run on the user's GPU box:
              latent-consistency/lcm-lora-sdv1-5 through --lora + --sampler lcm,
              4 steps, guidance 1.0 and 1.5. Check that every LoRA key of that
              file merges (the pico LoRA test covers attention, proj_in, ff,
              conv1 and time_emb_proj only).
        - [ ] Real LCM-distilled run on the user's GPU box:
              SimianLuo/LCM_Dreamshaper_v7, 4 steps, guidance 8.5, text-to-image
              and --init-image. Its unet is F32 safetensors; check the load and
              the image.
        - [ ] BuildControlNet ignores a ControlNet config's time_cond_proj_dim
              (no cond_proj wiring, no refusal); refuse it or wire it like
              BuildSDUNet if an LCM-distilled ControlNet shows up.
        - [ ] SDXL LCM-distilled UNets (latent-consistency/lcm-sdxl): the
              importer builds guidance + add_embeds together, but the pipeline
              refuses SDXL; untested.
        - [ ] Decide the LCM img2img schedule for LCM-LoRA. Today smLCM img2img
              follows LatentConsistencyModelImg2ImgPipeline (strength shrinks
              the origin grid, all steps run: 0.5 x 4 -> 499, 379, 259, 139).
              diffusers StableDiffusionImg2ImgPipeline + LCMScheduler (the
              documented LCM-LoRA img2img route, AutoPipelineForImage2Image)
              keeps the last int(steps * strength) of the full grid instead
              (-> 499, 259). Optionally offer that one too.
      - [ ] SD 2.x follow-ups (pico parity: tests/fixtures/tiny_sd2,
            TestStableDiffusion2*):
        - [ ] Real SD 2.1 run on the user's GPU box (768x768, v_prediction).
              stabilityai/stable-diffusion-2-1 is gone from the Hub; the
              sd2-community/stable-diffusion-2-1 mirror has the same files.
              Then 2.1-base (512, epsilon).
        - [ ] Inpainting with v_prediction has no fixture (inspection only: it
              noises with the same AddNoise as img2img). Add one if a v
              inpainting checkpoint shows up.
        - [ ] SD 2.x upcast_attention is ignored (all FP32); it matters only
              with the F16 item below.
        - [ ] SD 2.x depth-to-image follow-ups (TStableDiffusionDepth2Img,
              TestStableDiffusion2Depth2ImgParity):
          - [ ] Real stabilityai/stable-diffusion-2-depth run on the user's GPU
                box (512x512) with a depth map from another estimator
                (--depth-image).
          - [ ] DPT-Hybrid / plain-ViT DPT follow-ups (BuildDPT;
                TestDPTHybridDepthEstimationParity, TestDPTViTDepthEstimationParity):
            - [ ] Real Intel/dpt-hybrid-midas and Intel/dpt-large runs on the
                  user's GPU box (384x384; compare against transformers).
            - [ ] Non-native input size (TDPTConfig.InputWidth/InputHeight)
                  follow-ups:
              - [ ] Stock transformers refuses a non-native DPT-Hybrid input
                    and reshapes a non-square ViT DPT neck input to
                    sqrt(tokens); the oracle lifts both (image_size override,
                    patch grid handed to the neck). Our non-square DPT output
                    follows the DINOv2-backbone DPT path, not stock behaviour.
              - [ ] A DPT (not Depth Anything) neck level of odd size (an odd
                    patch grid with a 0.5 reassemble factor) is refused;
                    DPTFeatureFusionLayer bilinearly resizes the residual
                    instead.
              - [ ] DPTPreprocessImage has a size test only
                    (TestDPTResizeOutputSize); no pixel parity vs
                    DPTImageProcessor with keep_aspect_ratio.
            - [ ] readout "add" is built but has no fixture; BiT
                  preactivation layers, dilated stages (output_stride reached)
                  and batch-norm fusion units are refused.
            - [ ] FPC 3.x -O2 miscompiled `B <> (Flag and (I < 2))` in a for
                  loop in BuildDPT (garbage index register; rewritten with a
                  local boolean). Report upstream with a reduced case.
          - [ ] Depth2Img depth-estimator follow-ups (EstimateDepthMap;
                TestStableDiffusion2Depth2ImgEstimatorParity):
            - [ ] Real stabilityai/stable-diffusion-2-depth run with its own
                  depth_estimator on the user's GPU box (512x512); compare the
                  estimated depth with diffusers.
            - [ ] transformers with torchvision uses the torchvision uint8
                  resize for DPTImageProcessor; the oracle here is the PIL
                  backend (torchvision is not installed in ~/x).
        - [ ] SD 2.x x4 upscaler follow-ups (TStableDiffusionUpscaler,
              `--upscale-image` / `--noise-level`,
              TestStableDiffusionUpscalerParity):
          - [ ] Real stabilityai/stable-diffusion-x4-upscaler run on the
                user's GPU box (e.g. 128x128 -> 512x512, `--gpu`): check its
                unet/vae/scheduler configs load (the pico folder mirrors the
                layout from memory of the real configs), time it, and compare
                the image with diffusers.
          - [ ] diffusers' upscale pipeline takes a batch of images and
                num_images_per_prompt; the conditioner takes one image.
      - [ ] SD MultiDiffusion follow-ups (`PanoramaWindowWidth/Height`,
            `PanoramaStride`, `PanoramaCircular`, `VaeTileSize` over
            `TMultiDiffusionWindows`; examples/StableDiffusion `--panorama`,
            `--tile-upscale`, `--controlnet`; covered by
            TestStableDiffusionPanoramaOneWindowParity / TwoWindows /
            HintCrop and TestStableDiffusionVaeTiling):
        - [ ] diffusers StableDiffusionPanoramaPipeline oracle on a two-window
              pico canvas (needs a new fixture script; ask before adding
              Python). Same views when the stride divides side - window;
              diffusers steps the scheduler per view (equal for the linear
              samplers) and draws ancestral noise per view (here: once per
              canvas).
        - [ ] Real SD 1.5 runs on the user's GPU box: a 2048x512 panorama
              (default stride 64 vs 128 / 256, with and without
              `--panorama-wrap`) and a 512 -> 1024 / 2048 `--tile-upscale`
              with lllyasviel/control_v11f1e_sd15_tile (quality, time, peak
              memory); README outputs.
        - [ ] Tiled VAE encode follow-ups (`TiledVaeEncode` on the
              `TiledNetCompute` planner; `EncodeVaeImage` tiles under
              `VaeTileSize`, so img2img, `--tile-upscale` and every SD
              conditioner that encodes an image use it):
          - [ ] Measure tiled-vs-whole encode error on a real SD 1.5 VAE (the
                pico encoder is untrained: about 0.24 x max latent away from
                seams on a periodic image, about 2 x on an aperiodic one).
          - [ ] The same on the real Qwen-Image-2.1 VAE
                (`TQwenImage21VaeEncoder.EncodeTiled`, default 256/192, runs
                for every img2img / inpaint / edit encode) and its encode
                memory at 1024x1024; and on a real Marigold checkpoint with
                `TMarigoldPipeline.VaeTileSize` (encode and decode; e.g.
                `examples/MarigoldDepth --vae-tile 512` at 768 processing
                resolution) against the untiled depth map.
        - [ ] Refused with windows today: DeepCache over several windows (one
              cache set per window), GLIGEN (boxes in image coordinates mapped
              per window), Prompt-to-Prompt, DDIM / null-text inversion.
        - [ ] T2I-Adapter over several windows reruns the adapter per window
              and step; keep per-window features if it shows in a profile.
      - [ ] SDXL: two text encoders (CLIP-L + OpenCLIP-bigG) concatenated + the
            pooled vector and time ids into SDUNetDenoiseSDXL; scaling 0.13025;
            the pipeline refuses addition_embed_type "text_time" today.
      - [ ] F16 weights: the SD UNet / VAE / CLIP loaders are unverified on F16
            and *.fp16.safetensors variant files (pick the variant by name).
      - [ ] CLIP tokenizer.json (fast) path follow-ups:
        - [ ] The post_processor is not read: BosId / EosId come from the
              <|startoftext|> / <|endoftext|> added tokens. A CLIP-family
              tokenizer.json with other bos/eos names gets no BosId.
        - [ ] HF matches normalized added tokens in the NFC + whitespace-
              collapsed text; Encode matches them in the lowercased raw text
              (differs only on decomposed input).
        - [ ] IsWhitespaceCP vs the Rust regex \s (Unicode White_Space) of the
              CLIP Split: only ASCII whitespace, U+00A0 and U+3000 are tested.
      - [ ] int8 quality on a real SD 1.5 checkpoint (UNet + CLIP int8 vs FP32
            image, user's GPU box; `UNetInt8` / `TextEncoderInt8`, `--int8`),
            then int4 UNet weights (QuantizeWeightsInt4, the QwenImage --int4
            path) after the int8 check.
      - [ ] Kept-pipeline size-change follow-ups (weight owners,
            BuildWeightOwner):
        - [ ] --gpu: the owners stay on the host, so each size change uploads
              the UNet weights (int8 codes or FP32) into the new net's own
              OpenCL context again. Arming the owners in one shared context
              (CreateOpenCLContextNet + ArmWeightOwnerInOpenCLContext, as the
              Qwen-Image VAE does) would make a size change upload nothing;
              measure on the GPU box first.
        - [ ] A different IP-Adapter file still re-reads the UNet (the owner
              holds the IP-Adapter attention weights).
      - [ ] Per-prompt caching of the cross-attention text K/V (low payoff,
            ~0.4% of a 512x512 SD 1.5 pass: 2*77*768*sum(C)=1.5 GMAC over the
            16 attn2 blocks vs ~400 GMAC). Needs a TNNet mechanism to skip the
            layers that depend only on the text input (AddSDTransformer2D's
            TSDAttnLayers.Blocks[].Cross.K / Cross.V) while it is unchanged, and two
            caches, because CFG alternates the cond and uncond text every step
            (or run CFG as one batch-2 pass).
      - [ ] Follow-ups of the ControlNet / T2I-Adapter / inpainting conditioners
            (`TStableDiffusionControlNet`, `TStableDiffusionT2IAdapter`,
            `TStableDiffusionInpainter` on `TStableDiffusionConditioner`;
            examples/ControlNetCanny, T2IAdapterSketch, DiffusionInpainting):
        - [ ] Real-checkpoint runs on the user's GPU box (--gpu, also --int8):
              lllyasviel/sd-controlnet-canny, TencentARC/t2iadapter_sketch_sd15v2
              and runwayml/stable-diffusion-inpainting (plus the blended mode on
              SD 1.5). The conditioners' OpenCL and int8 paths only ran on the
              pico (CPU) so far. These runs (and the Marigold one) also re-check
              the UNet config head-count fix on real checkpoints.
        - [ ] T2I-Adapter on a plain DownBlock2D that has a downsampler (the
              feature goes after the downsampler, in place on the last skip):
              coded from the diffusers source, no fixture has such a block.
        - [ ] Composed-conditioner follow-ups: (1) parity for ControlNet / T2I on
              img2img and inpainting (diffusers StableDiffusionControlNetImg2Img /
              Inpaint pipelines; the multi-conditioner runs them, untested, and
              diffusers' ControlNet-inpaint prepares its control image with the
              mask); (2) IP-Adapter + img2img / inpainting parity (code path
              shared with the tested InstructPix2Pix case); (3) a T2I MultiAdapter
              test (diffusers MultiAdapter adapter_weights).
        - [ ] Real-checkpoint run of a global_pool_conditions ControlNet
              (lllyasviel/control_v11e_sd15_shuffle).
      - [ ] Textual inversion on a real checkpoint: SD 1.5 with a
            sd-concepts-library learned_embeds.safetensors and an A1111 .pt
            (e.g. EasyNegative); only the pico fixture is tested.
      - [ ] Textual inversion: mixed-case tokens never expand to their N
            vectors (diffusers' maybe_convert_prompt looks the lowercased
            token up by the token as given; ported as is). Decide whether to
            expand them anyway (A1111 does) at the cost of diffusers parity.
      - [ ] CLIP lowercasing (LowercaseCP: the text, AddToken and normalized
            added tokens) covers only ASCII, Latin-1 and Latin Extended-A; HF
            lowercases all of Unicode (Greek, Cyrillic, ... differ). The
            added-token match needs the lowercase to keep the byte length.
      - [ ] Old A1111 .pt embeddings saved before torch 1.6 (legacy non-zip
            pickle) are rejected by TNNetTorchBinReader.
      - [ ] Prompt control in TStableDiffusionPipeline follow-ups. Long prompts
            (75-token chunks, lpw_stable_diffusion layout, MaxPromptChunkCount /
            `--max-prompt-chunks`, PromptNotice) and A1111 prompt weighting
            (ParseSDPromptWeights, PromptWeighting / `--no-prompt-weighting`,
            lpw whole-prompt mean restore) are DONE with pico parity vs a
            float64 reimplementation of lpw's get_weighted_text_embeddings
            (tools/make_pico_sd_prompt_weighting_fixture.py; the community
            pipeline is not in the ~/x venv). Open:
        - [ ] Run a long weighted prompt on real SD 1.5 (pico only so far).
        - [ ] A1111 extras not done: moving a chunk split back to a nearby
              comma, per-chunk mean restore (lpw restores the whole prompt's
              mean, which is what is coded).
      - [ ] IP-Adapter image prompts in TStableDiffusionPipeline follow-ups
            (TStableDiffusionIPAdapter, h94/IP-Adapter ip-adapter_sd15,
            StableDiffusion --ip-adapter/--ip-image/--ip-scale):
        - [ ] Run h94/IP-Adapter ip-adapter_sd15 on real SD 1.5 (only the pico
              fixture is tested).
        - [ ] IP-Adapter extras: set_ip_adapter_scale per-block dicts
              (style/layout), several adapters/images, ip_adapter_masks,
              ip-adapter-plus (Resampler), FaceID, SDXL.
        - [ ] Conditioner follow-ups from the 2026-10-22 batch:
              (1) cross-check CannyEdges against a real cv2.Canny run (cv2 is
              not in the ~/x venv; needs user approval to install it);
              (2) rerun tools/make_pico_sd_conditioning_fixtures.py in full and
              confirm the committed ControlNet / T2I / inpaint fixtures come out
              byte-identical after the --nonsquare-only / --inpaint-strength-only
              / --ip-adapter-only refactors;
              (6) img2img / inpaint parity covers only images already at the
              output size: add a fixture whose sides are not a VAE-factor
              multiple (diffusers resizes a PIL image with lanczos after uint8
              rounding, a tensor with torch interpolate) and check the resize.
      - [ ] InstructPix2Pix instruction-based image editing follow-ups
            (TStableDiffusionInstructPix2Pix, checkpoint timbrooks/instruct-pix2pix;
            examples/StableDiffusion --edit-image / --image-guidance):
        - [ ] Run timbrooks/instruct-pix2pix on the user's GPU box (only the pico
              fixture is tested; its F32 safetensors / fp16 variants unchecked).
        - [ ] InstructPix2Pix extras: an --edit-image whose
              sides are not a VAE-factor multiple (resize parity, see (6)
              above). PAG and SAG are refused with InstructPix2Pix (diffusers
              has no such pipeline).
      - [ ] FreeU follow-ups (`TNNetFreeUFourierFilter`,
            examples/StableDiffusion `--freeu`):
        - [ ] FreeU follow-up: a with/without image pair from the GPU box on a
              real SD 1.5 checkpoint (`--freeu sd15`) for the README section,
              and its UNet pass time with and without FreeU.
        - [ ] FreeU follow-up: an `sdxl` preset (0.6, 0.4, 1.1, 1.2 in the diffusers
              FreeU docs) once an SDXL pipeline CLI exists; BuildSDUNet already
              applies FreeU to SDXL configs (up blocks 0 and 1).
      - [ ] DeepCache follow-ups (`TStableDiffusionPipeline.DeepCacheInterval`,
            examples/StableDiffusion `--deepcache N --deepcache-branch B`):
        - [ ] GPU box: SD 1.5, 50 steps DDIM, `--deepcache 2/3/5` (branch 0 and
              the default) vs off: time per image and image pairs in
              examples/StableDiffusion/README.md.
      - [ ] Self-Attention Guidance for the SD UNet: the with/without image
            pair on SD 1.5 from the GPU box in examples/StableDiffusion/README.md
            (`--sag-scale 0.75`). The pipeline side is done
            (`TStableDiffusionPipeline.SAGScale`, `--sag-scale`,
            TestStableDiffusionSAG*: pico parity vs diffusers with reference
            numbers hardcoded from an ad hoc run; tools/make_pico_sd_pipeline_fixture.py
            has no SAG mode).
      - [ ] Prompt-to-Prompt editing follow-ups in TStableDiffusionPipeline
            (Hertz et al. 2022, arXiv:2208.01626; `--edit-prompt`,
            `--invert-image`, `--null-text`):
        - [ ] Prompt-to-Prompt on a real checkpoint (SD 1.5) on the user's
              GPU box: swap, refine and reweight from a seed and from an
              inverted photo; LocalBlend (the reference's mask that limits
              the edit to the edited words' region) is not ported. Also run
              `--invert-image --null-text` (with and without `--edit-prompt`)
              there: reconstruction quality at guidance 7.5.
        - [ ] Null-text inversion follow-ups (b5f693ec..59dcdab1):
          - [ ] OpenCL: with OpenCL on, `NullTextInversion` still builds a
                trainable FP32 UNet on the CPU; 50 steps x 10 inner steps is
                up to 500 CPU forward+backward passes. Error-only on the
                OpenCL UNet was not tried: check that the device forward
                leaves every backward cache (GroupNorm's normalized input,
                attention probabilities, ...) on the host, or add a device
                backward.
          - [ ] Decide: DDIM / null-text inversion encode through
                `EncodeVaeImage`, so with `VaeTileSize` set on a large image
                they now invert a TILED encode (a948d626). They refuse
                panorama windows but not `VaeTileSize`; either refuse it too
                or pin that the tiled latent inverts acceptably.
- [ ] Differential Diffusion (Levin & Fried 2024, arXiv:2306.00950): the SD
      1.x/2.x part is DONE (`TStableDiffusionDifferentialImg2Img`, standalone
      `DifferentialDiffusionRegenerateMask` in neural/neuraldiffusion.pas,
      examples/StableDiffusion `--change-map` / `--invert-change-map`; pico
      parity vs exx8 SD2/diff_pipe.py's loop with DDIM, Euler, DPM++ 2M Karras,
      Euler Karras). The convention is white = change most; the reference's
      own maps are white = keep (its `map > i / N` keeps the image). Open:
      - QwenImage DONE: `TQwenImage21Pipeline.EncodeChangeMap` /
        `Denoise(..., ChangeLatents)`, examples/QwenImage `--change-map`,
        `--invert-change-map`, `/change-map`, `/invert-change-map`; the
        threshold index follows the Flux differential reference (after run
        step i blend with 1 - change > i / N, never after the last step);
        invariants in `TestQwenImage21ChangeMap`. Open:
        - [ ] Real Qwen-Image-2.1 run on the user's GPU box (horizontal
              gradient at strength 1 and 0.6).
        - [ ] Optional pico parity: an ad hoc ~/x venv loop of
              QwenImage21Pipeline img2img plus the Flux reference's blend
              (feeding our FPC noise), hardcoded into a test; today only
              invariants pin the QwenImage side.
        - [ ] SD and QwenImage use different step indices (SD: the exx8 SD2
              before-step-i convention, threshold (i+1)/N after step i;
              QwenImage: Flux's i/N) and different latent resizes (SD
              bilinear, QwenImage nearest as Flux): keep or unify only after
              a real side-by-side run.
      - [ ] Real checkpoint run (SD 1.5) on the user's GPU box: a horizontal
            gradient and a depth-derived map.
      - [ ] The 4-channel `TStableDiffusionInpainter` blend still refuses the
            continuous-sigma schedules (`AcceptsSigmaSchedule` = 9-channel
            only); `AfterStep` now gets the next noise level on both paths
            (sqrt(ab), sqrt(1-ab)), so it could accept them, after a pico
            parity vs diffusers' inpaint pipeline with Euler Karras.
- [ ] IC-Light relighting on the SD 1.5 pipeline (lllyasviel/IC-Light,
      Zhang et al. ICLR 2025, `gradio_demo.py` / `gradio_demo_bg.py`):
      `TStableDiffusionICLight` (`Relight`) and examples/StableDiffusion
      `--relight PNG --iclight FILE` landed; open:
      - [ ] (f) Real run on the user's GPU box (fc and fbc, `--int8` too). The
            img2img start / highres pass now follow the truncated Karras
            schedule (fed30990), so check the highres output there.
      - [ ] (g) A whole `Relight` parity test with DPM++ 2M SDE Karras (the
            example's default) against the gradio reference; today the four
            IC-Light tests sample with DDIM.
      - [ ] (h) `gradio_demo_bg.py` was not checked: the fbc low-res pass is
            assumed txt2img, guidance 2.0 is used for fbc (its default may be
            7.0), and its own light presets (gradients, grey) are not ported.
- [ ] Karras / continuous-sigma img2img follow-ups (fed30990, e9fe925f): pico
      parity vs a diffusers oracle for img2img with Euler linspace, Euler a
      Karras (per-step ancestral noise) and 9-channel inpainting below
      strength 1 with Karras (only DPM++ 2M, SDE and Euler Karras are pinned
      against diffusers; `TestImg2ImgSigmaScheduleTruncation` pins the t_start
      cut by hand). Needs a new oracle generator; no Python was added for
      it (ask the user how to produce it). The flow-match `sde-dpmsolver++` step is pinned only
      by hand-computed and invariant tests, with no diffusers trajectory, and the
      Qwen-Image pipeline draws its SDE noise from RandG (no step-noise hook).
- [ ] GLIGEN on a real checkpoint (`masterful/gligen-1-4-generation-text-box`)
      on the user's GPU box: boxes land where asked, timing with the fusers on
      vs after the gates close; and `gligen-1-4-inpainting-text-box` with
      `--gligen-inpaint-image` (only the boxes change).
- [ ] GLIGEN text-image variant (`gligen-1-4-generation-text-image-box`:
      `linears_text` / `linears_image`, CLIP image embeddings as grounding).
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
- [ ] Mask2Former universal-segmentation importer follow-ups
      (BuildMask2FormerFromSafeTensors[Ex], model_type "mask2former";
      masked-attention decoder + per-query mask/class heads, RunMask2FormerSemantic +
      DecodeMask2FormerSemantic, TestMask2FormerParity ~1.2e-7 < 1e-4,
      examples/UniversalSegmentation). Deferred follow-ups:
      (a) wire the full Swin backbone + FPN pixel decoder (mask_features + 3 multi-scale
          memory levels with level_embed + sine pos) into ONE forward — v1 takes the
          pixel-decoder outputs as PRECOMPUTED TNNetInput feature maps (mirroring Mask
          R-CNN v1's FPN-input stance); the masked-attention decoder + heads are complete;
      (c) multi-head cross-attention beyond the pico 1-head fixture is wired generically
          (Heads loop) but only parity-verified at Heads=1;
      (d) a real-checkpoint smoke once a swin-tiny-semantic checkpoint is obtainable
          offline (the importer is config-driven and ready).
      (f) the pico fixture's `semantic_map` is class 3 everywhere, so
          `TestMask2FormerParity`'s semantic check is weak (the synthetic cases in
          `TestMask2FormerSemanticDecodeReference` carry it); a fixture with varied
          output would make the real-forward check meaningful.
- [ ] NAFNet image-restoration importer follow-ups (BuildNAFNetFromSafeTensors[Ex]
      + TNAFNetConfig LANDED — TNNetSimpleGate gate layer + Simplified Channel
      Attention + U-Net of NAFBlocks with LayerNorm2d / depthwise 3x3 / PixelShuffle;
      pico parity TestNAFNetParity < 1e-4 vs a numpy float64 oracle;
      examples/ImageRestoration end-to-end CPU denoise):
  - [ ] Real official NAFNet denoise/deblur checkpoint parity (offline env — the
        weights are large / not redistributable).
  - [ ] NAFBlock dropout + the wider official width/block configs are wired by
        config but only the small pico shape is parity-pinned.
- [ ] SwinIR transformer image-restoration importer follow-ups
      (BuildSwinIRFromSafeTensors[Ex] + TSwinIRConfig LANDED — RSTB of Swin
      window/shifted-window attention + 3x3 conv residual + pixel-shuffle upsample
      tail, reusing the landed Swin attention blocks, NO new leaf layers; pico parity
      TestSwinIRParity / TestSwinIRConfigFromJSONFile < 1e-4 vs a float64 numpy
      oracle; examples/SwinIRRestore 2x SR smoke; the `nearest+conv` x4,
      upscale-1 denoise, two-step `pixelshuffle` x4 and lightweight
      `pixelshuffledirect` x3 tails are pinned by TestSwinIRNearestConvParity,
      TestSwinIRDenoiseParity, TestSwinIRPixelShuffleX4Parity and
      TestSwinIRPixelShuffleDirectParity):
  - [ ] Real-checkpoint parity deferred (official SwinIR .safetensors are large /
        not obtainable offline). Every tail can run now, e.g. the `nearest+conv`
        real-world x4 checkpoint (003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth)
        and the denoise checkpoints. They would also confirm
        d6d3c6d2's LeakyReLU 0.01 (conv_before_upsample) and erf-GELU changes.
- [ ] Restormer image-restoration follow-ups (`TNNetTransposedChannelAttention`,
      7ded244f; `BuildRestormerFromSafeTensors[Ex]` + `TRestormerConfig` +
      examples/Restormer, c7363ab4; pico parity `TestRestormerParity`,
      `TestRestormerDualPixelBiasFreeParity`, `TestRestormerTiledOpenCL` on PoCL):
  - [ ] Real-checkpoint run on the user's box (no official file was run;
        the presets' LayerNorm/bias flags come from the official options
        yml, and the builder rejects a preset that disagrees with the
        checkpoint).
  - [ ] Dual-pixel defocus follow-ups (examples/Restormer `--left/--right` +
        `RestormerPackDualPixel`, cee1d880):
    - [ ] `--input` (single-image tasks) still reads and writes 8 bits; check
          whether the official single-image defocus script reads DPDD at 16.
    - [ ] Not run on a real DPDD pair (needs `dual_pixel_defocus_deblurring.pth`).
  - [ ] GPU timing at 1-4 MP tiles, then coalescing `cai_transposed_attn_dots`,
        which reads q|k with a 3C stride per position (uncoalesced).
- [ ] RIFE video frame-interpolation importer follow-ups (BuildRIFEFromSafeTensors[Ex]
      + TRIFEConfig + TNNetBackwardWarp LANDED — IFNet intermediate-frame synthesis
      with accumulated flow residuals + sigmoid fusion-mask blend; pico parity
      TestRIFEParity < 1e-4 vs a float64 numpy oracle;
      examples/VideoFrameInterpolation --factor N / --t VALUE, inference-only, on CPU):
  - [ ] Real Practical-RIFE checkpoint parity (offline-download deferred): the real
          IFNet is MULTI-SCALE (stride-2 IFBlocks downsample then bilinearly upsample
          the predicted flow) and RE-FEEDS the warped frames + previous flow/mask into
          each successive block; v1 refines from the SAME [frame0|frame1] input at full
          resolution. Wire the scale schedule + warped-frame re-feeding and verify
          against an exported hzwer/Practical-RIFE checkpoint.
  - [ ] Timestep-plane follow-up (arbitrary t and the recursive Nx schedule,
        7322d80d):
    - [ ] Verify the timestep-plane path on a real Practical-RIFE v4 checkpoint
          (needs the multi-scale item above; v4 also feeds t to every refining
          block alongside the warped frames, which v1 does not re-feed).
  - [ ] Privileged-distillation teacher (the training-time IFNet that sees the GT
        middle frame); inference-only path is landed.
- [ ] GFPGAN follow-up (`BuildGFPGANFromSafeTensors[Ex]` + examples/FaceRestoration
      landed; pico parity and OpenCL parity only): run GFPGANv1.3.pth and
      GFPGANv1.4.pth on real aligned 512x512 faces (CPU and `--gpu`) against the
      official `gfpgan` inference output before README calls it supported.
- [ ] Whole-photo face restoration (port of facexlib `FaceRestoreHelper`, what
      `gfpgan`'s inference_gfpgan.py runs by default): the RetinaFace detector,
      5-point alignment, paste-back and the `FaceRestoration --detector`
      whole-photo mode are done; open:
  - [ ] Run the real detection_mobilenet0.25_Final.pth / detection_Resnet50_Final.pth
        on a real photo against facexlib `detect_faces` (oracle in the ~/x venv:
        facexlib 0.3.0 is not installed there; the wheel's retinaface*.py run with
        torchvision from /tmp/tvlib and a cv2 stub).
  - [ ] ParseNet paste mask (`BuildParseNet[FromSafeTensors[Ex]]`,
        `ParseNetFaceMask`, `ParseNetFaceMaskOfFace`, `RestoreFacesInImage(...,
        Parser)`, `FaceRestoration --parse`) is CODED with pico parity vs the
        facexlib 0.3.0 parsenet.py + cv2 mask (tools/parsenet_tiny_fixture.py);
        open: run the real `parsing_parsenet.pth` (v0.2.2 release) on a real
        restored 512x512 face against facexlib's `face_parse` + mask (argmax
        flips at near-tied logits are expected to be rare).
  - [ ] Real-checkpoint parity (needs the downloads; oracle in the ~/x venv with
        the facexlib 0.3.0 + gfpgan 1.3.8 wheels unpacked under /tmp and
        opencv-python-headless 4.10 unpacked to /tmp/cvlib4, all on sys.path
        via `python -I` + sys.path.insert, torchvision from /tmp/tvlib):
        (a) `FaceRestoration --model GFPGANv1.4.pth --detector
        detection_Resnet50_Final.pth --input photo.jpg --output out.png
        --save-faces f --noise const --upscale 2` on a real group photo; in
        Python run `FaceRestoreHelper(2, det_model='retinaface_resnet50',
        use_parse=False)` -> read_image -> get_face_landmarks_5(eye_dist_threshold=5)
        -> align_warp_face and compare `cropped_faces[i]` (BGR uint8) with
        `f_cropped_NN.png` (expect ~0.3 gray levels mean: cv2 rounds to uint8 and
        to 1/32-px positions) and the landmarks; (b) feed OUR restored faces
        (`f_restored_NN.png`, RGB->BGR) through add_restored_face ->
        get_inverse_affine -> paste_faces_to_input_image and compare the
        composite with out.png (isolates the paste from GFPGAN noise; expect
        < 1 gray level mean, facexlib truncates to uint8); (c) the full
        inference_gfpgan.py composite (`--parse parsing_parsenet.pth`, use_parse=True).
- [ ] CodeFormer follow-ups (Zhou et al., NeurIPS 2022, arXiv:2206.11253;
      sczhou/CodeFormer; S-Lab License 1.0, non-commercial, so state it in the
      README row). The importer core LANDED in neuralpretrained.pas:
      `TCodeFormerConfig` (shape fields from InferCodeFormerConfigFromTensors;
      n_head / norm_num_groups / fidelity_weight / adain from an optional
      config.json, default 8 / 32 / 0.5 / true), `BuildCodeFormer[FromSafeTensors[Ex]]`
      (reads `codeformer.pth`, `params_ema` unwrapped), `CodeFormerSetFidelity(NN, w)`
      (no rebuild; the w of each Fuse_sft_block is a TNNetMulByConstant),
      `CodeFormerRestoreFace` and the shared `FaceRestorerPrepareInput`. One TNNet,
      stock layers only: top-1 = softmax -> TNNetTopKGate(1) (exact one-hot) ->
      bias-free 1x1 conv holding the codebook; adain = TNNetAdaIN (on by default,
      as inference_codeformer.py). Pico parity vs the official arch files
      (tools/codeformer_tiny_fixture.py, tests/fixtures/tiny_codeformer*):
      TestCodeFormerParity (w = 0.7 1.5e-5, w = 0 5e-6, codes, round trip),
      TestCodeFormerNoAdaINParity (1.1e-5). Also LANDED: `RestoreFacesInImage`
      takes a restorer face size (`FaceRestorerRestoreFace`; overloads for
      `TGFPGANConfig` / `TCodeFormerConfig`), `IsCodeFormerCheckpoint`,
      examples/FaceRestoration `--model codeformer.pth --fidelity W` (both modes,
      `--gpu`), TestCodeFormerOpenCLParity (size verdict + ForceOpenCL, every
      layer) and TestRestoreFacesInImageCodeFormer. OPEN: real `codeformer.pth`
      (CPU and `--gpu`, aligned and whole-photo) on the user's box before the
      README calls it supported. A Fuse_sft_block at the latent level (the official table puts
      '16' after the 2nd ResBlock) is rejected; codeformer.pth does not use it.
  - [ ] `--int8` on CodeFormer is unchecked: quantization error can flip the
        transformer's top-1 code choices. Compare the code ids and the restored
        face of `--int8` against FP32 on the real `codeformer.pth`.
- [ ] DDColor follow-up (`BuildDDColorFromSafeTensors[Ex]` + examples/ColorizePhoto
      landed; pico parity and OpenCL parity only): run `piddnad/ddcolor_paper_tiny`
      and `ddcolor_modelscope` on real grayscale photos (CPU and `--gpu`) against
      the official `basicsr` inference output before README calls it supported.
- [ ] LPIPS follow-ups — the metric LANDED (ComputeLPIPSDistance /
      LPIPSStageDistance / LPIPSUnitNormalize in neuralpretrained.pas, reusing the
      VGG importer's 5 relu taps; unit-normalize -> squared-diff -> per-stage lin
      head -> spatial-mean -> sum; parity test TestLPIPSDistanceParity vs a numpy
      float64 oracle, LPIPS(x,x)=0). The lin head is a LOADABLE per-stage weight
      vector; nil = the unweighted (1/C channel-mean) lpips lin_layers=False
      baseline (what the oracle pins). OPEN: (a) ship/import the OFFICIAL
      richzhang/PerceptualSimilarity lin{0..4}.model 1x1-conv weights + parity vs
      the reference calibrated LPIPS (needs the `lpips` weights, unobtainable
      offline here). (b) TNNetPerceptualLoss (a TNeuralFit.OutputErrorFn hook)
      is wired into SuperResolutionTrain as --perceptual-vgg FILE
      [--perceptual-weight W]. OPEN from (b): (b1) a real-VGG SR training
      run (CIFAR-10 + timm/vgg16.tv_in1k safetensors) to pick a default
      --perceptual-weight and check the timm key layout loads; (b2) calibrated
      unit-normalized (LPIPS-style) perceptual gradient; (b3) per-thread VGG
      clones instead of the serializing lock if the perceptual step dominates.
- [ ] ControlNet spatial-conditioning importer follow-ups
      (BuildControlNetFromSafeTensors[Ex], lllyasviel/sd-controlnet-canny;
      hint stem + per-resolution zero-conv residual taps + ControlNetResiduals driver,
      TestControlNetParity 2.89e-6 < 1e-4). First conditioning-by-feature-injection model.
      DEFERRED follow-ups: (a) real-checkpoint slicer (slice sd-controlnet-canny like
      slice_llama.py) for parity vs real weights on top of the synthetic fixture;
      (b) T2I-Adapter (BuildT2IAdapterFromSafeTensors[Ex])
      follow-ups: real-checkpoint slicer (TencentARC/t2iadapter_*_sd15v2),
      light_adapter variant.
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
      int8 weight-only storage instead of the per-row dequantize-then-requantize
      of the row-streaming loader (Q4_0 already loads straight into int4 rows).
- [ ] FP16 (half-precision) OpenCL compute path for the FP32-weight
      dot-product/matmul offload. PARTLY LANDED: `--experimental-fp16` narrows the
      activation (B operand) and the split-K partials to half for the int8
      GEMV/GEMM (`cai_dot_product_int8_h`, `cai_f32_to_half`, `cai_im2col_h`);
      user-measured +10.2% decode on the 27B. That path uses half as STORAGE only,
      read through `vload_half` (core OpenCL, no `cl_khr_fp16`), so it runs and is
      parity-tested on the PoCL device here (TestInt8ConvFP16OpenCLParity). What
      remains is the non-quantized nets: every FP32 GEMM kernel in
      `neural/neural.cl` (cai_dot_product, simpleGEMMT, myGEMM5/6) still reads
      FP32 weights. For GPU LLM decode the binding constraint is device memory
      bandwidth, so half device-side weight/input buffers (vload_half storage,
      FP32 accumulate) would roughly halve buffer traffic. Scope: add an opt-in
      FP16 buffer/kernel path in neuralopencl.pas wired through the existing
      TDotProductSharedKernel offload (reuse the NewVAs/weight-residency
      machinery), keep the FP32 kernel as the default, and add a numerical test
      that checks the FP16 result matches the FP32 path within a half-precision
      tolerance on a fixed fixture. A `cl_khr_fp16` half-ARITHMETIC variant is
      optional and cannot be tested here (PoCL does not advertise the
      extension). Distinct from the int8 work and from CPU FP16/BF16 weight
      storage (still open in the quantized-inference item above).
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
- [ ] Preference-optimization follow-up on TNeuralDPOTrainer
      (neural/neuraldpo.pas, TNeuralPreferenceLossMode plmKTO, paired-batch
      simplification documented in the unit header): a full unpaired-batch KTO
      variant (current KTO uses the paired pair-mean KL point).
- [ ] Sequence-length warmup curriculum in neuralfit.pas: train at short
      context first and grow SeqLen on a schedule (rebuild the same
      architecture at a new width) — large early-epoch throughput win for LM
      pretraining. Needs CopyWeights across width rebuilds + a schedule
      hook; the landed TNeuralFitCallback (AddCallback, neuralfit.pas) is the
      natural home. Test: weights survive a width hop bit-for-bit, loss
      continuous across the hop.
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
  - [ ] EvaluateMMLU large-model path: batch questions through the left-padded
        DecodeBatchGreedy / KV-cache batch scoring instead of one
        ScoreSingleTokenCandidates forward per question (already one forward per
        question, not per letter; still slow at the full 14k-question x
        57-subject scale).
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
      + the `TNNetHiFiGAN` channel-major holder; conv weights fold weight_norm
      g/v at import; parity-gated only on the pico fixture,
      TestHiFiGANSynthesisParity). Open follow-ups:
  - [ ] real-checkpoint smoke: resynthesize a clip with a downloaded `hifigan` /
        SpeechT5HifiGan generator (weight_norm fold path) and write it via
        SaveVolumeToWav16 (offline + RAM-gated here, so deferred).
- [ ] OpenCL acceleration for the channel-major conv1d inner loops in the
      self-contained audio holders. These holders deliberately do the conv1d /
      ConvTranspose1d math DIRECTLY on channel-major arrays (not via TNNet
      layers). REMAINING:
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
- [ ] VITS / MMS-TTS text-to-speech importer follow-ups
      (`BuildVitsFromSafeTensors[Ex]` + the `TNNetVits` channel-major holder +
      examples/TextToSpeech; model_type `vits`: facebook/mms-tts-* and
      kakao-enterprise/vits-ljs; parity-gated only on the pico fixture,
      `TestVitsSynthesisParity`). Open follow-ups:
  - [ ] MULTI-SPEAKER models (`num_speakers>1` / `speaker_embedding_size!=0`, the
        global-conditioning `cond` convs into the WaveNet/duration/decoder) —
        rejected loudly.
  - [ ] real-checkpoint smoke: synthesize a sentence with a downloaded
        `facebook/mms-tts-eng` / `kakao-enterprise/vits-ljs` and write it via
        `SaveVolumeToWav16` (offline + RAM-gated here, so deferred).
- [ ] Kokoro-82M TTS (`hexgrad/Kokoro-82M`) follow-ups. The real checkpoint loads
      and speaks 24 kHz WAV (user-tested 2026-10-14, examples/KokoroTTS); stage A
      runs as `TNNet`s, stages B/C as plain code (`neuralplainops.pas`). Open
      layer follow-ups found during the port:
  Phase 0 — oracle and structure:
  - [ ] 0.4 Layers for the `TNNet` route:
        - [ ] `ShouldUseInterleavedDotProduct` picks `InterleavedDotProduct` for
          every conv with a multiple of 32 neurons; on 1D convs it is slower than
          `DotProductsTiled`: same conv at 250 ch (tiled) 199 ms serial / 89 ms
          threaded vs 256 ch (interleaved) 260 / 161-189 ms. Kokoro stage A's
          128/256/512 channel convs take the interleaved path. Measure on 2D convs
          too before changing the heuristic.
        - [ ] OpenCL weight borrowing (`BuildWeightOwner` + `EnableOpenCLInContextOf`)
          has no direct unit test for `TNNetConv1DLinear` (the CPU borrowing build
          has one); only the stage-A integration test `TestKokoroStageAOpenCLParity`
          covers it, and that test asserts only that some layer ran on OpenCL.
        - [ ] The unused backward variants (`BackpropagateCPU`, `BackpropagateFastCPU`,
          `BackpropagateFastCPUDev`, `BackpropagateOpenCL`) ignore `FPaddingY`
          (they raise when the layer is dilated); only `BackpropagateFastTiledCPU` is called.
        - [ ] `cai_conv_transpose` is untiled (one work-item per output cell and
          channel, like `cai_conv_implicit`); a large dense deconv (Kokoro
          generator ups) may want a tiled or phase-split GEMM. Not timed on a GPU.
        - [ ] Trainable deconvolutions and activations without a fused opcode
          (anything but Identity/ReLU/Sigmoid/Tanh/Swish/HardSwish) run on the
          CPU under OpenCL; the old host-scatter OpenCL path served trainable ones.
        - [ ] The new `cai_group_norm` and the OpenCL residency of the norm /
          activation chain are unmeasured on a GPU (PoCL parity only).
  - [ ] 0.5 Stage-net holder for Kokoro (stage A: `TNNetKokoro.PrepareStageA`;
        stages B/C run as plain code). Open:
        - [ ] `TNNetEmbedding` sizes its table in its constructor (before the net
          is known), so a borrowing build allocates and frees each table: 3
          tables, 114,176 floats on the real checkpoint. Other layers size none.
        - [ ] CPU profile of the real stage A: re-profile on the real checkpoint
          (not run). Before: `TNNetLSTMCell` ~51% of layer time,
          `TNNetPointwiseConvLinear` ~44%. The LSTM/GRU CPU forward now computes
          the input projection of every step as one `DotProductsPacked` GEMM;
          synthetic bi-LSTM SeqLen 100, In 640, Hidden 256: ~19 -> ~8 ms.
        - [ ] `TNNetLSTMCell` / `TNNetGRUCell` backward still does the input
          gradient (W_i^T dGates) and the W_i / W_h weight gradients as per-step
          rank-1 `MulAdd`s. Batching them needs a [SeqLen][4*Depth] dGates
          scratch and a transposing variant of `PackDotProductRows` (packed A^T
          for dX = dGates W_i, X^T / H^T for the weight gradients).
        - `TNNetVolume.AddDotProductRows` follow-ups (c4ef0979: the AVX2 4-row
          recurrent GEMV in `TNNetLSTMCell`, `TNNetGRUCell`, `TNNetSLSTMCell`,
          `TNNetEchoStateReservoir` and `TNNetPlainOps.RunLSTMDirection`):
          - [ ] No plain-AVX / 32-bit asm body: those builds take the per-row
            `DotProduct` path (unchanged speed).
          - [ ] The recurrent GEMV runs on one thread (no row split over the
            pool); not measured (1-core box).
          - [ ] Other per-row GEMV loops that fit `AddDotProductRows`:
            `TNNetPlainOps.ComputeStyleChannelAffine` (gamma/beta rows x
            style), the per-step q/k/v projections of `TNNetMLSTMCell`,
            `TNNetDeltaNet`, `TNNetClosedFormContinuous`.
        - [ ] Borrowing builds (stage A): each borrower cell packs its own
          `FInputRowsPacked` copy of W_i* (~2.6 MB per Kokoro cell, 10 cells) on
          its first forward after every `PrepareStageA`. The owner could hold
          one packed copy, as `ShareOwnerWeightCaches` does for the conv cache.
          Under OpenCL the CPU packed copy is sized but unused.
        - [ ] The input projection (`TNNetInputProjectionCell`, base of the
          LSTM/GRU/sLSTM/minGRU/minLSTM cells) runs on one thread (no position
          split over the pool); not measured (1-core box).
        - [ ] `TNNetMLSTMCell`, `TNNetDeltaNet` and `TNNetClosedFormContinuous`
          still compute their input projections as per-step dot products; they
          do not fit `TNNetInputProjectionCell` as is (q/k/v rows with a scale,
          a beta row, non-contiguous gate neurons). `TNNetConvLSTMCell`'s input
          convolution is recurrence-independent too.
        - [ ] Real-checkpoint stage A on a GPU: not run (PoCL parity on the tiny
          fixture only). The AdaLayerNorms after the bi-LSTMs run on the CPU
          unless forced, because the LSTM output is on RAM.
  Phase 2 — components, in graph order:
  - 2.1 ALBERT follow-up:
        - [ ] `bfAlbert` through `BuildBertFromSafeTensorsWithConfig` (HF
          `albert-*` checkpoints) is untested; no config reader for model_type
          "albert"; pooler / classifier / int8 refused.
  - 2.9 Forward complex STFT follow-ups:
        - [ ] `ComputeWhisperLogMel` runs `RealFramePowerSpectrum`'s two-factor
          path at N = 400; in the `TNNetForwardSTFT` A/B of fbe4661e that path
          (with complex output) was 2.6x slower than an AVX direct DFT over
          windowed tables (400: 118 vs 46 ms). Not measured in log-mel itself.
        - [ ] `TFFTPlan` is scalar Double; a Single AVX radix-2 would move the
          STFT (and inverse STFT) crossovers down.
        - [ ] The harmonic source runs scalar `pcr_sinf` (arguments up to ~4e4
          rad need its exact range reduction; the vector `Sin` was not checked
          at that range) on one thread; 41 ms total is small next to the nets.
  - 2.11 Generator follow-up:
        - [ ] `ISTFTOverlapAdd` (magnitude + phase, +sin sign) still builds its
          Re/Im volumes per call; no hot caller.
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
      + examples/ZeroShotAudioTag — LANDED: HTS-AT Swin audio tower over a log-mel
      image (the HF `BatchNorm2d`(mel) + `reshape_mel2img` folded into
      `ClapBatchNormMelImage`, caller supplies the `(time,mel,1)` image) + RoBERTa
      text tower; pico parity `TestClapParity` (`tools/clap_tiny_fixture.py`).
      Follow-ups:
  - [ ] "fused" CLAP (`enable_fusion = true`, clap-htsat-FUSED): the local/global
        mel-fusion patch-embed (`mel_conv2d` + the attention-feature-fusion block) is
        rejected. Distinct windowing — own importer branch + fixture.
  - [ ] Real laion checkpoint parity + the log-mel frontend wiring: `ClapFeatureExtractor`
        produces the 1024x64 mel (freq_ratio=4 is supported, 7bd3e0cd); wire `neuralaudio.pas`
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
- [ ] examples/SpeechCommands keyword-spotting trainer follow-up (LANDED:
      examples/SpeechCommands + scripts/download_speech_commands.sh; the default
      no-network smoke trains a 10-class confusable synthetic set through the real
      ComputeWhisperLogMel to ~99% held-out accuracy; `--full <dir>` loads real
      Speech Commands WAVs via LoadWavResampledToVolume, documented but NOT
      exercised by the smoke). Open follow-ups:
  - [ ] run the `--full` path on real downloaded Speech Commands v2 and record a
        real-data accuracy number (the synthetic smoke proves the path, not a
        hard benchmark).
- [ ] Whisper word-timestamp follow-ups (v1 landed, scoped to one 30 s
      greedy window). REMAINING:
  - [ ] (b) multi-window stitching for clips > 30 s (carry the running time offset
        and merge the per-window word lists).
  - [ ] (c rest) read `generation_config.alignment_heads` from arbitrary model
        configs when present (v1 still hardcodes the curated tiny/base/small/medium/
        large head lists and falls back to all-heads otherwise; no config-plumbing
        + fixture yet).
- [ ] Speaker diarization importer (`BuildPyannoteSegmentationFromSafeTensors[Ex]` +
      `TPyannoteConfig`/`ReadPyannoteConfigFromJSONFile`, model_type `pyannote`) — LANDED
      (SincConv front-end + bidirectional `TNNetLSTMCell` trunk + per-frame powerset
      head + `PyannotePowersetDecode`; pico parity `TestPyannoteParity` vs a numpy
      float64 oracle, since the `pyannote.audio` python package is NOT installed
      here; examples/SpeakerDiarization). Open follow-ups:
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
      (BatchNorm1d eval-mode stats, SpeechBrain TDNN naming) + a VoxCeleb EER smoke
      (network/RAM-gated).
- [ ] SAM mask decoder as a real TNNet layer graph (TNNetCrossAttention two-source
      wiring) instead of the plain-array RunSAMMaskDecoder forward, so the decoder is
      trainable / fine-tunable end-to-end (v1 is inference-only). Needs a builder that
      threads the per-step query/key positional-embedding additions through the
      asymmetric token<->image cross-attentions.
- [ ] SAM end-to-end processor follow-ups (TSAMPredictor LANDED: SAMPreprocessImage /
      SAMScaleCoordinates / SAMPostprocessMasks match SamImageProcessorPil + SamModel
      (TestSAMProcessorEndToEndParity); examples/SegmentAnything takes --image/--point/--box):
  - [ ] Run TSAMPredictor on a real facebook/sam-vit-base checkpoint + photo
        (only the random pico fixture is tested).
  - [ ] transformers 5.x defaults SamImageProcessor to the torchvision backend
        (tensor resize); the port matches the PIL backend (uint8 PIL bilinear).
        Measure the pixel_values gap on a real photo; add a torchvision-mode
        resize only if it changes masks.
  - [ ] Not exposed (the decoder lacks it): the input_masks prompt (previous
        low-res mask fed back).
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
  - [ ] Step 3 — add the real T5 tower (BuildT5FromSafeTensors) over a tokenized
        prompt, so the existing `--cfg` loop gets a real empty-caption uncond
        branch (today uncond is zero T5 states); ulimit-bounded demo that
        generates one small image. The CV-generative-stack-composes capstone.
        (Real PixArt/VAE checkpoints already load from the command line; the T5
        states are still DETERMINISTIC SYNTHETIC.)
  - [ ] Follow-up — the pico PixArt + matched pico VAE are RANDOM (smoke only);
        once a real checkpoint is wired (Step 3) re-verify the chain produces a
        sensible image, and expose the existing `smEulerAncestral` sampler /
        `tsKarras` spacing (neuraldiffusion.pas) as example flags.
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
- [ ] DBNet text detection (Liao et al., AAAI 2020, arXiv:1911.08947; the docTR /
      PaddleOCR / mmocr detector) + examples/DocumentOCR follow-ups. LANDED: docTR
      db_mobilenet_v3_large / db_resnet50 importer + DB post-processing (c71d2c74,
      pico parity 6e-8) and examples/DocumentOCR (a1697f45, DBNet boxes -> reading
      order -> TrOCR per box); CODED, not supported. Open:
  - [ ] Real docTR checkpoint: convert db_mobilenet_v3_large (and db_resnet50) to
        .safetensors in the ~/x venv and confirm the tensor names, which were
        recalled from docTR's source and never checked against a released checkpoint
        (feat_extractor.* IntermediateLayerGetter keys, fpn.in_branches /
        out_branches, prob_head.0/.1/.3/.4/.6); fix the importer if docTR's names
        differ. Then run `tools/dbnet_tiny_fixture.py --real CKPT` for a
        probability-map parity < 1e-4 at 1024x1024, then a scanned page and a street
        photo before README.md lists it as supported. Also check docTR's PreProcessor
        resize against `DBNetPrepareInput` on a large downscale (the pico case only
        downscales 120x90 -> 85x64), and docTR's padding removal against
        `DecodeDBNetTextBoxes`'s pixel mapping.
  - [ ] Real run: docTR db_mobilenet_v3_large + microsoft/trocr-small-printed on a
        scanned page and a street photo, after the docTR checkpoint item above,
        before README.md lists DocumentOCR as supported.
  - [ ] Tokenizer: check that the trocr-small `sentencepiece.bpe.model` (fairseq id
        layout, possibly offset by one) decodes correctly through `--tokenizer`, or
        that a tokenizer.json is available; fix the offset in the loader if not.
  - Word boxes -> line crops (`MergeDBNetTextBoxesIntoLines`, DocumentOCR
    `--merge-lines`) follow-up:
    - [ ] Compare the character accuracy of word-by-word vs `--merge-lines` crops
          on a real page (needs the real-run item above).
  - Rotated boxes (docTR assume_straight_pages=False): `DecodeDBNetRotatedTextBoxes`
    / `DetectDBNetRotatedTextBoxes` (convex hull `ConvexHullOfPoints`, rotating-
    calipers `MinAreaRectOfConvexHull`, the rectangle offset by D = hull area x
    unclip_ratio / hull perimeter via `OffsetRotatedRect`), upright crops by
    `CropRotatedRect`; DocumentOCR `--rotated`. Differences from docTR, none
    checked against a docTR oracle yet:
    - [ ] docTR offsets the CONTOUR polygon with pyclipper (round joins, area and
          perimeter of the possibly concave contour), then cv2.minAreaRect; here the
          hull's min-area rectangle is offset directly. Same for convex regions up
          to the orientation choice and the arc approximation; a concave region
          (curved text) gets a different D. Port a polygon offset if a real page shows it.
    - [ ] CropRotatedRect rounds the crop size and maps corner 1 to W - 1; docTR
          truncates (int(d1), int(d2)) and maps it to d1 - 1 (sub-pixel difference).
    - [ ] Not ported: cv2.findContours RETR_EXTERNAL drops a region that sits inside
          another region's hole; the labelling keeps it (both decoders).
    - [ ] Real rotated page (street photo) through `--rotated`, after the real-run item.
- [ ] EfficientNet (timm / torchvision efficientnet_b0..b7) importer follow-up
      (BuildEfficientNetFromSafeTensors LANDED, commit d35970c; AddEfficientNetMBConv
      reusing MobileNetV3 MBConv + conv-BN fold, b0..b7 share one JSON-driven builder,
      TestEfficientNetImageClassificationParity max|diff| < 1e-4 vs a numpy float64 oracle):
  - [ ] real torchvision/timm efficientnet_b0 checkpoint parity + top-1 via the landed
        ImageNet eval harness (the landed parity is the hand-built pico oracle only).

- [ ] StyleGAN2 generator importer follow-ups (BuildStyleGAN2Generator + NEW leaf layer
      TNNetModulatedConv2D LANDED, commit 8d72b95; mapping MLP z->w + synthesis tower of
      modulated conv + ReZero noise + LeakyReLU + summed toRGB skips, inference-only,
      TestStyleGAN2GeneratorParity < 1e-4 vs a numpy float64 oracle + examples/StyleGAN2Generate):
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
      injection into the frozen UNet (same wiring as the landed ControlNet zero-conv).
      Builds on the landed SD UNet importer (BuildSDUNetFromSafeTensors); the natural
      successor to the landed ControlNet. Pico parity vs a diffusers
      float64 oracle on one motion-module output for a fixed multi-frame latent; an
      examples/TextToVideo `--animatediff` mode (the example already writes PPM
      frames and `--gif`). Note:
      a cheaper no-UNet route to video is bolting the same temporal block onto the
      landed PixArt DiT.
- [ ] CogVideoX text-to-VIDEO DiT real-checkpoint follow-up
      (BuildCogVideoXFromSafeTensors[Ex] + TCogVideoXConfig + TCogVideoXVaeDecoder +
      examples/TextToVideo LANDED, reusing TNNetMRotaryEmbedding 3D RoPE +
      TNNetConvolution3D causal VAE, no new leaf; pico parity TestCogVideoXParity
      < 1e-4): real-checkpoint parity (THUDM/CogVideoX-2b) vs a sliced real checkpoint
      (the make_pico_*_fixture slicer pattern) once weights are available, incl. the
      real diffusers 3D RoPE frequency layout / temporal VAE up/down blocks and a real
      T5 encoder over a tokenized prompt feeding BuildT5FromSafeTensors instead of the
      synthetic text states. Further gaps (audit 2026-10-02), which together mean
      the repo has NO usable video generator today:
  - [ ] VAE follow-up: re-measure the whole-tile peak RSS per latent cell with
        the growable host slots (the 6.7 MB figure predates them; scaling by the net
        bytes gives ~3.8 MB, i.e. ~5.5 GB for a 30x45 tile) - part of the real
        THUDM run below.
  - [ ] Host output sharing follow-up: in-place layers (an elementwise layer
        writing into its only reader's dead source) and finding which layers
        hold the remaining unshared host bytes of RT-DETR / EfficientLoFTR
        (best-fit slots saved only 2-6% there).
  - [ ] VAE follow-up: real THUDM/CogVideoX-2b vae run (TCogVideoXVaeDecoder.Create
        on the downloaded vae folder, a 2-3 latent-frame low-res slice against
        diffusers decode); check the encoder tensors are ignored and no decoder
        tensor is left over. RAM estimate: 0.47 GB FP32 weights + ~3.8 MB per
        latent cell (scaled from the net bytes, RSS not re-measured), so
        ~5.5 GB for one 30x45 tile with EnableTiling and ~21 GB
        for the untiled 60x90 latent; a low-res slice (e.g. 8x8 latent) fits
        in ~1 GB.
  - [ ] VAE follow-up: OpenCL / speed. TNNetConvolution3D's CPU forward is one
        C-wide dot product per output tap; time the decoder under OpenCL
        (cai_conv_implicit) and check the TNNetGatherChannels frame pads and the
        per-norm zq branches (two 1x1x1 convs + gather + DeMaxPool each) stay
        resident; fold the per-norm zq branches if they dominate.
  - [ ] A `TCogVideoXPipeline` over a diffusers model_index.json folder (the
        TQwenImage21Pipeline shape: tokenizer + T5 -> transformer -> VAE, one
        component in memory at a time, OnPhase/OnStep) and a CLI that loads it;
        examples/TextToVideo takes only a single .safetensors today.
  - [ ] CFG in the TextToVideo loop with a real empty-prompt uncond branch.
        Blocked on the TCogVideoXPipeline item above (the landed T5 tower +
        tokenizer are not wired into the example yet): the example's text
        states are synthetic, so an "empty-prompt" branch would be invented
        too and `--cfg` would mean nothing. With the pipeline in place:
        eps = uncond + scale * (cond - uncond), two
        `TCogVideoXSampler.PredictNoise` calls per step, each branch with its
        own `TDiTFirstBlockCache` (PredictNoise takes the cache as a
        parameter).
  - [ ] Output: PPM/PNG frames and an animated GIF (`--gif`, neuralgif.pas)
        exist; a video-container (MP4/WebM) writer is still missing.
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
      the landed ImageNet top-1 harness is for CLASSIFIERS, but for the importers whose
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
        TNNetWindowAttention), and the GROUPED CHANNEL attention (attention ACROSS
        channels within groups, scale = num_tokens^-0.5). The closest layer is
        `TNNetTransposedChannelAttention` (Restormer MDTA), which L2-normalizes q/k and
        uses a learned per-head temperature instead; extend it with an option. Then drop the precomputed-feature
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
        `TQwen3VLVisionTower` / `TQwen3VLVisionMergerLayers` (2x2 patch merger,
        two-frame patch embed); build the variable-grid window-attention mask. Then drop the precomputed-embeds shortcut.
  - [ ] Qwen2.5-VL specifics (the temporal M-RoPE section for VIDEO, the per-frame
        time_interval, the windowed full-attention layer pattern, MRoPE rope_deltas for
        incremental decode) — the M-RoPE index builder already has the temporal slot.
  - [ ] Multi-modality prompts (images + video in one prompt; the N-still-image
        M-RoPE position overload already exists) and a multi-image Qwen2VLRunLogits
        (it takes ONE VisualTokens block + one ImageGridH/W today).
  - [ ] Real-checkpoint parity (Qwen/Qwen2-VL-2B-Instruct) + an examples/Qwen2VLDescribe
        that captions a tiny CPU image (ulimit-bounded).
- [ ] Image input for ChatTerminal / ChatServer on the multimodal Qwen3.5 / Qwen3.6 /
      Qwen3.8 checkpoints (`Qwen3_5ForConditionalGeneration`, model_type `qwen3_5` /
      `qwen3_5_moe`). Today `BuildQwen35FromSafeTensorsEx` SKIPS `model.visual.*`, and
      ChatServer rejects any `image_url` content part, so the user's tested chat models
      are text-only although their checkpoints ship a vision tower. transformers'
      `Qwen3_5VisionModel` is `Qwen3VLVisionModel` without DeepStack (no
      `deepstack_visual_indexes`), so the landed `TQwen3VLVisionTower` (Qwen-Image-2.1
      text encoder) should load it with an empty DeepStack array. Pieces:
  - [ ] Vision tower: build `TQwen3VLVisionTower.CreateFromReader` from the checkpoint's
        `vision_config` + `preprocessor_config.json`; a test that a qwen3_5 pico
        fixture (no DeepStack) encodes an image and matches HF `Qwen3_5VisionModel`
        `pooler_output` (oracle via the ~/x venv; ask before adding a new Python tool).
  - [ ] Text side: interleaved M-RoPE (`mrope_section`, `mrope_interleaved`) on the
        gated full-attention layers. `CreateRoPELayerForConfig` raises "M-RoPE has no
        partial rotary" today, and qwen3_5 combines both (partial_rotary_factor 0.25),
        so `TNNetInterleavedMRotaryEmbedding` needs the rotary-tile path. Text-only
        prompts must stay exact (all three axes equal), so the text path keeps working.
  - [ ] TChatEngine: splice the merged image tokens at the `<|image_pad|>` slots of
        the chat template (reuse `LlavaAssembleEmbeddings` / the Qwen2-VL position
        builder), and continue decode positions from max(position)+1 (HF rope_deltas),
        not from the token count. The prompt-prefix cache and `--cache-checkpoints`
        must key on the image bytes too, so a different image never resumes the cache.
  - [ ] Front ends: ChatTerminal `/image FILE` (attach to the next user message);
        ChatServer accepts OpenAI `image_url` parts as `data:` base64 URLs and local
        files (no network fetch). The tower runs on OpenCL under `--gpu` through
        the existing armed-net path.
  - [ ] Parity: HF `Qwen3_5ForConditionalGeneration` greedy tokens for one image +
        question on Qwen/Qwen3.5-0.8B (CPU, under the 2 GB cap), HF logit oracle first;
        README row stays "planned (coded)" until the user runs a real checkpoint.
- [ ] CLIPSeg text-prompted zero-shot segmentation importer follow-ups
      (BuildCLIPSegFromSafeTensors[WithConfig] + TCLIPSegConfig + RunCLIPSeg LANDED —
      frozen CLIP dual encoder + a FiLM-conditioned transposed-conv decoder emitting a
      single-channel mask for a free-text prompt; pico parity TestCLIPSegParity < 1e-4
      vs the REAL HF CLIPSegForImageSegmentation float64 oracle; examples/CLIPSegPrompt
      writes a binary-mask PPM; v1 = single text prompt -> one mask, inference-only):
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

- [ ] VGGT multi-view 3D reconstruction importer + examples/MultiViewReconstruct
      (BuildVGGTFromSafeTensors[Ex], facebook/VGGT-1B, model.safetensors via
      PyTorchModelHubMixin; CC BY-NC 4.0, VGGT-1B-Commercial is gated). Input: N
      photos of one scene (518px, patch 14). Output per frame: camera extrinsics +
      intrinsics, a depth map with confidence, and a world-space point map, in one
      forward pass (no SfM / bundle adjustment). Nothing in the tree estimates
      camera poses or writes a 3D point cloud today. Reuse: the DINOv2 patch
      embed (BuildDINOv2), TNNetQKNormAttention, TNNetVisionRoPE2D (VGGT's 2-D
      RoPE uses base 100 and shifts patch positions by +1 so the camera token and
      the 4 register tokens sit at position 0, i.e. unrotated — check that the
      NumPrefix path matches), and the DPT head from BuildDPTFromSafeTensors for
      the depth and point heads (expp1 confidence activation). New code: (1) the
      aggregator, 24 ALTERNATING blocks of frame attention (tokens of one frame)
      and global attention (tokens of all N frames), with frame 0 using its own
      camera/register token embeddings; (2) the camera head (4-block trunk with
      adaLN modulation, 4 refinement iterations, pose encoding = translation 3 +
      quaternion 4 + FoV 2) and a PoseEncodingToExtriIntri helper; (3) a binary
      PLY writer for coloured point clouds filtered by confidence percentile. The
      track head is out of scope. The example takes a folder of images and writes
      scene.ply + per-frame depth PNGs + cameras.txt. Parity: pico fixture against
      the reference `vggt` package — needs a NEW Python oracle script in tools/,
      so ask the user before writing it; the PLY writer and pose-encoding
      conversion can be tested by hand-derived cases without it.
- [ ] 3D Gaussian Splatting viewer: load a trained 3DGS scene and render novel
      views (Kerbl et al., SIGGRAPH 2023, arXiv:2308.04079). Input: the INRIA
      `point_cloud.ply` (binary_little_endian vertex list with x,y,z, nx,ny,nz,
      f_dc_0..2, f_rest_0..44 stored channel-major (15 coefficients per colour),
      opacity, scale_0..2, rot_0..3 as w,x,y,z). This file layout is shared by
      most 3DGS trainers (INRIA, gsplat, nerfstudio splatfacto exports). Today
      the tree has no 3D scene renderer (examples/TinyNeRF ray-marches a toy
      field it trains itself). Work: (1) a small PLY reader/writer unit
      (neural/neuralply.pas, header parse + typed binary properties), which the
      VGGT task above should reuse for its point-cloud writer instead of a second
      writer; (2) `TGaussianSplatScene` (load, activate: sigmoid opacity, exp
      scale, normalized quaternion; degree 0-3 SH colour + 0.5, clamp at 0);
      (3) a CPU tile rasterizer: per Gaussian, the 3D covariance R S S^T R^T,
      the 2D covariance J W Sigma W^T J^T + 0.3 I, a 3-sigma radius, then 16x16
      tile binning, depth sort per tile (key = tile id + depth), and
      front-to-back blending with alpha = min(0.99, opacity * exp(-0.5 d^T
      Sigma2D^-1 d)), skipping alpha < 1/255 and stopping at T < 1e-4. Output is
      an RGB TNNetVolume; (4) an OpenCL path: one preprocess kernel per Gaussian
      plus one blend kernel per tile (one work-group per tile, Gaussians staged
      through local memory in batches as in the reference CUDA code). The sort
      can stay on the host in v1. examples/GaussianSplatView (.lpr + .lpi +
      README.md, cthreads first): `--ply FILE --orbit N --size WxH [--gpu]`
      writes PNG frames and an animated GIF through TNNetAnimatedGIFWriter;
      camera intrinsics come from `--fov` or the INRIA `cameras.json`. Tests,
      all hand-derived (no Python oracle): one isotropic Gaussian at a known
      depth (peak pixel value and footprint radius), the degree-0 SH colour, two
      overlapping Gaussians (front-to-back order), a quaternion rotation of an
      anisotropic Gaussian, a PLY round-trip, and CPU vs OpenCL agreement.
      Memory: a real scene has 1-6 M Gaussians x 62 floats (~1.5 GB at 6 M), so
      dev-box runs must use a small scene under the 2 GB cap; real scenes are a
      user GPU-box check. Out of scope for v1: training (the differentiable
      backward), which would be a follow-up that fits a scene from posed photos.
      - Steps (1)-(4) and examples/GaussianSplatView (`--demo` on CPU and
        PoCL) DONE; the OpenCL path is `TGaussianSplatRasterizerCL` with
        BinTiles still on the host. neural/neuralply.pas (`TNNetPLYReader` /
        `TNNetPLYWriter`, ascii + both binary byte orders, streamed rows, list
        properties skipped on read), neural/neuralgaussiansplat.pas
        (`TGaussianSplatScene` Load/SaveToPLY, `TGaussianSplatCamera` +
        `InitGaussianSplatCamera` / `SetGaussianSplatCameraLookAt`,
        `TGaussianSplatRasterizer` Preprocess / BinTiles / BlendTile / Render
        with the per-Gaussian and per-tile results in flat public arrays for
        the OpenCL upload); tests in tests/TestNeuralGaussianSplat.pas. Open:
        - [ ] Real INRIA scene check (small scene on the dev box, big ones on
              the user's GPU box), including a `--cameras cameras.json` run.
        - [ ] Device-side sort + binning (radix sort on (tile, depth) keys,
              then per-tile ranges) so a frame needs no host round-trip; time
              it against the host BinTiles on a real 1-6 M Gaussian scene first.

- [X] RT-DETR real-time detection-transformer importer
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
      DONE: BuildRtDetrFromSafeTensors[Ex] + ReadRtDetrConfigFromJSONFile +
      DecodeRtDetrDetections (post_process_object_detection, no NMS); new
      layers TNNetMultiScaleDeformableAttention (forward + backward of value,
      offsets/logits and reference), TNNetTopKGatherTokens, TNNetInverseSigmoid;
      ObjectDetection --rtdetr [--image]. Parity < 1e-4 on two pico fixtures
      (tools/make_pico_rtdetr_fixture.py); agent run of PekingU/rtdetr_r18vd
      at 640 x 640: logits 3.4e-5, boxes 2.7e-4 over all 300 queries, same 5
      detections as transformers.
        - [ ] User run of a real checkpoint (r18vd and r50vd/r101vd; only r18vd
              was run, by the agent).
        - [ ] RAM: r18vd at 640 x 640 peaks at 1.86 GB RSS with
              ShareHostOutputsByLiveness on (80 MB of weights); profile which
              layers hold it (stem convs at 320 x 320, the valid-mask
              TNNetCellMul table of 8400 x 256) and trim.
        - [ ] TNNetInverseSigmoid OpenCL forward: the box-refinement chain
              (inverse_sigmoid -> TNNetSum -> TNNetSigmoid) runs on the host, so
              from decoder layer 1 on the deformable attention uploads its
              reference source (TestRtDetrOpenCL prints it).
        - [ ] OpenCL top-K selection (per-token max + top-K in a kernel) so the
              (tokens x classes) score map no longer comes to RAM each forward.
        - [ ] GPU-box timing of the RT-DETR OpenCL forward (r18vd/r50vd at
              640 x 640) against the CPU; PoCL on the 1-core VM is not
              representative.
        - [ ] Training: contrastive denoising queries, the varifocal class loss
              and the auxiliary per-layer losses (inference-only now; the
              decoder computes only the last layer's class head).
        - [ ] Unsupported config options raise: normalize_before = true,
              learn_initial_query = true, use_timm_backbone; RT-DETRv2
              (model_type rt_detr_v2: per-level point counts, discrete
              sampling) is a separate importer.

- [ ] D-FINE detector importer (BuildDFineFromSafeTensors[Ex], transformers
      model_type "d_fine", DFineForObjectDetection; Apache-2.0 checkpoints
      ustc-community/dfine-{nano,small,medium,large,xlarge}-coco and the obj365
      variants). D-FINE (Peng et al., ICLR 2025) is the RT-DETR design with a
      stronger box head: it is more accurate than RT-DETR at the same speed, and
      the nano/small models run in real time on a CPU. Reuse from the RT-DETR
      importer: the AIFI encoder layer, sine position table, top-K query
      selection (TNNetTopKGatherTokens), TNNetInverseSigmoid,
      TNNetMultiScaleDeformableAttention and the no-NMS post-process
      (DecodeRtDetrDetections). Share the code; do not copy it. New code:
        - [x] HGNetV2 backbone: DONE, BuildHGNetV2Backbone +
          ReadHGNetV2Config[FromJSONFile] (LearnableAffineBlock = a
          TNNetPolynomialActivation with a zero square term). Stem and stage
          parity < 1e-6 on both pico fixtures, CPU and OpenCL. Pico fixtures
          tiny_dfine / tiny_dfine_perlevel from tools/make_pico_dfine_fixture.py
          also hold the encoder outputs, enc_topk, logits, boxes and
          post-process. transformers 5.17 crashes when eval_idx is not the last
          layer, so the perlevel oracle runs with the decoder cut after eval_idx.
        - Neck blocks DFineRepNCSPELAN4 (RepVgg + CSP) and DFineSCDown
          (pointwise + strided depthwise) in place of the RT-DETR CCFM convs.
        - [x] Deformable attention with one point count per level
          (decoder_n_points as a list, num_points_scale = 1/n per point) and
          decoder_offset_scale: DONE, the per-level Create overload of
          TNNetMultiScaleDeformableAttention (FFloatSt [0] scale, [1+l]
          counts; old saves load unchanged) and cai_ms_deformable_attn.
          decoder_method "discrete" is not implemented (default is
          "default"). TNNetInverseSigmoid also runs in OpenCL now (opcode 73).
        - Decoder gate (DFineGate: sigmoid(Linear(concat)) mixes the residual
          and the cross-attention output, then LayerNorm).
        - Fine-grained Distribution Refinement: each layer predicts 4 x
          (max_num_bins + 1) logits; the box is the softmax over the bins
          times the fixed weighting_function(max_num_bins, up, reg_scale),
          then distance2bbox. The weighting vector is a constant computed at
          build time. Each layer adds its corners to the previous ones.
        - LQE head: top-k (top_prob_values) of each softmaxed side + their
          mean -> small MLP -> added to the class logits.
      Honour eval_idx (the output layer can be before the last decoder layer;
      do not build the layers after it). Add ObjectDetection --dfine
      [--image], a pico fixture generator next to
      tools/make_pico_rtdetr_fixture.py, and parity tests against transformers
      (logits and boxes < 1e-4) on two pico configs: one with equal and one
      with per-level point counts. Agent check on dfine-nano-coco at 640 x 640
      under the 2 GB cap: same detections as transformers. Update
      examples/ObjectDetection/README.md. Inference only; training (FDR/GO-LSD
      distillation losses) is out of scope.

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
      `TestRegNetConfigFromJSONFile`, `TestRegNet{X,Y}ImageClassificationParity` < 1e-4
      vs a REAL float64 HF RegNetForImageClassification):
  - [ ] real-checkpoint top-1 sweep (facebook/regnet-y-040 / regnet-x-040)
        through `ReadImagePreprocessConfig` + `PreprocessImageForVisionModel`.
  - [ ] timm naming variant.
  - [ ] downsample-in-first-stage=true real configs (code handles it, untested
        on a real ckpt).
- [ ] Qwen-Image-2.1 text-to-image and image editing (Qwen/Qwen-Image-2.1, diffusers
      `QwenImage21Pipeline`, Qwen Research License). 1024x1024 on the OpenCL GPU box,
      branch a19. One Opus agent per task, run serially, test + commit after each.
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
  Phase B — speed:
  - [ ] B1 follow-ups (B1 = OpenCL transformer step pass, int4 and int8 on the
        shared tiled OpenCL kernels, 88aad8ca..52e1d2c7, `examples/QwenImage --gpu`).
        Latest L4 measurements are in examples/QwenImage/README.md (1024x1024
        --int4, 10 steps: 13.05 s/step, VAE decode 35.2 s, load transformer
        55.3 s, peak RSS 12.7 GB).
    - [ ] B1d1 follow-up: Non-blocking row write + a queue Finish at the end of
          TNNet.Compute when the last layer ends on the host (these writes
          are the only mid-block waits).
    - [ ] B1d1 follow-up: the DiT/PixArt/MMDiT/F5/VAR/CogVideoX blocks and the
          EfficientNet/RegNet SE (and CBAM) importers take this OpenCL path with no
          OpenCL parity test (the RVM MobileNetV3 SE trunk and the Qwen-Image
          blocks are covered by TestRVMOpenCLParity / TestQwenImage21*OpenCL*).
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
        34.3 s int8 and 16.4 s int4 (~2.1x); at 1024x1024 (16x the tokens) today's
        rate is ~9 min per step. Step 1 (4de35106) found that the CPU kernels
        (`DotProductsTiledInt8`, `DotProductsTiledInt8Int8`, `DotProductsTiledInt4Int8`)
        already tile 16 tokens x 16 rows (`GetMaxDivisor`); they are bound by the
        inner one-row x one-token dot product (one weight load and one input load
        per MAC; int8 x FP32 re-converts each weight code 16 times), not by DRAM.
        int4 is faster than default int8 because it is int4 x int8 (`vpmaddubsw`),
        not int8 x FP32. Low-memory mode and `--prefill-window` change nothing on
        the CPU. Probe on this box (1 core VM, single TNNetPointwiseConvLinear, min
        of 3; indicative only), ms at 64 / 256 / 257 tokens: 4096->4096 int8 x FP32
        43 / 263 / 421, int8 x int8 21 / 163 / 305, int4 22 / 94 / 170; 12288->4096:
        194 / 1016 / 1464, 111 / 478 / 1260, 101 / 462 / 941. 257 tokens (prime,
        tile 1, no reuse) costs 1.4-2.6x more per token than 256.
    - [ ] B0 step 2, awaiting user approval. Proposed fix:
          - Option A, no code: run Qwen-Image with `--int8-input`. The probe shows
            1.6-2.1x on the int8 projections (263 -> 163 ms, 1016 -> 478 ms). Risk: one
            per-tensor activation scale per projection; image quality is unchecked.
          - Option B: one A row x 4 B rows (tokens) register-blocked micro-kernel per
            format, as pointer-form `TNNetVolume` class methods with an AVX2 body and
            a Pascal fallback (OPTIMIZATION-GUIDE "Authoring a new TNNetVolume
            method"). The three ranged `DotProductsTiled*` loops call them on groups of
            4 tokens inside each tile and keep the single dot for the remainder. The
            callers in neuralnetwork.pas do not change. Decode (1 token) keeps today's
            kernel. Each weight vector is loaded and converted/unpacked once per 4
            tokens: ~1.25 loads per MAC instead of 2. Try 2 rows x 4 tokens on the
            int8 x FP32 kernel (11 ymm registers).
          - Side fix: tile size fixed at 16 tokens (the ranged kernels already accept a
            partial trailing tile) instead of `GetMaxDivisor`, so awkward token counts
            keep the reuse (e.g. 63x63 = 3969 tokens gets 9 today).
          - Expected gain (estimate, unmeasured): 1.5-2.5x on int8 x FP32, 1.3-2x on
            int8 x int8 and int4. Gate: A/B the probe above at 256 tokens on this box
            before wiring; then the user times one 256x256 CPU step.
          - Tests: length sweeps of each new method against 4 single-dot calls (vector
            sizes 1..100 and 4096, int4 at multiples of 32); tiled parity for token
            counts 1..9, ranged BStart/BFinish and AStart/AFinish, partial tiles;
            int8 x int8 exact, float paths at 1e-5 relative; existing int8/int4 conv
            and `TestQwenImage21*` parity tests; the 3 suites plus one non-AVX build.
          - Risks: rule #17 (registers and fixed locals only, no buffer); asm must be
            PIC-safe (constants built in registers, as `AVXDotProductInt8Int8` does);
            float summation order changes (no bit parity required); int4 needs ~11
            ymm registers.
  - [ ] B2. VAE decode speed: 3x3 convolutions at 1152 channels. Measured at
        1024x1024: 528.2 s on the CPU, 35.2 s on OpenCL (L4, tiles 256/192).
    - [ ] B2b follow-up: Drop the armed nets' host FConcatedWeights/FConcatedWInter
          copies (~2 GB) after the upload.
    - [ ] B2b follow-up: Measure --serial vs the default parallel scheduler with
          OpenCL on the L4. The PoCL oversubscription (idle workers spinning hot)
          is fixed: only worker 0 stays hot while OpenCL is armed. Cap the workers
          for CPU OpenCL devices only if a --serial gap remains.
    - [ ] B2f. Big tiles / untiled decode (dims 1152/1152/1152/576/288/144,
          1.06 TFLOP per 256-px tile; 1024x1024 = 29.2 TFLOP at 256/192, 26.5
          at 512/384, 17.0 untiled). The planned OpenCL bytes per sized
          net are ~1.3 / ~1.9 / ~4.7 GB at a 256 tile / 512 /
          untiled 1024. a78b169f shares one weight copy across the tile-shape
          nets, so re-measure the time outside the nets. Untiled 1024 still
          overflows on the host (`TVolume.ReSize` integer size). In the REPL
          the transformer stays resident, so the VAE has ~10-14 GB on a 24 GB L4.
          - [ ] B2f5 follow-ups (the nearest-2x + 3x3 OpenCL fold, c42915c0):
                - [ ] GPU timing of the fold on the L4.
                - [ ] B2f5 follow-up: the "single reader" test is
                      `GetDepartingBranchesCnt() = 1`; a layer that reads the
                      TNNetDeMaxPool output without registering as a branch would
                      see an empty output. Audit readers that bypass AddLayer.
                - [ ] B2f5 follow-up: Swish/HardSwish/Rectangular conv subclasses
                      and the int8/int4 convs do not fold (the Qwen-Image VAE
                      needs none; SD VAE / UNet upsamplers might).
          - [ ] B2f6. Untiled path: memory estimator vs the device limits,
                `--vae-tile 0`; L4 runs 256/192 -> 512/384 -> untiled in the REPL;
                then the default decision with the user.
          Side note: `FInputBufferBs` (READ_ONLY) is written by cai_im2col
          (int8, FP16 and training paths) and
          `FResultBuffer` (WRITE_ONLY) is read by consumer kernels; works on
          NVIDIA/PoCL, undefined per spec; pool buffers must be READ_WRITE.
  - [ ] B3 follow-up (negative-prompt guidance, 90e08744: examples/QwenImage
        `--negative-prompt`, `--guidance-scale`, `--apg`, `--apg-velocity`):
    - [ ] Real-checkpoint quality check on the GPU box: same seed, scale 1
          vs 4 with a negative prompt, CFG vs `--apg 0,-0.5,15` (x0 and
          `--apg-velocity`); record the step time (two passes) and the images
          in examples/QwenImage/README.md.
  Phase C — editing and reference images (plan from source reading of
  diffusers 0.41.0.dev0 `pipeline_qwenimage21.py` / `transformer_qwenimage21.py`
  / `autoencoder_kl_qwenimage21.py` and transformers 5.17.0 `modeling_qwen3_vl.py`).
  diffusers has
  one pipeline for text-to-image and editing (`image=`): no strength, no mask, no
  second-image input; masks and circles are extra input images. Condition images
  are resized with our own resize, not PIL Lanczos (accepted divergence).
  - [ ] C4. Prefix pass on OpenCL, int8 prefix K/V and a windowed text-encoder
        prefill. Estimates at 1024^2: prefix ~4.1k tokens / ~4.3 GB FP32 K/V for one
        reference image, ~41k tokens / ~43 GB for ten; text-encoder scores ~13.6 GB
        unwindowed at ten. Decide scope with an L4 measurement.
- [ ] Qwen-Image inpainting (`--init-image FILE --mask FILE`, `/mask`; coded +
      pico parity `TestQwenImage21InpaintParity`): real 1024x1024 inpaint on the
      GPU box (user run), then the README example output.
- [ ] MultiDiffusion panoramas follow-up (examples/QwenImage `--panorama WxH`,
      `--panorama-stride N`, `--panorama-wrap`, `/panorama WxH|off` over
      `TMultiDiffusionWindows` in neuraldiffusion.pas; coded and covered by
      `TestQwenImage21PanoramaOneWindowParity`, `TestMultiDiffusionConstantSeams`
      and `TestMultiDiffusionCircularWrap`):
      - [ ] Real 2048x512 / 4096x1024 runs on the user's GPU box (quality,
            time per step, with and without `--panorama-wrap`); record them
            in examples/QwenImage/README.md.
- [ ] First-Block Cache follow-ups. `TDiTFirstBlockCache`
      (neural/neuralpretrained.pas, the decision of diffusers
      `hooks/first_block_cache.py`: residual of block 0's image hidden states,
      mean|r - r_prev| / mean|r_prev| > threshold runs the rest, r_prev and the
      tail residual refreshed only on full steps) is wired into
      `TQwenImage21Transformer.PredictVelocity` and
      `TQwenImage21Pipeline.FirstBlockCacheThreshold` (0 = off,
      `ImageStats.SkippedStepCount`), examples/QwenImage `--fbc-threshold X` /
      `/fbc X|off`. Covered by `TestDiTFirstBlockCacheDecision`,
      `TestQwenImage21FirstBlockCacheReference` (threshold 0 bit-identical, a
      skipping threshold against a plain-loop reference) and
      `TestQwenImage21FirstBlockCacheOpenCL`.
      - [ ] Speed and quality on the user's GPU box: 1024x1024, 18 steps,
            thresholds 0.05 / 0.1 / 0.2 vs off, same seed; record the skipped
            steps, the denoise time and the images in examples/QwenImage/README.md.
      - [ ] Device-side residual math on OpenCL. Today `PredictVelocity`
            downloads img_in's and block 0's outputs every step, the last
            block's output on a full step, and uploads block 0's output plus
            the tail residual on a skipped step (three (tokens x hidden)
            transfers per step). One reduction kernel for sum|r - r_prev| and
            sum|r_prev| plus one axpy, on the resident buffers, would leave
            only two floats crossing per step; worth it if the GPU-box
            measurement shows the transfers.
      - [ ] diffusers apply_first_block_cache pico oracle (needs user approval
            for a tools/ generator). Traps: diffusers 0.41 does not register
            `QwenImage21TransformerBlock` in `TransformerBlockRegistry`
            (register it with return_hidden_states_index=0,
            return_encoder_hidden_states_index=None), and its step 0 runs the
            "extract" pass over the whole joint sequence, so block 0's residual
            there has prefix rows and does not match the cached steps' shape;
            the oracle must slice the target rows (our step pass carries only
            them).
      - [ ] Reuse in later DiT pipelines (SD3/FLUX MMDiT, Wan): the same
            split of their block loop (run block 0, ask, run or skip the
            rest); a single-TNNet denoiser can follow TCogVideoXSampler's
            layer-range split.
- [ ] LaMa large-mask inpainting follow-ups. `BuildLaMaFromSafeTensors`
      (neuralpretrained.pas, `TestLaMaParity` on the pico fixture from
      `tools/make_pico_lama_fixture.py`) and examples/LaMaInpaint are coded; the
      real `anyisalin/big-lama` checkpoint is untested.
      - [ ] Real 512x512 run on the user's box, then promote the README status
            from "coded, real checkpoint untested". Memory estimate from the pico
            net scaled to big-lama: about 8.6 GB at 512x512, of which 3.3 GB is the
            im2col column matrix of the 7x7 64->3 output conv and 3.4 GB the
            column matrices of the 36 FFC-unit 3x3 convs; releasing
            `FInputPrepared` is the user-decision item under "Layer follow-ups".
- [ ] Image-generator LoRA follow-ups (`--lora FILE[:SCALE]` in examples/QwenImage,
      `TNNetLoRAMergeReader.CreateFromFile` / `CreateLoRAMergeChain`, 6eb26a48):
      - [ ] Run a real community SD 1.5 LoRA (kohya and diffusers files) on
            runwayml/stable-diffusion-v1-5 (user run).
      - [ ] Qwen-Image has no diffusers `fuse_lora` latent oracle; its reader
            now also takes `lora_adapter_metadata` (`transformer.r` /
            `transformer.lora_alpha`), checked on SD files only.
      - [ ] LyCORIS LoHa / LoKr / Tucker (`lora_mid`) and DoRA files are
            refused; SDXL LoRAs need the SDXL pipeline first.
- [ ] TAESD real-checkpoint check: run examples/LatentTextToImage with a
      downloaded `madebyollin/taesd` (`--tiny-vae DIR --fast-vae`) and compare
      the image against the full SD VAE decode. The original madebyollin
      `taesd_decoder.safetensors` key layout loads too, but only through
      `BuildTinyVaeDecoderFromSafeTensorsEx` with a hand-built config (that
      repo ships no config.json); checked on a re-keyed pico fixture, not on
      the real file.
- [ ] TAESD previews for examples/QwenImage: examples/QwenImage
      has NO matching tiny decoder: Qwen-Image-2.1's VAE is z_dim 64 at 16x
      spatial with RGBA out, while taesd* / taef1 are 4/16-ch at 8x and taehv's
      `taew2_1` is Wan-2.1 (16-ch, 8x); a QwenImage preview needs a tiny decoder
      trained for that latent space (or a cheap latent->RGB linear projection).
- [ ] Real-ESRGAN (`TRRDBNetUpscaler`, examples/RealESRGANUpscale, QwenImage
      `--upscale`) open checks:
      (1) memory: on the CPU path a tile holds ~2.7 MB per net-input pixel
      (x4plus layout, measured), almost all of it each convolution's own
      FInputCopy + FInputPrepared (im2col), which liveness output sharing does
      not cover; releasing or sharing `FInputPrepared` (the user-decision
      item under "Layer follow-ups") would allow 256-pixel tiles;
      (2) DEFERRED, needs the user's approval for a new Python tool: a pico
      basicsr x2plus fixture (`tools/` generator from the ~/x venv using basicsr's
      RRDBNet definition), parity < 1e-4 on the whole net. The x4 and scale-2
      nearest layouts have whole-net numpy oracles (`TestRRDBNetParity`,
      `TestRRDBNetParityScale2`); the x2plus unshuffle layout has only the
      Pascal-side stem check `TestRRDBNetPixelUnshuffleX2`;
      (3) a real `RealESRGAN_x4plus.pth` (and x2plus) run on the user's box,
      through both examples (QwenImage: `--width 512 --height 512 --upscale`);
      (4) not supported: the old ESRGAN `model.N.*` key layout.
- [ ] Jev-style typed decision models ("System One": a state plus typed questions in,
      calibrated option probabilities out, no text generated). Stages 0-2 are
      coded (`/v1/systemone` on ChatServer, AnyJev L0 in neuralanyjev.pas, Kev and
      Open-Jev via `--decision-adapter`; design in c54d68df). Open: the
      real-checkpoint checks below and the follow-ups.
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
        weights move the probabilities far more than FP32 rounding. A first run on a
        0.5B chat checkpoint gave a questionable yes/no answer; stays planned (coded).
  Stage 2 — Kev and Open-Jev:
  - [ ] D2.7 User-box real checks: Kev-0.8B (Kev's own test record) and
        Open-Jev-2B against their Python servers; commands written by the agents.
        Commands below (ChatServer.md *Trained decision checkpoints* has only the
        DecisionClient sanity run); PASS = max abs
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
  Follow-ups: Laya (ModernBERT encoder + decision head); AnyJev batch prior, L1, L2; Kev
  `KEV_DATE_FACTS`; `/v1/systemone/permute` and `/separate`; per-question
  `TruncateTo` downloads the resident KV cache on OpenCL (measure first); tokenizer
  "specials as plain text" flag for chat requests; live (unmerged) LoRA.
- [ ] Flow-matching sampler clean-ups surfaced by `TNNetFlowMatchEulerScheduler`
      (fd99162f):
  - [ ] `examples/FlowMatching/FlowMatching.lpr` (`SampleOne`): optionally drive the
        loop with `TNNetFlowMatchEulerScheduler`. The example integrates t from 0
        (noise) to 1 (data) with dt = 1/N and no shift, so it needs time = 1 - sigma
        and a reversed velocity sign; the gain is small (the loop is already one
        `MulAdd` per step).
- [ ] Flow-matching sampler follow-ups (`fsDPMSolverPP2M` / `fsUniPC` +
      examples/QwenImage `--sampler euler|dpm2m|unipc`):
      - [ ] User GPU-box quality check of dpm2m and unipc at 10-12 steps
            against euler at the current default count (18).
- [ ] Adaptive Projected Guidance follow-ups (`TNNetAdaptiveProjectedGuidance` in
      neural/neuraldiffusion.pas, 15c5aca5; `--apg` in examples/LatentTextToImage,
      examples/ConditionalDiffusion and examples/QwenImage). Later consumers: the
      open CFG items of the CogVideoX TextToVideo loop and the Stable Diffusion
      pipeline.
      - [ ] A diffusers `normalized_guidance` oracle test; needs the user's
            approval for a new Python tool.
- [ ] Perturbed-Attention Guidance (PAG, Ahn et al. 2024, arXiv:2403.17377) and
      Skip Layer Guidance (SLG) follow-ups. The Qwen-Image pipeline runs both
      (`TQwenImage21Pipeline.PerturbedGuidance`, `--pag-scale` / `--slg` in
      examples/QwenImage); the SD pipeline runs PAG
      (`TStableDiffusionPipeline.PerturbedGuidance`, `--pag-scale` in
      examples/StableDiffusion, diffusers parity `TestStableDiffusionPAGParity`).
      Open:
      - [ ] A with/without image pair and the measured denoise cost at the
            defaults (2 of 18 steps run the extra pass) from the user's GPU
            box, for examples/QwenImage/README.md; the default blocks (PAG 16,
            SLG 6,7,8) are untuned on the real checkpoint.
      - [ ] A with/without PAG image pair on a real SD 1.5 checkpoint
            (examples/StableDiffusion/README.md).
      - [ ] Later consumers: examples/LatentTextToImage (PixArt) and CogVideoX.
      A Qwen-Image diffusers oracle needs a patched block / processor registry
      and a Prefix + i identity rule in Python (diffusers 0.41.0.dev0 cannot run
      either guider on QwenImage21); ask the user before adding a Python tool.

- [ ] SuperPoint + LightGlue follow-ups (`BuildSuperPointFromSafeTensors[Ex]`,
      `BuildLightGlueFromSafeTensors[Ex]`, `TSuperPointKeypoints`,
      `TLightGlueMatcher` with early exit and point pruning, and
      examples/KeypointMatching are done):
      - [ ] LightGlue follow-ups: a real-checkpoint run of
            ETH-CVG/lightglue_superpoint on a real image pair, with a
            transformers comparison (with and without early exit/pruning) and
            the measured time; threading the attention (one head per pool
            thread) and an OpenCL path; a TNNet coordinate-rotary layer if a
            TNNet path is ever wanted.
      - [ ] SuperPoint follow-ups: a real-checkpoint run of
            magic-leap-community/superpoint (check its preprocessor_config
            do_grayscale and the tensor names); the torchvision-backend
            processor (uint8 grayscale before an antialiased resize) is not
            matched, only the PIL backend.

- [ ] EfficientLoFTR follow-ups (`BuildEfficientLoFTRFromSafeTensors[Ex]` +
      `TEfficientLoFTRMatcher`, `--matcher loftr` in examples/KeypointMatching
      and `TestEfficientLoFTR*` are done, 238641b4; real checkpoint checked
      against transformers at 256 x 192):
      - [ ] Memory: about 22 KB of RAM per input pixel (1.35 GB at 256 x 192,
            about 6.9 GB at 640 x 480). Every CPU 3x3 convolution keeps its
            im2col (`FInputPrepared`), its padded input copy (`FInputCopy`),
            `FOutputRaw` and `FBiasOutput` for the life of the net, and
            `ShareHostOutputsByLiveness` shares only `FOutput`. Candidates:
            one conv scratch shared by every layer of an inference-only net
            (it must stay safe with the parallel layer scheduler), and
            `CanShareHostOutput` overrides on `TNNetBilinearUpsample`,
            `TNNetLeakyReLU` and `TNNetTokenLayerNorm`. A 640 x 480 run fits
            the dev box only after this. The im2col part is the same
            library-wide question as "Decision for the user: inference-time
            release of the convolution im2col matrix" below; it now blocks
            big-lama 512x512, EfficientLoFTR 640x480 and RT-DETR 640x640.
      - [ ] Real-checkpoint run at 640 x 480 on a real image pair, user-tested,
            with the transformers comparison (the 256 x 192 check used
            docs/park.jpg and a crop of it).
      - [ ] The backbone, attention and pyramid weights are loaded twice, once
            per image branch (+64 MB at the real size). Borrow them with
            `LinkWeightsFrom` / `BuildWeightOwner` instead.
      - [ ] Speed: the coarse similarity (`DotProductsPacked`, 4800 x 4800 x
            256 at 640 x 480) and the per-match refinement run on one thread;
            no OpenCL path has been tested.
      - [ ] Image pairs of different sizes (the original code supports them,
            transformers does not).

- [ ] CoTracker3 point tracking importer + examples/PointTracking (Karaev et al.
      2024, *CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling
      Real Videos*, arXiv:2410.11831; checkpoint facebook/cotracker3,
      `scaled_offline.pth`). Video in plus query points `(t, x, y)`, out a 2-D
      track and a visibility/confidence score per point per frame. The library
      has dense two-frame flow (RAFT, RIFE) and sparse two-image matching
      (SuperPoint + LightGlue), but nothing follows a point across a whole clip.
      Reuse: `RaftBuildEncoder` / `RaftLoadEncoder` for the per-frame CNN
      encoder (CoTracker3 uses the same BasicEncoder at stride 4); the bilinear
      sampler of `TNNetGridSample` for the multi-scale local correlation around
      each track (4 pyramid levels, a 7x7 window, track-feature x frame-feature
      dot products, the same idea as `TNNetCorrelationLookup` but sampled at
      sparse points); the existing attention blocks for the update transformer,
      which alternates time attention per track with cross attention between
      the real tracks and 64 learned virtual tracks. The update runs K
      iterations (default 6) and refines track deltas plus visibility and
      confidence logits.
      - [ ] `BuildCoTrackerFromSafeTensors[Ex]` + `CoTrackerPredictTracks` in
            neural/neuralpretrained.pas (offline mode: the whole clip in one
            window). The checkpoint is a torch `.pth` with no safetensors copy
            (the torch.save zip reader neural/neuraltorchbin.pas can read it);
            a pico fixture plus oracle are needed: ask the user before adding
            the Python tool. Parity test against the reference
            `CoTrackerThreeOffline` forward on the pico fixture (tracks and
            visibility).
      - [ ] examples/PointTracking (.lpr + .lpi + README.md): read a frame
            sequence (a folder of images), track a regular grid of points or
            points given on the command line, and write the frames with the
            tracks drawn as colored trails plus an animated GIF with the
            existing GIF writer.
      - [ ] Later: online (sliding-window) mode for long videos, reusing the
            offline weights with the `window_len` / `step` overlap of the
            reference `CoTrackerOnlinePredictor`; a real-checkpoint run on a
            real clip from the user's GPU box.
- [ ] Real-checkpoint run of the photo -> OpenPose -> ControlNet path (code
      DONE: `EstimateViTPoseInBox`, `EstimateMultiPersonPose`,
      `DrawOpenPoseBodies`, examples/ControlNetCanny `--pose-photo --yolo
      --vitpose [--pose-out]`, examples/PoseEstimation `--openpose-out`; only
      the pico fixtures have run). On the user's GPU box: yolov8n +
      usyd-community/vitpose-base-simple + lllyasviel/sd-controlnet-openpose
      on SD 1.5 with a real multi-person photo; check the `--pose-out` image
      by eye against controlnet_aux DWpose / OpenposeDetector output, then
      mark both READMEs supported.
      `DrawOpenPoseBodies` and `tools/make_openpose_draw_oracle.py` were
      checked line by line against the controlnet_aux 0.0.10 sdist (equal to
      GitHub master) on 2026-10-23: dwpose/wholebody.py (neck and
      reorder), dwpose/__init__.py (/W, /H, score > 0.3) and
      dwpose/util.py (draw_bodypose) all match; nothing changed.
      OpenposeDetector (open_pose/__init__.py, open_pose/util.py)
      draws differently: limbs then dots PER PERSON, with color * 0.6 at fill
      time, so a later person's limbs cover an earlier person's dots. Compare
      the `--pose-out` image against DWpose output, not OpenposeDetector output.
      - [ ] Optional: no cv2 oracle covers `DrawOpenPoseBodiesViaDetectResolution`
            (cv2 `INTER_LINEAR` rounds uint8 with fixed-point weights; the
            route resizes in float), only size and thickness tests.

- [ ] Multi-object tracking follow-ups (ByteTrack, 013f18a8..31fccc90):
      - [ ] Real yolov8n run of examples/ObjectTracking on a real clip (user
            box, also `--gpu` and `--int8`), then mark its README supported.
      - [ ] The pico YOLO fixture config has no `names`, so the example smoke
            runs only show ids; the named path is covered by the unit test only.
      - [ ] Optional: BoT-SORT / OC-SORT style camera-motion compensation is
            not ported; ByteTrack only.

## OpenCL forward coverage — layers still on the host

Forward pass only; device backward is limited to `TNNetFullConnect`,
`TNNetPixelShuffle` and `TNNetBicubicUpsample` (convolution: see "Convolution
training on OpenCL"). About 69 of the 528 `TNNet*` classes in `neuralnetwork.pas` (2026-10-22)
declare a `ComputeOpenCL`, and `cai_activation` covers 73 elementwise opcodes on top of
that. `csActivationOpenCLMinSize` is 64 MB, so a small layer reaches the device
only when its source is ALREADY resident — which is why the ranking below is by
"does this layer evict a resident chain to RAM", not by FLOPs.

Every item: guard behind the existing `WillOpenCL()`/`FShouldOpenCL` verdict,
bind the resident source instead of downloading it (`ShouldBindPrevOutputOnOpenCL`)
and leave the result in OpenCL memory, keep the host path as the fallback, and
pin an exact-vs-CPU parity test that skips cleanly with no device. PoCL on the
dev box proves parity; throughput numbers have to come from the GPU box.

### Residency breakers on the decode path (small kernels, largest payoff)

- [ ] GPU-box check after 0ea67b17 (gather-copy layers now share OpenCL outputs by
      liveness; pipelines that enable output sharing, such as Restormer and other
      neuralpretrained.pas pipelines, now also share TNNetPad / Crop / Transpose /
      SpaceToDepth / upsample outputs; the host-tap contract is pinned by
      TestGatherCopyHostTapsShareOutputsByLiveness): a real Restormer / Qwen-Image
      VAE run on the GPU box against the parent commit (output and time).
- [ ] Follow-ups of the `TNNetTopKGate` / `TNNetBiasBalancedTopKGate` device
      forward (2ebe172b; `cai_topk_gate`) plus the device slot-map build
      (`cai_moe_build_slot_map`): router -> gate|up -> down has no host hop per
      token when the router output is resident. Open follow-ups:
  - [ ] Throughput of `cai_topk_gate` (one work-item per token, so a decode
        step is one serial work-item over the experts) and of the two bank
        kernels (one work-item per output element, no tiling) is unmeasured;
        time them on the GPU box.
  - [ ] `TNNetExpertChoiceGate` (top-k along the TOKEN axis, per expert) has no
        device forward; it is not on any fused-bank path today.
  - [ ] A top-k gate with 32 < k < experts stays on the host
        (`CAI_TOPK_GATE_MAX_K`, the kernel's private selection buffers).
- [ ] GPU-box check of `TNNet.ShareOpenCLHalfSourceCopies` (on by default): in
      `--experimental-fp16` the int8 FP16 consumers of one resident source now
      bind the half copy the first consumer cast in the same forward pass
      (`TNNetLayer.OpenCLHalfOutputCopy`), so a Qwen3.5/3.8 block casts 2
      sources instead of once per each of the 5 consumers. PoCL proves parity and the cast count
      (`TestFP16HalfSourceCopyShared`); measure the 27B decode tok/s in
      `--experimental-fp16` against the parent commit. Open follow-ups:
  - [ ] Per-layer queues (`--no-gpu-shared-kernel`) share nothing: sharing
        there needs an event between the cast and each reader on another queue.

### Real device-side work still running on the host

- [ ] Follow-ups of the `TNNetSelectiveSSM` device forward (0e9ed104), all
      three modes (DState = 1, the
      folded Mamba-1 scan, and the Jamba / falcon_mamba inner-norm scan), for the
      full-sequence forward, prefill windows and the one-token decode path.
      `TNNetSelectiveSSMCL` runs `cai_selective_ssm_project` (delta / B / C),
      `cai_selective_ssm_inner_norm` (Jamba only) and `cai_selective_ssm_scan`
      (one work-item per channel for the whole sequence) on one queue; the
      source is bound, the result and the (Depth, N) decode state stay in
      OpenCL memory, and the state slots of `TNNetStreamingDecoder` checkpoints
      are OpenCL buffers. Open follow-ups:
  - [ ] Throughput is unmeasured; time a Mamba / falcon_mamba / Jamba decode
        on the GPU box. `cai_selective_ssm_project` is one 64-lane work-group
        per (row, token), so a long prefill re-reads each weight row once per
        token; if a profile shows it, move the projections onto the tiled
        GEMM. The folded Mamba-1 `W_d` is a full Depth x Depth matrix; the
        unfolded low-rank dt path (x_proj_dt then dt_proj, as the Jamba mode
        already runs) would cut that traffic by Depth / dt_rank.
- [ ] Follow-ups of the `TNNetMamba2` device forward (fc5c2c40) and its
      streamed-decode state (329b0a91), the Mamba-2 and Nemotron-H mixer.
      `TNNetMamba2CL` runs `cai_mamba2_scan` (one work-item per d_inner
      channel for the whole sequence, the (d_inner, N) state advanced in
      place) and `cai_mamba2_gated_norm` (one work-group per token) on one
      queue; the source is bound, the result stays in OpenCL memory. The
      decode state now stays in OpenCL memory and checkpoint slots are OpenCL
      buffers. `TNNetRecurrentScanCL` is the new shared parent of the
      GatedDeltaNet, SelectiveSSM and Mamba-2 helpers (state bank plumbing).
      Open follow-ups:
  - [ ] Throughput is unmeasured; time a Mamba-2 and a Nemotron-H decode and
        prefill on the GPU box. One work-item per channel runs the whole
        sequence serially, so a long prefill is latency-bound; if a profile
        shows it, the chunked SSD scan is the next step (it is also the CPU
        stretch goal).
  - [ ] The gated RMSNorm normalizes over the full d_inner (CPU and OpenCL
        alike). That matches HF `modeling_mamba2.py` (`MambaRMSNormGated` over
        intermediate_size), but HF `modeling_nemotron_h.py` builds
        `Zamba2RMSNormGated(group_size = intermediate_size // n_groups)`. The
        tiny_nemotronh fixture uses n_groups = 1, so no test sees the
        difference; a real Nemotron-H checkpoint with n_groups > 1 would
        normalize wrong. Needs a decision: a norm-group count on `TNNetMamba2`
        (CPU forward, backward, `cai_mamba2_gated_norm` with one work-group per
        (token, group)), set by the Nemotron-H importer, plus an n_groups > 1
        Nemotron-H fixture. Also fix the main README's `TNNetMamba2` row,
        which claims a "grouped gated RMSNorm" the code does not do.
  - [ ] The chat engine's checkpoint and prefill-window routes have not been
        run end to end on a real Mamba-2 or Nemotron-H checkpoint since the
        layer became recurrent; only the pico fixtures cover it.
- [ ] Follow-ups of the `TNNetTokenShift` / `TNNetDiagonalSSM` / `TNNetWKV` device
      forward (`cai_token_shift_scan`, `cai_diagonal_ssm_scan`, `cai_wkv_scan` on
      the shared `TNNetChannelScanCL`; decode state and checkpoint slots stay in
      OpenCL memory; the tiny_rwkv full forward, streamed decode and checkpoint
      resume are covered on PoCL). Open follow-ups:
  - [ ] Throughput is unmeasured: each kernel is one work-item per channel,
        serial over the window, so a long prefill is latency-bound. Time an
        RWKV decode and prefill on the GPU box.
  - [ ] No real RWKV checkpoint has been decoded on a GPU; only the pico fixture.
- [ ] Follow-ups of the attention subclasses' device paths:
  - [ ] Follow-ups of the GPT-OSS attention sinks. `TNNetGptOssSinkAttention`
        (bc08ac04) runs its full forward on `cai_sdpa_flash`; the flash
        epilogue and `cai_sdpa_decode_merge` take a per-query-head sink buffer,
        and a split launch adds the sink in the merge only. Since 36da5698
        `TNNetFusedSDPA` has the same sinks (`pHasSinks`, one neuron of QHeads
        logits, option bit 2 of FStruct[7]) on the CPU paths, the backward and
        the resident cached step (`TNNetFusedSDPACL.ComputeStep`, split-row
        decode and flash, FP32 and int8 KV). The gpt-oss importer now builds
        one fused layer per decoder layer through
        `AddGQAAttentionFromSources(..., HasSinks=true)` (head-tiled YaRN RoPE
        on the whole q/k projections); `TNNetGptOssSinkAttention` is only the
        `NeuralAllowFusedAttention=false` fallback. Open follow-ups:
    - [ ] GPU-box: gpt-oss decode tok/s with the resident KV cache against
          the parent of the importer commit (per-head sink heads, host cached
          decode), FP32 and int8 KV.
    - [ ] GPU-box: gpt-oss prefill tok/s (uncached full forward on OpenCL,
          e.g. the perplexity run) against the parent of 1336ef5a.
    - [ ] No model-level int8-KV parity test for gpt-oss: on the pico a top-2
          routing flip turns int8 rounding into logit jumps, so int8 with sinks
          is covered per layer only (`TestFusedSDPASinkCachedDecodeParity`,
          `FusedSDPASink*OpenCLParity`).
  - [ ] Follow-ups of the `TNNetSinkAttention` full forward on `cai_sdpa_flash` (732bbaa5): the
        K sink slots are cache rows 0..K-1 (a prefix `ComputeFlashPacked`
        uploads only when the weights change or the cache buffer regrows), the
        real K/V sit at slot K, and CacheSlot = K gives mask mode 1 the keys
        [0, K + t + 1). Source bound, result resident, host fallback
        (`SinkAttentionFlashOpenCLParity`, max|diff| < 8e-7 on PoCL). Open:
    - [ ] GPU-box: prefill tok/s of an `avSink` stack against the parent commit,
          and decode tok/s of an `avSink` stack with the resident cache.
    - [ ] GPU-box: decode tok/s of an `avSDPA` stack with the resident cache.
  - [ ] Follow-ups of the `TNNetDifferentialAttention` full forward on `cai_sdpa_flash`
        (732bbaa5): one launch with two query heads per KV head
        (`cai_sdpa_diff_pack_q` writes [Q1,0 | 0,Q2], score scale
        1/sqrt(d_k/2)), then `cai_sdpa_diff_combine` writes Y1 - lambda*Y2
        into a second result buffer. Source bound, result resident, host
        fallback (`DifferentialAttentionFlashOpenCLParity`, max|diff| < 1.5e-6
        on PoCL). Open:
    - [ ] GPU-box: prefill tok/s of an `avDifferential` stack against the
          parent commit, and decode tok/s with the resident cache.
- [ ] `TNNetDeltaNet`, `TNNetGatedLinearAttention`, `TNNetRetention` and
      `TNNetCausalLinearAttention` (the non-causal `TNNetLinearAttention` already
      has a two-GEMM `ComputeOpenCL`). COUPLED, not independent: each is a
      strict per-token scan today (the causal linear attention is a prefix-sum
      `S_t = S_{t-1} + phi(K_t) x V_t`), so there is no GEMM-shaped kernel to
      offload until the chunked/parallel forward tasks in "Layer follow-ups that
      fix real limitations" are done (intra-chunk dense GEMM + inter-chunk
      running state). Sequence them after those, or fold the device path into
      each chunked rewrite, with an exact-vs-CPU parity test each.


### Vision and audio layers with no device forward

Companion to the two groups above, same contract (forward only, bind the
resident source, leave the result resident, host fallback, exact-vs-CPU parity
test). The ranking differs though: a vision or audio forward runs over large
tensors, so here the FLOPs matter alongside the transfers, and a kernel can pay
for itself even when it sits between two host layers.

Derived by walking the class graph: a layer counts as covered if it declares
`ComputeOpenCL`, sets an `FActivationOpcode`, dispatches to OpenCL inside its
own `Compute`, or inherits one of those without overriding `Compute`. Note the third case: a class that
overrides `Compute` with no device branch silently loses its parent's kernel
(`TNNetMaxPoolPortable` lost `TNNetMaxPool`'s that way; fixed in commit abd639f0).

Training-only layers (`TNNetDropout`, `TNNetDropPath`) are out of scope.

- [ ] GPU-box check for an a20 device path, unmeasured on the dev box:
      `TNNetRealFFT2D` / `TNNetInverseRealFFT2D` (d7c177dd, direct O(N^2)
      DFT per line) inside examples/LaMaInpaint at 512x512: FFT share of the
      generator forward, and whether the whole generator stays resident.
- [ ] GPU-box check: `cai_sdpa_flash` gained a uniform score-bias branch
      (13282df0) that the decode path also runs. Measure decode tok/s on a
      Qwen checkpoint before/after; if it costs anything, split the biased
      variant into its own entry point.
- [ ] GPU-box timing of the resident `TNNetModulatedConv2D` forward (be2196e5,
      `TDotProductSharedKernel.ComputeModulatedConv`) at 1024x1024: StyleGAN2
      generator forward with OpenCL, per-layer forward time of the 1024x1024
      `TNNetModulatedConv2D` layers against the parent commit (host w'' slab +
      upload every forward). Run `examples/StyleGAN2Generate --gpu
      --time-layers --time-passes 3 <1024 checkpoint>` (rows
      `#<idx> ModConv 1024x1024x<D>`); CPU vs `--gpu` matched byte-for-byte on
      the pico fixture under PoCL.
- [ ] Follow-ups of the vision attention device paths:
      - [ ] SwinV2 follow-up: `cai_sdpa_qk_l2norm_packed` is one work-item
        per (token, Q/K/V part) with a serial d_k loop; fine for Swin windows
        (49 tokens, d_k 32), use work-groups if a large-d_k user appears.
      - [ ] `TNNetSAMVisionAttention` follow-ups (OpenCL forward is
        `TNNetSAMVisionAttentionCL`). Time a SAM encoder on the GPU box first:
        - Not timed yet: run a SAM encoder on the GPU box and compare the
          decomposed-bias flash launch with the old full-table one.
        - The qkv and out projections run on `cai_row_dot_products` (see the
          `TNNetDotProducts` follow-up below).
      - [ ] `TNNetDotProducts` follow-up: `cai_row_dot_products` is one
        work-item per output with a serial depth loop and uncoalesced A-row
        reads. Time a CAI self-attention model on the GPU box; if the
        contraction shows up at long sequences, tile A and B rows through
        local memory.
- [ ] Check (not verified): `TNNetSpectralConv2D` (CPU and OpenCL) keeps only
      the low corner [0..ModesX-1][0..ModesY-1] of the spectrum and takes the
      real part of the inverse 2-D FFT. The reference FNO `SpectralConv2d`
      keeps TWO corners along the full-FFT axis (weights1 for the low modes,
      weights2 for the negative-frequency modes) and uses irfft2. Decide
      whether to add the second corner (needed to import a torch FNO
      checkpoint) or document the layer as a one-corner variant.

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
`cai_activation` kernel, which has 73 opcodes, each wired to its layer classes
through `ComputeActivationOnOpenCL()`. Every elementwise `TNNetReLUBase`
activation now has one (`TNNetRReLU` reuses 15 = LeakyReLU, device path only
while `Enabled = false`, i.e. at its fixed inference slope).

Both are forward-only by construction: neither produces `FOutputRaw` or the
derivative mask, so `IsActivationFunctionInOpenCL` requires `not FIsTrainable`
and the standalone path requires it too. `csActivationOpenCLMinSize` (64 MB)
keeps the standalone kernel off the device unless the source is already
resident or `ForceOpenCL` is set — deliberate, since a 1-flop-per-word kernel
cannot pay for its own transfer.

- [ ] GPU-box decode timing for the trap-free math of d8e5ec90: the -80 floor in
      the split-row decode (one `fmax` per score) and `cai_tanh` on the soft-cap;
      not measured here.

## Layer follow-ups that fix real limitations

- [ ] Dilated `TNNetConvolution` follow-ups (c331577c; the 6-argument
      `Create(..., SuppressBias, Dilation)`, FStruct[6]):
  - [ ] Winograd and the 2x-upsample fold decline dilation > 1 (plain path);
        fine unless a dilated 3x3 shows up hot in a GPU-box profile.

- [ ] Convolution training on OpenCL that pays for itself. Today a CNN trains
      on the CPU even under OpenCL (examples/SimpleImageClassifierGPU,
      VisualGAN, UNetSegmentation, VQGAN): `TNNetConvolution.BackpropagateOpenCL`
      exists (two device GEMMs: weight grad and input grad) but its call in
      `TNNetConvolution.Backpropagate` was commented out as "too slow" in
      37022401, and `BackpropagatePointwiseOpenCL` has no caller. The cost is
      the transfers, not the GEMMs: every backward computes OED on the host,
      re-lays out OED, the im2col patches and W^T on the host, uploads all
      three, downloads both results, and scatters the input gradient (col2im)
      on the host. Also, `ShouldBindPrevOutputOnOpenCL` refuses a trainable
      layer, so the training forward downloads every activation too.
      Plan, each step with a PoCL exact-vs-CPU gradient test (tolerance from
      docs/layer-authoring.md §3): (1) measure first, on the GPU box: per-layer
      backward time of SimpleImageClassifierGPU with the commented block
      re-enabled vs `BackpropagateFastTiledCPU`, split into upload/GEMM/download;
      (2) keep the forward im2col and `FOutputRaw` resident for the backward,
      compute OED on the device (activation-derivative opcode times the error)
      and bind it as the GEMM A operand; (3) a `cai_col2im` scatter kernel so
      the input gradient stays resident and the previous layer's backward binds
      it; (4) accumulate the weight delta on the device and download it once per
      `UpdateWeights`, not once per sample. Same route for `TNNetFullConnect`
      (whose `BackpropagateOpenCL` is live but uploads per call). Out of scope
      until (1)-(4) pay: `TNNetDepthwiseConv` / `TNNetDeconvolution` backward
      (both CPU-only today); when this lands, add `TNNetConvolution` to the
      device-backward list at the top of "OpenCL forward coverage".

- [ ] Decision for the user: inference-time release of the convolution im2col
      matrix (`FInputPrepared`). Every CPU, int8, Winograd or explicit-OpenCL
      convolution keeps it after the forward in both memory modes (the FP32
      inference implicit-GEMM OpenCL path, `ShouldOpenCLImplicitConv`, has none); for big-lama at 512x512 that is ~6.7 GB of the ~8.6 GB
      estimate (EfficientLoFTR at 640x480 needs ~6.9 GB for the same reason,
      and RT-DETR r18vd peaks at 1.86 GB at 640x640).
      Options: free/share it when the net is not trainable, or tile the
      im2col along Y. Library-wide; propose and wait for approval before coding.

- [ ] Faithful stacked-bidirectional torch import (`LoadTorchGRUInto`), GRU half
      and the multi-layer fixture: the LSTM half exists (`CellsReadInputWidth`
      builds projection-free `TNNetLSTMCell.Create(Hidden)` cells that
      `LoadTorchLSTMInto` loads), but `AddBidirectionalRecurrentStack` still
      inserts a learned `TNNetPointwiseConvLinear` projection on every
      incoming Depth≠Hidden feed for GRU cells, which torch nn.GRU does not
      have, so `LoadTorchGRUInto` rejects those stacks. Give `TNNetGRUCell`
      the same any-input-width form, then add a 2-layer bidirectional
      pico-oracle parity fixture for LSTM and GRU (the generator already
      supports the shape — `tools/torch_rnn_tiny_fixture.py`).

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
- [ ] Segment-mask MHA-builder test: `AddMultiHeadSDPAConcat` /
      `AddMultiHeadSelfAttention` already take `SegmentSource` (avSDPA only), but
      no test passes one. Assert a packed two-doc MHA stack matches the
      concatenation of independent per-document MHA runs (the builder-level
      analogue of TestSegmentMaskMatchesUnpackedBaseline).

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
      gradient and full weight-gradient FINITE-DIFFERENCE checks
      (TestDeconvolutionGradient central-differences a single weight only;
      TestDeconvolutionDepthVectorEquality checks both against an analytic
      reference; Upsample / DeMaxPool / DeAvgPool already covered). Use
      `LayerInputAndWeightGradientCheck(..., AUseDoubleAccumulator)`
      (tests/TestNeuralNumerical.pas), which checks every weight too.
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
- [ ] Backward audit for TNNetPointwiseNorm — its backward is the diagonal-
      only per-position `PointwiseMul(1/n)` approximation; TNNetL2Normalize now implements the
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
- [ ] Gradient-flow regression test — train a 12-layer ReLU MLP one epoch
      with and without a single TNNetLayerNorm/RMSNorm at the midpoint;
      assert per-layer gradient norms with the norm layer are uniformly
      bounded above the no-norm case.
- [ ] TNNetDotProducts numerical-gradient test — standalone class still
      ships; its input-gradient path (both branches A and B) is only
      forward-tested (`TestDotProductNumerical`).
- [ ] TNNetLocalConnect / TNNetDeLocalConnect input + weight gradient tests.
- [ ] Kink-region test parametric helper: with Clamp / HardShrink /
      SoftShrink / Threshold / ShiftedReLU / HardTanh all in tree, the
      "no-central-difference, hand-picked kink convention" pattern
      repeats. Capture as `AssertKinkDerivative(layer, x_kink, expected_dydx)`
      and use it first for the missing TNNetHardShrink / TNNetSoftShrink
      kink-region tests at hand-picked inputs (no central differences).
- [ ] FP-exception robustness for TNNetESwish at truly extreme inputs
      (surfaced 2026-05-31 while writing the saturation tests above): its
      `Exp(-beta*x)` overflows the RTL `Exp` around beta*x~-570 (EOverflow)
      instead of returning a finite value. The landed test stays inside the
      safe-but-saturating band (beta*x up to ±625) and documents the limit.
      Decide a policy: either clamp the offending intermediate (saturate the
      sigmoid to its asymptote) so the layer stays finite at any input like
      HardTanh/SoftCapping do, or document the input-range contract. Then
      extend the test to the far-extreme range under the chosen policy.
      TNNetSoftSign no longer squares 1+|x| (1641985d computes the derivative
      as InvDenom * InvDenom), so it should now be finite at |x|~1e30, but
      `TestSoftSignExtremeInputSaturation` still stops at ±1e6 and its comment
      still describes the old overflow: extend it to the far-extreme range.
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

## Vision / generative & accelerator batch 2026-06-27b

- [ ] NF4 double-quantized absmax (`neural/neuralnf4.pas` dequant, 90b6c931):
      handle `*.nested_absmax` / `*.nested_quant_map` / `*.quant_state.*` (the
      secondary int8 quant + nested absmax + offset). `LoadNF4QuantizedTensorFlat`
      RAISES a clear `EPretrainedImportError` on these nested/double-quant markers
      (and on a non-FP32 absmax sibling); `TestNF4ImporterRejectsDoubleQuant`
      covers the rejection (fixture `pico_nf4_doublequant.safetensors`). To
      finish: expand the int8-quantized absmax (nested absmax + offset +
      quant_map) to FP32 upstream, then drop the guard. A real `bnb-4bit`
      Llama/Qwen slicer fixture is RAM/network-gated; today's parity is the
      random pico numpy oracle only.

## Tiled VAE decode follow-ups (from 2e28a44b)

- [ ] Size-agnostic VAE-decoder Compute path. `TiledVaeDecode`
      (neuralpretrained.pas) already reuses one decoder net across tiles, but
      that net is built at a fixed tile latent size (its `TNNetInput` defines the
      tile and edge tiles are clamped to it), because `TNNet.Compute` errors on a
      latent whose size differs from the fixed `TNNetInput`. A size-agnostic
      forward (re-derive layer shapes from the actual input volume on each
      Compute, or a Resize-input hook) would let a single resident net decode
      arbitrary tile sizes — also unblocks variable-resolution inference
      generally. Scope first: audit which layers in the VAE decoder graph
      hard-bind to TNNetInput size.
- [ ] Tiled VAE decode is an APPROXIMATION because the decoder MID block has
      GLOBAL self-attention (same caveat as diffusers): per-tile attention can't
      see across seams. Follow-up: a windowed/overlapped-attention tiled mode, or
      document+measure the approximation error vs whole-image decode on a real
      (trained) checkpoint so users know the quality trade-off. Pure host /
      analysis; no new layer.
