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
  - [ ] real torchvision .pth (pickle) load path: today the importer reads
        safetensors only; the pico fixture is a numpy float64 oracle (no
        torchvision). Also: CAI maxpool (ceil sizing + edge-clamped windows + zero
        pad) diverges from PyTorch maxpool (floor + -inf pad) — the parity test
        mirrors CAI semantics; a real-checkpoint top-1 check needs reconciling the
        stem maxpool or documenting the gap.
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
    - [ ] real-checkpoint parity (runwayml/stable-diffusion-v1-5 unet) once
          diffusers is installable; verify the SD attention_head_dim->num_heads
          interpretation on the real config (the importer derives NumHeads =
          block_out_channels[0]/attention_head_dim and assumes a constant head
          count across blocks — SD1.5's 8-heads-everywhere case holds, but SDXL /
          configs with a per-block attention_head_dim list need a per-block heads
          array wired through TSDUNetConfig).
    - [ ] use_linear_projection=True variant (SD2.x / SDXL use Linear proj_in/out
          instead of 1x1 conv) + transformer_layers_per_block > 1 (SDXL stacks
          several BasicTransformerBlocks per Transformer2DModel) — the v1 importer
          hardcodes a 1x1-conv proj and exactly one transformer block.
    - [ ] add_embedding / class/time-aug conditioning (SDXL micro-conditioning),
          and the end-to-end LatentTextToImage capstone (CLIP text -> this UNet
          -> scheduler loop -> VAE decoder).
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
- [ ] Real-ESRGAN / ESRGAN importer follow-ups (BuildRRDBNet[FromSafeTensors][Ex]
      + TRRDBNetConfig LANDED in neuralpretrained.pas; RRDBNet x4 with
      NEAREST-interpolate conv upsampling via TNNetDeMaxPool(2), parametrized
      TNNetLeakyReLU.Create(pAlpha) 0.2 slope; pico parity TestRRDBNetParity
      max|diff| < 1e-4 vs a numpy float64 oracle): (a) realesrgan .pth pickle load
      (TNNetTorchBinReader path); (b) real x4 upscale of a tiny PNG end-to-end
      example; (c) scale=2 / other scales (currently only scale=4 = two upsample
      stages wired).
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
      offline here); (b) expose LPIPS as a backprop TRAINING LOSS head so the SR
      examples can opt into perceptual fine-tuning (the VGG build already enables
      input/error collection, so the gradient path exists).
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
- [ ] FP16 (half-precision) OpenCL compute path for the dot-product/matmul
      offload. ENV NOTE (2026-06-27): the only OpenCL device available here is
      PoCL on the host CPU, which does NOT advertise `cl_khr_fp16` (clinfo shows
      "Half-precision Floating-point support (n/a)"). The detection + opt-in
      buffer/kernel path can be written and the FP32-fallback verified, but the
      actual half-precision kernel CANNOT be exercised/parity-tested here — defer
      until a `cl_khr_fp16` device is available, or scope a host-side
      half<->float conversion test only and mark the GPU kernel unverified.
      offload. Every GEMM kernel in `neural/neural.cl` (cai_dot_product,
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
      still raise EHFTokenizerError. This now partially UNBLOCKS the mBART/NLLB
      BART-family follow-up (a, above) and the DeBERTa-v3 import. No residual
      non-ASCII parity gap. BYTE
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
  - [X] OpenCL offload of the same accumulation (via the shared dot-product kernel,
        like FullConnect/Convolution). FORWARD conv1d LANDED: EnCodec (`RunEnCodecConv`
        — `TEnCodecModel` + MusicGen EnCodec decode) and HiFiGAN/Vits (`RunHiFiGANConv`)
        run the per-stage im2col GEMM as one `TDotProductSharedKernel` call, opt-in +
        default OFF via the `EnableConvOpenCL`/`SetConvOpenCLMinWork` gate.
        `TestEnCodecOpenCLConvParity` GREEN on BOTH the default (CPU==CPU, trivially
        exact) and `-dOpenCL` builds. Verified on the real PoCL CPU device (FP32):
        end-to-end EnCodec decode max|diff| = 5.96e-8 << 1e-4 (per-stage forward conv
        GEMM matched the AVX DotProduct to < 2e-7). `EnableConvOpenCL` now self-tests
        the device kernel with a tiny known dot product and stays OFF (graceful CPU
        fallback, no crash) if it can't compute — covers the case where the kernel
        source `neural.cl` is not reachable from the run dir (added `../neural` and
        `../../neural` repo-relative lookups so the tests/examples dirs find it).
        REMAINING:
    - [X] OpenCL offload of the ConvTranspose1d (upsample) overlap-add accumulation
          (EnCodec + HiFiGAN/Vits); the per-(o,k2)-tap in-channel contraction is the
          same shared-kernel-shaped GEMM but scatters into an overlap-add buffer.
          DONE: RunConvTransposeGemmOpenCL runs the transposed-im2col GEMM as ONE
          TDotProductSharedKernel call (As = WT repacked [i*(OutCh*K)+(o*K+k2)],
          Bs = InT [t*InCh+i], Size=InCh, rows=OutCh*K, cols=InLen); the result
          Res[t*(OutCh*K)+(o*K+k2)] then scatters overlap-add into Full (EnCodec)
          / bias-preset Pad-trimmed OutSig (HiFiGAN) at stride positions. Same
          opt-in EnableConvOpenCL / SetConvOpenCLMinWork gate + ConvOpenCLSelfTest
          graceful-fallback as the forward path; AVX/CPU stays the fallback.
          TestEnCodecOpenCLConvParity (decoder upsamplers) + new
          TestHiFiGANOpenCLConvParity exercise the genuine PoCL FP32 device path
          (ConvOpenCLEnabled=TRUE): EnCodec end-to-end max|diff| 5.96e-8, HiFiGAN
          1.79e-7, both << 1e-4. Full suite green on default AND -dOpenCL builds.
    - [ ] OpenCL offload of the Mimi (`RunMimiConv`) / DAC (`RunDACConv`) holders —
          blocked on a Double-precision shared dot-product kernel (their oracle gate
          is `< 1e-4` against a float64 reference; a Float kernel would not hold it).
      Design note (the landed forward path follows this): channel-major layout
      means the contraction (sum over in-channels for a fixed kernel tap) is
      depth-axis-contiguous — exactly the case `TNNetVolume.DotProduct` / `MulAdd`
      already vectorize — so the win is reusing the existing AVX volume primitives
      (or a small im2col-into-DotProduct reshape) for the per-tap accumulation
      instead of triple-nested scalar loops. Gate any remaining rewrite on the
      existing parity tests (TestHiFiGANSynthesisParity / TestVitsSynthesisParity /
      EnCodec round-trip) staying `< 1e-4`, and re-profile decode wall-clock
      before/after.
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
  - [ ] Swap the `TNNetMinLSTM` trunk (gates depend on x_t only) for a VANILLA LSTM with a
        true cell state + recurrent gate feed (pyannote uses `nn.LSTM`), so real weights
        load without re-training; needs a landed vanilla-LSTM cell first.
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
  - [ ] No top-level BuildFromPretrained dispatch entry / classifier head /
        ForImageClassification wrapper yet (builder returns token hidden states;
        use_mean_pooling pooler LayerNorm + patch mean left to the caller).
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

## Layer follow-ups that fix real limitations

- [X] AVX-vectorize the scalar backward Jacobian of TNNetRMSNorm and
      TNNetLayerNorm. Their `Compute` already uses the AVX `GetSumSqr`/`Mul`
      primitives, but `Backpropagate` still walks `FData[Cnt]` element-by-element
      to form `dxhat = OutputError .* gamma`, the running reductions
      (`SumDxHat`, `SumDxHatXHat`) and the final
      `dx = invRMS*(dxhat - xhat*mean(dxhat*xhat))` scatter. The whole-volume case
      is contiguous, so each loop maps onto the existing neuralvolume primitives:
      `Mul` for the elementwise gate, `DotProduct(dxhat, xhat)` (and a plain
      `GetSum`/`DotProduct` against a ones-style accumulation for LayerNorm's
      mean-subtraction term), then two `MulAdd` writes into
      `FPrevLayer.FOutputError` — exactly the rewrite already landed for the
      per-token sibling `TNNetTokenRMSNorm.Backpropagate`, which can be copied as
      the template. Preserve the `-FLearningRate` convention and the gamma/beta
      delta path; the existing numerical-gradient tests pin correctness. Real value:
      RMSNorm/LayerNorm sit on the hot path of every transformer and ConvNeXt-style
      vision block, and the backward pass is currently the only scalar half left.

- [X] Port torch `nn.LSTMCell` / `nn.GRUCell` — the standard fully-connected
      recurrent cells with TRUE recurrent gating (each gate sees the previous
      hidden state h_{t-1}, and the LSTM carries a separate cell state c_t), i.e.
      a direct port of torch `nn.LSTMCell`/`nn.GRUCell`. This is a genuine gap, NOT
      a near-duplicate of anything in tree: `TNNetMinLSTM`/`TNNetMinGRU` deliberately
      make the gates depend on x_t ONLY (so they parallel-scan), `TNNetSLSTMCell`/
      `TNNetMLSTMCell` are the xLSTM exp-gated / matrix-memory variants, and
      `TNNetConvLSTMCell`/`TNNetConvGRUCell` are convolutional. None implement the
      classic Hochreiter/Cho gates `i,f,o,c̃ = σ/tanh(W_x·x_t + W_h·h_{t-1} + b)`
      with a real recurrent W_h. Full forward + BPTT (the frozen-h_{t-1} forward
      gotcha noted for sLSTM applies), input & weight numerical-gradient tests, and
      a weight layout that maps the torch fused `weight_ih`/`weight_hh` (4·hidden /
      3·hidden row-packed i,f,g,o gate order) so real `nn.LSTM`/`nn.GRU` checkpoints
      load without re-training. Immediate payoff: UNBLOCKS the pyannote
      real-checkpoint follow-up (lines 977-979, which needs exactly this to drop in
      `pyannote/segmentation-3.0`'s `nn.LSTM` trunk) and any future speech / seq2seq
      importer whose recurrent core is a stock LSTM/GRU. Bidirectional + multi-layer
      stacking reuse the existing forward+time-reversed-concat builder idiom used for
      the MinLSTM trunk (no new wiring needed).
      DONE: `TNNetLSTMCell` (torch i,f,g,o gate order, 12 tensors: `W_i*`/`W_h*`
      Depth×Depth + folded ih+hh biases, `b_f` init +1, separate cell state `c_t`)
      and `TNNetGRUCell` (torch r,z,n order, 11 tensors: `W_i*`/`W_h*` + folded
      `b_r`/`b_z`, SEPARATE `b_in`/`b_hn` because `b_hn` rides inside the reset-gated
      candidate term, 1 spare bias) in `neural/neuralnetwork.pas`. Both: full forward
      with frozen-`h_{t-1}` snapshot + exact BPTT (`dL/dc_t`+`dL/dh_t` for LSTM,
      `dL/dh_t` incl. `z_t⊙h_{t-1}` leakage and `r_t`/`whn` coupling for GRU);
      registered in both `CreateLayer` dispatch tables; input + weight
      numerical-gradient tests + serialization round-trip in `TestNeuralNumerical`
      (max abs grad err ~1.7e-3, well under 1e-2). README rows added.
      - [ ] FOLLOW-UP (still open): bidirectional + multi-layer stacking for
            `TNNetLSTMCell`/`TNNetGRUCell` (and a real `nn.LSTM`/`nn.GRU`
            num_layers>1 / bidirectional=True checkpoint importer). Reuse the
            existing forward + time-reversed-concat builder idiom from the MinLSTM
            trunk; no new layer math is needed, just a stacking/reverse builder and
            the per-direction/per-layer weight-slab wiring. Unblocks the pyannote
            `segmentation-3.0` bidirectional-LSTM trunk drop-in (lines ~977-995).

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

- [ ] AVX-vectorize the rank-1 outer-product state update in TNNetDeltaNet and
      TNNetGatedLinearAttention. Both already use the AVX `DotProduct` for the
      per-timestep input/key/value/gate projections, but the state matrix carry
      itself is a fully scalar nested `for d, e: S[d,e] := alpha*S[d,e] + beta*k[d]*v[e]`
      double loop in BOTH forward and backward. With the state laid out row-major
      (depth-contiguous in `e`) each output-row update is exactly one
      `MulAdd(ptrS_row, ptrV, beta*k[d], D)` after an optional `Mul` by the
      forget scale — a drop-in over the existing neuralvolume primitives, with the
      backward `dS`/`dk`/`dv` accumulation mapping onto `DotProduct`/`MulAdd` the
      same way. These O(SeqLen*D^2) state writes dominate the layer cost, so this is
      the largest remaining scalar hot loop in the linear-attention family. Preserve
      the `-FLearningRate` convention; the landed numerical-gradient tests pin
      correctness. (TNNetTitansMemory / TNNetTestTimeTraining share the identical
      per-row pattern and can reuse the same helper — see the next entry.)

- [ ] AVX-vectorize the test-time inner-optimizer weight-matrix updates in
      TNNetTestTimeTraining and TNNetTitansMemory. The per-timestep gradient-descent
      step on the inner memory (`W_lin -= eta*r outer k`, and for TitansMemory the
      momentum + forget-gated `S := (1-forget)*S + momentum*grad` over the D x H
      weight slabs) is the same rank-1 row-wise update as the DeltaNet/GLA entry
      above, currently a scalar nested loop forward and backward. Route each weight
      row through `MulAdd`/`Mul`/`DotProduct` over the contiguous H axis; share one
      `RankOneUpdate(ptrW, ptrA, ptrB, scale, rows, cols)` helper between this and
      the DeltaNet/GLA work so there is a single vectorized primitive, not four
      copies. These layers run an optimizer step per token, so the inner matmul is
      the dominant cost. Numerical-gradient tests stay the correctness oracle.

- [ ] AVX-vectorize the block / factor GEMMs of TNNetMonarchLinear and
      TNNetKroneckerLinear. Both never materialize the full weight, so their forward
      and backward are small dense matrix-vector products inside nested scalar
      `for row, col: acc += W[row,col]*x[col]` loops (Monarch's two per-block m x m
      butterflies, Kronecker's B-then-A two-phase contraction). Each inner
      accumulation is a contiguous dot product once the block/factor rows are walked
      row-contiguous, so they map onto `DotProduct` (forward) and `MulAdd` (backward
      weight/input gradient scatter) directly; where the natural axis is strided
      (Monarch's P / P^T permutes), reorder to the contiguous run first as was done
      for the attention output-accumulation loops. These structured-dense layers are
      pitched as a cheaper FullConnect, but the scalar inner loop currently gives up
      most of that win on AVX builds.

- [ ] AVX-vectorize the complex spectral matmul of TNNetSpectralConv1D. The FFT
      forward/inverse are already vectorized, but the channel-mixing contraction in
      the frequency domain is a scalar nested `for co, m, ci` complex multiply-add
      (`yr += Wr*xr - Wi*xi; yi += Wr*xi + Wi*xr`) in both ComputeCPU and
      BackpropagateCPU. With the real and imaginary mode weights stored as separate
      contiguous planes the per-output-channel reduction over input channels becomes
      four real `DotProduct`s (the standard 4-multiply complex GEMM), turning the
      O(Modes*InDepth*OutDepth) inner loop into AVX dot products. This is the FNO
      hot path; keep the Modes low-pass truncation and the learnable-complex-weight
      gradient exactly as-is, pinned by the existing numerical-gradient test.

- [ ] OpenCL offload for the spatial im2col GEMM of TNNetDeformableConv and
      TNNetGroupConvP4. Both spend their forward in a scalar
      `for oy, ox, co, fy, fx, ci` convolution accumulation (DeformableConv with a
      bilinear-sampled patch per tap, GroupConvP4 with the 4 rotation-tied weight
      copies). Once the (possibly sampled) input patch is gathered into an im2col
      buffer the accumulation is an ordinary patch x weight GEMM, so it can reuse the
      same `EnableConvOpenCL` / `SetConvOpenCLMinWork` gated kernel path already
      landed for the EnCodec/HiFiGAN conv1d and ConvTranspose1d offload (host
      round-trip as the fallback). Keep the bilinear-sample gather and the rotation
      weight-folding on the CPU; only the dense contraction goes to the device.
      Pin parity with a SDPAOpenCLParity-style exact-vs-CPU test on the PoCL device.

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
