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

- [ ] ResNet importer follow-ups (BuildResNetFromSafeTensors[Ex] LANDED, commit
      317a19c: torchvision resnet18/50 state_dict, conv-BN fold at load, config
      reader/ToString, resnet18 parity 1.0e-6 vs a numpy float64 oracle —
      torchvision not installed; ConvNeXt LayerScale+GRN+depthwise-7x7 is the
      modern-CNN stretch goal on the same path):
  - [ ] resnet50 Bottleneck (expansion 4) is CODED but not parity-tested — add a
        pico Bottleneck fixture + TestResNet50...Parity. resnet34 likewise coded,
        untested. resnet101/152 + ConvNeXt remain out of scope.
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
  - [ ] the SD UNet itself (the remaining piece for end-to-end latent text-to-image).
- [X] Segment Anything (SAM) image-ENCODER importer LANDED (facebook/sam-vit-base)
      — the repo's first PROMPTABLE-segmentation importer. BuildSAMFromSafeTensors
      [Ex] + the reusable BuildSAMVisionTower + TSAMConfig + ReadSAMConfigFromJSON
      File/SAMConfigToString (neuralpretrained.pas). The ViT-det image encoder:
      biased patch conv -> learned 2-D absolute pos_embed (TNNetCellBias, no CLS,
      no flatten) -> num_hidden_layers pre-LN blocks via the NEW self-contained
      leaf TNNetSAMVisionAttention (windowed attention with the SAM zero-padding
      window partition + global attention for global_attn_indexes blocks + the
      MViTv2 DECOMPOSED query-dependent rel-pos bias Q.rel_pos_h + Q.rel_pos_w,
      distinct from Swin's query-INdependent table) -> neck (conv1 1x1 no-bias ->
      LayerNorm2d-over-channels -> conv2 3x3 pad1 no-bias -> LayerNorm2d). MLP uses
      the exact-erf GELU (TNNetGELUErf). Pico parity < 1e-4 vs HF SamModel float64
      image embedding (TestSAMEncoderParity; tools/make_pico_sam_fixture.py ->
      tests/fixtures/tiny_sam.*). Example examples/SegmentAnything: encoder forward
      on a synthetic image + a one-click cosine-similarity mask PPM (encoder-only
      proxy). Gotchas baked in: CAI volume X axis is the FAST axis so the SAM grid
      maps gridW->X, gridH->Y (GetRawPtr(col,row)); rel_pos tensors arrive FLAT
      from LoadTensorFlat so the row count is Size div HeadDim (not SizeX);
      hidden_act='gelu' is exact-erf (NOT tanh). Follow-ups DEFERRED:
      (a) the lightweight TWO-WAY MASK DECODER (point/box prompt embeddings ->
          cross-attention blocks -> upscaled mask) — the stretch goal, reuse
          TNNetCrossAttention two-source wiring; the example's click->mask is a
          cosine proxy until it lands;
      (b) ViT-L / ViT-H variants (config-driven; only block count/width differ);
      (c) a real-checkpoint slicer (make_pico_sam_fixture style) for
          facebook/sam-vit-base parity from the real download;
      (d) multi-mask output + box/multi-point prompts (v1 is single-point intent).
  (DPT / Depth-Anything monocular-depth importer LANDED: BuildDPTFromSafeTensors
   + ReadDPTConfigFromJSONFile/DPTConfigToString, DINOv2 backbone reuse + DPT
   reassemble/RefineNet-fusion neck, pico parity < 1e-4 via tiny_dpt.* fixtures
   and TestDPT{Config,DepthEstimation}Parity. Follow-ups deferred: real-checkpoint
   slicer for depth-anything/Depth-Anything-V2-Small-hf, the dpt-hybrid (resnet
   backbone) variant, and an examples/MonoDepth grayscale demo.)
- [ ] Mask2Former universal-segmentation importer
      (BuildMask2FormerFromSafeTensors, e.g. facebook/mask2former-swin-tiny-*-semantic)
      — a third, architecturally DISTINCT segmentation vertical: mask-classification
      set-prediction (a fixed set of learned queries each predicting one binary mask +
      a class), unifying semantic/instance/panoptic in one head. Different from the
      landed SegFormer (per-PIXEL argmax) and the tracked Mask R-CNN (RoIAlign on
      region proposals) — there are NO proposals and NO per-pixel classifier. Reuses
      the landed DETR set-prediction machinery (learned object queries + a transformer
      decoder, already imported for detection) and a Swin/ResNet backbone (Swin is a
      landed classifier importer); the new pieces are the lightweight pixel decoder
      (a small FPN-style multi-scale conv pixel embedding, reusing the Mask R-CNN FPN
      blocks) and the masked-attention decoder layer (cross-attention restricted to the
      current mask foreground) plus the dot-product query-embedding x pixel-embedding
      -> per-query mask logits. Scope v1 to semantic inference (argmax over query
      class x mask). Pico parity vs HF float64 on the mask logits for a fixed image +
      an examples/UniversalSegmentation that writes one segmentation overlay on a tiny
      CPU image.
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
- [ ] Inception-v3 / GoogLeNet importer (BuildInceptionV3FromSafeTensors,
      torchvision) — PARTIAL (commit 503540b): the branch-concatenation block
      builder LANDED (AddInceptionAModule runs 4 parallel branches
      1x1 / 5x5 / 3x3dbl=5x5-as-two-3x3 / pool concatenated on the channel axis
      via TNNetDeepConcat), plus BuildInceptionV3 scaffold, config reader/ToString,
      ResNet conv-BN fold reuse (LoadResNetConvFoldBN), pooled-feature (FID backbone)
      tap via out PoolFeatureIdx, and a pico parity test <1e-4 vs a numpy float64
      oracle on the InceptionA-shaped sub-net. NOT yet a usable real-checkpoint
      importer. REMAINING to import a real torchvision inception_v3:
  - [ ] Strided grid-reduction modules InceptionB / InceptionD (parallel
        stride-2 conv + pool branches), the full stem (Conv2d_1a..4a + maxpools),
        InceptionE, and the torchvision avg-pool branch (the pico used a
        grid-preserving portable maxpool); wire the full module sequence + real
        weight loading and parity-test against a full-size float64 oracle.
  - [ ] Rewire neuralimagemetrics FID onto this backbone once the full net lands
        (today FID uses placeholder features).
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
- [ ] ControlNet spatial-conditioning importer (BuildControlNetFromSafeTensors, e.g.
      lllyasviel/sd-controlnet-canny) — adds spatial control (edge / depth / pose
      map -> image) to latent diffusion: a trainable COPY of the SD UNet encoder +
      mid block whose per-resolution residuals are added into the frozen base UNet
      via zero-initialised 1x1 convs, plus a small conv "hint" stem that embeds the
      control image into the latent grid. Genuinely new code is only the zero-conv
      residual injection wiring and the hint encoder; the encoder blocks reuse the
      SD UNet importer path. DEPENDS ON the open SD UNet importer (the VAE-decoder
      follow-up's deferred piece) — track as its natural successor. Pico parity vs a
      diffusers float64 oracle on the down/mid residual tensors for a fixed control
      image; an examples/ControlNetCanny that conditions generation on a hand-drawn
      edge map once the base UNet lands. First conditioning-by-feature-injection model.
- [ ] RandAugment / TrivialAugment automatic augmentation policy in
      neuraldatasets.pas — the repo has Mixup (landed) and CutMix (tracked) but NO
      single-image geometric/photometric augmentation policy; CV training augmentation
      is currently just flips + pad-crop. Port the torchvision transforms-v2 staple:
      a fixed op bank (autocontrast, equalize, rotate, shear-x/y, translate-x/y,
      posterize, solarize, color/contrast/brightness/sharpness) over a TNNetVolume,
      with RandAugment (N ops at fixed magnitude M) and the parameter-free
      TrivialAugment (one op, magnitude drawn uniformly) selection policies, plus
      RandomErasing/Cutout. Wire it as an optional hook in the TNeuralImageFit
      augmentation path so existing CIFAR examples can opt in. New code is the op
      bank + the two sampling policies; reuses existing volume rotate/shear/color
      primitives where present. Adds a measurable top-1 lift on the SimpleImageClassifier.
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
  - [ ] resblock type "2" (the alternative MRF wiring) — currently rejected loudly
        in ReadHiFiGANConfigFromJSONFile.
  - [ ] real-checkpoint smoke: resynthesize a clip with a downloaded `hifigan` /
        SpeechT5HifiGan generator (weight_norm fold path) and write it via
        SaveVolumeToWav16 (offline + RAM-gated here, so deferred).
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
  - [ ] the STOCHASTIC duration predictor (`VitsStochasticDurationPredictor`, the
        spline-flow `use_stochastic_duration_prediction=true` path) — currently
        rejected loudly in `ReadVitsConfigFromJSONFile`; the deterministic readout is
        implemented (the MMS-TTS default).
  - [ ] MULTI-SPEAKER models (`num_speakers>1` / `speaker_embedding_size!=0`, the
        global-conditioning `cond` convs into the WaveNet/duration/decoder) —
        rejected loudly.
  - [ ] the `VitsTokenizer` (uroman/phonemizer + char vocab) so a STRING can be
        synthesized; text is supplied as token ids for now.
  - [ ] real-checkpoint smoke: synthesize a sentence with a downloaded
        `facebook/mms-tts-eng` / `kakao-enterprise/vits-ljs` and write it via
        `SaveVolumeToWav16` (offline + RAM-gated here, so deferred).
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
  - [ ] freq_ratio > 1 (the real laion 1024-frame/64-mel `freq_ratio = 4` layout):
        the `reshape_mel2img` mel-crop reorganization + the final group-2D-CNN reshape
        before the avgpool are currently identity-only and loudly REJECTED. Add the
        general mel2img + group-CNN glue (a reshape/permute composition or a minimal
        helper) and a freq_ratio=4 pico fixture.
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
      DEPENDS ON the WAV writer + HiFi-GAN vocoder tasks above.
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
  - [ ] a harder synthetic task (overlapping classes / lower SNR / more keywords)
        so the smoke trains for more than ~2 epochs before validation hits the
        default TargetAccuracy and early-stops.
  - [ ] a 16 kHz resampler in neuralaudio so `--full` accepts non-16 kHz WAVs
        directly instead of requiring an ffmpeg pre-pass.
- [X] HTDemucs / Demucs music SOURCE-SEPARATION importer
      (BuildDemucsFromSafeTensors[Ex] + TDemucsConfig +
      ReadDemucsConfigFromJSONFile / DemucsConfigToString, facebook/demucs /
      htdemucs) — LANDED the TIME-DOMAIN (waveform) Demucs (v2/v3) U-Net, the
      FIRST audio source-separation model and a genuinely new audio OUTPUT
      modality: one mixed stereo track in, four stems out (drums / bass / other
      / vocals). Distinct from every landed audio model — EnCodec (codec),
      Wav2Vec2/Whisper (recognition), HiFi-GAN/VITS (synthesis), CLAP
      (embedding) — none separate sources. Built as a self-contained
      channel-major TNNetDemucs holder (the HiFi-GAN/VITS/EnCodec convention),
      REUSING THiFiGANConv / RunHiFiGANConv for every Conv1d / ConvTranspose1d
      (no new leaf layer): an ENCODER of `depth` blocks (strided Conv1d -> ReLU
      -> 1x1 Conv1d -> GLU, each saving a U-Net skip) -> a bi-LSTM bottleneck
      (`lstm_layers` stacked bidirectional cells run INLINE in the holder, there
      being no bidirectional-LSTM leaf layer, then Linear(2C -> C)) -> a DECODER
      of `depth` blocks (skip-add with Demucs center_trim -> Conv1d -> GLU ->
      ConvTranspose1d -> ReLU except the last) reshaped to (sources,
      audio_channels, time) and center-trimmed to the input length. Conv /
      nn.LSTM weights load as plain folded tensors (encoder.i.{0,2}, lstm.*_l*
      + *_reverse, lstm_linear, decoder.i.{0,2}). Pico parity vs a SELF-CONTAINED
      numpy float64 oracle (transformers ships no Demucs, so the oracle is
      hand-built from the published architecture math — the *_tiny_fixture
      precedent) on two short fixed stereo waveforms
      (tools/make_pico_demucs_fixture.py -> tests/fixtures/tiny_demucs*,
      TestDemucsSeparationParity max|diff| < 1e-4, PASSING) + an
      examples/MusicSourceSeparation that splits a short clip and writes the four
      stems via SaveVolumeToWav16 (no-arg pico smoke + real-checkpoint path).
      OPEN FOLLOW-UPS: the hybrid TIME+SPECTRAL HTDemucs spectral branch (STFT +
      a parallel 2-D spectral U-Net summed with the time branch); the v3
      CROSS-DOMAIN TRANSFORMER bottleneck (in place of the bi-LSTM); input/output
      NORMALIZATION (centering + std rescale, `normalize=true`) and weight
      rescaling; and a real downloaded checkpoint end-to-end separation within
      the ~5 min / ulimit budget.
- [X] Moonshine streaming-ASR importer (BuildMoonshineFromSafeTensors[Ex] +
      TMoonshineConfig + ReadMoonshineConfigFromJSONFile / MoonshineConfigToString +
      MoonshineEncoderLength, UsefulSensors/moonshine-tiny|base) — LANDED the ENCODER
      (the v1 parity surface). A SECOND speech-to-text architecture, distinct from the
      landed Whisper importer: NO fixed 30 s mel frontend — the raw-waveform conv stem
      (conv1 k127 s64 bias-free + tanh; GroupNorm(1) over the (T,C) block; conv2 k7 s3
      + erf-GELU; conv3 k3 s2 + erf-GELU) convolves features directly off the 16 kHz
      waveform so compute scales with actual audio length; PRE-norm BIDIRECTIONAL
      encoder with PARTIAL RoPE (partial_rotary_factor, the Phi-3 slice wiring reused
      verbatim), bias-free q/k/v/o, bias-free (gain-only) TNNetTokenLayerNorm norms,
      erf-GELU fc1/fc2 MLP. Pico parity vs the HF MoonshineModel float64 oracle on
      encoder hidden states (tools/make_pico_moonshine_fixture.py ->
      tests/fixtures/tiny_moonshine*, TestMoonshineEncoderParity, max|diff| < 1e-4,
      PASSING) + a DETERMINISTIC examples/MoonshineTranscribe smoke (no-arg fixture run
      contrasting the length-scaled frame count/latency vs Whisper's fixed cost; real
      checkpoint path supported). OPEN FOLLOW-UPS: the RoPE + SwiGLU DECODER (cross-
      attending the encoder states via T5EncoderStatesInput; reuse
      DecodeSeq2SeqGreedy/BeamSearch) + the GQA encoder_num_key_value_heads != heads
      path (wired but only exercised at full multi-head in the pico) + the
      MoonshineTokenizer for end-to-end WAV -> text transcription.
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

- [ ] ImageNet top-1 / top-5 parity eval harness for the imported vision backbones
      (EvaluateImageNet + ImageNetReport in neuralimagemetrics.pas or a sibling, plus
      examples/ImageNetEval). Today there is NO end-to-end accuracy check for the
      landed classifier importers (ResNet / ViT / Swin / DINOv2 / MobileNetV3 / VGG /
      Inception-v3): each importer's parity test only compares raw logits on one or
      two tensors, which catches a transposed weight but NOT a wrong preprocessing
      pipeline (resize/crop/normalize) or a label-permutation. The harness loads a
      small folder of labelled ImageNet-val JPEGs, applies the importer's declared
      ImageSize + csImageNetMean/csImageNetStd (already in neuraldatasets.pas) with
      the correct resize-then-center-crop, runs the net, and reports top-1 / top-5
      accuracy with a confusion sample. This is the missing import-VERIFICATION
      backstop mirroring what MMLUEval/PerplexityEval do for the LLM side.

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

- [ ] Mask R-CNN instance-segmentation importer + a RoIAlign primitive
      (BuildMaskRCNNFromSafeTensors, e.g. torchvision maskrcnn_resnet50_fpn) — the
      FIRST instance-segmentation vertical (per-OBJECT binary masks, distinct from
      DETR's boxes-only, SegFormer's single dense class map, and SAM's prompt-driven
      mask). Reuses two landed pieces — the ResNet-50 backbone importer and the
      conv-BN-fold loader — and adds the FPN top-down feature pyramid (lateral 1x1 +
      3x3 + nearest upsample, same blocks the tracked YOLO neck needs) plus the new
      RoIAlign pooling primitive (bilinear-sampled fixed-size crop of a proposal box
      from the chosen pyramid level — the genuinely new layer, sibling to the landed
      DeformableConv bilinear sampler, with full input numerical-gradient coverage in
      TestNeuralNumerical.pas). Scope v1 to INFERENCE with externally supplied
      proposal boxes (skip training the RPN/anchors; feed a handful of boxes) ->
      RoIAlign -> the box head (class + refined box) and the small mask head (4x conv
      -> deconv -> per-class HxW mask). Pico parity vs a torchvision float64 oracle on
      the mask-head logits for a fixed proposal + an examples/InstanceSegmentation that
      overlays one object mask on a tiny CPU image. RoIAlign also unblocks any future
      two-stage detector (Faster R-CNN box head).

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
- [ ] MMDiT (Stable Diffusion 3 / FLUX.1) text-to-image transformer importer
      (BuildMMDiTFromSafeTensors, e.g. stabilityai/stable-diffusion-3-medium or a
      small Flux-schnell config) — the DUAL-STREAM joint-attention DiT, architecturally
      DISTINCT from the landed class-conditional DiT (BuildDiTFromSafeTensors) and the
      tracked single-stream cross-attention PixArt-alpha. The genuinely new piece is the
      MMDiT block: image tokens and text tokens carry SEPARATE per-stream
      projections/MLPs/adaLN modulations but are CONCATENATED for ONE JOINT
      self-attention pass (text and image attend to each other symmetrically), then
      split back — not the image->text CROSS-attention of PixArt. Everything else is
      landed: patch embed, the T5 + CLIP prompt towers (BuildT5/BuildClip), the VAE
      decoder (BuildVaeDecoderFromSafeTensors), and a RECTIFIED-FLOW Euler sampler
      (the examples/FlowMatching velocity-field loop). Scope v1 to inference on one
      denoise step. The cheapest path to a REAL modern text-to-image checkpoint and a
      second route (besides the tracked PixArt) to the LatentTextToImage capstone with
      NO SD UNet. Pico parity vs a diffusers float64 oracle on one block's joint-attention
      output + reuse make_pico_*_fixture.py.
- [X] PaliGemma vision-language importer (BuildPaliGemmaFromSafeTensors[Ex] +
      TPaliGemmaConfig + ReadPaliGemmaConfigFromJSONFile/PaliGemmaConfigToString,
      neuralpretrained.pas) — LANDED the first PREFIX-LM vision-language importer. The
      genuinely NEW code is the block-bidirectional prefix mask: added a transient
      TNNetScaledDotProductAttention.PrefixLen property (NOT serialized) + the net-wide
      TNNet.SetAttentionPrefixLen driver (neuralnetwork.pas). In the SDPA score loop a
      key j is attendable for query i iff causal-allowed (j<=i) OR both i,j are in the
      prefix block [0..PrefixLen-1] — so the image tokens + prompt (token_type_id 0)
      attend bidirectionally and only the generated suffix (token_type_id 1) is causal.
      PrefixLen=0 default is bit-identical to the old pure-causal path (all SDPA
      numerical regressions pass). EVERYTHING else is reused: SigLIP tower
      (BuildSigLIPVisionTower in feature mode, SelectHiddenLayer=0 so post_layernorm IS
      applied — PaliGemma uses last_hidden_state, the OPPOSITE of LLaVA's -1 feature),
      the LLaVA splice (LlavaAssembleEmbeddings), and the stock Gemma decoder
      (BuildLlamaFromTensorReaderWithConfig, gemma model_type). NEW pieces:
      BuildPaliGemmaProjector (a SINGLE biased linear, no gelu / no 2nd layer unlike
      LLaVA) and PaliGemmaRunLogits (the prefix-LM forward, sets/restores PrefixLen).
      Pico parity TestPaliGemmaLogitParity: next-token logits < 1e-4 vs HF float64
      (PaliGemmaForConditionalGeneration with token_type_ids) AND a negative control that
      re-runs with PrefixLen=0 (causal-everywhere = LLaVA mask) and asserts the logits
      DIFFER (>1e-3) — proving the bidirectional mask is load-bearing (fixture self-check
      shows max|diff|=6.7). tools/make_pico_paligemma_fixture.py + committed
      tests/fixtures/tiny_paligemma.{safetensors,_config.json,_logits.json}.
      examples/PaliGemmaCaption (.lpr/.lpi + examples/README.md) smoke-captions the pico
      image. OPEN FOLLOW-UPS: (a) multi-image prompts (one prefix block per image);
      (b) a real-checkpoint slicer (tools/make_pico_*_fixture slice precedent) + the
      Gemma SentencePiece tokenizer's <image>-token expansion to NumPatches ids (the demo
      hardcodes the id layout); (c) 448/896 resolutions (more patches, larger pos table);
      (d) the M-RoPE Qwen2-VL path (a THIRD distinct VLM attention regime, still open).
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
      DEPENDS ON the open SD UNet importer (the VAE-decoder follow-up's deferred piece);
      track as its natural successor alongside ControlNet. Pico parity vs a diffusers
      float64 oracle on one motion-module output for a fixed multi-frame latent; an
      examples/TextToVideo that writes a short animated GIF/PPM sequence on CPU once the
      base UNet lands. Note: the cheaper no-UNet route to video is bolting the same
      temporal block onto the landed PixArt DiT — worth scoping if SD UNet stays blocked.
- [ ] Diffusion INPAINTING example (examples/ImageToImage --inpaint, or a sibling) — the
      one-flag follow-up unblocked by the landed SDEdit examples/ImageToImage driver: reuse
      the exact same encode->partial-noise->denoise->decode pipeline but, BEFORE each
      reverse step, overwrite the UNMASKED latent region with the (re-noised to that
      timestep) clean encoded latent, so only the masked region is regenerated while the
      rest stays pixel-faithful (the RePaint / SD-inpaint resample trick). No new model:
      it adds a mask volume + a per-step composite to the existing ImageToImage loop. The
      diffusion-based sibling of the tracked GAN context-encoder examples/Inpainting.
      CPU/ulimit-bounded smoke run on the same tiny fixtures, writing before/masked/after.
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
- [ ] Florence-2 unified vision importer (BuildFlorence2FromSafeTensors, e.g.
      microsoft/Florence-2-base) — a structurally DISTINCT VLM that does detection,
      segmentation, captioning AND OCR through ONE task-prompted seq2seq head, unlike the
      tracked single-task PaliGemma (prefix-LM caption) and LLaVA (causal chat). The
      input is an image + a short TASK TOKEN (e.g. <CAPTION>, <OD>, <OCR>) and the BART-
      style decoder emits a text/coordinate token stream that is parsed per task (boxes
      and polygons are encoded as quantized location tokens <loc_0..loc_999> in the
      vocabulary — the genuinely new idea: spatial outputs as text). Reuses the landed
      seq2seq enc-dec convention (T5EncoderStates two-net path, BART decoder as in TrOCR)
      and a ViT-style image tower; the new code is the DaViT-or-ViT vision encoder feeding
      visual tokens into the encoder prefix + the location-token (de)quantization for box/
      polygon parsing. Scope v1 to <CAPTION> + <OD> (detection) inference. Pico parity vs
      HF float64 on the decoder logits for a fixed image+task token + an examples/
      Florence2 that captions and box-detects one tiny CPU image. First "spatial-output-
      as-text" importer; complements the box/mask importers (DETR/SAM/Mask2Former).
- [ ] Qwen2-VL / Qwen2.5-VL vision-language importer (BuildQwen2VLFromSafeTensors, e.g.
      Qwen/Qwen2-VL-2B-Instruct or Qwen/Qwen2.5-VL-3B-Instruct) — referenced today only
      as "the open Qwen2-VL M-RoPE path" in the PaliGemma task but never tracked as a
      deliverable. The genuinely NEW piece, distinct from every landed/tracked VLM
      (LLaVA causal-everywhere, PaliGemma prefix-LM bidirectional), is **M-RoPE
      (Multimodal Rotary Position Embedding)**: the RoPE position index is split into
      three sections (temporal, height, width) so image/video tokens carry a 3-D grid
      position while text tokens fall back to the scalar 1-D index — a new rotary-index
      assignment threaded through the decoder's RoPE, NOT a new attention math. The
      vision side is a native-dynamic-resolution ViT (window attention over a variable
      patch grid + a small patch-merger MLP that 2x2-pools spatial tokens), reusing the
      landed BuildClipVisionTower/SigLIP ViT path and the LLaVA prompt-assembly splice
      (so it shares that open helper). Scope v1 to a SINGLE still image + text, greedy
      decode on CPU. Genuinely new code: the M-RoPE 3-D position builder (reuse the
      existing rotary tables, only the per-token index changes) + the spatial patch
      merger + the variable-grid window-attention mask. Pico parity vs HF float64 on
      next-token logits for a mixed image+text prompt (reuse make_pico_*_fixture.py) +
      an examples/Qwen2VLDescribe that captions a tiny image (ulimit-bounded). The
      M-RoPE index builder also unblocks Qwen2.5-VL video (the temporal section) later.
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
  - [ ] Complex transposed-conv upsample (use_complex_transposed_convolution=true) —
        the rd64-refined head uses the 3-stage Conv2d(3x3)+ReLU+2× ConvTranspose2d
        decoder instead of the single ConvTranspose2d v1 ships; add the 3x3 conv + the
        two-stage DepthToSpace upsample and a pico parity for that branch.
  - [ ] Image-prompt (visual) conditioning — CLIPSeg can also condition on a PROMPT
        IMAGE (conditional_pixel_values -> clip.get_image_features pooled embedding)
        instead of text; v1 does text only. Add a RunCLIPSegImagePrompt path reusing
        the vision tower's pooled class-token embedding as the conditional vector.
- [ ] TinyNeRF novel-view-synthesis example (examples/TinyNeRF) — a brand-new
      output modality for the tree: a learned implicit 3-D scene that renders an image
      from an arbitrary camera pose, the first differentiable VOLUME RENDERER in the
      repo. Distinct from every landed image generator (DiT/PixArt diffusion,
      VisualGAN/StyleGAN2, MaskGIT token gen) and from the landed implicit-function
      examples (SIREN/Fourier-feature image fitting are 2-D pixel regressions; this is
      3-D ray integration). The genuinely new code is the volume-rendering compositing
      step: cast rays from the camera, sample points along each ray, run a small
      positional-encoded MLP (reuse the landed Fourier/positional-encoding feature map +
      a couple of TNNetFullConnectReLU layers) to predict (RGB, density) per sample, then
      alpha-composite along the ray (C = sum T_i (1 - exp(-sigma_i delta_i)) c_i) with
      the standard transmittance weights — plus its analytic backward so the whole
      render is trainable end-to-end by the existing TNeuralFit MSE loss against ground-
      truth pixels. Scope v1 to a SINGLE tiny synthetic scene (a handful of posed low-res
      views shipped as a fixture, e.g. a procedurally generated colored cube or the
      classic tiny_nerf lego crop downsampled), train a few thousand steps on CPU, then
      render one HELD-OUT pose and write it as a PPM next to the ground truth. Pure CPU,
      ulimit/time-bounded smoke run; no importer, no external download. Establishes the
      ray-marching + alpha-compositing primitive that any future 3-D / view-synthesis
      work (instant-NGP hash grids, 3D Gaussian splatting) would build on.

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

- [ ] BLIP-2 Q-Former vision-language importer (BuildBlip2QFormerFromSafeTensors[Ex]
      + BuildBlip2FromSafeTensors, e.g. Salesforce/blip2-opt-2.7b). The landed VLM
      bridges (LLaVA `BuildLlavaProjector`, the SigLIP/CLIP towers) are simple
      linear/MLP projectors from a frozen vision tower into an LLM; BLIP-2's
      Q-Former is a genuinely DIFFERENT bridging module worth its own builder: a
      small BERT-style transformer fed a fixed set of LEARNED query tokens
      (`query_tokens`, e.g. 32) that, in each block, self-attend among themselves
      AND cross-attend (`crossattention`) into the frozen ViT patch features, then
      project the resulting query embeddings into the LLM token space. New code is
      the query-token soft-prompt (reuse TNNetSoftPrompt) + the interleaved
      self/cross-attention Q-Former block (reuse the landed two-source
      cross-attention wiring) + the OPT/LLM decode tail (Build* already exists).
      Establishes the querying-transformer primitive shared by InstructBLIP and
      many later VLMs. Scope v1 to image-conditioned text generation on a pico
      fixture; keep the ViT + LLM frozen.

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

- [ ] Latent Consistency Model (LCM) few-step sampler in neuraldiffusion.pas
      (TNNetLCMScheduler / LCMSample). The landed diffusion samplers (DDIM,
      DPM-Solver++(2M), Euler/Euler-ancestral, Karras spacing) are all ITERATIVE
      noise-prediction integrators needing ~20-50 steps and CFG-in-the-loop; an LCM
      sampler is a genuinely different inference rule — it evaluates a learned
      CONSISTENCY function f(x_t, t) that maps any noised latent directly toward the
      solution, so generation is 1-4 steps with guidance baked in (no
      cond/uncond double pass per step). New code is the consistency
      parameterization (c_skip/c_out boundary scalings) + the multistep LCM loop
      (predict x0, re-noise to the next, fewer timesteps from the
      already-landed TNNetDiffusionScheduler timestep table). Wire it as an opt-in
      sampler into the LatentTextToImage pipeline so the landed PixArt/DiT + VAE can
      render in a handful of steps. A short note on the matching LCM-distillation
      objective (consistency loss to a teacher) is enough for v1 — sampler first.

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
