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
- [ ] LLaVA-style GENERATIVE vision-language import — image-conditioned text
      generation, the capability step past the landed CLIP dual encoder
      (which only scores image/text similarity and cannot generate).
      Prefer the SigLIP vision tower (BuildSigLIPVisionTower with a
      pVisionFeatures skip-pooling/select-hidden-layer mode — exactly the
      LLaVA select-hidden mode) for Step 4's checkpoint: most modern open VLMs
      (PaliGemma/SmolVLM/Idefics) use SigLIP, not CLIP, as the tower.
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
  Suggested incremental breakdown (each step independently buildable + committable):
  - [ ] Step 3 — projector import + prompt-assembly helper: load the 2-layer MLP
        (gelu) projector, run the vision tower once, and concatenate
        [text-embeds | projected image tokens | text-embeds] as a TNNetInput-fed
        embedding sequence at the <image> placeholder (precedent:
        T5EncoderStatesInput). Test: projected visual tokens match HF float64 < 1e-4.
  - [ ] Step 4 — BuildLlavaFromSafeTensors[Ex] (two nets + projector, multi-shard
        index.json) on a small checkpoint (llava-interleave-qwen-0.5b-hf or a
        SmolVLM-class config); language side is the stock Llama/Qwen2 path. Pico
        parity fixture asserting next-token logits for a mixed image+text prompt
        vs HF float64 < 1e-4.
  - [ ] Step 5 — multimodal chat template ("USER: <image>\\n...ASSISTANT:" /
        ChatML-with-image) in neuralchat.pas + examples/LlavaDescribe demo that
        captions a small image on CPU (ulimit-bounded). Edit examples/README.md
        (NOT the main README) for the new example.
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
- [ ] Segment Anything (SAM) image-encoder + mask-decoder importer
      (facebook/sam-vit-base) — first PROMPTABLE-segmentation importer, a new
      output modality (dense binary masks, not class logits or embeddings).
      Two pieces: (a) the ViT-H/L/B image encoder (windowed attention + a few
      global-attention blocks + the neck 1x1+3x3 conv to a 256-ch image
      embedding) reusing the landed BuildClipVisionTower ViT path; (b) the
      lightweight two-way mask decoder (point/box prompt embeddings -> a couple
      of cross-attention blocks -> upscaled mask). Scope v1 to a single point
      prompt + single mask output to keep the decoder small. Pico parity vs HF
      float64 on the encoder embedding first (decoder is the stretch); demo:
      examples/SegmentAnything segments an object from one click on a tiny image.
- [ ] Depth Anything V2 / DPT monocular-depth importer
      (BuildDepthAnythingFromSafeTensors, e.g. depth-anything/Depth-Anything-V2-Small-hf
      or Intel/dpt-hybrid-midas) — the FIRST DENSE-REGRESSION vision importer and a
      brand-new output modality: one RGB image -> a per-pixel depth map (the landed
      examples/DepthEstimation is a from-scratch TRAINING toy, not a real-checkpoint
      importer). Reuses the landed DINOv2 ViT backbone wholesale (Depth Anything's
      encoder IS DINOv2) and adds the DPT head: (a) the "reassemble" step that takes
      patch tokens from 4 chosen encoder layers and projects+resizes them into a
      4-level feature pyramid (1x1 conv + transpose/identity/strided resampling), and
      (b) the RefineNet-style top-down fusion (residual conv units + nearest upsample)
      ending in a small conv depth head -> single-channel HxW map. The only genuinely
      new code is the multi-layer-token reassemble + the fusion-block wiring; resize
      reuses TNNetDeMaxPool / TNNetUpsample. Pico parity vs HF float64 on the depth
      logits < 1e-4 + an examples/MonoDepth that writes a grayscale depth map of one
      tiny CPU image. Distinct from SegFormer (class map), DETR/SAM (objects/masks)
      and the open Mask R-CNN — this is continuous per-pixel regression.
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
- [ ] DETR object-detection importer follow-up (BuildDetrFromSafeTensors/...Ex,
      facebook/detr-resnet-50, LANDED — TestDetrObjectDetectionParity in
      tests/TestNeuralPretrained.pas, tiny_detr.safetensors fixture; ResNet-50 backbone +
      transformer enc-dec + learned object queries + 2-D sinusoidal pos embedding +
      sigmoid-cxcywh box/class heads, inference-only, no Hungarian matcher):
  - [X] examples/ObjectDetection demo that draws boxes on one CPU image (the importer +
        parity test exist; no example yet).
- [ ] YOLO single-shot object-detection importer (ultralytics YOLOv8n safetensors)
      — a detection family STRUCTURALLY DISTINCT from the landed DETR importer
      (anchor-free fully-convolutional one-stage detector, no transformer): the
      CSP/C2f backbone + PANet feature-pyramid neck + the decoupled detect head
      (per-cell box distribution via DFL + class logits over 3 strides). New code is
      the C2f/SPPF blocks, the FPN top-down+bottom-up fusion (reuses Concat +
      upsample), and DFL box decoding; NMS is a small CPU post-process. Pico parity
      vs an ultralytics float64 oracle on the head outputs + an examples/YoloDetect
      that draws boxes on one CPU image. Reuses the conv-BN-fold loader path.
- [ ] RAFT optical-flow importer (BuildRaftFromSafeTensors, e.g. princeton-vl/raft or
      the torchvision raft_small weights) — the FIRST optical-flow vertical and the
      first model with a TWO-image input producing a dense 2-channel (dx, dy) motion
      field. Structurally distinct from every landed model: a shared feature encoder
      over both frames -> an all-pairs 4-D CORRELATION volume (the genuinely new
      primitive: dot-products between every pair of feature locations across a
      multi-scale pyramid) -> an iterative ConvGRU update operator that refines the
      flow over N steps (a convolutional recurrent cell over an (H,W,C) state —
      reuse / add a minimal conv-gated recurrent cell, the same building block the
      landed Next-frame VideoPrediction ConvLSTM needs). Scope v1 to raft_small +
      a fixed small iteration count, inference-only (no upsampling-mask training
      path needed). Pico parity vs a torchvision float64 oracle on the predicted
      flow + an examples/OpticalFlow that warps one tiny frame toward the next and
      writes a color-coded flow map. Unblocks video-stabilization / frame-interp work.
- [ ] Image inpainting example (examples/Inpainting) — the repo has unconditional
      (VisualGAN) and paired (Pix2Pix) image translation, but
      NO free-form mask completion (fill a hole in an image from its surroundings).
      Train a small context-encoder / partial-convolution U-Net that takes a masked
      image + binary mask and reconstructs the missing region, supervised by
      reconstruction loss (L1 + the landed SSIM loss) plus an optional adversarial
      term reusing the VisualGAN discriminator wiring. New code is the random
      rectangular-mask generator and the masked-region-weighted loss; the network is
      stock conv encoder-decoder + skip connections. CPU-friendly on CIFAR-10 /
      Tiny ImageNet; writes before/after triplets. Edit examples/README.md.
- [X] VideoMAE / TimeSformer spatiotemporal-transformer importer
      (LANDED: BuildVideoMAEFromSafeTensors[Ex/WithConfig] +
      ReadVideoMAEConfigFromJSONFile + RunVideoMAELogits in neuralpretrained.pas;
      tubelet 3-D conv (TNNetConvolution3D, non-overlapping temporal stride via
      SplitChannels+DeepConcat) + fixed sin-cos 3-D position table + stock CLIP
      pre-LN encoder (joint space-time attention) + mean-pool/fc_norm/classifier
      head; use_mean_pooling=True finetuned head only. Pico float64 parity
      TestVideoMAEClassificationParity (max|diff| < 1e-4) + tools/
      make_pico_videomae_fixture.py + examples/VideoAction. Open follow-ups:
      use_mean_pooling=False CLS-token head; DIVIDED space-time attention
      (TimeSformer); the masked-pretraining VideoMAEModel/decoder path.)
  OLD: VideoMAE / TimeSformer spatiotemporal-transformer importer
      (BuildVideoMAEFromSafeTensors, e.g. MCG-NJU/videomae-base-finetuned-kinetics)
      — the FIRST video-classification importer (a clip of T frames -> an action
      label), the natural pay-off of the landed TNNetConvolution3D layer. Tubelet
      embedding (a 3-D conv that splits the clip into TxPxP space-time patches —
      exactly the landed Conv3D primitive) -> the stock transformer encoder stack
      (reuses the ViT/BERT encoder path) with joint or divided space-time attention
      (TimeSformer) -> mean-pool + classifier. The only genuinely new code is the
      tubelet patchifier wiring + the (divided) space-time attention factorization;
      everything else is landed. Pico parity vs
      HF float64 on the logits + an examples/VideoAction that classifies a short
      Moving-MNIST-tubelet or a tiny clip on CPU. First image-sequence-in importer.
- [X] PixArt-alpha text-to-image importer (BuildPixArtFromSafeTensors, e.g.
      PixArt-alpha/PixArt-XL-2-512x512) — the TEXT-conditioned DiT variant that the
      landed class-conditional DiT importer (BuildDiTFromSafeTensors) explicitly
      defers (see the neuralpretrained.pas DiT header note). The ONLY structurally
      new pieces vs the landed DiT are: (a) the conditioning source — drop the
      y_embedder class table and instead feed T5 encoder states (reuse the landed
      BuildT5FromSafeTensors encoder as the prompt tower) into each block; (b) a
      CROSS-attention sublayer per DiTBlock between the (already landed)
      self-attention and the FFN, attending image tokens -> T5 tokens (reuse
      TNNetCrossAttention); (c) PixArt's SHARED adaLN-single (one global
      timestep-modulation table broadcast to all blocks, instead of per-block
      adaLN-Zero). Everything else — patch embed, sin-cos pos embed, the
      DDPM/DDIM/DPM-Solver++ sampler, the VAE decoder — is already landed, so this
      is the cheapest path to a REAL text-to-image checkpoint and directly unblocks
      the tracked LatentTextToImage example WITHOUT needing the SD UNet. Pico parity
      vs a diffusers float64 oracle on one denoise step + reuse make_pico_*_fixture.
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
- [ ] MaskGIT non-autoregressive image-generation example (examples/MaskGIT) — the
      VQModelImport README already names MaskGIT as the intended downstream generator
      for the landed VQ tokenizer, but no generator exists. Train a small
      bidirectional transformer (reuse the landed encoder-block builder) over the
      VQGAN/VqModel codebook indices to predict masked tokens, then GENERATE by
      iterative parallel decoding: start all-[MASK], predict every token, keep the
      most-confident fraction per step on a cosine mask schedule, remask the rest,
      repeat ~8-12 steps. Structurally distinct from every landed generator — the
      repo has autoregressive (TinyGPT), GAN (VisualGAN/StyleGAN2) and diffusion
      paths but NO masked-token parallel image decoding. New code is the random
      span-mask + confidence-based unmasking scheduler; the net and the VQ encode/
      decode are landed. CPU-friendly on CIFAR-10 latents; writes a generated grid.
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
- [ ] Euler-ancestral + Karras-sigma noise-schedule follow-up for
      TNNetDiffusionScheduler (neuraldiffusion.pas) — the landed sampler covers
      DDPM/DDIM/DPM-Solver++(2M) on the linear/cosine beta schedule, but the de-facto
      default of most diffusion UIs (Karras sigma spacing + the stochastic
      Euler-ancestral step) is absent. Add the Karras sigma rescaling (rho=7 spacing
      between sigma_min/sigma_max) as a schedule option and an Euler-ancestral
      sampler step (deterministic Euler drift + per-step ancestral noise injection),
      both reusing the existing TNNetDenoiseCallback interface. Small, self-contained,
      and a real quality/few-step-sampling lift for every diffusion example and the
      tracked PixArt/LatentTextToImage importers. Verify a 20-step Karras+Euler-a run
      on DiffusionMNIST matches the existing DDIM FID within tolerance.
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
  - [ ] LAMBADA last-word accuracy + ARC/PIQA/WinoGrande on the answer-letter
        scoring core (separate benchmarks, same EvaluateMMLU /
        EvaluateMultipleChoice machinery).
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
      that finally CHAINS the imported generative pieces into one pipeline on CPU:
      CLIP text encoder (BuildClipFromSafeTensors) -> a latent denoiser (the from
      DiT importer above, or the SD-UNet noted under the VAE task) -> VAE decoder
      (BuildVaeDecoderFromSafeTensors) -> RGB, driven by the existing
      TNNetDiffusionScheduler DDIM/DPM-Solver++ loop with classifier-free guidance
      (MakeUnconditionalTwin / the CFG follow-up). The individual importers exist but
      nothing demonstrates the full Stable-Diffusion-style txt2img path; an
      ulimit-bounded demo that generates one small image from a hard-coded prompt is
      the capstone proving the CV-generative stack composes. Edit examples/README.md.

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
- [ ] TNNetFlowWarp dense flow-field backward-warp primitive — the repo has
      TNNetAffineGridSample (a single global 2x3 affine theta -> sampling grid) but
      NO per-pixel DENSE flow warp: given a feature map and a two-channel (dx, dy)
      flow field of the same spatial size, produce the bilinearly-sampled
      backward-warped output `out(x,y) = in(x+dx, y+dy)`. This is the genuinely new
      sampling layer that the open RAFT optical-flow task names ("warps one tiny
      frame toward the next") and that the FrameInterpolation example below needs;
      it is the dense-field sibling of the landed DeformableConv bilinear sampler
      and the parametric TNNetAffineGridSample. Two sources (main features +
      flow-field source, wired like TNNetCrossAttention/TNNetAffineGridSample),
      full input numerical-gradient coverage in TestNeuralNumerical.pas for BOTH the
      sampled-features gradient and the flow-field gradient (the bilinear weights are
      differentiable in dx/dy), edge handling = border-clamp. Unblocks frame
      interpolation, optical-flow warping, and any future video-stabilization work.
- [ ] examples/FrameInterpolation — video FRAME INTERPOLATION (predict the MIDDLE
      frame between two given frames), structurally distinct from the landed
      examples/VideoPrediction (next-frame EXTRAPOLATION via TNNetConvLSTMCell) and
      from every image-generation example: the input is two endpoint frames and the
      target is the unseen in-between frame (the RIFE/FILM task). Scope v1 to the
      same self-contained synthetic Moving-MNIST-style blob data already used by
      VideoPrediction (no download): feed frames t and t+2, supervise on t+1 with an
      L1 + landed-SSIM reconstruction loss. Two CPU-friendly model variants worth
      comparing: (a) a direct conv encoder-decoder that synthesizes the middle frame,
      and (b) a flow-based path that predicts a mid->endpoints flow field and warps
      both frames via the new TNNetFlowWarp primitive, then blends — the textbook
      illustration of why warping beats direct synthesis for motion. Writes
      before/predicted/after triplets; edit examples/README.md (NOT the main README).
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
- [ ] PaliGemma vision-language importer (BuildPaliGemmaFromSafeTensors, e.g.
      google/paligemma-3b-mix-224) — a VLM follow-up that exercises a genuinely
      DIFFERENT attention regime from the tracked causal LLaVA path: PaliGemma is a
      PREFIX-LM, the image tokens AND the prompt tokens see each other with FULL
      BIDIRECTIONAL attention (a block-bidirectional mask over the prefix) and only the
      generated suffix is causal. The new code is that prefix-LM attention-mask wiring
      threaded through the decoder; nearly everything else is landed — the SigLIP tower
      (BuildSigLIPVisionTower), the Gemma decoder (BuildGemmaFromSafeTensors), and the
      linear multimodal projector + image-token splice are exactly the LLaVA
      prompt-assembly helper (so this depends on / shares that helper). Distinct from
      LLaVA's causal-everywhere mask and from the open Qwen2-VL M-RoPE path. Pico parity
      vs HF float64 on next-token logits for a mixed image+text prompt + an
      examples/PaliGemmaCaption that captions a tiny image on CPU (ulimit-bounded).

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
