#!/usr/bin/env python3
"""Generate the pico Stable Diffusion 1.x pipeline fixture for
TStableDiffusionPipeline and the CLIP vocab.json + merges.txt tokenizer path.

Every component is a tiny RANDOM-weight instance of the real diffusers /
transformers class (nothing downloaded), saved with the real folder layout:

  tests/fixtures/tiny_sd15/
    model_index.json                     StableDiffusionPipeline
    scheduler/scheduler_config.json      DDIMScheduler, SD 1.5 betas, leading
                                         spacing, steps_offset 1
    text_encoder/{config.json,model.safetensors}       CLIPTextModel, F32, keys
                                         under text_model. as in SD 1.5
    tokenizer/{vocab.json,merges.txt,tokenizer_config.json,
               special_tokens_map.json}  a CLIP BPE trained here on a small
                                         corpus (no tokenizer.json, as SD 1.5)
    unet/{config.json,diffusion_pytorch_model.safetensors}  F32
    vae/{config.json,diffusion_pytorch_model.safetensors}   F32, 2 blocks
  tests/fixtures/tiny_sd15_pipeline_io.json
    tokens          per prompt: raw ids and the 77-id padded pipeline input
    prompt / prompt_embeds / negative_prompt_embeds   encode_prompt output
    latents         the fixed initial latents (C,H,W)
    timesteps       diffusers DDIM timesteps for STEPS steps
    noise_pred_0    the guided noise prediction of step 0
    latents_after   latents after each step
    image           the decoded image (H,W,3) in [0,1]
  tests/fixtures/tiny_sd15_samplers_io.json
    per sampler (PNDM skip_prk_steps, EulerDiscrete, DPMSolverMultistep, each
    from_config of the DDIM config): its scheduler config, timesteps and the
    latents after each model eval, SAMPLER_STEPS steps, same latents/guidance.
    Euler latents are divided by sqrt(sigma_next^2 + 1): the model-input (VP)
    scale TStableDiffusionPipeline keeps. Values rounded to 10 digits.
  tests/fixtures/tiny_sd15_karras_io.json
    continuous-sigma sampling, the latents/guidance of the sampler file at
    KARRAS_STEPS steps, one case per KARRAS_CASES entry (from_config of the DDIM
    config with use_karras_sigmas and/or timestep_spacing "linspace"): the
    scheduler config, the float model timesteps, the sigmas and the latents
    after each step (Euler ones divided by sqrt(sigma_next^2 + 1)). The Euler a
    case injects step_noise (randn_tensor patched).
  tests/fixtures/tiny_sd15_nonsquare_io.json
    the DDIM run of tiny_sd15_pipeline_io.json at NONSQUARE_WIDTH x
    NONSQUARE_HEIGHT: latents (C,H,W), unet_cond_0 (the conditional UNet
    output of step 0), noise_pred_0, latents_after and image (H,W,3).
  tests/fixtures/tiny_sd15_oddsize_io.json
    the same at ODD_WIDTH x ODD_HEIGHT, a 7x5 latent grid that is not a
    multiple of the UNet's 2: diffusers resizes each upsample to its skip
    (upsample_size). check_inputs is skipped (it wants multiples of 8, the
    real VAE factor, not the pico 2).
  tests/fixtures/tiny_sd15_freeu_filter_io.json
    diffusers fourier_filter(threshold=1) (FreeU's skip filter) on random
    (C,H,W) inputs: odd, even and 1-wide sides, scale FREEU_FILTER_SCALE.
  tests/fixtures/tiny_sd15_freeu_nonsquare_io.json,
  tests/fixtures/tiny_sd15_freeu_oddsize_io.json
    the non-square / odd-size runs with pipe.enable_freeu(*FREEU_SD15) (the
    SD 1.5 preset): 16x8 latents (even sides at both FreeU up blocks) and 7x5
    latents (4x3 and 7x5 at the two FreeU up blocks).
  tests/fixtures/tiny_sd15_lcm_io.json
    LCMScheduler.from_config of the DDIM config (the LCM-LoRA recipe),
    LCM_STEPS steps on the tiny_sd15 UNet with the usual two-half CFG: text to
    image at guidance LCM_GUIDANCE and at 1.0 (one pass), and img2img at
    strength LCM_STRENGTH. The img2img case follows LatentConsistencyModel-
    Img2ImgPipeline's schedule: set_timesteps(strength=...) shrinks the origin
    grid and every step runs (StableDiffusionImg2ImgPipeline's get_timesteps
    cut is bypassed). Per-step noise (step_noise) and the image's VAE sample
    noise are injected; latents_after, timesteps and the decoded image per case.
  tests/fixtures/tiny_sd15_pag_io.json
    StableDiffusionPAGPipeline (Perturbed-Attention Guidance) on tiny_sd15,
    DDIM, STEPS steps from the fixed latents of tiny_sd15_pipeline_io.json, one
    case per PAG_CASES entry (guidance scale, pag_scale, pag_adaptive_scale,
    pag_applied_layers): timesteps, latents_after and the decoded image.
  tests/fixtures/tiny_sd15_lcm_distilled/{model_index.json,
                                          scheduler/scheduler_config.json, unet/}
    an LCM-distilled pico UNet: tiny_sd15's UNet weights plus a random
    time_embedding.cond_proj (time_cond_proj_dim LCM_COND_PROJ_DIM), an
    LCMScheduler, model_index LatentConsistencyModelPipeline. text_encoder/,
    tokenizer/ and vae/ are tiny_sd15's and not copied (tests assemble the
    folder).
  tests/fixtures/tiny_sd15_lcm_distilled_io.json
    LatentConsistencyModelPipeline (one UNet pass per step, guidance embedded)
    for LCM_STEPS steps at LCM_DISTILLED_GUIDANCE and 1.0, and
    LatentConsistencyModelImg2ImgPipeline at strength LCM_STRENGTH, with the
    same noise injection as the LCM file; unet_check is one UNet call
    (timestep_cond = the guidance-scale embedding) on the initial latents.

The pipeline is reloaded from the saved folder and run in float64, latents
passed explicitly so torch's RNG does not matter.

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv):
  python tools/make_pico_sd_pipeline_fixture.py
  python tools/make_pico_sd_pipeline_fixture.py --samplers-only
  python tools/make_pico_sd_pipeline_fixture.py --nonsquare-only
  python tools/make_pico_sd_pipeline_fixture.py --oddsize-only
  python tools/make_pico_sd_pipeline_fixture.py --lcm-only
  python tools/make_pico_sd_pipeline_fixture.py --lcm-distilled-only
  python tools/make_pico_sd_pipeline_fixture.py --freeu-only
  python tools/make_pico_sd_pipeline_fixture.py --pag-only
  python tools/make_pico_sd_pipeline_fixture.py --karras-only
(--samplers-only / --nonsquare-only / --oddsize-only / --lcm-only /
--freeu-only / --pag-only / --karras-only keep the committed folder and
rewrite only the sampler / non-square / odd-size / LCM / FreeU / PAG / Karras
files; --lcm-distilled-only
rewrites the distilled UNet folder and its file.)
"""
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter

import torch
from safetensors.torch import load_file, save_file
from diffusers.utils.torch_utils import fourier_filter
from diffusers import (AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler,
                       EulerDiscreteScheduler, LatentConsistencyModelImg2ImgPipeline,
                       LatentConsistencyModelPipeline, LCMScheduler, PNDMScheduler,
                       StableDiffusionImg2ImgPipeline, StableDiffusionPAGPipeline,
                       StableDiffusionPipeline, UNet2DConditionModel)
import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img as img2img_module
import diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_img2img as lcm_img2img_module
import diffusers.schedulers.scheduling_lcm as lcm_module
import diffusers.schedulers.scheduling_euler_ancestral_discrete as euler_a_module
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")
PICO_DIR = os.path.join(FIXTURES, "tiny_sd15")
IO_FILE = os.path.join(FIXTURES, "tiny_sd15_pipeline_io.json")
SAMPLERS_FILE = os.path.join(FIXTURES, "tiny_sd15_samplers_io.json")
NONSQUARE_FILE = os.path.join(FIXTURES, "tiny_sd15_nonsquare_io.json")
ODDSIZE_FILE = os.path.join(FIXTURES, "tiny_sd15_oddsize_io.json")
LCM_FILE = os.path.join(FIXTURES, "tiny_sd15_lcm_io.json")
LCM_DISTILLED_DIR = os.path.join(FIXTURES, "tiny_sd15_lcm_distilled")
LCM_DISTILLED_FILE = os.path.join(FIXTURES, "tiny_sd15_lcm_distilled_io.json")
FREEU_FILTER_FILE = os.path.join(FIXTURES, "tiny_sd15_freeu_filter_io.json")
FREEU_NONSQUARE_FILE = os.path.join(FIXTURES, "tiny_sd15_freeu_nonsquare_io.json")
FREEU_ODDSIZE_FILE = os.path.join(FIXTURES, "tiny_sd15_freeu_oddsize_io.json")
PAG_FILE = os.path.join(FIXTURES, "tiny_sd15_pag_io.json")
KARRAS_FILE = os.path.join(FIXTURES, "tiny_sd15_karras_io.json")

SEED = 20261021
MERGE_COUNT = 120
TEXT_HIDDEN = 32
IMAGE_SIZE = 16          # VAE factor 2 -> an 8x8 latent grid
NONSQUARE_WIDTH = 32     # -> a 16x8 latent grid, 8x4 at the UNet mid block
NONSQUARE_HEIGHT = 16
ODD_WIDTH = 14           # -> a 7x5 latent grid, 4x3 at the UNet mid block
ODD_HEIGHT = 10
STEPS = 3
SAMPLER_STEPS = 4        # PNDM reaches its 4th-order formula, DPM++ 2nd order
KARRAS_STEPS = 5         # linspace(0, 999, 5) is fractional (249.75 apart)
GUIDANCE = 7.5
LCM_STEPS = 4
LCM_GUIDANCE = 1.5
LCM_STRENGTH = 0.5
LCM_COND_PROJ_DIM = 16
LCM_DISTILLED_GUIDANCE = 8.5    # LatentConsistencyModelPipeline's default
LCM_UNET_CHECK_TIMESTEP = 499
FREEU_SD15 = (0.9, 0.2, 1.5, 1.6)   # s1, s2, b1, b2 of the FreeU paper for SD 1.5
FREEU_FILTER_SCALE = 0.37
FREEU_FILTER_SIZES = [(1, 1), (1, 4), (3, 1), (2, 2), (2, 5), (4, 4), (5, 3),
                      (6, 7), (8, 8)]   # (H, W)
FREEU_FILTER_CHANNELS = 3
# name, guidance_scale, pag_scale, pag_adaptive_scale, pag_applied_layers. The
# adaptive scale reaches 0 within the 3 DDIM steps (timesteps 667, 334, 1).
PAG_CASES = [
    ("cfg_mid", GUIDANCE, 3.0, 0.0, ["mid"]),
    ("no_cfg_down_up", 1.0, 2.0, 0.0, ["down", r"up_blocks\.1\.attentions\.1"]),
    ("cfg_adaptive_all", GUIDANCE, 3.0, 0.004, ["(down|mid|up)"]),
]
# name, scheduler class, from_config overrides (on the leading DDIM config).
KARRAS_CASES = [
    ("euler_karras_leading", EulerDiscreteScheduler, {"use_karras_sigmas": True}),
    ("euler_karras_linspace", EulerDiscreteScheduler,
     {"use_karras_sigmas": True, "timestep_spacing": "linspace"}),
    ("euler_linspace", EulerDiscreteScheduler, {"timestep_spacing": "linspace"}),
    ("euler_a_linspace", EulerAncestralDiscreteScheduler,
     {"timestep_spacing": "linspace"}),
    ("dpm_karras", DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
]
PROMPT = "a photo of an astronaut riding a horse on mars"
CORPUS = [
    "a photo of an astronaut riding a horse on mars",
    "a painting of a red fox in the snow, oil on canvas",
    "the cat's hat is on the mat",
    "hello world! it's a beautiful day for a photo",
    "high quality, detailed, sharp focus, trending on artstation",
    "an astronaut and a horse on the moon",
]
TOKEN_PROMPTS = [
    PROMPT,
    "  Hello,   WORLD!!  it's 2024 \t naïve café ",
    "",
    "<|startoftext|>A Photo<|endoftext|> of THE moon",
    "they'll say we've done it, I'm sure; you're right... isn't it?",
    " ".join(["astronaut riding horses"] * 30),
]


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def train_clip_bpe():
    """CLIP-layout vocab: bytes, bytes + '</w>', merges, then the two specials."""
    import regex
    byte_encoder = bytes_to_unicode()
    pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
                            regex.IGNORECASE)
    words = Counter()
    for line in CORPUS:
        for token in pattern.findall(line.lower()):
            symbols = [byte_encoder[b] for b in token.encode("utf-8")]
            symbols[-1] += "</w>"
            words[tuple(symbols)] += 1
    merges = []
    for _ in range(MERGE_COUNT):
        pairs = Counter()
        for word, freq in words.items():
            for left, right in zip(word, word[1:]):
                pairs[(left, right)] += freq
        if not pairs:
            break
        best = max(pairs.items(), key=lambda item: (item[1], item[0]))[0]
        merges.append(best)
        merged = Counter()
        for word, freq in words.items():
            out, pos = [], 0
            while pos < len(word):
                if pos + 1 < len(word) and (word[pos], word[pos + 1]) == best:
                    out.append(word[pos] + word[pos + 1])
                    pos += 2
                else:
                    out.append(word[pos])
                    pos += 1
            merged[tuple(out)] += freq
        words = merged
    vocab = list(byte_encoder.values())
    vocab = vocab + [v + "</w>" for v in vocab]
    vocab += ["".join(pair) for pair in merges]
    vocab += ["<|startoftext|>", "<|endoftext|>"]
    return {token: idx for idx, token in enumerate(vocab)}, merges


def write_tokenizer(folder, vocab, merges):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(folder, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for left, right in merges:
            f.write(left + " " + right + "\n")

    def added(content):
        return {"__type": "AddedToken", "content": content, "lstrip": False,
                "normalized": True, "rstrip": False, "single_word": False}
    # runwayml/stable-diffusion-v1-5 tokenizer/tokenizer_config.json layout.
    config = {"add_prefix_space": False, "bos_token": added("<|startoftext|>"),
              "do_lower_case": True, "eos_token": added("<|endoftext|>"),
              "errors": "replace", "model_max_length": 77,
              "pad_token": "<|endoftext|>", "tokenizer_class": "CLIPTokenizer",
              "unk_token": added("<|endoftext|>")}
    with open(os.path.join(folder, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    special = {"bos_token": "<|startoftext|>", "eos_token": "<|endoftext|>",
               "pad_token": "<|endoftext|>", "unk_token": "<|endoftext|>"}
    with open(os.path.join(folder, "special_tokens_map.json"), "w") as f:
        json.dump(special, f, indent=2)


def perturb(module, std=0.05):
    """Moves every parameter off its init so zero-init layers are exercised."""
    with torch.no_grad():
        for param in module.parameters():
            param.add_(torch.randn_like(param) * std)


def chw(tensor):
    return tensor.to(torch.float64).tolist()


def build_folder():
    torch.manual_seed(SEED)
    vocab, merges = train_clip_bpe()
    text_config = CLIPTextConfig(
        vocab_size=len(vocab), hidden_size=TEXT_HIDDEN, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=77,
        hidden_act="quick_gelu", layer_norm_eps=1e-5, projection_dim=TEXT_HIDDEN,
        bos_token_id=0, eos_token_id=2, pad_token_id=1)
    text_encoder = CLIPTextModel(text_config)
    perturb(text_encoder)
    unet = UNet2DConditionModel(
        sample_size=IMAGE_SIZE // 2, in_channels=4, out_channels=4,
        block_out_channels=(8, 16),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=1, cross_attention_dim=TEXT_HIDDEN, attention_head_dim=2,
        norm_num_groups=4)
    perturb(unet)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16), layers_per_block=1, latent_channels=4,
        norm_num_groups=4, sample_size=IMAGE_SIZE, scaling_factor=0.18215)
    perturb(vae)
    # SD 1.5 betas with the DDIM sampler our scheduler implements.
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False,
        steps_offset=1, prediction_type="epsilon", timestep_spacing="leading")
    tokenizer_dir = os.path.join(PICO_DIR, "tokenizer")
    if os.path.isdir(PICO_DIR):
        shutil.rmtree(PICO_DIR)
    write_tokenizer(tokenizer_dir, vocab, merges)
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False)
    pipe.save_pretrained(PICO_DIR, safe_serialization=True)
    # SD 1.5 ships the slow-tokenizer files only; keep exactly those.
    for name in os.listdir(tokenizer_dir):
        if name not in ("vocab.json", "merges.txt"):
            os.remove(os.path.join(tokenizer_dir, name))
    write_tokenizer(tokenizer_dir, vocab, merges)
    # transformers 5 saves CLIPTextModel keys without the text_model. prefix
    # that the SD 1.5 checkpoint (and our loader) use.
    te_file = os.path.join(PICO_DIR, "text_encoder", "model.safetensors")
    tensors = load_file(te_file)
    tensors = {(k if k.startswith("text_model.") else "text_model." + k): v
               for k, v in tensors.items()}
    save_file(tensors, te_file, metadata={"format": "pt"})


def make_oracle():
    pipe = StableDiffusionPipeline.from_pretrained(PICO_DIR, torch_dtype=torch.float64,
                                                   safety_checker=None)
    pipe.set_progress_bar_config(disable=True)
    tokenizer = pipe.tokenizer
    assert tokenizer.model_max_length == 77
    saved = load_file(os.path.join(PICO_DIR, "text_encoder", "model.safetensors"))
    for key, value in pipe.text_encoder.state_dict().items():
        stored = saved.get(key, saved.get("text_model." + key))
        assert stored is not None and torch.equal(stored.double(), value), key
    tokens = []
    for prompt in TOKEN_PROMPTS:
        padded = tokenizer(prompt, padding="max_length", max_length=77,
                           truncation=True).input_ids
        raw = tokenizer(prompt, add_special_tokens=False).input_ids
        tokens.append({"prompt": prompt, "raw_ids": raw, "input_ids": padded})
    prompt_embeds, negative_embeds = pipe.encode_prompt(PROMPT, "cpu", 1, True, None)

    generator = torch.Generator().manual_seed(SEED)
    grid = IMAGE_SIZE // pipe.vae_scale_factor
    latents = torch.randn((1, 4, grid, grid), generator=generator, dtype=torch.float64)

    captured = []

    def unet_hook(_module, _args, output):
        captured.append(output[0].detach().clone())
    hook = pipe.unet.register_forward_hook(unet_hook)
    after = []

    def on_step_end(_pipe, _step, _t, kwargs):
        after.append(chw(kwargs["latents"][0]))
        return kwargs
    result = pipe(PROMPT, height=IMAGE_SIZE, width=IMAGE_SIZE,
                  num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                  latents=latents.clone(), output_type="np",
                  callback_on_step_end=on_step_end)
    hook.remove()
    uncond, cond = captured[0].chunk(2)
    noise_pred_0 = uncond + GUIDANCE * (cond - uncond)
    payload = {
        "image_size": IMAGE_SIZE, "steps": STEPS, "guidance_scale": GUIDANCE,
        "prompt": PROMPT, "tokens": tokens,
        "prompt_embeds": prompt_embeds[0].tolist(),
        "negative_prompt_embeds": negative_embeds[0].tolist(),
        "latents": chw(latents[0]),
        "timesteps": [int(t) for t in pipe.scheduler.timesteps],
        "noise_pred_0": chw(noise_pred_0[0]),
        "latents_after": after,
        "image": result.images[0].tolist(),
    }
    with open(IO_FILE, "w") as f:
        json.dump(payload, f)
    print("timesteps", payload["timesteps"], "vocab", len(tokenizer))


def rounded(value):
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return float("%.10g" % value)


def make_sampler_oracle():
    pipe = StableDiffusionPipeline.from_pretrained(PICO_DIR, torch_dtype=torch.float64,
                                                   safety_checker=None)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator().manual_seed(SEED)
    grid = IMAGE_SIZE // pipe.vae_scale_factor
    latents = torch.randn((1, 4, grid, grid), generator=generator, dtype=torch.float64)
    base_config = pipe.scheduler.config
    samplers = [PNDMScheduler.from_config(base_config, skip_prk_steps=True),
                EulerDiscreteScheduler.from_config(base_config),
                DPMSolverMultistepScheduler.from_config(base_config)]
    payload = {"steps": SAMPLER_STEPS, "guidance_scale": GUIDANCE, "samplers": []}
    for scheduler in samplers:
        pipe.scheduler = scheduler
        is_euler = isinstance(scheduler, EulerDiscreteScheduler)
        after = []

        def on_step_end(_pipe, _step, _t, kwargs):
            sample = kwargs["latents"][0]
            if is_euler:
                sigma = float(scheduler.sigmas[scheduler.step_index])
                sample = sample / (sigma ** 2 + 1) ** 0.5
            after.append(rounded(chw(sample)))
            return kwargs
        pipe(PROMPT, height=IMAGE_SIZE, width=IMAGE_SIZE,
             num_inference_steps=SAMPLER_STEPS, guidance_scale=GUIDANCE,
             latents=latents.clone(), output_type="latent",
             callback_on_step_end=on_step_end)
        # JSON has no infinity (DPM lambda_min_clipped defaults to -inf).
        config = {key: value for key, value in dict(scheduler.config).items()
                  if not key.startswith("_") and not (
                      isinstance(value, float) and not math.isfinite(value))}
        config["_class_name"] = type(scheduler).__name__
        payload["samplers"].append({
            "class_name": type(scheduler).__name__, "config": config,
            "timesteps": [int(t) for t in scheduler.timesteps],
            "latents_after": after})
        print(type(scheduler).__name__, payload["samplers"][-1]["timesteps"])
    with open(SAMPLERS_FILE, "w") as f:
        json.dump(payload, f)


def make_karras_oracle():
    pipe = StableDiffusionPipeline.from_pretrained(PICO_DIR, torch_dtype=torch.float64,
                                                   safety_checker=None)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator().manual_seed(SEED)
    grid = IMAGE_SIZE // pipe.vae_scale_factor
    latents = torch.randn((1, 4, grid, grid), generator=generator, dtype=torch.float64)
    step_noise = [torch.randn(latents.shape, generator=generator, dtype=torch.float64)
                  for _ in range(KARRAS_STEPS)]
    pending_step_noise = []
    euler_a_module.randn_tensor = lambda *args, **kwargs: pending_step_noise.pop(0).clone()
    base_config = pipe.scheduler.config
    payload = {"steps": KARRAS_STEPS, "guidance_scale": GUIDANCE,
               "step_noise": [rounded(chw(item[0])) for item in step_noise],
               "cases": []}
    for name, scheduler_class, overrides in KARRAS_CASES:
        scheduler = scheduler_class.from_config(base_config, **overrides)
        pipe.scheduler = scheduler
        is_euler = scheduler_class is not DPMSolverMultistepScheduler
        pending_step_noise[:] = step_noise
        after = []

        def on_step_end(_pipe, _step, _t, kwargs):
            sample = kwargs["latents"][0]
            if is_euler:
                sigma = float(scheduler.sigmas[scheduler.step_index])
                sample = sample / (sigma ** 2 + 1) ** 0.5
            after.append(rounded(chw(sample)))
            return kwargs
        pipe(PROMPT, height=IMAGE_SIZE, width=IMAGE_SIZE,
             num_inference_steps=KARRAS_STEPS, guidance_scale=GUIDANCE,
             latents=latents.clone(), output_type="latent",
             callback_on_step_end=on_step_end)
        config = {key: value for key, value in dict(scheduler.config).items()
                  if not key.startswith("_") and not (
                      isinstance(value, float) and not math.isfinite(value))}
        config["_class_name"] = scheduler_class.__name__
        payload["cases"].append({
            "name": name, "config": config,
            "timesteps": [rounded(float(t)) for t in scheduler.timesteps],
            "sigmas": [rounded(float(s)) for s in scheduler.sigmas],
            "latents_after": after})
        print(name, payload["cases"][-1]["timesteps"], payload["cases"][-1]["sigmas"])
    with open(KARRAS_FILE, "w") as f:
        json.dump(payload, f)


def make_nonsquare_oracle(width=NONSQUARE_WIDTH, height=NONSQUARE_HEIGHT,
                          out_file=NONSQUARE_FILE, freeu=None):
    pipe = StableDiffusionPipeline.from_pretrained(PICO_DIR, torch_dtype=torch.float64,
                                                   safety_checker=None)
    pipe.set_progress_bar_config(disable=True)
    if freeu is not None:
        pipe.enable_freeu(*freeu)
    if width % 8 or height % 8:
        pipe.check_inputs = lambda *args, **kwargs: None
    generator = torch.Generator().manual_seed(SEED)
    factor = pipe.vae_scale_factor
    latents = torch.randn((1, 4, height // factor, width // factor),
                          generator=generator, dtype=torch.float64)
    captured = []

    def unet_hook(_module, _args, output):
        captured.append(output[0].detach().clone())
    hook = pipe.unet.register_forward_hook(unet_hook)
    after = []

    def on_step_end(_pipe, _step, _t, kwargs):
        after.append(rounded(chw(kwargs["latents"][0])))
        return kwargs
    result = pipe(PROMPT, height=height, width=width,
                  num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                  latents=latents.clone(), output_type="np",
                  callback_on_step_end=on_step_end)
    hook.remove()
    uncond, cond = captured[0].chunk(2)
    payload = {
        "width": width, "height": height, "steps": STEPS,
        "guidance_scale": GUIDANCE, "prompt": PROMPT,
        "latents": rounded(chw(latents[0])),
        "timesteps": [int(t) for t in pipe.scheduler.timesteps],
        "unet_cond_0": rounded(chw(cond[0])),
        "noise_pred_0": rounded(chw((uncond + GUIDANCE * (cond - uncond))[0])),
        "latents_after": after,
        "image": rounded(result.images[0].tolist()),
    }
    if freeu is not None:
        payload["freeu"] = list(freeu)
    with open(out_file, "w") as f:
        json.dump(payload, f)
    print("image", result.images[0].shape, "->", os.path.basename(out_file))


def make_oddsize_oracle():
    make_nonsquare_oracle(ODD_WIDTH, ODD_HEIGHT, ODDSIZE_FILE)


def make_pag_oracle():
    base = StableDiffusionPipeline.from_pretrained(PICO_DIR, torch_dtype=torch.float64,
                                                   safety_checker=None)
    generator = torch.Generator().manual_seed(SEED)
    grid = IMAGE_SIZE // base.vae_scale_factor
    latents = torch.randn((1, 4, grid, grid), generator=generator, dtype=torch.float64)
    cases = []
    for name, guidance, pag_scale, adaptive, layers in PAG_CASES:
        pipe = StableDiffusionPAGPipeline(**base.components, pag_applied_layers=layers)
        pipe.set_progress_bar_config(disable=True)
        after = []

        def on_step_end(_pipe, _step, _t, kwargs):
            after.append(rounded(chw(kwargs["latents"][0])))
            return kwargs
        result = pipe(PROMPT, height=IMAGE_SIZE, width=IMAGE_SIZE,
                      num_inference_steps=STEPS, guidance_scale=guidance,
                      pag_scale=pag_scale, pag_adaptive_scale=adaptive,
                      latents=latents.clone(), output_type="np",
                      callback_on_step_end=on_step_end)
        cases.append({"name": name, "guidance_scale": guidance,
                      "pag_scale": pag_scale, "pag_adaptive_scale": adaptive,
                      "pag_applied_layers": layers,
                      "timesteps": [int(t) for t in pipe.scheduler.timesteps],
                      "latents_after": after,
                      "image": rounded(result.images[0].tolist())})
        print("PAG", name, cases[-1]["timesteps"])
    payload = {"width": IMAGE_SIZE, "height": IMAGE_SIZE, "steps": STEPS,
               "prompt": PROMPT, "latents": rounded(chw(latents[0])),
               "cases": cases}
    with open(PAG_FILE, "w") as f:
        json.dump(payload, f)


def make_freeu_oracles():
    generator = torch.Generator().manual_seed(SEED + 50)
    cases = []
    for height, width in FREEU_FILTER_SIZES:
        sample = torch.randn((1, FREEU_FILTER_CHANNELS, height, width),
                             generator=generator, dtype=torch.float64)
        filtered = fourier_filter(sample, threshold=1, scale=FREEU_FILTER_SCALE)
        cases.append({"input": rounded(chw(sample[0])),
                      "output": rounded(chw(filtered[0]))})
    with open(FREEU_FILTER_FILE, "w") as f:
        json.dump({"scale": FREEU_FILTER_SCALE, "cases": cases}, f)
    make_nonsquare_oracle(NONSQUARE_WIDTH, NONSQUARE_HEIGHT, FREEU_NONSQUARE_FILE,
                          FREEU_SD15)
    make_nonsquare_oracle(ODD_WIDTH, ODD_HEIGHT, FREEU_ODDSIZE_FILE, FREEU_SD15)


def make_lcm_oracle():
    generator = torch.Generator().manual_seed(SEED + 40)
    grid = IMAGE_SIZE // 2
    noise = torch.randn((1, 4, grid, grid), generator=generator, dtype=torch.float64)
    image_noise = torch.randn(noise.shape, generator=generator, dtype=torch.float64)
    step_noise = [torch.randn(noise.shape, generator=generator, dtype=torch.float64)
                  for _ in range(LCM_STEPS)]
    init_image = torch.randint(0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE),
                               generator=generator).to(torch.float64) / 255.0
    pending_step_noise = []
    lcm_module.randn_tensor = lambda *args, **kwargs: pending_step_noise.pop(0).clone()
    img2img_module.randn_tensor = lambda *args, **kwargs: noise.clone()
    img2img_module.retrieve_latents = (
        lambda output, generator=None, sample_mode="sample":
        output.latent_dist.mean + output.latent_dist.std * image_noise)
    cases = []
    for name, guidance, strength in (("text2img", LCM_GUIDANCE, None),
                                     ("text2img_no_cfg", 1.0, None),
                                     ("img2img_strength", LCM_GUIDANCE, LCM_STRENGTH)):
        pipe_class = StableDiffusionPipeline if strength is None else StableDiffusionImg2ImgPipeline
        pipe = pipe_class.from_pretrained(PICO_DIR, torch_dtype=torch.float64,
                                          safety_checker=None)
        pipe.set_progress_bar_config(disable=True)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        extra = {"height": IMAGE_SIZE, "width": IMAGE_SIZE, "latents": noise.clone()}
        if strength is not None:
            base_set_timesteps = pipe.scheduler.set_timesteps
            pipe.scheduler.set_timesteps = (
                lambda *args, **kwargs: base_set_timesteps(*args, strength=strength, **kwargs))
            pipe.get_timesteps = lambda steps, _strength, _device: (
                pipe.scheduler.timesteps, len(pipe.scheduler.timesteps))
            extra = {"image": init_image, "strength": strength}
        pending_step_noise[:] = step_noise
        after = []

        def on_step_end(_pipe, _step, _t, kwargs):
            after.append(rounded(chw(kwargs["latents"][0])))
            return kwargs
        result = pipe(PROMPT, num_inference_steps=LCM_STEPS, guidance_scale=guidance,
                      output_type="np", callback_on_step_end=on_step_end, **extra)
        assert len(pending_step_noise) == 1, name
        config = dict(pipe.scheduler.config)
        config["_class_name"] = type(pipe.scheduler).__name__
        cases.append({"name": name, "guidance_scale": guidance,
                      "strength": 1.0 if strength is None else strength,
                      "scheduler_config": {key: value for key, value in config.items()
                                           if not key.startswith("_") or key == "_class_name"},
                      "timesteps": [int(t) for t in pipe.scheduler.timesteps],
                      "latents_after": after,
                      "image": rounded(result.images[0].tolist())})
        print("LCM", name, cases[-1]["timesteps"])
    payload = {"width": IMAGE_SIZE, "height": IMAGE_SIZE, "steps": LCM_STEPS,
               "prompt": PROMPT,
               "init_image": rounded(init_image[0].permute(1, 2, 0).tolist()),
               "latents": rounded(chw(noise[0])),
               "image_sample_noise": rounded(chw(image_noise[0])),
               "step_noise": [rounded(chw(item[0])) for item in step_noise],
               "cases": cases}
    with open(LCM_FILE, "w") as f:
        json.dump(payload, f)


def build_lcm_distilled_folder(folder):
    """Saves the full LatentConsistencyModelPipeline to folder; copies the
    scheduler, unet and model_index into LCM_DISTILLED_DIR."""
    torch.manual_seed(SEED + 50)
    base = StableDiffusionPipeline.from_pretrained(PICO_DIR, safety_checker=None)
    config = dict(base.unet.config)
    config["time_cond_proj_dim"] = LCM_COND_PROJ_DIM
    unet = UNet2DConditionModel.from_config(config)
    missing, unexpected = unet.load_state_dict(base.unet.state_dict(), strict=False)
    assert missing == ["time_embedding.cond_proj.weight"] and not unexpected, missing
    pipe = LatentConsistencyModelPipeline(
        vae=base.vae, text_encoder=base.text_encoder, tokenizer=base.tokenizer,
        unet=unet, scheduler=LCMScheduler.from_config(base.scheduler.config),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.save_pretrained(folder, safe_serialization=True)
    if os.path.isdir(LCM_DISTILLED_DIR):
        shutil.rmtree(LCM_DISTILLED_DIR)
    os.makedirs(LCM_DISTILLED_DIR)
    shutil.copy(os.path.join(folder, "model_index.json"), LCM_DISTILLED_DIR)
    for component in ("scheduler", "unet"):
        shutil.copytree(os.path.join(folder, component),
                        os.path.join(LCM_DISTILLED_DIR, component))


def make_lcm_distilled_oracle():
    generator = torch.Generator().manual_seed(SEED + 60)
    grid = IMAGE_SIZE // 2
    noise = torch.randn((1, 4, grid, grid), generator=generator, dtype=torch.float64)
    image_noise = torch.randn(noise.shape, generator=generator, dtype=torch.float64)
    step_noise = [torch.randn(noise.shape, generator=generator, dtype=torch.float64)
                  for _ in range(LCM_STEPS)]
    init_image = torch.randint(0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE),
                               generator=generator).to(torch.float64) / 255.0
    pending_step_noise = []
    lcm_module.randn_tensor = lambda *args, **kwargs: pending_step_noise.pop(0).clone()
    lcm_img2img_module.randn_tensor = lambda *args, **kwargs: noise.clone()
    lcm_img2img_module.retrieve_latents = (
        lambda output, generator=None, sample_mode="sample":
        output.latent_dist.mean + output.latent_dist.std * image_noise)
    with tempfile.TemporaryDirectory() as folder:
        build_lcm_distilled_folder(folder)
        cases = []
        for name, guidance, strength in (
                ("text2img", LCM_DISTILLED_GUIDANCE, None),
                ("text2img_w0", 1.0, None),
                ("img2img_strength", LCM_DISTILLED_GUIDANCE, LCM_STRENGTH)):
            pipe_class = (LatentConsistencyModelPipeline if strength is None
                          else LatentConsistencyModelImg2ImgPipeline)
            pipe = pipe_class.from_pretrained(folder, torch_dtype=torch.float64,
                                              safety_checker=None)
            pipe.set_progress_bar_config(disable=True)
            extra = {"height": IMAGE_SIZE, "width": IMAGE_SIZE, "latents": noise.clone()}
            if strength is not None:
                extra = {"image": init_image, "strength": strength}
            pending_step_noise[:] = step_noise
            after = []

            def on_step_end(_pipe, _step, _t, kwargs):
                after.append(rounded(chw(kwargs["latents"][0])))
                return kwargs
            result = pipe(PROMPT, num_inference_steps=LCM_STEPS, guidance_scale=guidance,
                          output_type="np", callback_on_step_end=on_step_end, **extra)
            assert len(pending_step_noise) == 1, name
            cases.append({"name": name, "guidance_scale": guidance,
                          "strength": 1.0 if strength is None else strength,
                          "timesteps": [int(t) for t in pipe.scheduler.timesteps],
                          "latents_after": after,
                          "image": rounded(result.images[0].tolist())})
            print("LCM distilled", name, cases[-1]["timesteps"])
        pipe = LatentConsistencyModelPipeline.from_pretrained(
            folder, torch_dtype=torch.float64, safety_checker=None)
        prompt_embeds, _ = pipe.encode_prompt(PROMPT, "cpu", 1, False, None)
        w_embedding = pipe.get_guidance_scale_embedding(
            torch.tensor([LCM_DISTILLED_GUIDANCE - 1.0]), LCM_COND_PROJ_DIM,
            torch.float64)
        with torch.no_grad():
            output = pipe.unet(noise, LCM_UNET_CHECK_TIMESTEP,
                               timestep_cond=w_embedding,
                               encoder_hidden_states=prompt_embeds).sample
            plain = pipe.unet(noise, LCM_UNET_CHECK_TIMESTEP,
                              encoder_hidden_states=prompt_embeds).sample
        print("cond embedding moves the UNet output by",
              float((output - plain).abs().max()))
    payload = {"width": IMAGE_SIZE, "height": IMAGE_SIZE, "steps": LCM_STEPS,
               "prompt": PROMPT, "time_cond_proj_dim": LCM_COND_PROJ_DIM,
               "init_image": rounded(init_image[0].permute(1, 2, 0).tolist()),
               "latents": rounded(chw(noise[0])),
               "image_sample_noise": rounded(chw(image_noise[0])),
               "step_noise": [rounded(chw(item[0])) for item in step_noise],
               "unet_check": {"timestep": LCM_UNET_CHECK_TIMESTEP,
                              "guidance_scale": LCM_DISTILLED_GUIDANCE,
                              "output": rounded(chw(output[0]))},
               "cases": cases}
    with open(LCM_DISTILLED_FILE, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    if "--lcm-distilled-only" in sys.argv:
        make_lcm_distilled_oracle()
        sys.exit(0)
    if "--lcm-only" in sys.argv:
        make_lcm_oracle()
        sys.exit(0)
    if "--nonsquare-only" in sys.argv:
        make_nonsquare_oracle()
        sys.exit(0)
    if "--oddsize-only" in sys.argv:
        make_oddsize_oracle()
        sys.exit(0)
    if "--freeu-only" in sys.argv:
        make_freeu_oracles()
        sys.exit(0)
    if "--pag-only" in sys.argv:
        make_pag_oracle()
        sys.exit(0)
    if "--karras-only" in sys.argv:
        make_karras_oracle()
        sys.exit(0)
    if "--samplers-only" not in sys.argv:
        build_folder()
        make_oracle()
    make_sampler_oracle()
    make_nonsquare_oracle()
    make_oddsize_oracle()
    make_lcm_oracle()
    make_lcm_distilled_oracle()
    make_freeu_oracles()
    make_pag_oracle()
    make_karras_oracle()
