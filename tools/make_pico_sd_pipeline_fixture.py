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

The pipeline is reloaded from the saved folder and run in float64, latents
passed explicitly so torch's RNG does not matter.

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv):
  python tools/make_pico_sd_pipeline_fixture.py
  python tools/make_pico_sd_pipeline_fixture.py --samplers-only
(--samplers-only keeps the committed folder and rewrites only the sampler file.)
"""
import json
import math
import os
import shutil
import sys
from collections import Counter

import torch
from safetensors.torch import load_file, save_file
from diffusers import (AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler,
                       EulerDiscreteScheduler, PNDMScheduler,
                       StableDiffusionPipeline, UNet2DConditionModel)
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")
PICO_DIR = os.path.join(FIXTURES, "tiny_sd15")
IO_FILE = os.path.join(FIXTURES, "tiny_sd15_pipeline_io.json")
SAMPLERS_FILE = os.path.join(FIXTURES, "tiny_sd15_samplers_io.json")

SEED = 20261021
MERGE_COUNT = 120
TEXT_HIDDEN = 32
IMAGE_SIZE = 16          # VAE factor 2 -> an 8x8 latent grid
STEPS = 3
SAMPLER_STEPS = 4        # PNDM reaches its 4th-order formula, DPM++ 2nd order
GUIDANCE = 7.5
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


if __name__ == "__main__":
    if "--samplers-only" not in sys.argv:
        build_folder()
        make_oracle()
    make_sampler_oracle()
