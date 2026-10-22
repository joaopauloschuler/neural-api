#!/usr/bin/env python3
"""Generate the pico ControlNet / T2I-Adapter / inpainting fixtures that run on
top of the pico Stable Diffusion folder tests/fixtures/tiny_sd15 (made by
tools/make_pico_sd_pipeline_fixture.py), with diffusers float64 oracles for
TStableDiffusionPipeline and its conditioners.

Every component is a tiny RANDOM-weight instance of the real diffusers class
(nothing downloaded), saved in the real folder layout:

  tests/fixtures/tiny_sd15_controlnet/{config.json,diffusion_pytorch_model.safetensors}
      ControlNetModel.from_unet(tiny_sd15 unet), conditioning embedding
      [8, 16] (one stride-2 stage = the pico VAE factor 2), weights perturbed
      so the zero convs are not zero
  tests/fixtures/tiny_sd15_t2i_adapter/{config.json,diffusion_pytorch_model.safetensors}
      T2IAdapter full_adapter, in_channels 1 (a sketch), channels [8, 16],
      downscale_factor 2
  tests/fixtures/tiny_sd15_inpaint/
      a copy of tiny_sd15 whose unet takes 9 channels (latents | mask |
      masked-image latents); model_index names StableDiffusionInpaintPipeline

and the oracle files (all from latents passed explicitly, so torch's RNG does
not matter; values rounded to 10 digits):

  tests/fixtures/tiny_sd15_controlnet_io.json
      StableDiffusionControlNetPipeline: control_image (H,W,3) in [0,1],
      conditioning_scale, latents (C,H,W), timesteps, latents_after, image
  tests/fixtures/tiny_sd15_t2i_adapter_io.json
      StableDiffusionAdapterPipeline: hint_image (H,W,1), conditioning_scale,
      latents, timesteps, latents_after, image
  tests/fixtures/tiny_sd15_inpaint_io.json
      StableDiffusionInpaintPipeline: image (H,W,3) and mask (H,W,1) in [0,1],
      then per case (4-channel tiny_sd15 with DDIM and with Euler, 9-channel
      tiny_sd15_inpaint with DDIM): latents, timesteps, latents_after and the
      decoded output. The VAE encode takes the latent mean (retrieve_latents is
      patched to argmax; the Pascal test sets zero sample noise). Euler latents
      are divided by sqrt(sigma_next^2 + 1), the model-input scale the
      pipeline keeps.
  tests/fixtures/tiny_sd15_inpaint_strength_io.json
      the same image and mask, STRENGTH_STEPS steps at strength STRENGTH (the
      4-channel tiny_sd15 with DDIM and Euler, the 9-channel tiny_sd15_inpaint
      with DDIM) and the 9-channel one at strength 1. The noise ("latents")
      and the unit noise of the image / masked-image latent samples are
      injected (randn_tensor and retrieve_latents patched), so the VAE latents
      are sampled as diffusers does: mean + std * noise.

  tests/fixtures/tiny_sd15_img2img_io.json
      StableDiffusionImg2ImgPipeline on tiny_sd15: a IMG2IMG_WIDTH x
      IMG2IMG_HEIGHT image, STRENGTH_STEPS steps at strength STRENGTH with
      DDIM, PNDM (PLMS), DPM++ 2M, Euler and Euler a, and DDIM at strength 1.
      The noise, the unit noise of the image latent sample and Euler a's
      per-step noise ("step_noise") are injected.

  tests/fixtures/tiny_sd15_controlnet_nonsquare_io.json
      the ControlNet run at CONTROLNET_WIDTH x CONTROLNET_HEIGHT, a 7x5 latent
      grid (odd, so the UNet resizes its upsamples to the skips); check_inputs
      is skipped (it wants multiples of 8, the real VAE factor)
  tests/fixtures/tiny_sd15_t2i_adapter_nonsquare_io.json
      the T2I-Adapter run at ADAPTER_WIDTH x ADAPTER_HEIGHT
  tests/fixtures/tiny_sd15_ip_adapter/{ip-adapter_sd15.safetensors,
      image_encoder/{config.json,model.safetensors}}
      an h94/IP-Adapter ip-adapter_sd15 layout: a random ImageProjModel
      (image_proj.proj / image_proj.norm, 4 tokens) and to_k_ip / to_v_ip per
      cross-attention (ip_adapter.1, .3, ... in unet.attn_processors order),
      plus a pico CLIPVisionModelWithProjection
  tests/fixtures/tiny_sd15_ip_adapter_io.json
      StableDiffusionPipeline + load_ip_adapter: ip_image (H,W,3) in 0..255,
      ip_scale, the image_embeds, latents, timesteps, latents_after, image
  tests/fixtures/tiny_sd15_ip2p/
      a copy of tiny_sd15 whose unet takes 8 channels (latents | image
      latents) and whose scheduler is EulerAncestralDiscreteScheduler, as
      timbrooks/instruct-pix2pix; model_index names
      StableDiffusionInstructPix2PixPipeline
  tests/fixtures/tiny_sd15_ip2p_io.json
      StableDiffusionInstructPix2PixPipeline on tiny_sd15_ip2p: a
      IMG2IMG_WIDTH x IMG2IMG_HEIGHT image, STEPS steps of Euler a and DDIM at
      guidance GUIDANCE / image guidance IMAGE_GUIDANCE (three UNet passes),
      and Euler a at 1 / 1 (one pass). The noise and Euler a's per-step noise
      are injected.

  tests/fixtures/tiny_sd15_lora/{peft.safetensors,kohya.safetensors}
      two random LoRAs on tiny_sd15's unet AND text encoder, linear and conv
      (1x1 proj_in, 3x3 resnet conv1) modules: peft.safetensors as
      save_lora_weights writes it (unet./text_encoder. lora_A/lora_B keys,
      lora_adapter_metadata header with lora_alpha != r), kohya.safetensors
      in the kohya/A1111 layout (lora_unet_/lora_te_ lora_down/lora_up keys
      with an alpha per module)
  tests/fixtures/tiny_sd15_lora_io.json
      StableDiffusionPipeline + load_lora_weights + fuse_lora on tiny_sd15,
      STEPS guided DDIM steps from injected latents, per case (each file at
      its own scale, and both stacked through set_adapters): files, scales,
      timesteps, latents_after, image

Images are multiples of 1/255 so the 0..255 inputs of the Pascal side are exact.

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv):
  python tools/make_pico_sd_conditioning_fixtures.py
  python tools/make_pico_sd_conditioning_fixtures.py --nonsquare-only
  python tools/make_pico_sd_conditioning_fixtures.py --inpaint-strength-only
  python tools/make_pico_sd_conditioning_fixtures.py --ip-adapter-only
  python tools/make_pico_sd_conditioning_fixtures.py --img2img-only
  python tools/make_pico_sd_conditioning_fixtures.py --ip2p-only
  python tools/make_pico_sd_conditioning_fixtures.py --lora-only
(--nonsquare-only keeps the committed folders and writes only the two
non-square files; --inpaint-strength-only writes only the strength file;
--ip-adapter-only writes only the IP-Adapter folder and its file;
--img2img-only writes only the img2img file; --ip2p-only writes only the
InstructPix2Pix folder and its file; --lora-only writes only the LoRA folder
and its file.)
"""
import json
import os
import shutil
import sys

import numpy
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
from diffusers import (ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler, StableDiffusionAdapterPipeline,
                       StableDiffusionControlNetPipeline,
                       StableDiffusionImg2ImgPipeline,
                       StableDiffusionInpaintPipeline,
                       StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline,
                       T2IAdapter, UNet2DConditionModel)
import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint as inpaint_module
import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img as img2img_module
import diffusers.schedulers.scheduling_euler_ancestral_discrete as euler_a_module

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")
PICO_DIR = os.path.join(FIXTURES, "tiny_sd15")
CONTROLNET_DIR = os.path.join(FIXTURES, "tiny_sd15_controlnet")
ADAPTER_DIR = os.path.join(FIXTURES, "tiny_sd15_t2i_adapter")
INPAINT_DIR = os.path.join(FIXTURES, "tiny_sd15_inpaint")
CONTROLNET_IO = os.path.join(FIXTURES, "tiny_sd15_controlnet_io.json")
ADAPTER_IO = os.path.join(FIXTURES, "tiny_sd15_t2i_adapter_io.json")
INPAINT_IO = os.path.join(FIXTURES, "tiny_sd15_inpaint_io.json")
INPAINT_STRENGTH_IO = os.path.join(FIXTURES, "tiny_sd15_inpaint_strength_io.json")
IMG2IMG_IO = os.path.join(FIXTURES, "tiny_sd15_img2img_io.json")
CONTROLNET_NONSQUARE_IO = os.path.join(FIXTURES, "tiny_sd15_controlnet_nonsquare_io.json")
ADAPTER_NONSQUARE_IO = os.path.join(FIXTURES, "tiny_sd15_t2i_adapter_nonsquare_io.json")
IP_ADAPTER_DIR = os.path.join(FIXTURES, "tiny_sd15_ip_adapter")
IP_ADAPTER_IO = os.path.join(FIXTURES, "tiny_sd15_ip_adapter_io.json")
IP2P_DIR = os.path.join(FIXTURES, "tiny_sd15_ip2p")
IP2P_IO = os.path.join(FIXTURES, "tiny_sd15_ip2p_io.json")
LORA_DIR = os.path.join(FIXTURES, "tiny_sd15_lora")
LORA_IO = os.path.join(FIXTURES, "tiny_sd15_lora_io.json")

SEED = 20261022
IMAGE_SIZE = 16
CONTROLNET_WIDTH = 14
CONTROLNET_HEIGHT = 10
ADAPTER_WIDTH = 24       # sides must be multiples of the adapter's 2 * 2
ADAPTER_HEIGHT = 8
STEPS = 3
GUIDANCE = 7.5
CONTROLNET_SCALE = 0.5
ADAPTER_SCALE = 0.8
STRENGTH = 0.6
STRENGTH_STEPS = 5
IMG2IMG_WIDTH = 16
IMG2IMG_HEIGHT = 12
IP_SCALE = 0.7
IP_IMAGE_WIDTH = 20      # resized (shortest edge) and center-cropped to 8x8
IP_IMAGE_HEIGHT = 13
IP_TOKENS = 4            # diffusers ImageProjection of an ip-adapter_sd15
IMAGE_GUIDANCE = 1.5
PROMPT = "a photo of an astronaut riding a horse on mars"
LORA_RANK = 2
LORA_ALPHA = 3.0         # peft.safetensors metadata: scale alpha / r = 1.5
KOHYA_ALPHA = 1.0        # kohya.safetensors per-module alpha: scale 0.5
LORA_UNET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "ff.net.2",
                     "conv1", "time_emb_proj"]
LORA_TEXT_MODULES = ["q_proj", "v_proj", "out_proj", "fc1"]
LORA_CASES = (("peft", ["peft.safetensors"], [0.8]),
              ("kohya", ["kohya.safetensors"], [0.7]),
              ("stack", ["peft.safetensors", "kohya.safetensors"], [0.8, -0.6]))


def perturb(module, std=0.05):
    with torch.no_grad():
        for param in module.parameters():
            param.add_(torch.randn_like(param) * std)


def rounded(value):
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return float("%.10g" % value)


def chw(tensor):
    return rounded(tensor.to(torch.float64).tolist())


def byte_image(generator, channels, width=IMAGE_SIZE, height=IMAGE_SIZE):
    """(1, C, H, W) float64 in [0, 1], multiples of 1/255."""
    levels = torch.randint(0, 256, (1, channels, height, width),
                           generator=generator)
    return levels.to(torch.float64) / 255.0


def hwc(image):
    return rounded(image[0].permute(1, 2, 0).tolist())


def initial_latents(width=IMAGE_SIZE, height=IMAGE_SIZE):
    generator = torch.Generator().manual_seed(SEED)
    return torch.randn((1, 4, height // 2, width // 2), generator=generator,
                       dtype=torch.float64)


def run(pipe, extra, latents, euler=False, legacy_callback=False,
        width=IMAGE_SIZE, height=IMAGE_SIZE, steps=STEPS, sized=True,
        guidance=GUIDANCE):
    pipe.set_progress_bar_config(disable=True)
    if width % 8 or height % 8:
        pipe.check_inputs = lambda *args, **kwargs: None
    after = []

    def record(sample):
        if euler:
            sigma = float(pipe.scheduler.sigmas[pipe.scheduler.step_index])
            sample = sample / (sigma ** 2 + 1) ** 0.5
        after.append(chw(sample))

    def on_step_end(_pipe, _step, _t, kwargs):
        record(kwargs["latents"][0])
        return kwargs
    if legacy_callback:
        # StableDiffusionAdapterPipeline has only callback(step, t, latents).
        extra = dict(extra, callback=lambda _step, _t, sample: record(sample[0]),
                     callback_steps=1)
    else:
        extra = dict(extra, callback_on_step_end=on_step_end)
    if sized:
        # img2img takes its size from the image and draws its own noise.
        extra = dict(extra, height=height, width=width,
                     latents=None if latents is None else latents.clone())
    result = pipe(PROMPT, num_inference_steps=steps, guidance_scale=guidance,
                  output_type="np", **extra)
    return {"timesteps": [int(t) for t in pipe.scheduler.timesteps],
            "latents_after": after,
            "image": rounded(result.images[0].tolist())}


def base_components():
    base = StableDiffusionPipeline.from_pretrained(PICO_DIR, torch_dtype=torch.float64,
                                                   safety_checker=None)
    return dict(vae=base.vae, text_encoder=base.text_encoder,
                tokenizer=base.tokenizer, unet=base.unet,
                scheduler=base.scheduler, safety_checker=None,
                feature_extractor=None, requires_safety_checker=False)


def make_controlnet():
    torch.manual_seed(SEED)
    parts = base_components()
    controlnet = ControlNetModel.from_unet(
        parts["unet"], conditioning_embedding_out_channels=(8, 16))
    perturb(controlnet)
    if os.path.isdir(CONTROLNET_DIR):
        shutil.rmtree(CONTROLNET_DIR)
    controlnet.to(torch.float32).save_pretrained(CONTROLNET_DIR, safe_serialization=True)
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_DIR, torch_dtype=torch.float64)
    pipe = StableDiffusionControlNetPipeline(controlnet=controlnet, **parts)
    image = byte_image(torch.Generator().manual_seed(SEED + 1), 3)
    latents = initial_latents()
    out = run(pipe, {"image": image, "controlnet_conditioning_scale": CONTROLNET_SCALE},
              latents)
    payload = {"image_size": IMAGE_SIZE, "steps": STEPS, "guidance_scale": GUIDANCE,
               "prompt": PROMPT, "conditioning_scale": CONTROLNET_SCALE,
               "control_image": hwc(image), "latents": chw(latents[0])}
    payload.update(out)
    with open(CONTROLNET_IO, "w") as f:
        json.dump(payload, f)
    print("controlnet timesteps", out["timesteps"])


def make_adapter():
    torch.manual_seed(SEED + 10)
    parts = base_components()
    adapter = T2IAdapter(in_channels=1, channels=[8, 16], num_res_blocks=1,
                         downscale_factor=2, adapter_type="full_adapter")
    perturb(adapter)
    if os.path.isdir(ADAPTER_DIR):
        shutil.rmtree(ADAPTER_DIR)
    adapter.to(torch.float32).save_pretrained(ADAPTER_DIR, safe_serialization=True)
    adapter = T2IAdapter.from_pretrained(ADAPTER_DIR, torch_dtype=torch.float64)
    pipe = StableDiffusionAdapterPipeline(adapter=adapter, **parts)
    hint = byte_image(torch.Generator().manual_seed(SEED + 11), 1)
    latents = initial_latents()
    out = run(pipe, {"image": hint, "adapter_conditioning_scale": ADAPTER_SCALE},
              latents, legacy_callback=True)
    payload = {"image_size": IMAGE_SIZE, "steps": STEPS, "guidance_scale": GUIDANCE,
               "prompt": PROMPT, "conditioning_scale": ADAPTER_SCALE,
               "hint_image": hwc(hint), "latents": chw(latents[0])}
    payload.update(out)
    with open(ADAPTER_IO, "w") as f:
        json.dump(payload, f)
    print("adapter timesteps", out["timesteps"])


def make_inpaint_folder():
    torch.manual_seed(SEED + 20)
    if os.path.isdir(INPAINT_DIR):
        shutil.rmtree(INPAINT_DIR)
    shutil.copytree(PICO_DIR, INPAINT_DIR)
    base = UNet2DConditionModel.from_pretrained(os.path.join(PICO_DIR, "unet"))
    config = dict(base.config)
    config["in_channels"] = 9
    unet = UNet2DConditionModel.from_config(config)
    perturb(unet)
    unet_dir = os.path.join(INPAINT_DIR, "unet")
    shutil.rmtree(unet_dir)
    unet.save_pretrained(unet_dir, safe_serialization=True)
    index_file = os.path.join(INPAINT_DIR, "model_index.json")
    with open(index_file) as f:
        index = json.load(f)
    index["_class_name"] = "StableDiffusionInpaintPipeline"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)


def make_inpaint():
    make_inpaint_folder()
    # Our VAE encoder returns the latent mean; diffusers samples by default.
    inpaint_module.retrieve_latents = (
        lambda output, generator=None, sample_mode="sample": output.latent_dist.mode())
    image, mask = inpaint_image_and_mask()
    latents = initial_latents()
    cases = []
    for name, folder, euler in (("unet4_ddim", PICO_DIR, False),
                                ("unet4_euler", PICO_DIR, True),
                                ("unet9_ddim", INPAINT_DIR, False)):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            folder, torch_dtype=torch.float64, safety_checker=None)
        if euler:
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        out = run(pipe, {"image": image, "mask_image": mask}, latents, euler)
        out["name"] = name
        out["folder"] = os.path.basename(folder)
        out["scheduler"] = type(pipe.scheduler).__name__
        cases.append(out)
        print(name, out["timesteps"])
    payload = {"image_size": IMAGE_SIZE, "steps": STEPS, "guidance_scale": GUIDANCE,
               "prompt": PROMPT, "init_image": hwc(image), "mask": hwc(mask),
               "latents": chw(latents[0]), "cases": cases}
    with open(INPAINT_IO, "w") as f:
        json.dump(payload, f)


def inpaint_image_and_mask():
    image = byte_image(torch.Generator().manual_seed(SEED + 21), 3)
    mask = torch.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float64)
    mask[:, :, 3:13, 8:] = 1.0
    mask[:, :, 0:4, 0:4] = 1.0
    return image, mask


def make_inpaint_strength():
    image, mask = inpaint_image_and_mask()
    noise = initial_latents()
    generator = torch.Generator().manual_seed(SEED + 22)
    image_noise = torch.randn(noise.shape, generator=generator, dtype=torch.float64)
    masked_image_noise = torch.randn(noise.shape, generator=generator,
                                     dtype=torch.float64)
    sample_noise = []

    def sampled(output, generator=None, sample_mode="sample"):
        dist = output.latent_dist
        return dist.mean + dist.std * sample_noise.pop(0)
    inpaint_module.retrieve_latents = sampled
    inpaint_module.randn_tensor = lambda *args, **kwargs: noise.clone()
    cases = []
    for name, folder, euler, strength in (
            ("unet4_ddim_strength", PICO_DIR, False, STRENGTH),
            ("unet4_euler_strength", PICO_DIR, True, STRENGTH),
            ("unet9_ddim_strength", INPAINT_DIR, False, STRENGTH),
            ("unet9_ddim_sampled", INPAINT_DIR, False, 1.0)):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            folder, torch_dtype=torch.float64, safety_checker=None)
        if euler:
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        # prepare_latents encodes the image (4 channels or strength < 1), then
        # prepare_mask_latents the masked image.
        if folder == INPAINT_DIR and strength == 1.0:
            sample_noise[:] = [masked_image_noise]
        else:
            sample_noise[:] = [image_noise, masked_image_noise]
        out = run(pipe, {"image": image, "mask_image": mask, "strength": strength},
                  None, euler, steps=STRENGTH_STEPS)
        assert not sample_noise, name
        # The steps that ran: get_timesteps keeps the last ones.
        out["timesteps"] = out["timesteps"][-len(out["latents_after"]):]
        out["name"] = name
        out["folder"] = os.path.basename(folder)
        out["scheduler"] = type(pipe.scheduler).__name__
        out["strength"] = strength
        cases.append(out)
        print(name, out["timesteps"], len(out["latents_after"]), "steps run")
    payload = {"image_size": IMAGE_SIZE, "steps": STRENGTH_STEPS,
               "guidance_scale": GUIDANCE, "prompt": PROMPT,
               "init_image": hwc(image), "mask": hwc(mask),
               "latents": chw(noise[0]), "image_sample_noise": chw(image_noise[0]),
               "masked_image_sample_noise": chw(masked_image_noise[0]),
               "cases": cases}
    with open(INPAINT_STRENGTH_IO, "w") as f:
        json.dump(payload, f)


def make_img2img():
    image = byte_image(torch.Generator().manual_seed(SEED + 30), 3,
                       IMG2IMG_WIDTH, IMG2IMG_HEIGHT)
    noise = initial_latents(IMG2IMG_WIDTH, IMG2IMG_HEIGHT)
    generator = torch.Generator().manual_seed(SEED + 31)
    image_noise = torch.randn(noise.shape, generator=generator, dtype=torch.float64)
    step_noise = [torch.randn(noise.shape, generator=generator, dtype=torch.float64)
                  for _ in range(STRENGTH_STEPS)]
    img2img_module.retrieve_latents = (
        lambda output, generator=None, sample_mode="sample":
        output.latent_dist.mean + output.latent_dist.std * image_noise)
    img2img_module.randn_tensor = lambda *args, **kwargs: noise.clone()
    pending_step_noise = []
    euler_a_module.randn_tensor = lambda *args, **kwargs: pending_step_noise.pop(0).clone()
    base = StableDiffusionImg2ImgPipeline.from_pretrained(
        PICO_DIR, torch_dtype=torch.float64, safety_checker=None)
    base_config = base.scheduler.config
    cases = []
    for name, scheduler, strength in (
            ("ddim_strength", None, STRENGTH),
            ("ddim_full_strength", None, 1.0),
            ("pndm_strength", PNDMScheduler.from_config(base_config, skip_prk_steps=True), STRENGTH),
            ("dpmpp2m_strength", DPMSolverMultistepScheduler.from_config(base_config), STRENGTH),
            ("euler_strength", EulerDiscreteScheduler.from_config(base_config), STRENGTH),
            ("euler_a_strength", EulerAncestralDiscreteScheduler.from_config(base_config), STRENGTH)):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            PICO_DIR, torch_dtype=torch.float64, safety_checker=None)
        if scheduler is not None:
            pipe.scheduler = scheduler
        pending_step_noise[:] = step_noise
        euler = isinstance(pipe.scheduler, (EulerDiscreteScheduler,
                                            EulerAncestralDiscreteScheduler))
        out = run(pipe, {"image": image, "strength": strength}, None, euler,
                  steps=STRENGTH_STEPS, sized=False)
        # The steps that ran: get_timesteps keeps the last ones.
        out["timesteps"] = out["timesteps"][-len(out["latents_after"]):]
        out["name"] = name
        out["scheduler"] = type(pipe.scheduler).__name__
        # Strict JSON: drop non-finite values (DPM++ lambda_min_clipped = -inf).
        out["scheduler_config"] = {
            key: value for key, value in pipe.scheduler.config.items()
            if not (isinstance(value, float) and not numpy.isfinite(value))}
        out["strength"] = strength
        cases.append(out)
        print(name, out["timesteps"], len(out["latents_after"]), "steps run")
    payload = {"width": IMG2IMG_WIDTH, "height": IMG2IMG_HEIGHT,
               "steps": STRENGTH_STEPS, "guidance_scale": GUIDANCE,
               "prompt": PROMPT, "init_image": hwc(image),
               "latents": chw(noise[0]), "image_sample_noise": chw(image_noise[0]),
               "step_noise": [chw(item[0]) for item in step_noise],
               "cases": cases}
    with open(IMG2IMG_IO, "w") as f:
        json.dump(payload, f)


def make_controlnet_nonsquare():
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_DIR, torch_dtype=torch.float64)
    pipe = StableDiffusionControlNetPipeline(controlnet=controlnet, **base_components())
    image = byte_image(torch.Generator().manual_seed(SEED + 2), 3,
                       CONTROLNET_WIDTH, CONTROLNET_HEIGHT)
    latents = initial_latents(CONTROLNET_WIDTH, CONTROLNET_HEIGHT)
    out = run(pipe, {"image": image, "controlnet_conditioning_scale": CONTROLNET_SCALE},
              latents, width=CONTROLNET_WIDTH, height=CONTROLNET_HEIGHT)
    payload = {"width": CONTROLNET_WIDTH, "height": CONTROLNET_HEIGHT, "steps": STEPS,
               "guidance_scale": GUIDANCE, "prompt": PROMPT,
               "conditioning_scale": CONTROLNET_SCALE,
               "control_image": hwc(image), "latents": chw(latents[0])}
    payload.update(out)
    with open(CONTROLNET_NONSQUARE_IO, "w") as f:
        json.dump(payload, f)
    print("controlnet non-square", len(out["image"][0]), "x", len(out["image"]))


def make_adapter_nonsquare():
    adapter = T2IAdapter.from_pretrained(ADAPTER_DIR, torch_dtype=torch.float64)
    pipe = StableDiffusionAdapterPipeline(adapter=adapter, **base_components())
    hint = byte_image(torch.Generator().manual_seed(SEED + 12), 1,
                      ADAPTER_WIDTH, ADAPTER_HEIGHT)
    latents = initial_latents(ADAPTER_WIDTH, ADAPTER_HEIGHT)
    out = run(pipe, {"image": hint, "adapter_conditioning_scale": ADAPTER_SCALE},
              latents, legacy_callback=True, width=ADAPTER_WIDTH, height=ADAPTER_HEIGHT)
    payload = {"width": ADAPTER_WIDTH, "height": ADAPTER_HEIGHT, "steps": STEPS,
               "guidance_scale": GUIDANCE, "prompt": PROMPT,
               "conditioning_scale": ADAPTER_SCALE,
               "hint_image": hwc(hint), "latents": chw(latents[0])}
    payload.update(out)
    with open(ADAPTER_NONSQUARE_IO, "w") as f:
        json.dump(payload, f)
    print("adapter non-square", len(out["image"][0]), "x", len(out["image"]))


def make_ip_adapter():
    torch.manual_seed(SEED + 30)
    parts = base_components()
    unet = parts["unet"]
    vision_config = CLIPVisionConfig(hidden_size=16, intermediate_size=24,
                                     num_hidden_layers=2, num_attention_heads=2,
                                     image_size=8, patch_size=4, projection_dim=12,
                                     hidden_act="quick_gelu")
    encoder = CLIPVisionModelWithProjection(vision_config)
    perturb(encoder)
    cross = unet.config.cross_attention_dim
    state = {
        "image_proj.proj.weight": torch.randn(IP_TOKENS * cross, 12) * 0.3,
        "image_proj.proj.bias": torch.randn(IP_TOKENS * cross) * 0.3,
        "image_proj.norm.weight": 1 + torch.randn(cross) * 0.1,
        "image_proj.norm.bias": torch.randn(cross) * 0.1}
    # load_ip_adapter numbers the cross-attention processors 1, 3, 5, ...
    key_id = 1
    for name, proc in unet.attn_processors.items():
        if name.endswith("attn1.processor"):
            continue
        hidden = unet.get_submodule(name[:-len(".processor")]).to_q.out_features
        print("ip_adapter.%d" % key_id, name, hidden)
        state["ip_adapter.%d.to_k_ip.weight" % key_id] = torch.randn(hidden, cross) * 0.3
        state["ip_adapter.%d.to_v_ip.weight" % key_id] = torch.randn(hidden, cross) * 0.3
        key_id += 2
    if os.path.isdir(IP_ADAPTER_DIR):
        shutil.rmtree(IP_ADAPTER_DIR)
    os.makedirs(IP_ADAPTER_DIR)
    save_file({k: v.to(torch.float32).contiguous() for k, v in state.items()},
              os.path.join(IP_ADAPTER_DIR, "ip-adapter_sd15.safetensors"))
    encoder.to(torch.float32).save_pretrained(
        os.path.join(IP_ADAPTER_DIR, "image_encoder"), safe_serialization=True)
    pipe = StableDiffusionPipeline(**parts)
    pipe.register_modules(image_encoder=None)
    pipe.load_ip_adapter(IP_ADAPTER_DIR, subfolder="",
                         weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(IP_SCALE)
    pipe.to(torch.float64)
    generator = torch.Generator().manual_seed(SEED + 31)
    levels = torch.randint(0, 256, (IP_IMAGE_HEIGHT, IP_IMAGE_WIDTH, 3),
                           generator=generator)
    ip_image = Image.fromarray(levels.numpy().astype(numpy.uint8), "RGB")
    pixels = pipe.feature_extractor(ip_image, return_tensors="pt").pixel_values
    image_embeds = pipe.image_encoder(pixels.to(torch.float64)).image_embeds
    latents = initial_latents()
    out = run(pipe, {"ip_adapter_image": ip_image}, latents)
    payload = {"image_size": IMAGE_SIZE, "steps": STEPS, "guidance_scale": GUIDANCE,
               "prompt": PROMPT, "ip_scale": IP_SCALE,
               "ip_image": levels.tolist(),
               "image_embeds": rounded(image_embeds[0].tolist()),
               "latents": chw(latents[0])}
    payload.update(out)
    with open(IP_ADAPTER_IO, "w") as f:
        json.dump(payload, f)
    print("ip-adapter timesteps", out["timesteps"])


def make_ip2p_folder():
    torch.manual_seed(SEED + 40)
    if os.path.isdir(IP2P_DIR):
        shutil.rmtree(IP2P_DIR)
    shutil.copytree(PICO_DIR, IP2P_DIR)
    base = UNet2DConditionModel.from_pretrained(os.path.join(PICO_DIR, "unet"))
    config = dict(base.config)
    config["in_channels"] = 8
    unet = UNet2DConditionModel.from_config(config)
    perturb(unet)
    unet_dir = os.path.join(IP2P_DIR, "unet")
    shutil.rmtree(unet_dir)
    unet.save_pretrained(unet_dir, safe_serialization=True)
    scheduler_file = os.path.join(IP2P_DIR, "scheduler", "scheduler_config.json")
    with open(scheduler_file) as f:
        scheduler = json.load(f)
    scheduler["_class_name"] = "EulerAncestralDiscreteScheduler"
    with open(scheduler_file, "w") as f:
        json.dump(scheduler, f, indent=2)
    index_file = os.path.join(IP2P_DIR, "model_index.json")
    with open(index_file) as f:
        index = json.load(f)
    index["_class_name"] = "StableDiffusionInstructPix2PixPipeline"
    index["scheduler"] = ["diffusers", "EulerAncestralDiscreteScheduler"]
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)


def make_ip2p():
    make_ip2p_folder()
    image = byte_image(torch.Generator().manual_seed(SEED + 41), 3,
                       IMG2IMG_WIDTH, IMG2IMG_HEIGHT)
    noise = initial_latents(IMG2IMG_WIDTH, IMG2IMG_HEIGHT)
    generator = torch.Generator().manual_seed(SEED + 42)
    step_noise = [torch.randn(noise.shape, generator=generator, dtype=torch.float64)
                  for _ in range(STEPS)]
    pending_step_noise = []
    euler_a_module.randn_tensor = lambda *args, **kwargs: pending_step_noise.pop(0).clone()
    cases = []
    for name, ddim, guidance, image_guidance in (
            ("euler_a", False, GUIDANCE, IMAGE_GUIDANCE),
            ("ddim", True, GUIDANCE, IMAGE_GUIDANCE),
            ("euler_a_unguided", False, 1.0, 1.0)):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            IP2P_DIR, torch_dtype=torch.float64, safety_checker=None)
        if ddim:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pending_step_noise[:] = step_noise
        euler = not ddim
        out = run(pipe, {"image": image, "image_guidance_scale": image_guidance,
                         "latents": noise.clone()},
                  None, euler, sized=False, guidance=guidance)
        out["name"] = name
        out["scheduler"] = type(pipe.scheduler).__name__
        out["guidance_scale"] = guidance
        out["image_guidance_scale"] = image_guidance
        cases.append(out)
        print(name, out["timesteps"])
    payload = {"width": IMG2IMG_WIDTH, "height": IMG2IMG_HEIGHT,
               "steps": STEPS, "prompt": PROMPT, "init_image": hwc(image),
               "latents": chw(noise[0]),
               "step_noise": [chw(item[0]) for item in step_noise],
               "cases": cases}
    with open(IP2P_IO, "w") as f:
        json.dump(payload, f)


def random_lora_layers(model, modules, seed):
    """Adds a LoRA adapter (r LORA_RANK, alpha LORA_ALPHA) on modules with
    random A and B; returns its config and peft state dict (float32)."""
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=modules,
                        init_lora_weights=False)
    model.add_adapter(config)
    generator = torch.Generator().manual_seed(seed)
    layers = get_peft_model_state_dict(model)
    for key in sorted(layers):
        value = torch.randn(layers[key].shape, generator=generator,
                            dtype=torch.float64) * 0.2
        layers[key] = value.to(torch.float32).contiguous()
    return config, layers


def adapter_metadata(config):
    """The LoraConfig dict save_lora_weights stores, minus the local path."""
    metadata = config.to_dict()
    metadata["base_model_name_or_path"] = None
    return metadata


def kohya_layers(layers, kohya_prefix, alpha):
    """peft keys "<m>.lora_A.weight" -> "<kohya_prefix><m with _>.lora_down
    .weight" (lora_up, alpha alongside)."""
    out = {}
    for key, value in layers.items():
        module, factor = key.rsplit(".lora_", 1)
        flat = kohya_prefix + module.replace(".", "_")
        out[flat + (".lora_down.weight" if factor.startswith("A") else
                    ".lora_up.weight")] = value
        out[flat + ".alpha"] = torch.tensor(alpha, dtype=torch.float32)
    return out


def make_lora():
    os.makedirs(LORA_DIR, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(PICO_DIR, torch_dtype=torch.float32,
                                                   safety_checker=None)
    unet_config, unet_layers = random_lora_layers(pipe.unet, LORA_UNET_MODULES,
                                                  SEED + 50)
    text_config, text_layers = random_lora_layers(pipe.text_encoder,
                                                  LORA_TEXT_MODULES, SEED + 51)
    for name in ("peft.safetensors", "kohya.safetensors"):
        if os.path.exists(os.path.join(LORA_DIR, name)):
            os.remove(os.path.join(LORA_DIR, name))
    StableDiffusionPipeline.save_lora_weights(
        LORA_DIR, unet_lora_layers=unet_layers, text_encoder_lora_layers=text_layers,
        weight_name="peft.safetensors",
        unet_lora_adapter_metadata=adapter_metadata(unet_config),
        text_encoder_lora_adapter_metadata=adapter_metadata(text_config))
    pipe = StableDiffusionPipeline.from_pretrained(PICO_DIR, torch_dtype=torch.float32,
                                                   safety_checker=None)
    _, unet_layers = random_lora_layers(pipe.unet, LORA_UNET_MODULES, SEED + 52)
    _, text_layers = random_lora_layers(pipe.text_encoder, LORA_TEXT_MODULES,
                                        SEED + 53)
    kohya = kohya_layers(unet_layers, "lora_unet_", KOHYA_ALPHA)
    kohya.update(kohya_layers(text_layers, "lora_te_", KOHYA_ALPHA))
    save_file(kohya, os.path.join(LORA_DIR, "kohya.safetensors"))
    latents = initial_latents()
    cases = []
    for name, files, scales in LORA_CASES:
        pipe = StableDiffusionPipeline.from_pretrained(
            PICO_DIR, torch_dtype=torch.float64, safety_checker=None)
        adapter_names = []
        for file_name in files:
            adapter_names.append(file_name.split(".")[0])
            pipe.load_lora_weights(LORA_DIR, weight_name=file_name,
                                   adapter_name=adapter_names[-1])
        pipe.set_adapters(adapter_names, scales)
        pipe.fuse_lora(adapter_names=adapter_names)
        pipe.unload_lora_weights()
        out = run(pipe, {}, latents)
        out.update(name=name, files=files, scales=scales)
        cases.append(out)
        print(name, out["timesteps"])
    payload = {"image_size": IMAGE_SIZE, "steps": STEPS, "guidance_scale": GUIDANCE,
               "prompt": PROMPT, "latents": chw(latents[0]), "cases": cases}
    with open(LORA_IO, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    if "--lora-only" in sys.argv:
        make_lora()
        sys.exit(0)
    if "--ip2p-only" in sys.argv:
        make_ip2p()
        sys.exit(0)
    if "--ip-adapter-only" in sys.argv:
        make_ip_adapter()
        sys.exit(0)
    if "--inpaint-strength-only" in sys.argv:
        make_inpaint_strength()
        sys.exit(0)
    if "--img2img-only" in sys.argv:
        make_img2img()
        sys.exit(0)
    if "--nonsquare-only" not in sys.argv:
        make_controlnet()
        make_adapter()
        make_inpaint()
        make_inpaint_strength()
        make_img2img()
    make_controlnet_nonsquare()
    make_adapter_nonsquare()
    if "--nonsquare-only" not in sys.argv:
        make_ip_adapter()
        make_ip2p()
        make_lora()
