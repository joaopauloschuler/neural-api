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

  tests/fixtures/tiny_sd15_controlnet_nonsquare_io.json
      the ControlNet run at CONTROLNET_WIDTH x CONTROLNET_HEIGHT, a 7x5 latent
      grid (odd, so the UNet resizes its upsamples to the skips); check_inputs
      is skipped (it wants multiples of 8, the real VAE factor)
  tests/fixtures/tiny_sd15_t2i_adapter_nonsquare_io.json
      the T2I-Adapter run at ADAPTER_WIDTH x ADAPTER_HEIGHT

Images are multiples of 1/255 so the 0..255 inputs of the Pascal side are exact.

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv):
  python tools/make_pico_sd_conditioning_fixtures.py
  python tools/make_pico_sd_conditioning_fixtures.py --nonsquare-only
  python tools/make_pico_sd_conditioning_fixtures.py --inpaint-strength-only
(--nonsquare-only keeps the committed folders and writes only the two
non-square files; --inpaint-strength-only writes only the strength file.)
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file
from diffusers import (ControlNetModel, EulerDiscreteScheduler,
                       StableDiffusionAdapterPipeline,
                       StableDiffusionControlNetPipeline,
                       StableDiffusionInpaintPipeline, StableDiffusionPipeline,
                       T2IAdapter, UNet2DConditionModel)
import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint as inpaint_module

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
CONTROLNET_NONSQUARE_IO = os.path.join(FIXTURES, "tiny_sd15_controlnet_nonsquare_io.json")
ADAPTER_NONSQUARE_IO = os.path.join(FIXTURES, "tiny_sd15_t2i_adapter_nonsquare_io.json")

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
PROMPT = "a photo of an astronaut riding a horse on mars"


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
        width=IMAGE_SIZE, height=IMAGE_SIZE, steps=STEPS):
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
    result = pipe(PROMPT, height=height, width=width,
                  num_inference_steps=steps, guidance_scale=GUIDANCE,
                  latents=None if latents is None else latents.clone(),
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


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    if "--inpaint-strength-only" in sys.argv:
        make_inpaint_strength()
        sys.exit(0)
    if "--nonsquare-only" not in sys.argv:
        make_controlnet()
        make_adapter()
        make_inpaint()
        make_inpaint_strength()
    make_controlnet_nonsquare()
    make_adapter_nonsquare()
