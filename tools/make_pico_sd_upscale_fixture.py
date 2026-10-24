#!/usr/bin/env python3
"""Generate the pico Stable Diffusion x4 upscaler fixture for
TStableDiffusionUpscaler: a tiny RANDOM-weight folder shaped like
stabilityai/stable-diffusion-x4-upscaler (nothing downloaded):

  tests/fixtures/tiny_sd_x4_upscaler/
    model_index.json                     StableDiffusionUpscalePipeline,
                                         max_noise_level 350
    scheduler/scheduler_config.json      DDIMScheduler, v_prediction (as x4)
    low_res_scheduler/scheduler_config.json  DDPMScheduler, clip_sample true
    text_encoder/, tokenizer/            copied from tests/fixtures/tiny_sd2
    unet/  in_channels 7 (latents | noised low-res RGB), num_class_embeds 1000
           (class_labels = noise_level), only_cross_attention [T, T, F] over
           DownBlock2D + 2 CrossAttnDownBlock2D, use_linear_projection
    vae/   3 blocks (factor 4), scaling_factor 0.08333

and two diffusers float64 oracles:

  tests/fixtures/tiny_sd_x4_upscaler_unet_io.json  one UNet forward: sample
      (7, H, W), timestep, class_label, encoder_hidden_states (S, D), noise
  tests/fixtures/tiny_sd_x4_upscaler_io.json  StableDiffusionUpscalePipeline:
      per case the low-res PIL image (H, W, 3) in 0..255 (one already a
      multiple of 4, one resized by the pipeline), noise_level, scheduler,
      the low_res_scheduler noise ("image_noise", C, H, W), timesteps,
      latents_after, image. The latents are passed explicitly.

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv):
  python tools/make_pico_sd_upscale_fixture.py
"""
import json
import os
import shutil

import torch
from PIL import Image
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       EulerDiscreteScheduler, StableDiffusionUpscalePipeline,
                       UNet2DConditionModel)
import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale as upscale_module
from transformers import CLIPTextModel, CLIPTokenizer

import make_pico_sd_conditioning_fixtures as conditioning_fixture

FIXTURES = conditioning_fixture.FIXTURES
SD2_DIR = os.path.join(FIXTURES, "tiny_sd2")
UPSCALER_DIR = os.path.join(FIXTURES, "tiny_sd_x4_upscaler")
UNET_IO = os.path.join(FIXTURES, "tiny_sd_x4_upscaler_unet_io.json")
PIPELINE_IO = os.path.join(FIXTURES, "tiny_sd_x4_upscaler_io.json")
SEED = 20261023
TEXT_HIDDEN = 32
LOW_RES_WIDTH = 12       # latent grid = the low-res size (a multiple of 4)
LOW_RES_HEIGHT = 8
STEPS = 3
GUIDANCE = 9.0           # the diffusers upscale default
# (name, scheduler, PIL image size, noise_level)
CASES = (("ddim", "ddim", (12, 8), 20),
         ("euler_resized", "euler", (13, 10), 150))


def build_folder():
    torch.manual_seed(SEED)
    if os.path.isdir(UPSCALER_DIR):
        shutil.rmtree(UPSCALER_DIR)
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(SD2_DIR, "text_encoder"))
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(SD2_DIR, "tokenizer"))
    unet = UNet2DConditionModel(
        sample_size=LOW_RES_WIDTH, in_channels=7, out_channels=4,
        block_out_channels=(8, 16, 16),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D",
                          "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        only_cross_attention=[True, True, False], num_class_embeds=1000,
        layers_per_block=1, cross_attention_dim=TEXT_HIDDEN,
        attention_head_dim=2, use_linear_projection=True, norm_num_groups=4)
    conditioning_fixture.perturb(unet)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=(8, 16, 16), layers_per_block=1, latent_channels=4,
        norm_num_groups=4, sample_size=LOW_RES_WIDTH * 4, scaling_factor=0.08333)
    conditioning_fixture.perturb(vae)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False,
        steps_offset=1, prediction_type="v_prediction")
    low_res_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="scaled_linear", clip_sample=True,
        variance_type="fixed_small")
    pipe = StableDiffusionUpscalePipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        low_res_scheduler=low_res_scheduler, scheduler=scheduler,
        max_noise_level=350)
    pipe.save_pretrained(UPSCALER_DIR, safe_serialization=True)
    # The tokenizer and text encoder files exactly as tiny_sd2 ships them.
    for component in ("tokenizer", "text_encoder"):
        shutil.rmtree(os.path.join(UPSCALER_DIR, component))
        shutil.copytree(os.path.join(SD2_DIR, component),
                        os.path.join(UPSCALER_DIR, component))


def load_pipeline(scheduler_name):
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        UPSCALER_DIR, torch_dtype=torch.float64)
    if scheduler_name == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    assert pipe.vae.config.scaling_factor == 0.08333
    assert pipe.unet.config.in_channels == 7
    return pipe


def make_unet_oracle():
    """One UNet forward with class_labels: 7-channel sample, both kinds of
    attention block, the class embedding added to the time embedding."""
    pipe = load_pipeline("ddim")
    generator = torch.Generator().manual_seed(SEED + 1)
    sample = torch.randn((1, 7, LOW_RES_HEIGHT, LOW_RES_WIDTH),
                         generator=generator, dtype=torch.float64)
    states = torch.randn((1, 5, TEXT_HIDDEN), generator=generator,
                         dtype=torch.float64)
    timestep, class_label = 417, 123
    noise = pipe.unet(sample, timestep, encoder_hidden_states=states,
                      class_labels=torch.tensor([class_label])).sample
    payload = {"sample": conditioning_fixture.chw(sample[0]),
               "timestep": timestep, "class_label": class_label,
               "encoder_hidden_states": conditioning_fixture.rounded(
                   states[0].tolist()),
               "noise": conditioning_fixture.chw(noise[0])}
    with open(UNET_IO, "w") as f:
        json.dump(payload, f)
    print("unet oracle: noise range", float(noise.min()), float(noise.max()))


def make_pipeline_oracle():
    latents = torch.randn((1, 4, LOW_RES_HEIGHT, LOW_RES_WIDTH),
                          generator=torch.Generator().manual_seed(SEED + 2),
                          dtype=torch.float64)
    image_noise = torch.randn((1, 3, LOW_RES_HEIGHT, LOW_RES_WIDTH),
                              generator=torch.Generator().manual_seed(SEED + 3),
                              dtype=torch.float64)
    # The pipeline draws the low_res_scheduler noise first; latents are given.
    upscale_module.randn_tensor = lambda *args, **kwargs: image_noise.clone()
    cases = []
    for case_pos, (name, scheduler_name, (width, height), noise_level) in \
            enumerate(CASES):
        levels = conditioning_fixture.byte_image(
            torch.Generator().manual_seed(SEED + 10 + case_pos), 3, width, height)
        pil_image = Image.fromarray(
            (levels[0].permute(1, 2, 0) * 255).round().to(torch.uint8).numpy())
        pipe = load_pipeline(scheduler_name)
        out = conditioning_fixture.run(
            pipe, {"image": pil_image, "noise_level": noise_level,
                   "latents": latents.clone()},
            None, euler=scheduler_name == "euler", legacy_callback=True,
            steps=STEPS, sized=False, guidance=GUIDANCE)
        out["name"] = name
        out["scheduler"] = type(pipe.scheduler).__name__
        out["noise_level"] = noise_level
        out["low_res_image"] = conditioning_fixture.rounded(
            (levels[0].permute(1, 2, 0) * 255).round().tolist())
        assert len(out["image"]) == LOW_RES_HEIGHT * 4
        assert len(out["image"][0]) == LOW_RES_WIDTH * 4
        cases.append(out)
        print(name, out["timesteps"], len(out["latents_after"]), "steps run")
    payload = {"width": LOW_RES_WIDTH * 4, "height": LOW_RES_HEIGHT * 4,
               "steps": STEPS, "guidance_scale": GUIDANCE,
               "prompt": conditioning_fixture.PROMPT,
               "latents": conditioning_fixture.chw(latents[0]),
               "image_noise": conditioning_fixture.chw(image_noise[0]),
               "cases": cases}
    with open(PIPELINE_IO, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    build_folder()
    make_unet_oracle()
    make_pipeline_oracle()
