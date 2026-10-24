#!/usr/bin/env python3
"""Generate the Qwen-Image-2.1 img2img (SDEdit) parity oracle (tasklist C2b)
from the pico folder that tools/make_pico_qwenimage21_fixture.py already wrote
to tests/fixtures/tiny_qwenimage21 (nothing is regenerated).

Qwen-Image-2.1 has no img2img pipeline. The schedule truncation and the
noising come from the Qwen-Image v1 QwenImageImg2ImgPipeline (get_timesteps,
scheduler.scale_noise) on the 2.1 scheduler set the way the 2.1 pipeline sets
it (sigmas = linspace(1, 1/N, N), mu from the target token count). The
end-to-end run is the 2.1 pipeline with sigmas = linspace(1, 1/N, N)[t_start:]
and latents = the noised latents: the time shift is elementwise and the
shift_terminal stretch reads only the last sigma, which the tail keeps, so the
tail schedule equals the full schedule from t_start (asserted below).

Outputs (tests/fixtures/):
  qwenimage21_img2img_oracle.json   t_start / steps run / sigma[t_start] per
      (N, strength), the cases with no step left, scale_noise on 6 values,
      PIL Lanczos resizes (float 'F' mode per channel, and two RGBA uint8
      images that PIL resizes with premultiplied alpha, one with low and zero
      alpha)
  tiny_qwenimage21_img2img_io.json  64x64 pico img2img: posterior-mean image
      latents, the noise, the noised latents and the step latents (the prompt
      embeds are those of tiny_qwenimage21_pipeline_64_io.json)
  tiny_qwenimage21_inpaint_io.json  the same run as blended inpainting (the
      Qwen-Image v1 QwenImageInpaintPipeline's mask processor, mask latents
      and per-step blend, on the 2.1 pipeline through callback_on_step_end):
      the mask image, the mask latents and the step latents after each blend

The init image is not stored: every value is ((i * 7919 + 13) mod 2049) / 1024
- 1 with i the row-major (C, H, W) index (the C2 encoder fixture's formula).

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv, torch CPU build):
  ( ulimit -v 3145728; python3 tools/make_pico_qwenimage21_img2img_fixture.py )
  --inpaint-only writes only tiny_qwenimage21_inpaint_io.json.
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pico_qwenimage21_fixture import (PICO_DIR, SCHEDULER_CONFIG, TE_DROP_IDX,  # noqa: E402
                                           AutoencoderKLQwenImage21,
                                           FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline,
                                           QwenImage21Transformer2DModel, TemplateOnlyProcessor,
                                           calculate_shift, install_float64_shims, max_abs_diff,
                                           rebuild_rope_tables, tensor_json, write_json)
from make_pico_qwenimage21_vae_encoder_fixture import formula_input  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.pipelines.qwenimage.pipeline_qwenimage_img2img import QwenImageImg2ImgPipeline  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures")
TRUNCATION_CASES = [(18, 0.6), (10, 0.3), (3, 0.5), (4, 1.0), (18, 0.05), (20, 0.75), (7, 0.999),
                    (5, 0.2), (3, 0.3), (5, 0.6), (2, 1.0), (50, 0.01), (6, 0.5)]
ZERO_STEP_CASES = [(18, 0.0), (10, 1e-20)]
TRUNCATION_SEQ_LEN = 16                 # a 64x64 pico image
PIPE_SIZE, PIPE_STEPS, PIPE_STRENGTH = 64, 5, 0.6
NOISE_SEED = 31
# (in_w, in_h, out_w, out_h): shrink, enlarge, mixed, same size
RESIZE_CASES = [(13, 9, 7, 5), (5, 4, 11, 8), (16, 6, 6, 10), (9, 7, 9, 7)]
RESIZE_CHANNELS = 3


class StubPipeline:
    """Just what QwenImageImg2ImgPipeline.get_timesteps reads."""

    def __init__(self, scheduler):
        self.scheduler = scheduler


def schedule_2_1(num_steps, image_seq_len):
    """The 2.1 pipeline's scheduler setup: linspace sigmas, mu from the token count."""
    mu = calculate_shift(image_seq_len, SCHEDULER_CONFIG["base_image_seq_len"],
                         SCHEDULER_CONFIG["max_image_seq_len"], SCHEDULER_CONFIG["base_shift"],
                         SCHEDULER_CONFIG["max_shift"])
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / num_steps, num_steps), mu=mu)
    return scheduler, mu


def shifted_sigmas_f64(num_steps, mu):
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    shifted = np.exp(mu) / (np.exp(mu) + (1.0 / sigmas - 1.0))
    one_minus = 1.0 - shifted
    return 1.0 - one_minus / (one_minus[-1] / (1.0 - SCHEDULER_CONFIG["shift_terminal"]))


def v1_truncate(num_steps, strength, image_seq_len):
    """(t_start, steps run, scheduler, mu) exactly as QwenImageImg2ImgPipeline computes them."""
    scheduler, mu = schedule_2_1(num_steps, image_seq_len)
    _, steps_run = QwenImageImg2ImgPipeline.get_timesteps(StubPipeline(scheduler), num_steps, strength, "cpu")
    return num_steps - steps_run, steps_run, scheduler, mu


def make_truncation_oracle():
    cases = []
    for num_steps, strength in TRUNCATION_CASES:
        t_start, steps_run, scheduler, mu = v1_truncate(num_steps, strength, TRUNCATION_SEQ_LEN)
        assert steps_run >= 1, (num_steps, strength)
        cases.append({"num_steps": num_steps, "strength": strength, "t_start": t_start,
                      "steps_run": steps_run, "mu": mu,
                      "sigma_start_f64": float(shifted_sigmas_f64(num_steps, mu)[t_start]),
                      "sigma_start_native": float(scheduler.sigmas[t_start])})
    zero_cases = []
    for num_steps, strength in ZERO_STEP_CASES:
        _, steps_run, _, _ = v1_truncate(num_steps, strength, TRUNCATION_SEQ_LEN)
        assert steps_run == 0, (num_steps, strength, steps_run)
        zero_cases.append({"num_steps": num_steps, "strength": strength})

    # scale_noise right after set_begin_index, as the img2img pipeline calls it.
    num_steps, strength = 18, 0.6
    t_start, _, scheduler, mu = v1_truncate(num_steps, strength, TRUNCATION_SEQ_LEN)
    gen = torch.Generator().manual_seed(20261004)
    sample = torch.randn(6, generator=gen, dtype=torch.float64)
    noise = torch.randn(6, generator=gen, dtype=torch.float64)
    native = scheduler.scale_noise(sample[None], scheduler.timesteps[t_start:t_start + 1], noise[None])[0]
    sigma = shifted_sigmas_f64(num_steps, mu)[t_start]
    expected = sigma * noise + (1.0 - sigma) * sample
    assert max_abs_diff(native, expected) < 1e-6
    return {"config": SCHEDULER_CONFIG, "image_seq_len": TRUNCATION_SEQ_LEN, "cases": cases,
            "zero_step_cases": zero_cases,
            "scale_noise": {"num_steps": num_steps, "strength": strength, "t_start": t_start,
                            "sample": sample.tolist(), "noise": noise.tolist(),
                            "noised_f64": expected.tolist(),
                            "native_maxabs_diff": max_abs_diff(native, expected)}}


def resize_formula(width, height, channels):
    """(H, W, C) float32 values in [0, 255] from the fixture formula."""
    index = np.arange(channels * height * width, dtype=np.int64).reshape(channels, height, width)
    values = ((index * 7919 + 13) % 2049).astype(np.float64) / 1024 * 127.5
    return values.transpose(1, 2, 0).astype(np.float32)


def make_resize_oracle():
    float_cases = []
    for in_w, in_h, out_w, out_h in RESIZE_CASES:
        source = resize_formula(in_w, in_h, RESIZE_CHANNELS)
        out = np.stack([np.asarray(Image.fromarray(source[:, :, c], mode="F").resize(
            (out_w, out_h), Image.LANCZOS)) for c in range(RESIZE_CHANNELS)], axis=2)
        float_cases.append({"in_w": in_w, "in_h": in_h, "out_w": out_w, "out_h": out_h,
                            "output_hwc": tensor_json(torch.from_numpy(out.astype(np.float64)))})
    # RGBA uint8, PIL resizes it premultiplied ("RGBa"): alpha 128..255, then
    # alpha 0..255 with every third pixel fully transparent.
    rgba_cases = []
    in_w, in_h, out_w, out_h = 12, 10, 7, 9
    index = np.arange(in_h * in_w * 4, dtype=np.int64).reshape(in_h, in_w, 4)
    for low_alpha in (False, True):
        rgba = ((index * 7919 + 13) % 256).astype(np.uint8)
        if low_alpha:
            alpha = (index[:, :, 3] * 37 + 11) % 256
            alpha[(index[:, :, 3] // 4) % 3 == 0] = 0
        else:
            alpha = 128 + (index[:, :, 3] * 31 + 5) % 128
        rgba[:, :, 3] = alpha.astype(np.uint8)
        out = np.asarray(Image.fromarray(rgba, mode="RGBA").resize((out_w, out_h), Image.LANCZOS))
        rgba_cases.append({"in_w": in_w, "in_h": in_h, "out_w": out_w, "out_h": out_h,
                           "low_alpha": low_alpha,
                           "input_hwc": tensor_json(torch.from_numpy(rgba.astype(np.float64))),
                           "output_hwc": tensor_json(torch.from_numpy(out.astype(np.float64)))})
    return {"float_cases": float_cases, "rgba_uint8_cases": rgba_cases,
            "note": "float_cases: each channel resized as a PIL mode 'F' image with Image.LANCZOS; "
                    "inputs are ((i * 7919 + 13) mod 2049) / 1024 * 127.5, i the row-major (C, H, W) "
                    "index. rgba_uint8_cases: PIL RGBA resize (premultiplied alpha, fixed-point "
                    "coefficients, uint8 per pass)."}


def make_pipeline_oracle():
    vae_dir = os.path.join(PICO_DIR, "vae")
    vae = AutoencoderKLQwenImage21.from_pretrained(vae_dir, torch_dtype=torch.float64).eval()
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        os.path.join(PICO_DIR, "transformer"), torch_dtype=torch.float64).eval()
    rebuild_rope_tables(transformer.pos_embed)
    with open(os.path.join(FIXTURES, "tiny_qwenimage21_pipeline_64_io.json")) as f:
        embeds_json = json.load(f)["prompt_embeds"]
    prompt_embeds = torch.tensor(embeds_json["data"], dtype=torch.float64).view(1, *embeds_json["shape"])
    z_dim = vae.config.z_dim
    latent_side = PIPE_SIZE // 16
    tokens = latent_side * latent_side

    with torch.no_grad():
        image = formula_input(vae.config.in_channels, PIPE_SIZE, PIPE_SIZE)
        posterior_mean = vae.encode(image).latent_dist.mean
        mean = torch.tensor(vae.config.latents_mean, dtype=torch.float64).view(1, z_dim, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, dtype=torch.float64).view(1, z_dim, 1, 1, 1)
        image_latents = ((posterior_mean - mean) / std)[0, :, 0].reshape(z_dim, tokens).T.contiguous()

        t_start, steps_run, scheduler, mu = v1_truncate(PIPE_STEPS, PIPE_STRENGTH, tokens)
        gen = torch.Generator().manual_seed(NOISE_SEED)
        noise = torch.randn(tokens, z_dim, generator=gen, dtype=torch.float64)
        noised = scheduler.scale_noise(image_latents[None], scheduler.timesteps[t_start:t_start + 1],
                                       noise[None])
        full_sigmas = [float(s) for s in scheduler.sigmas]

        pipe = QwenImage21Pipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG),
                                   vae=vae, text_encoder=None, processor=TemplateOnlyProcessor(TE_DROP_IDX),
                                   transformer=transformer)
        step_latents, timesteps = [], []

        def on_step(pipeline, index, t, callback_kwargs):
            step_latents.append(callback_kwargs["latents"].clone())
            timesteps.append(float(t))
            return callback_kwargs

        tail = np.linspace(1.0, 1.0 / PIPE_STEPS, PIPE_STEPS)[t_start:]
        pipe(prompt_embeds=prompt_embeds, height=PIPE_SIZE, width=PIPE_SIZE,
             num_inference_steps=steps_run, sigmas=list(tail), latents=noised.clone(),
             output_type="latent", callback_on_step_end=on_step)
        tail_sigmas = [float(s) for s in pipe.scheduler.sigmas]
    assert len(step_latents) == steps_run
    tail_vs_full = max(abs(a - b) for a, b in zip(tail_sigmas, full_sigmas[t_start:]))
    assert len(tail_sigmas) == len(full_sigmas) - t_start and tail_vs_full < 1e-6, tail_vs_full
    write_json("tiny_qwenimage21_img2img_io.json", {
        "height": PIPE_SIZE, "width": PIPE_SIZE, "num_inference_steps": PIPE_STEPS,
        "strength": PIPE_STRENGTH, "t_start": t_start, "steps_run": steps_run, "mu": mu,
        "tail_vs_full_sigma_maxabs": tail_vs_full,
        "timesteps": timesteps,
        "image_latents": tensor_json(image_latents),
        "noise": tensor_json(noise),
        "noised_latents": tensor_json(noised[0]),
        "step_latents": [tensor_json(latents[0]) for latents in step_latents],
        "note": "prompt_embeds: tiny_qwenimage21_pipeline_64_io.json. Init image: "
                "((i * 7919 + 13) mod 2049) / 1024 - 1, i = row-major (C, H, W) index, in [-1, 1]. "
                "image_latents = (posterior mean - latents_mean) / latents_std, (tokens, C) with "
                "token = h * latent_w + w. noised_latents = scheduler.scale_noise at t_start (float32 "
                "sigma). The run is QwenImage21Pipeline(sigmas=linspace(1, 1/N, N)[t_start:], "
                "latents=noised_latents, output_type='latent'); step_latents[i] = latents after run "
                "step i. The VAE decode is covered by the text-to-image pipeline fixtures.",
    })


def inpaint_mask_image(size):
    """RGB uint8 (H, W, 3). Only the top-left pixel of each 16x16 cell reaches the
    latent grid (nearest), so the other pixels are a formula; the corners hold
    white/black, the binarize edge (gray 128 -> 1, 127 -> 0) and colours whose
    PIL 'L' luma is either side of it."""
    index = np.arange(size * size * 3, dtype=np.int64).reshape(size, size, 3)
    pixels = ((index * 7919 + 13) % 256).astype(np.uint8)
    corners = [(255, 255, 255), (0, 0, 0), (128, 128, 128), (127, 127, 127),
               (0, 255, 0), (255, 0, 0), (0, 0, 255), (200, 200, 200),
               (0, 0, 0), (255, 255, 255), (30, 30, 30), (255, 255, 0),
               (10, 10, 10), (0, 0, 0), (255, 255, 255), (100, 120, 140)]
    cells = size // 16
    for cell, rgb in enumerate(corners[:cells * cells]):
        pixels[(cell // cells) * 16, (cell % cells) * 16] = rgb
    return pixels


def make_inpaint_oracle():
    vae_dir = os.path.join(PICO_DIR, "vae")
    vae = AutoencoderKLQwenImage21.from_pretrained(vae_dir, torch_dtype=torch.float64).eval()
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        os.path.join(PICO_DIR, "transformer"), torch_dtype=torch.float64).eval()
    rebuild_rope_tables(transformer.pos_embed)
    with open(os.path.join(FIXTURES, "tiny_qwenimage21_pipeline_64_io.json")) as f:
        embeds_json = json.load(f)["prompt_embeds"]
    prompt_embeds = torch.tensor(embeds_json["data"], dtype=torch.float64).view(1, *embeds_json["shape"])
    z_dim = vae.config.z_dim
    latent_side = PIPE_SIZE // 16
    tokens = latent_side * latent_side

    # QwenImageInpaintPipeline's mask processor and prepare_mask_latents, with
    # the 2.1 VAE factor (16) and the 2.1 packing (a plain spatial flatten).
    mask_pixels = inpaint_mask_image(PIPE_SIZE)
    mask_processor = VaeImageProcessor(vae_scale_factor=16 * 2, vae_latent_channels=z_dim,
                                       do_normalize=False, do_binarize=True, do_convert_grayscale=True)
    mask_condition = mask_processor.preprocess(Image.fromarray(mask_pixels, mode="RGB"),
                                               height=PIPE_SIZE, width=PIPE_SIZE).to(torch.float64)
    mask = torch.nn.functional.interpolate(mask_condition, size=(latent_side, latent_side))
    mask = mask.repeat(1, z_dim, 1, 1)
    mask = QwenImage21Pipeline._pack_latents(mask, 1, z_dim, latent_side, latent_side)[0]
    hole_tokens = [int(v) for v in mask[:, 0].tolist()]
    assert 0 < sum(hole_tokens) < tokens, hole_tokens

    with torch.no_grad():
        image = formula_input(vae.config.in_channels, PIPE_SIZE, PIPE_SIZE)
        posterior_mean = vae.encode(image).latent_dist.mean
        mean = torch.tensor(vae.config.latents_mean, dtype=torch.float64).view(1, z_dim, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, dtype=torch.float64).view(1, z_dim, 1, 1, 1)
        image_latents = ((posterior_mean - mean) / std)[0, :, 0].reshape(z_dim, tokens).T.contiguous()

        t_start, steps_run, scheduler, mu = v1_truncate(PIPE_STEPS, PIPE_STRENGTH, tokens)
        gen = torch.Generator().manual_seed(NOISE_SEED)
        noise = torch.randn(tokens, z_dim, generator=gen, dtype=torch.float64)
        noised = scheduler.scale_noise(image_latents[None], scheduler.timesteps[t_start:t_start + 1],
                                       noise[None])

        pipe = QwenImage21Pipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG),
                                   vae=vae, text_encoder=None, processor=TemplateOnlyProcessor(TE_DROP_IDX),
                                   transformer=transformer)
        step_latents, timesteps = [], []

        # The inpaint pipeline's lines after scheduler.step: scale_noise reads
        # the scheduler's step_index (= i + 1 here, as there).
        def on_step(pipeline, index, t, callback_kwargs):
            latents = callback_kwargs["latents"]
            init_latents_proper = image_latents[None]
            if index < steps_run - 1:
                noise_timestep = pipeline.scheduler.timesteps[index + 1]
                init_latents_proper = pipeline.scheduler.scale_noise(
                    init_latents_proper, torch.tensor([noise_timestep]), noise[None])
            latents = (1 - mask[None]) * init_latents_proper + mask[None] * latents
            step_latents.append(latents.clone())
            timesteps.append(float(t))
            return {"latents": latents}

        tail = np.linspace(1.0, 1.0 / PIPE_STEPS, PIPE_STEPS)[t_start:]
        pipe(prompt_embeds=prompt_embeds, height=PIPE_SIZE, width=PIPE_SIZE,
             num_inference_steps=steps_run, sigmas=list(tail), latents=noised.clone(),
             output_type="latent", callback_on_step_end=on_step)
    assert len(step_latents) == steps_run
    keep = mask[:, :1] == 0
    assert max_abs_diff(step_latents[-1][0] * keep, image_latents * keep) == 0
    write_json("tiny_qwenimage21_inpaint_io.json", {
        "height": PIPE_SIZE, "width": PIPE_SIZE, "num_inference_steps": PIPE_STEPS,
        "strength": PIPE_STRENGTH, "t_start": t_start, "steps_run": steps_run, "mu": mu,
        "timesteps": timesteps,
        "mask_hwc": tensor_json(torch.from_numpy(mask_pixels.astype(np.float64))),
        "hole_tokens": hole_tokens,
        "mask_latents": tensor_json(mask),
        "noise": tensor_json(noise),
        "step_latents": [tensor_json(latents[0]) for latents in step_latents],
        "note": "As tiny_qwenimage21_img2img_io.json (same init image, noise, strength and prompt "
                "embeds) with a mask: mask_hwc is RGB 0..255 (white = regenerate), mask_latents "
                "its binarized, nearest-resized, packed (tokens, C) mask. After every step the "
                "latents become (1 - mask) * known + mask * latents, known = scale_noise(image "
                "latents, timesteps[i + 1], noise), the clean image latents after the last step; "
                "step_latents[i] = latents after that blend.",
    })


def main():
    torch.set_num_threads(1)
    install_float64_shims(True)
    if "--inpaint-only" in sys.argv[1:]:
        make_inpaint_oracle()
        return
    payload = make_truncation_oracle()
    payload["resize"] = make_resize_oracle()
    write_json("qwenimage21_img2img_oracle.json", payload)
    make_pipeline_oracle()
    make_inpaint_oracle()


if __name__ == "__main__":
    main()
