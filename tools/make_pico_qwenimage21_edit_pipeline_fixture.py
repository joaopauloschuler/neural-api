#!/usr/bin/env python3
"""Generate the Qwen-Image-2.1 EDIT pipeline oracle (tasklist C3b) from the pico
folder that tools/make_pico_qwenimage21_fixture.py already wrote to
tests/fixtures/tiny_qwenimage21 (nothing is regenerated).

Output: tests/fixtures/tiny_qwenimage21_edit_pipeline_io.json, one case per
CASES entry: diffusers QwenImage21Pipeline(image=[...], output_resolution=64)
end to end on the pico text encoder (with its vision tower), transformer and
VAE in float64, fixed initial latents, use_kv_cache on:
  token_ids          the template ids with ONE <|image_pad|> per image (the
                     processor shim expands them, as Qwen3VLProcessor does)
  width, height      the target size (calculate_dimensions of the LAST image)
  prompt_embeds      the text-encoder rows after drop_idx (image slots kept)
  image_pad_mask     1 at the image-slot rows of prompt_embeds
  condition_latents  per image, (tokens, z) normalised posterior mean
  initial_latents, step_latents, and for two_images the image (output_type
                     'pt', (C, H, W) in [0, 1])
Floats are rounded to 8 significant digits (the Pascal side is float32).

The pico has no tokenizer: the shim processor maps the template to fixed ids
(system rows, "<imageN>" stand-ins, vision start/pad/end, prompt, assistant
tail) and runs Qwen2VLImageProcessorPil on the images the pipeline hands it
(already composited over white). The images are already at the size
calculate_dimensions gives them at 64x64, so PIL's resize is the identity,
except in resized_image: 100x60 -> 96x64, where PIL's 8-bit Lanczos differs
from the Pascal float resize (an accepted divergence; looser tolerance). That
case also records float_resize_condition_latents: the VAE encode of the image
resized as PIL float ('F') images, which the Pascal resize equals.

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv, torch CPU build):
  ( ulimit -v 2097152; python3 tools/make_pico_qwenimage21_edit_pipeline_fixture.py )
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions  # noqa: E402
from make_pico_qwenimage21_fixture import (PICO_DIR, PREPROCESSOR_CONFIG, SCHEDULER_CONFIG,  # noqa: E402
                                           AutoencoderKLQwenImage21,
                                           FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline,
                                           QwenImage21Transformer2DModel,
                                           Qwen3VLForConditionalGeneration, install_float64_shims,
                                           rebuild_rope_tables, tensor_json, vision_formula_image,
                                           write_json)

DROP_IDX = 14
IMAGE_PAD, VISION_START, VISION_END = 290, 292, 293
SYSTEM_IDS = [(37 * i + 11) % 280 for i in range(DROP_IDX)]
USER_IDS = [5, 17, 3]
PROMPT_IDS = [60, 61, 62, 63, 64]
TAIL_IDS = [7, 8, 9, 10, 11]
OUTPUT_RESOLUTION = 64
STEPS = 3
LATENT_SEED = 23
# (width, height, mode) per image. Both sizes are calculate_dimensions(64*64,
# aspect) of themselves; the two-image case takes its 64x64 target from the
# LAST image (the first alone would give 96x32).
CASES = [("one_image", [(96, 32, "RGBA")]),
         ("two_images", [(96, 32, "RGBA"), (64, 64, "RGB")]),
         ("resized_image", [(100, 60, "RGB")])]
IMAGE_CASE = "two_images"
SIGNIFICANT_DIGITS = 8


def float_resize_latents(vae, image, width, height):
    """Normalised posterior mean of image resized per channel as PIL 'F' images (no 8-bit rounding)."""
    rgba = np.asarray(image.convert("RGBA")).astype(np.float32)
    resized = np.stack([np.asarray(Image.fromarray(rgba[:, :, c], "F").resize((width, height), Image.LANCZOS))
                        for c in range(4)], axis=2)
    pixels = torch.from_numpy(np.clip(resized, 0, 255).astype(np.float64) / 127.5 - 1)
    with torch.no_grad():
        mean = vae.encode(pixels.permute(2, 0, 1)[None, :, None]).latent_dist.mean
    z_dim = vae.config.z_dim
    latents_mean = torch.tensor(vae.config.latents_mean, dtype=torch.float64).view(1, z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, dtype=torch.float64).view(1, z_dim, 1, 1, 1)
    return ((mean - latents_mean) / latents_std)[0, :, 0].reshape(z_dim, -1).T.contiguous()


def rounded_json(x):
    payload = tensor_json(x)
    payload["data"] = [float(f"{value:.{SIGNIFICANT_DIGITS}g}") for value in payload["data"]]
    return payload


def template_ids(image_count):
    ids = SYSTEM_IDS + USER_IDS
    for index in range(image_count):
        ids += [41 + index] if index == 0 else [23, 41 + index]
        ids += [VISION_START, IMAGE_PAD, VISION_END]
    return ids + PROMPT_IDS + TAIL_IDS


class EditProcessorShim:
    """What QwenImage21Pipeline reads from its processor, with fixed ids for the pico vocabulary."""

    class _Tokenizer:
        def encode(self, text):
            assert text == "<|image_pad|>", text
            return [IMAGE_PAD]

    def __init__(self):
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        self.tokenizer = self._Tokenizer()
        self.image_processor = Qwen2VLImageProcessorPil(
            **{k: v for k, v in PREPROCESSOR_CONFIG.items() if k != "image_processor_type"})

    def apply_chat_template(self, messages, tokenize=True, return_dict=False):
        return [[0] * DROP_IDX]

    def __call__(self, text=None, images=None, **kwargs):
        assert len(text) == 1
        image_count = text[0].count("<|image_pad|>")
        assert image_count == len(images), (image_count, len(images))
        vision = self.image_processor(images=images, return_tensors="pt")
        merge = PREPROCESSOR_CONFIG["merge_size"]
        ids = []
        image_pos = 0
        for token in template_ids(image_count):
            if token == IMAGE_PAD:
                _, grid_h, grid_w = vision["image_grid_thw"][image_pos].tolist()
                ids += [IMAGE_PAD] * (grid_h * grid_w // (merge * merge))
                image_pos += 1
            else:
                ids.append(token)
        input_ids = torch.tensor([ids])
        return BatchFeature({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
                             "pixel_values": vision["pixel_values"].to(torch.float64),
                             "image_grid_thw": vision["image_grid_thw"],
                             "mm_token_type_ids": (input_ids == IMAGE_PAD).to(torch.int64)})


def main():
    torch.set_num_threads(1)
    install_float64_shims(True)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(PICO_DIR, "text_encoder"), dtype=torch.float64).eval()
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        os.path.join(PICO_DIR, "transformer"), torch_dtype=torch.float64).eval()
    rebuild_rope_tables(transformer.pos_embed)
    vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(PICO_DIR, "vae"),
                                                   torch_dtype=torch.float64).eval()
    pipe = QwenImage21Pipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG),
                               vae=vae, text_encoder=text_encoder, processor=EditProcessorShim(),
                               transformer=transformer)
    assert pipe._drop_idx == DROP_IDX
    z_dim = vae.config.z_dim
    cases = []
    for label, specs in CASES:
        images = [Image.fromarray(vision_formula_image(w, h, len(mode)), mode) for w, h, mode in specs]
        width, height, _ = calculate_dimensions(OUTPUT_RESOLUTION * OUTPUT_RESOLUTION,
                                                specs[-1][0] / specs[-1][1])
        latent_tokens = (width // 16) * (height // 16)
        gen = torch.Generator().manual_seed(LATENT_SEED)
        initial = torch.randn(1, latent_tokens, z_dim, generator=gen, dtype=torch.float64)
        step_latents, condition_latents, captured = [], [], {}

        def on_step(pipeline, index, t, callback_kwargs):
            step_latents.append(callback_kwargs["latents"].clone())
            return callback_kwargs

        original_encode_vae = pipe._encode_vae_image
        original_encode_prompt = pipe.encode_prompt

        def recording_encode_vae(image, generator):
            latents = original_encode_vae(image, generator)
            condition_latents.append(latents[0, :, 0].reshape(z_dim, -1).T.contiguous())
            return latents

        def recording_encode_prompt(*args, **kwargs):
            result = original_encode_prompt(*args, **kwargs)
            captured["prompt_embeds"], _, captured["image_pad_mask"] = result
            return result

        pipe._encode_vae_image = recording_encode_vae
        pipe.encode_prompt = recording_encode_prompt
        try:
            with torch.no_grad():
                out = pipe(prompt="edit", image=images, num_inference_steps=STEPS,
                           output_resolution=OUTPUT_RESOLUTION, latents=initial.clone(),
                           output_type="pt", callback_on_step_end=on_step).images
        finally:
            del pipe._encode_vae_image
            del pipe.encode_prompt
        assert out.shape[-2:] == (height, width), out.shape
        assert len(step_latents) == STEPS and len(condition_latents) == len(images)
        case = {
            "label": label,
            "resized": any((w, h) != calculate_dimensions(OUTPUT_RESOLUTION ** 2, w / h)[:2]
                           for w, h, _ in specs),
            "images": [{"width": w, "height": h, "channels": len(mode)} for w, h, mode in specs],
            "token_ids": template_ids(len(images)),
            "drop_idx": DROP_IDX,
            "width": width, "height": height,
            "prompt_embeds": rounded_json(captured["prompt_embeds"][0]),
            "image_pad_mask": captured["image_pad_mask"][0].to(torch.int64).tolist(),
            "condition_latents": [rounded_json(latents) for latents in condition_latents],
            "initial_latents": rounded_json(initial[0]),
            "step_latents": [rounded_json(latents[0]) for latents in step_latents],
        }
        if case["resized"]:
            case["float_resize_condition_latents"] = [
                rounded_json(float_resize_latents(vae, image, *calculate_dimensions(
                    OUTPUT_RESOLUTION ** 2, image.size[0] / image.size[1])[:2]))
                for image in images]
        if label == IMAGE_CASE:
            case["image"] = rounded_json(out[0])
        cases.append(case)
        print(f"{label}: target {width}x{height}, {len(captured['image_pad_mask'][0])} prompt rows")
    write_json("tiny_qwenimage21_edit_pipeline_io.json", {
        "output_resolution": OUTPUT_RESOLUTION, "num_inference_steps": STEPS,
        "image_token_id": IMAGE_PAD, "cases": cases,
        "note": "QwenImage21Pipeline(prompt, image=[...], output_resolution=64, latents=initial) on "
                "the pico folder in float64, use_kv_cache on, true_cfg_scale 1. Images: uint8 "
                "((i * 7919 + 13) mod 256), i = row-major (y, x, channel) index. token_ids: one "
                "<|image_pad|> per image (expanded by the processor shim). Latents are packed "
                "(tokens, channels), token = h * latent_w + w. step_latents[i] = latents after "
                "scheduler step i; image (two_images only) = output_type 'pt' (C, H, W) in [0, 1]. "
                "resized: PIL resized a condition image. Floats: 8 significant digits.",
    })


if __name__ == "__main__":
    main()
