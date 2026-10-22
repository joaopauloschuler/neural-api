#!/usr/bin/env python3
"""Generate the pico clip skip oracle for TStableDiffusionPipeline.ClipSkip on
the tiny_sd15 pipeline (tools/make_pico_sd_pipeline_fixture.py, a 2-layer
CLIP text encoder).

  tests/fixtures/tiny_sd15_clip_skip_io.json
    prompt, negative_prompt and per case: clip_skip, prompt_embeds [77][32]
    from diffusers encode_prompt(clip_skip=...) (float64):
    hidden_states[-(clip_skip + 1)] through final_layer_norm. encode_prompt
    skips nothing on the negative prompt; TStableDiffusionPipeline skips on
    both (as A1111), so negative_prompt_embeds is that same formula.

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv):
  python tools/make_pico_sd_clip_skip_fixture.py
"""
import json
import os
import types

import torch
from diffusers import StableDiffusionPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")
PICO_DIR = os.path.join(FIXTURES, "tiny_sd15")
IO_FILE = os.path.join(FIXTURES, "tiny_sd15_clip_skip_io.json")

PROMPT = "a photo of an astronaut riding a horse on mars"
NEGATIVE_PROMPT = "blurry, low quality"
CLIP_SKIPS = [1]


def skipped(pipe, prompt, clip_skip):
    ids = pipe.tokenizer(prompt, padding="max_length", max_length=77,
                         truncation=True, return_tensors="pt").input_ids
    hidden = pipe.text_encoder(ids, output_hidden_states=True).hidden_states
    return final_layer_norm(pipe)(hidden[-(clip_skip + 1)])


def final_layer_norm(pipe):
    encoder = pipe.text_encoder
    return getattr(encoder, "text_model", encoder).final_layer_norm


def main():
    pipe = StableDiffusionPipeline.from_pretrained(PICO_DIR, torch_dtype=torch.float64,
                                                   safety_checker=None)
    if not hasattr(pipe.text_encoder, "text_model"):
        # transformers 5 flattened CLIPTextModel; diffusers' clip_skip path
        # still reads text_encoder.text_model.final_layer_norm.
        pipe.text_encoder.__dict__["text_model"] = types.SimpleNamespace(
            final_layer_norm=pipe.text_encoder.final_layer_norm)
    cases = []
    with torch.no_grad():
        plain, _ = pipe.encode_prompt(PROMPT, "cpu", 1, False, None)
        for clip_skip in CLIP_SKIPS:
            embeds, _ = pipe.encode_prompt(PROMPT, "cpu", 1, False, None,
                                           clip_skip=clip_skip)
            assert (embeds - plain).abs().max() > 1e-3, "clip skip changed nothing"
            assert (skipped(pipe, PROMPT, clip_skip) - embeds).abs().max() < 1e-12
            negative = skipped(pipe, NEGATIVE_PROMPT, clip_skip)
            cases.append({"clip_skip": clip_skip,
                          "prompt_embeds": embeds[0].tolist(),
                          "negative_prompt_embeds": negative[0].tolist()})
    with open(IO_FILE, "w") as f:
        json.dump({"prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT,
                   "num_hidden_layers": pipe.text_encoder.config.num_hidden_layers,
                   "cases": cases}, f)


if __name__ == "__main__":
    main()
