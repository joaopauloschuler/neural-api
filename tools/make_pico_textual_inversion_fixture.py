#!/usr/bin/env python3
"""Generate the pico textual inversion fixture for
TStableDiffusionPipeline.LoadTextualInversion, on the tiny_sd15 pipeline
(tools/make_pico_sd_pipeline_fixture.py; hidden size 32).

  tests/fixtures/tiny_sd15_ti/
    pico_cat.safetensors    diffusers learned_embeds: {"<pico-cat>": [2, 32]}
    pico_cat.bin            the same dict through torch.save
    pico_style.pt           A1111 layout: string_to_param {"*": [2, 32]},
                            name "PicoStyle" (mixed case: diffusers matches it
                            case-insensitively but never expands it)
    wrong_width.safetensors {"<wide>": [1, 16]} (hidden size mismatch)
  tests/fixtures/tiny_sd15_ti_io.json
    cases  per prompt, after loading pico_cat.safetensors then pico_style.pt:
           prompt, converted (maybe_convert_prompt), input_ids (77, padded),
           and for the first EMBED_CASES prompts prompt_embeds [77][32]
           (encode_prompt, float64).

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv):
  python tools/make_pico_textual_inversion_fixture.py
"""
import json
import os

import torch
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")
PICO_DIR = os.path.join(FIXTURES, "tiny_sd15")
TI_DIR = os.path.join(FIXTURES, "tiny_sd15_ti")
IO_FILE = os.path.join(FIXTURES, "tiny_sd15_ti_io.json")

SEED = 20261022
HIDDEN = 32
EMBED_CASES = 2
PROMPTS = [
    "a photo of <pico-cat> riding a horse",
    "<pico-cat> and PicoStyle, <pico-cat>",
    "a <PICO-CAT> in picostyle",
    "<pico-cat> then <pico-cat>_1",
    "a <pico-cat>_1 alone",
    "an astronaut on mars",
]


def round_list(values):
    return [round(float(v), 10) for v in values]


def main():
    torch.manual_seed(SEED)
    os.makedirs(TI_DIR, exist_ok=True)
    # Unit-scale rows, as trained concept vectors are, so they move the embeds.
    cat = torch.randn(2, HIDDEN, dtype=torch.float32)
    style = torch.randn(2, HIDDEN, dtype=torch.float32)
    cat_st = os.path.join(TI_DIR, "pico_cat.safetensors")
    style_pt = os.path.join(TI_DIR, "pico_style.pt")
    save_file({"<pico-cat>": cat}, cat_st)
    torch.save({"<pico-cat>": cat.clone()}, os.path.join(TI_DIR, "pico_cat.bin"))
    torch.save({"string_to_token": {"*": 265},
                "string_to_param": {"*": style.clone().requires_grad_(True)},
                "name": "PicoStyle", "step": 100,
                "sd_checkpoint": "0123abcd", "sd_checkpoint_name": "tiny_sd15"},
               style_pt)
    save_file({"<wide>": torch.randn(1, 16)},
              os.path.join(TI_DIR, "wrong_width.safetensors"))

    pipe = StableDiffusionPipeline.from_pretrained(
        PICO_DIR, safety_checker=None, requires_safety_checker=False)
    pipe.to(torch.float64)
    pipe.load_textual_inversion(cat_st)
    pipe.load_textual_inversion(style_pt)
    cases = []
    for pos, prompt in enumerate(PROMPTS):
        converted = pipe.maybe_convert_prompt(prompt, pipe.tokenizer)
        ids = pipe.tokenizer(converted, padding="max_length", max_length=77,
                             truncation=True).input_ids
        case = {"prompt": prompt, "converted": converted, "input_ids": ids}
        if pos < EMBED_CASES:
            with torch.no_grad():
                embeds, _ = pipe.encode_prompt(prompt, "cpu", 1, False)
            case["prompt_embeds"] = [round_list(row) for row in embeds[0]]
        cases.append(case)
    with open(IO_FILE, "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f)
    for case in cases:
        print(repr(case["prompt"]), "->", repr(case["converted"]),
              case["input_ids"][:12])


if __name__ == "__main__":
    main()
