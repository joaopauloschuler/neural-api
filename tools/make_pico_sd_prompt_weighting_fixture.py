#!/usr/bin/env python3
"""Generate the pico long-prompt / prompt-weighting oracle for
TStableDiffusionPipeline (TokenizePromptChunks, EncodePrompt with
PromptWeighting), on the tiny_sd15 pipeline (tools/make_pico_sd_pipeline_fixture.py)
and its textual inversion files (tools/make_pico_textual_inversion_fixture.py).

  tests/fixtures/tiny_sd15_lpw_io.json
    cases  per case: prompt, negative_prompt, weighting, inversion (load
           tiny_sd15_ti/pico_cat.safetensors first), chunk_count, chunk_ids /
           chunk_weights (flat, chunk_count * 77, as the text encoder sees them),
           the same for the negative prompt, and when "embeds" is set
           prompt_embeds (and negative_prompt_embeds of a non-empty negative prompt)
           [chunk_count * 77][32] (float64).

The diffusers community pipeline lpw_stable_diffusion is not installed in the
venv, so parse_prompt_attention, get_prompts_with_weights,
pad_tokens_and_weights, get_unweighted_text_embeddings and
get_weighted_text_embeddings (max_embeddings_multiples=3,
no_boseos_middle=False, its defaults) are reimplemented below as that file
writes them. The tiny_sd15 tokenizer pads with <|endoftext|>, so lpw's single
trailing eos and the per-chunk [bos] ids [eos] pad layout are the same ids.

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv):
  python tools/make_pico_sd_prompt_weighting_fixture.py
"""
import json
import os
import re

import torch
from diffusers import StableDiffusionPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")
PICO_DIR = os.path.join(FIXTURES, "tiny_sd15")
TI_FILE = os.path.join(FIXTURES, "tiny_sd15_ti", "pico_cat.safetensors")
IO_FILE = os.path.join(FIXTURES, "tiny_sd15_lpw_io.json")

MAX_EMBEDDINGS_MULTIPLES = 3
TAIL = "high quality, detailed, sharp focus, trending on artstation, "
CASES = [
    {"name": "long weighted",
     "prompt": "a (photo:1.3) of an ((astronaut)) riding a [horse] on mars, " +
               TAIL * 3 + "\\(oil on canvas\\), (red fox [in the snow]:0.8), " + TAIL * 4,
     "negative_prompt": "(blurry:1.2), [ugly], low quality",
     "weighting": True, "inversion": False, "embeds": True},
    {"name": "inversion weighted",
     "prompt": "a (<pico-cat>:1.4) on mars, " + TAIL * 6 + "[<pico-cat>]",
     "negative_prompt": "",
     "weighting": True, "inversion": True, "embeds": True},
    {"name": "cut at three chunks",
     "prompt": TAIL * 20 + "(astronaut:1.2)",
     "negative_prompt": "[the cat's hat]",
     "weighting": True, "inversion": False, "embeds": False},
    {"name": "parsing off",
     "prompt": "a (photo:1.3) of [mars], " + TAIL,
     "negative_prompt": "(blurry), " + TAIL * 7,
     "weighting": False, "inversion": False, "embeds": False},
]

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    res = []
    round_brackets = []
    square_brackets = []
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)
        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])
    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)
    if len(res) == 0:
        res = [["", 1.0]]
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res


def get_prompts_with_weights(pipe, prompt, max_length):
    tokens = []
    weights = []
    for text in prompt:
        text_token = []
        text_weight = []
        for word, weight in parse_prompt_attention(text):
            token = pipe.tokenizer(word).input_ids[1:-1]
            text_token += token
            text_weight += [weight] * len(token)
            if len(text_token) > max_length:
                break
        if len(text_token) > max_length:
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad,
                           chunk_length=77):
    """no_boseos_middle=False branch."""
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [pad] * (max_length - 1 - len(tokens[i]) - 1) + [eos]
        w = []
        if len(weights[i]) == 0:
            w = [1.0] * weights_length
        else:
            for j in range(max_embeddings_multiples):
                w.append(1.0)
                w += weights[i][j * (chunk_length - 2):
                                min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                w.append(1.0)
            w += [1.0] * (weights_length - len(w))
        weights[i] = w[:]
    return tokens, weights


def chunk_inputs(text_input, chunk_length):
    multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    chunks = []
    for i in range(multiples):
        chunk = text_input[:, i * (chunk_length - 2):(i + 1) * (chunk_length - 2) + 2].clone()
        chunk[:, 0] = text_input[0, 0]
        chunk[:, -1] = text_input[0, -1]
        chunks.append(chunk)
    return chunks


def get_unweighted_text_embeddings(pipe, text_input, chunk_length):
    """no_boseos_middle=False: every chunk keeps its bos and eos rows."""
    chunks = chunk_inputs(text_input, chunk_length)
    if len(chunks) == 1:
        return pipe.text_encoder(text_input)[0]
    return torch.concat([pipe.text_encoder(chunk)[0] for chunk in chunks], axis=1)


def get_weighted_text_embeddings(pipe, prompt, uncond_prompt, skip_parsing):
    model_max_length = pipe.tokenizer.model_max_length
    max_length = (model_max_length - 2) * MAX_EMBEDDINGS_MULTIPLES + 2
    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, [prompt], max_length - 2)
        uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, [uncond_prompt], max_length - 2)
    else:
        prompt_tokens = [token[1:-1] for token in pipe.tokenizer(
            [prompt], max_length=max_length, truncation=True).input_ids]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        uncond_tokens = [token[1:-1] for token in pipe.tokenizer(
            [uncond_prompt], max_length=max_length, truncation=True).input_ids]
        uncond_weights = [[1.0] * len(token) for token in uncond_tokens]
    max_length = max(max(len(token) for token in prompt_tokens),
                     max(len(token) for token in uncond_tokens))
    multiples = min(MAX_EMBEDDINGS_MULTIPLES, (max_length - 1) // (model_max_length - 2) + 1)
    multiples = max(1, multiples)
    max_length = (model_max_length - 2) * multiples + 2
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    pad = getattr(pipe.tokenizer, "pad_token_id", eos)
    assert pad == eos, "the fixture relies on pad == eos"
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens, prompt_weights, max_length, bos, eos, pad, model_max_length)
    uncond_tokens, uncond_weights = pad_tokens_and_weights(
        uncond_tokens, uncond_weights, max_length, bos, eos, pad, model_max_length)
    out = []
    for tokens, weights in ((prompt_tokens, prompt_weights), (uncond_tokens, uncond_weights)):
        text_input = torch.tensor(tokens, dtype=torch.long)
        ids = torch.concat(chunk_inputs(text_input, model_max_length), axis=1)[0].tolist()
        with torch.no_grad():
            embeds = get_unweighted_text_embeddings(pipe, text_input, model_max_length)
        weight_tensor = torch.tensor(weights, dtype=embeds.dtype)
        if not skip_parsing:
            previous_mean = embeds.float().mean(axis=[-2, -1]).to(embeds.dtype)
            embeds = embeds * weight_tensor.unsqueeze(-1)
            current_mean = embeds.float().mean(axis=[-2, -1]).to(embeds.dtype)
            embeds = embeds * (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        out.append((ids, weights[0], embeds[0]))
    return multiples, out


def rounded(values):
    return [float("%.7g" % float(v)) for v in values]


def main():
    pipe = StableDiffusionPipeline.from_pretrained(
        PICO_DIR, safety_checker=None, requires_safety_checker=False)
    pipe.to(torch.float64)
    ti_pipe = StableDiffusionPipeline.from_pretrained(
        PICO_DIR, safety_checker=None, requires_safety_checker=False)
    ti_pipe.to(torch.float64)
    ti_pipe.load_textual_inversion(TI_FILE)
    cases = []
    for spec in CASES:
        use = ti_pipe if spec["inversion"] else pipe
        prompt = use.maybe_convert_prompt(spec["prompt"], use.tokenizer)
        negative = use.maybe_convert_prompt(spec["negative_prompt"], use.tokenizer)
        raw_count = len(use.tokenizer(prompt, add_special_tokens=False).input_ids)
        multiples, ((ids, weights, embeds), (neg_ids, neg_weights, neg_embeds)) = \
            get_weighted_text_embeddings(use, prompt, negative, not spec["weighting"])
        case = {"name": spec["name"], "prompt": spec["prompt"],
                "negative_prompt": spec["negative_prompt"],
                "weighting": spec["weighting"], "inversion": spec["inversion"],
                "chunk_count": multiples, "chunk_ids": ids,
                "chunk_weights": rounded(weights), "negative_chunk_ids": neg_ids,
                "negative_chunk_weights": rounded(neg_weights)}
        if spec["embeds"]:
            case["prompt_embeds"] = [rounded(row) for row in embeds]
            if spec["negative_prompt"]:
                case["negative_prompt_embeds"] = [rounded(row) for row in neg_embeds]
        cases.append(case)
        print(spec["name"], "raw tokens", raw_count, "chunks", multiples,
              "weights", sorted(set(round(w, 6) for w in weights)))
    with open(IO_FILE, "w", encoding="utf-8") as f:
        json.dump({"max_chunk_count": MAX_EMBEDDINGS_MULTIPLES, "cases": cases}, f)


if __name__ == "__main__":
    main()
