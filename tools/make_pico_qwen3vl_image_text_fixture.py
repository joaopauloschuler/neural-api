#!/usr/bin/env python3
"""Generate the Qwen3-VL text-encoder-with-images oracle (tasklist C1b) from the
pico text encoder that tools/make_pico_qwenimage21_fixture.py already wrote to
tests/fixtures/tiny_qwenimage21 (nothing is regenerated).

Outputs (tests/fixtures/):
  tiny_qwen3vl_image_text_io.json   per case (1 and 2 images, non-square): the
      token ids, image_grid_thw, the get_rope_index positions (3, L), the
      image-pad mask and the last decoder layer before the final RMSNorm
      (Qwen3VLForConditionalGeneration with pixel_values, float64) after
      dropping the system rows; plus the interleaved M-RoPE pair -> axis map
      of recomposition_frequencies for a few mrope_section values
  qwenimage21_edit_prompt_tokens.json   the QwenImage21Pipeline edit template
      text for 1 and 2 images (captured from _get_qwen_prompt_embeds), its ids
      and the ids after the processor's <|image_pad|> expansion, tokenized with
      the real Qwen/Qwen-Image-2.1 tokenizer (skipped when it is not cached)

The pico has no tokenizer: the numeric cases build ids directly (system rows,
"<imageN>" stand-ins, <|vision_start|> 292, <|image_pad|> 290 x slots,
<|vision_end|> 293, prompt, assistant tail). Images are the C1a formula images.

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv, torch CPU build):
  ( ulimit -v 2097152; python3 tools/make_pico_qwen3vl_image_text_fixture.py )
"""
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pico_qwenimage21_fixture import (PICO_DIR, PREPROCESSOR_CONFIG, NativeReference,  # noqa: E402
                                           QwenImage21Pipeline, Qwen3VLForConditionalGeneration,
                                           install_float64_shims, max_abs_diff, qwen3vl_module,
                                           tensor_json, vision_formula_image, write_json)

DROP_IDX = 14
IMAGE_PAD, VISION_START, VISION_END = 290, 292, 293
SYSTEM_IDS = [(37 * i + 11) % 280 for i in range(DROP_IDX)]
USER_IDS = [5, 17, 3]
PROMPT_IDS = [60, 61, 62, 63, 64]
TAIL_IDS = [7, 8, 9, 10, 11]
# (width, height, mode) per image; merged grids 2x3, then 2x1 and 3x2.
CASES = [("one_image", [(96, 64, "RGBA")]),
         ("two_images", [(96, 64, "RGBA"), (32, 64, "RGB")]),
         ("two_images_tall_first", [(64, 96, "RGB"), (96, 64, "RGBA")])]
SECTION_CASES = [[24, 20, 20], [4, 2, 2], [3, 2, 1], [2, 3, 3]]
REAL_SNAPSHOT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots")
EDIT_PROMPTS = ["Make the sky purple", " "]


def condition_image(width, height, mode):
    image = Image.fromarray(vision_formula_image(width, height, len(mode)), mode)
    if mode == "RGBA":
        white = Image.new("RGB", image.size, (255, 255, 255))
        white.paste(image, mask=image.getchannel("A"))
        image = white
    return image


def case_ids(grids, merge):
    ids = SYSTEM_IDS + USER_IDS
    for index, (_, grid_h, grid_w) in enumerate(grids):
        ids += [41 + index] if index == 0 else [23, 41 + index]
        ids += [VISION_START] + [IMAGE_PAD] * (grid_h * grid_w // (merge * merge)) + [VISION_END]
    return ids + PROMPT_IDS + TAIL_IDS


def last_layer_before_norm(model, **inputs):
    text_model = model.model.language_model
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        outputs = model(**inputs, output_hidden_states=True)
    finally:
        handle.remove()
    return outputs.hidden_states[-1]


def make_numeric_oracle():
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(PICO_DIR, "text_encoder"), dtype=torch.float64).eval()
    processor = Qwen2VLImageProcessorPil(**{k: v for k, v in PREPROCESSOR_CONFIG.items()
                                            if k != "image_processor_type"})
    merge = PREPROCESSOR_CONFIG["merge_size"]
    cases = []
    for label, image_specs in CASES:
        images = [condition_image(*spec) for spec in image_specs]
        vision = processor(images=images, return_tensors="pt")
        grids = vision["image_grid_thw"].tolist()
        ids = case_ids(grids, merge)
        input_ids = torch.tensor([ids])
        mm_token_type_ids = (input_ids == IMAGE_PAD).to(torch.int64)
        inputs = dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                      pixel_values=vision["pixel_values"].to(torch.float64),
                      image_grid_thw=vision["image_grid_thw"], mm_token_type_ids=mm_token_type_ids)
        with torch.no_grad():
            positions, _ = model.model.get_rope_index(input_ids, mm_token_type_ids,
                                                      image_grid_thw=vision["image_grid_thw"])
            hidden = last_layer_before_norm(model, **inputs)[0]
            with NativeReference():
                native = last_layer_before_norm(model, **inputs)[0]
            text_only = last_layer_before_norm(model, input_ids=input_ids,
                                               attention_mask=torch.ones_like(input_ids))[0]
        image_rows = (input_ids[0] == IMAGE_PAD)
        cases.append({
            "label": label,
            "images": [{"width": w, "height": h, "channels": len(mode)} for w, h, mode in image_specs],
            "image_grid_thw": grids,
            "token_ids": ids,
            "drop_idx": DROP_IDX,
            "positions": tensor_json(positions[:, 0]),
            "image_pad_mask": image_rows[DROP_IDX:].to(torch.int64).tolist(),
            "prompt_embeds": tensor_json(hidden[DROP_IDX:]),
            "images_change_text_rows_maxabs": max_abs_diff(hidden[~image_rows], text_only[~image_rows]),
            "native_hf_maxabs_diff": max_abs_diff(hidden, native),
        })
        print(f"{label}: grids {grids}, {len(ids)} ids, native diff {cases[-1]['native_hf_maxabs_diff']:.3g}")

    rope = model.model.language_model.rotary_emb
    section_maps = []
    for section in SECTION_CASES:
        pairs = sum(section)
        rope.mrope_section = section
        axis = torch.arange(3, dtype=torch.float64)[:, None, None, None].expand(3, 1, 1, pairs).clone()
        recomposed = rope.recomposition_frequencies(axis)[0, 0, :pairs]
        section_maps.append({"mrope_section": section,
                             "axis_of_pair": recomposed.to(torch.int64).tolist()})
    write_json("tiny_qwen3vl_image_text_io.json", {
        "image_token_id": IMAGE_PAD, "vision_start_token_id": VISION_START,
        "vision_end_token_id": VISION_END,
        "cases": cases,
        "interleaved_sections": section_maps,
        "note": "Pico text encoder of tiny_qwenimage21 with the C1a formula images (RGBA composited "
                "over white), Qwen2VLImageProcessorPil with the pico preprocessor_config, "
                "Qwen3VLForConditionalGeneration in float64. positions = model.get_rope_index "
                "(3 rows T/H/W over all ids). prompt_embeds = last decoder layer before the final "
                "RMSNorm, rows drop_idx.. (the pipeline's split). image_pad_mask over the same rows. "
                "axis_of_pair[j] = the axis (0 T, 1 H, 2 W) recomposition_frequencies gives "
                "frequency pair j.",
    })


class RecordingProcessor:
    """Captures the text the pipeline hands to the processor, then stops the call."""

    class Stop(Exception):
        pass

    def __init__(self, tokenizer, drop_idx):
        self.tokenizer = tokenizer
        self.drop_idx = drop_idx
        self.texts = None

    def apply_chat_template(self, messages, tokenize=True, return_dict=False):
        return [self.tokenizer.encode("<|im_start|>system\n" + messages[0]["content"][0]["text"]
                                      + "<|im_end|>\n")]

    def __call__(self, text=None, **kwargs):
        self.texts = text
        raise self.Stop()


def make_template_oracle():
    from transformers import AutoTokenizer
    snapshot_dirs = []
    if os.path.isdir(REAL_SNAPSHOT_GLOB):
        snapshot_dirs = [os.path.join(REAL_SNAPSHOT_GLOB, d, "processor")
                         for d in sorted(os.listdir(REAL_SNAPSHOT_GLOB))]
    snapshot_dirs = [d for d in snapshot_dirs if os.path.isfile(os.path.join(d, "tokenizer.json"))]
    if not snapshot_dirs:
        print("SKIPPED qwenimage21_edit_prompt_tokens.json: real tokenizer not cached")
        return
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dirs[-1])
    image_pad_id = tokenizer.encode("<|image_pad|>")[0]
    recorder = RecordingProcessor(tokenizer, DROP_IDX)
    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=None, processor=recorder,
                               transformer=None)
    assert pipe._drop_idx == DROP_IDX and pipe._img_token_id == image_pad_id
    cases = []
    for image_count in (1, 2, 3):
        for prompt in EDIT_PROMPTS:
            images = [vision_formula_image(32, 32, 3)] * image_count
            try:
                pipe._get_qwen_prompt_embeds(prompt, image=images, device="cpu")
            except RecordingProcessor.Stop:
                pass
            text = recorder.texts[0]
            slots = [2 + index for index in range(image_count)]
            parts = text.split("<|image_pad|>")
            expanded = parts[0]
            for index, part in enumerate(parts[1:]):
                expanded += "<|image_pad|>" * slots[index] + part
            cases.append({"prompt": prompt, "image_count": image_count, "template_text": text,
                          "input_ids": tokenizer.encode(text), "slot_counts": slots,
                          "expanded_input_ids": tokenizer.encode(expanded)})
    write_json("qwenimage21_edit_prompt_tokens.json", {
        "image_pad_token_id": image_pad_id, "drop_idx": DROP_IDX, "cases": cases,
        "note": "template_text = the text QwenImage21Pipeline._get_qwen_prompt_embeds hands its "
                "processor for image_count images (captured; the templates are the strings "
                "__init__ sets). expanded_input_ids = the real tokenizer on template_text with the "
                "k-th <|image_pad|> repeated slot_counts[k] times (processing_qwen3_vl "
                "replace_image_token).",
    })


def main():
    torch.set_num_threads(1)
    install_float64_shims(True)
    assert qwen3vl_module is not None
    make_numeric_oracle()
    make_template_oracle()


if __name__ == "__main__":
    main()
