#!/usr/bin/env python3
"""Generate the Qwen-Image-2.1 transformer oracle WITH CONDITION IMAGES (tasklist
C3a) from the pico transformer that tools/make_pico_qwenimage21_fixture.py
already wrote to tests/fixtures/tiny_qwenimage21/transformer (nothing is
regenerated).

Output: tests/fixtures/tiny_qwenimage21_edit_transformer_io.json
  cases: one entry per CASES layout (text runs and condition-image grids):
    encoder_hidden_states   (L, context_in_dim): text-encoder rows, image slots
                            included (diffusers discards those rows)
    img_mask                (L,) 1 at the image slots of the prompt (no target)
    positions_fhw           (prefix_len + target h*w, 3) QwenImage21Rope indices
    condition_latents       (sum h*w, in_channels): condition images in order
    latents_1, latents_2    (target h*w, in_channels)
    cache_k, cache_v        per block, (prefix_len, heads, head_dim) post-RoPE
    extract_target_output   step 1 (timestep_1, extract mode), target rows
    cached_output           step 2 (timestep_2, cached mode)

Inputs are random (fixed seeds); the forwards run in float64 on the BF16 weights
with the float64 shims of the main generator.

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv, torch CPU build):
  ( ulimit -v 3145728; python3 tools/make_pico_qwenimage21_edit_transformer_fixture.py )
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pico_qwenimage21_fixture import (PICO_DIR, TR_CHANNELS, TR_GRID,  # noqa: E402
                                           TR_LAYERS, QwenImage21KVCache,
                                           QwenImage21Transformer2DModel,
                                           install_float64_shims, max_abs_diff,
                                           rebuild_rope_tables, rope_index_table,
                                           tensor_json, write_json)

TOKENS_PER_SLOT = 4
# (text run lengths, condition grids (h, w)): text[0], image[0], text[1], ...
CASES = [
    ([5, 3], [(2, 6)]),               # one condition image
    ([4, 0, 3], [(3, 4), (2, 2)]),    # two adjacent images, odd height first
]


def joint_img_mask(case):
    """The pipeline's img_mask: the prompt's slots, then one slot per 2x2 target latents."""
    target_slots = TR_GRID[1] * TR_GRID[2] // TOKENS_PER_SLOT
    return torch.cat([case["img_mask"], torch.ones(target_slots, dtype=torch.bool)])


def run(model, case, latents, t, kv_cache=None, kv_cache_mode=None):
    img_shapes = [[*[(1, h, w) for h, w in case["grids"]], TR_GRID]]
    img_mask = joint_img_mask(case)[None]
    hidden = torch.cat([case["condition_latents"], latents], dim=0)[None]
    return model(hidden_states=hidden, encoder_hidden_states=case["encoder_hidden_states"][None],
                 timestep=t, img_shapes=img_shapes, img_mask=img_mask, kv_cache=kv_cache,
                 kv_cache_mode=kv_cache_mode, return_dict=False)[0]


def main():
    torch.set_num_threads(1)
    install_float64_shims(True)
    model = QwenImage21Transformer2DModel.from_pretrained(
        os.path.join(PICO_DIR, "transformer"), torch_dtype=torch.float64).eval()
    rebuild_rope_tables(model.pos_embed)
    context_dim = model.config.context_in_dim
    target_tokens = TR_GRID[1] * TR_GRID[2]
    gen = torch.Generator().manual_seed(20261005)
    t_1 = torch.tensor([0.9], dtype=torch.float64)
    t_2 = torch.tensor([0.35], dtype=torch.float64)
    cases = []
    for text_lengths, grids in CASES:
        mask = []
        for run_pos, text_len in enumerate(text_lengths):
            mask += [False] * text_len
            if run_pos < len(grids):
                h, w = grids[run_pos]
                mask += [True] * (h * w // TOKENS_PER_SLOT)
        case = {
            "grids": grids,
            "img_mask": torch.tensor(mask, dtype=torch.bool),
            "encoder_hidden_states": torch.randn(len(mask), context_dim, generator=gen,
                                                 dtype=torch.float64),
            "condition_latents": torch.randn(sum(h * w for h, w in grids), TR_CHANNELS,
                                             generator=gen, dtype=torch.float64),
        }
        latents_1 = torch.randn(target_tokens, TR_CHANNELS, generator=gen, dtype=torch.float64)
        latents_2 = torch.randn(target_tokens, TR_CHANNELS, generator=gen, dtype=torch.float64)
        prefix_len = sum(text_lengths) + case["condition_latents"].shape[0]
        with torch.no_grad():
            full = run(model, case, latents_1, t_1)
            cache = QwenImage21KVCache(TR_LAYERS)
            extract = run(model, case, latents_1, t_1, cache, "extract")
            cached = run(model, case, latents_2, t_2, cache, "cached")
            full_2 = run(model, case, latents_2, t_2)
        assert max_abs_diff(full, extract) < 1e-12
        cached_vs_full = max_abs_diff(cached, full_2[:, prefix_len:])
        assert cached_vs_full < 1e-12, cached_vs_full
        assert cache.get_layer(0).k.shape[1] == prefix_len
        slot_mask = joint_img_mask(case)
        image_pad_mask = torch.repeat_interleave(
            slot_mask, torch.where(slot_mask, TOKENS_PER_SLOT, 1))
        positions = rope_index_table([*[(1, h, w) for h, w in grids], TR_GRID], image_pad_mask)
        cases.append({
            "text_lengths": text_lengths,
            "condition_grids": [list(grid) for grid in grids],
            "target_grid": [TR_GRID[1], TR_GRID[2]],
            "prefix_len": prefix_len,
            "img_mask": case["img_mask"].to(torch.int64).tolist(),
            "positions_fhw": tensor_json(positions),
            "encoder_hidden_states": tensor_json(case["encoder_hidden_states"]),
            "condition_latents": tensor_json(case["condition_latents"]),
            "latents_1": tensor_json(latents_1),
            "latents_2": tensor_json(latents_2),
            "cache_k": [tensor_json(cache.get_layer(i).k[0]) for i in range(TR_LAYERS)],
            "cache_v": [tensor_json(cache.get_layer(i).v[0]) for i in range(TR_LAYERS)],
            "extract_target_output": tensor_json(extract[0, prefix_len:]),
            "cached_output": tensor_json(cached[0]),
            "cached_vs_full_maxabs_diff": cached_vs_full,
        })
    write_json("tiny_qwenimage21_edit_transformer_io.json", {
        "timestep_1": float(t_1), "timestep_2": float(t_2),
        "cases": cases,
        "note": "Layout per case: text_lengths[0] rows, condition image 0 (h*w latent tokens, "
                "h*w/4 slots in img_mask), text_lengths[1], ...; the target image follows the "
                "prompt. Latents are packed (tokens, channels), token = h * w_count + w. "
                "cache_k/cache_v = extract-mode prefix K/V (text and condition-image rows, "
                "post-norm post-RoPE). timestep is t in [0,1] (pipeline t/1000).",
    })


if __name__ == "__main__":
    main()
