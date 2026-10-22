#!/usr/bin/env python3
"""Non-native input-size parity cases for the committed pico DPT fixtures
(tests/TestNeuralPretrained.pas TestDPT*ResizedParity). Nothing is downloaded
and no model file is rewritten: the models come from tests/fixtures.

  tests/fixtures/tiny_dpt_vit_resized_io.json      plain-ViT DPT (native 32):
      16x16 (stock transformers) and 48 wide x 16 high
  tests/fixtures/tiny_dpt_hybrid_resized_io.json   DPT-Hybrid (native 64):
      96 wide x 32 high
  tests/fixtures/tiny_depth_anything_v2_resized_io.json  Depth Anything V2
      (native 12): 16 wide x 10 high (odd 8x5 patch grid), stock transformers

DPTViT(Hybrid)Embeddings resize the position grid bilinearly
(align_corners=False) and Dinov2Embeddings bicubically. Two transformers limits
are lifted for the DPT oracle: DPTViTHybridEmbeddings refuses a non-native size
(image_size is overridden), and DPTModel hands the neck no patch grid when
backbone_config is None or is_hybrid, so a non-square map would be reshaped to
sqrt(tokens); the neck gets the grid, as it does for a DINOv2-backbone DPT.

Each io file: {"cases": [{"in_w", "in_h", "input": (H*W*C) in (y, x, c)
order, "out_grid_w", "out_grid_h", "output": (H*W) in (y, x) order}]}.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_dpt_resized_cases.py
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import (DepthAnythingConfig, DepthAnythingForDepthEstimation,
                          DPTForDepthEstimation)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")


def rounded(values):
    return [float("%.10g" % v) for v in values]


def give_neck_patch_grid(model, grid_h, grid_w):
    neck_forward = model.neck.forward

    def forward(hidden_states, patch_height=None, patch_width=None):
        return neck_forward(hidden_states, grid_h, grid_w)
    model.neck.forward = forward


def make_case(model, width, height, rng):
    x = (rng.standard_normal((1, 3, height, width)) * 0.8).astype(np.float32)
    x = x.astype(np.float64)
    with torch.no_grad():
        depth = model(torch.from_numpy(x)).predicted_depth[0].numpy()
    gh, gw = depth.shape
    print("  %dx%d -> depth %dx%d range %.4f..%.4f" %
          (width, height, gw, gh, depth.min(), depth.max()))
    return {"in_w": width, "in_h": height,
            "input": rounded(np.transpose(x[0], (1, 2, 0)).reshape(-1)),
            "out_grid_w": int(gw), "out_grid_h": int(gh),
            "output": rounded(depth.reshape(-1))}


def load_dpt(folder):
    return DPTForDepthEstimation.from_pretrained(
        os.path.join(FIXTURES, folder)).double().eval()


def write(io_name, cases):
    with open(os.path.join(FIXTURES, io_name), "w") as f:
        json.dump({"cases": cases}, f)


def make_vit():
    rng = np.random.default_rng(7301)
    patch = 8
    cases = [make_case(load_dpt("tiny_dpt_vit"), 16, 16, rng)]
    model = load_dpt("tiny_dpt_vit")
    give_neck_patch_grid(model, 16 // patch, 48 // patch)
    cases.append(make_case(model, 48, 16, rng))
    write("tiny_dpt_vit_resized_io.json", cases)


def make_hybrid():
    rng = np.random.default_rng(7302)
    patch = 16
    width, height = 96, 32
    model = load_dpt(os.path.join("tiny_sd2_depth_estimator", "depth_estimator"))
    model.dpt.embeddings.image_size = (height, width)
    give_neck_patch_grid(model, height // patch, width // patch)
    write("tiny_dpt_hybrid_resized_io.json", [make_case(model, width, height, rng)])


def make_depth_anything():
    rng = np.random.default_rng(7303)
    with open(os.path.join(FIXTURES, "tiny_depth_anything_v2_config.json")) as f:
        fields = json.load(f)
    # As the fixture generator built it: the neck reshapes the token maps.
    fields["backbone_config"].update(reshape_hidden_states=False,
                                     apply_layernorm=True)
    config = DepthAnythingConfig(**fields)
    model = DepthAnythingForDepthEstimation(config)
    missing, unexpected = model.load_state_dict(load_file(
        os.path.join(FIXTURES, "tiny_depth_anything_v2.safetensors")), strict=False)
    assert not unexpected and all(k.endswith("mask_token") for k in missing), \
        (missing, unexpected)
    model = model.double().eval()
    write("tiny_depth_anything_v2_resized_io.json",
          [make_case(model, 16, 10, rng)])


if __name__ == "__main__":
    make_vit()
    make_hybrid()
    make_depth_anything()
