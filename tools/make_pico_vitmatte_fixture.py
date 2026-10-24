#!/usr/bin/env python3
"""Generate a tiny RANDOM ViTMatte (VitMatteForImageMatting) parity fixture for
tests/TestNeuralPretrained.pas (TestViTMatteParity, TestViTMatteOpenCLParity).

No network access: the model is built from a pico config and randomly
initialized. The oracle is transformers' own VitMatteImageProcessor (for the
4-channel padded input) and VitMatteForImageMatting run in float64.

Exercised:
  - VitDet backbone with a 4-channel input (RGB + trimap): biased patch conv,
    absolute position table (pretrain_image_size 40 -> 5x5 grid) BICUBICALLY
    resized to the 8x4 input grid, windowed blocks (window 3, zero-padded
    windows) and global blocks with decomposed rel-pos; the global tables
    (image_size 32 -> 7 rows) are LINEARLY resized to 15 rows on the 8-wide
    axis and kept on the 4-high axis; residual bottleneck conv blocks.
  - detail capture: ConvStream (2 stride-2 conv+BN+ReLU stages), 3 fusion
    blocks (bilinear x2, concat, conv+BN+ReLU), matting head + sigmoid.
  - a 45x30 image + trimap padded to 64x32 by the processor.

Every weight and BatchNorm statistic is re-randomized to O(1) scale so each
quirk is visible above the 1e-4 parity tolerance.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python -I tools/make_pico_vitmatte_fixture.py
writes tests/fixtures/tiny_vitmatte/{model.safetensors,config.json} and
tests/fixtures/tiny_vitmatte_io.json:
  {"width", "height", "image": (H*W*3) uint8 in (y, x, c) order,
   "trimap": (H*W) uint8, "padded_width", "padded_height",
   "pixel_values": (PH*PW*4) in (y, x, c) order, "alphas": (PH*PW)}.
"""
import json
import os

import numpy as np
import torch
from transformers import (VitDetConfig, VitMatteConfig,
                          VitMatteForImageMatting, VitMatteImageProcessor)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "tiny_vitmatte")
IMAGE_W, IMAGE_H = 45, 30

torch.manual_seed(20261023)
rng = np.random.default_rng(20261023)

backbone = VitDetConfig(
    hidden_size=16, num_hidden_layers=4, num_attention_heads=2, mlp_ratio=2,
    image_size=32, pretrain_image_size=40, patch_size=8, num_channels=4,
    window_size=3, window_block_indices=[0, 2], residual_block_indices=[1, 2],
    use_relative_position_embeddings=True, out_features=["stage4"])
config = VitMatteConfig(backbone_config=backbone, hidden_size=16,
                        convstream_hidden_sizes=[4, 6],
                        fusion_hidden_sizes=[8, 6, 4])
model = VitMatteForImageMatting(config).eval()

with torch.no_grad():
    for name, tensor in model.state_dict().items():
        if name.endswith("num_batches_tracked"):
            continue
        if name.endswith("running_var"):
            tensor.copy_(torch.rand_like(tensor) + 0.5)
        elif name.endswith("running_mean"):
            tensor.normal_(0.0, 0.3)
        elif tensor.dim() == 1 and name.endswith(".weight"):
            tensor.normal_(1.0, 0.25)
        elif tensor.dim() == 1:
            tensor.normal_(0.0, 0.2)
        elif "rel_pos" in name or "position_embeddings" in name:
            tensor.normal_(0.0, 0.5)
        else:
            fan_in = tensor[0].numel()
            tensor.normal_(0.0, 1.0 / np.sqrt(fan_in))

os.makedirs(OUT_DIR, exist_ok=True)
model.float().save_pretrained(OUT_DIR)
model = model.double().eval()

image = rng.integers(0, 256, size=(IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
trimap = np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8)
trimap[:, IMAGE_W // 3:] = 128
trimap[:, 2 * IMAGE_W // 3:] = 255
trimap[rng.random((IMAGE_H, IMAGE_W)) < 0.1] = 128

processor = VitMatteImageProcessor()
pixel_values = processor(images=image, trimaps=trimap,
                         return_tensors="pt").pixel_values
with torch.no_grad():
    alphas = model(pixel_values.double()).alphas[0, 0].numpy()
padded_h, padded_w = alphas.shape
print("pixel_values", tuple(pixel_values.shape), "alphas range %.4f..%.4f std %.4f"
      % (alphas.min(), alphas.max(), alphas.std()))


def rounded(values):
    return [float("%.10g" % v) for v in values]


with open(os.path.join(FIXTURES, "tiny_vitmatte_io.json"), "w") as f:
    json.dump({
        "width": IMAGE_W, "height": IMAGE_H,
        "image": image.reshape(-1).tolist(),
        "trimap": trimap.reshape(-1).tolist(),
        "padded_width": int(padded_w), "padded_height": int(padded_h),
        "pixel_values": rounded(
            pixel_values[0].permute(1, 2, 0).reshape(-1).double().numpy()),
        "alphas": rounded(alphas.reshape(-1)),
    }, f)
