#!/usr/bin/env python3
"""Generate a tiny RANDOM ViTPose human-pose parity fixture for
tests/TestNeuralPretrained.pas.

No network access: a pico VitPoseForPoseEstimation (a plain ViT backbone + the
"simple" deconvolution head) is built from a small config and randomly
initialized (never downloaded). The reference forward is the REAL transformers
VitPoseForPoseEstimation run in float64 (the package is installed in this
environment); the test asserts the Pascal forward matches the HF `heatmaps`
per-joint heatmap tensor.

This is the repo's FIRST keypoint / human-pose importer: the output modality is
per-joint 2-D heatmaps (a stack of HxW maps, one per keypoint), decoded to (x,y)
coordinates by a spatial argmax. The genuinely new code is the deconvolution
heatmap head + the argmax decode; the ViT encoder is reused (BuildViT path).

Architecture exercised (vitpose-base-simple shape, shrunk, RECTANGULAR image):
  - ViT backbone: biased patch conv (k=stride=patch, PADDING=2 - a ViTPose
    quirk: nn.Conv2d(..., padding=2); for patch>=8 the floor of the output grid
    still equals image//patch) -> (gh,gw) patch grid flattened row-major -> NO
    class token kept; each patch token gets
    position_embeddings[1:] PLUS the (broadcast) cls position row [:1] -> N
    pre-LN encoder blocks (separate q/k/v, biased, gelu MLP) -> final LayerNorm.
  - reshape the N patch tokens back to a (gh,gw,hidden) grid.
  - simple decoder head: ReLU -> bilinear Upsample(scale_factor, align_corners=
    False) -> 3x3 pad-1 Conv2d to num_labels (= number of keypoints) channels ->
    one (gh*scale, gw*scale) heatmap per joint.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_vitpose_fixture.py
writes tests/fixtures/tiny_vitpose{.safetensors,_config.json,_io.json}.
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import (
    VitPoseBackboneConfig,
    VitPoseConfig,
    VitPoseForPoseEstimation,
)

# ---------------- pico config ----------------
# Rectangular image to exercise the general (gh != gw) path. The ViTPose patch
# conv hardcodes padding=2; for patch>=5 the floor of the padded output grid
# still equals image//patch (matching the num_patches the pos table was sized
# for). IMAGE=[24,16], PATCH=8 -> grid 3x2 = 6 patches (pos table 7 = 6 + 1 cls
# row). scale 4 -> heatmaps 12x8 per joint.
IMAGE_H = 24
IMAGE_W = 16
PATCH = 8
NUM_CHANNELS = 3
HIDDEN = 16
NUM_LAYERS = 2
NUM_HEADS = 2
MLP_RATIO = 2              # intermediate_size = hidden * mlp_ratio
INTERMEDIATE = HIDDEN * MLP_RATIO
NUM_KEYPOINTS = 4          # num_labels (joints)
SCALE_FACTOR = 4
LAYER_NORM_EPS = 1e-12

torch.manual_seed(1234)
np.random.seed(1234)

bc = VitPoseBackboneConfig(
    hidden_size=HIDDEN,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    image_size=[IMAGE_H, IMAGE_W],
    patch_size=[PATCH, PATCH],
    num_channels=NUM_CHANNELS,
    num_experts=1,
    layer_norm_eps=LAYER_NORM_EPS,
)
cfg = VitPoseConfig(
    backbone_config=bc,
    num_labels=NUM_KEYPOINTS,
    scale_factor=SCALE_FACTOR,
)
model = VitPoseForPoseEstimation(cfg).eval()

# Re-randomize on an O(1) scale: the HF default init (std 0.02) makes the encoder
# output near-constant, so the parity test would pass trivially. A modest std
# keeps the forward in a small dynamic range so the float32 Pascal path stays
# within ~1e-4 of the float64 oracle.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.09)
        elif "bias" in name:
            torch.nn.init.normal_(p, mean=0.0, std=0.05)
        else:
            torch.nn.init.normal_(p, mean=0.0, std=0.1)
    # The ReLU in the head clamps the encoder output's negatives to 0; bias the
    # head conv up so the reference heatmaps are not mostly the conv bias (a
    # vacuous parity check) and have a well-defined per-joint peak.
    model.head.conv.bias.add_(0.2)

# ---------------- float64 reference forward ----------------
model_d = model.double()
cases = []
for case in range(2):
    x = np.random.randn(1, NUM_CHANNELS, IMAGE_H, IMAGE_W).astype(np.float64) * 0.8
    with torch.no_grad():
        heatmaps = model_d(torch.from_numpy(x)).heatmaps  # (1, K, Hh, Hw)
    heatmaps = heatmaps.numpy()[0]  # (K, Hh, Hw)
    K, Hh, Hw = heatmaps.shape
    img_chw = x[0]  # (C, H, W)
    img_xyc = np.transpose(img_chw, (1, 2, 0)).reshape(-1)  # (H*W*C) order (y,x,c)
    out_cyx = heatmaps.reshape(-1)  # (K*Hh*Hw) order (c,y,x)
    # argmax (x,y) peak per joint, for the decode test.
    peaks = []
    for k in range(K):
        flat = int(np.argmax(heatmaps[k]))
        py, px = divmod(flat, Hw)
        peaks.append([int(px), int(py)])
    cases.append({
        "input": img_xyc.tolist(),
        "out_grid_w": int(Hw),
        "out_grid_h": int(Hh),
        "num_keypoints": int(K),
        "output": out_cyx.tolist(),
        "peaks": peaks,
    })

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

state = model.float().state_dict()
tensors = {}
for k, v in state.items():
    if k.endswith("num_batches_tracked"):
        continue
    tensors[k] = v.detach().cpu().numpy().astype(np.float32)
save_file(tensors, os.path.join(fixtures, "tiny_vitpose.safetensors"))

config = {
    "model_type": "vitpose",
    "num_labels": NUM_KEYPOINTS,
    "scale_factor": SCALE_FACTOR,
    "backbone_config": {
        "model_type": "vitpose_backbone",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "mlp_ratio": MLP_RATIO,
        "image_size": [IMAGE_H, IMAGE_W],
        "patch_size": [PATCH, PATCH],
        "num_channels": NUM_CHANNELS,
        "num_experts": 1,
        "hidden_act": "gelu",
        "layer_norm_eps": LAYER_NORM_EPS,
    },
}
with open(os.path.join(fixtures, "tiny_vitpose_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_vitpose_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

print("wrote tiny_vitpose.{safetensors,_config.json,_io.json} to", fixtures)
print(f"  image={IMAGE_H}x{IMAGE_W} patch={PATCH} hidden={HIDDEN} "
      f"layers={NUM_LAYERS} keypoints={NUM_KEYPOINTS} scale={SCALE_FACTOR}")
print(f"  heatmaps = {cases[0]['num_keypoints']} x "
      f"{cases[0]['out_grid_h']}x{cases[0]['out_grid_w']}")
print(f"  peaks case0 = {cases[0]['peaks']}")
print(f"  #tensors = {len(tensors)}")
