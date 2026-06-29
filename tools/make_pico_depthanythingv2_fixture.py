#!/usr/bin/env python3
"""Generate a tiny RANDOM Depth-Anything-V2 monocular relative-depth parity
fixture for tests/TestNeuralPretrained.pas (TestDepthAnythingV2Parity).

Depth Anything V2 IS the DPT depth stack on a DINOv2 ViT backbone (S/B/L); the
HF model_type is "depth_anything" and the reference class is
DepthAnythingForDepthEstimation - the SAME class the tiny_dpt fixture exercises.
The point of THIS distinct fixture is to pin the backbone `out_indices` wiring:
unlike tiny_dpt (which hooks ALL N backbone stages, so "last 4" trivially equals
"all"), this pico backbone has 6 layers and selects NON-contiguous, NON-last-4
stages out_indices=[2,3,5,6] (0-based blocks 1,2,4,5). That exercises the
importer's out_indices path (tap the selected encoder blocks, shared final
LayerNorm) rather than the hardcoded last-4 assumption.

No network access: a pico DepthAnythingForDepthEstimation is built from a small
config and randomly initialized (never downloaded). The float64 reference forward
is the REAL transformers DepthAnythingForDepthEstimation (installed here); the
test asserts the Pascal forward matches HF `predicted_depth` per-pixel at full
input resolution (max |diff| < 1e-4).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_depthanythingv2_fixture.py
writes tests/fixtures/tiny_depth_anything_v2{.safetensors,_config.json,_io.json}.
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import (
    Dinov2Config,
    DepthAnythingConfig,
    DepthAnythingForDepthEstimation,
)

# ---------------- pico config ----------------
# patch grid g = IMAGE/PATCH must be even so the reassemble factor-0.5 downsample
# (g -> g/2) lands on an integer grid. IMAGE=12, PATCH=2 -> g=6.
IMAGE = 12
PATCH = 2
NUM_CHANNELS = 3
HIDDEN = 16
NUM_LAYERS = 6                 # deeper than tiny_dpt so out_indices != last-4
NUM_HEADS = 2
MLP_RATIO = 2
NECK_HIDDEN = [4, 8, 12, 16]
FUSION_HIDDEN = 8
HEAD_HIDDEN = 4
REASSEMBLE_FACTORS = [4, 2, 1, 0.5]
LAYER_NORM_EPS = 1e-6
# NON-contiguous, NON-last-4 stage hooks (1-based) -> 0-based blocks [1,2,4,5].
OUT_INDICES = [2, 3, 5, 6]

torch.manual_seed(4242)
np.random.seed(4242)

bc = Dinov2Config(
    hidden_size=HIDDEN,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    image_size=IMAGE,
    patch_size=PATCH,
    num_channels=NUM_CHANNELS,
    layer_norm_eps=LAYER_NORM_EPS,
    out_indices=OUT_INDICES,
    apply_layernorm=True,
    reshape_hidden_states=False,
    use_swiglu_ffn=False,
)
cfg = DepthAnythingConfig(
    backbone_config=bc,
    reassemble_hidden_size=HIDDEN,
    patch_size=PATCH,
    neck_hidden_sizes=NECK_HIDDEN,
    fusion_hidden_size=FUSION_HIDDEN,
    head_in_index=-1,
    head_hidden_size=HEAD_HIDDEN,
    depth_estimation_type="relative",
    reassemble_factors=REASSEMBLE_FACTORS,
    max_depth=1,
)
model = DepthAnythingForDepthEstimation(cfg).eval()

# Sanity: confirm HF resolved the stage hooks exactly as we expect.
assert list(model.backbone.config.out_indices) == OUT_INDICES, \
    model.backbone.config.out_indices

# Re-randomize every parameter on an O(1) scale (HF default init std 0.02 makes
# the encoder output near-constant -> a vacuous parity check).
with torch.no_grad():
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.09)
        elif "bias" in name:
            torch.nn.init.normal_(p, mean=0.0, std=0.05)
        else:
            torch.nn.init.normal_(p, mean=0.0, std=0.1)
    # The final-ReLU depth head clamps negatives to 0; bias conv3 up so the
    # reference map is not mostly zeros.
    model.head.conv3.bias.add_(0.3)

# ---------------- float64 reference forward ----------------
model_d = model.double()
cases = []
for case in range(2):
    x = np.random.randn(1, NUM_CHANNELS, IMAGE, IMAGE).astype(np.float64) * 0.8
    with torch.no_grad():
        depth = model_d(torch.from_numpy(x)).predicted_depth  # (1, H, W)
    depth = depth.numpy()[0]  # (H, W)
    gh, gw = depth.shape
    img_chw = x[0]  # (C, H, W)
    img_xyc = np.transpose(img_chw, (1, 2, 0)).reshape(-1)  # (y,x,c)
    out_xyc = depth.reshape(-1)  # (y,x); single channel
    cases.append({
        "input": img_xyc.tolist(),
        "out_grid_w": int(gw),
        "out_grid_h": int(gh),
        "output": out_xyc.tolist(),
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
    if k.endswith("embeddings.mask_token"):
        continue
    tensors[k] = v.detach().cpu().numpy().astype(np.float32)
save_file(tensors, os.path.join(fixtures, "tiny_depth_anything_v2.safetensors"))

config = {
    "model_type": "depth_anything",
    "patch_size": PATCH,
    "reassemble_hidden_size": HIDDEN,
    "neck_hidden_sizes": NECK_HIDDEN,
    "fusion_hidden_size": FUSION_HIDDEN,
    "head_hidden_size": HEAD_HIDDEN,
    "head_in_index": -1,
    "reassemble_factors": REASSEMBLE_FACTORS,
    "depth_estimation_type": "relative",
    "max_depth": 1,
    "backbone_config": {
        "model_type": "dinov2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "mlp_ratio": MLP_RATIO,
        "image_size": IMAGE,
        "patch_size": PATCH,
        "num_channels": NUM_CHANNELS,
        "layer_norm_eps": LAYER_NORM_EPS,
        "hidden_act": "gelu",
        "out_indices": OUT_INDICES,
    },
}
with open(os.path.join(fixtures, "tiny_depth_anything_v2_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_depth_anything_v2_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

print("wrote tiny_depth_anything_v2.{safetensors,_config.json,_io.json} to",
      fixtures)
print(f"  image={IMAGE} patch={PATCH} hidden={HIDDEN} layers={NUM_LAYERS} "
      f"out_indices={OUT_INDICES} neck={NECK_HIDDEN} fusion={FUSION_HIDDEN}")
print(f"  out map = {cases[0]['out_grid_w']}x{cases[0]['out_grid_h']} depth")
print(f"  #tensors = {len(tensors)}")
