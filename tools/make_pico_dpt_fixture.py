#!/usr/bin/env python3
"""Generate a tiny RANDOM Depth-Anything (DPT) monocular-depth parity fixture for
tests/TestNeuralPretrained.pas.

No network access: a pico DepthAnythingForDepthEstimation (a DINOv2 ViT backbone
+ the DPT reassemble/fusion neck + a 3-conv depth head) is built from a small
config and randomly initialized (never downloaded). The reference forward is the
REAL transformers DepthAnythingForDepthEstimation run in float64 (the package is
installed in this environment); the test asserts the Pascal forward matches the
HF `predicted_depth` per-pixel map at full input resolution.

This is the repo's FIRST dense per-pixel REGRESSION vision importer (depth, not a
class map). The genuinely new code is the convolutional DECODER neck + head; the
DINOv2 encoder is reused (BuildDINOv2FromSafeTensors path).

Architecture exercised (Depth-Anything-V2-Small shape, shrunk):
  - DINOv2 backbone: biased patch conv (k=stride=patch, no pad) -> CLS-token row
    + learned position table -> N pre-LN encoder blocks (fused QKV, LayerScale on
    both residual branches, gelu MLP) -> final LayerNorm. The backbone hooks the
    output of all N blocks (out_features = stage1..N), applies the SHARED final
    LayerNorm to each, drops the CLS row and reshapes each to a (patch_h,patch_w)
    grid.
  - DPT reassemble stage: per-stage 1x1 projection to neck_hidden_sizes[i], then
    a per-stage resize by reassemble_factors [4,2,1,0.5]: factor 4/2 = a
    ConvTranspose2d (kernel=stride=factor, pad 0); factor 1 = identity; factor
    0.5 = a 3x3 stride-2 pad-1 Conv2d downsample. -> a 4-level feature pyramid.
  - neck 3x3 pad-1 bias-free convs map every level to fusion_hidden_size.
  - RefineNet-style fusion stage (reversed, coarse->fine): each level adds the
    upsampled-and-refined coarser feature (pre-act residual conv units), refines,
    bilinearly upsamples (align_corners=True, x2 or to the next level's size) and
    1x1-projects.
  - depth head: conv1 3x3 -> bilinear upsample to (patch_h*patch, patch_w*patch)
    (align_corners=True) -> conv2 3x3 -> ReLU -> conv3 1x1 -> ReLU (relative
    depth) * max_depth -> a single-channel per-pixel depth map.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_dpt_fixture.py
writes tests/fixtures/tiny_dpt{.safetensors,_config.json,_io.json}.
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
# (g -> g/2) lands on an integer grid. IMAGE=12, PATCH=2 -> g=6, reassemble grids
# 24/12/6/3, fusion walks 3->6->12->24, head upsamples to 12 (=g*PATCH).
IMAGE = 12
PATCH = 2
NUM_CHANNELS = 3
HIDDEN = 16
NUM_LAYERS = 4
NUM_HEADS = 2
MLP_RATIO = 2
NECK_HIDDEN = [4, 8, 12, 16]
FUSION_HIDDEN = 8
HEAD_HIDDEN = 4
REASSEMBLE_FACTORS = [4, 2, 1, 0.5]
LAYER_NORM_EPS = 1e-6

torch.manual_seed(1234)
np.random.seed(1234)

bc = Dinov2Config(
    hidden_size=HIDDEN,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    image_size=IMAGE,
    patch_size=PATCH,
    num_channels=NUM_CHANNELS,
    layer_norm_eps=LAYER_NORM_EPS,
    out_features=["stage%d" % (i + 1) for i in range(NUM_LAYERS)],
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

# Re-randomize every parameter on an O(1) scale: the HF default init (std 0.02)
# makes the encoder output near-constant, so the test would pass trivially. A
# modest std keeps the forward in a small dynamic range so the float32 Pascal
# path stays within ~1e-4 of the float64 oracle.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.09)
        elif "bias" in name:
            torch.nn.init.normal_(p, mean=0.0, std=0.05)
        else:
            torch.nn.init.normal_(p, mean=0.0, std=0.1)
    # The final-ReLU depth head clamps negatives to 0; bias the last conv up so
    # the reference map is not mostly zeros (a vacuous parity check).
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
    img_xyc = np.transpose(img_chw, (1, 2, 0)).reshape(-1)  # (H*W*C) order (y,x,c)
    out_xyc = depth.reshape(-1)  # (H*W) order (y,x); single channel
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
    # mask_token is unused at inference; drop it to keep the fixture small.
    if k.endswith("embeddings.mask_token"):
        continue
    tensors[k] = v.detach().cpu().numpy().astype(np.float32)
save_file(tensors, os.path.join(fixtures, "tiny_dpt.safetensors"))

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
        "out_features": ["stage%d" % (i + 1) for i in range(NUM_LAYERS)],
    },
}
with open(os.path.join(fixtures, "tiny_dpt_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_dpt_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

print("wrote tiny_dpt.{safetensors,_config.json,_io.json} to", fixtures)
print(f"  image={IMAGE} patch={PATCH} hidden={HIDDEN} layers={NUM_LAYERS} "
      f"neck={NECK_HIDDEN} fusion={FUSION_HIDDEN} head_hidden={HEAD_HIDDEN}")
print(f"  out map = {cases[0]['out_grid_w']}x{cases[0]['out_grid_h']} depth")
print(f"  #tensors = {len(tensors)}")
