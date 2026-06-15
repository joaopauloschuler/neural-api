#!/usr/bin/env python3
"""Generate a tiny RANDOM SegFormer semantic-segmentation parity fixture for
tests/TestNeuralPretrained.pas.

No network access: a pico SegformerForSemanticSegmentation is built from a small
config and randomly initialized (never downloaded). The reference forward is the
REAL transformers SegformerForSemanticSegmentation run in float64 (the package is
installed in this environment); the test asserts the Pascal forward matches the
HF decode-head logits (at input/4 resolution, BEFORE the user-side upsample to
full resolution, exactly what HF's `logits` returns).

Architecture exercised (MiT-b0 shape, shrunk):
  - 4-stage MiT encoder: overlap-patch conv (7/3/3/3, stride 4/2/2/2, pad k//2)
    + LayerNorm; depths blocks; each block = LN -> efficient self-attention
    (spatial-reduction sr; Q full-res, K/V from a sr x sr stride-sr conv + LN)
    -> residual -> LN -> Mix-FFN (fc1 -> depthwise 3x3 pad 1 -> gelu -> fc2)
    -> residual; per-stage final LayerNorm.
  - all-MLP decode head: per-stage Linear to decoder dim, bilinear upsample
    (align_corners=False) to stage-0 grid, concat (reversed), 1x1 fuse conv,
    BatchNorm, ReLU, 1x1 classifier.

The fixture writes the transformers-5.x RAW tensor names (segformer.stages.{i}...,
q_proj/k_proj/v_proj/o_proj, decode_head.linear_projections); the Pascal importer
BuildSegformerFromSafeTensors reads exactly these (and also accepts the legacy
spelling).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_segformer_fixture.py
writes tests/fixtures/tiny_segformer{.safetensors,_config.json,_io.json}.
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# ---------------- pico config ----------------
# CAI clamps a conv kernel to the input spatial size (PyTorch zero-pads), so the
# importer pads the Mix-FFN 3x3 dwconv explicitly (works at any grid). The 3x3
# overlap-patch convs (stages 1..3) have NO such guard, so the grid feeding each
# patch conv must stay >= 3: the stage-3 patch input is image/8. image=64 ->
# encoder grids 16/8/4/2 (patch inputs 64/16/8/4, all >= 3).
IMAGE = 64
NUM_CHANNELS = 3
DEPTHS = [1, 1, 1, 1]
HIDDEN = [8, 16, 24, 32]
SR = [2, 2, 2, 1]
HEADS = [1, 2, 3, 4]
PATCH = [7, 3, 3, 3]
STRIDE = [4, 2, 2, 2]
MLP = [4, 4, 4, 4]
DECODER = 16
NUM_LABELS = 5

torch.manual_seed(1234)
np.random.seed(1234)

cfg = SegformerConfig(
    num_channels=NUM_CHANNELS,
    num_encoder_blocks=4,
    depths=DEPTHS,
    sr_ratios=SR,
    hidden_sizes=HIDDEN,
    patch_sizes=PATCH,
    strides=STRIDE,
    num_attention_heads=HEADS,
    mlp_ratios=MLP,
    hidden_act="gelu",
    decoder_hidden_size=DECODER,
    num_labels=NUM_LABELS,
    layer_norm_eps=1e-6,
)
model = SegformerForSemanticSegmentation(cfg).eval()

# Re-randomize every parameter on an O(1) scale: the HF default init (std 0.02)
# makes the encoder output near-constant, so the test would pass trivially.
with torch.no_grad():
    # Modest std keeps the forward in a small dynamic range so the float32
    # Pascal path stays within ~1e-4 of the float64 oracle (a larger std blows
    # the logit magnitudes up and the absolute float32 error with them), while
    # still being non-trivial (the HF default std 0.02 makes the encoder output
    # near-constant and the test vacuous).
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.18)
        elif "bias" in name:
            torch.nn.init.normal_(p, mean=0.0, std=0.05)
        else:
            torch.nn.init.normal_(p, mean=0.0, std=0.1)
    # BatchNorm running stats: give them non-trivial, well-conditioned values.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(0.0, 0.3)
            m.running_var.uniform_(0.5, 1.5)
            torch.nn.init.normal_(m.weight, 1.0, 0.2)
            torch.nn.init.normal_(m.bias, 0.0, 0.1)

# ---------------- float64 reference forward ----------------
model_d = model.double()
cases = []
for case in range(2):
    x = np.random.randn(1, NUM_CHANNELS, IMAGE, IMAGE).astype(np.float64) * 0.8
    with torch.no_grad():
        logits = model_d(torch.from_numpy(x)).logits  # (1, NUM_LABELS, H/4, W/4)
    logits = logits.numpy()[0]  # (NUM_LABELS, h, w)
    nlab, gh, gw = logits.shape
    # store input as CAI (x, y, c) raster: index ((y*W+x)*C+c)
    img_chw = x[0]  # (C, H, W)
    img_xyc = np.transpose(img_chw, (1, 2, 0)).reshape(-1)  # (H*W*C), order (y,x,c)
    # store output as CAI (x, y, c): logits is (c, y, x) -> (y, x, c)
    out_xyc = np.transpose(logits, (1, 2, 0)).reshape(-1)
    cases.append({
        "input": img_xyc.tolist(),
        "out_grid_w": int(gw),
        "out_grid_h": int(gh),
        "out_labels": int(nlab),
        "output": out_xyc.tolist(),
    })

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

state = model.float().state_dict()
tensors = {}
for k, v in state.items():
    # skip num_batches_tracked (int scalar, not needed by the importer)
    if k.endswith("num_batches_tracked"):
        continue
    tensors[k] = v.detach().cpu().numpy().astype(np.float32)
save_file(tensors, os.path.join(fixtures, "tiny_segformer.safetensors"))

config = {
    "model_type": "segformer",
    "num_channels": NUM_CHANNELS,
    "depths": DEPTHS,
    "hidden_sizes": HIDDEN,
    "sr_ratios": SR,
    "num_attention_heads": HEADS,
    "patch_sizes": PATCH,
    "strides": STRIDE,
    "mlp_ratios": MLP,
    "hidden_act": "gelu",
    "decoder_hidden_size": DECODER,
    "image_size": IMAGE,
    "num_labels": NUM_LABELS,
    "layer_norm_eps": 1e-6,
}
with open(os.path.join(fixtures, "tiny_segformer_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_segformer_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

print("wrote tiny_segformer.{safetensors,_config.json,_io.json} to", fixtures)
print(f"  image={IMAGE} hidden={HIDDEN} sr={SR} heads={HEADS} "
      f"decoder={DECODER} labels={NUM_LABELS}")
print(f"  out grid = {cases[0]['out_grid_w']}x{cases[0]['out_grid_h']} "
      f"x {cases[0]['out_labels']} labels")
print(f"  #tensors = {len(tensors)}")
