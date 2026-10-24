#!/usr/bin/env python3
"""Generate a pico RANDOM EfficientLoFTR dense-matching parity fixture for
tests/TestNeuralPretrained.pas (TestEfficientLoFTR*).

No network access: a pico transformers EfficientLoFTRForKeypointMatching
(RepVGG stages [1, 1, 2, 2], widths [16, 16, 24, 32], 8 heads, 2 attention
layers) is built with random weights and random BatchNorm statistics. The
reference runs in float64 on exactly the float32 weights stored.

What the oracle stores and where it comes from:
  confidence    the coarse dual-softmax matrix (HW0 x HW1) that transformers
                hands to _get_matches_from_scores;
  matches       (i, j, score) of every mutual nearest neighbour above the
                threshold away from the border, in image-0 order; this equals
                transformers' second match row (asserted below);
  fine windows  transformers' own refinement_layer output (the 8x8 window of
                image 0 and the 10x10 window of image 1 per match);
  keypoints     the sub-pixel refinement of the ORIGINAL EfficientLoFTR code
                (zju3dv/EfficientLoFTR src/loftr/utils/fine_matching.py) run on
                those windows. transformers 5.17 does not follow the original
                here: its extra batch axis moves the fine dual softmax and the
                window-grid gather onto the keypoint axis, it pairs image-0
                rows ordered by image 1 with image-1 rows ordered by image 0,
                and it scales the first-stage features by the sliced width
                instead of the full fine width. The Pascal matcher follows the
                original, so the keypoints are checked against it.
A float32 rerun must give the same matches and the same first-stage window
argmax, so no decision sits on a float32 tie.

Writes tests/fixtures/tiny_efficientloftr{.safetensors,_config.json,
_io.safetensors}. Usage (from the repo root):
  systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 \\
    ~/x/bin/python -I tools/make_pico_efficientloftr_fixture.py

Coded by Claude (AI).
"""
import json
import math
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import EfficientLoFTRConfig, EfficientLoFTRForKeypointMatching

WIDTH = 128
HEIGHT = 96
SHIFT = 8
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
name = "tiny_efficientloftr"

# Seed 4 gives 7 confident interior matches on this pair.
torch.manual_seed(4)
np.random.seed(4)
cfg = EfficientLoFTRConfig(
    stage_num_blocks=[1, 1, 2, 2],
    out_features=[16, 16, 24, 32],
    hidden_size=32,
    num_attention_heads=8,
    num_attention_layers=2,
    attn_implementation="eager",
)
model = EfficientLoFTRForKeypointMatching(cfg).eval()
with torch.no_grad():
    for pname, p in model.named_parameters():
        if p.dim() >= 2:
            fan_in = int(np.prod(p.shape[1:]))
            torch.nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(fan_in))
            if "q_aggregation" in pname:
                p.abs_()  # a smoothing 4x4 depthwise filter
        elif pname.endswith("weight"):
            torch.nn.init.normal_(p, mean=1.0, std=0.1)

        else:
            torch.nn.init.normal_(p, mean=0.0, std=0.1)
    # Larger regress channels give the 3x3 sub-pixel step offsets of a few
    # tenths of a pixel instead of a near-uniform heatmap.
    last_conv = model.refinement_layer.out_conv_layers[-1].out_conv3.weight
    last_conv[-cfg.fine_matching_slice_dim:].mul_(12.0)
    for bname, b in model.named_buffers():
        if bname.endswith("running_mean"):
            b.copy_(torch.randn_like(b) * 0.1)
        elif bname.endswith("running_var"):
            b.copy_(torch.rand_like(b) + 0.5)
model = model.float()

state = {k: v.detach().cpu().numpy().astype(np.float32)
         for k, v in model.state_dict().items() if "num_batches_tracked" not in k}


def smooth_texture(width, height, seed):
    rng = np.random.RandomState(seed)
    coarse = rng.rand(height // 4 + 2, width // 4 + 2)
    t = torch.from_numpy(coarse)[None, None]
    up = torch.nn.functional.interpolate(t, scale_factor=4, mode="bilinear",
                                         align_corners=False)[0, 0].numpy()
    return up[:height, :width] * 0.8 + 0.1 * rng.rand(height, width)


big = smooth_texture(WIDTH + SHIFT, HEIGHT, 7)
image0 = big[:, SHIFT:].copy()
image1 = big[:, :WIDTH].copy()  # image 0 moved SHIFT pixels to the right


def original_fine(model, window0, window1, coarse_points, slice_dim, temperature):
    """zju3dv EfficientLoFTR FineMatching.forward on [M, WW, C] windows."""
    count, ww, channels = window0.shape
    w = int(round(math.sqrt(ww)))
    f0, f1 = window0[..., :-slice_dim], window1[..., :-slice_dim]
    ff0, ff1 = window0[..., -slice_dim:], window1[..., -slice_dim:]
    f0, f1 = f0 / channels ** .5, f1 / channels ** .5
    conf = torch.einsum("mlc,mrc->mlr", f0, f1)
    conf_ff = torch.einsum("mlc,mrc->mlr", ff0, ff1 / slice_dim ** .5)
    soft = torch.softmax(conf, 1) * torch.softmax(conf, 2)
    soft = soft.reshape(count, ww, w + 2, w + 2)[..., 1:-1, 1:-1].reshape(count, ww, ww)
    idx = soft.reshape(count, -1).argmax(-1)
    idx_l, idx_r = idx // ww, idx % ww
    grid_x = (torch.arange(ww) % w).to(window0.dtype) - w // 2 + 0.5
    grid_y = (torch.arange(ww) // w).to(window0.dtype) - w // 2 + 0.5
    first0 = coarse_points[0] + torch.stack([grid_x[idx_l], grid_y[idx_l]], -1)
    first1 = coarse_points[1] + torch.stack([grid_x[idx_r], grid_y[idx_r]], -1)
    conf_ff = conf_ff.reshape(count, ww, w + 2, w + 2)
    rows, cols = idx_r // w, idx_r % w
    heat = torch.zeros(count, 3, 3, dtype=window0.dtype)
    for m in range(count):
        for dy in range(3):
            for dx in range(3):
                # The original indexes the 10x10 grid with the 8x8 index plus
                # -1..1; Python wraps -1 to the last row/column.
                heat[m, dy, dx] = conf_ff[m, idx_l[m], rows[m] + dy - 1, cols[m] + dx - 1]
    heat = torch.softmax(heat.reshape(count, 9) / temperature, -1).reshape(count, 3, 3)
    lin = torch.tensor([-1.0, 0.0, 1.0], dtype=window0.dtype)
    expect_x = (heat.sum(1) * lin).sum(-1)
    expect_y = (heat.sum(2) * lin).sum(-1)
    final1 = first1 + torch.stack([expect_x, expect_y], -1)
    return idx_l, idx_r, first0, first1, final1


def run(dtype):
    m = model.to(dtype)
    captured = {}
    original = m._get_matches_from_scores

    def capture(scores):
        captured["confidence"] = scores.detach().clone()
        return original(scores)

    m._get_matches_from_scores = capture
    pv = np.stack([image0, image1])[None, :, None]
    pv = torch.from_numpy(np.repeat(pv, 3, axis=2)).to(dtype)
    with torch.no_grad():
        out = m(pv)
        features = m.efficientloftr(pv).feature_maps
        coarse = features[0]
        _, _, channels, coarse_h, coarse_w = coarse.shape
        windows0, windows1 = m.refinement_layer(coarse / cfg.hidden_size ** .5,
                                                features[1:])
    m._get_matches_from_scores = original
    conf = captured["confidence"].reshape(coarse_h * coarse_w, coarse_h * coarse_w)
    border = cfg.coarse_matching_border_removal
    mask = conf > cfg.coarse_matching_threshold
    mask = mask.reshape(coarse_h, coarse_w, coarse_h, coarse_w)
    mask[:border] = False
    mask[:, :border] = False
    mask[:, :, :border] = False
    mask[:, :, :, :border] = False
    mask[-border:] = False
    mask[:, -border:] = False
    mask[:, :, -border:] = False
    mask[:, :, :, -border:] = False
    mask = mask.reshape(conf.shape)
    mask &= conf == conf.max(1, keepdim=True).values
    mask &= conf == conf.max(0, keepdim=True).values
    i_ids, j_ids = torch.where(mask)
    scores = conf[i_ids, j_ids]
    # transformers' second row: for image-0 cell i, its image-1 match.
    hf_row = out.matches[0, 1]
    for i, j in zip(i_ids.tolist(), j_ids.tolist()):
        assert hf_row[i].item() == j, (i, j, hf_row[i].item())
    assert int((hf_row >= 0).sum()) == len(i_ids)
    scale = HEIGHT / coarse_h
    coarse_points = (
        torch.stack([i_ids % coarse_w, i_ids // coarse_w], -1).to(dtype) * scale,
        torch.stack([j_ids % coarse_w, j_ids // coarse_w], -1).to(dtype) * scale)
    win0 = windows0[0, i_ids]
    win1 = windows1[0, j_ids]
    fine = original_fine(m, win0, win1, coarse_points, cfg.fine_matching_slice_dim,
                         cfg.fine_matching_regress_temperature)
    return dict(confidence=conf, i=i_ids, j=j_ids, scores=scores, win0=win0,
                win1=win1, coarse_points=coarse_points, fine=fine)


ref = run(torch.float64)
check = run(torch.float32)
assert torch.equal(ref["i"], check["i"]) and torch.equal(ref["j"], check["j"])
assert torch.equal(ref["fine"][0], check["fine"][0])
assert torch.equal(ref["fine"][1], check["fine"][1])
count = len(ref["i"])
assert count >= 4, count
print("matches:", count, "max conf diff f32/f64:",
      float((ref["confidence"] - check["confidence"]).abs().max()))

os.makedirs(fixtures, exist_ok=True)
save_file(state, os.path.join(fixtures, name + ".safetensors"))
config_dict = cfg.to_dict()
for key in ("transformers_version", "_attn_implementation_autoset"):
    config_dict.pop(key, None)
config_dict["model_type"] = "efficientloftr"
with open(os.path.join(fixtures, name + "_config.json"), "w") as f:
    json.dump(config_dict, f, indent=2, sort_keys=True)


def f32(t):
    return np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32))


io = {
    "image0": f32(torch.from_numpy(image0)),
    "image1": f32(torch.from_numpy(image1)),
    "confidence": f32(ref["confidence"]),
    "matches": f32(torch.stack([ref["i"], ref["j"]], -1).to(torch.float64)),
    "match_scores": f32(ref["scores"]),
    "window0": f32(ref["win0"]),
    "window1": f32(ref["win1"]),
    "keypoints0": f32(ref["fine"][2]),
    "first_keypoints1": f32(ref["fine"][3]),
    "keypoints1": f32(ref["fine"][4]),
}
save_file(io, os.path.join(fixtures, name + "_io.safetensors"),
          metadata={"width": str(WIDTH), "height": str(HEIGHT)})
print("wrote", name, "fixtures")
