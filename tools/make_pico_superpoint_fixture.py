#!/usr/bin/env python3
"""Generate a pico RANDOM SuperPoint keypoint-detection parity fixture for
tests/TestNeuralPretrained.pas (TestSuperPoint*).

No network access: a pico transformers SuperPointForKeypointDetection is built
from a small config (4 encoder blocks, so the cell stays 8 and the score head
keeps 65 channels) and randomly initialized. The reference is the real
transformers forward in float64. Per case the oracle stores the post-process
parameters (nms_radius, keypoint_threshold, border_removal_distance,
max_keypoints), the input, the dense pre-NMS score map, the dense normalized
descriptor map, and the keypoints (pixel x, y), scores and descriptors.
A preprocessing case runs SuperPointImageProcessorPil (resize, rescale,
grayscale) on a random RGB byte image.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_superpoint_fixture.py
writes tests/fixtures/tiny_superpoint{.safetensors,_config.json,_io.json}.
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import SuperPointConfig, SuperPointForKeypointDetection
from transformers.models.superpoint.image_processing_pil_superpoint import (
    SuperPointImageProcessorPil,
)

WIDTH = 40
HEIGHT = 32
ENCODER = [4, 4, 6, 6]
DECODER_HIDDEN = 8
DESCRIPTOR_DIM = 8

torch.manual_seed(2026)
np.random.seed(2026)

cfg = SuperPointConfig(
    encoder_hidden_sizes=ENCODER,
    decoder_hidden_size=DECODER_HIDDEN,
    keypoint_decoder_dim=65,
    descriptor_decoder_dim=DESCRIPTOR_DIM,
    keypoint_threshold=0.005,
    max_keypoints=-1,
    nms_radius=4,
    border_removal_distance=4,
)
model = SuperPointForKeypointDetection(cfg).eval()

# O(1) activations: fan-in scaled weights; a sharper score head so the softmax
# is peaky and the threshold / NMS decisions are not borderline.
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.dim() == 4:
            fan_in = p.shape[1] * p.shape[2] * p.shape[3]
            torch.nn.init.normal_(p, mean=0.0, std=1.6 / np.sqrt(fan_in))
        else:
            torch.nn.init.normal_(p, mean=0.0, std=0.1)
    model.keypoint_decoder.conv_score_b.weight.mul_(1.5)

model_d = model.double()
kd = model.keypoint_decoder
dd = model.descriptor_decoder


def dense_maps(pixel_values):
    """Pre-NMS full-resolution score map (H, W) and normalized descriptor map
    (D, Hc, Wc), recomputed with the model's own modules."""
    with torch.no_grad():
        enc = model_d.encoder(pixel_values[:, :1]).last_hidden_state
        s = kd.conv_score_b(kd.relu(kd.conv_score_a(enc)))
        s = torch.nn.functional.softmax(s, 1)[:, :-1]
        b, _, h, w = s.shape
        s = s.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        s = s.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        d = dd.conv_descriptor_b(dd.relu(dd.conv_descriptor_a(enc)))
        d = torch.nn.functional.normalize(d, p=2, dim=1)
    return s[0].numpy(), d[0].numpy()


def run_case(gray, nms_radius, threshold, border, max_keypoints):
    kd.nms_radius = nms_radius
    kd.keypoint_threshold = threshold
    kd.border_removal_distance = border
    kd.max_keypoints = max_keypoints
    pv = torch.from_numpy(np.repeat(gray[None, None], 3, axis=1))
    with torch.no_grad():
        out = model_d(pv)
    n = int(out.mask[0].sum())
    kps = out.keypoints[0, :n].numpy() * np.array([WIDTH, HEIGHT])
    score_map, desc_map = dense_maps(pv)
    return {
        "nms_radius": nms_radius,
        "keypoint_threshold": threshold,
        "border_removal_distance": border,
        "max_keypoints": max_keypoints,
        "input": gray.reshape(-1).tolist(),  # (y, x)
        "score_map": score_map.reshape(-1).tolist(),  # (y, x)
        # (yc, xc, d): depth-contiguous like a TNNetVolume.
        "descriptor_map": np.transpose(desc_map, (1, 2, 0)).reshape(-1).tolist(),
        "keypoints": np.rint(kps).astype(int).reshape(-1).tolist(),  # x0,y0,..
        "scores": out.scores[0, :n].numpy().tolist(),
        "descriptors": out.descriptors[0, :n].numpy().reshape(-1).tolist(),
    }


# ---------------- preprocessing (PIL backend) ----------------
SRC_W, SRC_H = 23, 17
rgb = np.random.randint(0, 256, size=(SRC_H, SRC_W, 3), dtype=np.uint8)
proc = SuperPointImageProcessorPil(do_grayscale=True,
                                   size={"height": HEIGHT, "width": WIDTH})
pixel_values = proc(rgb, return_tensors="np")["pixel_values"][0]  # (3, H, W)
gray = pixel_values[0].astype(np.float64)
preprocess = {
    "source_width": SRC_W,
    "source_height": SRC_H,
    "source": rgb.reshape(-1).tolist(),  # (y, x, c) bytes
    "output": pixel_values[0].reshape(-1).tolist(),  # (y, x)
}

cases = [
    # End to end from the preprocessed image, all keypoints kept (row-major).
    run_case(gray, 4, 0.005, 4, -1),
    # Random input, smaller NMS, top-k (descending score order).
    run_case(np.random.rand(HEIGHT, WIDTH), 2, 0.01, 2, 12),
    # No border removal and a max_keypoints above the count.
    run_case(np.random.rand(HEIGHT, WIDTH), 1, 0.02, 0, 1000),
]
for c in cases:
    assert len(c["scores"]) > 3, len(c["scores"])
    # Distinct scores keep the top-k order defined.
    assert len(set(np.float32(c["scores"]))) == len(c["scores"]), c["scores"]

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

tensors = {k: v.detach().cpu().numpy().astype(np.float32)
           for k, v in model.float().state_dict().items()}
save_file(tensors, os.path.join(fixtures, "tiny_superpoint.safetensors"))

config = {
    "model_type": "superpoint",
    "encoder_hidden_sizes": ENCODER,
    "decoder_hidden_size": DECODER_HIDDEN,
    "keypoint_decoder_dim": 65,
    "descriptor_decoder_dim": DESCRIPTOR_DIM,
    "keypoint_threshold": 0.005,
    "max_keypoints": -1,
    "nms_radius": 4,
    "border_removal_distance": 4,
}
with open(os.path.join(fixtures, "tiny_superpoint_config.json"), "w") as f:
    json.dump(config, f, indent=2)
with open(os.path.join(fixtures, "tiny_superpoint_io.json"), "w") as f:
    json.dump({"width": WIDTH, "height": HEIGHT, "preprocess": preprocess,
               "cases": cases}, f)

print("wrote tiny_superpoint.{safetensors,_config.json,_io.json} to", fixtures)
print("  tensors:", sorted(tensors))
print("  keypoints per case:", [len(c["scores"]) for c in cases])
print("  max x per case:", [max(c["keypoints"][0::2]) for c in cases])
print("  score range per case:", [(min(c["scores"]), max(c["scores"])) for c in cases])
