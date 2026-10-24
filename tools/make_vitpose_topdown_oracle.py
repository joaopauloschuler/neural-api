#!/usr/bin/env python3
"""Oracle for the top-down ViTPose pipeline test in tests/TestNeuralPretrained.pas.

Loads the committed pico model (tests/fixtures/tiny_vitpose.safetensors, made by
make_pico_vitpose_fixture.py) into transformers' VitPoseForPoseEstimation, runs
VitPoseImageProcessor (box -> center/scale, affine crop, rescale, normalize) on a
small random uint8 photo with two NON-SQUARE person boxes (one tall, one wide and
partly outside the photo), and post_process_pose_estimation (argmax, DARK
refinement, transform_preds) back to photo pixels.

Writes tests/fixtures/tiny_vitpose_topdown.json:
  image: flat (y, x, c) uint8 values of the photo
  boxes[i]: box (COCO x, y, w, h), center, scale, pixel_sum / pixel_abs_sum
  (checksums of the normalized crop), keypoints [[x, y], ...], scores [...].

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python -I tools/make_vitpose_topdown_oracle.py
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import (
    VitPoseConfig,
    VitPoseForPoseEstimation,
    VitPoseImageProcessorPil,
)
from transformers.models.vitpose.image_processing_pil_vitpose import box_to_center_and_scale

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")

with open(os.path.join(FIXTURES, "tiny_vitpose_config.json")) as f:
    cfg = VitPoseConfig(**json.load(f))
model = VitPoseForPoseEstimation(cfg).eval()
model.load_state_dict(load_file(os.path.join(FIXTURES, "tiny_vitpose.safetensors")))

input_h, input_w = cfg.backbone_config.image_size
processor = VitPoseImageProcessorPil(size={"height": input_h, "width": input_w})

IMAGE_W, IMAGE_H = 20, 26
rng = np.random.default_rng(20261023)
image = rng.integers(0, 256, size=(IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
boxes = [[3.3, 2.7, 9.4, 17.1], [6.2, 9.1, 13.7, 11.3]]

inputs = processor(images=image, boxes=[boxes], return_tensors="pt")
with torch.no_grad():
    outputs = model(inputs["pixel_values"])
results = processor.post_process_pose_estimation(outputs, boxes=[boxes])[0]

box_records = []
for box_pos, box in enumerate(boxes):
    center, scale = box_to_center_and_scale(
        box, image_width=input_w, image_height=input_h)
    pixels = inputs["pixel_values"][box_pos].numpy().astype(np.float64)
    box_records.append({
        "box": box,
        "center": [float(v) for v in center],
        "scale": [float(v) for v in scale],
        "pixel_sum": float(pixels.sum()),
        "pixel_abs_sum": float(np.abs(pixels).sum()),
        "keypoints": [[float(v) for v in kp] for kp in results[box_pos]["keypoints"]],
        "scores": [float(v) for v in results[box_pos]["scores"]],
    })

with open(os.path.join(FIXTURES, "tiny_vitpose_topdown.json"), "w") as f:
    json.dump({"image_w": IMAGE_W, "image_h": IMAGE_H,
               "image": image.reshape(-1).tolist(), "boxes": box_records}, f)

for record in box_records:
    print(record["box"], record["center"], record["scale"])
    print("  keypoints", record["keypoints"])
    print("  scores", record["scores"])
