#!/usr/bin/env python3
"""Generate pico RANDOM RT-DETR object-detection parity fixtures for
tests/TestNeuralPretrained.pas (TestRtDetr*).

No network access: pico transformers RTDetrForObjectDetection and
RTDetrV2ForObjectDetection models are built with random weights and random
BatchNorm statistics, and the reference runs in float64 on exactly the float32
weights stored.
  tiny_rtdetr      ResNet-vd "basic" blocks (the rtdetr_r18vd layout), 3 feature
                   levels, 2 decoder layers, 3 sampling points.
  tiny_rtdetr_r50  ResNet-vd "bottleneck" blocks (the rtdetr_r50vd layout), a
                   4th decoder level from the 3x3/2 extra projection, 2 AIFI
                   encoder layers, hidden_expansion 0.5 (the CSP conv3).
  tiny_rtdetr_v2   rt_detr_v2 with the tiny_rtdetr layout and decoder_offset_scale
                   0.7 (bilinear decoder_method "default").
  tiny_rtdetr_v2_discrete  tiny_rtdetr_v2 with decoder_method "discrete"; the
                   script also asserts that every sampled loc * (W, H) + 0.5 is
                   at least 1e-3 from an integer, so a float32 rerun picks the
                   same cells.
The io file stores the input image (H, W, 3) in 0..1, the logits (Q, C), the
pred_boxes (Q, 4), the encoder top-K indices, and post_process_object_detection
(threshold 0.75, target size 480 x 640) scores, labels and xyxy boxes.
The script asserts that every top-K decision (encoder query selection and the
post-process ranking) has a float64 margin above 1e-4, so a float32 rerun ranks
the same way.

Writes tests/fixtures/tiny_rtdetr{,_r50,_v2,_v2_discrete}{.safetensors,
_config.json,_io.safetensors}; fixture names given as arguments limit the run
to those. Usage (from the repo root):
  systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 \\
    ~/x/bin/python -I tools/make_pico_rtdetr_fixture.py [tiny_rtdetr_v2]

Coded by Claude (AI).
"""
import json
import math
import os
import sys

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import (RTDetrConfig, RTDetrForObjectDetection,
                          RTDetrImageProcessor, RTDetrResNetConfig,
                          RTDetrV2Config, RTDetrV2ForObjectDetection)
from transformers.models.rt_detr_v2 import modeling_rt_detr_v2

here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
SIZE = 64
THRESHOLD = 0.75
TARGET_H, TARGET_W = 480, 640
DISCRETE_MARGIN = 1e-3

# The smallest distance of a discrete sampling position loc * (W, H) + 0.5 from
# an integer over the last forward (inf when nothing was sampled discretely).
discrete_margin = [math.inf]
_core = modeling_rt_detr_v2.multi_scale_deformable_attention_v2


def recording_core(value, shapes, locations, weights, points, method="default"):
    if method == "discrete":
        for level, grid in enumerate(locations.split(points, dim=-2)):
            height, width = shapes[level]
            scale = torch.tensor([width, height], dtype=grid.dtype)
            position = (grid * scale + 0.5).flatten(0, -2)
            # Only positions in [0.5, size) can round to a different cell.
            for axis, size in ((0, width), (1, height)):
                coord = position[:, axis]
                coord = coord[(coord >= 0.5) & (coord < size)]
                if coord.numel():
                    gap = (coord - coord.round()).abs().min().item()
                    discrete_margin[0] = min(discrete_margin[0], gap)
    return _core(value, shapes, locations, weights, points, method)


modeling_rt_detr_v2.multi_scale_deformable_attention_v2 = recording_core


def make_config(variant):
    if variant in ("basic", "v2", "v2_discrete"):
        backbone = RTDetrResNetConfig(
            embedding_size=8, hidden_sizes=[8, 16, 24, 32], depths=[1, 2, 1, 1],
            layer_type="basic", out_indices=[2, 3, 4])
        if variant == "basic":
            config_class, extra = RTDetrConfig, {}
        else:
            config_class = RTDetrV2Config
            extra = dict(decoder_n_levels=3, decoder_offset_scale=0.7,
                         decoder_method="discrete" if variant == "v2_discrete"
                         else "default")
        return config_class(
            backbone_config=backbone, encoder_in_channels=[16, 24, 32],
            encoder_hidden_dim=16, encoder_ffn_dim=32, encoder_attention_heads=2,
            encoder_layers=1, d_model=16, decoder_in_channels=[16, 16, 16],
            decoder_ffn_dim=32, num_feature_levels=3, decoder_n_points=3,
            decoder_layers=2, decoder_attention_heads=2, num_queries=10,
            num_labels=5, attn_implementation="eager", **extra)
    backbone = RTDetrResNetConfig(
        embedding_size=8, hidden_sizes=[16, 24, 32, 48], depths=[1, 2, 1, 1],
        layer_type="bottleneck", out_indices=[2, 3, 4])
    return RTDetrConfig(
        backbone_config=backbone, encoder_in_channels=[24, 32, 48],
        encoder_hidden_dim=16, encoder_ffn_dim=24, encoder_attention_heads=4,
        encoder_layers=2, d_model=16, decoder_in_channels=[16, 16, 16],
        decoder_ffn_dim=24, num_feature_levels=4, decoder_n_points=2,
        hidden_expansion=0.5,
        decoder_layers=3, decoder_attention_heads=4, num_queries=12,
        num_labels=6, attn_implementation="eager")


def randomize(model):
    with torch.no_grad():
        for pname, p in model.named_parameters():
            if p.dim() >= 2:
                fan_in = int(np.prod(p.shape[1:]))
                torch.nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(fan_in))
            elif pname.endswith("weight"):  # BatchNorm / LayerNorm scale
                torch.nn.init.normal_(p, mean=1.0, std=0.1)
            else:
                torch.nn.init.normal_(p, mean=0.0, std=0.1)
        # Sampling offsets of about one level cell, so some samples leave the
        # map (zero padding) and most fall between pixels.
        for layer in model.model.decoder.layers:
            layer.encoder_attn.sampling_offsets.weight.mul_(3.0)
            torch.nn.init.normal_(layer.encoder_attn.sampling_offsets.bias, 0, 1.0)
        for bname, b in model.named_buffers():
            if bname.endswith("running_mean"):
                b.copy_(torch.randn_like(b) * 0.1)
            elif bname.endswith("running_var"):
                b.copy_(torch.rand_like(b) + 0.5)
            elif bname.endswith(".weight") and "backbone" in bname:
                b.copy_(torch.randn_like(b) * 0.1 + 1.0)
            elif bname.endswith(".bias") and "backbone" in bname:
                b.copy_(torch.randn_like(b) * 0.1)


def min_gap(values, count):
    ordered = np.sort(values)[::-1]
    return float(np.min(ordered[:count] - ordered[1:count + 1]))


def build(variant, seed, name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = make_config(variant)
    if variant.startswith("v2"):
        model = RTDetrV2ForObjectDetection(cfg).eval()
    else:
        model = RTDetrForObjectDetection(cfg).eval()
    randomize(model)
    model = model.float()
    state = {k: v.detach().cpu().numpy().astype(np.float32)
             for k, v in model.state_dict().items()
             if "num_batches_tracked" not in k}
    model64 = model.double()
    rng = np.random.RandomState(seed + 100)
    image = rng.rand(SIZE, SIZE, 3)
    pixels = torch.from_numpy(image.transpose(2, 0, 1)[None].copy())
    with torch.no_grad():
        discrete_margin[0] = math.inf
        out = model64(pixel_values=pixels)
    if discrete_margin[0] < DISCRETE_MARGIN:
        return None
    enc_scores = out.enc_outputs_class[0].max(-1).values.numpy()
    gap = min_gap(enc_scores, cfg.num_queries)
    if gap < 1e-4:
        return None
    scores = torch.sigmoid(out.logits[0]).flatten().numpy()
    post_gap = min_gap(scores, cfg.num_queries)
    if post_gap < 1e-4:
        return None
    if np.min(np.abs(scores - THRESHOLD)) < 1e-3:
        return None
    _, topk = torch.topk(out.enc_outputs_class[0].max(-1).values,
                         cfg.num_queries)
    processor = RTDetrImageProcessor()
    post = processor.post_process_object_detection(
        out, threshold=THRESHOLD, target_sizes=[(TARGET_H, TARGET_W)])[0]
    if not 2 <= len(post["scores"]) < cfg.num_queries:
        return None
    io = {
        "image": image.astype(np.float32),
        "logits": out.logits[0].numpy().astype(np.float32),
        "pred_boxes": out.pred_boxes[0].numpy().astype(np.float32),
        "enc_topk": topk.numpy().astype(np.float32),
        "post_scores": post["scores"].numpy().astype(np.float32),
        "post_labels": post["labels"].numpy().astype(np.float32),
        "post_boxes": post["boxes"].numpy().astype(np.float32),
        "post_threshold": np.array([THRESHOLD], dtype=np.float32),
        "post_target_size": np.array([TARGET_H, TARGET_W], dtype=np.float32),
    }
    save_file(state, os.path.join(fixtures, name + ".safetensors"))
    save_file(io, os.path.join(fixtures, name + "_io.safetensors"))
    with open(os.path.join(fixtures, name + "_config.json"), "w") as f:
        f.write(cfg.to_json_string())
    print(name, "seed", seed, "enc gap", gap, "post gap", post_gap,
          "discrete margin", discrete_margin[0],
          "detections", len(post["scores"]), "weights", len(state))
    return name


FIXTURES = (("basic", "tiny_rtdetr"), ("bottleneck", "tiny_rtdetr_r50"),
            ("v2", "tiny_rtdetr_v2"), ("v2_discrete", "tiny_rtdetr_v2_discrete"))
for variant, name in FIXTURES:
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    for seed in range(1, 60):
        if build(variant, seed, name):
            break
    else:
        raise SystemExit("no seed with clear top-K margins for " + name)
