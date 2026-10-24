#!/usr/bin/env python3
"""Generate pico RANDOM EoMT universal-segmentation parity fixtures for
tests/TestNeuralPretrained.pas (TestEoMT*).

No network access: pico transformers EomtForUniversalSegmentation models are
built with random weights, and the reference runs in float64 on exactly the
float32 weights stored.
  tiny_eomt         hidden 16, plain MLP FFN, 3 layers, queries entering before
                    the last 2 (num_blocks 2), 2 register tokens, 32x32 image,
                    patch 8 (4x4 grid), 2 upscale blocks (16x16 masks).
  tiny_eomt_swiglu  SwiGLU FFN, 2 layers, num_blocks 1, 1 register token,
                    24x24 image, patch 8 (3x3 grid), 2 upscale blocks.
attn_mask_probs is stored as zeros, as in the released inference checkpoints
(masked attention is a training-time aid that inference skips at 0).
The io file stores, all float32:
  pixel_values           (H, W, 3) channels-last normalized input
  class_queries_logits   (Q, num_labels + 1)
  masks_queries_logits   (Q, Hm, Wm) at the upscaled grid

The processor_io files pin EomtImageProcessorPil (the PIL backend) on a raw
40 x 28 uint8 image (stored as float32, (H, W, 3)):
  tiny_eomt         do_pad (size 32/32): pixel_values, the float64 forward's
                    logits (float32), post_process_semantic / instance /
                    panoptic maps (H, W), the semantic scores (H, W, C) and
                    segments (N, 3: id, label, score).
  tiny_eomt_swiglu  do_split_image (shortest_edge 24) on a landscape and a
                    portrait image: windows (N, 24, 24, 3), patch_offsets,
                    logits per window, the merged semantic map and scores.

Writes tests/fixtures/tiny_eomt{,_swiglu}{.safetensors,_config.json,
_io.safetensors,_processor_io.safetensors,_preprocessor_config.json}. Usage (from the repo root):
  systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 \\
    ~/x/bin/python -I tools/make_pico_eomt_fixture.py

Coded by Claude (AI).
"""
import math
import os
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import EomtConfig, EomtForUniversalSegmentation
from transformers.models.eomt.image_processing_pil_eomt import EomtImageProcessorPil

here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")


def make_config(variant):
    if variant == "swiglu":
        return EomtConfig(
            hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
            mlp_ratio=2, image_size=24, patch_size=8, num_blocks=1,
            num_queries=4, num_register_tokens=1, num_upscale_blocks=2,
            use_swiglu_ffn=True, num_labels=3, attn_implementation="eager")
    return EomtConfig(
        hidden_size=16, num_hidden_layers=3, num_attention_heads=2,
        mlp_ratio=2, image_size=32, patch_size=8, num_blocks=2,
        num_queries=5, num_register_tokens=2, num_upscale_blocks=2,
        num_labels=4, attn_implementation="eager")


def randomize(model):
    with torch.no_grad():
        for pname, p in model.named_parameters():
            if pname.startswith("embeddings.") and "token" in pname:
                torch.nn.init.normal_(p, mean=0.0, std=1.0)
            elif pname.startswith("query.") or "position_embeddings" in pname:
                torch.nn.init.normal_(p, mean=0.0, std=0.5)
            elif pname.endswith("lambda1"):
                torch.nn.init.normal_(p, mean=0.5, std=0.2)
            elif p.dim() >= 2:
                fan_in = int(np.prod(p.shape[1:]))
                if "upscale_block" in pname and "conv1" in pname:
                    fan_in = p.shape[0]  # ConvTranspose2d weight is [in, out, k, k]
                torch.nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(fan_in))
            elif pname.endswith("weight"):  # LayerNorm scale
                torch.nn.init.normal_(p, mean=1.0, std=0.1)
            else:
                torch.nn.init.normal_(p, mean=0.0, std=0.1)
        model.attn_mask_probs.zero_()


def make_model(variant, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = make_config(variant)
    model = EomtForUniversalSegmentation(cfg).eval()
    randomize(model)
    return cfg, model.float()


def build(variant, seed, name):
    cfg, model = make_model(variant, seed)
    state = {k: v.detach().cpu().numpy().astype(np.float32)
             for k, v in model.state_dict().items()}
    model64 = model.double()
    rng = np.random.RandomState(seed + 100)
    image = rng.randn(cfg.image_size, cfg.image_size, 3).astype(np.float32)
    pixels = torch.from_numpy(image.astype(np.float64).transpose(2, 0, 1)[None].copy())
    with torch.no_grad():
        out = model64(pixel_values=pixels)
    io = {
        "pixel_values": image,
        "class_queries_logits":
            out.class_queries_logits[0].numpy().astype(np.float32),
        "masks_queries_logits":
            out.masks_queries_logits[0].numpy().astype(np.float32),
    }
    save_file(state, os.path.join(fixtures, name + ".safetensors"))
    save_file(io, os.path.join(fixtures, name + "_io.safetensors"))
    with open(os.path.join(fixtures, name + "_config.json"), "w") as f:
        f.write(cfg.to_json_string())
    print(name, "class", tuple(out.class_queries_logits.shape),
          "masks", tuple(out.masks_queries_logits.shape),
          "mask |max|", float(out.masks_queries_logits.abs().max()),
          "weights", len(state))


def raw_test_image():
    """A non-square 40 x 28 (W x H) uint8 RGB image with smooth structure."""
    rng = np.random.RandomState(7)
    y, x = np.mgrid[0:28, 0:40]
    base = np.stack([x * 6.0, y * 9.0, (x + y) * 3.5], axis=-1)
    noisy = base + rng.uniform(-20, 20, size=base.shape)
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def run_model(model64, pixel_values, patch_offsets=None):
    """float64 forward on the processor output; logits cast to float32."""
    with torch.no_grad():
        out = model64(pixel_values=pixel_values.double())
    return SimpleNamespace(
        class_queries_logits=out.class_queries_logits.float(),
        masks_queries_logits=out.masks_queries_logits.float(),
        patch_offsets=patch_offsets)


def segments_array(segments):
    """(N, 3) float32 rows of id, label_id, score (empty -> (0, 3))."""
    rows = [[s["id"], s["label_id"], s["score"]] for s in segments]
    return np.array(rows, dtype=np.float32).reshape(-1, 3)


def scores_channels_last(semantic):
    """segmentation_scores (C, H, W) as (H, W, C) float32."""
    return semantic.segmentation_scores.numpy().transpose(1, 2, 0).astype(np.float32).copy()


def channels_last(pixel_values):
    return pixel_values.numpy().transpose(0, 2, 3, 1).astype(np.float32).copy()


def build_padded_processor_case(variant, seed, name):
    """do_pad processor (panoptic / instance checkpoints) and the semantic,
    instance and panoptic post-processors on the logits of its output."""
    cfg, model = make_model(variant, seed)
    model64 = model.double()
    size = cfg.image_size
    processor = EomtImageProcessorPil(
        size={"shortest_edge": size, "longest_edge": size}, do_pad=True)
    image = raw_test_image()
    target = [image.shape[:2]]
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
    out = run_model(model64, pixel_values)
    semantic = processor.post_process_semantic_segmentation(
        out, target_sizes=target, return_segmentation_scores=True)[0]
    # thresholds the random pico weights pass: about half of the queries.
    probs = out.class_queries_logits[0].softmax(-1)
    instance_threshold = float(probs[:, :-1].max(-1).values.median()) * 0.5
    instance = processor.post_process_instance_segmentation(
        out, target_sizes=target, threshold=instance_threshold)[0]
    panoptic_threshold = float(probs.max(-1).values.min()) * 0.99
    # label 3 is the best class of pico queries 3 and 4, so both segments of
    # that stuff class merge; the low overlap threshold keeps the small ones.
    stuff_classes = [3]
    panoptic = processor.post_process_panoptic_segmentation(
        out, target_sizes=target, threshold=panoptic_threshold,
        mask_threshold=0.5, overlap_mask_area_threshold=0.02,
        stuff_classes=stuff_classes)[0]
    io = {
        "raw_image": image.astype(np.float32),
        "pixel_values": channels_last(pixel_values)[0],
        "class_queries_logits": out.class_queries_logits[0].numpy(),
        "masks_queries_logits": out.masks_queries_logits[0].numpy(),
        "semantic": semantic.segmentation.numpy().astype(np.float32),
        "semantic_scores": scores_channels_last(semantic),
        "instance_threshold": np.array([instance_threshold], dtype=np.float32),
        "instance": instance["segmentation"].numpy().astype(np.float32),
        "instance_segments": segments_array(instance["segments_info"]),
        "panoptic_threshold": np.array([panoptic_threshold], dtype=np.float32),
        "panoptic_stuff_classes": np.array(stuff_classes, dtype=np.float32),
        "panoptic_mask_threshold": np.array([0.5], dtype=np.float32),
        "panoptic_overlap_threshold": np.array([0.02], dtype=np.float32),
        "panoptic": panoptic["segmentation"].numpy().astype(np.float32),
        "panoptic_segments": segments_array(panoptic["segments_info"]),
    }
    save_file(io, os.path.join(fixtures, name + "_processor_io.safetensors"))
    with open(os.path.join(fixtures, name + "_preprocessor_config.json"), "w") as f:
        f.write(processor.to_json_string())
    print(name, "padded", tuple(pixel_values.shape),
          "instance segments", len(instance["segments_info"]),
          "panoptic segments", len(panoptic["segments_info"]))


def build_split_processor_case(variant, seed, name):
    """do_split_image processor (ADE20K semantic checkpoints) on a landscape and
    a portrait image, and post_process_semantic_segmentation with the patch
    offsets (merge_image_patches)."""
    cfg, model = make_model(variant, seed)
    model64 = model.double()
    processor = EomtImageProcessorPil(
        size={"shortest_edge": cfg.image_size, "longest_edge": None},
        do_split_image=True)
    io = {}
    for orientation, image in (("landscape", raw_test_image()),
                               ("portrait", raw_test_image().transpose(1, 0, 2).copy())):
        encoded = processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"]
        offsets = encoded["patch_offsets"]
        out = run_model(model64, pixel_values, offsets)
        semantic = processor.post_process_semantic_segmentation(
            out, target_sizes=[image.shape[:2]], return_segmentation_scores=True)[0]
        io[orientation + "_raw_image"] = image.astype(np.float32)
        io[orientation + "_pixel_values"] = channels_last(pixel_values)
        io[orientation + "_patch_offsets"] = torch.stack(offsets).numpy().astype(np.float32)
        io[orientation + "_class_queries_logits"] = out.class_queries_logits.numpy()
        io[orientation + "_masks_queries_logits"] = out.masks_queries_logits.numpy()
        io[orientation + "_semantic"] = semantic.segmentation.numpy().astype(np.float32)
        io[orientation + "_semantic_scores"] = scores_channels_last(semantic)
        print(name, orientation, "windows", tuple(pixel_values.shape),
              "offsets", [o.tolist() for o in offsets])
    save_file(io, os.path.join(fixtures, name + "_processor_io.safetensors"))
    with open(os.path.join(fixtures, name + "_preprocessor_config.json"), "w") as f:
        f.write(processor.to_json_string())


build("mlp", 1, "tiny_eomt")
build("swiglu", 2, "tiny_eomt_swiglu")
build_padded_processor_case("mlp", 1, "tiny_eomt")
build_split_processor_case("swiglu", 2, "tiny_eomt_swiglu")
