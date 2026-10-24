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

Writes tests/fixtures/tiny_eomt{,_swiglu}{.safetensors,_config.json,
_io.safetensors}. Usage (from the repo root):
  systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 \\
    ~/x/bin/python -I tools/make_pico_eomt_fixture.py

Coded by Claude (AI).
"""
import math
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import EomtConfig, EomtForUniversalSegmentation

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


def build(variant, seed, name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = make_config(variant)
    model = EomtForUniversalSegmentation(cfg).eval()
    randomize(model)
    model = model.float()
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


build("mlp", 1, "tiny_eomt")
build("swiglu", 2, "tiny_eomt_swiglu")
