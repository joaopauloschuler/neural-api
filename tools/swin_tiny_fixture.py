#!/usr/bin/env python3
"""Generate a tiny RANDOM Swin-Transformer parity fixture for
tests/TestNeuralPretrained.pas, using the real HF SwinForImageClassification as
the float64 oracle (Swin IS shipped in transformers).

Why HF-as-oracle (unlike the ResNet/MobileNetV3 fixtures, which hand-roll a
numpy oracle because torchvision is absent): the Swin forward path is intricate
(patch-merging, window partition + cyclic shift + relative-position bias) and
exactly mirroring it in numpy would be error-prone. transformers HAS Swin, so we
build a pico random-init SwinForImageClassification, save its safetensors +
config.json, and compute the reference logits by running the SAME model cast to
float64. CAI loads the float32 weights and must reproduce the logits < 1e-4.

The pico config deliberately exercises the FULL shifted-window path:
  image=8, patch=2  -> stage-0 token grid 4x4
  window=2          -> 4 windows per stage, shift_size = 1
  depths=[2,2]      -> stage 0 has an EVEN block (W-MSA, no shift) AND an ODD
                       block (SW-MSA: cyclic shift + attention mask)
  num_heads=[2,4]   -> multi-head, head_dim 2 then 3
  patch-merging between stage 0 and stage 1 (grid 4x4 -> 2x2, dim 4 -> 8)
  stage 1 grid 2x2 == window 2 -> HF clamps shift to 0 (W-MSA only)

The relative_position_bias_table parameters are re-randomized to non-trivial
values so the relative-position bias (and, in the shifted block, its interaction
with the cyclic-shift mask) actually moves the logits.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/swin_tiny_fixture.py
writes tests/fixtures/tiny_swin{.safetensors,_config.json,_logits.json}.
Needs torch + transformers + safetensors + numpy.
"""
import json
import numpy as np
import torch
from transformers import SwinConfig, SwinForImageClassification
from safetensors.torch import save_file

IMAGE = 8
PATCH = 2
NUM_CHANNELS = 3
EMBED_DIM = 4
DEPTHS = [2, 2]
NUM_HEADS = [2, 4]
WINDOW = 2
MLP_RATIO = 2.0
NUM_LABELS = 5
LN_EPS = 1e-5


def main():
    torch.manual_seed(20260614)
    cfg = SwinConfig(
        image_size=IMAGE, patch_size=PATCH, num_channels=NUM_CHANNELS,
        embed_dim=EMBED_DIM, depths=DEPTHS, num_heads=NUM_HEADS,
        window_size=WINDOW, mlp_ratio=MLP_RATIO, qkv_bias=True,
        hidden_act="gelu", layer_norm_eps=LN_EPS, num_labels=NUM_LABELS,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
    )
    model = SwinForImageClassification(cfg).eval()

    # Re-randomize the relative-position bias tables (default init is zero, which
    # would make the rel-pos-bias path vacuous) AND the LayerNorms / classifier
    # bias so every branch measurably affects the logits.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "relative_position_bias_table" in name:
                p.normal_(std=0.6)
            elif name.endswith("layernorm.bias") or name.endswith(
                    "layernorm_before.bias") or name.endswith(
                    "layernorm_after.bias") or name.endswith("norm.bias"):
                p.normal_(std=0.3)
            elif name == "classifier.bias":
                p.normal_(std=0.2)

    # Pinned deterministic input pixels (float32, dyadic so exact in f32 + JSON).
    pixels = np.zeros((NUM_CHANNELS, IMAGE, IMAGE), dtype=np.float64)
    for c in range(NUM_CHANNELS):
        for y in range(IMAGE):
            for x in range(IMAGE):
                pixels[c, y, x] = (((c * 64 + y * 8 + x) * 5) % 17 - 8) / 8.0

    # Round the state_dict through float32 (what CAI loads), then build a float64
    # copy of the model for the oracle so the reference is computed from the SAME
    # f32 weights at double precision.
    # SwinForImageClassification nests the backbone under ".swin"; strip that
    # prefix so the saved keys are the bare names our importer expects
    # (embeddings.*, encoder.layers.*, layernorm.*, classifier.*).
    def strip(k):
        return k[len("swin."):] if k.startswith("swin.") else k

    sd_f32 = {strip(k): v.detach().to(torch.float32)
              for k, v in model.state_dict().items()}

    def unstrip(k):
        # re-add the .swin prefix for everything except the classifier head
        return k if k.startswith("classifier.") else "swin." + k

    # float64 oracle forward
    model64 = SwinForImageClassification(cfg).eval().to(torch.float64)
    model64.load_state_dict(
        {unstrip(k): v.to(torch.float64) for k, v in sd_f32.items()})
    x64 = torch.tensor(pixels[None], dtype=torch.float64)
    with torch.no_grad():
        logits = model64(x64).logits[0].numpy()
    print("logits:", logits.tolist())

    # Save f32 safetensors (drop non-persistent buffers; state_dict already
    # excludes relative_position_index since it is registered non-persistent).
    save_file(sd_f32, "tests/fixtures/tiny_swin.safetensors")

    config = {
        "model_type": "swin",
        "image_size": IMAGE,
        "patch_size": PATCH,
        "num_channels": NUM_CHANNELS,
        "embed_dim": EMBED_DIM,
        "depths": DEPTHS,
        "num_heads": NUM_HEADS,
        "window_size": WINDOW,
        "mlp_ratio": MLP_RATIO,
        "qkv_bias": True,
        "hidden_act": "gelu",
        "layer_norm_eps": LN_EPS,
        "num_labels": NUM_LABELS,
    }
    with open("tests/fixtures/tiny_swin_config.json", "w") as f:
        json.dump(config, f, indent=1)
    with open("tests/fixtures/tiny_swin_logits.json", "w") as f:
        json.dump({
            "pixels": pixels.tolist(),         # [C][IMAGE][IMAGE]
            "logits": [logits.tolist()],       # [1][NUM_LABELS]
            "num_labels": NUM_LABELS,
        }, f)
    print(f"wrote tiny_swin.safetensors ({len(sd_f32)} tensors) + config + oracle")
    for k in sorted(sd_f32):
        print(f"  {k} {list(sd_f32[k].shape)}")

    # ---- fixture self-checks: each path must move the logits ----
    base = logits.copy()

    def effect(key):
        alt = {k: v.clone().to(torch.float64) for k, v in sd_f32.items()}
        alt[key] = torch.zeros_like(alt[key])
        m = SwinForImageClassification(cfg).eval().to(torch.float64)
        m.load_state_dict({unstrip(k): v for k, v in alt.items()})
        with torch.no_grad():
            return float(np.abs(m(x64).logits[0].numpy() - base).max())

    TH = 1e-6
    checks = [
        ("embeddings.patch_embeddings.projection.weight", "patch conv"),
        ("encoder.layers.0.blocks.0.attention.relative_position_bias."
         "relative_position_bias_table", "stage0 blk0 rel-pos bias (W-MSA)"),
        ("encoder.layers.0.blocks.1.attention.relative_position_bias."
         "relative_position_bias_table", "stage0 blk1 rel-pos bias (SW-MSA)"),
        ("encoder.layers.0.downsample.reduction.weight", "patch merging"),
        ("encoder.layers.1.blocks.0.attention.q_proj.weight", "stage1 q_proj"),
        ("layernorm.weight", "final layernorm"),
        ("classifier.bias", "classifier bias"),
    ]
    for key, label in checks:
        d = effect(key)
        assert d > TH, f"{label} had no effect ({d})"
        print(f"{label} effect: {d:.6f}")
    print("all fixture self-checks passed")


if __name__ == "__main__":
    main()
