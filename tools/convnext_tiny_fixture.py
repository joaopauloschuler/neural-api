#!/usr/bin/env python3
"""Generate tiny RANDOM ConvNeXt v1 (LayerScale) and v2 (GRN) parity fixtures
for tests/TestNeuralPretrained.pas.

The reference logits are computed with the REAL HuggingFace
ConvNext(V2)ForImageClassification forward (transformers is in the reusable
venv at /home/bpsa/x), so the test is a true parity check of the importer
against HF.

Architecture (pico, KB-scale): patch_size 4, image 32, depths [1,1,2,1],
hidden_sizes [8,16,24,32], 5 labels. The four stages exercise the stem
patchify, all three downsample transitions, and a 2-block stage.

GOTCHA (ModernBERT fixture note): HF's std-0.02 random init makes the
LayerNorm / GRN outputs nearly vacuous (everything ~0), so re-randomise the
conv / linear / norm parameters at O(1) scale and set the LayerScale gamma
(v1) and GRN gamma/bias (v2) to non-trivial values, otherwise the parity
test would pass even with a broken GRN / LayerScale.

The CAI input volume is channels-last interleaved FData[(y*W+x)*C+c]; the
test reads "pixels" as [C][H][W] and feeds the HF model pixel_values
(1,C,H,W) of the SAME array, so the two see identical pixels.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/convnext_tiny_fixture.py
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

# NOTE on geometry: CAI convolutions CLAMP the kernel to the input spatial
# size (TNNetConvolutionAbstract.SetPrevLayer: FeatureSize = Min(FeatureSize,
# InputSize)), unlike PyTorch which keeps the 7x7 kernel and pads. So for a
# faithful 7x7-depthwise parity the deepest stage must stay >= 7 wide. The
# four stages reduce the image by patch_size (stem) then /2 three times. With
# patch_size 1 and image 56 the stages are 56,28,14,7 - all >= 7 - which keeps
# the pixel array (and thus the committed _logits.json) KB-scale while still
# exercising every downsample transition and the real 7x7 depthwise.
IMAGE = 56
PATCH = 1
DEPTHS = [1, 1, 2, 1]
DIMS = [4, 4, 8, 8]
NUM_LABELS = 5
NUM_CHANNELS = 3


def reinit(model, rng):
    """Re-randomise every parameter at O(1) scale (see module docstring)."""
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            p.data = torch.tensor(
                rng.standard_normal(tuple(p.shape)) * 0.5, dtype=torch.float32)
        else:
            # 1-D: norms gamma ~ 1, biases / GRN ~ small but non-trivial.
            if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
                p.data = torch.tensor(
                    1.0 + rng.standard_normal(tuple(p.shape)) * 0.2,
                    dtype=torch.float32)
            else:
                p.data = torch.tensor(
                    rng.standard_normal(tuple(p.shape)) * 0.3,
                    dtype=torch.float32)


def emit(version):
    from transformers import (ConvNextConfig, ConvNextForImageClassification,
                              ConvNextV2Config,
                              ConvNextV2ForImageClassification)
    rng = np.random.default_rng(20260614 + version)
    if version == 1:
        cfg = ConvNextConfig(
            num_channels=NUM_CHANNELS, patch_size=PATCH, num_stages=4,
            hidden_sizes=DIMS, depths=DEPTHS, hidden_act="gelu",
            layer_scale_init_value=1e-6, image_size=IMAGE,
            num_labels=NUM_LABELS)
        model = ConvNextForImageClassification(cfg).eval()
        model_type = "convnext"
        prefix = "convnext"
        base = "tiny_convnext"
    else:
        cfg = ConvNextV2Config(
            num_channels=NUM_CHANNELS, patch_size=PATCH, num_stages=4,
            hidden_sizes=DIMS, depths=DEPTHS, hidden_act="gelu",
            image_size=IMAGE, num_labels=NUM_LABELS)
        model = ConvNextV2ForImageClassification(cfg).eval()
        model_type = "convnextv2"
        prefix = "convnextv2"
        base = "tiny_convnextv2"

    reinit(model, rng)
    # Make LayerScale (v1) / GRN gamma+bias (v2) non-trivial so the test
    # actually exercises them.
    sd = model.state_dict()
    for k in list(sd.keys()):
        if k.endswith("layer_scale_parameter"):
            sd[k] = torch.tensor(
                0.5 + rng.standard_normal(tuple(sd[k].shape)) * 0.3,
                dtype=torch.float32)
        if k.endswith("grn.weight"):
            sd[k] = torch.tensor(
                0.7 + rng.standard_normal(tuple(sd[k].shape)) * 0.3,
                dtype=torch.float32)
        if k.endswith("grn.bias"):
            sd[k] = torch.tensor(
                rng.standard_normal(tuple(sd[k].shape)) * 0.3,
                dtype=torch.float32)
    model.load_state_dict(sd)

    # Pixels [C][H][W]; feed the SAME array to HF as (1,C,H,W). Round to 4
    # decimals BEFORE the oracle so the committed JSON (which stores the rounded
    # values) and the CAI test see byte-identical pixels - keeps _logits.json
    # compact without breaking the < 1e-4 parity.
    pixels = np.round(
        rng.standard_normal((NUM_CHANNELS, IMAGE, IMAGE)), 4).astype(np.float64)
    pv = torch.tensor(pixels[None, ...], dtype=torch.float32)
    with torch.no_grad():
        logits = model(pixel_values=pv).logits.double().numpy()

    # Save weights (contiguous f32 safetensors).
    out_sd = {k: v.detach().contiguous().to(torch.float32)
              for k, v in model.state_dict().items()}
    save_file(out_sd, os.path.join(FIX, base + ".safetensors"))

    config = {
        "model_type": model_type,
        "depths": DEPTHS,
        "hidden_sizes": DIMS,
        "patch_size": PATCH,
        "image_size": IMAGE,
        "num_channels": NUM_CHANNELS,
        "num_labels": NUM_LABELS,
        "layer_norm_eps": 1e-6,
        "layer_scale_init_value": 1e-6,
    }
    with open(os.path.join(FIX, base + "_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Compact pixel serialisation: %.4f exactly reproduces the 4-decimal
    # rounding above, so both HF (which used `pixels`) and the CAI test (which
    # parses this JSON) see byte-identical values while the JSON stays KB-scale.
    def fmt_chw(arr):
        rows = []
        for c in range(arr.shape[0]):
            rows.append("[" + ",".join(
                "[" + ",".join("%.4f" % v for v in arr[c, y]) + "]"
                for y in range(arr.shape[1])) + "]")
        return "[" + ",".join(rows) + "]"

    with open(os.path.join(FIX, base + "_logits.json"), "w") as f:
        f.write('{"pixels":' + fmt_chw(pixels) + ',"logits":' +
                json.dumps(logits.tolist()) + "}")

    print(f"v{version}: wrote {base}.safetensors / _config.json / _logits.json"
          f"  logits[0]={logits[0]}")


if __name__ == "__main__":
    emit(1)
    emit(2)
