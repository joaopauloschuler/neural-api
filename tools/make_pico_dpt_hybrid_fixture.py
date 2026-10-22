#!/usr/bin/env python3
"""Generate pico RANDOM DPTForDepthEstimation parity fixtures (transformers
model_type "dpt") for tests/TestNeuralPretrained.pas. Nothing is downloaded.

  tests/fixtures/tiny_sd2_depth_estimator/depth_estimator/
      {config.json, model.safetensors}   DPT-Hybrid (is_hybrid, the
      Intel/dpt-hybrid-midas layout of stabilityai/stable-diffusion-2-depth):
      BiT bottleneck backbone (weight-standardized convs, GroupNorm, SAME /
      dynamic padding) -> ViT-hybrid embeddings (1x1 projection of the BiT
      stage-3 map, cls token, position table) -> ViT encoder; the neck takes
      the BiT stage-1/2 maps raw (neck_ignore_stages [0, 1]) and two encoder
      blocks through readout "project"; DPT fusion + depth head.
  tests/fixtures/tiny_sd2_depth_estimator/feature_extractor/
      preprocessor_config.json            DPTImageProcessor at the estimator
                                          size, mean/std 0.5, bilinear
  tests/fixtures/tiny_dpt_hybrid_io.json  float64 predicted_depth per case
  tests/fixtures/tiny_dpt_vit/{config.json, model.safetensors}
  tests/fixtures/tiny_dpt_vit_io.json     plain-ViT DPT (Intel/dpt-large
                                          layout): patch embeddings, all four
                                          stages through readout "project"

Each io file: {"cases": [{"input": (H*W*C) in (y, x, c) order,
"out_grid_w", "out_grid_h", "output": (H*W) in (y, x) order}]}.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_dpt_hybrid_fixture.py
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import BitConfig, DPTConfig, DPTForDepthEstimation

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")
ESTIMATOR_ROOT = os.path.join(FIXTURES, "tiny_sd2_depth_estimator")

# Hybrid: 64x64 input -> BiT stage maps 16 (stage1), 8 (stage2), 4 (stage3 =
# the ViT grid at patch 16); factor 0.5 -> 2; fusion 2->4->8->16->32; head 64.
HYBRID_IMAGE = 64
# Plain ViT: 32x32, patch 8 -> grid 4; factors 4/2/1/0.5 -> 16/8/4/2.
VIT_IMAGE = 32


def randomize(model, seed, weight_std=0.09):
    """O(1) weights so the float32 Pascal forward is a meaningful check."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("weight") and param.dim() >= 2:
                std = weight_std
            elif "norm" in name and name.endswith("weight"):
                param.copy_(1 + 0.1 * torch.randn(param.shape, generator=generator))
                continue
            elif name.endswith("bias"):
                std = 0.05
            else:
                std = 0.1
            param.copy_(std * torch.randn(param.shape, generator=generator))
        # The final ReLU clamps negatives; lift the last conv so the map is
        # not mostly zeros.
        model.head.head[4].bias.add_(0.3)


def save_model(model, folder):
    os.makedirs(folder, exist_ok=True)
    model.config.to_json_file(os.path.join(folder, "config.json"))
    tensors = {k: v.detach().float().cpu().numpy() for k, v in
               model.state_dict().items() if not k.endswith("num_batches_tracked")}
    save_file(tensors, os.path.join(folder, "model.safetensors"),
              metadata={"format": "pt"})


def rounded(values):
    """float32-exact inputs and 10 significant digits keep the files small."""
    return [float("%.10g" % v) for v in values]


def write_cases(model, image, io_file, seed):
    rng = np.random.default_rng(seed)
    model64 = model.double().eval()
    cases = []
    for _ in range(1):
        x = (rng.standard_normal((1, 3, image, image)) * 0.8).astype(np.float32)
        x = x.astype(np.float64)
        with torch.no_grad():
            depth = model64(torch.from_numpy(x)).predicted_depth[0].numpy()
        gh, gw = depth.shape
        cases.append({"input": rounded(np.transpose(x[0], (1, 2, 0)).reshape(-1)),
                      "out_grid_w": int(gw), "out_grid_h": int(gh),
                      "output": rounded(depth.reshape(-1))})
        print(io_file, "depth range", depth.min(), depth.max(),
              "zeros", float((depth == 0).mean()))
    with open(io_file, "w") as f:
        json.dump({"cases": cases}, f)
    model.float()


def make_hybrid():
    torch.manual_seed(20261022)
    backbone = BitConfig(
        num_channels=3, embedding_size=8, hidden_sizes=[16, 32, 64, 128],
        depths=[2, 1, 2], layer_type="bottleneck", hidden_act="relu",
        global_padding="same", num_groups=4, embedding_dynamic_padding=True,
        output_stride=32, width_factor=1,
        out_features=["stage1", "stage2", "stage3"])
    config = DPTConfig(
        hidden_size=16, num_hidden_layers=5, num_attention_heads=2,
        intermediate_size=32, hidden_act="gelu", layer_norm_eps=1e-12,
        image_size=HYBRID_IMAGE, patch_size=16, num_channels=3, is_hybrid=True,
        qkv_bias=True, backbone_out_indices=[0, 1, 2, 4], readout_type="project",
        reassemble_factors=[1, 1, 1, 0.5], neck_hidden_sizes=[16, 32, 16, 16],
        fusion_hidden_size=8, head_in_index=-1,
        use_batch_norm_in_fusion_residual=False,
        backbone_featmap_shape=[1, 64, HYBRID_IMAGE // 16, HYBRID_IMAGE // 16],
        neck_ignore_stages=[0, 1], backbone_config=backbone)
    model = DPTForDepthEstimation(config).eval()
    randomize(model, 7101)
    save_model(model, os.path.join(ESTIMATOR_ROOT, "depth_estimator"))
    feature_dir = os.path.join(ESTIMATOR_ROOT, "feature_extractor")
    os.makedirs(feature_dir, exist_ok=True)
    preprocessor = {
        "do_normalize": True, "do_rescale": True, "do_resize": True,
        "ensure_multiple_of": 1, "image_mean": [0.5, 0.5, 0.5],
        "image_processor_type": "DPTImageProcessor",
        "image_std": [0.5, 0.5, 0.5], "keep_aspect_ratio": False, "resample": 2,
        "rescale_factor": 0.00392156862745098,
        "size": {"height": HYBRID_IMAGE, "width": HYBRID_IMAGE}}
    with open(os.path.join(feature_dir, "preprocessor_config.json"), "w") as f:
        json.dump(preprocessor, f, indent=2)
    write_cases(model, HYBRID_IMAGE, os.path.join(FIXTURES, "tiny_dpt_hybrid_io.json"),
                7102)


def make_vit():
    torch.manual_seed(20261023)
    config = DPTConfig(
        hidden_size=16, num_hidden_layers=4, num_attention_heads=2,
        intermediate_size=32, hidden_act="gelu", layer_norm_eps=1e-12,
        image_size=VIT_IMAGE, patch_size=8, num_channels=3, is_hybrid=False,
        qkv_bias=True, backbone_out_indices=[0, 1, 2, 3], readout_type="project",
        reassemble_factors=[4, 2, 1, 0.5], neck_hidden_sizes=[4, 8, 16, 16],
        fusion_hidden_size=8, head_in_index=-1,
        use_batch_norm_in_fusion_residual=False)
    model = DPTForDepthEstimation(config).eval()
    randomize(model, 7201, weight_std=0.14)
    save_model(model, os.path.join(FIXTURES, "tiny_dpt_vit"))
    write_cases(model, VIT_IMAGE, os.path.join(FIXTURES, "tiny_dpt_vit_io.json"), 7202)


if __name__ == "__main__":
    make_hybrid()
    make_vit()
