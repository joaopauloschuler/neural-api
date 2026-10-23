#!/usr/bin/env python3
"""Generate pico RANDOM LightGlue keypoint-matching parity fixtures for
tests/TestNeuralPretrained.pas (TestLightGlue*).

No network access: a pico transformers LightGlueForKeypointMatching is built
with the pico SuperPoint of make_pico_superpoint_fixture.py as its keypoint
detector (its weights are copied in from tests/fixtures/tiny_superpoint.safetensors)
and a randomly initialized matcher. The reference is the real transformers
forward in float64 with the adaptive depth and the point pruning disabled
(depth_confidence = width_confidence = -1 on the module). The written config
keeps the checkpoint defaults (0.95 / 0.99) so the config reader is exercised
on real values. A second oracle ("adaptive" in the io json) reruns the same
image pairs with per-variant depth / width confidences that trigger point
pruning (and, on tiny_lightglue, an early stop): the token confidence heads are
scaled and shifted so their sigmoid straddles the layer thresholds.

Two variants:
  tiny_lightglue    descriptor_dim 16 > SuperPoint's 8 (input_projection),
                    2 heads, 3 layers;
  tiny_lightglue_gqa descriptor_dim 8 (identity projection), 2 query heads
                    sharing 1 key/value head, 2 layers.
Per case the oracle stores both images, the keypoints of each, the final
matcher descriptors (input of the last match assignment layer), and the
matches / matching scores of each side. A float32 run of the same model must
give the same keypoints and matches, so no decision sits on a float32 tie.

Coded by Claude (AI).

Usage (from the repo root, after make_pico_superpoint_fixture.py):
  /home/bpsa/x/bin/python tools/make_pico_lightglue_fixture.py
writes tests/fixtures/tiny_lightglue{,_gqa}{.safetensors,_config.json,_io.json}.
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import load_file, save_file
from transformers import LightGlueConfig, LightGlueForKeypointMatching

WIDTH = 64
HEIGHT = 48
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")

with open(os.path.join(fixtures, "tiny_superpoint_config.json")) as f:
    sp_config = json.load(f)
# More keypoints on a small image: a smaller NMS window.
sp_config["nms_radius"] = 2
sp_config["border_removal_distance"] = 2
sp_weights = load_file(os.path.join(fixtures, "tiny_superpoint.safetensors"))


def build(name, descriptor_dim, heads, kv_heads, layers, filter_threshold, seed,
          confidence_scale, confidence_biases):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = LightGlueConfig(
        keypoint_detector_config=dict(sp_config),
        descriptor_dim=descriptor_dim,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        depth_confidence=0.95,
        width_confidence=0.99,
        filter_threshold=filter_threshold,
        attn_implementation="eager",
    )
    model = LightGlueForKeypointMatching(cfg).eval()
    with torch.no_grad():
        for pname, p in model.named_parameters():
            if pname.startswith("keypoint_detector."):
                continue
            if "layer_norm" in pname:
                if pname.endswith("weight"):
                    torch.nn.init.normal_(p, mean=1.0, std=0.1)
                else:
                    torch.nn.init.normal_(p, mean=0.0, std=0.1)
            elif p.dim() == 2:
                std = 1.0 / np.sqrt(p.shape[1])
                if "positional_encoder" in pname:
                    std = 2.0  # angles of a few radians across the image
                if "fc2" in pname:
                    std *= 0.1  # keep the residual stream near the descriptors
                torch.nn.init.normal_(p, mean=0.0, std=std)
                if "input_projection" in pname:
                    # Stacked identity: keeps the descriptors' cosine geometry.
                    eye = torch.eye(p.shape[1], dtype=p.dtype)
                    p.mul_(0.2).add_(eye.repeat(p.shape[0] // p.shape[1], 1))
                if "final_projection" in pname:
                    # Near identity: the similarity follows the descriptors, so
                    # the shifted pair has many mutual matches.
                    p.mul_(0.3).add_(6.0 * torch.eye(p.shape[0], dtype=p.dtype))
            else:
                # Small biases: a shared offset would make every descriptor
                # alike and leave few mutual nearest neighbours.
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
                if "matchability.bias" in pname:
                    p.add_(2.0)  # most points matchable, as when trained
        # After the seeded init, so every other tensor keeps its values.
        for layer, head in enumerate(model.token_confidence):
            head.token.weight.mul_(confidence_scale)
            head.token.bias.add_(confidence_biases[layer])
        sd = model.keypoint_detector.state_dict()
        for key in sd:
            sd[key].copy_(torch.from_numpy(sp_weights[key]))
    model.depth_confidence = -1
    model.width_confidence = -1
    # The oracle runs on exactly the float32 weights the fixture stores.
    return cfg, model.float().double()


def run(model, image0, image1, dtype):
    m = model.to(dtype)
    captured = {}

    def hook(_module, args):
        captured["descriptors"] = args[0].detach().clone()

    handle = m.match_assignment_layers[-1].register_forward_pre_hook(hook)
    pv = np.stack([image0, image1])[None, :, None]  # (1, 2, 1, H, W)
    pv = torch.from_numpy(np.repeat(pv, 3, axis=2)).to(dtype)
    with torch.no_grad():
        out = m(pv)
    handle.remove()
    counts = [int(out.mask[0, i].sum()) for i in range(2)]
    result = {}
    for side in range(2):
        n = counts[side]
        kps = out.keypoints[0, side, :n].double().numpy() * np.array([WIDTH, HEIGHT])
        result["keypoints%d" % side] = np.rint(kps).astype(int).reshape(-1).tolist()
        result["descriptors%d" % side] = (
            captured["descriptors"][side, :n].double().numpy().reshape(-1).tolist())
        result["matches%d" % side] = out.matches[side, :n].numpy().astype(int).tolist()
        result["scores%d" % side] = out.matching_scores[side, :n].double().numpy().tolist()
    return result


def run_adaptive(model, image0, image1, dtype, depth, width):
    """transformers forward with early stop + pruning; returns the matches,
    the stop layer, per-layer token counts and the kept rows' descriptors."""
    m = model.to(dtype)
    m.depth_confidence = depth
    m.width_confidence = width
    stops = []
    captured = {}

    def hook(layer):
        def fn(_module, args):
            stops.append(layer)
            captured["descriptors"] = args[0].detach().clone()
        return fn

    handles = [head.register_forward_pre_hook(hook(layer))
               for layer, head in enumerate(m.match_assignment_layers)]
    original_remap = m._do_final_keypoint_pruning

    def remap(indices, matches, scores, num_keypoints):
        captured["remap"] = (indices.clone(), matches.clone(), scores.clone())
        return original_remap(indices, matches, scores, num_keypoints)

    m._do_final_keypoint_pruning = remap
    pv = np.stack([image0, image1])[None, :, None]
    pv = torch.from_numpy(np.repeat(pv, 3, axis=2)).to(dtype)
    with torch.no_grad():
        out = m(pv)
    del m._do_final_keypoint_pruning
    for handle in handles:
        handle.remove()
    m.depth_confidence = -1
    m.width_confidence = -1
    assert len(stops) == 1, stops
    stop = stops[0]
    counts = [int(out.mask[0, i].sum()) for i in range(2)]
    prune = out.prune[0].numpy()
    result = {"stop_layer": stop}
    # prune[i] = 1 + the number of pruning passes keypoint i survived.
    result["layer_counts"] = [[int((prune[side, :counts[side]] >= layer + 1).sum())
                               for side in range(2)] for layer in range(stop + 1)]
    kept = [[int(i) for i in np.nonzero(prune[side, :counts[side]] == stop + 1)[0]]
            for side in range(2)]
    # The remap of the kept rows' matches back to keypoint indices, without
    # transformers' quirk: the shorter side's -1 padding indices also write
    # (match -1, score 0) into the last keypoint slot.
    indices, matches, scores = (t.reshape(2, -1) for t in captured["remap"])
    torch_matches = out.matches.reshape(2, -1)
    torch_scores = out.matching_scores.reshape(2, -1)
    result["transformers_clobbered"] = []
    for side in range(2):
        n = counts[side]
        assert [int(i) for i in indices[side] if i >= 0] == kept[side]
        side_matches = [-1] * n
        side_scores = [0.0] * n
        for row, index in enumerate(kept[side]):
            partner = int(matches[side, row])
            if partner >= 0:
                side_matches[index] = kept[1 - side][partner]
            side_scores[index] = float(scores[side, row])
        for index in range(n):
            if (side_matches[index] != int(torch_matches[side, index]) or
                    abs(side_scores[index] - float(torch_scores[side, index])) > 1e-12):
                assert index == max(counts) - 1, (side, index)
                result["transformers_clobbered"].append([side, index])
        result["kept%d" % side] = kept[side]
        result["descriptors%d" % side] = (
            captured["descriptors"][side, :len(kept[side])].double().numpy()
            .reshape(-1).tolist())
        result["matches%d" % side] = side_matches
        result["scores%d" % side] = side_scores
    return result


def smooth_noise(rng):
    img = rng.random((HEIGHT // 4, WIDTH // 4))
    img = np.kron(img, np.ones((4, 4)))
    return img


def write_variant(name, cfg, model, cases, adaptive):
    tensors = {k: v.detach().cpu().numpy().astype(np.float32)
               for k, v in model.float().state_dict().items()}
    save_file(tensors, os.path.join(fixtures, name + ".safetensors"))
    config = cfg.to_dict()
    keep = ["model_type", "descriptor_dim", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "depth_confidence",
            "width_confidence", "filter_threshold", "hidden_act",
            "attention_bias"]
    out_cfg = {k: config[k] for k in keep}
    out_cfg["keypoint_detector_config"] = dict(sp_config)
    with open(os.path.join(fixtures, name + "_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)
    with open(os.path.join(fixtures, name + "_io.json"), "w") as f:
        json.dump({"width": WIDTH, "height": HEIGHT, "cases": cases,
                   "adaptive": adaptive}, f)
    print("wrote", name, "tensors:", len(tensors))


def make_cases(model, rng, depth, width):
    base = smooth_noise(rng)
    pairs = [
        # The same scene shifted by 8 pixels plus noise: the counts differ.
        (base, np.clip(np.roll(base, 8, axis=1) + 0.05 * rng.random(base.shape), 0, 1)),
        # Two unrelated images.
        (smooth_noise(rng), smooth_noise(rng)),
    ]
    cases = []
    adaptive_cases = []
    for image0, image1 in pairs:
        ref = run_adaptive(model, image0, image1, torch.float64, depth, width)
        check = run_adaptive(model, image0, image1, torch.float32, depth, width)
        for key in ["stop_layer", "layer_counts", "kept0", "kept1",
                    "matches0", "matches1"]:
            assert ref[key] == check[key], key
        adaptive_cases.append(ref)
        ref = run(model, image0, image1, torch.float64)
        check = run(model, image0, image1, torch.float32)
        for key in ["keypoints0", "keypoints1", "matches0", "matches1"]:
            assert ref[key] == check[key], key
        case = {"image0": image0.reshape(-1).tolist(),
                "image1": image1.reshape(-1).tolist()}
        case.update(ref)
        cases.append(case)
    counts = [(len(c["scores0"]), len(c["scores1"])) for c in cases]
    valid = [sum(1 for m in c["matches0"] if m >= 0) for c in cases]
    mutual_low = [sum(1 for m, s in zip(c["matches0"], c["scores0"])
                      if m < 0 and s > 0) for c in cases]
    print("  keypoints per case:", counts)
    print("  valid matches per case:", valid)
    print("  mutual below threshold per case:", mutual_low)
    print("  mutual scores:", [sorted(round(s, 4) for s in c["scores0"] if s > 0) for c in cases])
    assert any(a != b for a, b in counts), counts
    assert all(v > 0 for v in valid), valid
    print("  adaptive stop layer per case:", [c["stop_layer"] for c in adaptive_cases])
    print("  adaptive tokens per layer per case:", [c["layer_counts"] for c in adaptive_cases])
    print("  transformers' clobbered slots per case:",
          [c["transformers_clobbered"] for c in adaptive_cases])
    print("  adaptive valid matches per case:",
          [sum(1 for m in c["matches0"] if m >= 0) for c in adaptive_cases])
    assert all(c["layer_counts"][-1] != c["layer_counts"][0] for c in adaptive_cases)
    adaptive = {"depth_confidence": depth, "width_confidence": width,
                "cases": adaptive_cases}
    return cases, adaptive


# Prunes at layer 0, then stops early at layer 1 of 3.
cfg, model = build("tiny_lightglue", 16, 2, 2, 3, 0.05, 2026, 4.0, [2.2, 4.0])
cases, adaptive = make_cases(model.double(), np.random.default_rng(7), 0.8, 0.11)
assert all(c["stop_layer"] == 1 for c in adaptive["cases"])
write_variant("tiny_lightglue", cfg, model, cases, adaptive)

# Prunes at layer 0 and runs to the last layer: no early stop.
cfg, model = build("tiny_lightglue_gqa", 8, 2, 1, 2, 0.05, 2027, 4.0, [2.2])
cases, adaptive = make_cases(model.double(), np.random.default_rng(8), 0.95, 0.145)
assert all(c["stop_layer"] == 1 for c in adaptive["cases"])
write_variant("tiny_lightglue_gqa", cfg, model, cases, adaptive)
