#!/usr/bin/env python3
"""Generate the LPIPS perceptual-distance oracle for
tests/TestNeuralPretrained.pas (TestLPIPSDistanceParity).

LPIPS (Zhang et al. 2018, richzhang/PerceptualSimilarity) runs both images
through a VGG-16 backbone, reads the 5 per-stage relu taps, channel-unit-
normalizes each feature map, takes the squared per-channel difference, applies
the learned per-stage "lin" 1x1 conv (one non-negative scalar weight per
channel), spatial-averages, and sums across stages.

The official richzhang lin weights are tiny but are NOT obtainable offline in
this environment (the `lpips` pip package is not installed and there is no
network access). This oracle therefore pins the UNWEIGHTED variant (lin = 1/C
per-channel MEAN, i.e. lpips lin_layers=False / "uncalibrated" baseline), which
the Pascal helper ComputeLPIPSDistance(...) reproduces when LinWeights = nil.
The arithmetic is identical except for the per-channel weighting, so the
unweighted oracle exercises the entire feature-extraction + unit-normalize +
squared-diff + spatial-mean + stage-sum path. (Supplying official lin weights
to the helper is a one-line change once the weights are available.)

This script REUSES the committed tiny_vgg16 fixture (tools/vgg_tiny_fixture.py):
the same pico random-init VGG-16, the same numpy float64 forward oracle. It
builds a SECOND image (image B) from a different deterministic pattern, runs
both through the float64 forward to get the 5 relu taps, and writes
tests/fixtures/tiny_lpips.json with image B pixels + the per-stage and total
LPIPS distance for (A,B), plus the trivial (A,A)=0 check value.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/lpips_tiny_fixture.py
writes tests/fixtures/tiny_lpips.json. Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import load_file

# Reuse the exact tiny VGG config/forward from the VGG fixture generator.
import vgg_tiny_fixture as vgg

EPS = 1e-10


def unit_normalize(feat):
    """Channel-unit-normalize a (C,H,W) map along the channel axis."""
    norm = np.sqrt((feat ** 2).sum(axis=0, keepdims=True) + EPS)
    return feat / norm


def lpips_unweighted(taps_a, taps_b):
    """Sum over stages of the spatial-mean of the per-channel-MEAN squared
    difference between the unit-normalized tap maps (lin = 1/C)."""
    per_stage = {}
    total = 0.0
    for k in taps_a:
        a = unit_normalize(taps_a[k])          # (C,H,W)
        b = unit_normalize(taps_b[k])
        d = (a - b) ** 2                        # (C,H,W)
        # lin = 1/C per channel -> mean over channels, then spatial mean.
        stage = d.mean(axis=0).mean()          # scalar
        per_stage[k] = float(stage)
        total += stage
    return float(total), per_stage


def make_image(seed_mul):
    pix = np.zeros((vgg.NUM_CHANNELS, vgg.IMAGE, vgg.IMAGE), dtype=np.float64)
    for c in range(vgg.NUM_CHANNELS):
        for y in range(vgg.IMAGE):
            for x in range(vgg.IMAGE):
                pix[c, y, x] = (((c * 251 + y * 13 + x * seed_mul) * 7)
                               % 19 - 9) / 9.0
    return pix


def main():
    sd_f32 = load_file('tests/fixtures/tiny_vgg16.safetensors')
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    # Image A is the SAME pattern the VGG fixture pins (so reviewers can reuse
    # the logits fixture's pixels); image B is a distinct deterministic pattern.
    pix_a = np.zeros((vgg.NUM_CHANNELS, vgg.IMAGE, vgg.IMAGE), dtype=np.float64)
    for c in range(vgg.NUM_CHANNELS):
        for y in range(vgg.IMAGE):
            for x in range(vgg.IMAGE):
                pix_a[c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0
    pix_b = make_image(seed_mul=3)

    _, taps_a, _ = vgg.forward(pix_a, sd)
    _, taps_b, _ = vgg.forward(pix_b, sd)

    total_ab, per_stage_ab = lpips_unweighted(taps_a, taps_b)
    total_aa, _ = lpips_unweighted(taps_a, taps_a)

    print(f'LPIPS(A,B) unweighted = {total_ab:.10f}')
    for k in sorted(per_stage_ab):
        print(f'  stage {k}: {per_stage_ab[k]:.10f}')
    print(f'LPIPS(A,A) = {total_aa:.2e} (must be 0)')
    assert total_aa < 1e-12, f'LPIPS(A,A) not zero: {total_aa}'
    assert total_ab > 1e-3, f'LPIPS(A,B) suspiciously small: {total_ab}'

    out = {
        'note': 'unweighted (lin=1/C) LPIPS-VGG oracle over the tiny_vgg16 '
                'fixture; ComputeLPIPSDistance(LinWeights=nil) must match.',
        'image_b': pix_b.tolist(),
        'lpips_ab': total_ab,
        'lpips_aa': total_aa,
        'per_stage_ab': {k: per_stage_ab[k] for k in sorted(per_stage_ab)},
    }
    with open('tests/fixtures/tiny_lpips.json', 'w') as f:
        json.dump(out, f)
    print('wrote tests/fixtures/tiny_lpips.json')


if __name__ == '__main__':
    main()
