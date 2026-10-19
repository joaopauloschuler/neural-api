#!/usr/bin/env python3
"""Oracle for the Kokoro-82M per-element / per-channel layers (tasklist 0.4).

Built from the kokoro package's own module classes with random weights:
  AdaIN1d (istftnet.py; the InstanceNorm1d affine is randomized too, so the
  host fold (1+gamma)*w, (1+gamma)*b + beta is exercised), AdaLayerNorm and the
  TextEncoder channel LayerNorm (modules.py), the AdaINResBlock1 Snake with a
  per-channel alpha [1,C,1], and F.leaky_relu at slopes 0.2, 0.1 and 0.01.

Writes tests/fixtures/kokoro_norm_activation_oracle.json. Sequences are in
torch's [C][L] layout (time on L), the style vector is [style_dim]. Run:
  ( ulimit -v 3145728; /home/bpsa/x/bin/python -I tools/kokoro_norm_activation_oracle.py )
"""
import json
import os

import torch
import torch.nn.functional as F

from kokoro.istftnet import AdaIN1d
from kokoro.modules import AdaLayerNorm, LayerNorm

STYLE_DIM = 8


def rounded(values, digits):
    if isinstance(values, list):
        return [rounded(v, digits) for v in values]
    return round(values, digits)


def rand(*shape, scale=1.0):
    # Inputs and weights rounded to 4 decimals keep the JSON small; every
    # output is computed from the rounded values.
    return torch.round(torch.randn(*shape) * scale, decimals=4)


def round_params(module):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.round(p, decimals=4))


def main():
    torch.manual_seed(20261012)
    cases = []
    with torch.no_grad():
        for channels, length in ((6, 17), (4, 5)):
            m = AdaIN1d(STYLE_DIM, channels)
            m.norm.weight.copy_(1 + 0.3 * torch.randn(channels))
            m.norm.bias.copy_(0.3 * torch.randn(channels))
            m.fc.weight.copy_(0.3 * torch.randn(2 * channels, STYLE_DIM))
            round_params(m)
            m.eval()
            x = rand(1, channels, length, scale=2.0) + 0.5
            s = rand(1, STYLE_DIM)
            y = m(x, s)
            cases.append({
                "kind": "adain1d", "channels": channels, "length": length,
                "input": rounded(x[0].tolist(), 4),
                "style": rounded(s[0].tolist(), 4),
                "fc_weight": rounded(m.fc.weight.tolist(), 4),
                "fc_bias": rounded(m.fc.bias.tolist(), 4),
                "norm_weight": rounded(m.norm.weight.tolist(), 4),
                "norm_bias": rounded(m.norm.bias.tolist(), 4),
                "output": rounded(y[0].tolist(), 7)})
        for channels, length in ((6, 9), (5, 1)):
            m = AdaLayerNorm(STYLE_DIM, channels)
            m.fc.weight.copy_(0.3 * torch.randn(2 * channels, STYLE_DIM))
            round_params(m)
            x = rand(1, channels, length, scale=2.0) - 0.3
            s = rand(1, STYLE_DIM)
            # The DurationEncoder calls it with [B, T, C].
            y = m(x.transpose(1, 2), s).transpose(1, 2)
            cases.append({
                "kind": "ada_layer_norm", "channels": channels,
                "length": length,
                "input": rounded(x[0].tolist(), 4),
                "style": rounded(s[0].tolist(), 4),
                "fc_weight": rounded(m.fc.weight.tolist(), 4),
                "fc_bias": rounded(m.fc.bias.tolist(), 4),
                "output": rounded(y[0].tolist(), 7)})
        for channels, length in ((7, 11),):
            m = LayerNorm(channels)
            m.gamma.copy_(1 + 0.3 * torch.randn(channels))
            m.beta.copy_(0.3 * torch.randn(channels))
            round_params(m)
            x = rand(1, channels, length, scale=2.0) + 0.2
            y = m(x)
            cases.append({
                "kind": "layer_norm", "channels": channels, "length": length,
                "input": rounded(x[0].tolist(), 4),
                "gamma": rounded(m.gamma.tolist(), 4),
                "beta": rounded(m.beta.tolist(), 4),
                "output": rounded(y[0].tolist(), 7)})
        for channels, length in ((5, 13),):
            # AdaINResBlock1.forward, the Snake1D line, alpha [1, C, 1].
            a = torch.round(0.2 + 2 * torch.rand(1, channels, 1), decimals=4)
            x = rand(1, channels, length, scale=2.0)
            y = x + (1 / a) * (torch.sin(a * x) ** 2)
            cases.append({
                "kind": "snake", "channels": channels, "length": length,
                "input": rounded(x[0].tolist(), 4),
                "alpha": rounded(a[0, :, 0].tolist(), 4),
                "output": rounded(y[0].tolist(), 7)})
        for slope in (0.2, 0.1, 0.01):
            channels, length = 4, 9
            x = rand(1, channels, length, scale=2.0)
            y = F.leaky_relu(x, negative_slope=slope)
            cases.append({
                "kind": "leaky_relu", "slope": slope, "channels": channels,
                "length": length,
                "input": rounded(x[0].tolist(), 4),
                "output": rounded(y[0].tolist(), 7)})
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "tests", "fixtures", "kokoro_norm_activation_oracle.json")
    with open(path, "w") as f:
        json.dump({"torch": torch.__version__, "style_dim": STYLE_DIM,
                   "cases": cases}, f)
    print("wrote", os.path.normpath(path), len(cases), "cases")


if __name__ == "__main__":
    main()
