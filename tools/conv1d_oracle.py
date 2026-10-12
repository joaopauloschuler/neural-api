#!/usr/bin/env python3
"""torch.nn.Conv1d oracle for TNNetConvolution1D (tasklist Kokoro-82M, 0.4).

One case per Kokoro-82M forward Conv1d geometry, with the REAL kernel size,
stride, padding and dilation and small channel counts:
  TextEncoder k5 p2; AdainResBlk1d k3 p1; conv_post k7 p3; 1x1 with and
  without bias; F0_conv/N_conv k3 s2 p1 (1 -> 1 channel); noise_convs[0]
  k12 s6 p3; generator resblocks k3/k7/k11 at dilation 1/3/5 with torch
  padding d*(k-1)/2.

Writes tests/fixtures/conv1d_oracle.json. Layouts are torch's: input [C][L],
weight [Out][In][K], output [Out][Lout]. Run:
  ( ulimit -v 3145728; /home/bpsa/x/bin/python -I tools/conv1d_oracle.py )
"""
import json
import os

import torch

CASES = [
    # name, in, out, k, stride, pad, dilation, bias, length
    ("textenc_k5_p2", 6, 5, 5, 1, 2, 1, True, 19),
    ("adainres_k3_p1", 7, 4, 3, 1, 1, 1, True, 16),
    ("conv_post_k7_p3", 4, 3, 7, 1, 3, 1, True, 23),
    ("pointwise_k1", 5, 3, 1, 1, 0, 1, True, 11),
    ("pointwise_k1_nobias", 5, 3, 1, 1, 0, 1, False, 11),
    ("f0_conv_k3_s2_p1", 1, 1, 3, 2, 1, 1, True, 21),
    ("noise_conv_k12_s6_p3", 3, 4, 12, 6, 3, 1, True, 61),
    ("resblock_k3_d1", 4, 4, 3, 1, 1, 1, True, 30),
    ("resblock_k3_d3", 4, 4, 3, 1, 3, 3, True, 30),
    ("resblock_k3_d5", 4, 4, 3, 1, 5, 5, True, 30),
    ("resblock_k7_d1", 4, 4, 7, 1, 3, 1, True, 30),
    ("resblock_k7_d3", 4, 4, 7, 1, 9, 3, True, 30),
    ("resblock_k7_d5", 4, 4, 7, 1, 15, 5, True, 30),
    ("resblock_k11_d1", 3, 5, 11, 1, 5, 1, True, 34),
    ("resblock_k11_d3", 3, 5, 11, 1, 15, 3, True, 34),
    ("resblock_k11_d5", 3, 5, 11, 1, 25, 5, True, 34),
]


def rounded(values, digits):
    if isinstance(values, list):
        return [rounded(v, digits) for v in values]
    return round(values, digits)


def main():
    torch.manual_seed(20261012)
    out_cases = []
    for name, cin, cout, k, stride, pad, dil, bias, length in CASES:
        conv = torch.nn.Conv1d(cin, cout, k, stride=stride, padding=pad,
                               dilation=dil, bias=bias)
        # Inputs and weights rounded to 4 decimals keep the JSON small; the
        # output is computed from the rounded values.
        x = torch.round(torch.randn(1, cin, length), decimals=4)
        with torch.no_grad():
            conv.weight.copy_(torch.round(conv.weight, decimals=4))
            if bias:
                conv.bias.copy_(torch.round(conv.bias, decimals=4))
            y = conv(x)
        out_cases.append({
            "name": name, "in_channels": cin, "out_channels": cout,
            "kernel_size": k, "stride": stride, "padding": pad,
            "dilation": dil, "bias": bias, "length": length,
            "out_length": int(y.shape[2]),
            "input": rounded(x[0].tolist(), 4),
            "weight": rounded(conv.weight.detach().tolist(), 4),
            "bias_values": (rounded(conv.bias.detach().tolist(), 4)
                            if bias else []),
            "output": rounded(y[0].tolist(), 7),
        })
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "tests", "fixtures", "conv1d_oracle.json")
    with open(path, "w") as f:
        json.dump({"torch": torch.__version__, "cases": out_cases}, f)
    print("wrote", os.path.normpath(path), len(out_cases), "cases")


if __name__ == "__main__":
    main()
