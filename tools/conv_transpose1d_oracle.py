#!/usr/bin/env python3
"""torch.nn.ConvTranspose1d oracle for TNNetDeconvolution1D (tasklist Kokoro-82M, 0.4).

One case per Kokoro-82M transposed geometry, with the REAL kernel size, stride,
padding, output_padding and groups and small channel counts:
  generator ups[0] k20 s10 p5; generator ups[1] k12 s6 p3; AdainResBlk1d pool
  (depthwise, groups = channels) k3 s2 p1 output_padding 1. Plus a length-1
  input to ups[0] (kernel longer than the padded input), a general grouped case
  and a bias-free case.

Writes tests/fixtures/conv_transpose1d_oracle.json. Layouts are torch's: input
[C][L], weight [In][Out/groups][K], output [Out][Lout]. Run:
  ( ulimit -v 3145728; /home/bpsa/x/bin/python -I tools/conv_transpose1d_oracle.py )
"""
import json
import os

import torch

CASES = [
    # name, in, out, k, stride, pad, output_padding, groups, bias, length
    ("ups1_k20_s10_p5", 6, 4, 20, 10, 5, 0, 1, True, 7),
    ("ups1_k20_s10_p5_len1", 6, 4, 20, 10, 5, 0, 1, True, 1),
    ("ups2_k12_s6_p3", 5, 3, 12, 6, 3, 0, 1, True, 9),
    ("pool_depthwise_k3_s2_p1_op1", 6, 6, 3, 2, 1, 1, 6, True, 11),
    ("grouped2_k4_s2_p1", 6, 4, 4, 2, 1, 0, 2, True, 8),
    ("nobias_k3_s2_op1", 3, 2, 3, 2, 0, 1, 1, False, 5),
]


def rounded(values, digits):
    if isinstance(values, list):
        return [rounded(v, digits) for v in values]
    return round(values, digits)


def main():
    torch.manual_seed(20261012)
    out_cases = []
    for name, cin, cout, k, stride, pad, opad, groups, bias, length in CASES:
        conv = torch.nn.ConvTranspose1d(cin, cout, k, stride=stride,
                                        padding=pad, output_padding=opad,
                                        groups=groups, bias=bias)
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
            "output_padding": opad, "groups": groups, "bias": bias,
            "length": length, "out_length": int(y.shape[2]),
            "input": rounded(x[0].tolist(), 4),
            "weight": rounded(conv.weight.detach().tolist(), 4),
            "bias_values": (rounded(conv.bias.detach().tolist(), 4)
                            if bias else []),
            "output": rounded(y[0].tolist(), 7),
        })
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "tests", "fixtures", "conv_transpose1d_oracle.json")
    with open(path, "w") as f:
        json.dump({"torch": torch.__version__, "cases": out_cases}, f)
    print("wrote", os.path.normpath(path), len(out_cases), "cases")


if __name__ == "__main__":
    main()
