#!/usr/bin/env python3
"""torch F.interpolate(antialias=True) oracle for TNNetResize2D antialias=1.

Covers bilinear and bicubic: downscale by 2, a non-integer downscale (13 -> 5),
a mixed case (downscale X, upscale Y), align_corners=True, several channels.

Writes tests/fixtures/resize2d_antialias_oracle.json. Layouts are torch's:
input [C][H][W], output [C][OutH][OutW]. Run:
  ( ulimit -v 2097152; /home/bpsa/x/bin/python -I tools/make_resize2d_antialias_fixture.py )
"""
import json
import os

import torch
import torch.nn.functional as F

CASES = [
    # name, mode, align_corners, channels, in_h, in_w, out_h, out_w
    ("bilinear_down2", "bilinear", False, 3, 8, 8, 4, 4),
    ("bicubic_down2", "bicubic", False, 3, 8, 8, 4, 4),
    ("bilinear_13to5", "bilinear", False, 2, 13, 13, 5, 5),
    ("bicubic_13to5", "bicubic", False, 2, 13, 13, 5, 5),
    ("bilinear_mixed", "bilinear", False, 4, 5, 12, 9, 5),
    ("bicubic_mixed", "bicubic", False, 4, 5, 12, 9, 5),
    ("bilinear_ac1", "bilinear", True, 3, 7, 11, 3, 4),
    ("bicubic_ac1", "bicubic", True, 3, 7, 11, 3, 4),
]


def rounded(values, digits):
    if isinstance(values, list):
        return [rounded(v, digits) for v in values]
    return round(values, digits)


def main():
    torch.manual_seed(20261020)
    out_cases = []
    for name, mode, ac, c, ih, iw, oh, ow in CASES:
        # Inputs rounded to 4 decimals keep the JSON small; the output is
        # computed from the rounded values.
        x = torch.round(torch.randn(1, c, ih, iw), decimals=4)
        y = F.interpolate(x, size=(oh, ow), mode=mode, align_corners=ac,
                          antialias=True)
        out_cases.append({
            "name": name, "mode": 1 if mode == "bilinear" else 2,
            "align_corners": int(ac), "channels": c,
            "in_h": ih, "in_w": iw, "out_h": oh, "out_w": ow,
            "input": rounded(x[0].tolist(), 4),
            "output": rounded(y[0].tolist(), 7),
        })
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "tests", "fixtures", "resize2d_antialias_oracle.json")
    with open(path, "w") as f:
        json.dump({"torch": torch.__version__, "cases": out_cases}, f)
    print("wrote", os.path.normpath(path), len(out_cases), "cases")


if __name__ == "__main__":
    main()
