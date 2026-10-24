#!/usr/bin/env python3
"""Oracle for DrawOpenPoseBodies (neural/neuralpretrained.pas, test
TestOpenPoseDrawParity in tests/TestNeuralPretrained.pas).

controlnet_aux is not needed: this script repeats, with cv2, the two pieces of
controlnet_aux 0.0.x DWposeDetector that turn COCO-17 keypoints into the
OpenPose control image:
  * dwpose/wholebody.py: neck = mean of the shoulders (score 1 only when both
    shoulders > 0.3), inserted as joint 17, then the mmpose -> openpose reorder;
  * dwpose/__init__.py + dwpose/util.py draw_bodypose: coordinates / W, / H,
    joints with score > 0.3 kept, the 17 limbs as cv2.ellipse2Poly (stickwidth
    4) + cv2.fillConvexPoly in full color, canvas * 0.6 (uint8), then the 18
    joint dots as cv2.circle(radius 4, filled). Float32 throughout, as there.
Two persons on a 96 x 64 canvas: one fully visible (an ankle dot clipped by the
bottom edge), one with a low-score shoulder (no neck), low-score ears and a knee,
and a wrist and an ankle outside the canvas (limbs clipped by the edges).

Also 32 bare cv2.ellipse2Poly cases (center, axes, angle) and their points.

Writes tests/fixtures/openpose_draw_oracle.json:
  width, height, threshold, persons[i] = [[x, y, score] * 17] (COCO order),
  ellipses[i] = [cx, cy, ax, ay, angle], ellipse_points[i] = flat x, y pairs,
  nonzero_pos / nonzero_value: flat (y, x, c) positions and values of the
  uint8 RGB canvas that are not 0.

Usage (from the repo root; cv2 from an unpacked opencv-python-headless wheel):
  ~/x/bin/python -I tools/make_openpose_draw_oracle.py --extra-path /tmp/cvlib4

Coded by Claude (AI).
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--extra-path', action='append', default=[],
                help='directory prepended to sys.path (e.g. an unpacked cv2)')
ap.add_argument('--out', default='tests/fixtures/openpose_draw_oracle.json')
args = ap.parse_args()
for path in args.extra_path:
    sys.path.insert(0, path)

import cv2
import numpy as np

print('cv2', cv2.__version__)
W, H, THRESHOLD = 96, 64, 0.3

# COCO-17 (x, y, score) in photo pixels.
PERSONS = [
    [[30.3, 9.6, 0.92], [32.1, 8.2, 0.88], [28.4, 8.4, 0.9], [34.7, 9.1, 0.71],
     [25.6, 9.3, 0.66], [37.2, 17.4, 0.95], [22.9, 17.8, 0.93],
     [41.6, 27.3, 0.87], [17.3, 26.9, 0.85], [44.2, 36.7, 0.8],
     [12.4, 34.1, 0.78], [34.6, 38.2, 0.9], [25.8, 38.4, 0.91],
     [35.9, 49.6, 0.84], [24.3, 50.2, 0.86], [36.8, 61.3, 0.77],
     [23.1, 62.7, 0.9]],
    [[70.4, 12.2, 0.81], [72.2, 10.9, 0.79], [68.6, 11.1, 0.83],
     [74.9, 12.0, 0.1], [66.3, 12.3, 0.25], [79.1, 20.6, 0.2],
     [63.5, 21.1, 0.89], [88.7, 24.4, 0.74], [57.2, 30.5, 0.82],
     [99.3, 22.8, 0.69], [54.1, 40.2, 0.76], [76.6, 42.3, 0.85],
     [66.2, 42.9, 0.88], [78.4, 53.7, 0.72], [64.8, 54.6, 0.31],
     [80.2, 66.5, 0.6], [63.9, 63.4, 0.05]],
]

# dwpose/wholebody.py: COCO-17 -> OpenPose-18.
info = np.array(PERSONS, dtype=np.float32)
neck = np.mean(info[:, [5, 6]], axis=1)
neck[:, 2:4] = np.logical_and(info[:, 5, 2:4] > 0.3,
                              info[:, 6, 2:4] > 0.3).astype(int)
new_info = np.insert(info, 17, neck, axis=1)
mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
new_info[:, openpose_idx] = new_info[:, mmpose_idx]
candidate = new_info[..., :2].copy()
scores = new_info[..., 2].copy()

# dwpose/__init__.py DWposeDetector.__call__.
nums = candidate.shape[0]
candidate[..., 0] /= float(W)
candidate[..., 1] /= float(H)
body = candidate[:, :18].copy().reshape(nums * 18, 2)
subset = scores[:, :18].copy()
for i in range(len(subset)):
    for j in range(len(subset[i])):
        subset[i][j] = int(18 * i + j) if subset[i][j] > THRESHOLD else -1

# dwpose/util.py draw_bodypose.
canvas = np.zeros((H, W, 3), dtype=np.uint8)
stickwidth = 4
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
           [1, 16], [16, 18], [3, 17], [6, 18]]
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
          [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
          [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
          [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
          [255, 0, 170], [255, 0, 85]]
for i in range(17):
    for n in range(len(subset)):
        index = subset[n][np.array(limbSeq[i]) - 1]
        if -1 in index:
            continue
        Y = body[index.astype(int), 0] * float(W)
        X = body[index.astype(int), 1] * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth),
                                   int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, colors[i])
canvas = (canvas * 0.6).astype(np.uint8)
for i in range(18):
    for n in range(len(subset)):
        index = int(subset[n][i])
        if index == -1:
            continue
        x, y = body[index][0:2]
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

# cv2.ellipse2Poly alone: every angle of its sine table and negative angles.
rng = np.random.default_rng(7)
ellipses = [[40, 30, 60, 60, 0], [0, 0, 0, 0, 0], [5, -3, 0, 4, 90],
            [10, 10, 7, 4, -1]]
for _ in range(28):
    ellipses.append([int(rng.integers(-20, 120)), int(rng.integers(-20, 90)),
                     int(rng.integers(0, 50)), int(rng.integers(0, 9)),
                     int(rng.integers(-200, 400))])
ellipse_points = [cv2.ellipse2Poly((e[0], e[1]), (e[2], e[3]), e[4], 0, 360, 1)
                  .reshape(-1).tolist() for e in ellipses]

flat = canvas.reshape(-1)
nonzero = np.nonzero(flat)[0]
out = {
    'width': W, 'height': H, 'threshold': THRESHOLD,
    'persons': PERSONS,
    'ellipses': ellipses, 'ellipse_points': ellipse_points,
    'nonzero_pos': [int(p) for p in nonzero],
    'nonzero_value': [int(flat[p]) for p in nonzero],
}
with open(args.out, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print('wrote', args.out, len(nonzero), 'nonzero values;',
      'visible joints', int((subset >= 0).sum()))
