#!/usr/bin/env python3
"""Generate a tiny CLIP image-preprocessing parity fixture for
tests/TestNeuralPretrained.pas (Step 2 of the LLaVA import).

Pins a small preprocessor_config.json and two synthetic RGB images plus the
HF CLIPImageProcessor reference for the normalized RGB tensor:

  - parity image: ALREADY at the processor working size (crop == size), so
    resize and center-crop are identities and the rescale+normalize path is
    byte-exact vs HF (HF's default resample is bicubic - we do not claim
    bicubic parity, only the rescale/normalize path).
  - crop image: 2x larger than the crop, so center-crop is an EXACT integer
    crop (no interpolation) - exercises the crop offset arithmetic with
    byte-exact reference too (do_resize disabled in this pass).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/clip_image_preprocess_fixture.py
writes tests/fixtures/tiny_clip_preprocessor_config.json and
tests/fixtures/tiny_clip_preprocess.json. Needs transformers + numpy.
"""
import json

import numpy as np
from transformers import CLIPImageProcessor

SIZE = 8          # tiny working size (shortest_edge == crop == 8)
CROP = 8

config = dict(
    image_processor_type='CLIPImageProcessor',
    do_resize=True,
    size={'shortest_edge': SIZE},
    do_center_crop=True,
    crop_size={'height': CROP, 'width': CROP},
    do_rescale=True,
    rescale_factor=1.0 / 255.0,
    do_normalize=True,
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
    do_convert_rgb=True,
)
with open('tests/fixtures/tiny_clip_preprocessor_config.json', 'w') as f:
    json.dump(config, f, indent=1)

proc = CLIPImageProcessor(**{k: v for k, v in config.items()
                             if k != 'image_processor_type'})


def make_image(h, w):
    # deterministic byte image, HWC uint8, RGB
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                img[y, x, c] = (c * 256 + y * 16 + x * 7) % 256
    return img


# --- parity image: already at working size (resize+crop are identities) ---
parity_img = make_image(SIZE, SIZE)
# pixel_values: [1][3][CROP][CROP]
parity_ref = proc(images=parity_img, return_tensors='np').pixel_values[0]

# --- crop image: 2x crop, resize disabled so center-crop is exact integer ---
proc_nocrop = CLIPImageProcessor(
    do_resize=False, size={'shortest_edge': SIZE},
    do_center_crop=True, crop_size={'height': CROP, 'width': CROP},
    do_rescale=True, rescale_factor=1.0 / 255.0, do_normalize=True,
    image_mean=config['image_mean'], image_std=config['image_std'],
    do_convert_rgb=True)
crop_img = make_image(2 * CROP, 2 * CROP)
crop_ref = proc_nocrop(images=crop_img, return_tensors='np').pixel_values[0]

with open('tests/fixtures/tiny_clip_preprocess.json', 'w') as f:
    json.dump({
        'size': SIZE,
        'crop': CROP,
        # raw uint8 images, HWC layout (what the test loads into (W,H,3))
        'parity_image': parity_img.tolist(),         # [SIZE][SIZE][3]
        'parity_ref': parity_ref.tolist(),           # [3][CROP][CROP]
        'crop_image': crop_img.tolist(),             # [2C][2C][3]
        'crop_ref': crop_ref.tolist(),               # [3][CROP][CROP]
    }, f)

print('wrote tiny_clip_preprocessor_config.json + tiny_clip_preprocess.json')
print('parity_ref shape', parity_ref.shape, 'crop_ref shape', crop_ref.shape)
# self-check: normalize math is what we expect on a known pixel
mean = np.array(config['image_mean']).reshape(3, 1, 1)
std = np.array(config['image_std']).reshape(3, 1, 1)
manual = (parity_img.transpose(2, 0, 1).astype(np.float64) / 255.0
          - mean) / std
assert np.allclose(manual, parity_ref, atol=1e-5), 'normalize mismatch'
print('normalize self-check passed')
