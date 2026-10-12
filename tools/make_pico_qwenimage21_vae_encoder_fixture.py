#!/usr/bin/env python3
"""Generate the Qwen-Image-2.1 VAE ENCODER parity oracle (tasklist C2) from the
pico VAE that tools/make_pico_qwenimage21_fixture.py already wrote to
tests/fixtures/tiny_qwenimage21/vae (its encoder weights; nothing is regenerated).

Output: tests/fixtures/tiny_qwenimage21_vae_encoder_io.json
  images: one entry per (height, width) in ENCODE_SIZES:
    posterior_mean      DiagonalGaussianDistribution(quant_conv(encoder(x))).mean
    latents_normalized  (posterior_mean - latents_mean) / latents_std
  avg_down: QwenImage21AvgDown3D outputs for AVG_DOWN_CASES on one frame.

The inputs are not stored: every input value is
  ((i * 7919 + 13) mod 2049) / 1024 - 1
with i the row-major (C, H, W) index, exact in float32 and inside [-1, 1].
The encode runs in float64 on the F32 weights (the encoder has no float32
casts to shim); native_hf_maxabs_diff is the float32 run against it.

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv, torch CPU build):
  ( ulimit -v 3145728; python3 tools/make_pico_qwenimage21_vae_encoder_fixture.py )
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pico_qwenimage21_fixture import (PICO_DIR, AutoencoderKLQwenImage21,  # noqa: E402
                                           max_abs_diff, tensor_json, vae_module,
                                           write_json)

ENCODE_SIZES = [(32, 48), (64, 64)]                    # (height, width), multiples of 16
# (in_channels, out_channels, factor_t, factor_s)
AVG_DOWN_CASES = [(2, 2, 1, 2), (2, 4, 2, 2), (4, 8, 1, 2), (3, 6, 2, 2), (4, 4, 1, 1),
                  (4, 4, 2, 1), (2, 8, 2, 2)]
AVG_DOWN_HW = (4, 6)                                  # (height, width) of the AvgDown input


def formula_input(channels, height, width, dtype=torch.float64):
    index = torch.arange(channels * height * width, dtype=torch.int64)
    values = ((index * 7919 + 13) % 2049).to(dtype) / 1024 - 1
    return values.view(1, channels, 1, height, width)


def time_conv_calls(vae):
    calls = []
    handles = [module.time_conv.register_forward_hook(lambda m, a, o, name=name: calls.append(name))
               for name, module in vae.named_modules()
               if isinstance(module, vae_module.QwenImage21Resample) and hasattr(module, "time_conv")]
    return calls, handles


def main():
    torch.set_num_threads(1)
    vae_dir = os.path.join(PICO_DIR, "vae")
    vae = AutoencoderKLQwenImage21.from_pretrained(vae_dir, torch_dtype=torch.float64).eval()
    native = AutoencoderKLQwenImage21.from_pretrained(vae_dir, torch_dtype=torch.float32).eval()
    z_dim = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float64).view(1, z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float64).view(1, z_dim, 1, 1, 1)
    calls, handles = time_conv_calls(vae)
    images = []
    with torch.no_grad():
        for height, width in ENCODE_SIZES:
            x = formula_input(vae.config.in_channels, height, width)
            posterior_mean = vae.encode(x).latent_dist.mean
            native_mean = native.encode(x.to(torch.float32)).latent_dist.mean
            images.append({
                "height": height,
                "width": width,
                "posterior_mean": tensor_json(posterior_mean[0, :, 0]),
                "latents_normalized": tensor_json(((posterior_mean - mean) / std)[0, :, 0]),
                "native_hf_maxabs_diff": max_abs_diff(posterior_mean, native_mean),
            })
        avg_down = []
        for in_ch, out_ch, factor_t, factor_s in AVG_DOWN_CASES:
            x = formula_input(in_ch, *AVG_DOWN_HW)
            module = vae_module.QwenImage21AvgDown3D(in_ch, out_ch, factor_t, factor_s)
            avg_down.append({"in_channels": in_ch, "out_channels": out_ch, "factor_t": factor_t,
                             "factor_s": factor_s, "output": tensor_json(module(x)[0, :, 0])})
    for handle in handles:
        handle.remove()
    write_json("tiny_qwenimage21_vae_encoder_io.json", {
        "images": images,
        "avg_down_input_hw": list(AVG_DOWN_HW),
        "avg_down": avg_down,
        "time_conv_calls_during_encode": len(calls),
        "note": "Inputs are ((i * 7919 + 13) mod 2049) / 1024 - 1, i = row-major (C, H, W) index of a "
                "(1, C, 1, H, W) tensor. Tensors are (C, H, W) with the frame axis dropped. "
                "posterior_mean = vae.encode(x).latent_dist.mean in float64; latents_normalized = "
                "(posterior_mean - latents_mean) / latents_std per channel.",
    })
    print(f"time_conv calls during {len(ENCODE_SIZES)} encodes: {len(calls)}")


if __name__ == "__main__":
    main()
