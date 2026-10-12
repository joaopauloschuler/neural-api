#!/usr/bin/env python3
"""torch oracle for TNNetForwardSTFT and TNNetHarmonicSource (neuralaudio.pas;
tasklist Kokoro-82M, 2.8 and 2.9).

stft cases: torch.stft with the iSTFTNet call (center=True, reflect pad,
  periodic Hann window, onesided, return_complex) on short float32 signals;
  n_fft 20 / hop 5 (Kokoro), n_fft 16 / hop 4 and an odd n_fft 15 / hop 3.
  Stored: re, im, magnitude (torch.abs) and phase (torch.angle), [bins][frames].
harmonic_source case: the kokoro package's SourceModuleHnNSF (SineGen,
  flag_for_pulse off) with zero noise, small sample rate and upsample scale,
  an F0 curve with unvoiced, negative and high frames. Stored: the F0 frames,
  l_linear weights/bias, f0 upsampled, uv and har_source per sample.

Writes tests/fixtures/stft_harmonic_source_oracle.json. Run:
  ( ulimit -v 3145728; /home/bpsa/x/bin/python -I tools/stft_harmonic_source_oracle.py )
"""
import json
import math
import os

import torch
import kokoro.istftnet as istftnet


class _ZeroNoiseTorch:
    """Stands in for `torch` inside kokoro.istftnet: rand / randn_like return zeros."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def rand(*args, **kwargs):
        return torch.zeros(*args, **kwargs)

    @staticmethod
    def randn_like(tensor, *args, **kwargs):
        return torch.zeros_like(tensor, *args, **kwargs)


istftnet.torch = _ZeroNoiseTorch()

STFT_CASES = [
    # name, n_fft, hop, length, signal
    ("kokoro_sinusoid", 20, 5, 103, "sinusoid"),
    ("nfft16_random", 16, 4, 61, "random"),
    ("nfft15_odd_random", 15, 3, 40, "random"),
]


def stft_signal(kind, length, generator):
    if kind == "sinusoid":
        t = torch.arange(length, dtype=torch.float32)
        return 0.7 * torch.sin(2 * math.pi * 3.3 * t / 20 + 0.4) + 0.1
    return torch.round(torch.randn(length, generator=generator), decimals=4)


def main():
    generator = torch.Generator().manual_seed(20261012)
    stft_cases = []
    for name, n_fft, hop, length, kind in STFT_CASES:
        x = stft_signal(kind, length, generator)
        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
        spec = torch.stft(x[None], n_fft, hop, n_fft, window=window,
                          return_complex=True)[0]
        stft_cases.append({
            "name": name, "n_fft": n_fft, "hop": hop, "length": length,
            "input": x.tolist(),
            "re": spec.real.tolist(), "im": spec.imag.tolist(),
            "magnitude": spec.abs().tolist(), "phase": spec.angle().tolist(),
        })

    sample_rate, upsample_scale, harmonic_num, threshold = 240, 6, 3, 10
    source = istftnet.SourceModuleHnNSF(sample_rate, upsample_scale,
                                        harmonic_num=harmonic_num,
                                        voiced_threshod=threshold)
    with torch.no_grad():
        source.l_linear.weight.copy_(torch.tensor([[0.6, -0.35, 0.25, 0.8]]))
        source.l_linear.bias.copy_(torch.tensor([0.05]))
    f0 = torch.tensor([[0.0, 5.0, 12.0, 30.5, 47.0, 100.0, -3.0, 15.0, 11.0,
                        9.0, 60.0, 60.0, 333.3, 81.25]])
    upsample = torch.nn.Upsample(scale_factor=upsample_scale)
    with torch.no_grad():
        f0_up = upsample(f0[:, None]).transpose(1, 2)
        har_source, _noise, uv = source(f0_up)
    harmonic_case = {
        "sample_rate": sample_rate, "upsample_scale": upsample_scale,
        "harmonic_count": harmonic_num + 1, "voiced_threshold": threshold,
        "sine_amp": 0.1, "noise_std": 0.003,
        "f0": f0[0].tolist(),
        "merge_weights": source.l_linear.weight[0].tolist(),
        "merge_bias": source.l_linear.bias.item(),
        "f0_upsampled": f0_up[0, :, 0].tolist(),
        "uv": uv[0, :, 0].tolist(),
        "har_source": har_source[0, :, 0].tolist(),
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "tests", "fixtures", "stft_harmonic_source_oracle.json")
    with open(path, "w") as f:
        json.dump({"torch": torch.__version__, "stft_cases": stft_cases,
                   "harmonic_source": harmonic_case}, f)
    print("wrote", os.path.normpath(path), len(stft_cases), "stft cases")


if __name__ == "__main__":
    main()
