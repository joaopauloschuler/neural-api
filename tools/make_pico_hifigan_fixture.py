#!/usr/bin/env python3
"""Generate a tiny RANDOM HiFi-GAN parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a PICO config, never downloaded).

HiFi-GAN (the `hifigan` generator shipped with most TTS stacks - Tacotron2 /
FastSpeech2 / SpeechT5 / Bark / VITS) is a PURELY CONVOLUTIONAL neural
vocoder: a log-mel spectrogram (model_in_dim mel bands over time) ->
conv_pre -> a stack of (LeakyReLU -> ConvTranspose1d upsample -> Multi-
Receptive-Field residual module) stages -> LeakyReLU -> conv_post -> tanh ->
raw mono waveform. This fixture pins the WHOLE mel -> waveform pass of a pico
instance, using the HF SpeechT5HifiGan implementation as the float64 oracle.

Fixtures, KB-scale, pinned in tests/fixtures/:

  tiny_hifigan.safetensors: a SpeechT5HifiGan generator at pico width with
      every trait the importer must reproduce:
        - conv_pre Conv1d(model_in_dim -> upsample_initial_channel, k=7, pad3);
        - per-stage ConvTranspose1d upsample (k=upsample_kernel_sizes[i],
          stride=upsample_rates[i], pad=(k-stride)//2), channels halving;
        - MRF = AVERAGE (not sum) of num_kernels ResBlocks; each ResBlock is a
          chain of (LeakyReLU0.1 -> dilated Conv1d -> LeakyReLU0.1 -> Conv1d)
          residual adds with the per-block dilation list;
        - conv_post Conv1d(C -> 1, k=7, pad3) and a FINAL LeakyReLU at the
          PyTorch DEFAULT slope (0.01, not 0.1) before it, then tanh.
      SpeechT5HifiGan ships the weights as plain (folded) .weight tensors
      (it removes weight_norm at save); the importer ALSO folds weight_norm
      g/v pairs for the bare upstream `hifigan` generator (not exercised here).
      normalize_before=False (no mean/scale path).

  tiny_hifigan_config.json: the pico HF SpeechT5HifiGanConfig.

  tiny_hifigan_ref.json: the float64 oracle. For each pinned mel clip:
        - "mel": the input log-mel [frames][bands];
        - "waveform": the synthesized waveform (the array the Pascal test
          gates < 1e-4), computed by the HF model in float64.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_hifigan_fixture.py
writes tests/fixtures/tiny_hifigan{.safetensors,_config.json,_ref.json}.
Needs torch + transformers + safetensors + numpy.
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

SEED = 4242


def pico_config():
    return SpeechT5HifiGanConfig(
        model_in_dim=4,                 # n_mels
        upsample_initial_channel=8,
        upsample_rates=[2, 2],          # total upsample x4
        upsample_kernel_sizes=[4, 4],   # (k-stride)//2 = 1 padding each
        resblock_kernel_sizes=[3, 5],   # num_kernels = 2 (MRF average of 2)
        resblock_dilation_sizes=[[1, 2], [2, 4]],
        leaky_relu_slope=0.1,
        normalize_before=False,
        sampling_rate=8000,
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = pico_config()

    model = SpeechT5HifiGan(cfg).eval()
    # Re-randomize all weights onto an O(1) scale (the default HF init is
    # tiny, which would make the tanh output near-linear and the parity test
    # vacuous). Round onto a coarse grid so f32 vs f64 cannot disagree.
    with torch.no_grad():
        for p in model.parameters():
            base = torch.randn_like(p, dtype=torch.float64) * 0.3
            base = torch.round(base * 64.0) / 64.0
            p.copy_(base.to(p.dtype))

    # ---- pinned mel clips: a few frames, model_in_dim bands.
    rng = np.random.RandomState(SEED)
    n_frames = 5
    clips = []
    for _ in range(3):
        mel = (rng.randn(n_frames, cfg.model_in_dim).astype(np.float64) * 0.5)
        mel = np.round(mel * 64.0) / 64.0  # exactly representable
        clips.append(mel)

    # ---- float64 oracle.
    model_f64 = SpeechT5HifiGan(cfg).double().eval()
    model_f64.load_state_dict(model.state_dict())

    refs = []
    with torch.no_grad():
        for mel in clips:
            inp = torch.tensor(mel, dtype=torch.float64)  # (frames, bands)
            wav = model_f64(inp)  # un-batched -> (num_frames,)
            refs.append({
                "mel": mel.tolist(),
                "waveform": wav.detach().cpu().numpy().astype(np.float64).tolist(),
            })

    # ---- save safetensors (raw HF state dict, F32).
    os.makedirs(FIX, exist_ok=True)
    sd = {}
    for k, v in model.state_dict().items():
        sd[k] = v.to(torch.float32).contiguous()
    save_file(sd, os.path.join(FIX, "tiny_hifigan.safetensors"))

    with open(os.path.join(FIX, "tiny_hifigan_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1, default=str)

    with open(os.path.join(FIX, "tiny_hifigan_ref.json"), "w") as f:
        json.dump({
            "model_in_dim": cfg.model_in_dim,
            "sampling_rate": cfg.sampling_rate,
            "clips": refs,
        }, f)

    st = os.path.getsize(os.path.join(FIX, "tiny_hifigan.safetensors"))
    print("wrote tiny_hifigan.safetensors %d bytes" % st)
    print("frames", n_frames, "-> waveform len", len(refs[0]["waveform"]))


if __name__ == "__main__":
    main()
