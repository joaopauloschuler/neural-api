#!/usr/bin/env python3
"""Generate a tiny RANDOM VITS / MMS-TTS parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a PICO config, never downloaded).

VITS (Kim et al. 2021, the architecture behind facebook/mms-tts-* and
kakao-enterprise/vits-ljs) is an end-to-end text-to-speech model. At
INFERENCE (the only path imported here) it is:

  input_ids -> text_encoder (relative-position transformer) -> per-token
    prior mean/logvar + hidden;
  deterministic duration_predictor -> ceil(exp(log_dur)) frames per token;
  length regulator EXPANDS prior_mean/logvar along time by the durations;
  prior_latents = mean + z * exp(logvar) * noise_scale   (z = standard noise);
  flow (RealNVP/Glow residual coupling, run in REVERSE) -> spectrogram latent;
  HiFi-GAN decoder (the same generator the HiFi-GAN importer builds) -> wave.

This fixture uses the DETERMINISTIC duration predictor
(use_stochastic_duration_prediction=False) and a SINGLE speaker (no global
conditioning) - exactly the inference path the Pascal importer supports.

VITS sampling injects noise in two places (the posterior is dropped at
inference; only the prior noise z remains). For an EXACT parity test the
oracle's z is pinned and fed to the Pascal forward as an explicit input, so
the comparison does not depend on RNG matching.

Fixtures, KB-scale, pinned in tests/fixtures/:

  tiny_vits.safetensors: a VitsModel at pico width (deterministic duration,
      1 speaker) with every trait the importer must reproduce.
  tiny_vits_config.json: the pico HF VitsConfig.
  tiny_vits_ref.json: the float64 oracle. Pins:
        - input_ids;
        - "prior_means"/"prior_log_variances": text-encoder prior stats
          [tokens][flow_size];
        - "log_duration": deterministic duration predictor output [tokens];
        - "durations": ceil(exp(log_dur)) [tokens] (length regulator);
        - "z": the pinned standard-normal prior noise [flow_size][out_len];
        - "prior_latents": expanded mean + z*exp(logvar) [flow_size][out_len]
          (the flow INPUT);
        - "flow_out": flow(prior_latents, reverse=True) [flow_size][out_len]
          (the decoder INPUT / spectrogram);
        - "waveform": the final synthesized waveform [num_samples].

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_vits_fixture.py
writes tests/fixtures/tiny_vits{.safetensors,_config.json,_ref.json}.
Needs torch + transformers + safetensors + numpy.
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import VitsModel, VitsConfig
import transformers.models.vits.modeling_vits as mv

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

SEED = 4242


def pico_config():
    return VitsConfig(
        vocab_size=12,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        ffn_dim=16,
        ffn_kernel_size=3,
        flow_size=8,
        prior_encoder_num_flows=2,
        prior_encoder_num_wavenet_layers=2,
        wavenet_kernel_size=5,
        wavenet_dilation_rate=1,
        upsample_initial_channel=8,
        upsample_rates=[4, 4],
        upsample_kernel_sizes=[8, 8],
        resblock_kernel_sizes=[3, 7],
        resblock_dilation_sizes=[[1, 3], [1, 3]],
        use_stochastic_duration_prediction=False,
        duration_predictor_kernel_size=3,
        duration_predictor_filter_channels=16,
        window_size=4,
        sampling_rate=16000,
        speaking_rate=1.0,
        noise_scale=0.667,
        num_speakers=1,
        speaker_embedding_size=0,
        layer_norm_eps=1e-5,
        hidden_act="relu",
    )


def coarsen(p):
    base = torch.randn_like(p, dtype=torch.float64) * 0.3
    base = torch.round(base * 64.0) / 64.0
    return base


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = pico_config()

    model = VitsModel(cfg).eval()
    # Re-randomize weights onto an O(1), coarse-grid scale so f32 vs f64
    # cannot disagree and the tanh decoder is not near-linear. weight_norm
    # tensors (parametrizations.weight.original0/1) and plain weights alike.
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(coarsen(p).to(p.dtype))
        # Bias the duration log-output small so ceil(exp(log_dur)) stays in a
        # sane 1-3 frames/token range (keeps out_len short & the test fast).
        model.duration_predictor.proj.bias.copy_(
            torch.zeros_like(model.duration_predictor.proj.bias))
        model.duration_predictor.proj.weight.mul_(0.2)

    # The posterior_encoder is TRAINING-ONLY (inference runs the flow in
    # reverse, never the posterior) - drop it to keep the committed fixture
    # tiny. The importer must likewise never require posterior_encoder.*.
    sd_f32 = {k: v.to(torch.float32).contiguous()
              for k, v in model.state_dict().items()
              if not k.startswith("posterior_encoder.")}

    # ---- float64 oracle.
    model64 = VitsModel(cfg).double().eval()
    model64.load_state_dict(model.state_dict())

    input_ids = torch.tensor([[2, 5, 7, 3, 9, 1]], dtype=torch.long)
    pad = torch.ones_like(input_ids).unsqueeze(-1).double()  # (1,T,1)

    with torch.no_grad():
        te = model64.text_encoder(input_ids, pad, attention_mask=None,
                                  return_dict=True)
        hidden = te.last_hidden_state.transpose(1, 2)   # (1,H,T)
        prior_means = te.prior_means                    # (1,T,flow)
        prior_logv = te.prior_log_variances             # (1,T,flow)
        ppad = pad.transpose(1, 2)                       # (1,1,T)

        log_dur = model64.duration_predictor(hidden, ppad, None)  # (1,1,T)
        length_scale = 1.0
        duration = torch.ceil(torch.exp(log_dur) * ppad * length_scale)
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

        out_len = int(predicted_lengths.max())
        opad = (torch.arange(out_len).unsqueeze(0) < predicted_lengths.unsqueeze(1))
        opad = opad.unsqueeze(1).double()                # (1,1,out_len)
        attn_mask = torch.unsqueeze(ppad, 2) * torch.unsqueeze(opad, -1)
        bsz, _, output_length, input_length = attn_mask.shape
        cum = torch.cumsum(duration, -1).view(bsz * input_length, 1)
        idx = torch.arange(output_length).double()
        valid = (idx.unsqueeze(0) < cum).double().view(bsz, input_length, output_length)
        padded = valid - torch.nn.functional.pad(valid, [0, 0, 1, 0, 0, 0])[:, :-1]
        attn = padded.unsqueeze(1).transpose(2, 3) * attn_mask  # (1,1,out,in)

        em = torch.matmul(attn.squeeze(1), prior_means).transpose(1, 2)   # (1,flow,out)
        ev = torch.matmul(attn.squeeze(1), prior_logv).transpose(1, 2)    # (1,flow,out)

        # Pinned standard-normal z (the prior noise) - shape (1,flow,out).
        rng = np.random.RandomState(SEED)
        z_np = np.round(rng.randn(1, cfg.flow_size, out_len) * 64.0) / 64.0
        z = torch.tensor(z_np, dtype=torch.float64)

        prior_latents = em + z * torch.exp(ev) * cfg.noise_scale
        flow_out = model64.flow(prior_latents, opad, None, reverse=True)
        spectrogram = flow_out * opad
        waveform = model64.decoder(spectrogram, None).squeeze(1).squeeze(0)

    ref = {
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "flow_size": cfg.flow_size,
        "sampling_rate": cfg.sampling_rate,
        "noise_scale": cfg.noise_scale,
        "input_ids": input_ids.squeeze(0).tolist(),
        "out_len": out_len,
        "prior_means": prior_means.squeeze(0).cpu().numpy().tolist(),         # [T][flow]
        "prior_log_variances": prior_logv.squeeze(0).cpu().numpy().tolist(),  # [T][flow]
        "log_duration": log_dur.squeeze(0).squeeze(0).cpu().numpy().tolist(), # [T]
        "durations": duration.squeeze(0).squeeze(0).cpu().numpy().tolist(),   # [T]
        "z": z.squeeze(0).cpu().numpy().tolist(),                             # [flow][out]
        "prior_latents": prior_latents.squeeze(0).cpu().numpy().tolist(),     # [flow][out]
        "flow_out": flow_out.squeeze(0).cpu().numpy().tolist(),               # [flow][out]
        "waveform": waveform.cpu().numpy().tolist(),                          # [samples]
    }

    os.makedirs(FIX, exist_ok=True)
    save_file(sd_f32, os.path.join(FIX, "tiny_vits.safetensors"))
    with open(os.path.join(FIX, "tiny_vits_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1, default=str)
    with open(os.path.join(FIX, "tiny_vits_ref.json"), "w") as f:
        json.dump(ref, f)

    st = os.path.getsize(os.path.join(FIX, "tiny_vits.safetensors"))
    print("wrote tiny_vits.safetensors %d bytes" % st)
    print("tokens", len(ref["input_ids"]), "out_len", out_len,
          "waveform", len(ref["waveform"]))


if __name__ == "__main__":
    main()
