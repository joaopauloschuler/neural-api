#!/usr/bin/env python3
"""Generate a tiny RANDOM VITS parity fixture exercising the STOCHASTIC
duration predictor (use_stochastic_duration_prediction=True, the path used by
kakao-enterprise/vits-ljs) for tests/TestNeuralPretrained.pas.

This is the sibling of tools/make_pico_vits_fixture.py: same end-to-end VITS
inference graph, but the deterministic duration readout is replaced by the
stochastic-duration NORMALIZING FLOW (a stack of VitsConvFlow rational-quadratic
spline coupling layers + a VitsElementwiseAffine, run in REVERSE on a noise
sample, conditioned on the text-encoder hidden states via a
VitsDilatedDepthSeparableConv). The reverse SDP maps a pinned 2-channel noise
`z_dur` to per-token log-durations.

VITS injects noise in two places at inference: the duration-predictor noise
(z_dur, only for the stochastic path) and the prior noise z. BOTH are pinned
here and fed to the Pascal forward as explicit inputs so the comparison does
not depend on RNG matching.

Fixtures, KB-scale, pinned in tests/fixtures/:

  tiny_vits_sdp.safetensors: a VitsModel at pico width with the STOCHASTIC
      duration predictor (1 speaker).
  tiny_vits_sdp_config.json: the pico HF VitsConfig.
  tiny_vits_sdp_ref.json: the float64 oracle. Pins:
        - input_ids;
        - "prior_means"/"prior_log_variances": [tokens][flow_size];
        - "z_dur": the pinned stochastic-duration noise [2][tokens]
          (already multiplied by noise_scale_duration, the reverse SDP input);
        - "log_duration": SDP reverse output [tokens];
        - "durations": ceil(exp(log_dur)) [tokens];
        - "z": the pinned standard-normal prior noise [flow_size][out_len];
        - "prior_latents": expanded mean + z*exp(logvar) [flow_size][out_len];
        - "flow_out": flow(prior_latents, reverse=True) [flow_size][out_len];
        - "waveform": the final synthesized waveform [num_samples].

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_vits_sdp_fixture.py
writes tests/fixtures/tiny_vits_sdp{.safetensors,_config.json,_ref.json}.
Needs torch + transformers + safetensors + numpy.
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import VitsModel, VitsConfig

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
        use_stochastic_duration_prediction=True,
        duration_predictor_kernel_size=3,
        duration_predictor_filter_channels=16,
        duration_predictor_flow_bins=10,
        duration_predictor_tail_bound=5.0,
        duration_predictor_num_flows=2,
        depth_separable_channels=2,
        depth_separable_num_layers=3,
        window_size=4,
        sampling_rate=16000,
        speaking_rate=1.0,
        noise_scale=0.667,
        noise_scale_duration=0.8,
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
    # Re-randomize weights onto an O(1), coarse-grid scale so f32 vs f64 cannot
    # disagree and the spline / tanh paths are exercised away from the linear
    # regime. The random HF init (std 0.02) is vacuous (ModernBERT lesson).
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(coarsen(p).to(p.dtype))
        # Keep the SDP flow proj weights smallish so the rational-quadratic
        # spline arguments stay in a sane range and log-durations land in a
        # 1-3 frames/token band (short out_len -> fast test). The proj of each
        # VitsConvFlow is conv_proj (filter -> half*(3*bins-1)).
        for fl in model.duration_predictor.flows:
            if hasattr(fl, "conv_proj"):
                fl.conv_proj.weight.mul_(0.3)
                fl.conv_proj.bias.mul_(0.3)
        # Bias the elementwise-affine log_scale / translate toward zero so the
        # mapped log-durations stay modest.
        for fl in model.duration_predictor.flows:
            if hasattr(fl, "log_scale"):
                fl.log_scale.mul_(0.3)
                fl.translate.mul_(0.3)

    sd_f32 = {k: v.to(torch.float32).contiguous()
              for k, v in model.state_dict().items()
              if not k.startswith("posterior_encoder.")
              and not k.startswith("duration_predictor.post_")}

    # ---- float64 oracle.
    model64 = VitsModel(cfg).double().eval()
    model64.load_state_dict(model.state_dict())

    input_ids = torch.tensor([[2, 5, 7, 3, 9, 1]], dtype=torch.long)
    pad = torch.ones_like(input_ids).unsqueeze(-1).double()  # (1,T,1)
    T = input_ids.shape[1]

    # Pinned stochastic-duration noise. HF draws randn(bsz,2,T)*noise_scale and
    # the spline path requires |latents| < tail_bound (=5). Round to coarse grid.
    rng = np.random.RandomState(SEED + 7)
    zdur_np = np.round(rng.randn(1, 2, T) * 64.0) / 64.0          # standard
    zdur = torch.tensor(zdur_np, dtype=torch.float64)
    zdur_scaled = zdur * cfg.noise_scale_duration                 # the SDP input

    with torch.no_grad():
        te = model64.text_encoder(input_ids, pad, attention_mask=None,
                                  return_dict=True)
        hidden = te.last_hidden_state.transpose(1, 2)   # (1,H,T)
        prior_means = te.prior_means                    # (1,T,flow)
        prior_logv = te.prior_log_variances             # (1,T,flow)
        ppad = pad.transpose(1, 2)                       # (1,1,T)

        # Run the reverse SDP with the pinned noise. We replicate the HF reverse
        # path but inject our pinned `latents` (= zdur_scaled) instead of randn.
        dp = model64.duration_predictor
        inputs = dp.conv_pre(hidden)
        inputs = dp.conv_dds(inputs, ppad)
        inputs = dp.conv_proj(inputs) * ppad

        flows = list(reversed(dp.flows))
        flows = flows[:-2] + [flows[-1]]   # remove a useless vflow
        latents = zdur_scaled.clone()
        for flow in flows:
            latents = torch.flip(latents, [1])
            latents, _ = flow(latents, ppad, global_conditioning=inputs,
                              reverse=True)
        log_dur = latents[:, 0:1, :]                     # (1,1,T)

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

        rng2 = np.random.RandomState(SEED)
        z_np = np.round(rng2.randn(1, cfg.flow_size, out_len) * 64.0) / 64.0
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
        "noise_scale_duration": cfg.noise_scale_duration,
        "input_ids": input_ids.squeeze(0).tolist(),
        "out_len": out_len,
        "prior_means": prior_means.squeeze(0).cpu().numpy().tolist(),
        "prior_log_variances": prior_logv.squeeze(0).cpu().numpy().tolist(),
        "z_dur": zdur_scaled.squeeze(0).cpu().numpy().tolist(),                # [2][T]
        "log_duration": log_dur.squeeze(0).squeeze(0).cpu().numpy().tolist(),  # [T]
        "durations": duration.squeeze(0).squeeze(0).cpu().numpy().tolist(),    # [T]
        "z": z.squeeze(0).cpu().numpy().tolist(),
        "prior_latents": prior_latents.squeeze(0).cpu().numpy().tolist(),
        "flow_out": flow_out.squeeze(0).cpu().numpy().tolist(),
        "waveform": waveform.cpu().numpy().tolist(),
    }

    os.makedirs(FIX, exist_ok=True)
    save_file(sd_f32, os.path.join(FIX, "tiny_vits_sdp.safetensors"))
    with open(os.path.join(FIX, "tiny_vits_sdp_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1, default=str)
    with open(os.path.join(FIX, "tiny_vits_sdp_ref.json"), "w") as f:
        json.dump(ref, f)

    st = os.path.getsize(os.path.join(FIX, "tiny_vits_sdp.safetensors"))
    print("wrote tiny_vits_sdp.safetensors %d bytes" % st)
    print("tokens", len(ref["input_ids"]), "out_len", out_len,
          "log_dur", ref["log_duration"], "durations", ref["durations"])


if __name__ == "__main__":
    main()
