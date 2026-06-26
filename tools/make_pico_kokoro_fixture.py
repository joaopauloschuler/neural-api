#!/usr/bin/env python3
"""Generate a tiny RANDOM Kokoro / StyleTTS2 parity fixture for
tests/TestNeuralPretrained.pas.

The `kokoro` / HF python packages are NOT installed in this environment, so
this generator is fully SELF-CONTAINED: it defines a PICO StyleTTS2-shaped
forward graph in pure numpy float64 (the same stance the pyannote / ModernBERT
fixtures take) and pins its outputs as the oracle. The Pascal TNNetKokoro
holder reimplements EXACTLY this forward math, so the parity test gates the
Pascal importer against this float64 reference at < 1e-4.

Kokoro (hexgrad/Kokoro-82M, Apache-2.0) is a StyleTTS2 model. The three
distinctive StyleTTS2 pieces this fixture exercises (vs the landed VITS path):

  (1) STYLE-VECTOR CONDITIONING. A 256-d voice/style vector is split into a
      prosody half s_pred (first 128) and an acoustic/decoder half s_dec
      (last 128). Each is AdaIN/affine-injected:
          AdaIN1d(x, s) = gamma(s) * InstanceNorm_channels(x) + beta(s)
      where [gamma; beta] = fc(s) (a linear map s -> 2*channels). This is the
      new conditioning math vs VITS's WaveNet `cond` convs.

  (2) iSTFTNet DECODER. The generator predicts a magnitude + phase
      spectrogram and runs an INVERSE STFT (overlap-add) to the waveform,
      rather than HiFi-GAN's pure transposed-conv upsampling. Pascal reuses
      the LANDED ISTFTOverlapAdd(Mag, Phase, ...) primitive in neuralaudio.pas.

  (3) PROSODY / DURATION STACK. A style-conditioned duration predictor drives
      a length-regulator that expands the per-token text encoding along time
      (the monotonic alignment), then style-conditioned F0 and energy (N)
      predictors produce the prosody curves fed to the decoder.

SCOPE v1 (matching the tasklist entry): a single deterministic forward graph
phonemes -> waveform with the reference style vector as an EXPLICIT input. No
sampling, no g2p front-end (pre-phonemized integer ids are the input).

Re-randomized O(1)-scale weights per the ModernBERT fixture lesson (HF
std-0.02 init is vacuous at pico scale: AdaIN gammas near 1, convs near 0
would make every stage near-identity and parity vacuous).

Outputs (tests/fixtures/):
  tiny_kokoro.safetensors : the pico weights the importer loads.
  tiny_kokoro_config.json : the pico TKokoroConfig.
  tiny_kokoro_ref.json    : the float64 oracle (input ids, style vector, and
      every stage output the parity test checks: log_duration, durations,
      f0, energy, magnitude, phase, waveform).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_kokoro_fixture.py
Needs numpy + (torch | safetensors) for the .safetensors writer.
"""
import json
import os

import numpy as np

# --------------------------------------------------------------------------
# PICO config. Small but every dimension > 1 so the math is genuinely
# exercised. H is the model width AND the per-AdaIN channel count; the style
# vector is 2*H (prosody half + acoustic half).
# --------------------------------------------------------------------------
H = 8                 # hidden / channel width (each AdaIN half conditions H ch)
STYLE_DIM = 2 * H     # 16: s_pred = s[:H], s_dec = s[H:]
VOCAB = 24            # phoneme vocab
KENC = 5              # text-encoder conv kernel
KDUR = 3              # duration-predictor conv kernel
KF0 = 3               # f0 / energy conv kernel
KDEC = 3              # decoder conv kernel
NFFT = 16             # iSTFT n_fft -> NFFT//2+1 = 9 magnitude/phase bins
HOP = 4               # iSTFT hop length
SPEED = 1.0           # length scale (duration multiplier)
LN_EPS = 1e-5
SAMPLE_RATE = 24000

# seed 13 yields small, varied durations [1,1,2,1,1,1] (out_len 7) and O(1)
# magnitudes - a clean, non-vacuous pico fixture.
rng = np.random.default_rng(int(os.environ.get("KOKORO_SEED", "13")))


def rw(*shape):
    """Re-randomized O(1)-scale weight (std ~0.6, NOT HF's vacuous 0.02)."""
    return rng.standard_normal(shape) * 0.6


def rb(*shape):
    return rng.standard_normal(shape) * 0.3


# --------------------------------------------------------------------------
# Pico parameters (numpy float64). Channel-major convention throughout:
# a signal is [channels][time]. Conv weight is [out, in, k] ("same" pad k//2).
# --------------------------------------------------------------------------
P = {}
P["embed"] = rw(VOCAB, H)                       # phoneme embedding [V, H]

P["enc_w"] = rw(H, H, KENC); P["enc_b"] = rb(H)  # text-encoder conv

# AdaIN fc maps: s (H,) -> [gamma(H); beta(H)] (2H,). Init gamma near 1 so the
# normalized signal is not annihilated, but with O(1) spread so it bites.
def adain_fc():
    w = rw(2 * H, H) * 0.5
    b = np.concatenate([np.ones(H) + rb(H), rb(H)])  # gamma~1+noise, beta~noise
    return w, b

P["dur_adain_w"], P["dur_adain_b"] = adain_fc()
P["dur_w"] = rw(H, H, KDUR); P["dur_b"] = rb(H)     # duration conv
# small proj -> log_dur in a tight range so durations are small varied ints.
P["dur_proj_w"] = rw(1, H, 1) * 0.5; P["dur_proj_b"] = rb(1) * 0.2 + 0.6

P["f0_adain_w"], P["f0_adain_b"] = adain_fc()
P["f0_w"] = rw(H, H, KF0); P["f0_b"] = rb(H)
P["f0_proj_w"] = rw(1, H, 1); P["f0_proj_b"] = rb(1)

P["n_adain_w"], P["n_adain_b"] = adain_fc()
P["n_w"] = rw(H, H, KF0); P["n_b"] = rb(H)
P["n_proj_w"] = rw(1, H, 1); P["n_proj_b"] = rb(1)

# Decoder: takes expanded hidden (H) + f0 (1) + energy (1) = H+2 input channels.
DEC_IN = H + 2
P["dec_adain_w"] = rw(2 * H, H) * 0.5
P["dec_adain_b"] = np.concatenate([np.ones(H) + rb(H), rb(H)])
P["dec_in_w"] = rw(H, DEC_IN, KDEC); P["dec_in_b"] = rb(H)   # mix to H channels
NBINS = NFFT // 2 + 1
# magnitude head emits LOG-magnitude; small weights keep exp() O(1).
P["mag_w"] = rw(NBINS, H, KDEC) * 0.25; P["mag_b"] = rb(NBINS) * 0.25
P["phase_w"] = rw(NBINS, H, KDEC); P["phase_b"] = rb(NBINS)  # phase head


# --------------------------------------------------------------------------
# Forward math (pure numpy float64). This IS the oracle the Pascal holder
# must match exactly.
# --------------------------------------------------------------------------
def conv1d(x, w, b, k):
    """Channel-major 'same'-padded conv. x:[Cin][T], w:[Cout,Cin,k]."""
    cin, T = x.shape
    cout = w.shape[0]
    pad = k // 2
    xp = np.zeros((cin, T + 2 * pad))
    xp[:, pad:pad + T] = x
    y = np.zeros((cout, T))
    for t in range(T):
        win = xp[:, t:t + k]                 # [Cin, k]
        for o in range(cout):
            y[o, t] = b[o] + np.sum(w[o] * win)
    return y


def instance_norm_ch(x, eps):
    """Per-channel normalization over time (StyleTTS2 AdaIN uses InstanceNorm:
    each channel normalized by its own mean/var across the time axis)."""
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)       # population variance (ddof=0)
    return (x - mean) / np.sqrt(var + eps)


def adain(x, s, fcw, fcb):
    """AdaIN1d: gamma * InstanceNorm(x) + beta, [gamma;beta]=fc(s)."""
    h = fcw @ s + fcb                        # [2H]
    gamma = h[:H][:, None]                   # [H,1]
    beta = h[H:][:, None]                    # [H,1]
    return gamma * instance_norm_ch(x, LN_EPS) + beta


def relu(x):
    return np.maximum(x, 0.0)


def forward(ids, style):
    s_pred = style[:H]
    s_dec = style[H:]

    # ---- text encoder: embed -> conv(k) -> ReLU. channel-major [H][T].
    emb = P["embed"][ids]                    # [T, H]
    hid = emb.T.copy()                       # [H, T]
    hid = relu(conv1d(hid, P["enc_w"], P["enc_b"], KENC))

    # ---- duration predictor: AdaIN(s_pred) -> conv ReLU -> proj scalar.
    d = adain(hid, s_pred, P["dur_adain_w"], P["dur_adain_b"])
    d = relu(conv1d(d, P["dur_w"], P["dur_b"], KDUR))
    log_dur = conv1d(d, P["dur_proj_w"], P["dur_proj_b"], 1)[0]   # [T]
    # duration = round(exp(log_dur) * speed), clamped >= 1. log_dur is scaled
    # at the proj weights so durations land in a small, varied integer range
    # (the fixture pins them; the parity test checks them exactly).
    durations = np.maximum(np.round(np.exp(log_dur) * SPEED), 1).astype(int)

    # ---- length regulator: expand hidden along time by per-token durations.
    cols = []
    for t in range(len(ids)):
        for _ in range(int(durations[t])):
            cols.append(hid[:, t])
    expanded = np.stack(cols, axis=1)        # [H, L]

    # ---- F0 predictor: AdaIN(s_pred) -> conv ReLU -> proj.
    f = adain(expanded, s_pred, P["f0_adain_w"], P["f0_adain_b"])
    f = relu(conv1d(f, P["f0_w"], P["f0_b"], KF0))
    f0 = conv1d(f, P["f0_proj_w"], P["f0_proj_b"], 1)[0]          # [L]

    # ---- energy (N) predictor: AdaIN(s_pred) -> conv ReLU -> proj.
    n = adain(expanded, s_pred, P["n_adain_w"], P["n_adain_b"])
    n = relu(conv1d(n, P["n_w"], P["n_b"], KF0))
    energy = conv1d(n, P["n_proj_w"], P["n_proj_b"], 1)[0]        # [L]

    # ---- iSTFTNet decoder: [hidden; f0; energy] -> AdaIN(s_dec) on the H-ch
    # mix -> magnitude + phase heads -> ISTFT.
    dec_in = np.concatenate([expanded, f0[None, :], energy[None, :]], axis=0)
    mix = relu(conv1d(dec_in, P["dec_in_w"], P["dec_in_b"], KDEC))   # [H, L]
    mix = adain(mix, s_dec, P["dec_adain_w"], P["dec_adain_b"])
    mag = np.exp(conv1d(mix, P["mag_w"], P["mag_b"], KDEC))          # [NBINS, L] > 0
    phase = np.sin(conv1d(mix, P["phase_w"], P["phase_b"], KDEC))    # [NBINS, L] (bounded)

    wave = istft(mag, phase, NFFT, HOP)
    return dict(hid=hid, log_dur=log_dur, durations=durations,
                f0=f0, energy=energy, mag=mag, phase=phase, wave=wave)


def istft(mag, phase, nfft, hop):
    """Overlap-add inverse STFT, EXACTLY matching neuralaudio.pas
    ISTFTOverlapAddReIm (periodic-hann synthesis window, COLA/window_sumsquare
    normalization, the file's cos/sin twiddle convention). mag/phase are
    [NBINS][frames]; Re=mag*cos(phase), Im=mag*sin(phase)."""
    nbins, nframes = mag.shape
    assert nbins == nfft // 2 + 1
    re = mag * np.cos(phase)
    im = mag * np.sin(phase)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(nfft) / nfft)
    out_len = nfft + (nframes - 1) * hop
    acc = np.zeros(out_len)
    env = np.zeros(out_len)
    # twiddles
    tt = np.arange(nfft)
    for fr in range(nframes):
        start = fr * hop
        samp = np.zeros(nfft)
        for b in range(nbins):
            scale = 1.0 if (b == 0 or (b == nbins - 1 and nfft % 2 == 0)) else 2.0
            cos = np.cos(2.0 * np.pi * b * tt / nfft)
            sin = np.sin(2.0 * np.pi * b * tt / nfft)
            samp += scale * (re[b, fr] * cos + im[b, fr] * sin)
        samp /= nfft
        acc[start:start + nfft] += samp * window
        env[start:start + nfft] += window * window
    wave = np.zeros(out_len)
    nz = env > 1e-12
    wave[nz] = acc[nz] / env[nz]
    return wave


# --------------------------------------------------------------------------
# Run the oracle on a pinned input.
# --------------------------------------------------------------------------
ids = np.array([3, 7, 1, 12, 5, 9], dtype=np.int64)
style = (rng.standard_normal(STYLE_DIM) * 0.8)

out = forward(ids, style)

# Assert non-vacuous: durations vary, waveform is non-trivial.
assert out["durations"].min() >= 1
assert len(set(out["durations"].tolist())) > 1, "durations must vary"
assert np.abs(out["wave"]).max() > 1e-3, "waveform must be non-trivial"

# --------------------------------------------------------------------------
# Write fixtures.
# --------------------------------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
fixdir = os.path.join(here, "..", "tests", "fixtures")
fixdir = os.path.abspath(fixdir)

config = dict(
    model_type="kokoro",
    hidden_size=H,
    style_dim=STYLE_DIM,
    vocab_size=VOCAB,
    enc_kernel=KENC,
    dur_kernel=KDUR,
    f0_kernel=KF0,
    dec_kernel=KDEC,
    n_fft=NFFT,
    hop_length=HOP,
    speed=SPEED,
    layer_norm_eps=LN_EPS,
    sampling_rate=SAMPLE_RATE,
)
with open(os.path.join(fixdir, "tiny_kokoro_config.json"), "w") as f:
    json.dump(config, f, indent=2)

ref = dict(
    hidden_size=H,
    style_dim=STYLE_DIM,
    n_fft=NFFT,
    hop_length=HOP,
    out_len=int(out["f0"].shape[0]),
    n_bins=NBINS,
    input_ids=ids.tolist(),
    style=style.tolist(),
    log_duration=out["log_dur"].tolist(),
    durations=out["durations"].tolist(),
    f0=out["f0"].tolist(),
    energy=out["energy"].tolist(),
    magnitude=out["mag"].tolist(),       # [NBINS][L]
    phase=out["phase"].tolist(),         # [NBINS][L]
    waveform=out["wave"].tolist(),
)
with open(os.path.join(fixdir, "tiny_kokoro_ref.json"), "w") as f:
    json.dump(ref, f)

# Save weights as a safetensors file (float32 storage, matching the importer).
tensors = {
    "embed": P["embed"],
    "text_encoder.conv.weight": P["enc_w"], "text_encoder.conv.bias": P["enc_b"],
    "duration_predictor.adain.fc.weight": P["dur_adain_w"],
    "duration_predictor.adain.fc.bias": P["dur_adain_b"],
    "duration_predictor.conv.weight": P["dur_w"], "duration_predictor.conv.bias": P["dur_b"],
    "duration_predictor.proj.weight": P["dur_proj_w"], "duration_predictor.proj.bias": P["dur_proj_b"],
    "f0_predictor.adain.fc.weight": P["f0_adain_w"], "f0_predictor.adain.fc.bias": P["f0_adain_b"],
    "f0_predictor.conv.weight": P["f0_w"], "f0_predictor.conv.bias": P["f0_b"],
    "f0_predictor.proj.weight": P["f0_proj_w"], "f0_predictor.proj.bias": P["f0_proj_b"],
    "energy_predictor.adain.fc.weight": P["n_adain_w"], "energy_predictor.adain.fc.bias": P["n_adain_b"],
    "energy_predictor.conv.weight": P["n_w"], "energy_predictor.conv.bias": P["n_b"],
    "energy_predictor.proj.weight": P["n_proj_w"], "energy_predictor.proj.bias": P["n_proj_b"],
    "decoder.adain.fc.weight": P["dec_adain_w"], "decoder.adain.fc.bias": P["dec_adain_b"],
    "decoder.conv_in.weight": P["dec_in_w"], "decoder.conv_in.bias": P["dec_in_b"],
    "decoder.magnitude.weight": P["mag_w"], "decoder.magnitude.bias": P["mag_b"],
    "decoder.phase.weight": P["phase_w"], "decoder.phase.bias": P["phase_b"],
}

try:
    from safetensors.numpy import save_file
    save_file({k: np.ascontiguousarray(v.astype(np.float32)) for k, v in tensors.items()},
              os.path.join(fixdir, "tiny_kokoro.safetensors"))
except ImportError:
    import torch
    from safetensors.torch import save_file
    save_file({k: torch.tensor(v.astype(np.float32)) for k, v in tensors.items()},
              os.path.join(fixdir, "tiny_kokoro.safetensors"))

print("wrote tiny_kokoro fixtures to", fixdir)
print("  ids        =", ids.tolist())
print("  durations  =", out["durations"].tolist(), "-> out_len", ref["out_len"])
print("  wave len   =", len(out["wave"]), " max|wave| =", float(np.abs(out["wave"]).max()))
