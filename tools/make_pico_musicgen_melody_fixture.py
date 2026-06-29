#!/usr/bin/env python3
"""Generate a tiny RANDOM MusicGen MELODY parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a PICO config, never downloaded).

MusicGen Melody (facebook/musicgen-melody, model_type "musicgen_melody")
extends text-conditioned MusicGen with MELODY conditioning: a 12-bin
CHROMAGRAM of a reference waveform steers generation, optionally alongside a
T5 text prompt. The genuinely NEW pieces over the landed text-MusicGen path:

  (a) a CHROMA front-end (transformers MusicgenMelodyFeatureExtractor): a
      power spectrogram (n_fft=16384, hop=4096, hann periodic window,
      center=True / reflect pad, normalized by the window L2 energy, power=2)
      projected through librosa's chroma_filter_bank onto 12 pitch classes,
      per-frame inf-norm normalized, then ARGMAX one-hot (the dominant chroma
      gets 1, the rest 0).

  (b) the decoder is DECODER-ONLY (causal self-attention, NO cross-attention,
      unlike text-MusicGen's Marian/Pegasus cross-attention blocks): the
      conditioning is PREPENDED to the decoder sequence. The chroma is
      projected by audio_enc_to_dec_proj (num_chroma -> hidden) and the text
      states by enc_to_dec_proj (text_d_model -> hidden); the prefix is
      cat([chroma_proj, text_proj]) on the sequence axis (CHROMA FIRST), then
      the per-frame code embeddings + sinusoidal positions follow. Logits are
      read only at the decoder-frame positions.

This fixture pins (1) the chroma extractor for a fixed short synthesized
waveform and (2) ONE decoder forward step given chroma+text conditioning.
The T5 encoder and EnCodec decoder are covered by their own importers/tests
and reuse the landed text-MusicGen fixtures.

Fixtures, KB-scale, pinned in tests/fixtures/:

  tiny_musicgen_melody.safetensors: the MusicgenMelodyForConditionalGeneration
      decoder + enc_to_dec_proj + audio_enc_to_dec_proj at pico width (text
      encoder + EnCodec audio encoder tensors dropped).

  tiny_musicgen_melody_config.json: pico decoder sub-config + num_chroma /
      chroma_length + text_encoder d_model.

  tiny_musicgen_melody_ref.json: the float64 oracle:
      - "wave": the fixed reference waveform fed to the chroma extractor;
      - "chroma": [num_frames][num_chroma] one-hot chroma oracle (< 1e-4 gate);
      - "enc_states": fixed T5 text hidden states [enc_seq_len][text_d_model];
      - "chroma_cond": [chroma_len][num_chroma] one-hot chroma conditioning
        sequence (already repeat-tiled/truncated to chroma_length) fed to the
        decoder step;
      - "dec_codes": fixed decoder code stack [num_codebooks][dec_seq_len];
      - "logits": [num_codebooks][dec_seq_len][vocab_size] decoder logits at the
        decoder-frame positions (< 1e-4 gate).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_musicgen_melody_fixture.py
Needs torch + transformers + safetensors + numpy (NO torchaudio: the
spectrogram is replicated bit-for-bit via torch.stft + window-energy norm,
exactly what torchaudio.transforms.Spectrogram(normalized=True) computes).
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import EncodecConfig, T5Config
from transformers.audio_utils import chroma_filter_bank
from transformers.models.musicgen_melody.configuration_musicgen_melody import (
    MusicgenMelodyConfig, MusicgenMelodyDecoderConfig)
from transformers.models.musicgen_melody.modeling_musicgen_melody import (
    MusicgenMelodyForCausalLM, MusicgenMelodyForConditionalGeneration)

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

SEED = 4242

TEXT_DMODEL = 12   # T5 d_model (!= dec hidden, so enc_to_dec_proj is exercised)
DEC_HIDDEN = 8
VOCAB = 16         # codebook size (decoder vocab_size)
NUM_CODEBOOKS = 4
DEC_LAYERS = 2
DEC_HEADS = 2
FFN = 16
ENC_SEQ = 5        # T5 text sequence length
DEC_SEQ = 7        # decoder frame count for the parity step
NUM_CHROMA = 12
CHROMA_LENGTH = 6  # config.chroma_length (conditioning chroma frames)

# Chroma front-end params (the REAL musicgen_melody feature extractor defaults).
SAMPLING_RATE = 32000
N_FFT = 16384
HOP = 4096


def stft_spectrogram(wave):
    """Replicate torchaudio.transforms.Spectrogram(n_fft, win_length=n_fft,
    hop_length, power=2, center=True, normalized=True): torch.stft with the
    hann periodic window then divide by the window L2 energy, then |.|^2.
    Matches torchaudio.functional.spectrogram bit-for-bit."""
    w = torch.hann_window(N_FFT, periodic=True, dtype=wave.dtype)
    S = torch.stft(wave, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                   window=w, center=True, pad_mode="reflect", normalized=False,
                   onesided=True, return_complex=True)
    S = S / w.pow(2.0).sum().sqrt()
    return S.abs().pow(2.0)


def extract_chroma(wave, dtype=torch.float64):
    """Bit-for-bit replica of MusicgenMelodyFeatureExtractor.
    _torch_extract_fbank_features for a single (n_samples,) waveform. Returns
    (num_frames, num_chroma) one-hot float tensor."""
    wave = wave.to(dtype)
    cf = torch.from_numpy(
        chroma_filter_bank(sampling_rate=SAMPLING_RATE, num_frequency_bins=N_FFT,
                           tuning=0, num_chroma=NUM_CHROMA)).to(dtype)
    w = wave.unsqueeze(0)  # (1, n_samples)
    # pad to n_fft if too short (feature extractor branch)
    wav_length = w.shape[-1]
    if wav_length < N_FFT:
        pad = N_FFT - wav_length
        rest = 0 if pad % 2 == 0 else 1
        w = torch.nn.functional.pad(w, (pad // 2, pad // 2 + rest), "constant", 0)
    spec = stft_spectrogram(w)          # (1, nbins, ntime)
    raw_chroma = torch.einsum("cf, ...ft->...ct", cf, spec)  # (1, nchroma, ntime)
    norm_chroma = torch.nn.functional.normalize(
        raw_chroma, p=float("inf"), dim=-2, eps=1e-6)
    norm_chroma = norm_chroma.transpose(1, 2)  # (1, ntime, nchroma)
    idx = norm_chroma.argmax(-1, keepdim=True)
    norm_chroma[:] = 0
    norm_chroma.scatter_(dim=-1, index=idx, value=1)
    return norm_chroma.squeeze(0)       # (ntime, nchroma)


def build_config():
    t5 = T5Config(vocab_size=40, d_model=TEXT_DMODEL, d_ff=16, num_layers=2,
                  num_heads=2, d_kv=6, relative_attention_num_buckets=8,
                  relative_attention_max_distance=16)
    enc = EncodecConfig(sampling_rate=8000, audio_channels=1, normalize=False,
                        hidden_size=8, num_filters=2, num_residual_layers=1,
                        upsampling_ratios=[2, 2], kernel_size=7,
                        last_kernel_size=7, residual_kernel_size=3,
                        dilation_growth_rate=2, compress=2, num_lstm_layers=1,
                        codebook_size=VOCAB, codebook_dim=8,
                        target_bandwidths=[48.0])
    dec = MusicgenMelodyDecoderConfig(
        vocab_size=VOCAB, max_position_embeddings=128,
        num_hidden_layers=DEC_LAYERS, ffn_dim=FFN,
        num_attention_heads=DEC_HEADS, hidden_size=DEC_HIDDEN,
        num_codebooks=NUM_CODEBOOKS, audio_channels=1,
        scale_embedding=False, activation_function="gelu")
    return MusicgenMelodyConfig(text_encoder=t5.to_dict(),
                                audio_encoder=enc.to_dict(), decoder=dec.to_dict(),
                                num_chroma=NUM_CHROMA, chroma_length=CHROMA_LENGTH)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = build_config()
    model = MusicgenMelodyForConditionalGeneration(cfg).eval()

    # Amplify the conditioning projections to O(1) so the chroma/text prefix
    # genuinely steers generation (HF std-0.02 init makes the prefix vacuous);
    # same "re-randomize pico weights" trick as the text-MusicGen fixture. The
    # oracle below is recomputed from the SAME saved state dict, so importer
    # parity is preserved exactly.
    with torch.no_grad():
        model.enc_to_dec_proj.weight.mul_(8.0)
        model.audio_enc_to_dec_proj.weight.mul_(8.0)

    model_f64 = MusicgenMelodyForConditionalGeneration(cfg).double().eval()
    model_f64.load_state_dict(model.state_dict())

    rng = np.random.RandomState(SEED)

    # ---- fixed reference waveform -> chroma oracle ----
    # A short deterministic mixture of pure tones (> n_fft so no padding branch
    # is exercised). The waveform is described by a small RECIPE (freqs / amps /
    # n_samples) rather than dumped sample-by-sample, so the committed fixture
    # stays KB-scale; the Pascal test re-synthesizes the identical waveform from
    # the recipe (wave[i] = sum_j amp_j * sin(2*pi*freq_j*i/sr)).
    n_samples = N_FFT + HOP       # 2 STFT frames
    wave_freqs = [440.0, 660.0]
    wave_amps = [0.6, 0.3]
    tt = np.arange(n_samples, dtype=np.float64) / SAMPLING_RATE
    wave_np = np.zeros(n_samples, dtype=np.float64)
    for fr, am in zip(wave_freqs, wave_amps):
        wave_np += am * np.sin(2 * np.pi * fr * tt)
    wave = torch.tensor(wave_np, dtype=torch.float64)
    chroma = extract_chroma(wave, dtype=torch.float64)    # (nframes, 12)
    num_frames = chroma.shape[0]

    # ---- chroma conditioning sequence fed to the decoder step ----
    # The model repeat-tiles chroma to chroma_length then truncates. Use a
    # DISTINCT short chroma so the conditioning path is exercised cleanly: a
    # one-hot sequence of length 2, tiled/truncated to chroma_length.
    cond_short = torch.zeros((2, NUM_CHROMA), dtype=torch.float64)
    cond_short[0, 3] = 1.0
    cond_short[1, 7] = 1.0
    n_repeat = int(np.ceil(CHROMA_LENGTH / cond_short.shape[0]))
    chroma_cond = cond_short.repeat(n_repeat, 1)[:CHROMA_LENGTH]  # (chroma_len,12)

    # ---- fixed T5 text hidden states ----
    enc_states = rng.randn(ENC_SEQ, TEXT_DMODEL) * 0.5
    enc_states = np.round(enc_states * 64.0) / 64.0

    # ---- fixed decoder code stack [K, T] ----
    dec_codes = rng.randint(0, VOCAB, size=(NUM_CODEBOOKS, DEC_SEQ))

    # ---- one decoder forward step oracle ----
    # Build the conditioning prefix exactly as the model does:
    #   text_proj  = enc_to_dec_proj(text_states)          (ENC_SEQ, hidden)
    #   chroma_proj= audio_enc_to_dec_proj(chroma_cond)    (chroma_len, hidden)
    #   prefix     = cat([chroma_proj, text_proj], dim=seq)  (CHROMA FIRST)
    # then feed it as encoder_hidden_states to the causal-LM decoder, which
    # prepends it to the per-codebook frame embeddings + sinusoidal positions.
    with torch.no_grad():
        text_t = torch.tensor(enc_states, dtype=torch.float64).unsqueeze(0)
        text_proj = model_f64.enc_to_dec_proj(text_t)              # (1,ENC_SEQ,H)
        chroma_t = chroma_cond.unsqueeze(0)                        # (1,chroma_len,12)
        chroma_proj = model_f64.audio_enc_to_dec_proj(chroma_t)    # (1,chroma_len,H)
        prefix = torch.cat([chroma_proj, text_proj], dim=1)        # chroma first

        decoder: MusicgenMelodyForCausalLM = model_f64.decoder
        ids = torch.tensor(dec_codes, dtype=torch.long).reshape(
            NUM_CODEBOOKS, DEC_SEQ)
        out = decoder(input_ids=ids, encoder_hidden_states=prefix, use_cache=False)
        # logits: (bsz*K, full_seq, vocab); take the LAST DEC_SEQ frame positions.
        logits = out.logits.reshape(NUM_CODEBOOKS, -1, VOCAB)[:, -DEC_SEQ:, :]

    os.makedirs(FIX, exist_ok=True)

    # ---- save safetensors: decoder + the two projection layers ----
    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("audio_encoder") or k.startswith("text_encoder"):
            continue
        sd[k] = v.to(torch.float32).contiguous()
    save_file(sd, os.path.join(FIX, "tiny_musicgen_melody.safetensors"))

    out_cfg = {
        "model_type": "musicgen_melody",
        "text_d_model": TEXT_DMODEL,
        "num_chroma": NUM_CHROMA,
        "chroma_length": CHROMA_LENGTH,
        "decoder": {
            "vocab_size": VOCAB,
            "hidden_size": DEC_HIDDEN,
            "num_hidden_layers": DEC_LAYERS,
            "num_attention_heads": DEC_HEADS,
            "ffn_dim": FFN,
            "num_codebooks": NUM_CODEBOOKS,
            "max_position_embeddings": 128,
            "activation_function": "gelu",
            "scale_embedding": False,
            "audio_channels": 1,
        },
    }
    with open(os.path.join(FIX, "tiny_musicgen_melody_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=1)

    ref = {
        "text_d_model": TEXT_DMODEL,
        "dec_hidden": DEC_HIDDEN,
        "vocab_size": VOCAB,
        "num_codebooks": NUM_CODEBOOKS,
        "num_chroma": NUM_CHROMA,
        "chroma_length": CHROMA_LENGTH,
        "enc_seq_len": ENC_SEQ,
        "dec_seq_len": DEC_SEQ,
        "sampling_rate": SAMPLING_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP,
        "num_frames": int(num_frames),
        "wave_n_samples": int(n_samples),
        "wave_freqs": wave_freqs,
        "wave_amps": wave_amps,
        "chroma": chroma.to(torch.float64).tolist(),
        "enc_states": enc_states.tolist(),
        "chroma_cond": chroma_cond.to(torch.float64).tolist(),
        "dec_codes": dec_codes.astype(np.int64).tolist(),
        "logits": logits.to(torch.float64).tolist(),
    }
    with open(os.path.join(FIX, "tiny_musicgen_melody_ref.json"), "w") as f:
        json.dump(ref, f)

    st = os.path.getsize(os.path.join(FIX, "tiny_musicgen_melody.safetensors"))
    print("wrote tiny_musicgen_melody.safetensors %d bytes (%d tensors)"
          % (st, len(sd)))
    print("chroma shape", list(chroma.shape), "frames", num_frames)
    print("chroma argmax per frame:", chroma.argmax(-1).tolist())
    print("logits shape", list(logits.shape))
    refsz = os.path.getsize(os.path.join(FIX, "tiny_musicgen_melody_ref.json"))
    print("ref json %d bytes" % refsz)


if __name__ == "__main__":
    main()
