#!/usr/bin/env python3
"""Generate a tiny RANDOM MusicGen parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a PICO config, never downloaded).

MusicGen (facebook/musicgen-* family, model_type "musicgen") is a
text-to-music model: a T5 text ENCODER conditions a single-stage transformer
DECODER that autoregressively predicts the EnCodec code stack with the
DELAY-PATTERN codebook interleaving (each of the K codebooks offset by one
step, so a single set of K LM heads predicts them causally), decoded back to
audio through the EnCodec decoder.

The genuinely NEW code over the landed seq2seq enc-dec convention is the
delay-pattern (de)interleaving; the decoder itself is a BART/Marian-style
POST-norm cross-attention stack with BIAS-FREE linears, a SINUSOIDAL position
table (HF cat([cos, sin]) half-split, base log(10000)/(half-1), NO offset), K
embedding tables summed at the input, a final decoder LayerNorm, and K LM
heads. This fixture pins the DECODER forward pass (the new wiring) and the
delay-pattern round trip; the T5 encoder and EnCodec decoder are covered by
their own importers/tests.

Fixtures, KB-scale, pinned in tests/fixtures/:

  tiny_musicgen.safetensors: a MusicgenForConditionalGeneration decoder +
      enc_to_dec_proj at pico width (the T5 encoder + EnCodec audio encoder
      tensors are dropped - the Pascal parity test feeds a FIXED encoder
      hidden-state tensor and pins decoder logits, so only the decoder side
      is needed).

  tiny_musicgen_config.json: the pico decoder sub-config plus the dims the
      importer needs (text_encoder d_model for enc_to_dec_proj).

  tiny_musicgen_ref.json: the float64 oracle:
      - "enc_states": the fixed encoder hidden states fed to cross-attention
        [enc_seq_len][text_d_model];
      - "dec_codes": the fixed decoder code stack [num_codebooks][dec_seq_len]
        (already delay-shifted ids in [0, vocab_size]; vocab_size is the pad);
      - "logits": [num_codebooks][dec_seq_len][vocab_size] decoder logits, the
        array the Pascal test gates < 1e-4;
      - "delay": a small delay-pattern round-trip oracle: a raw code stack and
        the HF delayed/undelayed forms, to gate the (de)interleave helpers.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/musicgen_tiny_fixture.py
writes tests/fixtures/tiny_musicgen{.safetensors,_config.json,_ref.json}.
Needs torch + transformers + safetensors + numpy.
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import (EncodecConfig, MusicgenConfig,
                          MusicgenForConditionalGeneration, T5Config)
from transformers.models.musicgen.configuration_musicgen import (
    MusicgenDecoderConfig)
from transformers.models.musicgen.modeling_musicgen import MusicgenForCausalLM

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
ENC_SEQ = 5
DEC_SEQ = 7


def build_config():
    t5 = T5Config(vocab_size=40, d_model=TEXT_DMODEL, d_ff=16, num_layers=2,
                  num_heads=2, d_kv=6, relative_attention_num_buckets=8,
                  relative_attention_max_distance=16)
    enc = EncodecConfig(sampling_rate=8000, audio_channels=1, normalize=False,
                        hidden_size=8, num_filters=2, num_residual_layers=1,
                        upsampling_ratios=[2, 2], kernel_size=7,
                        last_kernel_size=7, residual_kernel_size=3,
                        dilation_growth_rate=2, compress=2, num_lstm_layers=1,
                        codebook_size=VOCAB, codebook_dim=8)
    dec = MusicgenDecoderConfig(
        vocab_size=VOCAB, max_position_embeddings=64,
        num_hidden_layers=DEC_LAYERS, ffn_dim=FFN,
        num_attention_heads=DEC_HEADS, hidden_size=DEC_HIDDEN,
        num_codebooks=NUM_CODEBOOKS, audio_channels=1,
        scale_embedding=False, activation_function="gelu")
    return MusicgenConfig(text_encoder=t5.to_dict(),
                          audio_encoder=enc.to_dict(), decoder=dec.to_dict())


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = build_config()
    model = MusicgenForConditionalGeneration(cfg).eval()

    # float64 oracle
    model_f64 = MusicgenForConditionalGeneration(cfg).double().eval()
    model_f64.load_state_dict(model.state_dict())

    rng = np.random.RandomState(SEED)

    # Fixed encoder hidden states (what the T5 encoder WOULD emit) in TEXT
    # d_model; the decoder cross-attention projects them via enc_to_dec_proj.
    enc_states = (rng.randn(ENC_SEQ, TEXT_DMODEL) * 0.5)
    enc_states = np.round(enc_states * 64.0) / 64.0

    # Fixed decoder code stack [K, T] of ids in [0, VOCAB-1]. These are the
    # (already delay-shifted) input ids the decoder embeds per codebook.
    dec_codes = rng.randint(0, VOCAB, size=(NUM_CODEBOOKS, DEC_SEQ))

    # Project enc_states through enc_to_dec_proj inside the model by calling
    # the decoder sub-model directly with encoder_hidden_states already in the
    # DECODER hidden dim. The decoder's cross-attention k/v project from
    # encoder_hidden_states, which in MusicgenForConditionalGeneration are the
    # enc_to_dec_proj(text_encoder_output). We replicate that projection here
    # so the Pascal importer (which also applies enc_to_dec_proj) matches.
    with torch.no_grad():
        proj = model_f64.enc_to_dec_proj
        enc_t = torch.tensor(enc_states, dtype=torch.float64).unsqueeze(0)
        enc_hidden = proj(enc_t)  # [1, ENC_SEQ, DEC_HIDDEN]

        decoder: MusicgenForCausalLM = model_f64.decoder
        # input_ids shape (bsz * num_codebooks, seq_len)
        ids = torch.tensor(dec_codes, dtype=torch.long).reshape(
            1 * NUM_CODEBOOKS, DEC_SEQ)
        out = decoder(
            input_ids=ids,
            encoder_hidden_states=enc_hidden,
            use_cache=False,
        )
        logits = out.logits  # [bsz, num_codebooks, seq_len, vocab]
        logits = logits.reshape(NUM_CODEBOOKS, DEC_SEQ, VOCAB)

    # ---- delay-pattern round-trip oracle (HF build/apply) ----
    # Build a delayed mask from a fresh code stack and record the delayed form
    # so the Pascal interleave helper can match it exactly.
    raw = rng.randint(0, VOCAB, size=(NUM_CODEBOOKS, DEC_SEQ))
    pad_id = VOCAB  # bos/pad token id used by the delay pattern
    with torch.no_grad():
        gen_decoder = model.decoder
        raw_ids = torch.tensor(raw, dtype=torch.long).reshape(
            NUM_CODEBOOKS, DEC_SEQ)
        # build_delay_pattern_mask expects (bsz*K, seq_len) and a max_length
        max_len = DEC_SEQ + NUM_CODEBOOKS
        delayed_ids, mask = gen_decoder.build_delay_pattern_mask(
            raw_ids, pad_token_id=pad_id, max_length=max_len)
        delayed_ids = delayed_ids.reshape(NUM_CODEBOOKS, -1)
        mask = mask.reshape(NUM_CODEBOOKS, -1)

    os.makedirs(FIX, exist_ok=True)

    # ---- save safetensors: decoder + enc_to_dec_proj only ----
    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("audio_encoder") or k.startswith("text_encoder"):
            continue
        sd[k] = v.to(torch.float32).contiguous()
    save_file(sd, os.path.join(FIX, "tiny_musicgen.safetensors"))

    out_cfg = {
        "model_type": "musicgen",
        "text_d_model": TEXT_DMODEL,
        "decoder": {
            "vocab_size": VOCAB,
            "hidden_size": DEC_HIDDEN,
            "num_hidden_layers": DEC_LAYERS,
            "num_attention_heads": DEC_HEADS,
            "ffn_dim": FFN,
            "num_codebooks": NUM_CODEBOOKS,
            "max_position_embeddings": 64,
            "activation_function": "gelu",
            "scale_embedding": False,
            "audio_channels": 1,
        },
    }
    with open(os.path.join(FIX, "tiny_musicgen_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=1)

    ref = {
        "text_d_model": TEXT_DMODEL,
        "dec_hidden": DEC_HIDDEN,
        "vocab_size": VOCAB,
        "num_codebooks": NUM_CODEBOOKS,
        "enc_seq_len": ENC_SEQ,
        "dec_seq_len": DEC_SEQ,
        "enc_states": enc_states.tolist(),
        "dec_codes": dec_codes.astype(np.int64).tolist(),
        "logits": logits.to(torch.float64).tolist(),
        "delay": {
            "pad_id": pad_id,
            "max_length": int(max_len),
            "raw": raw.astype(np.int64).tolist(),
            "delayed": delayed_ids.to(torch.int64).tolist(),
            "mask": mask.to(torch.int64).tolist(),
        },
    }
    with open(os.path.join(FIX, "tiny_musicgen_ref.json"), "w") as f:
        json.dump(ref, f)

    st = os.path.getsize(os.path.join(FIX, "tiny_musicgen.safetensors"))
    print("wrote tiny_musicgen.safetensors %d bytes" % st)
    print("logits shape", list(logits.shape),
          "delayed shape", list(delayed_ids.shape))


if __name__ == "__main__":
    main()
