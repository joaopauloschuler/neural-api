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
                          MusicgenForConditionalGeneration, T5Config,
                          T5ForConditionalGeneration)
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


def build_config(audio_channels=1, num_codebooks=NUM_CODEBOOKS):
    t5 = T5Config(vocab_size=40, d_model=TEXT_DMODEL, d_ff=16, num_layers=2,
                  num_heads=2, d_kv=6, relative_attention_num_buckets=8,
                  relative_attention_max_distance=16)
    # EnCodec codebook_size MUST equal the MusicGen decoder VOCAB so the
    # generated code ids are valid RVQ indices for the codec decoder. The
    # target_bandwidths list controls how many RVQ stages decode; the lowest
    # band already yields >= NUM_CODEBOOKS quantizers at this frame rate.
    enc = EncodecConfig(sampling_rate=8000, audio_channels=1, normalize=False,
                        hidden_size=8, num_filters=2, num_residual_layers=1,
                        upsampling_ratios=[2, 2], kernel_size=7,
                        last_kernel_size=7, residual_kernel_size=3,
                        dilation_growth_rate=2, compress=2, num_lstm_layers=1,
                        codebook_size=VOCAB, codebook_dim=8,
                        target_bandwidths=[48.0])
    # NOTE: stereo MusicGen keeps the EnCodec audio_encoder MONO
    # (audio_channels=1) - HF decodes each channel's codebooks through the same
    # mono codec and concatenates. Only the DECODER carries audio_channels=2 and
    # the doubled (2*K) interleaved-codebook layout.
    dec = MusicgenDecoderConfig(
        vocab_size=VOCAB, max_position_embeddings=64,
        num_hidden_layers=DEC_LAYERS, ffn_dim=FFN,
        num_attention_heads=DEC_HEADS, hidden_size=DEC_HIDDEN,
        num_codebooks=num_codebooks, audio_channels=audio_channels,
        scale_embedding=False, activation_function="gelu")
    return MusicgenConfig(text_encoder=t5.to_dict(),
                          audio_encoder=enc.to_dict(), decoder=dec.to_dict())


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = build_config()
    model = MusicgenForConditionalGeneration(cfg).eval()

    # The default HF init (std 0.02 / Xavier) makes the cross-attention
    # conditioning path so weak that, over a 16-way greedy argmax, the encoder
    # states barely move the predicted code ids - the generation looks
    # text-INDEPENDENT. Amplify the conditioning path (enc_to_dec_proj and the
    # decoder cross-attention k/v/q/out projections) to O(1) so the prompt
    # genuinely steers generation; this is the same "re-randomize pico weights
    # to a useful scale" trick used for the ModernBERT fixtures. The decoder
    # logit-parity oracle below is recomputed AFTER this scaling (and from the
    # SAME state dict that is saved), so importer parity is preserved exactly.
    with torch.no_grad():
        model.enc_to_dec_proj.weight.mul_(12.0)
        for layer in model.decoder.model.decoder.layers:
            ca = layer.encoder_attn
            for proj in (ca.q_proj, ca.k_proj, ca.v_proj, ca.out_proj):
                proj.weight.mul_(4.0)

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

    # ---- matched T5 text fixture (for examples/MusicGenText) ----
    # The end-to-end text-conditioned example wires the REAL T5 encoder, whose
    # d_model MUST equal TEXT_DMODEL so its hidden states feed enc_to_dec_proj.
    # MusicGen ships only a T5 ENCODER, but the standalone T5 importer
    # (BuildT5FromSafeTensors) builds the full encoder+decoder pair and needs
    # the decoder tower's weights too. So emit a FULL standard
    # T5ForConditionalGeneration checkpoint at the matched config (a drop-in
    # T5 fixture); the example only runs its ENCODER. The encoder hidden states
    # are identical to MusicGen's text encoder up to the random init - parity
    # between the two T5 instances is NOT required (the example/test only need a
    # deterministic, shape-matched encoder), so a fresh standard T5 is correct.
    t5_full = T5ForConditionalGeneration(
        T5Config(**cfg.text_encoder.to_dict())).eval()
    t5_sd = {k: v.to(torch.float32).contiguous()
             for k, v in t5_full.state_dict().items()}
    # HF T5 ties shared/embed_tokens/lm_head; drop the aliases the importer
    # reconstructs from shared.weight (matches real-checkpoint _tied keys).
    t5_sd.pop("encoder.embed_tokens.weight", None)
    t5_sd.pop("decoder.embed_tokens.weight", None)
    if cfg.text_encoder.tie_word_embeddings:
        t5_sd.pop("lm_head.weight", None)
    save_file(t5_sd, os.path.join(FIX, "tiny_musicgen_t5enc.safetensors"))
    t5cfg = cfg.text_encoder
    t5_out_cfg = {
        "model_type": "t5",
        "d_model": t5cfg.d_model,
        "d_kv": t5cfg.d_kv,
        "d_ff": t5cfg.d_ff,
        "num_layers": t5cfg.num_layers,
        "num_decoder_layers": t5cfg.num_decoder_layers,
        "num_heads": t5cfg.num_heads,
        "vocab_size": t5cfg.vocab_size,
        "relative_attention_num_buckets": t5cfg.relative_attention_num_buckets,
        "relative_attention_max_distance":
            t5cfg.relative_attention_max_distance,
        "layer_norm_epsilon": t5cfg.layer_norm_epsilon,
        "feed_forward_proj": t5cfg.feed_forward_proj,
        "tie_word_embeddings": t5cfg.tie_word_embeddings,
    }
    with open(os.path.join(FIX, "tiny_musicgen_t5enc_config.json"), "w") as f:
        json.dump(t5_out_cfg, f, indent=1)

    # ---- matched EnCodec DECODER fixture (for examples/MusicGenText) ----
    # codebook_size == VOCAB so MusicGen-generated ids are valid RVQ indices.
    ec_sd = {}
    for k, v in model.state_dict().items():
        if not k.startswith("audio_encoder."):
            continue
        nk = k[len("audio_encoder."):]
        ec_sd[nk] = v.to(torch.float32).contiguous()
    save_file(ec_sd, os.path.join(FIX, "tiny_musicgen_encodec.safetensors"))
    eccfg = cfg.audio_encoder
    ec_out_cfg = {
        "model_type": "encodec",
        "sampling_rate": eccfg.sampling_rate,
        "audio_channels": eccfg.audio_channels,
        "normalize": eccfg.normalize,
        "hidden_size": eccfg.hidden_size,
        "num_filters": eccfg.num_filters,
        "num_residual_layers": eccfg.num_residual_layers,
        "upsampling_ratios": eccfg.upsampling_ratios,
        "norm_type": eccfg.norm_type,
        "kernel_size": eccfg.kernel_size,
        "last_kernel_size": eccfg.last_kernel_size,
        "residual_kernel_size": eccfg.residual_kernel_size,
        "dilation_growth_rate": eccfg.dilation_growth_rate,
        "use_causal_conv": eccfg.use_causal_conv,
        "pad_mode": eccfg.pad_mode,
        "compress": eccfg.compress,
        "num_lstm_layers": eccfg.num_lstm_layers,
        "trim_right_ratio": eccfg.trim_right_ratio,
        "codebook_size": eccfg.codebook_size,
        "codebook_dim": eccfg.codebook_dim,
        "target_bandwidths": eccfg.target_bandwidths,
    }
    with open(os.path.join(FIX, "tiny_musicgen_encodec_config.json"), "w") as f:
        json.dump(ec_out_cfg, f, indent=1)
    print("wrote tiny_musicgen_t5enc.safetensors (%d tensors)" % len(t5_sd))
    print("wrote tiny_musicgen_encodec.safetensors (%d tensors)" % len(ec_sd))

    st = os.path.getsize(os.path.join(FIX, "tiny_musicgen.safetensors"))
    print("wrote tiny_musicgen.safetensors %d bytes" % st)
    print("logits shape", list(logits.shape),
          "delayed shape", list(delayed_ids.shape))


def make_stereo():
    """Stereo (audio_channels=2) MusicGen parity fixture.

    Stereo MusicGen doubles the decoder to 2*K codebook rows: the two channels'
    per-channel codebooks are INTERLEAVED (row 2c = left codebook c, row 2c+1 =
    right codebook c) and each pair shares delay offset c. The transformer trunk
    is otherwise channel-agnostic (2*K embedding tables summed in, 2*K LM heads),
    so this pins (a) the decoder forward at 2*K rows and (b) the stereo
    delay-pattern round trip (HF build_delay_pattern_mask, audio_channels==2).
    Writes tiny_musicgen_stereo{.safetensors,_config.json,_ref.json}.
    """
    stereo_channels = 2
    channel_cb = NUM_CODEBOOKS          # per-channel codebooks (K)
    total_cb = stereo_channels * channel_cb  # decoder rows (2*K)

    torch.manual_seed(SEED + 1)
    np.random.seed(SEED + 1)
    cfg = build_config(audio_channels=stereo_channels, num_codebooks=total_cb)
    model = MusicgenForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        model.enc_to_dec_proj.weight.mul_(12.0)
        for layer in model.decoder.model.decoder.layers:
            ca = layer.encoder_attn
            for proj in (ca.q_proj, ca.k_proj, ca.v_proj, ca.out_proj):
                proj.weight.mul_(4.0)

    model_f64 = MusicgenForConditionalGeneration(cfg).double().eval()
    model_f64.load_state_dict(model.state_dict())

    rng = np.random.RandomState(SEED + 1)
    enc_states = (rng.randn(ENC_SEQ, TEXT_DMODEL) * 0.5)
    enc_states = np.round(enc_states * 64.0) / 64.0
    # 2*K interleaved code rows.
    dec_codes = rng.randint(0, VOCAB, size=(total_cb, DEC_SEQ))

    with torch.no_grad():
        proj = model_f64.enc_to_dec_proj
        enc_t = torch.tensor(enc_states, dtype=torch.float64).unsqueeze(0)
        enc_hidden = proj(enc_t)
        decoder: MusicgenForCausalLM = model_f64.decoder
        ids = torch.tensor(dec_codes, dtype=torch.long).reshape(
            total_cb, DEC_SEQ)
        out = decoder(input_ids=ids, encoder_hidden_states=enc_hidden,
                      use_cache=False)
        logits = out.logits.reshape(total_cb, DEC_SEQ, VOCAB)

    # ---- stereo delay-pattern round-trip oracle ----
    raw = rng.randint(0, VOCAB, size=(total_cb, DEC_SEQ))
    pad_id = VOCAB
    with torch.no_grad():
        gen_decoder = model.decoder
        raw_ids = torch.tensor(raw, dtype=torch.long).reshape(total_cb, DEC_SEQ)
        # stereo max delay offset is channel_cb-1, so max_length needs the same
        # +channel_cb headroom the importer's Steps uses.
        max_len = DEC_SEQ + channel_cb
        delayed_ids, mask = gen_decoder.build_delay_pattern_mask(
            raw_ids, pad_token_id=pad_id, max_length=max_len)
        delayed_ids = delayed_ids.reshape(total_cb, -1)
        mask = mask.reshape(total_cb, -1)

    os.makedirs(FIX, exist_ok=True)
    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("audio_encoder") or k.startswith("text_encoder"):
            continue
        sd[k] = v.to(torch.float32).contiguous()
    save_file(sd, os.path.join(FIX, "tiny_musicgen_stereo.safetensors"))

    out_cfg = {
        "model_type": "musicgen",
        "text_d_model": TEXT_DMODEL,
        "decoder": {
            "vocab_size": VOCAB,
            "hidden_size": DEC_HIDDEN,
            "num_hidden_layers": DEC_LAYERS,
            "num_attention_heads": DEC_HEADS,
            "ffn_dim": FFN,
            "num_codebooks": total_cb,
            "max_position_embeddings": 64,
            "activation_function": "gelu",
            "scale_embedding": False,
            "audio_channels": 2,
        },
    }
    with open(os.path.join(FIX, "tiny_musicgen_stereo_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=1)

    ref = {
        "text_d_model": TEXT_DMODEL,
        "dec_hidden": DEC_HIDDEN,
        "vocab_size": VOCAB,
        "num_codebooks": total_cb,
        "audio_channels": 2,
        "enc_seq_len": ENC_SEQ,
        "dec_seq_len": DEC_SEQ,
        "enc_states": enc_states.tolist(),
        "dec_codes": dec_codes.astype(np.int64).tolist(),
        "logits": logits.to(torch.float64).tolist(),
        "delay": {
            "channels": stereo_channels,
            "pad_id": pad_id,
            "max_length": int(max_len),
            "raw": raw.astype(np.int64).tolist(),
            "delayed": delayed_ids.to(torch.int64).tolist(),
            "mask": mask.to(torch.int64).tolist(),
        },
    }
    with open(os.path.join(FIX, "tiny_musicgen_stereo_ref.json"), "w") as f:
        json.dump(ref, f)
    st = os.path.getsize(os.path.join(FIX, "tiny_musicgen_stereo.safetensors"))
    print("wrote tiny_musicgen_stereo.safetensors %d bytes" % st)
    print("stereo logits shape", list(logits.shape),
          "delayed shape", list(delayed_ids.shape))


if __name__ == "__main__":
    main()
    make_stereo()
