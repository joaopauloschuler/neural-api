#!/usr/bin/env python3
"""Generate a tiny RANDOM Parler-TTS parity fixture for
tests/TestNeuralPretrained.pas (no network access, no parler_tts package: a
SELF-CONTAINED numpy float64 oracle re-implements the Parler decoder step from
the published architecture, exactly as the F5-TTS / InternLM2 fixtures do for
models that are not in stock transformers).

Parler-TTS (parler-tts/parler-tts-mini-v1, model_type "parler_tts", Lyth &
King 2024) is a description-conditioned text-to-speech model:

  * a (By)T5 text ENCODER encodes a free-text STYLE DESCRIPTION; its hidden
    states condition the decoder through CROSS-ATTENTION (reuses
    BuildT5FromSafeTensors);
  * a codec-LM DECODER autoregressively predicts the DELAY-PATTERNED
    multi-codebook DAC code stack. It is architecturally the MusicGen decoder
    (BART/Marian-style PRE-norm cross-attention blocks, BIAS-FREE linears, K
    embedding tables summed at the input, a SINUSOIDAL position table, a final
    decoder LayerNorm, K LM heads, the standard delay pattern). The genuinely
    NEW Parler piece is the DUAL PROMPT: the TRANSCRIPT prompt token ids are
    embedded by a SEPARATE learned table (embed_prompts) and PREPENDED on the
    sequence axis before the codec frames, so the decoder attends over
    [transcript_prefix | codec_frames] while ALSO cross-attending the
    description. Logits are read only at the codec-frame positions.
  * a DAC decoder renders the waveform (reuses BuildDACFromSafeTensors).

This fixture pins the DECODER forward pass (the prefix-prepended,
cross-attended codec LM - the new wiring) to < 1e-4. The T5 encoder and DAC
decoder are covered by their own importers/tests; here the decoder is fed a
FIXED encoder-hidden-state tensor (what enc_to_dec_proj(T5) would emit) and a
FIXED transcript prefix, and the next-codebook logits are gated.

The oracle is pure numpy float64 and mirrors, op for op, the Pascal importer:
the SAME random state dict is saved to safetensors, so importer parity is
exact (not just architectural).

Fixtures, KB-scale, pinned in tests/fixtures/:
  tiny_parler.safetensors : decoder + enc_to_dec_proj + embed_prompts weights
  tiny_parler_config.json : pico decoder sub-config + text_d_model + prompt dims
  tiny_parler_ref.json    : the float64 oracle (enc_states, prompt ids, codec
                            codes, logits)

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/parler_tiny_fixture.py
"""
import json
import math
import os

import numpy as np
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

SEED = 4242

TEXT_DMODEL = 12     # T5 d_model (!= dec hidden -> enc_to_dec_proj exercised)
DEC_HIDDEN = 8
VOCAB = 16           # DAC codebook size (decoder vocab_size)
NUM_CODEBOOKS = 4
DEC_LAYERS = 2
DEC_HEADS = 2
FFN = 16
ENC_SEQ = 5          # description (cross-attn) length
PROMPT_VOCAB = 20    # transcript token vocabulary
PROMPT_LEN = 3       # transcript prefix length
CODEC_SEQ = 6        # codec frames (logits read here)
LN_EPS = 1e-5


def gelu(x):
    # exact erf GELU (HF "gelu")
    from math import erf
    f = np.vectorize(lambda v: 0.5 * v * (1.0 + erf(v / math.sqrt(2.0))))
    return f(x)


def layer_norm(x, w, b, eps=LN_EPS):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def mha(q, k, v, num_heads, causal):
    # q [Lq, D], k/v [Lk, D]
    Lq, D = q.shape
    Lk = k.shape[0]
    hd = D // num_heads
    out = np.zeros((Lq, D), dtype=np.float64)
    scale = 1.0 / math.sqrt(hd)
    for h in range(num_heads):
        qh = q[:, h * hd:(h + 1) * hd]
        kh = k[:, h * hd:(h + 1) * hd]
        vh = v[:, h * hd:(h + 1) * hd]
        scores = (qh @ kh.T) * scale
        if causal:
            mask = np.triu(np.ones((Lq, Lk)), k=1).astype(bool)
            scores = np.where(mask, -1e30, scores)
        att = softmax(scores, axis=-1)
        out[:, h * hd:(h + 1) * hd] = att @ vh
    return out


def main():
    rng = np.random.RandomState(SEED)

    def randn(*shape, scale=1.0):
        return (rng.randn(*shape) * scale).astype(np.float64)

    # ---- weights (the SAME dict saved to safetensors) -------------------
    W = {}
    # K embedding tables (vocab+1 rows; extra row = delay pad id = VOCAB)
    for k in range(NUM_CODEBOOKS):
        W["decoder.model.decoder.embed_tokens.%d.weight" % k] = randn(
            VOCAB + 1, DEC_HIDDEN, scale=0.5)
    # transcript prompt embedding table (the NEW Parler piece)
    W["embed_prompts.weight"] = randn(PROMPT_VOCAB, DEC_HIDDEN, scale=0.5)
    # enc_to_dec_proj (biased): hidden x text_d_model. Amplified so the
    # description genuinely steers (same trick as MusicGen/ModernBERT pico).
    W["enc_to_dec_proj.weight"] = randn(DEC_HIDDEN, TEXT_DMODEL, scale=0.5) * 12.0
    W["enc_to_dec_proj.bias"] = randn(DEC_HIDDEN, scale=0.1)
    # per-layer (bias-free q/k/v/out + fc1/fc2; LN with weight+bias)
    for li in range(DEC_LAYERS):
        bp = "decoder.model.decoder.layers.%d." % li
        for nm in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                   "self_attn.out_proj"):
            W[bp + nm + ".weight"] = randn(DEC_HIDDEN, DEC_HIDDEN, scale=0.5)
        for nm in ("encoder_attn.q_proj", "encoder_attn.k_proj",
                   "encoder_attn.v_proj", "encoder_attn.out_proj"):
            W[bp + nm + ".weight"] = randn(DEC_HIDDEN, DEC_HIDDEN, scale=0.5) * 2.0
        W[bp + "fc1.weight"] = randn(FFN, DEC_HIDDEN, scale=0.5)
        W[bp + "fc2.weight"] = randn(DEC_HIDDEN, FFN, scale=0.5)
        for nm in ("self_attn_layer_norm", "encoder_attn_layer_norm",
                   "final_layer_norm"):
            W[bp + nm + ".weight"] = (1.0 + randn(DEC_HIDDEN, scale=0.1))
            W[bp + nm + ".bias"] = randn(DEC_HIDDEN, scale=0.1)
    W["decoder.model.decoder.layer_norm.weight"] = (
        1.0 + randn(DEC_HIDDEN, scale=0.1))
    W["decoder.model.decoder.layer_norm.bias"] = randn(DEC_HIDDEN, scale=0.1)
    for k in range(NUM_CODEBOOKS):
        W["decoder.lm_heads.%d.weight" % k] = randn(VOCAB, DEC_HIDDEN, scale=0.5)

    # ---- fixed inputs ----------------------------------------------------
    enc_states = np.round(randn(ENC_SEQ, TEXT_DMODEL, scale=0.5) * 64.0) / 64.0
    prompt_ids = rng.randint(0, PROMPT_VOCAB, size=PROMPT_LEN)
    codec_codes = rng.randint(0, VOCAB, size=(NUM_CODEBOOKS, CODEC_SEQ))

    # ---- sinusoidal position table (HF cat([cos,sin]) half-split) --------
    SEQ = PROMPT_LEN + CODEC_SEQ
    half = DEC_HIDDEN // 2
    emb_const = math.log(10000.0) / (half - 1)
    pos = np.zeros((SEQ, DEC_HIDDEN), dtype=np.float64)
    for p in range(SEQ):
        for c in range(half):
            ang = p * math.exp(-c * emb_const)
            pos[p, c] = math.cos(ang)
            pos[p, half + c] = math.sin(ang)

    # ---- build the decoder input sequence --------------------------------
    # transcript prefix: embed_prompts[prompt_ids]; codec frames: sum of K
    # codebook lookups. Then add the per-position sinusoid. (HF Parler adds the
    # positional embedding over the FULL [prefix|frames] sequence.)
    h = np.zeros((SEQ, DEC_HIDDEN), dtype=np.float64)
    for i, tid in enumerate(prompt_ids):
        h[i] = W["embed_prompts.weight"][tid]
    for t in range(CODEC_SEQ):
        acc = np.zeros(DEC_HIDDEN, dtype=np.float64)
        for k in range(NUM_CODEBOOKS):
            acc += W["decoder.model.decoder.embed_tokens.%d.weight" % k][
                codec_codes[k][t]]
        h[PROMPT_LEN + t] = acc
    h = h + pos

    # ---- projected encoder (description) states for cross-attention ------
    enc_hidden = enc_states @ W["enc_to_dec_proj.weight"].T + \
        W["enc_to_dec_proj.bias"]

    # ---- pre-norm cross-attention decoder blocks -------------------------
    for li in range(DEC_LAYERS):
        bp = "decoder.model.decoder.layers.%d." % li
        # self-attention (causal over the full prefix+frames sequence)
        r = h
        x = layer_norm(h, W[bp + "self_attn_layer_norm.weight"],
                       W[bp + "self_attn_layer_norm.bias"])
        q = x @ W[bp + "self_attn.q_proj.weight"].T
        k = x @ W[bp + "self_attn.k_proj.weight"].T
        v = x @ W[bp + "self_attn.v_proj.weight"].T
        a = mha(q, k, v, DEC_HEADS, causal=True)
        a = a @ W[bp + "self_attn.out_proj.weight"].T
        h = r + a
        # cross-attention (over description states)
        r = h
        x = layer_norm(h, W[bp + "encoder_attn_layer_norm.weight"],
                       W[bp + "encoder_attn_layer_norm.bias"])
        q = x @ W[bp + "encoder_attn.q_proj.weight"].T
        k = enc_hidden @ W[bp + "encoder_attn.k_proj.weight"].T
        v = enc_hidden @ W[bp + "encoder_attn.v_proj.weight"].T
        a = mha(q, k, v, DEC_HEADS, causal=False)
        a = a @ W[bp + "encoder_attn.out_proj.weight"].T
        h = r + a
        # FFN
        r = h
        x = layer_norm(h, W[bp + "final_layer_norm.weight"],
                       W[bp + "final_layer_norm.bias"])
        x = gelu(x @ W[bp + "fc1.weight"].T)
        x = x @ W[bp + "fc2.weight"].T
        h = r + x

    h = layer_norm(h, W["decoder.model.decoder.layer_norm.weight"],
                   W["decoder.model.decoder.layer_norm.bias"])

    # ---- K LM heads, read at the CODEC-FRAME positions only --------------
    logits = np.zeros((NUM_CODEBOOKS, CODEC_SEQ, VOCAB), dtype=np.float64)
    for k in range(NUM_CODEBOOKS):
        hh = W["decoder.lm_heads.%d.weight" % k]
        for t in range(CODEC_SEQ):
            logits[k, t] = h[PROMPT_LEN + t] @ hh.T

    # ---- save ------------------------------------------------------------
    os.makedirs(FIX, exist_ok=True)
    sd = {kk: vv.astype(np.float32) for kk, vv in W.items()}
    save_file(sd, os.path.join(FIX, "tiny_parler.safetensors"))

    out_cfg = {
        "model_type": "parler_tts",
        "text_d_model": TEXT_DMODEL,
        "prompt_vocab_size": PROMPT_VOCAB,
        "decoder": {
            "vocab_size": VOCAB,
            "hidden_size": DEC_HIDDEN,
            "num_hidden_layers": DEC_LAYERS,
            "num_attention_heads": DEC_HEADS,
            "ffn_dim": FFN,
            "num_codebooks": NUM_CODEBOOKS,
            "max_position_embeddings": 64,
            "activation_function": "gelu",
            "audio_channels": 1,
        },
    }
    with open(os.path.join(FIX, "tiny_parler_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=1)

    ref = {
        "text_d_model": TEXT_DMODEL,
        "dec_hidden": DEC_HIDDEN,
        "vocab_size": VOCAB,
        "num_codebooks": NUM_CODEBOOKS,
        "prompt_vocab_size": PROMPT_VOCAB,
        "enc_seq_len": ENC_SEQ,
        "prompt_len": PROMPT_LEN,
        "codec_seq_len": CODEC_SEQ,
        "enc_states": enc_states.tolist(),
        "prompt_ids": prompt_ids.astype(np.int64).tolist(),
        "codec_codes": codec_codes.astype(np.int64).tolist(),
        "logits": logits.tolist(),
    }
    with open(os.path.join(FIX, "tiny_parler_ref.json"), "w") as f:
        json.dump(ref, f)

    st = os.path.getsize(os.path.join(FIX, "tiny_parler.safetensors"))
    print("wrote tiny_parler.safetensors %d bytes, %d tensors" % (st, len(sd)))
    print("logits shape", logits.shape)


if __name__ == "__main__":
    main()
