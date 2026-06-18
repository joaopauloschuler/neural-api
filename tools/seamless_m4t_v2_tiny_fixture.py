#!/usr/bin/env python3
# Generates a TINY (pico) SeamlessM4T-v2 speech-to-text (S2TT) fixture for the
# Free Pascal neural-api parity test (BuildSeamlessM4TFromSafeTensors).
#
# Scope: v1 = S2TT only. The conformer SPEECH encoder (feature_projection ->
# conformer layers -> encoder.layer_norm -> intermediate_ffn -> conformer
# ADAPTER -> inner_layer_norm) feeding an NLLB-style TEXT decoder that
# cross-attends to the adapted encoder states.
#
# Deliberate pico-config choice: position_embeddings_type="" DISABLES the
# v2 conformer "relative_key" distance-embedding attention bias, so the
# conformer self-attention is vanilla scaled-dot-product attention that the
# repo's TNNetScaledDotProductAttention can represent exactly. The relative_key
# bias + the T2ST unit vocoder + UnitY2 two-pass decoding are deferred
# follow-ups (see tasklist.md).
#
# HF default 0.02 init is vacuous at pico width, so every weight is re-drawn
# with a healthy std (recurring neural-api fixture gotcha) and the oracle is
# computed in float64.
#
# Writes (relative to repo root):
#   tests/fixtures/tiny_seamless_m4t_v2.safetensors
#   tests/fixtures/tiny_seamless_m4t_v2_config.json
#   tests/fixtures/tiny_seamless_m4t_v2_logits.json
#
# Run:  /home/bpsa/x/bin/python tools/seamless_m4t_v2_tiny_fixture.py

import json
import os
import torch

from transformers import SeamlessM4Tv2Config, SeamlessM4Tv2ForSpeechToText
from safetensors.torch import save_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "tests", "fixtures")

# ----- pico hyper-parameters -----
HIDDEN = 16
FEAT_IN = 12          # feature_projection_input_dim (log-mel-ish front-end dim)
SPEECH_LAYERS = 2
SPEECH_HEADS = 2
SPEECH_FFN = 32
DEPTHWISE_K = 5       # conv_depthwise_kernel_size (odd)
ADAPT_K = 4           # adaptor_kernel_size
ADAPT_S = 2           # adaptor_stride
DEC_LAYERS = 2
DEC_HEADS = 2
DEC_FFN = 32
VOCAB = 24
MAX_POS = 64

SPEECH_FRAMES = 6     # raw input feature frames
DEC_LEN = 5           # decoder teacher-forced length

torch.manual_seed(20260615)


def boost(model, std=0.6):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0.0, std)
            else:
                # biases / layernorm: small but non-trivial so each is testable
                p.normal_(0.0, 0.15)
            # LayerNorm weights default to 1.0; recentre around 1 so norm is sane
            if name.endswith("layer_norm.weight") or name.endswith(
                    "_layer_norm.weight") or name == "inner_layer_norm.weight":
                p.add_(1.0)


def main():
    cfg = SeamlessM4Tv2Config(
        hidden_size=HIDDEN,
        speech_encoder_layers=SPEECH_LAYERS,
        speech_encoder_attention_heads=SPEECH_HEADS,
        speech_encoder_intermediate_size=SPEECH_FFN,
        conv_depthwise_kernel_size=DEPTHWISE_K,
        position_embeddings_type="",          # disable relative_key bias (v1 scope)
        feature_projection_input_dim=FEAT_IN,
        adaptor_kernel_size=ADAPT_K,
        adaptor_stride=ADAPT_S,
        num_adapter_layers=1,
        add_adapter=True,
        speech_encoder_hidden_act="swish",
        decoder_layers=DEC_LAYERS,
        decoder_attention_heads=DEC_HEADS,
        decoder_ffn_dim=DEC_FFN,
        vocab_size=VOCAB,
        max_position_embeddings=MAX_POS,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        decoder_start_token_id=3,
        scale_embedding=True,
        activation_function="relu",
        layer_norm_eps=1e-5,
        use_cache=False,
    )

    model = SeamlessM4Tv2ForSpeechToText(cfg)
    boost(model)
    model.double().eval()

    # ----- inputs -----
    torch.manual_seed(7)
    input_features = torch.randn(1, SPEECH_FRAMES, FEAT_IN, dtype=torch.float64)
    # decoder ids: start (eos=3) then in-vocab non-pad tokens, no pad (0)
    dec_ids = torch.tensor([[3, 5, 9, 14, 20]], dtype=torch.long)
    assert dec_ids.shape[1] == DEC_LEN

    with torch.no_grad():
        se_out = model.speech_encoder(input_features=input_features)
        enc_hidden = se_out.last_hidden_state  # (1, adapted_len, hidden)
        adapted_len = enc_hidden.shape[1]
        out = model(input_features=input_features, decoder_input_ids=dec_ids)
        logits = out.logits  # (1, DEC_LEN, vocab)

    # ----- save safetensors (drop tied/derived tensors) -----
    sd = model.state_dict()
    drop = set()
    for k in list(sd.keys()):
        # tied head/embeddings + sinusoidal position buffer are regenerated
        if k == "lm_head.weight":
            drop.add(k)
        if k.endswith("embed_tokens.weight"):
            drop.add(k)
        if "embed_positions.weights" in k:
            drop.add(k)
    keep = {k: v.to(torch.float32).contiguous() for k, v in sd.items()
            if k not in drop and v.is_floating_point()}
    # ensure shared.weight kept (the single source of truth for embed + head)
    assert "shared.weight" in keep
    os.makedirs(FIX, exist_ok=True)
    save_file(keep, os.path.join(FIX, "tiny_seamless_m4t_v2.safetensors"))

    # ----- config json -----
    conf = {
        "model_type": "seamless_m4t_v2",
        "hidden_size": HIDDEN,
        "feature_projection_input_dim": FEAT_IN,
        "speech_encoder_layers": SPEECH_LAYERS,
        "speech_encoder_attention_heads": SPEECH_HEADS,
        "speech_encoder_intermediate_size": SPEECH_FFN,
        "conv_depthwise_kernel_size": DEPTHWISE_K,
        "position_embeddings_type": "",
        "adaptor_kernel_size": ADAPT_K,
        "adaptor_stride": ADAPT_S,
        "num_adapter_layers": 1,
        "add_adapter": True,
        "speech_encoder_hidden_act": "swish",
        "decoder_layers": DEC_LAYERS,
        "decoder_attention_heads": DEC_HEADS,
        "decoder_ffn_dim": DEC_FFN,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POS,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "decoder_start_token_id": 3,
        "scale_embedding": True,
        "activation_function": "relu",
        "layer_norm_eps": 1e-5,
    }
    with open(os.path.join(FIX, "tiny_seamless_m4t_v2_config.json"), "w") as f:
        json.dump(conf, f, indent=2)

    # ----- oracle json -----
    oracle = {
        "input_features": input_features[0].tolist(),   # (frames, feat_in)
        "dec_sequences": dec_ids[0].tolist(),
        "adapted_len": int(adapted_len),
        "enc_hidden": enc_hidden[0].tolist(),           # (adapted_len, hidden)
        "logits": logits[0].tolist(),                   # (dec_len, vocab)
    }
    with open(os.path.join(FIX, "tiny_seamless_m4t_v2_logits.json"), "w") as f:
        json.dump(oracle, f)

    print("speech frames:", SPEECH_FRAMES, "adapted_len:", adapted_len)
    print("logits shape:", tuple(logits.shape))
    print("wrote fixtures to", FIX)


if __name__ == "__main__":
    main()
