#!/usr/bin/env python3
# Generates a TINY (pico) SeamlessM4T-v2 speech-to-text (S2TT) fixture with the
# v2 conformer RELATIVE-POSITION distance-embedding attention bias ENABLED
# (position_embeddings_type="relative_key") for the Free Pascal neural-api
# parity test (BuildSeamlessM4TFromSafeTensors + TNNetConformerRelPosAttention).
#
# This is the relative_key sibling of seamless_m4t_v2_tiny_fixture.py (which
# pins position_embeddings_type="" / vanilla SDPA). The ONLY structural change
# is position_embeddings_type="relative_key" plus left/right_max_position_
# embeddings, which adds a per-layer distance_embedding table whose bias HF adds
# to the conformer self-attention logits BEFORE softmax:
#   score[i,j] = ( Q_i.K_j + Q_i.P[clamp(j-i,-L,R)+L] ) / sqrt(head_size)
#
# The conformer ADAPTER self-attn always uses use_position_embeddings=False in
# HF, so it stays vanilla SDPA regardless of this flag (the Pascal importer
# matches: it only swaps the ENCODER-layer self-attn for the relpos variant).
#
# The HF default 0.02 init is vacuous at pico width, so every weight is re-drawn
# with a healthy std (recurring neural-api fixture gotcha) and the oracle is
# computed in float64. HF itself (in float64) IS the oracle; the committed
# *_relkey_logits.json bakes it in so the Pascal test never needs transformers.
#
# Writes (relative to repo root):
#   tests/fixtures/tiny_seamless_m4t_v2_relkey.safetensors
#   tests/fixtures/tiny_seamless_m4t_v2_relkey_config.json
#   tests/fixtures/tiny_seamless_m4t_v2_relkey_logits.json
#
# Run:  /home/bpsa/x/bin/python tools/seamless_m4t_v2_relkey_tiny_fixture.py

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
LEFT_MAX = 4          # left_max_position_embeddings (past clamp)
RIGHT_MAX = 2         # right_max_position_embeddings (future clamp)

SPEECH_FRAMES = 6     # raw input feature frames
DEC_LEN = 5           # decoder teacher-forced length

torch.manual_seed(20260626)


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
        position_embeddings_type="relative_key",   # ENABLE the distance bias
        left_max_position_embeddings=LEFT_MAX,
        right_max_position_embeddings=RIGHT_MAX,
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

    # sanity: the relative_key distance_embedding tables must exist
    sd0 = model.state_dict()
    assert any("self_attn.distance_embedding.weight" in k for k in sd0), \
        "distance_embedding tables missing -- relative_key not active?"

    # ----- inputs -----
    torch.manual_seed(7)
    input_features = torch.randn(1, SPEECH_FRAMES, FEAT_IN, dtype=torch.float64)
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
        if k == "lm_head.weight":
            drop.add(k)
        if k.endswith("embed_tokens.weight"):
            drop.add(k)
        if "embed_positions.weights" in k:
            drop.add(k)
    keep = {k: v.to(torch.float32).contiguous() for k, v in sd.items()
            if k not in drop and v.is_floating_point()}
    assert "shared.weight" in keep
    os.makedirs(FIX, exist_ok=True)
    save_file(keep, os.path.join(FIX, "tiny_seamless_m4t_v2_relkey.safetensors"))

    # ----- config json -----
    conf = {
        "model_type": "seamless_m4t_v2",
        "hidden_size": HIDDEN,
        "feature_projection_input_dim": FEAT_IN,
        "speech_encoder_layers": SPEECH_LAYERS,
        "speech_encoder_attention_heads": SPEECH_HEADS,
        "speech_encoder_intermediate_size": SPEECH_FFN,
        "conv_depthwise_kernel_size": DEPTHWISE_K,
        "position_embeddings_type": "relative_key",
        "left_max_position_embeddings": LEFT_MAX,
        "right_max_position_embeddings": RIGHT_MAX,
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
    with open(os.path.join(FIX, "tiny_seamless_m4t_v2_relkey_config.json"),
              "w") as f:
        json.dump(conf, f, indent=2)

    # ----- oracle json -----
    oracle = {
        "input_features": input_features[0].tolist(),
        "dec_sequences": dec_ids[0].tolist(),
        "adapted_len": int(adapted_len),
        "enc_hidden": enc_hidden[0].tolist(),
        "logits": logits[0].tolist(),
    }
    with open(os.path.join(FIX, "tiny_seamless_m4t_v2_relkey_logits.json"),
              "w") as f:
        json.dump(oracle, f)

    print("relative_key fixture: speech frames", SPEECH_FRAMES,
          "adapted_len", adapted_len, "left/right_max", LEFT_MAX, RIGHT_MAX)
    print("logits shape:", tuple(logits.shape))
    print("wrote fixtures to", FIX)


if __name__ == "__main__":
    main()
