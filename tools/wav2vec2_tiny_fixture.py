#!/usr/bin/env python3
"""Generate a tiny RANDOM Wav2Vec2 (and HuBERT) CTC parity fixture for
tests/TestNeuralPretrained.pas (no network access: both models are
randomly initialized from a pico config, never downloaded).

Two fixtures, ~15 KB each, pinned in tests/fixtures/:

  tiny_wav2vec2.*: Wav2Vec2ForCTC, the canonical wav2vec2-base topology
      reduced to pico width. The traits that distinguish this self-
      supervised SPEECH encoder from the landed Whisper seq2seq and the
      BERT text encoders, all of which the parity test must exercise:
        - a raw-waveform multi-layer STRIDED 1-D conv feature extractor;
          conv_layers[0] is conv -> GroupNorm(num_groups=channels) -> GELU
          (the "group" feat_extract_norm), the rest are conv -> GELU only;
          convs carry NO bias (conv_bias False);
        - a feature_projection: LayerNorm over the last conv channels then
          a biased Linear projecting to hidden_size;
        - a conv-based relative POSITIONAL EMBEDDING (a grouped conv1d,
          even kernel so a SamePad layer drops the last frame, then GELU)
          ADDED to the encoder input, then encoder.layer_norm;
        - POST-LN transformer blocks (do_stable_layer_norm False):
            x := LN(x + Attn(x)); x := final_LN(x + FFN(x));
        - exact erf "gelu" everywhere; bidirectional attention;
        - a linear CTC head (lm_head) over the vocab.
      The reference (HF transformers, float64) is the encoder
      last_hidden_state AND the CTC logits for every clip.

  tiny_hubert.*: HubertForCTC with the IDENTICAL topology and CTC head -
      proves the same importer path (with the hubert flag) loads both.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/wav2vec2_tiny_fixture.py
writes tests/fixtures/tiny_wav2vec2{,_hubert}{.safetensors,_config.json,
_ref.json}. Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import (Wav2Vec2Config, Wav2Vec2ForCTC,
                          HubertConfig, HubertForCTC)

# Pico topology. The conv stack maps a raw clip of RAW_LEN samples to a
# short encoder sequence; the chosen kernels/strides keep that >= the
# pos-conv kernel so the relative-position conv fits.
CONV_DIM = (8, 8, 8)
CONV_STRIDE = (5, 2, 2)
CONV_KERNEL = (10, 3, 3)
HIDDEN = 16
N_LAYER = 2
N_HEAD = 2
INTER = 32
POS_KERNEL = 8       # num_conv_pos_embeddings (even -> SamePad drops 1)
POS_GROUPS = 4       # num_conv_pos_embedding_groups
VOCAB = 12
N_CLIPS = 3
RAW_LEN = 200        # raw audio samples per clip

torch.manual_seed(20260614)


def conv_out_len(n):
    for k, s in zip(CONV_KERNEL, CONV_STRIDE):
        n = (n - k) // s + 1
    return n


def make_cfg(cfg_cls, **extra):
    return cfg_cls(
        hidden_size=HIDDEN,
        num_hidden_layers=N_LAYER,
        num_attention_heads=N_HEAD,
        intermediate_size=INTER,
        conv_dim=CONV_DIM,
        conv_stride=CONV_STRIDE,
        conv_kernel=CONV_KERNEL,
        conv_bias=False,
        num_conv_pos_embeddings=POS_KERNEL,
        num_conv_pos_embedding_groups=POS_GROUPS,
        feat_extract_norm='group',
        do_stable_layer_norm=False,
        hidden_act='gelu',
        feat_extract_activation='gelu',
        vocab_size=VOCAB,
        layer_norm_eps=1e-5,
        attn_implementation='eager',
        **extra)


def boost(model):
    # HF inits with std 0.02 -> at pico width attention scores ~0 and the
    # softmax is near-uniform, hiding the bidirectional structure. Boost
    # q/k/v and FFN so the attention pattern is O(1) and exact-vs-tanh
    # GELU is genuinely visible.
    with torch.no_grad():
        for layer in model.wav2vec2.encoder.layers \
                if hasattr(model, 'wav2vec2') \
                else model.hubert.encoder.layers:
            layer.attention.q_proj.weight.normal_(0.0, 0.7)
            layer.attention.k_proj.weight.normal_(0.0, 0.7)
            layer.attention.v_proj.weight.normal_(0.0, 0.5)
            layer.feed_forward.intermediate_dense.weight.normal_(0.0, 1.0)


def dump(model, backbone, name, model_type):
    cfg_dict = {
        'model_type': model_type,
        'architectures': [model.__class__.__name__],
        'hidden_size': HIDDEN,
        'num_hidden_layers': N_LAYER,
        'num_attention_heads': N_HEAD,
        'intermediate_size': INTER,
        'conv_dim': list(CONV_DIM),
        'conv_stride': list(CONV_STRIDE),
        'conv_kernel': list(CONV_KERNEL),
        'conv_bias': False,
        'num_conv_pos_embeddings': POS_KERNEL,
        'num_conv_pos_embedding_groups': POS_GROUPS,
        'feat_extract_norm': 'group',
        'do_stable_layer_norm': False,
        'hidden_act': 'gelu',
        'feat_extract_activation': 'gelu',
        'vocab_size': VOCAB,
        'layer_norm_eps': 1e-5,
    }
    sd = {k: v.to(torch.float32).contiguous()
          for k, v in model.state_dict().items()}
    save_file(sd, f'tests/fixtures/{name}.safetensors')
    with open(f'tests/fixtures/{name}_config.json', 'w') as f:
        json.dump(cfg_dict, f, indent=1)

    clips = [[round((torch.sin(torch.tensor((c + 1) * 0.07 * t)) * 0.5
               + 0.1 * ((7 * t + 3 * c) % 5 - 2)).item(), 4)
              for t in range(RAW_LEN)] for c in range(N_CLIPS)]
    hidden, logits = [], []
    with torch.no_grad():
        for clip in clips:
            x = torch.tensor([clip], dtype=torch.float64)
            enc = backbone(x).last_hidden_state
            lg = model(x).logits
            hidden.append(enc[0].tolist())
            logits.append(lg[0].tolist())
    with open(f'tests/fixtures/{name}_ref.json', 'w') as f:
        json.dump({'clips': clips, 'hidden': hidden, 'logits': logits,
                   'enc_len': len(hidden[0])}, f)
    print(f'wrote {name}: {len(sd)} tensors, raw {RAW_LEN} -> enc '
          f'{len(hidden[0])} frames, {N_CLIPS} clips')

    # ---- self-checks: every quirk must be visible in the reference ----
    with torch.no_grad():
        x0 = torch.tensor([clips[0]], dtype=torch.float64)
        base = backbone(x0).last_hidden_state
        # bidirectionality: perturbing the LAST raw sample must move frame 0.
        pclip = list(clips[0]); pclip[-1] += 1.0
        pert = backbone(torch.tensor([pclip], dtype=torch.float64)) \
            .last_hidden_state
        bidir = (base[0, 0] - pert[0, 0]).abs().max().item()
        assert bidir > 1e-4, f'last sample did not reach frame 0 ({bidir})'
        print(f'  bidirectional flow (last sample -> frame 0): {bidir:.4f}')
        # exact-vs-tanh GELU must differ (catches a tanh importer).
        tanh_cfg = make_cfg(type(backbone.config))
        tanh_cfg.hidden_act = 'gelu_pytorch_tanh'
        tanh_cfg.feat_extract_activation = 'gelu_pytorch_tanh'
        tanh_model = type(model)(tanh_cfg)
        tanh_model.load_state_dict(model.state_dict())
        tanh_model = tanh_model.double().eval()
        tbb = tanh_model.wav2vec2 if hasattr(tanh_model, 'wav2vec2') \
            else tanh_model.hubert
        tout = tbb(x0).last_hidden_state
        gelu = (base - tout).abs().max().item()
        assert gelu > 2.5e-5, f'exact-vs-tanh GELU invisible ({gelu})'
        print(f'  exact-vs-tanh GELU effect: {gelu:.2e}')


print('conv output length for RAW_LEN', RAW_LEN, '=', conv_out_len(RAW_LEN),
      '(must be >=', POS_KERNEL, ')')
assert conv_out_len(RAW_LEN) >= POS_KERNEL

w = Wav2Vec2ForCTC(make_cfg(Wav2Vec2Config))
boost(w)
w = w.double().eval()
dump(w, w.wav2vec2, 'tiny_wav2vec2', 'wav2vec2')

h = HubertForCTC(make_cfg(HubertConfig))
boost(h)
h = h.double().eval()
dump(h, h.hubert, 'tiny_hubert', 'hubert')
