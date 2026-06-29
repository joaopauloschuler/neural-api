#!/usr/bin/env python3
"""Generate a tiny RANDOM MERT music-representation parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

MERT (m-a-p/MERT-v1-95M, model_type "mert_model"/"music2vec") is a self-
supervised MUSIC understanding encoder - the audio analogue of a frozen
vision backbone. With the released MERT-v1-95M config the forward is
architecturally IDENTICAL to HuBERT (feature_extractor_cqt False,
attention_relax -1.0, deepnorm False, do_stable_layer_norm False): a
raw-waveform strided 1-D conv feature extractor -> feature_projection
(LayerNorm + Linear) -> conv relative positional embedding + encoder
LayerNorm -> POST-LN transformer blocks. So the oracle backbone here is a
pico HubertModel; MERTModel(cqt off) == HubertModel for the forward math.

The MERT-specific piece is the WEIGHTED-LAYER-SUM music embedding: the
deep weighted sum over ALL transformer hidden states (the embeddings
output plus each of the N block outputs = N+1 states) with a learned
per-layer weight vector normalized by softmax (HF use_weighted_layer_sum,
the *ForSequenceClassification layer_weights head). The base MERTModel
ships no layer_weights, so the importer keeps them as an explicit config
field (default uniform); here we plant a RANDOM layer_weights vector and
write both the per-layer hidden states AND the resulting weighted-sum
embedding so the test pins the trunk transpose AND the weighted sum.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/make_pico_mert_fixture.py
writes tests/fixtures/tiny_mert{.safetensors,_config.json,_ref.json}.
Needs torch + transformers + safetensors.
"""
import json

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import HubertConfig, HubertModel

# Pico topology. The conv stack maps a raw clip of RAW_LEN samples to a
# short encoder sequence; the chosen kernels/strides keep that >= the
# pos-conv kernel so the relative-position conv fits.
CONV_DIM = (8, 8, 8)
CONV_STRIDE = (5, 2, 2)
CONV_KERNEL = (10, 3, 3)
HIDDEN = 16
N_LAYER = 3          # N+1 = 4 hidden states feed the weighted sum
N_HEAD = 2
INTER = 32
POS_KERNEL = 8       # num_conv_pos_embeddings (even -> SamePad drops 1)
POS_GROUPS = 4       # num_conv_pos_embedding_groups
N_CLIPS = 3
RAW_LEN = 200        # raw audio samples per clip

torch.manual_seed(20260626)


def conv_out_len(n):
    for k, s in zip(CONV_KERNEL, CONV_STRIDE):
        n = (n - k) // s + 1
    return n


def make_cfg(**extra):
    return HubertConfig(
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
        layer_norm_eps=1e-5,
        attn_implementation='eager',
        **extra)


def boost(model):
    # HF inits with std 0.02 -> at pico width attention scores ~0 and the
    # softmax is near-uniform, hiding the bidirectional structure. Boost
    # q/k/v and FFN so the attention pattern is O(1) and exact-vs-tanh
    # GELU is genuinely visible across the trunk.
    with torch.no_grad():
        for layer in model.encoder.layers:
            layer.attention.q_proj.weight.normal_(0.0, 0.7)
            layer.attention.k_proj.weight.normal_(0.0, 0.7)
            layer.attention.v_proj.weight.normal_(0.0, 0.5)
            layer.feed_forward.intermediate_dense.weight.normal_(0.0, 1.0)


def main():
    print('conv output length for RAW_LEN', RAW_LEN, '=',
          conv_out_len(RAW_LEN), '(must be >=', POS_KERNEL, ')')
    assert conv_out_len(RAW_LEN) >= POS_KERNEL

    model = HubertModel(make_cfg())
    boost(model)
    model = model.double().eval()

    # The MERT-specific learned per-layer weights for the weighted-layer-sum
    # music embedding (softmax-normalized over the N+1 hidden states).
    torch.manual_seed(7)
    layer_weights = torch.randn(N_LAYER + 1, dtype=torch.float64)

    cfg_dict = {
        'model_type': 'mert_model',
        'architectures': ['MERTModel'],
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
        'feature_extractor_cqt': False,
        'attention_relax': -1.0,
        'deepnorm': False,
        'feat_proj_layer_norm': True,
        'hidden_act': 'gelu',
        'feat_extract_activation': 'gelu',
        'layer_norm_eps': 1e-5,
        'sample_rate': 24000,
        # The planted weighted-layer-sum weights (raw, pre-softmax). The
        # importer reads these into the per-layer weight vector; default
        # (absent) is uniform.
        'layer_weights': [round(float(x), 6) for x in layer_weights.tolist()],
    }
    # MERTModel ships the HuBERT backbone tensors at the TOP level (no
    # "hubert."/"wav2vec2." prefix), so the importer's prefix is ''.
    sd = {k: v.to(torch.float32).contiguous()
          for k, v in model.state_dict().items()}
    save_file(sd, 'tests/fixtures/tiny_mert.safetensors')
    with open('tests/fixtures/tiny_mert_config.json', 'w') as f:
        json.dump(cfg_dict, f, indent=1)

    clips = [[round((torch.sin(torch.tensor((c + 1) * 0.07 * t)) * 0.5
               + 0.1 * ((7 * t + 3 * c) % 5 - 2)).item(), 4)
              for t in range(RAW_LEN)] for c in range(N_CLIPS)]

    norm_w = F.softmax(layer_weights, dim=-1)  # (N+1,)
    last_hidden, all_layers, embedding = [], [], []
    with torch.no_grad():
        for clip in clips:
            x = torch.tensor([clip], dtype=torch.float64)
            out = model(x, output_hidden_states=True)
            last = out.last_hidden_state                  # (1, T, H)
            states = out.hidden_states                    # tuple len N+1
            assert len(states) == N_LAYER + 1
            stacked = torch.stack(states, dim=1)          # (1, N+1, T, H)
            # weighted-layer-sum over the N+1 hidden states.
            emb = (stacked * norm_w.view(1, -1, 1, 1)).sum(dim=1)  # (1,T,H)
            last_hidden.append(last[0].tolist())
            all_layers.append([s[0].tolist() for s in states])
            embedding.append(emb[0].tolist())
    with open('tests/fixtures/tiny_mert_ref.json', 'w') as f:
        json.dump({'clips': clips,
                   'last_hidden': last_hidden,
                   'all_hidden': all_layers,
                   'embedding': embedding,
                   'layer_weights': cfg_dict['layer_weights'],
                   'num_hidden': N_LAYER + 1,
                   'enc_len': len(last_hidden[0])}, f)
    print(f'wrote tiny_mert: {len(sd)} tensors, raw {RAW_LEN} -> enc '
          f'{len(last_hidden[0])} frames, {N_LAYER + 1} hidden states, '
          f'{N_CLIPS} clips')

    # ---- self-checks: every quirk must be visible in the reference ----
    with torch.no_grad():
        x0 = torch.tensor([clips[0]], dtype=torch.float64)
        base = model(x0).last_hidden_state
        # bidirectionality: perturbing the LAST raw sample must move frame 0.
        pclip = list(clips[0]); pclip[-1] += 1.0
        pert = model(torch.tensor([pclip], dtype=torch.float64)) \
            .last_hidden_state
        bidir = (base[0, 0] - pert[0, 0]).abs().max().item()
        assert bidir > 1e-4, f'last sample did not reach frame 0 ({bidir})'
        print(f'  bidirectional flow (last sample -> frame 0): {bidir:.4f}')
        # the weighted sum must differ from the last hidden state (the
        # shallow layers carry real weight) - catches a "just take last".
        emb0 = torch.tensor(embedding[0], dtype=torch.float64)
        last0 = torch.tensor(last_hidden[0], dtype=torch.float64)
        wls = (emb0 - last0).abs().max().item()
        assert wls > 1e-3, f'weighted sum == last hidden ({wls})'
        print(f'  weighted-layer-sum vs last hidden: {wls:.4f}')
        # exact-vs-tanh GELU must differ (catches a tanh importer).
        tanh_cfg = make_cfg()
        tanh_cfg.hidden_act = 'gelu_pytorch_tanh'
        tanh_cfg.feat_extract_activation = 'gelu_pytorch_tanh'
        tanh_model = HubertModel(tanh_cfg)
        tanh_model.load_state_dict(model.state_dict())
        tanh_model = tanh_model.double().eval()
        tout = tanh_model(x0).last_hidden_state
        gelu = (base - tout).abs().max().item()
        assert gelu > 2.5e-5, f'exact-vs-tanh GELU invisible ({gelu})'
        print(f'  exact-vs-tanh GELU effect: {gelu:.2e}')


if __name__ == '__main__':
    main()
