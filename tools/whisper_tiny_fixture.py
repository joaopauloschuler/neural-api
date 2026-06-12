#!/usr/bin/env python3
"""Generate a tiny RANDOM Whisper parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded) plus the
log-mel FRONTEND oracle for tests of neural/neuralaudio.pas.

Fixtures, KB-scale, pinned in tests/fixtures/:

  tiny_whisper.*: WhisperForConditionalGeneration (the openai/whisper-*
      speech-to-text architecture) with every Whisper trait the importer
      must reproduce:
        - mel-spectrogram encoder frontend: Conv1d(num_mel_bins ->
          d_model, k=3, s=1, p=1) + GELU, Conv1d(d_model -> d_model,
          k=3, s=2, p=1) + GELU - the stride-2 conv HALVES the input
          frames (2*max_source_positions) to max_source_positions;
        - FIXED (non-learned) sinusoidal encoder positions in the
          Whisper layout: concat sin|cos halves with timescale exponent
          c/(half-1) - DROPPED from the checkpoint here so the importer
          regenerates them from the formula (real checkpoints carry the
          tensor; the importer loads it when present);
        - LEARNED decoder positions (saved, re-randomized);
        - PRE-norm blocks (LayerNorm BEFORE each sublayer, residual adds
          the raw stream) + a FINAL stack LayerNorm in BOTH stacks -
          plain biased nn.LayerNorm eps 1e-5;
        - EXACT erf GELU FFNs ("gelu", NOT the tanh gelu_new);
        - attention: q/v/out biased, k_proj BIAS-FREE, standard
          1/sqrt(head_dim) scaling; decoder self-attention causal;
          cross-attention reads the encoder states - RECTANGULAR
          max_source_positions x DecSeqLen scores;
        - tied lm head (proj_out = decoder.embed_tokens, bias-free).

  whisper_mel_oracle.json: WhisperFeatureExtractor log-mel output (80
      slaney-mel bins, 400-pt periodic hann STFT, hop 160, reflect
      center pad, log10 floored at 1e-10, global max-8 clamp, (x+4)/4)
      on a pinned 1-second 440 Hz / 0.5 amplitude sine at 16 kHz, padded
      to 30 s. Only the first 30 of the 3000 frames are committed (the
      tail is the all-zero padding plateau, asserted constant here and
      reproduced by the Pascal frontend).

The model reference is computed by HF transformers in float64 (the
oracle convention of the committed fixtures): encoder final hidden
states plus full encoder-decoder logits for a pinned float mel input
(dyadic values, exactly representable in BOTH float32 and JSON decimal).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/whisper_tiny_fixture.py
writes tests/fixtures/tiny_whisper{.safetensors,_config.json,_logits.json}
and tests/fixtures/whisper_mel_oracle.json.
Needs torch + transformers + safetensors + numpy.
"""
import copy
import json
import math

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import (WhisperConfig, WhisperFeatureExtractor,
                          WhisperForConditionalGeneration)
from transformers.models.whisper.modeling_whisper import sinusoids

N_LAYER = 2
N_HEAD = 2                    # head_dim = D_MODEL / N_HEAD = 6
D_MODEL = 12
D_FF = 24
VOCAB = 31
N_MELS = 80                   # FULL 80 mel bins - the real frontend width
MAX_SRC = 13                  # encoder length; mel input = 2*13 = 26 frames
MAX_TGT = 12
DEC_LEN = 6
N_SEQUENCES = 2
DEC_START = 2

torch.manual_seed(20260612)

cfg_dict = {
    'architectures': ['WhisperForConditionalGeneration'],
    'model_type': 'whisper',
    'd_model': D_MODEL,
    'encoder_layers': N_LAYER,
    'decoder_layers': N_LAYER,
    'encoder_attention_heads': N_HEAD,
    'decoder_attention_heads': N_HEAD,
    'encoder_ffn_dim': D_FF,
    'decoder_ffn_dim': D_FF,
    'vocab_size': VOCAB,
    'num_mel_bins': N_MELS,
    'max_source_positions': MAX_SRC,
    'max_target_positions': MAX_TGT,
    'activation_function': 'gelu',
    'scale_embedding': False,
    'tie_word_embeddings': True,
    'dropout': 0.0,
    'attention_dropout': 0.0,
    'activation_dropout': 0.0,
    'pad_token_id': 0,
    'bos_token_id': 1,
    'eos_token_id': 1,
    'decoder_start_token_id': DEC_START,
}
model = WhisperForConditionalGeneration(
    WhisperConfig(**cfg_dict, attn_implementation='eager'))

# HF inits with std 0.02 at this pico width - vacuously small (the
# ModernBERT lesson: near-identity blocks pass parity even when wired
# wrong). Re-randomize at O(1) scale so every quirk is visible in the
# oracle: O(1) attention scores, FFN/conv pre-activations in the region
# where erf-GELU and tanh-GELU genuinely differ, norms away from (1,0).
with torch.no_grad():
    enc = model.model.encoder
    dec = model.model.decoder
    enc.conv1.weight.normal_(0.0, 0.30)
    enc.conv1.bias.normal_(0.0, 0.25)
    enc.conv2.weight.normal_(0.0, 0.45)
    enc.conv2.bias.normal_(0.0, 0.25)
    dec.embed_tokens.weight.normal_(0.0, 0.55)
    dec.embed_positions.weight.normal_(0.0, 0.45)
    for stack_layers in (enc.layers, dec.layers):
        for layer in stack_layers:
            attns = [layer.self_attn]
            if hasattr(layer, 'encoder_attn'):
                attns.append(layer.encoder_attn)
            for attn in attns:
                for proj in (attn.q_proj, attn.k_proj, attn.v_proj,
                             attn.out_proj):
                    proj.weight.normal_(0.0, 0.45)
                    if proj.bias is not None:
                        proj.bias.normal_(0.0, 0.2)
            layer.fc1.weight.normal_(0.0, 0.55)
            layer.fc1.bias.normal_(0.0, 0.3)
            layer.fc2.weight.normal_(0.0, 0.4)
            layer.fc2.bias.normal_(0.0, 0.2)
            norms = [layer.self_attn_layer_norm, layer.final_layer_norm]
            if hasattr(layer, 'encoder_attn_layer_norm'):
                norms.append(layer.encoder_attn_layer_norm)
            for norm in norms:
                norm.weight.normal_(1.0, 0.25)
                norm.bias.normal_(0.0, 0.2)
    for norm in (enc.layer_norm, dec.layer_norm):
        norm.weight.normal_(1.0, 0.25)
        norm.bias.normal_(0.0, 0.2)
model = model.double().eval()

# k_proj must be bias-free and the head tied - the importer relies on it.
for layer in list(enc.layers) + list(dec.layers):
    assert layer.self_attn.k_proj.bias is None, 'k_proj grew a bias'
assert model.proj_out.weight is dec.embed_tokens.weight, 'head not tied'

# The encoder sinusoidal table must match the documented formula (it was
# NOT re-randomized above), then gets dropped from the checkpoint so the
# importer's regeneration path is exercised.
ref_tab = sinusoids(MAX_SRC, D_MODEL).double()
d = (enc.embed_positions.weight - ref_tab).abs().max().item()
assert d < 1e-6, f'unexpected encoder sinusoid layout ({d})'

drop = {
    'model.encoder.embed_positions.weight',  # sinusoidal - regenerated
    'proj_out.weight',                       # tied to embed_tokens
}
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items() if k not in drop}
save_file(sd, 'tests/fixtures/tiny_whisper.safetensors')
with open('tests/fixtures/tiny_whisper_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

# Pinned mel inputs: dyadic values (multiples of 1/256) in roughly the
# post-normalization range of real Whisper log-mels, so JSON decimal,
# float32 and float64 all agree EXACTLY on the input.
rng = np.random.default_rng(424242)
N_FRAMES = 2 * MAX_SRC
mel_inputs = [
    (rng.integers(-256, 385, size=(N_MELS, N_FRAMES)) / 256.0).tolist()
    for _ in range(N_SEQUENCES)]
dec_sequences = [[DEC_START] + [(3 * i + 2 * s + 1) % (VOCAB - 3) + 3
                                for i in range(DEC_LEN - 1)]
                 for s in range(N_SEQUENCES)]

enc_hidden, logits = [], []
with torch.no_grad():
    for mel, ds in zip(mel_inputs, dec_sequences):
        feats = torch.tensor([mel], dtype=torch.float64)
        dec_ids = torch.tensor([ds])
        enc_hidden.append(
            enc(input_features=feats).last_hidden_state[0].tolist())
        logits.append(
            model(input_features=feats,
                  decoder_input_ids=dec_ids).logits[0].tolist())
with open('tests/fixtures/tiny_whisper_logits.json', 'w') as f:
    json.dump({'mel_inputs': mel_inputs, 'dec_sequences': dec_sequences,
               'enc_hidden': enc_hidden, 'logits': logits}, f)
print(f'wrote tiny_whisper.safetensors ({len(sd)} tensors)'
      ' + config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---------------- frontend oracle: 440 Hz sine log-mel ----------------
SR = 16000
sine = (0.5 * np.sin(2.0 * np.pi * 440.0 * np.arange(SR) / SR)).astype(
    np.float64)
fe = WhisperFeatureExtractor(feature_size=80)
mel = fe(sine, sampling_rate=SR, padding='max_length',
         return_tensors='np').input_features[0]   # (80, 3000)
assert mel.shape == (80, 3000), mel.shape
# The padding tail must be one constant plateau (all-zero signal frames
# floored at 1e-10 then max-8-clamped) - the Pascal test asserts the same
# value, so the WHOLE 3000-frame normalization is pinned by the global
# max + this plateau, without committing 3000 frames.
tail = mel[:, 200:]
assert np.ptp(tail) < 1e-7, 'padding tail is not constant'
ORACLE_FRAMES = 30
oracle = {
    'description': '440Hz 0.5-amplitude 1s sine at 16kHz, '
                   'WhisperFeatureExtractor log-mel; frames 0..29 of 3000',
    'num_mel_bins': 80,
    'num_frames': 3000,
    'oracle_frames': ORACLE_FRAMES,
    'padding_tail_value': float(tail[0, 0]),
    'global_max': float(mel.max()),
    # frames-major: oracle_mel[frame][mel_bin]
    'oracle_mel': [[round(float(mel[m, t]), 8) for m in range(80)]
                   for t in range(ORACLE_FRAMES)],
}
with open('tests/fixtures/whisper_mel_oracle.json', 'w') as f:
    json.dump(oracle, f)
print('wrote whisper_mel_oracle.json '
      f'(global max {oracle["global_max"]:.6f}, '
      f'tail {oracle["padding_tail_value"]:.6f})')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
feats = torch.tensor([mel_inputs[0]], dtype=torch.float64)
dec_ids = torch.tensor([dec_sequences[0]])
GATE = 1e-4  # the Pascal parity gate; every quirk must move more than it
with torch.no_grad():
    base = model(input_features=feats, decoder_input_ids=dec_ids).logits
    base_eh = enc(input_features=feats).last_hidden_state

    # 1. The dropped encoder sinusoid table (regenerated by the importer)
    #    must matter.
    m2 = copy.deepcopy(model)
    m2.model.encoder.embed_positions.weight.zero_()
    d = (m2.model.encoder(input_features=feats).last_hidden_state -
         base_eh).abs().max().item()
    assert d > 1e-2, f'encoder positions had no effect ({d})'

    # 2. EXACT erf GELU vs the tanh approximation (gelu_new): using
    #    TNNetGELU would fail the gate.
    m2 = copy.deepcopy(model)
    m2.config.activation_function = 'gelu_new'
    import torch.nn as nn
    import torch.nn.functional as F

    class TanhGelu(nn.Module):
        def forward(self, x):
            return F.gelu(x, approximate='tanh')
    for layer in list(m2.model.encoder.layers) + list(
            m2.model.decoder.layers):
        layer.activation_fn = TanhGelu()
    d = (m2(input_features=feats, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 2 * GATE, f'erf vs tanh GELU indistinguishable ({d})'

    # 3. The conv GELUs are exact-erf too (the frontend path, before the
    #    transformer blocks - modeling code calls nn.functional.gelu).
    m2 = copy.deepcopy(model)
    conv1, conv2 = m2.model.encoder.conv1, m2.model.encoder.conv2

    def tanh_frontend(input_features):
        x = F.gelu(conv1(input_features), approximate='tanh')
        x = F.gelu(conv2(x), approximate='tanh')
        return x
    eh_tanh = None  # only check the conv output difference directly
    x_exact = F.gelu(conv2(F.gelu(conv1(feats))))
    d = (tanh_frontend(feats) - x_exact).abs().max().item()
    assert d > GATE, f'conv GELU exactness invisible ({d})'

    # 4. Learned decoder positions matter.
    m2 = copy.deepcopy(model)
    m2.model.decoder.embed_positions.weight.zero_()
    d = (m2(input_features=feats, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-2, f'decoder positions had no effect ({d})'

    # 5. Cross-attention: the mel input must reach the logits.
    feats2 = feats.clone()
    feats2[0, :, :5] += 0.5
    d = (model(input_features=feats2, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-2, f'mel input had no effect on logits ({d})'

    # 6. Decoder self-attention causality: a change in the LAST decoder
    #    token must not move earlier positions' logits.
    dec2 = dec_ids.clone()
    dec2[0, -1] = (int(dec2[0, -1]) + 1) % VOCAB
    out2 = model(input_features=feats, decoder_input_ids=dec2).logits
    d_early = (out2[:, :-1] - base[:, :-1]).abs().max().item()
    assert d_early < 1e-9, f'decoder attention is not causal ({d_early})'

    # 7. The stride-2 conv halves the frames: encoder hidden length is
    #    MAX_SRC for a 2*MAX_SRC-frame input (rectangular cross-attention
    #    MAX_SRC x DEC_LEN - asserted again on the Pascal side).
    assert base_eh.shape == (1, MAX_SRC, D_MODEL), base_eh.shape

    # 8. The final stack norms matter (pre-norm + final norm wiring).
    m2 = copy.deepcopy(model)
    m2.model.encoder.layer_norm.bias += 1.0
    d = (m2.model.encoder(input_features=feats).last_hidden_state -
         base_eh - 1.0).abs().max().item()
    assert d < 1e-9, f'encoder final norm is not the last op ({d})'

    # 9. k_proj bias ABSENT but q/v/out biases nonzero (loaded, not
    #    defaulted).
    for layer in list(model.model.encoder.layers) + list(
            model.model.decoder.layers):
        attns = [layer.self_attn]
        if hasattr(layer, 'encoder_attn'):
            attns.append(layer.encoder_attn)
        for attn in attns:
            for proj in (attn.q_proj, attn.v_proj, attn.out_proj):
                assert proj.bias.abs().max().item() > 1e-3, 'zero bias'
print('all fixture self-checks passed')
