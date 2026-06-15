#!/usr/bin/env python3
"""Generate a tiny RANDOM TrOCR optical-character-recognition parity fixture
for tests/TestNeuralPretrained.pas.

No network access: a pico VisionEncoderDecoderModel (a tiny DeiT image encoder
+ a TrOCR/BART-style causal text decoder that cross-attends to ALL of the image
patch tokens + a tied LM head) is built from a small config and randomly
initialized (never downloaded). The reference forward is the REAL transformers
VisionEncoderDecoderModel run in float64 (the package is installed in this
environment); the test asserts the Pascal forward matches the per-position
next-token `logits` over the vocabulary for a fixed image + decoder prefix.

This is the repo's FIRST OCR / image-to-text importer: a cropped text-line
image -> a transcribed string. Structurally an encoder-decoder seq2seq with a
VISION encoder (the DeiT/ViT primitive set: patch conv WITH bias + a class
token AND a distillation token + learned positions + PRE-LN encoder blocks +
final layernorm) feeding a BART-style POST-LN causal text decoder through
CROSS-ATTENTION (TNNetCrossAttention). The decoder cross-attends to the FULL
encoder last_hidden_state (cls + distillation + every patch token; NO pooling).
It reuses the two-net (encoder + cross-attending decoder) convention of the
T5/Marian/Pegasus/BLIP importers and the DecodeSeq2Seq* helpers.

TrOCR decoder traits the importer must reproduce:
  - learned ABSOLUTE position embeddings with BART's +2 padding offset
    (max_position_embeddings + 2 rows; token position p reads row p + 2);
  - a layernorm_embedding after token+position embeddings;
  - POST-norm blocks (residual add THEN biased nn.LayerNorm, eps 1e-5);
  - exact-erf GELU FFN; all q/k/v/out/fc Linears biased;
  - scale_embedding (sqrt(d_model) on the token embeddings);
  - the output_projection tied to embed_tokens, NO final_logits_bias.

The DeiT encoder traits:
  - patch conv WITH bias (unlike CLIP's bias-free conv);
  - cls_token AND distillation_token prepended (position_embeddings has
    num_patches + 2 rows); both folded into the learned position table;
  - PRE-norm blocks (layernorm_before / layernorm_after), q/k/v/o_proj,
    mlp.fc1/fc2, exact-erf GELU, then a FINAL layernorm over all tokens.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_trocr_fixture.py
writes tests/fixtures/tiny_trocr{.safetensors,_config.json,_io.json}.
"""
import json

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import (VisionEncoderDecoderModel, VisionEncoderDecoderConfig,
                          DeiTConfig, TrOCRConfig)

# ---------------- pico config ----------------
IMAGE = 16
PATCH = 8                 # 2x2 = 4 patches (+ cls + dist = 6 encoder tokens)
NUM_CHANNELS = 3
ENC_HIDDEN = 24
ENC_INTER = 48
ENC_LAYERS = 2
ENC_HEADS = 3
DEC_HIDDEN = 24           # d_model (== encoder hidden so cross-attn k/v fit)
DEC_FFN = 48
DEC_LAYERS = 2
DEC_HEADS = 3
VOCAB = 40
MAX_POS = 32
BOS = 0
EOS = 2
PAD = 1
DEC_START = 2
DEC_LEN = 5

torch.manual_seed(20260615)
np.random.seed(20260615)

enc_cfg = DeiTConfig(
    image_size=IMAGE, patch_size=PATCH, num_channels=NUM_CHANNELS,
    hidden_size=ENC_HIDDEN, num_hidden_layers=ENC_LAYERS,
    num_attention_heads=ENC_HEADS, intermediate_size=ENC_INTER,
    hidden_act='gelu', layer_norm_eps=1e-12, qkv_bias=True)
dec_cfg = TrOCRConfig(
    vocab_size=VOCAB, d_model=DEC_HIDDEN, decoder_layers=DEC_LAYERS,
    decoder_attention_heads=DEC_HEADS, decoder_ffn_dim=DEC_FFN,
    max_position_embeddings=MAX_POS, activation_function='gelu',
    scale_embedding=True, tie_word_embeddings=True,
    bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD,
    decoder_start_token_id=DEC_START)
cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc_cfg, dec_cfg)
model = VisionEncoderDecoderModel(cfg)

# HF inits with tiny stds at this pico width; boost so every quirk is visible
# in the oracle (O(1) attention scores, FFN pre-activations where gelu/relu
# differ, layer-norm gains/biases away from (1,0)).
with torch.no_grad():
    enc = model.encoder
    enc.embeddings.cls_token.normal_(0.0, 0.5)
    enc.embeddings.distillation_token.normal_(0.0, 0.5)
    enc.embeddings.position_embeddings.normal_(0.0, 0.4)
    enc.embeddings.patch_embeddings.projection.weight.normal_(0.0, 0.4)
    enc.embeddings.patch_embeddings.projection.bias.normal_(0.0, 0.2)
    for layer in enc.layers:
        for proj in (layer.attention.q_proj, layer.attention.k_proj,
                     layer.attention.v_proj, layer.attention.o_proj):
            proj.weight.normal_(0.0, 0.45)
            proj.bias.normal_(0.0, 0.2)
        layer.mlp.fc1.weight.normal_(0.0, 0.7)
        layer.mlp.fc1.bias.normal_(0.0, 0.3)
        layer.mlp.fc2.weight.normal_(0.0, 0.4)
        layer.mlp.fc2.bias.normal_(0.0, 0.2)
        for norm in (layer.layernorm_before, layer.layernorm_after):
            norm.weight.normal_(1.0, 0.25)
            norm.bias.normal_(0.0, 0.2)
    enc.layernorm.weight.normal_(1.0, 0.25)
    enc.layernorm.bias.normal_(0.0, 0.2)

    dec = model.decoder.model.decoder
    dec.embed_tokens.weight.normal_(0.0, 0.5)
    dec.layernorm_embedding.weight.normal_(1.0, 0.25)
    dec.layernorm_embedding.bias.normal_(0.0, 0.2)
    for layer in dec.layers:
        for attn in (layer.self_attn, layer.encoder_attn):
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
                proj.weight.normal_(0.0, 0.45)
                proj.bias.normal_(0.0, 0.2)
        layer.fc1.weight.normal_(0.0, 0.7)
        layer.fc1.bias.normal_(0.0, 0.3)
        layer.fc2.weight.normal_(0.0, 0.4)
        layer.fc2.bias.normal_(0.0, 0.2)
        for norm in (layer.self_attn_layer_norm,
                     layer.encoder_attn_layer_norm, layer.final_layer_norm):
            norm.weight.normal_(1.0, 0.25)
            norm.bias.normal_(0.0, 0.2)
    # keep output_projection tied to the (now-randomized) embed_tokens
    model.decoder.output_projection.weight.copy_(dec.embed_tokens.weight)

model = model.double().eval()

# ---------------- pinned inputs ----------------
pixel_values = torch.randn(1, NUM_CHANNELS, IMAGE, IMAGE, dtype=torch.float64)
dec_ids = [DEC_START] + [(7 * i + 3) % (VOCAB - 3) + 3 for i in range(DEC_LEN - 1)]
dec_tensor = torch.tensor([dec_ids])

with torch.no_grad():
    enc_hidden = model.encoder(
        pixel_values=pixel_values).last_hidden_state[0]
    logits = model(pixel_values=pixel_values,
                   decoder_input_ids=dec_tensor).logits[0]

# ---------------- serialize ----------------
sd_full = model.state_dict()
# Real HF VisionEncoderDecoder checkpoints drop the tied duplicate
# (output_projection aliases embed_tokens via tie_word_embeddings). Keep both
# here so the importer's tie path has the table to read from embed_tokens.
drop = set()
sd = {}
for k, v in sd_full.items():
    if k in drop:
        continue
    sd[k] = v.to(torch.float32).clone().contiguous().numpy()
save_file(sd, 'tests/fixtures/tiny_trocr.safetensors')

cfg_dict = {
    'model_type': 'vision-encoder-decoder',
    'encoder': {
        'model_type': 'deit',
        'image_size': IMAGE, 'patch_size': PATCH,
        'num_channels': NUM_CHANNELS, 'hidden_size': ENC_HIDDEN,
        'num_hidden_layers': ENC_LAYERS,
        'num_attention_heads': ENC_HEADS,
        'intermediate_size': ENC_INTER, 'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,
    },
    'decoder': {
        'model_type': 'trocr',
        'vocab_size': VOCAB, 'd_model': DEC_HIDDEN,
        'decoder_layers': DEC_LAYERS,
        'decoder_attention_heads': DEC_HEADS,
        'decoder_ffn_dim': DEC_FFN,
        'max_position_embeddings': MAX_POS,
        'activation_function': 'gelu', 'scale_embedding': True,
        'tie_word_embeddings': True,
        'bos_token_id': BOS, 'eos_token_id': EOS, 'pad_token_id': PAD,
        'decoder_start_token_id': DEC_START,
    },
}
with open('tests/fixtures/tiny_trocr_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

with open('tests/fixtures/tiny_trocr_io.json', 'w') as f:
    json.dump({
        'pixel_values': pixel_values[0].tolist(),  # (C, H, W)
        'dec_ids': dec_ids,
        'enc_hidden': enc_hidden.tolist(),          # (num_tokens, hidden)
        'logits': logits.tolist(),                  # (dec_len, vocab)
    }, f)

print(f'wrote tiny_trocr.safetensors ({len(sd)} tensors) + config + oracle')
print(f'  encoder tokens (cls+dist+patches): {enc_hidden.shape[0]}')
print(f'  logits shape: {list(logits.shape)}')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
import copy

with torch.no_grad():
    base = model(pixel_values=pixel_values,
                 decoder_input_ids=dec_tensor).logits

    # 1. The image must reach the logits through cross-attention.
    other = torch.randn(1, NUM_CHANNELS, IMAGE, IMAGE, dtype=torch.float64)
    d = (model(pixel_values=other,
               decoder_input_ids=dec_tensor).logits - base).abs().max().item()
    assert d > 1e-2, f'image had no effect on logits ({d})'
    print(f'image-input effect on logits: max |diff| = {d:.4f}')

    # 2. Decoder self-attention must be CAUSAL.
    dec2 = dec_tensor.clone()
    dec2[0, -1] = (dec2[0, -1] + 1) % (VOCAB - 3) + 3
    l2 = model(pixel_values=pixel_values, decoder_input_ids=dec2).logits
    d_early = (l2[0, :-1] - base[0, :-1]).abs().max().item()
    assert d_early < 1e-9, f'decoder attention is not causal ({d_early})'
    print('decoder causality check passed')

    # 3. The encoder emits cls + distillation + every patch token (no
    # pooling): num_patches + 2 = 6 cross-attendable tokens. A 1-CLS-token
    # or pooled importer would mis-shape the cross-attention K|V.
    n_patches = (IMAGE // PATCH) ** 2
    assert enc_hidden.shape[0] == n_patches + 2, \
        f'expected {n_patches + 2} encoder tokens, got {enc_hidden.shape[0]}'
    print(f'encoder emits {enc_hidden.shape[0]} tokens '
          f'(cls + distillation + {n_patches} patches), no pooling')

    # 4. The FINAL encoder layernorm must matter (a stack that forgets to
    # close with it FAILS).
    nf = copy.deepcopy(model)
    nf.encoder.layernorm.bias += 1.0
    d = (nf(pixel_values=pixel_values,
            decoder_input_ids=dec_tensor).logits - base).abs().max().item()
    assert d > 1e-2, f'encoder final layernorm had no effect ({d})'
    print(f'encoder-final-layernorm effect on logits: max |diff| = {d:.4f}')

    # 5. The layernorm_embedding must matter.
    nle = copy.deepcopy(model)
    nle.decoder.model.decoder.layernorm_embedding.bias += 1.0
    d = (nle(pixel_values=pixel_values,
             decoder_input_ids=dec_tensor).logits - base).abs().max().item()
    assert d > 1e-2, f'layernorm_embedding had no effect ({d})'
    print(f'layernorm_embedding effect on logits: max |diff| = {d:.4f}')

    # 6. Learned positions with the +2 offset must matter.
    npz = copy.deepcopy(model)
    npz.decoder.model.decoder.embed_positions.weight.zero_()
    d = (npz(pixel_values=pixel_values,
             decoder_input_ids=dec_tensor).logits - base).abs().max().item()
    assert d > 1e-2, f'decoder positions had no effect ({d})'
    print(f'decoder-position effect on logits: max |diff| = {d:.4f}')

    # 7. scale_embedding TRUE: sqrt(d_model) multiplies the token embeds.
    assert dec_cfg.scale_embedding, 'scale_embedding must be True'
    print(f'scale_embedding check passed (scale = {np.sqrt(DEC_HIDDEN):.4f})')

    # 8. output_projection tied to embed_tokens.
    op = sd_full['decoder.output_projection.weight']
    et = sd_full['decoder.model.decoder.embed_tokens.weight']
    assert torch.allclose(op, et), 'output_projection not tied'
    print('tied-head check passed')

    # 9. gelu vs relu must be distinguishable above the 1e-4 gate.
    alt = copy.deepcopy(model)
    for layer in alt.decoder.model.decoder.layers:
        layer.activation_fn = torch.nn.ReLU()
    d = (alt(pixel_values=pixel_values,
             decoder_input_ids=dec_tensor).logits - base).abs().max().item()
    assert d > 1e-3, f'gelu vs relu indistinguishable ({d})'
    print(f'gelu-vs-relu effect on logits: max |diff| = {d:.4f}')
print('all fixture self-checks passed')
