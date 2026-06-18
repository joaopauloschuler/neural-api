#!/usr/bin/env python3
"""Generate a tiny RANDOM CLAP parity fixture with freq_ratio = 4 for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

Sibling of tools/clap_tiny_fixture.py (which pins freq_ratio = 1). This one
exercises the REAL laion layout where spec_size = freq_ratio * num_mel_bins
(e.g. the 1024-frame / 64-mel / freq_ratio = 4 checkpoint), at pico scale:

  tiny_clap_fr4.*: ClapModel with num_mel_bins = SPEC // 4 so the encoder's
      reshape_mel2img genuinely folds the long time axis into freq_ratio
      chunks stacked along frequency (NOT the freq_ratio = 1 transpose), and
      the group-2D-CNN reshape before the avgpool is a real permutation.
      Everything else is identical to the freq_ratio = 1 fixture.

The input log-mel is (1, 1, time = spec_size * freq_ratio, num_mel_bins); the
importer-facing audio image is the batch-normed + reshape_mel2img'd square
(spec_size, spec_size) feature the AUDIO net takes as its Input (a fixed
affine the caller folds in via ClapBatchNormMelImage).

The reference is computed by HF transformers in float64.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/clap_tiny_fixture_fr4.py
writes tests/fixtures/tiny_clap_fr4{.safetensors,_config.json,_io.json}.
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import ClapConfig, ClapModel

# ---- pico config (freq_ratio = spec_size // num_mel_bins = 4) ----
TXT_HIDDEN = 16
TXT_INTER = 32
TXT_LAYERS = 1
TXT_HEADS = 2
VOCAB = 40
MAX_POS = 24
PAD_ID = 1
SEQ_LEN = 8

FREQ_RATIO = 4
MEL = 8             # num_mel_bins
SPEC = MEL * FREQ_RATIO    # spec_size = 32 -> freq_ratio = 4
TIME = SPEC * FREQ_RATIO   # the long time axis fed to reshape_mel2img (128)
PATCH = 2           # patch_size / stride -> 16x16 grid
EMBED = 8           # patch_embeds_hidden_size (stage-0 dim)
DEPTHS = [2, 2]
AHEADS = [2, 4]
WINDOW = 2
AUD_HIDDEN = EMBED * 2 ** (len(DEPTHS) - 1)   # 16, the num_features
PROJ = 12
N_TEXTS = 3

torch.manual_seed(20260616)

cfg = ClapConfig(
    text_config=dict(
        hidden_size=TXT_HIDDEN, intermediate_size=TXT_INTER,
        num_hidden_layers=TXT_LAYERS, num_attention_heads=TXT_HEADS,
        max_position_embeddings=MAX_POS, vocab_size=VOCAB,
        type_vocab_size=1, layer_norm_eps=1e-12, pad_token_id=PAD_ID,
        hidden_act='gelu', projection_hidden_act='relu',
    ),
    audio_config=dict(
        patch_embeds_hidden_size=EMBED, depths=DEPTHS,
        num_attention_heads=AHEADS, num_mel_bins=MEL, spec_size=SPEC,
        window_size=WINDOW, patch_size=PATCH, patch_stride=[PATCH, PATCH],
        hidden_size=AUD_HIDDEN, num_classes=10, enable_fusion=False,
        layer_norm_eps=1e-5, hidden_act='gelu', projection_hidden_act='relu',
    ),
    projection_dim=PROJ,
)
model = ClapModel(cfg)

# HF inits with tiny stds at pico width; boost so every quirk is visible in
# the float64 oracle above the 1e-4 parity gate.
with torch.no_grad():
    ae = model.audio_model.audio_encoder
    assert ae.freq_ratio == FREQ_RATIO, 'fixture assumes freq_ratio == 4'
    ae.patch_embed.proj.weight.normal_(0.0, 0.3)
    ae.patch_embed.proj.bias.normal_(0.0, 0.2)
    ae.patch_embed.norm.weight.normal_(1.0, 0.25)
    ae.patch_embed.norm.bias.normal_(0.0, 0.2)
    ae.batch_norm.weight.normal_(1.0, 0.25)
    ae.batch_norm.bias.normal_(0.0, 0.2)
    ae.batch_norm.running_mean.normal_(0.0, 0.3)
    ae.batch_norm.running_var.uniform_(0.5, 1.5)
    ae.norm.weight.normal_(1.0, 0.25)
    ae.norm.bias.normal_(0.0, 0.2)
    for layer in ae.layers:
        for blk in layer.blocks:
            blk.attention.self.relative_position_bias_table.normal_(0.0, 0.4)
            for lin in (blk.attention.self.query, blk.attention.self.key,
                        blk.attention.self.value, blk.attention.output.dense,
                        blk.intermediate.dense, blk.output.dense):
                lin.weight.normal_(0.0, 0.35)
                lin.bias.normal_(0.0, 0.2)
            for norm in (blk.layernorm_before, blk.layernorm_after):
                norm.weight.normal_(1.0, 0.25)
                norm.bias.normal_(0.0, 0.2)
        if layer.downsample is not None:
            layer.downsample.reduction.weight.normal_(0.0, 0.3)
            layer.downsample.norm.weight.normal_(1.0, 0.25)
            layer.downsample.norm.bias.normal_(0.0, 0.2)

    tm = model.text_model
    tm.embeddings.word_embeddings.weight.normal_(0.0, 0.5)
    tm.embeddings.position_embeddings.weight.normal_(0.0, 0.4)
    tm.embeddings.token_type_embeddings.weight.normal_(0.0, 0.3)
    tm.embeddings.LayerNorm.weight.normal_(1.0, 0.25)
    tm.embeddings.LayerNorm.bias.normal_(0.0, 0.2)
    for layer in tm.encoder.layer:
        for lin in (layer.attention.self.query, layer.attention.self.key,
                    layer.attention.self.value, layer.attention.output.dense,
                    layer.intermediate.dense, layer.output.dense):
            lin.weight.normal_(0.0, 0.35)
            lin.bias.normal_(0.0, 0.2)
        for norm in (layer.attention.output.LayerNorm,
                     layer.output.LayerNorm):
            norm.weight.normal_(1.0, 0.25)
            norm.bias.normal_(0.0, 0.2)
    tm.pooler.dense.weight.normal_(0.0, 0.35)
    tm.pooler.dense.bias.normal_(0.0, 0.2)

    for proj in (model.audio_projection, model.text_projection):
        proj.linear1.weight.normal_(0.0, 0.35)
        proj.linear1.bias.normal_(0.0, 0.2)
        proj.linear2.weight.normal_(0.0, 0.35)
        proj.linear2.bias.normal_(0.0, 0.2)
    model.logit_scale_a.fill_(2.0)
    model.logit_scale_t.fill_(2.0)

model = model.double().eval()

# ---- pinned inputs ----
# audio: a deterministic log-mel image (1, 1, T = SPEC*freq_ratio, F = MEL).
mel = torch.zeros(1, 1, TIME, MEL, dtype=torch.float64)
for t in range(TIME):
    for f in range(MEL):
        mel[0, 0, t, f] = (((t * 31 + f * 7) % 23) - 11) / 11.0
# text: 3 token sequences. NO pad token (id == PAD_ID) anywhere.
text_sequences = [
    [5, 12, 23, 7, 31, 2, 9, 38],
    [8, 19, 3, 27, 14, 6, 22, 30],
    [11, 4, 33, 21, 16, 28, 17, 9],
]
assert all(PAD_ID not in s for s in text_sequences), 'no pad tokens allowed'

with torch.no_grad():
    feats_in = mel
    a = model.get_audio_features(input_features=feats_in)
    audio_embeds = a if torch.is_tensor(a) else a.pooler_output
    ids = torch.tensor(text_sequences)
    t = model.get_text_features(input_ids=ids)
    text_embeds = t if torch.is_tensor(t) else t.pooler_output
    audio_n = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
    text_n = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    out = model(input_ids=ids, input_features=feats_in)

    # the importer-facing audio image: batch_norm over mel + reshape_mel2img,
    # the fixed affine the caller folds in before the net. Shape (SPEC, SPEC).
    bn = ae.batch_norm(feats_in.transpose(1, 3)).transpose(1, 3)
    aud_img = ae.reshape_mel2img(bn)[0, 0]    # (H = freq*fr, W = time//fr)

# ---- dump safetensors (drop buffers the importer reconstructs / ignores) ----
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()
      if not k.endswith('position_ids')
      and not k.endswith('num_batches_tracked')}
save_file(sd, 'tests/fixtures/tiny_clap_fr4.safetensors')
with open('tests/fixtures/tiny_clap_fr4_config.json', 'w') as f:
    json.dump(cfg.to_dict(), f, indent=1, default=str)
with open('tests/fixtures/tiny_clap_fr4_io.json', 'w') as f:
    json.dump({
        'mel': mel[0, 0].tolist(),               # (TIME, MEL) raw log-mel
        'audio_image': aud_img.tolist(),          # (SPEC, SPEC) reshape_mel2img
        'text_sequences': text_sequences,
        'audio_embeds': audio_n.tolist(),         # [1][PROJ] L2-normalized
        'text_embeds': text_n.tolist(),           # [N_TEXTS][PROJ] L2-normalized
        'logit_scale_a': model.logit_scale_a.item(),
        'logit_scale_t': model.logit_scale_t.item(),
        'logits_per_audio': out.logits_per_audio.tolist(),  # [1][N_TEXTS]
    }, f)
print(f'wrote tiny_clap_fr4.safetensors ({len(sd)} tensors) + config + oracle')

# ---- self-checks: freq_ratio = 4 is genuinely exercised ----
with torch.no_grad():
    assert ae.freq_ratio == 4, 'fixture assumes freq_ratio == 4'
    # 1. logits_per_audio == exp(logit_scale_a) * normalized cosine
    ref = model.logit_scale_a.exp() * audio_n @ text_n.T
    assert (ref - out.logits_per_audio).abs().max() < 1e-9, 'logit_scale_a head'
    # 2. mel2img folds the long (TIME) axis into a square SPEC x SPEC image
    #    (NOT the freq_ratio = 1 transpose, whose shape would be (MEL, TIME)).
    assert aud_img.shape == (SPEC, SPEC), f'image shape {aud_img.shape}'
    # 3. verify the exact index mapping the Pascal importer uses:
    #    Image[W][H] = bn[ t = (H//MEL)*SPEC + W ][ m = H%MEL ].
    for ww in range(SPEC):
        for hh in range(SPEC):
            m = hh % MEL
            c2 = hh // MEL
            srcT = c2 * SPEC + ww
            assert abs(aud_img[hh, ww].item() - bn[0, 0, srcT, m].item()) < 1e-9, \
                f'index mapping mismatch at ({ww},{hh})'
    print('freq_ratio=4 mel2img index-mapping + logit-scale checks passed')
print('all fixture self-checks passed')
