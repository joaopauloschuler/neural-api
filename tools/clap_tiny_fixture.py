#!/usr/bin/env python3
"""Generate a tiny RANDOM CLAP parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_clap.*: ClapModel (the laion/clap-htsat-unfused architecture) with
      every CLAP trait the importer must reproduce:
        - AUDIO tower = HTS-AT, a Swin hierarchical windowed-attention
          transformer over a log-mel spectrogram:
            * a BatchNorm2d over the mel-bin axis, then the "reshape mel to
              image" step (a freq<->time transpose at freq_ratio == 1),
            * a Conv2d patch-embed (kernel = stride = patch_size) + a
              LayerNorm over the patch tokens,
            * Swin STAGES (window/shifted-window MSA + MLP, the SAME blocks
              as the landed microsoft/swin importer but with the
              clap_audio_model key spelling attention.self.{query,key,value}
              / attention.output.dense / intermediate.dense / output.dense
              and attention.self.relative_position_bias_table), with patch
              merging between stages,
            * a final LayerNorm, the group-2D-CNN reshape (identity at
              freq_ratio == 1) and a mean-pool over the tokens,
            * the 2-layer ClapProjectionLayer (linear1 -> ReLU -> linear2).
        - TEXT tower = RoBERTa (clap_text_model): token + LEARNED absolute
          positions OFFSET past the pad id + a (single) token-type row,
          post-LN bidirectional encoder blocks, a BERT-style pooler
          (dense + tanh on token 0), then the 2-layer ClapProjectionLayer.
        - both embeds are L2-normalized; logits_per_audio =
          exp(logit_scale_a) * audio_n @ text_n^T.

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): audio_embeds / text_embeds are the
L2-NORMALIZED shared-space features (HF get_audio_features /
get_text_features then normalized), plus logits_per_audio for the pinned
pair.

freq_ratio is pinned to 1 (num_mel_bins == spec_size) so the mel2img
reshape is a plain freq<->time transpose and the final group-CNN reshape is
identity; the importer's audio net therefore takes the ALREADY
batch-normed + transposed mel image as its (F, T, 1) Input (a fixed affine
the caller applies up front, exactly as CLIP supplies normalized pixels).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/clap_tiny_fixture.py
writes tests/fixtures/tiny_clap{.safetensors,_config.json,_io.json}.
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import ClapConfig, ClapModel

# ---- pico config (freq_ratio = spec_size // num_mel_bins = 1) ----
TXT_HIDDEN = 16
TXT_INTER = 32
TXT_LAYERS = 1
TXT_HEADS = 2
VOCAB = 40
MAX_POS = 24
PAD_ID = 1
SEQ_LEN = 8

MEL = 16            # num_mel_bins
SPEC = 16           # spec_size  -> freq_ratio = 1
PATCH = 2           # patch_size / stride -> 8x8 grid
EMBED = 8           # patch_embeds_hidden_size (stage-0 dim)
DEPTHS = [2, 2]
AHEADS = [2, 4]
WINDOW = 2
AUD_HIDDEN = EMBED * 2 ** (len(DEPTHS) - 1)   # 16, the num_features
PROJ = 12
N_TEXTS = 3

torch.manual_seed(20260615)

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
# audio: a deterministic log-mel image (1, 1, T=SPEC, F=MEL)
mel = torch.zeros(1, 1, SPEC, MEL, dtype=torch.float64)
for t in range(SPEC):
    for f in range(MEL):
        mel[0, 0, t, f] = (((t * 31 + f * 7) % 23) - 11) / 11.0
# text: 3 token sequences. NO pad token (id == PAD_ID) anywhere - RoBERTa
# derives position ids by cumsum over the NON-pad mask (pads collapse to the
# padding position), which the importer's plain 0..N-1 offset scheme only
# reproduces when every position is a real token. A real deployment must
# left/right-trim to the unpadded length before calling the text net.
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

    # the importer-facing audio image: batch_norm over mel + mel2img
    # transpose, the fixed affine the caller applies before the net.
    bn = ae.batch_norm(feats_in.transpose(1, 3)).transpose(1, 3)
    aud_img = ae.reshape_mel2img(bn)[0, 0]    # (F, T) after the transpose

# ---- dump safetensors (drop buffers the importer reconstructs / ignores) ----
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()
      if not k.endswith('position_ids')
      and not k.endswith('num_batches_tracked')}
save_file(sd, 'tests/fixtures/tiny_clap.safetensors')
with open('tests/fixtures/tiny_clap_config.json', 'w') as f:
    json.dump(cfg.to_dict(), f, indent=1, default=str)
with open('tests/fixtures/tiny_clap_io.json', 'w') as f:
    json.dump({
        'mel': mel[0, 0].tolist(),               # (SPEC, MEL) raw log-mel
        'audio_image': aud_img.tolist(),          # (F, T) batch-normed+transposed
        'text_sequences': text_sequences,
        'audio_embeds': audio_n.tolist(),         # [1][PROJ] L2-normalized
        'text_embeds': text_n.tolist(),           # [N_TEXTS][PROJ] L2-normalized
        'logit_scale_a': model.logit_scale_a.item(),
        'logit_scale_t': model.logit_scale_t.item(),
        'logits_per_audio': out.logits_per_audio.tolist(),  # [1][N_TEXTS]
    }, f)
print(f'wrote tiny_clap.safetensors ({len(sd)} tensors) + config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- self-checks: every quirk visible in the oracle ----
with torch.no_grad():
    assert ae.freq_ratio == 1, 'fixture assumes freq_ratio == 1'
    # 1. logits_per_audio == exp(logit_scale_a) * normalized cosine
    ref = model.logit_scale_a.exp() * audio_n @ text_n.T
    assert (ref - out.logits_per_audio).abs().max() < 1e-9, 'logit_scale_a head'
    # 2. mel2img is the freq<->time transpose at freq_ratio 1
    assert torch.allclose(ae.reshape_mel2img(bn), bn.transpose(-1, -2)), \
        'mel2img not a transpose'
    # 3. text positions are OFFSET past the pad id (RoBERTa). Changing a
    #    non-pad token must move its sequence's embedding.
    ids2 = ids.clone(); ids2[0, 2] = (ids2[0, 2] + 5) % VOCAB
    t2 = model.get_text_features(input_ids=ids2)
    t2 = t2 if torch.is_tensor(t2) else t2.pooler_output
    d = (t2[0] - text_embeds[0]).abs().max()
    assert d > 1e-3, f'text token had no effect ({d})'
    print('logit-scale + mel2img-transpose + text-token-effect checks passed')
print('all fixture self-checks passed')
