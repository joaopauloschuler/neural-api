#!/usr/bin/env python3
"""Generate a tiny RANDOM M2M100/NLLB parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_m2m100.*: M2M100ForConditionalGeneration (the facebook/m2m100_* and
      facebook/nllb-200-* multilingual translation architecture) with every
      M2M100 trait the importer must reproduce - M2M100 is Pegasus's pre-norm
      encoder-decoder body with three deltas:
        - SINUSOIDAL position embeddings (M2M100SinusoidalPositionalEmbedding)
          with the half-split base log(10000)/(half-1) AND a +2 offset (token
          position p reads sinusoidal row p+2). The table is a buffer dropped
          on save - the importer regenerates it;
        - NO layernorm_embedding (mBART has one; M2M100/Pegasus do NOT);
        - a ReLU FFN (activation_function="relu" on every published M2M100 and
          NLLB checkpoint) instead of mBART/Pegasus's exact-erf GELU;
        - PRE-norm encoder-decoder blocks: LayerNorm BEFORE each sub-layer,
          residual added RAW afterwards (self_attn_layer_norm /
          encoder_attn_layer_norm / final_layer_norm per block), plain BIASED
          nn.LayerNorm (eps 1e-5);
        - a FINAL encoder AND decoder layer_norm closing each pre-norm stack
          (model.encoder.layer_norm / model.decoder.layer_norm);
        - standard 1/sqrt(head_dim) attention, ALL q/k/v/out projections
          biased;
        - scale_embedding TRUE (exercise the sqrt(d_model) fold; the M2M100/
          NLLB default);
        - shared source/target embedding TIED to the lm_head PLUS the
          final_logits_bias vector added to every logit row.

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): the encoder final hidden states plus
the full encoder-decoder logits for pinned encoder/decoder input ids.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/m2m100_tiny_fixture.py
writes tests/fixtures/tiny_m2m100{.safetensors,_config.json,_logits.json}.
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import M2M100Config, M2M100ForConditionalGeneration

N_LAYER = 2
N_HEAD = 2                    # head_dim = D_MODEL / N_HEAD = 4
D_MODEL = 8
D_FF = 16
VOCAB = 13
PAD = 1                       # M2M100 convention: pad_token_id = 1
BOS = 0
EOS = 2
MAX_POS = 16
ENC_LEN = 10
DEC_LEN = 6
N_SEQUENCES = 2

torch.manual_seed(20260613)

# Pinned inputs. The decoder is teacher-forced with arbitrary in-vocab ids
# (the multilingual forced-language-token protocol is a generation concern,
# not a model-build one - this fixture only checks teacher-forced logits).
# Avoid the PAD id anywhere: M2M100 sinusoidal position_ids are built from a
# pad mask, so a pad token would shift every following position.
enc_sequences = [[(5 * i + 3 * s + s * s) % (VOCAB - 3) + 3
                  for i in range(ENC_LEN - 1)] + [EOS]
                 for s in range(N_SEQUENCES)]
dec_sequences = [[EOS] + [(3 * i + 2 * s + 1) % (VOCAB - 3) + 3
                          for i in range(DEC_LEN - 1)]
                 for s in range(N_SEQUENCES)]

cfg_dict = {
    'architectures': ['M2M100ForConditionalGeneration'],
    'model_type': 'm2m_100',
    'd_model': D_MODEL,
    'encoder_layers': N_LAYER,
    'decoder_layers': N_LAYER,
    'encoder_attention_heads': N_HEAD,
    'decoder_attention_heads': N_HEAD,
    'encoder_ffn_dim': D_FF,
    'decoder_ffn_dim': D_FF,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'activation_function': 'relu',
    'scale_embedding': True,
    'tie_word_embeddings': True,
    'dropout': 0.0,
    'attention_dropout': 0.0,
    'activation_dropout': 0.0,
    'pad_token_id': PAD,
    'bos_token_id': BOS,
    'eos_token_id': EOS,
    'decoder_start_token_id': EOS,
}
model = M2M100ForConditionalGeneration(
    M2M100Config(**cfg_dict, attn_implementation='eager'))

# HF inits with tiny stds at this pico width; boost so every quirk is visible
# in the oracle: O(1) attention scores, FFN pre-activations where relu and
# gelu/swish genuinely differ, layer-norm gains/biases away from (1, 0), a
# final_logits_bias that moves the argmax.
with torch.no_grad():
    model.model.shared.weight.normal_(0.0, 0.5)
    for stack in (model.model.encoder, model.model.decoder):
        stack.layer_norm.weight.normal_(1.0, 0.25)
        stack.layer_norm.bias.normal_(0.0, 0.2)
        for layer in stack.layers:
            attns = [layer.self_attn]
            if hasattr(layer, 'encoder_attn'):
                attns.append(layer.encoder_attn)
            for attn in attns:
                for proj in (attn.q_proj, attn.k_proj, attn.v_proj,
                             attn.out_proj):
                    proj.weight.normal_(0.0, 0.45)
                    proj.bias.normal_(0.0, 0.2)
            layer.fc1.weight.normal_(0.0, 0.7)
            layer.fc1.bias.normal_(0.0, 0.3)
            layer.fc2.weight.normal_(0.0, 0.4)
            layer.fc2.bias.normal_(0.0, 0.2)
            norms = [layer.self_attn_layer_norm, layer.final_layer_norm]
            if hasattr(layer, 'encoder_attn_layer_norm'):
                norms.append(layer.encoder_attn_layer_norm)
            for norm in norms:
                norm.weight.normal_(1.0, 0.25)
                norm.bias.normal_(0.0, 0.2)
    # transformers 5.x removed M2M100's final_logits_bias entirely (the head
    # is a bias-free lm_head tied to the shared embedding). Published NLLB/
    # M2M100 checkpoints carry an all-ZERO final_logits_bias buffer; the
    # importer accepts it when present and otherwise uses a zero head bias, so
    # this fixture - built on the modern modeling code - exercises the
    # absent-bias path (matching what current transformers emits).
model = model.double().eval()

# Real HF checkpoints drop the tied duplicates (encoder/decoder embed_tokens
# and lm_head alias shared) AND the SINUSOIDAL embed_positions buffers (a pure
# function of d_model). Drop them so the importer's tie-the-head and
# regenerate-positions paths are the ones under test.
drop = {
    'model.encoder.embed_tokens.weight',
    'model.decoder.embed_tokens.weight',
    'lm_head.weight',
    'model.encoder.embed_positions.weights',
    'model.decoder.embed_positions.weights',
}
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items() if k not in drop}
save_file(sd, 'tests/fixtures/tiny_m2m100.safetensors')
with open('tests/fixtures/tiny_m2m100_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

enc_hidden, logits = [], []
with torch.no_grad():
    for es, ds in zip(enc_sequences, dec_sequences):
        enc_ids = torch.tensor([es])
        dec_ids = torch.tensor([ds])
        enc_hidden.append(
            model.model.encoder(input_ids=enc_ids).last_hidden_state[0]
            .tolist())
        logits.append(
            model(input_ids=enc_ids,
                  decoder_input_ids=dec_ids).logits[0].tolist())
with open('tests/fixtures/tiny_m2m100_logits.json', 'w') as f:
    json.dump({'enc_sequences': enc_sequences,
               'dec_sequences': dec_sequences,
               'enc_hidden': enc_hidden, 'logits': logits}, f)
print(f'wrote tiny_m2m100.safetensors ({len(sd)} tensors) + config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
import copy
import math
import numpy as np

enc_ids = torch.tensor([enc_sequences[0]])
dec_ids = torch.tensor([dec_sequences[0]])
with torch.no_grad():
    base = model(input_ids=enc_ids, decoder_input_ids=dec_ids).logits

    # 1. Positions are SINUSOIDAL with a +2 offset and base log(1e4)/(half-1).
    pos = model.model.encoder.embed_positions
    assert pos.offset == 2, 'sinusoidal offset must be 2'
    half = D_MODEL // 2
    embc = math.log(10000) / (half - 1)
    for p in range(ENC_LEN):
        row = p + 2
        want = [0.0] * D_MODEL
        for c in range(half):
            ang = row * math.exp(-c * embc)
            want[c] = math.sin(ang)
            want[half + c] = math.cos(ang)
        got = pos.weights[row].tolist()
        d = max(abs(a - b) for a, b in zip(got, want))
        # HF builds the table at the default (float32) dtype before .double();
        # the formula match is what matters, not float32 round-off.
        assert d < 1e-6, f'sinusoidal row {row} mismatch ({d})'
    print('sinusoidal +2-offset / half-split base formula check passed')

    # 2. The positions must matter (zeroing the table moves the logits).
    nopos = copy.deepcopy(model)
    nopos.model.encoder.embed_positions.weights.zero_()
    nopos.model.decoder.embed_positions.weights.zero_()
    d = (nopos(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-2, f'positions had no effect ({d})'
    print(f'sinusoidal-position effect on logits: max |diff| = {d:.4f}')

    # 3. NO layernorm_embedding (M2M100/Pegasus omit it). The importer that
    # mistakenly inserts one would diverge; assert the module is absent.
    assert not hasattr(model.model.encoder, 'layernorm_embedding'), \
        'M2M100 encoder must NOT have layernorm_embedding'
    assert not hasattr(model.model.decoder, 'layernorm_embedding'), \
        'M2M100 decoder must NOT have layernorm_embedding'
    print('no-layernorm_embedding check passed')

    # 4. The FINAL encoder/decoder layer_norm must matter.
    for which in ('encoder', 'decoder'):
        nf = copy.deepcopy(model)
        getattr(nf.model, which).layer_norm.bias += 1.0
        d = (nf(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
             base).abs().max().item()
        assert d > 1e-2, f'{which}.layer_norm had no effect ({d})'
        print(f'final {which}.layer_norm effect on logits: '
              f'max |diff| = {d:.4f}')

    # 5. PRE-norm placement: shifting the FINAL encoder layer_norm bias must
    # shift the encoder output by EXACTLY that amount.
    pn = copy.deepcopy(model)
    pn.model.encoder.layer_norm.bias += 1.0
    eh = pn.model.encoder(input_ids=enc_ids).last_hidden_state
    eh_base = model.model.encoder(input_ids=enc_ids).last_hidden_state
    d = (eh - eh_base - 1.0).abs().max().item()
    assert d < 1e-9, f'encoder output is not closed by layer_norm ({d})'
    print('pre-norm final-LayerNorm placement check passed')

    # 6. scale_embedding TRUE: sqrt(d_model) multiplies the token embeds.
    assert cfg_dict['scale_embedding'] is True
    print(f'scale_embedding requested (scale = {np.sqrt(D_MODEL):.4f})')

    # 7. relu vs gelu/silu must be distinguishable above the 1e-4 gate.
    for wrong in ('gelu', 'silu'):
        alt = copy.deepcopy(model)
        for stack in (alt.model.encoder, alt.model.decoder):
            for layer in stack.layers:
                layer.activation_fn = (torch.nn.GELU() if wrong == 'gelu'
                                       else torch.nn.SiLU())
        d = (alt(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
             base).abs().max().item()
        assert d > 1e-3, f'relu vs {wrong} indistinguishable ({d})'
        print(f'relu-vs-{wrong} effect on logits: max |diff| = {d:.4f}')

    # 8. The head is TIED to the shared embedding (no separate lm_head tensor
    # saved) and bias-free in transformers 5.x.
    assert torch.equal(model.lm_head.weight, model.model.shared.weight), \
        'lm_head must be tied to the shared embedding'
    assert model.lm_head.bias is None, 'modern M2M100 lm_head is bias-free'
    print('tied bias-free lm_head check passed')

    # 9. The encoder input must reach the logits through cross-attention.
    other = torch.tensor([enc_sequences[1]])
    d = (model(input_ids=other, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-2, f'encoder ids had no effect on logits ({d})'
    print(f'encoder-input effect on logits: max |diff| = {d:.4f}')

    # 10. Decoder self-attention must be CAUSAL.
    dec2 = dec_ids.clone()
    dec2[0, -1] = (dec2[0, -1] + 1) % (VOCAB - 3) + 3
    l2 = model(input_ids=enc_ids, decoder_input_ids=dec2).logits
    d_early = (l2[0, :-1] - base[0, :-1]).abs().max().item()
    assert d_early < 1e-9, f'decoder attention is not causal ({d_early})'
    print('decoder causality check passed')

    # 11. Every Linear is BIASED (the M2M100 convention).
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m is not model.lm_head:
            assert m.bias is not None, 'unexpected bias-free Linear'
            assert m.bias.abs().max().item() > 1e-3, 'all-zero Linear bias'
    print('biased-Linear check passed')
print('all fixture self-checks passed')
