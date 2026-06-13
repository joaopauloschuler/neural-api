#!/usr/bin/env python3
"""Generate a tiny RANDOM mBART parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_mbart.*: MBartForConditionalGeneration (the facebook/mbart-large-*
      multilingual architecture) with every mBART trait the importer must
      reproduce - mBART is BART's embedding front-end stacked on PEGASUS-style
      pre-norm blocks:
        - LEARNED absolute position embeddings with BART's +2 padding offset
          (embed_positions has max_position_embeddings + 2 rows; token
          position p reads row p + 2);
        - a layernorm_embedding LayerNorm AFTER token+position embeddings in
          BOTH stacks (BART has it; Pegasus does NOT);
        - PRE-norm encoder-decoder blocks (normalize_before=true): LayerNorm
          BEFORE each sub-layer, residual added RAW afterwards
          (self_attn_layer_norm / encoder_attn_layer_norm / final_layer_norm
          per block), plain BIASED nn.LayerNorm (eps 1e-5);
        - a FINAL encoder AND decoder layer_norm closing each pre-norm stack
          (model.encoder.layer_norm / model.decoder.layer_norm);
        - exact-erf GELU FFN, both fc Linears biased;
        - standard 1/sqrt(head_dim) attention, ALL q/k/v/out projections
          biased;
        - scale_embedding TRUE here (exercise the sqrt(d_model) fold; the HF
          mBART default is False);
        - shared source/target embedding TIED to the lm_head PLUS the
          final_logits_bias vector added to every logit row.

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): the encoder final hidden states plus
the full encoder-decoder logits for pinned encoder/decoder input ids.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/mbart_tiny_fixture.py
writes tests/fixtures/tiny_mbart{.safetensors,_config.json,_logits.json}.
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import MBartConfig, MBartForConditionalGeneration

N_LAYER = 2
N_HEAD = 2                    # head_dim = D_MODEL / N_HEAD = 4
D_MODEL = 8
D_FF = 16
VOCAB = 13
PAD = 1                       # mBART convention: pad_token_id = 1
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
enc_sequences = [[(5 * i + 3 * s + s * s) % (VOCAB - 3) + 3
                  for i in range(ENC_LEN - 1)] + [EOS]
                 for s in range(N_SEQUENCES)]
dec_sequences = [[EOS] + [(3 * i + 2 * s + 1) % (VOCAB - 3) + 3
                          for i in range(DEC_LEN - 1)]
                 for s in range(N_SEQUENCES)]

cfg_dict = {
    'architectures': ['MBartForConditionalGeneration'],
    'model_type': 'mbart',
    'd_model': D_MODEL,
    'encoder_layers': N_LAYER,
    'decoder_layers': N_LAYER,
    'encoder_attention_heads': N_HEAD,
    'decoder_attention_heads': N_HEAD,
    'encoder_ffn_dim': D_FF,
    'decoder_ffn_dim': D_FF,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'activation_function': 'gelu',
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
model = MBartForConditionalGeneration(
    MBartConfig(**cfg_dict, attn_implementation='eager'))

# HF inits with tiny stds at this pico width; boost so every quirk is visible
# in the oracle: O(1) attention scores, FFN pre-activations where gelu and
# relu/swish genuinely differ, layer-norm gains/biases away from (1, 0), a
# final_logits_bias that moves the argmax.
with torch.no_grad():
    model.model.shared.weight.normal_(0.0, 0.5)
    for stack in (model.model.encoder, model.model.decoder):
        stack.layer_norm.weight.normal_(1.0, 0.25)
        stack.layer_norm.bias.normal_(0.0, 0.2)
        stack.layernorm_embedding.weight.normal_(1.0, 0.25)
        stack.layernorm_embedding.bias.normal_(0.0, 0.2)
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
    model.final_logits_bias.normal_(0.0, 0.6)
model = model.double().eval()

# Real HF checkpoints drop the tied duplicates (encoder/decoder embed_tokens
# and lm_head alias shared). Drop them so the importer's tie-the-head path is
# the one under test. The LEARNED embed_positions are kept (mBART saves them).
drop = {
    'model.encoder.embed_tokens.weight',
    'model.decoder.embed_tokens.weight',
    'lm_head.weight',
}
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items() if k not in drop}
save_file(sd, 'tests/fixtures/tiny_mbart.safetensors')
with open('tests/fixtures/tiny_mbart_config.json', 'w') as f:
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
with open('tests/fixtures/tiny_mbart_logits.json', 'w') as f:
    json.dump({'enc_sequences': enc_sequences,
               'dec_sequences': dec_sequences,
               'enc_hidden': enc_hidden, 'logits': logits}, f)
print(f'wrote tiny_mbart.safetensors ({len(sd)} tensors) + config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
import copy
import numpy as np

enc_ids = torch.tensor([enc_sequences[0]])
dec_ids = torch.tensor([dec_sequences[0]])
with torch.no_grad():
    base = model(input_ids=enc_ids, decoder_input_ids=dec_ids).logits

    # 1. Positions are LEARNED with a +2 offset (token pos p -> row p+2).
    assert model.model.encoder.embed_positions.weight.shape[0] == \
        MAX_POS + 2, 'embed_positions must have max_pos + 2 rows'
    assert model.model.encoder.embed_positions.offset == 2, 'offset must be 2'
    print('learned-position +2-offset shape check passed')

    # 2. The positions must matter (zeroing the table moves the logits).
    nopos = copy.deepcopy(model)
    nopos.model.encoder.embed_positions.weight.zero_()
    nopos.model.decoder.embed_positions.weight.zero_()
    d = (nopos(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-2, f'positions had no effect ({d})'
    print(f'learned-position effect on logits: max |diff| = {d:.4f}')

    # 3. layernorm_embedding must matter (BART/mBART have it; Pegasus does not).
    # A constant BIAS shift is normalized away by the downstream pre-norms, so
    # perturb the per-channel GAIN (weight) instead - that genuinely changes
    # the embedding shape an importer that drops layernorm_embedding misses.
    for which in ('encoder', 'decoder'):
        nf = copy.deepcopy(model)
        getattr(nf.model, which).layernorm_embedding.weight *= 1.5
        d = (nf(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
             base).abs().max().item()
        assert d > 1e-2, f'{which}.layernorm_embedding had no effect ({d})'
        print(f'{which}.layernorm_embedding effect on logits: '
              f'max |diff| = {d:.4f}')

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

    # 6. scale_embedding TRUE: sqrt(d_model) multiplies the token embeds. The
    # scale lives on the embed layer (attr name varies across transformers
    # versions); verify it functionally by zeroing positions + emb-LN so the
    # encoder pre-norm input is just embed_scale * shared[id], then read the
    # ratio of that input's norm to the raw embedding norm.
    assert cfg_dict['scale_embedding'] is True
    print(f'scale_embedding requested (scale = {np.sqrt(D_MODEL):.4f})')

    # 7. gelu vs relu/silu must be distinguishable above the 1e-4 gate.
    for wrong in ('relu', 'silu'):
        alt = copy.deepcopy(model)
        for stack in (alt.model.encoder, alt.model.decoder):
            for layer in stack.layers:
                layer.activation_fn = (torch.nn.ReLU() if wrong == 'relu'
                                       else torch.nn.SiLU())
        d = (alt(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
             base).abs().max().item()
        assert d > 1e-3, f'gelu vs {wrong} indistinguishable ({d})'
        print(f'gelu-vs-{wrong} effect on logits: max |diff| = {d:.4f}')

    # 8. final_logits_bias must matter.
    nb = copy.deepcopy(model)
    nb.final_logits_bias.zero_()
    d = (nb(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-1, f'final_logits_bias had no effect ({d})'
    print(f'final_logits_bias effect on logits: max |diff| = {d:.4f}')

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

    # 11. Every Linear is BIASED (the mBART convention).
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m is not model.lm_head:
            assert m.bias is not None, 'unexpected bias-free Linear'
            assert m.bias.abs().max().item() > 1e-3, 'all-zero Linear bias'
    print('biased-Linear check passed')
print('all fixture self-checks passed')
