#!/usr/bin/env python3
"""Generate a tiny RANDOM Pegasus parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_pegasus.*: PegasusForConditionalGeneration (the google/pegasus-*
      abstractive-summarization architecture) with every Pegasus trait the
      importer must reproduce:
        - PRE-norm encoder-decoder blocks (normalize_before=true): LayerNorm
          BEFORE each sub-layer, residual added RAW afterwards
          (self_attn_layer_norm / encoder_attn_layer_norm / final_layer_norm
          per block), plain BIASED nn.LayerNorm (eps 1e-5);
        - STATIC sinusoidal position embeddings in the half-split layout
          (sin in cols 0..d/2-1, cos in d/2..), added with NO padding offset
          (position p reads row p; not BART's +2);
        - NO layernorm_embedding (BART has one);
        - a FINAL encoder AND decoder LayerNorm closing each pre-norm stack
          (model.encoder.layer_norm / model.decoder.layer_norm);
        - exact-erf GELU FFN (fc2(gelu(fc1(x))), both biased);
        - standard 1/sqrt(head_dim) attention, ALL q/k/v/out projections
          biased;
        - scale_embedding TRUE (sqrt(d_model) on the embeddings - the Pegasus
          default);
        - shared source/target embedding TIED to the lm_head PLUS the
          final_logits_bias vector added to every logit row;
        - decoder_start_token_id = pad_token_id (Pegasus's shift).

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): the encoder final hidden states
plus the full encoder-decoder logits for pinned encoder/decoder input ids.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/pegasus_tiny_fixture.py
writes tests/fixtures/tiny_pegasus{.safetensors,_config.json,_logits.json}.
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import PegasusConfig, PegasusForConditionalGeneration

N_LAYER = 2
N_HEAD = 2                    # head_dim = D_MODEL / N_HEAD = 4
D_MODEL = 8
D_FF = 16
VOCAB = 13
PAD = 0                       # Pegasus convention: pad_token_id = 0
EOS = 1
MAX_POS = 16
ENC_LEN = 10
DEC_LEN = 6
N_SEQUENCES = 2

torch.manual_seed(20260613)

# Pinned inputs: encoder ids end with eos (the Pegasus convention) and
# decoder ids start with decoder_start_token_id = pad (Pegasus's shift).
enc_sequences = [[(5 * i + 3 * s + s * s) % (VOCAB - 2) + 2
                  for i in range(ENC_LEN - 1)] + [EOS]
                 for s in range(N_SEQUENCES)]
dec_sequences = [[PAD] + [(3 * i + 2 * s + 1) % (VOCAB - 2) + 2
                          for i in range(DEC_LEN - 1)]
                 for s in range(N_SEQUENCES)]

cfg_dict = {
    'architectures': ['PegasusForConditionalGeneration'],
    'model_type': 'pegasus',
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
    'eos_token_id': EOS,
    'decoder_start_token_id': PAD,
}
model = PegasusForConditionalGeneration(
    PegasusConfig(**cfg_dict, attn_implementation='eager'))

# HF inits with tiny stds at this pico width; boost so every quirk is
# visible in the oracle: O(1) attention scores, FFN pre-activations in the
# region where gelu and relu/swish genuinely differ, layer-norm gains and
# biases away from (1, 0), a final_logits_bias that moves the argmax.
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
    model.final_logits_bias.normal_(0.0, 0.6)
model = model.double().eval()

# Real HF checkpoints drop the tied duplicates (encoder/decoder embed_tokens
# and lm_head alias shared) and the STATIC sinusoidal embed_positions
# (_keys_to_ignore_on_save). Drop them so the importer's
# regenerate-the-table / tie-the-head paths are the ones under test.
drop = {
    'model.encoder.embed_tokens.weight',
    'model.decoder.embed_tokens.weight',
    'lm_head.weight',
    'model.encoder.embed_positions.weight',
    'model.decoder.embed_positions.weight',
}
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items() if k not in drop}
save_file(sd, 'tests/fixtures/tiny_pegasus.safetensors')
with open('tests/fixtures/tiny_pegasus_config.json', 'w') as f:
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
with open('tests/fixtures/tiny_pegasus_logits.json', 'w') as f:
    json.dump({'enc_sequences': enc_sequences,
               'dec_sequences': dec_sequences,
               'enc_hidden': enc_hidden, 'logits': logits}, f)
print(f'wrote tiny_pegasus.safetensors ({len(sd)} tensors) + config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
import copy
import numpy as np

enc_ids = torch.tensor([enc_sequences[0]])
dec_ids = torch.tensor([dec_sequences[0]])
with torch.no_grad():
    base = model(input_ids=enc_ids, decoder_input_ids=dec_ids).logits

    # 1. Positions are STATIC sinusoids in the half-split layout (sin first
    # half, cos second). Verify HF builds exactly that and that the
    # importer's regenerated table will match.
    pos = model.model.encoder.embed_positions
    w = pos.weight if not hasattr(pos, 'create_weight') else \
        pos.create_weight()
    half = D_MODEL // 2
    n_pos = w.shape[0]
    ref = np.zeros((n_pos, D_MODEL))
    for p in range(n_pos):
        for c in range(half):
            ang = p / np.power(10000.0, (2 * c) / D_MODEL)
            ref[p, c] = np.sin(ang)
            ref[p, half + c] = np.cos(ang)
    d = np.abs(w.detach().numpy() - ref).max()
    assert d < 1e-5, f'half-split sinusoid mismatch ({d})'
    print(f'half-split sinusoid layout check passed (max |diff| = {d:.2e})')

    # 2. The positions must matter (zeroing the table moves the logits) - an
    # importer that forgets them FAILS parity.
    nopos = copy.deepcopy(model)
    nopos.model.encoder.embed_positions.weight.zero_()
    nopos.model.decoder.embed_positions.weight.zero_()
    d = (nopos(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-2, f'positions had no effect ({d})'
    print(f'sinusoidal-position effect on logits: max |diff| = {d:.4f}')

    # 3. The FINAL encoder/decoder layer_norm must matter (a pre-norm stack
    # that forgets to close with it FAILS).
    for which in ('encoder', 'decoder'):
        nf = copy.deepcopy(model)
        getattr(nf.model, which).layer_norm.bias += 1.0
        d = (nf(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
             base).abs().max().item()
        assert d > 1e-2, f'{which}.layer_norm had no effect ({d})'
        print(f'final {which}.layer_norm effect on logits: max |diff| = '
              f'{d:.4f}')

    # 4. PRE-norm placement: shifting the FINAL encoder layer_norm bias must
    # shift the encoder output by EXACTLY that amount (a post-norm wiring
    # would put a per-block norm last instead and dilute it).
    pn = copy.deepcopy(model)
    pn.model.encoder.layer_norm.bias += 1.0
    eh = pn.model.encoder(input_ids=enc_ids).last_hidden_state
    eh_base = model.model.encoder(input_ids=enc_ids).last_hidden_state
    d = (eh - eh_base - 1.0).abs().max().item()
    assert d < 1e-9, f'encoder output is not closed by layer_norm ({d})'
    print('pre-norm final-LayerNorm placement check passed')

    # 5. PRE-norm vs post-norm: shifting a per-block self_attn_layer_norm
    # bias must NOT move the output by exactly that amount (it feeds the
    # sub-layer, not the residual sum) - distinguishes pre- from post-norm.
    pb = copy.deepcopy(model)
    pb.model.encoder.layers[0].self_attn_layer_norm.bias += 1.0
    eh = pb.model.encoder(input_ids=enc_ids).last_hidden_state
    d = (eh - eh_base).abs().max().item()
    assert d > 1e-3, 'per-block norm had no effect'
    # in a post-norm model the LAST block norm would shift output by ~1.0;
    # here it is a pre-sublayer norm so the shift is diffused.
    print(f'per-block pre-norm shift propagates diffusely (|diff| = {d:.4f})')

    # 6. scale_embedding TRUE: sqrt(d_model) multiplies the token embeds.
    assert abs(model.model.encoder.embed_scale -
               np.sqrt(D_MODEL)) < 1e-9, 'embed_scale != sqrt(d_model)'
    print(f'scale_embedding check passed (scale = {np.sqrt(D_MODEL):.4f})')

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
    dec2[0, -1] = (dec2[0, -1] + 1) % (VOCAB - 2) + 2
    l2 = model(input_ids=enc_ids, decoder_input_ids=dec2).logits
    d_early = (l2[0, :-1] - base[0, :-1]).abs().max().item()
    assert d_early < 1e-9, f'decoder attention is not causal ({d_early})'
    print('decoder causality check passed')

    # 11. Every Linear is BIASED (the Pegasus convention).
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m is not model.lm_head:
            assert m.bias is not None, 'unexpected bias-free Linear'
            assert m.bias.abs().max().item() > 1e-3, 'all-zero Linear bias'
    print('biased-Linear check passed')
print('all fixture self-checks passed')
