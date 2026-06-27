#!/usr/bin/env python3
"""Generate tiny RANDOM torch nn.LSTM / nn.GRU parity fixtures for
tests/TestNeuralPretrained.pas (no network access needed: the modules are
randomly initialized, never downloaded).

The fixtures exercise the GENERIC torch-RNN safetensors importer
(LoadTorchLSTMInto / LoadTorchGRUInto in neuralpretrained.pas) which copies
the real torch nn.LSTM / nn.GRU
  weight_ih_l{k} / weight_hh_l{k} / bias_ih_l{k} / bias_hh_l{k}  (+ _reverse)
slabs into the AddBidirectionalLSTM / AddBidirectionalGRU stacked cells.

Each emitted fixture is a {safetensors weights, JSON oracle} pair, KB-scale,
pinned in tests/fixtures/. The oracle holds the full output SEQUENCE produced
by torch in float64 for a fixed pinned input (matches what the builder emits:
the per-step hidden states, [forward;backward] concatenated for bidirectional).

Configurations covered (LSTM and GRU each):
  - 1-layer unidirectional
  - 1-layer bidirectional
  - 2-layer bidirectional

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/torch_rnn_tiny_fixture.py
writes tests/fixtures/tiny_torch_{lstm,gru}_*.{safetensors,_ref.json}.
Needs torch + safetensors.
"""
import json
import os

import torch
from safetensors.torch import save_file

OUT = 'tests/fixtures'
SEQ_LEN = 5
# input_size == hidden so the AddBidirectional{LSTM,GRU} builder needs NO
# per-direction projection at layer 0 (the cells are shape-preserving). The
# builder inserts a learned PointwiseConvLinear projection whenever an incoming
# Depth != Hidden - that projection has no torch counterpart, so torch weights
# can only be loaded FAITHFULLY when every cell's input width is Hidden. That
# holds for: unidirectional stacks of any depth (each cell sees Hidden) and a
# SINGLE bidirectional layer (both directions see input_size==Hidden). A
# 2-layer BIDIRECTIONAL stack feeds layer 1 a 2*Hidden concat -> a projection
# fires -> NOT faithfully importable by this builder (the importer rejects it).
INPUT_SIZE = 4
HIDDEN = 4

# Pinned input sequence: (SEQ_LEN, INPUT_SIZE). Deterministic, O(1) scale so
# every gate is genuinely exercised (not stuck in a saturated region).
PINNED_X = [[round(0.7 * (i + 1) - 0.31 * j + (1 if (i + j) % 2 else -1) * 0.45, 6)
             for j in range(INPUT_SIZE)]
            for i in range(SEQ_LEN)]


def emit(kind, num_layers, bidirectional, seed):
    torch.manual_seed(seed)
    Cls = torch.nn.LSTM if kind == 'lstm' else torch.nn.GRU
    rnn = Cls(input_size=INPUT_SIZE, hidden_size=HIDDEN,
              num_layers=num_layers, bidirectional=bidirectional,
              batch_first=False, bias=True)
    # Boost the random init away from torch's tiny 1/sqrt(hidden) uniform so
    # the gates span their nonlinear range and biases genuinely matter.
    with torch.no_grad():
        for name, p in rnn.named_parameters():
            if 'weight' in name:
                p.normal_(0.0, 0.6)
            else:  # bias_ih / bias_hh
                p.normal_(0.0, 0.4)
    rnn = rnn.double().eval()

    x = torch.tensor(PINNED_X, dtype=torch.float64).unsqueeze(1)  # (T,1,in)
    with torch.no_grad():
        out, _ = rnn(x)  # out: (T, 1, num_directions*HIDDEN)
    ref = out.squeeze(1).tolist()  # (T, num_directions*HIDDEN)

    tag = f'{kind}_{"bi" if bidirectional else "uni"}_l{num_layers}'
    sd = {k: v.to(torch.float32).clone().contiguous()
          for k, v in rnn.state_dict().items()}
    st_path = os.path.join(OUT, f'tiny_torch_{tag}.safetensors')
    save_file(sd, st_path)

    ref_path = os.path.join(OUT, f'tiny_torch_{tag}_ref.json')
    with open(ref_path, 'w') as f:
        json.dump({'kind': kind, 'num_layers': num_layers,
                   'bidirectional': bidirectional,
                   'input_size': INPUT_SIZE, 'hidden': HIDDEN,
                   'seq_len': SEQ_LEN, 'input': PINNED_X, 'output': ref}, f)

    print(f'wrote {st_path} ({len(sd)} tensors) + {os.path.basename(ref_path)}')
    for k in sorted(sd):
        print(f'    {k} {list(sd[k].shape)}')

    # ---- fixture self-checks ----
    with torch.no_grad():
        # 1. Biases genuinely matter: zeroing all bias_ih/bias_hh changes output.
        import copy
        nb = copy.deepcopy(rnn)
        for name, p in nb.named_parameters():
            if 'bias' in name:
                p.zero_()
        d = (nb(x)[0] - out).abs().max().item()
        assert d > 1e-3, f'{tag}: biases had no effect ({d})'
        # 2. For bidirectional, the [forward;backward] concat order must hold:
        #    the forward half (first HIDDEN cols) of a bidir net equals a
        #    unidirectional net built from the same forward weights. Only
        #    meaningful for 1 layer (layer k>0 has a different input_size for
        #    the unidirectional rebuild).
        if bidirectional and num_layers == 1:
            uni = Cls(input_size=INPUT_SIZE, hidden_size=HIDDEN,
                      num_layers=num_layers, bidirectional=False,
                      bias=True).double().eval()
            usd = uni.state_dict()
            for k in usd:  # copy the forward-direction slabs verbatim
                usd[k] = rnn.state_dict()[k].clone()
            uni.load_state_dict(usd)
            fwd = uni(x)[0].squeeze(1)
            dd = (out.squeeze(1)[:, :HIDDEN] - fwd).abs().max().item()
            assert dd < 1e-9, f'{tag}: forward-half concat mismatch ({dd})'
    print(f'    self-checks passed for {tag}')


def main():
    os.makedirs(OUT, exist_ok=True)
    # Faithfully importable by AddBidirectional{LSTM,GRU} (input_size==Hidden,
    # so no learned projection is inserted at any cell boundary):
    emit('lstm', 1, False, 20260627)   # 1-layer unidirectional
    emit('lstm', 2, False, 20260628)   # 2-layer unidirectional
    emit('lstm', 1, True, 20260629)    # 1-layer bidirectional
    emit('gru', 1, False, 20260630)
    emit('gru', 2, False, 20260631)
    emit('gru', 1, True, 20260632)
    print('all torch-RNN fixtures written')


if __name__ == '__main__':
    main()
