#!/usr/bin/env python3
# make_pico_bark_fixture.py
#
# Builds a committed RE-RANDOMIZED PICO fixture for the Bark text-to-speech
# importer (BuildBarkFromSafeTensors / TestBarkParity), plus a self-contained
# float64 numpy/HF oracle of the THREE GPT-style sub-models' forward logits.
#
# Bark = three stacked GPT-2-style decoders chained:
#   * SEMANTIC   (BarkCausalModel): text+semantic tokens -> semantic logits.
#   * COARSE     (BarkCausalModel): semantic tokens -> coarse EnCodec codes.
#   * FINE       (BarkFineModel):   NON-causal over the codebook axis; given
#                codebooks 0..idx predicts codebook idx (full bidirectional
#                attention over time).
# The EnCodec decode tail reuses the LANDED tiny_musicgen_encodec.* fixture
# (codebook_size=16, 6 RVQ codebooks), so the Bark fine model here is sized so
# its emitted codes are valid RVQ indices: output_vocab_size <= 16, and
# n_codes_total == 6 (the EnCodec codebook count).
#
# Does NOT download the real suno/bark checkpoint: it constructs the HF Bark
# sub-models from tiny configs, re-randomizes every weight to O(1) scale (so
# every quirk -- merged input embedding sum, exact-erf GELU, non-causal fine
# attention, codebook-conditioned sum -- is visible in the oracle), runs HF
# itself in float64 as the oracle, and dumps F32 safetensors + JSON.
#
# Run from the repo root:  /home/bpsa/x/bin/python tools/make_pico_bark_fixture.py
#
# Coded by Claude (AI).
import json
import os

import torch
from safetensors.torch import save_file
from transformers.models.bark.configuration_bark import (
    BarkCoarseConfig,
    BarkFineConfig,
    BarkSemanticConfig,
)
from transformers.models.bark.modeling_bark import (
    BarkCoarseModel,
    BarkFineModel,
    BarkSemanticModel,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

SEED = 20260626

# Pico dims. Tiny n_layer/n_head/hidden, small vocab, few codebooks so the
# Pascal parity test runs in a fraction of a second under ulimit -v 3000000.
HIDDEN = 16
NUM_HEADS = 2
NUM_LAYERS = 2
BLOCK_SIZE = 32          # max positions (>= any sequence length we test)
SEM_VOCAB = 33           # semantic in/out vocab (text+semantic merged space)
COARSE_VOCAB = 20        # coarse in/out vocab
FINE_VOCAB = 16          # MUST be <= EnCodec codebook_size (16) -> valid codes
N_CODES_TOTAL = 6        # == tiny_musicgen_encodec RVQ codebook count
N_CODES_GIVEN = 1        # fine predicts codebooks 1..5 (5 lm_heads)
BIAS = True              # default Bark config has bias=True (real suno = False)


def rerandomize(model):
    """Push every parameter off HF's tiny-std init to O(1) scale so each
    quirk provably moves the oracle."""
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                p.copy_(torch.empty_like(p).normal_(0.0, 0.5, generator=g))
            else:
                # biases / layernorm: keep gains near 1 but perturbed, biases
                # spread so they are not vacuous.
                if "layernorm" in name and name.endswith(".weight"):
                    p.copy_(1.0 + torch.empty_like(p).normal_(0.0, 0.3, generator=g))
                else:
                    p.copy_(torch.empty_like(p).normal_(0.0, 0.3, generator=g))


def sub_state_dict(model):
    sd = {}
    for k, v in model.state_dict().items():
        if k.endswith(".attn.bias"):
            continue  # the causal-mask buffer; not a real weight
        sd[k] = v.to(torch.float32).clone().contiguous()
    return sd


def common_cfg(model_type):
    return dict(
        model_type=model_type,
        block_size=BLOCK_SIZE,
        input_vocab_size=0,   # filled per sub-model
        output_vocab_size=0,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        hidden_size=HIDDEN,
        dropout=0.0,
        bias=BIAS,
    )


def main():
    torch.manual_seed(SEED)

    # ---------------- SEMANTIC sub-model ----------------
    sem_cfg = BarkSemanticConfig(
        block_size=BLOCK_SIZE, input_vocab_size=SEM_VOCAB,
        output_vocab_size=SEM_VOCAB, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, hidden_size=HIDDEN, bias=BIAS, dropout=0.0)
    sem = BarkSemanticModel(sem_cfg)
    rerandomize(sem)
    sem = sem.double().eval()

    # ---------------- COARSE sub-model ----------------
    coarse_cfg = BarkCoarseConfig(
        block_size=BLOCK_SIZE, input_vocab_size=COARSE_VOCAB,
        output_vocab_size=COARSE_VOCAB, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, hidden_size=HIDDEN, bias=BIAS, dropout=0.0)
    coarse = BarkCoarseModel(coarse_cfg)
    rerandomize(coarse)
    coarse = coarse.double().eval()

    # ---------------- FINE sub-model (non-causal over codebooks) ----------
    fine_cfg = BarkFineConfig(
        block_size=BLOCK_SIZE, input_vocab_size=FINE_VOCAB,
        output_vocab_size=FINE_VOCAB, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, hidden_size=HIDDEN, bias=BIAS, dropout=0.0,
        n_codes_total=N_CODES_TOTAL, n_codes_given=N_CODES_GIVEN)
    fine = BarkFineModel(fine_cfg)
    rerandomize(fine)
    fine = fine.double().eval()

    # ---------------- write fixtures ----------------
    save_file(sub_state_dict(sem),
              os.path.join(FIX, "tiny_bark_semantic.safetensors"))
    save_file(sub_state_dict(coarse),
              os.path.join(FIX, "tiny_bark_coarse.safetensors"))
    save_file(sub_state_dict(fine),
              os.path.join(FIX, "tiny_bark_fine.safetensors"))

    # A single combined config JSON in the HF "bark" nested shape, plus the
    # per-sub-model flat fields the importer reads.
    cfg_out = {
        "model_type": "bark",
        "semantic_config": {
            "model_type": "semantic",
            "block_size": BLOCK_SIZE, "input_vocab_size": SEM_VOCAB,
            "output_vocab_size": SEM_VOCAB, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "hidden_size": HIDDEN, "bias": BIAS,
        },
        "coarse_acoustics_config": {
            "model_type": "coarse_acoustics",
            "block_size": BLOCK_SIZE, "input_vocab_size": COARSE_VOCAB,
            "output_vocab_size": COARSE_VOCAB, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "hidden_size": HIDDEN, "bias": BIAS,
        },
        "fine_acoustics_config": {
            "model_type": "fine_acoustics",
            "block_size": BLOCK_SIZE, "input_vocab_size": FINE_VOCAB,
            "output_vocab_size": FINE_VOCAB, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "hidden_size": HIDDEN, "bias": BIAS,
            "n_codes_total": N_CODES_TOTAL, "n_codes_given": N_CODES_GIVEN,
        },
    }
    with open(os.path.join(FIX, "tiny_bark_config.json"), "w") as f:
        json.dump(cfg_out, f, indent=1)

    # ---------------- float64 oracle ----------------
    rng = torch.Generator().manual_seed(SEED + 1)

    def rand_ids(n, vocab):
        return torch.randint(0, vocab, (n,), generator=rng).tolist()

    # SEMANTIC: >=3 fixed input id sequences -> per-position logits.
    sem_seqs = [rand_ids(5, SEM_VOCAB) for _ in range(3)]
    sem_logits = []
    with torch.no_grad():
        for s in sem_seqs:
            out = sem(input_ids=torch.tensor([s]), use_cache=False).logits[0]
            sem_logits.append(out.tolist())

    # COARSE: same, its own vocab.
    coarse_seqs = [rand_ids(5, COARSE_VOCAB) for _ in range(3)]
    coarse_logits = []
    with torch.no_grad():
        for s in coarse_seqs:
            out = coarse(input_ids=torch.tensor([s]), use_cache=False).logits[0]
            coarse_logits.append(out.tolist())

    # FINE: input_ids shape (1, seq, n_codes_total); for each codebook_idx in
    # 1..n_codes_total-1 the model conditions on codebooks 0..idx and predicts
    # codebook idx. We pin >=3 (seq, idx) cases.
    SEQ = 5
    fine_inputs = []   # list of (codes [seq][n_codes_total], codebook_idx)
    fine_logits = []
    fg = torch.Generator().manual_seed(SEED + 2)
    for case in range(3):
        codes = torch.randint(0, FINE_VOCAB, (1, SEQ, N_CODES_TOTAL),
                              generator=fg)
        cb_idx = case + 1   # 1, 2, 3 -> valid (each < n_codes_total, > 0)
        with torch.no_grad():
            out = fine(cb_idx, input_ids=codes).logits[0]  # (seq, out_vocab)
        fine_inputs.append({"codes": codes[0].tolist(),
                            "codebook_idx": cb_idx})
        fine_logits.append(out.tolist())

    oracle = {
        "hidden": HIDDEN, "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS,
        "block_size": BLOCK_SIZE,
        "semantic": {"vocab": SEM_VOCAB, "sequences": sem_seqs,
                     "logits": sem_logits},
        "coarse": {"vocab": COARSE_VOCAB, "sequences": coarse_seqs,
                   "logits": coarse_logits},
        "fine": {"vocab": FINE_VOCAB, "n_codes_total": N_CODES_TOTAL,
                 "n_codes_given": N_CODES_GIVEN,
                 "cases": fine_inputs, "logits": fine_logits},
    }
    with open(os.path.join(FIX, "tiny_bark_ref.json"), "w") as f:
        json.dump(oracle, f)

    # ---------------- fixture self-checks ----------------
    # Each quirk must provably move the oracle. We perturb one component and
    # assert the logits shift, so a Pascal bug that drops a quirk can't pass.
    import copy

    def max_logit(m, kind):
        if kind == "sem":
            return m(input_ids=torch.tensor([sem_seqs[0]]),
                     use_cache=False).logits[0]
        if kind == "fine":
            c = torch.tensor([fine_inputs[1]["codes"]])
            return m(fine_inputs[1]["codebook_idx"], input_ids=c).logits[0]

    base_sem = max_logit(sem, "sem")
    # (1) position embedding effect
    sem2 = copy.deepcopy(sem)
    with torch.no_grad():
        sem2.position_embeds_layer.weight.zero_()
    d = (base_sem - max_logit(sem2, "sem")).abs().max().item()
    assert d > 1e-2, f"position embedding has no effect: {d}"

    # (2) fine codebook-conditioning effect: zeroing the embedding of a
    # conditioning codebook (<= idx) must move the prediction.
    base_fine = max_logit(fine, "fine")
    fine2 = copy.deepcopy(fine)
    with torch.no_grad():
        fine2.input_embeds_layers[1].weight.zero_()  # codebook 1 <= idx 2
    d = (base_fine - max_logit(fine2, "fine")).abs().max().item()
    assert d > 1e-2, f"fine conditioning codebook has no effect: {d}"

    # (3) fine non-causal: a future-time code must influence an earlier
    # position's logits (bidirectional attention). Flip the last timestep's
    # conditioning code and check an EARLIER row moved.
    fine_codes = torch.tensor([fine_inputs[1]["codes"]])
    idx = fine_inputs[1]["codebook_idx"]
    with torch.no_grad():
        a = fine(idx, input_ids=fine_codes).logits[0]
        fc2 = fine_codes.clone()
        fc2[0, -1, 0] = (fc2[0, -1, 0] + 1) % FINE_VOCAB
        b = fine(idx, input_ids=fc2).logits[0]
    d = (a[0] - b[0]).abs().max().item()  # row 0 changed by a change at row -1
    assert d > 1e-2, f"fine attention is not bidirectional over time: {d}"

    print("Bark pico fixture written. Self-checks passed.")
    for fn in ("tiny_bark_semantic.safetensors", "tiny_bark_coarse.safetensors",
               "tiny_bark_fine.safetensors", "tiny_bark_config.json",
               "tiny_bark_ref.json"):
        p = os.path.join(FIX, fn)
        print("  %-34s %7d bytes" % (fn, os.path.getsize(p)))


if __name__ == "__main__":
    main()
