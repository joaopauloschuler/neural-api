#!/usr/bin/env python3
"""Fixtures for the torch.save metadata API of the restricted reader
(neural/neuraltorchbin.pas, TNNetTorchBinReader.GetMetadata*).

Generates, into tests/fixtures/torch_head/:

  kev/head.pt           a Kev-shaped head.pt written by kev's own
                        kev.checkpoint.write_meta: PointerHead(d=8, dp=4)
                        state_dict under "head" plus the Meta fields and
                        nested training metadata (dicts, lists of mixed
                        scalars, True/False/None, an empty list, a memo-shared
                        string, a negative and a 64-bit int).
  openjev/head.pt       an Open-Jev-shaped head: torch.save(Linear(8, 1)
                        .state_dict()), storage location 'cpu'.
  openjev_cuda/head.pt  the same tensors with storage location 'cuda:0', as
                        the released Open-Jev-2B/9B heads record it.
  evil_meta/head.pt     Kev-shaped, with a metadata value that REDUCEs the
                        non-whitelisted GLOBAL `os system`: must raise.
  cyclic/head.pt        Kev-shaped, with a metadata list that contains
                        itself (the pickle memo allows it): must raise.
  shared_bomb/head.pt   Kev-shaped, with 40 levels of [x, x] lists: 2^40
                        leaves when walked as a tree: must raise.
  expected.json         per file: every tensor (shape + values) and every
                        metadata leaf by dotted name, in the reader's naming.

The 'cuda:0' tag comes from torch.save itself: a location tagger registered
ahead of torch's CPU tagger (torch.serialization.register_package) names
every storage 'cuda:0'; the bytes are the CPU tensors' bytes. The script
checks with pickletools that the tag is in data.pkl.

Run from the repo root:  ~/x/bin/python -I tools/torch_head_fixture.py
Needs torch and kev (installed --no-deps; only kev.checkpoint and kev.model
are imported).
"""
import io
import json
import os
import pickletools
import zipfile

import torch
from kev.checkpoint import Meta, write_meta
from kev.model import PointerHead

FIXDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "tests", "fixtures", "torch_head")
HIDDEN = 8      # d: backbone hidden size
POINTER = 4     # dp: Kev pointer dimension (head_dim)


def flatten_metadata(value, name, out, lists):
    """Mirrors TNNetTorchBinReader's RecordMetadata naming."""
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, dict):
        prefix = name + "." if name else ""
        for k, v in value.items():
            if isinstance(k, str):
                flatten_metadata(v, prefix + k, out, lists)
    elif isinstance(value, (list, tuple)):
        lists[name] = len(value)
        for i, v in enumerate(value):
            flatten_metadata(v, f"{name}.{i}", out, lists)
    else:
        out[name] = value


def flatten_tensors(value, name, out):
    if isinstance(value, torch.Tensor):
        out[name] = {"shape": list(value.shape),
                     "values": [float(x) for x in value.reshape(-1).tolist()]}
    elif isinstance(value, dict):
        prefix = name + "." if name else ""
        for k, v in value.items():
            flatten_tensors(v, prefix + k, out)


def expected_of(obj):
    meta, lists, tensors = {}, {}, {}
    flatten_metadata(obj, "", meta, lists)
    flatten_tensors(obj, "", tensors)
    return {"tensors": tensors, "metadata": meta, "lists": lists}


def storage_locations(path):
    """Location strings of every persistent id in data.pkl (pickletools only;
    nothing is unpickled)."""
    with zipfile.ZipFile(path) as z:
        pkl = next(n for n in z.namelist() if n.endswith("/data.pkl"))
        data = z.read(pkl)
    text = io.StringIO()
    pickletools.dis(data, out=text)
    return [line.split("'")[1] for line in text.getvalue().splitlines()
            if "BINUNICODE" in line and ("'cpu'" in line or "'cuda" in line)]


class Evil:
    """Pickles to REDUCE(os.system, ('echo pwned',))."""

    def __reduce__(self):
        return (os.system, ("echo pwned",))


def main():
    torch.manual_seed(20261018)
    expected = {}

    # ---- Kev: through kev's own Meta.to_dict + torch.save ----
    head = PointerHead(HIDDEN, dp=POINTER)
    meta = Meta(base="Qwen/Qwen3.5-0.8B-Base",
                head=head.state_dict(),
                base_revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
                lora=16, head_dim=POINTER, option_isolation=False,
                special_embeddings=False, weights_dtype="fp32",
                temperature=2.3510958125672174, holdout=[], weights="lora")
    meta.extra = {
        "args": {"base": "Qwen/Qwen3.5-0.8B-Base", "n_per_source": 1000,
                 "epochs": 1, "lr": 2e-05, "head_lr": 0.0, "lora": 16,
                 "holdout": "", "train_sources": "", "max_state": 7552,
                 "seed": 1, "offset": -3, "big_seed": 2 ** 40,
                 "init_from": "jaredpalmer/kev-0.8b"},
        "suite_sha256": "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8"
                        "265af0edbb5514d2",
        "init_source": {"resolved": "/hub/models--jaredpalmer--kev-0.8b",
                        "adapter_tensors": 372},
        "temperature_fit": {
            "n": 1264,
            "method": "min micro mean NLL over a 121-point log grid",
            "cross_validation": {
                "raw": {"n": 1264, "ece": 0.10212917934363491},
                "ece_ci95": {"raw": [0.08454655575092548,
                                     0.12585968478253243],
                             "delta": [-0.08723685319536686,
                                       -0.04532170327333796]},
                "separated": True,
                "temperatures": [2.406050072164233, 2.3510958125672174],
                "mixed": [1, 2.5, "x", True, None, []],
                "folds": 5}},
        "lora_placement": "full",
    }
    kev_dir = os.path.join(FIXDIR, "kev")
    os.makedirs(kev_dir, exist_ok=True)
    write_meta(kev_dir, meta)
    expected["kev"] = expected_of(meta.to_dict())

    # ---- Open-Jev: plain Linear(d, 1) state_dict ----
    linear = torch.nn.Linear(HIDDEN, 1)
    ojev_dir = os.path.join(FIXDIR, "openjev")
    os.makedirs(ojev_dir, exist_ok=True)
    torch.save(linear.state_dict(), os.path.join(ojev_dir, "head.pt"))
    expected["openjev"] = expected_of(linear.state_dict())

    # ---- Open-Jev with storages tagged 'cuda:0' ----
    cuda_dir = os.path.join(FIXDIR, "openjev_cuda")
    os.makedirs(cuda_dir, exist_ok=True)
    registry = torch.serialization._package_registry
    before = list(registry)
    torch.serialization.register_package(
        0, lambda storage: "cuda:0", lambda storage, location: None)
    try:
        torch.save(linear.state_dict(), os.path.join(cuda_dir, "head.pt"))
    finally:
        registry[:] = before
    expected["openjev_cuda"] = expected["openjev"]

    # ---- security: a hostile metadata value, a self-containing list ----
    cyclic = []
    cyclic.append(cyclic)
    bomb = []
    for _ in range(40):
        bomb = [bomb, bomb]
    for name, hostile in (("evil_meta", Evil()), ("cyclic", cyclic),
                          ("shared_bomb", bomb)):
        bad_dir = os.path.join(FIXDIR, name)
        os.makedirs(bad_dir, exist_ok=True)
        torch.save({"head": linear.state_dict(), "args": {"x": hostile},
                    "temperature": 1.0}, os.path.join(bad_dir, "head.pt"))

    for name, want in (("kev", "cpu"), ("openjev", "cpu"),
                       ("openjev_cuda", "cuda:0")):
        path = os.path.join(FIXDIR, name, "head.pt")
        locations = storage_locations(path)
        assert locations and all(loc == want for loc in locations), \
            (name, locations)
        print(f"{name}/head.pt: {os.path.getsize(path)} bytes, "
              f"locations {locations}")

    with open(os.path.join(FIXDIR, "expected.json"), "w") as f:
        json.dump(expected, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
