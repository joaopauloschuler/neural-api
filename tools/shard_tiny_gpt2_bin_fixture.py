#!/usr/bin/env python3
"""Splits tests/fixtures/tiny_gpt2.bin (the torch.save twin of
tiny_gpt2.safetensors, written by tools/torch_bin_fixture.py) into a
2-shard HF-style sharded torch.save checkpoint for the
TNNetTorchBinReader sharded-load tests:

  tests/fixtures/tiny_gpt2-00001-of-00002.bin
  tests/fixtures/tiny_gpt2-00002-of-00002.bin
  tests/fixtures/tiny_gpt2.bin.index.json

The split mirrors how HF shards real .bin checkpoints (whole tensors
assigned to shards, each shard its own torch.save zip; here: wte/wpe +
block 0 in shard 1, the rest in shard 2). The index.json carries the
standard {"metadata": {"total_size": N}, "weight_map": {tensor: shard}}
layout - the SAME shape as model.safetensors.index.json. Run from the
repo root:

  python3 tools/shard_tiny_gpt2_bin_fixture.py

Needs torch only (CPU build is fine).
"""
import json
import os

import torch

FIXDIR = os.path.join("tests", "fixtures")
SRC = os.path.join(FIXDIR, "tiny_gpt2.bin")
SHARD1 = "tiny_gpt2-00001-of-00002.bin"
SHARD2 = "tiny_gpt2-00002-of-00002.bin"
INDEX = os.path.join(FIXDIR, "tiny_gpt2.bin.index.json")


def shard_of(name):
    if name in ("wte.weight", "wpe.weight") or name.startswith("h.0."):
        return SHARD1
    return SHARD2


def main():
    state = torch.load(SRC, weights_only=True)

    shards = {SHARD1: {}, SHARD2: {}}
    weight_map = {}
    total_size = 0
    for name, value in sorted(state.items()):
        shard = shard_of(name)
        shards[shard][name] = value
        weight_map[name] = shard
        total_size += value.numel() * value.element_size()
    assert shards[SHARD1] and shards[SHARD2], "both shards must be non-empty"

    for shard, content in shards.items():
        torch.save(content, os.path.join(FIXDIR, shard))

    with open(INDEX, "w") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            f,
            indent=1,
            sort_keys=True,
        )
        f.write("\n")
    for name in (SHARD1, SHARD2):
        p = os.path.join(FIXDIR, name)
        print(f"{name}: {os.path.getsize(p)} bytes "
              f"({len(shards[name])} tensors)")
    print(f"{INDEX}: {os.path.getsize(INDEX)} bytes")


if __name__ == "__main__":
    main()
