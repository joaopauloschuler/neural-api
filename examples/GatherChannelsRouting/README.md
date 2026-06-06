# Gather Channels Routing

A tiny, self-contained **forward-pass** demo of the `TNNet.AddGatherChannels`
builder for **channel routing / pruning**. No training — it just builds a small
net, runs one forward pass, and asserts the gathered channels match the
hand-picked source channels with a clear `PASS` / `FAIL`.

## What it does

1. A stem `TNNetPointwiseConvReLU(8)` projects 3 raw input features up to an
   **8-channel** feature map.
2. `NN.AddGatherChannels([5, 1, 1, 6])` keeps / reorders / duplicates a
   hand-picked subset of those channels in one line, producing a **4-channel**
   output where `out[k] = stem[ROUTE[k]]`. This single call demonstrates:
   - **pruning** — 8 channels down to 4 kept,
   - **reorder** — channel 5 routed to output position 0,
   - **reuse** — channel 1 duplicated into two output positions.

`TNNetGatherChannels` is **learnable-free**: the gather is a pure copy/route, so
the output values are byte-for-byte the selected input values (the program
asserts this exactly, tolerance `1e-6`).

## Picking the right channel-select layer

| Layer | Selection | Output Depth |
|-------|-----------|--------------|
| `TNNetGather(c)` | a **single** channel `c` (depth slice) | 1 |
| `TNNetSplitChannels(start, len)` / `([list])` | a **contiguous range** (or an explicit list) split out of the volume | `len` / list length |
| `TNNetGatherChannels([i0, i1, ...])` | an **arbitrary, ordered, possibly-repeated index list** | list length |

Use `AddGatherChannels` when you need an arbitrary subset, a reorder, or a
duplicate; use `TNNetSplitChannels` for a plain contiguous slice; use
`TNNetGather` for the degenerate single-channel case.

## How to run

```bash
cd examples/GatherChannelsRouting
lazbuild GatherChannelsRouting.lpi
../../bin/x86_64-linux/bin/GatherChannelsRouting
```

Runs in well under a second on CPU.

## Sample output

```
GatherChannelsRouting: channel routing / pruning demo
------------------------------------------------------
Stem output Depth   : 8
Gathered output Depth: 4  (route = [5,1,1,6])

out[k]   <- stem ch   gathered        source
-----------------------------------------------
out[0]   <- ch 5      ...              ...
out[1]   <- ch 1      ...              ...
out[2]   <- ch 1      ...              ...
out[3]   <- ch 6      ...              ...

PASS: gathered channels match the hand-picked source channels.
```

(The exact activation values depend on the random stem weights; the gathered
and source columns are always equal — that is what the `PASS` line checks.)
