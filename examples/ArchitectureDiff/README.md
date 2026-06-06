# ArchitectureDiff

Demonstrates `TNNet.DiffArchitecture` and `TNNet.DiffArchitectureFromString`:
two near-identical SimpleImageClassifier-style variants are built in
memory and the architectural difference is printed as a unified-diff-style
report (matching layers prefixed with a space, removed with `-`, added
with `+`). Pure-CPU, no training, runs in milliseconds.

Useful as:

- a builder regression aid (pin a "golden" architecture string from
  `SaveStructureToString`, diff your refactored builder against it),
- a review aid when a PR touches a network-builder helper,
- a teaching demo of the introspection API alongside `PrintSummary`.

Build & run:

```
lazbuild ArchitectureDiff.lpi
./../../bin/x86_64-linux/bin/ArchitectureDiff
```
