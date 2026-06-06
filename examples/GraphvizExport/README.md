# GraphvizExport

Tiny example for `TNNet.ToGraphvizDot`, which emits a [Graphviz](https://graphviz.org/)
DOT description of a network's layer DAG.

The program builds four small networks and prints the DOT for each:

1. **NET 1** - a plain sequential MLP (`8 -> 16 -> 16 -> 1`): a single chain
   of edges.
2. **NET 2** - a branched residual net whose short cut and longer path are
   merged by a `TNNetSum`. The sum node has **two incoming edges**, so the
   multi-input DAG is visible. This net's DOT is also written to
   `branched_net.dot`.
3. **NET 3** - a `TNNetDeepConcat` merge of two convolutional branches
   (another multi-input node).
4. **NET 4** - an empty network, showing the valid empty `digraph` guard.

Each node is labelled `<idx>: <ClassName>` plus the layer's output shape
(`SizeX x SizeY x Depth`). `ToGraphvizDot` is forward/structure-only: it never
trains and never runs a forward or backward pass.

## Build & run

```
cd examples/GraphvizExport
lazbuild GraphvizExport.lpi
../../bin/x86_64-linux/bin/GraphvizExport
```

Total runtime is well under a minute (no dataset, tiny synthetic nets).

## Render the DOT

Pipe any printed block (or the written `branched_net.dot`) through Graphviz:

```
dot -Tpng branched_net.dot -o branched_net.png
dot -Tsvg branched_net.dot -o branched_net.svg
```
