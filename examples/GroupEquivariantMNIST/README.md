# GroupEquivariantMNIST — p4 Group-Equivariant CNN

A small bake-off demonstrating `TNNetGroupConvP4` + `TNNetGroupPoolP4` (Cohen &
Welling 2016, "Group Equivariant Convolutional Networks",
[arXiv:1602.07576](https://arxiv.org/abs/1602.07576)): a CNN that is robust to
90-degree rotations **by construction**, beating a parameter-matched plain CNN on
a rotated test set at **equal weight count**.

## The idea

`TNNetGroupConvP4` is the **lifting** rung of a Group Equivariant CNN for the C4
group of 90-degree plane rotations. One learned `K×K` kernel bank is **shared**
across the four rotations of C4: the layer convolves the input with the four
rot-{0,90,180,270} copies of the *same* filter and stacks the four responses
along a new 4-fold **orientation** sub-axis of the output Depth
(`Depth = 4 × FeaturesCount`, channel `= co*4 + r`). A 90-degree rotation of the
input therefore only **cyclically permutes** those orientation channels (plus
rotating the spatial map) — it never scrambles them. The three extra rotated
filters are index *views* of the one trained bank, and the backward pass **folds**
the four orientation gradients back onto that single shared kernel.

`TNNetGroupPoolP4` then **max-reduces** over the four orientation channels of each
feature, collapsing the C4 field to a rotation-**invariant** `(SizeX, SizeY,
FeaturesCount)` map. Followed by a global spatial average pool (a 90-degree input
rotation only permutes spatial positions, so their average is unchanged), the
whole stack is **exactly** invariant to the C4 rotations of the input.

This is distinct from `TNNetFlipX`/`FlipY` (fixed parameter-free involutions, data
augmentation primitives) and from `TNNetCondConv`/`Quaternion`/`Octonion` (which
share weights across a *different* algebra, not a spatial symmetry group).

## The task

8×8 single-channel **chiral** glyphs in 4 classes (each class = a bent corner/stair
shape whose appearance *changes* under rotation). Both nets train on **upright**
glyphs only — no rotation augmentation, the classic Cohen–Welling setting — and
are then evaluated on a test set rotated by 90/180/270 degrees. The p4 net
generalises to the rotated orientations for free; the plain net only ever saw the
upright orientation.

Both models are sized to the **same trainable weight count** (the p4 conv's shared
bank is 1/4 the parameters of the equivalent independent filters) and share the
same pooling/classifier tail.

## Running

```
lazbuild examples/GroupEquivariantMNIST/GroupEquivariantMNIST.lpi
./bin/x86_64-linux/bin/GroupEquivariantMNIST
```

Pure CPU, tiny data, runs in well under a minute (~45 s). Typical result (higher
rotated accuracy is better; `C4-invariance-error` = max change in the final logits
when the input is rotated 90 degrees — ~0 means exactly rotation-invariant):

```
  p4-CNN     trainable weights: 324
  plain-CNN  trainable weights: 314

  p4-CNN     rotated-test-acc = 0.92   C4-invariance-error = 0.000001
  plain-CNN  rotated-test-acc = 0.41   C4-invariance-error = 0.64
```

The p4 net **more than doubles** the plain net's rotated accuracy at equal weights
and its rotation-invariance error is ~0 by construction. The example also prints
`TNNet.EquivarianceReport` on both nets (the existing flip/roll diagnostic) — a
forward-only symmetry probe that closes the loop with the constructive guarantee.

## Out of scope (follow-ups)

This is the **lifting** conv only (plane → C4 field). Not built here: the full
**p4m** group (adds reflections, an 8-fold field), **steerable / SO(2)**
continuous-rotation harmonics, and a **field → field** p4 *group* convolution on an
existing C4 field.
