# CycleGAN — unpaired image-to-image translation (cycle-consistency)

The repo's first **UNPAIRED** image-to-image translation example
([Zhu et al. 2017](https://arxiv.org/abs/1703.10593)). It is the unpaired
sibling of [Pix2Pix](../Pix2Pix) (which is *paired* supervised: every input has a
known aligned target). Here the two image **domains** are sampled from *separate
independent* random draws, so no per-sample target exists and an L1-to-target
loss is impossible. CycleGAN's trick is the **cycle-consistency** loss:
translating `A -> B -> A` must return to the original `A` (and `B -> A -> B` back
to `B`). That round-trip constraint pins the mapping down without pairs.

Pure CPU, synthetic data generated in code — no dataset download. The default
SMOKE run finishes well under five minutes; `--full` trains longer for a sharper
result.

## The task (red shapes ↔ green shapes)

* **Domain A**: one or two filled shapes (circles / rectangles) drawn in **RED**
  on a dark-blue background.
* **Domain B**: the *same kind* of shapes drawn in **GREEN**.

The two domains are drawn from independent random seeds, so a sample `a ∈ A` and
a sample `b ∈ B` share **no geometry** — they are unpaired. The learnable mapping
is therefore a **recoloring**: `G` turns red into green while keeping the
silhouette, and `F` turns green back into red. RGB is encoded in `[-1, 1]`
(`0 -> -1`, `1 -> +1`) to match each generator's `Tanh` output range.

## The networks

* **Two generators** (`TNNet.AddUNet`, the same builder as
  [UNetSegmentation](../UNetSegmentation) and [Pix2Pix](../Pix2Pix)):
  `G : A -> B` and `F : B -> A`, each `Input(grid,grid,3) -> AddUNet -> Tanh`.
* **Two PatchGAN discriminators** (Isola et al.), one per domain, composed from
  existing conv layers (NO new leaf class). Unlike Pix2Pix the discriminator
  input is the **image alone** (3 channels, no condition stacked in), because
  CycleGAN's adversary judges *domain membership*, not input-output
  consistency. The **least-squares GAN** objective (Mao et al. 2017) is used:
  `D` regresses real → 1 / fake → 0 and each generator is pushed toward 1.

## The loss

Total generator objective (minimised w.r.t. `G` and `F`):

```
L =  adv(D_B, G(a)) + adv(D_A, F(b))                    (fool both discriminators)
   + λ_cyc * ( |F(G(a)) - a|  +  |G(F(b)) - b| )         (round-trip / cycle)
   + λ_id  * ( |G(b)     - b|  +  |F(a)     - a| )         (identity / colour anchor)
```

with `λ_cyc = 10`, `λ_id = 5`. The identity term asks `G` to leave an
already-domain-B image unchanged (and `F` a domain-A image), stabilising colours
early.

### Cycle backprop through composed generators (the part that is new vs Pix2Pix)

For the forward cycle `a -> G -> g -> F -> rec`:

1. `g := G.Compute(a)`; `rec := F.Compute(g)`.
2. Cycle gradient at `F`'s output: `λ_cyc * sign(rec - a)`.
3. `F.Backpropagate(that)` — this **both** updates `F`'s weights on the cycle
   term **and**, because `F`'s input layer has `EnableErrorCollection`, leaves
   `d(cycle)/d(g)` in `F.Layers[0].OutputError`. That input gradient is the cycle
   contribution that must flow into `G`'s output.
4. `G`'s full output error = adversarial grad (from frozen `D_B`) + the cycle
   grad returned from `F` in step 3 + the identity grad. Seed it and
   `G.GetLastLayer().Backpropagate()`.

The backward cycle `b -> F -> G -> rec` is the mirror image. Both directions are
accumulated per step before the single weight-update backprop of each generator;
`SetBatchUpdate(true)` keeps the framework from applying partial updates
mid-accumulation, and `IncDepartingBranchesCnt` / `ResetBackpropCallCurrCnt`
allow the multiple per-step backward passes into each generator.

This is the **faithful** CycleGAN gradient (the cycle term genuinely backprops
through the *composed* `F∘G` / `G∘F`), implemented with sub-gradient `sign(·)`
for the L1 cycle/identity terms rather than autograd. No simplification of the
objective vs canonical CycleGAN; the only differences from the paper are scale
(toy 16×16 recoloring task, tiny U-Nets) and the use of LSGAN, which is itself a
standard CycleGAN choice.

## Output / metrics

Per-eval the program prints:

* **`cycA` / `cycB`** — mean cycle-reconstruction error `|F(G(a))-a|` and
  `|G(F(b))-b|` over a held-out set (down = the unpaired mapping is becoming
  invertible / content-preserving).
* **`transA->B` / `transB->A`** — translation-colour score: the fraction of
  source-shape pixels whose translated dominant colour matches the *target*
  domain (red→green should go green, green→red should go red).
* **`lossD_A` / `lossD_B`** — the two LSGAN discriminator losses.

It also renders an ASCII panel (`a | G(a) | F(G(a))` and
`b | F(b) | G(F(b))`) and writes a PPM strip to `cyclegan_sample.ppm`.

## Usage

```
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 CycleGAN.lpr
./CycleGAN            # smoke run (fast, default)
./CycleGAN --full     # longer training for a sharper result
```

The default SMOKE run (16×16, depth 2, 64 train / 32 test samples, 10 epochs,
~3.5 min on CPU) drives held-out cycle error from `≈0.68` down to `cycA≈0.05`
/ `cycB≈0.06` and both translation scores above `≈0.93` (`transA->B≈0.99`,
`transB->A≈0.93`).
