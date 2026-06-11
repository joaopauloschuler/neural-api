# DeepSeekMoE: shared + fine-grained experts with aux-loss-free balancing

Demonstrates `TNNet.AddDeepSeekMoE`, a DeepSeekMoE feed-forward block with:

* **Shared experts** — always active, added with no routing weight, so the
  routed experts are free to specialize.
* **Fine-grained routed experts** — express the paper's m-split by passing a
  large `NumRoutedExperts` with a small `ExpertHiddenDim` and
  `ActiveRouted = k*m` (same FLOP budget, far more expert combinations).
* **Auxiliary-loss-free load balancing** (`TNNetBiasBalancedTopKGate`,
  enabled by `BalanceBiasSpeed > 0`) — a per-expert bias is added to the
  router scores **for top-k selection only** (combine weights stay unbiased)
  and is nudged once per batch by `UpdateRoutingBias()`:
  `b_e := b_e - speed * sign(load_e - mean_load)`. No Switch-style auxiliary
  loss, so no interference gradient on the task loss.

The demo builds the same block twice with a deliberately skewed router
(expert 0 dominates) — once with a plain top-k gate and once with the
bias-balanced gate — runs 30 batches calling `UpdateRoutingBias()` on the
balanced net, and prints the per-expert load histograms: the unbalanced net
stays collapsed on expert 0 while the balanced net's histogram flattens.

Build and run (finishes in seconds, pure CPU):

```
lazbuild examples/DeepSeekMoE/DeepSeekMoE.lpi
bin/x86_64-linux/bin/DeepSeekMoE
```

## References

* Dai et al. 2024, *DeepSeekMoE: Towards Ultimate Expert Specialization in
  Mixture-of-Experts Language Models*, arXiv:2401.06066.
* Wang et al. 2024, *Auxiliary-Loss-Free Load Balancing Strategy for
  Mixture-of-Experts*, arXiv:2408.15664 (deployed in DeepSeek-V3,
  arXiv:2412.19437).
