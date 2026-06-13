# DPOFineTune — Direct Preference Optimization on a tiny char LM

Aligns a TinyGPT-style char-level causal transformer with **Direct
Preference Optimization** (DPO, [Rafailov et al. 2023](https://arxiv.org/abs/2305.18290))
using `TNeuralDPOTrainer` from `neural/neuraldpo.pas` — preference
fine-tuning **without a reward model and without RL**.

## What it does

1. **Pretrain** a tiny causal LM (one-hot char `Input(16,1,128) ->
   PointwiseConvLinear(32) -> AddPositionalEmbedding -> 1x
   AddTransformerEncoderBlock(Heads=2, d_ff=32, CausalMask=true) ->
   FullConnectLinear(128) -> SoftMax`) with plain next-char SGD on a corpus
   where the prompt `say: ` continues **equally often** into a "good"
   patterned completion (`ababab...`) and a "bad" noise completion
   (`qzkxwv...`) — so the pretrained model has no preference between the
   styles (DPO loss starts at exactly `ln 2`, margin `0`).
2. **Clone** the pretrained policy into a frozen **reference** net (the KL
   anchor of the DPO objective).
3. **DPO fine-tune** on 6 preference pairs (prompt, chosen=pattern,
   rejected=noise) with

   ```
   loss = -ln sigmoid(beta * ((logpi_c - logref_c) - (logpi_r - logref_r)))
   ```

   Each `Trainer.Step` backpropagates the exact scaled
   `(softmax - onehot)` gradient: positive sign on chosen completion
   tokens, negative on rejected ones, both scaled by
   `sigmoid(-beta*margin)`.
4. **Report** per epoch: the average margin and the preference accuracy
   (fraction of pairs ranked correctly, ties = 0.5) — both climb. Typical
   run (seeded): margin `0.00 -> ~26.7`, accuracy `50% -> 100%`, loss
   `0.6931 -> 0.0003`, in ~3 s pure CPU.

## Built-in self-checks (`Halt(1)` on failure)

* preference accuracy reaches 100% after DPO;
* the average margin rises substantially above its initial value;
* the average DPO loss drops below its `ln 2` starting point;
* a probed reference-net weight is bit-identical after training (frozen).

## See also

* `neural/neuraldpo.pas` — the trainer, including the full error-signal
  derivation for this codebase's softmax-output networks.
* [TinyGPT](../TinyGPT) — the base char-level GPT recipe this example
  miniaturizes.
* [LoRAFineTune](../LoRAFineTune) — parameter-efficient fine-tuning;
  composable with DPO (freeze the trunk, DPO-train only adapters).
