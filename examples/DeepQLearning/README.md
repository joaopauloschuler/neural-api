# Deep Q-Learning (DQN) on a grid-world

The suite's **first reinforcement-learning example**: a minimal but complete
**Deep Q-Network** (DQN; Mnih et al. 2015, *Human-level control through deep
reinforcement learning*) that learns an optimal navigation policy on a tiny,
fully self-contained, deterministic grid-world. The agent, the environment and
all the replay machinery live in the single `.lpr`; the Q-network is composed
from existing dense layers — **no new layer class**.

## The environment (5x5 grid-world)

```
S . . . .      S = start (0,0)
. . . X .      G = goal  (4,4)   reward +1.0, episode ends
. . . . .      X = pit   (1,3),(3,1)  reward -1.0, episode ends
. X . . .      every non-terminal step costs -0.02 (living penalty)
. . . . G      4 actions: up / down / left / right; wall move = no-op
```

The small per-step penalty pressures the agent toward the **shortest** path. The
state is fed to the network as a 25-d **one-hot** of the agent's cell, so inputs
are already well-conditioned (exactly one `1.0`) — important because the manual
`UpdateWeights` path bypasses gradient clipping, so stability relies on
normalised inputs plus a small learning rate.

## The agent (textbook DQN)

* **Q-network**: `25 -> FullConnectReLU(64) -> FullConnectReLU(64) ->
  FullConnectLinear(4)` (one linear output per action = `Q(s, ·)`), ~5.9k params.
* **Experience replay**: a ring buffer of `(s, a, r, s', done)` transitions;
  each learning step samples a random minibatch, decorrelating the updates.
* **Target network**: a frozen copy re-synced every `cTargetSync` env-steps via
  **`CopyWeights`** (not `LoadFromFile`), used to form a stable TD target.
* **Epsilon-greedy** exploration with exponential epsilon decay (`1.0 -> 0.05`).
* **TD update**: `y = r + gamma * max_a' Q_target(s', a')` for non-terminal `s'`,
  else `y = r`. We regress `Q(s,a)` toward `y` for the **taken action only**: the
  training target vector is set equal to the current online `Q(s,·)` output, then
  its taken-action component is overwritten with `y`, so the squared-error
  gradient is exactly zero on the untaken actions.
* **Minibatch gradients** are accumulated with **`SetBatchUpdate(True)`** (the
  per-sample default would zero each neuron's delta), then applied with one
  `UpdateWeights` + `ClearDeltas` per batch.

## Headline

A moving-average learning curve shows the agent climbing from random toward
optimal, then the **greedy (epsilon=0) policy** is rolled out from the fixed
start and the ASCII trajectory is printed as concrete proof it learned:

```
training 1500 episodes...
  episode | epsilon | MA(50) return | MA(50) steps
        1 |   0.995 |         -1.060 |           4.0
      100 |   0.606 |         -0.780 |          17.0
      200 |   0.367 |          0.076 |          11.2
      ...
     1500 |   0.050 |          0.779 |           8.0

optimal shortest path start->goal avoiding pits = 8 steps (return ~ 0.86)

=== final GREEDY policy rollout ===
  greedy trajectory from start (0,0):  reached goal = TRUE  steps = 8
     S . . . .
     * * . X .
     . * * . .
     . X * . .
     . . * * G

greedy success rate over all 21 non-terminal start cells = 100.0%
```

The return rises from -1.06 (random: hits pits / times out) toward ~0.78, steps
drop from ~17 to the **optimal 8**, the greedy rollout reaches the goal along the
shortest pit-avoiding path, and the learned greedy policy succeeds from **100% of
the 21 non-terminal start cells**.

## Build & run

```
lazbuild DeepQLearning.lpi
stdbuf -oL -eL ../../bin/x86_64-linux/bin/DeepQLearning
```

Pure CPU, tiny net + modest replay buffer, ~32 s on two cores. No binary files.
