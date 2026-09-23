# TestNeuralRL

Validation suite for the `neuralrl` unit (classical reinforcement learning).
Exercises every public component — environment, replay buffer, DQN trainer,
policy-gradient trainer, and both built-in environments — with deterministic
and learned-policy tests. All networks are tiny MLPs composed from existing
`TNNet` layers; **no new layer class**.

## Tests

| # | Test | What it verifies |
|---|------|-----------------|
| 1 | GridWorld deterministic path | Environment mechanics: 4 steps right → goal, reward = 7 |
| 2 | CartPole episode terminates | Physics: pole falls within 500 steps under a fixed action |
| 3 | ReplayBuffer | Add / Sample / capacity-cap semantics |
| 4 | DQN on GridWorld | Q-network learns a greedy policy that reaches the goal (400 episodes) |
| 5 | PolicyGradient on GridWorld | REINFORCE shows reward improvement over training (500 episodes) |

## The environments

* **GridWorld** (5×5): start (0,0), goal (1,0) for PG / (4,0) for DQN, pit
  (2,2). State = `[x, y, 0, 0]` (4-d), 4 actions (up/right/down/left),
  stochastic slip optional. Goal reward +10, pit −10, step cost −1.
* **CartPole-v1**: standard Gym parameters (M=1, m=0.1, L=0.5, g=9.8,
  dt=0.02). State = `[x, ẋ, θ, θ̇]`, 2 actions (push left/right).
  Episode ends at |x|>2.4 or |θ|>12° or 500 steps.

## The trainers

* **`TNeuralDQNTrainer`**: ε-greedy (1.0→0.05), target network (hard update
  every 50 steps), Double-DQN option, per-step gradient updates.
  Q-net: `Input(1,1,4) → FC-ReLU(32) → FC-ReLU(32) → FC-Linear(4)`.
* **`TNeuralPolicyGradTrainer`**: REINFORCE with ε-greedy exploration,
  raw-return advantage (scaled ×0.05, clamped ±0.5), probability floor 0.1.
  Policy net: `Input(1,1,4) → FC-ReLU(32) → FC-ReLU(32) → SoftMax(4)`.

Both trainers use the same three-call update pattern as `neuraldpo.pas`:
`Compute` → `Backpropagate(pseudoTarget)` → `UpdateWeights`.

## Build & run

```
lazbuild TestNeuralRL.lpi
../../bin/x86_64-linux/bin/TestNeuralRL
```

## Expected output

```
=== TestNeuralRL ===

Test 1: GridWorld deterministic path
[PASS] GridWorld reaches goal  steps=4, final=(4,0), reward=7.00

Test 2: CartPole episode terminates
[PASS] CartPole episode terminates  steps=8, total_reward=8.0

Test 3: ReplayBuffer
[PASS] ReplayBuffer size  size=50
[PASS] ReplayBuffer sample  batch=10
[PASS] ReplayBuffer capacity cap  size=100

Test 4: DQN on GridWorld (400 episodes)
  DQN episode 50: reward=3.00  eps=0.778
  DQN episode 100: reward=5.00  eps=0.606
  ...
  DQN episode 400: reward=7.00  eps=0.135
[PASS] DQN GridWorld learns  final_reward=7.00, pos=(4,0), done=True

Test 5: PolicyGradient on GridWorld (500 episodes)
  PG episode 50: reward=-30.00
  PG episode 100: reward=2.00
  ...
  PG episode 500: reward=-10.00
[PASS] PG GridWorld learns  early_avg=-16.26, late_avg=-15.90 (improvement=0.36)

=================================
Total: 7  Passed: 7  Failed: 0

ALL TESTS PASSED
```

Pure CPU, tiny nets, finishes in under 2 minutes. No binary files.
