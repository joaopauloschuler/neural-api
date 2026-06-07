# HamiltonianPendulum — structure-preserving (symplectic) learned dynamics

Learns the dynamics of an ideal (undamped) pendulum from **noisy** phase-space
samples with a `TNNetHamiltonianCell`, then rolls the model out autoregressively
for a long horizon and measures how well it conserves energy. This is the first
layer in the suite that is **energy-conserving by construction** (Hamiltonian
Neural Networks, Greydanus, Dzamba & Yosinski 2019, arXiv:1906.01563).

## The idea

A plain learned dynamics model regresses the next state (or a residual field)
directly; nothing forces it to conserve anything, so a long free-swing rollout
slowly **drifts** off its energy level set (spiralling in or blowing up). The
Hamiltonian cell instead parameterizes a *scalar* learned Hamiltonian
`H_theta(q,p)` with a small inner MLP and takes a **symplectic** step from the
gradient of that scalar energy:

```
dq/dt = +dH/dp        dp/dt = -dH/dq
```

The conserved quantity falls out of the geometry of the update, so the rollout
stays near its initial energy.

The true system here (unit mass / length / gravity) is

```
H(q,p) = 0.5*p^2 + (1 - cos q)      dq/dt = p,  dp/dt = -sin q
conserved energy  E = 0.5*p^2 + (1 - cos q)
```

## What the cell does

Over a `(T,1,2*D)` phase sequence (the Depth axis is the `q | p` pair, `D`
coords each), it maps each time-step's phase vector forward by integrating
`Steps` sequential **symplectic-Euler** sub-steps of size `dt`:

```
p_mid = p - dt*dH/dq(q, p)          (field eval at (q,p))
q_new = q + dt*dH/dp(q, p_mid)      (field eval at (q,p_mid))
```

The field `dH/dz = W1^T*(W2 (*) (1 - tanh^2(W1*z + b1)))` is one backward sweep
through the inner MLP. Because the forward pass already needs this first
derivative, the training backward pass differentiates **through** it — a
Hessian-vector product of `H`, implemented as a second tape pass over the same
MLP (no Hessian is ever materialized).

## Contrast arm

An unconstrained NeuralODE-style residual MLP field `z_new = z + f_theta(z)` of
identical hidden width and the same smooth (tanh) activation is trained on the
exact same one-step regression. It has no symplectic / energy structure.

## Running

```
lazbuild examples/HamiltonianPendulum/HamiltonianPendulum.lpi
./bin/x86_64-linux/bin/HamiltonianPendulum
```

Pure CPU, ~30 s, tiny memory footprint.

## Headline result (seed 424242)

```
one-step MSE  HNN = 0.000441   free MLP = 0.000457   (noise floor ~ 0.000400)

energy drift over 800 autoregressive steps (lower = better):
  true leapfrog : end |dE|=0.00189   max |dE|=0.00213   (reference)
  HNN (symplectic): end |dE|=0.11390   max |dE|=0.15949
  free MLP field  : end |dE|=0.62376   max |dE|=0.63201
```

Both models fit the one-step transition equally well (down to the observation
noise floor), yet over an 800-step free-swing rollout the **Hamiltonian cell
conserves energy ~4x better** than the unconstrained MLP field, which drifts off
its level set. (Exact numbers print at runtime.)
