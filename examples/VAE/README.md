# VAE - Continuous-latent Variational Autoencoder (MNIST)

A small, CPU-friendly **Variational Autoencoder** (Kingma & Welling 2014,
[*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114)) on MNIST.
This is the **continuous-latent / Gaussian** companion to the discrete
[`examples/VQVAE`](../VQVAE) (which uses a vector-quantized codebook
bottleneck). Here the bottleneck is a diagonal Gaussian posterior and the
latent is **sampled with the reparameterization trick** using the repository
layer [`TNNetGaussianReparameterize`](../../neural/neuralnetwork.pas).

## What it does

An encoder maps each 28x28 digit to two stacked vectors on the depth axis:
the posterior mean `mu` and log-variance `log_var`. The new layer draws

```
z = mu + sigma * eps,   sigma = exp(0.5*log_var),   eps ~ N(0,1)
```

(`eps` is sampled once per forward and **frozen** for the matching backward,
the same fixed-noise discipline as `TNNetGumbelSoftmax`/`TNNetDropout`). A
decoder reconstructs the image from `z`. Training minimizes

```
loss = reconstruction-MSE  +  beta * KL( q(z|x) || N(0,1) )
```

The KL term pulls the posterior toward a standard-normal prior so that, at
generation time, sampling `z ~ N(0,1)` and decoding yields plausible digits.

### KL composition

`TNNetGaussianReparameterize` contributes **only** the reconstruction
(reparameterization) gradient. The KL term is a **separate** penalty
(`dKL/dmu = mu`, `dKL/dlog_var = 0.5*(exp(log_var)-1)`, also packaged as the
loss head `TNNetVAEKLDivergence`). This example wires the composition
explicitly: the reconstruction gradient flows back through the reparameterize
layer, then a second KL-only backward pass adds the KL gradient at the
`(mu|log_var)` fork. The two gradients **sum** there, exactly as a two-headed
VAE would.

## Run

```
lazbuild --build-mode=Release VAE.lpi
./VAE            # SMOKE: ~1-2 min on one CPU
./VAE --full     # more epochs, sharper output
```

Requires the standard MNIST `*.idx*-ubyte` files in the working directory
(same files every MNIST example uses; copy them from `examples/VQVAE` or any
MNIST example). If they are missing the program prints a hint and exits
cleanly.

## Output

* `vae_reconstructions.png` - top rows = test originals, bottom rows =
  deterministic reconstructions (`z = mu`, sampling disabled).
* `vae_generated.png` - an 8x8 grid of digits decoded from fresh `z ~ N(0,1)`
  prior samples.

Smoke output is rough (reconstruction MSE settles around 0.02 on `[-1,1]`
pixels, KL stays finite, no NaN). Use `--full` for sharper digits.
