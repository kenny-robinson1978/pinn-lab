# PINN Lab

**Physics-informed neural networks, trained live in the browser on an autodiff engine written from scratch.**

PINN Lab shows why adding physics to a neural network matters. A damped harmonic oscillator,

```
ẍ + μẋ + kx = 0,   x(0) = 1,  ẋ(0) = 0
```

is used for two experiments:

| Experiment | Question | Result |
|---|---|---|
| **01 · Predict beyond the data** | Two identical networks see the same 10 samples from the first 0.4 s. One is also trained to obey the ODE. Which predicts the rest of the motion? | After 9,000 steps: standard NN has **141.5%** relative L2 error over the full motion. PINN has **0.9%**. |
| **02 · Discover hidden physics** | Given 40 noisy position readings (σ = 0.05), can the network work out the unknown μ and k? | Starting from a guess of μ = 0, k = 100, it reaches μ = 4.05 (true 4.0) and k = 409 (true 400) in 4,000 steps. |

## How it works

### Loss function

```
L(θ) = 1/N Σ (u_θ(tᵢ) − yᵢ)²  +  λ · 1/M Σ (ü_θ + μ u̇_θ + k u_θ)² |_{t_j}
         └── data misfit ──┘          └── ODE residual at M check points ──┘
```

The physics term is evaluated at check points across the whole time axis, including where there is no data. In the inverse problem, μ and k are extra trainable parameters (μ = 10·θ₁, k = 100·e^θ₂) and get their gradients from the residual term.

### The engine (no ML libraries)

* **Taylor-mode forward pass.** The physics loss needs u̇ and ü of the network with respect to its input. Instead of nested autodiff, each layer propagates the triple (a, ȧ, ä):
  ```
  z = W a_prev + b,  ż = W ȧ_prev,  z̈ = W ä_prev
  a = tanh z,  s = 1 − a²,  ȧ = s ż,  ä = s z̈ − 2 a s ż²
  ```
  One forward pass returns u, u̇ and ü.
* **Hand-derived reverse mode.** Backpropagation through the Taylor pass. The tanh step maps upstream gradients (ḡ_a, ḡ_ȧ, ḡ_ä) to
  ```
  ḡ_z  = ḡ_a s − 2 a s ż ḡ_ȧ − (2 a s z̈ + 2 ż² s (s − 2a²)) ḡ_ä
  ḡ_ż  = s ḡ_ȧ − 4 a s ż ḡ_ä
  ḡ_z̈  = s ḡ_ä
  ```
* **Gradient check.** On every page load, all analytic gradients (including the physical constants) are compared with central finite differences. Typical max relative error: **6 × 10⁻¹⁰**.
* **Training.** 1→32→32→32→1 tanh MLP, Xavier init, Adam, learning rate decayed from 3×10⁻³ to 10⁻⁴ over 12k steps, float64. About 550 steps/s for both networks together on a laptop CPU.

### Things I learned

* **Scale the residual.** Raw ODE residuals are about ω₀² ≈ 400 times larger than the data term. Dividing by ω₀² makes λ ≈ 1–10 a sensible weight.
* **Learning-rate decay matters.** With a constant learning rate, Adam reached 0.4% error and then drifted back to 3%. Exponential decay made convergence steady.
* **Identifiability is physics, not a bug.** With heavy damping (μ = 7), σ = 0.1 noise and 30 points, the PINN settles on k ≈ 273 instead of 250. Its curve fits the noisy data *better* than the true curve (MSE 0.0057 vs 0.0074) while satisfying the ODE to 10⁻⁵. The oscillation dies out early, so the data alone cannot distinguish the two.

## Run it

It is a single HTML file with no build step and no dependencies. Open `index.html` in any modern browser.

## Reference

M. Raissi, P. Perdikaris, G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics* 378 (2019) 686–707.
