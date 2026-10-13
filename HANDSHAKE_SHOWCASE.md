# Handshake showcase entry (copy and paste)

## Title
PINN Lab: Physics-Informed Neural Networks Built from Scratch

## One-line summary
An interactive web app where neural networks learn the laws of motion, predicting a damped oscillator far beyond their training data and recovering its hidden physical constants from noisy measurements.

## Description
Standard neural networks only know the data they've seen. As a physics major, I wanted to show what happens when you also make a network obey a differential equation.

I built PINN Lab, a browser app that trains physics-informed neural networks (PINNs) live, using an automatic-differentiation engine I wrote from scratch in JavaScript with no ML libraries.

**Experiment 1: Predict beyond the data.** Two identical networks see 10 samples from the start of a damped spring's motion. The standard network fits the data but fails afterwards (141% error). The physics-informed network, which is also penalised for violating ẍ + 2δẋ + ω₀²x = 0, predicts the full motion with 0.9% error.

**Experiment 2: Discover hidden physics.** Given only noisy position readings, the network treats the damping and stiffness constants as trainable parameters and recovers them within about 2% of their true values. This is an inverse problem.

**Technical highlights**
- Second-order Taylor-mode forward pass returns u, u̇ and ü in one pass. I hand-derived the backpropagation through it.
- Verified on every page load by a finite-difference gradient check (max relative error ~6 × 10⁻¹⁰).
- Adam optimiser with learning-rate decay, 64-bit floats, about 550 training steps per second in the browser.
- Studied when the inverse problem stops being identifiable: with heavy damping and strong noise, the data alone can't pin the constants down.

## Skills to tag
Machine Learning · Neural Networks · Scientific Computing · Physics · Differential Equations · JavaScript · Numerical Methods · Data Visualization · Optimization

## Links
- Live demo: <paste your shared artifact link>
- Code: <paste your GitHub repo link>

---

## Short version (if there's a character limit)
Built PINN Lab, an interactive app that trains physics-informed neural networks live in the browser on an autodiff engine I wrote from scratch. A PINN predicts a damped oscillator with 0.9% error where a standard network fails (141%), and it recovers unknown damping and stiffness from noisy data to within about 2%. Gradients verified by finite differences to 10⁻⁹.
