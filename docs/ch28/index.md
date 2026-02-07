# Chapter 27: Neural ODEs

## Overview

Neural Ordinary Differential Equations (Neural ODEs) represent one of the most elegant unifications of deep learning and dynamical systems theory. Rather than stacking discrete layers, a Neural ODE defines a continuous transformation governed by a learned differential equation:

$$\frac{dh}{dt} = f_\theta(h(t), t), \quad h(0) = h_0$$

The output $h(T)$ is obtained not by sequential layer evaluation but by numerically integrating the ODE from time $0$ to $T$. This continuous-depth perspective reveals that **residual networks are Euler discretizations of an underlying ODE**, and taking the continuous limit unlocks adaptive computation, constant-memory training via the adjoint method, and guaranteed invertibility of the learned map.

For quantitative finance, this framework is transformative. Financial dynamics are inherently continuous—asset prices follow stochastic differential equations, interest rates evolve along yield curves, and risk factors propagate through continuous-time models. Neural ODEs provide a principled way to learn these dynamics directly from data while respecting the mathematical structure that classical quantitative finance demands.

## Chapter Structure

### 27.1 Foundations

The foundational sections establish the mathematical and computational machinery required for Neural ODEs.

**ODE Fundamentals** covers the theory of ordinary differential equations as continuous-time dynamical systems, numerical integration methods (Euler, Runge-Kutta, adaptive solvers), the critical connection between ResNets and ODE discretization, and phase portrait analysis for understanding qualitative dynamics. All concepts are implemented from scratch in PyTorch before introducing `torchdiffeq`.

**Adjoint Method** addresses the central computational challenge: how to train Neural ODEs efficiently. Standard backpropagation through ODE solvers requires $O(N)$ memory where $N$ is the number of solver steps. The adjoint sensitivity method reduces this to $O(1)$ by solving an augmented ODE backwards in time, making Neural ODEs practical for long integration horizons and high-dimensional states.

### 27.2 Continuous Flows

The continuous flow sections extend Neural ODEs to density estimation and generative modeling.

**Continuous Normalizing Flows** reformulate normalizing flows in continuous time. Instead of composing discrete invertible transformations, a CNF defines a continuous change of variables governed by an ODE. The key result is the **instantaneous change of variables formula**, which replaces expensive determinant computations with a trace of the Jacobian.

**FFJORD** (Free-Form Jacobian of Reversible Dynamics) makes CNFs practical by using the Hutchinson trace estimator to avoid computing the full Jacobian. This enables free-form architectures—any neural network can serve as the dynamics function without architectural constraints on invertibility.

**Augmented Neural ODEs** address a fundamental limitation: standard Neural ODEs define homeomorphisms and cannot change the topology of the data manifold. By augmenting the state space with extra dimensions, the ODE gains the representational capacity to learn topologically complex transformations while preserving the continuous-time framework.

### 27.3 Finance Applications

**Continuous Dynamics** brings Neural ODEs into quantitative finance, covering continuous-time modeling of asset prices and risk factors, irregular time series handling for tick data and event-driven markets, physics-informed architectures that encode financial structure (no-arbitrage, mean reversion), and latent ODE models for hidden state estimation in financial systems.

## Prerequisites

- **Multivariable calculus:** Derivatives, integrals, chain rule, Jacobians
- **Linear algebra:** Eigenvalues, matrix exponentials, vector fields
- **PyTorch fundamentals:** Tensors, autograd, `nn.Module`, training loops
- **Residual networks:** Skip connections, deep network training (Chapter 8)
- **Normalizing flows:** Change of variables formula, density estimation (Chapter 25)

## Key References

1. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
2. Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*.
3. Dupont, E., Doucet, A., & Teh, Y. W. (2019). Augmented Neural ODEs. *NeurIPS*.
4. Rubanova, Y., Chen, R. T. Q., & Duvenaud, D. (2019). Latent ODEs for Irregularly-Sampled Time Series. *NeurIPS*.
5. Kidger, P. (2022). On Neural Differential Equations. *DPhil Thesis, University of Oxford*.
