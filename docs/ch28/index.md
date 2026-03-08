<<<<<<< HEAD
# Chapter Overview

This chapter covers **Number Theory**.

# Reference

[Introduction to Algorithms (CLRS), Chapter 31](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
# Chapter 28: Neural ODEs

Neural Ordinary Differential Equations treat neural networks as continuous-time dynamical systems, generalizing residual networks to infinite depth by defining transformations through ODEs solved with numerical integrators. This framework enables adaptive computation, memory-efficient training via the adjoint method, and natural handling of irregularly-sampled data. This chapter covers the mathematical foundations, continuous normalizing flows, advanced extensions including Neural SDEs and latent ODEs, evaluation methodology, and applications to quantitative finance.

---

## Neural ODE Foundations

- [ODE Fundamentals](neural_ode_foundations/fundamentals.md) -- Ordinary differential equations as continuous-time dynamical systems with numerical integration and the ResNet-ODE connection.
- [Neural ODEs Overview](neural_ode_foundations/neural_ode_models_overview.md) -- Comprehensive tutorial package covering continuous-depth networks, adjoint methods, and generative modeling.
- [Getting Started](neural_ode_foundations/getting_started.md) -- Quick start guide with installation, torchdiffeq setup, and a first Neural ODE example.
- [Module Index](neural_ode_foundations/module_index.md) -- Complete overview of all tutorial modules from beginner ODE basics to advanced applications.
- [Adjoint Sensitivity Method](neural_ode_foundations/adjoint.md) -- Memory-efficient gradient computation by solving a backward ODE instead of storing all activations.
- [Numerical Solvers](neural_ode_foundations/numerical_solvers.md) -- Euler, Runge-Kutta, and adaptive solvers with their accuracy, cost, and memory tradeoffs.
- [Forward vs Adjoint Sensitivity](neural_ode_foundations/forward_vs_adjoint.md) -- Comparing forward sensitivity and adjoint methods for gradient computation through ODE solves.
- [Computational Cost Analysis](neural_ode_foundations/computational_cost.md) -- Understanding cost structure, number of function evaluations, and practical deployment considerations.

## Continuous Flows

- [Continuous Normalizing Flows](continuous_flows/cnf.md) -- Generalizing normalizing flows from discrete to continuous transformations with the instantaneous change of variables formula.
- [FFJORD](continuous_flows/ffjord.md) -- Making CNFs practical with Hutchinson's stochastic trace estimator for unrestricted architectures.
- [Augmented Neural ODEs](continuous_flows/augmented.md) -- Overcoming topological limitations of standard Neural ODEs by augmenting the state space.

## Advanced Extensions

- [Neural SDEs](advanced/neural_sdes.md) -- Extending Neural ODEs with learnable diffusion terms for modeling stochastic processes and uncertainty.
- [Latent ODEs for Time Series](advanced/latent_odes.md) -- Combining Neural ODEs with variational autoencoders for irregularly-sampled time series modeling.
- [Regularization Techniques](advanced/regularization.md) -- Encouraging simpler dynamics through kinetic energy and Jacobian regularization to reduce computational cost.

## Evaluation

- [Accuracy vs Speed Tradeoffs](evaluation/accuracy_speed.md) -- Fundamental tradeoffs between solver tolerance, number of function evaluations, and solution accuracy.
- [Comparison with Discrete Models](evaluation/comparison.md) -- How Neural ODEs compare with ResNets and RNNs in accuracy, efficiency, and applicability.
- [Benchmarks](evaluation/benchmarks.md) -- Standardized benchmarking practices across computational, numerical, and practical evaluation dimensions.
- [Solver Selection Guide](evaluation/solver_selection.md) -- Practical guidance on choosing ODE solvers for different application domains and accuracy requirements.

## Finance Applications

- [Continuous Dynamics in Finance](finance_applications/continuous_dynamics.md) -- Neural ODEs for continuous-time financial modeling with irregular time series and physics-informed constraints.
- [Term Structure Dynamics](finance_applications/term_structure.md) -- Modeling yield curve evolution as continuous-time processes learned from market data.
- [Volatility Surface Modeling](finance_applications/volatility_surface.md) -- Modeling implied volatility surface dynamics as continuous-time processes respecting no-arbitrage constraints.
>>>>>>> 96f31bd (...)
