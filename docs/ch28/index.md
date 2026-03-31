# Chapter 28: Neural ODEs

Standard neural networks transform data through a fixed sequence of discrete layers. **Neural Ordinary Differential Equations** (Neural ODEs) replace this discrete sequence with a continuous-time dynamical system: the network's hidden state evolves according to an ODE parameterized by a neural network. This continuous-depth perspective generalizes residual networks to infinite depth, enables adaptive computation through ODE solver tolerances, and provides memory-efficient training via the **adjoint method**.

The core idea is simple. A residual network computes $\mathbf{h}_{t+1} = \mathbf{h}_t + f(\mathbf{h}_t, \theta)$. In the limit of infinitely many infinitesimal layers, this becomes

$$
\frac{d\mathbf{h}(t)}{dt} = f\!\bigl(\mathbf{h}(t), t, \theta\bigr)
$$

The output is obtained by integrating this ODE from time $t_0$ to $t_1$ using a numerical solver, and gradients are computed by solving a backward ODE (the adjoint equation) without storing intermediate activations.

This chapter covers the mathematical foundations, continuous normalizing flows, advanced extensions including Neural SDEs and latent ODEs, evaluation methodology, and applications to quantitative finance.

---

## Neural ODE Foundations

- [ODE Fundamentals](neural_ode_foundations/fundamentals.md) -- Ordinary differential equations as continuous-time dynamical systems with numerical integration and the ResNet-ODE connection
- Neural ODEs Overview -- Comprehensive tutorial package covering continuous-depth networks, adjoint methods, and generative modeling
- Getting Started -- Quick start guide with installation, torchdiffeq setup, and a first Neural ODE example
- Module Index -- Complete overview of all tutorial modules from beginner ODE basics to advanced applications
- Adjoint Sensitivity Method -- Memory-efficient gradient computation by solving a backward ODE instead of storing all activations
- [Numerical Solvers](neural_ode_foundations/numerical_solvers.md) -- Euler, Runge-Kutta, and adaptive solvers with their accuracy, cost, and memory tradeoffs
- Forward vs Adjoint Sensitivity -- Comparing forward sensitivity and adjoint methods for gradient computation through ODE solves
- Computational Cost Analysis -- Understanding cost structure, number of function evaluations, and practical deployment considerations

## Continuous Flows

- Continuous Normalizing Flows -- Generalizing normalizing flows from discrete to continuous transformations with the instantaneous change of variables formula
- FFJORD -- Making CNFs practical with Hutchinson's stochastic trace estimator for unrestricted architectures
- Augmented Neural ODEs -- Overcoming topological limitations of standard Neural ODEs by augmenting the state space

## Advanced Extensions

- Neural SDEs -- Extending Neural ODEs with learnable diffusion terms for modeling stochastic processes and uncertainty
- [Latent ODEs for Time Series](advanced/latent_odes.md) -- Combining Neural ODEs with variational autoencoders for irregularly-sampled time series modeling
- Regularization Techniques -- Encouraging simpler dynamics through kinetic energy and Jacobian regularization to reduce computational cost

## Evaluation

- Accuracy vs Speed Tradeoffs -- Fundamental tradeoffs between solver tolerance, number of function evaluations, and solution accuracy
- Comparison with Discrete Models -- How Neural ODEs compare with ResNets and RNNs in accuracy, efficiency, and applicability
- Benchmarks -- Standardized benchmarking practices across computational, numerical, and practical evaluation dimensions
- [Solver Selection Guide](evaluation/solver_selection.md) -- Practical guidance on choosing ODE solvers for different application domains and accuracy requirements

## Finance Applications

- Continuous Dynamics in Finance -- Neural ODEs for continuous-time financial modeling with irregular time series and physics-informed constraints
- [Term Structure Dynamics](finance_applications/term_structure.md) -- Modeling yield curve evolution as continuous-time processes learned from market data
- Volatility Surface Modeling -- Modeling implied volatility surface dynamics as continuous-time processes respecting no-arbitrage constraints
