# Continual Learning Taxonomy

## By Strategy

**Regularization-based**: $\mathcal{L} = \mathcal{L}_{task} + \lambda\sum_i \Omega_i(\theta_i - \theta_i^*)^2$ (EWC, SI, MAS). **Replay-based**: experience replay, generative replay, gradient replay (GEM, A-GEM). **Architecture-based**: parameter isolation, dynamic expansion, task-specific masks (Progressive Networks, PackNet, HAT). **Distillation-based**: $\mathcal{L} = \mathcal{L}_{task} + \alpha \cdot \text{KL}(p_{old} \| p_{new})$ (LwF, PODNet, LUCIR).

## By Information Access

Task-incremental (task ID at test, easiest): model knows which task head to use. Class-incremental (no task ID): must identify both task and class. Domain-incremental (same classes, shifting domain). Online continual (no task boundaries, streaming data, hardest).
