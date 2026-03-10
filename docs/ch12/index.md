# Chapter Overview


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

This chapter covers continual learning (also known as lifelong or incremental learning), which addresses the fundamental challenge of training neural networks on a sequence of tasks without catastrophically forgetting previously learned knowledge. We examine the stability-plasticity dilemma, formalize different learning scenarios, and survey the major families of methods -- regularization, replay, architecture, and distillation -- along with standard benchmarks for evaluation.

---

## Foundations

- Continual Learning Overview -- Comprehensive introduction to continual learning concepts, mathematical frameworks, and main approaches
- Usage Guide -- Practical quick-start guide with installation instructions and script organization
- [Catastrophic Forgetting](continual_learning/catastrophic_forgetting.md) -- Rigorous treatment of the forgetting phenomenon, including mathematical formalization and historical context
- [Evaluation Metrics](continual_learning/evaluation_metrics.md) -- Standard metrics including the accuracy matrix, backward/forward transfer, and forgetting measures
- Taxonomy -- Classification of methods by strategy (regularization, replay, architecture, distillation) and information access
- Stability-Plasticity Dilemma -- The fundamental tension between retaining old knowledge and learning new tasks

## Learning Scenarios

- Task-Incremental Learning -- Simplest scenario where task identity is provided at both training and test time
- Class-Incremental Learning -- Most challenging scenario requiring classification among all classes seen so far without task identifiers
- Online Continual Learning -- Most restrictive setting where data arrives as a stream and each sample is seen only once

## Regularization Methods

- [Overview](regularization_methods/overview.md) -- Introduction to regularization-based continual learning through parameter importance estimation
- [Elastic Weight Consolidation (EWC)](regularization_methods/ewc.md) -- Protecting important parameters using the Fisher information matrix with Bayesian foundations
- [Online EWC](regularization_methods/online_ewc.md) -- Extension of EWC with continuous online updating of parameter importance estimates
- Synaptic Intelligence (SI) -- Tracking parameter importance during training based on contribution to loss reduction
- Memory Aware Synapses (MAS) -- Task-agnostic importance estimation based on output sensitivity to parameter changes
- [Comparison of Regularization Methods](regularization_methods/comparison.md) -- Detailed comparison of EWC, SI, and MAS including strengths, weaknesses, and applicability

## Replay Methods

- Overview -- Introduction to replay-based continual learning through exemplar storage and rehearsal strategies
- [Experience Replay](replay_methods/experience_replay.md) -- Maintaining and rehearsing a memory buffer of examples from previous tasks
- Generative Replay -- Using a generative model to produce pseudo-samples instead of storing raw data
- Gradient Episodic Memory (GEM) -- Using stored examples as gradient constraints rather than for direct replay
- [A-GEM](replay_methods/agem.md) -- Averaged GEM with efficient gradient projection using average reference gradients
- Dark Experience Replay -- Learning task-specific weighting schemes for replayed data with dynamic priority adjustment

## Architecture Methods

- Overview -- Introduction to architecture-based methods using task-specific modules and dynamic expansion
- Progressive Neural Networks -- Preventing forgetting by freezing old columns and adding new capacity with lateral connections
- PackNet -- Packing multiple tasks into a single network through iterative pruning, retraining, and freezing
- Dynamically Expandable Networks (DEN) -- Combining selective retraining, dynamic expansion, and neuron duplication
- [Expert Gate](architecture_methods/expert_gate.md) -- Routing data to specialized expert modules via a learned gating network
- [Supermask in Superposition](architecture_methods/supermask.md) -- Binary masks selecting task-specific subnetworks from a single frozen base network

## Distillation Methods

- Overview -- Introduction to knowledge distillation for continual learning through soft target constraints
- [Learning Without Forgetting (LwF)](distillation_methods/lwf.md) -- Preserving old task performance by matching the model's outputs to its own previous predictions
- LUCIR -- Addressing classifier bias in class-incremental learning through cosine normalization and margin ranking
- PODNet -- Constraining intermediate features at every layer using pooled distillation loss
- [Feature Distillation](distillation_methods/feature_distillation.md) -- Preserving learned representations by matching intermediate layer features across task sequences
- [Comparison of Distillation Methods](distillation_methods/comparison.md) -- Comparative analysis of response-based, feature-based, attention-based, and relation-based distillation

## Benchmarks

- Benchmark Comparison -- Comprehensive comparison of continual learning methods across standard benchmarks
- Standard Benchmarks -- Split MNIST, Split CIFAR-100, Split ImageNet, and permutation benchmarks
- Stream Benchmarks -- CORe50 and online continual learning benchmarks with gradual distribution shifts
- Evaluation Protocols -- Task-incremental vs class-incremental evaluation, metrics, and controlled comparison guidelines
