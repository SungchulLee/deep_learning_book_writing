# Chapter Overview

<<<<<<< HEAD
This chapter covers **Efficient Sorting**.

# Reference

[Introduction to Algorithms (CLRS), Chapters 2, 7](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
This chapter covers continual learning (also known as lifelong or incremental learning), which addresses the fundamental challenge of training neural networks on a sequence of tasks without catastrophically forgetting previously learned knowledge. We examine the stability-plasticity dilemma, formalize different learning scenarios, and survey the major families of methods -- regularization, replay, architecture, and distillation -- along with standard benchmarks for evaluation.

---

## Foundations

- [Continual Learning Overview](continual_learning/continual_learning_overview.md) -- Comprehensive introduction to continual learning concepts, mathematical frameworks, and main approaches
- [Usage Guide](continual_learning/usage_guide.md) -- Practical quick-start guide with installation instructions and script organization
- [Catastrophic Forgetting](continual_learning/catastrophic_forgetting.md) -- Rigorous treatment of the forgetting phenomenon, including mathematical formalization and historical context
- [Evaluation Metrics](continual_learning/evaluation_metrics.md) -- Standard metrics including the accuracy matrix, backward/forward transfer, and forgetting measures
- [Taxonomy](continual_learning/taxonomy.md) -- Classification of methods by strategy (regularization, replay, architecture, distillation) and information access
- [Stability-Plasticity Dilemma](continual_learning/stability_plasticity.md) -- The fundamental tension between retaining old knowledge and learning new tasks

## Learning Scenarios

- [Task-Incremental Learning](learning_scenarios/task_incremental.md) -- Simplest scenario where task identity is provided at both training and test time
- [Class-Incremental Learning](learning_scenarios/class_incremental.md) -- Most challenging scenario requiring classification among all classes seen so far without task identifiers
- [Online Continual Learning](learning_scenarios/online_continual.md) -- Most restrictive setting where data arrives as a stream and each sample is seen only once

## Regularization Methods

- [Overview](regularization_methods/overview.md) -- Introduction to regularization-based continual learning through parameter importance estimation
- [Elastic Weight Consolidation (EWC)](regularization_methods/ewc.md) -- Protecting important parameters using the Fisher information matrix with Bayesian foundations
- [Online EWC](regularization_methods/online_ewc.md) -- Extension of EWC with continuous online updating of parameter importance estimates
- [Synaptic Intelligence (SI)](regularization_methods/synaptic_intelligence.md) -- Tracking parameter importance during training based on contribution to loss reduction
- [Memory Aware Synapses (MAS)](regularization_methods/mas.md) -- Task-agnostic importance estimation based on output sensitivity to parameter changes
- [Comparison of Regularization Methods](regularization_methods/comparison.md) -- Detailed comparison of EWC, SI, and MAS including strengths, weaknesses, and applicability

## Replay Methods

- [Overview](replay_methods/overview.md) -- Introduction to replay-based continual learning through exemplar storage and rehearsal strategies
- [Experience Replay](replay_methods/experience_replay.md) -- Maintaining and rehearsing a memory buffer of examples from previous tasks
- [Generative Replay](replay_methods/generative_replay.md) -- Using a generative model to produce pseudo-samples instead of storing raw data
- [Gradient Episodic Memory (GEM)](replay_methods/gem.md) -- Using stored examples as gradient constraints rather than for direct replay
- [A-GEM](replay_methods/agem.md) -- Averaged GEM with efficient gradient projection using average reference gradients
- [Dark Experience Replay](replay_methods/dark_er.md) -- Learning task-specific weighting schemes for replayed data with dynamic priority adjustment

## Architecture Methods

- [Overview](architecture_methods/overview.md) -- Introduction to architecture-based methods using task-specific modules and dynamic expansion
- [Progressive Neural Networks](architecture_methods/progressive_networks.md) -- Preventing forgetting by freezing old columns and adding new capacity with lateral connections
- [PackNet](architecture_methods/packnet.md) -- Packing multiple tasks into a single network through iterative pruning, retraining, and freezing
- [Dynamically Expandable Networks (DEN)](architecture_methods/den.md) -- Combining selective retraining, dynamic expansion, and neuron duplication
- [Expert Gate](architecture_methods/expert_gate.md) -- Routing data to specialized expert modules via a learned gating network
- [Supermask in Superposition](architecture_methods/supermask.md) -- Binary masks selecting task-specific subnetworks from a single frozen base network

## Distillation Methods

- [Overview](distillation_methods/overview.md) -- Introduction to knowledge distillation for continual learning through soft target constraints
- [Learning Without Forgetting (LwF)](distillation_methods/lwf.md) -- Preserving old task performance by matching the model's outputs to its own previous predictions
- [LUCIR](distillation_methods/lucir.md) -- Addressing classifier bias in class-incremental learning through cosine normalization and margin ranking
- [PODNet](distillation_methods/podnet.md) -- Constraining intermediate features at every layer using pooled distillation loss
- [Feature Distillation](distillation_methods/feature_distillation.md) -- Preserving learned representations by matching intermediate layer features across task sequences
- [Comparison of Distillation Methods](distillation_methods/comparison.md) -- Comparative analysis of response-based, feature-based, attention-based, and relation-based distillation

## Benchmarks

- [Benchmark Comparison](benchmarks/benchmarks.md) -- Comprehensive comparison of continual learning methods across standard benchmarks
- [Standard Benchmarks](benchmarks/standard_benchmarks.md) -- Split MNIST, Split CIFAR-100, Split ImageNet, and permutation benchmarks
- [Stream Benchmarks](benchmarks/stream_benchmarks.md) -- CORe50 and online continual learning benchmarks with gradual distribution shifts
- [Evaluation Protocols](benchmarks/evaluation_protocols.md) -- Task-incremental vs class-incremental evaluation, metrics, and controlled comparison guidelines
>>>>>>> 96f31bd (...)
