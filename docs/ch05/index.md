# Chapter Overview

<<<<<<< HEAD
This chapter covers **Stacks Queues and Deques**.

# Reference

[Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
This chapter covers the complete PyTorch training pipeline from data loading through model deployment. Each section addresses a critical stage: datasets and transforms, data loaders, the training loop, loss functions, optimizers, learning rate schedulers, overfitting diagnostics, evaluation metrics, hyperparameter tuning, and model saving. Together, these components form the end-to-end workflow for training, evaluating, and deploying deep learning models.

## Datasets

Loading, transforming, and augmenting data with PyTorch's Dataset abstraction.

- [Datasets Overview](datasets/datasets_overview.md) -- Tutorial package covering 16 Dataset and DataLoader examples
- [Dataset Class](datasets/dataset_class.md) -- The `torch.utils.data.Dataset` interface and its two core methods
- [Built-in Datasets](datasets/builtin_datasets.md) -- Ready-to-use datasets from torchvision, torchaudio, and torchtext
- [Custom Datasets](datasets/custom_datasets.md) -- Implementing datasets for CSV, images, and domain-specific data formats
- [Data Transforms](datasets/data_transforms.md) -- Transform pipelines with `torchvision.transforms` and `Compose`
- [Custom Transforms](datasets/custom_transforms.md) -- Writing domain-specific transforms as callable classes
- [Augmentation Basics](datasets/augmentation_basics.md) -- Random transformations for regularization and invariance learning

## DataLoaders

Batching, shuffling, sampling, and parallel loading for efficient training.

- [DataLoaders Overview](dataloaders/dataloaders_overview.md) -- Complete tutorial package from fundamentals to distributed training
- [Quick Start](dataloaders/quickstart.md) -- Get started with DataLoader in five minutes
- [Package Summary](dataloaders/package_summary.md) -- Overview of the tutorial package contents and structure
- [DataLoader Basics](dataloaders/dataloader_basics.md) -- Core interface, parameter overview, and basic iteration patterns
- [Batching Strategies](dataloaders/batching.md) -- Batch size effects on training dynamics, convergence, and generalization
- [Shuffling and Sampling](dataloaders/shuffling_sampling.md) -- Randomizing sample order and using built-in samplers
- [Custom Samplers](dataloaders/custom_samplers.md) -- Implementing the `Sampler` interface for application-specific strategies
- [Weighted Sampling](dataloaders/weighted_sampling.md) -- `WeightedRandomSampler` for handling class imbalance
- [Collate Functions](dataloaders/collate_functions.md) -- Merging samples into batches for variable-length and nested data
- [Multi-Process Loading](dataloaders/multiprocess_loading.md) -- Using `num_workers` to parallelize data preparation
- [Memory Pinning](dataloaders/memory_pinning.md) -- Page-locked memory for accelerated CPU-to-GPU transfers

## Training

The training loop, validation, logging, and training techniques.

- [Training Overview](training/tensorboard_and_logging_overview.md) -- MNIST training with TensorBoard visualization tutorial
- [Quick Start](training/quickstart.md) -- Five-minute installation and first training run
- [Training Loop](training/training_loop.md) -- The standard forward-backward-update pattern in PyTorch
- [Validation Loop](training/validation_loop.md) -- Evaluating generalization performance without gradient computation
- [Metrics Tracking](training/metrics_tracking.md) -- Recording losses, accuracies, and gradient statistics over time
- [TensorBoard](training/tensorboard.md) -- Interactive visualization with `SummaryWriter`
- [Weights & Biases](training/wandb.md) -- Cloud-hosted experiment tracking, sweeps, and collaborative dashboards
- [Gradient Clipping](training/gradient_clipping.md) -- Bounding gradient magnitudes to prevent exploding gradients
- [Mixed Precision Training](training/mixed_precision.md) -- FP16/BF16 computation with FP32 master weights for speed and memory savings
- [Reproducibility](training/reproducibility.md) -- Seeding, deterministic operations, and ensuring repeatable results

## Loss Functions

Specialized loss functions beyond the basics covered in Chapter 3.

- [Loss and Optimizer Overview](loss/loss_and_optimizer_overview.md) -- Tutorial package covering loss functions, optimizers, and scheduling
- [Getting Started](loss/getting_started.md) -- Installation and first steps with loss functions in PyTorch
- [Quick Reference](loss/quick_reference.md) -- Essential imports and cheat sheet for common loss/optimizer patterns
- [Loss Selection Guide](loss/loss_selection.md) -- Systematic framework for choosing loss functions by task and data characteristics
- [Focal Loss](loss/focal_loss.md) -- Down-weighting easy examples to focus on hard, misclassified samples
- [Huber Loss](loss/huber_loss.md) -- Smooth L1 loss combining MSE precision with MAE robustness to outliers
- [Hinge Loss](loss/hinge_loss.md) -- Maximum-margin loss for SVM-style classification
- [Custom Loss Functions](loss/custom_loss.md) -- Designing domain-specific objectives for segmentation, multi-task, and beyond

## Optimizers

Parameter update algorithms from vanilla SGD through modern adaptive methods.

- [Optimizers Overview](optimizers/optimizers_overview.md) -- Tutorial package for Adam, RMSprop, and Adagrad implementations
- [Optimizer Overview](optimizers/optimizer_overview.md) -- The optimization problem, PyTorch optimizer interface, and parameter groups
- [SGD](optimizers/sgd.md) -- Vanilla stochastic gradient descent
- [Momentum](optimizers/momentum.md) -- Exponentially decaying gradient average for faster convergence
- [Nesterov Accelerated Gradient](optimizers/nesterov.md) -- Look-ahead gradient computation for improved convergence near optima
- [Adagrad](optimizers/adagrad.md) -- Per-parameter adaptive learning rates based on historical gradients
- [Adadelta](optimizers/adadelta.md) -- Decaying gradient accumulation without requiring an initial learning rate
- [RMSprop](optimizers/rmsprop.md) -- Exponentially decaying average of squared gradients for adaptive rates
- [Adam](optimizers/adam.md) -- Combined momentum and adaptive learning rates with bias correction
- [AdamW](optimizers/adamw.md) -- Decoupled weight decay for proper regularization with adaptive optimizers
- [AMSGrad](optimizers/amsgrad.md) -- Maximum second-moment tracking to fix Adam's convergence issue
- [NAdam](optimizers/nadam.md) -- Nesterov momentum integrated into the Adam framework
- [RAdam](optimizers/radam.md) -- Rectified Adam with variance-aware warmup switching
- [LAMB](optimizers/lamb.md) -- Layer-wise adaptive rates for stable large-batch training
- [L-BFGS](optimizers/lbfgs.md) -- Quasi-Newton method using limited-memory inverse Hessian approximation
- [Optimizer Comparison](optimizers/comparison.md) -- Systematic comparison across convergence, memory, and generalization
- [Selection Guide](optimizers/selection_guide.md) -- Practical decision framework for choosing an optimizer
- [Practical Examples](optimizers/optimizer_examples.md) -- Complete runnable examples for image classification and NLP

## Schedulers

Learning rate scheduling strategies for improved convergence and final model quality.

- [Schedulers Overview](schedulers/learning_rate_schedulers_overview.md) -- Tutorial package overview for scheduling implementations
- [Scheduler Overview](schedulers/scheduler_overview.md) -- Motivation, the fixed-rate tension, and the scheduling landscape
- [Quick Start](schedulers/quickstart.md) -- Two-minute setup and first scheduler example
- [Scheduler Guide](schedulers/scheduler_guide.md) -- Complete guide with comparison table, decision tree, and formulas
- [Combined Guide](schedulers/combined_guide.md) -- Integrating built-in and custom schedulers together
- [Step LR](schedulers/step_lr.md) -- Fixed-interval multiplicative decay
- [Multi-Step LR](schedulers/multistep_lr.md) -- Decay at specified epoch milestones
- [Exponential LR](schedulers/exponential_lr.md) -- Constant multiplicative decay every epoch
- [Cosine Annealing](schedulers/cosine_annealing.md) -- Smooth cosine decay spending more time at low learning rates
- [Warmup Strategies](schedulers/warmup.md) -- Gradual learning rate ramp-up for stable early training
- [ReduceLROnPlateau](schedulers/reduce_on_plateau.md) -- Reactive decay when validation performance stalls
- [OneCycleLR](schedulers/one_cycle.md) -- Smith's 1cycle policy with learning rate and momentum co-scheduling
- [Custom Schedulers](schedulers/custom_schedulers.md) -- `LambdaLR` and the `LRScheduler` base class for domain-specific policies

## Overfitting and Generalization

Diagnosing and addressing overfitting through the bias-variance lens.

- [Overfitting Overview](overfitting/overfitting_bias_variance_overview.md) -- Tutorial package with polynomial regression demonstrations
- [Overfitting and Underfitting](overfitting/overfitting_underfitting.md) -- Definitions, symptoms, and the two failure modes of learning
- [Overfitting Detection](overfitting/overfitting_detection.md) -- Training-validation gap analysis and early warning signals
- [Bias-Variance Tradeoff](overfitting/bias_variance_tradeoff.md) -- The fundamental tension between model simplicity and flexibility
- [Bias-Variance Decomposition](overfitting/bias_variance_decomposition.md) -- Formal decomposition and empirical estimation of bias and variance
- [Mathematical Derivation](overfitting/mathematical_derivation.md) -- Rigorous step-by-step derivation of the decomposition from first principles
- [Cross-Validation](overfitting/cross_validation.md) -- K-fold cross-validation for robust generalization estimates
- [Train-Val-Test Split](overfitting/train_val_test_split.md) -- Three-way splitting for parameter optimization, selection, and evaluation

## Evaluation

Metrics for quantifying classification and regression model performance.

- [Evaluation Overview](evaluation/model_evaluation_metrics_overview.md) -- Comprehensive evaluation package with core modules
- [Metrics Overview](evaluation/metrics_overview.md) -- Choosing metrics aligned with problem objectives
- [Classification Metrics](evaluation/classification_metrics.md) -- Accuracy, precision, recall, F1, and threshold-dependent measures
- [Regression Metrics](evaluation/regression_metrics.md) -- MSE, MAE, R-squared, and outlier sensitivity analysis
- [Confusion Matrix](evaluation/confusion_matrix.md) -- Tabular summary of predictions vs actual labels
- [ROC and AUC](evaluation/roc_auc.md) -- Receiver operating characteristic curves and area under the curve
- [Precision-Recall Curves](evaluation/precision_recall.md) -- PR curves for imbalanced datasets where positives are rare
- [Calibration](evaluation/calibration.md) -- Expected calibration error and reliability diagrams for probability quality

## Hyperparameters

Systematic approaches to hyperparameter search and analysis.

- [Hyperparameter Tuning Overview](hyperparameters/hyperparameter_tuning_overview.md) -- Tutorial package for tuning techniques
- [Quick Start](hyperparameters/quickstart.md) -- Installation and first hyperparameter search
- [Grid Search](hyperparameters/grid_search.md) -- Exhaustive evaluation of all parameter combinations
- [Random Search](hyperparameters/random_search.md) -- Efficient sampling from hyperparameter distributions
- [Bayesian Optimization](hyperparameters/bayesian_optimization.md) -- Surrogate-model-guided search for expensive evaluations
- [Learning Rate Schedules](hyperparameters/lr_schedules.md) -- Learning rate range test and optimal schedule discovery
- [Importance Analysis](hyperparameters/importance_analysis.md) -- Identifying which hyperparameters matter most with fANOVA

## Save and Load

Model persistence, checkpointing, and deployment formats.

- [Save and Load Overview](save_load/save_and_load_models_overview.md) -- Complete tutorial collection for model saving and deployment
- [Quick Start](save_load/quickstart.md) -- Quick setup for model deployment workflows
- [State Dict](save_load/state_dict.md) -- PyTorch's recommended serialization format for portable model persistence
- [Checkpointing](save_load/checkpointing.md) -- Saving complete training state for resumable long-running experiments
- [ONNX Export](save_load/onnx_export.md) -- Cross-framework model format for production deployment
- [TorchScript](save_load/torchscript.md) -- Serializing models for C++ and non-Python execution environments
- [Save Best Model Example](save_load/save_best_hymenoptera.md) -- Complete training pipeline with checkpointing on Hymenoptera dataset
- [Model Deployment Overview](save_load/model_deployment_overview.md) -- Complete deployment pipeline: serialization, APIs, and containerization
- [Deployment Basics](save_load/model_deployment_overview_2.md) -- ONNX conversion, quantization, and inference optimization techniques
>>>>>>> 96f31bd (...)
