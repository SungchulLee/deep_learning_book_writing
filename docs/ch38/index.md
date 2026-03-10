# Chapter 38: Adversarial Robustness


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

This chapter provides a comprehensive treatment of adversarial robustness in deep learning, covering the discovery and theory of adversarial examples, attack methods ranging from white-box to physical-world scenarios, defense mechanisms including adversarial training and certified robustness, and evaluation best practices. Special emphasis is placed on financial applications where adversarial threats manifest as market manipulation, fraud evasion, and model extraction attacks.

---

## Foundations

Core theory, threat models, and mathematical framework for adversarial robustness.

- [Introduction to Adversarial Robustness](foundations/introduction.md) -- Discovery of adversarial examples, hypotheses for their existence, and implications for deep learning
- [Threat Models](foundations/threat_models.md) -- Formalizing adversary knowledge, capabilities, and objectives across white-box, black-box, and physical settings
- [Robustness Definitions](foundations/definitions.md) -- Pointwise robustness, minimum adversarial perturbation, and reusable PyTorch attack base classes
- [Perturbation Types](foundations/perturbations.md) -- L-infinity, L-2, and L-1 norm constraints with geometric properties and projection operators

## White-Box Attacks

Gradient-based attacks with full access to model parameters and architecture.

- [Fast Gradient Sign Method (FGSM)](white_box/fgsm.md) -- The foundational single-step gradient attack motivated by the linear hypothesis
- [Projected Gradient Descent (PGD)](white_box/pgd.md) -- Multi-step iterative attack that is the de facto standard for robustness evaluation
- [DeepFool](white_box/deepfool.md) -- Geometric attack finding the minimal perturbation to cross the nearest decision boundary
- [Carlini-Wagner (C&W) Attack](white_box/cw_attack.md) -- Optimization-based attack reformulating adversarial generation as unconstrained minimization
- [AutoAttack](white_box/autoattack.md) -- Parameter-free ensemble attack for reliable robustness evaluation without hyperparameter tuning

## Black-Box Attacks

Attacks without direct access to model internals, using only query responses or transferability.

- [Transfer Attacks](black_box/transfer.md) -- Zero-query attacks exploiting cross-model transferability of adversarial perturbations
- [Query-Based Attacks](black_box/query_based.md) -- Iterative attacks using model queries with score-based and decision-based variants
- [Score-Based Attacks](black_box/score_based.md) -- Exploiting output probability distributions for gradient estimation without model access
- [Decision-Based Attacks](black_box/decision_based.md) -- Attacks operating with only hard-label predictions in the most restrictive black-box setting

## Physical-World Attacks

Adversarial perturbations that survive real-world deployment conditions.

- [Adversarial Patches](physical/patches.md) -- Localized, physically printable perturbations that fool classifiers in the real world
- [3D Adversarial Attacks](physical/3d_attacks.md) -- Perturbations to object shape, texture, and lighting that are adversarial across viewpoints
- [Real-World Robustness](physical/real_world.md) -- Bridging the digital-to-physical gap with expectation over transformations (EOT)

## Adversarial Training

Training-time defenses that augment the learning process with adversarial examples.

- [Standard Adversarial Training](adversarial_training/standard.md) -- PGD-based robust optimization as the most effective defense against adversarial attacks
- [TRADES](adversarial_training/trades.md) -- Theoretically principled trade-off between clean accuracy and adversarial robustness
- [Free Adversarial Training](adversarial_training/free_at.md) -- Achieving comparable robustness at near standard training cost by recycling gradients
- [Fast Adversarial Training](adversarial_training/fast_at.md) -- Single-step FGSM training with random initialization to prevent catastrophic overfitting
- [MART](adversarial_training/mart.md) -- Misclassification-aware training that focuses defense effort on hard examples

## Certified Defenses

Provable robustness guarantees that hold against all possible attacks within a radius.

- [Randomized Smoothing](certified/randomized_smoothing.md) -- Transforming any classifier into a certifiably robust one via Gaussian noise averaging
- [Interval Bound Propagation (IBP)](certified/ibp.md) -- Certified L-infinity robustness by propagating interval bounds through network layers
- [CROWN](certified/crown.md) -- Convex relaxation-based certification with tighter bounds than IBP via linear relaxations
- [Lipschitz-Constrained Networks](certified/lipschitz.md) -- Certified robustness through explicit Lipschitz constant constraints on network sensitivity

## Detection Methods

Identifying adversarial inputs before they reach the classifier.

- [Statistical Detection](detection/statistical.md) -- Detecting adversarial examples via Mahalanobis distance and activation distribution analysis
- [Feature Squeezing](detection/feature_squeezing.md) -- Comparing predictions on original vs. reduced-complexity inputs to flag adversarial perturbations
- [Input Transformation Defenses](detection/input_transform.md) -- Preprocessing with JPEG compression, randomized resizing, and denoising to purify inputs

## Evaluation

Best practices for honest adversarial robustness assessment.

- [Adaptive Attacks and Gradient Masking](evaluation/adaptive.md) -- Detecting and circumventing defenses that obscure gradients rather than providing true robustness
- [Evaluation and Benchmarking](evaluation/benchmarks.md) -- Standardized evaluation protocols including AutoAttack and RobustBench
- [Certified Accuracy](evaluation/certified_accuracy.md) -- Provable lower bounds on robustness independent of attack strength

## Finance Applications

Adversarial robustness in financial machine learning systems.

- [Market Manipulation Detection](finance/manipulation.md) -- Detecting spoofing, wash trading, and strategic signal manipulation as adversarial attacks
- [Fraud Detection Robustness](finance/fraud.md) -- Building fraud detectors resilient to adaptive adversaries who evolve evasion strategies
- [Model Security in Financial Systems](finance/security.md) -- End-to-end security including model extraction, data poisoning, and secure deployment

## Additional Resources

Supplementary materials and code examples.

- Adversarial Robustness Overview -- Module overview covering attacks, defenses, and evaluation techniques
- [Adversarial Attacks on NLP Models](adversarial/nlp_adversarial.md) -- Character-level, word-level, and sentence-level attacks on text-based financial models
