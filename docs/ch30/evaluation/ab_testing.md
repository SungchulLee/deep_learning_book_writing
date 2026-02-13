# A/B Testing for RecSys

## Overview

A/B testing is the gold standard for evaluating recommender systems in production. It directly measures user behavior under different recommendation algorithms.

## Experimental Design

### Randomization Unit
- **User-level**: each user sees one algorithm consistently → measures user-level effects
- **Session-level**: each session is randomly assigned → more samples but potential inconsistency

### Sample Size

Required sample size depends on the minimum detectable effect (MDE):

$$n = \frac{(z_{\alpha/2} + z_\beta)^2 (\sigma_A^2 + \sigma_B^2)}{\delta^2}$$

where $\delta$ is the MDE, $\sigma^2$ are the variances, and $z$ values correspond to significance and power levels.

### Duration
Run tests for at least 1–2 weeks to account for day-of-week effects. Avoid starting/ending tests around holidays or special events.

## Common Pitfalls

### Network Effects
User A's recommendations may affect User B's behavior (e.g., viral content). This violates the independence assumption. Mitigation: cluster-based randomization.

### Novelty Effect
Users may engage more with a new algorithm simply because it is novel. Mitigation: run tests for longer periods and measure week-over-week trends.

### Peeking
Checking results before the planned end date inflates false positive rates. Mitigation: use sequential testing methods (always-valid p-values) or pre-commit to a fixed end date.

### Multiple Metrics
Testing multiple metrics simultaneously increases the chance of finding a spurious significant result. Mitigation: designate one primary metric (OEC — Overall Evaluation Criterion) and use Bonferroni or FDR correction for secondary metrics.

## Guardrail Metrics

In addition to the primary metric, monitor guardrail metrics that should not degrade: page load time, error rate, revenue, and user satisfaction. If a guardrail metric degrades significantly, the test should be stopped regardless of primary metric performance.
