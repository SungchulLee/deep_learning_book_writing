# Standard Benchmarks

## Split Benchmarks

**Split MNIST**: 5 tasks of 2 digits each. Very easy (>95% for most methods). Only for sanity checking. **Split CIFAR-100**: 10 or 20 tasks (10 or 5 classes per task). Standard benchmark for comparing methods. **Split ImageNet**: most challenging, requires significant compute.

## Permutation Benchmarks

**Permuted MNIST**: each task applies a fixed random pixel permutation. Same 10 classes across tasks (domain-incremental). Tests ability to handle distinct input distributions.

## Protocols

Task-incremental (task ID at test, per-task accuracy). Class-incremental (no task ID, overall accuracy across all seen classes). Domain-incremental (accuracy on current domain).

## Reporting

Average accuracy after all tasks, backward transfer (forgetting), forward transfer, learning curve (accuracy vs number of tasks), and buffer size (for replay methods).
