# Evaluation Protocols

## Standard Protocol

Sample 600+ test episodes, report mean accuracy with 95% confidence interval. Common configurations: 5-way 1-shot, 5-way 5-shot, 20-way 1-shot, 20-way 5-shot.

## Backbone Impact

The backbone significantly affects results on mini-ImageNet 5-way 1-shot: Conv-4 (~113K params, 45-55%), ResNet-12 (~8M, 60-67%), WRN-28-10 (~36M, 63-68%), ViT-Small (~22M, 65-72%). Always compare methods using the same backbone.

## Common Pitfalls

Transductive vs inductive methods must be explicitly reported (transductive uses query set statistics). Test-time augmentation boosts 1-3% and must be reported. Class split leakage invalidates results. Fewer than 600 episodes gives unreliable variance estimates; use 2000+ for robust comparison.
