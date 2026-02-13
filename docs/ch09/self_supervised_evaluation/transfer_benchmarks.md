# Transfer Benchmarks

## Overview

Transfer benchmarks evaluate whether self-supervised representations generalize beyond the pre-training domain.

## Standard Vision Benchmarks

ImageNet variants (V2, R, A) test robustness to distribution shift. Fine-grained benchmarks (CUB-200 birds, Stanford Cars, FGVC Aircraft) test detailed discrimination. Other domains (CIFAR-10/100, Places365, VOC07, DTD textures) test breadth of learned features.

## Evaluation Protocol

For each dataset: extract frozen features, train a linear classifier, report top-1 accuracy. Compare against supervised pre-training with the same architecture.

## NLP Transfer

GLUE/SuperGLUE (NLU task suite), SQuAD (extractive QA), CoNLL-2003 (NER).

## Reporting Standards

Include pre-training dataset and compute budget, evaluation protocol details, supervised baseline comparison with same architecture, and standard deviations across runs.
