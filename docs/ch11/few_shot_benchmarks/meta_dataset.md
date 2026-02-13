# Meta-Dataset

## Overview

Meta-Dataset (Triantafillou et al., 2020) addresses limitations of earlier benchmarks by spanning 10 diverse image classification datasets across multiple visual domains.

## Component Datasets

ILSVRC/ImageNet (712 classes), Omniglot (1623 character classes), Aircraft (100), CUB-200 (200 bird species), DTD (47 textures), Quick Draw (345 sketch categories), Fungi (1394), VGG Flower (102), Traffic Signs (43), MSCOCO (80).

## Key Features

Variable-way variable-shot episodes (vs fixed 5-way 5-shot in mini-ImageNet) more closely reflect real-world scenarios. Training on 8 datasets; Traffic Signs and MSCOCO are test-only to evaluate cross-domain generalization.

## Significance

Much harder to overfit to a specific episode structure. Reveals cross-domain generalization gaps that simpler benchmarks like mini-ImageNet cannot expose.
