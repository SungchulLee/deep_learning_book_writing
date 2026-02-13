# CORe50 and Stream Benchmarks

## CORe50

Lomonaco & Maltoni (2017): 50 objects from 10 categories across 11 sessions with different backgrounds and lighting. Three scenarios: New Instances (NI), New Classes (NC), and New Instances and Classes (NIC).

## Online Continual Learning

Each example seen only once (single pass). Much harder than multi-epoch. Streaming benchmarks with gradual distribution shifts have no clear task boundaries.

## Metrics

Online accuracy (averaged after each example), anytime inference readiness, and computational efficiency (wall-clock time and memory per example).

## Current Challenges

Handling gradual concept drift (vs abrupt task switches), scaling to high-dimensional data (video, long documents), and maintaining calibrated uncertainty throughout learning.
