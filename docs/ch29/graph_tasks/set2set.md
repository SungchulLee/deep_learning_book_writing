# 29.5.5 Set2Set

## Overview
**Set2Set** (Vinyals et al., 2016) produces order-invariant graph representations via attention-based LSTM processing. More expressive than simple pooling, it iteratively attends to the node set.

## Algorithm
For $T$ processing steps:
1. $q_t, c_t = \text{LSTM}([q_{t-1}, r_{t-1}], (q_{t-1}, c_{t-1}))$
2. $e_{t,i} = q_t^T h_i$ (attention energy)
3. $a_{t,i} = \text{softmax}(e_{t,i})$ (attention weights)
4. $r_t = \sum_i a_{t,i} h_i$ (weighted readout)

Output: $[q_T \| r_T]$ (2d-dimensional).

## Properties
- More expressive than sum/mean/max pooling
- Captures complex set statistics through iterative attention
- Commonly used in molecular property prediction (Gilmer et al., 2017)
