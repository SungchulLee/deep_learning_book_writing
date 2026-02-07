# AMSGrad

## Overview

AMSGrad is a variant of Adam that addresses a theoretical non-convergence issue by maintaining the maximum of all past second moment estimates, ensuring the effective learning rate is monotonically non-increasing.

## The Convergence Issue

In certain pathological cases, Adam's second moment estimate $v_t$ can decrease, causing the effective learning rate to increase at exactly the wrong time. This can prevent convergence to the optimum even for simple convex problems.

## Update Rule

The only change from Adam is a monotonic constraint on the second moment:

$$\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$$

This ensures the denominator $\sqrt{\hat{v}_t}$ never decreases, preventing the effective learning rate from increasing.

## PyTorch Implementation

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                             betas=(0.9, 0.999), amsgrad=True)
```

## Practical Impact

In practice, the convergence issue AMSGrad addresses is rarely encountered in typical deep learning training. Empirical studies show mixed results—AMSGrad sometimes provides modest improvements and sometimes performs identically to Adam. It is worth trying when Adam shows unstable training behavior.

## Key Takeaways

- AMSGrad constrains Adam's second moment to be monotonically non-decreasing.
- Addresses a theoretical non-convergence issue in Adam.
- Practical impact is often minimal; try when Adam training is unstable.
