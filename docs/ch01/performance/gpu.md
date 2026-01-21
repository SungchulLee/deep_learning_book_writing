# GPU and Performance

Modern deep learning performance relies heavily on hardware acceleration, especially GPUs.

---

## CPU vs GPU

GPUs excel at:
- large matrix operations,
- parallel computation,
- batched linear algebra.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

---

## Moving data to GPU

All tensors involved in computation must be on the same device:

```python
x = x.to(device)
y = y.to(device)
```

Mixing CPU and GPU tensors causes runtime errors.

---

## Performance tips

- Use larger batch sizes when possible.
- Avoid Python loops inside training steps.
- Minimize data transfer between CPU and GPU.

---

## Numerical precision

Using lower precision can improve speed:

```python
torch.float16
torch.bfloat16
```

But may affect numerical stability.

---

## Financial modeling

In finance, GPUs are valuable for:
- large-scale Monte Carlo,
- neural calibration,
- real-time risk scenarios.

---

## Key takeaways

- GPUs accelerate tensor operations.
- Keep tensors on the same device.
- Balance speed and numerical stability.
