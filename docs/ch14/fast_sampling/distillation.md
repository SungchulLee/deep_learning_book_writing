# Progressive Distillation

## Introduction

**Progressive distillation** trains a student model to match a teacher in fewer steps, iteratively halving the number of steps required.

## Method

### Two-Step Matching

Train student to match two teacher steps in one:

$$\mathcal{L} = \|x_{t-2}^{\text{student}} - x_{t-2}^{\text{teacher}}\|^2$$

### Iteration

1. Start: 1024-step teacher
2. Round 1: 512-step student → new teacher
3. Round 2: 256-step student → new teacher
4. Continue until 4-8 steps

## Key Results

| Steps | FID | Speed |
|-------|-----|-------|
| 1024 | 2.4 | 1× |
| 8 | 3.0 | 128× |
| 4 | 4.5 | 256× |

## Implementation Sketch

```python
def distillation_loss(student, teacher, x_0, t, t_minus_2):
    """Progressive distillation loss."""
    # Teacher: two steps
    x_t = add_noise(x_0, t)
    x_t_1_teacher = teacher_step(teacher, x_t, t)
    x_t_2_teacher = teacher_step(teacher, x_t_1, t-1)
    
    # Student: one step
    x_t_2_student = student_step(student, x_t, t, skip=2)
    
    return mse_loss(x_t_2_student, x_t_2_teacher)
```

## Summary

Progressive distillation achieves 100× speedup with minimal quality loss by iteratively training faster models.

## Navigation

- **Previous**: [Ancestral Sampling](ancestral.md)
- **Next**: [Consistency Models](consistency.md)
