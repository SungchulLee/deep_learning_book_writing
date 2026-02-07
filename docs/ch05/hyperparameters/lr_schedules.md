# Learning Rate Schedules

## Overview

The learning rate is the most impactful hyperparameter. This section covers the learning rate range test and strategies for finding the optimal learning rate and schedule.

## Learning Rate Range Test (LR Finder)

Train for one epoch while exponentially increasing the learning rate. Plot loss vs. learning rate:

```python
def lr_range_test(model, train_loader, loss_fn, optimizer,
                  start_lr=1e-7, end_lr=10, num_steps=100):
    lrs, losses = [], []
    lr_mult = (end_lr / start_lr) ** (1 / num_steps)

    optimizer.param_groups[0]['lr'] = start_lr
    best_loss = float('inf')

    for i, (x, y) in enumerate(train_loader):
        if i >= num_steps:
            break
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        if loss.item() > 4 * best_loss:
            break
        best_loss = min(best_loss, loss.item())

        optimizer.param_groups[0]['lr'] *= lr_mult

    return lrs, losses
```

Choose the learning rate roughly one order of magnitude below the loss minimum (the steepest descent region), not at the minimum itself.

## Schedule-Hyperparameter Interactions

The optimal learning rate depends on the schedule:

- **No schedule**: Set LR conservatively to ensure stable convergence.
- **Step decay**: Set initial LR higher; it will be reduced at milestones.
- **Cosine annealing**: Set initial LR to the value suggested by the LR range test.
- **OneCycleLR**: Set `max_lr` to the LR range test suggestion; the schedule handles warm-up and cool-down.

## Key Takeaways

- Use the LR range test to find a good initial learning rate.
- The optimal LR depends on the schedule, optimizer, batch size, and model.
- Learning rate and schedule should be tuned together, not independently.
