# Early Stopping

## Overview

Early stopping terminates training when validation performance stops improving, serving as an implicit regularizer that limits the effective model complexity. It is one of the most widely used and effective regularization techniques in deep learning.

## Implementation

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_score = None
        self.best_state = None
        self.counter = 0

    def __call__(self, val_loss, model):
        if self.best_score is None or val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best:
                    model.load_state_dict(self.best_state)
                return True  # Stop training
            return False

# Usage
early_stopping = EarlyStopping(patience=15, min_delta=1e-4)

for epoch in range(max_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    if early_stopping(val_loss, model):
        print(f"Early stopping at epoch {epoch}")
        break
```

## Key Parameters

**Patience**: Number of epochs to wait for improvement before stopping. Too low: stops prematurely before the model converges. Too high: overfits before stopping. Typical values: 5–20.

**Min delta**: Minimum improvement to qualify as progress. Prevents stopping from tiny noise-driven improvements.

**Restore best**: Whether to restore the model to the best validation state upon stopping.

## Early Stopping as Regularization

Early stopping limits the number of gradient steps, which constrains how far parameters can move from initialization. For linear models, this is formally equivalent to L2 regularization with a regularization strength inversely proportional to the number of training steps.

## Interaction with Learning Rate Scheduling

Early stopping interacts with learning rate schedulers. A decreasing learning rate may cause apparent plateaus in validation loss that would be resolved by further training. Options:

- Use early stopping with `ReduceLROnPlateau`: reduce LR first, then stop if still no improvement.
- Set patience to be longer than any scheduler transition period.

## Key Takeaways

- Early stopping terminates training when validation loss stops improving.
- It acts as an implicit regularizer by limiting effective model complexity.
- Always restore the best model state, not the final state.
- Coordinate patience with learning rate scheduling to avoid premature stopping.
