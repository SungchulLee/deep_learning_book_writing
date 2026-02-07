# Checkpointing

## Overview

Checkpointing saves the complete training state—model parameters, optimizer state, epoch number, and metrics—enabling training to resume from any saved point. Essential for long training runs and preemptible compute.

## Saving a Checkpoint

```python
def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'rng_state': torch.get_rng_state(),
    }
    torch.save(checkpoint, path)
```

## Loading a Checkpoint

```python
def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_val_loss']
```

## Checkpoint Strategy

```python
for epoch in range(start_epoch, num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    # Save periodic checkpoints
    if (epoch + 1) % save_every == 0:
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                        f'checkpoint_epoch_{epoch+1}.pt')

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                        'best_model.pt')
```

## Rotating Checkpoints

Keep only the $k$ most recent checkpoints to manage disk space:

```python
import os
import glob

def save_rotating_checkpoint(state, path, keep_last=3):
    torch.save(state, path)
    checkpoints = sorted(glob.glob('checkpoint_epoch_*.pt'))
    for ckpt in checkpoints[:-keep_last]:
        os.remove(ckpt)
```

## Key Takeaways

- Checkpoints include model state, optimizer state, scheduler state, and epoch.
- Save both periodic checkpoints (for resumption) and best-model checkpoints (for deployment).
- Rotating checkpoints prevent disk space exhaustion during long training runs.
