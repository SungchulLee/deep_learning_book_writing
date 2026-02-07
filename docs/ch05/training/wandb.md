# Logging with Weights & Biases

## Overview

Weights & Biases (W&B) is a cloud-hosted experiment tracking platform that provides automatic logging, hyperparameter sweeps, artifact versioning, and collaborative dashboards. It integrates with PyTorch via the `wandb` library.

## Setup

```bash
pip install wandb
wandb login  # Authenticate with API key
```

```python
import wandb

wandb.init(
    project='deep-learning-finance',
    name='experiment-001',
    config={
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100,
        'architecture': 'ResNet18',
        'optimizer': 'AdamW'
    }
)
```

## Logging Metrics

```python
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)

    wandb.log({
        'epoch': epoch,
        'train/loss': train_loss,
        'train/accuracy': train_acc,
        'val/loss': val_loss,
        'val/accuracy': val_acc,
        'learning_rate': optimizer.param_groups[0]['lr']
    })

wandb.finish()
```

## Logging Images and Media

```python
images, labels = next(iter(val_loader))
preds = model(images.to(device))

wandb.log({
    'predictions': [
        wandb.Image(img, caption=f'pred={p.argmax()}, true={t}')
        for img, p, t in zip(images[:8], preds[:8], labels[:8])
    ]
})
```

## Model Watching

Automatically log gradients and parameter histograms:

```python
wandb.watch(model, log='all', log_freq=100)
# 'all' logs gradients and parameters
# 'gradients' logs only gradients
# 'parameters' logs only parameters
```

## Hyperparameter Sweeps

W&B supports automated hyperparameter search:

```python
sweep_config = {
    'method': 'bayes',  # 'grid', 'random', or 'bayes'
    'metric': {'name': 'val/loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 1e-2, 'distribution': 'log_uniform_values'},
        'batch_size': {'values': [32, 64, 128]},
        'optimizer': {'values': ['Adam', 'AdamW', 'SGD']}
    }
}

sweep_id = wandb.sweep(sweep_config, project='deep-learning-finance')

def train_sweep():
    wandb.init()
    config = wandb.config
    # ... training with config.learning_rate, config.batch_size, etc.
    wandb.finish()

wandb.agent(sweep_id, function=train_sweep, count=50)
```

## Artifact Versioning

Track datasets and model checkpoints:

```python
# Log model checkpoint
artifact = wandb.Artifact('model-checkpoint', type='model')
artifact.add_file('best_model.pt')
wandb.log_artifact(artifact)

# Log dataset
data_artifact = wandb.Artifact('training-data', type='dataset')
data_artifact.add_dir('data/processed/')
wandb.log_artifact(data_artifact)
```

## TensorBoard vs. W&B

| Feature | TensorBoard | W&B |
|---------|-------------|-----|
| Hosting | Local | Cloud (free tier available) |
| Collaboration | Manual sharing | Built-in teams |
| Hyperparameter sweeps | Not built-in | Integrated |
| Artifact versioning | No | Yes |
| Setup complexity | Lower | Higher (account required) |
| Privacy | Full local control | Data sent to cloud |

## Key Takeaways

- W&B provides cloud-hosted experiment tracking with minimal code changes.
- `wandb.log()` accepts any dictionary of metrics; use `/` for grouping.
- `wandb.watch()` automatically logs gradient and parameter histograms.
- Sweeps automate hyperparameter search with Bayesian optimization.
- Use artifacts for reproducible dataset and model version tracking.
