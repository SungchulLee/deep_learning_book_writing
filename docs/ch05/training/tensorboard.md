# Logging with TensorBoard

## Overview

TensorBoard provides interactive visualization of training metrics, model graphs, histograms, and images. PyTorch integrates with TensorBoard through `torch.utils.tensorboard.SummaryWriter`.

## Setup

```bash
pip install tensorboard
```

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_01')
```

The `SummaryWriter` writes event files to the specified log directory. Launch the TensorBoard UI with:

```bash
tensorboard --logdir=runs
```

## Logging Scalars

```python
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

writer.close()
```

The `tag` string uses `/` as a grouping delimiter—`Loss/train` and `Loss/val` appear on the same chart.

## Logging Histograms

Track parameter and gradient distributions over training:

```python
for name, param in model.named_parameters():
    writer.add_histogram(f'Parameters/{name}', param, epoch)
    if param.grad is not None:
        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
```

## Logging Images

Visualize input batches, augmented samples, or model outputs:

```python
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images[:16], nrow=4, normalize=True)
writer.add_image('Training/batch_samples', grid, epoch)
```

## Logging Model Graph

```python
dummy_input = torch.randn(1, 3, 224, 224).to(device)
writer.add_graph(model, dummy_input)
```

## Logging Hyperparameters

```python
hparams = {'lr': 0.001, 'batch_size': 64, 'optimizer': 'Adam'}
metrics = {'hparam/best_val_loss': best_val_loss,
           'hparam/best_val_acc': best_val_acc}
writer.add_hparams(hparams, metrics)
```

## Comparing Experiments

Each `SummaryWriter` writes to a separate subdirectory. TensorBoard overlays all runs under the same `--logdir`:

```python
# Experiment comparison
writer1 = SummaryWriter('runs/lr_0.001')
writer2 = SummaryWriter('runs/lr_0.01')
writer3 = SummaryWriter('runs/lr_0.1')
```

## Key Takeaways

- `SummaryWriter` logs scalars, histograms, images, and model graphs to event files.
- Group related metrics with `/` in tag names for organized dashboards.
- Log both training and validation metrics on the same chart for overfitting detection.
- Use separate log directories for experiment comparison.
