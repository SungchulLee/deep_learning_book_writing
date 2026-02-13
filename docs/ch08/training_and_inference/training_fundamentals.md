# Training Fundamentals

## Loss Functions

The training objective depends on the task: next-token cross-entropy for language modeling (GPT), masked token cross-entropy for MLM (BERT), cross-entropy with teacher forcing for seq2seq, and cross-entropy on [CLS] representation for classification.

## Optimizer: AdamW

Standard for transformers, decoupling weight decay: $\theta_{t+1} = \theta_t - \eta(\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda\theta_t)$. Typical hyperparameters: lr 1e-4 to 5e-4 with warmup, $\beta_1=0.9$, $\beta_2=0.98{-}0.999$, weight decay 0.01-0.1.

## Gradient Accumulation

Simulate larger batches without extra memory:

```python
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

## Gradient Clipping

Transformers are susceptible to gradient spikes: $g \leftarrow g \cdot \min(1, \text{max\_norm} / \|g\|)$. Max norm of 1.0 is standard.

## Dropout Placement

After attention weights (attention dropout), after each sublayer before residual addition, and on embeddings. Rate: 0.1 for pre-training, 0.1-0.3 for fine-tuning on small datasets.
