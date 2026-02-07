# Generative Replay

Generative Replay (Shin et al., 2017) replaces stored examples with a generative model that produces pseudo-samples from previous task distributions. This avoids storing raw data while maintaining rehearsal capability.

## Key Idea

Instead of storing real examples, train a generator $G$ alongside the main model $M$:

1. When learning task $t+1$, use $G$ to generate samples resembling tasks $1, ..., t$
2. Train $M$ on both real data from task $t+1$ and generated data from $G$
3. Update $G$ to also model task $t+1$

## Implementation

```python
import torch
import torch.nn as nn


class GenerativeReplay:
    """Generative replay using a VAE or GAN for pseudo-rehearsal."""
    
    def __init__(self, solver, generator, device='cuda'):
        self.solver = solver.to(device)
        self.generator = generator.to(device)
        self.device = device
    
    def train_task(self, dataloader, task_id, epochs=10, replay_ratio=0.5):
        """Train on new task with generative replay of previous tasks."""
        solver_opt = torch.optim.Adam(self.solver.parameters(), lr=1e-3)
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for x_real, y_real in dataloader:
                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)
                batch_size = x_real.size(0)
                
                # Generate replay samples from previous tasks
                if task_id > 0:
                    n_replay = int(batch_size * replay_ratio)
                    with torch.no_grad():
                        z = torch.randn(n_replay, self.generator.latent_dim).to(self.device)
                        x_replay = self.generator.decode(z)
                        y_replay = self.solver(x_replay).argmax(1)
                    
                    # Combine real and replay data
                    x_combined = torch.cat([x_real, x_replay])
                    y_combined = torch.cat([y_real, y_replay])
                else:
                    x_combined = x_real
                    y_combined = y_real
                
                # Train solver
                solver_opt.zero_grad()
                logits = self.solver(x_combined)
                loss = criterion(logits, y_combined)
                loss.backward()
                solver_opt.step()
                
                # Train generator on current task data
                gen_opt.zero_grad()
                gen_loss = self.generator.train_step(x_real)
                gen_loss.backward()
                gen_opt.step()
```

## Advantages and Limitations

| Aspect | Advantage | Limitation |
|--------|-----------|-----------|
| Memory | No raw data storage | Generator requires parameters |
| Privacy | No real data retained | Generated samples may leak info |
| Quality | Unlimited replay samples | Generation quality limits performance |
| Scalability | Fixed memory regardless of tasks | Generator quality degrades over many tasks |

## References

1. Shin, H., et al. (2017). "Continual Learning with Deep Generative Replay." *NeurIPS*.
