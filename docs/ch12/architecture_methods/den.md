# Dynamically Expandable Networks (DEN)

DEN (Yoon et al., 2018) combines selective retraining, network expansion, and split/duplication to dynamically grow the network as needed for new tasks.

## Algorithm Overview

For each new task, DEN performs three steps:

1. **Selective retraining**: Identify and retrain only the neurons relevant to the new task using group sparse regularisation
2. **Dynamic expansion**: If selective retraining is insufficient, add new neurons
3. **Split/duplication**: If a neuron is too important for both old and new tasks, duplicate it

## Implementation Sketch

```python
import torch
import torch.nn as nn


class DynamicallyExpandableNetwork(nn.Module):
    """DEN: Dynamically Expandable Networks."""
    
    def __init__(self, input_dim, initial_hidden_dims, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + initial_hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        
        self.output = nn.Linear(initial_hidden_dims[-1], output_dim)
        self.task_outputs = nn.ModuleDict()
    
    def selective_retrain(self, task_loader, threshold, device='cuda'):
        """
        Step 1: Identify neurons relevant to the new task.
        Uses L1 regularisation to select sparse subset of neurons.
        """
        # Train with group sparsity
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        l1_lambda = 0.001
        
        for x, y in task_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = self.forward(x)
            loss = criterion(logits, y)
            
            # Group sparse regularisation
            for layer in self.layers:
                loss += l1_lambda * layer.weight.abs().sum()
            
            loss.backward()
            optimizer.step()
    
    def expand_network(self, layer_idx, num_new_neurons):
        """
        Step 2: Add new neurons to a layer.
        """
        old_layer = self.layers[layer_idx]
        new_in = old_layer.in_features
        new_out = old_layer.out_features + num_new_neurons
        
        new_layer = nn.Linear(new_in, new_out)
        with torch.no_grad():
            new_layer.weight[:old_layer.out_features] = old_layer.weight
            new_layer.bias[:old_layer.out_features] = old_layer.bias
        
        self.layers[layer_idx] = new_layer
        
        # Update next layer's input dimension
        if layer_idx + 1 < len(self.layers):
            next_layer = self.layers[layer_idx + 1]
            new_next = nn.Linear(new_out, next_layer.out_features)
            with torch.no_grad():
                new_next.weight[:, :old_layer.out_features] = next_layer.weight
            self.layers[layer_idx + 1] = new_next
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)
```

## Comparison of Architecture Methods

| Method | Growth | Forgetting | Forward transfer |
|--------|--------|-----------|-----------------|
| Progressive Networks | Linear (new column) | Zero | Via lateral connections |
| PackNet | Fixed (pruning) | Zero | Limited |
| DEN | Dynamic (as needed) | Low | Via shared neurons |

## References

1. Yoon, J., et al. (2018). "Lifelong Learning with Dynamically Expandable Networks." *ICLR*.
