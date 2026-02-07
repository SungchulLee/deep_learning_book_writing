# Progressive Neural Networks

Progressive Networks (Rusu et al., 2016) prevent forgetting entirely by freezing previously learned parameters and adding new capacity for each task. Lateral connections allow knowledge transfer from old tasks to new ones.

## Architecture

```
Task 1:     Task 2:           Task 3:
[Column 1]  [Column 2]        [Column 3]
   h₁¹         h₁²  ←─ h₁¹       h₁³  ←─ h₁¹, h₁²
   h₂¹         h₂²  ←─ h₂¹       h₂³  ←─ h₂¹, h₂²
   h₃¹         h₃²  ←─ h₃¹       h₃³  ←─ h₃¹, h₃²
   (frozen)    (trainable)     (trainable)
```

Each new task gets its own column. Previous columns are frozen. Lateral connections enable forward transfer.

## Implementation

```python
import torch
import torch.nn as nn


class ProgressiveColumn(nn.Module):
    """Single column in a progressive network."""
    
    def __init__(self, layer_dims, lateral_dims=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
        
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i + 1]),
                nn.ReLU()
            ))
            
            if lateral_dims and lateral_dims[i]:
                # Lateral connections from previous columns
                self.lateral_connections.append(
                    nn.Linear(sum(lateral_dims[i]), layer_dims[i + 1])
                )
            else:
                self.lateral_connections.append(None)
    
    def forward(self, x, lateral_inputs=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if lateral_inputs and self.lateral_connections[i] is not None:
                lateral = torch.cat(lateral_inputs[i], dim=1)
                x = x + self.lateral_connections[i](lateral)
        return x


class ProgressiveNetwork(nn.Module):
    """Progressive Neural Network for continual learning."""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.columns = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
    
    def add_task(self, num_classes):
        """Add a new column for a new task."""
        task_id = len(self.columns)
        
        # Freeze previous columns
        for col in self.columns:
            for param in col.parameters():
                param.requires_grad = False
        
        layer_dims = [self.input_dim] + self.hidden_dims
        
        # Lateral dims: for each layer, sizes of corresponding layers in previous columns
        lateral_dims = []
        for i in range(len(self.hidden_dims)):
            if task_id > 0:
                lateral_dims.append([self.hidden_dims[i]] * task_id)
            else:
                lateral_dims.append(None)
        
        column = ProgressiveColumn(layer_dims, lateral_dims)
        self.columns.append(column)
        self.heads.append(nn.Linear(self.hidden_dims[-1], num_classes))
    
    def forward(self, x, task_id):
        # Forward through all columns up to task_id
        column_outputs = []
        
        for col_idx in range(task_id + 1):
            lateral_inputs = None
            if col_idx > 0:
                # Collect lateral inputs from previous columns
                lateral_inputs = [
                    [column_outputs[prev_col][layer_idx]
                     for prev_col in range(col_idx)]
                    for layer_idx in range(len(self.hidden_dims))
                ]
            
            col_output = self.columns[col_idx](x, lateral_inputs)
            column_outputs.append(col_output)
        
        return self.heads[task_id](column_outputs[task_id])
```

## Advantages and Limitations

| Advantage | Limitation |
|-----------|-----------|
| Zero forgetting (frozen columns) | Linear growth in parameters |
| Forward transfer via laterals | Does not scale to many tasks |
| No regularisation needed | Requires task ID at test time |

## References

1. Rusu, A.A., et al. (2016). "Progressive Neural Networks." arXiv.
