"""
Model-Agnostic Meta-Learning (MAML)

Reference: Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation" (2017)

Key idea: Learn an initialization that allows for fast adaptation to new tasks
with just a few gradient steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class SimpleClassifier(nn.Module):
    """
    Simple neural network for MAML.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, params=None):
        """
        Forward pass with optional parameter override.
        
        Args:
            x: Input tensor
            params: Optional OrderedDict of parameters to use instead of self.parameters()
        """
        if params is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            x = F.relu(F.linear(x, params['fc2.weight'], params['fc2.bias']))
            x = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        return x


class MAML:
    """
    Model-Agnostic Meta-Learning algorithm.
    
    Args:
        model: Neural network model
        inner_lr: Learning rate for inner loop (task adaptation)
        meta_lr: Learning rate for outer loop (meta-update)
        num_inner_steps: Number of gradient steps in inner loop
    """
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, num_inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
    
    def inner_loop(self, support_x, support_y, query_x, query_y):
        """
        Perform inner loop adaptation on a single task.
        
        Returns:
            query_loss: Loss on query set after adaptation
        """
        # Copy model parameters for task-specific adaptation
        params = OrderedDict(self.model.named_parameters())
        
        # Inner loop: adapt to support set
        for step in range(self.num_inner_steps):
            # Forward pass with current params
            support_logits = self.model(support_x, params)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            # Compute gradients with respect to params
            grads = torch.autograd.grad(
                support_loss,
                params.values(),
                create_graph=True  # Important for second-order gradients
            )
            
            # Update params using gradient descent
            params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for ((name, param), grad) in zip(params.items(), grads)
            )
        
        # Evaluate on query set with adapted parameters
        query_logits = self.model(query_x, params)
        query_loss = F.cross_entropy(query_logits, query_y)
        
        return query_loss
    
    def meta_train_step(self, tasks):
        """
        Perform one meta-training step on a batch of tasks.
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples
        
        Returns:
            meta_loss: Average query loss across tasks
            meta_accuracy: Average accuracy on query sets
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        meta_accuracy = 0.0
        
        # Perform inner loop for each task and accumulate gradients
        for support_x, support_y, query_x, query_y in tasks:
            task_loss = self.inner_loop(support_x, support_y, query_x, query_y)
            meta_loss += task_loss
            
            # Compute accuracy for this task
            with torch.no_grad():
                # Get adapted parameters (re-do inner loop without graph)
                params = OrderedDict(self.model.named_parameters())
                for step in range(self.num_inner_steps):
                    support_logits = self.model(support_x, params)
                    support_loss = F.cross_entropy(support_logits, support_y)
                    grads = torch.autograd.grad(support_loss, params.values())
                    params = OrderedDict(
                        (name, param - self.inner_lr * grad)
                        for ((name, param), grad) in zip(params.items(), grads)
                    )
                
                query_logits = self.model(query_x, params)
                predictions = torch.argmax(query_logits, dim=1)
                accuracy = (predictions == query_y).float().mean()
                meta_accuracy += accuracy
        
        # Average over tasks
        meta_loss = meta_loss / len(tasks)
        meta_accuracy = meta_accuracy / len(tasks)
        
        # Meta-update: update initial parameters
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item(), meta_accuracy.item()
    
    def adapt(self, support_x, support_y, num_steps=None):
        """
        Adapt the model to a new task given support set.
        
        Returns:
            adapted_params: Parameters adapted to the task
        """
        if num_steps is None:
            num_steps = self.num_inner_steps
        
        params = OrderedDict(self.model.named_parameters())
        
        for step in range(num_steps):
            support_logits = self.model(support_x, params)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            grads = torch.autograd.grad(support_loss, params.values())
            params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for ((name, param), grad) in zip(params.items(), grads)
            )
        
        return params
    
    def predict(self, support_x, support_y, query_x):
        """
        Predict on query set after adapting to support set.
        """
        self.model.eval()
        with torch.no_grad():
            adapted_params = self.adapt(support_x, support_y)
            query_logits = self.model(query_x, adapted_params)
            predictions = torch.argmax(query_logits, dim=1)
        return predictions


# Example usage
if __name__ == "__main__":
    # Model configuration
    input_dim = 784  # 28x28 images flattened
    hidden_dim = 128
    output_dim = 5  # 5-way classification
    
    # Create model and MAML trainer
    model = SimpleClassifier(input_dim, hidden_dim, output_dim)
    maml = MAML(
        model,
        inner_lr=0.01,
        meta_lr=0.001,
        num_inner_steps=5
    )
    
    # Create batch of tasks (5-way 5-shot)
    num_tasks = 4
    n_way = 5
    k_shot = 5
    n_query = 15
    
    tasks = []
    for _ in range(num_tasks):
        support_x = torch.randn(n_way * k_shot, input_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        query_x = torch.randn(n_query, input_dim)
        query_y = torch.randint(0, n_way, (n_query,))
        tasks.append((support_x, support_y, query_x, query_y))
    
    # Meta-training step
    meta_loss, meta_acc = maml.meta_train_step(tasks)
    print(f"Meta-Loss: {meta_loss:.4f}, Meta-Accuracy: {meta_acc:.4f}")
    
    # Test adaptation to a new task
    test_support_x = torch.randn(n_way * k_shot, input_dim)
    test_support_y = torch.arange(n_way).repeat_interleave(k_shot)
    test_query_x = torch.randn(n_query, input_dim)
    
    predictions = maml.predict(test_support_x, test_support_y, test_query_x)
    print(f"Predictions: {predictions}")
