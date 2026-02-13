#!/usr/bin/env python3
"""
============================================================
TorchScript: Optimized Model Export for Production
============================================================

TorchScript is PyTorch's way of creating serializable and
optimizable models for production deployment.

Topics:
- Tracing vs Scripting
- TorchScript conversion
- Model optimization
- Mobile deployment preparation
"""

import torch
import torch.nn as nn

print("=" * 70)
print("TORCHSCRIPT EXPORT TUTORIAL")
print("=" * 70)

# Define models
class SimpleModel(nn.Module):
    """Simple feedforward network"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ModelWithControlFlow(nn.Module):
    """Model with control flow - requires scripting"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        if x.sum() > 0:
            x = self.fc(x)
        else:
            x = x * 2
        return x


# METHOD 1: TRACING
print("\n" + "=" * 70)
print("METHOD 1: TRACING")
print("=" * 70)

model = SimpleModel()
model.eval()

example_input = torch.randn(1, 10)

# Trace the model
traced_model = torch.jit.trace(model, example_input)
print("\nModel successfully traced")

# Save traced model
traced_path = "traced_model.pt"
traced_model.save(traced_path)
print(f"Traced model saved to '{traced_path}'")

# Test traced model
test_input = torch.randn(2, 10)

with torch.no_grad():
    original_output = model(test_input)
    traced_output = traced_model(test_input)

diff = torch.abs(original_output - traced_output).max().item()
print(f"Max difference: {diff:.6f}")

if diff < 1e-6:
    print("Traced model produces identical results")

# Cleanup
import os
if os.path.exists(traced_path):
    os.remove(traced_path)


# METHOD 2: SCRIPTING
print("\n" + "=" * 70)
print("METHOD 2: SCRIPTING")
print("=" * 70)

# Script simple model
simple_model = SimpleModel()
simple_model.eval()

scripted_simple = torch.jit.script(simple_model)
print("\nSimple model successfully scripted")

scripted_path = "scripted_simple.pt"
scripted_simple.save(scripted_path)
print(f"Scripted model saved to '{scripted_path}'")

# Cleanup
if os.path.exists(scripted_path):
    os.remove(scripted_path)

# Script model with control flow
print("\nScripting model with control flow...")
control_flow_model = ModelWithControlFlow()
control_flow_model.eval()

scripted_control = torch.jit.script(control_flow_model)
print("Control flow model successfully scripted")

# Test with different inputs
pos_input = torch.ones(1, 10)
neg_input = -torch.ones(1, 10)

with torch.no_grad():
    pos_result = scripted_control(pos_input)
    neg_result = scripted_control(neg_input)

print("Control flow preserved correctly")


# OPTIMIZATION
print("\n" + "=" * 70)
print("TORCHSCRIPT OPTIMIZATIONS")
print("=" * 70)

model = SimpleModel()
model.eval()

scripted_model = torch.jit.script(model)

print("\nApplying optimizations...")

# Freeze the model
frozen_model = torch.jit.freeze(scripted_model)
print("Model frozen")

# Optimize for inference
optimized_model = torch.jit.optimize_for_inference(frozen_model)
print("Optimized for inference")

print("\nOptimizations applied:")
print("- Constant propagation")
print("- Dead code elimination")
print("- Operator fusion")
print("- Memory layout optimization")

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)
