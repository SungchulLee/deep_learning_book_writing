"""
GRU Theory and Implementation from Scratch
==========================================

Gated Recurrent Unit (GRU) is a simplified variant of LSTM that combines
the forget and input gates into a single update gate, and merges the cell
state and hidden state.

Mathematical Formulation:
------------------------
For time step t:

Reset Gate:    r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
Update Gate:   z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
Candidate:     h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
Hidden State:  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

Where:
- σ is the sigmoid function
- ⊙ is element-wise multiplication
- W are weight matrices
- b are bias vectors

Key Differences from LSTM:
- 2 gates instead of 3 (no output gate)
- No separate cell state
- Fewer parameters (faster training)
- Comparable performance on many tasks
"""

import numpy as np
import matplotlib.pyplot as plt


class GRUCell:
    """A single GRU cell implementation from scratch."""
    
    def __init__(self, input_size, hidden_size):
        """
        Initialize GRU cell parameters.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights using Xavier initialization
        scale = 1.0 / np.sqrt(input_size + hidden_size)
        
        # Reset gate weights
        self.W_r = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_r = np.zeros((hidden_size, 1))
        
        # Update gate weights
        self.W_z = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_z = np.zeros((hidden_size, 1))
        
        # Candidate hidden state weights
        self.W_h = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_h = np.zeros((hidden_size, 1))
        
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """Hyperbolic tangent activation function."""
        return np.tanh(x)
    
    def forward(self, x, h_prev):
        """
        Forward pass through GRU cell.
        
        Args:
            x: Input at current time step (input_size, 1)
            h_prev: Previous hidden state (hidden_size, 1)
            
        Returns:
            h: New hidden state
            cache: Values needed for backward pass
        """
        # Concatenate previous hidden state and current input
        combined = np.vstack((h_prev, x))
        
        # Reset gate - determines how much past info to forget
        r = self.sigmoid(self.W_r @ combined + self.b_r)
        
        # Update gate - determines how much to update
        z = self.sigmoid(self.W_z @ combined + self.b_z)
        
        # Candidate hidden state using reset gate
        # Reset gate modulates the previous hidden state
        combined_reset = np.vstack((r * h_prev, x))
        h_tilde = self.tanh(self.W_h @ combined_reset + self.b_h)
        
        # New hidden state: interpolation between previous and candidate
        # z acts as a "forget gate" and "input gate" combined
        h = (1 - z) * h_prev + z * h_tilde
        
        # Cache for backward pass
        cache = (x, h_prev, combined, r, z, h_tilde)
        
        return h, cache


class GRU:
    """Multi-step GRU implementation."""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize GRU network.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            output_size: Dimension of output
        """
        self.cell = GRUCell(input_size, hidden_size)
        
        # Output layer weights
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        
        self.hidden_size = hidden_size
        
    def forward(self, inputs):
        """
        Forward pass through entire sequence.
        
        Args:
            inputs: List of input vectors, each of shape (input_size, 1)
            
        Returns:
            outputs: List of output predictions
            hidden_states: List of hidden states for visualization
        """
        h = np.zeros((self.hidden_size, 1))
        
        outputs = []
        hidden_states = []
        
        for x in inputs:
            h, _ = self.cell.forward(x, h)
            y = self.W_y @ h + self.b_y
            
            outputs.append(y)
            hidden_states.append(h.copy())
            
        return outputs, hidden_states


def demonstrate_gru_gates():
    """Demonstrate how GRU gates work with visualizations."""
    print("=" * 60)
    print("GRU Gate Mechanics Demonstration")
    print("=" * 60)
    
    # Create a simple GRU cell
    input_size = 3
    hidden_size = 4
    gru_cell = GRUCell(input_size, hidden_size)
    
    # Generate sample input sequence
    sequence_length = 20
    inputs = [np.random.randn(input_size, 1) for _ in range(sequence_length)]
    
    # Track gate activations
    reset_gates = []
    update_gates = []
    hidden_states = []
    candidates = []
    
    h = np.zeros((hidden_size, 1))
    
    for x in inputs:
        combined = np.vstack((h, x))
        
        r = gru_cell.sigmoid(gru_cell.W_r @ combined + gru_cell.b_r)
        z = gru_cell.sigmoid(gru_cell.W_z @ combined + gru_cell.b_z)
        
        combined_reset = np.vstack((r * h, x))
        h_tilde = gru_cell.tanh(gru_cell.W_h @ combined_reset + gru_cell.b_h)
        
        h = (1 - z) * h + z * h_tilde
        
        reset_gates.append(r.mean())
        update_gates.append(z.mean())
        candidates.append(h_tilde.mean())
        hidden_states.append(h.mean())
    
    # Visualize gate activations
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(reset_gates, 'r-', linewidth=2)
    plt.title('Reset Gate Activation')
    plt.ylabel('Activation (0-1)')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.subplot(2, 2, 2)
    plt.plot(update_gates, 'g-', linewidth=2)
    plt.title('Update Gate Activation')
    plt.ylabel('Activation (0-1)')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.subplot(2, 2, 3)
    plt.plot(candidates, 'b-', linewidth=2)
    plt.title('Candidate Hidden State')
    plt.ylabel('Mean Value')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(hidden_states, 'm-', linewidth=2)
    plt.title('Hidden State Evolution')
    plt.ylabel('Mean Value')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/lstm_gru_module/gru_gates.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gate activation plot saved as 'gru_gates.png'")
    plt.close()


def compare_gate_mechanics():
    """Compare how update gate interpolates between previous and new state."""
    print("\n" + "=" * 60)
    print("Understanding the Update Gate")
    print("=" * 60)
    
    print("\nThe update gate z_t controls:")
    print("h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t")
    print("\nWhen z_t ≈ 0: Keep old hidden state (h_t ≈ h_{t-1})")
    print("When z_t ≈ 1: Use new candidate (h_t ≈ h̃_t)")
    print("When z_t ≈ 0.5: Mix equally")
    
    # Demonstrate with examples
    h_prev = np.array([[1.0], [2.0], [3.0]])
    h_tilde = np.array([[5.0], [6.0], [7.0]])
    
    scenarios = [
        ("z ≈ 0 (Keep old)", np.array([[0.1], [0.1], [0.1]])),
        ("z ≈ 1 (Use new)", np.array([[0.9], [0.9], [0.9]])),
        ("z ≈ 0.5 (Mix)", np.array([[0.5], [0.5], [0.5]])),
    ]
    
    print("\nExample with h_{t-1} = [1, 2, 3] and h̃_t = [5, 6, 7]:")
    for name, z in scenarios:
        h = (1 - z) * h_prev + z * h_tilde
        print(f"\n{name}:")
        print(f"  Result: {h.flatten()}")


def parameter_comparison():
    """Compare parameter counts between LSTM and GRU."""
    print("\n" + "=" * 60)
    print("LSTM vs GRU Parameter Comparison")
    print("=" * 60)
    
    input_size = 100
    hidden_size = 200
    
    # LSTM parameters
    lstm_params = 4 * (hidden_size * (input_size + hidden_size) + hidden_size)
    
    # GRU parameters
    gru_params = 3 * (hidden_size * (input_size + hidden_size) + hidden_size)
    
    reduction = (1 - gru_params / lstm_params) * 100
    
    print(f"\nFor input_size={input_size}, hidden_size={hidden_size}:")
    print(f"LSTM parameters: {lstm_params:,}")
    print(f"GRU parameters:  {gru_params:,}")
    print(f"Reduction:       {reduction:.1f}%")
    
    print("\nGRU has ~25% fewer parameters than LSTM")
    print("This means:")
    print("  ✓ Faster training")
    print("  ✓ Less memory usage")
    print("  ✓ Less prone to overfitting on small datasets")


def main():
    """Run all GRU demonstrations."""
    print("\n" + "=" * 60)
    print("GRU: Gated Recurrent Unit Networks")
    print("=" * 60)
    
    print("\nKey Features of GRU:")
    print("1. Simpler than LSTM (2 gates vs 3)")
    print("2. No separate cell state")
    print("3. Fewer parameters → faster training")
    print("4. Often performs similarly to LSTM")
    
    print("\nGRU Gates:")
    print("- Reset gate (r): Controls how much past info to forget")
    print("- Update gate (z): Controls how much to update")
    print("  * Acts as both forget and input gate")
    
    # Run demonstrations
    demonstrate_gru_gates()
    compare_gate_mechanics()
    parameter_comparison()
    
    print("\n" + "=" * 60)
    print("GRU demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
