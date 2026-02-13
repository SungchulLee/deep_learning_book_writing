"""
LSTM Theory and Implementation from Scratch
============================================

Long Short-Term Memory (LSTM) networks solve the vanishing gradient problem
in traditional RNNs by introducing a gating mechanism and cell state.

Mathematical Formulation:
------------------------
For time step t:

Forget Gate:    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:     i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Cell Candidate: C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell State:     C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Output Gate:    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden State:   h_t = o_t ⊙ tanh(C_t)

Where:
- σ is the sigmoid function
- ⊙ is element-wise multiplication
- W are weight matrices
- b are bias vectors
"""

import numpy as np
import matplotlib.pyplot as plt


class LSTMCell:
    """A single LSTM cell implementation from scratch."""
    
    def __init__(self, input_size, hidden_size):
        """
        Initialize LSTM cell parameters.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights using Xavier initialization
        scale = 1.0 / np.sqrt(input_size + hidden_size)
        
        # Forget gate weights
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_f = np.zeros((hidden_size, 1))
        
        # Input gate weights
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_i = np.zeros((hidden_size, 1))
        
        # Cell candidate weights
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_c = np.zeros((hidden_size, 1))
        
        # Output gate weights
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_o = np.zeros((hidden_size, 1))
        
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """Hyperbolic tangent activation function."""
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass through LSTM cell.
        
        Args:
            x: Input at current time step (input_size, 1)
            h_prev: Previous hidden state (hidden_size, 1)
            c_prev: Previous cell state (hidden_size, 1)
            
        Returns:
            h: New hidden state
            c: New cell state
            cache: Values needed for backward pass
        """
        # Concatenate previous hidden state and current input
        combined = np.vstack((h_prev, x))
        
        # Forget gate
        f = self.sigmoid(self.W_f @ combined + self.b_f)
        
        # Input gate
        i = self.sigmoid(self.W_i @ combined + self.b_i)
        
        # Cell candidate
        c_tilde = self.tanh(self.W_c @ combined + self.b_c)
        
        # New cell state
        c = f * c_prev + i * c_tilde
        
        # Output gate
        o = self.sigmoid(self.W_o @ combined + self.b_o)
        
        # New hidden state
        h = o * self.tanh(c)
        
        # Cache for backward pass
        cache = (x, h_prev, c_prev, combined, f, i, c_tilde, c, o)
        
        return h, c, cache


class LSTM:
    """Multi-step LSTM implementation."""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize LSTM network.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            output_size: Dimension of output
        """
        self.cell = LSTMCell(input_size, hidden_size)
        
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
        c = np.zeros((self.hidden_size, 1))
        
        outputs = []
        hidden_states = []
        
        for x in inputs:
            h, c, _ = self.cell.forward(x, h, c)
            y = self.W_y @ h + self.b_y
            
            outputs.append(y)
            hidden_states.append(h.copy())
            
        return outputs, hidden_states


def demonstrate_lstm_gates():
    """Demonstrate how LSTM gates work with visualizations."""
    print("=" * 60)
    print("LSTM Gate Mechanics Demonstration")
    print("=" * 60)
    
    # Create a simple LSTM cell
    input_size = 3
    hidden_size = 4
    lstm_cell = LSTMCell(input_size, hidden_size)
    
    # Generate sample input sequence
    sequence_length = 20
    inputs = [np.random.randn(input_size, 1) for _ in range(sequence_length)]
    
    # Track gate activations
    forget_gates = []
    input_gates = []
    output_gates = []
    cell_states = []
    
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))
    
    for x in inputs:
        combined = np.vstack((h, x))
        
        f = lstm_cell.sigmoid(lstm_cell.W_f @ combined + lstm_cell.b_f)
        i = lstm_cell.sigmoid(lstm_cell.W_i @ combined + lstm_cell.b_i)
        c_tilde = lstm_cell.tanh(lstm_cell.W_c @ combined + lstm_cell.b_c)
        c = f * c + i * c_tilde
        o = lstm_cell.sigmoid(lstm_cell.W_o @ combined + lstm_cell.b_o)
        h = o * lstm_cell.tanh(c)
        
        forget_gates.append(f.mean())
        input_gates.append(i.mean())
        output_gates.append(o.mean())
        cell_states.append(c.mean())
    
    # Visualize gate activations
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(forget_gates, 'r-', linewidth=2)
    plt.title('Forget Gate Activation')
    plt.ylabel('Activation (0-1)')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.subplot(2, 2, 2)
    plt.plot(input_gates, 'g-', linewidth=2)
    plt.title('Input Gate Activation')
    plt.ylabel('Activation (0-1)')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.subplot(2, 2, 3)
    plt.plot(output_gates, 'b-', linewidth=2)
    plt.title('Output Gate Activation')
    plt.ylabel('Activation (0-1)')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.subplot(2, 2, 4)
    plt.plot(cell_states, 'm-', linewidth=2)
    plt.title('Cell State Evolution')
    plt.ylabel('Mean Cell State Value')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/lstm_gru_module/lstm_gates.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gate activation plot saved as 'lstm_gates.png'")
    plt.close()


def sequence_prediction_example():
    """Example: Predicting next value in a sine wave."""
    print("\n" + "=" * 60)
    print("LSTM Sequence Prediction Example: Sine Wave")
    print("=" * 60)
    
    # Generate sine wave data
    t = np.linspace(0, 20, 200)
    data = np.sin(t)
    
    # Prepare sequences (use past 10 points to predict next point)
    seq_length = 10
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    
    # Create LSTM
    lstm = LSTM(input_size=1, hidden_size=20, output_size=1)
    
    # Use first sequence to demonstrate
    test_seq = sequences[0]
    inputs = [np.array([[x]]) for x in test_seq]
    
    outputs, hidden_states = lstm.forward(inputs)
    
    print(f"\nInput sequence length: {len(inputs)}")
    print(f"Hidden state dimension: {hidden_states[0].shape}")
    print(f"Number of LSTM parameters:")
    print(f"  - Forget gate: {lstm.cell.W_f.size + lstm.cell.b_f.size}")
    print(f"  - Input gate: {lstm.cell.W_i.size + lstm.cell.b_i.size}")
    print(f"  - Cell candidate: {lstm.cell.W_c.size + lstm.cell.b_c.size}")
    print(f"  - Output gate: {lstm.cell.W_o.size + lstm.cell.b_o.size}")
    total_params = (lstm.cell.W_f.size + lstm.cell.b_f.size +
                    lstm.cell.W_i.size + lstm.cell.b_i.size +
                    lstm.cell.W_c.size + lstm.cell.b_c.size +
                    lstm.cell.W_o.size + lstm.cell.b_o.size +
                    lstm.W_y.size + lstm.b_y.size)
    print(f"  - Total: {total_params}")


def main():
    """Run all LSTM demonstrations."""
    print("\n" + "=" * 60)
    print("LSTM: Long Short-Term Memory Networks")
    print("=" * 60)
    
    print("\nKey Advantages of LSTM:")
    print("1. Solves vanishing gradient problem")
    print("2. Can learn long-term dependencies")
    print("3. Gates control information flow")
    print("4. Cell state acts as 'memory highway'")
    
    print("\nLSTM vs Traditional RNN:")
    print("- RNN: Simple recurrent connection")
    print("- LSTM: Gated architecture with cell state")
    print("- LSTM has 4x more parameters than RNN")
    
    # Run demonstrations
    demonstrate_lstm_gates()
    sequence_prediction_example()
    
    print("\n" + "=" * 60)
    print("LSTM demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
