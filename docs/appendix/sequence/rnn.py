#!/usr/bin/env python3
"""
Vanilla RNN - Recurrent Neural Network (Elman RNN)
Classic idea: maintain a hidden state that is updated sequentially over time.

Reference: "Finding Structure in Time" (1990), Jeffrey L. Elman (popularized simple RNN)
Key: h_t = tanh(W_x x_t + W_h h_{t-1} + b)

File: appendix/sequence/rnn.py
Note: Educational, fully commented implementation (single-layer, batch-first).
"""

import torch
import torch.nn as nn


class RNNCell(nn.Module):
    """
    A single vanilla RNN cell (one time step).

    Shapes:
      x_t     : (B, input_size)
      h_prev  : (B, hidden_size)
      h_t     : (B, hidden_size)

    Update:
      h_t = tanh( W_x * x_t + W_h * h_{t-1} + b )
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Linear layers to transform input and previous hidden state
        self.Wx = nn.Linear(input_size, hidden_size, bias=True)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # Combine transformed input and hidden state, then apply tanh nonlinearity
        h_t = torch.tanh(self.Wx(x_t) + self.Wh(h_prev))
        return h_t


class RNN(nn.Module):
    """
    Vanilla RNN (manual unroll across time).

    Input:
      x : (B, T, input_size)  batch-first sequence
    Output:
      y : (B, T, hidden_size) all hidden states across time
      h_T : (B, hidden_size) final hidden state
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        # Extract batch size and time length
        B, T, _ = x.shape
        device = x.device

        # Initialize hidden state (zeros) if not provided
        if h0 is None:
            h_t = torch.zeros(B, self.hidden_size, device=device)
        else:
            h_t = h0

        outputs = []
        for t in range(T):
            # Take input for time step t
            x_t = x[:, t, :]               # (B, input_size)

            # Update hidden state using the RNN cell
            h_t = self.cell(x_t, h_t)      # (B, hidden_size)

            # Store hidden state for this time step
            outputs.append(h_t)

        # Stack hidden states across time into a single tensor
        y = torch.stack(outputs, dim=1)     # (B, T, hidden_size)
        return y, h_t


if __name__ == "__main__":
    # Quick sanity check: run a forward pass and print shapes
    model = RNN(input_size=8, hidden_size=16)
    x = torch.randn(2, 5, 8)     # (B=2, T=5, input=8)
    y, hT = model(x)

    print("y :", y.shape)        # expected (2, 5, 16)
    print("hT:", hT.shape)       # expected (2, 16)
