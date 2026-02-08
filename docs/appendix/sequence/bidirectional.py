#!/usr/bin/env python3
"""
Bidirectional RNN / GRU / LSTM - Processing sequences in both directions
Key idea: run one RNN forward (t=1..T) and another backward (t=T..1),
then combine their outputs (concat or sum).

This file provides:
  - Bidirectional wrapper around a *cell-based* RNN (vanilla), GRU, or LSTM-like module
  - For simplicity and clarity, we implement a bidirectional vanilla RNN here.

File: appendix/sequence/bidirectional.py
Note: Educational, fully commented implementation (batch-first).
"""

import torch
import torch.nn as nn


class RNNCell(nn.Module):
    """
    Reuse a simple vanilla RNN cell.

    h_t = tanh(Wx x_t + Wh h_{t-1})
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.Wx = nn.Linear(input_size, hidden_size, bias=True)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.Wx(x_t) + self.Wh(h_prev))


class BidirectionalRNN(nn.Module):
    """
    Bidirectional vanilla RNN.

    We maintain:
      - forward hidden state h_f (left -> right)
      - backward hidden state h_b (right -> left)

    For each time step t:
      forward:  h_f[t] = f(x[t], h_f[t-1])
      backward: h_b[t] = f(x[t], h_b[t+1])   (computed by iterating reversed time)

    Output combination (common choices):
      - concat: y[t] = [h_f[t], h_b[t]]  -> dimension 2*hidden
      - sum:    y[t] = h_f[t] + h_b[t]  -> dimension hidden

    Here we implement concatenation (most common).
    """
    def __init__(self, input_size: int, hidden_size: int, concat: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.concat = concat

        # Two independent RNN cells: one forward, one backward
        self.cell_f = RNNCell(input_size, hidden_size)
        self.cell_b = RNNCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, h0_f: torch.Tensor | None = None, h0_b: torch.Tensor | None = None):
        """
        x: (B, T, input_size)

        Returns:
          y: (B, T, 2*hidden) if concat=True else (B, T, hidden)
          (hT_f, hT_b): final forward/backward hidden states (B, hidden)
        """
        B, T, _ = x.shape
        device = x.device

        # Initialize hidden states if not provided
        h_f = torch.zeros(B, self.hidden_size, device=device) if h0_f is None else h0_f
        h_b = torch.zeros(B, self.hidden_size, device=device) if h0_b is None else h0_b

        # ---- Forward pass (t = 0..T-1) ----
        forward_states = []
        for t in range(T):
            x_t = x[:, t, :]               # (B, input)
            h_f = self.cell_f(x_t, h_f)    # update forward hidden
            forward_states.append(h_f)     # store h_f[t]

        # ---- Backward pass (t = T-1..0) ----
        backward_states_reversed = []
        for t in reversed(range(T)):
            x_t = x[:, t, :]               # (B, input)
            h_b = self.cell_b(x_t, h_b)    # update backward hidden (moving right->left)
            backward_states_reversed.append(h_b)  # this is h_b[t], but collected reversed

        # Reverse backward list so it aligns with time order 0..T-1
        backward_states = list(reversed(backward_states_reversed))

        # ---- Combine forward and backward states per time step ----
        y_steps = []
        for t in range(T):
            hf_t = forward_states[t]       # (B, hidden)
            hb_t = backward_states[t]      # (B, hidden)

            if self.concat:
                # Concatenate along feature dimension -> (B, 2*hidden)
                y_t = torch.cat([hf_t, hb_t], dim=1)
            else:
                # Sum -> (B, hidden)
                y_t = hf_t + hb_t

            y_steps.append(y_t)

        # Stack to (B, T, feat_dim)
        y = torch.stack(y_steps, dim=1)

        # Final hidden states (after last updates in each direction)
        hT_f = forward_states[-1]          # (B, hidden)
        hT_b = backward_states[0]          # (B, hidden)  (backward final corresponds to t=0 in aligned order)

        return y, (hT_f, hT_b)


if __name__ == "__main__":
    # Quick sanity check
    model = BidirectionalRNN(input_size=8, hidden_size=16, concat=True)
    x = torch.randn(2, 5, 8)

    y, (hF, hB) = model(x)
    print("y :", y.shape)   # expected (2, 5, 32) when concat=True
    print("hF:", hF.shape)  # expected (2, 16)
    print("hB:", hB.shape)  # expected (2, 16)
