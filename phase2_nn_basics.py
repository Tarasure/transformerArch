
# phase2_nn_basics.py

import torch
import torch.nn as nn

# Fully connected linear layer with optional bias
class MyLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return x @ self.weight + self.bias

# Simple feedforward block with ReLU or GELU
class FeedForward(nn.Module):
    def __init__(self, dim, use_gelu=True):
        super().__init__()
        self.fc1 = MyLinear(dim, 4 * dim)
        self.fc2 = MyLinear(4 * dim, dim)
        self.activation = nn.GELU() if use_gelu else nn.ReLU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

# Training step with backpropagation
def training_step(model, x, y, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()
