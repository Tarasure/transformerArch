
# phase1_math_utils.py

import torch
import math

# 1. Matrix Multiplication
def matmul(a, b):
    return torch.matmul(a, b)

# 2. Dot Product (row-wise if batched)
def dot_product(a, b):
    return torch.sum(a * b, dim=-1)

# 3. Softmax
def softmax(x, dim=-1):
    x_exp = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

# 4. Layer Normalization
def Layer_norm(x, eps=1e-5):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return (x - mean) / (std + eps)

# 5. ReLU Activation
def relu(x):
    return torch.maximum(torch.tensor(0.0), x)

# 6. GELU Activation (approximation)
def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
