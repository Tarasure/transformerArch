
import torch
import torch.nn as nn
import torch.optim as optim
from tokenizer import build_tokenizer, encode, decode
from phase4_transformer_arch import TransformerModel

vocab_size = 1024
block_size = 32
d_model = 256
n_heads = 4
n_layers = 4
iterations = 4000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab, stoi, itos = build_tokenizer("input.txt", vocab_size)

def cross_entropy_loss(logits, targets):
    B, T, V = logits.shape
    logits = logits.view(B * T, V)
    targets = targets.view(B * T)
    return nn.functional.cross_entropy(logits, targets)

def train_model():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    encoded = encode(text, stoi)
    data = torch.tensor(encoded, dtype=torch.long)

    model = TransformerModel(vocab_size, d_model, n_heads, n_layers, block_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(iterations):
        ix = torch.randint(0, len(data) - block_size - 1, (32,))
        x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+1+block_size] for i in ix]).to(device)

        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size
    }, "trained_model.pt")
    print("✅ Model saved to trained_model.pt")

if __name__ == "__main__":
    train_model()
