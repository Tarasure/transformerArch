# # inference.py
import torch
from tokenizer import encode, decode
from phase4_transformer_arch import TransformerModel
block_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load checkpoint
checkpoint = torch.load("trained_model.pt", map_location=device)
vocab_size = checkpoint["vocab_size"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

model = TransformerModel(vocab_size, d_model=256, n_heads=4, n_layers=4, block_size=32).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(prompt, max_new_tokens=200, top_k=40, repetition_penalty=1.4):
    if not prompt.startswith("<|startofsong|>"):
        prompt = "<|startofsong|>\n" + prompt
    model.eval()
    input_ids = encode(prompt, stoi)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    generated = set(input_ids)  # Track used tokens

    for _ in range(max_new_tokens):
        input_crop = input_tensor[:, -block_size:]
        with torch.no_grad():
            logits = model(input_crop)
        
        logits = logits[:, -1, :]

        # Apply repetition penalty
        for token_id in generated:
            logits[0, token_id] /= repetition_penalty  # Penalize previously used tokens

        # Top-k sampling
        topk = torch.topk(logits, k=top_k, dim=-1)
        probs = torch.softmax(topk.values, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        next_token = topk.indices.gather(-1, sampled)

        # Append next token
        input_tensor = torch.cat([input_tensor, next_token], dim=1)
        generated.add(next_token.item())  # Track repetition

    output_ids = input_tensor[0].tolist()
    return decode(output_ids, itos)

if __name__ == "__main__":
    while(True):
        prompt = input("Enter a prompt: ")
        output = generate(prompt, max_new_tokens=100)
        print("📝 Generated:", output)
