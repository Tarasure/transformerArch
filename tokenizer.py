# # tokenizer.py
import re

def build_tokenizer(file_path, vocab_size=1024):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    # Clean and split text into words
    words = re.findall(r"\b\w+'\w+|\w+\b", text)
    unique_words = list(set(words))

    # Truncate or pad vocab
    pad_size = max(0, vocab_size - len(unique_words))
    vocab = unique_words[:vocab_size] + [f"<PAD{i}>" for i in range(pad_size)]
    
    # Add <UNK> if not present
    if "<UNK>" not in vocab:
        vocab[0] = "<UNK>"

    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for word, i in stoi.items()}

    return vocab, stoi, itos

def encode(text, stoi):
    words = re.findall(r"\b\w+'\w+|\w+\b", text.lower())
    return [stoi.get(w, stoi["<UNK>"]) for w in words]

def decode(indices, itos):
    return ' '.join([itos[i] for i in indices if itos[i] not in {"<UNK>"}])
