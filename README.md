# transformerllm
**Custom Transformer LLM –Lyrics Generator**

**Overview**

This project implements a Transformer-based Language Model (LLM) from scratch using PyTorch.
The model is trained on song lyrics and can generate new lyrics when prompted.

The implementation is divided into five phases for modularity:

Phase 1 – Math Utilities: Core math ops (matmul, dot product, softmax, ReLU, GELU, LayerNorm).

Phase 2 – Neural Network Basics: Basic building blocks (Linear layers, activations).

Phase 3 – Transformer Components: Scaled Dot-Product Attention, Multi-Head Attention, Feed-Forward blocks.

Phase 4 – Transformer Architecture: Full model assembly (stacked Transformer blocks, embeddings, positional encoding).

Phase 5 – Training Pipeline: Data loading, tokenization, training loop, checkpoint saving.

Additional:

tokenizer.py – Word-level tokenizer with padding, <UNK>, <PAD> handling.

inference.py – Load trained model and generate lyrics with top-k sampling.

**Setup Instructions**

1. Clone or Download Project

git clone https://github.com/Tarasure/transformerArch

cd transllm

2. Install Dependencies
   
Make sure Python 3.8+ is installed, then install required libraries:

pip install torch regex

**How to Run**

1. Prepare Dataset

Place your song lyrics inside input.txt in the format:

<|startofsong|>
Your song lyrics here...
<|endofsong|>

2. Train the Model
   
Run the training script:

python phase5_training_pipeline.py

Hyperparameters:

vocab_size = 1024

block_size = 32

d_model = 256

n_heads = 4

n_layers = 4

iterations = 4000 (≈5 min training)

Checkpoint is saved as trained_model.pt.

3. Generate Lyrics
   
Run the inference script:

python inference.py

Example prompt:

Enter a prompt: <|startofsong|> Saying goodbye is

Output(Example):
📝 Generated: startofsong saying goodbye is death by a thousand cuts i get drunk but it's not enough cause you're so hey let's be forever or with a thousand cuts flashbacks dressed like a part of me somewhere we stand but you heard about me somewhere i'm dyin to one for signs in my body my love the flashback starts do waking me juliet you'll come back each time you leave cause darling i'm a nightmare dressed like a daydream so it's gonna be afraid we'll make your hand paper cut stings from our paper thin plans my heart my wine my spirit my trust

**Key Concepts Implemented**

Attention (Query, Key, Value)

Multi-Head Self-Attention

Positional Encoding

Residual Connections + LayerNorm

Feed-Forward Networks

Top-k Sampling for Generation

**Features**

Implemented from scratch (no HuggingFace, minimal PyTorch usage).

Word-level tokenizer with 1024 vocab size (padded if fewer words).

Configurable hyperparameters: embedding size, heads, layers, block size.

Top-k sampling for creative generation.

Support for structured prompts like <|startofsong|>.
