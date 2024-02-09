import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# Hyperparameters
BATCH_SIZE = 32  # Number of independent sequences will be processed in parallel
BLOCK_SIZE = 8  # Max context length for predictions
MAX_ITERS = 10000
EVAL_INTERVAL = 1000
EVAL_ITERS = 200
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(1337)


def load_data(mode="train"):
    # Generate a small batch of data with input x and target y
    if mode == "train":
        data = train_data
    elif mode == "val":
        data = val_data

    ids = torch.randint(
        len(data) - BLOCK_SIZE, (BATCH_SIZE,)
    )  # id of the first character of x
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ids])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ids])
    x, y = x.to(device=DEVICE), y.to(device=DEVICE)

    return x, y


@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval()

    for mode in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = load_data(mode)
            logits, loss = model.forward(x, y)
            losses[i] = loss.item()
        output[mode] = losses.mean()
    model.train()

    return output


class BigramLanguageModel(nn.Module):
    """The simplest model in NLP"""

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, contexts, targets=None):
        logits = self.token_embedding_table(contexts)  # (n_batch, n_block, n_vocab)

        if targets is not None:
            # Reshape the logits and targets
            logits = logits.reshape(-1, 65)  # (n_batch * n_block, n_vocab)
            targets = targets.reshape(-1)  # (n_batch * n_block)
            # Calculate the loss
            loss = F.cross_entropy(logits, targets)

            return logits, loss

        return logits

    def generate(self, contexts, max_new_tokens):
        for _ in range(max_new_tokens):
            # Get the predictions
            logits = self.forward(contexts)
            logits = logits[:, -1, :]  # (n_batch, n_vocab)

            # Get the probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_contexts = torch.multinomial(probs, num_samples=1)  # (n_batch, 1)

            # Append sampled context to the running sequence
            # fmt: off
            contexts = torch.cat((contexts, next_contexts), dim=1) # (n_batch, n_block + 1)
            # fmt: on
        return contexts


if __name__ == "__main__":
    with open("./input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Unique characters occurring in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Create a mapping from characters to integers
    str_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_str = {i: ch for i, ch in enumerate(chars)}

    # Take a string, output a list of integers
    encode = lambda string: [str_to_int[ch] for ch in string]
    # Take a list of integers, output a string
    decode = lambda ints: "".join([int_to_str[i] for i in ints])

    # Encode the entire text dataset and store it into a tensor
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split the data into train and validation sets
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = BigramLanguageModel(vocab_size=vocab_size).to(device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in range(MAX_ITERS + 1):
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # Sample a batch of data
        x, y = load_data(mode="train")

        # Evaluate the loss
        logits, loss = model.forward(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(
        decode(
            model.generate(
                contexts=torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
                max_new_tokens=500,
            )[0].tolist()
        )
    )
