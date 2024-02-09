import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """The simplest model in NLP"""

    def __init__(self, block_size, vocab_size, device):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device

        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, y=None):
        logits = self.token_embedding_table(x)

        if y is not None:
            # Reshape the logits and targets
            logits = logits.reshape(-1, self.vocab_size)  # (n_batch * n_block, n_vocab)
            y = y.reshape(-1)  # (n_batch * n_block)
            # Calculate the loss
            loss = F.cross_entropy(logits, y)

            return logits, loss

        return logits

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # Get the predictions
            logits = self.forward(x)
            logits = logits[:, -1, :]  # (n_batch, n_vocab)

            # Get the probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_x = torch.multinomial(probs, num_samples=1)  # (n_batch, 1)

            # Append sampled contexts to the running sequence
            x = torch.cat((x, next_x), dim=1)  # (n_batch, n_block + 1)
        return x
