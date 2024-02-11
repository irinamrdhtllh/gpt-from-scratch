import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, block_size, channel_size = x.shape

        k = self.key(x)
        q = self.query(x)

        # Compute attention scores (affinities)
        weights = q @ k.transpose(-2, -1) * channel_size**-0.5
        weights = weights.masked_fill(
            self.tril[:block_size, :block_size] == 0, float("-inf")
        )
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # Perform the weighted aggregation of the values
        v = self.value(x)
        output = weights @ v

        return output


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, n_heads, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(
                    head_size=head_size,
                    n_embed=n_embed,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_heads)
            ]
        )
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.projection(output)
        output = self.dropout(output)

        return output


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, n_embed, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_heads
        self.self_attention = MultiHeadAttention(
            n_heads=n_heads,
            head_size=head_size,
            n_embed=n_embed,
            block_size=block_size,
            dropout=dropout,
        )
        self.feed_forward = FeedForward(n_embed=n_embed, dropout=dropout)
        self.norm_1 = nn.LayerNorm(n_embed)
        self.norm_2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attention(self.norm_1(x))
        x = x + self.feed_forward(self.norm_2(x))

        return x


class BigramLanguageModel(nn.Module):
    """The simplest model in NLP"""

    def __init__(
        self,
        n_layers,
        batch_size,
        block_size,
        n_embed,
        n_heads,
        vocab_size,
        dropout,
        device,
    ):
        super().__init__()
        self.block_size = block_size
        self.n_embed = n_embed
        self.vocab_size = vocab_size
        self.device = device

        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embed=n_embed,
                    n_heads=n_heads,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x, y=None):
        batch_size, block_size = x.shape
        token_embed = self.token_embedding_table(x)  # (n_batch, n_block, n_embed)
        pos_embed = self.position_embedding_table(
            torch.arange(block_size, device=self.device)
        )  # (n_block, n_embed)
        inputs = token_embed + pos_embed  # (n_batch, n_block, n_embed)
        inputs = self.blocks(inputs)
        inputs = self.final_norm(inputs)
        logits = self.lm_head(inputs)  # (n_batch, n_block, n_vocab)

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
            # Crop the x to the last block of tokens
            x_cond = x[:, -self.block_size :]

            # Get the predictions
            logits = self.forward(x_cond)
            logits = logits[:, -1, :]  # (n_batch, n_vocab)

            # Get the probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_x = torch.multinomial(probs, num_samples=1)  # (n_batch, 1)

            # Append sampled contexts to the running sequence
            x = torch.cat((x, next_x), dim=1)  # (n_batch, n_block + 1)

        return x
