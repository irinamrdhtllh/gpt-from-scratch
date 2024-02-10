import torch

from methods import load_data, estimate_loss
from models import BigramLanguageModel


# Hyperparameters
BATCH_SIZE = 32  # Number of independent sequences will be processed in parallel
BLOCK_SIZE = 8  # Max context length for predictions
N_EMBED = 32
MAX_ITERS = 10000
EVAL_INTERVAL = 1000
EVAL_ITERS = 200
LEARNING_RATE = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)


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

    model = BigramLanguageModel(
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        n_embed=N_EMBED,
        vocab_size=vocab_size,
        device=device,
    ).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in range(MAX_ITERS + 1):
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(
                model=model,
                eval_iters=EVAL_ITERS,
                batch_size=BATCH_SIZE,
                block_size=BLOCK_SIZE,
                train_data=train_data,
                val_data=val_data,
                device=device,
            )
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # Sample a batch of data
        x, y = load_data(
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
            train_data=train_data,
            val_data=val_data,
            device=device,
        )

        # Evaluate the loss
        logits, loss = model.forward(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(
        decode(
            model.generate(
                x=torch.zeros((1, 1), dtype=torch.long, device=device),
                max_new_tokens=500,
            )[0].tolist()
        )
    )
