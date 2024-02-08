import torch

torch.manual_seed(1337)


BATCH_SIZE = 4  # Number of independent sequences will be processed in parallel
BLOCK_SIZE = 8  # Max context length for predictions


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

    return x, y


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

    x, y = load_data(mode="train")

    for batch in range(BATCH_SIZE):
        for block in range(BLOCK_SIZE):
            context = x[batch, : block + 1]
            target = y[batch, block]
            print(f"when the input is {context}, the target is {target}")
