import torch


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
