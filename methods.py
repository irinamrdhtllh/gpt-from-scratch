import torch


def load_data(batch_size, block_size, train_data, val_data, device, mode="train"):
    # Generate a small batch of data with input x and target y
    if mode == "train":
        data = train_data
    elif mode == "val":
        data = val_data

    ids = torch.randint(
        len(data) - block_size, (batch_size,)
    )  # id of the first character of x
    x = torch.stack([data[i : i + block_size] for i in ids])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ids])
    x, y = x.to(device=device), y.to(device=device)

    return x, y


@torch.no_grad()
def estimate_loss(
    model, eval_iters, batch_size, block_size, train_data, val_data, device
):
    output = {}
    model.eval()

    for mode in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = load_data(
                batch_size=batch_size,
                block_size=block_size,
                train_data=train_data,
                val_data=val_data,
                device=device,
                mode=mode,
            )
            logits, loss = model.forward(x, y)
            losses[i] = loss.item()
        output[mode] = losses.mean()
    model.train()

    return output
