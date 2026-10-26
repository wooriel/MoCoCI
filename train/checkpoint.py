import torch


def save_checkpoint(model, optimizer, epoch, path, val_loss=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if val_loss is not None:
        checkpoint["val_loss"] = val_loss

    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    start_epoch = checkpoint["epoch"] + 1

    val_loss = checkpoint.get("val_loss", None)

    print(
        "Loaded checkpoint from epoch {}".format(checkpoint['epoch'])
    )

    return start_epoch, val_loss