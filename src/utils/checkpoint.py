# src/utils/checkpoint.py
import os
import torch

def save_checkpoint(model, optimizer, epoch, path="../outputs/checkpoints/simclr_latest.pth"):
    """
    Save model + optimizer state_dict to disk.
    Args:
        model: nn.Module
        optimizer: torch.optim.Optimizer
        epoch: int (current epoch number)
        path: str (file path to save)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, path)
    print(f"[Checkpoint] Saved at {path}")

def load_checkpoint(model, optimizer=None, path="../outputs/checkpoints/simclr_latest.pth", device="cpu"):
    """
    Load model + (optionally) optimizer state_dict.
    Args:
        model: nn.Module (with same architecture as saved checkpoint)
        optimizer: torch.optim.Optimizer or None
        path: str (file path to load)
        device: str ("cpu" or "cuda")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint.get("epoch", -1)
    print(f"[Checkpoint] Loaded from {path} (epoch {epoch})")
    return model, optimizer, epoch



