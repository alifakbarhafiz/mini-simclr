import os
import torch
from tqdm import tqdm
from typing import Optional

from src.losses.nt_xent import NTXentLoss

def train_simclr(
    encoder,
    projector,
    dataloader,
    device: str = "cpu",
    lr: float = 3e-4,
    temperature: float = 0.5,
    epochs: int = 10,
    save_path: Optional[str] = None,
):
    """
    Train SimCLR given encoder and projection head.

    Args:
        encoder: backbone producing features (B, feat_dim)
        projector: projection MLP producing (B, proj_dim)
        dataloader: yields ((x1,x2), label)
        device: "cpu" or "cuda"
        lr: learning rate
        temperature: contrastive loss temperature
        epochs: number of training epochs
        save_path: optional directory to save checkpoints
    """
    encoder = encoder.to(device)
    projector = projector.to(device)
    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = NTXentLoss(
        batch_size=dataloader.batch_size,
        temperature=temperature,
        device=device,
    ).to(device)

    encoder.train()
    projector.train()

    for epoch in range(epochs):
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for (x1, x2), _ in loop:
            x1, x2 = x1.to(device), x2.to(device)

            h1, h2 = encoder(x1), encoder(x2)
            z1, z2 = projector(h1), projector(h2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (loop.n + 1))

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished — avg loss: {avg_loss:.4f}")

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            ckpt = {
                "epoch": epoch + 1,
                "encoder_state": encoder.state_dict(),
                "projector_state": projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(save_path, f"simclr_epoch{epoch+1}.pth"))
            print(f"[Checkpoint] Saved at {save_path}/simclr_epoch{epoch+1}.pth")


