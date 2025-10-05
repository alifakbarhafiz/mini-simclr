# src/losses/nt_xent.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    Expects two batches zis, zjs of shape (batch_size, dim).
    """

    def __init__(self, batch_size: int, temperature: float = 0.5, device: str = "cpu"):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.register_buffer("mask", self._get_correlated_mask(batch_size), persistent=False)

    def _get_correlated_mask(self, batch_size: int):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask.fill_diagonal_(False)
        for i in range(batch_size):
            mask[i, i + batch_size] = False
            mask[i + batch_size, i] = False
        return mask

def forward(self, zis: torch.Tensor, zjs: torch.Tensor) -> torch.Tensor:
    """
    zis, zjs: (batch_size, dim)
    returns scalar loss
    """
    batch_size = zis.shape[0]
    assert batch_size == self.batch_size, "Batch size mismatch between loss and inputs."

    z = torch.cat([zis, zjs], dim=0)  # (2B, D)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    positives = torch.cat(
        [torch.diag(sim, self.batch_size), torch.diag(sim, -self.batch_size)],
        dim=0
    ).unsqueeze(1)  # (2B,1)

    mask = self.mask.to(sim.device).bool()

    negatives = sim[mask].view(2 * batch_size, -1)  # (2B, 2B-2)

    logits = torch.cat([positives, negatives], dim=1)  # (2B, 1 + 2B - 2)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=sim.device)  # positives are at index 0

    loss = self.criterion(logits, labels)
    loss = loss / (2 * batch_size)
    return loss



