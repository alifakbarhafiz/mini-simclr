import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple

from src.utils.visualization import plot_embeddings_tsne, plot_embeddings_umap

def train_linear_classifier(encoder, dataloader, num_classes: int, device="cpu", epochs=10, lr=1e-3) -> nn.Module:
    """
    Freeze encoder and train a linear classifier on top.
    Returns trained linear module.
    """
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        (x1, _), _ = sample_batch
        feat = encoder(x1.to(device))
        feat_dim = feat.shape[1]

    linear = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = optim.Adam(linear.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        linear.train()
        total_loss, correct, total = 0.0, 0, 0
        loop = tqdm(dataloader, desc=f"Linear Epoch {epoch+1}/{epochs}")
        for (x1, x2), labels in loop:
            x = x1.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                feats = encoder(x)

            logits = linear(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=total_loss/(loop.n+1), acc=100.*correct/total)

        print(f"Linear Epoch {epoch+1}: Loss {total_loss/len(dataloader):.4f} Acc {100.*correct/total:.2f}%")

    return linear

def evaluate_and_visualize(encoder, dataloader, device="cpu", use_tsne=True, use_umap=True):
    """
    Extracts embeddings from encoder using dataloader and runs t-SNE/UMAP visualizations.
    """
    encoder = encoder.to(device)
    encoder.eval()

    all_feats = []
    all_labels = []
    with torch.no_grad():
        for (x1, x2), labels in tqdm(dataloader, desc="Extracting features"):
            feats = encoder(x1.to(device))
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"Extracted {feats.shape[0]} embeddings (dim {feats.shape[1]})")

    if use_tsne:
        plot_embeddings_tsne(feats, labels)
    if use_umap:
        plot_embeddings_umap(feats, labels)


