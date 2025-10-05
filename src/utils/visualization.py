import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    from umap import UMAP
except ImportError:
    try:
        from umap.umap_ import UMAP
    except ImportError:
        UMAP = None 


def visualize_augmentations(dataset, n=5):
    fig, axes = plt.subplots(n, 2, figsize=(5, 2*n))

    for i in range(n):
        (x_i, x_j), _ = dataset[i] 
        
        axes[i, 0].imshow(x_i.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[i, 0].set_title("Aug 1")
        axes[i, 1].imshow(x_j.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[i, 1].set_title("Aug 2")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_embeddings_tsne(embeddings, labels, n_samples=2000, perplexity=30, random_state=42, save_path="outputs/tsne_visualization.png"):
    """
    Plots embeddings using t-SNE and saves the plot to a file.
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    if len(embeddings_np) > n_samples:
        idx = np.random.permutation(len(embeddings_np))[:n_samples]
        embeddings_np = embeddings_np[idx]
        labels_np = labels_np[idx]

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    reduced = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels_np, cmap="tab10", s=5, alpha=0.7)
    plt.legend(*scatter.legend_elements(), loc="best", title="Classes")
    plt.title(f"t-SNE visualization (N={len(embeddings_np)})")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close() 
    print(f"Saved t-SNE plot to {save_path}")


def plot_embeddings_umap(embeddings: torch.Tensor, labels: torch.Tensor, n_samples:int = 2000, n_neighbors:int = 15, min_dist:float = 0.1, random_state:int = 42, save_path="outputs/umap_visualization.png"):
    """
    Plots embeddings using UMAP and saves the plot to a file.
    """
    if UMAP is None:
        print("UMAP is not installed. Skipping UMAP visualization.")
        return

    emb = embeddings.detach().cpu().numpy()
    lab = labels.detach().cpu().numpy()

    if len(emb) > n_samples:
        idx = np.random.permutation(len(emb))[:n_samples]
        emb = emb[idx]; lab = lab[idx]

    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    reduced: np.ndarray = reducer.fit_transform(emb) 

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=lab, # Use the sampled labels
        cmap="tab10",
        s=5,
        alpha=0.8
    )
    plt.legend(*scatter.legend_elements(), loc="best", title="Classes")
    plt.title(f"UMAP visualization (N={len(emb)})")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close() 
    print(f"Saved UMAP plot to {save_path}")
