# src/main.py
import argparse
import torch
from torch.utils.data import DataLoader
import os

from src.datasets.cifar10 import get_cifar10_dataloaders
from src.models.resnet import get_resnet
from src.models.projection import ProjectionHead
from src.training.trainer import train_simclr
from src.training.linear_eval import train_linear_classifier, evaluate_and_visualize

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--save_path", type=str, default="outputs/checkpoints")
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_path, exist_ok=True)

    train_loader, test_loader = get_cifar10_dataloaders(batch_size=args.batch_size)

    encoder, feat_dim = get_resnet("resnet18", pretrained=False)
    projector = ProjectionHead(input_dim=feat_dim, hidden_dim=512, out_dim=128)

    train_simclr(encoder, projector, train_loader, device=device, lr=args.lr, temperature=args.temperature, epochs=args.epochs, save_path=args.save_path)

    final_ckpt_path = f"{args.save_path}/simclr_final.pth"
    ckpt_data = {"encoder_state": encoder.state_dict()}
    
    torch.save(ckpt_data, final_ckpt_path)
    print(f"[Checkpoint] Saved final encoder state at {final_ckpt_path}")

    linear = train_linear_classifier(encoder, train_loader, num_classes=10, device=device, epochs=5, lr=1e-3)
    evaluate_and_visualize(encoder, test_loader, device=device, use_tsne=True, use_umap=True)

if __name__ == "__main__":
    main()
