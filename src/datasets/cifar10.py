import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from .augmentations import SimCLRAugmentations

class CIFAR10Pair(CIFAR10):
    """
    Overrides CIFAR10 __getitem__ to return two augmented views + label.
    """
    def __init__(self, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._two_view_transform = transform if transform is not None else SimCLRAugmentations()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        x1, x2 = self._two_view_transform(img)
        return (x1, x2), target

def get_cifar10_dataloaders(batch_size=256, num_workers=2, image_size=32, root="./data"):
    """
    Returns train and test DataLoaders for CIFAR-10 with SimCLR augmentations.
    """
    transform = SimCLRAugmentations(image_size=image_size)

    train_ds = CIFAR10Pair(root=root, train=True, download=True, transform=transform)
    test_ds = CIFAR10Pair(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader

