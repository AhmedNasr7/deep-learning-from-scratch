"""
Data loaders for ViT experiments.

Datasets:
  - CIFAR-10:       32×32 images, 10 classes (torchvision)
  - Tiny ImageNet:  64×64 images, 200 classes (HuggingFace)
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = -1) -> tuple:
    """
    CIFAR-10 with standard augmentation.

    Returns: (train_loader, val_loader)
    """
    # num_workers=0 required on macOS MPS to avoid DataLoader deadlocks
    if num_workers < 0:
        num_workers = 0 if torch.backends.mps.is_available() else 4

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def get_tiny_imagenet_loaders(batch_size: int = 128, num_workers: int = -1) -> tuple:
    """
    Tiny ImageNet (200 classes, 64×64) via HuggingFace.

    Returns: (train_loader, val_loader)
    """
    if num_workers < 0:
        num_workers = 0 if torch.backends.mps.is_available() else 4

    from datasets import load_dataset

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    class TinyImageNetDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform):
            self.data = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            image = item["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.transform(image)
            return image, item["label"]

    hf_dataset = load_dataset("zh-plus/tiny-imagenet")

    train_dataset = TinyImageNetDataset(hf_dataset["train"], train_transform)
    val_dataset = TinyImageNetDataset(hf_dataset["valid"], val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
