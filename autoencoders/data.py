"""
Data pipelines for VAE experiments across 3 domains.

Datasets:
  1. CelebA faces     — 64×64 crops, 3 channels, ~200K images
  2. Speech Commands  — mel spectrograms treated as grayscale images
  3. Numerical        — Swiss roll / moons (2D toy data for latent space viz)
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple


# ---------------------------------------------------------------------------
# 1. CelebA Faces (64×64 crop)
# ---------------------------------------------------------------------------

def get_celeba_loaders(batch_size: int = 64, img_size: int = 64,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    CelebA face dataset — center-cropped and resized.

    Downloads to ./data/celeba/ on first run (~1.4 GB).
    If CelebA download fails (Google Drive quota), use the HuggingFace mirror:
        datasets.load_dataset("huggan/CelebA-faces")

    Returns: (train_loader, val_loader)
    """
    transform = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    try:
        dataset = torchvision.datasets.CelebA(
            root="./data", split="train", download=True, transform=transform
        )
        val_dataset = torchvision.datasets.CelebA(
            root="./data", split="valid", download=True, transform=transform
        )
    except Exception:
        # Fallback to HuggingFace mirror
        from datasets import load_dataset

        class HFCelebA(torch.utils.data.Dataset):
            def __init__(self, hf_split, transform):
                self.data = hf_split
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                img = self.data[idx]["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return self.transform(img), 0  # dummy label

        hf = load_dataset("huggan/CelebA-faces")
        n = len(hf["train"])
        val_size = min(5000, n // 10)
        dataset = HFCelebA(hf["train"].select(range(val_size, n)), transform)
        val_dataset = HFCelebA(hf["train"].select(range(val_size)), transform)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 2. Speech Commands → Mel Spectrograms
# ---------------------------------------------------------------------------

def get_speech_commands_loaders(batch_size: int = 64, n_mels: int = 64,
                                target_length: int = 32,
                                num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Google Speech Commands v0.02 — 1-second spoken words.

    Preprocesses waveforms → mel spectrograms → (1, n_mels, target_length).
    Treat as grayscale images for the VAE.

    Labels: 35 words including "yes", "no", "stop", "go", etc.

    Returns: (train_loader, val_loader)
    """
    import torchaudio
    from torchaudio.transforms import MelSpectrogram

    mel_transform = MelSpectrogram(
        sample_rate=16000,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=512,
    )

    class SpectrogramDataset(torch.utils.data.Dataset):
        def __init__(self, subset):
            self.data = subset
            self.mel = mel_transform
            self.target_length = target_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            waveform, sample_rate, label, *_ = self.data[idx]

            # Resample if needed
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

            # Pad/trim to 1 second
            if waveform.shape[1] < 16000:
                waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
            else:
                waveform = waveform[:, :16000]

            # Mel spectrogram → log scale → normalize to [0, 1]
            mel = self.mel(waveform)
            mel = torch.log1p(mel)
            mel = mel / (mel.max() + 1e-8)

            # Pad/trim time axis
            if mel.shape[2] < self.target_length:
                mel = torch.nn.functional.pad(mel, (0, self.target_length - mel.shape[2]))
            else:
                mel = mel[:, :, :self.target_length]

            return mel, label

    train_set = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True, subset="training")
    val_set = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True, subset="validation")

    train_loader = DataLoader(
        SpectrogramDataset(train_set), batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        SpectrogramDataset(val_set), batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 3. Numerical Toy Data (2D — for latent space visualization)
# ---------------------------------------------------------------------------

def get_toy_data_loaders(dataset_name: str = "swiss_roll", n_samples: int = 10000,
                         batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
    """
    2D toy datasets for VAE latent space experiments.

    Supports: "swiss_roll", "moons", "circles", "s_curve"

    Returns: (train_loader, val_loader) of (batch, 2) tensors
    """
    from sklearn.datasets import make_swiss_roll, make_moons, make_circles, make_s_curve

    if dataset_name == "swiss_roll":
        X, color = make_swiss_roll(n_samples, noise=0.3)
        X = X[:, [0, 2]]  # project to 2D
    elif dataset_name == "moons":
        X, color = make_moons(n_samples, noise=0.1)
    elif dataset_name == "circles":
        X, color = make_circles(n_samples, noise=0.05, factor=0.5)
    elif dataset_name == "s_curve":
        X, color = make_s_curve(n_samples, noise=0.1)
        X = X[:, [0, 2]]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Normalize to [-1, 1]
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X = torch.tensor(X, dtype=torch.float32)
    color = torch.tensor(color, dtype=torch.float32)

    val_size = n_samples // 5
    train_data = TensorDataset(X[val_size:], color[val_size:])
    val_data = TensorDataset(X[:val_size], color[:val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
