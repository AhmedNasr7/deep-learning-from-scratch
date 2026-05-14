"""
Training script for VAE experiments.

Usage:
    python -m autoencoders.train --dataset celeba --latent_dim 128 --epochs 50
    python -m autoencoders.train --dataset speech_commands --latent_dim 64
"""

import argparse
import os
import torch
from torch.optim import Adam
from .vae import VAE
from .data import get_celeba_loaders, get_speech_commands_loaders


def kl_weight(epoch, warmup=10):
    return min(1.0, epoch / warmup)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    if args.dataset == "celeba":
        train_loader, val_loader = get_celeba_loaders(args.batch_size)
        in_ch = 3
    elif args.dataset == "speech_commands":
        train_loader, val_loader = get_speech_commands_loaders(args.batch_size)
        in_ch = 1
    else:
        raise ValueError(f"Unknown: {args.dataset}")

    model = VAE(in_channels=in_ch, latent_dim=args.latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    os.makedirs(f"checkpoints/vae_{args.dataset}", exist_ok=True)
    best = float("inf")

    for epoch in range(args.epochs):
        model.train()
        beta = kl_weight(epoch, args.kl_warmup)
        t_loss, t_recon, t_kl = 0, 0, 0

        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            recon, mu, lv = model(imgs)
            L = VAE.loss_function(recon, imgs, mu, lv, beta)
            optimizer.zero_grad(); L["loss"].backward(); optimizer.step()
            t_loss += L["loss"].item(); t_recon += L["recon_loss"]; t_kl += L["kl_loss"]
            if i % 100 == 0:
                print(f"E{epoch+1} B{i} loss={L['loss'].item():.1f} recon={L['recon_loss']:.1f} kl={L['kl_loss']:.1f} β={beta:.2f}")

        n = len(train_loader)
        model.eval(); v = 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                r, m, l = model(imgs)
                v += VAE.loss_function(r, imgs, m, l)["loss"].item()
        avg_v = v / len(val_loader)
        print(f"Epoch {epoch+1}/{args.epochs} train={t_loss/n:.1f} val={avg_v:.1f} β={beta:.2f}")
        if avg_v < best:
            best = avg_v
            torch.save(model.state_dict(), f"checkpoints/vae_{args.dataset}/best.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="celeba", choices=["celeba", "speech_commands"])
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kl_warmup", type=int, default=10)
    args = p.parse_args()
    train(args)
