"""
Training script for GPT-2 on TinyStories.

Usage:
    python -m gpt.train --subset_size 10000 --epochs 5
    python -m gpt.train  # full TinyStories training
"""

import argparse
import os
import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import GPT2Config
from .model import GPT2
from .data import load_tinystories, build_corpus, get_dataloaders


def train(config: GPT2Config, tokenizer, subset_size=None):
    """Main training loop."""

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    train_loader, val_loader = get_dataloaders(
        tokenizer,
        max_len=config.max_seq_len,
        batch_size=config.batch_size,
        subset_size=subset_size,
    )

    # --- Model ---
    model = GPT2(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Optimizer + Scheduler ---
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # --- Loss ---
    criterion = nn.CrossEntropyLoss()

    # --- Training ---
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            logits = model(input_ids)

            # Loss: flatten (batch * seq_len, vocab) vs (batch * seq_len)
            loss = criterion(logits.reshape(-1, config.vocab_size), labels.reshape(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                ppl = math.exp(min(loss.item(), 20))  # cap to avoid overflow
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f} | PPL: {ppl:.1f}")

        avg_loss = total_loss / len(train_loader)
        avg_ppl = math.exp(min(avg_loss, 20))

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, config.vocab_size), labels.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_ppl = math.exp(min(avg_val_loss, 20))

        scheduler.step()

        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {avg_loss:.4f} (PPL {avg_ppl:.1f}) | "
              f"Val Loss: {avg_val_loss:.4f} (PPL {val_ppl:.1f})")

        # --- Generate sample ---
        model.eval()
        prompt = torch.tensor([[tokenizer.bos_id]], dtype=torch.long).to(device)
        generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
        sample_text = tokenizer.decode(generated[0].tolist())
        print(f"  Sample: {sample_text[:200]}")

        # --- Checkpoint ---
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
        }, f"checkpoints/gpt2_epoch_{epoch+1}.pt")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 on TinyStories")
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    args = parser.parse_args()

    config = GPT2Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )

    # TODO: Replace with your trained tokenizer
    # Example:
    #   from gpt.tokenizer import GPTTokenizer
    #   stories = load_tinystories("train", subset_size=args.subset_size)
    #   corpus = build_corpus(stories)
    #   tokenizer = GPTTokenizer()
    #   tokenizer.train(corpus, target_vocab_size=10_000)
    #   config.vocab_size = tokenizer.vocab_size

    print("⚠️  Set up your tokenizer above before running!")
    print("See comments in this file for examples.")
