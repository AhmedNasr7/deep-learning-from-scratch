"""
Training script for Transformer (Arabic ↔ English translation).

Usage:
    python -m transformer.train --subset_size 10000 --epochs 5
    python -m transformer.train  # full OPUS-100 training

Assembles: tokenizer → data → model → training loop.
"""

import argparse
import os
import math

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from .config import TransformerConfig
from .model import Transformer
from .data import load_opus100, get_dataloaders

# --- Tokenizer Setup ---
# Import your tokenizer of choice:
# from tokenizers import NaiveBPETokenizer, EfficientBPETokenizer, CharTokenizer
#
# For a quick start with char-level:
#   src_tokenizer = CharTokenizer().train(arabic_corpus)
#   tgt_tokenizer = CharTokenizer().train(english_corpus)
#
# For BPE:
#   src_tokenizer = NaiveBPETokenizer().train(arabic_corpus, target_vocab_size=8000)
#   tgt_tokenizer = NaiveBPETokenizer().train(english_corpus, target_vocab_size=8000)


def get_lr_scheduler(optimizer, d_model: int, warmup_steps: int):
    """
    Transformer learning rate schedule from "Attention Is All You Need":
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def lr_lambda(step):
        step = max(step, 1)
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    return LambdaLR(optimizer, lr_lambda)


def train(config: TransformerConfig, src_tokenizer, tgt_tokenizer, subset_size=None):
    """Main training loop."""

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    train_loader, val_loader = get_dataloaders(
        src_tokenizer, tgt_tokenizer,
        max_len=config.max_seq_len,
        batch_size=config.batch_size,
        subset_size=subset_size,
    )

    # --- Model ---
    model = Transformer(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Optimizer + Scheduler ---
    optimizer = Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_lr_scheduler(optimizer, config.d_model, config.warmup_steps)

    # --- Loss (ignore padding) ---
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id)

    # --- Training ---
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            src = batch["src_ids"].to(device)
            tgt = batch["tgt_ids"].to(device)

            # Teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            # Generate masks
            src_mask = Transformer.generate_padding_mask(src, config.pad_id)
            tgt_mask = Transformer.generate_causal_mask(tgt_input.size(1)).to(device)

            # Forward
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # Loss: flatten (batch * seq_len, vocab) vs (batch * seq_len)
            loss = criterion(logits.reshape(-1, config.tgt_vocab_size), tgt_target.reshape(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.epochs} | Avg Train Loss: {avg_loss:.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["src_ids"].to(device)
                tgt = batch["tgt_ids"].to(device)
                tgt_input = tgt[:, :-1]
                tgt_target = tgt[:, 1:]

                src_mask = Transformer.generate_padding_mask(src, config.pad_id)
                tgt_mask = Transformer.generate_causal_mask(tgt_input.size(1)).to(device)

                logits = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(logits.reshape(-1, config.tgt_vocab_size), tgt_target.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{config.epochs} | Avg Val Loss: {avg_val_loss:.4f}")

        # --- Checkpoint ---
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
        }, f"checkpoints/transformer_epoch_{epoch+1}.pt")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer on OPUS-100 ar-en")
    parser.add_argument("--subset_size", type=int, default=None, help="Use first N examples (fast iteration)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    args = parser.parse_args()

    config = TransformerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
    )

    # TODO: Replace with your trained tokenizers
    # Example:
    #   from tokenizers import CharTokenizer
    #   corpus = load_opus100("train", subset_size=args.subset_size)
    #   ar_texts = " ".join([p["ar"] for p in corpus])
    #   en_texts = " ".join([p["en"] for p in corpus])
    #   src_tokenizer = CharTokenizer().train(ar_texts)
    #   tgt_tokenizer = CharTokenizer().train(en_texts)
    #   config.src_vocab_size = src_tokenizer.vocab_size
    #   config.tgt_vocab_size = tgt_tokenizer.vocab_size

    print("⚠️  Set up your tokenizers above before running!")
    print("See comments in this file for examples.")
