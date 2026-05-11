"""
Data pipeline for Arabic ↔ English translation using OPUS-100.

Dataset: Helsinki-NLP/opus-100, config "ar-en"
Format:  {'translation': {'ar': '...', 'en': '...'}}

Uses your BPE tokenizer — train two instances (Arabic, English) on
the training corpus, or one shared bilingual tokenizer.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any


class TranslationDataset(Dataset):
    """
    Wraps OPUS-100 ar-en. Tokenizes on-the-fly using provided tokenizers.

    Args:
        data:          list of {'ar': str, 'en': str} pairs
        src_tokenizer: tokenizer for source language (Arabic)
        tgt_tokenizer: tokenizer for target language (English)
        max_len:       max sequence length (truncate or pad to this)

    Each __getitem__ returns:
        {
            'src_ids':  (max_len,) LongTensor  — <BOS> src tokens <EOS> <PAD...>
            'tgt_ids':  (max_len,) LongTensor  — <BOS> tgt tokens <EOS> <PAD...>
            'src_mask': (max_len,) BoolTensor  — True where NOT padded
            'tgt_mask': (max_len,) BoolTensor  — True where NOT padded
        }
    """

    def __init__(self, data: List[Dict[str, str]], src_tokenizer, tgt_tokenizer, max_len: int = 128):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.data[idx]
        src_text = pair["ar"]
        tgt_text = pair["en"]

        # Tokenize
        src_ids = self.src_tokenizer.encode(src_text)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text)

        # Truncate to max_len - 2 (room for BOS/EOS)
        src_ids = src_ids[: self.max_len - 2]
        tgt_ids = tgt_ids[: self.max_len - 2]

        # Add BOS/EOS
        # NOTE: Adjust these if your tokenizer uses different special token IDs
        # Default assumption: 0=PAD, 1=UNK, 2=BOS, 3=EOS (from your CharTokenizer)
        bos, eos, pad = 2, 3, 0
        src_ids = [bos] + src_ids + [eos]
        tgt_ids = [bos] + tgt_ids + [eos]

        # Pad
        src_pad_len = self.max_len - len(src_ids)
        tgt_pad_len = self.max_len - len(tgt_ids)

        src_mask = [True] * len(src_ids) + [False] * src_pad_len
        tgt_mask = [True] * len(tgt_ids) + [False] * tgt_pad_len

        src_ids = src_ids + [pad] * src_pad_len
        tgt_ids = tgt_ids + [pad] * tgt_pad_len

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_mask": torch.tensor(src_mask, dtype=torch.bool),
            "tgt_mask": torch.tensor(tgt_mask, dtype=torch.bool),
        }


def load_opus100(split: str = "train", subset_size: int = None) -> List[Dict[str, str]]:
    """
    Load OPUS-100 ar-en from HuggingFace.

    Args:
        split:       "train", "validation", or "test"
        subset_size: if set, only load first N examples (for fast iteration)

    Returns:
        list of {'ar': str, 'en': str} dicts
    """
    from datasets import load_dataset

    ds = load_dataset("Helsinki-NLP/opus-100", "ar-en", split=split)

    data = []
    for i, example in enumerate(ds):
        if subset_size and i >= subset_size:
            break
        data.append(example["translation"])

    return data


def get_dataloaders(
    src_tokenizer,
    tgt_tokenizer,
    max_len: int = 128,
    batch_size: int = 64,
    subset_size: int = None,
    num_workers: int = 0,
) -> tuple:
    """
    Build train and validation DataLoaders.

    Args:
        src_tokenizer: trained tokenizer for Arabic
        tgt_tokenizer: trained tokenizer for English
        max_len:       max sequence length
        batch_size:    batch size
        subset_size:   load only first N examples per split (None = all)
        num_workers:   DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    train_data = load_opus100("train", subset_size)
    val_data = load_opus100("validation", subset_size=min(subset_size or 5000, 5000))

    train_dataset = TranslationDataset(train_data, src_tokenizer, tgt_tokenizer, max_len)
    val_dataset = TranslationDataset(val_data, src_tokenizer, tgt_tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
