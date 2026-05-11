"""
Data pipeline for GPT-2 pretraining on TinyStories.

Dataset: roneneldan/TinyStories (~2.1M short stories)
Proven to produce coherent small LMs even at 1M–33M params.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List


class TextDataset(Dataset):
    """
    Tokenizes text and chunks into fixed-length sequences for LM training.

    Each item returns:
        input_ids: (max_len,)   — the sequence
        labels:    (max_len,)   — shifted by 1 (predict next token)

    Args:
        tokens:  pre-tokenized list of token IDs (the full corpus as one flat list)
        max_len: context window size
    """

    def __init__(self, tokens: List[int], max_len: int = 512):
        self.tokens = tokens
        self.max_len = max_len
        # Number of complete chunks we can extract
        self.n_chunks = (len(tokens) - 1) // max_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.max_len
        end = start + self.max_len

        input_ids = torch.tensor(self.tokens[start:end], dtype=torch.long)
        labels = torch.tensor(self.tokens[start + 1:end + 1], dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}


def load_tinystories(split: str = "train", subset_size: int = None) -> List[str]:
    """
    Load TinyStories from HuggingFace.

    Args:
        split:       "train" or "validation"
        subset_size: only load first N stories (for fast iteration)

    Returns:
        list of story strings
    """
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split=split)

    stories = []
    for i, example in enumerate(ds):
        if subset_size and i >= subset_size:
            break
        stories.append(example["text"])

    return stories


def load_wikitext103(split: str = "train", subset_size: int = None) -> List[str]:
    """
    Load WikiText-103 from HuggingFace — bigger English corpus.

    ~100M tokens of Wikipedia articles. Good step up from TinyStories
    when you want to test scaling or need richer vocabulary.

    Dataset: wikitext, config wikitext-103-raw-v1
    Splits:  train (~1.8M lines), validation (~3.7K lines), test (~4.3K lines)

    Args:
        split:       "train", "validation", or "test"
        subset_size: only load first N paragraphs

    Returns:
        list of text strings (paragraphs)
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    texts = []
    for i, example in enumerate(ds):
        if subset_size and i >= subset_size:
            break
        text = example["text"].strip()
        if text:  # skip empty lines
            texts.append(text)

    return texts


def load_arabic_wiki(subset_size: int = None) -> List[str]:
    """
    Load Arabic Wikipedia articles from HuggingFace.

    Clean Arabic text, ~300K+ articles. Manageable for small GPT experiments.

    Dataset: wikimedia/wikipedia, config 20231101.ar

    Args:
        subset_size: only load first N articles (recommended: 10K–50K for experiments)

    Returns:
        list of article text strings
    """
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train")

    texts = []
    for i, example in enumerate(ds):
        if subset_size and i >= subset_size:
            break
        text = example["text"].strip()
        if len(text) > 100:  # skip stubs
            texts.append(text)

    return texts


def build_corpus(texts: List[str], separator: str = "\n") -> str:
    """Join texts into a single training corpus."""
    return separator.join(texts)


def get_dataloaders(
    tokenizer,
    max_len: int = 512,
    batch_size: int = 32,
    subset_size: int = None,
    val_subset: int = 5000,
    num_workers: int = 0,
) -> tuple:
    """
    Build train and validation DataLoaders.

    Pipeline:
        1. Load TinyStories
        2. Join into corpus string
        3. Tokenize with your BPE tokenizer
        4. Chunk into fixed-length sequences
        5. Wrap in DataLoader

    Args:
        tokenizer:   trained tokenizer with .encode(text) -> List[int]
        max_len:     context window
        batch_size:  batch size
        subset_size: limit training stories (None = all)
        val_subset:  limit validation stories
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    # Load and tokenize
    train_stories = load_tinystories("train", subset_size)
    val_stories = load_tinystories("validation", val_subset)

    train_corpus = build_corpus(train_stories)
    val_corpus = build_corpus(val_stories)

    train_tokens = tokenizer.encode(train_corpus)
    val_tokens = tokenizer.encode(val_corpus)

    print(f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")

    # Build datasets
    train_dataset = TextDataset(train_tokens, max_len)
    val_dataset = TextDataset(val_tokens, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
