# Implementation Guide

## Phase 1: Shared Foundations

Implement these first — everything else depends on them.

| # | File | What to implement | Input → Output |
|---|---|---|---|
| 1 | `transformer/activations.py` | `ReLU.forward()`, `GELU.forward()` | `(any) → (same)` |
| 2 | `transformer/normalization.py` | `LayerNorm.forward()` | `(..., D) → (..., D)` |
| 3 | `attention/scaled_dot_product.py` | `ScaledDotProductAttention.forward()` | `Q,K,V → weighted V` |
| 4 | `attention/multi_head.py` | `MultiHeadAttention.forward()` | `(B,T,D)×3 → (B,T,D)` |
| 5 | `transformer/feed_forward.py` | `PositionWiseFeedForward.forward()` | `(B,T,D) → (B,T,D)` |
| 6 | `positional_encoding/sinusoidal.py` | `__init__()` + `forward()` | `(B,T,D) → (B,T,D)` |
| 7 | `transformer/embedding.py` | `__init__()` + `forward()` | `(B,T) → (B,T,D)` |
| 8 | `tokenizers/char_tokenizer.py` | `train()`, `encode()`, `decode()` | `str ↔ List[int]` |

---

## Phase 2: Pick a Project

### Option A: Transformer (Arabic↔English Translation)

| # | File | What to implement |
|---|---|---|
| 9 | `transformer/model.py` | ✅ Already done (assembly code) |
| 10 | `transformer/train.py` | ✅ Ready — plug in your tokenizer and run |

**Dataset:** `Helsinki-NLP/opus-100` (ar-en)
- Train: ~1M pairs | Val: ~2K pairs | Test: ~2K pairs
- Auto-downloads via HuggingFace

**Run:** `python -m transformer.train --subset_size 10000`

---

### Option B: ViT (Image Classification)

Requires Phase 1 complete (uses shared attention, normalization, feed_forward).

| # | File | What to implement | Input → Output |
|---|---|---|---|
| 9 | `vit/patch_embedding.py` | `PatchEmbedding.forward()` | `(B,C,H,W) → (B,N,D)` |
| 10 | `vit/model.py` | `ViTEncoderBlock.forward()` | `(B,T,D) → (B,T,D)` |
| 11 | `vit/model.py` | `VisionTransformer.forward()` | `(B,C,H,W) → (B,classes)` |

**Datasets:**
| Dataset | Size | Classes | Image Size | Loader |
|---|---|---|---|---|
| CIFAR-10 | 60K images | 10 | 32×32 | `torchvision` (auto-download) |
| Tiny ImageNet | 100K images | 200 | 64×64 | `zh-plus/tiny-imagenet` (HuggingFace) |

**Run:** `python -m vit.train --arch vit_tiny --dataset cifar10`

---

### Option C: GPT-2 (Language Modeling)

Requires Phase 1 complete (uses shared normalization, activations).

| # | File | What to implement | Input → Output |
|---|---|---|---|
| 9 | `gpt/layers.py` | `CausalSelfAttention.forward()` | `(B,T,D) → (B,T,D)` |
| 10 | `gpt/layers.py` | `GPTFeedForward.forward()` | `(B,T,D) → (B,T,D)` |
| 11 | `gpt/model.py` | `GPTBlock.forward()` | `(B,T,D) → (B,T,D)` |
| 12 | `gpt/model.py` | `GPT2.forward()` | `(B,T) → (B,T,V)` |
| 13 | `gpt/model.py` | `GPT2.generate()` | `(1,P) → (1,P+N)` |
| 14 | `gpt/tokenizer.py` | `GPTTokenizer.train/encode/decode` | `str ↔ List[int]` |

**Datasets:**
| Corpus | Size | Language | Loader |
|---|---|---|---|
| TinyStories | ~2.1M stories | English (simple) | `load_tinystories()` |
| WikiText-103 | ~100M tokens | English (Wikipedia) | `load_wikitext103()` |
| Arabic Wikipedia | ~300K articles | Arabic | `load_arabic_wiki()` |

**Run:** `python -m gpt.train --subset_size 10000`

---

### Option D: ViT (vision-models — production)

Self-contained, no dependency on Phase 1.

| # | File | What to implement |
|---|---|---|
| 1 | `models/vit.py` | `PatchEmbedding.forward()` |
| 2 | `models/vit.py` | `MultiHeadSelfAttention.forward()` |
| 3 | `models/vit.py` | `MLP.forward()` |
| 4 | `models/vit.py` | `TransformerBlock.forward()` |
| 5 | `models/vit.py` | `VisionTransformer.forward()` |

**Run:** `python train.py --model vit_tiny --dataset cifar10`

---

## Phase 3: Advanced Modules (Future)

| Module | Key files | Status |
|---|---|---|
| `fourier_features/` | `gaussian.py`, `positional.py` | Skeletons ready |
| `neural_implicit_representation/` | `inr.py`, `siren.py`, `wire.py`, `nerf.py`, `deep_sdf.py` | Skeletons ready |
| `autoencoders/` | `autoencoder.py`, `vae.py` | Skeletons ready |
| `diffusion/` | `ddpm.py`, `unet.py` | Skeletons ready |
| `flow_matching/` | `flow_matching.py` | Skeletons ready |
| `nn_modules/` | Linear, Conv2d, BatchNorm, Dropout, RNN | Planned |
| `losses_and_metrics/` | CrossEntropy, MSE, BLEU, Perplexity | Planned |
| `autograd/` | Custom tensor, computation graph, backward | Planned |
| `optimizers/` | SGD, Adam, AdamW, RMSProp, L-BFGS | Planned |

---

## Dataset Summary

| Dataset | Task | Size | Source |
|---|---|---|---|
| OPUS-100 ar-en | Translation | ~1M pairs | `Helsinki-NLP/opus-100` |
| CIFAR-10 | Classification | 60K × 32×32 | `torchvision` |
| Tiny ImageNet | Classification | 100K × 64×64 | `zh-plus/tiny-imagenet` |
| TinyStories | Language modeling | ~2.1M stories | `roneneldan/TinyStories` |
| WikiText-103 | Language modeling | ~100M tokens | `wikitext/wikitext-103-raw-v1` |
| Arabic Wikipedia | Language modeling | ~300K articles | `wikimedia/wikipedia 20231101.ar` |
