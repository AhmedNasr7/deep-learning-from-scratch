# Low-Rank Methods

This package is a scaffold for low-rank adaptation and compression methods.
It is designed to follow the repository's educational style with clear
placeholder classes and utility functions.

## Planned modules

- `lora.py` — LoRA adapter layer scaffold
- `dora.py` — DoRA adapter layer scaffold
- `svd_compress.py` — SVD-based compression utilities scaffold
- `apply.py` — helper functions to inject/adapt low-rank layers into models

## Usage pattern

The low-rank API is intended for later integration with GPT/Transformer models:

```python
from low_rank import LoRALinear, DoRALinear
from low_rank.apply import apply_lora, freeze_base_model

model = build_gpt2(config)
apply_lora(model, rank=4, target_modules=["W_q", "W_k", "W_v"])
freeze_base_model(model)
```

## Viability

Yes — low-rank adapters are a viable extension for GPT and transformer models.
A lightweight low-rank module can replace or augment attention and feed-forward
projection matrices while keeping the base architecture intact.
